"""
backend/hybrid_router.py - Hybrid Confidence Router
Coordinates inference between:
1. Primary Model: EfficientNet-B0 + GPU-Accelerated XGBoost (38 classes)
2. Secondary Model: PlantDoc ResNet50 (29 classes, semantically mapped)

Routing Logic:
1. Runs Primary Model (EfficientNet-B0 + XGBoost) to produce initial prediction & confidence.
2. If confidence >= PRIMARY_CONFIDENCE_THRESHOLD (default 70.0%):
     Confidence is high -> accepts primary prediction directly (reliability: HIGH).
3. If confidence < PRIMARY_CONFIDENCE_THRESHOLD:
     Invokes PlantDoc ResNet50 fallback model.
     Maps PlantDoc class to FarmGuardian taxonomy via ModelClassMapper.
     If mapped classes agree -> reinforces prediction (reliability: HIGH, source: 'hybrid').
     If mapped classes disagree -> marks prediction as uncertain (reliability: LOW, source: 'uncertain').
     If PlantDoc class is not mappable -> keeps primary prediction (reliability: MEDIUM, source: 'primary').
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image

import numpy as np
import torch
import xgboost as xgb
from torchvision import transforms, models

from model import resolve_checkpoint_path, resolve_mapping_path
from model_class_mapping import get_class_mapper, ModelClassMapper
from plantdoc_model import get_plantdoc_model, PlantDocModel

DEFAULT_CONFIDENCE_THRESHOLD = 70.0
FEATURE_DIM = 1280
NUM_CLASSES = 38


class HybridRouter:
    def __init__(
        self,
        efficientnet_path: Optional[str] = None,
        xgboost_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        device: Optional[str] = None
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else float(os.getenv("PRIMARY_CONFIDENCE_THRESHOLD", str(DEFAULT_CONFIDENCE_THRESHOLD)))
        )
        self.mode = os.getenv("MODE", "hybrid").lower()

        # Paths
        self.b0_checkpoint_path = resolve_checkpoint_path(efficientnet_path)
        self.backend_dir = Path(__file__).resolve().parent
        candidate_xgb = [
            Path(xgboost_path) if xgboost_path else None,
            self.backend_dir / "checkpoints" / "xgboost_efficientnet_b0.json",
            self.backend_dir.parent.parent / "plant-disease-detection" / "checkpoints" / "xgboost_efficientnet_b0.json",
            Path("C:/Users/DELL/Desktop/plant-disease-detection/checkpoints/xgboost_efficientnet_b0.json"),
        ]
        self.xgb_model_path = next((p for p in candidate_xgb if p and p.is_file()), candidate_xgb[1])

        # Lazy singletons
        self.feature_extractor: Optional[torch.nn.Module] = None
        self.xgboost_booster: Optional[xgb.Booster] = None
        self.plantdoc_model: Optional[PlantDocModel] = None
        self.class_mapper: ModelClassMapper = get_class_mapper()

        # Class names & mapping
        self.idx_to_class: Dict[int, str] = {}
        self._load_class_mapping()

        # Deterministic preprocessing for EfficientNet-B0 (matching feature extraction)
        self.b0_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_class_mapping(self):
        mapping_file = resolve_mapping_path()
        if mapping_file and mapping_file.is_file():
            import json
            with open(mapping_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.idx_to_class = {int(k): v for k, v in data["idx_to_class"].items()}
        else:
            # Fallback 38 classes
            print("[Router] Class mapping JSON not found. Loading classes from checkpoint...")
            ckpt = torch.load(self.b0_checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "class_names" in ckpt:
                self.idx_to_class = {i: c for i, c in enumerate(ckpt["class_names"])}

    def _init_primary_models(self):
        """Warm up EfficientNet-B0 feature extractor and XGBoost booster."""
        if self.feature_extractor is None:
            print(f"[Router] Initializing EfficientNet-B0 extractor on {self.device}...")
            model = models.efficientnet_b0(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(p=0.2, inplace=True),
                torch.nn.Linear(in_features, NUM_CLASSES)
            )

            ckpt = torch.load(self.b0_checkpoint_path, map_location=self.device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict)

            # Extract 1280-D deep features
            model.classifier = torch.nn.Identity()
            for p in model.parameters():
                p.requires_grad = False
            model.to(self.device)
            model.eval()
            self.feature_extractor = model
            print("[Router] EfficientNet-B0 feature extractor ready.")

        if self.xgboost_booster is None:
            if not self.xgb_model_path.exists():
                raise FileNotFoundError(f"XGBoost model file not found at: {self.xgb_model_path}")
            print(f"[Router] Loading XGBoost booster from: {self.xgb_model_path}")
            bst = xgb.Booster()
            bst.load_model(str(self.xgb_model_path))
            self.xgboost_booster = bst
            print("[Router] XGBoost booster ready.")

    def _predict_primary(self, image: Image.Image, top_k: int = 5) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Extract features via EfficientNet-B0 and classify via XGBoost."""
        self._init_primary_models()

        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = self.b0_transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            features = self.feature_extractor(tensor)  # Shape: [1, 1280]
            feat_np = features.float().cpu().numpy()

        dmat = xgb.DMatrix(feat_np)
        probs = self.xgboost_booster.predict(dmat)[0]  # Shape: [38]

        top_indices = np.argsort(probs)[::-1][:top_k]
        top_predictions = []
        for idx in top_indices:
            cls_name = self.idx_to_class.get(int(idx), f"Class_{idx}")
            conf_pct = round(float(probs[idx]) * 100.0, 1)
            top_predictions.append({
                "class_name": cls_name,
                "confidence": conf_pct,
                "index": int(idx)
            })

        primary = top_predictions[0]
        return primary, top_predictions

    def route(
        self,
        image: Image.Image,
        force_secondary: bool = True
    ) -> Dict[str, Any]:
        """
        Execute end-to-end routing strategy based on mode and confidence.
        force_secondary=True ensures secondary telemetry is available for the UI.
        """
        current_mode = os.getenv("MODE", self.mode).lower()

        # -------------------------------------------------------------
        # Mode: PlantDoc Only
        # -------------------------------------------------------------
        if current_mode == "plantdoc":
            plantdoc = get_plantdoc_model()
            sec_res = plantdoc.predict(image, top_k=5)
            mapped_cls = self.class_mapper.map_plantdoc_to_farmguardian(sec_res["class_index"])
            final_cls = mapped_cls if mapped_cls != "not_mappable" else sec_res["prediction"]

            return {
                "final_prediction": final_cls,
                "confidence": sec_res["confidence"],
                "reliability": "MEDIUM",
                "primary_model": None,
                "secondary_model": {
                    "name": "PlantDoc-ResNet50",
                    "prediction": sec_res["prediction"],
                    "confidence": sec_res["confidence"],
                    "mapped_class": mapped_cls
                },
                "agreement": True,
                "model_source": "plantdoc",
                "top_predictions": sec_res["top_predictions"]
            }

        # -------------------------------------------------------------
        # Mode: Primary (EfficientNet-B0 + XGBoost)
        # -------------------------------------------------------------
        primary_top, top_predictions = self._predict_primary(image, top_k=5)

        if current_mode == "primary":
            return {
                "final_prediction": primary_top["class_name"],
                "confidence": primary_top["confidence"],
                "reliability": "HIGH" if primary_top["confidence"] >= self.threshold else "MEDIUM",
                "primary_model": {
                    "name": "EfficientNet-B0 + XGBoost",
                    "prediction": primary_top["class_name"],
                    "confidence": primary_top["confidence"]
                },
                "secondary_model": None,
                "agreement": True,
                "model_source": "primary",
                "top_predictions": top_predictions
            }

        # -------------------------------------------------------------
        # Mode: Hybrid (Confidence Routing + Verification)
        # -------------------------------------------------------------
        needs_secondary = (primary_top["confidence"] < self.threshold) or force_secondary

        secondary_data = None
        agreement = True
        reliability = "HIGH" if primary_top["confidence"] >= self.threshold else "MEDIUM"
        model_source = "primary"

        if needs_secondary:
            plantdoc = get_plantdoc_model()
            sec_res = plantdoc.predict(image, top_k=3)
            agree, mapped_fg, pd_name = self.class_mapper.compare_predictions(
                primary_top["class_name"],
                sec_res["class_index"]
            )
            agreement = agree

            secondary_data = {
                "name": "PlantDoc-ResNet50",
                "prediction": pd_name,
                "confidence": sec_res["confidence"],
                "mapped_class": mapped_fg
            }

            if primary_top["confidence"] >= self.threshold:
                # Primary is very confident
                if agree:
                    reliability = "HIGH"
                    model_source = "hybrid"
                else:
                    # Minor conflict but primary is above threshold
                    reliability = "HIGH"
                    model_source = "primary"
            else:
                # Primary confidence is low (< threshold)
                if agree:
                    # Both models agree on the diagnosis!
                    reliability = "HIGH"
                    model_source = "hybrid"
                elif mapped_fg == "not_mappable":
                    # PlantDoc does not have this class; maintain primary
                    reliability = "MEDIUM"
                    model_source = "primary"
                else:
                    # Clear disagreement between two mappable classes!
                    reliability = "LOW"
                    model_source = "uncertain"

        return {
            "predicted_class": primary_top["class_name"],
            "final_prediction": primary_top["class_name"],
            "confidence": primary_top["confidence"],
            "reliability": reliability,
            "primary_model": {
                "name": "EfficientNet-B0 + XGBoost",
                "prediction": primary_top["class_name"],
                "confidence": primary_top["confidence"]
            },
            "secondary_model": secondary_data,
            "agreement": agreement,
            "model_source": model_source,
            "top_predictions": top_predictions
        }


    def load_models(self):
        self._init_primary_models()

    def predict(self, image: Image.Image, force_secondary: bool = False) -> Dict[str, Any]:
        return self.route(image, force_secondary=force_secondary)


_router_singleton: Optional[HybridRouter] = None


def get_hybrid_router() -> HybridRouter:
    global _router_singleton
    if _router_singleton is None:
        _router_singleton = HybridRouter()
    return _router_singleton
