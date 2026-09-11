"""
EfficientNet-B0 Model Loader & Inference Engine for Plant Disease Classification.
Loads trained weights from best_efficientnet_b0.pth and class mappings from class_mapping.json.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent

# Search candidate locations for the trained weights checkpoint
CANDIDATE_CHECKPOINTS = [
    BASE_DIR / "checkpoints" / "best_efficientnet_b0.pth",
    BASE_DIR / "best_efficientnet_b0.pth",
    BASE_DIR / "plantvillage_efficientnet_b0_fast_best.pth",
    BASE_DIR.parent / "crop-health-ai" / "backend" / "plantvillage_efficientnet_b0_fast_best.pth",
]

# Search candidate locations for class mapping json
CANDIDATE_MAPPINGS = [
    BASE_DIR / "class_mapping.json",
    BASE_DIR / "splits" / "class_mapping.json",
    BASE_DIR.parent / "crop-health-ai" / "backend" / "class_mapping.json",
]


def resolve_checkpoint_path(custom_path: Optional[str] = None) -> Path:
    if custom_path and Path(custom_path).is_file():
        return Path(custom_path)
    env_path = os.getenv("MODEL_PATH")
    if env_path and Path(env_path).is_file():
        return Path(env_path)
    for cand in CANDIDATE_CHECKPOINTS:
        if cand.is_file():
            return cand
    return CANDIDATE_CHECKPOINTS[0]


def resolve_mapping_path(custom_path: Optional[str] = None) -> Optional[Path]:
    if custom_path and Path(custom_path).is_file():
        return Path(custom_path)
    env_path = os.getenv("CLASS_MAPPING_PATH")
    if env_path and Path(env_path).is_file():
        return Path(env_path)
    for cand in CANDIDATE_MAPPINGS:
        if cand.is_file():
            return cand
    return None


class PlantDiseaseModel:
    def __init__(
        self,
        checkpoint_path: str = None,
        class_mapping_path: str = None,
        device: str = None
    ):
        resolved_ckpt = resolve_checkpoint_path(checkpoint_path)
        if not resolved_ckpt.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found at '{resolved_ckpt}'. "
                f"Please ensure best_efficientnet_b0.pth or plantvillage_efficientnet_b0_fast_best.pth is present."
            )
        self.checkpoint_path = str(resolved_ckpt)

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[Model] Initializing PlantDiseaseModel on device: {self.device}")
        print(f"[Model] Checkpoint: {self.checkpoint_path}")

        # Load trained checkpoint first
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        # Resolve class mappings
        resolved_mapping = resolve_mapping_path(class_mapping_path)
        if resolved_mapping and resolved_mapping.is_file():
            self.class_mapping_path = str(resolved_mapping)
            print(f"[Model] Class mapping: {self.class_mapping_path}")
            with open(self.class_mapping_path, "r", encoding="utf-8") as f:
                mapping_data = json.load(f)
                self.idx_to_class = {int(k): v for k, v in mapping_data["idx_to_class"].items()}
                self.class_to_idx = mapping_data["class_to_idx"]
                self.num_classes = mapping_data.get("num_classes", len(self.idx_to_class))
        elif isinstance(checkpoint, dict) and "class_names" in checkpoint:
            class_names = checkpoint["class_names"]
            self.idx_to_class = {i: c for i, c in enumerate(class_names)}
            self.class_to_idx = {c: i for i, c in enumerate(class_names)}
            self.num_classes = len(class_names)
            self.class_mapping_path = "loaded_from_checkpoint"
            print(f"[Model] Class mapping: extracted {self.num_classes} classes from checkpoint metadata")
        else:
            raise FileNotFoundError(
                "Could not find class_mapping.json and checkpoint does not contain 'class_names'."
            )

        # Build EfficientNet-B0
        self.model = models.efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, self.num_classes)
        )

        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        val_acc = checkpoint.get("val_acc", "93.98")
        print(f"[Model] Successfully loaded weights (Trained Val Accuracy: {val_acc}%)")

        # Determine image size from checkpoint or default to 160
        image_size = checkpoint.get("image_size", 160)
        if image_size == 160:
            self.transform = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def predict(self, image: Image.Image, top_k: int = 3) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Run inference on a PIL image.
        Returns:
            primary_prediction: Dict with class_name, confidence, idx
            top_predictions: List of Dicts with class_name, confidence, idx
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            top_probs, top_indices = torch.topk(probs, k=min(top_k, self.num_classes))

        top_predictions = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            cls_name = self.idx_to_class[idx]
            conf_pct = round(prob * 100.0, 1)
            top_predictions.append({
                "class_name": cls_name,
                "confidence": conf_pct,
                "index": idx
            })

        primary = top_predictions[0]
        return primary, top_predictions


_model_instance = None

def get_model() -> PlantDiseaseModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = PlantDiseaseModel()
    return _model_instance
