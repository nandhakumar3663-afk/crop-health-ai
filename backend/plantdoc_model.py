"""
backend/plantdoc_model.py - Hugging Face PlantDoc ResNet50 Model Loader & Inference Engine
Loads and serves predictions from Sami-06/PlantdocAI-ResNet50 as a secondary / fallback model.
- Downloads once and caches checkpoint to avoid repeated network traffic.
- Full GPU support when CUDA is available with graceful CPU fallback.
- Lazy singleton loading to preserve VRAM until needed.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
LOCAL_CKPT_NAME = "plantdoc_resnet50.pth"

# 29-class label sequence matching PlantDoc training configuration
PLANTDOC_RAW_CLASSES = [
    "Strawberry Healthy", "Strawberry Leaf Scorch", "Potato Early Blight", "Potato Healthy", "Potato Late Blight",
    "Grape Black Rot", "Grape Esca (Black Measles)", "Grape Healthy", "Grape Leaf Blight",
    "Apple Apple Scab", "Apple Black Rot", "Apple Cedar Apple Rust", "Apple Healthy",
    "Tomato Bacterial Spot", "Tomato Early Blight", "Tomato Healthy", "Tomato Late Blight",
    "Tomato Septoria Leaf Spot", "Tomato Yellow Leaf Curl Virus", "Bell Pepper Bacterial Spot",
    "Bell Pepper Healthy", "Corn Cercospora Leaf Spot", "Corn Common Rust", "Corn Healthy",
    "Corn Northern Leaf Blight", "Cherry Healthy", "Cherry Powdery Mildew", "Peach Bacterial Spot",
    "Peach Healthy"
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def download_plantdoc_checkpoint(dest_path: Path) -> Path:
    """Download PlantDoc checkpoint from Hugging Face if not cached."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.is_file() and dest_path.stat().st_size > 50 * 1024 * 1024:
        return dest_path

    print(f"[PlantDoc] Checkpoint not found locally. Downloading from Hugging Face...")
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id="Sami-06/PlantdocAI-ResNet50",
            filename="checkpoint.pth",
            local_dir=str(dest_path.parent),
            local_dir_use_symlinks=False
        )
        downloaded_path = Path(downloaded)
        if downloaded_path.resolve() != dest_path.resolve():
            import shutil
            shutil.copyfile(downloaded_path, dest_path)
        print(f"[PlantDoc] Download complete: {dest_path} ({dest_path.stat().st_size / (1024*1024):.1f} MB)")
        return dest_path
    except Exception as e:
        print(f"[PlantDoc] hf_hub_download failed: {e}. Falling back to requests stream...")
        import requests
        url = "https://huggingface.co/Sami-06/PlantdocAI-ResNet50/resolve/main/checkpoint.pth"
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        print(f"[PlantDoc] Direct download complete: {dest_path} ({dest_path.stat().st_size / (1024*1024):.1f} MB)")
        return dest_path


class PlantDocModel:
    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: Optional[nn.Module] = None
        self.classes: List[str] = PLANTDOC_RAW_CLASSES
        self.num_classes = len(self.classes)

        if checkpoint_path:
            self.checkpoint_path = Path(checkpoint_path)
        else:
            self.checkpoint_path = CHECKPOINT_DIR / LOCAL_CKPT_NAME

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def load(self):
        """Load PlantDoc ResNet50 model and weights."""
        if self.model is not None:
            return

        print(f"[PlantDoc] Initializing ResNet50 (29 classes) on device: {self.device}...")
        resolved_ckpt = download_plantdoc_checkpoint(self.checkpoint_path)

        # Build architecture: frozen backbone + 29-class linear head
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features  # 2048
        model.fc = nn.Linear(in_features, self.num_classes)

        print(f"[PlantDoc] Loading weights from: {resolved_ckpt}")
        ckpt_data = torch.load(str(resolved_ckpt), map_location=self.device, weights_only=False)
        state_dict = ckpt_data.get("model_state_dict", ckpt_data)
        model.load_state_dict(state_dict)

        model.to(self.device)
        model.eval()
        self.model = model
        val_acc = ckpt_data.get("val_acc", "97.88")
        print(f"[PlantDoc] Model successfully loaded (Reported Val Acc: {val_acc}%)")

    def is_loaded(self) -> bool:
        return self.model is not None

    def to_device(self, target_device: str):
        """Dynamic device migration to manage VRAM."""
        self.device = torch.device(target_device)
        if self.model is not None:
            self.model.to(self.device)

    def predict(self, image: Image.Image, top_k: int = 3) -> Dict[str, Any]:
        """
        Inference on leaf image.
        Returns:
            {
                "model": "PlantDoc-ResNet50",
                "prediction": str,
                "confidence": float,
                "class_index": int,
                "top_predictions": list
            }
        """
        if not self.is_loaded():
            self.load()

        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            top_probs, top_indices = torch.topk(probs, k=min(top_k, self.num_classes))

        top_predictions = []
        for p, idx in zip(top_probs.tolist(), top_indices.tolist()):
            cls_name = self.classes[idx] if idx < len(self.classes) else f"Class_{idx}"
            top_predictions.append({
                "class_name": cls_name,
                "confidence": round(p * 100.0, 1),
                "index": idx
            })

        primary = top_predictions[0]
        return {
            "model": "PlantDoc-ResNet50",
            "prediction": primary["class_name"],
            "confidence": primary["confidence"],
            "class_index": primary["index"],
            "top_predictions": top_predictions
        }


_plantdoc_singleton: Optional[PlantDocModel] = None


def get_plantdoc_model() -> PlantDocModel:
    global _plantdoc_singleton
    if _plantdoc_singleton is None:
        _plantdoc_singleton = PlantDocModel()
    return _plantdoc_singleton
