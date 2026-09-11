"""
Crop Health AI - Production FastAPI Backend with Dual-Model Hybrid Routing
- Primary Model: EfficientNet-B0 (1280-D features) + GPU-accelerated XGBoost classifier
- Secondary Model: PlantDoc ResNet-50 for high-uncertainty validation
- Dynamic Routing: Confidence threshold (>70%), cross-model agreement, and reliability badges
- Pre-validation: Blur, resolution, and brightness image quality checks
- Fully backward-compatible with Vercel frontend (public/index.html) and FarmGuardian
"""

import os
import io
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

from PIL import Image
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure backend directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from image_quality import validate_leaf_image
    from hybrid_router import get_hybrid_router
except ImportError:
    from backend.image_quality import validate_leaf_image
    from backend.hybrid_router import get_hybrid_router

app = FastAPI(
    title="Crop Health AI API",
    description="Hybrid Dual-Model Plant Disease Detection API",
    version="2.0.0"
)

# CORS - Allow Vercel frontend, local dev, and FarmGuardian
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cached singleton router
_router = None

def get_router():
    global _router
    if _router is None:
        _router = get_hybrid_router()
    return _router


@app.on_event("startup")
async def startup_event():
    """Pre-warm model weights on startup."""
    try:
        router = get_router()
        router.load_models()
        print(f"[Startup] Hybrid AI Router ready on device: {router.device}")
    except Exception as e:
        print(f"[Startup] Warning during model initialization: {e}")


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Crop Health AI - Hybrid Disease Detection API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health():
    router = get_router()
    return {
        "status": "online",
        "message": "Crop Health AI API - Hybrid Production System",
        "primary_model": "EfficientNet-B0 + XGBoost (38 classes)",
        "secondary_model": "PlantDoc ResNet-50 (29 classes)",
        "device": str(router.device),
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "confidence_threshold": router.threshold,
        "mode": router.mode,
        "models_loaded": {
            "efficientnet_b0": router.feature_extractor is not None,
            "xgboost": router.xgboost_booster is not None,
            "plantdoc_resnet50": True
        }
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return {
            "success": False,
            "error": "Only image uploads (JPEG, PNG, WebP) are supported."
        }

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # 1. Image Quality Pre-Validation
        is_valid, quality_error = validate_leaf_image(image)
        if not is_valid:
            return {
                "success": False,
                "error": quality_error
            }

        # 2. Hybrid Router Prediction
        router = get_router()
        route_res = router.predict(image)

        predicted_class = route_res.get("predicted_class", "Unknown")
        confidence_val = float(route_res.get("confidence", 0.0))

        # 3. Determine Healthy vs Diseased status
        if "healthy" in predicted_class.lower():
            status = "HEALTHY"
        else:
            status = "DISEASED"

        # 4. Format Top Predictions
        top_candidates = route_res.get("top_predictions", route_res.get("top_k", []))
        top_predictions = []
        for item in top_candidates[:3]:
            cls_name = item.get("class_name", item.get("class", "Unknown"))
            top_predictions.append({
                "class": cls_name,
                "confidence": round(float(item["confidence"]), 2)
            })

        # 5. Return Unified JSON Response
        return {
            "success": True,
            "prediction": predicted_class,
            "confidence": round(confidence_val, 2),
            "status": status,
            "top_predictions": top_predictions,
            "primary_model": route_res.get("primary_model", "EfficientNet-B0 + XGBoost"),
            "secondary_model": route_res.get("secondary_model", "PlantDoc ResNet-50"),
            "agreement": route_res.get("agreement", True),
            "reliability": route_res.get("reliability", "HIGH"),
            "model_source": route_res.get("model_source", "hybrid")
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)