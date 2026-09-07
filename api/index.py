import os
from io import BytesIO

import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from torchvision import models, transforms


# =========================================================
# APP
# =========================================================

app = FastAPI()


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "scripts",
    "models",
    "plantvillage_efficientnet_b0_fast_best.pth"
)

IMAGE_SIZE = 160

DEVICE = torch.device("cpu")


# =========================================================
# LOAD MODEL
# =========================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

class_names = checkpoint["class_names"]

model = models.efficientnet_b0(
    weights=None
)

model.classifier[1] = torch.nn.Linear(
    model.classifier[1].in_features,
    len(class_names)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)
model.eval()


# =========================================================
# IMAGE TRANSFORM
# =========================================================

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# =========================================================
# HOME / HEALTH CHECK
# =========================================================

@app.get("/")
def home():

    return {
        "status": "online",
        "message": "Crop Health AI API",
        "model": "EfficientNet-B0",
        "classes": len(class_names)
    }


# =========================================================
# PREDICTION
# =========================================================

@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...)
):

    try:

        # Read uploaded image
        contents = await file.read()

        # Convert image
        image = Image.open(
            BytesIO(contents)
        ).convert("RGB")

        # Transform image
        image_tensor = transform(
            image
        ).unsqueeze(0).to(DEVICE)

        # Model prediction
        with torch.no_grad():

            output = model(
                image_tensor
            )

            probabilities = F.softmax(
                output,
                dim=1
            )

            confidence, predicted = torch.max(
                probabilities,
                dim=1
            )

        # Main prediction
        predicted_class = class_names[
            predicted.item()
        ]

        confidence_value = (
            confidence.item() * 100
        )

        # Top 3 predictions
        top_probs, top_indices = torch.topk(
            probabilities,
            k=3,
            dim=1
        )

        top_predictions = []

        for i in range(3):

            top_predictions.append({
                "class": class_names[
                    top_indices[0][i].item()
                ],
                "confidence": round(
                    top_probs[0][i].item() * 100,
                    2
                )
            })

        # Healthy / diseased
        if "healthy" in predicted_class.lower():
            status = "HEALTHY"
        else:
            status = "DISEASED"

        # Return JSON
        return {
            "success": True,
            "prediction": predicted_class,
            "confidence": round(
                confidence_value,
                2
            ),
            "status": status,
            "top_predictions": top_predictions
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }