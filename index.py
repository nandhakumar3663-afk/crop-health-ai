import os
import torch
import torch.nn.functional as F

from PIL import Image
from io import BytesIO

from fastapi import FastAPI, UploadFile, File
from torchvision import models, transforms


app = FastAPI()


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
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
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "model": "EfficientNet-B0",
        "classes": len(class_names)
    }


# =========================================================
# PREDICTION API
# =========================================================

@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...)
):

    contents = await file.read()

    image = Image.open(
        BytesIO(contents)
    ).convert("RGB")

    image_tensor = transform(
        image
    ).unsqueeze(0).to(DEVICE)


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


    predicted_class = class_names[
        predicted.item()
    ]

    confidence_value = (
        confidence.item() * 100
    )


    # Top 3 predictions

    top_probs, top_indices = torch.topk(
        probabilities,
        3,
        dim=1
    )

    top_predictions = []

    for i in range(3):

        top_predictions.append({

            "class": class_names[
                top_indices[0][i].item()
            ],

            "confidence":
                top_probs[0][i].item() * 100

        })


    is_healthy = (
        "healthy"
        in predicted_class.lower()
    )


    return {

        "prediction": predicted_class,

        "confidence": round(
            confidence_value,
            2
        ),

        "status":
            "HEALTHY"
            if is_healthy
            else "DISEASED",

        "top_predictions":
            top_predictions

    }