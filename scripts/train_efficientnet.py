import os
import copy
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)

from PIL import Image
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = os.path.expanduser(
    "~/Downloads/PlantVillage-Dataset-master/raw/color"
)

MODEL_DIR = os.path.expanduser(
    "~/Downloads/crop-health-ai/models"
)

RESULTS_DIR = os.path.expanduser(
    "~/Downloads/crop-health-ai/results"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "tomato_efficientnet_b0.pth"
)

IMAGE_SIZE = 224

# Keep these low because your laptop has 8 GB RAM
MAX_IMAGES_PER_CLASS = 150
BATCH_SIZE = 8

EPOCHS = 8

LEARNING_RATE = 1e-4

# Stop training if validation does not improve
PATIENCE = 2

SEED = 42


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("🌱 TOMATO CROP DISEASE DETECTION")
print("EfficientNet-B0")
print("=" * 60)

print(f"Device: {DEVICE}")

if DEVICE.type == "cpu":
    print("ℹ️ CPU training enabled")
else:
    print("🚀 CUDA GPU detected")


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# TOMATO CLASSES
# ============================================================

TOMATO_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )


print("\n📂 Dataset:")
print(DATASET_PATH)


# ============================================================
# LOAD ORIGINAL DATASET WITHOUT TRANSFORM
# ============================================================

full_dataset = datasets.ImageFolder(
    root=DATASET_PATH
)

available_classes = full_dataset.classes

print("\n📋 Dataset classes detected:")
print(f"Total classes in PlantVillage: {len(available_classes)}")


# ============================================================
# FIND TOMATO CLASS INDICES
# ============================================================

tomato_class_indices = {}

for tomato_class in TOMATO_CLASSES:

    if tomato_class not in full_dataset.class_to_idx:

        raise ValueError(
            f"\n❌ Class not found:\n{tomato_class}"
        )

    tomato_class_indices[tomato_class] = (
        full_dataset.class_to_idx[tomato_class]
    )


# ============================================================
# COLLECT IMAGE PATHS
# ============================================================

print("\n🔍 Collecting Tomato images...")


class_to_images = {}

for tomato_class in TOMATO_CLASSES:

    class_dir = os.path.join(
        DATASET_PATH,
        tomato_class
    )

    images = []

    for filename in os.listdir(class_dir):

        filepath = os.path.join(
            class_dir,
            filename
        )

        if os.path.isfile(filepath):

            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
            ):

                images.append(filepath)

    # Shuffle deterministically
    random.Random(SEED).shuffle(images)

    # Limit images for low-RAM training
    images = images[:MAX_IMAGES_PER_CLASS]

    class_to_images[tomato_class] = images

    print(
        f"{tomato_class}: {len(images)} images"
    )


# ============================================================
# CREATE FILEPATH + LABEL LIST
# ============================================================

image_paths = []
labels = []


for label_id, tomato_class in enumerate(TOMATO_CLASSES):

    for filepath in class_to_images[tomato_class]:

        image_paths.append(filepath)

        labels.append(label_id)


image_paths = np.array(image_paths)
labels = np.array(labels)


print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print(
    f"Total images: {len(image_paths)}"
)

print(
    f"Total classes: {len(TOMATO_CLASSES)}"
)


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# First split:
# 80% train
# 20% temporary

train_paths, temp_paths, train_labels, temp_labels = (
    train_test_split(
        image_paths,
        labels,
        test_size=0.20,
        random_state=SEED,
        stratify=labels
    )
)


# Second split:
# temporary -> 10% validation + 10% test

val_paths, test_paths, val_labels, test_labels = (
    train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_labels
    )
)


print("\n📊 Split:")
print(f"Training:   {len(train_paths)}")
print(f"Validation: {len(val_paths)}")
print(f"Testing:    {len(test_paths)}")


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=15
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# CUSTOM DATASET
# ============================================================

class TomatoDataset(Dataset):

    def __init__(
        self,
        image_paths,
        labels,
        transform=None
    ):

        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]

        label = int(
            self.labels[index]
        )

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as e:

            raise RuntimeError(
                f"Could not read image:\n"
                f"{image_path}\n"
                f"Error: {e}"
            )

        if self.transform:

            image = self.transform(image)

        return image, label


# ============================================================
# CREATE DATASETS
# ============================================================

train_dataset = TomatoDataset(
    train_paths,
    train_labels,
    train_transform
)

val_dataset = TomatoDataset(
    val_paths,
    val_labels,
    val_transform
)

test_dataset = TomatoDataset(
    test_paths,
    test_labels,
    val_transform
)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    # IMPORTANT FOR YOUR UBUNTU/PYTHON SETUP
    num_workers=0,

    pin_memory=False
)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0,

    pin_memory=False
)


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0,

    pin_memory=False
)


# ============================================================
# LOAD PRETRAINED EFFICIENTNET-B0
# ============================================================

print("\n🧠 Loading EfficientNet-B0...")

weights = EfficientNet_B0_Weights.DEFAULT

model = models.efficientnet_b0(
    weights=weights
)


# ============================================================
# REPLACE CLASSIFIER
# ============================================================

num_features = (
    model.classifier[1].in_features
)

model.classifier[1] = nn.Linear(
    num_features,
    len(TOMATO_CLASSES)
)


model = model.to(DEVICE)


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=1e-4
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="min",

    factor=0.5,

    patience=1
)


# ============================================================
# TRAINING HISTORY
# ============================================================

train_losses = []
train_accuracies = []

val_losses = []
val_accuracies = []


best_val_accuracy = 0.0

best_model_state = None

epochs_without_improvement = 0


# ============================================================
# TRAINING LOOP
# ============================================================

print("\n🚀 Starting training...")

print("=" * 60)


for epoch in range(EPOCHS):

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    print("-" * 40)


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    progress_bar = tqdm(
        train_loader,
        desc="Training"
    )


    for images, labels_batch in progress_bar:

        images = images.to(DEVICE)

        labels_batch = labels_batch.to(DEVICE)


        # Clear gradients

        optimizer.zero_grad()


        # Forward pass

        outputs = model(images)


        # Calculate loss

        loss = criterion(
            outputs,
            labels_batch
        )


        # Backpropagation

        loss.backward()


        # Update weights

        optimizer.step()


        # Statistics

        running_loss += (
            loss.item() * images.size(0)
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        correct += (
            predictions == labels_batch
        ).sum().item()


        total += labels_batch.size(0)


        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )


    train_loss = (
        running_loss / total
    )

    train_accuracy = (
        100.0 * correct / total
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_running_loss = 0.0

    val_correct = 0

    val_total = 0


    with torch.no_grad():

        for images, labels_batch in val_loader:

            images = images.to(DEVICE)

            labels_batch = labels_batch.to(DEVICE)


            outputs = model(images)


            loss = criterion(
                outputs,
                labels_batch
            )


            val_running_loss += (
                loss.item() * images.size(0)
            )


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            val_correct += (
                predictions == labels_batch
            ).sum().item()


            val_total += (
                labels_batch.size(0)
            )


    val_loss = (
        val_running_loss / val_total
    )

    val_accuracy = (
        100.0 * val_correct / val_total
    )


    # Scheduler

    scheduler.step(val_loss)


    # Store history

    train_losses.append(
        train_loss
    )

    train_accuracies.append(
        train_accuracy
    )

    val_losses.append(
        val_loss
    )

    val_accuracies.append(
        val_accuracy
    )


    # Print results

    print(
        f"\nTrain Loss:      {train_loss:.4f}"
    )

    print(
        f"Train Accuracy:  {train_accuracy:.2f}%"
    )

    print(
        f"Val Loss:        {val_loss:.4f}"
    )

    print(
        f"Val Accuracy:    {val_accuracy:.2f}%"
    )


    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Learning Rate:   {current_lr:.7f}"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        epochs_without_improvement = 0

        print(
            "✅ New best model!"
        )

    else:

        epochs_without_improvement += 1

        print(
            f"⚠️ No improvement "
            f"({epochs_without_improvement}/{PATIENCE})"
        )


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if epochs_without_improvement >= PATIENCE:

        print(
            "\n🛑 Early stopping triggered."
        )

        break


# ============================================================
# LOAD BEST MODEL
# ============================================================

if best_model_state is not None:

    model.load_state_dict(
        best_model_state
    )


# ============================================================
# SAVE MODEL
# ============================================================

checkpoint = {

    "model_state_dict":
        model.state_dict(),

    "classes":
        TOMATO_CLASSES,

    "image_size":
        IMAGE_SIZE,

    "best_val_accuracy":
        best_val_accuracy

}


torch.save(
    checkpoint,
    MODEL_PATH
)


print("\n" + "=" * 60)

print("✅ BEST MODEL SAVED")

print("=" * 60)

print(
    f"Model: {MODEL_PATH}"
)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.2f}%"
)


# ============================================================
# TEST MODEL
# ============================================================

print("\n🧪 Evaluating on test set...")

model.eval()

all_predictions = []

all_true_labels = []


with torch.no_grad():

    for images, labels_batch in tqdm(
        test_loader,
        desc="Testing"
    ):

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )


        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_true_labels.extend(
            labels_batch.numpy()
        )


# ============================================================
# TEST ACCURACY
# ============================================================

test_accuracy = (
    accuracy_score(
        all_true_labels,
        all_predictions
    ) * 100
)


print("\n" + "=" * 60)

print(
    f"🎯 TEST ACCURACY: "
    f"{test_accuracy:.2f}%"
)

print("=" * 60)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    all_true_labels,

    all_predictions,

    target_names=TOMATO_CLASSES,

    digits=4,

    zero_division=0
)


print("\n📊 CLASSIFICATION REPORT\n")

print(report)


# Save report

report_path = os.path.join(
    RESULTS_DIR,
    "classification_report.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    all_true_labels,

    all_predictions
)


display = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=[
        name.replace(
            "Tomato___",
            ""
        )
        for name in TOMATO_CLASSES
    ]
)


fig, ax = plt.subplots(
    figsize=(14, 12)
)


display.plot(
    ax=ax,
    xticks_rotation=90,
    cmap="Blues",
    colorbar=False
)


plt.title(
    "Tomato Disease Detection - Confusion Matrix"
)

plt.tight_layout()


cm_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.png"
)


plt.savefig(
    cm_path,
    dpi=200
)


plt.close()


# ============================================================
# TRAINING CURVES
# ============================================================

epochs_completed = range(
    1,
    len(train_losses) + 1
)


# Accuracy

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    epochs_completed,
    train_accuracies,
    marker="o",
    label="Training Accuracy"
)


plt.plot(
    epochs_completed,
    val_accuracies,
    marker="o",
    label="Validation Accuracy"
)


plt.xlabel("Epoch")

plt.ylabel("Accuracy (%)")

plt.title(
    "EfficientNet-B0 Training Accuracy"
)

plt.legend()

plt.grid(True)


accuracy_path = os.path.join(
    RESULTS_DIR,
    "accuracy_curve.png"
)


plt.savefig(
    accuracy_path,
    dpi=200
)


plt.close()


# Loss

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    epochs_completed,
    train_losses,
    marker="o",
    label="Training Loss"
)


plt.plot(
    epochs_completed,
    val_losses,
    marker="o",
    label="Validation Loss"
)


plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "EfficientNet-B0 Training Loss"
)

plt.legend()

plt.grid(True)


loss_path = os.path.join(
    RESULTS_DIR,
    "loss_curve.png"
)


plt.savefig(
    loss_path,
    dpi=200
)


plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)

print("🌱 TRAINING COMPLETE")

print("=" * 60)

print(
    f"Classes:                {len(TOMATO_CLASSES)}"
)

print(
    f"Training images:        {len(train_paths)}"
)

print(
    f"Validation images:      {len(val_paths)}"
)

print(
    f"Test images:            {len(test_paths)}"
)

print(
    f"Best validation acc:    "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"Test accuracy:          "
    f"{test_accuracy:.2f}%"
)

print("\n📁 Files created:")

print(
    f"Model:      {MODEL_PATH}"
)

print(
    f"Report:     {report_path}"
)

print(
    f"Confusion:  {cm_path}"
)

print(
    f"Accuracy:   {accuracy_path}"
)

print(
    f"Loss:       {loss_path}"
)

print("\n✅ Done!")