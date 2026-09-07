import os
import json
import random
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = r"C:\Datasets\PlantVillage\PlantVillage-Dataset-master\raw\color"

MODEL_DIR = r"models"
RESULTS_DIR = r"results"

IMAGE_SIZE = 160

EPOCHS = 3

# CPU-friendly
BATCH_SIZE = 16
NUM_WORKERS = 0

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

SEED = 42

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# SEED
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# DEVICE
# ============================================================

def get_device():
    if torch.cuda.is_available():
        print("\nGPU DETECTED")
        print("GPU:", torch.cuda.get_device_name(0))
        return torch.device("cuda")

    print("\nCPU MODE")
    print("PyTorch CUDA is not available.")
    return torch.device("cpu")


# ============================================================
# CHECK DATASET
# ============================================================

def check_dataset():

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"\nDataset not found:\n{DATASET_PATH}"
        )

    class_count = len([
        x for x in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, x))
    ])

    total_images = 0

    for root, _, files in os.walk(DATASET_PATH):
        for file in files:
            if file.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            ):
                total_images += 1

    print("\n================ DATASET ================")
    print("Path:", DATASET_PATH)
    print("Classes:", class_count)
    print("Images :", f"{total_images:,}")

    if class_count != 38:
        print("WARNING: Expected 38 classes.")

    if total_images < 50000:
        print("WARNING: Dataset seems smaller than expected.")

    print("==========================================")


# ============================================================
# TRANSFORMS
# ============================================================

def get_transforms():

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    return train_transform, eval_transform


# ============================================================
# DATASETS
# ============================================================

def create_datasets():

    train_transform, eval_transform = get_transforms()

    base_dataset = datasets.ImageFolder(DATASET_PATH)

    targets = np.array(base_dataset.targets)
    indices = np.arange(len(base_dataset))

    # 80% TRAIN
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=SEED,
        stratify=targets
    )

    # 10% VAL + 10% TEST
    temp_targets = targets[temp_idx]

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_targets
    )

    train_full = datasets.ImageFolder(
        DATASET_PATH,
        transform=train_transform
    )

    eval_full = datasets.ImageFolder(
        DATASET_PATH,
        transform=eval_transform
    )

    train_dataset = Subset(
        train_full,
        train_idx
    )

    val_dataset = Subset(
        eval_full,
        val_idx
    )

    test_dataset = Subset(
        eval_full,
        test_idx
    )

    print("\n================ SPLIT ================")
    print("Train:", f"{len(train_dataset):,}")
    print("Val  :", f"{len(val_dataset):,}")
    print("Test :", f"{len(test_dataset):,}")
    print("Total:",
          f"{len(train_dataset) + len(val_dataset) + len(test_dataset):,}")
    print("=======================================")

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        base_dataset.classes
    )


# ============================================================
# DATALOADERS
# ============================================================

def create_loaders(train_dataset, val_dataset, test_dataset):

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return train_loader, val_loader, test_loader


# ============================================================
# MODEL
# ============================================================

def create_model(num_classes, device):

    print("\nLoading EfficientNet-B0...")

    weights = EfficientNet_B0_Weights.DEFAULT

    model = models.efficientnet_b0(
        weights=weights
    )

    # FREEZE FEATURE EXTRACTOR
    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        num_classes
    )

    model = model.to(device)

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    print("\n================ MODEL ================")
    print("Architecture : EfficientNet-B0")
    print("Classes      :", num_classes)
    print("Trainable    :", f"{trainable:,}")
    print("Total params :", f"{total:,}")
    print("=======================================")

    return model


# ============================================================
# TRAIN
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    progress = tqdm(
        loader,
        desc="Training"
    )

    for images, labels in progress:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()
        optimizer.step()

        total_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        accuracy = (
            correct / total
        ) * 100

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{accuracy:.2f}%"
        )

    return (
        total_loss / total,
        correct / total * 100
    )


# ============================================================
# VALIDATE / TEST
# ============================================================

def evaluate(
    model,
    loader,
    criterion,
    device,
    desc
):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        progress = tqdm(
            loader,
            desc=desc
        )

        for images, labels in progress:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    accuracy = (
        correct / total
    ) * 100

    return (
        total_loss / total,
        accuracy,
        all_labels,
        all_predictions
    )


# ============================================================
# CURVES
# ============================================================

def save_curves(history):

    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )

    # LOSS
    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "loss_curve.png"
        ),
        dpi=200
    )

    plt.close()

    # ACCURACY
    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        history["train_accuracy"],
        label="Train Accuracy"
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "accuracy_curve.png"
        ),
        dpi=200
    )

    plt.close()


# ============================================================
# CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    labels,
    predictions,
    class_names
):

    cm = confusion_matrix(
        labels,
        predictions
    )

    plt.figure(
        figsize=(20, 18)
    )

    sns.heatmap(
        cm,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("PlantVillage Confusion Matrix")

    plt.xticks(
        rotation=90,
        fontsize=7
    )

    plt.yticks(
        fontsize=7
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "confusion_matrix.png"
        ),
        dpi=200
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    check_dataset()

    device = get_device()

    (
        train_dataset,
        val_dataset,
        test_dataset,
        class_names
    ) = create_datasets()

    (
        train_loader,
        val_loader,
        test_loader
    ) = create_loaders(
        train_dataset,
        val_dataset,
        test_dataset
    )

    model = create_model(
        len(class_names),
        device
    )

    # Only classifier parameters are trainable
    trainable_parameters = filter(
        lambda p: p.requires_grad,
        model.parameters()
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }

    best_val_accuracy = 0.0

    best_model_path = os.path.join(
        MODEL_DIR,
        "plantvillage_efficientnet_b0_fast_best.pth"
    )

    # ========================================================
    # TRAINING
    # ========================================================

    for epoch in range(EPOCHS):

        print("\n")
        print("=" * 60)
        print(
            f"EPOCH {epoch + 1}/{EPOCHS}"
        )
        print("=" * 60)

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_accuracy, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device,
            "Validation"
        )

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["val_accuracy"].append(
            val_accuracy
        )

        print("\nRESULTS")

        print(
            f"Train Loss     : {train_loss:.4f}"
        )

        print(
            f"Train Accuracy : {train_accuracy:.2f}%"
        )

        print(
            f"Val Loss       : {val_loss:.4f}"
        )

        print(
            f"Val Accuracy   : {val_accuracy:.2f}%"
        )

        # SAVE BEST MODEL
        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "class_names":
                        class_names,

                    "image_size":
                        IMAGE_SIZE,

                    "architecture":
                        "EfficientNet-B0"
                },
                best_model_path
            )

            print(
                "\nBEST MODEL SAVED ✅"
            )

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    checkpoint = torch.load(
        best_model_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # ========================================================
    # FINAL TEST
    # ========================================================

    print("\n")
    print("=" * 60)
    print("FINAL TEST")
    print("=" * 60)

    test_loss, test_accuracy, y_true, y_pred = evaluate(
        model,
        test_loader,
        criterion,
        device,
        "Testing"
    )

    print(
        f"\nTest Loss     : {test_loss:.4f}"
    )

    print(
        f"Test Accuracy : {test_accuracy:.2f}%"
    )

    print(
        f"Best Val Acc  : {best_val_accuracy:.2f}%"
    )

    # ========================================================
    # REPORT
    # ========================================================

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )

    report_path = os.path.join(
        RESULTS_DIR,
        "classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"Test Accuracy: {test_accuracy:.4f}%\n\n"
        )

        f.write(report)

    # ========================================================
    # CLASS NAMES
    # ========================================================

    with open(
        os.path.join(
            RESULTS_DIR,
            "class_names.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            class_names,
            f,
            indent=4
        )

    # ========================================================
    # HISTORY
    # ========================================================

    with open(
        os.path.join(
            RESULTS_DIR,
            "training_history.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            history,
            f,
            indent=4
        )

    # ========================================================
    # PLOTS
    # ========================================================

    save_curves(history)

    save_confusion_matrix(
        y_true,
        y_pred,
        class_names
    )

    # ========================================================
    # DONE
    # ========================================================

    print("\n")
    print("=" * 60)
    print("TRAINING COMPLETE 🚀")
    print("=" * 60)

    print(
        f"\nFinal Test Accuracy: "
        f"{test_accuracy:.2f}%"
    )

    print(
        f"\nModel:\n{best_model_path}"
    )

    print(
        "\nResults saved in:"
    )

    print(
        f"{RESULTS_DIR}"
    )


# ============================================================
# WINDOWS
# ============================================================

if __name__ == "__main__":

    multiprocessing.freeze_support()

    main()