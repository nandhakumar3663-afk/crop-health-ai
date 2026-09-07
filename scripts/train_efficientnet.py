import os
import random
import json
import copy

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset

from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from tqdm import tqdm


# =====================================================
# PATH CONFIGURATION
# =====================================================


DATASET_PATH = r"C:\Datasets\PlantVillage\PlantVillage-Dataset-master\raw\color"


MODEL_DIR = r"C:\Users\DELL\PycharmProjects\crop-health-ai\models"


RESULT_DIR = r"C:\Users\DELL\PycharmProjects\crop-health-ai\results"


os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


MODEL_PATH = os.path.join(
    MODEL_DIR,
    "efficientnet_plantvillage.pth"
)



# =====================================================
# TRAINING CONFIG
# =====================================================


IMAGE_SIZE = 224

BATCH_SIZE = 16

EPOCHS = 10

LR = 0.0001

SEED = 42



# =====================================================
# DEVICE
# =====================================================


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


print("="*60)
print("🌱 PLANT DISEASE DETECTION")
print("EfficientNet-B0 + PlantVillage")
print("="*60)


print("Device:", DEVICE)


if DEVICE.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0)
            .total_memory/1024**3,
            2
        ),
        "GB"
    )

else:

    print(
        "Running on CPU"
    )



# =====================================================
# SEED
# =====================================================


random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)



# =====================================================
# DATASET CHECK
# =====================================================


if not os.path.exists(DATASET_PATH):

    raise Exception(
        f"Dataset not found:\n{DATASET_PATH}"
    )


print("\nLoading dataset...")


dataset_original = datasets.ImageFolder(
    DATASET_PATH
)


class_names = dataset_original.classes


print(
    "Classes:",
    len(class_names)
)


print(
    "Images:",
    len(dataset_original)
)



# =====================================================
# TRANSFORMS
# =====================================================


train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(20),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        [0.485,0.456,0.406],

        [0.229,0.224,0.225]
    )

])



test_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE,IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        [0.485,0.456,0.406],

        [0.229,0.224,0.225]
    )

])



# =====================================================
# TRAIN VALIDATION TEST SPLIT
# =====================================================


targets = np.array(
    dataset_original.targets
)


indices = np.arange(
    len(dataset_original)
)



train_idx, temp_idx = train_test_split(

    indices,

    test_size=0.2,

    random_state=SEED,

    stratify=targets
)



val_idx, test_idx = train_test_split(

    temp_idx,

    test_size=0.5,

    random_state=SEED,

    stratify=targets[temp_idx]

)



print("\nDATA SPLIT")

print(
    "Train:",
    len(train_idx)
)

print(
    "Validation:",
    len(val_idx)
)

print(
    "Test:",
    len(test_idx)
)



# =====================================================
# DATASETS
# =====================================================


train_data = datasets.ImageFolder(

    DATASET_PATH,

    transform=train_transform

)


test_data = datasets.ImageFolder(

    DATASET_PATH,

    transform=test_transform

)



train_dataset = Subset(

    train_data,

    train_idx

)


val_dataset = Subset(

    test_data,

    val_idx

)


test_dataset = Subset(

    test_data,

    test_idx

)



# =====================================================
# DATALOADER
# =====================================================


train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0

)



val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0

)



test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0

)



# =====================================================
# MODEL
# =====================================================


print("\nLoading EfficientNet-B0...")


model = models.efficientnet_b0(

    weights=EfficientNet_B0_Weights.DEFAULT

)



features = model.classifier[1].in_features



model.classifier[1] = nn.Linear(

    features,

    len(class_names)

)



model = model.to(DEVICE)



# =====================================================
# LOSS OPTIMIZER
# =====================================================


criterion = nn.CrossEntropyLoss()



optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LR

)



# =====================================================
# TRAIN FUNCTION
# =====================================================


def train_epoch():

    model.train()

    total=0

    correct=0

    loss_sum=0


    for images,labels in tqdm(train_loader):

        images=images.to(DEVICE)

        labels=labels.to(DEVICE)



        optimizer.zero_grad()



        output=model(images)



        loss=criterion(

            output,

            labels

        )



        loss.backward()


        optimizer.step()



        loss_sum += loss.item()



        pred=torch.argmax(

            output,

            dim=1

        )


        correct += (

            pred==labels

        ).sum().item()


        total += labels.size(0)



    return (

        loss_sum/len(train_loader),

        correct/total*100

    )



# =====================================================
# VALIDATION
# =====================================================


def validate(loader):

    model.eval()

    correct=0

    total=0

    loss_sum=0



    with torch.no_grad():

        for images,labels in tqdm(loader):


            images=images.to(DEVICE)

            labels=labels.to(DEVICE)



            output=model(images)



            loss=criterion(

                output,

                labels

            )


            loss_sum += loss.item()



            pred=torch.argmax(

                output,

                dim=1

            )



            correct += (

                pred==labels

            ).sum().item()



            total += labels.size(0)



    return (

        loss_sum/len(loader),

        correct/total*100

    )



# =====================================================
# TRAINING LOOP
# =====================================================


best_accuracy=0



for epoch in range(EPOCHS):


    print("\n")

    print(
        f"Epoch {epoch+1}/{EPOCHS}"
    )



    train_loss,train_acc=train_epoch()



    val_loss,val_acc=validate(
        val_loader
    )



    print(
        f"""
Train Loss     : {train_loss:.4f}
Train Accuracy : {train_acc:.2f}%

Val Loss       : {val_loss:.4f}
Val Accuracy   : {val_acc:.2f}%
"""
    )



    if val_acc > best_accuracy:


        best_accuracy=val_acc


        torch.save(

            {

            "model":
            model.state_dict(),

            "classes":
            class_names

            },

            MODEL_PATH

        )


        print("✅ Best model saved")




# =====================================================
# FINAL TEST
# =====================================================


checkpoint=torch.load(

    MODEL_PATH,

    map_location=DEVICE

)



model.load_state_dict(

    checkpoint["model"]

)



model.eval()



y_true=[]

y_pred=[]



with torch.no_grad():


    for images,labels in tqdm(test_loader):


        images=images.to(DEVICE)



        output=model(images)



        pred=torch.argmax(

            output,

            dim=1

        )



        y_pred.extend(

            pred.cpu().numpy()

        )



        y_true.extend(

            labels.numpy()

        )





accuracy = accuracy_score(

    y_true,

    y_pred

)*100



print("\n")

print("="*60)

print(
    f"🎯 FINAL TEST ACCURACY : {accuracy:.2f}%"
)

print("="*60)



# =====================================================
# REPORT
# =====================================================


report=classification_report(

    y_true,

    y_pred,

    target_names=class_names

)



with open(

    os.path.join(
        RESULT_DIR,
        "classification_report.txt"
    ),

    "w"

) as f:

    f.write(report)



with open(

    os.path.join(
        RESULT_DIR,
        "classes.json"
    ),

    "w"

) as f:


    json.dump(

        class_names,

        f,

        indent=4

    )



print("\nDONE 🚀")

print(
    "Model saved:",
    MODEL_PATH
)