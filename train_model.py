
from pathlib import Path
import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

ImageFile.LOAD_TRUNCATED_IMAGES = False

# --------------------------------------------------
# 1. PROJECT SETTINGS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_DIR
    / "datasets"
    / "prepared"
    / "image_manifest_deduplicated.csv"
)

MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "manipulated_image_model.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.0001
RANDOM_SEED = 42

CLASS_NAMES = ["authentic", "tampered"]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# --------------------------------------------------
# 2. REPRODUCIBILITY
# --------------------------------------------------

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("MANIPULATED IMAGE DETECTION - MODEL TRAINING")
print("=" * 60)
print("Project:", PROJECT_DIR)
print("Device:", DEVICE)

if DEVICE.type == "cpu":
    print("Training will use your CPU.")
else:
    print("Training will use your NVIDIA CUDA GPU.")

# --------------------------------------------------
# 3. CHECK AND LOAD THE MANIFEST
# --------------------------------------------------

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(
        f"Manifest not found: {MANIFEST_PATH}\n"
        "Check that image_manifest_deduplicated.csv exists."
    )

df = pd.read_csv(MANIFEST_PATH)

print("\nManifest columns:", list(df.columns))
print("Total manifest rows:", len(df))

# Find the image path column.
path_candidates = [
    "path", "image_path", "filepath",
    "file_path", "image", "filename"
]

path_column = next(
    (column for column in path_candidates if column in df.columns),
    None
)

if path_column is None:
    raise ValueError(
        "Cannot identify the image path column. "
        "Expected a column such as 'path' or 'image_path'."
    )

if "label" not in df.columns or "split" not in df.columns:
    raise ValueError(
        "The manifest must contain 'label' and 'split' columns."
    )

# Normalize class labels to two standard names.
def normalize_label(value):
    text = str(value).strip().lower()

    if text in {
        "authentic", "au", "real", "original", "genuine", "0"
    }:
        return "authentic"

    if text in {
        "tampered", "tp", "manipulated", "fake", "forged", "1"
    }:
        return "tampered"

    raise ValueError(f"Unknown class label: {value!r}")


df["label"] = df["label"].apply(normalize_label)
df["split"] = df["split"].astype(str).str.strip().str.lower()
df[path_column] = df[path_column].astype(str)

required_splits = {"train", "validation", "test"}

if not required_splits.issubset(set(df["split"])):
    raise ValueError(
        "The manifest must contain train, validation and test splits."
    )

# Resolve relative paths against the project directory.
def resolve_image_path(value):
    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_DIR / path

    return str(path.resolve())


df["resolved_path"] = df[path_column].apply(resolve_image_path)

# Verify that files exist and can be opened.
print("\nChecking image files...")

valid_rows = []
bad_files = []

for row in df.itertuples(index=False):
    row_data = row._asdict()
    image_path = row_data["resolved_path"]

    try:
        with Image.open(image_path) as image:
            image.verify()

        valid_rows.append(row_data)

    except Exception as error:
        bad_files.append((image_path, str(error)))

if bad_files:
    print(f"\nUnreadable or missing files: {len(bad_files)}")
    for path, error in bad_files[:10]:
        print(path, "->", error)

    raise RuntimeError(
        "Fix the unreadable files and update the manifest before training."
    )

df = pd.DataFrame(valid_rows)

for split_name in ["train", "validation", "test"]:
    subset = df[df["split"] == split_name]

    print(f"\n{split_name.upper()}: {len(subset)} images")
    print(subset["label"].value_counts().to_dict())

    if set(subset["label"]) != set(CLASS_NAMES):
        raise ValueError(
            f"{split_name} must contain both authentic and tampered images."
        )

# --------------------------------------------------
# 4. IMAGE TRANSFORMATIONS
# --------------------------------------------------

# ImageNet normalization is required for the pretrained ResNet18.
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

evaluation_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# --------------------------------------------------
# 5. CREATE A PYTORCH DATASET
# --------------------------------------------------

class ManipulatedImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_path = row["resolved_path"]
        label_name = row["label"]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = CLASS_NAMES.index(label_name)

        return image, label


train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()

train_dataset = ManipulatedImageDataset(
    train_df, train_transform
)

val_dataset = ManipulatedImageDataset(
    val_df, evaluation_transform
)

test_dataset = ManipulatedImageDataset(
    test_df, evaluation_transform
)

# num_workers=0 is a safe starting point on Windows.
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

print("\nData loaders created successfully.")

# --------------------------------------------------
# 6. LOAD A PRETRAINED CNN
# --------------------------------------------------

print("\nLoading pretrained ResNet18...")

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# Freeze most pretrained layers initially.
for parameter in model.parameters():
    parameter.requires_grad = False

# Fine-tune the final residual block.
for parameter in model.layer4.parameters():
    parameter.requires_grad = True

# Replace the original 1000-class output layer with 2 classes.
input_features = model.fc.in_features

model.fc = nn.Linear(input_features, len(CLASS_NAMES))

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    filter(lambda parameter: parameter.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)

# --------------------------------------------------
# 7. TRAINING AND VALIDATION
# --------------------------------------------------

best_val_loss = float("inf")
best_state = copy.deepcopy(model.state_dict())

print("\nStarting training...\n")

for epoch in range(EPOCHS):

    # Training phase.
    model.train()

    train_loss_total = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss_total += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)

    train_loss = train_loss_total / train_total
    train_accuracy = train_correct / train_total

    # Validation phase.
    model.eval()

    val_loss_total = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss_total += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss_total / val_total
    val_accuracy = val_correct / val_total

    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"  Train loss     : {train_loss:.4f}")
    print(f"  Train accuracy : {train_accuracy:.4f}")
    print(f"  Val loss       : {val_loss:.4f}")
    print(f"  Val accuracy   : {val_accuracy:.4f}")

    # Keep the model with the lowest validation loss.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())

        torch.save(
            {
                "model_state_dict": best_state,
                "class_names": CLASS_NAMES,
                "image_size": IMAGE_SIZE,
                "mean": mean,
                "std": std
            },
            MODEL_PATH
        )

        print("  Best model saved.")

    print("-" * 45)

# Restore the best validation checkpoint.
model.load_state_dict(best_state)
model.eval()

# --------------------------------------------------
# 8. FINAL TEST EVALUATION
# --------------------------------------------------

print("\nEvaluating on the untouched test split...")

all_true = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)

        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu().numpy()

        all_predictions.extend(predictions)
        all_true.extend(labels.numpy())

test_accuracy = accuracy_score(all_true, all_predictions)

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

print("\nClassification report:")
print(
    classification_report(
        all_true,
        all_predictions,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        zero_division=0
    )
)

print("Confusion matrix (rows=true, columns=predicted):")
print(confusion_matrix(all_true, all_predictions, labels=[0, 1]))

print("\nTrained model saved at:")
print(MODEL_PATH)

print("\nTraining and evaluation completed.")
