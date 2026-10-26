
from pathlib import Path
import argparse

import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18


# --------------------------------------------------
# 1. PATHS AND DEVICE
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "manipulated_image_model.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# 2. LOAD THE TRAINED MODEL
# --------------------------------------------------

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Run python src/train_model.py first."
        )

    # Only load checkpoints created by your own training script.
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    class_names = checkpoint["class_names"]
    image_size = checkpoint["image_size"]
    mean = checkpoint["mean"]
    std = checkpoint["std"]

    model = resnet18(weights=None)

    input_features = model.fc.in_features

    model.fc = nn.Linear(
        input_features,
        len(class_names)
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return model, class_names, image_transform


# --------------------------------------------------
# 3. PREDICT AN IMAGE
# --------------------------------------------------

def predict_image(image_path, model, class_names, image_transform):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    with Image.open(image_path) as image:
        image = image.convert("RGB")

    image_tensor = image_transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)[0]

        predicted_index = probabilities.argmax().item()
        predicted_label = class_names[predicted_index]
        confidence = probabilities[predicted_index].item()

    print("\n" + "=" * 45)
    print("MANIPULATED IMAGE DETECTION RESULT")
    print("=" * 45)

    print("Image:", image_path.name)

    if predicted_label == "authentic":
        print("Prediction: AUTHENTIC")
    else:
        print("Prediction: MANIPULATED")

    print(f"Model confidence: {confidence * 100:.2f}%")

    print("\nClass probabilities:")

    for index, name in enumerate(class_names):
        print(
            f"{name.capitalize()}: "
            f"{probabilities[index].item() * 100:.2f}%"
        )

    print("\nNote: Confidence is not a guarantee of correctness.")
    print("Performance on unfamiliar images may differ from the test set.")


# --------------------------------------------------
# 4. MAIN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Detect potentially manipulated images."
    )

    parser.add_argument(
        "image_path",
        nargs="?",
        help="Full path to the image you want to test."
    )

    args = parser.parse_args()

    if args.image_path:
        selected_image = args.image_path
    else:
        selected_image = input(
            "Enter the full path of the image to test: "
        ).strip().strip('"')

    model, class_names, image_transform = load_model()

    predict_image(
        selected_image,
        model,
        class_names,
        image_transform
    )
