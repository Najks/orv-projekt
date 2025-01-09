import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 512)
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 3)
)
model.eval()

model.load_state_dict(torch.load('face_recognition_model.pth'))
model = model.to(device)

class_mapping = {'domen': 1, 'nik': 2, 'nejc': 3}

def compare_images(image_paths, threshold=50):
    images = []
    for image_path in image_paths:
        image = Image.open(image_path)
        image = data_transforms(image).unsqueeze(0)
        images.append(image)

    class_names = ['domen', 'nejc', 'nik']
    predictions = []
    total_confidence = 0

    with torch.no_grad():
        confident_predictions = []
        for image in images:
            image = image.to(device)
            outputs = model.classifier(model(image))
            probs = F.softmax(outputs, dim=1)
            max_prob, preds = torch.max(probs, 1)
            confidence = max_prob.item() * 100
            total_confidence += confidence
            if preds.item() >= len(class_names):
                print(f"Warning: Model predicted an index {preds.item()} outside of class names range")
                continue  # Skip invalid predictions

            if confidence > threshold:
                confident_predictions.append((class_names[preds.item()], confidence))
            predictions.append((class_names[preds.item()], confidence))
            print(f"Image result: {class_names[preds.item()]}, Accuracy: {confidence}%")

        if len(predictions) == 0:
            print("No valid predictions were made.")
            return 0  # No valid predictions

        average_confidence = total_confidence / len(predictions)
        print(f"Average confidence: {average_confidence}%")

        if len(confident_predictions) < len(predictions) / 2:
            return 0
        else:
            class_counts = Counter(prediction[0] for prediction in confident_predictions)
            most_common_class = class_counts.most_common(1)[0]
            return class_mapping[most_common_class[0]]

if __name__ == '__main__':
    # Example usage
    test_images_folder = 'comparing'
    images = [os.path.join(test_images_folder, f) for f in os.listdir(test_images_folder)]
    result = compare_images(images)
    print(result)
