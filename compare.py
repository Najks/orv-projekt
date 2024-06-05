import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
from collections import Counter

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)
    model.load_state_dict(torch.load('face_recognition_model.pth'))
    model = model.to(device)
    model.eval()

    test_images_folder = 'comparing'
    images = []

    for filename in os.listdir(test_images_folder):
        image_path = os.path.join(test_images_folder, filename)
        image = Image.open(image_path)
        image = data_transforms(image).unsqueeze(0)
        images.append(image)

    class_names = ['domen', 'nejc', 'nik', 'undefined']
    predictions = []
    confidence_threshold = 0.6

    with torch.no_grad():
        for image in images:
            image = image.to(device)
            outputs = model(image)
            probs = F.softmax(outputs, dim=1)
            max_prob, preds = torch.max(probs, 1)
            if max_prob.item() < confidence_threshold:
                predictions.append((class_names[-1], max_prob.item() * 100))  # 'undefined'
            else:
                predictions.append((class_names[preds.item()], max_prob.item() * 100))

    for filename, prediction in zip(os.listdir(test_images_folder), predictions):
        print(f"Image: {filename}, Predicted class: {prediction[0]}, Confidence: {prediction[1]:.2f}%")

    class_counts = Counter(prediction[0] for prediction in predictions)

    most_common_class = class_counts.most_common(1)[0]

    print(f"\nThe most common class is '{most_common_class[0]}' with {most_common_class[1]} occurrences.")
