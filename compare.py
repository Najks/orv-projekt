import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
from collections import Counter
from facenet_pytorch import InceptionResnetV1

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model = InceptionResnetV1(pretrained='vggface2').eval()
    num_ftrs = model.last_linear.in_features
    model.last_linear = nn.Linear(num_ftrs, 512)
    model.classifier = nn.Linear(512, 4)
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

    class_names = ['domen', 'nejc', 'nik', 'unknown']
    predictions = []

    with torch.no_grad():
        for image in images:
            image = image.to(device)
            outputs = model.classifier(model(image))
            probs = F.softmax(outputs, dim=1)
            max_prob, preds = torch.max(probs, 1)
            predictions.append((class_names[preds.item()], max_prob.item() * 100))

    for filename, prediction in zip(os.listdir(test_images_folder), predictions):
        print(f"Image: {filename}, Predicted class: {prediction[0]}, Confidence: {prediction[1]:.2f}%")

    class_counts = Counter(prediction[0] for prediction in predictions)

    most_common_class = class_counts.most_common(1)[0]

    print(f"\nThe most common class is '{most_common_class[0]}' with {most_common_class[1]} occurrences.")