import os
import shutil
import random
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, models
import projekt_orv

# Define the classes
classes = ['domen', 'nejc', 'nik']

# Define the model architecture
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(classes))

# Load the saved state dictionary
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_image(image_path, model, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
    softmax = torch.nn.functional.softmax(outputs, dim=1)
    _, predicted = torch.max(outputs, 1)
    return predicted.item(), softmax.cpu().numpy()[0]

def compare():
    total_scores = np.zeros(len(classes))
    total_probabilities = np.zeros(len(classes))
    count = 0

    image_dir = 'augmented'
    image_names = os.listdir(image_dir)

    for image_name in image_names:
        image_path = os.path.join(image_dir, image_name)
        predicted_class, probabilities = predict_image(image_path, model, transform)
        total_scores[predicted_class] += 1
        total_probabilities[predicted_class] += probabilities[predicted_class]
        count += 1
        print(f'Image: {image_name}, Predicted Class: {classes[predicted_class]}, Probability: {probabilities[predicted_class]}')

    average_probabilities = total_probabilities / total_scores
    max_class_index = np.argmax(total_scores)
    print('Total Scores:', total_scores)
    print(f'Class with most count: {classes[max_class_index]}, Average Probability: {average_probabilities[max_class_index]}')

def clear_directory(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

if __name__ == '__main__':
    projekt_orv.preprocess_dataset()
    projekt_orv.augment_dataset()
    compare()
    clear_directory('processed')
    clear_directory('augmented')
