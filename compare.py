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
    _, predicted = torch.max(outputs, 1)
    return predicted.item()


def compare():
    total_scores = np.zeros(len(classes))
    class_correct = list(0. for _ in range(len(classes)))
    class_total = list(0. for _ in range(len(classes)))

    image_dir = 'augmented'
    image_names = os.listdir(image_dir)

    for image_name in image_names:
        image_path = os.path.join(image_dir, image_name)
        predicted_class = predict_image(image_path, model, transform)
        true_class = classes.index(image_name.split('_')[0])  # Extract true class from image name
        total_scores[predicted_class] += 1
        if predicted_class == true_class:
            class_correct[predicted_class] += 1
        class_total[true_class] += 1

    # Print individual class accuracies
    for i, cls in enumerate(classes):
        accuracy = 100 * class_correct[i] / class_total[i]
        print(f'Accuracy for {cls}: {accuracy:.2f}%')

    # Calculate overall accuracy
    total_correct = sum(class_correct)
    total_images = sum(class_total)
    overall_accuracy = 100 * total_correct / total_images
    print(f'Overall Accuracy: {overall_accuracy:.2f}%')

    print('Total Scores:', total_scores)



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
