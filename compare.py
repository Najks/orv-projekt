import torch
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
import random
import projekt_orv

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(['domen', 'nejc', 'nik']))
model.load_state_dict(torch.load('model.pth'))
model.eval()

total_scores = np.zeros(len(['domen', 'nejc', 'nik']))
count = 0

image_dir = 'learning/learning_nik'
image_names = os.listdir(image_dir)

random_image_names = random.sample(image_names, 10)

for image_name in random_image_names:
    image_path = os.path.join(image_dir, image_name)
    image = Image.open(image_path).convert('RGB')

    image = transform(image).unsqueeze(0)

    output = model(image)

    probabilities = torch.nn.functional.softmax(output, dim=1).detach().numpy()

    probabilities = probabilities.flatten()

    total_scores += probabilities

    count += 1

average_scores = total_scores / count

predicted_class = np.argmax(average_scores)

recognition_threshold = 0.5

if average_scores[predicted_class] < recognition_threshold:
    print("The model cannot recognize the person.")
else:
    class_mapping = {0: 'domen', 1: 'nejc', 2: 'nik'}
    predicted_class_name = class_mapping[predicted_class]

    print(f"The model predicts class {predicted_class_name} with average score {average_scores[predicted_class]*100:.2f}%")

if __name__ == '__main__':
    projekt_orv.preprocess_image()
    projekt_orv.augment_dataset()