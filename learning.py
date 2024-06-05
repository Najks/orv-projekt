import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the dataset
    data_dir = 'learning'
    image_dataset = datasets.ImageFolder(data_dir, data_transforms)
    dataloader = DataLoader(image_dataset, batch_size=32, shuffle=True, num_workers=4)
    dataset_size = len(image_dataset)
    class_names = image_dataset.classes

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    def train_model(model, criterion, optimizer, dataloader, dataset_size, num_epochs=25, patience=5):
        best_model_wts = model.state_dict()
        best_acc = 0.0
        no_improvement_count = 0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            model.train()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= patience:
                print(f"No improvement in {patience} epochs. Early stopping...")
                break

        model.load_state_dict(best_model_wts)
        return model

    model = train_model(model, criterion, optimizer, dataloader, dataset_size, num_epochs=25, patience=5)

    torch.save(model.state_dict(), 'face_recognition_model.pth')
