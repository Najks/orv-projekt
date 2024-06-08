import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
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

    data_dir = 'learning'
    image_dataset = datasets.ImageFolder(data_dir, data_transforms)
    dataloader = DataLoader(image_dataset, batch_size=32, shuffle=True, num_workers=4)
    dataset_size = len(image_dataset)
    class_names = image_dataset.classes

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 512)
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, len(class_names))
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # underfitting = increase lr, decrease weight decay
    # overfitting = decrease lr, increase weight decay
    optimizer = optim.Adam(model.parameters(), lr=0.000005, weight_decay=0.02)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    from tqdm import tqdm

    def train_model(model, criterion, optimizer, dataloader, dataset_size, num_epochs=50, patience=3):
        best_model_wts = model.state_dict()
        best_acc = 0.0
        no_improvement_count = 0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            model.train()

            running_loss = 0.0
            running_corrects = 0

            pbar = tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}", dynamic_ncols=True)

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model.classifier(model(inputs))
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                pbar.set_postfix({'loss': running_loss / ((pbar.n + 1) * dataloader.batch_size)})
                pbar.update()

            pbar.close()

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            scheduler.step(epoch_loss)

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

    model = train_model(model, criterion, optimizer, dataloader, dataset_size, num_epochs=50, patience=5)

    torch.save(model.state_dict(), 'face_recognition_model.pth')
