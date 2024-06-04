import os
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import random_split, DataLoader
from torch.utils.data import ConcatDataset

transform = transforms.Compose([
    transforms.toTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_dir = 'learning'
classes = ['domen', 'nejc', 'nik']
datasets = {x: datasets.ImageFolder(os.path.join(data_dir, f'learning_{x}'), transform) for x in classes}

all_datasets = ConcatDataset([datasets[x] for x in classes])

total_size = len(all_datasets)
train_size = int(0.8 * total_size)
val_size = total_size - train_size

train_dataset, val_dataset = random_split(all_datasets, [train_size, val_size])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = models.resnet50(pretrained=True)

num_fts = model.fc.in_features
model.fc = nn.Linear(num_fts, len(classes))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}')

torch.save(model.state_dict(), 'model.pth')