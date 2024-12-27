import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Training function
def train_model(train_path, valid_path, save_path, device, epochs=50):
    # Data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Datasets and loaders
    train_dataset = ImageFolder(train_path, transform=data_transforms['train'])
    valid_dataset = ImageFolder(valid_path, transform=data_transforms['valid'])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    # Load ResNet50 pre-trained model
    resnet = models.resnet50(pretrained=True)
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = nn.Linear(resnet.fc.in_features, len(train_dataset.classes))
    resnet = resnet.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet.fc.parameters(), lr=0.001)

    # Training loop
    for epoch in range(epochs):
        resnet.train()
        train_loss, train_correct = 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = resnet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels)

        # Validation phase
        resnet.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = resnet(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels)

        # Log results
        train_acc = train_correct.double() / len(train_dataset)
        val_acc = val_correct.double() / len(valid_dataset)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    # Save the model
    torch.save(resnet.state_dict(), save_path)
    print("Model training complete and saved!")

if __name__ == '__main__':
    base_path = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_path, "dataset", "Images", "Train")
    valid_path = os.path.join(base_path, "dataset", "Images", "Test")
    save_path = os.path.join(base_path, "models", "model_resnet50.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(train_path, valid_path, save_path, device)
