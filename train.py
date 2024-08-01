import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model import ConvNeuralNetwork
from custom_dataset import CustomDataset

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

def evaluate_model(model, data_loader):
    model.eval()
    accuracy = 0
    count = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            y_pred = model(inputs)
            accuracy += (torch.argmax(y_pred, 1) == labels).float().sum()
            count += len(labels)
    return accuracy / count

def main():
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((150, 150)),
        transforms.ToTensor()
    ])

    # Load datasets
    from data_preparation import train_df, test_df, val_df

    train_dataset = CustomDataset(root_dir="./Dataset/chest_xray/train", annotation_file=train_df, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = CustomDataset(root_dir="./Dataset/chest_xray/test", annotation_file=test_df, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    val_dataset = CustomDataset(root_dir="./Dataset/chest_xray/val", annotation_file=val_df, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    # Model training and evaluation
    num_classes = 2
    img_size = 150
    model = ConvNeuralNetwork(num_classes, img_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.005, momentum=0.9)

    num_epochs = 10
    train_model(model, train_loader, criterion, optimizer, num_epochs)

    test_accuracy = evaluate_model(model, test_loader)
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")

    val_accuracy = evaluate_model(model, val_loader)
    print(f"Validation Accuracy: {val_accuracy*100:.2f}%")

    # Save model
    torch.save(model.state_dict(), "lungs.pth")

if __name__ == "__main__":
    main()
