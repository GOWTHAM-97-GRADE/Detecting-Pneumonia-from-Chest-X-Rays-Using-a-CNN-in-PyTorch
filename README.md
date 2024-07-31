# Pneumonia Detection from Chest X-Rays Using a CNN in PyTorch

This repository contains a convolutional neural network (CNN) implemented in PyTorch to detect pneumonia from chest X-ray images. The project includes data preprocessing, CNN model definition, training, evaluation, and saving the trained model.

## Dataset

The dataset used in this project is the [Chest X-ray Images (Pneumonia) dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

The dataset used in this project is the Chest X-ray Images (Pneumonia) dataset. The dataset is organized into three folders: `train`, `test`, and `val`, with two subfolders for each class (`PNEUMONIA` and `NORMAL`).

## Overview

This repository contains a PyTorch implementation of a CNN designed to classify chest X-ray images into two categories: pneumonia and normal. The project includes data preparation, model definition, training, and evaluation scripts.

## Key Functionalities

- **Data Preparation**: Processes images and labels into a format suitable for training and evaluation.
- **CNN Model**: Defines a Convolutional Neural Network for image classification.
- **Training Script**: Trains the CNN model and evaluates its performance on test and validation datasets.
- **Model Saving**: Saves the trained model for future use.

## Components

### Data Preparation

The `customDataset` class handles the loading and preprocessing of images. Images are resized, converted to grayscale, and transformed into tensors for training and evaluation.

### CNN Model

The `convNeuralNetwork` class defines a CNN with the following components:

- **Convolutional Layers**: Extract features from the input images.
- **Pooling Layers**: Reduce the spatial dimensions of feature maps.
- **Fully Connected Layers**: Classify the extracted features into the target classes (pneumonia or normal).
- **Activation Functions**: Apply ReLU activation to introduce non-linearity.

### Training Script

The training script performs the following:

- **Data Loading**: Uses `DataLoader` to handle training, validation, and testing datasets.
- **Model Training**: Uses Cross-Entropy Loss and Stochastic Gradient Descent (SGD) for optimization.
- **Accuracy Evaluation**: Computes accuracy on test and validation datasets after each epoch.

### Model Saving

The trained model is saved to a file named `lungs.pth` for future inference.

## Usage

### Define Hyperparameters

Set the hyperparameters for training the CNN:

```python
batch_size = 32
num_classes = 2
learning_rate = 0.001
num_epochs = 10
```

### Prepare the Data

Ensure your image data is organized in the `./Dataset/chest_xray` directory, with subdirectories for `train`, `test`, and `val`, each containing folders for `PNEUMONIA` and `NORMAL` images.

### Initialize and Train the Model

Run the training script:

```python
# Model, criterion, and optimizer
model = convNeuralNetwork(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)

# Training loop
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
    
    # Evaluate on test set
    model.eval()
    test_acc = 0
    count = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            y_pred = model(inputs)
            test_acc += (torch.argmax(y_pred, 1) == labels).float().sum()
            count += len(labels)
    test_acc /= count
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Test Accuracy: {test_acc*100:.2f}%")

# Evaluate on validation set
val_acc = 0
val_count = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        y_pred = model(inputs)
        val_acc += (torch.argmax(y_pred, 1) == labels).float().sum()
        val_count += len(labels)
val_acc /= val_count
print(f"Validation Accuracy: {val_acc*100:.2f}%")

# Save model
torch.save(model.state_dict(), "lungs.pth")
```

### Running the Code

To run the training and evaluation, ensure you have the required dependencies installed and execute the script in your Python environment:

```bash
python your_script_name.py
```

## Acknowledgments

This implementation follows best practices for image classification using CNNs and PyTorch. Special thanks to the PyTorch documentation and tutorials that provided foundational knowledge.
