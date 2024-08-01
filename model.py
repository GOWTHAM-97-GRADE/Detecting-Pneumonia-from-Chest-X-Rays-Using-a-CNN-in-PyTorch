import torch.nn as nn

class ConvNeuralNetwork(nn.Module):
    def __init__(self, num_classes, img_size):
        super(ConvNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, padding=3)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=11, padding=4)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self._initialize_fc_input_size(img_size)
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.maxpool1(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.maxpool2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out
