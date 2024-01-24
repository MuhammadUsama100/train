# train.py

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader

# Define a simple neural network
class SimpleModel(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleModel, self).__init__()
        self.resnet = resnet18(pretrained=False)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Function to train the model
def train_model():
    # Your training logic here

    # Example: Training a simple model
    model = SimpleModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Your training loop here

    # Example: Saving the model
    torch.save(model.state_dict(), "model.pth")

    # Log the model to MLflow
    mlflow.pytorch.log_model(model, "model")

if __name__ == "__main__":
    train_model()

