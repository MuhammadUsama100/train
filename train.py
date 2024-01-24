# train.py
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)

def train():
    # Your training logic here
    model = SimpleModel()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Simulated training loop
    for epoch in range(10):
        x = torch.rand(10, 1)
        y_true = 2 * x + 1

        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()

        # Log metrics and parameters to MLflow
        mlflow.log_param("epoch", epoch)
        mlflow.log_metric("loss", loss.item())

    # Save the trained model
    mlflow.pytorch.log_model(model, "model")

if __name__ == "__main__":
    # Start an MLflow run
    with mlflow.start_run():
        train()

