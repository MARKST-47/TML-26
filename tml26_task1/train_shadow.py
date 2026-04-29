import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18
import torchvision.transforms as transforms
from pathlib import Path

BASE = Path(__file__).parent


# Model Architecture
def get_model():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 9)
    return model


# Normalize Data
MEAN = [0.7406, 0.5331, 0.7059]
STD = [0.1491, 0.1864, 0.1301]
transform = transforms.Compose(
    [
        transforms.Resize(32),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
)

pub_ds = torch.load(BASE / "pub.pt", weights_only=False)
pub_ds.transform = transform


# Training Loop Function
def train_shadow(model_name, dataset, epochs=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Training {model_name}")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for _, imgs, labels, _ in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), BASE / f"{model_name}.pt")
    print(f"Saved {model_name}.pt\n")


# Splitting data and train 2 models
if __name__ == "__main__":
    subset_1, subset_2 = random_split(pub_ds, [0.5, 0.5])
    train_shadow("shadow_1", subset_1)
    train_shadow("shadow_2", subset_2)
