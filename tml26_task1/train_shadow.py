import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.models import resnet18
import torchvision.transforms as transforms
from pathlib import Path
from typing import Tuple

BASE = Path(__file__).parent


class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]:
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)


class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int, int]:
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]


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
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
)

pub_ds = torch.load(BASE / "pub.pt", weights_only=False)
pub_ds.transform = transform


# Training Loop Function
def train_shadow(model_name, dataset, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    criterion = nn.CrossEntropyLoss()

    # Added weight_decay
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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

        scheduler.step()
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), BASE / f"{model_name}.pt")
    print(f"Saved {model_name}.pt\n")


if __name__ == "__main__":
    num_shadows = 5
    for i in range(1, num_shadows + 1):
        # Independent 50% split for every model
        subset, _ = random_split(pub_ds, [0.5, 0.5])
        train_shadow(f"shadow_{i}", subset, epochs=40)
