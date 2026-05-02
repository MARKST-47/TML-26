import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import resnet18
import torchvision.transforms as transforms
from pathlib import Path

BASE = Path(__file__).parent

MEAN = [0.7406, 0.5331, 0.7059]
STD = [0.1491, 0.1864, 0.1301]


class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids, self.imgs, self.labels = [], [], []
        self.transform = transform

    def __getitem__(self, i):
        img = self.transform(self.imgs[i]) if self.transform else self.imgs[i]
        return self.ids[i], img, self.labels[i]

    def __len__(self):
        return len(self.ids)


class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, i):
        id_, img, label = super().__getitem__(i)
        return id_, img, label, self.membership[i]


def get_model():
    m = resnet18(weights=None)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, 9)
    return m


transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
)

pub_ds = torch.load(BASE / "pub.pt", weights_only=False)

# Only train on NON-MEMBERS (membership == 0)
nonmember_idx = [i for i, m in enumerate(pub_ds.membership) if m == 0]
print(f"Non-members in pub.pt: {len(nonmember_idx)}")

pub_ds.transform = transform_train
ref_subset = Subset(pub_ds, nonmember_idx)
loader = DataLoader(ref_subset, batch_size=128, shuffle=True, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model().to(device)

EPOCHS = 50
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for _, imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {total_loss / len(loader):.4f}")

torch.save(model.state_dict(), BASE / "reference.pt")
print("Saved reference.pt")
