
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

# -------------------------------
# Pretrained ViT Model
# -------------------------------
class BloodCellViT(nn.Module):
    def __init__(self, num_classes=4):
        super(BloodCellViT, self).__init__()
        weights = ViT_B_16_Weights.DEFAULT
        self.vit = vit_b_16(weights=weights)
        num_ftrs = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.vit(x)

# -------------------------------
# Custom ViT Model
# -------------------------------
class CustomViT(nn.Module):
    def __init__(self, num_classes=4):
        super(CustomViT, self).__init__()
        weights = ViT_B_16_Weights.DEFAULT
        self.vit = vit_b_16(weights=weights)
        num_ftrs = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.vit(x)

# -------------------------------
# Performer Model (Example)
# -------------------------------
class PerformerModel(nn.Module):
    def __init__(self, num_classes=4):
        super(PerformerModel, self).__init__()
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        return self.fc(x)

# -------------------------------
# GAN Generator (Example)
# -------------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 3*64*64),  # 64x64 RGB output
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 3, 64, 64)
        return img
