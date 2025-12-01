
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

# -------------------------------
# Pretrained ViT Model
# -------------------------------
class BloodCellViT(nn.Module):
    def __init__(self, num_classes=4):
        super(BloodCellViT, self).__init__()
        # Load pre-trained ViT-Base/16
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
        # Load pre-trained ViT-Base/16
        weights = ViT_B_16_Weights.DEFAULT
        self.vit = vit_b_16(weights=None)
        num_ftrs = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.vit(x)

# -------------------------------
# Performer Model (Example)
# -------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
def linear_attention(q, k, v):
    attention_weights = torch.matmul(q, k.transpose(-2, -1))
    attention_weights = F.softmax(attention_weights, dim=-1)
    output = torch.matmul(attention_weights, v)
    return output

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class PerformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_drop_rate, batch_first=True) # Using standard MultiheadAttention for now, would replace with efficient attention
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(drop_rate)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class PerformerModel(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, num_layers=12, num_heads=12, mlp_ratio=4., num_classes=4, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            PerformerBlock(embed_dim, num_heads, mlp_ratio, drop_rate, attn_drop_rate)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


# -------------------------------
# GAN Generator (Example)
# -------------------------------
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, z_dim=100, ngf=64, nc=3):
        super().__init__()
        self.net = nn.Sequential(
            # z -> ngf*16 x 4 x 4
            nn.ConvTranspose2d(z_dim, ngf*16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*16), nn.ReLU(True),

            # -> ngf*8 x 8 x 8
            nn.ConvTranspose2d(ngf*16, ngf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*8), nn.ReLU(True),

            # -> ngf*4 x 16 x 16
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4), nn.ReLU(True),

            # -> ngf*2 x 32 x 32
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2), nn.ReLU(True),

            # -> ngf x 64 x 64
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),

            # -> nc x 128 x 128
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf*8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*8, ndf*16, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf*16), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1, 1).squeeze(1)

