# 定义PatchEmbedding类
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=1, embed_dim=64):
        super(PatchEmbedding, self).__init__()
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
    
    def forward(self, x):
        x = self.projection(x)  # 投影为嵌入表示
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # 将卷积层的输出展平并转置得到(B, num_patches, embed_dim)
        return x

# 定义ViT类
class ViT(nn.Module):
    def __init__(self, img_size=256, patch_size=32, in_channels=1, embed_dim=64, num_classes=5, num_heads=16, num_layers=4, hidden_dim=256, dropout=0.1):
        super(ViT, self).__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.positional_embedding = nn.Parameter(torch.randn(1, self.patch_embedding.num_patches + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, hidden_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.positional_embedding[:, :(self.patch_embedding.num_patches + 1)]
        x = self.dropout(x)

        x = self.transformer_encoder(x)
        x = x[:, 0]  # 取出CLS token的输出
        x = self.fc(x)
        return x
