# multiview_effnet.py
import torch
import torch.nn as nn
from timm import create_model

class MultiviewEfficientNetB4(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = create_model('efficientnet_b4', pretrained=pretrained, features_only=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(1792, 512)  # EfficientNet-B4 final feature dim
        self.regressor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # [calories, protein, fat, carbs]
        )

    def forward(self, x):
        B, V, C, H, W = x.shape  # B: batch, V: views (4), C: channels
        x = x.view(B * V, C, H, W)
        feats = self.backbone(x)[-1]  # Use final feature map
        pooled = self.pool(feats).squeeze(-1).squeeze(-1)  # (B*V, 1792)
        projected = self.proj(pooled)
        projected = projected.view(B, V, -1)  # (B, 4, 512)
        fused = projected.mean(dim=1)  # Fusion by mean across views → (B, 512)
        out = self.regressor(fused)  # (B, 4)
        return out
