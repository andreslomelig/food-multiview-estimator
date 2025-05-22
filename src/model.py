import torch
import torch.nn as nn
import timm

# Backbone CNN using EfficientNet (or any model from timm)
class CNNBackbone(nn.Module):
    def __init__(self, output_dim=512, model_name="efficientnet_b0"):
        super().__init__()
        self.encoder = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.projector = nn.Linear(self.encoder.num_features, output_dim)

    def forward(self, x):
        x = self.encoder(x)
        return self.projector(x)

# Transformer encoder for cross-view attention
class CrossViewTransformer(nn.Module):
    def __init__(self, dim, num_heads=4, depth=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):  # x: [V, B, D]
        return self.transformer(x)

# Full multiview nutrition estimation model
class NutritionEstimator(nn.Module):
    def __init__(self, feat_dim=512, num_outputs=5, backbone_name="efficientnet_b0"):
        super().__init__()
        self.backbone = CNNBackbone(output_dim=feat_dim, model_name=backbone_name)
        self.cross_view = CrossViewTransformer(dim=feat_dim)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs)
        )

    def forward(self, x):  # x: [B, V, C, H, W]
        B, V, C, H, W = x.shape
        x = x.view(B * V, C, H, W)
        feats = self.backbone(x)  # [B*V, D]
        feats = feats.view(B, V, -1).permute(1, 0, 2)  # [V, B, D]
        fused = self.cross_view(feats)  # [V, B, D]
        agg = fused.mean(dim=0)  # [B, D]
        return self.head(agg)  # [B, 5]
