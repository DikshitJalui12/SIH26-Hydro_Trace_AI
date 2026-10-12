"""
Training utility for Attention U-Net on SAR dual-pol oil spill detection.
Trains model on simulated multi-look C-band radar backscatter to segment dark formations.
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .attention_unet import AttentionUNet
from ..ingestion.normalization import SARNormalizer
from data.sample_generator import generate_synthetic_sar_scene

class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        dice_score = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1.0 - dice_score
        
        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss

def train_attention_unet(
    output_weights_path: str = "vision_ai/models/attention_unet_s1.pth",
    num_scenes: int = 12,
    epochs: int = 25,
    device: str = "cpu"
) -> AttentionUNet:
    """
    Synthesizes training SAR scenes and trains the Attention U-Net to segment oil slicks.
    """
    dev = torch.device(device)
    model = AttentionUNet(in_channels=2, out_channels=1, base_filters=16).to(dev)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = DiceBCELoss()
    normalizer = SARNormalizer()

    temp_dir = Path("data/train_temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate multi-scene training data
    train_tensors = []
    train_masks = []
    
    print(f"Synthesizing {num_scenes} training SAR scenes...")
    for i in range(num_scenes):
        tif_path = temp_dir / f"scene_{i}.tif"
        c_lat = 20.0 + (i * 1.5)
        c_lon = 50.0 + (i * 2.0)
        dn_vv, dn_vh, gt = generate_synthetic_sar_scene(
            str(tif_path), height=256, width=256, center_lat=c_lat, center_lon=c_lon
        )
        
        # Linear to dB
        vv_lin = (dn_vv.astype(np.float32) / 300.0)**2
        vh_lin = (dn_vh.astype(np.float32) / 300.0)**2
        vv_db = 10.0 * np.log10(np.maximum(vv_lin, 1e-7))
        vh_db = 10.0 * np.log10(np.maximum(vh_lin, 1e-7))
        
        norm_dual = normalizer.normalize_dual_pol(vv_db, vh_db)
        
        # Ground truth: 1 for oil spill (exclude biogenic = 2)
        binary_gt = (gt == 1).astype(np.float32)
        
        train_tensors.append(norm_dual)
        train_masks.append(binary_gt)

    X = torch.from_numpy(np.stack(train_tensors, axis=0)).to(dev) # (N, 2, 256, 256)
    Y = torch.from_numpy(np.stack(train_masks, axis=0)).unsqueeze(1).to(dev) # (N, 1, 256, 256)

    print(f"Training Attention U-Net on {dev} for {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, Y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            probs = torch.sigmoid(logits)
            pred_bin = (probs >= 0.5).float()
            intersection = (pred_bin * Y).sum().item()
            total = pred_bin.sum().item() + Y.sum().item()
            iou = (intersection / (total - intersection + 1e-6)) if (total - intersection) > 0 else 1.0
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | IoU: {iou:.4f}")

    # Save weights
    out_p = Path(output_weights_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(out_p))
    print(f"Trained model saved to {out_p}")
    return model

if __name__ == "__main__":
    train_attention_unet()
