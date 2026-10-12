"""
Attention U-Net architecture for dual-polarization (VV/VH) SAR Oil Spill Segmentation.
Implements spatial/channel additive attention gates to focus on dark low-backscatter
marine oil slicks while suppressing homogeneous ocean clutter and wind streaks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    Additive Attention Gate (Oktay et al., 2018).
    Conditions skip-connection features 'x' on gating signal 'g' from deeper layers.
    Suppresses irrelevant ocean clutter background while highlighting low-backscatter slick contours.
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Match spatial dimensions if needed
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)
            
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        net = self.relu(g1 + x1)
        alpha = self.psi(net)
        return x * alpha


class AttentionUNet(nn.Module):
    """
    Attention U-Net configured for dual-polarization C-band SAR input (VV, VH).
    Outputs 1-channel logit / probability map for oceanic oil slicks.
    """
    def __init__(self, in_channels: int = 2, out_channels: int = 1, base_filters: int = 32):
        super().__init__()
        f = base_filters

        # Encoder
        self.conv1 = DoubleConv(in_channels, f)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = DoubleConv(f, f * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = DoubleConv(f * 2, f * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = DoubleConv(f * 4, f * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = DoubleConv(f * 8, f * 16)

        # Decoder with Attention Gates
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.att4 = AttentionGate(F_g=f * 8, F_l=f * 8, F_int=f * 4)
        self.conv_up4 = DoubleConv(f * 16, f * 8)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=f * 4, F_l=f * 4, F_int=f * 2)
        self.conv_up3 = DoubleConv(f * 8, f * 4)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=f * 2, F_l=f * 2, F_int=f)
        self.conv_up2 = DoubleConv(f * 4, f * 2)

        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=f, F_l=f, F_int=f // 2)
        self.conv_up1 = DoubleConv(f * 2, f)

        # Final 1x1 convolution
        self.out_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        # Center Bottleneck
        bn = self.bottleneck(p4)

        # Decoder Stage 4
        d4 = self.up4(bn)
        x4 = self.att4(g=d4, x=c4)
        d4 = torch.cat([x4, d4], dim=1)
        d4 = self.conv_up4(d4)

        # Decoder Stage 3
        d3 = self.up3(d4)
        x3 = self.att3(g=d3, x=c3)
        d3 = torch.cat([x3, d3], dim=1)
        d3 = self.conv_up3(d3)

        # Decoder Stage 2
        d2 = self.up2(d3)
        x2 = self.att2(g=d2, x=c2)
        d2 = torch.cat([x2, d2], dim=1)
        d2 = self.conv_up2(d2)

        # Decoder Stage 1
        d1 = self.up1(d2)
        x1 = self.att1(g=d1, x=c1)
        d1 = torch.cat([x1, d1], dim=1)
        d1 = self.conv_up1(d1)

        logits = self.out_conv(d1)
        return logits
