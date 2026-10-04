"""
PyTorch 3-Class U-Net Semantic Segmentation Architecture.

Designed for Banana Foliar Disease Severity Assessment:
- Class 0: Background
- Class 1: Healthy Cavendish leaf lamina
- Class 2: Diseased foliar lesion (Sigatoka streak, Cordana patch, BBTV symptom)

Architecture:
- 4 Encoder levels with DoubleConv (Conv-BN-ReLU x 2) and MaxPool2d
- Central Bottleneck (1024 channels for base_filters=64)
- 4 Decoder levels with transposed convolutions and skip-connections
- 1x1 convolution classification head yielding 3 output channel logits
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv with skip connection"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad if dimension mismatch due to odd dimensions
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        if diff_x != 0 or diff_y != 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])

        # Concatenate along channel axis (skip connection)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution to map feature maps to class logits"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    3-Class PyTorch U-Net for Banana Foliar Disease Segmentation.
    """

    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 3,
        base_filters: int = 64,
        bilinear: bool = False
    ):
        """
        Args:
            n_channels: Input image channels (3 for RGB).
            n_classes: Number of target segmentation classes (3: background, leaf, lesion).
            base_filters: Base number of feature channels (default 64).
            bilinear: If True, uses bilinear interpolation instead of transposed convolutions.
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.base_filters = base_filters
        self.bilinear = bilinear

        f = base_filters

        # Encoder path
        self.inc = DoubleConv(n_channels, f)
        self.down1 = Down(f, f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(f * 8, f * 16 // factor)

        # Decoder path
        self.up1 = Up(f * 16, f * 8 // factor, bilinear)
        self.up2 = Up(f * 8, f * 4 // factor, bilinear)
        self.up3 = Up(f * 4, f * 2 // factor, bilinear)
        self.up4 = Up(f * 2, f, bilinear)

        # Output head
        self.outc = OutConv(f, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass yielding unnormalized class logits.

        Args:
            x: Tensor of shape (B, 3, H, W).

        Returns:
            Logits tensor of shape (B, n_classes, H, W).
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs inference and returns discrete class labels per pixel.

        Args:
            x: Tensor of shape (B, 3, H, W) or (3, H, W).

        Returns:
            Class tensor of shape (B, H, W) or (H, W) with values in {0, 1, 2}.
        """
        was_single = False
        if x.ndim == 3:
            was_single = True
            x = x.unsqueeze(0)

        self.eval()
        logits = self.forward(x)
        preds = torch.argmax(logits, dim=1)  # (B, H, W)

        if was_single:
            return preds.squeeze(0)
        return preds
