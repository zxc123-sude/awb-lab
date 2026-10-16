"""A compact fully convolutional global illuminant estimator."""

import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURE = "tiny_illuminant_v1"


class TinyIlluminantNet(nn.Module):
    """Learn positive local RGB and confidence, then pool valid spatial cells.

    Input is normalized linear RGB plus a fourth valid-coverage channel.
    Confidence-weighted local pooling is inspired by the general FC4 concept;
    this small architecture and preprocessing are independent, not a paper
    reproduction. No pretrained parameters or external weights are used.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        )
        self.local_rgb = nn.Conv2d(64, 3, kernel_size=1)
        self.confidence = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 4:
            raise ValueError("model expects N x 4 x H x W RGB plus valid coverage")
        features = self.features(image)
        local_rgb = F.softplus(self.local_rgb(features)) + 1e-6
        confidence = F.softplus(self.confidence(features)) + 1e-6
        coverage = F.adaptive_avg_pool2d(image[:, 3:4], features.shape[-2:])
        weights = confidence * coverage
        pooled = (local_rgb * weights).sum(dim=(2, 3)) / weights.sum(dim=(2, 3)).clamp_min(1e-12)
        return F.normalize(pooled, p=2, dim=1, eps=1e-12)


def cosine_loss(estimate, target):
    """Stable 1-cos loss; arccos is reserved for evaluation reporting."""
    target = F.normalize(target, p=2, dim=1, eps=1e-12)
    return (1.0 - (estimate * target).sum(dim=1).clamp(-1.0, 1.0)).mean()
