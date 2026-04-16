"""Baseline 4: Chen et al. HEVC Intra Compressed-Domain Detection (EUSIPCO 2021).

Detection using HEVC intra-frame compressed-domain features: partition depth maps,
intra prediction mode maps, and residual data with iterative restoration.

Reference: Chen et al., "Object Detection in HEVC Intra Compressed Images",
EUSIPCO 2021.

Note: This baseline operates on I-frames only (no temporal propagation).
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.metrics import COCOMetrics


class IterativeRestorationModule(nn.Module):
    """Iterative restoration of HEVC compressed-domain features.

    Progressively refines compressed-domain features to approximate
    pixel-domain features through iterative denoising.
    """

    def __init__(self, in_channels=3, hidden_channels=64, num_iterations=3):
        super().__init__()
        self.num_iterations = num_iterations

        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU()
        )

        # Shared restoration block (applied iteratively)
        self.restoration_block = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: (B, in_channels, H, W) compressed-domain features

        Returns:
            (B, hidden_channels, H, W) restored features
        """
        h = self.initial_conv(x)

        for _ in range(self.num_iterations):
            residual = self.restoration_block(h)
            h = self.relu(h + residual)

        return h


class HEVCIntraDetector(nn.Module):
    """HEVC intra compressed-domain object detector.

    Uses CU partition depth, intra prediction modes, and residual data
    as input features. Applies iterative restoration before detection.
    """

    def __init__(self, num_classes=10, hidden_channels=64):
        super().__init__()
        self.num_classes = num_classes

        # Input: partition_depth (1) + pred_mode (1) + residual_energy (1) = 3 channels
        self.restoration = IterativeRestorationModule(
            in_channels=3, hidden_channels=hidden_channels, num_iterations=3
        )

        # Detection backbone (lightweight)
        self.backbone = nn.Sequential(
            nn.Conv2d(hidden_channels, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # Detection heads
        self.cls_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

        self.reg_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 4, kernel_size=1)
        )

        self.objectness_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, partition_depth, pred_mode, residual_energy):
        """
        Args:
            partition_depth: (B, 1, H, W) CU partition depth map
            pred_mode: (B, 1, H, W) prediction mode map
            residual_energy: (B, 1, H, W) residual energy map

        Returns:
            dict with detection outputs
        """
        # Concatenate compressed-domain features
        x = torch.cat([partition_depth, pred_mode, residual_energy], dim=1)

        # Upsample to reasonable resolution
        x = F.interpolate(x, size=(180, 320), mode='bilinear', align_corners=False)

        # Iterative restoration
        restored = self.restoration(x)

        # Backbone
        features = self.backbone(restored)

        # Detection heads
        cls_map = self.cls_head(features)       # (B, num_classes, H', W')
        reg_map = self.reg_head(features)       # (B, 4, H', W')
        obj_map = self.objectness_head(features)  # (B, 1, H', W')

        return {
            'class_map': cls_map,
            'regression_map': reg_map,
            'objectness_map': obj_map,
            'features': features
        }

    def decode_detections(self, outputs, conf_threshold=0.25, img_h=720, img_w=1280):
        """Decode dense detection maps into box predictions.

        Args:
            outputs: dict from forward()
            conf_threshold: confidence threshold
            img_h, img_w: original image size

        Returns:
            list of detection dicts per image
        """
        cls_map = outputs['class_map']   # (B, C, H', W')
        reg_map = outputs['regression_map']  # (B, 4, H', W')
        obj_map = outputs['objectness_map']  # (B, 1, H', W')

        B, _, H, W = cls_map.shape
        results = []

        for b in range(B):
            # Get objectness scores
            obj_scores = obj_map[b, 0]  # (H, W)

            # Find high-confidence locations
            mask = obj_scores > conf_threshold
            if not mask.any():
                results.append({
                    'boxes': torch.zeros(0, 4, device=cls_map.device),
                    'scores': torch.zeros(0, device=cls_map.device),
                    'labels': torch.zeros(0, dtype=torch.long, device=cls_map.device)
                })
                continue

            # Get indices
            ys, xs = torch.where(mask)

            # Decode boxes (convert grid positions + offsets to image coords)
            stride_h = img_h / H
            stride_w = img_w / W

            cx = (xs.float() + 0.5) * stride_w + reg_map[b, 0, ys, xs] * stride_w
            cy = (ys.float() + 0.5) * stride_h + reg_map[b, 1, ys, xs] * stride_h
            w = reg_map[b, 2, ys, xs].exp() * stride_w
            h = reg_map[b, 3, ys, xs].exp() * stride_h

            boxes = torch.stack([
                cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            ], dim=1).clamp(min=0)
            boxes[:, 2].clamp_(max=img_w)
            boxes[:, 3].clamp_(max=img_h)

            # Class scores
            cls_scores = cls_map[b, :, ys, xs].T  # (N, num_classes)
            scores_per_class = cls_scores.softmax(dim=1)
            scores, labels = scores_per_class.max(dim=1)
            scores = scores * obj_scores[ys, xs]

            results.append({
                'boxes': boxes,
                'scores': scores,
                'labels': labels
            })

        return results
