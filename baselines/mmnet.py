"""Baseline 3: MMNet-style Feature Propagation (Wang et al., ICCV 2019).

Feature-level propagation using motion vectors and residuals to warp CNN
features from I-frames to P-frames. Core idea: warp deep features using MVs,
then add residual correction.

Reference: Wang et al., "Looking Fast and Slow: Memory-Guided Mobile Video
Object Detection", ICCV 2019.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
import torchvision.models as models

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.metrics import COCOMetrics, LatencyTracker, DecodeBudgetTracker


class FeatureWarper(nn.Module):
    """Warp feature maps using motion vectors (MMNet-style)."""

    def __init__(self):
        super().__init__()

    def forward(self, features, mv_tensor, feat_h, feat_w):
        """Warp features using motion vectors.

        Args:
            features: (B, C, H, W) feature maps from I-frame
            mv_tensor: (B, 2, H_mv, W_mv) motion vectors
            feat_h, feat_w: target feature map size

        Returns:
            (B, C, H, W) warped features
        """
        B = features.shape[0]

        # Resize MVs to feature map resolution
        mvs = F.interpolate(mv_tensor, size=(feat_h, feat_w),
                            mode='bilinear', align_corners=False)

        # Normalize MVs to [-1, 1] for grid_sample
        mvs_norm = mvs.clone()
        mvs_norm[:, 0] = mvs_norm[:, 0] / (feat_w / 2)  # x displacement
        mvs_norm[:, 1] = mvs_norm[:, 1] / (feat_h / 2)  # y displacement

        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, feat_h, device=features.device),
            torch.linspace(-1, 1, feat_w, device=features.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        # Add MV displacement to grid
        flow = mvs_norm.permute(0, 2, 3, 1)  # (B, H, W, 2)
        warped_grid = grid + flow

        # Warp features
        warped = F.grid_sample(features, warped_grid, mode='bilinear',
                               padding_mode='border', align_corners=True)
        return warped


class ResidualCorrector(nn.Module):
    """Residual correction network to refine warped features."""

    def __init__(self, feat_channels=256, residual_channels=1):
        super().__init__()
        self.residual_encoder = nn.Sequential(
            nn.Conv2d(residual_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, feat_channels, kernel_size=3, padding=1),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(feat_channels * 2, feat_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1),
        )

    def forward(self, warped_features, residual_energy):
        """
        Args:
            warped_features: (B, C, H, W) warped features
            residual_energy: (B, 1, H_r, W_r) residual energy map

        Returns:
            (B, C, H, W) corrected features
        """
        H, W = warped_features.shape[2], warped_features.shape[3]
        res_upsampled = F.interpolate(residual_energy, size=(H, W),
                                       mode='bilinear', align_corners=False)
        res_encoded = self.residual_encoder(res_upsampled)
        combined = torch.cat([warped_features, res_encoded], dim=1)
        corrected = self.fusion(combined)
        return warped_features + corrected  # Residual connection


class MMNetDetector(nn.Module):
    """MMNet-style compressed-domain detector.

    1. Extract deep features on I-frame using a backbone
    2. For P-frames, warp I-frame features using MVs + residual correction
    3. Run detection head on warped features
    """

    def __init__(self, num_classes=10, feat_channels=256):
        super().__init__()
        self.num_classes = num_classes

        # Feature backbone (ResNet-50 truncated at layer3)
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3
        )
        # layer3 outputs 1024 channels
        self.channel_proj = nn.Conv2d(1024, feat_channels, kernel_size=1)

        self.warper = FeatureWarper()
        self.corrector = ResidualCorrector(feat_channels=feat_channels)

        # Simple detection head
        self.det_head = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.cls_head = nn.Conv2d(feat_channels, num_classes, kernel_size=1)
        self.reg_head = nn.Conv2d(feat_channels, 4, kernel_size=1)

    def extract_iframe_features(self, iframe_rgb):
        """Extract deep features from I-frame.

        Args:
            iframe_rgb: (B, 3, H, W) normalized I-frame

        Returns:
            (B, feat_channels, H', W') feature maps
        """
        features = self.backbone(iframe_rgb)
        return self.channel_proj(features)

    def propagate_features(self, iframe_features, mv_tensor, residual_energy):
        """Propagate I-frame features to P-frame using MVs + residuals.

        Args:
            iframe_features: (B, C, H, W) I-frame features
            mv_tensor: (B, 2, H_mv, W_mv) motion vectors
            residual_energy: (B, 1, H_r, W_r) residual energy

        Returns:
            (B, C, H, W) propagated features for P-frame
        """
        H, W = iframe_features.shape[2], iframe_features.shape[3]
        warped = self.warper(iframe_features, mv_tensor, H, W)
        corrected = self.corrector(warped, residual_energy)
        return corrected

    def detect_from_features(self, features, boxes_xyxy):
        """Run detection head on features.

        For simplicity, refine provided anchor boxes rather than dense detection.

        Args:
            features: (B, C, H, W) feature maps
            boxes_xyxy: list of (N_i, 4) anchor boxes

        Returns:
            list of detection dicts
        """
        det_features = self.det_head(features)

        results = []
        for b in range(features.shape[0]):
            boxes = boxes_xyxy[b]
            if boxes.numel() == 0:
                results.append({
                    'boxes': torch.zeros(0, 4),
                    'scores': torch.zeros(0),
                    'labels': torch.zeros(0, dtype=torch.long)
                })
                continue

            # RoI-Align
            batch_idx = torch.zeros(boxes.shape[0], 1, device=boxes.device)
            rois = torch.cat([batch_idx, boxes], dim=1)
            spatial_scale = det_features.shape[-1] / 1280.0
            pooled = roi_align(det_features[b:b+1], rois, output_size=7,
                               spatial_scale=spatial_scale, aligned=True)

            # Classification and regression
            cls_out = self.cls_head(pooled).mean(dim=(2, 3))  # (N, num_classes)
            reg_out = self.reg_head(pooled).mean(dim=(2, 3))  # (N, 4)

            scores, labels = cls_out.softmax(dim=1).max(dim=1)
            refined_boxes = boxes + reg_out

            results.append({
                'boxes': refined_boxes,
                'scores': scores,
                'labels': labels
            })

        return results

    def forward(self, iframe_rgb, p_frame_mvs, p_frame_residuals, anchor_boxes):
        """Full forward pass for a GOP.

        Args:
            iframe_rgb: (1, 3, H, W) I-frame
            p_frame_mvs: list of (1, 2, H, W) MV tensors per P-frame
            p_frame_residuals: list of (1, 1, H, W) residual energy per P-frame
            anchor_boxes: list of (N, 4) anchor boxes from I-frame

        Returns:
            list of detection dicts per frame
        """
        # Extract I-frame features
        iframe_features = self.extract_iframe_features(iframe_rgb)

        results = []
        for mv, res in zip(p_frame_mvs, p_frame_residuals):
            # Propagate features
            p_features = self.propagate_features(iframe_features, mv, res)
            # Detect
            dets = self.detect_from_features(p_features, anchor_boxes)
            results.append(dets[0])

        return results
