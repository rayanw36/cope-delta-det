"""Baseline 5: BAFE — Box-Aligned Feature Extraction with BiLSTM (Duché et al., 2026).

Adapted from MPEG-4 to HEVC. Uses box-aligned feature extraction from
compressed-domain features + BiLSTM temporal head for temporal fusion.

This isolates the transformer vs BiLSTM comparison in the temporal fusion head.

Reference: Duché et al., "BAFE: Box-Aligned Feature Extraction for Video
Object Detection in Compressed Domain", 2026.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.box_utils import xyxy_to_xywh, xywh_to_xyxy


class BAFEEncoder(nn.Module):
    """Box-Aligned Feature Extraction from compressed-domain features.

    Extracts per-box features from MV and residual maps using RoI-Align,
    similar to the Δ-Det Encoder but without the cross-attention transformer.
    """

    def __init__(self, embed_dim=256, roi_output_size=7):
        super().__init__()
        self.embed_dim = embed_dim
        self.roi_output_size = roi_output_size

        # MV encoder
        self.mv_encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Residual encoder
        self.res_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2 * roi_output_size * roi_output_size,
                      embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, mv_tensor, residual_energy, boxes_xyxy,
                img_h=720, img_w=1280):
        """
        Args:
            mv_tensor: (B, 2, H, W) motion vectors
            residual_energy: (B, 1, H, W) residual energy
            boxes_xyxy: list of (N_i, 4) boxes per image

        Returns:
            list of (N_i, embed_dim) per-box features per image
        """
        B = mv_tensor.shape[0]

        # Encode feature maps
        mv_features = self.mv_encoder(mv_tensor)    # (B, D, H', W')
        res_features = self.res_encoder(residual_energy)  # (B, D, H', W')

        feat_h, feat_w = mv_features.shape[2], mv_features.shape[3]
        spatial_scale = feat_w / img_w

        all_features = []
        for b in range(B):
            boxes = boxes_xyxy[b]
            if boxes.numel() == 0:
                all_features.append(
                    torch.zeros(0, self.embed_dim, device=mv_tensor.device)
                )
                continue

            N = boxes.shape[0]
            batch_idx = torch.zeros(N, 1, device=boxes.device)
            rois = torch.cat([batch_idx, boxes], dim=1)

            # RoI-Align from both feature maps
            mv_roi = roi_align(mv_features[b:b+1], rois,
                               output_size=self.roi_output_size,
                               spatial_scale=spatial_scale, aligned=True)
            res_roi = roi_align(res_features[b:b+1], rois,
                                output_size=self.roi_output_size,
                                spatial_scale=spatial_scale, aligned=True)

            # Concatenate and fuse
            combined = torch.cat([mv_roi, res_roi], dim=1)  # (N, 2D, 7, 7)
            combined_flat = combined.flatten(1)  # (N, 2D*7*7)
            fused = self.fusion(combined_flat)  # (N, embed_dim)

            all_features.append(fused)

        return all_features


class BAFETemporalHead(nn.Module):
    """BiLSTM temporal head for BAFE.

    Processes per-box features across P-frames using a BiLSTM to capture
    temporal dependencies.
    """

    def __init__(self, embed_dim=256, hidden_dim=256, num_layers=2,
                 num_classes=10, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Combine anchor embedding with per-frame features
        self.input_proj = nn.Linear(embed_dim * 2, embed_dim)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        out_dim = 2 * hidden_dim

        self.box_head = nn.Sequential(
            nn.Linear(out_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 4)
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(out_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )

        self.class_head = nn.Sequential(
            nn.Linear(out_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, anchor_embedding, frame_features_sequence):
        """
        Args:
            anchor_embedding: (N, embed_dim) anchor box embedding
            frame_features_sequence: list of T tensors, each (N, embed_dim)

        Returns:
            list of T prediction dicts
        """
        N = anchor_embedding.shape[0]
        T = len(frame_features_sequence)

        if N == 0:
            device = anchor_embedding.device
            return [{
                'box_deltas': torch.zeros(0, 4, device=device),
                'confidence': torch.zeros(0, device=device),
                'class_scores': torch.zeros(0, self.num_classes, device=device)
            } for _ in range(T)]

        # Build sequence: concatenate anchor with each frame's features
        sequence = []
        for feat in frame_features_sequence:
            combined = torch.cat([anchor_embedding, feat], dim=1)
            projected = self.input_proj(combined)
            sequence.append(projected)

        # Stack into (N, T, embed_dim) sequence
        seq_tensor = torch.stack(sequence, dim=1)

        # BiLSTM
        lstm_out, _ = self.lstm(seq_tensor)  # (N, T, 2*hidden_dim)

        # Decode per-frame predictions
        predictions = []
        for t in range(T):
            h = lstm_out[:, t, :]  # (N, 2*hidden_dim)
            predictions.append({
                'box_deltas': self.box_head(h),
                'confidence': self.confidence_head(h).squeeze(-1),
                'class_scores': self.class_head(h)
            })

        return predictions


class BAFEDetector(nn.Module):
    """Complete BAFE detector adapted for HEVC."""

    def __init__(self, num_classes=10, embed_dim=256):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.encoder = BAFEEncoder(embed_dim=embed_dim)
        self.temporal_head = BAFETemporalHead(
            embed_dim=embed_dim, num_classes=num_classes
        )

        # Anchor embedding (same as CoPE-Δ-Det for fair comparison)
        self.anchor_mlp = nn.Sequential(
            nn.Linear(5 + num_classes, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, anchor_boxes_xywh, anchor_scores, anchor_labels,
                p_frame_mvs, p_frame_residuals):
        """
        Args:
            anchor_boxes_xywh: (N, 4) anchor boxes
            anchor_scores: (N,) scores
            anchor_labels: (N,) labels
            p_frame_mvs: list of (1, 2, H, W) MV tensors per P-frame
            p_frame_residuals: list of (1, 1, H, W) residual energy per P-frame

        Returns:
            list of prediction dicts per P-frame
        """
        N = anchor_boxes_xywh.shape[0]

        # Create anchor embeddings
        boxes_norm = anchor_boxes_xywh.clone()
        boxes_norm[:, 0] /= 1280.0
        boxes_norm[:, 1] /= 720.0
        boxes_norm[:, 2] /= 1280.0
        boxes_norm[:, 3] /= 720.0

        one_hot = torch.zeros(N, self.num_classes, device=anchor_boxes_xywh.device)
        if N > 0:
            one_hot.scatter_(1, anchor_labels.unsqueeze(1), 1.0)

        anchor_feat = torch.cat([boxes_norm, anchor_scores.unsqueeze(1), one_hot], dim=1)
        anchor_emb = self.anchor_mlp(anchor_feat)

        # Extract per-box features for each P-frame
        anchor_boxes_xyxy = xywh_to_xyxy(anchor_boxes_xywh)
        frame_features = []
        for mv, res in zip(p_frame_mvs, p_frame_residuals):
            box_feats = self.encoder(mv, res, [anchor_boxes_xyxy])
            frame_features.append(box_feats[0])  # (N, embed_dim)

        # Temporal fusion via BiLSTM
        predictions = self.temporal_head(anchor_emb, frame_features)

        return predictions
