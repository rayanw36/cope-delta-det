"""Box utility functions: IoU, GIoU, RoI-Align wrappers, NMS, box format conversions."""

import torch
import torch.nn as nn
from torchvision.ops import roi_align, nms, box_iou


def xyxy_to_xywh(boxes):
    """Convert (x1, y1, x2, y2) to (cx, cy, w, h)."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def xywh_to_xyxy(boxes):
    """Convert (cx, cy, w, h) to (x1, y1, x2, y2)."""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def compute_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes in xyxy format.

    Args:
        boxes1: (N, 4) tensor
        boxes2: (M, 4) tensor

    Returns:
        (N, M) IoU matrix
    """
    return box_iou(boxes1, boxes2)


def compute_giou(boxes1, boxes2):
    """Compute Generalized IoU between paired boxes.

    Args:
        boxes1: (N, 4) in xyxy format
        boxes2: (N, 4) in xyxy format

    Returns:
        (N,) GIoU values
    """
    x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1 + area2 - inter

    iou = inter / union.clamp(min=1e-6)

    # Enclosing box
    ex1 = torch.min(boxes1[:, 0], boxes2[:, 0])
    ey1 = torch.min(boxes1[:, 1], boxes2[:, 1])
    ex2 = torch.max(boxes1[:, 2], boxes2[:, 2])
    ey2 = torch.max(boxes1[:, 3], boxes2[:, 3])
    enclose_area = (ex2 - ex1) * (ey2 - ey1)

    giou = iou - (enclose_area - union) / enclose_area.clamp(min=1e-6)
    return giou


def giou_loss(pred_boxes, target_boxes):
    """GIoU loss = 1 - GIoU."""
    return 1 - compute_giou(pred_boxes, target_boxes)


def apply_box_deltas(boxes, deltas):
    """Apply (Δcx, Δcy, Δw, Δh) refinements to boxes in xywh format.

    Args:
        boxes: (N, 4) in xywh format (cx, cy, w, h)
        deltas: (N, 4) refinements

    Returns:
        (N, 4) refined boxes in xywh format
    """
    return boxes + deltas


def box_with_context_padding(boxes_xyxy, padding_ratio, img_h, img_w):
    """Expand boxes by a padding ratio and clip to image bounds.

    Args:
        boxes_xyxy: (N, 4) boxes in xyxy format
        padding_ratio: float, e.g. 0.2 for 20% padding
        img_h, img_w: image dimensions

    Returns:
        (N, 4) padded boxes in xyxy format
    """
    w = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
    h = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
    pad_w = w * padding_ratio / 2
    pad_h = h * padding_ratio / 2

    padded = boxes_xyxy.clone()
    padded[:, 0] = (padded[:, 0] - pad_w).clamp(min=0)
    padded[:, 1] = (padded[:, 1] - pad_h).clamp(min=0)
    padded[:, 2] = (padded[:, 2] + pad_w).clamp(max=img_w)
    padded[:, 3] = (padded[:, 3] + pad_h).clamp(max=img_h)
    return padded


def batched_roi_align(features, boxes_list, output_size=7, spatial_scale=1.0):
    """RoI-Align wrapper for batched inputs.

    Args:
        features: (B, C, H, W) feature maps
        boxes_list: list of B tensors, each (N_i, 4) in xyxy format (image coords)
        output_size: int, spatial size of output
        spatial_scale: float, ratio of feature map size to input image size

    Returns:
        (sum(N_i), C, output_size, output_size) pooled features
    """
    # Build roi tensor with batch indices
    rois = []
    for batch_idx, boxes in enumerate(boxes_list):
        if boxes.numel() == 0:
            continue
        batch_indices = torch.full(
            (boxes.shape[0], 1), batch_idx,
            dtype=boxes.dtype, device=boxes.device
        )
        rois.append(torch.cat([batch_indices, boxes], dim=1))

    if not rois:
        return torch.empty(0, features.shape[1], output_size, output_size,
                           device=features.device)

    rois = torch.cat(rois, dim=0)
    return roi_align(features, rois, output_size=output_size,
                     spatial_scale=spatial_scale, aligned=True)


def apply_nms(boxes, scores, iou_threshold=0.45):
    """Apply NMS to boxes.

    Args:
        boxes: (N, 4) in xyxy format
        scores: (N,) confidence scores
        iou_threshold: float

    Returns:
        indices of kept boxes
    """
    return nms(boxes, scores, iou_threshold)
