"""Debug script to investigate why YOLO Full baseline achieves only 17% mAP@50.

Checks:
1. COCO->BDD100K class mapping correctness
2. GT annotation format and distribution
3. YOLO detection quality on individual frames
4. Confidence threshold effects
5. Box format consistency (xyxy vs xywh)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
from collections import Counter

from data.dataset import BDD100KCoPEDataset
from models.yolo_anchor import YOLOAnchor
from utils.metrics import COCOMetrics
from utils.box_utils import xyxy_to_xywh, xywh_to_xyxy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# BDD100K class names (index -> name)
BDD_CLASSES = {
    0: 'pedestrian', 1: 'rider', 2: 'car', 3: 'truck',
    4: 'bus', 5: 'train', 6: 'motorcycle', 7: 'bicycle',
    8: 'traffic light', 9: 'traffic sign'
}

# COCO class names for relevant classes
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
    5: 'bus', 6: 'train', 7: 'truck', 9: 'traffic light', 11: 'stop sign'
}

# Current mapping in yolo_anchor.py
COCO_TO_BDD = {0: 0, 1: 7, 2: 2, 3: 6, 5: 4, 6: 5, 7: 3, 9: 8, 11: 9}

print("\n=== 1. CLASS MAPPING ANALYSIS ===")
print(f"{'COCO ID':>8} {'COCO Name':<15} -> {'BDD ID':>6} {'BDD Name':<15}")
print("-" * 55)
for coco_id, bdd_id in sorted(COCO_TO_BDD.items()):
    coco_name = COCO_CLASSES.get(coco_id, '???')
    bdd_name = BDD_CLASSES.get(bdd_id, '???')
    print(f"{coco_id:>8} {coco_name:<15} -> {bdd_id:>6} {bdd_name:<15}")

print("\nPotential issues:")
print("  - COCO 'person' (0) -> BDD 'pedestrian' (0): OK but BDD has 'rider' (1) too")
print("  - COCO 'bicycle' (1) -> BDD 'bicycle' (7): OK")
print("  - COCO 'stop sign' (11) -> BDD 'traffic sign' (9): BDD traffic sign != stop sign")
print("  - BDD 'rider' (1) has NO COCO mapping! Riders detected as 'person' go to class 0")

# Load dataset
print("\n=== 2. GT ANNOTATION ANALYSIS ===")
base_dir = Path("D:/cope-delta-det2/data/bdd100k")
dataset = BDD100KCoPEDataset(root_dir=str(base_dir), split='train', gop_length=16, annotated_only=True)
print(f"Dataset size: {len(dataset)} annotated GOPs")

# Analyze GT class distribution
gt_class_counts = Counter()
gt_box_sizes = []
gt_frames_with_boxes = 0
gt_total_frames = 0
gt_boxes_per_frame = []

for idx in range(min(len(dataset), 100)):
    sample = dataset[idx]
    targets = sample['targets']
    for t, tgt in enumerate(targets):
        gt_total_frames += 1
        boxes = tgt['boxes']
        labels = tgt['labels']
        if len(boxes) > 0:
            gt_frames_with_boxes += 1
            gt_boxes_per_frame.append(len(boxes))
            for lbl in labels:
                gt_class_counts[lbl.item()] += 1
            # Box sizes (in xywh format from dataset)
            for box in boxes:
                w, h = box[2].item(), box[3].item()
                gt_box_sizes.append((w, h))

print(f"Frames analyzed: {gt_total_frames}")
print(f"Frames with GT boxes: {gt_frames_with_boxes} ({gt_frames_with_boxes/gt_total_frames*100:.1f}%)")
if gt_boxes_per_frame:
    print(f"Avg boxes per annotated frame: {np.mean(gt_boxes_per_frame):.1f}")

print(f"\nGT class distribution:")
for cls_id in sorted(gt_class_counts.keys()):
    name = BDD_CLASSES.get(cls_id, f'unknown-{cls_id}')
    print(f"  {cls_id}: {name:<15} {gt_class_counts[cls_id]:>5} boxes")

print(f"\nGT box sizes (w x h pixels):")
if gt_box_sizes:
    ws = [s[0] for s in gt_box_sizes]
    hs = [s[1] for s in gt_box_sizes]
    print(f"  Width:  min={min(ws):.0f}, median={np.median(ws):.0f}, max={max(ws):.0f}")
    print(f"  Height: min={min(hs):.0f}, median={np.median(hs):.0f}, max={max(hs):.0f}")
    small = sum(1 for w, h in gt_box_sizes if w < 32 or h < 32)
    print(f"  Small boxes (<32px): {small}/{len(gt_box_sizes)} ({small/len(gt_box_sizes)*100:.1f}%)")

# Now run YOLO on I-frames and compare
print("\n=== 3. YOLO DETECTION ANALYSIS ===")
yolo = YOLOAnchor(model_size='yolov8m.pt', device=device)

det_class_counts = Counter()
det_conf_values = []
det_boxes_per_frame = []
all_coco_classes = Counter()

num_test = 50
for idx in range(min(len(dataset), num_test)):
    sample = dataset[idx]
    iframe_rgb = sample['iframe_rgb'].unsqueeze(0).to(device)

    # Also run raw YOLO to see COCO class distribution before filtering
    import torch.nn.functional as Fpad
    _, _, h, w = iframe_rgb.shape
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    padded = Fpad.pad(iframe_rgb, (0, pad_w, 0, pad_h)) if (pad_h or pad_w) else iframe_rgb
    raw_results = yolo._yolo_wrap[0](padded, verbose=False)
    for r in raw_results:
        if len(r.boxes) > 0:
            for cls_id in r.boxes.cls.long().cpu().numpy():
                all_coco_classes[cls_id] += 1

    anchors = yolo.get_anchor_boxes(iframe_rgb)
    for a in anchors:
        if a.shape[0] > 0:
            det_boxes_per_frame.append(a.shape[0])
            confs = a[:, 4].cpu().numpy()
            classes = a[:, 5].long().cpu().numpy()
            det_conf_values.extend(confs.tolist())
            for c in classes:
                det_class_counts[c] += 1

print(f"Analyzed {num_test} I-frames:")
if det_boxes_per_frame:
    print(f"  Avg detections per frame: {np.mean(det_boxes_per_frame):.1f}")
    print(f"  Frames with detections: {len(det_boxes_per_frame)}/{num_test}")

print(f"\nYOLO detection class distribution (BDD-mapped):")
for cls_id in sorted(det_class_counts.keys()):
    name = BDD_CLASSES.get(cls_id, f'unknown-{cls_id}')
    print(f"  {cls_id}: {name:<15} {det_class_counts[cls_id]:>5} detections")

print(f"\nRaw COCO class distribution (before BDD filtering):")
for cls_id in sorted(all_coco_classes.keys()):
    name = COCO_CLASSES.get(cls_id, f'coco-{cls_id}')
    print(f"  {cls_id}: {name:<15} {all_coco_classes[cls_id]:>5}")

print(f"\nConfidence distribution:")
if det_conf_values:
    confs = np.array(det_conf_values)
    print(f"  Min: {confs.min():.3f}, Mean: {confs.mean():.3f}, Max: {confs.max():.3f}")
    for thresh in [0.1, 0.25, 0.5, 0.7]:
        pct = (confs >= thresh).mean() * 100
        print(f"  Above {thresh}: {pct:.1f}%")

# Per-frame mAP diagnostic
print("\n=== 4. PER-FRAME mAP DIAGNOSTIC (10 frames) ===")
for idx in range(min(len(dataset), 10)):
    sample = dataset[idx]
    iframe_rgb = sample['iframe_rgb'].unsqueeze(0).to(device)
    targets = sample['targets']

    # Only check I-frame (t=0)
    tgt = targets[0]
    gt_boxes = tgt['boxes'].numpy()
    gt_labels = tgt['labels'].numpy()

    # Convert xywh -> xyxy for GT
    if gt_boxes.shape[0] > 0:
        gt_xyxy = gt_boxes.copy()
        gt_xyxy[:, 2] = gt_xyxy[:, 0] + gt_xyxy[:, 2]
        gt_xyxy[:, 3] = gt_xyxy[:, 1] + gt_xyxy[:, 3]
    else:
        gt_xyxy = np.zeros((0, 4))

    anchors = yolo.get_anchor_boxes(iframe_rgb)
    if anchors[0].shape[0] > 0:
        pred_boxes = anchors[0][:, :4].cpu().numpy()
        pred_conf = anchors[0][:, 4].cpu().numpy()
        pred_cls = anchors[0][:, 5].long().cpu().numpy()
    else:
        pred_boxes = np.zeros((0, 4))
        pred_conf = np.zeros(0)
        pred_cls = np.zeros(0, dtype=np.int64)

    # Compute IoU between each pred and GT box
    n_match = 0
    if gt_xyxy.shape[0] > 0 and pred_boxes.shape[0] > 0:
        from torchvision.ops import box_iou
        iou = box_iou(
            torch.from_numpy(pred_boxes).float(),
            torch.from_numpy(gt_xyxy).float()
        ).numpy()
        max_iou_per_gt = iou.max(axis=0) if iou.shape[1] > 0 else np.array([])
        n_match = (max_iou_per_gt >= 0.5).sum()

        print(f"\n  GOP {idx}: GT={gt_xyxy.shape[0]} boxes, Pred={pred_boxes.shape[0]} dets")
        print(f"    GT classes: {gt_labels.tolist()}")
        print(f"    Pred classes: {pred_cls.tolist()}")
        print(f"    Max IoU per GT: {max_iou_per_gt.tolist()[:5]}")
        print(f"    Matched (IoU>=0.5): {n_match}/{gt_xyxy.shape[0]}")

        # Check class alignment for matched pairs
        for gt_i in range(min(gt_xyxy.shape[0], 5)):
            best_pred = iou[:, gt_i].argmax()
            best_iou = iou[best_pred, gt_i]
            if best_iou >= 0.5:
                gt_c = gt_labels[gt_i]
                pred_c = pred_cls[best_pred]
                match = "OK" if gt_c == pred_c else f"MISMATCH (gt={gt_c}, pred={pred_c})"
                print(f"    GT box {gt_i} (class {gt_c}={BDD_CLASSES.get(gt_c, '?')}) <-> "
                      f"Pred {best_pred} (class {pred_c}={BDD_CLASSES.get(pred_c, '?')}) IoU={best_iou:.2f} {match}")
    else:
        print(f"\n  GOP {idx}: GT={gt_xyxy.shape[0]} boxes, Pred={pred_boxes.shape[0]} dets (no overlap)")

print("\n=== 5. QUICK mAP SANITY CHECK (I-frames only, 50 GOPs) ===")
metrics = COCOMetrics(num_classes=10)
for idx in range(min(len(dataset), 50)):
    sample = dataset[idx]
    iframe_rgb = sample['iframe_rgb'].unsqueeze(0).to(device)
    tgt = sample['targets'][0]

    gt_boxes = tgt['boxes'].numpy()
    gt_labels = tgt['labels'].numpy()

    if gt_boxes.shape[0] > 0:
        gt_xyxy = gt_boxes.copy()
        gt_xyxy[:, 2] = gt_xyxy[:, 0] + gt_xyxy[:, 2]
        gt_xyxy[:, 3] = gt_xyxy[:, 1] + gt_xyxy[:, 3]
    else:
        gt_xyxy = np.zeros((0, 4))

    with torch.no_grad():
        anchors = yolo.get_anchor_boxes(iframe_rgb)

    if anchors[0].shape[0] > 0:
        pred_boxes = anchors[0][:, :4].cpu().numpy()
        pred_conf = anchors[0][:, 4].cpu().numpy()
        pred_cls = anchors[0][:, 5].long().cpu().numpy()
    else:
        pred_boxes = np.zeros((0, 4))
        pred_conf = np.zeros(0)
        pred_cls = np.zeros(0, dtype=np.int64)

    if gt_xyxy.shape[0] == 0 and pred_boxes.shape[0] == 0:
        continue

    metrics.update(pred_boxes, pred_conf, pred_cls, gt_xyxy, gt_labels)

mAP = metrics.compute()
print(f"I-frame only mAP@50:     {mAP['mAP_50']*100:.2f}%")
print(f"I-frame only mAP@[.5:.95]: {mAP['mAP_50_95']*100:.2f}%")

if 'per_class_ap_50' in mAP:
    print(f"\nPer-class AP@50:")
    for cls_id, ap in enumerate(mAP['per_class_ap_50']):
        name = BDD_CLASSES.get(cls_id, f'class-{cls_id}')
        print(f"  {cls_id}: {name:<15} AP={ap*100:.2f}%")
