"""Check YOLO mAP on ONLY the annotated frame vs all 16 propagated frames.

The annotation is at GOP offset 12 (frame 300). Evaluating on all 16 frames
with MV-propagated GT dilutes accuracy. Let's measure the difference.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
from data.dataset import BDD100KCoPEDataset
from models.yolo_anchor import YOLOAnchor
from utils.metrics import COCOMetrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_dir = Path("D:/cope-delta-det2/data/bdd100k")
dataset = BDD100KCoPEDataset(root_dir=str(base_dir), split='train', gop_length=16, annotated_only=True)
yolo = YOLOAnchor(model_size='yolov8m.pt', device=device)

NUM_GOPS = 50

# Test 1: Only annotated frame (offset 12)
print("=== Test 1: YOLO on annotated frame ONLY (offset 12) ===")
metrics_ann = COCOMetrics(num_classes=10)
for idx in range(min(len(dataset), NUM_GOPS)):
    sample = dataset[idx]
    targets = sample['targets']

    # The annotated frame is at offset 12
    ann_offset = 12
    if ann_offset >= len(targets):
        continue

    # Get the P-frame RGB for the annotated frame
    pframe_rgbs = sample['pframe_rgbs']  # [N-1, 3, H, W] (P-frames only, offset by 1)
    pframe_idx = ann_offset - 1  # P-frame index (0-based, first P-frame is 0)

    if pframe_idx < 0 or pframe_idx >= len(pframe_rgbs):
        continue

    rgb = pframe_rgbs[pframe_idx].unsqueeze(0).to(device)

    with torch.no_grad():
        anchors = yolo.get_anchor_boxes(rgb)

    tgt = targets[ann_offset]
    gt_boxes = tgt['boxes'].numpy()
    gt_labels = tgt['labels'].numpy()

    if gt_boxes.shape[0] > 0:
        gt_boxes_xyxy = gt_boxes.copy()
        gt_boxes_xyxy[:, 2] += gt_boxes_xyxy[:, 0]
        gt_boxes_xyxy[:, 3] += gt_boxes_xyxy[:, 1]
    else:
        gt_boxes_xyxy = np.zeros((0, 4))

    if anchors[0].shape[0] > 0:
        pred_boxes = anchors[0][:, :4].cpu().numpy()
        pred_conf = anchors[0][:, 4].cpu().numpy()
        pred_cls = anchors[0][:, 5].long().cpu().numpy()
    else:
        pred_boxes = np.zeros((0, 4))
        pred_conf = np.zeros(0)
        pred_cls = np.zeros(0, dtype=np.int64)

    if gt_boxes_xyxy.shape[0] == 0 and pred_boxes.shape[0] == 0:
        continue

    metrics_ann.update(pred_boxes, pred_conf, pred_cls, gt_boxes_xyxy, gt_labels)

mAP_ann = metrics_ann.compute()
print(f"mAP@50 (annotated frame only): {mAP_ann['mAP_50']*100:.2f}%")
print(f"mAP@[.5:.95]:                  {mAP_ann['mAP_50_95']*100:.2f}%")

# Test 2: All 16 frames with propagated GT (current approach)
print("\n=== Test 2: YOLO Full on all 16 frames (propagated GT) ===")
metrics_all = COCOMetrics(num_classes=10)
for idx in range(min(len(dataset), NUM_GOPS)):
    sample = dataset[idx]
    targets = sample['targets']

    iframe_rgb = sample['iframe_rgb'].unsqueeze(0).to(device)
    pframe_rgbs = sample['pframe_rgbs']

    # I-frame
    with torch.no_grad():
        anchors = yolo.get_anchor_boxes(iframe_rgb)

    for t in range(len(targets)):
        if t == 0:
            # I-frame detections
            det = anchors
        else:
            # P-frame detections
            p_rgb = pframe_rgbs[t-1].unsqueeze(0).to(device)
            with torch.no_grad():
                det = yolo.get_anchor_boxes(p_rgb)

        tgt = targets[t]
        gt_boxes = tgt['boxes'].numpy()
        gt_labels = tgt['labels'].numpy()

        if gt_boxes.shape[0] > 0:
            gt_xyxy = gt_boxes.copy()
            gt_xyxy[:, 2] += gt_xyxy[:, 0]
            gt_xyxy[:, 3] += gt_xyxy[:, 1]
        else:
            gt_xyxy = np.zeros((0, 4))

        if det[0].shape[0] > 0:
            pred_boxes = det[0][:, :4].cpu().numpy()
            pred_conf = det[0][:, 4].cpu().numpy()
            pred_cls = det[0][:, 5].long().cpu().numpy()
        else:
            pred_boxes = np.zeros((0, 4))
            pred_conf = np.zeros(0)
            pred_cls = np.zeros(0, dtype=np.int64)

        if gt_xyxy.shape[0] == 0 and pred_boxes.shape[0] == 0:
            continue

        metrics_all.update(pred_boxes, pred_conf, pred_cls, gt_xyxy, gt_labels)

mAP_all = metrics_all.compute()
print(f"mAP@50 (all 16 frames):  {mAP_all['mAP_50']*100:.2f}%")
print(f"mAP@[.5:.95]:            {mAP_all['mAP_50_95']*100:.2f}%")

# Test 3: Check what the annotated_offset actually is
print("\n=== Test 3: Annotation offset verification ===")
for idx in range(min(3, len(dataset))):
    gop = dataset.gops[idx]
    print(f"GOP {idx}: video={gop['video_name']}, start={gop['start_frame']}, "
          f"annotated_offset={gop.get('annotated_offset', 'N/A')}, "
          f"has_annotation={gop.get('has_annotation', False)}")
    targets = dataset[idx]['targets']
    for t in range(len(targets)):
        n_boxes = len(targets[t]['boxes'])
        if n_boxes > 0:
            print(f"  Frame {t}: {n_boxes} GT boxes")
