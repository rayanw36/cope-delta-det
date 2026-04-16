"""Baseline 2: Mean-MV Propagation.

Simplest baseline: average motion vectors within each anchor box and shift
the box frame-by-frame. No learning involved.

Reports: mAP, latency, decode %
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.yolo_anchor import YOLOAnchorDetector
from data.dataset import GOPDataset, gop_collate_fn
from utils.metrics import COCOMetrics, LatencyTracker, DecodeBudgetTracker
from utils.box_utils import xyxy_to_xywh, xywh_to_xyxy


class MeanMVPropagation:
    """Mean motion vector box propagation baseline."""

    def __init__(self, model_name='yolov8m.pt', num_classes=10):
        self.detector = YOLOAnchorDetector(
            model_name=model_name, num_classes=num_classes
        )
        self.latency = LatencyTracker()
        self.decode_tracker = DecodeBudgetTracker()

    def propagate_boxes(self, boxes_xyxy, mv_tensor, img_h=720, img_w=1280):
        """Propagate boxes using mean motion vectors within each box.

        Args:
            boxes_xyxy: (N, 4) boxes in xyxy format
            mv_tensor: (2, H_feat, W_feat) motion vector tensor
            img_h, img_w: image dimensions

        Returns:
            (N, 4) propagated boxes in xyxy format
        """
        if boxes_xyxy.numel() == 0:
            return boxes_xyxy.clone()

        h_feat, w_feat = mv_tensor.shape[1], mv_tensor.shape[2]
        propagated = boxes_xyxy.clone().float()

        for i in range(boxes_xyxy.shape[0]):
            x1, y1, x2, y2 = boxes_xyxy[i].tolist()

            # Map box to feature grid
            fx1 = max(0, int(x1 / img_w * w_feat))
            fy1 = max(0, int(y1 / img_h * h_feat))
            fx2 = min(w_feat, int(np.ceil(x2 / img_w * w_feat)))
            fy2 = min(h_feat, int(np.ceil(y2 / img_h * h_feat)))

            if fx2 <= fx1 or fy2 <= fy1:
                continue

            # Average MVs within the box region
            mv_region = mv_tensor[:, fy1:fy2, fx1:fx2]
            mean_mvx = mv_region[0].mean().item()
            mean_mvy = mv_region[1].mean().item()

            # Shift box by mean MV (MVs are in quarter-pixel units typically)
            # Scale MV from feature grid to image coordinates
            scale_x = img_w / w_feat
            scale_y = img_h / h_feat

            propagated[i, 0] += mean_mvx * scale_x
            propagated[i, 1] += mean_mvy * scale_y
            propagated[i, 2] += mean_mvx * scale_x
            propagated[i, 3] += mean_mvy * scale_y

        # Clip to image bounds
        propagated[:, 0].clamp_(min=0)
        propagated[:, 1].clamp_(min=0)
        propagated[:, 2].clamp_(max=img_w)
        propagated[:, 3].clamp_(max=img_h)

        return propagated

    def evaluate(self, dataset):
        """Run evaluation on GOP dataset.

        Args:
            dataset: GOPDataset instance

        Returns:
            dict with mAP metrics, latency, and decode budget
        """
        metrics = COCOMetrics(num_classes=10)
        self.latency.reset()
        self.decode_tracker.reset()

        for idx in tqdm(range(len(dataset)), desc="Mean-MV Propagation"):
            sample = dataset[idx]
            gt_boxes_gop = sample['gt_boxes']
            gt_labels_gop = sample['gt_labels']

            # I-frame: use GT boxes as anchor (simulating YOLO detection)
            self.latency.start('i_frame_detect')
            anchor_boxes = gt_boxes_gop[0].float()
            anchor_labels = gt_labels_gop[0]
            anchor_scores = torch.ones(anchor_boxes.shape[0])
            self.latency.stop()
            self.decode_tracker.record_iframe()

            # I-frame evaluation
            metrics.update(
                anchor_boxes.numpy(), anchor_scores.numpy(), anchor_labels.numpy(),
                gt_boxes_gop[0].numpy(), gt_labels_gop[0].numpy()
            )

            # P-frames: propagate using mean MV
            current_boxes = anchor_boxes.clone()
            for t, feat in enumerate(sample['p_frame_features']):
                self.latency.start('mv_propagate')
                mv_tensor = feat['mv_tensor']  # (2, H, W)
                current_boxes = self.propagate_boxes(current_boxes, mv_tensor)
                self.latency.stop()
                self.decode_tracker.record_pframe(was_refreshed=False)

                gt_idx = t + 1
                if gt_idx < len(gt_boxes_gop):
                    metrics.update(
                        current_boxes.numpy(), anchor_scores.numpy(),
                        anchor_labels.numpy(),
                        gt_boxes_gop[gt_idx].numpy(),
                        gt_labels_gop[gt_idx].numpy()
                    )

        mAP = metrics.compute()
        latency = self.latency.summary()
        decode = self.decode_tracker.summary()

        return {
            'mAP_50': mAP['mAP_50'],
            'mAP_50_95': mAP['mAP_50_95'],
            'decode_budget': decode['decode_percent'],
            'latency': latency,
            'decode_stats': decode
        }


def main():
    parser = argparse.ArgumentParser(description='Baseline 2: Mean-MV Propagation')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    feat_subdir = f"qp{config['hevc']['default_qp']}_gop{config['hevc']['default_gop']}"
    dataset = GOPDataset(
        images_dir=config['paths']['bdd100k_root'] + '/images/val',
        features_dir=config['paths']['extracted_features'] + f'/{feat_subdir}',
        annotations_path=config['paths']['bdd100k_root'] + '/annotations_val_coco.json',
        gop_index_path=config['paths']['bdd100k_root'] + '/gop_index_val.json',
        gop_length=config['hevc']['default_gop'],
        split='val'
    )

    baseline = MeanMVPropagation()
    results = baseline.evaluate(dataset)

    print(f"\n{'='*50}")
    print(f"Mean-MV Propagation Results")
    print(f"{'='*50}")
    print(f"mAP@0.5:      {results['mAP_50']:.4f}")
    print(f"mAP@[.5:.95]: {results['mAP_50_95']:.4f}")
    print(f"Decode budget: {results['decode_budget']:.1f}%")


if __name__ == '__main__':
    main()
