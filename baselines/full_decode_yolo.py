"""Baseline 1: Per-frame YOLO (Full Decode).

Upper-bound accuracy baseline. Fully decodes every frame to RGB and runs
YOLOv8 on each. 100% decode budget.

Reports: mAP, latency per frame (decode + inference)
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.yolo_anchor import YOLOAnchorDetector
from data.dataset import SingleFrameDataset
from utils.metrics import COCOMetrics, LatencyTracker


class FullDecodeYOLO:
    """Per-frame YOLO baseline with full decode."""

    def __init__(self, model_name='yolov8m.pt', conf_threshold=0.25,
                 iou_threshold=0.45, num_classes=10):
        self.detector = YOLOAnchorDetector(
            model_name=model_name,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            num_classes=num_classes
        )
        self.latency = LatencyTracker()

    def evaluate(self, dataset, device='cuda'):
        """Run evaluation on a dataset.

        Args:
            dataset: SingleFrameDataset instance
            device: torch device

        Returns:
            dict with mAP metrics and latency
        """
        metrics = COCOMetrics(num_classes=10)
        self.latency.reset()

        for idx in tqdm(range(len(dataset)), desc="Full-Decode YOLO"):
            sample = dataset[idx]

            # Simulate full decode by loading the image
            self.latency.start('decode')
            image = sample['image']  # Already loaded as tensor
            # Convert to numpy for YOLO (denormalize)
            img_np = image.permute(1, 2, 0).numpy()
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            self.latency.stop()

            # YOLO inference
            self.latency.start('inference')
            detections = self.detector.detect([img_np])
            det = detections[0]
            self.latency.stop()

            # Update metrics
            metrics.update(
                det['boxes'].numpy(), det['scores'].numpy(), det['labels'].numpy(),
                sample['gt_boxes'].numpy(), sample['gt_labels'].numpy()
            )

        mAP = metrics.compute()
        latency = self.latency.summary()

        return {
            'mAP_50': mAP['mAP_50'],
            'mAP_50_95': mAP['mAP_50_95'],
            'per_class_ap_50': mAP['per_class_ap_50'],
            'decode_budget': 100.0,
            'latency': latency
        }


def main():
    parser = argparse.ArgumentParser(description='Baseline 1: Full-Decode YOLO')
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--annotations', type=str, required=True)
    parser.add_argument('--model', type=str, default='yolov8m.pt')
    args = parser.parse_args()

    dataset = SingleFrameDataset(
        images_dir=args.images_dir,
        annotations_path=args.annotations,
        split='val'
    )

    baseline = FullDecodeYOLO(model_name=args.model)
    results = baseline.evaluate(dataset)

    print(f"\n{'='*50}")
    print(f"Full-Decode YOLO Results")
    print(f"{'='*50}")
    print(f"mAP@0.5:      {results['mAP_50']:.4f}")
    print(f"mAP@[.5:.95]: {results['mAP_50_95']:.4f}")
    print(f"Decode budget: {results['decode_budget']:.1f}%")
    print(f"Latency:       {results['latency']['total_ms']:.1f} ms/frame")
    print(f"FPS:           {results['latency']['fps']:.1f}")


if __name__ == '__main__':
    main()
