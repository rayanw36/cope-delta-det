"""Evaluation metrics: COCO-style mAP, latency measurement, decode budget tracking."""

import time
import numpy as np
import torch
from collections import defaultdict


class COCOMetrics:
    """COCO-style mAP evaluation for object detection."""

    def __init__(self, num_classes=10, iou_thresholds=None):
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05)
        self.reset()

    def reset(self):
        self.predictions = []  # list of (boxes, scores, labels) per image
        self.ground_truths = []  # list of (boxes, labels) per image

    def update(self, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels):
        """Add predictions and ground truths for one image.

        All inputs are numpy arrays or tensors on CPU.
        """
        if isinstance(pred_boxes, torch.Tensor):
            pred_boxes = pred_boxes.cpu().numpy()
            pred_scores = pred_scores.cpu().numpy()
            pred_labels = pred_labels.cpu().numpy()
            gt_boxes = gt_boxes.cpu().numpy()
            gt_labels = gt_labels.cpu().numpy()

        self.predictions.append((pred_boxes, pred_scores, pred_labels))
        self.ground_truths.append((gt_boxes, gt_labels))

    def _compute_ap(self, recall, precision):
        """Compute AP using 101-point interpolation (COCO style)."""
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))

        # Make precision monotonically decreasing
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # 101-point interpolation
        recall_points = np.linspace(0, 1, 101)
        precision_interp = np.zeros_like(recall_points)
        for i, r in enumerate(recall_points):
            idx = np.where(mrec >= r)[0]
            if idx.size > 0:
                precision_interp[i] = mpre[idx[0]]

        return precision_interp.mean()

    def _compute_iou_matrix(self, boxes1, boxes2):
        """Compute IoU between two sets of boxes."""
        if boxes1.size == 0 or boxes2.size == 0:
            return np.zeros((len(boxes1), len(boxes2)))

        x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
        y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
        x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
        y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

        inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1[:, None] + area2[None, :] - inter

        return inter / np.maximum(union, 1e-6)

    def compute(self):
        """Compute mAP@0.5 and mAP@[.5:.95].

        Returns:
            dict with 'mAP_50', 'mAP_50_95', and per-class APs.
        """
        all_aps = defaultdict(list)  # iou_thresh -> list of per-class APs

        for cls_id in range(self.num_classes):
            # Gather all predictions and GTs for this class
            all_scores = []
            all_tp = {t: [] for t in self.iou_thresholds}
            total_gt = 0

            for img_idx in range(len(self.predictions)):
                pred_boxes, pred_scores, pred_labels = self.predictions[img_idx]
                gt_boxes, gt_labels = self.ground_truths[img_idx]

                # Filter to this class
                pred_mask = pred_labels == cls_id
                gt_mask = gt_labels == cls_id

                p_boxes = pred_boxes[pred_mask]
                p_scores = pred_scores[pred_mask]
                g_boxes = gt_boxes[gt_mask]

                total_gt += len(g_boxes)

                if len(p_boxes) == 0:
                    continue

                # Sort by score descending
                order = np.argsort(-p_scores)
                p_boxes = p_boxes[order]
                p_scores = p_scores[order]
                all_scores.extend(p_scores.tolist())

                if len(g_boxes) == 0:
                    for t in self.iou_thresholds:
                        all_tp[t].extend([0] * len(p_boxes))
                    continue

                iou_matrix = self._compute_iou_matrix(p_boxes, g_boxes)

                for t in self.iou_thresholds:
                    matched_gt = set()
                    for pred_idx in range(len(p_boxes)):
                        best_iou = 0
                        best_gt = -1
                        for gt_idx in range(len(g_boxes)):
                            if gt_idx in matched_gt:
                                continue
                            if iou_matrix[pred_idx, gt_idx] > best_iou:
                                best_iou = iou_matrix[pred_idx, gt_idx]
                                best_gt = gt_idx
                        if best_iou >= t and best_gt >= 0:
                            all_tp[t].append(1)
                            matched_gt.add(best_gt)
                        else:
                            all_tp[t].append(0)

            if total_gt == 0:
                continue

            # Sort all predictions by score
            if not all_scores:
                for t in self.iou_thresholds:
                    all_aps[t].append(0.0)
                continue

            sorted_indices = np.argsort(-np.array(all_scores))

            for t in self.iou_thresholds:
                tp_arr = np.array(all_tp[t])[sorted_indices]
                cum_tp = np.cumsum(tp_arr)
                cum_fp = np.cumsum(1 - tp_arr)
                recall = cum_tp / total_gt
                precision = cum_tp / (cum_tp + cum_fp)
                ap = self._compute_ap(recall, precision)
                all_aps[t].append(ap)

        results = {}
        # mAP@0.5
        if all_aps[0.5]:
            results['mAP_50'] = np.mean(all_aps[0.5])
        else:
            results['mAP_50'] = 0.0

        # mAP@[.5:.95]
        mean_aps = []
        for t in self.iou_thresholds:
            if all_aps[t]:
                mean_aps.append(np.mean(all_aps[t]))
        results['mAP_50_95'] = np.mean(mean_aps) if mean_aps else 0.0

        # Per-class AP at 0.5
        results['per_class_ap_50'] = all_aps.get(0.5, [])

        return results


class LatencyTracker:
    """Tracks per-component latency for the pipeline."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.timings = defaultdict(list)

    def start(self, name):
        """Start timing a component."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start_time = time.perf_counter()
        self._current_name = name

    def stop(self):
        """Stop timing and record."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - self._start_time) * 1000  # ms
        self.timings[self._current_name].append(elapsed)
        return elapsed

    def summary(self):
        """Return mean and std for each component."""
        results = {}
        total = 0
        for name, times in self.timings.items():
            arr = np.array(times)
            results[name] = {
                'mean_ms': arr.mean(),
                'std_ms': arr.std(),
                'count': len(times)
            }
            total += arr.mean()
        results['total_ms'] = total
        results['fps'] = 1000.0 / total if total > 0 else 0
        return results


class DecodeBudgetTracker:
    """Tracks the percentage of frames that were fully decoded."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_frames = 0
        self.decoded_frames = 0

    def record_iframe(self):
        self.total_frames += 1
        self.decoded_frames += 1

    def record_pframe(self, was_refreshed=False):
        self.total_frames += 1
        if was_refreshed:
            self.decoded_frames += 1

    @property
    def decode_ratio(self):
        if self.total_frames == 0:
            return 0.0
        return self.decoded_frames / self.total_frames

    def summary(self):
        return {
            'total_frames': self.total_frames,
            'decoded_frames': self.decoded_frames,
            'decode_ratio': self.decode_ratio,
            'decode_percent': self.decode_ratio * 100
        }
