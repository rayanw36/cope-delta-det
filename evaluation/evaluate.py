"""Main evaluation script for CoPE-Δ-Det2.

Runs full evaluation: mAP, latency, decode budget on BDD100K validation set.
Adapted to use the new CoPE-Δ-Det2 forwarding pass.
"""

import argparse
import sys
import json
import time
from pathlib import Path

import torch
import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import BDD100KCoPEDataset
from models.cope_delta_det import CoPEDeltaDet
from models.refresh_policy import SaliencyRefreshPolicy
from utils.metrics import COCOMetrics, LatencyTracker, DecodeBudgetTracker
from utils.box_utils import xyxy_to_xywh, xywh_to_xyxy

def evaluate_cope_delta_det(model, dataset, device, measure_latency=True, policy_w1=1.0, policy_w2=1.0, policy_thresh=0.5, num_classes=10, max_eval=None):
    """Evaluate CoPE-Δ-Det on a dataset using the unified forward pass.

    Set policy_thresh >= 999 to disable the refresh policy entirely (pure CoPE,
    delta encoder runs on every P-frame, decode budget ≈ 6.25%).
    """
    model.eval()
    metrics = COCOMetrics(num_classes=num_classes)
    latency = LatencyTracker() if measure_latency else None
    decode_tracker = DecodeBudgetTracker()

    policy_disabled = policy_thresh >= 999
    refresh_policy = SaliencyRefreshPolicy(init_threshold=policy_thresh).to(device)
    refresh_policy.w1.data = torch.tensor(policy_w1, device=device)
    refresh_policy.w2.data = torch.tensor(policy_w2, device=device)

    n_total = len(dataset) if max_eval is None else min(max_eval, len(dataset))
    with torch.no_grad():
        for idx in tqdm(range(n_total), desc="Evaluating CoPE-Δ-Det"):
            sample = dataset[idx]
            
            # Map sample elements to device
            iframe_rgb = sample['iframe_rgb'].unsqueeze(0).to(device) # Add batch dim
            pframe_rgbs = sample['pframe_rgbs'] # Full decoded RGBs for fallback
            
            mvs = sample['pframe_mvs'].to(device) # [N, 2, H, W]
            res = sample['pframe_res'].to(device) # [N, 1, H, W]
            depths = sample['pframe_depths'].to(device) # [N, 1, H, W]
            modes = sample['pframe_modes'].to(device) # [N, 1, H, W]
            
            targets = sample['targets'] # List of dicts
            num_pframes = mvs.shape[0]
            
            # Frame 0 (I-Frame) Native Decode
            decode_tracker.record_iframe()
            if latency: latency.start('inference_total')
            
            # Manual frame-by-frame forward pass to allow Saliency interruptions
            anchor_results = model.anchor_detector.get_anchor_boxes(iframe_rgb)
            current_boxes = [res[:, :4] if res.shape[0] > 0 else torch.empty((0, 4), device=device) for res in anchor_results]
            current_confs = [res[:, 4:5] if res.shape[0] > 0 else torch.empty((0, 1), device=device) for res in anchor_results]
            current_classes = [res[:, 5:6] if res.shape[0] > 0 else torch.empty((0, 1), device=device) for res in anchor_results]
            
            predictions = [{'boxes': current_boxes, 'confs': current_confs, 'classes': current_classes}]
            
            refresh_policy.reset_states()
            
            for t in range(num_pframes):
                # Construct App Tensor with proper [B, C, H, W] dimensions
                r_feat = res[t].permute(2, 0, 1) # [1, 45, 80]
                d_feat = depths[t].permute(2, 0, 1).float()
                m_feat = modes[t].permute(2, 0, 1).float()
                
                app_t = torch.cat([r_feat, d_feat, m_feat], dim=0).unsqueeze(0) # [1, 3, 45, 80]
                mvs_t = mvs[t].permute(2, 0, 1).unsqueeze(0) # [1, 2, 45, 80]
                
                # 1. Ask Saliency Policy if we need to refresh
                # using the residual (1 channel) and MVs
                res_1c = r_feat.unsqueeze(0) # [1, 1, 45, 80]
                if policy_disabled:
                    do_refresh = False
                else:
                    refresh_flags, _ = refresh_policy(res_1c, mvs_t, current_boxes)
                    do_refresh = refresh_flags.any().item()

                if do_refresh:
                    # REFRESH! Run YOLO on P-frame RGB!
                    decode_tracker.record_iframe() # Counts as full decode
                    fallback_rgb = pframe_rgbs[t].unsqueeze(0).to(device)
                    
                    anchor_results = model.anchor_detector.get_anchor_boxes(fallback_rgb)
                    current_boxes = [r[:, :4] if r.shape[0] > 0 else torch.empty((0, 4), device=device) for r in anchor_results]
                    current_confs = [r[:, 4:5] if r.shape[0] > 0 else torch.empty((0, 1), device=device) for r in anchor_results]
                    current_classes = [r[:, 5:6] if r.shape[0] > 0 else torch.empty((0, 1), device=device) for r in anchor_results]
                    
                    if not policy_disabled:
                        refresh_policy.reset_states()

                else:
                    # STANDARD COPE DECODE!
                    decode_tracker.record_pframe(was_refreshed=False)
                    # We must run Delta Encoder and Temporal Fusion
                    flat_boxes = torch.cat(current_boxes, dim=0) if current_boxes[0].shape[0] > 0 else torch.empty((0, 4), device=device)
                    
                    if flat_boxes.shape[0] > 0:
                        b_ids = torch.zeros(flat_boxes.shape[0], dtype=torch.long, device=device)
                        flat_confs = torch.cat(current_confs, dim=0)

                        # Convert xyxy -> xywh for fusion head
                        boxes_xywh = xyxy_to_xywh(flat_boxes)
                        flat_anchors = torch.cat([boxes_xywh, flat_confs], dim=1)

                        delta_tokens = model.delta_encoder(mvs_t, app_t, flat_boxes, b_ids)
                        box_deltas, conf_updates, _ = model.fusion_head(flat_anchors, delta_tokens)

                        # Apply deltas in xywh space, convert back to xyxy
                        updated_xywh = boxes_xywh + box_deltas
                        current_boxes = [xywh_to_xyxy(updated_xywh)]
                        current_confs = [flat_confs * conf_updates]
                        
                predictions.append({'boxes': current_boxes, 'confs': current_confs, 'classes': current_classes})
                
            if latency: latency.stop()
            
            # Record metrics for each frame in GOP
            for t, pred_t in enumerate(predictions):
                if t >= len(targets):
                    break

                target_t = targets[t]
                gt_boxes = target_t['boxes'].numpy() if len(target_t['boxes']) > 0 else np.zeros((0,4))
                gt_labels = target_t['labels'].numpy() if len(target_t['labels']) > 0 else np.zeros(0, dtype=np.int64)

                # Convert [x1, y1, width, height] from dataset JSON to [x1, y1, x2, y2]
                if gt_boxes.shape[0] > 0:
                    gt_boxes[:, 2] = gt_boxes[:, 0] + gt_boxes[:, 2]
                    gt_boxes[:, 3] = gt_boxes[:, 1] + gt_boxes[:, 3]

                if len(pred_t['boxes']) > 0 and pred_t['boxes'][0].shape[0] > 0:
                    pred_boxes = pred_t['boxes'][0].cpu().numpy()
                    pred_conf = pred_t['confs'][0].cpu().numpy().squeeze()

                    # Ensure confidence array is correct shape
                    if pred_conf.ndim == 0:
                        pred_conf = np.array([pred_conf])

                    # Handle classes — may be [N, num_classes] logits or [N, 1] IDs
                    raw_cls = pred_t['classes'][0].cpu()
                    if raw_cls.ndim == 2 and raw_cls.shape[1] > 1:
                        # Logits from fusion head — take argmax
                        pred_cls = raw_cls.argmax(dim=1).numpy().astype(np.int64)
                    else:
                        pred_cls = raw_cls.numpy().squeeze().astype(np.int64)
                        if pred_cls.ndim == 0:
                            pred_cls = np.array([pred_cls])
                else:
                    pred_boxes = np.zeros((0, 4))
                    pred_conf = np.zeros(0)
                    pred_cls = np.zeros(0, dtype=np.int64)

                # Skip frames where both GT and predictions are empty
                if gt_boxes.shape[0] == 0 and pred_boxes.shape[0] == 0:
                    continue

                metrics.update(pred_boxes, pred_conf, pred_cls, gt_boxes, gt_labels)

    mAP = metrics.compute()
    decode_stats = decode_tracker.summary()

    results = {
        'mAP_50': mAP['mAP_50'],
        'mAP_50_95': mAP['mAP_50_95'],
        'per_class_ap_50': mAP['per_class_ap_50'],
        'decode_budget': decode_stats['decode_percent'],
        'decode_stats': decode_stats,
    }

    if latency:
        results['latency'] = latency.summary()

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate CoPE-Δ-Det2')
    parser.add_argument('--config', type=str, default='configs/eval.yaml')
    parser.add_argument('--checkpoint', type=str, default='None')
    parser.add_argument('--output', type=str, default='results/eval_results.json')
    parser.add_argument('--dataset', type=str, default='bdd100k',
                        choices=['bdd100k', 'imagenetvid'])
    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--features', type=str, default='features',
                        choices=['features', 'features_pyav'])
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--yolo_weights', type=str, default='yolov8m.pt')
    parser.add_argument('--class_mapping', type=str, default=None,
                        help="'coco_to_bdd', 'coco_to_vid', 'identity', or omit for dataset default")
    parser.add_argument('--annotated_only', action='store_true')
    parser.add_argument('--policy_thresh', type=float, default=0.5,
                        help='Saliency refresh threshold. Set to 999 to disable (pure CoPE, no refresh).')
    parser.add_argument('--max_eval', type=int, default=None,
                        help='Max GOPs to evaluate (default: all)')
    args = parser.parse_args()

    # Dataset-specific defaults
    if args.root is None:
        args.root = ('D:/cope-delta-det2/data/bdd100k' if args.dataset == 'bdd100k'
                     else 'D:/cope-delta-det2/data/imagenetvid')
    if args.num_classes is None:
        args.num_classes = 10 if args.dataset == 'bdd100k' else 30
    if args.class_mapping is None:
        args.class_mapping = 'coco_to_bdd' if args.dataset == 'bdd100k' else 'coco_to_vid'
    annotated_only_flag = args.annotated_only or (args.dataset == 'bdd100k')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on {args.dataset} ({args.split}), num_classes={args.num_classes}")
    print(f"  root={args.root}  yolo_weights={args.yolo_weights}")

    model = CoPEDeltaDet(
        yolo_size=args.yolo_weights,
        embed_dim=256,
        num_classes=args.num_classes,
        device=device
    ).to(device)

    # Propagate class_mapping to YOLO anchor
    # coco_to_vid: COCO 80-class IDs → ImageNet VID 30-class IDs
    COCO_TO_VID = {
        4: 0,   # airplane
        21: 2,  # bear
        1: 3,   # bicycle
        14: 4,  # bird
        5: 5,   # bus
        2: 6,   # car
        7: 6,   # truck → car
        19: 7,  # cow → cattle
        16: 8,  # dog
        15: 9,  # cat → domestic cat
        20: 10, # elephant
        17: 14, # horse
        3: 18,  # motorcycle
        18: 21, # sheep
        6: 25,  # train
        8: 27,  # boat → watercraft
        22: 29, # zebra
    }
    try:
        if hasattr(model.anchor_detector, 'class_mapping'):
            if args.class_mapping in (None, 'identity'):
                model.anchor_detector.class_mapping = None
            elif args.class_mapping == 'coco_to_bdd':
                model.anchor_detector.class_mapping = {0: 0, 1: 7, 2: 2, 3: 6, 5: 4, 6: 5, 7: 3, 9: 8, 11: 9}
            elif args.class_mapping == 'coco_to_vid':
                model.anchor_detector.class_mapping = COCO_TO_VID
                print(f"  class_mapping: coco_to_vid ({len(COCO_TO_VID)} COCO→VID class pairs)")
        print(f"  policy_thresh: {args.policy_thresh}  max_eval: {args.max_eval or 'all'}")
    except Exception as e:
        print(f"  (note) could not set class_mapping on anchor: {e}")
    
    if args.checkpoint != 'None' and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        if 'delta_encoder' in ckpt:
            model.delta_encoder.load_state_dict(ckpt['delta_encoder'])
            model.fusion_head.load_state_dict(ckpt['fusion_head'])
            print(f"Loaded finetuned checkpoint (epoch {ckpt.get('epoch', '?')})")
        elif 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
            print(f"Loaded checkpoint from {args.checkpoint}")
        else:
            model.load_state_dict(ckpt, strict=False)
            print(f"Loaded checkpoint (legacy format)")
    else:
        print("Running with untrained weights for skeleton check.")

    dataset = BDD100KCoPEDataset(
        root_dir=args.root,
        split=args.split,
        gop_length=16,
        annotated_only=annotated_only_flag,
        features_subdir=args.features,
        dataset_type=args.dataset,
    )

    # Evaluate
    results = evaluate_cope_delta_det(model, dataset, device, policy_w1=1.0, policy_w2=1.0,
                                      policy_thresh=args.policy_thresh,
                                      num_classes=args.num_classes,
                                      max_eval=args.max_eval)

    # Print results
    print(f"\n{'='*60}")
    print(f"CoPE-Δ-Det2 Evaluation Results")
    print(f"{'='*60}")
    print(f"mAP@0.5:      {results['mAP_50']:.4f}")
    print(f"mAP@[.5:.95]: {results['mAP_50_95']:.4f}")
    print(f"Decode budget: {results['decode_budget']:.1f}%")

    if results.get('latency'):
        lat = results['latency']
        print(f"\nLatency:")
        for name, stats in lat.items():
            if isinstance(stats, dict):
                print(f"  {name:20s}: {stats['mean_ms']:.2f} ms")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump({
                'features': args.features,
                'dataset': args.dataset,
                'split': args.split,
                'class_mapping': args.class_mapping,
                'checkpoint': args.checkpoint,
                'policy_thresh': args.policy_thresh,
                'max_eval': args.max_eval,
                'yolo_weights': args.yolo_weights,
                'mAP_50': results['mAP_50'],
                'mAP_50_95': results['mAP_50_95'],
                'decode_budget': results['decode_budget'],
            }, f, indent=2)
        print(f"\nResults saved to {out_path}")

if __name__ == '__main__':
    main()
