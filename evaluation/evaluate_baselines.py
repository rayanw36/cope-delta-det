import argparse
import sys
import json
import time
from pathlib import Path

import torch
from torch.utils.data import Subset
import json
import time
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import BDD100KCoPEDataset
from models.cope_delta_det import CoPEDeltaDet
from utils.metrics import COCOMetrics, LatencyTracker, DecodeBudgetTracker
from utils.box_utils import xyxy_to_xywh, xywh_to_xyxy

def evaluate_baseline(model, dataset, device, mode='cope', num_classes=10, max_eval=50):
    """
    Evaluates Video Obj Detection on specific baseline architectures:
    * 'yolo_full' : Runs heavy YOLO object detection on every single frame.
    * 'copy_paste': Runs YOLO on I-frame, and just locks boxes into place on P-frames
                    (zero tracking math, lowest possible cost but terrible accuracy).
    * 'cope'      : Runs our custom CoPE-Delta-Det2 framework.
    """
    model.eval()
    metrics = COCOMetrics(num_classes=num_classes)
    latency = LatencyTracker()
    decode_tracker = DecodeBudgetTracker()

    with torch.no_grad():
        evaluated_count = 0
        
        pbar = tqdm(total=max_eval, desc=f"Evaluating mode: {mode}")
        for idx in range(len(dataset)):
            if evaluated_count >= max_eval:
                break
                
            sample = dataset[idx]
            # Since some videos are not locally downloaded, check if the iframe is fully black/zero
            if sample['iframe_rgb'].sum().item() == 0:
                continue
                
            evaluated_count += 1
            pbar.update(1)
            
            iframe_rgb = sample['iframe_rgb'].unsqueeze(0).to(device)
            pframe_rgbs = sample['pframe_rgbs'] # Tensors [N, 3, H, W]
            
            mvs = sample['pframe_mvs'].to(device)
            res = sample['pframe_res'].to(device)
            depths = sample['pframe_depths'].to(device)
            modes = sample['pframe_modes'].to(device)
            targets = sample['targets']
            num_pframes = mvs.shape[0]
            
            decode_tracker.record_iframe()
            latency.start('inference_total')
            
            # --- Frame 0 (I-FRAME) is identically evaluated for all baselines ---
            anchor_results = model.anchor_detector.get_anchor_boxes(iframe_rgb)
            current_boxes = [res[:, :4] if res.shape[0] > 0 else torch.empty((0, 4), device=device) for res in anchor_results]
            current_confs = [res[:, 4:5] if res.shape[0] > 0 else torch.empty((0, 1), device=device) for res in anchor_results]
            current_classes = [res[:, 5:6] if res.shape[0] > 0 else torch.empty((0, 1), device=device) for res in anchor_results]
            
            predictions = [{'boxes': current_boxes, 'confs': current_confs, 'classes': current_classes}]
            
            # --- P-FRAME Tracks ---
            for t in range(num_pframes):
                if mode == 'yolo_full':
                    # YOLO FULL BASELINE: Complete RGB processing everywhere
                    decode_tracker.record_iframe() # Counts as max compute
                    fallback_rgb = pframe_rgbs[t].unsqueeze(0).to(device)
                    anchor_results = model.anchor_detector.get_anchor_boxes(fallback_rgb)
                    current_boxes = [r[:, :4] if r.shape[0] > 0 else torch.empty((0, 4), device=device) for r in anchor_results]
                    current_confs = [r[:, 4:5] if r.shape[0] > 0 else torch.empty((0, 1), device=device) for r in anchor_results]
                    current_classes = [r[:, 5:6] if r.shape[0] > 0 else torch.empty((0, 1), device=device) for r in anchor_results]
                    
                elif mode == 'copy_paste':
                    # NAIVE COPY-PASTE BASELINE: Do literally nothing. Just push I-Frame anchors forward blindly
                    decode_tracker.record_pframe(was_refreshed=False) 
                    # boxes remain identically `current_boxes` from before
                    
                elif mode == 'cope':
                    # OUR ARCHITECTURE: Fast HEVC tracking updates
                    decode_tracker.record_pframe(was_refreshed=False)
                    r_feat = res[t].permute(2, 0, 1) 
                    d_feat = depths[t].permute(2, 0, 1).float()
                    m_feat = modes[t].permute(2, 0, 1).float()
                    app_t = torch.cat([r_feat, d_feat, m_feat], dim=0).unsqueeze(0) 
                    mvs_t = mvs[t].permute(2, 0, 1).unsqueeze(0) 
                    
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
                
            latency.stop()
            
            # --- Target Validation ---
            for t, pred_t in enumerate(predictions):
                if t >= len(targets):
                    break
                
                target_t = targets[t]
                gt_boxes = target_t['boxes'].numpy() if len(target_t['boxes']) > 0 else np.zeros((0,4))
                gt_labels = target_t['labels'].numpy() if len(target_t['labels']) > 0 else np.zeros(0, dtype=np.int64)
                
                # IMPORTANT: BDD100K JSON stores boxes as [x1, y1, width, height]
                # metrics module requires [x1, y1, x2, y2] to compute intersection boundaries!
                if gt_boxes.shape[0] > 0:
                    gt_boxes[:, 2] = gt_boxes[:, 0] + gt_boxes[:, 2] # x2 = x1 + w
                    gt_boxes[:, 3] = gt_boxes[:, 1] + gt_boxes[:, 3] # y2 = y1 + h
                
                if len(pred_t['boxes']) > 0 and pred_t['boxes'][0].shape[0] > 0:
                    pred_boxes = pred_t['boxes'][0].cpu().numpy()
                    pred_conf = pred_t['confs'][0].cpu().numpy().squeeze()
                    if pred_conf.ndim == 0: pred_conf = np.array([pred_conf])
                        
                    pred_cls = pred_t['classes'][0].cpu().numpy().squeeze()
                    if pred_cls.ndim == 0: pred_cls = np.array([pred_cls]).astype(np.int64)
                    else: pred_cls = pred_cls.astype(np.int64)
                else:
                    pred_boxes = np.zeros((0, 4))
                    pred_conf = np.zeros(0)
                    pred_cls = np.zeros(0, dtype=np.int64)

                metrics.update(pred_boxes, pred_conf, pred_cls, gt_boxes, gt_labels)

    mAP = metrics.compute()
    decode_stats = decode_tracker.summary()

    results = {
        'mode': mode,
        'mAP_50': mAP['mAP_50'],
        'mAP_50_95': mAP['mAP_50_95'],
        'decode_budget': decode_stats['decode_percent'],
        'latency_ms': latency.summary()['inference_total']['mean_ms']
    }
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run baseline evaluations.')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['yolo_full', 'copy_paste', 'cope', 'all'])
    parser.add_argument('--dataset', type=str, default='bdd100k',
                        choices=['bdd100k', 'imagenetvid'])
    parser.add_argument('--root', type=str, default=None,
                        help='Dataset root dir (defaults based on --dataset)')
    parser.add_argument('--split', type=str, default=None,
                        help="Dataset split (default: 'train' for bdd, 'val' for vid)")
    parser.add_argument('--num_classes', type=int, default=None,
                        help='Class count (defaults: bdd100k=10, imagenetvid=30)')
    parser.add_argument('--yolo_weights', type=str, default='yolov8m.pt',
                        help='YOLO checkpoint (use fine-tuned best.pt for VID)')
    parser.add_argument('--class_mapping', type=str, default=None,
                        help="'coco_to_bdd', 'coco_to_vid', 'identity', or omit for dataset default")
    parser.add_argument('--features', type=str, default='features',
                        choices=['features', 'features_pyav'])
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to finetuned CoPE checkpoint (delta_encoder+fusion_head)')
    parser.add_argument('--gop_length', type=int, default=16)
    parser.add_argument('--max_eval', type=int, default=50,
                        help='Max GOPs to evaluate per mode (default 50)')
    parser.add_argument('--annotated_only', action='store_true',
                        help='Filter to annotated GOPs only (BDD legacy)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save JSON results (default: checkpoints/baseline_comparison[_vid].json)')
    args = parser.parse_args()

    # Dataset-specific defaults
    if args.root is None:
        args.root = ('D:/cope-delta-det2/data/bdd100k' if args.dataset == 'bdd100k'
                     else 'D:/cope-delta-det2/data/imagenetvid')
    if args.num_classes is None:
        args.num_classes = 10 if args.dataset == 'bdd100k' else 30
    if args.class_mapping is None:
        # identity = correct for VID-finetuned YOLO (nc=30, outputs VID IDs directly)
        # coco_to_vid = correct for COCO-pretrained yolov8m.pt (nc=80)
        args.class_mapping = 'coco_to_bdd' if args.dataset == 'bdd100k' else 'identity'
    if args.split is None:
        args.split = 'train' if args.dataset == 'bdd100k' else 'val'
    if args.checkpoint is None:
        args.checkpoint = ('D:/cope-delta-det2/checkpoints/finetuned_best.pt'
                           if args.dataset == 'bdd100k'
                           else 'D:/cope-delta-det2/checkpoints/vid_finetuned_best.pt')
    annotated_only_flag = args.annotated_only or (args.dataset == 'bdd100k')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Preparing baseline benchmarks on {device}...")
    print(f"  Dataset: {args.dataset}  root={args.root}  split={args.split}  "
          f"num_classes={args.num_classes}")
    print(f"  YOLO weights: {args.yolo_weights}  class_mapping={args.class_mapping}")
    print(f"  Features: {args.features}  Checkpoint: {args.checkpoint}")
    print(f"  Max eval GOPs per mode: {args.max_eval}")

    dataset = BDD100KCoPEDataset(root_dir=args.root, split=args.split,
                                  gop_length=args.gop_length,
                                  annotated_only=annotated_only_flag,
                                  features_subdir=args.features,
                                  dataset_type=args.dataset)

    model = CoPEDeltaDet(yolo_size=args.yolo_weights, embed_dim=256,
                         num_classes=args.num_classes, device=device).to(device)

    # Propagate class_mapping to the YOLO anchor inside the model, if supported
    COCO_TO_VID = {
        4: 0, 21: 2, 1: 3, 14: 4, 5: 5, 2: 6, 7: 6,
        19: 7, 16: 8, 15: 9, 20: 10, 17: 14, 3: 18,
        18: 21, 6: 25, 8: 27, 22: 29,
    }
    try:
        if hasattr(model.anchor_detector, 'class_mapping'):
            if args.class_mapping in (None, 'identity'):
                model.anchor_detector.class_mapping = None
                print(f"  class_mapping: identity (VID-finetuned YOLO outputs VID IDs directly)")
            elif args.class_mapping == 'coco_to_bdd':
                model.anchor_detector.class_mapping = {
                    0: 0, 1: 7, 2: 2, 3: 6, 5: 4, 6: 5, 7: 3, 9: 8, 11: 9,
                }
            elif args.class_mapping == 'coco_to_vid':
                model.anchor_detector.class_mapping = COCO_TO_VID
                print(f"  class_mapping: coco_to_vid ({len(COCO_TO_VID)} pairs)")
    except Exception as e:
        print(f"  (note) could not set class_mapping on anchor: {e}")

    # Load Stage 2 Finetuned tracking weights (only delta_encoder + fusion_head)
    ckpt_path = args.checkpoint
    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        if 'delta_encoder' in ckpt:
            model.delta_encoder.load_state_dict(ckpt['delta_encoder'])
            model.fusion_head.load_state_dict(ckpt['fusion_head'])
            loss_val = ckpt.get('loss', None)
            loss_str = f"{loss_val:.1f}" if isinstance(loss_val, (int, float)) else str(loss_val)
            print(f"Loaded finetuned CoPE tracker (epoch {ckpt.get('epoch', '?')}, loss {loss_str})")
        else:
            # Legacy: full model state_dict
            model.load_state_dict(ckpt, strict=False)
            print("Loaded checkpoint (legacy format, strict=False)")
    else:
        print(f"WARNING: Could not find checkpoint at {ckpt_path}; running with untrained CoPE components.")

    modes_to_test = ['yolo_full', 'copy_paste', 'cope'] if args.mode == 'all' else [args.mode]

    final_reports = []
    print("\n" + "="*50)
    model.eval()

    for m in modes_to_test:
        res = evaluate_baseline(model, dataset, device, mode=m,
                                num_classes=args.num_classes,
                                max_eval=args.max_eval)
        final_reports.append(res)

        print(f"\n--- RESULTS FOR: {m.upper()} ---")
        print(f"Decode Budget (Compute Cost): {res['decode_budget']:.1f}%")
        print(f"mAP@50:                       {res['mAP_50']*100:.2f}%")
        print(f"mAP@[.5:.95]:                 {res['mAP_50_95']*100:.2f}%")
        print(f"Latency Per GOP:              {res['latency_ms']:.2f}ms")
        print("="*50)

    # Save to JSON
    if args.output:
        out_path = args.output
    else:
        out_name = 'baseline_comparison.json' if args.dataset == 'bdd100k' else 'baseline_comparison_vid.json'
        out_path = f"D:/cope-delta-det2/checkpoints/{out_name}"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(final_reports, f, indent=4)
    print(f"\nAll baseline results saved to {out_path}!")
