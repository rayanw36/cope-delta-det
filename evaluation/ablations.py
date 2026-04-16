"""Run ablation experiments for CoPE-Δ-Det.

Ablation studies:
1. MV branch only vs Residual branch only vs Both
2. With vs without CU partition depth and prediction mode features
3. Transformer fusion head vs BiLSTM
4. With vs without refresh policy
5. Number of Δ-tokens per object (4, 8, 16)

Usage:
    python evaluation/ablations.py --config configs/eval.yaml \
                                   --checkpoint checkpoints/finetune/best.pt
"""

import argparse
import sys
import json
from pathlib import Path

import torch
import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.evaluate import evaluate_cope_delta_det
from data.dataset import GOPDataset
from models.cope_delta_det import CoPEDeltaDet, CoPEDeltaDetAblation
from utils.metrics import COCOMetrics, DecodeBudgetTracker
from utils.box_utils import xyxy_to_xywh, xywh_to_xyxy


def run_branch_ablation(config, checkpoint_path, dataset, device):
    """Ablation 1: MV branch only vs Residual branch only vs Both.

    Returns:
        dict with results for each configuration
    """
    results = {}

    for mode in ['full', 'mv_only', 'residual_only']:
        print(f"\n--- Branch ablation: {mode} ---")
        model = CoPEDeltaDet(config).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        model.eval()

        metrics = COCOMetrics(num_classes=10)
        decode_tracker = DecodeBudgetTracker()

        with torch.no_grad():
            for idx in tqdm(range(len(dataset)), desc=f"Ablation: {mode}"):
                sample = dataset[idx]
                gt_boxes_gop = sample['gt_boxes']
                gt_labels_gop = sample['gt_labels']

                anchor_boxes = gt_boxes_gop[0].to(device)
                anchor_labels = gt_labels_gop[0].to(device)
                anchor_scores = torch.ones(anchor_boxes.shape[0], device=device)
                decode_tracker.record_iframe()

                if anchor_boxes.numel() == 0:
                    for _ in sample['p_frame_features']:
                        decode_tracker.record_pframe()
                    continue

                anchor_xywh = xyxy_to_xywh(anchor_boxes)
                anchor_emb = model.anchor_embed(
                    anchor_xywh, anchor_scores, anchor_labels,
                    num_classes=model.num_classes
                )

                current_xywh = anchor_xywh.clone()
                current_conf = anchor_scores.clone()

                for t, feat in enumerate(sample['p_frame_features']):
                    mv = feat['mv_tensor'].unsqueeze(0).to(device)
                    res = feat['residual_energy'].unsqueeze(0).to(device)
                    depth = feat['partition_depth'].unsqueeze(0).to(device)
                    pred_mode = feat['pred_mode'].unsqueeze(0).to(device)

                    current_xyxy = xywh_to_xyxy(current_xywh)

                    if mode == 'mv_only':
                        delta_tokens = model.delta_encoder.forward_motion_only(
                            mv, [current_xyxy]
                        )
                    elif mode == 'residual_only':
                        delta_tokens = model.delta_encoder.forward_residual_only(
                            res, depth, pred_mode, [current_xyxy]
                        )
                    else:
                        delta_tokens = model.delta_encoder(
                            mv, res, depth, pred_mode, [current_xyxy]
                        )

                    pred = model.fusion_head(
                        anchor_emb, delta_tokens[0], temporal_idx=t
                    )

                    current_xywh = current_xywh + pred['box_deltas']
                    current_conf = current_conf * pred['confidence']
                    decode_tracker.record_pframe()

                    gt_idx = t + 1
                    if gt_idx < len(gt_boxes_gop):
                        pred_xyxy = xywh_to_xyxy(current_xywh)
                        metrics.update(
                            pred_xyxy.cpu().numpy(),
                            current_conf.cpu().numpy(),
                            anchor_labels.cpu().numpy(),
                            gt_boxes_gop[gt_idx].numpy(),
                            gt_labels_gop[gt_idx].numpy()
                        )

        mAP = metrics.compute()
        results[mode] = {
            'mAP_50': mAP['mAP_50'],
            'mAP_50_95': mAP['mAP_50_95'],
            'decode_budget': decode_tracker.summary()['decode_percent']
        }
        print(f"  mAP@0.5: {results[mode]['mAP_50']:.4f}")

    return results


def run_fusion_ablation(config, checkpoint_path, dataset, device):
    """Ablation 3: Transformer vs BiLSTM fusion head."""
    results = {}

    for head_type in ['transformer', 'bilstm']:
        print(f"\n--- Fusion ablation: {head_type} ---")

        if head_type == 'bilstm':
            model = CoPEDeltaDetAblation(config, ablation_mode='bilstm').to(device)
        else:
            model = CoPEDeltaDet(config).to(device)

        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)

        eval_results = evaluate_cope_delta_det(
            model, dataset, device, enable_refresh=False, measure_latency=True
        )
        results[head_type] = {
            'mAP_50': eval_results['mAP_50'],
            'mAP_50_95': eval_results['mAP_50_95'],
            'latency': eval_results.get('latency', {})
        }
        print(f"  mAP@0.5: {results[head_type]['mAP_50']:.4f}")

    return results


def run_refresh_ablation(config, checkpoint_path, dataset, device):
    """Ablation 4: With vs without refresh policy."""
    results = {}

    model = CoPEDeltaDet(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    for enable in [True, False]:
        label = 'with_refresh' if enable else 'without_refresh'
        print(f"\n--- Refresh ablation: {label} ---")

        eval_results = evaluate_cope_delta_det(
            model, dataset, device, enable_refresh=enable
        )
        results[label] = {
            'mAP_50': eval_results['mAP_50'],
            'mAP_50_95': eval_results['mAP_50_95'],
            'decode_budget': eval_results['decode_budget']
        }
        print(f"  mAP@0.5: {results[label]['mAP_50']:.4f}, "
              f"decode: {results[label]['decode_budget']:.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description='Run ablation experiments')
    parser.add_argument('--config', type=str, default='configs/eval.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results/ablations')
    parser.add_argument('--ablations', nargs='+',
                        default=['branch', 'fusion', 'refresh'],
                        choices=['branch', 'fusion', 'refresh'])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feat_subdir = f"qp{config['hevc']['default_qp']}_gop{config['hevc']['default_gop']}"
    dataset = GOPDataset(
        images_dir=config['paths']['bdd100k_root'] + '/images/val',
        features_dir=config['paths']['extracted_features'] + f'/{feat_subdir}',
        annotations_path=config['paths']['bdd100k_root'] + '/annotations_val_coco.json',
        gop_index_path=config['paths']['bdd100k_root'] + '/gop_index_val.json',
        gop_length=config['hevc']['default_gop'],
        split='val'
    )

    all_results = {}

    if 'branch' in args.ablations:
        all_results['branch'] = run_branch_ablation(
            config, args.checkpoint, dataset, device
        )

    if 'fusion' in args.ablations:
        all_results['fusion'] = run_fusion_ablation(
            config, args.checkpoint, dataset, device
        )

    if 'refresh' in args.ablations:
        all_results['refresh'] = run_refresh_ablation(
            config, args.checkpoint, dataset, device
        )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'ablation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))

    # Print summary table
    print(f"\n{'='*60}")
    print(f"Ablation Summary")
    print(f"{'='*60}")
    for ablation_name, ablation_results in all_results.items():
        print(f"\n{ablation_name.upper()}:")
        for variant, res in ablation_results.items():
            print(f"  {variant:25s}  mAP@0.5={res['mAP_50']:.4f}  "
                  f"mAP@[.5:.95]={res['mAP_50_95']:.4f}")

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
