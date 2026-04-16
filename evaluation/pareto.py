"""Generate Pareto curves: mAP vs decode budget.

Varies GOP length and refresh threshold to trace the accuracy/efficiency
trade-off frontier for CoPE-Δ-Det vs baselines.

Usage:
    python evaluation/pareto.py --config configs/eval.yaml \
                                --checkpoint checkpoints/finetune/best.pt
"""

import argparse
import sys
import json
from pathlib import Path

import torch
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.evaluate import evaluate_cope_delta_det
from data.dataset import GOPDataset
from models.cope_delta_det import CoPEDeltaDet
from utils.visualization import plot_pareto_curve


def generate_pareto_points(model, config, device, gop_lengths=None,
                            thresholds=None):
    """Generate Pareto curve points by varying GOP length and refresh threshold.

    Args:
        model: CoPEDeltaDet model
        config: configuration dict
        device: torch device
        gop_lengths: list of GOP lengths to evaluate
        thresholds: list of refresh threshold values

    Returns:
        list of dicts with (decode_budget, mAP_50, mAP_50_95, gop, threshold)
    """
    if gop_lengths is None:
        gop_lengths = [8, 16, 32]
    if thresholds is None:
        thresholds = [0.2, 0.3, 0.5, 0.7, 1.0, float('inf')]  # inf = no refresh

    points = []

    for gop in gop_lengths:
        feat_subdir = f"qp{config['hevc']['default_qp']}_gop{gop}"
        feat_dir = config['paths']['extracted_features'] + f'/{feat_subdir}'

        if not Path(feat_dir).exists():
            print(f"Skipping GOP={gop}: features not found at {feat_dir}")
            continue

        dataset = GOPDataset(
            images_dir=config['paths']['bdd100k_root'] + '/images/val',
            features_dir=feat_dir,
            annotations_path=config['paths']['bdd100k_root'] + '/annotations_val_coco.json',
            gop_index_path=config['paths']['bdd100k_root'] + '/gop_index_val.json',
            gop_length=gop,
            split='val'
        )

        for thresh in thresholds:
            print(f"\nEvaluating GOP={gop}, threshold={thresh}")

            # Update refresh threshold
            model.refresh_policy.threshold.data = torch.tensor(thresh)

            enable_refresh = thresh < float('inf')
            results = evaluate_cope_delta_det(
                model, dataset, device,
                enable_refresh=enable_refresh,
                measure_latency=False
            )

            point = {
                'gop_length': gop,
                'refresh_threshold': thresh if thresh < float('inf') else 'none',
                'decode_budget': results['decode_budget'],
                'mAP_50': results['mAP_50'],
                'mAP_50_95': results['mAP_50_95'],
            }
            points.append(point)
            print(f"  decode={point['decode_budget']:.1f}%, "
                  f"mAP@0.5={point['mAP_50']:.4f}")

    return points


def main():
    parser = argparse.ArgumentParser(description='Generate Pareto curves')
    parser.add_argument('--config', type=str, default='configs/eval.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results/pareto')
    parser.add_argument('--gop', type=int, nargs='+', default=[8, 16, 32])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CoPEDeltaDet(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    # Generate points
    points = generate_pareto_points(model, config, device, gop_lengths=args.gop)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'pareto_points.json', 'w') as f:
        json.dump(points, f, indent=2)

    # Plot Pareto curve
    decode_budgets = [[p['decode_budget'] for p in points]]
    mAPs = [[p['mAP_50'] for p in points]]

    plot_pareto_curve(
        decode_budgets, mAPs,
        method_names=['CoPE-Δ-Det'],
        save_path=str(output_dir / 'pareto_curve.png')
    )

    print(f"\nPareto analysis saved to {output_dir}")


if __name__ == '__main__':
    main()
