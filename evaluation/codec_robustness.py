"""Codec Robustness Sweep: Evaluate across QP x GOP grid.

Evaluates CoPE-Δ-Det at each combination of QP ∈ {22, 27, 32, 37} and
GOP ∈ {8, 16, 32} to analyze robustness to codec parameters.

Generates a 4x3 heatmap of mAP values and analyzes how MV quality and
residual energy statistics change with QP.

Usage:
    python evaluation/codec_robustness.py --config configs/eval.yaml \
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
from models.cope_delta_det import CoPEDeltaDet
from utils.visualization import plot_codec_robustness_heatmap


def compute_feature_statistics(dataset, num_samples=100):
    """Compute statistics of extracted codec features.

    Args:
        dataset: GOPDataset
        num_samples: number of GOPs to analyze

    Returns:
        dict with feature statistics
    """
    mv_magnitudes = []
    residual_energies = []
    partition_depths = []

    for idx in range(min(num_samples, len(dataset))):
        sample = dataset[idx]
        for feat in sample['p_frame_features']:
            mv = feat['mv_tensor']
            mv_mag = torch.sqrt(mv[0]**2 + mv[1]**2).mean().item()
            mv_magnitudes.append(mv_mag)

            res = feat['residual_energy']
            residual_energies.append(res.mean().item())

            depth = feat['partition_depth']
            partition_depths.append(depth.mean().item())

    return {
        'mv_magnitude': {
            'mean': np.mean(mv_magnitudes),
            'std': np.std(mv_magnitudes)
        },
        'residual_energy': {
            'mean': np.mean(residual_energies),
            'std': np.std(residual_energies)
        },
        'partition_depth': {
            'mean': np.mean(partition_depths),
            'std': np.std(partition_depths)
        }
    }


def run_codec_sweep(model, config, device, qp_values=None, gop_lengths=None):
    """Run full QP x GOP evaluation sweep.

    Args:
        model: CoPEDeltaDet model
        config: configuration dict
        device: torch device
        qp_values: list of QP values
        gop_lengths: list of GOP lengths

    Returns:
        dict with results grid and feature statistics
    """
    if qp_values is None:
        qp_values = [22, 27, 32, 37]
    if gop_lengths is None:
        gop_lengths = [8, 16, 32]

    results_grid = {}
    mAP_matrix = np.zeros((len(qp_values), len(gop_lengths)))
    decode_matrix = np.zeros((len(qp_values), len(gop_lengths)))
    feature_stats = {}

    for qi, qp in enumerate(qp_values):
        for gi, gop in enumerate(gop_lengths):
            key = f"qp{qp}_gop{gop}"
            feat_dir = config['paths']['extracted_features'] + f'/{key}'

            if not Path(feat_dir).exists():
                print(f"Skipping {key}: features not found")
                mAP_matrix[qi, gi] = np.nan
                continue

            print(f"\nEvaluating {key}...")

            dataset = GOPDataset(
                images_dir=config['paths']['bdd100k_root'] + '/images/val',
                features_dir=feat_dir,
                annotations_path=config['paths']['bdd100k_root'] + '/annotations_val_coco.json',
                gop_index_path=config['paths']['bdd100k_root'] + '/gop_index_val.json',
                gop_length=gop,
                split='val'
            )

            # Evaluate
            eval_results = evaluate_cope_delta_det(
                model, dataset, device,
                enable_refresh=True, measure_latency=False
            )

            # Feature statistics
            stats = compute_feature_statistics(dataset)

            results_grid[key] = {
                'qp': qp,
                'gop': gop,
                'mAP_50': eval_results['mAP_50'],
                'mAP_50_95': eval_results['mAP_50_95'],
                'decode_budget': eval_results['decode_budget'],
                'feature_stats': stats
            }

            mAP_matrix[qi, gi] = eval_results['mAP_50']
            decode_matrix[qi, gi] = eval_results['decode_budget']
            feature_stats[key] = stats

            print(f"  mAP@0.5: {eval_results['mAP_50']:.4f}, "
                  f"decode: {eval_results['decode_budget']:.1f}%")

    return {
        'results_grid': results_grid,
        'mAP_matrix': mAP_matrix,
        'decode_matrix': decode_matrix,
        'feature_stats': feature_stats,
        'qp_values': qp_values,
        'gop_lengths': gop_lengths
    }


def main():
    parser = argparse.ArgumentParser(description='Codec Robustness Sweep')
    parser.add_argument('--config', type=str, default='configs/eval.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results/codec_robustness')
    parser.add_argument('--qp', type=int, nargs='+', default=[22, 27, 32, 37])
    parser.add_argument('--gop', type=int, nargs='+', default=[8, 16, 32])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CoPEDeltaDet(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    # Run sweep
    sweep_results = run_codec_sweep(
        model, config, device,
        qp_values=args.qp, gop_lengths=args.gop
    )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON (convert numpy arrays)
    save_results = {k: v for k, v in sweep_results.items()
                    if k not in ['mAP_matrix', 'decode_matrix']}
    save_results['mAP_matrix'] = sweep_results['mAP_matrix'].tolist()
    save_results['decode_matrix'] = sweep_results['decode_matrix'].tolist()

    with open(output_dir / 'codec_sweep_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)

    # Generate heatmaps
    plot_codec_robustness_heatmap(
        args.qp, args.gop, sweep_results['mAP_matrix'],
        save_path=str(output_dir / 'mAP_heatmap.png')
    )

    # Print summary table
    print(f"\n{'='*60}")
    print(f"Codec Robustness Summary — mAP@0.5")
    print(f"{'='*60}")
    header = f"{'QP':>6}" + "".join(f"{'GOP='+str(g):>12}" for g in args.gop)
    print(header)
    print("-" * len(header))
    for qi, qp in enumerate(args.qp):
        row = f"{qp:>6}"
        for gi, gop in enumerate(args.gop):
            val = sweep_results['mAP_matrix'][qi, gi]
            row += f"{val:>12.4f}" if not np.isnan(val) else f"{'N/A':>12}"
        print(row)

    # Feature statistics summary
    print(f"\nFeature Statistics by QP:")
    for qp in args.qp:
        key = f"qp{qp}_gop{args.gop[1]}"  # Use middle GOP for comparison
        if key in sweep_results['feature_stats']:
            stats = sweep_results['feature_stats'][key]
            print(f"  QP={qp}: MV_mag={stats['mv_magnitude']['mean']:.2f}, "
                  f"Res_energy={stats['residual_energy']['mean']:.2f}, "
                  f"Depth={stats['partition_depth']['mean']:.2f}")

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
