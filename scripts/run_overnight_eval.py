"""
Post-training evaluation chain for CoPE-Delta-Det2.

Runs all evaluations in sequence after training finishes at epoch 30:
  1. Full-val baselines: YOLO Full, Copy-Paste (10,753 GOPs)
  2. Full-val CoPE (block-match) and CoPE (PyAV) (10,753 GOPs)
  3. Ablation study on 500 GOPs: full, mv_only, appearance_only
  4. Generate overnight_summary.txt with all results + training curve

Usage:
  python scripts/run_overnight_eval.py [--wait] [--checkpoint PATH]

Flags:
  --wait          Poll until checkpoints/vid_finetuned_model_epoch_30.pt exists
                  before starting (use if training is still running)
  --checkpoint    Path to CoPE checkpoint (default: checkpoints/vid_finetuned_best.pt)
  --yolo          YOLO weights (default: runs/detect/yolov8m_vid/weights/best.pt)
  --max_full      GOPs for the full eval (default: None = all ~10753)
  --max_ablation  GOPs for the ablation runs (default: 500)
  --dry_run       Print commands without executing them
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
CHECKPOINTS = ROOT / 'checkpoints'
REPORT = ROOT / 'report'
EVAL_PY = ROOT / 'evaluation' / 'evaluate.py'
BASELINES_PY = ROOT / 'evaluation' / 'evaluate_baselines.py'
HISTORY_JSON = CHECKPOINTS / 'vid_loss_history.json'


def run(cmd: list[str], dry_run: bool) -> int:
    joined = ' '.join(str(c) for c in cmd)
    print(f'\n>>> {joined}')
    if dry_run:
        print('    [dry_run — skipped]')
        return 0
    result = subprocess.run(cmd, env=os.environ)
    return result.returncode


def wait_for_training(epoch: int = 30, poll_secs: int = 300):
    """Block until the epoch-30 checkpoint appears on disk."""
    target = CHECKPOINTS / f'vid_finetuned_model_epoch_{epoch}.pt'
    print(f'Waiting for training to finish ({target}) ...')
    while not target.exists():
        print(f'  [{datetime.now():%H:%M:%S}] not found yet — sleeping {poll_secs}s')
        time.sleep(poll_secs)
    print(f'  [{datetime.now():%H:%M:%S}] Found! Training complete.')


def load_json(path: Path) -> dict | None:
    """Load a result JSON; unwrap single-element lists from evaluate_baselines.py."""
    try:
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return data[0] if len(data) == 1 else data[0]  # baselines always saves [{}]
        return data
    except Exception:
        return None


def best_checkpoint(args_checkpoint: str) -> str:
    """Return the checkpoint to use for CoPE runs."""
    p = Path(args_checkpoint)
    if p.exists():
        return str(p)
    fallback = CHECKPOINTS / 'vid_finetuned_best.pt'
    print(f'WARNING: {p} not found; falling back to {fallback}')
    return str(fallback)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', action='store_true',
                        help='Poll until epoch-30 checkpoint exists before starting')
    parser.add_argument('--checkpoint', default=str(CHECKPOINTS / 'vid_finetuned_best.pt'))
    parser.add_argument('--yolo', default='runs/detect/yolov8m_vid/weights/best.pt')
    parser.add_argument('--max_full', type=int, default=None,
                        help='GOP cap for full-val runs (None = all)')
    parser.add_argument('--max_ablation', type=int, default=500,
                        help='GOP cap for ablation runs (default 500)')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.wait:
        wait_for_training(epoch=30)

    ckpt = best_checkpoint(args.checkpoint)

    # ---------- shared CLI fragments ----------
    py = sys.executable
    dataset_flags = ['--dataset', 'imagenetvid', '--split', 'val',
                     '--class_mapping', 'identity',
                     '--yolo_weights', args.yolo,
                     '--num_classes', '30']
    cope_flags = ['--checkpoint', ckpt, '--policy_thresh', '999']
    # Baselines use max_eval too; None means all GOPs (no --max_eval flag passed = new default)
    max_full_baseline = ['--max_eval', str(args.max_full)] if args.max_full else []
    max_full_cope = ['--max_eval', str(args.max_full)] if args.max_full else []
    max_ablation = ['--max_eval', str(args.max_ablation)]

    print('\n' + '='*60)
    print(' OVERNIGHT EVAL — CoPE-Delta-Det2')
    print(f' Started: {datetime.now():%Y-%m-%d %H:%M:%S}')
    print(f' Checkpoint: {ckpt}')
    print(f' YOLO:       {args.yolo}')
    print('='*60)

    errors = []

    # ----------------------------------------------------------------
    # Task 1a — YOLO Full baseline (decode every frame)
    # ----------------------------------------------------------------
    out_yolo = RESULTS / 'full_eval_yolo_full.json'
    rc = run([py, BASELINES_PY,
              '--mode', 'yolo_full',
              *dataset_flags,
              '--output', str(out_yolo),
              *max_full_baseline], args.dry_run)
    if rc: errors.append(f'yolo_full (rc={rc})')

    # ----------------------------------------------------------------
    # Task 1b — Copy-Paste baseline
    # ----------------------------------------------------------------
    out_cp = RESULTS / 'full_eval_copy_paste.json'
    rc = run([py, BASELINES_PY,
              '--mode', 'copy_paste',
              *dataset_flags,
              '--checkpoint', ckpt,
              '--output', str(out_cp),
              *max_full_baseline], args.dry_run)
    if rc: errors.append(f'copy_paste (rc={rc})')

    # ----------------------------------------------------------------
    # Task 2a — CoPE (block-matched features)
    # ----------------------------------------------------------------
    out_cope_block = RESULTS / 'full_eval_cope_block.json'
    rc = run([py, EVAL_PY,
              '--features', 'features',
              *dataset_flags,
              *cope_flags,
              '--output', str(out_cope_block),
              *max_full_cope], args.dry_run)
    if rc: errors.append(f'cope_block (rc={rc})')

    # ----------------------------------------------------------------
    # Task 2b — CoPE (PyAV real codec MVs, zero-shot)
    # ----------------------------------------------------------------
    out_cope_pyav = RESULTS / 'full_eval_cope_pyav.json'
    rc = run([py, EVAL_PY,
              '--features', 'features_pyav',
              *dataset_flags,
              *cope_flags,
              '--output', str(out_cope_pyav),
              *max_full_cope], args.dry_run)
    if rc: errors.append(f'cope_pyav (rc={rc})')

    # ----------------------------------------------------------------
    # Task 3 — Ablation study (500 GOPs, block-matched features)
    # ----------------------------------------------------------------
    ablation_results = {}
    for mode in ['full', 'mv_only', 'appearance_only']:
        out_abl = RESULTS / f'ablation_{mode}.json'
        rc = run([py, EVAL_PY,
                  '--features', 'features',
                  *dataset_flags,
                  *cope_flags,
                  '--ablation', mode,
                  '--output', str(out_abl),
                  *max_ablation], args.dry_run)
        if rc:
            errors.append(f'ablation_{mode} (rc={rc})')
        else:
            ablation_results[mode] = load_json(out_abl)

    # ----------------------------------------------------------------
    # Task 4 — Collect all results and write summary report
    # ----------------------------------------------------------------
    def pct(v): return f'{v*100:.2f}%' if v is not None else 'N/A'

    r_yolo  = load_json(out_yolo)
    r_cp    = load_json(out_cp)
    r_block = load_json(out_cope_block)
    r_pyav  = load_json(out_cope_pyav)

    # loss history
    loss_hist = load_json(HISTORY_JSON) or []

    now = datetime.now()
    summary_path = REPORT / f'overnight_summary_{now:%Y_%m_%d}.txt'
    REPORT.mkdir(parents=True, exist_ok=True)

    lines = []
    def L(s=''): lines.append(s)

    L('=' * 76)
    L('         CoPE-Delta-Det2 — Overnight Evaluation Summary')
    L(f'         Generated: {now:%Y-%m-%d %H:%M:%S}')
    L('=' * 76)
    L()
    L('TRAINING CURVE (vid_loss_history.json)')
    L('-' * 44)
    if loss_hist:
        best_ep = min(loss_hist, key=lambda h: h['loss'])
        for h in loss_hist:
            marker = ' <-- BEST' if h['epoch'] == best_ep['epoch'] else ''
            L(f"  Epoch {h['epoch']:3d}: {h['loss']:.4f}{marker}")
        L()
        L(f"  Best loss:  {best_ep['loss']:.4f}  (epoch {best_ep['epoch']})")
    else:
        L('  (loss history not found)')
    L()

    # helper to pull a field safely
    def get(r, *keys, default=None):
        for k in keys:
            if r and k in r: return r[k]
        return default

    L('FULL-VAL RESULTS (ImageNet VID val split)')
    L('-' * 76)
    rows = [
        ('YOLO Full (100% decode)',     r_yolo,  None),
        ('Copy-Paste (6.2% decode)',    r_cp,    None),
        ('CoPE-Det block-match',        r_block, None),
        ('CoPE-Det PyAV (zero-shot)',   r_pyav,  None),
    ]
    L(f"  {'Method':<32} {'mAP@0.5':>8} {'mAP@.5:.95':>10} {'Decode':>8}")
    L('  ' + '-'*62)
    for label, r, _ in rows:
        m50   = pct(get(r, 'mAP_50'))
        m5095 = pct(get(r, 'mAP_50_95'))
        dec   = f"{get(r, 'decode_budget', default=0.0):.1f}%"
        L(f"  {label:<32} {m50:>8} {m5095:>10} {dec:>8}")
    L()

    L('ABLATION STUDY (500 GOPs, block-match features, CoPE delta encoder)')
    L('-' * 76)
    L(f"  {'Ablation mode':<24} {'mAP@0.5':>8} {'mAP@.5:.95':>10}  Notes")
    L('  ' + '-'*62)
    abl_labels = {
        'full':             ('Full (MV + Appearance)',  'baseline for ablation'),
        'mv_only':          ('MV only (res/depth=0)',   'measures MV contribution alone'),
        'appearance_only':  ('Appearance only (MV=0)',  'measures residual/depth/mode'),
    }
    for mode, (label, note) in abl_labels.items():
        r = ablation_results.get(mode) or {}
        m50   = pct(r.get('mAP_50'))
        m5095 = pct(r.get('mAP_50_95'))
        L(f"  {label:<24} {m50:>8} {m5095:>10}  {note}")
    L()

    L('KEY FINDINGS')
    L('-' * 44)
    if r_yolo and r_block:
        ratio = r_block['mAP_50'] / r_yolo['mAP_50'] * 100
        L(f"  - CoPE-Det achieves {ratio:.1f}% of YOLO Full mAP at 6.2% decode cost.")
    if r_block and r_pyav:
        delta = (r_block['mAP_50'] - r_pyav['mAP_50']) * 100
        L(f"  - Zero-shot PyAV transfer gap: {delta:+.2f}pp vs block-match.")
    if ablation_results.get('full') and ablation_results.get('mv_only'):
        d = (ablation_results['full']['mAP_50'] - ablation_results['mv_only']['mAP_50']) * 100
        L(f"  - Appearance contribution: {d:+.2f}pp mAP@0.5 (full minus mv_only).")
    if ablation_results.get('full') and ablation_results.get('appearance_only'):
        d = (ablation_results['full']['mAP_50'] - ablation_results['appearance_only']['mAP_50']) * 100
        L(f"  - MV contribution:         {d:+.2f}pp mAP@0.5 (full minus appearance_only).")
    L()

    if errors:
        L('ERRORS')
        L('-' * 44)
        for e in errors:
            L(f'  [FAILED] {e}')
        L()

    L('=' * 76)
    L('                       End of Overnight Summary')
    L('=' * 76)

    report_text = '\n'.join(lines)
    print('\n' + report_text)

    summary_path.write_text(report_text)
    print(f'\nSummary written to {summary_path}')

    # Also save a machine-readable JSON with all numbers
    combined = {
        'generated': now.isoformat(),
        'checkpoint': ckpt,
        'yolo': args.yolo,
        'training': {
            'best_epoch': best_ep['epoch'] if loss_hist else None,
            'best_loss': best_ep['loss'] if loss_hist else None,
            'history': loss_hist,
        },
        'full_eval': {
            'yolo_full':    r_yolo,
            'copy_paste':   r_cp,
            'cope_block':   r_block,
            'cope_pyav':    r_pyav,
        },
        'ablations': ablation_results,
        'errors': errors,
    }
    combined_path = RESULTS / f'overnight_combined_{now:%Y_%m_%d}.json'
    combined_path.write_text(json.dumps(combined, indent=2))
    print(f'Combined JSON written to {combined_path}')

    sys.exit(1 if errors else 0)


if __name__ == '__main__':
    main()
