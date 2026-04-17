"""Plot YOLOv8 training loss curves from Ultralytics' ``results.csv`` files.

Ultralytics writes ``runs/detect/<name>/results.csv`` after every epoch with
train/val losses + mAP metrics. This script loads one or more such CSVs
(useful when a training run was resumed and produced multiple run dirs) and
writes a combined loss-vs-epoch + mAP-vs-epoch plot.

Usage:
    # Single run
    python training/plot_yolo_loss.py `
        --csv runs/detect/yolov8m_vid/results.csv `
        --output runs/detect/yolov8m_vid/loss_curve.png

    # Multiple runs stitched together (original + resumed continuation)
    python training/plot_yolo_loss.py `
        --csv runs/detect/yolov8m_vid/results.csv `
              runs/detect/yolov8m_vid2/results.csv `
        --output runs/detect/yolov8m_vid_combined_loss.png
"""

import argparse
import csv
from pathlib import Path
import sys


def load_csv(path: Path):
    """Return list of dict rows with numeric fields parsed as float."""
    rows = []
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                k = k.strip()
                v = v.strip() if isinstance(v, str) else v
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            rows.append(parsed)
    return rows


def stitch_runs(csv_paths):
    """Load + concatenate multiple results.csv files with epochs renumbered so
    the continuation picks up after the last epoch of the previous run."""
    combined = []
    offset = 0
    for p in csv_paths:
        rows = load_csv(Path(p))
        if not rows:
            continue
        for r in rows:
            # Ultralytics epoch is 1-indexed already
            r['_global_epoch'] = int(r.get('epoch', 0)) + offset
            combined.append(r)
        # Offset for the next run is the highest epoch seen in this one
        max_e = max(int(r.get('epoch', 0)) for r in rows)
        offset += max_e
    return combined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', nargs='+', required=True,
                    help='One or more results.csv paths (in chronological order)')
    ap.add_argument('--output', type=str, default=None,
                    help='Output PNG path (default: alongside the first CSV)')
    ap.add_argument('--title', type=str, default='YOLOv8m on ImageNet VID')
    args = ap.parse_args()

    csv_paths = [Path(p) for p in args.csv]
    for p in csv_paths:
        if not p.exists():
            print(f"ERROR: {p} not found")
            sys.exit(1)

    rows = stitch_runs(csv_paths)
    if not rows:
        print("ERROR: no rows loaded")
        sys.exit(1)
    print(f"Loaded {len(rows)} total epochs across {len(csv_paths)} run(s)")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib not installed. pip install matplotlib")
        sys.exit(1)

    # Extract series (tolerant of column-name variations)
    def _get(row, *keys):
        for k in keys:
            if k in row and isinstance(row[k], float):
                return row[k]
        return None

    epochs = [r['_global_epoch'] for r in rows]
    train_box = [_get(r, 'train/box_loss') for r in rows]
    train_cls = [_get(r, 'train/cls_loss') for r in rows]
    train_dfl = [_get(r, 'train/dfl_loss') for r in rows]
    val_box   = [_get(r, 'val/box_loss')   for r in rows]
    val_cls   = [_get(r, 'val/cls_loss')   for r in rows]
    val_dfl   = [_get(r, 'val/dfl_loss')   for r in rows]
    map50     = [_get(r, 'metrics/mAP50(B)', 'metrics/mAP_0.5', 'metrics/mAP50') for r in rows]
    map5095   = [_get(r, 'metrics/mAP50-95(B)', 'metrics/mAP_0.5:0.95', 'metrics/mAP50-95') for r in rows]

    def _total(b, c, d):
        out = []
        for bb, cc, dd in zip(b, c, d):
            if bb is None or cc is None or dd is None:
                out.append(None)
            else:
                out.append(bb + cc + dd)
        return out

    train_total = _total(train_box, train_cls, train_dfl)
    val_total   = _total(val_box, val_cls, val_dfl)

    # Figure layout: 2 rows x 2 cols
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(args.title, fontsize=14, fontweight='bold')

    # 1. Total loss
    ax = axes[0, 0]
    ax.plot(epochs, train_total, 'b-o', markersize=4, label='train total', linewidth=1.8)
    ax.plot(epochs, val_total,   'r-s', markersize=4, label='val total',   linewidth=1.8)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Total loss (box+cls+dfl)')
    ax.set_title('Total loss (train vs val)')
    ax.grid(True, alpha=0.3); ax.legend()

    # 2. Component losses (train)
    ax = axes[0, 1]
    ax.plot(epochs, train_box, label='box',  linewidth=1.6)
    ax.plot(epochs, train_cls, label='cls',  linewidth=1.6)
    ax.plot(epochs, train_dfl, label='dfl',  linewidth=1.6)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Training loss')
    ax.set_title('Training loss components')
    ax.grid(True, alpha=0.3); ax.legend()

    # 3. Component losses (val)
    ax = axes[1, 0]
    ax.plot(epochs, val_box, label='box',  linewidth=1.6)
    ax.plot(epochs, val_cls, label='cls',  linewidth=1.6)
    ax.plot(epochs, val_dfl, label='dfl',  linewidth=1.6)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Validation loss')
    ax.set_title('Validation loss components')
    ax.grid(True, alpha=0.3); ax.legend()

    # 4. mAP curves
    ax = axes[1, 1]
    ax.plot(epochs, map50,   'g-o', markersize=4, label='mAP@0.5',    linewidth=1.8)
    ax.plot(epochs, map5095, 'm-s', markersize=4, label='mAP@[.5:.95]', linewidth=1.8)
    if any(v is not None for v in map50):
        best_i = max(range(len(map50)),
                     key=lambda i: (map50[i] if map50[i] is not None else -1))
        best_e = epochs[best_i]; best_v = map50[best_i]
        ax.axvline(x=best_e, color='g', linestyle='--', alpha=0.5)
        ax.annotate(f'best mAP50: {best_v:.3f} @ ep{best_e}',
                    xy=(best_e, best_v), xytext=(best_e + 1, best_v - 0.05),
                    color='g', fontsize=9)
    ax.set_xlabel('Epoch'); ax.set_ylabel('mAP')
    ax.set_title('Validation mAP')
    ax.grid(True, alpha=0.3); ax.legend()
    ax.set_ylim(0, 1)

    # If multiple CSVs were concatenated, draw a vertical divider where the
    # continuation started so it's obvious in the plot
    if len(csv_paths) > 1:
        run_ends = []
        cum = 0
        for p in csv_paths[:-1]:
            rs = load_csv(p)
            if rs:
                cum += max(int(r.get('epoch', 0)) for r in rs)
                run_ends.append(cum)
        for a in axes.flat:
            for e in run_ends:
                a.axvline(x=e + 0.5, color='k', linestyle=':', alpha=0.5)

    plt.tight_layout()
    out_path = args.output or str(csv_paths[0].parent / 'loss_curve.png')
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved plot to: {out_path}")

    # Tiny summary
    if train_total[0] is not None and train_total[-1] is not None:
        print(f"  train total: {train_total[0]:.3f} -> {train_total[-1]:.3f}  "
              f"({(train_total[0] - train_total[-1]) / train_total[0] * 100:+.1f}%)")
    if val_total[0] is not None and val_total[-1] is not None:
        print(f"  val   total: {val_total[0]:.3f} -> {val_total[-1]:.3f}")
    if any(v is not None for v in map50):
        vals = [v for v in map50 if v is not None]
        print(f"  mAP50 min/max: {min(vals):.3f} / {max(vals):.3f}")


if __name__ == '__main__':
    main()
