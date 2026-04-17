"""Resume YOLOv8 fine-tuning for N more epochs, then plot combined loss curve.

Loads ``runs/detect/<prev_name>/weights/last.pt`` (default: yolov8m_vid) and
trains for ``--additional_epochs`` more, producing a new run directory with
the original's best weights available to copy forward. After training, calls
``plot_yolo_loss.py`` to generate a combined plot stitching both runs.

This does NOT modify ``finetune_yolo_vid.py``; it's a standalone companion
that uses Ultralytics' ability to continue training from an arbitrary
checkpoint.

Usage:
    python training/resume_yolo_vid.py `
        --vid_root ./data/imagenetvid `
        --prev_name yolov8m_vid `
        --additional_epochs 30 `
        --batch 16 --imgsz 640
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vid_root', default='./data/imagenetvid')
    ap.add_argument('--prev_name', default='yolov8m_vid',
                    help='Ultralytics run name to resume from (runs/detect/<name>)')
    ap.add_argument('--new_name', default=None,
                    help='Name for the new continuation run '
                         '(default: <prev_name>_resume)')
    ap.add_argument('--additional_epochs', type=int, default=30,
                    help='How many MORE epochs to train beyond the previous run')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--device', default='0')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--weights', default=None,
                    help="Override: explicit checkpoint path (default: "
                         "runs/detect/<prev_name>/weights/last.pt)")
    ap.add_argument('--skip_plot', action='store_true',
                    help='Skip the combined loss plot at the end')
    ap.add_argument('--optimizer', default='SGD',
                    help="Optimizer: 'SGD' (default, matches original run), "
                         "'AdamW', or 'auto' (uses Ultralytics default which "
                         "may be MuSGD — known to OOM on some systems)")
    ap.add_argument('--lr0', type=float, default=0.01,
                    help='Initial learning rate (SGD default 0.01)')
    ap.add_argument('--lrf', type=float, default=0.01,
                    help='Final LR fraction (cosine schedule end)')
    args = ap.parse_args()

    vid_root = Path(args.vid_root)
    data_yaml = vid_root / 'yolo_vid' / 'data.yaml'
    if not data_yaml.exists():
        print(f"ERROR: {data_yaml} not found. "
              "Run training/finetune_yolo_vid.py first (for label prep).")
        sys.exit(1)

    prev_run_dir = Path('runs/detect') / args.prev_name
    prev_results_csv = prev_run_dir / 'results.csv'
    weights_path = Path(args.weights) if args.weights else (prev_run_dir / 'weights' / 'last.pt')
    if not weights_path.exists():
        print(f"ERROR: checkpoint not found: {weights_path}")
        sys.exit(1)
    print(f"Resuming from: {weights_path}")

    new_name = args.new_name or f"{args.prev_name}_resume"
    print(f"New run dir:   runs/detect/{new_name}")
    print(f"Training for {args.additional_epochs} more epochs")

    from ultralytics import YOLO
    model = YOLO(str(weights_path))
    # Note: we do NOT use resume=True — that would re-use the original run's
    # args.yaml and be capped at its original `epochs` count. Instead we start
    # a fresh run initialized from last.pt, which lets us specify any number
    # of additional epochs.
    model.train(
        data=str(data_yaml),
        epochs=args.additional_epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        name=new_name,
        patience=10,
        amp=True,
        optimizer=args.optimizer,   # 'SGD' to avoid Muon bad-alloc crash
        lr0=args.lr0,
        lrf=args.lrf,
    )

    new_run_dir = Path('runs/detect') / new_name
    new_results_csv = new_run_dir / 'results.csv'
    print(f"\nFine-tune done. Weights:")
    print(f"  best: {new_run_dir / 'weights' / 'best.pt'}")
    print(f"  last: {new_run_dir / 'weights' / 'last.pt'}")

    if args.skip_plot:
        return

    # Combined plot — stitches prev + new results.csv
    if not prev_results_csv.exists():
        print(f"(note) {prev_results_csv} missing; plotting new run only")
        csv_args = [str(new_results_csv)]
    elif not new_results_csv.exists():
        print(f"(note) {new_results_csv} missing; plotting prev run only")
        csv_args = [str(prev_results_csv)]
    else:
        csv_args = [str(prev_results_csv), str(new_results_csv)]

    plot_out = new_run_dir / 'combined_loss_curve.png'
    cmd = [sys.executable, 'training/plot_yolo_loss.py',
           '--csv', *csv_args,
           '--output', str(plot_out),
           '--title', f'YOLOv8m on ImageNet VID ({args.prev_name} + {new_name})']
    print(f"\nGenerating combined loss plot:\n  {' '.join(cmd)}")
    subprocess.run(cmd, check=False)


if __name__ == '__main__':
    main()
