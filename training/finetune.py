import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cope_delta_det import CoPEDeltaDet
from training.losses import DetectionLoss
from tqdm import tqdm
from torchvision.ops import box_iou
from scipy.optimize import linear_sum_assignment

def build_optimizer_and_scheduler(model, lr=1e-4, warmup_steps=1000, total_steps=100000):
    """
    Builds AdamW and Cosine Annealing with Warmup scheduler.
    """
    # Exclude YOLO anchor from optimization (it's frozen)
    optimized_params = [p for n, p in model.named_parameters() if p.requires_grad and 'anchor_detector' not in n]
    
    optimizer = AdamW(optimized_params, lr=lr)
    
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    
    return optimizer, scheduler

def match_predictions_to_targets(pred_boxes, tgt_boxes):
    """
    Bipartite matching via Hungarian algorithm using IoU cost.
    Returns matched pred and tgt indices.
    """
    if len(pred_boxes) == 0 or len(tgt_boxes) == 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        
    # Calculate pairwise IoU. Shape: [num_preds, num_tgts]
    iou_matrix = box_iou(pred_boxes, tgt_boxes)
    
    # We want to MAXIMIZE IoU, so minimize 1 - IoU
    cost_matrix = 1.0 - iou_matrix.detach().cpu().numpy()
    
    # Scipy linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Filter out matches where IoU is basically zero (e.g. < 0.1)
    matched_rows = []
    matched_cols = []
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] > 0.1:
            matched_rows.append(r)
            matched_cols.append(c)
            
    return torch.tensor(matched_rows, dtype=torch.long), torch.tensor(matched_cols, dtype=torch.long)

def train_finetune_epoch(dataloader, model, optimizer, scheduler, loss_fn, device='cuda',
                         scaler=None, accum_steps=4, num_classes=10):
    """Train one epoch with mixed precision and gradient accumulation."""
    model.train()
    total_loss = 0.0
    num_batches_with_loss = 0
    use_amp = scaler is not None

    pbar = tqdm(dataloader, desc="Fine-tuning")
    optimizer.zero_grad()

    for step, batch in enumerate(pbar):
        # Unpack elements matching dataset.py return structure
        video_names, iframe_rgbs, pframe_rgbs, pframe_mvs, pframe_res, pframe_depths, pframe_modes, targets = batch

        # We process batches frame by frame. iframe_rgbs shape is [B, 3, H, W]
        iframe_rgb = torch.stack([x.to(device) for x in iframe_rgbs]) if isinstance(iframe_rgbs, (list, tuple)) else iframe_rgbs.to(device)

        # MVS/APP are lists of [num_pframes, C, H_p, W_p]
        B = len(video_names)
        num_pframes = pframe_mvs[0].shape[0] if B > 0 else 0

        formatted_mvs = []
        formatted_app = []

        for t in range(num_pframes):
            batch_mvs = []
            batch_app = []
            for b_idx in range(B):
                mvs_t = pframe_mvs[b_idx][t].permute(2, 0, 1).to(device)
                res_t = pframe_res[b_idx][t].permute(2, 0, 1).to(device)
                depths_t = pframe_depths[b_idx][t].permute(2, 0, 1).to(device).float()
                modes_t = pframe_modes[b_idx][t].permute(2, 0, 1).to(device).float()
                app_t = torch.cat([res_t, depths_t, modes_t], dim=0)
                batch_mvs.append(mvs_t)
                batch_app.append(app_t)

            formatted_mvs.append(torch.stack(batch_mvs))
            formatted_app.append(torch.stack(batch_app))

        # Forward pass with mixed precision
        with torch.amp.autocast('cuda', enabled=use_amp):
            predictions = model(iframe_rgb, formatted_mvs, formatted_app)

            batch_loss = 0.0

            for t in range(1, len(predictions)):
                pred_t = predictions[t]
                target_t = [targets[b_idx][t] for b_idx in range(len(targets))]

                pred_boxes = torch.cat(pred_t['boxes'], dim=0) if len(pred_t['boxes']) > 0 else torch.empty((0,4), device=device)
                pred_conf = torch.cat(pred_t['confs'], dim=0) if len(pred_t['confs']) > 0 else torch.empty((0,1), device=device)
                pred_cls = torch.cat(pred_t['classes'], dim=0) if len(pred_t['classes']) > 0 else torch.empty((0, num_classes), device=device)

                tgt_boxes = []
                tgt_cls_list = []
                for b_idx in range(len(target_t)):
                    if 'boxes' in target_t[b_idx]:
                        b_boxes = target_t[b_idx]['boxes'].to(device).clone()
                        if b_boxes.shape[0] > 0:
                            b_boxes[:, 2] = b_boxes[:, 0] + b_boxes[:, 2]
                            b_boxes[:, 3] = b_boxes[:, 1] + b_boxes[:, 3]
                        tgt_boxes.append(b_boxes)
                        tgt_cls_list.append(target_t[b_idx]['labels'].to(device))

                tgt_boxes = torch.cat(tgt_boxes, dim=0) if len(tgt_boxes) > 0 else torch.empty((0,4), device=device)
                tgt_cls = torch.cat(tgt_cls_list, dim=0) if len(tgt_cls_list) > 0 else torch.empty((0,), dtype=torch.long, device=device)

                pred_idx, tgt_idx = match_predictions_to_targets(pred_boxes, tgt_boxes)

                if len(pred_idx) > 0:
                    matched_pred_boxes = pred_boxes[pred_idx]
                    matched_pred_cls = pred_cls[pred_idx]
                    matched_tgt_boxes = tgt_boxes[tgt_idx]
                    matched_tgt_cls = tgt_cls[tgt_idx].long()

                    loss, _ = loss_fn(
                        matched_pred_boxes,
                        matched_pred_cls,
                        matched_tgt_boxes,
                        matched_tgt_cls
                    )
                    batch_loss += loss

        if isinstance(batch_loss, torch.Tensor) and batch_loss.requires_grad:
            scaled_loss = batch_loss / accum_steps
            if use_amp:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            total_loss += batch_loss.item()
            num_batches_with_loss += 1
            pbar.set_postfix({'loss': f'{batch_loss.item():.1f}'})

    return total_loss / max(num_batches_with_loss, 1)

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from data.dataset import BDD100KCoPEDataset, collate_fn
    import os
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Stage 2: Fine-tuning CoPE-Delta-Det')
    parser.add_argument('--features', type=str, default='features',
                        choices=['features', 'features_pyav'],
                        help="Feature subdir: 'features' (block-matched) or 'features_pyav' (H.264 proxy)")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--prefix', type=str, default='',
                        help="Checkpoint filename prefix, e.g. 'pyav_'")
    parser.add_argument('--dataset', type=str, default='bdd100k',
                        choices=['bdd100k', 'imagenetvid'])
    parser.add_argument('--root', type=str, default=None,
                        help='Dataset root dir (defaults based on --dataset)')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='Class count (defaults: bdd100k=10, imagenetvid=30)')
    parser.add_argument('--yolo_weights', type=str, default='yolov8m.pt',
                        help='YOLO checkpoint (use fine-tuned best.pt for VID)')
    parser.add_argument('--class_mapping', type=str, default=None,
                        help="'coco_to_bdd', 'identity', or omit for dataset default")
    parser.add_argument('--annotated_only', action='store_true',
                        help='Filter to annotated GOPs only (BDD legacy)')
    parser.add_argument('--resume', type=str, default=None,
                        help="Path to a {prefix}finetuned_model_epoch_N.pt to resume from; "
                             "training continues from epoch N+1")
    parser.add_argument('--num_workers', type=int, default=8,
                        help="Dataloader workers (higher = more I/O overlap)")
    parser.add_argument('--gop_length', type=int, default=16)
    args = parser.parse_args()

    # Dataset-specific defaults
    if args.root is None:
        args.root = ('D:/cope-delta-det2/data/bdd100k' if args.dataset == 'bdd100k'
                     else 'D:/cope-delta-det2/data/imagenetvid')
    if args.num_classes is None:
        args.num_classes = 10 if args.dataset == 'bdd100k' else 30
    if args.class_mapping is None:
        args.class_mapping = 'coco_to_bdd' if args.dataset == 'bdd100k' else 'identity'
    # For VID we want all GOPs (every frame has real GT); for BDD legacy, annotated_only.
    annotated_only_flag = args.annotated_only or (args.dataset == 'bdd100k')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Executing Stage 2: Fine-tuning Sequence on {device}...")
    print(f"  Dataset: {args.dataset}  root={args.root}  num_classes={args.num_classes}")
    print(f"  Features: {args.features}")
    print(f"  YOLO weights: {args.yolo_weights}  class_mapping={args.class_mapping}")

    # 1. Dataset
    dataset = BDD100KCoPEDataset(root_dir=args.root, split='train',
                                  gop_length=args.gop_length,
                                  annotated_only=annotated_only_flag,
                                  features_subdir=args.features,
                                  dataset_type=args.dataset)
    # batch_size=1 (GOP-level), but we use gradient accumulation (accum_steps=4) for effective batch=4
    # num_workers high to overlap JPEG decode + NPZ reads with GPU compute
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=True, prefetch_factor=4)

    # 2. Model
    model = CoPEDeltaDet(yolo_size=args.yolo_weights, embed_dim=256,
                         num_classes=args.num_classes, device=device).to(device)
    # Propagate class_mapping to the YOLO anchor inside the model, if supported
    try:
        if hasattr(model.anchor_detector, 'class_mapping'):
            if args.class_mapping in (None, 'identity'):
                model.anchor_detector.class_mapping = None
            elif args.class_mapping == 'coco_to_bdd':
                model.anchor_detector.class_mapping = {0: 0, 1: 7, 2: 2, 3: 6, 5: 4, 6: 5, 7: 3, 9: 8, 11: 9}
    except Exception as e:
        print(f"  (note) could not set class_mapping on anchor: {e}")

    # Attempt to load Pretrained Weights from Stage 1 if they exist
    stage1_weights = "D:/cope-delta-det2/checkpoints/delta_encoder_epoch_1.pt"
    if os.path.exists(stage1_weights):
        print(f"Loading Stage 1 pretrained weights from {stage1_weights}")
        model.delta_encoder.load_state_dict(torch.load(stage1_weights, map_location=device))
    else:
        print("WARNING: Stage 1 weights not found. Fine-tuning from scratch!")

    # 3. Optimizers & Loss
    num_epochs = args.epochs
    opt, sch = build_optimizer_and_scheduler(model, lr=1e-4, warmup_steps=500, total_steps=num_epochs * len(dataloader))
    loss_fn = DetectionLoss(lambda_box=5.0, lambda_giou=2.0, lambda_cls=2.0).to(device)

    os.makedirs('D:/cope-delta-det2/checkpoints', exist_ok=True)

    # Mixed precision for faster GPU compute (FP16 forward, FP32 backward)
    scaler = torch.amp.GradScaler('cuda')
    accum_steps = 4  # Gradient accumulation for effective batch size of 4

    prefix = args.prefix
    best_loss = float('inf')
    start_epoch = 0

    # Resume from a previous checkpoint if requested
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        if 'delta_encoder' in ckpt:
            model.delta_encoder.load_state_dict(ckpt['delta_encoder'])
        if 'fusion_head' in ckpt:
            model.fusion_head.load_state_dict(ckpt['fusion_head'])
        start_epoch = int(ckpt.get('epoch', 0))
        prev_loss = float(ckpt.get('loss', float('inf')))
        best_loss = min(best_loss, prev_loss)
        # Fast-forward LR scheduler to the correct step count
        # (number of optimizer steps that already happened = start_epoch * (len(dataloader) // accum_steps))
        opt_steps_done = start_epoch * max(1, len(dataloader) // accum_steps)
        for _ in range(opt_steps_done):
            sch.step()
        print(f"  Resumed at epoch {start_epoch}, prev_loss={prev_loss:.4f}, "
              f"scheduler fast-forwarded by {opt_steps_done} steps")

    print(f"Starting Fine-tuning: {num_epochs} epochs, {len(dataloader)} steps/epoch")
    print(f"  Mixed precision: enabled, Gradient accumulation: {accum_steps} steps")
    print(f"  DataLoader workers: {args.num_workers}, pin_memory: True")
    print(f"  Checkpoint prefix: '{prefix}'")
    print(f"  Starting at epoch {start_epoch+1}")

    # ── Loss history ─────────────────────────────────────────────────────
    # Records (epoch, loss) so we can plot convergence / decide if more
    # epochs are worthwhile. JSON history is resumable across runs.
    import json
    history_path = f"D:/cope-delta-det2/checkpoints/{prefix}loss_history.json"
    loss_history = []
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                loss_history = json.load(f)
            # Drop entries >= start_epoch to avoid duplicates on resume
            loss_history = [h for h in loss_history if h['epoch'] <= start_epoch]
            print(f"  Loaded {len(loss_history)} prior loss records from {history_path}")
        except Exception as e:
            print(f"  (note) could not load prior history: {e}")
            loss_history = []

    for epoch in range(start_epoch, num_epochs):
        loss = train_finetune_epoch(dataloader, model, opt, sch, loss_fn, device=device,
                                     scaler=scaler, accum_steps=accum_steps,
                                     num_classes=args.num_classes)
        print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {loss:.4f}")

        # Record loss history + persist to disk after every epoch so a crash
        # doesn't lose data
        loss_history.append({'epoch': epoch + 1, 'loss': float(loss)})
        with open(history_path, 'w') as f:
            json.dump(loss_history, f, indent=2)

        # Save only trainable components (not frozen YOLO which fuses BN at runtime)
        ckpt = {
            'delta_encoder': model.delta_encoder.state_dict(),
            'fusion_head': model.fusion_head.state_dict(),
            'epoch': epoch + 1,
            'loss': loss,
        }
        torch.save(ckpt, f"D:/cope-delta-det2/checkpoints/{prefix}finetuned_model_epoch_{epoch+1}.pt")

        if loss < best_loss:
            best_loss = loss
            torch.save(ckpt, f"D:/cope-delta-det2/checkpoints/{prefix}finetuned_best.pt")
            print(f"  -> New best model (loss={loss:.4f})")

    print(f"Fine-tuning complete. Best loss: {best_loss:.4f}")

    # ── Plot loss curve ──────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')  # headless backend
        import matplotlib.pyplot as plt

        epochs_plot = [h['epoch'] for h in loss_history]
        losses_plot = [h['loss'] for h in loss_history]

        plt.figure(figsize=(10, 6))
        plt.plot(epochs_plot, losses_plot, 'b-o', linewidth=2, markersize=6, label='Train loss')

        # Mark the best epoch
        if losses_plot:
            best_idx = losses_plot.index(min(losses_plot))
            plt.axhline(y=losses_plot[best_idx], color='g', linestyle='--', alpha=0.5,
                        label=f'Best (epoch {epochs_plot[best_idx]}): {losses_plot[best_idx]:.4f}')

        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title(f"CoPE-Delta-Det Fine-tuning Loss ({args.dataset}, {args.num_classes} classes)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plot_path = f"D:/cope-delta-det2/checkpoints/{prefix}loss_curve.png"
        plt.savefig(plot_path, dpi=120)
        plt.close()
        print(f"Loss curve saved to {plot_path}")
        print(f"Loss history JSON saved to {history_path}")
    except ImportError:
        print("(note) matplotlib not installed; skipping loss plot")
    except Exception as e:
        print(f"(note) plot failed: {e}")
