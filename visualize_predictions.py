"""Visualize ground-truth vs model predictions on random GOPs from the dataset.

For each sampled GOP, produces a 2x2 quadrant video where each quadrant shows
the same RGB frame but with boxes from a different source:

    +---------------------+---------------------+
    | Ground Truth (GT)   | YOLO-full           |
    +---------------------+---------------------+
    | Copy-paste          | CoPE-Delta-Det      |
    +---------------------+---------------------+

Writes an AVI video at `visualize_predictions_{split}.avi`.

Usage:
    python visualize_predictions.py `
        --dataset imagenetvid `
        --root D:/cope-delta-det2/data/imagenetvid `
        --num_classes 30 `
        --yolo_weights runs/detect/yolov8m_vid/weights/best.pt `
        --class_mapping identity `
        --split val `
        --checkpoint D:/cope-delta-det2/checkpoints/vid_finetuned_model_epoch_3.pt `
        --num_gops 8 `
        --random
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.dataset import BDD100KCoPEDataset
from models.cope_delta_det import CoPEDeltaDet
from utils.box_utils import xyxy_to_xywh, xywh_to_xyxy


IMAGENET_VID_CLASSES = [
    'airplane', 'antelope', 'bear', 'bicycle', 'bird',
    'bus', 'car', 'cattle', 'dog', 'domestic_cat',
    'elephant', 'fox', 'giant_panda', 'hamster', 'horse',
    'lion', 'lizard', 'monkey', 'motorcycle', 'rabbit',
    'red_panda', 'sheep', 'snake', 'squirrel', 'tiger',
    'train', 'turtle', 'watercraft', 'whale', 'zebra',
]

BDD_CLASSES = ['person', 'rider', 'car', 'truck', 'bus',
               'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign']

# BGR colours (consistent across the four quadrants so it's easy to compare)
COLOR_GT         = (0, 255,   0)   # green
COLOR_YOLO_FULL  = (255, 128, 0)   # blue-ish
COLOR_COPY_PASTE = (0, 200, 255)   # yellow
COLOR_COPE       = (0, 0, 255)     # red


def get_class_names(dataset_name: str):
    return IMAGENET_VID_CLASSES if dataset_name == 'imagenetvid' else BDD_CLASSES


def tensor_frame_to_bgr(rgb_t: torch.Tensor) -> np.ndarray:
    """[3,H,W] float in [0,1] -> BGR uint8."""
    rgb = (rgb_t.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    rgb = np.transpose(rgb, (1, 2, 0))          # [H,W,3]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def draw_boxes(img: np.ndarray,
               boxes_xyxy: np.ndarray,
               labels: np.ndarray,
               scores: np.ndarray,
               class_names,
               color,
               show_score: bool = True) -> None:
    """Draw boxes [x1,y1,x2,y2] with class label and optional score."""
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return
    for i in range(len(boxes_xyxy)):
        x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
        x1 = max(0, min(x1, img.shape[1] - 1))
        y1 = max(0, min(y1, img.shape[0] - 1))
        x2 = max(0, min(x2, img.shape[1] - 1))
        y2 = max(0, min(y2, img.shape[0] - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        cid = int(labels[i]) if labels is not None and i < len(labels) else -1
        name = class_names[cid] if 0 <= cid < len(class_names) else f'cls{cid}'
        if show_score and scores is not None and i < len(scores):
            label = f"{name} {float(scores[i]):.2f}"
        else:
            label = name

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th - 5)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def add_title(tile: np.ndarray, title: str, color) -> np.ndarray:
    """Add a coloured header bar with the panel name."""
    bar_h = 28
    bar = np.zeros((bar_h, tile.shape[1], 3), dtype=np.uint8)
    bar[:] = (30, 30, 30)
    cv2.rectangle(bar, (0, 0), (6, bar_h), color, -1)
    cv2.putText(bar, title, (12, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return np.vstack([bar, tile])


def run_yolo_per_frame(model, rgb_bchw: torch.Tensor):
    """Return (boxes_xyxy [N,4], conf [N], cls [N]) for a single image."""
    anchor_results = model.anchor_detector.get_anchor_boxes(rgb_bchw)
    r = anchor_results[0]
    if r.shape[0] == 0:
        return (np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int64))
    boxes = r[:, :4].cpu().numpy()
    conf = r[:, 4].cpu().numpy()
    cls = r[:, 5].cpu().numpy().astype(np.int64)
    return boxes, conf, cls


def run_cope_pframe(model, current_boxes_xyxy: torch.Tensor,
                    current_confs: torch.Tensor,
                    mvs_thwc_frame: torch.Tensor,
                    res_thwc_frame: torch.Tensor,
                    depths_thwc_frame: torch.Tensor,
                    modes_thwc_frame: torch.Tensor,
                    device):
    """One CoPE P-frame update. Returns updated (boxes_xyxy, confs, cls_scores)."""
    if current_boxes_xyxy.shape[0] == 0:
        return current_boxes_xyxy, current_confs, None

    mvs_t = mvs_thwc_frame.permute(2, 0, 1).unsqueeze(0).to(device)
    r_feat = res_thwc_frame.permute(2, 0, 1).to(device)
    d_feat = depths_thwc_frame.permute(2, 0, 1).float().to(device)
    m_feat = modes_thwc_frame.permute(2, 0, 1).float().to(device)
    app_t = torch.cat([r_feat, d_feat, m_feat], dim=0).unsqueeze(0)

    b_ids = torch.zeros(current_boxes_xyxy.shape[0], dtype=torch.long, device=device)
    boxes_xywh = xyxy_to_xywh(current_boxes_xyxy)
    flat_anchors = torch.cat([boxes_xywh, current_confs], dim=1)

    delta_tokens = model.delta_encoder(mvs_t, app_t, current_boxes_xyxy, b_ids)
    box_deltas, conf_updates, cls_scores = model.fusion_head(flat_anchors, delta_tokens)

    updated_xyxy = xywh_to_xyxy(boxes_xywh + box_deltas)
    updated_confs = current_confs * conf_updates
    return updated_xyxy, updated_confs, cls_scores


def build_video_groups(dataset):
    """Group GOP indices by source video and keep them in temporal order."""
    groups = defaultdict(list)
    for idx, gop in enumerate(dataset.gops):
        groups[gop['video_name']].append(idx)
    for video_name in groups:
        groups[video_name].sort(key=lambda i: dataset.gops[i]['start_frame'])
    return groups


def estimate_video_motion(dataset, gop_indices, max_gops_to_scan=4):
    """Estimate how dynamic a video is using mean MV magnitude over a few GOPs."""
    scores = []
    for idx in gop_indices[:max_gops_to_scan]:
        gop_meta = dataset.gops[idx]
        mvs_t, _, _, _, _ = dataset.load_frame_features(
            gop_meta['video_name'],
            gop_meta['start_frame'],
            gop_meta.get('num_frames', dataset.gop_length),
        )
        if mvs_t.numel() == 0:
            continue
        mv_mag = torch.linalg.vector_norm(mvs_t.float(), dim=-1).mean().item()
        scores.append(mv_mag)
    return float(np.mean(scores)) if scores else 0.0


def select_gop_indices(dataset, args):
    """Return GOP indices and a short description of the selection policy."""
    video_groups = build_video_groups(dataset)
    all_video_names = sorted(video_groups.keys())

    if args.video_name:
        matched_names = [v for v in all_video_names if args.video_name.lower() in v.lower()]
        if not matched_names:
            raise ValueError(f"No video_name matched '{args.video_name}'")
        selected_videos = matched_names[:args.num_videos]
        description = f"video filter '{args.video_name}'"
    elif args.sort_videos_by_motion:
        scored_videos = []
        for video_name in all_video_names:
            score = estimate_video_motion(dataset, video_groups[video_name], max_gops_to_scan=args.motion_scan_gops)
            if score >= args.min_motion_score:
                scored_videos.append((score, video_name))
        scored_videos.sort(reverse=True)
        selected_videos = [video_name for _, video_name in scored_videos[:args.num_videos]]
        description = f"top-{args.num_videos} motion-ranked videos"
    else:
        selected_videos = all_video_names[:]
        if args.random:
            random.shuffle(selected_videos)
        selected_videos = selected_videos[:args.num_videos]
        description = "random videos" if args.random else "first videos"

    if args.sort_videos_by_motion:
        print("\nSelected videos by motion:")
        for rank, video_name in enumerate(selected_videos, start=1):
            score = estimate_video_motion(dataset, video_groups[video_name], max_gops_to_scan=args.motion_scan_gops)
            print(f"  {rank:2d}. {video_name}  motion_score={score:.3f}")

    selected_indices = []
    for video_name in selected_videos:
        indices = video_groups[video_name]
        if args.random and not args.video_name and not args.sort_videos_by_motion:
            start_offset = random.randint(0, max(0, len(indices) - args.gops_per_video))
            indices = indices[start_offset:]
        selected_indices.extend(indices[:args.gops_per_video])

    return selected_indices, description


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='imagenetvid',
                    choices=['bdd100k', 'imagenetvid'])
    ap.add_argument('--root', type=str, default=None)
    ap.add_argument('--split', type=str, default=None)
    ap.add_argument('--num_classes', type=int, default=None)
    ap.add_argument('--yolo_weights', type=str, default='yolov8m.pt')
    ap.add_argument('--class_mapping', type=str, default=None)
    ap.add_argument('--features', type=str, default='features',
                    choices=['features', 'features_pyav'])
    ap.add_argument('--checkpoint', type=str, default=None)
    ap.add_argument('--gop_length', type=int, default=16)
    ap.add_argument('--num_gops', type=int, default=8)
    ap.add_argument('--random', action='store_true', help='Random GOPs rather than first N')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num_videos', type=int, default=1,
                    help='Number of source videos to visualize when using consecutive GOP mode')
    ap.add_argument('--gops_per_video', type=int, default=4,
                    help='How many consecutive GOPs to render per selected video')
    ap.add_argument('--video_name', type=str, default=None,
                    help='Substring match for a specific video_name to visualize')
    ap.add_argument('--sort_videos_by_motion', action='store_true',
                    help='Rank videos by estimated MV magnitude and pick the most dynamic ones')
    ap.add_argument('--motion_scan_gops', type=int, default=4,
                    help='How many GOPs per video to scan when estimating motion')
    ap.add_argument('--min_motion_score', type=float, default=0.0,
                    help='Minimum motion score when using --sort_videos_by_motion')
    ap.add_argument('--min_gt_boxes', type=int, default=1,
                    help='Skip GOPs whose I-frame has fewer than this many GT boxes')
    ap.add_argument('--output', type=str, default=None)
    ap.add_argument('--fps', type=float, default=6.0)
    ap.add_argument('--conf_threshold', type=float, default=0.25,
                    help='Hide predicted boxes below this score')
    args = ap.parse_args()

    # Dataset-specific defaults
    if args.root is None:
        args.root = ('D:/cope-delta-det2/data/bdd100k' if args.dataset == 'bdd100k'
                     else 'D:/cope-delta-det2/data/imagenetvid')
    if args.num_classes is None:
        args.num_classes = 10 if args.dataset == 'bdd100k' else 30
    if args.class_mapping is None:
        args.class_mapping = 'coco_to_bdd' if args.dataset == 'bdd100k' else 'identity'
    if args.split is None:
        args.split = 'train' if args.dataset == 'bdd100k' else 'val'
    if args.checkpoint is None:
        args.checkpoint = ('D:/cope-delta-det2/checkpoints/finetuned_best.pt'
                           if args.dataset == 'bdd100k'
                           else 'D:/cope-delta-det2/checkpoints/vid_finetuned_best.pt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_names = get_class_names(args.dataset)

    print(f"Loading dataset: {args.dataset} split={args.split} root={args.root}")
    annotated_only = args.dataset == 'bdd100k'
    dataset = BDD100KCoPEDataset(root_dir=args.root, split=args.split,
                                 gop_length=args.gop_length,
                                 annotated_only=annotated_only,
                                 features_subdir=args.features,
                                 dataset_type=args.dataset)
    print(f"Dataset size: {len(dataset)}")

    print(f"Loading model: yolo={args.yolo_weights}  cope={args.checkpoint}")
    model = CoPEDeltaDet(yolo_size=args.yolo_weights, embed_dim=256,
                         num_classes=args.num_classes, device=device).to(device)
    if hasattr(model.anchor_detector, 'class_mapping'):
        if args.class_mapping in (None, 'identity'):
            model.anchor_detector.class_mapping = None
        elif args.class_mapping == 'coco_to_bdd':
            model.anchor_detector.class_mapping = {
                0: 0, 1: 7, 2: 2, 3: 6, 5: 4, 6: 5, 7: 3, 9: 8, 11: 9}
        elif args.class_mapping == 'coco_to_vid':
            model.anchor_detector.class_mapping = {
                4: 0, 21: 2, 1: 3, 14: 4, 5: 5, 2: 6, 7: 6, 19: 7,
                16: 8, 15: 9, 20: 10, 17: 14, 3: 18, 18: 21, 6: 25,
                8: 27, 22: 29,
            }

    if Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        if 'delta_encoder' in ckpt:
            model.delta_encoder.load_state_dict(ckpt['delta_encoder'])
            model.fusion_head.load_state_dict(ckpt['fusion_head'])
            print(f"  loaded CoPE weights (epoch {ckpt.get('epoch', '?')})")
        else:
            model.load_state_dict(ckpt, strict=False)
    else:
        print(f"  WARNING: no checkpoint at {args.checkpoint}; running with untrained CoPE")

    model.eval()

    # Pick GOP indices
    random.seed(args.seed)
    if args.video_name or args.sort_videos_by_motion or args.gops_per_video != 1 or args.num_videos != 1:
        indices, selection_desc = select_gop_indices(dataset, args)
        print(f"Selection mode: {selection_desc}")
    else:
        indices = list(range(len(dataset)))
        if args.random:
            random.shuffle(indices)
        indices = indices[:args.num_gops]
        selection_desc = "random GOPs" if args.random else "first GOPs"
        print(f"Selection mode: {selection_desc}")

    # Video writer (2x2 tiles of 640x360, plus header bars => 640x (360+28) per tile,
    # full canvas = 2*640 x 2*388 = 1280 x 776)
    tile_w, tile_h = 640, 360
    canvas_w = 2 * tile_w
    canvas_h = 2 * (tile_h + 28)
    out_path = args.output or f"visualize_predictions_{args.dataset}_{args.split}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video = cv2.VideoWriter(out_path, fourcc, args.fps, (canvas_w, canvas_h))

    gops_used = 0
    total_frames = 0
    target_gops = len(indices) if indices else args.num_gops
    pbar = tqdm(total=target_gops, desc=f"Visualising ({args.dataset}/{args.split})")

    with torch.no_grad():
        for idx in indices:
            try:
                sample = dataset[idx]
            except Exception as e:
                continue

            # Skip GOPs whose I-frame RGB is all-black (unavailable video)
            if sample['iframe_rgb'].sum().item() == 0:
                continue

            targets = sample['targets']
            if len(targets) == 0:
                continue
            if targets[0]['boxes'].shape[0] < args.min_gt_boxes:
                continue

            iframe_rgb = sample['iframe_rgb'].unsqueeze(0).to(device)     # [1,3,720,1280]
            pframe_rgbs = sample['pframe_rgbs']                            # [N,3,720,1280]
            mvs = sample['pframe_mvs']                                     # [N,45,80,2]
            res_feat = sample['pframe_res']
            depths_feat = sample['pframe_depths']
            modes_feat = sample['pframe_modes']
            num_pframes = int(mvs.shape[0])

            video_name = sample.get('video_name', f'gop{idx}')

            # --- Frame 0 predictions (shared initial I-frame detection) ---
            i_boxes, i_conf, i_cls = run_yolo_per_frame(model, iframe_rgb)

            # State for each tracker
            yolo_boxes_t, yolo_conf_t, yolo_cls_t = i_boxes, i_conf, i_cls
            cp_boxes,   cp_conf,   cp_cls   = i_boxes, i_conf, i_cls

            # CoPE state as tensors (for the model forward)
            if i_boxes.shape[0] > 0:
                cope_boxes_xyxy = torch.from_numpy(i_boxes).float().to(device)
                cope_confs      = torch.from_numpy(i_conf).float().unsqueeze(1).to(device)
                cope_cls_ids    = i_cls.copy()  # will be replaced by argmax on P-frames
            else:
                cope_boxes_xyxy = torch.empty((0, 4), device=device)
                cope_confs      = torch.empty((0, 1), device=device)
                cope_cls_ids    = np.zeros(0, dtype=np.int64)

            # Build frame 0 tiles
            def build_frame_tiles(frame_bgr, gt_target,
                                  yolo_b, yolo_s, yolo_c,
                                  cp_b, cp_s, cp_c,
                                  cope_b, cope_s, cope_c,
                                  frame_header):
                def _mk(panel_title, color, boxes, scores, cls):
                    tile = cv2.resize(frame_bgr, (tile_w, tile_h))
                    # Boxes are in 1280x720 space; we must scale to 640x360
                    if boxes is not None and len(boxes) > 0:
                        sx = tile_w / 1280.0
                        sy = tile_h / 720.0
                        scaled = boxes.copy().astype(np.float32)
                        scaled[:, [0, 2]] *= sx
                        scaled[:, [1, 3]] *= sy
                    else:
                        scaled = np.zeros((0, 4))
                    # Filter by confidence (not for GT)
                    if scores is not None and panel_title != 'Ground Truth':
                        keep = scores >= args.conf_threshold
                        scaled = scaled[keep]
                        scores = scores[keep]
                        if cls is not None:
                            cls = cls[keep]
                    draw_boxes(tile, scaled, cls, scores, class_names, color,
                               show_score=(panel_title != 'Ground Truth'))
                    return add_title(tile, panel_title, color)

                # GT boxes are [x,y,w,h] -> convert to xyxy
                gt_np = gt_target['boxes'].cpu().numpy().copy() if gt_target['boxes'].shape[0] > 0 else np.zeros((0, 4))
                gt_labels = gt_target['labels'].cpu().numpy() if gt_target['labels'].shape[0] > 0 else np.zeros(0, dtype=np.int64)
                if gt_np.shape[0] > 0:
                    gt_np[:, 2] = gt_np[:, 0] + gt_np[:, 2]
                    gt_np[:, 3] = gt_np[:, 1] + gt_np[:, 3]

                tl = _mk('Ground Truth', COLOR_GT,  gt_np,  None,    gt_labels)
                tr = _mk('YOLO-full',    COLOR_YOLO_FULL,  yolo_b, yolo_s, yolo_c)
                bl = _mk('Copy-paste',   COLOR_COPY_PASTE, cp_b,   cp_s,   cp_c)
                br = _mk('CoPE-Delta',   COLOR_COPE,       cope_b, cope_s, cope_c)

                top = np.hstack([tl, tr])
                bot = np.hstack([bl, br])
                canvas = np.vstack([top, bot])

                # Overlay a thin bottom bar with the frame info
                h, w = canvas.shape[:2]
                bar_h = 22
                bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
                cv2.putText(bar, frame_header, (8, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)
                return np.vstack([canvas, bar])[:canvas_h]  # clip to output size

            # --- Emit I-frame tile ---
            iframe_bgr = tensor_frame_to_bgr(sample['iframe_rgb'])
            header = f"GOP {gops_used+1}/{target_gops}  {video_name}  frame 0/{num_pframes} (I)"
            tile = build_frame_tiles(
                iframe_bgr, targets[0],
                yolo_boxes_t, yolo_conf_t, yolo_cls_t,
                cp_boxes,     cp_conf,     cp_cls,
                i_boxes,      i_conf,      i_cls,     # CoPE at t=0 == YOLO
                header)
            out_video.write(tile)
            total_frames += 1

            # --- P-frames ---
            for t in range(num_pframes):
                pframe_bgr = tensor_frame_to_bgr(pframe_rgbs[t])
                rgb_batched = pframe_rgbs[t].unsqueeze(0).to(device)

                # yolo_full: detect on every frame
                yolo_boxes_t, yolo_conf_t, yolo_cls_t = run_yolo_per_frame(model, rgb_batched)

                # copy_paste: carry frame 0 predictions unchanged — cp_boxes/conf/cls already set

                # CoPE-Delta
                cope_boxes_xyxy, cope_confs, cls_scores = run_cope_pframe(
                    model, cope_boxes_xyxy, cope_confs,
                    mvs[t], res_feat[t], depths_feat[t], modes_feat[t], device)
                if cls_scores is not None and cls_scores.shape[0] > 0:
                    cope_cls_ids = cls_scores.argmax(dim=1).cpu().numpy().astype(np.int64)

                cope_b_np = cope_boxes_xyxy.cpu().numpy() if cope_boxes_xyxy.shape[0] > 0 else np.zeros((0, 4))
                cope_s_np = cope_confs.squeeze(-1).cpu().numpy() if cope_confs.shape[0] > 0 else np.zeros(0)

                # target at t+1 (targets indexing: frame 0 = I-frame)
                gt_idx = t + 1
                gt_target = targets[gt_idx] if gt_idx < len(targets) else {
                    'boxes': torch.empty((0, 4)), 'labels': torch.empty((0,), dtype=torch.long)}

                header = (f"GOP {gops_used+1}/{target_gops}  {video_name}  "
                          f"frame {gt_idx}/{num_pframes} (P)")
                tile = build_frame_tiles(
                    pframe_bgr, gt_target,
                    yolo_boxes_t, yolo_conf_t, yolo_cls_t,
                    cp_boxes,     cp_conf,     cp_cls,
                    cope_b_np,    cope_s_np,   cope_cls_ids,
                    header)
                out_video.write(tile)
                total_frames += 1

            # Black separator
            sep = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            cv2.putText(sep, f"--- end of GOP {gops_used+1} ---",
                        (canvas_w // 2 - 180, canvas_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            for _ in range(4):
                out_video.write(sep)
                total_frames += 1

            gops_used += 1
            pbar.update(1)

    pbar.close()
    out_video.release()
    print(f"\nSaved {total_frames} frames across {gops_used} GOPs -> {out_path}")
    print(f"Playback at {args.fps} fps -> ~{total_frames / args.fps:.0f}s video")
    print("\nLegend:")
    print("  GREEN  = Ground truth")
    print("  BLUE   = YOLO-full (detects every frame)")
    print("  YELLOW = Copy-paste (I-frame boxes frozen)")
    print("  RED    = CoPE-Delta-Det (tracked boxes)")


if __name__ == '__main__':
    main()
