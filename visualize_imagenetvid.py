"""Visualize ImageNet VID GOPs with per-frame bounding box annotations.

Reads the GOP index JSON + source JPEGs directly, draws boxes with class
labels, and writes an AVI video for visual QA. Similar to visualize_dataset.py
but adapted for the VID per-frame GT format.

Usage:
    python visualize_imagenetvid.py --num_gops 10 --split train
    python visualize_imagenetvid.py --num_gops 5 --split val
"""

import cv2
import json
import numpy as np
from pathlib import Path
import argparse
import random

IMAGENET_VID_CLASSES = [
    'airplane', 'antelope', 'bear', 'bicycle', 'bird',
    'bus', 'car', 'cattle', 'dog', 'domestic_cat',
    'elephant', 'fox', 'giant_panda', 'hamster', 'horse',
    'lion', 'lizard', 'monkey', 'motorcycle', 'rabbit',
    'red_panda', 'sheep', 'snake', 'squirrel', 'tiger',
    'train', 'turtle', 'watercraft', 'whale', 'zebra',
]

# 30 distinct BGR colors for the 30 classes
CLASS_COLORS = [
    (0, 255, 0),    (255, 255, 0),  (0, 0, 255),    (0, 165, 255),
    (255, 0, 255),  (128, 128, 0),  (0, 255, 255),   (255, 0, 0),
    (0, 128, 255),  (128, 0, 255),  (50, 205, 50),   (255, 140, 0),
    (220, 20, 60),  (0, 191, 255),  (255, 215, 0),   (147, 20, 255),
    (34, 139, 34),  (70, 130, 180), (255, 99, 71),   (144, 238, 144),
    (218, 112, 214),(255, 182, 193),(107, 142, 35),  (72, 61, 139),
    (0, 206, 209),  (210, 105, 30), (127, 255, 212), (255, 69, 0),
    (173, 216, 230),(189, 183, 107),
]


def main():
    parser = argparse.ArgumentParser(description='Visualize ImageNet VID annotations')
    parser.add_argument('--vid_root', type=str, default='D:/cope-delta-det2/data/imagenetvid')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'])
    parser.add_argument('--num_gops', type=int, default=10)
    parser.add_argument('--random', action='store_true', help='Pick random GOPs instead of first N')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video path (default: vid_annotation_check_{split}.avi)')
    parser.add_argument('--fps', type=float, default=10.0)
    args = parser.parse_args()

    vid_root = Path(args.vid_root)
    jpeg_root = vid_root / 'ILSVRC2015' / 'Data' / 'VID'
    index_path = vid_root / f'gop_index_{args.split}.json'

    if not index_path.exists():
        print(f"ERROR: GOP index not found: {index_path}")
        return

    print(f"Loading GOP index: {index_path}")
    with open(index_path, 'r') as f:
        all_gops = json.load(f)

    # Filter to GOPs that actually have annotations
    annotated_gops = [g for g in all_gops if g.get('has_annotation', False)]
    print(f"Total GOPs: {len(all_gops)}, Annotated: {len(annotated_gops)}")

    if args.random:
        selected = random.sample(annotated_gops, min(args.num_gops, len(annotated_gops)))
    else:
        selected = annotated_gops[:args.num_gops]

    out_path = args.output or f"vid_annotation_check_{args.split}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video = cv2.VideoWriter(out_path, fourcc, args.fps, (1280, 720))
    total_frames = 0

    for gop_idx, gop in enumerate(selected):
        video_name = gop['video_name']
        frame_names = gop.get('frame_names', [])
        annotations = gop.get('annotations', [])
        num_frames = gop.get('num_frames', len(frame_names))

        n_boxes_total = sum(len(a) for a in annotations)
        print(f"  GOP {gop_idx+1}/{len(selected)}: {video_name} | "
              f"{num_frames} frames | {n_boxes_total} total boxes")

        for t in range(num_frames):
            # Load JPEG
            if t < len(frame_names):
                jpg_path = jpeg_root / frame_names[t]
            else:
                jpg_path = None

            if jpg_path and jpg_path.exists():
                frame_bgr = cv2.imread(str(jpg_path))
                if frame_bgr is None:
                    frame_bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
                elif frame_bgr.shape[:2] != (720, 1280):
                    frame_bgr = cv2.resize(frame_bgr, (1280, 720))
            else:
                frame_bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(frame_bgr, f"MISSING: {frame_names[t] if t < len(frame_names) else '?'}",
                            (100, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Draw annotations
            anns_t = annotations[t] if t < len(annotations) else []
            for ann in anns_t:
                bbox = ann['bbox']  # [x, y, w, h] COCO format
                cat_id = ann['category_id']  # 0-29
                track_id = ann.get('track_id', -1)

                x, y, w, h = bbox
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)

                color = CLASS_COLORS[cat_id % len(CLASS_COLORS)]
                class_name = IMAGENET_VID_CLASSES[cat_id] if cat_id < len(IMAGENET_VID_CLASSES) else f'cls{cat_id}'

                # Draw box
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

                # Draw label background + text
                label = f"{class_name}"
                if track_id >= 0:
                    label += f" #{track_id}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame_bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(frame_bgr, label, (x1 + 2, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Info overlay at top
            frame_type = "I-frame" if t == 0 else f"P-frame {t}"
            info = f"GOP {gop_idx+1} | {video_name} | Frame {t}/{num_frames} ({frame_type}) | {len(anns_t)} objs"
            cv2.putText(frame_bgr, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            out_video.write(frame_bgr)
            total_frames += 1

        # Black separator between GOPs
        black = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(black, "--- Next GOP ---", (480, 360),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        for _ in range(5):
            out_video.write(black)
            total_frames += 1

    out_video.release()
    print(f"\nSaved {total_frames} frames ({len(selected)} GOPs) to: {out_path}")
    print(f"Playback at {args.fps} fps -> ~{total_frames/args.fps:.0f}s video")


if __name__ == '__main__':
    main()
