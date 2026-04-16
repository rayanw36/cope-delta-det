"""Visualize multiple GOPs from the dataset with MV-propagated bounding boxes."""
import cv2
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import BDD100KCoPEDataset

# BDD100K class names for labels
BDD100K_CLASSES = [
    'pedestrian', 'rider', 'car', 'truck', 'bus',
    'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign'
]

# Colors per class (BGR)
CLASS_COLORS = [
    (0, 255, 0),    # pedestrian - green
    (255, 255, 0),  # rider - cyan
    (0, 0, 255),    # car - red
    (0, 165, 255),  # truck - orange
    (255, 0, 255),  # bus - magenta
    (128, 128, 0),  # train - teal
    (0, 255, 255),  # motorcycle - yellow
    (255, 0, 0),    # bicycle - blue
    (0, 128, 255),  # traffic light - orange-red
    (128, 0, 255),  # traffic sign - purple
]


def visualize_video(num_gops=5):
    print("Loading Dataset (annotated GOPs only)...")
    dataset = BDD100KCoPEDataset(
        root_dir='D:/cope-delta-det2/data/bdd100k',
        split='train',
        gop_length=16,
        annotated_only=True
    )

    num_gops = min(num_gops, len(dataset))
    print(f"Found {len(dataset)} annotated GOPs, visualizing {num_gops}")

    out_path = str(Path(__file__).resolve().parent / "gop_mask_playback.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video = cv2.VideoWriter(out_path, fourcc, 10.0, (1280, 720))

    total_frames = 0

    for gop_idx in range(num_gops):
        sample = dataset[gop_idx]
        video_name = sample['video_name']
        targets = sample['targets']

        # Build frame list: I-frame + P-frames
        iframe_rgb = sample['iframe_rgb'].numpy().transpose(1, 2, 0)
        if iframe_rgb.max() <= 1.05:
            iframe_rgb = (iframe_rgb * 255).astype(np.uint8)
        frames = [cv2.cvtColor(iframe_rgb, cv2.COLOR_RGB2BGR)]

        pframe_rgbs = sample['pframe_rgbs']
        for t in range(pframe_rgbs.shape[0]):
            p_rgb = pframe_rgbs[t].numpy().transpose(1, 2, 0)
            if p_rgb.max() <= 1.05:
                p_rgb = (p_rgb * 255).astype(np.uint8)
            frames.append(cv2.cvtColor(p_rgb, cv2.COLOR_RGB2BGR))

        n_objs = targets[0]['boxes'].shape[0] if targets else 0
        print(f"  GOP {gop_idx+1}/{num_gops}: {video_name} | "
              f"{len(frames)} frames | {n_objs} objects")

        for t, frame in enumerate(frames):
            frame_vis = frame.copy()

            if t < len(targets):
                boxes = targets[t]['boxes'].numpy()
                labels = targets[t]['labels'].numpy()

                for i, box in enumerate(boxes):
                    x, y, w, h = box
                    x1, y1 = int(x), int(y)
                    x2, y2 = int(x + w), int(y + h)

                    label_id = int(labels[i]) if i < len(labels) else 0
                    color = CLASS_COLORS[label_id % len(CLASS_COLORS)]
                    class_name = BDD100K_CLASSES[label_id] if label_id < len(BDD100K_CLASSES) else '?'

                    cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, 2)
                    text = f"{class_name}"
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame_vis, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
                    cv2.putText(frame_vis, text, (x1, y1 - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Info overlay
            frame_type = "I-frame" if t == 0 else f"P-frame {t}"
            n_obj = targets[t]['boxes'].shape[0] if t < len(targets) else 0
            info = f"GOP {gop_idx+1} | {video_name} | Frame {t} ({frame_type}) | {n_obj} objs"
            cv2.putText(frame_vis, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            out_video.write(frame_vis)
            total_frames += 1

        # Add a brief black separator between GOPs (5 frames)
        black = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(black, f"--- Next GOP ---", (480, 360),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        for _ in range(5):
            out_video.write(black)
            total_frames += 1

    out_video.release()
    print(f"\nSaved {total_frames} frames ({num_gops} GOPs) to: {out_path}")
    print(f"Playback at 10 fps => ~{total_frames/10:.0f}s video")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gops', type=int, default=10,
                        help='Number of GOPs to visualize')
    args = parser.parse_args()
    visualize_video(num_gops=args.num_gops)
