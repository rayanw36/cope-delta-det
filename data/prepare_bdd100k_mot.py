"""Prepare BDD100K MOT 2020 per-frame annotations for CoPE-Δ-Det.

BDD100K MOT 2020 provides per-frame bounding box annotations with tracking IDs
for video sequences. This replaces the MV-propagated single-frame GT with
real per-frame annotations.

Expected MOT annotation format (Scalabel JSON):
    {
        "name": "video_name",
        "videoName": "video_name",
        "frames": [
            {
                "frameIndex": 0,
                "timestamp": 0,
                "labels": [
                    {
                        "id": "tracking_id",
                        "category": "car",
                        "box2d": {"x1": ..., "y1": ..., "x2": ..., "y2": ...}
                    }, ...
                ]
            }, ...
        ]
    }

If MOT annotations are not available, falls back to single-frame annotations
replicated across the GOP (current behavior).

Usage:
    python data/prepare_bdd100k_mot.py --bdd100k_root ./data/bdd100k --gop_length 16

Download MOT 2020 labels from:
    https://bdd-data.berkeley.edu/ -> Labels -> MOT 2020
    Place in: data/bdd100k/labels/box_track_20/train/ (or val/)
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

# Same annotation category mapping as prepare_bdd100k.py
ANNOTATION_CATEGORY_MAP = {
    'person': 0, 'pedestrian': 0,
    'rider': 1,
    'car': 2,
    'truck': 3,
    'bus': 4,
    'train': 5,
    'motor': 6, 'motorcycle': 6,
    'bike': 7, 'bicycle': 7,
    'traffic light': 8,
    'traffic sign': 9,
}


def load_mot_annotations(mot_dir, video_name):
    """Load MOT per-frame annotations for a video.

    Returns:
        dict mapping frame_index -> list of annotation dicts
        Each annotation: {'bbox': [x, y, w, h], 'category_id': int, 'track_id': str}
    """
    # Try different file naming conventions
    candidates = [
        mot_dir / f"{video_name}.json",
        mot_dir / f"{video_name}.jpg.json",
    ]

    for path in candidates:
        if path.exists():
            with open(path) as f:
                data = json.load(f)

            frame_anns = {}
            for frame in data.get('frames', []):
                frame_idx = frame.get('frameIndex', frame.get('frame_index', -1))
                if frame_idx < 0:
                    continue

                anns = []
                for label in frame.get('labels', []):
                    category = label.get('category', '')
                    if category not in ANNOTATION_CATEGORY_MAP:
                        continue

                    box2d = label.get('box2d', None)
                    if box2d is None:
                        continue

                    x1 = box2d['x1']
                    y1 = box2d['y1']
                    x2 = box2d['x2']
                    y2 = box2d['y2']

                    anns.append({
                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'category_id': ANNOTATION_CATEGORY_MAP[category],
                        'track_id': label.get('id', ''),
                    })

                frame_anns[frame_idx] = anns

            return frame_anns

    return None  # MOT annotations not found


def create_gop_index_mot(bdd100k_root, split, gop_length=16, hevc_config='qp22_gop16'):
    """Create GOP index using MOT per-frame annotations.

    Falls back to single-frame (100K) annotations with MV propagation
    if MOT annotations are not available for a video.
    """
    root = Path(bdd100k_root)

    # Check for MOT annotation directories
    mot_dirs = [
        root / 'labels' / 'box_track_20' / split,
        root / 'labels' / 'mot' / split,
        root / 'mot_labels' / split,
    ]
    mot_dir = None
    for d in mot_dirs:
        if d.exists():
            mot_dir = d
            break

    if mot_dir is None:
        print(f"  WARNING: MOT annotations not found in any of: {mot_dirs}")
        print(f"  Falling back to single-frame annotations.")
        print(f"  Download MOT 2020 from https://bdd-data.berkeley.edu/")
        return None

    print(f"  Using MOT annotations from: {mot_dir}")

    # Load feature counts for frame counts
    features_dir = root / 'features' / hevc_config
    frame_counts = {}
    if features_dir.exists():
        for vdir in features_dir.iterdir():
            if vdir.is_dir():
                n = len(list(vdir.glob('*.npz')))
                if n > 0:
                    frame_counts[vdir.name] = n

    # Build GOPs
    hevc_dir = root / 'hevc' / hevc_config
    hevc_files = sorted(hevc_dir.glob('*.hevc')) if hevc_dir.exists() else []
    video_names = [f.stem for f in hevc_files]

    gops = []
    mot_found = 0
    mot_missing = 0

    for video_name in video_names:
        n_frames = frame_counts.get(video_name, 0)
        if n_frames < gop_length:
            continue

        mot_anns = load_mot_annotations(mot_dir, video_name)

        if mot_anns is not None:
            mot_found += 1
        else:
            mot_missing += 1
            continue  # Skip videos without MOT annotations

        for start in range(0, n_frames - gop_length + 1, gop_length):
            end = start + gop_length

            # Get per-frame annotations from MOT data
            gop_annotations = []
            has_any_annotation = False

            for offset in range(gop_length):
                frame_idx = start + offset
                if frame_idx in mot_anns and mot_anns[frame_idx]:
                    gop_annotations.append(mot_anns[frame_idx])
                    has_any_annotation = True
                else:
                    gop_annotations.append([])

            gop = {
                'video_name': video_name,
                'start_frame': start,
                'num_frames': gop_length,
                'has_annotation': has_any_annotation,
                'annotations': gop_annotations,
                'annotation_source': 'mot2020',
            }
            gops.append(gop)

    annotated = sum(1 for g in gops if g['has_annotation'])
    print(f"  MOT annotations found for {mot_found}/{len(video_names)} videos")
    print(f"  Created {len(gops)} GOPs ({annotated} with annotations)")

    return gops


def main():
    parser = argparse.ArgumentParser(description='Prepare BDD100K MOT GOP index')
    parser.add_argument('--bdd100k_root', type=str, default='./data/bdd100k')
    parser.add_argument('--gop_length', type=int, default=16)
    parser.add_argument('--hevc_config', type=str, default='qp22_gop16')
    args = parser.parse_args()

    root = Path(args.bdd100k_root)

    for split in ['train']:
        print(f"\n{'='*60}")
        print(f"Building MOT GOP index for {split}")
        print(f"{'='*60}")

        gops = create_gop_index_mot(root, split,
                                     gop_length=args.gop_length,
                                     hevc_config=args.hevc_config)

        if gops is not None:
            out_path = root / f'gop_index_mot_{split}.json'
            with open(out_path, 'w') as f:
                json.dump(gops, f)
            print(f"  Saved to {out_path}")
        else:
            print(f"  MOT index not created — annotations not available")
            print(f"  Using existing single-frame index: gop_index_{split}.json")


if __name__ == '__main__':
    main()
