"""Prepare BDD100K dataset for CoPE-Δ-Det.

Builds GOP indices from HEVC-encoded video sequences.

BDD100K structure:
    bdd100k/
    ├── 100k/
    │   ├── train/   # Per-video JSON annotations (1 annotated frame each)
    │   └── val/
    ├── videos/
    │   ├── train/   # Original .mov video files
    │   └── val/
    ├── hevc/        # HEVC re-encoded videos (from encode_hevc.py)
    │   ├── qp22_gop16/
    │   └── ...
    └── features/    # Extracted features (from extract_features.py)
        ├── qp22_gop16/
        └── ...

Key insight: In BDD100K, each full filename (e.g., '0000f77c-6257be58') is a UNIQUE video.
The 100K image annotations label ONE frame per video at timestamp=10000ms (~frame 300 at 30fps).

GOP construction:
    - Each HEVC video (~1200 frames) is split into consecutive GOPs of gop_length frames
    - The annotation (at frame ~300) is assigned to the GOP containing that frame
    - Within that GOP, the annotation is replicated to all frames (objects move < few pixels in 0.5s)
    - GOPs without annotations still get empty annotation lists (useful for self-supervised training)

Usage:
    python data/prepare_bdd100k.py --bdd100k_root ./data/bdd100k --gop_length 16
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict
import av

# BDD100K class mapping
# Canonical class names (used throughout our pipeline)
BDD100K_CLASSES = [
    'pedestrian', 'rider', 'car', 'truck', 'bus',
    'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign'
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(BDD100K_CLASSES)}

# BDD100K annotation files use DIFFERENT category names than the standard class list.
# Map annotation category names → our class IDs.
ANNOTATION_CATEGORY_MAP = {
    'person': 0,       # annotation uses 'person', we call it 'pedestrian'
    'pedestrian': 0,   # some annotation formats use 'pedestrian'
    'rider': 1,
    'car': 2,
    'truck': 3,
    'bus': 4,
    'train': 5,
    'motor': 6,        # annotation uses 'motor', we call it 'motorcycle'
    'motorcycle': 6,   # some formats use 'motorcycle'
    'bike': 7,         # annotation uses 'bike', we call it 'bicycle'
    'bicycle': 7,      # some formats use 'bicycle'
    'traffic light': 8,
    'traffic sign': 9,
}


def load_video_annotation(json_path):
    """Load a per-video BDD100K annotation JSON.

    Returns list of annotation dicts with 'bbox' [x, y, w, h] and 'category_id'.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    annotations = []
    frames = data.get('frames', [])
    if not frames:
        return annotations

    # BDD100K 100K labels have 1 frame at timestamp=10000
    frame = frames[0]
    for obj in frame.get('objects', []):
        category = obj.get('category', '')
        if category not in ANNOTATION_CATEGORY_MAP:
            continue

        box2d = obj.get('box2d', None)
        if box2d is None:
            continue

        x1 = box2d['x1']
        y1 = box2d['y1']
        x2 = box2d['x2']
        y2 = box2d['y2']
        w = x2 - x1
        h = y2 - y1

        annotations.append({
            'bbox': [x1, y1, w, h],  # COCO format
            'category_id': ANNOTATION_CATEGORY_MAP[category]
        })

    return annotations


def get_video_frame_count(video_path):
    """Get frame count from a video file."""
    try:
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        # If metadata frame count is available, use it
        n_frames = stream.frames
        if n_frames == 0:
            # Fallback: count frames by decoding
            n_frames = sum(1 for _ in container.decode(stream))
        container.close()
        return n_frames
    except Exception as e:
        print(f"  Error reading {video_path}: {e}")
        return 0


def create_gop_index(bdd100k_root, split, gop_length=16, hevc_config='qp22_gop16'):
    """Create GOP index from actual video sequences.

    For each video:
    1. Determine frame count from the HEVC file (or .mov)
    2. Split into consecutive GOPs of gop_length frames
    3. Load the single annotation and assign it to the correct GOP
    4. Replicate annotation across all frames in that GOP

    Returns list of GOP dicts.
    """
    root = Path(bdd100k_root)

    # Find annotation JSON files
    ann_dir = root / '100k' / split
    if not ann_dir.exists():
        print(f"Annotation directory not found: {ann_dir}")
        return []

    # Find video source (HEVC preferred for accurate frame counts)
    hevc_dir = root / 'hevc' / hevc_config
    mov_dirs = [
        root / 'videos' / split,
        root / 'bdd100k' / 'videos' / split,
    ]

    # Build set of available videos
    available_videos = set()
    video_source = {}

    if hevc_dir.exists():
        for f in hevc_dir.glob('*.hevc'):
            name = f.stem
            available_videos.add(name)
            video_source[name] = f
        print(f"  Found {len(available_videos)} HEVC videos in {hevc_dir}")
    else:
        for md in mov_dirs:
            if md.exists():
                for f in md.glob('*.mov'):
                    name = f.stem
                    available_videos.add(name)
                    video_source[name] = f
                print(f"  Found {len(available_videos)} .mov videos in {md}")
                break

    if not available_videos:
        print(f"  No video files found!")
        return []

    # Load annotations for available videos
    ann_files = list(ann_dir.glob('*.json'))
    print(f"  Found {len(ann_files)} annotation files in {ann_dir}")

    video_annotations = {}
    for af in ann_files:
        video_name = af.stem
        if video_name in available_videos:
            anns = load_video_annotation(af)
            video_annotations[video_name] = anns

    print(f"  Matched annotations for {len(video_annotations)}/{len(available_videos)} videos")

    # Determine frame counts
    # For efficiency, use feature files if available, otherwise sample a few videos
    features_dir = root / 'features' / hevc_config
    frame_counts = {}

    if features_dir.exists():
        # Count NPZ files per video
        for vdir in features_dir.iterdir():
            if vdir.is_dir():
                n = len(list(vdir.glob('*.npz')))
                if n > 0:
                    frame_counts[vdir.name] = n
        print(f"  Got frame counts from features dir for {len(frame_counts)} videos")

    # For videos without feature counts, get from video file
    missing = available_videos - set(frame_counts.keys())
    if missing:
        print(f"  Getting frame counts from video files for {len(missing)} videos...")
        for i, name in enumerate(sorted(missing)):
            if name in video_source:
                frame_counts[name] = get_video_frame_count(video_source[name])
            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{len(missing)}...")

    # BDD100K annotation timestamp = 10000ms → frame 300 at 30fps
    ANNOTATED_FRAME = 300

    # Build GOPs
    gops = []
    videos_with_annotated_gop = 0

    for video_name in sorted(available_videos):
        n_frames = frame_counts.get(video_name, 0)
        if n_frames < gop_length:
            continue

        anns = video_annotations.get(video_name, [])

        # Split video into consecutive GOPs
        for start in range(0, n_frames - gop_length + 1, gop_length):
            end = start + gop_length

            # Check if annotated frame falls in this GOP
            has_annotation = (start <= ANNOTATED_FRAME < end) and len(anns) > 0

            if has_annotation:
                # Replicate annotation to all frames in this GOP
                # Objects barely move in 0.5s at 30fps
                gop_annotations = [anns for _ in range(gop_length)]
                videos_with_annotated_gop += 1
            else:
                gop_annotations = [[] for _ in range(gop_length)]

            gop = {
                'video_name': video_name,
                'start_frame': start,
                'num_frames': gop_length,
                'frame_names': [f"{video_name}.jpg"],  # Reference name for the video
                'has_annotation': has_annotation,
                'annotations': gop_annotations
            }
            gops.append(gop)

    annotated_gops = sum(1 for g in gops if g['has_annotation'])
    print(f"  Created {len(gops)} GOPs ({annotated_gops} with annotations) "
          f"from {len(available_videos)} videos")

    return gops


def main():
    parser = argparse.ArgumentParser(description='Prepare BDD100K GOP index')
    parser.add_argument('--bdd100k_root', type=str, default='./data/bdd100k',
                        help='Root directory of BDD100K dataset')
    parser.add_argument('--gop_length', type=int, default=16,
                        help='GOP length for indexing')
    parser.add_argument('--hevc_config', type=str, default='qp22_gop16',
                        help='HEVC config to use for frame counts')
    args = parser.parse_args()

    root = Path(args.bdd100k_root)

    for split in ['train', 'val']:
        print(f"\n{'='*60}")
        print(f"Building GOP index for {split} split")
        print(f"{'='*60}")

        gops = create_gop_index(root, split,
                                gop_length=args.gop_length,
                                hevc_config=args.hevc_config)

        if gops:
            gop_index_path = root / f'gop_index_{split}.json'
            with open(gop_index_path, 'w') as f:
                json.dump(gops, f)
            print(f"  Saved {len(gops)} GOPs to {gop_index_path}")
        else:
            print(f"  No GOPs created for {split}")

    print("\nDone!")


if __name__ == '__main__':
    main()
