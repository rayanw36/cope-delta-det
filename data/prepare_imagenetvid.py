"""Prepare ImageNet VID (ILSVRC 2015) dataset for CoPE-Δ-Det.

Builds GOP indices from per-frame annotated video snippets. Unlike BDD100K
(where we had a single annotated frame per video and had to propagate boxes
with MVs), ImageNet VID provides ground-truth bounding boxes for every frame
— so each entry in ``gop['annotations']`` contains the *real* boxes for that
specific frame.

Expected layout after downloading from `guanxiongsun/imagenetvid` on HF:

    imagenetvid/
    ├── annotations/
    │   ├── imagenet_vid_train.json   # COCO-style JSON with per-frame anns
    │   └── imagenet_vid_val.json
    ├── ILSVRC2015/
    │   └── Data/VID/
    │       ├── train/
    │       │   └── ILSVRC2015_VID_train_0000/
    │       │       └── ILSVRC2015_train_00000000/
    │       │           ├── 000000.JPEG
    │       │           ├── 000001.JPEG
    │       │           └── ...
    │       └── val/
    │           └── ILSVRC2015_val_00000000/
    │               └── 000000.JPEG
    ├── hevc/qp22_gop16/              # (filled by encode_imagenetvid_hevc.py)
    │   └── <video_name>.hevc
    └── features/qp22_gop16/          # (filled by extract_features.py)
        └── <video_name>/frame_XXXX.npz

The ``video_name`` used throughout the pipeline is the *full* snippet path
relative to Data/VID, with forward slashes replaced by ``__`` so the result is
a single safe filename. For example::

    train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00000000
      → train__ILSVRC2015_VID_train_0000__ILSVRC2015_train_00000000

All output video_names follow this scheme. This keeps the HEVC and features
directory flat while still round-trippable to the JPEG directory.

Usage:
    python data/prepare_imagenetvid.py --vid_root ./data/imagenetvid --gop_length 16
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


# ImageNet VID categories — 30 classes, 1-indexed in the source JSON.
# We convert to 0-indexed ids to match BDD100K pipeline conventions.
IMAGENET_VID_CLASSES = [
    'airplane', 'antelope', 'bear', 'bicycle', 'bird',
    'bus', 'car', 'cattle', 'dog', 'domestic_cat',
    'elephant', 'fox', 'giant_panda', 'hamster', 'horse',
    'lion', 'lizard', 'monkey', 'motorcycle', 'rabbit',
    'red_panda', 'sheep', 'snake', 'squirrel', 'tiger',
    'train', 'turtle', 'watercraft', 'whale', 'zebra',
]
NUM_CLASSES = len(IMAGENET_VID_CLASSES)
assert NUM_CLASSES == 30


def video_id_to_name(video_path):
    """Convert a video JSON 'name' field to a flat, filesystem-safe video_name.

    Example:
        'train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00000000'
        → 'train__ILSVRC2015_VID_train_0000__ILSVRC2015_train_00000000'
    """
    return video_path.replace('/', '__')


def build_gop_index(coco_json_path, gop_length=16, max_gops_per_video=None):
    """Build a GOP index list from an ImageNet VID COCO-style JSON file.

    Parameters
    ----------
    coco_json_path : str | Path
        Path to imagenet_vid_{train,val}.json
    gop_length : int
        Frames per GOP (16 for QP22/GOP16 pipeline).
    max_gops_per_video : int | None
        If set, spread at most this many GOPs across each video's frame range.
        ``None`` means use every consecutive GOP.

    Returns
    -------
    gops : list[dict]
        Each dict has the same schema as the BDD100K gop_index entries, with
        the annotation list now containing *real* per-frame boxes.
    """
    print(f"Loading {coco_json_path} ...")
    with open(coco_json_path, 'r') as f:
        data = json.load(f)

    cats = data['categories']
    # The JSON uses 1..30; we remap to 0..29 for our training code.
    catid_to_idx = {c['id']: c['id'] - 1 for c in cats}

    # Index images by video_id → [image, ...] sorted by frame_id
    imgs_by_vid = defaultdict(list)
    for img in data['images']:
        imgs_by_vid[img['video_id']].append(img)
    for v in imgs_by_vid.values():
        v.sort(key=lambda x: x['frame_id'])

    # Index annotations by image_id
    anns_by_img = defaultdict(list)
    for ann in data['annotations']:
        anns_by_img[ann['image_id']].append(ann)

    # Video id → name
    vid_id_to_name = {v['id']: v['name'] for v in data['videos']}

    gops = []
    total_frames_used = 0
    total_anns_used = 0
    videos_with_gops = 0

    for video_id, frames in imgs_by_vid.items():
        n_frames = len(frames)
        if n_frames < gop_length:
            continue

        video_name = video_id_to_name(vid_id_to_name[video_id])

        # Decide GOP start positions
        max_full_gops = n_frames // gop_length
        if max_gops_per_video is None or max_full_gops <= max_gops_per_video:
            # All consecutive non-overlapping GOPs
            start_indices = list(range(0, max_full_gops * gop_length, gop_length))
        else:
            # Evenly spread `max_gops_per_video` GOPs across the video length.
            # We use the *frame offset* (not the contiguous gop_length-stride
            # math) so the samples actually span the whole snippet.
            span = n_frames - gop_length
            step = span / (max_gops_per_video - 1) if max_gops_per_video > 1 else 0
            start_indices = sorted({
                int(round(i * step)) for i in range(max_gops_per_video)
            })

        had_gop = False
        for start in start_indices:
            gop_frames = frames[start:start + gop_length]
            if len(gop_frames) < gop_length:
                continue

            # Collect per-frame annotations for every frame in this GOP
            # Scale boxes to the canonical 720x1280 resolution used by
            # the pipeline (dataset.py resizes all frames to this size).
            TARGET_W, TARGET_H = 1280, 720
            per_frame_anns = []
            any_ann = False
            for img in gop_frames:
                orig_w = img['width']
                orig_h = img['height']
                sx = TARGET_W / orig_w
                sy = TARGET_H / orig_h
                anns = []
                for a in anns_by_img.get(img['id'], []):
                    cat_idx = catid_to_idx.get(a['category_id'])
                    if cat_idx is None:
                        continue
                    bbox = a['bbox']  # COCO [x, y, w, h] in original coords
                    if bbox[2] <= 0 or bbox[3] <= 0:
                        continue
                    anns.append({
                        'bbox': [float(bbox[0] * sx), float(bbox[1] * sy),
                                 float(bbox[2] * sx), float(bbox[3] * sy)],
                        'category_id': int(cat_idx),
                    })
                if anns:
                    any_ann = True
                per_frame_anns.append(anns)

            gops.append({
                'video_name': video_name,
                'start_frame': int(start),
                'num_frames': gop_length,
                'frame_names': [img['file_name'] for img in gop_frames],
                'has_annotation': any_ann,
                'annotations': per_frame_anns,
                'per_frame_gt': True,   # flag for the dataset loader
            })
            had_gop = True
            total_frames_used += gop_length
            total_anns_used += sum(len(a) for a in per_frame_anns)

        if had_gop:
            videos_with_gops += 1

    print(f"  videos_with_gops = {videos_with_gops} / {len(imgs_by_vid)}")
    print(f"  total GOPs        = {len(gops)}")
    print(f"  total frames used = {total_frames_used}")
    print(f"  total anns used   = {total_anns_used}")
    return gops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vid_root', type=str, default='./data/imagenetvid')
    parser.add_argument('--gop_length', type=int, default=16)
    parser.add_argument('--max_gops_per_video_train', type=int, default=6,
                        help='Evenly-sampled GOPs per training video (set to 0 for all).')
    parser.add_argument('--max_gops_per_video_val', type=int, default=0,
                        help='Evenly-sampled GOPs per validation video (0 = all).')
    args = parser.parse_args()

    root = Path(args.vid_root)
    ann_dir = root / 'annotations'

    for split, cap in [
        ('train', args.max_gops_per_video_train),
        ('val', args.max_gops_per_video_val),
    ]:
        print(f"\n{'='*60}")
        print(f"Building GOP index — {split}")
        print(f"{'='*60}")
        coco_json = ann_dir / f'imagenet_vid_{split}.json'
        if not coco_json.exists():
            print(f"  MISSING {coco_json} — skipping")
            continue

        cap_val = cap if cap > 0 else None
        gops = build_gop_index(coco_json, gop_length=args.gop_length,
                               max_gops_per_video=cap_val)

        out_path = root / f'gop_index_{split}.json'
        with open(out_path, 'w') as f:
            json.dump(gops, f)
        print(f"  -> wrote {len(gops)} GOPs to {out_path}")

    print("\nClasses:")
    for i, name in enumerate(IMAGENET_VID_CLASSES):
        print(f"  {i:2d} {name}")
    print("\nDone.")


if __name__ == '__main__':
    main()
