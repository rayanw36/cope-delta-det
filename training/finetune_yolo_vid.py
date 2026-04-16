"""Fine-tune YOLOv8m on ImageNet VID (30 classes).

Pipeline:
  1. Generate YOLO-format labels (.txt, one per image) from the COCO-style
     imagenet_vid_{train,val}.json files. We only use the frames marked with
     ``is_vid_train_frame=True`` for training (the official sparse set,
     ~57k frames) and 1 frame per val video for quick validation. This keeps
     fine-tuning tractable (a few hours on a single GPU) without sacrificing
     accuracy for our downstream CoPE-Δ-Det task.

  2. Write a data.yaml file pointing to these labels + images.

  3. Call ``ultralytics.YOLO('yolov8m.pt').train(...)``.

After training, the fine-tuned weights (``runs/detect/yolov8m_vid/weights/best.pt``)
become the new anchor detector for ``yolo_anchor.py``.

Usage:
    python training/finetune_yolo_vid.py --vid_root ./data/imagenetvid \
        --epochs 30 --batch 16 --imgsz 640
"""

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path


IMAGENET_VID_CLASSES = [
    'airplane', 'antelope', 'bear', 'bicycle', 'bird',
    'bus', 'car', 'cattle', 'dog', 'domestic_cat',
    'elephant', 'fox', 'giant_panda', 'hamster', 'horse',
    'lion', 'lizard', 'monkey', 'motorcycle', 'rabbit',
    'red_panda', 'sheep', 'snake', 'squirrel', 'tiger',
    'train', 'turtle', 'watercraft', 'whale', 'zebra',
]


def collect_frames_to_use(coco_json, split):
    """Return ``{image_id: image_dict}`` for the frames we want to train on."""
    data = json.load(open(coco_json, 'r'))

    imgs_by_id = {img['id']: img for img in data['images']}

    if split == 'train':
        # Only use the official sparse training frames
        selected = {img_id: img for img_id, img in imgs_by_id.items()
                    if img.get('is_vid_train_frame', False)}
    else:
        # For val, use one frame per video (first frame of each video)
        by_video = defaultdict(list)
        for img in data['images']:
            by_video[img['video_id']].append(img)
        selected = {}
        for vid_id, frames in by_video.items():
            frames.sort(key=lambda x: x['frame_id'])
            selected[frames[0]['id']] = frames[0]

    anns_by_img = defaultdict(list)
    for ann in data['annotations']:
        if ann['image_id'] in selected:
            anns_by_img[ann['image_id']].append(ann)

    return selected, anns_by_img, data['categories']


def write_yolo_labels(selected, anns_by_img, cats, vid_data_root, labels_dir,
                      images_txt_out):
    """Write a .txt label file per image and a .txt list of image paths."""
    cat_id_to_idx = {c['id']: c['id'] - 1 for c in cats}  # 1..30 -> 0..29

    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    written = 0
    for img_id, img in selected.items():
        h, w = img['height'], img['width']
        anns = anns_by_img.get(img_id, [])
        if not anns:
            continue  # skip frames with no GT — they'd just slow training

        # Compute a safe label filename based on the flat image path
        rel = img['file_name']
        stem = rel.replace('/', '__').replace('.JPEG', '')
        lbl_path = labels_dir / f"{stem}.txt"

        with open(lbl_path, 'w') as lf:
            for a in anns:
                cat = cat_id_to_idx.get(a['category_id'])
                if cat is None:
                    continue
                x, y, bw, bh = a['bbox']  # COCO
                if bw <= 0 or bh <= 0:
                    continue
                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                nw = bw / w
                nh = bh / h
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                nw = max(0.0, min(1.0, nw))
                nh = max(0.0, min(1.0, nh))
                lf.write(f"{cat} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        # YOLO expects labels next to images with the same stem. Because our
        # JPEGs live in deeply nested directories we use the list-file form
        # instead: train.txt contains absolute JPEG paths, and YOLO derives
        # label paths by swapping "/images/" -> "/labels/". We sidestep this
        # by providing a symlink tree. See below.
        abs_jpeg = (Path(vid_data_root) / rel).resolve()
        lines.append(str(abs_jpeg))
        written += 1

    with open(images_txt_out, 'w') as f:
        f.write('\n'.join(lines) + ('\n' if lines else ''))

    print(f"  wrote {written} images, labels in {labels_dir}")
    return written


def build_symlink_tree(selected, vid_data_root, images_flat_dir, labels_flat_dir,
                       labels_src_dir):
    """YOLOv8 derives label paths by swapping the 'images/' directory in the
    image path for 'labels/'. To stay compatible we create a flat directory
    of symlinks (or copies on Windows if symlinks fail) for both images and
    labels."""
    images_flat_dir = Path(images_flat_dir); images_flat_dir.mkdir(parents=True, exist_ok=True)
    labels_flat_dir = Path(labels_flat_dir); labels_flat_dir.mkdir(parents=True, exist_ok=True)
    labels_src_dir = Path(labels_src_dir)

    n_img = 0
    n_lbl = 0
    for img_id, img in selected.items():
        rel = img['file_name']
        stem = rel.replace('/', '__').replace('.JPEG', '')
        src_img = (Path(vid_data_root) / rel).resolve()
        dst_img = images_flat_dir / f"{stem}.JPEG"
        if not dst_img.exists():
            if not src_img.exists():
                continue  # Source JPEG missing (e.g. val split not yet downloaded)
            try:
                os.symlink(src_img, dst_img)
            except (OSError, NotImplementedError):
                # Fallback: copy (slow but robust on Windows without dev mode)
                shutil.copy2(src_img, dst_img)
        n_img += 1

        src_lbl = labels_src_dir / f"{stem}.txt"
        dst_lbl = labels_flat_dir / f"{stem}.txt"
        if src_lbl.exists() and not dst_lbl.exists():
            shutil.copy2(src_lbl, dst_lbl)
            n_lbl += 1

    print(f"  symlink/copy: {n_img} images, {n_lbl} labels")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vid_root', default='./data/imagenetvid')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default='0')
    parser.add_argument('--model', default='yolov8m.pt')
    parser.add_argument('--name', default='yolov8m_vid')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--prep_only', action='store_true',
                        help='Only write labels + data.yaml; skip training')
    args = parser.parse_args()

    vid_root = Path(args.vid_root)
    ann_dir = vid_root / 'annotations'
    data_root = vid_root / 'ILSVRC2015' / 'Data' / 'VID'
    yolo_root = vid_root / 'yolo_vid'      # everything for YOLO lives here
    yolo_root.mkdir(parents=True, exist_ok=True)

    for split in ('train', 'val'):
        print(f"\n=== {split} ===")
        coco_json = ann_dir / f'imagenet_vid_{split}.json'
        if not coco_json.exists():
            print(f"  MISSING {coco_json}, skipping")
            continue

        selected, anns_by_img, cats = collect_frames_to_use(coco_json, split)
        print(f"  selected {len(selected)} images")

        labels_src = yolo_root / 'labels_src' / split
        images_txt = yolo_root / f'{split}.txt'
        write_yolo_labels(selected, anns_by_img, cats,
                          vid_data_root=data_root,
                          labels_dir=labels_src,
                          images_txt_out=images_txt)

        images_flat = yolo_root / 'images' / split
        labels_flat = yolo_root / 'labels' / split
        build_symlink_tree(selected, data_root, images_flat, labels_flat,
                           labels_src)

    # Write data.yaml
    data_yaml = yolo_root / 'data.yaml'
    with open(data_yaml, 'w') as f:
        f.write("# Auto-generated by training/finetune_yolo_vid.py\n")
        f.write(f"path: {yolo_root.resolve().as_posix()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(IMAGENET_VID_CLASSES)}\n")
        f.write("names:\n")
        for i, n in enumerate(IMAGENET_VID_CLASSES):
            f.write(f"  {i}: {n}\n")
    print(f"\nWrote {data_yaml}")

    if args.prep_only:
        print("prep_only set; skipping training.")
        return

    # Train
    from ultralytics import YOLO
    model = YOLO(args.model)
    print(f"\nFine-tuning {args.model} for {args.epochs} epochs on ImageNet VID")
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        name=args.name,
        patience=10,
        amp=True,
    )
    print("\nFine-tune done. Best weights:")
    print(f"  runs/detect/{args.name}/weights/best.pt")


if __name__ == '__main__':
    main()
