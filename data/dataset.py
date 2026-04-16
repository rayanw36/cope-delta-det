import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

ANNOTATED_FRAME = 300   # BDD100K keyframe at t=10s, 30fps
BLOCK_SIZE = 16         # MV grid block size (720/16=45, 1280/16=80)


def propagate_boxes_with_mvs(boxes_xywh, mvs_np, start_frame, num_frames,
                             annotated_offset, img_h=720, img_w=1280):
    """Propagate bounding boxes across a GOP using extracted motion vectors.

    The annotation lives at one specific offset inside the GOP.  For every other
    frame we shift each box by the *average MV inside its region*, accumulated
    frame-by-frame so that drift stays small.

    Args:
        boxes_xywh:  (N, 4) ndarray  [x, y, w, h]  — the ground-truth boxes
        mvs_np:      list of (45, 80, 2) MV arrays for P-frames (indices 0..num_frames-2,
                     where index i corresponds to GOP frame i+1)
        start_frame: absolute frame index of the first GOP frame
        num_frames:  number of frames in the GOP
        annotated_offset: which offset inside the GOP holds the annotation
        img_h, img_w: frame dimensions for clamping

    Returns:
        List[ndarray] of length num_frames, each (N, 4) [x, y, w, h].
    """
    N = boxes_xywh.shape[0]
    if N == 0:
        return [boxes_xywh.copy() for _ in range(num_frames)]

    per_frame_boxes = [None] * num_frames
    per_frame_boxes[annotated_offset] = boxes_xywh.copy()

    def _avg_mv_in_box(mv_map, box):
        """Average MV (dx, dy) within a bounding box region on the 45×80 grid."""
        x, y, w, h = box
        # Convert pixel coords → grid coords
        gx1 = max(0, int(x / BLOCK_SIZE))
        gy1 = max(0, int(y / BLOCK_SIZE))
        gx2 = min(mv_map.shape[1], int((x + w) / BLOCK_SIZE) + 1)
        gy2 = min(mv_map.shape[0], int((y + h) / BLOCK_SIZE) + 1)
        if gx2 <= gx1 or gy2 <= gy1:
            return 0.0, 0.0
        region = mv_map[gy1:gy2, gx1:gx2]          # (rows, cols, 2)
        return float(np.median(region[:, :, 0])), float(np.median(region[:, :, 1]))

    def _shift_boxes(boxes, mv_map, direction=1):
        """Shift boxes by median MV.  direction=+1 forward, -1 backward."""
        shifted = boxes.copy()
        for i in range(N):
            dx, dy = _avg_mv_in_box(mv_map, shifted[i])
            shifted[i, 0] += direction * dx   # x
            shifted[i, 1] += direction * dy   # y
            # Clamp to image bounds
            shifted[i, 0] = np.clip(shifted[i, 0], 0, img_w - shifted[i, 2])
            shifted[i, 1] = np.clip(shifted[i, 1], 0, img_h - shifted[i, 3])
        return shifted

    # ── Forward propagation: annotated_offset → annotated_offset+1 → … → end
    prev_boxes = per_frame_boxes[annotated_offset]
    for t in range(annotated_offset + 1, num_frames):
        mv_idx = t - 1          # mvs_np[i] is MV for GOP frame i+1
        if mv_idx < len(mvs_np):
            prev_boxes = _shift_boxes(prev_boxes, mvs_np[mv_idx], direction=+1)
        per_frame_boxes[t] = prev_boxes.copy()

    # ── Backward propagation: annotated_offset → annotated_offset-1 → … → 0
    prev_boxes = per_frame_boxes[annotated_offset]
    for t in range(annotated_offset - 1, -1, -1):
        mv_idx = t              # mvs_np[t] is MV for GOP frame t+1 (motion from t→t+1)
        if mv_idx < len(mvs_np):
            prev_boxes = _shift_boxes(prev_boxes, mvs_np[mv_idx], direction=-1)
        per_frame_boxes[t] = prev_boxes.copy()

    return per_frame_boxes


class BDD100KCoPEDataset(Dataset):
    """
    Dataset for CoPE-Δ-Det.
    Loads GOPs containing the I-frame RGB and P-frame HEVC primitives.

    Each GOP = gop_length consecutive frames from a single video.
    Frame 0 = I-frame (decoded to RGB for YOLO anchor detection).
    Frames 1..N = P-frames (use HEVC codec primitives for Δ-Det).

    Two operating modes:
      * BDD100K (single annotated frame + MV-based box propagation)
      * ImageNet VID ``dataset_type='imagenetvid'`` (real per-frame GT, JPEG
        RGBs, no propagation). The GOP index file created by
        ``prepare_imagenetvid.py`` carries ``per_frame_gt=True`` and
        ``frame_names`` listing the source JPEGs relative to
        ``ILSVRC2015/Data/VID``.
    """
    def __init__(self, root_dir, split='train', gop_length=16, transform=None,
                 annotated_only=False, features_subdir='features',
                 dataset_type='bdd100k'):
        self.root_dir = Path(root_dir)
        self.split = split
        self.gop_length = gop_length
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
        ])
        self.annotated_only = annotated_only
        self.dataset_type = dataset_type.lower()
        assert self.dataset_type in ('bdd100k', 'imagenetvid')

        # Load GOP index
        index_path = self.root_dir / f'gop_index_{split}.json'
        with open(index_path, 'r') as f:
            all_gops = json.load(f)

        # Optionally filter to only annotated GOPs (for evaluation)
        if annotated_only:
            self.gops = [g for g in all_gops if g.get('has_annotation', False)]
        else:
            self.gops = all_gops

        # Feature and video directories
        # features_subdir can be 'features' (block-matched) or 'features_pyav' (H.264 proxy codec MVs)
        self.features_dir = self.root_dir / features_subdir / f'qp22_gop{gop_length}'

        # ImageNet VID JPEG root (only used when dataset_type=='imagenetvid')
        self.vid_data_root = self.root_dir / 'ILSVRC2015' / 'Data' / 'VID'

        # Search paths for source videos (BDD100K only)
        self.video_dirs = [
            self.root_dir / 'bdd100k' / 'videos' / 'train',
            self.root_dir / 'bdd100k' / 'videos' / 'val',
            self.root_dir / 'videos' / 'train',
            self.root_dir / 'videos' / 'val',
        ]

    def __len__(self):
        return len(self.gops)

    def _find_video(self, video_name):
        """Find the source .mov video file for a given video name."""
        for vdir in self.video_dirs:
            p = vdir / f"{video_name}.mov"
            if p.exists():
                return p
        return None

    def load_frame_features(self, video_name, start_frame, num_frames):
        """Load pre-extracted .npz primitives for the P-frames in the GOP.

        Returns:
            mvs_t:    (N-1, 45, 80, 2)  motion vectors for P-frames
            res_t:    (N-1, 45, 80, 1)  residual energy
            depths_t: (N-1, 45, 80, 1)  partition depth
            modes_t:  (N-1, 45, 80, 1)  prediction mode
            mvs_raw:  list of (45, 80, 2) numpy arrays (for box propagation)
        """
        vid_dir = self.features_dir / video_name

        mvs = []
        res = []
        depths = []
        modes = []
        mvs_raw = []   # Keep numpy copies for box propagation

        # Frame 0 is I-frame (no P-frame primitives needed)
        # Frames 1..N-1 are P-frames
        for offset in range(1, num_frames):
            frame_idx = start_frame + offset
            feat_path = vid_dir / f'frame_{frame_idx:04d}.npz'

            if feat_path.exists():
                data = np.load(str(feat_path))
                mv_np = data['mv']
                mvs.append(mv_np)
                mvs_raw.append(mv_np)
                res.append(data['res_energy'])
                depths.append(data['part_depth'])
                modes.append(data['pred_mode'])
            else:
                zero_mv = np.zeros((45, 80, 2), dtype=np.float32)
                mvs.append(zero_mv)
                mvs_raw.append(zero_mv)
                res.append(np.zeros((45, 80, 1), dtype=np.float32))
                depths.append(np.zeros((45, 80, 1), dtype=np.int8))
                modes.append(np.zeros((45, 80, 1), dtype=np.int8))

        if mvs:
            mvs_t = torch.from_numpy(np.stack(mvs))
            res_t = torch.from_numpy(np.stack(res))
            depths_t = torch.from_numpy(np.stack(depths)).long()
            modes_t = torch.from_numpy(np.stack(modes)).long()
        else:
            mvs_t = torch.empty((0, 45, 80, 2))
            res_t = torch.empty((0, 45, 80, 1))
            depths_t = torch.empty((0, 45, 80, 1)).long()
            modes_t = torch.empty((0, 45, 80, 1)).long()

        return mvs_t, res_t, depths_t, modes_t, mvs_raw

    def load_video_rgbs(self, video_name, start_frame, num_frames, frame_names=None):
        """Load decoded RGB frames.

        BDD100K: seek inside the source .mov file.
        ImageNet VID: open the JPEG files listed in ``frame_names``.
        """
        if self.dataset_type == 'imagenetvid':
            assert frame_names is not None, "frame_names required for imagenetvid"
            rgbs_list = []
            for fn in frame_names:
                jpg_path = self.vid_data_root / fn
                try:
                    img = Image.open(jpg_path).convert('RGB')
                    # Normalise to 720x1280 if the snippet has a different size
                    if img.size != (1280, 720):
                        img = img.resize((1280, 720), Image.BILINEAR)
                    rgbs_list.append(self.transform(img))
                except Exception as e:
                    print(f"WARNING: failed to read {jpg_path}: {e}")
                    rgbs_list.append(torch.zeros((3, 720, 1280)))
            return rgbs_list

        vid_path = self._find_video(video_name)

        if vid_path is None:
            print(f"WARNING: Video not found: {video_name}")
            return [torch.zeros((3, 720, 1280)) for _ in range(num_frames)]

        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            print(f"WARNING: Could not open video: {vid_path}")
            return [torch.zeros((3, 720, 1280)) for _ in range(num_frames)]

        rgbs_list = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for offset in range(num_frames):
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                rgbs_list.append(self.transform(img))
            else:
                rgbs_list.append(torch.zeros((3, 720, 1280)))
        cap.release()

        return rgbs_list

    def __getitem__(self, idx):
        gop_meta = self.gops[idx]
        video_name = gop_meta['video_name']
        start_frame = gop_meta['start_frame']
        num_frames = gop_meta.get('num_frames', self.gop_length)
        frame_names = gop_meta.get('frame_names', None)
        per_frame_gt = gop_meta.get('per_frame_gt', False) or self.dataset_type == 'imagenetvid'

        # 1. Load RGB frames from source video / JPEGs
        rgbs_list = self.load_video_rgbs(video_name, start_frame, num_frames,
                                         frame_names=frame_names)

        iframe_tensor = rgbs_list[0]
        if len(rgbs_list) > 1:
            pframe_rgbs = torch.stack(rgbs_list[1:])
        else:
            pframe_rgbs = torch.empty((0, 3, 720, 1280))

        # 2. Load P-frame HEVC primitives
        mvs, res, depths, modes, mvs_raw = self.load_frame_features(
            video_name, start_frame, num_frames)

        # 3. Build per-frame targets
        if per_frame_gt:
            # Real per-frame annotations — no MV propagation.
            targets = []
            for t in range(num_frames):
                anns_t = gop_meta['annotations'][t] if t < len(gop_meta['annotations']) else []
                if anns_t:
                    boxes_t = np.array([a['bbox'] for a in anns_t], dtype=np.float32)
                    labels_t = np.array([a['category_id'] for a in anns_t], dtype=np.int64)
                    targets.append({
                        'boxes': torch.from_numpy(boxes_t),
                        'labels': torch.from_numpy(labels_t),
                    })
                else:
                    targets.append({
                        'boxes': torch.empty((0, 4)),
                        'labels': torch.empty((0,), dtype=torch.long),
                    })
        else:
            # Legacy BDD100K path: single annotated frame at ANNOTATED_FRAME,
            # propagated with MVs.
            annotated_offset = ANNOTATED_FRAME - start_frame
            annotated_offset = max(0, min(annotated_offset, num_frames - 1))

            ref_anns = gop_meta['annotations'][annotated_offset]
            ref_boxes = np.array([a['bbox'] for a in ref_anns], dtype=np.float32) if ref_anns else np.empty((0, 4), dtype=np.float32)
            ref_labels = np.array([a['category_id'] for a in ref_anns], dtype=np.int64) if ref_anns else np.empty((0,), dtype=np.int64)

            if ref_boxes.shape[0] > 0 and len(mvs_raw) > 0:
                propagated = propagate_boxes_with_mvs(
                    ref_boxes, mvs_raw, start_frame, num_frames, annotated_offset)
            else:
                propagated = [ref_boxes.copy() for _ in range(num_frames)]

            targets = []
            for t in range(num_frames):
                boxes_t = propagated[t]
                targets.append({
                    'boxes': torch.from_numpy(boxes_t) if boxes_t.shape[0] > 0 else torch.empty((0, 4)),
                    'labels': torch.from_numpy(ref_labels.copy()) if ref_labels.shape[0] > 0 else torch.empty((0,), dtype=torch.long)
                })

        return {
            'video_name': video_name,
            'iframe_rgb': iframe_tensor,
            'pframe_rgbs': pframe_rgbs,
            'pframe_mvs': mvs,
            'pframe_res': res,
            'pframe_depths': depths,
            'pframe_modes': modes,
            'targets': targets
        }

def collate_fn(batch):
    """Custom collate to handle lists of dicts with varying tensor sizes."""
    return tuple(zip(*[b.values() for b in batch]))
