"""HEVC bitstream parsing utilities.

Extracts motion vectors, residual energy, CU partition depth, and prediction modes
from HEVC compressed video without full pixel reconstruction.

Uses PyAV (FFmpeg wrapper) for prototyping. For final experiments, replace with
HM reference decoder modifications.
"""

import numpy as np
import subprocess
import json
import os
from pathlib import Path

try:
    import av
    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False


def extract_motion_vectors_pyav(video_path, frame_idx=None):
    """Extract motion vectors from HEVC video using PyAV.

    Args:
        video_path: path to HEVC-encoded video
        frame_idx: specific frame index to extract (None = all frames)

    Returns:
        list of dicts per frame, each containing:
            - 'frame_idx': int
            - 'frame_type': str ('I', 'P', 'B')
            - 'motion_vectors': np.array of shape (N, 10) where columns are:
                [source, blockw, blockh, srcx, srcy, dstx, dsty, mvx, mvy, flags]
    """
    if not HAS_PYAV:
        raise ImportError("PyAV is required. Install with: pip install av")

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    stream.codec_context.export_mvs = True

    results = []
    for idx, frame in enumerate(container.decode(stream)):
        if frame_idx is not None and idx != frame_idx:
            if idx > frame_idx:
                break
            continue

        mvs_sd = frame.side_data.get('MOTION_VECTORS')
        frame_type = frame.pict_type.name  # 'I', 'P', 'B'

        if mvs_sd is not None:
            mvs = mvs_sd.to_ndarray()
        else:
            mvs = np.zeros((0, 10), dtype=np.int32)

        results.append({
            'frame_idx': idx,
            'frame_type': frame_type,
            'motion_vectors': mvs
        })

    container.close()
    return results


def motion_vectors_to_tensor(mvs, height, width, block_size=16):
    """Convert raw motion vectors to a dense tensor.

    Args:
        mvs: np.array of shape (N, 10) from PyAV
        height, width: video dimensions
        block_size: grid resolution for tensorization

    Returns:
        np.array of shape (H//block_size, W//block_size, 2) with (mvx, mvy)
    """
    grid_h = height // block_size
    grid_w = width // block_size
    mv_tensor = np.zeros((grid_h, grid_w, 2), dtype=np.float32)
    count = np.zeros((grid_h, grid_w), dtype=np.float32)

    if len(mvs) == 0:
        return mv_tensor

    for mv in mvs:
        # mv columns: source, blockw, blockh, srcx, srcy, dstx, dsty, mvx, mvy, flags
        dstx, dsty = mv[5], mv[6]
        mvx, mvy = mv[7], mv[8]

        # Map to grid cell
        gx = min(max(int(dstx // block_size), 0), grid_w - 1)
        gy = min(max(int(dsty // block_size), 0), grid_h - 1)

        mv_tensor[gy, gx, 0] += mvx
        mv_tensor[gy, gx, 1] += mvy
        count[gy, gx] += 1

    # Average overlapping MVs
    mask = count > 0
    mv_tensor[mask, 0] /= count[mask]
    mv_tensor[mask, 1] /= count[mask]

    return mv_tensor


def extract_residual_energy_ffmpeg(video_path, output_dir, qp=27):
    """Extract residual energy approximation using FFmpeg.

    Since FFmpeg doesn't directly expose residual coefficients, we approximate
    residual energy by computing the difference between the reconstructed frame
    and the predicted frame (motion-compensated reference).

    For accurate residual extraction, use HM reference decoder.

    Args:
        video_path: path to HEVC video
        output_dir: directory to save residual energy maps
        qp: quantization parameter (affects residual magnitude)

    Returns:
        list of residual energy map paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not HAS_PYAV:
        raise ImportError("PyAV is required.")

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    prev_frame = None
    results = []

    for idx, frame in enumerate(container.decode(stream)):
        rgb = frame.to_ndarray(format='gray').astype(np.float32)

        if prev_frame is not None and frame.pict_type.name != 'I':
            # Approximate residual as frame difference (simplified)
            residual = np.abs(rgb - prev_frame)
            # Compute block-wise energy (L2 norm per 16x16 block)
            h, w = residual.shape
            bh, bw = h // 16, w // 16
            energy = np.zeros((bh, bw), dtype=np.float32)
            for by in range(bh):
                for bx in range(bw):
                    block = residual[by*16:(by+1)*16, bx*16:(bx+1)*16]
                    energy[by, bx] = np.sqrt(np.sum(block ** 2))
        else:
            h, w = rgb.shape
            energy = np.zeros((h // 16, w // 16), dtype=np.float32)

        path = output_dir / f"residual_energy_{idx:06d}.npy"
        np.save(path, energy)
        results.append(str(path))
        prev_frame = rgb

    container.close()
    return results


def extract_cu_partition_depth(video_path, frame_idx):
    """Extract CU partition depth map.

    Note: This requires HM decoder modification to expose CU partition info.
    For prototyping, we approximate partition depth from motion vector block sizes
    available through PyAV.

    Args:
        video_path: path to HEVC video
        frame_idx: frame index

    Returns:
        np.array of shape (H//16, W//16, 1) with partition depth values (0-3)
    """
    mvs_data = extract_motion_vectors_pyav(video_path, frame_idx=frame_idx)

    if not mvs_data:
        return None

    data = mvs_data[0]
    mvs = data['motion_vectors']

    # Get video dimensions
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    width = stream.codec_context.width
    height = stream.codec_context.height
    container.close()

    grid_h = height // 16
    grid_w = width // 16
    depth_map = np.zeros((grid_h, grid_w, 1), dtype=np.float32)

    if len(mvs) == 0:
        return depth_map

    for mv in mvs:
        blockw, blockh = mv[1], mv[2]
        dstx, dsty = mv[5], mv[6]

        # Approximate partition depth from block size
        # CTU=64: depth 0, 32: depth 1, 16: depth 2, 8: depth 3
        min_dim = min(blockw, blockh)
        if min_dim >= 64:
            depth = 0
        elif min_dim >= 32:
            depth = 1
        elif min_dim >= 16:
            depth = 2
        else:
            depth = 3

        gx = min(max(int(dstx // 16), 0), grid_w - 1)
        gy = min(max(int(dsty // 16), 0), grid_h - 1)
        depth_map[gy, gx, 0] = depth

    return depth_map


def extract_prediction_modes(video_path, frame_idx):
    """Extract prediction mode map.

    Note: Accurate prediction mode extraction requires HM decoder modification.
    For prototyping, we infer modes from motion vector properties.

    Mode encoding:
        0: Skip (MV present but zero residual)
        1: Merge (MV present, from merge candidate list)
        2: AMVP (MV present, from AMVP)
        3: Intra (no MV, I-frame or intra-coded block)

    Args:
        video_path: path to HEVC video
        frame_idx: frame index

    Returns:
        np.array of shape (H//16, W//16, 1) with prediction mode values
    """
    mvs_data = extract_motion_vectors_pyav(video_path, frame_idx=frame_idx)

    if not mvs_data:
        return None

    data = mvs_data[0]
    mvs = data['motion_vectors']
    frame_type = data['frame_type']

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    width = stream.codec_context.width
    height = stream.codec_context.height
    container.close()

    grid_h = height // 16
    grid_w = width // 16
    mode_map = np.zeros((grid_h, grid_w, 1), dtype=np.float32)

    if frame_type == 'I':
        mode_map[:] = 3  # All intra
        return mode_map

    # Default to intra for blocks without MVs
    mode_map[:] = 3

    for mv in mvs:
        dstx, dsty = mv[5], mv[6]
        mvx, mvy = mv[7], mv[8]
        flags = mv[9]

        gx = min(max(int(dstx // 16), 0), grid_w - 1)
        gy = min(max(int(dsty // 16), 0), grid_h - 1)

        # Approximate mode from MV properties
        if mvx == 0 and mvy == 0:
            mode_map[gy, gx, 0] = 0  # Skip
        elif flags == 0:
            mode_map[gy, gx, 0] = 1  # Merge-like
        else:
            mode_map[gy, gx, 0] = 2  # AMVP-like

    return mode_map


def extract_all_primitives(video_path, output_dir):
    """Extract all HEVC codec primitives for an entire video.

    Saves per-frame .npz files containing:
        - mv_tensor: (H//16, W//16, 2) motion vectors
        - residual_energy: (H//16, W//16, 1) residual energy
        - partition_depth: (H//16, W//16, 1) CU partition depth
        - pred_mode: (H//16, W//16, 1) prediction mode
        - frame_type: str ('I', 'P', 'B')

    Args:
        video_path: path to HEVC video
        output_dir: directory to save features
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not HAS_PYAV:
        raise ImportError("PyAV is required.")

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    stream.codec_context.export_mvs = True
    width = stream.codec_context.width
    height = stream.codec_context.height

    grid_h = height // 16
    grid_w = width // 16

    prev_gray = None

    for idx, frame in enumerate(container.decode(stream)):
        frame_type = frame.pict_type.name

        # Motion vectors
        mvs_sd = frame.side_data.get('MOTION_VECTORS')
        if mvs_sd is not None:
            mvs = mvs_sd.to_ndarray()
            mv_tensor = motion_vectors_to_tensor(mvs, height, width, block_size=16)
        else:
            mv_tensor = np.zeros((grid_h, grid_w, 2), dtype=np.float32)

        # Residual energy approximation
        gray = frame.to_ndarray(format='gray').astype(np.float32)
        if prev_gray is not None and frame_type != 'I':
            residual = np.abs(gray - prev_gray)
            energy = np.zeros((grid_h, grid_w, 1), dtype=np.float32)
            for by in range(grid_h):
                for bx in range(grid_w):
                    block = residual[by*16:(by+1)*16, bx*16:(bx+1)*16]
                    energy[by, bx, 0] = np.sqrt(np.sum(block ** 2))
        else:
            energy = np.zeros((grid_h, grid_w, 1), dtype=np.float32)

        # CU partition depth (approximated from MV block sizes)
        depth_map = np.zeros((grid_h, grid_w, 1), dtype=np.float32)
        if mvs_sd is not None:
            for mv in mvs:
                blockw, blockh = mv[1], mv[2]
                dstx, dsty = mv[5], mv[6]
                min_dim = min(blockw, blockh)
                depth = 3 if min_dim < 16 else (2 if min_dim < 32 else (1 if min_dim < 64 else 0))
                gx = min(max(int(dstx // 16), 0), grid_w - 1)
                gy = min(max(int(dsty // 16), 0), grid_h - 1)
                depth_map[gy, gx, 0] = depth

        # Prediction mode (approximated)
        mode_map = np.full((grid_h, grid_w, 1), 3, dtype=np.float32)
        if frame_type != 'I' and mvs_sd is not None:
            for mv in mvs:
                dstx, dsty = mv[5], mv[6]
                mvx, mvy = mv[7], mv[8]
                gx = min(max(int(dstx // 16), 0), grid_w - 1)
                gy = min(max(int(dsty // 16), 0), grid_h - 1)
                if mvx == 0 and mvy == 0:
                    mode_map[gy, gx, 0] = 0
                else:
                    mode_map[gy, gx, 0] = 2

        # Save
        np.savez_compressed(
            output_dir / f"frame_{idx:06d}.npz",
            mv_tensor=mv_tensor,
            residual_energy=energy,
            partition_depth=depth_map,
            pred_mode=mode_map,
            frame_type=frame_type
        )

        prev_gray = gray

    container.close()
