"""Extract HEVC codec primitives using real codec MVs via H.264 proxy.

Since FFmpeg's HEVC decoder does NOT populate MOTION_VECTORS side_data,
we use an H.264 proxy approach:
  1. Decode the HEVC source to get raw frames
  2. Encode frames to H.264 in-memory with matching GOP settings (ultrafast, same g=GOP)
  3. Decode the H.264 stream with export_mvs enabled
  4. Extract real codec MVs from side_data['MOTION_VECTORS']

This produces genuine codec MVs (not block-matching estimates), giving the
Delta Encoder access to the actual motion field the codec computed.

Output format is IDENTICAL to extract_features.py:
  - mv:         (grid_h, grid_w, 2)  float32  motion vectors (dx, dy)
  - res_energy: (grid_h, grid_w, 1)  float32  block residual energy
  - part_depth: (grid_h, grid_w, 1)  int8     partition depth (0-3)
  - pred_mode:  (grid_h, grid_w, 1)  int8     prediction mode (0=skip, 1=merge, 2=AMVP)

Note: Residual energy, partition depth, and prediction modes are still estimated
from frame differences (same as block-matching version), since H.264 proxy only
gives us MVs. The key improvement is in the MV quality.
"""

import os
import io
import numpy as np
import argparse
from pathlib import Path
import time
import tempfile
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed

import av
import torch
import torch.nn.functional as F


# ─── MV Extraction from H.264 side_data ─────────────────────────────────────

def mvs_side_data_to_dense(mv_array, grid_h, grid_w, block_size=16):
    """Convert sparse MV side_data entries to a dense (grid_h, grid_w, 2) map.

    mv_array is a structured numpy array with fields:
        source, w, h, src_x, src_y, dst_x, dst_y, flags, motion_x, motion_y, motion_scale

    We aggregate MVs into a block_size grid by averaging all MVs that fall
    within each grid cell. motion_x/motion_y are in quarter-pel units
    (divide by motion_scale for pixel units).
    """
    mv_map = np.zeros((grid_h, grid_w, 2), dtype=np.float32)
    mv_count = np.zeros((grid_h, grid_w), dtype=np.float32)

    if mv_array is None or len(mv_array) == 0:
        return mv_map

    # Filter to forward-predicted MVs only (source=-1 means future ref)
    # source >= 0 means past reference frame
    src = mv_array['source']
    mask = src >= 0
    if not mask.any():
        # Fall back to all MVs if no forward ones
        mask = np.ones(len(mv_array), dtype=bool)

    filtered = mv_array[mask]

    # Get pixel-unit motion vectors
    scale = filtered['motion_scale'].astype(np.float32)
    scale[scale == 0] = 1  # Avoid division by zero
    mx = filtered['motion_x'].astype(np.float32) / scale
    my = filtered['motion_y'].astype(np.float32) / scale

    # dst_x, dst_y is the position of the block in the current frame
    dst_x = filtered['dst_x'].astype(np.int32)
    dst_y = filtered['dst_y'].astype(np.int32)
    bw = filtered['w'].astype(np.int32)
    bh = filtered['h'].astype(np.int32)

    # Map each MV to the grid cell(s) it covers
    for i in range(len(filtered)):
        # Center of the block
        cx = dst_x[i] + bw[i] // 2
        cy = dst_y[i] + bh[i] // 2

        gx = cx // block_size
        gy = cy // block_size

        if 0 <= gy < grid_h and 0 <= gx < grid_w:
            mv_map[gy, gx, 0] += mx[i]
            mv_map[gy, gx, 1] += my[i]
            mv_count[gy, gx] += 1

    # Average where we had multiple MVs per cell
    valid = mv_count > 0
    mv_map[valid, 0] /= mv_count[valid]
    mv_map[valid, 1] /= mv_count[valid]

    return mv_map


# ─── GPU-based residual/depth/mode estimation ───────────────────────────────

def compute_residual_energy_gpu(prev_gray_t, curr_gray_t, block_size=16):
    """Block-wise residual energy on GPU."""
    diff_sq = torch.abs(curr_gray_t - prev_gray_t) ** 2
    t = diff_sq.unsqueeze(0).unsqueeze(0)
    block_sum = F.avg_pool2d(t, kernel_size=block_size, stride=block_size) * (block_size ** 2)
    energy = torch.sqrt(block_sum).squeeze()
    return energy


def compute_partition_depth_gpu(mv_mag, grid_h, grid_w):
    """Estimate partition depth from local MV variance."""
    mag_pad = F.pad(mv_mag.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate')
    mean_x = F.avg_pool2d(mag_pad, kernel_size=3, stride=1, padding=0)
    mean_x2 = F.avg_pool2d(mag_pad ** 2, kernel_size=3, stride=1, padding=0)
    local_var = (mean_x2 - mean_x ** 2).squeeze()

    depth = torch.zeros_like(local_var, dtype=torch.int8)
    depth[local_var >= 1.0] = 1
    depth[local_var >= 5.0] = 2
    depth[local_var >= 20.0] = 3
    return depth


def compute_pred_mode_gpu(mv_mag, res_energy):
    """Estimate prediction mode from MV magnitude and residual energy."""
    mode = torch.full_like(mv_mag, 2, dtype=torch.int8)  # Default: AMVP
    mode[mv_mag < 2.0] = 1   # Merge-like
    mode[(mv_mag < 0.5) & (res_energy < 10.0)] = 0  # Skip
    return mode


# ─── H.264 Proxy MV Extraction Pipeline ─────────────────────────────────────

def extract_h264_proxy_mvs(video_path, gop_size=16):
    """Decode HEVC video, re-encode to H.264, extract real codec MVs.

    Returns:
        frames_gray: list of numpy arrays (H, W) uint8 grayscale frames
        frame_mvs:   list of (grid_h, grid_w, 2) float32 MV maps (None for I-frames)
        frame_types: list of str ('I', 'P', 'B')
    """
    # Step 1: Decode HEVC source to raw frames
    container_in = av.open(str(video_path))
    stream_in = container_in.streams.video[0]
    height = stream_in.codec_context.height
    width = stream_in.codec_context.width
    grid_h = height // 16
    grid_w = width // 16

    raw_frames = []
    for frame in container_in.decode(stream_in):
        raw_frames.append(frame)
    container_in.close()

    if not raw_frames:
        return [], [], []

    # Step 2: Encode to H.264 temp file with matching GOP
    tmp_path = tempfile.mktemp(suffix='.h264')
    try:
        container_out = av.open(tmp_path, mode='w', format='h264')
        stream_out = container_out.add_stream('libx264', rate=30)
        stream_out.width = width
        stream_out.height = height
        stream_out.pix_fmt = 'yuv420p'
        stream_out.options = {
            'preset': 'ultrafast',
            'g': str(gop_size),
            'sc_threshold': '0',  # No scene-cut to match HEVC GOP structure
            'bf': '0',  # No B-frames for simpler MV extraction
        }

        for raw_frame in raw_frames:
            # Convert to yuv420p for H.264 encoding
            yuv_frame = raw_frame.reformat(format='yuv420p')
            for packet in stream_out.encode(yuv_frame):
                container_out.mux(packet)

        # Flush encoder
        for packet in stream_out.encode():
            container_out.mux(packet)
        container_out.close()

        # Step 3: Decode H.264 with export_mvs enabled
        container_mv = av.open(tmp_path)
        stream_mv = container_mv.streams.video[0]
        stream_mv.codec_context.options = {'flags2': '+export_mvs'}

        frames_gray = []
        frame_mvs = []
        frame_types = []

        for frame in container_mv.decode(stream_mv):
            # Get grayscale
            gray = frame.to_ndarray(format='gray').squeeze()
            frames_gray.append(gray)

            # Get frame type
            ptype = frame.pict_type
            if isinstance(ptype, int):
                ftype = {1: 'I', 2: 'P', 3: 'B'}.get(ptype, 'P')
            else:
                ftype = getattr(ptype, 'name', 'P')
            frame_types.append(ftype)

            # Extract MVs from side_data
            sd = frame.side_data
            if sd and 'MOTION_VECTORS' in sd:
                mv_data = sd['MOTION_VECTORS'].to_ndarray()
                mv_dense = mvs_side_data_to_dense(mv_data, grid_h, grid_w, block_size=16)
                frame_mvs.append(mv_dense)
            else:
                frame_mvs.append(None)

        container_mv.close()

    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return frames_gray, frame_mvs, frame_types


# ─── Full Video Processing ───────────────────────────────────────────────────

def process_video_features(video_path, output_dir, device='cuda', gop_size=16):
    """Extract features from a single video using H.264 proxy MVs + GPU residuals."""
    base_name = Path(video_path).stem
    vid_output_dir = Path(output_dir) / base_name

    # Skip if already extracted
    if vid_output_dir.exists():
        existing = len(list(vid_output_dir.glob("*.npz")))
        if existing > 100:
            return
    vid_output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # Extract H.264 proxy MVs for ALL frames
    try:
        frames_gray, frame_mvs, frame_types = extract_h264_proxy_mvs(
            video_path, gop_size=gop_size
        )
    except Exception as e:
        print(f"  Error extracting MVs from {video_path}: {e}")
        return

    num_frames = len(frames_gray)
    if num_frames == 0:
        print(f"  No frames decoded from {video_path}")
        return

    height, width = frames_gray[0].shape
    grid_h = height // 16
    grid_w = width // 16

    # Process each frame: combine H.264 MVs with GPU-computed residuals
    prev_gray_t = None

    for idx in range(num_frames):
        gray_t = torch.from_numpy(frames_gray[idx].astype(np.float32)).to(device)

        # Get MV map (from H.264 proxy or zeros for I-frames)
        if frame_mvs[idx] is not None:
            mv_np = frame_mvs[idx]
        else:
            mv_np = np.zeros((grid_h, grid_w, 2), dtype=np.float32)

        # Compute residual features on GPU
        if prev_gray_t is not None and frame_types[idx] != 'I':
            res_energy = compute_residual_energy_gpu(prev_gray_t, gray_t, block_size=16)

            mv_t = torch.from_numpy(mv_np).to(device)
            mv_mag = torch.sqrt(mv_t[:, :, 0] ** 2 + mv_t[:, :, 1] ** 2)

            part_depth = compute_partition_depth_gpu(mv_mag, grid_h, grid_w)
            pred_mode = compute_pred_mode_gpu(mv_mag, res_energy)

            res_np = res_energy.unsqueeze(-1).cpu().numpy()
            depth_np = part_depth.unsqueeze(-1).cpu().numpy()
            mode_np = pred_mode.unsqueeze(-1).cpu().numpy()
        else:
            # I-frame: zero features
            res_np = np.zeros((grid_h, grid_w, 1), dtype=np.float32)
            depth_np = np.zeros((grid_h, grid_w, 1), dtype=np.int8)
            mode_np = np.full((grid_h, grid_w, 1), 3, dtype=np.int8)

        # Save
        save_path = str(vid_output_dir / f"frame_{idx:04d}.npz")
        np.savez(save_path,
                 mv=mv_np,
                 res_energy=res_np,
                 part_depth=depth_np,
                 pred_mode=mode_np)

        prev_gray_t = gray_t

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t_start
            fps = (idx + 1) / elapsed
            print(f"    {idx + 1}/{num_frames} frames ({fps:.1f} fps)...")

    elapsed = time.time() - t_start
    fps = num_frames / elapsed if elapsed > 0 else 0
    print(f"  {base_name}: {num_frames} frames in {elapsed:.1f}s ({fps:.1f} fps)")


def process_video_wrapper(args):
    """Wrapper for thread pool."""
    video_path, output_dir, device, gop_size = args
    try:
        process_video_features(video_path, output_dir, device=device, gop_size=gop_size)
    except Exception as e:
        print(f"  ERROR processing {Path(video_path).stem}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract features using H.264 proxy MVs (real codec MVs)"
    )
    parser.add_argument("--hevc_dir", default="./data/bdd100k/hevc")
    parser.add_argument("--output_dir", default="./data/bdd100k/features_pyav")
    parser.add_argument("--configs", type=str, nargs="+", default=[],
                        help="Specific configs to process, e.g. 'qp22_gop16'")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrent videos (default 1 — H.264 proxy is more memory-intensive)")
    parser.add_argument("--gop_size", type=int, default=16,
                        help="GOP size for H.264 proxy encoding (should match HEVC GOP)")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Concurrent workers: {args.workers}")
    print(f"H.264 proxy GOP size: {args.gop_size}")
    print(f"NOTE: Using H.264 proxy approach for real codec MVs")
    print(f"      (HEVC decoder does not export MVs via side_data)")

    in_dir = Path(args.hevc_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        print(f"HEVC directory not found: {in_dir}")
        return

    for qp_gop_dir in sorted(in_dir.iterdir()):
        if not qp_gop_dir.is_dir():
            continue
        if args.configs and qp_gop_dir.name not in args.configs:
            continue

        print(f"\n{'='*60}")
        print(f"Processing config: {qp_gop_dir.name}")
        print(f"{'='*60}")
        feat_set_dir = out_dir / qp_gop_dir.name

        hevc_files = sorted(qp_gop_dir.glob("*.hevc"))
        total = len(hevc_files)
        print(f"  Found {total} HEVC files")

        # Count already-done
        done = 0
        todo_files = []
        for hf in hevc_files:
            vdir = feat_set_dir / hf.stem
            if vdir.exists() and len(list(vdir.glob("*.npz"))) > 100:
                done += 1
            else:
                todo_files.append(hf)

        print(f"  Already extracted: {done}/{total}")
        print(f"  Remaining: {len(todo_files)}")

        if not todo_files:
            print("  Nothing to do, skipping.")
            continue

        t0 = time.time()
        tasks = [(str(f), str(feat_set_dir), device, args.gop_size) for f in todo_files]

        if args.workers <= 1:
            for task in tasks:
                process_video_wrapper(task)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(process_video_wrapper, t) for t in tasks]
                for f in as_completed(futures):
                    f.result()

        elapsed = time.time() - t0
        print(f"\n  Config {qp_gop_dir.name} complete: {len(todo_files)} videos in {elapsed:.1f}s")

    print("\nAll extraction complete.")


if __name__ == '__main__':
    main()
