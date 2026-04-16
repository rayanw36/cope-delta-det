"""Extract HEVC codec primitives for CoPE-Δ-Det — GPU-accelerated version.

Extracts Motion Vectors, Residual Energy, Partition Depth and Prediction Modes
from HEVC-encoded video using PyAV for decoding and PyTorch GPU for fast computation.

Motion vectors are estimated via GPU-accelerated block correlation matching.
All block-level operations are fully vectorized (no Python loops in hot path).

Multi-video concurrent processing to maximize GPU utilization.
"""

import os
import numpy as np
import argparse
from pathlib import Path
import time
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import av
    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False

import torch
import torch.nn.functional as F


# ─── GPU Feature Computation Functions ───────────────────────────────────────

def compute_block_mvs_gpu(prev_gray_t, curr_gray_t, block_size=16, search_range=16, device='cuda'):
    """Fully-vectorized GPU block motion estimation.

    Processes one row of dy offsets at a time (all dx in parallel),
    using avg_pool2d for block-level SAD. Minimal Python loops.
    Returns (grid_h, grid_w, 2) motion vectors.
    """
    H, W = curr_gray_t.shape
    grid_h = H // block_size
    grid_w = W // block_size
    sr = search_range
    bs2 = block_size * block_size

    # Pad previous frame
    prev_padded = F.pad(prev_gray_t.unsqueeze(0).unsqueeze(0),
                        (sr, sr, sr, sr), mode='constant', value=0).squeeze()

    # Generate offsets (step=2)
    offsets_y = torch.arange(-sr, sr + 1, 2, device=device)
    offsets_x = torch.arange(-sr, sr + 1, 2, device=device)
    n_oy = len(offsets_y)
    n_ox = len(offsets_x)

    curr_expanded = curr_gray_t.unsqueeze(0)  # (1, H, W)

    best_sad = torch.full((grid_h, grid_w), float('inf'), device=device)
    best_idx = torch.zeros(grid_h, grid_w, dtype=torch.long, device=device)

    offset_counter = 0
    for iy in range(n_oy):
        dy = offsets_y[iy].item()
        # Extract the y-shifted row and unfold along x to get all dx offsets at once
        # row shape: (H, W + 2*sr), unfold(1, W, 2) → (H, n_ox, W), permute → (n_ox, H, W)
        row = prev_padded[sr + dy:sr + dy + H, :]
        shifted_batch = row.unfold(1, W, 2).permute(1, 0, 2).contiguous()  # (n_ox, H, W)

        # Batch SAD: all dx offsets for this dy in one call
        abs_diff = torch.abs(shifted_batch - curr_expanded)
        block_sad = F.avg_pool2d(abs_diff.unsqueeze(1), kernel_size=block_size,
                                  stride=block_size).squeeze(1) * bs2

        chunk_best_sad, chunk_best_ix = block_sad.min(dim=0)
        improved = chunk_best_sad < best_sad
        if improved.any():
            best_sad[improved] = chunk_best_sad[improved]
            best_idx[improved] = (offset_counter + chunk_best_ix[improved]).long()
        offset_counter += n_ox

    # Convert flat index back to (dy, dx) offsets
    best_iy = best_idx // n_ox
    best_ix = best_idx % n_ox
    mv_dy = offsets_y[best_iy.flatten()].view(grid_h, grid_w).float()
    mv_dx = offsets_x[best_ix.flatten()].view(grid_h, grid_w).float()

    mv_map = torch.stack([mv_dx, mv_dy], dim=-1)
    return mv_map


def compute_residual_energy_gpu(prev_gray_t, curr_gray_t, block_size=16):
    """Compute block-wise residual energy on GPU. Fully vectorized."""
    diff_sq = torch.abs(curr_gray_t - prev_gray_t) ** 2
    t = diff_sq.unsqueeze(0).unsqueeze(0)
    block_sum = F.avg_pool2d(t, kernel_size=block_size, stride=block_size) * (block_size ** 2)
    energy = torch.sqrt(block_sum).squeeze()
    return energy


def compute_partition_depth_gpu(mv_mag, grid_h, grid_w):
    """Estimate partition depth from local MV variance. Vectorized."""
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
    """Estimate prediction mode from MV magnitude and residual energy. Vectorized."""
    mode = torch.full_like(mv_mag, 2, dtype=torch.int8)  # Default: AMVP
    mode[mv_mag < 2.0] = 1   # Merge-like
    mode[(mv_mag < 0.5) & (res_energy < 10.0)] = 0  # Skip
    return mode


# ─── Pipelined Video Processing ─────────────────────────────────────────────

def _decode_frames_thread(video_path, frame_queue, stop_event):
    """Decode frames in a background thread, pushing (idx, frame_type, gray_np) to queue.

    This runs on CPU and releases the GIL during PyAV decode and numpy operations,
    allowing the GPU compute thread to run in parallel.
    """
    try:
        container = av.open(str(video_path))
        stream = container.streams.video[0]

        for idx, frame in enumerate(container.decode(stream)):
            if stop_event.is_set():
                break

            ptype = frame.pict_type
            if isinstance(ptype, int):
                frame_type = {1: 'I', 2: 'P', 3: 'B'}.get(ptype, 'P')
            else:
                frame_type = getattr(ptype, 'name', 'P')

            gray_np = frame.to_ndarray(format='gray').squeeze()
            frame_queue.put((idx, frame_type, gray_np))

        container.close()
    except Exception as e:
        frame_queue.put(('error', str(e), None))
    finally:
        frame_queue.put(None)  # Sentinel


def _save_frame_thread(save_queue, stop_event):
    """Save features to disk in a background thread to avoid blocking GPU."""
    while not stop_event.is_set():
        item = save_queue.get()
        if item is None:
            break
        path, arrays = item
        np.savez(path, **arrays)  # Uncompressed — much faster than savez_compressed


def process_video_features(video_path, output_dir, device='cuda'):
    """Extract features using pipelined decode → GPU compute → save."""
    if not HAS_PYAV:
        print(f"  PyAV not installed, skipping {video_path}")
        return

    base_name = Path(video_path).stem
    vid_output_dir = Path(output_dir) / base_name

    # Skip if already extracted
    if vid_output_dir.exists():
        existing = len(list(vid_output_dir.glob("*.npz")))
        if existing > 100:
            return  # Already done
        elif existing > 0:
            pass  # Re-extract partial

    vid_output_dir.mkdir(parents=True, exist_ok=True)

    # Get video dimensions
    try:
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        actual_h = stream.codec_context.height
        actual_w = stream.codec_context.width
        container.close()
    except Exception as e:
        print(f"  Error opening {video_path}: {e}")
        return

    grid_h = actual_h // 16
    grid_w = actual_w // 16

    # Set up pipeline: decode_thread → GPU compute → save_thread
    frame_queue = Queue(maxsize=8)   # Decoded frames waiting for GPU
    save_queue = Queue(maxsize=16)   # Computed features waiting for disk save
    stop_event = threading.Event()

    decode_thread = threading.Thread(
        target=_decode_frames_thread,
        args=(video_path, frame_queue, stop_event),
        daemon=True
    )
    save_thread = threading.Thread(
        target=_save_frame_thread,
        args=(save_queue, stop_event),
        daemon=True
    )
    decode_thread.start()
    save_thread.start()

    prev_gray_t = None
    frame_count = 0
    t_start = time.time()

    while True:
        item = frame_queue.get()
        if item is None:
            break  # End of video
        if item[0] == 'error':
            print(f"  Decode error: {item[1]}")
            break

        idx, frame_type, gray_np = item

        # Transfer to GPU
        gray_t = torch.from_numpy(gray_np.astype(np.float32)).to(device, non_blocking=True)

        if prev_gray_t is not None and frame_type != 'I':
            # GPU compute
            mv_map = compute_block_mvs_gpu(prev_gray_t, gray_t, block_size=16,
                                           search_range=16, device=device)
            res_energy = compute_residual_energy_gpu(prev_gray_t, gray_t, block_size=16)
            mv_mag = torch.sqrt(mv_map[:, :, 0] ** 2 + mv_map[:, :, 1] ** 2)
            part_depth = compute_partition_depth_gpu(mv_mag, grid_h, grid_w)
            pred_mode = compute_pred_mode_gpu(mv_mag, res_energy)

            # Transfer back to CPU (non-blocking)
            mv_np = mv_map.cpu().numpy()
            res_np = res_energy.unsqueeze(-1).cpu().numpy()
            depth_np = part_depth.unsqueeze(-1).cpu().numpy()
            mode_np = pred_mode.unsqueeze(-1).cpu().numpy()
        else:
            # I-frame: zero features
            mv_np = np.zeros((grid_h, grid_w, 2), dtype=np.float32)
            res_np = np.zeros((grid_h, grid_w, 1), dtype=np.float32)
            depth_np = np.zeros((grid_h, grid_w, 1), dtype=np.int8)
            mode_np = np.full((grid_h, grid_w, 1), 3, dtype=np.int8)

        # Send to save thread (non-blocking disk I/O)
        save_path = str(vid_output_dir / f"frame_{idx:04d}.npz")
        save_queue.put((save_path, {
            'mv': mv_np,
            'res_energy': res_np,
            'part_depth': depth_np,
            'pred_mode': mode_np
        }))

        prev_gray_t = gray_t
        frame_count += 1

        if frame_count % 200 == 0:
            elapsed = time.time() - t_start
            fps = frame_count / elapsed
            print(f"    {frame_count} frames ({fps:.1f} fps)...")

    # Wait for all saves to finish
    save_queue.put(None)
    save_thread.join()
    stop_event.set()
    decode_thread.join(timeout=2)

    elapsed = time.time() - t_start
    fps = frame_count / elapsed if elapsed > 0 else 0
    print(f"  {base_name}: {frame_count} frames in {elapsed:.1f}s ({fps:.1f} fps)")


# ─── Multi-Video Parallel Processing ────────────────────────────────────────

def process_video_wrapper(args):
    """Wrapper for thread pool — processes one video."""
    video_path, output_dir, device = args
    try:
        process_video_features(video_path, output_dir, device=device)
    except Exception as e:
        print(f"  ERROR processing {Path(video_path).stem}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hevc_dir", default="./data/bdd100k/hevc")
    parser.add_argument("--output_dir", default="./data/bdd100k/features")
    parser.add_argument("--configs", type=str, nargs="+", default=[],
                        help="Specific configs to process, e.g. 'qp22_gop16'")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: 'cuda' or 'cpu'")
    parser.add_argument("--workers", type=int, default=3,
                        help="Number of videos to process concurrently (default: 3)")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Concurrent workers: {args.workers}")

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
        print(f"Processing set: {qp_gop_dir.name}")
        print(f"{'='*60}")
        feat_set_dir = out_dir / qp_gop_dir.name

        hevc_files = sorted(qp_gop_dir.glob("*.hevc"))
        total = len(hevc_files)
        print(f"  Found {total} HEVC files")

        # Count already-done videos
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
        completed = 0

        # Process videos concurrently using thread pool
        # Each thread gets its own decode pipeline but shares GPU
        tasks = [(str(hf), str(feat_set_dir), device) for hf in todo_files]

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_video_wrapper, t): t for t in tasks}
            for future in as_completed(futures):
                completed += 1
                elapsed = time.time() - t0
                videos_per_hour = completed / elapsed * 3600 if elapsed > 0 else 0
                remaining = len(todo_files) - completed
                eta_hours = remaining / videos_per_hour if videos_per_hour > 0 else 0
                if completed % 10 == 0 or completed == len(todo_files):
                    print(f"\n  Progress: {completed}/{len(todo_files)} videos "
                          f"({videos_per_hour:.0f} videos/hr, ETA: {eta_hours:.1f}h)")

        total_elapsed = time.time() - t0
        print(f"\n  Set {qp_gop_dir.name} complete: {len(todo_files)} videos in "
              f"{total_elapsed/3600:.1f}h ({len(todo_files)/total_elapsed*3600:.0f} videos/hr)")

    print("\nFeature extraction complete.")


if __name__ == "__main__":
    main()
