"""Validate PyAV H.264 proxy MVs against block-matched MVs on sample videos.

Compares:
1. MV magnitude distributions
2. MV direction correlation
3. Coverage (% of grid cells with non-zero MVs)
4. Frame-by-frame visual comparison stats
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
import time

from data.extract_features_pyav import extract_h264_proxy_mvs


def validate_single_video(hevc_path, block_match_dir, max_frames=50):
    """Compare PyAV MVs vs block-matched MVs for one video."""
    video_name = Path(hevc_path).stem
    bm_dir = Path(block_match_dir) / video_name

    if not bm_dir.exists():
        print(f"  No block-match features found for {video_name}")
        return None

    print(f"\n{'='*60}")
    print(f"Validating: {video_name}")
    print(f"{'='*60}")

    # Extract PyAV MVs
    t0 = time.time()
    frames_gray, frame_mvs, frame_types = extract_h264_proxy_mvs(
        hevc_path, gop_size=16
    )
    pyav_time = time.time() - t0
    print(f"  PyAV extraction: {len(frames_gray)} frames in {pyav_time:.1f}s")

    # Count MV frames
    mv_frames = sum(1 for mv in frame_mvs if mv is not None)
    print(f"  Frames with MVs: {mv_frames}/{len(frames_gray)}")
    print(f"  Frame types: I={frame_types.count('I')}, P={frame_types.count('P')}, B={frame_types.count('B')}")

    # Compare frame by frame
    stats = {
        'pyav_mag_mean': [],
        'bm_mag_mean': [],
        'pyav_coverage': [],
        'bm_coverage': [],
        'direction_corr': [],
    }

    num_compared = 0
    for idx in range(min(len(frame_mvs), max_frames)):
        bm_path = bm_dir / f"frame_{idx:04d}.npz"
        if not bm_path.exists():
            continue

        bm_data = np.load(str(bm_path))
        bm_mv = bm_data['mv']  # (45, 80, 2)

        pyav_mv = frame_mvs[idx]
        if pyav_mv is None:
            pyav_mv = np.zeros_like(bm_mv)

        # MV magnitudes
        bm_mag = np.sqrt(bm_mv[:,:,0]**2 + bm_mv[:,:,1]**2)
        pyav_mag = np.sqrt(pyav_mv[:,:,0]**2 + pyav_mv[:,:,1]**2)

        stats['bm_mag_mean'].append(bm_mag.mean())
        stats['pyav_mag_mean'].append(pyav_mag.mean())

        # Coverage (% cells with |mv| > 0.5)
        bm_cov = (bm_mag > 0.5).mean() * 100
        pyav_cov = (pyav_mag > 0.5).mean() * 100
        stats['bm_coverage'].append(bm_cov)
        stats['pyav_coverage'].append(pyav_cov)

        # Direction correlation (cosine similarity of MV vectors)
        bm_flat = bm_mv.reshape(-1, 2)
        pyav_flat = pyav_mv.reshape(-1, 2)
        # Only compare where both have non-trivial MVs
        mask = (np.linalg.norm(bm_flat, axis=1) > 0.5) & (np.linalg.norm(pyav_flat, axis=1) > 0.5)
        if mask.sum() > 10:
            bm_norm = bm_flat[mask] / (np.linalg.norm(bm_flat[mask], axis=1, keepdims=True) + 1e-8)
            pyav_norm = pyav_flat[mask] / (np.linalg.norm(pyav_flat[mask], axis=1, keepdims=True) + 1e-8)
            cos_sim = (bm_norm * pyav_norm).sum(axis=1).mean()
            stats['direction_corr'].append(cos_sim)

        num_compared += 1

    if num_compared == 0:
        print("  No frames to compare!")
        return None

    # Print summary
    print(f"\n  Compared {num_compared} frames:")
    print(f"  {'Metric':<25} {'Block-Match':>12} {'PyAV H.264':>12}")
    print(f"  {'-'*50}")
    print(f"  {'Avg MV magnitude':<25} {np.mean(stats['bm_mag_mean']):>12.2f} {np.mean(stats['pyav_mag_mean']):>12.2f}")
    print(f"  {'Avg coverage (%)':<25} {np.mean(stats['bm_coverage']):>12.1f} {np.mean(stats['pyav_coverage']):>12.1f}")
    if stats['direction_corr']:
        print(f"  {'Direction cosine sim':<25} {'---':>12} {np.mean(stats['direction_corr']):>12.3f}")

    # Show per-frame detail for first 10 P-frames
    print(f"\n  Per-frame MV magnitudes (first 10 P-frames):")
    print(f"  {'Frame':<8} {'Type':<6} {'BM mag':>8} {'PyAV mag':>10} {'BM cov%':>8} {'PyAV cov%':>10}")
    shown = 0
    for i in range(min(len(stats['bm_mag_mean']), 20)):
        if stats['bm_mag_mean'][i] > 0 or stats['pyav_mag_mean'][i] > 0:
            ftype = frame_types[i] if i < len(frame_types) else '?'
            print(f"  {i:<8} {ftype:<6} {stats['bm_mag_mean'][i]:>8.2f} {stats['pyav_mag_mean'][i]:>10.2f} "
                  f"{stats['bm_coverage'][i]:>8.1f} {stats['pyav_coverage'][i]:>10.1f}")
            shown += 1
            if shown >= 10:
                break

    return stats


if __name__ == '__main__':
    hevc_dir = Path("D:/cope-delta-det2/data/bdd100k/hevc/qp22_gop16")
    bm_feat_dir = Path("D:/cope-delta-det2/data/bdd100k/features/qp22_gop16")

    # Test on first 3 videos
    hevc_files = sorted(hevc_dir.glob("*.hevc"))[:3]

    for hf in hevc_files:
        validate_single_video(str(hf), str(bm_feat_dir), max_frames=50)

    print("\n\nValidation complete.")
