"""Validate PyAV bitstream motion vectors against the block-matched reference.

For a handful of sample videos and specific P-frame indices:
  1. Load block-matched MVs from the pre-computed `features/` NPZs.
  2. Freshly extract PyAV MVs from the same HEVC file via
     `extract_features_pyav.extract_real_mvs_for_frame_idx`.
  3. Render side-by-side quiver plots (block-matched = blue, PyAV = red) over
     the decoded RGB frame.
  4. Report per-video / overall statistics (mean magnitude, non-zero fraction,
     cosine similarity, endpoint error) and a PASS / AMBIGUOUS / FAIL verdict.

Pass criteria (per `report/pyav_mv_design.md` section 5.4):
  PASS       : mean cosine sim >= 0.7 AND median endpoint error <= 5 px.
  AMBIGUOUS  : mean cosine sim in [0.3, 0.7].  Try toggling --flip_sign and/or
               --dst_is_topleft.
  FAIL       : mean cosine sim < 0.3 or median endpoint error > 20 px (or no
               MVs were emitted at all).
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np

# Make data/ importable whether run as module or script
sys.path.insert(0, str(Path(__file__).parent))

import av  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from extract_features_pyav import (  # noqa: E402
    extract_real_mvs_for_frame_idx,
    open_hevc_with_mvs,
    extract_real_mvs_single_frame,
    iter_proxy_mvs_for_frames,
)


def _decode_rgb_frame(hevc_path, frame_idx):
    """Decode frame `frame_idx` of `hevc_path` and return it as an RGB ndarray."""
    container = av.open(str(hevc_path))
    stream = container.streams.video[0]
    rgb = None
    try:
        for idx, frame in enumerate(container.decode(stream)):
            if idx == frame_idx:
                rgb = frame.to_ndarray(format='rgb24')
                break
    finally:
        container.close()
    return rgb


def _decode_all_target_frames(hevc_path, frame_indices, flip_sign,
                              dst_is_topleft, backend='h264_proxy',
                              gop_size=16):
    """Collect (RGB frame, PyAV MV grid) for each index in `frame_indices`.

    RGB always comes from a direct HEVC decode (so validation images show the
    original pixels).  MVs come from whichever backend was requested:
      hevc_direct  : pulled from HEVC side_data (usually empty upstream).
      h264_proxy   : transcode to H.264 once, read MVs from H.264 side_data.
    """
    wanted = set(int(i) for i in frame_indices)
    rgb_by_idx = {}

    # RGB frames from the original HEVC decode
    container = av.open(str(hevc_path))
    stream = container.streams.video[0]
    try:
        for idx, frame in enumerate(container.decode(stream)):
            if idx in wanted:
                rgb_by_idx[idx] = frame.to_ndarray(format='rgb24')
            if len(rgb_by_idx) == len(wanted):
                break
    finally:
        container.close()

    if backend == 'hevc_direct':
        mv_by_idx = {}
        container, stream = open_hevc_with_mvs(hevc_path)
        fh = stream.codec_context.height
        fw = stream.codec_context.width
        try:
            for idx, frame in enumerate(container.decode(stream)):
                if idx in wanted:
                    mv_by_idx[idx] = extract_real_mvs_single_frame(
                        frame, grid_h=fh // 16, grid_w=fw // 16,
                        frame_h=fh, frame_w=fw,
                        flip_sign=flip_sign, dst_is_topleft=dst_is_topleft,
                    )
                if len(mv_by_idx) == len(wanted):
                    break
        finally:
            container.close()
    else:
        mv_by_idx = iter_proxy_mvs_for_frames(
            hevc_path, wanted,
            flip_sign=flip_sign, dst_is_topleft=dst_is_topleft,
            gop_size=gop_size,
        )

    return rgb_by_idx, mv_by_idx


def _load_block_mv(block_dir, stem, frame_idx):
    """Load the `mv` array from features/<stem>/frame_XXXX.npz."""
    npz_path = Path(block_dir) / stem / f'frame_{frame_idx:04d}.npz'
    if not npz_path.exists():
        return None
    with np.load(npz_path) as d:
        return d['mv'].astype(np.float32)


# --- metrics ---------------------------------------------------------------

def _compute_stats(mv_block, mv_pyav, nz_mag_thresh=0.5):
    """Return dict of comparison metrics over cells where either MV magnitude > thresh."""
    mag_b = np.sqrt((mv_block ** 2).sum(-1))
    mag_p = np.sqrt((mv_pyav ** 2).sum(-1))
    mask = (mag_b > nz_mag_thresh) | (mag_p > nz_mag_thresh)

    stats = {
        'mean_mag_block': float(mag_b.mean()),
        'mean_mag_pyav': float(mag_p.mean()),
        'nonzero_frac_block': float((mag_b > nz_mag_thresh).mean()),
        'nonzero_frac_pyav': float((mag_p > nz_mag_thresh).mean()),
        'n_mask': int(mask.sum()),
    }

    if mask.sum() == 0:
        stats['cosine'] = float('nan')
        stats['endpoint_error_mean'] = float('nan')
        stats['endpoint_error_median'] = float('nan')
        return stats

    b = mv_block[mask]
    p = mv_pyav[mask]
    # Cosine over flattened vector pair
    bf = b.reshape(-1)
    pf = p.reshape(-1)
    denom = float(np.linalg.norm(bf) * np.linalg.norm(pf))
    stats['cosine'] = float(bf @ pf / denom) if denom > 0 else float('nan')

    ee = np.sqrt(((b - p) ** 2).sum(-1))
    stats['endpoint_error_mean'] = float(ee.mean())
    stats['endpoint_error_median'] = float(np.median(ee))
    return stats


# --- visualisation ---------------------------------------------------------

def _quiver_on_axis(ax, rgb, mv, title, color, stride=2):
    """Render a quiver of `mv` (grid_h, grid_w, 2) over `rgb` on `ax`."""
    H, W = rgb.shape[:2]
    gh, gw = mv.shape[:2]
    cell_h = H / gh
    cell_w = W / gw
    ys = np.arange(gh) * cell_h + cell_h / 2
    xs = np.arange(gw) * cell_w + cell_w / 2
    X, Y = np.meshgrid(xs, ys)
    U = mv[..., 0]
    V = mv[..., 1]

    ax.imshow(rgb)
    # Subsample for readability
    Xs = X[::stride, ::stride]
    Ys = Y[::stride, ::stride]
    Us = U[::stride, ::stride]
    Vs = V[::stride, ::stride]
    ax.quiver(Xs, Ys, Us, Vs, color=color, angles='xy',
              scale_units='xy', scale=1.0, width=0.002)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def _save_side_by_side(out_path, rgb, mv_block, mv_pyav, stem, frame_idx):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _quiver_on_axis(axes[0], rgb, mv_block,
                    f'{stem} frame {frame_idx} - block-matched', 'blue')
    _quiver_on_axis(axes[1], rgb, mv_pyav,
                    f'{stem} frame {frame_idx} - PyAV bitstream', 'red')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=90, bbox_inches='tight')
    plt.close(fig)


# --- driver ----------------------------------------------------------------

def _select_videos(hevc_dir, block_dir, num_videos, seed=0):
    """Pick `num_videos` HEVC files that also have block-matched features."""
    rng = random.Random(seed)
    hevc_dir = Path(hevc_dir)
    block_dir = Path(block_dir)
    all_hevc = sorted(hevc_dir.glob('*.hevc'))
    candidates = [h for h in all_hevc if (block_dir / h.stem).exists()]
    if not candidates:
        return []
    rng.shuffle(candidates)
    return candidates[:num_videos]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hevc_dir', default='./data/imagenetvid/hevc/qp22_gop16')
    ap.add_argument('--block_dir', default='./data/imagenetvid/features/qp22_gop16')
    ap.add_argument('--num_videos', type=int, default=5)
    ap.add_argument('--frames', type=str, default='1,5,10,15',
                    help='Comma-separated list of frame indices to sample')
    ap.add_argument('--output_dir', default='./debug/pyav_validation')
    ap.add_argument('--flip_sign', action='store_true')
    ap.add_argument('--dst_is_topleft', action='store_true')
    ap.add_argument('--backend', type=str, default='h264_proxy',
                    choices=['h264_proxy', 'hevc_direct'])
    ap.add_argument('--gop_size', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    frame_idxs = [int(x) for x in args.frames.split(',') if x.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = _select_videos(args.hevc_dir, args.block_dir, args.num_videos, args.seed)
    if not videos:
        print(f'No HEVC files in {args.hevc_dir} with matching block features.')
        return

    print(f'Sampling {len(videos)} videos, frames {frame_idxs}')
    print(f'Backend: {args.backend}  (GOP={args.gop_size})')
    print(f'Flags: flip_sign={args.flip_sign}, dst_is_topleft={args.dst_is_topleft}')
    print(f'Writing visualisations to {out_dir}\n')

    all_cos = []
    all_ee_med = []
    per_video_summaries = []

    for hf in videos:
        stem = hf.stem
        print(f'--- {stem} ---')
        rgbs, mvs_pyav = _decode_all_target_frames(
            hf, frame_idxs,
            flip_sign=args.flip_sign, dst_is_topleft=args.dst_is_topleft,
            backend=args.backend, gop_size=args.gop_size,
        )

        cos_list = []
        ee_med_list = []
        for fi in frame_idxs:
            if fi not in rgbs:
                print(f'  frame {fi}: could not decode, skip')
                continue
            mv_block = _load_block_mv(args.block_dir, stem, fi)
            if mv_block is None:
                print(f'  frame {fi}: no block-matched NPZ, skip')
                continue
            mv_pyav = mvs_pyav[fi]

            st = _compute_stats(mv_block, mv_pyav)
            print(f'  frame {fi:4d}: '
                  f'|mv_b|={st["mean_mag_block"]:.2f} '
                  f'|mv_p|={st["mean_mag_pyav"]:.2f} '
                  f'nz_b={st["nonzero_frac_block"]*100:.1f}% '
                  f'nz_p={st["nonzero_frac_pyav"]*100:.1f}% '
                  f'cos={st["cosine"]:.3f} '
                  f'EE_med={st["endpoint_error_median"]:.2f}px')

            if not np.isnan(st['cosine']):
                cos_list.append(st['cosine'])
            if not np.isnan(st['endpoint_error_median']):
                ee_med_list.append(st['endpoint_error_median'])

            _save_side_by_side(
                out_dir / f'{stem}_frame_{fi:04d}.png',
                rgbs[fi], mv_block, mv_pyav, stem, fi,
            )

        vmean_cos = float(np.mean(cos_list)) if cos_list else float('nan')
        vmed_ee = float(np.median(ee_med_list)) if ee_med_list else float('nan')
        per_video_summaries.append((stem, vmean_cos, vmed_ee))
        if cos_list:
            all_cos.extend(cos_list)
        if ee_med_list:
            all_ee_med.extend(ee_med_list)
        print(f'  video mean cos={vmean_cos:.3f} median EE={vmed_ee:.2f}px')

    print('\n=== Summary ===')
    for stem, c, e in per_video_summaries:
        print(f'  {stem}: cos={c:.3f}, EE_med={e:.2f}px')
    overall_cos = float(np.mean(all_cos)) if all_cos else float('nan')
    overall_ee = float(np.median(all_ee_med)) if all_ee_med else float('nan')
    print(f'\nOverall: mean cosine={overall_cos:.3f}, '
          f'median endpoint error={overall_ee:.2f}px')

    # Verdict
    pyav_nonzero_any = any(
        (s[1] != 0.0 and not np.isnan(s[1])) for s in per_video_summaries
    )
    if not pyav_nonzero_any and all_cos == []:
        # no PyAV MVs at all across any sample
        verdict = 'FAIL'
        detail = ('PyAV emitted no MVs for any sampled frame. Your FFmpeg '
                  'build likely lacks HEVC export_mvs support; this is a '
                  'decoder-capability issue, not an extraction-code bug.')
    elif np.isnan(overall_cos):
        verdict = 'FAIL'
        detail = 'No non-zero cells across samples; cannot compare.'
    elif overall_cos >= 0.7 and overall_ee <= 5.0:
        verdict = 'PASS'
        detail = 'Real HEVC MVs agree with block-matched reference.'
    elif overall_cos < 0.3 or overall_ee > 20.0:
        verdict = 'FAIL'
        detail = ('MV fields disagree strongly. Check decoder support, or '
                  'try toggling --flip_sign / --dst_is_topleft.')
    else:
        verdict = 'AMBIGUOUS'
        detail = ('Partial agreement. Re-run with --flip_sign and/or '
                  '--dst_is_topleft to probe sign / center conventions.')

    print(f'\nVerdict: {verdict}')
    print(f'  {detail}')


if __name__ == '__main__':
    main()
