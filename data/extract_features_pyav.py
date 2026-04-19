"""Extract HEVC codec primitives using real codec motion vectors via PyAV.

Two backends are provided:

  * `--backend h264_proxy`  (default) — transcode HEVC -> H.264 in-memory with
    libx264 (preset=ultrafast, bf=0, matching GOP), then read real bitstream
    motion vectors from the H.264 decoder's `MOTION_VECTORS` side data.  This
    gives genuine codec MVs (not block-SAD estimates) on every FFmpeg build,
    because the H.264 decoder universally supports `export_mvs`.  The MVs are
    of course H.264's re-computed MVs rather than the original HEVC MVs, but
    they reflect the same underlying motion.

  * `--backend hevc_direct` — read MVs directly from HEVC `side_data`.  Only
    useful on FFmpeg builds whose HEVC decoder emits
    `AV_FRAME_DATA_MOTION_VECTORS`; upstream FFmpeg's hevc decoder does NOT
    as of libavcodec 62 (empirically: every P-frame yields empty side data).
    Kept in place so that when upstream adds HEVC MV export, a one-flag switch
    activates it.

The other three channels (`res_energy`, `part_depth`, `pred_mode`) are computed
from decoded luma exactly as in `extract_features.py` — those helpers are
imported unchanged.

Output NPZ schema (per frame, 720x1280 -> 45x80 grid):
  mv          (45, 80, 2)  float32  (dx, dy) in pixel units
  res_energy  (45, 80, 1)  float32
  part_depth  (45, 80, 1)  int8
  pred_mode   (45, 80, 1)  int8
"""

import argparse
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

try:
    import av
    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False

import torch

sys.path.insert(0, str(Path(__file__).parent))
from extract_features import (  # noqa: E402
    compute_residual_energy_gpu,
    compute_partition_depth_gpu,
    compute_pred_mode_gpu,
)


# FFmpeg AV_CODEC_FLAG2_EXPORT_MVS = 1 << 28
FLAG2_EXPORT_MVS = 1 << 28


# --- side_data -> grid projection ------------------------------------------

def _project_mv_entries_to_grid(mvs, frame_h, frame_w, grid_h=45, grid_w=80,
                                dst_is_topleft=False, flip_sign=False):
    """Area-weighted projection of variable-size MV entries onto a fixed grid.

    Works identically for H.264 and HEVC side_data — both codecs use the
    same AVMotionVector struct.
    """
    cell_h = frame_h / float(grid_h)
    cell_w = frame_w / float(grid_w)
    out_zero = np.zeros((grid_h, grid_w, 2), dtype=np.float32)

    if mvs is None or len(mvs) == 0:
        return out_zero

    # Forward-only (source == -1).  With bf=0 there is no backward prediction,
    # but filter anyway for safety.
    mask = mvs['source'] == -1
    mvs = mvs[mask]
    if len(mvs) == 0:
        return out_zero

    scales = mvs['motion_scale'].astype(np.float64)
    scales = np.where(scales > 0, scales, 1.0)
    mx = mvs['motion_x'].astype(np.float64) / scales
    my = mvs['motion_y'].astype(np.float64) / scales
    if flip_sign:
        mx = -mx
        my = -my

    bws = mvs['w'].astype(np.int32)
    bhs = mvs['h'].astype(np.int32)
    if dst_is_topleft:
        bxs = mvs['dst_x'].astype(np.int32)
        bys = mvs['dst_y'].astype(np.int32)
    else:
        bxs = mvs['dst_x'].astype(np.int32) - (bws // 2)
        bys = mvs['dst_y'].astype(np.int32) - (bhs // 2)

    mv_sum = np.zeros((grid_h, grid_w, 2), dtype=np.float64)
    mv_wt = np.zeros((grid_h, grid_w), dtype=np.float64)

    for mvx, mvy, bx, by, bw, bh in zip(mx, my, bxs, bys, bws, bhs):
        bx0 = max(0, int(bx))
        by0 = max(0, int(by))
        bx1 = min(frame_w, int(bx) + int(bw))
        by1 = min(frame_h, int(by) + int(bh))
        if bx1 <= bx0 or by1 <= by0:
            continue

        gy_start = max(0, int(by0 / cell_h))
        gx_start = max(0, int(bx0 / cell_w))
        gy_end = min(grid_h, int(np.ceil(by1 / cell_h)))
        gx_end = min(grid_w, int(np.ceil(bx1 / cell_w)))

        for gy in range(gy_start, gy_end):
            cy0 = gy * cell_h
            cy1 = cy0 + cell_h
            oy = min(by1, cy1) - max(by0, cy0)
            if oy <= 0:
                continue
            for gx in range(gx_start, gx_end):
                cx0 = gx * cell_w
                cx1 = cx0 + cell_w
                ox = min(bx1, cx1) - max(bx0, cx0)
                if ox <= 0:
                    continue
                w = ox * oy
                mv_sum[gy, gx, 0] += mvx * w
                mv_sum[gy, gx, 1] += mvy * w
                mv_wt[gy, gx] += w

    nz = mv_wt > 0
    out = np.zeros((grid_h, grid_w, 2), dtype=np.float32)
    out[nz, 0] = (mv_sum[nz, 0] / mv_wt[nz]).astype(np.float32)
    out[nz, 1] = (mv_sum[nz, 1] / mv_wt[nz]).astype(np.float32)
    return out


def extract_real_mvs_single_frame(frame, grid_h=45, grid_w=80,
                                  frame_h=720, frame_w=1280,
                                  flip_sign=False, dst_is_topleft=False):
    """Return (grid_h, grid_w, 2) MV grid for one decoded PyAV frame.

    I-frames and frames without MV side data yield all-zeros.
    """
    side = frame.side_data.get('MOTION_VECTORS')
    if side is None:
        return np.zeros((grid_h, grid_w, 2), dtype=np.float32)
    try:
        mvs = side.to_ndarray()
    except Exception:
        return np.zeros((grid_h, grid_w, 2), dtype=np.float32)
    return _project_mv_entries_to_grid(
        mvs, frame_h=frame_h, frame_w=frame_w,
        grid_h=grid_h, grid_w=grid_w,
        dst_is_topleft=dst_is_topleft, flip_sign=flip_sign,
    )


# --- HEVC direct backend ---------------------------------------------------

def open_hevc_with_mvs(video_path):
    """Open an HEVC file with `export_mvs` enabled."""
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    try:
        stream.thread_type = 'NONE'
        stream.thread_count = 1
    except Exception:
        pass
    ctx = stream.codec_context
    try:
        ctx.options = {'flags2': '+export_mvs'}
    except Exception:
        pass
    try:
        ctx.flags2 = ctx.flags2 | FLAG2_EXPORT_MVS
    except Exception:
        pass
    return container, stream


def extract_real_mvs_for_frame_idx(hevc_path, frame_idx,
                                   grid_h=45, grid_w=80,
                                   flip_sign=False, dst_is_topleft=False,
                                   backend='h264_proxy', gop_size=16):
    """Stand-alone helper for the validator — returns the MV grid for one frame.

    For `h264_proxy` backend this transcodes the whole video to a temp H.264
    file (expensive) and then decodes up to `frame_idx`.  For repeated calls on
    the same video the caller should instead batch frames through
    `iter_proxy_mvs_for_frames`.
    """
    if backend == 'hevc_direct':
        container, stream = open_hevc_with_mvs(hevc_path)
        fh = stream.codec_context.height
        fw = stream.codec_context.width
        mv = None
        try:
            for idx, frame in enumerate(container.decode(stream)):
                if idx == frame_idx:
                    mv = extract_real_mvs_single_frame(
                        frame, grid_h=grid_h, grid_w=grid_w,
                        frame_h=fh, frame_w=fw,
                        flip_sign=flip_sign, dst_is_topleft=dst_is_topleft,
                    )
                    break
        finally:
            container.close()
        return mv if mv is not None else np.zeros((grid_h, grid_w, 2), dtype=np.float32)

    # h264_proxy
    results = iter_proxy_mvs_for_frames(
        hevc_path, [frame_idx],
        grid_h=grid_h, grid_w=grid_w,
        flip_sign=flip_sign, dst_is_topleft=dst_is_topleft,
        gop_size=gop_size,
    )
    return results.get(frame_idx, np.zeros((grid_h, grid_w, 2), dtype=np.float32))


# --- H.264 proxy backend ---------------------------------------------------

def _pick_h264_encoder():
    """Return 'h264_nvenc' if available on this machine, else 'libx264'.

    h264_nvenc is ~5-10x faster than libx264 for the ultrafast re-encode step
    and is available whenever an NVIDIA GPU and the matching FFmpeg build are
    present.  The output is still decoded by the software H.264 decoder, which
    exports MVs normally.
    """
    try:
        av.Codec('h264_nvenc', 'w')
        return 'h264_nvenc'
    except Exception:
        return 'libx264'


_H264_ENCODER = _pick_h264_encoder()


def _transcode_hevc_to_h264(hevc_path, h264_path, gop_size=16, encoder=None):
    """Decode HEVC and re-encode to H.264 (bf=0, matching GOP).

    Uses h264_nvenc by default when available (GPU-accelerated; ~5-10x faster
    than libx264 ultrafast), otherwise falls back to libx264.  The H.264
    decoder fully supports `export_mvs` regardless of which encoder was used.
    """
    enc = encoder or _H264_ENCODER
    cin = av.open(str(hevc_path))
    sin = cin.streams.video[0]
    width = sin.codec_context.width
    height = sin.codec_context.height
    rate = sin.average_rate or 30

    cout = av.open(str(h264_path), mode='w', format='h264')
    sout = cout.add_stream(enc, rate=rate)
    sout.width = width
    sout.height = height
    sout.pix_fmt = 'yuv420p'

    if enc == 'h264_nvenc':
        sout.options = {
            'preset': 'p1',          # fastest NVENC preset (low quality, fine for MV)
            'g': str(gop_size),
            'bf': '0',               # no B-frames
            'rc': 'constqp',
            'qp': '28',
        }
    else:
        sout.options = {
            'preset': 'ultrafast',
            'g': str(gop_size),
            'keyint_min': str(gop_size),
            'sc_threshold': '0',
            'bf': '0',
            'tune': 'zerolatency',
        }

    n_frames = 0
    try:
        for frame in cin.decode(sin):
            yuv = frame.reformat(format='yuv420p')
            yuv.pts = n_frames
            for packet in sout.encode(yuv):
                cout.mux(packet)
            n_frames += 1
        for packet in sout.encode():
            cout.mux(packet)
    finally:
        cout.close()
        cin.close()
    return n_frames, width, height


def _open_h264_with_mvs(h264_path):
    """Open the transcoded H.264 file with `export_mvs` enabled."""
    container = av.open(str(h264_path))
    stream = container.streams.video[0]
    try:
        stream.thread_type = 'NONE'
        stream.thread_count = 1
    except Exception:
        pass
    ctx = stream.codec_context
    try:
        ctx.options = {'flags2': '+export_mvs'}
    except Exception:
        pass
    try:
        ctx.flags2 = ctx.flags2 | FLAG2_EXPORT_MVS
    except Exception:
        pass
    return container, stream


def iter_proxy_mvs_for_frames(hevc_path, frame_indices,
                              grid_h=None, grid_w=None,
                              flip_sign=False, dst_is_topleft=False,
                              gop_size=16, block_size=16, encoder=None):
    """H.264-proxy MV extraction for a specific set of frame indices.

    Transcodes `hevc_path` to a temp H.264 file (ultrafast libx264, bf=0,
    matching GOP), decodes it with export_mvs, and returns a dict
    {frame_idx: (grid_h, grid_w, 2) float32} for the requested indices.

    If `grid_h`/`grid_w` are None they are inferred from the actual stream
    dimensions as `height // block_size` / `width // block_size`.
    """
    wanted = set(int(i) for i in frame_indices)
    if not wanted:
        return {}
    results = {}

    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.h264')
    os.close(tmp_fd)
    try:
        _transcode_hevc_to_h264(hevc_path, tmp_path, gop_size=gop_size, encoder=encoder)
        container, stream = _open_h264_with_mvs(tmp_path)
        fh = stream.codec_context.height
        fw = stream.codec_context.width
        gh = grid_h if grid_h is not None else fh // block_size
        gw = grid_w if grid_w is not None else fw // block_size
        try:
            for idx, frame in enumerate(container.decode(stream)):
                if idx in wanted:
                    results[idx] = extract_real_mvs_single_frame(
                        frame, grid_h=gh, grid_w=gw,
                        frame_h=fh, frame_w=fw,
                        flip_sign=flip_sign, dst_is_topleft=dst_is_topleft,
                    )
                if len(results) == len(wanted):
                    break
        finally:
            container.close()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    return results


# --- per-video processing --------------------------------------------------

def _save_thread(save_q, stop_event):
    while not stop_event.is_set():
        item = save_q.get()
        if item is None:
            break
        path, arrays = item
        np.savez(path, **arrays)


def _process_video_hevc_direct(video_path, output_dir, device='cuda',
                               dst_is_topleft=False, flip_sign=False,
                               skip_existing=True):
    stem = Path(video_path).stem
    vid_out = Path(output_dir) / stem
    if skip_existing and vid_out.exists() and len(list(vid_out.glob('*.npz'))) > 100:
        return
    vid_out.mkdir(parents=True, exist_ok=True)

    try:
        container, stream = open_hevc_with_mvs(video_path)
    except Exception as e:
        print(f"  Error opening {video_path}: {e}")
        return
    fh = stream.codec_context.height
    fw = stream.codec_context.width
    grid_h = fh // 16
    grid_w = fw // 16
    if not (fw == 1280 and fh == 720):
        print(f"  WARN: {stem} dims {fw}x{fh}; expected 1280x720")

    save_q = Queue(maxsize=16)
    stop_event = threading.Event()
    saver = threading.Thread(target=_save_thread, args=(save_q, stop_event), daemon=True)
    saver.start()

    prev_gray_t = None
    frame_count = 0
    p_frames = 0
    p_with_mvs = 0
    t0 = time.time()
    try:
        for idx, frame in enumerate(container.decode(stream)):
            ptype = frame.pict_type
            ftype = ({1: 'I', 2: 'P', 3: 'B'}.get(ptype, 'P')
                     if isinstance(ptype, int) else getattr(ptype, 'name', 'P'))
            gray_np = frame.to_ndarray(format='gray').squeeze()
            gray_t = torch.from_numpy(gray_np.astype(np.float32)).to(device, non_blocking=True)

            if ftype == 'I' or prev_gray_t is None:
                mv_np = np.zeros((grid_h, grid_w, 2), dtype=np.float32)
                res_np = np.zeros((grid_h, grid_w, 1), dtype=np.float32)
                depth_np = np.zeros((grid_h, grid_w, 1), dtype=np.int8)
                mode_np = np.full((grid_h, grid_w, 1), 3, dtype=np.int8)
            else:
                p_frames += 1
                mv_np = extract_real_mvs_single_frame(
                    frame, grid_h=grid_h, grid_w=grid_w, frame_h=fh, frame_w=fw,
                    flip_sign=flip_sign, dst_is_topleft=dst_is_topleft)
                if np.any(mv_np != 0):
                    p_with_mvs += 1
                res = compute_residual_energy_gpu(prev_gray_t, gray_t, block_size=16)
                mv_mag = torch.from_numpy(
                    np.sqrt(mv_np[..., 0] ** 2 + mv_np[..., 1] ** 2)
                ).to(device, non_blocking=True)
                part = compute_partition_depth_gpu(mv_mag, grid_h, grid_w)
                mode = compute_pred_mode_gpu(mv_mag, res)
                res_np = res.unsqueeze(-1).cpu().numpy()
                depth_np = part.unsqueeze(-1).cpu().numpy()
                mode_np = mode.unsqueeze(-1).cpu().numpy()

            save_q.put((str(vid_out / f'frame_{idx:04d}.npz'), {
                'mv': mv_np.astype(np.float32),
                'res_energy': res_np.astype(np.float32),
                'part_depth': depth_np.astype(np.int8),
                'pred_mode': mode_np.astype(np.int8),
            }))
            prev_gray_t = gray_t
            frame_count += 1
    finally:
        save_q.put(None)
        saver.join()
        stop_event.set()
        container.close()

    elapsed = time.time() - t0
    fps = frame_count / elapsed if elapsed > 0 else 0.0
    frac = (p_with_mvs / p_frames) if p_frames else 0.0
    print(f"  {stem}: {frame_count} frames in {elapsed:.1f}s ({fps:.1f} fps) "
          f"[hevc_direct] P-frames with MVs: {p_with_mvs}/{p_frames} ({frac*100:.1f}%)")
    if p_frames > 0 and p_with_mvs == 0:
        print(f"  NOTE: no HEVC MV side-data. Use --backend h264_proxy "
              f"on FFmpeg builds without HEVC export_mvs.")


def _process_video_h264_proxy(video_path, output_dir, device='cuda',
                              dst_is_topleft=False, flip_sign=False,
                              skip_existing=True, gop_size=16, encoder=None):
    """Transcode HEVC -> H.264 (libx264 bf=0 g=gop_size), extract MVs from H.264.

    For the non-MV channels we use the luma of the *H.264-decoded* frames.
    Those are a one-generation re-encode of the HEVC reconstruction, so the
    residual energy differs slightly from using the original HEVC luma, but
    the pipeline stays simple (one decode, not two) and the approximation
    quality is comparable to the block-matched reference.
    """
    stem = Path(video_path).stem
    vid_out = Path(output_dir) / stem
    if skip_existing and vid_out.exists() and len(list(vid_out.glob('*.npz'))) > 100:
        return
    vid_out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Step 1: transcode to temp H.264
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.h264')
    os.close(tmp_fd)

    frame_count = 0
    p_frames = 0
    p_with_mvs = 0

    try:
        try:
            n_in, width, height = _transcode_hevc_to_h264(
                video_path, tmp_path, gop_size=gop_size, encoder=encoder)
        except Exception as e:
            print(f"  Error transcoding {video_path}: {e}")
            return
        if not (width == 1280 and height == 720):
            print(f"  WARN: {stem} dims {width}x{height}; expected 1280x720")
        grid_h = height // 16
        grid_w = width // 16

        # Step 2: decode H.264 with MV export, pipeline GPU + save
        container, stream = _open_h264_with_mvs(tmp_path)
        save_q = Queue(maxsize=16)
        stop_event = threading.Event()
        saver = threading.Thread(target=_save_thread, args=(save_q, stop_event), daemon=True)
        saver.start()

        prev_gray_t = None
        try:
            for idx, frame in enumerate(container.decode(stream)):
                ptype = frame.pict_type
                ftype = ({1: 'I', 2: 'P', 3: 'B'}.get(ptype, 'P')
                         if isinstance(ptype, int) else getattr(ptype, 'name', 'P'))
                gray_np = frame.to_ndarray(format='gray').squeeze()
                gray_t = torch.from_numpy(gray_np.astype(np.float32)).to(device, non_blocking=True)

                if ftype == 'I' or prev_gray_t is None:
                    mv_np = np.zeros((grid_h, grid_w, 2), dtype=np.float32)
                    res_np = np.zeros((grid_h, grid_w, 1), dtype=np.float32)
                    depth_np = np.zeros((grid_h, grid_w, 1), dtype=np.int8)
                    mode_np = np.full((grid_h, grid_w, 1), 3, dtype=np.int8)
                else:
                    p_frames += 1
                    mv_np = extract_real_mvs_single_frame(
                        frame, grid_h=grid_h, grid_w=grid_w,
                        frame_h=height, frame_w=width,
                        flip_sign=flip_sign, dst_is_topleft=dst_is_topleft)
                    if np.any(mv_np != 0):
                        p_with_mvs += 1
                    res = compute_residual_energy_gpu(prev_gray_t, gray_t, block_size=16)
                    mv_mag = torch.from_numpy(
                        np.sqrt(mv_np[..., 0] ** 2 + mv_np[..., 1] ** 2)
                    ).to(device, non_blocking=True)
                    part = compute_partition_depth_gpu(mv_mag, grid_h, grid_w)
                    mode = compute_pred_mode_gpu(mv_mag, res)
                    res_np = res.unsqueeze(-1).cpu().numpy()
                    depth_np = part.unsqueeze(-1).cpu().numpy()
                    mode_np = mode.unsqueeze(-1).cpu().numpy()

                save_q.put((str(vid_out / f'frame_{idx:04d}.npz'), {
                    'mv': mv_np.astype(np.float32),
                    'res_energy': res_np.astype(np.float32),
                    'part_depth': depth_np.astype(np.int8),
                    'pred_mode': mode_np.astype(np.int8),
                }))
                prev_gray_t = gray_t
                frame_count += 1
        finally:
            save_q.put(None)
            saver.join()
            stop_event.set()
            container.close()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    elapsed = time.time() - t0
    fps = frame_count / elapsed if elapsed > 0 else 0.0
    frac = (p_with_mvs / p_frames) if p_frames else 0.0
    enc_used = encoder or _H264_ENCODER
    print(f"  {stem}: {frame_count} frames in {elapsed:.1f}s ({fps:.1f} fps) "
          f"[h264_proxy/{enc_used}] P-frames with MVs: {p_with_mvs}/{p_frames} ({frac*100:.1f}%)")


def process_video_features_pyav(video_path, output_dir, device='cuda',
                                dst_is_topleft=False, flip_sign=False,
                                skip_existing=True, backend='h264_proxy',
                                gop_size=16, encoder=None):
    if not HAS_PYAV:
        print(f"  PyAV not installed, skipping {video_path}")
        return
    if backend == 'hevc_direct':
        _process_video_hevc_direct(
            video_path, output_dir, device=device,
            dst_is_topleft=dst_is_topleft, flip_sign=flip_sign,
            skip_existing=skip_existing)
    elif backend == 'h264_proxy':
        _process_video_h264_proxy(
            video_path, output_dir, device=device,
            dst_is_topleft=dst_is_topleft, flip_sign=flip_sign,
            skip_existing=skip_existing, gop_size=gop_size, encoder=encoder)
    else:
        raise ValueError(f'Unknown backend: {backend!r}')


# --- multi-video driver ----------------------------------------------------

def _worker(args):
    (video_path, out_dir, device, dst_is_topleft, flip_sign,
     skip_existing, backend, gop_size, encoder) = args
    try:
        process_video_features_pyav(
            video_path, out_dir, device=device,
            dst_is_topleft=dst_is_topleft, flip_sign=flip_sign,
            skip_existing=skip_existing, backend=backend,
            gop_size=gop_size, encoder=encoder)
    except Exception as e:
        print(f"  ERROR {Path(video_path).stem}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hevc_dir', default='./data/imagenetvid/hevc')
    parser.add_argument('--output_dir', default='./data/imagenetvid/features_pyav')
    parser.add_argument('--configs', type=str, nargs='+', default=[])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--backend', type=str, default='h264_proxy',
                        choices=['h264_proxy', 'hevc_direct'],
                        help='MV source: H.264 proxy transcode (default, always works) '
                             'or direct HEVC side_data (requires FFmpeg HEVC MV support).')
    parser.add_argument('--gop_size', type=int, default=16,
                        help='GOP size for H.264 proxy re-encode (match HEVC source).')
    parser.add_argument('--single_video', type=str, default=None,
                        help='Process only this specific HEVC file (smoke-test).')
    parser.add_argument('--flip_sign', action='store_true',
                        help='Negate MVs before projection.')
    parser.add_argument('--dst_is_topleft', action='store_true',
                        help='Treat dst_x/dst_y as top-left rather than center.')
    parser.add_argument('--no_skip_existing', action='store_true',
                        help='Re-extract even if NPZ dir already exists.')
    parser.add_argument('--encoder', type=str, default=None,
                        help='H.264 encoder for proxy transcode: h264_nvenc (GPU, default '
                             'when available) or libx264 (CPU fallback). '
                             'Auto-detected if omitted.')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    encoder = args.encoder  # None → auto-detect inside _pick_h264_encoder()
    print(f'Using device: {device}')
    if device == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'Backend: {args.backend}  (GOP={args.gop_size})')
    print(f'H.264 encoder: {encoder or _H264_ENCODER} (auto={encoder is None})')
    print(f'Workers: {args.workers}, flip_sign={args.flip_sign}, '
          f'dst_is_topleft={args.dst_is_topleft}')

    skip_existing = not args.no_skip_existing

    if args.single_video:
        hf = Path(args.single_video)
        if not hf.exists():
            print(f'File not found: {hf}')
            return
        config = hf.parent.name
        out_dir = Path(args.output_dir) / config
        out_dir.mkdir(parents=True, exist_ok=True)
        process_video_features_pyav(
            str(hf), str(out_dir), device=device,
            dst_is_topleft=args.dst_is_topleft, flip_sign=args.flip_sign,
            skip_existing=skip_existing, backend=args.backend,
            gop_size=args.gop_size, encoder=encoder)
        return

    in_dir = Path(args.hevc_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not in_dir.exists():
        print(f'HEVC directory not found: {in_dir}')
        return

    for qp_gop_dir in sorted(in_dir.iterdir()):
        if not qp_gop_dir.is_dir():
            continue
        if args.configs and qp_gop_dir.name not in args.configs:
            continue

        print(f"\n{'='*60}\nProcessing set: {qp_gop_dir.name}\n{'='*60}")
        feat_set_dir = out_dir / qp_gop_dir.name
        feat_set_dir.mkdir(parents=True, exist_ok=True)

        hevc_files = sorted(qp_gop_dir.glob('*.hevc'))
        total = len(hevc_files)
        todo = []
        done = 0
        for hf in hevc_files:
            vdir = feat_set_dir / hf.stem
            if skip_existing and vdir.exists() and len(list(vdir.glob('*.npz'))) > 100:
                done += 1
            else:
                todo.append(hf)
        print(f'  Found {total}, already extracted {done}, remaining {len(todo)}')
        if not todo:
            continue

        tasks = [(str(hf), str(feat_set_dir), device,
                  args.dst_is_topleft, args.flip_sign, skip_existing,
                  args.backend, args.gop_size, encoder) for hf in todo]

        t0 = time.time()
        completed = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_worker, t): t for t in tasks}
            for fut in as_completed(futures):
                completed += 1
                elapsed = time.time() - t0
                vph = completed / elapsed * 3600 if elapsed > 0 else 0
                eta = (len(todo) - completed) / vph if vph > 0 else 0
                if completed % 10 == 0 or completed == len(todo):
                    print(f"  Progress: {completed}/{len(todo)} "
                          f"({vph:.0f} vid/hr, ETA {eta:.1f}h)")
        print(f'  {qp_gop_dir.name}: done in {(time.time()-t0)/3600:.1f}h')

    print('\nPyAV feature extraction complete.')


if __name__ == '__main__':
    main()
