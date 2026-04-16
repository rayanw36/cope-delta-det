"""Re-encode ImageNet VID JPEG frame sequences into HEVC clips.

ImageNet VID ships as individual JPEGs per frame. For the CoPE-Δ-Det pipeline
we need HEVC-encoded video files so that ``extract_features.py`` can operate
on exactly the same compressed-domain primitives used for BDD100K.

Each snippet (a directory of ``%06d.JPEG``) becomes a single ``.hevc`` file
whose name is the flattened video_name used by ``prepare_imagenetvid.py``
(slashes replaced with ``__``).

Uses ``ffmpeg`` with the ``hevc_nvenc`` encoder when available (GPU), falling
back to ``libx265`` on CPU. Settings match ``data/encode_hevc.py``:
QP=22, GOP=16, no B-frames, keyint_min=gop, sc_threshold=0.

Usage:
    python data/encode_imagenetvid_hevc.py \\
        --vid_root ./data/imagenetvid \\
        --qp 22 --gop 16 --workers 4
"""

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path
from multiprocessing import Pool


def _find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    # Fallback to the path observed in encode_hevc.py on this machine
    fallback = os.path.expanduser(
        r"~\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe"
    )
    return fallback if os.path.exists(fallback) else "ffmpeg"


FFMPEG = _find_ffmpeg()


def _has_nvenc():
    try:
        r = subprocess.run([FFMPEG, "-hide_banner", "-encoders"],
                           capture_output=True, text=True, timeout=10)
        return "hevc_nvenc" in r.stdout
    except Exception:
        return False


USE_NVENC = _has_nvenc()


def video_name_from_dir(split, snippet_dir, vid_root_data):
    """Derive the flat video_name from a snippet directory path.

    snippet_dir: Path to the JPEG directory
    vid_root_data: Path to ILSVRC2015/Data/VID (for making rel path)
    """
    rel = snippet_dir.relative_to(vid_root_data).as_posix()
    # e.g. 'train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00000000'
    return rel.replace('/', '__')


def encode_one(task):
    snippet_dir, out_path, qp, gop = task
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
        return ('skip', out_path, 0.0)

    # Verify the snippet has a sequential %06d.JPEG pattern
    first = snippet_dir / "000000.JPEG"
    if not first.exists():
        # Some snippets may be 1-indexed; fall back to auto-detect
        jpegs = sorted(snippet_dir.glob("*.JPEG"))
        if not jpegs:
            return ('empty', str(snippet_dir), 0.0)
        start = int(jpegs[0].stem)
    else:
        start = 0

    input_pattern = str(snippet_dir / "%06d.JPEG")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    common = [
        FFMPEG, "-hide_banner", "-loglevel", "error",
        "-framerate", "30",
        "-start_number", str(start),
        "-i", input_pattern,
    ]
    if USE_NVENC:
        enc = [
            "-c:v", "hevc_nvenc", "-preset", "p6", "-tune", "hq",
            "-rc", "constqp", "-qp", str(qp),
            "-g", str(gop), "-keyint_min", str(gop),
            "-bf", "0", "-sc_threshold", "0",
        ]
    else:
        enc = [
            "-c:v", "libx265", "-preset", "medium",
            "-x265-params",
            f"qp={qp}:keyint={gop}:min-keyint={gop}:bframes=0:scenecut=0",
        ]
    cmd = common + enc + ["-pix_fmt", "yuv420p", "-y", out_path]

    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.time() - t0
        if r.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) < 1024:
            return ('fail', f"{snippet_dir}: {r.stderr[:400]}", dt)
        return ('ok', out_path, dt)
    except Exception as e:
        return ('fail', f"{snippet_dir}: {e}", time.time() - t0)


def gather_tasks(vid_root, qp, gop):
    """Enumerate every leaf snippet directory under ILSVRC2015/Data/VID."""
    data_root = Path(vid_root) / "ILSVRC2015" / "Data" / "VID"
    out_root = Path(vid_root) / "hevc" / f"qp{qp}_gop{gop}"

    tasks = []
    for split in ("train", "val"):
        split_root = data_root / split
        if not split_root.exists():
            print(f"  (missing) {split_root}")
            continue

        if split == "train":
            # train has one extra level: ILSVRC2015_VID_train_0000/
            bundles = list(split_root.iterdir())
            snippet_dirs = []
            for b in bundles:
                if b.is_dir():
                    snippet_dirs.extend([d for d in b.iterdir() if d.is_dir()])
        else:
            snippet_dirs = [d for d in split_root.iterdir() if d.is_dir()]

        for sd in sorted(snippet_dirs):
            name = video_name_from_dir(split, sd, data_root)
            out_path = out_root / f"{name}.hevc"
            tasks.append((sd, str(out_path), qp, gop))

    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid_root", default="./data/imagenetvid")
    parser.add_argument("--qp", type=int, default=22)
    parser.add_argument("--gop", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0,
                        help="Debug: encode only the first N snippets (0 = all)")
    args = parser.parse_args()

    print(f"ffmpeg: {FFMPEG}")
    print(f"nvenc available: {USE_NVENC}")

    tasks = gather_tasks(args.vid_root, args.qp, args.gop)
    print(f"snippet count: {len(tasks)}")
    if args.limit > 0:
        tasks = tasks[: args.limit]
        print(f"limited to first {len(tasks)} snippets")

    done = ok = fail = skip = 0
    t0 = time.time()
    with Pool(args.workers) as pool:
        for status, info, dt in pool.imap_unordered(encode_one, tasks):
            done += 1
            if status == 'ok':
                ok += 1
            elif status == 'skip':
                skip += 1
            else:
                fail += 1
                if fail <= 5:
                    print(f"  [FAIL] {info}")
            if done % 50 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-3)
                eta = (len(tasks) - done) / max(rate, 1e-3)
                print(f"  {done}/{len(tasks)}  ok={ok} skip={skip} fail={fail}  "
                      f"rate={rate:.1f}/s  eta={eta/60:.1f}min", flush=True)

    print(f"\nDone. ok={ok} skip={skip} fail={fail} / total={len(tasks)}")


if __name__ == "__main__":
    main()
