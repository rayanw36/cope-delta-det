"""One-shot script: extract PyAV (h264_proxy) features for the ImageNet VID val split only.

Reads gop_index_val.json to get the 553 val video stems, then calls
process_video_features_pyav() for each one that hasn't been extracted yet.
"""
import json
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))
from extract_features_pyav import process_video_features_pyav, _H264_ENCODER  # noqa: E402

import torch  # noqa: E402

DATA_ROOT    = Path(__file__).parent          # data/
HEVC_DIR     = DATA_ROOT / 'imagenetvid' / 'hevc'       / 'qp22_gop16'
OUT_DIR      = DATA_ROOT / 'imagenetvid' / 'features_pyav' / 'qp22_gop16'
INDEX_PATH   = DATA_ROOT / 'imagenetvid' / 'gop_index_val.json'
GOP_SIZE     = 16
WORKERS      = 4
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'


def _worker(args):
    hevc_path, out_dir, device = args
    stem = Path(hevc_path).stem
    try:
        process_video_features_pyav(
            hevc_path, str(out_dir),
            device=device, backend='h264_proxy',
            gop_size=GOP_SIZE, skip_existing=True,
            encoder=None,  # auto: h264_nvenc when available, else libx264
        )
    except Exception as e:
        print(f'  ERROR {stem}: {e}', flush=True)


def main():
    with open(INDEX_PATH) as f:
        gops = json.load(f)
    stems = sorted(set(g['video_name'] for g in gops))

    todo = []
    done = 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for s in stems:
        hf = HEVC_DIR / (s + '.hevc')
        if not hf.exists():
            print(f'  MISSING HEVC: {s}')
            continue
        vdir = OUT_DIR / s
        if vdir.exists() and len(list(vdir.glob('*.npz'))) > 10:
            done += 1
        else:
            todo.append(str(hf))

    print(f'Val videos: {len(stems)} total, {done} already done, {len(todo)} to extract')
    print(f'Device: {DEVICE}  Workers: {WORKERS}  Encoder: {_H264_ENCODER}')
    if not todo:
        print('Nothing to do.')
        return

    tasks = [(hf, OUT_DIR, DEVICE) for hf in todo]
    t0 = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(_worker, t): t for t in tasks}
        for fut in as_completed(futures):
            completed += 1
            elapsed = time.time() - t0
            vph = completed / elapsed * 3600 if elapsed > 0 else 0
            eta = (len(todo) - completed) / vph if vph > 0 else 0
            if completed % 20 == 0 or completed == len(todo):
                print(f'  [{completed}/{len(todo)}] {vph:.0f} vid/hr  ETA {eta:.1f}h',
                      flush=True)

    elapsed = time.time() - t0
    print(f'\nDone: {len(todo)} videos in {elapsed/60:.1f} min '
          f'({len(todo)/elapsed*3600:.0f} vid/hr)')


if __name__ == '__main__':
    main()
