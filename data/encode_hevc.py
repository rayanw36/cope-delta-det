import os
import shutil
import subprocess
import argparse
from pathlib import Path
from multiprocessing import Pool

def encode_video(args):
    input_path, output_path, qp, gop = args
    if os.path.exists(output_path):
        return
        
    ffmpeg_exe = shutil.which("ffmpeg")
    if not ffmpeg_exe:
        ffmpeg_exe = os.path.expanduser(r'~\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe')

    cmd = [
        ffmpeg_exe, 
        "-hwaccel", "cuda",
        "-i", input_path,
        "-c:v", "hevc_nvenc",
        "-preset", "p6",
        "-tune", "hq",
        "-rc", "constqp",
        "-qp", str(qp),
        "-g", str(gop),
        "-keyint_min", str(gop),
        "-sc_threshold", "0",
        "-fps_mode", "passthrough",
        "-y",
        output_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error encoding {input_path} with QP={qp}, GOP={gop}")
            print(f"  stderr: {result.stderr[:500]}")
    except Exception as e:
        print(f"Exception encoding {input_path} with QP={qp}, GOP={gop}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Encode BDD100k videos to HEVC at different QPs and GOPs")
    parser.add_argument("--input_dir", type=str, default="./data/bdd100k/bdd100k/videos/train")
    parser.add_argument("--output_base", type=str, default="./data/bdd100k/hevc")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel encoding workers")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Input directory {input_dir} does not exist.")
        return
        
    qps = [22, 27, 32, 37]
    gops = [8, 16, 32]
    
    tasks = []
    
    video_files = list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.mov"))
    print(f"Found {len(video_files)} videos in {input_dir}")
    
    for qp in qps:
        for gop in gops:
            out_dir = Path(args.output_base) / f"qp{qp}_gop{gop}"
            out_dir.mkdir(parents=True, exist_ok=True)
            
            for vf in video_files:
                out_path = out_dir / (vf.stem + ".hevc")
                tasks.append((str(vf), str(out_path), qp, gop))
                
    print(f"Total encoding tasks: {len(tasks)}")
    
    with Pool(args.workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(encode_video, tasks)):
            if (i+1) % 10 == 0:
                print(f"Progress: {i+1}/{len(tasks)}")
                
    print("Done encoding.")

if __name__ == "__main__":
    main()
