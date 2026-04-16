import os
import requests
from tqdm import tqdm

def download_file_with_resume(url, dest):
    """Downloads a file with resume support via ranges."""
    # Check if we have a partial file
    headers = {}
    if os.path.exists(dest):
        downloaded = os.path.getsize(dest)
        headers['Range'] = f'bytes={downloaded}-'
    else:
        downloaded = 0

    # Test the connection to see if it works and what the size is
    try:
        response = requests.get(url, headers=headers, stream=True)
        # 416 means Requested Range Not Satisfiable (i.e. we fully downloaded it)
        if response.status_code == 416:
            print(f"{dest} already fully downloaded.")
            return
        elif response.status_code not in (200, 206):
            print(f"Failed to download {url}: Status {response.status_code}")
            return
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return

    # If new download it's 200, if partial resume it's 206
    total_size = int(response.headers.get('content-length', 0)) + downloaded
    mode = 'ab' if downloaded > 0 else 'wb'

    print(f"Downloading to {dest}: {total_size / (1024*1024*1024):.2f} GB")

    with open(dest, mode) as f, tqdm(
        desc=os.path.basename(dest),
        initial=downloaded,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=1024*1024*10): # 10MB chunks
            if chunk:
                size = f.write(chunk)
                pbar.update(size)

def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'bdd100k')
    os.makedirs(out_dir, exist_ok=True)
    
    files = [
        ("http://128.32.162.150/bdd100k/video_parts/bdd100k_videos_train_00.zip", os.path.join(out_dir, "bdd100k_videos_train_00.zip")),
        ("http://128.32.162.150/bdd100k/video_parts/bdd100k_videos_train_01.zip", os.path.join(out_dir, "bdd100k_videos_train_01.zip")),
        ("http://128.32.162.150/bdd100k/bdd100k_det_20_labels.zip", os.path.join(out_dir, "bdd100k_det_20_labels.zip"))
    ]
    
    for url, dest in files:
        print(f"Processing {url}")
        download_file_with_resume(url, dest)

if __name__ == '__main__':
    main()
