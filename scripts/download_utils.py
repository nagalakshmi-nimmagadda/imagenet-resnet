import sys
import academictorrents as at
from tqdm import tqdm

class ProgressCallback:
    def __init__(self, desc):
        self.pbar = None
        self.desc = desc

    def __call__(self, bytes_downloaded, total_bytes):
        if self.pbar is None and total_bytes:
            self.pbar = tqdm(
                total=total_bytes,
                desc=self.desc,
                unit='B',
                unit_scale=True,
                unit_divisor=1024
            )
        if self.pbar:
            self.pbar.update(bytes_downloaded - self.pbar.n)

def download_with_progress(torrent_hash, datastore, desc):
    try:
        callback = ProgressCallback(desc)
        path = at.get(
            torrent_hash,
            datastore=datastore,
            progress_callback=callback
        )
        return path
    except Exception as e:
        print(f"Error downloading {desc}: {str(e)}", file=sys.stderr)
        raise 