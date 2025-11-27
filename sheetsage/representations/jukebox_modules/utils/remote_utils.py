import os
import subprocess
import sys
from urllib.request import urlretrieve
from tqdm import tqdm


class DownloadProgressBar:
    """Progress bar for file downloads."""
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc="Downloading"
            )
        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(block_size)
        else:
            self.pbar.close()


def download(remote_path, local_path, async_download=False, show_progress=True):
    """
    Download a file from remote_path to local_path.
    
    Args:
        remote_path: URL to download from
        local_path: Local file path to save to
        async_download: If True, download in background (not recommended)
        show_progress: If True, show download progress bar
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # Check if file already exists
    if os.path.exists(local_path):
        file_size = os.path.getsize(local_path)
        if file_size > 0:
            print(f"✓ Checkpoint already exists: {local_path} ({file_size / (1024**2):.1f} MB)")
            return True
    
    print(f"Downloading {os.path.basename(local_path)}...")
    print(f"  From: {remote_path}")
    print(f"  To: {local_path}")
    
    try:
        if show_progress and not async_download:
            # Use urllib with progress bar
            progress = DownloadProgressBar()
            urlretrieve(remote_path, local_path, reporthook=progress)
            print(f"✓ Download complete: {local_path}")
        else:
            # Fallback to wget if urllib fails
            args = ['wget', '-c', '-O', local_path, remote_path]
            if not show_progress:
                args.append('-q')
            result = subprocess.call(args)
            if result != 0:
                raise RuntimeError(f"wget failed with exit code {result}")
            print(f"✓ Download complete: {local_path}")
        return True
    except Exception as e:
        print(f"✗ Download failed: {e}")
        # Clean up partial download
        if os.path.exists(local_path):
            try:
                os.remove(local_path)
            except:
                pass
        raise


def check_file_exists(local_path):
    """Check if a checkpoint file exists and has non-zero size."""
    return os.path.exists(local_path) and os.path.getsize(local_path) > 0


