"""Model-weight asset registry: resolves a named asset to a checksum-verified
local file, downloading it into `CACHE_DIR` on first use.

Each `sheetsage/assets/*.json` (hooktheory, jukebox, rwc, sheetsage, test)
is a manifest of {name: {url, checksum, ...}} entries; `retrieve_asset(name)`
looks a name up across all manifests and returns its local path. These
downloaded weights are CC BY-NC-SA (see LICENSE/NOTICE), unlike this
package's MIT-licensed code.

Reads: sheetsage/assets/*.json, .utils (compute_checksum); read by:
sheetsage.representations.handcrafted, sheetsage.infer (_init_model)
"""

import json
import logging
import pathlib
import shutil
import urllib.request

from . import CACHE_DIR, LIB_DIR
from .utils import compute_checksum

_DEFAULT_CHUNK_SIZE = 4096
_ASSETS = None


def _init_assets():
    global _ASSETS
    if _ASSETS is not None:
        raise Exception("Should only run this once")

    _ASSETS = {}
    asset_paths = set()
    for json_path in sorted(pathlib.Path(LIB_DIR, "assets").rglob("*.json")):
        with open(json_path) as f:
            d = json.load(f)
        for _tag, asset in d.items():
            if "checksum" not in asset:
                raise AssertionError("Missing checksum")
            try:
                asset["path"] = pathlib.PurePosixPath(asset["path"].strip())
            except Exception:
                raise AssertionError("Invalid path") from None
            if asset["path"] in asset_paths:
                raise AssertionError("Duplicate path")
            asset_paths.add(asset["path"])
            asset["path_abs"] = pathlib.Path(CACHE_DIR, asset["path"])
        _ASSETS.update(d)


_init_assets()


def get_asset_tags():
    return set(_ASSETS.keys())


def get_asset_checksum(tag):
    """Return a tagged asset's checksum without downloading or verifying the file itself.

    For callers (e.g. Phonon's provider wrapper) that need to publish a checkpoint's
    identity for observability purposes but never touch the file directly -- `get_asset_tags`
    only exposes the set of tag names, not the checksum each one resolves to.

    Raises
    ------
    :class:`ValueError`
       Unknown asset tag.
    """
    if tag not in _ASSETS:
        raise ValueError(f"Unknown asset tag: {tag}")
    return _ASSETS[tag]["checksum"]


def _download_from_url(url, dest_path, chunk_size=_DEFAULT_CHUNK_SIZE, log=True):
    if log:
        logging.info(f"Downloading from: {url}")
    if "mega.nz" in url.lower():
        import subprocess

        if shutil.which("megadl"):
            result = subprocess.run(
                ["megadl", url, "--path", str(dest_path)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise Exception(f"Mega.nz download failed: {result.stderr}")
            return
        raise Exception("Mega.nz links require megatools. Install: sudo apt-get install megatools")

    with open(dest_path, "wb") as f:
        r = urllib.request.urlopen(url)
        while True:
            chunk = r.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)


def _download_from_huggingface(asset, dest_path, log=True):
    repo_id = asset.get("huggingface_repo")
    filename = asset.get("huggingface_filename")
    if not repo_id or not filename:
        raise Exception("Missing Hugging Face repository information for asset")
    repo_type = asset.get("huggingface_repo_type", "model")
    revision = asset.get("huggingface_revision")
    if log:
        logging.info("Downloading from Hugging Face: repo=%s file=%s", repo_id, filename)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise Exception(
            "huggingface-hub is required for Hugging Face downloads. Install with `pip install huggingface-hub`."
        ) from exc

    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=revision,
        )
    except Exception as exc:
        raise Exception(
            f"Failed to download {filename} from Hugging Face repo {repo_id}: {exc}"
        ) from exc

    shutil.copyfile(downloaded_path, dest_path)


def _download(asset, dest_path, chunk_size=_DEFAULT_CHUNK_SIZE, log=True):
    errors = []

    url = asset.get("url")
    if url:
        try:
            _download_from_url(url, dest_path, chunk_size=chunk_size, log=log)
            return
        except Exception as exc:
            errors.append(f"URL download failed ({exc})")
            if dest_path.exists():
                dest_path.unlink()

    if asset.get("huggingface_repo") and asset.get("huggingface_filename"):
        try:
            _download_from_huggingface(asset, dest_path, log=log)
            return
        except Exception as exc:
            errors.append(f"Hugging Face download failed ({exc})")
            if dest_path.exists():
                dest_path.unlink()

    if errors:
        raise Exception("; ".join(errors))

    raise Exception("File is missing and cannot be downloaded")


def retrieve_asset(tag, delete_wrong=False, chunk_size=_DEFAULT_CHUNK_SIZE, log=True):
    """Attempts to acquire and/or verify existence of a tagged asset in the cache.

    Returns
    -------
    str
       Absolute file path for asset, if verified.

    Raises
    ------
    :class:`ValueError`
       Invalid asset tag.
    :class:`Exception`
       Asset could not be verified.
    """
    # Retrieve asset
    if tag not in _ASSETS:
        raise ValueError()
    asset = _ASSETS[tag]
    path = asset["path_abs"]
    checksum = asset["checksum"]
    if log:
        logging.info(f"Verifying asset: {tag}")
        logging.info(f"Asset location: {path}")

    # Create parent directory
    if not path.parent.is_dir():
        if log:
            logging.info(f"Creating parent: {path.parent}")
        path.parent.mkdir(parents=True)

    def verify():
        assert path.is_file()
        if checksum is not None:
            if len(checksum) == 32:
                algorithm = "md5"
            elif len(checksum) == 40:
                algorithm = "sha1"
            elif len(checksum) == 64:
                algorithm = "sha256"
            else:
                raise AssertionError("Unknown checksum algorithm")
            computed = compute_checksum(path, algorithm=algorithm, chunk_size=chunk_size)
            if computed != checksum:
                raise Exception(f"File {path} has wrong checksum.")

    # Delete incorrect files
    already_verified = False
    if delete_wrong and path.is_file():
        try:
            verify()
            already_verified = True
        except Exception:
            logging.warning(f"Deleting file with bad checksum: {path}")
            path.unlink()

    # Attempt to download
    if not path.is_file():
        try:
            _download(asset, path, chunk_size=chunk_size, log=log)
        except Exception as e:
            raise Exception(f"Download failed: {e}") from e
    assert path.is_file()

    # Ensure file integrity
    if not already_verified:
        verify()
    if log:
        logging.info("Verified!")

    return path


if __name__ == "__main__":
    import multiprocessing
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("startswith", nargs="?")
    parser.add_argument("--delete_wrong", action="store_true", dest="delete_wrong")
    parser.add_argument("--num_parallel", "-n", type=int)

    parser.set_defaults(startswith=None, num_parallel=1, delete_wrong=False)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    tags = sorted(get_asset_tags())
    if args.startswith is not None:
        tags = [t for t in tags if t.startswith(args.startswith.strip().upper())]

    def task(t):
        logging.info("-" * 80)
        try:
            retrieve_asset(t, delete_wrong=args.delete_wrong)
        except Exception as e:
            logging.error(e)
            raise e

    with multiprocessing.Pool(args.num_parallel) as p:
        p.map(task, tags)
