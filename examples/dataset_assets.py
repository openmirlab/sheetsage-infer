"""Repository-only dataset acquisition for the HookTheory examples.

Reads: dataset_catalogs/*.json and an explicit output path or legacy dataset cache.
This helper and its catalogs are excluded from source and wheel distributions;
the inference package never imports them. Downloads occur only on explicit calls.
"""

import hashlib
import json
import logging
import os
import tempfile
import urllib.request
from pathlib import Path


def get_dataset_asset(tag):
    """Read one preserved dataset entry without importing the inference runtime."""
    for path in sorted(Path(__file__).with_name("dataset_catalogs").glob("*.json")):
        entries = json.loads(path.read_text())
        if tag in entries:
            return entries[tag]
    raise ValueError(f"Unknown repository dataset tag: {tag}")


def retrieve_dataset_asset(tag, *, cache_dir=None, output_path=None, log=True):
    """Acquire a dataset explicitly, verifying SHA-256 before accepting its bytes.

    Existing incorrect files fail without modification. A missing manual-only
    asset fails with its required location. The default cache preserves the old
    examples' SHEETSAGE_CACHE_DIR / ~/.sheetsage location.
    """
    if cache_dir is not None and output_path is not None:
        raise ValueError("Choose cache_dir or output_path, not both")
    asset = get_dataset_asset(tag)
    cache = Path(cache_dir or os.environ.get("SHEETSAGE_CACHE_DIR", Path.home() / ".sheetsage"))
    destination = Path(output_path) if output_path is not None else cache / asset["path"]
    expected = asset["checksum"]

    def verify(path):
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Dataset {tag} has wrong SHA-256: {actual}; expected {expected}")

    if destination.is_file():
        verify(destination)
        return destination
    if "url" not in asset:
        raise FileNotFoundError(f"Dataset {tag} requires a manually supplied file at {destination}")
    if log:
        logging.info("Downloading repository dataset %s to %s", tag, destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as target:
            temporary = Path(target.name)
            with urllib.request.urlopen(asset["url"]) as response:
                while chunk := response.read(1 << 20):
                    target.write(chunk)
        verify(temporary)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination
