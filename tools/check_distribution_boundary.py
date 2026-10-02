"""Reject dataset catalogs/helpers/data in a built wheel or source archive.

Reads: explicitly supplied .whl/.tar.gz files; checks required inference assets.
No archives are extracted, installed, or modified.
"""

import argparse
import tarfile
import zipfile
from pathlib import Path


def check(path):
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    else:
        with tarfile.open(path) as archive:
            names = archive.getnames()
    forbidden = ("hooktheory.json", "rwc.json", "dataset_assets.py", "dataset_catalogs/",
                 "hooktheory_data/", "hooktheory_transcription_results/")
    assert not [name for name in names if any(value in name for value in forbidden)], path
    for required in ("sheetsage/assets/sheetsage.json", "sheetsage/assets/jukebox.json",
                     "sheetsage/assets/test.json", "sheetsage/config/checkpoints.toml"):
        assert any(name.endswith(required) for name in names), (path, required)
    print(f"{path.name}: inference assets present; dataset surface absent")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    for archive_path in parser.parse_args().archives:
        check(archive_path)
