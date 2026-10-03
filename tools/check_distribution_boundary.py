"""Reject dataset catalogs/helpers/data in a built wheel or source archive.

Reads: explicitly supplied .whl/.tar.gz files; checks required inference assets.
No archives are extracted, installed, or modified.
"""

import argparse
import json
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
    baseline = Path(__file__).resolve().parents[1] / "tests/fixtures/delivery_baseline.json"
    if baseline.is_file():
        expected = set(json.loads(baseline.read_text())["runtime_sha256"])
        expected.add("sheetsage/__about__.py")
        actual = {"sheetsage/" + name.split("sheetsage/", 1)[1]
                  for name in names if "sheetsage/" in name and not name.endswith("/")}
        # tar may list directories without a trailing slash; only members representing files count.
        if path.suffix != ".whl":
            with tarfile.open(path) as archive:
                actual = {"sheetsage/" + member.name.split("sheetsage/", 1)[1]
                          for member in archive.getmembers()
                          if member.isfile() and "sheetsage/" in member.name}
            for required in ("tests/fixtures/cpu_boundary_baseline/arrays.npz",
                             "tests/fixtures/cpu_boundary_baseline/metadata.json",
                             "tools/verify_installed.py", "tests/fixtures/delivery_baseline.json"):
                assert any(name.endswith(required) for name in names), (path, required)
        assert actual == expected, (path, sorted(actual ^ expected))
    print(f"{path.name}: inference assets present; dataset surface absent")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    for archive_path in parser.parse_args().archives:
        check(archive_path)
