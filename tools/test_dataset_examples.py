"""Repository-only checks for examples deliberately omitted from distributions.

Reads: example helper/catalogs, mocked download bytes, immutable source hashes.
Run explicitly with pytest tools/test_dataset_examples.py; no network is used.
"""

import hashlib
import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import dataset_assets


def test_preserved_catalog_bytes():
    root = Path(__file__).resolve().parents[1]
    baseline = json.loads((root / "tests/fixtures/cpu_boundary_baseline/metadata.json").read_text())
    for name in ("hooktheory.json", "rwc.json"):
        actual = (root / "examples/dataset_catalogs" / name).read_bytes()
        assert hashlib.sha256(actual).hexdigest() == baseline["source_sha256"][f"sheetsage/assets/{name}"]


def test_download_verifies_before_accepting_and_reuses_cache(monkeypatch, tmp_path):
    content = b"independent test dataset"
    calls = []
    monkeypatch.setattr(dataset_assets, "get_dataset_asset", lambda tag: {
        "path": "example/data.json", "checksum": hashlib.sha256(content).hexdigest(),
        "url": "https://example.invalid/data",
    })
    monkeypatch.setattr(dataset_assets.urllib.request, "urlopen",
                        lambda url: calls.append(url) or io.BytesIO(content))
    first = dataset_assets.retrieve_dataset_asset("TEST", cache_dir=tmp_path)
    second = dataset_assets.retrieve_dataset_asset("TEST", cache_dir=tmp_path)
    assert first == second == tmp_path / "example/data.json"
    assert first.read_bytes() == content
    assert len(calls) == 1
    first.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="wrong SHA-256"):
        dataset_assets.retrieve_dataset_asset("TEST", cache_dir=tmp_path)
    assert first.read_bytes() == b"corrupt"
    assert len(calls) == 1


def test_corrupt_download_is_never_installed(monkeypatch, tmp_path):
    monkeypatch.setattr(dataset_assets, "get_dataset_asset", lambda tag: {
        "path": "data", "checksum": "0" * 64, "url": "https://example.invalid/data",
    })
    monkeypatch.setattr(dataset_assets.urllib.request, "urlopen", lambda url: io.BytesIO(b"bad"))
    with pytest.raises(ValueError, match="wrong SHA-256"):
        dataset_assets.retrieve_dataset_asset("TEST", cache_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_manual_only_asset_and_unknown_tag_fail_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="manually supplied"):
        dataset_assets.retrieve_dataset_asset("RWC_AUDIO_P012", cache_dir=tmp_path)
    with pytest.raises(ValueError, match="Unknown repository dataset tag"):
        dataset_assets.retrieve_dataset_asset("UNKNOWN", cache_dir=tmp_path)


def test_both_example_consumers_use_repository_helper(monkeypatch, tmp_path):
    import hooktheory_example
    import hooktheory_simple

    calls = []
    monkeypatch.setattr(hooktheory_example, "retrieve_asset",
                        lambda tag, **kwargs: calls.append(tag) or tmp_path / tag)
    monkeypatch.setattr(hooktheory_simple, "retrieve_dataset_asset",
                        lambda tag, **kwargs: calls.append(tag) or tmp_path / tag)
    hooktheory_example.download_hooktheory_segments("TRAIN")
    hooktheory_example.download_hooktheory_midi("VALID")
    hooktheory_simple.download_file(hooktheory_simple.HOOKTHEORY_URLS["TEST_SEGMENTS"], tmp_path / "x")
    assert calls == ["HOOKTHEORY_TRAIN_SEGMENTS", "HOOKTHEORY_VALID_MIDI", "HOOKTHEORY_TEST_SEGMENTS"]
