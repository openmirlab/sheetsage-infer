"""Offline integrity and resolver contracts for SheetSage2 checkpoint bundles."""

import hashlib
import json
from pathlib import Path

import pytest

from sheetsage.sheetsage2_checkpoints import CheckpointStore


REVISION = "0123456789abcdef0123456789abcdef01234567"
BASE_REVISION = "abcdef0123456789abcdef0123456789abcdef01"


def _artifact(path, url=None):
    data = path.read_bytes()
    return {
        "url": url or path.as_uri(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
    }


def _manifest(tmp_path, *, custom_code=False):
    source = tmp_path / "source"
    source.mkdir()
    config = source / "config.json"
    weights = source / "model.safetensors"
    config.write_bytes(b'{"model_type":"tiny"}')
    weights.write_bytes(b"fake weights, intentionally tiny")
    components = {}
    for key, revision in (("sheetsage2", REVISION), ("mert2", BASE_REVISION)):
        artifacts = {
            "config.json": _artifact(config),
            "model.safetensors": _artifact(weights),
        }
        if custom_code:
            code = source / "configuration.py"
            code.write_bytes(b"class Config: pass\n")
            artifacts["configuration.py"] = _artifact(code)
        components[key] = {
            "repository": f"test/{key}", "revision": revision,
            "license": "test", "provenance": "test fixture", "artifacts": artifacts,
        }
    manifest_path = tmp_path / "manifest.json"
    document = {"schema_version": 1, "components": components}
    manifest_path.write_text(json.dumps(document))
    return manifest_path, document


def _store(tmp_path, **kwargs):
    manifest, _ = _manifest(tmp_path)
    return CheckpointStore(checkpoint_config=manifest, cache_dir=tmp_path / "cache", **kwargs)


def test_manifest_is_fully_validated_and_accepts_optional_code(tmp_path):
    manifest, _ = _manifest(tmp_path, custom_code=True)
    store = CheckpointStore(checkpoint_config=manifest, cache_dir=tmp_path / "cache")
    assert "configuration.py" in store.manifest["components"]["sheetsage2"]["artifacts"]


@pytest.mark.parametrize("source_change", ["modify", "delete"])
def test_provenance_snapshots_validated_bytes_once_and_survives_source_changes(
    tmp_path, monkeypatch, source_change,
):
    manifest, document = _manifest(tmp_path)
    # Preserve exact bytes, including whitespace, rather than reserializing JSON.
    original = b"\n  " + manifest.read_bytes() + b"\n"
    manifest.write_bytes(original)
    expected = {
        "manifest_sha256": hashlib.sha256(original).hexdigest(),
        "components": {
            key: {"repository": value["repository"], "revision": value["revision"]}
            for key, value in document["components"].items()
        },
    }
    reads = []
    read_bytes = Path.read_bytes

    def counted_read(path):
        reads.append(path.resolve())
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read)
    monkeypatch.chdir(tmp_path)
    store = CheckpointStore(checkpoint_config="manifest.json", cache_dir="cache")
    assert reads == [manifest]
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    if source_change == "modify":
        document["components"]["sheetsage2"]["revision"] = "0" * 40
        manifest.write_text(json.dumps(document))
    else:
        manifest.unlink()

    def no_network(*args, **kwargs):
        pytest.fail("provenance attempted a download")

    monkeypatch.setattr("sheetsage.sheetsage2_checkpoints.urlopen", no_network)
    snapshot = store.provenance()
    assert snapshot == expected
    snapshot["manifest_sha256"] = "changed"
    snapshot["components"]["sheetsage2"]["revision"] = "changed"
    snapshot["components"].pop("mert2")
    assert store.provenance() == expected
    assert reads == [manifest]
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("damage", ["malformed", "version", "revision", "hash", "unsafe"])
def test_invalid_manifest_fails_at_store_construction(tmp_path, damage):
    manifest_path, document = _manifest(tmp_path)
    if damage == "malformed":
        manifest_path.write_text("{")
    elif damage == "version":
        document["schema_version"] = 2
    elif damage == "revision":
        document["components"]["mert2"]["revision"] = "main"
    elif damage == "hash":
        document["components"]["sheetsage2"]["artifacts"]["model.safetensors"]["sha256"] = "bad"
    else:
        artifacts = document["components"]["sheetsage2"]["artifacts"]
        artifacts["../escape.py"] = artifacts["config.json"]
    if damage != "malformed":
        manifest_path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Invalid SheetSage2 checkpoint manifest"):
        CheckpointStore(checkpoint_config=manifest_path)


def test_cache_info_and_prepare_share_api_env_and_explicit_path_resolver(tmp_path, monkeypatch):
    manifest, _ = _manifest(tmp_path)
    api_cache = tmp_path / "api-cache"
    monkeypatch.setenv("SHEETSAGE2_CACHE_DIR", str(api_cache))
    api_store = CheckpointStore(checkpoint_config=manifest)
    assert Path(api_store.cache_info()["cache_dir"]) == api_cache.resolve()
    assert api_store.directory("sheetsage2").name == f"sheetsage2_{REVISION}"
    override = tmp_path / "chosen-adapter"
    override.mkdir()
    direct_store = CheckpointStore(checkpoint_config=manifest, model_path=override)
    assert direct_store.cache_info()["components"]["sheetsage2"]["path"] == str(override.resolve())
    assert direct_store.directory("sheetsage2") == override.resolve()
    assert direct_store.directory("mert2").name == f"mert2_{BASE_REVISION}"

    api_override = tmp_path / "api-override"
    api_store = CheckpointStore(checkpoint_config=manifest, cache_dir=api_override)
    assert Path(api_store.cache_info()["cache_dir"]) == api_override.resolve()
    assert api_store.directory("sheetsage2").parent == api_override.resolve()


def test_explicit_base_directory_uses_same_prepare_and_inspection_resolution(tmp_path):
    manifest, _ = _manifest(tmp_path)
    source = tmp_path / "source"
    base_directory = tmp_path / "provided-base"
    base_directory.mkdir()
    for filename in ("config.json", "model.safetensors"):
        (base_directory / filename).write_bytes((source / filename).read_bytes())
    store = CheckpointStore(
        checkpoint_config=manifest, cache_dir=tmp_path / "unused-cache",
        base_model_path=base_directory,
    )
    inspected = store.cache_info()["components"]["mert2"]
    assert inspected["path"] == str(base_directory.resolve())
    assert inspected["complete"]
    prepared = store.prepare()
    assert prepared["mert2"] == base_directory.resolve()
    assert store.cache_info()["components"]["mert2"]["complete"]
    assert not (tmp_path / "unused-cache" / f"mert2_{BASE_REVISION}").exists()


def test_inspection_is_non_downloading_and_does_not_import_backend(tmp_path, monkeypatch):
    store = _store(tmp_path)
    import sheetsage.sheetsage2_checkpoints as checkpoints

    def no_network(*args, **kwargs):
        pytest.fail("inspection attempted a download")

    monkeypatch.setattr(checkpoints, "urlopen", no_network)
    info = store.cache_info()
    assert not any(component["complete"] for component in info["components"].values())
    assert all(not component["integrity_verified"] for component in info["components"].values())
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("mode", ["offline", "explicit"])
def test_missing_local_checkpoint_fails_before_network_or_optional_imports(tmp_path, monkeypatch, mode):
    store = _store(tmp_path, **({"model_path": tmp_path / "missing"} if mode == "explicit" else {}))
    import sheetsage.sheetsage2_checkpoints as checkpoints

    monkeypatch.setattr(checkpoints, "urlopen", lambda *a, **k: pytest.fail("network used"))
    with pytest.raises(FileNotFoundError, match="Missing SheetSage2 artifact"):
        store.prepare(local_files_only=(mode == "offline"))
    assert not (tmp_path / "cache").exists()


def test_prepare_downloads_file_urls_and_offline_reload_reuses_verified_cache(tmp_path):
    store = _store(tmp_path)
    resolved = store.prepare()
    assert set(resolved) == {"sheetsage2", "mert2"}
    info = store.cache_info()
    assert all(component["complete"] for component in info["components"].values())
    assert store.prepare(local_files_only=True) == resolved


@pytest.mark.parametrize("corruption", ["size", "same_size_hash"])
def test_prepare_rejects_corrupt_cached_file(tmp_path, corruption):
    store = _store(tmp_path)
    resolved = store.prepare()
    target = resolved["sheetsage2"] / "config.json"
    original = target.read_bytes()
    target.write_bytes(original + b"x" if corruption == "size" else b"x" * len(original))
    with pytest.raises(ValueError, match="size mismatch" if corruption == "size" else "SHA-256 mismatch"):
        store.prepare(local_files_only=True)


def test_bad_download_checksum_leaves_no_partial_file(tmp_path, monkeypatch):
    manifest, document = _manifest(tmp_path)
    component = document["components"]["sheetsage2"]
    artifact = component["artifacts"]["config.json"]
    artifact["url"] = (tmp_path / "source" / "model.safetensors").as_uri()
    artifact["size"] = 1
    manifest.write_text(json.dumps(document))
    store = CheckpointStore(checkpoint_config=manifest, cache_dir=tmp_path / "cache")
    with pytest.raises(ValueError, match="size mismatch"):
        store.prepare()
    component_dir = store.directory("sheetsage2")
    assert not (component_dir / "config.json").exists()
    assert list(component_dir.glob(".download-*")) == []
