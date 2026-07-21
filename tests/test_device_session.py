"""Focused device, session-cache, and checkpoint-resolution contract tests.

All model construction and assets are doubles; these tests never download weights.
"""

import pytest
import torch

from sheetsage.device import resolve_device
from sheetsage.session import SheetSageSession


def test_auto_device_resolves_to_cpu_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == torch.device("cpu")


def test_explicit_unavailable_or_invalid_cuda_raises(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="explicitly requested"):
        resolve_device("cuda")
    with pytest.raises(ValueError, match="Invalid device"):
        resolve_device("not-a-device")


@pytest.mark.parametrize(
    ("use_jukebox", "expected"),
    [(False, "cpu"), (True, "cuda")],
)
def test_no_device_one_shot_preserves_historical_feature_backend(monkeypatch, use_jukebox, expected):
    import sheetsage.infer as infer_module

    selected = []

    def capture(device):
        selected.append(device)
        raise RuntimeError("stop before model loading")

    monkeypatch.setattr("sheetsage.device.resolve_device", capture)
    with pytest.raises(RuntimeError, match="stop before model loading"):
        infer_module.sheetsage("unused.wav", use_jukebox=use_jukebox)
    assert selected == [expected]


def test_session_forwards_resolved_device_and_reuses_components(monkeypatch):
    import sheetsage.infer as infer_module
    import sheetsage.pipeline as pipeline

    calls = []
    components = object()
    monkeypatch.setattr(
        pipeline,
        "load_components",
        lambda input_feats, melody, harmony, device: calls.append((input_feats, melody, harmony, device))
        or components,
    )
    inferred = []
    monkeypatch.setattr(infer_module, "sheetsage", lambda audio, **kwargs: inferred.append((audio, kwargs)) or "ok")

    session = SheetSageSession(device="cpu").load()
    assert session.infer("first") == "ok"
    assert session.infer("second") == "ok"
    assert len(calls) == 1
    assert calls[0][-1] == torch.device("cpu")
    assert [kwargs["_components"] for _, kwargs in inferred] == [components, components]
    assert [kwargs["device"] for _, kwargs in inferred] == [torch.device("cpu"), torch.device("cpu")]


def test_release_discards_components_and_a_later_load_rebuilds(monkeypatch):
    import sheetsage.pipeline as pipeline

    builds = []
    monkeypatch.setattr(pipeline, "load_components", lambda *args: builds.append(object()) or builds[-1])
    session = SheetSageSession(device="cpu").load()
    first = session._components
    session.release()
    assert session.status == "new"
    assert session._components is None
    session.load()
    assert len(builds) == 2
    assert session._components is not first


def test_cache_info_and_loading_share_asset_path_resolver(monkeypatch, tmp_path):
    import sheetsage.assets as assets

    tag = "SHEETSAGE_V02_HANDCRAFTED_MELODY_CFG"
    resolved = tmp_path / "checkpoint.json"
    resolved.write_text("{}")
    calls = []
    monkeypatch.setattr(assets, "resolve_asset_path", lambda requested: calls.append(requested) or resolved)
    monkeypatch.setattr(assets, "compute_checksum", lambda *args, **kwargs: assets._ASSETS[tag]["checksum"])

    assert assets.retrieve_asset(tag, log=False) == resolved
    SheetSageSession(device="cpu").cache_info()
    assert calls.count(tag) == 2


def test_checkpoint_catalog_has_honest_integrity_metadata():
    from sheetsage import LIB_DIR
    from sheetsage.assets import tomllib

    with open(LIB_DIR / "config" / "checkpoints.toml", "rb") as f:
        catalog = tomllib.load(f)["artifacts"]
    for artifact in catalog.values():
        expected_length = {"md5": 32, "sha1": 40, "sha256": 64}[artifact["integrity_algorithm"]]
        assert len(artifact["integrity_digest"]) == expected_length
        assert all(char in "0123456789abcdef" for char in artifact["integrity_digest"])


def test_checkpoint_catalog_preserves_legacy_resolver_legs():
    import json
    from sheetsage import LIB_DIR
    from sheetsage.assets import _ASSETS, tomllib

    with open(LIB_DIR / "assets" / "sheetsage.json") as f:
        legacy = json.load(f)
    with open(LIB_DIR / "config" / "checkpoints.toml", "rb") as f:
        catalog = tomllib.load(f)["artifacts"]
    assert {artifact["tag"] for artifact in catalog.values()} == set(legacy)
    for artifact in catalog.values():
        tag = artifact["tag"]
        for field in ("path", "url", "huggingface_repo", "huggingface_filename"):
            assert str(_ASSETS[tag].get(field)) == str(legacy[tag].get(field))
        assert _ASSETS[tag]["checksum"] == legacy[tag]["checksum"] == artifact["integrity_digest"]
        assert len(legacy[tag]["checksum"]) == {
            "md5": 32,
            "sha1": 40,
            "sha256": 64,
        }[artifact["integrity_algorithm"]]


def test_jukebox_oom_is_not_retried_on_cpu(monkeypatch):
    import sheetsage.representations.jukebox as jukebox

    calls = []
    monkeypatch.setattr(jukebox, "resolve_device", lambda device: torch.device("cuda:0"))
    monkeypatch.setattr(
        jukebox,
        "make_model",
        lambda *args, **kwargs: calls.append(args[1]) or (_ for _ in ()).throw(RuntimeError("CUDA out of memory")),
    )
    with pytest.raises(RuntimeError, match="CUDA out of memory.*cuda:0"):
        jukebox.JukeboxEmbeddings(auto_download=False)
    assert calls == [torch.device("cuda:0")]
