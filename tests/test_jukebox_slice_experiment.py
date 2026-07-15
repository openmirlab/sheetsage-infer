"""Parity and wiring checks for the reversible Jukebox slice experiment."""

import importlib


def test_backend_modes_expose_the_same_required_symbols(monkeypatch):
    monkeypatch.setenv("SHEETSAGE_JUKEBOX_BACKEND", "slice")
    seam = importlib.reload(importlib.import_module("sheetsage._jukebox.slice"))
    sliced = seam.get_backend()
    external = seam.get_backend("external")
    assert sliced.mode == "slice"
    assert external.mode == "external"
    assert sliced.Hyperparams is not external.Hyperparams
    params = sliced.Hyperparams(sample_length_in_seconds=24.0)
    assert params.sample_length_in_seconds == 24.0
    assert sliced.make_model is external.make_model
    assert sliced.conditioners is external.conditioners
    assert sliced.empty_cache is external.empty_cache


def test_public_representation_keeps_legacy_symbol_identity(monkeypatch):
    monkeypatch.setenv("SHEETSAGE_JUKEBOX_BACKEND", "external")
    jukebox = importlib.reload(importlib.import_module("sheetsage.representations.jukebox"))
    models = importlib.import_module("jukebox_infer.make_models")
    assert jukebox.make_model is models.make_model
