"""Smoke test: every public module imports cleanly with declared dependencies only.

Guards against the two structural issues this test suite was added to catch
regressions on: (1) `sheetsage.representations.jukebox` importing the real
`jukebox_infer` package (previously a vendored fork with a missing
`jukebox_modules.data` submodule that made `use_jukebox=True` crash at
import time -- see CHANGELOG), and (2) `madmom_infer.features.downbeats`
exposing the two classes `sheetsage.beat_track` needs (`RNNDownBeatProcessor`,
`DBNDownBeatTrackingProcessor`) -- as of 0.2.1 this is a plain PyPI dependency,
no git install or install-honesty workaround needed (see CHANGELOG).
"""

import importlib


def test_import_sheetsage():
    importlib.import_module("sheetsage")


def test_import_infer():
    importlib.import_module("sheetsage.infer")


def test_import_representations():
    mod = importlib.import_module("sheetsage.representations")
    assert hasattr(mod, "Handcrafted")
    assert hasattr(mod, "JukeboxEmbeddings")


def test_import_jukebox_representation_uses_real_package():
    """The vendored jukebox_modules/ fork is gone; this must come from jukebox_infer."""
    jukebox = importlib.import_module("sheetsage.representations.jukebox")
    import jukebox_infer

    assert jukebox.make_model is jukebox_infer.make_models.make_model


def test_import_theory():
    importlib.import_module("sheetsage.theory")


def test_import_beat_track():
    """sheetsage.beat_track must import cleanly (madmom-infer is a plain pip dependency now)."""
    importlib.import_module("sheetsage.beat_track")


def test_import_madmom_infer():
    """madmom-infer must expose the two classes beat_track.py's DBN path calls."""
    from madmom_infer.features.downbeats import (
        DBNDownBeatTrackingProcessor,
        RNNDownBeatProcessor,
    )

    assert DBNDownBeatTrackingProcessor is not None
    assert RNNDownBeatProcessor is not None
