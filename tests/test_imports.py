"""Smoke test: every public module imports cleanly with declared dependencies only.

Guards against the two structural issues this test suite was added to catch
regressions on: (1) `sheetsage.representations.jukebox` importing the real
`jukebox_infer` package (previously a vendored fork with a missing
`jukebox_modules.data` submodule that made `use_jukebox=True` crash at
import time -- see CHANGELOG), and (2) `madmom` resolving to *something*
importable in this environment (it is git-installed, not the broken PyPI
0.16.1 default -- see pyproject.toml's install-honesty comments).
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


def test_import_madmom():
    """madmom must resolve to the git-installed fix, not PyPI's broken 0.16.1."""
    import madmom

    assert madmom.__version__ != "0.16.1", (
        "madmom resolved to the broken PyPI 0.16.1 release -- expected the git "
        "HEAD version pinned in pyproject.toml's [tool.uv.sources]"
    )
