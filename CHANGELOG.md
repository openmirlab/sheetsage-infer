# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed

- **`use_jukebox=True` was broken.** The vendored `sheetsage/representations/
  jukebox_modules/` fork was missing its `data/` submodule (`labels.py`,
  `text_processor.py`, `artist_genre_processor.py`), so any call that built
  Jukebox's prior stack (i.e. every real use of the Jukebox feature path)
  crashed with `ModuleNotFoundError: No module named 'jukebox_modules.data'`.
  This was hit unconditionally, not on an edge case.
- Along the way, uncovered and worked around a real bug in the
  `jukebox_infer` package that the fix above now depends on:
  `RangeEmbedding.__init__` stores `n_time` without casting it to `int`;
  arithmetic upstream in `make_vqvae` taints it as `numpy.float64`, and
  `RangeEmbedding.forward`'s `t.arange(...).view(1, n_time)` then raises
  `TypeError` because `torch.Tensor.view` rejects `numpy.float64`. Worked
  around locally in `sheetsage/representations/jukebox.py` via a small,
  documented monkeypatch (the vendored fork had independently discovered
  and fixed the same bug by casting to `int`); the proper fix belongs
  upstream in `jukebox_infer`.

### Changed

- **Replaced the vendored Jukebox fork with the real `jukebox_infer`
  package.** `sheetsage/representations/jukebox_modules/` (~5.4k LOC,
  loaded via a `sys.path.insert` hack) is gone; `sheetsage/representations/
  jukebox.py` now imports `jukebox_infer` directly. `jukebox-infer` was
  already declared as a pyproject dependency but nothing imported it before
  this change -- it was dead weight next to the vendored duplicate.
  Verified: the CPU (handcrafted-features) lead-sheet output is
  byte-identical before/after (untouched code path); the GPU (Jukebox)
  path could not be compared byte-for-byte because it did not previously
  produce output at all (see Fixed above) -- post-swap it now runs to
  completion and produces finite, plausibly-ranged fp16 activations.
- Diffed every vendored file against `jukebox_infer` 0.1.1 file-by-file
  before swapping: all core inference-path files (`hparams.py`,
  `utils/*.py`, `prior/*.py`, `transformer/*.py`, `vqvae/*.py`) were
  cosmetic-only divergence (black reformatting, quote style, dict-literal
  vs `dict()`), *except* the two real divergences noted above (missing
  `data/` package; the `n_time` int-cast fix) and one dead-code divergence
  in `api.py`/`cli.py` (a generation-oriented API sheetsage never imports).
  `make_models.py` also carried an additive checkpoint-corruption-retry
  feature not present in `jukebox_infer`; deliberately not ported back in
  this pass (exceptional-path robustness, doesn't affect representation
  correctness) -- flagged for a future upstream contribution instead.

### Documentation

- README's Installation section now documents the real install path: a
  bare `pip install openmirlab-sheetsage-infer` resolves `madmom` to the
  broken PyPI 0.16.1 release, so an explicit `pip install git+https://
  github.com/CPJKU/madmom.git` step is required for anyone not using `uv`
  (which already picks up the git pin via `[tool.uv.sources]`).
  `pyproject.toml` now carries inline comments explaining this, plus a
  `TODO` marking the dependency swap to `madmom-infer` (the org's
  maintained replacement) as blocked until it is published to PyPI
  (Phase 1+2 are done locally, not yet released).
- Added `NOTICE` documenting the MIT (code) vs CC BY-NC-SA (downloaded
  weights/HookTheory-derived data, plus madmom's bundled DBN models) split
  that `LICENSE` already asserted but nothing else in the repo explained.
- Added nav-style headers (what it is, what it reads/is read by) to the
  larger load-bearing modules: `infer.py`, `representations/jukebox.py`,
  `utils.py`, `beat_track.py`, `assets.py`.

### Added

- `tests/` did not exist despite README instructing `pytest tests/`. Added:
  import smoke tests (including one asserting the Jukebox path resolves to
  the real `jukebox_infer` package, not a re-appearing vendor fork); a
  bit-exact CPU regression test for `Handcrafted` features against a
  fixture recorded in this environment (skips, not fails, on a
  torch/librosa/numpy/python mismatch -- confirmed necessary: the
  project's own *original* `TEST_MP3_OAFMELSPEC_REF` fixture no longer
  matches under librosa>=0.11, off by ~12.8 max-abs-diff); a discrete
  (LilyPond text) end-to-end lead-sheet regression test on the same
  env-guard convention; and an opt-in GPU smoke test for the Jukebox path
  (`SHEETSAGE_RUN_JUKEBOX_TESTS=1`, skipped by default -- loads a ~5GB
  checkpoint).
- `.github/workflows/publish.yml` now runs `pytest tests/` before building
  the package.
- Removed `sheetsage/representations/__init__bak.py`, an unreferenced
  backup file.
- Removed the stale, unreferenced `requirements.txt` (duplicated
  `pyproject.toml`'s dependency list and had drifted out of sync -- it
  listed neither `madmom` nor `jukebox-infer`, so `pip install -r
  requirements.txt` would have silently produced a broken install).
  `pyproject.toml` is the single source of truth for dependencies.

## [0.1.1] - see prior git history for changes up to this release.
