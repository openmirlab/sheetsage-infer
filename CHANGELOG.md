# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.2.1] - 2026-07-12

### Changed

- **Package renamed: `openmirlab-sheetsage-infer` -> `sheetsage-infer`.** The PyPI name
  `sheetsage-infer` was free and the org's trusted-publisher config was moved to it in the same
  change window (see `pyproject.toml`'s `[project.urls]` and the constitution's packaging
  article). `openmirlab-sheetsage-infer` is deprecated as of this release and will not receive
  further updates. The import name is unchanged (`import sheetsage`).
- **`madmom` replaced by [`madmom-infer`](https://pypi.org/project/madmom-infer/).** The bare
  `madmom` PyPI dependency (broken 0.16.1 release) plus its `[tool.uv.sources]` git-HEAD pin are
  gone; `sheetsage/beat_track.py` now depends on plain, `pip install`-able `madmom-infer>=0.1.0`.
  This also changes *how* it's called: the original `madmom` package installed a
  `DBNDownBeatTracker` console script that `beat_track.py` invoked via `subprocess`;
  `madmom-infer` is a pure-Python/numpy library with no console script, so `beat_track.py` now
  calls its Python API directly (`RNNDownBeatProcessor` + `DBNDownBeatTrackingProcessor`).
  Verified bit-identical against real upstream `madmom` (git HEAD) on the bundled `TEST_WAV`
  asset: same beat times, same first-downbeat index, same detected beats-per-bar. One gap
  bridged in the process: `madmom-infer` has no ffmpeg-backed resampler, so non-44.1kHz audio is
  now resampled with `librosa` before being handed to it (not bit-identical to `madmom`'s ffmpeg
  resample, but avoids a hard crash on non-44.1kHz input where `madmom-infer` would otherwise
  raise `NotImplementedError`; doesn't affect `TEST_WAV`, which is 44.1kHz native).
- `.github/workflows/publish.yml` no longer needs the `numpy`+`cython`-first,
  `--no-build-isolation` git-`madmom` install dance -- `pip install -e ".[dev]"` alone now
  resolves every dependency from PyPI.
- README/CLAUDE.md: removed the `madmom` install-workaround sections entirely (plain
  `pip install` / `uv add` now just works); added the rename notice above.

### Fixed

- **`tests/test_lead_sheet_regression.py`'s fixture was recorded against a silently-degraded
  path, not real madmom DBN tracking.** Investigating why this test's output changed after the
  swap surfaced that the *previously recorded* fixture (0.2.0) matches this project's own
  `librosa` beat-tracking **fallback** exactly (3/4 time, tempo 120, first-downbeat assumed at
  index 0 -- `_librosa_fallback`'s documented behavior), not the DBN's real detection for this
  audio (4/4 time, tempo 170, first downbeat at index 3 -- confirmed against real upstream
  `madmom` git HEAD, byte-for-byte identical beat times to the new `madmom-infer` path). In
  other words: whatever environment recorded the 0.2.0 fixture never actually exercised the
  advertised premium madmom DBN path -- it silently fell back to the librosa heuristic tracker
  the whole time, and the regression test was unknowingly pinning that degraded output. The
  fixture (`tests/fixtures/lead_sheet_test_wav_ref.ly`) is re-recorded against the now-working
  `madmom-infer` DBN path; `tests/_env_fixtures.py`'s environment fingerprint now also records
  `madmom_infer`'s version so a future madmom-infer release that shifts its DBN decode
  invalidates (skips, not silently passes) this fixture the same way a torch/numpy bump does.

### Removed

- Google Colab quickstart notebook (`examples/sheetsage_infer_colab.ipynb`) --
  maintaining a separate notebook environment alongside the PyPI package was
  more upkeep than the audience justified. No README/docs referenced it.

### Internal

- Split `infer.py`'s pipeline steps into `sheetsage/pipeline/` -- no behavior change. The four
  enums and module-level constants moved to `sheetsage/pipeline/types.py`; the private `_*`
  step helpers (`_init_extractor`, `_init_model`, `_closest_idx`, `_beat_tracking_with_hints`,
  `_split_into_chunks`, `_extract_features`, `_transcribe_chunks`, `_format_lead_sheet`) moved
  to `sheetsage/pipeline/steps.py`, verbatim. `infer.py` now keeps only the public `sheetsage()`
  function and the `main()` CLI, and re-exports every moved name so existing imports from
  `sheetsage.infer` keep working unchanged.

## [0.2.0] - 2026-07-11

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
