# sheetsage-infer

Inference-only fork of [SheetSage](https://github.com/chrisdonahue/sheetsage) (audio -> lead
sheet: melody + chords). Packaged for `pip install openmirlab-sheetsage-infer`.

## Orientation

- `sheetsage/infer.py` -- the pipeline (`sheetsage()`): beat-track -> chunk -> extract features
  -> transcribe -> assemble `LeadSheet`. Start here.
- `sheetsage/representations/` -- feature extractors: `handcrafted.py` (CPU, always available),
  `jukebox.py` (GPU, optional, imports the `jukebox_infer` package).
- `sheetsage/theory/` -- music-theory classes (`Note`, `Chord`, `LeadSheet`, LilyPond/MIDI export).
- `sheetsage/assets.py` -- resolves named model weights / test fixtures to checksum-verified
  local files, downloaded on first use into `~/.sheetsage` (see `sheetsage/assets/*.json`).
- `sheetsage/beat_track.py` -- madmom DBN downbeat tracker with a librosa fallback.

## Known constraints (read before touching install/dependency config)

- **madmom**: the bare `madmom` PyPI release (0.16.1) is broken under plain `pip install`; this
  project needs the git HEAD fix. `uv` picks this up automatically via `[tool.uv.sources]` in
  `pyproject.toml`; plain `pip` users must separately run
  `pip install git+https://github.com/CPJKU/madmom.git`. See README's Installation section.
  A TODO in `pyproject.toml` tracks swapping to `madmom-infer` (org-maintained replacement) once
  it's published to PyPI -- it is not yet (Phase 1+2 done locally only).
- **jukebox_infer**: `sheetsage/representations/jukebox.py` imports the real, published
  `jukebox_infer` package (not vendored). It carries a small documented monkeypatch working
  around a real upstream bug (`RangeEmbedding` not casting `n_time` to `int`, which surfaces as
  a `TypeError` from `torch.Tensor.view`) -- see the comment block at the top of that file. If
  `jukebox_infer` ships a fix, remove the patch.
- **Licensing is two-tier**: code is MIT; weights and HookTheory-derived data downloaded via
  `sheetsage.assets` are CC BY-NC-SA (as is madmom's bundled DBN downbeat model). See `NOTICE`.

## Testing

`tests/` has import smoke tests, bit-exact regression fixtures (env-guarded: they SKIP rather
than fail on a torch/librosa/numpy/python mismatch, since those genuinely shift floating-point
output -- don't "fix" a skip by loosening a fixture's tolerance without checking whether the env
actually matches first), and an opt-in GPU Jukebox smoke test
(`SHEETSAGE_RUN_JUKEBOX_TESTS=1 uv run pytest tests/test_jukebox_smoke.py`). Run the default
suite with `uv run pytest tests/`.

When changing `sheetsage/representations/jukebox.py`, `make_models.py`-adjacent code, or
anything in the feature-extraction path: re-run the CPU regression test at minimum, and the GPU
smoke test if you have a free GPU -- a prior campaign found the Jukebox path silently broken
end-to-end (see CHANGELOG's Unreleased section) precisely because nothing exercised it.

## Conventions

- Nav-style headers (what a file is, what it reads/is read by) on load-bearing modules -- keep
  them in sync when you restructure imports.
- CHANGELOG.md: add entries under `[Unreleased]` for behavior changes, not just version bumps.
- Commit messages / PRs: no emojis.
