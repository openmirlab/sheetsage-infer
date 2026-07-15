# SheetSage-Infer

**Inference-only version of SheetSage for music transcription.**

[![PyPI](https://img.shields.io/pypi/v/sheetsage-infer)](https://pypi.org/project/sheetsage-infer/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

AI-powered music transcription system that converts audio to lead sheets (melody + chord symbols) using deep learning models.

> **Renamed from `openmirlab-sheetsage-infer`.** Releases through 0.2.0 were published
> as `openmirlab-sheetsage-infer` on PyPI; that name is now deprecated and will not
> receive further releases. Starting with 0.2.1, this package publishes as
> **`sheetsage-infer`** -- update your `pip install` / `uv add` commands accordingly.
> The import name is unchanged (`import sheetsage`).

---

## Why this exists

[SheetSage](https://github.com/chrisdonahue/sheetsage) is Chris Donahue's
audio-to-lead-sheet transcription system, introduced in the ISMIR 2022 paper
*Melody Transcription via Generative Pre-Training* (see [Citation](#citation)
below). The original repository targets Python 3.12+ only, builds with Hatch,
and depends on Jukebox and madmom in ways that make it hard to install as a
library: Jukebox required vendoring the OpenAI Jukebox codebase directly, and
madmom's own PyPI release has for years been git-HEAD-only / broken.

**SheetSage-Infer** is an inference-only repackaging aimed at library use:
same models, same theory classes, same output formats, but published as an
ordinary `pip install`-able package with explicit dependency pins, support
for Python 3.10-3.12, and its two hardest dependencies swapped for
OpenMIRLab's own maintained, PyPI-published replacements
([`jukebox-infer`](https://pypi.org/project/jukebox-infer/) and
[`madmom-infer`](https://pypi.org/project/madmom-infer/)) instead of a
vendored copy and a broken upstream package.

## Acknowledgments

SheetSage-Infer is an independent repackaging built on top of Chris Donahue's
research. It is not affiliated with or endorsed by the original author.

- **Original research & repository**: [SheetSage](https://github.com/chrisdonahue/sheetsage)
  was created by **Chris Donahue** (repository created 2022-06-10, per
  GitHub). See [Citation](#citation) below for the paper's full author list
  (Donahue, Thickstun, and Liang).
- **Source repository**: [github.com/chrisdonahue/sheetsage](https://github.com/chrisdonahue/sheetsage)
- **Weights & data hosts**: SheetSage's own trained checkpoints are fetched
  from its S3 bucket (`sheetsage.s3.amazonaws.com`) for the handcrafted-feature
  models, and from a third-party HuggingFace mirror
  ([`mku121/sheetsage`](https://huggingface.co/mku121/sheetsage), originally
  distributed via mega.nz) for the Jukebox-feature models. HookTheory-derived
  training/eval data comes from
  [github.com/chrisdonahue/sheetsage-data](https://github.com/chrisdonahue/sheetsage-data).
  The Jukebox model itself (`JUKEBOX_VQVAE`, `JUKEBOX_LM`) is OpenAI's
  original release, hosted at `openaipublic.azureedge.net` and mirrored on
  the same HuggingFace repo.
- **[`jukebox-infer`](https://pypi.org/project/jukebox-infer/)**: OpenMIRLab's
  maintained, PyPI-published replacement for vendoring OpenAI's Jukebox
  codebase directly.
- **[`madmom-infer`](https://pypi.org/project/madmom-infer/)**: OpenMIRLab's
  maintained, PyPI-published replacement for `madmom` (whose own PyPI release
  has long been git-HEAD-only/broken). The underlying
  [madmom](https://github.com/CPJKU/madmom) project and its bundled DBN
  downbeat-tracking model are by the original madmom project (Institute of
  Computational Perception, JKU Linz / CPJKU) -- see that repository for its
  own author/contributor list.

## Citation

Please cite using the following bibtex entry:

```bibtex
@inproceedings{donahue2022melody,
  title={Melody Transcription via Generative Pre-Training},
  author={Donahue, Chris and Thickstun, John and Liang, Percy},
  booktitle={Proceedings of the 23rd International Society for Music Information Retrieval Conference (ISMIR)},
  year={2022}
}
```

*(Corrected 2026-07-12: this README previously cited a nonexistent title,
"SheetSage: A Hierarchical Transformer for Audio to Lead Sheet Transcription"
(Donahue, ISMIR 2024). The real paper, verified against its
[arXiv page](https://arxiv.org/abs/2212.01884), is "Melody Transcription via
Generative Pre-Training" by Donahue, Thickstun, and Liang, published at
ISMIR 2022. The copyright year in [NOTICE](NOTICE)/[LICENSE](LICENSE), 2022,
was already correct -- it matches the original repository's creation date --
so only the Citation entry needed fixing.)*

**If you use SheetSage-Infer in your research, please cite the original paper above.** This package is a maintenance fork to ensure easier deployment and continued compatibility - all credit for the models, algorithms, and research belongs to the original author.

---

## ✨ Features

- ✅ **CPU & GPU Support** - Handcrafted features (CPU) or Jukebox embeddings (GPU, via
  [`jukebox-infer`](https://pypi.org/project/jukebox-infer/))
- ✅ **Multiple Export Formats** - LilyPond notation, MIDI files, PDF generation
- ✅ **Audio from URLs** - Support for YouTube, Bandcamp, and other sources
- ✅ **Simple API** - High-level `sheetsage()` function

---

## Scope

**In scope:**
- Inference-only transcription: audio -> lead sheet (melody + chord symbols),
  via handcrafted CPU features or Jukebox GPU embeddings
- LilyPond / MIDI / PDF export via the `LeadSheet` API
- Library-first packaging: pip-installable, explicit dependency pins,
  Python 3.10-3.12
- Swapping the original's hard-to-install dependencies (vendored Jukebox,
  PyPI-broken madmom) for maintained PyPI replacements (`jukebox-infer`,
  `madmom-infer`)

**Out of scope, forever:**
- Training or fine-tuning SheetSage's models -- this is an inference-only
  fork; see [SheetSage](https://github.com/chrisdonahue/sheetsage) for
  training code
- Redistributing SheetSage's trained weights or HookTheory-derived data --
  these remain CC BY-NC-SA (non-commercial, share-alike) and are downloaded
  at runtime, never bundled (see
  ["What this project will NEVER bundle"](#what-this-project-will-never-bundle))
- Time signatures beyond 4/4 and 3/4 (inherited limitation of the original
  model)

**Known constraints:**
- Jukebox features require a GPU with ≥12GB VRAM
- Jukebox features require ≥60s of (reported) total audio length -- the 5B
  prior's conditioning asserts total length is within `[60s, 600s)`; this is
  an inherent Jukebox architecture constraint (not something this project's
  code controls), so very short clips raise an `AssertionError` on the
  `use_jukebox=True` path even though the CPU (handcrafted-features) path
  has no such floor
- LilyPond is required for PDF export only (optional dependency)
- Best transcription results on 30-300 second segments

---

## 🚀 Quick Start

### Installation

**From PyPI:**

```bash
# Using pip
pip install sheetsage-infer
```

```bash
# Using uv (recommended - faster)
uv pip install sheetsage-infer

# Or add to your project with uv
uv add sheetsage-infer
```

madmom was replaced by [`madmom-infer`](https://pypi.org/project/madmom-infer/) (our
maintained, PyPI-published replacement) as of 0.2.1 -- plain `pip install` now works
with no extra steps or git installs.

**For Development:**

```bash
git clone https://github.com/openmirlab/sheetsage-infer.git
cd sheetsage-infer
pip install -e ".[dev]"
# Or, with uv:
uv sync --extra dev
```

### Prerequisites

- **Python**: ≥3.10 (tested on 3.10, 3.11, 3.12)
- **[LilyPond](https://lilypond.org/)** (optional, for PDF generation)
  - **Linux**: `sudo apt-get install lilypond`
  - **macOS**: `brew install lilypond`
  - **Windows**: Download from [lilypond.org](https://lilypond.org/download.html)

### Simple API (Recommended for Python)

```python
from sheetsage.infer import sheetsage
from sheetsage.utils import engrave
from sheetsage.align import create_beat_to_time_fn

# Transcribe audio URL
lead_sheet, segment_beats, segment_beats_times = sheetsage(
    'https://example.com/audio.mp3',
    use_jukebox=False,           # Use fast CPU-based features
    segment_start_hint=30,       # Start at 30 seconds
    segment_end_hint=60,         # End at 60 seconds
    beats_per_minute_hint=120    # Hint for BPM (improves accuracy)
)

# Export to LilyPond
lily_code = lead_sheet.as_lily()
print(lily_code)

# Export to MIDI
beat_to_time_fn = create_beat_to_time_fn(segment_beats, segment_beats_times)
midi_bytes = lead_sheet.as_midi(beat_to_time_fn)

# Save MIDI file
with open('output.mid', 'wb') as f:
    f.write(midi_bytes)

# Generate PDF (requires LilyPond)
pdf_bytes = engrave(lily_code, out_format='pdf')
with open('leadsheet.pdf', 'wb') as f:
    f.write(pdf_bytes)
```

### Using Jukebox Features (Higher Quality, GPU Required)

```python
from sheetsage.infer import sheetsage

# Requires GPU with >=12GB VRAM
lead_sheet, beats, beat_times = sheetsage(
    'audio.mp3',
    use_jukebox=True,  # Use Jukebox embeddings (vendored)
    segment_start_hint=0,
    segment_end_hint=30,
    beats_per_minute_hint=100
)
```

**Note**: Jukebox features require GPU with ≥12GB VRAM. Vendored modules work without external installation.

### Command-Line Interface

```bash
# Basic transcription
python -m sheetsage.infer audio.mp3

# With options
python -m sheetsage.infer audio.mp3 \
    --segment_start_hint 30 \
    --segment_end_hint 60 \
    --beats_per_minute_hint 120 \
    --output_dir ./output

# See all options
python -m sheetsage.infer --help
```

---

## 📋 Requirements

- **Python**: ≥3.10
- **PyTorch**: ≥2.0.0
- **GPU**: Optional, but recommended for Jukebox features (12GB+ VRAM)
- **OS**: Linux, macOS, Windows

---

## ⚡ Performance

Transcription speed depends on audio length and feature extraction method:

- **Handcrafted features (CPU)**: ~1-5 seconds per minute of audio
- **Jukebox features (GPU)**: ~30-60 seconds per minute of audio (requires GPU with ≥12GB VRAM)

> **Note**: Performance depends on audio length, hardware, and feature extraction method. Jukebox features provide higher quality but are slower.

---

## 📚 Examples

See `examples/` directory for usage examples:
- `basic_transcription.py` - Basic usage
- `jukebox_transcription.py` - GPU-based transcription
- `hooktheory_example.py` - Working with Hooktheory data

---

## 🏗️ Project Structure

```
sheetsage-infer/
├── sheetsage/                    # Main package
│   ├── infer.py                 # Main transcription pipeline (public sheetsage()/CLI)
│   ├── pipeline/                 # Pipeline enums/constants + step helpers (used by infer.py)
│   ├── align.py                 # Beat-to-time alignment
│   ├── beat_track.py             # Beat detection
│   ├── utils.py                 # LilyPond engraving, audio I/O
│   ├── assets.py                 # Asset management
│   ├── assets/                   # Asset JSON files
│   │   ├── hooktheory.json
│   │   ├── jukebox.json
│   │   ├── rwc.json
│   │   ├── sheetsage.json
│   │   └── test.json
│   ├── modules/                  # Neural network models
│   │   └── modules.py            # Transformer architectures
│   ├── representations/          # Feature extractors
│   │   ├── handcrafted.py       # CPU-based mel-spectrograms
│   │   └── jukebox.py            # Jukebox embedding interface (imports jukebox-infer)
│   └── theory/                   # Music theory classes
│       ├── lead_sheet.py         # LeadSheet class with export methods
│       ├── basic.py              # Basic music theory primitives
│       ├── internal.py           # Internal theory classes
│       ├── theorytab.py          # TheoryTab integration
│       └── utils.py              # Theory utilities
├── tests/                        # Import smoke tests + env-guarded regression fixtures
├── examples/                     # Example scripts
│   ├── basic_transcription.py    # Basic usage
│   ├── jukebox_transcription.py  # GPU-based transcription
│   ├── hooktheory_example.py     # Hooktheory data examples
│   ├── hooktheory_simple.py     # Simple Hooktheory example
│   └── transcribe_hooktheory_segments.py  # Hooktheory segment transcription
├── hooktheory_data/              # Test data
│   ├── Hooktheory_Test_MIDI.tar.gz
│   └── Hooktheory_Test_Segments.json
├── .github/                      # GitHub configuration
│   └── workflows/
│       └── publish.yml           # PyPI publishing workflow (runs tests before build)
├── pyproject.toml               # Project configuration (single source of truth for deps)
├── uv.lock                      # UV lock file
├── CHANGELOG.md                 # Notable changes
├── CLAUDE.md                    # Orientation for AI coding agents working in this repo
├── LICENSE                      # MIT License (code)
├── NOTICE                       # License layering: code (MIT) vs weights/data (CC BY-NC-SA)
└── README.md                    # This file
```

---

## What this project will NEVER bundle

sheetsage-infer downloads trained model weights and HookTheory-derived data
at runtime via `sheetsage.assets` (`sheetsage/assets/*.json` manifests,
resolved by `retrieve_asset()` in `sheetsage/assets.py`) into a local cache
directory (`~/.sheetsage` by default). None of these are ever committed to
this repository or bundled into the PyPI sdist/wheel:

- **SheetSage's own trained checkpoints** (handcrafted-feature and
  Jukebox-feature harmony/melody models) are fetched from SheetSage's S3
  bucket or a third-party HuggingFace mirror on first use, checksum-verified
  against the manifest before being trusted.
- **HookTheory-derived training/eval data** (segments, MIDI) is fetched from
  `github.com/chrisdonahue/sheetsage-data` on first use. Because it's derived
  from user contributions on HookTheory (see HookTheory's
  [ToS](https://forum.hooktheory.com/tos)), it is **CC BY-NC-SA 3.0** --
  non-commercial, share-alike -- a materially different, more restrictive
  license than this repo's own MIT code license. See [NOTICE](NOTICE).
- **The Jukebox model weights** (`JUKEBOX_VQVAE`, `JUKEBOX_LM`) are OpenAI's
  original release, fetched from OpenAI's own CDN (or the HuggingFace mirror)
  under OpenAI's Jukebox terms -- see
  [github.com/openai/jukebox](https://github.com/openai/jukebox) for the
  upstream license. This package does not modify or relicense them.
- **madmom's bundled DBN downbeat-tracking model** (used via `madmom-infer`)
  is separately CC BY-NC-SA 4.0, distinct from madmom's own BSD-2-Clause
  source code.

If you redistribute anything built with this package -- transcriptions
derived from HookTheory data, a model fine-tuned on the downloaded
checkpoints, etc. -- check the license of whichever asset you actually used;
this package's own MIT license only covers its own code, not what it
downloads on your behalf. See [NOTICE](NOTICE) for the full breakdown.

---

## Development

We welcome contributions! Please:

1. Follow the code style (ruff/black)
2. Add tests for new features
3. Submit PRs with clear descriptions

```bash
# Install dependencies
pip install -e ".[dev]"
# Or, with uv:
uv sync --extra dev

# Run tests
uv run pytest tests/

# Format and lint code
uv run ruff format . && uv run ruff check .
```

See [`CLAUDE.md`](CLAUDE.md) for architecture notes, known constraints, and
orientation for AI coding agents working in this repo.

---

## 📄 License

Licensing is two-tier — see [NOTICE](NOTICE) for the full breakdown:

- **Code** (this repository, including code adapted from
  [SheetSage](https://github.com/chrisdonahue/sheetsage)): **MIT License**.
  Copyright (c) 2022 Chris Donahue (Original SheetSage); Copyright (c) 2025
  SheetSage-Infer contributors. See [LICENSE](LICENSE) for details.
- **Weights and data** downloaded at runtime via `sheetsage.assets` (trained
  model checkpoints, HookTheory-derived segments/MIDI) are **CC BY-NC-SA 3.0**,
  since they derive from user contributions on HookTheory. madmom's bundled
  DBN downbeat-tracking model is similarly CC BY-NC-SA (separate from
  madmom's own BSD-2-Clause source code). These are fetched on demand, not
  bundled in this package's source distribution or wheel.

---

## 📞 Support

- **GitHub Issues**: [github.com/openmirlab/sheetsage-infer/issues](https://github.com/openmirlab/sheetsage-infer/issues)
- **Examples**: `examples/` directory
- **Original SheetSage**: https://github.com/chrisdonahue/sheetsage
- **This Repository**: https://github.com/openmirlab/sheetsage-infer
- **PyPI Package**: https://pypi.org/project/sheetsage-infer/ (previously
  `openmirlab-sheetsage-infer`, deprecated as of 0.2.1)

---

**Made with ❤️ for the ML community**

Based on the excellent work by Chris Donahue and the SheetSage project.
# Lifecycle API

Use `SheetSageSession` when an explicit lifecycle is required:
`load()` → ready-only `infer()` → `release()`/`close()`. Sessions are
independent; the package-owned `sheetsage/config/checkpoints.toml` records
checkpoint URLs and provenance. The existing `sheetsage()` one-shot function
remains available for backward compatibility.
