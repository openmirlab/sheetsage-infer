# SheetSage-Infer

**Inference-only version of SheetSage for music transcription.**

[![PyPI](https://img.shields.io/pypi/v/openmirlab-sheetsage-infer)](https://pypi.org/project/openmirlab-sheetsage-infer/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

AI-powered music transcription system that converts audio to lead sheets (melody + chord symbols) using deep learning models.

---

## 📌 Overview

**SheetSage-Infer** is an inference-only version of [SheetSage](https://github.com/chrisdonahue/sheetsage) for music transcription, optimized for easy deployment with vendored Jukebox modules.

---

## ✨ Features

- ✅ **CPU & GPU Support** - Handcrafted features (CPU) or Jukebox embeddings (GPU, via
  [`jukebox-infer`](https://pypi.org/project/jukebox-infer/))
- ✅ **Multiple Export Formats** - LilyPond notation, MIDI files, PDF generation
- ✅ **Audio from URLs** - Support for YouTube, Bandcamp, and other sources
- ✅ **Simple API** - High-level `sheetsage()` function

---

## 🚀 Quick Start

### Installation

**From PyPI:**

```bash
# Using pip
pip install openmirlab-sheetsage-infer

# madmom is required at runtime; a bare `pip install` resolves it to PyPI's
# madmom==0.16.1, which is broken (numpy incompatibilities). Install the
# git HEAD fix explicitly:
pip install git+https://github.com/CPJKU/madmom.git
```

```bash
# Using uv (recommended - faster): picks up the git-madmom fix
# automatically via this repo's [tool.uv.sources], no extra step needed
uv pip install openmirlab-sheetsage-infer

# Or add to your project with uv
uv add openmirlab-sheetsage-infer
```

> **Why the extra step?** `pyproject.toml` declares a bare `madmom` dependency
> (so `pip`'s resolver is satisfied) plus a `[tool.uv.sources]` pin to git
> HEAD (so `uv` resolves it correctly). Plain `pip install` has no equivalent
> to `[tool.uv.sources]`, so it silently installs the broken PyPI release
> unless you install the git version yourself, as above. We plan to replace
> this dependency with `madmom-infer` (our maintained, PyPI-published
> replacement) once it ships; see `pyproject.toml` for the tracking `TODO`.

**For Development:**

```bash
git clone https://github.com/openmirlab/sheetsage-infer.git
cd sheetsage-infer
pip install -e ".[dev]"
pip install git+https://github.com/CPJKU/madmom.git  # see note above
# Or, with uv (handles madmom automatically):
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

## 🔄 Changes from Original SheetSage

**SheetSage-Infer** has been modified from the [original SheetSage](https://github.com/chrisdonahue/sheetsage) to make it more suitable for library use and easier to maintain.

### Key Improvements

| Feature | Original | This Version |
|---------|----------|--------------|
| **Jukebox Dependency** | External, complex install | `pip install`-able via [`jukebox-infer`](https://pypi.org/project/jukebox-infer/) |
| **Test Coverage** | Limited | Import smoke tests + env-guarded regression fixtures |
| **Python Support** | 3.12+ only | 3.10, 3.11, 3.12 |
| **Build System** | Hatch | Setuptools (standard) |
| **Dependency Pins** | Loose | Explicit versions |

### What We Maintain

- ✅ All core transcription functionality
- ✅ Same neural network models
- ✅ Same output formats (LeadSheet, LilyPond, MIDI)
- ✅ Same API interface for `sheetsage()` function
- ✅ Same theory classes (Note, Chord, Melody, Harmony, etc.)

### What We Changed

- **Jukebox via `jukebox-infer`**: Uses the published, org-maintained
  [`jukebox-infer`](https://pypi.org/project/jukebox-infer/) package instead of
  vendoring a copy of the Jukebox codebase
- **Library-First Design**: Optimized for `pip install` and programmatic use
- **Better Dependency Management**: Explicit version pins and compatibility

---

## 🙏 Acknowledgments

### Original Research by Chris Donahue

**SheetSage-Infer** is built upon the excellent work of [SheetSage](https://github.com/chrisdonahue/sheetsage) by Chris Donahue. The original SheetSage represents a major advancement in music transcription, achieving state-of-the-art results through hierarchical transformer architectures.

### Research Paper

**[SheetSage: A Hierarchical Transformer for Audio to Lead Sheet Transcription](https://github.com/chrisdonahue/sheetsage)**

This work introduced hierarchical music transcription with melody and harmony extraction, enabling high-quality lead sheet generation from audio.

### Original Author

- **Chris Donahue** - Original SheetSage creator

### About This Implementation

This package was created to continue the excellent work by providing easier deployment (via `pip`-installable dependencies, including [`jukebox-infer`](https://pypi.org/project/jukebox-infer/) for Jukebox features), while preserving 100% of the original model quality and algorithms.

**What we maintain:**
- PyTorch 2.0+ compatibility
- Modern dependency management
- Inference-only packaging

**What remains unchanged:**
- All model architectures (100% original)
- All transcription algorithms (100% original)
- All model weights (100% original)
- All output formats (100% original)

---

## 📄 Citation

Please cite using the following bibtex entry:

```bibtex
@inproceedings{donahue2024sheetsage,
  title={SheetSage: A Hierarchical Transformer for Audio to Lead Sheet Transcription},
  author={Donahue, Chris},
  booktitle={ISMIR},
  year={2024}
}
```

**If you use SheetSage-Infer in your research, please cite the original SheetSage paper above.** This package is a maintenance fork to ensure easier deployment and continued compatibility - all credit for the models, algorithms, and research belongs to the original author.

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

## ⚠️ Limitations

- **Inference only** - No training capabilities
- **Jukebox features require GPU** - 12GB+ VRAM recommended for Jukebox embeddings
- **Jukebox features require ≥60s of (reported) total audio length** - the 5B
  prior's conditioning asserts total length is within `[60s, 600s)`; this is
  an inherent Jukebox architecture constraint (not something this project's
  code controls), so very short clips will raise an `AssertionError` on the
  `use_jukebox=True` path even though the CPU (handcrafted-features) path has
  no such floor
- **LilyPond required for PDF** - Optional dependency for PDF generation
- **Time signatures** - Currently supports 4/4 and 3/4 only
- **Audio length** - Best results with segments 30-300 seconds

---

## 🤝 Contributing

We welcome contributions! Please:

1. Follow the code style (ruff/black)
2. Add tests for new features
3. Submit PRs with clear descriptions

### Development Setup

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format and lint code
ruff format . && ruff check .
```

---

## 📞 Support

For issues and questions:
- **GitHub Issues**: [github.com/openmirlab/sheetsage-infer/issues](https://github.com/openmirlab/sheetsage-infer/issues)
- **Examples**: `examples/` directory

---

## 🔗 Links

- **Original SheetSage**: https://github.com/chrisdonahue/sheetsage
- **This Repository**: https://github.com/openmirlab/sheetsage-infer
- **PyPI Package**: https://pypi.org/project/openmirlab-sheetsage-infer/

---

**Made with ❤️ for the ML community**

Based on the excellent work by Chris Donahue and the SheetSage project.
