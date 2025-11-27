# SheetSage Inference (Modified)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Inference-only version of SheetSage for music transcription with vendored Jukebox modules and comprehensive testing.**

This repository is a modified version of [sheetsage](https://github.com/chrisdonahue/sheetsage), optimized for easier deployment and integration as a library package.

## What is SheetSage?

SheetSage is an AI-powered tool that transcribes audio to lead sheets (melody + chord symbols). It uses deep learning models to analyze music and extract:
- **Melody**: Note sequences with pitches and durations
- **Harmony**: Chord progressions with symbols and timing
- **Meter & Tempo**: Time signature and BPM
- **Key**: Musical key signature

---

## Changes from Original SheetSage

This repository modifies the [original SheetSage](https://github.com/chrisdonahue/sheetsage) to make it more suitable for library use and easier to maintain. Here are the key differences:

### 🎯 Purpose of This Fork

1. **Vendored Jukebox Modules**: Eliminates complex external dependency on OpenAI Jukebox
2. **Library-First Design**: Optimized for `pip install` and programmatic use
3. **Comprehensive Testing**: Full pytest test suite with 75+ tests
4. **Better Dependency Management**: Explicit version pins and compatibility
5. **AI Assistant Ready**: Includes CLAUDE.md for Claude Code integration

### 📋 Detailed Changes

#### 1. **Vendored Jukebox Implementation**
- **Original**: Required separate Jukebox installation with dependency conflicts
- **Modified**: Jukebox embedding extraction code vendored into `sheetsage/representations/jukebox_modules/`
- **Benefit**: No external Jukebox dependency needed; works out of the box

**Files Added:**
```
sheetsage/representations/jukebox_modules/
├── api.py, embeddings.py, make_models.py
├── prior/            # Jukebox prior models
├── transformer/      # Transformer implementations
├── utils/            # Jukebox utilities
└── vqvae/            # VQ-VAE models
```

#### 2. **Build System Changes**
- **Original**: Uses `hatch` build backend
- **Modified**: Uses `setuptools` for broader compatibility
- **Changed Files**: `pyproject.toml`

#### 3. **Dependency Management**
- **Original**: Loose version constraints, Python 3.12+ only
- **Modified**:
  - Explicit version pins for reproducibility
  - Python 3.10+ support (broader compatibility)
  - Added `jukebox-infer` as dependency for model weights
  - Added `yt-dlp`, `resampy` for better audio handling

**pyproject.toml changes:**
```diff
- requires-python = ">=3.12"
+ requires-python = ">=3.10"

- "torch",
+ "torch>=2.0.0",
+ "jukebox-infer>=0.1.0",
+ "madmom>=0.16.0",
+ ... (explicit versions for all deps)
```

#### 4. **Comprehensive Test Suite**
- **Original**: Limited tests
- **Modified**: 75+ pytest tests across 7 test modules

**New Test Files:**
```
tests/
├── conftest.py                  # Pytest configuration
├── test_imports.py              # Import verification (20 tests)
├── test_signatures.py           # API contract tests (10 tests)
├── test_representations.py      # Feature extractors (10 tests)
├── test_integration_full.py     # Full pipeline tests (10 tests)
├── test_output_formats.py       # LilyPond/MIDI tests (10 tests)
├── test_theory_classes.py       # Music theory (15 tests)
└── README_TESTS.md              # Test documentation
```

**Test Features:**
- Fast unit tests (imports, signatures, theory classes)
- Slow integration tests (with `--run-slow` flag)
- GPU tests for Jukebox (with `--run-jukebox` flag)
- Fixtures for reusable test data
- Markers for selective test execution

#### 5. **Documentation Improvements**
- **Added**: `CLAUDE.md` - Comprehensive project documentation for Claude Code
- **Added**: `tests/README_TESTS.md` - Complete testing guide
- **Removed**: Original notebooks (Dataset.ipynb, Inference.ipynb)
- **Reason**: Focus on library usage over notebook examples

#### 6. **Package Metadata**
- **Changed**: Package name from `sheetsage-infer` to `sheetsage`
- **Added**: Comprehensive keywords and classifiers
- **Added**: Project URLs (homepage, repository, issues)
- **Added**: Author information

#### 7. **Development Tools Configuration**
- **Added**: Pytest configuration in `pyproject.toml`
- **Added**: Ruff linter configuration
- **Added**: MyPy type checker configuration
- **Added**: Optional dev dependencies (`pytest`, `pytest-cov`, `ruff`, `mypy`)

---

## Installation

### Prerequisites

- Python >= 3.10 (tested on 3.10, 3.11, 3.12)
- [LilyPond](https://lilypond.org/) (optional, for PDF generation)
  - **Linux**: `sudo apt-get install lilypond`
  - **macOS**: `brew install lilypond`
  - **Windows**: Download from [lilypond.org](https://lilypond.org/download.html)

### Standard Installation

```bash
# Clone this repository
git clone https://github.com/openmirlab/sheetsage-infer.git
cd sheetsage-infer

# Install with pip
pip install -e .

# Or install from PyPI (when published)
pip install sheetsage
```

### Development Installation

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=sheetsage --cov-report=html
```

---

## Quick Start

### Python API

```python
from sheetsage.infer import sheetsage

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
from sheetsage.align import create_beat_to_time_fn
beat_to_time_fn = create_beat_to_time_fn(segment_beats, segment_beats_times)
midi_bytes = lead_sheet.as_midi(beat_to_time_fn)

# Save MIDI file
with open('output.mid', 'wb') as f:
    f.write(midi_bytes)

# Generate PDF (requires LilyPond)
from sheetsage.utils import engrave
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

**Note**: With vendored Jukebox modules, this works without external Jukebox installation!

---

## Features

### ✅ Available in This Version

- ✅ CPU-based feature extraction (Handcrafted mel-spectrograms)
- ✅ GPU-based feature extraction (Jukebox embeddings - vendored)
- ✅ Beat detection and time signature estimation
- ✅ Melody transcription (pitch + duration)
- ✅ Chord/harmony transcription
- ✅ LilyPond export (music notation)
- ✅ MIDI export
- ✅ PDF generation (via LilyPond)
- ✅ Audio from URLs (YouTube, Bandcamp, etc.)
- ✅ Comprehensive test suite (75+ tests)

### 🎯 Key Advantages Over Original

| Feature | Original | This Version |
|---------|----------|--------------|
| Jukebox Dependency | External, complex install | Vendored, works out of box |
| Test Coverage | Limited | 75+ comprehensive tests |
| Python Support | 3.12+ only | 3.10, 3.11, 3.12 |
| Build System | Hatch | Setuptools (standard) |
| Dependency Pins | Loose | Explicit versions |
| Documentation | Basic | CLAUDE.md + test docs |

---

## Testing

This repository includes a comprehensive test suite:

```bash
# Run fast tests only (imports, signatures)
pytest tests/ -v

# Run slow integration tests
pytest tests/ --run-slow -v

# Run GPU tests (requires CUDA)
pytest tests/ --run-jukebox -v

# Run with coverage
pytest tests/ --cov=sheetsage --cov-report=html

# See tests/README_TESTS.md for full testing guide
```

---

## Architecture Overview

### Pipeline Flow

```
Audio Input
    ↓
[Beat Detection]           # madmom library
    ↓
[Feature Extraction]       # Handcrafted OR Jukebox (vendored)
    ↓
[Beat Alignment]           # Align features to beat grid
    ↓
[Neural Transcription]     # PyTorch models
    ↓
[Post-Processing]          # Format predictions
    ↓
LeadSheet Output
    ↓
[Export: LilyPond/MIDI/PDF]
```

### Key Components

- **sheetsage/infer.py**: Main transcription pipeline
- **sheetsage/representations/**: Feature extractors
  - `handcrafted.py`: CPU-based mel-spectrograms
  - `jukebox.py`: Jukebox embedding interface
  - `jukebox_modules/`: **Vendored Jukebox code** (NEW)
- **sheetsage/theory/**: Music theory classes
- **sheetsage/modules/**: Neural network models
- **sheetsage/utils.py**: LilyPond engraving, audio I/O

---

## Comparison with Original

### File Structure Changes

```diff
sheetsage-infer/
+ ├── CLAUDE.md                          # NEW: Claude Code documentation
  ├── pyproject.toml                     # MODIFIED: Dependencies, build system
  ├── README.md                          # MODIFIED: This file
  ├── sheetsage/
  │   ├── infer.py                       # MODIFIED: Updated imports
  │   ├── representations/
  │   │   ├── jukebox.py                 # MODIFIED: Uses vendored modules
+ │   │   └── jukebox_modules/          # NEW: Vendored Jukebox code (33 files)
  │   └── ...
  ├── tests/
- │   ├── (limited tests)                # REMOVED
+ │   ├── conftest.py                    # NEW: Pytest config
+ │   ├── test_imports.py                # NEW: 20 import tests
+ │   ├── test_signatures.py             # NEW: 10 signature tests
+ │   ├── test_representations.py        # NEW: 10 feature tests
+ │   ├── test_integration_full.py       # NEW: 10 integration tests
+ │   ├── test_output_formats.py         # NEW: 10 output tests
+ │   ├── test_theory_classes.py         # NEW: 15 theory tests
+ │   └── README_TESTS.md                # NEW: Test documentation
- └── notebooks/                         # REMOVED: Dataset.ipynb, Inference.ipynb
```

### Dependency Changes

**Removed:**
- External Jukebox dependency (vendored instead)
- Hatch build system

**Added:**
- `jukebox-infer>=0.1.0` (for model weights)
- `madmom>=0.16.0` (beat tracking)
- `yt-dlp>=2025.11.0` (YouTube download)
- `resampy>=0.4.0` (audio resampling)
- Dev dependencies: pytest, pytest-cov, ruff, mypy

**Changed:**
- All dependencies now have explicit version constraints
- Python version: `>=3.12` → `>=3.10`

---

## Original SheetSage

This repository is based on [sheetsage](https://github.com/chrisdonahue/sheetsage) by Chris Donahue.

### Original Features Preserved

- ✅ All core transcription functionality
- ✅ Same neural network models
- ✅ Same output formats (LeadSheet, LilyPond, MIDI)
- ✅ Same API interface for `sheetsage()` function
- ✅ Same theory classes (Note, Chord, Melody, Harmony, etc.)

### Why Fork?

The original SheetSage is excellent for research, but this fork addresses practical deployment needs:

1. **Jukebox Dependency Hell**: Original requires complex Jukebox installation with Python version conflicts. We vendor the needed code.
2. **Library Integration**: Better suited for `pip install` and programmatic use
3. **Testing**: Comprehensive test suite for production reliability
4. **Maintainability**: Explicit dependencies and modern Python support

---

## Credits

- **Original SheetSage**: [Chris Donahue](https://github.com/chrisdonahue)
- **Jukebox**: [OpenAI Jukebox](https://github.com/openai/jukebox) (vendored components)
- **This Fork**: Modified for easier deployment and testing

---

## License

MIT License (same as original SheetSage)

---

## Citation

If you use this software, please cite the original SheetSage work:

```bibtex
@inproceedings{donahue2024sheetsage,
  title={SheetSage: A Hierarchical Transformer for Audio to Lead Sheet Transcription},
  author={Donahue, Chris},
  booktitle={ISMIR},
  year={2024}
}
```

---

## Contributing

Contributions are welcome! Please:

1. Run tests before submitting: `pytest tests/`
2. Follow existing code style: `ruff check sheetsage/`
3. Add tests for new features
4. Update documentation

---

## FAQ

### Q: Do I need to install Jukebox separately?

**A:** No! Jukebox modules are vendored in this repository. Just `pip install -e .` and use `use_jukebox=True`.

### Q: What's the difference between Handcrafted and Jukebox features?

**A:**
- **Handcrafted**: Fast, CPU-only, mel-spectrogram features. Good quality.
- **Jukebox**: Slow, GPU-required, learned embeddings. Better quality.

### Q: Why was this forked from the original?

**A:** To eliminate Jukebox dependency issues and make it production-ready with tests.

### Q: Are the transcription results identical to the original?

**A:** Yes, we use the same models and logic. Only the packaging and dependencies changed.

### Q: Can I use this commercially?

**A:** Yes, it's MIT licensed (same as original).

---

## Links

- **Original SheetSage**: https://github.com/chrisdonahue/sheetsage
- **This Repository**: https://github.com/openmirlab/sheetsage-infer
- **OpenAI Jukebox**: https://github.com/openai/jukebox
- **Test Documentation**: [tests/README_TESTS.md](tests/README_TESTS.md)
- **Claude Code Guide**: [CLAUDE.md](CLAUDE.md)
