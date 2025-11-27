# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SheetSage is a music transcription system that converts audio to lead sheets (melody + chord symbols). This is an **inference-only** minimal version designed for easy integration. The system uses deep learning models (PyTorch-based) to extract musical features and transcribe audio into symbolic music notation.

## Core Architecture

### Main Pipeline Flow

The transcription pipeline (`sheetsage/infer.py:sheetsage()`) follows these stages:

1. **Audio Retrieval** - Load from file, bytes, or URL (`Status.FETCHING_AUDIO`)
2. **Beat Detection** - Detect beats and time signature using madmom (`Status.DETECTING_BEATS`)
3. **Feature Extraction** - Extract audio features at beat-synchronized intervals (`Status.EXTRACTING_FEATURES`)
4. **Transcription** - Run neural models to predict melody notes and chords (`Status.TRANSCRIBING`)
5. **Formatting** - Convert predictions to structured lead sheet format (`Status.FORMATTING`)

### Key Components

**Feature Extractors** (`sheetsage/representations/`)
- `Handcrafted` - Fast, CPU-based mel-spectrogram features (default)
- `JukeboxEmbeddings` - High-quality learned features from OpenAI Jukebox (GPU-required, optional dependency)

**Neural Models** (`sheetsage/modules/modules.py`)
- `EncOnlyTransducer` - Main transcription model (encoder-only architecture)
- `TransformerEncoder` - Transformer-based feature encoder
- `IdentityEncoder` - Passthrough encoder for linear probing

**Music Theory** (`sheetsage/theory/`)
- `LeadSheet` - Main output format containing meter, tempo, key, harmony, and melody
- `Melody` - Sequence of notes with pitch and duration
- `Harmony` - Sequence of chord symbols
- Music theory primitives: `Note`, `Chord`, `Key`, `Meter`, `Tempo`
- Format converters: `.as_lily()` (LilyPond), `.as_midi()` (MIDI)

**Beat Processing** (`sheetsage/beat_track.py`, `sheetsage/align.py`)
- Beat tracking with madmom library
- Time signature detection (3/4 or 4/4)
- Beat-to-time alignment functions

### Model Asset Management

Models are cached locally and downloaded automatically on first use:

- **Cache directory**: `~/.sheetsage/` (override with `SHEETSAGE_CACHE_DIR` env var)
- **Asset retrieval**: `sheetsage/assets.py:retrieve_asset()` handles downloads
- **Model initialization**: Models are lazy-loaded with `@cache()` decorator in `infer.py:_init_model()` and `_init_extractor()`

Model naming convention: `SHEETSAGE_V02_{INPUT_FEATS}_{TASK}_{MODEL|CFG|MOMENTS}`
- Input features: `HANDCRAFTED` or `JUKEBOX`
- Tasks: `MELODY` or `HARMONY`

## Development Commands

### Installation

```bash
# Standard installation (no Jukebox)
pip install -e .

# With optional dependencies for development
pip install -e ".[dev]"

# Install Jukebox separately (optional, for improved quality)
# Note: Jukebox has dependency conflicts and must be installed with --no-deps
pip install git+https://github.com/chrisdonahue/jukebox.git@7e0a38b679ff3f64987d8297d9d0eb5a046880c1 --no-deps
```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_inference.py

# Run with coverage
pytest --cov=sheetsage --cov-report=html

# Run tests verbosely
pytest -v
```

### Linting and Type Checking

```bash
# Run ruff linter
ruff check sheetsage/

# Run ruff with auto-fix
ruff check --fix sheetsage/

# Format code
ruff format sheetsage/

# Type checking with mypy
mypy sheetsage/
```

### Running Transcription

```bash
# Command-line interface
sheetsage audio.mp3 -o ./output

# With hints for better accuracy
sheetsage audio.mp3 -s 30 -e 90 --beats_per_minute_hint 120 -o ./output

# Use Jukebox embeddings (requires GPU with >=12GB VRAM)
sheetsage audio.mp3 -j -o ./output

# Get full CLI help
sheetsage --help
```

## Important Implementation Details

### Jukebox Dependency Quirk

**CRITICAL**: Jukebox cannot be included directly in `pyproject.toml` dependencies because:
- Jukebox's `requirements.txt` specifies `numba==0.48.0`
- `numba==0.48.0` is incompatible with Python 3.12
- Modern pip's strict dependency resolver will fail
- However, Jukebox works fine with newer numba versions at runtime

**Solution**: Install Jukebox separately with `--no-deps` flag (see `DOC_JUKEBOX_DEPENDENCY_EXPLANATION_2025-11-13.md`)

### Beat Resampling

Features are extracted at a fixed frame rate, then resampled to beat-synchronized "tertiaries" (sixteenth notes):
- Frame rates: `HANDCRAFTED` = 31.25 fps, `JUKEBOX` = 344.53 fps
- Tertiaries per beat: 4 (sixteenth note grid)
- Beat resampling averages feature frames between consecutive tertiaries

**Note**: Handcrafted features are normalized AFTER beat resampling (acknowledged as unusual but required by trained models).

### Chunking Strategy

Models have a maximum input length (`_MAX_TERTIARIES_PER_CHUNK = 384`). Long audio is split into chunks:
- Default: 8 measures per chunk (`measures_per_chunk=8`)
- Max supported: 24 measures per chunk
- Chunks cannot exceed `_JUKEBOX_CHUNK_DURATION_EDGE = 23.75` seconds
- Use `avoid_chunking_if_possible=True` (default) to process entire segment if it fits

### Legacy Behavior Mode

The `legacy_behavior=True` flag enables compatibility with original research code:
- Ignores `segment_end_hint`
- Transcribes exactly one max-length chunk from `segment_start_hint`
- Uses different padding logic for beat detection

## File Organization

```
sheetsage/
├── __init__.py          - Cache directory setup
├── infer.py             - Main transcription pipeline and CLI
├── assets.py            - Model download and asset management
├── beat_track.py        - Beat detection with madmom
├── align.py             - Beat-to-time alignment utilities
├── utils.py             - Audio I/O, engraving (LilyPond → PDF)
├── modules/
│   └── modules.py       - Neural network architectures
├── representations/
│   ├── __init__.py      - Feature extractor exports
│   ├── base.py          - Abstract base class
│   ├── handcrafted.py   - Mel-spectrogram features
│   ├── jukebox.py       - Jukebox embedding extractor
│   └── jukebox_modules/ - Vendored Jukebox code (embedding extraction only)
└── theory/
    ├── __init__.py      - Music theory exports
    ├── basic.py         - Pitch, interval primitives
    ├── internal.py      - Note, Chord, Key, Meter, Tempo classes
    ├── lead_sheet.py    - LeadSheet class with export methods
    ├── theorytab.py     - TheoryTab format conversions
    └── utils.py         - Key estimation utilities
```

## Common Patterns

### Adding a New Feature Extractor

1. Subclass `Representation` in `sheetsage/representations/base.py`
2. Implement `__call__(self, audio_path, offset=None, duration=None)` → `(frame_rate, features)`
3. Add to `InputFeats` enum in `infer.py`
4. Update `_INPUT_TO_FRAME_RATE` and `_INPUT_TO_DIM` dicts
5. Add initialization case in `_init_extractor()`

### Modifying the Transcription Model

Models are loaded from cached checkpoints (`.cfg.json` and `.pt` files). To use a different model:
1. Place config and weights in `~/.sheetsage/sheetsage/v0.2/`
2. Update `_init_model()` to load your checkpoint
3. Ensure output vocabulary size matches `_TASK_TO_VOCAB_SIZE`

### Extending Music Theory Classes

The theory classes (`Melody`, `Harmony`, `LeadSheet`) support:
- Serialization: `.as_lily()` (LilyPond), `.as_midi()` (MIDI)
- Deserialization: `.from_theorytab()` for TheoryTab format
- Internal representation uses pulse-based timing (beats × tertiaries_per_beat)

## Testing Approach

- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test full transcription pipeline with short audio clips
- **Pattern tests**: Verify expected usage patterns match documentation

When adding features:
1. Add unit tests for new functions/classes
2. Update integration tests if pipeline changes
3. Add pattern tests for new API usage examples

## External Dependencies

- **PyTorch** - Neural network inference
- **librosa** - Audio loading and processing
- **madmom** - Beat tracking and time signature detection
- **LilyPond** (system binary) - PDF generation from LilyPond notation
- **Jukebox** (optional) - High-quality audio embeddings

## Cache and Asset Locations

- Models: `~/.sheetsage/sheetsage/v0.2/`
- Jukebox models: `~/.sheetsage/jukebox/models/5b/` (if using Jukebox)
- Override cache dir: `export SHEETSAGE_CACHE_DIR=/custom/path`
