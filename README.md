# SheetSage Inference

Minimal inference-only version of SheetSage for music transcription.

## Overview

SheetSage is a tool for transcribing audio to lead sheets (melody + chord symbols). This minimal version contains only the essential code needed for inference, making it easy to integrate into your projects.

## Installation

### Prerequisites

- Python >= 3.12
- [LilyPond](https://lilypond.org/) (for generating PDF lead sheets)
  - **Linux**: `sudo apt-get install lilypond` or `sudo yum install lilypond`
  - **macOS**: `brew install lilypond`
  - **Windows**: Download from [lilypond.org](https://lilypond.org/download.html)

### Install Package

```bash
# Clone or download this repository
cd sheetsage-infer

# Install in development mode
pip install -e .

# Or install dependencies from requirements.txt
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Command Line Usage

After installation, you can use the `sheetsage` command:

```bash
# Basic transcription
sheetsage audio.mp3

# Specify output directory
sheetsage audio.mp3 -o ./output

# Add metadata
sheetsage audio.mp3 -t "Song Title" -a "Artist Name" -o ./output

# Transcribe a segment (from 30s to 90s)
sheetsage audio.mp3 -s 30 -e 90 -o ./output

# Use Jukebox embeddings (requires GPU with >=12GB VRAM)
sheetsage audio.mp3 -j -o ./output

# Get help
sheetsage --help
```

### Python API Usage

```python
from sheetsage.infer import sheetsage
from sheetsage.utils import engrave
from sheetsage.align import create_beat_to_time_fn
import pathlib

# Basic transcription
lead_sheet, segment_beats, segment_beats_times = sheetsage(
    "path/to/audio.mp3",
    detect_melody=True,
    detect_harmony=True
)

# Save as LilyPond file
lily = lead_sheet.as_lily(title="My Song", artist="Artist Name")
with open("output.ly", "w") as f:
    f.write(lily)

# Generate PDF
pdf_bytes = engrave(lily, out_format="pdf")
with open("output.pdf", "wb") as f:
    f.write(pdf_bytes)

# Save MIDI
beat_to_time_fn = create_beat_to_time_fn(segment_beats, segment_beats_times)
midi_bytes = lead_sheet.as_midi(beat_to_time_fn)
with open("output.midi", "wb") as f:
    f.write(midi_bytes)
```

### Advanced Usage

```python
# Transcription with hints for better accuracy
lead_sheet, segment_beats, segment_beats_times = sheetsage(
    "audio.mp3",
    segment_start_hint=30.0,      # Start at 30 seconds
    segment_end_hint=90.0,        # End at 90 seconds
    beats_per_minute_hint=120,    # Approximate BPM
    beats_per_measure_hint=4,     # 4/4 time signature
    use_jukebox=True,             # Use Jukebox embeddings (GPU required)
    measures_per_chunk=8,         # Process 8 measures at a time
    detect_melody=True,
    detect_harmony=True
)

# Transcribe from URL
lead_sheet, _, _ = sheetsage("https://example.com/song.mp3")

# Adjust sensitivity thresholds
lead_sheet, _, _ = sheetsage(
    "audio.mp3",
    melody_threshold=0.3,   # Lower = more notes detected
    harmony_threshold=0.5   # Lower = more chords detected
)
```

See `examples/basic_transcription.py` for more complete examples.

## Model Downloads

### Automatic Downloads

Most model assets are downloaded automatically on first use to `~/.sheetsage`. You can change this location by setting the `SHEETSAGE_CACHE_DIR` environment variable.

### Manual Download Instructions

If automatic downloads fail (e.g., due to network issues or 403 errors), you can manually download the models:

#### Option 1: Using megatools (Recommended)

1. Install megatools:
   ```bash
   # Linux (Debian/Ubuntu)
   sudo apt-get install megatools
   
   # macOS
   brew install megatools
   ```

2. The models will be automatically downloaded when needed, or you can download manually:
   ```bash
   megadl https://mega.nz/file/j9YyWbAK#LjROFI9qxGq6Om9dx9HaORd5NaiYOcV8ULVUBKB0bcg
   ```

#### Option 2: Manual Download from Browser

1. **Download the archive**:
   - Visit: https://mega.nz/file/j9YyWbAK#LjROFI9qxGq6Om9dx9HaORd5NaiYOcV8ULVUBKB0bcg
   - Download the archive file using your browser

2. **Extract the archive**:
   ```bash
   # Extract to a temporary location first
   unzip downloaded_file.zip  # or tar -xzf downloaded_file.tar.gz
   ```

3. **Place files in the correct location**:
   
   The models should be placed in `~/.sheetsage/` with the following structure:
   ```
   ~/.sheetsage/
   ├── sheetsage/
   │   └── v0.2/
   │       ├── 0920_00_e0830_jukebox53/  # Jukebox Melody models
   │       │   ├── e968ecb8349156b2e9761ae61606454988fa614d.cfg.json
   │       │   ├── step.pkl
   │       │   └── model.pt
   │       └── 0920_01_e0908_jukebox53/  # Jukebox Harmony models
   │           ├── f94f45ed03c8696f187a8bfded0f0d65476b4d48.cfg.json
   │           ├── step.pkl
   │           └── model.pt
   └── jukebox/
       └── models/
           └── 5b/  # Jukebox embedding models (if included)
               ├── vqvae.pth.tar
               └── prior_level_2.pth.tar
   ```

4. **Verify the installation**:
   ```bash
   # Check if files are in the right place
   ls -la ~/.sheetsage/sheetsage/v0.2/0920_00_e0830_jukebox53/
   ls -la ~/.sheetsage/sheetsage/v0.2/0920_01_e0908_jukebox53/
   ```

5. **Alternative: Use local directory**:
   
   You can also place models in a local `./sheetsage` directory in the project root, and the code will automatically detect and use them.

### Jukebox Embedding Model (Optional)

For improved transcription quality, download the Jukebox embedding model:

**Download from**: https://mega.nz/file/j9YyWbAK#LjROFI9qxGq6Om9dx9HaORd5NaiYOcV8ULVUBKB0bcg

After downloading, place the model files in `~/.sheetsage/jukebox/models/5b/` (the exact path will be shown in error messages if the model is missing).

**Note**: Using Jukebox requires a GPU with at least 12GB VRAM and significantly increases processing time.

## Requirements

- Python >= 3.12
- PyTorch
- librosa
- numpy
- scipy
- audioread
- SoundFile
- validators
- pillow
- scikit-learn
- pretty-midi
- tqdm

All dependencies are listed in `pyproject.toml` and `requirements.txt`.

## Output Files

The transcription process generates:

- **`.ly` file**: LilyPond notation (human-readable music notation)
- **`.pdf` file**: Rendered lead sheet (requires LilyPond)
- **`.midi` file**: MIDI file with synchronized timing

## Troubleshooting

### "LilyPond not found" error

Install LilyPond (see Prerequisites above) and ensure it's in your system PATH. You can verify with:
```bash
lilypond --version
```

### "Model not found" errors

- Models are downloaded automatically on first use
- Check that you have internet connectivity
- Verify write permissions to `~/.sheetsage` directory
- For Jukebox model, manually download and place in the correct directory

### Slow transcription

- Jukebox embeddings are slow but improve quality
- For faster transcription, use `use_jukebox=False` (default)
- Processing time scales with audio length

### Beat detection issues

- Provide `beats_per_minute_hint` if tempo detection is off
- Use `segment_start_hint` and `segment_end_hint` to focus on specific segments
- Try `beats_per_measure_hint=4` for 4/4 time or `3` for 3/4 time

### GPU/VRAM errors with Jukebox

- Jukebox requires GPU with >=12GB VRAM
- If you don't have sufficient VRAM, use `use_jukebox=False`
- CPU-only mode is supported but slower

### Audio format issues

Supported formats: MP3, WAV, FLAC, OGG, M4A, AAC, and others supported by librosa.

## Examples

See the `examples/` directory for:
- `basic_transcription.py`: Simple transcription examples
- `notebooks/Inference.ipynb`: Jupyter notebook with interactive examples

## License

See `LICENSE` file for details.

## Citation

If you use SheetSage in your research, please cite the original paper (see the main SheetSage repository).

