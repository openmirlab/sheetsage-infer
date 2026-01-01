# Testing Jukebox Transcription with CUDA_VISIBLE_DEVICES=2

This document describes how to test the SheetSage transcription pipeline with Jukebox features on a specific GPU device.

## Test Files Created

1. **`test_jukebox_cuda.py`** - Main test script for running transcription with Jukebox on GPU 2
2. **`test_jukebox_cuda_diagnostic.py`** - Environment diagnostic script
3. **`jukebox_transcription.py`** - Full-featured example with CLI

## Prerequisites

- Python 3.10+
- PyTorch with CUDA support
- SheetSage installed: `pip install -e .`
- GPU with >=12GB VRAM (recommended for Jukebox)
- Audio file for testing

## Test Audio File

A test audio file has been created from Hooktheory MIDI:
- **Location**: `hooktheory_data/test_audio.wav` (5.83 MB)
- **Source**: Converted from `hooktheory_data/-WeglAPLxrY.mid` using fluidsynth

## Running the Test

### Option 1: Using the Test Script

```bash
# From repository root
cd sheetsage-infer

# Run transcription with Jukebox on GPU 2
CUDA_VISIBLE_DEVICES=2 python examples/test_jukebox_cuda.py hooktheory_data/test_audio.wav
```

### Option 2: Using the Full Example Script

```bash
# Basic usage
CUDA_VISIBLE_DEVICES=2 python examples/jukebox_transcription.py hooktheory_data/test_audio.wav

# With output directory
CUDA_VISIBLE_DEVICES=2 python examples/jukebox_transcription.py hooktheory_data/test_audio.wav -o ./jukebox_output

# With timing hints
CUDA_VISIBLE_DEVICES=2 python examples/jukebox_transcription.py hooktheory_data/test_audio.wav --start 0 --end 30 --bpm 120
```

### Option 3: Direct Python API

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from sheetsage.infer import sheetsage
from sheetsage.align import create_beat_to_time_fn

# Run transcription
lead_sheet, segment_beats, segment_beats_times = sheetsage(
    'hooktheory_data/test_audio.wav',
    use_jukebox=True,  # Enable Jukebox embeddings
    detect_melody=True,
    detect_harmony=True
)

# Save outputs
lily = lead_sheet.as_lily()
with open('output_jukebox.ly', 'w') as f:
    f.write(lily)

beat_to_time_fn = create_beat_to_time_fn(segment_beats, segment_beats_times)
midi_bytes = lead_sheet.as_midi(beat_to_time_fn)
with open('output_jukebox.midi', 'wb') as f:
    f.write(midi_bytes)
```

## Environment Diagnostic

Before running transcription, check your environment:

```bash
CUDA_VISIBLE_DEVICES=2 python examples/test_jukebox_cuda_diagnostic.py hooktheory_data/test_audio.wav
```

This will check:
- CUDA_VISIBLE_DEVICES setting
- PyTorch installation and CUDA availability
- GPU device information and memory
- SheetSage importability
- Audio file existence

## Expected Output

When successful, the test will:
1. Load SheetSage and Jukebox modules
2. Extract Jukebox embeddings from audio (slower, but higher quality)
3. Run melody and harmony transcription
4. Save outputs to `jukebox_test_output/`:
   - `test_jukebox.ly` - LilyPond notation
   - `test_jukebox.midi` - MIDI file
   - `test_jukebox.pdf` - PDF (if LilyPond is installed)

## Performance Notes

- **Jukebox features**: ~30-60 seconds per minute of audio (requires GPU)
- **Handcrafted features**: ~1-5 seconds per minute of audio (CPU)
- Jukebox provides better transcription quality but is significantly slower

## Troubleshooting

### CUDA Not Available
```
Error: CUDA not available
```
- Check GPU drivers: `nvidia-smi`
- Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Ensure CUDA_VISIBLE_DEVICES is set correctly

### Insufficient VRAM
```
Error: Out of memory
```
- Jukebox requires >=12GB VRAM
- Try shorter audio segments
- Use `segment_start_hint` and `segment_end_hint` to process smaller chunks

### Module Not Found
```
Error: No module named 'sheetsage'
```
- Install SheetSage: `pip install -e .`
- Ensure you're in the correct Python environment
- Check PYTHONPATH if needed

### Jukebox Import Error
```
Error: Jukebox module not installed
```
- Jukebox modules are vendored in this repository
- Check that `sheetsage/representations/jukebox_modules/` exists
- Verify all dependencies are installed

## Verification

After running, verify the outputs:

```bash
# Check output files
ls -lh jukebox_test_output/

# View LilyPond file
cat jukebox_test_output/test_jukebox.ly | head -50

# Check MIDI file (if you have a MIDI viewer)
# The MIDI should contain melody, harmony, and click track
```

## Comparison with Handcrafted Features

To compare quality, run both methods:

```bash
# Handcrafted (fast, CPU)
python examples/basic_transcription.py hooktheory_data/test_audio.wav -o ./output_handcrafted

# Jukebox (slow, GPU, better quality)
CUDA_VISIBLE_DEVICES=2 python examples/jukebox_transcription.py hooktheory_data/test_audio.wav -o ./output_jukebox

# Compare the .ly files to see quality differences
diff output_handcrafted/lead_sheet.ly output_jukebox/lead_sheet_jukebox.ly
```

## Notes

- The `CUDA_VISIBLE_DEVICES=2` environment variable restricts PyTorch to use only GPU device 2
- This is useful when you have multiple GPUs and want to use a specific one
- The setting must be set before importing PyTorch
- Jukebox feature extraction is the slowest part of the pipeline
- Processing time scales roughly linearly with audio duration

