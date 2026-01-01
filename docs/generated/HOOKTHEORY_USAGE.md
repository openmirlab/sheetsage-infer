# Using Hooktheory Data with SheetSage

This guide demonstrates how to use Hooktheory data with SheetSage for music transcription.

## Setup Summary

Based on the README.md setup instructions:

1. **Installation**: The project requires Python 3.10+ (currently detected: Python 3.9.19)
2. **Dependencies**: Install with `pip install -e .` (requires Python 3.10+)
3. **Optional**: Install LilyPond for PDF generation

## Hooktheory Data Structure

The Hooktheory dataset contains:
- **1480 test segments** with audio timing information
- **MIDI files** in tar.gz archives
- Each segment includes:
  - `audio_tag`: Reference to audio source (e.g., YouTube video ID)
  - `segment_start`: Start time in seconds
  - `segment_end`: End time in seconds

## Example Scripts Created

### 1. `hooktheory_simple.py`
A simple script that:
- Downloads Hooktheory test segments JSON
- Downloads Hooktheory MIDI archive
- Extracts sample MIDI files
- Shows data structure

**Usage:**
```bash
python examples/hooktheory_simple.py
```

**Output:**
- Downloads data to `./hooktheory_data/`
- Extracts sample MIDI file
- Provides instructions for transcription

### 2. `hooktheory_example.py`
A more complete example that:
- Downloads Hooktheory data using SheetSage's asset system
- Converts MIDI to audio
- Runs transcription (if SheetSage is fully installed)

## Current Status

✅ **Completed:**
- Hooktheory data downloaded successfully
- Sample MIDI file extracted: `hooktheory_data/-WeglAPLxrY.mid`
- Data structure documented

⚠️ **Limitations:**
- Python 3.9 detected (requires 3.10+)
- SheetSage dependencies not fully installed
- MIDI needs conversion to audio for transcription

## Next Steps to Run Transcription

### Option 1: Convert MIDI to Audio, Then Transcribe

```bash
# 1. Convert MIDI to audio (requires fluidsynth)
fluidsynth -F audio.wav /usr/share/sounds/sf2/FluidR3_GM.sf2 hooktheory_data/-WeglAPLxrY.mid

# 2. Run transcription
python examples/basic_transcription.py audio.wav ./output
```

### Option 2: Use YouTube Audio (if available)

The segments include `audio_tag` which may reference YouTube videos. You could:
1. Extract the YouTube video ID from `audio_tag`
2. Download audio from YouTube
3. Use the `segment_start` and `segment_end` to extract the relevant portion
4. Run SheetSage transcription

### Option 3: Use Hooktheory Segments with Audio URLs

If you have access to the original audio files:
```python
from sheetsage.infer import sheetsage

# Transcribe with segment timing hints
lead_sheet, beats, beat_times = sheetsage(
    'audio_file.mp3',
    segment_start_hint=78.16,  # From segment_start
    segment_end_hint=98.57,    # From segment_end
    beats_per_minute_hint=120  # Optional BPM hint
)
```

## Data Files Downloaded

- `hooktheory_data/Hooktheory_Test_Segments.json` - 1480 test segments
- `hooktheory_data/Hooktheory_Test_MIDI.tar.gz` - MIDI files archive
- `hooktheory_data/-WeglAPLxrY.mid` - Sample MIDI file

## Example Segment Structure

```json
{
  "pyvgPG-dmYq": {
    "audio_tag": "YOUTUBE_OrTyD7rjBpw",
    "segment_start": 78.16552734375,
    "segment_end": 98.575439453125
  }
}
```

## Full Workflow Example

Once Python 3.10+ and dependencies are installed:

```python
from sheetsage.infer import sheetsage
from sheetsage.utils import engrave
from sheetsage.align import create_beat_to_time_fn
import json

# Load Hooktheory segments
with open('hooktheory_data/Hooktheory_Test_Segments.json', 'r') as f:
    segments = json.load(f)

# Get first segment
segment_id = list(segments.keys())[0]
segment = segments[segment_id]

# Transcribe audio (assuming you have the audio file)
lead_sheet, segment_beats, segment_beats_times = sheetsage(
    'audio_file.mp3',
    segment_start_hint=segment['segment_start'],
    segment_end_hint=segment['segment_end'],
    detect_melody=True,
    detect_harmony=True
)

# Export results
lily = lead_sheet.as_lily()
with open('output/lead_sheet.ly', 'w') as f:
    f.write(lily)

beat_to_time_fn = create_beat_to_time_fn(segment_beats, segment_beats_times)
midi_bytes = lead_sheet.as_midi(beat_to_time_fn)
with open('output/lead_sheet.midi', 'wb') as f:
    f.write(midi_bytes)
```

## Notes

- Hooktheory data is used for evaluation/testing of SheetSage models
- The segments provide ground truth timing for audio segments
- MIDI files can be used for comparison with transcription results
- For production use, you would typically use your own audio files

