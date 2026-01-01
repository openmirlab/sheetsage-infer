# How Hooktheory Data Was Obtained

This document shows the code used to download Hooktheory data and create `test_audio.wav`.

---

## Overview

The Hooktheory data was downloaded using two methods:

1. **Direct URL download** (in `hooktheory_simple.py`)
2. **SheetSage asset system** (in `hooktheory_example.py`)

The `test_audio.wav` file was created by:
1. Downloading the MIDI archive
2. Extracting a MIDI file
3. Converting MIDI to WAV using `fluidsynth`

---

## Method 1: Direct URL Download

**File**: `examples/hooktheory_simple.py`

### Step 1: Download Test Segments JSON

```python
# Hooktheory data URLs (from assets/hooktheory.json)
HOOKTHEORY_URLS = {
    "TEST_SEGMENTS": "https://github.com/chrisdonahue/sheetsage-data/raw/refs/heads/main/hooktheory/Hooktheory_Test_Segments.json",
    "TEST_MIDI": "https://github.com/chrisdonahue/sheetsage-data/raw/refs/heads/main/hooktheory/Hooktheory_Test_MIDI.tar.gz",
}

def download_file(url, output_path):
    """Download a file from URL."""
    print(f"Downloading {url}...")
    print(f"  -> {output_path}")
    
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"✓ Downloaded successfully")
        return output_path
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return None

# Usage:
segments_path = download_file(
    HOOKTHEORY_URLS["TEST_SEGMENTS"],
    "./hooktheory_data/Hooktheory_Test_Segments.json"
)
```

### Step 2: Download MIDI Archive

```python
midi_tar_path = download_file(
    HOOKTHEORY_URLS["TEST_MIDI"],
    "./hooktheory_data/Hooktheory_Test_MIDI.tar.gz"
)
```

### Step 3: Extract MIDI File

```python
def extract_sample_midi(tar_path, output_dir):
    """Extract the first MIDI file from tar.gz."""
    print(f"Extracting MIDI files from {tar_path}...")
    
    with tarfile.open(tar_path, 'r:gz') as tar:
        members = [m for m in tar.getmembers() if m.name.endswith('.mid') or m.name.endswith('.midi')]
        
        if not members:
            print("No MIDI files found in archive")
            return None
        
        # Extract first MIDI file
        first_midi = members[0]
        tar.extract(first_midi, output_dir)
        extracted_path = pathlib.Path(output_dir) / first_midi.name
        print(f"✓ Extracted: {extracted_path}")
        return extracted_path

# Usage:
extracted_midi = extract_sample_midi(midi_tar_path, "./hooktheory_data")
# Result: hooktheory_data/-WeglAPLxrY.mid
```

---

## Method 2: Using SheetSage Asset System

**File**: `examples/hooktheory_example.py`

### Step 1: Download Using `retrieve_asset`

```python
from sheetsage.assets import retrieve_asset

def download_hooktheory_segments(segment_type="TEST"):
    """Download Hooktheory segments JSON file."""
    tag = f"HOOKTHEORY_{segment_type}_SEGMENTS"
    print(f"Downloading {tag}...")
    
    try:
        segments_path = retrieve_asset(tag, log=True)
        print(f"✓ Downloaded to: {segments_path}")
        return segments_path
    except Exception as e:
        print(f"Error downloading {tag}: {e}")
        return None

def download_hooktheory_midi(segment_type="TEST"):
    """Download Hooktheory MIDI tar.gz file."""
    tag = f"HOOKTHEORY_{segment_type}_MIDI"
    print(f"Downloading {tag}...")
    
    try:
        midi_path = retrieve_asset(tag, log=True)
        print(f"✓ Downloaded to: {midi_path}")
        return midi_path
    except Exception as e:
        print(f"Error downloading {tag}: {e}")
        return None

# Usage:
segments_path = download_hooktheory_segments("TEST")
midi_tar_path = download_hooktheory_midi("TEST")
```

The `retrieve_asset` function:
- Reads URLs and checksums from `sheetsage/assets/hooktheory.json`
- Downloads files to cache directory
- Verifies checksums
- Returns path to downloaded file

---

## Creating test_audio.wav from MIDI

**File**: `examples/hooktheory_example.py` (function `midi_to_audio`)

### Step 1: Extract MIDI from Archive

```python
def extract_midi_from_tar(tar_path, midi_filename, output_dir):
    """Extract a specific MIDI file from tar.gz."""
    print(f"Extracting {midi_filename} from {tar_path}...")
    
    with tarfile.open(tar_path, 'r:gz') as tar:
        try:
            tar.extract(midi_filename, output_dir)
            extracted_path = pathlib.Path(output_dir) / midi_filename
            print(f"✓ Extracted to: {extracted_path}")
            return extracted_path
        except KeyError:
            print(f"✗ MIDI file {midi_filename} not found in archive")
            return None

# Usage:
extracted_midi = extract_midi_from_tar(
    "./hooktheory_data/Hooktheory_Test_MIDI.tar.gz",
    "-WeglAPLxrY.mid",
    "./hooktheory_data"
)
```

### Step 2: Convert MIDI to WAV using fluidsynth

```python
def midi_to_audio(midi_path, output_audio_path, sample_rate=44100):
    """
    Convert MIDI to audio using fluidsynth or similar tool.
    
    Requires: fluidsynth and a soundfont file
    """
    print(f"Converting MIDI to audio: {midi_path} -> {output_audio_path}")
    
    # Try fluidsynth first
    if shutil.which("fluidsynth"):
        # Use a default soundfont (user may need to specify their own)
        soundfont = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
        if not pathlib.Path(soundfont).exists():
            soundfont = "/usr/share/sounds/sf2/default.sf2"
            if not pathlib.Path(soundfont).exists():
                print("Warning: No soundfont found. Trying alternative methods...")
                return None
        
        cmd = [
            "fluidsynth",
            "-F", str(output_audio_path),  # Output file
            "-r", str(sample_rate),         # Sample rate (44100 Hz)
            soundfont,                      # Soundfont file
            str(midi_path)                  # Input MIDI file
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✓ Converted to: {output_audio_path}")
            return output_audio_path
        except subprocess.CalledProcessError as e:
            print(f"✗ Fluidsynth conversion failed: {e.stderr.decode()}")
            return None
    else:
        print("✗ fluidsynth not found. Install with: sudo apt-get install fluidsynth")
        return None

# Usage:
audio_path = midi_to_audio(
    "./hooktheory_data/-WeglAPLxrY.mid",
    "./hooktheory_data/test_audio.wav",
    sample_rate=44100
)
# Result: hooktheory_data/test_audio.wav (5.83 MB, ~24 seconds)
```

### Command Line Equivalent

The same conversion can be done directly with fluidsynth:

```bash
fluidsynth -F hooktheory_data/test_audio.wav \
           -r 44100 \
           /usr/share/sounds/sf2/FluidR3_GM.sf2 \
           hooktheory_data/-WeglAPLxrY.mid
```

---

## Complete Workflow Summary

Here's the complete process that was used:

```python
# 1. Download segments JSON
segments_path = download_file(
    "https://github.com/chrisdonahue/sheetsage-data/raw/refs/heads/main/hooktheory/Hooktheory_Test_Segments.json",
    "./hooktheory_data/Hooktheory_Test_Segments.json"
)

# 2. Download MIDI archive
midi_tar_path = download_file(
    "https://github.com/chrisdonahue/sheetsage-data/raw/refs/heads/main/hooktheory/Hooktheory_Test_MIDI.tar.gz",
    "./hooktheory_data/Hooktheory_Test_MIDI.tar.gz"
)

# 3. Extract MIDI file
with tarfile.open(midi_tar_path, 'r:gz') as tar:
    members = [m for m in tar.getmembers() if m.name.endswith('.mid')]
    first_midi = members[0]
    tar.extract(first_midi, "./hooktheory_data")
    # Result: hooktheory_data/-WeglAPLxrY.mid

# 4. Convert MIDI to WAV
subprocess.run([
    "fluidsynth",
    "-F", "./hooktheory_data/test_audio.wav",
    "-r", "44100",
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "./hooktheory_data/-WeglAPLxrY.mid"
], check=True)
# Result: hooktheory_data/test_audio.wav (5.83 MB, ~24 seconds)
```

---

## Files Created

After running the scripts, the following files were created:

```
hooktheory_data/
├── Hooktheory_Test_Segments.json      (196K) - 1480 test segments
├── Hooktheory_Test_MIDI.tar.gz        (290K) - MIDI files archive
├── -WeglAPLxrY.mid                    (166B) - Extracted MIDI file
└── test_audio.wav                     (5.9M) - Converted audio file
```

---

## Asset Configuration

The URLs and checksums are defined in `sheetsage/assets/hooktheory.json`:

```json
{
  "HOOKTHEORY_TEST_SEGMENTS": {
    "path": "hooktheory/Hooktheory_Test_Segments.json",
    "url": "https://github.com/chrisdonahue/sheetsage-data/raw/refs/heads/main/hooktheory/Hooktheory_Test_Segments.json",
    "checksum": "72be80045d4d28842352383e605e8712d50b3437a07b15faa541ee9d17283d5a"
  },
  "HOOKTHEORY_TEST_MIDI": {
    "path": "hooktheory/Hooktheory_Test_MIDI.tar.gz",
    "url": "https://github.com/chrisdonahue/sheetsage-data/raw/refs/heads/main/hooktheory/Hooktheory_Test_MIDI.tar.gz",
    "checksum": "3baebe9d4e19a5006d0f24bc7f0c92a4f66039ab376be86bf7b37a136d4fb6c8"
  }
}
```

---

## Quick Reference

To recreate the test audio file:

```bash
# 1. Download segments
wget -O hooktheory_data/Hooktheory_Test_Segments.json \
  https://github.com/chrisdonahue/sheetsage-data/raw/refs/heads/main/hooktheory/Hooktheory_Test_Segments.json

# 2. Download MIDI archive
wget -O hooktheory_data/Hooktheory_Test_MIDI.tar.gz \
  https://github.com/chrisdonahue/sheetsage-data/raw/refs/heads/main/hooktheory/Hooktheory_Test_MIDI.tar.gz

# 3. Extract MIDI file
tar -xzf hooktheory_data/Hooktheory_Test_MIDI.tar.gz -C hooktheory_data/ --wildcards "*.mid" | head -1

# 4. Convert to WAV
fluidsynth -F hooktheory_data/test_audio.wav \
           -r 44100 \
           /usr/share/sounds/sf2/FluidR3_GM.sf2 \
           hooktheory_data/-WeglAPLxrY.mid
```

Or use the Python scripts:

```bash
# Method 1: Simple script
python examples/hooktheory_simple.py

# Method 2: Full example with SheetSage asset system
python examples/hooktheory_example.py
```

