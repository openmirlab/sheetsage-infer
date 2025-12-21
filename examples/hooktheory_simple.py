#!/usr/bin/env python3
"""
Simple example for downloading and using Hooktheory data with SheetSage.

This script:
1. Downloads Hooktheory test segments JSON
2. Shows sample data structure
3. Downloads a sample MIDI file
4. Demonstrates how to use it with SheetSage (if available)
"""

import json
import pathlib
import tarfile
import tempfile
import urllib.request

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
        print("✓ Downloaded successfully")
        return output_path
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return None


def load_segments(segments_path):
    """Load segments from JSON file."""
    with open(segments_path) as f:
        segments = json.load(f)
    return segments


def extract_sample_midi(tar_path, output_dir):
    """Extract the first MIDI file from tar.gz."""
    print(f"Extracting MIDI files from {tar_path}...")

    with tarfile.open(tar_path, "r:gz") as tar:
        members = [
            m for m in tar.getmembers() if m.name.endswith(".mid") or m.name.endswith(".midi")
        ]

        if not members:
            print("No MIDI files found in archive")
            return None

        # Extract first MIDI file
        first_midi = members[0]
        tar.extract(first_midi, output_dir)
        extracted_path = pathlib.Path(output_dir) / first_midi.name
        print(f"✓ Extracted: {extracted_path}")
        return extracted_path


def main():
    """Main function."""
    print("=" * 70)
    print("SheetSage Hooktheory Data Example")
    print("=" * 70)

    # Create output directory
    output_dir = pathlib.Path("./hooktheory_data")
    output_dir.mkdir(exist_ok=True)

    # Step 1: Download test segments
    print("\n[Step 1] Downloading Hooktheory test segments...")
    segments_path = output_dir / "Hooktheory_Test_Segments.json"

    if not segments_path.exists():
        segments_path = download_file(HOOKTHEORY_URLS["TEST_SEGMENTS"], segments_path)
        if not segments_path:
            print("Failed to download segments. Exiting.")
            return
    else:
        print(f"✓ Using existing file: {segments_path}")

    # Step 2: Load and display segments
    print("\n[Step 2] Loading segments...")
    segments = load_segments(segments_path)
    print(f"✓ Loaded {len(segments)} segments")

    if len(segments) == 0:
        print("No segments found. Exiting.")
        return

    # Display sample segment
    print("\n[Step 3] Sample segment structure:")
    # Handle both list and dict formats
    if isinstance(segments, list):
        sample = segments[0]
    elif isinstance(segments, dict):
        # Get first key-value pair
        first_key = list(segments.keys())[0]
        sample = segments[first_key]
        print(f"  Segment ID: {first_key}")
    else:
        print(f"  Unexpected segments type: {type(segments)}")
        sample = None

    if sample:
        print(f"  Keys: {list(sample.keys())}")
        for key, value in list(sample.items())[:10]:  # Show first 10 items
            if isinstance(value, (str, int, float)):
                val_str = str(value)
                if len(val_str) > 50:
                    val_str = val_str[:50] + "..."
                print(f"  {key}: {val_str}")
            elif isinstance(value, list):
                print(f"  {key}: list with {len(value)} items")
            elif isinstance(value, dict):
                print(f"  {key}: dict with keys {list(value.keys())[:5]}")

    # Step 4: Download MIDI archive
    print("\n[Step 4] Downloading Hooktheory MIDI archive...")
    midi_tar_path = output_dir / "Hooktheory_Test_MIDI.tar.gz"

    if not midi_tar_path.exists():
        midi_tar_path = download_file(HOOKTHEORY_URLS["TEST_MIDI"], midi_tar_path)
        if not midi_tar_path:
            print("Failed to download MIDI archive. Exiting.")
            return
    else:
        print(f"✓ Using existing file: {midi_tar_path}")

    # Step 5: Extract sample MIDI
    print("\n[Step 5] Extracting sample MIDI file...")
    with tempfile.TemporaryDirectory() as tmpdir:
        extracted_midi = extract_sample_midi(midi_tar_path, tmpdir)

        if extracted_midi:
            # Copy to output directory for easy access
            final_midi = output_dir / extracted_midi.name
            import shutil

            shutil.copy(extracted_midi, final_midi)
            print(f"✓ Copied to: {final_midi}")

            # Step 6: Try to use with SheetSage
            print("\n[Step 6] Attempting to use with SheetSage...")
            print("\nTo transcribe this MIDI file:")
            print("  1. Convert MIDI to audio (e.g., using fluidsynth)")
            print(f"     fluidsynth -F audio.wav /usr/share/sounds/sf2/FluidR3_GM.sf2 {final_midi}")
            print("  2. Run SheetSage transcription:")
            print("     python -m sheetsage.infer audio.wav -o ./output")
            print("\nOr use the basic_transcription.py example:")
            print("     python examples/basic_transcription.py audio.wav")

            # Try to import and use sheetsage if available
            try:
                from sheetsage.align import create_beat_to_time_fn
                from sheetsage.infer import sheetsage
                from sheetsage.utils import engrave

                print("\n✓ SheetSage is available!")
                print("  Note: MIDI files need to be converted to audio first.")
                print("  SheetSage requires audio input (WAV, MP3, etc.), not MIDI.")

            except ImportError:
                print("\n⚠ SheetSage not fully installed.")
                print("  Install dependencies: pip install -e .")
                print("  (Requires Python 3.10+)")

    print("\n" + "=" * 70)
    print("✓ Hooktheory data downloaded successfully!")
    print(f"  Data location: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
