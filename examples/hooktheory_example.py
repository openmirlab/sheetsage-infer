#!/usr/bin/env python3
"""
Example script for transcribing audio using SheetSage with Hooktheory data.

This script demonstrates:
1. Downloading Hooktheory segments
2. Extracting a sample entry
3. Converting MIDI to audio (if needed)
4. Running transcription
"""

import json
import pathlib
import shutil
import tarfile
import tempfile
import subprocess
import sys

# Try to import sheetsage (may fail if dependencies not installed)
SHEETSAGE_AVAILABLE = False
retrieve_asset = None
sheetsage = None
engrave = None
create_beat_to_time_fn = None

try:
    from sheetsage.infer import sheetsage
    from sheetsage.utils import engrave
    from sheetsage.align import create_beat_to_time_fn
    from sheetsage.assets import retrieve_asset
    SHEETSAGE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import sheetsage: {e}")
    print("This script will download Hooktheory data but cannot run transcription.")
    # Try to import just assets module
    try:
        import sys
        import pathlib
        # Add parent directory to path
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from sheetsage.assets import retrieve_asset
        print("✓ Assets module loaded (transcription will be skipped)")
    except ImportError as e2:
        print(f"Could not load assets module: {e2}")
        retrieve_asset = None


def download_hooktheory_segments(segment_type="TEST"):
    """
    Download Hooktheory segments JSON file.
    
    Args:
        segment_type: "TRAIN", "VALID", or "TEST"
    
    Returns:
        Path to downloaded segments file
    """
    if retrieve_asset is None:
        print("Error: retrieve_asset not available. Cannot download assets.")
        return None
    
    tag = f"HOOKTHEORY_{segment_type}_SEGMENTS"
    print(f"Downloading {tag}...")
    
    try:
        segments_path = retrieve_asset(tag, log=True)
        print(f"✓ Downloaded to: {segments_path}")
        return segments_path
    except Exception as e:
        print(f"Error downloading {tag}: {e}")
        return None


def load_segments(segments_path):
    """Load and return segments from JSON file."""
    with open(segments_path, 'r') as f:
        segments = json.load(f)
    print(f"✓ Loaded {len(segments)} segments")
    return segments


def download_hooktheory_midi(segment_type="TEST"):
    """
    Download Hooktheory MIDI tar.gz file.
    
    Args:
        segment_type: "TRAIN", "VALID", or "TEST"
    
    Returns:
        Path to downloaded MIDI tar.gz file
    """
    if retrieve_asset is None:
        print("Error: retrieve_asset not available. Cannot download assets.")
        return None
    
    tag = f"HOOKTHEORY_{segment_type}_MIDI"
    print(f"Downloading {tag}...")
    
    try:
        midi_path = retrieve_asset(tag, log=True)
        print(f"✓ Downloaded to: {midi_path}")
        return midi_path
    except Exception as e:
        print(f"Error downloading {tag}: {e}")
        return None


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
            # List available files
            print("Available files in archive:")
            for member in tar.getmembers()[:10]:  # Show first 10
                print(f"  - {member.name}")
            return None


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
            "-F", str(output_audio_path),
            "-r", str(sample_rate),
            soundfont,
            str(midi_path)
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
        print("  Or use another MIDI to audio converter")
        return None


def transcribe_audio(audio_path, output_dir="./output"):
    """Transcribe audio file using SheetSage."""
    if not SHEETSAGE_AVAILABLE:
        print("✗ Cannot transcribe: sheetsage not available")
        return None
    
    print(f"\nTranscribing {audio_path}...")
    
    try:
        # Transcribe the audio
        lead_sheet, segment_beats, segment_beats_times = sheetsage(
            str(audio_path),
            detect_melody=True,
            detect_harmony=True
        )
        
        # Create output directory
        output_path = pathlib.Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save lead sheet as LilyPond file
        lily = lead_sheet.as_lily()
        lily_path = output_path / "lead_sheet.ly"
        with open(lily_path, "w", encoding="utf-8") as f:
            f.write(lily)
        print(f"✓ Saved LilyPond file to {lily_path}")
        
        # Save MIDI file
        beat_to_time_fn = create_beat_to_time_fn(segment_beats, segment_beats_times)
        midi_bytes = lead_sheet.as_midi(beat_to_time_fn)
        midi_path = output_path / "lead_sheet.midi"
        with open(midi_path, "wb") as f:
            f.write(midi_bytes)
        print(f"✓ Saved MIDI file to {midi_path}")
        
        # Try to generate PDF (requires LilyPond)
        try:
            pdf_bytes = engrave(lily, out_format="pdf", transparent=False, trim=False, hide_footer=False)
            pdf_path = output_path / "lead_sheet.pdf"
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)
            print(f"✓ Saved PDF to {pdf_path}")
        except Exception as e:
            print(f"⚠ Could not generate PDF: {e}")
        
        return lead_sheet, output_path
        
    except Exception as e:
        print(f"✗ Transcription failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function to demonstrate Hooktheory data usage."""
    print("=" * 60)
    print("SheetSage Hooktheory Example")
    print("=" * 60)
    
    # Step 1: Download Hooktheory test segments
    print("\n[Step 1] Downloading Hooktheory test segments...")
    segments_path = download_hooktheory_segments("TEST")
    if not segments_path:
        print("Failed to download segments. Exiting.")
        return
    
    # Step 2: Load and display sample segments
    print("\n[Step 2] Loading segments...")
    segments = load_segments(segments_path)
    
    if len(segments) == 0:
        print("No segments found. Exiting.")
        return
    
    # Display first few segments
    print(f"\nFirst 3 segments:")
    for i, segment in enumerate(segments[:3]):
        print(f"\nSegment {i+1}:")
        print(f"  Keys: {list(segment.keys())}")
        if 'artist' in segment:
            print(f"  Artist: {segment.get('artist', 'N/A')}")
        if 'title' in segment:
            print(f"  Title: {segment.get('title', 'N/A')}")
        if 'midi' in segment:
            print(f"  MIDI: {segment.get('midi', 'N/A')}")
    
    # Step 3: Download MIDI archive
    print("\n[Step 3] Downloading Hooktheory MIDI archive...")
    midi_tar_path = download_hooktheory_midi("TEST")
    if not midi_tar_path:
        print("Failed to download MIDI archive. Exiting.")
        return
    
    # Step 4: Extract a sample MIDI file
    print("\n[Step 4] Extracting sample MIDI file...")
    sample_segment = segments[0]
    midi_filename = sample_segment.get('midi', None)
    
    if not midi_filename:
        print("No MIDI filename in segment. Trying to find one...")
        # Look for any MIDI file in the archive
        with tarfile.open(midi_tar_path, 'r:gz') as tar:
            members = tar.getmembers()
            if members:
                midi_filename = members[0].name
                print(f"Using first file in archive: {midi_filename}")
            else:
                print("Archive is empty. Exiting.")
                return
    
    with tempfile.TemporaryDirectory() as tmpdir:
        extracted_midi = extract_midi_from_tar(midi_tar_path, midi_filename, tmpdir)
        
        if not extracted_midi:
            print("Failed to extract MIDI. Exiting.")
            return
        
        # Step 5: Convert MIDI to audio (if needed for transcription)
        print("\n[Step 5] Converting MIDI to audio...")
        audio_path = pathlib.Path(tmpdir) / f"{pathlib.Path(midi_filename).stem}.wav"
        converted_audio = midi_to_audio(extracted_midi, audio_path)
        
        if converted_audio and SHEETSAGE_AVAILABLE:
            # Step 6: Transcribe audio
            print("\n[Step 6] Transcribing audio...")
            result = transcribe_audio(converted_audio, output_dir="./hooktheory_output")
            
            if result:
                print("\n" + "=" * 60)
                print("✓ Transcription complete!")
                print("=" * 60)
                print(f"\nOutput files saved to: ./hooktheory_output/")
        elif not converted_audio:
            print("\n⚠ Could not convert MIDI to audio.")
            print("  To complete the example:")
            print("  1. Install fluidsynth: sudo apt-get install fluidsynth")
            print("  2. Or use another MIDI to audio converter")
            print(f"  3. Convert {extracted_midi} to audio manually")
            print("  4. Run transcription on the audio file")
        else:
            print("\n⚠ SheetSage not available. Data downloaded successfully.")
            print(f"  MIDI file: {extracted_midi}")
            print("  Install SheetSage dependencies to run transcription.")


if __name__ == "__main__":
    main()

