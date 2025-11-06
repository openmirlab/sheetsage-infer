#!/usr/bin/env python3
"""
Basic example script for transcribing audio to lead sheets using SheetSage.

This script demonstrates the simplest way to use SheetSage for transcription.
"""

from sheetsage.infer import sheetsage
from sheetsage.utils import engrave
from sheetsage.align import create_beat_to_time_fn
import pathlib

# Example 1: Basic transcription from a local audio file
def transcribe_audio_file(audio_path, output_dir="./output"):
    """
    Transcribe an audio file to a lead sheet.
    
    Args:
        audio_path: Path to the audio file (supports .mp3, .wav, .flac, etc.)
        output_dir: Directory to save the output files
    """
    print(f"Transcribing {audio_path}...")
    
    # Transcribe the audio
    lead_sheet, segment_beats, segment_beats_times = sheetsage(
        audio_path,
        detect_melody=True,
        detect_harmony=True
    )
    
    # Create output directory
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save lead sheet as LilyPond file
    lily = lead_sheet.as_lily()
    with open(output_path / "lead_sheet.ly", "w", encoding="utf-8") as f:
        f.write(lily)
    print(f"Saved LilyPond file to {output_path / 'lead_sheet.ly'}")
    
    # Generate and save PDF
    pdf_bytes = engrave(lily, out_format="pdf", transparent=False, trim=False, hide_footer=False)
    with open(output_path / "lead_sheet.pdf", "wb") as f:
        f.write(pdf_bytes)
    print(f"Saved PDF to {output_path / 'lead_sheet.pdf'}")
    
    # Save MIDI file
    beat_to_time_fn = create_beat_to_time_fn(segment_beats, segment_beats_times)
    midi_bytes = lead_sheet.as_midi(beat_to_time_fn)
    with open(output_path / "lead_sheet.midi", "wb") as f:
        f.write(midi_bytes)
    print(f"Saved MIDI file to {output_path / 'lead_sheet.midi'}")
    
    return lead_sheet, output_path


# Example 2: Transcription with hints for better accuracy
def transcribe_with_hints(audio_path, start_time=None, end_time=None, bpm=None):
    """
    Transcribe with optional hints for better beat detection.
    
    Args:
        audio_path: Path to the audio file
        start_time: Approximate start time in seconds (optional)
        end_time: Approximate end time in seconds (optional)
        bpm: Approximate beats per minute (optional)
    """
    print(f"Transcribing {audio_path} with hints...")
    
    lead_sheet, segment_beats, segment_beats_times = sheetsage(
        audio_path,
        segment_start_hint=start_time,
        segment_end_hint=end_time,
        beats_per_minute_hint=bpm,
        detect_melody=True,
        detect_harmony=True
    )
    
    return lead_sheet, segment_beats, segment_beats_times


# Example 3: Transcription from URL
def transcribe_from_url(audio_url, output_dir="./output"):
    """
    Transcribe audio from a URL.
    
    Args:
        audio_url: URL to the audio file
        output_dir: Directory to save the output files
    """
    print(f"Transcribing audio from {audio_url}...")
    
    lead_sheet, segment_beats, segment_beats_times = sheetsage(
        audio_url,
        detect_melody=True,
        detect_harmony=True
    )
    
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save outputs (same as Example 1)
    lily = lead_sheet.as_lily()
    with open(output_path / "lead_sheet.ly", "w", encoding="utf-8") as f:
        f.write(lily)
    
    pdf_bytes = engrave(lily, out_format="pdf", transparent=False, trim=False, hide_footer=False)
    with open(output_path / "lead_sheet.pdf", "wb") as f:
        f.write(pdf_bytes)
    
    beat_to_time_fn = create_beat_to_time_fn(segment_beats, segment_beats_times)
    midi_bytes = lead_sheet.as_midi(beat_to_time_fn)
    with open(output_path / "lead_sheet.midi", "wb") as f:
        f.write(midi_bytes)
    
    print(f"Transcription complete! Files saved to {output_path}")
    return lead_sheet, output_path


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python basic_transcription.py <audio_file_or_url> [output_dir]")
        print("\nExample:")
        print("  python basic_transcription.py song.mp3")
        print("  python basic_transcription.py song.mp3 ./my_output")
        print("  python basic_transcription.py https://example.com/song.mp3")
        sys.exit(1)
    
    audio_input = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./output"
    
    # Check if it's a URL
    if audio_input.startswith("http://") or audio_input.startswith("https://"):
        transcribe_from_url(audio_input, output_dir)
    else:
        transcribe_audio_file(audio_input, output_dir)

