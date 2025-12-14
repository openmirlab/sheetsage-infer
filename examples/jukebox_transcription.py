#!/usr/bin/env python3
"""
Example script for transcribing audio using SheetSage with Jukebox embeddings.

This script demonstrates how to use Jukebox features for higher-quality transcription.
Jukebox features require:
- GPU with >=12GB VRAM (recommended)
- Jukebox modules (vendored in this repository)
- Slower processing time but better transcription quality

Note: Jukebox features are slower but provide better transcription quality
compared to handcrafted mel-spectrogram features.
"""

from sheetsage.infer import sheetsage
from sheetsage.utils import engrave
from sheetsage.align import create_beat_to_time_fn
import pathlib
import sys


def transcribe_with_jukebox(audio_path, output_dir="./output", **kwargs):
    """
    Transcribe an audio file using Jukebox embeddings for higher quality.
    
    Args:
        audio_path: Path to the audio file (supports .mp3, .wav, .flac, etc.)
        output_dir: Directory to save the output files
        **kwargs: Additional arguments to pass to sheetsage() function
    
    Returns:
        Tuple of (lead_sheet, segment_beats, segment_beats_times, output_path)
    """
    print("=" * 70)
    print("SheetSage Transcription with Jukebox Features")
    print("=" * 70)
    print(f"\nTranscribing: {audio_path}")
    print("Using: Jukebox embeddings (higher quality, requires GPU)")
    print("\nNote: This may take several minutes depending on audio length...")
    print("      Jukebox feature extraction is slower but provides better results.\n")
    
    try:
        # Transcribe the audio with Jukebox features
        lead_sheet, segment_beats, segment_beats_times = sheetsage(
            audio_path,
            use_jukebox=True,  # Enable Jukebox embeddings
            detect_melody=True,
            detect_harmony=True,
            **kwargs  # Pass through any additional arguments
        )
        
        print("✓ Transcription complete!")
        
        # Create output directory
        output_path = pathlib.Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save lead sheet as LilyPond file
        print("\n[1/3] Saving LilyPond notation...")
        lily = lead_sheet.as_lily()
        lily_path = output_path / "lead_sheet_jukebox.ly"
        with open(lily_path, "w", encoding="utf-8") as f:
            f.write(lily)
        print(f"✓ Saved to: {lily_path}")
        
        # Save MIDI file
        print("\n[2/3] Saving MIDI file...")
        beat_to_time_fn = create_beat_to_time_fn(segment_beats, segment_beats_times)
        midi_bytes = lead_sheet.as_midi(beat_to_time_fn)
        midi_path = output_path / "lead_sheet_jukebox.midi"
        with open(midi_path, "wb") as f:
            f.write(midi_bytes)
        print(f"✓ Saved to: {midi_path}")
        
        # Generate and save PDF (requires LilyPond)
        print("\n[3/3] Generating PDF...")
        try:
            pdf_bytes = engrave(
                lily, 
                out_format="pdf", 
                transparent=False, 
                trim=False, 
                hide_footer=False
            )
            pdf_path = output_path / "lead_sheet_jukebox.pdf"
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)
            print(f"✓ Saved to: {pdf_path}")
        except Exception as e:
            print(f"⚠ Could not generate PDF: {e}")
            print("  (LilyPond may not be installed or audio may be too short)")
        
        print("\n" + "=" * 70)
        print("✓ All outputs saved successfully!")
        print(f"  Output directory: {output_path}")
        print("=" * 70)
        
        return lead_sheet, segment_beats, segment_beats_times, output_path
        
    except Exception as e:
        print(f"\n✗ Transcription failed: {e}")
        print("\nCommon issues:")
        print("  - GPU not available or insufficient VRAM (<12GB)")
        print("  - CUDA not properly configured")
        print("  - Audio file not found or invalid format")
        print("  - Jukebox modules not properly installed")
        print("\nTry using handcrafted features instead:")
        print("  python examples/basic_transcription.py <audio_file>")
        import traceback
        traceback.print_exc()
        return None


def transcribe_with_hints(audio_path, start_time=None, end_time=None, bpm=None, output_dir="./output"):
    """
    Transcribe with Jukebox features and optional hints for better accuracy.
    
    Args:
        audio_path: Path to the audio file
        start_time: Approximate start time in seconds (optional)
        end_time: Approximate end time in seconds (optional)
        bpm: Approximate beats per minute (optional)
        output_dir: Directory to save the output files
    """
    print(f"\nTranscribing {audio_path} with Jukebox features and hints...")
    if start_time is not None:
        print(f"  Start time: {start_time}s")
    if end_time is not None:
        print(f"  End time: {end_time}s")
    if bpm is not None:
        print(f"  BPM hint: {bpm}")
    
    return transcribe_with_jukebox(
        audio_path,
        output_dir=output_dir,
        segment_start_hint=start_time,
        segment_end_hint=end_time,
        beats_per_minute_hint=bpm
    )


def compare_handcrafted_vs_jukebox(audio_path, output_dir="./output"):
    """
    Compare transcription results using both handcrafted and Jukebox features.
    
    This function runs transcription twice - once with handcrafted features
    and once with Jukebox features - to demonstrate the quality difference.
    
    Args:
        audio_path: Path to the audio file
        output_dir: Directory to save the output files
    """
    print("=" * 70)
    print("Comparing Handcrafted vs Jukebox Features")
    print("=" * 70)
    
    from sheetsage.infer import sheetsage
    
    # Run with handcrafted features (fast, CPU)
    print("\n[1/2] Running with handcrafted features (CPU, fast)...")
    try:
        lead_sheet_hc, beats_hc, times_hc = sheetsage(
            audio_path,
            use_jukebox=False,  # Handcrafted features
            detect_melody=True,
            detect_harmony=True
        )
        print("✓ Handcrafted transcription complete")
        
        output_path = pathlib.Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        lily_hc = lead_sheet_hc.as_lily()
        with open(output_path / "lead_sheet_handcrafted.ly", "w", encoding="utf-8") as f:
            f.write(lily_hc)
        print(f"  Saved: {output_path / 'lead_sheet_handcrafted.ly'}")
        
    except Exception as e:
        print(f"✗ Handcrafted transcription failed: {e}")
        lead_sheet_hc = None
    
    # Run with Jukebox features (slow, GPU)
    print("\n[2/2] Running with Jukebox features (GPU, slower but better quality)...")
    try:
        lead_sheet_jb, beats_jb, times_jb = sheetsage(
            audio_path,
            use_jukebox=True,  # Jukebox features
            detect_melody=True,
            detect_harmony=True
        )
        print("✓ Jukebox transcription complete")
        
        lily_jb = lead_sheet_jb.as_lily()
        with open(output_path / "lead_sheet_jukebox.ly", "w", encoding="utf-8") as f:
            f.write(lily_jb)
        print(f"  Saved: {output_path / 'lead_sheet_jukebox.ly'}")
        
    except Exception as e:
        print(f"✗ Jukebox transcription failed: {e}")
        print("  (May require GPU with >=12GB VRAM)")
        lead_sheet_jb = None
    
    print("\n" + "=" * 70)
    print("Comparison complete!")
    print("=" * 70)
    print("\nCompare the two .ly files to see the quality difference:")
    print(f"  - Handcrafted: {output_path / 'lead_sheet_handcrafted.ly'}")
    print(f"  - Jukebox: {output_path / 'lead_sheet_jukebox.ly'}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Transcribe audio using SheetSage with Jukebox embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic transcription with Jukebox
  python jukebox_transcription.py audio.mp3
  
  # With timing hints
  python jukebox_transcription.py audio.mp3 --start 30 --end 60 --bpm 120
  
  # Compare handcrafted vs Jukebox
  python jukebox_transcription.py audio.mp3 --compare
  
  # Custom output directory
  python jukebox_transcription.py audio.mp3 -o ./my_output

Note: Jukebox features require GPU with >=12GB VRAM and are slower
      but provide better transcription quality.
        """
    )
    
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to the audio file (supports .mp3, .wav, .flac, etc.)"
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default="./output",
        help="Directory to save the output files (default: ./output)"
    )
    parser.add_argument(
        "-s", "--start",
        type=float,
        default=None,
        help="Approximate start time in seconds (optional)"
    )
    parser.add_argument(
        "-e", "--end",
        type=float,
        default=None,
        help="Approximate end time in seconds (optional)"
    )
    parser.add_argument(
        "-b", "--bpm",
        type=int,
        default=None,
        help="Approximate beats per minute (optional, improves accuracy)"
    )
    parser.add_argument(
        "-c", "--compare",
        action="store_true",
        help="Compare handcrafted vs Jukebox features (runs both)"
    )
    
    args = parser.parse_args()
    
    # Check if audio file exists
    audio_path = pathlib.Path(args.audio_path)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)
    
    # Run comparison if requested
    if args.compare:
        compare_handcrafted_vs_jukebox(str(audio_path), args.output_dir)
    else:
        # Run with Jukebox features
        transcribe_with_hints(
            str(audio_path),
            start_time=args.start,
            end_time=args.end,
            bpm=args.bpm,
            output_dir=args.output_dir
        )

