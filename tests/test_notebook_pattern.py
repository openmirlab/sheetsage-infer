#!/usr/bin/env python3
"""
Test using the exact usage pattern from Inference.ipynb notebook.
Tests the complete transcription pipeline with Jukebox.
"""

import logging
from sheetsage.infer import sheetsage
from sheetsage.utils import engrave
from sheetsage.align import create_beat_to_time_fn
from io import BytesIO
import pathlib

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Configuration from notebook
AUDIO_URL = 'https://foodgroup.bandcamp.com/track/universe'
USE_JUKEBOX = True
SEGMENT_START_HINT = 69
SEGMENT_END_HINT = 88
BPM_HINT = 76


def test_transcription_pipeline():
    """Test complete transcription pipeline matching notebook pattern."""
    print("=" * 80)
    print("Testing Notebook Usage Pattern")
    print("=" * 80)
    
    # Transcribe
    lead_sheet, segment_beats, segment_beats_times = sheetsage(
        AUDIO_URL,
        use_jukebox=USE_JUKEBOX,
        segment_start_hint=SEGMENT_START_HINT,
        segment_end_hint=SEGMENT_END_HINT,
        beats_per_minute_hint=BPM_HINT
    )
    
    assert len(lead_sheet) > 0, "Lead sheet should have measures"
    assert len(segment_beats) > 0, "Should have detected beats"
    print(f"✓ Transcription: {len(lead_sheet)} measures, {len(segment_beats)} beats")
    
    # Generate PNG
    lead_sheet_png = engrave(lead_sheet.as_lily())
    assert len(lead_sheet_png) > 0, "PNG should be generated"
    print(f"✓ PNG generated: {len(lead_sheet_png) / 1024:.1f} KB")
    
    # Create MIDI
    import pretty_midi
    beat_to_time_fn = create_beat_to_time_fn(segment_beats, segment_beats_times)
    midi_bytes = lead_sheet.as_midi(beat_to_time_fn)
    midi = pretty_midi.PrettyMIDI(BytesIO(midi_bytes))
    assert midi.get_end_time() > 0, "MIDI should have duration"
    print(f"✓ MIDI created: {midi.get_end_time():.2f}s, {len(midi.instruments)} instruments")
    
    # Save LilyPond
    lily = lead_sheet.as_lily()
    assert len(lily) > 0, "LilyPond should be generated"
    print(f"✓ LilyPond generated: {len(lily) / 1024:.1f} KB")
    
    print("=" * 80)
    print("All tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    test_transcription_pipeline()

