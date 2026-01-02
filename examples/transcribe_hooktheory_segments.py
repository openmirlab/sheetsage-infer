#!/usr/bin/env python3
"""Transcribe audio from Hooktheory Test Segments."""

import json
import pathlib
import sys
import tempfile

import soundfile
import yt_dlp

from sheetsage.align import create_beat_to_time_fn
from sheetsage.infer import sheetsage
from sheetsage.utils import decode_audio, engrave


def construct_youtube_url(audio_tag):
    """Convert audio_tag to YouTube URL."""
    if audio_tag.startswith("YOUTUBE_"):
        return f"https://www.youtube.com/watch?v={audio_tag.replace('YOUTUBE_', '')}"
    return None


def download_audio_segment(youtube_url, segment_start, segment_end, output_path):
    """Download audio segment from YouTube and save as WAV."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': str(pathlib.Path(tmpdir) / 'audio.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
            downloaded_files = list(pathlib.Path(tmpdir).glob('audio.*'))
            if not downloaded_files:
                return False
            sr, audio = decode_audio(
                str(downloaded_files[0]),
                offset=segment_start,
                duration=segment_end - segment_start,
                sr=44100
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            soundfile.write(str(output_path), audio, sr, format='WAV')
            return True
        except Exception:
            return False


def transcribe_segment(segment_id, segment_data, output_dir, use_jukebox=False):
    """Transcribe a single segment."""
    audio_tag = segment_data.get("audio_tag", "")
    segment_start = segment_data.get("segment_start", 0)
    segment_end = segment_data.get("segment_end", 0)
    youtube_url = construct_youtube_url(audio_tag)
    if not youtube_url:
        return False

    segment_dir = output_dir / segment_id
    segment_dir.mkdir(parents=True, exist_ok=True)

    download_audio_segment(youtube_url, segment_start, segment_end, segment_dir / f"{segment_id}.wav")

    try:
        lead_sheet, segment_beats, segment_beats_times = sheetsage(
            youtube_url,
            use_jukebox=use_jukebox,
            segment_start_hint=segment_start,
            segment_end_hint=segment_end,
            detect_melody=True,
            detect_harmony=True
        )

        lily_code = lead_sheet.as_lily()
        with open(segment_dir / f"{segment_id}.ly", 'w', encoding='utf-8') as f:
            f.write(lily_code)

        beat_to_time_fn = create_beat_to_time_fn(segment_beats, segment_beats_times)
        midi_bytes = lead_sheet.as_midi(beat_to_time_fn)
        with open(segment_dir / f"{segment_id}.mid", 'wb') as f:
            f.write(midi_bytes)

        try:
            pdf_bytes = engrave(lily_code, out_format='pdf')
            with open(segment_dir / f"{segment_id}.pdf", 'wb') as f:
                f.write(pdf_bytes)
        except Exception:
            pass

        metadata = {
            "segment_id": segment_id,
            "audio_tag": audio_tag,
            "youtube_url": youtube_url,
            "segment_start": segment_start,
            "segment_end": segment_end,
            "duration": segment_end - segment_start,
            "use_jukebox": use_jukebox,
            "num_beats": len(segment_beats),
            "lead_sheet_length": len(lead_sheet)
        }
        with open(segment_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        return True
    except Exception:
        return False


def main():
    """Main function."""
    segments_file = pathlib.Path("hooktheory_data/Hooktheory_Test_Segments.json")
    output_dir = pathlib.Path("hooktheory_transcription_results")
    num_segments = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    use_jukebox = sys.argv[2].lower() == "true" if len(sys.argv) > 2 else False

    if not segments_file.exists():
        print(f"ERROR: Segments file not found: {segments_file}")
        return 1

    with open(segments_file, encoding='utf-8') as f:
        segments = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    segment_ids = list(segments.keys())[:num_segments]
    results = {"success": [], "failed": []}

    for segment_id in segment_ids:
        if transcribe_segment(segment_id, segments[segment_id], output_dir, use_jukebox):
            results["success"].append(segment_id)
        else:
            results["failed"].append(segment_id)

    print(f"SUCCESS: {len(results['success'])}/{len(segment_ids)} successful")
    if results["failed"]:
        print(f"FAILED: {', '.join(results['failed'])}")
    return 0 if len(results["failed"]) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

