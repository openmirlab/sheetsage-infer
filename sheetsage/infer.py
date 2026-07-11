"""Main transcription pipeline: audio -> LeadSheet (melody + chords).

`sheetsage()` is the public entry point (also exposed as a CLI via `main()`
/ `python -m sheetsage.infer`): it downbeat-detects, splits long audio into
model-sized chunks, extracts either handcrafted (CPU) or Jukebox (GPU)
features per chunk, runs the transcription transducer, and assembles the
per-chunk results into one `LeadSheet`. The enums/constants and the private
`_*` step helpers (single-call, sequential steps of that one pipeline: init
model -> beat-track -> chunk -> extract features -> transcribe -> format)
live in `sheetsage/pipeline/` and are re-exported here so every name that
used to be importable from this module still is.

Reads: .pipeline, .align, .utils; read by: examples/*, README.md Quick Start
/ CLI sections
"""

import logging
import pathlib

import validators

from .align import create_beat_to_time_fn
from .pipeline import (
    _FAMILY_TO_INTERVALS,
    _HARMONY_FAMILIES,
    _INPUT_TO_DIM,
    _INPUT_TO_FRAME_RATE,
    _JUKEBOX_CHUNK_DURATION_EDGE,
    _MAX_TERTIARIES_PER_CHUNK,
    _MELODY_PITCH_MIN,
    _TASK_TO_VOCAB_SIZE,
    _TERTIARIES_PER_BEAT,
    InputFeats,
    Model,
    Status,
    Task,
    _beat_tracking_with_hints,
    _closest_idx,
    _extract_features,
    _format_lead_sheet,
    _init_extractor,
    _init_model,
    _split_into_chunks,
    _transcribe_chunks,
)
from .utils import retrieve_audio_bytes


def sheetsage(
    audio_path_bytes_or_url,
    segment_start_hint=None,
    segment_end_hint=None,
    use_jukebox=False,
    measures_per_chunk=8,
    segment_hints_are_downbeats=False,
    beats_per_measure_hint=None,
    beats_per_minute_hint=None,
    detect_melody=True,
    detect_harmony=True,
    melody_threshold=None,
    harmony_threshold=None,
    beat_detection_padding=15.0,
    avoid_chunking_if_possible=True,
    legacy_behavior=False,
    status_change_callback=lambda s: logging.info(s.name),
    return_intermediaries=False,
    tqdm=lambda x: x,
):
    """Main driver function for Sheet Sage: music audio -> lead sheet.

    Parameters
    ----------
    audio_path_bytes_or_url : :class:`pathlib.Path`, bytes, or str
       The filepath, raw bytes, or string URL of the audio to transcribe.
    segment_start_hint : float or None
       Approximate timestamp of start downbeat (to transcribe a segment of the audio).
    segment_end_hint : float or None
       Approximate timestamp of end downbeat (to transcribe a segment of the audio).
    use_jukebox : bool
       If True, improves transcription quality by using OpenAI Jukebox (requires GPU w/
       >=12GB VRAM).
    measures_per_chunk : int
       The number of measures which Sheet Sage transcribes at a time (for best results,
       set to phrase length).
    segment_hints_are_downbeats: bool
       If True, overrides downbeat detection using the specified segment hints (note
       that the hints must be *very* precise for this to work as intended).
    beats_per_measure_hint : int or None
       If specified, overrides time signature detection (4 for "4/4" or 3 for "3/4").
    beats_per_minute_hint : int or None
       If specified, helps the beat detector find the right tempo. Useful if detected
       tempo is a factor of 2 off from real tempo.
    detect_melody : bool
       If False, skips melody transcription.
    detect_harmony : bool
       If False, skips chord recognition.
    melody_threshold : float
       If specified, overrides default melody threshold (0-1, lower for more notes.)
    harmony_threshold : float
       If specified, overrides default harmony threshold (0-1, lower for more chords.)
    beat_detection_padding : float
       Amount of audio padding to use when running beat detection on segment.
    avoid_chunking_if_possible : bool
       If False, uses chunking even for segments shorter than training length.
    legacy_behavior : bool
       If True, ignores segment_end_hint and transcribes exactly one max-length chunk.
    status_change_callback : Callable
       If specified, calls this method upon changes in `Status`.
    return_intermediaries : bool
       If True, returns intermediate high-level results.

    Returns
    -------
    :class:`sheetsage.LeadSheet`
       Pass
    Callable[float, float]
       Metronome function for converting beat values to timestamps
    """
    # Check values
    if segment_start_hint is not None and segment_start_hint < 0:
        raise ValueError("Segment start hint cannot be negative")
    if segment_end_hint is not None and segment_end_hint < 0:
        raise ValueError("Segment end hint cannot be negative")
    if (
        segment_start_hint is not None
        and segment_end_hint is not None
        and segment_end_hint <= segment_start_hint
    ):
        raise ValueError("Segment end hint should be greater than start hint")
    if measures_per_chunk <= 0:
        raise ValueError("Invalid measures per chunk specified")
    if measures_per_chunk > 24:
        # TODO: Allow 32 if time signature is 3/4??
        raise ValueError("Sheet Sage can only transcribe 24 measures per chunk")
    if beats_per_measure_hint is not None and beats_per_measure_hint not in [3, 4]:
        raise ValueError("Currently, Sheet Sage only supports 4/4 and 3/4 time signatures")
    if beat_detection_padding < 0:
        raise ValueError("Beat detection padding cannot be negative")
    input_feats = InputFeats.JUKEBOX if use_jukebox else InputFeats.HANDCRAFTED

    # Disambiguate between URL and file path for string inputs and retrieve URL
    audio_path_or_bytes = audio_path_bytes_or_url
    if isinstance(audio_path_bytes_or_url, str):
        if validators.url(audio_path_bytes_or_url):
            status_change_callback(Status.FETCHING_AUDIO)
            logging.info(f"Retrieving audio from {audio_path_bytes_or_url}")
            audio_path_or_bytes = retrieve_audio_bytes(audio_path_bytes_or_url)
        else:
            logging.info(f"Loading audio from {audio_path_bytes_or_url}")
            audio_path_or_bytes = pathlib.Path(audio_path_bytes_or_url).resolve()
    if isinstance(audio_path_or_bytes, pathlib.Path) and not audio_path_or_bytes.exists():
        raise FileNotFoundError(audio_path_or_bytes)

    # Run beat detection
    status_change_callback(Status.DETECTING_BEATS)
    (
        beats_per_measure,
        beats,
        beats_times,
        tertiaries,
        tertiaries_times,
        segment_start_downbeat,
        segment_end_beat,
    ) = _beat_tracking_with_hints(
        audio_path_or_bytes,
        segment_start_hint,
        segment_end_hint,
        segment_hints_are_downbeats,
        beats_per_measure_hint,
        beats_per_minute_hint,
        beat_detection_padding,
        legacy_behavior,
    )

    # Identify suitable chunks for running through transcription model
    chunks_tertiaries = _split_into_chunks(
        tertiaries_times,
        measures_per_chunk,
        beats_per_measure,
        segment_start_downbeat,
        segment_end_beat,
        avoid_chunking_if_possible,
        legacy_behavior,
    )

    # Extract features
    status_change_callback(Status.EXTRACTING_FEATURES)
    if use_jukebox:
        logging.info("Feature extraction w/ Jukebox could take several minutes.")
    chunks_features = _extract_features(
        audio_path_or_bytes, input_feats, tertiaries_times, chunks_tertiaries, tqdm
    )

    # Transcribe chunks
    status_change_callback(Status.TRANSCRIBING)
    melody_logits, harmony_logits = _transcribe_chunks(
        chunks_features, input_feats, detect_melody, detect_harmony
    )

    # Create lead sheet
    status_change_callback(Status.FORMATTING)
    total_num_tertiary = sum([c.shape[0] for c in chunks_features])
    lead_sheet, segment_beats, segment_beats_times = _format_lead_sheet(
        melody_logits,
        harmony_logits,
        beats_per_measure,
        beats,
        beats_times,
        segment_start_downbeat,
        segment_end_beat,
        total_num_tertiary,
        melody_threshold=melody_threshold,
        harmony_threshold=harmony_threshold,
    )

    status_change_callback(Status.DONE)
    result = lead_sheet, segment_beats, segment_beats_times
    if return_intermediaries:
        result = result + (chunks_tertiaries, melody_logits, harmony_logits)
    return result


def main():
    """Entry point for the sheetsage CLI command."""
    import pathlib
    import uuid
    from argparse import ArgumentParser

    from tqdm import tqdm

    from .utils import engrave

    parser = ArgumentParser()

    parser.add_argument(
        "audio_path_or_url",
        type=str,
        help="The filepath or URL of the audio to transcribe.",
    )
    parser.add_argument(
        "-s",
        "--segment_start_hint",
        type=float,
        help="Approximate timestamp of start downbeat (to transcribe a segment of the audio).",
    )
    parser.add_argument(
        "-e",
        "--segment_end_hint",
        type=float,
        help="Approximate timestamp of end downbeat (to transcribe a segment of the audio).",
    )
    parser.add_argument(
        "-t",
        "--title",
        type=str,
        help="Title of the song.",
    )
    parser.add_argument(
        "-a",
        "--artist",
        type=str,
        help="Name of the artist or composer.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="Directory to save the output files (lead sheet PDF, synchronized MIDI, etc.).",
    )
    parser.add_argument(
        "-j",
        "--use_jukebox",
        action="store_true",
        help="If set, improves transcription quality by using OpenAI Jukebox (requires GPU w/ >=12GB VRAM).",
    )
    parser.add_argument(
        "--measures_per_chunk",
        type=int,
        help="The number of measures which Sheet Sage transcribes at a time (for best results, set to phrase length).",
    )
    parser.add_argument(
        "--segment_hints_are_downbeats",
        action="store_true",
        help="If set, overrides downbeat detection using the specified segment hints (note that the hints must be *very* precise for this to work as intended).",
    )
    parser.add_argument(
        "--beats_per_measure",
        type=int,
        choices=[3, 4],
        help="If specified, overrides time signature detection (4 for '4/4' or 3 for '3/4').",
    )
    parser.add_argument(
        "--beats_per_minute_hint",
        type=int,
        help="If specified, helps the beat detector find the right tempo. Useful if detected tempo is a factor of 2 off from real tempo.",
    )
    parser.add_argument(
        "--melody_threshold",
        type=float,
        help="If specified, overrides default melody threshold (0-1, lower for more notes.)",
    )
    parser.add_argument(
        "--harmony_threshold",
        type=float,
        help="If specified, overrides default harmony threshold (0-1, lower for more chords.)",
    )
    parser.add_argument(
        "--skip_melody",
        action="store_false",
        dest="detect_melody",
        help="If set, skips melody transcription.",
    )
    parser.add_argument(
        "--skip_harmony",
        action="store_false",
        dest="detect_harmony",
        help="If set, skips chord recognition.",
    )
    parser.add_argument(
        "--legacy_behavior",
        action="store_true",
        dest="legacy_behavior",
        help="If set, ignores segment_end_hint and transcribes exactly one max-length chunk.",
    )

    parser.set_defaults(
        segment_start_hint=None,
        segment_end_hint=None,
        title=None,
        artist=None,
        output_dir="./output",
        use_jukebox=False,
        measures_per_chunk=8,
        segment_hints_are_downbeats=False,
        beats_per_measure=None,
        beats_per_minute_hint=None,
        melody_threshold=None,
        harmony_threshold=None,
        detect_melody=True,
        detect_harmony=True,
        legacy_behavior=False,
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    lead_sheet, segment_beats, segment_beats_times = sheetsage(
        args.audio_path_or_url,
        segment_start_hint=args.segment_start_hint,
        segment_end_hint=args.segment_end_hint,
        use_jukebox=args.use_jukebox,
        measures_per_chunk=args.measures_per_chunk,
        segment_hints_are_downbeats=args.segment_hints_are_downbeats,
        beats_per_measure_hint=args.beats_per_measure,
        beats_per_minute_hint=args.beats_per_minute_hint,
        detect_melody=args.detect_melody,
        detect_harmony=args.detect_harmony,
        melody_threshold=args.melody_threshold,
        harmony_threshold=args.harmony_threshold,
        legacy_behavior=args.legacy_behavior,
        tqdm=tqdm,
    )

    # Create output directory
    output_dir = pathlib.Path(args.output_dir).resolve()
    if output_dir == pathlib.Path("./output").resolve():
        uuid = uuid.uuid4().hex
        output_dir = pathlib.Path(output_dir, uuid)
    logging.info(f"Writing to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write lead sheet
    lily = lead_sheet.as_lily(artist=args.artist, title=args.title)
    with open(pathlib.Path(output_dir, "output.ly"), "w") as f:
        f.write(lily)
    with open(pathlib.Path(output_dir, "output.pdf"), "wb") as f:
        f.write(engrave(lily, out_format="pdf", transparent=False, trim=False, hide_footer=False))

    # Write MIDI
    with open(pathlib.Path(output_dir, "output.midi"), "wb") as f:
        f.write(
            lead_sheet.as_midi(
                pulse_to_time_fn=create_beat_to_time_fn(segment_beats, segment_beats_times)
            )
        )


if __name__ == "__main__":
    main()
