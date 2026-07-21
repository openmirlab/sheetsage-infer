"""Pipeline steps: the sequential, single-call helpers `sheetsage()` drives.

Each `_*` function here is one step of the one pipeline (init model -> beat-track
-> chunk -> extract features -> transcribe -> format) rather than an independent
utility, moved verbatim out of `sheetsage/infer.py` so that file can stay focused
on the public `sheetsage()`/`main()` entry points. Order below matches call order.

Reads: .types, ..align, ..assets, ..beat_track, ..modules, ..representations,
..theory, ..utils; read by: sheetsage.infer (via sheetsage.pipeline)
"""

import json
import tempfile
from dataclasses import dataclass

import numpy as np
import torch
from scipy.special import softmax

from ..align import create_beat_to_time_fn
from ..assets import retrieve_asset
from ..beat_track import madmom
from ..modules import EncOnlyTransducer, IdentityEncoder, TransformerEncoder
from ..representations import Handcrafted, JukeboxEmbeddings
from ..theory import (
    Chord,
    Harmony,
    KeyChanges,
    LeadSheet,
    Melody,
    MeterChanges,
    Note,
    TempoChanges,
    estimate_key_changes,
)
from ..utils import decode_audio
from .types import (
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
    Task,
)


def _init_extractor(input_feats, device="auto"):
    if input_feats == InputFeats.HANDCRAFTED:
        extractor = Handcrafted()
    elif input_feats == InputFeats.JUKEBOX:
        extractor = JukeboxEmbeddings(device=device)
    else:
        raise ValueError()
    return extractor


def _init_model(task, input_feats, model, device="cpu"):
    if model == Model.LINEAR:
        # NOTE: Just need to catalogue these configs / weights
        raise NotImplementedError()

    asset_prefix = f"SHEETSAGE_V02_{input_feats.name}_{task.name}"
    with open(retrieve_asset(f"{asset_prefix}_CFG", log=False)) as f:
        cfg = json.load(f)
    assert cfg["src_max_len"] == _MAX_TERTIARIES_PER_CHUNK

    src_dim = _INPUT_TO_DIM[input_feats]
    output_dim = _TASK_TO_VOCAB_SIZE[task]

    if cfg["model"] == "probe":
        model = EncOnlyTransducer(
            output_dim,
            src_emb_mode="identity",
            src_vocab_size=None,
            src_dim=src_dim,
            src_emb_dim=None,
            src_pos_emb=False,
            src_dropout_p=0.0,
            enc_cls=IdentityEncoder,
            enc_kwargs={},
        )
    elif cfg["model"] == "transformer":
        model = EncOnlyTransducer(
            output_dim,
            src_emb_mode="project",
            src_vocab_size=None,
            src_dim=src_dim,
            src_emb_dim=512,
            src_pos_emb="pos_emb" in cfg["hacks"],
            src_dropout_p=0.1,
            enc_cls=TransformerEncoder,
            enc_kwargs={
                "model_dim": 512,
                "num_heads": 8,
                "num_layers": 4 if "4layers" in cfg["hacks"] else 6,
                "feedforward_dim": 2048,
                "dropout_p": 0.1,
            },
        )
    else:
        raise ValueError()

    device = torch.device(device)
    model.to(device)
    model.load_state_dict(
        torch.load(retrieve_asset(f"{asset_prefix}_MODEL", log=False), map_location=device)
    )
    model.eval()
    return model


@dataclass
class PipelineComponents:
    extractor: object
    melody_model: object | None
    harmony_model: object | None


def load_components(input_feats, detect_melody, detect_harmony, device):
    """Construct the session-owned feature extractor and requested transducers once."""
    return PipelineComponents(
        extractor=_init_extractor(input_feats, device),
        melody_model=(
            _init_model(Task.MELODY, input_feats, Model.TRANSFORMER, device)
            if detect_melody
            else None
        ),
        harmony_model=(
            _init_model(Task.HARMONY, input_feats, Model.TRANSFORMER, device)
            if detect_harmony
            else None
        ),
    )


def _closest_idx(x, lst):
    assert len(lst) > 0
    return int(np.argmin([abs(li - x) for li in lst]) + 1e-6)


def _beat_tracking_with_hints(
    audio_path_or_bytes,
    segment_start_hint,
    segment_end_hint,
    segment_hints_are_downbeats,
    beats_per_measure_hint,
    beats_per_minute_hint,
    beat_detection_padding,
    legacy_behavior,
):
    # Decode a segment of the audio
    beat_detection_start = 0.0 if segment_start_hint is None else segment_start_hint
    beat_detection_start = max(beat_detection_start - beat_detection_padding, 0.0)
    beat_detection_end = None if segment_end_hint is None else segment_end_hint
    beat_detection_end = (
        None if beat_detection_end is None else beat_detection_end + beat_detection_padding
    )
    if legacy_behavior:
        left = segment_start_hint - beat_detection_padding
        right = segment_start_hint + _JUKEBOX_CHUNK_DURATION_EDGE + beat_detection_padding
        sr, audio = decode_audio(audio_path_or_bytes)
        audio_duration = audio.shape[0] / sr
        left, right = [round(t * sr) for t in (left, right)]
        left = max(0, left)
        right = min(audio.shape[0], right)
        assert right > left
        audio = audio[left:right]
    else:
        sr, audio = decode_audio(
            audio_path_or_bytes,
            offset=beat_detection_start,
            duration=None
            if beat_detection_end is None
            else beat_detection_end - beat_detection_start,
        )

    # Run beat detection on segment
    first_downbeat_idx, beats_per_measure, beats = madmom(
        sr,
        audio,
        beats_per_bar=beats_per_measure_hint if beats_per_measure_hint is not None else [3, 4],
        beats_per_minute_hint=beats_per_minute_hint,
    )
    if first_downbeat_idx is None or beats_per_measure is None or len(beats) == 0:
        raise ValueError("Audio too short to detect time signature")
    assert first_downbeat_idx >= 0 and first_downbeat_idx < beats_per_measure
    assert beats_per_measure in [3, 4]
    beats = [beat_detection_start + t for t in beats]
    downbeats = [t for i, t in enumerate(beats) if i % beats_per_measure == first_downbeat_idx]
    assert len(beats) > 0
    assert len(downbeats) > 0

    # Convert beats into tertiary (sixteenth note) timestamps
    # NOTE: Yes, this is super ugly, but sometimes you gotta do what you gotta do
    beat_to_time_fn = create_beat_to_time_fn(list(range(len(beats))), beats)
    tertiaries = np.arange(0, len(beats) - 1 + 1e-6, 1 / _TERTIARIES_PER_BEAT)
    assert tertiaries.shape[0] == (len(beats) - 1) * _TERTIARIES_PER_BEAT + 1
    tertiaries_centered = tertiaries - (1 / _TERTIARIES_PER_BEAT) / 2
    tertiaries_times = beat_to_time_fn(tertiaries_centered)
    tertiaries_times = np.maximum(tertiaries_times, 0.0)
    tertiaries_times = np.minimum(tertiaries_times, beats[-1])

    # Find first downbeat of the song from optional hint
    if segment_start_hint is None:
        segment_start = downbeats[0]
    else:
        if segment_hints_are_downbeats:
            segment_start = segment_start_hint
        else:
            segment_start = downbeats[_closest_idx(segment_start_hint, downbeats)]
    segment_start_downbeat = _closest_idx(segment_start, beats)
    downbeats = [
        t
        for i, t in enumerate(beats)
        if i % beats_per_measure == segment_start_downbeat % beats_per_measure
    ]

    # Find last downbeat of the song from optional hint
    if segment_end_hint is None:
        segment_end = downbeats[-1]
    else:
        if segment_hints_are_downbeats:
            segment_end = segment_end_hint
        else:
            segment_end = downbeats[_closest_idx(segment_end_hint, downbeats)]
    segment_end_beat = _closest_idx(segment_end, beats)
    if segment_end_beat == segment_start_downbeat:
        raise ValueError("Specified segment is too short (<1 measure).")

    # NOTE on naming conventions: segment_start_downbeat *is* an (internally-consistent)
    # downbeat, but segment_end_beat may not be (if segment_hints_are_downbeats is true
    # and user specifies an inaccurate timestamp).

    if legacy_behavior:
        beats = beats[segment_start_downbeat:]

        beat_to_time_fn = create_beat_to_time_fn(list(range(len(beats))), beats)
        tertiaries = np.arange(0, len(beats) + 1e-6, 1 / _TERTIARIES_PER_BEAT)
        assert tertiaries.shape[0] > 0
        tertiaries -= (1 / _TERTIARIES_PER_BEAT) / 2
        tertiaries_times = beat_to_time_fn(tertiaries)
        tertiaries_times = np.maximum(tertiaries_times, 0.0)
        tertiaries_times = np.minimum(tertiaries_times, audio_duration)
        segment_offset = tertiaries_times[0]
        tertiaries_times = [
            t for t in tertiaries_times if t < segment_offset + _JUKEBOX_CHUNK_DURATION_EDGE
        ]
        tertiaries_times[-1] - segment_offset
        tertiaries = (np.arange(len(tertiaries_times)) * (1 / _TERTIARIES_PER_BEAT)).tolist()

        segment_end_beat = segment_start_downbeat + len(tertiaries) / _TERTIARIES_PER_BEAT
        if abs(segment_end_beat - round(segment_end_beat)) < 1e-6:
            segment_end_beat = round(segment_end_beat)
        else:
            segment_end_beat = int(np.ceil(segment_end_beat) + 1e-6)
        tertiaries = np.array(tertiaries)
        tertiaries_times = np.array(tertiaries_times)

    return (
        beats_per_measure,
        list(range(len(beats))),
        beats,
        tertiaries,
        tertiaries_times,
        segment_start_downbeat,
        segment_end_beat,
    )


def _split_into_chunks(
    tertiaries_times,
    measures_per_chunk,
    beats_per_measure,
    segment_start_downbeat,
    segment_end_beat,
    avoid_chunking_if_possible,
    legacy_behavior,
):
    chunks = []

    if legacy_behavior:
        chunk_slice = slice(None, None)
        chunk_tertiaries_times = tertiaries_times[chunk_slice]
        duration = chunk_tertiaries_times[-1] - chunk_tertiaries_times[0]
        assert duration > 0 and duration <= _JUKEBOX_CHUNK_DURATION_EDGE
        chunks.append(chunk_slice)
    else:
        beats_per_chunk = beats_per_measure * measures_per_chunk
        if avoid_chunking_if_possible:
            chunk_start_tertiary = segment_start_downbeat * _TERTIARIES_PER_BEAT
            chunk_end_tertiary = (segment_end_beat * _TERTIARIES_PER_BEAT) + 1
            chunk_slice = slice(chunk_start_tertiary, chunk_end_tertiary)
            chunk_tertiaries_times = tertiaries_times[chunk_slice]
            duration = chunk_tertiaries_times[-1] - chunk_tertiaries_times[0]
            if duration <= _JUKEBOX_CHUNK_DURATION_EDGE:
                beats_per_chunk = segment_end_beat

        for b in range(segment_start_downbeat, segment_end_beat, beats_per_chunk):
            chunk_start_tertiary = b * _TERTIARIES_PER_BEAT
            chunk_end_tertiary = ((b + beats_per_chunk) * _TERTIARIES_PER_BEAT) + 1
            chunk_end_tertiary = min(
                chunk_end_tertiary, (segment_end_beat * _TERTIARIES_PER_BEAT) + 1
            )
            assert chunk_end_tertiary <= tertiaries_times.shape[0]
            chunk_slice = slice(chunk_start_tertiary, chunk_end_tertiary)
            chunk_tertiaries_times = tertiaries_times[chunk_slice]
            duration = chunk_tertiaries_times[-1] - chunk_tertiaries_times[0]
            assert duration > 0
            if duration > _JUKEBOX_CHUNK_DURATION_EDGE:
                raise NotImplementedError(
                    "Dynamic chunking not implemented. Try halving measures_per_chunk."
                )
            chunks.append(chunk_slice)

    return chunks


def _extract_features(
    audio_path_or_bytes, input_feats, tertiaries_times, chunks_tertiaries, tqdm, extractor=None, device="auto"
):
    tertiary_diff_frames = np.diff(tertiaries_times) * _INPUT_TO_FRAME_RATE[input_feats]
    if np.any(tertiary_diff_frames.astype(np.int64) == 0):
        raise ValueError("Tempo too fast for beat-informed feature resampling")

    extractor = extractor or _init_extractor(input_feats, device)
    chunks_features = []
    with tempfile.NamedTemporaryFile("wb") as f:
        if isinstance(audio_path_or_bytes, bytes):
            f.write(audio_path_or_bytes)
            f.flush()
            audio_path = f.name
        else:
            audio_path = audio_path_or_bytes

        for chunk_slice in tqdm(chunks_tertiaries):
            chunk_tertiaries_times = tertiaries_times[chunk_slice]
            offset = chunk_tertiaries_times[0]
            duration = chunk_tertiaries_times[-1] - offset
            assert duration <= _JUKEBOX_CHUNK_DURATION_EDGE
            fr, feats = extractor(audio_path, offset=offset, duration=duration)
            beat_resampled = []
            for i in range(chunk_tertiaries_times.shape[0] - 1):
                s = int((chunk_tertiaries_times[i] - offset) * fr)
                e = int((chunk_tertiaries_times[i + 1] - offset) * fr)
                assert e > s
                beat_resampled.append(np.mean(feats[s:e], axis=0, keepdims=True))
            beat_resampled = np.concatenate(beat_resampled, axis=0)
            chunks_features.append(beat_resampled)

    # Normalize handcrafted features (after beat resampling)
    # NOTE: Normalizing after beat resampling is probably a bug in retrospect, but it's
    # what the model expects.
    if input_feats == InputFeats.HANDCRAFTED:
        moments = np.load(retrieve_asset(f"SHEETSAGE_V02_{input_feats.name}_MOMENTS", log=False))
        for chunk in chunks_features:
            chunk -= moments[0]
            chunk /= moments[1]

    return chunks_features


def _transcribe_chunks(chunks_features, input_feats, detect_melody, detect_harmony, device="cpu", components=None):
    melody_logits = None
    if detect_melody:
        melody_model = components.melody_model if components else _init_model(Task.MELODY, input_feats, Model.TRANSFORMER, device)
        melody_logits = []

    harmony_logits = None
    if detect_harmony:
        harmony_model = components.harmony_model if components else _init_model(Task.HARMONY, input_feats, Model.TRANSFORMER, device)
        harmony_logits = []

    if detect_melody or detect_harmony:
        device = torch.device(device)
        with torch.no_grad():
            for src in chunks_features:
                src_len = src.shape[0]
                src = np.pad(src, [(0, _MAX_TERTIARIES_PER_CHUNK - src_len), (0, 0)])
                src = src[:, np.newaxis]
                src = torch.tensor(src).float()
                src_len = torch.tensor(src_len).long().view(-1)
                src = src.to(device)
                src_len = src_len.to(device)

                if detect_melody:
                    chunk_melody_logits = melody_model(src, src_len, None, None)
                    chunk_melody_logits = chunk_melody_logits[: src_len.item(), 0]
                    melody_logits.append(chunk_melody_logits.cpu().numpy())
                if detect_harmony:
                    chunk_harmony_logits = harmony_model(src, src_len, None, None)
                    chunk_harmony_logits = chunk_harmony_logits[: src_len.item(), 0]
                    harmony_logits.append(chunk_harmony_logits.cpu().numpy())

    total_num_tertiary = sum([c.shape[0] for c in chunks_features])
    if detect_melody:
        assert sum([c.shape[0] for c in melody_logits]) == total_num_tertiary
    if detect_harmony:
        assert sum([c.shape[0] for c in harmony_logits]) == total_num_tertiary

    return melody_logits, harmony_logits


def _format_lead_sheet(
    melody_logits,
    harmony_logits,
    beats_per_measure,
    beats,
    beats_times,
    segment_start_downbeat,
    segment_end_beat,
    total_num_tertiary,
    melody_threshold=None,
    harmony_threshold=None,
):
    def decode(logits, threshold=None):
        if threshold is None:
            preds = np.argmax(logits, axis=-1)
        else:
            probs_nonnull = 1 - softmax(logits, axis=-1)[:, 0]
            preds_nonnull = 1 + np.argmax(logits[:, 1:], axis=-1)
            preds = np.where(probs_nonnull >= threshold, preds_nonnull, 0)
        return preds

    # Decode melody
    if melody_logits is None:
        melody = Melody()
    else:
        melody_logits = np.concatenate(melody_logits, axis=0)
        assert melody_logits.shape[0] == total_num_tertiary
        melody_preds = decode(melody_logits, threshold=melody_threshold)
        melody_onsets = []
        for o, p in enumerate(melody_preds):
            if p != 0:
                assert p >= 1
                p -= 1
                p = (p + _MELODY_PITCH_MIN).tolist()
                melody_onsets.append((o, Note(p % 12, p // 12)))
        melody = []
        for i, (o, n) in enumerate(melody_onsets):
            if i + 1 < len(melody_onsets):
                d = melody_onsets[i + 1][0] - o
            else:
                d = total_num_tertiary - o
            melody.append((o, d, n))
        melody = Melody(*melody)

    # Decode harmony
    if harmony_logits is None:
        harmony = Harmony()
    else:
        harmony_logits = np.concatenate(harmony_logits, axis=0)
        assert harmony_logits.shape[0] == total_num_tertiary
        harmony_preds = decode(harmony_logits, threshold=harmony_threshold)
        harmony = []
        last_chord = None
        for o, c in enumerate(harmony_preds):
            if c != 0:
                assert c >= 1
                c -= 1
                c = c.tolist()
                c = (
                    c // len(_HARMONY_FAMILIES),
                    _HARMONY_FAMILIES[c % len(_HARMONY_FAMILIES)],
                )
                chord = Chord(c[0], _FAMILY_TO_INTERVALS[c[1]])
                if chord != last_chord:
                    harmony.append((o, chord))
                last_chord = chord
        harmony = Harmony(*harmony)

    # Extract tempo
    measures_bps = []
    for b in range(segment_start_downbeat, segment_end_beat, beats_per_measure):
        m0_time = beats_times[b]
        try:
            mp1_time = beats_times[b + beats_per_measure]
        except IndexError:
            break
        assert mp1_time >= m0_time
        if mp1_time > m0_time:
            bps = beats_per_measure / (mp1_time - m0_time)
            measures_bps.append(bps)
    if len(measures_bps) > 0:
        beats_per_second = np.median(measures_bps)
    else:
        beats_per_second = 2

    meter_changes = MeterChanges((0, (beats_per_measure, 2, 2)))
    tempo_changes = TempoChanges((0, (round(beats_per_second * 60),)))
    try:
        key_changes = estimate_key_changes(meter_changes, harmony, melody)
    except Exception:
        # NOTE: C major by default
        key_changes = KeyChanges((0, (0, (2, 2, 1, 2, 2, 2))))
    lead_sheet = LeadSheet(
        meter_changes, tempo_changes, key_changes, harmony, melody, total_num_tertiary
    )

    assert beats[0] == 0
    segment_beats = [b - segment_start_downbeat for b in beats]

    return lead_sheet, segment_beats, beats_times
