"""Downbeat/beat detection: madmom-infer's DBN tracker with a librosa fallback.

`madmom()` is the function `infer.py` calls; it tries madmom-infer's
`RNNDownBeatProcessor` + `DBNDownBeatTrackingProcessor` pipeline first (best
quality) and falls back to librosa's beat tracker (logged as a warning) if
madmom-infer isn't installed or its DBN decode fails on the given input.

As of 0.2.1 this calls madmom-infer's Python API directly
(`madmom_infer.features.downbeats`) instead of shelling out to the
`DBNDownBeatTracker` CLI binary the original `madmom` package installed --
madmom-infer is a pure-Python/numpy library with no console script of its
own. One behavior difference this rewrite has to bridge: madmom-infer's
`RNNDownBeatProcessor` requires its input WAV to already be 44.1kHz (it has
no ffmpeg-backed resampler, see `madmom_infer.audio.signal`'s module
docstring), whereas original madmom transparently resampled any input rate
to 44.1kHz internally. `madmom()` below resamples non-44.1kHz audio with
librosa before handing it to madmom-infer to preserve that behavior (not
bit-identical to madmom's ffmpeg resampler, but avoids a hard crash /
silent quality loss on non-44.1kHz input; the bundled TEST_WAV regression
fixture is already 44.1kHz native, so this path doesn't affect it).

Reads: madmom_infer.features.downbeats (optional, see pyproject), librosa,
scipy; read by: sheetsage.infer
"""

import logging
import math
import tempfile
import warnings

try:
    from scipy.io.wavfile import write as wavwrite
except ImportError as exc:  # pragma: no cover - dependency guardrail
    raise ImportError("scipy is required for beat tracking") from exc

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - dependency guardrail
    raise ImportError("numpy is required for beat tracking") from exc


def _normalize_beats_per_bar(beats_per_bar):
    if beats_per_bar is None:
        return None
    if isinstance(beats_per_bar, list):
        return [int(round(b)) for b in beats_per_bar]
    return int(round(beats_per_bar))


def _librosa_fallback(sr, audio, beats_per_bar=None, beats_per_minute_hint=None):
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - librosa is a core dependency
        raise RuntimeError("librosa must be installed for beat tracking fallback") from exc

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32, copy=False)

    start_bpm = 120.0  # default for librosa (None not accepted in librosa >=0.10)
    if beats_per_minute_hint is not None and beats_per_minute_hint > 0:
        start_bpm = float(beats_per_minute_hint)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _tempo, beat_frames = librosa.beat.beat_track(
            y=audio, sr=sr, trim=False, start_bpm=start_bpm
        )

    if beat_frames.size == 0:
        return None, None, []

    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    beats_per_bar_normalized = _normalize_beats_per_bar(beats_per_bar)
    if isinstance(beats_per_bar_normalized, list):
        detected_beats_per_bar = beats_per_bar_normalized[0]
    elif beats_per_bar_normalized is None:
        detected_beats_per_bar = 4
    else:
        detected_beats_per_bar = beats_per_bar_normalized

    detected_beats_per_bar = max(1, int(detected_beats_per_bar))

    # Assume the first beat is a downbeat for the fallback implementation
    first_downbeat = 0

    return first_downbeat, detected_beats_per_bar, beat_times


def _madmom_infer_dbn(sr, audio, beats_per_bar_normalized, beats_per_minute_hint):
    """Run madmom-infer's RNN + DBN downbeat tracker; returns the parsed result.

    Mirrors the parsing/validation the old CLI-output-parsing code did, so
    the return contract (`first_downbeat_idx, beats_per_measure, beat_times`)
    is unchanged. Raises on any processing failure; the caller decides
    whether to fall back to librosa.
    """
    from madmom_infer.features.downbeats import (
        DBNDownBeatTrackingProcessor,
        RNNDownBeatProcessor,
    )

    beat_audio = audio
    beat_sr = sr
    if sr != 44100:
        # madmom-infer has no ffmpeg-backed resampler (see module docstring);
        # resample here to preserve madmom's original "always run the DBN at
        # 44.1kHz" behavior instead of hard-failing on other input rates.
        import librosa

        mono_audio = audio if audio.ndim == 1 else np.mean(audio, axis=1)
        beat_audio = librosa.resample(
            mono_audio.astype(np.float32, copy=False), orig_sr=sr, target_sr=44100
        )
        beat_sr = 44100

    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        wavwrite(f.name, beat_sr, beat_audio)
        activations = RNNDownBeatProcessor()(f.name)

    dbn_kwargs = {
        "beats_per_bar": (
            beats_per_bar_normalized if beats_per_bar_normalized is not None else [3, 4]
        ),
        "fps": 100,
    }
    if beats_per_minute_hint is not None:
        dbn_kwargs["min_bpm"] = beats_per_minute_hint * math.pow(2, -0.5)
        dbn_kwargs["max_bpm"] = beats_per_minute_hint * math.pow(2, 0.5)

    result = np.asarray(DBNDownBeatTrackingProcessor(**dbn_kwargs)(activations))

    if result.size == 0:
        times, positions = [], []
    else:
        # Round to centiseconds: matches the precision the old CLI's text
        # output format actually carried (the old code asserted parsed times
        # were already exact at this precision before proceeding).
        times = [round(t, 2) for t in result[:, 0].tolist()]
        positions = [int(round(p)) for p in result[:, 1].tolist()]

    dbts = [t for t, p in zip(times, positions, strict=True) if p == 1]
    bts = [t for t, p in zip(times, positions, strict=True) if p != 1]

    assert all(t >= 0 for t in dbts + bts)
    assert sorted(dbts) == dbts
    assert sorted(bts) == bts
    assert len(set(dbts)) == len(dbts)
    assert len(set(bts)) == len(bts)
    assert len(set(dbts).intersection(set(bts))) == 0

    first_downbeat = None
    detected_beats_per_bar = None
    merged = sorted(dbts + bts)
    if len(dbts) > 0:
        first_downbeat = merged.index(dbts[0])
        partial_head = [t for t in bts if t < dbts[0]]
        partial_tail = [t for t in bts if t > dbts[-1]]
        for i in range(len(dbts) - 1):
            beats_this_bar = 1
            beats_this_bar += len([t for t in bts if dbts[i] < t < dbts[i + 1]])
            if detected_beats_per_bar is None:
                detected_beats_per_bar = beats_this_bar
            assert beats_this_bar == detected_beats_per_bar
        assert detected_beats_per_bar is None or len(partial_head) < detected_beats_per_bar
        assert detected_beats_per_bar is None or len(partial_tail) < detected_beats_per_bar
        if beats_per_bar_normalized is not None and detected_beats_per_bar is not None:
            if isinstance(beats_per_bar_normalized, list):
                assert detected_beats_per_bar in beats_per_bar_normalized
            else:
                assert detected_beats_per_bar == beats_per_bar_normalized

    return first_downbeat, detected_beats_per_bar, merged


def madmom(sr, audio, beats_per_bar=None, beats_per_minute_hint=None):
    if beats_per_minute_hint is not None and beats_per_minute_hint < 0:
        raise ValueError()

    beats_per_bar_normalized = _normalize_beats_per_bar(beats_per_bar)

    try:
        import madmom_infer.features.downbeats  # noqa: F401
    except ImportError:
        logging.info("madmom-infer not installed. Using librosa beat tracker fallback.")
    else:
        try:
            return _madmom_infer_dbn(sr, audio, beats_per_bar_normalized, beats_per_minute_hint)
        except (
            OSError,
            ValueError,
            RuntimeError,
            AssertionError,
            NotImplementedError,
            IndexError,
        ) as exc:  # pragma: no cover - defensive logging
            logging.warning(
                "madmom-infer DBN downbeat tracking failed (%s). Falling back to librosa "
                "beat tracker.",
                exc,
            )

    return _librosa_fallback(
        sr,
        audio,
        beats_per_bar=beats_per_bar_normalized,
        beats_per_minute_hint=beats_per_minute_hint,
    )
