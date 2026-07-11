"""Downbeat/beat detection: madmom DBN tracker with a librosa fallback.

`madmom()` is the function `infer.py` calls; it tries madmom's
DBNDownBeatTrackingProcessor first (best quality, needs the git-installed
madmom -- see README/pyproject for the PyPI-vs-git install note) and falls
back to librosa's beat tracker (logged as a warning) if madmom's CLI/model
isn't usable in the current environment.

Reads: madmom (optional, see pyproject [tool.uv.sources]), librosa, scipy;
read by: sheetsage.infer
"""

import logging
import math
import shutil
import tempfile
import warnings

try:
    from scipy.io.wavfile import write as wavwrite
except ImportError as exc:  # pragma: no cover - dependency guardrail
    raise ImportError("scipy is required for beat tracking") from exc

from .utils import run_cmd_sync

# Monkey-patch numpy for madmom compatibility (Cython extensions need np.int)
try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - dependency guardrail
    raise ImportError("numpy is required for beat tracking") from exc

if not hasattr(np, "int"):
    np.int = int
    np.float = float
    np.complex = complex

# Note: madmom downbeats.py line 287 has been patched directly in the installed package
# to fix numpy 1.26+ incompatibility with inhomogeneous array shapes


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


def madmom(sr, audio, beats_per_bar=None, beats_per_minute_hint=None):
    if beats_per_minute_hint is not None and beats_per_minute_hint < 0:
        raise ValueError()

    beats_per_bar_normalized = _normalize_beats_per_bar(beats_per_bar)

    # Check for DBNDownBeatTracker command
    if shutil.which("DBNDownBeatTracker"):
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                wavwrite(f.name, sr, audio)
                beats_per_bar_arg = ""
                if beats_per_bar_normalized is not None:
                    if isinstance(beats_per_bar_normalized, list):
                        beats_per_bar_str = ",".join([str(b) for b in beats_per_bar_normalized])
                    else:
                        beats_per_bar_str = str(beats_per_bar_normalized)
                    beats_per_bar_arg = f"--beats_per_bar {beats_per_bar_str}"
                beats_per_minute_arg = ""
                if beats_per_minute_hint is not None:
                    min_bpm = beats_per_minute_hint * math.pow(2, -0.5)
                    max_bpm = beats_per_minute_hint * math.pow(2, 0.5)
                    beats_per_minute_arg = f"--min_bpm {min_bpm} --max_bpm {max_bpm}"
                result, stdout, stderr = run_cmd_sync(
                    f"DBNDownBeatTracker {beats_per_bar_arg} {beats_per_minute_arg} single {f.name}"
                )
                if result == 0:
                    dbts = []
                    bts = []
                    for line in stdout.splitlines():
                        t, p = line.split()
                        t = float(t)
                        if int(p) == 1:
                            dbts.append(t)
                        else:
                            bts.append(t)

                    assert all(abs(t - (round(t * 100) / 100)) < 1e-8 for t in bts + dbts)
                    dbts = [round(t * 100) for t in dbts]
                    bts = [round(t * 100) for t in bts]

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
                            beats_this_bar += len(
                                [t for t in bts if t > dbts[i] and t < dbts[i + 1]]
                            )
                            if detected_beats_per_bar is None:
                                detected_beats_per_bar = beats_this_bar
                            assert beats_this_bar == detected_beats_per_bar
                        assert (
                            detected_beats_per_bar is None
                            or len(partial_head) < detected_beats_per_bar
                        )
                        assert (
                            detected_beats_per_bar is None
                            or len(partial_tail) < detected_beats_per_bar
                        )
                        if (
                            beats_per_bar_normalized is not None
                            and detected_beats_per_bar is not None
                        ):
                            if isinstance(beats_per_bar_normalized, list):
                                assert detected_beats_per_bar in beats_per_bar_normalized
                            else:
                                assert detected_beats_per_bar == beats_per_bar_normalized

                    return first_downbeat, detected_beats_per_bar, [t / 100 for t in merged]

                logging.warning(
                    "DBNDownBeatTracker exited with status %s. stderr: %s",
                    result,
                    stderr.strip(),
                )
        except (
            OSError,
            ValueError,
            RuntimeError,
            AssertionError,
        ) as exc:  # pragma: no cover - defensive logging
            logging.warning(
                "DBNDownBeatTracker failed (%s). Falling back to librosa beat tracker.",
                exc,
            )
    else:
        logging.info("DBNDownBeatTracker not found. Using librosa beat tracker fallback.")

    return _librosa_fallback(
        sr,
        audio,
        beats_per_bar=beats_per_bar_normalized,
        beats_per_minute_hint=beats_per_minute_hint,
    )
