import math
import shutil
import sys
import tempfile

from scipy.io.wavfile import write as wavwrite

from .utils import run_cmd_sync

# Monkey-patch numpy for madmom compatibility (Cython extensions need np.int)
import numpy as np
if not hasattr(np, 'int'):
    np.int = int
    np.float = float
    np.complex = complex

# Note: madmom downbeats.py line 287 has been patched directly in the installed package
# to fix numpy 1.26+ incompatibility with inhomogeneous array shapes


def madmom(sr, audio, beats_per_bar=None, beats_per_minute_hint=None):
    if beats_per_minute_hint is not None and beats_per_minute_hint < 0:
        raise ValueError()
    
    # Check for DBNDownBeatTracker command
    if not shutil.which("DBNDownBeatTracker"):
        raise FileNotFoundError(
            "Command 'DBNDownBeatTracker' not found in PATH. "
            "This is provided by the madmom package. "
            "Install with: pip install madmom"
        )

    # Run madmom
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        wavwrite(f.name, sr, audio)
        beats_per_bar_arg = ""
        if beats_per_bar is not None:
            if isinstance(beats_per_bar, list):
                beats_per_bar_str = ",".join([str(b) for b in beats_per_bar])
            else:
                beats_per_bar_str = str(beats_per_bar)
            beats_per_bar_arg = f"--beats_per_bar {beats_per_bar_str}"
        beats_per_minute_arg = ""
        if beats_per_minute_hint is not None:
            min_bpm = beats_per_minute_hint * math.pow(2, -0.5)
            max_bpm = beats_per_minute_hint * math.pow(2, 0.5)
            beats_per_minute_arg = f"--min_bpm {min_bpm} --max_bpm {max_bpm}"
        result, stdout, stderr = run_cmd_sync(
            f"DBNDownBeatTracker {beats_per_bar_arg} {beats_per_minute_arg} single {f.name}"
        )
        if result != 0:
            raise Exception(stderr)

    # Parse output
    dbts = []
    bts = []
    for l in stdout.splitlines():
        t, p = l.split()
        t = float(t)
        if int(p) == 1:
            dbts.append(t)
        else:
            bts.append(t)

    # Make sure 100Hz and convert to discrete
    assert all(abs(t - (round(t * 100) / 100)) < 1e-8 for t in bts + dbts)
    dbts = [round(t * 100) for t in dbts]
    bts = [round(t * 100) for t in bts]

    # Sanity check (assumptions about madmom output)
    assert all(t >= 0 for t in dbts + bts)
    assert sorted(dbts) == dbts
    assert sorted(bts) == bts
    assert len(set(dbts)) == len(dbts)
    assert len(set(bts)) == len(bts)
    assert len(set(dbts).intersection(set(bts))) == 0

    # Detect beats per bar
    # NOTE: This logic asserts that madmom does *not* change the time signature
    first_downbeat = None
    detected_beats_per_bar = None
    merged = sorted(dbts + bts)
    if len(dbts) > 0:
        first_downbeat = merged.index(dbts[0])
        partial_head = [t for t in bts if t < dbts[0]]
        partial_tail = [t for t in bts if t > dbts[-1]]
        detected_beats_per_bar = None
        for i in range(len(dbts) - 1):
            beats_this_bar = 1
            beats_this_bar += len([t for t in bts if t > dbts[i] and t < dbts[i + 1]])
            if detected_beats_per_bar is None:
                detected_beats_per_bar = beats_this_bar
            assert beats_this_bar == detected_beats_per_bar
        assert (
            detected_beats_per_bar is None or len(partial_head) < detected_beats_per_bar
        )
        assert (
            detected_beats_per_bar is None or len(partial_tail) < detected_beats_per_bar
        )
        if beats_per_bar is not None and detected_beats_per_bar is not None:
            if isinstance(beats_per_bar, list):
                assert detected_beats_per_bar in beats_per_bar
            else:
                assert detected_beats_per_bar == beats_per_bar

    return first_downbeat, detected_beats_per_bar, [t / 100 for t in merged]
