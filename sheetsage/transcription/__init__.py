"""Shared transcription facade with faithful native results available alongside it.

Reads: api, options, result; read by: sheetsage and public callers.
"""

from .api import SheetSageTranscriber, TranscriptionCapabilities, transcribe
from .options import SheetSage1Options, SheetSage2Options
from .result import (
                     BeatEvent,
                     ChordInterval,
                     LabelInterval,
                     MeterInterval,
                     NoteEvent,
                     OutputStatus,
                     TranscriptionResult,
                     result_semantics,
)

__all__ = ["SheetSageTranscriber", "transcribe", "SheetSage1Options", "SheetSage2Options",
           "TranscriptionCapabilities", "TranscriptionResult", "NoteEvent", "LabelInterval",
           "ChordInterval", "BeatEvent", "MeterInterval", "OutputStatus", "result_semantics"]
