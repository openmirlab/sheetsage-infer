"""Pipeline package: re-exports everything `types.py` and `steps.py` define.

Lets `sheetsage.infer` (and, transitively, anything that used to import these
names from `sheetsage.infer`) do `from .pipeline import <name>` for any enum,
constant, or private `_*` helper without knowing which of the two submodules
actually defines it.

Reads: .types, .steps; read by: sheetsage.infer
"""

from .steps import (
    _beat_tracking_with_hints,
    _closest_idx,
    _extract_features,
    _format_lead_sheet,
    _init_extractor,
    _init_model,
    load_components,
    _split_into_chunks,
    _transcribe_chunks,
)
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
    Status,
    Task,
)

__all__ = [
    "InputFeats",
    "Model",
    "Status",
    "Task",
    "_FAMILY_TO_INTERVALS",
    "_HARMONY_FAMILIES",
    "_INPUT_TO_DIM",
    "_INPUT_TO_FRAME_RATE",
    "_JUKEBOX_CHUNK_DURATION_EDGE",
    "_MAX_TERTIARIES_PER_CHUNK",
    "_MELODY_PITCH_MIN",
    "_TASK_TO_VOCAB_SIZE",
    "_TERTIARIES_PER_BEAT",
    "_beat_tracking_with_hints",
    "_closest_idx",
    "_extract_features",
    "_format_lead_sheet",
    "_init_extractor",
    "_init_model",
    "load_components",
    "_split_into_chunks",
    "_transcribe_chunks",
]
