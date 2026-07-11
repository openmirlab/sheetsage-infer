"""Pipeline enums and module-level constants shared by `sheetsage.pipeline.steps`.

Defines the four small `Enum`s that tag the pipeline's inputs/tasks/models/status
(`InputFeats`, `Task`, `Model`, `Status`) and the frozen lookup tables/constants
(frame rates, dims, chunking edges, harmony vocab, etc.) that `steps.py` and
`sheetsage.infer` read from.

Reads: (leaf module: stdlib `enum` only); read by: sheetsage.pipeline.steps,
sheetsage.infer
"""

from enum import Enum


class InputFeats(Enum):
    HANDCRAFTED = 0
    JUKEBOX = 1


class Task(Enum):
    MELODY = 0
    HARMONY = 1


class Model(Enum):
    LINEAR = 0
    TRANSFORMER = 1


class Status(Enum):
    FETCHING_AUDIO = 0
    DETECTING_BEATS = 1
    EXTRACTING_FEATURES = 2
    TRANSCRIBING = 3
    FORMATTING = 4
    DONE = 5


_INPUT_TO_FRAME_RATE = {
    InputFeats.HANDCRAFTED: 16000 / 512,
    InputFeats.JUKEBOX: 44100 / 128,
}
_INPUT_TO_DIM = {
    InputFeats.HANDCRAFTED: 229,
    InputFeats.JUKEBOX: 4800,
}
_JUKEBOX_CHUNK_DURATION_EDGE = 23.75
_TERTIARIES_PER_BEAT = 4
_MELODY_PITCH_MIN = 21
_HARMONY_FAMILIES = ["", "m", "m7", "7", "maj7", "sus", "dim", "aug"]
_FAMILY_TO_INTERVALS = {
    "": (4, 3),
    "m": (3, 4),
    "m7": (3, 4, 3),
    "7": (4, 3, 3),
    "maj7": (4, 3, 4),
    "sus": (5, 2),
    "dim": (3, 3),
    "aug": (4, 4),
}
_TASK_TO_VOCAB_SIZE = {Task.MELODY: 89, Task.HARMONY: 97}
_MAX_TERTIARIES_PER_CHUNK = 384
