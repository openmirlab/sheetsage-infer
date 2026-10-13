"""Immutable backend options for the shared transcription API.

Reads: native session/inference contracts; read by: transcription.api and callers.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, fields
from numbers import Real
from os import PathLike
from typing import Any, Literal

_FULL_PROMPTS = (
    "timestamp", "downbeat_meter", "structure", "key", "chord_full", "melody_full",
)
_PROMPTS = (
    "timestamp", "downbeat_meter", "structure", "key", "chord_majmin", "chord_full",
    "melody_vocal", "melody_full",
)


@dataclass(frozen=True)
class SheetSage1Options:
    """Native V1 options; callbacks set to None preserve native defaults."""

    use_jukebox: bool = False
    segment_start_hint: float | None = None
    segment_end_hint: float | None = None
    measures_per_chunk: int = 8
    segment_hints_are_downbeats: bool = False
    beats_per_measure_hint: int | None = None
    beats_per_minute_hint: float | None = None
    detect_melody: bool = True
    detect_harmony: bool = True
    melody_threshold: float | None = None
    harmony_threshold: float | None = None
    beat_detection_padding: float = 15.0
    avoid_chunking_if_possible: bool = True
    legacy_behavior: bool = False
    status_change_callback: Callable | None = None
    return_intermediaries: bool = False
    tqdm: Callable | None = None

    def session_kwargs(self) -> dict[str, Any]:
        return {"use_jukebox": self.use_jukebox}

    def inference_kwargs(self) -> dict[str, Any]:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "use_jukebox"
            and not (field.name in {"status_change_callback", "tqdm"}
                     and getattr(self, field.name) is None)
        }


@dataclass(frozen=True)
class SheetSage2Options:
    """Native V2 options, including immutable session checkpoint configuration.

    native_output_dir saves upstream artifacts, not the shared JSON/MIDI exports.
    Sampling rate applies only to array/tensor inputs. Rendering remains available
    only outside this experimental integration.
    """

    cache_dir: str | PathLike | None = None
    model_path: str | PathLike | None = None
    base_model_path: str | PathLike | None = None
    checkpoint_config: str | PathLike | None = None
    local_files_only: bool = False
    native_output_dir: str | PathLike | None = None
    dtype: Literal["bf16", "fp32"] = "bf16"
    melody_only: bool = False
    prompts: tuple[str, ...] = _FULL_PROMPTS
    sampling_rate: int | None = None
    max_seconds: float | None = None
    preset: Literal["default", "paper"] = "default"
    overlap_seconds: float | None = None
    lookahead_seconds: float | None = None
    progress: Callable | None = None
    export_logits: bool = False
    export_scores: bool = False
    export_embeddings: bool = False
    output_hidden_states: bool = False

    def __post_init__(self):
        if isinstance(self.prompts, str):
            raise TypeError("prompts must be a sequence of task names, not a string")
        prompts = tuple(str(name).strip() for name in self.prompts)
        prompts = tuple(name[2:-2] if name.startswith("<|") and name.endswith("|>")
                        else name for name in prompts)
        unknown = set(prompts) - set(_PROMPTS)
        if unknown:
            raise ValueError(f"Unknown SheetSage2 prompts: {sorted(unknown)}")
        if "timestamp" not in prompts:
            raise ValueError("timestamp prompt is required for timed transcription")
        for alternatives in ({"chord_majmin", "chord_full"}, {"melody_vocal", "melody_full"}):
            if alternatives.issubset(prompts):
                raise ValueError(f"Mutually exclusive prompts: {sorted(alternatives)}")
        object.__setattr__(self, "prompts", tuple(name for name in _PROMPTS if name in prompts))
        if self.dtype not in {"bf16", "fp32"}:
            raise ValueError("dtype must be bf16 or fp32")
        if self.preset not in {"default", "paper"}:
            raise ValueError("preset must be default or paper")
        if not isinstance(self.melody_only, bool):
            raise TypeError("melody_only must be a bool")
        for name in ("sampling_rate", "max_seconds", "overlap_seconds", "lookahead_seconds"):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                    raise ValueError(f"{name} must be finite")
                if value < 0 or (name in {"sampling_rate", "max_seconds"} and value == 0):
                    raise ValueError(f"{name} is outside the supported range")
                if name == "sampling_rate" and value != int(value):
                    raise ValueError("sampling_rate must be a positive integer")
        if self.preset == "paper" and (
            self.overlap_seconds not in (None, 100.0)
            or self.lookahead_seconds not in (None, 0.0)
        ):
            raise ValueError("paper preset fixes overlap_seconds=100 and lookahead_seconds=0")

    def session_kwargs(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in (
            "cache_dir", "model_path", "base_model_path", "checkpoint_config", "local_files_only",
        )}

    def inference_kwargs(self) -> dict[str, Any]:
        session_fields = self.session_kwargs()
        return {
            "output_dir" if field.name == "native_output_dir" else field.name: getattr(self, field.name)
            for field in fields(self) if field.name not in session_fields
        }


__all__ = ["SheetSage1Options", "SheetSage2Options"]
