"""Shared transcription entry point over independent native model sessions.

Reads: options, result, adapters, native sessions; read by: public callers.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from .options import SheetSage1Options, SheetSage2Options


@dataclass(frozen=True)
class TranscriptionCapabilities:
    """Model support, independent of tasks selected for a particular inference."""

    notes: bool = True
    chords: bool = True
    beats: bool = True
    keys: bool = True
    meters: bool = True
    structure: bool = False
    native_formats: tuple[str, ...] = ("midi", "lilypond")


class SheetSageTranscriber:
    """Own one backend and return normalized results while retaining native output.

    Construction is lazy. Call load() before infer(), or use a context manager.
    Per-call options replace the complete option object; session configuration
    must match construction. A transcriber supports sequential inference calls.
    """

    def __init__(self, model="sheetsage1", *, device="auto", options=None,
                 require_structure=False):
        if model not in {"sheetsage1", "sheetsage2"}:
            raise ValueError("model must be sheetsage1 or sheetsage2")
        if not isinstance(require_structure, bool):
            raise TypeError("require_structure must be a bool")
        self._model = model
        self._device = device
        self._require_structure = require_structure
        self._options = self._validate_options(options)
        if model == "sheetsage1":
            from ..session import SheetSageSession

            self._session = SheetSageSession(device=device, **self._options.session_kwargs())
        else:
            from ..sheetsage2 import SheetSage2Session

            self._session = SheetSage2Session(device=device, **self._options.session_kwargs())

    @property
    def model(self):
        return self._model

    @property
    def options(self):
        return self._options

    @property
    def capabilities(self):
        if self.model == "sheetsage2":
            return TranscriptionCapabilities(structure=True, native_formats=("midi", "abc", "lab", "json", "tsv"))
        return TranscriptionCapabilities()

    @property
    def semantics(self):
        from .result import result_semantics

        return result_semantics(self.model)

    @property
    def status(self):
        return self._session.status

    def _validate_options(self, options):
        expected = SheetSage1Options if self.model == "sheetsage1" else SheetSage2Options
        if options is None:
            options = expected()
        if not isinstance(options, expected):
            raise TypeError(f"{self.model} requires {expected.__name__}")
        if self._require_structure:
            if self.model == "sheetsage1":
                raise ValueError("sheetsage1 does not support structure; select sheetsage2")
            if "structure" not in options.prompts:
                raise ValueError("require_structure=True requires the structure prompt")
        return options

    def load(self):
        self._session.load()
        return self

    def infer(self, audio, *, options=None):
        if self.status != "ready":
            raise RuntimeError("transcriber is not ready; call load() first")
        selected = self._options if options is None else self._validate_options(options)
        if selected.session_kwargs() != self._options.session_kwargs():
            raise ValueError("Backend session configuration is fixed at construction")
        _validate_audio(audio, self.model, selected)
        from . import adapters

        kwargs = selected.inference_kwargs()
        native = self._session.infer(audio, **kwargs)
        adapter = adapters.from_sheetsage1 if self.model == "sheetsage1" else adapters.from_sheetsage2
        provenance = ({"provenance": self._session.checkpoint_provenance()}
                      if self.model == "sheetsage2" else {})
        return adapter(native, {**selected.session_kwargs(), **kwargs,
                                "requested_device": self._device}, **provenance)

    def release(self):
        self._session.release()

    def close(self):
        self._session.close()

    def cache_info(self):
        return self._session.cache_info()

    def __enter__(self):
        return self.load()

    def __exit__(self, exc_type, exc, tb):
        self.close()


def _validate_audio(audio, model, options):
    encoded = isinstance(audio, (str, Path, bytes))
    if model == "sheetsage1":
        if not encoded:
            raise TypeError("sheetsage1 accepts a path, URL, or encoded audio bytes")
        return
    if isinstance(audio, str) and "://" in audio:
        raise ValueError("sheetsage2 accepts local paths, not URLs")
    encoded = encoded or isinstance(audio, (bytearray, memoryview)) or callable(getattr(audio, "read", None))
    if encoded:
        if options.sampling_rate is not None:
            raise ValueError("sampling_rate applies only to waveform arrays/tensors, not encoded audio")
        if options.preset == "paper":
            import torchaudio

            if not callable(getattr(torchaudio, "info", None)):
                raise RuntimeError(
                    "The upstream paper file decoder requires torchaudio.info(), absent in this "
                    "torchaudio version. Supply a decoded NumPy/Torch waveform with sampling_rate, "
                    "or select preset='default'."
                )
        return
    import numpy as np

    torch = sys.modules.get("torch")
    if not isinstance(audio, np.ndarray) and not (torch is not None and isinstance(audio, torch.Tensor)):
        raise TypeError("sheetsage2 accepts local paths, encoded bytes, binary streams, or NumPy/Torch waveforms")
    if options.sampling_rate is None:
        raise ValueError("Provide sampling_rate in SheetSage2Options for a waveform")


def transcribe(audio, *, model="sheetsage1", device="auto", options=None, require_structure=False):
    """Transcribe once with a fresh backend and close it after success or failure."""
    session = SheetSageTranscriber(model, device=device, options=options,
                                  require_structure=require_structure)
    try:
        session.load()
        return session.infer(audio)
    finally:
        session.close()


__all__ = ["SheetSageTranscriber", "TranscriptionCapabilities", "transcribe"]
