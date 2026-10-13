"""Experimental SheetSage2 transcription with an independent model lifetime.

Loads checksum-verified upstream code at runtime and preserves its rich output
without converting it to SheetSage1's LeadSheet. Optional dependencies remain
lazy; no upstream source or weights are distributed here.
Reads: sheetsage2_checkpoints, device, transformers; read by: public callers.
"""

from __future__ import annotations


class SheetSage2Session:
    """Own a pinned SheetSage2 model; call load() before infer().

    Supports CPU and CUDA. Upstream Python code executes from verified local
    files during load. Explicit model_path/base_model_path directories must
    contain the pinned unmerged adapter/encoder files; they are never modified.
    Inference options are passed to upstream transcribe(), except rendering.
    A session is intended for sequential calls, not concurrent inference.
    """

    def __init__(self, *, device="auto", cache_dir=None, model_path=None,
                 base_model_path=None, checkpoint_config=None, local_files_only=False,
                 **defaults):
        from .sheetsage2_checkpoints import CheckpointStore

        self.device = device
        self.local_files_only = local_files_only
        self.defaults = dict(defaults)
        self._store = CheckpointStore(
            cache_dir=cache_dir, model_path=model_path, base_model_path=base_model_path,
            checkpoint_config=checkpoint_config,
        )
        self._model = None
        self._resolved_device = None
        self._status = "new"

    @property
    def status(self):
        return self._status

    def load(self):
        """Download/verify once and construct the model; failures can be retried."""
        if self._status == "closed":
            raise RuntimeError("session is closed")
        if self._status == "ready":
            return self
        try:
            from .device import resolve_device

            self._resolved_device = resolve_device(self.device)
            try:
                import mir_eval  # noqa: F401 — required by upstream MIDI chord spelling
                import safetensors  # noqa: F401
                import torchaudio  # noqa: F401
                from transformers import AutoModel
            except ImportError as exc:
                raise ImportError(
                    'SheetSage2 requires the optional backend: pip install ".[sheetsage2]" '
                    'from this checkout to install sheetsage-infer[sheetsage2].'
                ) from exc
            except OSError as exc:
                raise ImportError(
                    "SheetSage2 could not load the Torch/torchaudio native libraries. "
                    "Install compatible builds from the same CPU or CUDA wheel index; "
                    "see the README's SheetSage2 installation instructions."
                ) from exc
            directories = self._store.prepare(local_files_only=self.local_files_only)
            # Both bundles are verified and local. Parent downloads and mutable
            # remote-code resolution are disabled, including in upstream loaders.
            model = AutoModel.from_pretrained(
                str(directories["sheetsage2"]),
                base_model_path=str(directories["mert2"]),
                trust_remote_code=True, local_files_only=True,
            )
            self._model = model.eval().to(self._resolved_device)
        except BaseException:
            self._model = None
            self._resolved_device = None
            self._status = "failed"
            raise
        self._status = "ready"
        return self

    def infer(self, audio, **kwargs):
        """Return upstream ABC, MIDI, events, annotations and requested tensors.

        Input may be a local path, encoded bytes, a binary stream, or a waveform.
        Arrays/tensors require sampling_rate and use channels-first layout.
        output_dir saves upstream artifacts; omit it to retain results in memory.
        """
        if self._status != "ready":
            raise RuntimeError("session is not ready; call load() first")
        options = dict(self.defaults)
        options.update(kwargs)
        if options.get("render_audio") or options.get("render_score"):
            raise ValueError("SheetSage2 rendering is not supported; export ABC/MIDI with output_dir instead")
        return self._model.transcribe(audio, **options)

    def release(self):
        """Drop model references; retain checkpoint files and allow reloading."""
        self._model = None
        self._resolved_device = None
        if self._status != "closed":
            self._status = "released"

    def close(self):
        """Terminal, idempotent release, also used by context-manager exit."""
        self.release()
        self._status = "closed"

    def cache_info(self):
        return self._store.cache_info()

    def checkpoint_provenance(self):
        """Report checkpoint identities captured at construction without loading."""
        return self._store.provenance()

    def __enter__(self):
        return self.load()

    def __exit__(self, exc_type, exc, tb):
        self.close()


def sheetsage2(audio, *, device="auto", cache_dir=None, model_path=None,
               base_model_path=None, checkpoint_config=None, local_files_only=False,
               **kwargs):
    """Transcribe once with a fresh session; use SheetSage2Session for reuse."""
    with SheetSage2Session(
        device=device, cache_dir=cache_dir, model_path=model_path,
        base_model_path=base_model_path, checkpoint_config=checkpoint_config,
        local_files_only=local_files_only,
    ) as session:
        return session.infer(audio, **kwargs)


__all__ = ["SheetSage2Session", "sheetsage2"]
