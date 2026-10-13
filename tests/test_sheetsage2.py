"""Offline contracts for the experimental SheetSage2 adapter."""

import builtins
import subprocess
import sys
import types

import pytest

from sheetsage.sheetsage2 import SheetSage2Session


class FakeModel:
    def __init__(self):
        self.transcriptions = []
        self.eval_calls = 0
        self.devices = []

    def eval(self):
        self.eval_calls += 1
        return self

    def to(self, device):
        self.devices.append(device)
        return self

    def transcribe(self, audio, **options):
        result = {"audio": audio, "options": options}
        self.transcriptions.append(result)
        return result


def _install_backend(monkeypatch, model=None, calls=None):
    model = model or FakeModel()
    calls = calls if calls is not None else []
    for name in ("mir_eval", "safetensors", "torchaudio"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    transformers = types.ModuleType("transformers")

    class AutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.append((args, kwargs))
            return model

    transformers.AutoModel = AutoModel
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr("sheetsage.device.resolve_device", lambda device: device)
    return model, calls


def test_load_is_lazy_idempotent_and_two_inferences_reuse_model(monkeypatch):
    model, calls = _install_backend(monkeypatch)
    prepares = []
    monkeypatch.setattr("sheetsage.sheetsage2_checkpoints.CheckpointStore.prepare",
                        lambda self, **kwargs: prepares.append(kwargs) or {
                            "sheetsage2": "/cache/adapter", "mert2": "/cache/mert"})
    session = SheetSage2Session(device="cpu", melody_only=True)
    assert session.status == "new"
    assert calls == []
    assert session.load() is session
    assert session.load() is session
    assert session.status == "ready"
    assert len(calls) == len(prepares) == 1
    assert calls[0] == (("/cache/adapter",), {
        "base_model_path": "/cache/mert", "trust_remote_code": True,
        "local_files_only": True,
    })
    assert prepares == [{"local_files_only": False}]
    assert model.eval_calls == 1
    assert model.devices == ["cpu"]
    first = session.infer("one.wav")
    second = session.infer("two.wav", melody_only=False, export_logits=True)
    assert first == {"audio": "one.wav", "options": {"melody_only": True}}
    assert second == {"audio": "two.wav", "options": {
        "melody_only": False, "export_logits": True,
    }}
    assert len(calls) == 1


def test_infer_passes_result_and_options_without_changing_identity(monkeypatch):
    model, _ = _install_backend(monkeypatch)
    monkeypatch.setattr("sheetsage.sheetsage2_checkpoints.CheckpointStore.prepare",
                        lambda self, **kwargs: {"sheetsage2": "/a", "mert2": "/m"})
    defaults = {"melody_only": True, "export_logits": False}
    session = SheetSage2Session(device="cpu", **defaults).load()
    audio = object()
    result = object()
    captured = {}

    def transcribe(got_audio, **options):
        captured.update(audio=got_audio, options=options)
        return result

    model.transcribe = transcribe
    assert session.infer(audio, melody_only=False, output_dir="exports") is result
    assert captured["audio"] is audio
    assert captured["options"] == {
        "melody_only": False, "export_logits": False, "output_dir": "exports"
    }
    assert defaults == {"melody_only": True, "export_logits": False}


@pytest.mark.parametrize("status", ["new", "failed", "released", "closed"])
def test_infer_requires_ready_state(status):
    session = SheetSage2Session()
    session._status = status
    with pytest.raises(RuntimeError, match="not ready"):
        session.infer("unused")


@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt])
def test_load_failure_clears_state_and_retry_succeeds(monkeypatch, failure_type):
    model = FakeModel()
    attempts = []
    _install_backend(monkeypatch, model=model)

    def prepare(self, **kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise failure_type("temporary failure")
        return {"sheetsage2": "/a", "mert2": "/m"}

    monkeypatch.setattr("sheetsage.sheetsage2_checkpoints.CheckpointStore.prepare", prepare)
    session = SheetSage2Session(device="cpu")
    with pytest.raises(failure_type):
        session.load()
    assert session.status == "failed"
    assert session._model is None
    assert session._resolved_device is None
    with pytest.raises(RuntimeError, match="not ready"):
        session.infer("unused")
    assert session.load() is session
    assert session.status == "ready"
    assert session._model is model
    assert len(attempts) == 2


def test_release_allows_reload_and_close_is_terminal_and_context_exit_closes(monkeypatch):
    model, calls = _install_backend(monkeypatch)
    monkeypatch.setattr("sheetsage.sheetsage2_checkpoints.CheckpointStore.prepare",
                        lambda self, **kwargs: {"sheetsage2": "/a", "mert2": "/m"})
    session = SheetSage2Session(device="cpu").load()
    session.release()
    assert session.status == "released"
    assert session._model is None
    session.load()
    assert session._model is model
    assert len(calls) == 2
    with pytest.raises(ValueError, match="body"):
        with session as active:
            assert active is session
            raise ValueError("body")
    assert session.status == "closed"
    assert session._model is None
    with pytest.raises(RuntimeError, match="closed"):
        session.load()
    session.release()
    assert session.status == "closed"


def test_missing_optional_dependency_has_actionable_install_hint(monkeypatch):
    real_import = builtins.__import__
    monkeypatch.setattr("sheetsage.device.resolve_device", lambda device: device)
    for name in ("mir_eval", "safetensors", "torchaudio"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    def missing_transformers(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("missing transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_transformers)
    session = SheetSage2Session()
    with pytest.raises(ImportError, match=r'pip install .*sheetsage-infer\[sheetsage2\]'):
        session.load()
    assert session.status == "failed"
    assert session._model is None


def test_public_root_import_does_not_import_optional_model_stack():
    script = (
        "import sys, sheetsage; "
        "assert not any(name == 'torch' or name.startswith(('transformers', 'torchaudio')) "
        "for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_incompatible_torchaudio_binary_has_actionable_error(monkeypatch):
    real_import = builtins.__import__
    monkeypatch.setattr("sheetsage.device.resolve_device", lambda device: device)
    for name in ("mir_eval", "safetensors"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    def incompatible_audio(name, *args, **kwargs):
        if name == "torchaudio":
            raise OSError("libcudart.so.13: cannot open shared object file")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", incompatible_audio)
    session = SheetSage2Session(device="cpu")
    with pytest.raises(ImportError, match="same CPU or CUDA wheel index") as error:
        session.load()
    assert isinstance(error.value.__cause__, OSError)
    assert session.status == "failed"


def test_model_move_failure_cleans_owned_state_and_can_retry(monkeypatch):
    class FailOnceModel(FakeModel):
        fail = True

        def to(self, device):
            if self.fail:
                self.fail = False
                raise RuntimeError("device move failed")
            return super().to(device)

    model = FailOnceModel()
    _install_backend(monkeypatch, model=model)
    monkeypatch.setattr("sheetsage.sheetsage2_checkpoints.CheckpointStore.prepare",
                        lambda self, **kwargs: {"sheetsage2": "/a", "mert2": "/m"})
    session = SheetSage2Session(device="cpu")
    with pytest.raises(RuntimeError, match="device move"):
        session.load()
    assert session.status == "failed"
    assert session._model is None
    assert session._resolved_device is None
    assert session.load() is session
    assert session.status == "ready"


@pytest.mark.parametrize("option", ["render_audio", "render_score"])
def test_unsupported_rendering_is_rejected_before_upstream_call(monkeypatch, option):
    model, _ = _install_backend(monkeypatch)
    monkeypatch.setattr("sheetsage.sheetsage2_checkpoints.CheckpointStore.prepare",
                        lambda self, **kwargs: {"sheetsage2": "/a", "mert2": "/m"})
    session = SheetSage2Session(device="cpu").load()
    with pytest.raises(ValueError, match="rendering is not supported"):
        session.infer("unused", **{option: True})
    assert model.transcriptions == []


def test_one_shot_closes_on_success_and_inference_error_and_sessions_are_independent(monkeypatch):
    import importlib

    adapter = importlib.import_module("sheetsage.sheetsage2")
    models = []
    loader_calls = []

    class TrackingModel(FakeModel):
        def transcribe(self, audio, **options):
            if audio == "bad":
                raise ValueError("upstream failed")
            return super().transcribe(audio, **options)

    def loader(*args, **kwargs):
        loader_calls.append((args, kwargs))
        model = TrackingModel()
        models.append(model)
        return model

    monkeypatch.setattr("sheetsage.device.resolve_device", lambda device: device)
    monkeypatch.setattr("sheetsage.sheetsage2_checkpoints.CheckpointStore.prepare",
                        lambda self, **kwargs: {"sheetsage2": "/a", "mert2": "/m"})
    # Install the optional imports without replacing the real torch module.
    for name in ("mir_eval", "safetensors", "torchaudio"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    transformers = types.ModuleType("transformers")
    transformers.AutoModel = types.SimpleNamespace(from_pretrained=loader)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    closed = []
    original_close = adapter.SheetSage2Session.close

    def recording_close(session):
        original_close(session)
        closed.append(session.status)

    monkeypatch.setattr(adapter.SheetSage2Session, "close", recording_close)
    assert adapter.sheetsage2("good", device="cpu")["audio"] == "good"
    with pytest.raises(ValueError, match="upstream failed"):
        adapter.sheetsage2("bad", device="cpu")
    assert len(loader_calls) == 2
    assert len(models) == 2 and models[0] is not models[1]
    assert closed == ["closed", "closed"]


def test_requested_mps_and_unavailable_cuda_fail_before_checkpoint_prepare(monkeypatch):
    import torch

    prepares = []
    monkeypatch.setattr("sheetsage.sheetsage2_checkpoints.CheckpointStore.prepare",
                        lambda self, **kwargs: prepares.append(kwargs))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    session = SheetSage2Session(device="mps")
    with pytest.raises(ValueError, match="Unsupported device"):
        session.load()
    session = SheetSage2Session(device="cuda")
    with pytest.raises(RuntimeError, match="CUDA was explicitly requested"):
        session.load()
    assert prepares == []
