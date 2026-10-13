"""Shared facade contracts exercised without checkpoints or model construction."""

from dataclasses import FrozenInstanceError, replace
from io import BytesIO
from types import ModuleType
import sys

import numpy as np
import pytest

from sheetsage.transcription.api import SheetSageTranscriber, transcribe
from sheetsage.transcription.options import SheetSage1Options, SheetSage2Options


@pytest.fixture
def backends(monkeypatch):
    """Record native ownership and forwarding, isolating conversion tests elsewhere."""
    sessions = []

    class NativeSession:
        def __init__(self, **kwargs):
            self.config = kwargs
            self.status = "new"
            self.calls = []
            self.builds = 0
            self.error = None
            self.load_error = None
            self.result = object()
            sessions.append(self)

        def load(self):
            if self.status == "closed":
                raise RuntimeError("session is closed")
            if self.load_error:
                self.status = "failed"
                raise self.load_error
            if self.status != "ready":
                self.builds += 1
            self.status = "ready"
            return self

        def infer(self, audio, **kwargs):
            self.calls.append((audio, kwargs))
            if self.error:
                raise self.error
            return self.result

        def release(self):
            if self.status != "closed":
                self.status = "released"

        def close(self):
            self.status = "closed"

        def checkpoint_provenance(self):
            return {"manifest_sha256": "original"}

        def cache_info(self):
            return {"cached": True}

    import sheetsage.session
    from importlib import import_module

    monkeypatch.setattr(sheetsage.session, "SheetSageSession", NativeSession)
    monkeypatch.setattr(import_module("sheetsage.sheetsage2"), "SheetSage2Session", NativeSession)
    adapters = ModuleType("sheetsage.transcription.adapters")
    adapters.from_sheetsage1 = lambda native, options: ("sheetsage1", native, options)
    adapters.from_sheetsage2 = lambda native, options, **kwargs: ("sheetsage2", native, options, kwargs)
    monkeypatch.setitem(sys.modules, "sheetsage.transcription.adapters", adapters)
    monkeypatch.setattr(import_module("sheetsage.transcription"), "adapters", adapters, raising=False)
    return sessions


@pytest.mark.parametrize("model", ["sheetsage1", "sheetsage2"])
def test_reuses_backend_and_delegates_lifecycle(backends, model):
    session = SheetSageTranscriber(model, device="cpu")
    native = backends[0]
    assert native.builds == 0
    assert session.status == "new"
    assert session.cache_info() == {"cached": True}
    with pytest.raises(RuntimeError, match="not ready"):
        session.infer(b"audio")
    assert session.load() is session
    assert session.load() is session
    audio = b"encoded audio"
    first = session.infer(audio)
    second = session.infer(audio)
    assert first[0] == model
    assert first[1] is native.result
    assert second[1] is native.result
    assert first[2]["requested_device"] == "cpu"
    assert native.calls[0][0] is audio
    if model == "sheetsage2":
        assert first[3] == {"provenance": {"manifest_sha256": "original"}}
    assert native.builds == 1
    session.release()
    assert session.status == "released"
    session.load()
    assert native.builds == 2
    session.close()
    session.release()
    assert session.status == "closed"
    with pytest.raises(RuntimeError, match="closed"):
        session.load()


def test_capabilities_immutable_and_structure_rejected_before_construction(backends):
    with pytest.raises(ValueError, match="does not support structure"):
        SheetSageTranscriber(require_structure=True)
    with pytest.raises(ValueError, match="structure prompt"):
        SheetSageTranscriber("sheetsage2", require_structure=True,
                             options=SheetSage2Options(prompts=("timestamp",)))
    assert backends == []
    old = SheetSageTranscriber()
    new = SheetSageTranscriber("sheetsage2")
    assert not old.capabilities.structure
    assert new.capabilities.structure
    assert "abc" in new.capabilities.native_formats
    with pytest.raises(FrozenInstanceError):
        new.capabilities.structure = False


def test_structure_only_and_per_call_requirement(backends):
    options = SheetSage2Options(prompts=("timestamp", "structure"))
    with SheetSageTranscriber("sheetsage2", options=options, require_structure=True) as session:
        session.infer(b"audio")
        assert backends[0].calls[-1][1]["prompts"] == ("timestamp", "structure")
        with pytest.raises(ValueError, match="structure prompt"):
            session.infer(b"audio", options=replace(options, prompts=("timestamp",)))
    assert len(backends[0].calls) == 1
    assert session.status == "closed"


@pytest.mark.parametrize("model,wrong", [
    ("sheetsage1", SheetSage2Options()), ("sheetsage2", SheetSage1Options()),
    ("sheetsage2", {"preset": "paper"}),
])
def test_wrong_backend_options_fail_early(backends, model, wrong):
    with pytest.raises(TypeError, match="requires SheetSage"):
        SheetSageTranscriber(model, options=wrong)
    assert backends == []
    with SheetSageTranscriber(model) as session:
        with pytest.raises(TypeError, match="requires SheetSage"):
            session.infer(b"audio", options=wrong)
    assert backends[0].calls == []


@pytest.mark.parametrize("model,options,field,value", [
    ("sheetsage1", SheetSage1Options(), "use_jukebox", True),
    ("sheetsage2", SheetSage2Options(), "cache_dir", "/other"),
    ("sheetsage2", SheetSage2Options(), "model_path", "/other"),
    ("sheetsage2", SheetSage2Options(), "base_model_path", "/other"),
    ("sheetsage2", SheetSage2Options(), "checkpoint_config", "/other"),
    ("sheetsage2", SheetSage2Options(), "local_files_only", True),
])
def test_session_configuration_cannot_change_per_call(backends, model, options, field, value):
    with SheetSageTranscriber(model, options=options) as session:
        with pytest.raises(ValueError, match="fixed at construction"):
            session.infer(b"audio", options=replace(options, **{field: value}))
    assert backends[0].calls == []


def test_native_options_preserve_callbacks_intermediaries_and_artifact_directory(backends, tmp_path):
    def callback(event):
        return None
    options = SheetSage1Options(return_intermediaries=True, status_change_callback=callback,
                               segment_start_hint=3.5, use_jukebox=True)
    with SheetSageTranscriber(options=options) as session:
        result = session.infer(tmp_path / "song.wav", options=replace(options, detect_harmony=False))
        assert result[2]["use_jukebox"] is True
    native = backends[0]
    assert native.config["use_jukebox"] is True
    assert native.calls[0][1]["return_intermediaries"] is True
    assert native.calls[0][1]["status_change_callback"] is callback
    assert native.calls[0][1]["segment_start_hint"] == 3.5
    assert native.calls[0][1]["detect_harmony"] is False
    assert "tqdm" not in native.calls[0][1]
    options2 = SheetSage2Options(native_output_dir=tmp_path, export_logits=True, progress=callback)
    transcribe(b"audio", model="sheetsage2", options=options2)
    forwarded = backends[1].calls[0][1]
    assert forwarded["output_dir"] == tmp_path
    assert forwarded["progress"] is callback
    assert forwarded["export_logits"] is True
    assert "native_output_dir" not in forwarded
    assert "cache_dir" not in forwarded


@pytest.mark.parametrize("stage", ["success", "inference", "load", "adapter"])
def test_one_shot_owns_fresh_sessions_and_closes_on_all_exits(backends, monkeypatch, stage):
    original = SheetSageTranscriber.load
    error = RuntimeError("failed")

    def load(session):
        if stage == "inference":
            backends[-1].error = error
        if stage == "load":
            backends[-1].load_error = error
        return original(session)

    monkeypatch.setattr(SheetSageTranscriber, "load", load)
    if stage == "adapter":
        def fail(*args):
            raise error
        monkeypatch.setattr(sys.modules["sheetsage.transcription.adapters"], "from_sheetsage1", fail)
    for _ in range(2):
        if stage == "success":
            transcribe(b"audio")
        else:
            with pytest.raises(RuntimeError) as caught:
                transcribe(b"audio")
            assert caught.value is error
        assert backends[-1].status == "closed"
    assert len(backends) == 2


def test_input_contract_preserves_objects_and_rejects_conflicting_sample_rate(backends):
    with SheetSageTranscriber() as old:
        old.infer("https://example.test/audio.wav")
        with pytest.raises(TypeError, match="path, URL"):
            old.infer(np.zeros(4096))
    waveform = np.zeros(4096, dtype=np.float32)
    with SheetSageTranscriber("sheetsage2") as new:
        stream = BytesIO(b"audio")
        new.infer(stream)
        assert backends[-1].calls[-1][0] is stream
        with pytest.raises(ValueError, match="not URLs"):
            new.infer("https://example.test/audio.wav")
        with pytest.raises(ValueError, match="Provide sampling_rate"):
            new.infer(waveform)
        options = SheetSage2Options(sampling_rate=24000)
        new.infer(waveform, options=options)
        assert backends[-1].calls[-1][0] is waveform
        with pytest.raises(ValueError, match="not encoded audio"):
            new.infer(b"audio", options=options)
        with pytest.raises(TypeError, match="NumPy/Torch"):
            new.infer([0.0] * 4096, options=options)


@pytest.mark.parametrize("prompts,match", [
    (("structure",), "timestamp"),
    (("timestamp", "chorus"), "Unknown"),
    (("timestamp", "chord_full", "chord_majmin"), "Mutually exclusive"),
    (("timestamp", "melody_full", "melody_vocal"), "Mutually exclusive"),
])
def test_invalid_task_combinations_rejected(prompts, match):
    with pytest.raises(ValueError, match=match):
        SheetSage2Options(prompts=prompts)


def test_options_canonicalize_prompts_and_reject_unknown_parameters():
    options = SheetSage2Options(prompts=["structure", "<|timestamp|>", "structure"])
    assert options.prompts == ("timestamp", "structure")
    with pytest.raises(FrozenInstanceError):
        options.preset = "paper"
    with pytest.raises(TypeError):
        SheetSage2Options(use_jukebox=True)
    with pytest.raises(TypeError):
        SheetSage1Options(preset="paper")
    with pytest.raises(TypeError):
        SheetSage2Options(render_score=True)
    with pytest.raises(ValueError, match="model must"):
        SheetSageTranscriber("unknown")


def test_paper_encoded_input_explains_torchaudio_compatibility(backends, monkeypatch):
    audio_module = ModuleType("torchaudio")
    monkeypatch.setitem(sys.modules, "torchaudio", audio_module)
    with SheetSageTranscriber("sheetsage2", options=SheetSage2Options(preset="paper")) as model:
        with pytest.raises(RuntimeError, match="decoded NumPy/Torch waveform"):
            model.infer(b"audio")
        assert backends[-1].calls == []
        model.infer(np.zeros(4096), options=SheetSage2Options(preset="paper", sampling_rate=24000))
        audio_module.info = lambda *a, **kw: None
        model.infer(b"audio")
        assert len(backends[-1].calls) == 2
