"""Session state and owned-reference contracts, without model downloads.

Reads: the public session and controlled device/component/inference seams.
Real CPU output parity is verified separately against the committed baseline.
"""

import gc
import weakref

import pytest
import torch

from sheetsage import SheetSageSession


def test_construction_is_lazy_and_successful_load_is_idempotent(monkeypatch):
    calls = []
    components = object()
    monkeypatch.setattr("sheetsage.device.resolve_device",
                        lambda device: calls.append(("resolve", device)) or torch.device("cpu"))
    monkeypatch.setattr("sheetsage.pipeline.load_components",
                        lambda *args: calls.append(("build", args)) or components)
    session = SheetSageSession(device="cpu")
    assert session.status == "new"
    assert calls == []
    assert session.load() is session
    assert session.load() is session
    assert session.status == "ready"
    assert [kind for kind, _ in calls] == ["resolve", "build"]


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("stage", ["device", "components"])
def test_failed_or_interrupted_load_cleans_owned_state_and_allows_retry(monkeypatch, error_type, stage):
    failure = error_type("fail once")
    attempts = []
    components = object()

    def fail_once(value):
        attempts.append(value)
        if len(attempts) == 1:
            raise failure

    def resolve(device):
        if stage == "device":
            fail_once(device)
        return torch.device("cpu")

    def build(*args):
        if stage == "components":
            fail_once(args)
        return components

    monkeypatch.setattr("sheetsage.device.resolve_device", resolve)
    monkeypatch.setattr("sheetsage.pipeline.load_components", build)
    session = SheetSageSession(device="cpu")
    with pytest.raises(error_type) as caught:
        session.load()
    assert caught.value is failure
    assert session.status == "failed"
    assert session._components is None
    assert session._resolved_device is None
    with pytest.raises(RuntimeError, match="not ready"):
        session.infer("unused")
    assert session.load() is session
    assert session.status == "ready"
    assert session._components is components
    assert session.load() is session
    assert len(attempts) == 2


@pytest.mark.parametrize("operation", ["release", "close"])
def test_owned_components_are_retained_until_release_or_close(monkeypatch, tmp_path, operation):
    class Components:
        pass

    references = []

    def build(*args):
        component = Components()
        references.append(weakref.ref(component))
        return component

    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"existing cached weights")
    monkeypatch.setattr("sheetsage.pipeline.load_components", build)
    monkeypatch.setattr("sheetsage.assets.resolve_asset_path", lambda tag: checkpoint)
    session = SheetSageSession(device="cpu").load()
    gc.collect()
    assert references[0]() is session._components
    assert session.cache_info()["assets"]
    getattr(session, operation)()
    gc.collect()
    assert references[0]() is None
    assert session._components is None
    assert session._resolved_device is None
    assert checkpoint.read_bytes() == b"existing cached weights"
    expected = "released" if operation == "release" else "closed"
    assert session.status == expected
    getattr(session, operation)()
    assert session.status == expected


@pytest.mark.parametrize("loaded", [False, True])
def test_close_is_terminal_before_or_after_load(monkeypatch, loaded):
    builds = []
    monkeypatch.setattr("sheetsage.pipeline.load_components",
                        lambda *args: builds.append(object()) or builds[-1])
    session = SheetSageSession(device="cpu")
    if loaded:
        session.load()
    session.close()
    session.close()
    session.release()
    assert session.status == "closed"
    assert session._components is None
    assert session._resolved_device is None
    for attempt in (session.load, session.__enter__):
        with pytest.raises(RuntimeError, match="closed"):
            attempt()
    with pytest.raises(RuntimeError, match="not ready"):
        session.infer("unused")
    assert len(builds) == int(loaded)


@pytest.mark.parametrize("body_error", [False, True])
def test_context_exit_closes_even_when_body_raises(monkeypatch, body_error):
    monkeypatch.setattr("sheetsage.pipeline.load_components", lambda *args: object())
    session = SheetSageSession(device="cpu")
    failure = ValueError("body failed")
    try:
        with session as active:
            assert active is session
            assert active.status == "ready"
            if body_error:
                raise failure
    except ValueError as error:
        assert body_error
        assert error is failure
    else:
        assert not body_error
    assert session.status == "closed"
    assert session._components is None
    assert session._resolved_device is None


def test_release_before_load_is_visible_and_reloadable(monkeypatch):
    builds = []
    monkeypatch.setattr("sheetsage.pipeline.load_components",
                        lambda *args: builds.append(object()) or builds[-1])
    session = SheetSageSession(device="cpu")
    with pytest.raises(RuntimeError, match="not ready"):
        session.infer("unused")
    session.release()
    session.release()
    assert session.status == "released"
    with pytest.raises(RuntimeError, match="not ready"):
        session.infer("unused")
    assert builds == []
    session.load()
    assert session.status == "ready"
    assert len(builds) == 1
