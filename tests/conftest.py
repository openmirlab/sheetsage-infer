#!/usr/bin/env python3
"""Pytest configuration file with fixtures and custom options."""

import pytest


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests (integration tests with real audio)",
    )
    parser.addoption(
        "--run-jukebox",
        action="store_true",
        default=False,
        help="Run Jukebox tests (requires GPU and Jukebox installed)",
    )
    parser.addoption(
        "--with-lilypond",
        action="store_true",
        default=False,
        help="Run tests requiring LilyPond binary",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (integration tests)")
    config.addinivalue_line("markers", "gpu: marks tests as requiring GPU")
    config.addinivalue_line("markers", "jukebox: marks tests as requiring Jukebox")
    config.addinivalue_line("markers", "lilypond: marks tests as requiring LilyPond binary")


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command-line options."""

    # Skip slow tests unless --run-slow is given
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    # Skip GPU tests unless --run-jukebox is given
    if not config.getoption("--run-jukebox"):
        skip_gpu = pytest.mark.skip(reason="need --run-jukebox option to run")
        for item in items:
            if "gpu" in item.keywords or "jukebox" in item.keywords:
                item.add_marker(skip_gpu)

    # Skip LilyPond tests unless --with-lilypond is given
    if not config.getoption("--with-lilypond"):
        skip_lily = pytest.mark.skip(reason="need --with-lilypond option to run")
        for item in items:
            if "lilypond" in item.keywords:
                item.add_marker(skip_lily)


# Common fixtures
@pytest.fixture(scope="session")
def test_audio_url():
    """Test audio URL for integration tests."""
    return "https://foodgroup.bandcamp.com/track/universe"


@pytest.fixture(scope="session")
def test_params():
    """Common test parameters."""
    return {"segment_start_hint": 69, "segment_end_hint": 88, "beats_per_minute_hint": 76}


@pytest.fixture
def handcrafted_params(test_params):
    """Parameters for handcrafted feature tests."""
    params = test_params.copy()
    params["use_jukebox"] = False
    return params


@pytest.fixture
def jukebox_params(test_params):
    """Parameters for Jukebox feature tests."""
    params = test_params.copy()
    params["use_jukebox"] = True
    return params
