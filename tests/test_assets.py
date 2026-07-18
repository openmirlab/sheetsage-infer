"""Unit tests for `sheetsage.assets.get_asset_checksum`.

No network access: only exercises the in-memory `_ASSETS` registry built at import time
from `sheetsage/assets/*.json`, never `retrieve_asset` (which downloads).
"""

import pytest

from sheetsage.assets import get_asset_checksum, get_asset_tags


def test_get_asset_checksum_matches_the_manifest_for_every_tag():
    from sheetsage.assets import _ASSETS

    for tag in get_asset_tags():
        assert get_asset_checksum(tag) == _ASSETS[tag]["checksum"]


def test_get_asset_checksum_returns_the_known_handcrafted_model_checksums():
    # Regression: these are the two checkpoints Phonon's sheetsage-infer wrapper actually
    # exercises (use_jukebox=False), pinned here so a manifest edit that silently changes
    # either checksum is caught by this suite, not discovered downstream.
    assert (
        get_asset_checksum("SHEETSAGE_V02_HANDCRAFTED_MELODY_MODEL")
        == "70f10a4146da8f1294597516622901d93621c5cd1bbb4e9dc831f9c43c081ef4"
    )
    assert (
        get_asset_checksum("SHEETSAGE_V02_HANDCRAFTED_HARMONY_MODEL")
        == "d7f6dae6902618ba285d78459010a5388002efcd685f1f2ce334297eec6c799f"
    )


def test_get_asset_checksum_raises_a_clear_error_for_an_unknown_tag():
    with pytest.raises(ValueError, match="Unknown asset tag"):
        get_asset_checksum("NOT_A_REAL_TAG")
