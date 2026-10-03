"""The installed inference registry contains no training/evaluation dataset surface.

Reads: packaged asset catalogs and the public acquisition/checksum APIs.
These checks never download any asset.
"""

import json

import pytest

from sheetsage import LIB_DIR
from sheetsage.assets import get_asset_checksum, get_asset_tags, retrieve_asset

DATASET_TAGS = (
    "HOOKTHEORY", "HOOKTHEORY_RAW",
    *(f"HOOKTHEORY_{split}_{kind}" for split in ("TRAIN", "VALID", "TEST")
      for kind in ("SEGMENTS", "MIDI")),
    *(f"RWC_{group}_{kind}" for group in ("RYY", "RYYVOX") for kind in ("SEGMENTS", "MIDI")),
    *(f"RWC_AUDIO_{name}" for name in ("P012", "P038", "P060", "P070", "P079",
                                      "G002", "G010", "G036", "G068", "G072")),
)


@pytest.mark.parametrize("tag", DATASET_TAGS)
def test_dataset_tags_are_not_runtime_download_options(tag):
    assert tag not in get_asset_tags()
    with pytest.raises(ValueError):
        retrieve_asset(tag)
    with pytest.raises(ValueError, match="Unknown asset tag"):
        get_asset_checksum(tag)


def test_all_inference_and_test_assets_remain_available():
    expected = {}
    for name in ("sheetsage.json", "jukebox.json", "test.json"):
        expected.update(json.loads((LIB_DIR / "assets" / name).read_text()))
    assert len(expected) == 24
    assert get_asset_tags() == set(expected)
    for tag, asset in expected.items():
        assert get_asset_checksum(tag) == asset["checksum"]
