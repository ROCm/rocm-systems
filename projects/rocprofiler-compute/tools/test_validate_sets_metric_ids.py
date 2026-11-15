# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for tools/validate_sets_metric_ids.py.

Run from the project directory: python3 -m pytest tools/test_validate_sets_metric_ids.py
"""

import copy
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from validate_sets_metric_ids import SETS_DIR, library_set_errors

ARCH = "gfx942"


def test_library_sets_match_the_sets_here():
    """The shipped sets agree; a set with another title, and a set with a row less,
    differ from their metric library sets."""
    path = SETS_DIR / f"{ARCH}_sets.yaml"
    sets = yaml.safe_load(path.read_text(encoding="utf-8"))["sets"]
    assert library_set_errors(ARCH, sets) == []

    changed = copy.deepcopy(sets)
    changed[0]["title"] = "Another title"
    changed[1]["metric"].pop()
    assert library_set_errors(ARCH, changed) == [
        f"[{ARCH}] metric library set '{set_option}' is not the set here"
        for set_option in sorted(entry["set_option"] for entry in changed[:2])
    ]
