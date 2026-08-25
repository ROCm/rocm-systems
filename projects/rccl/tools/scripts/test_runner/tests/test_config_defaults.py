#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""Host tests for configuration/suite-level init-pipeline defaults.

schema.json blesses warmup_profile / perf_sensitive / fork_expand at test_configuration
and test_suite level, but the loader dropped all three silently: the keys vanished and
the run died later with the confusing "resolved ZERO pipeline entries" fatal.
"""

import json
import os
import sys

_RUNNER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RUNNER_ROOT not in sys.path:
    sys.path.insert(0, _RUNNER_ROOT)

# Aliased so pytest does not try to collect the __init__-bearing class.
from lib.test_config import TestConfigProcessor as ConfigProcessor  # noqa: E402
from lib.pipeline_runner import resolve_test  # noqa: E402


def _load(tmp_path, config):
    path = os.path.join(str(tmp_path), "c.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f)
    return ConfigProcessor(path).parse_test_suites()


_FORK_EXPAND = {"num_gpus": [8], "process_mask": 1, "ranks_per_gpu": 1}


def test_configuration_level_pipeline_keys_reach_the_tests(tmp_path):
    suites = _load(tmp_path, {
        "test_configurations": {
            "c": {
                "binary": "rccl-UnitTests",
                "warmup_profile": "fork_coll",
                "perf_sensitive": True,
                "fork_expand": _FORK_EXPAND,
                "tests": [{"name": "T", "test_filter": "A.B"}],
            }
        },
        "test_suites": [{"name": "s", "config": "c", "enabled": True}],
    })
    t = suites[0]["tests"][0]
    assert t["warmup_profile"] == "fork_coll"
    assert t["perf_sensitive"] is True
    assert t["fork_expand"] == _FORK_EXPAND


def test_suite_level_overrides_configuration_level(tmp_path):
    suites = _load(tmp_path, {
        "test_configurations": {
            "c": {"binary": "rccl-UnitTests", "warmup_profile": "fork_coll",
                  "tests": [{"name": "T", "test_filter": "A.B"}]}
        },
        "test_suites": [{"name": "s", "config": "c", "enabled": True,
                         "warmup_profile": "none"}],
    })
    assert suites[0]["tests"][0]["warmup_profile"] == "none"


def test_test_level_still_wins(tmp_path):
    suites = _load(tmp_path, {
        "test_configurations": {
            "c": {"binary": "rccl-UnitTests", "warmup_profile": "fork_coll",
                  "tests": [{"name": "T", "test_filter": "A.B", "warmup_profile": "none"}]}
        },
        "test_suites": [{"name": "s", "config": "c", "enabled": True}],
    })
    assert suites[0]["tests"][0]["warmup_profile"] == "none"


def test_inherited_profile_is_suite_provenance_not_an_implicit_default(tmp_path):
    """The overbroad-default guard must see 'suite', never 'fallback' -- an inherited
    profile is an explicit config decision, but not a per-entry one."""
    suites = _load(tmp_path, {
        "test_configurations": {
            "c": {"binary": "rccl-UnitTests", "warmup_profile": "mpi_coll",
                  "tests": [{"name": "T", "test_filter": "A.B"}]}
        },
        "test_suites": [{"name": "s", "config": "c", "enabled": True}],
    })
    t = suites[0]["tests"][0]
    assert t["_warmup_profile_from_suite"] is True
    assert resolve_test(t, exec_mode="init-pipeline")["provenance"] == "suite"


def test_explicit_test_level_profile_keeps_entry_provenance(tmp_path):
    suites = _load(tmp_path, {
        "test_configurations": {
            "c": {"binary": "rccl-UnitTestsMPI",
                  "tests": [{"name": "T", "test_filter": "A.B",
                             "warmup_profile": "mpi_coll"}]}
        },
        "test_suites": [{"name": "s", "config": "c", "enabled": True}],
    })
    t = suites[0]["tests"][0]
    assert "_warmup_profile_from_suite" not in t
    assert resolve_test(t, exec_mode="init-pipeline")["provenance"] == "entry"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
