# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the static instruction execution pipeline lookup."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from pc_sampling import instruction_type


@pytest.fixture(autouse=True)
def clear_table_cache():
    """Drop the per-process table cache so each test loads it again."""
    instruction_type._load_instruction_pipelines.cache_clear()
    yield
    instruction_type._load_instruction_pipelines.cache_clear()


@pytest.fixture
def pipeline_table(tmp_path):
    """Point the loader at a small generated table."""
    analysis_configs = tmp_path / "rocprof_compute_soc" / "analysis_configs"
    analysis_configs.mkdir(parents=True)
    (analysis_configs / instruction_type.INSTRUCTION_PIPELINES_FILE).write_text(
        json.dumps({
            "commit": "0" * 40,
            "pipelines": {
                "v_mov_b32_e32": "VALU",
                "s_waitcnt": "INTERNAL",
                "v_mfma_f32_16x16x16f16": "MATRIX",
            },
        }),
        encoding="utf-8",
    )
    with patch.object(instruction_type.config, "rocprof_compute_home", tmp_path):
        yield


@pytest.mark.parametrize(
    "instruction, expected",
    [
        ("v_mov_b32_e32 v1, 0", "VALU"),
        ("s_waitcnt", "INTERNAL"),
        ("v_mfma_f32_16x16x16f16 a[0:3], v0, v1, a[0:3]", "MATRIX"),
        ("v_not_a_real_instruction v0", None),
        (None, None),
        ("", None),
    ],
)
def test_classify_reads_the_leading_mnemonic(pipeline_table, instruction, expected):
    assert instruction_type.classify(instruction) == expected


def test_classify_without_a_table_warns_once(tmp_path):
    """A missing table leaves every type unset instead of failing analyze."""
    with patch.object(instruction_type.config, "rocprof_compute_home", tmp_path):
        with patch.object(instruction_type, "console_warning") as console_warning:
            assert instruction_type.classify("v_mov_b32_e32 v1, 0") is None
            assert instruction_type.classify("s_waitcnt") is None

    console_warning.assert_called_once()
