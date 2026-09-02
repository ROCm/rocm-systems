# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils/specs.py."""

from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

import utils.specs as specs
from utils.specs import (
    MachineSpecs,
    MachineSpecsCDNA,
    MachineSpecsRDNA35,
    generate_machine_specs,
    spec_family_for_arch,
)


@pytest.mark.misc
@pytest.mark.parametrize(
    "mock_kwargs",
    [
        {"side_effect": FileNotFoundError("missing")},
        {
            "return_value": CompletedProcess(
                args=["rocminfo"], returncode=1, stdout="", stderr="boom"
            )
        },
    ],
    ids=["missing_binary", "nonzero_exit"],
)
def test_run_fails_fast(mock_kwargs):
    with patch.object(specs.subprocess, "run", **mock_kwargs), pytest.raises(
        SystemExit
    ):
        specs._run_command(["rocminfo"])


@pytest.mark.misc
def test_set_cache_sizes_cdna_uses_l1():
    """CDNA models (e.g. mi300x_a1) should key vL1D as 'L1'."""
    cache_info = {
        "cache": [
            {
                "cache_level": 1,
                "cache_properties": ["DATA_CACHE"],
                "cache_size": 16,
                "num_cache_instance": 304,
            }
        ]
    }
    result = specs.set_cache_sizes(
        "mi300x_a1", 304, cache_info, num_dies=4, num_se=8, num_sa_se=2
    )
    assert "L1" in result
    assert "L0" not in result
    assert result["L1"] == 16 * 1024


@pytest.mark.misc
def test_set_cache_sizes_rdna_uses_l0_and_l1():
    """RDNA models (e.g. rdna35_halo) populate both L0 (per-CU GL0) and L1 (per-SA GL1).
    Both are reported as level-1 DATA_CACHE by amd-smi; L0 has higher instance count.
    """
    cache_info = {
        "cache": [
            {
                "cache_level": 1,
                "cache_properties": ["DATA_CACHE"],
                "cache_size": 32,
                "num_cache_instance": 40,  # GL0: one per CU (40 CUs total)
            },
            {
                "cache_level": 1,
                "cache_properties": ["DATA_CACHE"],
                "cache_size": 128,
                "num_cache_instance": 8,  # GL1: one per SA (num_se * num_sa_se = 4*2)
            },
        ]
    }
    result = specs.set_cache_sizes(
        "rdna35_halo", 40, cache_info, num_dies=1, num_se=4, num_sa_se=2
    )
    assert result["L0"] == 32 * 1024
    assert result["L1"] == 128 * 1024


@pytest.mark.misc
def test_set_cache_sizes_l2_and_mall():
    """L2 and MALL (L3) cache entries are correctly parsed and scaled by num_dies."""
    cache_info = {
        "cache": [
            {
                "cache_level": 2,
                "cache_size": 512,
                "cache_properties": [],
                "num_cache_instance": 1,
            },
            {
                "cache_level": 3,
                "cache_size": 256,
                "cache_properties": [],
                "num_cache_instance": 1,
            },
        ]
    }
    result = specs.set_cache_sizes(
        "mi300x_a1", 304, cache_info, num_dies=4, num_se=8, num_sa_se=2
    )
    assert result["L2"] == 512 * 1024
    assert result["MALL"] == 256 * 1024 // 4


@pytest.mark.misc
def test_set_cache_sizes_selects_vl1d_by_max_instance_count():
    """Harvested GPU: num_cu reported by rocminfo may be less than the max
    num_cache_instance in amd-smi. set_cache_sizes must select vL1D by the
    highest num_cache_instance, not by exact match to num_cu.
    """
    cache_info = {
        "cache": [
            {
                "cache_level": 1,
                "cache_properties": ["DATA_CACHE"],
                "cache_size": 16,
                "num_cache_instance": 228,  # max instances (some CUs harvested)
            },
            {
                "cache_level": 1,
                "cache_properties": ["DATA_CACHE"],
                "cache_size": 16,
                "num_cache_instance": 190,  # lower instance count entry
            },
        ]
    }
    # num_cu=224 simulates harvested GPU (fewer than 228 active CUs)
    result = specs.set_cache_sizes(
        "mi300x_a1", 224, cache_info, num_dies=4, num_se=8, num_sa_se=2
    )
    assert "L1" in result
    assert result["L1"] == 16 * 1024


@pytest.mark.misc
def test_kw_only_rejects_positional_arguments():
    """MachineSpecs construction must be keyword-only (regression for the
    _kw_only decorator that previously accepted positional args silently).
    """
    with pytest.raises(TypeError):
        MachineSpecs("gfx942")

    spec = MachineSpecs(gpu_arch="gfx942")
    assert spec.gpu_arch == "gfx942"


@pytest.mark.misc
@pytest.mark.parametrize(
    "series, expected_cls",
    [
        ("RDNA3.5", MachineSpecsRDNA35),
        ("mi300", MachineSpecsCDNA),
        (None, MachineSpecsCDNA),
    ],
    ids=["rdna35", "cdna", "unknown"],
)
def test_spec_family_for_arch_dispatch(series, expected_cls):
    with patch.object(specs.mi_gpu_specs, "get_gpu_series", return_value=series):
        assert spec_family_for_arch("gfx_test") is expected_cls


@pytest.mark.misc
@pytest.mark.parametrize(
    "se_per_gpu, sa_per_se, vram_bit_width, expected_gl1c, expected_channels",
    [
        (4, 2, 256, "8", "8"),
        (4, 2, None, "8", "32"),
        (None, 2, 256, None, "8"),
    ],
    ids=["bit_width", "fallback_total_l2_chan", "missing_se"],
)
def test_rdna35_finalize_soc_fields(
    se_per_gpu, sa_per_se, vram_bit_width, expected_gl1c, expected_channels
):
    """RDNA 3.5 derives num_gl1c from SE*SA and num_memory_channels from the
    VRAM bus width, falling back to total_l2_chan when width is unavailable.
    """
    spec = MachineSpecsRDNA35(
        gpu_arch="gfx1151",
        gpu_model="rdna35_halo",
        se_per_gpu=se_per_gpu,
        sa_per_se=sa_per_se,
        l2_banks="8",
        total_l2_chan="32",
    )
    gpu_info = {
        "num_compute_units": 0,
        "gpu_cache_info": {},
        "vram_bit_width": vram_bit_width,
    }
    with patch.object(specs, "set_cache_sizes", return_value={}), patch.object(
        specs.mi_gpu_specs, "get_num_dies", return_value=1
    ), patch.object(specs, "totall2_banks", return_value="32"):
        spec.finalize_soc_fields(gpu_info)

    assert spec.num_gl1c == expected_gl1c
    assert spec.num_memory_channels == expected_channels


@pytest.mark.misc
def test_reconstruct_specs_from_sysinfo_round_trip():
    """generate_machine_specs reconstructs the arch-specific subclass from a
    saved sysinfo dict.
    """
    sysinfo = {
        "version": "3",
        "gpu_arch": "gfx1151",
        "num_gl1c": "8",
        "num_memory_channels": "8",
    }
    with patch.object(
        specs, "get_version", return_value={"version": "3.0.0"}
    ), patch.object(specs.mi_gpu_specs, "get_gpu_series", return_value="RDNA3.5"):
        spec = generate_machine_specs(None, sysinfo)

    assert isinstance(spec, MachineSpecsRDNA35)
    assert spec.num_gl1c == "8"
    assert spec.num_memory_channels == "8"


@pytest.mark.misc
def test_reconstruct_specs_from_sysinfo_missing_version_raises():
    """A sysinfo dict without a version key aborts via console_error/KeyError."""
    with pytest.raises((KeyError, SystemExit)):
        generate_machine_specs(None, {"gpu_arch": "gfx1151"})


@pytest.mark.misc
def test_get_rocm_ver_env_override(monkeypatch, tmp_path):
    """ROCM_VER overrides detection when no .info/ directory is present."""
    monkeypatch.setenv("ROCM_PATH", str(tmp_path))
    monkeypatch.setenv("ROCM_VER", "6.9.9")
    assert specs.get_rocm_ver() == "6.9.9"


@pytest.mark.misc
def test_get_rocm_ver_undetectable_errors(monkeypatch, tmp_path):
    """Missing .info/ and unset ROCM_VER terminates via console_error."""
    monkeypatch.setenv("ROCM_PATH", str(tmp_path))
    monkeypatch.delenv("ROCM_VER", raising=False)
    with patch.object(specs, "console_error") as console_error_mock:
        specs.get_rocm_ver()
    console_error_mock.assert_called_once()
    assert (
        "Unable to detect a complete local ROCm installation"
        in (console_error_mock.call_args.args[0])
    )
