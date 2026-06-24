# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Tests for gfx1151 (Strix Halo, RDNA 3.5 APU) support.

Covers:
  1. mi_gpu_spec.yaml — gfx1151 registration and perfmon config
  2. is_apu_arch() — APU detection for gfx115x prefix
  3. MachineSpecs family split — CDNA vs RDNA memory channel fields
  4. num_gl1c field — RDNA-only GL1 cache count for bandwidth calculations
  5. soc_gfx1151 — SoC class initialization and gfx1151-specific values
  6. Built-in variables — family-specific $num_hbm_channels vs $num_memory_channels
  7. Fixture schema — RDNA workloads use correct field names
"""

from __future__ import annotations

import argparse
import os
import sys
import types

import pytest

# ---------------------------------------------------------------------------
# Path setup (mirrors conftest.py)
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def _ensure_vendored_yaml() -> None:
    """Inject system PyYAML into the ``vendored`` namespace if the C-extension
    build artefact (``vendored/pyyaml/lib/``) is absent.

    The project vendors PyYAML so that profile mode has zero runtime
    dependencies, but in a sparse-checkout / development environment the
    CMake build step that copies ``pyyaml/lib/`` has not been run.
    The system-installed ``yaml`` package is API-compatible, so we can
    inject it transparently for tests.
    """
    if "vendored" not in sys.modules:
        try:
            import yaml  # system PyYAML (or pyyaml pip package)
        except ImportError:
            pytest.skip("System yaml not available; skipping vendored yaml patch")
            return
        vendor_mod = types.ModuleType("vendored")
        vendor_mod.yaml = yaml
        sys.modules["vendored"] = vendor_mod


_ensure_vendored_yaml()

# ---------------------------------------------------------------------------
# Project imports — all pure-Python; no hardware required
# ---------------------------------------------------------------------------
try:
    from src.utils.mi_gpu_spec import mi_gpu_specs
    from src.utils.specs import (
        MachineSpecs,
        MachineSpecsCDNA,
        MachineSpecsRDNA,
        is_apu_arch,
    )
    from src.utils.utils_counter_defs import get_build_in_vars
except ImportError:
    from utils.mi_gpu_spec import mi_gpu_specs
    from utils.specs import (
        MachineSpecs,
        MachineSpecsCDNA,
        MachineSpecsRDNA,
        is_apu_arch,
    )
    from utils.utils_counter_defs import get_build_in_vars


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_minimal_mspec(**kwargs) -> MachineSpecs:
    """Return a MachineSpecs instance with rocminfo_lines=None (no HW needed)
    plus any extra keyword arguments set as attributes.
    """
    mspec = MachineSpecs(rocminfo_lines=None)
    for k, v in kwargs.items():
        setattr(mspec, k, v)
    return mspec


# ===========================================================================
# 1. mi_gpu_spec.yaml — gfx1151 registration
# ===========================================================================


class TestMIGPUSpecYamlGfx1151:
    """Verify the gfx1151 / Strix Halo entries exist in mi_gpu_spec.yaml."""

    def test_gfx1151_in_supported_gpu_series_dict(self):
        """gfx1151 must appear in the supported-arch → series mapping."""
        series_dict = mi_gpu_specs.get_gpu_series_dict()
        assert "gfx1151" in series_dict, (
            "gfx1151 not found in gpu_series_dict; check mi_gpu_spec.yaml"
        )

    def test_gfx1151_gpu_series_is_rdna35(self):
        """get_gpu_series should return 'RDNA3.5' (uppercased) for gfx1151."""
        result = mi_gpu_specs.get_gpu_series("gfx1151")
        assert result == "RDNA3.5", f"Expected 'RDNA3.5', got {result!r}"

    def test_gfx1151_chip_id_5510_in_dict(self):
        """Chip ID 5510 (0x1586) must map to rdna35_halo model."""
        chip_id_dict = mi_gpu_specs.get_chip_id_dict()
        assert 5510 in chip_id_dict, "Chip ID 5510 (0x1586) missing from chip_id_dict"
        assert chip_id_dict[5510].lower() == "rdna35_halo"

    def test_gfx1151_perfmon_config_nonempty(self):
        """gfx1151 perfmon_config must be populated (non-empty dict)."""
        config = mi_gpu_specs.get_perfmon_config("gfx1151")
        assert isinstance(config, dict) and len(config) > 0, (
            "perfmon_config for gfx1151 is empty or not a dict"
        )

    def test_gfx1151_perfmon_config_has_standard_blocks(self):
        """gfx1151 perfmon_config must contain the RDNA 3.5 IP blocks."""
        config = mi_gpu_specs.get_perfmon_config("gfx1151")
        for block in (
            "SQ",
            "TA",
            "TCP",
            "CPC",
            "SPI",
            "GRBM",
            "GCEA",
            "GL1A",
            "GL1C",
            "GL2A",
            "GL2C",
        ):
            assert block in config, (
                f"Block {block!r} missing from gfx1151 perfmon_config"
            )


# ===========================================================================
# 2. is_apu_arch() helper
# ===========================================================================


class TestIsApuArch:
    """Unit tests for specs.is_apu_arch()."""

    def test_gfx1151_is_apu(self):
        assert is_apu_arch("gfx1151") is True

    def test_gfx1150_is_apu(self):
        """Any gfx115x should be detected as APU."""
        assert is_apu_arch("gfx1150") is True

    def test_gfx942_is_not_apu(self):
        assert is_apu_arch("gfx942") is False

    def test_none_is_not_apu(self):
        assert is_apu_arch(None) is False


# ===========================================================================
# 3. MachineSpecs — family-specific memory-channel fields
# ===========================================================================


class TestMemoryChannelFields:
    """num_memory_channels is defined on the base class and inherited by both
    CDNA and RDNA families.
    """

    def test_channel_field_on_base_class(self):
        from dataclasses import fields as dc_fields

        base = {f.name for f in dc_fields(MachineSpecs)}
        cdna = {f.name for f in dc_fields(MachineSpecsCDNA)}
        rdna = {f.name for f in dc_fields(MachineSpecsRDNA)}
        assert "num_memory_channels" in base
        assert "num_memory_channels" in cdna
        assert "num_memory_channels" in rdna

    def test_channel_display_name(self):
        from dataclasses import fields as dc_fields

        base = {f.name: f.metadata.get("name") for f in dc_fields(MachineSpecs)}
        assert base["num_memory_channels"] == "Memory Channels"

    def test_cdna_memory_channels_in_repr_and_members(self):
        mspec = MachineSpecsCDNA(rocminfo_lines=None, num_memory_channels="32")
        assert "Memory Channels" in repr(mspec)
        assert mspec.get_class_members().get("num_memory_channels") == "32"

    def test_rdna_memory_channels_in_repr_and_members(self):
        mspec = MachineSpecsRDNA(rocminfo_lines=None, num_memory_channels="8")
        assert "Memory Channels" in repr(mspec)
        assert mspec.get_class_members().get("num_memory_channels") == "8"


# ===========================================================================
# 4. MachineSpecs — num_gl1c field
# ===========================================================================


class TestNumGl1cField:
    """num_gl1c is an RDNA-only spec (one GL1 per Shader Array).

    It lives on MachineSpecsRDNA (not the CDNA-shaped base), is shown in the
    --specs table, and is serialized so the GL1 bandwidth-ceiling analysis
    formulas can reference $num_gl1c.
    """

    def test_num_gl1c_is_rdna_only_field(self):
        from dataclasses import fields as dc_fields

        rdna_fields = {f.name for f in dc_fields(MachineSpecsRDNA)}
        base_fields = {f.name for f in dc_fields(MachineSpecs)}
        assert "num_gl1c" in rdna_fields
        assert "num_gl1c" not in base_fields

    def test_num_gl1c_shown_in_table(self):
        from dataclasses import fields as dc_fields

        for f in dc_fields(MachineSpecsRDNA):
            if f.name == "num_gl1c":
                assert f.metadata.get("show_in_table") is True
                break
        else:
            raise AssertionError("num_gl1c field not found on MachineSpecsRDNA")

    def test_num_gl1c_derived_from_shader_arrays(self):
        """num_gl1c == se_per_gpu * sa_per_se (one GL1 per Shader Array)."""
        mspec = MachineSpecsRDNA(
            rocminfo_lines=None,
            gpu_arch="gfx1151",
            gpu_model="rdna35_halo",
            se_per_gpu="2",
            sa_per_se="2",
            l2_banks=8,
        )
        gpu_info = {
            "compute_partition": "N/A",
            "memory_partition": "N/A",
            "num_compute_units": 32,
            "gpu_cache_info": {
                "cache": [
                    {
                        "cache_level": 1,
                        "cache_properties": ["DATA_CACHE"],
                        "cache_size": 32,
                        "num_cache_instance": 32,
                    },
                    {
                        "cache_level": 2,
                        "cache_properties": ["DATA_CACHE"],
                        "cache_size": 2048,
                        "num_cache_instance": 1,
                    },
                ]
            },
            "vram_bit_width": 256,
        }
        mspec.finalize_soc_fields(gpu_info)
        assert mspec.num_gl1c == "4"
        # 256-bit bus / 32-bit per channel = 8 memory channels.
        assert mspec.num_memory_channels == "8"


# ===========================================================================
# 5. soc_gfx1151 — SoC class initialisation
# ===========================================================================


class TestSocGfx1151:
    """Tests for the gfx1151_soc class (no hardware required)."""

    def _make_soc(self, cu_per_gpu="32"):
        """Create a gfx1151_soc instance with a minimal MachineSpecs."""
        try:
            from src.rocprof_compute_soc.soc_gfx1151 import gfx1151_soc
        except ImportError:
            from rocprof_compute_soc.soc_gfx1151 import gfx1151_soc

        args = argparse.Namespace()
        mspec = make_minimal_mspec(
            gpu_arch="gfx1151",
            cu_per_gpu=cu_per_gpu,
        )
        soc = gfx1151_soc(args, mspec)
        return soc

    def test_soc_instantiates_without_hardware(self):
        """gfx1151_soc must not raise with rocminfo_lines=None."""
        soc = self._make_soc()
        assert soc is not None

    def test_soc_sets_l2_banks_8(self):
        soc = self._make_soc()
        assert int(soc._mspec.l2_banks) == 8

    def test_soc_sets_lds_banks_per_cu_32(self):
        soc = self._make_soc()
        assert int(soc._mspec.lds_banks_per_cu) == 32

    def test_soc_sets_pipes_per_gpu_2(self):
        soc = self._make_soc()
        assert int(soc._mspec.pipes_per_gpu) == 2

    def test_soc_compatible_profilers(self):
        """Must list rocprofv3 / rocprofiler-sdk as compatible."""
        soc = self._make_soc()
        profilers = soc.get_compatible_profilers()
        assert "rocprofv3" in profilers or "rocprofiler-sdk" in profilers


# ===========================================================================
# 6. memory-channel built-ins + analysis variables
# ===========================================================================


def _base_sys_info() -> "dict":
    """Common sysinfo fields required by create_sys_vars (no channel field)."""
    return {
        "se_per_gpu": 2,
        "pipes_per_gpu": 2,
        "cu_per_gpu": 32,
        "simd_per_cu": 2,
        "sqc_per_gpu": 16,
        "lds_banks_per_cu": 32,
        "cur_sclk": 2800.0,
        "cur_mclk": 2133.0,
        "max_mclk": 2133.0,
        "max_sclk": 2800.0,
        "max_waves_per_cu": 32,
        "num_xcd": 1,
        "wave_size": 32,
        "total_l2_chan": 8,
    }


class TestMemoryChannelVars:
    """Memory-channel built-ins and analysis variables are family-specific."""

    def test_hbm_bandwidth_formula_uses_num_memory_channels(self):
        """CDNA hbmBandwidth must use $num_memory_channels."""
        formula = get_build_in_vars("MI300").get("hbmBandwidth", "")
        assert "$num_memory_channels" in formula

    def test_cdna_sysinfo_emits_num_memory_channels(self):
        import pandas as pd

        from utils.metrics.evaluation_pipeline import create_sys_vars

        sys_info = pd.Series({
            **_base_sys_info(),
            "gpu_arch": "gfx942",
            "num_memory_channels": 32.0,
        })
        result = create_sys_vars(sys_info)
        assert result["ammolite__num_memory_channels"] == 32.0

    def test_rdna_sysinfo_emits_num_memory_channels(self):
        import pandas as pd

        from utils.metrics.evaluation_pipeline import create_sys_vars

        sys_info = pd.Series({
            **_base_sys_info(),
            "gpu_arch": "gfx1151",
            "num_memory_channels": 8.0,
        })
        result = create_sys_vars(sys_info)
        assert result["ammolite__num_memory_channels"] == 8.0


# ===========================================================================
# 7. committed RDNA workload fixtures match the RDNA family schema
# ===========================================================================

_RDNA_WORKLOAD_DIR = os.path.join(ROOT, "tests", "workloads")
_RDNA_WORKLOADS = ["dispatch_0", "ipblocks_CU", "kernel", "no_roof", "path", "vcopy"]

# CDNA-only columns that MachineSpecsRDNA does not define. A fixture carrying
# any of these is stale: generate_machine_specs() would drop it on load and the
# RDNA-only metrics below would silently never populate.
_CDNA_ONLY_COLUMNS = [
    "compute_partition",
    "memory_partition",
    "num_xcd",
]


class TestRdnaWorkloadFixtures:
    """The committed RDNA35_HALO fixtures must use the RDNA family schema.

    Guards against fixture drift: the analyze-path integration tests only
    assert exit code, so a fixture written with the legacy CDNA field names
    would degrade memory-bandwidth / GL1 roofline ceilings without failing.
    """

    @pytest.mark.parametrize("workload", _RDNA_WORKLOADS)
    def test_fixture_has_no_legacy_cdna_columns(self, workload: str) -> None:
        import pandas as pd

        path = os.path.join(_RDNA_WORKLOAD_DIR, workload, "RDNA35_HALO", "sysinfo.csv")
        columns = pd.read_csv(path).columns
        stale = [c for c in _CDNA_ONLY_COLUMNS if c in columns]
        assert not stale, f"{workload}: stale CDNA-only columns present: {stale}"

    @pytest.mark.parametrize("workload", _RDNA_WORKLOADS)
    def test_fixture_resolves_rdna_sys_vars(self, workload: str) -> None:
        import pandas as pd

        from utils.metrics.evaluation_pipeline import create_sys_vars

        path = os.path.join(_RDNA_WORKLOAD_DIR, workload, "RDNA35_HALO", "sysinfo.csv")
        sys_info = pd.read_csv(path).iloc[0]
        result = create_sys_vars(sys_info)

        # num_memory_channels (LPDDR5X) feeds the memory-bandwidth ceiling.
        assert result["ammolite__num_memory_channels"] == 8.0
        # num_gl1c (one GL1 per Shader Array) feeds the GL1 bandwidth ceiling.
        assert result["ammolite__num_gl1c"] == 4
