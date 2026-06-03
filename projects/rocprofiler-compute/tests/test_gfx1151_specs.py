# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Tests for gfx1151 (Strix Halo, RDNA 3.5 APU) support added in
the gfx115x_fixes PR.  Covers:

  1. mi_gpu_spec.yaml — gfx1151 data registration
  2. specs.is_apu_arch() helper
  3. MachineSpecs chip-ID hexadecimal display
  4. MachineSpecs APU field hiding (compute/memory partition, num_xcd)
  5. MachineSpecs num_memory_channels rename + get_memory_channels()
  6. MachineSpecs num_gl1c field
  7. soc_gfx1151 SoC class initialisation
  8. parser.py num_memory_channels variable usage
"""

from __future__ import annotations

import argparse
import sys
import os
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
    from src.utils.mi_gpu_spec import MIGPUSpecs, mi_gpu_specs
    from src.utils.specs import MachineSpecs, is_apu_arch, _APU_HIDDEN_FIELDS
    from src.utils.parser import BUILD_IN_VARS
except ImportError:
    from utils.mi_gpu_spec import MIGPUSpecs, mi_gpu_specs
    from utils.specs import MachineSpecs, is_apu_arch, _APU_HIDDEN_FIELDS
    from utils.parser import BUILD_IN_VARS


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
        """Chip ID 5510 (0x1586) must map to strix_halo model."""
        chip_id_dict = mi_gpu_specs.get_chip_id_dict()
        assert 5510 in chip_id_dict, "Chip ID 5510 (0x1586) missing from chip_id_dict"
        assert chip_id_dict[5510].lower() == "strix_halo"

    def test_gfx1151_get_gpu_model_from_chip_id(self):
        """get_gpu_model('gfx1151', '5510') must return 'STRIX_HALO'."""
        model = mi_gpu_specs.get_gpu_model("gfx1151", "5510")
        assert model is not None
        assert model.upper() == "STRIX_HALO"

    def test_gfx1151_num_xcds_na_partition_returns_1(self):
        """For APU (compute_partition='N/A'), get_num_xcds must return 1."""
        result = mi_gpu_specs.get_num_xcds(
            gpu_arch="gfx1151",
            gpu_model="STRIX_HALO",
            compute_partition="N/A",
        )
        assert result == 1, (
            f"Expected 1 XCD for APU gfx1151 with N/A partition, got {result}"
        )

    def test_gfx1151_perfmon_config_nonempty(self):
        """gfx1151 perfmon_config must be populated (non-empty dict)."""
        config = mi_gpu_specs.get_perfmon_config("gfx1151")
        assert isinstance(config, dict) and len(config) > 0, (
            "perfmon_config for gfx1151 is empty or not a dict"
        )

    def test_gfx1151_perfmon_config_has_standard_blocks(self):
        """gfx1151 perfmon_config must contain the standard GFX IP blocks."""
        config = mi_gpu_specs.get_perfmon_config("gfx1151")
        for block in ("SQ", "TA", "TD", "TCP", "TCC", "CPC", "CPF", "SPI", "GRBM"):
            assert block in config, f"Block {block!r} missing from gfx1151 perfmon_config"

    def test_gfx1151_not_in_legacy_archs_but_returns_1(self):
        """Even though gfx1151 is NOT in LEGACY_ARCHS, the YAML n/a entry
        guarantees get_num_xcds returns 1 for compute_partition='N/A'.
        Verify both model-name and arch-name lookup paths."""
        # arch-based lookup path
        result = mi_gpu_specs.get_num_xcds(
            gpu_arch="gfx1151", compute_partition="N/A"
        )
        assert result == 1

        # model-based lookup path
        result = mi_gpu_specs.get_num_xcds(
            gpu_model="STRIX_HALO", compute_partition="N/A"
        )
        assert result == 1


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

    def test_gfx1152_is_apu(self):
        assert is_apu_arch("gfx1152") is True

    def test_gfx942_is_not_apu(self):
        assert is_apu_arch("gfx942") is False

    def test_gfx90a_is_not_apu(self):
        assert is_apu_arch("gfx90a") is False

    def test_gfx950_is_not_apu(self):
        assert is_apu_arch("gfx950") is False

    def test_none_is_not_apu(self):
        assert is_apu_arch(None) is False

    def test_empty_string_is_not_apu(self):
        assert is_apu_arch("") is False


# ===========================================================================
# 3. MachineSpecs — Chip ID hexadecimal display
# ===========================================================================

class TestChipIdHexDisplay:
    """Chip ID must be shown as 0x<HEX> in --specs output."""

    def _make_apu_mspec(self, chip_id="5510"):
        return make_minimal_mspec(
            gpu_arch="gfx1151",
            gpu_chip_id=chip_id,
            gpu_series="RDNA3.5",
            gpu_model="STRIX_HALO",
            total_l2_chan="8",
            num_memory_channels="8",
        )

    def test_decimal_chip_id_formatted_as_hex_in_repr(self):
        mspec = self._make_apu_mspec("5510")
        repr_str = repr(mspec)
        assert "0x1586" in repr_str, (
            "Chip ID 5510 should be displayed as '0x1586' in --specs repr"
        )
        # Raw decimal should NOT appear as a standalone value
        assert "5510" not in repr_str, (
            "Decimal '5510' should not appear in --specs repr (use hex)"
        )

    def test_decimal_chip_id_formatted_as_hex_in_get_class_members(self):
        mspec = self._make_apu_mspec("5510")
        df = mspec.get_class_members()
        assert "gpu_chip_id" in df.columns
        assert df["gpu_chip_id"].iloc[0] == "0x1586"

    def test_non_numeric_chip_id_not_reformatted(self):
        """Non-decimal chip_id (e.g. 'N/A') must pass through unchanged."""
        mspec = self._make_apu_mspec("N/A")
        repr_str = repr(mspec)
        assert "N/A" in repr_str

    def test_different_decimal_chip_id_formatted_correctly(self):
        """Verify hex formatting works for a different known chip ID."""
        # MI300X_A1 chip ID 29857 = 0x74A1
        mspec = make_minimal_mspec(
            gpu_arch="gfx942",
            gpu_chip_id="29857",
            total_l2_chan="16",
            num_memory_channels="32",
        )
        df = mspec.get_class_members()
        assert df["gpu_chip_id"].iloc[0] == "0x74A1"


# ===========================================================================
# 4. MachineSpecs — APU field hiding
# ===========================================================================

class TestApuFieldHiding:
    """compute_partition, memory_partition, num_xcd must be hidden for APU."""

    APU_HIDDEN = {"compute_partition", "memory_partition", "num_xcd"}
    APU_HIDDEN_PRETTY = {"Compute Partition", "Memory Partition", "Num XCDs"}

    def _make_apu_mspec(self):
        return make_minimal_mspec(
            gpu_arch="gfx1151",
            gpu_series="RDNA3.5",
            gpu_model="STRIX_HALO",
            total_l2_chan="8",
            num_memory_channels="8",
            compute_partition="N/A",
            memory_partition="N/A",
            num_xcd="1",
        )

    def _make_mi300_mspec(self):
        return make_minimal_mspec(
            gpu_arch="gfx942",
            gpu_series="MI300",
            gpu_model="MI300X_A1",
            total_l2_chan="16",
            num_memory_channels="32",
            compute_partition="SPX",
            memory_partition="NPS1",
            num_xcd="8",
        )

    # --- get_class_members ---

    def test_apu_hidden_fields_absent_from_get_class_members(self):
        df = self._make_apu_mspec().get_class_members()
        for field_name in self.APU_HIDDEN:
            assert field_name not in df.columns, (
                f"Field '{field_name}' should be hidden for APU but found in DataFrame"
            )

    def test_non_apu_hidden_fields_present_in_get_class_members(self):
        df = self._make_mi300_mspec().get_class_members()
        for field_name in self.APU_HIDDEN:
            assert field_name in df.columns, (
                f"Field '{field_name}' should be visible for non-APU but missing"
            )

    # --- __repr__ ---

    def test_apu_hidden_fields_absent_from_repr(self):
        repr_str = repr(self._make_apu_mspec())
        for pretty_name in self.APU_HIDDEN_PRETTY:
            assert pretty_name not in repr_str, (
                f"'{pretty_name}' should be hidden in APU --specs repr"
            )

    def test_non_apu_fields_visible_in_repr(self):
        repr_str = repr(self._make_mi300_mspec())
        for pretty_name in self.APU_HIDDEN_PRETTY:
            assert pretty_name in repr_str, (
                f"'{pretty_name}' should appear in non-APU --specs repr"
            )

    def test_apu_hidden_fields_constant(self):
        """_APU_HIDDEN_FIELDS must contain exactly the expected set."""
        assert _APU_HIDDEN_FIELDS == frozenset({
            "compute_partition",
            "memory_partition",
            "num_xcd",
        })


# ===========================================================================
# 5. MachineSpecs — num_memory_channels rename + get_memory_channels
# ===========================================================================

class TestNumMemoryChannels:
    """num_hbm_channels was renamed to num_memory_channels."""

    def test_num_memory_channels_field_exists(self):
        """MachineSpecs must have a num_memory_channels field."""
        from dataclasses import fields as dc_fields
        field_names = {f.name for f in dc_fields(MachineSpecs)}
        assert "num_memory_channels" in field_names
        assert "num_hbm_channels" not in field_names, (
            "Stale field num_hbm_channels still present; rename incomplete"
        )

    def test_num_memory_channels_metadata_name(self):
        """Display name for num_memory_channels must be 'Memory Channels'."""
        from dataclasses import fields as dc_fields
        for f in dc_fields(MachineSpecs):
            if f.name == "num_memory_channels":
                assert f.metadata.get("name") == "Memory Channels"
                break

    def test_get_memory_channels_method_exists(self):
        """MachineSpecs must have get_memory_channels() method."""
        assert hasattr(MachineSpecs, "get_memory_channels"), (
            "get_memory_channels() method is missing from MachineSpecs"
        )
        assert not hasattr(MachineSpecs, "get_hbm_channels"), (
            "Stale get_hbm_channels() still present; rename incomplete"
        )

    def test_get_memory_channels_returns_total_l2_chan_when_no_nps(self):
        """Without NPS partition, fallback returns total_l2_chan."""
        mspec = make_minimal_mspec(
            total_l2_chan="32",
            memory_partition="N/A",
        )
        assert mspec.get_memory_channels() == "32"

    def test_get_memory_channels_returns_total_l2_chan_when_partition_none(self):
        mspec = make_minimal_mspec(total_l2_chan="16", memory_partition=None)
        assert mspec.get_memory_channels() == "16"

    def test_get_memory_channels_nps1_returns_128(self):
        mspec = make_minimal_mspec(
            total_l2_chan="128",
            memory_partition="NPS1",
        )
        assert mspec.get_memory_channels() == "128"

    def test_get_memory_channels_nps4_returns_32(self):
        mspec = make_minimal_mspec(
            total_l2_chan="128",
            memory_partition="NPS4",
        )
        assert mspec.get_memory_channels() == "32"

    def test_get_memory_channels_nps8_returns_16(self):
        mspec = make_minimal_mspec(
            total_l2_chan="128",
            memory_partition="NPS8",
        )
        assert mspec.get_memory_channels() == "16"

    def test_num_memory_channels_shown_in_repr(self):
        """'Memory Channels' should appear in the --specs repr."""
        mspec = make_minimal_mspec(
            gpu_arch="gfx1151",
            total_l2_chan="8",
            num_memory_channels="8",
        )
        repr_str = repr(mspec)
        assert "Memory Channels" in repr_str

    def test_num_memory_channels_in_get_class_members_df(self):
        mspec = make_minimal_mspec(
            gpu_arch="gfx1151",
            total_l2_chan="8",
            num_memory_channels="8",
        )
        df = mspec.get_class_members()
        assert "num_memory_channels" in df.columns
        assert df["num_memory_channels"].iloc[0] == "8"


# ===========================================================================
# 6. MachineSpecs — num_gl1c field
# ===========================================================================

class TestNumGl1cField:
    """num_gl1c must exist but must NOT appear in the --specs table."""

    def test_num_gl1c_field_exists(self):
        from dataclasses import fields as dc_fields
        field_names = {f.name for f in dc_fields(MachineSpecs)}
        assert "num_gl1c" in field_names

    def test_num_gl1c_not_shown_in_table(self):
        """num_gl1c has show_in_table=False so it must not appear in repr or DF."""
        from dataclasses import fields as dc_fields
        for f in dc_fields(MachineSpecs):
            if f.name == "num_gl1c":
                assert f.metadata.get("show_in_table") is False
                break

    def test_num_gl1c_absent_from_repr(self):
        mspec = make_minimal_mspec(gpu_arch="gfx1151", num_gl1c="8")
        repr_str = repr(mspec)
        assert "Num GL1 Caches" not in repr_str
        assert "num_gl1c" not in repr_str

    def test_num_gl1c_absent_from_get_class_members(self):
        mspec = make_minimal_mspec(gpu_arch="gfx1151", num_gl1c="8")
        df = mspec.get_class_members()
        assert "num_gl1c" not in df.columns

    def test_num_gl1c_attribute_settable(self):
        """num_gl1c attribute must be readable/writable on MachineSpecs."""
        mspec = make_minimal_mspec()
        mspec.num_gl1c = "10"
        assert mspec.num_gl1c == "10"


# ===========================================================================
# 7. soc_gfx1151 — SoC class initialisation
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

    def test_soc_arch_is_gfx1151(self):
        soc = self._make_soc()
        assert soc.get_arch() == "gfx1151"

    def test_soc_sets_l2_banks_8(self):
        soc = self._make_soc()
        assert int(soc._mspec.l2_banks) == 8

    def test_soc_sets_lds_banks_per_cu_32(self):
        soc = self._make_soc()
        assert int(soc._mspec.lds_banks_per_cu) == 32

    def test_soc_sets_pipes_per_gpu_2(self):
        soc = self._make_soc()
        assert int(soc._mspec.pipes_per_gpu) == 2

    def test_soc_sets_num_memory_channels_8(self):
        """LPDDR5X 256-bit bus / 32-bit per channel = 8 channels."""
        soc = self._make_soc()
        assert soc._mspec.num_memory_channels == "8"

    def test_soc_sets_num_gl1c_for_32cu(self):
        """32 CU / 4 CU per SA = 8 GL1 caches."""
        soc = self._make_soc(cu_per_gpu="32")
        assert soc._mspec.num_gl1c == "8"

    def test_soc_sets_num_gl1c_for_40cu(self):
        """40 CU / 4 CU per SA = 10 GL1 caches."""
        soc = self._make_soc(cu_per_gpu="40")
        assert soc._mspec.num_gl1c == "10"

    def test_soc_sets_num_gl1c_for_16cu(self):
        """16 CU / 4 CU per SA = 4 GL1 caches."""
        soc = self._make_soc(cu_per_gpu="16")
        assert soc._mspec.num_gl1c == "4"

    def test_soc_num_gl1c_none_when_cu_per_gpu_none(self):
        """num_gl1c should remain None if cu_per_gpu is not set."""
        try:
            from src.rocprof_compute_soc.soc_gfx1151 import gfx1151_soc
        except ImportError:
            from rocprof_compute_soc.soc_gfx1151 import gfx1151_soc

        args = argparse.Namespace()
        mspec = make_minimal_mspec(gpu_arch="gfx1151", cu_per_gpu=None)
        soc = gfx1151_soc(args, mspec)
        assert soc._mspec.num_gl1c is None

    def test_soc_compatible_profilers(self):
        """Must list rocprofv3 / rocprofiler-sdk as compatible."""
        soc = self._make_soc()
        profilers = soc.get_compatible_profilers()
        assert "rocprofv3" in profilers or "rocprofiler-sdk" in profilers

    def test_soc_perfmon_config_nonempty(self):
        """Perfmon config must be a non-empty dict after init."""
        soc = self._make_soc()
        # Access via the private mangled name
        config = soc._OmniSoC_Base__perfmon_config
        assert isinstance(config, dict) and len(config) > 0

    def test_num_memory_channels_constant_is_8(self):
        try:
            from src.rocprof_compute_soc.soc_gfx1151 import gfx1151_soc
        except ImportError:
            from rocprof_compute_soc.soc_gfx1151 import gfx1151_soc
        assert gfx1151_soc._NUM_MEMORY_CHANNELS == 8


# ===========================================================================
# 8. parser.py — num_memory_channels variable
# ===========================================================================

class TestParserNumMemoryChannels:
    """Verify parser.py references num_memory_channels, not num_hbm_channels."""

    def test_hbm_bandwidth_formula_uses_num_memory_channels(self):
        """BUILD_IN_VARS['hbmBandwidth'] must use $num_memory_channels."""
        formula = BUILD_IN_VARS.get("hbmBandwidth", "")
        assert "$num_memory_channels" in formula, (
            f"hbmBandwidth formula still uses old name. Got: {formula!r}"
        )
        assert "$num_hbm_channels" not in formula, (
            "hbmBandwidth formula still contains stale $num_hbm_channels"
        )

    def test_create_sys_vars_uses_num_memory_channels(self):
        """create_sys_vars() must read from sys_info.num_memory_channels."""
        try:
            import src.utils.parser as parser_mod
        except ImportError:
            import utils.parser as parser_mod

        import inspect
        source = inspect.getsource(parser_mod.create_sys_vars)
        assert "num_memory_channels" in source, (
            "create_sys_vars() does not reference num_memory_channels"
        )
        assert "num_hbm_channels" not in source, (
            "create_sys_vars() still contains stale num_hbm_channels"
        )

    def test_create_sys_vars_evaluates_num_memory_channels(self):
        """create_sys_vars() must produce ammolite__num_memory_channels key."""
        try:
            from src.utils.parser import create_sys_vars
        except ImportError:
            from utils.parser import create_sys_vars
        import pandas as pd

        # Minimal sys_info with the renamed field
        sys_info = pd.Series({
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
            "num_memory_channels": 8.0,
            "num_xcd": 1,
            "wave_size": 32,
            "total_l2_chan": 8,
        })

        result = create_sys_vars(sys_info)
        assert "ammolite__num_memory_channels" in result, (
            "create_sys_vars() did not produce ammolite__num_memory_channels"
        )
        assert "ammolite__num_hbm_channels" not in result, (
            "create_sys_vars() still produces stale ammolite__num_hbm_channels"
        )
        assert result["ammolite__num_memory_channels"] == 8.0
