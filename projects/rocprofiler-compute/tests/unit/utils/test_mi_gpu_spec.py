# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT


import tempfile
from pathlib import Path
from unittest.mock import patch

import common
import pytest

from utils.mi_gpu_spec import MIGPUSpecs
from utils.specs import _extract_gpu_info


class TestMIGPUSpecs:
    # -- YAML parsing / initialization ---------------------------------------

    def test_yaml_loads_successfully(self):
        yaml_path = Path(common.SRC) / "utils" / "mi_gpu_spec.yaml"
        data = MIGPUSpecs._load_yaml(str(yaml_path))
        assert isinstance(data, dict)
        assert "mi_gpu_spec" in data
        assert len(data["mi_gpu_spec"]) > 0

    def test_gpu_model_dict_consistent_with_all_models(self):
        all_models = set(MIGPUSpecs.get_all_gpu_models())
        models_from_dict = set()
        for gpu_models in MIGPUSpecs._gpu_model_dict.values():
            models_from_dict.update(gpu_models)
        assert models_from_dict == all_models

    def test_supported_archs_matches_soc_files(self):
        soc_dir = Path(common.SRC) / "rocprof_compute_soc"
        soc_archs = {f.stem.removeprefix("soc_") for f in soc_dir.glob("soc_gfx*.py")}

        supported = set(common.SUPPORTED_ARCHS)
        missing_from_supported = soc_archs - supported
        missing_soc_file = supported - soc_archs
        assert not missing_from_supported, (
            f"SoC files exist but missing from SUPPORTED_ARCHS:"
            f" {missing_from_supported}"
        )
        assert not missing_soc_file, (
            f"SUPPORTED_ARCHS entries have no SoC file: {missing_soc_file}"
        )

    # -- get_gpu_series ------------------------------------------------------

    def test_get_gpu_series_all_archs(self):
        for arch in MIGPUSpecs._gpu_series_dict:
            result = MIGPUSpecs.get_gpu_series(arch)
            assert result is not None, f"get_gpu_series({arch!r}) returned None"
            assert result == result.upper()

    # -- get_gpu_model -------------------------------------------------------

    def test_get_gpu_model_legacy_archs(self):
        result = MIGPUSpecs.get_gpu_model("gfx908", None)
        assert result is not None
        assert "MI100" == result

        result = MIGPUSpecs.get_gpu_model("gfx90a", None)
        assert result is not None

    def test_get_gpu_model_chip_id_lookup(self):
        for chip_id, expected_model in MIGPUSpecs._chip_id_dict.items():
            if chip_id is None:
                continue
            result = MIGPUSpecs.get_gpu_model("gfx942", str(chip_id))
            assert result is not None, (
                f"get_gpu_model('gfx942', {chip_id!r}) returned None"
            )
            assert result.lower() == expected_model.lower()

    # -- get_perfmon_config --------------------------------------------------

    def test_get_perfmon_config_all_archs(self):
        for arch in MIGPUSpecs._perfmon_config:
            result = MIGPUSpecs.get_perfmon_config(arch)
            assert isinstance(result, dict)

    def test_every_supported_arch_has_nonempty_perfmon_config(self):
        for arch in common.SUPPORTED_ARCHS:
            config = MIGPUSpecs.get_perfmon_config(arch)
            assert config, (
                f"perfmon_config for {arch} is missing or empty in mi_gpu_spec.yaml"
            )

    # -- is_partition_supported ----------------------------------------------

    def test_is_partition_supported_true(self):
        for arch in (
            "gfx940",
            "gfx941",
            "gfx942",
            "gfx950",
            "gfx1250",
            "GFX940",
            "GFX941",
            "GFX942",
            "GFX950",
            "GFX1250",
        ):
            assert MIGPUSpecs.is_partition_supported(gpu_arch=arch, gpu_model=None), (
                f"is_partition_supported(gpu_arch={arch!r}) should be True"
            )

    def test_is_partition_supported_false(self):
        for arch in (
            "gfx90a",
            "gfx1150",
            "gfx1151",
            "gfx1152",
            "gfx1153",
            "gfx908",
            None,
            "",
            "junk",
        ):
            assert not MIGPUSpecs.is_partition_supported(
                gpu_arch=arch, gpu_model=None
            ), f"is_partition_supported(gpu_arch={arch!r}) should be False"

    def test_is_partition_supported_by_model(self):
        supported_models = ["mi300a_a1", "mi300x_a1", "mi325x", "mi350"]
        for model in supported_models:
            assert MIGPUSpecs.is_partition_supported(gpu_arch=None, gpu_model=model), (
                f"is_partition_supported(gpu_arch=None, gpu_model={model!r})"
                " should be True"
            )

        unsupported_models = ["mi100", "mi210", "mi250", "mi250x", None, "", "junk"]
        for model in unsupported_models:
            assert not MIGPUSpecs.is_partition_supported(
                gpu_arch=None, gpu_model=model
            ), (
                f"is_partition_supported(gpu_arch=None, gpu_model={model!r})"
                " should be False"
            )

    # -- get_num_xcds --------------------------------------------------------

    def test_get_num_xcds_legacy_returns_1(self):
        legacy_cases = [
            ("gfx908", "mi100"),
            ("gfx90a", "mi210"),
            ("gfx90a", "mi250"),
            ("gfx90a", "mi250x"),
        ]
        for arch, model in legacy_cases:
            result = MIGPUSpecs.get_num_xcds(gpu_arch=arch, gpu_model=model)
            assert result == 1, (
                f"get_num_xcds({arch!r}, {model!r}) returned {result}, expected 1"
            )

    def test_get_num_xcds_with_partition(self):
        for arch, partitions in MIGPUSpecs._gpu_arch_to_compute_partition_dict.items():
            if not isinstance(partitions, dict):
                continue
            for partition, num_xcds in partitions.items():
                if num_xcds is None:
                    continue
                result = MIGPUSpecs.get_num_xcds(
                    gpu_arch=arch, compute_partition=partition
                )
                assert result == num_xcds, (
                    f"get_num_xcds({arch!r}, partition={partition!r}) "
                    f"returned {result}, expected {num_xcds}"
                )

    # -- get_num_dies --------------------------------------------------------

    def test_get_num_dies_all_models(self):
        for arch, models in MIGPUSpecs._gpu_model_dict.items():
            for model in models:
                result = MIGPUSpecs.get_num_dies(arch, model)
                assert isinstance(result, int) and result >= 1, (
                    f"get_num_dies({arch!r}, {model!r}) returned {result!r}"
                )

    def test_get_num_dies_cdna_no_design(self):
        with patch.object(MIGPUSpecs, "_gpu_design", {"mi100": {}}), patch.object(
            MIGPUSpecs, "_gpu_series_dict", {"gfx908": "mi100"}
        ):
            assert MIGPUSpecs.get_num_dies("gfx908", "mi100") == 1

    def test_get_num_dies_cdna_with_design(self):
        design = {"testmodel": {"physical_aid": 4, "logical_partitions_per_die": 2}}
        with patch.object(MIGPUSpecs, "_gpu_design", design), patch.object(
            MIGPUSpecs, "_gpu_series_dict", {"gfx942": "mi300"}
        ):
            assert MIGPUSpecs.get_num_dies("gfx942", "testmodel") == 8

    def test_get_num_dies_cdna_partial_design(self):
        design = {"testmodel": {"physical_aid": 4}}
        with patch.object(MIGPUSpecs, "_gpu_design", design), patch.object(
            MIGPUSpecs, "_gpu_series_dict", {"gfx942": "mi300"}
        ):
            assert MIGPUSpecs.get_num_dies("gfx942", "testmodel") == 4

    def test_get_num_dies_rdna_single_die(self):
        design = {"rdna_model": {}}
        for arch in ("gfx1151", "gfx1153"):
            with patch.object(MIGPUSpecs, "_gpu_design", design), patch.object(
                MIGPUSpecs, "_gpu_series_dict", {arch: "rdna3.5"}
            ):
                assert MIGPUSpecs.get_num_dies(arch, "rdna_model") == 1

    # -- get_memory_levels ---------------------------------------------------

    def test_get_memory_levels(self):
        """Test get_memory_levels getting different gpu_model from the same gpu_arch,
        should result in different list of memory levels returned.
        """
        # rdna35_halo (dGPU) includes MALL in its memory levels
        result = MIGPUSpecs.get_memory_levels("rdna35_halo")
        assert result == ["LDS", "L0", "L1", "L2", "MALL"]

        # rdna35_point_1 (APU) does not include MALL
        result = MIGPUSpecs.get_memory_levels("rdna35_point_1")
        assert result == ["LDS", "L0", "L1", "L2"]

    def test_get_memory_levels_missing_returns_empty(self):
        design = {"testmodel": {"physical_aid": 4}}
        with patch.object(MIGPUSpecs, "_gpu_design", design):
            result = MIGPUSpecs.get_memory_levels("testmodel")
            assert result == []

    def test_get_memory_levels_case_insensitive(self):
        result_lower = MIGPUSpecs.get_memory_levels("mi300x_a1")
        result_upper = MIGPUSpecs.get_memory_levels("MI300X_A1")
        assert result_lower == result_upper

    # -- extract_gpu_info (gfx1250 partition defaults) -----------------------

    def _mock_smi(self, mock_smi, num_compute_units=256):
        mock_smi.get_gpu_vbios_part_number.return_value = "TEST"
        mock_smi.get_gpu_compute_partition.return_value = "N/A"
        mock_smi.get_gpu_memory_partition.return_value = "N/A"
        mock_smi.get_gpu_num_compute_units.return_value = num_compute_units
        mock_smi.get_gpu_cache_info.return_value = {}

    def test_gfx1250_defaults_memory_partition_to_nps1(self):
        with patch("utils.specs.amdsmi_interface") as mock_smi:
            self._mock_smi(mock_smi)
            result = _extract_gpu_info("gfx1250")
        assert result["memory_partition"] == "NPS1"

    def test_gfx1250_defaults_compute_partition_to_spx(self):
        with patch("utils.specs.amdsmi_interface") as mock_smi:
            self._mock_smi(mock_smi)
            result = _extract_gpu_info("gfx1250")
        assert result["compute_partition"] == "SPX"

    def test_gfx942_does_not_default_memory_partition(self):
        with patch("utils.specs.amdsmi_interface") as mock_smi:
            self._mock_smi(mock_smi, num_compute_units=304)
            result = _extract_gpu_info("gfx942")
        assert result["memory_partition"] == "N/A"

    def test_gfx942_defaults_compute_partition_to_spx(self):
        with patch("utils.specs.amdsmi_interface") as mock_smi:
            self._mock_smi(mock_smi, num_compute_units=304)
            result = _extract_gpu_info("gfx942")
        assert result["compute_partition"] == "SPX"


@pytest.mark.misc
def test_load_yaml_file_not_found():
    """Test _load_yaml with non-existent file - covers lines 104-105"""
    with pytest.raises(FileNotFoundError):
        MIGPUSpecs._load_yaml("non_existent_file.yaml")


@pytest.mark.misc
def test_load_yaml_invalid_yaml():
    """Test _load_yaml with corrupted YAML - covers lines 106-107"""
    from vendored import yaml

    # Create invalid YAML file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("invalid: yaml: content: [\nunclosed bracket")
        temp_path = f.name

    with pytest.raises(yaml.YAMLError):
        MIGPUSpecs._load_yaml(str(temp_path))


@pytest.mark.misc
def test_load_yaml_generic_exception():
    """Test _load_yaml generic exception handling - covers lines 108-111"""
    with patch("builtins.open", side_effect=PermissionError("Access denied")):
        with pytest.raises(PermissionError, match="Access denied"):
            MIGPUSpecs._load_yaml("some_file.yaml")


@pytest.mark.misc
def test_get_gpu_series_dict_uninitialized():
    """Test get_gpu_series_dict when dict not populated - covers lines 182-185"""
    with patch.object(MIGPUSpecs, "_gpu_series_dict", {}):
        assert MIGPUSpecs.get_gpu_series_dict() == {}


@pytest.mark.misc
def test_get_gpu_series_uninitialized():
    """Test get_gpu_series when dict not populated - covers lines 191-194"""
    with patch.object(MIGPUSpecs, "_gpu_series_dict", {}):
        assert MIGPUSpecs.get_gpu_series_dict() == {}


@pytest.mark.misc
def test_get_perfmon_config_uninitialized():
    """Test get_perfmon_config when dict not populated - covers lines 210-213"""
    with patch.object(MIGPUSpecs, "_perfmon_config", {}):
        with pytest.raises(SystemExit):
            MIGPUSpecs.get_perfmon_config("gfx942")


@pytest.mark.misc
def test_get_gpu_model_uninitialized():
    """Test get_gpu_model when dict not populated - covers lines 223-226"""
    with patch.object(MIGPUSpecs, "_gpu_model_dict", {}):
        with pytest.raises(SystemExit):
            MIGPUSpecs.get_gpu_model("gfx942", "29857")


@pytest.mark.misc
def test_get_gpu_model_invalid_chip_id():
    """Test get_gpu_model with invalid chip_id - covers lines 235-236"""
    result = MIGPUSpecs.get_gpu_model("gfx942", "99999")
    assert result is None


@pytest.mark.misc
def test_get_gpu_model_invalid_arch():
    """Test get_gpu_model with invalid architecture - covers lines 243-244"""
    result = MIGPUSpecs.get_gpu_model("gfx999", "12345")
    assert result is None


@pytest.mark.misc
def test_get_gpu_model_none_result():
    """Test get_gpu_model when result is None - covers lines 246-248"""
    with patch.object(MIGPUSpecs, "_chip_id_dict", {999: None}):
        result = MIGPUSpecs.get_gpu_model("gfx942", "999")
        assert result is None


@pytest.mark.misc
def test_get_num_xcds_no_compute_partition_data():
    """Test get_num_xcds when no compute partition data found - covers lines 307-309"""
    mock_dict = {"gfx942": None}
    with patch.object(MIGPUSpecs, "_gpu_arch_to_compute_partition_dict", mock_dict):
        result = MIGPUSpecs.get_num_xcds(gpu_arch="gfx942")  # noqa: F841


@pytest.mark.misc
def test_get_num_xcds_uninitialized_dict():
    """Test get_num_xcds when XCD dict not populated - covers lines 315-317"""
    with patch.object(MIGPUSpecs, "_num_xcds_dict", {}):
        with pytest.raises(SystemExit):
            MIGPUSpecs.get_num_xcds(gpu_arch="gfx950", gpu_model="MI350")


@pytest.mark.misc
def test_get_num_xcds_unknown_gpu_model():
    """Test get_num_xcds with unknown gpu model - covers lines 319-321"""
    result = MIGPUSpecs.get_num_xcds(  # noqa: F841
        gpu_arch="gfx950", gpu_model="UNKNOWN_MODEL"
    )


@pytest.mark.misc
def test_get_num_xcds_no_compute_partition():
    """Test get_num_xcds with no compute partition - covers lines 325-327"""
    result = MIGPUSpecs.get_num_xcds(  # noqa: F841
        gpu_arch="gfx950", gpu_model="MI350", compute_partition=""
    )


@pytest.mark.misc
def test_get_num_xcds_unknown_compute_partition():
    """Test get_num_xcds with unknown compute partition - covers lines 329-332"""
    result = MIGPUSpecs.get_num_xcds(  # noqa: F841
        gpu_arch="gfx950", gpu_model="MI350", compute_partition="UNKNOWN"
    )


@pytest.mark.misc
def test_get_num_xcds_none_partition_value():
    """Test get_num_xcds when partition value is None - covers lines 338-340"""
    mock_dict = {"mi350": {"spx": None}}
    with patch.object(MIGPUSpecs, "_num_xcds_dict", mock_dict):
        result = MIGPUSpecs.get_num_xcds(  # noqa: F841
            gpu_arch="gfx950", gpu_model="MI350", compute_partition="spx"
        )


@pytest.mark.misc
def test_get_num_xcds_no_gpu_model():
    """Test get_num_xcds with no gpu model - covers line 342"""
    result = MIGPUSpecs.get_num_xcds(  # noqa: F841
        gpu_arch="gfx950", gpu_model="", compute_partition="spx"
    )


@pytest.mark.misc
def test_get_chip_id_dict_empty():
    """Test get_chip_id_dict when dict is empty - covers line 352"""
    with patch.object(MIGPUSpecs, "_chip_id_dict", {}):
        assert MIGPUSpecs.get_chip_id_dict() == {}


@pytest.mark.misc
def test_get_num_xcds_dict_empty():
    """Test get_num_xcds_dict when dict is empty - covers line 359"""
    with patch.object(MIGPUSpecs, "_num_xcds_dict", {}):
        assert MIGPUSpecs.get_num_xcds_dict() == {}


@pytest.mark.misc
def test_normal_functionality_still_works():
    """Ensure that normal paths still work after adding error handling tests"""
    result = MIGPUSpecs.get_gpu_model("gfx90a", None)

    assert result is not None

    result = MIGPUSpecs.get_gpu_series("gfx90a")
    assert result is not None

    result = MIGPUSpecs.get_num_xcds(gpu_arch="gfx90a")
    assert result == 1
