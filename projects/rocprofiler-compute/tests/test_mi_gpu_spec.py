# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT


from pathlib import Path
from unittest.mock import patch

import common

from utils.mi_gpu_spec import MIGPUSpecs


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

    # -- is_partition_supported ----------------------------------------------

    def test_is_partition_supported_true(self):
        for arch in (
            "gfx940",
            "gfx941",
            "gfx942",
            "gfx950",
            "GFX940",
            "GFX941",
            "GFX942",
            "GFX950",
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
        with (
            patch.object(MIGPUSpecs, "_gpu_design", {"mi100": {}}),
            patch.object(MIGPUSpecs, "_gpu_series_dict", {"gfx908": "mi100"}),
        ):
            assert MIGPUSpecs.get_num_dies("gfx908", "mi100") == 1

    def test_get_num_dies_cdna_with_design(self):
        design = {"testmodel": {"physical_aid": 4, "logical_partitions_per_die": 2}}
        with (
            patch.object(MIGPUSpecs, "_gpu_design", design),
            patch.object(MIGPUSpecs, "_gpu_series_dict", {"gfx942": "mi300"}),
        ):
            assert MIGPUSpecs.get_num_dies("gfx942", "testmodel") == 8

    def test_get_num_dies_cdna_partial_design(self):
        design = {"testmodel": {"physical_aid": 4}}
        with (
            patch.object(MIGPUSpecs, "_gpu_design", design),
            patch.object(MIGPUSpecs, "_gpu_series_dict", {"gfx942": "mi300"}),
        ):
            assert MIGPUSpecs.get_num_dies("gfx942", "testmodel") == 4

    def test_get_num_dies_rdna_with_memory_die(self):
        design = {"rdna_model": {"memory_die": 3}}
        with (
            patch.object(MIGPUSpecs, "_gpu_design", design),
            patch.object(MIGPUSpecs, "_gpu_series_dict", {"gfx1151": "navi3"}),
        ):
            assert MIGPUSpecs.get_num_dies("gfx1151", "rdna_model") == 3

    def test_get_num_dies_rdna_no_memory_die(self):
        design = {"rdna_model": {}}
        with (
            patch.object(MIGPUSpecs, "_gpu_design", design),
            patch.object(MIGPUSpecs, "_gpu_series_dict", {"gfx1151": "navi3"}),
        ):
            assert MIGPUSpecs.get_num_dies("gfx1151", "rdna_model") == 1

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
