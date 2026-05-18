# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI, config-file, instrumentation, and HPC workflow tests for region filtering."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import RocprofsysTest, check_use_perfetto
from rocprofsys.selective_region_config import (
    combined_output_text,
    output_has_region_filter_log,
    output_indicates_successful_trace,
)
pytest_plugins = ["selective_regions_conftest"]

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.selective_regions,
    pytest.mark.selective_regions_config,
    pytest.mark.timeout(180),
    pytest.mark.rocm,
]

_REGION1_KERNELS_PASS = [
    "CodeBlock_B",
    "CodeBlock_C",
    "CodeBlock_D",
    "CodeBlock_F",
]
_REGION1_KERNELS_FAIL = ["CodeBlock_A", "CodeBlock_E", "CodeBlock_G"]
_ALL_CODEBLOCKS = [
    "CodeBlock_A",
    "CodeBlock_B",
    "CodeBlock_C",
    "CodeBlock_D",
    "CodeBlock_E",
    "CodeBlock_F",
    "CodeBlock_G",
]


@pytest.fixture
def selective_regions_rocpd_rules(validation_rules_dir) -> list[Path]:
    rules = validation_rules_dir / "selective-regions" / "validation-rules.json"
    return [rules]


@pytest.mark.class_name("selective-region-config")
class TestSelectiveRegionsConfig(RocprofsysTest):
    def _assert_region1_filtered(self, result, *, check_rocpd: bool = True) -> None:
        self.assert_regex(result)
        if not check_use_perfetto():
            return
        self.assert_perfetto(
            result,
            subtest_name="Region1 filtered kernels",
            categories=["rocm_hip_stream"],
            pass_regex=_REGION1_KERNELS_PASS,
            fail_regex=_REGION1_KERNELS_FAIL,
        )
        self.assert_perfetto(
            result,
            subtest_name="Region1 filtered markers",
            categories=["rocm_marker_api"],
            pass_regex=["Region1", "Region2"],
            fail_regex=["Region3"],
        )

    def test_cli_selected_regions_flag(self, config_sys_run_env):
        """--selected-regions on rocprof-sys-run."""
        env = config_sys_run_env.copy()
        env.pop("ROCPROFSYS_SELECTED_REGIONS", None)
        result = self.run_test(
            "sys_run",
            "selective_region",
            env=env,
            sysrun_args=["--selected-regions", "Region1"],
            check_target_arch=True,
        )
        self._assert_region1_filtered(result, check_rocpd=False)

    @pytest.mark.xfail(
        reason="preset JSON tracing.region leaves profiler in Init with rocprof-sys-run",
        strict=False,
    )
    def test_preset_json_config_tracing_region(
        self, config_sys_run_env, selective_region_preset_json_path
    ):
        """ROCPROFSYS_CONFIG_FILE with hierarchical tracing.region."""
        env = config_sys_run_env.copy()
        env.pop("ROCPROFSYS_SELECTED_REGIONS", None)
        env["ROCPROFSYS_CONFIG_FILE"] = str(selective_region_preset_json_path)
        result = self.run_test(
            "sys_run",
            "selective_region",
            env=env,
            check_target_arch=True,
        )
        self._assert_region1_filtered(result, check_rocpd=False)

    def test_flat_text_config_selected_regions(
        self, config_sys_run_env, selective_region_flat_config
    ):
        """Flat cfg file sets ROCPROFSYS_SELECTED_REGIONS."""
        env = config_sys_run_env.copy()
        env.pop("ROCPROFSYS_SELECTED_REGIONS", None)
        env["ROCPROFSYS_CONFIG_FILE"] = str(selective_region_flat_config)
        result = self.run_test(
            "sys_run",
            "selective_region",
            env=env,
            check_target_arch=True,
        )
        combined = combined_output_text(result)
        assert output_has_region_filter_log(combined)
        self._assert_region1_filtered(result, check_rocpd=False)

    def test_xml_config_trace_completes(
        self, config_sys_run_env, selective_region_xml_config
    ):
        """XML cfg from rocprof-sys-avail: run should finish and write Perfetto."""
        env = config_sys_run_env.copy()
        env.pop("ROCPROFSYS_SELECTED_REGIONS", None)
        env["ROCPROFSYS_CONFIG_FILE"] = str(selective_region_xml_config)
        result = self.run_test(
            "sys_run",
            "selective_region",
            env=env,
            check_target_arch=True,
        )
        self.assert_regex(result)
        combined = combined_output_text(result)
        assert output_indicates_successful_trace(combined)
        assert result.perfetto_file is not None
        self.assert_file_exists([result.perfetto_file], subtest_name="XML Perfetto trace")

    @pytest.mark.xfail(
        reason="ROCPROFSYS_SELECTED_REGIONS in XML not applied before sdk init (v1.6.0)",
        strict=False,
    )
    def test_xml_config_selected_regions_filter(
        self, config_sys_run_env, selective_region_xml_config
    ):
        """XML cfg should honor ROCPROFSYS_SELECTED_REGIONS=Region1."""
        env = config_sys_run_env.copy()
        env.pop("ROCPROFSYS_SELECTED_REGIONS", None)
        env["ROCPROFSYS_CONFIG_FILE"] = str(selective_region_xml_config)
        result = self.run_test(
            "sys_run",
            "selective_region",
            env=env,
            check_target_arch=True,
        )
        combined = combined_output_text(result)
        assert output_has_region_filter_log(combined)
        self._assert_region1_filtered(result, check_rocpd=False)

    def test_binary_rewrite_with_region_filter(self, config_sys_run_env):
        env = config_sys_run_env.copy()
        env["ROCPROFSYS_SELECTED_REGIONS"] = "Region1"
        result = self.run_test(
            "binary_rewrite",
            "selective_region",
            env=env,
            check_target_arch=True,
        )
        self._assert_region1_filtered(result, check_rocpd=False)

    @pytest.mark.ci_disable("all")
    def test_runtime_instrument_with_region_filter(self, config_sys_run_env):
        env = config_sys_run_env.copy()
        env["ROCPROFSYS_SELECTED_REGIONS"] = "Region1"
        inst = self.run_test(
            "runtime_instrument",
            "selective_region",
            env=env,
            check_target_arch=True,
            skip_on_error=True,
        )
        self.assert_regex(inst)
        result = self.run_test(
            "sys_run",
            "selective_region",
            env=env,
            check_target_arch=True,
        )
        self._assert_region1_filtered(result, check_rocpd=False)

    def test_unknown_region_name_excludes_kernels(self, config_sys_run_env):
        env = config_sys_run_env.copy()
        env["ROCPROFSYS_SELECTED_REGIONS"] = "BadRegion"
        result = self.run_test(
            "sys_run",
            "selective_region",
            env=env,
            check_target_arch=True,
        )
        self.assert_regex(result)
        if check_use_perfetto():
            self.assert_perfetto(
                result,
                subtest_name="BadRegion — no CodeBlock kernels",
                categories=["rocm_hip_stream"],
                fail_regex=_ALL_CODEBLOCKS,
            )

    @pytest.mark.skip(reason="manual Perfetto UI check")
    def test_perfetto_ui_not_automated(self):
        pass

    @pytest.mark.openmp
    def test_openmp_hotphase_region_filter(self, config_sys_run_env):
        env = config_sys_run_env.copy()
        env["ROCPROFSYS_USE_OMPT"] = "ON"
        env["ROCPROFSYS_SELECTED_REGIONS"] = "HotPhase"
        result = self.run_test(
            "sys_run",
            "omp_app",
            env=env,
            check_target_arch=True,
        )
        self.assert_regex(result)
        if check_use_perfetto():
            self.assert_perfetto(
                result,
                subtest_name="HotPhase filtered kernels",
                categories=["rocm_kernel_dispatch"],
                pass_regex=["CodeBlock_Hot"],
                fail_regex=["CodeBlock_Warmup", "CodeBlock_Cooldown"],
            )
            self.assert_perfetto(
                result,
                subtest_name="HotPhase marker present",
                categories=["rocm_marker_api"],
                pass_regex=["HotPhase"],
            )

    @pytest.mark.mpi_optional("selective_region")
    def test_mpi_multirank_region_filter(self, config_sys_run_env):
        env = config_sys_run_env.copy()
        env["ROCPROFSYS_USE_MPIP"] = "ON"
        env["ROCPROFSYS_SELECTED_REGIONS"] = "Region1"
        result = self.run_test(
            "sys_run",
            "selective_region",
            env=env,
            launcher="mpi",
            num_procs=2,
            check_target_arch=True,
        )
        self._assert_region1_filtered(result, check_rocpd=False)

    def test_pushpop_ranges_ignore_region_filter(self, config_sys_run_env):
        env = config_sys_run_env.copy()
        env["ROCPROFSYS_SELECTED_REGIONS"] = "Region1"
        result = self.run_test(
            "sys_run",
            "selective_region_pushpop",
            env=env,
            check_target_arch=True,
        )
        self.assert_regex(result)
        if check_use_perfetto():
            self.assert_perfetto(
                result,
                subtest_name="Push/Pop — filter ignored (no Region1 kernels)",
                categories=["rocm_hip_stream"],
                fail_regex=_REGION1_KERNELS_PASS,
            )

    @pytest.mark.rocpd("config_sys_run_env")
    def test_rocpd_with_region_filter(
        self,
        config_sys_run_env,
        selective_regions_rocpd_rules,
        assert_no_leakage_outside_regions,
    ):
        env = config_sys_run_env.copy()
        env["ROCPROFSYS_SELECTED_REGIONS"] = "Region1"
        result = self.run_test(
            "sys_run",
            "selective_region",
            env=env,
            check_target_arch=True,
        )
        self._assert_region1_filtered(result, check_rocpd=False)
        self.assert_rocpd(result, rules_files=selective_regions_rocpd_rules)
        assert_no_leakage_outside_regions(
            result,
            "Region1",
            check_counters=True,
            check_rocpd=True,
        )
