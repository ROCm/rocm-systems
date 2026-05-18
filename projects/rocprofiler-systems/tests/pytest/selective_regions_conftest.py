# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared fixtures for selective-region pytest modules."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import check_use_perfetto, check_use_rocpd
from rocprofsys.config import RocprofsysConfig
from rocprofsys.region_filter_leakage import (
    DEFAULT_LEAKAGE_CHECKS,
    LeakageCheck,
    validate_region_filter_leakage,
)
from rocprofsys.selective_region_config import (
    find_preset_json_path,
    generate_selective_region_xml,
    write_flat_config,
)
from rocprofsys.validators import validate_region_filter_rocpd_leakage


@pytest.fixture
def selective_region_flat_config(test_output_dir) -> Path:
    """Flat text config with ROCPROFSYS_SELECTED_REGIONS=Region1."""
    path = test_output_dir / "selective-region-flat.cfg"
    return write_flat_config(path)


@pytest.fixture
def selective_region_xml_config(rocprof_config: RocprofsysConfig, test_output_dir) -> Path:
    """XML config from rocprof-sys-avail with selective-region overrides."""
    path = test_output_dir / "selective-region.xml"
    return generate_selective_region_xml(rocprof_config, path)


@pytest.fixture
def selective_region_preset_json_path(rocprof_config: RocprofsysConfig) -> Path:
    """Preset JSON with tracing.region for config-file tests."""
    found = find_preset_json_path(rocprof_config)
    if found is None:
        pytest.skip("rocprof-sys-selected-region1.cfg not found")
    return found


@pytest.fixture
def selective_region_env() -> dict[str, str]:
    """Environment variables for selective region tests."""
    return {
        "ROCPROFSYS_ROCM_DOMAINS": "hip_runtime_api,marker_api,kernel_dispatch,marker_core_range_api",
    }


@pytest.fixture
def sys_run_base_env() -> dict[str, str]:
    """Base rocprof-sys-run env: CI/sampling off, ROCPD on."""
    return {
        "ROCPROFSYS_CI": "OFF",
        "ROCPROFSYS_USE_SAMPLING": "OFF",
        "ROCPROFSYS_USE_PROCESS_SAMPLING": "OFF",
        "ROCPROFSYS_USE_ROCPD": "ON",
    }


@pytest.fixture
def config_sys_run_env(selective_region_env, sys_run_base_env) -> dict[str, str]:
    """Region domains + sys_run_base_env for test_selective_regions_config.py."""
    return merge_selective_env("sys_run", selective_region_env, sys_run_base_env)


def merge_selective_env(
    mode: str, base: dict[str, str], sys_run_base: dict[str, str]
) -> dict[str, str]:
    """Merge base env with sys_run_base_env when mode is sys_run."""
    env = base.copy()
    if mode == "sys_run":
        env.update(sys_run_base)
    return env


@pytest.fixture
def assert_no_leakage_outside_regions(subtests, record_subtest_failure, request):
    """Assert Perfetto/ROCPD events are inside ROCPROFSYS_SELECTED_REGIONS windows."""

    def _assert(
        result,
        selected_regions: str,
        *,
        checks: tuple[LeakageCheck, ...] = DEFAULT_LEAKAGE_CHECKS,
        check_counters: bool = True,
        check_rocpd: bool = False,
        counter_max_outside: int = 0,
    ) -> None:
        if not check_use_perfetto():
            pytest.skip("Perfetto is disabled")

        perfetto = result.perfetto_file
        if not perfetto or not perfetto.exists():
            pytest.fail(f"Perfetto trace file not found for {result.output_dir}")

        for check in checks:
            subtest_name = f"No leakage: {check.label}"
            with subtests.test(subtest_name):
                validation = validate_region_filter_leakage(
                    perfetto,
                    selected_regions,
                    checks=(check,),
                    check_counters=False,
                )
                if not validation.is_valid:
                    record_subtest_failure(subtest_name)
                    pytest.fail(
                        f"Region-filter leakage ({check.label}):\n{validation.message}",
                        pytrace=False,
                    )

        if check_counters:
            subtest_name = "No leakage: GPU PMC / AMD-SMI counters"
            with subtests.test(subtest_name):
                validation = validate_region_filter_leakage(
                    perfetto,
                    selected_regions,
                    checks=(),
                    check_counters=True,
                    counter_max_outside=counter_max_outside,
                )
                if not validation.is_valid:
                    record_subtest_failure(subtest_name)
                    pytest.fail(
                        f"Region-filter leakage (counters):\n{validation.message}",
                        pytrace=False,
                    )

        if check_rocpd:
            subtest_name = "No leakage: ROCPD timestamps"
            with subtests.test(subtest_name):
                if not check_use_rocpd():
                    pytest.skip("ROCpd is disabled")
                rocpd_file = result.rocpd_file
                if rocpd_file is None or not rocpd_file.exists():
                    record_subtest_failure(subtest_name)
                    pytest.fail("ROCpd database not found", pytrace=False)
                validation = validate_region_filter_rocpd_leakage(
                    rocpd_file, selected_regions
                )
                if not validation.is_valid:
                    record_subtest_failure(subtest_name)
                    pytest.fail(
                        f"Region-filter ROCPD leakage:\n{validation.message}",
                        pytrace=False,
                    )

    return _assert
