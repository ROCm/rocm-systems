# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Integration tests for profile output paths and directory templating."""

import os
import socket
from pathlib import Path

import common

from tests.integration import common as integration_common
from tests.integration.common import (
    CSVS,
    GPU_MODEL,
    SLURM_RANK_VAR,
    SLURM_SIZE_VAR,
    MockProfiler,
    clear_rank_env,
    config,
    mock_generate_machine_specs,
    mock_load_soc_specs,
    num_devices,
    num_kernels,
)

WAVE_COUNTERS = {
    "gfx908": "SQ_WAVES",
    "gfx90a": "SQ_WAVES",
    "gfx940": "SQ_WAVES",
    "gfx941": "SQ_WAVES",
    "gfx942": "SQ_WAVES",
    "gfx950": "SQ_WAVES",
    "gfx1150": "SQ_WAVES_sum",
    "gfx1151": "SQ_WAVES_sum",
    "gfx1152": "SQ_WAVES_sum",
    "gfx1153": "SQ_WAVES_sum",
    "gfx1250": "SPI_ALL_WAVE_sum",
}

VALU_ARITHMETIC_COUNTERS = (
    "SQ_INSTS_VALU_ADD_F16",
    "SQ_INSTS_VALU_MUL_F16",
    "SQ_INSTS_VALU_TRANS_F16",
    "SQ_INSTS_VALU_FMA_F16",
    "SQ_INSTS_VALU_ADD_F32",
    "SQ_INSTS_VALU_MUL_F32",
    "SQ_INSTS_VALU_TRANS_F32",
    "SQ_INSTS_VALU_FMA_F32",
    "SQ_INSTS_VALU_ADD_F64",
    "SQ_INSTS_VALU_MUL_F64",
    "SQ_INSTS_VALU_TRANS_F64",
    "SQ_INSTS_VALU_FMA_F64",
)

ROCFLOP_COUNTERS = {
    "gfx908": (),  # This architecture has no defined FLOP metric for block 2.1.0.
    "gfx90a": VALU_ARITHMETIC_COUNTERS,
    "gfx940": VALU_ARITHMETIC_COUNTERS,
    "gfx941": VALU_ARITHMETIC_COUNTERS,
    "gfx942": VALU_ARITHMETIC_COUNTERS,
    "gfx950": VALU_ARITHMETIC_COUNTERS,
    "gfx1150": ("SQ_INSTS_VALU_sum",),
    "gfx1151": ("SQ_INSTS_VALU_sum",),
    "gfx1152": ("SQ_INSTS_VALU_sum",),
    "gfx1153": ("SQ_INSTS_VALU_sum",),
    "gfx1250": ("SP_VALU_FLOPS_FP16_sum",),
}


def test_path(binary_handler_profile_rocprof_compute):
    workload_dir = common.get_output_dir()
    binary_handler_profile_rocprof_compute(config, workload_dir)

    file_dict = integration_common.check_csv_files(
        workload_dir, num_devices, num_kernels
    )

    assert sorted(list(file_dict.keys())) == CSVS

    sysinfo = common.check_sysinfo(Path(workload_dir) / "sysinfo.csv")
    assert set(sysinfo["gpu_arch"]) == {integration_common.gpu_soc()[0]}
    wave_counter = WAVE_COUNTERS[sysinfo["gpu_arch"].iloc[0]]
    results = common.check_counter_results(
        common.read_counter_results(workload_dir), required_counters=(wave_counter,)
    )
    waves = results.loc[results["Counter_Name"] == wave_counter, "Counter_Value"]
    assert (waves > 0).all()
    common.clean_output_dir(config["cleanup"], workload_dir)


def test_path_rocflop(binary_handler_profile_rocprof_compute):
    # Test whether multiprocess workloads like rocflop are handled correctly
    workload_dir = common.get_output_dir()
    options = ["--block", "2.1.0"]
    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=False,
        app_name="rocflop",
    )
    integration_common.check_csv_files(workload_dir, num_devices, num_kernels)
    sysinfo = common.check_sysinfo(Path(workload_dir) / "sysinfo.csv")
    arch = sysinfo["gpu_arch"].iloc[0]
    results = common.check_counter_results(
        common.read_counter_results(workload_dir),
        required_counters=ROCFLOP_COUNTERS[arch],
    )
    if arch in {"gfx90a", "gfx940", "gfx941", "gfx942", "gfx950"}:
        fp16_fma = results.loc[
            results["Counter_Name"] == "SQ_INSTS_VALU_FMA_F16", "Counter_Value"
        ]
        assert (fp16_fma > 0).any()
    common.clean_output_dir(config["cleanup"], workload_dir)


def test_path_no_native(binary_handler_profile_rocprof_compute):
    workload_dir = common.get_output_dir()
    options = ["--no-native-tool"]
    binary_handler_profile_rocprof_compute(config, workload_dir, options)

    file_dict = integration_common.check_csv_files(
        workload_dir, num_devices, num_kernels
    )

    assert sorted(list(file_dict.keys())) == CSVS

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_path_rocpd(
    binary_handler_profile_rocprof_compute, binary_handler_analyze_rocprof_compute
):
    workload_dir = common.get_output_dir()
    binary_handler_profile_rocprof_compute(config, workload_dir)

    # Validate profile outputs (results_*.csv for rocpd format)
    integration_common.check_csv_files(workload_dir, num_devices, num_kernels)

    code = binary_handler_analyze_rocprof_compute(["analyze", "--path", workload_dir])
    assert code == 0

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_output_directory_hostname(binary_handler_profile_rocprof_compute, monkeypatch):
    """Test that %hostname% placeholder is replaced with the actual hostname."""
    from rocprof_compute_base import RocProfCompute

    hostname = "test_node"

    monkeypatch.setattr(RocProfCompute, "create_profiler", lambda self: MockProfiler())
    monkeypatch.setattr(socket, "gethostname", lambda: hostname)

    workload_base_dir = common.get_output_dir(param_id="hostname")
    workload_dir = os.path.join(workload_base_dir, "%hostname%")

    binary_handler_profile_rocprof_compute(config, workload_dir)

    workload_dir = workload_dir.replace("%hostname%", hostname)
    assert os.path.exists(workload_dir)

    common.clean_output_dir(config["cleanup"], workload_base_dir)


def test_output_directory_gpumodel(binary_handler_profile_rocprof_compute, monkeypatch):
    """Test that %gpumodel% placeholder is replaced with the GPU model name."""
    from rocprof_compute_base import RocProfCompute

    monkeypatch.setattr(RocProfCompute, "create_profiler", lambda self: MockProfiler())
    monkeypatch.setattr(
        RocProfCompute, "generate_machine_specs", mock_generate_machine_specs
    )
    monkeypatch.setattr(RocProfCompute, "load_soc_specs", mock_load_soc_specs)

    workload_base_dir = common.get_output_dir(param_id="gpumodel")
    workload_dir = os.path.join(workload_base_dir, "%gpumodel%_output")

    binary_handler_profile_rocprof_compute(config, workload_dir)

    workload_dir = workload_dir.replace("%gpumodel%", GPU_MODEL)
    assert os.path.exists(workload_dir)

    common.clean_output_dir(config["cleanup"], workload_base_dir)


def test_output_directory_rank_ignored_without_mpi(
    binary_handler_profile_rocprof_compute, monkeypatch
):
    """Test that %rank% is ignored when no MPI rank env var is set."""
    from rocprof_compute_base import RocProfCompute

    clear_rank_env(monkeypatch, SLURM_RANK_VAR, SLURM_SIZE_VAR)
    monkeypatch.setattr(RocProfCompute, "create_profiler", lambda self: MockProfiler())

    workload_base_dir = common.get_output_dir(param_id="no_rank")
    workload_dir = os.path.join(workload_base_dir, "%rank%_output")

    binary_handler_profile_rocprof_compute(config, workload_dir)

    workload_dir = workload_dir.replace("%rank%", "")
    assert os.path.exists(workload_dir)

    common.clean_output_dir(config["cleanup"], workload_base_dir)


def test_output_directory_rank_replaced_with_mpi(
    binary_handler_profile_rocprof_compute, monkeypatch
):
    """Test that %rank% is replaced with the rank value when SLURM env vars are set."""
    from rocprof_compute_base import RocProfCompute

    rank = "3"

    monkeypatch.setattr(RocProfCompute, "create_profiler", lambda self: MockProfiler())
    monkeypatch.setenv(SLURM_RANK_VAR, rank)
    monkeypatch.setenv(SLURM_SIZE_VAR, "4")

    workload_base_dir = common.get_output_dir(param_id="rank_env_SLURM")
    workload_dir = os.path.join(workload_base_dir, "%rank%_output")

    binary_handler_profile_rocprof_compute(config, workload_dir)

    workload_dir = workload_dir.replace("%rank%", rank)
    assert os.path.exists(workload_dir)

    common.clean_output_dir(config["cleanup"], workload_base_dir)
    clear_rank_env(monkeypatch, SLURM_RANK_VAR, SLURM_SIZE_VAR)


def test_output_directory_env_variable(
    binary_handler_profile_rocprof_compute, monkeypatch
):
    """Test that %env{VAR}% is replaced with the environment variable value."""
    from rocprof_compute_base import RocProfCompute

    monkeypatch.setenv("ENV_1", "custom_env")
    monkeypatch.setattr(RocProfCompute, "create_profiler", lambda self: MockProfiler())

    workload_base_dir = common.get_output_dir(param_id="env")
    workload_dir = os.path.join(workload_base_dir, "%env{ENV_1}%")

    binary_handler_profile_rocprof_compute(config, workload_dir)

    workload_dir = workload_dir.replace("%env{ENV_1}%", "custom_env")
    assert os.path.exists(workload_dir)

    common.clean_output_dir(config["cleanup"], workload_base_dir)
    monkeypatch.delenv("ENV_1", raising=False)


def test_output_directory_env_variable_unset(
    binary_handler_profile_rocprof_compute, monkeypatch
):
    """Test that %env{VAR}% resolves to empty string when the var is unset."""
    from rocprof_compute_base import RocProfCompute

    monkeypatch.delenv("ENV_2", raising=False)
    monkeypatch.setattr(RocProfCompute, "create_profiler", lambda self: MockProfiler())

    workload_base_dir = common.get_output_dir(param_id="no_env")
    workload_dir = os.path.join(workload_base_dir, "%env{ENV_2}%")

    binary_handler_profile_rocprof_compute(config, workload_dir)
    workload_dir = workload_dir.replace("%env{ENV_2}%", "")

    assert os.path.exists(workload_dir)
    common.clean_output_dir(config["cleanup"], workload_base_dir)


def test_output_directory_all_placeholders_combined(
    binary_handler_profile_rocprof_compute, monkeypatch
):
    """Test that all placeholders work together in a single path."""
    from rocprof_compute_base import RocProfCompute

    hostname = "test_node"
    rank = "3"

    monkeypatch.setattr(RocProfCompute, "create_profiler", lambda self: MockProfiler())
    monkeypatch.setattr(socket, "gethostname", lambda: hostname)
    monkeypatch.setattr(
        RocProfCompute, "generate_machine_specs", mock_generate_machine_specs
    )
    monkeypatch.setattr(RocProfCompute, "load_soc_specs", mock_load_soc_specs)
    monkeypatch.setenv("ENV_1", "custom_env")
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", rank)
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")

    workload_base_dir = common.get_output_dir(param_id="host_gpu_env_rank")
    workload_dir = os.path.join(
        workload_base_dir,
        "%hostname%_%gpumodel%_%env{ENV_1}%_%rank%_output",
    )

    binary_handler_profile_rocprof_compute(config, workload_dir)

    workload_dir = (
        workload_dir
        .replace("%hostname%", hostname)
        .replace("%gpumodel%", GPU_MODEL)
        .replace("%env{ENV_1}%", "custom_env")
        .replace("%rank%", rank)
    )
    assert os.path.exists(workload_dir)

    common.clean_output_dir(config["cleanup"], workload_base_dir)
    monkeypatch.delenv("OMPI_COMM_WORLD_RANK", raising=False)
    monkeypatch.delenv("ENV_1", raising=False)


def test_output_directory_default_with_rank(
    binary_handler_profile_rocprof_compute, monkeypatch
):
    """
    Test that rank is appended to the default output
    directory when MPI rank is set.
    """
    from rocprof_compute_base import RocProfCompute

    rank = "3"
    original_cwd = os.getcwd()

    monkeypatch.setattr(RocProfCompute, "create_profiler", lambda self: MockProfiler())
    monkeypatch.setattr(
        RocProfCompute, "generate_machine_specs", mock_generate_machine_specs
    )
    monkeypatch.setattr(RocProfCompute, "load_soc_specs", mock_load_soc_specs)
    monkeypatch.setenv("PMI_RANK", rank)
    monkeypatch.setenv("PMI_SIZE", "4")

    workload_base_dir = common.get_output_dir(param_id="rank_def_dir")
    p = Path(workload_base_dir)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    os.chdir(workload_base_dir)

    binary_handler_profile_rocprof_compute(
        config, workload_dir=workload_base_dir, workload_dir_type="default"
    )

    workload_dir = os.path.join(
        workload_base_dir,
        "workloads",
        "app_1",
        rank,
    )

    os.chdir(original_cwd)

    assert os.path.exists(workload_dir)

    common.clean_output_dir(config["cleanup"], workload_base_dir)
    monkeypatch.delenv("PMI_RANK", raising=False)


def test_output_directory_default_without_rank(
    binary_handler_profile_rocprof_compute, monkeypatch
):
    """Test default output directory layout when no MPI rank is set."""
    from rocprof_compute_base import RocProfCompute

    clear_rank_env(monkeypatch, SLURM_RANK_VAR, SLURM_SIZE_VAR)
    original_cwd = os.getcwd()

    monkeypatch.setattr(RocProfCompute, "create_profiler", lambda self: MockProfiler())
    monkeypatch.setattr(
        RocProfCompute, "generate_machine_specs", mock_generate_machine_specs
    )
    monkeypatch.setattr(RocProfCompute, "load_soc_specs", mock_load_soc_specs)

    workload_base_dir = common.get_output_dir(param_id="no_rank_def_dir")
    p = Path(workload_base_dir)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    os.chdir(workload_base_dir)

    binary_handler_profile_rocprof_compute(
        config, workload_dir=workload_base_dir, workload_dir_type="default"
    )

    os.chdir(original_cwd)

    workload_dir = os.path.join(
        workload_base_dir,
        "workloads",
        "app_1",
        GPU_MODEL,
    )
    assert os.path.exists(workload_dir)

    common.clean_output_dir(config["cleanup"], workload_base_dir)


def test_output_directory_no_name_with_output_dir(
    binary_handler_profile_rocprof_compute, monkeypatch
):
    """Test that --output-directory works without --name."""
    from rocprof_compute_base import RocProfCompute

    monkeypatch.setattr(RocProfCompute, "create_profiler", lambda self: MockProfiler())
    monkeypatch.setattr(
        RocProfCompute, "generate_machine_specs", mock_generate_machine_specs
    )
    monkeypatch.setattr(RocProfCompute, "load_soc_specs", mock_load_soc_specs)

    workload_dir = common.get_output_dir(param_id="dir_no_name")

    binary_handler_profile_rocprof_compute(
        config, workload_dir=workload_dir, skip_app_name=True
    )

    assert os.path.exists(workload_dir)

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_output_directory_no_name_no_output_dir(
    binary_handler_profile_rocprof_compute, monkeypatch
):
    """Test that profiling fails when neither --name nor --output-directory is given."""
    from rocprof_compute_base import RocProfCompute

    monkeypatch.setattr(RocProfCompute, "create_profiler", lambda self: MockProfiler())
    monkeypatch.setattr(
        RocProfCompute, "generate_machine_specs", mock_generate_machine_specs
    )
    monkeypatch.setattr(RocProfCompute, "load_soc_specs", mock_load_soc_specs)

    workload_dir = common.get_output_dir(param_id="no_name_no_dir")

    error_code = binary_handler_profile_rocprof_compute(
        config,
        skip_app_name=True,
        workload_dir=workload_dir,
        check_success=False,
        workload_dir_type="default",
    )

    assert error_code == 1

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_comprehensive_error_paths():
    """Simplified test for error path coverage"""

    from utils.parser import build_comparable_columns

    columns = build_comparable_columns("ms")
    expected = [
        "Count(ms)",
        "Sum(ms)",
        "Mean(ms)",
        "Median(ms)",
        "Standard Deviation(ms)",
    ]
    for expected_col in expected:
        assert expected_col in columns
