# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hazard mutation testing: HIP shaders → assembly, mutate ``s_wait*``, run."""

from .benchmark import Benchmarker, PerfCollector
from .kernel_processor import KernelBuilder, KernelBuilderConfig
from .models import DetectionLabel, MutantResult, ShaderReport, WaitInstruction
from .pipeline import (
    DEFAULT_ARCH,
    DEFAULT_EXCLUDED_WAITS,
    DEFAULT_TIMEOUT,
    KernelRunner,
    create_mutant_asm,
    discover_shaders,
    find_tool,
    find_wait_instructions,
    perf_record_command,
    process_shader,
    read_hazard_report,
    run_command,
    select_rocm_path,
    shader_required_archs,
    shader_supports_arch,
)
from .report_writer import (
    compute_summary,
    is_false_negative,
    write_benchmark_report,
    write_csv_report,
    write_json_report,
    write_markdown_report,
    write_terminal_report,
)
from .rocjitsu_runner import RocjitsuRunner, build_rocjitsu_runner

__all__ = [
    "DEFAULT_ARCH",
    "DEFAULT_EXCLUDED_WAITS",
    "DEFAULT_TIMEOUT",
    "Benchmarker",
    "PerfCollector",
    "DetectionLabel",
    "KernelBuilder",
    "KernelBuilderConfig",
    "KernelRunner",
    "MutantResult",
    "ShaderReport",
    "RocjitsuRunner",
    "WaitInstruction",
    "build_rocjitsu_runner",
    "compute_summary",
    "create_mutant_asm",
    "discover_shaders",
    "find_tool",
    "find_wait_instructions",
    "is_false_negative",
    "perf_record_command",
    "process_shader",
    "read_hazard_report",
    "run_command",
    "select_rocm_path",
    "shader_required_archs",
    "shader_supports_arch",
    "write_benchmark_report",
    "write_csv_report",
    "write_json_report",
    "write_markdown_report",
    "write_terminal_report",
]
