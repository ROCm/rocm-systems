# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hazard mutation testing: HIP shaders → assembly, mutate ``s_wait*``, run."""

from .benchmark import Benchmarker, PerfCollector
from .kernel_processor import KernelBuilder, KernelBuilderConfig
from .models import DetectionLabel, MutantResult, ShaderReport, WaitInstruction
from .report_writer import (
    compute_summary,
    is_false_negative,
    write_benchmark_report,
    write_csv_report,
    write_json_report,
    write_markdown_report,
    write_terminal_report,
)

# Pipeline and CLI helpers live in __main__; import lazily avoids any future cycles.
from . import __main__ as _main

RocjitsuRunner = _main.RocjitsuRunner
build_rocjitsu_runner = _main.build_rocjitsu_runner
create_mutant_asm = _main.create_mutant_asm
discover_shaders = _main.discover_shaders
find_tool = _main.find_tool
find_wait_instructions = _main.find_wait_instructions
perf_record_command = _main.perf_record_command
process_shader = _main.process_shader
read_hazard_report = _main.read_hazard_report
run_command = _main.run_command

DEFAULT_ARCH = _main.DEFAULT_ARCH
DEFAULT_EXCLUDED_WAITS = _main.DEFAULT_EXCLUDED_WAITS
DEFAULT_TIMEOUT = _main.DEFAULT_TIMEOUT

__all__ = [
    "DEFAULT_ARCH",
    "DEFAULT_EXCLUDED_WAITS",
    "DEFAULT_TIMEOUT",
    "Benchmarker",
    "PerfCollector",
    "DetectionLabel",
    "KernelBuilder",
    "KernelBuilderConfig",
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
    "write_benchmark_report",
    "write_csv_report",
    "write_json_report",
    "write_markdown_report",
    "write_terminal_report",
]
