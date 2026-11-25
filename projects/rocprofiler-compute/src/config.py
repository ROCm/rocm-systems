# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from pathlib import Path

# NB: Creating a new module to share global vars across modules
rocprof_compute_home = Path(__file__).resolve().parent
analysis_configs_dir = rocprof_compute_home / "rocprof_compute_soc" / "analysis_configs"
# Built by tools/instruction_pipeline_generator.py, read by PC sampling analysis.
instruction_pipelines_path = (
    rocprof_compute_home / "pc_sampling" / "instruction_pipelines.yaml"
)
PROJECT_NAME = "rocprofiler-compute"

TIME_UNITS = {"s": 10**9, "ms": 10**6, "us": 10**3, "ns": 1}
