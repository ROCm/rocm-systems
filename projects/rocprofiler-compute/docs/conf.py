# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re

import yaml

with open("../VERSION", encoding="utf-8") as f:
    match = re.search(r"([0-9.]+)[^0-9.]+", f.read())
    if not match:
        raise ValueError("VERSION not found!")
    version_number = match[1]

# project info
project = "ROCm Compute Profiler"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved."
version = version_number
release = version_number

extensions = [
    "rocm_docs",
    "sphinx.ext.extlinks",
    "sphinxcontrib.datatemplates",
    "sphinx_jinja",
]
html_theme = "rocm_docs_theme"
html_theme_options = {"flavor": "rocm"}
html_title = f"{project} {version_number} documentation"
exclude_patterns = ["archive", "*/includes"]

html_static_path = ["sphinx/static/css"]
html_css_files = ["o_custom.css"]

# Load per-arch CDNA metrics YAMLs
arch_metrics = {}
for arch in ["gfx908", "gfx90a", "gfx942", "gfx950"]:
    with open(f"data/metrics/{arch}_metrics.yaml") as f:
        arch_metrics[arch] = yaml.safe_load(f)

# CDNA section name mapping: jinja context id -> YAML section name
cdna_section_map = {
    "wavefront-launch-stats": "Wavefront launch stats",
    "wavefront-runtime-stats": "Wavefront runtime stats",
    "instruction-mix": "Overall instruction mix",
    "valu-arith-instruction-mix": "VALU arithmetic instruction mix",
    "matrix-instruction-mix": "Matrix instruction mix",
    "compute-speed-of-light": "Compute Speed-of-Light",
    "pipeline-stats": "Pipeline statistics",
    "arithmetic-operations": "Arithmetic operations",
    "lds-sol": "LDS Speed-of-Light",
    "lds-stats": "LDS Statistics",
    "vl1d-sol": "vL1D Speed-of-Light",
    "ta-busy-stall": "Busy / stall metrics",
    "ta-instruction-counts": "Instruction counts",
    "ta-spill-stack": "Spill / stack metrics",
    "desc-utcl1": "L1 Unified Translation Cache (UTCL1)",
    "vl1d-cache-stall-metrics": "vL1D cache stall metrics",
    "vl1d-cache-access-metrics": "vL1D cache access metrics",
    "desc-td": "Vector L1 data-return path or Texture Data (TD)",
    "l2-sol": "L2 Speed-of-Light",
    "l2-cache-accesses": "L2 cache accesses",
    "l2-fabric-metrics": "L2-Fabric interface metrics",
    "l2-detailed-metrics": "L2 - Fabric interface detailed metrics",
    "l2-fabric-stalls": "L2 - Fabric Interface stalls",
    "desc-sl1d-sol": "Scalar L1D Speed-of-Light",
    "desc-sl1d-stats": "Scalar L1D cache accesses",
    "desc-sl1d-l2-interface": "Scalar L1D Cache - L2 Interface",
    "desc-l1i-sol": "L1I Speed-of-Light",
    "desc-l1i-stats": "L1I cache accesses",
    "desc-l1i-l2-interface": "L1I <-> L2 interface",
    "spi-util": "Workgroup manager utilizations",
    "spi-resc-util": "Workgroup Manager - Resource Allocation",
    "cpf-metrics": "Command processor fetcher (CPF)",
    "cpc-metrics": "Command processor packet processor (CPC)",
    "sys-sol": "System Speed-of-Light",
}

# Generate per-arch CDNA jinja contexts
jinja_contexts = {}
for context_name, section_name in cdna_section_map.items():
    for arch in ["gfx908", "gfx90a", "gfx942", "gfx950"]:
        if section_name in arch_metrics[arch]:
            jinja_contexts[f"{context_name}-{arch}"] = {
                "data": arch_metrics[arch][section_name],
            }

# Load gfx115x (RDNA 3.5) metrics YAML
with open("data/metrics/gfx115x_metrics.yaml") as f:
    gfx115x_metrics = yaml.safe_load(f)

# RDNA gfx115x section mapping: jinja context id -> YAML section name
rdna_gfx115x_section_map = {
    "sys-sol-gfx115x": "System Speed-of-Light",
    "rdna115x-roofline-performance-rates-gfx115x": "Roofline Performance Rates",
    "rdna115x-roofline-plot-points-gfx115x": "Roofline Plot Points",
    "rdna115x-wgp-utilization-gfx115x": "WGP Utilization",
    "rdna115x-wavefront-launch-stats-gfx115x": "Wavefront Launch Stats",
    "rdna115x-wave-dispatch-gfx115x": "Wave Dispatch",
    "rdna115x-wave-life-gfx115x": "Wave Life",
    "rdna115x-wave-instruction-mix-gfx115x": "Wave Instruction Mix",
    "rdna115x-vmem-instruction-mix-gfx115x": "VMEM Instruction Mix",
    "rdna115x-lds-instruction-mix-gfx115x": "LDS Instruction Mix",
    "rdna115x-wait-state-analysis-gfx115x": "Wait State Analysis",
    "rdna115x-wgp-instruction-cache-gfx115x": "WGP Instruction Cache",
    "rdna115x-wgp-scalar-data-cache-gfx115x": "WGP Scalar Data Cache",
    "rdna115x-gpu-utilization-gfx115x": "GPU Utilization",
    "rdna115x-shader-engine-utilization-gfx115x": "Shader Engine Utilization",
    "rdna115x-spi-utilization-gfx115x": "SPI Utilization",
    "rdna115x-wave-dispatch-statistics-gfx115x": "Wave Dispatch Statistics",
    "rdna115x-cpc-utilization-gfx115x": "CPC Utilization",
    "rdna115x-cpc-interface-utilization-gfx115x": "CPC Interface Utilization",
    "rdna115x-mec-stall-cycles-gfx115x": "MEC Stall Cycles",
    "rdna115x-cpc-memory-requests-gfx115x": "CPC Memory Requests",
    "rdna115x-mec-instruction-cache-gfx115x": "MEC Instruction Cache",
    "rdna115x-gl0-utilization-gfx115x": "GL0 Utilization",
    "rdna115x-gl0-request-statistics-gfx115x": "GL0 Request Statistics",
    "rdna115x-gl0-cache-performance-gfx115x": "GL0 Cache Performance",
    "rdna115x-gl0-gl1-interface-gfx115x": "GL0-GL1 Interface",
    "rdna115x-gl0-stalls-gfx115x": "GL0 Stalls",
    "rdna115x-gl1-cache-utilization-gfx115x": "GL1 Cache Utilization",
    "rdna115x-gl1-cache-request-statistics-gfx115x": "GL1 Cache Request Statistics",
    "rdna115x-gl1-cache-performance-gfx115x": "GL1 Cache Performance",
    "rdna115x-gl1-cache-stalls-gfx115x": "GL1 Cache Stalls",
    "rdna115x-gl1-gl2-interface-gfx115x": "GL1-GL2 Interface",
    "rdna115x-gl2-cache-performance-gfx115x": "GL2 Cache Performance",
    "rdna115x-gl2-cache-request-statistics-gfx115x": "GL2 Cache Request Statistics",
    "rdna115x-gl2-cache-bandwidth-gfx115x": "GL2 Cache Bandwidth",
    "rdna115x-dram-read-interface-gfx115x": "DRAM Read Interface",
    "rdna115x-dram-write-interface-gfx115x": "DRAM Write Interface",
    "rdna115x-system-arbiter-sarb-gfx115x": "System Arbiter (SARB)",
    "rdna115x-return-interface-gfx115x": "Return Interface",
    "rdna115x-memory-chart-instruction-cache-gfx115x": (
        "Memory chart - Instruction Cache"
    ),
    "rdna115x-memory-chart-scalar-data-cache-gfx115x": (
        "Memory chart - Scalar Data Cache"
    ),
    "rdna115x-memory-chart-tcp-cache-gfx115x": ("Memory chart - TCP Cache"),
    "rdna115x-memory-chart-lds-local-data-share-gfx115x": (
        "Memory chart - LDS (Local Data Share)"
    ),
    "rdna115x-memory-chart-tcp-gl1-interface-gfx115x": (
        "Memory chart - TCP-GL1 Interface"
    ),
    "rdna115x-memory-chart-gl1-cache-gfx115x": ("Memory chart - GL1 Cache"),
    "rdna115x-memory-chart-gl1-gl2-interface-gfx115x": (
        "Memory chart - GL1-GL2 Interface"
    ),
    "rdna115x-memory-chart-gl2-cache-gfx115x": ("Memory chart - GL2 Cache"),
    "rdna115x-memory-chart-gcea-to-system-memory-gfx115x": (
        "Memory chart - GCEA to System Memory"
    ),
}

# Generate gfx115x jinja contexts
for context_name, section_name in rdna_gfx115x_section_map.items():
    jinja_contexts[context_name] = {
        "data": gfx115x_metrics.get(section_name, {}),
    }

external_toc_path = "./sphinx/_toc.yml"
external_projects_current_project = "rocprofiler-compute"

# frequently used external resources
extlinks = {
    "dev-sample": (
        "https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-compute/sample/%s",
        "%s",
    ),
    "prod-page": (
        "https://www.amd.com/en/products/accelerators/instinct/%s.html",
        "%s",
    ),
    "llvm-docs": ("https://llvm.org/docs/AMDGPUUsage.html#%s", "%s"),
    "amd-lab-note": ("https://gpuopen.com/learn/amd-lab-notes/%s", "%s"),
    "cdna2-white-paper": (
        "https://www.amd.com/system/files/documents/amd-cdna2-white-paper.pdf#page=%s",
        "CDNA2 white paper (page %s)",
    ),
    "gcn-crash-course": (
        "https://www.slideshare.net/DevCentralAMD/gs4106-the-amd-gcn-architecture-a-crash-course-by-layla-mah#%s",
        "The AMD GCN Architecture - A Crash Course (slide %s)",
    ),
    "hip-training-pdf": (
        "https://www.olcf.ornl.gov/wp-content/uploads/2019/09/AMD_GPU_HIP_training_20190906.pdf#page=%s",
        "Introduction to AMD GPU Programming with HIP (slide %s)",
    ),
    "mantor-gcn-pdf": (
        "https://old.hotchips.org/wp-content/uploads/hc_archives/hc24/HC24-3-ManyCore/HC24.28.315-AMD.GCN.mantor_v1.pdf#page=%s",
        "AMD Radeon HD7970 with GCN Architecture (slide %s)",
    ),
    "mantor-vega10-pdf": (
        "https://old.hotchips.org/wp-content/uploads/hc_archives/hc29/HC29.21-Monday-Pub/HC29.21.10-GPU-Gaming-Pub/HC29.21.120-Radeon-Vega10-Mantor-AMD-f1.pdf#page=%s",
        "AMD Radeon Next Generation GPU Architecture - Vega10 (slide %s)",
    ),
    "mi200-isa-pdf": (
        "https://www.amd.com/system/files/TechDocs/instinct-mi200-cdna2-instruction-set-architecture.pdf#page=%s",
        "AMD Instinct MI200 ISA Reference Guide (page %s)",
    ),
    "hsa-runtime-pdf": (
        "http://hsafoundation.com/wp-content/uploads/2021/02/HSA-Runtime-1.2.pdf#page=%s",
        "HSA Runtime Programmer's Reference Manual (page %s)",
    ),
}

# Uncomment if facing rate limit exceed issue with local build
external_projects_remote_repository = ""
