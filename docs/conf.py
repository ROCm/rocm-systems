##############################################################################bl
# MIT License
#
# Copyright (c) 2021 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##############################################################################el

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

with open("data/metrics_description.yaml", "r") as f:
    metrics_data = yaml.safe_load(f)
jinja_contexts = {
    "wavefront-launch-stats": {
        "data": metrics_data["Wavefront launch stats"],
    },
    "wavefront-runtime-stats": {
        "data": metrics_data["Wavefront runtime stats"],
    },
    "instruction-mix": {
        "data": metrics_data["Overall instruction mix"],
    },
    "valu-arith-instruction-mix": {
        "data": metrics_data["VALU arithmetic instruction mix"],
    },
    "mfma-instruction-mix": {
        "data": metrics_data["MFMA instruction mix"],
    },
    "compute-speed-of-light": {
        "data": metrics_data["Compute Speed-of-Light"],
    },
    "pipeline-stats": {
        "data": metrics_data["Pipeline statistics"],
    },
    "arithmetic-operations": {
        "data": metrics_data["Arithmetic operations"],
    },
    "lds-sol": {
        "data": metrics_data["LDS Speed-of-Light"],
    },
    "lds-stats": {
        "data": metrics_data["LDS Statistics"],
    },
    "vl1d-sol": {
        "data": metrics_data["vL1D Speed-of-Light"],
    },
    "ta-busy-stall": {
        "data": metrics_data["Busy / stall metrics"],
    },
    "ta-instruction-counts": {
        "data": metrics_data["Instruction counts"],
    },
    "ta-spill-stack": {
        "data": metrics_data["Spill / stack metrics"],
    },
    "desc-utcl1": {
        "data": metrics_data["L1 Unified Translation Cache (UTCL1)"],
    },
    "vl1d-cache-stall-metrics": {
        "data": metrics_data["vL1D cache stall metrics"],
    },
    "vl1d-cache-access-metrics": {
        "data": metrics_data["vL1D cache access metrics"],
    },
    "desc-td": {
        "data": metrics_data["Vector L1 data-return path or Texture Data (TD)"],
    },
    "l2-sol": {
        "data": metrics_data["L2 Speed-of-Light"],
    },
    "l2-cache-accesses": {
        "data": metrics_data["L2 cache accesses"],
    },
    "l2-fabric-metrics": {
        "data": metrics_data["L2-Fabric interface metrics"],
    },
    "l2-detailed-metrics": {
        "data": metrics_data["L2 - Fabric interface detailed metrics"],
    },
    "l2-fabric-stalls": {
        "data": metrics_data["L2 - Fabric Interface stalls"],
    },
    "desc-sl1d-sol": {
        "data": metrics_data["Scalar L1D Speed-of-Light"],
    },
    "desc-sl1d-stats": {
        "data": metrics_data["Scalar L1D cache accesses"],
    },
    "desc-sl1d-l2-interface": {
        "data": metrics_data["Scalar L1D Cache - L2 Interface"],
    },
    "desc-l1i-sol": {
        "data": metrics_data["L1I Speed-of-Light"],
    },
    "desc-l1i-stats": {
        "data": metrics_data["L1I cache accesses"],
    },
    "desc-l1i-l2-interface": {
        "data": metrics_data["L1I <-> L2 interface"],
    },
    "spi-util": {
        "data": metrics_data["Workgroup manager utilizations"],
    },
    "spi-resc-util": {
        "data": metrics_data["Workgroup Manager - Resource Allocation"],
    },
    "cpf-metrics": {
        "data": metrics_data["Command processor fetcher (CPF)"],
    },
    "cpc-metrics": {
        "data": metrics_data["Command processor packet processor (CPC)"],
    },
    "sys-sol": {
        "data": metrics_data["System Speed-of-Light"],
    },
}

external_toc_path = "./sphinx/_toc.yml"
external_projects_current_project = "rocprofiler-compute"

# frequently used external resources
extlinks = {
    "dev-sample": (
        "https://github.com/ROCm/rocprofiler-compute/blob/amd-mainline/sample/%s",
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