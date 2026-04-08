##############################################################################
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
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

##############################################################################

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re
import warnings
from pathlib import Path

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

_docs_root = Path(__file__).resolve().parent
_metrics_description = _docs_root / "data" / "metrics_description.yaml"
_per_arch_fallback = (
    _docs_root.parent
    / "tools"
    / "per_arch_metric_definitions"
    / "gfx942_metrics_description.yaml"
)
if _metrics_description.is_file():
    with _metrics_description.open(encoding="utf-8") as f:
        metrics_data = yaml.safe_load(f)
elif _per_arch_fallback.is_file():
    warnings.warn(
        f"{_metrics_description} not found; using {_per_arch_fallback.name} for "
        "jinja metric tables. Run "
        "`python tools/config_management/metric_description_manager.py --generate-docs` "
        "or restore docs/data/metrics_description.yaml for the canonical bundle.",
        stacklevel=1,
    )
    with _per_arch_fallback.open(encoding="utf-8") as f:
        metrics_data = yaml.safe_load(f)
else:
    raise FileNotFoundError(
        f"Neither {_metrics_description} nor {_per_arch_fallback} exists; "
        "cannot build metric description contexts."
    )

# Sphinx ``.. jinja::`` context id -> YAML section title (same keys in bundled
# ``metrics_description.yaml``, per-arch ``docs/data/metrics/<arch>_metrics.yaml``,
# and ``tools/per_arch_metric_definitions/gfx*_metrics_description.yaml``).
_METRIC_JINJA_BINDINGS: tuple[tuple[str, str], ...] = (
    ("wavefront-launch-stats", "Wavefront launch stats"),
    ("wavefront-runtime-stats", "Wavefront runtime stats"),
    ("instruction-mix", "Overall instruction mix"),
    ("valu-arith-instruction-mix", "VALU arithmetic instruction mix"),
    ("mfma-instruction-mix", "Matrix instruction mix"),
    ("compute-speed-of-light", "Compute Speed-of-Light"),
    ("pipeline-stats", "Pipeline statistics"),
    ("arithmetic-operations", "Arithmetic operations"),
    ("lds-sol", "LDS Speed-of-Light"),
    ("lds-stats", "LDS Statistics"),
    ("vl1d-sol", "vL1D Speed-of-Light"),
    ("ta-busy-stall", "Busy / stall metrics"),
    ("ta-instruction-counts", "Instruction counts"),
    ("ta-spill-stack", "Spill / stack metrics"),
    ("desc-utcl1", "L1 Unified Translation Cache (UTCL1)"),
    ("vl1d-cache-stall-metrics", "vL1D cache stall metrics"),
    ("vl1d-cache-access-metrics", "vL1D cache access metrics"),
    ("desc-td", "Vector L1 data-return path or Texture Data (TD)"),
    ("l2-sol", "L2 Speed-of-Light"),
    ("l2-cache-accesses", "L2 cache accesses"),
    ("l2-fabric-metrics", "L2-Fabric interface metrics"),
    ("l2-detailed-metrics", "L2 - Fabric interface detailed metrics"),
    ("l2-fabric-stalls", "L2 - Fabric Interface stalls"),
    ("desc-sl1d-sol", "Scalar L1D Speed-of-Light"),
    ("desc-sl1d-stats", "Scalar L1D cache accesses"),
    ("desc-sl1d-l2-interface", "Scalar L1D Cache - L2 Interface"),
    ("desc-l1i-sol", "L1I Speed-of-Light"),
    ("desc-l1i-stats", "L1I cache accesses"),
    ("desc-l1i-l2-interface", "L1I <-> L2 interface"),
    ("spi-util", "Workgroup manager utilizations"),
    ("spi-resc-util", "Workgroup Manager - Resource Allocation"),
    ("cpf-metrics", "Command processor fetcher (CPF)"),
    ("cpc-metrics", "Command processor packet processor (CPC)"),
    ("sys-sol", "System Speed-of-Light"),
)

if not isinstance(metrics_data, dict):
    raise TypeError("metrics bundle must deserialize to a dict")

jinja_contexts = {
    ctx: {"data": metrics_data.get(sec) or {}}
    for ctx, sec in _METRIC_JINJA_BINDINGS
}

_metrics_docs_dir = _docs_root / "data" / "metrics"
for _arch in ("gfx908", "gfx90a", "gfx942", "gfx950"):
    _per_arch_metrics_path = _metrics_docs_dir / f"{_arch}_metrics.yaml"
    if not _per_arch_metrics_path.is_file():
        continue
    with _per_arch_metrics_path.open(encoding="utf-8") as _f:
        _arch_bundle = yaml.safe_load(_f)
    if not isinstance(_arch_bundle, dict):
        continue
    for _ctx, _sec in _METRIC_JINJA_BINDINGS:
        jinja_contexts[f"{_ctx}-{_arch}"] = {
            "data": _arch_bundle.get(_sec) or {},
        }


def _merge_gfx1151_jinja_contexts(target: dict) -> None:
    """Populate RDNA gfx1151 ``.. jinja::`` contexts from panel analysis YAMLs."""
    import sys

    _sphinx_py = _docs_root / "sphinx"
    if str(_sphinx_py) not in sys.path:
        sys.path.insert(0, str(_sphinx_py))
    from gfx1151_jinja_metrics import GFX1151_JINJA_CONTEXT_IDS, build_gfx1151_jinja_contexts

    _gfx1151_cfg = (
        _docs_root.parent / "src" / "rocprof_compute_soc" / "analysis_configs" / "gfx1151"
    )
    if _gfx1151_cfg.is_dir():
        _payload = build_gfx1151_jinja_contexts(_gfx1151_cfg)
        for _ctx in GFX1151_JINJA_CONTEXT_IDS:
            target[_ctx] = _payload.get(_ctx, {"data": {}})
    else:
        for _ctx in GFX1151_JINJA_CONTEXT_IDS:
            target[_ctx] = {"data": {}}


_merge_gfx1151_jinja_contexts(jinja_contexts)

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
# external_projects_remote_repository = ""
