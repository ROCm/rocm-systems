#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################

"""
Tier 0: Static source code analysis for GPU profiling planning.

Scans a source directory for GPU-related patterns (HIP/CUDA kernels, memory
operations, synchronization points, ROCm library calls) and produces a
structured profiling plan — recommending which rocprofv3/rocprof-compute
commands to run and which hardware counters to collect.

This module is intentionally self-contained (stdlib only, no rocpd DB
imports) so it can be used independently and tested without a database.

Usage:
    from rocinsight.ai_analysis.source_analyzer import SourceAnalyzer
    from pathlib import Path

    analyzer = SourceAnalyzer(Path("./my_app/src"))
    plan = analyzer.analyze()
    print(plan.programming_model)
    print(plan.suggested_first_command)
"""

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Source file extensions to scan
# ---------------------------------------------------------------------------
_GPU_EXTENSIONS = frozenset(
    {
        ".hip",
        ".cpp",
        ".cu",
        ".cuh",
        ".cl",
        ".h",
        ".hpp",
        ".hxx",
        ".cc",
    }
)
_PYTHON_EXTENSIONS = frozenset({".py", ".pyx"})
_ALL_EXTENSIONS = _GPU_EXTENSIONS | _PYTHON_EXTENSIONS

# Directories to skip unconditionally
_SKIP_DIRS = frozenset(
    {
        "build",
        "_build",
        ".build",
        "__pycache__",
        ".git",
        ".svn",
        "node_modules",
        "vendor",
        ".cache",
        "dist",
        "site-packages",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        "CMakeFiles",
        "CMakeCache",
        ".deps",
        "third_party",
        "thirdparty",
        "extern",
    }
)

# Max limits to avoid hanging on huge repos
_MAX_FILES = 500
_MAX_FILE_SIZE_BYTES = 512 * 1024  # 512 KB

# Maximum number of hardware counters per hardware block per rocprofv3 run.
#
# AMD GPUs limit how many performance counters from the same hardware block can
# be collected simultaneously in a single kernel dispatch pass.  The block name
# is the prefix before the first "_" in the counter name:
#
#   SQ_WAVES        → block "SQ"   (shader/wavefront counters)
#   GRBM_COUNT      → block "GRBM" (GPU register bus)
#   TCP_TOTAL_ACCESSES → block "TCP" (L1 cache per CU)
#   TCC_HIT         → block "TCC"  (L2 cache)
#
# Exceeding this limit causes rocprofv3 to abort with error code 38:
#   "Request exceeds the capabilities of the hardware to collect"
#
# Per-block limits vary by GPU generation (MI100/MI200/MI300X) and block type.
# Use _PMC_BLOCK_LIMIT as the safe default; override per-block in _PMC_BLOCK_LIMITS.
_PMC_BLOCK_LIMIT_DEFAULT: int = 4  # safe conservative default for any block
_PMC_BLOCK_LIMITS: Dict[str, int] = {
    # SQ (shader/wave counters): up to 8 on gfx942+ (MI300X); use 4 to be safe
    "SQ": 4,
    # GRBM (GPU register bus): 2–4 counters per pass
    "GRBM": 4,
    # Cache blocks: 4–8 per pass
    "TCP": 4,
    "TCC": 4,
    "TA": 4,
    "TD": 4,
}

# FETCH_SIZE and WRITE_SIZE are DERIVED metrics, not raw hardware counters.
# Internally rocprofv3 expands them to multiple TCC hardware block counters:
#   FETCH_SIZE → TCC_BUBBLE + TCC_EA0_RDREQ + GRBM_GUI_ACTIVE  (TCC block)
#   WRITE_SIZE → TCC_EA0_WRREQ + TCC_EA0_WRREQ_64B             (TCC block)
# Combined they require ~4 TCC hardware counter slots across 32 TCC instances.
# They MUST be collected in a separate pass from SQ counters to avoid error code 38.
_TCC_DERIVED_COUNTERS: frozenset = frozenset({"FETCH_SIZE", "WRITE_SIZE"})


def _pmc_block(counter: str) -> str:
    """Return the hardware block name for a counter (prefix before first '_')."""
    return counter.split("_")[0]


def _pmc_block_limit(block: str) -> int:
    """Return the per-pass limit for the given hardware block."""
    return _PMC_BLOCK_LIMITS.get(block, _PMC_BLOCK_LIMIT_DEFAULT)


def _pmc_commands(
    counters: List[str],
    flags: List[str],
    output_dir: str,
    output_prefix: str,
    description: str,
    app_placeholder: str = "./app",
) -> List[Dict[str, Any]]:
    """
    Return one or more rocprofv3 command dicts covering all *counters*, splitting
    across multiple passes when any hardware block exceeds its per-pass limit.

    Splitting strategy:
    - FETCH_SIZE and WRITE_SIZE are TCC-derived metrics that expand internally to
      multiple TCC hardware counters (FETCH_SIZE→3 TCC counters, WRITE_SIZE→2).
      Together they exceed the TCC block per-pass limit, so each derived counter
      MUST be in its own dedicated pass, isolated from all other counters.
    - For all other (raw) counters: group by hardware block (prefix before '_'),
      determine passes needed as max(ceil(block_count / block_limit)), distribute
      counters round-robin per block.
    """
    from collections import defaultdict

    if not counters:
        return []

    # Each TCC-derived counter must be in its own dedicated pass.
    derived = [c for c in counters if c in _TCC_DERIVED_COUNTERS]
    regular = [c for c in counters if c not in _TCC_DERIVED_COUNTERS]

    if derived and (len(derived) > 1 or regular):
        # Multiple derived counters can't share a pass (combined TCC hw counter count
        # exceeds the block limit). Each derived counter gets its own dedicated pass;
        # regular counters are handled together as a separate group.
        all_cmds = []
        if regular:
            all_cmds.extend(
                _pmc_commands(
                    regular,
                    flags,
                    output_dir,
                    output_prefix,
                    description,
                    app_placeholder,
                )
            )
        for dc in derived:
            # Single derived counter: build its command directly (no recursion).
            pmc_str = dc
            flags_str = " ".join(flags)
            all_cmds.append(
                {
                    "tool": "rocprofv3",
                    "description": description,
                    "flags": list(flags),
                    "args": [
                        {"name": "--pmc", "value": pmc_str},
                        {"name": "-d", "value": output_dir},
                        {"name": "-o", "value": output_prefix},
                    ],
                    "full_command": (
                        f"rocprofv3 {flags_str} --pmc {pmc_str}"
                        f" -d {output_dir} -o {output_prefix} -- {app_placeholder}"
                    ).strip(),
                }
            )
        # Re-number pass suffixes across the combined list.
        n = len(all_cmds)
        if n > 1:
            for idx, cmd in enumerate(all_cmds):
                cmd["description"] = f"{description} (pass {idx + 1}/{n})"
                out_name = f"{output_prefix}_pass{idx + 1}"
                for arg in cmd["args"]:
                    if arg["name"] == "-o":
                        arg["value"] = out_name
                pmc_val = next(a["value"] for a in cmd["args"] if a["name"] == "--pmc")
                flags_str = " ".join(flags)
                cmd["full_command"] = (
                    f"rocprofv3 {flags_str} --pmc {pmc_val}"
                    f" -d {output_dir} -o {out_name} -- {app_placeholder}"
                ).strip()
        return all_cmds

    # Standard path: group by block and distribute round-robin.
    block_groups: Dict[str, List[str]] = defaultdict(list)
    for c in counters:
        block_groups[_pmc_block(c)].append(c)

    n_passes = max(
        (len(cs) + _pmc_block_limit(blk) - 1) // max(_pmc_block_limit(blk), 1)
        for blk, cs in block_groups.items()
    )

    pass_counters: List[List[str]] = [[] for _ in range(n_passes)]
    for blk, cs in block_groups.items():
        limit = _pmc_block_limit(blk)
        for pass_idx in range(n_passes):
            chunk = cs[pass_idx * limit : (pass_idx + 1) * limit]
            pass_counters[pass_idx].extend(chunk)

    pass_counters = [p for p in pass_counters if p]
    n = len(pass_counters)

    cmds: List[Dict[str, Any]] = []
    for idx, pctrs in enumerate(pass_counters):
        suffix = f" (pass {idx + 1}/{n})" if n > 1 else ""
        out_name = f"{output_prefix}_pass{idx + 1}" if n > 1 else output_prefix
        pmc_str = " ".join(pctrs)
        flags_str = " ".join(flags)
        full_cmd = (
            f"rocprofv3 {flags_str} --pmc {pmc_str}"
            f" -d {output_dir} -o {out_name} -- {app_placeholder}"
        ).strip()
        cmds.append(
            {
                "tool": "rocprofv3",
                "description": f"{description}{suffix}",
                "flags": list(flags),
                "args": [
                    {"name": "--pmc", "value": pmc_str},
                    {"name": "-d", "value": output_dir},
                    {"name": "-o", "value": out_name},
                ],
                "full_command": full_cmd,
            }
        )
    return cmds


# ---------------------------------------------------------------------------
# Pattern table
# Each entry: (pattern_id, compiled_regex_or_None, severity, category, description_template)
# None regex = synthetic pattern computed from aggregate stats after all files are scanned.
# ---------------------------------------------------------------------------
@dataclass
class _PatternDef:
    pattern_id: str
    regex: Optional[re.Pattern]
    severity: str  # "high" | "medium" | "low" | "info"
    category: str
    description: str  # may contain {count} placeholder


_PATTERN_DEFS: List[_PatternDef] = [
    # ── Kernel definitions ────────────────────────────────────────────────
    _PatternDef(
        "GLOBAL_KERNEL_DEF",
        re.compile(r"\b__global__\s+\w[\w\s*&]*?\s+(\w+)\s*\(", re.MULTILINE),
        "info",
        "GPU Kernels",
        "Custom __global__ kernel definition(s) found",
    ),
    _PatternDef(
        "HIP_KERNEL_LAUNCH",
        re.compile(
            r"\bhipLaunchKernelGGL\s*\(\s*(?:\(void\s*\*\)\s*)?(\w+)", re.MULTILINE
        ),
        "info",
        "GPU Kernels",
        "hipLaunchKernelGGL kernel launch(es) found",
    ),
    _PatternDef(
        "TRIPLE_ANGLE_LAUNCH",
        re.compile(r"(\w+)\s*<<<\s*[^>]+>>>", re.MULTILINE),
        "info",
        "GPU Kernels",
        "Triple-angle-bracket kernel launch(es) found (<<<>>>)",
    ),
    _PatternDef(
        "HIP_KERNEL_NAME",
        re.compile(r"\bHIP_KERNEL_NAME\s*\(\s*(\w+)", re.MULTILINE),
        "info",
        "GPU Kernels",
        "HIP_KERNEL_NAME macro usage found",
    ),
    # ── Memory operations ─────────────────────────────────────────────────
    _PatternDef(
        "BLOCKING_MEMCPY",
        re.compile(r"\bhipMemcpy\s*\(", re.MULTILINE),
        "medium",
        "Memory",
        "Blocking hipMemcpy call(s) — consider hipMemcpyAsync to overlap with compute",
    ),
    _PatternDef(
        "ASYNC_MEMCPY",
        re.compile(r"\bhipMemcpyAsync\s*\(", re.MULTILINE),
        "info",
        "Memory",
        "hipMemcpyAsync call(s) found (good practice)",
    ),
    _PatternDef(
        "HOST_MALLOC_PINNED",
        re.compile(r"\bhipHostMalloc\s*\(", re.MULTILINE),
        "info",
        "Memory",
        "hipHostMalloc (pinned host memory) found — enables fast DMA transfers",
    ),
    _PatternDef(
        "DEVICE_MALLOC",
        re.compile(r"\bhipMalloc\s*\(", re.MULTILINE),
        "info",
        "Memory",
        "hipMalloc device allocation(s) found",
    ),
    _PatternDef(
        "MEMSET",
        re.compile(r"\bhipMemset\b", re.MULTILINE),
        "low",
        "Memory",
        "hipMemset call(s) — may create implicit sync; consider fusing with kernel init",
    ),
    _PatternDef(
        "MANAGED_MEMORY",
        re.compile(r"\bhipMallocManaged\s*\(", re.MULTILINE),
        "medium",
        "Memory",
        "hipMallocManaged (unified memory) found — can incur page-fault overhead on MI series",
    ),
    # ── Synchronization ───────────────────────────────────────────────────
    _PatternDef(
        "DEVICE_SYNC",
        re.compile(r"\bhipDeviceSynchronize\s*\(", re.MULTILINE),
        "high",
        "Synchronization",
        "hipDeviceSynchronize call(s) — serializes entire GPU pipeline; profile to confirm frequency",
    ),
    _PatternDef(
        "STREAM_SYNC",
        re.compile(r"\bhipStreamSynchronize\s*\(", re.MULTILINE),
        "medium",
        "Synchronization",
        "hipStreamSynchronize call(s) — stream-level sync; ensure not called in a hot loop",
    ),
    _PatternDef(
        "EVENT_SYNC",
        re.compile(r"\bhipEventSynchronize\s*\(", re.MULTILINE),
        "low",
        "Synchronization",
        "hipEventSynchronize call(s) found — event-based sync (preferred over device sync)",
    ),
    _PatternDef(
        "STREAM_WAIT_EVENT",
        re.compile(r"\bhipStreamWaitEvent\s*\(", re.MULTILINE),
        "info",
        "Synchronization",
        "hipStreamWaitEvent found — inter-stream dependency (good concurrency pattern)",
    ),
    # ── Streams / concurrency ─────────────────────────────────────────────
    _PatternDef(
        "STREAM_CREATE",
        re.compile(r"\bhipStreamCreate(?:WithFlags|WithPriority)?\s*\(", re.MULTILINE),
        "info",
        "Concurrency",
        "hipStreamCreate call(s) found — streams enable overlapping transfers and kernels",
    ),
    # ── ROCm library calls ────────────────────────────────────────────────
    _PatternDef(
        "ROCBLAS",
        re.compile(r"\brocblas_\w+\s*\(", re.MULTILINE),
        "info",
        "ROCm Libraries",
        "rocBLAS call(s) found — library is GPU-optimized; profile with rocprof-compute MFMA blocks",
    ),
    _PatternDef(
        "ROCSOLVER",
        re.compile(r"\brocsolver_\w+\s*\(", re.MULTILINE),
        "info",
        "ROCm Libraries",
        "rocSOLVER call(s) found",
    ),
    _PatternDef(
        "MIOPEN",
        re.compile(r"\bmiopen\w+\s*\(", re.MULTILINE),
        "info",
        "ROCm Libraries",
        "MIOpen call(s) found — deep learning primitives; profile conv/gemm with rocprof-compute",
    ),
    _PatternDef(
        "ROCFFT",
        re.compile(r"\brocfft_\w+\s*\(", re.MULTILINE),
        "info",
        "ROCm Libraries",
        "rocFFT call(s) found",
    ),
    _PatternDef(
        "HIPSPARSE",
        re.compile(r"\bhipsparse\w+\s*\(", re.MULTILINE),
        "info",
        "ROCm Libraries",
        "hipSparse call(s) found — sparse operations can be memory-bandwidth-bound",
    ),
    _PatternDef(
        "HIPBLAS",
        re.compile(r"\bhipblas\w+\s*\(", re.MULTILINE),
        "info",
        "ROCm Libraries",
        "hipBLAS call(s) found (portable BLAS interface)",
    ),
    _PatternDef(
        "ROCRAND",
        re.compile(r"\brocrand_\w+\s*\(", re.MULTILINE),
        "info",
        "ROCm Libraries",
        "rocRAND call(s) found",
    ),
    _PatternDef(
        "RCCL",
        re.compile(
            r"\bncclAll\w+\s*\(|\brcclAll\w+\s*\(|\bncclBcast\s*\(|\brcclBcast\s*\(",
            re.MULTILINE,
        ),
        "info",
        "Multi-GPU",
        "RCCL/NCCL collective operation(s) found — multi-GPU communication; profile inter-GPU bandwidth",
    ),
    # ── ROCTx instrumentation ─────────────────────────────────────────────
    _PatternDef(
        "ROCTX_RANGE_PUSH",
        re.compile(r"\broctxRangePush\s*\(", re.MULTILINE),
        "info",
        "Instrumentation",
        "roctxRangePush found — application is already instrumented with ROCTx markers",
    ),
    _PatternDef(
        "ROCTX_RANGE_POP",
        re.compile(r"\broctxRangePop\s*\(", re.MULTILINE),
        "info",
        "Instrumentation",
        "roctxRangePop found",
    ),
    _PatternDef(
        "ROCTX_MARK",
        re.compile(r"\broctxMark\s*\(", re.MULTILINE),
        "info",
        "Instrumentation",
        "roctxMark found",
    ),
    # ── Python GPU patterns ───────────────────────────────────────────────
    _PatternDef(
        "TORCH_CUDA_DEVICE",
        re.compile(
            r'\.cuda\(\)|\.to\(["\']cuda["\']|\.to\(device\s*=\s*["\']cuda["\']',
            re.MULTILINE,
        ),
        "info",
        "PyTorch",
        "PyTorch .cuda() / .to('cuda') tensor operation(s) found",
    ),
    _PatternDef(
        "TORCH_COMPILE",
        re.compile(r"\btorch\.compile\s*\(", re.MULTILINE),
        "info",
        "PyTorch",
        "torch.compile() found — compiled kernels; use torch profiler + rocprof-sys",
    ),
    _PatternDef(
        "TORCH_PROFILER",
        re.compile(r"\btorch\.profiler\b|\btorch\.autograd\.profiler\b", re.MULTILINE),
        "info",
        "PyTorch",
        "PyTorch profiler already in use",
    ),
    _PatternDef(
        "JAX_JIT",
        re.compile(r"\bjax\.jit\s*\(|\b@jax\.jit\b", re.MULTILINE),
        "info",
        "JAX",
        "JAX jit-compiled function(s) found",
    ),
]

# IDs of kernel-definition patterns (used to decide if any kernels found)
_KERNEL_PATTERN_IDS = frozenset(
    {
        "GLOBAL_KERNEL_DEF",
        "HIP_KERNEL_LAUNCH",
        "TRIPLE_ANGLE_LAUNCH",
        "HIP_KERNEL_NAME",
    }
)

# IDs of ROCTx instrumentation patterns
_ROCTX_PATTERN_IDS = frozenset(
    {
        "ROCTX_RANGE_PUSH",
        "ROCTX_RANGE_POP",
        "ROCTX_MARK",
    }
)

# Stable IDs for Tier 0 recommendations
_T0_CATEGORY_IDS = {
    "Initial Profiling": "ROCPD-T0-INIT-001",
    "Synchronization": "ROCPD-T0-SYNC-001",
    "Memory Transfer": "ROCPD-T0-MEM-001",
    "No Streams": "ROCPD-T0-STREAMS-001",
    "Managed Memory": "ROCPD-T0-MANAGED-001",
    "ROCm Libraries": "ROCPD-T0-LIBS-001",
    "Instrumentation": "ROCPD-T0-ROCTX-001",
    "PyTorch": "ROCPD-T0-PYTORCH-001",
    "Multi-GPU": "ROCPD-T0-MULTIGPU-001",
}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DetectedKernel:
    """A GPU kernel found in source code."""

    name: str
    file: str  # relative path from source_dir
    line: int
    launch_type: str  # "GLOBAL_KERNEL_DEF" | "HIP_KERNEL_LAUNCH" | "TRIPLE_ANGLE_LAUNCH" | "HIP_KERNEL_NAME"


@dataclass
class DetectedPattern:
    """A detected GPU programming pattern in source code."""

    pattern_id: str
    severity: str  # "high" | "medium" | "low" | "info"
    category: str
    description: str
    count: int  # total occurrences across all files
    locations: List[str] = field(default_factory=list)  # "file.cpp:42"


@dataclass
class ProfilingPlan:
    """
    Complete Tier 0 analysis result from static source scanning.

    Represents a profiling plan derived entirely from source code —
    before any profiling data has been collected.
    """

    source_dir: str
    analysis_timestamp: str
    programming_model: str  # "HIP" | "HIP+ROCm_Libraries" | "OpenCL" | "PyTorch_HIP" | "JAX_HIP" | "CUDA" | "Unknown"

    files_scanned: int
    files_skipped: int

    detected_kernels: List[DetectedKernel]
    kernel_count: int

    detected_patterns: List[DetectedPattern]
    risk_areas: List[str]

    already_instrumented: bool
    roctx_marker_count: int

    # Same dict structure as generate_recommendations() output
    recommendations: List[Dict[str, Any]]

    suggested_counters: List[str]
    suggested_first_command: str

    llm_explanation: Optional[str] = None


# ---------------------------------------------------------------------------
# Main analyzer class
# ---------------------------------------------------------------------------


class SourceAnalyzer:
    """
    Scans a source directory for GPU programming patterns and produces
    a ProfilingPlan with structured rocprofv3/rocprof-compute recommendations.

    Self-contained: stdlib + dataclasses only. No rocpd DB imports.

    Example:
        analyzer = SourceAnalyzer(Path("./src"))
        plan = analyzer.analyze()
        print(plan.programming_model)
        print(plan.suggested_first_command)
    """

    def __init__(
        self,
        source_dir: Path,
        max_files: int = _MAX_FILES,
        max_file_size_bytes: int = _MAX_FILE_SIZE_BYTES,
        verbose: bool = False,
    ):
        from .exceptions import SourceDirectoryNotFoundError

        if not source_dir.exists():
            raise SourceDirectoryNotFoundError(
                f"Source directory not found: {source_dir}"
            )
        if not source_dir.is_dir():
            raise SourceDirectoryNotFoundError(f"Not a directory: {source_dir}")

        self._source_dir = source_dir
        self._max_files = max_files
        self._max_file_size_bytes = max_file_size_bytes
        self._verbose = verbose

        # Aggregated state populated during analyze()
        self._pattern_counts: Dict[str, int] = {}  # pattern_id → total count
        self._pattern_locations: Dict[str, List[str]] = {}  # pattern_id → ["file:line"]
        self._detected_kernels: List[DetectedKernel] = []
        self._files_scanned: int = 0
        self._files_skipped: int = 0
        self._scan_truncated: bool = False  # True when _MAX_FILES limit is reached
        self._has_python: bool = False
        self._has_hip: bool = False
        self._has_opencl: bool = False

    # ── Public API ─────────────────────────────────────────────────────────

    def analyze(self) -> ProfilingPlan:
        """Scan the source directory and return a ProfilingPlan."""
        from .exceptions import SourceAnalysisError

        try:
            files = self._collect_files()
            for path in files:
                self._scan_file(path)

            self._apply_synthetic_patterns()

            programming_model = self._classify_programming_model()
            risk_areas = self._assess_risks()

            # Warn when the file limit was reached so users know results may
            # be incomplete for very large repositories.
            if self._scan_truncated:
                risk_areas.append(
                    f"File scan truncated: {self._files_skipped} files skipped "
                    f"(limit: {self._max_files}). Large repositories may need "
                    f"a more specific --source-dir path."
                )

            patterns = self._build_pattern_list()
            roctx_count = (
                self._pattern_counts.get("ROCTX_RANGE_PUSH", 0)
                + self._pattern_counts.get("ROCTX_RANGE_POP", 0)
                + self._pattern_counts.get("ROCTX_MARK", 0)
            )
            already_instrumented = roctx_count > 0
            suggested_counters = self._suggest_counters()
            recommendations = self._generate_profiling_commands(programming_model)
            first_cmd = self._pick_first_command(recommendations)

            return ProfilingPlan(
                source_dir=str(self._source_dir),
                analysis_timestamp=datetime.now().isoformat(),
                programming_model=programming_model,
                files_scanned=self._files_scanned,
                files_skipped=self._files_skipped,
                detected_kernels=self._detected_kernels,
                kernel_count=len(self._detected_kernels),
                detected_patterns=patterns,
                risk_areas=risk_areas,
                already_instrumented=already_instrumented,
                roctx_marker_count=roctx_count,
                recommendations=recommendations,
                suggested_counters=suggested_counters,
                suggested_first_command=first_cmd,
            )
        except Exception as e:
            if not isinstance(e, SourceAnalysisError):
                from .exceptions import SourceAnalysisError as SAE

                raise SAE(f"Source analysis failed: {e}") from e
            raise

    # ── File collection ────────────────────────────────────────────────────

    def _collect_files(self) -> List[Path]:
        """Walk source_dir and return files to scan, respecting limits."""
        collected: List[Path] = []

        for root, dirs, files in os.walk(self._source_dir):
            # Prune skip dirs in-place so os.walk doesn't recurse into them
            dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]

            for fname in files:
                if len(collected) >= self._max_files:
                    self._files_skipped += 1
                    self._scan_truncated = True
                    continue

                path = Path(root) / fname
                ext = path.suffix.lower()
                if ext not in _ALL_EXTENSIONS:
                    continue

                try:
                    size = path.stat().st_size
                except OSError:
                    self._files_skipped += 1
                    continue

                if size > self._max_file_size_bytes:
                    self._files_skipped += 1
                    if self._verbose:
                        print(f"[Tier0] Skipping large file ({size // 1024} KB): {path}")
                    continue

                collected.append(path)

        if self._verbose:
            print(f"[Tier0] Collected {len(collected)} files to scan")

        return collected

    # ── File scanning ──────────────────────────────────────────────────────

    def _scan_file(self, path: Path) -> None:
        """Scan a single file for all patterns and detected kernels."""
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            self._files_skipped += 1
            return

        rel = str(path.relative_to(self._source_dir))
        ext = path.suffix.lower()

        if ext in _PYTHON_EXTENSIONS:
            self._has_python = True
        elif ext in {".cl"}:
            self._has_opencl = True
        elif ext in _GPU_EXTENSIONS:
            self._has_hip = True

        self._files_scanned += 1

        # Strip block and line comments before pattern matching to reduce
        # false positives from commented-out code.
        clean = self._strip_comments(text)

        for pdef in _PATTERN_DEFS:
            if pdef.regex is None:
                continue  # synthetic — handled in _apply_synthetic_patterns
            matches = list(pdef.regex.finditer(clean))
            if not matches:
                continue

            self._pattern_counts[pdef.pattern_id] = self._pattern_counts.get(
                pdef.pattern_id, 0
            ) + len(matches)

            locs = self._pattern_locations.setdefault(pdef.pattern_id, [])
            for m in matches[:10]:  # cap locations per file to avoid bloat
                lineno = clean[: m.start()].count("\n") + 1
                locs.append(f"{rel}:{lineno}")

        # Extract kernel names for kernel-definition patterns
        self._extract_kernels(clean, rel)

    @staticmethod
    def _strip_comments(text: str) -> str:
        """Remove C/C++ // and /* */ comments from text (best-effort)."""
        # Remove block comments /* ... */ (non-greedy, including newlines)
        text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
        # Remove line comments // ... (up to end of line)
        text = re.sub(r"//[^\n]*", " ", text)
        return text

    def _extract_kernels(self, clean: str, rel_path: str) -> None:
        """Extract kernel names from all kernel-definition/launch patterns."""
        seen: set = set()

        # __global__ void kernel_name(
        for m in re.finditer(
            r"\b__global__\s+\w[\w\s*&]*?\s+(\w+)\s*\(", clean, re.MULTILINE
        ):
            name = m.group(1)
            if name not in seen:
                seen.add(name)
                lineno = clean[: m.start()].count("\n") + 1
                self._detected_kernels.append(
                    DetectedKernel(
                        name=name,
                        file=rel_path,
                        line=lineno,
                        launch_type="GLOBAL_KERNEL_DEF",
                    )
                )

        # hipLaunchKernelGGL(kernel_name, ...
        for m in re.finditer(
            r"\bhipLaunchKernelGGL\s*\(\s*(?:\(void\s*\*\)\s*)?(\w+)", clean, re.MULTILINE
        ):
            name = m.group(1)
            if name not in seen:
                seen.add(name)
                lineno = clean[: m.start()].count("\n") + 1
                self._detected_kernels.append(
                    DetectedKernel(
                        name=name,
                        file=rel_path,
                        line=lineno,
                        launch_type="HIP_KERNEL_LAUNCH",
                    )
                )

        # kernel_name<<<grid, block>>>(
        for m in re.finditer(r"\b(\w+)\s*<<<\s*[^>]+>>>", clean, re.MULTILINE):
            name = m.group(1)
            # Skip common non-kernel names that appear before <<<
            if name in {"if", "for", "while", "switch", "else", "return"}:
                continue
            if name not in seen:
                seen.add(name)
                lineno = clean[: m.start()].count("\n") + 1
                self._detected_kernels.append(
                    DetectedKernel(
                        name=name,
                        file=rel_path,
                        line=lineno,
                        launch_type="TRIPLE_ANGLE_LAUNCH",
                    )
                )

    # ── Synthetic patterns ─────────────────────────────────────────────────

    def _apply_synthetic_patterns(self) -> None:
        """Compute patterns derived from aggregate statistics."""
        has_kernels = any(
            self._pattern_counts.get(pid, 0) > 0 for pid in _KERNEL_PATTERN_IDS
        )
        has_streams = self._pattern_counts.get("STREAM_CREATE", 0) > 0

        # NO_STREAMS: custom kernels launched but no stream management
        if has_kernels and not has_streams:
            self._pattern_counts["NO_STREAMS"] = 1
            self._pattern_locations["NO_STREAMS"] = []

        # LOOP_DEVICE_SYNC: hipDeviceSynchronize count > 5 suggests it's in a loop
        dev_sync_count = self._pattern_counts.get("DEVICE_SYNC", 0)
        if dev_sync_count > 5:
            self._pattern_counts["LOOP_DEVICE_SYNC"] = dev_sync_count
            self._pattern_locations["LOOP_DEVICE_SYNC"] = self._pattern_locations.get(
                "DEVICE_SYNC", []
            )[:5]

    # ── Pattern list builder ───────────────────────────────────────────────

    def _build_pattern_list(self) -> List[DetectedPattern]:
        """Convert accumulated pattern counts into DetectedPattern objects."""
        result: List[DetectedPattern] = []

        # Known patterns from the table
        for pdef in _PATTERN_DEFS:
            count = self._pattern_counts.get(pdef.pattern_id, 0)
            if count == 0:
                continue
            result.append(
                DetectedPattern(
                    pattern_id=pdef.pattern_id,
                    severity=pdef.severity,
                    category=pdef.category,
                    description=pdef.description,
                    count=count,
                    locations=self._pattern_locations.get(pdef.pattern_id, [])[:20],
                )
            )

        # Synthetic patterns not in _PATTERN_DEFS
        _synthetic = [
            (
                "NO_STREAMS",
                "medium",
                "Concurrency",
                "No hipStreamCreate found — concurrent kernel/transfer overlap not possible",
            ),
            (
                "LOOP_DEVICE_SYNC",
                "high",
                "Synchronization",
                "hipDeviceSynchronize called many times — likely inside a loop (severe serialization)",
            ),
        ]
        for pid, sev, cat, desc in _synthetic:
            count = self._pattern_counts.get(pid, 0)
            if count == 0:
                continue
            result.append(
                DetectedPattern(
                    pattern_id=pid,
                    severity=sev,
                    category=cat,
                    description=desc,
                    count=count,
                    locations=self._pattern_locations.get(pid, [])[:10],
                )
            )

        # Sort: high → medium → low → info
        _sev_order = {"high": 0, "medium": 1, "low": 2, "info": 3}
        result.sort(key=lambda p: _sev_order.get(p.severity, 9))
        return result

    # ── Programming model classification ───────────────────────────────────

    def _classify_programming_model(self) -> str:
        """Infer the GPU programming model from detected patterns and file types."""
        has_torch = bool(
            self._pattern_counts.get("TORCH_CUDA_DEVICE", 0)
            or self._pattern_counts.get("TORCH_COMPILE", 0)
        )
        has_jax = bool(self._pattern_counts.get("JAX_JIT", 0))
        has_libs = bool(
            self._pattern_counts.get("ROCBLAS", 0)
            or self._pattern_counts.get("ROCSOLVER", 0)
            or self._pattern_counts.get("MIOPEN", 0)
            or self._pattern_counts.get("ROCFFT", 0)
            or self._pattern_counts.get("HIPSPARSE", 0)
            or self._pattern_counts.get("HIPBLAS", 0)
        )
        has_kernels = any(
            self._pattern_counts.get(pid, 0) > 0 for pid in _KERNEL_PATTERN_IDS
        )
        has_rccl = bool(self._pattern_counts.get("RCCL", 0))

        if has_torch:
            return "PyTorch_HIP"
        if has_jax:
            return "JAX_HIP"
        if self._has_opencl and not has_kernels:
            return "OpenCL"
        if has_kernels and has_libs:
            return "HIP+ROCm_Libraries"
        if has_kernels:
            return "HIP"
        if has_libs:
            return "HIP+ROCm_Libraries"
        if has_rccl:
            return "HIP+ROCm_Libraries"
        if self._has_hip:
            return "HIP"
        if self._has_python:
            return "Python_GPU"
        return "Unknown"

    # ── Risk assessment ────────────────────────────────────────────────────

    def _assess_risks(self) -> List[str]:
        """Produce a list of human-readable risk area descriptions."""
        risks: List[str] = []

        dev_sync = self._pattern_counts.get("DEVICE_SYNC", 0)
        if dev_sync > 5:
            risks.append(
                f"{dev_sync} hipDeviceSynchronize calls detected — "
                "frequent global synchronization will stall the GPU pipeline"
            )
        elif dev_sync > 0:
            risks.append(
                f"{dev_sync} hipDeviceSynchronize call(s) — "
                "verify these are not inside hot loops"
            )

        blocking_memcpy = self._pattern_counts.get("BLOCKING_MEMCPY", 0)
        async_memcpy = self._pattern_counts.get("ASYNC_MEMCPY", 0)
        if blocking_memcpy > 0 and async_memcpy == 0:
            risks.append(
                f"{blocking_memcpy} blocking hipMemcpy call(s) with no hipMemcpyAsync — "
                "transfers cannot overlap with kernel execution"
            )

        if self._pattern_counts.get("MANAGED_MEMORY", 0) > 0:
            risks.append(
                "hipMallocManaged (unified memory) detected — "
                "page migration overhead can be significant on MI-series GPUs"
            )

        if self._pattern_counts.get("NO_STREAMS", 0):
            risks.append(
                "No hipStreamCreate found — all work serialized on default stream; "
                "concurrent kernel/transfer overlap is not possible"
            )

        roctx = self._pattern_counts.get(
            "ROCTX_RANGE_PUSH", 0
        ) + self._pattern_counts.get("ROCTX_MARK", 0)
        if roctx == 0 and len(self._detected_kernels) > 0:
            risks.append(
                "No ROCTx markers found — adding roctxRangePush/Pop around key regions "
                "will make trace timelines much easier to interpret"
            )

        if not risks:
            risks.append(
                "No major static risk factors detected — "
                "run rocprofv3 --sys-trace to collect baseline profiling data"
            )

        return risks

    # ── Counter suggestions ────────────────────────────────────────────────

    def _suggest_counters(self) -> List[str]:
        """Suggest hardware counters relevant to the detected patterns."""
        counters: List[str] = ["GRBM_COUNT", "GRBM_GUI_ACTIVE", "SQ_WAVES"]

        has_libs = bool(
            self._pattern_counts.get("ROCBLAS", 0)
            or self._pattern_counts.get("HIPBLAS", 0)
            or self._pattern_counts.get("MIOPEN", 0)
        )
        if has_libs:
            # MFMA-heavy workloads — add VALU + memory bandwidth counters
            counters += ["SQ_INSTS_VALU", "FETCH_SIZE", "WRITE_SIZE"]

        has_custom_kernels = any(
            self._pattern_counts.get(pid, 0) > 0
            for pid in {"GLOBAL_KERNEL_DEF", "TRIPLE_ANGLE_LAUNCH", "HIP_KERNEL_LAUNCH"}
        )
        if has_custom_kernels:
            counters += ["SQ_INSTS_VMEM_RD", "SQ_INSTS_VMEM_WR", "SQ_INSTS_LDS"]

        has_memcpy = bool(
            self._pattern_counts.get("BLOCKING_MEMCPY", 0)
            or self._pattern_counts.get("ASYNC_MEMCPY", 0)
        )
        if has_memcpy:
            # PCIe/HBM bandwidth analysis
            if "FETCH_SIZE" not in counters:
                counters.append("FETCH_SIZE")
            if "WRITE_SIZE" not in counters:
                counters.append("WRITE_SIZE")

        # Deduplicate while preserving order
        seen: set = set()
        deduped: List[str] = []
        for c in counters:
            if c not in seen:
                seen.add(c)
                deduped.append(c)
        return deduped

    # ── Recommendation / command generation ───────────────────────────────

    def _generate_profiling_commands(
        self, programming_model: str
    ) -> List[Dict[str, Any]]:
        """
        Generate profiling recommendations in the same dict format as
        generate_recommendations() in analyze.py, so the existing formatters
        can render Tier 0 recommendations without modification.

        Recommendation dict structure:
            priority, category, issue, suggestion, actions[], estimated_impact, commands[]

        Command dict structure:
            tool, description, flags[], args[], full_command
        """
        recommendations: List[Dict[str, Any]] = []
        counters = self._suggest_counters()

        has_kernels = len(self._detected_kernels) > 0

        # ── Rec 1: Baseline sys-trace (always if any GPU code found) ────────
        # Split counters across multiple passes if they exceed _MAX_PMC_PER_PASS
        # to avoid hardware error code 38 ("Request exceeds the capabilities of
        # the hardware to collect").
        if has_kernels or self._pattern_counts.get("ROCBLAS", 0):
            # Determine how many passes the block-aware splitter will generate
            from collections import defaultdict as _dd

            _bg: Dict[str, int] = _dd(int)
            for c in counters:
                _bg[_pmc_block(c)] += 1
            n_passes = (
                max(
                    (_cnt + _pmc_block_limit(_blk) - 1) // max(_pmc_block_limit(_blk), 1)
                    for _blk, _cnt in _bg.items()
                )
                if _bg
                else 1
            )
            pass_note = (
                f" Hardware counter limits require {n_passes} separate collection passes."
                if n_passes > 1
                else ""
            )
            recommendations.append(
                {
                    "priority": "HIGH",
                    "category": "Initial Profiling",
                    "issue": (
                        f"No profiling data yet — {len(self._detected_kernels)} kernel(s) "
                        f"found in source. Establish a baseline trace first."
                    ),
                    "suggestion": (
                        "Run a baseline sys-trace to capture kernel timings and memory transfers."
                        + pass_note
                    ),
                    "actions": [
                        "Collect a sys-trace with hardware counters (split across passes if needed)",
                        "Open the resulting .db file with 'rocinsight analyze -i output.db'",
                        "Identify the top 3 kernels by time before deeper optimization",
                    ],
                    "estimated_impact": "Establishes ground truth for all subsequent optimization",
                    "commands": _pmc_commands(
                        counters,
                        flags=["--sys-trace"],
                        output_dir="./rocpd_output",
                        output_prefix="baseline",
                        description="Baseline trace + hardware counters",
                    ),
                }
            )

        # ── Rec 2: Synchronization risk ─────────────────────────────────────
        dev_sync = self._pattern_counts.get("DEVICE_SYNC", 0)
        loop_sync = self._pattern_counts.get("LOOP_DEVICE_SYNC", 0)
        if loop_sync > 0:
            sync_locations = self._pattern_locations.get("DEVICE_SYNC", [])[:3]
            locations_str = (
                "; ".join(sync_locations) if sync_locations else "multiple locations"
            )
            recommendations.append(
                {
                    "priority": "HIGH",
                    "category": "Synchronization",
                    "issue": (
                        f"{dev_sync} hipDeviceSynchronize call(s) detected — likely inside a loop "
                        f"({locations_str}). This serializes the entire GPU pipeline on every iteration."
                    ),
                    "suggestion": "Profile synchronization overhead and replace with hipStreamSynchronize or hipEventSynchronize",
                    "actions": [
                        "Use rocprof-sys to see exact CPU↔GPU synchronization gaps in the timeline",
                        "Replace hipDeviceSynchronize() with per-stream sync (hipStreamSynchronize)",
                        "Use hipEventRecord/hipEventSynchronize for fine-grained dependencies",
                        "Consider double-buffering to overlap computation and data transfers",
                    ],
                    "estimated_impact": "Can reduce idle GPU time by 20-60% in sync-heavy workloads",
                    "commands": [
                        {
                            "tool": "rocprofv3",
                            "description": "Capture synchronization events and gaps in kernel timeline",
                            "flags": ["--sys-trace"],
                            "args": [
                                {"name": "-d", "value": "./sync_output"},
                                {"name": "-o", "value": "sync_profile"},
                            ],
                            "full_command": "rocprofv3 --sys-trace -d ./sync_output -o sync_profile -- ./app",
                        },
                        {
                            "tool": "rocprof-sys",
                            "description": "System-level timeline showing CPU/GPU sync points and idle gaps",
                            "flags": ["--trace"],
                            "args": [],
                            "full_command": "rocprof-sys --trace -- ./app",
                        },
                    ],
                }
            )
        elif dev_sync > 0:
            recommendations.append(
                {
                    "priority": "MEDIUM",
                    "category": "Synchronization",
                    "issue": (
                        f"{dev_sync} hipDeviceSynchronize call(s) detected — "
                        "verify these are not in hot paths."
                    ),
                    "suggestion": "Profile to confirm sync frequency and duration at runtime",
                    "actions": [
                        "Check if hipDeviceSynchronize is inside a loop or called per-iteration",
                        "Replace with stream-level sync where possible",
                    ],
                    "estimated_impact": "Depends on call frequency; 5-30% improvement if in hot loop",
                    "commands": [
                        {
                            "tool": "rocprof-sys",
                            "description": "Timeline view to identify CPU/GPU synchronization points",
                            "flags": ["--trace"],
                            "args": [],
                            "full_command": "rocprof-sys --trace -- ./app",
                        },
                    ],
                }
            )

        # ── Rec 3: Blocking memcpy without async ────────────────────────────
        blocking = self._pattern_counts.get("BLOCKING_MEMCPY", 0)
        async_mc = self._pattern_counts.get("ASYNC_MEMCPY", 0)
        if blocking > 0 and async_mc == 0:
            recommendations.append(
                {
                    "priority": "MEDIUM",
                    "category": "Memory Transfer",
                    "issue": (
                        f"{blocking} blocking hipMemcpy call(s) with no hipMemcpyAsync — "
                        "transfers block the CPU until complete and cannot overlap with kernels."
                    ),
                    "suggestion": "Convert to hipMemcpyAsync and add hipHostMalloc for pinned buffers",
                    "actions": [
                        "Allocate host buffers with hipHostMalloc(size, hipHostMallocDefault) for DMA access",
                        "Replace hipMemcpy with hipMemcpyAsync(dst, src, size, kind, stream)",
                        "Create at least 2 streams to overlap H2D transfers with D2H or kernel execution",
                        "Profile with rocprofv3 --sys-trace to confirm transfer/kernel overlap",
                    ],
                    "estimated_impact": "15-40% reduction in total runtime when transfers are a bottleneck",
                    "commands": [
                        {
                            "tool": "rocprofv3",
                            "description": "Trace memory copies and kernel launches to measure overlap opportunity",
                            "flags": ["--sys-trace"],
                            "args": [
                                {"name": "-d", "value": "./memcpy_output"},
                                {"name": "-o", "value": "memcpy_profile"},
                            ],
                            "full_command": "rocprofv3 --sys-trace -d ./memcpy_output -o memcpy_profile -- ./app",
                        },
                    ],
                }
            )

        # ── Rec 4: No streams ────────────────────────────────────────────────
        if self._pattern_counts.get("NO_STREAMS", 0) and has_kernels:
            recommendations.append(
                {
                    "priority": "MEDIUM",
                    "category": "No Streams",
                    "issue": (
                        f"No hipStreamCreate found in {self._files_scanned} scanned files. "
                        "All work runs on the default stream (serialized)."
                    ),
                    "suggestion": "Add hipStream_t to enable concurrent kernel execution and transfer overlap",
                    "actions": [
                        "Create 2-4 streams with hipStreamCreate(&stream)",
                        "Pass stream to kernel launches: kernel<<<grid, block, 0, stream>>>(...)",
                        "Use hipMemcpyAsync with streams to overlap H2D and D2H with compute",
                        "Use hipStreamSynchronize(stream) instead of hipDeviceSynchronize()",
                    ],
                    "estimated_impact": "10-50% throughput improvement for workloads with independent work",
                    "commands": [
                        {
                            "tool": "rocprof-sys",
                            "description": "Visualize kernel concurrency gaps on the default stream",
                            "flags": ["--trace"],
                            "args": [],
                            "full_command": "rocprof-sys --trace -- ./app",
                        },
                    ],
                }
            )

        # ── Rec 5: Managed memory ────────────────────────────────────────────
        if self._pattern_counts.get("MANAGED_MEMORY", 0) > 0:
            recommendations.append(
                {
                    "priority": "MEDIUM",
                    "category": "Managed Memory",
                    "issue": (
                        "hipMallocManaged (unified/managed memory) detected. "
                        "On MI-series GPUs, page migration can add significant latency."
                    ),
                    "suggestion": "Replace hipMallocManaged with explicit hipMalloc + hipMemcpy for predictable performance",
                    "actions": [
                        "Profile page-fault overhead with rocprof-sys --trace-gpu-memory",
                        "Replace hipMallocManaged with hipMalloc (device) + hipHostMalloc (host) pairs",
                        "Use explicit hipMemcpy to control when data moves between host and device",
                    ],
                    "estimated_impact": "Can eliminate page-migration stalls; 2-10x improvement in some cases",
                    "commands": [
                        {
                            "tool": "rocprof-sys",
                            "description": "Trace GPU memory page migrations and access patterns",
                            "flags": [],
                            "args": [{"name": "--trace-gpu-memory", "value": None}],
                            "full_command": "rocprof-sys --trace-gpu-memory -- ./app",
                        },
                    ],
                }
            )

        # ── Rec 6: ROCm libraries → rocprof-compute deep dive ──────────────
        has_libs = bool(
            self._pattern_counts.get("ROCBLAS", 0)
            or self._pattern_counts.get("HIPBLAS", 0)
            or self._pattern_counts.get("MIOPEN", 0)
            or self._pattern_counts.get("ROCFFT", 0)
        )
        if has_libs:
            lib_names = []
            if self._pattern_counts.get("ROCBLAS", 0) or self._pattern_counts.get(
                "HIPBLAS", 0
            ):
                lib_names.append("rocBLAS/hipBLAS")
            if self._pattern_counts.get("MIOPEN", 0):
                lib_names.append("MIOpen")
            if self._pattern_counts.get("ROCFFT", 0):
                lib_names.append("rocFFT")
            recommendations.append(
                {
                    "priority": "MEDIUM",
                    "category": "ROCm Libraries",
                    "issue": (
                        f"{', '.join(lib_names)} call(s) detected. "
                        "Library kernels are pre-tuned but may be memory- or compute-bound "
                        "depending on problem size."
                    ),
                    "suggestion": "Profile library kernels with rocprof-compute to identify roofline position",
                    "actions": [
                        "Run rocprof-compute to see MFMA utilization and HBM bandwidth for library kernels",
                        "Check if matrix/tensor dimensions are optimal for the hardware (multiples of 16/64)",
                        "Try different batch sizes to find the efficiency sweet spot on the roofline",
                    ],
                    "estimated_impact": "Library tuning can yield 1.5-4x on GEMM-heavy workloads",
                    "commands": [
                        {
                            "tool": "rocprof-compute",
                            "description": "Roofline model and MFMA utilization for library kernels",
                            "flags": [],
                            "args": [{"name": "profile", "value": None}],
                            "full_command": "rocprof-compute profile -- ./app",
                        },
                    ],
                }
            )

        # ── Rec 7: ROCTx instrumentation suggestion ──────────────────────────
        roctx_count = self._pattern_counts.get(
            "ROCTX_RANGE_PUSH", 0
        ) + self._pattern_counts.get("ROCTX_MARK", 0)
        if roctx_count == 0 and has_kernels:
            recommendations.append(
                {
                    "priority": "LOW",
                    "category": "Instrumentation",
                    "issue": (
                        "No ROCTx markers found. Without markers, profiling timelines have "
                        "no application-level context — all you see is kernel names."
                    ),
                    "suggestion": "Add roctxRangePush/Pop around key computation phases",
                    "actions": [
                        "#include <roctx.h> and link with -lroctx64",
                        'Wrap major phases: roctxRangePush("forward_pass"); ...; roctxRangePop();',
                        'Add roctxMark("iteration_N") at loop checkpoints',
                        "Re-run rocprofv3 --sys-trace --marker-trace after adding markers",
                    ],
                    "estimated_impact": "No runtime impact; dramatically improves trace readability",
                    "commands": [
                        {
                            "tool": "rocprofv3",
                            "description": "Collect sys-trace with ROCTx marker tracing enabled",
                            "flags": ["--sys-trace", "--marker-trace"],
                            "args": [
                                {"name": "-d", "value": "./marked_output"},
                                {"name": "-o", "value": "marked_profile"},
                            ],
                            "full_command": "rocprofv3 --sys-trace --marker-trace -d ./marked_output -o marked_profile -- ./app",
                        },
                    ],
                }
            )

        # ── Rec 8: PyTorch / Python path ────────────────────────────────────
        if programming_model in ("PyTorch_HIP", "JAX_HIP", "Python_GPU"):
            fw = (
                "PyTorch"
                if "Torch" in programming_model or programming_model == "PyTorch_HIP"
                else "JAX"
            )
            recommendations.append(
                {
                    "priority": "HIGH",
                    "category": "PyTorch",
                    "issue": (
                        f"{fw} GPU code detected. Framework-level profiling is required "
                        "before dropping to rocprofv3."
                    ),
                    "suggestion": f"Use {fw} profiler for operator-level insight, then rocprof-sys for system-level timeline",
                    "actions": [
                        "Wrap your training/inference loop with torch.profiler.profile(activities=[ProfilerActivity.CUDA])",
                        "Use with_stack=True to get Python call stacks",
                        "Export to Chrome trace for visualization",
                        "Use rocprof-sys for system-level GPU timeline alongside the framework profiler",
                    ],
                    "estimated_impact": "Framework profiler reveals op-level bottlenecks before HW counter collection",
                    "commands": [
                        {
                            "tool": "rocprof-sys",
                            "description": f"System-level trace capturing {fw} GPU kernel timeline",
                            "flags": ["--trace"],
                            "args": [],
                            "full_command": "rocprof-sys --trace -- python ./train.py",
                        },
                        {
                            "tool": "rocprofv3",
                            "description": "Hardware counters for GPU kernels dispatched by PyTorch",
                            "flags": ["--sys-trace"],
                            "args": [
                                {
                                    "name": "--pmc",
                                    "value": "GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES",
                                },
                                {"name": "-d", "value": "./pytorch_output"},
                                {"name": "-o", "value": "pytorch_profile"},
                            ],
                            "full_command": "rocprofv3 --sys-trace --pmc GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES -d ./pytorch_output -o pytorch_profile -- python ./train.py",
                        },
                    ],
                }
            )

        # ── Rec 9: Multi-GPU ─────────────────────────────────────────────────
        if self._pattern_counts.get("RCCL", 0):
            recommendations.append(
                {
                    "priority": "MEDIUM",
                    "category": "Multi-GPU",
                    "issue": "RCCL/NCCL collective operations detected — inter-GPU communication may be a bottleneck.",
                    "suggestion": "Profile inter-GPU bandwidth and collective operation overlap",
                    "actions": [
                        "Enable RCCL_DEBUG=INFO to see collective sizes and durations",
                        "Profile with rocprof-sys to see NIC/PCIe/NVLink bandwidth utilization",
                        "Check RCCL_TREE_THRESHOLD and ring vs tree algorithm selection",
                    ],
                    "estimated_impact": "Communication optimizations can yield 1.2-2x on multi-GPU workloads",
                    "commands": [
                        {
                            "tool": "rocprof-sys",
                            "description": "System timeline showing inter-GPU communication and kernel overlap",
                            "flags": ["--trace"],
                            "args": [],
                            "full_command": "rocprof-sys --trace -- ./app",
                        },
                    ],
                }
            )

        # ── Rec 10: Default if nothing else triggered ────────────────────────
        if not recommendations:
            recommendations.append(
                {
                    "priority": "INFO",
                    "category": "Initial Profiling",
                    "issue": "No GPU source patterns detected. Ensure the correct source directory is provided.",
                    "suggestion": "Run a baseline sys-trace to capture any GPU activity",
                    "actions": [
                        "Verify --source-dir points to your GPU source files (.hip, .cpp, .cu)",
                        "Run rocprofv3 --sys-trace as a baseline even without detected patterns",
                    ],
                    "estimated_impact": "Baseline trace establishes ground truth for further analysis",
                    "commands": [
                        {
                            "tool": "rocprofv3",
                            "description": "Baseline sys-trace to capture any GPU activity",
                            "flags": ["--sys-trace"],
                            "args": [
                                {"name": "-d", "value": "./rocpd_output"},
                                {"name": "-o", "value": "baseline"},
                            ],
                            "full_command": "rocprofv3 --sys-trace -d ./rocpd_output -o baseline -- ./app",
                        },
                    ],
                }
            )

        return recommendations

    @staticmethod
    def _pick_first_command(recommendations: List[Dict[str, Any]]) -> str:
        """Return the full_command of the highest-priority recommendation."""
        for priority in ("HIGH", "MEDIUM", "LOW", "INFO"):
            for rec in recommendations:
                if rec.get("priority") == priority:
                    commands = rec.get("commands", [])
                    if commands:
                        return commands[0].get("full_command", "")
        return "rocprofv3 --sys-trace -d ./rocpd_output -o baseline -- ./app"
