###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
###############################################################################

"""
Tier 3: Advanced Thread Trace (ATT) analysis.
"""

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Tier 3: Advanced Thread Trace (ATT) analysis
# ---------------------------------------------------------------------------

_ATT_STALL_CATEGORY_MAP: Dict[str, str] = {
    # VMEM instructions -> HBM/cache latency
    "GLOBAL_LOAD": "att_vmem_latency",
    "GLOBAL_STORE": "att_vmem_latency",
    "FLAT_LOAD": "att_vmem_latency",
    "FLAT_STORE": "att_vmem_latency",
    "BUFFER_LOAD": "att_vmem_latency",
    "BUFFER_STORE": "att_vmem_latency",
    # LDS instructions -> LDS bank conflict or throughput
    "DS_READ": "att_lds_conflict",
    "DS_WRITE": "att_lds_conflict",
    "DS_BPERMUTE": "att_lds_conflict",
    # Wait instructions -> in-flight memory dependency
    "S_WAITCNT": "att_dependency_chain",
    "S_WAIT_": "att_dependency_chain",
    # Branch instructions -> divergence
    "S_BRANCH": "att_divergence",
    "S_CBRANCH": "att_divergence",
    "V_CMP": "att_divergence",
}

_ATT_MIN_HITCOUNT = 6400  # 100 wavefronts x 64 threads -- below this, stall ratio is noise


def _att_stall_category(instruction_id: str) -> str:
    """Classify ATT instruction stall by PC offset prefix / ISA mnemonic hint."""
    prefix = instruction_id.upper()
    for key, cat in _ATT_STALL_CATEGORY_MAP.items():
        if prefix.startswith(key):
            return cat
    return "att_vmem_latency"  # default: assume memory latency


def analyze_thread_trace(att_dir: str) -> Dict[str, Any]:
    """
    Tier 3: Parse Advanced Thread Trace (ATT) CSV output from rocprofv3 --att.

    Reads ``stats_<kernel_name>.csv`` files produced by rocprof-trace-decoder in
    ``att_dir``.  Returns stall analysis: top stalling instructions per kernel,
    stall ratios, and bottleneck classification.

    Args:
        att_dir: Path to directory containing ``stats_*.csv`` files.

    Returns:
        Dict with keys:
        - ``has_att_data`` (bool)
        - ``kernels`` (list of per-kernel dicts)
        - ``summary`` (aggregated stats)
        - ``reason`` (error string when has_att_data=False)
    """
    import csv as _csv
    from pathlib import Path as _AttPath

    att_dir_path = _AttPath(att_dir)
    if not att_dir_path.is_dir():
        return {
            "has_att_data": False,
            "kernels": [],
            "summary": {},
            "reason": f"ATT directory not found: {att_dir}",
        }

    csv_files = sorted(att_dir_path.glob("stats_*.csv"))
    if not csv_files:
        # Also search one level deep (rocprofv3 may create a sub-directory)
        csv_files = sorted(att_dir_path.glob("*/stats_*.csv"))

    if not csv_files:
        return {
            "has_att_data": False,
            "kernels": [],
            "summary": {},
            "reason": (
                f"No stats_*.csv files found in {att_dir}. "
                "Ensure rocprofv3 --att was used and rocprof-trace-decoder is installed."
            ),
        }

    kernels_data: List[Dict[str, Any]] = []
    high_stall_count = 0

    for csv_path in csv_files:
        # Fallback name from filename; overridden below by the kernel comment row
        kernel_name = csv_path.stem[len("stats_") :]
        instructions: List[Dict[str, Any]] = []

        try:
            with open(csv_path, newline="") as fh:
                # Skip comment/header lines starting with '#'
                lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
            reader = _csv.DictReader(lines)

            # Normalise header names (strip whitespace + surrounding quotes, lowercase)
            for row in reader:
                row_lower = {
                    k.strip().strip('"').lower(): v.strip().strip('"')
                    for k, v in row.items()
                }

                def _col(*candidates: str) -> str:
                    for c in candidates:
                        if c in row_lower:
                            return row_lower[c]
                    return ""

                # Real rocprofv3 --att CSV: "CodeObj","Vaddr","Instruction",
                # "Hitcount","Latency","Stall","Idle","Source"
                instr_name = _col(
                    "instruction", "instruction id", "pc_offset", "pc offset", "offset"
                )
                hitcount_s = _col("hitcount", "hit count", "count")
                latency_s = _col("latency (cycles)", "latency", "total latency")
                stall_s = _col("stall cycles", "stall", "stalls")
                source_line = _col("source line", "source", "file:line")

                try:
                    hitcount = int(hitcount_s) if hitcount_s else 0
                    total_latency = int(latency_s) if latency_s else 0
                    stall_cycles = int(stall_s) if stall_s else 0
                except ValueError:
                    continue

                # Comment rows (Hitcount=0, Latency=0) embed the demangled kernel
                # name in the Source column: "; _Zmangled...", Source="demangled()"
                if hitcount == 0 and total_latency == 0:
                    demangled = (
                        source_line  # e.g. "heavy_elementwise_kernel(float*, int)"
                    )
                    if demangled:
                        # Use just the function name without arguments
                        kernel_name = demangled.split("(")[0].strip()
                    continue

                stall_ratio = stall_cycles / total_latency if total_latency > 0 else 0.0
                weighted_stall = stall_cycles * hitcount

                instructions.append(
                    {
                        "pc_offset": instr_name,
                        "hitcount": hitcount,
                        "total_latency_cycles": total_latency,
                        "stall_cycles": stall_cycles,
                        "stall_ratio": round(stall_ratio, 4),
                        "weighted_stall": weighted_stall,
                        "source_line": source_line,
                    }
                )
        except Exception as exc:
            # Malformed CSV -- skip this kernel but continue
            kernels_data.append(
                {
                    "name": kernel_name,
                    "csv_file": str(csv_path),
                    "error": str(exc),
                    "top_stalling_instructions": [],
                    "avg_stall_ratio": 0.0,
                    "stall_category": "unknown",
                    "total_weighted_stall": 0,
                }
            )
            continue

        if not instructions:
            continue

        # Sort by weighted stall descending; take top 10
        instructions.sort(key=lambda x: x["weighted_stall"], reverse=True)
        top_stalling = instructions[:10]

        total_weighted = sum(i["weighted_stall"] for i in instructions)
        avg_stall = (
            sum(i["stall_ratio"] for i in instructions) / len(instructions)
            if instructions
            else 0.0
        )

        # Classify bottleneck from the top stalling instruction's PC prefix
        top_pc = top_stalling[0]["pc_offset"] if top_stalling else ""
        stall_category = _att_stall_category(top_pc)

        is_high_stall = (
            top_stalling
            and top_stalling[0]["stall_ratio"] >= 0.60
            and top_stalling[0]["hitcount"] >= _ATT_MIN_HITCOUNT
        )
        if is_high_stall:
            high_stall_count += 1

        kernels_data.append(
            {
                "name": kernel_name,
                "csv_file": str(csv_path),
                "instruction_count": len(instructions),
                "top_stalling_instructions": top_stalling,
                "avg_stall_ratio": round(avg_stall, 4),
                "stall_category": stall_category,
                "total_weighted_stall": total_weighted,
            }
        )

    # Sort kernels by total weighted stall descending (worst first)
    kernels_data.sort(key=lambda k: k.get("total_weighted_stall", 0), reverse=True)

    return {
        "has_att_data": bool(kernels_data),
        "kernels": kernels_data,
        "summary": {
            "kernel_count": len(kernels_data),
            "high_stall_kernels": high_stall_count,
        },
        "reason": (
            "" if kernels_data else "No valid ATT data could be parsed from CSV files"
        ),
    }
