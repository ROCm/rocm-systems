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
PMC (Performance Monitor Counter) utilities: block limits, pass splitting,
and collection-context constants.
"""

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Collection-context detection
# ---------------------------------------------------------------------------

# Flags that --sys-trace subsumes.  Any flag in this set is redundant when
# kernel + memory-copy trace data is already present in the database.
_SYS_TRACE_IMPLIED: frozenset = frozenset(
    {
        "--sys-trace",
        "--hip-trace",
        "--hip-api-trace",
        "--hsa-trace",
        "--kernel-trace",
        "--memory-copy-trace",
        "--marker-trace",
        "--roctx-trace",
    }
)

# Args that only specify output location -- not considered "new data collection"
_OUTPUT_ONLY_ARGS: frozenset = frozenset(
    {"-d", "-o", "--output-directory", "--output-file"}
)

# Hardware counter collection limits for rocprofv3 --pmc.
#
# AMD GPUs limit how many performance counters from the SAME hardware block can
# be collected simultaneously in a single kernel dispatch pass.  The "block name"
# is the prefix before the first "_" in a counter name:
#
#   SQ_WAVES        -> block "SQ"    (shader / wavefront counters)
#   GRBM_COUNT      -> block "GRBM"  (GPU register bus manager)
#   TCP_*           -> block "TCP"   (L1 vector cache)
#   TCC_*           -> block "TCC"   (L2 cache)
#
# IMPORTANT: FETCH_SIZE and WRITE_SIZE are DERIVED metrics, not raw hardware counters.
# Internally rocprofv3 expands them to TCC hardware block counters:
#   FETCH_SIZE -> TCC_BUBBLE + TCC_EA0_RDREQ + GRBM_GUI_ACTIVE  (TCC block)
#   WRITE_SIZE -> TCC_EA0_WRREQ + TCC_EA0_WRREQ_64B              (TCC block)
# Combined they require ~4 TCC hardware counter slots (across 32 TCC instances on MI300X).
# They MUST be isolated in their own pass whenever SQ counters are also requested.
#
# Exceeding a block's per-pass limit causes rocprofv3 to abort with error code 38:
#   "Request exceeds the capabilities of the hardware to collect"
#
# Actual limits vary by GPU generation (MI100/MI200/MI300X) and block type.
# The values below are conservative safe defaults; some blocks (e.g. SQ on
# gfx942/MI300X) support up to 8 counters per pass in practice.
_PMC_BLOCK_LIMIT_DEFAULT: int = 4
_PMC_BLOCK_LIMITS: Dict[str, int] = {
    "SQ": 4,  # shader/wave; gfx942 supports up to 8 -- use 4 as safe default
    "GRBM": 4,  # GPU register bus manager
    "TCP": 4,  # L1 vector cache
    "TCC": 4,  # L2 cache
    "TA": 4,  # texture addressing
    "TD": 4,  # texture data
}

# FETCH_SIZE and WRITE_SIZE are derived metrics that each expand to multiple TCC
# hardware counters (FETCH_SIZE -> 3 counters, WRITE_SIZE -> 2 counters; combined 5
# exceed the TCC per-pass limit). Each must be in its own dedicated pass, isolated
# from all other counters -- including each other.
_TCC_DERIVED_COUNTERS: frozenset = frozenset({"FETCH_SIZE", "WRITE_SIZE"})


def _pmc_block(counter: str) -> str:
    """Return the hardware block name for a counter (prefix before first '_')."""
    return counter.split("_")[0]


def _pmc_block_limit(block: str) -> int:
    """Return the per-pass counter limit for the given hardware block."""
    return _PMC_BLOCK_LIMITS.get(block, _PMC_BLOCK_LIMIT_DEFAULT)


def _split_pmc_into_passes(
    counters: List[str],
    base_flags: List[str],
    base_args: List[Dict[str, Any]],
    output_dir: str,
    output_prefix: str,
    description: str,
    app_placeholder: str = "./app",
) -> List[Dict[str, Any]]:
    """
    Split a counter list into the minimum number of rocprofv3 commands so that
    no hardware block exceeds its per-pass collection limit.

    Strategy:
    - FETCH_SIZE and WRITE_SIZE are TCC-derived metrics that expand internally to
      multiple TCC hardware counters (FETCH_SIZE->3 TCC counters, WRITE_SIZE->2).
      Together they exceed the TCC block per-pass limit, so each derived counter
      MUST be in its own dedicated pass, isolated from all other counters.
    - For all other counters: group by hardware block (prefix before '_'),
      passes needed = max(ceil(block_count / block_limit)), distribute evenly.

    Returns a list of command dicts. Single-element when one pass suffices.
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
                _split_pmc_into_passes(
                    regular,
                    base_flags,
                    base_args,
                    output_dir,
                    output_prefix,
                    description,
                    app_placeholder,
                )
            )
        for dc in derived:
            # Single derived counter: build its command directly (no recursion).
            pmc_str = dc
            flags_str = " ".join(base_flags)
            non_pmc = [a for a in base_args if a.get("name") not in ("--pmc",)]
            args = list(non_pmc) + [
                {"name": "--pmc", "value": pmc_str},
                {"name": "-d", "value": output_dir},
                {"name": "-o", "value": output_prefix},
            ]
            all_cmds.append(
                {
                    "tool": "rocprofv3",
                    "description": description,
                    "flags": list(base_flags),
                    "args": args,
                    "full_command": (
                        f"rocprofv3 {flags_str} --pmc {pmc_str}"
                        f" -d {output_dir} -o {output_prefix} -- {app_placeholder}"
                    ).strip(),
                }
            )
        n = len(all_cmds)
        if n > 1:
            for idx, cmd in enumerate(all_cmds):
                out_name = f"{output_prefix}_pass{idx + 1}"
                pmc_val = next(
                    (a["value"] for a in cmd["args"] if a["name"] == "--pmc"), ""
                )
                flags_str = " ".join(base_flags)
                cmd["description"] = f"{description} (pass {idx + 1}/{n})"
                for arg in cmd["args"]:
                    if arg["name"] == "-o":
                        arg["value"] = out_name
                cmd["full_command"] = (
                    f"rocprofv3 {flags_str} --pmc {pmc_val}"
                    f" -d {output_dir} -o {out_name} -- {app_placeholder}"
                ).strip()
        return all_cmds

    # Standard path: group by block and distribute round-robin.
    block_groups: Dict[str, List[str]] = defaultdict(list)
    for c in counters:
        block_groups[_pmc_block(c)].append(c)

    if not block_groups:
        return []

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

    commands = []
    for idx, pctrs in enumerate(pass_counters):
        suffix = f" (pass {idx + 1}/{n})" if n > 1 else ""
        out_name = f"{output_prefix}_pass{idx + 1}" if n > 1 else output_prefix
        pmc_str = " ".join(pctrs)
        flags_str = " ".join(base_flags)
        non_pmc_args = [a for a in base_args if a.get("name") not in ("--pmc",)]
        args = list(non_pmc_args) + [
            {"name": "--pmc", "value": pmc_str},
            {"name": "-d", "value": output_dir},
            {"name": "-o", "value": out_name},
        ]
        full_cmd = (
            f"rocprofv3 {flags_str} --pmc {pmc_str}"
            f" -d {output_dir} -o {out_name} -- {app_placeholder}"
        ).strip()
        commands.append(
            {
                "tool": "rocprofv3",
                "description": f"{description}{suffix}",
                "flags": list(base_flags),
                "args": args,
                "full_command": full_cmd,
            }
        )
    return commands
