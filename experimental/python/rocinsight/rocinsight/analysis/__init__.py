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
analysis package -- pure analysis logic extracted from analyze.py.

Re-exports every public and semi-public symbol for backward compatibility.
"""

from .core import (
    compute_time_breakdown,
    identify_hotspots,
    analyze_memory_copies,
    analyze_hardware_counters,
)

from .att import (
    _ATT_STALL_CATEGORY_MAP,
    _ATT_MIN_HITCOUNT,
    _att_stall_category,
    analyze_thread_trace,
)

from .pmc import (
    _SYS_TRACE_IMPLIED,
    _OUTPUT_ONLY_ARGS,
    _PMC_BLOCK_LIMIT_DEFAULT,
    _PMC_BLOCK_LIMITS,
    _TCC_DERIVED_COUNTERS,
    _pmc_block,
    _pmc_block_limit,
    _split_pmc_into_passes,
)

from .recommendations import (
    _INIT_OVERHEAD_MAX_KERNEL_PCT,
    _INIT_OVERHEAD_MAX_RUNTIME_NS,
    _detect_already_collected,
    _filter_rec_commands,
    generate_recommendations,
    _is_code_change_rec,
)

__all__ = [
    # core.py
    "compute_time_breakdown",
    "identify_hotspots",
    "analyze_memory_copies",
    "analyze_hardware_counters",
    # att.py
    "_ATT_STALL_CATEGORY_MAP",
    "_ATT_MIN_HITCOUNT",
    "_att_stall_category",
    "analyze_thread_trace",
    # pmc.py
    "_SYS_TRACE_IMPLIED",
    "_OUTPUT_ONLY_ARGS",
    "_PMC_BLOCK_LIMIT_DEFAULT",
    "_PMC_BLOCK_LIMITS",
    "_TCC_DERIVED_COUNTERS",
    "_pmc_block",
    "_pmc_block_limit",
    "_split_pmc_into_passes",
    # recommendations.py
    "_INIT_OVERHEAD_MAX_KERNEL_PCT",
    "_INIT_OVERHEAD_MAX_RUNTIME_NS",
    "_detect_already_collected",
    "_filter_rec_commands",
    "generate_recommendations",
    "_is_code_change_rec",
]
