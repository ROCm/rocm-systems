# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Checks that analyze filter ids name rows of the workload.

The CLI parser and the analysis database both call these checks, so an unknown
id exits with the same message on either path.
"""

import re
from typing import Any, Iterable, List, Optional, Sequence, Union

from utils.logger import console_error
from utils.utils_common import normalize_filter_to_str_list

# One filter id as the CLI or a caller passes it. A '> n' dispatch filter is a str.
FilterId = Union[str, int]


def validate_gpu_filter_ids(
    filter_gpu_ids: Union[FilterId, Sequence[FilterId]],
    available_gpu_ids: Iterable[Any],
) -> None:
    """Exit when any --gpu-id names a GPU the rows do not contain."""
    available = {str(gpu_id) for gpu_id in available_gpu_ids}
    invalid = [
        gpu_id
        for gpu_id in normalize_filter_to_str_list(filter_gpu_ids)
        if gpu_id not in available
    ]
    if invalid:
        console_error("analysis", f"{invalid} is an invalid gpu-id")


def validate_dispatch_filter_ids(
    filter_dispatch_ids: Sequence[FilterId],
    available_dispatch_ids: Iterable[Any],
) -> None:
    """Exit when a --kernel-iteration-range names no dispatch of the rows.

    A '> n' id is valid when n does not exceed the highest available dispatch id.
    """
    available = {int(dispatch_id) for dispatch_id in available_dispatch_ids}
    if available:
        available_ids_hint = (
            f"Dispatch ids run from {min(available)} to {max(available)}."
        )
    else:
        available_ids_hint = "This workload has no dispatches."

    for dispatch_id in filter_dispatch_ids:
        threshold = dispatch_threshold(dispatch_id)
        if threshold is not None:
            valid = bool(available) and threshold <= max(available)
        else:
            valid = int(dispatch_id) in available
        if not valid:
            console_error(
                "analysis",
                f"{dispatch_id} is an invalid dispatch id. {available_ids_hint}",
            )


def validate_kernel_filter_ids(
    filter_kernel_ids: List[int],
    kernel_count: int,
) -> None:
    """Exit with a readable message when a -k id names no kernel."""
    if kernel_count == 0:
        console_error("analysis", "No kernels found in this workload.")

    for kernel_id in filter_kernel_ids:
        if not 0 <= kernel_id < kernel_count:
            console_error(
                "analysis",
                f"{kernel_id} is an invalid kernel id. "
                f"Please enter an id between 0-{kernel_count - 1}",
            )


def dispatch_threshold(dispatch_id: FilterId) -> Optional[int]:
    """Return n for a '> n' dispatch filter, or None for a plain dispatch id.

    This is the only place that reads the '> n' grammar, so every caller accepts
    and rejects the same ids. A '>' id without a whole-number bound, such as
    '> x', exits with a readable message.
    """
    match = re.match(r"\s*>\s*(\d+)", str(dispatch_id))
    if match:
        return int(match.group(1))
    if str(dispatch_id).lstrip().startswith(">"):
        console_error(
            "analysis",
            f"{dispatch_id} is an invalid dispatch filter. "
            "Use '> n' to keep the dispatches after dispatch n.",
        )
    return None
