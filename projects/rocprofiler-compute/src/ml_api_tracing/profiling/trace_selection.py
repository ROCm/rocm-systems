# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Resolve the frameworks that --trace-ml-api selects in profile mode."""

from typing import List, Optional

from ml_api_tracing.constants import KNOWN_ML_API_BACKENDS


def select_traced_frameworks(trace_ml_api: Optional[List[str]]) -> Optional[List[str]]:
    """Return the frameworks selected by --trace-ml-api, or None when it is off.

    A bare flag selects every known framework. A repeated name counts once, in
    the order first given. User ROCTx regions are always recorded, so "user" is
    not a framework value.
    """
    if trace_ml_api is None:
        return None
    if not trace_ml_api:
        return list(KNOWN_ML_API_BACKENDS)
    return list(dict.fromkeys(trace_ml_api))
