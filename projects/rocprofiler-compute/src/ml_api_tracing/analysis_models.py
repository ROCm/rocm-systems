# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Shared records for ML API tracing analysis data."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class MlApiTracePair:
    """Marker rows from one profiling pass and the sibling counter CSV."""

    marker_df: pd.DataFrame
    counter_path: Path
    joined_df: Optional[pd.DataFrame] = None
