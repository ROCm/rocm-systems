# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for metric debug row extraction."""

import pandas as pd

from utils.metrics.debug_row_tracker import _collect_debug_column_data


def test_debug_column_data_includes_pass_qualified_columns() -> None:
    frame = pd.DataFrame({"GRBM_GUI_ACTIVE@pass:pmc_perf_1": [200]})
    rows, _width = _collect_debug_column_data(
        "to_avg(raw_pmc_df['GRBM_GUI_ACTIVE@pass:pmc_perf_1'])",
        frame,
    )
    assert rows == [("raw_pmc_df['GRBM_GUI_ACTIVE@pass:pmc_perf_1']", [200])]
