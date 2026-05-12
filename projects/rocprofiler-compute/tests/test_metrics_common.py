# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils.metrics.common."""

from unittest.mock import patch

import pandas as pd

from utils.metrics.common import ValuDualIssueDetector


class TestValuDualIssueDetector:
    """Tests for utils.metrics.common.ValuDualIssueDetector."""

    def test_check_skips_non_candidate_metric(self):
        """Non-VALU metric names never trigger a warning, even when over peak."""
        detector = ValuDualIssueDetector(gpu_arch="gfx942")
        with patch("utils.metrics.common.console_warning") as console_warning_mock:
            detector.check("IPC", value=10.0, peak=5.0)
        console_warning_mock.assert_not_called()

    def test_check_skips_when_value_below_peak(self):
        """Candidate metric does not warn when value is within peak."""
        detector = ValuDualIssueDetector(gpu_arch="gfx942")
        with patch("utils.metrics.common.console_warning") as console_warning_mock:
            detector.check("VALU Utilization", value=80.0, peak=100.0)
        console_warning_mock.assert_not_called()

    def test_check_skips_when_peak_not_positive(self):
        """A non-positive peak is treated as missing and never warns."""
        detector = ValuDualIssueDetector(gpu_arch="gfx942")
        with patch("utils.metrics.common.console_warning") as console_warning_mock:
            detector.check("VALU Utilization", value=10.0, peak=0.0)
            detector.check("VALU Utilization", value=10.0, peak=-1.0)
        console_warning_mock.assert_not_called()

    def test_check_emits_valu_utilization_warning_above_peak(self):
        """VALU Utilization above peak emits the dual-issue warning."""
        detector = ValuDualIssueDetector(gpu_arch="gfx942")
        with patch("utils.metrics.common.console_warning") as console_warning_mock:
            detector.check("VALU Utilization", value=150.0, peak=100.0)
        console_warning_mock.assert_called_once()
        msg = console_warning_mock.call_args.args[0]
        assert "VALU Utilization can go up to 200%" in msg
        assert ValuDualIssueDetector.FAQ_URL in msg
        assert "SQ_ACTIVE_INST_VALU2" not in msg

    def test_check_emits_valu_flops_warning_above_peak(self):
        """VALU FLOPs (F64) above peak emits the FLOPs-flavored warning."""
        detector = ValuDualIssueDetector(gpu_arch="gfx942")
        with patch("utils.metrics.common.console_warning") as console_warning_mock:
            detector.check("VALU FLOPs (F64)", value=600.0, peak=400.0)
        console_warning_mock.assert_called_once()
        msg = console_warning_mock.call_args.args[0]
        assert "VALU FLOPs can exceed the peak value" in msg
        assert ValuDualIssueDetector.FAQ_URL in msg
        assert "SQ_ACTIVE_INST_VALU2" not in msg

    def test_check_appends_valu2_suffix_on_gfx950(self):
        """gfx950 with non-zero SQ_ACTIVE_INST_VALU2 appends the confirmation."""
        detector = ValuDualIssueDetector(
            gpu_arch="gfx950",
            valu2_series=pd.Series([0, 1, 2]),
        )
        with patch("utils.metrics.common.console_warning") as console_warning_mock:
            detector.check("VALU Utilization", value=150.0, peak=100.0)
        msg = console_warning_mock.call_args.args[0]
        assert "Dual-issue activity detected via SQ_ACTIVE_INST_VALU2 counter" in msg

    def test_check_omits_valu2_suffix_when_counter_sum_zero(self):
        """gfx950 with all-zero SQ_ACTIVE_INST_VALU2 omits the confirmation."""
        detector = ValuDualIssueDetector(
            gpu_arch="gfx950",
            valu2_series=pd.Series([0, 0, 0]),
        )
        with patch("utils.metrics.common.console_warning") as console_warning_mock:
            detector.check("VALU Utilization", value=150.0, peak=100.0)
        msg = console_warning_mock.call_args.args[0]
        assert "SQ_ACTIVE_INST_VALU2" not in msg

    def test_check_omits_valu2_suffix_when_counter_absent(self):
        """gfx950 without the SQ_ACTIVE_INST_VALU2 series omits the confirmation."""
        detector = ValuDualIssueDetector(gpu_arch="gfx950", valu2_series=None)
        with patch("utils.metrics.common.console_warning") as console_warning_mock:
            detector.check("VALU Utilization", value=150.0, peak=100.0)
        msg = console_warning_mock.call_args.args[0]
        assert "SQ_ACTIVE_INST_VALU2" not in msg

    def test_check_ignores_valu2_on_non_gfx950(self):
        """Non-gfx950 archs never append the SQ_ACTIVE_INST_VALU2 suffix."""
        detector = ValuDualIssueDetector(
            gpu_arch="gfx942",
            valu2_series=pd.Series([1, 2, 3]),
        )
        with patch("utils.metrics.common.console_warning") as console_warning_mock:
            detector.check("VALU Utilization", value=150.0, peak=100.0)
        msg = console_warning_mock.call_args.args[0]
        assert "SQ_ACTIVE_INST_VALU2" not in msg
