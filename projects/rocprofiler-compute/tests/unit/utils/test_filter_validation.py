# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils.filter_validation, the id checks shared by the CLI parser
and the analysis database."""

import common
import pytest

from utils.filter_validation import (
    dispatch_threshold,
    validate_dispatch_filter_ids,
    validate_gpu_filter_ids,
    validate_kernel_filter_ids,
)

# =============================================================================
# Helpers
# =============================================================================


def record_only_stub():
    """Return a calls list and a console_error stub that records without exiting."""
    calls = []

    def stub(*args, **_kwargs):
        calls.append(args)

    return calls, stub


# =============================================================================
# validate_gpu_filter_ids
# =============================================================================


class TestValidateGpuFilterIds:
    """Tests for utils.filter_validation.validate_gpu_filter_ids."""

    @pytest.mark.parametrize("filter_gpu_ids", [["0", "1"], "0", [1], ["1"]])
    def test_known_gpu_ids_are_accepted(self, monkeypatch, filter_gpu_ids) -> None:
        """Scalar, string and integer GPU ids that the rows contain pass."""
        calls, record = record_only_stub()
        common.patch_console(
            monkeypatch, "utils.filter_validation", "error", error=record
        )

        validate_gpu_filter_ids(filter_gpu_ids, [0, 1])

        assert calls == []

    def test_unknown_gpu_id_reports_the_list(self, monkeypatch) -> None:
        """A GPU id absent from the rows exits with the gpu-id message."""
        calls, record_and_exit = common.record_and_exit_stub()
        common.patch_console(
            monkeypatch, "utils.filter_validation", "error", error=record_and_exit
        )

        with pytest.raises(SystemExit):
            validate_gpu_filter_ids(["9"], [0, 1])

        assert calls == [("analysis", "['9'] is an invalid gpu-id")]

    def test_partly_invalid_list_reports_only_the_unknown_ids(
        self, monkeypatch
    ) -> None:
        """A valid id beside an unknown one still exits, naming only the unknown id."""
        calls, record_and_exit = common.record_and_exit_stub()
        common.patch_console(
            monkeypatch, "utils.filter_validation", "error", error=record_and_exit
        )

        with pytest.raises(SystemExit):
            validate_gpu_filter_ids(["0", "9"], [0, 1])

        assert calls == [("analysis", "['9'] is an invalid gpu-id")]

    def test_gpu_filter_on_rows_without_gpus_exits(self, monkeypatch) -> None:
        """A GPU filter on a frame with no rows names every id as invalid."""
        calls, record_and_exit = common.record_and_exit_stub()
        common.patch_console(
            monkeypatch, "utils.filter_validation", "error", error=record_and_exit
        )

        with pytest.raises(SystemExit):
            validate_gpu_filter_ids(["0"], [])

        assert calls == [("analysis", "['0'] is an invalid gpu-id")]


# =============================================================================
# validate_dispatch_filter_ids
# =============================================================================


class TestValidateDispatchFilterIds:
    """Tests for utils.filter_validation.validate_dispatch_filter_ids."""

    @pytest.mark.parametrize(
        "filter_dispatch_ids",
        [["1", "3"], [2], [">0"], [">4"], [" >4"], ["> 4"]],
    )
    def test_known_and_threshold_ids_are_accepted(
        self, monkeypatch, filter_dispatch_ids
    ) -> None:
        """Ids present in the rows pass, and so does a '> n' with n <= the max id."""
        calls, record = record_only_stub()
        common.patch_console(
            monkeypatch, "utils.filter_validation", "error", error=record
        )

        validate_dispatch_filter_ids(filter_dispatch_ids, [1, 2, 3, 4])

        assert calls == []

    def test_unknown_dispatch_id_reports_the_range(self, monkeypatch) -> None:
        """An absent dispatch id exits with the range hint."""
        calls, record_and_exit = common.record_and_exit_stub()
        common.patch_console(
            monkeypatch, "utils.filter_validation", "error", error=record_and_exit
        )

        with pytest.raises(SystemExit):
            validate_dispatch_filter_ids(["0"], [1, 2, 3, 4])

        assert calls == [
            (
                "analysis",
                "0 is an invalid dispatch id. Dispatch ids run from 1 to 4.",
            )
        ]

    def test_partly_invalid_list_exits_on_the_unknown_id(self, monkeypatch) -> None:
        """A valid id beside an unknown one exits, naming the unknown id."""
        calls, record_and_exit = common.record_and_exit_stub()
        common.patch_console(
            monkeypatch, "utils.filter_validation", "error", error=record_and_exit
        )

        with pytest.raises(SystemExit):
            validate_dispatch_filter_ids(["1", "9"], [1, 2, 3, 4])

        assert calls == [
            (
                "analysis",
                "9 is an invalid dispatch id. Dispatch ids run from 1 to 4.",
            )
        ]

    def test_threshold_past_the_highest_id_exits(self, monkeypatch) -> None:
        """A '> n' with n above the highest id exits with the range hint."""
        calls, record_and_exit = common.record_and_exit_stub()
        common.patch_console(
            monkeypatch, "utils.filter_validation", "error", error=record_and_exit
        )

        with pytest.raises(SystemExit):
            validate_dispatch_filter_ids([">5"], [1, 2, 3, 4])

        assert calls == [
            (
                "analysis",
                ">5 is an invalid dispatch id. Dispatch ids run from 1 to 4.",
            )
        ]

    def test_dispatch_filter_without_dispatches_exits(self, monkeypatch) -> None:
        """A dispatch filter on rows with no dispatches exits with that hint."""
        calls, record_and_exit = common.record_and_exit_stub()
        common.patch_console(
            monkeypatch, "utils.filter_validation", "error", error=record_and_exit
        )

        with pytest.raises(SystemExit):
            validate_dispatch_filter_ids([">0"], [])

        assert calls == [
            (
                "analysis",
                ">0 is an invalid dispatch id. This workload has no dispatches.",
            )
        ]


# =============================================================================
# dispatch_threshold
# =============================================================================


class TestDispatchThreshold:
    """Tests for utils.filter_validation.dispatch_threshold."""

    @pytest.mark.parametrize(
        ("dispatch_id", "expected"),
        [
            ("2", None),
            (2, None),
            (">2", 2),
            (" >2", 2),
            ("> 2", 2),
        ],
    )
    def test_reads_the_threshold_grammar_only(self, dispatch_id, expected) -> None:
        """A '> n' gives n, with or without spaces, and a plain id gives None."""
        assert dispatch_threshold(dispatch_id) == expected


# =============================================================================
# validate_kernel_filter_ids
# =============================================================================


class TestValidateKernelFilterIds:
    """Tests for utils.filter_validation.validate_kernel_filter_ids."""

    def test_ids_within_range_are_accepted(self, monkeypatch) -> None:
        mock_error = common.patch_console(
            monkeypatch, "utils.filter_validation", "error"
        )["error"]
        validate_kernel_filter_ids([0, 2], kernel_count=3)
        mock_error.assert_not_called()

    @pytest.mark.parametrize("kernel_id", [99, -1])
    def test_id_outside_range_reports_the_valid_range(
        self, monkeypatch, kernel_id
    ) -> None:
        """Ids above the last kernel and negative ids both name no kernel."""
        mock_error = common.patch_console(
            monkeypatch, "utils.filter_validation", "error"
        )["error"]

        # The mock records instead of exiting, so validation runs to completion.
        validate_kernel_filter_ids([kernel_id], kernel_count=3)

        assert f"{kernel_id} is an invalid kernel id" in mock_error.call_args.args[1]
        assert "0-2" in mock_error.call_args.args[1]

    def test_no_kernels_is_reported_as_such(self, monkeypatch) -> None:
        """A workload with no kernels reports that, not an empty range."""
        mock_error = common.patch_console(
            monkeypatch, "utils.filter_validation", "error"
        )["error"]

        validate_kernel_filter_ids([0], kernel_count=0)

        assert "No kernels found" in mock_error.call_args_list[0].args[1]
