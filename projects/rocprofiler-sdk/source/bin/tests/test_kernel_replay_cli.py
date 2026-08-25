#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""GPU-free unit tests for the kernel-replay option rules in the rocprofv3 launcher.

These rules are the only thing between a user and a run whose output they would read as complete.
A combination that should be rejected but is not produces a file in which several executions of one
kernel are indistinguishable, and nothing in that file says so -- so the rules are worth testing
even though they are only a handful of conditionals.
"""

import pytest


def _parse(rocprofv3, argv):
    args, _ = rocprofv3.parse_arguments(argv)
    return args


def test_replay_requires_counter_collection(rocprofv3):
    args = _parse(rocprofv3, ["--kernel-replay-beta-enabled", "--", "./app"])

    with pytest.raises(SystemExit) as excinfo:
        rocprofv3.validate_kernel_replay_args(args)

    assert excinfo.value.code != 0


def test_replay_with_counter_groups_is_accepted(rocprofv3):
    args = _parse(
        rocprofv3,
        [
            "--kernel-replay-beta-enabled",
            "--pmc",
            "SQ_WAVES",
            "--pmc",
            "GRBM_COUNT",
            "--",
            "./app",
        ],
    )

    rocprofv3.validate_kernel_replay_args(args)


# One dispatch executed N times under a single dispatch id: a consumer of these modes' output has no
# way to separate the executions, so the run is rejected rather than allowed to produce it.
#
# The sibling options are here for a reason. Each of these modes turns on from any member of its
# option group -- rocprofv3 keys SPM off `spm or spm_sample_interval or spm_sample_interval_unit`,
# and PC sampling off the union of its unit/method/interval -- so a check written against the
# headline flag alone would pass this suite for `--spm` and still let `--spm-sample-interval`
# through.
@pytest.mark.parametrize(
    "extra_argv",
    [
        pytest.param(["--att"], id="advanced-thread-trace"),
        pytest.param(["--att-no-intercept"], id="advanced-thread-trace-no-intercept"),
        pytest.param(["--pc-sampling-method", "stochastic"], id="pc-sampling"),
        pytest.param(["--pc-sampling-unit", "cycles"], id="pc-sampling-unit-only"),
        pytest.param(["--pc-sampling-interval", "1000"], id="pc-sampling-interval-only"),
        pytest.param(["--spm", "SQ_WAVES"], id="spm"),
        pytest.param(["--spm-sample-interval", "100"], id="spm-interval-only"),
    ],
)
def test_replay_rejects_per_dispatch_modes(rocprofv3, extra_argv):
    args = _parse(
        rocprofv3,
        ["--kernel-replay-beta-enabled", "--pmc", "SQ_WAVES"]
        + extra_argv
        + ["--", "./app"],
    )

    with pytest.raises(SystemExit) as excinfo:
        rocprofv3.validate_kernel_replay_args(args)

    assert excinfo.value.code != 0


# Tracing alongside replay is allowed: the records are correct, there really were N executions. Only
# the counts derived from them are inflated, which is a warning rather than a refusal, because a
# trace read for structure is still useful.
@pytest.mark.parametrize(
    "extra_argv",
    [
        pytest.param(["--kernel-trace"], id="kernel-trace"),
        pytest.param(["--stats"], id="stats"),
    ],
)
def test_replay_warns_but_allows_tracing(rocprofv3, extra_argv, capsys):
    args = _parse(
        rocprofv3,
        ["--kernel-replay-beta-enabled", "--pmc", "SQ_WAVES"]
        + extra_argv
        + ["--", "./app"],
    )

    rocprofv3.validate_kernel_replay_args(args)

    stderr = capsys.readouterr().err
    assert "kernel replay" in stderr
    assert "number of passes" in stderr, (
        "the warning must say what is inflated, not just that something is: a user who reads "
        "'results may differ' has no idea which column to distrust"
    )


# Counter collection alone must stay quiet, or the warning becomes noise users learn to ignore.
def test_replay_without_tracing_is_quiet(rocprofv3, capsys):
    args = _parse(
        rocprofv3,
        ["--kernel-replay-beta-enabled", "--pmc", "SQ_WAVES", "--", "./app"],
    )

    rocprofv3.validate_kernel_replay_args(args)

    assert capsys.readouterr().err == ""


# Every counter group must end up in a single run: the whole point of replay is to avoid the
# per-group child-process relaunch, and a silent fallback to relaunching would look like success
# while measuring a different application execution per group.
def test_replay_merges_every_counter_group_into_one_run(rocprofv3):
    argv = [
        "--kernel-replay-beta-enabled",
        "--pmc",
        "SQ_WAVES",
        "--pmc",
        "GRBM_COUNT",
        "--pmc",
        "SQ_INSTS_VALU",
        "--",
        "./app",
    ]
    args = _parse(rocprofv3, argv)

    assert args.pmc == [["SQ_WAVES"], ["GRBM_COUNT"], ["SQ_INSTS_VALU"]], (
        "three --pmc flags must parse as three groups; if they collapse, replay would run one "
        "pass and report a single group as if it were all of them"
    )


# pmc_groups comes from an input file rather than the command line, so it is simply absent from a
# namespace built by parse_arguments. A check that reads the attribute directly raises
# AttributeError here instead of validating.
def test_replay_accepts_counter_groups_from_an_input_file(rocprofv3):
    args = _parse(rocprofv3, ["--kernel-replay-beta-enabled", "--", "./app"])
    args.pmc_groups = [["SQ_WAVES"], ["GRBM_COUNT"]]

    rocprofv3.validate_kernel_replay_args(args)


# Kernel filtering is the documented remediation for a footprint too large to snapshot, so it has to
# combine with replay rather than be caught by a broad "no other options" rule.
@pytest.mark.parametrize(
    "extra_argv",
    [
        pytest.param(["--kernel-include-regex", "gemm.*"], id="include-regex"),
        pytest.param(["--kernel-exclude-regex", "memset.*"], id="exclude-regex"),
    ],
)
def test_replay_allows_kernel_filtering(rocprofv3, extra_argv, capsys):
    args = _parse(
        rocprofv3,
        ["--kernel-replay-beta-enabled", "--pmc", "SQ_WAVES"]
        + extra_argv
        + ["--", "./app"],
    )

    rocprofv3.validate_kernel_replay_args(args)
    assert capsys.readouterr().err == ""


# API tracing does not attribute anything per dispatch, so it neither blocks replay nor inflates a
# count. It must stay silent, or the warning stops carrying information.
def test_replay_allows_api_tracing_without_warning(rocprofv3, capsys):
    args = _parse(
        rocprofv3,
        [
            "--kernel-replay-beta-enabled",
            "--pmc",
            "SQ_WAVES",
            "--hip-trace",
            "--",
            "./app",
        ],
    )

    rocprofv3.validate_kernel_replay_args(args)
    assert capsys.readouterr().err == ""
