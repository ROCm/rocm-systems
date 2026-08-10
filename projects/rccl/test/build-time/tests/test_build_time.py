# *************************************************************************
#  * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************
"""Build-time regression gate, one reported result per GPU target.

Two mutually exclusive gates, selected by --compare-base:

  * comparison (CI): clean-build the PR's base commit and its head on this
    same runner and fail if head is more than --max-regression-pct slower.
    A ratio cancels out machine speed, so it needs no per-machine tuning.
  * absolute: clean-build head only and fail if it exceeds --threshold-sec.
    Wall clock is only meaningful relative to a known machine, so this is for
    a runner whose budget someone has actually calibrated.

Both drive real from-scratch cmake builds and cost minutes per target; see
../README.md for how CI invokes them.
"""

import build_time as bt
import pytest


@pytest.mark.build_time
@pytest.mark.compare
def test_no_regression_against_base(
    gpu_target, settings, build_root, base_source_dir, record_property
):
    """Head builds no more than --max-regression-pct slower than its base."""
    target = gpu_target

    base_s, head_s, ok, fail_log = bt.compare_target(
        target, bt.RCCL_ROOT, base_source_dir, build_root,
        settings.jobs, settings.repeat, settings.keep,
    )
    if not ok:
        pytest.fail("clean build of %s failed, see %s" % (target, fail_log))

    result = bt.Comparison(target, bt.PASS, "", base_s, head_s)
    pct = result.delta_pct
    if pct is None:
        # A zero-second base build means the timing, not the code, is broken;
        # reporting it as a 0% delta would hide that.
        pytest.fail("base build of %s measured %s, so the ratio is meaningless"
                    % (target, bt.fmt(base_s)))

    record_property("target", target)
    record_property("base_seconds", "%.2f" % base_s)
    record_property("head_seconds", "%.2f" % head_s)
    record_property("delta_pct", "%.2f" % pct)

    assert pct <= settings.max_regression_pct, (
        "%s clean build is %.1f%% slower than base (limit %.1f%%): "
        "base %s -> head %s over %d round(s) at -j%d"
        % (target, pct, settings.max_regression_pct,
           bt.fmt(base_s), bt.fmt(head_s), settings.repeat, settings.jobs)
    )


@pytest.mark.build_time
@pytest.mark.absolute
def test_under_absolute_threshold(gpu_target, settings, build_root, record_property):
    """Head builds within --threshold-sec on a runner with a known budget."""
    if settings.compare:
        pytest.skip("running the comparison gate instead (--compare-base)")
    target = gpu_target

    total_s, ok, log_path = bt.time_best_of(
        target, bt.RCCL_ROOT, build_root, "head", settings.jobs, 0, settings.keep,
    )
    if not ok:
        pytest.fail("clean build of %s failed, see %s" % (target, log_path))

    record_property("target", target)
    record_property("total_seconds", "%.2f" % total_s)

    assert total_s <= settings.threshold_sec, (
        "%s clean build took %s, over the %s budget at -j%d"
        % (target, bt.fmt(total_s), bt.fmt(settings.threshold_sec), settings.jobs)
    )
