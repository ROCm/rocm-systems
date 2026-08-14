# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils/metrics/noise_clamper.py."""

import pandas as pd

# =============================================================================
# TESTS FOR NOISE_CLAMP: Multi-Pass Profiling Variance Handling
# =============================================================================


def test_noise_clamp_clamping_behavior():
    """Core behavior: positives unchanged, negatives clamped to 0."""
    import numpy as np

    from utils.metrics.noise_clamper import to_noise_clamp

    # Scalar: positive unchanged
    assert to_noise_clamp(1000.0, 100000.0) == 1000.0
    # Scalar: negative clamped
    assert to_noise_clamp(-100.0, 1000000.0) == 0.0

    # Series: mixed values
    diff = pd.Series([100.0, -50.0, 200.0, -100.0])
    ref = pd.Series([1e6, 1e6, 1e6, 1e6])
    result = to_noise_clamp(diff, ref)
    pd.testing.assert_series_equal(result, pd.Series([100.0, 0.0, 200.0, 0.0]))

    # NumPy array
    diff_np = np.array([100.0, -50.0])
    ref_np = np.array([1e6, 1e6])
    result_np = to_noise_clamp(diff_np, ref_np)
    np.testing.assert_array_equal(result_np, np.array([100.0, 0.0]))


def test_noise_clamp_zero_reference():
    """Edge case: zero reference should not cause division by zero."""
    from utils.metrics.noise_clamper import to_noise_clamp

    assert to_noise_clamp(-100.0, 0.0) == 0.0
    result = to_noise_clamp(pd.Series([-100.0]), pd.Series([0.0]))
    assert result.iloc[0] == 0.0


def test_noise_clamp_warning_above_threshold():
    """Warning recorded when relative error >= 1%."""
    from utils.metrics.noise_clamper import (
        clear_noise_clamp_warnings,
        get_noise_clamp_warnings,
        to_noise_clamp,
    )

    clear_noise_clamp_warnings()

    # 2% error (above 1% threshold) - should record
    to_noise_clamp(pd.Series([-20000.0]), pd.Series([1000000.0]))

    stats = get_noise_clamp_warnings()
    assert stats["count"] == 1
    assert stats["max_rel"] >= 0.01


def test_noise_clamp_no_warning_below_threshold():
    """No warning when relative error < 1%."""
    from utils.metrics.noise_clamper import (
        clear_noise_clamp_warnings,
        get_noise_clamp_warnings,
        to_noise_clamp,
    )

    clear_noise_clamp_warnings()

    # 0.5% error (below 1% threshold) - still clamped, no warning
    result = to_noise_clamp(pd.Series([-5000.0]), pd.Series([1000000.0]))
    assert result.iloc[0] == 0.0
    assert get_noise_clamp_warnings()["count"] == 0


def test_noise_clamp_empty_input():
    """Empty inputs should return empty without error."""
    from utils.metrics.noise_clamper import to_noise_clamp

    result = to_noise_clamp(pd.Series([], dtype=float), pd.Series([], dtype=float))
    assert len(result) == 0


def test_noise_clamp_threshold_boundary():
    """Exactly 1% error should trigger warning (>= not >)."""
    from utils.metrics.noise_clamper import (
        clear_noise_clamp_warnings,
        get_noise_clamp_warnings,
        to_noise_clamp,
    )

    clear_noise_clamp_warnings()

    # Exactly 1% error: -10000 / 1000000 = 0.01
    to_noise_clamp(pd.Series([-10000.0]), pd.Series([1000000.0]))
    assert get_noise_clamp_warnings()["count"] == 1


def test_noise_clamper_instance_isolation():
    """Separate NoiseClamper instances should have independent state."""
    import numpy as np

    from utils.metrics.noise_clamper import NoiseClamper

    clamper1 = NoiseClamper()
    clamper2 = NoiseClamper()

    clamper1.clamp(pd.Series([-20000.0]), pd.Series([1000000.0]))

    assert clamper1.get_stats()["count"] == 1
    assert clamper2.get_stats()["count"] == 0

    clamper1.clear()
    assert clamper1.get_stats()["count"] == 0
    assert clamper2.get_stats()["count"] == 0

    clamper1.clamp(np.array([-50000.0]), np.array([1000000.0]))
    clamper2.clamp(np.array([-30000.0, -40000.0]), np.array([1000000.0, 1000000.0]))

    assert clamper1.get_stats()["count"] == 1
    assert clamper2.get_stats()["count"] == 2
