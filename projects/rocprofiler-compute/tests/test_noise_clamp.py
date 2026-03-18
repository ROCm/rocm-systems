##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
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

##############################################################################

import numpy as np
import pandas as pd

from utils.parser import (
    NoiseClamper,
    clear_noise_clamp_warnings,
    get_noise_clamp_warnings,
    to_noise_clamp,
)


def test_noise_clamp_clamping_behavior():
    """Core behavior: positives unchanged, negatives clamped to 0."""
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
    assert to_noise_clamp(-100.0, 0.0) == 0.0
    result = to_noise_clamp(pd.Series([-100.0]), pd.Series([0.0]))
    assert result.iloc[0] == 0.0


def test_noise_clamp_warning_above_threshold():
    """Warning recorded when relative error >= 1%."""
    clear_noise_clamp_warnings()

    # 2% error (above 1% threshold) - should record
    to_noise_clamp(pd.Series([-20000.0]), pd.Series([1000000.0]))

    stats = get_noise_clamp_warnings()
    assert stats["count"] == 1
    assert stats["max_rel"] >= 0.01


def test_noise_clamp_no_warning_below_threshold():
    """No warning when relative error < 1%."""
    clear_noise_clamp_warnings()

    # 0.5% error (below 1% threshold) - still clamped, no warning
    result = to_noise_clamp(pd.Series([-5000.0]), pd.Series([1000000.0]))
    assert result.iloc[0] == 0.0
    assert get_noise_clamp_warnings()["count"] == 0


def test_noise_clamp_empty_input():
    """Empty inputs should return empty without error."""
    result = to_noise_clamp(pd.Series([], dtype=float), pd.Series([], dtype=float))
    assert len(result) == 0


def test_noise_clamp_threshold_boundary():
    """Exactly 1% error should trigger warning (>= not >)."""
    clear_noise_clamp_warnings()

    # Exactly 1% error: -10000 / 1000000 = 0.01
    to_noise_clamp(pd.Series([-10000.0]), pd.Series([1000000.0]))
    assert get_noise_clamp_warnings()["count"] == 1


def test_noise_clamper_instance_isolation():
    """Separate NoiseClamper instances should have independent state."""
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


def test_noise_clamp_scalar_exceeds_reference():
    """Scalar values exceeding reference are clamped to reference."""
    assert to_noise_clamp(150.0, 100.0) == 100.0
    assert to_noise_clamp(100.0, 100.0) == 100.0
    assert to_noise_clamp(99.99, 100.0) == 99.99


def test_noise_clamp_series_exceeds_reference():
    """Series values exceeding their per-element reference are clamped."""
    diff = pd.Series([50.0, 150.0, 100.0, 200.0])
    ref = pd.Series([100.0, 100.0, 100.0, 100.0])
    result = to_noise_clamp(diff, ref)
    pd.testing.assert_series_equal(result, pd.Series([50.0, 100.0, 100.0, 100.0]))


def test_noise_clamp_ndarray_exceeds_reference():
    """NumPy array values exceeding reference are clamped."""
    diff = np.array([50.0, 150.0, 200.0])
    ref = np.array([100.0, 100.0, 100.0])
    result = to_noise_clamp(diff, ref)
    np.testing.assert_array_equal(result, np.array([50.0, 100.0, 100.0]))


def test_noise_clamp_scalar_reference_with_array():
    """Scalar reference broadcasts correctly against an array difference."""
    diff = pd.Series([50.0, 110.0, 99.0])
    result = to_noise_clamp(diff, 100.0)
    pd.testing.assert_series_equal(result, pd.Series([50.0, 100.0, 99.0]))


def test_noise_clamp_mixed_negative_and_exceeding():
    """Both min-clamp and max-clamp fire in a single call."""
    diff = pd.Series([-10.0, 50.0, 150.0, 100.0])
    ref = pd.Series([100.0, 100.0, 100.0, 100.0])
    result = to_noise_clamp(diff, ref)
    pd.testing.assert_series_equal(result, pd.Series([0.0, 50.0, 100.0, 100.0]))

    assert to_noise_clamp(100 - 105, 100) == 0.0
    assert to_noise_clamp(100 - 80, 100) == 20.0


def test_noise_clamp_upper_warning_above_threshold():
    """Warning recorded when upper-bound excess >= 1%."""
    clear_noise_clamp_warnings()

    to_noise_clamp(pd.Series([105.0]), pd.Series([100.0]))

    stats = get_noise_clamp_warnings()
    assert stats["count"] == 1
    assert stats["max_rel"] >= 0.01


def test_noise_clamp_upper_no_warning_below_threshold():
    """No warning when upper-bound excess < 1%, but value is still clamped."""
    clear_noise_clamp_warnings()

    result = to_noise_clamp(pd.Series([100.005]), pd.Series([100.0]))
    assert result.iloc[0] == 100.0
    assert get_noise_clamp_warnings()["count"] == 0


def test_noise_clamp_nan_with_exceeding_values():
    """NaN and None inputs return np.nan regardless of reference."""
    assert np.isnan(to_noise_clamp(None, 100.0))
    assert np.isnan(to_noise_clamp(float("nan"), 100.0))


def test_noise_clamp_per_element_reference_array():
    """Each element is clamped against its own reference value."""
    diff = pd.Series([60.0, 200.0, 80.0])
    ref = pd.Series([100.0, 150.0, 50.0])
    result = to_noise_clamp(diff, ref)
    pd.testing.assert_series_equal(result, pd.Series([60.0, 150.0, 50.0]))
