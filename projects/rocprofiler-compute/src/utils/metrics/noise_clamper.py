# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Threshold-based clamping for multi-pass profiling noise."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.logger import console_warning


class NoiseClamper:
    """
    Tracks and clamps negative values from multi-pass counter variance.

    Negative counts are physically impossible - they result from run-to-run
    variance when counters are collected across multiple profiling passes.
    This class clamps negatives to 0 and tracks deviations for diagnostics.
    """

    WARN_THRESHOLD = 0.01  # 1% relative error threshold

    def __init__(self) -> None:
        self._count = 0
        self._max_rel_error = 0.0

    def clamp(
        self,
        difference: pd.Series | float | np.ndarray,
        reference: pd.Series | float | np.ndarray,
    ) -> pd.Series | float | np.ndarray:
        """Clamp negative values to 0 and track significant deviations."""
        if difference is None or (np.isscalar(difference) and pd.isna(difference)):
            return np.nan
        if np.isscalar(difference):
            return self._clamp_scalar(difference, reference)
        return self._clamp_array(difference, reference)

    def _clamp_scalar(self, difference: float, reference: float) -> float:
        """Clamp a single scalar value."""
        if difference >= 0:
            return difference
        rel_error = self._compute_relative_error(abs(difference), reference)
        self._record_if_significant(1, rel_error)
        return 0.0

    def _clamp_array(
        self,
        difference: pd.Series | np.ndarray,
        reference: pd.Series | np.ndarray | float,
    ) -> pd.Series | np.ndarray:
        """Clamp negative values in an array or Series."""
        result = difference.copy()
        negative_mask = result < 0

        if not np.any(negative_mask):
            return result

        safe_ref = self._make_safe_reference(reference)
        rel_errors = self._compute_relative_errors(result, negative_mask, safe_ref)
        result = self._apply_clamp(result, negative_mask)
        self._record_significant_deviations(rel_errors)

        return result

    def _make_safe_reference(
        self,
        reference: pd.Series | np.ndarray | float,
    ) -> pd.Series | np.ndarray | float:
        """Replace zero values with NaN to avoid division errors."""
        if isinstance(reference, pd.Series):
            return reference.replace(0, np.nan)
        if isinstance(reference, np.ndarray):
            return np.where(reference == 0, np.nan, reference)
        return reference if reference != 0 else np.nan

    def _compute_relative_error(self, abs_diff: float, reference: float) -> float:
        """Compute relative error for a scalar, handling zero reference."""
        if reference == 0:
            return 0.0
        return abs_diff / abs(reference)

    def _compute_relative_errors(
        self,
        result: pd.Series | np.ndarray,
        negative_mask: pd.Series | np.ndarray,
        safe_ref: pd.Series | np.ndarray | float,
    ) -> np.ndarray:
        """Compute relative errors for all negative values."""
        ref_vals = (
            safe_ref[negative_mask]
            if hasattr(safe_ref, "__getitem__") and not np.isscalar(safe_ref)
            else safe_ref
        )
        return np.abs(result[negative_mask]) / np.abs(ref_vals)

    def _apply_clamp(
        self,
        result: pd.Series | np.ndarray,
        negative_mask: pd.Series | np.ndarray,
    ) -> pd.Series | np.ndarray:
        """Set negative values to zero."""
        if isinstance(result, pd.Series):
            result.loc[negative_mask] = 0
        else:
            result[negative_mask] = 0
        return result

    def _record_if_significant(self, count: int, rel_error: float) -> None:
        """Record stats if error exceeds threshold."""
        if rel_error >= self.WARN_THRESHOLD:
            self._record_stats(count, rel_error)

    def _record_significant_deviations(self, rel_errors: np.ndarray) -> None:
        """Record stats for all values exceeding threshold."""
        warn_mask = rel_errors >= self.WARN_THRESHOLD
        if np.any(warn_mask):
            self._record_stats(int(np.sum(warn_mask)), float(np.max(rel_errors)))

    def _record_stats(self, count: int, max_rel: float) -> None:
        """Update running statistics."""
        self._count += count
        self._max_rel_error = max(self._max_rel_error, max_rel)

    def clear(self) -> None:
        """Reset collected statistics."""
        self._count = 0
        self._max_rel_error = 0.0

    def get_stats(self) -> dict:
        """Return copy of current statistics."""
        return {"count": self._count, "max_rel": self._max_rel_error}

    def print_summary(self) -> None:
        """Print summary if significant variance was detected."""
        if self._count == 0:
            return
        max_pct = self._max_rel_error * 100
        console_warning(
            f"Counter variance corrected: {self._count} value(s) adjusted "
            f"(max {max_pct:.1f}% deviation from multi-pass collection)."
        )


# Global instance for backward compatibility with YAML expressions
_noise_clamper = NoiseClamper()


def to_noise_clamp(
    difference: pd.Series | float | np.ndarray,
    reference: pd.Series | float | np.ndarray,
) -> pd.Series | float | np.ndarray:
    """Clamp negative values from multi-pass variance. Delegates to global tracker."""
    return _noise_clamper.clamp(difference, reference)


def clear_noise_clamp_warnings() -> None:
    """Clear collected stats."""
    _noise_clamper.clear()


def get_noise_clamp_warnings() -> dict:
    """Return collected stats."""
    return _noise_clamper.get_stats()


def print_noise_clamp_summary() -> None:
    """Print summary if significant variance was detected."""
    _noise_clamper.print_summary()
