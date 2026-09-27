#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
###############################################################################
"""Fit the log-scale residual from measured kernel times.

The target is ``log((measured - latency - launch) / t_throughput)``. A
one-shot EMA on the same ratio (``Calibrator``) is the better first step;
this fit is for a corpus.
"""

from __future__ import annotations

import math

from kernel_sched_hint.devices import Device
from kernel_sched_hint.model import (
    KernelWork,
    SCALE_MAX,
    SCALE_MIN,
    evaluate,
    phi_vector,
)


def _solve(matrix: list[list[float]]) -> list[float]:
    n = len(matrix)
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(matrix[row][col]))
        if abs(matrix[pivot][col]) < 1e-12:
            raise ValueError("normal equations are singular")
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        scale = matrix[col][col]
        for j in range(col, n + 1):
            matrix[col][j] /= scale
        for row in range(n):
            if row == col:
                continue
            factor = matrix[row][col]
            for j in range(col, n + 1):
                matrix[row][j] -= factor * matrix[col][j]
    return [matrix[i][n] for i in range(n)]


def fit_weights(
    rows: list[tuple[KernelWork, Device, int, int, float]],
    ridge: float = 1e-3,
) -> list[float]:
    """Each row is ``(work, device, waves, trip_count, measured_s)``."""

    xs: list[list[float]] = []
    ys: list[float] = []
    for work, dev, waves, trip_count, measured_s in rows:
        terms = evaluate(work, dev, waves, trip_count)
        variable = measured_s - terms.t_latency - terms.t_launch
        if terms.t_throughput <= 0.0 or variable <= 0.0:
            continue
        ratio = variable / terms.t_throughput
        ratio = min(SCALE_MAX, max(SCALE_MIN, ratio))
        xs.append(phi_vector(work, dev, waves, terms))
        ys.append(math.log(ratio))
    if len(xs) < 2:
        raise ValueError("need at least two usable measurements")

    dim = len(xs[0])
    xtx = [[0.0 for _ in range(dim)] for _ in range(dim)]
    xty = [0.0 for _ in range(dim)]
    for x, y in zip(xs, ys):
        for i in range(dim):
            xty[i] += x[i] * y
            for j in range(dim):
                xtx[i][j] += x[i] * x[j]
    for i in range(dim):
        xtx[i][i] += ridge
    augmented = [row + [xty[i]] for i, row in enumerate(xtx)]
    return _solve(augmented)
