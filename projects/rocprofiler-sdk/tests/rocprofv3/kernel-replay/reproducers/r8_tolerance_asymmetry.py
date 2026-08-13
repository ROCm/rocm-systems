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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Reproducer R8: the counter tolerance is asymmetric in drift direction.

validate.py declares COUNTER_TOLERANCE = 10%, but _approx_equal scales the allowed
difference by max(|a|, |b|, 1). When one replay pass reports a drifted value, the side
the drift falls on decides the effective threshold:

    over-reporting  a, a(1+f):  |diff| = a*f,  scale = a(1+f)  ->  fires when f > tol/(1-tol)
    under-reporting a, a(1-f):  |diff| = a*f,  scale = a        ->  fires when f > tol

So a restore that under-reports by 10.5% is caught and one that over-reports by 10.5% is
not. Needs no GPU: it exercises the comparison directly.

    ./r8_tolerance_asymmetry.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import validate  # noqa: E402

TOL = validate.COUNTER_TOLERANCE
BASE = 16384.0


def fires(base, drifted):
    """True when the validator's comparison would reject this pair."""
    return not validate._approx_equal(base, drifted)


def first_threshold(sign, lo=0.0, hi=1.0, iters=60):
    """Bisect for the smallest |f| the comparison rejects."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        if fires(BASE, BASE * (1.0 + sign * mid)):
            hi = mid
        else:
            lo = mid
    return hi


def main():
    over_predicted = TOL / (1.0 - TOL)
    under_predicted = TOL
    over_measured = first_threshold(+1.0)
    under_measured = first_threshold(-1.0)

    print(f"declared COUNTER_TOLERANCE            {TOL:.4%}")
    print(f"predicted over-reporting threshold    {over_predicted:.4%}   tol/(1-tol)")
    print(f"measured  over-reporting threshold    {over_measured:.4%}")
    print(f"predicted under-reporting threshold   {under_predicted:.4%}   tol")
    print(f"measured  under-reporting threshold   {under_measured:.4%}")

    print("\nconsequence, a drift of exactly 10.5% either way:")
    for sign, label in ((+1.0, "over-reports"), (-1.0, "under-reports")):
        drifted = BASE * (1.0 + sign * 0.105)
        verdict = "REJECTED" if fires(BASE, drifted) else "accepted"
        print(f"  pass {label} ({BASE:.0f} vs {drifted:.0f}): {verdict}")

    ok = (
        abs(over_measured - over_predicted) < 1e-6
        and abs(under_measured - under_predicted) < 1e-6
        and not fires(BASE, BASE * 1.105)
        and fires(BASE, BASE * 0.895)
    )
    print(
        "\nREPRODUCED: the same 10.5% error is caught in one direction and missed in the"
        " other."
        if ok
        else "\nNOT REPRODUCED: thresholds no longer match the asymmetric prediction."
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
