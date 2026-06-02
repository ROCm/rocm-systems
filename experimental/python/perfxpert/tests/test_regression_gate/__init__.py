"""Regression-gate false-positive audit — runs every proven_optimizations seed
through runtime/gate_cascade.py and confirms they PASS (i.e. the gate does not
reject a known-good optimization).

Audit-gate exit criterion: ≤ 5% false-positive rate on the seed corpus
(10 seed cases → at most 0 rejections at the strict 5% threshold; 10 seeds
means even 1 rejection fails the gate, per spec risk R17).
"""
