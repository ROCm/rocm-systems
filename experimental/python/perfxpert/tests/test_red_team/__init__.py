"""Red-team suite — 14 adversarial attacks.

Audit-gate exit criterion: 100% pass (all 14 attacks defeated).

Eight gate-evasion attacks exercise runtime/gate_cascade.py directly.
Six prompt-injection attacks exercise the sanitization pipeline + tool-class split.

Unlike the parity suite, every single test MUST pass — aggregate thresholds do
NOT apply. A red-team failure is a hard block on breaking-change merges.
"""
