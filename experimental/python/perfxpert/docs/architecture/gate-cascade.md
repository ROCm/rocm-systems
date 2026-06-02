# Gate Cascade (5-Gate Correctness Middleware)

Every code change the Recommendation / specialist agents propose is
validated by a strict 5-gate cascade before the change is considered
accepted. The cascade runs as **middleware** — not inside any agent
— so no agent can reorder or skip gates. The Correctness agent
receives an immutable `GateVerdict` and narrates it; that's the only
agent contact surface.

Source of truth: `perfxpert/runtime/gate_cascade.py` (spec §5, §5.0).

Cross-links:
- [Agent hierarchy](agent-hierarchy.md) — where Correctness fits in
  the tier map
- [Agentic mode](../guides/agentic-mode.md) — how verdicts surface to
  end-users (LLM vs air-gap)

## The 5 gates

Strict order — ALL must pass:

| # | Gate | Tool delegated to | Reject condition |
|---|------|---------------------|-------------------|
| 1 | Compile / Run | `tools.compile.build` | Build failure (non-zero exit) |
| 2 | SOL sanity (anti-Sakana) | `tools.sol.sanity_check` | See below — two-tier |
| 3 | Bitwise / Numeric | `tools.patch.verify_output` | Output drift > tolerance |
| 4 | Regression (weighted-geomean) | `tools.regression.compare_runs` | See below — three conditions |
| 5 | Test Anchors | `tools.anchors.check` | Any prior-pass test now fails |

Gates 1, 3, 4, and 5 invoke **EXECUTION-class** tools; see
[mcp-server.md](../integration/mcp-server.md) for why this matters.
Gate 2 invokes a READ_ONLY tool but is owned by the middleware layer
for verdict-coherence reasons.

### Two-tier SOL check (Gate 2)

Gate 2 is the primary anti-Sakana defense against reward-hacking (LLM
claims a wildly unrealistic speedup to satisfy the optimization target).

It runs in two tiers:

1. **Hard cap on speedup** — `SOL_MAX_REASONABLE_SPEEDUP = 50.0`.
   Any `claimed_speedup > 50×` is rejected immediately, regardless of
   architecture or absolute FLOPS. This catches gross reward-hacking
   with no GPU-specific setup required.

2. **Absolute-FLOPS check (optional)** — when
   `achieved_flops_per_sec` is known, the gate compares it to the
   architecture's published peak (via `arch.lookup_peaks`). Any
   optimized run that claims `flops > peak` is rejected.

Either failing tier produces `GateVerdict(status="reject", failing_gate="sol", ...)`.

### Regression gate (Gate 4) — three reject paths

The regression gate passes only if **all three** of these are satisfied:

| Check | Threshold | Constant |
|-------|-----------|----------|
| Any individual hot kernel regressed | > 10% → reject | `HOT_KERNEL_INDIVIDUAL_THRESHOLD_PCT` |
| Weighted-geomean degradation on tail | > 5% → reject | `TAIL_GEOMEAN_THRESHOLD_PCT` |
| Total regression vs baseline | > 3% → reject | `REGRESSION_NOISE_THRESHOLD_PCT` |

"Hot kernel" = top-K covering 80% of cumulative runtime, plus any
kernel ≥ 3% individually (spec §5 definition). Weighted-geomean is the
noise-robust metric for the tail of non-hot kernels: sensitive enough
to catch cross-the-board regressions that individual thresholds miss,
but immune to single-run jitter.

The gate returns `status="regressed"` (not `"reject"`) to distinguish
"code is semantically correct but slower" from "code is broken".

### Test-anchor gate (Gate 5) — last line of defense

Test anchors are a small set of known-good unit or integration tests
that must stay green across every optimization cycle. If any anchor
test that previously passed now fails, the gate returns
`status="reject", failing_gate="anchors"`.

Anchors are NOT the full test suite — they're a curated subset chosen
to catch regressions the earlier gates miss (e.g. numerical drift
inside a tolerance band that the bitwise gate accepted).

## Verdict propagation

`evaluate()` / `run_gate_cascade()` return a single immutable
`GateVerdict`:

```python
from perfxpert.runtime.gate_cascade import run_gate_cascade, GateInput

# Minimum required fields — everything else is optional and defaults to
# None. In this shape the cascade runs gates 2 (SOL) + 4 (regression)
# only; gates 1/3/5 short-circuit when their optional inputs are absent.
# For a full 5-gate run, supply source_file + diff_payload (compile),
# verify_output_baseline/new (bitwise), and test_anchor_baseline/new.
verdict = run_gate_cascade(GateInput(
    kernel_name="my_kernel",
    claimed_speedup=1.15,
    arch="gfx942",
    baseline_runtime_ns=1_000_000,
    achieved_runtime_ns=870_000,
    patch_sha="abc123",
))

print(verdict.status)         # 'pass' | 'reject' | 'regressed'
print(verdict.failing_gate)   # None on pass; gate name on reject
print(verdict.detail)         # Human-readable reason
print(verdict.delta_pct)      # Only meaningful for regression verdicts
```

Statuses:

- `pass` — all 5 gates green; the patch is accepted.
- `reject` — a non-regression gate failed; `failing_gate` says which.
- `regressed` — the code compiled + ran + matched output but was
  measurably slower. The Recommendation agent treats this as a
  weaker rejection (may try a different technique from the same
  specialist's knowledge file).

Note that `GateVerdict` is a frozen dataclass — agents that receive it
cannot mutate it. The Correctness agent's fence slice
(`perfxpert/agents/fence/correctness.md`) explicitly instructs it to
narrate `detail` verbatim and never override `status`.

## Debug-loop caps (spec §5)

To bound cost in agentic-LLM mode, the cascade is invoked under three
session-level caps:

| Constant | Value | Meaning |
|----------|-------|---------|
| `MAX_OPTIMIZATION_CYCLES_PER_KERNEL` | 5 | How many patch-then-verify rounds per hot kernel before giving up |
| `MAX_CONSECUTIVE_FAILURES` | 3 | Stop after 3 consecutive verdicts ≠ pass |
| `MAX_SESSION_LLM_TURNS` | 100 | Hard cap on total LLM turns in one session |

These constants are verified at runtime; see
`tests/test_runtime/test_gate_cascade.py` for coverage.

## Why middleware, not agent

Design-review C1 in the spec: if the Correctness agent *ran* the
gates, it could re-order or skip them. Keeping gates in middleware
means the verdict is structurally incapable of being tampered with by
any agent — the adversarial model includes the LLM itself. This
invariant is test-enforced in
`tests/test_agents/test_correctness.py` (see
`test_correctness_has_no_gate_tools`), which asserts Correctness's
`AgentSpec.tools` allowlist contains zero EXECUTION tools.
