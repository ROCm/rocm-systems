# Formal models

Small, self-contained models of state machines that the GPU tests can only sample and that
the GPU-free tests cannot reach at all. They are a development aid, not a build dependency:
without the checkers installed both ctest entries report `SKIP`.

| Model | Checker | File | Needs |
|---|---|---|---|
| Serialization refcount | CBMC | `refcount.c` | queue hooks |
| Per-pass service isolation | CBMC | `perpass.c` | kernel replay |
| Replay isolation window | TLC | `ReplayIsolation.tla` | kernel replay |

A model is only checked on a branch that carries the code it models. The callback-removal
lineage has no `kernel_replay`, so the two replay models are simply absent there and the
runner reports them as skipped rather than stubbing or duplicating them.

## Running

```bash
sudo apt-get install cbmc
curl -L -o tests/formal/tla2tools.jar \
  https://github.com/tlaplus/tlaplus/releases/latest/download/tla2tools.jar

ctest -R test-formal            # or:
python3 tests/formal/run_formal.py cbmc
python3 tests/formal/run_formal.py tlc
```

## Two of these models are *expected* to report a violation

That is the point of them. Each pins a known gap so that a change in its shape surfaces as a
test failure rather than as a surprise in production. `run_formal.py` declares the verdict it
expects for every case and fails if the verdict changes in either direction — including a gap
that silently closes, which usually means the model stopped modelling the real code.

**`refcount.c` — the serialization refcount.** With every caller pairing acquire and release,
three properties hold: the serializer never drifts from the refcount, counts never go negative,
and an agent stays serialized while any client still holds a claim on it. Allow one unbalanced
release (`-DALLOW_UNBALANCED`) and the third property fails in two steps: client C acquires for
`{gpu0, gpu1}`, client A releases for `{gpu0}` having never acquired, and C loses serialization it
still depends on. The saturating decrement prevents underflow but not theft. Every current caller
is balanced, so this is not a live defect — it records that the correctness argument rests on a
caller contract that nothing enforces.

**`perpass.c` — one service per replay pass.** Among the override-aware services (counters, SPM,
thread trace) at most one collects per pass, and that is proven. Add PC sampling, which is
agent-wide and never consults `local_context_override()`, and the property fails. Because a
replayed dispatch reserves one dispatch id for all its passes, an N-pass replay attributes about N
times the PC samples to a single dispatch.

**`ReplayIsolation.tla` — the replay window.** With the per-agent reader/writer lock and the
agent-wide drain in place, no ordinary dispatch is ever in flight between `snap()` and the final
`restore()`; TLC confirms this exhaustively. Setting `BYPASS = TRUE` models the HIP-graph fast path
skipping the reader lock and `async_started()`, and the invariant fails in five states — which is
exactly why that fast path is excluded whenever a replay service is active.

## Scope

These models abstract away real HSA and aqlprofile behaviour, compiled memory ordering, and the
hardware. CBMC's results are bounded (up to 4 clients, 3 agents, 8 operations); the TLA+ model is
one agent with an abstracted lock. They say the mechanisms are sound under their stated contracts.
They cannot say the contracts are correctly wired to the hardware — that is what the GPU tests are
for.
