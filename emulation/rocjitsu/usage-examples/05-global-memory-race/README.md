# Example 5: Global Memory Race — a limit of `RJ_RACE=1`

## Objective

Show a **real** data race that rocjitsu's race detector **does not** report, and
how to catch and fix it anyway. Many threads across multiple blocks do a
non-atomic read-modify-write on one global counter.

This is the deliberate counterpart to [Example 4](../04-data-race-simple/): that
LDS race **is** reported by `RJ_RACE=1`; this global race is **not**. Together
they draw the boundary of what the detector covers.

## What `RJ_RACE=1` does and does not catch

| Detected | **Not** detected |
|---|---|
| Intra-workgroup **LDS** hazards (missing `__syncthreads()`) | **Inter-workgroup global** races (this example) |
| **VGPR / SGPR** hazards (missing `s_waitcnt`) | Write-after-write (WAW) conflicts |

The detector is **per-workgroup** and tracks register/LDS synchronization; it has
no cross-workgroup view and does not track contention on a global address. See
[race-detector.md](../../docs/race-detector.md) ("Intra-workgroup only", "No WAW").

## The bug

```cpp
__global__ void sum_with_race(int *result, const int *data, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    *result += data[idx];   // RACE: non-atomic read-modify-write
  }
}
```

Every thread reads `*result`, adds, and writes back. With no atomic, concurrent
threads read the same value and overwrite each other → updates are lost.

### The fix

```cpp
atomicAdd(result, data[idx]);   // hardware-serialized read-modify-write
```

## Files

- `src/global_race.cpp` — non-atomic global counter (buggy) + host golden check
- `src/global_fixed.cpp` — `atomicAdd` fix
- `Makefile` — build and run targets

## Build and run

```bash
cd usage-examples/05-global-memory-race
make
make run-race              # buggy + RJ_RACE=1 (detector silent; host detects)
make run-race-no-detector  # buggy without detector (identical result)
make run-fixed             # atomicAdd fix
make compare               # side-by-side
```

## Expected output

Output below is verbatim from `sharkmi300x-4` (ROCm 7.2.1,
`amdgpu_cdna4_kmd.json`) and is deterministic across runs.

### Buggy + `RJ_RACE=1`

Detector output on **stderr** — note there are **no `RACE` lines**, only the
dispatch metadata:

```text
[rocjitsu] Race detection enabled (RJ_RACE)
[rocjitsu] Kernel dispatch: "?"
[rocjitsu] Kernel dispatch: "?"
[rocjitsu] Kernel dispatch: "?"
[rocjitsu] Kernel dispatch: "?"
```

Application stdout — the **host golden check** is what exposes the bug:

```text
Global Counter Example - MULTI-BLOCK RACE (host-detected)
  Elements: 1000 (each = 1)
  Launch: grid(10, 1, 1), block(100, 1, 1)
  Note: RJ_RACE=1 does NOT flag inter-block global RMW —
        this bug is caught by the host golden check below.

Verification: FAILED
  Expected sum: 1000
  Actual sum: 8
  Lost updates: 992

Lost updates from a non-atomic global RMW. RJ_RACE=1 stays silent
(inter-workgroup global races are out of scope); fix with atomicAdd
(see global_fixed.cpp) and re-check the host sum.
```

`make run-race-no-detector` produces the **same** `Actual sum: 8` — proving the
detector adds nothing for this class of bug.

### Fixed + `RJ_RACE=1`

```text
Global Counter Example - FIXED (atomicAdd)
  Elements: 1000 (each = 1)
  Launch: grid(10, 1, 1), block(100, 1, 1)

Verification: PASSED
  Expected sum: 1000
  Actual sum: 1000
  All updates accounted for (atomicAdd).
```

Still no `RACE` lines — the detector is silent for the fixed version too. The
**host check** is what proves the fix.

## Debugging workflow

1. `make run-race` — notice `RJ_RACE=1` reports nothing, yet the host sum is wrong.
2. Recognize the shape: a shared global counter updated without atomics → this is
   an **inter-workgroup global** race, outside the detector's scope.
3. Fix with `atomicAdd` (or a two-level reduction: block-local partial in shared
   memory, then one atomic per block).
4. `make run-fixed` — host check passes.

## How to actually catch this class of bug

`RJ_RACE=1` will not help here. Use:

- **Host golden reference** (as in this example) — compare device result to a CPU
  computation.
- **`atomicAdd` / atomics** for shared counters and accumulators.
- **Privatization** — per-block partial sums, then a single atomic merge, to
  reduce contention.

## Key takeaways

- A silent race detector does **not** mean race-free code.
- `RJ_RACE=1` targets **intra-workgroup** LDS/VGPR/SGPR sync hazards, not global
  inter-block contention or WAW.
- Global counters need **atomics**; verify with a **host golden** check.

## Related

- [Example 4: LDS Data Race](../04-data-race-simple/) — a race `RJ_RACE=1` **does** report
- [race-detector.md](../../docs/race-detector.md) — detector scope and limitations
