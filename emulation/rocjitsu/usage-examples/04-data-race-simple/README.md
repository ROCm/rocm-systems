# Example 4: LDS Data Race Detection

## Objective

Learn to detect and fix **intra-workgroup LDS races** with rocjitsu's race detector (`RJ_RACE=1`):
- Missing `__syncthreads()` before cross-wave shared-memory reads
- Interpreting real `RACE type=LDS` reports
- Fixing with a barrier; verifying with host golden checks

## What rocjitsu detects (and what it does not)

| Detected by `RJ_RACE=1` | **Not** detected |
|---|---|
| LDS read/write without `s_barrier` / `__syncthreads()` within a workgroup | Inter-workgroup global races (`bins[bin]++` across blocks) |
| VGPR / SGPR hazards (missing `s_waitcnt`) | Host–device sync bugs |

Global histogram races without atomics are covered in [Example 5](../05-global-memory-race/).

## The bug

Each thread stores its bin index in `__shared__ thread_bin[]`, then reads **another wave's slot** before a barrier:

```cpp
__shared__ int thread_bin[128];
// ...
thread_bin[tid] = data[gid] % num_bins;
// BUG: missing __syncthreads()
int peer = (tid + 64) % 128;
int bin = thread_bin[peer];          // cross-wave LDS read — race
atomicAdd(&bins[bin], 1);
```

**Block size must be 128** (2 waves of 64 on CDNA) so wave 1 reads slots wave 0 may still be writing.

### Fixed code

```cpp
thread_bin[tid] = data[gid] % num_bins;
__syncthreads();                     // FIXED
int bin = thread_bin[tid];           // read own slot after barrier
atomicAdd(&bins[bin], 1);
```

## Files

- `src/histogram_race.cpp` — missing barrier (LDS race)
- `src/histogram_fixed.cpp` — adds `__syncthreads()`
- `Makefile` — build and run targets

## Build

```bash
cd usage-examples/04-data-race-simple
make
```

## Running

### 1. Buggy kernel — wrong bins + LDS race reports

```bash
make run-race
```

Equivalent:

```bash
RJ_RACE=1 rocjitsu --config ../../configs/amdgpu_cdna4_kmd.json -- ./build/histogram_race
```

### 2. Fixed kernel — correct bins, no races

```bash
make run-fixed
```

### 3. Buggy kernel without detector (silent wrong bins)

```bash
make run-race-no-detector
```

### 4. Save race log to file

```bash
make run-to-file
# → logs/race.log
```

## Expected output

Output below is copied verbatim from a run on `sharkmi300x-4` (ROCm 7.2.1,
`amdgpu_cdna4_kmd.json`) and is deterministic across runs.

### Buggy + `RJ_RACE=1`

Race reports go to **stderr** (or `race.log` with `RJ_SINKS=file`):

```text
[rocjitsu] Race detection enabled (RJ_RACE)
[rocjitsu] Kernel dispatch: "?"
[rocjitsu] Kernel dispatch: "?"
RACE type=LDS reg=256 wave=0 lane=0 wg=7,0,0 conflict=unknown
Race on LDS byte 256 [workgroup (7, 0, 0), wave 0, lane 0]
  ==>  0x5400021a10  ds_write_b32 v1, v2  ; <-- wave 1
  ==>  0x5400021a18  ds_read_b32 v0, v0  ; <-- wave 0 lane 0
END_RACE
[rocjitsu] Kernel dispatch: "?"
```

Reading the report:

- `type=LDS` — a shared-memory (LDS) hazard; `reg=256` is the **LDS byte
  address**, not a register.
- `wave=0 lane=0 wg=7,0,0` — the read that observed the unsynchronized write.
- `ds_write_b32 ... ; <-- wave 1` wrote the byte; `ds_read_b32 ... ; <-- wave 0`
  read it with no `s_barrier` in between. The leading `0x5400...` is the PC.
- `conflict=unknown` — this header field is currently hardcoded; the actual
  conflicting instruction is shown on the `==>` lines.

**Only one block appears** even though every workgroup has the bug: the plugin
de-duplicates by PC pair, so the first workgroup to hit this read/write PC pair
is reported (here `wg=7,0,0`) and the rest are suppressed.

Application stdout (host golden check):

```text
Histogram Example - LDS SYNC BUG (for RJ_RACE=1)
  Input size: 10000 elements
  Number of bins: 256
  Block size: 128 (2 waves per block)
  Run with: RJ_RACE=1 rocjitsu -- ... ./build/histogram_race

Launching kernel: grid(79, 1, 1), block(128, 1, 1)

  Bin 0: expected=41, actual=57
  Bin 19: expected=37, actual=35
  Bin 20: expected=47, actual=46
  Bin 37: expected=53, actual=52
  Bin 38: expected=34, actual=33
Verification: FAILED
  Expected sum: 10000
  Actual sum: 10000
  Mismatched bins: 16

Incorrect histogram — stale peer reads from missing __syncthreads().
Check stderr for RJ_RACE=1 reports (RACE type=LDS ... END_RACE).
```

Note the **total sum stays 10000** while **16 bins hold the wrong counts** —
each thread still does exactly one `atomicAdd`, just to the wrong bin because it
read a peer slot before that peer had written it. A sum check alone would miss
this; `RJ_RACE=1` points straight at the cause.

### Fixed + `RJ_RACE=1`

```text
Histogram Example - FIXED (LDS barrier before peer read)
  Input size: 10000 elements
  Number of bins: 256
  Block size: 128

Launching kernel: grid(79, 1, 1), block(128, 1, 1)

Verification: PASSED
  Expected sum: 10000
  Actual sum: 10000
  All histogram bins correct!

NO LDS RACES expected under RJ_RACE=1.
```

No `RACE` / `END_RACE` lines on stderr.

## Why the sum still equals 10000

This is the instructive part of the example: the histogram is **wrong** yet the
total is **exactly right**. A sum check would pass while the data is corrupt.

**The sum only counts how many `atomicAdd` calls happened — not where they
landed.** Every in-range thread runs exactly one increment:

```cpp
int bin = thread_bin[peer];   // may be the WRONG bin (unsynchronized read)
if (gid < N)
  atomicAdd(&bins[bin], 1);   // but still exactly one atomic +1
```

- 10000 threads have `gid < N` → exactly 10000 increments.
- `atomicAdd` is atomic, so **no increment is ever lost**.
- ⇒ `sum(bins) == 10000` no matter what `bin` holds. The bug corrupts *which*
  bin each `+1` targets, not *how many* there are — "wrong data, right count."

> Contrast with a `bins[bin]++` global race (no atomics, [Example 5](../05-global-memory-race/)):
> there the read-modify-write itself races and `+1`s are genuinely **lost**, so the
> sum drops **below** 10000. Here the final write is atomic, so only the
> distribution is wrong.

**Where the 16 wrong bins come from — the last, partial block.** The grid is
`79 × 128 = 10112` threads but only 10000 are in range. Each thread counts its
*peer's* bin, `thread_bin[(tid+64)%128]`:

- The 78 **full** blocks stay correct: `peer` just swaps the two 64-lane halves,
  a permutation, so the per-block multiset of counted bins is unchanged.
- **Block 78 is partial:** only `tid 0..15` are in range (gids 9984–9999); the
  rest took the `else` branch and stored `0`. Those 16 valid threads read peer
  slots `64..79` — all `0` — so all 16 land in **bin 0** (`+16`), while the 16
  elements' true bins each lose a count (`−16` across 15 bins).

`+16` and `−16` cancel → **sum unchanged, 16 bins wrong**. The takeaway: totals
and checksums can hide real corruption; `RJ_RACE=1` flags the missing barrier
directly.

## Debugging workflow

1. `make run-race-no-detector` — confirm wrong per-bin counts (no race lines).
2. `RJ_RACE=1 make run-race` — read `RACE type=LDS` reports; note `ds_write` vs `ds_read` without barrier.
3. `make run-fixed` — confirm PASSED and no race lines.
4. For global races across workgroups, see Example 5 and use atomics or privatization.

## Key environment variables

| Variable | Purpose |
|---|---|
| `RJ_RACE=1` | **Enable** race detector (not `RJ_SINKS=race_detector`) |
| `RJ_SINKS=file RJ_SINK_DIR=logs` | Write reports to `race.log` |
| `RJ_LOG=1` | Kernel dispatch metadata |

See [race-detector.md](../../docs/race-detector.md) for full scope and limitations.

## Key takeaways

- Use **`RJ_RACE=1`** to enable the race detector plugin.
- The detector targets **intra-workgroup LDS/VGPR/SGPR** sync hazards.
- A missing `__syncthreads()` before shared reads can corrupt results even when the total element count looks fine.
- Always cross-check with a **host golden** reference.

## Next steps

- [Example 5: Global Memory Races](../05-global-memory-race/) — inter-workgroup global RMW
- [race-detector.md](../../docs/race-detector.md) — report format and limitations
