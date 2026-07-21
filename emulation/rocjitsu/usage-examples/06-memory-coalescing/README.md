# Example 6: Instruction/Hook Profiling (strided vs coalesced)

## Objective

Run a strided and a coalesced kernel under rocjitsu's **hook profiler** and read
the per-kernel `HOOK_PROFILE` summary.

> **Important — rocjitsu does not model memory coalescing.** It is a functional
> (correctness) emulator, not a bandwidth/timing model. There are **no**
> memory-transaction counts, cache-line metrics, or cycle timings. This example
> therefore does **not** show a coalescing/bandwidth difference; see
> ["What you can and cannot learn"](#what-you-can-and-cannot-learn) below.

## Key Concepts

- The hook profiler (`RJ_USE_PROFILED_EXECUTION_PLUGIN_GROUP=1`) counts
  **emulator hook invocations** and estimates **CPU** time per hook.
- Strided vs coalesced access affects **instruction count** (address math), not
  the emulator's memory-transaction count.

## Files

- `src/memory_pattern.cpp` — a single binary with both kernels
  (`strided_access` and `coalesced_access`) plus a `main` that launches each.
- `Makefile`

## Kernels

```cpp
// Strided (extra address arithmetic: i * stride)
__global__ void strided_access(float *out, float *in, int N, int stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) out[i] = in[i * stride];
}

// Coalesced (contiguous)
__global__ void coalesced_access(float *out, float *in, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) out[i] = in[i];
}
```

Both launch `<<<16, 64>>>` = 1024 threads (16 workgroups, 16 wavefronts on wave64).

## Build and run

```bash
cd usage-examples/06-memory-coalescing
make
make run          # plain functional run (app output only)
make run-profile  # run with the hook profiler; also saved to logs/profile.log
```

`make run-profile` enables `RJ_USE_PROFILED_EXECUTION_PLUGIN_GROUP=1`. The
profiler writes its `HOOK_PROFILE` summary to **stderr**, so the Makefile tees
stderr into `logs/profile.log` (see the note in ["logs/"](#the-logs-directory)).

## Expected output

Plain run:

```text
Testing memory access patterns...
Strided access: completed
Coalesced access: completed
```

Profiling run — one `HOOK_PROFILE` block per kernel (verbatim from
`sharkmi300x-4`, ROCm 7.2.1):

```text
HOOK_PROFILE --- ? ---
HOOK_PROFILE beforeExecuteInstruction        calls=384           est_total=0.0 ms
HOOK_PROFILE afterExecuteInstruction         calls=383           est_total=0.0 ms
HOOK_PROFILE readVgpr                        calls=5264          est_total=0.3 ms
HOOK_PROFILE readSgpr                        calls=256           est_total=0.0 ms
HOOK_PROFILE routeMemoryInstruction          calls=80            est_total=0.0 ms
HOOK_PROFILE workgroupDispatched             calls=16            est_total=0.0 ms
HOOK_PROFILE workgroupCompleted              calls=16            est_total=0.0 ms
HOOK_PROFILE wavefrontDispatched             calls=16            est_total=0.0 ms
HOOK_PROFILE wavefrontHalted                 calls=16            est_total=0.0 ms
HOOK_PROFILE ---
Strided access: completed
HOOK_PROFILE --- ? ---
HOOK_PROFILE beforeExecuteInstruction        calls=304           est_total=0.0 ms
HOOK_PROFILE afterExecuteInstruction         calls=303           est_total=0.0 ms
HOOK_PROFILE readVgpr                        calls=5216          est_total=0.3 ms
HOOK_PROFILE readSgpr                        calls=240           est_total=0.0 ms
HOOK_PROFILE routeMemoryInstruction          calls=80            est_total=0.0 ms
HOOK_PROFILE workgroupDispatched             calls=16            est_total=0.0 ms
HOOK_PROFILE workgroupCompleted              calls=16            est_total=0.0 ms
HOOK_PROFILE wavefrontDispatched             calls=16            est_total=0.0 ms
HOOK_PROFILE wavefrontHalted                 calls=16            est_total=0.0 ms
HOOK_PROFILE ---
Coalesced access: completed
```

### Reading each line

| Field | Meaning |
|---|---|
| `--- ? ---` | Kernel-name header; `?` = symbol name not resolved |
| `beforeExecuteInstruction` / `afterExecuteInstruction` | ISA instructions executed (÷16 waves ≈ per-wave count) |
| `readVgpr` / `readSgpr` | Vector / scalar register reads during execution |
| `routeMemoryInstruction` | Memory instructions (loads/stores) routed |
| `workgroupDispatched` / `workgroupCompleted` | Workgroups (blocks) started / finished |
| `wavefrontDispatched` / `wavefrontHalted` | Wavefronts started / finished |
| `calls=` | Exact invocation count for that hook |
| `est_total=… ms` | Sampled **CPU** time in that hook (emulator overhead, **not** GPU time) |

## What you can and cannot learn

**Can learn:**
- **Launch geometry** — 16 workgroups, 16 wavefronts, ~24 (strided) vs ~19
  (coalesced) instructions per wave.
- **Extra work in the strided kernel** — more instructions (384 vs 304) and SGPR
  reads (256 vs 240) from computing `i * stride` and reading the `stride` arg.
- **Emulator hotspots** — `readVgpr` dominates the emulator's CPU cost (~0.3 ms).

**Cannot learn:**
- **Anything about coalescing or bandwidth.** `routeMemoryInstruction` is **80 for
  both** kernels — the emulator routes the same number of memory instructions
  regardless of address pattern. It does not count memory transactions, model
  cache lines, or produce real timings. `est_total` is CPU time spent in emulator
  hooks, not device execution time.

## The `logs/` directory

`make run-profile` writes `logs/profile.log`. Note this is done by **teeing
stderr** in the Makefile, not by rocjitsu's file sink: the profiler group emits
`HOOK_PROFILE` to stderr and does **not** honor `RJ_SINKS=file` /
`RJ_SINK_DIR` (unlike, e.g., the race detector's `race.log` in Example 4). See
the tracker (`../rocjitsu_questions.md`) for details.

## Key takeaways

- rocjitsu profiling = **emulator hook counts + CPU time**, not GPU performance.
- Use it to understand instruction mix and launch geometry, **not** to measure
  coalescing, bandwidth, or latency.
- For real memory-performance analysis, use hardware tools (e.g. `rocprofv3` /
  `omniperf`) on an actual GPU.

## Related

- [Example 4: LDS Data Race](../04-data-race-simple/) — a plugin that **does** use file sinks
- [race-detector.md](../../docs/race-detector.md) — "a correctness tool, not a performance tool"
