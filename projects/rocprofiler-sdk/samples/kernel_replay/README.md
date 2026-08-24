# Kernel replay samples

Small **custom tools** plus a HIP application, following the same layout as
`samples/counter_collection/`: each sample is a shared-library client (`.so`) preloaded
onto a minimal app (`main.cpp`). These are **not** `rocprofv3` integration tests.

Build with `-DROCPROFILER_BUILD_SAMPLES=ON`.

## Samples vs integration tests

| | **Samples** (`samples/kernel_replay/`) | **Integration tests** (`tests/kernel-replay-*`) |
|---|---|---|
| Purpose | Teach tool authors: subscribe to replay, wire one service | Regression coverage (local toggles, concurrency) |
| Tool | Small LD_PRELOAD client with only the feature under demo | Env-driven client in `tests/` |
| App | Shared `main.cpp` HIP kernels | Dedicated test app per suite |
| Run | `ctest -R '^kernel-replay-'` in the samples build dir | `ctest -R kernel-replay-local-context` in main build |

For JSON output validation with the shared test harness, see `tests/counter-collection/`
(`rocprofiler-sdk-json-tool`). Kernel replay JSON/tool wiring is planned for the stacked
`rocprofv3` PR (#10439), not these SDK samples.

## What each sample shows

Start with the **basic** samples — each client is small and only wires kernel replay plus one
service (same idea as `samples/counter_collection/`).

| Sample | Passes | Services |
|---|---|---|
| `kernel-replay-basic` | 4 | Replay only. The app still sees one kernel completion. |
| `kernel-replay-counters` | 3 | Dispatch counters; each pass selects a different counter configuration. |
| `kernel-replay-counters-then-pc-sampling` | 4 | Counters on passes 0–2, PC sampling on pass 3 only. |
| `kernel-replay-att` | 2 | Counters on pass 0, ATT on pass 1. |
| `kernel-replay-spm` | 2 | Counters on pass 0, SPM on pass 1. |
| `kernel-replay-opt-out` | 3 / 1 | Replays the `bump` kernel (`block.x == 67`); leaves `nudge` unreplayed. |
| `kernel-replay-early-exit` | 4 / 2 | Sets `replay_continue_cb` to stop after pass 1 even though `pass_count_cb` returns 4. |

### Advanced: multi-service pass ordering

These two targets share one larger client (`service_sequence_client.cpp`) that runs **five**
passes with PC sampling, ATT, SPM, and counters in a fixed order. Use them when you want to see
how incompatible services are kept on separate passes in a single replay loop.

| Sample | Pass order (5 passes) |
|---|---|
| `kernel-replay-services-first` | PC sampling → ATT → SPM → counters → counters |
| `kernel-replay-services-last` | Counters → counters → PC sampling → ATT → SPM |

Set `KR_SERVICE_ORDER=services-first` or `services-last` (the CTest targets set this for you).
Requires `ROCPROFILER_PC_SAMPLING_BETA_ENABLED=ON` and `ROCPROFILER_SPM_BETA_ENABLED=True`.

## Incompatible services use separate passes

Dispatch counter collection turns clock gating back on around a kernel. PC sampling on
MI2xx/MI3xx requires clock gating off. Running both on the **same** replay pass can hang the GPU.

ATT and SPM also cannot safely share one pass because both inject AQL instrumentation around the
dispatch. Use **separate contexts** and **separate passes** with local toggles.

`ROCPROFILER_PC_SAMPLING_BETA_ENABLED=ON` is required for the PC-sampling sample.
`ROCPROFILER_SPM_BETA_ENABLED=True` is required for the SPM sample.

## Run

From the sample build directory:

```bash
ctest -R '^kernel-replay-' --output-on-failure
```

Or manually:

```bash
export LD_PRELOAD=./libkernel-replay-counters-client.so
./kernel-replay-counters
```
