# Kernel replay samples

Custom tools that subscribe to `ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY`.
These samples preload a client `.so`; they do not use `rocprofv3`.

Build with `-DROCPROFILER_BUILD_SAMPLES=ON`.

## What they show

| Sample | Passes | Services |
|---|---|---|
| `kernel-replay-basic` | 4 | Replay only. The app still sees one kernel completion. |
| `kernel-replay-counters` | 3 | Dispatch counters; each pass selects a different counter configuration. |
| `kernel-replay-counters-then-pc-sampling` | 4 | Counters on passes 0–2, PC sampling on pass 3 only. |
| `kernel-replay-services-first` | 5 | PC sampling → ATT → SPM → counters → counters. |
| `kernel-replay-services-last` | 5 | Counters → counters → PC sampling → ATT → SPM. |
| `kernel-replay-opt-out` | 3 / 1 | Replays the `bump` kernel (`block.x == 67`); leaves `nudge` (`block.x == 64`) unreplayed. |

## Incompatible services use separate passes

Dispatch counter collection turns clock gating back on around a kernel. PC sampling on MI2xx/MI3xx requires clock gating off. Running both on the **same** replay pass can hang the GPU.

ATT and SPM also cannot safely share one pass because both inject AQL instrumentation around the
dispatch. SPM and PC sampling compete for SQ/performance-monitor resources. Therefore “PC sampling
plus ATT plus SPM first/last” means three adjacent, exclusive passes—not all three services on one
pass.

Use **separate contexts** and **separate passes**:

1. Start the counter context globally.
2. Configure PC sampling but leave that context globally stopped.
3. On the last `PASS PHASE_ENTER`, locally stop counters and locally start PC sampling.
4. Kernel replay starts PC sampling on the replaying agent only, then restores the globally stopped state.

`ROCPROFILER_PC_SAMPLING_BETA_ENABLED=ON` is required for the PC-sampling sample.
The service-sequence samples additionally require `ROCPROFILER_SPM_BETA_ENABLED=True`.

Run the service-order examples directly:

```bash
KR_SERVICE_ORDER=services-first \
ROCPROFILER_PC_SAMPLING_BETA_ENABLED=ON \
ROCPROFILER_SPM_BETA_ENABLED=True \
LD_PRELOAD=./libkernel-replay-services-first-client.so \
./kernel-replay-services-first

KR_SERVICE_ORDER=services-last \
ROCPROFILER_PC_SAMPLING_BETA_ENABLED=ON \
ROCPROFILER_SPM_BETA_ENABLED=True \
LD_PRELOAD=./libkernel-replay-services-last-client.so \
./kernel-replay-services-last
```

## Run

From the sample build directory:

```bash
ctest -R kernel-replay --output-on-failure
```

Or:

```bash
export LD_PRELOAD=./libkernel-replay-basic-client.so
./kernel-replay-basic
```
