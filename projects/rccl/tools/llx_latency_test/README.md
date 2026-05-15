# llx_latency_test

Measures one-way latency and unidirectional bandwidth for three NCCL-style
LL flag/data signaling protocols between two GPUs on the same node, sweeping
message sizes from 8 B to 128 MB.

## Protocols

| Protocol | Line size | Wire efficiency | Ordering mechanism |
|---|---|---|---|
| `LL` | 16 B | 50% | Data and flag packed into each 64-bit store; receiver checks both flag halves |
| `LL<N>-sc` | N bytes | (N−8)/N | `s_waitcnt vmcnt(0)` drains the wavefront's data stores before the flag lane writes the flag; receiver uses `__any()` ballot |
| `LL<N>-tf` | N bytes | (N−8)/N | `__threadfence_system()` orders all of a block's data stores before the flag lane writes the flag; receiver polls flag then re-reads data |

Wire efficiencies for supported line sizes:

| LINE_BYTES | Data per line | Wire efficiency |
|---|---|---|
| 64 | 56 B (7 × 8 B) | 87.5% |
| 128 | 120 B (15 × 8 B) | 93.75% |
| 256 | 248 B (31 × 8 B) | 96.9% |

## Build

```bash
# All three line sizes
make -j

# Single line size
make llx_latency_test_64
make llx_latency_test_128
make llx_latency_test_256
```

Requires ROCm in `$ROCM_PATH` (default `/opt/rocm`). The arch is auto-detected
via `--offload-arch=native`.

## Run

```bash
./llx_latency_test_64  [warmup [iters [timeout_s [max_bytes]]]]
./llx_latency_test_128 [warmup [iters [timeout_s [max_bytes]]]]
./llx_latency_test_256 [warmup [iters [timeout_s [max_bytes]]]]
```

Defaults: warmup=10, iters=100, timeout=30 s, max_bytes=128M.
`max_bytes` accepts K/M/G suffixes (e.g. `16M`, `1G`).

## Output

```
LL<N>-sc    16777216 B  nlines=299594  nb=64  RTT=  392.18 us  BW=42.76 GB/s  [OK]
```

- `nlines` — number of wire lines transferred per direction
- `nb` — thread blocks launched (scales with message size, capped at 64)
- `RTT` — round-trip time averaged over `iters` iterations
- `BW` — effective per-direction bandwidth: `data_bytes / RTT`. For pipelined protocols
  (sc, sc+wb) both directions run simultaneously so RTT ≈ T\_one\_way and BW reflects
  the true per-link throughput. For sequential protocols (LL baseline) RTT ≈ 2×T\_one\_way
  so BW reflects half the link rate — the cost of serialization.
- `[OK]` / `[ERRORS: N]` — result of the correctness verification pass

## Correctness verification

Between warmup and timing, one iteration runs with a non-zero deterministic
payload (a hash of group index, lane, and flag value). The receiver checks
every recovered data word and reports any mismatch. This catches stale or
zeroed data that a zero-initialized payload would miss.
