# llx_latency_bench

Measures `T(GPU0 -> GPU1)` and one-way bandwidth for four RCCL-style LL
flag/data signaling protocols between two GPUs on the same node, sweeping
message sizes from 8 B to 128 MB.

The send is one-way: GPU0 streams data, GPU1 polls each iteration's flag.
A single 8-byte ACK per iteration from GPU1 back to GPU0 provides the
minimum backpressure needed to keep GPU0 from overwriting unread flags.
Per-iter cost ≈ `T(A->B) + ~1 µs` for the ACK round trip.

## Protocols

| Protocol | Line size | Wire efficiency | Ordering mechanism |
|---|---|---|---|
| `LL` | 16 B | 50% | Data and flag packed into each 64-bit store; receiver checks both flag halves |
| `LL<N>-sc` | N bytes | (N−8)/N | `s_waitcnt vmcnt(0)` drains the wavefront's data stores before the flag lane writes the flag; receiver uses `__any()` ballot |
| `LL<N>-sc+wb` | N bytes | (N−8)/N | Same as `sc` but adds `__threadfence_system()` per line between `s_waitcnt` and the flag store (extra cache writeback) |
| `LL<N>-tf` | N bytes | (N−8)/N | One `__threadfence_system()` per iteration between a bulk data pass and a bulk flag pass; receiver polls flag then re-reads data |

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
make llx_latency_bench_64
make llx_latency_bench_128
make llx_latency_bench_256
```

Requires ROCm in `$ROCM_PATH` (default `/opt/rocm`). The arch is auto-detected
via `--offload-arch=native`. The `.c` source is compiled as HIP via `-x hip`.

## Run

```bash
./llx_latency_bench_64  [warmup [iters [timeout_s [max_bytes]]]]
./llx_latency_bench_128 [warmup [iters [timeout_s [max_bytes]]]]
./llx_latency_bench_256 [warmup [iters [timeout_s [max_bytes]]]]
```

Defaults: warmup=10, iters=100, timeout=30 s, max_bytes=128M.
`max_bytes` accepts K/M/G suffixes (e.g. `16M`, `1G`). `iters` and
`warmup` must be ≤ 4096 (size of the per-iter ACK counter array).

## Output

```
LL128-sc      16777216 B  nlines=139811  nb=64  T(A->B)=  377.65 us  BW= 44.426 GB/s  [OK]
```

- `nlines` — number of wire lines transferred per iteration
- `nb` — thread blocks launched (scales with message size, capped at 64)
- `T(A->B)` — average GPU0 -> GPU1 transfer time per iteration in microseconds
  (includes the per-iter 8-byte ACK round trip; ≈1 µs of overhead, negligible
  beyond ~1 KB)
- `BW` — `data_bytes / T(A->B)` in GB/s. Bounded by the per-link unidirectional
  rate by construction.
- `[OK]` / `[ERRORS: N]` — result of the correctness verification pass

## Correctness verification

Between warmup and timing, one iteration runs with a non-zero deterministic
payload (a hash of group index, lane, and flag value). The receiver checks
every recovered data word and reports any mismatch. This catches stale or
zeroed data that a zero-initialized payload would miss.
