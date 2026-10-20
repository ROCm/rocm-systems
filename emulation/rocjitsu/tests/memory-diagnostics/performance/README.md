# Address-race performance workloads

These standalone HIP programs are the synthetic workloads used by the address
race-check performance analyses:

- [`race_bench.hip`](race_bench.hip) contains the stream, LDS reduction, tiled
  GEMM, histogram, and compute-control kernels.
- [`hazard_isolation.hip`](hazard_isolation.hip) isolates the global gate,
  global shadow, LDS lifecycle, barrier, and LDS history paths.
- [`direct_to_lds_bench.hip`](direct_to_lds_bench.hip) repeatedly executes a
  gfx950 direct-to-LDS transfer.
- [`tensor_dma_bench.hip`](tensor_dma_bench.hip) repeatedly executes a gfx1250
  tensor-DMA transfer.

Each program contains its own host launch and result validation. It prints one
`kernel_ms=<time>` value for the device-synchronized timed region and prints
`PASS` only after validating the output.

See
[Address-race checks: the 2% decision](../../../docs/address-race-overhead.md)
and
[Core race families versus the legacy plugin](../../../docs/synthetic-race-performance-comparison.md)
for the measured results and statistical method.

## Build

Use the same ROCm SDK that built the simulator. For example:

```bash
src=tests/memory-diagnostics/performance
out=/tmp/rj-race-perf
mkdir -p "$out"

"$AMDCXX" -x hip --offload-arch=gfx950 --rocm-path="$ROCM_PATH" \
  -std=c++20 -O2 "$src/race_bench.hip" -o "$out/race_bench_gfx950"
"$AMDCXX" -x hip --offload-arch=gfx950 --rocm-path="$ROCM_PATH" \
  -std=c++20 -O2 "$src/hazard_isolation.hip" -o "$out/hazard_isolation_gfx950"
"$AMDCXX" -x hip --offload-arch=gfx950 --rocm-path="$ROCM_PATH" \
  -std=c++20 -O2 "$src/direct_to_lds_bench.hip" -o "$out/direct_to_lds_gfx950"
"$AMDCXX" -x hip --offload-arch=gfx1250 --rocm-path="$ROCM_PATH" \
  -std=c++20 -O2 "$src/tensor_dma_bench.hip" -o "$out/tensor_dma_gfx1250"
```

Record the compiler, SDK, simulator, config, and workload hashes with every
campaign. Do not compare binaries built by different toolchains.

## General synthetic shapes

The reported small and large sets use the following arguments:

| Workload | Small | Large |
|---|---:|---:|
| Stream | `stream 262144` | `stream 1048576` |
| LDS reduction | `lds_reduce 262144` | `lds_reduce 1048576` |
| GEMM | `gemm 128` | `gemm 256` |
| Histogram | `histogram 262144` | `histogram 1048576` |

For example, one large race-off measurement is:

```bash
taskset -c "$CPUS" "$ROCJITSU" --config "$CONFIG" \
  --race-checking=off -- "$out/race_bench_gfx950" stream 1048576
```

Select one core family from the same launcher binary with:

```bash
taskset -c "$CPUS" "$ROCJITSU" --config "$CONFIG" \
  --race-checking=on --race-checking-checks=global -- \
  "$out/race_bench_gfx950" stream 1048576
```

The family campaigns rotate race-off and selected-family processes rather than
running every sample of one configuration consecutively. Use at least nine
repetitions for the general synthetic matrix. Pair repetitions within each
workload, take the ratio of variant and race-off medians, and geometrically
average the workload ratios. The analyses use 20,000 paired bootstrap samples
for confidence intervals.

## Focused shapes

The focused address-domain campaign uses:

| Mode | Rounds |
|---|---:|
| `global_gate` | 8 |
| `global_full` | 8 |
| `lds_lifecycle` | 4096 |
| `barrier_only` | 65536 |
| `lds_only` | 65536 |
| `lds_barrier` | 32768 |

The direct-to-LDS and tensor-DMA source campaigns use 1,024 rounds and 15
rotating repetitions. A source selector must include at least one address
domain, for example:

```bash
taskset -c "$CPUS" "$ROCJITSU" --config "$CONFIG" \
  --race-checking=on \
  --race-checking-checks=global,lds-same-wave,lds-cross-wave,direct-to-lds \
  -- "$out/direct_to_lds_gfx950" 1024
```

Use a gfx1250 simulator config and replace `direct-to-lds` with `tensor-dma`
for the tensor workload.

## Plugin comparison

Enable the legacy race detector through a config whose top level contains:

```json
{
  "require_all_plugins": true,
  "plugins": {
    "race": {}
  },
  "sinks": {
    "types": ["stderr"]
  }
}
```

Run that config with `--race-checking=off`. `RJ_RACE=1` is not a supported
fallback. Confirm activation from the plugin's kernel-dispatch record before
including the timing.

The plugin and core checker are not coverage-equivalent. The plugin primarily
models in-flight memory/register wait hazards and LDS overlap; it does not
provide the core global-address shadow or tensor-DMA memory-access observation.
