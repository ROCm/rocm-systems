<!-- Copyright (c) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->

# HSA memory registry contention probe

`registry_contention.py` measures how long a metadata query waits while a
second thread runs host `hsa_memory_copy` or `hsa_amd_memory_fill`. Both paths
currently retain the process registry lock through their CPU work. The probe
compares each workload with its own query-only baseline and prints p50, p95,
p99, and maximum query latency, operation count, and elapsed time. `ctypes`
releases the Python GIL during C calls, so the query and copy/fill calls can
overlap.

Build the shared image from `runtimes/`, then run on a GPU/KFD host qualified
for this early-access runtime:

```sh
cargo build --workspace --all-features --frozen
python3 ddi/libhsa/tests/perf/registry_contention.py \
  --library target/debug/librocddi_runtime.so --mib 64 --operations 64
```

Run several trials with the same machine load and keep the raw CSV output.
Compare query latency, copy/fill throughput, and total elapsed time before and
after a proposed concurrency change. A change that releases the registry lock
must first retain every allocation or loader mapping reached by the CPU worker
until it finishes, including through shutdown and failed native cleanup.

The current development host returns `0x1000` from `hsa_init`; the probe exits
with code 77 in that case. No runtime contention result is inferred from that
skip. The Python timer includes FFI overhead, so use within-host comparisons
rather than treating its absolute microseconds as native API latency.
