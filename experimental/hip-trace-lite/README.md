# hip-trace-lite

Minimal LD_PRELOAD library that registers a HIP tracer callback via
`hipRegisterTracerCallback` and dumps `activity_record_t` records to a
binary file. Built for benchmarking the CLR-callback delivery mechanism.

See `design.md` for the design and `plan.md` for the build steps.

## What it captures

- **HIP_OPS** records (kernel completions and async copies as exposed by
  CLR) — always.
- **HIP_API** records (sync API entry/exit) — only when
  `HTL_TRACE_API=1`.

## What it does NOT do

- Correlate HIP_API ↔ HIP_OPS records
- Collect PMC/derived counters
- Intercept HSA queues itself (CLR did that already; we just receive the
  pre-baked records)
- Coexist with other tools that call `hipRegisterTracerCallback` — see
  caveats below.

## Build

Requires CMake 3.16+, a C++17 compiler, libpthread, libdl. HIP is needed
only for the smoke test; the shared library and decoder build with no
ROCm dependency.

```bash
cmake -B build experimental/hip-trace-lite
cmake --build build -j
```

Artifacts:
- `build/libhiptracelite.so`
- `build/htl_dump`
- `build/test_ring`, `build/test_record` (unit tests)
- `build/smoke` (only if HIP was found by CMake)

## Run

Drop-in: preload the library against any HIP application.

```bash
LD_PRELOAD=$PWD/build/libhiptracelite.so ./your_hip_app
```

Choose where the trace lands:

```bash
HTL_OUTPUT_FILE=/tmp/run.htl \
LD_PRELOAD=$PWD/build/libhiptracelite.so \
./your_hip_app
```

Also capture HIP API entry/exit spans:

```bash
HTL_TRACE_API=1 \
HTL_OUTPUT_FILE=/tmp/run.htl \
LD_PRELOAD=$PWD/build/libhiptracelite.so \
./your_hip_app
```

Decode to CSV:

```bash
./build/htl_dump /tmp/run.htl > run.csv
head run.csv
```

Run the smoke test (built only if HIP was found by CMake):

```bash
LD_PRELOAD=$PWD/build/libhiptracelite.so \
HTL_OUTPUT_FILE=/tmp/smoke.htl \
./build/smoke
./build/htl_dump /tmp/smoke.htl
```

You should see `[hip-trace-lite] registered, output=...` on stderr at
startup and `[hip-trace-lite] shutdown: N records written, 0 dropped` at
exit. The decoded CSV must contain at least one `domain=2,op=0` row whose
`kernel` field is non-empty.

## Environment variables

| Var | Default | Effect |
|---|---|---|
| `HTL_OUTPUT_FILE` | `./hiptrace.bin` | Output file path (truncated on open) |
| `HTL_TRACE_API`   | `0`              | When `1`, also capture HIP_API records |

## Caveats

- **Single-slot collision.** CLR's `hipRegisterTracerCallback` writes one
  atomic function pointer. If rocprofiler-sdk, rocprofv1/v2, legacy
  roctracer, or any other tracer also tries to register, the last writer
  wins and the other tool's records vanish silently. **Run isolated.**
- **No correlation joining.** `correlation_id` is recorded as-is; pairing
  HIP_API with HIP_OPS is left to downstream analysis.
- **Async-copy `bytes` field** is taken straight from the CLR record;
  CLR's semantics apply (typically the linear size of the copy).
- **Callback runs on the rocclr completion thread.** The callback is hot:
  it does only an atomic ring enqueue and a `strncpy` of the kernel name.
  The writer thread does all I/O.
- **Backpressure.** If the ring fills (default 65536 slots), records are
  dropped and counted in the file footer. Increase `HTL_RING_SIZE` (when
  exposed in a later iteration) or reduce the producer rate.

## Files

```
experimental/hip-trace-lite/
├── design.md             design + format spec
├── plan.md               step-by-step implementation plan
├── README.md             this file
├── CMakeLists.txt
├── src/                  library sources
├── tools/htl_dump.cpp    offline decoder
└── test/                 unit + smoke tests
```
