# hip-trace-lite — Design

Date: 2026-04-20
Status: Approved (brainstorm); implementation plan pending
Location: `experimental/hip-trace-lite/`

## Purpose

A minimal LD_PRELOAD shared library that registers a HIP tracer callback via
`hipRegisterTracerCallback` and writes received `activity_record_t` records
to a binary file. Built specifically to **benchmark the CLR-callback delivery
mechanism** in isolation — no correlation joining, no PMC counters, no
HSA-side queue intercept. The benchmarking harness itself is a separate task
and is **out of scope** for this spec.

## Non-goals

- Production tracer or replacement for roctracer / rocprofiler-sdk
- Correlation between HIP_API and HIP_OPS records
- Counter collection (PMC, derived metrics) or PC sampling
- HSA-side queue intercept or signal injection (that path is what
  `~/tmp/rocm-trace-lite` already exercises; we deliberately avoid it)
- Multi-tool coexistence (assumes nothing else has registered against the
  single CLR tracer-callback slot for the run)
- Python bindings, install rules, packaging

## Architecture

A single C++17 shared library `libhiptracelite.so` consumed via
`LD_PRELOAD`. On load, an `__attribute__((constructor))` opens an output
file, spawns a writer thread, and registers a callback against
`libamdhip64.so`. On unload, an `__attribute__((destructor))` flushes,
joins the writer, and closes the file.

The hot path is a single function — `htl_callback(activity_domain_t,
uint32_t op, void* data)` — which:

1. Handles the enablement probe (`op == 0x1`, `data == nullptr`) by returning
   "enabled" for the requested op.
2. For real records (`data` points to an `activity_record_t`), copies the
   fields we care about plus the `kernel_name` string into a fixed-size
   ring slot and returns immediately.
3. Performs no I/O, no allocation, no locking on the hot path.

The writer thread loops on the SPSC ring, batches drained slots, and
`writev()`s them to the file.

### Default scope

- **HIP_OPS records** (kernel completions, async copies as exposed by CLR):
  always captured.
- **HIP_API records** (sync API entry/exit): captured only when
  `HTL_TRACE_API=1` is set in the environment. When the flag is off, the
  HIP_API activity table is never populated, so CLR's
  `api_callbacks_spawner_t` short-circuits and the API hot path is not
  touched.

### Single-slot caveat

`hipRegisterTracerCallback` writes into one atomic function pointer in CLR.
If any other tracer (rocprofiler-sdk, rocprofv1/v2, legacy roctracer) is
also loaded, the last writer wins. The README must call this out as a
"run isolated" requirement for valid benchmark numbers.

## Components & file layout

```
experimental/hip-trace-lite/
├── CMakeLists.txt
├── README.md                       (build + run + env vars + caveats)
├── design.md                       (this spec)
├── src/
│   ├── htl_loader.cpp              ctor/dtor; dlsym hipRegisterTracerCallback; env parse
│   ├── htl_callback.cpp            single TracerCallback function; enqueues to ring
│   ├── htl_ring.hpp                header-only bounded SPSC ring (power-of-2 size)
│   ├── htl_writer.cpp              writer thread; writev batching
│   ├── htl_record.hpp              packed on-disk record + file header (magic/version)
│   └── htl_prof_protocol.hpp       inline minimal copy of activity_record_t fields
├── tools/
│   └── htl_dump.cpp                offline binary → CSV/text decoder
└── test/
    └── smoke.cpp                   minimal HIP kernel launch; verify file non-empty
```

### Disk format

A 64-byte file header followed by a stream of fixed-size records, then a
trailing string section.

```
File header (64 bytes):
  char     magic[4]      = "HTL0"
  uint32_t version       = 1
  uint64_t start_ns      // host monotonic ns at file open
  uint64_t pid
  uint32_t record_size   // sizeof(htl_record_t)
  uint32_t header_size   // 64
  uint8_t  reserved[32]

htl_record_t (packed, fixed width):
  uint8_t  domain          // ACTIVITY_DOMAIN_HIP_OPS or HIP_API
  uint8_t  op              // OP_ID_DISPATCH / COPY / BARRIER / api id
  uint16_t flags
  uint32_t correlation_id
  uint64_t begin_ns
  uint64_t end_ns
  uint32_t process_id
  uint32_t thread_id
  uint32_t device_id
  uint32_t queue_id
  uint64_t bytes           // for copies; 0 otherwise
  uint64_t kernel_name_off // byte offset into trailing string section; 0 if none

String section: appended after the last record on shutdown. Each entry is
length-prefixed (uint32 length, then bytes, no NUL).
```

Records are fixed-width so the writer can do a contiguous `writev()` without
per-record framing. Strings are deferred to shutdown to keep the hot path
copy-only (the kernel name pointer from CLR is owned by CLR; we copy the
bytes into a TLS-bumped string arena keyed by ring-slot index, then flush
the arena into the trailing section at close).

### No roctracer build dependency

`htl_prof_protocol.hpp` declares only the layout we consume from
`activity_record_t` (the prefix matching `prof_protocol.h`). No headers
from the `projects/roctracer` tree are pulled in.

### Build

- CMake subproject. Finds `hip` only for header include paths (we do not
  link against HIP; `hipRegisterTracerCallback` is resolved at load time
  via `dlsym(RTLD_DEFAULT, ...)` against the host process's already-loaded
  `libamdhip64.so`).
- Out-of-tree build:
  `cmake -B build experimental/hip-trace-lite && cmake --build build`
- Artifacts: `libhiptracelite.so`, `htl_dump`, `smoke` (test binary).
- C++17, no external deps beyond libc/libpthread/libdl.

### Run path (must be documented in README)

The README must include a **Running it** section with concrete copy-paste
commands covering at minimum:

1. Building (`cmake -B build … && cmake --build build`)
2. Running any HIP application with the library preloaded:
   `LD_PRELOAD=$PWD/build/libhiptracelite.so ./your_hip_app`
3. Picking the output file: `HTL_OUTPUT_FILE=/tmp/run.htl LD_PRELOAD=… ./app`
4. Toggling HIP_API capture: `HTL_TRACE_API=1 LD_PRELOAD=… ./app`
5. Decoding: `./build/htl_dump /tmp/run.htl > run.csv`
6. The "run isolated" caveat (do not preload alongside rocprofv1/v2 or any
   rocprofiler-sdk tool — the single CLR callback slot will be overwritten).
7. Smoke test invocation: `./build/smoke && ls -l hiptrace.bin`

### Environment variables

| Var | Default | Effect |
|---|---|---|
| `HTL_OUTPUT_FILE` | `./hiptrace.bin` | Output file path (truncated on open) |
| `HTL_TRACE_API` | `0` | When `1`, also capture HIP_API records |
| `HTL_RING_SIZE` | `65536` | SPSC ring slot count (power of two) |
| `HTL_DROP_ON_FULL` | `1` | When `1`, drop records if ring is full and bump a counter; when `0`, busy-wait (skews benchmarks) |

### Error handling

- `dlsym(hipRegisterTracerCallback)` failure: log to stderr and become a
  no-op (do not crash the host app).
- File open failure: same — log and no-op.
- Ring full: drop the record and increment a counter that's emitted in the
  file footer at shutdown so post-run analysis can see the loss rate.

### Testing

- `smoke.cpp` launches one trivial HIP kernel, exits, then a small shell
  step (or a follow-up CTest assertion) verifies `hiptrace.bin` exists,
  has the correct magic, contains at least one HIP_OPS record with
  non-zero `begin_ns`/`end_ns`, and that `htl_dump` parses it without
  error.
- No GPU-class-specific assertions — anything HIP-runnable should work.

## Open items (intentional, not blocking the spec)

- Pinning the writer thread / setting its scheduling policy is left to a
  later iteration if benchmarks show it matters.
- `writev` batch size is a small constant; tuning is a benchmark concern.
- The benchmark harness comparing this to `rocm-trace-lite` (HSA-intercept)
  and rocprofiler-sdk lives in a separate spec.
