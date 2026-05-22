# HIP Record & Replay (HRR) — In-Tree Dispatch Table Design

## Summary of Changes from Original Out-of-Tree HRR

The original HRR system (`hip_hrr.cpp` + out-of-tree proxy DLL / LD_PRELOAD interposer) has
been replaced by an in-tree capture layer built directly into `amdhip64`. Key changes:

| Area                           | Before                                    | After                                              |
|--------------------------------|-------------------------------------------|----------------------------------------------------|
| **API coverage**               | ~20 hand-maintained APIs                  | Every HIP API (~529 total, fully generated)        |
| **Platform support**           | Separate Windows DLL + Linux LD_PRELOAD   | Single in-tree code path                           |
| **Capture enable**             | Hard-coded at build time                  | `HIP_HRR_CAPTURE_OUTPUT=<dir>` env var             |
| **Kernel arg introspection**   | ~360 lines of ELF + msgpack parsing       | `kernel->signature()` — zero added lines           |
| **Kernel arg pointer detect**  | `value_kind` string matching              | `desc.type_ == T_POINTER`                          |
| **Handle IDs**                 | Sequential `uint32_t` per type            | Raw pointer cast to `uint64_t`                     |
| **Stream capture**             | Not captured                              | Full stream create/destroy + event record           |
| **Graph capture**              | Not captured                              | Full `hipStreamBeginCapture` → `hipGraphLaunch`    |
| **Fat-binary launch path**     | Not captured                              | Compiler dispatch table slot + TLS                 |
| **Capture shim generation**    | Manual updates per new API                | `gen_hrr_api_args.py` auto-generates all shims     |
| **D2H data validation**        | None                                      | Blob compare on every D2H memcpy                   |
| **Pinned host mem (register)** | Not captured                              | `hipHostRegister` blob snapshot + playback restore |
| **Multi-threaded replay**      | No                                        | `--multi-thread`: one CPU thread per captured thread (default: single-threaded) |
| **Kernel filter + warm-up**    | No                                        | `--kernel-filter STR` with silent warm-up pass     |
| **Timing**                     | No                                        | `--timing`: wall time + GPU kernel time + GPU graph time |
| **GPU error surfacing**        | No                                        | `--sync-after-launch`: device sync after every launch |
| **MT-safe replay context**     | No                                        | `shared_mutex` reads, atomic stats counters        |
| **Archive info (no GPU)**      | Separate `hrr-info` tool                  | `hrr-playback --info [--events]`                   |
| **Playback tools**             | `hrr-replay` + `hrr-info`                 | Single `hrr-playback` (`--info` flag)         |
| **Playback built**             | Separate CMake tree                       | Part of main `amdhip` CMake build                  |
| **Code-object layout**         | `hipamd/src'                              | All under `hipamd/src/hrr/`                        |
| **Copyright**                  | Ad-hoc                                    | SPDX `MIT` on every file + generated output        |

### Quick Start

```bash
# Capture
HIP_HRR_CAPTURE_OUTPUT=./my_capture.hrr ./my_hip_app

# Replay + D2H validation
hrr-playback ./my_capture.hrr

# Archive info (no GPU required)
hrr-playback ./my_capture.hrr --info [--events]
```

---

## Implementation Summary

### What It Does
HRR captures every HIP API call made by an application into a binary archive (`.hrr` directory), then replays that archive against a live GPU to reproduce the original workload exactly — including multi-threaded submission, graph execution, and all GPU memory transfers. Primary uses: bug reproduction, performance regression testing, and kernel benchmarking without the original application.

### Capture Layer (`hipamd/src/hrr/`)
At HIP init time the capture layer snapshots the real `HipDispatchTable` and installs a parallel capture table with ~529 shim function pointers. Every HIP API call flows through one of these shims, which records a fixed-size `hrr_event_header` (32 bytes: type, sequence ID, timestamp, thread ID) followed by a variable-length payload of serialised arguments into a streaming `events.bin` file.

Twenty-seven APIs require hand-written shims (`MANUAL_CAPTURE_APIS`): kernel launches (4 variants), host-to-device memcpy (4 variants), module load (3 variants), `__hipRegisterFatBinary`, host memory registration (`hipHostRegister`, `hipHostUnregister`), all `hipMemcpy3D` variants (4), array creation, VMM, stream/memory attribute APIs, and others where struct pointer serialisation requires custom handling. All others are code-generated by `gen_hrr_api_args.py` from `hip_api_trace.hpp`. Fat binaries registered before shim install (C++ static-init) are recovered retroactively via `ForEachFatBinaryBlob` at init.

H2D memcpy captures the host source buffer as a content-addressed blob (FNV-1a-128 hash). D2H memcpy captures the expected output blob after a forced stream sync. Kernel argument pointers are detected via `kernel->signature()` (in-tree access to `amd::KernelSignature`) — no ELF parsing needed.

### Only Successful Calls Are Captured

All capture shims guard the event write with `if (r == hipSuccess)`.
This means API calls that fail in the captured application are not recorded.
Applications that handle and recover from API failures (e.g. `hipMalloc` OOM
followed by a retry with a smaller allocation) will not replay faithfully —
the retry branch produces a different allocation layout than the successful original.
This is a known limitation with no current workaround.

### Archive Format (v3)
```
capture.hrr/
  events.bin         hrr_file_header(8) + [EventHeader(32) + payload]*
  blobs/<2hex>/      content-addressed host buffers keyed by FNV-1a-128 hash
  code_objects/      .hsaco ELFs (unused in current fat-binary path)
```
All handle values (stream, event, module, graph, device pointer) are stored as raw `uint64_t` pointer casts. Sequence IDs are a global atomic counter providing causal ordering across threads.

### Playback Tool (`hrr-playback`)
The reader (`hrr_reader`) loads all events sorted by sequence ID. One CPU thread is spawned per captured OS thread. Each thread replays its slice of events in order; a global atomic `next_seq` counter enforces the original interleave between threads — each thread spin-waits until its event's sequence ID is current before dispatching, then advances the counter.

Handle translation maps (one per HIP handle type) remap captured raw pointers to live playback handles. Device pointer translation supports sub-allocation offsets via range search through `alloc_map`. D2H memcpy validation compares live GPU output against captured expected blobs, reporting pass/fail counts.

Fifty-six APIs have hand-written playback handlers (`MANUAL_PLAYBACK_APIS`): everything in the capture manual set plus all malloc/free variants, stream/event create/destroy, `hipModuleGetFunction`, host memory registration, the full HIP graph chain (`hipStreamBeginCapture` → `hipStreamEndCapture` → `hipGraphInstantiate` → `hipGraphLaunch`), VMM APIs (`hipMemAddressReserve/Free`, `hipMemCreate/Release`, `hipMemMap/Unmap`), and others requiring handle translation or non-trivial replay logic. All others are code-generated.

### Ordering — `needs_ordering`
Handle-creating and handle-destroying events (malloc, free, stream/event/module create/destroy, graph instantiate, etc.) must execute in global capture order so that handle maps are populated before any dependent thread reads them. Kernel launches and syncs are excluded — GPU stream ordering handles those. The `SeqAdvance` RAII guard in `dispatch_event` releases `next_seq` either immediately (non-ordering events) or on scope exit (ordering events), using a lookahead at the next event to decide.

### Timing (`--timing`)
`hipModuleLaunchKernel` and `hipGraphLaunch` wrap their GPU submission with `hipEventRecord` before/after, synchronise the stop event, and accumulate elapsed time into `total_kernel_ms` / `total_graph_ms` (both guarded by `map_mutex`). Events are created once per replay thread (`thread_local`) and reused for every launch. Graph-capture streams are excluded from timing (recording events into a captured stream corrupts the graph). The summary reports kernel time, graph time, and combined total.

### Code Generation
`gen_hrr_api_args.py` parses `hip_api_trace.hpp` and emits three files:

| File | Contents |
|------|----------|
| `hrr_api_args.h` | 529 `hrr_args_*` structs + `hrr_api_id_t` enum |
| `hip_capture_generated.cpp` | ~517 capture shims + `hip_capture_build_table()` |
| `hip_playback_generated.cpp` | ~504 playback shims + `hrr_playback_dispatch[]` |

Run from `hipamd/src/hrr/`: `python gen_hrr_api_args.py`

### Build
```bash
# Capture DLL
cmake --build C:/MIGraphX/amdhip --config Release -j 6 --target amdhip64

# Playback tool (same tree)
cmake --build C:/MIGraphX/amdhip --config Release --target hrr-playback
# Output: hipamd/src/hrr/playback/Release/hrr-playback.exe
```

### Usage
```bash
HIP_HRR_CAPTURE_OUTPUT=./out.hrr ./my_hip_app   # capture
hrr-playback ./out.hrr                           # replay + D2H validation
hrr-playback ./out.hrr --timing                  # with kernel + graph GPU timing
hrr-playback ./out.hrr --info --events           # inspect archive (no GPU)
hrr-playback ./out.hrr --kernel-filter saxpy     # bench one kernel (warm-up pass first)
```

---

## Background

The out-of-tree HRR proxy DLL (`hrr_proxy_win.c`) and Linux LD_PRELOAD interposer
intercept a hand-maintained subset of HIP calls. This approach has several gaps:

- `hipStreamCreate` / `hipStreamDestroy` — not captured; stream topology lost
- `hipEventRecord` / `hipEventSynchronize` — not captured; cross-stream ordering lost
- `hipGraph*` — not captured
- `__hipRegisterFatBinary` / fat-binary launch path — not captured
- Separate Windows and Linux code paths with duplicated logic
- Must be manually updated whenever a new HIP API needs recording

`hip_api_trace.hpp` already provides a **versioned dispatch table** (`HipDispatchTable`)
with a function pointer slot for every HIP API. `hip_table_interface.cpp` routes every
`extern "C" hipFoo(...)` call through `GetHipDispatchTable()->hipFoo_fn(...)`.

This is the correct hook point. An in-tree capture layer installs capture shims into
the dispatch table at init time, replacing the out-of-tree proxy entirely.

## Directory Structure

```
hipamd/src/hrr/
  DESIGN.md                       — this file
  gen_hrr_api_args.py             — generator: produces hrr_api_args.h,
                                    hip_capture_generated.cpp,
                                    playback/hip_playback_generated.cpp
  hrr_api_args.h                  — AUTO-GENERATED: format constants, hrr_event_header,
                                    one hrr_args_* struct per HIP API, hrr_api_id_t enum
  hip_capture.h                   — public API (init/shutdown) + Hash128 type
  hip_capture.cpp                 — 27 manual shims (kernel launch ×4, memcpy H2D+D2H ×8,
                                    module load ×3, fat binary, hipHostRegister/Unregister,
                                    hipMemcpy3D variants ×4, hipArrayCreate, hipArray3DCreate,
                                    hipStreamSetAttribute, hipMemGetAllocationGranularity,
                                    hipMemPoolSetAccess, hipMemSetAccess, hipMemcpyWithStream)
                                    + dispatch table install/uninstall
  hip_capture_writer.h/.cpp       — streaming event serialization to events.bin / blobs/
  hip_capture_handles.h/.cpp      — stream/event/module ID maps (mutex-guarded)
  hip_capture_generated.cpp       — AUTO-GENERATED: ~502 capture shims + build_table()
  CMakeLists.txt                  — adds hrr/ sources to amdhip64 target;
                                    add_subdirectory(playback) to build tools

  playback/
    hrr_reader.h/.cpp             — archive loader, v3 format
    hip_playback.h                — PlaybackContext (11 handle maps + host_reg_bufs), PlaybackFn typedef
    hip_playback.cpp              — 56 manual playback shims (all graph APIs, malloc/free variants,
                                    stream/event create/destroy, hipModuleGetFunction,
                                    hipHostRegister/Unregister, hipMemcpy3D variants,
                                    hipArrayCreate, VMM APIs, and more)
    hip_playback_generated.cpp    — AUTO-GENERATED: ~201 playback shims + dispatch table
    hrr_playback.cpp              — hrr-playback tool (replay + D2H validation + --info)
    CMakeLists.txt                — child build: hrr_reader + hrr_playback_lib + hrr-playback exe

test/ (separate tree, not part of this PR)
  hip_raw_trace.cpp               — multi-threaded integration test workload
                                    (threadFunc ×4 + graphFunc + pinnedFunc + hostRegisterFunc)
                                    capture: HIP_HRR_CAPTURE_OUTPUT=out.hrr hip_raw_trace.exe
                                    replay:  hrr-playback out.hrr
```

## Core Mechanism

At init time the capture layer snapshots `g_real_table` from `GetHipDispatchTable()`,
builds `g_cap_table` as a copy with the 12 manual shims overridden, then installs it
atomically. All shims call through `g_real_table` — never back into the dispatch table —
preventing re-entrancy. Uninstall restores the snapshot.

### `<<<>>>` Launch Path — Compiler Dispatch Table

`kernel<<<grid,block,shared,stream>>>` compiles to `__hipPushCallConfiguration` (saves
grid/block/shared/stream into thread-local storage) then `hipLaunchByPtr`. Both slots
are hooked in `HipCompilerDispatchTable`; `hipLaunchByPtr` reads the TLS values and
calls `serialize_kernel_launch()`.

## Generator — `gen_hrr_api_args.py`

A single Python script produces three files from `hip_api_trace.hpp`:

| Output | Description |
|--------|-------------|
| `hrr/hrr_api_args.h` | Shared header: format constants, `hrr_event_header`, one `hrr_args_*` struct per API, `hrr_api_id_t` enum, `hrr_api_names[]` |
| `hrr/hip_capture_generated.cpp` | ~502 capture shims for APIs not in `MANUAL_CAPTURE_APIS` |
| `hrr/playback/hip_playback_generated.cpp` | ~201 playback shims + dispatch table for APIs not in `MANUAL_PLAYBACK_APIS` |

Script location: `hipamd/src/hrr/gen_hrr_api_args.py`

Run from `hipamd/src/hrr/`: `C:/Users/gandryey/AppData/Local/Programs/Python/Python312/python.exe gen_hrr_api_args.py`

The generator classifies each API:
- **`MANUAL_CAPTURE_APIS`** (27): kernel launches ×4, memcpy H2D+D2H ×8, module load ×3, `__hipRegisterFatBinary`, `hipHostRegister/Unregister`, `hipMemcpy3D` variants ×4, array creation ×2, VMM, stream/memory attribute APIs, `hipMemcpyWithStream`
- **`MANUAL_PLAYBACK_APIS`** (56): above plus malloc ×4, free ×2, stream create/destroy ×4, event create/destroy ×3, `hipModuleGetFunction`, the full graph chain ×4, VMM APIs ×6, and other APIs requiring non-trivial handle translation or replay logic
- **`NOOP_PLAYBACK_APIS`** (~271): APIs where replay is not meaningful (stale handles, context-destroying calls, missing in ROCm SDK 6.4) — emit `return hipSuccess;` without calling the real function. These are **not replay-tested**; tests that exercise only these APIs verify capture does not crash, not that replay works.

Generated capture shims for manual APIs are pass-throughs (no `write_event()`).

## Archive Format (v3)

Single-authority definition in `hrr_api_args.h` (auto-generated):

```
HRR_MAGIC   = 0x52524845  ("HRRE")
HRR_VERSION = 3
```

```
<output_dir>/
  manifest.json      { version, capture_mode, event_count, blob_count }
  events.bin         8-byte hrr_file_header, then repeated records
  blobs/<2hex>/      FNV-1a-128 content-addressed raw buffers (.blob ext)
  code_objects/      .hsaco ELFs keyed by hash
```

### `events.bin` Layout

```
[0..7]    hrr_file_header  { magic:u32, version:u16, reserved:u16 }
[8..]     records, back-to-back, no padding:
            hrr_event_header (32 bytes, pack(1)):
              event_type     u16   hrr_api_id_t (0..528)
              sequence_id    u64   monotonically increasing (atomic)
              timestamp_ns   u64   MONOTONIC wall clock
              thread_id      u64   OS thread ID (cached per-thread)
              payload_length u16   total size INCLUDING this 32-byte header
              reserved       u8[4] zero
            payload bytes    (payload_length - 32 bytes)
              ret            i32   hipError_t return value
              params         per hrr_args_* struct fields
              [extra fields] blob_hash_lo/hi, co_hash_lo/hi, module_id
```

### `hrr_args_*` Struct Layout Rules

All structs are `#pragma pack(1)`. First field is always `hrr_event_header hdr`,
then `int32_t ret`, then API parameters:

| HIP type | Stored as |
|----------|-----------|
| pointer (`void*`, device ptr) | `uint64_t` |
| handle (`hipStream_t`, `hipEvent_t`, `hipModule_t`, ...) | `uint64_t` (raw ptr cast) |
| `dim3` | three `uint32_t` (`_x`, `_y`, `_z`) |
| `size_t` | `uint64_t` |
| enum | `int32_t` |
| `hipError_t` return | `int32_t ret` |
| scalar int/uint/float/double | native type |

**Playback payload rule:** `payload` in all playback functions points to the START of the
full `hrr_args_*` struct — i.e. `payload[0..31]` is the `hrr_event_header`, `payload[32]`
is `ret`, `payload[36]` is the first field. Cast directly:
```cpp
const auto* a = reinterpret_cast<const hrr_args_hipFoo*>(payload);
// then: a->ret, a->fieldName, etc.
```
Exception: `replay_kernel_launch` has a variable-length format and must skip the header
manually: `const uint8_t* p = pl + sizeof(hrr_event_header);`

### Extra Fields (appended after normal params)

| API | Extra fields |
|-----|-------------|
| `__hipRegisterFatBinary` | `blob_hash_lo u64`, `blob_hash_hi u64`, `blob_size u64` |
| `hipModuleLoadData`, `hipModuleLoadDataEx`, `hipModuleLoad` | `co_hash_lo u64`, `co_hash_hi u64`, `module_id u32` |
| `hipMemcpy`, `hipMemcpyAsync`, `hipMemcpyHtoD`, `hipMemcpyHtoDAsync` | `blob_hash_lo u64`, `blob_hash_hi u64` |
| `hipHostRegister` | `blob_hash_lo u64`, `blob_hash_hi u64` |

## Handle Storage — Raw Pointer Casts

All opaque HIP handles are stored as raw `uint64_t` pointer casts. No sequential ID
registry. At playback, `PlaybackContext::translate_*()` methods map recorded values to
live handles. Device pointers use the same scheme, with sub-allocation offsets resolved
via `alloc_map` range search. `nullptr` / default stream store as `0` naturally.

## Blob Capture — H2D, D2H, and Host Registration

### Memcpy blobs

The `blob_hash_lo/hi` extra fields in memcpy events serve double duty:

| Direction | What is captured | When |
|-----------|-----------------|------|
| `hipMemcpyHostToDevice` | host `src` buffer (input data) | immediately (src valid before call) |
| `hipMemcpyDeviceToHost` (sync) | host `dst` buffer (expected output) | after call returns |
| `hipMemcpyDeviceToHost` (async) | host `dst` buffer (expected output) | after `hipStreamSynchronize(stream)` |
| `hipMemcpyDeviceToDevice` | not captured | — |

For D2H async, the capture shim forces a stream sync to ensure the host buffer is valid
before snapshotting. This adds latency to the captured workload but produces correct
expected-output blobs.

### hipHostRegister / hipHostUnregister

`hipHostRegister` registers existing host (CPU) memory as pinned for faster GPU transfers
or direct GPU access via the mapped flag. At capture time:

- `capture_hipHostRegister` calls the real function, then snapshots the host buffer as a
  content-addressed blob. The blob hash is stored in `hrr_args_hipHostRegister.blob_hash_lo/hi`.
- A process-global `g_pinned_reg_map` (`mutex` + `unordered_map<void*, size_t>`) records
  `(hostPtr → sizeBytes)`. This is needed because `hipHostUnregister` receives only the
  pointer with no size.
- `capture_hipHostUnregister` calls the real function and erases from `g_pinned_reg_map`.

At playback:

- `playback_hipHostRegister` allocates a fresh 64-byte-aligned host buffer (`_aligned_malloc`
  on Windows, `posix_memalign` on Linux), restores the blob snapshot into it, calls
  `hipHostRegister` on the new buffer, and records `(recorded_ptr → live_ptr)` in `alloc_map`
  (for kernel-arg pointer translation) and `host_reg_bufs` (for cleanup at Unregister).
- `playback_hipHostUnregister` translates the pointer, calls `hipHostUnregister`, frees the
  backing buffer, and removes both map entries.

### Sysmem Update Tracking (planned — not yet implemented)

**Problem:** `hipHostRegister` captures the buffer contents at registration time only.
After that, the CPU may modify the buffer without going through any HIP API:

```
hipHostRegister(ptr, sz)   ← blob captured (initial state)
// CPU writes to ptr[]     ← invisible to capture layer
hipMemcpyAsync(d, ptr, sz, H2D, stream)   ← H2D blob captures current contents ✓
// CPU writes to ptr[] again
hipModuleLaunchKernel(...)  ← kernel reads ptr directly via mapped flag ✗ stale
```

H2D memcpy is already handled — the src buffer is re-snapshotted at each call regardless
of registration. The gap is direct GPU reads from registered host memory (mapped flag)
after a CPU write that was not routed through a memcpy.

**Planned design:**

1. New synthetic event `HRR_SYSMEM_UPDATE` (not a real HIP API):
   fields: `hostPtr u64`, `sizeBytes u64`, `blob_hash_lo u64`, `blob_hash_hi u64`.

2. Per-region `last_hash` stored alongside `g_pinned_reg_map`. Before each H2D memcpy src
   check and before each kernel launch for pointer args in registered ranges: hash the
   current contents, compare with `last_hash`. Emit `HRR_SYSMEM_UPDATE` + write blob only
   if hash changed. Content-addressed storage deduplicates unchanged regions automatically.

3. Optional sync-gated dirty flag: set `dirty=true` for all registered regions after any
   `hipStreamSynchronize` / `hipDeviceSynchronize` / `hipEventSynchronize`. Only hash
   (step 2) if `dirty==true`. Avoids hashing in pure-GPU loops where the CPU never
   touches the buffer between launches. Falls back to always-hash if sync events are
   not captured.

4. At replay: `HRR_SYSMEM_UPDATE` handler `memcpy`s the blob into the live registered
   buffer, ordered by sequence ID like all other events.

## Kernel Argument Capture

In-tree access to `kernel->signature()` (`amd::KernelSignature`) gives arg types,
offsets, and sizes directly — no ELF/msgpack parsing needed (the out-of-tree proxy
required ~360 lines for this). Hidden runtime-injected args are skipped via
`desc.info_.hidden_`. Each visible arg is tagged as scalar or gpu-pointer
(`desc.type_ == T_POINTER`) and its bytes copied from the kernarg buffer.

`hipModuleLaunchKernel` / `hipExtModuleLaunchKernel` support two arg-passing modes:
`extra[]` (pre-packed buffer pointed to by `HIP_LAUNCH_PARAM_BUFFER_POINTER`) and
`kernelParams[]` (void** array reconstructed into a packed buffer using the signature).

`co_hash` is always written as 0 in kernel launch events — no cheap mapping from
kernel to code object at dispatch time. Playback resolves kernels by name, scanning
all loaded modules.

## Fat Binary Registration

`__hipRegisterFatBinary` fires at C++ static-init time before shims are installed.
At init, all already-registered fat binaries are recovered retroactively via
`ForEachFatBinaryBlob`. At playback, each captured blob is loaded with
`hipModuleLoadData`; the resulting `hipModule_t` is stored in `module_map` keyed by
hash and scanned for kernel name lookups.

## Multi-Threaded Capture

Events from concurrent threads are written in file-arrival order. The `sequence_id`
field (atomic counter, incremented per event) records logical call order per thread.
The reader (`hrr_reader.cpp`) sorts all events by `sequence_id` after loading to
restore causal ordering within each thread.

Cross-thread causal ordering (e.g. a stream created by thread A before thread B calls
BeginCapture on it) may still be violated if the writes race. The playback handles
missing stream handles by creating a temporary stream.

## HIP Graph Capture / Replay

The full `hipStreamBeginCapture` → `hipStreamEndCapture` → `hipGraphInstantiate` →
`hipGraphLaunch` chain is supported. All four are `MANUAL_PLAYBACK_APIS`. `BeginCapture`
falls back to `hipStreamCaptureModeGlobal` if `ThreadLocal` fails. `EndCapture` records
the returned `hipGraph_t` into `graph_map`. `GraphInstantiate` uses
`hipGraphInstantiateWithFlags(..., 0)` and records the exec into `graph_exec_map`.
`GraphLaunch` translates both handles and, when `--timing` is active, wraps the call
with `hipEventRecord` to accumulate elapsed time into `total_graph_ms`.

## Init / Shutdown

`hip_capture_init()` is called from `hip_context.cpp` at HIP init. It always snapshots
the dispatch tables; if `HIP_HRR_CAPTURE_OUTPUT` is set it opens the writer, recovers
pre-init fat binaries, installs the capture shims, and registers `hip_capture_shutdown`
via `atexit`. Shutdown uninstalls shims and flushes `events.bin` + `manifest.json`.

## Enable Flag

```
HIP_HRR_CAPTURE_OUTPUT=<directory>
```

Defined as a `cstring` release flag in `rocclr/utils/flags.hpp`. Capture is active
only when the variable is set (non-default) and non-empty.

## Playback Tools

| Tool | Source | Purpose |
|------|--------|---------|
| `hrr-playback` | `hrr_playback.cpp` | Full replay + D2H validation + archive info |
| `hrr-validate` | `hrr_validate.cpp` (test/) | Kernel-count + NaN/Inf correctness check |

### `hrr-playback` Options

```
hrr-playback <capture.hrr> [options]
  --info                Print archive summary and exit (no GPU required)
  --events              With --info: also print the full event log
  --verbose             Print each event as it is processed
  --skip-device-sync    Skip hipDeviceSynchronize / hipStreamSynchronize events
  --multi-thread        One replay thread per captured thread (default: single-threaded)
  --timing              Report wall time, GPU kernel time, GPU graph time, and combined total
  --kernel-filter STR   Only launch kernels whose name contains STR
                        (silent full warm-up pass runs first to populate GPU state)
  --sync-after-launch   hipDeviceSynchronize after every kernel (surfaces GPU errors)
  --sync-after-event    hipDeviceSynchronize after every event (HW hang debug; very slow)
```

For each D2H memcpy event that has a captured expected-data blob:
1. Translates the recorded device src pointer to the live address
2. Copies the result back to a host buffer via `hipMemcpy`
3. Compares byte-for-byte with the expected blob from the archive
4. Increments `ctx.d2h_pass` or `ctx.d2h_fail`

Exits 0 if all checks pass (or no D2H blobs in archive), 1 on any failure.

### Multi-Threaded Replay

One CPU thread is spawned per captured thread (keyed by OS thread ID stored in every
event header). Each thread replays its own event slice in sequence-ID order. GPU-side
parallelism is preserved through stream handles exactly as during capture. CPU-side
synchronisation between threads is not replicated — see Limitations section.

## Build System

The playback tools are now built as part of the main `amdhip` CMake tree via
`add_subdirectory(playback)` in `hrr/CMakeLists.txt`. No separate CMake tree is needed.

### Capture DLL + Playback Tools (combined)
```bash
cd C:/MIGraphX/amdhip
cmake --build . --config Release -j 6 --target amdhip64
cmake --build . --config Release --target hrr-playback
# DLL: Release/amdhip64_7.dll
# Tool: Release/hrr-playback.exe  (or playback/Release/hrr-playback.exe)
```

### Deploy
```bash
copy Release\amdhip64_7.dll C:\MIGraphX\test\build2\
```

### Validation Tool (separate test tree)
```bash
cmake --build C:/MIGraphX/test/build2 --config Release --target hrr-validate
# Produces: hrr-validate.exe
```

## Comparison with Out-of-Tree Proxy

| Property                        | Out-of-tree proxy         | In-tree dispatch hook           |
|---------------------------------|---------------------------|---------------------------------|
| API coverage                    | Manual subset (~20 APIs)  | Every HIP API (~529 total)      |
| Windows support                 | Separate proxy DLL        | Single code path                |
| Linux support                   | Separate LD_PRELOAD       | Single code path                |
| Stream handle capture           | No                        | Yes                             |
| Event capture / ordering        | No                        | Yes                             |
| `hipGraph*`                     | No                        | Yes (full capture + replay)     |
| D2H data validation             | No                        | Yes (`hrr-playback` default)    |
| Fat-binary launch path          | No                        | Yes (compiler table slot)       |
| ROCm version coupling           | Breaks on ABI shift       | Owns the table, always in sync  |
| Kernel arg type info            | ELF + msgpack parse       | `kernel->signature()` directly  |
| Kernel arg pointer detection    | `value_kind` string match | `desc.type_ == T_POINTER`       |
| Kernel name lookup              | `g.funcs[]` side table    | `kernel->name()`                |
| Supporting code for args        | ~360 lines (hrr_code_object.c) | 0 lines                    |
| Handle IDs                      | Sequential u32 per type   | Raw pointer cast to u64         |

## Known Limitations

### CPU Thread Synchronisation — Fundamentally Uncapturable

The capture layer intercepts HIP API calls only. Any CPU-side synchronisation between
application threads — `std::mutex`, `std::condition_variable`, `std::atomic`, barriers,
`pthread_join`, Win32 `WaitForSingleObject`, etc. — is invisible to the capture layer
and therefore not recorded.

At replay time, one CPU thread is spawned per captured thread. Each thread replays its
own event slice in call-sequence order, but the threads run independently. If the
original application used a mutex to gate thread B until thread A completed a GPU
memcpy, that ordering is **not reproduced**. The replay may submit GPU work in a
different relative order than the original.

In practice this is safe for most workloads because GPU-visible ordering is enforced
through HIP stream and event APIs (which _are_ captured), but it can cause incorrect
results if application code uses CPU-side synchronisation to sequence GPU memory
writes that are later read by a different thread.

There is no practical fix: capturing general CPU synchronisation would require
instrumentation of the application binary, OS synchronisation primitives, or a
language runtime — all of which are out of scope for a HIP API trace.

### `__hipRegisterVar` / `__hipRegisterManagedVar` — Not Replayed

These compiler-generated functions populate the runtime's internal host-symbol →
device-symbol table used by `hipMemcpyToSymbol`, `hipMemcpyFromSymbol`,
`hipGetSymbolAddress`, and `hipGetSymbolSize`.

The capture shims record events for `__hipRegisterVar` and `__hipRegisterManagedVar`
but write empty payloads — the variable name, host pointer, size, and flags are not
captured. The playback shims are no-ops.

**Impact:** workloads that use `hipMemcpyToSymbol` / `hipMemcpyFromSymbol` will fail
at replay because the symbol table is never populated. Kernel launches and all
pointer-based memcpy APIs are unaffected.

**Workaround:** none currently. A fix would require recording the variable name and
host pointer, then at playback calling `hipModuleGetGlobal` on the loaded module to
resolve the device address and rebuilding the symbol table entry.

### `hipLaunchByPtr` — Kernel Args Not Captured

The `<<<>>>` launch path (`__hipPushCallConfiguration` + `hipLaunchByPtr`) does not
support kernel argument capture. Grid/block/shared/stream are saved in thread-local
storage and recorded correctly. However `hipLaunchByPtr` receives only a function
pointer — the kernarg buffer is assembled by the compiler-generated wrapper _after_
`hipLaunchByPtr` returns, and is never accessible to the capture shim.

**Impact:** kernels launched via `<<<>>>` syntax are recorded with correct
grid/block/stream but with zero arguments. At replay the kernel is launched with a
zeroed kernarg buffer, which gives wrong results for any kernel that reads its
arguments.

**Workaround:** capture workloads that use `hipModuleLaunchKernel` /
`hipExtModuleLaunchKernel` directly, or modify the workload to use the explicit API.

### `co_hash` Always Zero in Kernel Launch Events

At dispatch time there is no cheap way to determine which code object a kernel
function belongs to. `co_hash_lo` and `co_hash_hi` are always written as 0 in kernel
launch payload. The playback resolves kernels by name, scanning all loaded modules.

**Impact:** if two loaded code objects define kernels with the same mangled name, the
playback may call the wrong kernel.

### Fat Binary Registration Race

`__hipRegisterFatBinary` fires at C++ static-initialisation time, before
`hip_capture_init()` installs shims. All fat binaries registered before the shim is
active are missed by the normal capture path and recovered retroactively via
`PlatformState::Instance().StatCO().ForEachFatBinaryBlob(...)` at init time.

If an application loads additional shared libraries _after_ HIP init that register
their own fat binaries, those will be captured normally through the shim. The edge
case is a fat binary registered from a thread that runs concurrently with
`hip_capture_init()` — the retroactive sweep and the shim install are not atomic, so
there is a narrow window where such a fat binary could be missed entirely.

### `hipMemcpyDeviceToHost` Async — Forced Stream Sync

To snapshot the expected output blob for a D2H async memcpy, the capture shim calls
`hipStreamSynchronize(stream)` immediately after the copy completes. This adds a CPU
stall to the captured workload that does not exist in the original, which can
significantly alter timing characteristics for latency-sensitive workloads.

### Device Pointer Sub-Allocation

The replayer handles sub-allocations (pointers within a `hipMalloc` region) via a
linear range search through `alloc_map`. If the workload allocates a large pool and
passes many sub-pointers as kernel arguments, this scan becomes O(N) per pointer per
kernel dispatch. For archives with thousands of allocations and tight kernel launch
loops this can make replay significantly slower than the original.

### GPU Allocator Address Non-Determinism

Recorded device pointers are the GPU virtual addresses from the original run. At
replay, `hipMalloc` returns different addresses. Translation is performed via
`alloc_map`. If the application constructs a device pointer by arithmetic on a
recorded address that was never passed through a HIP API (e.g. computed from an
`hipGetDeviceProperties` query or a device-side `malloc`), the capture layer cannot
know about it and translation will fail.

### `hipMemcpyDeviceToDevice` — Not Captured

Source data for D2D copies is not snapshotted. At replay, D2D copies execute correctly
as long as the source allocation was populated by prior H2D or kernel events that were
themselves replayed. If the source was populated by a path not captured (e.g. device
peer access from another process), the copy will operate on uninitialised memory.

### `hipMallocManaged` — Unified Memory Not Validated

Managed memory allocations are tracked in `alloc_map` and translated like any other
device pointer. However the capture layer does not snapshot managed memory contents
at any point. D2H validation blobs cover only explicitly synchronised
`hipMemcpy`-family calls; data accessed by the CPU directly through a managed pointer
is never captured.

### Single-Device Replay

All captured device pointers are translated relative to device 0. Multi-GPU workloads
that allocate memory on specific devices (via `hipSetDevice` + `hipMalloc`) or use
peer-to-peer transfers are not supported at replay. `hipSetDevice` events with an
out-of-range device ordinal are silently clamped to device 0.

### No Replay of `hipIpcMemHandle` / `hipExternalMemory`

IPC memory handles, external memory imports (`hipImportExternalMemory`), and
semaphore imports (`hipImportExternalSemaphore`) are recorded as raw events but the
playback shims are no-ops. Workloads that communicate with other processes via IPC
handles cannot be replayed in isolation.

### `hipStreamCaptureModeThreadLocal` Downgraded

If the workload uses `hipStreamBeginCapture` with `hipStreamCaptureModeThreadLocal`,
the playback falls back to `hipStreamCaptureModeGlobal` because thread-local capture
mode interacts with the single-submission-thread replay model. The graphs are
reconstructed correctly but the capture mode semantics differ.

### Archive Size — No Incremental / Ring-Buffer Mode

Events are written to `events.bin` as they are captured (streaming, not buffered in memory).
Blobs are written atomically via temp-file + rename on first occurrence; subsequent
captures of the same content hash are skipped via an in-memory `g_written_blobs` set.
The only unbounded in-memory state is `g_written_blobs` (one string entry per unique blob)
and the per-event `hrr_args_*` struct (freed after `fwrite`). There is no windowed or
ring-buffer capture mode that would limit archive size to a fixed window of events.

### System Memory Allocations and Kernel Launches

The capture layer records `hipHostMalloc` / `hipHostRegister` allocations and replays
them, but GPU kernels may read or write system memory (sysmem) regions via pointers
passed as kernel arguments. The results at replay are likely to be invalid unless the
sysmem contents were explicitly snapshotted.

Sysmem capture is a fundamentally hard problem. Two approaches exist:

- **Brute-force (capture everything, always):** Before every kernel launch, snapshot
  all known registered and pinned host regions. This is correct but imposes very high
  overhead on both capture (copying potentially gigabytes of host memory per launch)
  and playback (restoring them before every kernel).
- **Smart (capture dirty regions):** Track which sysmem regions have been written since
  the last snapshot (e.g. via memory-protection page faults or a dirty-tracking shadow),
  and emit `HRR_SYSMEM_UPDATE` synthetic events only when content changes. This is
  significantly more complex to implement and may not be fully feasible on all platforms.

Neither approach is currently implemented.

### Partial-Replay APIs — H2D and D2H Work, Kernel Launches Do Not

Several allocation APIs are captured but their playback handlers are no-ops that
return `hipSuccess` without calling the real HIP function or recording any entry
in `alloc_map`:

| API | Playback handler |
|-----|-----------------|
| `hipHostAlloc` | no-op |
| `hipMallocHost` | no-op |
| `hipMemAllocHost` | no-op |
| `hipMalloc3DArray` | no-op |

These APIs allocate pinned host memory or 3D arrays. At replay, no memory is
actually allocated and no pointer is added to `alloc_map`.

**What still works:**

- **H2D memcpy** whose source is one of these host buffers: the capture layer
  recorded the host data as a blob at capture time. At playback, `replay_memcpy_impl`
  loads the blob from disk and calls `hipMemcpy(dst_dev, blob.data(), ...)`. The
  destination is a translated device pointer from `alloc_map` (a real `hipMalloc`).
  Result: correct.

- **D2H memcpy** into one of these host buffers: the destination is a raw host
  pointer, not a device address, so `translate_ptr` is not called on it. The source
  is a device allocation in `alloc_map`. The copy and the D2H blob comparison both
  execute correctly.

**What does not work:**

- **Kernel launches** that pass the device-side alias of a `hipHostAlloc` /
  `hipMallocHost` buffer as a kernel argument. The alias is obtained at capture
  time via `hipHostGetDevicePointer`, and that device address is never recorded
  in `alloc_map`. At replay, `translate_ptr` returns `nullptr` for that argument,
  and the kernel receives a null pointer — producing wrong results or a GPU fault.

**Impact:** workloads that use pinned host memory exclusively for staging H2D and
D2H transfers (the common case) replay correctly. Workloads that pass pinned memory
pointers directly to GPU kernels will produce incorrect output.

**Workaround:** replace `hipHostAlloc` / `hipMallocHost` with `hipHostRegister` on
an existing `malloc` buffer — `hipHostRegister` _is_ fully captured and replayed,
including the device alias registration.

### Texture Objects — H2D and D2H Work, Texture Reads Do Not

`hipCreateTextureObject`, `hipDestroyTextureObject`, and `hipTexObjectCreate` are
captured but their playback handlers are no-ops. The texture object handle is never
created at replay time.

**What still works:**

- H2D memcpy that populates the underlying device array (e.g. `hipMemcpyToArray`)
  is fully replayed from its blob.
- D2H memcpy that reads back from a plain device allocation using the texture
  workload's `hipMalloc` buffer replays and validates correctly.

**What does not work:**

- Kernels that read from a texture object handle: the handle translates to
  `nullptr` (no entry in the handle maps), so the kernel reads from address 0
  and produces wrong results.

**Impact:** workloads that only create and destroy texture objects, then operate on
the underlying device memory via regular memcpy, replay correctly. Workloads where
GPU kernels perform texture fetches (the primary purpose of texture objects) will
produce incorrect output.

**Workaround:** none currently. Implementing texture replay would require recording
the `hipResourceDesc` and `hipTextureDesc` at capture time (already in
`hrr_args_hipCreateTextureObject`), then calling `hipCreateTextureObject` at
replay with the translated underlying array handle.

### Multi-Process Capture and Playback

The capture layer intercepts HIP API calls within a single process. Workloads that span
multiple processes — using `hipIpcMemHandle`, `hipExternalMemory`, or any OS-level IPC
to share GPU buffers between processes — cannot be captured as a single self-contained
archive.

Replaying such workloads would require coordinating replay across multiple process
instances, handling inter-process communication interfaces, and re-establishing IPC
handle relationships at replay time. This adds substantial complexity and is not
currently supported.

### Single-File Archive Format

The capture archive is a directory containing `events.bin`, a `blobs/` subdirectory,
and a `code_objects/` subdirectory. Managing a directory tree is less convenient than
a single portable artifact, particularly for transfer, archival, and CI pipelines.

Ideally the archive would be packaged as a single file — a format such as PKZIP/ZIP
(which supports streaming append and random-access extraction) or a similar container
would enable this. Alternative approaches (a custom flat-file format, tar, etc.) are
possible. This is not currently implemented.

### Multithreaded Playback — Missing CPU Synchronisation

Playback is optionally multithreaded (`--multi-thread`), which is necessary to achieve
performance close to the original workload and to reproduce GPU-visible ordering in
workloads that exploit CPU parallelism to pipeline GPU submissions.

However, the capture layer does not record CPU synchronisation points
(`std::mutex`, `std::condition_variable`, barriers, `pthread_join`, Win32 wait
functions, etc.). The playback therefore lacks the information needed to correctly
synchronise CPU threads. GPU-API-level ordering is preserved (via captured HIP stream
and event dependencies), but CPU-level thread sequencing is not reproduced. This
introduces correctness limitations and edge cases in multithreaded replay.

Single-threaded playback (`--multi-thread` omitted, the default) avoids the ordering
problem by serialising all events, but introduces a different correctness issue: the
HIP runtime maintains per-thread state in TLS (current device set by `hipSetDevice`,
current context, default stream, and other runtime-internal fields). In the original
workload each thread had its own TLS. Replaying events from multiple captured threads
on a single OS thread shares one TLS instance, so events that depend on TLS state set
by a different captured thread will see wrong values. For example, if thread A called
`hipSetDevice(0)` and thread B called `hipSetDevice(1)`, replaying both threads
sequentially on one OS thread leaves device 1 active for thread A's subsequent events.

Neither mode is universally safe for multi-threaded captures: multithreaded replay
preserves per-thread TLS but loses CPU synchronisation ordering; single-threaded replay
preserves event ordering but corrupts per-thread TLS. The correct choice depends on
the workload. Single-thread is the default only because it is simpler to reason about
for single-threaded captures and simple multi-threaded workloads where HIP TLS does
not vary across threads.

### HIP-Specific Polling Mechanisms

Certain HIP features interact poorly with a pure trace-and-replay model. In particular,
**polling-based mechanisms** such as `hipEventQuery` allow the application to loop on
CPU until a GPU event completes without issuing a blocking HIP synchronisation call.
The capture records the successful `hipSuccess` return, but at replay the event may not
yet be complete at the same point in execution (replay timing differs from original),
causing the application logic that relied on polling to diverge.

Other HIP features with similar complications include device-side enqueue, cooperative
groups with CPU-side barriers, and persistent kernels with CPU-side steering. These are
not currently handled.

## Relationship to Original HRR Code

- `hip_hrr.cpp` / `hip_hrr.h` — former in-tree recording hooks, superseded by this layer.
  They are compiled but no longer called (wiring removed from `hip_context.cpp`).
- `out_of_tree/hrr_proxy_win.c` — Windows proxy DLL; still useful for recording against
  a stock ROCm install where patching `amdhip64` is not possible.

## Copyright Policy

All hand-written HRR source files carry the AMD SPDX header:

```cpp
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
```

The generator (`gen_hrr_api_args.py`) emits this header at the top of all three
auto-generated files (`hrr_api_args.h`, `hip_capture_generated.cpp`,
`hip_playback_generated.cpp`) followed by an `AUTO-GENERATED` notice.
