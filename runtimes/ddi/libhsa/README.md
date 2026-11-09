<!-- Copyright (c) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->

# libhsa

libhsa implements the HSA C ABI over [rocddi](../rocddi/README.md). It reports
HSA runtime version 1.21 and AMD extension version 1.36. This early frontend
does not yet provide complete ROCr compatibility. The
[HSA headers](../../api-headers/include/hsa/) define the public ABI; an
exported entry point does not imply that every request is supported.

## Runtime model

`hsa_init` and `hsa_shut_down` manage one reference-counted HSA state per
process. It holds CPU and GPU agent snapshots; focused modules track queues,
signals, memory handles, and executables. rocddi supplies native device
mechanisms, while
[rocloader](../../loader/rocloader/README.md) handles HSA code objects and
executable state. Final shutdown waits for outstanding native calls and stops
runtime workers. On Linux KFD, a primary process context may remain after
shutdown; later runtime generations use an instance-owned context when KFD
supports it.

The final reference closes one process admission gate under the lifecycle
lock and requests worker stop. Calls already admitted retain their generation;
nested work on the same thread can finish while its parent lease is live.
Unbounded signal waits observe the stop request and return their last observed
value. Shutdown waits for calls on other threads. A shutdown invoked from an
admitted callback or runtime worker uses a helper to finish teardown after
that caller returns. Application code must avoid holding a lock needed by a
worker callback while calling `hsa_shut_down`, since shutdown joins workers.

Final shutdown has three outcomes:

| Outcome | Ownership and next call |
| --- | --- |
| Complete | HSA state and its rocddi session retire; a later `hsa_init` can create a new generation. |
| Native retry | HSA objects have retired, but the detached rocddi session retains failed native cleanup. Another `hsa_shut_down` retries it; `hsa_init` remains blocked until it succeeds. |
| Retained generation | Worker, callback, or HSA object cleanup cannot prove that native access has stopped. The remaining owner graph stays live until process exit. Further init and shutdown calls report invalid runtime state. |

An unwind during detached shutdown also retains the remaining owner. During
initialization, failed native rollback retains its session and prevents a new
generation. The process gate and its reference count belong to libhsa because
they implement the HSA C ABI. rocddi owns the independent native session and
its retry mechanism; AMDF instances use that mechanism without joining the
HSA process gate.

The single HSA CPU agent is frontend policy built from rocddi's passive host
topology snapshot. It does not activate a CPU device. HSA prefers CPU cache
records from OS NUMA node zero and uses the system scan when that node is
absent. Cache handles, names, callback order, and agent attributes belong to
libhsa.

The CMake build stages `libhsa-runtime64.so.1` with `ROCR_1` symbol versions.
Its `libamdf.so` alias exports AMDF from the same image. The two ABIs have
separate frontend lifecycles and rocddi sessions; process-lifetime KFD context
resources are shared by rocddi. HSA calls do not route through AMDF.

## Current support

- CPU and GPU agents, memory regions and pools, virtual memory, SVM, IPC,
  signals, AQL and SDMA queues, and synchronous and asynchronous copies.
- Code-object and executable loading for matching GFX12 targets through
  rocloader. The image extension is advertised only on gfx1200 and gfx1201,
  for supported formats and layouts.
- AMD PC sampling on gfx1200, gfx1201, or gfx1250 when KFD reports native
  configurations. Its GPU behavior still needs hardware qualification.
- `rocprofiler-register` and legacy `HSA_TOOLS_LIB` loading on Linux x86-64.
  libhsa retains each registrar table and dispatches through stable bridges
  so existing SDK late attachment remains visible. The current registrar ABI
  provides no update notification or reader lock: a concurrent table rewrite
  and bridge read remain a known race until that ABI gains synchronization.
  The existing SDK wrapper stops its completion monitor before native HSA
  shutdown; libhsa does not call private SDK lifecycle entry points. A late
  registrar replay must not overlap final `hsa_shut_down`, because it can
  restart that monitor after the wrapper stops it. The existing ABI does not
  provide a fence for this overlap. Each full HSA reinitialization creates
  another registrar instance, which the current registrar limits to eight;
  profiler resource reuse across those generations is not qualified.
  Legacy tools must finish table edits before `OnLoad` returns. libhsa
  snapshots those edits into atomic dispatch slots and retains every staging
  table through process exit, so a tool may keep a table pointer across
  `OnUnload` and attach again on a later HSA generation. Final shutdown closes
  HSA API admission when the last reference is released, then waits for
  admitted calls and callback producers before `OnUnload`. Runtime-dependent
  copied API functions called from `OnUnload` report `NOT_INITIALIZED`; signal
  and queue index functions with no status result use their existing fallback.
  The pure `hsa_status_string` query remains callable. A tool's
  `hsa_shut_down` wrapper may still be returning after `OnUnload`, so the tool
  must retain any state that wrapper needs until it returns. The AMD profiler
  and AQLprofile extension tables are not advertised.

Availability depends on the selected agent and driver. Query system and agent
extensions, memory pools, and queue capabilities before relying on a feature.
The finalizer extension and external semaphore operations are not supported.

## Build and verification

See the [workspace build instructions](../../README.md) for the CMake build,
library aliases, and Cargo commands. The current platform adapter is Linux;
the runtime builds on x86-64 and AArch64. GPU execution was previously
observed on one x86-64 GFX1201 host. CTest runs Rust tests and the native C
smoke test without a GPU. The [GPU probes](tests/gpu/README.md) cover
HSA callbacks, logging, sampling, and kernel dispatch on a qualified host.
