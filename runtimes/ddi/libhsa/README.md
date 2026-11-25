<!-- Copyright (c) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->

# libhsa

libhsa implements the HSA C ABI over [rocddi](../rocddi/README.md). It reports
HSA runtime version 1.21 and AMD extension version 1.39. The frontend
does not yet provide complete ROCr compatibility. The
[HSA headers](../../api-headers/include/hsa/) define the public ABI; an
exported entry point does not imply that every request is supported.

## Runtime model

`hsa_init` and `hsa_shut_down` manage one reference-counted `HsaSession` per
process. The lifecycle coordinator owns publication and retirement transitions;
the session owns public handle registries, agent views, callbacks, and its
private rocddi native session. Resource modules track queues, signals,
memory handles, and executables. rocddi supplies native device mechanisms, while
[rocloader](../../loader/rocloader/README.md) handles HSA code objects and
executable state. Final shutdown waits for outstanding native calls and stops
runtime workers. On Linux KFD, clean later sessions reuse the exact retained
primary connection and GPU VMs. This preserves primary-context SVM and profiling
support across HSA sessions. A pending native disable, changed GPU identity,
sticky device loss, or uncertain cleanup still prevents reuse.

The final reference closes public admission under the lifecycle lock and
cancels ordinary waits and native work. Calls already admitted retain their
session; nested work on the same thread can finish while its parent lease
is live. Scoped closures install and restore thread-local admission state;
transferable tokens retain work that moves to another thread. Unbounded signal
waits return their last observed value. When a
compatible profiler is registered, preparation gives it explicit generation
leases to stop profiling engines while native queues and completion callbacks
remain available. The rocddi session stays open throughout tool preparation and
tool retirement. Shutdown closes cleanup delegation, drains its admitted descendants,
joins callbacks, and checks native queue inactivation before tools release their
operands. Only after tools finish does the coordinator close native session
admission and retire HSA objects. A shutdown
invoked from an admitted callback or runtime worker uses a helper to finish
teardown after that caller returns. Application code must avoid holding a lock needed by a
worker callback while calling `hsa_shut_down`, since shutdown joins workers.
Each admitted-work drain waits up to five seconds. A timeout retains the
session at that checkpoint and reports `HSA_STATUS_ERROR_RESOURCE_BUSY`.
Public admission stays closed, while already-admitted operations and their
nested calls retain access to their HSA session. Another `hsa_shut_down` resumes
the incomplete drain; it does not repeat successful tool preparation. A timeout
never authorizes releasing storage that native work may still use.

Final shutdown distinguishes four outcomes:

| Outcome | Ownership and next call |
| --- | --- |
| Complete | HSA state and its rocddi session retire; a later `hsa_init` can create a new HSA session. |
| Pending drain | Known admitted work still owns the HSA session. Another `hsa_shut_down` resumes the ordinary-call or tool-cleanup drain; `hsa_init` cannot create a replacement. |
| Native retry | HSA objects have retired, but the detached rocddi session retains failed native cleanup. Another `hsa_shut_down` retries it; `hsa_init` remains blocked until it succeeds. |
| Retained HSA session | Worker, callback, or HSA object cleanup cannot prove that native access has stopped. The remaining owner graph stays live until process exit. Further init and shutdown calls report invalid runtime state. |

An unwind during detached shutdown also retains the remaining owner. Failed
initialization returns the original startup error together with an internal,
typed rollback outcome: complete, retryable native retirement, or uncertain
frontend cleanup. A later `hsa_init` retries pending native retirement before
constructing an HSA session or acquiring a public reference. Uncertain cleanup
prevents further initialization.

The process gate and its reference count belong to libhsa because they implement
the HSA C ABI. rocddi owns the independent native session and its retry mechanism;
AMDF instances use that mechanism without joining the HSA process gate. The
[architecture guide](../../docs/architecture.md#hsa-session-ownership) maps
these responsibilities to their source owners. `HsaSession` extends native
session ownership through composition; generation numbers distinguish its
successive lifetimes. Shared rocddi storage handles explicit retention after
uncertain cleanup for both HSA and AMDF.

On Linux, the first `hsa_init` binds the runtime image to its creating process.
After `fork`, the child must `exec` before using HSA. The inherited child's
`hsa_init` reports `HSA_STATUS_ERROR`, and runtime-dependent status APIs report
`HSA_STATUS_ERROR_NOT_INITIALIZED`; raw signal and queue-index APIs return
their existing fallback values. Public dispatch and profiler bridges bypass
inherited tool wrappers before reaching those guards. The parent retains its
HSA state. The dispatch guard requires Linux `MADV_WIPEONFORK` support; if
that advice cannot be installed, `hsa_init` fails instead of publishing an
unguarded runtime. Applications should `exec` or `_exit` promptly in the
child, especially when other threads existed at `fork`.
If a callback itself calls `fork`, its child must `exec` or `_exit` before
returning from the callback into the inherited runtime call.

HSA CPU agents are frontend projections of rocddi's passive host topology.
Linux CPU nodes retain their memory-bank and NUMA associations; a generic
host CPU projection is used when native CPU nodes are absent. Cache selection
follows explicit locality and CPU sharing records. Missing CPU model names
receive the HSA display fallback. These projections do not activate CPU devices;
cache handles, names, callback order and public attributes remain frontend policy.
Generations with no selected GPU use host pages for ordinary signals and
system-pool signal batches, including waits and asynchronous callbacks.

The CMake build stages `libhsa-runtime64.so.1` with `ROCR_1` symbol versions.
Its `libamdf.so` alias exports AMDF from the same image. The two ABIs have
separate frontend lifecycles and rocddi sessions; process-lifetime KFD context
resources are shared by rocddi. HSA calls do not route through AMDF.

## Debugger and trap ownership

Before native activation, HSA requests a process-lifetime debugger rendezvous
with ABI version 11. Matching runtime trap handlers and TTMP setup provide the
trap ABI expected by current rocdbgapi. Loaded code objects enter the native
debugger list only after executable freeze completes their writes and code-cache
publication; they leave that list before executable backing is released.

An already active AMDF or other primary-context registration can prevent HSA
from owning that rendezvous. HSA initialization and execution remain available
in that case; the existing registration and trap binding are preserved. GPU
debugging through this HSA session is unavailable, and AMD PC sampling
creation reports `HSA_STATUS_ERROR_NOT_SUPPORTED` because HSA cannot safely
restore an unowned trap binding. Activate HSA before the other native runtime
when these facilities are needed. An active KFD runtime cannot safely change its
debugger registration while another client owns queues.

PC sampling retains the normal debugger trap while installing its sampling
handler and restores the normal handler on destruction. Trap code, arguments,
and executable images remain retained after ambiguous native retirement.
The adjacent [trap sources and license](src/debugger/traps/LICENSE) and
[regeneration script](src/debugger/traps/regenerate.py) reproduce the embedded
handler programs without depending on an external ROCr checkout at build time.

`HSA_XNACK=0` and `HSA_XNACK=1` request the corresponding native fault mode before
activation. Requests are advisory: a rejected change falls back to a confirmed
live mode, which is also used for ISA names and code-object compatibility.
Other values leave the mode unchanged. A failed live-mode query is an
initialization error rather than an assumed mode.

## Current support

- CPU and GPU agents, memory regions and pools, virtual memory, SVM, IPC,
  signals, AQL and SDMA queues, and synchronous and asynchronous copies.
- Modern code-object loading on gfx908, gfx90a, gfx942, gfx950, selected
  gfx10.1/gfx10.3/gfx11.0/gfx11.5 targets, gfx1200, gfx1201, and gfx1250.
  Admission requires a matching loader, AQL/scratch implementation, and
  compute-copy asset. Loader extension 1.04 supports caller-owned executable
  destinations, published at freeze without taking ownership of their mapping.
- Images and samplers on those GFX9/GFX10/GFX11 and GFX12.0 targets, including
  imported-image transfers, packed/depth/sRGB formats, and sampler-v3
  anisotropy. Format and geometry queries report the supported combinations.
- SVM pointer classification and discard/prefetch batches when XNACK is
  enabled, rectangular copy batches, non-peer staging, and compute copy/fill.
- Scratch occupancy reduction, use-once reclamation, and asynchronous
  reclamation on firmware supporting the CP protocol. Local queue descriptors
  require a coherent host XGMI route with atomic access. SDMA queues support
  endpoint fault callbacks and normal scheduling priority.
- AMD PC sampling on gfx1200, gfx1201, or gfx1250 when KFD reports native
  configurations. Its GPU behavior still needs hardware qualification.
- Legacy `HSA_TOOLS_LIB` loading on Linux x86-64 and AArch64.
  Legacy tools must finish table edits before `OnLoad` returns. libhsa
  snapshots those edits into atomic dispatch slots and retains every staging
  table through process exit, so a tool may keep a table pointer across
  `OnUnload` and attach again on a later HSA session. Final shutdown closes
  HSA API admission when the last reference is released, then waits for
  admitted calls and callback producers before `OnUnload`. Synchronous HSA
  calls from `OnUnload` retain cleanup admission while the native session and
  public objects remain available. Ordinary callers stay excluded, and unload
  does not reopen the earlier window for delegated asynchronous cleanup.
  A tool's `hsa_shut_down` wrapper may still be returning after `OnUnload`, so it
  must retain any state that wrapper needs until it returns. The AMD profiler
  and AQLprofile extension tables are not advertised.

The runtime also implements an optional version 2 registrar boundary for
immutable profiler snapshots, generation-scoped worker admission, and separate
prepare/retire callbacks. These entry points are exercised by a local registrar
fixture. The registrar and SDK currently in this checkout do not implement that
protocol; their companion changes are deferred. SDK registration, concurrent
late attachment, and SDK resource reuse across HSA sessions are therefore
not supported by the current component combination. Runtime initialization and
legacy tools remain available when the version 2 symbols are absent. The older
mutable-table registrar ABI is not used because it lacks a writer/shutdown fence.

The private versioned descriptor-interception hook preserves native batch
errors, priority, CU masks, requested ring/control placement, and cooperative
queue references. Integration of that hook into the SDK and attachment helper
also requires the deferred companion changes. It does not change the public
HSA headers or make the current SDK wrappers descriptor-aware.

Availability depends on the selected agent and driver. Query system and agent
extensions, memory pools, and queue capabilities before relying on a feature.
The finalizer extension and external semaphore operations are not supported.
`hsa_amd_svm_prefetch_async` requires confirmed migration completion. Linux KFD
supports this only when the native endpoint advertises the explicitly negotiated
checked-prefetch protocol. Ordinary drivers and inactive contexts report
`HSA_STATUS_ERROR_NOT_SUPPORTED`. The local kernel correction is qualified on gfx1201;
its exact driver/kernel coverage is recorded in
[the SVM investigation](../../docs/testing/rocr-gfx1201-svm-prefetch.md).

Default fatal handling captures GPU core dumps before aborting on unhandled
AQL queue errors and registered system events. It does not write to application
stderr, which may block indefinitely. Application callbacks retain priority.
The runtime does not yet start a process-wide fault observer without an
application system-event registration. See the
[capture qualification](../../docs/testing/rocr-gfx1201-core-dump.md) for the
output format and limits, including unavailable private VRAM contents.

The current target profile excludes gfx11.7 because no matching compute-copy
object can be generated by the selected compiler. gfx1250 images remain
unavailable because that compiler cannot generate the required image kernel.
gfx1251 has loader definitions but no established native address/queue profile
in the selected kernel baseline. These are implementation/toolchain limits,
separate from whether a particular device has been tested. Legacy V1/V2 code
objects are rejected instead of being decoded as modern descriptors.

## Configuration

These controls are sampled when an HSA session initializes. Native device
identity is independent of the resulting HSA agent order.

| Control | Behavior |
| --- | --- |
| `ROCR_VISIBLE_DEVICES` | Ordered GPU ordinals or unique UUID prefixes. An empty value exposes CPU topology only; an invalid or repeated token ends the accepted prefix. |
| `HSA_XNACK` | Advisory `0`/`1` request before activation; ISA reporting uses the confirmed effective mode. |
| `HSA_CU_MASK` | Process CU restrictions intersect every queue mask, including later mask resets. |
| `HSA_ENABLE_SDMA`, `HSA_ENABLE_PEER_SDMA` | `0` disables the selected SDMA path; copies use the compute/staging implementation. |
| `HSA_NO_SCRATCH_RECLAIM` | `1` disables scratch reclamation. |
| `HSA_ENABLE_SCRATCH_ASYNC_RECLAIM` | `0` disables asynchronous reclaim on supported firmware. |
| `HSA_NO_SCRATCH_THREAD_LIMITER` | `1` disables occupancy reduction after scratch allocation pressure. |
| `HSA_SCRATCH_SINGLE_LIMIT`, `HSA_SCRATCH_SINGLE_LIMIT_ASYNC` | Per-queue byte thresholds, clamped to the device scratch aperture. The asynchronous threshold can also be updated through the AMD agent API. |
| `HSA_DISABLE_FRAGMENT_ALLOCATOR` | `1` disables allocation fragments. |
| `HSA_ENABLE_YIELDING_POLL` | `1` enables bounded sleep backoff for polling-only scalar signal waits after a 200 microsecond hot window (10 milliseconds for an active hint). |
| `HSA_ENABLE_QUEUE_FAULT_MESSAGE` | `0` disables default fatal handling for AQL queue errors; explicit application callbacks remain active. |
| `HSA_DISABLE_COREDUMP_ON_EXCEPTION` | `1` suppresses GPU core output while preserving fatal termination. |
| `HSA_COREDUMP_PATTERN` | Nonempty file pattern or pipe command for GPU cores, with `%p/%i/%h/%e/%t/%%` substitutions. Otherwise use the kernel pattern with GPU fallback. File output honors `RLIMIT_CORE`; pipe output has a bounded write/exit deadline. |
| `GPU_MAX_HW_QUEUES` | Limits counted hardware queues per agent and priority (default `4`); `0` prevents acquisition. Unset, empty, or invalid values use the default. |
| `HSA_FORCE_FINE_GRAIN_PCIE` | `1` exposes the fine-grain PCIe pool policy. |

## Build and verification

See the [workspace build instructions](../../README.md) for the CMake build,
library aliases, and Cargo commands. Native adapters now distinguish Linux
handles from pointer-sized Windows handles and qualify native sharing and trace
protocols. HSA Rust sources and tests type-check for Windows x86-64 MSVC;
Windows DLL export generation is deferred. Complete SDK/C++ linkage and native
Windows execution remain unqualified. Windows GPU
initialization, tool dispatch and borrowed CRT logging remain unavailable.
The Linux runtime builds on x86-64 and AArch64. GPU execution was previously
observed on one x86-64 GFX1201 host. CTest runs Rust tests and the native C
smoke test without a GPU. The [GPU probes](tests/gpu/README.md) cover
HSA callbacks, logging, sampling, and kernel dispatch on a qualified host.
