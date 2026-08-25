(kernel-replay-memory-snapshot)=
# Kernel Replay — Memory Snapshot and Restore

Replaying a kernel is only meaningful if every pass sees the same inputs. Kernel replay achieves that
by capturing the replaying agent's device memory before the first pass and writing it back between
passes. This page describes what is captured, what is deliberately excluded, what happens when
capture fails, how this differs from the earlier hashed snapshot, and the limitations that follow
from the capture mechanism.

Implementation lives in `source/lib/rocprofiler-sdk/kernel_replay/memory_snapshot.cpp` (capture and
restore) and `memory_tracker.cpp` plus `utils.cpp` (the allocation inventory).

## Snapshot mechanism: this version vs hashing

**This version is a plain in-memory full copy.** `snap(agent)` copies every tracked device allocation
owned by `agent` into host RAM; `restore()` copies the whole region back. Every region, every pass.
There is **no hashing, no dirty-page diffing, no chunking, and no disk backing**.

| | This version | Expected later |
|---|---|---|
| Backing store | Host buffer per region | Open; hashing aims to cut bytes moved |
| Copy shape | One `hsa_memory_copy` per region | Restore only dirty regions |
| Dirty detection | None — every byte of every region is written back | Host-side hashing (after a device-to-host read) and/or device-side hashing (hash on the GPU, restore only dirty pages) |
| Host RAM | The **entire** tracked footprint is duplicated for the replay | Reduced if hashing restores only dirty bytes |

Dirty-page hashing is **not** implemented. It is expected in a future version, on the host and/or
the device, because snapshot and restore are bandwidth-bound and scale linearly with the tracked
footprint. An earlier prototype hashed dirty pages and stored snapshots on disk; do not assume that
behavior here.

## Allocation tracking

The tracker maintains an inventory of live device allocations. It is populated by wrappers installed
over four HSA entry points:

| Wrapped function | Table |
|---|---|
| `hsa_amd_memory_pool_allocate` | AMD extension |
| `hsa_amd_memory_pool_free` | AMD extension |
| `hsa_memory_allocate` | core |
| `hsa_memory_free` | core |

Wrappers are installed only for the first library instance; a later instance would capture the
already-installed wrapper as its own chained `next_` pointer and recurse. Recording is gated on a
relaxed atomic flag that is set when a tool configures the kernel replay callback tracing service, so
a run that never uses replay pays only that load plus the chained call.

The inventory itself lives behind a `common::Synchronized` wrapper, so every access goes through
`rlock`/`wlock` rather than a bare mutex.

### Teardown guard

The installed wrappers stay live for the whole process, which means HIP's own `__cxa_finalize` can
call the free wrapper *after* rocprofiler has destroyed the inventory's static object.
`record_alloc()`, `record_free()` and `snap_inventory()` therefore early-out on
`registration::get_fini_status() > 0`. Without that guard they lock a freed mutex, which throws
`std::system_error` into HIP's `noexcept` teardown and aborts the process.

### What is tracked

Each recorded pointer is classified by `query_alloc()`, which queries `hsa_amd_pointer_info` and
keeps the allocation only when it is **coarse-grained and not kernarg**. The exclusions are
load-bearing, not incidental:

| Excluded | Why |
|---|---|
| Kernarg memory | It holds the pointer arguments of kernels. A torn or stale restore landing on kernarg faults the GPU. |
| Host and fine-grained memory | Out of scope for replay and precarious to restore. |
| Executable allocations | HIP places its per-stream and per-graph kernarg pools — and rocprofiler its trace buffers — in the coarse-grained segment with the executable flag, so they would otherwise pass the kernarg check. They hold live kernel arguments and runtime state rather than application data; snapshotting them means a later `restore()` clobbers the in-flight kernargs of a concurrent dispatch. This is filtered from the allocation flags directly, before the `hsa_amd_pointer_info` query. |

Each surviving entry is tagged with its owning agent (`hsa_amd_pointer_info::agentOwner`), which is
what makes snapshots agent-scoped.

## What `snap()` captures

`snap(agent)` produces a `device_snapshot_t` holding one host-side copy per captured region. It
captures two categories:

1. **Tracked allocations** owned by `agent`, as filtered above.
2. **Module-scope `__device__` and `__constant__` variables** visible to `agent`.

Capture is a plain device-to-host copy through `hsa_memory_copy`, taken from rocprofiler's captured
HSA table so it uses the original, un-wrapped function.

### Module-scope variables

Module-scope variables are not `hipMalloc` allocations. They live in the loaded executable's data
segment, so the allocation tracker never sees them — yet a kernel that mutates a `__device__` global
would leak that mutation into the next pass. `snap()` therefore discovers them separately: it
iterates the loaded code objects and, for each one, calls `hsa_executable_iterate_agent_symbols` for
the replaying agent, collecting symbols of kind `HSA_SYMBOL_KIND_VARIABLE` along with their device
address and size. Symbols with a zero address or size are skipped, as is anything larger than 1 GiB
(a sanity cap).

Iterating for a specific agent naturally scopes the walk to executables loaded on that agent —
others yield no symbols — which matches `snap()`'s per-agent contract.

This discovery must run at snapshot time rather than at executable-load time, because constant memory
is not necessarily populated when the executable loads.

## Failure handling: an incomplete snapshot declines replay

Capture of a single region fails if the host allocation fails under memory pressure (a `std::bad_alloc`
from the resize) or if the device-to-host copy fails. Either leaves the snapshot incomplete, and
`snap()` returns with `ok == false`.

The replay loop treats that as a decision point, not an error to push through: restoring a partial
snapshot between passes would write back some regions and leave others carrying a later pass's
mutations, corrupting application data. So on `ok == false` the loop warns, closes the CONFIG
callback sequence, and runs the dispatch **exactly once** with its original completion signal, still
under the agent writer lock. The application sees a normal, un-replayed execution.

## What `restore()` writes back

`restore()` copies each captured region back host-to-device and returns the number of regions
successfully restored. The two categories are handled differently:

- **Tracked allocations** are re-validated first, by `region_is_restorable()`. The lookup and the copy
  happen together under the inventory read lock, so a concurrent free cannot race between the check
  and the write. A region that fails the check is skipped with a warning rather than written to
  memory that no longer belongs to it.
- **Module-scope variables** live in the loaded executable and are always present, so they are
  restored directly.

### Why an address is not an identity

Re-validating on `(base, size)` is not sufficient, and the gap is not theoretical. The allocate and
free wrappers deliberately sit outside the per-agent replay lock, so between `snap()` recording a
region and `restore()` writing it back, another thread can free that allocation and be handed the
same base address for a new one. With a caching or pooling allocator — which is what most frameworks
put in front of the device allocator — same-address reuse is the *expected* outcome of a free
followed by an allocation of similar size, not an unlikely race. Both base and size can match while
the memory belongs to something else entirely.

Every tracked allocation therefore carries a process-monotonic `generation` stamp assigned at
allocation time. `snap()` records the generation it observed; `restore()` refuses to write unless the
address still carries it. A mismatch means the address was reused, and the region is skipped.

The asymmetry is deliberate. Skipping a region that was in fact still ours costs accuracy on later
passes and is reported on the outcome line. Writing into an allocation that is *not* ours corrupts
live application data and would be reported nowhere. So every ambiguous case — freed, shrunk, grown,
or restamped — refuses.

A generation of `0` means "any live allocation of at least this size" and is used only for regions the
tracker does not own (module-scope variables). No tracked allocation is ever stamped `0`, so a tracked
region cannot fall through to the weaker check by accident.

A failed host-to-device copy on a live region is different from a skip: the snapshot is then only
partially applied, some regions hold pre-kernel bytes and others post-kernel bytes, and nothing can
put that back. `restore()` returns false and the replay window aborts rather than hand the
application a mixture it cannot detect.

## HIP graphs

HIP graph launches are not replayed. The interceptor warns once (process-wide) and the graph runs
once on the ordinary path. Excluding `graph_launch_active` from the replay gate is what makes that
a graceful decline rather than an abort: snapshot and restore around a graph's runtime-managed
memory and ordering is undefined, so the dispatch is not replayed.

A graph with no other consumers runs un-replayed through the interceptor's fast path. The warning
is emitted once so a large graph workload does not flood the log.

Graph replay is future work.

## Known limitations

Stated plainly, because each one has a concrete cause in the mechanism above.

- **No dirty-page hashing.** Every tracked byte is copied on every restore. Host-side and/or
  device-side hashing is expected later; see [Snapshot mechanism](#snapshot-mechanism-this-version-vs-hashing).
- **Virtual-memory mappings are counted, not snapshotted.** Memory that becomes addressable through
  `hsa_amd_vmem_map` rather than through one of the wrapped allocate entry points cannot be captured:
  the snapshot has no record of the underlying physical handle. This covers **`hipMallocAsync` and
  other pool-backed, stream-ordered allocations**, `hipMemAddressReserve`/`hipMemMap`, and the
  expandable-segment allocators built on them. A kernel writing such a buffer will not have those
  writes reverted between passes.

  The map and unmap entry points *are* wrapped, so those regions are counted even though they cannot
  be restored, and the replay window declines by default when any are live rather than reporting
  counters gathered on mutated inputs. That turns the limitation from a silent wrong answer into a
  named decline. See
  [Concurrency and isolation](kernel_replay_concurrency_and_isolation.md#admission-control).
- **Async SDMA copies are not fenced by the replay window.** `hsa_amd_memory_async_copy` and its
  variants are not kernel dispatches, so they bypass both the AQL queues and the per-agent replay
  gate. The source marks serializing them as a follow-up. See
  [Concurrency and isolation](kernel_replay_concurrency_and_isolation.md#what-is-not-isolated).
- **Coarse-grained device memory only.** Kernarg, host, fine-grained and executable allocations are
  excluded by design, as is unified and managed memory.
- **HIP graph launches are not replayed** (warn once, run once), as described above.
- **Only single-packet, single-dispatch submissions are replayed.** The replay gate requires exactly
  one packet in the batch and exactly one dispatch packet in it; anything else takes the normal path.
- **The tracked device footprint is duplicated in host RAM** for the lifetime of the replay, and
  every region is copied on every restore. Snapshot and restore cost scales linearly with the tracked
  footprint, so for large footprints `N` passes of snap plus restore can cost more than re-running
  the application `N` times.
- **Single process.** There is no cross-process or multi-process coordination, so replaying a kernel
  that participates in inter-process collectives is not safe.

## Source reference

All paths are relative to `projects/rocprofiler-sdk/`.

| Component | File | Symbol |
|---|---|---|
| Snapshot capture | `source/lib/rocprofiler-sdk/kernel_replay/memory_snapshot.cpp` | `snap()` |
| Module-variable discovery | `source/lib/rocprofiler-sdk/kernel_replay/memory_snapshot.cpp` | `discover_module_variables()`, `collect_module_variable()` |
| Restore | `source/lib/rocprofiler-sdk/kernel_replay/memory_snapshot.cpp` | `restore()` |
| Restore identity gate | `source/lib/rocprofiler-sdk/kernel_replay/memory_snapshot.cpp` | `region_is_restorable()` |
| Footprint estimate | `source/lib/rocprofiler-sdk/kernel_replay/memory_snapshot.cpp` | `estimate_footprint()` |
| Snapshot types | `source/lib/rocprofiler-sdk/kernel_replay/memory_snapshot.hpp` | `device_snapshot_t`, `mem_block_t` |
| Allocation generation stamp | `source/lib/rocprofiler-sdk/kernel_replay/memory_tracker.hpp` | `alloc_info_t::generation` |
| Untracked-memory accounting | `source/lib/rocprofiler-sdk/kernel_replay/memory_tracker.cpp` | `untracked_device_memory()`, `record_vmem_map()` |
| HSA hook installation | `source/lib/rocprofiler-sdk/kernel_replay/memory_tracker.cpp` | `memory_tracker_init()` |
| Allocation recording | `source/lib/rocprofiler-sdk/kernel_replay/memory_tracker.cpp` | `record_alloc()`, `record_free()` |
| Agent-scoped inventory | `source/lib/rocprofiler-sdk/kernel_replay/memory_tracker.cpp` | `snap_inventory()` |
| Trackability classifier | `source/lib/rocprofiler-sdk/kernel_replay/utils.cpp` | `query_alloc()` |
| Tracking activation | `source/lib/rocprofiler-sdk/callback_tracing.cpp` | `rocprofiler_configure_callback_tracing_service()` |
| Snapshot-declined fallback | `source/lib/rocprofiler-sdk/hsa/queue.cpp` | `snapshot.ok` branch in `WriteInterceptor` |
| HIP graph warn-once + run-once | `source/lib/rocprofiler-sdk/hsa/queue.cpp` | `_warned_graph_replay`; `graph_launch_active` excluded from the replay gate |
| Tests (GPU) | `source/lib/rocprofiler-sdk/kernel_replay/tests/` | `snap_restore.cpp`, `snap_allocator_patterns.cpp`, `snap_region_scaling.cpp`, `snap_bandwidth.cpp`, `snap_kernels.{hpp,cpp}` |
| Tests (no GPU) | `source/lib/rocprofiler-sdk/kernel_replay/tests/` | `replay_diagnostics.cpp` |
