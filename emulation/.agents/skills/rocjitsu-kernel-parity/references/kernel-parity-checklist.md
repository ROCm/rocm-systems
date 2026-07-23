# Linux AMDGPU/KFD parity checklist

Use the applicable sections to construct a behavior trace. Search by public
UAPI operation and symbol rather than assuming a fixed Linux directory layout.

## ABI and dispatch

- UAPI struct size, field width, signedness, padding, alignment, reserved bits,
  compat behavior, and in/out direction.
- Ioctl command number, flags, version/capability gates, and device node.
- Input copy timing, output initialization, partial output, and copy failure.
- Return convention and the first error selected when multiple inputs are
  invalid.

## Object and process model

- Which process, PASID, file, device, VM, or session owns the object.
- Lookup keys, duplicate creation, stale handles, cross-device use, and
  cross-process visibility.
- Reference acquisition/release and behavior during close, process exit, exec,
  fork, reset, or device removal.
- Locking scope and ordering around lookup, mutation, wait, and destruction.

## Queues and doorbells

- Queue type and property validation, ring size/alignment, priority, percentage,
  CU masks, EOP/context-save areas, and architecture-specific requirements.
- Queue ID allocation/reuse and update/destroy behavior.
- Doorbell allocation, mapping offset, width, page sharing, permissions, and
  lifetime.
- Ordering between user writes, packet visibility, queue activation, preemption,
  eviction, and teardown.

## Events and synchronization

- Event type, auto/manual reset, signal state, event-page slots, and limits.
- Wait-any/wait-all, zero events, duplicate IDs, timeout conversion/rounding,
  interruption, spurious wakeups, and destruction while waiting.
- Memory ordering between state publication and wakeup.
- Exception/event payload population and reset semantics.

## Memory and mappings

- Allocation flags, permitted combinations, size/alignment/overflow, device and
  NUMA policy, and accounting.
- VA reservation, map/unmap granularity, overlap, partial ranges, peer devices,
  and rollback after partial success.
- `mmap` offset encoding, page protection, caching, permissions, and mapping
  lifetime after handle/file close.
- Userptr pinning, invalidation, process memory, coherence, and fault behavior.
- Scratch, LDS, GWS, VRAM, system memory, and aperture distinctions relevant to
  the operation.

## Topology and properties

- Node/device enumeration stability, IDs, capability flags, cache/link
  properties, and public sysfs formatting.
- Behavior when data is absent, zero, unknown, hot-added, or removed.
- Units and conversions for clocks, memory, cache, links, and firmware fields.

## Error, rollback, and observability

- Exact `errno` and validation precedence.
- State remaining after allocation, mapping, registration, or copyout fails.
- Idempotence of cleanup and behavior for duplicate destroy/unmap.
- Logs and diagnostics do not substitute for return behavior.
- Tests observe public outcomes, not private implementation coincidences.
