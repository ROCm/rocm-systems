# MSC Atomic Alias Coherence Design

## Context

`L2Cache::atomic_rmw` acquires the device-wide atomic boundary before accessing
backing memory. The boundary flushes every registered L2 and then advances the
global coherence epoch while holding the L2 maintenance locks.

The direct-memory path therefore observes the flushed value. The linked-port
path instead reads through `MemorySideCache` (MSC). MSC is write-through and
keys clean cache lines by `(VMID, GPU virtual address)`. If two such keys alias
the same host storage, a clean line under one key can remain stale after another
key publishes dirty L2 data. An atomic RMW through the stale key can consequently
use the old value.

## Goals

- Make linked-port atomic RMWs observe the authoritative value after all L2
  dirty data has been published.
- Ensure stale MSC aliases are not visible after a coherence epoch change.
- Preserve the existing device-wide atomic serialization and L2 locking model.
- Avoid scanning or invalidating the entire 128 MiB MSC on every atomic.
- Add regression coverage for two L2s sharing one MSC and HBM through aliased
  VMID/virtual-address mappings.
- Remove the unused `register_cache` helper so there is one L2 registration
  implementation and one ordering invariant.

## Design

Each MSC cache tag will record the device-coherence epoch at which its line was
filled or refreshed. MSC already protects tag lookup and replacement with the
corresponding stripe lock, so the epoch metadata will use that same protection.

On an MSC access:

1. Look up the `(VMID, GPU virtual address)` cache line under its stripe lock.
2. Compare the tag's recorded epoch with
   `DeviceCacheCoherence::current_epoch()`.
3. If the epochs differ, require the old entry to be clean, invalidate it, and
   refill the line from backing HBM.
4. Stamp a newly filled or refreshed entry with the current epoch.
5. Continue the requested read or write.

The atomic boundary's existing order remains unchanged:

1. Acquire the global exclusive coherence lock.
2. Lock every registered L2 maintenance mutex in sorted address order.
3. Flush dirty L2 data through MSC to HBM.
4. Advance the global coherence epoch.
5. Perform the linked-port RMW while the boundary remains held.

After step 4, any MSC alias retained from an earlier epoch is stale. The RMW
read therefore invalidates and refills that alias from HBM, where step 3
published the authoritative value. Other old aliases are invalidated lazily on
their next access.

MSC is write-through, so an epoch-stale entry is expected to be clean and can
be discarded without losing data. No MSC registration or new global lock is
needed.

## Alternatives

### Eagerly flush and invalidate the entire MSC

Registering MSC with the atomic boundary and calling `flush_all()` would be
simple, but it scans 1,048,576 tags and locks all 256 stripes for every atomic.
That cost is disproportionate to an individual RMW.

### Bypass MSC only for atomic backing accesses

Sending the RMW directly to HBM would obtain the correct operand, but clean MSC
aliases could remain stale and be returned after the atomic. A generation or
invalidation mechanism would still be required, while the backing protocol
would become more complex.

## Validation

Add a regression with two L2 caches linked to one MSC and HBM controller:

1. Map different `(VMID, GPU virtual address)` pairs to the same host word.
2. Read through the second alias to retain a clean MSC value of `0`.
3. Dirty `40` through the first L2 without immediately publishing it.
4. Increment through the second L2's atomic RMW.
5. Verify the callback observes `40`, the atomic result and host word are `41`,
   and subsequent reads through both aliases return `41`.

Run the focused L2/MSC tests in Release and under TSan, followed by the relevant
non-benchmark test suite.

Measure end-to-end performance against commit `b685943638` using representative
large kernels from `rocjitsu-test-corpus`, including a large softmax, a large
pack/unpack case, and a large matrix multiplication. Use the same pinned-CPU,
alternating-run methodology as the previously published PR benchmark where the
harness permits it.
