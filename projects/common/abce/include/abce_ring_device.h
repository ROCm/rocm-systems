/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Accelerated Blit Copy Engine (ABCE) — device-side SDMA ring.
//
// The device wrapper around the shared reserve/commit protocol in
// abce_ring_core.h — the same algorithm the host RingBuffer in abce_ring_host.h
// runs. It adds only the concerns the host wrapper cannot express: dword stores
// into the ring, and releasing them to system scope as the doorbell is rung.
//
// Requirements: the ring buffer and its read/write/doorbell control words must
// be DEVICE-VISIBLE and the doorbell device-writable (an hsa_amd_queue_create
// SDMA queue whose read/write pointer and doorbell addresses are exposed via
// hsa_amd_queue_get_info). Host and device producers may share one ring only if
// both point at the same SharedRingControl.

#ifndef ABCE_RING_DEVICE_H_
#define ABCE_RING_DEVICE_H_

#include <cstdint>

#include "abce_ring_core.h"  // shared reserve/commit protocol (host+device)

namespace abce {

/// @brief Device-side SDMA ring handle. Trivially copyable; may be passed to a
/// kernel by value or kept in device memory and shared across launches.
struct DeviceRing {
  uint32_t* base = nullptr;                ///< ring buffer (dwords), device-visible.
  uint64_t size = 0;                       ///< ring size in bytes; MUST be power of two.
  volatile uint64_t* read_ptr = nullptr;   ///< hw read index (bytes, monotonic).
  volatile uint64_t* write_ptr = nullptr;  ///< hw write index (bytes) the engine reads.
  volatile uint64_t* doorbell = nullptr;   ///< doorbell word (bytes); poke to run.

  SharedRingControl* control = nullptr;  ///< state shared with host producers.

#if defined(__HIPCC__) || defined(__CUDACC__)
  __host__ __device__ bool IsValid() const {
    return base && size != 0 && (size & (size - 1)) == 0 && read_ptr && write_ptr && doorbell &&
           control;
  }

  // Wrap / reservation come from the shared host+device core (abce_ring_core.h)
  // — the same algorithm the host RingBuffer runs.
  __device__ uint64_t Wrap(uint64_t index) const { return RingWrap(index, size); }

  /// Reserve `bytes` of contiguous ring space. If the payload would cross the
  /// ring end, publish the wrap tail as NOPs first and retry at offset zero.
  /// Multi-producer safe via CAS on reserve_cursor.
  __device__ RingStatus Reserve(uint64_t bytes, RingReservation& reservation) {
    if (!IsValid() || (bytes % sizeof(uint32_t)) != 0) return RingStatus::kInvalidArgument;
    return RingReserve(&control->reserve_cursor, read_ptr, size, bytes, reservation,
                       [this](const RingReservation& padding) {
                         StoreZeroDwords(padding.start, padding.bytes());
                         return Submit(padding);
                       });
  }

  /// Satisfy the builders' zeroed-buffer contract over a held reservation.
  __device__ RingStatus Zero(const RingReservation& reservation) {
    if (!IsValid() || !reservation.valid()) return RingStatus::kInvalidArgument;
    StoreZeroDwords(reservation.start, reservation.bytes());
    return RingStatus::kSuccess;
  }

  /// Address of the @p byte_count bytes at monotonic @p write_index. Does not
  /// re-validate the ring: Reserve() already did, and every caller holds a
  /// reservation it handed out. Returns null only if the write would leave the
  /// buffer, which a reservation cannot do — Reserve never returns one that
  /// straddles the physical end.
  __device__ char* WriteAddress(uint64_t write_index, uint64_t byte_count) const {
    const uint64_t offset = Wrap(write_index);
    if (offset + byte_count > size) return nullptr;
    return reinterpret_cast<char*>(base) + offset;
  }

  /// Publish [start, end): wait until earlier reservations have committed, then
  /// update the hardware write pointer and ring the doorbell in order. The
  /// in-order commit skeleton is the shared core's RingPublish; the doorbell
  /// mechanics are the device-specific publish step.
  ///
  /// THE DOORBELL STORE IS THIS SUBMISSION'S SYSTEM-SCOPE RELEASE POINT. It is
  /// what orders the packet dwords — and everything the calling thread wrote
  /// beforehand, including the buffers the packets name — ahead of the engine
  /// being told to run. Saying that with a release store rather than hand-rolled
  /// waits is what makes it portable: the compiler emits each target's own
  /// writeback and waits for it to land before the store (buffer_wbl2 sc0 sc1 +
  /// s_waitcnt vmcnt(0) on gfx9, global_wb SCOPE_SYS + s_wait_storecnt on
  /// gfx125+), which is the same drain the older explicit waitcnt did by hand.
  __device__ RingStatus Submit(const RingReservation& reservation) {
    if (!IsValid()) return RingStatus::kInvalidArgument;
    return RingPublish(&control->commit_cursor, reservation, [this](uint64_t new_write_index) {
      // Relaxed: the engine cannot look at this until the doorbell below, and
      // that store's release orders this one ahead of it.
      __hip_atomic_store(write_ptr, new_write_index, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      __hip_atomic_store(doorbell, new_write_index, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
      // High-water mark for RingBuffer::Drain(), which only compares it against
      // the engine's read index — no data is published through it, so it needs
      // the value to be coherent and nothing more. Relaxed keeps the release
      // above from being paid for a second time on every publish.
      __hip_atomic_store(&control->max_write_index, new_write_index, __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_SYSTEM);
    });
  }

 private:
  __device__ void StoreZeroDwords(uint64_t write_index, uint64_t byte_count) {
    const uint64_t base_dword = Wrap(write_index) / sizeof(uint32_t);
    const uint64_t num_dwords = byte_count / sizeof(uint32_t);
    for (uint64_t dword_idx = 0; dword_idx < num_dwords; ++dword_idx)
      __hip_atomic_store(base + base_dword + dword_idx, 0u, __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_AGENT);
  }
#endif  // __HIPCC__ || __CUDACC__
};

}  // namespace abce

#endif  // ABCE_RING_DEVICE_H_
