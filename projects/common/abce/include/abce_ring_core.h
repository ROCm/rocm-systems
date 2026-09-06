/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Accelerated Blit Copy Engine (ABCE) — shared SDMA ring reservation core.
//
// Implements the SDMA ring's reserve / wrap / in-order commit protocol as
// __host__ __device__ free functions over primitive pointers, so the host
// RingBuffer (abce_ring_host.h) and the device DeviceRing (abce_device.h) share
// one implementation. The atomics resolve to __hip_atomic_* under device
// compilation and __atomic_* on the host.
//
// The ring is single-consumer (the SDMA engine drains it in order) and
// multi-producer. A producer:
//   1. RingReserve(): advances a monotonic reserve cursor by a CAS. If a payload
//      would cross the ring end, it first reserves and publishes the wrap tail
//      as a separate NOP region, then retries the payload at offset 0.
//   2. writes its packets into the region (the host zeroes first for the
//      builders' zeroed-buffer contract; the device stores dwords directly).
//   3. RingPublish(): waits its turn on a monotonic commit cursor, invokes the
//      caller's publish step (advance the write pointer, ring the doorbell),
//      then advances the commit cursor. The write index thus only advances in
//      reservation order.

#ifndef ABCE_RING_CORE_H_
#define ABCE_RING_CORE_H_

#include <cstdint>

// Host+device qualifier (mirrors abce_builder.h; guarded so either may define it).
#ifndef ABCE_HD
#if defined(__HIPCC__) || defined(__CUDACC__)
#define ABCE_HD __host__ __device__
#else
#define ABCE_HD
#endif
#endif  // ABCE_HD

#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__)
#include <thread>  // std::this_thread::yield() for host spin-waits
#endif

namespace abce {

// ---------------------------------------------------------------------------
// Atomic + pause primitives (device: __hip_atomic_* / s_sleep; host: __atomic_*)
// ---------------------------------------------------------------------------

ABCE_HD inline uint64_t RingAtomicLoad(const uint64_t* p) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
  return __atomic_load_n(p, __ATOMIC_RELAXED);
#endif
}

ABCE_HD inline uint64_t RingAtomicLoadAcquire(const uint64_t* p) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
  return __atomic_load_n(p, __ATOMIC_ACQUIRE);
#endif
}

/// Load a hardware-updated word (engine read pointer): same relaxed/system read,
/// but the pointer is volatile because the engine writes it.
ABCE_HD inline uint64_t RingAtomicLoadHw(const volatile uint64_t* p) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __hip_atomic_load(const_cast<const uint64_t*>(p), __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_SYSTEM);
#else
  return __atomic_load_n(const_cast<const uint64_t*>(p), __ATOMIC_RELAXED);
#endif
}

ABCE_HD inline bool RingAtomicCas(uint64_t* p, uint64_t& expected, uint64_t desired) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __hip_atomic_compare_exchange_strong(p, &expected, desired, __ATOMIC_RELAXED,
                                              __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
  return __atomic_compare_exchange_n(p, &expected, desired, /*weak=*/false, __ATOMIC_RELAXED,
                                     __ATOMIC_RELAXED);
#endif
}

ABCE_HD inline void RingAtomicStore(uint64_t* p, uint64_t v) {
#if defined(__HIP_DEVICE_COMPILE__)
  __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
  __atomic_store_n(p, v, __ATOMIC_RELAXED);
#endif
}

ABCE_HD inline void RingAtomicStoreRelease(uint64_t* p, uint64_t v) {
#if defined(__HIP_DEVICE_COMPILE__)
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
  __atomic_store_n(p, v, __ATOMIC_RELEASE);
#endif
}

/// Back off inside a spin-wait: sleep a wave on device, yield the CPU on host.
ABCE_HD inline void RingPause() {
#if defined(__HIP_DEVICE_COMPILE__)
  __builtin_amdgcn_s_sleep(1);
#else
  std::this_thread::yield();
#endif
}

// ---------------------------------------------------------------------------
// Reserve / wrap / commit (shared by host + device)
// ---------------------------------------------------------------------------

enum class RingStatus : uint8_t {
  kSuccess,
  kInvalidArgument,
  kTooLarge,
};

struct RingReservation {
  uint64_t start = 0;
  uint64_t end = 0;

  ABCE_HD bool valid() const { return end > start; }
  ABCE_HD uint64_t bytes() const { return end - start; }
};

/// CPU/GPU-visible coordination state shared by every producer of one ring.
/// Keeping the control block exactly one cache line avoids false sharing with
/// unrelated runtime state and gives HIP a stable internal device layout.
struct alignas(64) SharedRingControl {
  uint64_t reserve_cursor = 0;
  uint64_t commit_cursor = 0;
  uint64_t max_write_index = 0;
};

static_assert(sizeof(SharedRingControl) == 64, "shared ring control must occupy one cache line");

/// Wrap a monotonic byte index into a ring offset.  @p size MUST be a power of 2.
ABCE_HD inline uint64_t RingWrap(uint64_t index, uint64_t size) { return index & (size - 1); }

/// True if writing up to @p upto (exclusive, monotonic) will not clobber packets
/// the engine has not drained yet.
ABCE_HD inline bool RingCanWriteUpto(const volatile uint64_t* read_ptr, uint64_t size,
                                     uint64_t upto) {
  return (upto - RingAtomicLoadHw(read_ptr)) < size;
}

/// Accessor-backed variant used by host queues whose public API exposes the
/// read index through a function rather than a directly accessible control
/// word.
ABCE_HD inline bool RingCanWriteUpto(uint64_t read_index, uint64_t size, uint64_t upto) {
  return (upto - read_index) < size;
}

/// Wrap padding (bytes to the ring end) a reservation of @p bytes needs if it
/// would otherwise straddle the physical end starting at monotonic @p start.
/// Pure function of (start, bytes, size) so producer and publisher agree without
/// sharing state.
ABCE_HD inline uint64_t RingPadBytes(uint64_t start, uint64_t bytes, uint64_t size) {
  const uint64_t offset = RingWrap(start, size);
  return (offset + bytes > size) ? (size - offset) : 0;
}

/// Shared reservation implementation. The read-index callable returns the
/// engine's monotonic byte index. When a payload would cross the physical ring
/// end, this function CAS-reserves only the wrap tail, asks @p publish_padding
/// to publish that zero/NOP region, then retries the payload at offset zero.
/// Keeping padding separate means every payload smaller than the ring can make
/// progress once the ring is empty, even when padding + payload exceeds the
/// ring size.
template <typename LoadReadIndexFn, typename PublishPaddingFn>
ABCE_HD inline RingStatus RingReserveImpl(uint64_t* reserve_cursor, uint64_t size, uint64_t bytes,
                                         RingReservation& reservation,
                                         LoadReadIndexFn load_read_index,
                                         PublishPaddingFn publish_padding) {
  if (!reserve_cursor || size == 0 || (size & (size - 1)) != 0 || bytes == 0)
    return RingStatus::kInvalidArgument;
  if (bytes >= size) return RingStatus::kTooLarge;

  while (true) {
    uint64_t current = RingAtomicLoad(reserve_cursor);
    const uint64_t pad_bytes = RingPadBytes(current, bytes, size);
    const uint64_t reservation_bytes = pad_bytes != 0 ? pad_bytes : bytes;
    const uint64_t next = current + reservation_bytes;
    if (RingCanWriteUpto(load_read_index(), size, next) &&
        RingAtomicCas(reserve_cursor, current, next)) {
      if (pad_bytes != 0) {
        RingReservation padding;
        padding.start = current;
        padding.end = next;
        const RingStatus padding_status = publish_padding(padding);
        if (padding_status != RingStatus::kSuccess) return padding_status;
        continue;
      }

      reservation.start = current;
      reservation.end = next;
      return RingStatus::kSuccess;
    }
    RingPause();
  }
}

/// Direct-pointer reservation path used by DeviceRing.
template <typename PublishPaddingFn>
ABCE_HD inline RingStatus RingReserve(uint64_t* reserve_cursor,
                                      const volatile uint64_t* read_ptr, uint64_t size,
                                      uint64_t bytes, RingReservation& reservation,
                                      PublishPaddingFn publish_padding) {
  if (!read_ptr) return RingStatus::kInvalidArgument;
  return RingReserveImpl(
      reserve_cursor, size, bytes, reservation,
      [read_ptr]() { return RingAtomicLoadHw(read_ptr); }, publish_padding);
}

/// Accessor-backed reservation path used by host queues whose public API keeps
/// the hardware read index opaque.
template <typename LoadReadIndexFn, typename PublishPaddingFn>
ABCE_HD inline RingStatus RingReserveWithReadIndex(uint64_t* reserve_cursor, uint64_t size,
                                                   uint64_t bytes, RingReservation& reservation,
                                                   LoadReadIndexFn load_read_index,
                                                   PublishPaddingFn publish_padding) {
  return RingReserveImpl(reserve_cursor, size, bytes, reservation, load_read_index,
                         publish_padding);
}

/// Publish [start, end) in reservation order: wait indefinitely until it is
/// this producer's turn (commit cursor reached @p start), invoke @p publish(end)
/// to advance the hardware write pointer + ring the doorbell, then release the
/// next producer by advancing the commit cursor to @p end. A commit cannot
/// safely time out after reservation because that would strand a permanent
/// unpublished hole in the ring.
template <typename PublishFn, typename WaitFn>
ABCE_HD inline RingStatus RingPublish(uint64_t* commit_cursor, const RingReservation& reservation,
                                      PublishFn publish, WaitFn wait) {
  if (!commit_cursor || !reservation.valid()) return RingStatus::kInvalidArgument;
  uint64_t observed_commit_index = RingAtomicLoadAcquire(commit_cursor);
  while (observed_commit_index != reservation.start) {
    wait(commit_cursor, observed_commit_index);
    observed_commit_index = RingAtomicLoadAcquire(commit_cursor);
  }
  publish(reservation.end);
  RingAtomicStoreRelease(commit_cursor, reservation.end);
  return RingStatus::kSuccess;
}

struct RingPauseWait {
  ABCE_HD void operator()(const uint64_t*, uint64_t) const { RingPause(); }
};

template <typename PublishFn>
ABCE_HD inline RingStatus RingPublish(uint64_t* commit_cursor, const RingReservation& reservation,
                                      PublishFn publish) {
  return RingPublish(commit_cursor, reservation, publish, RingPauseWait{});
}

}  // namespace abce

#endif  // ABCE_RING_CORE_H_
