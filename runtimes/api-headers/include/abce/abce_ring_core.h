/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Accelerated Blit Copy Engine (ABCE) — shared SDMA ring reservation core.
//
// Implements the SDMA ring's reserve / wrap / in-order commit protocol as free
// functions over primitive pointers, so the host ring (abce_ring_host.h) and a
// device-side producer share one implementation. The atomics resolve to
// __hip_atomic_* under device compilation and __atomic_* on the host.
//
// Where the C++ version took the read-index and publish steps as template
// parameters instantiated on lambdas, this takes them as function pointers
// bundled with one context pointer: abce_ring_producer_t for the reserve side
// and abce_ring_publisher_t for the commit side. Both callbacks of a producer
// share a context because in every caller they are the same object.
//
// The ring is single-consumer (the SDMA engine drains it in order) and
// multi-producer. A producer:
//   1. abce_ring_reserve(): advances a monotonic reserve cursor by a CAS. If a
//      payload would cross the ring end, it first reserves and publishes the
//      wrap tail as a separate NOP region, then retries the payload at offset 0.
//   2. writes its packets into the region (the host zeroes first for the
//      builders' zeroed-buffer contract; a device producer stores dwords
//      directly).
//   3. abce_ring_publish(): waits its turn on a monotonic commit cursor,
//      invokes the caller's publish step (advance the write pointer, ring the
//      doorbell), then advances the commit cursor. The write index thus only
//      advances in reservation order.

#ifndef ABCE_RING_CORE_H_
#define ABCE_RING_CORE_H_

#include <stdbool.h>
#include <stdint.h>

#include "abce_config.h"

#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__)
#include <sched.h>  // sched_yield() for host spin-waits
#endif

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Atomic + pause primitives
//
// Device: __hip_atomic_* / s_sleep. Host: __atomic_* / sched_yield.
//===----------------------------------------------------------------------===//

static inline uint64_t abce_ring_atomic_load(const uint64_t* ptr) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
  return __atomic_load_n(ptr, __ATOMIC_RELAXED);
#endif  // __HIP_DEVICE_COMPILE__
}

static inline uint64_t abce_ring_atomic_load_acquire(const uint64_t* ptr) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __hip_atomic_load(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
  return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
#endif  // __HIP_DEVICE_COMPILE__
}

// Loads a hardware-updated word (the engine read pointer): the same relaxed,
// system-scope read, but the pointer is volatile because the engine writes it.
static inline uint64_t abce_ring_atomic_load_hw(const volatile uint64_t* ptr) {
  const uint64_t* stable_ptr = (const uint64_t*)(uintptr_t)ptr;
#if defined(__HIP_DEVICE_COMPILE__)
  return __hip_atomic_load(stable_ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
  return __atomic_load_n(stable_ptr, __ATOMIC_RELAXED);
#endif  // __HIP_DEVICE_COMPILE__
}

static inline bool abce_ring_atomic_cas(uint64_t* ptr, uint64_t* expected, uint64_t desired) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __hip_atomic_compare_exchange_strong(ptr, expected, desired, __ATOMIC_RELAXED,
                                              __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
  return __atomic_compare_exchange_n(ptr, expected, desired, /*weak=*/false, __ATOMIC_RELAXED,
                                     __ATOMIC_RELAXED);
#endif  // __HIP_DEVICE_COMPILE__
}

static inline void abce_ring_atomic_store(uint64_t* ptr, uint64_t value) {
#if defined(__HIP_DEVICE_COMPILE__)
  __hip_atomic_store(ptr, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
  __atomic_store_n(ptr, value, __ATOMIC_RELAXED);
#endif  // __HIP_DEVICE_COMPILE__
}

static inline void abce_ring_atomic_store_release(uint64_t* ptr, uint64_t value) {
#if defined(__HIP_DEVICE_COMPILE__)
  __hip_atomic_store(ptr, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
  __atomic_store_n(ptr, value, __ATOMIC_RELEASE);
#endif  // __HIP_DEVICE_COMPILE__
}

// Backs off inside a spin-wait: sleeps a wave on device, yields the CPU on host.
static inline void abce_ring_pause(void) {
#if defined(__HIP_DEVICE_COMPILE__)
  __builtin_amdgcn_s_sleep(1);
#else
  sched_yield();
#endif  // __HIP_DEVICE_COMPILE__
}

//===----------------------------------------------------------------------===//
// abce_ring_reservation_t / abce_shared_ring_control_t
//===----------------------------------------------------------------------===//

typedef struct abce_ring_reservation_t {
  uint64_t start;
  uint64_t end;
} abce_ring_reservation_t;

static inline bool abce_ring_reservation_is_valid(const abce_ring_reservation_t* reservation) {
  return reservation->end > reservation->start;
}

static inline uint64_t abce_ring_reservation_bytes(const abce_ring_reservation_t* reservation) {
  return reservation->end - reservation->start;
}

// CPU/GPU-visible coordination state shared by every producer of one ring.
// Keeping the control block exactly one cache line avoids false sharing with
// unrelated runtime state and gives a device view a stable internal layout.
// The alignment rides on the first member because C does not accept _Alignas
// between `struct` and the tag; it still gives the whole struct 64-byte
// alignment, which the size assertion below confirms.
typedef struct abce_shared_ring_control_t {
  ABCE_ALIGNAS(64) uint64_t reserve_cursor;
  uint64_t commit_cursor;
  uint64_t max_write_index;
} abce_shared_ring_control_t;

ABCE_STATIC_ASSERT(sizeof(abce_shared_ring_control_t) == 64,
                   "shared ring control must occupy one cache line");

//===----------------------------------------------------------------------===//
// Producer / publisher callbacks
//===----------------------------------------------------------------------===//

// Returns the engine's monotonic read index in bytes. Rings whose read pointer
// is a directly mapped word implement this as a call to
// abce_ring_atomic_load_hw().
typedef uint64_t (*abce_ring_load_read_index_fn_t)(void* user_data);

// Publishes an already-zeroed wrap-tail region as NOPs.
typedef abce_status_t (*abce_ring_publish_padding_fn_t)(
    void* user_data, const abce_ring_reservation_t* padding);

// Advances the hardware write pointer to |new_write_index| and rings the
// doorbell.
typedef void (*abce_ring_publish_fn_t)(void* user_data, uint64_t new_write_index);

// Optional efficient wait while an out-of-order producer waits its turn on the
// commit cursor. Null uses abce_ring_pause().
typedef void (*abce_ring_commit_wait_fn_t)(void* user_data, const uint64_t* commit_cursor,
                                           uint64_t observed_commit_index);

// The reserve side of one ring.
typedef struct abce_ring_producer_t {
  uint64_t* reserve_cursor;
  uint64_t size;  // ring size in bytes; MUST be a power of two.
  abce_ring_load_read_index_fn_t load_read_index;
  abce_ring_publish_padding_fn_t publish_padding;
  void* user_data;
} abce_ring_producer_t;

// The commit side of one ring.
typedef struct abce_ring_publisher_t {
  uint64_t* commit_cursor;
  abce_ring_publish_fn_t publish;
  abce_ring_commit_wait_fn_t wait;  // optional.
  void* user_data;
} abce_ring_publisher_t;

//===----------------------------------------------------------------------===//
// Reserve / wrap / commit
//===----------------------------------------------------------------------===//

// Wraps a monotonic byte index into a ring offset. |size| MUST be a power of 2.
static inline uint64_t abce_ring_wrap(uint64_t index, uint64_t size) {
  return index & (size - 1);
}

// True if writing up to |upto| (exclusive, monotonic) will not clobber packets
// the engine has not drained yet.
static inline bool abce_ring_can_write_upto(uint64_t read_index, uint64_t size, uint64_t upto) {
  return (upto - read_index) < size;
}

// Wrap padding (bytes to the ring end) a reservation of |bytes| needs if it
// would otherwise straddle the physical end starting at monotonic |start|. A
// pure function of (start, bytes, size) so producer and publisher agree without
// sharing state.
static inline uint64_t abce_ring_pad_bytes(uint64_t start, uint64_t bytes, uint64_t size) {
  const uint64_t offset = abce_ring_wrap(start, size);
  return (offset + bytes > size) ? (size - offset) : 0;
}

// Reserves |bytes| of contiguous ring space.
//
// When a payload would cross the physical ring end, this CAS-reserves only the
// wrap tail, asks the producer's publish_padding to publish that zero/NOP
// region, then retries the payload at offset zero. Keeping padding separate
// means every payload smaller than the ring can make progress once the ring is
// empty, even when padding + payload exceeds the ring size.
//
// Blocks until the engine has freed enough space. Fails only when the request
// can never fit.
static inline abce_status_t abce_ring_reserve(const abce_ring_producer_t* producer, uint64_t bytes,
                                              abce_ring_reservation_t* out_reservation) {
  if (!producer || !producer->reserve_cursor || !producer->load_read_index ||
      !producer->publish_padding || !out_reservation)
    return ABCE_STATUS_INVALID_ARGUMENT;
  const uint64_t size = producer->size;
  if (!abce_is_power_of_two(size) || bytes == 0) return ABCE_STATUS_INVALID_ARGUMENT;
  if (bytes >= size) return ABCE_STATUS_OUT_OF_RANGE;

  for (;;) {
    uint64_t current = abce_ring_atomic_load(producer->reserve_cursor);
    const uint64_t pad_bytes = abce_ring_pad_bytes(current, bytes, size);
    const uint64_t reservation_bytes = pad_bytes != 0 ? pad_bytes : bytes;
    const uint64_t next = current + reservation_bytes;
    if (abce_ring_can_write_upto(producer->load_read_index(producer->user_data), size, next) &&
        abce_ring_atomic_cas(producer->reserve_cursor, &current, next)) {
      if (pad_bytes != 0) {
        abce_ring_reservation_t padding;
        padding.start = current;
        padding.end = next;
        const abce_status_t status =
            producer->publish_padding(producer->user_data, &padding);
        if (!abce_status_is_ok(status)) return status;
        continue;
      }
      out_reservation->start = current;
      out_reservation->end = next;
      return ABCE_STATUS_OK;
    }
    abce_ring_pause();
  }
}

// Publishes [start, end) in reservation order: waits indefinitely until it is
// this producer's turn (the commit cursor reached |reservation->start|),
// invokes publish() to advance the hardware write pointer and ring the
// doorbell, then releases the next producer by advancing the commit cursor to
// |reservation->end|.
//
// A commit cannot safely time out after reservation because that would strand a
// permanent unpublished hole in the ring.
static inline abce_status_t abce_ring_publish(const abce_ring_publisher_t* publisher,
                                              const abce_ring_reservation_t* reservation) {
  if (!publisher || !publisher->commit_cursor || !publisher->publish || !reservation ||
      !abce_ring_reservation_is_valid(reservation))
    return ABCE_STATUS_INVALID_ARGUMENT;

  uint64_t observed_commit_index = abce_ring_atomic_load_acquire(publisher->commit_cursor);
  while (observed_commit_index != reservation->start) {
    if (publisher->wait) {
      publisher->wait(publisher->user_data, publisher->commit_cursor, observed_commit_index);
    } else {
      abce_ring_pause();
    }
    observed_commit_index = abce_ring_atomic_load_acquire(publisher->commit_cursor);
  }
  publisher->publish(publisher->user_data, reservation->end);
  abce_ring_atomic_store_release(publisher->commit_cursor, reservation->end);
  return ABCE_STATUS_OK;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ABCE_RING_CORE_H_
