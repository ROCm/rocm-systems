/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Accelerated Blit Copy Engine (ABCE) — host-side SDMA ring.
//
// The host wrapper around the shared reserve/commit protocol in
// abce_ring_core.h. See that header for the core; this file adds only the
// host-side concerns (zeroing, wptr/doorbell writes, the doorbell-notify
// callback).

#ifndef ABCE_RING_HOST_H_
#define ABCE_RING_HOST_H_

#include "abce_config.h"

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "abce_ring_core.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Queue-control callbacks
//===----------------------------------------------------------------------===//

typedef void (*abce_doorbell_notify_fn_t)(void* user_data, uint64_t new_write_index);
typedef void (*abce_queue_publish_fn_t)(void* user_data, volatile uint64_t* write_ptr,
                                        volatile uint64_t* doorbell, uint64_t new_write_index);
typedef uint64_t (*abce_queue_load_index_fn_t)(void* user_data);
typedef void (*abce_queue_publish_index_fn_t)(void* user_data, uint64_t new_write_index);
// Writes one region the engine will read: |src_bytes| from |src| (none for
// padding), then zeros up to |dst_bytes|.
typedef void (*abce_ring_write_fn_t)(void* user_data, char* dst, const void* src, size_t src_bytes,
                                     size_t dst_bytes);

// Queue-control operations for APIs that intentionally keep hardware control
// words opaque. All three callbacks must be supplied together.
typedef struct abce_ring_queue_ops_t {
  abce_queue_load_index_fn_t load_read_index;
  abce_queue_load_index_fn_t load_write_index;
  abce_queue_publish_index_fn_t publish_write_index;
  void* user_data;
} abce_ring_queue_ops_t;

#if ABCE_ENABLE_IMPLEMENTATION
static inline bool abce_ring_queue_ops_empty(const abce_ring_queue_ops_t* ops) {
  return ops->load_read_index == NULL && ops->load_write_index == NULL &&
         ops->publish_write_index == NULL && ops->user_data == NULL;
}
#else
extern bool abce_ring_queue_ops_empty(const abce_ring_queue_ops_t* ops);
#endif  // ABCE_ENABLE_IMPLEMENTATION

#if ABCE_ENABLE_IMPLEMENTATION
static inline bool abce_ring_queue_ops_complete(const abce_ring_queue_ops_t* ops) {
  return ops->load_read_index != NULL && ops->load_write_index != NULL &&
         ops->publish_write_index != NULL;
}
#else
extern bool abce_ring_queue_ops_complete(const abce_ring_queue_ops_t* ops);
#endif  // ABCE_ENABLE_IMPLEMENTATION

//===----------------------------------------------------------------------===//
// abce_ring_config_t
//===----------------------------------------------------------------------===//

// Static configuration for one SDMA ring buffer.
//
// The caller supplies exactly one queue-control backend: either queue_ops, or
// the directly mapped write_ptr/read_ptr/doorbell fields. The ring never
// allocates or frees the ring memory or its queue-control state.
typedef struct abce_ring_config_t {
  // Base address of the ring buffer (queue command memory).
  char* base;

  // Ring size in bytes. MUST be a power of two.
  size_t size;

  // Monotonic write index (in bytes) consumed by the engine. Written by
  // release/pad in reservation order. Device-visible.
  volatile uint64_t* write_ptr;

  // Monotonic read index (in bytes) advanced by the engine as it drains
  // packets. Read-only from the host.
  volatile uint64_t* read_ptr;

  // Doorbell word the engine polls; written with the new write index.
  volatile uint64_t* doorbell;

  // Opaque queue-control backend. Use this instead of the three directly mapped
  // control pointers when queue indices and publication are APIs.
  abce_ring_queue_ops_t queue_ops;

  // Pad every submission up to at least this many bytes (0 = no minimum). The
  // pad tail is left as zero DWORDs, which the engine interprets as NOPs.
  size_t min_submission_size;

  // When true, additionally pad the submission size up to a 64-byte multiple
  // (required by some virtualized/DXG paths).
  bool align64;

  // Optional efficient wait used while an out-of-order producer waits for the
  // commit cursor. ROCr integrations can use this hook for mwaitx. The default
  // uses abce_ring_pause() (a thread yield on the host).
  abce_ring_commit_wait_fn_t commit_wait;
  void* commit_wait_user_data;

  // Optional platform-specific write-pointer/doorbell publication routine. When
  // null, the ring uses volatile writes with a release fence.
  abce_queue_publish_fn_t publish;
  void* publish_user_data;

  // Optional writer for ring memory. When set, acquire leaves the region as it
  // is and every region is written once, whole, through this hook, which must
  // leave its stores ordered ahead of a later doorbell store. A ring the CPU
  // reaches through a write-combined BAR uses it to write full lines (e.g.
  // MOVDIR64B) rather than zeroing a region and then copying into it.
  abce_ring_write_fn_t write;
  void* write_user_data;

  // The CPU writes ring memory through a BAR (device memory), while the doorbell
  // reaches the engine by another path and can arrive first. An SDMA engine
  // executes whatever lies before the write index, so it would run stale bytes:
  // NOPs that skip the copy, or after a wrap, old packets. When set, every
  // publish of written ring memory first reads back the last line written; the
  // read cannot pass the stores. An HSA client sets it for a ring in the agent's
  // memory from HSA_AMD_AGENT_INFO_PCIE_STORES_NEED_FLUSH.
  bool readback_before_doorbell;
} abce_ring_config_t;

//===----------------------------------------------------------------------===//
// abce_ring_t
//===----------------------------------------------------------------------===//

// Single-consumer (SDMA engine) ring with multi-producer host writers.
//
// A thin host wrapper over the shared reserve/commit protocol in
// abce_ring_core.h, adding only the host-side concerns: zeroing acquired
// regions for the builders' zeroed-buffer contract, the hardware wptr/doorbell
// writes, and the optional doorbell notify callback.
//
// The SDMA packet processor consumes the ring strictly in order and does not
// tolerate the write index moving out of reservation order, so:
//   1. abce_ring_acquire(n): reserve a contiguous n-byte region. If it would
//      cross the ring end, reserve and publish the wrap tail as NOPs first, then
//      retry the payload at offset zero. Zero the payload and return it.
//   2. The caller writes packets into the region. Zeroing on acquire both
//      satisfies the builders' zeroed-buffer contract and turns any unused pad
//      tail into NOPs.
//   3. abce_ring_release(reservation): publish the region. Blocks until every
//      earlier reservation has been published, then advances wptr + rings the
//      doorbell.
//
// Concurrency model (see abce_ring_core.h):
//   - The reserve cursor is advanced lock-free via a single CAS, so producers
//     write disjoint regions concurrently — there is no reservation lock.
//   - The commit cursor enforces in-order publication: release for region R
//     spins until it is R's turn, so the doorbell is monotone even when
//     producers finish writing out of order.
//
// The ring must remain at a stable address: the core's callbacks take it as
// their context. It is neither copyable nor movable in any meaningful sense.
typedef struct abce_ring_t {
  abce_ring_config_t config;
  abce_doorbell_notify_fn_t notify;
  void* notify_user_data;

  // Host threads are the only producers, so the cursors live here.
  abce_ring_control_t control;
} abce_ring_t;

// Orders every earlier store ahead of the doorbell, write-combined ones
// included. A release fence does not do that on x86, where it is only a
// compiler barrier, and packet stores into a VRAM ring reached through the BAR
// are write-combined: without this they can still sit in the CPU's WC buffers
// when the doorbell lands, and the engine fetches the old ring contents.
#if ABCE_ENABLE_IMPLEMENTATION
static inline void abce_ring_host_store_fence(void) {
#if ABCE_COMPILER_MSVC
  _mm_sfence();
#elif defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_sfence();
#else
  abce_host_fence_release();
#endif  // __x86_64__ || __i386__
}
#endif  // ABCE_ENABLE_IMPLEMENTATION

// Wraps a monotonic index into a ring offset.
#if ABCE_ENABLE_IMPLEMENTATION
static inline uint32_t abce_ring_wrap_index(const abce_ring_t* ring, uint64_t index) {
  return (uint32_t)abce_ring_wrap(index, ring->config.size);
}
#else
extern uint32_t abce_ring_wrap_index(const abce_ring_t* ring, uint64_t index);
#endif  // ABCE_ENABLE_IMPLEMENTATION

#if ABCE_ENABLE_IMPLEMENTATION
static inline uint64_t abce_ring_load_read_index(const abce_ring_t* ring) {
  if (abce_ring_queue_ops_complete(&ring->config.queue_ops))
    return ring->config.queue_ops.load_read_index(ring->config.queue_ops.user_data);
  return abce_ring_atomic_load_hw(ring->config.read_ptr);
}
#else
extern uint64_t abce_ring_load_read_index(const abce_ring_t* ring);
#endif  // ABCE_ENABLE_IMPLEMENTATION

#if ABCE_ENABLE_IMPLEMENTATION
static inline uint64_t abce_ring_load_write_index(const abce_ring_t* ring) {
  if (abce_ring_queue_ops_complete(&ring->config.queue_ops))
    return ring->config.queue_ops.load_write_index(ring->config.queue_ops.user_data);
  return abce_ring_atomic_load_hw(ring->config.write_ptr);
}
#else
extern uint64_t abce_ring_load_write_index(const abce_ring_t* ring);
#endif  // ABCE_ENABLE_IMPLEMENTATION

#if ABCE_ENABLE_IMPLEMENTATION
static inline void abce_ring_publish_write_index(const abce_ring_t* ring,
                                                 uint64_t new_write_index) {
  if (abce_ring_queue_ops_complete(&ring->config.queue_ops)) {
    ring->config.queue_ops.publish_write_index(ring->config.queue_ops.user_data, new_write_index);
  } else if (ring->config.publish) {
    ring->config.publish(ring->config.publish_user_data, ring->config.write_ptr,
                         ring->config.doorbell, new_write_index);
  } else {
    // The write pointer before the doorbell; every release path has already
    // ordered the packets ahead of both.
    *ring->config.write_ptr = new_write_index;
    abce_host_fence_release();
    *ring->config.doorbell = new_write_index;
  }
}
#else
extern void abce_ring_publish_write_index(const abce_ring_t* ring,
                                                 uint64_t new_write_index);
#endif  // ABCE_ENABLE_IMPLEMENTATION

// Initializes |out_ring| against |config|. |notify| is optional.
#if ABCE_ENABLE_IMPLEMENTATION
static inline void abce_ring_initialize(const abce_ring_config_t* config,
                                        abce_doorbell_notify_fn_t notify, void* notify_user_data,
                                        abce_ring_t* out_ring) {
  assert(config->base != NULL);
  const bool has_raw_backend =
      config->write_ptr != NULL || config->read_ptr != NULL || config->doorbell != NULL;
  const bool raw_backend_complete =
      config->write_ptr != NULL && config->read_ptr != NULL && config->doorbell != NULL;
  const bool has_ops_backend = !abce_ring_queue_ops_empty(&config->queue_ops);
  (void)has_raw_backend;
  (void)raw_backend_complete;
  (void)has_ops_backend;
  assert(((raw_backend_complete && !has_ops_backend) ||
          (abce_ring_queue_ops_complete(&config->queue_ops) && !has_raw_backend)) &&
         "configure exactly one complete queue-control backend");
  assert(!has_ops_backend || config->publish == NULL);
  assert(abce_is_power_of_two(config->size) && "ring size must be a power of two");

  memset(out_ring, 0, sizeof(*out_ring));
  out_ring->config = *config;
  out_ring->notify = notify;
  out_ring->notify_user_data = notify_user_data;
  // Seed the reserve/commit cursors from the queue's current monotonic write
  // pointer. Attaching to an already-in-use queue and starting from 0 would let
  // the first acquire hand out bytes the engine still owns, corrupting the ring,
  // so start the cursors at the queue's write pointer.
  const uint64_t initial_write_index = abce_ring_load_write_index(out_ring);
  abce_ring_cursor_store(&out_ring->control.reserve_cursor, initial_write_index);
  abce_ring_cursor_store(&out_ring->control.commit_cursor, initial_write_index);
  abce_ring_cursor_store(&out_ring->control.max_write_index, initial_write_index);
}
#else
extern void abce_ring_initialize(const abce_ring_config_t* config,
                                        abce_doorbell_notify_fn_t notify, void* notify_user_data,
                                        abce_ring_t* out_ring);
#endif  // ABCE_ENABLE_IMPLEMENTATION

// Total reservation size for a payload of |payload_bytes|, after applying the
// configured minimum submission size and optional 64B rounding. Pass the result
// to both acquire and release.
#if ABCE_ENABLE_IMPLEMENTATION
static inline abce_status_t abce_ring_padded_size(const abce_ring_t* ring, uint32_t payload_bytes,
                                                  uint32_t* out_reserve_bytes) {
  if (payload_bytes == 0) return ABCE_STATUS_INVALID_ARGUMENT;
  uint64_t padded = payload_bytes;
  if (padded < ring->config.min_submission_size) padded = ring->config.min_submission_size;
  if (ring->config.align64) padded = abce_align_up(padded, 64);
  if (padded > UINT32_MAX || padded >= ring->config.size) return ABCE_STATUS_OUT_OF_RANGE;
  *out_reserve_bytes = (uint32_t)padded;
  return ABCE_STATUS_OK;
}
#else
extern abce_status_t abce_ring_padded_size(const abce_ring_t* ring, uint32_t payload_bytes,
                                                  uint32_t* out_reserve_bytes);
#endif  // ABCE_ENABLE_IMPLEMENTATION

#if ABCE_ENABLE_IMPLEMENTATION
static inline abce_status_t abce_ring_release_ordered(abce_ring_t* ring,
                                                      const abce_ring_reservation_t* reservation);
#else
extern abce_status_t abce_ring_release_ordered(abce_ring_t* ring,
                                                      const abce_ring_reservation_t* reservation);
#endif  // ABCE_ENABLE_IMPLEMENTATION

//===----------------------------------------------------------------------===//
// Core callbacks (the C++ version passed these as lambdas)
//===----------------------------------------------------------------------===//

#if ABCE_ENABLE_IMPLEMENTATION
static inline uint64_t abce_ring_load_read_index_callback(void* user_data) {
  return abce_ring_load_read_index((const abce_ring_t*)user_data);
}
#endif  // ABCE_ENABLE_IMPLEMENTATION

// Fills an acquired region with |src_bytes| from |src| and zeros after it. A
// region from abce_ring_acquire is already zero unless the ring has a writer.
#if ABCE_ENABLE_IMPLEMENTATION
static inline void abce_ring_fill(const abce_ring_t* ring, char* buffer, const void* src,
                                  size_t src_bytes, size_t reserve_bytes) {
  if (ring->config.write) {
    ring->config.write(ring->config.write_user_data, buffer, src, src_bytes, reserve_bytes);
  } else if (src_bytes != 0) {
    memcpy(buffer, src, src_bytes);
  }
}
#else
extern void abce_ring_fill(const abce_ring_t* ring, char* buffer, const void* src,
                                  size_t src_bytes, size_t reserve_bytes);
#endif  // ABCE_ENABLE_IMPLEMENTATION

// Waits for every earlier store to reach |byte|'s device by reading it back
// (see abce_ring_config_t::readback_before_doorbell). A read of any width
// cannot pass the stores before it, and one byte stays inside the smallest
// region at any alignment: a reservation can be a single 4-byte NOP.
#if ABCE_ENABLE_IMPLEMENTATION
static inline void abce_ring_read_back(const void* byte) {
  abce_host_fence_seq_cst();
  (void)*(const volatile uint8_t*)byte;
}
#endif  // ABCE_ENABLE_IMPLEMENTATION

// The last byte of a filled region, the one abce_ring_read_back reads.
#if ABCE_ENABLE_IMPLEMENTATION
static inline const char* abce_ring_region_last_byte(const char* buffer, size_t bytes) {
  return buffer + bytes - 1;
}
#endif  // ABCE_ENABLE_IMPLEMENTATION

// Orders the stores that filled |region| ahead of its doorbell: a readback
// where the ring sets readback_before_doorbell, otherwise a store fence unless
// a ring writer, which orders its own stores, filled it.
#if ABCE_ENABLE_IMPLEMENTATION
static inline void abce_ring_order_region(const abce_ring_t* ring,
                                          const abce_ring_reservation_t* region,
                                          bool filled_by_writer) {
  if (ring->config.readback_before_doorbell) {
    const char* buffer = ring->config.base + abce_ring_wrap_index(ring, region->start);
    abce_ring_read_back(
        abce_ring_region_last_byte(buffer, (size_t)abce_ring_reservation_bytes(region)));
  } else if (!filled_by_writer) {
    abce_ring_host_store_fence();
  }
}
#endif  // ABCE_ENABLE_IMPLEMENTATION

#if ABCE_ENABLE_IMPLEMENTATION
static inline abce_status_t abce_ring_publish_padding_callback(
    void* user_data, const abce_ring_reservation_t* padding) {
  abce_ring_t* ring = (abce_ring_t*)user_data;
  char* buffer = ring->config.base + abce_ring_wrap_index(ring, padding->start);
  const size_t bytes = (size_t)abce_ring_reservation_bytes(padding);
  if (ring->config.write) {
    ring->config.write(ring->config.write_user_data, buffer, NULL, 0, bytes);
  } else {
    // Padding can run to the end of the ring, large enough for memset to switch
    // to non-temporal stores.
    memset(buffer, 0, bytes);
  }
  abce_ring_order_region(ring, padding, ring->config.write != NULL);
  return abce_ring_release_ordered(ring, padding);
}
#endif  // ABCE_ENABLE_IMPLEMENTATION

#if ABCE_ENABLE_IMPLEMENTATION
static inline void abce_ring_publish_callback(void* user_data, uint64_t new_write_index) {
  abce_ring_t* ring = (abce_ring_t*)user_data;
  abce_ring_publish_write_index(ring, new_write_index);
  abce_ring_cursor_store_release(&ring->control.max_write_index, new_write_index);
  if (ring->notify) ring->notify(ring->notify_user_data, new_write_index);
}
#endif  // ABCE_ENABLE_IMPLEMENTATION

#if ABCE_ENABLE_IMPLEMENTATION
static inline void abce_ring_commit_wait_callback(void* user_data, const uint64_t* commit_cursor,
                                                  uint64_t observed_commit_index) {
  abce_ring_t* ring = (abce_ring_t*)user_data;
  if (ring->config.commit_wait) {
    ring->config.commit_wait(ring->config.commit_wait_user_data, commit_cursor,
                             observed_commit_index);
  } else {
    abce_ring_pause();
  }
}
#endif  // ABCE_ENABLE_IMPLEMENTATION

//===----------------------------------------------------------------------===//
// Acquire / release / drain
//===----------------------------------------------------------------------===//

// Reserves a contiguous, zeroed |reserve_bytes| region of the ring.
//
// Blocks until the engine has freed enough space. On success sets
// |out_reservation|, which must be passed verbatim to abce_ring_release, and
// |out_buffer|, a zeroed pointer into the ring. Fails only if |reserve_bytes|
// is at or above the ring size (the request can never fit).
#if ABCE_ENABLE_IMPLEMENTATION
static inline abce_status_t abce_ring_acquire(abce_ring_t* ring, uint32_t reserve_bytes,
                                              abce_ring_reservation_t* out_reservation,
                                              char** out_buffer) {
  *out_buffer = NULL;

  abce_ring_producer_t producer;
  producer.reserve_cursor = &ring->control.reserve_cursor;
  // The engine writes the read pointer, so loading it costs a cache miss (and,
  // through the HSA adapter, two runtime calls) on every submission.
  producer.cached_read_index = &ring->control.cached_read_index;
  producer.size = ring->config.size;
  producer.load_read_index = abce_ring_load_read_index_callback;
  producer.publish_padding = abce_ring_publish_padding_callback;
  producer.user_data = ring;

  const abce_status_t status = abce_ring_reserve(&producer, reserve_bytes, out_reservation);
  if (!abce_status_is_ok(status)) return status;

  // Wrap padding, when needed, was already published as a separate NOP
  // reservation. This reservation contains only the contiguous payload.
  char* buffer = ring->config.base + abce_ring_wrap_index(ring, out_reservation->start);
  if (!ring->config.write) memset(buffer, 0, reserve_bytes);
  *out_buffer = buffer;
  return ABCE_STATUS_OK;
}
#else
extern abce_status_t abce_ring_acquire(abce_ring_t* ring, uint32_t reserve_bytes,
                                              abce_ring_reservation_t* out_reservation,
                                              char** out_buffer);
#endif  // ABCE_ENABLE_IMPLEMENTATION

// abce_ring_release for a caller that has already ordered every store the
// engine reads ahead of the doorbell: submit, which fences once for a whole
// batch, and padding and cancel, which order their own fill.
#if ABCE_ENABLE_IMPLEMENTATION
static inline abce_status_t abce_ring_release_ordered(abce_ring_t* ring,
                                                      const abce_ring_reservation_t* reservation) {
  abce_ring_publisher_t publisher;
  publisher.commit_cursor = &ring->control.commit_cursor;
  publisher.publish = abce_ring_publish_callback;
  publisher.wait = abce_ring_commit_wait_callback;
  publisher.user_data = ring;
  return abce_ring_publish(&publisher, reservation);
}
#else
extern abce_status_t abce_ring_release_ordered(abce_ring_t* ring,
                                                      const abce_ring_reservation_t* reservation);
#endif  // ABCE_ENABLE_IMPLEMENTATION

// Publishes a previously acquired region and rings the doorbell. Blocks until
// all earlier reservations have been published so the write index advances
// strictly in reservation order.
//
// Orders the caller's writes to the region ahead of the doorbell first,
// including the non-temporal stores a large memcpy can issue on x86.
#if ABCE_ENABLE_IMPLEMENTATION
static inline abce_status_t abce_ring_release(abce_ring_t* ring,
                                              const abce_ring_reservation_t* reservation) {
  abce_ring_order_region(ring, reservation, /*filled_by_writer=*/false);
  return abce_ring_release_ordered(ring, reservation);
}
#else
extern abce_status_t abce_ring_release(abce_ring_t* ring,
                                              const abce_ring_reservation_t* reservation);
#endif  // ABCE_ENABLE_IMPLEMENTATION

// Publishes an acquired reservation as NOPs. Without a writer the NOPs are
// abce_ring_acquire's zeroing.
#if ABCE_ENABLE_IMPLEMENTATION
static inline abce_status_t abce_ring_cancel(abce_ring_t* ring,
                                             const abce_ring_reservation_t* reservation) {
  if (ring->config.write) {
    char* buffer = ring->config.base + abce_ring_wrap_index(ring, reservation->start);
    abce_ring_fill(ring, buffer, NULL, 0, (size_t)abce_ring_reservation_bytes(reservation));
  }
  abce_ring_order_region(ring, reservation, ring->config.write != NULL);
  return abce_ring_release_ordered(ring, reservation);
}
#else
extern abce_status_t abce_ring_cancel(abce_ring_t* ring,
                                             const abce_ring_reservation_t* reservation);
#endif  // ABCE_ENABLE_IMPLEMENTATION

// Waits until the engine has consumed everything published when this call
// snapshots the high-water mark.
#if ABCE_ENABLE_IMPLEMENTATION
static inline abce_status_t abce_ring_drain(const abce_ring_t* ring) {
  const uint64_t target = abce_ring_cursor_load_acquire(&ring->control.max_write_index);
  while (abce_ring_load_read_index(ring) < target) {
    abce_ring_pause();
  }
  return ABCE_STATUS_OK;
}
#else
extern abce_status_t abce_ring_drain(const abce_ring_t* ring);
#endif  // ABCE_ENABLE_IMPLEMENTATION

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ABCE_RING_HOST_H_
