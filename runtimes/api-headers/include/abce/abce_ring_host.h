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

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "abce_config.h"
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

// Queue-control operations for APIs that intentionally keep hardware control
// words opaque. All three callbacks must be supplied together.
typedef struct abce_ring_queue_ops_t {
  abce_queue_load_index_fn_t load_read_index;
  abce_queue_load_index_fn_t load_write_index;
  abce_queue_publish_index_fn_t publish_write_index;
  void* user_data;
} abce_ring_queue_ops_t;

static inline bool abce_ring_queue_ops_empty(const abce_ring_queue_ops_t* ops) {
  return ops->load_read_index == NULL && ops->load_write_index == NULL &&
         ops->publish_write_index == NULL && ops->user_data == NULL;
}

static inline bool abce_ring_queue_ops_complete(const abce_ring_queue_ops_t* ops) {
  return ops->load_read_index != NULL && ops->load_write_index != NULL &&
         ops->publish_write_index != NULL;
}

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

  // Optional CPU/GPU-visible producer state. When null, the ring uses private
  // host-only state. Shared host/device submission requires this.
  abce_shared_ring_control_t* shared_control;

  // Pad every submission up to at least this many bytes (0 = no minimum). The
  // pad tail is left as zero DWORDs, which the engine interprets as NOPs.
  size_t min_submission_size;

  // When true, additionally pad the submission size up to a 64-byte multiple
  // (required by some virtualized/DXG paths).
  bool align64;

  // Optional efficient wait used while an out-of-order producer waits for the
  // commit cursor. ROCr integrations can use this hook for mwaitx. The default
  // uses abce_ring_pause() (sched_yield on the host).
  abce_ring_commit_wait_fn_t commit_wait;
  void* commit_wait_user_data;

  // Optional platform-specific write-pointer/doorbell publication routine. When
  // null, the ring uses volatile writes with a release fence.
  abce_queue_publish_fn_t publish;
  void* publish_user_data;
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

  abce_shared_ring_control_t local_control;
  abce_shared_ring_control_t* control;
} abce_ring_t;

// Orders every earlier store ahead of the doorbell, write-combined ones
// included. A release fence does not do that on x86, where it is only a
// compiler barrier, and packet stores into a VRAM ring reached through the BAR
// are write-combined: without this they can still sit in the CPU's WC buffers
// when the doorbell lands, and the engine fetches the old ring contents.
static inline void abce_ring_host_store_fence(void) {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_sfence();
#else
  __atomic_thread_fence(__ATOMIC_RELEASE);
#endif  // __x86_64__ || __i386__
}

// Wraps a monotonic index into a ring offset.
static inline uint32_t abce_ring_wrap_index(const abce_ring_t* ring, uint64_t index) {
  return (uint32_t)abce_ring_wrap(index, ring->config.size);
}

static inline uint64_t abce_ring_load_read_index(const abce_ring_t* ring) {
  if (abce_ring_queue_ops_complete(&ring->config.queue_ops))
    return ring->config.queue_ops.load_read_index(ring->config.queue_ops.user_data);
  return abce_ring_atomic_load_hw(ring->config.read_ptr);
}

static inline uint64_t abce_ring_load_write_index(const abce_ring_t* ring) {
  if (abce_ring_queue_ops_complete(&ring->config.queue_ops))
    return ring->config.queue_ops.load_write_index(ring->config.queue_ops.user_data);
  return abce_ring_atomic_load_hw(ring->config.write_ptr);
}

static inline void abce_ring_publish_write_index(const abce_ring_t* ring,
                                                 uint64_t new_write_index) {
  if (abce_ring_queue_ops_complete(&ring->config.queue_ops)) {
    ring->config.queue_ops.publish_write_index(ring->config.queue_ops.user_data, new_write_index);
  } else if (ring->config.publish) {
    ring->config.publish(ring->config.publish_user_data, ring->config.write_ptr,
                         ring->config.doorbell, new_write_index);
  } else {
    // Packets before the write pointer, which an engine that polls it can act
    // on without the doorbell.
    abce_ring_host_store_fence();
    *ring->config.write_ptr = new_write_index;
    *ring->config.doorbell = new_write_index;
  }
}

// Initializes |out_ring| against |config|. |notify| is optional.
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
  out_ring->control =
      out_ring->config.shared_control ? out_ring->config.shared_control : &out_ring->local_control;
  abce_ring_atomic_store(&out_ring->control->reserve_cursor, initial_write_index);
  abce_ring_atomic_store(&out_ring->control->commit_cursor, initial_write_index);
  abce_ring_atomic_store(&out_ring->control->max_write_index, initial_write_index);
}

// Total reservation size for a payload of |payload_bytes|, after applying the
// configured minimum submission size and optional 64B rounding. Pass the result
// to both acquire and release.
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

static inline abce_status_t abce_ring_release(abce_ring_t* ring,
                                              const abce_ring_reservation_t* reservation);

//===----------------------------------------------------------------------===//
// Core callbacks (the C++ version passed these as lambdas)
//===----------------------------------------------------------------------===//

static inline uint64_t abce_ring_load_read_index_callback(void* user_data) {
  return abce_ring_load_read_index((const abce_ring_t*)user_data);
}

static inline abce_status_t abce_ring_publish_padding_callback(
    void* user_data, const abce_ring_reservation_t* padding) {
  abce_ring_t* ring = (abce_ring_t*)user_data;
  memset(ring->config.base + abce_ring_wrap_index(ring, padding->start), 0,
         (size_t)abce_ring_reservation_bytes(padding));
  return abce_ring_release(ring, padding);
}

static inline void abce_ring_publish_callback(void* user_data, uint64_t new_write_index) {
  abce_ring_t* ring = (abce_ring_t*)user_data;
  abce_ring_publish_write_index(ring, new_write_index);
  abce_ring_atomic_store_release(&ring->control->max_write_index, new_write_index);
  if (ring->notify) ring->notify(ring->notify_user_data, new_write_index);
}

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

//===----------------------------------------------------------------------===//
// Acquire / release / drain
//===----------------------------------------------------------------------===//

// Reserves a contiguous, zeroed |reserve_bytes| region of the ring.
//
// Blocks until the engine has freed enough space. On success sets
// |out_reservation|, which must be passed verbatim to abce_ring_release, and
// |out_buffer|, a zeroed pointer into the ring. Fails only if |reserve_bytes|
// is at or above the ring size (the request can never fit).
static inline abce_status_t abce_ring_acquire(abce_ring_t* ring, uint32_t reserve_bytes,
                                              abce_ring_reservation_t* out_reservation,
                                              char** out_buffer) {
  *out_buffer = NULL;

  abce_ring_producer_t producer;
  producer.reserve_cursor = &ring->control->reserve_cursor;
  producer.size = ring->config.size;
  producer.load_read_index = abce_ring_load_read_index_callback;
  producer.publish_padding = abce_ring_publish_padding_callback;
  producer.user_data = ring;

  const abce_status_t status = abce_ring_reserve(&producer, reserve_bytes, out_reservation);
  if (!abce_status_is_ok(status)) return status;

  // Wrap padding, when needed, was already published as a separate NOP
  // reservation. This reservation contains only the contiguous payload.
  char* buffer = ring->config.base + abce_ring_wrap_index(ring, out_reservation->start);
  memset(buffer, 0, reserve_bytes);
  *out_buffer = buffer;
  return ABCE_STATUS_OK;
}

// Publishes a previously acquired region and rings the doorbell. Blocks until
// all earlier reservations have been published so the write index advances
// strictly in reservation order.
static inline abce_status_t abce_ring_release(abce_ring_t* ring,
                                              const abce_ring_reservation_t* reservation) {
  abce_ring_publisher_t publisher;
  publisher.commit_cursor = &ring->control->commit_cursor;
  publisher.publish = abce_ring_publish_callback;
  publisher.wait = abce_ring_commit_wait_callback;
  publisher.user_data = ring;
  return abce_ring_publish(&publisher, reservation);
}

// Publishes an already-zeroed reservation as NOPs.
static inline abce_status_t abce_ring_cancel(abce_ring_t* ring,
                                             const abce_ring_reservation_t* reservation) {
  return abce_ring_release(ring, reservation);
}

// Waits until the engine has consumed everything published when this call
// snapshots the high-water mark.
static inline abce_status_t abce_ring_drain(const abce_ring_t* ring) {
  if (!ring->control) return ABCE_STATUS_INVALID_ARGUMENT;
  const uint64_t target = abce_ring_atomic_load_acquire(&ring->control->max_write_index);
  while (abce_ring_load_read_index(ring) < target) {
    abce_ring_pause();
  }
  return ABCE_STATUS_OK;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ABCE_RING_HOST_H_
