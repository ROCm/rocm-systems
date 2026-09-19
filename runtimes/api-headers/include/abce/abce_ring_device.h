/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Accelerated Blit Copy Engine (ABCE) — device-side SDMA ring.
//
// A producer over the shared reserve/commit protocol in abce_ring_core.h, the
// same functions the host ring in abce_ring_host.h runs, driven from a shader.
// It adds only what the host ring does not need: stores into the ring from a
// kernel, and releasing them to system scope as the write pointer is published.
//
// Requirements: the ring buffer and its read/write/doorbell words must be
// DEVICE-VISIBLE and the doorbell device-writable (an hsa_amd_queue_create SDMA
// queue whose read/write pointer and doorbell addresses are exposed through
// hsa_amd_queue_get_info). Host and device producers may share one ring only if
// both reserve and commit through the same abce_shared_ring_control_t.

#ifndef ABCE_RING_DEVICE_H_
#define ABCE_RING_DEVICE_H_

#include <stdbool.h>
#include <stdint.h>

#include "abce_config.h"
#include "abce_ring_core.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// One SDMA ring as a kernel sees it. Trivially copyable: the host fills it in
// and a kernel only reads it.
typedef struct abce_device_ring_t {
  uint32_t* base;                       // ring buffer, device-visible.
  uint64_t size;                        // bytes; MUST be a power of two.
  volatile uint64_t* read_ptr;          // hw read index (bytes, monotonic).
  volatile uint64_t* write_ptr;         // hw write index (bytes) the engine reads.
  volatile uint64_t* doorbell;          // storing the write index here runs the ring.
  abce_shared_ring_control_t* control;  // cursors shared with host producers.
} abce_device_ring_t;

// Address of the bytes at monotonic |write_index|. abce_ring_reserve() never
// hands out a region that straddles the physical end, so a reservation is
// contiguous from here.
static inline char* abce_device_ring_write_address(const abce_device_ring_t* ring,
                                                   uint64_t write_index) {
  return (char*)ring->base + abce_ring_wrap(write_index, ring->size);
}

// Fills a reservation with NOPs: a zero dword is an SDMA NOP. Only wrap padding
// needs it; the submitter stores its packets whole.
static inline void abce_device_ring_zero(const abce_device_ring_t* ring,
                                         const abce_ring_reservation_t* reservation) {
  uint32_t* words = (uint32_t*)abce_device_ring_write_address(ring, reservation->start);
  const uint64_t num_words = abce_ring_reservation_bytes(reservation) / sizeof(uint32_t);
  for (uint64_t word_idx = 0; word_idx < num_words; ++word_idx) words[word_idx] = 0;
}

static inline uint64_t abce_device_ring_load_read_index(void* user_data) {
  const abce_device_ring_t* ring = (const abce_device_ring_t*)user_data;
  return abce_ring_atomic_load_hw(ring->read_ptr);
}

// THE WRITE-POINTER STORE IS THE SUBMISSION'S SYSTEM-SCOPE RELEASE POINT. It
// orders the packet dwords, and everything the calling thread wrote beforehand
// including the buffers the packets name, ahead of the engine seeing new work.
// That has to be the write pointer rather than the doorbell: where KFD enables
// write-pointer polling for SDMA user queues
static inline void abce_device_ring_publish_write_index(void* user_data, uint64_t new_write_index) {
  abce_device_ring_t* ring = (abce_device_ring_t*)user_data;
  abce_ring_atomic_store_release((uint64_t*)ring->write_ptr, new_write_index);
  // Relaxed: the release above already made the packets visible, and the
  // doorbell carries the write index itself.
  abce_ring_atomic_store((uint64_t*)ring->doorbell, new_write_index);
  // The high-water mark is only compared against the read index by the host
  // ring's drain; no data is published through it, so relaxed avoids paying the
  // release above a second time.
  abce_ring_atomic_store(&ring->control->max_write_index, new_write_index);
}

// Publishes [start, end) in reservation order and rings the doorbell.
static inline abce_status_t abce_device_ring_submit(abce_device_ring_t* ring,
                                                    const abce_ring_reservation_t* reservation) {
  abce_ring_publisher_t publisher;
  publisher.commit_cursor = &ring->control->commit_cursor;
  publisher.publish = abce_device_ring_publish_write_index;
  publisher.wait = NULL;
  publisher.user_data = ring;
  return abce_ring_publish(&publisher, reservation);
}

static inline abce_status_t abce_device_ring_publish_padding(
    void* user_data, const abce_ring_reservation_t* padding) {
  abce_device_ring_t* ring = (abce_device_ring_t*)user_data;
  abce_device_ring_zero(ring, padding);
  return abce_device_ring_submit(ring, padding);
}

// Reserves |bytes| of contiguous ring space, blocking while the ring is full.
// Fails only with ABCE_STATUS_OUT_OF_RANGE, when |bytes| can never fit.
static inline abce_status_t abce_device_ring_reserve(abce_device_ring_t* ring, uint64_t bytes,
                                                     abce_ring_reservation_t* out_reservation) {
  abce_ring_producer_t producer;
  producer.reserve_cursor = &ring->control->reserve_cursor;
  producer.size = ring->size;
  producer.load_read_index = abce_device_ring_load_read_index;
  producer.publish_padding = abce_device_ring_publish_padding;
  producer.user_data = ring;
  return abce_ring_reserve(&producer, bytes, out_reservation);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ABCE_RING_DEVICE_H_
