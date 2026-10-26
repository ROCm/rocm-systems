/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Accelerated Blit Copy Engine (ABCE) — device-side SDMA ring.
//
// A producer over the reserve/commit protocol in abce_ring_core.h, the same
// functions the host ring in abce_ring_host.h runs, driven from a shader. It
// adds only what the host ring does not need: stores into the ring from a
// kernel, and making them visible to the engine as a submission is published.
//
// A device ring belongs to the kernels of one device; the host never submits to
// it. That keeps a kernel's copy from queueing behind host work that may itself
// wait on the kernel, and lets the ring buffer and its control block live in
// that device's memory with agent-scope cursors.
//
// Requirements: the ring buffer and its read/write/doorbell words must be
// DEVICE-VISIBLE and the doorbell device-writable (an hsa_amd_queue_create SDMA
// queue whose read/write pointer and doorbell addresses are exposed through
// hsa_amd_queue_get_info), and the ring buffer mapped UNCACHED, as ROCr maps it.

#ifndef ABCE_RING_DEVICE_H_
#define ABCE_RING_DEVICE_H_

#include <stdbool.h>
#include <stdint.h>

#include "abce_config.h"
#include "abce_ring_core.h"
#include "sdma_packets.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// One SDMA ring as a kernel sees it. Trivially copyable: the host fills it in
// and a kernel only reads it.
typedef struct abce_device_ring_t {
  uint32_t* base;                 // ring buffer, device-visible.
  uint64_t size;                  // bytes; MUST be a power of two.
  volatile uint64_t* read_ptr;    // hw read index (bytes, monotonic).
  volatile uint64_t* write_ptr;   // hw write index (bytes) the engine reads.
  volatile uint64_t* doorbell;    // storing the write index here runs the ring.
  abce_ring_control_t* control;   // cursors, shared by the device's waves.
} abce_device_ring_t;

// Address of the bytes at monotonic |write_index|. abce_ring_reserve() never
// hands out a region that straddles the physical end, so a reservation is
// contiguous from here.
static inline char* abce_device_ring_write_address(const abce_device_ring_t* ring,
                                                   uint64_t write_index) {
  return (char*)ring->base + abce_ring_wrap(write_index, ring->size);
}

// Fills a reservation with NOPs: one counted header over the first
// SDMA_NOP_MAX_DWORDS, so the engine skips them as one packet, and zero dwords
// (one-dword NOPs) after it. Only wrap padding needs it; the submitter stores
// its packets whole.
static inline void abce_device_ring_pad(const abce_device_ring_t* ring,
                                        const abce_ring_reservation_t* reservation) {
  uint32_t* words = (uint32_t*)abce_device_ring_write_address(ring, reservation->start);
  const uint64_t num_words = abce_ring_reservation_bytes(reservation) / sizeof(uint32_t);
  for (uint64_t word_idx = 0; word_idx < num_words; ++word_idx) words[word_idx] = 0;
  const uint64_t nop_words = num_words < SDMA_NOP_MAX_DWORDS ? num_words : SDMA_NOP_MAX_DWORDS;
  words[0] = (uint32_t)(nop_words - 1) << SDMA_NOP_COUNT_SHIFT;
}

static inline uint64_t abce_device_ring_load_read_index(void* user_data) {
  const abce_device_ring_t* ring = (const abce_device_ring_t*)user_data;
  return abce_ring_atomic_load_hw(ring->read_ptr);
}

// The packets land before either store below (gfx9/gfx10 start on the doorbell,
// gfx11+ also poll the write pointer). Nothing else the thread wrote is
// released.
static inline void abce_device_ring_publish_write_index(void* user_data, uint64_t new_write_index) {
  abce_device_ring_t* ring = (abce_device_ring_t*)user_data;
  abce_ring_wait_stores();
  abce_ring_atomic_store_hw(ring->write_ptr, new_write_index);
  abce_ring_atomic_store_hw(ring->doorbell, new_write_index);
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

// Reserves |bytes| of contiguous ring space. While the ring is full it waits if
// |block|, else returns ABCE_STATUS_NOT_READY. Wrap padding is folded in
// (abce_ring_reserve()): the payload is the last |bytes| of the reservation, and
// any tail before it must be filled with NOPs before submit. |bytes| may be at
// most half the ring; ABCE_STATUS_OUT_OF_RANGE otherwise.
static inline abce_status_t abce_device_ring_reserve(abce_device_ring_t* ring, uint64_t bytes,
                                                     bool block,
                                                     abce_ring_reservation_t* out_reservation) {
  abce_ring_producer_t producer;
  producer.reserve_cursor = &ring->control->reserve_cursor;
  producer.cached_read_index = &ring->control->cached_read_index;
  producer.size = ring->size;
  producer.load_read_index = abce_device_ring_load_read_index;
  producer.publish_padding = NULL;
  producer.user_data = ring;
  return abce_ring_reserve_(&producer, bytes, block, out_reservation);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ABCE_RING_DEVICE_H_
