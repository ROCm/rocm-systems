/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Accelerated Blit Copy Engine (ABCE) — device-initiated SDMA submission.
//
// A kernel submits one linear copy or one inline write per call, with an
// optional dependency and an optional completion signal, onto a device ring
// (abce_ring_device.h). No batching, fan-out or interrupts; those are the host
// planner's (abce_host.h).
//
// The host picks the ring: a handle names one device ring, chosen by the host's
// policy for a transfer class, so the kernel does no routing and a sequence on
// one handle stays on one ring.
//
// KERNEL USE: include this before any other ABCE header in a HIP translation
// unit; it wraps them in `#pragma clang force_cuda_host_device`.
//
// CONCURRENCY: any number of waves may submit concurrently. Lanes of one wave
// take turns, or a caller that submits from one lane uses the *_one_lane calls,
// which skip the turn loop. Device rings are separate from the host's.
//
// MEMORY VISIBILITY and LIVENESS are the caller's; see
// abce_device_submit_linear().

#ifndef ABCE_DEVICE_H_
#define ABCE_DEVICE_H_

#if defined(__HIP__)
// Outside the host+device region, so these keep their own device overloads.
#include <hip/hip_runtime.h>
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#if !defined(__HIP_DEVICE_COMPILE__)
#include <sched.h>
#endif  // !__HIP_DEVICE_COMPILE__
#pragma clang force_cuda_host_device begin
#define ABCE_DEVICE_FORCE_HOST_DEVICE_ 1
#endif  // __HIP__

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "abce_builder.h"
#include "abce_config.h"
#include "abce_ring_core.h"
#include "abce_ring_device.h"
#include "sdma_packets.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Bumped on any change to abce_device_engine_t or the submit contract; a
// mismatched kernel and runtime fail with ABCE_STATUS_INVALID_ARGUMENT.
#define ABCE_DEVICE_ABI_VERSION 1u

// One device ring and what a kernel needs to submit to it. Built on the host by
// abce_device_engine_initialize() and published to device memory before any
// handle names it; read-only to kernels, which write only the ring and its
// control block.
typedef struct abce_device_engine_t {
  ABCE_ALIGNAS(64) uint32_t abi_version;
  // SDMA engine the ring runs on; informational.
  uint32_t engine_id;
  // SDMA atomics reach the signal words (required for ADD/SUB signals).
  bool device_atomic_support;
  abce_builder_t builder;
  abce_device_ring_t ring;
} abce_device_engine_t;

// Passed to kernels by value. Valid only on the device the ring belongs to.
typedef struct abce_device_handle_t {
  abce_device_engine_t* engine;
  uint32_t abi_version;
} abce_device_handle_t;

typedef enum abce_device_signal_op_t {
  ABCE_DEVICE_SIGNAL_SET = 0,  // *address = value.
  ABCE_DEVICE_SIGNAL_ADD = 1,  // *address += value, wrapping.
  ABCE_DEVICE_SIGNAL_SUB = 2,  // *address -= value, wrapping.
} abce_device_signal_op_t;

// By default a signal lands only after its own submission and every earlier one
// on its ring, so k ADD signals on a counter mean the first k submissions are
// done. UNORDERED lets a signal ride in a gfx125+ copy's own packet (one packet,
// not chunked), so later copies may overlap it: the signal then covers that copy
// alone, and a counter says how many such copies landed, not which. Elsewhere
// it changes nothing.
#define ABCE_DEVICE_SIGNAL_UNORDERED (1u << 0)

// Submit flag: never block and never park the engine. If the ring has no room or
// the dependency is unmet, return ABCE_STATUS_NOT_READY with nothing queued; the
// caller retries later. The submit still waits its turn behind submissions
// already reserved, which finish without outside help.
#define ABCE_DEVICE_SUBMIT_NONBLOCKING (1u << 0)

// Completion signal: once the submission's writes have landed, the engine
// applies |op| with |value| to the word at |address| (8-byte aligned,
// SDMA-visible). The caller owns and initializes the word; ABCE never writes it
// from the kernel.
typedef struct abce_device_signal_t {
  uint64_t* address;
  uint64_t value;
  uint32_t op;     // abce_device_signal_op_t.
  uint32_t flags;  // ABCE_DEVICE_SIGNAL_*.
} abce_device_signal_t;

// Unsigned compares.
typedef enum abce_device_wait_compare_t {
  ABCE_DEVICE_WAIT_GEQ = 0,  // *address >= value.
  ABCE_DEVICE_WAIT_LEQ = 1,  // *address <= value.
} abce_device_wait_compare_t;

// A condition on a word, used as a dependency or polled.
typedef struct abce_device_wait_t {
  uint64_t* address;
  uint64_t value;
  uint32_t compare;  // abce_device_wait_compare_t.
} abce_device_wait_t;

static inline bool abce_device_wait_holds(const abce_device_wait_t* wait, uint64_t word) {
  return wait->compare == ABCE_DEVICE_WAIT_LEQ ? word <= wait->value : word >= wait->value;
}

static inline uint32_t abce_device_wait_function(const abce_device_wait_t* wait) {
  return wait->compare == ABCE_DEVICE_WAIT_LEQ ? SDMA_FUNC_LESS_EQUAL : SDMA_FUNC_GREATER_EQUAL;
}

static inline bool abce_device_signal_op_is_atomic(uint32_t op) {
  return op == ABCE_DEVICE_SIGNAL_ADD || op == ABCE_DEVICE_SIGNAL_SUB;
}

//===----------------------------------------------------------------------===//
// Host side: building an engine
//===----------------------------------------------------------------------===//

// UNIMPLEMENTED for parts whose copies need GCR packets, which the device
// submitter does not emit.
static inline abce_status_t abce_device_engine_initialize(const abce_builder_t* builder,
                                                          const abce_device_ring_t* ring,
                                                          uint32_t engine_id,
                                                          bool device_atomic_support,
                                                          abce_device_engine_t* out_engine) {
  if (abce_builder_requires_gcr(builder)) return ABCE_STATUS_UNIMPLEMENTED;
  memset(out_engine, 0, sizeof(*out_engine));
  out_engine->abi_version = ABCE_DEVICE_ABI_VERSION;
  out_engine->engine_id = engine_id;
  out_engine->device_atomic_support = device_atomic_support;
  out_engine->builder = *builder;
  out_engine->ring = *ring;
  return ABCE_STATUS_OK;
}

static inline bool abce_device_handle_valid(const abce_device_handle_t* handle) {
  return handle && handle->engine && handle->abi_version == ABCE_DEVICE_ABI_VERSION &&
         handle->engine->abi_version == ABCE_DEVICE_ABI_VERSION;
}

// A kernel's code object runs only on its own target, so for the targets below
// the packet format is fixed at compile time and only the host's size limits
// are read from the engine.
#if defined(__HIP_DEVICE_COMPILE__)
#if defined(__gfx90a__)
#define ABCE_DEVICE_TARGET_ISA_ {9u, 0u, 10u}
#elif defined(__gfx942__)
#define ABCE_DEVICE_TARGET_ISA_ {9u, 4u, 2u}
#elif defined(__gfx950__)
#define ABCE_DEVICE_TARGET_ISA_ {9u, 5u, 0u}
#elif defined(__gfx1250__)
#define ABCE_DEVICE_TARGET_ISA_ {12u, 5u, 0u}
#endif
#endif  // __HIP_DEVICE_COMPILE__

static inline void abce_device_load_builder(const abce_builder_t* engine_builder,
                                            abce_builder_t* out_builder) {
#if defined(ABCE_DEVICE_TARGET_ISA_)
  const abce_isa_version_t isa = ABCE_DEVICE_TARGET_ISA_;
  const abce_packet_caps_t packet_caps = abce_detect_packet_caps(isa);
  out_builder->isa = isa;
  out_builder->packet_caps = packet_caps;
  out_builder->scope_fields = packet_caps.scope_fields;
  out_builder->is_gfx125plus = packet_caps.gfx125plus;
  out_builder->use_extended_count = engine_builder->use_extended_count;
  out_builder->max_copy_size = engine_builder->max_copy_size;
  out_builder->max_fill_dwords = engine_builder->max_fill_dwords;
#else
  *out_builder = *engine_builder;
#endif
}

// Engine fields loaded once before the lane loop, so they are not reloaded
// after each lane's ring stores. The ring is loaded per lane, in emit.
typedef struct abce_device_submit_view_t {
  const abce_device_engine_t* engine;
  abce_builder_t builder;
  bool device_atomic_support;
} abce_device_submit_view_t;

//===----------------------------------------------------------------------===//
// Submission
//===----------------------------------------------------------------------===//

typedef enum abce_device_submit_kind_t {
  ABCE_DEVICE_SUBMIT_COPY,
  ABCE_DEVICE_SUBMIT_WRITE,
  ABCE_DEVICE_SUBMIT_SIGNAL,  // The completion alone.
} abce_device_submit_kind_t;

typedef struct abce_device_request_t {
  abce_device_submit_kind_t kind;
  void* dst;        // COPY and WRITE.
  const void* src;  // COPY only.
  uint64_t size;    // COPY: bytes. WRITE: 1, 2, 4 or 8.
  uint64_t value;   // WRITE only.
  const abce_device_wait_t* dependency;
  const abce_device_signal_t* completion;
  // Applied right after |completion|, in the same frame; needs |completion|. Both land after the
  // operation's writes, the second after the first.
  const abce_device_signal_t* second_completion;
  uint32_t flags;  // ABCE_DEVICE_SUBMIT_*.
} abce_device_request_t;

static inline uint64_t abce_device_wait_packet_bytes(const abce_builder_t* builder) {
  return abce_builder_is_gfx125plus(builder) ? sizeof(SDMA_PKT_POLL_MEM_64B_GFX125PLUS)
                                             : sizeof(SDMA_PKT_POLL_REGMEM);
}

static inline uint64_t abce_device_completion_packet_bytes(const abce_builder_t* builder,
                                                           const abce_device_signal_t* completion) {
  if (abce_device_signal_op_is_atomic(completion->op)) return sizeof(SDMA_PKT_ATOMIC);
  return abce_builder_is_gfx125plus(builder) ? sizeof(SDMA_PKT_FENCE_64B_GFX125PLUS)
                                             : 2 * sizeof(SDMA_PKT_FENCE);
}

// Packets are built in zeroed locals and stored whole; the ring is never read.
static inline size_t abce_device_store_wait(const abce_builder_t* builder, char* out,
                                            const abce_device_wait_t* wait) {
  if (abce_builder_is_gfx125plus(builder)) {
    SDMA_PKT_POLL_MEM_64B_GFX125PLUS poll;
    memset(&poll, 0, sizeof(poll));
    abce_builder_build_poll_64b_compare(builder, (char*)&poll, wait->address, wait->value,
                                        UINT64_MAX, abce_device_wait_function(wait));
    *(SDMA_PKT_POLL_MEM_64B_GFX125PLUS*)out = poll;
    return sizeof(poll);
  }
  SDMA_PKT_POLL_REGMEM poll;
  memset(&poll, 0, sizeof(poll));
  abce_builder_build_poll_compare(builder, (char*)&poll, wait->address, (uint32_t)wait->value,
                                  UINT32_MAX, abce_device_wait_function(wait));
  *(SDMA_PKT_POLL_REGMEM*)out = poll;
  return sizeof(poll);
}

static inline size_t abce_device_store_completion(const abce_builder_t* builder, char* out,
                                                  const abce_device_signal_t* completion) {
  if (abce_device_signal_op_is_atomic(completion->op)) {
    // SUB is ADD64 of the negated value.
    const uint64_t addend =
        completion->op == ABCE_DEVICE_SIGNAL_SUB ? (uint64_t)0 - completion->value : completion->value;
    SDMA_PKT_ATOMIC atomic;
    memset(&atomic, 0, sizeof(atomic));
    abce_builder_build_atomic_add(builder, (char*)&atomic, completion->address, addend);
    *(SDMA_PKT_ATOMIC*)out = atomic;
    return sizeof(atomic);
  }
  if (abce_builder_is_gfx125plus(builder)) {
    SDMA_PKT_FENCE_64B_GFX125PLUS fence;
    memset(&fence, 0, sizeof(fence));
    abce_builder_build_fence_64b(builder, (char*)&fence, completion->address, completion->value);
    *(SDMA_PKT_FENCE_64B_GFX125PLUS*)out = fence;
    return sizeof(fence);
  }
  // Low word first: a rising value crossing 2^32 never transiently overshoots.
  uint32_t* signal_words = (uint32_t*)completion->address;
  SDMA_PKT_FENCE* fence_packets = (SDMA_PKT_FENCE*)out;
  for (uint32_t word_idx = 0; word_idx < 2; ++word_idx) {
    SDMA_PKT_FENCE fence;
    memset(&fence, 0, sizeof(fence));
    abce_builder_build_fence(builder, (char*)&fence, &signal_words[word_idx],
                             (uint32_t)(completion->value >> (32 * word_idx)));
    fence_packets[word_idx] = fence;
  }
  return 2 * sizeof(SDMA_PKT_FENCE);
}

// The packets of one submission and their sizes, decided before reserving.
typedef struct abce_device_frame_t {
  uint64_t num_packets;  // copy chunks.
  bool fused;            // gfx125+ COPY_LINEAR_WAITSIGNAL.
  bool wait;
  bool fused_signal;
  uint64_t wait_bytes;
  uint64_t body_bytes;
  uint64_t completion_bytes;
  uint64_t second_completion_bytes;
  uint64_t bytes;
} abce_device_frame_t;

// Frame: [wait] body [signal] [second signal]. gfx125+ copies fold the wait, and
// an UNORDERED signal, into COPY_LINEAR_WAITSIGNAL; otherwise POLL, body, then
// ATOMIC or FENCE, which land after everything earlier on the ring. The second
// signal is always standalone.
static inline abce_status_t abce_device_plan_frame(const abce_builder_t* builder,
                                                   const abce_device_request_t* request,
                                                   abce_device_frame_t* out_frame) {
  const bool copy = request->kind == ABCE_DEVICE_SUBMIT_COPY;
  const bool fused = copy && abce_builder_is_gfx125plus(builder);
  const uint64_t num_packets = copy ? abce_builder_num_copy_packets(builder, (size_t)request->size)
                                    : 0;
  if (num_packets > ABCE_MAX_OP_PACKETS) return ABCE_STATUS_OUT_OF_RANGE;

  // A blocking submit always queues the wait: checking it first costs a load
  // from wherever the word lives, often host or peer memory. A non-blocking one
  // must check, and queues no wait.
  const abce_device_wait_t* dependency = request->dependency;
  const bool block = !(request->flags & ABCE_DEVICE_SUBMIT_NONBLOCKING);
  if (dependency && !block &&
      !abce_device_wait_holds(dependency, abce_ring_atomic_load_hw(dependency->address)))
    return ABCE_STATUS_NOT_READY;
  const bool wait = dependency && block;
  const abce_device_signal_t* completion = request->completion;
  // A chunked copy cannot fuse its signal: each chunk's signal would cover only
  // that chunk, and ABCE does not arm the caller's word for a per-chunk count.
  // Its signal is then a standalone packet, which waits for every chunk.
  const bool fused_signal = fused && num_packets == 1 && completion &&
                            (completion->flags & ABCE_DEVICE_SIGNAL_UNORDERED);

  out_frame->num_packets = num_packets;
  out_frame->fused = fused;
  out_frame->wait = wait;
  out_frame->fused_signal = fused_signal;
  out_frame->wait_bytes = wait && !fused ? abce_device_wait_packet_bytes(builder) : 0;
  // Compacted fused packets: header + body per chunk, the wait on the first, and
  // the signal (single chunk only) on it too.
  out_frame->body_bytes =
      fused ? (num_packets * (1u + 6u) + (wait ? 7u : 0u) + (fused_signal ? 5u : 0u)) *
                  sizeof(uint32_t)
      : copy                                      ? num_packets * sizeof(SDMA_PKT_COPY_LINEAR)
      : request->kind == ABCE_DEVICE_SUBMIT_SIGNAL ? 0
      : request->size < 4u                        ? request->size * sizeof(SDMA_PKT_CONSTANT_FILL)
                                                  : sizeof(SDMA_PKT_WRITE_UNTILED) + request->size;
  out_frame->completion_bytes =
      completion && !fused_signal ? abce_device_completion_packet_bytes(builder, completion) : 0;
  out_frame->second_completion_bytes =
      request->second_completion
          ? abce_device_completion_packet_bytes(builder, request->second_completion)
          : 0;
  out_frame->bytes = out_frame->wait_bytes + out_frame->body_bytes + out_frame->completion_bytes +
                     out_frame->second_completion_bytes;
  return ABCE_STATUS_OK;
}

static inline void abce_device_write_frame(const abce_builder_t* builder,
                                           const abce_device_request_t* request,
                                           const abce_device_frame_t* frame, char* cursor) {
  const abce_device_wait_t* dependency = request->dependency;
  const abce_device_signal_t* completion = request->completion;
  if (frame->wait_bytes) cursor += abce_device_store_wait(builder, cursor, dependency);
  if (frame->fused) {
    abce_wait_signal_params_t params = abce_wait_signal_none();
    if (frame->wait) {
      params.wait_addr = dependency->address;
      params.wait_reference = dependency->value;
      params.wait_function = abce_device_wait_function(dependency);
    }
    if (frame->fused_signal) {
      params.signal_addr = completion->address;
      params.signal_operation = completion->op == ABCE_DEVICE_SIGNAL_SUB   ? SDMA_SIGNAL_OP_SUB64
                                : completion->op == ABCE_DEVICE_SIGNAL_ADD ? SDMA_SIGNAL_OP_ADD64
                                                                          : SDMA_SIGNAL_OP_WRITE64;
      params.signal_value = completion->value;
    }
    params.wait_first_chunk_only = true;
    abce_builder_build_wait_signal_copy(builder, cursor, request->dst, request->src,
                                        (size_t)request->size, &params);
  } else if (request->kind == ABCE_DEVICE_SUBMIT_COPY) {
    SDMA_PKT_COPY_LINEAR* copy_packets = (SDMA_PKT_COPY_LINEAR*)cursor;
    uint64_t copied = 0;
    for (uint64_t packet_idx = 0; packet_idx < frame->num_packets; ++packet_idx) {
      const uint32_t chunk_size =
          (uint32_t)ABCE_MIN(request->size - copied, (uint64_t)builder->max_copy_size);
      SDMA_PKT_COPY_LINEAR packet;
      memset(&packet, 0, sizeof(packet));
      abce_builder_form_copy_packet(builder, &packet, (char*)request->dst + copied,
                                    (const char*)request->src + copied, chunk_size);
      copy_packets[packet_idx] = packet;
      copied += chunk_size;
    }
  } else if (request->kind == ABCE_DEVICE_SUBMIT_WRITE) {
    if (request->size < 4u) {
      // WRITE_LINEAR is dword-granular: one byte fill per byte.
      for (uint32_t byte_idx = 0; byte_idx < (uint32_t)request->size; ++byte_idx)
        abce_builder_build_fill_byte(builder, cursor + byte_idx * sizeof(SDMA_PKT_CONSTANT_FILL),
                                     (char*)request->dst + byte_idx,
                                     (uint8_t)(request->value >> (8 * byte_idx)));
    } else {
      const uint32_t data_words[2] = {(uint32_t)request->value, (uint32_t)(request->value >> 32)};
      abce_builder_build_write(builder, cursor, request->dst, data_words,
                               (uint32_t)(request->size / sizeof(uint32_t)));
    }
  }
  cursor += frame->body_bytes;
  if (frame->completion_bytes) {
    abce_device_store_completion(builder, cursor, completion);
    cursor += frame->completion_bytes;
  }
  if (frame->second_completion_bytes)
    abce_device_store_completion(builder, cursor, request->second_completion);
}

// Reserves |bytes|, folding in any wrap tail as NOPs, and returns where the
// payload goes: the reservation's last |bytes|.
static inline abce_status_t abce_device_reserve_payload(abce_device_ring_t* ring, uint64_t bytes,
                                                        bool block,
                                                        abce_ring_reservation_t* out_reservation,
                                                        uint64_t* out_payload_start) {
  // Padding is folded into the reservation, so a payload may use at most half
  // the ring; otherwise reserve fails only for a non-blocking submit with no
  // room.
  const abce_status_t status = abce_device_ring_reserve(ring, bytes, block, out_reservation);
  if (!abce_status_is_ok(status)) return status;
  abce_ring_reservation_t padding;
  padding.start = out_reservation->start;
  padding.end = out_reservation->end - bytes;
  if (padding.end != padding.start) abce_device_ring_pad(ring, &padding);
  *out_payload_start = padding.end;
  return ABCE_STATUS_OK;
}

static inline abce_status_t abce_device_emit(const abce_builder_t* builder,
                                             const abce_device_ring_t* engine_ring,
                                             const abce_device_request_t* request) {
  // Register copy, as with abce_device_submit_view_t.
  abce_device_ring_t ring_copy = *engine_ring;
  abce_device_ring_t* ring = &ring_copy;
  abce_device_frame_t frame;
  abce_status_t status = abce_device_plan_frame(builder, request, &frame);
  if (!abce_status_is_ok(status)) return status;
  abce_ring_reservation_t reservation;
  uint64_t payload_start = 0;
  status = abce_device_reserve_payload(ring, frame.bytes,
                                       !(request->flags & ABCE_DEVICE_SUBMIT_NONBLOCKING),
                                       &reservation, &payload_start);
  if (!abce_status_is_ok(status)) return status;
  abce_device_write_frame(builder, request, &frame,
                          abce_device_ring_write_address(ring, payload_start));
  return abce_device_ring_submit(ring, &reservation);
}

static inline bool abce_device_word_well_formed(const uint64_t* address) {
  return address && ((uintptr_t)address % sizeof(uint64_t)) == 0;
}

static inline abce_status_t abce_device_validate_request(const abce_device_submit_view_t* view,
                                                         const abce_device_request_t* request) {
  if ((request->flags & ~ABCE_DEVICE_SUBMIT_NONBLOCKING) != 0) return ABCE_STATUS_INVALID_ARGUMENT;
  if (request->kind == ABCE_DEVICE_SUBMIT_SIGNAL) {
    if (!request->completion) return ABCE_STATUS_INVALID_ARGUMENT;
  } else if (!request->dst) {
    return ABCE_STATUS_INVALID_ARGUMENT;
  } else if (request->kind == ABCE_DEVICE_SUBMIT_COPY) {
    if (!request->src || request->size == 0) return ABCE_STATUS_INVALID_ARGUMENT;
  } else {
    const uint64_t size = request->size;
    const uint64_t alignment = size < 4u ? size : 4u;
    if ((size != 1 && size != 2 && size != 4 && size != 8) ||
        ((uintptr_t)request->dst % alignment) != 0)
      return ABCE_STATUS_INVALID_ARGUMENT;
  }
  const abce_device_wait_t* dependency = request->dependency;
  if (dependency &&
      (!abce_device_word_well_formed(dependency->address) ||
       (dependency->compare != ABCE_DEVICE_WAIT_GEQ && dependency->compare != ABCE_DEVICE_WAIT_LEQ) ||
       (!abce_builder_is_gfx125plus(&view->builder) && dependency->value > UINT32_MAX)))
    return ABCE_STATUS_INVALID_ARGUMENT;
  if (request->second_completion && !request->completion) return ABCE_STATUS_INVALID_ARGUMENT;
  const abce_device_signal_t* completions[2] = {request->completion, request->second_completion};
  for (uint32_t completion_idx = 0; completion_idx < 2; ++completion_idx) {
    const abce_device_signal_t* completion = completions[completion_idx];
    if (!completion) continue;
    if (!abce_device_word_well_formed(completion->address) ||
        (completion->op != ABCE_DEVICE_SIGNAL_SET && !abce_device_signal_op_is_atomic(completion->op)) ||
        (completion->flags & ~ABCE_DEVICE_SIGNAL_UNORDERED) != 0)
      return ABCE_STATUS_INVALID_ARGUMENT;
    if (abce_device_signal_op_is_atomic(completion->op) && !view->device_atomic_support)
      return ABCE_STATUS_UNIMPLEMENTED;
  }
  return ABCE_STATUS_OK;
}

static inline abce_status_t abce_device_submit_lane(const abce_device_submit_view_t* view,
                                                    const abce_device_request_t* request) {
  const abce_status_t status = abce_device_validate_request(view, request);
  if (!abce_status_is_ok(status)) return status;
  return abce_device_emit(&view->builder, &view->engine->ring, request);
}

#if defined(__HIP_DEVICE_COMPILE__)
static inline uint32_t abce_device_lane_id(void) {
  return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}

static inline uint64_t abce_device_lane_mask(bool predicate) {
  return __builtin_amdgcn_ballot_w64(predicate);
}

// |value| in lane |lane|, which must be the same in every active lane.
static inline uint64_t abce_device_read_lane(uint64_t value, uint32_t lane) {
  const uint32_t low = (uint32_t)__builtin_amdgcn_readlane((int)(uint32_t)value, (int)lane);
  const uint32_t high = (uint32_t)__builtin_amdgcn_readlane((int)(uint32_t)(value >> 32), (int)lane);
  return ((uint64_t)high << 32) | low;
}

// Every active lane, all on view's engine and all blocking, submits under one
// reservation and one publish: frames go back to back in lane order, each lane
// writing its own. A lane whose request fails validation drops out. Groups past
// half the ring take turns instead.
static inline abce_status_t abce_device_submit_lanes_together(const abce_device_submit_view_t* view,
                                                              const abce_device_request_t* request,
                                                              uint32_t lane) {
  abce_device_frame_t frame;
  abce_status_t status = abce_device_validate_request(view, request);
  if (abce_status_is_ok(status)) status = abce_device_plan_frame(&view->builder, request, &frame);
  const uint64_t members = abce_device_lane_mask(abce_status_is_ok(status));
  if (!abce_status_is_ok(status)) return status;

  uint64_t offset = 0;
  uint64_t total_bytes = 0;
  for (uint64_t pending = members; pending != 0; pending &= pending - 1) {
    const uint32_t member = (uint32_t)abce_count_trailing_zeros_u64(pending);
    if (member == lane) offset = total_bytes;
    total_bytes += abce_device_read_lane(frame.bytes, member);
  }
  // Register copy, as with abce_device_submit_view_t.
  abce_device_ring_t ring_copy = view->engine->ring;
  abce_device_ring_t* ring = &ring_copy;
  if (total_bytes > ring->size / 2) {
    for (uint64_t pending = members; pending != 0; pending &= pending - 1) {
      if ((uint32_t)abce_count_trailing_zeros_u64(pending) == lane)
        status = abce_device_emit(&view->builder, &view->engine->ring, request);
    }
    return status;
  }

  // Blocking, and within half the ring: reserve and publish cannot fail.
  const uint32_t leader = (uint32_t)abce_count_trailing_zeros_u64(members);
  abce_ring_reservation_t reservation;
  uint64_t payload_start = 0;
  if (lane == leader)
    abce_device_reserve_payload(ring, total_bytes, true, &reservation, &payload_start);
  payload_start = abce_device_read_lane(payload_start, leader);
  abce_device_write_frame(&view->builder, request, &frame,
                          abce_device_ring_write_address(ring, payload_start + offset));
  // The publish's store wait covers every lane's stores: the count is the wave's.
  if (lane == leader) abce_device_ring_submit(ring, &reservation);
  return ABCE_STATUS_OK;
}
#endif  // __HIP_DEVICE_COMPILE__

// |one_lane| is a constant at every call site: with it set, the lane loop is not
// compiled in.
static inline abce_status_t abce_device_submit(const abce_device_handle_t* handle,
                                               const abce_device_request_t* request,
                                               bool one_lane) {
  if (!handle || !handle->engine) return ABCE_STATUS_INVALID_ARGUMENT;
  const abce_device_engine_t* engine = handle->engine;
  // Load before checking, so the loads overlap.
  const uint32_t engine_abi_version = engine->abi_version;
  abce_device_submit_view_t view;
  view.engine = engine;
  abce_device_load_builder(&engine->builder, &view.builder);
  view.device_atomic_support = engine->device_atomic_support;
  if (handle->abi_version != ABCE_DEVICE_ABI_VERSION ||
      engine_abi_version != ABCE_DEVICE_ABI_VERSION)
    return ABCE_STATUS_INVALID_ARGUMENT;

#if defined(__HIP_DEVICE_COMPILE__)
  if (one_lane) {
    // Two lanes of one wave in here would each spin on the other's turn.
    if (__builtin_popcountll(__builtin_amdgcn_read_exec()) != 1)
      return ABCE_STATUS_INVALID_ARGUMENT;
    return abce_device_submit_lane(&view, request);
  }
  const uint32_t lane = abce_device_lane_id();
  abce_status_t status = ABCE_STATUS_OK;
  // Non-blocking lanes take turns, so one lane's NOT_READY holds back no other.
  const bool together = !(request->flags & ABCE_DEVICE_SUBMIT_NONBLOCKING);
  for (uint64_t waiting = abce_device_lane_mask(!together); waiting != 0; waiting &= waiting - 1) {
    if ((uint32_t)abce_count_trailing_zeros_u64(waiting) == lane)
      status = abce_device_submit_lane(&view, request);
  }
  // The rest submit together, one group per engine.
  const uint64_t engine_address = (uint64_t)(uintptr_t)view.engine;
  for (uint64_t waiting = abce_device_lane_mask(together); waiting != 0;) {
    const uint64_t group_engine =
        abce_device_read_lane(engine_address, (uint32_t)abce_count_trailing_zeros_u64(waiting));
    const bool in_group = ((waiting >> lane) & 1u) && engine_address == group_engine;
    waiting &= ~abce_device_lane_mask(in_group);
    if (in_group) status = abce_device_submit_lanes_together(&view, request, lane);
  }
  return status;
#else
  (void)one_lane;
  return abce_device_submit_lane(&view, request);
#endif  // __HIP_DEVICE_COMPILE__
}

static inline abce_device_request_t abce_device_request(abce_device_submit_kind_t kind,
                                                        const abce_device_wait_t* dependency,
                                                        const abce_device_signal_t* completion,
                                                        uint32_t flags) {
  abce_device_request_t request;
  request.kind = kind;
  request.dst = NULL;
  request.src = NULL;
  request.size = 0;
  request.value = 0;
  request.dependency = dependency;
  request.completion = completion;
  request.second_completion = NULL;
  request.flags = flags;
  return request;
}

static inline abce_device_request_t abce_device_linear_request(
    void* dst, const void* src, uint64_t size, const abce_device_wait_t* dependency,
    const abce_device_signal_t* completion, uint32_t flags) {
  abce_device_request_t request =
      abce_device_request(ABCE_DEVICE_SUBMIT_COPY, dependency, completion, flags);
  request.dst = dst;
  request.src = src;
  request.size = size;
  return request;
}

static inline abce_device_request_t abce_device_write_request(
    void* dst, uint64_t value, uint32_t size, const abce_device_wait_t* dependency,
    const abce_device_signal_t* completion, uint32_t flags) {
  abce_device_request_t request =
      abce_device_request(ABCE_DEVICE_SUBMIT_WRITE, dependency, completion, flags);
  request.dst = dst;
  request.size = size;
  request.value = value;
  return request;
}

// Submits one linear copy of |size| bytes from |src| to |dst| on the handle's
// ring. Returns once the copy is queued, not complete. Submissions on one ring
// start in order but may run concurrently; only signals say what has finished.
//
// |completion| (optional): signal applied once the copy's writes land, ordered
//   as ABCE_DEVICE_SIGNAL_UNORDERED describes. ADD and SUB need
//   device_atomic_support.
// |dependency| (optional): the engine waits for it; the thread does not. Before
//   gfx125 only the low 32 bits are compared, so the value must fit in 32 bits
//   and the word must stay below 2^32 while waiting.
// |flags|: ABCE_DEVICE_SUBMIT_*.
//
// MEMORY VISIBILITY IS THE CALLER'S. The submit releases nothing for the
// caller. Before submitting, release at system scope the source, prior stores
// to the destination and the signal's initial value. For data other waves of
// the workgroup wrote, that is a workgroup barrier and then
// __threadfence_system() from the submitting lane; data from other workgroups
// needs their own release (a fence writes back only its own XCC's L2). One
// release may cover many submissions. A waiter that sees the condition hold
// acquires at system scope before reading what the engine wrote; that covers
// its own CU and XCC only.
//
// Any number of lanes may call it; lanes of a wave take turns. For a caller that
// submits from one lane, abce_device_submit_linear_one_lane() is the same
// without the turn loop, at roughly half the registers.
//
// LIVENESS. A blocking submit waits while the ring is full, and a dependency
// parks the ring, and every later submission on it, until it holds. So never
// make a submission depend, directly or through the ring filling up, on work
// that can only run after it: a signal from a later submission on the same
// ring, from a workgroup that may not be resident yet, or from a GPU that waits
// on this one. Use ABCE_DEVICE_SUBMIT_NONBLOCKING where that cannot be ruled
// out. Kernel completion does not wait for submitted copies; wait on a signal
// (from a kernel, or the host and streams) before reusing a source or reading a
// destination.
//
// Returns INVALID_ARGUMENT (bad handle, pointer, size, signal, wait or flags),
// UNIMPLEMENTED (ADD/SUB without atomics), OUT_OF_RANGE (never fits the ring),
// or NOT_READY (non-blocking only).
static inline abce_status_t abce_device_submit_linear(const abce_device_handle_t* handle,
                                                      void* dst, const void* src, uint64_t size,
                                                      const abce_device_wait_t* dependency,
                                                      const abce_device_signal_t* completion,
                                                      uint32_t flags) {
  const abce_device_request_t request =
      abce_device_linear_request(dst, src, size, dependency, completion, flags);
  return abce_device_submit(handle, &request, false);
}

// abce_device_submit_linear() for exactly one calling lane per wave (several
// waves may call at once). INVALID_ARGUMENT if more lanes of a wave call it.
static inline abce_status_t abce_device_submit_linear_one_lane(
    const abce_device_handle_t* handle, void* dst, const void* src, uint64_t size,
    const abce_device_wait_t* dependency, const abce_device_signal_t* completion, uint32_t flags) {
  const abce_device_request_t request =
      abce_device_linear_request(dst, src, size, dependency, completion, flags);
  return abce_device_submit(handle, &request, true);
}

// abce_device_submit_linear() with a second completion, applied right after
// |completion| (required) under the same reservation and publish: one submit
// where a copy then a standalone signal would be two. For a copy that is
// announced to its destination and to its source side at once, say. Both land
// once the copy's writes have, the second after the first.
static inline abce_status_t abce_device_submit_linear_two_completions(
    const abce_device_handle_t* handle, void* dst, const void* src, uint64_t size,
    const abce_device_wait_t* dependency, const abce_device_signal_t* completion,
    const abce_device_signal_t* second_completion, uint32_t flags) {
  abce_device_request_t request =
      abce_device_linear_request(dst, src, size, dependency, completion, flags);
  request.second_completion = second_completion;
  return abce_device_submit(handle, &request, false);
}

// abce_device_submit_linear_two_completions() for exactly one calling lane per
// wave.
static inline abce_status_t abce_device_submit_linear_two_completions_one_lane(
    const abce_device_handle_t* handle, void* dst, const void* src, uint64_t size,
    const abce_device_wait_t* dependency, const abce_device_signal_t* completion,
    const abce_device_signal_t* second_completion, uint32_t flags) {
  abce_device_request_t request =
      abce_device_linear_request(dst, src, size, dependency, completion, flags);
  request.second_completion = second_completion;
  return abce_device_submit(handle, &request, true);
}

// Writes the low |size| (1, 2, 4 or 8) bytes of |value|, carried in the packet,
// to |dst| aligned to min(size, 4); no source buffer. 4 and 8 bytes are one
// WRITE_LINEAR, 1 and 2 one byte CONST_FILL per byte; bytes around them are
// untouched. It is data, not a completion: on gfx125+ it may land before
// earlier copies, so flag readiness with a signal. Otherwise as
// abce_device_submit_linear().
static inline abce_status_t abce_device_submit_write(const abce_device_handle_t* handle,
                                                     void* dst, uint64_t value, uint32_t size,
                                                     const abce_device_wait_t* dependency,
                                                     const abce_device_signal_t* completion,
                                                     uint32_t flags) {
  const abce_device_request_t request =
      abce_device_write_request(dst, value, size, dependency, completion, flags);
  return abce_device_submit(handle, &request, false);
}

// abce_device_submit_write() for exactly one calling lane per wave.
static inline abce_status_t abce_device_submit_write_one_lane(
    const abce_device_handle_t* handle, void* dst, uint64_t value, uint32_t size,
    const abce_device_wait_t* dependency, const abce_device_signal_t* completion, uint32_t flags) {
  const abce_device_request_t request =
      abce_device_write_request(dst, value, size, dependency, completion, flags);
  return abce_device_submit(handle, &request, true);
}

// Queues |completion| (required) alone: it lands once every earlier submission
// on the ring has, which drains the ring for the waiter. Otherwise as
// abce_device_submit_linear().
static inline abce_status_t abce_device_submit_signal(const abce_device_handle_t* handle,
                                                     const abce_device_signal_t* completion,
                                                     const abce_device_wait_t* dependency,
                                                     uint32_t flags) {
  const abce_device_request_t request =
      abce_device_request(ABCE_DEVICE_SUBMIT_SIGNAL, dependency, completion, flags);
  return abce_device_submit(handle, &request, false);
}

// abce_device_submit_signal() for exactly one calling lane per wave.
static inline abce_status_t abce_device_submit_signal_one_lane(
    const abce_device_handle_t* handle, const abce_device_signal_t* completion,
    const abce_device_wait_t* dependency, uint32_t flags) {
  const abce_device_request_t request =
      abce_device_request(ABCE_DEVICE_SUBMIT_SIGNAL, dependency, completion, flags);
  return abce_device_submit(handle, &request, true);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#if defined(ABCE_DEVICE_FORCE_HOST_DEVICE_)
#pragma clang force_cuda_host_device end
#undef ABCE_DEVICE_FORCE_HOST_DEVICE_
#endif  // ABCE_DEVICE_FORCE_HOST_DEVICE_

#endif  // ABCE_DEVICE_H_
