/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Accelerated Blit Copy Engine (ABCE) — per-engine frame composition.

#ifndef ABCE_FRAME_H_
#define ABCE_FRAME_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "abce_builder.h"
#include "abce_config.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Coordination word
//===----------------------------------------------------------------------===//

// Start gate for a gated fan-out, held in bit 62 of the coordination scratch
// word (see abce_signal_ref_t::coordination_scratch) whose low 32 bits are the
// fan-in counter. The gate is deliberately not in the signal's value: a gate bit
// there would make the signal transiently hold something that is not a legal
// signal state, visible to any client or tool that reads it mid-flight. Bit 62
// rather than 63 so the word stays positive when read as a signed 64-bit value,
// which keeps it printable by the same tools that dump signal values.
#define ABCE_FAN_OUT_START_GATE (((uint64_t)1) << 62)

//===----------------------------------------------------------------------===//
// abce_signal_ref_t
//===----------------------------------------------------------------------===//

// Reference to a signal's device-visible state.
//
// A thin adapter that carries the fields ABCE actually touches on a signal: the
// value location, the completion value, the optional interrupt mailbox, and the
// scratch word used to coordinate a fan-out.
typedef struct abce_signal_ref_t {
  void* value;  // 64-bit signal value (device-visible).
  // Value after the final hardware completion update. FENCE writes it directly;
  // ATOMIC decrements from completion_value + 1, so a client using the atomic
  // path arms the signal to that pending value before submitting.
  uint64_t completion_value;
  void* event_mailbox;  // non-null => emit interrupt after completion.
  uint32_t event_id;
  // Device-visible 64-bit scratch word, private to ABCE, holding the fan-in
  // counter (low 32 bits) and the start gate (bit 62) of a gated fan-out. The
  // SDMA engines poll it and atomically update it, so it must live in memory
  // they can reach. Required only for a fan-out whose coordinator drains fan-in;
  // single-engine plans and fan-outs that complete through their own body
  // signals never touch it. abce_hsa_signal_ref() points it at the signal
  // object's own reserved words, so it costs no separate allocation. The
  // protocol drains the word back to zero (the coordinator subtracts the gate,
  // each frame decrements the count), so a completed plan leaves no residue.
  void* coordination_scratch;
} abce_signal_ref_t;

static inline abce_signal_ref_t abce_signal_ref_make(void* value, uint64_t completion_value,
                                                     void* event_mailbox, uint32_t event_id,
                                                     void* coordination_scratch) {
  abce_signal_ref_t ref;
  ref.value = value;
  ref.completion_value = completion_value;
  ref.event_mailbox = event_mailbox;
  ref.event_id = event_id;
  ref.coordination_scratch = coordination_scratch;
  return ref;
}

//===----------------------------------------------------------------------===//
// abce_timestamp_surface_t
//===----------------------------------------------------------------------===//

// Optional profiling timestamp surface holding two device-visible 64-bit SDMA
// global-clock slots. The coordinator frame writes start in its prologue
// (before the copy body) and end in its epilogue (after the body, before
// completion is signalled, so a reader that observes completion also sees a
// valid end). Null slots disable profiling; it is all-or-nothing. Passed on
// abce_copy_metadata_t the same way as the completion abce_signal_ref_t.
typedef struct abce_timestamp_surface_t {
  void* start_value;
  void* end_value;
} abce_timestamp_surface_t;

static inline bool abce_timestamp_surface_enabled(const abce_timestamp_surface_t* surface) {
  return surface->start_value != NULL && surface->end_value != NULL;
}

//===----------------------------------------------------------------------===//
// abce_dep_signal_t
//===----------------------------------------------------------------------===//

// A dependency signal a frame must wait on before proceeding.
typedef struct abce_dep_signal_t {
  void* value;  // 64-bit signal value location (device-visible).
  // Value observed at record time. Used to omit an already-satisfied wait and,
  // on non-gfx125plus, determine whether the high 32 bits also need a poll.
  uint64_t observed_value;
  uint64_t reference_value;  // value the signal must reach before proceeding.
} abce_dep_signal_t;

static inline abce_dep_signal_t abce_dep_signal_make(void* value, uint64_t observed,
                                                     uint64_t reference) {
  abce_dep_signal_t dep;
  dep.value = value;
  dep.observed_value = observed;
  dep.reference_value = reference;
  return dep;
}

//===----------------------------------------------------------------------===//
// abce_op_kind_t
//===----------------------------------------------------------------------===//

// Copy primitive. Clients typically use LINEAR / MULTICAST / FILL; BROADCAST is
// mainly produced internally (small 2-dst copies on non-gfx125plus) but may be
// requested directly for an explicit 2-destination broadcast.
//   * SWAP      — bidirectional exchange of two buffers (dst<->src); fusible on
//                 gfx125plus, plain packet elsewhere. Endpoint sizes may differ
//                 (size / size2); a distinct size2 is honored only on the fused
//                 path.
//   * INDIRECT  — gfx125plus-only gather/scatter copy whose src and/or dst
//                 address is read indirectly; always emitted via the
//                 wait/signal packet.
//   * COPY_RECT — 2D/3D strided sub-window copy (abce_pitched_ptr_t + offsets +
//                 range); not fusible (no fused rect packet).
typedef enum abce_op_kind_t {
  ABCE_OP_KIND_LINEAR = 0,
  ABCE_OP_KIND_MULTICAST = 1,
  ABCE_OP_KIND_BROADCAST = 2,
  ABCE_OP_KIND_SWAP = 3,
  ABCE_OP_KIND_INDIRECT = 4,
  ABCE_OP_KIND_COPY_RECT = 5,
  ABCE_OP_KIND_FILL = 6,
} abce_op_kind_t;

// Geometry for an ABCE_OP_KIND_COPY_RECT op (caller-owned; borrowed by
// abce_copy_op_t / abce_engine_op_t like the multicast `dsts` array). Mirrors
// abce_builder_build_copy_rect().
typedef struct abce_copy_rect_desc_t {
  abce_pitched_ptr_t dst;
  abce_dim3_t dst_offset;
  abce_pitched_ptr_t src;
  abce_dim3_t src_offset;
  abce_dim3_t range;
} abce_copy_rect_desc_t;

//===----------------------------------------------------------------------===//
// abce_coherency_control_t
//===----------------------------------------------------------------------===//

typedef union abce_coherency_control_t {
  struct {
    uint32_t emit_hdp_flush : 1;
    uint32_t emit_gcr : 1;
    uint32_t reserved : 30;
  } flags;
  uint32_t value;
} abce_coherency_control_t;

// Both triggers default on; zero-initializing the union does NOT give the
// default.
static inline abce_coherency_control_t abce_coherency_control_default(void) {
  abce_coherency_control_t coherency;
  coherency.value = 0;
  coherency.flags.emit_hdp_flush = 1;
  coherency.flags.emit_gcr = 1;
  return coherency;
}

//===----------------------------------------------------------------------===//
// abce_copy_metadata_t
//===----------------------------------------------------------------------===//

// A whole batch's submission parameters (one completion signal).
typedef struct abce_copy_metadata_t {
  const abce_dep_signal_t* deps;  // dependencies gating the whole batch.
  uint32_t num_deps;

  abce_signal_ref_t out;  // the single completion signal (+ optional interrupt).

  // Per-batch coherency packet triggers. A trigger is additionally gated by the
  // corresponding platform capability / driver-ownership policy.
  abce_coherency_control_t coherency;

  // Restrict the batch to this set of registered engines (bit i => engine i).
  // 0 selects all registered engines.
  uint64_t engine_mask;

  // Cap the number of engines used (0 = no cap). Applied after the mask.
  uint32_t max_engines;

  // ROCr-compatible multi-linear strategy. AUTOMATIC keeps a batch whose TOTAL
  // bytes are at or below ABCE_LINEAR_B2B_MAX_TOTAL_DEFAULT back-to-back on one
  // engine, and fans larger batches out. The threshold is on the total, not on
  // any one entry: see the measurements on that constant in abce_host.h for why
  // a per-copy window mis-sizes every batch shape it was not tuned on.
  abce_linear_batch_mode_t linear_batch_mode;

  // gfx125plus multicast selection. AUTOMATIC uses multicast through 256 KiB and
  // fans out larger operations across engines.
  abce_multicast_mode_t multicast_mode;

  // Optional profiling timestamps. When set, the coordinator frame records the
  // global clock into timestamps.start_value in its prologue and
  // timestamps.end_value in its epilogue (before the completion signal), just
  // like a fused/single copy still carries a prologue + epilogue for this.
  abce_timestamp_surface_t timestamps;

  // Optional 32-bit slot for an execution descriptor (see abce_host.h): which
  // hardware engines ran the batch, what kind of transfer it was, and the
  // coordinator's engine class. Submit stamps it with a plain host store once
  // the engines are final; hardware never touches it, so unlike timestamps it
  // costs no packets and needs only 4-byte alignment. Null disables it.
  //
  // Held as an address rather than a field so ABCE commits to no particular
  // signal layout: point it at whatever a reader of this signal will look at.
  void* execution_descriptor;

  // Prefer fused wait/signal copy packets when the device supports them. This
  // enables direct completion for single-frame work and integrated fan-out
  // synchronization for multi-frame work. Ignored on non-fused hardware.
  bool prefer_fused;
} abce_copy_metadata_t;

// Zero-initializing this struct is NOT the default configuration: both
// coherency triggers and prefer_fused default on. Always initialize through
// here.
static inline void abce_copy_metadata_initialize(abce_copy_metadata_t* out_metadata) {
  memset(out_metadata, 0, sizeof(*out_metadata));
  out_metadata->coherency = abce_coherency_control_default();
  out_metadata->linear_batch_mode = ABCE_LINEAR_BATCH_MODE_AUTOMATIC;
  out_metadata->multicast_mode = ABCE_MULTICAST_MODE_AUTOMATIC;
  out_metadata->prefer_fused = true;
}

//===----------------------------------------------------------------------===//
// abce_engine_op_t
//===----------------------------------------------------------------------===//

// One decomposed copy placed on a single engine. Produced by the orchestrator's
// map phase (lives on the stack there, never exposed in the public
// abce_plan_t) and consumed by the frame composer to size and emit packets.
typedef struct abce_engine_op_t {
  const void* src;
  void* dst;
  void* const* dsts;
  const abce_copy_rect_desc_t* rect;
  size_t size;
  size_t size2;  // secondary size (SWAP endpoint B); 0 => same as size.
  uint32_t engine;
  uint32_t num_dsts;
  uint32_t fill_value;
  abce_op_kind_t kind;
  uint8_t num_ranked_engines;
  bool indirect_src;
  bool indirect_dst;
  bool has_engine_preference;
  bool fused;
  uint32_t ranked_engines[ABCE_MAX_ENGINE_CHOICES];
} abce_engine_op_t;

//===----------------------------------------------------------------------===//
// abce_frame_job_t
//===----------------------------------------------------------------------===//

// Everything the frame composer needs to size / emit one frame.
//
// A frame is the group of packets written to a single engine's ring in one
// reservation. `ops` points at the whole batch's decomposed ops; the composer
// filters to those matching `engine` (there are `frame_num_ops` of them). The
// coordinator frame (`coordinator == true`) additionally carries the prologue +
// epilogue. For a fan-out (`multi == true`) the frames coordinate through
// `coordination_word`: when ordering work requires it, non-coordinator frames
// wait for the start gate (bit 62) to clear before their first body. Every
// participating frame without a start gate, or every non-coordinator frame with
// one, decrements the word after its last body, and the coordinator waits for
// that count to drain.
typedef struct abce_frame_job_t {
  const abce_engine_op_t* ops;  // all ops in the batch (filtered by `engine`).
  uint32_t num_ops;             // total ops in `ops`.
  const abce_copy_metadata_t* metadata;

  uint32_t engine;         // this frame's engine / ring index.
  uint32_t frame_num_ops;  // ops on THIS frame (for the last-body signal).
  bool coordinator;        // carries the prologue + epilogue.
  bool multi;              // batch fans out across engines.
  // Whether non-coordinator frames must wait for the coordinator to release the
  // bit-62 start gate. gfx125+ fused fan-out without dependencies can start
  // directly and has every frame signal the fan-in count.
  bool start_gate_required;
  // Whether the coordinator must drain fan-in and perform completion work after
  // its body. A dependency-free gfx125+ fused fan-out can instead let the final
  // body SIGNAL reach the completion value directly.
  bool epilogue_required;

  // The 64-bit word this plan's start gate and fan-in counter live in.
  //
  // It is the private coordination scratch
  // (abce_signal_ref_t::coordination_scratch) whenever the coordinator drains
  // fan-in and then writes completion itself. It is the completion signal's own
  // value in the two cases where the body signals *are* the completion
  // transitions and nothing drains them: a single-frame direct fused
  // completion, and a fan-out with no epilogue. Those cases only ever decrement
  // toward the completion value, so they need no scratch and never leave the
  // signal in an illegal state.
  void* coordination_word;
} abce_frame_job_t;

//===----------------------------------------------------------------------===//
// abce_frame_composer_t
//===----------------------------------------------------------------------===//

// Composes SDMA builder packets into per-engine frames.
//
// Stateless apart from the builder + a few device capability flags. Owns the
// size/emit pairs (prologue_bytes/emit_prologue, body_bytes/emit_body, ...) so
// the byte count and the packet stream can never drift apart. Platform
// capabilities are supplied at initialization; packet-layout capabilities come
// from the builder.
typedef struct abce_frame_composer_t {
  const abce_builder_t* builder;
  abce_platform_caps_t caps;
  bool is_gfx125plus;
} abce_frame_composer_t;

static inline void abce_frame_composer_initialize(const abce_builder_t* builder,
                                                  abce_platform_caps_t caps,
                                                  abce_frame_composer_t* out_composer) {
  out_composer->builder = builder;
  out_composer->caps = caps;
  out_composer->is_gfx125plus = abce_builder_is_gfx125plus(builder);
}

//===----------------------------------------------------------------------===//
// Per-command sizes (bytes)
//===----------------------------------------------------------------------===//

#define ABCE_FRAME_POLL32_BYTES ((uint32_t)sizeof(SDMA_PKT_POLL_REGMEM))
#define ABCE_FRAME_POLL64_BYTES ((uint32_t)sizeof(SDMA_PKT_POLL_MEM_64B_GFX125PLUS))
#define ABCE_FRAME_FENCE32_BYTES ((uint32_t)sizeof(SDMA_PKT_FENCE))
#define ABCE_FRAME_FENCE64_BYTES ((uint32_t)sizeof(SDMA_PKT_FENCE_64B_GFX125PLUS))
#define ABCE_FRAME_ATOMIC_BYTES ((uint32_t)sizeof(SDMA_PKT_ATOMIC))
#define ABCE_FRAME_TIMESTAMP_BYTES ((uint32_t)sizeof(SDMA_PKT_TIMESTAMP))
#define ABCE_FRAME_TRAP_BYTES ((uint32_t)sizeof(SDMA_PKT_TRAP))
#define ABCE_FRAME_FLUSH_BYTES ((uint32_t)sizeof(SDMA_PKT_HDP_FLUSH))
#define ABCE_FRAME_COPY_LINEAR_BYTES ((uint32_t)sizeof(SDMA_PKT_COPY_LINEAR))
#define ABCE_FRAME_FILL_BYTES ((uint32_t)sizeof(SDMA_PKT_CONSTANT_FILL))

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

// Whether the body at |body_idx| of |job| carries the start-gate wait and the
// completion / fan-in signal, and which word each one targets. Derived once
// here so the sizing pass and the emit pass can never disagree about which
// bodies are synchronizing.
typedef struct abce_body_sync_t {
  bool wait;
  bool signal;
  void* wait_addr;
  void* signal_addr;
} abce_body_sync_t;

static inline abce_body_sync_t abce_body_sync_for(const abce_frame_job_t* job, uint32_t body_idx) {
  const bool wait_for_start = job->multi && job->start_gate_required && !job->coordinator;
  const bool signals_completion_directly = !job->multi && !job->epilogue_required;
  const bool contributes_to_fan_in =
      job->multi && (!job->coordinator || !job->start_gate_required);
  abce_body_sync_t sync;
  sync.wait_addr = NULL;
  sync.signal_addr = NULL;
  sync.wait = wait_for_start && (body_idx == 0);
  sync.signal = (signals_completion_directly || contributes_to_fan_in) &&
                (body_idx == job->frame_num_ops - 1);
  // Both roles read the same word for a given plan: abce_frame_job_t's
  // coordination_word already resolves to the scratch or to the signal value
  // depending on whether anything drains the count.
  if (sync.wait) sync.wait_addr = job->coordination_word;
  if (sync.signal) sync.signal_addr = job->coordination_word;
  return sync;
}

static inline bool abce_frame_should_emit_gcr(const abce_frame_composer_t* composer) {
  return abce_builder_requires_gcr(composer->builder) && !composer->caps.driver_manages_gcr;
}

static inline uint32_t abce_frame_gcr_bytes(const abce_frame_composer_t* composer) {
  if (!abce_frame_should_emit_gcr(composer)) return 0;
  return composer->is_gfx125plus ? (uint32_t)sizeof(SDMA_PKT_GCR_GFX125PLUS)
                                 : (uint32_t)sizeof(SDMA_PKT_GCR);
}

static inline uint32_t abce_frame_poll_bytes(const abce_frame_composer_t* composer) {
  return composer->is_gfx125plus ? ABCE_FRAME_POLL64_BYTES : ABCE_FRAME_POLL32_BYTES;
}

static inline uint32_t abce_frame_multicast_pkt_bytes(uint32_t num_dsts) {
  return (5u + 2u * num_dsts) * (uint32_t)sizeof(uint32_t);
}

// Effective secondary size for a two-region op: the caller's size2 when
// available, otherwise size (symmetric). Only SWAP's fused wait/signal path
// consumes a distinct value; the single-COUNT plain swap packet is
// symmetric-only.
static inline size_t abce_engine_op_size2(const abce_engine_op_t* op) {
  return op->size2 ? op->size2 : op->size;
}

//===----------------------------------------------------------------------===//
// Sizing
//===----------------------------------------------------------------------===//

static inline size_t abce_frame_dep_waits_bytes(const abce_frame_composer_t* composer,
                                                const abce_dep_signal_t* deps, uint32_t num_deps) {
  size_t bytes = 0;
  for (uint32_t dep_idx = 0; dep_idx < num_deps; ++dep_idx) {
    if (deps[dep_idx].observed_value == deps[dep_idx].reference_value) continue;
    bytes += composer->is_gfx125plus ? ABCE_FRAME_POLL64_BYTES : ABCE_FRAME_POLL32_BYTES;
    if (!composer->is_gfx125plus &&
        (deps[dep_idx].observed_value >> 32) != (deps[dep_idx].reference_value >> 32))
      bytes += ABCE_FRAME_POLL32_BYTES;
  }
  return bytes;
}

// Rect tiling can fail the same way emission does; a failure is reported as a
// zero payload and surfaces when the orchestrator re-runs the emit.
static inline size_t abce_frame_payload_bytes(const abce_frame_composer_t* composer,
                                              const abce_engine_op_t* op) {
  const abce_builder_t* builder = composer->builder;
  switch (op->kind) {
    case ABCE_OP_KIND_LINEAR:
      return (size_t)abce_builder_num_copy_packets(builder, op->size) *
             ABCE_FRAME_COPY_LINEAR_BYTES;
    case ABCE_OP_KIND_MULTICAST:
      return (size_t)abce_builder_num_multicast_packets(builder, op->size) *
             abce_frame_multicast_pkt_bytes(op->num_dsts);
    case ABCE_OP_KIND_BROADCAST:
      return (size_t)abce_builder_num_broadcast_packets(builder, op->size) *
             sizeof(SDMA_PKT_COPY_LINEAR_BROADCAST);
    case ABCE_OP_KIND_SWAP:
      return (size_t)abce_builder_num_swap_packets(builder, op->size) *
             sizeof(SDMA_PKT_COPY_LINEAR_SWAP);
    case ABCE_OP_KIND_COPY_RECT: {
      uint32_t num_packets = 0;
      if (!abce_status_is_ok(abce_builder_num_rect_packets(
              builder, &op->rect->dst, &op->rect->dst_offset, &op->rect->src,
              &op->rect->src_offset, &op->rect->range, &num_packets)))
        return 0;
      return (size_t)num_packets * sizeof(SDMA_PKT_COPY_LINEAR_RECT);
    }
    case ABCE_OP_KIND_INDIRECT:
      return sizeof(SDMA_PKT_COPY_LINEAR_WAITSIGNAL_INDIRECT_GFX125PLUS);
    case ABCE_OP_KIND_FILL:
      return (size_t)abce_builder_num_fill_packets(builder, op->size / sizeof(uint32_t)) *
             ABCE_FRAME_FILL_BYTES;
  }
  return 0;
}

static inline size_t abce_frame_fused_bytes(const abce_frame_composer_t* composer,
                                            const abce_engine_op_t* op, bool wait, bool signal) {
  const abce_builder_t* builder = composer->builder;
  const uint32_t wait_dwords = wait ? 7u : 0u;
  const uint32_t signal_dwords = signal ? 5u : 0u;
  if (op->kind == ABCE_OP_KIND_INDIRECT)
    return (size_t)(1u + wait_dwords + 6u + signal_dwords) * sizeof(uint32_t);

  if (op->kind == ABCE_OP_KIND_LINEAR || op->kind == ABCE_OP_KIND_SWAP) {
    const uint64_t num_packets =
        op->kind == ABCE_OP_KIND_SWAP
            ? abce_builder_num_wait_signal_swap_packets(builder, op->size, abce_engine_op_size2(op))
            : abce_builder_num_copy_packets(builder, op->size);
    const uint64_t total_dwords = num_packets * (1u + 6u) + wait_dwords + signal_dwords;
    return (size_t)total_dwords * sizeof(uint32_t);
  }
  if (op->kind == ABCE_OP_KIND_MULTICAST) {
    const uint64_t packet_core_dwords = 1u + 4u + 2u * (uint64_t)op->num_dsts;
    const uint64_t total_dwords =
        abce_builder_num_multicast_packets(builder, op->size) * packet_core_dwords + wait_dwords +
        signal_dwords;
    return (size_t)total_dwords * sizeof(uint32_t);
  }
  return 0;
}

static inline size_t abce_frame_body_bytes(const abce_frame_composer_t* composer,
                                           const abce_engine_op_t* op, bool wait_here,
                                           bool signal_here) {
  if (op->fused) return abce_frame_fused_bytes(composer, op, wait_here, signal_here);
  size_t bytes = 0;
  if (wait_here) bytes += abce_frame_poll_bytes(composer);
  bytes += abce_frame_payload_bytes(composer, op);
  if (signal_here) bytes += ABCE_FRAME_ATOMIC_BYTES;
  return bytes;
}

static inline size_t abce_frame_prologue_bytes(const abce_frame_composer_t* composer,
                                               const abce_copy_metadata_t* metadata, bool multi,
                                               bool start_gate_required) {
  size_t bytes = abce_frame_dep_waits_bytes(composer, metadata->deps, metadata->num_deps);
  if (abce_timestamp_surface_enabled(&metadata->timestamps)) bytes += ABCE_FRAME_TIMESTAMP_BYTES;
  if (composer->caps.emit_hdp_flush && metadata->coherency.flags.emit_hdp_flush)
    bytes += ABCE_FRAME_FLUSH_BYTES;
  if (metadata->coherency.flags.emit_gcr) bytes += abce_frame_gcr_bytes(composer);
  if (multi && start_gate_required) bytes += ABCE_FRAME_ATOMIC_BYTES;
  return bytes;
}

static inline uint32_t abce_frame_completion_bytes(const abce_frame_composer_t* composer,
                                                   const abce_signal_ref_t* target) {
  if (composer->caps.device_atomic_support) return ABCE_FRAME_ATOMIC_BYTES;
  if (composer->is_gfx125plus) return ABCE_FRAME_FENCE64_BYTES;
  return (target->completion_value > UINT32_MAX) ? 2 * ABCE_FRAME_FENCE32_BYTES
                                                 : ABCE_FRAME_FENCE32_BYTES;
}

static inline size_t abce_frame_epilogue_bytes(const abce_frame_composer_t* composer,
                                               const abce_copy_metadata_t* metadata, bool multi) {
  size_t bytes = 0;
  if (multi) bytes += abce_frame_poll_bytes(composer);
  if (metadata->coherency.flags.emit_gcr) bytes += abce_frame_gcr_bytes(composer);
  if (abce_timestamp_surface_enabled(&metadata->timestamps)) bytes += ABCE_FRAME_TIMESTAMP_BYTES;
  bytes += abce_frame_completion_bytes(composer, &metadata->out);
  if (metadata->out.event_mailbox) bytes += ABCE_FRAME_FENCE32_BYTES + ABCE_FRAME_TRAP_BYTES;
  return bytes;
}

// Total packet bytes this frame will occupy.
static inline size_t abce_frame_composer_frame_bytes(const abce_frame_composer_t* composer,
                                                     const abce_frame_job_t* job) {
  const abce_copy_metadata_t* metadata = job->metadata;
  size_t bytes = 0;
  if (job->coordinator)
    bytes += abce_frame_prologue_bytes(composer, metadata, job->multi, job->start_gate_required);
  uint32_t body_idx = 0;
  for (uint32_t op_idx = 0; op_idx < job->num_ops; ++op_idx) {
    if (job->ops[op_idx].engine != job->engine) continue;
    const abce_body_sync_t sync = abce_body_sync_for(job, body_idx);
    bytes += abce_frame_body_bytes(composer, &job->ops[op_idx], sync.wait, sync.signal);
    body_idx++;
  }
  if (job->coordinator && job->epilogue_required)
    bytes += abce_frame_epilogue_bytes(composer, metadata, job->multi);
  return bytes;
}

//===----------------------------------------------------------------------===//
// Emission
//===----------------------------------------------------------------------===//

static inline size_t abce_frame_emit_dep_waits(const abce_frame_composer_t* composer, char* out,
                                               const abce_dep_signal_t* deps, uint32_t num_deps) {
  char* cursor = out;
  for (uint32_t dep_idx = 0; dep_idx < num_deps; ++dep_idx) {
    if (deps[dep_idx].observed_value == deps[dep_idx].reference_value) continue;
    if (composer->is_gfx125plus) {
      abce_builder_build_poll_64b(composer->builder, cursor, deps[dep_idx].value,
                                  deps[dep_idx].reference_value, UINT64_MAX);
      cursor += ABCE_FRAME_POLL64_BYTES;
    } else {
      uint32_t* words = (uint32_t*)deps[dep_idx].value;
      if ((deps[dep_idx].observed_value >> 32) != (deps[dep_idx].reference_value >> 32)) {
        abce_builder_build_poll(composer->builder, cursor, &words[1],
                                (uint32_t)(deps[dep_idx].reference_value >> 32), UINT32_MAX);
        cursor += ABCE_FRAME_POLL32_BYTES;
      }
      abce_builder_build_poll(composer->builder, cursor, &words[0],
                              (uint32_t)deps[dep_idx].reference_value, UINT32_MAX);
      cursor += ABCE_FRAME_POLL32_BYTES;
    }
  }
  return (size_t)(cursor - out);
}

static inline size_t abce_frame_emit_prologue(const abce_frame_composer_t* composer, char* out,
                                              const abce_copy_metadata_t* metadata, bool multi,
                                              bool start_gate_required, void* coordination_word) {
  char* cursor = out;
  cursor += abce_frame_emit_dep_waits(composer, cursor, metadata->deps, metadata->num_deps);
  if (abce_timestamp_surface_enabled(&metadata->timestamps)) {
    abce_builder_build_get_global_timestamp(composer->builder, cursor,
                                            metadata->timestamps.start_value);
    cursor += ABCE_FRAME_TIMESTAMP_BYTES;
  }
  if (composer->caps.emit_hdp_flush && metadata->coherency.flags.emit_hdp_flush) {
    abce_builder_build_hdp_flush(composer->builder, cursor);
    cursor += ABCE_FRAME_FLUSH_BYTES;
  }
  if (metadata->coherency.flags.emit_gcr && abce_frame_should_emit_gcr(composer)) {
    abce_builder_build_gcr(composer->builder, cursor, /*invalidate=*/true);
    cursor += abce_frame_gcr_bytes(composer);
  }
  if (multi && start_gate_required) {
    abce_builder_build_atomic_add(composer->builder, cursor, coordination_word,
                                  (uint64_t)0 - ABCE_FAN_OUT_START_GATE);
    cursor += ABCE_FRAME_ATOMIC_BYTES;
  }
  return (size_t)(cursor - out);
}

static inline size_t abce_frame_write_start_gate_poll(const abce_frame_composer_t* composer,
                                                      char* out, void* coordination_word) {
  if (composer->is_gfx125plus) {
    abce_builder_build_poll_64b(composer->builder, out, coordination_word, /*reference=*/0,
                                ABCE_FAN_OUT_START_GATE);
    return ABCE_FRAME_POLL64_BYTES;
  }
  // Without a 64-bit poll, watch bit 30 of the high word — the same bit 62.
  uint32_t* coordination_words = (uint32_t*)coordination_word;
  const uint32_t start_gate_high_word = ((uint32_t)1) << 30;
  abce_builder_build_poll(composer->builder, out, &coordination_words[1], /*reference=*/0,
                          start_gate_high_word);
  return ABCE_FRAME_POLL32_BYTES;
}

// Rect emission appends tiles sequentially into the frame cursor.
typedef struct abce_frame_rect_cursor_t {
  char* cursor;
} abce_frame_rect_cursor_t;

static inline void* abce_frame_rect_append(void* user_data, size_t packet_bytes) {
  abce_frame_rect_cursor_t* state = (abce_frame_rect_cursor_t*)user_data;
  char* slot = state->cursor;
  state->cursor += packet_bytes;
  return slot;
}

static inline size_t abce_frame_emit_body(const abce_frame_composer_t* composer, char* out,
                                          const abce_engine_op_t* op,
                                          const abce_body_sync_t* sync) {
  char* cursor = out;
  const abce_builder_t* builder = composer->builder;

  if (op->fused) {
    abce_wait_signal_params_t params = abce_wait_signal_none();
    params.wait_addr = sync->wait_addr;
    params.signal_addr = sync->signal_addr;
    params.wait_reference = 0;
    params.wait_mask = ABCE_FAN_OUT_START_GATE;

    if (op->kind == ABCE_OP_KIND_INDIRECT) {
      abce_builder_build_wait_signal_indirect_copy(builder, cursor, op->dst, op->src, op->size,
                                                   op->indirect_src, op->indirect_dst, &params);
    } else if (op->kind == ABCE_OP_KIND_LINEAR) {
      params.boundary_wait_signal = true;
      abce_builder_build_wait_signal_copy(builder, cursor, op->dst, op->src, op->size, &params);
    } else if (op->kind == ABCE_OP_KIND_SWAP) {
      params.boundary_wait_signal = true;
      abce_builder_build_wait_signal_swap(builder, cursor, op->dst, (void*)(uintptr_t)op->src,
                                          op->size, abce_engine_op_size2(op), &params);
    } else if (op->kind == ABCE_OP_KIND_MULTICAST) {
      params.boundary_wait_signal = true;
      abce_builder_build_multicast_wait_signal_copy(builder, cursor, op->dsts, op->num_dsts,
                                                    op->src, op->size, &params);
    }
    return abce_frame_fused_bytes(composer, op, sync->wait, sync->signal);
  }

  if (sync->wait) cursor += abce_frame_write_start_gate_poll(composer, cursor, sync->wait_addr);
  switch (op->kind) {
    case ABCE_OP_KIND_LINEAR:
      abce_builder_build_copy(builder, cursor, op->dst, op->src, op->size);
      break;
    case ABCE_OP_KIND_MULTICAST:
      abce_builder_build_multicast_copy(builder, cursor, op->dsts, op->num_dsts, op->src, op->size);
      break;
    case ABCE_OP_KIND_BROADCAST:
      abce_builder_build_broadcast_copy(builder, cursor, op->dsts[0], op->dsts[1], op->src,
                                        op->size);
      break;
    case ABCE_OP_KIND_SWAP:
      abce_builder_build_swap_copy(builder, cursor, op->dst, (void*)(uintptr_t)op->src, op->size);
      break;
    case ABCE_OP_KIND_COPY_RECT: {
      abce_frame_rect_cursor_t rect_state;
      rect_state.cursor = cursor;
      (void)abce_builder_build_copy_rect(builder, abce_frame_rect_append, &rect_state,
                                         &op->rect->dst, &op->rect->dst_offset, &op->rect->src,
                                         &op->rect->src_offset, &op->rect->range);
      break;
    }
    case ABCE_OP_KIND_INDIRECT:
      break;
    case ABCE_OP_KIND_FILL:
      abce_builder_build_fill(builder, cursor, op->dst, op->fill_value,
                              op->size / sizeof(uint32_t));
      break;
  }
  cursor += abce_frame_payload_bytes(composer, op);
  if (sync->signal) {
    abce_builder_build_atomic_decrement(builder, cursor, sync->signal_addr);
    cursor += ABCE_FRAME_ATOMIC_BYTES;
  }
  return (size_t)(cursor - out);
}

// Wait for every participating frame to have decremented the fan-in counter.
// The counter is the low 32 bits of a word ABCE owns outright, so it drains to
// zero rather than to a value biased by the caller's completion value.
static inline size_t abce_frame_write_fan_in_poll(const abce_frame_composer_t* composer, char* out,
                                                  void* coordination_word) {
  if (composer->is_gfx125plus) {
    abce_builder_build_poll_64b(composer->builder, out, coordination_word, /*reference=*/0,
                                UINT32_MAX);
    return ABCE_FRAME_POLL64_BYTES;
  }
  abce_builder_build_poll(composer->builder, out, coordination_word, /*reference=*/0, UINT32_MAX);
  return ABCE_FRAME_POLL32_BYTES;
}

static inline size_t abce_frame_write_completion(const abce_frame_composer_t* composer, char* out,
                                                 const abce_signal_ref_t* target) {
  if (composer->caps.device_atomic_support) {
    abce_builder_build_atomic_decrement(composer->builder, out, target->value);
    return ABCE_FRAME_ATOMIC_BYTES;
  }
  if (composer->is_gfx125plus) {
    abce_builder_build_fence_64b(composer->builder, out, target->value, target->completion_value);
    return ABCE_FRAME_FENCE64_BYTES;
  }
  char* cursor = out;
  uint32_t* words = (uint32_t*)target->value;
  if (target->completion_value > UINT32_MAX) {
    abce_builder_build_fence(composer->builder, cursor, &words[1],
                             (uint32_t)(target->completion_value >> 32));
    cursor += ABCE_FRAME_FENCE32_BYTES;
  }
  abce_builder_build_fence(composer->builder, cursor, &words[0],
                           (uint32_t)target->completion_value);
  cursor += ABCE_FRAME_FENCE32_BYTES;
  return (size_t)(cursor - out);
}

static inline size_t abce_frame_write_mailbox(const abce_frame_composer_t* composer, char* out,
                                              const abce_signal_ref_t* target) {
  if (!target->event_mailbox) return 0;
  char* cursor = out;
  abce_builder_build_fence(composer->builder, cursor, (uint32_t*)target->event_mailbox,
                           target->event_id);
  cursor += ABCE_FRAME_FENCE32_BYTES;
  abce_builder_build_trap(composer->builder, cursor, target->event_id);
  cursor += ABCE_FRAME_TRAP_BYTES;
  return (size_t)(cursor - out);
}

static inline size_t abce_frame_emit_epilogue(const abce_frame_composer_t* composer, char* out,
                                              const abce_copy_metadata_t* metadata, bool multi,
                                              void* coordination_word) {
  char* cursor = out;
  if (multi) cursor += abce_frame_write_fan_in_poll(composer, cursor, coordination_word);
  if (metadata->coherency.flags.emit_gcr && abce_frame_should_emit_gcr(composer)) {
    abce_builder_build_gcr(composer->builder, cursor, /*invalidate=*/false);
    cursor += abce_frame_gcr_bytes(composer);
  }
  if (abce_timestamp_surface_enabled(&metadata->timestamps)) {
    abce_builder_build_get_global_timestamp(composer->builder, cursor,
                                            metadata->timestamps.end_value);
    cursor += ABCE_FRAME_TIMESTAMP_BYTES;
  }
  cursor += abce_frame_write_completion(composer, cursor, &metadata->out);
  cursor += abce_frame_write_mailbox(composer, cursor, &metadata->out);
  return (size_t)(cursor - out);
}

// Emits the whole frame into |buffer|, returning the bytes written. The buffer
// must be at least abce_frame_composer_frame_bytes() long and zeroed.
static inline size_t abce_frame_composer_emit_frame(const abce_frame_composer_t* composer,
                                                    char* buffer, const abce_frame_job_t* job) {
  const abce_copy_metadata_t* metadata = job->metadata;
  char* cursor = buffer;

  if (job->coordinator)
    cursor += abce_frame_emit_prologue(composer, cursor, metadata, job->multi,
                                       job->start_gate_required, job->coordination_word);

  uint32_t body_idx = 0;
  for (uint32_t op_idx = 0; op_idx < job->num_ops; ++op_idx) {
    if (job->ops[op_idx].engine != job->engine) continue;
    const abce_body_sync_t sync = abce_body_sync_for(job, body_idx);
    cursor += abce_frame_emit_body(composer, cursor, &job->ops[op_idx], &sync);
    body_idx++;
  }

  if (job->coordinator && job->epilogue_required)
    cursor += abce_frame_emit_epilogue(composer, cursor, metadata, job->multi,
                                       job->coordination_word);
  return (size_t)(cursor - buffer);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ABCE_FRAME_H_
