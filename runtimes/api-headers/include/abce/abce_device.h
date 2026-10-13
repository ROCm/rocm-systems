/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Accelerated Blit Copy Engine (ABCE) — device-initiated copy submission.
//
// A GPU kernel hands this header one source, one destination and a pre-armed
// completion signal. It forms the SDMA packets into a device ring
// (abce_ring_device.h) and rings that ring's doorbell, so a copy is launched
// from a shader with no host round trip.
//
// SCOPE: one linear copy per call, and deliberately nothing else. No batching,
// no multi-destination fan-out, no dependency gating, no interrupt. One copy is
// one frame on one ring, which keeps the submitter a short scalar routine with
// no scratch and no LDS, and removes every reason for a coordinator: with a
// single frame the caller's arming is exact and that frame's one decrement
// retires the signal. The wide cases are the host planner's (abce_host.h).
//
// What it still decides is which ring. The host precomputes, from its own SDMA
// policy, a route per transfer class — host to this device, this device to
// host, local, to or from each peer, and one for a copy whose ends it cannot
// tell — so a kernel's copy lands on the ring the host path would have chosen
// and never on one the policy reserves for other traffic. A caller placing
// copies itself may name the ring instead.
//
// KERNEL USE: under HIP this header wraps itself and the ABCE headers it
// includes in `#pragma clang force_cuda_host_device`, which is what makes the C
// functions callable from a kernel. Include it before any other ABCE header in
// a HIP translation unit: one included first keeps its functions host-only, and
// calling them from a kernel then fails to compile.
//
// CONCURRENCY: the ring protocol is multi-producer, so several submitters —
// including the host orchestrator sharing a ring — may run concurrently, all
// through the ring's one abce_shared_ring_control_t. Any number of lanes may
// submit at once, but lanes of one wave take turns inside the submit; see
// abce_device_submit_linear().

#ifndef ABCE_DEVICE_H_
#define ABCE_DEVICE_H_

#if defined(__HIP__)
// Declared before the region below. Pulled in inside it these would become
// host+device and collide with the device versions, and the HIP runtime has to
// come first so the builders' memset/memcpy resolve to its device overloads.
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

// Bumped whenever abce_device_engine_set_t's layout or the submitter's contract
// changes, so a kernel built against an older header refuses a newer runtime's
// view instead of misreading it.
#define ABCE_DEVICE_ABI_VERSION 1u

// One ring per registered SDMA engine.
#define ABCE_DEVICE_MAX_ENGINES ABCE_MAX_ENGINES

// Agent ids a class route covers: KFD node ids, the numbering the host policy
// ranks engines by.
#define ABCE_DEVICE_MAX_AGENTS ABCE_MAX_TOPOLOGY_DEVICES

// Client device ordinals the peer map covers. A client names devices in its own
// numbering (HIP ordinals, say), which is not the agent numbering; see
// abce_device_engine_set_t::agent_for_device.
#define ABCE_DEVICE_MAX_PEERS 72u

// Lets the submitter pick the ring. Distinct from every engine index, so "no
// opinion" and "ring 0" stay tellable apart.
#define ABCE_DEVICE_AUTO_ENGINE UINT32_MAX

// Agent ids naming one end of a copy. Values >= 0 are KFD node ids.
#define ABCE_DEVICE_AGENT_HOST ((int8_t)-1)
#define ABCE_DEVICE_AGENT_UNKNOWN ((int8_t)-2)

// What the host policy says about one transfer class: an affinity anchor, and
// the registered rings it ruled out. A ring missing from the policy's ranked
// list is forbidden for that class — that is how a reservation such as
// gfx90a's H2D-only SDMA0 is enforced — so a forbidden ring is never used even
// as a fallback. Rings kept out of the host orchestrator (device-owned ones)
// are unknown to the policy and so never forbidden.
typedef struct abce_device_route_t {
  uint64_t forbidden;
  uint32_t preferred;  // ABCE_DEVICE_MAX_ENGINES when the policy named none.
} abce_device_route_t;

// Read-only from the device: the host publishes every field before the view
// reaches a kernel, and the submitter writes only through the rings' pointers.
// That lets the host keep this in ordinary memory and put only the shared ring
// cursors — which host and device CAS concurrently — in an uncached pool.
typedef struct abce_device_engine_set_t {
  ABCE_ALIGNAS(64) uint32_t abi_version;
  uint32_t num_engines;
  // Whether SDMA can retire the completion signal with an atomic packet.
  bool device_atomic_support;
  // Agent these rings belong to, so a kernel can tag the local end of a copy.
  int8_t local_agent;

  // Rings a KERNEL may submit to (bit i => ring i). Not every registered ring:
  // a ring whose doorbell a shader store does not reach is withheld here while
  // staying available to host copies. A bit also asserts engines[i] is
  // complete, so the submitter treats this as the whole answer.
  uint64_t device_usable_engines;

  // Rings no host producer submits to, always a subset of the above. With a
  // kernel the only producer their cursors can live in device memory, which
  // makes a submit through them materially cheaper, so automatic ring choice
  // prefers them. Zero is the normal state.
  uint64_t device_owned_engines;

  abce_builder_t builder;
  abce_device_ring_t engines[ABCE_DEVICE_MAX_ENGINES];

  // Routes per transfer class, relative to local_agent.
  abce_device_route_t host_to_local;
  abce_device_route_t local_to_host;
  abce_device_route_t local_to_local;
  abce_device_route_t local_to_peer[ABCE_DEVICE_MAX_AGENTS];
  abce_device_route_t peer_to_local[ABCE_DEVICE_MAX_AGENTS];
  // For a copy whose class cannot be told — an end is unknown, or the copy does
  // not touch this device. The host fills it with the policy's answer for a copy
  // between two unknown devices, the same thing its own planner does with
  // unknown device ids, so a caller that names no agents still stays off every
  // ring the policy reserves.
  abce_device_route_t unknown;

  // Client device ordinal -> agent id; ABCE_DEVICE_AGENT_UNKNOWN when not
  // published, which costs affinity and never correctness.
  int8_t agent_for_device[ABCE_DEVICE_MAX_PEERS];
} abce_device_engine_set_t;

// Handle a kernel receives as a launch argument.
typedef struct abce_device_handle_t {
  abce_device_engine_set_t* engine_set;
  uint32_t abi_version;
} abce_device_handle_t;

// What a caller polls to learn a copy has retired: the completion signal and
// the value it lands on.
typedef struct abce_device_token_t {
  uint64_t* signal;
  uint64_t reference;
} abce_device_token_t;

//===----------------------------------------------------------------------===//
// Host side: building the view
//===----------------------------------------------------------------------===//

static inline void abce_device_route_initialize(abce_device_route_t* out_route) {
  out_route->forbidden = 0;
  out_route->preferred = ABCE_DEVICE_MAX_ENGINES;
}

// Initializes an empty view around a copy of |builder|: no rings, no published
// agents, and every route open, which is the right answer for a host with no
// policy to consult. The host fills the rest before handing out a handle.
static inline void abce_device_engine_set_initialize(const abce_builder_t* builder,
                                                     abce_device_engine_set_t* out_engine_set) {
  memset(out_engine_set, 0, sizeof(*out_engine_set));
  out_engine_set->abi_version = ABCE_DEVICE_ABI_VERSION;
  out_engine_set->local_agent = ABCE_DEVICE_AGENT_UNKNOWN;
  out_engine_set->builder = *builder;
  abce_device_route_initialize(&out_engine_set->host_to_local);
  abce_device_route_initialize(&out_engine_set->local_to_host);
  abce_device_route_initialize(&out_engine_set->local_to_local);
  for (uint32_t agent = 0; agent < ABCE_DEVICE_MAX_AGENTS; ++agent) {
    abce_device_route_initialize(&out_engine_set->local_to_peer[agent]);
    abce_device_route_initialize(&out_engine_set->peer_to_local[agent]);
  }
  abce_device_route_initialize(&out_engine_set->unknown);
  for (uint32_t ordinal = 0; ordinal < ABCE_DEVICE_MAX_PEERS; ++ordinal)
    out_engine_set->agent_for_device[ordinal] = ABCE_DEVICE_AGENT_UNKNOWN;
}

//===----------------------------------------------------------------------===//
// Handle and agents
//===----------------------------------------------------------------------===//

static inline bool abce_device_handle_valid(const abce_device_handle_t* handle) {
  return handle && handle->engine_set && handle->abi_version == ABCE_DEVICE_ABI_VERSION &&
         handle->engine_set->abi_version == ABCE_DEVICE_ABI_VERSION;
}

// Agent id for client device ordinal |device_ordinal|, for the agent arguments
// of a submit. ABCE_DEVICE_AGENT_UNKNOWN for an unpublished ordinal, which
// routes the copy as one between unknown devices rather than failing it.
static inline int8_t abce_device_agent_for(const abce_device_handle_t* handle,
                                           uint32_t device_ordinal) {
  if (!abce_device_handle_valid(handle) || device_ordinal >= ABCE_DEVICE_MAX_PEERS)
    return ABCE_DEVICE_AGENT_UNKNOWN;
  return handle->engine_set->agent_for_device[device_ordinal];
}

// Agent id of the device these rings belong to.
static inline int8_t abce_device_local_agent(const abce_device_handle_t* handle) {
  return abce_device_handle_valid(handle) ? handle->engine_set->local_agent
                                          : ABCE_DEVICE_AGENT_UNKNOWN;
}

//===----------------------------------------------------------------------===//
// Ring choice — the one decision the submitter makes, host-callable so it can
// be tested without a GPU queue
//===----------------------------------------------------------------------===//

// The view fields a submit reads whatever its copy. abce_device_submit_linear()
// copies them into registers before any lane's turn: inside the turn loop the
// compiler cannot tell that earlier turns' ring stores left the view alone, and
// would reload them lane by lane with vector loads. The route and ring a copy
// uses depend on its own arguments, so those are read in its turn.
typedef struct abce_device_submit_view_t {
  abce_device_engine_set_t* engine_set;
  abce_builder_t builder;
  uint64_t usable_engines;
  uint64_t owned_engines;
  int8_t local_agent;
} abce_device_submit_view_t;

static inline void abce_device_submit_view_load(abce_device_engine_set_t* engine_set,
                                                abce_device_submit_view_t* out_view) {
  out_view->engine_set = engine_set;
  out_view->builder = engine_set->builder;
  out_view->usable_engines =
      engine_set->device_usable_engines & ((1ull << ABCE_DEVICE_MAX_ENGINES) - 1u);
  out_view->owned_engines = engine_set->device_owned_engines;
  out_view->local_agent = engine_set->local_agent;
}

// The route for a copy between |src_agent| and |dst_agent|: its class route
// when the pair falls in a class the host routed, otherwise the unknown route.
static inline const abce_device_route_t* abce_device_route(const abce_device_submit_view_t* view,
                                                           int8_t src_agent, int8_t dst_agent) {
  const abce_device_engine_set_t* engine_set = view->engine_set;
  const int8_t local = view->local_agent;
  if (local < 0) return &engine_set->unknown;
  if (src_agent == local) {
    if (dst_agent == local) return &engine_set->local_to_local;
    if (dst_agent == ABCE_DEVICE_AGENT_HOST) return &engine_set->local_to_host;
    if (dst_agent >= 0 && (uint32_t)dst_agent < ABCE_DEVICE_MAX_AGENTS)
      return &engine_set->local_to_peer[dst_agent];
  } else if (dst_agent == local) {
    if (src_agent == ABCE_DEVICE_AGENT_HOST) return &engine_set->host_to_local;
    if (src_agent >= 0 && (uint32_t)src_agent < ABCE_DEVICE_MAX_AGENTS)
      return &engine_set->peer_to_local[src_agent];
  }
  return &engine_set->unknown;
}

// Resolves the ring a copy runs on into |out_engine|.
//
// A named ring is honoured or refused, never silently substituted: a caller
// that named one is placing the copy on purpose and would rather hear that it
// cannot than have the work land elsewhere. ABCE_DEVICE_AUTO_ENGINE takes the
// route's preferred ring when permitted, then a device-owned ring, then the
// lowest permitted one.
//
// Returns ABCE_STATUS_NO_ENGINE when no ring a kernel may use qualifies, and
// ABCE_STATUS_NO_LEGAL_ENGINE when rings exist but the route forbids them all.
static inline abce_status_t abce_device_resolve_engine(const abce_device_submit_view_t* view,
                                                       uint32_t engine, int8_t src_agent,
                                                       int8_t dst_agent, uint32_t* out_engine) {
  const uint64_t candidates = view->usable_engines;
  if (!candidates) return ABCE_STATUS_NO_ENGINE;
  const abce_device_route_t* route = abce_device_route(view, src_agent, dst_agent);
  const uint64_t allowed = candidates & ~route->forbidden;

  if (engine != ABCE_DEVICE_AUTO_ENGINE) {
    if (engine >= ABCE_DEVICE_MAX_ENGINES || !((candidates >> engine) & 1u))
      return ABCE_STATUS_NO_ENGINE;
    if (!((allowed >> engine) & 1u)) return ABCE_STATUS_NO_LEGAL_ENGINE;
    *out_engine = engine;
    return ABCE_STATUS_OK;
  }

  if (!allowed) return ABCE_STATUS_NO_LEGAL_ENGINE;
  if (route->preferred < ABCE_DEVICE_MAX_ENGINES && ((allowed >> route->preferred) & 1u)) {
    *out_engine = route->preferred;
    return ABCE_STATUS_OK;
  }
  const uint64_t owned = allowed & view->owned_engines;
  *out_engine = (uint32_t)abce_count_trailing_zeros_u64(owned ? owned : allowed);
  return ABCE_STATUS_OK;
}

//===----------------------------------------------------------------------===//
// Submission
//===----------------------------------------------------------------------===//

// Emits one linear copy that retires |completion_signal| onto |ring|.
//
// Two packet forms, one decrement either way, so arming is identical: gfx125+
// emits COPY_LINEAR_WAITSIGNAL, which carries its own completion; everything
// else emits COPY_LINEAR plus an atomic decrement. The packet count follows from the size — the builder
// chunks against its per-packet maximum — so a large copy becomes several
// packets in one frame and callers never reason about packet counts.
static inline abce_status_t abce_device_emit_linear(const abce_builder_t* builder,
                                                    const abce_device_ring_t* view_ring, void* dst,
                                                    const void* src, uint64_t size,
                                                    void* completion_signal) {
  // A register copy, for the reason abce_device_submit_view_t is one: read
  // through the view, the ring's pointers would be reloaded after the packet
  // stores.
  abce_device_ring_t ring_copy = *view_ring;
  abce_device_ring_t* ring = &ring_copy;
  const uint64_t num_packets = abce_builder_num_copy_packets(builder, (size_t)size);
  if (num_packets > ABCE_MAX_OP_PACKETS) return ABCE_STATUS_OUT_OF_RANGE;
  const bool fused = abce_builder_is_gfx125plus(builder);
  const uint64_t copy_bytes = num_packets * sizeof(SDMA_PKT_COPY_LINEAR);
  // Fused packets are compacted — header plus body per chunk, with the single
  // signal block on the last chunk. Same accounting as abce_frame_fused_bytes().
  const uint64_t frame_bytes = fused ? (num_packets * (1u + 6u) + 5u) * sizeof(uint32_t)
                                     : copy_bytes + sizeof(SDMA_PKT_ATOMIC);

  // Past the packet cap, reserve is the authority on whether a frame fits: it
  // fails only with OUT_OF_RANGE when the frame can never fit, and otherwise
  // blocks until the engine has freed room.
  abce_ring_reservation_t reservation;
  const abce_status_t status = abce_device_ring_reserve(ring, frame_bytes, &reservation);
  if (!abce_status_is_ok(status)) return status;
  char* frame = abce_device_ring_write_address(ring, reservation.start);

  // Every ring dword is stored once and never read back. The fused builder
  // already writes whole dwords; the fixed-layout packets are formed in a zeroed
  // local and stored whole, since setting their bitfields in place would need
  // the reservation zeroed first and a ring-memory round trip per field.
  if (fused) {
    abce_wait_signal_params_t params = abce_wait_signal_none();
    params.signal_addr = completion_signal;
    params.boundary_wait_signal = true;
    abce_builder_build_wait_signal_copy(builder, frame, dst, src, (size_t)size, &params);
  } else {
    SDMA_PKT_COPY_LINEAR* copy_packets = (SDMA_PKT_COPY_LINEAR*)frame;
    uint64_t copied = 0;
    for (uint64_t packet_idx = 0; packet_idx < num_packets; ++packet_idx) {
      const uint32_t chunk_size =
          (uint32_t)ABCE_MIN(size - copied, (uint64_t)builder->max_copy_size);
      SDMA_PKT_COPY_LINEAR packet;
      memset(&packet, 0, sizeof(packet));
      abce_builder_form_copy_packet(builder, &packet, (char*)dst + copied,
                                    (const char*)src + copied, chunk_size);
      copy_packets[packet_idx] = packet;
      copied += chunk_size;
    }
    SDMA_PKT_ATOMIC completion;
    memset(&completion, 0, sizeof(completion));
    abce_builder_build_atomic_decrement(builder, (char*)&completion, completion_signal);
    *(SDMA_PKT_ATOMIC*)(frame + copy_bytes) = completion;
  }
  return abce_device_ring_submit(ring, &reservation);
}

// One lane's submission, in its turn; abce_device_submit_linear() is the entry
// point.
static inline abce_status_t abce_device_submit_linear_lane(
    const abce_device_submit_view_t* view, void* dst, const void* src, uint64_t size,
    int8_t src_agent, int8_t dst_agent, uint32_t engine, uint64_t* completion_signal,
    uint64_t completion_reference, abce_device_token_t* out_token) {
  if (!out_token || !dst || !src || size == 0 || !completion_signal ||
      ((uintptr_t)completion_signal % sizeof(uint64_t)) != 0)
    return ABCE_STATUS_INVALID_ARGUMENT;

  uint32_t ring_index = ABCE_DEVICE_MAX_ENGINES;
  abce_status_t status =
      abce_device_resolve_engine(view, engine, src_agent, dst_agent, &ring_index);
  if (!abce_status_is_ok(status)) return status;
  status = abce_device_emit_linear(&view->builder, &view->engine_set->engines[ring_index], dst,
                                   src, size, completion_signal);
  if (!abce_status_is_ok(status)) return status;

  out_token->signal = completion_signal;
  out_token->reference = completion_reference;
  return ABCE_STATUS_OK;
}

// Submits one linear copy of |size| bytes and fills |out_token| to poll on.
//
// The whole device-side API: form the packets, ring the doorbell, return. The
// copy is published, not complete.
//
// SIGNAL OWNERSHIP. |completion_signal| must be 8-byte aligned and SDMA-visible,
// and the caller arms it to |completion_reference| + 1. That arming is exact: one
// copy is one frame, so exactly one 64-bit decrement lands and this never adds
// to the signal.
//
// RING CHOICE. |src_agent| and |dst_agent| steer which ring serves the copy;
// ABCE_DEVICE_AGENT_UNKNOWN is always safe and routes the copy as one between
// unknown devices. |engine| names a ring outright, or is ABCE_DEVICE_AUTO_ENGINE.
// See abce_device_resolve_engine().
//
// LANES. Any number of lanes may call this at once, each with its own copy. Lanes
// of one wave have no independent forward progress — a lane spinning on the
// commit cursor holds its whole wave in the loop, so a sibling lane whose turn
// it is never gets to publish — so the wave's active lanes take turns, each
// running its whole submission while the others are masked off. Waves and the
// host still submit concurrently.
//
// MEMORY VISIBILITY. This call is a SYSTEM-SCOPE RELEASE POINT, and owning that
// is deliberately the API's job rather than the caller's. Everything the calling
// thread stored before it — the source this copy reads, and anything it wrote
// into the destination — is ordered ahead of the engine seeing the copy, so
// SDMA cannot read the source before the stores that filled it land, and an
// in-flight store to the destination cannot arrive after the copy and
// overwrite it. Callers therefore neither need nor should add a release of
// their own, which would pay the cache writeback twice for one guarantee. The
// release covers the CALLING
// THREAD; stores made by other threads are covered once the caller has
// synchronised with them by its own means first.
//
// Blocks while the ring is full and while earlier reservations publish, so it
// must not be called from a thread the completion of this copy depends on.
//
// Returns ABCE_STATUS_INVALID_ARGUMENT for a bad handle, pointer, size or signal
// address; ABCE_STATUS_UNIMPLEMENTED when SDMA cannot retire the signal on this
// device or its copies need GCR; the resolve statuses above; or
// ABCE_STATUS_OUT_OF_RANGE when the copy's frame can never fit in its ring.
static inline abce_status_t abce_device_submit_linear(const abce_device_handle_t* handle,
                                                      void* dst, const void* src, uint64_t size,
                                                      int8_t src_agent, int8_t dst_agent,
                                                      uint32_t engine, uint64_t* completion_signal,
                                                      uint64_t completion_reference,
                                                      abce_device_token_t* out_token) {
  if (!abce_device_handle_valid(handle)) return ABCE_STATUS_INVALID_ARGUMENT;
  abce_device_engine_set_t* engine_set = handle->engine_set;
  // The frame carries no GCR packets, so a part whose copies need them is not
  // one this submitter can serve.
  if (!engine_set->device_atomic_support || abce_builder_requires_gcr(&engine_set->builder))
    return ABCE_STATUS_UNIMPLEMENTED;
  abce_device_submit_view_t view;
  abce_device_submit_view_load(engine_set, &view);

#if defined(__HIP_DEVICE_COMPILE__)
  const uint32_t lane = __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
  abce_status_t status = ABCE_STATUS_OK;
  for (uint64_t waiting = __builtin_amdgcn_read_exec(); waiting != 0; waiting &= waiting - 1) {
    if ((uint32_t)abce_count_trailing_zeros_u64(waiting) == lane)
      status = abce_device_submit_linear_lane(&view, dst, src, size, src_agent, dst_agent, engine,
                                              completion_signal, completion_reference, out_token);
  }
  return status;
#else
  return abce_device_submit_linear_lane(&view, dst, src, size, src_agent, dst_agent, engine,
                                        completion_signal, completion_reference, out_token);
#endif  // __HIP_DEVICE_COMPILE__
}

// Whether the copy behind a token filled by a successful submit has retired.
static inline bool abce_device_token_retired(const abce_device_token_t* token) {
  return abce_ring_atomic_load_acquire(token->signal) == token->reference;
}

// Spins up to |max_spins| times for |token|, then checks once more. Bounded so a
// kernel cannot hang forever on a lost or mis-armed signal; false leaves the
// copy outstanding rather than failed.
static inline bool abce_device_token_wait(const abce_device_token_t* token, uint64_t max_spins) {
  for (uint64_t spin = 0; spin < max_spins; ++spin) {
    if (abce_device_token_retired(token)) return true;
    abce_ring_pause();
  }
  return abce_device_token_retired(token);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#if defined(ABCE_DEVICE_FORCE_HOST_DEVICE_)
#pragma clang force_cuda_host_device end
#undef ABCE_DEVICE_FORCE_HOST_DEVICE_
#endif  // ABCE_DEVICE_FORCE_HOST_DEVICE_

#endif  // ABCE_DEVICE_H_
