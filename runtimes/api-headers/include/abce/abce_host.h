/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Accelerated Blit Copy Engine (ABCE) — host mapping and submission API.

#ifndef ABCE_HOST_H_
#define ABCE_HOST_H_

#include <assert.h>
#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "abce_builder.h"
#include "abce_config.h"
#include "abce_frame.h"
#include "abce_ring_host.h"
#include "abce_topology.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Endpoints and transfer classification
//===----------------------------------------------------------------------===//

// Where one end of a copy lives. Lets the engine policy tell H2D/D2H
// (host<->device) from device<->device, and same-device (local D2D) from
// cross-device (P2P) transfers, without the orchestrator ever dereferencing a
// pointer to discover locality.
typedef enum abce_endpoint_kind_t {
  ABCE_ENDPOINT_KIND_HOST = 0,
  ABCE_ENDPOINT_KIND_DEVICE = 1,
} abce_endpoint_kind_t;

// "Any / unknown device" sentinel for abce_copy_endpoint_t::device_id.
#define ABCE_ANY_DEVICE UINT32_MAX

// One end of a copy: whether it is host or device memory, and (for device
// memory) which device it lives on. device_id is an ABCE-local index the client
// assigns at topology-load / registration time; it is only meaningful when
// kind == ABCE_ENDPOINT_KIND_DEVICE.
typedef struct abce_copy_endpoint_t {
  abce_endpoint_kind_t kind;
  uint32_t device_id;
} abce_copy_endpoint_t;

static inline abce_copy_endpoint_t abce_copy_endpoint_make(abce_endpoint_kind_t kind,
                                                           uint32_t device_id) {
  abce_copy_endpoint_t endpoint;
  endpoint.kind = kind;
  endpoint.device_id = device_id;
  return endpoint;
}

// How a transfer crosses the memory topology.
//
// Derived purely from the endpoint classification the client supplies, so it
// costs nothing to know and never dereferences a user pointer. Engine selection
// keys its ranking bands off this; it is also the natural label for a profiler
// attributing a copy, which is why it is reported on the plan.
typedef enum abce_transfer_kind_t {
  ABCE_TRANSFER_KIND_UNKNOWN = 0,          // host->host: rejected before selection.
  ABCE_TRANSFER_KIND_HOST_TO_DEVICE = 1,   // H2D
  ABCE_TRANSFER_KIND_DEVICE_TO_HOST = 2,   // D2H
  ABCE_TRANSFER_KIND_PEER_TO_PEER = 3,     // D2D across devices (or with an unknown end)
  ABCE_TRANSFER_KIND_LOCAL_DEVICE_TO_DEVICE = 4,  // D2D within one device
} abce_transfer_kind_t;

// Classifies a transfer from its endpoints. Single source of truth: both engine
// ranking and the reported plan label go through here, so they cannot disagree
// about what kind of copy something is.
static inline abce_transfer_kind_t abce_classify_transfer(const abce_copy_endpoint_t* src,
                                                          const abce_copy_endpoint_t* dst) {
  if (src->kind == ABCE_ENDPOINT_KIND_DEVICE && dst->kind == ABCE_ENDPOINT_KIND_DEVICE) {
    // Local D2D requires both ids to be known: ABCE_ANY_DEVICE means "unknown",
    // so two unknown ends are a cross-device copy, not a same-device one, which
    // would steer a real P2P transfer away from the xGMI band.
    const bool same_device =
        src->device_id != ABCE_ANY_DEVICE && src->device_id == dst->device_id;
    return same_device ? ABCE_TRANSFER_KIND_LOCAL_DEVICE_TO_DEVICE
                       : ABCE_TRANSFER_KIND_PEER_TO_PEER;
  }
  if (src->kind == ABCE_ENDPOINT_KIND_HOST && dst->kind == ABCE_ENDPOINT_KIND_DEVICE)
    return ABCE_TRANSFER_KIND_HOST_TO_DEVICE;
  if (src->kind == ABCE_ENDPOINT_KIND_DEVICE && dst->kind == ABCE_ENDPOINT_KIND_HOST)
    return ABCE_TRANSFER_KIND_DEVICE_TO_HOST;
  return ABCE_TRANSFER_KIND_UNKNOWN;
}

// Bit for |kind| inside abce_plan_t::transfer_kinds.
static inline uint32_t abce_transfer_kind_bit(abce_transfer_kind_t kind) {
  return ((uint32_t)1) << (uint32_t)kind;
}

//===----------------------------------------------------------------------===//
// Engines
//===----------------------------------------------------------------------===//

// "No hardware engine id" sentinel for abce_engine_affinity_t::hw_engine_id.
#define ABCE_ANY_ENGINE UINT32_MAX

// Per-engine metadata supplied at registration.
typedef struct abce_engine_affinity_t {
  // Hardware SDMA engine id (the policy's hw-id space; maps to this registered
  // index). ABCE_ANY_ENGINE uses the registered index as the hardware id.
  //
  // There is deliberately no engine-class field. Which engines suit which band
  // is not a client-declared property: it is either measured (the H2D/D2H
  // heatmap) or reported by the driver (KFD's non-xGMI/xGMI engine split and
  // per-link recommended engine mask). On MI300X the two are not even
  // correlated -- KFD classes engines 2-15 as xGMI, yet the measured H2D
  // ranking puts four of them ahead of both non-xGMI engines -- so treating a
  // class as a filter for host copies would discard the fastest engines.
  uint32_t hw_engine_id;
} abce_engine_affinity_t;

static inline void abce_engine_affinity_initialize(abce_engine_affinity_t* out_affinity) {
  out_affinity->hw_engine_id = ABCE_ANY_ENGINE;
}

// Lowest set bit index of |mask|, or ABCE_MAX_ENGINES if empty.
static inline uint32_t abce_first_engine(uint64_t mask) {
  return mask ? (uint32_t)abce_count_trailing_zeros_u64(mask) : ABCE_MAX_ENGINES;
}

//===----------------------------------------------------------------------===//
// abce_copy_op_t
//===----------------------------------------------------------------------===//

// One logical copy operation from the client.
//
// TODO(perf, revisit at integration): this is a flat 88-byte AoS struct (2
// cachelines). For the dominant linear H2D/D2H batch only kind/src/dst/size/
// endpoints are live; the per-kind tail (dsts+num_dsts, fill_value, size2, rect,
// indirect_*) is mutually exclusive and pollutes the line. When we wire this to
// the hipMemcpyBatchAsync path, consider a hot core + kind-keyed union to bring
// the common op to <=64B / one cacheline (mirror in abce_engine_op_t). Measure
// typical batch sizes first — only worth the ergonomic hit if these batches are
// hot.
typedef struct abce_copy_op_t {
  abce_op_kind_t kind;

  const void* src;  // linear source / swap endpoint B / indirect src.
  void* dst;        // linear/fill destination / swap endpoint A / indirect dst.

  void* const* dsts;  // multicast destinations.
  uint32_t num_dsts;  // multicast destination count.

  uint32_t fill_value;  // fill pattern (FILL).

  size_t size;  // transfer size in bytes (fill: byte count, multiple of 4).

  // Optional secondary size for two-region ops. For SWAP it is endpoint B's
  // (src) byte count; 0 => same as size. Ignored by other kinds.
  size_t size2;

  bool indirect_src;  // INDIRECT: `src` points to an address list.
  bool indirect_dst;  // INDIRECT: `dst` points to an address list.

  const abce_copy_rect_desc_t* rect;  // COPY_RECT geometry (required for COPY_RECT).

  abce_copy_endpoint_t src_end;  // source residence for the policy.
  abce_copy_endpoint_t dst_end;  // destination residence for the policy.
} abce_copy_op_t;

// Zero-initializing this struct is NOT the default: both endpoints default to
// ABCE_ANY_DEVICE rather than device 0. Always initialize through here.
static inline void abce_copy_op_initialize(abce_copy_op_t* out_copy) {
  memset(out_copy, 0, sizeof(*out_copy));
  out_copy->kind = ABCE_OP_KIND_LINEAR;
  out_copy->src_end = abce_copy_endpoint_make(ABCE_ENDPOINT_KIND_HOST, ABCE_ANY_DEVICE);
  out_copy->dst_end = abce_copy_endpoint_make(ABCE_ENDPOINT_KIND_HOST, ABCE_ANY_DEVICE);
}

//===----------------------------------------------------------------------===//
// abce_execution_descriptor_t
//===----------------------------------------------------------------------===//

// Which hardware block executed a copy. Three bits inside an execution
// descriptor, because a completion signal is not the property of any one
// engine: the same signal object that an SDMA batch completes can instead be
// completed by a compute blit kernel, or by a host memcpy, and a reader has no
// way to tell from the signal value.
//
// NONE is the zero value and means nobody stamped the descriptor, which is what
// makes a descriptor self-validating.
//
// ABCE only ever writes SDMA -- it builds nothing else. The other values exist
// so that a client whose fallback paths complete the *same* signal can stamp
// the same slot, giving one field a consumer can trust instead of a heuristic.
// ROCr, for instance, currently distinguishes SDMA from a blit kernel by zeroing
// its SDMA timestamps before a copy and testing them afterwards
// (SharedSignal::CopyPrep / GetRawTs); the two paths do not even record
// timestamps in the same place, so this field also tells a reader which pair to
// look at.
// Three bits rather than two: the spare values cost nothing here (the reserved
// tail absorbs them) and leave room for executors that have not come up yet,
// which matters for a field written by more than one component.
typedef enum abce_execution_engine_kind_t {
  ABCE_EXECUTION_ENGINE_KIND_NONE = 0,     // not stamped.
  ABCE_EXECUTION_ENGINE_KIND_SDMA = 1,     // SDMA / DMA engine. What ABCE emits.
  ABCE_EXECUTION_ENGINE_KIND_COMPUTE = 2,  // compute shader (a blit kernel).
  ABCE_EXECUTION_ENGINE_KIND_CPU = 3,      // host processor (a memcpy).
} abce_execution_engine_kind_t;

// A 32-bit summary of how a batch actually executed, for whoever reads the
// completion signal afterwards.
//
// Timestamps tell a profiler *when* a copy ran; this tells it *what ran and
// where*, which is the part a tool otherwise has to guess (ROCr, for instance,
// infers "SDMA or blit kernel?" from whether its SDMA timestamps came back
// non-zero). Submit writes it, not the map phase: a client may retarget
// abce_plan_frame_t::engine in between, so only submit knows the engines really
// used.
//
// Engines are reported as a mask of *hardware* engine ids rather than ABCE's
// registered indices, which are a client-local numbering that means nothing to
// an external reader. A mask, rather than a single id, is what makes a fan-out
// representable.
//
// | bits  | field |
// |-------|-------|
// | 0-15  | instance mask, interpreted per engine kind (see below) |
// | 16-18 | abce_transfer_kind_t |
// | 19    | batch mixed transfer kinds, so the transfer kind field reads UNKNOWN |
// | 20-22 | abce_execution_engine_kind_t |
// | 23-31 | reserved, zero |
//
// The instance mask is only meaningful relative to the engine kind, which is why
// the kind is part of the contract rather than assumed. For SDMA, bit *h* means
// hardware SDMA engine *h* ran part of the batch. Another kind defines its own
// instances, or leaves the mask zero when it has none to report -- a blit kernel
// has no equivalent of an SDMA engine id.
//
// A descriptor is valid iff its engine kind is not NONE. Validity deliberately
// does not hang off the instance mask: that would force every writer to invent
// an instance, and would make a legitimately mask-less stamp unreadable.
//
// An engine *class* -- xGMI versus PCIe within SDMA -- is deliberately not
// reported, which is a different question from engine kind. ABCE no longer has
// one to report (see abce_engine_affinity_t), and it would mislead if it did: on
// MI300X the driver classes engines 2-15 as xGMI while the measured H2D ranking
// prefers four of them over both non-xGMI engines, so "an xGMI engine ran an H2D
// copy" is the normal case rather than an anomaly worth a bit. Engine kind is
// the opposite: it distinguishes hardware blocks with genuinely different
// behaviour, including where they record their timestamps.
#define ABCE_EXECUTION_DESCRIPTOR_INSTANCE_MASK_SHIFT 0u
#define ABCE_EXECUTION_DESCRIPTOR_INSTANCE_MASK_MASK 0xFFFFu
#define ABCE_EXECUTION_DESCRIPTOR_TRANSFER_KIND_SHIFT 16u
#define ABCE_EXECUTION_DESCRIPTOR_TRANSFER_KIND_MASK 0x7u
#define ABCE_EXECUTION_DESCRIPTOR_MIXED_KINDS_BIT (((uint32_t)1) << 19)
#define ABCE_EXECUTION_DESCRIPTOR_ENGINE_KIND_SHIFT 20u
#define ABCE_EXECUTION_DESCRIPTOR_ENGINE_KIND_MASK 0x7u

ABCE_STATIC_ASSERT(ABCE_MAX_ENGINES <= 16, "instance mask field is 16 bits wide");

static inline uint32_t abce_execution_descriptor_encode(abce_execution_engine_kind_t engine_kind,
                                                        uint32_t instance_mask,
                                                        abce_transfer_kind_t transfer_kind,
                                                        bool mixed_kinds) {
  return ((instance_mask & ABCE_EXECUTION_DESCRIPTOR_INSTANCE_MASK_MASK)
          << ABCE_EXECUTION_DESCRIPTOR_INSTANCE_MASK_SHIFT) |
         (((uint32_t)transfer_kind & ABCE_EXECUTION_DESCRIPTOR_TRANSFER_KIND_MASK)
          << ABCE_EXECUTION_DESCRIPTOR_TRANSFER_KIND_SHIFT) |
         (mixed_kinds ? ABCE_EXECUTION_DESCRIPTOR_MIXED_KINDS_BIT : 0) |
         (((uint32_t)engine_kind & ABCE_EXECUTION_DESCRIPTOR_ENGINE_KIND_MASK)
          << ABCE_EXECUTION_DESCRIPTOR_ENGINE_KIND_SHIFT);
}

static inline abce_execution_engine_kind_t abce_execution_descriptor_engine_kind(uint32_t word) {
  return (abce_execution_engine_kind_t)((word >> ABCE_EXECUTION_DESCRIPTOR_ENGINE_KIND_SHIFT) &
                                        ABCE_EXECUTION_DESCRIPTOR_ENGINE_KIND_MASK);
}

static inline bool abce_execution_descriptor_valid(uint32_t word) {
  return abce_execution_descriptor_engine_kind(word) != ABCE_EXECUTION_ENGINE_KIND_NONE;
}

static inline uint32_t abce_execution_descriptor_instance_mask(uint32_t word) {
  return (word >> ABCE_EXECUTION_DESCRIPTOR_INSTANCE_MASK_SHIFT) &
         ABCE_EXECUTION_DESCRIPTOR_INSTANCE_MASK_MASK;
}

static inline abce_transfer_kind_t abce_execution_descriptor_transfer_kind(uint32_t word) {
  return (abce_transfer_kind_t)((word >> ABCE_EXECUTION_DESCRIPTOR_TRANSFER_KIND_SHIFT) &
                                ABCE_EXECUTION_DESCRIPTOR_TRANSFER_KIND_MASK);
}

static inline bool abce_execution_descriptor_mixed_kinds(uint32_t word) {
  return (word & ABCE_EXECUTION_DESCRIPTOR_MIXED_KINDS_BIT) != 0;
}

//===----------------------------------------------------------------------===//
// Engine selection policy
//===----------------------------------------------------------------------===//

// Returns a disposition plus the number of ranked engine indices written.
// NO_PREFERENCE permits ascending/round-robin fallback; NO_LEGAL_ENGINE is a
// hard rejection and never falls back.
typedef enum abce_policy_disposition_t {
  ABCE_POLICY_DISPOSITION_NO_PREFERENCE = 0,
  ABCE_POLICY_DISPOSITION_RANKED = 1,
  ABCE_POLICY_DISPOSITION_NO_LEGAL_ENGINE = 2,
} abce_policy_disposition_t;

typedef struct abce_policy_result_t {
  abce_policy_disposition_t disposition;
  uint32_t count;
} abce_policy_result_t;

// Optional engine-selection policy callback.
//
// Called per-copy during the map phase to rank engines for a specific transfer.
// If not supplied (null), the orchestrator uses ascending-index / round-robin.
//
// Inputs (read-only):
//   user_data      — opaque policy pointer (e.g. topology table).
//   src            — source endpoint of this copy (host/device + device_id).
//   dst            — destination endpoint of this copy.
//   candidate_mask — bitmask of legal engines for this batch (bit i = engine i).
//   max            — max entries the caller can accept in out_engines.
//
// Output:
//   out_engines[0..return) — up to |max| engine indices, ranked best-first. Each
//                            must be a bit set in |candidate_mask|.
typedef abce_policy_result_t (*abce_engine_policy_fn_t)(void* user_data,
                                                        const abce_copy_endpoint_t* src,
                                                        const abce_copy_endpoint_t* dst,
                                                        uint64_t candidate_mask,
                                                        uint32_t* out_engines, uint32_t max);

// Measured host-copy (H2D/D2H) engine ordering for a specific part.
//
// SPX bandwidth per SDMA engine is not uniform: some engines are far better for
// D2H than others (e.g. on gfx94x/95x). Each profile lists hardware SDMA engine
// ids best->worst for each direction; the policy resolves those ids to
// registered indices and returns the top few. Parts are distinguished by
// (gfx minor version, total SDMA engine count).
typedef struct abce_heatmap_profile_t {
  uint8_t minor;                        // gfx9 minor version.
  uint8_t total_sdma;                   // total SDMA engines (host + xGMI).
  uint8_t d2h[ABCE_MAX_ENGINES];        // D2H hw engine ids, best->worst.
  uint8_t d2h_count;
  uint8_t h2d[ABCE_MAX_ENGINES];        // H2D hw engine ids, best->worst.
  uint8_t h2d_count;
} abce_heatmap_profile_t;

// Selects the heatmap profile for |minor| / |total_sdma|, or null when the part
// is unrecognized (ranking then degrades to ascending order).
static inline const abce_heatmap_profile_t* abce_select_heatmap(uint8_t minor,
                                                                uint8_t total_sdma) {
  // clang-format off
  static const abce_heatmap_profile_t kProfiles[] = {
    // These profiles were measured through ROCr's copy_on_engine selector. On
    // gfx94x/95x ROCr swaps its two host blit queues: selector 0 targets physical
    // SDMA1 and selector 1 targets physical SDMA0. ABCE queues are targeted by
    // physical id directly, so the rankings below have selector ids 0/1 swapped.
    // gfx942, 16 SDMA (MI300X): physical engines 8-15 are poor for D2H
    // (~20 GB/s), while H2D peak bandwidth is ~flat (~55). Use the same tier
    // pattern for both directions and defer physical engines 0 and 3, which
    // have a large medium-copy penalty through ABCE.
    {4, 16, {2, 4, 6, 5, 1, 7, 0, 3}, 8, {2, 4, 6, 5, 1, 7, 0, 3}, 8},
    // gfx942, 8 SDMA (MI308X): only selector 0 (physical SDMA1) is slow for D2H
    // (~48 vs ~56 on the rest), while H2D is flat (~55). Match the direction
    // patterns and keep physical SDMA1 last.
    {4,  8, {0, 3, 4, 5, 2, 7, 6, 1}, 8, {0, 3, 4, 5, 2, 7, 6, 1}, 8},
    // gfx950, 16 SDMA (MI350X): selectors 0-3 are the fast tier for both
    // directions (~56). Selectors 4-7 are the second H2D tier (~51) but fall
    // to ~13 for D2H; selectors 8-15 are slower still and are omitted.
    {5, 16, {1, 0, 2, 3, 4, 5, 6, 7}, 8, {1, 0, 2, 3, 4, 5, 6, 7}, 8},
  };
  // clang-format on
  for (size_t profile_idx = 0; profile_idx < ABCE_ARRAYSIZE(kProfiles); ++profile_idx) {
    if (kProfiles[profile_idx].minor == minor && kProfiles[profile_idx].total_sdma == total_sdma)
      return &kProfiles[profile_idx];
  }
  return NULL;
}

typedef struct abce_sdma_engine_policy_t abce_sdma_engine_policy_t;

// Per-band ranking hooks. Each writes up to |max| hardware SDMA engine ids
// (best->worst) into |out_hwids| and returns the count. Omitting an engine from
// the list is how a hard reservation is enforced (it can never be chosen for
// that band).
//
// This vtable replaces the C++ virtual hierarchy; abce_sdma_policy_base_vtable
// is the former base class's behaviour.
typedef struct abce_sdma_engine_policy_vtable_t {
  uint32_t (*rank_h2d)(const abce_sdma_engine_policy_t* policy, const abce_copy_endpoint_t* src,
                       const abce_copy_endpoint_t* dst, uint8_t* out_hwids, uint32_t max);
  uint32_t (*rank_d2h)(const abce_sdma_engine_policy_t* policy, const abce_copy_endpoint_t* src,
                       const abce_copy_endpoint_t* dst, uint8_t* out_hwids, uint32_t max);
  uint32_t (*rank_p2p)(const abce_sdma_engine_policy_t* policy, const abce_copy_endpoint_t* src,
                       const abce_copy_endpoint_t* dst, uint8_t* out_hwids, uint32_t max);
  uint32_t (*rank_local_d2d)(const abce_sdma_engine_policy_t* policy,
                             const abce_copy_endpoint_t* src, const abce_copy_endpoint_t* dst,
                             uint8_t* out_hwids, uint32_t max);
} abce_sdma_engine_policy_vtable_t;

// Topology-aware SDMA engine-selection policy.
//
// Given a (src, dst) endpoint pair, produces a ranked list of engines. Three
// bands drive the ranking:
//   - local   (same-device copies)
//   - xGMI    (cross-device P2P copies)
//   - H2D/D2H (host copies)
//
// Which engines belong to a band is never declared by the client. Host copies
// are ordered by the measured heatmap, and the xGMI band comes from the driver's
// own engine split (abce_sdma_policy_set_engine_split, filled by
// abce_sdma_policy_load_topology_from_kfd). Absent both, every band resolves to
// all registered engines and selection degrades to pure load balancing, which is
// the right answer for parts where the engines are equivalent.
//
// Arch variants install their own vtable; the shared helpers below provide
// hw-id-to-register mapping and the ranking skeleton.
struct abce_sdma_engine_policy_t {
  const abce_sdma_engine_policy_vtable_t* vtable;
  const abce_heatmap_profile_t* heatmap;
  int hwid_to_reg[ABCE_MAX_ENGINES];
  uint8_t mapped_hwids[ABCE_MAX_ENGINES];
  uint32_t num_mapped;
  uint32_t num_non_xgmi;
  abce_topology_data_t topology;
};

//===----------------------------------------------------------------------===//
// Policy band helpers (shared by every vtable)
//===----------------------------------------------------------------------===//

// Appends every mapped hw engine id, in registration order, to |out_hwids|.
static inline uint32_t abce_sdma_policy_emit_all(const abce_sdma_engine_policy_t* policy,
                                                 uint8_t* out_hwids, uint32_t max) {
  const uint32_t count = policy->num_mapped < max ? policy->num_mapped : max;
  for (uint32_t idx = 0; idx < count; ++idx) out_hwids[idx] = policy->mapped_hwids[idx];
  return count;
}

// Appends the mapped hw engine ids in the driver's xGMI band. Falls back to
// every engine when the split is unknown or when the band came out empty (a
// client may have registered only non-xGMI engines), because offering a slower
// engine beats reporting that a P2P copy cannot be mapped at all.
static inline uint32_t abce_sdma_policy_emit_xgmi(const abce_sdma_engine_policy_t* policy,
                                                  uint8_t* out_hwids, uint32_t max) {
  uint32_t count = 0;
  if (policy->num_non_xgmi != 0) {
    for (uint32_t idx = 0; idx < policy->num_mapped && count < max; ++idx)
      if (policy->mapped_hwids[idx] >= policy->num_non_xgmi)
        out_hwids[count++] = policy->mapped_hwids[idx];
  }
  return count ? count : abce_sdma_policy_emit_all(policy, out_hwids, max);
}

static inline int abce_sdma_policy_xgmi_physical_id(const abce_sdma_engine_policy_t* policy,
                                                    uint32_t device_id) {
  return abce_topology_data_xgmi_physical_id(&policy->topology, device_id);
}

static inline uint64_t abce_sdma_policy_hive_id(const abce_sdma_engine_policy_t* policy,
                                                uint32_t device_id) {
  return abce_topology_data_hive_id(&policy->topology, device_id);
}

static inline uint32_t abce_sdma_policy_copy_hwids(const uint8_t* src, uint32_t src_count,
                                                   uint8_t* out_hwids, uint32_t max) {
  const uint32_t count = src_count < max ? src_count : max;
  for (uint32_t idx = 0; idx < count; ++idx) out_hwids[idx] = src[idx];
  return count;
}

//===----------------------------------------------------------------------===//
// Base policy vtable
//===----------------------------------------------------------------------===//

static inline uint32_t abce_sdma_policy_base_rank_all(const abce_sdma_engine_policy_t* policy,
                                                      const abce_copy_endpoint_t* src,
                                                      const abce_copy_endpoint_t* dst,
                                                      uint8_t* out_hwids, uint32_t max) {
  (void)src;
  (void)dst;
  return abce_sdma_policy_emit_all(policy, out_hwids, max);
}

static inline uint32_t abce_sdma_policy_base_rank_p2p(const abce_sdma_engine_policy_t* policy,
                                                      const abce_copy_endpoint_t* src,
                                                      const abce_copy_endpoint_t* dst,
                                                      uint8_t* out_hwids, uint32_t max) {
  (void)src;
  (void)dst;
  return abce_sdma_policy_emit_xgmi(policy, out_hwids, max);
}

static const abce_sdma_engine_policy_vtable_t abce_sdma_policy_base_vtable = {
    abce_sdma_policy_base_rank_all,
    abce_sdma_policy_base_rank_all,
    abce_sdma_policy_base_rank_p2p,
    abce_sdma_policy_base_rank_all,
};

//===----------------------------------------------------------------------===//
// gfx94x policy (gfx9, minor 4 or 5)
//
// H2D/D2H order comes from the measured heatmap profile; P2P uses the
// xGMI_physical_id SDMA-affinity table (even hardware ids only, *2) for the
// optimal first choice, then appends every xGMI engine in registration order.
// Same-hive is required for the xGMI band (rank_p2p returns none when the two
// GPUs' hives are known and differ).
//===----------------------------------------------------------------------===//

// SDMA-affinity map keyed by xGMI physical id (src row, dst col) -> logical xGMI
// engine.
static const int abce_gfx94x_sdma_affinity_map[8][8] = {
    {0, 7, 6, 1, 2, 4, 5, 3}, {7, 0, 1, 5, 4, 2, 3, 6}, {5, 1, 0, 6, 7, 3, 2, 4},
    {1, 6, 5, 0, 3, 7, 4, 2}, {2, 4, 7, 3, 0, 5, 6, 1}, {4, 2, 3, 7, 6, 0, 1, 5},
    {5, 3, 2, 4, 6, 1, 0, 7}, {3, 6, 4, 2, 1, 5, 7, 0}};

// The heatmap is a performance ranking, not a reservation, so both host bands
// append every remaining engine behind it: the measured order is preserved
// (the ranking skeleton drops the duplicates) while a client that registered
// engines the profile does not mention -- or that the profile deliberately ranks
// last, like the gfx950 tail -- still gets a legal engine instead of a failed
// map.
static inline uint32_t abce_gfx94x_rank_h2d(const abce_sdma_engine_policy_t* policy,
                                            const abce_copy_endpoint_t* src,
                                            const abce_copy_endpoint_t* dst, uint8_t* out_hwids,
                                            uint32_t max) {
  (void)src;
  (void)dst;
  const uint32_t count =
      policy->heatmap
          ? abce_sdma_policy_copy_hwids(policy->heatmap->h2d, policy->heatmap->h2d_count,
                                        out_hwids, max)
          : 0;
  return count < max ? count + abce_sdma_policy_emit_all(policy, out_hwids + count, max - count)
                     : count;
}

static inline uint32_t abce_gfx94x_rank_d2h(const abce_sdma_engine_policy_t* policy,
                                            const abce_copy_endpoint_t* src,
                                            const abce_copy_endpoint_t* dst, uint8_t* out_hwids,
                                            uint32_t max) {
  (void)src;
  (void)dst;
  const uint32_t count =
      policy->heatmap
          ? abce_sdma_policy_copy_hwids(policy->heatmap->d2h, policy->heatmap->d2h_count,
                                        out_hwids, max)
          : 0;
  return count < max ? count + abce_sdma_policy_emit_all(policy, out_hwids + count, max - count)
                     : count;
}

static inline uint32_t abce_gfx94x_rank_p2p(const abce_sdma_engine_policy_t* policy,
                                            const abce_copy_endpoint_t* src,
                                            const abce_copy_endpoint_t* dst, uint8_t* out_hwids,
                                            uint32_t max) {
  // Dedicated xGMI engines can only drive a directly-connected peer: require the
  // two GPUs share a hive when both hive ids are known.
  const uint64_t src_hive = abce_sdma_policy_hive_id(policy, src->device_id);
  const uint64_t dst_hive = abce_sdma_policy_hive_id(policy, dst->device_id);
  if (src_hive && dst_hive && src_hive != dst_hive) return 0;

  uint32_t count = 0;
  const int src_affinity = abce_sdma_policy_xgmi_physical_id(policy, src->device_id);
  const int dst_affinity = abce_sdma_policy_xgmi_physical_id(policy, dst->device_id);
  if (src_affinity >= 0 && src_affinity < 8 && dst_affinity >= 0 && dst_affinity < 8 &&
      count < max) {
    // Even engines only.
    out_hwids[count++] = (uint8_t)(abce_gfx94x_sdma_affinity_map[src_affinity][dst_affinity] * 2);
  }

  if (count < max) count += abce_sdma_policy_emit_xgmi(policy, out_hwids + count, max - count);
  return count;
}

static const abce_sdma_engine_policy_vtable_t abce_sdma_policy_gfx94x_vtable = {
    abce_gfx94x_rank_h2d,
    abce_gfx94x_rank_d2h,
    abce_gfx94x_rank_p2p,
    abce_sdma_policy_base_rank_all,
};

//===----------------------------------------------------------------------===//
// gfx90a policy (gfx9, minor 0, stepping 10)
//
// Due to a RAS issue SDMA0 can only drive H2D copies, so SDMA0 is reserved
// exclusively for that band: H2D = {SDMA0}, D2H = {SDMA1} (SDMA0 is simply never
// listed for D2H). P2P uses the dedicated xGMI band; the orchestrator spreads
// successive P2P ops across it.
//===----------------------------------------------------------------------===//

static inline uint32_t abce_gfx90a_rank_h2d(const abce_sdma_engine_policy_t* policy,
                                            const abce_copy_endpoint_t* src,
                                            const abce_copy_endpoint_t* dst, uint8_t* out_hwids,
                                            uint32_t max) {
  (void)policy;
  (void)src;
  (void)dst;
  if (max == 0) return 0;
  out_hwids[0] = 0;  // SDMA0 only (hard RAS reservation).
  return 1;
}

static inline uint32_t abce_gfx90a_rank_d2h(const abce_sdma_engine_policy_t* policy,
                                            const abce_copy_endpoint_t* src,
                                            const abce_copy_endpoint_t* dst, uint8_t* out_hwids,
                                            uint32_t max) {
  (void)policy;
  (void)src;
  (void)dst;
  if (max == 0) return 0;
  out_hwids[0] = 1;  // SDMA1 only; SDMA0 is reserved for H2D and never listed here.
  return 1;
}

static inline uint32_t abce_gfx90a_rank_p2p(const abce_sdma_engine_policy_t* policy,
                                            const abce_copy_endpoint_t* src,
                                            const abce_copy_endpoint_t* dst, uint8_t* out_hwids,
                                            uint32_t max) {
  const uint64_t src_hive = abce_sdma_policy_hive_id(policy, src->device_id);
  const uint64_t dst_hive = abce_sdma_policy_hive_id(policy, dst->device_id);
  if (src_hive && dst_hive && src_hive != dst_hive) return 0;
  return abce_sdma_policy_emit_xgmi(policy, out_hwids, max);
}

static const abce_sdma_engine_policy_vtable_t abce_sdma_policy_gfx90a_vtable = {
    abce_gfx90a_rank_h2d,
    abce_gfx90a_rank_d2h,
    abce_gfx90a_rank_p2p,
    abce_sdma_policy_base_rank_all,
};

//===----------------------------------------------------------------------===//
// Policy configuration
//===----------------------------------------------------------------------===//

static inline void abce_sdma_policy_initialize(const abce_sdma_engine_policy_vtable_t* vtable,
                                               abce_sdma_engine_policy_t* out_policy) {
  memset(out_policy, 0, sizeof(*out_policy));
  out_policy->vtable = vtable;
  for (uint32_t idx = 0; idx < ABCE_MAX_ENGINES; ++idx) out_policy->hwid_to_reg[idx] = -1;
  abce_topology_data_initialize(&out_policy->topology);
}

// Configures |out_policy| for a gfx version. Returns false for parts with no
// specialized policy (engine selection then degrades to ascending /
// round-robin), leaving |out_policy| untouched. The policy is otherwise
// unconfigured: the caller still maps its engines and loads topology / heatmap
// before use.
static inline bool abce_sdma_policy_initialize_for_isa(abce_isa_version_t isa,
                                                       abce_sdma_engine_policy_t* out_policy) {
  if (isa.major == 9) {
    // gfx90a: major 9, minor 0, stepping 10 — SDMA0 RAS reservation.
    if (isa.minor == 0 && isa.stepping == 10) {
      abce_sdma_policy_initialize(&abce_sdma_policy_gfx90a_vtable, out_policy);
      return true;
    }
    // gfx94x / gfx95x: heatmap + xGMI SDMA-affinity map.
    if (isa.minor == 4 || isa.minor == 5) {
      abce_sdma_policy_initialize(&abce_sdma_policy_gfx94x_vtable, out_policy);
      return true;
    }
  }
  return false;
}

// Associates a hardware SDMA engine id with a registered engine index. Call once
// per registered engine.
static inline void abce_sdma_policy_map_engine(abce_sdma_engine_policy_t* policy,
                                               uint32_t hw_engine_id, uint32_t reg_index) {
  if (hw_engine_id >= ABCE_MAX_ENGINES || reg_index >= ABCE_MAX_ENGINES ||
      policy->hwid_to_reg[hw_engine_id] >= 0)
    return;
  policy->hwid_to_reg[hw_engine_id] = (int)reg_index;
  if (policy->num_mapped < ABCE_MAX_ENGINES)
    policy->mapped_hwids[policy->num_mapped++] = (uint8_t)hw_engine_id;
}

// Records the driver's SDMA engine split: |num_non_xgmi| engines are not xGMI,
// and KFD numbers those first, so hardware ids at or above it form the xGMI band
// used for P2P. Zero means "unknown", which widens the P2P band to every
// registered engine rather than emptying it.
//
// This replaces a per-engine class declaration. The distinction is only ever
// used to *prefer* engines for P2P, never to exclude them from host copies: on
// MI300X the fastest measured H2D engines are xGMI engines.
static inline void abce_sdma_policy_set_engine_split(abce_sdma_engine_policy_t* policy,
                                                     uint32_t num_non_xgmi) {
  policy->num_non_xgmi = num_non_xgmi < ABCE_MAX_ENGINES ? num_non_xgmi : 0;
}

static inline uint32_t abce_sdma_policy_registered_engine(const abce_sdma_engine_policy_t* policy,
                                                          uint32_t hw_engine_id) {
  return hw_engine_id < ABCE_MAX_ENGINES && policy->hwid_to_reg[hw_engine_id] >= 0
             ? (uint32_t)policy->hwid_to_reg[hw_engine_id]
             : ABCE_MAX_ENGINES;
}

// Selects the host-copy heatmap profile for this part.
static inline void abce_sdma_policy_select_heatmap_profile(abce_sdma_engine_policy_t* policy,
                                                           uint8_t minor, uint8_t total_sdma) {
  policy->heatmap = abce_select_heatmap(minor, total_sdma);
}

// Installs caller-supplied topology. The route for platforms with no KFD sysfs:
// fill an abce_topology_data_t from whatever the OS exposes, then pair this with
// abce_sdma_policy_select_heatmap_profile() or
// abce_copy_orchestrator_init_device_profile().
static inline void abce_sdma_policy_set_topology(abce_sdma_engine_policy_t* policy,
                                                 const abce_topology_data_t* topology) {
  policy->topology = *topology;
}

// Loads KFD topology into a temporary snapshot and replaces active state only
// after a GPU is found. Returns false where KFD sysfs does not exist
// (ABCE_HAS_KFD_TOPOLOGY == 0), leaving topology untouched. |base_path| may be
// null for the default.
static inline bool abce_sdma_policy_load_topology_from_kfd(abce_sdma_engine_policy_t* policy,
                                                           const char* base_path) {
#if ABCE_HAS_KFD_TOPOLOGY
  abce_topology_data_t replacement;
  abce_topology_load_result_t result;
  if (!abce_topology_load_from_kfd(base_path, &replacement, &result)) return false;
  policy->topology = replacement;
  abce_sdma_policy_select_heatmap_profile(policy, result.gfx_minor, result.total_sdma);
  abce_sdma_policy_set_engine_split(policy, result.num_non_xgmi_sdma);
  return true;
#else
  (void)policy;
  (void)base_path;
  return false;
#endif  // ABCE_HAS_KFD_TOPOLOGY
}

// Classifies the transfer, asks the matching band hook for an ordered hw-id
// list, then resolves each id to its registered index, keeps only candidates,
// dedups, and caps at |max|.
static inline abce_policy_result_t abce_sdma_policy_rank_engines(
    const abce_sdma_engine_policy_t* policy, const abce_copy_endpoint_t* src,
    const abce_copy_endpoint_t* dst, uint64_t candidate_mask, uint32_t* out_engines,
    uint32_t max) {
  uint8_t hwids[ABCE_MAX_ENGINES];
  uint32_t num_hwids = 0;
  abce_policy_result_t result;
  result.disposition = ABCE_POLICY_DISPOSITION_NO_PREFERENCE;
  result.count = 0;

  switch (abce_classify_transfer(src, dst)) {
    case ABCE_TRANSFER_KIND_LOCAL_DEVICE_TO_DEVICE:
      num_hwids = policy->vtable->rank_local_d2d(policy, src, dst, hwids, ABCE_MAX_ENGINES);
      break;
    case ABCE_TRANSFER_KIND_PEER_TO_PEER:
      num_hwids = policy->vtable->rank_p2p(policy, src, dst, hwids, ABCE_MAX_ENGINES);
      break;
    case ABCE_TRANSFER_KIND_HOST_TO_DEVICE:
      num_hwids = policy->vtable->rank_h2d(policy, src, dst, hwids, ABCE_MAX_ENGINES);
      break;
    case ABCE_TRANSFER_KIND_DEVICE_TO_HOST:
      num_hwids = policy->vtable->rank_d2h(policy, src, dst, hwids, ABCE_MAX_ENGINES);
      break;
    case ABCE_TRANSFER_KIND_UNKNOWN:
      break;
  }

  if (num_hwids == 0) {
    result.disposition = ABCE_POLICY_DISPOSITION_NO_LEGAL_ENGINE;
    return result;
  }

  uint32_t count = 0;
  uint64_t emitted = 0;
  for (uint32_t idx = 0; idx < num_hwids && count < max; ++idx) {
    const uint8_t hw_engine_id = hwids[idx];
    if (hw_engine_id >= ABCE_MAX_ENGINES || policy->hwid_to_reg[hw_engine_id] < 0) continue;
    const uint32_t reg_index = (uint32_t)policy->hwid_to_reg[hw_engine_id];
    const uint64_t engine_bit = 1ull << reg_index;
    if ((emitted & engine_bit) || !(candidate_mask & engine_bit)) continue;
    emitted |= engine_bit;
    out_engines[count++] = reg_index;
  }
  result.disposition =
      count ? ABCE_POLICY_DISPOSITION_RANKED : ABCE_POLICY_DISPOSITION_NO_LEGAL_ENGINE;
  result.count = count;
  return result;
}

// Adapter matching abce_engine_policy_fn_t, so the built-in policy plugs into
// the same slot a caller-supplied one does.
static inline abce_policy_result_t abce_sdma_policy_trampoline(void* user_data,
                                                               const abce_copy_endpoint_t* src,
                                                               const abce_copy_endpoint_t* dst,
                                                               uint64_t candidate_mask,
                                                               uint32_t* out_engines,
                                                               uint32_t max) {
  return abce_sdma_policy_rank_engines((const abce_sdma_engine_policy_t*)user_data, src, dst,
                                       candidate_mask, out_engines, max);
}

//===----------------------------------------------------------------------===//
// abce_plan_t (output of the map phase)
//===----------------------------------------------------------------------===//

// One frame of the mapping decision: the group of packets that submit writes to
// a single engine's ring, in one reservation (one doorbell).
//
// A fan-out (e.g. a hipMemcpyBatchAsync spreading work over 4 devices) maps to
// several frames — one per ring index — so a single map can return up to
// ABCE_MAX_ENGINES frames to submit in parallel. frames[0] is always the
// coordinator (carries the prologue + epilogue).
typedef struct abce_plan_frame_t {
  uint32_t engine;   // selected registry index / ring. The client may replace this
                     // with any entry in ranked_engines before submit.
  uint32_t num_ops;  // bodies on this frame.
  uint32_t bytes;    // total packet bytes for this frame.
  uint8_t num_ranked_engines;      // valid entries in ranked_engines.
  bool has_engine_preference;      // policy ranked the alternatives; false means equal.
  bool coordinator;                // frames[0]: carries the prologue + epilogue.
  uint32_t ranked_engines[ABCE_MAX_ENGINE_CHOICES];  // legal alternatives; best first
                                                     // when has_engine_preference.
  // Non-owning view of this frame's `bytes` formed packets. Points into the
  // orchestrator's per-thread packet scratch (see the map phase); valid only
  // until the same thread maps another plan. Submit copies it into the ring.
  const char* packets;
} abce_plan_frame_t;

// The output of the map phase: a fully materialized batch ready for submit.
//
// Outputs:
//   frames[0..num_frames) — one per participating ring; frames[0] is the
//                           coordinator (carries prologue + epilogue). Each
//                           frame also exposes up to ABCE_MAX_ENGINE_CHOICES
//                           legal alternatives and records whether policy ranked
//                           them. A client may replace frame.engine with one of
//                           those registered indices before submit;
//                           multi-frame clients must keep the selected frame
//                           engines distinct.
//   num_frames       — number of distinct engines used.
//   multi            — true when the copy is spread across more than one SDMA
//                      engine (num_frames > 1), so the frames run in parallel
//                      and need cross-engine start/fan-in coordination.
//   coordination_signal / coordination_initial_value — the word submit arms
//                      before publishing, carrying the low-32 fan-in count and,
//                      when required, the bit-62 start gate. Normally
//                      abce_signal_ref_t::coordination_scratch; the output
//                      signal's own value only when the body signals are
//                      themselves the completion transitions.
typedef struct abce_plan_t {
  abce_plan_frame_t frames[ABCE_MAX_ENGINES];
  uint32_t num_frames;

  bool multi;
  bool start_gate_required;
  bool epilogue_required;
  uint64_t fan_in_count;
  void* coordination_signal;
  uint64_t coordination_initial_value;

  // Set of abce_transfer_kind_t values present in the batch, as
  // abce_transfer_kind_bit() flags. A batch may legitimately mix kinds (an H2D
  // and a D2H in one submission), so this is a set rather than a single label;
  // use abce_plan_uniform_transfer_kind() when a single label is what you want.
  uint32_t transfer_kinds;

  // Where submit stamps the execution descriptor, or null if the client did not
  // ask for one. Carried from the metadata because submit does not see it.
  void* execution_descriptor;

  // Index of the operation that failed to map, or UINT32_MAX. Meaningful only
  // when the map call returned a failure status.
  uint32_t failed_op;

  bool valid;
  bool submitted;
} abce_plan_t;

static inline void abce_plan_initialize(abce_plan_t* out_plan) {
  memset(out_plan, 0, sizeof(*out_plan));
  out_plan->epilogue_required = true;
  out_plan->failed_op = UINT32_MAX;
}

// Whether the batch spans more than one transfer kind.
static inline bool abce_plan_mixed_transfer_kinds(const abce_plan_t* plan) {
  return (plan->transfer_kinds & (plan->transfer_kinds - 1)) != 0;
}

// The batch's single transfer kind, or UNKNOWN if it mixes kinds. A profiler
// attributing one completion signal to one kind of copy can only do so when this
// is not UNKNOWN.
static inline abce_transfer_kind_t abce_plan_uniform_transfer_kind(const abce_plan_t* plan) {
  if (plan->transfer_kinds == 0 || abce_plan_mixed_transfer_kinds(plan))
    return ABCE_TRANSFER_KIND_UNKNOWN;
  return (abce_transfer_kind_t)abce_count_trailing_zeros_u64(plan->transfer_kinds);
}

//===----------------------------------------------------------------------===//
// Map-level limits and retuning knobs
//===----------------------------------------------------------------------===//

#define ABCE_MAX_BATCH_ENTRIES (64u * ABCE_KI)

// The back-to-back path serializes a whole batch onto one ring, so what it
// trades against fan-out is the batch's *total* bytes, not the size of any one
// entry: one ring pays the full total while E rings each pay roughly total/E,
// set against a fixed coordination cost for fanning out at all. Measured on the
// 8-rank all-gather batch (MI300X, 16 SDMA engines), fan-out sits at a ~27 us
// floor no matter how small the copies are, while one ring starts near 13 us,
// and the two curves cross just under 1 MiB of total batch traffic: at 512 KiB
// total one ring wins 22.4 vs. 28.2 us, at 1 MiB fan-out edges ahead 29.6 vs.
// 31.6 us, and by 4 MiB it is decisive, 37.2 vs. 92.2 us. So keep one ring
// through the region where it is ahead and hand everything past the crossover to
// fan-out. Override via ABCE_LINEAR_B2B_MAX_TOTAL.
//
// Writing this rule per copy is what made an 8 x 256 KiB batch serialize onto a
// single ring at 51.6 us where fan-out needed 32.3: a per-copy window is only
// valid at the entry count it was tuned on, and silently mis-sizes every other
// batch shape, since 64 x 8 KiB moves exactly as many bytes as 8 x 64 KiB.
#define ABCE_LINEAR_B2B_MAX_TOTAL_DEFAULT ((size_t)(512u * ABCE_KI))
#define ABCE_MULTICAST_MAX_SIZE ((size_t)(256u * ABCE_KI))
#define ABCE_LARGE_COPY_MIN_SIZE ((size_t)(((uint64_t)1) << 30))
#define ABCE_MAX_COPIES_PER_ENGINE 8u
#define ABCE_BROADCAST_MAX_SIZE_DEFAULT ((size_t)(16u * ABCE_KI))
#define ABCE_MULTICAST_MAX_DSTS ((uint32_t)ABCE_KI)
#define ABCE_BROADCAST_MAX_DSTS 2u

// Retuning knobs (bytes). The linear back-to-back window and the
// broadcast-packet cutoff can be overridden at process start via environment so
// they can be swept without recompiling. Accepts decimal or 0x-hex.
// Unset/empty falls back to the compiled default.
static inline size_t abce_env_size_or(const char* name, size_t fallback) {
  const char* raw = getenv(name);
  if (raw == NULL || *raw == '\0') return fallback;
  char* end = NULL;
  const unsigned long long parsed = strtoull(raw, &end, 0);
  if (end == raw) return fallback;
  return (size_t)parsed;
}

// Cached on first use. The race between two threads computing this is benign:
// parsing the same environment string is idempotent, so both store the same
// value. UINT64_MAX is the "not yet parsed" sentinel, which no real size reaches.
static inline size_t abce_cached_env_size(uint64_t* cache, const char* name, size_t fallback) {
  uint64_t value = __atomic_load_n(cache, __ATOMIC_RELAXED);
  if (value == UINT64_MAX) {
    value = (uint64_t)abce_env_size_or(name, fallback);
    __atomic_store_n(cache, value, __ATOMIC_RELAXED);
  }
  return (size_t)value;
}

static inline size_t abce_linear_b2b_max_total(void) {
  static uint64_t cache = UINT64_MAX;
  return abce_cached_env_size(&cache, "ABCE_LINEAR_B2B_MAX_TOTAL",
                              ABCE_LINEAR_B2B_MAX_TOTAL_DEFAULT);
}

static inline size_t abce_broadcast_max_size(void) {
  static uint64_t cache = UINT64_MAX;
  return abce_cached_env_size(&cache, "ABCE_BROADCAST_MAX", ABCE_BROADCAST_MAX_SIZE_DEFAULT);
}

//===----------------------------------------------------------------------===//
// Per-thread packet scratch
//===----------------------------------------------------------------------===//

// The map phase builds packets into a per-thread scratch buffer (relocatable —
// no ring offset embedded). Reusing the scratch avoids a per-call heap
// allocation on the hot map->submit path; it grows once to the batch high-water
// mark. A pthread key gives the buffer a destructor so it is released at thread
// exit, which is what the C++ thread_local unique_ptr did.
typedef struct abce_packet_scratch_t {
  char* buffer;
  size_t capacity;
} abce_packet_scratch_t;

static inline void abce_packet_scratch_destroy(void* scratch) {
  abce_packet_scratch_t* state = (abce_packet_scratch_t*)scratch;
  if (!state) return;
  free(state->buffer);
  free(state);
}

static pthread_key_t abce_packet_scratch_key;
static pthread_once_t abce_packet_scratch_key_once = PTHREAD_ONCE_INIT;

static inline void abce_packet_scratch_key_create(void) {
  (void)pthread_key_create(&abce_packet_scratch_key, abce_packet_scratch_destroy);
}

// Returns a per-thread packet scratch buffer of at least |bytes|, growing it to
// the high-water mark. The buffer is reused across map calls on the same thread,
// so a plan's packet views are valid only until that thread maps another plan.
// Returns null only on allocation failure.
static inline char* abce_acquire_packet_scratch(size_t bytes) {
  (void)pthread_once(&abce_packet_scratch_key_once, abce_packet_scratch_key_create);
  abce_packet_scratch_t* state =
      (abce_packet_scratch_t*)pthread_getspecific(abce_packet_scratch_key);
  if (!state) {
    state = (abce_packet_scratch_t*)calloc(1, sizeof(*state));
    if (!state) return NULL;
    if (pthread_setspecific(abce_packet_scratch_key, state) != 0) {
      free(state);
      return NULL;
    }
  }
  if (bytes == 0) return state->buffer;
  if (state->capacity < bytes) {
    char* grown = (char*)realloc(state->buffer, bytes);
    if (!grown) return NULL;
    state->buffer = grown;
    state->capacity = bytes;
  }
  return state->buffer;
}

//===----------------------------------------------------------------------===//
// abce_copy_orchestrator_t — the ABCE front door
//===----------------------------------------------------------------------===//

// Maps batches of copies onto registered SDMA engines, builds the SDMA packets,
// then submits them.
//
// Two phases:
//   * abce_copy_orchestrator_map_copy() — pick engines, decompose copies, and
//     emit packets into owned buffers. Does not touch a ring or signal.
//   * abce_copy_orchestrator_submit()   — reserve every ring, arm the output
//     signal, memcpy each frame's pre-built packets in, then release.
//
// The orchestrator must remain at a stable address while plans reference it.
typedef struct abce_copy_orchestrator_t {
  const abce_builder_t* builder;
  bool is_gfx125plus;
  abce_platform_caps_t caps;
  abce_frame_composer_t composer;

  abce_ring_t* engines[ABCE_MAX_ENGINES];
  abce_engine_affinity_t affinity[ABCE_MAX_ENGINES];
  uint64_t registered_mask;

  // Auto-selected for the part; held by value rather than behind a pointer
  // because it is plain data plus a vtable. has_policy is false on parts with
  // no specialized policy.
  abce_sdma_engine_policy_t owned_policy;
  bool has_policy;

  abce_engine_policy_fn_t policy;
  void* policy_user_data;
} abce_copy_orchestrator_t;

static inline void abce_copy_orchestrator_initialize_with_caps(
    const abce_builder_t* builder, abce_platform_caps_t caps,
    abce_copy_orchestrator_t* out_orchestrator) {
  memset(out_orchestrator, 0, sizeof(*out_orchestrator));
  out_orchestrator->builder = builder;
  out_orchestrator->is_gfx125plus = abce_builder_is_gfx125plus(builder);
  out_orchestrator->caps = caps;
  abce_frame_composer_initialize(builder, caps, &out_orchestrator->composer);
  out_orchestrator->has_policy =
      abce_sdma_policy_initialize_for_isa(builder->isa, &out_orchestrator->owned_policy);
  if (out_orchestrator->has_policy) {
    out_orchestrator->policy = abce_sdma_policy_trampoline;
    out_orchestrator->policy_user_data = &out_orchestrator->owned_policy;
  }
}

static inline void abce_copy_orchestrator_initialize(
    const abce_builder_t* builder, abce_copy_orchestrator_t* out_orchestrator) {
  abce_copy_orchestrator_initialize_with_caps(
      builder, abce_detect_default_platform_caps(builder->isa), out_orchestrator);
}

// The SDMA engine policy auto-selected for this part, or null if the part has
// none. Engines are mapped automatically by registration (via
// abce_engine_affinity_t); use this to load the remaining topology. The
// device-wide heatmap profile is selected by
// abce_sdma_policy_load_topology_from_kfd() or
// abce_copy_orchestrator_init_device_profile().
static inline abce_sdma_engine_policy_t* abce_copy_orchestrator_sdma_policy(
    abce_copy_orchestrator_t* orchestrator) {
  return orchestrator->has_policy ? &orchestrator->owned_policy : NULL;
}

// Replaces the auto-selected policy with a caller-supplied ranking callback.
// Passing null restores no-preference fallback behavior.
static inline void abce_copy_orchestrator_set_engine_policy(
    abce_copy_orchestrator_t* orchestrator, abce_engine_policy_fn_t policy, void* user_data) {
  orchestrator->policy = policy;
  orchestrator->policy_user_data = policy ? user_data : NULL;
}

// Establishes device-wide selection state that does not depend on which
// individual engines are registered. Selects the host-copy heatmap profile for
// this part from its total SDMA engine count (host + xGMI) — a fixed device
// property, so this is chosen once. Call before registering engines.
//
// The KFD/topology path (abce_sdma_policy_load_topology_from_kfd) selects the
// same profile from the count KFD reports; call this when KFD topology loading
// is disabled.
//
// |num_non_xgmi_sdma_engines| is KFD's num_sdma_engines: how many engines are
// not xGMI, the driver numbering them first. It establishes the P2P band the way
// the KFD path does. Zero leaves the band unknown, which widens P2P to every
// registered engine rather than emptying it.
static inline void abce_copy_orchestrator_init_device_profile(
    abce_copy_orchestrator_t* orchestrator, uint32_t total_sdma_engines,
    uint32_t num_non_xgmi_sdma_engines) {
  if (!orchestrator->has_policy) return;
  const abce_isa_version_t isa = orchestrator->builder->isa;
  if (isa.major == 9 && (isa.minor == 4 || isa.minor == 5))
    abce_sdma_policy_select_heatmap_profile(&orchestrator->owned_policy, (uint8_t)isa.minor,
                                            (uint8_t)total_sdma_engines);
  abce_sdma_policy_set_engine_split(&orchestrator->owned_policy, num_non_xgmi_sdma_engines);
}

// Registers an engine's ring and maps its hardware id into the auto-selected
// SDMA policy, so the policy is populated without a separate map-engine call.
// |affinity| may be null for the default (registered index as hardware id).
static inline bool abce_copy_orchestrator_register_engine(
    abce_copy_orchestrator_t* orchestrator, uint32_t index, abce_ring_t* ring,
    const abce_engine_affinity_t* affinity) {
  if (index >= ABCE_MAX_ENGINES || !ring || orchestrator->engines[index] != NULL) return false;
  abce_engine_affinity_t resolved_affinity;
  if (affinity) {
    resolved_affinity = *affinity;
  } else {
    abce_engine_affinity_initialize(&resolved_affinity);
  }
  if (resolved_affinity.hw_engine_id == ABCE_ANY_ENGINE) resolved_affinity.hw_engine_id = index;
  // A hardware id the policy cannot map would leave the engine registered but
  // unrankable, so refuse it here rather than half-register it.
  if (resolved_affinity.hw_engine_id >= ABCE_MAX_ENGINES) return false;
  orchestrator->engines[index] = ring;
  orchestrator->registered_mask |= (1ull << index);
  orchestrator->affinity[index] = resolved_affinity;
  if (orchestrator->has_policy)
    abce_sdma_policy_map_engine(&orchestrator->owned_policy, resolved_affinity.hw_engine_id,
                                index);
  return true;
}

// Affinity a registered engine was registered with, resolved (so hw_engine_id is
// never ABCE_ANY_ENGINE for a registered index). Reporting a copy to anything
// outside ABCE wants the hardware id from here, not the registered index, which
// is a client-local numbering.
static inline const abce_engine_affinity_t* abce_copy_orchestrator_engine_affinity(
    const abce_copy_orchestrator_t* orchestrator, uint32_t index) {
  return &orchestrator->affinity[index];
}

// Summarizes a plan's final engine assignment for a reader of its completion
// signal. Public so a client that keeps this metadata somewhere other than
// abce_copy_metadata_t::execution_descriptor can encode the same word itself.
static inline uint32_t abce_copy_orchestrator_describe_execution(
    const abce_copy_orchestrator_t* orchestrator, const abce_plan_t* plan) {
  uint32_t hw_engine_mask = 0;
  for (uint32_t frame_idx = 0; frame_idx < plan->num_frames; ++frame_idx)
    hw_engine_mask |= ((uint32_t)1)
                      << orchestrator->affinity[plan->frames[frame_idx].engine].hw_engine_id;
  return abce_execution_descriptor_encode(ABCE_EXECUTION_ENGINE_KIND_SDMA, hw_engine_mask,
                                          abce_plan_uniform_transfer_kind(plan),
                                          abce_plan_mixed_transfer_kinds(plan));
}

//===----------------------------------------------------------------------===//
// Candidate resolution and policy invocation
//===----------------------------------------------------------------------===//

static inline uint64_t abce_copy_orchestrator_candidates(
    const abce_copy_orchestrator_t* orchestrator, const abce_copy_metadata_t* metadata) {
  return metadata->engine_mask ? (metadata->engine_mask & orchestrator->registered_mask)
                               : orchestrator->registered_mask;
}

static inline uint32_t abce_engine_at(uint64_t mask, uint32_t nth) {
  for (uint32_t engine = 0; engine < ABCE_MAX_ENGINES; ++engine) {
    if (mask & (1ull << engine)) {
      if (nth == 0) return engine;
      --nth;
    }
  }
  return ABCE_MAX_ENGINES;
}

static inline uint32_t abce_popcount_engines(uint64_t mask) {
  return (uint32_t)abce_popcount_u64(mask);
}

static inline abce_policy_result_t abce_copy_orchestrator_invoke_policy(
    const abce_copy_orchestrator_t* orchestrator, const abce_copy_op_t* op, uint64_t candidates,
    uint32_t* ranked) {
  abce_policy_result_t result;
  result.disposition = ABCE_POLICY_DISPOSITION_NO_PREFERENCE;
  result.count = 0;
  if (!orchestrator->policy) return result;

  result = orchestrator->policy(orchestrator->policy_user_data, &op->src_end, &op->dst_end,
                                candidates, ranked, ABCE_MAX_ENGINES);
  if (result.disposition != ABCE_POLICY_DISPOSITION_RANKED) {
    result.count = 0;
    return result;
  }

  const uint32_t reported_count =
      result.count < ABCE_MAX_ENGINES ? result.count : ABCE_MAX_ENGINES;
  uint64_t emitted = 0;
  uint32_t valid_count = 0;
  for (uint32_t rank_idx = 0; rank_idx < reported_count; ++rank_idx) {
    const uint32_t engine = ranked[rank_idx];
    if (engine >= ABCE_MAX_ENGINES) continue;
    const uint64_t engine_bit = 1ull << engine;
    if (!(candidates & engine_bit) || (emitted & engine_bit)) continue;
    emitted |= engine_bit;
    ranked[valid_count++] = engine;
  }
  result.disposition =
      valid_count ? ABCE_POLICY_DISPOSITION_RANKED : ABCE_POLICY_DISPOSITION_NO_LEGAL_ENGINE;
  result.count = valid_count;
  return result;
}

static inline uint32_t abce_copy_orchestrator_best_engine(
    const abce_copy_orchestrator_t* orchestrator, const abce_copy_op_t* op, uint64_t candidates) {
  uint32_t ranked[ABCE_MAX_ENGINES] = {0};
  const abce_policy_result_t result =
      abce_copy_orchestrator_invoke_policy(orchestrator, op, candidates, ranked);
  if (result.disposition == ABCE_POLICY_DISPOSITION_NO_LEGAL_ENGINE) return ABCE_MAX_ENGINES;
  if (result.disposition == ABCE_POLICY_DISPOSITION_RANKED) return ranked[0];
  return abce_first_engine(candidates);
}

static inline uint32_t abce_copy_orchestrator_back_to_back_engine(
    const abce_copy_orchestrator_t* orchestrator, const abce_copy_op_t* first_copy,
    uint64_t candidates) {
  const abce_isa_version_t isa = orchestrator->builder->isa;
  if (orchestrator->has_policy && isa.major == 9 && isa.minor >= 4) {
    const uint32_t reversed_host_engine =
        abce_sdma_policy_registered_engine(&orchestrator->owned_policy, 1);
    if (reversed_host_engine < ABCE_MAX_ENGINES &&
        (candidates & (((uint64_t)1) << reversed_host_engine)))
      return reversed_host_engine;
  }
  return abce_copy_orchestrator_best_engine(orchestrator, first_copy, candidates);
}

static inline uint8_t abce_copy_orchestrator_rank_engine_choices(
    const abce_copy_orchestrator_t* orchestrator, const abce_copy_op_t* op, uint64_t candidates,
    uint32_t* out_choices, bool* out_has_engine_preference) {
  uint32_t ranked[ABCE_MAX_ENGINES] = {0};
  const abce_policy_result_t result =
      abce_copy_orchestrator_invoke_policy(orchestrator, op, candidates, ranked);
  *out_has_engine_preference = result.disposition == ABCE_POLICY_DISPOSITION_RANKED;
  if (result.disposition == ABCE_POLICY_DISPOSITION_NO_LEGAL_ENGINE) return 0;

  const uint32_t num_candidates = abce_popcount_engines(candidates);
  const uint32_t num_choices =
      result.disposition == ABCE_POLICY_DISPOSITION_RANKED
          ? (result.count < ABCE_MAX_ENGINE_CHOICES ? result.count : ABCE_MAX_ENGINE_CHOICES)
          : (num_candidates < ABCE_MAX_ENGINE_CHOICES ? num_candidates
                                                      : ABCE_MAX_ENGINE_CHOICES);
  for (uint32_t choice_idx = 0; choice_idx < num_choices; ++choice_idx) {
    out_choices[choice_idx] = result.disposition == ABCE_POLICY_DISPOSITION_RANKED
                                  ? ranked[choice_idx]
                                  : abce_engine_at(candidates, choice_idx);
  }
  return (uint8_t)num_choices;
}

// Up to |capacity| legal engines for |copy|, ranked best-first.
static inline uint32_t abce_copy_orchestrator_rank_legal_engines(
    const abce_copy_orchestrator_t* orchestrator, const abce_copy_op_t* copy, uint32_t* out_engines,
    uint32_t capacity);

static inline uint32_t abce_copy_orchestrator_select_engine(
    const abce_copy_orchestrator_t* orchestrator, const abce_copy_op_t* op, uint64_t candidates,
    bool can_fanout, uint32_t max_participants, uint32_t* round_robin, uint64_t* selected_mask,
    uint32_t* out_ranked_engines, uint8_t* out_num_ranked_engines,
    bool* out_has_engine_preference) {
  const bool participant_cap_reached =
      abce_popcount_engines(*selected_mask) >= max_participants;
  const uint64_t unselected_candidates = candidates & ~*selected_mask;
  const bool add_participant =
      can_fanout && !participant_cap_reached && unselected_candidates != 0;
  const uint64_t allowed =
      add_participant ? unselected_candidates
                      : ((*selected_mask != 0 && (!can_fanout || participant_cap_reached))
                             ? *selected_mask
                             : candidates);
  *out_num_ranked_engines = abce_copy_orchestrator_rank_engine_choices(
      orchestrator, op, candidates, out_ranked_engines, out_has_engine_preference);
  uint32_t allowed_ranked_engines[ABCE_MAX_ENGINE_CHOICES] = {0};
  bool allowed_has_engine_preference = false;
  const uint8_t num_allowed_ranked_engines = abce_copy_orchestrator_rank_engine_choices(
      orchestrator, op, allowed, allowed_ranked_engines, &allowed_has_engine_preference);
  if (num_allowed_ranked_engines == 0) return ABCE_MAX_ENGINES;
  const uint32_t engine =
      (add_participant || !can_fanout)
          ? allowed_ranked_engines[0]
          : allowed_ranked_engines[(*round_robin)++ % num_allowed_ranked_engines];
  if (engine < ABCE_MAX_ENGINES) *selected_mask |= ((uint64_t)1) << engine;
  return engine;
}

//===----------------------------------------------------------------------===//
// Fan-out engine assignment
//
// Fan-out spreads a batch across engines for parallelism. The rule is "parallel
// first, preferred only when doubling up": every ring legal for a copy takes a
// copy before any ring takes a second, and once the copies outnumber the legal
// rings the extra copies prefer their recommended (affinity map) ring. A
// per-ring load counter drives this.
//
//   * gfx94x/95x — a copy's legal set is class-based (P2P -> xGMI rings; local
//     D2D / host copies -> host rings) with the affinity ring ranked first. The
//     classes are disjoint, so P2P copies balance across the xGMI rings while
//     local copies run in parallel on host rings, with no cross-class
//     contention.
//   * gfx1250 — every ring is equivalent: the legal set is simply every
//     candidate ring and there is no recommended ring, so the balancer
//     degenerates to round-robin, independent of src/dst.
//===----------------------------------------------------------------------===//

// Least-loaded legal ring, preferring |recommended| on a tie so a copy stays on
// its affinity ring until that ring is as busy as the alternatives. Returns
// ABCE_MAX_ENGINES when |legal_mask| is empty.
static inline uint32_t abce_pick_balanced_engine(uint64_t legal_mask, uint32_t recommended,
                                                 const uint32_t* load) {
  uint32_t min_load = UINT32_MAX;
  for (uint32_t engine = 0; engine < ABCE_MAX_ENGINES; ++engine)
    if ((legal_mask & (((uint64_t)1) << engine)) && load[engine] < min_load)
      min_load = load[engine];
  if (min_load == UINT32_MAX) return ABCE_MAX_ENGINES;
  if (recommended < ABCE_MAX_ENGINES && (legal_mask & (((uint64_t)1) << recommended)) &&
      load[recommended] == min_load)
    return recommended;
  for (uint32_t engine = 0; engine < ABCE_MAX_ENGINES; ++engine)
    if ((legal_mask & (((uint64_t)1) << engine)) && load[engine] == min_load) return engine;
  return ABCE_MAX_ENGINES;
}

// Legal ring set for |op|. On gfx1250 every candidate ring is legal and no ring
// is preferred; elsewhere the policy ranks the class-legal rings, with the
// recommended (affinity) ring first. Also fills |out_ranked_engines| for the
// engine op so downstream ring substitution still has the ordered choices.
static inline uint64_t abce_copy_orchestrator_legal_engine_set(
    const abce_copy_orchestrator_t* orchestrator, const abce_copy_op_t* op, uint64_t candidates,
    uint32_t* out_recommended, uint32_t* out_ranked_engines, uint8_t* out_num_ranked_engines,
    bool* out_has_engine_preference) {
  if (orchestrator->is_gfx125plus) {
    *out_num_ranked_engines = 0;
    for (uint32_t engine = 0;
         engine < ABCE_MAX_ENGINES && *out_num_ranked_engines < ABCE_MAX_ENGINE_CHOICES; ++engine)
      if (candidates & (((uint64_t)1) << engine))
        out_ranked_engines[(*out_num_ranked_engines)++] = engine;
    *out_has_engine_preference = false;
    *out_recommended = ABCE_MAX_ENGINES;
    return candidates;
  }
  *out_num_ranked_engines = abce_copy_orchestrator_rank_engine_choices(
      orchestrator, op, candidates, out_ranked_engines, out_has_engine_preference);
  uint64_t legal_mask = 0;
  for (uint32_t choice_idx = 0; choice_idx < *out_num_ranked_engines; ++choice_idx)
    legal_mask |= ((uint64_t)1) << out_ranked_engines[choice_idx];
  *out_recommended = *out_num_ranked_engines ? out_ranked_engines[0] : ABCE_MAX_ENGINES;
  return legal_mask;
}

static inline uint32_t abce_copy_orchestrator_rank_legal_engines(
    const abce_copy_orchestrator_t* orchestrator, const abce_copy_op_t* copy, uint32_t* out_engines,
    uint32_t capacity) {
  if (!out_engines || capacity == 0) return 0;
  uint32_t ranked[ABCE_MAX_ENGINE_CHOICES] = {0};
  uint32_t recommended = ABCE_MAX_ENGINES;
  uint8_t num_ranked = 0;
  bool has_engine_preference = false;
  (void)abce_copy_orchestrator_legal_engine_set(orchestrator, copy,
                                                orchestrator->registered_mask, &recommended,
                                                ranked, &num_ranked, &has_engine_preference);
  const uint32_t count = num_ranked < capacity ? num_ranked : capacity;
  for (uint32_t rank = 0; rank < count; ++rank) out_engines[rank] = ranked[rank];
  return count;
}

//===----------------------------------------------------------------------===//
// Validation
//===----------------------------------------------------------------------===//

// Most packets |copy| can take on whichever path the planner picks: a swap may
// go out plain or fused, and a multicast decomposed into per-destination linear
// copies takes the same count as the fused packet. Rect tiling is counted by its
// own check, and an indirect copy is one packet.
static inline uint64_t abce_copy_op_max_packets(const abce_builder_t* builder,
                                                const abce_copy_op_t* copy) {
  switch (copy->kind) {
    case ABCE_OP_KIND_LINEAR:
    case ABCE_OP_KIND_MULTICAST:
    case ABCE_OP_KIND_BROADCAST:
      return abce_builder_num_copy_packets(builder, copy->size);
    case ABCE_OP_KIND_SWAP:
      return ABCE_MAX(abce_builder_num_swap_packets(builder, copy->size),
                      abce_builder_num_wait_signal_swap_packets(
                          builder, copy->size, copy->size2 ? copy->size2 : copy->size));
    case ABCE_OP_KIND_FILL:
      return abce_builder_num_fill_packets(builder, copy->size / sizeof(uint32_t));
    case ABCE_OP_KIND_INDIRECT:
    case ABCE_OP_KIND_COPY_RECT:
      break;
  }
  return 1;
}

static inline abce_status_t abce_copy_orchestrator_validate_request(
    const abce_copy_orchestrator_t* orchestrator, const abce_copy_op_t* copies,
    uint32_t num_copies, const abce_copy_metadata_t* metadata, uint32_t* out_failed_op) {
  if (!copies || num_copies == 0 || num_copies > ABCE_MAX_BATCH_ENTRIES || !metadata->out.value ||
      (metadata->num_deps != 0 && !metadata->deps))
    return ABCE_STATUS_INVALID_ARGUMENT;

  for (uint32_t op_idx = 0; op_idx < num_copies; ++op_idx) {
    const abce_copy_op_t* copy = &copies[op_idx];
    *out_failed_op = op_idx;
    if (copy->src_end.kind == ABCE_ENDPOINT_KIND_HOST &&
        copy->dst_end.kind == ABCE_ENDPOINT_KIND_HOST)
      return ABCE_STATUS_INVALID_ARGUMENT;
    if (copy->kind == ABCE_OP_KIND_COPY_RECT) {
      if (!copy->rect || !copy->rect->src.base || !copy->rect->dst.base ||
          copy->rect->range.x == 0 || copy->rect->range.y == 0 || copy->rect->range.z == 0)
        return ABCE_STATUS_INVALID_ARGUMENT;
      // Tiling can reject a pitch/slice the packet cannot encode. The C++
      // version discovered this as an exception thrown out of frame sizing;
      // checking here reports it against the operation that caused it.
      uint32_t num_rect_packets = 0;
      const abce_status_t rect_status = abce_builder_num_rect_packets(
          orchestrator->builder, &copy->rect->dst, &copy->rect->dst_offset, &copy->rect->src,
          &copy->rect->src_offset, &copy->rect->range, &num_rect_packets);
      if (!abce_status_is_ok(rect_status)) return rect_status;
      continue;
    }
    if (copy->size == 0) return ABCE_STATUS_INVALID_ARGUMENT;

    switch (copy->kind) {
      case ABCE_OP_KIND_LINEAR:
      case ABCE_OP_KIND_SWAP:
        if (!copy->src || !copy->dst) return ABCE_STATUS_INVALID_ARGUMENT;
        break;
      case ABCE_OP_KIND_INDIRECT:
        if (!orchestrator->is_gfx125plus) return ABCE_STATUS_UNIMPLEMENTED;
        if (!copy->src || !copy->dst || copy->size > (((uint64_t)1) << 30))
          return ABCE_STATUS_INVALID_ARGUMENT;
        break;
      case ABCE_OP_KIND_MULTICAST:
      case ABCE_OP_KIND_BROADCAST:
        if (!copy->src || !copy->dsts || copy->num_dsts == 0) return ABCE_STATUS_INVALID_ARGUMENT;
        if (copy->kind == ABCE_OP_KIND_BROADCAST && copy->num_dsts != ABCE_BROADCAST_MAX_DSTS)
          return ABCE_STATUS_INVALID_ARGUMENT;
        for (uint32_t dst_idx = 0; dst_idx < copy->num_dsts; ++dst_idx)
          if (!copy->dsts[dst_idx]) return ABCE_STATUS_INVALID_ARGUMENT;
        break;
      case ABCE_OP_KIND_FILL:
        if (!copy->dst || (copy->size % sizeof(uint32_t)) != 0)
          return ABCE_STATUS_INVALID_ARGUMENT;
        break;
      case ABCE_OP_KIND_COPY_RECT:
        break;
    }
    if (abce_copy_op_max_packets(orchestrator->builder, copy) > ABCE_MAX_OP_PACKETS)
      return ABCE_STATUS_OUT_OF_RANGE;
  }
  *out_failed_op = UINT32_MAX;
  return ABCE_STATUS_OK;
}

//===----------------------------------------------------------------------===//
// Decomposition
//===----------------------------------------------------------------------===//

// Bytes the back-to-back path would push through a single ring. A multicast op
// decomposed on non-gfx125plus writes its payload once per destination, so its
// share scales with the destination count rather than being one copy.
static inline size_t abce_back_to_back_bytes(const abce_copy_op_t* copies, uint32_t num_copies) {
  size_t total = 0;
  for (uint32_t copy_idx = 0; copy_idx < num_copies; ++copy_idx) {
    const abce_copy_op_t* copy = &copies[copy_idx];
    const uint32_t writes = copy->kind == ABCE_OP_KIND_MULTICAST ? copy->num_dsts : 1;
    total += copy->size * writes;
  }
  return total;
}

static inline bool abce_copy_orchestrator_is_back_to_back_batch(
    const abce_copy_orchestrator_t* orchestrator, const abce_copy_op_t* copies,
    uint32_t num_copies, const abce_copy_metadata_t* metadata) {
  if (metadata->linear_batch_mode == ABCE_LINEAR_BATCH_MODE_FORCE_FAN_OUT) return false;
  // A single multicast is eligible because non-gfx125plus decomposes it into one
  // write per destination, which is the same back-to-back shape a linear batch
  // has. Every other single-op batch has nothing to serialize.
  const bool decomposed_multicast = num_copies == 1 &&
                                    copies[0].kind == ABCE_OP_KIND_MULTICAST &&
                                    !orchestrator->is_gfx125plus;
  if (!decomposed_multicast) {
    if (num_copies <= 1) return false;
    for (uint32_t copy_idx = 0; copy_idx < num_copies; ++copy_idx)
      if (copies[copy_idx].kind != ABCE_OP_KIND_LINEAR) return false;
  }
  if (metadata->linear_batch_mode == ABCE_LINEAR_BATCH_MODE_FORCE_BACK_TO_BACK) return true;
  return abce_back_to_back_bytes(copies, num_copies) <= abce_linear_b2b_max_total();
}

static inline bool abce_push_engine_op(abce_engine_op_t* ops, uint32_t* num_ops, uint32_t capacity,
                                       const abce_engine_op_t* op) {
  if (*num_ops >= capacity) return false;
  ops[(*num_ops)++] = *op;
  return true;
}

static inline void abce_engine_op_initialize(abce_engine_op_t* out_op) {
  memset(out_op, 0, sizeof(*out_op));
  out_op->kind = ABCE_OP_KIND_LINEAR;
}

static inline void abce_set_engine_choices(abce_engine_op_t* op, const uint32_t* ranked_engines,
                                           uint8_t num_ranked_engines,
                                           bool has_engine_preference) {
  op->num_ranked_engines = num_ranked_engines;
  op->has_engine_preference = has_engine_preference;
  for (uint32_t choice_idx = 0; choice_idx < num_ranked_engines; ++choice_idx)
    op->ranked_engines[choice_idx] = ranked_engines[choice_idx];
}

static inline void abce_copy_orchestrator_set_fixed_engine(
    const abce_copy_orchestrator_t* orchestrator, abce_engine_op_t* op,
    const abce_copy_op_t* copy, uint64_t candidates, uint32_t engine) {
  op->engine = engine;
  bool has_engine_preference = false;
  op->num_ranked_engines = abce_copy_orchestrator_rank_engine_choices(
      orchestrator, copy, candidates, op->ranked_engines, &has_engine_preference);
  op->has_engine_preference = has_engine_preference;
  if (op->num_ranked_engines == 0) {
    op->ranked_engines[0] = engine;
    op->num_ranked_engines = 1;
  }
}

static inline bool abce_copy_orchestrator_decompose_multi_dst(
    const abce_copy_orchestrator_t* orchestrator, const abce_copy_op_t* op,
    const abce_copy_metadata_t* metadata, uint64_t candidates, bool can_fanout,
    uint32_t max_participants, bool force_back_to_back, uint32_t fixed_engine,
    uint32_t* round_robin, uint64_t* selected_mask, abce_engine_op_t* ops, uint32_t* num_ops,
    uint32_t capacity) {
  // gfx125plus multicast: one packet handles up to ABCE_MULTICAST_MAX_DSTS
  // destinations. If num_dsts exceeds that, chunk into multiple ops spread
  // across engines.
  const bool use_multicast =
      metadata->multicast_mode == ABCE_MULTICAST_MODE_FORCE_MULTICAST ||
      (metadata->multicast_mode == ABCE_MULTICAST_MODE_AUTOMATIC &&
       op->size <= ABCE_MULTICAST_MAX_SIZE);
  if (orchestrator->is_gfx125plus && use_multicast) {
    uint32_t ranked_engines[ABCE_MAX_ENGINE_CHOICES] = {0};
    uint8_t num_ranked_engines = 0;
    bool has_engine_preference = false;
    const uint32_t multicast_engine =
        force_back_to_back
            ? fixed_engine
            : abce_copy_orchestrator_select_engine(
                  orchestrator, op, candidates, can_fanout, max_participants, round_robin,
                  selected_mask, ranked_engines, &num_ranked_engines, &has_engine_preference);
    if (multicast_engine == ABCE_MAX_ENGINES) return false;
    if (force_back_to_back) {
      num_ranked_engines = abce_copy_orchestrator_rank_engine_choices(
          orchestrator, op, candidates, ranked_engines, &has_engine_preference);
      if (num_ranked_engines == 0) {
        ranked_engines[0] = fixed_engine;
        num_ranked_engines = 1;
      }
    }
    for (uint32_t base = 0; base < op->num_dsts; base += ABCE_MULTICAST_MAX_DSTS) {
      const uint32_t chunk_dsts = (op->num_dsts - base < ABCE_MULTICAST_MAX_DSTS)
                                      ? (op->num_dsts - base)
                                      : ABCE_MULTICAST_MAX_DSTS;
      abce_engine_op_t engine_op;
      abce_engine_op_initialize(&engine_op);
      engine_op.kind = ABCE_OP_KIND_MULTICAST;
      engine_op.src = op->src;
      engine_op.dsts = &op->dsts[base];
      engine_op.num_dsts = chunk_dsts;
      engine_op.size = op->size;
      engine_op.engine = multicast_engine;
      abce_set_engine_choices(&engine_op, ranked_engines, num_ranked_engines,
                              has_engine_preference);
      if (!abce_push_engine_op(ops, num_ops, capacity, &engine_op)) return false;
    }
    return true;
  }

  if (!orchestrator->is_gfx125plus && op->size < abce_broadcast_max_size()) {
    uint32_t ranked_engines[ABCE_MAX_ENGINE_CHOICES] = {0};
    uint8_t num_ranked_engines = 0;
    bool has_engine_preference = false;
    const uint32_t bcast_engine =
        force_back_to_back
            ? fixed_engine
            : abce_copy_orchestrator_select_engine(
                  orchestrator, op, candidates, can_fanout, max_participants, round_robin,
                  selected_mask, ranked_engines, &num_ranked_engines, &has_engine_preference);
    if (bcast_engine == ABCE_MAX_ENGINES) return false;
    if (force_back_to_back) {
      num_ranked_engines = abce_copy_orchestrator_rank_engine_choices(
          orchestrator, op, candidates, ranked_engines, &has_engine_preference);
      if (num_ranked_engines == 0) {
        ranked_engines[0] = fixed_engine;
        num_ranked_engines = 1;
      }
    }
    for (uint32_t base = 0; base < op->num_dsts; base += ABCE_BROADCAST_MAX_DSTS) {
      const uint32_t chunk_dsts = (op->num_dsts - base < ABCE_BROADCAST_MAX_DSTS)
                                      ? (op->num_dsts - base)
                                      : ABCE_BROADCAST_MAX_DSTS;
      abce_engine_op_t engine_op;
      abce_engine_op_initialize(&engine_op);
      engine_op.src = op->src;
      engine_op.dsts = &op->dsts[base];
      engine_op.size = op->size;
      engine_op.engine = bcast_engine;
      abce_set_engine_choices(&engine_op, ranked_engines, num_ranked_engines,
                              has_engine_preference);
      if (chunk_dsts == ABCE_BROADCAST_MAX_DSTS) {
        engine_op.kind = ABCE_OP_KIND_BROADCAST;
        engine_op.num_dsts = ABCE_BROADCAST_MAX_DSTS;
      } else {
        engine_op.kind = ABCE_OP_KIND_LINEAR;
        engine_op.dst = op->dsts[base];
      }
      if (!abce_push_engine_op(ops, num_ops, capacity, &engine_op)) return false;
    }
    return true;
  }

  const uint32_t num_candidates = abce_popcount_engines(candidates);
  const uint32_t num_group_engines =
      max_participants < num_candidates ? max_participants : num_candidates;
  const bool use_large_copy_grouping =
      orchestrator->is_gfx125plus && can_fanout && op->size >= ABCE_LARGE_COPY_MIN_SIZE;

  for (uint32_t dst_idx = 0; dst_idx < op->num_dsts; ++dst_idx) {
    abce_engine_op_t engine_op;
    abce_engine_op_initialize(&engine_op);
    engine_op.kind = ABCE_OP_KIND_LINEAR;
    engine_op.src = op->src;
    engine_op.dst = op->dsts[dst_idx];
    engine_op.size = op->size;
    if (force_back_to_back) {
      abce_copy_orchestrator_set_fixed_engine(orchestrator, &engine_op, op, candidates,
                                              fixed_engine);
    } else if (use_large_copy_grouping) {
      const uint32_t engine_slot = (dst_idx / ABCE_MAX_COPIES_PER_ENGINE) % num_group_engines;
      abce_copy_orchestrator_set_fixed_engine(orchestrator, &engine_op, op, candidates,
                                              abce_engine_at(candidates, engine_slot));
    } else {
      bool has_engine_preference = false;
      engine_op.engine = abce_copy_orchestrator_select_engine(
          orchestrator, op, candidates, can_fanout, max_participants, round_robin, selected_mask,
          engine_op.ranked_engines, &engine_op.num_ranked_engines, &has_engine_preference);
      engine_op.has_engine_preference = has_engine_preference;
    }
    if (engine_op.engine == ABCE_MAX_ENGINES) return false;
    if (!abce_push_engine_op(ops, num_ops, capacity, &engine_op)) return false;
  }
  return true;
}

static inline bool abce_copy_orchestrator_decompose(
    const abce_copy_orchestrator_t* orchestrator, const abce_copy_op_t* copies,
    uint32_t num_copies, const abce_copy_metadata_t* metadata, uint64_t candidates,
    bool force_back_to_back, uint32_t fixed_engine, abce_engine_op_t* ops, uint32_t* num_ops,
    uint32_t capacity, abce_status_t* out_status, uint32_t* out_failed_op) {
  const uint32_t num_candidates = abce_popcount_engines(candidates);
  const bool can_fanout =
      !force_back_to_back && (num_candidates > 1) && orchestrator->caps.device_atomic_support;
  const uint32_t max_participants =
      force_back_to_back
          ? 1
          : (metadata->max_engines == 0
                 ? (can_fanout ? ABCE_MAX_ENGINES : 1)
                 : (metadata->max_engines < ABCE_MAX_ENGINES ? metadata->max_engines
                                                             : ABCE_MAX_ENGINES));
  const uint32_t num_group_engines =
      max_participants < num_candidates ? max_participants : num_candidates;
  bool use_large_copy_grouping = orchestrator->is_gfx125plus && can_fanout;
  for (uint32_t copy_idx = 0; copy_idx < num_copies && use_large_copy_grouping; ++copy_idx) {
    use_large_copy_grouping = copies[copy_idx].kind == ABCE_OP_KIND_LINEAR &&
                              copies[copy_idx].size >= ABCE_LARGE_COPY_MIN_SIZE;
  }
  uint32_t round_robin = 0;
  uint64_t selected_mask = 0;
  // Per-ring copy count that drives the fan-out balancer (see
  // abce_pick_balanced_engine).
  uint32_t engine_load[ABCE_MAX_ENGINES] = {0};

  for (uint32_t op_idx = 0; op_idx < num_copies; ++op_idx) {
    const abce_copy_op_t* copy = &copies[op_idx];
    *out_failed_op = op_idx;

    if (copy->kind == ABCE_OP_KIND_MULTICAST) {
      if (!abce_copy_orchestrator_decompose_multi_dst(
              orchestrator, copy, metadata, candidates, can_fanout, max_participants,
              force_back_to_back, fixed_engine, &round_robin, &selected_mask, ops, num_ops,
              capacity)) {
        *out_status = *num_ops >= capacity ? ABCE_STATUS_TOO_MANY_OPERATIONS
                                           : ABCE_STATUS_NO_LEGAL_ENGINE;
        return false;
      }
      continue;
    }

    abce_engine_op_t engine_op;
    abce_engine_op_initialize(&engine_op);
    engine_op.kind = copy->kind;
    engine_op.src = copy->src;
    engine_op.dst = copy->dst;
    engine_op.dsts = copy->dsts;
    engine_op.num_dsts = copy->num_dsts;
    engine_op.fill_value = copy->fill_value;
    engine_op.size = copy->size;
    engine_op.size2 = copy->size2;
    engine_op.indirect_src = copy->indirect_src;
    engine_op.indirect_dst = copy->indirect_dst;
    engine_op.rect = copy->rect;
    if (force_back_to_back) {
      abce_copy_orchestrator_set_fixed_engine(orchestrator, &engine_op, copy, candidates,
                                              fixed_engine);
    } else if (use_large_copy_grouping) {
      const uint32_t engine_slot = (op_idx / ABCE_MAX_COPIES_PER_ENGINE) % num_group_engines;
      abce_copy_orchestrator_set_fixed_engine(orchestrator, &engine_op, copy, candidates,
                                              abce_engine_at(candidates, engine_slot));
    } else if (can_fanout) {
      // Parallel-first placement: least-loaded legal ring, biased to the
      // recommended (affinity map) ring so a copy stays on it until the ring is
      // as busy as its alternatives.
      bool has_engine_preference = false;
      uint32_t recommended = ABCE_MAX_ENGINES;
      const uint64_t legal_mask = abce_copy_orchestrator_legal_engine_set(
          orchestrator, copy, candidates, &recommended, engine_op.ranked_engines,
          &engine_op.num_ranked_engines, &has_engine_preference);
      engine_op.has_engine_preference = has_engine_preference;
      const bool can_add_participant =
          abce_popcount_engines(selected_mask) < max_participants;
      const uint64_t unselected_legal = legal_mask & ~selected_mask;
      const uint64_t selected_legal = legal_mask & selected_mask;
      const uint64_t allowed_mask =
          (can_add_participant && unselected_legal != 0) ? unselected_legal : selected_legal;
      engine_op.engine = abce_pick_balanced_engine(allowed_mask, recommended, engine_load);
      if (engine_op.engine < ABCE_MAX_ENGINES) {
        selected_mask |= ((uint64_t)1) << engine_op.engine;
        ++engine_load[engine_op.engine];
      }
    } else {
      bool has_engine_preference = false;
      engine_op.engine = abce_copy_orchestrator_select_engine(
          orchestrator, copy, candidates, can_fanout, max_participants, &round_robin,
          &selected_mask, engine_op.ranked_engines, &engine_op.num_ranked_engines,
          &has_engine_preference);
      engine_op.has_engine_preference = has_engine_preference;
    }
    if (engine_op.engine == ABCE_MAX_ENGINES) {
      *out_status = ABCE_STATUS_NO_LEGAL_ENGINE;
      return false;
    }
    if (!abce_push_engine_op(ops, num_ops, capacity, &engine_op)) {
      *out_status = ABCE_STATUS_TOO_MANY_OPERATIONS;
      return false;
    }
  }
  *out_status = ABCE_STATUS_OK;
  *out_failed_op = UINT32_MAX;
  return true;
}

//===----------------------------------------------------------------------===//
// Frame assembly
//===----------------------------------------------------------------------===//

static inline void abce_initialize_frame_choices(abce_plan_frame_t* frame,
                                                 const abce_engine_op_t* op) {
  frame->num_ranked_engines = op->num_ranked_engines;
  frame->has_engine_preference = op->has_engine_preference;
  for (uint32_t choice_idx = 0; choice_idx < op->num_ranked_engines; ++choice_idx)
    frame->ranked_engines[choice_idx] = op->ranked_engines[choice_idx];
}

static inline void abce_intersect_frame_choices(abce_plan_frame_t* frame,
                                                const abce_engine_op_t* op) {
  frame->has_engine_preference = frame->has_engine_preference && op->has_engine_preference;
  uint32_t common[ABCE_MAX_ENGINE_CHOICES] = {0};
  uint8_t num_common = 0;
  for (uint32_t frame_choice_idx = 0;
       frame_choice_idx < frame->num_ranked_engines && num_common < ABCE_MAX_ENGINE_CHOICES;
       ++frame_choice_idx) {
    const uint32_t candidate = frame->ranked_engines[frame_choice_idx];
    for (uint32_t op_choice_idx = 0; op_choice_idx < op->num_ranked_engines; ++op_choice_idx) {
      if (candidate == op->ranked_engines[op_choice_idx]) {
        common[num_common++] = candidate;
        break;
      }
    }
  }
  frame->num_ranked_engines = num_common;
  for (uint32_t choice_idx = 0; choice_idx < num_common; ++choice_idx)
    frame->ranked_engines[choice_idx] = common[choice_idx];
}

static inline void abce_build_frames(const abce_engine_op_t* ops, uint32_t num_ops,
                                     abce_plan_t* plan) {
  uint64_t seen = 0;
  uint8_t engine_to_frame[ABCE_MAX_ENGINES] = {0};
  plan->num_frames = 0;
  for (uint32_t op_idx = 0; op_idx < num_ops; ++op_idx) {
    const uint32_t engine = ops[op_idx].engine;
    if (!(seen & (1ull << engine))) {
      seen |= (1ull << engine);
      abce_plan_frame_t* frame = &plan->frames[plan->num_frames];
      memset(frame, 0, sizeof(*frame));
      frame->engine = engine;
      abce_initialize_frame_choices(frame, &ops[op_idx]);
      frame->coordinator = (plan->num_frames == 0);
      engine_to_frame[engine] = (uint8_t)plan->num_frames;
      plan->num_frames++;
    } else {
      abce_intersect_frame_choices(&plan->frames[engine_to_frame[engine]], &ops[op_idx]);
    }
    plan->frames[engine_to_frame[engine]].num_ops++;
  }
}

// Bundles the current mapping state for one frame into the descriptor the frame
// composer sizes / emits from.
static inline abce_frame_job_t abce_make_frame_job(const abce_engine_op_t* ops, uint32_t num_ops,
                                                   const abce_copy_metadata_t* metadata,
                                                   const abce_plan_t* plan,
                                                   const abce_plan_frame_t* frame) {
  abce_frame_job_t job;
  job.ops = ops;
  job.num_ops = num_ops;
  job.metadata = metadata;
  job.engine = frame->engine;
  job.frame_num_ops = frame->num_ops;
  job.coordinator = frame->coordinator;
  job.multi = plan->multi;
  job.start_gate_required = plan->start_gate_required;
  job.epilogue_required = plan->epilogue_required;
  job.coordination_word = plan->coordination_signal;
  return job;
}

//===----------------------------------------------------------------------===//
// Phase 1: map
//===----------------------------------------------------------------------===//

// Decomposes |copies| into |out_plan| with pre-built packets. On failure the
// plan is left invalid and out_plan->failed_op names the offending operation
// where one is identifiable. The plan's packet views live in per-thread scratch
// and are valid only until the same thread maps another plan.
static inline abce_status_t abce_copy_orchestrator_map_copy(
    abce_copy_orchestrator_t* orchestrator, const abce_copy_op_t* copies, uint32_t num_copies,
    const abce_copy_metadata_t* metadata, abce_plan_t* out_plan) {
  abce_plan_initialize(out_plan);

  abce_status_t status = abce_copy_orchestrator_validate_request(
      orchestrator, copies, num_copies, metadata, &out_plan->failed_op);
  if (!abce_status_is_ok(status)) return status;

  const uint64_t candidates = abce_copy_orchestrator_candidates(orchestrator, metadata);
  if (!candidates) return ABCE_STATUS_NO_ENGINE;
  out_plan->execution_descriptor = metadata->execution_descriptor;

  uint32_t capacity = 0;
  for (uint32_t copy_idx = 0; copy_idx < num_copies; ++copy_idx) {
    const uint32_t expansion =
        copies[copy_idx].kind == ABCE_OP_KIND_MULTICAST ? copies[copy_idx].num_dsts : 1;
    if (expansion > ABCE_MAX_BATCH_ENTRIES - capacity) {
      out_plan->failed_op = copy_idx;
      return ABCE_STATUS_TOO_MANY_OPERATIONS;
    }
    capacity += expansion;
    out_plan->transfer_kinds |= abce_transfer_kind_bit(
        abce_classify_transfer(&copies[copy_idx].src_end, &copies[copy_idx].dst_end));
  }

  abce_engine_op_t* local_ops = (abce_engine_op_t*)calloc(capacity, sizeof(abce_engine_op_t));
  if (!local_ops) return ABCE_STATUS_RESOURCE_EXHAUSTED;

  uint32_t num_ops = 0;
  const bool force_back_to_back =
      abce_copy_orchestrator_is_back_to_back_batch(orchestrator, copies, num_copies, metadata);
  const uint32_t fixed_engine =
      force_back_to_back
          ? abce_copy_orchestrator_back_to_back_engine(orchestrator, &copies[0], candidates)
          : ABCE_MAX_ENGINES;
  if (force_back_to_back && fixed_engine == ABCE_MAX_ENGINES) {
    free(local_ops);
    out_plan->failed_op = 0;
    return ABCE_STATUS_NO_LEGAL_ENGINE;
  }

  if (!abce_copy_orchestrator_decompose(orchestrator, copies, num_copies, metadata, candidates,
                                        force_back_to_back, fixed_engine, local_ops, &num_ops,
                                        capacity, &status, &out_plan->failed_op)) {
    free(local_ops);
    return status;
  }
  if (num_ops == 0) {
    free(local_ops);
    return ABCE_STATUS_INVALID_ARGUMENT;
  }

  abce_build_frames(local_ops, num_ops, out_plan);
  out_plan->multi = (out_plan->num_frames > 1);

  // Mark ops that use the fused wait/signal packet. gfx125+ can fuse direct
  // completion into a single-frame batch as well as fan-out synchronization into
  // a multi-frame batch.
  const bool use_fused_packets = metadata->prefer_fused && orchestrator->is_gfx125plus;
  bool all_bodies_fused = true;
  for (uint32_t op_idx = 0; op_idx < num_ops; ++op_idx) {
    const abce_op_kind_t kind = local_ops[op_idx].kind;
    if (kind == ABCE_OP_KIND_INDIRECT) {
      local_ops[op_idx].fused = true;
    } else if (use_fused_packets &&
               (kind == ABCE_OP_KIND_LINEAR || kind == ABCE_OP_KIND_MULTICAST ||
                kind == ABCE_OP_KIND_SWAP)) {
      local_ops[op_idx].fused = true;
    }
    all_bodies_fused = all_bodies_fused && local_ops[op_idx].fused;
  }

  // Match ROCr's gfx125+ dependency-free fan-out path: when every body is a
  // fused wait/signal packet and the coordinator has no ordering work, all
  // frames can start directly. Their final packets signal the fan-in count,
  // including the coordinator's. Other plans retain the bit-62 start gate.
  const bool effective_hdp_flush =
      orchestrator->caps.emit_hdp_flush && metadata->coherency.flags.emit_hdp_flush;
  const bool effective_gcr = metadata->coherency.flags.emit_gcr &&
                             abce_builder_requires_gcr(orchestrator->builder) &&
                             !orchestrator->caps.driver_manages_gcr;
  const bool coordinator_has_ordering_work =
      metadata->num_deps != 0 || abce_timestamp_surface_enabled(&metadata->timestamps) ||
      effective_hdp_flush || effective_gcr;
  out_plan->start_gate_required =
      out_plan->multi && (!all_bodies_fused || coordinator_has_ordering_work);
  // ROCr's gfx125+ fast path also omits the fan-in poll and final completion
  // update when no post-copy ordering or interrupt work remains. The last fused
  // body SIGNAL then transitions the output directly to its done value.
  out_plan->epilogue_required = !all_bodies_fused || out_plan->start_gate_required ||
                                abce_timestamp_surface_enabled(&metadata->timestamps) ||
                                effective_gcr || metadata->out.event_mailbox != NULL;
  out_plan->fan_in_count =
      out_plan->multi ? out_plan->num_frames - (out_plan->start_gate_required ? 1u : 0u) : 0;

  // Where the gate and fan-in counter live. When the coordinator drains fan-in
  // and then writes completion itself, they belong in ABCE's own scratch word:
  // keeping them out of the signal's value means the value only ever holds legal
  // signal states, and lets the counter drain to a plain zero instead of a value
  // biased by the caller's completion value. A fan-out with no epilogue has
  // nothing to drain — its body signals are themselves the completion
  // transitions — so it counts down the signal value directly, as does a
  // single-frame direct fused completion.
  const bool coordinator_drains_fan_in = out_plan->multi && out_plan->epilogue_required;
  if (coordinator_drains_fan_in) {
    if (!metadata->out.coordination_scratch) {
      free(local_ops);
      return ABCE_STATUS_MISSING_COORDINATION_SCRATCH;
    }
    // The gfx125+ 64-bit poll encodes only addr[63:3], so a misaligned
    // coordination word would be silently polled at the wrong address.
    ABCE_ASSERT(((uintptr_t)metadata->out.coordination_scratch % 8) == 0 &&
                "coordination_scratch must be 8-byte aligned");
    out_plan->coordination_signal = metadata->out.coordination_scratch;
    out_plan->coordination_initial_value =
        (out_plan->start_gate_required ? ABCE_FAN_OUT_START_GATE : 0) | out_plan->fan_in_count;
  } else {
    out_plan->coordination_signal = metadata->out.value;
    if (out_plan->multi) {
      if (metadata->out.completion_value > UINT32_MAX ||
          out_plan->fan_in_count > UINT32_MAX - metadata->out.completion_value) {
        free(local_ops);
        return ABCE_STATUS_TOO_MANY_OPERATIONS;
      }
      out_plan->coordination_initial_value =
          out_plan->fan_in_count + metadata->out.completion_value;
    }
  }

  // Size each frame.
  for (uint32_t frame_idx = 0; frame_idx < out_plan->num_frames; ++frame_idx) {
    const abce_frame_job_t job = abce_make_frame_job(local_ops, num_ops, metadata, out_plan,
                                                     &out_plan->frames[frame_idx]);
    const size_t frame_bytes = abce_frame_composer_frame_bytes(&orchestrator->composer, &job);
    if (frame_bytes > UINT32_MAX) {
      free(local_ops);
      return ABCE_STATUS_TOO_MANY_OPERATIONS;
    }
    out_plan->frames[frame_idx].bytes = (uint32_t)frame_bytes;
  }

  // Build packets into the per-thread scratch buffer. Frames are laid out
  // contiguously and each frame gets a non-owning view. The zeroed-buffer
  // contract mirrors the ring's acquire memset.
  size_t total_bytes = 0;
  for (uint32_t frame_idx = 0; frame_idx < out_plan->num_frames; ++frame_idx)
    total_bytes += out_plan->frames[frame_idx].bytes;

  char* scratch = abce_acquire_packet_scratch(total_bytes);
  if (!scratch && total_bytes > 0) {
    free(local_ops);
    return ABCE_STATUS_RESOURCE_EXHAUSTED;
  }
  size_t scratch_offset = 0;
  for (uint32_t frame_idx = 0; frame_idx < out_plan->num_frames; ++frame_idx) {
    abce_plan_frame_t* frame = &out_plan->frames[frame_idx];
    char* frame_buffer = scratch + scratch_offset;
    memset(frame_buffer, 0, frame->bytes);
    const abce_frame_job_t job =
        abce_make_frame_job(local_ops, num_ops, metadata, out_plan, frame);
    const size_t emitted =
        abce_frame_composer_emit_frame(&orchestrator->composer, frame_buffer, &job);
    (void)emitted;
    assert(emitted == frame->bytes && "ABCE emit/size divergence");
    frame->packets = frame_buffer;
    scratch_offset += frame->bytes;
  }

  free(local_ops);
  out_plan->valid = true;
  return ABCE_STATUS_OK;
}

//===----------------------------------------------------------------------===//
// Phase 2: submit
//===----------------------------------------------------------------------===//

typedef struct abce_submit_pending_t {
  abce_ring_t* ring;
  abce_ring_reservation_t reservation;
  uint32_t reserve_bytes;
  char* buffer;
} abce_submit_pending_t;

// Reserves every ring, arms the output signal for fan-out, copies the frames
// into their reservations, and publishes them.
static inline abce_status_t abce_copy_orchestrator_submit(abce_copy_orchestrator_t* orchestrator,
                                                          abce_plan_t* plan) {
  if (!plan->valid || plan->submitted) return ABCE_STATUS_INVALID_PLAN;

  abce_submit_pending_t pending[ABCE_MAX_ENGINES];
  memset(pending, 0, sizeof(pending));
  uint8_t acquired_frames[ABCE_MAX_ENGINES] = {0};
  uint32_t num_pending = 0;

  // Validate every frame before reserving any ring so deterministic failures
  // cannot leave an unpublished reservation behind.
  if (plan->num_frames == 0 || plan->num_frames > ABCE_MAX_ENGINES)
    return ABCE_STATUS_INVALID_PLAN;
  uint64_t selected_ring_mask = 0;
  for (uint32_t frame_idx = 0; frame_idx < plan->num_frames; ++frame_idx) {
    const abce_plan_frame_t* frame = &plan->frames[frame_idx];
    if (frame->engine >= ABCE_MAX_ENGINES || frame->packets == NULL || frame->bytes == 0 ||
        (selected_ring_mask & (((uint64_t)1) << frame->engine))) {
      return ABCE_STATUS_INVALID_PLAN;
    }
    selected_ring_mask |= ((uint64_t)1) << frame->engine;
    abce_ring_t* ring = orchestrator->engines[frame->engine];
    if (!ring) return ABCE_STATUS_RING_UNAVAILABLE;
    abce_submit_pending_t* entry = &pending[frame_idx];
    entry->ring = ring;
    if (!abce_status_is_ok(
            abce_ring_padded_size(entry->ring, frame->bytes, &entry->reserve_bytes)))
      return ABCE_STATUS_OUT_OF_RANGE;
  }

  // All submitters acquire rings in the same global order. Without this, plans
  // whose coordinator/first-op choices produce opposite frame orders can each
  // hold one unpublished reservation while waiting forever for the other's ring.
  for (uint32_t engine_idx = 0; engine_idx < ABCE_MAX_ENGINES; ++engine_idx) {
    for (uint32_t frame_idx = 0; frame_idx < plan->num_frames; ++frame_idx) {
      if (plan->frames[frame_idx].engine != engine_idx) continue;
      abce_submit_pending_t* entry = &pending[frame_idx];
      const abce_status_t acquire_status =
          abce_ring_acquire(entry->ring, entry->reserve_bytes, &entry->reservation,
                            &entry->buffer);
      if (!abce_status_is_ok(acquire_status)) {
        for (uint32_t acquired_idx = 0; acquired_idx < num_pending; ++acquired_idx) {
          abce_submit_pending_t* acquired_entry = &pending[acquired_frames[acquired_idx]];
          abce_ring_cancel(acquired_entry->ring, &acquired_entry->reservation);
        }
        return ABCE_STATUS_RING_UNAVAILABLE;
      }
      acquired_frames[num_pending++] = (uint8_t)frame_idx;
    }
  }

  // Arm the coordination word only after every reservation succeeds. No frame is
  // visible to hardware yet, so a deterministic/acquire failure cannot strand a
  // gate or a fan-in count that nothing will ever clear.
  if (plan->multi) {
    __atomic_store_n((uint64_t*)plan->coordination_signal, plan->coordination_initial_value,
                     __ATOMIC_RELEASE);
  }

  // Stamp the execution descriptor here, not in the map phase: a client may
  // retarget a frame's engine after mapping, so the engines are only final now.
  // The ring publish below is what orders this store ahead of the copy
  // completing.
  if (plan->execution_descriptor) {
    __atomic_store_n((uint32_t*)plan->execution_descriptor,
                     abce_copy_orchestrator_describe_execution(orchestrator, plan),
                     __ATOMIC_RELEASE);
  }

  for (uint32_t pending_idx = 0; pending_idx < num_pending; ++pending_idx) {
    const abce_plan_frame_t* frame = &plan->frames[pending_idx];
    memcpy(pending[pending_idx].buffer, frame->packets, frame->bytes);
  }

  // Publish in the same global ring order used for acquisition. Publishing in
  // frame order can create a cross-ring commit cycle between concurrent plans
  // whose coordinator choices produce opposite frame orders.
  for (uint32_t acquired_idx = 0; acquired_idx < num_pending; ++acquired_idx) {
    abce_submit_pending_t* acquired_entry = &pending[acquired_frames[acquired_idx]];
    if (!abce_status_is_ok(abce_ring_release(acquired_entry->ring, &acquired_entry->reservation)))
      return ABCE_STATUS_RING_UNAVAILABLE;
  }

  plan->submitted = true;
  return ABCE_STATUS_OK;
}

// Convenience: map and submit.
static inline abce_status_t abce_copy_orchestrator_dispatch(
    abce_copy_orchestrator_t* orchestrator, const abce_copy_op_t* copies, uint32_t num_copies,
    const abce_copy_metadata_t* metadata) {
  abce_plan_t plan;
  const abce_status_t status =
      abce_copy_orchestrator_map_copy(orchestrator, copies, num_copies, metadata, &plan);
  if (!abce_status_is_ok(status)) return status;
  return abce_copy_orchestrator_submit(orchestrator, &plan);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ABCE_HOST_H_
