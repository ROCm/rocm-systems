/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Accelerated Blit Copy Engine (ABCE) — device-initiated copy submission.
//
// A GPU kernel hands this header one source, one destination and a pre-armed
// completion signal. It forms the SDMA packets into a device ring
// (abce_ring_device.h) and rings that ring's doorbell, so a copy can be launched
// from a shader with no host round-trip.
//
// SCOPE: one linear copy per call, and deliberately nothing else. No batching,
// no multi-destination fan-out, no dependency gating, no interrupt. One copy is
// one frame on one ring, which is what keeps the submitter a short scalar
// routine with no scratch and no LDS, and what removes every reason for a
// coordinator: there is only ever one frame, so the caller's arming is exact and
// the signal retires when that frame does. Anything wider is the host planner's
// job — abce_host.h decomposes a batch across rings and coordinates the frames.
//
// What it still decides is which ring. DeviceEngineSet::engine_table biases the
// choice for the agent pair the caller declares, so a peer copy lands on a ring
// that peer actually prefers. The host fills that table only where engines are
// asymmetric (gfx94x/95x) and leaves it empty otherwise, which degrades to
// taking the lowest permitted ring — the right answer when rings are
// interchangeable. A caller placing copies itself may name the ring instead.
//
// CONCURRENCY: the ring protocol is multi-producer, so several submitters —
// including the host CopyOrchestrator sharing the same ring — may run
// concurrently. Every participant must reference the same SharedRingControl.
// One thread calls DeviceSubmitLinear per submission; it is a scalar producer,
// not a wave-cooperative routine.

#ifndef ABCE_DEVICE_H_
#define ABCE_DEVICE_H_

#include <cstdint>

#include "abce_builder.h"       // ABCE packet builders (ABCE_HD)
#include "abce_engine_table.h"  // agent ids + EngineTable (shared with the host path)
#include "abce_ring_device.h"   // DeviceRing (device-side SDMA ring)
#include "sdma_packets.h"

namespace abce {

/// Bumped whenever DeviceEngineSet's layout or the submitter's contract
/// changes, so a kernel built against an older header refuses a newer runtime's
/// view instead of misreading it.
constexpr uint32_t kDeviceAbiVersion = 1;

/// One ring per registered SDMA engine; matches abce::kMaxEngines.
constexpr uint32_t kDeviceMaxEngines = kMaxEngineChoices;

/// Client device ids the peer map covers. A client names devices in its own
/// numbering (HIP device ordinals, say), which is NOT the agent numbering the
/// engine table is indexed by — see DeviceEngineSet::agent_for_device.
constexpr uint32_t kDeviceMaxPeers = 72;

/// Let the submitter pick the ring. Distinct from every valid engine index, so
/// "no opinion" and "ring 0" stay tellable apart.
constexpr uint32_t kDeviceAutoEngine = 0xFFFFFFFFu;

/// Every outcome this header reports, submission and polling alike. One enum so
/// a status never has to be translated on its way out; kNotReady and kTimeout
/// are the two a poll adds, and neither is an error.
enum class DeviceStatus : uint32_t {
  kSuccess,
  kInvalidHandle,    ///< null/mismatched-ABI engine set.
  kInvalidArgument,  ///< bad pointer, size, signal address or reference value.
  kNoEngine,         ///< no ring a kernel may submit to, or the named one is not one of them.
  kTooLarge,         ///< the frame cannot fit in its ring.
  kUnsupported,      ///< no SDMA completion mechanism on this device.
  kNotReady,         ///< submitted, not yet retired.
  kTimeout,          ///< a bounded wait gave up; the copy is still outstanding.
};

/// Read-only from the device: the host publishes every field before the view
/// reaches a kernel, and the submitter only ever writes through DeviceRing's
/// pointers. That lets the host place this in ordinary cacheable memory and
/// keep just the SharedRingControl blocks — which host and device CAS
/// concurrently — in an uncached fine-grained pool.
struct alignas(64) DeviceEngineSet {
  uint32_t abi_version = kDeviceAbiVersion;
  uint32_t num_engines = 0;
  /// Whether SDMA can retire the completion signal with an atomic packet.
  bool device_atomic_support = false;

  /// Rings a KERNEL may submit to (bit i => ring i). Not every registered ring:
  /// on some parts a shader store to a doorbell is silently dropped, leaving the
  /// engine idle with work published, and the host cannot call those rings
  /// invalid because its own submissions work. A bit also asserts @c engines[i]
  /// is complete, so the submitter treats this as the whole answer.
  uint64_t device_usable_engines = 0;

  /// Rings no host producer submits to (bit i => ring i), always a subset of
  /// @c device_usable_engines. Because a kernel is the only producer, their
  /// SharedRingControl can sit in device memory instead of the uncached host
  /// memory a shared cursor needs, which is worth ~2.4 us of a submit —
  /// so automatic ring choice prefers them. Zero is the normal state.
  uint64_t device_owned_engines = 0;

  ABCE builder;
  DeviceRing engines[kDeviceMaxEngines];

  /// Ring preferences per agent pair — the same EngineTable the host forms and
  /// selects with, copied in wholesale, so both paths answer "which engine for
  /// this pair" from one source. Empty cells mean no opinion, which is the
  /// correct permanent content on parts whose engines are interchangeable, and
  /// preference only biases the choice, so a stale cell costs bandwidth and
  /// never correctness.
  EngineTable engine_table{};

  /// Client device id -> agent id, because the two are different numberings and
  /// only the host knows the correspondence. kAgentUnknown means "not
  /// published", which costs affinity and never correctness.
  int8_t agent_for_device[kDeviceMaxPeers];

  /// Agent this engine set's rings belong to. A kernel cannot name the device it
  /// is running on, so without this it could not tag the local end of its own
  /// H2D or D2H copy.
  int8_t local_agent = kAgentUnknown;

  /// The builder has no device-callable constructor, so the host builds it once
  /// and this trivially copyable snapshot is what kernels read.
  explicit DeviceEngineSet(const ABCE& packet_builder) : builder(packet_builder) {
    for (uint32_t device_id = 0; device_id < kDeviceMaxPeers; ++device_id)
      agent_for_device[device_id] = kAgentUnknown;
  }
};

/// Opaque-by-convention handle a kernel receives as a launch argument.
struct DeviceEngineSetHandle {
  DeviceEngineSet* engine_set = nullptr;
  uint32_t abi_version = kDeviceAbiVersion;
};

ABCE_HD inline bool DeviceEngineSetValid(const DeviceEngineSetHandle& handle) {
  return handle.engine_set != nullptr && handle.abi_version == kDeviceAbiVersion &&
         handle.engine_set->abi_version == kDeviceAbiVersion;
}

/// @brief Agent id for a client device id, for the agent arguments of a submit.
///
/// Returns kAgentUnknown for an out-of-range or unpublished id, which places the
/// copy by ring availability alone rather than failing it.
ABCE_HD inline int8_t DeviceAgentFor(const DeviceEngineSetHandle& handle, uint32_t device_id) {
  if (!DeviceEngineSetValid(handle) || device_id >= kDeviceMaxPeers) return kAgentUnknown;
  return handle.engine_set->agent_for_device[device_id];
}

/// @brief Agent id of the device these rings belong to, for tagging the local
/// end of a copy without knowing which device id names it.
ABCE_HD inline int8_t DeviceLocalAgent(const DeviceEngineSetHandle& handle) {
  return DeviceEngineSetValid(handle) ? handle.engine_set->local_agent : kAgentUnknown;
}

/// What a caller polls to learn a copy has retired: the completion signal and
/// the value it lands on, so a query costs one load and needs nothing the
/// caller did not already own.
struct DeviceSubmissionToken {
  uint64_t* signal = nullptr;
  uint64_t reference = 0;
};

// Ring choice. Host-callable on purpose: this is the only decision the submitter
// makes, so it is also the only part worth testing without a GPU queue. Nothing
// here touches a ring, a doorbell or the completion signal.
namespace device_detail {

/// What the engine table says about one agent pair: an affinity anchor and any
/// rings that must not serve it.
struct AgentPreference {
  uint32_t preferred = kDeviceMaxEngines;
  uint64_t forbidden = 0;
};

ABCE_HD inline AgentPreference PreferenceFor(const DeviceEngineSet& engine_set, int8_t src_agent,
                                             int8_t dst_agent) {
  const EngineOrder& order = engine_set.engine_table.Cell(src_agent, dst_agent);
  AgentPreference preference;
  preference.preferred = order.num_ranked ? order.ranked[0] : kDeviceMaxEngines;
  preference.forbidden = order.forbidden_mask;
  return preference;
}

/// @brief Ring for one copy out of @p candidates, or kDeviceMaxEngines if none.
///
/// Affinity first: the pair's preferred ring when it named one and that ring is
/// permitted. Otherwise a device-owned ring, whose cursors live in device memory
/// and so make a submit materially cheaper. Otherwise the lowest permitted ring.
/// The forbidden mask is a hard rule and is never widened past; everything else
/// is a preference, so an empty table or a build that reserved no ring simply
/// falls through to the lowest permitted ring.
ABCE_HD inline uint32_t ChooseEngine(const DeviceEngineSet& engine_set, int8_t src_agent,
                                     int8_t dst_agent, uint64_t candidates) {
  const AgentPreference preference = PreferenceFor(engine_set, src_agent, dst_agent);
  const uint64_t allowed = candidates & ~preference.forbidden;
  if (allowed == 0) return kDeviceMaxEngines;
  if (preference.preferred < kDeviceMaxEngines &&
      (allowed & (uint64_t{1} << preference.preferred)) != 0)
    return preference.preferred;
  const uint64_t owned = allowed & engine_set.device_owned_engines;
  return static_cast<uint32_t>(detail::CountTrailingZeros64(owned != 0 ? owned : allowed));
}

/// @brief Resolve the caller's ring request, or kDeviceMaxEngines when none serves.
///
/// A named ring is honoured or refused, never silently substituted: a caller
/// that named one is placing the copy on purpose — spreading a set of copies
/// over rings itself, say — and would rather hear that it cannot than have the
/// work land somewhere else.
ABCE_HD inline uint32_t ResolveEngine(const DeviceEngineSet& engine_set, uint32_t engine,
                                      int8_t src_agent, int8_t dst_agent) {
  const uint64_t candidates =
      engine_set.device_usable_engines & ((uint64_t{1} << kDeviceMaxEngines) - 1);
  if (engine == kDeviceAutoEngine)
    return ChooseEngine(engine_set, src_agent, dst_agent, candidates);
  return engine < kDeviceMaxEngines && (candidates & (uint64_t{1} << engine)) != 0
             ? engine
             : kDeviceMaxEngines;
}

}  // namespace device_detail

#if defined(__HIPCC__)

namespace device_detail {

/// @brief Emit one linear copy that retires @p completion_signal onto @p engine.
///
/// Two packet forms, one decrement either way, so arming is identical: gfx125+
/// emits COPY_LINEAR_WAITSIGNAL, which carries its own completion; everything
/// else emits COPY_LINEAR plus an atomic decrement. The packet count follows
/// from the size — BuildCopyCommand chunks against the builder's per-packet
/// maximum — so a large copy becomes several back-to-back packets in the same
/// frame rather than one malformed packet, and callers never reason about
/// packet counts.
__device__ inline DeviceStatus EmitLinearFrame(DeviceEngineSet& engine_set, uint32_t engine,
                                               const void* src, void* dst, uint64_t size,
                                               void* completion_signal) {
  const ABCE builder = engine_set.builder;
  const uint32_t num_packets = builder.NumCopyPackets(static_cast<size_t>(size));
  const bool fused = builder.IsGfx125plus();
  const uint64_t copy_bytes = uint64_t{num_packets} * sizeof(SDMA_PKT_COPY_LINEAR);
  // Fused packets are compacted — header plus body per chunk, with the single
  // signal block on the last chunk (boundary_wait_signal). Same accounting as
  // the host's FrameBuilder::FusedBytes.
  const uint64_t frame_bytes = fused
      ? (uint64_t{num_packets} * (1u + 6u) + 5u) * sizeof(uint32_t)
      : copy_bytes + sizeof(SDMA_PKT_ATOMIC);

  // Reserve is the single authority on whether a frame fits, so there is no
  // separate size check, and a frame that does not fit is its only failure: the
  // ring is known valid (device_usable_engines asserts it) and packet sizes are
  // dword multiples.
  DeviceRing& ring = engine_set.engines[engine];
  RingReservation reservation;
  if (ring.Reserve(frame_bytes, reservation) != RingStatus::kSuccess)
    return DeviceStatus::kTooLarge;
  // The fixed-layout builders only set meaningful fields (abce_builder.h's
  // zeroed-buffer contract), so the region must start out zero. The fused
  // builders write every dword, so that route skips it.
  if (!fused) ring.Zero(reservation);
  char* const frame = ring.WriteAddress(reservation.start, reservation.bytes());

  if (fused) {
    builder.BuildWaitSignalCopyCommand(frame, dst, src, static_cast<size_t>(size),
                                       /*wait_addr=*/nullptr, completion_signal,
                                       /*wait_reference=*/0, UINT64_MAX,
                                       /*boundary_wait_signal=*/true);
  } else {
    builder.BuildCopyCommand(frame, dst, src, static_cast<size_t>(size));
    builder.BuildAtomicDecrementCommand(frame + copy_bytes, completion_signal);
  }

  ring.Submit(reservation);
  return DeviceStatus::kSuccess;
}

}  // namespace device_detail

/// @brief Submit one linear copy of @p size bytes and hand back a @p token.
///
/// The whole device-side API: form the packets, ring the doorbell, return. The
/// copy is published, not complete — poll @p token to learn that.
///
/// SIGNAL OWNERSHIP. @p completion_signal must be 8-byte aligned and
/// SDMA-visible, and the caller arms it to @p completion_reference + 1. That
/// arming is exact rather than a floor: one copy is one frame, so exactly one
/// decrement ever lands and this never adds to the signal. @p
/// completion_reference must be below UINT32_MAX, because the SDMA atomic
/// carries a 32-bit operand and a wider target could never be reached.
///
/// RING CHOICE. @p src_agent and @p dst_agent are consulted only for affinity;
/// leaving them at kAgentUnknown is always safe and places the copy by ring
/// availability alone, which is what happens anyway on parts whose engines are
/// symmetric. Use DeviceAgentFor() and DeviceLocalAgent() to fill them. @p
/// engine names a ring outright, or kDeviceAutoEngine to let the submitter pick.
///
/// MEMORY VISIBILITY. This call is a SYSTEM-SCOPE RELEASE POINT, and owning that
/// is deliberately the API's job rather than the caller's. Everything the
/// calling thread stored before it — the source this copy reads, and anything it
/// wrote into the destination — is ordered ahead of the doorbell, so SDMA cannot
/// read the source before the stores that filled it land, and an in-flight store
/// to the destination cannot arrive after the copy and overwrite it. Callers
/// therefore do NOT need to release those buffers themselves, and should not: a
/// caller-side release on top of this one pays the cache writeback twice for one
/// guarantee. The release covers the CALLING THREAD; stores made by other
/// threads or waves are covered once the caller has synchronised with them by
/// its own means (a workgroup barrier, say) before calling.
///
/// Blocks while the ring is full and while earlier reservations publish, so it
/// must not be called from a thread the completion of this copy depends on.
__device__ inline DeviceStatus DeviceSubmitLinear(
    const DeviceEngineSetHandle& handle, void* dst, const void* src, uint64_t size,
    uint64_t* completion_signal, uint64_t completion_reference, DeviceSubmissionToken& token,
    int8_t src_agent = kAgentUnknown, int8_t dst_agent = kAgentUnknown,
    uint32_t engine = kDeviceAutoEngine) {
  using namespace device_detail;

  token = {};
  if (!DeviceEngineSetValid(handle)) return DeviceStatus::kInvalidHandle;
  if (src == nullptr || dst == nullptr || size == 0) return DeviceStatus::kInvalidArgument;
  if (completion_signal == nullptr ||
      (reinterpret_cast<uintptr_t>(completion_signal) % sizeof(uint64_t)) != 0)
    return DeviceStatus::kInvalidArgument;
  if (completion_reference >= UINT32_MAX) return DeviceStatus::kInvalidArgument;

  DeviceEngineSet& engine_set = *handle.engine_set;
  if (!engine_set.device_atomic_support) return DeviceStatus::kUnsupported;

  const uint32_t ring_index = ResolveEngine(engine_set, engine, src_agent, dst_agent);
  if (ring_index >= kDeviceMaxEngines) return DeviceStatus::kNoEngine;

  const DeviceStatus status =
      EmitLinearFrame(engine_set, ring_index, src, dst, size, completion_signal);
  if (status != DeviceStatus::kSuccess) return status;

  token.signal = completion_signal;
  token.reference = completion_reference;
  return DeviceStatus::kSuccess;
}

/// Whether @p token's copy has retired. kNotReady is the ordinary answer for a
/// copy still in flight, not an error.
__device__ inline DeviceStatus DeviceTokenQuery(const DeviceSubmissionToken& token) {
  if (token.signal == nullptr) return DeviceStatus::kInvalidArgument;
  const uint64_t value =
      __hip_atomic_load(token.signal, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
  return value == token.reference ? DeviceStatus::kSuccess : DeviceStatus::kNotReady;
}

/// Spin up to @p max_spins times for @p token, then check once more. Bounded so
/// a kernel cannot hang forever on a lost or mis-armed signal; kTimeout leaves
/// the copy outstanding rather than failed.
__device__ inline DeviceStatus DeviceTokenWait(const DeviceSubmissionToken& token,
                                               uint64_t max_spins) {
  for (uint64_t spin = 0; spin < max_spins; ++spin) {
    const DeviceStatus status = DeviceTokenQuery(token);
    if (status != DeviceStatus::kNotReady) return status;
    RingPause();
  }
  const DeviceStatus status = DeviceTokenQuery(token);
  return status == DeviceStatus::kNotReady ? DeviceStatus::kTimeout : status;
}

#endif  // __HIPCC__

}  // namespace abce

#endif  // ABCE_DEVICE_H_
