/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Accelerated Blit Copy Engine (ABCE) — host mapping and submission API.

#ifndef ABCE_HOST_H_
#define ABCE_HOST_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <utility>

#include "abce_builder.h"
#include "abce_frame.h"
#include "abce_ring_host.h"
#include "abce_topology.h"

namespace abce {

/// @brief Where one end of a copy lives.  Lets the engine policy tell H2D/D2H
/// (host<->device) from device<->device, and same-device (local D2D) from
/// cross-device (P2P) transfers, without the orchestrator ever dereferencing a
/// pointer to discover locality.
enum class EndpointKind : uint8_t { kHost, kDevice };

/// @brief "Any / unknown device" sentinel for CopyEndpoint::device_id.
constexpr uint32_t kAnyDevice = UINT32_MAX;

/// @brief One end of a copy: whether it is host or device memory, and (for
/// device memory) which device it lives on. @c device_id is an ABCE-local
/// index the client assigns at topology-load / registration time; it is only
/// meaningful when @c kind == kDevice.
struct CopyEndpoint {
  EndpointKind kind = EndpointKind::kHost;
  uint32_t device_id = kAnyDevice;
};

/// @brief How a transfer crosses the memory topology.
///
/// Derived purely from the endpoint classification the client supplies, so it
/// costs nothing to know and never dereferences a user pointer. Engine selection
/// keys its ranking bands off this; it is also the natural label for a profiler
/// attributing a copy, which is why it is reported on Plan.
enum class TransferKind : uint8_t {
  kUnknown,               ///< host->host: rejected before it reaches selection.
  kHostToDevice,          ///< H2D
  kDeviceToHost,          ///< D2H
  kPeerToPeer,            ///< D2D across devices (or with an unknown end)
  kLocalDeviceToDevice,   ///< D2D within one device
};

/// Classify a transfer from its endpoints. Single source of truth: both engine
/// ranking and the reported Plan label go through here, so they cannot disagree
/// about what kind of copy something is.
inline TransferKind ClassifyTransfer(const CopyEndpoint& src, const CopyEndpoint& dst) {
  if (src.kind == EndpointKind::kDevice && dst.kind == EndpointKind::kDevice) {
    // Local D2D requires both ids to be known: kAnyDevice means "unknown", so
    // two unknown ends are a cross-device copy, not a same-device one, which
    // would steer a real P2P transfer away from the xGMI band.
    const bool same_device = src.device_id != kAnyDevice && src.device_id == dst.device_id;
    return same_device ? TransferKind::kLocalDeviceToDevice : TransferKind::kPeerToPeer;
  }
  if (src.kind == EndpointKind::kHost && dst.kind == EndpointKind::kDevice)
    return TransferKind::kHostToDevice;
  if (src.kind == EndpointKind::kDevice && dst.kind == EndpointKind::kHost)
    return TransferKind::kDeviceToHost;
  return TransferKind::kUnknown;
}

/// Bit for @p kind inside Plan::transfer_kinds.
inline constexpr uint32_t TransferKindBit(TransferKind kind) {
  return uint32_t{1} << static_cast<uint8_t>(kind);
}


/// @brief Maximum number of SDMA engines the orchestrator tracks (also the
/// engine-bitmask width).
constexpr uint32_t kMaxEngines = 16;

/// @brief "No hardware engine id" sentinel for EngineAffinity::hw_engine_id.
constexpr uint32_t kAnyEngine = UINT32_MAX;

/// @brief One logical copy operation from the client.
///
/// TODO(perf, revisit at integration): this is a flat 88-byte AoS struct (2
/// cachelines).  For the dominant linear H2D/D2H batch only kind/src/dst/size/
/// endpoints are live; the per-kind tail (dsts+num_dsts, fill_value, size2, rect,
/// indirect_*) is mutually exclusive and pollutes the line.  When we wire this to
/// the hipMemcpyBatchAsync path, consider a hot core + kind-keyed union to bring
/// the common op to <=64B / one cacheline (mirror in EngineOp).  Measure typical
/// batch sizes first — only worth the ergonomic hit if these batches are hot.
struct CopyOp {
  OpKind kind = OpKind::kLinear;

  const void* src = nullptr;  ///< linear source / swap endpoint B / indirect src.
  void* dst = nullptr;        ///< linear/fill destination / swap endpoint A / indirect dst.

  void* const* dsts = nullptr;  ///< multicast destinations.
  uint32_t num_dsts = 0;        ///< multicast destination count.

  uint32_t fill_value = 0;  ///< fill pattern (kFill).

  size_t size = 0;  ///< transfer size in bytes (fill: byte count, multiple of 4).

  /// Optional secondary size for two-region ops.  For kSwap it is endpoint B's
  /// (@c src) byte count; 0 => same as @c size.  Ignored by other kinds.
  size_t size2 = 0;

  bool indirect_src = false;  ///< kIndirect: `src` points to an address list.
  bool indirect_dst = false;  ///< kIndirect: `dst` points to an address list.

  const CopyRectDesc* rect = nullptr;  ///< kCopyRect geometry (required for kCopyRect).

  CopyEndpoint src_end{};  ///< source residence (host/device + device id) for the policy.
  CopyEndpoint dst_end{};  ///< destination residence for the policy.
};

/// @brief Per-engine metadata supplied at registration.
struct EngineAffinity {
  /// Hardware SDMA engine id (the policy's hw-id space; maps to this registered
  /// index). kAnyEngine uses the registered index as the hardware id.
  ///
  /// There is deliberately no engine-class field. Which engines suit which band
  /// is not a client-declared property: it is either measured (the H2D/D2H
  /// heatmap) or reported by the driver (KFD's non-xGMI/xGMI engine split and
  /// per-link recommended engine mask). On MI300X the two are not even
  /// correlated -- KFD classes engines 2-15 as xGMI, yet the measured H2D
  /// ranking puts four of them ahead of both non-xGMI engines -- so treating a
  /// class as a filter for host copies would discard the fastest engines.
  uint32_t hw_engine_id = kAnyEngine;
};

/// @brief Which hardware block executed a copy. Three bits inside an
/// ExecutionDescriptor, because a completion signal is not the property of any
/// one engine: the same signal object that an SDMA batch completes can instead
/// be completed by a compute blit kernel, or by a host memcpy, and a reader has
/// no way to tell from the signal value.
///
/// kNone is the zero value and means nobody stamped the descriptor, which is
/// what makes a descriptor self-validating.
///
/// ABCE only ever writes kSdma -- it builds nothing else. The other values exist
/// so that a client whose fallback paths complete the *same* signal can stamp
/// the same slot, giving one field a consumer can trust instead of a heuristic.
/// ROCr, for instance, currently distinguishes SDMA from a blit kernel by zeroing
/// its SDMA timestamps before a copy and testing them afterwards
/// (SharedSignal::CopyPrep / GetRawTs); the two paths do not even record
/// timestamps in the same place, so this field also tells a reader which pair to
/// look at.
/// Three bits rather than two: the spare values cost nothing here (the reserved
/// tail absorbs them) and leave room for executors that have not come up yet,
/// which matters for a field written by more than one component.
enum class ExecutionEngineKind : uint8_t {
  kNone = 0,     ///< not stamped.
  kSdma = 1,     ///< SDMA / DMA engine. What ABCE emits.
  kCompute = 2,  ///< compute shader (a blit kernel).
  kCpu = 3,      ///< host processor (a memcpy).
};

/// @brief A 32-bit summary of how a batch actually executed, for whoever reads
/// the completion signal afterwards.
///
/// Timestamps tell a profiler *when* a copy ran; this tells it *what ran and
/// where*, which is the part a tool otherwise has to guess (ROCr, for instance,
/// infers "SDMA or blit kernel?" from whether its SDMA timestamps came back
/// non-zero). Submit writes it, not MapCopy: a client may retarget
/// PlanFrame::engine in between, so only Submit knows the engines really used.
///
/// Engines are reported as a mask of *hardware* engine ids rather than ABCE's
/// registered indices, which are a client-local numbering that means nothing to
/// an external reader. A mask, rather than a single id, is what makes a fan-out
/// representable.
///
/// | bits  | field |
/// |-------|-------|
/// | 0-15  | instance mask, interpreted per engine kind (see below) |
/// | 16-18 | TransferKind |
/// | 19    | batch mixed transfer kinds, so the TransferKind field reads kUnknown |
/// | 20-22 | ExecutionEngineKind |
/// | 23-31 | reserved, zero |
///
/// The instance mask is only meaningful relative to the engine kind, which is why
/// the kind is part of the contract rather than assumed. For kSdma, bit *h* means
/// hardware SDMA engine *h* ran part of the batch. Another kind defines its own
/// instances, or leaves the mask zero when it has none to report -- a blit kernel
/// has no equivalent of an SDMA engine id.
///
/// A descriptor is valid iff its engine kind is not kNone. Validity deliberately
/// does not hang off the instance mask: that would force every writer to invent
/// an instance, and would make a legitimately mask-less stamp unreadable.
///
/// An engine *class* -- xGMI versus PCIe within SDMA -- is deliberately not
/// reported, which is a different question from engine kind. ABCE no longer has
/// one to report (see EngineAffinity), and it would mislead if it did: on MI300X
/// the driver classes engines 2-15 as xGMI while the measured H2D ranking prefers
/// four of them over both non-xGMI engines, so "an xGMI engine ran an H2D copy"
/// is the normal case rather than an anomaly worth a bit. Engine kind is the
/// opposite: it distinguishes hardware blocks with genuinely different behaviour,
/// including where they record their timestamps.
struct ExecutionDescriptor {
  static constexpr uint32_t kInstanceMaskShift = 0;
  static constexpr uint32_t kInstanceMaskMask = 0xFFFFu;
  static constexpr uint32_t kTransferKindShift = 16;
  static constexpr uint32_t kTransferKindMask = 0x7u;
  static constexpr uint32_t kMixedKindsBit = uint32_t{1} << 19;
  static constexpr uint32_t kEngineKindShift = 20;
  static constexpr uint32_t kEngineKindMask = 0x7u;

  static_assert(kMaxEngines <= 16, "instance mask field is 16 bits wide");

  static uint32_t Encode(ExecutionEngineKind engine_kind, uint32_t instance_mask,
                         TransferKind transfer_kind, bool mixed_kinds) {
    return ((instance_mask & kInstanceMaskMask) << kInstanceMaskShift) |
           ((static_cast<uint32_t>(transfer_kind) & kTransferKindMask) << kTransferKindShift) |
           (mixed_kinds ? kMixedKindsBit : 0) |
           ((static_cast<uint32_t>(engine_kind) & kEngineKindMask) << kEngineKindShift);
  }

  static bool Valid(uint32_t word) { return EngineKind(word) != ExecutionEngineKind::kNone; }
  static ExecutionEngineKind EngineKind(uint32_t word) {
    return static_cast<ExecutionEngineKind>((word >> kEngineKindShift) & kEngineKindMask);
  }
  static uint32_t InstanceMask(uint32_t word) {
    return (word >> kInstanceMaskShift) & kInstanceMaskMask;
  }
  static TransferKind Kind(uint32_t word) {
    return static_cast<TransferKind>((word >> kTransferKindShift) & kTransferKindMask);
  }
  static bool MixedKinds(uint32_t word) { return (word & kMixedKindsBit) != 0; }
};

// ===========================================================================
// Engine selection policy
// ===========================================================================

/// @brief Optional engine-selection policy callback.
///
/// Called per-CopyOp during MapCopy to rank engines for a specific transfer.
/// If not supplied (nullptr), the orchestrator uses ascending-index / round-robin.
///
/// Inputs (read-only):
///   @p context        — opaque policy pointer (e.g. topology table).
///   @p src            — source endpoint of this copy (host/device + device_id).
///   @p dst            — destination endpoint of this copy.
///   @p candidate_mask — bitmask of legal engines for this batch (bit i = engine i).
///   @p max            — max entries the caller can accept in @p out.
///
/// Output:
///   @p out[0..return) — up to @p max engine indices, ranked best-first.  Each
///                       must be a bit set in @p candidate_mask.
///
enum class PolicyDisposition : uint8_t {
  kNoPreference,
  kRanked,
  kNoLegalEngine,
};

struct PolicyResult {
  PolicyDisposition disposition = PolicyDisposition::kNoPreference;
  uint32_t count = 0;
};

/// Returns a disposition plus the number of ranked engine indices written.
/// kNoPreference permits ascending/round-robin fallback; kNoLegalEngine is a
/// hard rejection and never falls back.
using EnginePolicyFn = PolicyResult (*)(void* context, const CopyEndpoint& src,
                                        const CopyEndpoint& dst, uint64_t candidate_mask,
                                        uint32_t* out, uint32_t max);

/// @brief Lowest set bit index of @p mask, or kMaxEngines if empty.
inline uint32_t FirstEngine(uint64_t mask) {
  return mask ? static_cast<uint32_t>(detail::CountTrailingZeros64(mask)) : kMaxEngines;
}

/// @brief Measured host-copy (H2D/D2H) engine ordering for a specific part.
///
/// SPX bandwidth per SDMA engine is not uniform: some engines are far better for
/// D2H than others (e.g. on gfx94x/95x).  Each profile lists hardware SDMA engine
/// engine ids best->worst for each direction; the policy resolves those ids to
/// registered indices and returns the top few.
/// Parts are distinguished by (gfx minor version, total SDMA engine count) —

struct HeatmapProfile {
  uint8_t minor;             ///< gfx9 minor version.
  uint8_t total_sdma;        ///< total SDMA engines (host + xGMI).
  uint8_t d2h[kMaxEngines];  ///< D2H hw engine ids, best->worst.
  uint8_t d2h_count;
  uint8_t h2d[kMaxEngines];  ///< H2D hw engine ids, best->worst.
  uint8_t h2d_count;
};

/// @brief Select the heatmap profile for @p minor / @p total_sdma, or nullptr
/// when the part is unrecognized (ranking then degrades to ascending order).
inline const HeatmapProfile* SelectHeatmap(uint8_t minor, uint8_t total_sdma) {
  // clang-format off
  static const HeatmapProfile kProfiles[] = {
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
  for (const auto& profile : kProfiles)
    if (profile.minor == minor && profile.total_sdma == total_sdma) return &profile;
  return nullptr;
}

/// @brief Topology-aware SDMA engine-selection policy.
///
/// Given a (src, dst) CopyEndpoint pair, produces a ranked list of engines.
/// Three bands drive the ranking:
///   - local   (same-device copies)
///   - xGMI    (cross-device P2P copies)
///   - H2D/D2H (host copies)
///
/// Which engines belong to a band is never declared by the client. Host copies
/// are ordered by the measured heatmap, and the xGMI band comes from the
/// driver's own engine split (SetSdmaEngineSplit, filled by
/// LoadTopologyFromKfd). Absent both, every band resolves to all registered
/// engines and selection degrades to pure load balancing, which is the right
/// answer for parts where the engines are equivalent.
///
/// Arch subclasses override per-band hooks; the base provides hw-id-to-register
/// mapping and the RankEngines skeleton.
class SdmaEnginePolicy {
 public:
  SdmaEnginePolicy() {
    for (auto& entry : hwid_to_reg_) entry = -1;
  }
  virtual ~SdmaEnginePolicy() = default;

  /// Associate a hardware SDMA engine id with a registered engine index.
  /// Call once per registered engine.
  void MapEngine(uint32_t hw_engine_id, uint32_t reg_index) {
    if (hw_engine_id >= kMaxEngines || reg_index >= kMaxEngines || hwid_to_reg_[hw_engine_id] >= 0)
      return;
    hwid_to_reg_[hw_engine_id] = static_cast<int>(reg_index);
    if (num_mapped_ < kMaxEngines) mapped_hwids_[num_mapped_++] = static_cast<uint8_t>(hw_engine_id);
  }

  /// Record the driver's SDMA engine split: @p num_non_xgmi engines are not
  /// xGMI, and KFD numbers those first, so hardware ids at or above it form the
  /// xGMI band used for P2P. Zero means "unknown", which widens the P2P band to
  /// every registered engine rather than emptying it.
  ///
  /// This replaces a per-engine class declaration. The distinction is only ever
  /// used to *prefer* engines for P2P, never to exclude them from host copies:
  /// on MI300X the fastest measured H2D engines are xGMI engines.
  void SetSdmaEngineSplit(uint32_t num_non_xgmi) {
    num_non_xgmi_ = num_non_xgmi < kMaxEngines ? num_non_xgmi : 0;
  }

  uint32_t RegisteredEngine(uint32_t hw_engine_id) const {
    return hw_engine_id < kMaxEngines && hwid_to_reg_[hw_engine_id] >= 0
               ? static_cast<uint32_t>(hwid_to_reg_[hw_engine_id])
               : kMaxEngines;
  }

  /// Select the host-copy heatmap profile for this part.
  void SelectHeatmapProfile(uint8_t minor, uint8_t total_sdma) {
    hm_ = SelectHeatmap(minor, total_sdma);
  }

  /// Install caller-supplied topology. The route for platforms with no KFD
  /// sysfs: fill a TopologyData from whatever the OS exposes, then pair this
  /// with SelectHeatmapProfile() or InitDeviceProfile().
  void SetTopology(const TopologyData& topology) { topology_ = topology; }

  /// Load KFD topology into a temporary snapshot and replace active state only
  /// after a GPU is found. Returns false where KFD sysfs does not exist
  /// (ABCE_HAS_KFD_TOPOLOGY == 0), leaving topology untouched.
  bool LoadTopologyFromKfd(const std::string& base_path = "/sys/class/kfd/kfd/topology/nodes") {
#if ABCE_HAS_KFD_TOPOLOGY
    LinuxTopologyProvider provider(base_path);
    TopologyData replacement;
    const TopologyLoadResult result = provider.Populate(replacement);
    if (!result.found_gpu) return false;
    topology_ = replacement;
    SelectHeatmapProfile(result.gfx_minor, result.total_sdma);
    SetSdmaEngineSplit(result.num_non_xgmi_sdma);
    return true;
#else
    (void)base_path;
    return false;
#endif
  }

 protected:
  // Each writes up to `max` hardware SDMA engine ids (best->worst) into
  // `out_hwids` and returns the count.  Omitting an engine from the list is how
  // a hard reservation is enforced (it can never be chosen for that band).

  virtual uint32_t RankH2D(const CopyEndpoint&, const CopyEndpoint&, uint8_t* out,
                           uint32_t max) const {
    return EmitAll(out, max);
  }
  virtual uint32_t RankD2H(const CopyEndpoint&, const CopyEndpoint&, uint8_t* out,
                           uint32_t max) const {
    return EmitAll(out, max);
  }
  virtual uint32_t RankP2P(const CopyEndpoint&, const CopyEndpoint&, uint8_t* out,
                           uint32_t max) const {
    return EmitXgmi(out, max);
  }
  virtual uint32_t RankLocalD2D(const CopyEndpoint&, const CopyEndpoint&, uint8_t* out,
                                uint32_t max) const {
    return EmitAll(out, max);
  }

  /// Append every mapped hw engine id, in registration order, to @p out.
  uint32_t EmitAll(uint8_t* out, uint32_t max) const {
    const uint32_t n = num_mapped_ < max ? num_mapped_ : max;
    for (uint32_t i = 0; i < n; ++i) out[i] = mapped_hwids_[i];
    return n;
  }

  /// Append the mapped hw engine ids in the driver's xGMI band. Falls back to
  /// every engine when the split is unknown or when the band came out empty
  /// (a client may have registered only non-xGMI engines), because offering a
  /// slower engine beats reporting that a P2P copy cannot be mapped at all.
  uint32_t EmitXgmi(uint8_t* out, uint32_t max) const {
    uint32_t n = 0;
    if (num_non_xgmi_ != 0)
      for (uint32_t i = 0; i < num_mapped_ && n < max; ++i)
        if (mapped_hwids_[i] >= num_non_xgmi_) out[n++] = mapped_hwids_[i];
    return n ? n : EmitAll(out, max);
  }

  int XgmiPhysicalId(uint32_t device_id) const { return topology_.XgmiPhysicalId(device_id); }
  uint64_t HiveId(uint32_t device_id) const { return topology_.HiveId(device_id); }
  uint64_t RecommendedMask(uint32_t source_device_id, uint32_t destination_device_id) const {
    return topology_.RecommendedMask(source_device_id, destination_device_id);
  }

  const HeatmapProfile* hm_ = nullptr;

 private:
  friend class CopyOrchestrator;

  static PolicyResult Trampoline(void* context, const CopyEndpoint& src, const CopyEndpoint& dst,
                                 uint64_t candidate_mask, uint32_t* out, uint32_t max) {
    return static_cast<const SdmaEnginePolicy*>(context)->RankEngines(src, dst, candidate_mask, out,
                                                                      max);
  }

  /// Classify the transfer, ask the matching hook for an ordered hw-id list,
  /// then resolve each id to its registered index, keep only candidates, dedup,
  /// and cap at @p max.
  PolicyResult RankEngines(const CopyEndpoint& src, const CopyEndpoint& dst,
                           uint64_t candidate_mask, uint32_t* out, uint32_t max) const {
    uint8_t hwids[kMaxEngines];
    uint32_t num_hwids = 0;
    switch (ClassifyTransfer(src, dst)) {
      case TransferKind::kLocalDeviceToDevice:
        num_hwids = RankLocalD2D(src, dst, hwids, kMaxEngines);
        break;
      case TransferKind::kPeerToPeer:
        num_hwids = RankP2P(src, dst, hwids, kMaxEngines);
        break;
      case TransferKind::kHostToDevice:
        num_hwids = RankH2D(src, dst, hwids, kMaxEngines);
        break;
      case TransferKind::kDeviceToHost:
        num_hwids = RankD2H(src, dst, hwids, kMaxEngines);
        break;
      case TransferKind::kUnknown:
        break;
    }

    if (num_hwids == 0) return {PolicyDisposition::kNoLegalEngine, 0};

    uint32_t n = 0;
    uint64_t emitted = 0;
    for (uint32_t i = 0; i < num_hwids && n < max; ++i) {
      const uint8_t hw = hwids[i];
      if (hw >= kMaxEngines || hwid_to_reg_[hw] < 0) continue;
      const uint32_t reg = static_cast<uint32_t>(hwid_to_reg_[hw]);
      const uint64_t bit = 1ull << reg;
      if ((emitted & bit) || !(candidate_mask & bit)) continue;
      emitted |= bit;
      out[n++] = reg;
    }
    return {n ? PolicyDisposition::kRanked : PolicyDisposition::kNoLegalEngine, n};
  }

  int hwid_to_reg_[kMaxEngines];
  uint8_t mapped_hwids_[kMaxEngines]{};
  uint32_t num_mapped_ = 0;
  uint32_t num_non_xgmi_ = 0;
  TopologyData topology_;
};

/// @brief gfx94x host-copy + xGMI policy (gfx9, minor 4 or 5).
///
/// H2D/D2H order comes from the measured heatmap profile; P2P uses the
/// xGMI_physical_id SDMA-affinity table (even hardware ids only, *2) for the
/// optimal first choice, then appends every xGMI engine in registration order.
/// Same-hive is required for the xGMI band (RankP2P returns none when the two
/// GPUs' hives are known and differ).
class Gfx94xSdmaPolicy : public SdmaEnginePolicy {
 protected:
  // The heatmap is a performance ranking, not a reservation, so both host bands
  // append every remaining engine behind it: the measured order is preserved
  // (RankEngines drops the duplicates) while a client that registered engines
  // the profile does not mention -- or that the profile deliberately ranks last,
  // like the gfx950 tail -- still gets a legal engine instead of a failed map.
  uint32_t RankH2D(const CopyEndpoint&, const CopyEndpoint&, uint8_t* out,
                   uint32_t max) const override {
    const uint32_t n = hm_ ? Copy(hm_->h2d, hm_->h2d_count, out, max) : 0;
    return n < max ? n + EmitAll(out + n, max - n) : n;
  }
  uint32_t RankD2H(const CopyEndpoint&, const CopyEndpoint&, uint8_t* out,
                   uint32_t max) const override {
    const uint32_t n = hm_ ? Copy(hm_->d2h, hm_->d2h_count, out, max) : 0;
    return n < max ? n + EmitAll(out + n, max - n) : n;
  }
  uint32_t RankP2P(const CopyEndpoint& src, const CopyEndpoint& dst, uint8_t* out,
                   uint32_t max) const override {
    // Dedicated xGMI engines can only drive a directly-connected peer: require
    // the two GPUs share a hive when both hive ids are known.
    const uint64_t src_hive = HiveId(src.device_id), dst_hive = HiveId(dst.device_id);
    if (src_hive && dst_hive && src_hive != dst_hive) return 0;

    uint32_t n = 0;
    const int src_affinity = XgmiPhysicalId(src.device_id);
    const int dst_affinity = XgmiPhysicalId(dst.device_id);
    if (src_affinity >= 0 && src_affinity < 8 && dst_affinity >= 0 && dst_affinity < 8 && n < max)
      out[n++] = static_cast<uint8_t>(kMap[src_affinity][dst_affinity] * 2);  // even engines only.

    if (n < max) n += EmitXgmi(out + n, max - n);
    return n;
  }

 private:
  static uint32_t Copy(const uint8_t* src, uint32_t src_count, uint8_t* out, uint32_t max) {
    const uint32_t n = src_count < max ? src_count : max;
    for (uint32_t i = 0; i < n; ++i) out[i] = src[i];
    return n;
  }

  // SDMA-affinity map keyed by xGMI physical id (src row, dst col) -> logical xGMI engine.
  static constexpr int kMap[8][8] = {{0, 7, 6, 1, 2, 4, 5, 3}, {7, 0, 1, 5, 4, 2, 3, 6},
                                     {5, 1, 0, 6, 7, 3, 2, 4}, {1, 6, 5, 0, 3, 7, 4, 2},
                                     {2, 4, 7, 3, 0, 5, 6, 1}, {4, 2, 3, 7, 6, 0, 1, 5},
                                     {5, 3, 2, 4, 6, 1, 0, 7}, {3, 6, 4, 2, 1, 5, 7, 0}};
};

/// @brief gfx90a host-copy + xGMI policy (gfx9, minor 0, stepping 10).
///
/// Due to a RAS issue SDMA0 can only drive H2D copies, so SDMA0 is reserved
/// exclusively for that band: H2D = {SDMA0}, D2H = {SDMA1} (SDMA0 is simply
/// never listed for D2H).  P2P uses the dedicated xGMI band; the orchestrator
/// spreads successive P2P ops across it.
class Gfx90aSdmaPolicy : public SdmaEnginePolicy {
 protected:
  uint32_t RankH2D(const CopyEndpoint&, const CopyEndpoint&, uint8_t* out,
                   uint32_t max) const override {
    if (max == 0) return 0;
    out[0] = 0;  // SDMA0 only (hard RAS reservation).
    return 1;
  }
  uint32_t RankD2H(const CopyEndpoint&, const CopyEndpoint&, uint8_t* out,
                   uint32_t max) const override {
    if (max == 0) return 0;
    out[0] = 1;  // SDMA1 only; SDMA0 is reserved for H2D and never listed here.
    return 1;
  }
  uint32_t RankP2P(const CopyEndpoint& src, const CopyEndpoint& dst, uint8_t* out,
                   uint32_t max) const override {
    const uint64_t src_hive = HiveId(src.device_id), dst_hive = HiveId(dst.device_id);
    if (src_hive && dst_hive && src_hive != dst_hive) return 0;
    return EmitXgmi(out, max);
  }
};

/// @brief Construct the SDMA engine policy matching a gfx version, or nullptr for
/// parts with no specialized policy (engine selection then degrades to ascending /
/// round-robin).  The returned policy is unconfigured: the caller still maps its
/// engines (MapEngine) and loads topology / heatmap before use.
inline std::unique_ptr<SdmaEnginePolicy> MakeSdmaPolicy(uint32_t gfx_major, uint32_t gfx_minor,
                                                        uint32_t gfx_stepping) {
  if (gfx_major == 9) {
    // gfx90a: major 9, minor 0, stepping 10 — SDMA0 RAS reservation.
    if (gfx_minor == 0 && gfx_stepping == 10) return std::make_unique<Gfx90aSdmaPolicy>();
    // gfx94x / gfx95x: heatmap + xGMI SDMA-affinity map.
    if (gfx_minor == 4 || gfx_minor == 5) return std::make_unique<Gfx94xSdmaPolicy>();
  }
  return nullptr;
}

// ===========================================================================
// Plan (output of MapCopy)
// ===========================================================================

enum class MapStatus : uint8_t {
  kSuccess,
  kInvalidArgument,
  kNoEngine,
  kNoLegalEngine,
  kUnsupportedOperation,
  kTooManyOperations,
  kAllocationFailed,
  /// The batch fanned out across engines and needs a coordination word to
  /// sequence them, but SignalRef::coordination_scratch is null. Distinct from
  /// kInvalidArgument because it depends on batch size and engine count, so the
  /// same client code can map smaller batches successfully and only trip here
  /// once a copy is large enough to fan out.
  kMissingCoordinationScratch,
};

enum class SubmitStatus : uint8_t {
  kSuccess,
  kInvalidPlan,
  kFrameTooLarge,
  kRingUnavailable,
};

/// @brief One frame of the mapping decision: the group of packets that Submit
/// writes to a single engine's ring, in one reservation (one doorbell).
///
/// A fan-out (e.g. a hipMemcpyBatchAsync spreading work over 4 devices) maps to
/// several frames — one per ring index — so a single MapCopy can return up to
/// @c kMaxEngines frames to submit in parallel.  @c frames[0] is always the
/// coordinator (carries the prologue + epilogue).
struct PlanFrame {
  uint32_t engine = 0;  ///< selected registry index / ring. The client may replace this with
                        ///< any entry in @c ranked_engines before Submit().
  uint32_t num_ops = 0;  ///< bodies on this frame.
  uint32_t bytes = 0;    ///< total packet bytes for this frame.
  uint8_t num_ranked_engines = 0;  ///< valid entries in @c ranked_engines.
  bool has_engine_preference = false;  ///< policy ranked the alternatives; false means equal.
  bool coordinator = false;  ///< frames[0]: carries the prologue + epilogue.
  uint32_t ranked_engines[kMaxEngineChoices]{};  ///< legal alternatives; best first when
                                                 ///< @c has_engine_preference is true.
  /// Non-owning view of this frame's @c bytes formed packets. Points into the
  /// orchestrator's per-thread packet scratch (see MapCopy); valid only until
  /// the same thread maps another plan. Submit copies it into the ring.
  const char* packets = nullptr;
};

/// @brief The output of MapCopy: a fully materialized batch ready for Submit.
///
/// Outputs (filled by MapCopy):
///   frames[0..num_frames) — one per participating ring; frames[0] is the
///                           coordinator (carries prologue + epilogue). Each
///                           frame also exposes up to @c kMaxEngineChoices
///                           legal alternatives and records whether policy
///                           ranked them. A client may replace @c frame.engine
///                           with one of those registered indices before
///                           Submit(); multi-frame clients must keep the
///                           selected frame engines distinct.
///   num_frames       — number of distinct engines used.
///   multi            — true when the copy is spread across more than one SDMA
///                      engine (num_frames > 1), so the frames run in parallel
///                      and need cross-engine start/fan-in coordination.
///   coordination_signal / coordination_initial_value — the word Submit arms
///                           before publishing, carrying the low-32 fan-in count
///                           and, when required, the bit-62 start gate. Normally
///                           SignalRef::coordination_scratch; the output signal's
///                           own value only when the body signals are themselves
///                           the completion transitions.
///   status           — typed mapping result.
struct Plan {
  Plan() = default;
  Plan(const Plan&) = delete;
  Plan& operator=(const Plan&) = delete;

  Plan(Plan&& other) noexcept { MoveFrom(other); }
  Plan& operator=(Plan&& other) noexcept {
    if (this != &other) {
      MoveFrom(other);
    }
    return *this;
  }

  ~Plan() = default;

  PlanFrame frames[kMaxEngines];
  uint32_t num_frames = 0;

  bool multi = false;
  bool start_gate_required = false;
  bool epilogue_required = true;
  uint64_t fan_in_count = 0;
  void* coordination_signal = nullptr;
  uint64_t coordination_initial_value = 0;

  /// Set of TransferKind values present in the batch, as TransferKindBit()
  /// flags. A batch may legitimately mix kinds (an H2D and a D2H in one
  /// submission), so this is a set rather than a single label; use
  /// UniformTransferKind() when a single label is what you want.
  uint32_t transfer_kinds = 0;

  /// The batch's single transfer kind, or kUnknown if it mixes kinds. A profiler
  /// attributing one completion signal to one kind of copy can only do so when
  /// this is not kUnknown.
  TransferKind UniformTransferKind() const {
    if (transfer_kinds == 0 || MixedTransferKinds()) return TransferKind::kUnknown;
    return static_cast<TransferKind>(detail::CountTrailingZeros64(transfer_kinds));
  }

  /// Whether the batch spans more than one transfer kind.
  bool MixedTransferKinds() const { return (transfer_kinds & (transfer_kinds - 1)) != 0; }

  /// Where Submit stamps the ExecutionDescriptor, or null if the client did not
  /// ask for one. Carried from CopyMetadata because Submit does not see metadata.
  void* execution_descriptor = nullptr;

  MapStatus status = MapStatus::kInvalidArgument;
  uint32_t failed_op = UINT32_MAX;

  bool valid() const { return status == MapStatus::kSuccess; }

 private:
  friend class CopyOrchestrator;

  void MoveFrom(Plan& other) {
    for (uint32_t frame_idx = 0; frame_idx < kMaxEngines; ++frame_idx)
      frames[frame_idx] = std::move(other.frames[frame_idx]);
    num_frames = other.num_frames;
    multi = other.multi;
    start_gate_required = other.start_gate_required;
    epilogue_required = other.epilogue_required;
    fan_in_count = other.fan_in_count;
    coordination_signal = other.coordination_signal;
    coordination_initial_value = other.coordination_initial_value;
    transfer_kinds = other.transfer_kinds;
    execution_descriptor = other.execution_descriptor;
    status = other.status;
    failed_op = other.failed_op;
    submitted_ = other.submitted_;

    other.num_frames = 0;
    other.multi = false;
    other.start_gate_required = false;
    other.epilogue_required = true;
    other.fan_in_count = 0;
    other.coordination_signal = nullptr;
    other.coordination_initial_value = 0;
    other.transfer_kinds = 0;
    other.execution_descriptor = nullptr;
    other.status = MapStatus::kInvalidArgument;
    other.failed_op = UINT32_MAX;
    other.submitted_ = false;
  }

  bool submitted_ = false;
};

struct SubmitResult {
  SubmitStatus status = SubmitStatus::kInvalidPlan;
};

// ===========================================================================
// CopyOrchestrator — the ABCE front door
// ===========================================================================

/// @brief Maps batches of copies onto registered SDMA engines, builds the SDMA
/// packets, then submits them.
///
/// Two phases:
///   * MapCopy() — pick engines, decompose copies, and emit packets into owned
///                 buffers. Does not touch a ring or signal.
///   * Submit()  — reserve every ring, arm the output signal, memcpy
///                 each frame's pre-built packets in, then release.
class CopyOrchestrator {
 public:
  explicit CopyOrchestrator(const ABCE& builder)
      : CopyOrchestrator(builder, DetectDefaultPlatformCaps(builder.Isa())) {}

  CopyOrchestrator(const ABCE& builder, PlatformCaps caps)
      : builder_(builder),
        is_gfx125plus_(builder.IsGfx125plus()),
        caps_(caps),
        composer_(builder) {
    composer_.Configure(caps_);
    const IsaVersion& isa = builder_.Isa();
    owned_policy_ = MakeSdmaPolicy(isa.major, isa.minor, isa.stepping);
    if (owned_policy_) {
      policy_ = &SdmaEnginePolicy::Trampoline;
      policy_ctx_ = owned_policy_.get();
    }
  }

  /// @brief The SDMA engine policy auto-selected for this part, or nullptr
  /// if the part has none.  Engines are mapped automatically by RegisterEngine (via
  /// EngineAffinity); use this to load the remaining topology. The device-wide
  /// heatmap profile is selected by LoadTopologyFromKfd or InitDeviceProfile.
  SdmaEnginePolicy* sdma_policy() { return owned_policy_.get(); }

  /// Replace the auto-selected policy with a caller-supplied ranking callback.
  /// Passing nullptr restores no-preference fallback behavior.
  void SetEnginePolicy(EnginePolicyFn policy, void* context = nullptr) {
    policy_ = policy;
    policy_ctx_ = policy ? context : nullptr;
  }

  // ---- Device init ----

  /// @brief Establish device-wide selection state that does not depend on which
  /// individual engines are registered.  Selects the host-copy heatmap profile
  /// for this part from its total SDMA engine count (host + xGMI) — a fixed
  /// device property, so this is chosen once.  Call before RegisterEngine.
  ///
  /// The KFD/topology path (sdma_policy()->LoadTopologyFromKfd) selects the same
  /// profile from the count KFD reports; call this when KFD topology loading is
  /// disabled.
  ///
  /// @p num_non_xgmi_sdma_engines is KFD's num_sdma_engines: how many engines are
  /// not xGMI, the driver numbering them first. It establishes the P2P band the
  /// way LoadTopologyFromKfd does. Zero leaves the band unknown, which widens
  /// P2P to every registered engine rather than emptying it.
  void InitDeviceProfile(uint32_t total_sdma_engines, uint32_t num_non_xgmi_sdma_engines = 0) {
    if (!owned_policy_) return;
    const IsaVersion& isa = builder_.Isa();
    if (isa.major == 9 && (isa.minor == 4 || isa.minor == 5))
      owned_policy_->SelectHeatmapProfile(static_cast<uint8_t>(isa.minor),
                                          static_cast<uint8_t>(total_sdma_engines));
    owned_policy_->SetSdmaEngineSplit(num_non_xgmi_sdma_engines);
  }

  // ---- Registration ----

  /// Register an engine's ring and map its hardware id into the auto-selected
  /// SDMA policy, so the policy is populated without a separate MapEngine() call.
  bool RegisterEngine(uint32_t index, RingBuffer* ring, const EngineAffinity& affinity = {}) {
    if (index >= kMaxEngines || !ring || engines_[index] != nullptr) return false;
    EngineAffinity resolved_affinity = affinity;
    if (resolved_affinity.hw_engine_id == kAnyEngine) resolved_affinity.hw_engine_id = index;
    // A hardware id the policy cannot map would leave the engine registered but
    // unrankable, so refuse it here rather than half-register it.
    if (resolved_affinity.hw_engine_id >= kMaxEngines) return false;
    engines_[index] = ring;
    registered_mask_ |= (1ull << index);
    affinity_[index] = resolved_affinity;
    if (owned_policy_) owned_policy_->MapEngine(resolved_affinity.hw_engine_id, index);
    return true;
  }

  /// Affinity a registered engine was registered with, resolved (so
  /// hw_engine_id is never kAnyEngine for a registered index). Reporting a copy
  /// to anything outside ABCE wants the hardware id from here, not the
  /// registered index, which is a client-local numbering.
  const EngineAffinity& EngineAffinityFor(uint32_t index) const { return affinity_[index]; }

  /// Summarize a plan's final engine assignment for a reader of its completion
  /// signal. Public so a client that keeps this metadata somewhere other than
  /// CopyMetadata::execution_descriptor can encode the same word itself.
  uint32_t DescribeExecution(const Plan& plan) const {
    uint32_t hw_engine_mask = 0;
    for (uint32_t frame_idx = 0; frame_idx < plan.num_frames; ++frame_idx)
      hw_engine_mask |= uint32_t{1} << affinity_[plan.frames[frame_idx].engine].hw_engine_id;
    return ExecutionDescriptor::Encode(ExecutionEngineKind::kSdma, hw_engine_mask,
                                       plan.UniformTransferKind(), plan.MixedTransferKinds());
  }

  uint32_t RankLegalEngines(const CopyOp& copy, uint32_t* engines, uint32_t capacity) const {
    if (!engines || capacity == 0) return 0;
    uint32_t ranked[kMaxEngineChoices]{};
    uint32_t recommended = kMaxEngines;
    uint8_t num_ranked = 0;
    bool has_engine_preference = false;
    (void)LegalEngineSet(copy, registered_mask_, recommended, ranked, num_ranked,
                         has_engine_preference);
    const uint32_t count = num_ranked < capacity ? num_ranked : capacity;
    for (uint32_t rank = 0; rank < count; ++rank) engines[rank] = ranked[rank];
    return count;
  }

  // ---- Phase 1: Map ----

  /// @brief Decompose @p copies into a Plan with pre-built packets.  Returns an
  /// invalid Plan (valid==false) if no engine is available.  The returned Plan
  /// owns its packet buffers.
  Plan MapCopy(const CopyOp* copies, uint32_t num_copies, const CopyMetadata& metadata) {
    Plan plan{};
    plan.status = ValidateRequest(copies, num_copies, metadata, plan.failed_op);
    if (plan.status != MapStatus::kSuccess) return plan;

    const uint64_t candidates = Candidates(metadata);
    if (!candidates) {
      plan.status = MapStatus::kNoEngine;
      return plan;
    }
    plan.execution_descriptor = metadata.execution_descriptor;

    uint32_t capacity = 0;
    for (uint32_t copy_idx = 0; copy_idx < num_copies; ++copy_idx) {
      const uint32_t expansion =
          copies[copy_idx].kind == OpKind::kMulticast ? copies[copy_idx].num_dsts : 1;
      if (expansion > kMaxBatchEntries - capacity) {
        plan.status = MapStatus::kTooManyOperations;
        plan.failed_op = copy_idx;
        return plan;
      }
      capacity += expansion;
      plan.transfer_kinds |=
          TransferKindBit(ClassifyTransfer(copies[copy_idx].src_end, copies[copy_idx].dst_end));
    }
    std::unique_ptr<EngineOp[]> local_ops(new (std::nothrow) EngineOp[capacity]);
    if (!local_ops) {
      plan.status = MapStatus::kAllocationFailed;
      return plan;
    }
    uint32_t num_ops = 0;
    const bool force_back_to_back = IsBackToBackBatch(copies, num_copies, metadata);
    const uint32_t fixed_engine =
        force_back_to_back ? BackToBackEngine(copies[0], candidates) : kMaxEngines;
    if (force_back_to_back && fixed_engine == kMaxEngines) {
      plan.status = MapStatus::kNoLegalEngine;
      plan.failed_op = 0;
      return plan;
    }

    if (!Decompose(copies, num_copies, metadata, candidates, force_back_to_back, fixed_engine,
                   local_ops.get(), num_ops, capacity, plan.status, plan.failed_op))
      return plan;
    if (num_ops == 0) {
      plan.status = MapStatus::kInvalidArgument;
      return plan;
    }

    BuildFrames(local_ops.get(), num_ops, plan);
    plan.multi = (plan.num_frames > 1);

    // Mark ops that use the fused wait/signal packet. gfx125+ can fuse direct
    // completion into a single-frame batch as well as fan-out synchronization
    // into a multi-frame batch.
    const bool use_fused_packets = metadata.prefer_fused && is_gfx125plus_;
    bool all_bodies_fused = true;
    for (uint32_t op_idx = 0; op_idx < num_ops; ++op_idx) {
      const OpKind kind = local_ops[op_idx].kind;
      if (kind == OpKind::kIndirect) {
        local_ops[op_idx].fused = true;
      } else if (use_fused_packets &&
                 (kind == OpKind::kLinear || kind == OpKind::kMulticast || kind == OpKind::kSwap)) {
        local_ops[op_idx].fused = true;
      }
      all_bodies_fused = all_bodies_fused && local_ops[op_idx].fused;
    }

    // Match ROCr's gfx125+ dependency-free fan-out path: when every body is a
    // fused wait/signal packet and the coordinator has no ordering work, all
    // frames can start directly. Their final packets signal the fan-in count,
    // including the coordinator's. Other plans retain the bit-62 start gate.
    const bool effective_hdp_flush =
        caps_.emit_hdp_flush && metadata.coherency.flags.emit_hdp_flush;
    const bool effective_gcr =
        metadata.coherency.flags.emit_gcr && builder_.RequiresGcr() &&
        !caps_.driver_manages_gcr;
    const bool coordinator_has_ordering_work =
        metadata.num_deps != 0 || metadata.timestamps.enabled() ||
        effective_hdp_flush || effective_gcr;
    plan.start_gate_required =
        plan.multi && (!all_bodies_fused || coordinator_has_ordering_work);
    // ROCr's gfx125+ fast path also omits the fan-in poll and final completion
    // update when no post-copy ordering or interrupt work remains. The last
    // fused body SIGNAL then transitions the output directly to its done value.
    plan.epilogue_required =
        !all_bodies_fused || plan.start_gate_required || metadata.timestamps.enabled() ||
        effective_gcr || metadata.out.event_mailbox != nullptr;
    plan.fan_in_count =
        plan.multi ? plan.num_frames - (plan.start_gate_required ? 1u : 0u) : 0;

    // Where the gate and fan-in counter live. When the coordinator drains fan-in
    // and then writes completion itself, they belong in ABCE's own scratch word:
    // keeping them out of the signal's value means the value only ever holds
    // legal signal states, and lets the counter drain to a plain zero instead of
    // a value biased by the caller's completion value. A fan-out with no
    // epilogue has nothing to drain — its body signals are themselves the
    // completion transitions — so it counts down the signal value directly, as
    // does a single-frame direct fused completion.
    const bool coordinator_drains_fan_in = plan.multi && plan.epilogue_required;
    if (coordinator_drains_fan_in) {
      if (!metadata.out.coordination_scratch) {
        plan.status = MapStatus::kMissingCoordinationScratch;
        return plan;
      }
      // The gfx125+ 64-bit poll encodes only addr[63:3], so a misaligned
      // coordination word would be silently polled at the wrong address.
      ABCE_ASSERT((reinterpret_cast<uintptr_t>(metadata.out.coordination_scratch) % 8) == 0 &&
                  "coordination_scratch must be 8-byte aligned");
      plan.coordination_signal = metadata.out.coordination_scratch;
      plan.coordination_initial_value =
          (plan.start_gate_required ? kFanOutStartGate : 0) | plan.fan_in_count;
    } else {
      plan.coordination_signal = metadata.out.value;
      if (plan.multi) {
        if (metadata.out.completion_value > UINT32_MAX ||
            plan.fan_in_count > UINT32_MAX - metadata.out.completion_value) {
          plan.status = MapStatus::kTooManyOperations;
          return plan;
        }
        plan.coordination_initial_value = plan.fan_in_count + metadata.out.completion_value;
      }
    }

    // Size each frame.
    try {
      for (uint32_t frame_idx = 0; frame_idx < plan.num_frames; ++frame_idx) {
        const size_t frame_bytes = composer_.FrameBytes(
            MakeJob(local_ops.get(), num_ops, metadata, plan, plan.frames[frame_idx]));
        if (frame_bytes > UINT32_MAX) {
          plan.status = MapStatus::kTooManyOperations;
          return plan;
        }
        plan.frames[frame_idx].bytes = static_cast<uint32_t>(frame_bytes);
      }
    } catch (const std::invalid_argument&) {
      plan.status = MapStatus::kInvalidArgument;
      return plan;
    }

    // Build packets into a per-thread scratch buffer (relocatable — no ring
    // offset embedded). Reusing the scratch avoids a per-call heap allocation
    // on the hot Map->Submit path; it grows once to the batch high-water mark.
    // Frames are laid out contiguously and each frame gets a non-owning view.
    // The zeroed-buffer contract mirrors the ring's Acquire memset.
    size_t total_bytes = 0;
    for (uint32_t frame_idx = 0; frame_idx < plan.num_frames; ++frame_idx)
      total_bytes += plan.frames[frame_idx].bytes;

    char* scratch = AcquirePacketScratch(total_bytes);
    if (!scratch && total_bytes > 0) {
      plan.status = MapStatus::kAllocationFailed;
      return plan;
    }
    size_t scratch_offset = 0;
    for (uint32_t frame_idx = 0; frame_idx < plan.num_frames; ++frame_idx) {
      PlanFrame& frame = plan.frames[frame_idx];
      char* frame_buffer = scratch + scratch_offset;
      std::memset(frame_buffer, 0, frame.bytes);
      const size_t emitted = composer_.EmitFrame(
          frame_buffer, MakeJob(local_ops.get(), num_ops, metadata, plan, frame));
      (void)emitted;
      assert(emitted == frame.bytes && "ABCE emit/size divergence");
      frame.packets = frame_buffer;
      scratch_offset += frame.bytes;
    }

    plan.status = MapStatus::kSuccess;
    return plan;
  }

  // ---- Phase 2: Submit ----

  /// @brief Reserve every ring, arm the output signal for fan-out, copy the
  /// frames into their reservations, and publish them.
  SubmitResult Submit(Plan& plan) {
    SubmitResult result{};
    if (!plan.valid() || plan.submitted_) return result;

    struct Pending {
      RingBuffer* ring = nullptr;
      RingReservation reservation{};
      uint32_t reserve_bytes = 0;
      char* buffer = nullptr;
    };
    Pending pending[kMaxEngines]{};
    uint8_t acquired_frames[kMaxEngines]{};
    uint32_t num_pending = 0;

    // Validate every frame before reserving any ring so deterministic failures
    // cannot leave an unpublished reservation behind.
    if (plan.num_frames == 0 || plan.num_frames > kMaxEngines) return result;
    uint64_t selected_ring_mask = 0;
    for (uint32_t frame_idx = 0; frame_idx < plan.num_frames; ++frame_idx) {
      const PlanFrame& frame = plan.frames[frame_idx];
      if (frame.engine >= kMaxEngines || frame.packets == nullptr || frame.bytes == 0 ||
          (selected_ring_mask & (uint64_t{1} << frame.engine))) {
        return result;
      }
      selected_ring_mask |= uint64_t{1} << frame.engine;
      RingBuffer* ring = engines_[frame.engine];
      if (!ring) {
        result.status = SubmitStatus::kRingUnavailable;
        return result;
      }
      Pending& entry = pending[frame_idx];
      entry.ring = ring;
      if (entry.ring->PaddedSize(frame.bytes, entry.reserve_bytes) != RingStatus::kSuccess) {
        result.status = SubmitStatus::kFrameTooLarge;
        return result;
      }
    }

    // All submitters acquire rings in the same global order. Without this,
    // plans whose coordinator/first-op choices produce opposite frame orders
    // can each hold one unpublished reservation while waiting forever for the
    // other's ring.
    for (uint32_t engine_idx = 0; engine_idx < kMaxEngines; ++engine_idx) {
      for (uint32_t frame_idx = 0; frame_idx < plan.num_frames; ++frame_idx) {
        if (plan.frames[frame_idx].engine != engine_idx) continue;
        Pending& entry = pending[frame_idx];
        const RingStatus acquire_status =
            entry.ring->Acquire(entry.reserve_bytes, entry.reservation, entry.buffer);
        if (acquire_status != RingStatus::kSuccess) {
          for (uint32_t acquired_idx = 0; acquired_idx < num_pending; ++acquired_idx) {
            Pending& acquired_entry = pending[acquired_frames[acquired_idx]];
            acquired_entry.ring->Cancel(acquired_entry.reservation);
          }
          result.status = SubmitStatus::kRingUnavailable;
          return result;
        }
        acquired_frames[num_pending++] = static_cast<uint8_t>(frame_idx);
      }
    }

    // Arm the coordination word only after every reservation succeeds. No frame
    // is visible to hardware yet, so a deterministic/acquire failure cannot
    // strand a gate or a fan-in count that nothing will ever clear.
    if (plan.multi) {
      __atomic_store_n(static_cast<uint64_t*>(plan.coordination_signal),
                       plan.coordination_initial_value, __ATOMIC_RELEASE);
    }

    // Stamp the execution descriptor here, not in MapCopy: a client may retarget
    // PlanFrame::engine after mapping, so the engines are only final now. The
    // ring publish below is what orders this store ahead of the copy completing.
    if (plan.execution_descriptor) {
      __atomic_store_n(static_cast<uint32_t*>(plan.execution_descriptor),
                       DescribeExecution(plan), __ATOMIC_RELEASE);
    }

    for (uint32_t pending_idx = 0; pending_idx < num_pending; ++pending_idx) {
      const PlanFrame& frame = plan.frames[pending_idx];
      std::memcpy(pending[pending_idx].buffer, frame.packets, frame.bytes);
    }

    // Publish in the same global ring order used for acquisition. Publishing
    // in frame order can create a cross-ring commit cycle between concurrent
    // plans whose coordinator choices produce opposite frame orders.
    for (uint32_t acquired_idx = 0; acquired_idx < num_pending; ++acquired_idx) {
      Pending& acquired_entry = pending[acquired_frames[acquired_idx]];
      if (acquired_entry.ring->Release(acquired_entry.reservation) != RingStatus::kSuccess) {
        result.status = SubmitStatus::kRingUnavailable;
        return result;
      }
    }

    plan.submitted_ = true;
    result.status = SubmitStatus::kSuccess;
    return result;
  }

  /// @brief Convenience: map and submit.
  SubmitResult Dispatch(const CopyOp* copies, uint32_t num_copies, const CopyMetadata& metadata) {
    Plan plan = MapCopy(copies, num_copies, metadata);
    if (!plan.valid()) return {};
    return Submit(plan);
  }

 private:
  // ---- Map-level limits ----

  static constexpr uint32_t kMaxBatchEntries = 64 * kKi;
  // The back-to-back path serializes a whole batch onto one ring, so what it
  // trades against fan-out is the batch's *total* bytes, not the size of any one
  // entry: one ring pays the full total while E rings each pay roughly total/E,
  // set against a fixed coordination cost for fanning out at all. Measured on
  // the 8-rank all-gather batch (MI300X, 16 SDMA engines), fan-out sits at a ~27
  // us floor no matter how small the copies are, while one ring starts near 13
  // us, and the two curves cross just under 1 MiB of total batch traffic: at 512
  // KiB total one ring wins 22.4 vs. 28.2 us, at 1 MiB fan-out edges ahead 29.6
  // vs. 31.6 us, and by 4 MiB it is decisive, 37.2 vs. 92.2 us. So keep one ring
  // through the region where it is ahead and hand everything past the crossover
  // to fan-out. Override via ABCE_LINEAR_B2B_MAX_TOTAL.
  //
  // Writing this rule per copy is what made an 8 x 256 KiB batch serialize onto a
  // single ring at 51.6 us where fan-out needed 32.3: a per-copy window is only
  // valid at the entry count it was tuned on, and silently mis-sizes every other
  // batch shape, since 64 x 8 KiB moves exactly as many bytes as 8 x 64 KiB.
  static constexpr size_t kLinearB2BMaxTotalDefault = 512 * kKi;
  static constexpr size_t kMulticastMaxSize = 256 * kKi;
  static constexpr size_t kLargeCopyMinSize = uint64_t{1} << 30;
  static constexpr uint32_t kMaxCopiesPerEngine = 8;
  static constexpr size_t kBroadcastMaxSizeDefault = 16 * kKi;
  static constexpr uint32_t kMulticastMaxDsts = kKi;
  static constexpr uint32_t kBroadcastMaxDsts = 2;

  // Retuning knobs (bytes). The linear back-to-back window and the broadcast-
  // packet cutoff can be overridden at process start via environment so they can
  // be swept without recompiling. Parsed once (thread-safe local static); accepts
  // decimal or 0x-hex. Unset/empty falls back to the compiled default.
  static size_t EnvSizeOr(const char* name, size_t fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') return fallback;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(raw, &end, 0);
    if (end == raw) return fallback;
    return static_cast<size_t>(parsed);
  }
  static size_t LinearB2BMaxTotal() {
    static const size_t value =
        EnvSizeOr("ABCE_LINEAR_B2B_MAX_TOTAL", kLinearB2BMaxTotalDefault);
    return value;
  }
  static size_t BroadcastMaxSize() {
    static const size_t value = EnvSizeOr("ABCE_BROADCAST_MAX", kBroadcastMaxSizeDefault);
    return value;
  }

  // ---- Candidate resolution ----

  uint64_t Candidates(const CopyMetadata& metadata) const {
    return metadata.engine_mask ? (metadata.engine_mask & registered_mask_) : registered_mask_;
  }

  /// Bytes the back-to-back path would push through a single ring. A multicast
  /// op decomposed on non-gfx125plus writes its payload once per destination, so
  /// its share scales with the destination count rather than being one copy.
  static size_t BackToBackBytes(const CopyOp* copies, uint32_t num_copies) {
    size_t total = 0;
    for (uint32_t copy_idx = 0; copy_idx < num_copies; ++copy_idx) {
      const CopyOp& copy = copies[copy_idx];
      const uint32_t writes = copy.kind == OpKind::kMulticast ? copy.num_dsts : 1;
      total += copy.size * writes;
    }
    return total;
  }

  bool IsBackToBackBatch(const CopyOp* copies, uint32_t num_copies,
                         const CopyMetadata& metadata) const {
    if (metadata.linear_batch_mode == LinearBatchMode::kForceFanOut) return false;
    // A single multicast is eligible because non-gfx125plus decomposes it into
    // one write per destination, which is the same back-to-back shape a linear
    // batch has. Every other single-op batch has nothing to serialize.
    const bool decomposed_multicast =
        num_copies == 1 && copies[0].kind == OpKind::kMulticast && !is_gfx125plus_;
    if (!decomposed_multicast) {
      if (num_copies <= 1) return false;
      for (uint32_t copy_idx = 0; copy_idx < num_copies; ++copy_idx)
        if (copies[copy_idx].kind != OpKind::kLinear) return false;
    }
    if (metadata.linear_batch_mode == LinearBatchMode::kForceBackToBack) return true;
    return BackToBackBytes(copies, num_copies) <= LinearB2BMaxTotal();
  }

  static uint32_t EngineAt(uint64_t mask, uint32_t nth) {
    for (uint32_t engine = 0; engine < kMaxEngines; ++engine) {
      if (mask & (1ull << engine)) {
        if (nth == 0) return engine;
        --nth;
      }
    }
    return kMaxEngines;
  }

  static uint32_t PopCount(uint64_t mask) {
    return static_cast<uint32_t>(detail::PopCount64(mask));
  }

  static constexpr uint32_t kRankMax = kMaxEngines;

  PolicyResult InvokePolicy(const CopyOp& op, uint64_t candidates,
                            uint32_t (&ranked)[kRankMax]) const {
    if (!policy_) return {};
    PolicyResult result =
        policy_(policy_ctx_, op.src_end, op.dst_end, candidates, ranked, kRankMax);
    if (result.disposition != PolicyDisposition::kRanked) {
      result.count = 0;
      return result;
    }

    const uint32_t reported_count = result.count < kRankMax ? result.count : kRankMax;
    uint64_t emitted = 0;
    uint32_t valid_count = 0;
    for (uint32_t rank_idx = 0; rank_idx < reported_count; ++rank_idx) {
      const uint32_t engine = ranked[rank_idx];
      if (engine >= kMaxEngines) continue;
      const uint64_t engine_bit = 1ull << engine;
      if (!(candidates & engine_bit) || (emitted & engine_bit)) continue;
      emitted |= engine_bit;
      ranked[valid_count++] = engine;
    }
    return {valid_count ? PolicyDisposition::kRanked : PolicyDisposition::kNoLegalEngine,
            valid_count};
  }

  uint32_t BestEngine(const CopyOp& op, uint64_t candidates) const {
    uint32_t ranked[kRankMax] = {};
    const PolicyResult result = InvokePolicy(op, candidates, ranked);
    if (result.disposition == PolicyDisposition::kNoLegalEngine) return kMaxEngines;
    if (result.disposition == PolicyDisposition::kRanked) return ranked[0];
    return FirstEngine(candidates);
  }

  uint32_t BackToBackEngine(const CopyOp& first_copy, uint64_t candidates) const {
    const IsaVersion& isa = builder_.Isa();
    if (owned_policy_ && isa.major == 9 && isa.minor >= 4) {
      const uint32_t reversed_host_engine = owned_policy_->RegisteredEngine(1);
      if (reversed_host_engine < kMaxEngines &&
          (candidates & (uint64_t{1} << reversed_host_engine)))
        return reversed_host_engine;
    }
    return BestEngine(first_copy, candidates);
  }

  uint8_t RankEngineChoices(const CopyOp& op, uint64_t candidates,
                            uint32_t (&choices)[kMaxEngineChoices],
                            bool& has_engine_preference) const {
    uint32_t ranked[kRankMax] = {};
    const PolicyResult result = InvokePolicy(op, candidates, ranked);
    has_engine_preference = result.disposition == PolicyDisposition::kRanked;
    if (result.disposition == PolicyDisposition::kNoLegalEngine) return 0;

    const uint32_t num_choices =
        result.disposition == PolicyDisposition::kRanked
            ? (result.count < kMaxEngineChoices ? result.count : kMaxEngineChoices)
            : (PopCount(candidates) < kMaxEngineChoices ? PopCount(candidates)
                                                        : kMaxEngineChoices);
    for (uint32_t choice_idx = 0; choice_idx < num_choices; ++choice_idx) {
      choices[choice_idx] =
          result.disposition == PolicyDisposition::kRanked
              ? ranked[choice_idx]
              : EngineAt(candidates, choice_idx);
    }
    return static_cast<uint8_t>(num_choices);
  }

  uint32_t SelectEngine(const CopyOp& op, uint64_t candidates, bool can_fanout,
                        uint32_t max_participants, uint32_t& round_robin,
                        uint64_t& selected_mask,
                        uint32_t (&ranked_engines)[kMaxEngineChoices],
                        uint8_t& num_ranked_engines, bool& has_engine_preference) const {
    const bool participant_cap_reached = PopCount(selected_mask) >= max_participants;
    const uint64_t unselected_candidates = candidates & ~selected_mask;
    const bool add_participant =
        can_fanout && !participant_cap_reached && unselected_candidates != 0;
    const uint64_t allowed =
        add_participant
            ? unselected_candidates
            : (selected_mask != 0 && (!can_fanout || participant_cap_reached))
                  ? selected_mask
                  : candidates;
    num_ranked_engines =
        RankEngineChoices(op, candidates, ranked_engines, has_engine_preference);
    uint32_t allowed_ranked_engines[kMaxEngineChoices]{};
    bool allowed_has_engine_preference = false;
    const uint8_t num_allowed_ranked_engines =
        RankEngineChoices(op, allowed, allowed_ranked_engines, allowed_has_engine_preference);
    if (num_allowed_ranked_engines == 0) return kMaxEngines;
    const uint32_t engine = add_participant || !can_fanout
                                ? allowed_ranked_engines[0]
                                : allowed_ranked_engines[round_robin++ %
                                                         num_allowed_ranked_engines];
    if (engine < kMaxEngines) selected_mask |= uint64_t{1} << engine;
    return engine;
  }

  // ---- Fan-out engine assignment ----
  //
  // Fan-out spreads a batch across engines for parallelism. The rule is
  // "parallel first, preferred only when doubling up": every ring legal for a
  // copy takes a copy before any ring takes a second, and once the copies
  // outnumber the legal rings the extra copies prefer their recommended (kMap
  // affinity) ring. A per-ring load counter drives this.
  //
  //   * gfx94x/95x — a copy's legal set is class-based (P2P -> xGMI rings; local
  //     D2D / host copies -> host rings) with the kMap affinity ring ranked
  //     first. The classes are disjoint, so P2P copies balance across the xGMI
  //     rings while local copies run in parallel on host rings, with no
  //     cross-class contention.
  //   * gfx1250 — every ring is equivalent: the legal set is simply every
  //     candidate ring and there is no recommended ring, so the balancer
  //     degenerates to round-robin, independent of src/dst.

  /// Least-loaded legal ring, preferring @p recommended on a tie so a copy stays
  /// on its affinity ring until that ring is as busy as the alternatives.
  /// Returns kMaxEngines when @p legal_mask is empty.
  static uint32_t PickBalancedEngine(uint64_t legal_mask, uint32_t recommended,
                                     const uint32_t (&load)[kMaxEngines]) {
    uint32_t min_load = UINT32_MAX;
    for (uint32_t engine = 0; engine < kMaxEngines; ++engine)
      if ((legal_mask & (uint64_t{1} << engine)) && load[engine] < min_load)
        min_load = load[engine];
    if (min_load == UINT32_MAX) return kMaxEngines;
    if (recommended < kMaxEngines && (legal_mask & (uint64_t{1} << recommended)) &&
        load[recommended] == min_load)
      return recommended;
    for (uint32_t engine = 0; engine < kMaxEngines; ++engine)
      if ((legal_mask & (uint64_t{1} << engine)) && load[engine] == min_load)
        return engine;
    return kMaxEngines;
  }

  /// Legal ring set for @p op. On gfx1250 every candidate ring is legal and no
  /// ring is preferred; elsewhere the policy ranks the class-legal rings, with
  /// the recommended (affinity) ring first. Also fills @p ranked_engines for the
  /// EngineOp so downstream ring substitution still has the ordered choices.
  uint64_t LegalEngineSet(const CopyOp& op, uint64_t candidates, uint32_t& recommended,
                          uint32_t (&ranked_engines)[kMaxEngineChoices],
                          uint8_t& num_ranked_engines, bool& has_engine_preference) const {
    if (is_gfx125plus_) {
      num_ranked_engines = 0;
      for (uint32_t engine = 0; engine < kMaxEngines && num_ranked_engines < kMaxEngineChoices;
           ++engine)
        if (candidates & (uint64_t{1} << engine))
          ranked_engines[num_ranked_engines++] = engine;
      has_engine_preference = false;
      recommended = kMaxEngines;
      return candidates;
    }
    num_ranked_engines = RankEngineChoices(op, candidates, ranked_engines, has_engine_preference);
    uint64_t legal_mask = 0;
    for (uint32_t choice_idx = 0; choice_idx < num_ranked_engines; ++choice_idx)
      legal_mask |= uint64_t{1} << ranked_engines[choice_idx];
    recommended = num_ranked_engines ? ranked_engines[0] : kMaxEngines;
    return legal_mask;
  }

  // ---- Decomposition ----

  MapStatus ValidateRequest(const CopyOp* copies, uint32_t num_copies, const CopyMetadata& metadata,
                            uint32_t& failed_op) const {
    if (!copies || num_copies == 0 || num_copies > kMaxBatchEntries || !metadata.out.value ||
        (metadata.num_deps != 0 && !metadata.deps))
      return MapStatus::kInvalidArgument;

    for (uint32_t op_idx = 0; op_idx < num_copies; ++op_idx) {
      const CopyOp& copy = copies[op_idx];
      failed_op = op_idx;
      if (copy.src_end.kind == EndpointKind::kHost && copy.dst_end.kind == EndpointKind::kHost)
        return MapStatus::kInvalidArgument;
      if (copy.kind == OpKind::kCopyRect) {
        if (!copy.rect || !copy.rect->src.base || !copy.rect->dst.base || copy.rect->range.x == 0 ||
            copy.rect->range.y == 0 || copy.rect->range.z == 0)
          return MapStatus::kInvalidArgument;
        continue;
      }
      if (copy.size == 0) return MapStatus::kInvalidArgument;

      switch (copy.kind) {
        case OpKind::kLinear:
        case OpKind::kSwap:
          if (!copy.src || !copy.dst) return MapStatus::kInvalidArgument;
          break;
        case OpKind::kIndirect:
          if (!is_gfx125plus_) return MapStatus::kUnsupportedOperation;
          if (!copy.src || !copy.dst || copy.size > (uint64_t{1} << 30))
            return MapStatus::kInvalidArgument;
          break;
        case OpKind::kMulticast:
        case OpKind::kBroadcast:
          if (!copy.src || !copy.dsts || copy.num_dsts == 0) return MapStatus::kInvalidArgument;
          if (copy.kind == OpKind::kBroadcast && copy.num_dsts != kBroadcastMaxDsts)
            return MapStatus::kInvalidArgument;
          for (uint32_t dst_idx = 0; dst_idx < copy.num_dsts; ++dst_idx)
            if (!copy.dsts[dst_idx]) return MapStatus::kInvalidArgument;
          break;
        case OpKind::kFill:
          if (!copy.dst || (copy.size % sizeof(uint32_t)) != 0) return MapStatus::kInvalidArgument;
          break;
        case OpKind::kCopyRect:
          break;
      }
    }
    failed_op = UINT32_MAX;
    return MapStatus::kSuccess;
  }

  static bool PushOp(EngineOp* ops, uint32_t& num_ops, uint32_t capacity, const EngineOp& op) {
    if (num_ops >= capacity) return false;
    ops[num_ops++] = op;
    return true;
  }

  static void SetEngineChoices(EngineOp& op,
                               const uint32_t (&ranked_engines)[kMaxEngineChoices],
                               uint8_t num_ranked_engines, bool has_engine_preference) {
    op.num_ranked_engines = num_ranked_engines;
    op.has_engine_preference = has_engine_preference;
    for (uint32_t choice_idx = 0; choice_idx < num_ranked_engines; ++choice_idx)
      op.ranked_engines[choice_idx] = ranked_engines[choice_idx];
  }

  void SetFixedEngine(EngineOp& op, const CopyOp& copy, uint64_t candidates,
                      uint32_t engine) const {
    op.engine = engine;
    bool has_engine_preference = false;
    op.num_ranked_engines =
        RankEngineChoices(copy, candidates, op.ranked_engines, has_engine_preference);
    op.has_engine_preference = has_engine_preference;
    if (op.num_ranked_engines == 0) {
      op.ranked_engines[0] = engine;
      op.num_ranked_engines = 1;
    }
  }

  bool Decompose(const CopyOp* copies, uint32_t num_copies, const CopyMetadata& metadata,
                 uint64_t candidates, bool force_back_to_back, uint32_t fixed_engine, EngineOp* ops,
                 uint32_t& num_ops, uint32_t capacity, MapStatus& status,
                 uint32_t& failed_op) const {
    const uint32_t num_candidates = PopCount(candidates);
    const bool can_fanout =
        !force_back_to_back && (num_candidates > 1) && caps_.device_atomic_support;
    const uint32_t max_participants =
        force_back_to_back ? 1
        : metadata.max_engines == 0
            ? (can_fanout ? kMaxEngines : 1)
            : (metadata.max_engines < kMaxEngines ? metadata.max_engines : kMaxEngines);
    const uint32_t num_group_engines =
        max_participants < num_candidates ? max_participants : num_candidates;
    bool use_large_copy_grouping = is_gfx125plus_ && can_fanout;
    for (uint32_t copy_idx = 0; copy_idx < num_copies && use_large_copy_grouping; ++copy_idx) {
      use_large_copy_grouping =
          copies[copy_idx].kind == OpKind::kLinear &&
          copies[copy_idx].size >= kLargeCopyMinSize;
    }
    uint32_t round_robin = 0;
    uint64_t selected_mask = 0;
    // Per-ring copy count that drives the fan-out balancer (see PickBalancedEngine).
    uint32_t engine_load[kMaxEngines] = {};

    for (uint32_t op_idx = 0; op_idx < num_copies; ++op_idx) {
      const CopyOp& cop = copies[op_idx];
      failed_op = op_idx;

      if (cop.kind == OpKind::kMulticast) {
        if (!DecomposeMultiDst(cop, metadata, candidates, can_fanout, max_participants,
                               force_back_to_back, fixed_engine, round_robin, selected_mask, ops,
                               num_ops, capacity))
          return status = num_ops >= capacity ? MapStatus::kTooManyOperations
                                              : MapStatus::kNoLegalEngine,
                 false;
        continue;
      }

      EngineOp engine_op{};
      engine_op.kind = cop.kind;
      engine_op.src = cop.src;
      engine_op.dst = cop.dst;
      engine_op.dsts = cop.dsts;
      engine_op.num_dsts = cop.num_dsts;
      engine_op.fill_value = cop.fill_value;
      engine_op.size = cop.size;
      engine_op.size2 = cop.size2;
      engine_op.indirect_src = cop.indirect_src;
      engine_op.indirect_dst = cop.indirect_dst;
      engine_op.rect = cop.rect;
      if (force_back_to_back) {
        SetFixedEngine(engine_op, cop, candidates, fixed_engine);
      } else if (use_large_copy_grouping) {
        const uint32_t engine_slot =
            (op_idx / kMaxCopiesPerEngine) % num_group_engines;
        SetFixedEngine(engine_op, cop, candidates, EngineAt(candidates, engine_slot));
      } else if (can_fanout) {
        // Parallel-first placement: least-loaded legal ring, biased to the
        // recommended (kMap affinity) ring so a copy stays on it until the ring
        // is as busy as its alternatives.
        bool has_engine_preference = false;
        uint32_t recommended = kMaxEngines;
        const uint64_t legal_mask =
            LegalEngineSet(cop, candidates, recommended, engine_op.ranked_engines,
                           engine_op.num_ranked_engines, has_engine_preference);
        engine_op.has_engine_preference = has_engine_preference;
        const bool can_add_participant = PopCount(selected_mask) < max_participants;
        const uint64_t unselected_legal = legal_mask & ~selected_mask;
        const uint64_t selected_legal = legal_mask & selected_mask;
        const uint64_t allowed_mask =
            (can_add_participant && unselected_legal != 0) ? unselected_legal : selected_legal;
        engine_op.engine = PickBalancedEngine(allowed_mask, recommended, engine_load);
        if (engine_op.engine < kMaxEngines) {
          selected_mask |= uint64_t{1} << engine_op.engine;
          ++engine_load[engine_op.engine];
        }
      } else {
        bool has_engine_preference = false;
        engine_op.engine =
            SelectEngine(cop, candidates, can_fanout, max_participants, round_robin,
                         selected_mask, engine_op.ranked_engines,
                         engine_op.num_ranked_engines, has_engine_preference);
        engine_op.has_engine_preference = has_engine_preference;
      }
      if (engine_op.engine == kMaxEngines) {
        status = MapStatus::kNoLegalEngine;
        return false;
      }
      if (!PushOp(ops, num_ops, capacity, engine_op)) {
        status = MapStatus::kTooManyOperations;
        return false;
      }
    }
    status = MapStatus::kSuccess;
    failed_op = UINT32_MAX;
    return true;
  }

  bool DecomposeMultiDst(const CopyOp& op, const CopyMetadata& metadata, uint64_t candidates,
                         bool can_fanout, uint32_t max_participants, bool force_back_to_back,
                         uint32_t fixed_engine, uint32_t& round_robin, uint64_t& selected_mask,
                         EngineOp* ops, uint32_t& num_ops, uint32_t capacity) const {
    // gfx125plus multicast: one packet handles up to kMulticastMaxDsts destinations.
    // If num_dsts exceeds that, chunk into multiple ops spread across engines.
    const bool use_multicast =
        metadata.multicast_mode == MulticastMode::kForceMulticast ||
        (metadata.multicast_mode == MulticastMode::kAutomatic && op.size <= kMulticastMaxSize);
    if (is_gfx125plus_ && use_multicast) {
      uint32_t ranked_engines[kMaxEngineChoices]{};
      uint8_t num_ranked_engines = 0;
      bool has_engine_preference = false;
      const uint32_t multicast_engine =
          force_back_to_back
              ? fixed_engine
              : SelectEngine(op, candidates, can_fanout, max_participants, round_robin,
                             selected_mask, ranked_engines, num_ranked_engines,
                             has_engine_preference);
      if (multicast_engine == kMaxEngines) return false;
      if (force_back_to_back) {
        num_ranked_engines =
            RankEngineChoices(op, candidates, ranked_engines, has_engine_preference);
        if (num_ranked_engines == 0) {
          ranked_engines[0] = fixed_engine;
          num_ranked_engines = 1;
        }
      }
      for (uint32_t base = 0; base < op.num_dsts; base += kMulticastMaxDsts) {
        const uint32_t chunk_dsts =
            (op.num_dsts - base < kMulticastMaxDsts) ? (op.num_dsts - base) : kMulticastMaxDsts;
        EngineOp engine_op{};
        engine_op.kind = OpKind::kMulticast;
        engine_op.src = op.src;
        engine_op.dsts = &op.dsts[base];
        engine_op.num_dsts = chunk_dsts;
        engine_op.size = op.size;
        engine_op.engine = multicast_engine;
        SetEngineChoices(engine_op, ranked_engines, num_ranked_engines, has_engine_preference);
        if (!PushOp(ops, num_ops, capacity, engine_op)) return false;
      }
      return true;
    }

    if (!is_gfx125plus_ && op.size < BroadcastMaxSize()) {
      uint32_t ranked_engines[kMaxEngineChoices]{};
      uint8_t num_ranked_engines = 0;
      bool has_engine_preference = false;
      const uint32_t bcast_engine =
          force_back_to_back
              ? fixed_engine
              : SelectEngine(op, candidates, can_fanout, max_participants, round_robin,
                             selected_mask, ranked_engines, num_ranked_engines,
                             has_engine_preference);
      if (bcast_engine == kMaxEngines) return false;
      if (force_back_to_back) {
        num_ranked_engines =
            RankEngineChoices(op, candidates, ranked_engines, has_engine_preference);
        if (num_ranked_engines == 0) {
          ranked_engines[0] = fixed_engine;
          num_ranked_engines = 1;
        }
      }
      for (uint32_t base = 0; base < op.num_dsts; base += kBroadcastMaxDsts) {
        const uint32_t chunk_dsts =
            (op.num_dsts - base < kBroadcastMaxDsts) ? (op.num_dsts - base) : kBroadcastMaxDsts;
        EngineOp engine_op{};
        engine_op.src = op.src;
        engine_op.dsts = &op.dsts[base];
        engine_op.size = op.size;
        engine_op.engine = bcast_engine;
        SetEngineChoices(engine_op, ranked_engines, num_ranked_engines, has_engine_preference);
        if (chunk_dsts == kBroadcastMaxDsts) {
          engine_op.kind = OpKind::kBroadcast;
          engine_op.num_dsts = kBroadcastMaxDsts;
        } else {
          engine_op.kind = OpKind::kLinear;
          engine_op.dst = op.dsts[base];
        }
        if (!PushOp(ops, num_ops, capacity, engine_op)) return false;
      }
      return true;
    }

    const uint32_t num_candidates = PopCount(candidates);
    const uint32_t num_group_engines =
        max_participants < num_candidates ? max_participants : num_candidates;
    const bool use_large_copy_grouping =
        is_gfx125plus_ && can_fanout && op.size >= kLargeCopyMinSize;

    for (uint32_t dst_idx = 0; dst_idx < op.num_dsts; ++dst_idx) {
      EngineOp engine_op{};
      engine_op.kind = OpKind::kLinear;
      engine_op.src = op.src;
      engine_op.dst = op.dsts[dst_idx];
      engine_op.size = op.size;
      if (force_back_to_back) {
        SetFixedEngine(engine_op, op, candidates, fixed_engine);
      } else if (use_large_copy_grouping) {
        const uint32_t engine_slot =
            (dst_idx / kMaxCopiesPerEngine) % num_group_engines;
        SetFixedEngine(engine_op, op, candidates, EngineAt(candidates, engine_slot));
      } else {
        bool has_engine_preference = false;
        engine_op.engine =
            SelectEngine(op, candidates, can_fanout, max_participants, round_robin,
                         selected_mask, engine_op.ranked_engines,
                         engine_op.num_ranked_engines, has_engine_preference);
        engine_op.has_engine_preference = has_engine_preference;
      }
      if (engine_op.engine == kMaxEngines) return false;
      if (!PushOp(ops, num_ops, capacity, engine_op)) return false;
    }
    return true;
  }

  static void InitializeFrameChoices(PlanFrame& frame, const EngineOp& op) {
    frame.num_ranked_engines = op.num_ranked_engines;
    frame.has_engine_preference = op.has_engine_preference;
    for (uint32_t choice_idx = 0; choice_idx < op.num_ranked_engines; ++choice_idx)
      frame.ranked_engines[choice_idx] = op.ranked_engines[choice_idx];
  }

  static void IntersectFrameChoices(PlanFrame& frame, const EngineOp& op) {
    frame.has_engine_preference =
        frame.has_engine_preference && op.has_engine_preference;
    uint32_t common[kMaxEngineChoices]{};
    uint8_t num_common = 0;
    for (uint32_t frame_choice_idx = 0;
         frame_choice_idx < frame.num_ranked_engines && num_common < kMaxEngineChoices;
         ++frame_choice_idx) {
      const uint32_t candidate = frame.ranked_engines[frame_choice_idx];
      for (uint32_t op_choice_idx = 0; op_choice_idx < op.num_ranked_engines;
           ++op_choice_idx) {
        if (candidate == op.ranked_engines[op_choice_idx]) {
          common[num_common++] = candidate;
          break;
        }
      }
    }
    frame.num_ranked_engines = num_common;
    for (uint32_t choice_idx = 0; choice_idx < num_common; ++choice_idx)
      frame.ranked_engines[choice_idx] = common[choice_idx];
  }

  void BuildFrames(const EngineOp* ops, uint32_t num_ops, Plan& plan) const {
    uint64_t seen = 0;
    uint8_t engine_to_frame[kMaxEngines] = {};
    plan.num_frames = 0;
    for (uint32_t op_idx = 0; op_idx < num_ops; ++op_idx) {
      const uint32_t engine = ops[op_idx].engine;
      if (!(seen & (1ull << engine))) {
        seen |= (1ull << engine);
        PlanFrame& frame = plan.frames[plan.num_frames];
        frame = PlanFrame{};
        frame.engine = engine;
        InitializeFrameChoices(frame, ops[op_idx]);
        frame.coordinator = (plan.num_frames == 0);
        engine_to_frame[engine] = static_cast<uint8_t>(plan.num_frames);
        plan.num_frames++;
      } else {
        IntersectFrameChoices(plan.frames[engine_to_frame[engine]], ops[op_idx]);
      }
      plan.frames[engine_to_frame[engine]].num_ops++;
    }
  }

  // ---- Frame job assembly ----

  /// Bundle the current mapping state for one frame into the descriptor
  /// FrameComposer sizes / emits from.
  FrameJob MakeJob(const EngineOp* ops, uint32_t num_ops, const CopyMetadata& metadata,
                   const Plan& plan, const PlanFrame& frame) const {
    FrameJob job;
    job.ops = ops;
    job.num_ops = num_ops;
    job.metadata = &metadata;
    job.engine = frame.engine;
    job.frame_num_ops = frame.num_ops;
    job.coordinator = frame.coordinator;
    job.multi = plan.multi;
    job.start_gate_required = plan.start_gate_required;
    job.epilogue_required = plan.epilogue_required;
    job.coordination_word = plan.coordination_signal;
    return job;
  }

  /// @brief Return a per-thread packet scratch buffer of at least @p bytes,
  /// growing it to the high-water mark. The buffer is reused across MapCopy
  /// calls on the same thread, so a plan's packet views are valid only until
  /// that thread maps another plan. Returns nullptr only on allocation failure.
  static char* AcquirePacketScratch(size_t bytes) {
    thread_local std::unique_ptr<char[]> scratch;
    thread_local size_t capacity = 0;
    if (bytes == 0) return scratch.get();
    if (capacity < bytes) {
      scratch.reset(new (std::nothrow) char[bytes]);
      capacity = scratch ? bytes : 0;
    }
    return scratch.get();
  }

  const ABCE& builder_;
  bool is_gfx125plus_ = false;
  PlatformCaps caps_{};
  FrameComposer composer_;

  RingBuffer* engines_[kMaxEngines]{};
  EngineAffinity affinity_[kMaxEngines]{};
  uint64_t registered_mask_ = 0;

  std::unique_ptr<SdmaEnginePolicy> owned_policy_;  ///< auto-selected by Init (may be null).
  EnginePolicyFn policy_ = nullptr;
  void* policy_ctx_ = nullptr;
};

}  // namespace abce

#endif  // ABCE_HOST_H_
