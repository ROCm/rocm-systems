/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Accelerated Blit Copy Engine (ABCE) — SDMA engine preference table.
//
// ABCE treats every SDMA engine as an SDMA engine. The driver reports some as
// xGMI and some as not, and hardware gives them different measured bandwidth,
// but neither is a class ABCE reasons about: both collapse into one question —
// for a copy between these two agents, which engines are best, and are any
// forbidden? That question is answered by a lookup, not a policy.
//
// The answer is a table indexed by (source agent, destination agent), formed
// once when engines are registered and immutable afterwards. It has two sources:
//
//   * a compile-time per-part profile for host copies, because the good order
//     there is measured and cannot be derived. The same IP ships with different
//     engine counts and different measured orders, so profiles are keyed on
//     (gfx minor, total SDMA engines) rather than on the IP alone.
//
//   * the driver's own per-link recommendation for device-to-device copies. On
//     gfx94x KFD's recommended_sdma_engine_id_mask reproduces the xGMI
//     SDMA-affinity map exactly, agent-pair indexed and already correct for the
//     board, so ABCE derives nothing and reads no topology of its own.
//
// A pair the driver says nothing about, on a part with no profile, is simply
// empty — no opinion, and selection degrades to load balancing. That is the
// right answer wherever the engines really are interchangeable (gfx12+), and a
// safe one everywhere else, since engine choice only ever costs bandwidth.

#ifndef ABCE_ENGINE_TABLE_H_
#define ABCE_ENGINE_TABLE_H_

#include <cstdint>

#include "abce_types.h"

namespace abce {

/// Device agents the table covers, plus one slot for the host and one for
/// "unknown". Agent ids are the client's ABCE device ids — the same values it
/// puts in CopyEndpoint::device_id — so the table is indexed in whatever space
/// the client already names devices in.
constexpr uint32_t kMaxTableAgents = 16;

/// Agent id conventions. Values >= 0 are device agent ids.
constexpr int8_t kAgentHost = -1;     ///< CPU / host memory, whichever NUMA node.
constexpr int8_t kAgentUnknown = -2;  ///< untagged; no opinion is available.

/// Slots are: 0 for the host, n + 1 for device agent n, and one trailing slot
/// that absorbs kAgentUnknown and any agent id the table is too small to cover.
/// That trailing slot is a real, permanently empty row and column, so an
/// out-of-range lookup needs no special case — it lands on "no opinion".
constexpr uint32_t kTableSlots = kMaxTableAgents + 2;

ABCE_HD inline uint32_t AgentSlot(int32_t agent) {
  if (agent == kAgentHost) return 0;
  if (agent < 0 || static_cast<uint32_t>(agent) >= kMaxTableAgents) return kTableSlots - 1;
  return static_cast<uint32_t>(agent) + 1;
}

/// @brief Which engines to prefer for one ordered agent pair.
///
/// @c ranked is a PREFIX, not a whitelist. An engine missing from it stays legal
/// and simply sorts behind, which is what keeps a measured bandwidth order from
/// hardening into a reservation: a client that registered engines the profile
/// never mentions still gets a legal engine. Say "never" with @c forbidden_mask.
///
/// Entries are REGISTERED ENGINE INDICES, not hardware ids. The translation
/// happens once during formation, so nothing on the selection path has to know
/// the hardware numbering.
struct EngineOrder {
  uint8_t ranked[kMaxEngineChoices]{};
  uint8_t num_ranked = 0;
  /// Registered engines that must never run this pair (bit i => index i).
  uint64_t forbidden_mask = 0;

  ABCE_HD bool empty() const { return num_ranked == 0 && forbidden_mask == 0; }
};

/// @brief The formed table. Immutable once built; a plain value, so it can be
/// copied into a device-visible buffer as-is.
struct EngineTable {
  EngineOrder cells[kTableSlots][kTableSlots];

  ABCE_HD const EngineOrder& Cell(int32_t src_agent, int32_t dst_agent) const {
    return cells[AgentSlot(src_agent)][AgentSlot(dst_agent)];
  }
};

// ===========================================================================
// Placement policy
//
// The table answers which engines a pair prefers. This answers which one a
// given piece of a decomposed batch actually gets, once preference, load and
// the caller's caps are all taken into account. Both producers share it — the
// host planner in abce_host.h and the device planner in abce_device.h — so a
// batch is placed identically whichever side submits it.
// ===========================================================================

/// Pieces one engine takes before placement moves to the next. Only
/// kDepthFirst consults it: a breadth-first fan-out spreads long before it
/// would ever stack this deep.
constexpr uint32_t kMaxPiecesPerEngine = 8;

/// How placement fills the permitted engines.
enum class FillOrder : uint8_t {
  /// Every permitted engine takes a piece before any takes a second; extras
  /// then go to the least loaded. What a fan-out wants — a broadcast to N peers
  /// reaches N engines instead of serializing behind one.
  kBreadthFirst,
  /// Fill one engine to kMaxPiecesPerEngine before moving to the next. What a
  /// back-to-back batch wants: a short batch stays on a single ring, and so in
  /// a single frame with no coordinator to pay for.
  kDepthFirst,
};

/// @brief Placement state threaded through one decomposition.
struct EnginePlacement {
  /// Engines the caller permits (bit i => registered engine index i).
  uint64_t candidates = 0;
  /// Distinct engines this batch may spread over; 0 means no cap.
  uint32_t max_participants = 0;
  FillOrder order = FillOrder::kBreadthFirst;

  uint64_t selected_mask = 0;  ///< engines already carrying a piece.

  /// Pieces placed per engine, 4 bits each. Packed so placement holds no array:
  /// an array indexed by a runtime engine number is what forces a kernel into
  /// scratch. Saturates at 15, which cannot mislead the cap (8) or the balancer,
  /// since reaching 15 needs a batch with no choice of engine left.
  uint64_t load = 0;
};

static_assert(kMaxEngineChoices <= 16, "EnginePlacement::load packs 4 bits per engine");
static_assert(kMaxPiecesPerEngine < 15, "depth cap must be distinguishable from saturation");

ABCE_HD inline uint32_t LoadOf(uint64_t load, uint32_t engine) {
  return static_cast<uint32_t>((load >> (engine * 4)) & 0xf);
}

/// Least-loaded engine in @p allowed, preferring @p preferred on a tie so a
/// piece stays on its affinity engine until that engine is as busy as the rest.
ABCE_HD inline uint32_t LeastLoadedEngine(uint64_t allowed, uint32_t preferred, uint64_t load) {
  uint32_t min_load = UINT32_MAX;
  for (uint32_t engine = 0; engine < kMaxEngineChoices; ++engine)
    if ((allowed & (uint64_t{1} << engine)) && LoadOf(load, engine) < min_load)
      min_load = LoadOf(load, engine);
  if (min_load == UINT32_MAX) return kMaxEngineChoices;
  if (preferred < kMaxEngineChoices && (allowed & (uint64_t{1} << preferred)) &&
      LoadOf(load, preferred) == min_load)
    return preferred;
  for (uint32_t engine = 0; engine < kMaxEngineChoices; ++engine)
    if ((allowed & (uint64_t{1} << engine)) && LoadOf(load, engine) == min_load) return engine;
  return kMaxEngineChoices;
}

/// @brief Engine for one piece, or kMaxEngineChoices when none is permitted.
///
/// @p preferred is the pair's affinity anchor and @p forbidden the engines that
/// must never serve it, both straight out of the pair's EngineOrder.
///
/// The depth cap is SOFT: once every permitted engine is full, pieces keep
/// landing on the least loaded rather than failing. A deep ring is slow, not
/// wrong, and a legal engine does still exist.
ABCE_HD inline uint32_t PlacePiece(EnginePlacement& placement, uint32_t preferred,
                                   uint64_t forbidden) {
  // forbidden is a hard rule, not a preference — it is how a part's engine
  // reservations (gfx90a's RAS band) are expressed, so nothing below may widen
  // past it. An empty result means the pair genuinely has no legal engine.
  const uint64_t allowed = placement.candidates & ~forbidden;
  if (allowed == 0) return kMaxEngineChoices;

  uint64_t under_cap = 0;
  for (uint32_t engine = 0; engine < kMaxEngineChoices; ++engine) {
    const uint64_t bit = uint64_t{1} << engine;
    if ((allowed & bit) && LoadOf(placement.load, engine) < kMaxPiecesPerEngine)
      under_cap |= bit;
  }
  const uint64_t pool = under_cap != 0 ? under_cap : allowed;

  const uint64_t started = pool & placement.selected_mask;
  const uint64_t untouched = pool & ~placement.selected_mask;
  const bool room_for_more =
      placement.max_participants == 0 ||
      static_cast<uint32_t>(detail::PopCount64(placement.selected_mask)) <
          placement.max_participants;

  uint64_t target;
  if (placement.order == FillOrder::kDepthFirst)
    // Stay on an engine already in use until the cap retires it from the pool.
    target = started != 0 ? started : (room_for_more ? untouched : pool);
  else
    // Claim an unused engine while the participant cap allows, then balance.
    target = (room_for_more && untouched != 0) ? untouched : (started != 0 ? started : pool);
  if (target == 0) target = pool;

  const uint32_t engine = LeastLoadedEngine(target, preferred, placement.load);
  if (engine < kMaxEngineChoices) {
    placement.selected_mask |= uint64_t{1} << engine;
    if (LoadOf(placement.load, engine) < 0xf) placement.load += uint64_t{1} << (engine * 4);
  }
  return engine;
}

// ===========================================================================
// Per-part host-copy profiles
// ===========================================================================

/// @brief Measured host-copy engine order for one part, in HARDWARE SDMA engine
/// ids, best first.
///
/// Host copies are the one band whose good order cannot be derived from
/// anything the driver reports: it is measured. Parts are distinguished by
/// (gfx minor, total SDMA engines) because the same IP ships with different
/// engine counts and genuinely different orders.
struct HostCopyProfile {
  uint8_t minor;
  uint8_t total_sdma;
  uint8_t h2d[kMaxEngineChoices];
  uint8_t h2d_count;
  uint8_t d2h[kMaxEngineChoices];
  uint8_t d2h_count;
  /// Engines this part must never use outside the H2D band, as a hardware-id
  /// mask. gfx90a's SDMA0 RAS reservation is the only current user.
  uint64_t h2d_only_hw_mask;
  /// Whether the H2D and D2H orders are exclusive: no engine outside them may
  /// serve that direction. Only gfx90a needs this; elsewhere the order is a
  /// ranking and the tail stays legal.
  bool host_bands_exclusive;
};

/// Profile for @p minor / @p total_sdma, or nullptr when the part has none —
/// which leaves every host cell empty and selection on pure load balancing.
inline const HostCopyProfile* SelectHostCopyProfile(uint8_t minor, uint8_t total_sdma) {
  // clang-format off
  static const HostCopyProfile kProfiles[] = {
    // Measured through ROCr's copy_on_engine selector. On gfx94x/95x ROCr swaps
    // its two host blit queues: selector 0 targets physical SDMA1 and selector 1
    // targets physical SDMA0. ABCE queues are targeted by physical id directly,
    // so the orders below have selector ids 0/1 swapped.
    //
    // gfx942, 16 SDMA (MI300X): physical engines 8-15 are poor for D2H
    // (~20 GB/s), while H2D peak bandwidth is ~flat (~55). Same tier pattern for
    // both directions, deferring physical engines 0 and 3, which have a large
    // medium-copy penalty through ABCE.
    {4, 16, {2, 4, 6, 5, 1, 7, 0, 3}, 8, {2, 4, 6, 5, 1, 7, 0, 3}, 8, 0, false},
    // gfx942, 8 SDMA (MI308X): only selector 0 (physical SDMA1) is slow for D2H
    // (~48 vs ~56 on the rest), while H2D is flat (~55). Keep physical SDMA1 last.
    {4,  8, {0, 3, 4, 5, 2, 7, 6, 1}, 8, {0, 3, 4, 5, 2, 7, 6, 1}, 8, 0, false},
    // gfx950, 16 SDMA (MI350X): selectors 0-3 are the fast tier for both
    // directions (~56). Selectors 4-7 are the second H2D tier (~51) but fall to
    // ~13 for D2H; selectors 8-15 are slower still and are omitted.
    {5, 16, {1, 0, 2, 3, 4, 5, 6, 7}, 8, {1, 0, 2, 3, 4, 5, 6, 7}, 8, 0, false},
    // gfx90a (MI200): a RAS issue means SDMA0 may drive H2D copies and nothing
    // else, so it is forbidden outside that band. The bands are exclusive here
    // because this is a hardware workaround rather than a bandwidth ranking:
    // widening it is a hardware question, not a tuning one.
    {0,  0, {0}, 1, {1}, 1, uint64_t{1} << 0, true},
  };
  // clang-format on
  for (const auto& profile : kProfiles)
    if (profile.minor == minor && profile.total_sdma == total_sdma) return &profile;
  return nullptr;
}

/// Profile for an ISA, or nullptr. gfx90a is keyed on stepping rather than an
/// engine count, and gfx12+ intentionally has none: its engines are symmetric,
/// so "no opinion" is not a gap but the correct answer.
inline const HostCopyProfile* SelectHostCopyProfile(const IsaVersion& isa, uint32_t total_sdma) {
  if (isa.major != 9) return nullptr;
  if (isa.minor == 0 && isa.stepping == 10) return SelectHostCopyProfile(0, 0);
  if (isa.minor == 4 || isa.minor == 5)
    return SelectHostCopyProfile(static_cast<uint8_t>(isa.minor),
                                 static_cast<uint8_t>(total_sdma));
  return nullptr;
}

// ===========================================================================
// Formation
// ===========================================================================

/// @brief Builds an EngineTable from a part profile plus whatever the driver
/// says about individual links.
///
/// Usage: construct, RegisterEngine() once per ring, SetPeerRecommendation()
/// for each link the driver describes, then Build(). Everything is in hardware
/// engine ids on the way in and registered indices on the way out.
class EngineTableBuilder {
 public:
  EngineTableBuilder(const IsaVersion& isa, uint32_t total_sdma)
      : profile_(SelectHostCopyProfile(isa, total_sdma)) {
    for (auto& entry : hw_to_registered_) entry = kUnmapped;
  }

  /// Associate a registered index with its hardware SDMA engine id.
  void RegisterEngine(uint32_t index, uint32_t hw_engine_id) {
    if (index >= kMaxEngineChoices || hw_engine_id >= kMaxHwEngines) return;
    if (hw_to_registered_[hw_engine_id] != kUnmapped) return;
    hw_to_registered_[hw_engine_id] = static_cast<uint8_t>(index);
    registered_mask_ |= uint64_t{1} << index;
  }

  /// The driver's recommended engines for one ordered agent pair, as a mask of
  /// HARDWARE engine ids (KFD's recommended_sdma_engine_id_mask). A pair with no
  /// recommendation is left alone; do not call, or pass 0.
  void SetPeerRecommendation(int32_t src_agent, int32_t dst_agent, uint64_t hw_mask) {
    const uint32_t src_slot = AgentSlot(src_agent);
    const uint32_t dst_slot = AgentSlot(dst_agent);
    if (src_slot >= kTableSlots - 1 || dst_slot >= kTableSlots - 1) return;
    peer_hw_mask_[src_slot][dst_slot] = hw_mask;
  }

  EngineTable Build() const {
    EngineTable table;
    for (uint32_t src_slot = 0; src_slot < kTableSlots; ++src_slot) {
      for (uint32_t dst_slot = 0; dst_slot < kTableSlots; ++dst_slot) {
        // The trailing slot is the "unknown agent" absorber and stays empty, as
        // does host-to-host, which ABCE rejects before selection.
        const bool src_unknown = src_slot == kTableSlots - 1;
        const bool dst_unknown = dst_slot == kTableSlots - 1;
        if (src_unknown || dst_unknown) continue;
        const bool src_host = src_slot == 0;
        const bool dst_host = dst_slot == 0;
        if (src_host && dst_host) continue;

        EngineOrder& cell = table.cells[src_slot][dst_slot];
        if (src_host || dst_host) {
          FillHostCell(cell, /*to_device=*/src_host);
        } else {
          FillDeviceCell(cell, peer_hw_mask_[src_slot][dst_slot]);
        }
      }
    }
    return table;
  }

 private:
  static constexpr uint32_t kMaxHwEngines = 64;
  static constexpr uint8_t kUnmapped = 0xFF;

  /// Translate a hardware-id order into registered indices, dropping ids no ring
  /// was registered for and de-duplicating.
  void AppendHwOrder(EngineOrder& cell, const uint8_t* hw_ids, uint32_t count) const {
    uint64_t emitted = 0;
    for (uint32_t rank = 0; rank < count && cell.num_ranked < kMaxEngineChoices; ++rank) {
      const uint8_t hw_id = hw_ids[rank];
      if (hw_id >= kMaxHwEngines) continue;
      const uint8_t index = hw_to_registered_[hw_id];
      if (index == kUnmapped) continue;
      const uint64_t bit = uint64_t{1} << index;
      if (emitted & bit) continue;
      emitted |= bit;
      cell.ranked[cell.num_ranked++] = index;
    }
  }

  /// Registered indices matching a hardware-id mask, ascending.
  uint64_t TranslateHwMask(uint64_t hw_mask) const {
    uint64_t registered = 0;
    for (uint32_t hw_id = 0; hw_id < kMaxHwEngines; ++hw_id) {
      if (!(hw_mask & (uint64_t{1} << hw_id))) continue;
      const uint8_t index = hw_to_registered_[hw_id];
      if (index != kUnmapped) registered |= uint64_t{1} << index;
    }
    return registered;
  }

  void FillHostCell(EngineOrder& cell, bool to_device) const {
    if (!profile_) return;
    const uint8_t* order = to_device ? profile_->h2d : profile_->d2h;
    const uint32_t count = to_device ? profile_->h2d_count : profile_->d2h_count;
    AppendHwOrder(cell, order, count);
    // An engine restricted to H2D may not serve the reverse direction. Applied
    // unconditionally: this and an exclusive band say different things and
    // compose, so the restriction must not depend on the band also being set.
    if (!to_device) cell.forbidden_mask |= TranslateHwMask(profile_->h2d_only_hw_mask);
    if (profile_->host_bands_exclusive) {
      // Everything outside the band is forbidden for this direction.
      uint64_t allowed = 0;
      for (uint32_t rank = 0; rank < cell.num_ranked; ++rank)
        allowed |= uint64_t{1} << cell.ranked[rank];
      cell.forbidden_mask |= registered_mask_ & ~allowed;
    }
  }

  void FillDeviceCell(EngineOrder& cell, uint64_t peer_hw_mask) const {
    // The driver's recommendation, when it has one. Ascending within the mask:
    // it names the engines with affinity to this link, not an order among them.
    const uint64_t recommended = TranslateHwMask(peer_hw_mask);
    for (uint32_t index = 0; index < kMaxEngineChoices && cell.num_ranked < kMaxEngineChoices;
         ++index)
      if (recommended & (uint64_t{1} << index)) cell.ranked[cell.num_ranked++] = index;

    if (profile_) cell.forbidden_mask = TranslateHwMask(profile_->h2d_only_hw_mask);
  }

  const HostCopyProfile* profile_ = nullptr;
  uint8_t hw_to_registered_[kMaxHwEngines]{};
  uint64_t registered_mask_ = 0;
  uint64_t peer_hw_mask_[kTableSlots][kTableSlots]{};
};

}  // namespace abce

#endif  // ABCE_ENGINE_TABLE_H_
