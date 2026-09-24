// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// \file
/// Abstract state and transfer operations for forward waitcheck dataflow.
///
/// A PendingState combines possible pending events/hazards (may facts) with
/// guaranteed counter progress and available register generations (must facts).
/// States are ordered by loss of information: a less precise state may contain
/// more hazards, smaller lower bounds on progress, and fewer readiness facts.
/// For a CFG block B, the intended dataflow equations are
///   IN[B] = join(OUT[P] for each reachable predecessor P),
///   OUT[B] = transfer_B(IN[B]),
/// with an explicit initial state at the entry. The driver owns instruction
/// issue/aging, block traversal and fixed-point iteration; this file supplies
/// joins and wait transfers independently of traversal and diagnostic emission.
///
/// merge_predecessors distinguishes unvisited outputs (bottom, tracked by
/// output_initialized) from an initialized state with no pending events. The
/// latter is a real path and must participate in intersections of must facts.
/// merge_into joins initialized states as follows:
/// - Pending event identities and possible hazards are unioned. Events are
///   sorted by static identity for stable equality, not by hardware issue order.
/// - min_younger is a lower bound on younger requests on that event's counter.
///   Matching events join by minimum age; absence on a path imposes no bound.
///   Per-kind ages similarly summarize the newest possibly pending request;
///   they include counter-only requests and are not additional event counts.
/// - ready_regs and each matching event's old_value_regs are intersected.
///   Readiness describes an available committed generation, not completion of
///   every outstanding producer of that register.
/// - Event presence on every incoming path is a must fact: matching events
///   intersect it, and an absent event clears it. For paths missing the event,
///   ready_on_absent_paths intersects their already-committed register facts.
///   Retirement may establish readiness only with pending-or-ready coverage of
///   every path. Different uncompleted producers on different paths still cannot
///   establish readiness after the join.
/// - Conflicting mode values become unknown; possible scalar-memory presence
///   and uncertain ordering are ORed. Delayed clears survive only if guaranteed
///   on both paths, with the later of their earliest guaranteed clear times.
///
/// Wait transfers refine these facts using guaranteed completion: they remove
/// possibly pending events once the wait proves retirement, and establish must
/// readiness only when every path has an available committed value.
/// For an ordered counter, threshold n proves retirement at min_younger >= n;
/// zero proves completion of the whole counter. Architecture-aware transfers
/// withhold partial retirement when completion may be out of order and propagate
/// guarantees to related counters. See PendingState and PendingEvent for the
/// representation and make_retired_generations_ready for generation handling.

#include "rocjitsu/code/analysis/waitcheck/target.h"
#include "rocjitsu/isa/arch/amdgpu/shared/vgpr_msb.h"

#include <array>
#include <bitset>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocjitsu::waitcheck_detail {

/// Lower bounds on the age of the newest possibly pending request of each kind.
/// kNoPendingEventAge denotes absence. These summaries include counter-only
/// requests and may overlap materialized PendingEvents; they are not token counts.
/// Byte storage keeps the dense per-counter summaries compact. Ages saturate at
/// the target dependency-wait limit, below the reserved absence value.
struct PendingEventAges {
  std::array<uint8_t, kWaitEventKindCount> values = [] {
    std::array<uint8_t, kWaitEventKindCount> result;
    result.fill(kNoPendingEventAge);
    return result;
  }();

  bool operator==(const PendingEventAges &) const = default;
};

inline constexpr uint8_t kVgprLow16Mask = 0x1;
inline constexpr uint8_t kVgprHigh16Mask = 0x2;
inline constexpr uint8_t kVgprFull32Mask = kVgprLow16Mask | kVgprHigh16Mask;

/// One physical register access, with a mask selecting its low/high 16-bit halves.
/// Full-lane accesses use kVgprFull32Mask; partial accesses select the relevant half.
struct PartialRegisterAccess {
  RegisterRef reg;
  uint8_t mask = kVgprFull32Mask;
};

/// A possibly outstanding operation on one wait counter, identified by its static
/// instruction location, event kind, register payload and hazard-check flags.
/// One instruction can have multiple counter facets with the same generation.
/// regs describes affected lanes; produces_regs distinguishes pending destinations
/// from source-use hazards. Age, path coverage and available older values are
/// dataflow facts, excluded from the static identity. PendingState stores events
/// sorted by that identity, with at most one event for each identity per counter.
struct PendingEvent {
  WaitCounterKind counter = WaitCounterKind::Load;
  WaitEventKind kind = WaitEventKind::Unknown;
  RegisterSet regs;
  // Subset of regs with a committed value available while this producer remains
  // pending. A consumer that permits older generations can exclude these lanes
  // from this event's use hazards; this does not prove the new result completed.
  RegisterSet old_value_regs;
  // LLVM tracks the low/high 16-bit physical subregisters used by D16 memory
  // operations independently. Keep the exceptional partial destination
  // sparse: almost every event still covers whole 32-bit register lanes.
  std::optional<RegisterRef> partial_reg;
  uint8_t partial_reg_mask = kVgprFull32Mask;
  std::optional<RegisterRef> special_reg;
  std::optional<int64_t> barrier_id;
  bool produces_regs = false;
  // A newly issued event is pending on every path reaching that instruction.
  // Joins clear this must fact when any incoming path lacks the event. It is
  // state information, not part of the static event identity.
  bool present_on_all_paths = true;
  // Subset of regs already ready on every path where this event is absent.
  // Empty when present_on_all_paths is true (there are no absent paths). When
  // false, retirement can establish readiness only for these covered lanes.
  // The available value need not be from this producer: readiness promises an
  // available committed generation, not the identity of the latest write.
  RegisterSet ready_on_absent_paths;
  bool check_uses = true;
  bool check_defs = true;
  bool check_exec_defs = false;
  std::string section_name;
  uint64_t section_offset = 0;
  uint64_t file_offset = 0;
  std::string instruction;
  bool check_memory_order = false;
  bool check_program_end = false;
  bool check_counter_parity_order = false;
  // Lower bound on younger requests issued on this counter, independent of the
  // event's canonical vector position. Matching events join by minimum. Use the
  // wait-threshold width for arithmetic; only dense per-kind summaries are packed.
  uint32_t min_younger = 0;

  bool operator==(const PendingEvent &) const = default;
};

/// A representative instruction location used to explain a scheduling hazard.
/// Joins can retain either incoming witness for the same hazard; the chosen
/// location affects diagnostics, not the hazard's completion guarantee.
struct SgprHazardProducer {
  std::string section_name;
  uint64_t section_offset = 0;
  uint64_t file_offset = 0;
  std::string instruction;

  bool operator==(const SgprHazardProducer &) const = default;
};

constexpr uint8_t kSgprHazardSalu = 1u << 0u;
constexpr uint8_t kSgprHazardValu = 1u << 1u;

/// Scalar-register scheduling hazards and progress toward clearing them.
/// tracked_pairs/tracked_vcc record scalar sources whose later writes may need
/// a dependency wait; SALU
/// and VALU hazard bits select lanes requiring the corresponding wait.
/// Sparse producer maps retain diagnostic witnesses. Joins union possible hazards
/// and take the minimum guaranteed consecutive DS-NOP progress.
struct SgprHazardState {
  std::bitset<64> tracked_pairs;
  bool tracked_vcc = false;
  std::bitset<128> salu_hazards;
  std::bitset<128> valu_hazards;
  uint8_t vcc_hazard = 0;
  std::unordered_map<uint16_t, SgprHazardProducer> salu_producers;
  std::unordered_map<uint16_t, SgprHazardProducer> valu_producers;
  std::optional<SgprHazardProducer> salu_vcc_producer;
  std::optional<SgprHazardProducer> valu_vcc_producer;
  uint8_t consecutive_ds_nops = 0;

  bool operator==(const SgprHazardState &) const = default;
};

/// One destination hazard tracked by VA_VDST age, with a diagnostic witness.
/// age is guaranteed progress; trans_since records a possible intervening
/// transcendental operation, which makes a nonzero wait insufficient.
struct VaVdstHazard {
  uint8_t age = 0;
  bool trans_since = false;
  SgprHazardProducer producer;

  bool operator==(const VaVdstHazard &) const = default;
};

/// Sparse per-VGPR VA_VDST hazards. Absence means no tracked hazard for that lane;
/// joins retain hazards from either path and the least guaranteed progress.
struct VaVdstHazardState {
  std::unordered_map<uint16_t, VaVdstHazard> hazards;

  bool operator==(const VaVdstHazardState &) const = default;
};

/// Packed VGPR-bank selection for the four operand roles. known=false means
/// incoming modes disagree; the driver must handle that uncertainty before using
/// for_role, which extracts bits from a known mode.
struct VgprMsbState {
  uint8_t mode = 0;
  bool known = true;

  [[nodiscard]] uint32_t for_role(amdgpu::VgprMsbRole role) const;

  bool operator==(const VgprMsbState &) const = default;
};

/// The dependency domain cleared when a delayed ALU wait takes effect.
enum class DelayAluEffect : uint8_t {
  None,
  Valu,
  Salu,
};

/// A scheduled dependency clear that becomes effective after countdown steps.
/// The driver advances the countdown; a join retains an effect only when every
/// incoming path guarantees it, at the latest guaranteed clear time.
struct PendingDelayAlu {
  uint8_t countdown = 0;
  DelayAluEffect effect = DelayAluEffect::None;

  bool operator==(const PendingDelayAlu &) const = default;
};

/// Whether instructions use expert scheduling. Conflicting incoming values set
/// known=false and conservatively retain enabled=true until the mode is resolved.
struct ExpertSchedulingState {
  bool enabled = false;
  bool known = true;

  bool operator==(const ExpertSchedulingState &) const = default;
};

/// Dataflow facts at an instruction or block boundary on an initialized path.
/// Counter-indexed events/ages and auxiliary hazards describe possible pending
/// work; readiness, progress and known modes describe guarantees. The default
/// state has no pending work or readiness proofs. It is a real empty path, not
/// the unvisited bottom state, which the driver records separately.
struct PendingState {
  // Vectors stay sorted by static event identity so CFG equality is stable.
  // min_younger, rather than vector position, represents hardware issue order.
  std::array<std::vector<PendingEvent>, kCounterCount> pending;
  // LLVM keeps the newest score for every hardware-event kind, including
  // counter-only operations with no register or ordering payload.  Their
  // presence determines whether a counter may retire out of order.  Store the
  // equivalent age rather than materializing one PendingEvent per token.
  std::array<PendingEventAges, kCounterCount> pending_event_ages;
  // Keep scalar-memory presence even for counter-only requests that do not
  // need a full PendingEvent. Scalar memory makes its counter out of order.
  std::array<bool, kCounterCount> pending_smem{};
  std::array<bool, kCounterCount> uncertain_order{};
  // Lanes with a committed generation available on every incoming path. A later
  // producer may still be pending; when it is issued, this fact seeds that
  // event's old_value_regs. This is not a promise that the latest write is done.
  RegisterSet ready_regs;
  SgprHazardState sgpr_hazards;
  VaVdstHazardState va_vdst_hazards;
  VgprMsbState vgpr_msb;
  bool vgpr_msb_setreg_hazard = false;
  bool previous_vm_vsrc_zero_wait = false;
  std::optional<SgprHazardProducer> async_barrier_post_wait;
  std::vector<PendingDelayAlu> delay_alu;
  ExpertSchedulingState expert_scheduling;

  bool operator==(const PendingState &) const = default;
};

static_assert(sizeof(PendingState) < 4096,
              "waitcheck CFG states must keep architecture-specific hazard storage sparse");

/// Stateless joins and wait transfers shared by straight-line and CFG drivers.
/// Inputs must maintain PendingState's canonical event order and counter facts.
/// The driver owns issue/aging and traversal; these operations update supplied
/// states without emitting diagnostics. Architecture-aware transfers report
/// unsupported targets through Result/FailureOr.
struct WaitcheckStateOps : WaitcheckTarget {
  [[nodiscard]] static bool same_event_identity(const PendingEvent &lhs, const PendingEvent &rhs);

  [[nodiscard]] static bool register_set_less(const RegisterSet &lhs, const RegisterSet &rhs);

  [[nodiscard]] static bool event_identity_less(const PendingEvent &lhs, const PendingEvent &rhs);

  static void merge_va_vdst_hazards(VaVdstHazardState &dst, const VaVdstHazardState &src);

  static void merge_delay_alu(std::vector<PendingDelayAlu> &dst,
                              const std::vector<PendingDelayAlu> &src);

  // Join two initialized states. An unvisited predecessor is not an empty state.
  static void merge_into(PendingState &dst, const PendingState &src);

  // Skip valid unvisited predecessors; fail if any index is outside either
  // input array. An invalid edge must not silently drop hazards or must facts.
  [[nodiscard]] static util::FailureOr<PendingState>
  merge_predecessors(std::span<const size_t> predecessors, const std::vector<PendingState> &outputs,
                     std::span<const uint8_t> output_initialized);

  [[nodiscard]] static bool same_register_generation(const PendingEvent &lhs,
                                                     const PendingEvent &rhs);

  // Retirement proves readiness when every path either has this producer or
  // already has a committed value, and all facets of this generation retire. A
  // different, still-pending producer does not erase this older value: record it in both
  // ready_regs and that producer's old_value_regs for consumers permitting it.
  static void make_retired_generations_ready(PendingState &state,
                                             std::span<const PendingEvent> retired_events);

  static void apply_wait_to_event_ages(PendingState &state, WaitCounterKind counter,
                                       uint32_t count);

  static void apply_wait(PendingState &state, WaitCounterKind counter, uint32_t count);

  static util::Result apply_kmcnt_wait(PendingState &state, uint32_t count, rj_code_arch_t arch);

  static void apply_implied_vm_vsrc_wait(PendingState &state, WaitCounterKind counter,
                                         uint32_t count);

  [[nodiscard]] static bool is_xcnt_smem_event(const PendingEvent &event);

  [[nodiscard]] static bool is_xcnt_vmem_event(const PendingEvent &event);

  [[nodiscard]] static bool is_xcnt_store_event(const PendingEvent &event);

  static void apply_xcnt_wait(PendingState &state, uint32_t count);

  static void apply_xcnt_wait_implied_by_kmcnt(PendingState &state, uint32_t count);

  static void apply_xcnt_wait_implied_by_loadcnt(PendingState &state, uint32_t count);

  [[nodiscard]] static bool scalar_memory_makes_counter_out_of_order(const PendingState &state,
                                                                     WaitCounterKind counter,
                                                                     WaitcntModel model);

  [[nodiscard]] static bool counter_has_event_kind(const PendingState &state,
                                                   WaitCounterKind counter, WaitEventKind kind);

  [[nodiscard]] static bool flat_memory_makes_counter_out_of_order(const PendingState &state,
                                                                   WaitCounterKind counter,
                                                                   rj_code_arch_t arch);

  [[nodiscard]] static util::FailureOr<bool>
  counter_out_of_order(const PendingState &state, WaitCounterKind counter, rj_code_arch_t arch);

  static util::Result apply_counter_wait(PendingState &state, WaitCounterKind counter,
                                         uint32_t count, rj_code_arch_t arch);

  [[nodiscard]] static util::FailureOr<uint32_t>
  dependency_required_count(const PendingState &state, const PendingEvent &event,
                            rj_code_arch_t arch);

  static util::Result apply_memory_wait(PendingState &state, WaitCounterKind counter,
                                        uint32_t count, rj_code_arch_t arch);

  static void clear_salu_sgpr_hazards(SgprHazardState &state);

  static void clear_valu_sgpr_hazards(SgprHazardState &state);

  static void clear_valu_vcc_hazard(SgprHazardState &state);

  static void merge_lane_producers(std::unordered_map<uint16_t, SgprHazardProducer> &dst,
                                   const std::unordered_map<uint16_t, SgprHazardProducer> &src);

  static void merge_sgpr_hazards(SgprHazardState &dst, const SgprHazardState &src);

  static void apply_sgpr_hazard_wait(SgprHazardState &state, uint32_t depctr);

  static util::Result apply_wait_fields(PendingState &state, const WaitFields &fields,
                                        rj_code_arch_t arch);

  static util::Result apply_embedded_waitcnt(PendingState &state, const Instruction &inst,
                                             rj_code_arch_t arch);

  static util::Result apply_waitcnt(PendingState &state, const Instruction &inst,
                                    rj_code_arch_t arch);

private:
  // Counter facets of one static operation can track different registers and
  // hazard flags. Location, instruction and event kind identify their operation.
  [[nodiscard]] static bool same_operation(const PendingEvent &lhs, const PendingEvent &rhs);

  [[nodiscard]] static auto register_ref_key(const std::optional<RegisterRef> &ref);

  template <typename Predicate>
  static void retire_events(PendingState &state, std::vector<PendingEvent> &events,
                            Predicate should_retire);

  template <typename Predicate>
  static void retire_event_kind_ages(PendingState &state, WaitCounterKind counter,
                                     Predicate belongs_to_group, uint32_t minimum_age = 0);

  template <typename Predicate>
  static void apply_filtered_vm_vsrc_wait(PendingState &state, Predicate is_implied,
                                          uint32_t count);

  template <typename Predicate>
  [[nodiscard]] static bool has_xcnt_event(const PendingState &state, Predicate predicate);

  template <typename Predicate>
  static void retire_xcnt_group(PendingState &state, Predicate belongs_to_group);
};

} // namespace rocjitsu::waitcheck_detail
