// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// \file
/// Abstract state and transfer operations for forward waitcheck dataflow.
///
/// A PendingState combines possible pending events/hazards (may facts) with
/// guaranteed counter progress and operation coverage (must facts).
/// States are ordered by loss of information: a less precise state may contain
/// more hazards, smaller lower bounds on progress, and fewer coverage facts.
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
/// - Event presence on every incoming path is a must fact: matching events
///   intersect it, and an absent event clears it. Implied cross-counter waits
///   use this coverage when proving completion of another facet of an operation.
/// - Conflicting mode values become unknown; possible scalar-memory presence
///   and uncertain ordering are ORed. Delayed clears survive only if guaranteed
///   on both paths, with the later of their earliest guaranteed clear times.
///
/// Wait transfers refine these facts using guaranteed completion: they remove
/// possibly pending events once the wait proves retirement. A register remains
/// unavailable while any conflicting physical access is still pending.
/// For a completion FIFO, threshold n proves retirement at min_younger >= n;
/// zero proves completion of the whole counter. Unordered events have no younger
/// FIFO credits. Admission and waits use those lower bounds even across mixed
/// queues and CFG joins. See PendingState and PendingEvent for the
/// representation.

#include "rocjitsu/code/analysis/waitcheck/target.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_selectors.h"
#include "rocjitsu/isa/arch/amdgpu/shared/vgpr_msb.h"

#include <array>
#include <bitset>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocjitsu::waitcheck_detail {

/// Lower bounds on the age of the newest possibly pending request of each kind.
/// kNoPendingEventAge denotes absence. These summaries include counter-only
/// requests and may overlap materialized PendingEvents; they are not token counts.
/// Byte storage keeps the dense per-counter summaries compact. Ages stay below
/// the reserved absence value and may exceed a dependency wait's encodable range
/// when another counter can apply a wider implied wait.
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
/// One instruction can have multiple counter facets. regs describes the affected
/// physical registers. Age and path coverage are dataflow facts, excluded from
/// static identity. PendingState stores events sorted by that identity, with at
/// most one event for each identity per counter.
struct PendingEvent {
  RegisterSet regs;
  // Subset of regs with a committed value available while this producer remains
  // pending. A consumer that permits older generations can exclude these lanes
  // from this event's use hazards; this does not prove the new result completed.
  RegisterSet old_value_regs;
  // Subset of regs already ready on every path where this event is absent.
  // Empty when present_on_all_paths is true (there are no absent paths). When
  // false, retirement can establish readiness only for these covered lanes.
  // The available value need not be from this producer: readiness promises an
  // available committed generation, not the identity of the latest write.
  RegisterSet ready_on_absent_paths;
  // RegisterSet treats VCC as a singleton. Keep the actual physical dwords
  // alongside it for memory-result and replay operand overlap checks.
  uint8_t vcc_halves = 0;
  // LLVM tracks the low/high 16-bit physical subregisters used by D16 memory
  // operations independently. Keep the exceptional partial destination
  // sparse: almost every event still covers whole 32-bit register lanes.
  std::optional<RegisterRef> partial_reg;
  uint8_t partial_reg_mask = kVgprFull32Mask;
  std::optional<RegisterRef> special_reg;
  std::optional<int64_t> barrier_id;
  // A newly issued event is pending on every path reaching that instruction.
  // Joins clear this must fact when any incoming path lacks the event. It is
  // state information, not part of the static event identity.
  bool present_on_all_paths = true;
  // If this GFX11 LDS result or expert memory-source read is pending,
  // its issuing EXEC was nonempty until the mask is overwritten.
  bool implies_active_exec = false;
  bool crossed_exec_expansion = false;
  // While this source is still live, a younger active same-FIFO source
  // cannot have completed its primary operation before reading its operands.
  uint8_t younger_source_primaries = 0;
  bool check_uses = true;
  bool check_defs = true;
  bool check_exec_defs = false;
  std::string section_name;
  uint64_t section_offset = 0;
  uint64_t file_offset = 0;
  std::string instruction;
  WaitEventKind kind = WaitEventKind::Unknown;
  // Lower bound on younger requests issued on this counter, independent of the
  // event's canonical vector position. Matching events join by minimum. Use the
  // wait-threshold width for arithmetic; only dense per-kind summaries are packed.
  uint32_t min_younger = 0;
  WaitCounterKind counter = WaitCounterKind::Load;
  bool produces_regs = false;
  bool check_memory_order = false;
  bool check_program_end = false;
  bool check_counter_parity_order = false;
  // Guaranteed younger requests in this event's own typed EXPCNT FIFO.
  // Other export types can overtake it and contribute no progress proof.
  uint8_t min_younger_in_fifo = 0;
  // For an XCNT or dedicated VM_VSRC store source, guaranteed younger primary FIFO credits.
  // For FLAT, count only same-family credits guaranteed on both possible routes.
  uint8_t min_younger_primary = 0;
  // For CDNA5 WMMA, an explicitly retired core VALU beyond this instruction
  // window proves the matrix register access has finished across pipelines.
  uint8_t wmma_wait_states = 0;
  uint8_t min_younger_core = 0;

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

  auto operator<=>(const SgprHazardProducer &) const = default;
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

/// A possible VA_VDST producer, ordered by least VALU progress then stable location.
struct VaVdstHazard {
  uint8_t age = 0;
  SgprHazardProducer producer;

  auto operator<=>(const VaVdstHazard &) const = default;
};

// RDNA3 TRANS results can feed a VALU for five intervening VALUs and one
// intervening TRANS. Separate the two TRANS counts at CFG joins so ages
// from different paths cannot manufacture a qualifying pair of distances.
struct TransUseHazard {
  uint8_t valu_age = 0;
  SgprHazardProducer producer;

  bool operator==(const TransUseHazard &) const = default;
};
// Keyed by the physical 16-bit VGPR half and intervening TRANS count.
using TransUseHazards = std::map<std::pair<uint16_t, uint8_t>, TransUseHazard>;

/// Both paths expire at age 15. A path containing a TRANS needs a zero wait
/// until then. Keep its age and witness separate from an ordinary path.
struct VaVdstHazardState {
  using Key = std::pair<uint16_t, bool>; // VGPR and whether a TRANS has issued.
  std::map<Key, VaVdstHazard> hazards;

  void merge_hazard(Key key, const VaVdstHazard &hazard);
  [[nodiscard]] std::pair<const VaVdstHazard *, uint32_t> requirement(uint16_t reg) const;
  [[nodiscard]] bool contains_producer(uint16_t reg, const SgprHazardProducer &producer) const;

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

/// Possible values of a scalar source for MOVREL. An empty set means unknown.
/// Once a join exceeds the bound, retain every possible register alias.
struct ScalarOffsetValues {
  std::array<uint32_t, 16> literals{};
  unsigned count = 0;

  [[nodiscard]] std::span<const uint32_t> values() const { return {literals.data(), count}; }
  void merge(const ScalarOffsetValues &other);
  bool operator==(const ScalarOffsetValues &) const = default;
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
  SgprHazardState sgpr_hazards;
  VaVdstHazardState va_vdst_hazards;
  TransUseHazards trans_use_hazards;
  VgprMsbState vgpr_msb;
  ScalarOffsetValues m0;
  std::unordered_map<uint16_t, ScalarOffsetValues> sgpr_offsets;
  std::optional<bool> prefetch_enabled;
  std::optional<bool> vector_skip = false;
  bool expert_exec_empty = false; ///< All lanes are known inactive for expert memory sources.
  // A live vector operand implies the producer had an active lane. The same
  // EXEC remains active until an intervening write can disable every lane.
  bool expert_valu_may_skip = false;
  bool expert_exec_nonzero = false;
  std::bitset<amdgpu::kVccSelectorLast + 1> expert_saved_nonzero_exec;
  bool exec_empty = false; ///< Every incoming path has a proven zero EXEC mask.
  bool exec_nonzero = false; ///< All incoming paths have at least one active lane.
  bool vgpr_msb_setreg_hazard = false;
  bool previous_vm_vsrc_zero_wait = false;
  std::optional<SgprHazardProducer> async_barrier_post_wait;
  std::vector<PendingDelayAlu> delay_alu;
  ExpertSchedulingState expert_scheduling;
  ExpertValuDelayState expert_delay;

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
  static bool tracks_d16_subregisters(rj_code_arch_t arch, uint32_t flags = 0, uint8_t hsa_abi = 0);
  static std::optional<PartialRegisterAccess> partial_d16_load_def(const Instruction &inst,
                                                                   bool tracks_halves);
  static void update_expert_valu_mask(PendingState &state, const Instruction &inst,
                                      const RegisterSet &defs, uint32_t wave_size,
                                      uint32_t allocated_sgprs, bool writes_exec);
  [[nodiscard]] static std::optional<bool> constant_exec_nonzero(const Instruction &inst,
                                                                 uint32_t wave_size);
  [[nodiscard]] static std::optional<uint32_t>
  track_issue_units(PendingState &state, const Instruction &inst, const ClassifiedEvent &event,
                    const RegisterSet &defs, rj_code_arch_t arch);
  [[nodiscard]] static bool preserves_exec_lanes(const Instruction &inst);
  [[nodiscard]] static bool vm_source_results_allocated(const Instruction &inst,
                                                        rj_code_arch_t arch,
                                                        const VgprMsbState &banks,
                                                        uint32_t min_vgprs);
  [[nodiscard]] static bool dedicated_source_issues(const Instruction &inst);
  [[nodiscard]] static uint8_t dedicated_source_primary(WaitEventKind kind);
  [[nodiscard]] static bool pending_source_forces_issue(WaitEventKind older, WaitEventKind younger,
                                                        bool empty_exec);
  static void age_events(PendingState &state, const PendingEvent &event, rj_code_arch_t arch,
                         uint32_t maximum_wait, uint32_t units = 1,
                         bool expanded_source_issues = false, bool results_allocated = true);

  [[nodiscard]] static bool is_completion_counter(WaitCounterKind counter);
  [[nodiscard]] static std::optional<uint32_t> guaranteed_issue_order(WaitCounterKind counter,
                                                                      WaitEventKind kind,
                                                                      rj_code_arch_t arch,
                                                                      bool exec_nonzero);
  static void apply_admission(PendingState &state, const Instruction &inst,
                              std::span<const ClassifiedEvent> events, rj_code_arch_t arch,
                              bool after_source_reads);
  [[nodiscard]] static bool ordered_waw(const PendingEvent &event,
                                        std::span<const ClassifiedEvent> incoming,
                                        rj_code_arch_t arch);

  [[nodiscard]] static bool same_event_identity(const PendingEvent &lhs, const PendingEvent &rhs);

  [[nodiscard]] static bool register_set_less(const RegisterSet &lhs, const RegisterSet &rhs);

  [[nodiscard]] static bool event_identity_less(const PendingEvent &lhs, const PendingEvent &rhs);

  static void merge_va_vdst_hazards(VaVdstHazardState &dst, const VaVdstHazardState &src);

  static void apply_expert_valu_delay(PendingState &state, const Instruction &inst,
                                      rj_code_arch_t arch, bool may_skip);
  static void merge_delay_alu(std::vector<PendingDelayAlu> &dst,
                              const std::vector<PendingDelayAlu> &src);

  // Join two initialized states. An unvisited predecessor is not an empty state.
  static void merge_into(PendingState &dst, const PendingState &src);

  // Skip valid unvisited predecessors; fail if any index is outside either
  // input array. An invalid edge must not silently drop hazards or must facts.
  [[nodiscard]] static util::FailureOr<PendingState>
  merge_predecessors(std::span<const size_t> predecessors, const std::vector<PendingState> &outputs,
                     std::span<const uint8_t> output_initialized);

  static void apply_wait_to_event_ages(PendingState &state, WaitCounterKind counter,
                                       uint32_t count);

  static void apply_wait(PendingState &state, WaitCounterKind counter, uint32_t count);

  static util::Result apply_kmcnt_wait(PendingState &state, uint32_t count, rj_code_arch_t arch);

  static void apply_implied_vm_vsrc_wait(PendingState &state, WaitCounterKind counter,
                                         uint32_t count);

  [[nodiscard]] static bool is_xcnt_smem_event(const PendingEvent &event);

  [[nodiscard]] static bool is_xcnt_vmem_event(const PendingEvent &event);

  static void apply_xcnt_wait(PendingState &state, uint32_t count);
  static void apply_xcnt_group_wait(PendingState &state, uint32_t count);
  static void apply_xcnt_smem_group_wait(PendingState &state);

  // Includes counter-only requests as well as materialized register hazards.
  [[nodiscard]] static bool has_xcnt_smem(const PendingState &state);
  [[nodiscard]] static bool has_xcnt_vmem(const PendingState &state);

  static void apply_xcnt_wait_implied_by_kmcnt(PendingState &state, uint32_t count);

  static void apply_xcnt_wait_implied_by_memory(PendingState &state, WaitCounterKind counter,
                                                uint32_t count);

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

  [[nodiscard]] static util::FailureOr<std::optional<amdgpu::WaitFields>>
  resolved_wait_fields(const PendingState &state, const Instruction &inst, rj_code_arch_t arch);
  static util::Result apply_waitcnt(PendingState &state, const Instruction &inst,
                                    rj_code_arch_t arch);

private:
  // Counter facets of one static operation can track different registers and
  // hazard flags. Location, instruction and event kind identify their operation.
  [[nodiscard]] static bool same_operation(const PendingEvent &lhs, const PendingEvent &rhs);

  [[nodiscard]] static auto register_ref_key(const std::optional<RegisterRef> &ref);

  template <typename Predicate>
  static void retire_event_kind_ages(PendingState &state, WaitCounterKind counter,
                                     Predicate belongs_to_group, uint32_t minimum_age = 0);

  template <typename Predicate>
  static void apply_filtered_vm_vsrc_wait(PendingState &state, Predicate is_implied,
                                          uint32_t count);

  template <typename Predicate>
  [[nodiscard]] static bool has_xcnt_event(const PendingState &state, Predicate predicate);

  template <typename Predicate>
  static void retire_completed_xcnt(PendingState &state, Predicate completed);

  template <typename Predicate>
  static void retire_xcnt_group(PendingState &state, Predicate belongs_to_group,
                                uint32_t count = 0);
};

} // namespace rocjitsu::waitcheck_detail
