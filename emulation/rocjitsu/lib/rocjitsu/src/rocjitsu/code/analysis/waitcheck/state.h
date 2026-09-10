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
/// - Conflicting mode values become unknown; possible scalar-memory presence
///   and uncertain ordering are ORed. Delayed clears survive only if guaranteed
///   on both paths, with the later of their earliest guaranteed clear times.
///
/// A wait with threshold n retires an ordered event only when min_younger >= n;
/// a zero wait retires all events on that counter. Architecture-aware wrappers
/// reject partial waits when completion may be out of order and propagate
/// proven completion to related counters. Shared producing generations become
/// ready only after their relevant counter facets retire. Auxiliary hazard maps
/// retain representative producer locations when multiple paths explain the
/// same hazard; the choice of witness does not change its semantic guarantee.

#include "rocjitsu/code/analysis/waitcheck/target.h"
#include "rocjitsu/isa/arch/amdgpu/shared/vgpr_msb.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"

#include <algorithm>
#include <bitset>
#include <span>
#include <tuple>
#include <unordered_map>

namespace rocjitsu::waitcheck_detail {

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

struct PartialRegisterAccess {
  RegisterRef reg;
  uint8_t mask = kVgprFull32Mask;
};

struct PendingEvent {
  WaitCounterKind counter = WaitCounterKind::Load;
  WaitEventKind kind = WaitEventKind::Unknown;
  RegisterSet regs;
  RegisterSet old_value_regs;
  // LLVM tracks the low/high 16-bit physical subregisters used by D16 memory
  // operations independently. Keep the exceptional partial destination
  // sparse: almost every event still covers whole 32-bit register lanes.
  std::optional<RegisterRef> partial_reg;
  uint8_t partial_reg_mask = kVgprFull32Mask;
  std::optional<RegisterRef> special_reg;
  std::optional<int64_t> barrier_id;
  bool produces_regs = false;
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
  uint32_t min_younger = 0;

  bool operator==(const PendingEvent &) const = default;
};

struct SgprHazardProducer {
  std::string section_name;
  uint64_t section_offset = 0;
  uint64_t file_offset = 0;
  std::string instruction;

  bool operator==(const SgprHazardProducer &) const = default;
};

constexpr uint8_t kSgprHazardSalu = 1u << 0u;
constexpr uint8_t kSgprHazardValu = 1u << 1u;

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

struct VaVdstHazard {
  uint8_t age = 0;
  bool trans_since = false;
  SgprHazardProducer producer;

  bool operator==(const VaVdstHazard &) const = default;
};

struct VaVdstHazardState {
  std::unordered_map<uint16_t, VaVdstHazard> hazards;

  bool operator==(const VaVdstHazardState &) const = default;
};

struct VgprMsbState {
  uint8_t mode = 0;
  bool known = true;

  [[nodiscard]] uint32_t for_role(amdgpu::VgprMsbRole role) const {
    switch (role) {
    case amdgpu::VgprMsbRole::Src0:
      return mode & 0x3u;
    case amdgpu::VgprMsbRole::Src1:
      return (mode >> 2u) & 0x3u;
    case amdgpu::VgprMsbRole::Src2:
      return (mode >> 4u) & 0x3u;
    case amdgpu::VgprMsbRole::Dst:
      return (mode >> 6u) & 0x3u;
    case amdgpu::VgprMsbRole::None:
      return 0;
    }
    return 0;
  }

  bool operator==(const VgprMsbState &) const = default;
};

enum class DelayAluEffect : uint8_t {
  None,
  Valu,
  Salu,
};

struct PendingDelayAlu {
  uint8_t countdown = 0;
  DelayAluEffect effect = DelayAluEffect::None;

  bool operator==(const PendingDelayAlu &) const = default;
};

struct ExpertSchedulingState {
  bool enabled = false;
  bool known = true;

  bool operator==(const ExpertSchedulingState &) const = default;
};

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

// State transfer and joins are independent of diagnostic emission and graph
// traversal. Keep their original inline definitions together with the state
// representation so both straight-line and CFG drivers use the same rules.
struct WaitcheckStateOps : WaitcheckTarget {
  [[nodiscard]] static bool same_event_identity(const PendingEvent &lhs, const PendingEvent &rhs) {
    return lhs.counter == rhs.counter && lhs.kind == rhs.kind && lhs.regs == rhs.regs &&
           lhs.partial_reg == rhs.partial_reg && lhs.partial_reg_mask == rhs.partial_reg_mask &&
           lhs.special_reg == rhs.special_reg && lhs.barrier_id == rhs.barrier_id &&
           lhs.produces_regs == rhs.produces_regs && lhs.check_uses == rhs.check_uses &&
           lhs.check_defs == rhs.check_defs && lhs.check_exec_defs == rhs.check_exec_defs &&
           lhs.section_name == rhs.section_name && lhs.section_offset == rhs.section_offset &&
           lhs.file_offset == rhs.file_offset && lhs.instruction == rhs.instruction &&
           lhs.check_memory_order == rhs.check_memory_order &&
           lhs.check_program_end == rhs.check_program_end &&
           lhs.check_counter_parity_order == rhs.check_counter_parity_order;
  }

  [[nodiscard]] static auto register_ref_key(const std::optional<RegisterRef> &ref) {
    return std::make_tuple(ref.has_value(), ref ? static_cast<uint8_t>(ref->cls) : uint8_t{0},
                           ref ? ref->index : uint16_t{0}, ref ? ref->width : uint8_t{0});
  }

  [[nodiscard]] static bool register_set_less(const RegisterSet &lhs, const RegisterSet &rhs) {
    if (lhs == rhs)
      return false;

    std::vector<RegisterRef> lhs_regs;
    std::vector<RegisterRef> rhs_regs;
    lhs_regs.reserve(lhs.size());
    rhs_regs.reserve(rhs.size());
    lhs.for_each([&](RegisterRef ref) { lhs_regs.push_back(ref); });
    rhs.for_each([&](RegisterRef ref) { rhs_regs.push_back(ref); });
    auto less = [](RegisterRef lhs_ref, RegisterRef rhs_ref) {
      return std::make_tuple(static_cast<uint8_t>(lhs_ref.cls), lhs_ref.index, lhs_ref.width) <
             std::make_tuple(static_cast<uint8_t>(rhs_ref.cls), rhs_ref.index, rhs_ref.width);
    };
    return std::lexicographical_compare(lhs_regs.begin(), lhs_regs.end(), rhs_regs.begin(),
                                        rhs_regs.end(), less);
  }

  [[nodiscard]] static bool event_identity_less(const PendingEvent &lhs, const PendingEvent &rhs) {
    const auto lhs_key =
        std::tie(lhs.section_name, lhs.section_offset, lhs.file_offset, lhs.instruction,
                 lhs.counter, lhs.kind, lhs.barrier_id, lhs.produces_regs, lhs.check_uses,
                 lhs.check_defs, lhs.check_exec_defs, lhs.check_memory_order, lhs.check_program_end,
                 lhs.check_counter_parity_order);
    const auto rhs_key =
        std::tie(rhs.section_name, rhs.section_offset, rhs.file_offset, rhs.instruction,
                 rhs.counter, rhs.kind, rhs.barrier_id, rhs.produces_regs, rhs.check_uses,
                 rhs.check_defs, rhs.check_exec_defs, rhs.check_memory_order, rhs.check_program_end,
                 rhs.check_counter_parity_order);
    if (lhs_key != rhs_key)
      return lhs_key < rhs_key;
    if (register_ref_key(lhs.special_reg) != register_ref_key(rhs.special_reg))
      return register_ref_key(lhs.special_reg) < register_ref_key(rhs.special_reg);
    if (register_ref_key(lhs.partial_reg) != register_ref_key(rhs.partial_reg))
      return register_ref_key(lhs.partial_reg) < register_ref_key(rhs.partial_reg);
    if (lhs.partial_reg_mask != rhs.partial_reg_mask)
      return lhs.partial_reg_mask < rhs.partial_reg_mask;
    return register_set_less(lhs.regs, rhs.regs);
  }

  static void merge_va_vdst_hazards(VaVdstHazardState &dst, const VaVdstHazardState &src) {
    for (const auto &[index, src_hazard] : src.hazards) {
      auto [dst_it, inserted] = dst.hazards.try_emplace(index, src_hazard);
      if (inserted)
        continue;
      VaVdstHazard &dst_hazard = dst_it->second;
      if (dst_hazard == src_hazard)
        continue;

      dst_hazard.trans_since = dst_hazard.trans_since || src_hazard.trans_since;
      dst_hazard.age = dst_hazard.trans_since ? 0 : std::min(dst_hazard.age, src_hazard.age);
    }
  }

  static void merge_delay_alu(std::vector<PendingDelayAlu> &dst,
                              const std::vector<PendingDelayAlu> &src) {
    std::vector<PendingDelayAlu> common;
    for (DelayAluEffect effect : {DelayAluEffect::Valu, DelayAluEffect::Salu}) {
      auto earliest = [effect](const std::vector<PendingDelayAlu> &delays) {
        std::optional<uint8_t> result;
        for (const PendingDelayAlu &delay : delays)
          if (delay.effect == effect && (!result || delay.countdown < *result))
            result = delay.countdown;
        return result;
      };
      const auto lhs = earliest(dst);
      const auto rhs = earliest(src);
      // A delayed clear is guaranteed only if every incoming path scheduled
      // one. Wait until the latest of their earliest guaranteed clears.
      if (lhs && rhs)
        common.push_back({std::max(*lhs, *rhs), effect});
    }
    dst = std::move(common);
  }

  // Join two initialized states. An unvisited predecessor is not an empty state.
  static void merge_into(PendingState &dst, const PendingState &src) {
    dst.ready_regs &= src.ready_regs;
    dst.previous_vm_vsrc_zero_wait &= src.previous_vm_vsrc_zero_wait;
    if (dst.vgpr_msb != src.vgpr_msb)
      dst.vgpr_msb = {.mode = 0, .known = false};
    if (dst.expert_scheduling != src.expert_scheduling)
      dst.expert_scheduling = {.enabled = true, .known = false};
    for (size_t i = 0; i < kCounterCount; ++i) {
      if (!dst.pending[i].empty() && !src.pending[i].empty() && dst.pending[i] != src.pending[i])
        dst.uncertain_order[i] = true;
      for (size_t kind = 0; kind < kWaitEventKindCount; ++kind) {
        const uint8_t src_age = src.pending_event_ages[i].values[kind];
        uint8_t &dst_age = dst.pending_event_ages[i].values[kind];
        if (src_age != kNoPendingEventAge && (dst_age == kNoPendingEventAge || src_age < dst_age)) {
          dst_age = src_age;
        }
      }
      dst.pending_smem[i] = dst.pending_smem[i] || src.pending_smem[i];
      dst.uncertain_order[i] = dst.uncertain_order[i] || src.uncertain_order[i];
      for (const auto &event : src.pending[i]) {
        auto position = std::ranges::lower_bound(dst.pending[i], event, event_identity_less);
        if (position == dst.pending[i].end() || !same_event_identity(*position, event)) {
          dst.pending[i].insert(position, event);
        } else {
          position->min_younger = std::min(position->min_younger, event.min_younger);
          position->old_value_regs &= event.old_value_regs;
        }
      }
    }
    merge_sgpr_hazards(dst.sgpr_hazards, src.sgpr_hazards);
    merge_va_vdst_hazards(dst.va_vdst_hazards, src.va_vdst_hazards);
    dst.vgpr_msb_setreg_hazard = dst.vgpr_msb_setreg_hazard || src.vgpr_msb_setreg_hazard;
    if (!dst.async_barrier_post_wait && src.async_barrier_post_wait)
      dst.async_barrier_post_wait = src.async_barrier_post_wait;
    merge_delay_alu(dst.delay_alu, src.delay_alu);
  }

  [[nodiscard]] static PendingState
  merge_predecessors(std::span<const size_t> predecessors, const std::vector<PendingState> &outputs,
                     std::span<const uint8_t> output_initialized) {
    PendingState merged;
    bool initialized = false;
    for (size_t predecessor : predecessors) {
      if (predecessor >= outputs.size() || predecessor >= output_initialized.size() ||
          output_initialized[predecessor] == 0)
        continue;
      if (!initialized) {
        merged = outputs[predecessor];
        initialized = true;
      } else {
        merge_into(merged, outputs[predecessor]);
      }
    }
    return merged;
  }

  [[nodiscard]] static bool same_register_generation(const PendingEvent &lhs,
                                                     const PendingEvent &rhs) {
    return lhs.produces_regs && rhs.produces_regs && lhs.regs == rhs.regs &&
           lhs.section_name == rhs.section_name && lhs.section_offset == rhs.section_offset &&
           lhs.file_offset == rhs.file_offset && lhs.instruction == rhs.instruction;
  }

  static void make_retired_generations_ready(PendingState &state,
                                             std::span<const PendingEvent> retired_events) {
    for (const PendingEvent &retired : retired_events) {
      if (!retired.produces_regs)
        continue;
      retired.regs.for_each([&](RegisterRef reg) {
        if (reg.cls != RegClass::VGPR && reg.cls != RegClass::ACC_VGPR)
          return;
        const bool generation_still_pending =
            std::ranges::any_of(state.pending, [&](const std::vector<PendingEvent> &events) {
              return std::ranges::any_of(events, [&](const PendingEvent &pending) {
                return pending.regs.contains(reg) && same_register_generation(retired, pending);
              });
            });
        if (generation_still_pending)
          return;

        state.ready_regs.expand(reg);
        for (auto &events : state.pending) {
          for (PendingEvent &pending : events) {
            if (pending.produces_regs && pending.regs.contains(reg))
              pending.old_value_regs.expand(reg);
          }
        }
      });
    }
  }

  template <typename Predicate>
  static void retire_events(PendingState &state, std::vector<PendingEvent> &events,
                            Predicate should_retire) {
    std::vector<PendingEvent> retired_events;
    const auto retained =
        std::remove_if(events.begin(), events.end(), [&](const PendingEvent &event) {
          if (!should_retire(event))
            return false;
          retired_events.push_back(event);
          return true;
        });
    events.erase(retained, events.end());
    make_retired_generations_ready(state, retired_events);
  }

  static void apply_wait_to_event_ages(PendingState &state, WaitCounterKind counter,
                                       uint32_t count) {
    auto &ages = state.pending_event_ages[counter_index(counter)].values;
    for (uint8_t &age : ages) {
      if (age != kNoPendingEventAge && (count == 0 || age >= count))
        age = kNoPendingEventAge;
    }
  }

  static void apply_wait(PendingState &state, WaitCounterKind counter, uint32_t count) {
    const size_t idx = counter_index(counter);
    auto &pending = state.pending[idx];
    apply_wait_to_event_ages(state, counter, count);
    if (count == 0) {
      retire_events(state, pending, [](const PendingEvent &) { return true; });
      state.pending_smem[idx] = false;
      state.uncertain_order[idx] = false;
      return;
    }
    retire_events(state, pending,
                  [count](const PendingEvent &event) { return event.min_younger >= count; });
    if (pending.empty())
      state.uncertain_order[idx] = false;
  }

  static util::Result apply_kmcnt_wait(PendingState &state, uint32_t count, rj_code_arch_t arch) {
    const auto out_of_order = counter_out_of_order(state, WaitCounterKind::Km, arch);
    if (out_of_order.failed())
      return util::Result::failure();
    // LLVM cannot use a partial wait to advance any part of a counter whose
    // pending event kinds may complete out of order.
    if (count != 0 && out_of_order.value())
      return util::Result::success();
    apply_wait(state, WaitCounterKind::Km, count);
    apply_xcnt_wait_implied_by_kmcnt(state, count);
    return util::Result::success();
  }

  [[nodiscard]] static bool vm_vsrc_event_implied_by_wait(WaitEventKind kind,
                                                          WaitCounterKind counter) {
    switch (counter) {
    case WaitCounterKind::Load:
      return kind == WaitEventKind::VmemNoSamplerLoad || kind == WaitEventKind::FlatLoad;
    case WaitCounterKind::Store:
      return kind == WaitEventKind::VmemStore || kind == WaitEventKind::FlatStore;
    case WaitCounterKind::Ds:
      return kind == WaitEventKind::Ds || kind == WaitEventKind::FlatLoad ||
             kind == WaitEventKind::FlatStore;
    case WaitCounterKind::Sample:
      return kind == WaitEventKind::Sample;
    case WaitCounterKind::Bvh:
      return kind == WaitEventKind::Bvh;
    default:
      return false;
    }
  }

  template <typename Predicate>
  static void retire_event_kind_ages(PendingState &state, WaitCounterKind counter,
                                     Predicate belongs_to_group, uint32_t minimum_age = 0) {
    const size_t idx = counter_index(counter);
    auto &ages = state.pending_event_ages[idx].values;
    for (size_t kind = 0; kind < ages.size(); ++kind) {
      PendingEvent probe;
      probe.counter = counter;
      probe.kind = static_cast<WaitEventKind>(kind);
      if (belongs_to_group(probe) && ages[kind] != kNoPendingEventAge && ages[kind] >= minimum_age)
        ages[kind] = kNoPendingEventAge;
    }
    PendingEvent smem;
    smem.counter = counter;
    smem.kind = WaitEventKind::Smem;
    if (minimum_age == 0 && belongs_to_group(smem))
      state.pending_smem[idx] = false;
  }

  template <typename Predicate>
  static void apply_filtered_vm_vsrc_wait(PendingState &state, Predicate is_implied,
                                          uint32_t count) {
    const size_t idx = counter_index(WaitCounterKind::VmVsrc);
    auto &pending = state.pending[idx];
    if (count == 0) {
      retire_events(state, pending, is_implied);
      retire_event_kind_ages(state, WaitCounterKind::VmVsrc, is_implied);
      if (pending.empty())
        state.uncertain_order[idx] = false;
      return;
    }

    const size_t matching = static_cast<size_t>(std::ranges::count_if(pending, is_implied));
    if (state.uncertain_order[idx] || count >= matching)
      return;

    // Pending vectors are canonicalized by event identity for stable CFG
    // equality, so their physical order is not the hardware issue order.  An
    // event's counter age is the ordering fact: keep the `count` youngest
    // matching events and retire the older ones.  Equal ages at the boundary
    // mean the merge lost their relative order, in which case a partial wait
    // cannot prove that either one retired.
    std::vector<uint32_t> matching_ages;
    matching_ages.reserve(matching);
    for (const PendingEvent &event : pending) {
      if (is_implied(event))
        matching_ages.push_back(event.min_younger);
    }
    std::ranges::sort(matching_ages);
    if (matching_ages[count - 1] == matching_ages[count]) {
      state.uncertain_order[idx] = true;
      return;
    }
    const uint32_t youngest_retired_age = matching_ages[count];
    retire_events(state, pending, [&](const PendingEvent &event) {
      return is_implied(event) && event.min_younger >= youngest_retired_age;
    });
    retire_event_kind_ages(state, WaitCounterKind::VmVsrc, is_implied, youngest_retired_age);
  }

  static void apply_implied_vm_vsrc_wait(PendingState &state, WaitCounterKind counter,
                                         uint32_t count) {
    apply_filtered_vm_vsrc_wait(
        state,
        [&](const PendingEvent &event) {
          return vm_vsrc_event_implied_by_wait(event.kind, counter);
        },
        count);
  }

  [[nodiscard]] static bool is_xcnt_smem_event(const PendingEvent &event) {
    return event.counter == WaitCounterKind::X && event.kind == WaitEventKind::Smem;
  }

  [[nodiscard]] static bool is_xcnt_vmem_event(const PendingEvent &event) {
    return event.counter == WaitCounterKind::X && is_xcnt_vmem_kind(event.kind);
  }

  [[nodiscard]] static bool is_xcnt_store_event(const PendingEvent &event) {
    return event.counter == WaitCounterKind::X &&
           (event.kind == WaitEventKind::VmemStore || event.kind == WaitEventKind::FlatStore);
  }

  template <typename Predicate>
  [[nodiscard]] static bool has_xcnt_event(const PendingState &state, Predicate predicate) {
    const size_t idx = counter_index(WaitCounterKind::X);
    if (std::ranges::any_of(state.pending[idx], predicate))
      return true;
    for (size_t kind = 0; kind < kWaitEventKindCount; ++kind) {
      PendingEvent probe;
      probe.counter = WaitCounterKind::X;
      probe.kind = static_cast<WaitEventKind>(kind);
      if (predicate(probe) && (state.pending_event_ages[idx].values[kind] != kNoPendingEventAge ||
                               (probe.kind == WaitEventKind::Smem && state.pending_smem[idx])))
        return true;
    }
    return false;
  }

  static void apply_xcnt_wait(PendingState &state, uint32_t count) {
    // SIInsertWaitcnts treats X_CNT as out of order while an SMEM
    // translation is pending. Only xcnt(0) proves that a particular scalar
    // source has been released.
    if (count != 0 && has_xcnt_event(state, is_xcnt_smem_event))
      return;
    apply_wait(state, WaitCounterKind::X, count);
  }

  template <typename Predicate>
  static void retire_xcnt_group(PendingState &state, Predicate belongs_to_group) {
    const size_t idx = counter_index(WaitCounterKind::X);
    auto &pending = state.pending[idx];
    retire_events(state, pending, belongs_to_group);
    retire_event_kind_ages(state, WaitCounterKind::X, belongs_to_group);
    if (pending.empty())
      state.uncertain_order[idx] = false;
  }

  static void apply_xcnt_wait_implied_by_kmcnt(PendingState &state, uint32_t count) {
    if (count != 0)
      return;
    if (!has_xcnt_event(state, is_xcnt_smem_event))
      return;
    if (has_xcnt_event(state, is_xcnt_vmem_event)) {
      retire_xcnt_group(state, is_xcnt_smem_event);
      return;
    }
    apply_xcnt_wait(state, 0);
  }

  static void apply_xcnt_wait_implied_by_loadcnt(PendingState &state, uint32_t count) {
    if (!has_xcnt_event(state, is_xcnt_vmem_event) || has_xcnt_event(state, is_xcnt_store_event))
      return;
    if (has_xcnt_event(state, is_xcnt_smem_event)) {
      if (count == 0)
        retire_xcnt_group(state, is_xcnt_vmem_event);
      return;
    }
    apply_xcnt_wait(state, count);
  }

  [[nodiscard]] static bool scalar_memory_makes_counter_out_of_order(const PendingState &state,
                                                                     WaitCounterKind counter,
                                                                     WaitcntModel model) {
    return counter == smem_wait_counter(model) && state.pending_smem[counter_index(counter)];
  }

  [[nodiscard]] static bool counter_has_event_kind(const PendingState &state,
                                                   WaitCounterKind counter, WaitEventKind kind) {
    return state.pending_event_ages[counter_index(counter)].values[static_cast<size_t>(kind)] !=
           kNoPendingEventAge;
  }

  [[nodiscard]] static std::optional<WaitEventKind>
  normalized_hardware_event_kind(WaitCounterKind counter, WaitEventKind kind, WaitcntModel model) {
    switch (counter) {
    case WaitCounterKind::Load:
      // Generic FLAT and ordinary VMEM loads both raise VMEM_READ_ACCESS.
      // GLOBAL_INV is explicitly ignored by LLVM's LOAD_CNT out-of-order
      // test. Pre-gfx12 image event kinds share that same hardware event.
      if (kind == WaitEventKind::GlobalInv)
        return std::nullopt;
      if (kind == WaitEventKind::FlatLoad || kind == WaitEventKind::LdsDirect ||
          (uses_legacy_waitcnt(model) &&
           (kind == WaitEventKind::Sample || kind == WaitEventKind::Bvh))) {
        return WaitEventKind::VmemNoSamplerLoad;
      }
      return kind;
    case WaitCounterKind::Ds:
      // A generic FLAT access raises the same LDS_ACCESS event as native DS.
      if (kind == WaitEventKind::FlatLoad || kind == WaitEventKind::FlatStore)
        return WaitEventKind::Ds;
      return kind;
    case WaitCounterKind::Store:
      if (kind == WaitEventKind::GlobalWb)
        return WaitEventKind::VmemStore;
      return kind;
    case WaitCounterKind::X:
      // X_CNT distinguishes VMEM_GROUP from SMEM_GROUP, not the underlying
      // load/store/image operation.
      if (kind == WaitEventKind::Smem)
        return WaitEventKind::Smem;
      if (is_xcnt_vmem_kind(kind))
        return WaitEventKind::VmemNoSamplerLoad;
      return kind;
    case WaitCounterKind::VmVsrc:
      if (kind == WaitEventKind::Ds)
        return WaitEventKind::Ds;
      if (kind == WaitEventKind::FlatLoad || kind == WaitEventKind::FlatStore)
        return WaitEventKind::FlatLoad;
      if (is_xcnt_vmem_kind(kind))
        return WaitEventKind::VmemNoSamplerLoad;
      return kind;
    case WaitCounterKind::Async:
      // Load, store, and barrier forms all raise ASYNC_ACCESS.
      return WaitEventKind::AsyncLdsLoad;
    case WaitCounterKind::Tensor:
      return WaitEventKind::TensorLdsLoad;
    default:
      return kind;
    }
  }

  [[nodiscard]] static bool flat_memory_makes_counter_out_of_order(const PendingState &state,
                                                                   WaitCounterKind counter,
                                                                   rj_code_arch_t arch) {
    if ((arch != ROCJITSU_CODE_ARCH_CDNA3 && arch != ROCJITSU_CODE_ARCH_CDNA4) ||
        (counter != WaitCounterKind::Load && counter != WaitCounterKind::Ds)) {
      return false;
    }
    return counter_has_event_kind(state, counter, WaitEventKind::FlatLoad) ||
           counter_has_event_kind(state, counter, WaitEventKind::FlatStore);
  }

  [[nodiscard]] static util::FailureOr<bool>
  counter_out_of_order(const PendingState &state, WaitCounterKind counter, rj_code_arch_t arch) {
    const auto model = waitcnt_model(arch);
    if (model.failed())
      return util::Result::failure();
    // Match WaitcntBrackets::counterOutOfOrder. Scalar memory can always
    // complete out of order on its accounting counter (and on X_CNT).
    if (scalar_memory_makes_counter_out_of_order(state, counter, model.value()) ||
        (counter == WaitCounterKind::X && has_xcnt_event(state, is_xcnt_smem_event)) ||
        flat_memory_makes_counter_out_of_order(state, counter, arch)) {
      return true;
    }

    // Before VScnt, VMEM loads and stores share LOAD_CNT but are mutually
    // ordered, so LLVM deliberately does not treat their distinct event bits
    // as an out-of-order mixture.
    if (counter == WaitCounterKind::Load && model.value() == WaitcntModel::LegacyNoVscnt) {
      return false;
    }

    std::bitset<kWaitEventKindCount> kinds;
    const auto &ages = state.pending_event_ages[counter_index(counter)].values;
    for (size_t kind_index = 0; kind_index < ages.size(); ++kind_index) {
      if (ages[kind_index] == kNoPendingEventAge)
        continue;
      const auto normalized = normalized_hardware_event_kind(
          counter, static_cast<WaitEventKind>(kind_index), model.value());
      if (normalized)
        kinds.set(static_cast<size_t>(*normalized));
    }
    return kinds.count() > 1;
  }

  static util::Result apply_counter_wait(PendingState &state, WaitCounterKind counter,
                                         uint32_t count, rj_code_arch_t arch) {
    const auto out_of_order = counter_out_of_order(state, counter, arch);
    if (out_of_order.failed())
      return util::Result::failure();
    if (count != 0 && out_of_order.value())
      return util::Result::success();
    apply_wait(state, counter, count);
    return util::Result::success();
  }

  [[nodiscard]] static util::FailureOr<uint32_t>
  dependency_required_count(const PendingState &state, const PendingEvent &event,
                            rj_code_arch_t arch) {
    const auto out_of_order = counter_out_of_order(state, event.counter, arch);
    if (out_of_order.failed())
      return util::Result::failure();
    if (state.uncertain_order[counter_index(event.counter)] || out_of_order.value())
      return 0;
    return event.min_younger;
  }

  static util::Result apply_memory_wait(PendingState &state, WaitCounterKind counter,
                                        uint32_t count, rj_code_arch_t arch) {
    const auto out_of_order = counter_out_of_order(state, counter, arch);
    if (out_of_order.failed())
      return util::Result::failure();
    // SIInsertWaitcnts cannot use a nonzero wait to retire a particular event
    // while scalar memory is pending on this counter. This notably covers
    // legacy LGKMCNT shared by SMEM and DS operations.
    if (count != 0 && out_of_order.value())
      return util::Result::success();
    apply_wait(state, counter, count);
    apply_implied_vm_vsrc_wait(state, counter, count);
    if (counter == WaitCounterKind::Load)
      apply_xcnt_wait_implied_by_loadcnt(state, count);
    return util::Result::success();
  }

  static void clear_salu_sgpr_hazards(SgprHazardState &state) {
    state.salu_hazards.reset();
    state.salu_producers.clear();
    state.vcc_hazard = static_cast<uint8_t>(state.vcc_hazard & ~kSgprHazardSalu);
    state.salu_vcc_producer.reset();
  }

  static void clear_valu_sgpr_hazards(SgprHazardState &state) {
    state.valu_hazards.reset();
    state.valu_producers.clear();
  }

  static void clear_valu_vcc_hazard(SgprHazardState &state) {
    state.vcc_hazard = static_cast<uint8_t>(state.vcc_hazard & ~kSgprHazardValu);
    state.valu_vcc_producer.reset();
  }

  static void merge_lane_producers(std::unordered_map<uint16_t, SgprHazardProducer> &dst,
                                   const std::unordered_map<uint16_t, SgprHazardProducer> &src) {
    for (const auto &[index, producer] : src)
      dst.try_emplace(index, producer);
  }

  static void merge_sgpr_hazards(SgprHazardState &dst, const SgprHazardState &src) {
    dst.consecutive_ds_nops = std::min(dst.consecutive_ds_nops, src.consecutive_ds_nops);
    dst.tracked_pairs |= src.tracked_pairs;
    dst.tracked_vcc = dst.tracked_vcc || src.tracked_vcc;
    dst.salu_hazards |= src.salu_hazards;
    dst.valu_hazards |= src.valu_hazards;
    dst.vcc_hazard |= src.vcc_hazard;
    merge_lane_producers(dst.salu_producers, src.salu_producers);
    merge_lane_producers(dst.valu_producers, src.valu_producers);
    if (!dst.salu_vcc_producer && src.salu_vcc_producer)
      dst.salu_vcc_producer = src.salu_vcc_producer;
    if (!dst.valu_vcc_producer && src.valu_vcc_producer)
      dst.valu_vcc_producer = src.valu_vcc_producer;
  }

  static void apply_sgpr_hazard_wait(SgprHazardState &state, uint32_t depctr) {
    constexpr uint32_t kDepctrSaSdstShift = 0;
    constexpr uint32_t kDepctrVaVccShift = 1;
    constexpr uint32_t kDepctrVaSdstShift = 9;
    constexpr uint32_t kDepctrSaSdstWidth = 1;
    constexpr uint32_t kDepctrVaVccWidth = 1;
    constexpr uint32_t kDepctrVaSdstWidth = 3;

    if (depctr_field(depctr, kDepctrSaSdstShift, kDepctrSaSdstWidth) == 0)
      clear_salu_sgpr_hazards(state);
    if (depctr_field(depctr, kDepctrVaVccShift, kDepctrVaVccWidth) == 0)
      clear_valu_vcc_hazard(state);
    if (depctr_field(depctr, kDepctrVaSdstShift, kDepctrVaSdstWidth) == 0)
      clear_valu_sgpr_hazards(state);
  }

  static util::Result apply_wait_fields(PendingState &state, const WaitFields &fields,
                                        rj_code_arch_t arch) {
    if (waitcnt_model(arch).failed())
      return util::Result::failure();
    for (size_t i = 0; i < fields.size(); ++i) {
      if (!fields[i])
        continue;
      const WaitCounterKind counter = static_cast<WaitCounterKind>(i);
      const uint32_t count = *fields[i];
      switch (counter) {
      case WaitCounterKind::Load:
      case WaitCounterKind::Store:
      case WaitCounterKind::Ds:
      case WaitCounterKind::Sample:
      case WaitCounterKind::Bvh:
        if (apply_memory_wait(state, counter, count, arch).failed())
          return util::Result::failure();
        break;
      case WaitCounterKind::Km:
        if (apply_kmcnt_wait(state, count, arch).failed())
          return util::Result::failure();
        break;
      case WaitCounterKind::X:
        apply_xcnt_wait(state, count);
        break;
      case WaitCounterKind::VaVdst:
        std::erase_if(state.va_vdst_hazards.hazards, [count](const auto &entry) {
          const VaVdstHazard &hazard = entry.second;
          return count == 0 || (!hazard.trans_since && hazard.age >= count);
        });
        if (apply_counter_wait(state, counter, count, arch).failed())
          return util::Result::failure();
        break;
      default:
        if (apply_counter_wait(state, counter, count, arch).failed())
          return util::Result::failure();
        break;
      }
    }
    return util::Result::success();
  }

  static util::Result apply_embedded_waitcnt(PendingState &state, const Instruction &inst,
                                             rj_code_arch_t arch) {
    const auto fields = embedded_wait_fields(inst, arch);
    if (fields.failed())
      return util::Result::failure();
    if (fields.value())
      return apply_wait_fields(state, *fields.value(), arch);
    return util::Result::success();
  }

  static util::Result apply_waitcnt(PendingState &state, const Instruction &inst,
                                    rj_code_arch_t arch) {
    const auto fields = explicit_wait_fields(inst, arch);
    if (fields.failed())
      return util::Result::failure();
    if (!fields.value())
      return util::Result::success();
    if (inst.mnemonic() == "s_wait_idle") {
      VgprMsbState vgpr_msb = state.vgpr_msb;
      ExpertSchedulingState expert_scheduling = state.expert_scheduling;
      state = {};
      state.vgpr_msb = vgpr_msb;
      state.expert_scheduling = expert_scheduling;
      return util::Result::success();
    }
    if (inst.mnemonic() == "s_wait_alu")
      apply_sgpr_hazard_wait(state.sgpr_hazards,
                             static_cast<uint32_t>(inst.src_operand(0)->encoding_value()));
    return apply_wait_fields(state, *fields.value(), arch);
  }
};

} // namespace rocjitsu::waitcheck_detail
