// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/analysis/waitcheck/state.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"

#include <algorithm>
#include <tuple>

namespace rocjitsu::waitcheck_detail {

bool WaitcheckStateOps::is_completion_counter(WaitCounterKind counter) {
  return counter != WaitCounterKind::X && counter != WaitCounterKind::VmVsrc &&
         counter != WaitCounterKind::VaVdst && counter != WaitCounterKind::Depctr;
}

namespace {
bool has_conditional_route(WaitEventKind kind) {
  return kind == WaitEventKind::FlatLoad || kind == WaitEventKind::FlatStore;
}

bool gfx11_flat_ds(WaitCounterKind counter, WaitEventKind kind, rj_code_arch_t arch) {
  return counter == WaitCounterKind::Ds && has_conditional_route(kind) &&
         (arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5);
}

bool has_conditional_issue(WaitCounterKind counter, WaitEventKind kind, rj_code_arch_t arch,
                           bool exec_nonzero) {
  // A live VMcnt/VScnt position forces issue on GFX11. A live LGKM position
  // alone does not. Queue summaries are upper bounds, so a different VMEM
  // instruction may already have completed.
  if (gfx11_flat_ds(counter, kind, arch) && !exec_nonzero)
    return true;
  // CDNA5 3.2.2.1: ASYNC may remain outstanding while the LDS arrive skips
  // with EXEC=0 and DSCNT=0. A possible issue cannot age an older ASYNC load.
  if (kind == WaitEventKind::AsyncBarrier && !exec_nonzero)
    return true;
  return has_conditional_route(kind) && !WaitcheckTarget::flat_issues_both_counters(arch);
}
} // namespace

std::optional<uint32_t> WaitcheckStateOps::guaranteed_issue_order(WaitCounterKind counter,
                                                                  WaitEventKind kind,
                                                                  rj_code_arch_t arch,
                                                                  bool exec_nonzero) {
  // GFX11 and GFX12 issue both queue positions regardless of the lanes in each route.
  if (has_conditional_issue(counter, kind, arch, exec_nonzero))
    return std::nullopt;
  return completion_order_kind(counter, kind, arch);
}

bool WaitcheckStateOps::ordered_waw(const PendingEvent &event,
                                    std::span<const ClassifiedEvent> incoming,
                                    rj_code_arch_t arch) {
  WaitEventKind ordered = event.kind;
  if (event.counter == WaitCounterKind::Load && amdgpu::has_ordered_vmem_writeback(arch)) {
    // Only the VMEM facet of an older FLAT result can be superseded by a
    // native VMEM write. The incoming generic FLAT route remains unknown.
    if (ordered == WaitEventKind::FlatLoad)
      ordered = WaitEventKind::VmemNoSamplerLoad;
    if (ordered != WaitEventKind::VmemNoSamplerLoad && ordered != WaitEventKind::Sample &&
        ordered != WaitEventKind::Bvh)
      return false;
  } else if (!(event.counter == WaitCounterKind::Ds && ordered == WaitEventKind::Ds) &&
             !(event.counter == WaitCounterKind::Exp && ordered == WaitEventKind::LdsDirect &&
               amdgpu::has_ordered_lds_direct_writeback(arch))) {
    return false;
  }
  return std::ranges::any_of(incoming, [&](const ClassifiedEvent &other) {
    return other.counter == event.counter && other.kind == ordered &&
           other.registers == TrackedRegisterSource::Defs;
  });
}

void WaitcheckStateOps::apply_admission(PendingState &state, const Instruction &inst,
                                        std::span<const ClassifiedEvent> events,
                                        rj_code_arch_t arch) {
  for (const auto &event : events) {
    if (!is_completion_counter(event.counter) ||
        has_conditional_issue(event.counter, event.kind, arch, state.exec_nonzero))
      continue;
    // The incoming instruction must reserve all its credits before accessing
    // its registers. Only guaranteed younger members of an ordered FIFO count.
    const auto capacity = maximum_dependency_wait(arch, event.counter).value() + 1;
    const uint32_t remaining = capacity - issue_units(inst, event, arch);
    const auto idx = counter_index(event.counter);
    const bool reaches_capacity =
        std::ranges::any_of(
            state.pending[idx],
            [&](const auto &pending) { return pending.min_younger >= remaining; }) ||
        std::ranges::any_of(state.pending_event_ages[idx].values, [&](uint8_t age) {
          return age != kNoPendingEventAge && age >= remaining;
        });
    const auto &translations = state.pending[counter_index(WaitCounterKind::X)];
    const bool releases_translation = std::ranges::any_of(translations, [&](const auto &pending) {
      return vm_vsrc_event_implied_by_wait(pending.kind, event.counter) &&
             pending.min_younger_primary >= remaining;
    });
    if (reaches_capacity || releases_translation)
      (void)apply_memory_wait(state, event.counter, remaining, arch);
  }
}

void WaitcheckStateOps::age_events(PendingState &state, const PendingEvent &event,
                                   rj_code_arch_t arch, uint32_t maximum_wait, uint32_t units) {
  const size_t idx = counter_index(event.counter);
  const uint32_t max_age = event.counter == WaitCounterKind::VmVsrc ? 62 : maximum_wait;
  const bool completion = is_completion_counter(event.counter);
  const auto incoming =
      completion ? guaranteed_issue_order(event.counter, event.kind, arch, state.exec_nonzero)
                 : std::nullopt;
  auto age = [&](WaitEventKind older, auto &value) {
    // Direct async VMEM may need to retain its sources for replay. Its X
    // position is unqualified, so neither side can use it to prove X ordering.
    if (event.counter == WaitCounterKind::X &&
        (is_async_transfer_kind(event.kind) || is_async_transfer_kind(older)))
      return;
    if (!completion || (incoming && completion_order_kind(event.counter, older, arch) == incoming))
      value = std::min<uint32_t>(value + units, max_age);
  };
  if (arch == ROCJITSU_CODE_ARCH_CDNA5 && incoming) {
    // X and primary sequence numbers diverge as soon as different VMEM families
    // interleave. Count only definitely-issued younger credits in the same FIFO.
    for (auto &translation : state.pending[counter_index(WaitCounterKind::X)]) {
      const bool same_completion =
          (is_async_transfer_kind(translation.kind) && event.counter == WaitCounterKind::Async) ||
          (is_xcnt_vmem_event(translation) &&
           vm_vsrc_event_implied_by_wait(translation.kind, event.counter));
      if (same_completion && !has_conditional_route(translation.kind) &&
          completion_order_kind(event.counter, translation.kind, arch) == incoming)
        translation.min_younger_primary =
            std::min<uint32_t>(maximum_wait, translation.min_younger_primary + units);
    }
  }
  auto &ages = state.pending_event_ages[idx].values;
  for (size_t kind = 0; kind < ages.size(); ++kind)
    if (ages[kind] != kNoPendingEventAge)
      age(static_cast<WaitEventKind>(kind), ages[kind]);
  ages[static_cast<size_t>(event.kind)] = 0;
  for (auto &pending : state.pending[idx]) {
    if (!incoming && pending.implies_active_exec &&
        gfx11_flat_ds(event.counter, event.kind, arch) &&
        completion_order_kind(event.counter, pending.kind, arch) ==
            completion_order_kind(event.counter, event.kind, arch))
      pending.min_younger = std::min<uint32_t>(pending.min_younger + units, max_age);
    else
      age(pending.kind, pending.min_younger);
  }
}

namespace {
bool flat_has_other_pending_counters(const PendingState &state, WaitEventKind kind,
                                     WaitCounterKind waited_counter) {
  if (kind != WaitEventKind::FlatLoad && kind != WaitEventKind::FlatStore)
    return false;
  // Static FLAT routing is unknown. Completing one route does not prove that
  // the other route has consumed its VGPR sources. Include counter-only facets,
  // which can exist solely in the per-kind summaries (notably stores).
  for (auto counter : {WaitCounterKind::Load, WaitCounterKind::Store, WaitCounterKind::Ds}) {
    if (counter != waited_counter &&
        WaitcheckStateOps::counter_has_event_kind(state, counter, kind))
      return true;
  }
  return false;
}
} // namespace

uint32_t VgprMsbState::for_role(amdgpu::VgprMsbRole role) const {
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

bool WaitcheckStateOps::same_event_identity(const PendingEvent &lhs, const PendingEvent &rhs) {
  return lhs.counter == rhs.counter && lhs.kind == rhs.kind && lhs.regs == rhs.regs &&
         lhs.vcc_halves == rhs.vcc_halves && lhs.partial_reg == rhs.partial_reg &&
         lhs.partial_reg_mask == rhs.partial_reg_mask && lhs.special_reg == rhs.special_reg &&
         lhs.barrier_id == rhs.barrier_id && lhs.check_uses == rhs.check_uses &&
         lhs.check_defs == rhs.check_defs && lhs.check_exec_defs == rhs.check_exec_defs &&
         lhs.section_name == rhs.section_name && lhs.section_offset == rhs.section_offset &&
         lhs.file_offset == rhs.file_offset && lhs.instruction == rhs.instruction &&
         lhs.check_memory_order == rhs.check_memory_order &&
         lhs.check_program_end == rhs.check_program_end &&
         lhs.check_counter_parity_order == rhs.check_counter_parity_order;
}

auto WaitcheckStateOps::register_ref_key(const std::optional<RegisterRef> &ref) {
  return std::make_tuple(ref.has_value(), ref ? static_cast<uint8_t>(ref->cls) : uint8_t{0},
                         ref ? ref->index : uint16_t{0}, ref ? ref->width : uint8_t{0});
}

bool WaitcheckStateOps::register_set_less(const RegisterSet &lhs, const RegisterSet &rhs) {
  if (lhs == rhs)
    return false;

  std::vector<RegisterRef> lhs_regs;
  std::vector<RegisterRef> rhs_regs;
  lhs_regs.reserve(lhs.size());
  rhs_regs.reserve(rhs.size());
  // Inline these callbacks to avoid packing RegisterRef through the aggregate ABI.
  lhs.for_each([&](RegisterRef ref) __attribute__((always_inline)) { lhs_regs.push_back(ref); });
  rhs.for_each([&](RegisterRef ref) __attribute__((always_inline)) { rhs_regs.push_back(ref); });
  auto less = [](RegisterRef lhs_ref, RegisterRef rhs_ref) {
    return std::make_tuple(static_cast<uint8_t>(lhs_ref.cls), lhs_ref.index, lhs_ref.width) <
           std::make_tuple(static_cast<uint8_t>(rhs_ref.cls), rhs_ref.index, rhs_ref.width);
  };
  return std::ranges::lexicographical_compare(lhs_regs, rhs_regs, less);
}

bool WaitcheckStateOps::event_identity_less(const PendingEvent &lhs, const PendingEvent &rhs) {
  const auto lhs_key =
      std::tie(lhs.section_name, lhs.section_offset, lhs.file_offset, lhs.instruction, lhs.counter,
               lhs.kind, lhs.barrier_id, lhs.check_uses, lhs.check_defs, lhs.check_exec_defs,
               lhs.check_memory_order, lhs.check_program_end, lhs.check_counter_parity_order);
  const auto rhs_key =
      std::tie(rhs.section_name, rhs.section_offset, rhs.file_offset, rhs.instruction, rhs.counter,
               rhs.kind, rhs.barrier_id, rhs.check_uses, rhs.check_defs, rhs.check_exec_defs,
               rhs.check_memory_order, rhs.check_program_end, rhs.check_counter_parity_order);
  if (lhs_key != rhs_key)
    return lhs_key < rhs_key;
  if (register_ref_key(lhs.special_reg) != register_ref_key(rhs.special_reg))
    return register_ref_key(lhs.special_reg) < register_ref_key(rhs.special_reg);
  if (register_ref_key(lhs.partial_reg) != register_ref_key(rhs.partial_reg))
    return register_ref_key(lhs.partial_reg) < register_ref_key(rhs.partial_reg);
  if (lhs.partial_reg_mask != rhs.partial_reg_mask)
    return lhs.partial_reg_mask < rhs.partial_reg_mask;
  if (lhs.vcc_halves != rhs.vcc_halves)
    return lhs.vcc_halves < rhs.vcc_halves;
  return register_set_less(lhs.regs, rhs.regs);
}

void WaitcheckStateOps::merge_va_vdst_hazards(VaVdstHazardState &dst,
                                              const VaVdstHazardState &src) {
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

void WaitcheckStateOps::merge_delay_alu(std::vector<PendingDelayAlu> &dst,
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

void ScalarOffsetValues::merge(const ScalarOffsetValues &other) {
  if (count == 0 || *this == other)
    return;
  if (other.count == 0) {
    *this = {};
    return;
  }
  for (uint32_t value : other.values()) {
    if (std::ranges::find(values(), value) != values().end())
      continue;
    if (count == literals.size()) {
      *this = {};
      return;
    }
    literals[count++] = value;
  }
  std::ranges::sort(literals.begin(), literals.begin() + count);
}

void WaitcheckStateOps::merge_into(PendingState &dst, const PendingState &src) {
  dst.exec_nonzero &= src.exec_nonzero;
  dst.previous_vm_vsrc_zero_wait &= src.previous_vm_vsrc_zero_wait;
  if (dst.prefetch_enabled != src.prefetch_enabled)
    dst.prefetch_enabled.reset();
  dst.m0.merge(src.m0);
  for (auto it = dst.sgpr_offsets.begin(); it != dst.sgpr_offsets.end();) {
    const auto other = src.sgpr_offsets.find(it->first);
    if (other != src.sgpr_offsets.end()) {
      it->second.merge(other->second);
      if (it->second.count != 0) {
        ++it;
        continue;
      }
    }
    it = dst.sgpr_offsets.erase(it);
  }
  if (dst.vgpr_msb != src.vgpr_msb)
    dst.vgpr_msb = {.mode = 0, .known = false};
  if (dst.expert_scheduling != src.expert_scheduling)
    dst.expert_scheduling = {.enabled = true, .known = false};
  for (size_t i = 0; i < kCounterCount; ++i) {
    // Path coverage does not change hardware issue order.
    const bool same_order = std::ranges::equal(
        dst.pending[i], src.pending[i], [](const PendingEvent &lhs, const PendingEvent &rhs) {
          return same_event_identity(lhs, rhs) && lhs.min_younger == rhs.min_younger;
        });
    if (!dst.pending[i].empty() && !src.pending[i].empty() && !same_order)
      dst.uncertain_order[i] = true;
    for (PendingEvent &event : dst.pending[i]) {
      const auto position = std::ranges::lower_bound(src.pending[i], event, event_identity_less);
      const bool found = position != src.pending[i].end() && same_event_identity(*position, event);
      event.present_on_all_paths &= found && position->present_on_all_paths;
    }
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
        position = dst.pending[i].insert(position, event);
        position->present_on_all_paths = false;
      } else {
        position->implies_active_exec &= event.implies_active_exec;
        position->min_younger = std::min(position->min_younger, event.min_younger);
        position->min_younger_primary =
            std::min(position->min_younger_primary, event.min_younger_primary);
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

util::FailureOr<PendingState>
WaitcheckStateOps::merge_predecessors(std::span<const size_t> predecessors,
                                      const std::vector<PendingState> &outputs,
                                      std::span<const uint8_t> output_initialized) {
  PendingState merged;
  bool initialized = false;
  for (size_t predecessor : predecessors) {
    if (predecessor >= outputs.size() || predecessor >= output_initialized.size())
      return util::Result::failure();
    if (output_initialized[predecessor] == 0)
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

void WaitcheckStateOps::apply_wait_to_event_ages(PendingState &state, WaitCounterKind counter,
                                                 uint32_t count) {
  auto &ages = state.pending_event_ages[counter_index(counter)].values;
  for (uint8_t &age : ages) {
    if (age != kNoPendingEventAge && (count == 0 || age >= count))
      age = kNoPendingEventAge;
  }
}

void WaitcheckStateOps::apply_wait(PendingState &state, WaitCounterKind counter, uint32_t count) {
  const size_t idx = counter_index(counter);
  auto &pending = state.pending[idx];
  apply_wait_to_event_ages(state, counter, count);
  if (count == 0) {
    pending.clear();
    state.pending_smem[idx] = false;
    state.uncertain_order[idx] = false;
    return;
  }
  std::erase_if(pending, [count](const PendingEvent &event) { return event.min_younger >= count; });
  if (pending.empty())
    state.uncertain_order[idx] = false;
}

util::Result WaitcheckStateOps::apply_kmcnt_wait(PendingState &state, uint32_t count,
                                                 rj_code_arch_t arch) {
  if (amdgpu::waitcnt_model(arch).failed())
    return util::Result::failure();
  apply_wait(state, WaitCounterKind::Km, count);
  apply_xcnt_wait_implied_by_kmcnt(state, count);
  return util::Result::success();
}

template <typename Predicate>
void WaitcheckStateOps::retire_event_kind_ages(PendingState &state, WaitCounterKind counter,
                                               Predicate belongs_to_group, uint32_t minimum_age) {
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
void WaitcheckStateOps::apply_filtered_vm_vsrc_wait(PendingState &state, Predicate is_implied,
                                                    uint32_t count) {
  const size_t idx = counter_index(WaitCounterKind::VmVsrc);
  auto &pending = state.pending[idx];
  if (count == 0) {
    std::erase_if(pending, is_implied);
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
  std::erase_if(pending, [&](const PendingEvent &event) {
    return is_implied(event) && event.min_younger >= youngest_retired_age;
  });
  retire_event_kind_ages(state, WaitCounterKind::VmVsrc, is_implied, youngest_retired_age);
}

void WaitcheckStateOps::apply_implied_vm_vsrc_wait(PendingState &state, WaitCounterKind counter,
                                                   uint32_t count) {
  apply_filtered_vm_vsrc_wait(
      state,
      [&](const PendingEvent &event) {
        return vm_vsrc_event_implied_by_wait(event.kind, counter) &&
               !flat_has_other_pending_counters(state, event.kind, counter);
      },
      count);
}

bool WaitcheckStateOps::is_xcnt_smem_event(const PendingEvent &event) {
  return event.counter == WaitCounterKind::X && event.kind == WaitEventKind::Smem;
}

bool WaitcheckStateOps::is_xcnt_vmem_event(const PendingEvent &event) {
  return event.counter == WaitCounterKind::X && is_xcnt_vmem_kind(event.kind);
}

template <typename Predicate>
bool WaitcheckStateOps::has_xcnt_event(const PendingState &state, Predicate predicate) {
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

bool WaitcheckStateOps::has_xcnt_smem(const PendingState &state) {
  return has_xcnt_event(state, is_xcnt_smem_event);
}

bool WaitcheckStateOps::has_xcnt_vmem(const PendingState &state) {
  return has_xcnt_event(state, is_xcnt_vmem_event);
}

void WaitcheckStateOps::apply_xcnt_wait(PendingState &state, uint32_t count) {
  // SIInsertWaitcnts treats X_CNT as out of order while an SMEM
  // translation is pending. Only xcnt(0) proves that a particular scalar
  // source has been released.
  if (count != 0 && has_xcnt_event(state, is_xcnt_smem_event))
    return;
  apply_wait(state, WaitCounterKind::X, count);
}

template <typename Predicate>
void WaitcheckStateOps::retire_xcnt_group(PendingState &state, Predicate belongs_to_group,
                                          uint32_t count) {
  const size_t idx = counter_index(WaitCounterKind::X);
  auto &pending = state.pending[idx];
  std::erase_if(pending, [&](const PendingEvent &event) {
    return belongs_to_group(event) && event.min_younger >= count;
  });
  retire_event_kind_ages(state, WaitCounterKind::X, belongs_to_group, count);
  if (pending.empty())
    state.uncertain_order[idx] = false;
}

void WaitcheckStateOps::apply_xcnt_group_wait(PendingState &state, uint32_t count) {
  // Switching translations or ordering a VMEM write only retires definite
  // group members. A possible async replay source needs its own completion
  // or a full architectural X drain.
  if (count && has_xcnt_smem(state))
    return;
  retire_xcnt_group(
      state, [](const PendingEvent &event) { return !is_async_transfer_kind(event.kind); }, count);
}

void WaitcheckStateOps::apply_xcnt_wait_implied_by_kmcnt(PendingState &state, uint32_t count) {
  if (count != 0)
    return;
  if (!has_xcnt_event(state, is_xcnt_smem_event))
    return;
  retire_xcnt_group(state, is_xcnt_smem_event);
}

void WaitcheckStateOps::apply_xcnt_wait_implied_by_memory(PendingState &state,
                                                          WaitCounterKind counter, uint32_t count) {
  const size_t index = counter_index(WaitCounterKind::X);
  auto &translations = state.pending[index];
  auto linked = [&](const PendingEvent &event) {
    if (is_async_transfer_kind(event.kind))
      return counter == WaitCounterKind::Async;
    return is_xcnt_vmem_event(event) && vm_vsrc_event_implied_by_wait(event.kind, counter) &&
           !flat_has_other_pending_counters(state, event.kind, counter);
  };
  // A completed translation also proves that older VMEM translations have
  // finished. Compare X ages only on an ordered path, using an event present
  // on every path, and never let a scalar-group alternative establish a prefix.
  std::optional<uint32_t> prefix_age;
  // Equal saturated ages do not establish relative order between two sources.
  const auto maximum_x_age =
      maximum_dependency_wait(ROCJITSU_CODE_ARCH_CDNA5, WaitCounterKind::X).value();
  if (!state.uncertain_order[index] && !has_xcnt_smem(state)) {
    for (const auto &event : translations) {
      if (is_xcnt_vmem_event(event) && event.present_on_all_paths && linked(event) &&
          event.min_younger_primary >= count && event.min_younger < maximum_x_age)
        prefix_age = std::min(prefix_age.value_or(event.min_younger), event.min_younger);
    }
  }
  auto in_prefix = [&](uint32_t age) { return prefix_age && age >= *prefix_age; };
  std::erase_if(translations, [&](const PendingEvent &event) {
    return (linked(event) && event.min_younger_primary >= count) ||
           (is_xcnt_vmem_event(event) && in_prefix(event.min_younger));
  });
  // A primary zero proves every request of its class complete, including
  // summary-only entries. A partial wait only proves the ordered prefix.
  if (count == 0)
    retire_event_kind_ages(state, WaitCounterKind::X, linked);
  if (prefix_age)
    retire_event_kind_ages(state, WaitCounterKind::X, is_xcnt_vmem_event, *prefix_age);
  if (translations.empty())
    state.uncertain_order[index] = false;
}

bool WaitcheckStateOps::scalar_memory_makes_counter_out_of_order(const PendingState &state,
                                                                 WaitCounterKind counter,
                                                                 WaitcntModel model) {
  return counter == amdgpu::smem_wait_counter(model) && state.pending_smem[counter_index(counter)];
}

bool WaitcheckStateOps::counter_has_event_kind(const PendingState &state, WaitCounterKind counter,
                                               WaitEventKind kind) {
  return state.pending_event_ages[counter_index(counter)].values[static_cast<size_t>(kind)] !=
         kNoPendingEventAge;
}

bool WaitcheckStateOps::flat_memory_makes_counter_out_of_order(const PendingState &state,
                                                               WaitCounterKind counter,
                                                               rj_code_arch_t arch) {
  // Legacy FLAT stores can use an unknown memory path. GFX11 and GFX12 FLAT
  // instructions issue both queue positions even when all lanes use one route.
  if (counter == WaitCounterKind::Store && !flat_issues_both_counters(arch) &&
      counter_has_event_kind(state, counter, WaitEventKind::FlatStore))
    return true;
  if ((arch != ROCJITSU_CODE_ARCH_CDNA1 && arch != ROCJITSU_CODE_ARCH_CDNA2 &&
       arch != ROCJITSU_CODE_ARCH_CDNA3 && arch != ROCJITSU_CODE_ARCH_CDNA4) ||
      (counter != WaitCounterKind::Load && counter != WaitCounterKind::Ds)) {
    return false;
  }
  return counter_has_event_kind(state, counter, WaitEventKind::FlatLoad) ||
         counter_has_event_kind(state, counter, WaitEventKind::FlatStore);
}

util::FailureOr<bool> WaitcheckStateOps::counter_out_of_order(const PendingState &state,
                                                              WaitCounterKind counter,
                                                              rj_code_arch_t arch) {
  const auto model = amdgpu::waitcnt_model(arch);
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

util::Result WaitcheckStateOps::apply_counter_wait(PendingState &state, WaitCounterKind counter,
                                                   uint32_t count, rj_code_arch_t arch) {
  const auto out_of_order = counter_out_of_order(state, counter, arch);
  if (out_of_order.failed())
    return util::Result::failure();
  if (count != 0 && !is_completion_counter(counter) && out_of_order.value())
    return util::Result::success();
  apply_wait(state, counter, count);
  return util::Result::success();
}

util::FailureOr<uint32_t> WaitcheckStateOps::dependency_required_count(const PendingState &state,
                                                                       const PendingEvent &event,
                                                                       rj_code_arch_t arch) {
  if (is_completion_counter(event.counter)) {
    if (amdgpu::waitcnt_model(arch).failed())
      return util::Result::failure();
    return event.min_younger;
  }
  const auto out_of_order = counter_out_of_order(state, event.counter, arch);
  if (out_of_order.failed())
    return util::Result::failure();
  if (state.uncertain_order[counter_index(event.counter)] || out_of_order.value())
    return 0;
  const auto max_wait = maximum_dependency_wait(arch, event.counter);
  if (max_wait.failed())
    return util::Result::failure();
  return std::min(event.min_younger, max_wait.value());
}

bool WaitcheckStateOps::same_operation(const PendingEvent &lhs, const PendingEvent &rhs) {
  return lhs.kind == rhs.kind && lhs.section_name == rhs.section_name &&
         lhs.section_offset == rhs.section_offset && lhs.file_offset == rhs.file_offset &&
         lhs.instruction == rhs.instruction;
}

util::Result WaitcheckStateOps::apply_memory_wait(PendingState &state, WaitCounterKind counter,
                                                  uint32_t count, rj_code_arch_t arch) {
  if (amdgpu::waitcnt_model(arch).failed())
    return util::Result::failure();
  if (count != 0) {
    // A primary-counter wait can complete an operation even when a younger
    // counter-only request has no VM_VSRC facet. Correlate existing facets by
    // operation before removing primary events; their register payloads differ.
    const auto &primary = state.pending[counter_index(counter)];
    auto &sources = state.pending[counter_index(WaitCounterKind::VmVsrc)];
    const auto kind_completed = [&](const PendingEvent &source) {
      const uint8_t age =
          state.pending_event_ages[counter_index(counter)].values[static_cast<size_t>(source.kind)];
      return vm_vsrc_event_implied_by_wait(source.kind, counter) && age != kNoPendingEventAge &&
             age >= count && !flat_has_other_pending_counters(state, source.kind, counter);
    };
    std::erase_if(sources, [&](const PendingEvent &source) {
      if (!vm_vsrc_event_implied_by_wait(source.kind, counter) ||
          flat_has_other_pending_counters(state, source.kind, counter))
        return false;
      // The primary kind's newest request being complete proves the entire
      // kind complete, including operations with no materialized primary facet.
      if (kind_completed(source))
        return true;
      bool covered_on_all_paths = false;
      for (const PendingEvent &event : primary) {
        if (!same_operation(event, source))
          continue;
        // Every pending instance/variant must be old enough, and a common
        // primary facet must prove the operation's coverage across the join.
        if (event.min_younger < count)
          return false;
        covered_on_all_paths |= event.present_on_all_paths;
      }
      return covered_on_all_paths;
    });
    // Individual operation proofs cannot clear a whole kind's summary. Use
    // the primary counter's all-requests-complete proof for age-only facts.
    retire_event_kind_ages(state, WaitCounterKind::VmVsrc, kind_completed);
    if (sources.empty())
      state.uncertain_order[counter_index(WaitCounterKind::VmVsrc)] = false;
  }
  apply_wait(state, counter, count);
  // Partial waits above use the operation's actual completion FIFO. A count
  // over VM_VSRC alone cannot distinguish ordered results from invalidates.
  if (count == 0)
    apply_implied_vm_vsrc_wait(state, counter, count);
  if (arch == ROCJITSU_CODE_ARCH_CDNA5)
    apply_xcnt_wait_implied_by_memory(state, counter, count);
  return util::Result::success();
}

void WaitcheckStateOps::clear_salu_sgpr_hazards(SgprHazardState &state) {
  state.salu_hazards.reset();
  state.salu_producers.clear();
  state.vcc_hazard = static_cast<uint8_t>(state.vcc_hazard & ~kSgprHazardSalu);
  state.salu_vcc_producer.reset();
}

void WaitcheckStateOps::clear_valu_sgpr_hazards(SgprHazardState &state) {
  state.valu_hazards.reset();
  state.valu_producers.clear();
}

void WaitcheckStateOps::clear_valu_vcc_hazard(SgprHazardState &state) {
  state.vcc_hazard = static_cast<uint8_t>(state.vcc_hazard & ~kSgprHazardValu);
  state.valu_vcc_producer.reset();
}

void WaitcheckStateOps::merge_lane_producers(
    std::unordered_map<uint16_t, SgprHazardProducer> &dst,
    const std::unordered_map<uint16_t, SgprHazardProducer> &src) {
  for (const auto &[index, producer] : src)
    dst.try_emplace(index, producer);
}

void WaitcheckStateOps::merge_sgpr_hazards(SgprHazardState &dst, const SgprHazardState &src) {
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

void WaitcheckStateOps::apply_sgpr_hazard_wait(SgprHazardState &state, uint32_t depctr) {
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

util::Result WaitcheckStateOps::apply_wait_fields(PendingState &state, const WaitFields &fields,
                                                  rj_code_arch_t arch) {
  if (amdgpu::waitcnt_model(arch).failed())
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
    case WaitCounterKind::Async:
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

util::Result WaitcheckStateOps::apply_embedded_waitcnt(PendingState &state, const Instruction &inst,
                                                       rj_code_arch_t arch) {
  const auto fields = embedded_wait_fields(inst, arch);
  if (fields.failed())
    return util::Result::failure();
  if (fields.value())
    return apply_wait_fields(state, *fields.value(), arch);
  return util::Result::success();
}

util::Result WaitcheckStateOps::apply_waitcnt(PendingState &state, const Instruction &inst,
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
  if (inst.mnemonic() == "s_wait_alu" || inst.mnemonic() == "s_waitcnt_depctr")
    apply_sgpr_hazard_wait(state.sgpr_hazards,
                           static_cast<uint32_t>(inst.src_operand(0)->encoding_value()));
  return apply_wait_fields(state, *fields.value(), arch);
}

} // namespace rocjitsu::waitcheck_detail
