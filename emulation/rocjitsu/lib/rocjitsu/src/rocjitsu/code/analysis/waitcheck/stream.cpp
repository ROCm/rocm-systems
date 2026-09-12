// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/analysis/waitcheck/stream.h"

#include "rocjitsu/code/analysis/control_flow.h"
#include "rocjitsu/code/analysis/waitcheck/state.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <cassert>

namespace rocjitsu {
namespace {
using namespace waitcheck_detail;
using Ops = WaitcheckStateOps;

bool has_committed_generations(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4;
}

std::optional<RegisterRef> first_intersection(const RegisterSet &a, const RegisterSet &b) {
  std::optional<RegisterRef> first;
  (a & b).for_each([&](RegisterRef ref) {
    if (!first)
      first = ref;
  });
  return first;
}

RegisterSet vector_registers(const RegisterSet &regs) {
  RegisterSet result;
  regs.for_each([&](RegisterRef ref) {
    if (ref.cls == RegClass::VGPR || ref.cls == RegClass::ACC_VGPR)
      result.expand(ref);
  });
  return result;
}

// Reuse the original checker's whole-register view, including selector-encoded
// special registers. Liveness deliberately omits those singleton aliases, so
// its ordinary-register projection is insufficient for memory source lifetimes.
struct RegisterEffects {
  RegisterSet defs;
  RegisterSet uses;
};

// SVE controls scratch vector addressing; an enabled register index of zero is v0.
bool scratch_vector_address_enabled(const Instruction &inst, rj_code_arch_t arch) {
  // Only successfully decoded scratch instructions reach this private helper.
  const uint32_t *words = inst.raw_encoding();
  assert(words && inst.size() >= static_cast<int>(2 * sizeof(uint32_t)));
  constexpr uint32_t kCdnaScratchSveBit = 1u << 13;
  constexpr uint32_t kRdna3ScratchSveBit = 1u << 23;
  constexpr uint32_t kGfx12ScratchSveBit = 1u << 17;
  if (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4)
    return (words[0] & kCdnaScratchSveBit) != 0;
  if (arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5)
    return (words[1] & kRdna3ScratchSveBit) != 0;
  assert(arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5);
  return (words[1] & kGfx12ScratchSveBit) != 0;
}

RegisterEffects register_effects(const Instruction &inst, rj_code_arch_t arch, uint32_t wave_size) {
  RegisterEffects effects;
  const auto name = inst.mnemonic();
  const bool scalar_address = [&] {
    if (!name.starts_with("global_") && !name.starts_with("scratch_"))
      return false;
    for (int i = 1; i < inst.num_src_operands(); ++i) {
      const auto *op = inst.src_operand(i);
      if (op) {
        const auto ref = op->to_register_ref();
        if (ref && ref->cls == RegClass::SGPR)
          return true;
      }
    }
    return false;
  }();
  if (!Ops::is_cdna4_mubuf_lds_load(inst, arch)) {
    for (int i = 0; i < inst.num_dst_operands(); ++i) {
      const auto *op = inst.dst_operand(i);
      if (!op)
        continue;
      if (auto ref = op->to_register_ref()) {
        // Compare, carry-out and divide-scale scalar destinations are wave
        // masks; encoded operands can retain their maximum two-SGPR width.
        const bool carry_out = name == "v_add_co_u32" || name == "v_add_co_ci_u32" ||
                               name == "v_sub_co_u32" || name == "v_sub_co_ci_u32" ||
                               name == "v_subrev_co_u32" || name == "v_subrev_co_ci_u32" ||
                               name == "v_mad_co_u64_u32" || name == "v_mad_co_i64_i32" ||
                               name == "v_mad_u64_u32" || name == "v_mad_i64_i32";
        const bool wave_mask = (i == 0 && name.starts_with("v_cmp")) ||
                               (i == 1 && (carry_out || name.starts_with("v_div_scale_")));
        if (wave_mask && ref->cls == RegClass::SGPR)
          ref->width = wave_size / 32;
        effects.defs.expand(*ref);
      }
    }
  }
  inst.implicit_defs(effects.defs);
  for (int i = 0; i < inst.num_src_operands(); ++i) {
    const auto *op = inst.src_operand(i);
    if (!op)
      continue;
    auto ref = op->to_register_ref();
    if (!ref)
      continue;
    if (i == 0 && name.starts_with("scratch_") && !scratch_vector_address_enabled(inst, arch))
      continue;
    if (i == 0 && scalar_address && ref->cls == RegClass::VGPR)
      ref->width = 1;
    if (i == 2 && ref->cls == RegClass::SGPR &&
        (name.starts_with("v_cndmask_") || name == "v_add_co_ci_u32" || name == "v_sub_co_ci_u32" ||
         name == "v_subrev_co_ci_u32"))
      ref->width = wave_size / 32;
    effects.uses.expand(*ref);
  }
  inst.implicit_uses(effects.uses);
  if (inst.flags() & WRITES_EXEC)
    effects.defs.expand({RegClass::EXEC, 0, 1});
  return effects;
}

std::optional<std::string_view> deferred_reason(const Instruction &inst,
                                                std::span<const ClassifiedEvent> events) {
  const auto name = inst.mnemonic();
  if (is_control_flow_transfer(inst) && !is_program_path_terminator(inst))
    return "control flow requires CFG analysis";
  if (name == "s_set_vgpr_msb" || name.starts_with("s_setreg") ||
      name.starts_with("s_set_gpr_idx") || name == "s_setvskip" ||
      name.starts_with("s_waitcnt_depctr") || name == "s_wait_alu") {
    // Dependency waits themselves are handled by the transfer layer, below.
    if (!inst.is_waitcnt())
      return "scheduling or register-mode state is not analyzed";
  }
  if (name.starts_with("v_movrel") || name.starts_with("v_swaprel") || name.starts_with("s_movrel"))
    return "relative-register operands are not analyzed";
  if (name.find("_d16") != std::string_view::npos || name.starts_with("v_dual_"))
    return "partial-register or dual-issue semantics are not analyzed";
  if (name.starts_with("image_") || name.find("atomic") != std::string_view::npos)
    return "image and atomic operand semantics are not analyzed";
  for (const auto &event : events) {
    if (event.check_counter_parity_order)
      return "counter parity ordering is not analyzed";
    if (event.counter == WaitCounterKind::Async || event.counter == WaitCounterKind::Tensor ||
        event.check_memory_order || event.barrier_id)
      return "async, tensor or barrier ordering is not analyzed";
  }
  if (inst.is_barrier())
    return "barrier ordering is not analyzed";
  if (inst.is_memory_op() && events.empty())
    return "memory instruction has no event policy";
  if (name.starts_with("s_") && events.empty() && !inst.is_waitcnt() &&
      !is_program_path_terminator(inst) && name != "s_nop" && name != "s_delay_alu" &&
      inst.num_src_operands() == 0 && inst.num_dst_operands() == 0)
    return "unmodeled scalar control instruction";
  return std::nullopt;
}

RegisterSet registers_for_event(const ClassifiedEvent &event, const RegisterEffects &effects,
                                const Instruction &inst) {
  switch (event.registers) {
  case TrackedRegisterSource::None:
    return {};
  case TrackedRegisterSource::Defs:
    return effects.defs;
  case TrackedRegisterSource::Uses:
    return effects.uses;
  case TrackedRegisterSource::VectorUses:
    return vector_registers(effects.uses);
  case TrackedRegisterSource::StoreDataUses: {
    const auto name = inst.mnemonic();
    const int index = name.starts_with("buffer_") || name.starts_with("tbuffer_") ? 0 : 1;
    RegisterSet regs;
    if (const auto *op = inst.src_operand(index)) {
      if (auto ref = op->to_register_ref())
        regs.expand(*ref);
    }
    return regs;
  }
  }
  return {};
}

bool ordered_waw(const PendingEvent &event, std::span<const ClassifiedEvent> current,
                 rj_code_arch_t arch) {
  if (!has_ordered_vmem_writeback(arch))
    return false;
  if (event.kind == WaitEventKind::FlatLoad) {
    if (event.counter != WaitCounterKind::Load)
      return false;
    return std::ranges::any_of(current, [](const auto &other) {
      return other.counter == WaitCounterKind::Load &&
             other.kind == WaitEventKind::VmemNoSamplerLoad;
    });
  }
  if (event.kind != WaitEventKind::VmemNoSamplerLoad)
    return false;
  return std::ranges::any_of(current, [&](const auto &other) { return other.kind == event.kind; });
}

class StreamAnalyzer {
public:
  StreamAnalyzer(rj_code_arch_t arch, WaitcheckStreamOptions options, WaitcheckStreamReport &report)
      : arch_(arch), options_(options), report_(report) {
    state_.expert_scheduling.enabled = options.expert_scheduling;
  }

  // Control-flow operands are modeled even though this driver does not follow
  // their edges. Check them without issuing events or transferring state.
  util::Result check_control_flow_operands(const Instruction &inst, uint64_t offset) {
    return check_dependencies(inst, register_effects(inst, arch_, options_.wave_size), {}, offset);
  }

  util::Result analyze(const Instruction &inst, std::span<const ClassifiedEvent> events,
                       uint64_t offset) {
    if (inst.is_waitcnt())
      return Ops::apply_waitcnt(state_, inst, arch_);
    if (Ops::apply_embedded_waitcnt(state_, inst, arch_).failed())
      return util::Result::failure();
    const auto effects = register_effects(inst, arch_, options_.wave_size);
    apply_translation_ordering(effects, events);
    if (check_dependencies(inst, effects, events, offset).failed())
      return util::Result::failure();
    RegisterSet async_defs;
    for (const auto &event : events) {
      if (event.registers == TrackedRegisterSource::Defs)
        async_defs |= registers_for_event(event, effects, inst);
      if (add_event(event, effects, inst, offset).failed())
        return util::Result::failure();
    }
    update_committed_registers(effects, async_defs);
    return util::Result::success();
  }

private:
  rj_code_arch_t arch_;
  WaitcheckStreamOptions options_;
  PendingState state_;
  RegisterSet local_ready_;
  WaitcheckStreamReport &report_;

  util::Result check_dependencies(const Instruction &inst, const RegisterEffects &effects,
                                  std::span<const ClassifiedEvent> current, uint64_t offset) {
    for (const auto &pending : state_.pending) {
      for (const auto &event : pending) {
        std::optional<RegisterRef> reg;
        auto access = WaitcheckAccess::Read;
        if (event.check_uses)
          reg = first_intersection(event.regs - event.old_value_regs, effects.uses);
        if (!reg && event.check_defs) {
          // A visible old value can satisfy a read, but an outstanding memory
          // response can still overwrite a newer synchronous definition.
          if (!ordered_waw(event, current, arch_))
            reg = first_intersection(event.regs, effects.defs);
          access = WaitcheckAccess::Write;
        }
        if (!reg && event.check_exec_defs && effects.defs.contains({RegClass::EXEC, 0, 1})) {
          reg = RegisterRef{RegClass::EXEC, 0, 1};
          access = WaitcheckAccess::Write;
        }
        if (!reg)
          continue;
        const auto count = Ops::dependency_required_count(state_, event, arch_);
        if (count.failed())
          return util::Result::failure();
        auto wait = wait_expression(event.counter, count.value(), arch_);
        if (wait.failed())
          return util::Result::failure();
        report_.diagnostics.push_back({event.section_offset, offset, event.counter, *reg, access,
                                       count.value(), event.instruction, inst.disassemble(),
                                       std::move(wait).value()});
      }
    }
    return util::Result::success();
  }

  util::Result add_event(const ClassifiedEvent &classification, const RegisterEffects &effects,
                         const Instruction &inst, uint64_t offset) {
    PendingEvent event;
    event.counter = classification.counter;
    event.kind = classification.kind;
    event.regs = registers_for_event(classification, effects, inst);
    event.produces_regs = has_committed_generations(arch_) &&
                          classification.registers == TrackedRegisterSource::Defs &&
                          event.regs == vector_registers(event.regs);
    if (event.produces_regs)
      event.old_value_regs = event.regs & (state_.ready_regs | local_ready_);
    event.check_uses = classification.check_uses;
    event.check_defs = classification.check_defs;
    event.check_exec_defs = classification.check_exec_defs;
    event.special_reg = classification.special_reg;
    event.section_offset = offset;
    event.instruction = inst.disassemble();
    const auto idx = Ops::counter_index(event.counter);
    if (event.kind == WaitEventKind::Smem)
      state_.pending_smem[idx] = true;
    const auto max_wait = Ops::maximum_dependency_wait(arch_, event.counter);
    if (max_wait.failed())
      return util::Result::failure();
    auto &ages = state_.pending_event_ages[idx].values;
    for (auto &age : ages) {
      if (age != kNoPendingEventAge && age < max_wait.value())
        ++age;
    }
    ages[static_cast<size_t>(event.kind)] = 0;
    for (auto &pending : state_.pending[idx]) {
      if (pending.min_younger < max_wait.value())
        ++pending.min_younger;
    }
    // Counter-only operations must age older requests without accumulating a
    // dependency record of their own (notably long sequences of stores).
    if (!event.regs.none() || event.check_exec_defs || event.special_reg)
      state_.pending[idx].push_back(std::move(event));
    return util::Result::success();
  }

  void update_committed_registers(const RegisterEffects &effects, const RegisterSet &async_defs) {
    if (!has_committed_generations(arch_))
      return;
    RegisterSet unavailable;
    for (const auto &pending : state_.pending) {
      for (const auto &event : pending) {
        if (event.produces_regs)
          unavailable |= event.regs - event.old_value_regs;
      }
    }
    local_ready_ |= vector_registers(effects.uses - unavailable);
    const auto synchronous = vector_registers(effects.defs - async_defs);
    state_.ready_regs |= synchronous;
    for (auto &pending : state_.pending) {
      for (auto &event : pending) {
        if (event.produces_regs)
          event.old_value_regs |= event.regs & synchronous;
      }
    }
  }

  void apply_translation_ordering(const RegisterEffects &effects,
                                  std::span<const ClassifiedEvent> current) {
    const bool vmem = std::ranges::any_of(current, [](const auto &event) {
      return event.counter == WaitCounterKind::X && Ops::is_xcnt_vmem_kind(event.kind);
    });
    const bool smem = std::ranges::any_of(current, [](const auto &event) {
      return event.counter == WaitCounterKind::X && event.kind == WaitEventKind::Smem;
    });
    if ((vmem && Ops::has_xcnt_smem(state_)) || (smem && Ops::has_xcnt_vmem(state_))) {
      // Switching translation groups implicitly drains the old group.
      Ops::apply_xcnt_wait(state_, 0);
      return;
    }
    if (!vmem)
      return;
    std::optional<uint32_t> count;
    for (const auto &event : state_.pending[Ops::counter_index(WaitCounterKind::X)]) {
      if (Ops::is_xcnt_vmem_event(event) && event.regs.intersects(effects.defs))
        count = count ? std::min(*count, event.min_younger) : event.min_younger;
    }
    // A newer VMEM result cannot overwrite an older VMEM translation input
    // before that translation completes. Other outstanding inputs remain live.
    if (count)
      Ops::apply_xcnt_wait(state_, *count);
  }
};
} // namespace

util::FailureOr<WaitcheckStreamReport>
analyze_waitcheck_stream(std::span<const uint32_t> words, rj_code_arch_t arch,
                         WaitcheckStreamOptions options,
                         const util::DiagnosticEmitter &emit_error) {
  if (waitcheck_detail::waitcnt_model(arch).failed())
    return emit_error.emit() << "unsupported waitcheck architecture";
  if ((options.wave_size != 32 && options.wave_size != 64) ||
      (has_committed_generations(arch) && options.wave_size != 64))
    return emit_error.emit() << "invalid waitcheck wave size";
  if (options.expert_scheduling && !waitcheck_detail::supports_expert_scheduling(arch))
    return emit_error.emit() << "expert scheduling is unavailable on this architecture";
  auto decoder = Decoder::create(arch);
  if (!decoder)
    return emit_error.emit() << "missing waitcheck ISA decoder";
  WaitcheckStreamReport report;
  StreamAnalyzer analyzer{arch, options, report};
  size_t index = 0;
  while (index < words.size()) {
    const uint64_t offset = index * sizeof(uint32_t);
    auto decoded = decoder->decode_window(words.subspan(index), offset, emit_error);
    if (decoded.failed())
      return util::Result::failure();
    const auto &inst = *decoded.value();
    auto events = waitcheck_detail::WaitcheckTarget::classify_events(inst, arch);
    if (events.failed())
      return util::Result::failure();
    if (auto reason = deferred_reason(inst, events.value())) {
      if (is_control_flow_transfer(inst) &&
          analyzer.check_control_flow_operands(inst, offset).failed())
        return emit_error.emit() << "waitcheck dependency check failed at byte " << offset;
      report.incomplete = WaitcheckStreamStop{offset, inst.disassemble(), std::string(*reason)};
      break;
    }
    // Classification describes target capability; the driver applies the entry mode.
    if (!options.expert_scheduling) {
      std::erase_if(events.value(), [](const auto &event) {
        return event.counter == WaitCounterKind::VmVsrc || event.counter == WaitCounterKind::VaVdst;
      });
    }
    if (analyzer.analyze(inst, events.value(), offset).failed())
      return emit_error.emit() << "waitcheck transfer failed at byte " << offset;
    ++report.instructions_analyzed;
    index += static_cast<size_t>(inst.size()) / sizeof(uint32_t);
    if (is_program_path_terminator(inst))
      break;
  }
  return report;
}
} // namespace rocjitsu
