// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/memory_wait_scoreboard.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_read.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <bit>

namespace rocjitsu::amdgpu {

std::optional<RegisterRef> RegisterAccess::source_register(const Operand &op) const {
  if (!op.reads_value() || op.size_bits() == 0 || op.const_value())
    return std::nullopt;
  const auto &wf = wavefront();
  const auto width = static_cast<uint8_t>(std::max(1, op.size_bits() / 32));
  if (auto base = op.simd_vgpr_base(wf)) {
    if (!cu_->owns_vgpr_range(wf, *base, width))
      return std::nullopt;
    return RegisterRef{RegClass::VGPR, static_cast<uint16_t>(*base - wf.vgpr_alloc().base), width};
  }
  if (auto reg = op.to_register_ref())
    return reg;
  if (auto special = op.to_special_reg_class())
    return RegisterRef{*special, 0, width};
  if (!op.is_fieldless() && !op.is_vgpr())
    if (auto range = resolve_scalar_register_range(wf, op.encoding_value(), width))
      return range->register_ref();
  return std::nullopt;
}

void check_active_memory_wait(RegisterRef reg, uint64_t lanes, uint8_t bytes, bool write) {
  if (write && reg.cls == RegClass::VCC && suppress_memory_wait_vcc_write)
    return;
  if (active_memory_wait_check)
    active_memory_wait_check->access(reg, lanes, bytes, write);
}

void MemoryWaitScoreboard::clear() {
  events_.clear();
  pending_.reset();
  issued_.fill(0);
  retired_.fill(0);
  unordered_.fill(false);
  ordering_kinds_.fill(0);
  xcnt_scalar_ = false;
  translations_.clear();
}

uint64_t MemoryWaitScoreboard::issue(WaitCounterKind counter, bool unordered,
                                     uint32_t ordering_kind, uint32_t units) {
  auto i = static_cast<size_t>(counter);
  ordering_kinds_[i] |= ordering_kind;
  unordered_[i] |= unordered || std::popcount(ordering_kinds_[i]) > 1;
  issued_[i] += units;
  return issued_[i];
}

void MemoryWaitScoreboard::add(Event event) {
  if (!event.lanes || !event.bytes || !event.reg.width)
    return;
  // A newer result covering the entire old destination is the stronger
  // dependency on an ordered counter. Discard the superseded record rather
  // than growing with a loop that repeatedly loads an unused destination.
  bool overlap = false;
  for (unsigned r = 0; r < event.reg.width; ++r) {
    auto reg = event.reg;
    reg.index += r;
    const auto i = index(reg);
    overlap |= i < kRegisters && pending_.test(i, true);
  }
  if (overlap)
    std::erase_if(events_, [&](const Event &old) {
      return old.counter == event.counter && old.reg.cls == event.reg.cls &&
             event.reg.index <= old.reg.index &&
             event.reg.index + event.reg.width >= old.reg.index + old.reg.width &&
             (event.lanes & old.lanes) == old.lanes && (event.bytes & old.bytes) == old.bytes;
    });
  events_.push_back(event);
  for (unsigned r = 0; r < event.reg.width; ++r) {
    auto reg = event.reg;
    reg.index += r;
    if (auto i = index(reg); i < kRegisters)
      pending_.set(i, event.counter == WaitCounterKind::X ? MemoryWaitShadow::kReplaySource
                                                          : MemoryWaitShadow::kResult);
  }
}

void MemoryWaitScoreboard::clear_destination(const Event &event) {
  for (unsigned r = 0; r < event.reg.width; ++r) {
    auto reg = event.reg;
    reg.index += r;
    if (auto i = index(reg); i < kRegisters)
      pending_.clear(i);
  }
}

void MemoryWaitScoreboard::rebuild_mask() {
  // Erasing a record cleared its destinations. Restore any overlapping records
  // that remain; unrelated shadow bytes already have the right value.
  for (const auto &event : events_)
    for (unsigned r = 0; r < event.reg.width; ++r) {
      auto reg = event.reg;
      reg.index += r;
      if (auto i = index(reg); i < kRegisters)
        pending_.set(i, event.counter == WaitCounterKind::X ? MemoryWaitShadow::kReplaySource
                                                            : MemoryWaitShadow::kResult);
    }
}

void MemoryWaitScoreboard::wait(WaitCounterKind counter, uint32_t threshold) {
  // The caller normalizes split and legacy spellings into one sequence
  // domain for each architectural counter.
  const size_t i = static_cast<size_t>(counter);
  if (threshold && unordered_[i])
    return;
  const uint64_t through = issued_[i] > threshold ? issued_[i] - threshold : 0;
  retired_[i] = std::max(retired_[i], through);
  const auto size = events_.size();
  std::erase_if(events_, [&](const Event &event) {
    const bool retire = event.counter == counter && event.sequence <= through;
    if (retire)
      clear_destination(event);
    return retire;
  });
  if (!threshold) {
    unordered_[i] = false;
    ordering_kinds_[i] = 0;
  }
  if (size != events_.size())
    rebuild_mask();
  if (counter == WaitCounterKind::X) {
    std::erase_if(translations_, [&](const auto &entry) { return entry.sequence <= through; });
  } else if (outstanding(WaitCounterKind::X)) {
    if (xcnt_scalar_) {
      if (counter == WaitCounterKind::Km && !threshold)
        wait(WaitCounterKind::X, 0);
    } else {
      // Completion proves translation. Map the waited counter position back
      // into the ordered X queue; counts from mixed families are not X ages.
      uint64_t translated = 0;
      for (const auto &entry : translations_)
        if (entry.completion == counter && entry.completion_sequence <= retired_[i])
          translated = std::max(translated, entry.sequence);
      if (translated)
        wait(WaitCounterKind::X,
             static_cast<uint32_t>(issued_[static_cast<size_t>(WaitCounterKind::X)] - translated));
    }
  }
}

void MemoryWaitScoreboard::xcnt_group(bool scalar) {
  if (xcnt_scalar_ != scalar && outstanding(WaitCounterKind::X))
    wait(WaitCounterKind::X, 0);
  xcnt_scalar_ = scalar;
}

uint64_t MemoryWaitScoreboard::issue_xcnt(WaitCounterKind completion, bool scalar) {
  const auto sequence = issue(WaitCounterKind::X, scalar);
  if (!scalar)
    translations_.push_back({sequence, issued_[static_cast<size_t>(completion)], completion});
  return sequence;
}

void MemoryWaitScoreboard::xcnt_ordered_write(RegisterRef reg, uint64_t lanes, uint8_t bytes) {
  if (xcnt_scalar_ || !pending_.pending(reg, true))
    return;
  uint64_t through = 0;
  for (const auto &event : events_)
    if (event.counter == WaitCounterKind::X && event.reg.cls == reg.cls && (event.lanes & lanes) &&
        (event.bytes & bytes) && reg.index < event.reg.index + event.reg.width &&
        event.reg.index < reg.index + reg.width)
      through = std::max(through, event.sequence);
  if (through)
    wait(WaitCounterKind::X,
         static_cast<uint32_t>(issued_[static_cast<size_t>(WaitCounterKind::X)] - through));
}

void MemoryWaitScoreboard::before(const Instruction &inst, rj_code_arch_t arch) {
  using namespace waitcheck_detail;
  if (arch == ROCJITSU_CODE_ARCH_CDNA5 && outstanding(WaitCounterKind::X) &&
      WaitcheckTarget::is_xcnt_drain(inst))
    wait(WaitCounterKind::X, 0);
  const auto fields = inst.is_waitcnt() ? WaitcheckTarget::explicit_wait_fields(inst, arch)
                                        : WaitcheckTarget::embedded_wait_fields(inst, arch);
  if (fields.failed() || !fields.value())
    return;
  for (auto counter :
       {WaitCounterKind::Load, WaitCounterKind::Store, WaitCounterKind::Ds, WaitCounterKind::Km,
        WaitCounterKind::Exp, WaitCounterKind::Sample, WaitCounterKind::Bvh, WaitCounterKind::Async,
        WaitCounterKind::Tensor, WaitCounterKind::X})
    if (const auto value = (*fields.value())[static_cast<size_t>(counter)])
      wait(counter, *value);
}

void MemoryWaitScoreboard::access_pending(RegisterRef reg, uint64_t lanes, uint8_t bytes,
                                          bool write, WaitCounterKind ordered_write_counter) {
  for (auto &event : events_) {
    // The writeback shortcut is valid only while this queue remains ordered.
    // FLAT or mixed event kinds must retain pending overwrite protection.
    if (event.reported || (!write && event.counter == WaitCounterKind::X) ||
        (write && event.counter == ordered_write_counter &&
         !unordered_[static_cast<size_t>(event.counter)]) ||
        event.reg.cls != reg.cls || !(event.lanes & lanes) || !(event.bytes & bytes) ||
        reg.index >= event.reg.index + event.reg.width || event.reg.index >= reg.index + reg.width)
      continue;
    event.reported = true;
    if (reporter_) {
      const auto i = static_cast<size_t>(event.counter);
      const auto required = unordered_[i] ? 0 : issued_[i] - event.sequence;
      reporter_(context_, {event, pc_, reg, write,
                           static_cast<uint32_t>(std::min<uint64_t>(required, UINT32_MAX))});
    }
  }
  const auto size = events_.size();
  std::erase_if(events_, [&](const Event &event) {
    if (event.reported)
      clear_destination(event);
    return event.reported;
  });
  if (size != events_.size())
    rebuild_mask();
}
} // namespace rocjitsu::amdgpu
