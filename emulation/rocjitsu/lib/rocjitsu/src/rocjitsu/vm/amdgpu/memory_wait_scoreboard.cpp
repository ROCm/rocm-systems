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
  orders_.clear();
  last_order_.fill(kUnordered);
  xcnt_scalar_ = false;
  translations_.clear();
}

uint64_t MemoryWaitScoreboard::issue(WaitCounterKind counter, bool unordered,
                                     uint32_t ordering_kind, uint32_t units,
                                     bool backpressure_ordered) {
  auto i = static_cast<size_t>(counter);
  ordering_kinds_[i] |= ordering_kind;
  unordered_[i] |= unordered || std::popcount(ordering_kinds_[i]) > 1;
  issued_[i] += units;
  last_order_[i] = kUnordered;
  if (!unordered && backpressure_ordered) {
    const auto it = std::ranges::find_if(orders_, [&](const Order &order) {
      return order.counter == counter && order.kind == ordering_kind;
    });
    const auto slot = static_cast<uint16_t>(it - orders_.begin());
    if (it == orders_.end())
      orders_.push_back({counter, ordering_kind});
    orders_[slot].issued += units;
    last_order_[i] = slot;
  }
  return issued_[i];
}

uint64_t MemoryWaitScoreboard::issue(const waitcheck_detail::ClassifiedEvent &event,
                                     rj_code_arch_t arch, uint32_t units) {
  using namespace waitcheck_detail;
  const auto model = waitcnt_model(arch).value();
  const bool legacy_flat =
      model == WaitcntModel::LegacyNoVscnt &&
      (event.kind == WaitEventKind::FlatLoad || event.kind == WaitEventKind::FlatStore);
  uint32_t kind = 0;
  if (!(model == WaitcntModel::LegacyNoVscnt && event.counter == WaitCounterKind::Load))
    if (auto normalized =
            WaitcheckTarget::normalized_hardware_event_kind(event.counter, event.kind, model))
      kind = uint32_t{1} << static_cast<unsigned>(*normalized);
  // ASYNC loads and stores share a counter but return done out of order with
  // respect to each other (CDNA5 ISA 10.8). Barrier arrive orders with loads.
  if (event.counter == WaitCounterKind::Async)
    kind = uint32_t{1} << static_cast<unsigned>(event.kind == WaitEventKind::AsyncLdsStore
                                                    ? WaitEventKind::AsyncLdsStore
                                                    : WaitEventKind::AsyncLdsLoad);
  // Older RDNA samples and BVH operations share VMcnt, not a completion FIFO.
  if (model == WaitcntModel::LegacyVscnt && event.counter == WaitCounterKind::Load &&
      (event.kind == WaitEventKind::Sample || event.kind == WaitEventKind::Bvh))
    kind = uint32_t{1} << static_cast<unsigned>(event.kind);
  const bool ordered =
      event.kind != WaitEventKind::Gds && event.kind != WaitEventKind::Export &&
      event.kind != WaitEventKind::SqMessage && event.kind != WaitEventKind::SccWrite &&
      event.kind != WaitEventKind::GlobalInv && event.kind != WaitEventKind::Unknown;
  return issue(event.counter, !ordered || event.kind == WaitEventKind::Smem || legacy_flat, kind,
               units, ordered);
}

bool MemoryWaitScoreboard::completed(WaitCounterKind counter, uint64_t sequence, uint16_t order,
                                     uint64_t order_sequence) const {
  return sequence <= retired_[static_cast<size_t>(counter)] ||
         (order != kUnordered && order_sequence <= orders_[order].retired);
}

void MemoryWaitScoreboard::backpressure(WaitCounterKind counter, uint32_t capacity) {
  assert(capacity != 0);
  bool changed = false;
  for (auto &order : orders_) {
    if (order.counter != counter || order.issued < capacity)
      continue;
    // Even an unordered incoming operation needs one counter slot. If an old
    // result still had capacity-1 younger operations in its own ordered class,
    // none could have completed and that slot could not be available.
    const auto through = order.issued - (capacity - 1);
    changed |= through > order.retired;
    order.retired = std::max(order.retired, through);
  }
  if (changed)
    retire_completed();
}

void MemoryWaitScoreboard::stamp_order(Event &event) const {
  const auto counter = static_cast<size_t>(event.counter);
  if (event.sequence == issued_[counter] && last_order_[counter] != kUnordered) {
    event.order = last_order_[counter];
    event.order_sequence = orders_[event.order].issued;
  }
}

uint32_t MemoryWaitScoreboard::required_wait(const Event &event) const {
  const auto i = static_cast<size_t>(event.counter);
  const auto required = !unordered_[i] ? issued_[i] - event.sequence
                        : event.order != kUnordered
                            ? orders_[event.order].issued - event.order_sequence
                            : 0;
  return static_cast<uint32_t>(std::min<uint64_t>(required, UINT32_MAX));
}

void MemoryWaitScoreboard::add(Event event) {
  if (!event.lanes || !event.bytes || !event.reg.width)
    return;
  stamp_order(event);
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
  const uint64_t through = issued_[i] > threshold ? issued_[i] - threshold : 0;
  if (!threshold || !unordered_[i])
    retired_[i] = std::max(retired_[i], through);
  // A mixed counter still bounds every ordered class independently. It does
  // not prove readiness of an unordered result such as SMEM.
  for (auto &order : orders_)
    if (order.counter == counter && order.issued > threshold)
      order.retired = std::max(order.retired, order.issued - threshold);
  if (!threshold) {
    unordered_[i] = false;
    ordering_kinds_[i] = 0;
    for (auto &order : orders_)
      if (order.counter == counter)
        order.retired = order.issued;
  }
  retire_completed();
  if (counter == WaitCounterKind::X) {
    std::erase_if(translations_, [&](const auto &entry) { return entry.sequence <= through; });
  } else if (xcnt_scalar_ && counter == WaitCounterKind::Km && !threshold) {
    wait(WaitCounterKind::X, 0);
  }
}

void MemoryWaitScoreboard::retire_completed() {
  const auto size = events_.size();
  std::erase_if(events_, [&](const Event &event) {
    const bool retire = completed(event.counter, event.sequence, event.order, event.order_sequence);
    if (retire)
      clear_destination(event);
    return retire;
  });
  if (size != events_.size())
    rebuild_mask();
  if (!xcnt_scalar_ && outstanding(WaitCounterKind::X)) {
    // Completion proves translation. Map the waited counter position back
    // into the ordered X queue; counts from mixed families are not X ages.
    uint64_t translated = 0;
    for (const auto &entry : translations_)
      if (completed(entry.completion, entry.completion_sequence, entry.order, entry.order_sequence))
        translated = std::max(translated, entry.sequence);
    if (translated > retired_[static_cast<size_t>(WaitCounterKind::X)])
      wait(WaitCounterKind::X,
           static_cast<uint32_t>(issued_[static_cast<size_t>(WaitCounterKind::X)] - translated));
  }
}

void MemoryWaitScoreboard::xcnt_group(bool scalar) {
  if (xcnt_scalar_ != scalar && outstanding(WaitCounterKind::X))
    wait(WaitCounterKind::X, 0);
  xcnt_scalar_ = scalar;
}

uint64_t MemoryWaitScoreboard::issue_xcnt(std::optional<WaitCounterKind> completion, bool scalar) {
  const auto sequence = issue(WaitCounterKind::X, scalar);
  if (!scalar && completion) {
    const auto i = static_cast<size_t>(*completion);
    const auto order = last_order_[i];
    translations_.push_back({sequence, issued_[i], *completion, order,
                             order == kUnordered ? 0 : orders_[order].issued});
  }
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
  // Most kernels wait well before a queue is full. Avoid classifying the
  // incoming instruction twice unless some ordered class can force progress.
  const bool full =
      inst.is_memory_wait_producer() && std::ranges::any_of(orders_, [&](const Order &order) {
        const auto maximum = WaitcheckTarget::maximum_dependency_wait(arch, order.counter);
        return maximum.succeeded() && order.issued - order.retired > maximum.value();
      });
  if (full) {
    const auto events = WaitcheckTarget::classify_events(inst, arch);
    if (events.succeeded())
      for (const auto &event : events.value()) {
        if (event.counter == WaitCounterKind::VmVsrc || event.counter == WaitCounterKind::VaVdst ||
            event.counter == WaitCounterKind::Depctr)
          continue;
        const auto maximum = WaitcheckTarget::maximum_dependency_wait(arch, event.counter);
        if (maximum.succeeded())
          backpressure(event.counter, maximum.value() + 1);
      }
  }
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
      reporter_(context_, {event, pc_, reg, write, required_wait(event)});
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
