// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/wave_slot.h"

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <cassert>

namespace rocjitsu::amdgpu {

void WaveSlot::launch(Wavefront &wave, uint64_t arrival) {
  // Anything the previous occupant left in flight keeps its old generation
  // and is ignored when it returns.
  halt();
  wf = &wave;
  generation = wave.dispatch_generation();
  age = arrival;
}

void WaveSlot::halt() {
  ++epoch;
  buffer.clear();
  reserved = 0;
  fetch_pending = false;
  fetch_stopped = false;
  in_ready_list = false;
  waiting.reset();
  outstanding = {};
  memory_tokens = 0;
  pending_writes_.clear();
  generation = 0;
}

bool WaveSlot::active() const {
  return wf && generation != 0 && wf->dispatch_generation() == generation && !wf->is_halted();
}

void WaveSlot::squash(bool flush) {
  ++epoch;
  reserved = 0;
  in_ready_list = false;
  // Instructions still on their way to issue keep their resource reservations
  // until issue discards them; their register reservations go now.
  std::erase_if(pending_writes_, [](const PendingWrite &write) { return !write.issued; });
  if (flush) {
    buffer.clear();
    fetch_pending = false;
    fetch_stopped = false;
  }
}

bool WaveSlot::has_hazard(const InstructionTiming &inst, uint64_t cycle) {
  std::erase_if(pending_writes_,
                [&](const PendingWrite &write) { return write.ready_cycle <= cycle; });
  for (const auto &write : pending_writes_)
    if (write.registers.intersects(inst.hazards))
      return true;
  return false;
}

void WaveSlot::reserve(const InstructionTiming &inst, uint64_t seq) {
  if (!inst.writes.none())
    pending_writes_.push_back({.registers = inst.writes, .seq = seq});
  memory_tokens += inst.counters.size;
  ++reserved;
}

void WaveSlot::cancel(uint64_t seq, uint8_t tokens) {
  memory_tokens -= tokens;
  std::erase_if(pending_writes_, [&](const PendingWrite &write) { return write.seq == seq; });
}

void WaveSlot::issued(uint64_t seq) {
  // Each instruction reserves at most one entry.
  for (auto &write : pending_writes_)
    if (write.seq == seq) {
      write.issued = true;
      return;
    }
}

void WaveSlot::written_back(uint64_t seq, uint64_t cycle) {
  for (auto &write : pending_writes_)
    if (write.seq == seq) {
      write.ready_cycle = cycle;
      return;
    }
}

void WaveSlot::issued_memory(const InstructionTiming &inst) {
  for (uint8_t i = 0; i < inst.counters.size; ++i)
    outstanding.increment(inst.counters.types[i]);
}

void WaveSlot::memory_returned(uint64_t seq, const MemoryPipeline::WaitCounterTokens &counters,
                               uint64_t cycle) {
  for (uint8_t i = 0; i < counters.size; ++i)
    outstanding.decrement(counters.types[i]);
  memory_tokens -= counters.size;
  written_back(seq, cycle);
}

bool WaveSlot::waiting_on_counters() {
  if (!waiting)
    return false;
  if (!waiting->satisfied(outstanding))
    return true;
  waiting.reset();
  return false;
}

WaveSlots::WaveSlots(ComputeUnitCore &cu, uint32_t cu_index)
    : cu_(cu), cu_index_(cu_index), slots_(cu.num_wf_slots()) {
  cu_.set_wave_launch_handler([this](Wavefront &wave) { launch(wave); });
}

WaveSlots::~WaveSlots() { cu_.set_wave_launch_handler(nullptr); }

void WaveSlots::launch(Wavefront &wave) {
  assert(wave.wf_id() < slots_.size() && "a wave launched into a slot the CU does not have");
  slots_[wave.wf_id()].launch(wave, next_arrival_++);
}

WaveSlot *WaveSlots::live(const WaveKey &key) {
  if (key.cu != cu_index_ || key.slot >= slots_.size())
    return nullptr;
  WaveSlot &slot = slots_[key.slot];
  return slot.active() && slot.generation == key.generation ? &slot : nullptr;
}

} // namespace rocjitsu::amdgpu
