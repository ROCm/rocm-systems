// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/wavefront_slot.h"

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <cassert>

namespace rocjitsu::amdgpu {

void WavefrontSlot::launch(Wavefront &wave, uint64_t arrival) {
  // Anything the previous occupant left in flight keeps its old generation
  // and is ignored when it returns.
  halt();
  wf = &wave;
  generation = wave.dispatch_generation();
  age = arrival;
  fetch_pc = wave.pc;
}

void WavefrontSlot::halt() {
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

bool WavefrontSlot::active() const {
  return wf && generation != 0 && wf->dispatch_generation() == generation && !wf->is_halted();
}

void WavefrontSlot::squash(bool flush) {
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

bool WavefrontSlot::has_hazard(const InstructionTiming &inst, uint64_t cycle) {
  std::erase_if(pending_writes_,
                [&](const PendingWrite &write) { return write.ready_cycle <= cycle; });
  for (const auto &write : pending_writes_)
    if (write.registers.intersects(inst.hazards))
      return true;
  return false;
}

void WavefrontSlot::reserve(const InstructionTiming &inst, uint64_t seq) {
  if (!inst.writes.none())
    pending_writes_.push_back({.registers = inst.writes, .seq = seq});
  memory_tokens += inst.counters.size;
  ++reserved;
}

void WavefrontSlot::cancel(uint64_t seq, uint8_t tokens) {
  assert(memory_tokens >= tokens && "cancelling more counter capacity than is reserved");
  memory_tokens -= tokens;
  // squash() has already dropped the write; this only guards a caller that
  // discards an instruction it never handed to the squash.
  std::erase_if(pending_writes_, [&](const PendingWrite &write) { return write.seq == seq; });
}

void WavefrontSlot::issued(uint64_t seq) {
  // Each instruction reserves at most one entry.
  for (auto &write : pending_writes_)
    if (write.seq == seq) {
      write.issued = true;
      return;
    }
}

void WavefrontSlot::written_back(uint64_t seq, uint64_t cycle) {
  for (auto &write : pending_writes_)
    if (write.seq == seq) {
      write.ready_cycle = cycle;
      return;
    }
}

void WavefrontSlot::issued_memory(const InstructionTiming &inst,
                                  const MemoryPipeline::WaitCounterTokens &charged) {
  assert(charged.size <= inst.counters.size && "charged more counters than were reserved");
  assert(memory_tokens >= inst.counters.size && "issued an instruction that was not reserved");
  for (uint8_t i = 0; i < charged.size; ++i)
    outstanding.increment(charged.types[i]);
  memory_tokens -= inst.counters.size - charged.size;
}

void WavefrontSlot::memory_returned(uint64_t seq, const MemoryPipeline::WaitCounterTokens &counters,
                                    uint64_t cycle) {
  assert(memory_tokens >= counters.size && "returned more counters than were issued");
  for (uint8_t i = 0; i < counters.size; ++i)
    outstanding.decrement(counters.types[i]);
  memory_tokens -= counters.size;
  written_back(seq, cycle);
}

bool WavefrontSlot::waiting_on_counters() {
  if (!waiting)
    return false;
  if (!waiting->satisfied(outstanding))
    return true;
  waiting.reset();
  return false;
}

WavefrontSlots::WavefrontSlots(ComputeUnitCore &cu, uint32_t cu_index)
    : cu_(cu), cu_index_(cu_index), slots_(cu.num_wf_slots()) {
  cu_.set_wave_reservation_handler([this](Wavefront &wave) { slot_reserved(wave); });
}

WavefrontSlots::~WavefrontSlots() { cu_.set_wave_reservation_handler(nullptr); }

void WavefrontSlots::slot_reserved(Wavefront &wave) {
  assert(wave.wf_id() < slots_.size() && "the CU reserved a slot it does not have");
  slots_[wave.wf_id()].launch(wave, next_arrival_++);
}

WavefrontSlot *WavefrontSlots::live(const WavefrontKey &key) {
  if (key.cu != cu_index_ || key.slot >= slots_.size())
    return nullptr;
  WavefrontSlot &slot = slots_[key.slot];
  return slot.active() && slot.generation == key.generation ? &slot : nullptr;
}

} // namespace rocjitsu::amdgpu
