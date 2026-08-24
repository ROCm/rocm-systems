// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "data_hazard_state.h"
#include "detail/access_handling.h"

#include <algorithm>
#include <ranges>
#include <unordered_set>

namespace hazard_core {

namespace {

bool has_raw_isa(const std::array<uint32_t, 4> &raw_isa) {
  return std::ranges::any_of(raw_isa, [](uint32_t word) { return word != 0; });
}

void merge_hazards(InstructionHazardSemantics &current, const InstructionHazardSemantics &update) {
  if (update.vector_write_wait != WaitCntType::NONE)
    current.vector_write_wait = update.vector_write_wait;
  if (update.vector_read_wait != WaitCntType::NONE)
    current.vector_read_wait = update.vector_read_wait;
  if (update.vector_write_also_waits_lds)
    current.vector_write_also_waits_lds = true;
  if (update.scalar_write_wait != WaitCntType::NONE)
    current.scalar_write_wait = update.scalar_write_wait;
  if (update.local_write_wait != WaitCntType::NONE)
    current.local_write_wait = update.local_write_wait;
  if (update.memory_op_wait != WaitCntType::NONE)
    current.memory_op_wait = update.memory_op_wait;
}

void merge_instruction_context(EngineInstructionContext &current,
                               const EngineInstructionContext &update) {
  if (wait_action_has_effect(update.wait_action))
    current.wait_action = update.wait_action;
  merge_hazards(current.hazards, update.hazards);

  if (update.pc != 0)
    current.pc = update.pc;
  if (has_raw_isa(update.raw_isa))
    current.raw_isa = update.raw_isa;

  current.instruction_id = update.instruction_id;
  current.wave_id = update.wave_id;
  current.workgroup_id = update.workgroup_id;
  current.dispatch_id = update.dispatch_id;
  current.cluster_id = update.cluster_id;
  current.wavegroup_id = update.wavegroup_id;
}

} // namespace

void EngineState::reset() {
  std::lock_guard<std::shared_mutex> lock(mutex);
  waves.clear();
  workgroups.clear();
  wave_index.clear();
  wave_to_workgroup.clear();
  reported_lds_races.clear();
  reported_wavegroup_lds_races.clear();
  global_shadow.clear();
  warnings.clear();
  wave_cache_generation.fetch_add(1, std::memory_order_release);
}

EngineInstructionContext make_engine_instruction_context(const InstructionEvent &instruction) {
  EngineInstructionContext ctx = make_engine_instruction_context(instruction.instruction);
  ctx.wait_action = instruction.wait_action;
  ctx.hazards = instruction.hazards;
  return ctx;
}

EngineInstructionContext make_engine_instruction_context(const InstructionDescriptor &instruction) {
  EngineInstructionContext ctx;
  ctx.instruction_id = instruction.instruction_id;
  ctx.wave_id = instruction.execution.wave_id;
  ctx.workgroup_id = instruction.execution.workgroup_id;
  ctx.dispatch_id = instruction.execution.dispatch_id;
  ctx.cluster_id = instruction.execution.cluster_id;
  ctx.wavegroup_id = instruction.execution.wavegroup_id;
  ctx.pc = instruction.pc;
  for (int i = 0; i < 4; ++i)
    ctx.raw_isa[i] = instruction.raw_isa[i];
  return ctx;
}

void track_instruction_context(EngineWaveState &wave, const InstructionEvent &instruction) {
  EngineInstructionContext ctx = make_engine_instruction_context(instruction);
  auto [it, inserted] = wave.instruction_contexts.emplace(ctx.instruction_id, ctx);
  if (!inserted)
    merge_instruction_context(it->second, ctx);
}

const EngineInstructionContext &get_instruction_context(const EngineWaveState &wave,
                                                        EntityId instruction_id) {
  static const EngineInstructionContext kEmpty{};
  auto it = wave.instruction_contexts.find(instruction_id);
  if (it == wave.instruction_contexts.end())
    return kEmpty;
  return it->second;
}

std::array<uint32_t, 4> get_pending_raw_isa(const EngineWaveState &wave, EntityId instruction_id) {
  auto it = wave.pending_raw_isa.find(instruction_id);
  if (it == wave.pending_raw_isa.end())
    return {0, 0, 0, 0};
  return it->second;
}

void prune_pending_raw_isa(EngineWaveState &wave) {
  std::unordered_set<EntityId> active_ids;
  active_ids.reserve(wave.pending_raw_isa.size());

  for_each_pending_instruction_id(&wave.core, [&](EntityId id) { active_ids.insert(id); });

  for (auto it = wave.pending_raw_isa.begin(); it != wave.pending_raw_isa.end();) {
    if (active_ids.find(it->first) == active_ids.end()) {
      it = wave.pending_raw_isa.erase(it);
    } else {
      ++it;
    }
  }
}

} // namespace hazard_core
