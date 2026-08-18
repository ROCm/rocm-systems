// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Simulator-neutral state containers used by DataHazardEngine. Adapters own
// translation of simulator callbacks into these structures so the core engine
// remains frontend-agnostic.

#pragma once

#include "hash_utils.h"
#include "hazard_events.h"
#include "spin_lock.h"
#include "types.h"

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace hazard_core {

struct EngineWorkgroupKey {
  EntityId dispatch_id = 0;
  EntityId cluster_id = 0;
  EntityId workgroup_id = 0;

  bool operator==(const EngineWorkgroupKey &other) const noexcept {
    return dispatch_id == other.dispatch_id && cluster_id == other.cluster_id &&
           workgroup_id == other.workgroup_id;
  }
};

struct EngineWorkgroupKeyHash {
  size_t operator()(const EngineWorkgroupKey &key) const noexcept {
    size_t seed = 0;
    hash_combine(seed, key.dispatch_id);
    hash_combine(seed, key.cluster_id);
    hash_combine(seed, key.workgroup_id);
    return seed;
  }
};

struct EngineWaveKey {
  EntityId dispatch_id = 0;
  EntityId cluster_id = 0;
  EntityId workgroup_id = 0;
  EntityId wave_id = 0;

  bool operator==(const EngineWaveKey &other) const noexcept {
    return dispatch_id == other.dispatch_id && cluster_id == other.cluster_id &&
           workgroup_id == other.workgroup_id && wave_id == other.wave_id;
  }
};

struct EngineWaveKeyHash {
  size_t operator()(const EngineWaveKey &key) const noexcept {
    size_t seed = 0;
    hash_combine(seed, key.dispatch_id);
    hash_combine(seed, key.cluster_id);
    hash_combine(seed, key.workgroup_id);
    hash_combine(seed, key.wave_id);
    return seed;
  }
};

struct EngineLdsAccessRecord {
  EntityId wave_id = 0;
  EntityId instruction_id = 0;
  EntityId dispatch_id = 0;
  uint64_t pc = 0;
  uint32_t address = 0;
  uint32_t size = 0;
  bool is_write = false;
  bool is_atomic = false;
  std::array<uint32_t, 4> raw_isa{};
};

struct EngineInstructionContext {
  WaitAction wait_action{};
  InstructionHazardSemantics hazards{};
  EntityId instruction_id = 0;
  EntityId wave_id = 0;
  EntityId workgroup_id = 0;
  EntityId dispatch_id = 0;
  EntityId cluster_id = 0;
  EntityId wavegroup_id = 0;
  uint64_t pc = 0;
  std::array<uint32_t, 4> raw_isa{};
};

struct EngineWorkgroupState {
  SpinLock mutex;
  std::vector<EngineLdsAccessRecord> lds_epoch;
};

struct EngineWaveState {
  SpinLock mutex;
  WaveState core;
  std::unordered_map<EntityId, EngineInstructionContext> instruction_contexts;
  std::unordered_map<EntityId, std::array<uint32_t, 4>> pending_raw_isa;
  std::shared_ptr<EngineWorkgroupState> workgroup;
};

struct EngineWaveSnapshot {
  ExecutionKey key;
  WaveState core;
  size_t instruction_context_count = 0;
};

struct EngineLdsRaceKey {
  EntityId dispatch_id;
  EntityId cluster_id;
  EntityId workgroup_id;
  uint32_t address;
  hazard_core::EntityId wave_a;
  hazard_core::EntityId wave_b;

  bool operator==(const EngineLdsRaceKey &other) const noexcept {
    return dispatch_id == other.dispatch_id && cluster_id == other.cluster_id &&
           workgroup_id == other.workgroup_id && address == other.address &&
           wave_a == other.wave_a && wave_b == other.wave_b;
  }
};

struct EngineLdsRaceKeyHash {
  size_t operator()(const EngineLdsRaceKey &key) const noexcept {
    size_t seed = 0;
    hash_combine(seed, key.dispatch_id);
    hash_combine(seed, key.cluster_id);
    hash_combine(seed, key.workgroup_id);
    hash_combine(seed, key.address);
    hash_combine(seed, key.wave_a);
    hash_combine(seed, key.wave_b);
    return seed;
  }
};

struct EngineGlobalAccessInfo {
  EntityId dispatch_id = 0;
  EntityId cluster_id = 0;
  EntityId workgroup_id = 0;
  EntityId wave_id = 0;
  EntityId instruction_id = 0;
  uint64_t pc = 0;
  std::array<uint32_t, 4> raw_isa{};
  bool valid = false;
};

struct EngineGlobalShadowEntry {
  EngineGlobalAccessInfo writer;
  /// Readers held for two distinct workgroups, which is all a write needs: if
  /// its own workgroup fills the first slot, the second is by construction a
  /// different one. A single slot would let the workgroup that read last
  /// overwrite the others and hide its own conflict with them. One entry exists
  /// per four bytes of a dispatch, so readers are not retained per workgroup.
  std::array<EngineGlobalAccessInfo, 2> readers;
  bool race_reported = false;
};

struct EngineGlobalShadowKey {
  EntityId dispatch_id = 0;
  uint64_t address = 0;

  bool operator==(const EngineGlobalShadowKey &other) const noexcept {
    return dispatch_id == other.dispatch_id && address == other.address;
  }
};

struct EngineGlobalShadowKeyHash {
  size_t operator()(const EngineGlobalShadowKey &key) const noexcept {
    size_t seed = 0;
    hash_combine(seed, key.dispatch_id);
    hash_combine(seed, key.address);
    return seed;
  }
};

struct EngineWarning {
  HazardFinding finding;
  std::array<uint32_t, 4> source_raw_isa{};
  uint64_t source_pc = 0;
  bool has_source = false;
  std::string message;
  std::string suggestion;
};

class HazardWarningSink {
public:
  virtual ~HazardWarningSink() = default;
  virtual void emit_warning(const EngineWarning &warning) const = 0;
};

struct EngineState {
  mutable std::shared_mutex mutex;
  std::atomic<uint64_t> wave_cache_generation{0};
  std::unordered_map<ExecutionKey, std::shared_ptr<EngineWaveState>, ExecutionKeyHash> waves;
  std::unordered_map<EngineWorkgroupKey, std::shared_ptr<EngineWorkgroupState>,
                     EngineWorkgroupKeyHash>
      workgroups;
  std::unordered_map<EngineWaveKey, ExecutionKey, EngineWaveKeyHash> wave_index;
  std::unordered_map<ExecutionKey, EngineWorkgroupKey, ExecutionKeyHash> wave_to_workgroup;
  std::unordered_set<EngineLdsRaceKey, EngineLdsRaceKeyHash> reported_lds_races;
  std::unordered_map<EngineGlobalShadowKey, EngineGlobalShadowEntry, EngineGlobalShadowKeyHash>
      global_shadow;
  std::vector<EngineWarning> warnings;

  void reset();
};

EngineInstructionContext make_engine_instruction_context(const InstructionEvent &instruction);
EngineInstructionContext make_engine_instruction_context(const InstructionDescriptor &instruction);
void track_instruction_context(EngineWaveState &wave, const InstructionEvent &instruction);
const EngineInstructionContext &get_instruction_context(const EngineWaveState &wave,
                                                        EntityId instruction_id);

std::array<uint32_t, 4> get_pending_raw_isa(const EngineWaveState &wave, EntityId instruction_id);
void prune_pending_raw_isa(EngineWaveState &wave);

} // namespace hazard_core
