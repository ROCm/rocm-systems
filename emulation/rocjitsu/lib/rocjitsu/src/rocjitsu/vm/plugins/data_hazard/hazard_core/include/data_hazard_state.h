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

/// Identifies one wavegroup, the level of the hierarchy between a workgroup and
/// a wave whose waves synchronize with semaphores instead of a full barrier.
struct EngineWavegroupKey {
  EntityId dispatch_id = 0;
  EntityId cluster_id = 0;
  EntityId workgroup_id = 0;
  EntityId wavegroup_id = 0;

  bool operator==(const EngineWavegroupKey &other) const noexcept {
    return dispatch_id == other.dispatch_id && cluster_id == other.cluster_id &&
           workgroup_id == other.workgroup_id && wavegroup_id == other.wavegroup_id;
  }
};

struct EngineWavegroupKeyHash {
  size_t operator()(const EngineWavegroupKey &key) const noexcept {
    size_t seed = 0;
    hash_combine(seed, key.dispatch_id);
    hash_combine(seed, key.cluster_id);
    hash_combine(seed, key.workgroup_id);
    hash_combine(seed, key.wavegroup_id);
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
  /// Lets the workgroup detector leave pairs from one wavegroup to the
  /// wavegroup detector, which knows the semaphores that order them. Zero means
  /// the access belongs to no wavegroup and stays workgroup-scoped.
  EntityId wavegroup_id = 0;
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
  /// One LDS epoch per wavegroup, closed by that wavegroup's s_sema_wait or by
  /// the wavegroup ending. Held here rather than in a wavegroup of their own
  /// because they share the workgroup's lock and lifetime, and kept apart from
  /// lds_epoch because a semaphore closes only its own wavegroup's accesses.
  std::unordered_map<EntityId, std::vector<EngineLdsAccessRecord>> wavegroup_lds_epochs;
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

/// Dedup key for wavegroup-scope LDS races. Distinct from EngineLdsRaceKey so
/// that a workgroup-scope and a wavegroup-scope report about the same address
/// and wave pair do not suppress one another: they name different missing
/// synchronization, so both are worth reporting.
struct EngineWavegroupLdsRaceKey {
  EntityId dispatch_id;
  EntityId cluster_id;
  EntityId workgroup_id;
  EntityId wavegroup_id;
  uint32_t address;
  EntityId wave_a;
  EntityId wave_b;

  bool operator==(const EngineWavegroupLdsRaceKey &other) const noexcept {
    return dispatch_id == other.dispatch_id && cluster_id == other.cluster_id &&
           workgroup_id == other.workgroup_id && wavegroup_id == other.wavegroup_id &&
           address == other.address && wave_a == other.wave_a && wave_b == other.wave_b;
  }
};

struct EngineWavegroupLdsRaceKeyHash {
  size_t operator()(const EngineWavegroupLdsRaceKey &key) const noexcept {
    size_t seed = 0;
    hash_combine(seed, key.dispatch_id);
    hash_combine(seed, key.cluster_id);
    hash_combine(seed, key.workgroup_id);
    hash_combine(seed, key.wavegroup_id);
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
  /// Bytes of the shadow entry this access covers, one bit per byte. An entry
  /// spans four bytes, so accesses narrower than a dword share one without
  /// addressing the same memory; they conflict only where their masks meet.
  uint8_t byte_mask = 0;
  bool valid = false;
};

/// Accesses held for two distinct workgroups, which is all a conflicting access
/// needs: if its own workgroup fills the first slot, the second is by
/// construction a different one. A single slot would let the workgroup that
/// accessed last overwrite the others and hide its own conflict with them. One
/// entry exists per four bytes of a dispatch, so accesses are not retained per
/// workgroup; the byte masks keep disjoint ones inside an entry apart.
using EngineGlobalAccessSlots = std::array<EngineGlobalAccessInfo, 2>;

struct EngineGlobalShadowEntry {
  EngineGlobalAccessSlots writers;
  EngineGlobalAccessSlots readers;
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
  std::unordered_set<EngineWavegroupLdsRaceKey, EngineWavegroupLdsRaceKeyHash>
      reported_wavegroup_lds_races;
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
