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
  /// Only two kinds of instruction are still worth a context: the one whose
  /// accesses are arriving now, and those a pending operation can still be
  /// reported against. Everything else is pruned, so a loop that issues
  /// millions of instructions costs a map sized by what is in flight rather
  /// than by how long the wave has run.
  std::unordered_map<EntityId, EngineInstructionContext> instruction_contexts;
  std::unordered_map<EntityId, std::array<uint32_t, 4>> pending_raw_isa;
  EntityId current_instruction_id = 0;
  /// Contexts left by the last prune. Pruning waits for the map to grow a whole
  /// slack past it, so a wave holding many instructions in flight prunes just as
  /// rarely as one holding none: measuring the growth against anything else
  /// risks a map that is already at its bound asking for a prune per
  /// instruction, each one walking every pending queue.
  size_t contexts_at_last_prune = 0;
  std::shared_ptr<EngineWorkgroupState> workgroup;
};

struct EngineWaveSnapshot {
  ExecutionKey key;
  WaveState core;
  /// Contexts the wave is still holding, not instructions it has run: retired
  /// ones are pruned, so this tracks what is in flight.
  size_t instruction_context_count = 0;
};

/// One report per racing instruction pair, per wave pair and address, for the
/// life of the dispatch. The instructions are identified by PC because an
/// instruction id is a per-execution ordinal: keying on ids would report every
/// loop iteration, while keying on neither hides a second defect between waves
/// that already raced somewhere else.
struct EngineLdsRaceKey {
  EntityId dispatch_id;
  EntityId cluster_id;
  EntityId workgroup_id;
  uint32_t address;
  hazard_core::EntityId wave_a;
  hazard_core::EntityId wave_b;
  uint64_t producer_pc;
  uint64_t consumer_pc;

  bool operator==(const EngineLdsRaceKey &other) const noexcept {
    return dispatch_id == other.dispatch_id && cluster_id == other.cluster_id &&
           workgroup_id == other.workgroup_id && address == other.address &&
           wave_a == other.wave_a && wave_b == other.wave_b && producer_pc == other.producer_pc &&
           consumer_pc == other.consumer_pc;
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
    hash_combine(seed, key.producer_pc);
    hash_combine(seed, key.consumer_pc);
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
  /// Whether the access wrote the bytes it covers, which is what decides
  /// whether a later read conflicts with it. An atomic read-modify-write
  /// counts as a write.
  bool is_write = false;
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
  /// Atomic accesses, kept in slots of their own. They have to be retained or
  /// an ordinary access arriving after one has nothing to conflict with, and
  /// they have to stay apart from the ordinary history or an atomic would
  /// displace the ordinary access of its own workgroup and excuse a later
  /// atomic that the ordinary access still races.
  EngineGlobalAccessSlots atomics;
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

/// Waits are the natural moment to prune, but a wave can run a long stretch
/// without one, so tracking also prunes once the contexts it has kept outgrow
/// what is in flight by this much. Walking the pending queues is the cost of a
/// prune, and this slack keeps that off the per-instruction path.
inline constexpr size_t kInstructionContextSlack = 256;

/// Drops the context and raw ISA of every instruction that has left the pending
/// queues, keeping the instruction now issuing so its accesses still find it.
void prune_retired_instructions(EngineWaveState &wave);

} // namespace hazard_core
