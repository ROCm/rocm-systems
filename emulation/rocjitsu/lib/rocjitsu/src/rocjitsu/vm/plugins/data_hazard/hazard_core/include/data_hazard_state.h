// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Simulator-neutral state containers used by DataHazardEngine. Adapters own
// translation of simulator callbacks into these structures so the core engine
// remains frontend-agnostic.

#pragma once

#include "hash_utils.h"
#include "hazard_events.h"
#include "types.h"
#include "util/spinlock.h"

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
  /// The counter this access was still outstanding on when an earlier barrier
  /// resolved, which carried it into this epoch; NONE for an access made in it.
  WaitCntType pending_at_barrier = WaitCntType::NONE;
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
  util::Spinlock mutex;
  std::vector<EngineLdsAccessRecord> lds_epoch;
};

struct EngineWaveState {
  util::Spinlock mutex;
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

/// What an LDS race's message and suggestion say once the waves and addresses
/// are taken out, which is what the report folds occurrences on. The wave a
/// carried access came from stays: the suggestion names it.
struct EngineLdsRaceShapeKey {
  EntityId dispatch_id;
  uint64_t producer_pc;
  uint64_t consumer_pc;
  uint32_t producer_size;
  uint32_t consumer_size;
  bool consumer_is_write;
  /// NONE when neither access was carried across a barrier.
  WaitCntType carried_pending;
  EntityId carried_wave;
  bool carried_is_write;

  bool operator==(const EngineLdsRaceShapeKey &other) const noexcept {
    return dispatch_id == other.dispatch_id && producer_pc == other.producer_pc &&
           consumer_pc == other.consumer_pc && producer_size == other.producer_size &&
           consumer_size == other.consumer_size && consumer_is_write == other.consumer_is_write &&
           carried_pending == other.carried_pending && carried_wave == other.carried_wave &&
           carried_is_write == other.carried_is_write;
  }
};

struct EngineLdsRaceShapeKeyHash {
  size_t operator()(const EngineLdsRaceShapeKey &key) const noexcept {
    size_t seed = 0;
    hash_combine(seed, key.dispatch_id);
    hash_combine(seed, key.producer_pc);
    hash_combine(seed, key.consumer_pc);
    hash_combine(seed, key.producer_size);
    hash_combine(seed, key.consumer_size);
    hash_combine(seed, key.consumer_is_write);
    hash_combine(seed, static_cast<int>(key.carried_pending));
    hash_combine(seed, key.carried_wave);
    hash_combine(seed, key.carried_is_write);
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

/// A global shadow entry spans four bytes (one aligned dword); retention is
/// partitioned into one bucket per byte.
inline constexpr size_t kGlobalShadowEntryBytes = 4;

/// Distinct foreign workgroups retained per byte. Two is the tight bound for the
/// pairwise cross-workgroup race check: any later access to a byte differs from
/// at least one of the two workgroups held for it, so a conflicting partner is
/// always still there to find. A third distinct workgroup on a byte evicts the
/// older of its two, and the byte still keeps two distinct foreign workgroups.
inline constexpr size_t kGlobalBytePartners = 2;

/// Accesses held per four-byte entry so a later conflicting access can still
/// find the one it races. Accesses narrower than a dword share the entry through
/// their byte masks, so disjoint sub-dword accesses from different workgroups
/// accumulate without racing each other. Too few retained partners evict an
/// earlier access before a later one can conflict with it, missing the race
/// (writers self-report, but reads and atomics do not).
///
/// Retention is partitioned by byte, not pooled across the entry: a flood of
/// accesses on one byte can only evict within that byte's own bucket, never
/// displace another byte's retained partner. That makes the bound hold by
/// construction -- eight entries in total, but as four bytes each keeping two
/// distinct foreign workgroups -- so a write to any byte still finds that byte's
/// partner however the accesses interleave. A flat pool of the same eight would
/// be order-dependent: seven readers of one byte could consume the slots a later
/// byte's reader needs, and its WAR race would be missed.
///
/// A slot names a record of the entry by index rather than holding the access
/// itself: an access covering several bytes, or retained as both reader and
/// writer, is then stored once instead of once per slot. 0 is an empty slot;
/// any other value is one past the record's index in
/// EngineGlobalShadowEntry::records.
using EngineGlobalSlot = uint8_t;
using EngineGlobalBytePartners = std::array<EngineGlobalSlot, kGlobalBytePartners>;
using EngineGlobalAccessSlots = std::array<EngineGlobalBytePartners, kGlobalShadowEntryBytes>;

/// The entry compacts its records to the ones its slots still name once it holds
/// this many. Slots can name at most 3 * 4 * 2 = 24 records, so a compacted
/// entry stays far below the 255 a slot can index.
inline constexpr size_t kGlobalShadowRecordCompactAt = 64;

struct EngineGlobalShadowEntry {
  /// Every access recorded in the entry, each once. Records no slot names any
  /// more are dropped when the entry compacts.
  std::vector<EngineGlobalAccessInfo> records;
  EngineGlobalAccessSlots writers{};
  EngineGlobalAccessSlots readers{};
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

/// The global shadow is split by address into shards, each with its own lock,
/// so accesses to unrelated memory from different threads do not serialise on
/// one lock. An entry lives in exactly one shard and is read and updated only
/// under that shard's lock, so splitting changes nothing about what is found.
inline constexpr size_t kGlobalShadowShards = 256;
/// Addresses within one 128-byte span share a shard, so a lane's access, and
/// usually a whole coalesced row of them, takes one lock.
inline constexpr unsigned kGlobalShadowShardShift = 7;

struct EngineGlobalShadowShard {
  std::mutex mutex;
  std::unordered_map<EngineGlobalShadowKey, EngineGlobalShadowEntry, EngineGlobalShadowKeyHash>
      entries;
};

inline size_t global_shadow_shard(const EngineGlobalShadowKey &key) {
  size_t seed = 0;
  hash_combine(seed, key.dispatch_id);
  hash_combine(seed, key.address >> kGlobalShadowShardShift);
  return seed % kGlobalShadowShards;
}

struct EngineWarning {
  HazardFinding finding;
  std::array<uint32_t, 4> source_raw_isa{};
  uint64_t source_pc = 0;
  bool has_source = false;
  std::string message;
  std::string suggestion;
  /// Nonzero when later occurrences of this finding may arrive through
  /// HazardWarningSink::emit_repeat under the same id instead of as warnings.
  uint64_t fold_id = 0;
};

class HazardWarningSink {
public:
  virtual ~HazardWarningSink() = default;
  virtual void emit_warning(const EngineWarning &warning) const = 0;
  /// One more occurrence of the finding emitted with this fold_id. It can
  /// arrive before that warning does when another thread is still formatting
  /// it. Only called with DataHazardEngine::set_fold_repeated_lds_races on.
  virtual void emit_repeat(uint64_t fold_id) const { (void)fold_id; }
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
  /// The fold id each LDS race shape was first reported under, when repeated
  /// LDS races are folded.
  std::unordered_map<EngineLdsRaceShapeKey, uint64_t, EngineLdsRaceShapeKeyHash> lds_race_folds;
  /// Guarded by each shard's own lock, not by mutex above.
  std::array<EngineGlobalShadowShard, kGlobalShadowShards> global_shadow;
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
