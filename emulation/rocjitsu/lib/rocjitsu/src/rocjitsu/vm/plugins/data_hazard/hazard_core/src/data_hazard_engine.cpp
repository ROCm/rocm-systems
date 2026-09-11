// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "data_hazard_engine.h"
#include "detail/access_handling.h"
#include "detail/wait_handling.h"

#include <algorithm>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace hazard_core {

namespace {

constexpr uint64_t kGlobalShadowAlignment = 4;
constexpr uint64_t kGlobalShadowAlignMask = ~(kGlobalShadowAlignment - 1);

// How one vector register file is looked up in wave state and named in a
// report. VGPRs and accumulator VGPRs run the same hazard logic over their own
// pending state, so the handlers below differ only by this descriptor.
struct VectorRegisterFileInfo {
  VectorRegisterFile file;
  HazardRegisterKind hazard_kind;
  ResourceKind resource_kind;
  const char *file_label;
  char reg_prefix;
  uint32_t max_index;
};

VectorRegisterFileInfo vector_register_file(ResourceKind resource_kind) {
  if (resource_kind == ResourceKind::AccumVectorRegister) {
    return {VectorRegisterFile::AccumVector,
            HazardRegisterKind::AccumVector,
            ResourceKind::AccumVectorRegister,
            "AccVGPR",
            'a',
            MAX_ACC_VGPR_INDEX};
  }
  return {VectorRegisterFile::Vector,
          HazardRegisterKind::Vector,
          ResourceKind::VectorRegister,
          "VGPR",
          'v',
          MAX_VGPR_INDEX};
}

constexpr WaitCntType kIdleWaitCounters[] = {
    WaitCntType::VMEM, WaitCntType::STORE, WaitCntType::SMEM, WaitCntType::LDS, WaitCntType::TENSOR,
};

/// Suggestion for a hazard on @p resource, asking the frontend first so a
/// simulator can word it its own way, then falling back to the shared wording.
std::string wait_suggestion(const SimulatorInstructionFormatter *formatter, WaitCntType kind,
                            HazardAccessKind access, HazardResourceLabel resource) {
  if (formatter) {
    std::string suggestion = formatter->format_wait_suggestion(kind, access, resource);
    if (!suggestion.empty())
      return suggestion;
  }
  return make_wait_suggestion(kind, access, resource);
}

std::string lds_write_suggestion(const SimulatorInstructionFormatter *formatter, WaitCntType kind) {
  return wait_suggestion(formatter, kind, HazardAccessKind::Write, HazardResourceLabel::LdsAddress);
}

std::string register_write_suggestion(const SimulatorInstructionFormatter *formatter,
                                      WaitCntType kind) {
  return wait_suggestion(formatter, kind, HazardAccessKind::Write, HazardResourceLabel::Register);
}

std::string lds_read_suggestion(const SimulatorInstructionFormatter *formatter, WaitCntType kind) {
  return wait_suggestion(formatter, kind, HazardAccessKind::Read, HazardResourceLabel::LdsAddress);
}

std::string register_read_suggestion(const SimulatorInstructionFormatter *formatter,
                                     WaitCntType kind) {
  return wait_suggestion(formatter, kind, HazardAccessKind::Read, HazardResourceLabel::Register);
}

std::string flat_load_read_suggestion(const SimulatorInstructionFormatter *formatter) {
  if (formatter) {
    std::string suggestion = formatter->format_flat_load_wait_suggestion();
    if (!suggestion.empty())
      return suggestion;
  }
  return make_flat_load_wait_suggestion();
}

std::string format_pending_message(const char *hazard, const std::string &resource,
                                   const char *action, const char *op,
                                   const PendingAsyncOp &pending) {
  std::ostringstream msg;
  msg << hazard << " hazard: " << resource << " " << action << " before async " << op
      << " completes (" << op << " at pc 0x" << std::hex << pending.pc << ", instruction "
      << std::dec << pending.instruction_id << ")";
  return msg.str();
}

// Register hazards are suppressed by two independent rules: the per-register
// set, which stops the same register being reported twice, and the pair set,
// which stops a wide operand reporting once per register when one producer and
// one remedy cover all of them. Both are pure checks so that a suppressed
// report never claims a key, which would hide a later genuine one.
template <typename RegisterKey>
bool claim_register_hazard(std::set<RegisterKey> &by_register,
                           std::set<hazard_core::RegisterHazardKey> &by_pair,
                           const RegisterKey &register_key,
                           const hazard_core::RegisterHazardKey &pair_key) {
  if (by_register.count(register_key) || by_pair.count(pair_key))
    return false;
  by_register.insert(register_key);
  by_pair.insert(pair_key);
  return true;
}

// First register of the span that has not been reported yet, or nothing if all
// of them have.
//
// A span must not be discarded just because it starts on a register an earlier
// consumer already claimed: a narrow read of s10 followed by a wide read of
// s[10:11] still owes a warning for s11. Skipping the claimed prefix is also
// what keeps the output the same whether a frontend hands a wide operand over
// as one access or as one per register.
template <typename Set, typename MakeKey>
std::optional<uint32_t> first_unreported_register(const Set &by_register,
                                                  const hazard_core::PendingRegisterSpan &span,
                                                  MakeKey make_key) {
  for (uint32_t reg = span.first_register; reg <= span.last_register; ++reg) {
    if (!by_register.count(make_key(reg)))
      return reg;
  }
  return std::nullopt;
}

// "SGPR s4" for a single register, "SGPR s[4:5]" when one producer covers a
// run of them.
std::string register_label(const char *file, char prefix, uint32_t first, uint32_t last) {
  std::string label = std::string(file) + " " + prefix;
  if (first == last)
    return label + std::to_string(first);
  return label + "[" + std::to_string(first) + ":" + std::to_string(last) + "]";
}

std::array<uint32_t, 4> raw_isa_from_context(const EngineInstructionContext &ctx) {
  return ctx.raw_isa;
}

void track_pending_raw_isa(EngineWaveState &wave, const EngineInstructionContext &ctx) {
  wave.pending_raw_isa[ctx.instruction_id] = raw_isa_from_context(ctx);
}

EngineGlobalAccessInfo make_global_access_info(const EngineInstructionContext &ctx,
                                               const ResourceAccessEvent &event) {
  EngineGlobalAccessInfo info;
  info.dispatch_id = ctx.dispatch_id;
  info.cluster_id = ctx.cluster_id;
  info.workgroup_id = ctx.workgroup_id;
  info.wave_id = ctx.wave_id;
  info.instruction_id = ctx.instruction_id;
  info.pc = ctx.pc;
  info.raw_isa = ctx.raw_isa;
  info.is_write = event.is_write;
  info.valid = true;
  return info;
}

InstructionHazardSemantics
merge_resource_semantics(const InstructionHazardSemantics &instruction_hazards,
                         const ResourceAccessEvent &event) {
  InstructionHazardSemantics hazards = instruction_hazards;
  if (event.hazards.memory_op_wait != WaitCntType::NONE)
    hazards.memory_op_wait = event.hazards.memory_op_wait;

  switch (event.resource_kind) {
  case ResourceKind::VectorRegister:
  case ResourceKind::AccumVectorRegister:
    if (event.hazards.write_wait != WaitCntType::NONE)
      hazards.vector_write_wait = event.hazards.write_wait;
    if (event.hazards.read_wait != WaitCntType::NONE)
      hazards.vector_read_wait = event.hazards.read_wait;
    hazards.vector_write_also_waits_lds =
        hazards.vector_write_also_waits_lds || event.hazards.write_also_waits_lds;
    break;
  case ResourceKind::ScalarRegister:
    if (event.hazards.write_wait != WaitCntType::NONE)
      hazards.scalar_write_wait = event.hazards.write_wait;
    break;
  case ResourceKind::LocalMemory:
    if (event.hazards.write_wait != WaitCntType::NONE)
      hazards.local_write_wait = event.hazards.write_wait;
    break;
  case ResourceKind::GlobalMemory:
  case ResourceKind::ScratchMemory:
  case ResourceKind::MemoryRegister:
  case ResourceKind::Unknown:
    break;
  }
  return hazards;
}

/// Record a memory operation that holds a wait counter slot without leaving
/// pending register state, which is what an ordinary vector memory store does.
/// The frontend names the counter it is outstanding on, VMEM where one counter
/// covers loads and stores and STORE where they are counted apart. A frontend
/// may report one store once per lane or per access, so an instruction already
/// at the back of the queue is not queued again.
void track_memory_operation(WaveState &wave, const EngineInstructionContext &ctx,
                            WaitCntType memory_op_wait) {
  if (memory_op_wait != WaitCntType::VMEM && memory_op_wait != WaitCntType::STORE)
    return;
  if (!wave.vmem_store_ops.empty() &&
      wave.vmem_store_ops.back().instruction_id == ctx.instruction_id)
    return;

  PendingAsyncOp op;
  op.instruction_id = ctx.instruction_id;
  op.pc = ctx.pc;
  op.wait_type = memory_op_wait;
  wave.vmem_store_ops.push_back(op);
}

bool same_workgroup(const EngineGlobalAccessInfo &info, const EngineInstructionContext &ctx) {
  return info.valid && info.dispatch_id == ctx.dispatch_id && info.cluster_id == ctx.cluster_id &&
         info.workgroup_id == ctx.workgroup_id;
}

/// Bits of the four-byte shadow entry at @p entry_address that [@p start, @p
/// end) covers. An access reaching neither end of the entry leaves the bytes it
/// misses clear, so a disjoint access to the same entry does not conflict.
uint8_t entry_byte_mask(uint64_t entry_address, uint64_t start, uint64_t end) {
  uint8_t mask = 0;
  // entry_address is aligned down, so adding a byte index cannot wrap.
  for (uint64_t byte = 0; byte < kGlobalShadowAlignment; ++byte) {
    const uint64_t address = entry_address + byte;
    if (address >= start && address < end)
      mask = static_cast<uint8_t>(mask | (1u << byte));
  }
  return mask;
}

/// A retained access that @p ctx would race with over @p byte_mask, or null when
/// every overlapping access belongs to the accessing workgroup. @p writers_only
/// narrows the search to accesses that wrote, which is all a read conflicts
/// with; slots holding a single kind of access pass it as false.
const EngineGlobalAccessInfo *find_conflicting_access(const EngineGlobalAccessSlots &slots,
                                                      const EngineInstructionContext &ctx,
                                                      uint8_t byte_mask, bool writers_only) {
  for (const auto &slot : slots) {
    if (slot.valid && (slot.byte_mask & byte_mask) != 0 && !same_workgroup(slot, ctx) &&
        (!writers_only || slot.is_write))
      return &slot;
  }
  return nullptr;
}

/// Which of @p byte_mask the accessing workgroup itself already covered. Bytes
/// outside the returned mask remain the caller's to retain, since a workgroup
/// that covered part of an access says nothing about the rest of it.
uint8_t coverage_by_same_workgroup(const EngineGlobalAccessSlots &slots,
                                   const EngineInstructionContext &ctx, uint8_t byte_mask) {
  uint8_t covered = 0;
  for (const auto &slot : slots) {
    if (slot.valid && same_workgroup(slot, ctx))
      covered = static_cast<uint8_t>(covered | slot.byte_mask);
  }
  return static_cast<uint8_t>(covered & byte_mask);
}

/// Retains @p current while keeping the slots on distinct workgroups: a repeat
/// from a workgroup already held widens that slot's byte coverage instead of
/// consuming the slot holding the other workgroup, and a newcomer takes a free
/// slot before displacing the second.
void record_access(EngineGlobalAccessSlots &slots, const EngineInstructionContext &ctx,
                   const EngineGlobalAccessInfo &current) {
  for (auto &slot : slots) {
    if (slot.valid && same_workgroup(slot, ctx)) {
      const uint8_t covered = static_cast<uint8_t>(slot.byte_mask | current.byte_mask);
      slot = current;
      slot.byte_mask = covered;
      return;
    }
  }
  for (auto &slot : slots) {
    if (!slot.valid) {
      slot = current;
      return;
    }
  }
  slots[1] = current;
}

EngineWarning make_global_race_warning(const EngineInstructionContext &ctx,
                                       const ResourceAccessEvent &event, uint64_t address,
                                       const EngineGlobalAccessInfo &conflicting_access,
                                       const char *conflict_type) {
  std::ostringstream msg;
  msg << "Global memory " << conflict_type << " data race at address 0x" << std::hex << address
      << ": Workgroup " << std::dec << ctx.workgroup_id << " (wave " << ctx.wave_id
      << ") conflicts with Workgroup " << conflicting_access.workgroup_id << " (wave "
      << conflicting_access.wave_id << ", instruction " << conflicting_access.instruction_id
      << " at pc 0x" << std::hex << conflicting_access.pc << ")" << std::dec;

  EngineWarning warning;
  warning.finding.kind = HazardKind::GlobalMemoryRace;
  warning.finding.resource_kind = ResourceKind::GlobalMemory;
  warning.finding.access_kind = event.is_write ? HazardAccessKind::Write : HazardAccessKind::Read;
  warning.finding.instruction = event.instruction;
  warning.finding.instruction.pc = ctx.pc;
  warning.finding.instruction.raw_isa = ctx.raw_isa;
  warning.finding.address = address;
  warning.finding.size_bytes = event.size_bytes;
  warning.finding.has_source_instruction = true;
  warning.finding.source_instruction.execution.dispatch_id = conflicting_access.dispatch_id;
  warning.finding.source_instruction.execution.cluster_id = conflicting_access.cluster_id;
  warning.finding.source_instruction.execution.workgroup_id = conflicting_access.workgroup_id;
  warning.finding.source_instruction.execution.wave_id = conflicting_access.wave_id;
  warning.finding.source_instruction.instruction_id = conflicting_access.instruction_id;
  warning.finding.source_instruction.pc = conflicting_access.pc;
  warning.finding.source_instruction.raw_isa = conflicting_access.raw_isa;
  warning.finding.message_template = msg.str();
  warning.finding.suggestion_template =
      "Use atomic operations or add proper synchronization between workgroups";
  warning.message = msg.str();
  warning.suggestion = "Use atomic operations or add proper synchronization between workgroups";
  warning.source_raw_isa = conflicting_access.raw_isa;
  warning.source_pc = conflicting_access.pc;
  warning.has_source = true;
  return warning;
}

} // namespace

DataHazardEngine &data_hazard_engine() {
  static DataHazardEngine engine;
  return engine;
}

void DataHazardEngine::reset() {
  state_.reset();
  rejected_events_.store(0, std::memory_order_relaxed);
}

void DataHazardEngine::set_instruction_formatter(const SimulatorInstructionFormatter *formatter) {
  instruction_formatter_ = formatter;
}

void DataHazardEngine::set_warning_sink(const HazardWarningSink *sink) { warning_sink_ = sink; }

void DataHazardEngine::set_diagnostic_handler(EngineDiagnosticHandler handler) {
  diagnostic_handler_ = std::move(handler);
}

uint64_t DataHazardEngine::rejected_event_count() const {
  return rejected_events_.load(std::memory_order_relaxed);
}

void DataHazardEngine::reject_event(const std::string &message) const {
  rejected_events_.fetch_add(1, std::memory_order_relaxed);
  if (diagnostic_handler_)
    diagnostic_handler_(message);
}

bool DataHazardEngine::check_access_size(const char *handler, uint32_t size_bytes) const {
  if (size_bytes != 0 && size_bytes <= hazard_core::MAX_ACCESS_SIZE_BYTES)
    return true;
  reject_event(std::string(handler) + ": invalid size_bytes " + std::to_string(size_bytes));
  return false;
}

std::vector<EngineWarning> DataHazardEngine::warning_snapshot() const {
  std::shared_lock<std::shared_mutex> lock(state_.mutex);
  return state_.warnings;
}

std::optional<EngineWaveSnapshot>
DataHazardEngine::wave_snapshot(const ExecutionKey &wave_key) const {
  std::shared_ptr<EngineWaveState> wave;
  {
    std::shared_lock<std::shared_mutex> lock(state_.mutex);
    auto it = state_.waves.find(wave_key);
    if (it == state_.waves.end())
      return std::nullopt;
    wave = it->second;
  }

  std::lock_guard<SpinLock> lock(wave->mutex);
  return EngineWaveSnapshot{
      wave_key,
      wave->core,
      wave->instruction_contexts.size(),
  };
}

std::vector<EngineWaveSnapshot> DataHazardEngine::wave_snapshots() const {
  std::vector<std::pair<ExecutionKey, std::shared_ptr<EngineWaveState>>> waves;
  {
    std::shared_lock<std::shared_mutex> lock(state_.mutex);
    waves.reserve(state_.waves.size());
    for (const auto &[key, wave] : state_.waves)
      waves.push_back({key, wave});
  }

  std::vector<EngineWaveSnapshot> snapshots;
  snapshots.reserve(waves.size());
  for (const auto &[key, wave] : waves) {
    std::lock_guard<SpinLock> lock(wave->mutex);
    snapshots.push_back(EngineWaveSnapshot{
        key,
        wave->core,
        wave->instruction_contexts.size(),
    });
  }
  return snapshots;
}

size_t DataHazardEngine::wave_count() const {
  std::shared_lock<std::shared_mutex> lock(state_.mutex);
  return state_.waves.size();
}

void DataHazardEngine::on_dispatch_begin(EntityId dispatch_id) {
  reset_dispatch(dispatch_id, /*flush_epochs=*/false);
}

void DataHazardEngine::on_dispatch_end(EntityId dispatch_id) {
  reset_dispatch(dispatch_id, /*flush_epochs=*/true);
}

void DataHazardEngine::on_workgroup_begin(EntityId dispatch_id, EntityId cluster_id,
                                          EntityId workgroup_id) {
  auto workgroup =
      get_or_create_workgroup(EngineWorkgroupKey{dispatch_id, cluster_id, workgroup_id});
  if (workgroup) {
    std::lock_guard<SpinLock> lock(workgroup->mutex);
    workgroup->lds_epoch.clear();
  }
}

void DataHazardEngine::on_workgroup_end(EntityId dispatch_id, EntityId cluster_id,
                                        EntityId workgroup_id) {
  const EngineWorkgroupKey engine_key{dispatch_id, cluster_id, workgroup_id};
  flush_workgroup_epoch(engine_key);
  {
    std::lock_guard<std::shared_mutex> lock(state_.mutex);
    state_.workgroups.erase(engine_key);
  }
}

void DataHazardEngine::on_wave_begin(const ExecutionKey &wave_key) {
  const EngineWorkgroupKey workgroup_key{wave_key.dispatch_id, wave_key.cluster_id,
                                         wave_key.workgroup_id};
  const EngineWaveKey wave_index{wave_key.dispatch_id, wave_key.cluster_id, wave_key.workgroup_id,
                                 wave_key.wave_id};
  auto workgroup = get_or_create_workgroup(workgroup_key);
  auto wave = get_or_create_wave(wave_key, workgroup_key, wave_index);
  if (!wave)
    return;

  std::lock_guard<SpinLock> lock(wave->mutex);
  wave->workgroup = workgroup;
  wave->core.reset();
  wave->instruction_contexts.clear();
  wave->pending_raw_isa.clear();
  wave->current_instruction_id = 0;
  wave->contexts_at_last_prune = 0;
}

void DataHazardEngine::on_wave_end(const ExecutionKey &wave_key) {
  erase_wave(wave_key, EngineWaveKey{wave_key.dispatch_id, wave_key.cluster_id,
                                     wave_key.workgroup_id, wave_key.wave_id});
}

void DataHazardEngine::on_instruction(const InstructionEvent &instruction) {
  const ExecutionKey wave_key = instruction.instruction.execution;
  auto wave = get_or_create_wave(
      wave_key,
      EngineWorkgroupKey{wave_key.dispatch_id, wave_key.cluster_id, wave_key.workgroup_id},
      EngineWaveKey{wave_key.dispatch_id, wave_key.cluster_id, wave_key.workgroup_id,
                    wave_key.wave_id});
  if (!wave)
    return;

  std::vector<EngineLdsAccessRecord> epoch_to_check;
  EngineWorkgroupKey epoch_key{};
  {
    std::lock_guard<SpinLock> lock(wave->mutex);
    track_instruction_context(*wave, instruction);

    const EngineInstructionContext ctx =
        get_instruction_context(*wave, instruction.instruction.instruction_id);

    // Ahead of the wait handling below, which a plain store returns before
    // reaching because it carries no wait action of its own.
    track_memory_operation(wave->core, ctx, instruction.hazards.memory_op_wait);

    const WaitAction &action = instruction.wait_action;
    if (!wait_action_has_effect(action))
      return;

    if (action.waits_for_idle) {
      for (WaitCntType counter : kIdleWaitCounters)
        hazard_core::clear_pending_ops(&wave->core, counter, 0);
    }

    for (const auto &counter : action.counters) {
      if (!hazard_core::clear_pending_ops(&wave->core, counter))
        reject_event("on_instruction: wait counter has no pending queue to drain");
    }
    prune_retired_instructions(*wave);

    if (action.is_workgroup_barrier && wave->workgroup) {
      std::lock_guard<SpinLock> wg_lock(wave->workgroup->mutex);
      epoch_to_check = std::move(wave->workgroup->lds_epoch);
      wave->workgroup->lds_epoch.clear();
      epoch_key = EngineWorkgroupKey{ctx.dispatch_id, ctx.cluster_id, ctx.workgroup_id};
    }
  }

  if (!epoch_to_check.empty())
    check_lds_epoch_for_races(epoch_key, epoch_to_check);
}

void DataHazardEngine::on_resource_access(const ResourceAccessEvent &event) {
  const ExecutionKey wave_key = event.instruction.execution;
  auto wave = get_or_create_wave(
      wave_key,
      EngineWorkgroupKey{wave_key.dispatch_id, wave_key.cluster_id, wave_key.workgroup_id},
      EngineWaveKey{wave_key.dispatch_id, wave_key.cluster_id, wave_key.workgroup_id,
                    wave_key.wave_id});
  if (!wave)
    return;

  std::unique_lock<SpinLock> lock(wave->mutex);
  EngineInstructionContext ctx = get_instruction_context(*wave, event.instruction.instruction_id);
  if (ctx.instruction_id == 0 && event.instruction.instruction_id != 0)
    ctx = make_engine_instruction_context(event.instruction);

  InstructionHazardSemantics hazards = merge_resource_semantics(ctx.hazards, event);
  if (event.hazards.memory_op_wait != WaitCntType::NONE) {
    track_memory_operation(wave->core, ctx, event.hazards.memory_op_wait);
    // A frontend that reports counter occupancy on its own, rather than
    // alongside an access it already reports, sends no address to examine.
    if (!event.is_read && !event.is_write)
      return;
  }

  switch (event.resource_kind) {
  case ResourceKind::VectorRegister:
  case ResourceKind::AccumVectorRegister:
    handle_vector_access(*wave, ctx, event, hazards.vector_write_wait,
                         hazards.vector_write_also_waits_lds, hazards.vector_read_wait);
    break;
  case ResourceKind::ScalarRegister:
    handle_scalar_access(*wave, ctx, event, hazards.scalar_write_wait);
    break;
  case ResourceKind::LocalMemory:
    handle_lds_access(*wave, ctx, event, hazards.local_write_wait);
    break;
  case ResourceKind::GlobalMemory: {
    lock.unlock();
    check_global_access_for_races(ctx, event);
    return;
  }
  case ResourceKind::ScratchMemory:
  case ResourceKind::MemoryRegister:
  case ResourceKind::Unknown:
    break;
  }
}

void DataHazardEngine::on_barrier(const BarrierEvent &barrier) {
  if (barrier.kind == BarrierKind::LocalMemoryAtomic ||
      barrier.kind == BarrierKind::LocalMemoryAtomicAsync) {
    for (const auto &wave : waves_for_barrier(barrier)) {
      std::lock_guard<SpinLock> lock(wave->mutex);
      hazard_core::clear_pending_ops(&wave->core, WaitCntType::LDS, 0);
      prune_retired_instructions(*wave);
    }
  } else if (barrier.kind == BarrierKind::Workgroup) {
    for (const auto &key : workgroup_keys_for_barrier(barrier))
      flush_workgroup_epoch(key);
  } else if (barrier.kind == BarrierKind::Cluster) {
    for (const auto &key : workgroup_keys_for_barrier(barrier))
      flush_workgroup_epoch(key);
  }
}

void DataHazardEngine::on_shutdown() {
  std::vector<EngineWorkgroupKey> workgroups;
  {
    std::shared_lock<std::shared_mutex> lock(state_.mutex);
    workgroups.reserve(state_.workgroups.size());
    for (const auto &[key, _] : state_.workgroups)
      workgroups.push_back(key);
  }

  for (const auto &key : workgroups)
    flush_workgroup_epoch(key);
}

std::shared_ptr<EngineWaveState>
DataHazardEngine::get_or_create_wave(const ExecutionKey &key,
                                     const EngineWorkgroupKey &workgroup_key,
                                     const EngineWaveKey &partial) {
  struct CachedWave {
    const DataHazardEngine *engine = nullptr;
    uint64_t generation = 0;
    ExecutionKey key{};
    std::shared_ptr<EngineWaveState> wave;
  };

  static thread_local CachedWave cached_wave;
  const uint64_t generation = state_.wave_cache_generation.load(std::memory_order_acquire);
  if (cached_wave.engine == this && cached_wave.generation == generation && cached_wave.wave &&
      cached_wave.key == key) {
    return cached_wave.wave;
  }

  std::shared_ptr<EngineWaveState> wave;
  std::shared_ptr<EngineWorkgroupState> workgroup;
  {
    std::shared_lock<std::shared_mutex> lock(state_.mutex);
    auto wave_it = state_.waves.find(key);
    if (wave_it != state_.waves.end() && wave_it->second) {
      cached_wave = CachedWave{this, generation, key, wave_it->second};
      return wave_it->second;
    }
  }

  {
    std::lock_guard<std::shared_mutex> lock(state_.mutex);
    auto wave_it = state_.waves.find(key);
    if (wave_it != state_.waves.end() && wave_it->second) {
      const uint64_t current_generation =
          state_.wave_cache_generation.load(std::memory_order_acquire);
      cached_wave = CachedWave{this, current_generation, key, wave_it->second};
      return wave_it->second;
    }

    auto &workgroup_entry = state_.workgroups[workgroup_key];
    if (!workgroup_entry)
      workgroup_entry = std::make_shared<EngineWorkgroupState>();
    workgroup = workgroup_entry;

    auto &entry = state_.waves[key];
    if (!entry)
      entry = std::make_shared<EngineWaveState>();
    entry->workgroup = workgroup;

    state_.wave_index[partial] = key;
    if (partial.dispatch_id != 0) {
      const EngineWaveKey dispatchless_partial{0, partial.cluster_id, partial.workgroup_id,
                                               partial.wave_id};
      state_.wave_index[dispatchless_partial] = key;
    }

    state_.wave_to_workgroup[key] = workgroup_key;
    wave = entry;
  }
  const uint64_t current_generation = state_.wave_cache_generation.load(std::memory_order_acquire);
  cached_wave = CachedWave{this, current_generation, key, wave};
  return wave;
}

std::shared_ptr<EngineWaveState> DataHazardEngine::get_wave(const ExecutionKey &key) {
  std::shared_lock<std::shared_mutex> lock(state_.mutex);
  auto it = state_.waves.find(key);
  if (it == state_.waves.end())
    return nullptr;
  return it->second;
}

std::shared_ptr<EngineWaveState> DataHazardEngine::get_wave(const EngineWaveKey &partial) {
  std::shared_lock<std::shared_mutex> lock(state_.mutex);
  auto it = state_.wave_index.find(partial);
  if (it == state_.wave_index.end())
    return nullptr;
  auto wave_it = state_.waves.find(it->second);
  if (wave_it == state_.waves.end())
    return nullptr;
  return wave_it->second;
}

std::vector<std::shared_ptr<EngineWaveState>>
DataHazardEngine::waves_for_barrier(const BarrierEvent &barrier) const {
  std::shared_lock<std::shared_mutex> lock(state_.mutex);

  if (barrier.wave.dispatch_id != 0) {
    auto it = state_.waves.find(barrier.wave);
    if (it == state_.waves.end() || !it->second)
      return {};
    return {it->second};
  }

  std::vector<std::shared_ptr<EngineWaveState>> waves;
  for (const auto &[key, wave] : state_.waves) {
    if (key.cluster_id == barrier.wave.cluster_id &&
        key.workgroup_id == barrier.wave.workgroup_id && key.wave_id == barrier.wave.wave_id &&
        wave) {
      waves.push_back(wave);
    }
  }
  return waves;
}

std::shared_ptr<EngineWorkgroupState>
DataHazardEngine::get_or_create_workgroup(const EngineWorkgroupKey &key) {
  std::shared_ptr<EngineWorkgroupState> workgroup;
  {
    std::lock_guard<std::shared_mutex> lock(state_.mutex);
    auto &entry = state_.workgroups[key];
    if (!entry)
      entry = std::make_shared<EngineWorkgroupState>();
    workgroup = entry;
  }
  return workgroup;
}

void DataHazardEngine::reset_dispatch(EntityId dispatch_id, bool flush_epochs) {
  if (flush_epochs) {
    std::vector<EngineWorkgroupKey> workgroups;
    {
      std::shared_lock<std::shared_mutex> lock(state_.mutex);
      for (const auto &[key, _] : state_.workgroups) {
        if (key.dispatch_id == dispatch_id)
          workgroups.push_back(key);
      }
    }

    for (const auto &key : workgroups)
      flush_workgroup_epoch(key);
  }

  std::lock_guard<std::shared_mutex> lock(state_.mutex);

  for (auto it = state_.waves.begin(); it != state_.waves.end();) {
    if (it->first.dispatch_id == dispatch_id) {
      it = state_.waves.erase(it);
    } else {
      ++it;
    }
  }

  for (auto it = state_.wave_to_workgroup.begin(); it != state_.wave_to_workgroup.end();) {
    if (it->first.dispatch_id == dispatch_id) {
      it = state_.wave_to_workgroup.erase(it);
    } else {
      ++it;
    }
  }

  for (auto it = state_.workgroups.begin(); it != state_.workgroups.end();) {
    if (it->first.dispatch_id == dispatch_id) {
      it = state_.workgroups.erase(it);
    } else {
      ++it;
    }
  }

  for (auto it = state_.wave_index.begin(); it != state_.wave_index.end();) {
    if (it->first.dispatch_id == dispatch_id || it->second.dispatch_id == dispatch_id) {
      it = state_.wave_index.erase(it);
    } else {
      ++it;
    }
  }

  for (auto it = state_.reported_lds_races.begin(); it != state_.reported_lds_races.end();) {
    if (it->dispatch_id == dispatch_id) {
      it = state_.reported_lds_races.erase(it);
    } else {
      ++it;
    }
  }

  for (auto it = state_.global_shadow.begin(); it != state_.global_shadow.end();) {
    if (it->first.dispatch_id == dispatch_id) {
      it = state_.global_shadow.erase(it);
    } else {
      ++it;
    }
  }
  state_.wave_cache_generation.fetch_add(1, std::memory_order_release);
}

void DataHazardEngine::erase_wave(const ExecutionKey &key, const EngineWaveKey &partial) {
  std::lock_guard<std::shared_mutex> lock(state_.mutex);
  state_.waves.erase(key);
  state_.wave_index.erase(partial);
  if (partial.dispatch_id != 0) {
    const EngineWaveKey dispatchless_partial{0, partial.cluster_id, partial.workgroup_id,
                                             partial.wave_id};
    auto it = state_.wave_index.find(dispatchless_partial);
    if (it != state_.wave_index.end() && it->second == key)
      state_.wave_index.erase(it);
  }
  state_.wave_to_workgroup.erase(key);
  state_.wave_cache_generation.fetch_add(1, std::memory_order_release);
}

std::vector<EngineWorkgroupKey>
DataHazardEngine::workgroup_keys_for_barrier(const BarrierEvent &barrier) const {
  const EngineWorkgroupKey explicit_key{barrier.wave.dispatch_id, barrier.wave.cluster_id,
                                        barrier.wave.workgroup_id};
  if (barrier.wave.dispatch_id != 0)
    return {explicit_key};

  std::vector<EngineWorkgroupKey> keys;
  std::shared_lock<std::shared_mutex> lock(state_.mutex);

  for (const auto &[key, _] : state_.workgroups) {
    if (key.cluster_id == barrier.wave.cluster_id && key.workgroup_id == barrier.wave.workgroup_id)
      keys.push_back(key);
  }
  return keys;
}

void DataHazardEngine::record_warning(
    const EngineInstructionContext &ctx, const ResourceAccessEvent &event, HazardKind kind,
    ResourceKind resource_kind, HazardAccessKind access_kind, uint32_t resource_index,
    uint64_t address, uint32_t size_bytes, WaitCntType required_wait, const PendingAsyncOp &pending,
    const std::array<uint32_t, 4> &source_raw_isa, uint64_t source_pc, const std::string &message,
    const std::string &suggestion) {
  HazardFinding finding;
  finding.kind = kind;
  finding.resource_kind = resource_kind;
  finding.access_kind = access_kind;
  finding.instruction = event.instruction;
  finding.address = address;
  finding.resource_index = resource_index;
  finding.size_bytes = size_bytes;
  finding.required_wait = required_wait;
  finding.message_template = message;
  finding.suggestion_template = suggestion;
  finding.instruction.pc = ctx.pc;
  finding.instruction.raw_isa = ctx.raw_isa;

  if (pending.instruction_id != 0) {
    finding.has_source_instruction = true;
    finding.source_instruction.instruction_id = pending.instruction_id;
    finding.source_instruction.pc = source_pc;
    finding.source_instruction.execution = event.instruction.execution;
    finding.source_instruction.raw_isa = source_raw_isa;
  }

  EngineWarning warning;
  warning.finding = finding;
  warning.source_raw_isa = source_raw_isa;
  warning.source_pc = source_pc;
  warning.has_source = pending.instruction_id != 0;
  warning.message = message;
  warning.suggestion = suggestion;

  {
    std::lock_guard<std::shared_mutex> lock(state_.mutex);
    state_.warnings.push_back(warning);
  }
  emit_warning(warning);
}

void DataHazardEngine::handle_vector_access(
    EngineWaveState &wave, const EngineInstructionContext &ctx, const ResourceAccessEvent &event,
    WaitCntType vector_write_wait, bool vector_write_also_waits_lds, WaitCntType vector_read_wait) {
  if (!check_access_size("handle_vector_access", event.size_bytes))
    return;

  const uint32_t reg = event.resource_index;
  const uint32_t count = hazard_core::bytes_to_dwords(event.size_bytes);

  if (event.is_read)
    check_vector_raw_hazards(wave, ctx, event, event.resource_kind, reg, count);
  if (event.is_write)
    check_vector_war_waw_hazards(wave, ctx, event, event.resource_kind, reg, count);
  track_vector_pending(wave, ctx, event, event.resource_kind, reg, count, vector_write_wait,
                       vector_write_also_waits_lds, vector_read_wait);
}

void DataHazardEngine::check_vector_raw_hazards(EngineWaveState &wave,
                                                const EngineInstructionContext &ctx,
                                                const ResourceAccessEvent &event,
                                                ResourceKind resource_kind, uint32_t reg,
                                                uint32_t count) {
  const VectorRegisterFileInfo regs = vector_register_file(resource_kind);
  const auto &pending_writes = pending_registers(wave.core, regs.file, PendingRegisterSet::Writes);
  const auto &pending_writes_ds =
      pending_registers(wave.core, regs.file, PendingRegisterSet::DsWrites);
  const EntityId current_id = event.instruction.instruction_id;

  // These handlers walk register by register because the pending vmem and LDS
  // sides have to be consulted separately, but only the first register of a
  // run reports: the rest are folded into it by the pair rule. Naming the run
  // keeps the folded registers visible.
  const uint32_t last_access_reg = reg + count - 1;
  const auto reg_label = [&](const std::unordered_map<uint32_t, PendingAsyncOp> &pending_map,
                             uint32_t first, EntityId producer) {
    return register_label(regs.file_label, regs.reg_prefix, first,
                          hazard_core::pending_span_end(pending_map, first, last_access_reg,
                                                        regs.max_index, producer));
  };

  for (uint32_t i = 0; i < count; ++i) {
    const uint32_t read_reg = reg + i;
    if (read_reg > regs.max_index)
      break;

    const auto vmem_hazard = check_vector_hazard_info(
        &wave.core, regs.file, PendingRegisterSet::Writes, read_reg, 1, current_id);
    const auto ds_hazard = check_vector_hazard_info(
        &wave.core, regs.file, PendingRegisterSet::DsWrites, read_reg, 1, current_id);

    const PendingAsyncOp *vmem_pending = vmem_hazard.pending;
    const PendingAsyncOp *ds_pending = ds_hazard.pending;
    const bool report_combined_flat = vmem_pending && ds_pending && ds_pending->is_flat_ds &&
                                      vmem_pending->instruction_id == ds_pending->instruction_id;

    if (report_combined_flat) {
      if (claim_register_hazard(
              wave.core.reported_raw_hazards, wave.core.reported_raw_pairs,
              WawHazardKey{vmem_pending->instruction_id, regs.hazard_kind, read_reg},
              {vmem_pending->instruction_id, current_id, regs.hazard_kind})) {
        const std::string message = format_pending_message(
            "RAW", reg_label(pending_writes, read_reg, vmem_pending->instruction_id), "read",
            "load", *vmem_pending);
        record_warning(ctx, event, HazardKind::RAW, regs.resource_kind, HazardAccessKind::Read,
                       read_reg, 0, event.size_bytes, vmem_pending->wait_type, *vmem_pending,
                       get_pending_raw_isa(wave, vmem_pending->instruction_id), vmem_pending->pc,
                       message, flat_load_read_suggestion(instruction_formatter()));
      }
      continue;
    }

    if (const auto *pending = vmem_pending) {
      if (claim_register_hazard(wave.core.reported_raw_hazards, wave.core.reported_raw_pairs,
                                WawHazardKey{pending->instruction_id, regs.hazard_kind, read_reg},
                                {pending->instruction_id, current_id, regs.hazard_kind})) {
        const std::string message = format_pending_message(
            "RAW", reg_label(pending_writes, read_reg, pending->instruction_id), "read", "load",
            *pending);
        record_warning(ctx, event, HazardKind::RAW, regs.resource_kind, HazardAccessKind::Read,
                       read_reg, 0, event.size_bytes, pending->wait_type, *pending,
                       get_pending_raw_isa(wave, pending->instruction_id), pending->pc, message,
                       register_read_suggestion(instruction_formatter(), pending->wait_type));
      }
    }

    if (const auto *pending = ds_pending; pending && !pending->is_flat_ds) {
      if (!claim_register_hazard(wave.core.reported_raw_hazards, wave.core.reported_raw_pairs,
                                 WawHazardKey{pending->instruction_id, regs.hazard_kind, read_reg},
                                 {pending->instruction_id, current_id, regs.hazard_kind}))
        continue;
      const std::string message = format_pending_message(
          "RAW", reg_label(pending_writes_ds, read_reg, pending->instruction_id), "read", "load",
          *pending);
      record_warning(ctx, event, HazardKind::RAW, regs.resource_kind, HazardAccessKind::Read,
                     read_reg, 0, event.size_bytes, pending->wait_type, *pending,
                     get_pending_raw_isa(wave, pending->instruction_id), pending->pc, message,
                     register_read_suggestion(instruction_formatter(), pending->wait_type));
    }
  }
}

// The WAR half of this only fires for pending entries put in by track_vector_read,
// which needs an adapter to set ResourceHazardSemantics::read_wait on a store's
// data source. No rocjitsu store path does that today, so register WAR is
// currently unreported there even though the engine can represent it.
void DataHazardEngine::check_vector_war_waw_hazards(EngineWaveState &wave,
                                                    const EngineInstructionContext &ctx,
                                                    const ResourceAccessEvent &event,
                                                    ResourceKind resource_kind, uint32_t reg,
                                                    uint32_t count) {
  const VectorRegisterFileInfo regs = vector_register_file(resource_kind);
  const auto &pending_writes = pending_registers(wave.core, regs.file, PendingRegisterSet::Writes);
  const auto &pending_reads = pending_registers(wave.core, regs.file, PendingRegisterSet::Reads);
  const EntityId current_id = event.instruction.instruction_id;
  const uint32_t last_access_reg = reg + count - 1;
  const auto reg_label = [&](const std::unordered_map<uint32_t, PendingAsyncOp> &pending_map,
                             uint32_t first, EntityId producer) {
    return register_label(regs.file_label, regs.reg_prefix, first,
                          hazard_core::pending_span_end(pending_map, first, last_access_reg,
                                                        regs.max_index, producer));
  };

  for (uint32_t i = 0; i < count; ++i) {
    const uint32_t written_reg = reg + i;
    if (written_reg > regs.max_index)
      break;
    const auto war_hazard = check_vector_hazard_info(
        &wave.core, regs.file, PendingRegisterSet::Reads, written_reg, 1, current_id);
    if (!war_hazard.pending)
      continue;

    const auto &pending = *war_hazard.pending;
    if (!claim_register_hazard(wave.core.reported_war_hazards, wave.core.reported_war_pairs,
                               WawHazardKey{current_id, regs.hazard_kind, written_reg},
                               {pending.instruction_id, current_id, regs.hazard_kind}))
      continue;
    const std::string message =
        format_pending_message("WAR", reg_label(pending_reads, written_reg, pending.instruction_id),
                               "written", "store", pending);
    record_warning(ctx, event, HazardKind::WAR, regs.resource_kind, HazardAccessKind::Write,
                   written_reg, 0, event.size_bytes, pending.wait_type, pending,
                   get_pending_raw_isa(wave, pending.instruction_id), pending.pc, message,
                   register_write_suggestion(instruction_formatter(), pending.wait_type));
  }

  for (uint32_t i = 0; i < count; ++i) {
    const uint32_t written_reg = reg + i;
    if (written_reg > regs.max_index)
      break;

    // Memory-read destinations are not available until the relevant wait
    // counter drains. Overwriting the destination before completion is a WAW
    // ordering hazard because the pending memory read can still write the
    // old value later.
    const auto waw_hazard =
        check_vector_waw_hazard_info(&wave.core, regs.file, written_reg, 1, current_id);
    if (const auto *pending = waw_hazard.pending) {
      // A flat load's DScnt-side marker can remain after LOADcnt clears.
      // By itself it is not a separate destination-write hazard.
      if (pending->is_flat_ds)
        continue;

      if (claim_register_hazard(wave.core.reported_waw_hazards, wave.core.reported_waw_pairs,
                                {current_id, regs.hazard_kind, written_reg},
                                {pending->instruction_id, current_id, regs.hazard_kind})) {
        const std::string message = format_pending_message(
            "WAW", reg_label(pending_writes, written_reg, pending->instruction_id), "written",
            "load", *pending);
        record_warning(ctx, event, HazardKind::WAW, regs.resource_kind, HazardAccessKind::Write,
                       written_reg, 0, event.size_bytes, pending->wait_type, *pending,
                       get_pending_raw_isa(wave, pending->instruction_id), pending->pc, message,
                       register_write_suggestion(instruction_formatter(), pending->wait_type));
      }
    }
  }
}

void DataHazardEngine::track_vector_pending(
    EngineWaveState &wave, const EngineInstructionContext &ctx, const ResourceAccessEvent &event,
    ResourceKind resource_kind, uint32_t reg, uint32_t count, WaitCntType vector_write_wait,
    bool vector_write_also_waits_lds, WaitCntType vector_read_wait) {
  const VectorRegisterFile file = vector_register_file(resource_kind).file;
  // A read-modify-write access produces neither a pending read nor a pending
  // write, so it tracks nothing.
  if (event.is_read == event.is_write)
    return;

  const EntityId current_id = event.instruction.instruction_id;

  if (event.is_read) {
    if (vector_read_wait == WaitCntType::NONE)
      return;
    track_pending_raw_isa(wave, ctx);
    track_vector_read(&wave.core, file, current_id, ctx.pc, reg, count, vector_read_wait);
    return;
  }

  if (vector_write_wait == WaitCntType::NONE)
    return;

  track_pending_raw_isa(wave, ctx);

  if (vector_write_wait == WaitCntType::LDS) {
    track_vector_flat_ds(&wave.core, file, current_id, ctx.pc, reg, count, /*is_flat_ds=*/false);
    return;
  }

  track_vector_write(&wave.core, file, current_id, ctx.pc, reg, count, vector_write_wait);

  if (vector_write_also_waits_lds)
    track_vector_flat_ds(&wave.core, file, current_id, ctx.pc, reg, count, /*is_flat_ds=*/true);
}

void DataHazardEngine::handle_scalar_access(EngineWaveState &wave,
                                            const EngineInstructionContext &ctx,
                                            const ResourceAccessEvent &event,
                                            WaitCntType scalar_write_wait) {
  if (!check_access_size("handle_scalar_access", event.size_bytes))
    return;

  const uint32_t sgpr = event.resource_index;
  const uint32_t count = hazard_core::bytes_to_dwords(event.size_bytes);
  const EntityId current_id = event.instruction.instruction_id;

  if (event.is_read) {
    hazard_core::for_each_pending_register_span(
        wave.core.pending_sgpr_writes, sgpr, count, hazard_core::MAX_SGPR_INDEX, current_id,
        [&](const hazard_core::PendingRegisterSpan &span) {
          const PendingAsyncOp &pending = *span.pending;
          const auto report_from =
              first_unreported_register(wave.core.reported_raw_hazards, span, [&](uint32_t reg) {
                return hazard_core::WawHazardKey{pending.instruction_id,
                                                 hazard_core::HazardRegisterKind::Scalar, reg};
              });
          if (!report_from)
            return;
          if (!claim_register_hazard(
                  wave.core.reported_raw_hazards, wave.core.reported_raw_pairs,
                  hazard_core::WawHazardKey{pending.instruction_id,
                                            hazard_core::HazardRegisterKind::Scalar, *report_from},
                  {pending.instruction_id, current_id, hazard_core::HazardRegisterKind::Scalar}))
            return;

          const std::string message = format_pending_message(
              "RAW", register_label("SGPR", 's', *report_from, span.last_register), "read", "load",
              pending);
          record_warning(ctx, event, HazardKind::RAW, ResourceKind::ScalarRegister,
                         HazardAccessKind::Read, *report_from, 0, event.size_bytes,
                         pending.wait_type, pending,
                         get_pending_raw_isa(wave, pending.instruction_id), pending.pc, message,
                         register_read_suggestion(instruction_formatter(), pending.wait_type));
        });
  }

  if (event.is_write) {
    // Scalar reads may return out of order, including reads targeting the
    // same SGPR. A later write to a pending scalar-load destination therefore
    // needs the matching KMCNT wait.
    hazard_core::for_each_pending_register_span(
        wave.core.pending_sgpr_writes, sgpr, count, hazard_core::MAX_SGPR_INDEX, current_id,
        [&](const hazard_core::PendingRegisterSpan &span) {
          const PendingAsyncOp &pending = *span.pending;
          const auto report_from =
              first_unreported_register(wave.core.reported_waw_hazards, span, [&](uint32_t reg) {
                return hazard_core::WawHazardKey{current_id,
                                                 hazard_core::HazardRegisterKind::Scalar, reg};
              });
          if (!report_from)
            return;
          if (!claim_register_hazard(
                  wave.core.reported_waw_hazards, wave.core.reported_waw_pairs,
                  {current_id, hazard_core::HazardRegisterKind::Scalar, *report_from},
                  {pending.instruction_id, current_id, hazard_core::HazardRegisterKind::Scalar}))
            return;

          const std::string message = format_pending_message(
              "WAW", register_label("SGPR", 's', *report_from, span.last_register), "written",
              "load", pending);
          record_warning(ctx, event, HazardKind::WAW, ResourceKind::ScalarRegister,
                         HazardAccessKind::Write, *report_from, 0, event.size_bytes,
                         pending.wait_type, pending,
                         get_pending_raw_isa(wave, pending.instruction_id), pending.pc, message,
                         register_write_suggestion(instruction_formatter(), pending.wait_type));
        });
  }

  if (!event.is_write || event.is_read || scalar_write_wait == WaitCntType::NONE)
    return;

  track_pending_raw_isa(wave, ctx);
  hazard_core::track_sgpr_write(&wave.core, current_id, ctx.pc, sgpr, scalar_write_wait, count);
}

void DataHazardEngine::handle_lds_access(EngineWaveState &wave, const EngineInstructionContext &ctx,
                                         const ResourceAccessEvent &event,
                                         WaitCntType local_write_wait) {
  if (!check_access_size("handle_lds_access", event.size_bytes))
    return;
  if (event.address > std::numeric_limits<uint32_t>::max()) {
    reject_event("handle_lds_access: address exceeds 32-bit LDS address range");
    return;
  }

  const uint32_t address = static_cast<uint32_t>(event.address);
  const uint32_t size = event.size_bytes;
  const EntityId current_id = event.instruction.instruction_id;

  if (event.is_read) {
    auto report_overlap = [&](const PendingAsyncOp *pending, WaitCntType fallback_wait_type) {
      if (!pending)
        return;
      const WaitCntType wait_type =
          pending->wait_type != WaitCntType::NONE ? pending->wait_type : fallback_wait_type;
      hazard_core::LdsHazardKey key{pending->instruction_id, event.instruction.instruction_id};
      if (wave.core.reported_lds_hazards.count(key))
        return;
      wave.core.reported_lds_hazards.insert(key);

      std::ostringstream resource;
      resource << "LDS address 0x" << std::hex << address << " (size " << std::dec << size << ")";
      const std::string message =
          format_pending_message("RAW", resource.str(), "read", "write", *pending);
      record_warning(ctx, event, HazardKind::RAW, ResourceKind::LocalMemory, HazardAccessKind::Read,
                     0, address, size, wait_type, *pending,
                     get_pending_raw_isa(wave, pending->instruction_id), pending->pc, message,
                     lds_read_suggestion(instruction_formatter(), wait_type));
    };

    report_overlap(hazard_core::check_lds_read_hazard(&wave.core, address, size), WaitCntType::LDS);
    report_overlap(hazard_core::check_tensor_lds_hazard(&wave.core, address, size),
                   WaitCntType::TENSOR);

    if (!event.is_write) {
      // A read outstanding on a counter of its own, as the LDS a tensor store
      // streams out is, retires with that counter rather than with DScnt.
      const WaitCntType read_wait =
          event.hazards.read_wait != WaitCntType::NONE ? event.hazards.read_wait : WaitCntType::LDS;
      track_pending_raw_isa(wave, ctx);
      hazard_core::track_lds_read(&wave.core, current_id, ctx.pc, address, size, read_wait);
    }
  }

  if (event.is_write && local_write_wait == WaitCntType::TENSOR) {
    // WAW: an earlier DScnt-tracked LDS write (e.g. a ds_store) to an overlapping
    // address may still be in flight. It retires on a different counter than this
    // tensor DMA, so without an intervening s_wait_dscnt the two writes are unordered
    // and the ds_store can land after the DMA and clobber the tensor data. The read
    // path scans both write FIFOs for RAW; the write path must do the same for WAW.
    if (const auto *pending = hazard_core::find_pending_lds_write(&wave.core, address, size)) {
      hazard_core::LdsHazardKey key{pending->instruction_id, current_id};
      if (!wave.core.reported_lds_hazards.count(key)) {
        wave.core.reported_lds_hazards.insert(key);
        std::ostringstream resource;
        resource << "LDS address 0x" << std::hex << address << " (size " << std::dec << size << ")";
        const std::string message =
            format_pending_message("WAW", resource.str(), "written", "write", *pending);
        record_warning(ctx, event, HazardKind::WAW, ResourceKind::LocalMemory,
                       HazardAccessKind::Write, 0, address, size, pending->wait_type, *pending,
                       get_pending_raw_isa(wave, pending->instruction_id), pending->pc, message,
                       lds_write_suggestion(instruction_formatter(), pending->wait_type));
      }
    }
    track_pending_raw_isa(wave, ctx);
    hazard_core::track_tensor_lds(&wave.core, current_id, ctx.pc, address, size);
  }

  if (event.is_write && !event.is_read && local_write_wait != WaitCntType::TENSOR) {
    if (const auto *pending = hazard_core::check_lds_write_hazard(&wave.core, address, size)) {
      hazard_core::LdsHazardKey key{pending->instruction_id, current_id};
      if (!wave.core.reported_lds_hazards.count(key)) {
        wave.core.reported_lds_hazards.insert(key);
        std::ostringstream resource;
        resource << "LDS address 0x" << std::hex << address << " (size " << std::dec << size << ")";
        const std::string message =
            format_pending_message("WAR", resource.str(), "written", "read", *pending);
        record_warning(ctx, event, HazardKind::WAR, ResourceKind::LocalMemory,
                       HazardAccessKind::Write, 0, address, size, pending->wait_type, *pending,
                       get_pending_raw_isa(wave, pending->instruction_id), pending->pc, message,
                       lds_write_suggestion(instruction_formatter(), pending->wait_type));
      }
    }

    if (local_write_wait != WaitCntType::NONE) {
      track_pending_raw_isa(wave, ctx);
      hazard_core::track_lds_write(&wave.core, current_id, ctx.pc, address, size, local_write_wait);
    }
  }

  if (wave.workgroup) {
    EngineLdsAccessRecord record{};
    record.wave_id = ctx.wave_id;
    record.instruction_id = event.instruction.instruction_id;
    record.dispatch_id = ctx.dispatch_id;
    record.pc = ctx.pc;
    record.address = address;
    record.size = size;
    record.is_write = event.is_write;
    record.is_atomic = event.is_atomic;
    record.raw_isa = ctx.raw_isa;
    std::lock_guard<SpinLock> lock(wave.workgroup->mutex);
    wave.workgroup->lds_epoch.push_back(record);
  }
}

void DataHazardEngine::check_global_access_for_races(const EngineInstructionContext &ctx,
                                                     const ResourceAccessEvent &event) {
  if (!check_access_size("check_global_access_for_races", event.size_bytes))
    return;

  constexpr uint64_t max_address = std::numeric_limits<uint64_t>::max();
  const uint64_t aligned_start = event.address & kGlobalShadowAlignMask;
  const uint64_t end = event.address > max_address - event.size_bytes
                           ? max_address
                           : event.address + event.size_bytes;
  auto current = make_global_access_info(ctx, event);

  std::vector<EngineWarning> warnings_to_emit;
  {
    std::lock_guard<std::shared_mutex> lock(state_.mutex);
    for (uint64_t addr = aligned_start; addr < end;) {
      auto &entry = state_.global_shadow[EngineGlobalShadowKey{ctx.dispatch_id, addr}];
      const uint8_t byte_mask = entry_byte_mask(addr, event.address, end);
      current.byte_mask = byte_mask;

      if (!entry.race_reported) {
        const EngineGlobalAccessInfo *conflicting_access = nullptr;
        const char *conflict_type = nullptr;
        // The first conflict found is the one reported, so a write conflict is
        // looked for before a read conflict rather than replacing it later.
        auto take = [&](const EngineGlobalAccessInfo *candidate, const char *type) {
          if (conflicting_access != nullptr || candidate == nullptr)
            return;
          conflicting_access = candidate;
          conflict_type = type;
        };

        // Named for what this access does to what it found: reaching an earlier
        // writer is WAW or RAW, and only a write conflicts with an earlier
        // reader, which makes that one WAR.
        const char *const against_writer = event.is_write ? "WAW" : "RAW";
        take(find_conflicting_access(entry.writers, ctx, byte_mask, /*writers_only=*/false),
             against_writer);
        // Atomics order themselves against each other, so an atomic weighs only
        // the ordinary history while an ordinary access also weighs the atomics.
        if (!event.is_atomic)
          take(find_conflicting_access(entry.atomics, ctx, byte_mask, /*writers_only=*/true),
               against_writer);
        if (event.is_write) {
          take(find_conflicting_access(entry.readers, ctx, byte_mask, /*writers_only=*/false),
               "WAR");
          if (!event.is_atomic)
            take(find_conflicting_access(entry.atomics, ctx, byte_mask, /*writers_only=*/false),
                 "WAR");
        }

        if (conflicting_access) {
          entry.race_reported = true;
          EngineWarning warning =
              make_global_race_warning(ctx, event, addr, *conflicting_access, conflict_type);
          state_.warnings.push_back(warning);
          warnings_to_emit.push_back(std::move(warning));
        }
      }

      if (event.is_atomic) {
        record_access(entry.atomics, ctx, current);
      } else {
        if (event.is_write)
          record_access(entry.writers, ctx, current);
        if (event.is_read) {
          // Bytes this workgroup wrote itself are already spoken for by the
          // retained writer, which a later foreign access conflicts with just as
          // it would with the read. The bytes it did not write still need a
          // reader of their own, or a foreign write to them finds nothing.
          EngineGlobalAccessInfo read_access = current;
          read_access.byte_mask = static_cast<uint8_t>(
              byte_mask & ~coverage_by_same_workgroup(entry.writers, ctx, byte_mask));
          if (read_access.byte_mask != 0)
            record_access(entry.readers, ctx, read_access);
        }
      }

      if (addr > max_address - kGlobalShadowAlignment)
        break;
      addr += kGlobalShadowAlignment;
    }
  }

  for (const auto &warning : warnings_to_emit)
    emit_warning(warning);
}

void DataHazardEngine::flush_workgroup_epoch(const EngineWorkgroupKey &key) {
  std::shared_ptr<EngineWorkgroupState> workgroup;
  {
    std::shared_lock<std::shared_mutex> lock(state_.mutex);
    auto it = state_.workgroups.find(key);
    if (it != state_.workgroups.end())
      workgroup = it->second;
  }
  if (!workgroup)
    return;

  std::vector<EngineLdsAccessRecord> epoch;
  {
    std::lock_guard<SpinLock> lock(workgroup->mutex);
    epoch = std::move(workgroup->lds_epoch);
    workgroup->lds_epoch.clear();
  }
  if (!epoch.empty())
    check_lds_epoch_for_races(key, epoch);
}

void DataHazardEngine::check_lds_epoch_for_races(const EngineWorkgroupKey &key,
                                                 const std::vector<EngineLdsAccessRecord> &epoch) {
  static constexpr const char *kMessageSuffix = "without workgroup barrier";
  static constexpr const char *kSuggestion =
      "Add s_barrier_signal / s_barrier_wait between conflicting LDS accesses";

  if (epoch.size() < 2)
    return;

  std::vector<size_t> indices(epoch.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&](size_t lhs, size_t rhs) {
    const auto &left = epoch[lhs];
    const auto &right = epoch[rhs];
    if (left.address != right.address)
      return left.address < right.address;
    if (left.wave_id != right.wave_id)
      return left.wave_id < right.wave_id;
    return left.instruction_id < right.instruction_id;
  });

  for (size_t ii = 0; ii < indices.size(); ++ii) {
    const auto &first = epoch[indices[ii]];
    uint64_t first_end = static_cast<uint64_t>(first.address) + first.size;

    for (size_t jj = ii + 1; jj < indices.size(); ++jj) {
      const auto &second = epoch[indices[jj]];
      if (second.address >= first_end)
        break;
      if (first.wave_id == second.wave_id)
        continue;
      if (!first.is_write && !second.is_write)
        continue;
      if (first.is_atomic && second.is_atomic)
        continue;

      const EntityId wave_lo = std::min(first.wave_id, second.wave_id);
      const EntityId wave_hi = std::max(first.wave_id, second.wave_id);
      const uint32_t overlap_addr = std::max(first.address, second.address);

      const auto &writer = first.is_write ? first : second;
      const auto &other = first.is_write ? second : first;
      const char *other_action = other.is_write ? "writes" : "reads";

      {
        const EngineLdsRaceKey race_key{key.dispatch_id, key.cluster_id, key.workgroup_id,
                                        overlap_addr,    wave_lo,        wave_hi,
                                        writer.pc,       other.pc};
        std::lock_guard<std::shared_mutex> lock(state_.mutex);
        if (!state_.reported_lds_races.insert(race_key).second)
          continue;
      }

      std::ostringstream msg;
      msg << "LDS data race: Wave " << writer.wave_id << " writes LDS address 0x" << std::hex
          << writer.address << " (size " << std::dec << writer.size << ") and Wave "
          << other.wave_id << " " << other_action << " LDS address 0x" << std::hex << other.address
          << " (size " << std::dec << other.size << ") " << kMessageSuffix;

      EngineWarning warning;
      warning.finding.kind = HazardKind::LocalMemoryRace;
      warning.finding.resource_kind = ResourceKind::LocalMemory;
      warning.finding.access_kind =
          other.is_write ? HazardAccessKind::Write : HazardAccessKind::Read;
      warning.finding.instruction.instruction_id = other.instruction_id;
      warning.finding.instruction.execution.dispatch_id = key.dispatch_id;
      warning.finding.instruction.execution.cluster_id = key.cluster_id;
      warning.finding.instruction.execution.workgroup_id = key.workgroup_id;
      warning.finding.instruction.execution.wave_id = other.wave_id;
      warning.finding.instruction.pc = other.pc;
      warning.finding.instruction.raw_isa = other.raw_isa;
      warning.finding.address = overlap_addr;
      warning.finding.size_bytes = other.size;
      warning.finding.has_source_instruction = true;
      warning.finding.source_instruction.instruction_id = writer.instruction_id;
      warning.finding.source_instruction.execution.dispatch_id = key.dispatch_id;
      warning.finding.source_instruction.execution.cluster_id = key.cluster_id;
      warning.finding.source_instruction.execution.workgroup_id = key.workgroup_id;
      warning.finding.source_instruction.execution.wave_id = writer.wave_id;
      warning.finding.source_instruction.pc = writer.pc;
      warning.finding.source_instruction.raw_isa = writer.raw_isa;
      warning.finding.message_template = msg.str();
      warning.finding.suggestion_template = kSuggestion;
      warning.message = msg.str();
      warning.suggestion = kSuggestion;
      warning.source_raw_isa = writer.raw_isa;
      warning.source_pc = writer.pc;
      warning.has_source = true;
      {
        std::lock_guard<std::shared_mutex> lock(state_.mutex);
        state_.warnings.push_back(warning);
      }
      emit_warning(warning);
    }
  }
}

void DataHazardEngine::emit_warning(const EngineWarning &warning) {
  if (warning_sink_)
    warning_sink_->emit_warning(warning);
}

const SimulatorInstructionFormatter *DataHazardEngine::instruction_formatter() const {
  return instruction_formatter_;
}

} // namespace hazard_core
