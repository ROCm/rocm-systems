// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/command_processor.h"

#include "rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h"

#include <algorithm>
#include <limits>
#include <type_traits>

namespace rocjitsu::amdgpu {

// Runtime scheduling and resource requirements. Architectural registers remain in the
// debugger-editable CWSR records; this image is the authority for allocations.
struct CommandProcessor::QueueContext {
  struct Group {
    uint32_t dispatch = 0;
    uint32_t wg = 0;
    uint32_t lds_bytes = 0;
    uint32_t remaining_waves = 0;
    bool has_barriers = false;
    ComputeUnitCore::WorkgroupBarriers barriers{};
  };
  struct Wave {
    uint32_t group = 0;
    uint32_t sgprs = 0;
    uint32_t vgprs = 0;
    uint32_t width = 0;
    uint32_t packet = 0;
    uint32_t wave_in_group = 0;
    uint32_t scratch_lane_size = 0;
    // Allocation ownership is separate from the debugger-editable FLAT_SCRATCH pointer.
    uint64_t scratch_backing_base = 0;
    uint64_t scratch_backing_bytes = 0;
    uint64_t debug_id = 0;
    uint64_t code_bias = 0;
    std::array<uint32_t, 3> coordinates{};
    WfState state = WfState::RUNNING;
    uint32_t sleep_cycles = 0;
    uint32_t sched_mode = 0;
    bool vgpr_msb_hazard = false;
    bool runtime_suspended = false;
    Wavefront::BarrierWaitState barrier;
    Wavefront::DebugStopState stop;
    uint64_t queue_exceptions = 0;
    uint64_t runtime_exceptions = 0;
    uint32_t trap_saved_status = 0;
    uint64_t trap_saved_exec = 0;
    bool interrupt_sent = false;
    bool self_halted = false;
  };
  std::vector<Group> groups;
  std::vector<Wave> waves;
};

namespace {

// MODEL ASSUMPTION: AMD's public MQD definitions do not specify a complete CP
// firmware checkpoint encoding. This is a rocjitsu-private, versioned format,
// NOT an AMD register layout: each scalar occupies one little-endian uint64.
// The field visitors below are shared by writing and reading. Host pointers,
// callbacks, and address-space leases never enter the saved image.
constexpr uint64_t kQueueContextPayloadMagic = 0x3158544350434a52ULL; // "RJCPCTX1"
constexpr uint64_t kQueueContextVersion = 3;

class ContextWriter {
public:
  template <typename T> bool value(const T &field) {
    static_assert(std::is_integral_v<T> || std::is_enum_v<T>);
    const uint64_t word = static_cast<uint64_t>(field);
    for (unsigned byte = 0; byte < sizeof(word); ++byte)
      bytes.push_back(static_cast<uint8_t>(word >> (byte * 8)));
    return true;
  }

  template <typename T, size_t N> bool value(const std::array<T, N> &fields) {
    for (const auto &field : fields)
      value(field);
    return true;
  }

  template <typename... T> bool operator()(const T &...fields) { return (value(fields) && ...); }

  template <size_t N> bool sparse(const std::array<uint32_t, N> &fields) {
    value(static_cast<uint64_t>(std::ranges::count_if(fields, [](auto v) { return v != 0; })));
    for (size_t index = 0; index < N; ++index)
      if (fields[index] != 0)
        (*this)(index, fields[index]);
    return true;
  }

  std::vector<uint8_t> bytes;
};

class ContextReader {
public:
  explicit ContextReader(std::span<const uint8_t> bytes) : bytes_(bytes) {}

  template <typename T> bool value(T &field) {
    if constexpr (std::is_enum_v<T>) {
      std::underlying_type_t<T> raw;
      if (!value(raw))
        return false;
      field = static_cast<T>(raw);
    } else {
      static_assert(std::is_integral_v<T>);
      if (bytes_.size() < sizeof(uint64_t))
        return false;
      uint64_t word = 0;
      for (unsigned byte = 0; byte < sizeof(word); ++byte)
        word |= static_cast<uint64_t>(bytes_[byte]) << (byte * 8);
      bytes_ = bytes_.subspan(sizeof(word));
      if (word > static_cast<uint64_t>(std::numeric_limits<T>::max()))
        return false;
      field = static_cast<T>(word);
    }
    return true;
  }

  template <typename T, size_t N> bool value(std::array<T, N> &fields) {
    for (auto &field : fields)
      if (!value(field))
        return false;
    return true;
  }

  template <typename... T> bool operator()(T &...fields) { return (value(fields) && ...); }

  template <size_t N> bool sparse(std::array<uint32_t, N> &fields) {
    uint64_t count;
    if (!value(count) || count > N || count > bytes_.size() / (2 * sizeof(uint64_t)))
      return false;
    fields.fill(0);
    uint64_t previous = 0;
    for (uint64_t i = 0; i < count; ++i) {
      uint64_t index;
      uint32_t data;
      if (!(*this)(index, data) || index >= N || (i != 0 && index <= previous) || data == 0)
        return false;
      fields[index] = data;
      previous = index;
    }
    return true;
  }

  size_t remaining() const { return bytes_.size(); }

private:
  std::span<const uint8_t> bytes_;
};

template <typename Archive, typename Entry> bool dispatch_fields(Archive &archive, Entry &entry) {
  return archive(
      entry.dispatch_id, entry.queue_packet_id, entry.aql_packet_id, entry.aql_packet_address,
      entry.owns_aql_ring_slot, entry.kernel_entry_pc, entry.code_load_bias,
      entry.wfs_per_workgroup, entry.sgprs_per_wf, entry.vgprs_per_wf, entry.kernarg_addr,
      entry.kernarg_size, entry.num_user_sgprs, entry.kernel_code_properties,
      entry.num_named_barriers, entry.kernel_wave_size, entry.kernarg_preload,
      entry.initial_mode_raw, entry.dispatch_ptr, entry.queue_ptr, entry.workgroup_id_offset,
      entry.grid_size_x, entry.grid_size_y, entry.grid_size_z, entry.workgroup_origin,
      entry.grid_wgs_x, entry.grid_wgs_y, entry.grid_wgs_z, entry.grid_yz_valid,
      entry.cluster_count_x, entry.cluster_count_y, entry.cluster_count_z, entry.cluster_size_x,
      entry.cluster_size_y, entry.cluster_size_z, entry.wgp_mode, entry.enable_wg_id_x,
      entry.enable_wg_id_y, entry.enable_wg_id_z, entry.enable_wg_info,
      entry.enable_vgpr_workitem_id, entry.workgroup_size_x, entry.workgroup_size_y,
      entry.workgroup_size_z, entry.scratch_backing_addr, entry.private_segment_fixed_size,
      entry.scratch_wave_limit_per_se, entry.scratch_wave_stride_per_se, entry.scratch_wave_bytes,
      entry.group_segment_fixed_size, entry.scratch_use_once, entry.total_wgs, entry.dispatched_wgs,
      entry.completed_wgs, entry.completion_signal, entry.profiling_start_timestamp, entry.kind,
      entry.host_signal, entry.wait_for_predecessors, entry.blocks_following,
      entry.aql_pm4_ib_address, entry.aql_pm4_ib_dwords, entry.completion_notified,
      entry.execution_begun, entry.acquire_invalidate, entry.terminal_faulted,
      entry.completion_publication.phase);
}

template <typename Archive, typename Queue> bool queue_fields(Archive &archive, Queue &queue) {
  return archive(queue.ring_base_va, queue.ring_size, queue.read_ptr_va, queue.write_ptr_va,
                 queue.doorbell_offset, queue.packet_format, queue.scheduling_percentage,
                 queue.process_id, queue.queue_desc_va, queue.uses_kfd_queue_abi,
                 queue.fetch_cursor, queue.next_dispatch_idx, queue.implicit_barrier_next,
                 queue.last_doorbell, queue.runtime_suspended, queue.exception_suspended,
                 queue.faulted, queue.activity_generation, queue.commands.indirect_base,
                 queue.commands.predicate_pass) &&
         archive.sparse(queue.commands.sh_registers) &&
         archive.sparse(queue.commands.context_registers) &&
         archive.sparse(queue.commands.uconfig_registers) &&
         archive(queue.scratch_request.phase, queue.scratch_request.packet_index,
                 queue.scratch_request.status, queue.scratch_request.signal_address,
                 queue.scratch_request.mailbox_pointer, queue.scratch_request.event_id,
                 queue.scratch_reclaim.phase, queue.scratch_reclaim.signal_address,
                 queue.scratch_reclaim.mailbox_pointer, queue.scratch_reclaim.event_id);
}

template <typename Archive, typename Group> bool group_fields(Archive &archive, Group &group) {
  if (!archive(group.dispatch, group.wg, group.lds_bytes, group.remaining_waves,
               group.has_barriers))
    return false;
  if (!group.has_barriers)
    return true;
  if (!archive(group.barriers.allocated_count))
    return false;
  for (auto &counter : group.barriers.named)
    if (!archive(counter.member_count, counter.signal_count))
      return false;
  for (auto &counter : group.barriers.workgroup)
    if (!archive(counter.member_count, counter.signal_count))
      return false;
  return true;
}

template <typename Archive, typename Wave> bool wave_fields(Archive &archive, Wave &wave) {
  return archive(wave.group, wave.sgprs, wave.vgprs, wave.width, wave.packet, wave.wave_in_group,
                 wave.scratch_lane_size, wave.scratch_backing_base, wave.scratch_backing_bytes,
                 wave.debug_id, wave.code_bias, wave.coordinates, wave.state, wave.sleep_cycles,
                 wave.sched_mode, wave.vgpr_msb_hazard, wave.runtime_suspended,
                 wave.barrier.named_barrier_id, wave.barrier.complete, wave.barrier.waiting_bit,
                 wave.stop.trapsts, wave.stop.mode_raw, wave.stop.gfx12_trap_ctrl_raw,
                 wave.stop.trap_id, wave.stop.debug_halted, wave.stop.single_step,
                 wave.stop.fatal_exception_pending, wave.stop.fatal_exception_cwsr_valid,
                 wave.queue_exceptions, wave.runtime_exceptions, wave.trap_saved_status,
                 wave.trap_saved_exec, wave.interrupt_sent, wave.self_halted);
}

bool supported_dispatch(const DispatchEntry &entry) {
  // Current MES placement is not fanned out. Cross-XCD ownership and external
  // PM4 callbacks require their own context model rather than saving pointers.
  // Cluster support does not yet cover the MES save/release/rebuild lifecycle.
  // Cluster checkpoint support will be added separately.
  return !entry.grid_completion && !entry.enabled_cus && !entry.pm4_abi && !entry.pm4_failure &&
         !entry.pm4_scratch_pool && !entry.fanout_peer && entry.shard.is_unsharded() &&
         !entry.wgp_mode && entry.cluster_size() == 1;
}

} // namespace

VmAccessOutcome CommandProcessor::suspend_queue_context(const QueueIdentity &identity) {
  std::unique_lock structure_lock(queue_structure_mutex_);
  std::lock_guard lock(hw_queue_mutex_);
  ComputeQueueRecord *queue = find_compute_queue(identity.queue_id, identity.process_id);
  if (!queue || queue->address_space != identity.address_space)
    return VmAccessOutcome::Malformed;
  if (!queue->context_save_suspension)
    queue->context_save_suspension.emplace(queue->debug_suspended);
  set_queue_debug_suspended(identity.queue_id, identity.process_id, true);
  bool pending = false;
  for (auto *cu : cus_)
    cu->with_wave_state_locked([&] {
      for (uint32_t slot = 0; slot < cu->num_wf_slots(); ++slot) {
        auto *wave = cu->wf(slot);
        if (!wave || wave->is_halted() || wave->process_id() != identity.process_id ||
            wave->queue_id() != identity.queue_id ||
            wave->address_space() != identity.address_space)
          continue;
        // Let the handler return before freezing its architectural PC.
        if (wave->in_trap_handler()) {
          pending = true;
          continue;
        }
        if (!wave->debug_suspended()) {
          queue->context_save_suspension->newly_suspended.push_back(
              {cu, slot, wave->dispatch_generation()});
          wave->set_debug_suspended(true);
        }
      }
    });
  return pending ? VmAccessOutcome::Unavailable : VmAccessOutcome::Complete;
}

void CommandProcessor::cancel_queue_context_save(const QueueIdentity &identity) {
  std::unique_lock structure_lock(queue_structure_mutex_);
  std::lock_guard lock(hw_queue_mutex_);
  ComputeQueueRecord *queue = find_compute_queue(identity.queue_id, identity.process_id);
  if (!queue || queue->address_space != identity.address_space || !queue->context_save_suspension)
    return;
  auto suspension = std::move(*queue->context_save_suspension);
  queue->context_save_suspension.reset();
  for (const auto &saved : suspension.newly_suspended) {
    saved.cu->with_wave_state_locked([&] {
      auto *wave = saved.cu->wf(saved.slot);
      // A slot may have been retired or reused while a save was pending.
      if (wave && !wave->is_halted() && wave->dispatch_generation() == saved.generation &&
          wave->process_id() == identity.process_id && wave->queue_id() == identity.queue_id &&
          wave->address_space() == identity.address_space)
        wave->set_debug_suspended(false);
    });
    saved.cu->schedule_work_async();
  }
  set_queue_debug_suspended(identity.queue_id, identity.process_id, suspension.debug_suspended);
}

QueueContextSaveOutcome CommandProcessor::save_queue_context(const QueueIdentity &identity,
                                                             std::vector<uint8_t> &context) {
  std::unique_lock structure_lock(queue_structure_mutex_);
  std::lock_guard lock(hw_queue_mutex_);
  ComputeQueueRecord *queue = find_compute_queue(identity.queue_id, identity.process_id);
  if (!queue || queue->address_space != identity.address_space || !queue->debug_suspended)
    return VmAccessOutcome::Malformed;
  if (queue->xcd_fanout || queue->fanout_replica || !queue->commands.submissions.empty() ||
      !queue->dispatches.entries.empty() || queue->commands.saved_context_registers)
    return VmAccessOutcome::Faulted;
  drain_pending_wg_completions();
  if (queue->read_pointer_journal.publication_pending() || queue->aql_slot_release ||
      queue->idle_publication.active() || queue->scratch_request.publication_pending() ||
      queue->scratch_reclaim.publication_pending() || queue->command_access)
    return VmAccessOutcome::Unavailable;
  for (const auto &entry : queue->entries) {
    if (!supported_dispatch(entry))
      return Unsupported{};
    if (entry.completion_publication.phase != CompletionPublicationPhase::ExecutionEnd &&
        entry.completion_publication.phase != CompletionPublicationPhase::Complete)
      return VmAccessOutcome::Unavailable;
  }
  ContextWriter writer;
  uint64_t reclaim_entry = 0;
  if (queue->scratch_reclaim.active()) {
    const auto entry = std::ranges::find(queue->entries, queue->scratch_reclaim.dispatch_id,
                                         &DispatchEntry::dispatch_id);
    if (entry == queue->entries.end())
      return VmAccessOutcome::Malformed;
    reclaim_entry = std::distance(queue->entries.begin(), entry);
  }
  writer(kQueueContextPayloadMagic, kQueueContextVersion);
  queue_fields(writer, *queue);
  writer(queue->read_pointer_journal.initialized(), queue->read_pointer_journal.cursor(),
         queue->entries.size(), reclaim_entry);
  for (const auto &entry : queue->entries) {
    dispatch_fields(writer, entry);
    const auto metadata = dispatch_launch_metadata_.find(entry.dispatch_id);
    const bool has_metadata = metadata != dispatch_launch_metadata_.end();
    if (!has_metadata && entry.kernel_code_properties != 0 && entry.queue_ptr != 0)
      return VmAccessOutcome::Malformed;
    writer(has_metadata);
    if (has_metadata)
      writer(metadata->second.scratch_resource_descriptor, metadata->second.write_dispatch_id);
  }
  QueueContext saved;
  for (auto *cu : cus_) {
    const auto status = cu->with_wave_state_locked([&]() -> QueueContextSaveOutcome {
      for (uint32_t slot = 0; slot < cu->num_wf_slots(); ++slot) {
        auto *wave = cu->wf(slot);
        if (!wave || wave->is_halted() || wave->process_id() != identity.process_id ||
            wave->queue_id() != identity.queue_id ||
            wave->address_space() != identity.address_space)
          continue;
        if (!wave->debug_suspended() || wave->in_trap_handler() || !wave->wait_counters().empty())
          return VmAccessOutcome::Unavailable;
        if (wave->state() == WfState::GWS_WAIT || wave->cluster_size() != 1 ||
            &wave->lds() != &cu->lds())
          return Unsupported{};
        auto group = std::ranges::find_if(saved.groups, [&](const auto &saved_group) {
          return saved_group.dispatch == wave->dispatch_id() && saved_group.wg == wave->wg_id();
        });
        if (group == saved.groups.end()) {
          QueueContext::Group saved_group{
              .dispatch = wave->dispatch_id(), .wg = wave->wg_id(), .lds_bytes = wave->lds_size()};
          const auto key = wg_key(saved_group.dispatch, saved_group.wg);
          if (const auto remaining = cu->active_wgs_.find(key); remaining != cu->active_wgs_.end())
            saved_group.remaining_waves = remaining->second;
          if (const auto barriers = cu->barrier_wgs_.find(key);
              barriers != cu->barrier_wgs_.end()) {
            saved_group.has_barriers = true;
            saved_group.barriers = barriers->second;
          }
          saved.groups.push_back(saved_group);
          group = std::prev(saved.groups.end());
        }
        QueueContext::Wave saved_wave;
        saved_wave.group = std::distance(saved.groups.begin(), group);
        saved_wave.sgprs = wave->num_sgprs();
        saved_wave.vgprs = wave->num_vgprs();
        saved_wave.width = wave->wf_size();
        saved_wave.packet = wave->queue_packet_id();
        saved_wave.wave_in_group = wave->wave_in_group();
        saved_wave.scratch_lane_size = wave->scratch_lane_size();
        if (const auto &scratch = wave->scratch_lease()) {
          saved_wave.scratch_backing_base = scratch->base();
          saved_wave.scratch_backing_bytes = scratch->size();
        } else if (saved_wave.scratch_lane_size != 0) {
          return VmAccessOutcome::Faulted;
        }
        saved_wave.debug_id = wave->debug_wave_id();
        saved_wave.code_bias = wave->code_load_bias();
        saved_wave.coordinates = wave->wg_coord();
        saved_wave.state = wave->state();
        // Quiesced memory has no pipeline to wait on after reallocation.
        if (saved_wave.state == WfState::WAITCNT || saved_wave.state == WfState::VM_RETRY)
          saved_wave.state = WfState::RUNNING;
        saved_wave.sleep_cycles = wave->sleep_cycles();
        saved_wave.sched_mode = wave->wave_sched_mode_raw();
        saved_wave.vgpr_msb_hazard = wave->setreg_vgpr_msb_hazard();
        saved_wave.runtime_suspended = wave->runtime_suspended();
        saved_wave.barrier = wave->barrier_wait_state();
        saved_wave.stop = wave->debug_stop_state();
        saved_wave.queue_exceptions = wave->trap_queue_exception_status();
        saved_wave.runtime_exceptions = wave->trap_runtime_exception_status();
        saved_wave.trap_saved_status = wave->trap_saved_status();
        saved_wave.trap_saved_exec = wave->trap_saved_exec();
        saved_wave.interrupt_sent = wave->trap_interrupt_sent();
        saved_wave.self_halted = wave->self_halted();
        saved.waves.push_back(saved_wave);
      }
      return VmAccessOutcome::Complete;
    });
    if (status != QueueContextSaveOutcome{VmAccessOutcome::Complete})
      return status;
  }
  writer(saved.groups.size(), saved.waves.size());
  for (auto &group : saved.groups)
    group_fields(writer, group);
  for (auto &wave : saved.waves)
    wave_fields(writer, wave);
  context = std::move(writer.bytes);
  return VmAccessOutcome::Complete;
}

VmAccessOutcome CommandProcessor::restore_queue_context(const QueueIdentity &identity,
                                                        std::span<const uint8_t> context) {
  std::unique_lock structure_lock(queue_structure_mutex_);
  std::lock_guard lock(hw_queue_mutex_);
  ComputeQueueRecord *queue = find_compute_queue(identity.queue_id, identity.process_id);
  if (!queue || queue->address_space != identity.address_space || !queue->debug_suspended ||
      queue->has_pending_commands() || queue->idle_publication.active() ||
      queue->scratch_request.active() || queue->scratch_reclaim.active())
    return VmAccessOutcome::Malformed;
  ContextReader reader(context);
  uint64_t magic, version;
  ComputeQueueRecord restored(static_cast<const ComputeQueueConfig &>(*queue));
  bool consumer_initialized;
  uint64_t consumer, count, reclaim_entry;
  if (!reader(magic, version) || magic != kQueueContextPayloadMagic ||
      version != kQueueContextVersion || !queue_fields(reader, restored) ||
      !reader(consumer_initialized, consumer, count, reclaim_entry) ||
      count > reader.remaining() / sizeof(uint64_t) || restored.next_dispatch_idx > count ||
      (restored.scratch_reclaim.active() && reclaim_entry >= count) ||
      restored.packet_format != queue->packet_format || queue->xcd_fanout ||
      queue->fanout_replica || restored.ring_base_va != queue->ring_base_va ||
      restored.ring_size != queue->ring_size || restored.read_ptr_va != queue->read_ptr_va ||
      restored.write_ptr_va != queue->write_ptr_va ||
      restored.doorbell_offset != queue->doorbell_offset ||
      restored.process_id != queue->process_id || restored.queue_desc_va != queue->queue_desc_va ||
      restored.uses_kfd_queue_abi != queue->uses_kfd_queue_abi ||
      (restored.scratch_request.phase != QueueScratchRequestPhase::Inactive &&
       restored.scratch_request.phase != QueueScratchRequestPhase::WaitingForRuntime) ||
      (restored.scratch_reclaim.phase != QueueScratchReclaimPhase::Inactive &&
       restored.scratch_reclaim.phase != QueueScratchReclaimPhase::WaitingForDispatch &&
       restored.scratch_reclaim.phase != QueueScratchReclaimPhase::WaitingForRuntime))
    return VmAccessOutcome::Malformed;
  auto access = gpu_vm_->snapshot_pinned(identity.address_space);
  if (!access)
    return VmAccessOutcome::Unavailable;
  auto execution_access = std::make_shared<const GpuVmAccess>(*access);
  std::vector<std::optional<DispatchLaunchMetadata>> launch_metadata;
  for (uint64_t i = 0; i < count; ++i) {
    DispatchEntry entry{};
    if (!dispatch_fields(reader, entry) || entry.kind <= DispatchPacketKind::Unset ||
        entry.kind > DispatchPacketKind::NonKernel || entry.completed_wgs > entry.dispatched_wgs ||
        entry.dispatched_wgs > entry.total_wgs ||
        (entry.completion_publication.phase != CompletionPublicationPhase::ExecutionEnd &&
         entry.completion_publication.phase != CompletionPublicationPhase::Complete))
      return VmAccessOutcome::Malformed;
    bool has_metadata;
    if (!reader(has_metadata) ||
        (!has_metadata && entry.kernel_code_properties != 0 && entry.queue_ptr != 0))
      return VmAccessOutcome::Malformed;
    std::optional<DispatchLaunchMetadata> metadata;
    if (has_metadata) {
      metadata.emplace();
      if (!reader(metadata->scratch_resource_descriptor, metadata->write_dispatch_id))
        return VmAccessOutcome::Malformed;
    }
    launch_metadata.push_back(std::move(metadata));
    entry.process_id = identity.process_id;
    entry.queue_id = identity.queue_id;
    entry.address_space = identity.address_space;
    entry.execution_access = execution_access;
    entry.interrupt_sink = queue->interrupt_sink;
    entry.doorbell_offset = queue->doorbell_offset;
    restored.entries.push_back(std::move(entry));
  }
  QueueContext saved;
  uint64_t group_count, wave_count;
  if (!reader(group_count, wave_count) || group_count > reader.remaining() / sizeof(uint64_t) ||
      wave_count > reader.remaining() / sizeof(uint64_t))
    return VmAccessOutcome::Malformed;
  for (uint64_t i = 0; i < group_count; ++i) {
    QueueContext::Group group;
    if (!group_fields(reader, group) ||
        group.barriers.allocated_count > ComputeUnitCore::kMaxNamedBarriers)
      return VmAccessOutcome::Malformed;
    for (const auto &previous : saved.groups)
      if (previous.dispatch == group.dispatch && previous.wg == group.wg)
        return VmAccessOutcome::Malformed;
    saved.groups.push_back(group);
  }
  for (uint64_t i = 0; i < wave_count; ++i) {
    QueueContext::Wave wave;
    if (!wave_fields(reader, wave) || wave.group >= saved.groups.size())
      return VmAccessOutcome::Malformed;
    if ((wave.scratch_lane_size != 0) != (wave.scratch_backing_bytes != 0) ||
        wave.scratch_backing_bytes < uint64_t{wave.scratch_lane_size} * wave.width ||
        (wave.scratch_backing_bytes != 0 &&
         wave.scratch_backing_bytes - 1 > UINT64_MAX - wave.scratch_backing_base))
      return VmAccessOutcome::Malformed;
    if (wave.sgprs == 0 || wave.vgprs == 0 || (wave.width != 32 && wave.width != 64) ||
        !std::ranges::any_of(cus_,
                             [&](const auto *cu) {
                               return wave.sgprs <= cu->config().sgprs_per_wf &&
                                      wave.vgprs <= cu->config().vgprs_per_wf &&
                                      wave.width <= isa_properties(cu->config().arch).wave_size_max;
                             }) ||
        wave.state == WfState::HALTED || wave.state == WfState::GWS_WAIT ||
        wave.state > WfState::ENDING ||
        wave.barrier.named_barrier_id > ComputeUnitCore::kMaxNamedBarriers ||
        (wave.barrier.waiting_bit != 0xff &&
         wave.barrier.waiting_bit >= wave.barrier.complete.size()))
      return VmAccessOutcome::Malformed;
    saved.waves.push_back(wave);
  }
  for (size_t i = 0; i < saved.groups.size(); ++i) {
    const auto count = std::ranges::count(saved.waves, i, &QueueContext::Wave::group);
    if (count == 0 ||
        (saved.groups[i].remaining_waves != 0 && saved.groups[i].remaining_waves != count))
      return VmAccessOutcome::Malformed;
  }
  if (reader.remaining() != 0)
    return VmAccessOutcome::Malformed;
  // Saved dispatch IDs are references within the image, not new execution IDs.
  std::unordered_map<uint32_t, uint32_t> dispatch_ids;
  for (size_t i = 0; i < restored.entries.size(); ++i) {
    auto &entry = restored.entries[i];
    if (dispatch_ids.contains(entry.dispatch_id))
      return VmAccessOutcome::Malformed;
    dispatch_ids.emplace(entry.dispatch_id, 0);
  }
  for (const auto &group : saved.groups) {
    if (!restored.entries.empty() && !dispatch_ids.contains(group.dispatch))
      return VmAccessOutcome::Malformed;
    dispatch_ids.try_emplace(group.dispatch, 0);
  }

  // Keep all affected CUs stable until resource admission and rollback finish.
  std::vector<std::unique_lock<std::recursive_mutex>> wave_locks;
  for (auto *cu : cus_)
    wave_locks.emplace_back(cu->wave_state_mutex_);
  std::vector<std::pair<ComputeUnitCore *, Wavefront *>> reserved;
  std::vector<uint32_t> lds_bases;
  std::vector<ComputeUnitCore *> group_cus;
  std::vector<Wavefront *> waves(saved.waves.size());
  const auto rollback = [&] {
    for (const auto &[cu, wave] : reserved)
      cu->free_wavefront_resources(*wave);
    for (size_t i = 0; i < lds_bases.size(); ++i)
      group_cus[i]->release_lds(lds_bases[i], saved.groups[i].lds_bytes);
  };
  for (size_t i = 0; i < saved.groups.size(); ++i) {
    const auto &group = saved.groups[i];
    const uint32_t count = std::ranges::count(saved.waves, i, &QueueContext::Wave::group);
    const auto placement = std::ranges::find_if(cus_, [&](const auto *cu) {
      return cu->can_accept_workgroup(count, group.lds_bytes) &&
             std::ranges::all_of(saved.waves, [&](const auto &saved_wave) {
               return saved_wave.group != i ||
                      (saved_wave.sgprs <= cu->config().sgprs_per_wf &&
                       saved_wave.vgprs <= cu->config().vgprs_per_wf &&
                       saved_wave.width <= isa_properties(cu->config().arch).wave_size_max);
             });
    });
    if (placement == cus_.end()) {
      rollback();
      return VmAccessOutcome::Unavailable;
    }
    auto *cu = *placement;
    group_cus.push_back(cu);
    lds_bases.push_back(cu->allocate_lds(group.lds_bytes));
    for (size_t index = 0; index < saved.waves.size(); ++index) {
      const auto &saved_wave = saved.waves[index];
      if (saved_wave.group != i)
        continue;
      auto *wave =
          cu->dispatch_wf(group.wg, 0, saved_wave.sgprs, saved_wave.vgprs, saved_wave.width);
      if (!wave) {
        rollback();
        return VmAccessOutcome::Unavailable;
      }
      wave->set_debug_suspended(true);
      waves[index] = wave;
      reserved.emplace_back(cu, wave);
      if (saved_wave.scratch_backing_bytes != 0) {
        auto scratch = scratch_reservations_->try_reserve(identity.address_space,
                                                          saved_wave.scratch_backing_base,
                                                          saved_wave.scratch_backing_bytes);
        if (!scratch) {
          rollback();
          return VmAccessOutcome::Unavailable;
        }
        wave->set_scratch_lease(std::move(scratch));
      }
    }
  }
  for (auto &[old_id, new_id] : dispatch_ids)
    new_id = allocate_dispatch_id();
  for (size_t i = 0; i < restored.entries.size(); ++i) {
    auto &entry = restored.entries[i];
    entry.dispatch_id = dispatch_ids.at(entry.dispatch_id);
    if (launch_metadata[i])
      dispatch_launch_metadata_.insert_or_assign(entry.dispatch_id, *launch_metadata[i]);
  }
  for (size_t i = 0; i < saved.groups.size(); ++i) {
    const auto &group = saved.groups[i];
    auto *cu = group_cus[i];
    const auto dispatch = dispatch_ids.at(group.dispatch);
    if (group.remaining_waves != 0) {
      cu->begin_workgroup(dispatch, group.wg, group.remaining_waves);
      if (group.has_barriers)
        cu->barrier_wgs_[wg_key(dispatch, group.wg)] = group.barriers;
    }
    for (size_t index = 0; index < saved.waves.size(); ++index) {
      const auto &saved_wave = saved.waves[index];
      if (saved_wave.group != i)
        continue;
      auto *wave = waves[index];
      wave->set_dispatch_id(dispatch);
      wave->set_queue_packet_id(saved_wave.packet);
      wave->set_wave_in_group(saved_wave.wave_in_group);
      wave->set_debug_wave_id(saved_wave.debug_id);
      wave->set_code_load_bias(saved_wave.code_bias);
      wave->set_wg_coord(saved_wave.coordinates[0], saved_wave.coordinates[1],
                         saved_wave.coordinates[2]);
      wave->set_queue_id(identity.queue_id);
      wave->set_process_id(identity.process_id);
      wave->set_address_space(identity.address_space);
      wave->set_vm_access(execution_access);
      wave->set_doorbell_id((restored.doorbell_offset / sizeof(uint64_t)) & 0x3ffu);
      wave->set_lds_base(lds_bases[i]);
      wave->set_lds_size(group.lds_bytes);
      wave->set_scratch_lane_size(saved_wave.scratch_lane_size);
      wave->set_wave_sched_mode_raw(saved_wave.sched_mode);
      wave->set_sleep_cycles(saved_wave.sleep_cycles);
      if (saved_wave.vgpr_msb_hazard)
        wave->arm_setreg_vgpr_msb_hazard();
      wave->set_runtime_suspended(saved_wave.runtime_suspended);
      wave->restore_barrier_wait_state(saved_wave.barrier);
      wave->restore_debug_stop_state(saved_wave.stop);
      wave->add_trap_queue_exception_status(saved_wave.queue_exceptions);
      wave->add_trap_runtime_exception_status(saved_wave.runtime_exceptions);
      wave->set_trap_saved_status(saved_wave.trap_saved_status);
      wave->set_trap_saved_exec(saved_wave.trap_saved_exec);
      wave->set_trap_interrupt_sent(saved_wave.interrupt_sent);
      wave->set_self_halted(saved_wave.self_halted);
      wave->set_state(saved_wave.state);
    }
  }
  if (restored.scratch_reclaim.active())
    restored.scratch_reclaim.dispatch_id = restored.entries[reclaim_entry].dispatch_id;
  restored.read_pointer_journal.reset(consumer_initialized ? std::optional<uint64_t>(consumer)
                                                           : std::nullopt);
  if (restored.scratch_request.active())
    restored.scratch_request.access = std::make_shared<GpuVmAccess>(*access);
  if (restored.scratch_reclaim.active())
    restored.scratch_reclaim.access = std::make_shared<GpuVmAccess>(*access);
  restored.registration_id = queue->registration_id;
  restored.address_space_lease = std::move(queue->address_space_lease);
  restored.debug_suspended = true;
  restored.debug_work_deferred = true; // A new binding needs a service pass on resume.
  *queue = std::move(restored);
  return VmAccessOutcome::Complete;
}

void CommandProcessor::release_queue_context(const QueueIdentity &identity) {
  std::unique_lock structure_lock(queue_structure_mutex_);
  std::lock_guard lock(hw_queue_mutex_);
  if (ComputeQueueRecord *queue = find_compute_queue(identity.queue_id, identity.process_id);
      queue && queue->address_space == identity.address_space)
    queue->context_save_suspension.reset();
  for (auto *cu : cus_)
    cu->with_wave_state_locked([&] {
      struct Group {
        uint32_t dispatch;
        uint32_t wg;
        uint32_t lds_base;
        uint32_t lds_size;
      };
      std::vector<Group> groups;
      for (uint32_t slot = 0; slot < cu->num_wf_slots(); ++slot) {
        auto *wave = cu->wf(slot);
        if (wave && !wave->is_halted() && wave->queue_id() == identity.queue_id &&
            wave->process_id() == identity.process_id &&
            wave->address_space() == identity.address_space)
          groups.push_back(
              {wave->dispatch_id(), wave->wg_id(), wave->lds_base(), wave->lds_size()});
      }
      const auto key = [](const Group &group) { return wg_key(group.dispatch, group.wg); };
      std::ranges::sort(groups, {}, key);
      const auto duplicates = std::ranges::unique(groups, {}, key);
      groups.erase(duplicates.begin(), duplicates.end());
      for (const auto &group : groups) {
        // Return the allocation before abort_workgroup can reset an idle CU.
        cu->release_lds(group.lds_base, group.lds_size);
        cu->abort_workgroup(group.dispatch, group.wg);
      }
    });
}

} // namespace rocjitsu::amdgpu
