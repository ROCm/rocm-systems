// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h"
#include "rocjitsu/vm/amdgpu/command_processor.h"

#include <algorithm>
#include <limits>
#include <type_traits>

namespace rocjitsu::amdgpu {

// Runtime scheduling and resource requirements. Architectural registers remain in the
// debugger-editable CWSR records; this image is the authority for allocations.
struct CommandProcessor::QueueContext {
  struct Group {
    uint32_t dispatch = 0, wg = 0, lds_bytes = 0;
    uint32_t remaining_waves = 0;
    bool has_barriers = false;
    ComputeUnitCore::WorkgroupBarriers barriers{};
  };
  struct Wave {
    uint32_t group = 0, sgprs = 0, vgprs = 0, width = 0;
    uint32_t packet = 0, wave_in_group = 0, scratch_lane_size = 0;
    uint64_t debug_id = 0, code_bias = 0;
    std::array<uint32_t, 3> coordinates{};
    WfState state = WfState::RUNNING;
    uint32_t sleep_cycles = 0, sched_mode = 0;
    bool vgpr_msb_hazard = false, runtime_suspended = false;
    Wavefront::BarrierWaitState barrier;
    Wavefront::DebugStopState stop;
    uint64_t queue_exceptions = 0, runtime_exceptions = 0;
    uint32_t trap_saved_status = 0;
    uint64_t trap_saved_exec = 0;
    bool interrupt_sent = false, self_halted = false;
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
constexpr uint64_t kQueueContextMagic = 0x3158544350434a52ULL; // "RJCPCTX1"
constexpr uint64_t kQueueContextVersion = 2;

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

template <typename Archive, typename Entry> bool dispatch_fields(Archive &a, Entry &e) {
  return a(e.dispatch_id, e.queue_packet_id, e.aql_packet_id, e.aql_packet_address,
           e.owns_aql_ring_slot, e.kernel_entry_pc, e.code_load_bias, e.wfs_per_workgroup,
           e.sgprs_per_wf, e.vgprs_per_wf, e.kernarg_addr, e.kernarg_size, e.num_user_sgprs,
           e.kernel_code_properties, e.num_named_barriers, e.kernel_wave_size, e.kernarg_preload,
           e.initial_mode_raw, e.dispatch_ptr, e.queue_ptr, e.workgroup_id_offset, e.grid_size_x,
           e.grid_size_y, e.grid_size_z, e.workgroup_origin, e.grid_wgs_x, e.grid_wgs_y,
           e.grid_wgs_z, e.grid_yz_valid, e.cluster_count_x, e.cluster_count_y, e.cluster_count_z,
           e.cluster_size_x, e.cluster_size_y, e.cluster_size_z, e.wgp_mode, e.enable_wg_id_x,
           e.enable_wg_id_y, e.enable_wg_id_z, e.enable_wg_info, e.enable_vgpr_workitem_id,
           e.workgroup_size_x, e.workgroup_size_y, e.workgroup_size_z, e.scratch_backing_addr,
           e.private_segment_fixed_size, e.scratch_wave_limit_per_se, e.scratch_wave_stride_per_se,
           e.scratch_wave_bytes, e.group_segment_fixed_size, e.scratch_use_once, e.total_wgs,
           e.dispatched_wgs, e.completed_wgs, e.completion_signal, e.profiling_start_timestamp,
           e.kind, e.host_signal, e.wait_for_predecessors, e.blocks_following, e.aql_pm4_ib_address,
           e.aql_pm4_ib_dwords, e.completion_notified, e.execution_begun, e.acquire_invalidate,
           e.terminal_faulted, e.completion_publication.phase);
}

template <typename Archive, typename Queue> bool queue_fields(Archive &a, Queue &q) {
  return a(q.ring_base_va, q.ring_size, q.read_ptr_va, q.write_ptr_va, q.doorbell_offset,
           q.packet_format, q.scheduling_percentage, q.process_id, q.queue_desc_va,
           q.uses_kfd_queue_abi, q.fetch_cursor, q.next_dispatch_idx, q.implicit_barrier_next,
           q.last_doorbell, q.runtime_suspended, q.exception_suspended, q.faulted,
           q.activity_generation, q.commands.indirect_base, q.commands.predicate_pass) &&
         a.sparse(q.commands.sh_registers) && a.sparse(q.commands.context_registers) &&
         a.sparse(q.commands.uconfig_registers) &&
         a(q.scratch_request.phase, q.scratch_request.packet_index, q.scratch_request.status,
           q.scratch_request.signal_address, q.scratch_request.mailbox_pointer,
           q.scratch_request.event_id, q.scratch_reclaim.phase, q.scratch_reclaim.signal_address,
           q.scratch_reclaim.mailbox_pointer, q.scratch_reclaim.event_id);
}

template <typename Archive, typename Group> bool group_fields(Archive &a, Group &g) {
  if (!a(g.dispatch, g.wg, g.lds_bytes, g.remaining_waves, g.has_barriers))
    return false;
  if (!g.has_barriers)
    return true;
  if (!a(g.barriers.allocated_count))
    return false;
  for (auto &counter : g.barriers.named)
    if (!a(counter.member_count, counter.signal_count))
      return false;
  for (auto &counter : g.barriers.workgroup)
    if (!a(counter.member_count, counter.signal_count))
      return false;
  return true;
}

template <typename Archive, typename Wave> bool wave_fields(Archive &a, Wave &w) {
  return a(w.group, w.sgprs, w.vgprs, w.width, w.packet, w.wave_in_group, w.scratch_lane_size,
           w.debug_id, w.code_bias, w.coordinates, w.state, w.sleep_cycles, w.sched_mode,
           w.vgpr_msb_hazard, w.runtime_suspended, w.barrier.named_barrier_id, w.barrier.complete,
           w.barrier.waiting_bit, w.stop.trapsts, w.stop.mode_raw, w.stop.gfx12_trap_ctrl_raw,
           w.stop.trap_id, w.stop.debug_halted, w.stop.single_step, w.stop.fatal_exception_pending,
           w.stop.fatal_exception_cwsr_valid, w.queue_exceptions, w.runtime_exceptions,
           w.trap_saved_status, w.trap_saved_exec, w.interrupt_sent, w.self_halted);
}

bool supported_dispatch(const DispatchEntry &entry) {
  // Current MES placement is not fanned out. Cross-XCD ownership and external
  // PM4 callbacks require their own context model rather than saving pointers.
  return !entry.grid_completion && !entry.enabled_cus && !entry.pm4_abi && !entry.pm4_failure &&
         !entry.pm4_scratch_pool && !entry.fanout_peer && entry.shard.is_unsharded() &&
         !entry.wgp_mode && entry.cluster_size() == 1;
}

} // namespace

VmAccessOutcome CommandProcessor::save_queue_context(const QueueIdentity &identity,
                                                     std::vector<uint8_t> &context) {
  std::unique_lock structure_lock(queue_structure_mutex_);
  std::lock_guard lock(hw_queue_mutex_);
  auto *queue = find_compute_queue(identity.queue_id, identity.process_id);
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
      return VmAccessOutcome::Faulted;
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
  writer(kQueueContextMagic, kQueueContextVersion);
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
    const auto status = cu->with_wave_state_locked([&] {
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
          return VmAccessOutcome::Faulted;
        auto group = std::ranges::find_if(saved.groups, [&](const auto &g) {
          return g.dispatch == wave->dispatch_id() && g.wg == wave->wg_id();
        });
        if (group == saved.groups.end()) {
          QueueContext::Group g{
              .dispatch = wave->dispatch_id(), .wg = wave->wg_id(), .lds_bytes = wave->lds_size()};
          const auto key = wg_key(g.dispatch, g.wg);
          if (const auto remaining = cu->active_wgs_.find(key); remaining != cu->active_wgs_.end())
            g.remaining_waves = remaining->second;
          if (const auto barriers = cu->barrier_wgs_.find(key);
              barriers != cu->barrier_wgs_.end()) {
            g.has_barriers = true;
            g.barriers = barriers->second;
          }
          saved.groups.push_back(g);
          group = std::prev(saved.groups.end());
        }
        QueueContext::Wave w;
        w.group = std::distance(saved.groups.begin(), group);
        w.sgprs = wave->num_sgprs();
        w.vgprs = wave->num_vgprs();
        w.width = wave->wf_size();
        w.packet = wave->queue_packet_id();
        w.wave_in_group = wave->wave_in_group();
        w.scratch_lane_size = wave->scratch_lane_size();
        w.debug_id = wave->debug_wave_id();
        w.code_bias = wave->code_load_bias();
        w.coordinates = wave->wg_coord();
        w.state = wave->state();
        // Quiesced memory has no pipeline to wait on after reallocation.
        if (w.state == WfState::WAITCNT || w.state == WfState::VM_RETRY)
          w.state = WfState::RUNNING;
        w.sleep_cycles = wave->sleep_cycles();
        w.sched_mode = wave->wave_sched_mode_raw();
        w.vgpr_msb_hazard = wave->setreg_vgpr_msb_hazard();
        w.runtime_suspended = wave->runtime_suspended();
        w.barrier = wave->barrier_wait_state();
        w.stop = wave->debug_stop_state();
        w.queue_exceptions = wave->trap_queue_exception_status();
        w.runtime_exceptions = wave->trap_runtime_exception_status();
        w.trap_saved_status = wave->trap_saved_status();
        w.trap_saved_exec = wave->trap_saved_exec();
        w.interrupt_sent = wave->trap_interrupt_sent();
        w.self_halted = wave->self_halted();
        saved.waves.push_back(w);
      }
      return VmAccessOutcome::Complete;
    });
    if (status != VmAccessOutcome::Complete)
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
  auto *queue = find_compute_queue(identity.queue_id, identity.process_id);
  if (!queue || queue->address_space != identity.address_space || !queue->debug_suspended ||
      queue->has_pending_commands() || queue->idle_publication.active() ||
      queue->scratch_request.active() || queue->scratch_reclaim.active())
    return VmAccessOutcome::Malformed;
  ContextReader reader(context);
  uint64_t magic, version;
  ComputeQueueRecord restored(static_cast<const ComputeQueueConfig &>(*queue));
  bool consumer_initialized;
  uint64_t consumer, count, reclaim_entry;
  if (!reader(magic, version) || magic != kQueueContextMagic || version != kQueueContextVersion ||
      !queue_fields(reader, restored) ||
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
  for (const auto &group : saved.groups)
    dispatch_ids.try_emplace(group.dispatch, 0);

  // Keep all affected CUs stable until resource admission and rollback finish.
  std::vector<std::unique_lock<std::recursive_mutex>> wave_locks;
  std::vector<uint32_t> lds_cursors;
  for (auto *cu : cus_) {
    wave_locks.emplace_back(cu->wave_state_mutex_);
    lds_cursors.push_back(cu->next_lds_alloc_);
  }
  std::vector<std::pair<ComputeUnitCore *, Wavefront *>> reserved;
  std::vector<uint32_t> lds_bases;
  std::vector<ComputeUnitCore *> group_cus;
  std::vector<Wavefront *> waves(saved.waves.size());
  const auto rollback = [&] {
    for (const auto &[cu, wave] : reserved)
      cu->free_wavefront_resources(*wave);
    for (size_t i = 0; i < cus_.size(); ++i)
      cus_[i]->next_lds_alloc_ = lds_cursors[i];
  };
  for (size_t i = 0; i < saved.groups.size(); ++i) {
    const auto &group = saved.groups[i];
    const uint32_t count = std::ranges::count(saved.waves, i, &QueueContext::Wave::group);
    const auto placement = std::ranges::find_if(cus_, [&](const auto *cu) {
      return cu->can_accept_workgroup(count, group.lds_bytes) &&
             std::ranges::all_of(saved.waves, [&](const auto &w) {
               return w.group != i || (w.sgprs <= cu->config().sgprs_per_wf &&
                                       w.vgprs <= cu->config().vgprs_per_wf &&
                                       w.width <= isa_properties(cu->config().arch).wave_size_max);
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
      const auto &w = saved.waves[index];
      if (w.group != i)
        continue;
      auto *wave = cu->dispatch_wf(group.wg, 0, w.sgprs, w.vgprs, w.width);
      if (!wave) {
        rollback();
        return VmAccessOutcome::Unavailable;
      }
      wave->set_debug_suspended(true);
      waves[index] = wave;
      reserved.emplace_back(cu, wave);
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
      const auto &w = saved.waves[index];
      if (w.group != i)
        continue;
      auto *wave = waves[index];
      wave->set_dispatch_id(dispatch);
      wave->set_queue_packet_id(w.packet);
      wave->set_wave_in_group(w.wave_in_group);
      wave->set_debug_wave_id(w.debug_id);
      wave->set_code_load_bias(w.code_bias);
      wave->set_wg_coord(w.coordinates[0], w.coordinates[1], w.coordinates[2]);
      wave->set_queue_id(identity.queue_id);
      wave->set_process_id(identity.process_id);
      wave->set_address_space(identity.address_space);
      wave->set_vm_access(execution_access);
      wave->set_doorbell_id((restored.doorbell_offset / sizeof(uint64_t)) & 0x3ffu);
      wave->set_lds_base(lds_bases[i]);
      wave->set_lds_size(group.lds_bytes);
      wave->set_scratch_lane_size(w.scratch_lane_size);
      wave->set_wave_sched_mode_raw(w.sched_mode);
      wave->set_sleep_cycles(w.sleep_cycles);
      if (w.vgpr_msb_hazard)
        wave->arm_setreg_vgpr_msb_hazard();
      wave->set_runtime_suspended(w.runtime_suspended);
      wave->restore_barrier_wait_state(w.barrier);
      wave->restore_debug_stop_state(w.stop);
      wave->add_trap_queue_exception_status(w.queue_exceptions);
      wave->add_trap_runtime_exception_status(w.runtime_exceptions);
      wave->set_trap_saved_status(w.trap_saved_status);
      wave->set_trap_saved_exec(w.trap_saved_exec);
      wave->set_trap_interrupt_sent(w.interrupt_sent);
      wave->set_self_halted(w.self_halted);
      wave->set_state(w.state);
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
  for (auto *cu : cus_)
    cu->with_wave_state_locked([&] {
      std::vector<std::pair<uint32_t, uint32_t>> groups;
      for (uint32_t slot = 0; slot < cu->num_wf_slots(); ++slot) {
        auto *wave = cu->wf(slot);
        if (wave && !wave->is_halted() && wave->queue_id() == identity.queue_id &&
            wave->process_id() == identity.process_id &&
            wave->address_space() == identity.address_space)
          groups.emplace_back(wave->dispatch_id(), wave->wg_id());
      }
      std::ranges::sort(groups);
      const auto duplicates = std::ranges::unique(groups);
      groups.erase(duplicates.begin(), duplicates.end());
      for (const auto &[dispatch, wg] : groups)
        cu->abort_workgroup(dispatch, wg);
    });
}

} // namespace rocjitsu::amdgpu
