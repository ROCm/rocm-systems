// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/code/kernel_symbol.h"
#include "rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/hsa_clock.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/pm4/pm4_packet_processor.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/amd_ext_aql_packet.h"
#include "hsa/amd_hsa_queue.h"
RJ_DIAGNOSTIC_POP

#include "simdojo/sim/message.h"
#include "simdojo/sim/simulation.h"
#include "util/bit.h"
#include "util/log.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <elf.h>
#include <format>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <thread>

namespace rocjitsu {
namespace amdgpu {

CommandProcessor::CommandProcessor(std::string name, simdojo::ExecMode exec_mode)
    : simdojo::Component(std::move(name)),
      aql_packet_processor_({
          .load_signal = [this](const AqlPacketProcessRequest &request,
                                uint64_t address) { return read_gpu_u64(request.access, address); },
          .admit =
              [this](const AqlPacketProcessRequest &request, AqlPreparedPacket prepared) {
                return admit_aql_packet(request, std::move(prepared));
              },
      }),
      exec_mode_(exec_mode) {
  // Bind the doorbell handler at construction, not in startup(): register_queue()
  // may start the doorbell poll thread (which fires doorbell_event_ via
  // schedule_event_now) as soon as a host-accessible queue is registered, which can
  // happen before startup() runs. Binding here removes that ordering hazard — a
  // handlerless doorbell_event_ would be silently dropped by the engine.
  doorbell_event_.set_handler(
      [this](simdojo::Tick ts, simdojo::Message *) { handle_doorbell(ts); });
  retry_event_.set_handler([this](simdojo::Tick ts, simdojo::Message *) {
    retry_event_pending_.store(false, std::memory_order_release);
    handle_doorbell(ts);
  });
  stall_recheck_event_.set_handler([this](simdojo::Tick ts, simdojo::Message *message) {
    if (!message || message->payload() != stall_recheck_generation_)
      return;
    stall_recheck_pending_ = false;
    stall_recheck_tick_ = simdojo::TICK_MAX;
    handle_doorbell(ts);
  });
  dispatch_continuation_event_.set_handler([this](simdojo::Tick ts, simdojo::Message *message) {
    if (!message || message->payload() != dispatch_continuation_generation_)
      return;
    dispatch_continuation_pending_ = false;
    dispatch_continuation_tick_ = simdojo::TICK_MAX;
    handle_doorbell_sync(ts);
  });
}

CommandProcessor::~CommandProcessor() { stop_doorbell_monitor(); }

CommandProcessor::QueueRegistrationTransaction::QueueRegistrationTransaction(
    CommandProcessor &owner)
    : owner_(&owner) {
  std::lock_guard<std::recursive_mutex> lock(owner_->hw_queue_mutex_);
  ++owner_->active_queue_registrations_;
}

CommandProcessor::QueueRegistrationTransaction::~QueueRegistrationTransaction() {
  std::lock_guard<std::recursive_mutex> lock(owner_->hw_queue_mutex_);
  assert(owner_->active_queue_registrations_ != 0);
  --owner_->active_queue_registrations_;
}

void CommandProcessor::set_gpu_vm(GpuVm *gpu_vm, AddressSpaceHandle default_address_space) {
  if (gpu_vm != nullptr && default_address_space && !gpu_vm->lookup(default_address_space))
    throw std::invalid_argument("command processor default address space is not registered");
  if (gpu_vm_ != gpu_vm || default_address_space_ != default_address_space) {
    std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
    if (!compute_queues_.empty() || active_queue_registrations_ != 0)
      throw std::logic_error("cannot replace the command processor VM while queues exist");
    gpu_vm_ = gpu_vm;
    default_address_space_ = gpu_vm_ ? default_address_space : AddressSpaceHandle{};
  }
  for (ComputeUnitCore *cu : cus_)
    cu->set_gpu_vm(gpu_vm_);
}

uint64_t CommandProcessor::register_pm4_queue(Pm4QueueConfig config) {
  // Compatibility entry point: PM4 is initial state of the same compute queue.
  return register_queue({.address_space = config.address_space,
                         .ring_base_va = config.ring_base,
                         .ring_size = config.ring_size_bytes,
                         .read_ptr_va = config.consumer_pointer_address,
                         .last_doorbell = config.initial_consumer_cursor.value_or(~uint64_t(0)),
                         .packet_format = QueuePacketFormat::Pm4,
                         .initial_consumer_cursor = config.initial_consumer_cursor,
                         .packet_callbacks = std::move(config.packet_callbacks)});
}

bool CommandProcessor::unregister_pm4_queue_registration(uint64_t id) noexcept {
  return unregister_queue_registration(id);
}

QueuePrepareCloseStatus
CommandProcessor::prepare_unregister_pm4_queue_registration(uint64_t id) noexcept {
  return prepare_unregister_queue_registration(id);
}

QueueReconfigureStatus
CommandProcessor::update_pm4_queue_registration(uint64_t id,
                                                const QueueReconfigureRequest &request) {
  std::lock_guard lock(hw_queue_mutex_);
  const auto queue = std::ranges::find(compute_queues_, id, &ComputeQueueRecord::registration_id);
  if (queue == compute_queues_.end())
    return QueueReconfigureStatus::Stale;
  if (queue->faulted || queue->command_fault_pending || request.scheduling_percentage > 100)
    return QueueReconfigureStatus::Invalid;
  const bool disabled = !request.ring_base_address || !request.scheduling_percentage;
  const uint64_t base = disabled ? queue->ring_base_va : request.ring_base_address;
  const uint32_t bytes = disabled ? queue->ring_size : request.ring_size_bytes;
  if (!disabled && (!base || base % 4 || bytes < 4 || bytes % 4))
    return QueueReconfigureStatus::Invalid;
  if ((base != queue->ring_base_va || bytes != queue->ring_size) && queue->has_pending_commands())
    return QueueReconfigureStatus::Busy;
  // Ring reconfiguration and runtime suspension share the same wave gates as AQL.
  if (!update_queue_registration(id, base, bytes, disabled ? 0 : request.scheduling_percentage))
    return QueueReconfigureStatus::Invalid;
  return disabled ? QueueReconfigureStatus::Disabled : QueueReconfigureStatus::Applied;
}

QueueSubmissionStatus CommandProcessor::notify_pm4_queue_doorbell(uint64_t id, uint64_t producer) {
  std::lock_guard lock(hw_queue_mutex_);
  const auto queue = std::ranges::find(compute_queues_, id, &ComputeQueueRecord::registration_id);
  if (queue == compute_queues_.end() || queue->faulted || queue->command_fault_pending ||
      queue->runtime_suspended || queue->packet_format != QueuePacketFormat::Pm4 ||
      queue->submission_queue)
    return QueueSubmissionStatus::Faulted;
  const uint64_t reference = queue->last_doorbell == ~uint64_t(0)
                                 ? queue->read_pointer_journal.cursor()
                                 : queue->last_doorbell;
  const auto normalized = normalize_pm4_producer_cursor(producer, reference, queue->ring_size / 4);
  if (!normalized) {
    // CU cleanup belongs to the event thread, even for a rejected notification.
    queue->command_fault_pending = true;
    if (engine())
      engine()->schedule_event_now(doorbell_event());
    else
      fail_pm4_queue(*queue, queue->dispatches);
    return QueueSubmissionStatus::Faulted;
  }
  queue->last_doorbell = *normalized;
  if (engine()) {
    // Validate synchronously; packet effects remain on the CP event context.
    engine()->schedule_event_now(doorbell_event());
    return QueueSubmissionStatus::Accepted;
  }
  for (auto &record : compute_queues_)
    record.command_retry_pending = false;
  service_command_streams(0);
  return queue->faulted                  ? QueueSubmissionStatus::Faulted
         : queue->has_pending_commands() ? QueueSubmissionStatus::Retry
                                         : QueueSubmissionStatus::Accepted;
}

size_t CommandProcessor::registered_pm4_queue_count_for_test() const {
  std::lock_guard lock(hw_queue_mutex_);
  return std::ranges::count_if(
      compute_queues_, [](const auto &q) { return q.packet_format == QueuePacketFormat::Pm4; });
}

bool CommandProcessor::has_registered_queues() const {
  std::lock_guard lock(hw_queue_mutex_);
  return active_queue_registrations_ != 0 || !compute_queues_.empty();
}

void CommandProcessor::configure_for_arch(rj_code_arch_t arch) {
  // Matches LLVM's FeaturePackedTID: gfx90a and later CDNA targets, plus
  // GFX11 and later RDNA targets, receive work-item IDs packed in v0.
  packed_tid_ = arch == ROCJITSU_CODE_ARCH_CDNA2 || arch == ROCJITSU_CODE_ARCH_CDNA3 ||
                arch == ROCJITSU_CODE_ARCH_CDNA4 || arch == ROCJITSU_CODE_ARCH_RDNA3 ||
                arch == ROCJITSU_CODE_ARCH_RDNA3_5 || arch == ROCJITSU_CODE_ARCH_RDNA4 ||
                arch == ROCJITSU_CODE_ARCH_CDNA5;
}

namespace {

// The supported cluster size must fit the M0 multicast mask captured at issue time.
constexpr uint32_t kMaxClusterWorkgroups = kClusterMulticastMaskBits;
static_assert(kMaxClusterWorkgroups <= kClusterMulticastMaskBits);
static_assert(kMaxClusterWorkgroups <= 16,
              "TTMP6 cluster max and max-flat-ID fields are 4 bits wide");

// GFX12 launch-state TTMP indices used by compiler-generated workgroup and
// cluster identity sequences. These are indices into the wave's trap-temporary
// file (Wavefront::ttmp()), not SGPR numbers: the shader reaches them through
// the TTMP operand encodings (scalar selectors 108..123), which the ISA decoder
// routes to that file rather than to the SGPR allocation.
constexpr uint32_t kGfx12Ttmp6 = 6;
constexpr uint32_t kGfx12Ttmp7 = 7;
constexpr uint32_t kGfx12Ttmp8 = 8;
constexpr uint32_t kGfx12Ttmp9 = 9;

// LLVM's gfx1250 architected-SGPR ABI maps TTMP6 as seven 4-bit fields:
// cluster-local XYZ, cluster-max XYZ, and max-flat-ID from low to high bits.
// TTMP7 holds 16-bit cluster-grid Y/Z IDs. TTMP8 holds queue-packet ID
// [24:0], wave-in-workgroup [29:25], grid-Y/Z-valid [30], and debug-mark
// [31]. TTMP9 holds cluster-grid X.
constexpr uint32_t kGfx12Ttmp6ClusterLocalXShift = 0;
constexpr uint32_t kGfx12Ttmp6ClusterLocalYShift = 4;
constexpr uint32_t kGfx12Ttmp6ClusterLocalZShift = 8;
constexpr uint32_t kGfx12Ttmp6ClusterMaxXShift = 12;
constexpr uint32_t kGfx12Ttmp6ClusterMaxYShift = 16;
constexpr uint32_t kGfx12Ttmp6ClusterMaxZShift = 20;
constexpr uint32_t kGfx12Ttmp6ClusterMaxFlatIdShift = 24;
constexpr uint32_t kGfx12Ttmp7ClusterGridDimensionMask = 0xFFFFu;
constexpr uint32_t kGfx12Ttmp8QueuePacketIdMask = 0x1FFFFFFu;
constexpr uint32_t kGfx12Ttmp8WaveIdInGroupShift = 25;
constexpr uint32_t kGfx12Ttmp8GridYzValidShift = 30;

struct PlannedWorkgroup {
  uint32_t local_wg_id = 0;
  uint32_t global_wg_id = 0;
  ComputeUnitCore *cu = nullptr;
};

uint32_t nonzero_or_one(uint32_t v) { return v == 0 ? 1 : v; }

AqlAdmissionResult admission_from_vm_outcome(VmAccessOutcome outcome) {
  switch (outcome) {
  case VmAccessOutcome::Complete:
    return {.status = AqlAdmissionStatus::Complete};
  case VmAccessOutcome::Unavailable:
    return {.status = AqlAdmissionStatus::Blocked};
  case VmAccessOutcome::Faulted:
    return {.status = AqlAdmissionStatus::Faulted};
  case VmAccessOutcome::Malformed:
    return {.status = AqlAdmissionStatus::Malformed};
  case VmAccessOutcome::Revoked:
    return {.status = AqlAdmissionStatus::Faulted};
  }
  return {.status = AqlAdmissionStatus::Malformed};
}

std::optional<AqlPacketDiagnostic> validate_cluster_shape(const DispatchEntry &dp) {
  if (!dp.has_workgroup_clusters())
    return std::nullopt;
  auto cluster_size =
      static_cast<uint64_t>(dp.cluster_size_x) * dp.cluster_size_y * dp.cluster_size_z;
  // This also keeps every TTMP6 cluster dimension/max field within 4 bits.
  if (cluster_size == 0 || cluster_size > kMaxClusterWorkgroups)
    return AqlPacketDiagnostic::InvalidClusterShape;
  if (!dp.cluster_grid_is_complete())
    return AqlPacketDiagnostic::InvalidClusterShape;
  const uint64_t rank_period = dp.cluster_rank_period();
  if (dp.workgroup_id_offset % rank_period != 0)
    return AqlPacketDiagnostic::InvalidClusterShape;
  return std::nullopt;
}

uint32_t aligned_lds_bytes_per_workgroup(const DispatchEntry &entry) {
  // Match ComputeUnitCore::allocate_lds()/can_accept_workgroup() granularity for all dispatches.
  return util::align_up(entry.group_segment_fixed_size, 256u);
}

bool any_active_wavefronts(const std::vector<ComputeUnitCore *> &cus) {
  return std::ranges::any_of(cus, [](const auto *cu) { return cu->has_active_wfs(); });
}

bool plan_cluster_workgroups(const DispatchEntry &entry, uint32_t cluster_base_local_wg_id,
                             size_t next_cu, const std::vector<ComputeUnitCore *> &cus,
                             std::vector<PlannedWorkgroup> &plan, size_t &planned_next_cu) {
  plan.clear();
  uint32_t cluster_size = entry.cluster_size();
  const uint32_t lds_bytes_per_wg = aligned_lds_bytes_per_workgroup(entry);
  constexpr auto kU32Max = std::numeric_limits<uint32_t>::max();
  std::vector<uint32_t> planned_per_cu(cus.size(), 0);
  size_t last_cu_idx = next_cu;

  for (uint32_t rank = 0; rank < cluster_size; ++rank) {
    bool assigned = false;
    uint32_t local_wg_id = entry.cluster_peer_local_wg_id(cluster_base_local_wg_id, rank);
    for (size_t attempt = 0; attempt < cus.size(); ++attempt) {
      size_t cu_idx = (next_cu + rank + attempt) % cus.size();
      auto *cu = cus[cu_idx];
      if (!entry.allows_cu(cu))
        continue;

      uint32_t reserved_wgs = planned_per_cu[cu_idx] + 1;
      uint64_t reserved_wfs = static_cast<uint64_t>(entry.wfs_per_workgroup) * reserved_wgs;
      uint64_t reserved_lds = static_cast<uint64_t>(lds_bytes_per_wg) * reserved_wgs;
      if (reserved_wfs > kU32Max || reserved_lds > kU32Max)
        continue;
      if (!cu->can_accept_workgroup(static_cast<uint32_t>(reserved_wfs),
                                    static_cast<uint32_t>(reserved_lds),
                                    entry.scratch_wave_limit_per_se))
        continue;

      plan.push_back({local_wg_id, local_wg_id + entry.workgroup_id_offset, cu});
      ++planned_per_cu[cu_idx];
      last_cu_idx = cu_idx;
      assigned = true;
      break;
    }
    if (!assigned) {
      plan.clear();
      return false;
    }
  }

  planned_next_cu = (last_cu_idx + 1) % cus.size();
  return true;
}

bool sgpr_count_is_descriptor_encoded(rj_code_arch_t arch, uint32_t sgpr_gran) {
  if (sgpr_gran != 0)
    return true;
  return isa_properties(arch).descriptor_sgpr_count_encoded;
}

bool compute_pgm_rsrc1_mode_preserves_dx10_ieee(rj_code_arch_t arch) {
  /*
   * New ISA families should classify descriptor-to-MODE field initialization
   * for the architecture's MODE layout.
   */
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
  case ROCJITSU_CODE_ARCH_CDNA2:
  case ROCJITSU_CODE_ARCH_CDNA3:
  case ROCJITSU_CODE_ARCH_CDNA4:
  case ROCJITSU_CODE_ARCH_RDNA1:
  case ROCJITSU_CODE_ARCH_RDNA2:
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return true;
  case ROCJITSU_CODE_ARCH_RDNA4:
  case ROCJITSU_CODE_ARCH_CDNA5:
  case ROCJITSU_CODE_ARCH_RV32I:
  case ROCJITSU_CODE_ARCH_RV64I:
  case ROCJITSU_CODE_ARCH_NUM_ARCHS:
    return false;
  }
  // Handle out-of-range values without a default, so -Wswitch catches new architectures.
  return false;
}

bool compute_pgm_rsrc1_mode_has_debug_field(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
  case ROCJITSU_CODE_ARCH_CDNA2:
  case ROCJITSU_CODE_ARCH_CDNA3:
  case ROCJITSU_CODE_ARCH_CDNA4:
  case ROCJITSU_CODE_ARCH_RDNA1:
  case ROCJITSU_CODE_ARCH_RDNA2:
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return true;
  case ROCJITSU_CODE_ARCH_RDNA4:
  case ROCJITSU_CODE_ARCH_CDNA5:
  case ROCJITSU_CODE_ARCH_RV32I:
  case ROCJITSU_CODE_ARCH_RV64I:
  case ROCJITSU_CODE_ARCH_NUM_ARCHS:
    return false;
  }
  // Handle out-of-range values without a default, so -Wswitch catches new architectures.
  return false;
}

uint32_t initial_mode_from_compute_pgm_rsrc1(uint32_t rsrc1, rj_code_arch_t arch) {
  using namespace rocr::llvm::amdhsa;

  uint32_t mode = 0;
  mode |= AMDHSA_BITS_GET(rsrc1, COMPUTE_PGM_RSRC1_FLOAT_ROUND_MODE_32) << 0;
  mode |= AMDHSA_BITS_GET(rsrc1, COMPUTE_PGM_RSRC1_FLOAT_ROUND_MODE_16_64) << 2;
  mode |= AMDHSA_BITS_GET(rsrc1, COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_32) << 4;
  mode |= AMDHSA_BITS_GET(rsrc1, COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_16_64) << 6;
  if (compute_pgm_rsrc1_mode_preserves_dx10_ieee(arch)) {
    mode |= AMDHSA_BITS_GET(rsrc1, COMPUTE_PGM_RSRC1_ENABLE_DX10_CLAMP) << 8;
    mode |= AMDHSA_BITS_GET(rsrc1, COMPUTE_PGM_RSRC1_ENABLE_IEEE_MODE) << 9;
  }
  if (compute_pgm_rsrc1_mode_has_debug_field(arch))
    mode |= AMDHSA_BITS_GET(rsrc1, COMPUTE_PGM_RSRC1_DEBUG_MODE) << 11;
  if (AMDHSA_BITS_GET(rsrc1, COMPUTE_PGM_RSRC1_FP16_OVFL))
    mode |= Wavefront::FP16_OVFL_BIT;
  return mode;
}

} // namespace

void CommandProcessor::set_shared_dispatch_pool(CpuDispatchPool *pool) {
  shared_dispatch_pool_ = pool;
  if (shared_dispatch_pool_)
    local_dispatch_pool_.reset();
}

void CommandProcessor::set_dispatch_threads(uint32_t threads) {
  threads = std::max(threads, 1u);
  if (exec_mode_ != simdojo::ExecMode::FUNCTIONAL)
    threads = 1;
  if (dispatch_threads_ == threads)
    return;

  const bool was_pool_driven = dispatch_threads_ > 1;
  const bool pool_driven = threads > 1;
  dispatch_threads_ = threads;
  local_dispatch_pool_.reset();

  if (was_pool_driven == pool_driven) {
    for (auto *cu : cus_)
      cu->set_pool_driven(pool_driven);
    return;
  }

  simdojo::Tick now = 0;
  if (engine())
    now = engine()->context(partition_id()).current_tick();

  if (pool_driven) {
    pooled_due_ticks_.clear();
    for (auto *cu : cus_) {
      const simdojo::Tick serial_due = cu->suspend_scheduled_work();
      cu->set_pool_driven(true);
      if (cu->has_runnable_wfs()) {
        simdojo::Tick tick = serial_due == simdojo::TICK_MAX ? now + 1 : serial_due;
        if (tick < now)
          tick = now + 1;
        pooled_due_ticks_[cu] = tick;
      }
    }
    const simdojo::Tick next = next_pooled_due_tick();
    if (next != simdojo::TICK_MAX)
      arm_dispatch_continuation(next);
    return;
  }

  cancel_dispatch_continuation();
  for (auto *cu : cus_) {
    cu->set_pool_driven(false);
    if (!cu->has_active_wfs())
      continue;
    auto due = pooled_due_ticks_.find(cu);
    simdojo::Tick tick = due == pooled_due_ticks_.end() ? now + 1 : due->second;
    if (tick < now)
      tick = now + 1;
    cu->schedule_work_at(tick);
  }
  pooled_due_ticks_.clear();
}

void CommandProcessor::arm_dispatch_continuation(simdojo::Tick tick) {
  if (!engine())
    return;
  if (dispatch_continuation_pending_ && dispatch_continuation_tick_ <= tick)
    return;

  dispatch_continuation_pending_ = true;
  dispatch_continuation_tick_ = tick;
  const uintptr_t generation = ++dispatch_continuation_generation_;
  schedule_event(&dispatch_continuation_event_, tick,
                 std::make_unique<simdojo::Message>(simdojo::MessageHeader{}, generation));
}

void CommandProcessor::cancel_dispatch_continuation() {
  dispatch_continuation_pending_ = false;
  dispatch_continuation_tick_ = simdojo::TICK_MAX;
  ++dispatch_continuation_generation_;
}

VmAccessOutcome CommandProcessor::init_wavefront_regs(ComputeUnitCore *cu, Wavefront *wf,
                                                      const DispatchEntry &pkt,
                                                      uint32_t global_wg_id,
                                                      uint32_t wf_index_in_wg) {
  using namespace rocr::llvm::amdhsa;
  uint32_t sbase = wf->sgpr_alloc().base;
  uint32_t kcp = pkt.kernel_code_properties;

  // User SGPRs per AMDHSA ABI: placed sequentially based on enable bits.
  // Order: private_segment_buffer(4), dispatch_ptr(2), queue_ptr(2),
  //        kernarg_segment_ptr(2), dispatch_id(2), flat_scratch_init(2),
  //        private_segment_size(1).
  // When kernel_code_properties is 0 (internal test dispatches), fall back to
  // the legacy layout: kernarg at s[0:1].
  int flat_scratch_init_sgpr = -1;
  if (pkt.pm4_abi) {
    for (uint32_t i = 0; i < pkt.num_user_sgprs; ++i)
      cu->write_sgpr(sbase + i, pkt.user_sgprs[i]);
  } else if (kcp != 0) {
    const DispatchLaunchMetadata *launch_metadata = nullptr;
    if (pkt.queue_ptr != 0) {
      const std::unordered_map<uint32_t, DispatchLaunchMetadata>::const_iterator metadata_entry =
          dispatch_launch_metadata_.find(pkt.dispatch_id);
      if (metadata_entry == dispatch_launch_metadata_.end())
        return VmAccessOutcome::Malformed;
      launch_metadata = &metadata_entry->second;
    }
    uint32_t idx = 0;
    if (AMDHSA_BITS_GET(kcp, KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER)) {
      if (pkt.queue_ptr != 0) {
        for (uint32_t word = 0; word < 4; ++word)
          cu->write_sgpr(sbase + idx + word, launch_metadata->scratch_resource_descriptor[word]);
      }
      idx += 4;
    }
    if (AMDHSA_BITS_GET(kcp, KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR)) {
      cu->write_sgpr(sbase + idx, static_cast<uint32_t>(pkt.dispatch_ptr));
      cu->write_sgpr(sbase + idx + 1, static_cast<uint32_t>(pkt.dispatch_ptr >> 32));
      idx += 2;
    }
    if (AMDHSA_BITS_GET(kcp, KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR)) {
      cu->write_sgpr(sbase + idx, static_cast<uint32_t>(pkt.queue_ptr));
      cu->write_sgpr(sbase + idx + 1, static_cast<uint32_t>(pkt.queue_ptr >> 32));
      idx += 2;
    }
    if (AMDHSA_BITS_GET(kcp, KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR)) {
      cu->write_sgpr(sbase + idx, static_cast<uint32_t>(pkt.kernarg_addr));
      cu->write_sgpr(sbase + idx + 1, static_cast<uint32_t>(pkt.kernarg_addr >> 32));
      util::Logger::vm("CP: init_wf kernarg s[", idx, ":", idx + 1, "] = 0x", std::hex,
                       pkt.kernarg_addr, std::dec, " sbase=", sbase);
      idx += 2;
    }
    if (AMDHSA_BITS_GET(kcp, KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID)) {
      const uint64_t dispatch_id = pkt.queue_ptr == 0 ? 0 : launch_metadata->write_dispatch_id;
      cu->write_sgpr(sbase + idx, static_cast<uint32_t>(dispatch_id));
      cu->write_sgpr(sbase + idx + 1, static_cast<uint32_t>(dispatch_id >> 32));
      idx += 2;
    }
    if (AMDHSA_BITS_GET(kcp, KERNEL_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT)) {
      flat_scratch_init_sgpr = static_cast<int>(idx);
      idx += 2;
    }
    if (AMDHSA_BITS_GET(kcp, KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE)) {
      cu->write_sgpr(sbase + idx, pkt.private_segment_fixed_size);
      idx += 1;
    }

    uint32_t preload_length = AMDHSA_BITS_GET(pkt.kernarg_preload, KERNARG_PRELOAD_SPEC_LENGTH);
    uint32_t preload_offset = AMDHSA_BITS_GET(pkt.kernarg_preload, KERNARG_PRELOAD_SPEC_OFFSET);
    if (preload_length != 0) {
      if (pkt.kernarg_addr == 0)
        return VmAccessOutcome::Malformed;
      if (idx + preload_length > pkt.num_user_sgprs)
        return VmAccessOutcome::Malformed;
      uint32_t preload_end = preload_offset + preload_length;
      // Some assembly code objects leave descriptor kernarg_size at zero while
      // carrying the real size in metadata; treat zero as unknown.
      if (pkt.kernarg_size != 0 && preload_end > pkt.kernarg_size / sizeof(uint32_t))
        return VmAccessOutcome::Malformed;

      uint64_t preload_addr = pkt.kernarg_addr + static_cast<uint64_t>(preload_offset) * 4;
      for (uint32_t preload_index = 0; preload_index < preload_length; ++preload_index) {
        const uint64_t address = preload_addr + preload_index * 4;
        const AtomicLoadResult loaded =
            pkt.metadata_kernargs
                ? AtomicLoadResult{.outcome = VmAccessOutcome::Complete,
                                   .value = (*pkt.metadata_kernargs)[preload_index]}
            : pkt.execution_access ? read_gpu_u32(*pkt.execution_access, address)
                                   : read_gpu_u32(pkt.address_space, address);
        if (loaded.outcome != VmAccessOutcome::Complete)
          return loaded.outcome;
        cu->write_sgpr(sbase + idx + preload_index, static_cast<uint32_t>(loaded.value));
      }
      util::Logger::vm("CP: init_wf kernarg preload s[", idx, ":", idx + preload_length - 1,
                       "] length=", preload_length, " offset=", preload_offset, " sbase=", sbase);
      idx += preload_length;
    }
  } else {
    // Legacy: kernarg at s[0:1].
    if (pkt.kernarg_addr != 0) {
      cu->write_sgpr(sbase + 0, static_cast<uint32_t>(pkt.kernarg_addr));
      cu->write_sgpr(sbase + 1, static_cast<uint32_t>(pkt.kernarg_addr >> 32));
    }
  }

  uint32_t gx = pkt.grid_wgs_x > 0 ? pkt.grid_wgs_x : 1;
  uint32_t gy = pkt.grid_wgs_y > 0 ? pkt.grid_wgs_y : 1;
  uint32_t grid_wg_id_x = global_wg_id % gx + pkt.workgroup_origin[0];
  uint32_t wg_id_y = (global_wg_id / gx) % gy + pkt.workgroup_origin[1];
  uint32_t wg_id_z = global_wg_id / (gx * gy) + pkt.workgroup_origin[2];
  uint32_t wg_id_x =
      (pkt.pm4_abi || pkt.enable_wg_id_y || pkt.enable_wg_id_z) ? grid_wg_id_x : global_wg_id;

  // System SGPRs: workgroup_id_{x,y,z} placed sequentially after user SGPRs.
  // Only the IDs whose enable bits are set in compute_pgm_rsrc2 are written.
  // When kernel_code_properties is 0 (internal test dispatches), always write
  // workgroup_id_x as a fallback since internal kernels expect it.
  uint32_t sys_idx = pkt.num_user_sgprs;
  const auto properties = isa_properties(cu->arch());
  const bool pm4_ttmp_ids = pkt.pm4_abi && properties.uses_ttmp_workgroup_ids;
  {
    bool kcp_zero = (pkt.kernel_code_properties == 0) && !pkt.pm4_abi;
    if (!pm4_ttmp_ids && (pkt.enable_wg_id_x || kcp_zero))
      cu->write_sgpr(sbase + sys_idx++, wg_id_x);
    if (!pm4_ttmp_ids && pkt.enable_wg_id_y)
      cu->write_sgpr(sbase + sys_idx++, wg_id_y);
    if (!pm4_ttmp_ids && pkt.enable_wg_id_z)
      cu->write_sgpr(sbase + sys_idx++, wg_id_z);
    if (pkt.enable_wg_info) {
      // RDNA CS SGPR ABI: wave count [5:0] and ordered append term [17:6].
      // With ordered append disabled, the term is the wave index. RDNA1–3.5
      // also have first-wave [31]; RDNA2–3.5 add wave ID [24:20]. RDNA4
      // supplies the wave ID in TTMP8 and leaves those SGPR bits clear.
      uint32_t info = pkt.wfs_per_workgroup | (wf_index_in_wg << 6);
      if (!pm4_ttmp_ids) {
        info |= wf_index_in_wg == 0 ? 1u << 31 : 0;
        if (cu->arch() != ROCJITSU_CODE_ARCH_RDNA1)
          info |= wf_index_in_wg << 20;
      }
      cu->write_sgpr(sbase + sys_idx++, info);
    }
  }
  if (properties.uses_ttmp_workgroup_ids) {
    // The ordinary TTMP ABI uses grid coordinates. Targets advertising the
    // clustered extension reinterpret these fields below.
    uint32_t ttmp6 = 0;
    uint32_t ttmp7 = ((wg_id_z & kGfx12Ttmp7ClusterGridDimensionMask) << 16) |
                     (wg_id_y & kGfx12Ttmp7ClusterGridDimensionMask);
    uint32_t ttmp8 = pkt.queue_packet_id & kGfx12Ttmp8QueuePacketIdMask;
    ttmp8 |= wf_index_in_wg << kGfx12Ttmp8WaveIdInGroupShift;
    if (pkt.grid_yz_valid)
      ttmp8 |= 1u << kGfx12Ttmp8GridYzValidShift;
    uint32_t ttmp9 = grid_wg_id_x;
    if (properties.uses_cluster_ttmp_workgroup_ids) {
      const uint32_t cluster_size_x = nonzero_or_one(pkt.cluster_size_x);
      const uint32_t cluster_size_y = nonzero_or_one(pkt.cluster_size_y);
      const uint32_t cluster_size_z = nonzero_or_one(pkt.cluster_size_z);
      const WorkgroupCoord cluster_local = pkt.cluster_local_wg_coord_for_flat_wg_id(global_wg_id);
      const uint32_t cluster_max_x = cluster_size_x - 1;
      const uint32_t cluster_max_y = cluster_size_y - 1;
      const uint32_t cluster_max_z = cluster_size_z - 1;
      const uint32_t cluster_max_flat_id = cluster_size_x * cluster_size_y * cluster_size_z - 1;

      ttmp6 = (cluster_local.x << kGfx12Ttmp6ClusterLocalXShift) |
              (cluster_local.y << kGfx12Ttmp6ClusterLocalYShift) |
              (cluster_local.z << kGfx12Ttmp6ClusterLocalZShift) |
              (cluster_max_x << kGfx12Ttmp6ClusterMaxXShift) |
              (cluster_max_y << kGfx12Ttmp6ClusterMaxYShift) |
              (cluster_max_z << kGfx12Ttmp6ClusterMaxZShift) |
              (cluster_max_flat_id << kGfx12Ttmp6ClusterMaxFlatIdShift);
      const uint32_t cluster_grid_y =
          (wg_id_y / cluster_size_y) & kGfx12Ttmp7ClusterGridDimensionMask;
      const uint32_t cluster_grid_z =
          (wg_id_z / cluster_size_z) & kGfx12Ttmp7ClusterGridDimensionMask;
      ttmp7 = (cluster_grid_z << 16) | cluster_grid_y;
      ttmp9 = grid_wg_id_x / cluster_size_x;
    }
    wf->set_ttmp(kGfx12Ttmp6, ttmp6);
    wf->set_ttmp(kGfx12Ttmp7, ttmp7);
    wf->set_ttmp(kGfx12Ttmp8, ttmp8);
    wf->set_ttmp(kGfx12Ttmp9, ttmp9);
  }

  // Workitem IDs per AMDHSA ABI. The SPI decomposes the flat thread index
  // into (x, y, z) using the AQL packet's workgroup dimensions.
  // enable_vgpr_workitem_id (TIDIG_COMP_CNT from compute_pgm_rsrc2):
  //   0 = v0 only (workitem_id_x)
  //   1 = v0 + v1 (workitem_id_x, workitem_id_y)
  //   2 = v0 + v1 + v2 (workitem_id_x, workitem_id_y, workitem_id_z)
  // On packed-TID targets (CDNA3/4 and GFX11+): v0[9:0]=X, v0[19:10]=Y,
  // v0[29:20]=Z. TIDIG_COMP_CNT controls which components the SPI supplies;
  // unused packed components are zero.
  uint32_t vbase = wf->vgpr_alloc().base;
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    const WorkitemCoord id = workitem_local_coord(pkt, wf_index_in_wg, lane, wf->wf_size());
    if (packed_tid_) {
      cu->write_vgpr(vbase, lane, pack_workitem_id(id, pkt.enable_vgpr_workitem_id));
    } else {
      cu->write_vgpr(vbase, lane, id.x);
      if (pkt.enable_vgpr_workitem_id >= 1)
        cu->write_vgpr(vbase + 1, lane, id.y);
      if (pkt.enable_vgpr_workitem_id >= 2)
        cu->write_vgpr(vbase + 2, lane, id.z);
    }
  }

  // Scratch (private segment) setup.
  // Each wavefront gets a unique slice of scratch memory. The per-lane
  // private size is private_segment_fixed_size; the per-wave region is
  // that multiplied by wf_size. The global wave index is derived from
  // (global_wg_id, wf_index_in_wg) to ensure non-overlapping scratch
  // across all CUs and workgroups in the dispatch.
  if (pkt.private_segment_fixed_size > 0) {
    uint64_t scratch_pool = pkt.scratch_backing_addr;
    if (scratch_pool == 0)
      scratch_pool = 0x1'0000'0000ULL;
    // Round the per-wave region to the target's COMPUTE_TMPRING_SIZE.WAVESIZE
    // granule, so that each wave's base equals
    // scratch_pool + scoreboard_id * wavesize,
    // which is exactly what rocm-dbgapi computes to locate a wave's private
    // memory (rocdbgapi architecture.cpp scratch_memory_region).
    uint64_t raw_per_wave = static_cast<uint64_t>(pkt.private_segment_fixed_size) * wf->wf_size();
    const uint64_t granule = properties.compute_tmpring_wavesize_granule;
    const uint64_t per_wave_size = pkt.scratch_wave_size_bytes != 0
                                       ? pkt.scratch_wave_size_bytes
                                       : ((raw_per_wave + granule - 1) / granule) * granule;
    uint32_t wg_total_size = static_cast<uint32_t>(pkt.workgroup_size_x) *
                             std::max<uint16_t>(1, pkt.workgroup_size_y) *
                             std::max<uint16_t>(1, pkt.workgroup_size_z);
    uint32_t waves_per_wg = (wg_total_size + wf->wf_size() - 1) / wf->wf_size();
    uint64_t global_wave_idx = static_cast<uint64_t>(global_wg_id) * waves_per_wg + wf_index_in_wg;
    uint64_t scratch_slot = global_wave_idx;
    if (pkt.pm4_abi) {
      auto lease = pkt.pm4_scratch_pool->acquire();
      scratch_slot = *lease;
      wf->set_scratch_lease(std::move(lease));
    } else if (cu->arch() == ROCJITSU_CODE_ARCH_CDNA5) {
      const uint32_t shader_engine_count =
          std::max(scratch_wave_divisor_, scratch_shader_engine_count_);
      const uint32_t shader_engine_id = wf->shader_engine_id();
      const uint32_t scoreboard_id = wf->scratch_scoreboard_id();
      assert(shader_engine_id < shader_engine_count);
      assert(scoreboard_id < scratch_waves_per_se_);
      if (scoreboard_id >= pkt.scratch_wave_limit_per_se)
        return VmAccessOutcome::Malformed;
      const uint32_t scratch_wave_stride_per_se = pkt.scratch_wave_stride_per_se == 0
                                                      ? scratch_waves_per_se_
                                                      : pkt.scratch_wave_stride_per_se;
      if (scoreboard_id >= scratch_wave_stride_per_se)
        return VmAccessOutcome::Malformed;
      scratch_slot =
          (static_cast<uint64_t>(scratch_xcc_id_) * shader_engine_count + shader_engine_id) *
              scratch_wave_stride_per_se +
          scoreboard_id;
    } else {
      // Legacy CWSR records use the dispatch-wide logical scratch slot.
      wf->set_scratch_scoreboard_id(static_cast<uint32_t>(global_wave_idx));
    }
    if (scratch_slot > (std::numeric_limits<uint64_t>::max() - scratch_pool) / per_wave_size)
      return VmAccessOutcome::Malformed;
    uint64_t wave_scratch = scratch_pool + scratch_slot * per_wave_size;
    std::optional<GpuVmAccess> fallback_scratch_access;
    const GpuVmAccess *scratch_access = pkt.execution_access.get();
    if (scratch_access == nullptr) {
      fallback_scratch_access = snapshot_gpu_access(pkt.address_space);
      scratch_access = fallback_scratch_access ? &*fallback_scratch_access : nullptr;
    }
    auto scratch_range_outcome = [&](bool report_fault) {
      if (!scratch_access)
        return VmAccessOutcome::Unavailable;
      return report_fault
                 ? scratch_access->probe(wave_scratch, static_cast<size_t>(per_wave_size),
                                         VmAccessKind::Atomic)
                 : scratch_access->query_access(wave_scratch, static_cast<size_t>(per_wave_size),
                                                VmAccessKind::Atomic);
    };
    VmAccessOutcome scratch_outcome = scratch_range_outcome(false);

    if (!pkt.pm4_abi && scratch_allocator_ && pkt.scratch_wave_stride_per_se == 0) {
      // Size against the whole grid, not this XCD's share: every XCD of a
      // fanned-out dispatch shares the allocation. CDNA5 uses the complete
      // physical XCC/SE/scoreboard address space instead of logical grid slots.
      uint64_t scratch_slots = static_cast<uint64_t>(pkt.grid_total_wgs()) * waves_per_wg;
      if (cu->arch() == ROCJITSU_CODE_ARCH_CDNA5) {
        const uint32_t shader_engine_count =
            std::max(scratch_wave_divisor_, scratch_shader_engine_count_);
        scratch_slots =
            static_cast<uint64_t>(scratch_xcc_count_) * shader_engine_count * scratch_waves_per_se_;
      }
      if (scratch_slots == 0 || per_wave_size > std::numeric_limits<size_t>::max() / scratch_slots)
        return VmAccessOutcome::Malformed;
      const size_t total_scratch = static_cast<size_t>(per_wave_size * scratch_slots);
      // Provision the complete pool before this shard admits its first wave.
      // A smaller pool left by a preceding dispatch can cover that wave while
      // a later XCD needs more backing. Let the allocator check its allocation
      // records instead of probing every unused slot's host pages. Provisioning
      // is idempotent and preserves backing used by overlapping dispatches.
      const bool first_wave = pkt.dispatched_wgs == 0 && wf_index_in_wg == 0;
      if (first_wave || scratch_outcome != VmAccessOutcome::Complete) {
        if (!scratch_allocator_(pkt.process_id, scratch_pool, total_scratch))
          return VmAccessOutcome::Faulted;
        if (!pkt.execution_access) {
          fallback_scratch_access = snapshot_gpu_access(pkt.address_space);
          scratch_access = fallback_scratch_access ? &*fallback_scratch_access : nullptr;
        }
        scratch_outcome =
            scratch_access ? scratch_range_outcome(true) : VmAccessOutcome::Unavailable;
      }
    }

    // A successful allocator result is only a provisioning claim. Require the
    // complete per-wave slice to be readable and writable before publishing it
    // to the wave; checking the first byte would admit a truncated final page.
    if (scratch_outcome != VmAccessOutcome::Complete)
      return scratch_outcome == VmAccessOutcome::Unavailable ? VmAccessOutcome::Faulted
                                                             : scratch_outcome;

    wf->set_scratch_base(wave_scratch);
    wf->set_scratch_lane_size(pkt.private_segment_fixed_size);
    // CDNA compiler-generated functions use s32 as the private stack pointer
    // and s33 as its current frame value for explicit scratch SADDR operands.
    // The pointer is an offset within the per-wave scratch slice, not the SRD
    // base address supplied in the user SGPR block.
    if (cu->config().arch == ROCJITSU_CODE_ARCH_CDNA3 ||
        cu->config().arch == ROCJITSU_CODE_ARCH_CDNA4) {
      cu->write_sgpr(sbase + 32, 32);
      cu->write_sgpr(sbase + 33, 0);
    }
    util::Logger::cp([&](auto &os) {
      os << std::format("SCRATCH wf{} pool={:#x} wave_scratch={:#x} per_wave={} priv_size={} "
                        "backing_addr={:#x} mapped={}",
                        wf->wf_id(), scratch_pool, wave_scratch, per_wave_size,
                        pkt.private_segment_fixed_size, pkt.scratch_backing_addr,
                        scratch_outcome == VmAccessOutcome::Complete);
    });

    if (flat_scratch_init_sgpr >= 0) {
      cu->write_sgpr(sbase + flat_scratch_init_sgpr, static_cast<uint32_t>(wave_scratch));
      cu->write_sgpr(sbase + flat_scratch_init_sgpr + 1, static_cast<uint32_t>(wave_scratch >> 32));
    }
  }
  return VmAccessOutcome::Complete;
}

void CommandProcessor::startup() {
  // INVARIANT: this CP and every CU it dispatches to share one partition (engine
  // thread). on_cu_idle() dispatches inline and calls ComputeUnitCore::schedule_work()
  // on those CUs, which mutates their non-atomic executing_/tick_event_ and pushes to
  // the partition event queue without synchronization — safe only same-partition. The
  // generic balanced partitioner could in principle split a CP from a CU under
  // num_threads > 1; assert here (after partitioning, before the run loop) so any
  // such split fails loudly rather than silently racing.
  for ([[maybe_unused]] const auto *cu : cus_)
    assert(cu->partition_id() == partition_id() &&
           "CommandProcessor and its compute units must share one partition");
  // doorbell_event_'s handler is bound in the constructor (see there) so it is live
  // before register_queue() can start the poll thread; nothing to (re)bind here.
  if (gpu_vm_ == nullptr)
    throw std::logic_error("CommandProcessor requires a GPU VM before startup");
  completion_ = std::make_unique<CompletionTracker>(*gpu_vm_, cus_, l2_caches_);
  completion_->set_plugin_group(plugin_group_);
  completion_->set_dispatch_retirement_gate(
      [this](ComputeQueueRecord &queue, const DispatchEntry &entry) {
        return gate_dispatch_retirement(queue, entry);
      });
  completion_->set_dispatch_retired_callback([this](const DispatchEntry &entry) {
    erase_cluster_workgroups(entry.dispatch_id);
    dispatch_launch_metadata_.erase(entry.dispatch_id);
    // Resume the ring after the blocking packet's completion is durable.
    if (entry.blocks_following && engine()) {
      stall_recheck_backoff_ = 1;
      arm_stall_recheck(engine()->context(partition_id()).current_tick());
    }
  });
  completion_->set_grid_retired_callback([this](const DispatchEntry &) { wake_all_xcds(); });
  // Waves admitted before engine attachment could not notify the pool driver.
  // Seed those CUs once; admission and resume callbacks maintain the set later.
  if (dispatch_threads_ > 1)
    for (auto *cu : cus_)
      on_cu_pool_ready(cu);
}

void CommandProcessor::shutdown() {
  stop_doorbell_monitor();
  retry_event_pending_.store(false, std::memory_order_release);
  stall_recheck_pending_ = false;
  stall_recheck_tick_ = simdojo::TICK_MAX;
  ++stall_recheck_generation_;
  if (is_primary_ && engine()) {
    engine()->primary_release();
    is_primary_ = false;
  }
  completion_.reset();
}

void CommandProcessor::set_xcd_topology(uint32_t rank, std::vector<CommandProcessor *> peers) {
  assert(!peers.empty() && "XCD topology must contain at least this CP");
  assert(peers.size() <= MAX_NUM_XCC && "XCD topology exceeds the queue ABI capacity");
  assert(rank < peers.size() && "XCD rank must index its own SoC's CP list");
  assert(peers[rank] == this && "XCD rank must be this CP's own position");
  xcd_rank_ = rank;
  // Fan-out and scratch address the same physical XCD topology. Keeping a
  // frontend-owned scratch identity lets PCI/MES queues leave every CP at XCC
  // zero, so corresponding wave slots on different XCDs alias one another.
  scratch_xcc_id_ = rank;
  scratch_xcc_count_ = static_cast<uint32_t>(peers.size());
  xcd_peers_ = std::move(peers);
  // Carve this XCD its own dispatch-id space; see allocate_dispatch_id().
  dispatch_id_stride_ = static_cast<uint32_t>(xcd_peers_.size());
  dispatch_id_base_ = 1 + rank;
  next_dispatch_id_ = dispatch_id_base_;
}

void CommandProcessor::set_scratch_slots_per_cu(uint32_t slots) {
  configured_scratch_slots_per_cu_ = std::max(slots, 1u);
  scratch_waves_per_se_ = 1;
  for (ComputeUnitCore *cu : cus_) {
    cu->set_scratch_slots_per_cu(configured_scratch_slots_per_cu_);
    scratch_waves_per_se_ =
        std::max(scratch_waves_per_se_, cu->scratch_scoreboard_base() + cu->scratch_slots_per_cu());
  }
}

void CommandProcessor::set_scratch_xcc_layout_for_test(uint32_t xcc_id, uint32_t xcc_count) {
  assert(xcc_count != 0 && xcc_count <= MAX_NUM_XCC);
  assert(xcc_id < xcc_count);
  scratch_xcc_id_ = xcc_id;
  scratch_xcc_count_ = xcc_count;
}

ComputeQueueRecord *CommandProcessor::find_compute_queue(uint32_t queue_id, uint32_t process_id) {
  for (ComputeQueueRecord &queue : compute_queues_) {
    if (queue.queue_id == queue_id && queue.process_id == process_id)
      return &queue;
  }
  return nullptr;
}

void CommandProcessor::accept_fanout_shard(DispatchEntry shard) {
  accept_fanout_shard(std::move(shard), {});
}

void CommandProcessor::accept_fanout_shard(DispatchEntry shard,
                                           DispatchLaunchMetadata launch_metadata) {
  {
    util::Logger::cp([&](auto &os) {
      os << std::format("{}: FANOUT_SHARD d={} rank={}/{} wgs={}", name(), shard.dispatch_id,
                        shard.shard.rank(), shard.shard.stride(), shard.total_wgs);
    });
    // Deliberately NOT hw_queue_mutex_. The caller runs under its own CP's
    // hw_queue_mutex_ (fan-out happens inside handle_doorbell), so taking a peer's
    // hw_queue_mutex_ here would let two CPs fanning out concurrently acquire each
    // other's locks in opposite orders. Nothing that can lead back to another CP's
    // hw_queue_mutex_ is acquired while holding this one, and it is held only for
    // the push so a peer's engine thread never blocks on it for long.
    std::lock_guard<std::mutex> lock(fanout_inbox_mutex_);
    if (!shard.is_non_kernel())
      fanout_launch_metadata_inbox_.insert_or_assign(shard.dispatch_id, std::move(launch_metadata));
    fanout_inbox_.push_back(std::move(shard));
  }
  // Cross-thread and cross-partition safe: the engine buffers the event and drains
  // it into this CP's partition at its next safe point. Dispatching inline here
  // would reach into another partition's compute units.
  if (engine())
    engine()->schedule_event_now(doorbell_event());
}

void CommandProcessor::accept_dispatch_fault(DispatchFaultNotification fault) {
  {
    std::lock_guard<std::mutex> lock(dispatch_fault_inbox_mutex_);
    dispatch_fault_inbox_.push_back(fault);
  }
  if (engine())
    engine()->schedule_event_now(doorbell_event());
}

void CommandProcessor::drain_dispatch_fault_inbox() {
  std::vector<DispatchFaultNotification> faults;
  {
    std::lock_guard<std::mutex> lock(dispatch_fault_inbox_mutex_);
    faults.swap(dispatch_fault_inbox_);
  }
  for (const DispatchFaultNotification &fault : faults) {
    (void)fault_dispatch_local(fault.queue_id, fault.process_id, fault.dispatch_id, fault.outcome);
  }
}

void CommandProcessor::drain_fanout_inbox() {
  std::vector<DispatchEntry> inbox;
  std::unordered_map<uint32_t, DispatchLaunchMetadata> launch_metadata;
  {
    std::lock_guard<std::mutex> lock(fanout_inbox_mutex_);
    inbox.swap(fanout_inbox_);
    launch_metadata.swap(fanout_launch_metadata_inbox_);
  }
  // Real work arrived: the next wait this CP takes starts from a tight re-check.
  // Reset here rather than in accept_fanout_shard(), which runs on the OWNER's
  // thread -- writing this CP's backoff from there races the reads and writes its
  // own partition thread makes in arm_stall_recheck(). The shard is not visible to
  // this CP until it is drained anyway, and the drain runs before the re-arm in the
  // same handler pass, so resetting here is both correct and correctly ordered.
  if (inbox.empty())
    return;
  stall_recheck_backoff_ = 1;

  std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
  for (DispatchEntry &shard : inbox) {
    const std::vector<ComputeQueueRecord>::iterator queue =
        std::ranges::find_if(compute_queues_, [&](const ComputeQueueRecord &candidate) {
          return candidate.queue_id == shard.queue_id && candidate.process_id == shard.process_id;
        });
    if (queue == compute_queues_.end()) {
      // The replica was destroyed between the owner handing this shard over and
      // this drain. Drop it, exactly as unregister_queue drops a share it never
      // published: these workgroups have not run and this XCD's caches have not
      // been written back, so crediting the share would let the owner retire the
      // grid and fire the completion signal for work that never executed.
      //
      // Nothing is stranded by dropping it, for the reason unregister_queue
      // already relies on: a fan-out queue is destroyed on every XCD at once, so
      // the teardown that removed this replica removes the owner too. KFD
      // teardown is what reaches this window -- for_each_cp removes replicas in
      // XCD order while a later owner is still registered, and an
      // already-scheduled peer doorbell can drain concurrently.
      continue;
    }
    // A terminal VM fault closes execution admission for the queue. A peer may
    // already have emitted another shard before it observed the shared fault;
    // dropping it here prevents future work from appearing behind the fault.
    if (queue->faulted)
      continue;
    // Honour the packet's acquire fence on this XCD too. The owner invalidated
    // only its own CUs; this runs on our partition's thread, so ours are safe
    // to touch here and the peer ends up with the same view the owner has.
    if (shard.acquire_invalidate)
      flush_gpu_caches();
    if (!shard.is_non_kernel()) {
      const std::unordered_map<uint32_t, DispatchLaunchMetadata>::iterator metadata =
          launch_metadata.find(shard.dispatch_id);
      if (metadata == launch_metadata.end()) {
        queue->faulted = true;
        continue;
      }
      dispatch_launch_metadata_.insert_or_assign(shard.dispatch_id, std::move(metadata->second));
    }
    queue->push_entry(std::move(shard));
  }
}

void CommandProcessor::wake_all_xcds() {
  // Cross-partition safe: the engine buffers each event into the target's own
  // partition and never re-enters the component, so this is callable while
  // holding hw_queue_mutex_.
  for (auto *peer : xcd_peers_) {
    if (peer && peer->engine())
      peer->engine()->schedule_event_now(peer->doorbell_event());
  }
}

void CommandProcessor::fan_out_dispatch(DispatchEntry &dp,
                                        const DispatchLaunchMetadata &launch_metadata) {
  const auto num_xcds = static_cast<uint32_t>(xcd_peers_.size());
  if (num_xcds <= 1)
    return;

  // A masked queue may have no usable CU on its owning XCD. Only eligible
  // XCDs receive work; the owner still tracks whole-grid completion.
  std::vector<uint32_t> participants;
  if (dp.enabled_cus) {
    for (uint32_t rank = 0; rank < num_xcds; ++rank) {
      if (std::ranges::any_of(xcd_peers_[rank]->cus_,
                              [&](const auto *cu) { return dp.allows_cu(cu); }))
        participants.push_back(rank);
    }
  }
  const uint32_t participant_count =
      dp.enabled_cus ? static_cast<uint32_t>(participants.size()) : num_xcds;
  assert(participant_count != 0);

  const uint32_t grid_wgs = dp.total_wgs;
  auto grid = std::make_shared<GridCompletion>();
  grid->grid_wgs = grid_wgs;

  const auto apply_share = [&](DispatchEntry &entry, uint32_t physical_rank) {
    if (!entry.enabled_cus) {
      entry.apply_shard(XcdShard(physical_rank, num_xcds));
      return;
    }
    const auto participant = std::ranges::find(participants, physical_rank);
    if (participant == participants.end())
      entry.total_wgs = 0;
    else
      entry.apply_shard(
          XcdShard(static_cast<uint32_t>(participant - participants.begin()), participant_count));
  };

  // Excluded XCDs need empty shares too: a later CU-mask change can route work
  // to them, and barriers must still wait for the same predecessors everywhere.
  for (uint32_t physical_rank = 0; physical_rank < num_xcds; ++physical_rank) {
    if (physical_rank == xcd_rank_)
      continue;
    DispatchEntry shard = dp;
    shard.grid_completion = grid;
    shard.fanout_peer = true;
    // The peer must not fire the dispatch's completion signal; the owning XCD
    // does that once the grid counter shows every share retired.
    shard.completion_signal = 0;
    apply_share(shard, physical_rank);
    xcd_peers_[physical_rank]->accept_fanout_shard(std::move(shard), launch_metadata);
  }

  dp.grid_completion = std::move(grid);
  apply_share(dp, xcd_rank_);
}

void CommandProcessor::replicate_non_kernel_entry(const DispatchEntry &dp) {
  const auto num_xcds = static_cast<uint32_t>(xcd_peers_.size());
  if (num_xcds <= 1)
    return;

  assert(dp.is_non_kernel() && "only packets that run no shader are replicated whole");
  for (uint32_t rank = 0; rank < num_xcds; ++rank) {
    if (rank == xcd_rank_)
      continue;
    DispatchEntry copy = dp;
    copy.fanout_peer = true;
    copy.completion_signal = 0;
    xcd_peers_[rank]->accept_fanout_shard(std::move(copy));
  }
}

uint64_t CommandProcessor::register_queue(ComputeQueueConfig config) {
  return register_queue(std::move(config), false);
}

uint64_t CommandProcessor::register_queue(ComputeQueueConfig config, bool fanout_replica) {
  QueueRegistrationTransaction registration(*this);
  GpuVm *registration_vm = nullptr;
  {
    std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
    if (!config.address_space)
      config.address_space = default_address_space_;
    registration_vm = gpu_vm_;
  }
  if (registration_vm == nullptr || !config.address_space)
    return 0;
  if (!config.submission_queue) {
    const bool valid = config.packet_format == QueuePacketFormat::Aql
                           ? valid_aql_queue_layout(config.ring_base_va, config.ring_size,
                                                    config.read_ptr_va, config.write_ptr_va)
                           : config.packet_format == QueuePacketFormat::Pm4 &&
                                 config.ring_base_va != 0 && config.ring_base_va % 4 == 0 &&
                                 config.ring_size >= 4 && config.ring_size % 4 == 0 &&
                                 config.read_ptr_va != 0 && config.read_ptr_va % 4 == 0;
    if (!valid)
      return 0;
  }
  // ROCr enables metadata prefetch on GFX12.5 and newer, using 256-byte companions.
  if (config.metadata_ring_size &&
      (config.packet_format != QueuePacketFormat::Aql ||
       !aql_metadata::valid_ring_layout(config.ring_base_va, config.ring_size,
                                        config.metadata_ring_size) ||
       cus_.empty() || cus_[0]->config().arch != ROCJITSU_CODE_ARCH_CDNA5))
    return 0;
  // Native PM4 has no VM-doorbell poller. Reject it until polling is implemented.
  if (config.packet_format == QueuePacketFormat::Pm4 &&
      config.doorbell_mode == QueueDoorbellMode::VmPolled)
    return 0;
  // A host-polled queue may be registered before its doorbell page is mapped;
  // set_process_doorbell_base() publishes that mapping later. VM polling has no
  // equivalent deferred binding and therefore requires an address up front.
  if (config.doorbell_mode == QueueDoorbellMode::VmPolled && config.doorbell_va == 0) {
    return 0;
  }
  std::optional<GpuVmBindingLease> address_space_lease =
      registration_vm->retain_binding(config.address_space);
  if (!address_space_lease)
    return 0;
  ComputeQueueRecord queue(std::move(config));
  queue.address_space_lease = std::move(*address_space_lease);
  queue.fanout_replica = fanout_replica;
  if (!fanout_replica && queue.uses_kfd_queue_abi && queue.queue_desc_va != 0 && !cus_.empty() &&
      cus_[0]->config().arch == ROCJITSU_CODE_ARCH_CDNA5 &&
      publish_async_scratch_capability(queue) != VmAccessOutcome::Complete) {
    return 0;
  }
  class PeerRegistrationRollback {
  public:
    PeerRegistrationRollback(const std::vector<CommandProcessor *> &peers,
                             const std::vector<uint32_t> &registered_ranks, uint32_t queue_id,
                             uint32_t process_id)
        : peers_(peers), registered_ranks_(registered_ranks), queue_id_(queue_id),
          process_id_(process_id) {}
    PeerRegistrationRollback(const PeerRegistrationRollback &) = delete;
    PeerRegistrationRollback &operator=(const PeerRegistrationRollback &) = delete;
    ~PeerRegistrationRollback() {
      if (committed_)
        return;
      for (const uint32_t rank : registered_ranks_)
        peers_[rank]->unregister_queue(queue_id_, process_id_);
    }

    void commit() { committed_ = true; }

  private:
    const std::vector<CommandProcessor *> &peers_;
    const std::vector<uint32_t> &registered_ranks_;
    uint32_t queue_id_ = 0;
    uint32_t process_id_ = 0;
    bool committed_ = false;
  };

  class OwnerRegistrationRollback {
  public:
    explicit OwnerRegistrationRollback(std::vector<ComputeQueueRecord> &queues)
        : queues_(queues), initial_queue_count_(queues.size()) {}
    OwnerRegistrationRollback(const OwnerRegistrationRollback &) = delete;
    OwnerRegistrationRollback &operator=(const OwnerRegistrationRollback &) = delete;
    ~OwnerRegistrationRollback() {
      if (committed_)
        return;
      while (queues_.size() > initial_queue_count_)
        queues_.pop_back();
    }

    void commit() { committed_ = true; }

  private:
    std::vector<ComputeQueueRecord> &queues_;
    std::size_t initial_queue_count_ = 0;
    bool committed_ = false;
  };

  util::Logger::cp([&](auto &os) {
    os << std::format("{}: REGISTER_QUEUE id={} pid={} ring={:#x} size={} rptr={:#x} wptr={:#x} "
                      "doorbell_off={} db_base={}",
                      name(), queue.queue_id, queue.process_id, queue.ring_base_va, queue.ring_size,
                      queue.read_ptr_va, queue.write_ptr_va, queue.doorbell_offset,
                      reinterpret_cast<uintptr_t>(queue.doorbell_base));
  });
  // A replica exists only to receive dispatch shards from the XCD that owns the
  // queue. It must never read the ring or poll the doorbell, or the same packets
  // would be dispatched once per XCD.
  const bool start_poll =
      queue.doorbell_mode == QueueDoorbellMode::HostPolled && !queue.fanout_replica;
  // Replicate onto the peer XCDs without holding this CP's lock:
  // accept_fanout_shard() and the peers' register_queue() take their own locks.
  // A shard can arrive only after this function returns, so the owner may be
  // committed last without exposing an incomplete topology.
  {
    // Checked before replicating, so a rejection cannot leave replicas behind on
    // the peers. Shard routing keys on (queue_id, process_id), so a duplicate would
    // silently deliver every shard to whichever slot matched first -- a wrong-answer
    // bug rather than a crash, which is precisely what an assert compiled out of a
    // release build would let through. Enforced in every build for that reason.
    std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
    if (find_compute_queue(queue.queue_id, queue.process_id) != nullptr)
      return 0;
    queue.registration_id = next_queue_registration_id_++;
    if (next_queue_registration_id_ == 0)
      next_queue_registration_id_ = 1;
  }
  const uint64_t registration_id = queue.registration_id;
  const auto num_xcds = static_cast<uint32_t>(xcd_peers_.size());
  bool replicate = queue.xcd_fanout && num_xcds > 1;
  std::vector<uint32_t> registered_peer_ranks;
  PeerRegistrationRollback peer_rollback(xcd_peers_, registered_peer_ranks, queue.queue_id,
                                         queue.process_id);
  if (replicate) {
    registered_peer_ranks.reserve(num_xcds - 1);
    for (uint32_t rank = 0; rank < num_xcds; ++rank) {
      if (rank == xcd_rank_)
        continue;
      ComputeQueueConfig replica = queue;
      replica.xcd_fanout = false;
      if (xcd_peers_[rank]->register_queue(std::move(replica), true) == 0)
        return 0;
      registered_peer_ranks.push_back(rank);
    }
  }
  {
    std::unique_lock<std::shared_mutex> structure_lock(queue_structure_mutex_);
    std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
    if (find_compute_queue(queue.queue_id, queue.process_id) != nullptr)
      return 0;
    OwnerRegistrationRollback owner_rollback(compute_queues_);
    compute_queues_.push_back(std::move(queue));
    // KFD queues rely on the VM-level primary (rj_vm.cpp); only internal test
    // queues (no host-accessible queue anywhere on this CP) need the CP to own the
    // primary lifecycle. Gate on the same aggregate predicate as the teardown
    // release (!has_kfd_queues()) — checked AFTER the push_back so it reflects the
    // new queue — so a CP can never register a primary it will never release.
    if (!is_primary_ && engine() && !has_kfd_queues()) {
      engine()->register_as_primary();
      is_primary_ = true;
    } else if (is_primary_ && engine() && has_kfd_queues()) {
      // A KFD queue joined a CP that had registered a test-owned primary; the
      // VM-level primary now anchors this CP's lifecycle, so drop the CP-owned
      // primary to keep register/release symmetric (the teardown path only
      // releases when !has_kfd_queues()).
      engine()->primary_release();
      is_primary_ = false;
    }
    owner_rollback.commit();
  }
  peer_rollback.commit();
  // Start (or restart) the doorbell poll thread for KFD (host-accessible) queues
  // AFTER releasing hw_queue_mutex_. ensure_doorbell_monitor() serializes on its
  // own doorbell_thread_mutex_. Keeping that lock order consistent with the stop
  // path avoids joining a monitor while holding the queue mutex it needs to finish
  // a scan. Internal test queues inject doorbell events directly via
  // schedule_event_now() and need no monitor.
  if (start_poll)
    ensure_doorbell_monitor();
  return registration_id;
}

void CommandProcessor::notify_queue_doorbell(uint64_t registration_id, uint64_t value) {
  if (registration_id == 0)
    return;
  {
    // Transport callbacks never wait behind queue execution.  The owner thread
    // validates the stable registration id when it drains this leaf inbox.
    std::lock_guard lock(doorbell_inbox_mutex_);
    doorbell_inbox_.push_back({.registration_id = registration_id, .value = value});
  }
  if (engine())
    engine()->schedule_event_now(&doorbell_event_);
}

void CommandProcessor::drain_doorbell_inbox() {
  std::vector<DoorbellNotification> notifications;
  {
    std::lock_guard lock(doorbell_inbox_mutex_);
    notifications.swap(doorbell_inbox_);
  }
  if (notifications.empty())
    return;

  std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
  for (const DoorbellNotification &notification : notifications) {
    const std::vector<ComputeQueueRecord>::iterator queue = std::ranges::find(
        compute_queues_, notification.registration_id, &ComputeQueueRecord::registration_id);
    if (queue != compute_queues_.end())
      queue->last_doorbell = notification.value;
  }
}

bool CommandProcessor::signal_queue_exception(uint32_t queue_id, uint32_t process_id,
                                              uint64_t status, bool publish_interrupt) {
  {
    std::lock_guard<std::recursive_mutex> lk(hw_queue_mutex_);
    auto queue = std::find_if(
        compute_queues_.begin(), compute_queues_.end(), [&](const ComputeQueueRecord &candidate) {
          return candidate.queue_id == queue_id && candidate.process_id == process_id;
        });
    if (queue == compute_queues_.end() || queue->exception_status_va == 0)
      return false;
    queue->exception_suspended = true;
  }

  for (auto *cu : cus_) {
    cu->with_wave_state_locked([&] {
      for (uint32_t slot = 0; slot < cu->num_wf_slots(); ++slot) {
        auto *wave = cu->wf(slot);
        if (wave && !wave->is_halted() && wave->process_id() == process_id &&
            wave->queue_id() == queue_id) {
          wave->set_fatal_exception_pending(true);
          wave->set_debug_suspended(true);
        }
      }
    });
  }
  if (!publish_interrupt)
    return true;
  return publish_queue_exception(queue_id, process_id, status);
}

bool CommandProcessor::publish_queue_exception(uint32_t queue_id, uint32_t process_id,
                                               uint64_t status, bool wait_for_ack) {
  if (!gpu_vm_)
    return false;

  uint64_t exception_status_va = 0;
  uint32_t exception_event_id = 0;
  AddressSpaceHandle address_space;
  InterruptSink interrupt_sink;
  std::chrono::milliseconds ack_timeout;
  {
    std::lock_guard<std::recursive_mutex> lk(hw_queue_mutex_);
    auto queue = std::ranges::find_if(compute_queues_, [&](const ComputeQueueRecord &candidate) {
      return candidate.queue_id == queue_id && candidate.process_id == process_id;
    });
    if (queue == compute_queues_.end() || queue->exception_status_va == 0)
      return false;
    exception_status_va = queue->exception_status_va;
    exception_event_id = queue->exception_event_id;
    ack_timeout = runtime_exception_ack_timeout_;
    address_space = queue->address_space;
    interrupt_sink = queue->interrupt_sink;
  }

  const AtomicLoadResult previous =
      gpu_vm_->atomic_load(address_space, exception_status_va, sizeof(uint64_t));
  if (previous.outcome != VmAccessOutcome::Complete)
    return false;
  const uint64_t combined_status = previous.value | status;
  if (gpu_vm_->atomic_store(address_space, exception_status_va, sizeof(combined_status),
                            combined_status) != VmAccessOutcome::Complete)
    return false;
  interrupt_sink.deliver(process_id, exception_event_id);
  if (!wait_for_ack)
    return true;
  const auto deadline = std::chrono::steady_clock::now() + ack_timeout;
  AtomicLoadResult exception_status =
      gpu_vm_->atomic_load(address_space, exception_status_va, sizeof(uint64_t));
  while (exception_status.outcome == VmAccessOutcome::Complete &&
         exception_status.value == combined_status && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
    exception_status = gpu_vm_->atomic_load(address_space, exception_status_va, sizeof(uint64_t));
  }
  return exception_status.outcome == VmAccessOutcome::Complete &&
         exception_status.value != combined_status;
}

QueuePrepareCloseStatus
CommandProcessor::prepare_unregister_queue_registration(uint64_t registration_id) noexcept {
  if (!engine()) {
    std::lock_guard lock(hw_queue_mutex_);
    for (auto &queue : compute_queues_)
      queue.command_retry_pending = false;
    service_command_streams(0);
  }
  return close_queue_registration(registration_id, false);
}

bool CommandProcessor::unregister_queue_registration(uint64_t registration_id) {
  return close_queue_registration(registration_id, true) == QueuePrepareCloseStatus::Ready;
}

QueuePrepareCloseStatus CommandProcessor::close_queue_registration(uint64_t registration_id,
                                                                   bool force) noexcept {
  if (registration_id == 0)
    return QueuePrepareCloseStatus::Faulted;

  bool drop_replicas = false;
  bool retry_close = false;
  uint32_t queue_id = 0;
  uint32_t process_id = 0;
  {
    std::unique_lock<std::shared_mutex> structure_lock(queue_structure_mutex_);
    // Holds hw_queue_mutex_ across with_wave_state_locked(), which is the order
    // the dispatch path uses too (handle_doorbell -> dispatch_workgroups ->
    // dispatch_wf). Nothing takes them the other way any more: a wave reaching
    // s_endpgm under the wave-state lock queues its completion instead of sending
    // it, and WaveStateGuard delivers it after that lock is dropped. Keep it that
    // way -- a CU-side call back into the CP while the wave-state lock is held
    // would deadlock a DESTROY_QUEUE against the engine worker.
    std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
    const std::vector<ComputeQueueRecord>::iterator queue =
        std::ranges::find(compute_queues_, registration_id, &ComputeQueueRecord::registration_id);
    if (queue == compute_queues_.end())
      return QueuePrepareCloseStatus::Faulted;
    if (!force) {
      if (queue->publication_faulted || queue->command_fault_pending ||
          (queue->faulted && (!queue->entries.empty() || !queue->dispatches.entries.empty())))
        return QueuePrepareCloseStatus::Faulted;
      retry_close = queue->read_pointer_journal.publication_pending() ||
                    queue->publication_retry_pending || queue->idle_publication.active() ||
                    // After delivery, ROCr owns the scratch request and may suspend the queue
                    // while allocating backing. Only an unfinished notification publication
                    // must delay that removal.
                    queue->scratch_request.publication_pending() ||
                    queue->scratch_reclaim.active() || queue->has_pending_commands();
    }
    if (!retry_close) {
      queue_id = queue->queue_id;
      process_id = queue->process_id;
      for (auto *cu : cus_) {
        cu->with_wave_state_locked([&] {
          for (uint32_t slot = 0; slot < cu->num_wf_slots(); ++slot) {
            auto *wave = cu->wf(slot);
            if (wave && !wave->is_halted() && wave->process_id() == process_id &&
                wave->queue_id() == queue_id)
              wave->halt();
          }
        });
      }
      drop_replicas = queue->xcd_fanout;
      for (const DispatchEntry &entry : queue->entries)
        dispatch_launch_metadata_.erase(entry.dispatch_id);
      // Cancel every child stream before releasing the queue's VM lease.
      if (!queue->commands.submissions.empty() || !queue->dispatches.entries.empty())
        fail_pm4_queue(*queue, queue->dispatches);
      // Any shares still unpublished here are simply dropped. They cannot be
      // credited to the grid from this thread: publish_share is the release edge
      // that must follow this XCD's cache write-back, and flushing walks cus_,
      // which belong to the engine partition rather than to the caller. Crediting
      // without the flush would let the owner fire the completion signal with this
      // XCD's results still cached.
      //
      // Dropping them is safe because a fan-out queue is only ever destroyed on
      // every XCD at once: the KFD paths sweep all command processors, and an
      // owner cascades to its replicas below. No XCD is left holding a grid that
      // can no longer retire. Unregistering a lone replica is not supported.
      compute_queues_.erase(queue);
    }
  }
  if (retry_close) {
    if (engine())
      engine()->schedule_event_now(&doorbell_event_);
    return QueuePrepareCloseStatus::Busy;
  }
  // Tear the replicas down outside our own lock: a peer's unregister_queue takes
  // that peer's lock, and holding both would fix no order between two CPs whose
  // queues are being destroyed concurrently.
  if (drop_replicas) {
    const auto num_xcds = static_cast<uint32_t>(xcd_peers_.size());
    for (uint32_t rank = 0; rank < num_xcds; ++rank) {
      if (rank != xcd_rank_)
        xcd_peers_[rank]->unregister_queue(queue_id, process_id);
    }
  }

  // Reap the monitor when the last host queue is removed. This runs after
  // releasing hw_queue_mutex_: the poller needs that mutex to finish its current
  // scan. The poll loop may also be in the engine event-queue path or the
  // interrupt/event-state callback, but neither path enters a KFD ioctl or acquires
  // KfdProcess::op_mutex_, which the production callers hold here. Preserve that
  // invariant: no poll-loop callback may wait for a lock held by an
  // unregister_queue() caller. The synchronous join makes queue-destroy latency
  // include at most the current poll iteration and its bounded callbacks. The
  // helper rechecks the queue set while holding the lifecycle mutex, so a concurrent
  // registration either keeps this monitor alive or starts a new one after the join.
  stop_doorbell_monitor_if_idle();
  return QueuePrepareCloseStatus::Ready;
}

void CommandProcessor::unregister_queue(uint32_t queue_id, uint32_t process_id) {
  uint64_t registration_id = 0;
  {
    std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
    const ComputeQueueRecord *queue = find_compute_queue(queue_id, process_id);
    if (queue != nullptr)
      registration_id = queue->registration_id;
  }
  (void)unregister_queue_registration(registration_id);
}

void CommandProcessor::set_queue_cu_selection(uint32_t queue_id, uint32_t process_id,
                                              const QueueCuSelection &enabled_cus) {
  {
    std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
    for (auto &queue : compute_queues_) {
      if (queue.queue_id == queue_id && queue.process_id == process_id)
        queue.enabled_cus = enabled_cus;
    }
  }
  if (engine())
    engine()->schedule_event_now(doorbell_event());
}

bool CommandProcessor::update_queue(uint32_t queue_id, uint32_t process_id, uint64_t ring_base_va,
                                    uint32_t ring_size, uint32_t queue_percentage) {
  uint64_t registration_id = 0;
  {
    std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
    const ComputeQueueRecord *queue = find_compute_queue(queue_id, process_id);
    if (queue != nullptr)
      registration_id = queue->registration_id;
  }
  return update_queue_registration(registration_id, ring_base_va, ring_size, queue_percentage);
}

bool CommandProcessor::update_queue_registration(uint64_t registration_id, uint64_t ring_base_va,
                                                 uint32_t ring_size, uint32_t queue_percentage) {
  if (registration_id == 0)
    return false;
  if (ring_base_va == 0 || ring_base_va % 4 || ring_size < 4 || ring_size % 4)
    return false;

  const bool suspended = queue_percentage == 0;
  bool found = false;
  bool changed = false;
  bool wake_command_processor = false;
  bool update_replicas = false;
  uint32_t queue_id = 0;
  uint32_t process_id = 0;
  {
    std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
    for (auto &q : compute_queues_) {
      if (q.registration_id == registration_id) {
        if (q.packet_format == QueuePacketFormat::Aql &&
            !valid_aql_packet_ring(ring_base_va, ring_size))
          return false;
        if (!aql_metadata::valid_ring_layout(ring_base_va, ring_size, q.metadata_ring_size))
          return false;
        if (q.packet_format == QueuePacketFormat::Pm4 &&
            (q.ring_base_va != ring_base_va || q.ring_size != ring_size)) {
          if (q.has_pending_commands())
            return false;
          q.read_pointer_journal.reset();
          q.command_access.reset();
          q.last_doorbell = ~uint64_t(0);
        }
        found = true;
        queue_id = q.queue_id;
        process_id = q.process_id;
        q.ring_base_va = ring_base_va;
        q.ring_size = ring_size;
        update_replicas = q.xcd_fanout;
        changed = q.runtime_suspended != suspended;
        q.runtime_suspended = suspended;
        // Only consume the deferral once *no* reason still gates the queue.
        // debug_work_deferred is shared by both suspend reasons, so clearing it
        // here while the debugger still holds the gate would leave the later
        // debugger resume with nothing to release, and the already-fetched
        // packets would sit until an unrelated doorbell arrived.
        if (changed && !suspended && !q.debug_suspended && !q.exception_suspended)
          wake_command_processor = std::exchange(q.debug_work_deferred, false);
        break;
      }
    }
  }
  if (update_replicas) {
    const uint32_t num_xcds = static_cast<uint32_t>(xcd_peers_.size());
    for (uint32_t rank = 0; rank < num_xcds; ++rank) {
      if (rank != xcd_rank_)
        (void)xcd_peers_[rank]->update_queue(queue_id, process_id, ring_base_va, ring_size,
                                             queue_percentage);
    }
  }
  if (!changed)
    return found;
  for (auto *cu : cus_) {
    cu->with_wave_state_locked([&] {
      for (uint32_t slot = 0; slot < cu->num_wf_slots(); ++slot) {
        auto *wave = cu->wf(slot);
        if (wave && !wave->is_halted() && wave->process_id() == process_id &&
            wave->queue_id() == queue_id)
          // The runtime's own pause reason. Writing the debugger's bit here let
          // a runtime resume clear a debugger pause, and a debugger or CWSR
          // resume clear an active runtime pause.
          wave->set_runtime_suspended(suspended);
      }
    });
    if (!suspended)
      cu->schedule_work_async();
  }
  // Runtime resume has to release deferred queue work the same way a debugger
  // resume does, or already-fetched work sits until the next doorbell.
  if ((wake_command_processor || (changed && !suspended)) && engine())
    engine()->schedule_event_now(&doorbell_event_);
  return found;
}

void CommandProcessor::set_queue_debug_suspended(uint32_t queue_id, uint32_t process_id,
                                                 bool suspended, bool resolve_exception) {
  bool wake_command_processor = false;
  {
    std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
    for (ComputeQueueRecord &q : compute_queues_) {
      if (q.queue_id == queue_id && q.process_id == process_id) {
        const bool exception_resolved = resolve_exception && q.exception_suspended;
        if (q.debug_suspended == suspended && !exception_resolved)
          continue;
        if (exception_resolved)
          q.exception_suspended = false;
        q.debug_suspended = suspended;
        if (suspended) {
          // Existing queue work needs a resume pass only when the gate, rather
          // than an earlier incomplete dispatch, is what prevents it from
          // running. Resident waves are reactivated directly by KFD resume.
          // Accumulate: the flag is shared with the runtime's suspend reason,
          // and fetch_from_queue() may already have recorded a deferral for a
          // queue the runtime had gated. Assigning would discard it, leaving
          // neither resume path with anything to release.
          if (q.next_dispatch_idx < q.entries.size()) {
            const auto &entry = q.entries[q.next_dispatch_idx];
            const bool barrier_ready =
                !entry.wait_for_predecessors || barrier_satisfied(q, q.next_dispatch_idx);
            q.debug_work_deferred |=
                barrier_ready && (entry.is_non_kernel() || !entry.fully_dispatched());
          }
        } else if (!q.runtime_suspended && !q.exception_suspended) {
          wake_command_processor |= std::exchange(q.debug_work_deferred, false);
        }
      }
    }
  }
  if (wake_command_processor && engine())
    engine()->schedule_event_now(&doorbell_event_);
}

void CommandProcessor::set_doorbell_base(uint32_t process_id, void *base) {
  std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
  for (auto &q : compute_queues_) {
    if (q.process_id == process_id)
      q.doorbell_base = base;
  }
}

void CommandProcessor::ensure_doorbell_monitor() {
  std::lock_guard<std::mutex> lock(doorbell_thread_mutex_);
  // A monitor is already servicing this CP's queues — nothing to do. (The
  // register→ring window is fine: the live monitor scans every registered queue
  // each pass, so it will pick up the queue this call just added.)
  if (doorbell_running_)
    return;
  util::Logger::cp([&](auto &os) { os << std::format("{}: STARTING doorbell thread", name()); });
  // Construct the thread BEFORE setting doorbell_running_: if the jthread
  // constructor throws (std::system_error on thread-creation failure) the flag
  // must stay false so a later ensure_doorbell_monitor() retries instead of
  // no-oping forever. We still hold doorbell_thread_mutex_, so teardown cannot
  // observe the new handle until both it and the running flag are published.
  assert(!doorbell_thread_.joinable());
  doorbell_thread_ = std::jthread([this](std::stop_token stop) { doorbell_poll_loop(stop); });
  doorbell_running_ = true;
}

void CommandProcessor::stop_doorbell_monitor() {
  std::lock_guard<std::mutex> lock(doorbell_thread_mutex_);
  if (doorbell_thread_.joinable()) {
    doorbell_thread_.request_stop();
    doorbell_thread_.join();
  }
  doorbell_running_ = false;
}

void CommandProcessor::stop_doorbell_monitor_if_idle() {
  std::lock_guard<std::mutex> thread_lock(doorbell_thread_mutex_);
  {
    std::lock_guard<std::recursive_mutex> queue_lock(hw_queue_mutex_);
    // polls_kfd_queues(), not has_kfd_queues(): a fan-out replica is
    // host-accessible but is never polled, so keying this on presence would
    // strand a monitor on a CP whose own queue was destroyed while a replica of
    // some other queue happened to remain.
    if (polls_kfd_queues())
      return;
  }
  if (doorbell_thread_.joinable()) {
    doorbell_thread_.request_stop();
    doorbell_thread_.join();
  }
  doorbell_running_ = false;
}

std::optional<GpuVmAccess>
CommandProcessor::snapshot_gpu_access(AddressSpaceHandle address_space) const {
  if (gpu_vm_ == nullptr || !address_space)
    return std::nullopt;
  return gpu_vm_->snapshot(address_space);
}

AtomicLoadResult CommandProcessor::read_gpu_u64(AddressSpaceHandle address_space,
                                                uint64_t va) const {
  uint64_t val = 0;
  const std::optional<GpuVmAccess> access = snapshot_gpu_access(address_space);
  if (!access)
    return {.outcome = VmAccessOutcome::Faulted, .value = 0};
  if ((va & (alignof(uint64_t) - 1)) != 0) {
    const VmAccessOutcome outcome =
        access->read(va, std::as_writable_bytes(std::span<uint64_t, 1>(&val, 1)));
    return {.outcome = outcome, .value = val};
  }
  return access->atomic_load(va, sizeof(val));
}

AtomicLoadResult CommandProcessor::read_gpu_u64(const GpuVmAccess &access, uint64_t va) const {
  uint64_t value = 0;
  if ((va & (alignof(uint64_t) - 1)) == 0)
    return access.atomic_load(va, sizeof(value));
  const VmAccessOutcome outcome =
      access.read(va, std::as_writable_bytes(std::span<uint64_t, 1>(&value, 1)));
  return {.outcome = outcome, .value = value};
}

AtomicLoadResult CommandProcessor::read_gpu_u32(AddressSpaceHandle address_space,
                                                uint64_t va) const {
  uint32_t val = 0;
  const std::optional<GpuVmAccess> access = snapshot_gpu_access(address_space);
  if (!access)
    return {.outcome = VmAccessOutcome::Faulted, .value = 0};
  const VmAccessOutcome outcome =
      access->read(va, std::as_writable_bytes(std::span<uint32_t, 1>(&val, 1)));
  return {.outcome = outcome, .value = val};
}

AtomicLoadResult CommandProcessor::read_gpu_u32(const GpuVmAccess &access, uint64_t va) const {
  uint32_t value = 0;
  const VmAccessOutcome outcome =
      access.read(va, std::as_writable_bytes(std::span<uint32_t, 1>(&value, 1)));
  return {.outcome = outcome, .value = value};
}

VmAccessOutcome CommandProcessor::read_gpu_block(AddressSpaceHandle address_space, uint64_t va,
                                                 void *dst, size_t size) const {
  const std::optional<GpuVmAccess> access = snapshot_gpu_access(address_space);
  return access ? access->read(va, std::span<std::byte>(static_cast<std::byte *>(dst), size))
                : VmAccessOutcome::Faulted;
}

VmAccessOutcome CommandProcessor::read_gpu_block(const GpuVmAccess &access, uint64_t va, void *dst,
                                                 size_t size) const {
  return access.read(va, std::span<std::byte>(static_cast<std::byte *>(dst), size));
}

VmAccessOutcome CommandProcessor::write_gpu_block(AddressSpaceHandle address_space, uint64_t va,
                                                  const void *src, size_t size) {
  const std::optional<GpuVmAccess> access = snapshot_gpu_access(address_space);
  return access ? access->write(
                      va, std::span<const std::byte>(static_cast<const std::byte *>(src), size))
                : VmAccessOutcome::Faulted;
}

VmAccessOutcome CommandProcessor::write_gpu_block(const GpuVmAccess &access, uint64_t va,
                                                  const void *src, size_t size) {
  return access.write(va, std::span<const std::byte>(static_cast<const std::byte *>(src), size));
}

/// @brief Scan all AQL queues for doorbell changes; return true if any changed.
/// Caller must NOT hold hw_queue_mutex_.
bool CommandProcessor::scan_doorbells() {
  bool found = false;
  std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
  for (auto &q : compute_queues_) {
    // A replica shares the owner's ring and doorbell. Only the owning XCD may
    // consume them, or every XCD would dispatch the whole grid.
    if (q.fanout_replica)
      continue;
    uint64_t val;
    if (q.doorbell_mode == QueueDoorbellMode::Explicit)
      continue;
    if (q.doorbell_mode == QueueDoorbellMode::HostPolled) {
      if (!q.doorbell_base)
        continue;
      val = std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t *>(
                                          static_cast<char *>(q.doorbell_base) + q.doorbell_offset))
                .load(std::memory_order_acquire);
    } else {
      if (q.doorbell_va == 0)
        continue;
      const AtomicLoadResult loaded = read_gpu_u64(q.address_space, q.doorbell_va);
      if (loaded.outcome == VmAccessOutcome::Unavailable)
        continue;
      if (loaded.outcome != VmAccessOutcome::Complete) {
        q.faulted = true;
        continue;
      }
      val = loaded.value;
    }
    if (val != q.last_doorbell) {
      util::Logger::cp([&](auto &os) {
        os << std::format("{}: DOORBELL_CHANGE pid={} qid={} old={:#x} new={:#x} "
                          "db_base={} db_off={}",
                          name(), q.process_id, q.queue_id, q.last_doorbell, val,
                          reinterpret_cast<uintptr_t>(q.doorbell_base), q.doorbell_offset);
      });
      q.last_doorbell = val;
      found = true;
    }
  }
  return found;
}

bool CommandProcessor::schedule_retry_event() {
  bool expected = false;
  if (!retry_event_pending_.compare_exchange_strong(expected, true, std::memory_order_acq_rel,
                                                    std::memory_order_acquire))
    return false;
  engine()->schedule_event_next_tick(&retry_event_);
  return true;
}

void CommandProcessor::doorbell_poll_loop(std::stop_token stop) {
  using namespace std::chrono_literals;
  uint64_t poll_count = 0;
  // INVARIANT: this loop must re-read invalid_pending_/stall_pending_ (below) on
  // EVERY iteration, unconditionally. Those flags are level-triggered — the engine
  // clears them at handle_doorbell entry and re-sets them if a stall is still
  // unsatisfied — so this unconditional 100us heartbeat re-check is what guarantees a
  // pending stall is eventually retried. A future change that lets the loop skip the
  // flag re-read on some iterations (an early continue before the retry check) would
  // reintroduce a lost-wakeup.
  while (!stop.stop_requested()) {
    bool doorbell_changed = scan_doorbells();
    // Retry on a pending INVALID packet (the runtime has not finished writing it
    // yet) OR a pending barrier/dependency stall (waiting on a signal a peer rank
    // or another queue will write) even when no doorbell value changed. Pace those
    // retries with the same 100us idle wait rather than spinning: a real doorbell
    // change fires the event immediately (latency-sensitive), but a pending retry
    // only needs to poll until the awaited state changes, so it must not burn a
    // core. The wall-clock delay paces those polls; the following-tick timestamp
    // below prevents the resulting async retry stream from starving device work
    // already queued for that simulated tick.
    bool retry = !doorbell_changed && (invalid_pending_.load(std::memory_order_acquire) ||
                                       stall_pending_.load(std::memory_order_acquire));
    if (doorbell_changed)
      engine()->schedule_event_now(&doorbell_event_);
    else if (retry) {
      std::this_thread::sleep_for(100us);
      // This is a level-triggered poll, not a newly arrived doorbell. Put it at
      // the following simulated tick so a producer that polls faster than the
      // engine drains retries cannot indefinitely outrank CU work already queued
      // for that tick. Same-tick async/local arbitration then guarantees bounded
      // progress for both sources.
      schedule_retry_event();
    } else
      std::this_thread::sleep_for(100us);
    ++poll_count;

    // HQD idle monitoring: periodically fire HQD_IDLE for queues that are
    // currently empty. On real hardware the CP continuously monitors queue
    // activity and fires the idle interrupt whenever the queue is inactive.
    // Our drain_completions fires on the non-empty→empty transition, but a
    // process may create a new event AFTER that transition and miss the
    // signal. Re-broadcasting every ~10ms ensures late-created events see
    // the idle state within a bounded window.
    if (poll_count % 100 == 0) {
      // Snapshot the idle queues' process ids under the lock, then deliver interrupts
      // OUTSIDE it. Subscribers are external frontend callbacks whose internal
      // locking is opaque to the CP; invoking it while holding hw_queue_mutex_ risks a
      // lock-order inversion if that callback ever takes a lock held elsewhere while
      // acquiring hw_queue_mutex_.
      std::vector<std::pair<InterruptSink, uint32_t>> idle_queues;
      {
        std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
        for (size_t queue_index = 0; queue_index < compute_queues_.size(); ++queue_index) {
          // A replica does not own the queue, so it must not report it idle: its
          // shards drain before the owner's and the same KFD queue would otherwise
          // raise this from several CPs at once.
          if (compute_queues_[queue_index].fanout_replica ||
              compute_queues_[queue_index].packet_format != QueuePacketFormat::Aql)
            continue;
          if (!compute_queues_[queue_index].has_pending_commands() &&
              compute_queues_[queue_index].process_id != 0) {
            idle_queues.emplace_back(compute_queues_[queue_index].interrupt_sink,
                                     compute_queues_[queue_index].process_id);
          }
        }
      }
      for (const auto &[interrupt_sink, process_id] : idle_queues)
        interrupt_sink.deliver(process_id, 0);
    }

    if (poll_count % 5000 == 1) {
      std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
      for (auto &q : compute_queues_) {
        uint64_t current = q.last_doorbell;
        if (q.doorbell_mode == QueueDoorbellMode::HostPolled && q.doorbell_base) {
          current = std::atomic_ref<uint64_t>(
                        *reinterpret_cast<uint64_t *>(static_cast<char *>(q.doorbell_base) +
                                                      q.doorbell_offset))
                        .load(std::memory_order_acquire);
        } else if (q.doorbell_mode == QueueDoorbellMode::VmPolled && q.doorbell_va != 0) {
          const AtomicLoadResult loaded = read_gpu_u64(q.address_space, q.doorbell_va);
          if (loaded.outcome == VmAccessOutcome::Complete)
            current = loaded.value;
          else if (loaded.outcome != VmAccessOutcome::Unavailable)
            q.faulted = true;
        }
        util::Logger::cp([&](auto &os) {
          os << std::format("{}: DOORBELL_POLL pid={} qid={} current={:#x} last={:#x} "
                            "monitor_base={} db_off={} polls={}",
                            name(), q.process_id, q.queue_id, current, q.last_doorbell,
                            reinterpret_cast<uintptr_t>(q.doorbell_base), q.doorbell_offset,
                            poll_count);
        });
      }
    }
  }
}

ComputeQueueRecord *CommandProcessor::schedule_next_queue() {
  if (compute_queues_.empty())
    return nullptr;
  size_t start = next_queue_idx_;
  for (size_t i = 0; i < compute_queues_.size(); ++i) {
    size_t idx = (start + i) % compute_queues_.size();
    auto &qs = compute_queues_[idx];
    if (qs.faulted || qs.suspended())
      continue;
    if (qs.next_dispatch_idx < qs.entries.size()) {
      next_queue_idx_ = (idx + 1) % compute_queues_.size();
      return &qs;
    }
  }
  return nullptr;
}

bool CommandProcessor::barrier_satisfied(const ComputeQueueRecord &qs, size_t idx) const {
  if (idx == 0 && !qs.implicit_barrier_next)
    return true;

  // Barrier bit: all prior entries must be fully completed, device-wide. A prior
  // entry that is one XCD's share of a fanned-out dispatch is not done just
  // because this XCD finished it, so gate on the whole grid or this XCD would run
  // the next packet while a peer is still executing the previous one.
  for (size_t i = 0; i < idx; ++i) {
    if (!qs.entries[i].grid_fully_completed())
      return false;
  }
  return true;
}

void CommandProcessor::drain_pending_wg_completions() {
  for (const auto completion : pending_wg_completions_) {
    plugin_group_->onAmdgpuWorkgroupCompleted(completion.dispatch_id, completion.wg_id);
    for (auto *spi : spis_)
      if (spi->release_wgp_workgroup(completion.dispatch_id, completion.wg_id))
        break;
    mark_cluster_workgroup_complete(completion.dispatch_id, completion.wg_id);
    if (completion_)
      completion_->notify_wg_complete(completion.dispatch_id, completion.wg_id, compute_queues_);
    for (auto &queue : compute_queues_)
      for (auto &entry : queue.dispatches.entries)
        if (entry.dispatch_id == completion.dispatch_id) {
          ++entry.completed_wgs;
          if (engine())
            engine()->schedule_event_now(doorbell_event());
        }
  }
  pending_wg_completions_.clear();
}

void CommandProcessor::drain_pending_cluster_barrier_completions() {
  std::vector<PendingClusterBarrierCompletion> completions;
  {
    std::lock_guard<std::recursive_mutex> lock(cluster_placements_mutex_);
    completions.swap(pending_cluster_barrier_completions_);
  }

  for (auto &completion : completions) {
    std::vector<Wavefront *> members;
    for (auto [cu, peer_wg_id] : completion.peers) {
      auto peer_members =
          cu->complete_barrier(completion.dispatch_id, peer_wg_id, completion.completion_bit);
      members.insert(members.end(), peer_members.begin(), peer_members.end());
    }
    if (!members.empty())
      plugin_group_->onAmdgpuBarrierResolved(std::span<Wavefront *>(members));
  }
}

void CommandProcessor::register_cluster_workgroup(const DispatchEntry &entry, uint32_t local_wg_id,
                                                  uint32_t global_wg_id, ComputeUnitCore *cu,
                                                  uint32_t lds_base) {
  if (!entry.has_workgroup_clusters())
    return;
  std::lock_guard<std::recursive_mutex> lock(cluster_placements_mutex_);
  uint32_t cluster_base_wg_id =
      entry.cluster_base_local_wg_id(local_wg_id) + entry.workgroup_id_offset;
  uint64_t cluster_key = wg_key(entry.dispatch_id, cluster_base_wg_id);
  cu->pin_lds_until_cluster_retired(cluster_key);
  ClusterWorkgroupPlacement placement{};
  placement.cu = cu;
  placement.lds_base = lds_base;
  placement.cluster_key = cluster_key;
  placement.cluster_rank = entry.cluster_rank_for_flat_wg_id(global_wg_id);
  placement.cluster_size = entry.cluster_size();
  placement.peer_wg_ids.reserve(placement.cluster_size);
  for (uint32_t rank = 0; rank < placement.cluster_size; ++rank) {
    uint32_t peer_local_wg_id = entry.cluster_peer_local_wg_id(local_wg_id, rank);
    placement.peer_wg_ids.push_back(peer_local_wg_id + entry.workgroup_id_offset);
  }
  cluster_wg_placements_[wg_key(entry.dispatch_id, global_wg_id)] = std::move(placement);
  auto &barriers = cluster_barriers_[cluster_key];
  if (barriers.expected_member_count == 0) {
    barriers.expected_member_count = entry.cluster_size();
    barriers.member_count = entry.cluster_size();
  }
  barriers.registered_workgroups.insert(global_wg_id);
}

bool CommandProcessor::find_valid_cluster_barrier_locked(const Wavefront &wf, int32_t barrier_id,
                                                         ClusterWorkgroupPlacement *&placement,
                                                         ClusterBarrierState *&barriers) {
  if (barrier_id != kClusterBarrierId && barrier_id != kClusterTrapBarrierId)
    return false;
  auto placement_it = cluster_wg_placements_.find(wg_key(wf.dispatch_id(), wf.wg_id()));
  if (placement_it == cluster_wg_placements_.end())
    return false;
  auto barriers_it = cluster_barriers_.find(placement_it->second.cluster_key);
  if (barriers_it == cluster_barriers_.end() || barriers_it->second.member_count == 0 ||
      barriers_it->second.registered_workgroups.size() != barriers_it->second.expected_member_count)
    return false;
  placement = &placement_it->second;
  barriers = &barriers_it->second;
  return true;
}

bool CommandProcessor::find_valid_cluster_barrier_locked(
    const Wavefront &wf, int32_t barrier_id, const ClusterWorkgroupPlacement *&placement,
    const ClusterBarrierState *&barriers) const {
  if (barrier_id != kClusterBarrierId && barrier_id != kClusterTrapBarrierId)
    return false;
  auto placement_it = cluster_wg_placements_.find(wg_key(wf.dispatch_id(), wf.wg_id()));
  if (placement_it == cluster_wg_placements_.end())
    return false;
  auto barriers_it = cluster_barriers_.find(placement_it->second.cluster_key);
  if (barriers_it == cluster_barriers_.end() || barriers_it->second.member_count == 0 ||
      barriers_it->second.registered_workgroups.size() != barriers_it->second.expected_member_count)
    return false;
  placement = &placement_it->second;
  barriers = &barriers_it->second;
  return true;
}

bool CommandProcessor::cluster_barrier_valid(const Wavefront &wf, int32_t barrier_id) const {
  std::lock_guard<std::recursive_mutex> lock(cluster_placements_mutex_);
  const ClusterWorkgroupPlacement *placement = nullptr;
  const ClusterBarrierState *barriers = nullptr;
  return find_valid_cluster_barrier_locked(wf, barrier_id, placement, barriers);
}

uint32_t CommandProcessor::cluster_barrier_state(const Wavefront &wf, int32_t barrier_id,
                                                 uint32_t allocation_blocks) const {
  std::lock_guard<std::recursive_mutex> lock(cluster_placements_mutex_);
  const ClusterWorkgroupPlacement *placement = nullptr;
  const ClusterBarrierState *barriers = nullptr;
  if (!find_valid_cluster_barrier_locked(wf, barrier_id, placement, barriers))
    return 0;
  const uint32_t index = static_cast<uint32_t>(-barrier_id - kClusterBarrierBit);
  return 1u | ((barriers->member_count & 0x7fu) << 4) |
         ((barriers->signaled_workgroups[index].size() & 0x7fu) << 16) |
         ((allocation_blocks & 0x7u) << 24);
}

bool CommandProcessor::cluster_barrier_signal(Wavefront &wf, int32_t barrier_id) {
  bool is_first = false;
  const uint8_t completion_bit = static_cast<uint8_t>(-barrier_id);
  {
    std::lock_guard<std::recursive_mutex> lock(cluster_placements_mutex_);
    ClusterWorkgroupPlacement *placement = nullptr;
    ClusterBarrierState *barriers = nullptr;
    if (!find_valid_cluster_barrier_locked(wf, barrier_id, placement, barriers))
      return false;

    const uint32_t index = static_cast<uint32_t>(completion_bit - kClusterBarrierBit);
    auto [_, inserted] = barriers->signaled_workgroups[index].insert(wf.wg_id());
    if (!inserted)
      return false;
    is_first = barriers->signaled_workgroups[index].size() == 1;
    if (barriers->signaled_workgroups[index].size() < barriers->member_count)
      return is_first;

    barriers->signaled_workgroups[index].clear();
    PendingClusterBarrierCompletion completion{wf.dispatch_id(), completion_bit, {}};
    auto &peers = completion.peers;
    peers.reserve(placement->peer_wg_ids.size());
    for (uint32_t peer_wg_id : placement->peer_wg_ids) {
      auto peer = cluster_wg_placements_.find(wg_key(wf.dispatch_id(), peer_wg_id));
      if (peer != cluster_wg_placements_.end() && peer->second.cu)
        peers.emplace_back(peer->second.cu, peer_wg_id);
    }
    pending_cluster_barrier_completions_.push_back(std::move(completion));
  }

  // In serial mode this instruction already runs on the CP/engine thread, so
  // preserve immediate barrier resolution. Pool workers leave the compact
  // record queued until the fan-out rejoins below.
  if (dispatch_threads_ <= 1)
    drain_pending_cluster_barrier_completions();
  return is_first;
}

void CommandProcessor::mark_cluster_workgroup_complete(uint32_t dispatch_id, uint32_t wg_id) {
  // Cluster barriers resolve here, but complete_barrier() and the LDS reclaim
  // both reach into a CU. Collect them under the lock and act after it is
  // dropped -- see cluster_placements_mutex_.
  std::array<std::vector<std::pair<ComputeUnitCore *, uint32_t>>, 2> resolved_peers;
  std::vector<std::pair<ComputeUnitCore *, uint64_t>> unpin;
  {
    std::lock_guard<std::recursive_mutex> lock(cluster_placements_mutex_);
    auto it = cluster_wg_placements_.find(wg_key(dispatch_id, wg_id));
    if (it == cluster_wg_placements_.end() || it->second.completed)
      return;

    it->second.completed = true;
    const uint64_t cluster_key = it->second.cluster_key;
    const auto peer_wg_ids = it->second.peer_wg_ids;
    auto barrier_it = cluster_barriers_.find(cluster_key);
    if (barrier_it != cluster_barriers_.end() && barrier_it->second.member_count != 0) {
      auto &barriers = barrier_it->second;
      --barriers.member_count;
      for (uint32_t index = 0; index < barriers.signaled_workgroups.size(); ++index) {
        barriers.signaled_workgroups[index].erase(wg_id);
        if (barriers.member_count == 0 ||
            barriers.signaled_workgroups[index].size() < barriers.member_count)
          continue;
        barriers.signaled_workgroups[index].clear();
        for (uint32_t peer_wg_id : peer_wg_ids) {
          auto peer = cluster_wg_placements_.find(wg_key(dispatch_id, peer_wg_id));
          if (peer != cluster_wg_placements_.end() && !peer->second.completed && peer->second.cu)
            resolved_peers[index].emplace_back(peer->second.cu, peer_wg_id);
        }
      }
    }

    const bool all_completed = std::ranges::all_of(peer_wg_ids, [&](uint32_t peer_wg_id) {
      auto peer = cluster_wg_placements_.find(wg_key(dispatch_id, peer_wg_id));
      return peer != cluster_wg_placements_.end() && peer->second.completed;
    });
    if (all_completed) {
      for (uint32_t peer_wg_id : peer_wg_ids) {
        auto peer = cluster_wg_placements_.find(wg_key(dispatch_id, peer_wg_id));
        if (peer != cluster_wg_placements_.end() && peer->second.cu)
          unpin.emplace_back(peer->second.cu, cluster_key);
        cluster_wg_placements_.erase(wg_key(dispatch_id, peer_wg_id));
      }
      cluster_barriers_.erase(cluster_key);
    }
  }

  for (uint32_t index = 0; index < resolved_peers.size(); ++index) {
    std::vector<Wavefront *> members;
    const uint8_t completion_bit = static_cast<uint8_t>(kClusterBarrierBit + index);
    for (auto [cu, peer_wg_id] : resolved_peers[index]) {
      auto peer_members = cu->complete_barrier(dispatch_id, peer_wg_id, completion_bit);
      members.insert(members.end(), peer_members.begin(), peer_members.end());
    }
    if (!members.empty())
      plugin_group_->onAmdgpuBarrierResolved(std::span<Wavefront *>(members));
  }

  release_cluster_lds_pins(unpin);
}

// The waves halted (and freed) before their pin was released, so reclaim each
// peer CU's LDS once the whole cluster is done. Keep CU callbacks outside
// cluster_placements_mutex_. maybe_reset_lds_alloc() now checks the atomic
// activity count and LDS pin flag without taking the wave-state lock.
void CommandProcessor::release_cluster_lds_pins(
    const std::vector<std::pair<ComputeUnitCore *, uint64_t>> &unpin) {
  for (const auto &[cu, cluster_key] : unpin) {
    cu->unpin_lds_for_cluster(cluster_key);
    cu->maybe_reset_lds_alloc();
  }
}

void CommandProcessor::erase_cluster_workgroup(uint32_t dispatch_id, uint32_t wg_id) {
  // Collect pins under the placements lock, then release them and check for
  // LDS reclamation outside it, preserving the CU callback ordering.
  std::vector<std::pair<ComputeUnitCore *, uint64_t>> unpin;
  {
    std::lock_guard<std::recursive_mutex> lock(cluster_placements_mutex_);
    auto it = cluster_wg_placements_.find(wg_key(dispatch_id, wg_id));
    if (it == cluster_wg_placements_.end())
      return;
    if (it->second.cu)
      unpin.emplace_back(it->second.cu, it->second.cluster_key);
    cluster_barriers_.erase(it->second.cluster_key);
    cluster_wg_placements_.erase(it);
  }
  release_cluster_lds_pins(unpin);
}

void CommandProcessor::erase_cluster_workgroups(uint32_t dispatch_id) {
  std::vector<std::pair<ComputeUnitCore *, uint64_t>> unpin;
  {
    std::lock_guard<std::recursive_mutex> lock(cluster_placements_mutex_);
    for (auto it = cluster_wg_placements_.begin(); it != cluster_wg_placements_.end();) {
      if ((it->first >> 32) == dispatch_id) {
        cluster_barriers_.erase(it->second.cluster_key);
        if (it->second.cu)
          unpin.emplace_back(it->second.cu, it->second.cluster_key);
        it = cluster_wg_placements_.erase(it);
      } else {
        ++it;
      }
    }
  }
  release_cluster_lds_pins(unpin);
}

std::vector<ClusterLdsTarget>
CommandProcessor::cluster_lds_targets(uint32_t dispatch_id, uint32_t wg_id, uint32_t mcast_mask) {
  std::lock_guard<std::recursive_mutex> lock(cluster_placements_mutex_);
  std::vector<ClusterLdsTarget> targets;
  auto src_it = cluster_wg_placements_.find(wg_key(dispatch_id, wg_id));
  if (src_it == cluster_wg_placements_.end())
    return targets;

  const auto &src = src_it->second;
  const uint32_t self_mask = cluster_multicast_rank_mask(src.cluster_rank);
  // Defensive for direct helper callers; the issue path handles mask 0 locally.
  if (mcast_mask == 0 || (src.cluster_size <= 1 && (mcast_mask & self_mask) != 0)) {
    targets.push_back({src.cu, wg_id, src.lds_base, src.cluster_rank});
    return targets;
  }
  if (src.cluster_size <= 1)
    return targets;

  for (uint32_t rank = 0; rank < src.cluster_size && rank < kClusterMulticastMaskBits; ++rank) {
    if ((mcast_mask & (1u << rank)) == 0)
      continue;
    uint32_t peer_wg_id = src.peer_wg_ids[rank];
    auto peer_it = cluster_wg_placements_.find(wg_key(dispatch_id, peer_wg_id));
    if (peer_it == cluster_wg_placements_.end()) {
      throw std::runtime_error(std::format(
          "cluster multicast target is not resident: dispatch={} source_wg={} peer_wg={} rank={}",
          dispatch_id, wg_id, peer_wg_id, rank));
    }
    const auto &peer = peer_it->second;
    targets.push_back({peer.cu, peer_wg_id, peer.lds_base, peer.cluster_rank});
  }

  if (targets.empty() && (mcast_mask & self_mask) != 0)
    targets.push_back({src.cu, wg_id, src.lds_base, src.cluster_rank});
  return targets;
}

CommandProcessor::DispatchWorkgroupResult
CommandProcessor::dispatch_workgroups(DispatchEntry &entry) {
  assert(!cus_.empty() && "command processor has no compute units");

  // A dispatch may remain queued after its admission snapshot is invalidated.
  // Fall back to current operation snapshots for subsequent wave setup rather
  // than retrying the permanently revoked snapshot forever.
  if (entry.execution_access && entry.execution_access->revoked())
    entry.execution_access.reset();

  // A peer can publish the shared terminal-fault latch before this CP drains
  // its fault inbox. Stop placement as soon as that publication is visible;
  // the inbox drain will remove the entry and abort any waves already resident.
  if (entry.grid_faulted())
    return {.dispatched = 0, .outcome = VmAccessOutcome::Complete};

  // All waves in one workgroup currently land on one physical CU so the
  // existing barrier implementation remains local. WGP mode additionally
  // reserves that CU's sibling and binds the waves to their shared LDS pool.
  // Query the complete placement before dispatching for all-or-nothing setup.
  uint32_t dispatched = 0;
  // Places one workgroup's waves on the chosen CU. A terminal result is returned
  // only after every reservation made for this placement has been released.
  auto dispatch_to_placement =
      [&](uint32_t local_wg_id, uint32_t global_wg_id,
          const ShaderProcessorInput::WorkgroupPlacement &placement) -> VmAccessOutcome {
    // Fire the dispatch-execution-begin hook exactly once, on the first workgroup
    // actually placed on a CU, guarded by the per-dispatch flag. One begin per
    // dispatch, not per XCD. This cannot be pinned to the XCD that read the
    // packet: when the grid is smaller than the XCD count that XCD's share may be
    // empty, so it never places anything. Let whichever XCD places the grid's
    // first workgroup claim the report. The shared claim must run under the
    // plugin-group callback lock: otherwise the winner can be descheduled after
    // claiming while a peer publishes this dispatch's first wave callback.
    if (!entry.execution_begun) {
      entry.execution_begun = true;
      plugin_group_->onAmdgpuDispatchExecutionBeginOnce(entry.dispatch_id, [&]() {
        return !entry.grid_completion || entry.grid_completion->claim_execution_begin();
      });
    }
    ComputeUnitCore *cu = placement.cu;
    uint32_t lds_base = placement.lds_base;
    // Reserve AND fully initialize all waves BEFORE committing WG-completion
    // bookkeeping. begin_workgroup() installs the WG refcount and
    // register_cluster_workgroup() installs the LDS pin; both are released only via
    // release_wf() when the waves halt. Committing them first and then failing
    // mid-workgroup would orphan the refcount and pin, permanently blocking
    // maybe_reset_lds_alloc() on this CU. Two failure modes are covered by doing all
    // fallible work up front: a dispatch_wf() null (placement gating makes this
    // unreachable, but the assert is compiled out in release), and a terminal
    // register-initialization result (for example malformed launch metadata).
    // On either, release the reserved-but-uncommitted waves. Use
    // free_wavefront_resources() rather than halt(): these waves never executed, so
    // firing halt()'s onAmdgpuWavefrontHalted hook would feed observers a spurious
    // "completed" wave. free_wavefront_resources() frees the SGPR/VGPR blocks and
    // resets the slot without the hook or a CP completion notify — and since
    // begin_workgroup() has not run, there is no active_wgs_ entry / cluster pin to
    // unwind either. Also reclaim the placement's LDS/WGP reservation symmetrically
    // with how it was reserved:
    //   - CU / cluster mode: the SPI (or the direct CU path) advanced the CU's
    //     next_lds_alloc_ via allocate_lds(); maybe_reset_lds_alloc() rolls it back
    //     iff the CU is now idle (freeing these waves left no active waves) and
    //     unpinned. It correctly no-ops when peers of the same dispatch are resident.
    //   - WGP mode: allocate_workgroup() reserved SPI-side state (wgp.next_lds_alloc,
    //     wgp.active_workgroups, resident_wgp_workgroups_) that is NOT CU-local, so
    //     maybe_reset_lds_alloc() cannot reach it; release_wgp_workgroup() is the
    //     matching release (the same call notify_wg_complete uses on the normal path).
    // Without the WGP release a failed WGP dispatch would permanently pin that WGP.
    // Most workgroups need at most 32 waves. Keep their temporary reservation
    // list local, with dynamic storage for larger internal workgroups.
    std::array<Wavefront *, 32> local_wavefronts;
    std::vector<Wavefront *> large_wavefronts;
    std::span<Wavefront *> wg_wavefronts;
    if (entry.wfs_per_workgroup <= local_wavefronts.size()) {
      wg_wavefronts = std::span(local_wavefronts).first(entry.wfs_per_workgroup);
    } else {
      large_wavefronts.resize(entry.wfs_per_workgroup);
      wg_wavefronts = large_wavefronts;
    }
    uint32_t reserved_wavefronts = 0;
    const auto free_reserved = [&]() {
      for (auto *claimed : wg_wavefronts.first(reserved_wavefronts))
        cu->free_wavefront_resources(*claimed);
      if (entry.wgp_mode) {
        for (auto *spi : spis_)
          if (spi->release_wgp_workgroup(entry.dispatch_id, global_wg_id))
            break;
      }
      cu->maybe_reset_lds_alloc();
    };
    for (uint32_t w = 0; w < entry.wfs_per_workgroup; ++w) {
      Wavefront *wf = cu->dispatch_wf(global_wg_id, entry.kernel_entry_pc, entry.sgprs_per_wf,
                                      entry.vgprs_per_wf, entry.kernel_wave_size,
                                      entry.scratch_wave_limit_per_se);
      if (!wf) {
        assert(false && "dispatch_wf failed after placement was reserved");
        free_reserved();
        return VmAccessOutcome::Malformed;
      }
      wg_wavefronts[reserved_wavefronts++] = wf;
    }
    for (uint32_t w = 0; w < entry.wfs_per_workgroup; ++w) {
      Wavefront *wf = wg_wavefronts[w];
      wf->set_lds_base(lds_base);
      wf->set_lds_size(aligned_lds_bytes_per_workgroup(entry));
      wf->set_lds(placement.lds);
      wf->set_dispatch_id(entry.dispatch_id);
      wf->set_aql_packet_id(entry.aql_packet_id);
      wf->set_code_load_bias(entry.code_load_bias);
      wf->set_wave_in_group(w);
      wf->set_address_space(entry.address_space);
      wf->set_vm_access(entry.execution_access);
      wf->set_process_id(entry.process_id);
      // Live PM4 applications share the host monotonic clock with DRM query
      // timestamps. Internal simulation workloads retain the modeled clock.
      wf->set_system_clock(entry.pm4_abi);
      wf->set_pm4_failure(entry.pm4_failure);
      wf->set_mode_raw(entry.initial_mode_raw);
      wf->set_queue_id(entry.queue_id);
      wf->set_exec(initial_exec_mask_for_wave(entry, global_wg_id, w, wf->wf_size()));
      const uint32_t relative_wg_id = global_wg_id - entry.workgroup_id_offset;
      const WorkgroupCoord coord = entry.local_wg_coord(relative_wg_id);
      wf->set_wg_coord(coord.x, coord.y, coord.z);
      wf->set_cluster_info(entry.cluster_rank_for_flat_wg_id(global_wg_id), entry.cluster_size());
      const VmAccessOutcome initialization = init_wavefront_regs(cu, wf, entry, global_wg_id, w);
      if (initialization != VmAccessOutcome::Complete) {
        free_reserved();
        if (initialization == VmAccessOutcome::Revoked && !entry.pm4_abi) {
          // No wave in this workgroup has executed. Discard its partial register
          // setup and recapture access on retry; scratch provisioning is idempotent.
          // Keep a retry scheduled even when releasing these reservations left
          // every CU idle. PM4 retains its submission-failure path.
          entry.execution_access.reset();
          if (engine())
            arm_stall_recheck(engine()->context(partition_id()).current_tick());
          return VmAccessOutcome::Unavailable;
        }
        return initialization;
      }
    }

    // All fallible per-wave work succeeded: commit WG bookkeeping and the cluster pin.
    cu->begin_workgroup(entry.dispatch_id, global_wg_id, entry.wfs_per_workgroup,
                        entry.num_named_barriers);
    register_cluster_workgroup(entry, local_wg_id, global_wg_id, cu, lds_base);

    plugin_group_->onAmdgpuWorkgroupDispatched(entry.dispatch_id, global_wg_id,
                                               cu->vgpr_allocation_block_size(),
                                               cu->sgpr_allocation_block_size(), wg_wavefronts);
    for (auto *wf : wg_wavefronts)
      plugin_group_->onAmdgpuWavefrontDispatched(*wf);

    ++entry.dispatched_wgs;
    ++dispatched;
    dispatched_workgroups_.fetch_add(1, std::memory_order_relaxed);
    return VmAccessOutcome::Complete;
  };

  while (entry.dispatched_wgs < entry.total_wgs) {
    if (entry.grid_faulted() ||
        (entry.pm4_scratch_pool && !entry.pm4_scratch_pool->available(entry.wfs_per_workgroup)))
      break;
    if (entry.has_workgroup_clusters()) {
      assert(!entry.wgp_mode && "workgroup clusters are gfx1250-only and use CU mode");
      // The SPI interface chooses one WG at a time and cannot reserve all peers
      // in a cluster atomically. Plan clusters directly across the CP-visible CU
      // list until SPI grows an all-or-nothing cluster placement API.
      uint32_t cluster_size = entry.cluster_size();
      assert(entry.dispatched_wgs % cluster_size == 0 &&
             "clustered dispatch advances by whole clusters");
      assert(entry.total_wgs - entry.dispatched_wgs >= cluster_size &&
             "validate_cluster_shape guarantees a complete trailing cluster");
      // dispatched_wgs counts workgroups; the chunk here is a whole cluster, so
      // convert to a cluster index before asking the shard for its ordinal.
      uint32_t cluster_ordinal = entry.chunk_ordinal_for(entry.dispatched_wgs / cluster_size);
      uint32_t local_wg_id = entry.cluster_base_local_wg_id_for_ordinal(cluster_ordinal);
      std::vector<PlannedWorkgroup> plan;
      size_t planned_next_cu = next_cu_;
      if (!plan_cluster_workgroups(entry, local_wg_id, next_cu_, cus_, plan, planned_next_cu)) {
        if (!any_active_wavefronts(cus_))
          return {.dispatched = dispatched, .outcome = VmAccessOutcome::Malformed};
        break;
      }
      next_cu_ = planned_next_cu;
      // A cluster is all-or-nothing: dispatch_to_placement() commits each peer's WG
      // bookkeeping (begin_workgroup) and LDS cluster pin (register_cluster_workgroup)
      // as it succeeds. If a later peer fails, the already-committed peers would
      // otherwise keep their refcount and pin forever, permanently blocking
      // maybe_reset_lds_alloc() on those CUs. Track the committed peers and roll
      // them back before returning the terminal outcome.
      std::vector<std::pair<ComputeUnitCore *, uint32_t>> committed_peers;
      committed_peers.reserve(plan.size());
      for (const PlannedWorkgroup &planned_workgroup : plan) {
        if (entry.grid_faulted()) {
          for (const auto &[compute_unit, global_workgroup_id] : committed_peers) {
            erase_cluster_workgroup(entry.dispatch_id, global_workgroup_id);
            compute_unit->abort_workgroup(entry.dispatch_id, global_workgroup_id);
          }
          const uint32_t rolled_back = static_cast<uint32_t>(committed_peers.size());
          entry.dispatched_wgs -= rolled_back;
          dispatched -= rolled_back;
          return {.dispatched = dispatched, .outcome = VmAccessOutcome::Complete};
        }
        ShaderProcessorInput::WorkgroupPlacement placement{
            planned_workgroup.cu, &planned_workgroup.cu->lds(),
            planned_workgroup.cu->allocate_lds(entry.group_segment_fixed_size)};
        const VmAccessOutcome placement_outcome = dispatch_to_placement(
            planned_workgroup.local_wg_id, planned_workgroup.global_wg_id, placement);
        if (placement_outcome != VmAccessOutcome::Complete) {
          // dispatch_to_placement already unwound its own uncommitted waves.
          // Remove every earlier peer's pin, waves, and completion bookkeeping
          // before the caller faults and erases the dispatch entry.
          for (const auto &[compute_unit, global_workgroup_id] : committed_peers) {
            erase_cluster_workgroup(entry.dispatch_id, global_workgroup_id);
            compute_unit->abort_workgroup(entry.dispatch_id, global_workgroup_id);
          }
          const uint32_t rolled_back = static_cast<uint32_t>(committed_peers.size());
          entry.dispatched_wgs -= rolled_back;
          dispatched -= rolled_back;
          return {.dispatched = dispatched, .outcome = placement_outcome};
        }
        committed_peers.emplace_back(planned_workgroup.cu, planned_workgroup.global_wg_id);
      }
      continue;
    }

    // Unclustered: the chunk is a single workgroup, so dispatched_wgs indexes
    // the shard's chunks directly and the shard maps that to a grid-wide id.
    // An unsharded entry maps the ordinal to itself.
    uint32_t local_wg_id = entry.chunk_ordinal_for(entry.dispatched_wgs);
    uint32_t global_wg_id = local_wg_id + entry.workgroup_id_offset;

    // SPI selects the CU or sibling-CU WGP based on descriptor mode and
    // resource availability.
    std::optional<ShaderProcessorInput::WorkgroupPlacement> placement;
    if (!spis_.empty()) {
      for (auto *spi : spis_) {
        placement = spi->allocate_workgroup(entry, global_wg_id);
        if (placement)
          break;
      }
    } else if (!entry.wgp_mode) {
      for (size_t attempt = 0; attempt < cus_.size(); ++attempt) {
        size_t cu_idx = (next_cu_ + attempt) % cus_.size();
        if (!entry.allows_cu(cus_[cu_idx]))
          continue;
        if (cus_[cu_idx]->can_accept_workgroup(entry.wfs_per_workgroup,
                                               entry.group_segment_fixed_size,
                                               entry.scratch_wave_limit_per_se)) {
          auto *cu = cus_[cu_idx];
          placement = ShaderProcessorInput::WorkgroupPlacement{
              cu, &cu->lds(), cu->allocate_lds(entry.group_segment_fixed_size)};
          next_cu_ = (cu_idx + 1) % cus_.size();
          break;
        }
      }
    }

    if (!placement) {
      if (!any_active_wavefronts(cus_))
        return {.dispatched = dispatched, .outcome = VmAccessOutcome::Malformed};
      break;
    }

    const VmAccessOutcome placement_outcome =
        dispatch_to_placement(local_wg_id, global_wg_id, *placement);
    if (placement_outcome != VmAccessOutcome::Complete)
      return {.dispatched = dispatched, .outcome = placement_outcome};
  }
  return {.dispatched = dispatched, .outcome = VmAccessOutcome::Complete};
}

// ---------------------------------------------------------------------------
// Completion notification from CU
// ---------------------------------------------------------------------------

void CommandProcessor::notify_wg_complete(uint32_t dispatch_id, uint32_t wg_id) {
  util::Logger::cp(
      [&](auto &os) { os << std::format("WG_COMPLETE d={} wg={}", dispatch_id, wg_id); });
  std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
  pending_wg_completions_.push_back({dispatch_id, wg_id});
  if (dispatch_threads_ <= 1) {
    drain_pending_wg_completions();
    // Without worker fan-out, completion arrives on the CP's engine thread, so
    // retire promptly while another resident dispatch may wait on this signal.
    // Parallel workers defer all stateful retirement until their batch rejoins.
    (void)drain_completions();
  }
}

bool CommandProcessor::drain_completions() {
  if (!completion_)
    return true;

  bool stop_processing = false;
  bool retry_pending = false;
  for (;;) {
    const CompletionDrainResult result = completion_->drain_completions(compute_queues_);
    retry_pending |= result.retry_pending;
    if (!result.terminal_fault)
      break;

    stop_processing = true;
    const CompletionDrainFault &fault = *result.terminal_fault;
    if (!fault.queue_idle) {
      const std::vector<ComputeQueueRecord>::iterator queue =
          std::ranges::find_if(compute_queues_, [&](const ComputeQueueRecord &candidate) {
            return candidate.queue_id == fault.queue_id && candidate.process_id == fault.process_id;
          });
      if (queue != compute_queues_.end())
        queue->publication_faulted = true;
      notify_dispatch_vm_fault(fault.queue_id, fault.process_id, fault.dispatch_id, fault.outcome);
      continue;
    }

    const std::vector<ComputeQueueRecord>::iterator queue =
        std::ranges::find_if(compute_queues_, [&](const ComputeQueueRecord &candidate) {
          return candidate.queue_id == fault.queue_id && candidate.process_id == fault.process_id;
        });
    if (queue != compute_queues_.end()) {
      queue->publication_faulted = true;
      queue->faulted = true;
    }
  }

  if (retry_pending) {
    arm_stall_recheck(engine()->context(partition_id()).current_tick());
  }
  return !stop_processing;
}

bool CommandProcessor::fault_dispatch_local(uint32_t queue_id, uint32_t process_id,
                                            uint64_t dispatch_id, VmAccessOutcome outcome) {
  bool found = false;
  {
    // A shard may still be in transit from its owner. Remove it under the leaf
    // inbox lock without taking the queue lock, preserving the existing
    // cross-CP lock ordering.
    std::lock_guard<std::mutex> lock(fanout_inbox_mutex_);
    const size_t old_size = fanout_inbox_.size();
    std::erase_if(fanout_inbox_, [&](DispatchEntry &entry) {
      if (entry.queue_id != queue_id || entry.process_id != process_id ||
          entry.dispatch_id != dispatch_id)
        return false;
      entry.terminal_faulted = true;
      if (entry.grid_completion)
        entry.grid_completion->mark_faulted();
      fanout_launch_metadata_inbox_.erase(entry.dispatch_id);
      return true;
    });
    found = fanout_inbox_.size() != old_size;
  }

  std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
  const std::vector<ComputeQueueRecord>::iterator queue =
      std::ranges::find_if(compute_queues_, [&](const ComputeQueueRecord &candidate) {
        return candidate.queue_id == queue_id && candidate.process_id == process_id;
      });
  if (queue == compute_queues_.end())
    return false;
  ComputeQueueRecord &state = *queue;

  size_t index = 0;
  for (std::deque<DispatchEntry>::iterator entry = state.entries.begin();
       entry != state.entries.end();) {
    if (entry->dispatch_id != dispatch_id) {
      ++entry;
      ++index;
      continue;
    }
    found = true;
    entry->terminal_faulted = true;
    if (entry->grid_completion)
      entry->grid_completion->mark_faulted();
    if (index < state.next_dispatch_idx)
      --state.next_dispatch_idx;
    entry = state.entries.erase(entry);
  }
  if (!found)
    return false;

  dispatch_launch_metadata_.erase(static_cast<uint32_t>(dispatch_id));
  queue->faulted = true;
  // A peer fault can cancel an AQL entry while its vendor PM4 stream is blocked.
  // Release that stream and its VM snapshot along with the owning dispatch.
  if (state.packet_format == QueuePacketFormat::Aql && !state.commands.submissions.empty())
    fail_pm4_queue(state, state.dispatches);
  util::Logger::cp([&](auto &os) {
    os << std::format("{}: terminal VM fault pid={} qid={} dispatch={} outcome={}", name(),
                      process_id, queue_id, dispatch_id, static_cast<unsigned>(outcome));
  });
  erase_cluster_workgroups(static_cast<uint32_t>(dispatch_id));
  for (ComputeUnitCore *cu : cus_)
    cu->abort_dispatch(static_cast<uint32_t>(dispatch_id));
  return true;
}

void CommandProcessor::notify_dispatch_vm_fault(uint32_t queue_id, uint32_t process_id,
                                                uint64_t dispatch_id, VmAccessOutcome outcome) {
  if (outcome == VmAccessOutcome::Complete || outcome == VmAccessOutcome::Unavailable ||
      dispatch_id > std::numeric_limits<uint32_t>::max())
    return;

  if (!fault_dispatch_local(queue_id, process_id, dispatch_id, outcome))
    return;
  // A fault can originate on any shard. Cross-XCD delivery uses a leaf inbox
  // and the peer's event thread, exactly like dispatch fan-out, so this path
  // never acquires another CP's queue mutex.
  for (CommandProcessor *peer : xcd_peers_) {
    if (peer != nullptr && peer != this) {
      peer->accept_dispatch_fault({.queue_id = queue_id,
                                   .process_id = process_id,
                                   .dispatch_id = dispatch_id,
                                   .outcome = outcome});
    }
  }
}

// INVARIANT: on_cu_idle() runs on the owning partition's engine thread — it is
// invoked from CU::execute_quantum() (the CU's own per-partition tick event), so
// the CU, this CP, and the CUs it dispatches to all share one partition. Dispatch
// therefore happens inline (same-partition, non-thread-safe path); a
// schedule_event_now() here would collapse ticks and break causal ordering.
void CommandProcessor::on_cu_idle() {
  if (cus_.empty())
    return;

  std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);

  drain_pending_cluster_barrier_completions();
  drain_pending_wg_completions();
  if (!drain_completions())
    return;

  // Execute and retire eligible non-kernel entries at the head, then drain
  // again so a dependent kernel behind them can proceed.
  for (ComputeQueueRecord &qs : compute_queues_) {
    if (qs.faulted || qs.suspended() || qs.publication_retry_pending)
      continue;
    while (qs.next_dispatch_idx < qs.entries.size()) {
      auto &e = qs.entries[qs.next_dispatch_idx];
      if (e.wait_for_predecessors && !barrier_satisfied(qs, qs.next_dispatch_idx))
        break;
      if (!e.is_non_kernel())
        break;
      if (!execute_aql_pm4(qs, e, engine()->context(partition_id()).current_tick()))
        break;
      const uint32_t queue_id = e.queue_id;
      const uint32_t process_id = e.process_id;
      const uint32_t dispatch_id = e.dispatch_id;
      e.completed_wgs = e.total_wgs;
      const VmAccessOutcome outcome =
          completion_ ? completion_->complete_non_kernel(e) : VmAccessOutcome::Complete;
      if (outcome == VmAccessOutcome::Unavailable) {
        qs.publication_retry_pending = true;
        if (engine())
          arm_stall_recheck(engine()->context(partition_id()).current_tick());
        break;
      }
      if (outcome != VmAccessOutcome::Complete) {
        notify_dispatch_vm_fault(queue_id, process_id, dispatch_id, outcome);
        break;
      }
      ++qs.next_dispatch_idx;
    }
  }
  if (!drain_completions())
    return;

  // Continue dispatching pending workgroups onto the just-freed CU in this same
  // tick rather than deferring to a now+1 doorbell event. Deferring routed every
  // dispatch continuation through the doorbell path; across the many tiny kernels
  // of an RCCL collective the engine would idle a full tick between steps, adding
  // latency the host socket layer then paid for. dispatch_workgroups() schedules
  // the CU's own tick event in the ordinary case. Record quiesced CUs because a
  // CU containing only debug-halted waves needs an explicit activation when a
  // newly dispatched wave makes it runnable again.
  std::vector<bool> was_idle(cus_.size());
  for (size_t i = 0; i < cus_.size(); ++i)
    was_idle[i] = cus_[i]->is_idle();
  for (ComputeQueueRecord &qs : compute_queues_) {
    if (qs.faulted || qs.suspended() || qs.publication_retry_pending)
      continue;
    if (qs.next_dispatch_idx < qs.entries.size()) {
      auto &entry = qs.entries[qs.next_dispatch_idx];
      if (entry.wait_for_predecessors && !barrier_satisfied(qs, qs.next_dispatch_idx))
        continue;
      if (!entry.is_non_kernel() && !entry.fully_dispatched()) {
        const uint32_t queue_id = entry.queue_id;
        const uint32_t process_id = entry.process_id;
        const uint32_t dispatch_id = entry.dispatch_id;
        const DispatchWorkgroupResult result = dispatch_workgroups(entry);
        if (result.outcome != VmAccessOutcome::Complete) {
          notify_dispatch_vm_fault(queue_id, process_id, dispatch_id, result.outcome);
          continue;
        }
        if (result.dispatched > 0 && entry.fully_dispatched())
          ++qs.next_dispatch_idx;
      }
    }
  }
  for (size_t i = 0; i < cus_.size(); ++i) {
    if (was_idle[i] && !cus_[i]->is_idle())
      cus_[i]->schedule_work();
  }

  // The last workgroup of this XCD's share retires here, not in handle_doorbell,
  // so this is where a fanned-out shard parks to wait for its peers.
  arm_grid_wait_recheck();
}

void CommandProcessor::on_cu_pool_ready(ComputeUnitCore *cu) {
  if (dispatch_threads_ <= 1 || !engine() || !cu->has_runnable_wfs())
    return;

  std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
  const simdojo::Tick now = engine()->context(partition_id()).current_tick();
  // schedule_work() also runs when a new wave joins an already-active CU. Keep
  // that CU's established due tick, just as the serial driver keeps its queued
  // tick while executing_, rather than pulling resident work forward.
  auto due = pooled_due_ticks_.try_emplace(cu, now + 1).first;
  arm_dispatch_continuation(due->second);
}

bool CommandProcessor::step() {
  // Process dispatches across all queues.
  process_queues();
  return pending_entries() > 0;
}

void CommandProcessor::process_queues() {
  if (engine())
    service_command_streams(engine()->context(partition_id()).current_tick());
  for (ComputeQueueRecord &qs : compute_queues_) {
    if (qs.faulted || qs.suspended() || qs.publication_retry_pending)
      continue;
    while (qs.next_dispatch_idx < qs.entries.size()) {
      auto &entry = qs.entries[qs.next_dispatch_idx];

      if (entry.wait_for_predecessors && !barrier_satisfied(qs, qs.next_dispatch_idx))
        break; // Stalled on barrier bit.

      if (entry.is_non_kernel()) {
        if (!execute_aql_pm4(qs, entry, engine()->context(partition_id()).current_tick()))
          break;
        const uint32_t queue_id = entry.queue_id;
        const uint32_t process_id = entry.process_id;
        const uint32_t dispatch_id = entry.dispatch_id;
        entry.completed_wgs = entry.total_wgs;
        const VmAccessOutcome outcome =
            completion_ ? completion_->complete_non_kernel(entry) : VmAccessOutcome::Complete;
        if (outcome == VmAccessOutcome::Unavailable) {
          qs.publication_retry_pending = true;
          if (engine())
            arm_stall_recheck(engine()->context(partition_id()).current_tick());
          break;
        }
        if (outcome != VmAccessOutcome::Complete) {
          notify_dispatch_vm_fault(queue_id, process_id, dispatch_id, outcome);
          break;
        }
        ++qs.next_dispatch_idx;
        continue;
      }

      const uint32_t queue_id = entry.queue_id;
      const uint32_t process_id = entry.process_id;
      const uint32_t dispatch_id = entry.dispatch_id;
      const DispatchWorkgroupResult result = dispatch_workgroups(entry);
      if (result.outcome != VmAccessOutcome::Complete) {
        notify_dispatch_vm_fault(queue_id, process_id, dispatch_id, result.outcome);
        break;
      }
      if (entry.fully_dispatched())
        ++qs.next_dispatch_idx;
      if (result.dispatched == 0)
        break; // CU backpressure.
    }
  }
}

void CommandProcessor::prune_pooled_due_ticks() {
  // Admission and resume notify on_cu_pool_ready(), which inserts runnable
  // CUs. Only previously scheduled CUs can need pruning after a pause or
  // cancellation; idle CUs need no wave-state lock or slot scan.
  std::erase_if(pooled_due_ticks_, [](const auto &due) { return !due.first->has_runnable_wfs(); });
}

simdojo::Tick CommandProcessor::next_pooled_due_tick() {
  prune_pooled_due_ticks();
  simdojo::Tick next = simdojo::TICK_MAX;
  for (const auto &[_, tick] : pooled_due_ticks_)
    next = std::min(next, tick);
  return next;
}

FunctionalQuantumResult CommandProcessor::run_active_cus_once(simdojo::Tick now) {
  prune_pooled_due_ticks();
  active_cu_scratch_.clear();
  // Pruning already checked each CU's runnable-wave count. Keep SPI order
  // while selecting due work without repeating those checks.
  const auto append_due = [&](const auto &cus) {
    for (auto *cu : cus) {
      const auto due = pooled_due_ticks_.find(cu);
      if (due != pooled_due_ticks_.end() && due->second <= now)
        active_cu_scratch_.push_back(cu);
    }
  };
  if (!spis_.empty()) {
    for (auto *spi : spis_)
      append_due(spi->compute_units());
  } else {
    append_due(cus_);
  }

  if (active_cu_scratch_.empty())
    return {};

  quantum_result_scratch_.resize(active_cu_scratch_.size());
  uint32_t effective_threads =
      std::min<uint32_t>(dispatch_threads_, static_cast<uint32_t>(active_cu_scratch_.size()));
  FunctionalQuantumResult result;
  if (shared_dispatch_pool_)
    result =
        shared_dispatch_pool_->run(active_cu_scratch_, effective_threads, quantum_result_scratch_);
  else if (effective_threads > 1) {
    if (!local_dispatch_pool_ || local_dispatch_pool_->thread_count() < effective_threads)
      local_dispatch_pool_ = std::make_unique<CpuDispatchPool>(effective_threads);
    result =
        local_dispatch_pool_->run(active_cu_scratch_, effective_threads, quantum_result_scratch_);
  } else {
    quantum_result_scratch_.front() = active_cu_scratch_.front()->run_quantum();
    result = quantum_result_scratch_.front();
  }

  for (size_t i = 0; i < active_cu_scratch_.size(); ++i) {
    auto *cu = active_cu_scratch_[i];
    if (cu->has_runnable_wfs())
      pooled_due_ticks_[cu] = now + std::max<uint64_t>(1, quantum_result_scratch_[i].iterations);
    else
      pooled_due_ticks_.erase(cu);
  }
  return result;
}

bool CommandProcessor::register_drm_queue(ComputeQueueConfig config) {
  config.packet_format = QueuePacketFormat::Pm4;
  config.submission_queue = true;
  return register_queue(std::move(config)) != 0;
}

void CommandProcessor::unregister_drm_queues(uint32_t process_id) {
  std::vector<uint64_t> ids;
  {
    std::lock_guard lock(hw_queue_mutex_);
    for (const auto &queue : compute_queues_)
      if (queue.submission_queue && queue.process_id == process_id)
        ids.push_back(queue.registration_id);
  }
  for (const auto id : ids)
    (void)unregister_queue_registration(id);
}

void CommandProcessor::unregister_drm_queue(uint32_t queue_id, uint32_t process_id) {
  unregister_queue(queue_id, process_id);
}

void CommandProcessor::service_pm4_ring(ComputeQueueRecord &queue, simdojo::Tick now) {
  if (queue.packet_format != QueuePacketFormat::Pm4 || queue.submission_queue ||
      queue.fanout_replica || queue.faulted || queue.command_retry_pending)
    return;
  auto outcome = queue.read_pointer_journal.publish();
  if (outcome == VmAccessOutcome::Unavailable) {
    queue.command_retry_pending = true;
    arm_stall_recheck(now);
    return;
  }
  if (outcome != VmAccessOutcome::Complete) {
    queue.publication_faulted = true;
    fail_pm4_queue(queue, queue.dispatches);
    return;
  }
  if (queue.suspended()) {
    queue.debug_work_deferred = true;
    return;
  }
  if (!queue.commands.submissions.empty())
    return;
  if (queue.last_doorbell == ~uint64_t(0))
    return;
  if (!queue.command_access)
    queue.command_access = gpu_vm_->snapshot_pinned(queue.address_space);
  const auto &access = queue.command_access;
  if (!access) {
    fail_pm4_queue(queue, queue.dispatches);
    return;
  }
  // Publication replaces the GART binding; an unready pinned snapshot cannot observe it.
  if (!access->info().ready) {
    queue.command_access.reset();
    queue.command_retry_pending = true;
    arm_stall_recheck(now);
    return;
  }
  outcome = queue.read_pointer_journal.initialize(*access);
  if (outcome == VmAccessOutcome::Unavailable) {
    queue.command_retry_pending = true;
    arm_stall_recheck(now);
    return;
  }
  const auto consumer = queue.read_pointer_journal.cursor();
  const auto ring_dwords = queue.ring_size / 4;
  // Native PM4 compares ring offsets. A client may lift its software producer
  // by whole rings without changing the hardware consumer's starting offset.
  const auto notified = queue.doorbell_mode == QueueDoorbellMode::HostPolled
                            ? queue.last_doorbell % ring_dwords
                            : queue.last_doorbell;
  const auto producer = normalize_pm4_producer_cursor(notified, consumer, ring_dwords);
  if (outcome != VmAccessOutcome::Complete || !producer) {
    fail_pm4_queue(queue, queue.dispatches);
    return;
  }
  if (*producer == consumer) {
    queue.command_access.reset();
    return;
  }
  Pm4Submission submission;
  submission.buffers.push_back({.address = consumer * 4,
                                .dwords = static_cast<uint32_t>(*producer - consumer),
                                .ring_base = queue.ring_base_va,
                                .ring_bytes = queue.ring_size});
  submission.failure->wake = [this] {
    if (engine())
      engine()->schedule_event_now(doorbell_event());
  };
  queue.commands.submissions.push_back(std::move(submission));
}

void CommandProcessor::service_command_streams(simdojo::Tick now) {
  for (auto &queue : compute_queues_) {
    // AQL vendor streams advance only while their owning entry is eligible.
    if (queue.packet_format == QueuePacketFormat::Aql)
      continue;
    auto &state = queue.dispatches;
    if (queue.command_fault_pending) {
      queue.command_fault_pending = false;
      fail_pm4_queue(queue, state);
    }
    // Publication of a committed packet must finish even while execution is paused.
    service_pm4_ring(queue, now);
    if (queue.suspended() && !queue.commands.submissions.empty())
      queue.debug_work_deferred = true;
    if (queue.faulted || queue.suspended() || queue.command_retry_pending)
      continue;
    if (!queue.commands.submissions.empty() &&
        queue.commands.submissions.front().failure->failed.load(std::memory_order_acquire)) {
      fail_pm4_queue(queue, state);
      continue;
    }
    while (!state.entries.empty() && state.entries.front().fully_completed()) {
      const auto &entry = state.entries.front();
      flush_gpu_caches();
      if (entry.execution_begun)
        plugin_group_->onAmdgpuDispatchExecutionEnd(entry.dispatch_id);
      erase_cluster_workgroups(entry.dispatch_id);
      state.entries.pop_front();
    }
    fetch_pm4(queue, state, now);
    service_pm4_ring(queue, now);
    if (!queue.commands.submissions.empty() && state.entries.empty() && !queue.faulted)
      arm_stall_recheck(now);
    if (queue.faulted || state.entries.empty())
      continue;
    auto &entry = state.entries.front();
    // All DRM queues for this GPU are routed to the same CP. Delay a new
    // dispatch while another queue owns any overlapping scratch range.
    if (entry.pm4_scratch_pool && !entry.execution_begun) {
      const auto scratch_bytes = [&](const DispatchEntry &dispatch) -> uint64_t {
        return uint64_t{dispatch.pm4_scratch_waves_per_se} *
               std::max(scratch_wave_divisor_, scratch_shader_engine_count_) *
               dispatch.private_segment_fixed_size * dispatch.kernel_wave_size;
      };
      const bool busy = std::ranges::any_of(compute_queues_, [&](const auto &other) {
        if (&other == &queue || other.address_space != queue.address_space)
          return false;
        return std::ranges::any_of(other.dispatches.entries, [&](const auto &active) {
          if (!active.pm4_scratch_pool || !active.execution_begun || active.fully_completed())
            return false;
          const auto a = entry.scratch_backing_addr, b = active.scratch_backing_addr;
          return a <= b ? b - a < scratch_bytes(entry) : a - b < scratch_bytes(active);
        });
      });
      if (busy)
        continue;
    }
    try {
      const auto result = dispatch_workgroups(entry);
      if (result.outcome != VmAccessOutcome::Complete) {
        entry.pm4_failure->fail();
        fail_pm4_queue(queue, state);
      } else if (entry.fully_completed() && engine()) {
        engine()->schedule_event_now(doorbell_event());
      }
    } catch (const std::exception &error) {
      util::Logger::warn("PM4 launch failed: ", error.what());
      entry.pm4_failure->fail();
      fail_pm4_queue(queue, state);
    }
  }
}

bool CommandProcessor::submit_pm4(uint32_t queue_id, uint32_t process_id,
                                  Pm4Submission submission) {
  {
    std::lock_guard lock(hw_queue_mutex_);
    auto it = std::ranges::find_if(compute_queues_, [&](const ComputeQueueRecord &queue) {
      return queue.queue_id == queue_id && queue.process_id == process_id;
    });
    if (it == compute_queues_.end() || !it->submission_queue)
      throw std::runtime_error("PM4 submission has no registered queue");
    if (it->faulted)
      return false;
    submission.failure->wake = [this] {
      if (engine())
        engine()->schedule_event_now(doorbell_event());
    };
    it->commands.submissions.push_back(std::move(submission));
    if (!is_primary_ && engine() && !has_kfd_queues()) {
      engine()->register_as_primary();
      is_primary_ = true;
    }
  }
  if (engine())
    engine()->schedule_event_now(doorbell_event());
  return true;
}

void CommandProcessor::dispatch_pm4(const ComputeQueueRecord &queue, Pm4DispatchState &qs,
                                    const std::array<uint32_t, 4> &dimensions) {
  if (!queue.commands.submissions.front().allow_dispatch)
    throw std::runtime_error("shader dispatch inside an AQL PM4 IB is unsupported");
  using namespace rocr::llvm::amdhsa;
  const auto &regs = queue.commands.sh_registers;
  const uint32_t initiator = dimensions[3];
  if (!(initiator & 1))
    return;
  if (cus_.empty())
    throw std::runtime_error("PM4 dispatch requires a compute unit");
  const bool thread_dimensions = initiator & (1u << 5);
  const auto arch = cus_[0]->config().arch;
  const uint32_t rsrc1 = regs[kPm4ComputePgmRsrc1], rsrc2 = regs[kPm4ComputePgmRsrc2];
  DispatchEntry dp;
  dp.kind = DispatchPacketKind::Kernel;
  dp.dispatch_id = allocate_dispatch_id();
  dp.address_space = queue.address_space;
  dp.process_id = queue.process_id;
  dp.queue_id = queue.queue_id;
  dp.interrupt_sink = queue.interrupt_sink;
  dp.enabled_cus = queue.enabled_cus;
  dp.kernel_wave_size = (initiator & (1u << 15)) ? 32 : 64;
  if (AMDHSA_BITS_GET(rsrc2, COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT)) {
    dp.pm4_scratch_waves_per_se = regs[kPm4ComputeTmpringSize] & 0xfff;
    const auto properties = isa_properties(arch);
    const uint32_t wave_bytes = ((regs[kPm4ComputeTmpringSize] >> 12) &
                                 util::mask<uint32_t>(properties.compute_tmpring_wavesize_bits)) *
                                properties.compute_tmpring_wavesize_granule;
    if (!dp.pm4_scratch_waves_per_se || !wave_bytes)
      throw std::runtime_error("PM4 scratch enabled with an empty descriptor");
    dp.private_segment_fixed_size = wave_bytes / dp.kernel_wave_size;
    dp.pm4_scratch_pool = std::make_shared<Pm4ScratchPool>(
        dp.pm4_scratch_waves_per_se *
        std::max(scratch_wave_divisor_, scratch_shader_engine_count_));
    uint64_t scratch = ((uint64_t{regs[kPm4ComputeScratchHi]} << 32) | regs[kPm4ComputeScratchLo])
                       << 8;
    dp.scratch_backing_addr = static_cast<uint64_t>(static_cast<int64_t>(scratch << 16) >> 16);
  }
  // Program addresses have 256-byte granularity and are sign-extended from 48 bits.
  uint64_t pc = ((uint64_t{regs[kPm4ComputePgmHi]} << 32) | regs[kPm4ComputePgmLo]) << 8;
  dp.kernel_entry_pc = static_cast<uint64_t>(static_cast<int64_t>(pc << 16) >> 16);
  dp.num_user_sgprs = AMDHSA_BITS_GET(rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT);
  if (dp.num_user_sgprs > dp.user_sgprs.size())
    throw std::runtime_error("PM4 launch exceeds compute user-data registers");
  dp.pm4_abi = true;
  dp.pm4_failure = queue.commands.submissions.front().failure;
  std::ranges::copy_n(regs.begin() + kPm4ComputeUserData0, dp.num_user_sgprs,
                      dp.user_sgprs.begin());
  dp.sgprs_per_wf = cus_[0]->config().sgprs_per_wf;
  const auto granule = descriptor_vgpr_count_granule_for_wavefront(arch, dp.kernel_wave_size);
  if (!granule)
    throw std::runtime_error("unsupported PM4 wave size");
  dp.vgprs_per_wf =
      (AMDHSA_BITS_GET(rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT) + 1) * *granule;
  if (dp.vgprs_per_wf > cus_[0]->vgpr_allocation_block_size())
    throw std::runtime_error("PM4 launch exceeds available VGPRs");
  dp.initial_mode_raw = initial_mode_from_compute_pgm_rsrc1(rsrc1, arch);
  dp.enable_wg_id_x = AMDHSA_BITS_GET(rsrc2, COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X);
  dp.enable_wg_id_y = AMDHSA_BITS_GET(rsrc2, COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y);
  dp.enable_wg_id_z = AMDHSA_BITS_GET(rsrc2, COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z);
  dp.enable_wg_info = AMDHSA_BITS_GET(rsrc2, COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO);
  dp.enable_vgpr_workitem_id = AMDHSA_BITS_GET(rsrc2, COMPUTE_PGM_RSRC2_ENABLE_VGPR_WORKITEM_ID);
  dp.wgp_mode = AMDHSA_BITS_GET(rsrc1, COMPUTE_PGM_RSRC1_WGP_MODE);
  dp.group_segment_fixed_size = AMDHSA_BITS_GET(rsrc2, COMPUTE_PGM_RSRC2_GRANULATED_LDS_SIZE) * 512;
  for (uint32_t i = 0; i < 3; ++i)
    if (!(regs[kPm4ComputeNumThreadX + i] & 0xffff) ||
        (regs[kPm4ComputeNumThreadX + i] & 0xffff) > 1024)
      throw std::runtime_error("invalid PM4 workgroup dimension");
  dp.workgroup_size_x = regs[kPm4ComputeNumThreadX] & 0xffff;
  dp.workgroup_size_y = regs[kPm4ComputeNumThreadY] & 0xffff;
  dp.workgroup_size_z = regs[kPm4ComputeNumThreadZ] & 0xffff;
  uint64_t threads = uint64_t{dp.workgroup_size_x} * dp.workgroup_size_y * dp.workgroup_size_z;
  if (threads > 1024)
    throw std::runtime_error("PM4 workgroup exceeds 1024 threads");
  dp.wfs_per_workgroup = (threads + dp.kernel_wave_size - 1) / dp.kernel_wave_size;
  std::array<uint32_t, 3> counts{};
  for (uint32_t i = 0; i < 3; ++i) {
    dp.workgroup_origin[i] = (initiator & (1u << 2)) ? 0 : regs[kPm4ComputeStartX + i];
    if (thread_dimensions && dp.workgroup_origin[i])
      throw std::runtime_error("thread-dimension PM4 dispatch requires a zero origin");
    const uint32_t full = regs[kPm4ComputeNumThreadX + i] & 0xffff;
    const uint32_t end =
        thread_dimensions ? dimensions[i] / full + (dimensions[i] % full != 0) : dimensions[i];
    counts[i] = end > dp.workgroup_origin[i] ? end - dp.workgroup_origin[i] : 0;
  }
  dp.grid_wgs_x = counts[0];
  dp.grid_wgs_y = counts[1];
  dp.grid_wgs_z = counts[2];
  dp.grid_yz_valid = true;
  uint64_t total = counts[0];
  for (uint32_t i = 1; i < 3; ++i) {
    if (counts[i] && total > UINT32_MAX / counts[i])
      throw std::runtime_error("PM4 grid exceeds supported workgroup count");
    total *= counts[i];
  }
  dp.total_wgs = total;
  if (dp.pm4_scratch_pool && dp.total_wgs) {
    if (!dp.pm4_scratch_pool->available(dp.wfs_per_workgroup))
      throw std::runtime_error("PM4 scratch cannot accommodate one workgroup");
    const uint64_t slots = uint64_t{dp.pm4_scratch_waves_per_se} *
                           std::max(scratch_wave_divisor_, scratch_shader_engine_count_);
    const uint64_t bytes = slots * dp.private_segment_fixed_size * dp.kernel_wave_size;
    const auto access = snapshot_gpu_access(queue.address_space);
    if (!access || access->query_access(dp.scratch_backing_addr, bytes, VmAccessKind::Atomic) !=
                       VmAccessOutcome::Complete)
      throw std::runtime_error("PM4 scratch descriptor exceeds its mapped buffer");
  }
  if (!thread_dimensions && ((dp.grid_wgs_x && dp.workgroup_size_x > UINT32_MAX / dp.grid_wgs_x) ||
                             (dp.grid_wgs_y && dp.workgroup_size_y > UINT32_MAX / dp.grid_wgs_y) ||
                             (dp.grid_wgs_z && dp.workgroup_size_z > UINT32_MAX / dp.grid_wgs_z)))
    throw std::runtime_error("PM4 grid exceeds supported invocation count");
  dp.grid_size_x = thread_dimensions ? dimensions[0] : dp.grid_wgs_x * dp.workgroup_size_x;
  dp.grid_size_y = thread_dimensions ? dimensions[1] : dp.grid_wgs_y * dp.workgroup_size_y;
  dp.grid_size_z = thread_dimensions ? dimensions[2] : dp.grid_wgs_z * dp.workgroup_size_z;
  if (!thread_dimensions && (initiator & 2)) {
    uint32_t *sizes[] = {&dp.grid_size_x, &dp.grid_size_y, &dp.grid_size_z};
    for (uint32_t i = 0; i < 3; ++i) {
      uint32_t partial = regs[kPm4ComputeNumThreadX + i] >> 16;
      uint32_t full = regs[kPm4ComputeNumThreadX + i] & 0xffff;
      if (partial > full)
        throw std::runtime_error("invalid PM4 partial workgroup size");
      if (partial && counts[i])
        *sizes[i] -= full - partial;
    }
  }
  dp.wait_for_predecessors = true;
  util::Logger::vm("PM4 dispatch pc=", std::hex, dp.kernel_entry_pc, " rsrc1=", rsrc1,
                   " rsrc2=", rsrc2, std::dec, " workgroups=", total);
  flush_gpu_caches();
  KernelDispatchInfo info{};
  info.dispatch_id = dp.dispatch_id;
  info.entry_pc = dp.kernel_entry_pc;
  info.kernel_name = "PM4 compute";
  info.code_target = cus_[0]->config().target;
  info.lds_size_bytes = dp.group_segment_fixed_size;
  info.wave_size = dp.kernel_wave_size;
  info.grid_size_x = dp.grid_size_x;
  info.grid_size_y = dp.grid_size_y;
  info.grid_size_z = dp.grid_size_z;
  info.workgroup_size_x = dp.workgroup_size_x;
  info.workgroup_size_y = dp.workgroup_size_y;
  info.workgroup_size_z = dp.workgroup_size_z;
  info.workgroup_count = dp.total_wgs;
  info.wfs_per_workgroup = dp.wfs_per_workgroup;
  info.sgprs_per_wf = dp.sgprs_per_wf;
  info.vgprs_per_wf = dp.vgprs_per_wf;
  plugin_group_->onAmdgpuDispatchPacketProcessed(info);
  ++total_dispatched_;
  qs.push_entry(std::move(dp));
}

void CommandProcessor::fail_pm4_queue(ComputeQueueRecord &queue, Pm4DispatchState &qs) {
  queue.faulted = true;
  for (auto &entry : queue.entries)
    if (entry.grid_completion)
      entry.grid_completion->mark_faulted();
  // The CP owns the queue lock and CU workers have rejoined. Stop all resident
  // waves before releasing the submission's BO references or publishing failure.
  for (auto *cu : cus_) {
    cu->with_wave_state_locked([&] {
      for (uint32_t slot = 0; slot < cu->num_wf_slots(); ++slot) {
        auto *wave = cu->wf(slot);
        if (wave && !wave->is_halted() && wave->process_id() == queue.process_id &&
            wave->queue_id() == queue.queue_id)
          wave->halt();
      }
    });
  }
  flush_gpu_caches();
  qs.entries.clear();
  for (auto &submission : queue.commands.submissions)
    if (submission.complete)
      submission.complete(false);
  queue.commands.submissions.clear();
  queue.command_access.reset();
}

void CommandProcessor::fetch_pm4(ComputeQueueRecord &queue, Pm4DispatchState &qs,
                                 simdojo::Tick now) {
  process_pm4_packets(
      queue, gpu_vm_,
      {
          .arch = cus_.empty() ? ROCJITSU_CODE_ARCH_INVALID : cus_[0]->config().arch,
          .xcc_id = scratch_xcc_id_,
          .flush_caches = [this] { flush_gpu_caches(); },
          .dispatch =
              [this, &queue, &qs](const std::array<uint32_t, 4> &dimensions) {
                dispatch_pm4(queue, qs, dimensions);
              },
          .retry =
              [this, &queue, now] {
                queue.command_retry_pending = true;
                arm_stall_recheck(now);
              },
          .wake =
              [this, now] {
                stall_recheck_backoff_ = 1;
                arm_stall_recheck(now);
              },
          .fault_queue = [this, &queue, &qs] { fail_pm4_queue(queue, qs); },
      });
}

CommandProcessor::KernelDescriptorReadResult
CommandProcessor::read_kernel_descriptor(const GpuVmAccess &transaction_access,
                                         uint64_t kernel_object) const {
  using namespace rocr::llvm::amdhsa;
  kernel_descriptor_t kd{};
  const VmAccessOutcome outcome =
      read_gpu_block(transaction_access, kernel_object, &kd, sizeof(kd));
  return {.outcome = outcome, .descriptor = kd};
}

VmAccessOutcome
CommandProcessor::publish_async_scratch_capability(const ComputeQueueRecord &queue) const {
  if (!gpu_vm_ || !queue.address_space)
    return VmAccessOutcome::Unavailable;
  std::optional<GpuVmAccess> access = snapshot_gpu_access(queue.address_space);
  if (!access)
    return VmAccessOutcome::Unavailable;

  const uint64_t caps_address = queue.queue_desc_va + offsetof(amd_queue_v2_t, caps);
  AtomicLoadResult loaded = access->atomic_load(caps_address, sizeof(uint32_t));
  if (loaded.outcome != VmAccessOutcome::Complete)
    return loaded.outcome;

  for (;;) {
    const uint32_t observed = static_cast<uint32_t>(loaded.value);
    if ((observed & AMD_QUEUE_CAPS_CP_ASYNC_RECLAIM) != 0)
      return VmAccessOutcome::Complete;
    const AtomicCompareExchangeResult exchanged = access->compare_exchange(
        caps_address, sizeof(uint32_t), observed, observed | AMD_QUEUE_CAPS_CP_ASYNC_RECLAIM);
    if (exchanged.outcome != VmAccessOutcome::Complete)
      return exchanged.outcome;
    if (exchanged.exchanged)
      return VmAccessOutcome::Complete;
    loaded.value = exchanged.observed;
  }
}

VmAccessOutcome CommandProcessor::record_async_scratch_use(const ComputeQueueRecord &queue,
                                                           const GpuVmAccess &access,
                                                           uint64_t packet_index,
                                                           bool alternate) const {
  const uint64_t first_index = queue.xcd_fanout ? 0 : scratch_xcc_id_;
  const uint64_t end_index = queue.xcd_fanout ? scratch_xcc_count_ : scratch_xcc_id_ + 1;
  if (end_index > MAX_NUM_XCC)
    return VmAccessOutcome::Malformed;

  for (uint64_t xcc = first_index; xcc < end_index; ++xcc) {
    const uint64_t index_address = queue.queue_desc_va +
                                   offsetof(amd_queue_v2_t, scratch_last_used_index) +
                                   xcc * sizeof(scratch_last_used_index_xcc_t) +
                                   (alternate ? offsetof(scratch_last_used_index_xcc_t, alt)
                                              : offsetof(scratch_last_used_index_xcc_t, main));
    const VmAccessOutcome outcome =
        access.atomic_store(index_address, sizeof(uint64_t), packet_index);
    if (outcome != VmAccessOutcome::Complete)
      return outcome;
  }
  return VmAccessOutcome::Complete;
}

AqlAdmissionResult CommandProcessor::request_dynamic_scratch(ComputeQueueRecord &queue,
                                                             const GpuVmAccess &transaction_access,
                                                             uint64_t packet_index,
                                                             uint64_t status) {
  constexpr uint64_t kQueueInactiveSignalOffset = offsetof(amd_queue_t, queue_inactive_signal);
  constexpr uint32_t kSignalValueOffset = 8;
  constexpr uint32_t kMailboxPointerOffset = 16;
  constexpr uint32_t kEventIdOffset = 24;

  QueueScratchRequestState &state = queue.scratch_request;
  if (!state.active()) {
    if (queue.queue_desc_va == 0)
      return {.status = AqlAdmissionStatus::Faulted};
    state.phase = QueueScratchRequestPhase::PublishReadPointer;
    state.access = std::make_shared<GpuVmAccess>(transaction_access);
    state.packet_index = packet_index;
    state.status = status;
    util::Logger::cp([&](auto &os) {
      os << std::format("DYNAMIC_SCRATCH_REQUEST q={} packet={} status={:#x}", queue.queue_id,
                        packet_index, status);
    });
  } else if (state.packet_index != packet_index || state.status != status) {
    return {.status = AqlAdmissionStatus::Malformed};
  }

  for (;;) {
    switch (state.phase) {
    case QueueScratchRequestPhase::Inactive:
      return {.status = AqlAdmissionStatus::Malformed};

    case QueueScratchRequestPhase::PublishReadPointer: {
      // ROCr scans from read_dispatch_id to locate the scratch-needing packet.
      // Publish every earlier admission before raising the event, while leaving
      // this packet itself unconsumed at packet_index.
      if (!queue.read_pointer_journal.publication_pending()) {
        queue.read_pointer_journal.retire(state.packet_index, state.packet_index, *state.access);
      }
      const VmAccessOutcome outcome = queue.read_pointer_journal.publish();
      if (outcome != VmAccessOutcome::Complete)
        return admission_from_vm_outcome(outcome);
      state.phase = QueueScratchRequestPhase::ReadSignalHandle;
      continue;
    }

    case QueueScratchRequestPhase::ReadSignalHandle: {
      const VmAccessOutcome outcome =
          read_gpu_block(*state.access, queue.queue_desc_va + kQueueInactiveSignalOffset,
                         &state.signal_address, sizeof(state.signal_address));
      if (outcome != VmAccessOutcome::Complete)
        return admission_from_vm_outcome(outcome);
      if (state.signal_address == 0 || (state.signal_address & 0x3f) != 0)
        return {.status = AqlAdmissionStatus::Faulted};
      state.phase = QueueScratchRequestPhase::StoreStatus;
      continue;
    }

    case QueueScratchRequestPhase::StoreStatus: {
      const AtomicCompareExchangeResult stored = state.access->compare_exchange(
          state.signal_address + kSignalValueOffset, sizeof(uint64_t), 0, state.status);
      if (stored.outcome != VmAccessOutcome::Complete)
        return admission_from_vm_outcome(stored.outcome);
      if (!stored.exchanged && stored.observed != state.status)
        return {.status = AqlAdmissionStatus::Blocked};
      state.phase = QueueScratchRequestPhase::ReadMailboxPointer;
      continue;
    }

    case QueueScratchRequestPhase::ReadMailboxPointer: {
      const VmAccessOutcome outcome =
          read_gpu_block(*state.access, state.signal_address + kMailboxPointerOffset,
                         &state.mailbox_pointer, sizeof(state.mailbox_pointer));
      if (outcome != VmAccessOutcome::Complete)
        return admission_from_vm_outcome(outcome);
      state.phase = QueueScratchRequestPhase::ReadEventId;
      continue;
    }

    case QueueScratchRequestPhase::ReadEventId: {
      const VmAccessOutcome outcome =
          read_gpu_block(*state.access, state.signal_address + kEventIdOffset, &state.event_id,
                         sizeof(state.event_id));
      if (outcome != VmAccessOutcome::Complete)
        return admission_from_vm_outcome(outcome);
      state.phase = QueueScratchRequestPhase::StoreMailbox;
      continue;
    }

    case QueueScratchRequestPhase::StoreMailbox:
      if (state.mailbox_pointer != 0) {
        const VmAccessOutcome outcome = state.access->atomic_store(
            state.mailbox_pointer, sizeof(uint64_t), uint64_t(state.event_id));
        if (outcome != VmAccessOutcome::Complete)
          return admission_from_vm_outcome(outcome);
      }
      state.phase = QueueScratchRequestPhase::DeliverInterrupt;
      continue;

    case QueueScratchRequestPhase::DeliverInterrupt:
      state.phase = QueueScratchRequestPhase::WaitingForRuntime;
      util::Logger::cp([&](auto &os) {
        os << std::format("DYNAMIC_SCRATCH_INTERRUPT q={} packet={} status={:#x}", queue.queue_id,
                          state.packet_index, state.status);
      });
      queue.interrupt_sink.deliver(queue.process_id, state.event_id);
      return {.status = AqlAdmissionStatus::Blocked};

    case QueueScratchRequestPhase::WaitingForRuntime:
      return {.status = AqlAdmissionStatus::Blocked};
    }
  }
}

VmAccessOutcome CommandProcessor::advance_scratch_reclaim(ComputeQueueRecord &queue,
                                                          const DispatchEntry &entry) {
  constexpr uint64_t kQueueInactiveSignalOffset = offsetof(amd_queue_t, queue_inactive_signal);
  constexpr uint64_t kQueuePropertiesOffset = offsetof(amd_queue_t, queue_properties);
  constexpr uint64_t kScratchBackingOffset = offsetof(amd_queue_t, scratch_backing_memory_location);
  constexpr uint64_t kComputeTmpringSizeOffset = offsetof(amd_queue_t, compute_tmpring_size);
  constexpr uint32_t kSignalValueOffset = 8;
  constexpr uint32_t kMailboxPointerOffset = 16;
  constexpr uint32_t kEventIdOffset = 24;
  constexpr uint64_t kLargeScratchReclaim = 0x200;

  QueueScratchReclaimState &state = queue.scratch_reclaim;
  if (!state.active())
    state.begin(entry.dispatch_id);
  if (state.dispatch_id != entry.dispatch_id)
    return VmAccessOutcome::Malformed;

  for (;;) {
    switch (state.phase) {
    case QueueScratchReclaimPhase::Inactive:
      return VmAccessOutcome::Malformed;

    case QueueScratchReclaimPhase::WaitingForDispatch:
      util::Logger::cp([&](auto &os) {
        os << std::format("DYNAMIC_SCRATCH_RECLAIM q={} dispatch={}", queue.queue_id,
                          entry.dispatch_id);
      });
      state.phase = QueueScratchReclaimPhase::CaptureAccess;
      continue;

    case QueueScratchReclaimPhase::CaptureAccess:
      if (queue.queue_desc_va == 0 || !entry.address_space)
        return VmAccessOutcome::Faulted;
      if (std::optional<GpuVmAccess> access = snapshot_gpu_access(entry.address_space))
        state.access = std::make_shared<GpuVmAccess>(std::move(*access));
      else
        return VmAccessOutcome::Faulted;
      state.phase = QueueScratchReclaimPhase::ReadSignalHandle;
      continue;

    case QueueScratchReclaimPhase::ReadSignalHandle: {
      const VmAccessOutcome outcome =
          read_gpu_block(*state.access, queue.queue_desc_va + kQueueInactiveSignalOffset,
                         &state.signal_address, sizeof(state.signal_address));
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
      if (state.signal_address == 0 || (state.signal_address & 0x3f) != 0)
        return VmAccessOutcome::Faulted;
      state.phase = QueueScratchReclaimPhase::StoreStatus;
      continue;
    }

    case QueueScratchReclaimPhase::StoreStatus: {
      const AtomicCompareExchangeResult stored = state.access->compare_exchange(
          state.signal_address + kSignalValueOffset, sizeof(uint64_t), 0, kLargeScratchReclaim);
      if (stored.outcome != VmAccessOutcome::Complete)
        return stored.outcome;
      if (!stored.exchanged && stored.observed != kLargeScratchReclaim)
        return VmAccessOutcome::Unavailable;
      state.phase = QueueScratchReclaimPhase::ReadMailboxPointer;
      continue;
    }

    case QueueScratchReclaimPhase::ReadMailboxPointer: {
      const VmAccessOutcome outcome =
          read_gpu_block(*state.access, state.signal_address + kMailboxPointerOffset,
                         &state.mailbox_pointer, sizeof(state.mailbox_pointer));
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
      state.phase = QueueScratchReclaimPhase::ReadEventId;
      continue;
    }

    case QueueScratchReclaimPhase::ReadEventId: {
      const VmAccessOutcome outcome =
          read_gpu_block(*state.access, state.signal_address + kEventIdOffset, &state.event_id,
                         sizeof(state.event_id));
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
      state.phase = QueueScratchReclaimPhase::StoreMailbox;
      continue;
    }

    case QueueScratchReclaimPhase::StoreMailbox:
      if (state.mailbox_pointer != 0) {
        const VmAccessOutcome outcome = state.access->atomic_store(
            state.mailbox_pointer, sizeof(uint64_t), uint64_t(state.event_id));
        if (outcome != VmAccessOutcome::Complete)
          return outcome;
      }
      state.phase = QueueScratchReclaimPhase::DeliverInterrupt;
      continue;

    case QueueScratchReclaimPhase::DeliverInterrupt:
      state.phase = QueueScratchReclaimPhase::WaitingForRuntime;
      queue.interrupt_sink.deliver(queue.process_id, state.event_id);
      // Re-enter through the normal paced retry path even when an in-process
      // test handler acknowledges synchronously. In a guest, ROCr runs on a
      // different CPU after this interrupt and cannot have completed yet.
      return VmAccessOutcome::Unavailable;

    case QueueScratchReclaimPhase::WaitingForRuntime: {
      const AtomicLoadResult signal_status =
          read_gpu_u64(*state.access, state.signal_address + kSignalValueOffset);
      if (signal_status.outcome != VmAccessOutcome::Complete)
        return signal_status.outcome;

      uint32_t queue_properties = 0;
      VmAccessOutcome outcome =
          read_gpu_block(*state.access, queue.queue_desc_va + kQueuePropertiesOffset,
                         &queue_properties, sizeof(queue_properties));
      if (outcome != VmAccessOutcome::Complete)
        return outcome;

      const AtomicLoadResult scratch_backing =
          read_gpu_u64(*state.access, queue.queue_desc_va + kScratchBackingOffset);
      if (scratch_backing.outcome != VmAccessOutcome::Complete)
        return scratch_backing.outcome;

      uint32_t compute_tmpring_size = 0;
      outcome = read_gpu_block(*state.access, queue.queue_desc_va + kComputeTmpringSizeOffset,
                               &compute_tmpring_size, sizeof(compute_tmpring_size));
      if (outcome != VmAccessOutcome::Complete)
        return outcome;

      if (signal_status.value != 0 ||
          (queue_properties & AMD_QUEUE_PROPERTIES_USE_SCRATCH_ONCE) != 0 ||
          scratch_backing.value != 0 || compute_tmpring_size != 0) {
        return VmAccessOutcome::Unavailable;
      }

      util::Logger::cp([&](auto &os) {
        os << std::format("DYNAMIC_SCRATCH_RECLAIMED q={} dispatch={}", queue.queue_id,
                          entry.dispatch_id);
      });
      state.reset();
      return VmAccessOutcome::Complete;
    }
    }
  }
}

VmAccessOutcome CommandProcessor::gate_dispatch_retirement(ComputeQueueRecord &queue,
                                                           const DispatchEntry &entry) {
  if (entry.scratch_use_once && !entry.fanout_peer)
    return advance_scratch_reclaim(queue, entry);
  return VmAccessOutcome::Complete;
}

/// Scan backward from ptr to find the ELF header (\x7fELF) at a page boundary.
/// Both ptr and limit must be readable host memory.
static const uint8_t *find_elf_base(const uint8_t *ptr, const uint8_t *limit) {
  auto *page = reinterpret_cast<const uint8_t *>(reinterpret_cast<uintptr_t>(ptr) & ~0xFFFULL);
  for (; page >= limit; page -= 0x1000) {
    if (page[0] == 0x7f && page[1] == 'E' && page[2] == 'L' && page[3] == 'F')
      return page;
  }
  return nullptr;
}

AqlAdmissionResult CommandProcessor::admit_kernel_dispatch(
    const hsa_kernel_dispatch_packet_t &pkt, ComputeQueueRecord &queue,
    const GpuVmAccess &transaction_access, uint64_t pkt_addr, uint32_t queue_packet_id,
    uint64_t aql_packet_id, ClusterDispatchShape cluster_shape,
    std::span<const uint32_t> metadata) {
  const bool uses_kfd_queue_abi = queue.uses_kfd_queue_abi;
  using namespace rocr::llvm::amdhsa;
  kernel_descriptor_t kd{};
  if (!metadata.empty()) {
    const std::span<const std::byte> descriptor = aql_metadata::kernel_descriptor(metadata);
    std::memcpy(reinterpret_cast<std::byte *>(&kd) + aql_metadata::kKernelDescriptorOffset,
                descriptor.data(), descriptor.size());
    kd.group_segment_fixed_size = pkt.group_segment_size;
    kd.private_segment_fixed_size = pkt.private_segment_size;
    if (AMDHSA_BITS_GET(kd.kernarg_preload, KERNARG_PRELOAD_SPEC_LENGTH) >
        aql_metadata::kPreloadWords)
      return {.status = AqlAdmissionStatus::Malformed};
  } else {
    const KernelDescriptorReadResult descriptor =
        read_kernel_descriptor(transaction_access, pkt.kernel_object);
    if (descriptor.outcome != VmAccessOutcome::Complete)
      return admission_from_vm_outcome(descriptor.outcome);
    kd = descriptor.descriptor;
  }
  uint32_t vgpr_gran =
      AMDHSA_BITS_GET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  uint32_t sgpr_gran =
      AMDHSA_BITS_GET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);
  rj_code_arch_t arch = cus_.empty() ? ROCJITSU_CODE_ARCH_CDNA1 : cus_[0]->config().arch;
  const uint32_t wave_size = kernel_wavefront_size(arch, kd);
  const auto vgpr_granularity = descriptor_vgpr_count_granule_for_wavefront(arch, wave_size);
  if (!vgpr_granularity)
    return {.status = AqlAdmissionStatus::Unsupported,
            .diagnostic = AqlPacketDiagnostic::UnsupportedKernelWaveSize};
  uint32_t vgprs = (vgpr_gran + 1) * *vgpr_granularity;
  uint32_t sgprs = sgpr_count_is_descriptor_encoded(arch, sgpr_gran) ? (sgpr_gran + 1) * 8 : 0;
  uint32_t user_sgprs = kernel_descriptor_user_sgpr_count(arch, kd);
  uint64_t entry_pc = pkt.kernel_object + static_cast<uint64_t>(kd.kernel_code_entry_byte_offset);
  uint64_t code_load_bias = 0;

  uint32_t wg_size =
      static_cast<uint32_t>(pkt.workgroup_size_x) * pkt.workgroup_size_y * pkt.workgroup_size_z;
  uint32_t wfs_per_wg = (wg_size + wave_size - 1) / wave_size;

  uint32_t num_dims = pkt.setup & 0x3;
  uint32_t grid_wgs_x =
      util::ceil_div_or_one(pkt.grid_size_x, static_cast<uint32_t>(pkt.workgroup_size_x));
  uint32_t grid_wgs_y =
      util::ceil_div_or_one(pkt.grid_size_y, static_cast<uint32_t>(pkt.workgroup_size_y));
  uint32_t grid_wgs_z =
      util::ceil_div_or_one(pkt.grid_size_z, static_cast<uint32_t>(pkt.workgroup_size_z));
  uint32_t total_wgs = grid_wgs_x * grid_wgs_y * grid_wgs_z;

  DispatchLaunchMetadata launch_metadata{};
  uint64_t queue_ptr = 0;
  uint64_t scratch_backing_addr = 0;
  uint32_t scratch_wave_limit_per_se = std::numeric_limits<uint32_t>::max();
  uint32_t scratch_wave_stride_per_se = 0;
  uint32_t scratch_wave_size_bytes = 0;
  bool scratch_use_once = false;
  bool scratch_uses_alternate = false;
  const uint32_t private_segment_fixed_size =
      std::max(kd.private_segment_fixed_size, pkt.private_segment_size);
  if (uses_kfd_queue_abi) {
    queue_ptr = queue.read_ptr_va - offsetof(amd_queue_t, read_dispatch_id);
    if (AMDHSA_BITS_GET(kd.kernel_code_properties,
                        KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER)) {
      const uint64_t descriptor_va = queue_ptr + offsetof(amd_queue_t, scratch_resource_descriptor);
      const VmAccessOutcome outcome = read_gpu_block(
          transaction_access, descriptor_va, launch_metadata.scratch_resource_descriptor.data(),
          sizeof(launch_metadata.scratch_resource_descriptor));
      if (outcome != VmAccessOutcome::Complete)
        return admission_from_vm_outcome(outcome);
    }
    if (AMDHSA_BITS_GET(kd.kernel_code_properties, KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID)) {
      const uint64_t dispatch_id_va = queue_ptr + offsetof(amd_queue_t, write_dispatch_id);
      const AtomicLoadResult loaded = read_gpu_u64(transaction_access, dispatch_id_va);
      if (loaded.outcome != VmAccessOutcome::Complete)
        return admission_from_vm_outcome(loaded.outcome);
      launch_metadata.write_dispatch_id = loaded.value;
    }

    if (private_segment_fixed_size > 0) {
      uint32_t queue_caps = 0;
      VmAccessOutcome outcome =
          read_gpu_block(transaction_access, queue_ptr + offsetof(amd_queue_v2_t, caps),
                         &queue_caps, sizeof(queue_caps));
      if (outcome != VmAccessOutcome::Complete)
        return admission_from_vm_outcome(outcome);
      const bool async_scratch =
          (queue_caps & (AMD_QUEUE_CAPS_CP_ASYNC_RECLAIM | AMD_QUEUE_CAPS_SW_ASYNC_RECLAIM)) ==
          (AMD_QUEUE_CAPS_CP_ASYNC_RECLAIM | AMD_QUEUE_CAPS_SW_ASYNC_RECLAIM);

      const uint64_t scratch_loc_va =
          queue_ptr + offsetof(amd_queue_t, scratch_backing_memory_location);
      const AtomicLoadResult loaded = read_gpu_u64(transaction_access, scratch_loc_va);
      if (loaded.outcome != VmAccessOutcome::Complete)
        return admission_from_vm_outcome(loaded.outcome);
      uint32_t compute_tmpring_size = 0;
      outcome = read_gpu_block(transaction_access,
                               queue_ptr + offsetof(amd_queue_t, compute_tmpring_size),
                               &compute_tmpring_size, sizeof(compute_tmpring_size));
      if (outcome != VmAccessOutcome::Complete)
        return admission_from_vm_outcome(outcome);
      if (queue.simulator_scratch &&
          (loaded.value != queue.simulator_scratch->backing_address ||
           compute_tmpring_size != queue.simulator_scratch->tmpring_size))
        queue.simulator_scratch.reset();
      const bool simulator_scratch = loaded.value == 0 || queue.simulator_scratch.has_value();
      scratch_backing_addr = loaded.value;
      if (scratch_backing_addr == 0 && scratch_resolver_)
        scratch_backing_addr = scratch_resolver_(queue.process_id);
      const auto properties = isa_properties(arch);
      const uint32_t wavesize_mask = util::mask<uint32_t>(properties.compute_tmpring_wavesize_bits);
      const uint64_t raw_per_wave = static_cast<uint64_t>(private_segment_fixed_size) * wave_size;
      const uint64_t granule = properties.compute_tmpring_wavesize_granule;
      const uint64_t per_wave_stride = ((raw_per_wave + granule - 1) / granule) * granule;
      const uint64_t required_wavesize =
          per_wave_stride / properties.compute_tmpring_wavesize_granule;
      const auto scratch_wave_limit = [&](uint32_t tmpring_size) -> std::optional<uint32_t> {
        const uint32_t provisioned_waves = tmpring_size & 0xFFFu;
        const uint32_t provisioned_wavesize = (tmpring_size >> 12) & wavesize_mask;
        if (provisioned_waves == 0 || provisioned_wavesize < required_wavesize)
          return std::nullopt;
        if (arch == ROCJITSU_CODE_ARCH_CDNA5) {
          // A workgroup stays on one CU; insufficient scratch is a runtime
          // replacement request, not a fault after consuming the packet.
          if (provisioned_waves < wfs_per_wg)
            return std::nullopt;
          return provisioned_waves;
        }
        // ROCr FillComputeTmpRingSize publishes waves per XCC;
        // Gfx11/Gfx12 publish waves per shader engine. Older AQL execution
        // uses logical grid slots, so admit only grids fitting that backing.
        const bool waves_per_se = arch == ROCJITSU_CODE_ARCH_RDNA3 ||
                                  arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
                                  arch == ROCJITSU_CODE_ARCH_RDNA4;
        const uint64_t capacity =
            uint64_t{provisioned_waves} * scratch_xcc_count_ *
            (waves_per_se ? std::max(scratch_wave_divisor_, scratch_shader_engine_count_) : 1u);
        if (uint64_t{total_wgs} * wfs_per_wg > capacity)
          return std::nullopt;
        return std::numeric_limits<uint32_t>::max();
      };
      bool main_scratch_usable = !simulator_scratch && scratch_backing_addr != 0;
      bool requires_dynamic_scratch = !main_scratch_usable;
      if (!requires_dynamic_scratch) {
        const std::optional<uint32_t> wave_limit = scratch_wave_limit(compute_tmpring_size);
        main_scratch_usable = wave_limit.has_value();
        if (main_scratch_usable) {
          scratch_wave_limit_per_se = *wave_limit;
          scratch_wave_stride_per_se = *wave_limit;
          scratch_wave_size_bytes = ((compute_tmpring_size >> 12) & wavesize_mask) *
                                    properties.compute_tmpring_wavesize_granule;
        }
        requires_dynamic_scratch = !main_scratch_usable;
      }

      if (async_scratch && main_scratch_usable) {
        const AtomicLoadResult max_use = read_gpu_u64(
            transaction_access, queue_ptr + offsetof(amd_queue_v2_t, scratch_max_use_index));
        if (max_use.outcome != VmAccessOutcome::Complete)
          return admission_from_vm_outcome(max_use.outcome);
        main_scratch_usable = aql_packet_id <= max_use.value;
        requires_dynamic_scratch = !main_scratch_usable;
      }

      if (async_scratch && requires_dynamic_scratch) {
        const AtomicLoadResult alt_backing =
            read_gpu_u64(transaction_access,
                         queue_ptr + offsetof(amd_queue_v2_t, alt_scratch_backing_memory_location));
        if (alt_backing.outcome != VmAccessOutcome::Complete)
          return admission_from_vm_outcome(alt_backing.outcome);
        const AtomicLoadResult alt_max_use = read_gpu_u64(
            transaction_access, queue_ptr + offsetof(amd_queue_v2_t, alt_scratch_max_use_index));
        if (alt_max_use.outcome != VmAccessOutcome::Complete)
          return admission_from_vm_outcome(alt_max_use.outcome);

        uint32_t alt_wave64_lane_bytes = 0;
        uint32_t alt_compute_tmpring_size = 0;
        std::array<uint32_t, 3> alt_dispatch_limits{};
        outcome =
            read_gpu_block(transaction_access,
                           queue_ptr + offsetof(amd_queue_v2_t, alt_scratch_wave64_lane_byte_size),
                           &alt_wave64_lane_bytes, sizeof(alt_wave64_lane_bytes));
        if (outcome != VmAccessOutcome::Complete)
          return admission_from_vm_outcome(outcome);
        outcome = read_gpu_block(transaction_access,
                                 queue_ptr + offsetof(amd_queue_v2_t, alt_scratch_dispatch_limit_x),
                                 alt_dispatch_limits.data(), sizeof(alt_dispatch_limits));
        if (outcome != VmAccessOutcome::Complete)
          return admission_from_vm_outcome(outcome);
        outcome = read_gpu_block(transaction_access,
                                 queue_ptr + offsetof(amd_queue_v2_t, alt_compute_tmpring_size),
                                 &alt_compute_tmpring_size, sizeof(alt_compute_tmpring_size));
        if (outcome != VmAccessOutcome::Complete)
          return admission_from_vm_outcome(outcome);

        const uint64_t required_wave_bytes =
            static_cast<uint64_t>(private_segment_fixed_size) * wave_size;
        const uint64_t provisioned_wave_bytes = static_cast<uint64_t>(alt_wave64_lane_bytes) * 64;
        const std::optional<uint32_t> alt_wave_limit = scratch_wave_limit(alt_compute_tmpring_size);
        const bool alt_scratch_usable =
            alt_backing.value != 0 && required_wave_bytes <= provisioned_wave_bytes &&
            alt_wave_limit.has_value() && pkt.grid_size_x <= alt_dispatch_limits[0] &&
            pkt.grid_size_y <= alt_dispatch_limits[1] &&
            pkt.grid_size_z <= alt_dispatch_limits[2] && aql_packet_id <= alt_max_use.value;
        if (alt_scratch_usable) {
          scratch_backing_addr = alt_backing.value;
          scratch_uses_alternate = true;
          scratch_wave_limit_per_se = *alt_wave_limit;
          scratch_wave_stride_per_se = *alt_wave_limit;
          scratch_wave_size_bytes = ((alt_compute_tmpring_size >> 12) & wavesize_mask) *
                                    properties.compute_tmpring_wavesize_granule;
          requires_dynamic_scratch = false;
          if (AMDHSA_BITS_GET(kd.kernel_code_properties,
                              KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER)) {
            outcome = read_gpu_block(transaction_access,
                                     queue_ptr +
                                         offsetof(amd_queue_v2_t, alt_scratch_resource_descriptor),
                                     launch_metadata.scratch_resource_descriptor.data(),
                                     sizeof(launch_metadata.scratch_resource_descriptor));
            if (outcome != VmAccessOutcome::Complete)
              return admission_from_vm_outcome(outcome);
          }
        }
      }

      if (async_scratch && !requires_dynamic_scratch) {
        outcome = record_async_scratch_use(queue, transaction_access, aql_packet_id,
                                           scratch_uses_alternate);
        if (outcome != VmAccessOutcome::Complete)
          return admission_from_vm_outcome(outcome);
      }

      // Runtime backing is finite and may be reclaimed. Only simulator-owned
      // backing may grow; request replacement for rejected runtime scratch.
      if (requires_dynamic_scratch &&
          (async_scratch || !scratch_allocator_ || !simulator_scratch)) {
        constexpr uint64_t kInsufficientScratchWave64 = 0x1;
        constexpr uint64_t kInsufficientScratchWave32 = 0x401;
        const uint64_t status =
            wave_size == 32 ? kInsufficientScratchWave32 : kInsufficientScratchWave64;
        return request_dynamic_scratch(queue, transaction_access, aql_packet_id, status);
      }
      if (requires_dynamic_scratch) {
        scratch_wave_limit_per_se = std::numeric_limits<uint32_t>::max();
        scratch_wave_stride_per_se = 0;
        scratch_wave_size_bytes = 0;
      }
      queue.scratch_request.reset();

      uint32_t queue_properties = 0;
      const VmAccessOutcome properties_outcome =
          read_gpu_block(transaction_access, queue_ptr + offsetof(amd_queue_t, queue_properties),
                         &queue_properties, sizeof(queue_properties));
      if (properties_outcome != VmAccessOutcome::Complete)
        return admission_from_vm_outcome(properties_outcome);
      scratch_use_once = (queue_properties & AMD_QUEUE_PROPERTIES_USE_SCRATCH_ONCE) != 0;
      util::Logger::cp([&](auto &os) {
        os << std::format("DYNAMIC_SCRATCH_READY q={} packet={} private={} backing={:#x} "
                          "caps={:#x} properties={:#x} alt={} use_once={}",
                          queue.queue_id, aql_packet_id, private_segment_fixed_size,
                          scratch_backing_addr, queue_caps, queue_properties,
                          scratch_uses_alternate, scratch_use_once);
      });
      if (scratch_use_once && queue.scratch_reclaim.active())
        return {.status = AqlAdmissionStatus::Malformed};
    }
  }

  if (private_segment_fixed_size > 0 && arch == ROCJITSU_CODE_ARCH_CDNA5)
    scratch_wave_limit_per_se = std::min(scratch_wave_limit_per_se, scratch_waves_per_se_);

  DispatchEntry dp{};
  dp.queue_id = queue.queue_id;
  dp.enabled_cus = queue.enabled_cus;
  dp.queue_packet_id = queue_packet_id;
  dp.address_space = queue.address_space;
  dp.execution_access = std::make_shared<GpuVmAccess>(transaction_access);
  dp.interrupt_sink = queue.interrupt_sink;
  dp.process_id = queue.process_id;
  dp.aql_packet_id = static_cast<uint32_t>(aql_packet_id);
  dp.kernel_entry_pc = entry_pc;
  dp.total_wgs = total_wgs;
  dp.kind = DispatchPacketKind::Kernel;
  dp.dispatched_wgs = 0;
  dp.completed_wgs = 0;
  dp.wfs_per_workgroup = wfs_per_wg;
  uint32_t sgpr_limit = cus_.empty() ? 112 : cus_[0]->config().sgprs_per_wf;
  uint32_t vgpr_limit = cus_.empty() ? 256 : cus_[0]->vgpr_allocation_block_size();
  uint32_t required_sgprs = sgprs > 0 ? sgprs : sgpr_limit;
  if (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4)
    required_sgprs = std::max(required_sgprs, 34u); // s32 stack pointer, s33 frame pointer
  dp.sgprs_per_wf = std::min(required_sgprs, sgpr_limit);
  dp.vgprs_per_wf = std::min(vgprs > 0 ? vgprs : vgpr_limit, vgpr_limit);
  dp.kernarg_addr = reinterpret_cast<uint64_t>(pkt.kernarg_address);
  dp.kernarg_size = kd.kernarg_size;
  dp.num_user_sgprs = user_sgprs;
  dp.kernel_code_properties = kd.kernel_code_properties;
  if (arch == ROCJITSU_CODE_ARCH_CDNA5) {
    const uint32_t named_barrier_blocks =
        AMDHSA_BITS_GET(kd.compute_pgm_rsrc3, COMPUTE_PGM_RSRC3_GFX125_NAMED_BAR_CNT);
    dp.num_named_barriers = std::min(named_barrier_blocks * 4u, ComputeUnitCore::kMaxNamedBarriers);
  }
  dp.kernel_wave_size = wave_size;
  dp.kernarg_preload = kd.kernarg_preload;
  if (!metadata.empty()) {
    dp.metadata_event_id = aql_metadata::event_id(metadata);
    dp.metadata_kernargs = aql_metadata::kernargs(metadata);
  }
  dp.initial_mode_raw = initial_mode_from_compute_pgm_rsrc1(kd.compute_pgm_rsrc1, arch);
  dp.private_segment_fixed_size = private_segment_fixed_size;
  dp.scratch_wave_limit_per_se = scratch_wave_limit_per_se;
  dp.scratch_wave_stride_per_se = scratch_wave_stride_per_se;
  dp.scratch_wave_size_bytes = scratch_wave_size_bytes;
  dp.group_segment_fixed_size = std::max(kd.group_segment_fixed_size, pkt.group_segment_size);
  dp.scratch_use_once = scratch_use_once;
  dp.wgp_mode = isa_properties(arch).supports_wgp_mode &&
                AMDHSA_BITS_GET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_WGP_MODE) != 0;
  dp.workgroup_id_offset = workgroup_id_offset_;
  dp.grid_size_x = pkt.grid_size_x;
  dp.grid_size_y = (num_dims >= 2) ? pkt.grid_size_y : 1;
  dp.grid_size_z = (num_dims >= 3) ? pkt.grid_size_z : 1;
  // For WG ID decomposition, use the dispatch dimensionality (setup field).
  // A 1D dispatch flattens the entire grid into workgroup_id_x.
  dp.grid_wgs_x = (num_dims <= 1) ? total_wgs : grid_wgs_x;
  dp.grid_wgs_y = (num_dims >= 2) ? grid_wgs_y : 1;
  dp.grid_wgs_z = (num_dims >= 3) ? grid_wgs_z : 1;
  dp.grid_yz_valid = num_dims >= 2;
  dp.cluster_size_x = nonzero_or_one(cluster_shape.size_x);
  dp.cluster_size_y = nonzero_or_one(cluster_shape.size_y);
  dp.cluster_size_z = nonzero_or_one(cluster_shape.size_z);
  dp.cluster_count_x = cluster_shape.count_x == 0
                           ? (dp.grid_wgs_x + dp.cluster_size_x - 1) / dp.cluster_size_x
                           : cluster_shape.count_x;
  dp.cluster_count_y = cluster_shape.count_y == 0
                           ? (dp.grid_wgs_y + dp.cluster_size_y - 1) / dp.cluster_size_y
                           : cluster_shape.count_y;
  dp.cluster_count_z = cluster_shape.count_z == 0
                           ? (dp.grid_wgs_z + dp.cluster_size_z - 1) / dp.cluster_size_z
                           : cluster_shape.count_z;

  uint64_t lds_capacity = 0;
  if (dp.wgp_mode) {
    for (const auto *spi : spis_)
      lds_capacity = std::max<uint64_t>(lds_capacity, spi->max_wgp_lds_bytes());
  } else {
    for (const auto *cu : cus_)
      lds_capacity =
          std::max<uint64_t>(lds_capacity, static_cast<uint64_t>(cu->config().lds_size_kb) * 1024u);
  }
  const uint64_t aligned_lds =
      (static_cast<uint64_t>(dp.group_segment_fixed_size) + 255u) & ~uint64_t{255u};
  if (dp.wgp_mode && lds_capacity == 0) {
    return {.status = AqlAdmissionStatus::Unsupported,
            .diagnostic = AqlPacketDiagnostic::WgpTopologyUnavailable};
  }
  if (aligned_lds > lds_capacity) {
    return {.status = AqlAdmissionStatus::Malformed,
            .diagnostic = AqlPacketDiagnostic::LdsCapacityExceeded};
  }
  if (const std::optional<AqlPacketDiagnostic> diagnostic = validate_cluster_shape(dp))
    return {.status = AqlAdmissionStatus::Malformed, .diagnostic = *diagnostic};

  // No guest-controlled validation may fail after this point: admitting a
  // dispatch can now allocate scratch, publish queue metadata, flush caches,
  // notify plugins, and fan out work to peer XCDs.
  dp.dispatch_id = allocate_dispatch_id();
  dp.profiling_start_timestamp = hsa_system_timestamp();

  // For KFD dispatches, provide pointers the kernel may need via user SGPRs.
  // The queue_ptr and dispatch_ptr are GPU VAs that the kernel reads via SMEM.
  if (uses_kfd_queue_abi) {
    dp.dispatch_ptr = pkt_addr;
    dp.queue_ptr = queue_ptr;
    if (dp.private_segment_fixed_size > 0) {
      uint64_t scratch_loc_va =
          dp.queue_ptr + offsetof(amd_queue_t, scratch_backing_memory_location);
      dp.scratch_backing_addr = scratch_backing_addr;

      // Publish the scratch backing location and COMPUTE_TMPRING_SIZE into the
      // ABI-stable part of amd_queue_t so rocm-dbgapi can compute each wave's
      // private (scratch) memory region, letting ROCgdb read scratch-resident
      // variables. Real CP firmware populates these when it assigns scratch to
      // the queue; the emulator's ROCr instead sets the backing via
      // SET_SCRATCH_BACKING_VA and leaves these fields zero, so the CP fills
      // them here. Field layout per rocdbgapi architecture.cpp
      // scratch-memory region. The WAVES field is common, while ISA properties
      // describe the generation-specific WAVESIZE unit and field width.
      if (scratch_allocator_ && dp.scratch_backing_addr != 0 && !cus_.empty() &&
          dp.scratch_wave_stride_per_se == 0 && !scratch_uses_alternate) {
        const uint64_t per_wave_bytes =
            static_cast<uint64_t>(dp.private_segment_fixed_size) * wave_size;
        // Encode the actual stride used by init_wavefront_regs(), using the target's
        // allocation granule and this dispatch's wave width. This keeps flat_scratch
        // consistent with rocm-dbgapi for every scoreboard slot after slot zero.
        const auto properties = isa_properties(arch);
        const uint32_t wavesize_unit = properties.compute_tmpring_wavesize_granule;
        const uint64_t per_wave_stride =
            ((per_wave_bytes + wavesize_unit - 1) / wavesize_unit) * wavesize_unit;
        assert(wavesize_unit != 0 && properties.compute_tmpring_wavesize_bits != 0);
        const uint32_t wavesize_field = static_cast<uint32_t>(per_wave_stride / wavesize_unit);
        uint32_t waves_field = 0;
        if (arch == ROCJITSU_CODE_ARCH_CDNA5) {
          // gfx12 interprets WAVES as the number of physical scratch slots per
          // shader engine. The CWSR wave word supplies the SE plus its per-SE
          // scoreboard slot, so publish the same capacity used by allocation.
          waves_field = scratch_waves_per_se_;
        } else {
          // Older debugger layouts interpret WAVES as a device-wide count and
          // require it to be divisible by the shader-engine count.
          uint32_t se = std::max(1u, scratch_wave_divisor_);
          uint64_t total_waves = static_cast<uint64_t>(total_wgs) * wfs_per_wg;
          waves_field =
              static_cast<uint32_t>(((std::max<uint64_t>(1, total_waves) + se - 1) / se) * se);
        }
        const uint32_t wavesize_mask =
            util::mask<uint32_t>(properties.compute_tmpring_wavesize_bits);
        uint32_t tmpring = (waves_field & 0xFFFu) | ((wavesize_field & wavesize_mask) << 12);
        (void)write_gpu_block(transaction_access, scratch_loc_va, &dp.scratch_backing_addr,
                              sizeof(dp.scratch_backing_addr));
        (void)write_gpu_block(transaction_access,
                              dp.queue_ptr + offsetof(amd_queue_t, compute_tmpring_size), &tmpring,
                              sizeof(tmpring));
        queue.simulator_scratch =
            ComputeQueueRecord::SimulatorScratch{dp.scratch_backing_addr, tmpring};
      }
    }
  }

  dp.enable_wg_id_x =
      AMDHSA_BITS_GET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X);
  dp.enable_wg_id_y =
      AMDHSA_BITS_GET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y);
  dp.enable_wg_id_z =
      AMDHSA_BITS_GET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z);
  dp.enable_vgpr_workitem_id = static_cast<uint8_t>(
      AMDHSA_BITS_GET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_ENABLE_VGPR_WORKITEM_ID));
  dp.workgroup_size_x = pkt.workgroup_size_x;
  dp.workgroup_size_y = pkt.workgroup_size_y;
  dp.workgroup_size_z = pkt.workgroup_size_z;
  dp.completion_signal = pkt.completion_signal.handle;
  dp.host_signal = false;
  dp.wait_for_predecessors = (pkt.header >> HSA_PACKET_HEADER_BARRIER) & 1;

  // Process AQL acquire fence: invalidate caches so the kernel sees the
  // latest host/agent writes (kernarg data, input buffers, etc.).
  // On real hardware the CP issues GL1_INV + GL2_INV for SYSTEM/AGENT scope.
  // Only this XCD's CUs are reachable from here; a peer XCD's caches belong to
  // another partition and must not be touched from this thread. The shard carries
  // the fence instead, and each peer performs the same invalidate on its own thread
  // when it takes delivery -- see drain_fanout_inbox().
  uint32_t acquire_scope = (pkt.header >> HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) & 0x3;
  dp.acquire_invalidate = acquire_scope >= HSA_FENCE_SCOPE_AGENT;
  if (dp.acquire_invalidate && !cus_.empty()) {
    // Deliberately the per-CU walk, not the deduplicated flush_gpu_caches(). The
    // two are equivalent per invocation, but collapsing the repeated sweeps on the
    // release path caused peer ranks to hang on flags left unpublished in L2, and
    // that mechanism is still not understood. Until it is, this path -- which
    // predates fan-out -- keeps exactly the cache behaviour it had, and only the
    // new peer-side fence in drain_fanout_inbox() uses the collapsed form.
    for (ComputeUnitCore *cu : cus_)
      cu->flush_all(queue.process_id);
  }

  std::string kernel_symbol;
  if (uses_kfd_queue_abi) {
    auto [host_range_base, host_range_size] = transaction_access.host_range(pkt.kernel_object);
    auto *kernel_object_host_ptr =
        reinterpret_cast<uint8_t *>(transaction_access.resolve_host_pointer(pkt.kernel_object));
    if (host_range_base != 0 && kernel_object_host_ptr) {
      auto *host_range_begin = reinterpret_cast<const uint8_t *>(host_range_base);
      auto *elf_base = find_elf_base(kernel_object_host_ptr, host_range_begin);
      if (elf_base) {
        const uint64_t kernel_offset = static_cast<uint64_t>(kernel_object_host_ptr - elf_base);
        if (pkt.kernel_object >= kernel_offset)
          code_load_bias = pkt.kernel_object - kernel_offset;
        uint64_t elf_accessible =
            host_range_size - static_cast<uint64_t>(elf_base - host_range_begin);
        kernel_symbol = find_kernel_symbol(kernel_object_host_ptr, elf_base, elf_accessible);
      }
    }
  }
  dp.code_load_bias = code_load_bias;
  std::string kernel_name = kernel_display_name(kernel_symbol);
  ++total_dispatched_;

  KernelDispatchInfo dispatch_info{};
  dispatch_info.dispatch_id = dp.dispatch_id;
  dispatch_info.kernel_object = pkt.kernel_object;
  dispatch_info.entry_pc = entry_pc;
  dispatch_info.kernel_symbol = kernel_symbol;
  dispatch_info.kernel_name = kernel_name;
  dispatch_info.lds_size_bytes = dp.group_segment_fixed_size;
  dispatch_info.wave_size = wave_size;
  dispatch_info.code_target =
      cus_.empty() ? ROCJITSU_CODE_TARGET_INVALID : cus_[0]->config().target;
  dispatch_info.grid_size_x = pkt.grid_size_x;
  dispatch_info.grid_size_y = pkt.grid_size_y;
  dispatch_info.grid_size_z = pkt.grid_size_z;
  dispatch_info.workgroup_size_x = pkt.workgroup_size_x;
  dispatch_info.workgroup_size_y = pkt.workgroup_size_y;
  dispatch_info.workgroup_size_z = pkt.workgroup_size_z;
  dispatch_info.cluster_size_x = dp.cluster_size_x;
  dispatch_info.cluster_size_y = dp.cluster_size_y;
  dispatch_info.cluster_size_z = dp.cluster_size_z;
  dispatch_info.workgroup_count = total_wgs;
  dispatch_info.wfs_per_workgroup = wfs_per_wg;
  dispatch_info.sgprs_per_wf = dp.sgprs_per_wf;
  dispatch_info.vgprs_per_wf = dp.vgprs_per_wf;
  plugin_group_->onAmdgpuDispatchPacketProcessed(dispatch_info);

  util::Logger::vm([&](auto &os) {
    os << std::format("dispatch #{} d={} \"{}\" symbol=\"{}\" grid=[{},{},{}] wg=[{},{},{}] wgs={} "
                      "lds={} mode={} sgpr={} vgpr={} sig={:#x}",
                      total_dispatched_, dp.dispatch_id, dispatch_info.kernelNameOrUnknown(),
                      dispatch_info.kernelSymbolOrUnknown(), pkt.grid_size_x, pkt.grid_size_y,
                      pkt.grid_size_z, pkt.workgroup_size_x, pkt.workgroup_size_y,
                      pkt.workgroup_size_z, total_wgs, kd.group_segment_fixed_size,
                      dp.wgp_mode ? "WGP" : "CU", dp.sgprs_per_wf, dp.vgprs_per_wf,
                      dp.completion_signal);
  });
  util::Logger::cp([&](auto &os) {
    os << std::format("DISPATCH #{} d={} \"{}\" symbol=\"{}\" wgs={} wfs/wg={} sig={:#x} pid={} "
                      "ko={:#x} pc={:#x} kernarg={:#x} user_sgprs={}",
                      total_dispatched_, dp.dispatch_id, dispatch_info.kernelNameOrUnknown(),
                      dispatch_info.kernelSymbolOrUnknown(), total_wgs, wfs_per_wg,
                      dp.completion_signal, dp.process_id, pkt.kernel_object, entry_pc,
                      dp.kernarg_addr, dp.num_user_sgprs);
    auto *ko_ptr = transaction_access.resolve_host_pointer(pkt.kernel_object);
    auto *pc_ptr = transaction_access.resolve_host_pointer(entry_pc, sizeof(uint32_t));
    os << std::format(" ko_mapped={} pc_mapped={}", ko_ptr != nullptr, pc_ptr != nullptr);
    if (pc_ptr) {
      uint32_t first_word;
      std::memcpy(&first_word, pc_ptr, sizeof(first_word));
      os << std::format(" first_inst={:#010x}", first_word);
    }
  });

  dispatch_launch_metadata_.insert_or_assign(dp.dispatch_id, launch_metadata);
  if (dp.scratch_use_once)
    queue.scratch_reclaim.begin(dp.dispatch_id);
  if (queue.xcd_fanout)
    fan_out_dispatch(dp, launch_metadata);

  queue.push_entry(std::move(dp));
  return {.status = AqlAdmissionStatus::Complete};
}

bool CommandProcessor::execute_aql_pm4(ComputeQueueRecord &queue, DispatchEntry &entry,
                                       simdojo::Tick now) {
  // A peer can fault the grid before this CP drains its fault inbox. Leave
  // cancellation to the inbox without starting or resuming the command buffer.
  if (entry.grid_faulted())
    return false;
  if (!entry.aql_pm4_ib_dwords || entry.completed_wgs == entry.total_wgs)
    return true;
  if (!entry.execution_begun) {
    Pm4Submission submission;
    submission.allow_dispatch = false;
    submission.buffers.push_back({entry.aql_pm4_ib_address, entry.aql_pm4_ib_dwords});
    queue.commands.submissions.push_back(std::move(submission));
    entry.execution_begun = true;
  }
  fetch_pm4(queue, queue.dispatches, now);
  if (queue.faulted) {
    notify_dispatch_vm_fault(entry.queue_id, entry.process_id, entry.dispatch_id,
                             VmAccessOutcome::Faulted);
    return false;
  }
  return queue.commands.submissions.empty();
}

AqlAdmissionResult CommandProcessor::admit_aql_packet(const AqlPacketProcessRequest &request,
                                                      AqlPreparedPacket prepared) {
  const std::vector<ComputeQueueRecord>::iterator queue = std::ranges::find(
      compute_queues_, request.registration_id, &ComputeQueueRecord::registration_id);
  if (queue == compute_queues_.end() || queue->process_id != request.process_id ||
      queue->queue_id != request.queue_id || queue->faulted || queue->fanout_replica) {
    return {.status = AqlAdmissionStatus::Malformed};
  }

  if (prepared.kind == AqlPreparedPacketKind::KernelDispatch) {
    return admit_kernel_dispatch(prepared.kernel_dispatch, *queue, request.access,
                                 request.packet_address, request.ring_slot, request.packet_index,
                                 prepared.cluster_shape, request.metadata);
  }

  DispatchEntry entry{
      .dispatch_id = allocate_dispatch_id(),
      .queue_id = queue->queue_id,
      .address_space = queue->address_space,
      .execution_access = {},
      .interrupt_sink = queue->interrupt_sink,
      .process_id = queue->process_id,
      .completion_signal = prepared.completion_signal,
      .kind = DispatchPacketKind::NonKernel,
      .wait_for_predecessors = prepared.barrier_bit,
      .blocks_following = prepared.blocks_following,
      .aql_pm4_ib_address = prepared.pm4_ib_address,
      .aql_pm4_ib_dwords = prepared.pm4_ib_dwords,
  };
  if (prepared.kind == AqlPreparedPacketKind::Pm4Ib) {
    // One completion token per XCD prevents automatic retirement of an unexecuted
    // non-kernel packet and keeps its successor blocked until every IB finishes.
    entry.total_wgs = entry.dispatched_wgs = 1;
    if (queue->xcd_fanout) {
      entry.grid_completion = std::make_shared<GridCompletion>();
      entry.grid_completion->grid_wgs = xcd_peers_.size();
    }
  }
  if (!request.metadata.empty())
    entry.metadata_event_id = aql_metadata::event_id(request.metadata);
  if (queue->xcd_fanout)
    replicate_non_kernel_entry(entry);
  queue->push_entry(std::move(entry));
  return {.status = AqlAdmissionStatus::Complete};
}

void CommandProcessor::arm_grid_wait_recheck() {
  // A shard whose own share is done but whose grid is still running on another
  // XCD is a stall like any other, and has to be re-armed as one.
  //
  // The XCD that retires the grid does call wake_all_xcds(), but that wake
  // travels the engine's cross-thread async queue, which neither contributes to
  // LBTS nor counts as outstanding work when the engine tests for termination.
  // With one partition per XCD, every partition can publish TICK_MAX in the same
  // epoch the wake is deposited, and the run ends on that before the next epoch
  // drains it -- so the XCD holding the completion signal never re-drains and
  // never writes it. Keeping a re-check on this CP's own event queue holds its
  // partition's next-event time finite for exactly as long as it is waiting,
  // which leaves the wake an optimization rather than the only thing standing
  // between the grid retiring and the signal firing.
  //
  // Caller must hold hw_queue_mutex_ and must be on this CP's own partition
  // thread, which is where the re-check is enqueued.
  for (const auto &qs : compute_queues_) {
    if (qs.entries.empty())
      continue;
    const auto &head = qs.entries.front();
    if (head.fully_completed() && !head.grid_fully_completed()) {
      arm_stall_recheck(engine()->context(partition_id()).current_tick());
      return;
    }
  }
}

void CommandProcessor::arm_stall_recheck(simdojo::Tick now) {
  if (!engine())
    return;
  // A doorbell poll thread runs only for queues this CP polls; it re-checks
  // stall_pending_ at its 100us cadence, so the engine can idle instead of spinning.
  // Internal test queues have no poll thread — they are driven by engine->run()/
  // step() — and neither do fan-out replicas, so in both cases the re-check must be
  // kept alive on the main event queue instead.
  if (polls_kfd_queues()) {
    stall_pending_.store(true, std::memory_order_release);
    return;
  }
  // Back the re-check off rather than re-arming on the very next tick. What
  // actually ends one of these waits is an external event -- a peer's shard, or
  // the wake the retiring XCD sends -- and each of those resets the backoff, so
  // the wait still ends promptly. This event only has to keep the partition's
  // next-event time finite so the engine cannot decide the run is over while a
  // cross-thread wake is still undelivered (see arm_grid_wait_recheck).
  //
  // At one tick it is a spin, and a costly one: with fan-out every peer XCD waits
  // on the owner's grid, and a peer re-entered this handler once per simulated
  // tick -- 17M times on a corpus case that needs 24 doorbells without fan-out,
  // which is where a 12x slowdown came from. Backing off is nearly free in
  // simulated time: with no other event pending the engine jumps straight to the
  // re-check, so a longer interval skips idle ticks rather than adding latency.
  // Several queues or retry sites can remain blocked on the same pass. Keep
  // one live re-check; otherwise each retry can enqueue several more. Fresh
  // work can reset the backoff and pull the deadline earlier. Only that case
  // leaves a stale event, which the generation check ignores when it fires.
  const simdojo::Tick tick = now + stall_recheck_backoff_;
  if (stall_recheck_pending_ && stall_recheck_tick_ <= tick)
    return;
  stall_recheck_pending_ = true;
  stall_recheck_tick_ = tick;
  schedule_event(
      &stall_recheck_event_, tick,
      std::make_unique<simdojo::Message>(simdojo::MessageHeader{}, ++stall_recheck_generation_));
  stall_recheck_backoff_ = std::min(stall_recheck_backoff_ * 2, kMaxStallRecheckBackoff);
}

void CommandProcessor::fetch_from_queue(ComputeQueueRecord &queue, simdojo::Tick now) {
  if (queue.packet_format != QueuePacketFormat::Aql || queue.submission_queue)
    return;
  // A replica's work arrives as dispatch shards, not from the ring. Reading the
  // ring here would also advance a read pointer the owning XCD owns, and its
  // suspension flags are the owner's copy rather than state this CP maintains.
  if (queue.fanout_replica)
    return;
  const auto release_slot = [&]() {
    if (!queue.aql_slot_release)
      return true;
    auto &release = *queue.aql_slot_release;
    for (;;) {
      const bool releasing_companion_header =
          release.metadata_address && release.metadata_block < aql_metadata::kBlocks;
      const uint64_t address =
          releasing_companion_header
              ? release.metadata_address + release.metadata_block * aql_metadata::kBlockBytes
              : release.address;
      const auto outcome = release.access.atomic_store(
          address, sizeof(uint32_t),
          releasing_companion_header ? uint32_t{HSA_PACKET_TYPE_INVALID} : release.header);
      if (outcome == VmAccessOutcome::Complete) {
        if (releasing_companion_header) {
          ++release.metadata_block;
          continue;
        }
        queue.aql_slot_release.reset();
        return true;
      }
      if (outcome == VmAccessOutcome::Unavailable)
        arm_stall_recheck(now);
      else {
        queue.publication_faulted = true;
        queue.faulted = true;
      }
      return false;
    }
  };
  if (!release_slot())
    return;
  if (queue.read_pointer_journal.publication_pending()) {
    const VmAccessOutcome outcome = queue.read_pointer_journal.publish();
    if (outcome == VmAccessOutcome::Unavailable) {
      arm_stall_recheck(now);
      return;
    }
    if (outcome != VmAccessOutcome::Complete) {
      queue.publication_faulted = true;
      queue.faulted = true;
      return;
    }
  }
  // A packet or execution fault stops new admission, but it must not strand a
  // cursor retirement that committed before the fault became visible.
  if (queue.faulted)
    return;
  // ROCr owns a use-once scratch allocation until the dispatch retires and the
  // reclaim notification is acknowledged. Do not inspect a following packet:
  // doing so could request or reuse scratch that ROCr has not released yet.
  if (queue.scratch_reclaim.active())
    return;
  // Barrier packets serialize later fetch, not merely their first admission
  // pass. Keep the successor in the ring until the blocking entry has completed
  // and its completion publication is durable.
  if (std::ranges::any_of(queue.entries, [](const DispatchEntry &entry) {
        return entry.blocks_following && !entry.completion_notified;
      }))
    return;

  // One queue-service transaction observes one immutable address-space
  // generation. A root replacement may affect the next retry, but it cannot
  // splice new translations into queue-pointer, ring, dependency, admission,
  // or consumer-pointer accesses that belong to this attempt.
  std::optional<GpuVmAccess> transaction_access =
      gpu_vm_ != nullptr ? gpu_vm_->snapshot_pinned(queue.address_space) : std::nullopt;
  if (!transaction_access) {
    queue.faulted = true;
    return;
  }
  const GpuVmAccess &access = *transaction_access;

  auto load_queue_pointer = [&](uint64_t address, uint64_t &value) {
    const AtomicLoadResult loaded = read_gpu_u64(access, address);
    if (loaded.outcome == VmAccessOutcome::Complete) {
      value = loaded.value;
      return true;
    }
    if (loaded.outcome == VmAccessOutcome::Unavailable)
      arm_stall_recheck(now);
    else
      queue.faulted = true;
    return false;
  };
  if (queue.suspended()) {
    // A command-processor event can race a debugger suspension even when this
    // queue has no new packets. Do not turn that stale event into an endless
    // resume/event chain: request a resume pass only when packet fetch really
    // was deferred.
    uint64_t write_idx = 0;
    uint64_t read_idx = 0;
    if (!load_queue_pointer(queue.read_ptr_va, read_idx) ||
        !load_queue_pointer(queue.write_ptr_va, write_idx)) {
      return;
    }
    const uint64_t fetch_idx = std::max(read_idx, queue.fetch_cursor);
    queue.debug_work_deferred |= fetch_idx < write_idx;
    return;
  }
  if ((queue.doorbell_mode == QueueDoorbellMode::HostPolled && queue.doorbell_base == nullptr) ||
      (queue.doorbell_mode == QueueDoorbellMode::VmPolled && queue.doorbell_va == 0))
    return;

  // Read producer and consumer indices through the transaction's immutable VM
  // snapshot, regardless of which frontend supplied the queue.
  uint64_t write_idx = 0;
  uint64_t read_idx = 0;
  if (!load_queue_pointer(queue.read_ptr_va, read_idx) ||
      !load_queue_pointer(queue.write_ptr_va, write_idx)) {
    return;
  }
  util::Logger::vm([&](auto &os) {
    static uint64_t fetch_count = 0;
    if (write_idx != read_idx && ++fetch_count <= 50)
      os << std::format("FETCH q={} w={} r={} delta={}", queue.queue_id, write_idx, read_idx,
                        write_idx - read_idx);
  });

  // AQL doorbell clamping (compute queues only).
  // Use the CP-private fetch cursor as the authoritative next-packet index. It
  // normally equals read_ptr_va (the CP is the sole writer of a compute queue's
  // read pointer), but while the debugger holds read_ptr_va at a trapped
  // dispatch, the cursor stays ahead so already-dispatched packets are not
  // re-fetched.
  const uint64_t published_read_idx = read_idx;
  read_idx = std::max(read_idx, queue.fetch_cursor);
  uint64_t process_limit = write_idx;
  if (queue.doorbell_mode != QueueDoorbellMode::VmPolled) {
    const uint64_t doorbell = queue.last_doorbell;
    // KFD initializes a new queue's doorbell to UINT64_MAX, meaning that no
    // packet has been published yet. The producer may reserve a ring slot by
    // advancing write_ptr before it initializes the packet and release-stores
    // the header. Treating the sentinel as an unclamped producer cursor lets a
    // concurrent CP read that partially initialized slot.
    if (doorbell == std::numeric_limits<uint64_t>::max())
      return;
    uint64_t doorbell_limit = doorbell + 1;
    if (doorbell_limit < process_limit)
      process_limit = doorbell_limit;
    if (read_idx >= process_limit)
      return;
  } else if (read_idx >= process_limit) {
    return;
  }

  static_assert(sizeof(hsa_kernel_dispatch_packet_t) == kAqlPacketBytes);
  const uint32_t num_slots = queue.ring_size / kAqlPacketBytes;

  while (read_idx < process_limit) {
    const uint32_t slot = static_cast<uint32_t>(read_idx % num_slots);
    const uint64_t pkt_addr = queue.ring_base_va + slot * kAqlPacketBytes;

    // ROCr can publish just the 16-bit header; other producers release-store
    // the first dword. Acquire both widths before reading the body. In particular,
    // do not read setup in the dword until the 16-bit publisher has released it.
    // Once valid, the producer leaves the packet untouched until slot release.
    AtomicLoadResult header_load = access.atomic_load(pkt_addr, sizeof(uint16_t));
    if (header_load.outcome == VmAccessOutcome::Complete &&
        (header_load.value & 0xFF) != HSA_PACKET_TYPE_INVALID)
      header_load = access.atomic_load(pkt_addr, sizeof(uint32_t));
    if (header_load.outcome != VmAccessOutcome::Complete) {
      process_limit = read_idx;
      if (header_load.outcome == VmAccessOutcome::Unavailable)
        arm_stall_recheck(now);
      else
        queue.faulted = true;
      break;
    }
    const uint16_t published_header = static_cast<uint16_t>(header_load.value);
    const uint8_t published_type = published_header & 0xFF;
    if (published_type == HSA_PACKET_TYPE_INVALID) {
      process_limit = read_idx;
      invalid_pending_.store(true, std::memory_order_release);
      break;
    }

    hsa_kernel_dispatch_packet_t pkt{};
    const VmAccessOutcome packet_read = read_gpu_block(
        access, pkt_addr + sizeof(uint32_t), reinterpret_cast<std::byte *>(&pkt) + sizeof(uint32_t),
        kAqlPacketBytes - sizeof(uint32_t));
    if (packet_read != VmAccessOutcome::Complete) {
      process_limit = read_idx;
      if (packet_read == VmAccessOutcome::Unavailable)
        arm_stall_recheck(now);
      else
        queue.faulted = true;
      break;
    }
    pkt.header = published_header;
    pkt.setup = static_cast<uint16_t>(header_load.value >> 16);

    uint8_t pkt_type = pkt.header & 0xFF;
    util::Logger::cp([&](auto &os) {
      auto type_name = [](uint8_t t) -> const char * {
        switch (t) {
        case 0:
          return "VENDOR_SPECIFIC";
        case 1:
          return "INVALID";
        case 2:
          return "KERNEL_DISPATCH";
        case 3:
          return "BARRIER_AND";
        case 5:
          return "BARRIER_OR";
        default:
          return "UNKNOWN";
        }
      };
      os << std::format("PKT q={} slot={} type={}({}) header={:#x} barrier_bit={} read_idx={}",
                        queue.queue_id, slot, pkt_type, type_name(pkt_type), pkt.header,
                        (pkt.header >> HSA_PACKET_HEADER_BARRIER) & 1, read_idx);
    });

    aql_metadata::Packet metadata{};
    uint64_t metadata_address = 0;
    if (queue.metadata_ring_size) {
      AmdExtKernelDispatchPacket vendor{};
      std::memcpy(&vendor, &pkt, sizeof(vendor));
      const bool kernel = pkt_type == HSA_PACKET_TYPE_KERNEL_DISPATCH ||
                          (pkt_type == HSA_PACKET_TYPE_VENDOR_SPECIFIC &&
                           vendor.amd_format == kHsaAmdPacketTypeExtKernelDispatch);
      const bool barrier = pkt_type == HSA_PACKET_TYPE_BARRIER_AND ||
                           pkt_type == HSA_PACKET_TYPE_BARRIER_OR ||
                           (pkt_type == HSA_PACKET_TYPE_VENDOR_SPECIFIC &&
                            vendor.amd_format == kHsaAmdPacketTypeBarrierValue);
      if (!kernel && !barrier) {
        queue.faulted = true;
        process_limit = read_idx;
        break;
      }
      metadata_address = queue.ring_base_va + queue.ring_size + slot * aql_metadata::kPacketBytes;
      const uint32_t blocks = kernel ? aql_metadata::kBlocks : 1;
      bool ready = true;
      bool absent = true;
      for (uint32_t block = 0; block < blocks; ++block) {
        const AtomicLoadResult header = access.atomic_load(
            metadata_address + block * aql_metadata::kBlockBytes, sizeof(uint32_t));
        if (header.outcome != VmAccessOutcome::Complete) {
          ready = false;
          if (header.outcome == VmAccessOutcome::Unavailable)
            arm_stall_recheck(now);
          else
            queue.faulted = true;
          break;
        }
        metadata[aql_metadata::header_word(block)] = static_cast<uint32_t>(header.value);
        absent &= header.value == HSA_PACKET_TYPE_INVALID;
      }
      // CLR publishes metadata before the AQL header. All INVALID headers mean
      // the producer supplied no companion (for example DEBUG_CLR_ENABLE_KDQ=0).
      // Use descriptor/kernarg/signal memory in that case, not an endless retry.
      if (ready && absent) {
        metadata_address = 0;
      } else if (ready) {
        for (uint32_t block = 0; block < blocks; ++block) {
          const uint32_t header = metadata[aql_metadata::header_word(block)];
          if (header != pkt_type) {
            ready = false;
            if (header == HSA_PACKET_TYPE_INVALID)
              arm_stall_recheck(now);
            else
              queue.faulted = true;
            break;
          }
          const VmAccessOutcome outcome = read_gpu_block(
              access, metadata_address + block * aql_metadata::kBlockBytes + sizeof(uint32_t),
              metadata.data() + aql_metadata::header_word(block) + 1,
              aql_metadata::kBlockBytes - sizeof(uint32_t));
          if (outcome != VmAccessOutcome::Complete) {
            ready = false;
            if (outcome == VmAccessOutcome::Unavailable)
              arm_stall_recheck(now);
            else
              queue.faulted = true;
            break;
          }
        }
      }
      if (!ready) {
        process_limit = read_idx;
        break;
      }
    }
    const auto packet_bytes =
        std::as_bytes(std::span<const hsa_kernel_dispatch_packet_t, 1>(&pkt, 1));
    const AqlPacketProcessResult result = aql_packet_processor_.process({
        .packet = packet_bytes,
        .access = access,
        .registration_id = queue.registration_id,
        .process_id = queue.process_id,
        .queue_id = queue.queue_id,
        .ring_slot = slot,
        .packet_index = read_idx,
        .packet_address = pkt_addr,
        .kernel_admission_enabled = !queue.enabled_cus || !queue.enabled_cus->empty(),
        .metadata =
            metadata_address ? std::span<const uint32_t>(metadata) : std::span<const uint32_t>{},
    });
    const PacketProcessResult &packet_result = result.packet_result();
    if (!valid_packet_process_result(packet_result, packet_bytes.size(), kAqlPacketBytes))
      throw std::logic_error("AQL processor returned an invalid common packet result");

    if (packet_result.status == PacketProcessStatus::Complete) {
      if (packet_result.retirement_bytes != kAqlPacketBytes) {
        throw std::logic_error("AQL processor completed without retiring exactly one packet");
      }
      ++read_idx;
      // HSA System Architecture 1.2, section 2.8.3: release the format field
      // using a 32-bit atomic transaction before exposing the slot for reuse.
      // Admission is already durable, so retain this store rather than replay it.
      queue.aql_slot_release.emplace(ComputeQueueRecord::AqlSlotRelease{
          .access = access,
          .address = pkt_addr,
          .header = (static_cast<uint32_t>(header_load.value) & ~0xffu) | HSA_PACKET_TYPE_INVALID,
          .metadata_address = metadata_address});
      if (!release_slot() || result.blocks_following || queue.scratch_reclaim.active()) {
        process_limit = read_idx;
        break;
      }
      continue;
    }

    // Only a successful, durable admission may advance the AQL consumer cursor.
    process_limit = read_idx;
    if (packet_result.status == PacketProcessStatus::Blocked &&
        result.blocked_reason == AqlBlockedReason::HeaderInvalid) {
      util::Logger::cp([&](auto &os) {
        os << std::format("{}: INVALID_RETRY q={} slot={} read_idx={} va={:#x}", name(),
                          queue.queue_id, slot, read_idx, pkt_addr);
      });
      // The runtime has not finished publishing this packet's header. The KFD
      // poll thread notices invalid_pending_ and retries at its paced interval.
      invalid_pending_.store(true, std::memory_order_release);
    } else if (packet_result.status == PacketProcessStatus::Blocked) {
      // Dependencies, transient VM access, and a temporarily unavailable
      // admission path all retry from the same unretired packet.
      arm_stall_recheck(now);
    } else if (packet_result.status == PacketProcessStatus::Unsupported) {
      queue.faulted = true;
    } else if (packet_result.status == PacketProcessStatus::NeedInput) {
      throw std::logic_error("fixed-size AQL processor requested additional packet bytes");
    } else {
      queue.faulted = true;
    }
    break;
  }

  // Advance the CP-private cursor to match. read_ptr_va may subsequently be
  // lowered by the debugger to hold a trapped dispatch; the cursor is not, so
  // the next fetch resumes here rather than re-fetching held packets.
  queue.fetch_cursor = process_limit;

  // A blocked head packet does not create a cursor-publication dependency when
  // the guest-visible cursor is already current. A debugger may deliberately
  // hold the published cursor behind fetch_cursor, in which case this still
  // publishes the previously retired progress using the transaction snapshot.
  if (process_limit == published_read_idx)
    return;

  queue.read_pointer_journal.retire(process_limit, process_limit, std::move(*transaction_access));
  if (queue.aql_slot_release)
    return;
  const VmAccessOutcome publication = queue.read_pointer_journal.publish();

  if (publication == VmAccessOutcome::Unavailable) {
    arm_stall_recheck(now);
    return;
  }
  if (publication != VmAccessOutcome::Complete) {
    queue.publication_faulted = true;
    queue.faulted = true;
  }
}

void CommandProcessor::handle_doorbell(simdojo::Tick timestamp) {
  doorbell_handle_count_.fetch_add(1, std::memory_order_relaxed);
  handle_doorbell_sync(timestamp);
}

void CommandProcessor::handle_doorbell_sync(simdojo::Tick now) {
  // Release so the doorbell poll thread's acquire-load cannot observe a stale
  // "pending" after this handler has re-fetched; pairs with the release-stores at
  // the INVALID-packet and barrier/dependency stall sites. A site that is still
  // unsatisfied on this pass re-sets its flag below, re-arming the paced re-check.
  invalid_pending_.store(false, std::memory_order_release);
  stall_pending_.store(false, std::memory_order_release);

  // Queue storage must remain stable while a pooled CU batch temporarily drops
  // hw_queue_mutex_. Registration and removal take this mutex exclusively.
  std::shared_lock<std::shared_mutex> structure_lock(queue_structure_mutex_);

  // Apply terminal faults before accepting any later shards. These inbox
  // drains release their leaf locks before acquiring the AQL queue lock.
  drain_dispatch_fault_inbox();
  drain_fanout_inbox();
  drain_doorbell_inbox();

  // AQL admission and PM4 execution share the queue lock and lifetime.
  std::unique_lock<std::recursive_mutex> lock(hw_queue_mutex_);
  util::Logger::cp([&](auto &os) {
    os << std::format("{}: DOORBELL queues={}", name(), compute_queues_.size());
  });

  // Finish a prior durable publication before admitting a new queue generation.
  if (!drain_completions())
    return;

  for (auto &queue : compute_queues_)
    queue.command_retry_pending = false;
  service_command_streams(now);

  size_t entries_before = 0;
  for (const ComputeQueueRecord &queue : compute_queues_)
    entries_before += queue.entries.size();

  for (ComputeQueueRecord &queue : compute_queues_)
    if (!queue.publication_retry_pending)
      fetch_from_queue(queue, now);

  size_t entries_after = 0;
  for (const ComputeQueueRecord &queue : compute_queues_)
    entries_after += queue.entries.size();
  if (entries_after != entries_before)
    stall_recheck_backoff_ = 1;
  util::Logger::cp([&](auto &os) {
    os << std::format("{}: FETCHED {} new entries (total={})", name(),
                      entries_after - entries_before, entries_after);
  });

  auto complete_non_kernel = [&](ComputeQueueRecord &queue, DispatchEntry &entry) {
    if (!execute_aql_pm4(queue, entry, now))
      return false;
    const uint32_t queue_id = entry.queue_id;
    const uint32_t process_id = entry.process_id;
    const uint32_t dispatch_id = entry.dispatch_id;
    entry.completed_wgs = entry.total_wgs;
    const VmAccessOutcome outcome =
        completion_ ? completion_->complete_non_kernel(entry) : VmAccessOutcome::Complete;
    if (outcome == VmAccessOutcome::Unavailable) {
      queue.publication_retry_pending = true;
      arm_stall_recheck(now);
      return false;
    }
    if (outcome != VmAccessOutcome::Complete) {
      notify_dispatch_vm_fault(queue_id, process_id, dispatch_id, outcome);
      return false;
    }
    ++queue.next_dispatch_idx;
    return true;
  };

  bool completion_drain_failed = false;
  auto run_dispatch_workers = [&]() {
    if (exec_mode_ != simdojo::ExecMode::FUNCTIONAL || dispatch_threads_ <= 1 ||
        pooled_due_ticks_.empty())
      return FunctionalQuantumResult{};

    lock.unlock();
    FunctionalQuantumResult result = run_active_cus_once(now);
    lock.lock();

    drain_pending_cluster_barrier_completions();
    drain_pending_wg_completions();
    if (!drain_completions())
      completion_drain_failed = true;
    return result;
  };

  // Dispatch every runnable queue, then execute at most one pooled quantum.
  // A rescan catches packets published while this handler was running.
  bool rescan = true;
  bool process_refetched_entries = false;
  bool yield_to_event_loop = false;
  while (rescan && !completion_drain_failed) {
    bool progress = true;
    while (progress && !yield_to_event_loop && !completion_drain_failed) {
      progress = false;

      for (ComputeQueueRecord &queue : compute_queues_) {
        if (queue.faulted || queue.suspended() || queue.publication_retry_pending)
          continue;

        while (queue.next_dispatch_idx < queue.entries.size()) {
          DispatchEntry &entry = queue.entries[queue.next_dispatch_idx];
          if (entry.wait_for_predecessors && !barrier_satisfied(queue, queue.next_dispatch_idx))
            break;

          if (entry.is_non_kernel()) {
            if (!complete_non_kernel(queue, entry))
              break;
            if (!drain_completions()) {
              completion_drain_failed = true;
              break;
            }
            if (queue.publication_retry_pending)
              break;
            progress = true;
            continue;
          }

          const uint32_t dispatch_id = entry.dispatch_id;
          bool backpressure = false;
          for (;;) {
            if (queue.next_dispatch_idx >= queue.entries.size())
              break;
            DispatchEntry &current = queue.entries[queue.next_dispatch_idx];
            if (current.dispatch_id != dispatch_id)
              break;

            const uint32_t queue_id = current.queue_id;
            const uint32_t process_id = current.process_id;
            const DispatchWorkgroupResult result = dispatch_workgroups(current);
            if (result.outcome != VmAccessOutcome::Complete) {
              notify_dispatch_vm_fault(queue_id, process_id, dispatch_id, result.outcome);
              backpressure = true;
              break;
            }
            if (result.dispatched > 0)
              progress = true;

            if (queue.next_dispatch_idx >= queue.entries.size())
              break;
            DispatchEntry &post = queue.entries[queue.next_dispatch_idx];
            if (post.dispatch_id != dispatch_id)
              break;
            if (post.fully_dispatched()) {
              ++queue.next_dispatch_idx;
              break;
            }
            if (result.dispatched == 0) {
              backpressure = true;
              break;
            }
          }
          if (backpressure)
            break;
        }

        if (completion_drain_failed)
          break;
      }

      if (!yield_to_event_loop && !completion_drain_failed) {
        const FunctionalQuantumResult worker_result = run_dispatch_workers();
        if (worker_result.ran) {
          progress = true;
          yield_to_event_loop = true;
        }
      }
    }

    if (completion_drain_failed)
      return;

    drain_pending_cluster_barrier_completions();
    drain_pending_wg_completions();
    if (!drain_completions())
      return;

    // Refill resources released by the completed pool batch, but leave their
    // execution for the next continuation event.
    if (yield_to_event_loop) {
      process_queues();
      if (!drain_completions())
        return;
    }

    util::Logger::cp([&](auto &os) {
      size_t remaining = 0;
      for (const ComputeQueueRecord &queue : compute_queues_)
        remaining += queue.entries.size();
      uint32_t active_cus = 0;
      for (const ComputeUnitCore *cu : cus_)
        if (cu->has_active_wfs())
          ++active_cus;
      os << std::format("{}: PHASE1_DONE remaining={} active_cus={}/{}", name(), remaining,
                        active_cus, cus_.size());
      for (size_t queue_index = 0; queue_index < compute_queues_.size(); ++queue_index) {
        ComputeQueueRecord &queue = compute_queues_[queue_index];
        if (queue.entries.empty())
          continue;
        os << std::format("\\n  queue[{}] entries={} next_disp={} implicit_barrier={}", queue_index,
                          queue.entries.size(), queue.next_dispatch_idx,
                          queue.implicit_barrier_next);
        for (size_t entry_index = 0; entry_index < queue.entries.size(); ++entry_index) {
          const DispatchEntry &entry = queue.entries[entry_index];
          os << std::format(
              "\\n    [{}] d={} qid={} total_wgs={} disp={} comp={} wait_pred={} sig={:#x} "
              "non_kern={}",
              entry_index, entry.dispatch_id, entry.queue_id, entry.total_wgs, entry.dispatched_wgs,
              entry.completed_wgs, entry.wait_for_predecessors, entry.completion_signal,
              entry.is_non_kernel());
        }
      }
    });

    if (yield_to_event_loop)
      break;

    const uint32_t dispatch_id_before_refetch = next_dispatch_id_;
    if (!process_refetched_entries) {
      for (ComputeQueueRecord &queue : compute_queues_)
        if (!queue.publication_retry_pending)
          fetch_from_queue(queue, now);
    }
    process_refetched_entries = false;

    for (ComputeQueueRecord &queue : compute_queues_) {
      if (queue.faulted || queue.suspended() || queue.publication_retry_pending)
        continue;
      while (queue.next_dispatch_idx < queue.entries.size()) {
        DispatchEntry &entry = queue.entries[queue.next_dispatch_idx];
        if (entry.wait_for_predecessors && !barrier_satisfied(queue, queue.next_dispatch_idx))
          break;
        if (!entry.is_non_kernel())
          break;
        if (!complete_non_kernel(queue, entry))
          break;
      }
    }
    if (!drain_completions())
      return;

    rescan = next_dispatch_id_ != dispatch_id_before_refetch;
    // A successful tail refetch must receive one execution pass, but that pass
    // must not immediately refetch the same producer pointer a third time. A
    // later producer notification schedules its own event.
    process_refetched_entries = rescan;
  }

  arm_grid_wait_recheck();

  const bool kfd = has_kfd_queues();
  if (!is_primary_ && pending_entries() > 0 && !kfd) {
    engine()->register_as_primary();
    is_primary_ = true;
  }

  if (dispatch_threads_ > 1) {
    const simdojo::Tick next = next_pooled_due_tick();
    if (next != simdojo::TICK_MAX)
      arm_dispatch_continuation(next);
    else
      cancel_dispatch_continuation();
  } else {
    for (size_t i = 0; i < cus_.size(); ++i) {
      if (!cus_[i]->is_idle()) {
        if (dispatch_ports_[i]->link())
          dispatch_ports_[i]->send(std::make_unique<simdojo::Message>(simdojo::MessageHeader{}));
        else
          cus_[i]->schedule_work();
      }
    }
  }

  const bool drm_done = std::ranges::all_of(compute_queues_, [](const auto &queue) {
    return queue.dispatches.entries.empty() && queue.commands.submissions.empty();
  });
  const bool all_done = drm_done && completion_ && completion_->all_complete(compute_queues_);
  const bool should_release = all_done && is_primary_ && !kfd;

  util::Logger::cp([&](auto &os) {
    os << std::format("{}: TEARDOWN_CHECK all_done={} kfd={} primary={} release={}", name(),
                      all_done, kfd, is_primary_.load(), should_release);
  });

  lock.unlock();

  if (should_release) {
    stop_doorbell_monitor();
    engine()->primary_release();
    is_primary_ = false;
  }
}

void CommandProcessor::flush_gpu_caches() {
  // Both L1 caches are write-through, so discard their clean snapshots around
  // direct backing writes. Flush dirty L2 data before the direct write so a
  // later L2 flush cannot overwrite it.
  for (auto *cu : cus_)
    cu->l1_scalar().invalidate_all();
  for (auto *l2 : l2_caches_)
    l2->flush_all();
  for (auto *cu : cus_) {
    cu->l1_vector().invalidate_all();
    // A direct backing write may land on code, and the I$ is not coherent with
    // data writes any more than the hardware one is.
    cu->instruction_cache().invalidate_all();
  }
}

} // namespace amdgpu
} // namespace rocjitsu
