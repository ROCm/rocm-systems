// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/code/kernel_symbol.h"
#include "rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/hsa_clock.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/pm4_queue_binding_factory.h"
#include "rocjitsu/vm/amdgpu/pm4_queue_controller.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/amd_hsa_queue.h"
RJ_DIAGNOSTIC_POP

#include "simdojo/sim/message.h"
#include "simdojo/sim/simulation.h"
#include "util/bit.h"
#include "util/log.h"

#include <algorithm>
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
    if (!aql_queues_.empty() || active_queue_registrations_ != 0)
      throw std::logic_error("cannot replace the command processor VM while AQL queues exist");
    if (pm4_queue_controller_ && pm4_queue_controller_->active_queues() != 0) {
      throw std::logic_error("cannot replace the command processor VM while PM4 queues exist");
    }
    gpu_vm_ = gpu_vm;
    default_address_space_ = gpu_vm_ ? default_address_space : AddressSpaceHandle{};
    pm4_queue_controller_ = gpu_vm_ ? std::make_unique<Pm4QueueController>(*gpu_vm_) : nullptr;
  }
  for (ComputeUnitCore *cu : cus_)
    cu->set_gpu_vm(gpu_vm_);
}

std::shared_ptr<QueueBindingFactory>
CommandProcessor::make_pm4_queue_binding_factory(Pm4PacketCallbacks callbacks) {
  return amdgpu::make_pm4_queue_binding_factory(*this, std::move(callbacks));
}

uint64_t CommandProcessor::register_pm4_queue(Pm4QueueConfig config) {
  std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
  if (pm4_queue_controller_ == nullptr)
    return 0;
  return pm4_queue_controller_->attach(std::move(config));
}

bool CommandProcessor::unregister_pm4_queue_registration(uint64_t registration_id) noexcept {
  std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
  return pm4_queue_controller_ != nullptr && pm4_queue_controller_->detach(registration_id);
}

QueuePrepareCloseStatus
CommandProcessor::prepare_unregister_pm4_queue_registration(uint64_t registration_id) noexcept {
  std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
  if (pm4_queue_controller_ == nullptr)
    return QueuePrepareCloseStatus::Faulted;
  const QueuePrepareCloseStatus status = pm4_queue_controller_->prepare_detach(registration_id);
  if (status == QueuePrepareCloseStatus::Busy) {
    if (engine())
      engine()->schedule_event_now(&doorbell_event_);
    else
      (void)pm4_queue_controller_->service();
  }
  return status;
}

QueueReconfigureStatus
CommandProcessor::update_pm4_queue_registration(uint64_t registration_id,
                                                const QueueReconfigureRequest &request) {
  std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
  if (pm4_queue_controller_ == nullptr)
    return QueueReconfigureStatus::Stale;
  return pm4_queue_controller_->update(registration_id, request);
}

QueueSubmissionStatus CommandProcessor::notify_pm4_queue_doorbell(uint64_t registration_id,
                                                                  uint64_t producer_cursor) {
  std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
  if (pm4_queue_controller_ == nullptr)
    return QueueSubmissionStatus::Faulted;
  const QueueSubmissionStatus status =
      pm4_queue_controller_->notify(registration_id, producer_cursor);
  if (status != QueueSubmissionStatus::Accepted)
    return status;
  if (engine()) {
    engine()->schedule_event_now(&doorbell_event_);
  } else {
    // Unit-level CP fixtures without a SimulationEngine still exercise the
    // same controller. Production frontends always take the event path above.
    if (pm4_queue_controller_->service())
      return QueueSubmissionStatus::Retry;
  }
  return status;
}

size_t CommandProcessor::registered_pm4_queue_count_for_test() const {
  std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
  return pm4_queue_controller_ ? pm4_queue_controller_->active_queues() : 0;
}

bool CommandProcessor::has_registered_queues() const {
  std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
  return active_queue_registrations_ != 0 || !aql_queues_.empty() ||
         (pm4_queue_controller_ && pm4_queue_controller_->active_queues() != 0);
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
  return std::any_of(cus.begin(), cus.end(), [](const auto *cu) { return cu->has_active_wfs(); });
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

      uint32_t reserved_wgs = planned_per_cu[cu_idx] + 1;
      uint64_t reserved_wfs = static_cast<uint64_t>(entry.wfs_per_workgroup) * reserved_wgs;
      uint64_t reserved_lds = static_cast<uint64_t>(lds_bytes_per_wg) * reserved_wgs;
      if (reserved_wfs > kU32Max || reserved_lds > kU32Max)
        continue;
      if (!cu->can_accept_workgroup(static_cast<uint32_t>(reserved_wfs),
                                    static_cast<uint32_t>(reserved_lds)))
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
    const simdojo::Tick next = next_pooled_due_tick(now);
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
  if (kcp != 0) {
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
        const AtomicLoadResult loaded =
            read_gpu_u32(pkt.address_space, preload_addr + preload_index * 4);
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
  uint32_t grid_wg_id_x = global_wg_id % gx;
  uint32_t wg_id_y = (global_wg_id / gx) % gy;
  uint32_t wg_id_z = global_wg_id / (gx * gy);
  uint32_t wg_id_x = (pkt.enable_wg_id_y || pkt.enable_wg_id_z) ? grid_wg_id_x : global_wg_id;

  // System SGPRs: workgroup_id_{x,y,z} placed sequentially after user SGPRs.
  // Only the IDs whose enable bits are set in compute_pgm_rsrc2 are written.
  // When kernel_code_properties is 0 (internal test dispatches), always write
  // workgroup_id_x as a fallback since internal kernels expect it.
  uint32_t sys_idx = pkt.num_user_sgprs;
  {
    bool kcp_zero = (pkt.kernel_code_properties == 0);
    if (pkt.enable_wg_id_x || kcp_zero)
      cu->write_sgpr(sbase + sys_idx++, wg_id_x);
    if (pkt.enable_wg_id_y)
      cu->write_sgpr(sbase + sys_idx++, wg_id_y);
    if (pkt.enable_wg_id_z)
      cu->write_sgpr(sbase + sys_idx++, wg_id_z);
  }
  const auto properties = isa_properties(cu->arch());
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
    // Round the per-wave region to the 1 KB COMPUTE_TMPRING_SIZE.WAVESIZE granule
    // so that each wave's base equals scratch_pool + scoreboard_id * wavesize,
    // which is exactly what rocm-dbgapi computes to locate a wave's private
    // memory (rocdbgapi architecture.cpp scratch_memory_region).
    uint64_t raw_per_wave = static_cast<uint64_t>(pkt.private_segment_fixed_size) * wf->wf_size();
    uint64_t per_wave_size = ((raw_per_wave + 1023) / 1024) * 1024;
    uint32_t wg_total_size = static_cast<uint32_t>(pkt.workgroup_size_x) *
                             std::max<uint16_t>(1, pkt.workgroup_size_y) *
                             std::max<uint16_t>(1, pkt.workgroup_size_z);
    uint32_t waves_per_wg = (wg_total_size + wf->wf_size() - 1) / wf->wf_size();
    uint64_t global_wave_idx = static_cast<uint64_t>(global_wg_id) * waves_per_wg + wf_index_in_wg;
    uint64_t scratch_slot = global_wave_idx;
    if (cu->arch() == ROCJITSU_CODE_ARCH_CDNA5) {
      const uint32_t shader_engine_count =
          std::max(scratch_wave_divisor_, scratch_shader_engine_count_);
      const uint32_t shader_engine_id = wf->shader_engine_id();
      const uint32_t scoreboard_id = wf->scratch_scoreboard_id();
      assert(shader_engine_id < shader_engine_count);
      assert(scoreboard_id < scratch_waves_per_se_);
      scratch_slot =
          (static_cast<uint64_t>(scratch_xcc_id_) * shader_engine_count + shader_engine_id) *
              scratch_waves_per_se_ +
          scoreboard_id;
    } else {
      // Legacy CWSR records use the dispatch-wide logical scratch slot.
      wf->set_scratch_scoreboard_id(static_cast<uint32_t>(global_wave_idx));
    }
    if (scratch_slot > (std::numeric_limits<uint64_t>::max() - scratch_pool) / per_wave_size)
      return VmAccessOutcome::Malformed;
    uint64_t wave_scratch = scratch_pool + scratch_slot * per_wave_size;
    std::optional<GpuVmAccess> scratch_access = snapshot_gpu_access(pkt.address_space);
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

    if (scratch_outcome != VmAccessOutcome::Complete && scratch_allocator_) {
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
      if (!scratch_allocator_(pkt.process_id, scratch_pool, total_scratch))
        return VmAccessOutcome::Faulted;
      scratch_access = snapshot_gpu_access(pkt.address_space);
      scratch_outcome = scratch_access ? scratch_range_outcome(true) : VmAccessOutcome::Unavailable;
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
  completion_->set_dispatch_retired_callback([this](const DispatchEntry &entry) {
    erase_cluster_workgroups(entry.dispatch_id);
    dispatch_launch_metadata_.erase(entry.dispatch_id);
  });
  completion_->set_grid_retired_callback([this](const DispatchEntry &) { wake_all_xcds(); });
}

void CommandProcessor::shutdown() {
  stop_doorbell_monitor();
  retry_event_pending_.store(false, std::memory_order_release);
  if (is_primary_ && engine()) {
    engine()->primary_release();
    is_primary_ = false;
  }
  {
    std::unique_lock<std::shared_mutex> structure_lock(queue_structure_mutex_);
    std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
    aql_queues_.clear();
    dispatch_launch_metadata_.clear();
  }
  if (pm4_queue_controller_)
    pm4_queue_controller_->close_all();
  {
    std::lock_guard lock(fanout_inbox_mutex_);
    fanout_inbox_.clear();
    fanout_launch_metadata_inbox_.clear();
  }
  {
    std::lock_guard lock(dispatch_fault_inbox_mutex_);
    dispatch_fault_inbox_.clear();
  }
  {
    std::lock_guard lock(doorbell_inbox_mutex_);
    doorbell_inbox_.clear();
  }
  completion_.reset();
  pm4_queue_controller_.reset();
  gpu_vm_ = nullptr;
  default_address_space_ = {};
  for (ComputeUnitCore *cu : cus_)
    cu->set_gpu_vm(nullptr);
}

void CommandProcessor::set_xcd_topology(uint32_t rank, std::vector<CommandProcessor *> peers) {
  assert(rank < peers.size() && "XCD rank must index its own SoC's CP list");
  assert(peers[rank] == this && "XCD rank must be this CP's own position");
  xcd_rank_ = rank;
  xcd_peers_ = std::move(peers);
  // Carve this XCD its own dispatch-id space; see allocate_dispatch_id().
  dispatch_id_stride_ = static_cast<uint32_t>(xcd_peers_.size());
  dispatch_id_base_ = 1 + rank;
  next_dispatch_id_ = dispatch_id_base_;
}

AqlQueueRecord *CommandProcessor::find_aql_queue(uint32_t queue_id, uint32_t process_id) {
  for (AqlQueueRecord &queue : aql_queues_) {
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
    const std::vector<AqlQueueRecord>::iterator queue =
        std::ranges::find_if(aql_queues_, [&](const AqlQueueRecord &candidate) {
          return candidate.queue_id == shard.queue_id && candidate.process_id == shard.process_id;
        });
    if (queue == aql_queues_.end()) {
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

  const uint32_t grid_wgs = dp.total_wgs;
  auto grid = std::make_shared<GridCompletion>();
  grid->grid_wgs = grid_wgs;

  for (uint32_t rank = 0; rank < num_xcds; ++rank) {
    if (rank == xcd_rank_)
      continue;
    DispatchEntry shard = dp;
    shard.grid_completion = grid;
    shard.fanout_peer = true;
    // The peer must not fire the dispatch's completion signal; the owning XCD
    // does that once the grid counter shows every share retired.
    shard.completion_signal = 0;
    shard.apply_shard(XcdShard(rank, num_xcds));
    xcd_peers_[rank]->accept_fanout_shard(std::move(shard), launch_metadata);
  }

  dp.grid_completion = std::move(grid);
  dp.apply_shard(XcdShard(xcd_rank_, num_xcds));
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

uint64_t CommandProcessor::register_queue(AqlQueueConfig config) {
  return register_queue(std::move(config), false);
}

uint64_t CommandProcessor::register_queue(AqlQueueConfig config, bool fanout_replica) {
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
  if (!valid_aql_queue_layout(config.ring_base_va, config.ring_size, config.read_ptr_va,
                              config.write_ptr_va)) {
    return 0;
  }
  // A host-polled queue may be registered before its doorbell page is mapped;
  // set_process_doorbell_base() publishes that mapping later. VM polling has no
  // equivalent deferred binding and therefore requires an address up front.
  if (config.doorbell_mode == QueueDoorbellMode::VmPolled && config.doorbell_va == 0) {
    return 0;
  }
  std::optional<GpuVmBindingLease> address_space_lease =
      registration_vm->retain_binding(config.address_space);
  if (!address_space_lease || address_space_lease->info().vmid != config.process_id)
    return 0;
  AqlQueueRecord queue(std::move(config));
  queue.address_space_lease = std::move(*address_space_lease);
  queue.fanout_replica = fanout_replica;
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
    explicit OwnerRegistrationRollback(std::vector<AqlQueueRecord> &queues)
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
    std::vector<AqlQueueRecord> &queues_;
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
    if (find_aql_queue(queue.queue_id, queue.process_id) != nullptr)
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
      AqlQueueConfig replica = queue;
      replica.xcd_fanout = false;
      if (xcd_peers_[rank]->register_queue(std::move(replica), true) == 0)
        return 0;
      registered_peer_ranks.push_back(rank);
    }
  }
  {
    std::unique_lock<std::shared_mutex> structure_lock(queue_structure_mutex_);
    std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
    if (find_aql_queue(queue.queue_id, queue.process_id) != nullptr)
      return 0;
    OwnerRegistrationRollback owner_rollback(aql_queues_);
    aql_queues_.push_back(std::move(queue));
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
    const std::vector<AqlQueueRecord>::iterator queue = std::ranges::find(
        aql_queues_, notification.registration_id, &AqlQueueRecord::registration_id);
    if (queue != aql_queues_.end())
      queue->last_doorbell = notification.value;
  }
}

bool CommandProcessor::signal_queue_exception(uint32_t queue_id, uint32_t process_id,
                                              uint64_t status) {
  uint64_t exception_status_va = 0;
  uint32_t exception_event_id = 0;
  AddressSpaceHandle address_space;
  InterruptSink interrupt_sink;
  {
    std::lock_guard<std::recursive_mutex> lk(hw_queue_mutex_);
    auto queue =
        std::find_if(aql_queues_.begin(), aql_queues_.end(), [&](const AqlQueueRecord &candidate) {
          return candidate.queue_id == queue_id && candidate.process_id == process_id;
        });
    if (queue == aql_queues_.end() || queue->exception_status_va == 0)
      return false;
    exception_status_va = queue->exception_status_va;
    exception_event_id = queue->exception_event_id;
    address_space = queue->address_space;
    interrupt_sink = queue->interrupt_sink;
  }

  if (write_gpu_block(address_space, exception_status_va, &status, sizeof(status)) !=
      VmAccessOutcome::Complete) {
    return false;
  }
  interrupt_sink.deliver(process_id, exception_event_id);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
  AtomicLoadResult exception_status = read_gpu_u64(address_space, exception_status_va);
  while (exception_status.outcome == VmAccessOutcome::Complete &&
         exception_status.value == status && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
    exception_status = read_gpu_u64(address_space, exception_status_va);
  }

  for (auto *cu : cus_) {
    cu->with_wave_state_locked([&] {
      for (uint32_t slot = 0; slot < cu->num_wf_slots(); ++slot) {
        auto *wave = cu->wf(slot);
        if (!wave->is_halted() && wave->process_id() == process_id && wave->queue_id() == queue_id)
          wave->set_debug_suspended(true);
      }
    });
  }
  return true;
}

QueuePrepareCloseStatus
CommandProcessor::prepare_unregister_queue_registration(uint64_t registration_id) noexcept {
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
    const std::vector<AqlQueueRecord>::iterator queue =
        std::ranges::find(aql_queues_, registration_id, &AqlQueueRecord::registration_id);
    if (queue == aql_queues_.end())
      return QueuePrepareCloseStatus::Faulted;
    if (!force) {
      if (queue->publication_faulted)
        return QueuePrepareCloseStatus::Faulted;
      retry_close = queue->read_pointer_journal.publication_pending() ||
                    queue->publication_retry_pending || queue->idle_publication.active() ||
                    !queue->entries.empty();
    }
    if (!retry_close) {
      queue_id = queue->queue_id;
      process_id = queue->process_id;
      for (auto *cu : cus_) {
        cu->with_wave_state_locked([&] {
          for (uint32_t slot = 0; slot < cu->num_wf_slots(); ++slot) {
            auto *wave = cu->wf(slot);
            if (!wave->is_halted() && wave->process_id() == process_id &&
                wave->queue_id() == queue_id)
              wave->halt();
          }
        });
      }
      drop_replicas = queue->xcd_fanout;
      for (const DispatchEntry &entry : queue->entries)
        dispatch_launch_metadata_.erase(entry.dispatch_id);
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
      aql_queues_.erase(queue);
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
    const AqlQueueRecord *queue = find_aql_queue(queue_id, process_id);
    if (queue != nullptr)
      registration_id = queue->registration_id;
  }
  (void)unregister_queue_registration(registration_id);
}

bool CommandProcessor::update_queue(uint32_t queue_id, uint32_t process_id, uint64_t ring_base_va,
                                    uint32_t ring_size, uint32_t queue_percentage) {
  uint64_t registration_id = 0;
  {
    std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
    const AqlQueueRecord *queue = find_aql_queue(queue_id, process_id);
    if (queue != nullptr)
      registration_id = queue->registration_id;
  }
  return update_queue_registration(registration_id, ring_base_va, ring_size, queue_percentage);
}

bool CommandProcessor::update_queue_registration(uint64_t registration_id, uint64_t ring_base_va,
                                                 uint32_t ring_size, uint32_t queue_percentage) {
  if (registration_id == 0)
    return false;
  if (!valid_aql_packet_ring(ring_base_va, ring_size))
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
    for (auto &q : aql_queues_) {
      if (q.registration_id == registration_id) {
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
        if (changed && !suspended && !q.debug_suspended)
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
        if (!wave->is_halted() && wave->process_id() == process_id && wave->queue_id() == queue_id)
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
  if (wake_command_processor && engine())
    engine()->schedule_event_now(&doorbell_event_);
  return found;
}

void CommandProcessor::set_queue_debug_suspended(uint32_t queue_id, uint32_t process_id,
                                                 bool suspended) {
  bool wake_command_processor = false;
  {
    std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
    for (AqlQueueRecord &q : aql_queues_) {
      if (q.queue_id == queue_id && q.process_id == process_id) {
        if (q.debug_suspended == suspended)
          continue;
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
        } else if (!q.runtime_suspended) {
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
  for (auto &q : aql_queues_) {
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
  for (auto &q : aql_queues_) {
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
        for (size_t queue_index = 0; queue_index < aql_queues_.size(); ++queue_index) {
          // A replica does not own the queue, so it must not report it idle: its
          // shards drain before the owner's and the same KFD queue would otherwise
          // raise this from several CPs at once.
          if (aql_queues_[queue_index].fanout_replica)
            continue;
          if (aql_queues_[queue_index].entries.empty() &&
              aql_queues_[queue_index].process_id != 0) {
            idle_queues.emplace_back(aql_queues_[queue_index].interrupt_sink,
                                     aql_queues_[queue_index].process_id);
          }
        }
      }
      for (const auto &[interrupt_sink, process_id] : idle_queues)
        interrupt_sink.deliver(process_id, 0);
    }

    if (poll_count % 5000 == 1) {
      std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
      for (auto &q : aql_queues_) {
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

AqlQueueRecord *CommandProcessor::schedule_next_queue() {
  if (aql_queues_.empty())
    return nullptr;
  size_t start = next_queue_idx_;
  for (size_t i = 0; i < aql_queues_.size(); ++i) {
    size_t idx = (start + i) % aql_queues_.size();
    auto &qs = aql_queues_[idx];
    if (qs.faulted || qs.debug_suspended || qs.runtime_suspended)
      continue;
    if (qs.next_dispatch_idx < qs.entries.size()) {
      next_queue_idx_ = (idx + 1) % aql_queues_.size();
      return &qs;
    }
  }
  return nullptr;
}

bool CommandProcessor::barrier_satisfied(const AqlQueueRecord &qs, size_t idx) const {
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
      completion_->notify_wg_complete(completion.dispatch_id, completion.wg_id, aql_queues_);
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
// peer CU's LDS once the whole cluster is done. Runs with
// cluster_placements_mutex_ released: maybe_reset_lds_alloc() reaches the CU's
// wave-state lock, which is ordered ahead of it.
void CommandProcessor::release_cluster_lds_pins(
    const std::vector<std::pair<ComputeUnitCore *, uint64_t>> &unpin) {
  for (const auto &[cu, cluster_key] : unpin) {
    cu->unpin_lds_for_cluster(cluster_key);
    cu->maybe_reset_lds_alloc();
  }
}

void CommandProcessor::erase_cluster_workgroup(uint32_t dispatch_id, uint32_t wg_id) {
  // maybe_reset_lds_alloc() takes the CU's wave-state lock, so it runs after the
  // placements lock is dropped -- see cluster_placements_mutex_.
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
    // first workgroup claim the report.
    if (!entry.execution_begun) {
      entry.execution_begun = true;
      if (!entry.grid_completion || entry.grid_completion->claim_execution_begin())
        plugin_group_->onAmdgpuDispatchExecutionBegin(entry.dispatch_id);
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
    std::vector<Wavefront *> wg_wavefronts;
    wg_wavefronts.reserve(entry.wfs_per_workgroup);
    const auto free_reserved = [&]() {
      for (auto *claimed : wg_wavefronts)
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
                                      entry.vgprs_per_wf, entry.kernel_wave_size);
      if (!wf) {
        assert(false && "dispatch_wf failed after placement was reserved");
        free_reserved();
        return VmAccessOutcome::Malformed;
      }
      wg_wavefronts.push_back(wf);
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
      wf->set_process_id(entry.process_id);
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
        return initialization;
      }
    }

    // All fallible per-wave work succeeded: commit WG bookkeeping and the cluster pin.
    cu->begin_workgroup(entry.dispatch_id, global_wg_id, entry.wfs_per_workgroup,
                        entry.num_named_barriers);
    register_cluster_workgroup(entry, local_wg_id, global_wg_id, cu, lds_base);

    plugin_group_->onAmdgpuWorkgroupDispatched(
        entry.dispatch_id, global_wg_id, cu->vgpr_allocation_block_size(),
        cu->sgpr_allocation_block_size(), std::span<Wavefront *>(wg_wavefronts));
    for (auto *wf : wg_wavefronts)
      plugin_group_->onAmdgpuWavefrontDispatched(*wf);

    ++entry.dispatched_wgs;
    ++dispatched;
    dispatched_workgroups_.fetch_add(1, std::memory_order_relaxed);
    return VmAccessOutcome::Complete;
  };

  while (entry.dispatched_wgs < entry.total_wgs) {
    if (entry.grid_faulted())
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
        if (cus_[cu_idx]->can_accept_workgroup(entry.wfs_per_workgroup,
                                               entry.group_segment_fixed_size)) {
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
    const CompletionDrainResult result = completion_->drain_completions(aql_queues_);
    retry_pending |= result.retry_pending;
    if (!result.terminal_fault)
      break;

    stop_processing = true;
    const CompletionDrainFault &fault = *result.terminal_fault;
    if (!fault.queue_idle) {
      const std::vector<AqlQueueRecord>::iterator queue =
          std::ranges::find_if(aql_queues_, [&](const AqlQueueRecord &candidate) {
            return candidate.queue_id == fault.queue_id && candidate.process_id == fault.process_id;
          });
      if (queue != aql_queues_.end())
        queue->publication_faulted = true;
      notify_dispatch_vm_fault(fault.queue_id, fault.process_id, fault.dispatch_id, fault.outcome);
      continue;
    }

    const std::vector<AqlQueueRecord>::iterator queue =
        std::ranges::find_if(aql_queues_, [&](const AqlQueueRecord &candidate) {
          return candidate.queue_id == fault.queue_id && candidate.process_id == fault.process_id;
        });
    if (queue != aql_queues_.end()) {
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
  const std::vector<AqlQueueRecord>::iterator queue =
      std::ranges::find_if(aql_queues_, [&](const AqlQueueRecord &candidate) {
        return candidate.queue_id == queue_id && candidate.process_id == process_id;
      });
  if (queue == aql_queues_.end())
    return false;
  AqlQueueRecord &state = *queue;

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

  // Retire any non-kernel entries (barrier-kind packets) that are now at
  // the head, then drain again so a dependent kernel behind them can proceed.
  for (AqlQueueRecord &qs : aql_queues_) {
    if (qs.faulted || qs.debug_suspended || qs.runtime_suspended || qs.publication_retry_pending)
      continue;
    while (qs.next_dispatch_idx < qs.entries.size()) {
      auto &e = qs.entries[qs.next_dispatch_idx];
      if (e.wait_for_predecessors && !barrier_satisfied(qs, qs.next_dispatch_idx))
        break;
      if (!e.is_non_kernel())
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
  for (AqlQueueRecord &qs : aql_queues_) {
    if (qs.faulted || qs.debug_suspended || qs.runtime_suspended || qs.publication_retry_pending)
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
  for (AqlQueueRecord &qs : aql_queues_) {
    if (qs.faulted || qs.debug_suspended || qs.runtime_suspended || qs.publication_retry_pending)
      continue;
    while (qs.next_dispatch_idx < qs.entries.size()) {
      auto &entry = qs.entries[qs.next_dispatch_idx];

      if (entry.wait_for_predecessors && !barrier_satisfied(qs, qs.next_dispatch_idx))
        break; // Stalled on barrier bit.

      if (entry.is_non_kernel()) {
        const uint32_t queue_id = entry.queue_id;
        const uint32_t process_id = entry.process_id;
        const uint32_t dispatch_id = entry.dispatch_id;
        entry.completed_wgs = entry.total_wgs; // 0 == 0, immediately complete.
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

bool CommandProcessor::has_runnable_cus() const {
  for (auto *cu : cus_) {
    if (cu->has_runnable_wfs())
      return true;
  }
  return false;
}

void CommandProcessor::refresh_pooled_due_ticks(simdojo::Tick now) {
  for (auto it = pooled_due_ticks_.begin(); it != pooled_due_ticks_.end();) {
    if (!it->first->has_runnable_wfs())
      it = pooled_due_ticks_.erase(it);
    else
      ++it;
  }
  for (auto *cu : cus_)
    if (cu->has_runnable_wfs())
      pooled_due_ticks_.try_emplace(cu, now + 1);
}

simdojo::Tick CommandProcessor::next_pooled_due_tick(simdojo::Tick now) {
  refresh_pooled_due_ticks(now);
  simdojo::Tick next = simdojo::TICK_MAX;
  for (const auto &[_, tick] : pooled_due_ticks_)
    next = std::min(next, tick);
  return next;
}

FunctionalQuantumResult CommandProcessor::run_active_cus_once(simdojo::Tick now) {
  refresh_pooled_due_ticks(now);
  active_cu_scratch_.clear();
  if (!spis_.empty()) {
    for (auto *spi : spis_)
      spi->append_active_cus(active_cu_scratch_);
  } else {
    for (auto *cu : cus_) {
      if (cu->has_runnable_wfs())
        active_cu_scratch_.push_back(cu);
    }
  }
  std::erase_if(active_cu_scratch_, [&](ComputeUnitCore *cu) {
    auto due = pooled_due_ticks_.find(cu);
    return !cu->has_runnable_wfs() || due == pooled_due_ticks_.end() || due->second > now;
  });

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

CommandProcessor::KernelDescriptorReadResult
CommandProcessor::read_kernel_descriptor(const GpuVmAccess &transaction_access,
                                         uint64_t kernel_object) const {
  using namespace rocr::llvm::amdhsa;
  kernel_descriptor_t kd{};
  const VmAccessOutcome outcome =
      read_gpu_block(transaction_access, kernel_object, &kd, sizeof(kd));
  return {.outcome = outcome, .descriptor = kd};
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
    const hsa_kernel_dispatch_packet_t &pkt, AqlQueueRecord &queue,
    const GpuVmAccess &transaction_access, uint64_t pkt_addr, uint32_t queue_packet_id,
    uint64_t aql_packet_id, ClusterDispatchShape cluster_shape) {
  const bool uses_kfd_queue_abi = queue.uses_kfd_queue_abi;
  using namespace rocr::llvm::amdhsa;
  const KernelDescriptorReadResult descriptor =
      read_kernel_descriptor(transaction_access, pkt.kernel_object);
  if (descriptor.outcome != VmAccessOutcome::Complete)
    return admission_from_vm_outcome(descriptor.outcome);
  const kernel_descriptor_t &kd = descriptor.descriptor;
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

    const uint32_t private_segment_fixed_size =
        std::max(kd.private_segment_fixed_size, pkt.private_segment_size);
    if (private_segment_fixed_size > 0) {
      const uint64_t scratch_loc_va =
          queue_ptr + offsetof(amd_queue_t, scratch_backing_memory_location);
      const AtomicLoadResult loaded = read_gpu_u64(transaction_access, scratch_loc_va);
      if (loaded.outcome != VmAccessOutcome::Complete)
        return admission_from_vm_outcome(loaded.outcome);
      scratch_backing_addr = loaded.value;
    }
  }

  DispatchEntry dp{};
  dp.queue_id = queue.queue_id;
  dp.queue_packet_id = queue_packet_id;
  dp.address_space = queue.address_space;
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
  dp.initial_mode_raw = initial_mode_from_compute_pgm_rsrc1(kd.compute_pgm_rsrc1, arch);
  dp.private_segment_fixed_size = std::max(kd.private_segment_fixed_size, pkt.private_segment_size);
  dp.group_segment_fixed_size = std::max(kd.group_segment_fixed_size, pkt.group_segment_size);
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
      if (dp.scratch_backing_addr == 0 && scratch_resolver_)
        dp.scratch_backing_addr = scratch_resolver_(queue.process_id);

      // Publish the scratch backing location and COMPUTE_TMPRING_SIZE into the
      // ABI-stable part of amd_queue_t so rocm-dbgapi can compute each wave's
      // private (scratch) memory region, letting ROCgdb read scratch-resident
      // variables. Real CP firmware populates these when it assigns scratch to
      // the queue; the emulator's ROCr instead sets the backing via
      // SET_SCRATCH_BACKING_VA and leaves these fields zero, so the CP fills
      // them here. Field layout per rocdbgapi architecture.cpp
      // scratch-memory region. The WAVES field is common, while ISA properties
      // describe the generation-specific WAVESIZE unit and field width.
      if (dp.scratch_backing_addr != 0 && !cus_.empty()) {
        uint64_t per_wave_bytes =
            static_cast<uint64_t>(dp.private_segment_fixed_size) * cus_[0]->wf_size();
        // setup_wavefront() allocates scratch slots at a 1 KiB boundary. Encode
        // that actual stride, rather than merely rounding to the register's
        // unit, so flat_scratch agrees with rocm-dbgapi for every scoreboard
        // slot after slot zero.
        const uint64_t per_wave_stride = ((per_wave_bytes + 1023) / 1024) * 1024;
        const auto properties = isa_properties(arch);
        const uint32_t wavesize_unit = properties.compute_tmpring_wavesize_granule;
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
  dispatch_info.grid_size_x = pkt.grid_size_x;
  dispatch_info.grid_size_y = pkt.grid_size_y;
  dispatch_info.grid_size_z = pkt.grid_size_z;
  dispatch_info.workgroup_size_x = pkt.workgroup_size_x;
  dispatch_info.workgroup_size_y = pkt.workgroup_size_y;
  dispatch_info.workgroup_size_z = pkt.workgroup_size_z;
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
  if (queue.xcd_fanout)
    fan_out_dispatch(dp, launch_metadata);

  queue.push_entry(std::move(dp));
  return {.status = AqlAdmissionStatus::Complete};
}

AqlAdmissionResult CommandProcessor::admit_aql_packet(const AqlPacketProcessRequest &request,
                                                      AqlPreparedPacket prepared) {
  const std::vector<AqlQueueRecord>::iterator queue =
      std::ranges::find(aql_queues_, request.registration_id, &AqlQueueRecord::registration_id);
  if (queue == aql_queues_.end() || queue->process_id != request.process_id ||
      queue->queue_id != request.queue_id || queue->faulted || queue->fanout_replica) {
    return {.status = AqlAdmissionStatus::Malformed};
  }

  if (prepared.kind == AqlPreparedPacketKind::KernelDispatch) {
    return admit_kernel_dispatch(prepared.kernel_dispatch, *queue, request.access,
                                 request.packet_address, request.ring_slot, request.packet_index,
                                 prepared.cluster_shape);
  }

  DispatchEntry entry{
      .dispatch_id = allocate_dispatch_id(),
      .queue_id = queue->queue_id,
      .address_space = queue->address_space,
      .interrupt_sink = queue->interrupt_sink,
      .process_id = queue->process_id,
      .completion_signal = prepared.completion_signal,
      .kind = DispatchPacketKind::NonKernel,
      .wait_for_predecessors = prepared.barrier_bit,
  };
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
  for (const auto &qs : aql_queues_) {
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
  schedule_event(&doorbell_event_, now + stall_recheck_backoff_);
  stall_recheck_backoff_ = std::min(stall_recheck_backoff_ * 2, kMaxStallRecheckBackoff);
}

void CommandProcessor::fetch_from_queue(AqlQueueRecord &queue, simdojo::Tick now) {
  // A replica's work arrives as dispatch shards, not from the ring. Reading the
  // ring here would also advance a read pointer the owning XCD owns, and its
  // suspension flags are the owner's copy rather than state this CP maintains.
  if (queue.fanout_replica)
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

  // One queue-service transaction observes one immutable address-space
  // generation. A root replacement may affect the next retry, but it cannot
  // splice new translations into queue-pointer, ring, dependency, admission,
  // or consumer-pointer accesses that belong to this attempt.
  std::optional<GpuVmAccess> transaction_access = snapshot_gpu_access(queue.address_space);
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
  if (queue.debug_suspended || queue.runtime_suspended) {
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
    if (doorbell != std::numeric_limits<uint64_t>::max()) {
      uint64_t doorbell_limit = doorbell + 1;
      if (doorbell_limit < process_limit)
        process_limit = doorbell_limit;
    }
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

    hsa_kernel_dispatch_packet_t pkt{};
    const VmAccessOutcome packet_read = read_gpu_block(access, pkt_addr, &pkt, kAqlPacketBytes);
    if (packet_read != VmAccessOutcome::Complete) {
      process_limit = read_idx;
      if (packet_read == VmAccessOutcome::Unavailable)
        arm_stall_recheck(now);
      else
        queue.faulted = true;
      break;
    }

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
    });
    const PacketProcessResult &packet_result = result.packet_result();
    if (!valid_packet_process_result(packet_result, packet_bytes.size(), kAqlPacketBytes))
      throw std::logic_error("AQL processor returned an invalid common packet result");

    if (packet_result.status == PacketProcessStatus::Complete) {
      if (packet_result.retirement_bytes != kAqlPacketBytes) {
        throw std::logic_error("AQL processor completed without retiring exactly one packet");
      }
      ++read_idx;
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

  queue.read_pointer_journal.retire(process_limit, std::move(*transaction_access));
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

  // PM4 owns its queue lock and executes VM/register callbacks without the AQL
  // lock, so service it before serializing AQL admission and completion.
  const bool pm4_needs_retry = pm4_queue_controller_ && pm4_queue_controller_->service();

  std::unique_lock<std::recursive_mutex> lock(hw_queue_mutex_);
  if (pm4_needs_retry)
    arm_stall_recheck(now);
  util::Logger::cp(
      [&](auto &os) { os << std::format("{}: DOORBELL queues={}", name(), aql_queues_.size()); });

  // Finish a prior durable publication before admitting a new queue generation.
  if (!drain_completions())
    return;

  size_t entries_before = 0;
  for (const AqlQueueRecord &queue : aql_queues_)
    entries_before += queue.entries.size();

  for (AqlQueueRecord &queue : aql_queues_)
    if (!queue.publication_retry_pending)
      fetch_from_queue(queue, now);

  size_t entries_after = 0;
  for (const AqlQueueRecord &queue : aql_queues_)
    entries_after += queue.entries.size();
  if (entries_after != entries_before)
    stall_recheck_backoff_ = 1;
  util::Logger::cp([&](auto &os) {
    os << std::format("{}: FETCHED {} new entries (total={})", name(),
                      entries_after - entries_before, entries_after);
  });

  auto complete_non_kernel = [&](AqlQueueRecord &queue, DispatchEntry &entry) {
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
        !has_runnable_cus())
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

      for (AqlQueueRecord &queue : aql_queues_) {
        if (queue.faulted || queue.debug_suspended || queue.runtime_suspended ||
            queue.publication_retry_pending)
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
      for (const AqlQueueRecord &queue : aql_queues_)
        remaining += queue.entries.size();
      uint32_t active_cus = 0;
      for (const ComputeUnitCore *cu : cus_)
        if (cu->has_active_wfs())
          ++active_cus;
      os << std::format("{}: PHASE1_DONE remaining={} active_cus={}/{}", name(), remaining,
                        active_cus, cus_.size());
      for (size_t queue_index = 0; queue_index < aql_queues_.size(); ++queue_index) {
        AqlQueueRecord &queue = aql_queues_[queue_index];
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
      for (AqlQueueRecord &queue : aql_queues_)
        if (!queue.publication_retry_pending)
          fetch_from_queue(queue, now);
    }
    process_refetched_entries = false;

    for (AqlQueueRecord &queue : aql_queues_) {
      if (queue.faulted || queue.debug_suspended || queue.runtime_suspended ||
          queue.publication_retry_pending)
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
    const simdojo::Tick next = next_pooled_due_tick(now);
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

  const bool all_done = completion_ && completion_->all_complete(aql_queues_);
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
