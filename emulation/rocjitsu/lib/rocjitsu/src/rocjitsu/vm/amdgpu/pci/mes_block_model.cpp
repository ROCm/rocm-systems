// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pci/mes_block_model.h"

#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/amdgpu/command_processor_queue_backend.h"
#include "rocjitsu/vm/amdgpu/pci/physical_memory_access.h"
#include "rocjitsu/vm/amdgpu/pci/sdma_block_model.h"
#include "rocjitsu/vm/amdgpu/pm4_bootstrap_executor.h"
#include "rocjitsu/vm/soc.h"
#include "simdojo/components/pci_device.h"
#include "util/log.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <exception>
#include <format>
#include <optional>
#include <utility>

namespace rocjitsu {
namespace {

constexpr uint32_t kGcSegment = 0;
constexpr uint32_t kGcControlSegment = 1;

// The bank selected by GRBM_GFX_INDEX while the driver initializes a MES HQD.
constexpr uint32_t kQueueActive = 0x1fab;
constexpr uint32_t kMqdBaseLow = 0x1fa9;
constexpr uint32_t kMqdBaseHigh = 0x1faa;
constexpr uint32_t kQueueVmid = 0x1fac;
constexpr uint32_t kQueuePersistentState = 0x1fad;
constexpr uint32_t kQueueBaseLow = 0x1fb1;
constexpr uint32_t kQueueBaseHigh = 0x1fb2;
constexpr uint32_t kQueueReadPointer = 0x1fb3;
constexpr uint32_t kQueueReadPointerAddressLow = 0x1fb4;
constexpr uint32_t kQueueReadPointerAddressHigh = 0x1fb5;
constexpr uint32_t kQueueWritePointerAddressLow = 0x1fb6;
constexpr uint32_t kQueueWritePointerAddressHigh = 0x1fb7;
constexpr uint32_t kQueueDoorbellControl = 0x1fb8;
constexpr uint32_t kQueueControl = 0x1fba;
constexpr uint32_t kMqdControl = 0x1fcb;
constexpr uint32_t kQueueWritePointerLow = 0x1fdf;
constexpr uint32_t kQueueWritePointerHigh = 0x1fe0;
constexpr uint32_t kMesControl = 0x2807;
constexpr uint32_t kGrbmGfxIndex = 0x2200;
constexpr uint32_t kRlcCpSchedulers = 0x098a;

constexpr uint32_t kDoorbellEnable = 0x40000000;
constexpr uint32_t kDoorbellOffsetMask = 0x0ffffffc;
constexpr uint32_t kQueueSizeMask = 0x3f;
constexpr uint32_t kQueueActiveBit = 0x1;

constexpr uint32_t kMesApiTypeScheduler = 1;
constexpr uint32_t kMesApiSetHwResources = 0;
constexpr uint32_t kMesApiSetSchedulingConfig = 1;
constexpr uint32_t kMesApiAddQueue = 2;
constexpr uint32_t kMesApiRemoveQueue = 3;
constexpr uint32_t kMesApiPerformYield = 4;
constexpr uint32_t kMesApiChangeGangPriority = 5;
constexpr uint32_t kMesApiSuspend = 6;
constexpr uint32_t kMesApiResume = 7;
constexpr uint32_t kMesApiReset = 8;
constexpr uint32_t kMesApiSetLogBuffer = 9;
constexpr uint32_t kMesApiQuerySchedulerStatus = 11;
constexpr uint32_t kMesApiSetDebugVmid = 13;
constexpr uint32_t kMesApiMisc = 14;
constexpr uint32_t kMesApiUpdateRootPageTable = 15;
constexpr uint32_t kMesApiAmdLog = 16;
constexpr uint32_t kMesApiSetSeMode = 17;
constexpr uint32_t kMesApiSetGangSubmit = 18;
constexpr uint32_t kMesApiSetHwResources1 = 19;
constexpr uint32_t kMesApiInvalidateTlbs = 20;
constexpr uint32_t kMesBackendQueueIdBase = uint32_t{1} << 31;
constexpr uint32_t kMesFrameDwords = 64;
constexpr uint32_t kMesFrameBytes = kMesFrameDwords * sizeof(uint32_t);
constexpr uint32_t kAddQueueMqdAddressDword = 20;
constexpr uint32_t kAddQueueDoorbellDword = 18;
constexpr uint32_t kAddQueueWritePointerAddressDword = 22;
constexpr uint32_t kAddQueueTypeDword = 28;
constexpr uint32_t kAddQueueSizeDword = 30;
constexpr uint32_t kAddQueueProcessContextAddressDword = 10;
constexpr uint32_t kRemoveQueueDoorbellDword = 1;
constexpr uint32_t kInvalidateTlbsSelectorDword = 6;
constexpr uint32_t kUpdateRootPageTableBaseDword = 2;
constexpr uint32_t kUpdateRootProcessContextAddressDword = 4;
constexpr uint32_t kMesQueueTypeGfx = 0;
constexpr uint32_t kMesQueueTypeCompute = 1;
constexpr uint32_t kMesQueueTypeSdma = 2;
constexpr uint32_t kMesQueueTypeScheduler = 3;

constexpr uint32_t kScratchRegister0 = 0x2040;

constexpr uint32_t kMqdFirstDword = 130;
constexpr uint32_t kMqdVmidDword = 131;
constexpr uint32_t kMqdQueueBaseLowDword = 136;
constexpr uint32_t kMqdQueueBaseHighDword = 137;
constexpr uint32_t kMqdReadPointerAddressLowDword = 139;
constexpr uint32_t kMqdReadPointerAddressHighDword = 140;
constexpr uint32_t kMqdWritePointerAddressLowDword = 141;
constexpr uint32_t kMqdWritePointerAddressHighDword = 142;
constexpr uint32_t kMqdDoorbellControlDword = 143;
constexpr uint32_t kMqdQueueControlDword = 145;
constexpr uint32_t kMqdAqlControlDword = 181;
constexpr uint32_t kMqdNoUpdateReadPointer = 0x08000000;
constexpr uint32_t kMqdLastDword = kMqdAqlControlDword;

constexpr uint32_t kSdmaMqdQueueControlDword = 0;
constexpr uint32_t kSdmaMqdQueueBaseLowDword = 1;
constexpr uint32_t kSdmaMqdQueueBaseHighDword = 2;
constexpr uint32_t kSdmaMqdReadPointerLowDword = 3;
constexpr uint32_t kSdmaMqdReadPointerHighDword = 4;
constexpr uint32_t kSdmaMqdReadPointerAddressLowDword = 7;
constexpr uint32_t kSdmaMqdReadPointerAddressHighDword = 8;
constexpr uint32_t kSdmaMqdDoorbellDword = 17;
constexpr uint32_t kSdmaMqdWritePointerAddressLowDword = 24;
constexpr uint32_t kSdmaMqdWritePointerAddressHighDword = 25;
constexpr uint32_t kSdmaMqdEngineIdDword = 126;
constexpr uint32_t kSdmaMqdQueueIdDword = 127;
constexpr uint32_t kSdmaQueueSizeMask = 0x0000003e;
constexpr uint32_t kSdmaQueueSizeShift = 1;

constexpr std::array<uint32_t, 15> kWritableRegisters = {
    kQueueActive,
    kMqdBaseLow,
    kMqdBaseHigh,
    kQueueVmid,
    kQueuePersistentState,
    kQueueBaseLow,
    kQueueBaseHigh,
    kQueueReadPointer,
    kQueueReadPointerAddressLow,
    kQueueReadPointerAddressHigh,
    kQueueWritePointerAddressLow,
    kQueueWritePointerAddressHigh,
    kQueueDoorbellControl,
    kQueueControl,
    kMqdControl,
};

uint64_t join(uint32_t low, uint32_t high) {
  return static_cast<uint64_t>(low) | (static_cast<uint64_t>(high) << 32);
}

uint32_t dword(const std::array<std::byte, kMesFrameBytes> &frame, uint32_t index) {
  uint32_t value = 0;
  std::memcpy(&value, frame.data() + index * sizeof(value), sizeof(value));
  return value;
}

uint64_t qword(const std::array<std::byte, kMesFrameBytes> &frame, uint32_t index) {
  uint64_t value = 0;
  std::memcpy(&value, frame.data() + index * sizeof(uint32_t), sizeof(value));
  return value;
}

uint32_t dword(std::span<const std::byte> bytes, uint32_t index) {
  uint32_t value = 0;
  std::memcpy(&value, bytes.data() + index * sizeof(value), sizeof(value));
  return value;
}

uint64_t qword(std::span<const std::byte> bytes, uint32_t index) {
  uint64_t value = 0;
  std::memcpy(&value, bytes.data() + index * sizeof(uint32_t), sizeof(value));
  return value;
}

std::optional<uint32_t> status_dword(uint32_t opcode) {
  switch (opcode) {
  case kMesApiSetHwResources:
    return 50;
  case kMesApiSetSchedulingConfig:
    return 34;
  case kMesApiAddQueue:
    return 38;
  case kMesApiRemoveQueue:
  case kMesApiSetLogBuffer:
  case kMesApiUpdateRootPageTable:
  case kMesApiAmdLog:
  case kMesApiSetSeMode:
    return 6;
  case kMesApiPerformYield:
  case kMesApiQuerySchedulerStatus:
  case kMesApiSetDebugVmid:
  case kMesApiMisc:
  case kMesApiSetGangSubmit:
  case kMesApiSetHwResources1:
  case kMesApiInvalidateTlbs:
    return 2;
  case kMesApiChangeGangPriority:
  case kMesApiSuspend:
    return 8;
  case kMesApiResume:
    return 4;
  case kMesApiReset:
    return 28;
  default:
    return std::nullopt;
  }
}

} // namespace

MesBlockModel::MesBlockModel(IpRegisterWindow registers)
    : IpBlockModel("the MES kernel queue", std::move(registers)) {}

std::unique_ptr<MesBlockModel> MesBlockModel::create(const IpBlock &block,
                                                     IpRegisterWindow registers) {
  if (block.hardware_id == IpHardwareId::Gc && block.major == 12 && block.minor == 1 &&
      block.revision == 0) {
    return std::unique_ptr<MesBlockModel>(new MesBlockModel(std::move(registers)));
  }
  util::Logger::warn(std::format(
      "{}: no MES kernel-queue layout is known for GC {}.{}.{}, so MES startup cannot run",
      registers.owner(), block.major, block.minor, block.revision));
  return nullptr;
}

std::vector<RegisterClaim> MesBlockModel::claims() const {
  std::vector<RegisterClaim> claimed;
  claimed.reserve(kWritableRegisters.size() + 5);
  for (const uint32_t reg : kWritableRegisters) {
    claimed.push_back({.segment_index = kGcSegment, .first_dword = reg, .count = 1});
  }
  claimed.push_back(
      {.segment_index = kGcSegment, .first_dword = kQueueWritePointerLow, .count = 2});
  claimed.push_back({.segment_index = kGcControlSegment, .first_dword = kMesControl, .count = 1});
  claimed.push_back({.segment_index = kGcControlSegment, .first_dword = kGrbmGfxIndex, .count = 1});
  claimed.push_back(
      {.segment_index = kGcControlSegment, .first_dword = kRlcCpSchedulers, .count = 1});
  return claimed;
}

bool MesBlockModel::reset() {
  if (!teardown_queues())
    return false;
  next_queue_ordinal_ = 0;
  bool defined = true;
  for (const uint32_t reg : kWritableRegisters) {
    defined = registers_.define(kGcSegment, reg, 0) && defined;
  }
  defined = registers_.define(kGcSegment, kQueueWritePointerLow, 0) && defined;
  defined = registers_.define(kGcSegment, kQueueWritePointerHigh, 0) && defined;
  defined = registers_.define(kGcControlSegment, kMesControl, 0) && defined;
  defined = registers_.define(kGcControlSegment, kGrbmGfxIndex, 0) && defined;
  defined = registers_.define(kGcControlSegment, kRlcCpSchedulers, 0) && defined;
  return defined;
}

bool MesBlockModel::teardown_queues() {
  // Teardown is a transaction boundary: once it starts destroying queue state,
  // no pre-reset semantic may later publish success through a retained snapshot,
  // even if destruction makes partial progress and the caller reports failure.
  pending_mes_frames_.clear();
  pending_compute_runs_.clear();

  bool complete = true;
  for (auto queue = mapped_queues_.begin(); queue != mapped_queues_.end();) {
    bool released = true;
    if (queue->queue_handle) {
      released = soc_ != nullptr && soc_->queue_service().destroy(queue->queue_handle);
    } else if (queue->kind == QueueKind::Sdma) {
      released = sdma_ != nullptr && sdma_->unmap_user_queue(queue->doorbell_offset);
    }
    if (!released) {
      complete = false;
      ++queue;
      continue;
    }
    queue = mapped_queues_.erase(queue);
  }

  if (soc_ != nullptr) {
    for (auto address_space = address_spaces_.begin(); address_space != address_spaces_.end();) {
      const uint32_t process_id = address_space->first;
      const bool still_used =
          std::ranges::any_of(mapped_queues_, [process_id](const Queue &candidate) {
            return candidate.process_id == process_id && candidate.address_space;
          });
      if (still_used || !soc_->gpu_vm().unregister_address_space(address_space->second.handle)) {
        complete = false;
        ++address_space;
        continue;
      }
      address_space = address_spaces_.erase(address_space);
    }
  } else if (!address_spaces_.empty()) {
    complete = false;
  }
  return complete;
}

MesBlockModel::Queue MesBlockModel::direct_queue() const {
  const uint32_t queue_size = registers_.read(kGcSegment, kQueueControl) & kQueueSizeMask;
  return {
      .ring_base = join(registers_.read(kGcSegment, kQueueBaseLow),
                        registers_.read(kGcSegment, kQueueBaseHigh))
                   << 8,
      .read_pointer_address = join(registers_.read(kGcSegment, kQueueReadPointerAddressLow),
                                   registers_.read(kGcSegment, kQueueReadPointerAddressHigh)),
      .write_pointer_address = join(registers_.read(kGcSegment, kQueueWritePointerAddressLow),
                                    registers_.read(kGcSegment, kQueueWritePointerAddressHigh)),
      .ring_dwords = queue_size < 30 ? uint64_t{1} << (queue_size + 1) : 0,
      .doorbell_offset = registers_.read(kGcSegment, kQueueDoorbellControl) & kDoorbellOffsetMask,
      .active = (registers_.read(kGcSegment, kQueueDoorbellControl) & kDoorbellEnable) != 0 &&
                (registers_.read(kGcSegment, kQueueActive) & kQueueActiveBit) != 0,
      .aql = false,
      .address_space = {},
      .queue_handle = {},
      .kind = QueueKind::Mes,
  };
}

MesBlockModel::QueueLookup
MesBlockModel::queue_from_mqd(uint64_t mqd_address, QueueKind kind,
                              const amdgpu::GpuVmAccess &gart_access) const {
  if (kind == QueueKind::Sdma) {
    constexpr std::size_t kSdmaMqdBytes = (kSdmaMqdQueueIdDword + 1) * sizeof(uint32_t);
    std::array<std::byte, kSdmaMqdBytes> mqd{};
    if (mqd_address == 0)
      return {.outcome = amdgpu::VmAccessOutcome::Malformed, .queue = std::nullopt};
    const amdgpu::VmAccessOutcome outcome = gart_access.read(mqd_address, mqd);
    if (outcome != amdgpu::VmAccessOutcome::Complete)
      return {.outcome = outcome, .queue = std::nullopt};
    const uint32_t queue_size =
        (dword(mqd, kSdmaMqdQueueControlDword) & kSdmaQueueSizeMask) >> kSdmaQueueSizeShift;
    if (queue_size >= 30)
      return {.outcome = amdgpu::VmAccessOutcome::Malformed, .queue = std::nullopt};
    return {
        .outcome = amdgpu::VmAccessOutcome::Complete,
        .queue = Queue{
            .ring_base =
                join(dword(mqd, kSdmaMqdQueueBaseLowDword), dword(mqd, kSdmaMqdQueueBaseHighDword))
                << 8,
            .read_pointer_address = join(dword(mqd, kSdmaMqdReadPointerAddressLowDword),
                                         dword(mqd, kSdmaMqdReadPointerAddressHighDword)),
            .write_pointer_address = join(dword(mqd, kSdmaMqdWritePointerAddressLowDword),
                                          dword(mqd, kSdmaMqdWritePointerAddressHighDword)),
            .initial_read_pointer = join(dword(mqd, kSdmaMqdReadPointerLowDword),
                                         dword(mqd, kSdmaMqdReadPointerHighDword)),
            .ring_dwords = uint64_t{1} << queue_size,
            .doorbell_offset = dword(mqd, kSdmaMqdDoorbellDword) & kDoorbellOffsetMask,
            .queue_id = dword(mqd, kSdmaMqdQueueIdDword),
            .engine_id = dword(mqd, kSdmaMqdEngineIdDword),
            .active = true,
            .address_space = {},
            .queue_handle = {},
            .kind = kind,
        }};
  }

  constexpr std::size_t kMqdBytes = (kMqdLastDword - kMqdFirstDword + 1) * sizeof(uint32_t);
  std::array<std::byte, kMqdBytes> mqd{};
  if (mqd_address == 0)
    return {.outcome = amdgpu::VmAccessOutcome::Malformed, .queue = std::nullopt};
  const amdgpu::VmAccessOutcome outcome =
      gart_access.read(mqd_address + kMqdFirstDword * sizeof(uint32_t), mqd);
  if (outcome != amdgpu::VmAccessOutcome::Complete)
    return {.outcome = outcome, .queue = std::nullopt};
  const auto field = [&mqd](uint32_t absolute_dword) {
    return dword(mqd, absolute_dword - kMqdFirstDword);
  };
  const uint32_t queue_size = field(kMqdQueueControlDword) & kQueueSizeMask;
  if (queue_size >= 30)
    return {.outcome = amdgpu::VmAccessOutcome::Malformed, .queue = std::nullopt};
  return {.outcome = amdgpu::VmAccessOutcome::Complete,
          .queue = Queue{
              .ring_base = join(field(kMqdQueueBaseLowDword), field(kMqdQueueBaseHighDword)) << 8,
              .read_pointer_address = join(field(kMqdReadPointerAddressLowDword),
                                           field(kMqdReadPointerAddressHighDword)),
              .write_pointer_address = join(field(kMqdWritePointerAddressLowDword),
                                            field(kMqdWritePointerAddressHighDword)),
              .ring_dwords = uint64_t{1} << (queue_size + 1),
              .doorbell_offset = field(kMqdDoorbellControlDword) & kDoorbellOffsetMask,
              .process_id = field(kMqdVmidDword),
              .queue_id = static_cast<uint32_t>(
                  (field(kMqdDoorbellControlDword) & kDoorbellOffsetMask) / sizeof(uint32_t)),
              // ADD_QUEUE is the operation that activates the hardware queue. The
              // compute self-test deliberately supplies an MQD with HQD_ACTIVE clear
              // and no legacy DOORBELL_EN bit, and expects MES to make it runnable
              // while mapping it.
              .active = true,
              .aql = (field(kMqdQueueControlDword) & kMqdNoUpdateReadPointer) != 0 ||
                     field(kMqdAqlControlDword) != 0,
              .address_space = {},
              .queue_handle = {},
              .kind = kind,
          }};
}

bool MesBlockModel::bind_process_address_space(Queue &queue, PciMemoryAccess &memory,
                                               bool &created_address_space) {
  created_address_space = false;
  if (soc_ == nullptr || queue.page_table_base == 0) {
    return false;
  }

  auto physical_memory = std::make_shared<PciPhysicalMemoryAccess>(memory);
  auto existing_address_space = address_spaces_.find(queue.process_id);
  if (existing_address_space == address_spaces_.end()) {
    const amdgpu::AddressSpaceHandle handle = soc_->gpu_vm().register_gfx12_address_space(
        queue.process_id, queue.page_table_base, physical_memory);
    if (!handle)
      return false;
    existing_address_space =
        address_spaces_
            .emplace(queue.process_id,
                     AddressSpace{.handle = handle,
                                  .page_table_base = queue.page_table_base,
                                  .process_context_address = queue.process_context_address})
            .first;
    created_address_space = true;
  } else {
    AddressSpace &address_space = existing_address_space->second;
    if (address_space.process_context_address != 0 && queue.process_context_address != 0 &&
        address_space.process_context_address != queue.process_context_address) {
      return false;
    }
    // ADD_QUEUE consumes the process binding; it does not own an already-live
    // PASID's root transition. UPDATE_ROOT_PAGE_TABLE performs that operation
    // explicitly. Rejecting a mismatched root here keeps every existing queue
    // on its committed translation until that separate transaction succeeds.
    if (address_space.page_table_base != queue.page_table_base)
      return false;
  }

  queue.address_space = existing_address_space->second.handle;
  return true;
}

bool MesBlockModel::update_process_address_space(uint64_t process_context_address,
                                                 uint64_t page_table_base,
                                                 PciMemoryAccess &memory) {
  if (soc_ == nullptr || process_context_address == 0 || page_table_base == 0)
    return false;

  auto matching = address_spaces_.end();
  for (auto candidate = address_spaces_.begin(); candidate != address_spaces_.end(); ++candidate) {
    if (candidate->second.process_context_address != process_context_address)
      continue;
    if (matching != address_spaces_.end())
      return false;
    matching = candidate;
  }
  if (matching == address_spaces_.end())
    return false;

  auto physical_memory = std::make_shared<PciPhysicalMemoryAccess>(memory);
  if (!soc_->gpu_vm().replace_gfx12_address_space_root(matching->second.handle, page_table_base,
                                                       physical_memory))
    return false;

  matching->second.page_table_base = page_table_base;
  for (Queue &queue : mapped_queues_) {
    if (queue.address_space == matching->second.handle)
      queue.page_table_base = page_table_base;
  }
  return true;
}

bool MesBlockModel::register_aql_queue(Queue &queue) {
  if (queue.write_pointer_address == 0 || queue.ring_dwords > UINT32_MAX / sizeof(uint32_t)) {
    return false;
  }
  if (soc_ == nullptr || next_queue_ordinal_ >= kMesBackendQueueIdBase)
    return false;
  amdgpu::CommandProcessor *owner = soc_->assign_queue_owner_cp(next_queue_ordinal_);
  if (owner == nullptr)
    return false;
  queue.queue_id = kMesBackendQueueIdBase | next_queue_ordinal_;
  try {
    queue.queue_handle = soc_->queue_service().create({
        .address_space = queue.address_space,
        .interrupt_sink = interrupt_sink_,
        .backend = amdgpu::make_command_processor_queue_backend(*owner),
        .process_id = queue.process_id,
        .queue_id = queue.queue_id,
        .ring_base_va = queue.ring_base,
        .ring_size = static_cast<uint32_t>(queue.ring_dwords * sizeof(uint32_t)),
        .read_ptr_va = queue.read_pointer_address,
        .write_ptr_va = queue.write_pointer_address,
        .initial_read_pointer = std::nullopt,
        .doorbell_offset = static_cast<uint32_t>(queue.doorbell_offset),
        // The vfio-user front end traps the MMIO doorbell and wakes this queue
        // explicitly. A nonzero value keeps the internal-queue fetch path enabled.
        .doorbell_va = queue.write_pointer_address,
        .host_accessible = false,
        .xcd_fanout = true,
    });
  } catch (const std::exception &error) {
    util::Logger::warn(
        std::format("{}: cannot create AQL queue backend: {}", registers_.owner(), error.what()));
    return false;
  }
  return static_cast<bool>(queue.queue_handle);
}

bool MesBlockModel::register_sdma_queue(Queue &queue, PciMemoryAccess &memory) {
  if (sdma_ == nullptr || soc_ == nullptr || queue.ring_dwords == 0 ||
      queue.ring_dwords > UINT32_MAX / sizeof(uint32_t) ||
      next_queue_ordinal_ >= kMesBackendQueueIdBase) {
    return false;
  }
  amdgpu::CommandProcessor *owner = soc_->assign_queue_owner_cp(next_queue_ordinal_);
  queue.queue_id = kMesBackendQueueIdBase | next_queue_ordinal_;

  // A configured SoC executes MES-created SDMA queues through the same core
  // command-processor backend as legacy KFD. Device-only register tests have no
  // XCD and retain the synchronous block-model fallback.
  std::shared_ptr<amdgpu::QueueBackend> backend =
      owner != nullptr ? amdgpu::make_command_processor_queue_backend(*owner)
                       : sdma_->make_user_queue_backend(memory, queue.engine_id);
  try {
    queue.queue_handle = soc_->queue_service().create({
        .address_space = queue.address_space,
        .interrupt_sink = interrupt_sink_,
        .backend = std::move(backend),
        .process_id = queue.process_id,
        .queue_id = queue.queue_id,
        .ring_base_va = queue.ring_base,
        .ring_size = static_cast<uint32_t>(queue.ring_dwords * sizeof(uint32_t)),
        .read_ptr_va = queue.read_pointer_address,
        .write_ptr_va = queue.write_pointer_address,
        .initial_read_pointer = queue.initial_read_pointer,
        .doorbell_offset = static_cast<uint32_t>(queue.doorbell_offset),
        .doorbell_va = owner != nullptr ? queue.write_pointer_address : 0,
        .kind = amdgpu::QueueKind::Sdma,
    });
  } catch (const std::exception &error) {
    util::Logger::warn(
        std::format("{}: cannot create SDMA queue backend: {}", registers_.owner(), error.what()));
    return false;
  }
  return static_cast<bool>(queue.queue_handle);
}

void MesBlockModel::rollback_created_address_space(const Queue &queue, bool created_address_space) {
  if (!created_address_space || soc_ == nullptr || !queue.address_space)
    return;
  (void)soc_->gpu_vm().unregister_address_space(queue.address_space);
  address_spaces_.erase(queue.process_id);
}

void MesBlockModel::release_unused_address_space(uint32_t process_id) {
  if (soc_ == nullptr)
    return;
  const bool process_still_used =
      std::ranges::any_of(mapped_queues_, [process_id](const Queue &candidate) {
        return candidate.process_id == process_id && candidate.address_space;
      });
  if (process_still_used)
    return;
  const auto address_space = address_spaces_.find(process_id);
  if (address_space != address_spaces_.end() &&
      soc_->gpu_vm().unregister_address_space(address_space->second.handle)) {
    address_spaces_.erase(address_space);
  }
}

bool MesBlockModel::commit_queue(Queue queue, bool created_address_space) {
  util::Logger::cp([&](auto &os) {
    os << std::format("{}: map queue kind={} ring={:#x} rptr={:#x} wptr={:#x} pt={:#x} "
                      "dwords={} doorbell={:#x} active={} aql={}",
                      registers_.owner(), static_cast<unsigned>(queue.kind), queue.ring_base,
                      queue.read_pointer_address, queue.write_pointer_address,
                      queue.page_table_base, queue.ring_dwords, queue.doorbell_offset, queue.active,
                      queue.aql);
  });
  // A retained publication is identified by this doorbell and its queue
  // configuration. Replacing that queue would either strand the journal or let
  // it acknowledge work against a different queue generation.
  if (pending_mes_frames_.contains(queue.doorbell_offset) ||
      pending_compute_runs_.contains(queue.doorbell_offset)) {
    if (queue.queue_handle && soc_ != nullptr)
      (void)soc_->queue_service().destroy(queue.queue_handle);
    rollback_created_address_space(queue, created_address_space);
    return false;
  }
  const auto existing =
      std::ranges::find(mapped_queues_, queue.doorbell_offset, &Queue::doorbell_offset);
  if (existing == mapped_queues_.end()) {
    try {
      mapped_queues_.push_back(queue);
    } catch (...) {
      if (queue.queue_handle && soc_ != nullptr)
        (void)soc_->queue_service().destroy(queue.queue_handle);
      rollback_created_address_space(queue, created_address_space);
      return false;
    }
    if (queue.queue_handle)
      ++next_queue_ordinal_;
    if (queue.address_space && queue.process_context_address != 0) {
      AddressSpace &address_space = address_spaces_.at(queue.process_id);
      if (address_space.process_context_address == 0)
        address_space.process_context_address = queue.process_context_address;
    }
    return true;
  }

  const Queue replaced = *existing;
  bool detached = true;
  if (replaced.queue_handle) {
    detached = soc_ != nullptr && soc_->queue_service().destroy(replaced.queue_handle);
  } else if (replaced.kind == QueueKind::Sdma) {
    detached = sdma_ != nullptr && sdma_->unmap_user_queue(replaced.doorbell_offset);
  }
  if (!detached) {
    if (queue.queue_handle && soc_ != nullptr)
      (void)soc_->queue_service().destroy(queue.queue_handle);
    rollback_created_address_space(queue, created_address_space);
    return false;
  }

  *existing = queue;
  if (queue.queue_handle)
    ++next_queue_ordinal_;
  if (queue.address_space && queue.process_context_address != 0) {
    AddressSpace &address_space = address_spaces_.at(queue.process_id);
    if (address_space.process_context_address == 0)
      address_space.process_context_address = queue.process_context_address;
  }
  if (replaced.process_id != queue.process_id)
    release_unused_address_space(replaced.process_id);
  return true;
}

bool MesBlockModel::remove_queue(uint64_t doorbell_offset) {
  // Destruction of the queue that owns a pending frame would remove the only
  // route by which its completion/read-pointer publication can resume.
  if (pending_mes_frames_.contains(doorbell_offset) ||
      pending_compute_runs_.contains(doorbell_offset))
    return false;

  const auto queue = std::ranges::find(mapped_queues_, doorbell_offset, &Queue::doorbell_offset);
  if (queue == mapped_queues_.end())
    return false;

  const uint32_t process_id = queue->process_id;
  if (queue->queue_handle) {
    if (soc_ == nullptr || !soc_->queue_service().destroy(queue->queue_handle))
      return false;
  } else if (queue->kind == QueueKind::Sdma &&
             (sdma_ == nullptr || !sdma_->unmap_user_queue(doorbell_offset))) {
    return false;
  }
  mapped_queues_.erase(queue);

  release_unused_address_space(process_id);
  return true;
}

bool MesBlockModel::read_gpu(PciMemoryAccess &memory, uint64_t address,
                             std::span<std::byte> bytes) const {
  (void)memory;
  if (soc_ == nullptr)
    return false;
  const amdgpu::AddressSpaceHandle gart = soc_->gpu_vm().gart_address_space();
  return gart && soc_->gpu_vm().read(gart, address, bytes) == amdgpu::VmAccessOutcome::Complete;
}

bool MesBlockModel::write_gpu(PciMemoryAccess &memory, uint64_t address,
                              std::span<const std::byte> bytes) const {
  (void)memory;
  if (soc_ == nullptr)
    return false;
  const amdgpu::AddressSpaceHandle gart = soc_->gpu_vm().gart_address_space();
  return gart && soc_->gpu_vm().write(gart, address, bytes) == amdgpu::VmAccessOutcome::Complete;
}

bool MesBlockModel::read_queue_gpu(PciMemoryAccess &memory, const Queue &queue, uint64_t address,
                                   std::span<std::byte> bytes) const {
  if (!queue.address_space) {
    return queue.page_table_base == 0 && read_gpu(memory, address, bytes);
  }
  return soc_ != nullptr && soc_->gpu_vm().read(queue.address_space, address, bytes) ==
                                amdgpu::VmAccessOutcome::Complete;
}

bool MesBlockModel::write_queue_gpu(PciMemoryAccess &memory, const Queue &queue, uint64_t address,
                                    std::span<const std::byte> bytes) const {
  if (!queue.address_space) {
    return queue.page_table_base == 0 && write_gpu(memory, address, bytes);
  }
  return soc_ != nullptr && soc_->gpu_vm().write(queue.address_space, address, bytes) ==
                                amdgpu::VmAccessOutcome::Complete;
}

DoorbellDisposition MesBlockModel::process_queue(Queue queue, uint64_t write_pointer,
                                                 PciMemoryAccess &memory) {
  switch (queue.kind) {
  case QueueKind::Mes:
    return process_mes_queue(queue, write_pointer, memory);
  case QueueKind::Compute:
    return process_compute_queue(queue, write_pointer, memory);
  case QueueKind::Sdma:
    return DoorbellDisposition::Faulted;
  }
  return DoorbellDisposition::Faulted;
}

amdgpu::VmAccessOutcome
MesBlockModel::execute_mes_semantic(std::span<const std::byte> frame, uint32_t opcode,
                                    PciMemoryAccess &memory,
                                    const amdgpu::GpuVmAccess &gart_access) {
  if (opcode == kMesApiAddQueue) {
    const uint64_t mqd_address = qword(frame, kAddQueueMqdAddressDword);
    const uint32_t queue_type = dword(frame, kAddQueueTypeDword);
    QueueKind kind = QueueKind::Mes;
    if (queue_type == kMesQueueTypeGfx || queue_type == kMesQueueTypeCompute) {
      kind = QueueKind::Compute;
    } else if (queue_type == kMesQueueTypeSdma) {
      kind = QueueKind::Sdma;
    } else if (queue_type != kMesQueueTypeScheduler) {
      util::Logger::warn(
          std::format("{}: unsupported MES queue type {}", registers_.owner(), queue_type));
      return amdgpu::VmAccessOutcome::Malformed;
    }
    QueueLookup lookup = queue_from_mqd(mqd_address, kind, gart_access);
    if (lookup.outcome != amdgpu::VmAccessOutcome::Complete || !lookup.queue) {
      util::Logger::warn(std::format("{}: cannot read the MES ADD_QUEUE MQD at {:#x}",
                                     registers_.owner(), mqd_address));
      return lookup.outcome;
    }
    Queue &mapped = *lookup.queue;
    mapped.page_table_base = qword(frame, 2);
    mapped.process_context_address = qword(frame, kAddQueueProcessContextAddressDword);
    // MES's process_id is the PASID. CP_HQD_VMID is a hardware VMID and is
    // still zero in the MQD when the driver hands the queue to MES, before
    // firmware would assign one.
    mapped.process_id = dword(frame, 1);
    bool created_address_space = false;
    if (mapped.kind == QueueKind::Compute &&
        !bind_process_address_space(mapped, memory, created_address_space)) {
      util::Logger::warn(std::format("{}: cannot attach compute queue at doorbell {:#x}",
                                     registers_.owner(), mapped.doorbell_offset));
      return amdgpu::VmAccessOutcome::Faulted;
    }
    if (mapped.kind == QueueKind::Sdma) {
      mapped.doorbell_offset =
          static_cast<uint64_t>(dword(frame, kAddQueueDoorbellDword)) * sizeof(uint32_t);
      mapped.write_pointer_address = qword(frame, kAddQueueWritePointerAddressDword);
      if (const uint32_t ring_dwords = dword(frame, kAddQueueSizeDword); ring_dwords != 0)
        mapped.ring_dwords = ring_dwords;
      if (!bind_process_address_space(mapped, memory, created_address_space) ||
          !register_sdma_queue(mapped, memory)) {
        rollback_created_address_space(mapped, created_address_space);
        util::Logger::warn(std::format("{}: cannot attach SDMA queue at doorbell {:#x}",
                                       registers_.owner(), mapped.doorbell_offset));
        return amdgpu::VmAccessOutcome::Faulted;
      }
    }
    if (mapped.kind == QueueKind::Compute && mapped.aql && !register_aql_queue(mapped)) {
      rollback_created_address_space(mapped, created_address_space);
      util::Logger::warn(std::format("{}: cannot attach AQL queue at doorbell {:#x}",
                                     registers_.owner(), mapped.doorbell_offset));
      return amdgpu::VmAccessOutcome::Faulted;
    }
    if (!commit_queue(mapped, created_address_space)) {
      util::Logger::warn(std::format("{}: cannot commit queue at doorbell {:#x}",
                                     registers_.owner(), mapped.doorbell_offset));
      return amdgpu::VmAccessOutcome::Faulted;
    }
  } else if (opcode == kMesApiRemoveQueue) {
    const uint64_t doorbell_offset =
        static_cast<uint64_t>(dword(frame, kRemoveQueueDoorbellDword)) * sizeof(uint32_t);
    if (!remove_queue(doorbell_offset)) {
      util::Logger::warn(std::format("{}: cannot remove queue at doorbell {:#x}",
                                     registers_.owner(), doorbell_offset));
      return amdgpu::VmAccessOutcome::Faulted;
    }
  } else if (opcode == kMesApiUpdateRootPageTable) {
    const uint64_t page_table_base = qword(frame, kUpdateRootPageTableBaseDword);
    const uint64_t process_context_address = qword(frame, kUpdateRootProcessContextAddressDword);
    if (!update_process_address_space(process_context_address, page_table_base, memory)) {
      util::Logger::warn(std::format("{}: cannot update root {:#x} for process context {:#x}",
                                     registers_.owner(), page_table_base, process_context_address));
      return amdgpu::VmAccessOutcome::Faulted;
    }
  } else if (opcode == kMesApiInvalidateTlbs) {
    const uint32_t selector = dword(frame, kInvalidateTlbsSelectorDword);
    const uint8_t selector_kind = static_cast<uint8_t>(selector);
    const uint16_t selector_id = static_cast<uint16_t>(selector >> 16);
    if (selector_kind != 0 || soc_ == nullptr) {
      util::Logger::warn(std::format("{}: unsupported MES TLB invalidation selector {} for id {}",
                                     registers_.owner(), selector_kind, selector_id));
      return amdgpu::VmAccessOutcome::Malformed;
    }
    const std::optional<amdgpu::AddressSpaceHandle> address_space =
        soc_->gpu_vm().find_vmid(selector_id);
    if (!address_space || !soc_->gpu_vm().invalidate(*address_space)) {
      util::Logger::warn(
          std::format("{}: cannot invalidate PASID {}", registers_.owner(), selector_id));
      return amdgpu::VmAccessOutcome::Faulted;
    }
  }
  return amdgpu::VmAccessOutcome::Complete;
}

amdgpu::VmAccessOutcome MesBlockModel::publish_mes_frame(PendingMesFrame &pending) {
  if (pending.phase == MesFramePhase::Semantic)
    return amdgpu::VmAccessOutcome::Malformed;
  if (pending.phase == MesFramePhase::Terminal)
    return pending.terminal_outcome;

  if (pending.phase == MesFramePhase::Completion) {
    const auto raw_completion =
        std::bit_cast<std::array<std::byte, sizeof(uint64_t)>>(pending.completion_value);
    const amdgpu::VmAccessOutcome outcome =
        pending.access.write(pending.completion_address, raw_completion);
    if (outcome == amdgpu::VmAccessOutcome::Unavailable)
      return outcome;
    if (outcome != amdgpu::VmAccessOutcome::Complete) {
      pending.phase = MesFramePhase::Terminal;
      pending.terminal_outcome = outcome;
      return outcome;
    }
    pending.phase = MesFramePhase::ReadPointer;
  }

  const auto raw_pointer =
      std::bit_cast<std::array<std::byte, sizeof(uint64_t)>>(pending.next_read_pointer);
  const amdgpu::VmAccessOutcome outcome =
      pending.access.write(pending.queue.read_pointer_address, raw_pointer);
  if (outcome != amdgpu::VmAccessOutcome::Complete &&
      outcome != amdgpu::VmAccessOutcome::Unavailable) {
    pending.phase = MesFramePhase::Terminal;
    pending.terminal_outcome = outcome;
  }
  return outcome;
}

DoorbellDisposition MesBlockModel::process_mes_queue(Queue queue, uint64_t write_pointer,
                                                     PciMemoryAccess &memory) {
  const uint64_t ring_dwords = queue.ring_dwords;
  const uint64_t ring_bytes = ring_dwords * sizeof(uint32_t);
  if (!queue.active || queue.ring_base == 0 || queue.read_pointer_address == 0 ||
      ring_dwords == 0 || write_pointer < kMesFrameDwords) {
    return DoorbellDisposition::Faulted;
  }

  uint64_t read_pointer = 0;
  std::optional<amdgpu::GpuVmAccess> access;
  const auto pending_frame = pending_mes_frames_.find(queue.doorbell_offset);
  if (pending_frame != pending_mes_frames_.end()) {
    PendingMesFrame &pending = pending_frame->second;
    if (pending.phase == MesFramePhase::Terminal)
      return DoorbellDisposition::Faulted;
    const std::shared_ptr<simdojo::PciTransportSession> retained_session =
        pending.transport_session.lock();
    const std::shared_ptr<simdojo::PciTransportSession> current_session =
        memory.capture_transport_session();
    if (retained_session == nullptr || current_session != retained_session ||
        retained_session->generation() != pending.transport_generation) {
      pending.phase = MesFramePhase::Terminal;
      pending.terminal_outcome = amdgpu::VmAccessOutcome::Malformed;
      return DoorbellDisposition::Faulted;
    }
    if (pending.queue.ring_base != queue.ring_base ||
        pending.queue.read_pointer_address != queue.read_pointer_address ||
        pending.queue.ring_dwords != queue.ring_dwords ||
        pending.next_read_pointer > write_pointer) {
      util::Logger::warn(
          std::format("{}: MES queue {:#x} changed while frame {} awaited publication",
                      registers_.owner(), queue.doorbell_offset, pending.frame_read_pointer));
      pending.phase = MesFramePhase::Terminal;
      pending.terminal_outcome = amdgpu::VmAccessOutcome::Malformed;
      return DoorbellDisposition::Faulted;
    }
    const amdgpu::VmAccessOutcome outcome = publish_mes_frame(pending);
    if (outcome == amdgpu::VmAccessOutcome::Unavailable)
      return DoorbellDisposition::Retry;
    if (outcome != amdgpu::VmAccessOutcome::Complete)
      return DoorbellDisposition::Faulted;
    read_pointer = pending.next_read_pointer;
    pending_mes_frames_.erase(pending_frame);
  } else {
    if (soc_ == nullptr)
      return DoorbellDisposition::Faulted;
    const amdgpu::AddressSpaceHandle gart = soc_->gpu_vm().gart_address_space();
    access = soc_->gpu_vm().snapshot(gart);
    if (!access)
      return DoorbellDisposition::Faulted;
    std::array<std::byte, sizeof(uint64_t)> raw_read_pointer{};
    const amdgpu::VmAccessOutcome outcome =
        access->read(queue.read_pointer_address, raw_read_pointer);
    if (outcome != amdgpu::VmAccessOutcome::Complete) {
      util::Logger::warn(std::format("{}: cannot read the MES queue read pointer at {:#x}",
                                     registers_.owner(), queue.read_pointer_address));
      return outcome == amdgpu::VmAccessOutcome::Unavailable ? DoorbellDisposition::Retry
                                                             : DoorbellDisposition::Faulted;
    }
    read_pointer = std::bit_cast<uint64_t>(raw_read_pointer);
  }

  const uint64_t pending = write_pointer - read_pointer;
  if (pending > ring_dwords || (pending % kMesFrameDwords) != 0) {
    util::Logger::warn(std::format("{}: MES doorbell moved from {} to {} dwords in a {}-dword ring",
                                   registers_.owner(), read_pointer, write_pointer, ring_dwords));
    return DoorbellDisposition::Faulted;
  }

  if (read_pointer != write_pointer && !access) {
    if (soc_ == nullptr)
      return DoorbellDisposition::Faulted;
    access = soc_->gpu_vm().snapshot(soc_->gpu_vm().gart_address_space());
    if (!access)
      return DoorbellDisposition::Faulted;
  }

  while (read_pointer != write_pointer) {
    std::array<std::byte, kMesFrameBytes> frame{};
    const uint64_t ring_offset = (read_pointer % ring_dwords) * sizeof(uint32_t);
    const std::size_t first =
        std::min<std::size_t>(frame.size(), static_cast<std::size_t>(ring_bytes - ring_offset));
    amdgpu::VmAccessOutcome outcome =
        access->read(queue.ring_base + ring_offset, std::span(frame).first(first));
    if (outcome == amdgpu::VmAccessOutcome::Complete && first != frame.size())
      outcome = access->read(queue.ring_base, std::span(frame).subspan(first));
    if (outcome != amdgpu::VmAccessOutcome::Complete) {
      util::Logger::warn(std::format("{}: cannot read a MES frame at {:#x}", registers_.owner(),
                                     queue.ring_base + ring_offset));
      return outcome == amdgpu::VmAccessOutcome::Unavailable ? DoorbellDisposition::Retry
                                                             : DoorbellDisposition::Faulted;
    }

    const uint32_t header = dword(frame, 0);
    const uint32_t type = header & 0xf;
    const uint32_t opcode = (header >> 4) & 0xff;
    const uint32_t frame_dwords = (header >> 12) & 0xff;
    const std::optional<uint32_t> status_at = status_dword(opcode);
    if (type != kMesApiTypeScheduler || frame_dwords != kMesFrameDwords || !status_at) {
      util::Logger::warn(std::format("{}: unsupported MES frame type {}, opcode {}, size {} dwords",
                                     registers_.owner(), type, opcode, frame_dwords));
      return DoorbellDisposition::Faulted;
    }

    const uint64_t completion_address = qword(frame, *status_at);
    const uint64_t completion_value = qword(frame, *status_at + 2);
    if (completion_address == 0) {
      util::Logger::warn(std::format("{}: cannot publish MES opcode {} completion at {:#x}",
                                     registers_.owner(), opcode, completion_address));
      return DoorbellDisposition::Faulted;
    }

    const std::shared_ptr<simdojo::PciTransportSession> transport_session =
        memory.capture_transport_session();

    PendingMesFrame pending_frame{
        .queue = queue,
        .access = *access,
        .frame_read_pointer = read_pointer,
        .next_read_pointer = read_pointer + kMesFrameDwords,
        .completion_address = completion_address,
        .completion_value = completion_value,
        .opcode = opcode,
        .phase = MesFramePhase::Semantic,
        .transport_session = transport_session,
        .transport_generation = transport_session != nullptr ? transport_session->generation() : 0,
    };
    auto [pending_it, inserted] =
        pending_mes_frames_.try_emplace(queue.doorbell_offset, std::move(pending_frame));
    if (!inserted)
      return DoorbellDisposition::Faulted;

    const amdgpu::VmAccessOutcome semantic = execute_mes_semantic(frame, opcode, memory, *access);
    if (semantic == amdgpu::VmAccessOutcome::Unavailable) {
      // The only retryable semantic dependency is the ADD_QUEUE MQD read,
      // which occurs before any queue, address-space, or TLB state is changed.
      pending_mes_frames_.erase(pending_it);
      return DoorbellDisposition::Retry;
    }
    if (semantic != amdgpu::VmAccessOutcome::Complete) {
      // Semantic helpers either commit completely or roll back before reporting
      // failure. Leave a rejected frame unretired so the guest may correct it;
      // only a failure after a successful semantic commit is terminalized.
      pending_mes_frames_.erase(pending_it);
      return DoorbellDisposition::Faulted;
    }
    pending_it->second.phase = MesFramePhase::Completion;
    const amdgpu::VmAccessOutcome publication = publish_mes_frame(pending_it->second);
    if (publication == amdgpu::VmAccessOutcome::Unavailable)
      return DoorbellDisposition::Retry;
    if (publication != amdgpu::VmAccessOutcome::Complete)
      return DoorbellDisposition::Faulted;

    read_pointer = pending_it->second.next_read_pointer;
    pending_mes_frames_.erase(pending_it);
  }

  (void)registers_.write(kGcSegment, kQueueReadPointer, static_cast<uint32_t>(read_pointer));
  (void)registers_.write(kGcSegment, kQueueWritePointerLow, static_cast<uint32_t>(write_pointer));
  (void)registers_.write(kGcSegment, kQueueWritePointerHigh,
                         static_cast<uint32_t>(write_pointer >> 32));
  return DoorbellDisposition::Complete;
}

DoorbellDisposition MesBlockModel::process_compute_queue(Queue queue, uint64_t write_pointer,
                                                         PciMemoryAccess &memory) {
  if (!queue.active || queue.ring_base == 0 || queue.read_pointer_address == 0 ||
      queue.ring_dwords == 0 || soc_ == nullptr) {
    return DoorbellDisposition::Faulted;
  }

  const amdgpu::AddressSpaceHandle address_space =
      queue.address_space ? queue.address_space : soc_->gpu_vm().gart_address_space();
  if (!queue.address_space && queue.page_table_base != 0)
    return DoorbellDisposition::Faulted;

  auto pending = pending_compute_runs_.find(queue.doorbell_offset);
  if (pending == pending_compute_runs_.end()) {
    std::optional<amdgpu::GpuVmAccess> access = soc_->gpu_vm().snapshot(address_space);
    if (!access)
      return DoorbellDisposition::Faulted;
    const amdgpu::AtomicLoadResult loaded =
        access->atomic_load(queue.read_pointer_address, sizeof(uint32_t));
    if (loaded.outcome != amdgpu::VmAccessOutcome::Complete) {
      return loaded.outcome == amdgpu::VmAccessOutcome::Unavailable ? DoorbellDisposition::Retry
                                                                    : DoorbellDisposition::Faulted;
    }
    const std::shared_ptr<simdojo::PciTransportSession> transport_session =
        memory.capture_transport_session();
    pending =
        pending_compute_runs_
            .try_emplace(queue.doorbell_offset,
                         PendingComputeRun{
                             .queue = queue,
                             .access = std::move(*access),
                             .read_pointer = static_cast<uint32_t>(loaded.value),
                             .phase = ComputeRunPhase::Execute,
                             .transport_session = transport_session,
                             .transport_generation =
                                 transport_session != nullptr ? transport_session->generation() : 0,
                         })
            .first;
  }

  PendingComputeRun &run = pending->second;
  if (run.phase == ComputeRunPhase::Terminal)
    return DoorbellDisposition::Faulted;
  const std::shared_ptr<simdojo::PciTransportSession> retained_session =
      run.transport_session.lock();
  const std::shared_ptr<simdojo::PciTransportSession> current_session =
      memory.capture_transport_session();
  if (retained_session == nullptr || retained_session != current_session ||
      retained_session->generation() != run.transport_generation ||
      run.queue.ring_base != queue.ring_base ||
      run.queue.read_pointer_address != queue.read_pointer_address ||
      run.queue.ring_dwords != queue.ring_dwords || write_pointer < run.read_pointer ||
      write_pointer - run.read_pointer > queue.ring_dwords) {
    run.phase = ComputeRunPhase::Terminal;
    run.terminal_outcome = amdgpu::VmAccessOutcome::Malformed;
    return DoorbellDisposition::Faulted;
  }

  amdgpu::Pm4BootstrapExecutor executor(
      {.write_uconfig_register = [this](uint64_t register_dword, uint32_t value) {
        const std::optional<uint64_t> scratch =
            registers_.resolve(kGcControlSegment, kScratchRegister0);
        return scratch && register_dword == *scratch / sizeof(uint32_t) &&
               registers_.write(kGcControlSegment, kScratchRegister0, value);
      }});
  for (;;) {
    if (run.phase == ComputeRunPhase::Execute) {
      const amdgpu::Pm4BootstrapResult result = executor.execute(
          run.access, queue.ring_base, queue.ring_dwords, run.read_pointer, write_pointer);
      run.read_pointer = result.read_pointer;
      if (result.outcome == amdgpu::Pm4BootstrapOutcome::Unavailable)
        return DoorbellDisposition::Retry;
      if (result.outcome == amdgpu::Pm4BootstrapOutcome::UnsupportedPacket) {
        util::Logger::warn(std::format("{}: unsupported compute packet {:#010x}",
                                       registers_.owner(), result.packet_header));
      } else if (result.outcome == amdgpu::Pm4BootstrapOutcome::RegisterWriteRejected) {
        util::Logger::warn(std::format("{}: unsupported compute register write to {:#x}",
                                       registers_.owner(), result.register_dword));
      }
      if (result.outcome != amdgpu::Pm4BootstrapOutcome::Complete) {
        run.phase = ComputeRunPhase::Terminal;
        run.terminal_outcome = result.outcome == amdgpu::Pm4BootstrapOutcome::Faulted
                                   ? amdgpu::VmAccessOutcome::Faulted
                                   : amdgpu::VmAccessOutcome::Malformed;
        return DoorbellDisposition::Faulted;
      }
      run.phase = ComputeRunPhase::ReadPointer;
    }

    const auto raw_pointer = std::bit_cast<std::array<std::byte, sizeof(uint32_t)>>(
        static_cast<uint32_t>(run.read_pointer));
    const amdgpu::VmAccessOutcome published =
        run.access.write(queue.read_pointer_address, raw_pointer);
    if (published == amdgpu::VmAccessOutcome::Unavailable)
      return DoorbellDisposition::Retry;
    if (published != amdgpu::VmAccessOutcome::Complete) {
      run.phase = ComputeRunPhase::Terminal;
      run.terminal_outcome = published;
      return DoorbellDisposition::Faulted;
    }
    if (run.read_pointer == write_pointer) {
      pending_compute_runs_.erase(pending);
      return DoorbellDisposition::Complete;
    }
    run.phase = ComputeRunPhase::Execute;
  }
}

DoorbellDisposition MesBlockModel::observe_doorbell_write(uint64_t byte_offset,
                                                          uint64_t write_pointer, std::size_t width,
                                                          PciMemoryAccess &memory) {
  if (width != sizeof(uint32_t) && width != sizeof(uint64_t)) {
    return DoorbellDisposition::Ignored;
  }
  util::Logger::cp([&](auto &os) {
    os << std::format("{}: doorbell {:#x} <- {}", registers_.owner(), byte_offset, write_pointer);
  });
  const auto mapped = std::ranges::find(mapped_queues_, byte_offset, &Queue::doorbell_offset);
  if (mapped != mapped_queues_.end()) {
    if (mapped->queue_handle && soc_ != nullptr) {
      const amdgpu::QueueDoorbellResult result =
          soc_->queue_service().notify_doorbell_result(mapped->queue_handle, write_pointer);
      if (!result)
        return DoorbellDisposition::Faulted;
      switch (result.disposition) {
      case amdgpu::QueueDoorbellDisposition::Complete:
        return DoorbellDisposition::Complete;
      case amdgpu::QueueDoorbellDisposition::Retry:
        return DoorbellDisposition::Retry;
      case amdgpu::QueueDoorbellDisposition::Faulted:
        return DoorbellDisposition::Faulted;
      }
    }
    return process_queue(*mapped, write_pointer, memory);
  }
  Queue queue = direct_queue();
  if (queue.doorbell_offset == byte_offset)
    return process_queue(queue, write_pointer, memory);
  return DoorbellDisposition::Ignored;
}

} // namespace rocjitsu
