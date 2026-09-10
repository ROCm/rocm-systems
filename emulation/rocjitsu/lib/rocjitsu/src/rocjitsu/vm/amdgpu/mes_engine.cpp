// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/mes_engine.h"

#include "rocjitsu/vm/amdgpu/sdma_queue_binding_factory.h"
#include "util/log.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <exception>
#include <format>
#include <optional>
#include <utility>

namespace rocjitsu::amdgpu {
namespace {
constexpr uint32_t kDoorbellOffsetMask = 0x0ffffffc;
constexpr uint32_t kQueueSizeMask = 0x3f;

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
constexpr uint32_t kMesManagedQueueIdBase = uint32_t{1} << 31;
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

bool MesEngine::attach_frontend(std::string diagnostic_name,
                                std::shared_ptr<SdmaQueueBindingFactory> sdma_queue_binding_factory,
                                InterruptSink interrupt_sink, MesFrontendCallbacks callbacks) {
  reap_orphaned_address_spaces();
  if (frontend_attached_ || !mapped_queues_.empty() || !pending_mes_frames_.empty()) {
    return false;
  }
  diagnostic_name_ = std::move(diagnostic_name);
  sdma_queue_binding_factory_ = std::move(sdma_queue_binding_factory);
  interrupt_sink_ = std::move(interrupt_sink);
  callbacks_ = std::move(callbacks);
  frontend_attached_ = true;
  return true;
}

bool MesEngine::detach_frontend() {
  if (!mapped_queues_.empty() || !pending_mes_frames_.empty()) {
    return false;
  }
  // An address-space unregister can be refused while another core client still
  // holds a queue reference. The GpuVm binding then owns its own retained
  // backing and lifetime; it must not keep this frontend's MES bookkeeping or
  // prevent an unrelated replacement frontend from attaching after the holder
  // releases and unregisters it.
  for (const auto &[process_id, address_space] : address_spaces_) {
    (void)process_id;
    orphaned_address_spaces_.push_back(address_space.handle);
  }
  address_spaces_.clear();
  sdma_queue_binding_factory_ = nullptr;
  interrupt_sink_ = {};
  callbacks_ = {};
  diagnostic_name_ = "MES";
  frontend_attached_ = false;
  return true;
}

bool MesEngine::reset() {
  if (!teardown_queues())
    return false;
  next_queue_ordinal_ = 0;
  return true;
}

bool MesEngine::teardown_queues() {
  reap_orphaned_address_spaces();
  // Teardown is a transaction boundary: once it starts destroying queue state,
  // no pre-reset semantic may later publish success through a retained snapshot,
  // even if destruction makes partial progress and the caller reports failure.
  pending_mes_frames_.clear();

  bool complete = true;
  for (std::vector<Queue>::iterator queue = mapped_queues_.begin();
       queue != mapped_queues_.end();) {
    bool released = true;
    if (queue->queue_handle)
      released = static_cast<bool>(
          queue_registry_.unregister_queue(queue->queue_handle, QueueCloseMode::ForceCancel));
    if (!released) {
      complete = false;
      ++queue;
      continue;
    }
    queue = mapped_queues_.erase(queue);
  }

  for (std::unordered_map<uint32_t, AddressSpace>::iterator address_space = address_spaces_.begin();
       address_space != address_spaces_.end();) {
    const uint32_t process_id = address_space->first;
    const bool still_used =
        std::ranges::any_of(mapped_queues_, [process_id](const Queue &candidate) {
          return candidate.process_id == process_id && candidate.address_space;
        });
    if (still_used || !gpu_vm_.unregister_address_space(address_space->second.handle)) {
      complete = false;
      ++address_space;
      continue;
    }
    address_space = address_spaces_.erase(address_space);
  }
  return complete;
}

void MesEngine::reap_orphaned_address_spaces() {
  std::erase_if(orphaned_address_spaces_, [this](AddressSpaceHandle handle) {
    return !gpu_vm_.lookup(handle) || gpu_vm_.unregister_address_space(handle);
  });
}

MesEngine::Queue MesEngine::kernel_queue(const MesKernelQueue &queue) {
  return {
      .ring_base = queue.ring_base,
      .read_pointer_address = queue.read_pointer_address,
      .write_pointer_address = queue.write_pointer_address,
      .ring_dwords = queue.ring_dwords,
      .doorbell_offset = queue.doorbell_offset,
      .active = queue.active,
      .aql = false,
      .address_space = {},
      .queue_handle = {},
      .kind = QueueKind::Mes,
  };
}

MesEngine::QueueLookup MesEngine::queue_from_mqd(uint64_t mqd_address, QueueKind kind,
                                                 const amdgpu::GpuVmAccess &gart_access) const {
  if (kind == QueueKind::Sdma) {
    constexpr std::size_t kSdmaMqdBytes = (kSdmaMqdQueueIdDword + 1) * sizeof(uint32_t);
    std::array<std::byte, kSdmaMqdBytes> mqd{};
    if (mqd_address == 0)
      return {amdgpu::VmAccessOutcome::Malformed, std::nullopt};
    const amdgpu::VmAccessOutcome outcome = gart_access.read(mqd_address, mqd);
    if (outcome != amdgpu::VmAccessOutcome::Complete)
      return {outcome, std::nullopt};
    const uint32_t queue_size =
        (dword(mqd, kSdmaMqdQueueControlDword) & kSdmaQueueSizeMask) >> kSdmaQueueSizeShift;
    if (queue_size >= 30)
      return {amdgpu::VmAccessOutcome::Malformed, std::nullopt};
    return {
        amdgpu::VmAccessOutcome::Complete,
        Queue{
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
    return {amdgpu::VmAccessOutcome::Malformed, std::nullopt};
  const amdgpu::VmAccessOutcome outcome =
      gart_access.read(mqd_address + kMqdFirstDword * sizeof(uint32_t), mqd);
  if (outcome != amdgpu::VmAccessOutcome::Complete)
    return {outcome, std::nullopt};
  const auto field = [&mqd](uint32_t absolute_dword) {
    return dword(mqd, absolute_dword - kMqdFirstDword);
  };
  const uint32_t queue_size = field(kMqdQueueControlDword) & kQueueSizeMask;
  if (queue_size >= 30)
    return {amdgpu::VmAccessOutcome::Malformed, std::nullopt};
  return {amdgpu::VmAccessOutcome::Complete,
          Queue{
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

bool MesEngine::bind_process_address_space(Queue &queue,
                                           std::shared_ptr<PhysicalMemoryAccess> physical_memory,
                                           bool &created_address_space) {
  created_address_space = false;
  if (queue.page_table_base == 0 || physical_memory == nullptr) {
    return false;
  }

  std::unordered_map<uint32_t, AddressSpace>::iterator existing_address_space =
      address_spaces_.find(queue.process_id);
  if (existing_address_space == address_spaces_.end()) {
    const amdgpu::AddressSpaceHandle handle = gpu_vm_.register_gfx12_address_space(
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

bool MesEngine::update_process_address_space(uint64_t process_context_address,
                                             uint64_t page_table_base,
                                             std::shared_ptr<PhysicalMemoryAccess> memory) {
  if (process_context_address == 0 || page_table_base == 0 || memory == nullptr)
    return false;

  std::unordered_map<uint32_t, AddressSpace>::iterator matching = address_spaces_.end();
  for (std::unordered_map<uint32_t, AddressSpace>::iterator candidate = address_spaces_.begin();
       candidate != address_spaces_.end(); ++candidate) {
    if (candidate->second.process_context_address != process_context_address)
      continue;
    if (matching != address_spaces_.end())
      return false;
    matching = candidate;
  }
  if (matching == address_spaces_.end())
    return false;

  if (!gpu_vm_.replace_gfx12_address_space_root(matching->second.handle, page_table_base,
                                                std::move(memory)))
    return false;

  matching->second.page_table_base = page_table_base;
  for (Queue &queue : mapped_queues_) {
    if (queue.address_space == matching->second.handle)
      queue.page_table_base = page_table_base;
  }
  return true;
}

bool MesEngine::register_aql_queue(Queue &queue) {
  if (queue.write_pointer_address == 0 || queue.ring_dwords > UINT32_MAX / sizeof(uint32_t)) {
    return false;
  }
  if (next_queue_ordinal_ >= kMesManagedQueueIdBase)
    return false;
  const std::optional<ComputeQueueBindingPlan> plan =
      compute_bindings_.make_aql(next_queue_ordinal_);
  if (!plan || !plan->factory)
    return false;
  queue.queue_id = kMesManagedQueueIdBase | next_queue_ordinal_;
  try {
    queue.queue_handle = queue_registry_.register_queue({
        .identity = {.address_space = queue.address_space,
                     .interrupt_sink = interrupt_sink_,
                     .process_id = queue.process_id,
                     .queue_id = queue.queue_id},
        .ring = {.base_address = queue.ring_base,
                 .size_bytes = static_cast<uint32_t>(queue.ring_dwords * sizeof(uint32_t)),
                 .consumer_pointer_address = queue.read_pointer_address,
                 .producer_pointer_address = queue.write_pointer_address},
        .doorbell = callbacks_.queue_doorbell(static_cast<uint32_t>(queue.doorbell_offset),
                                              QueuePacketFormat::Aql),
        .binding_factory = plan->factory,
        .xcd_fanout = plan->xcd_fanout,
        .type = amdgpu::QueueType::Compute,
        .packet_format = amdgpu::QueuePacketFormat::Aql,
    });
  } catch (const std::exception &error) {
    util::Logger::warn(
        std::format("{}: cannot create AQL queue binding: {}", diagnostic_name_, error.what()));
    return false;
  }
  return static_cast<bool>(queue.queue_handle);
}

bool MesEngine::register_pm4_queue(Queue &queue) {
  if (queue.ring_dwords == 0 || queue.ring_dwords > UINT32_MAX / sizeof(uint32_t) ||
      next_queue_ordinal_ >= kMesManagedQueueIdBase) {
    return false;
  }
  queue.queue_id = kMesManagedQueueIdBase | next_queue_ordinal_;
  amdgpu::Pm4PacketCallbacks packet_callbacks{
      .write_uconfig_register = [writer = callbacks_.pm4_uconfig_writer()](uint64_t register_dword,
                                                                           uint32_t value) {
        if (!writer || !writer(register_dword, value))
          return amdgpu::Pm4RegisterWriteStatus::Rejected;
        return amdgpu::Pm4RegisterWriteStatus::Complete;
      }};
  const std::optional<ComputeQueueBindingPlan> plan =
      compute_bindings_.make_pm4(next_queue_ordinal_, std::move(packet_callbacks));
  if (!plan || !plan->factory)
    return false;
  try {
    queue.queue_handle = queue_registry_.register_queue({
        .identity = {.address_space = queue.address_space,
                     .interrupt_sink = interrupt_sink_,
                     .process_id = queue.process_id,
                     .queue_id = queue.queue_id},
        .ring = {.base_address = queue.ring_base,
                 .size_bytes = static_cast<uint32_t>(queue.ring_dwords * sizeof(uint32_t)),
                 .consumer_pointer_address = queue.read_pointer_address,
                 .producer_pointer_address = queue.write_pointer_address},
        .doorbell = callbacks_.queue_doorbell(static_cast<uint32_t>(queue.doorbell_offset),
                                              QueuePacketFormat::Pm4),
        .binding_factory = plan->factory,
        .xcd_fanout = plan->xcd_fanout,
        .type = amdgpu::QueueType::Compute,
        .packet_format = amdgpu::QueuePacketFormat::Pm4,
    });
  } catch (const std::exception &error) {
    util::Logger::warn(
        std::format("{}: cannot create PM4 queue binding: {}", diagnostic_name_, error.what()));
    return false;
  }
  return static_cast<bool>(queue.queue_handle);
}

bool MesEngine::register_sdma_queue(Queue &queue) {
  if (sdma_queue_binding_factory_ == nullptr || queue.ring_dwords == 0 ||
      queue.ring_dwords > UINT32_MAX / sizeof(uint32_t) ||
      next_queue_ordinal_ >= kMesManagedQueueIdBase) {
    return false;
  }
  queue.queue_id = kMesManagedQueueIdBase | next_queue_ordinal_;
  try {
    queue.queue_handle = queue_registry_.register_queue({
        .identity = {.address_space = queue.address_space,
                     .interrupt_sink = interrupt_sink_,
                     .process_id = queue.process_id,
                     .queue_id = queue.queue_id},
        .ring = {.base_address = queue.ring_base,
                 .size_bytes = static_cast<uint32_t>(queue.ring_dwords * sizeof(uint32_t)),
                 .consumer_pointer_address = queue.read_pointer_address,
                 .producer_pointer_address = queue.write_pointer_address},
        .doorbell = callbacks_.queue_doorbell(static_cast<uint32_t>(queue.doorbell_offset),
                                              QueuePacketFormat::Sdma),
        .binding_factory = sdma_queue_binding_factory_,
        .initial_consumer_cursor = queue.initial_read_pointer,
        .engine_id = queue.engine_id,
        .type = amdgpu::QueueType::Sdma,
        .packet_format = amdgpu::QueuePacketFormat::Sdma,
    });
  } catch (const std::exception &error) {
    util::Logger::warn(
        std::format("{}: cannot create SDMA queue binding: {}", diagnostic_name_, error.what()));
    return false;
  }
  return static_cast<bool>(queue.queue_handle);
}

void MesEngine::rollback_created_address_space(const Queue &queue, bool created_address_space) {
  if (!created_address_space || !queue.address_space)
    return;
  (void)gpu_vm_.unregister_address_space(queue.address_space);
  address_spaces_.erase(queue.process_id);
}

void MesEngine::release_unused_address_space(uint32_t process_id) {
  const bool process_still_used =
      std::ranges::any_of(mapped_queues_, [process_id](const Queue &candidate) {
        return candidate.process_id == process_id && candidate.address_space;
      });
  if (process_still_used)
    return;
  const std::unordered_map<uint32_t, AddressSpace>::iterator address_space =
      address_spaces_.find(process_id);
  if (address_space != address_spaces_.end() &&
      gpu_vm_.unregister_address_space(address_space->second.handle)) {
    address_spaces_.erase(address_space);
  }
}

VmAccessOutcome MesEngine::commit_queue(Queue queue, bool created_address_space) {
  util::Logger::cp([&](auto &os) {
    os << std::format("{}: map queue kind={} ring={:#x} rptr={:#x} wptr={:#x} pt={:#x} "
                      "dwords={} doorbell={:#x} active={} aql={}",
                      diagnostic_name_, static_cast<unsigned>(queue.kind), queue.ring_base,
                      queue.read_pointer_address, queue.write_pointer_address,
                      queue.page_table_base, queue.ring_dwords, queue.doorbell_offset, queue.active,
                      queue.aql);
  });
  // A retained publication is identified by this doorbell and its queue
  // configuration. Replacing that queue would either strand the journal or let
  // it acknowledge work against a different queue generation.
  if (pending_mes_frames_.contains(queue.doorbell_offset)) {
    if (queue.queue_handle)
      (void)queue_registry_.unregister_queue(queue.queue_handle, QueueCloseMode::ForceCancel);
    rollback_created_address_space(queue, created_address_space);
    return VmAccessOutcome::Unavailable;
  }
  const std::vector<Queue>::iterator existing =
      std::ranges::find(mapped_queues_, queue.doorbell_offset, &Queue::doorbell_offset);
  if (existing == mapped_queues_.end()) {
    try {
      mapped_queues_.push_back(queue);
    } catch (...) {
      if (queue.queue_handle)
        (void)queue_registry_.unregister_queue(queue.queue_handle, QueueCloseMode::ForceCancel);
      rollback_created_address_space(queue, created_address_space);
      return VmAccessOutcome::Faulted;
    }
    if (queue.queue_handle)
      ++next_queue_ordinal_;
    if (queue.address_space && queue.process_context_address != 0) {
      AddressSpace &address_space = address_spaces_.at(queue.process_id);
      if (address_space.process_context_address == 0)
        address_space.process_context_address = queue.process_context_address;
    }
    return VmAccessOutcome::Complete;
  }

  const Queue replaced = *existing;
  QueueCloseResult detached{.found = true, .status = QueueCloseStatus::Closed};
  if (replaced.queue_handle)
    detached = queue_registry_.unregister_queue(replaced.queue_handle);
  if (!detached) {
    if (queue.queue_handle)
      (void)queue_registry_.unregister_queue(queue.queue_handle, QueueCloseMode::ForceCancel);
    rollback_created_address_space(queue, created_address_space);
    return detached.status == QueueCloseStatus::Busy ? VmAccessOutcome::Unavailable
                                                     : VmAccessOutcome::Faulted;
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
  return VmAccessOutcome::Complete;
}

VmAccessOutcome MesEngine::remove_queue(uint64_t doorbell_offset) {
  // Destruction of the queue that owns a pending frame would remove the only
  // route by which its completion/read-pointer publication can resume.
  if (pending_mes_frames_.contains(doorbell_offset))
    return VmAccessOutcome::Unavailable;

  const std::vector<Queue>::iterator queue =
      std::ranges::find(mapped_queues_, doorbell_offset, &Queue::doorbell_offset);
  if (queue == mapped_queues_.end())
    return VmAccessOutcome::Malformed;

  const uint32_t process_id = queue->process_id;
  if (queue->queue_handle) {
    const QueueCloseResult closed = queue_registry_.unregister_queue(queue->queue_handle);
    if (!closed) {
      return closed.status == QueueCloseStatus::Busy ? VmAccessOutcome::Unavailable
                                                     : VmAccessOutcome::Faulted;
    }
  }
  if (!queue->queue_handle && queue->kind == QueueKind::Sdma) {
    return VmAccessOutcome::Faulted;
  }
  mapped_queues_.erase(queue);

  release_unused_address_space(process_id);
  return VmAccessOutcome::Complete;
}

MesDoorbellDisposition MesEngine::process_queue(Queue queue, uint64_t write_pointer,
                                                const MesDoorbellContext &context) {
  switch (queue.kind) {
  case QueueKind::Mes:
    return process_mes_queue(queue, write_pointer, context);
  case QueueKind::Compute:
  case QueueKind::Sdma:
    return MesDoorbellDisposition::Faulted;
  }
  return MesDoorbellDisposition::Faulted;
}

amdgpu::VmAccessOutcome MesEngine::execute_mes_semantic(std::span<const std::byte> frame,
                                                        uint32_t opcode,
                                                        const MesDoorbellContext &context,
                                                        const amdgpu::GpuVmAccess &gart_access) {
  if (opcode == kMesApiAddQueue) {
    reap_orphaned_address_spaces();
    const uint64_t mqd_address = qword(frame, kAddQueueMqdAddressDword);
    const uint32_t queue_type = dword(frame, kAddQueueTypeDword);
    QueueKind kind = QueueKind::Mes;
    if (queue_type == kMesQueueTypeCompute) {
      kind = QueueKind::Compute;
    } else if (queue_type == kMesQueueTypeSdma) {
      kind = QueueKind::Sdma;
    } else if (queue_type == kMesQueueTypeGfx || queue_type != kMesQueueTypeScheduler) {
      util::Logger::warn(
          std::format("{}: unsupported MES queue type {}", diagnostic_name_, queue_type));
      return amdgpu::VmAccessOutcome::Malformed;
    }
    QueueLookup lookup = queue_from_mqd(mqd_address, kind, gart_access);
    if (lookup.outcome_ != amdgpu::VmAccessOutcome::Complete || !lookup.queue_) {
      util::Logger::warn(std::format("{}: cannot read the MES ADD_QUEUE MQD at {:#x}",
                                     diagnostic_name_, mqd_address));
      return lookup.outcome_;
    }
    Queue &mapped = *lookup.queue_;
    mapped.page_table_base = qword(frame, 2);
    mapped.process_context_address = qword(frame, kAddQueueProcessContextAddressDword);
    // MES's process_id is the PASID. CP_HQD_VMID is a hardware VMID and is
    // still zero in the MQD when the driver hands the queue to MES, before
    // firmware would assign one.
    mapped.process_id = dword(frame, 1);
    bool created_address_space = false;
    std::shared_ptr<PhysicalMemoryAccess> physical_memory;
    if (mapped.kind == QueueKind::Compute || mapped.kind == QueueKind::Sdma) {
      physical_memory = context.make_physical_memory();
    }
    if (mapped.kind == QueueKind::Compute &&
        !bind_process_address_space(mapped, physical_memory, created_address_space)) {
      util::Logger::warn(std::format("{}: cannot attach compute queue at doorbell {:#x}",
                                     diagnostic_name_, mapped.doorbell_offset));
      return amdgpu::VmAccessOutcome::Faulted;
    }
    if (mapped.kind == QueueKind::Sdma) {
      mapped.doorbell_offset =
          static_cast<uint64_t>(dword(frame, kAddQueueDoorbellDword)) * sizeof(uint32_t);
      mapped.write_pointer_address = qword(frame, kAddQueueWritePointerAddressDword);
      if (const uint32_t ring_dwords = dword(frame, kAddQueueSizeDword); ring_dwords != 0)
        mapped.ring_dwords = ring_dwords;
      if (!bind_process_address_space(mapped, std::move(physical_memory), created_address_space) ||
          !register_sdma_queue(mapped)) {
        rollback_created_address_space(mapped, created_address_space);
        util::Logger::warn(std::format("{}: cannot attach SDMA queue at doorbell {:#x}",
                                       diagnostic_name_, mapped.doorbell_offset));
        return amdgpu::VmAccessOutcome::Faulted;
      }
    }
    if (mapped.kind == QueueKind::Compute) {
      const bool registered = mapped.aql ? register_aql_queue(mapped) : register_pm4_queue(mapped);
      if (!registered) {
        rollback_created_address_space(mapped, created_address_space);
        util::Logger::warn(std::format("{}: cannot attach {} queue at doorbell {:#x}",
                                       diagnostic_name_, mapped.aql ? "AQL" : "PM4",
                                       mapped.doorbell_offset));
        return amdgpu::VmAccessOutcome::Faulted;
      }
    }
    const amdgpu::VmAccessOutcome committed = commit_queue(mapped, created_address_space);
    if (committed != amdgpu::VmAccessOutcome::Complete) {
      util::Logger::warn(std::format("{}: cannot commit queue at doorbell {:#x}", diagnostic_name_,
                                     mapped.doorbell_offset));
      return committed;
    }
  } else if (opcode == kMesApiRemoveQueue) {
    const uint64_t doorbell_offset =
        static_cast<uint64_t>(dword(frame, kRemoveQueueDoorbellDword)) * sizeof(uint32_t);
    const amdgpu::VmAccessOutcome removed = remove_queue(doorbell_offset);
    if (removed != amdgpu::VmAccessOutcome::Complete) {
      util::Logger::warn(std::format("{}: cannot remove queue at doorbell {:#x}", diagnostic_name_,
                                     doorbell_offset));
      return removed;
    }
  } else if (opcode == kMesApiUpdateRootPageTable) {
    const uint64_t page_table_base = qword(frame, kUpdateRootPageTableBaseDword);
    const uint64_t process_context_address = qword(frame, kUpdateRootProcessContextAddressDword);
    std::shared_ptr<PhysicalMemoryAccess> physical_memory = context.make_physical_memory();
    if (!update_process_address_space(process_context_address, page_table_base,
                                      std::move(physical_memory))) {
      util::Logger::warn(std::format("{}: cannot update root {:#x} for process context {:#x}",
                                     diagnostic_name_, page_table_base, process_context_address));
      return amdgpu::VmAccessOutcome::Faulted;
    }
  } else if (opcode == kMesApiInvalidateTlbs) {
    const uint32_t selector = dword(frame, kInvalidateTlbsSelectorDword);
    const uint8_t selector_kind = static_cast<uint8_t>(selector);
    const uint16_t selector_id = static_cast<uint16_t>(selector >> 16);
    if (selector_kind != 0) {
      util::Logger::warn(std::format("{}: unsupported MES TLB invalidation selector {} for id {}",
                                     diagnostic_name_, selector_kind, selector_id));
      return amdgpu::VmAccessOutcome::Malformed;
    }
    const std::optional<amdgpu::AddressSpaceHandle> address_space = gpu_vm_.find_vmid(selector_id);
    if (!address_space || !gpu_vm_.invalidate(*address_space)) {
      util::Logger::warn(
          std::format("{}: cannot invalidate PASID {}", diagnostic_name_, selector_id));
      return amdgpu::VmAccessOutcome::Faulted;
    }
  }
  return amdgpu::VmAccessOutcome::Complete;
}

amdgpu::VmAccessOutcome MesEngine::publish_mes_frame(PendingMesFrame &pending) {
  if (pending.phase_ == MesFramePhase::Semantic)
    return amdgpu::VmAccessOutcome::Malformed;
  if (pending.phase_ == MesFramePhase::Terminal)
    return pending.terminal_outcome_;

  if (pending.phase_ == MesFramePhase::Completion) {
    const std::array<std::byte, sizeof(uint64_t)> raw_completion =
        std::bit_cast<std::array<std::byte, sizeof(uint64_t)>>(pending.completion_value_);
    const amdgpu::VmAccessOutcome outcome =
        pending.access_.write(pending.completion_address_, raw_completion);
    if (outcome == amdgpu::VmAccessOutcome::Unavailable)
      return outcome;
    if (outcome != amdgpu::VmAccessOutcome::Complete) {
      pending.phase_ = MesFramePhase::Terminal;
      pending.terminal_outcome_ = outcome;
      return outcome;
    }
    pending.phase_ = MesFramePhase::ReadPointer;
  }

  const std::array<std::byte, sizeof(uint64_t)> raw_pointer =
      std::bit_cast<std::array<std::byte, sizeof(uint64_t)>>(pending.next_read_pointer_);
  const amdgpu::VmAccessOutcome outcome =
      pending.access_.write(pending.queue_.read_pointer_address, raw_pointer);
  if (outcome != amdgpu::VmAccessOutcome::Complete &&
      outcome != amdgpu::VmAccessOutcome::Unavailable) {
    pending.phase_ = MesFramePhase::Terminal;
    pending.terminal_outcome_ = outcome;
  }
  return outcome;
}

bool MesEngine::same_source(const PendingMesFrame &pending,
                            const MesDoorbellContext &context) const {
  return pending.source_identity_ != 0 && pending.source_identity_ == context.source_identity() &&
         pending.source_generation_ == context.source_generation();
}

MesDoorbellDisposition MesEngine::process_mes_queue(Queue queue, uint64_t write_pointer,
                                                    const MesDoorbellContext &context) {
  const uint64_t ring_dwords = queue.ring_dwords;
  const uint64_t ring_bytes = ring_dwords * sizeof(uint32_t);
  if (!queue.active || queue.ring_base == 0 || queue.read_pointer_address == 0 ||
      ring_dwords == 0 || write_pointer < kMesFrameDwords) {
    return MesDoorbellDisposition::Faulted;
  }

  uint64_t read_pointer = 0;
  std::optional<amdgpu::GpuVmAccess> access;
  const std::unordered_map<uint64_t, PendingMesFrame>::iterator pending_frame =
      pending_mes_frames_.find(queue.doorbell_offset);
  if (pending_frame != pending_mes_frames_.end()) {
    PendingMesFrame &pending = pending_frame->second;
    if (pending.phase_ == MesFramePhase::Terminal)
      return MesDoorbellDisposition::Faulted;
    if (!same_source(pending, context)) {
      pending.phase_ = MesFramePhase::Terminal;
      pending.terminal_outcome_ = amdgpu::VmAccessOutcome::Malformed;
      return MesDoorbellDisposition::Faulted;
    }
    if (pending.queue_.ring_base != queue.ring_base ||
        pending.queue_.read_pointer_address != queue.read_pointer_address ||
        pending.queue_.ring_dwords != queue.ring_dwords ||
        pending.next_read_pointer_ > write_pointer) {
      util::Logger::warn(
          std::format("{}: MES queue {:#x} changed while frame {} awaited publication",
                      diagnostic_name_, queue.doorbell_offset, pending.frame_read_pointer_));
      pending.phase_ = MesFramePhase::Terminal;
      pending.terminal_outcome_ = amdgpu::VmAccessOutcome::Malformed;
      return MesDoorbellDisposition::Faulted;
    }
    const amdgpu::VmAccessOutcome outcome = publish_mes_frame(pending);
    if (outcome == amdgpu::VmAccessOutcome::Unavailable)
      return MesDoorbellDisposition::Retry;
    if (outcome != amdgpu::VmAccessOutcome::Complete)
      return MesDoorbellDisposition::Faulted;
    read_pointer = pending.next_read_pointer_;
    pending_mes_frames_.erase(pending_frame);
  } else {
    const amdgpu::AddressSpaceHandle gart = gpu_vm_.gart_address_space();
    access = gpu_vm_.snapshot(gart);
    if (!access)
      return MesDoorbellDisposition::Faulted;
    std::array<std::byte, sizeof(uint64_t)> raw_read_pointer{};
    const amdgpu::VmAccessOutcome outcome =
        access->read(queue.read_pointer_address, raw_read_pointer);
    if (outcome != amdgpu::VmAccessOutcome::Complete) {
      util::Logger::warn(std::format("{}: cannot read the MES queue read pointer at {:#x}",
                                     diagnostic_name_, queue.read_pointer_address));
      return outcome == amdgpu::VmAccessOutcome::Unavailable ? MesDoorbellDisposition::Retry
                                                             : MesDoorbellDisposition::Faulted;
    }
    read_pointer = std::bit_cast<uint64_t>(raw_read_pointer);
  }

  const uint64_t pending = write_pointer - read_pointer;
  if (pending > ring_dwords || (pending % kMesFrameDwords) != 0) {
    util::Logger::warn(std::format("{}: MES doorbell moved from {} to {} dwords in a {}-dword ring",
                                   diagnostic_name_, read_pointer, write_pointer, ring_dwords));
    return MesDoorbellDisposition::Faulted;
  }

  if (read_pointer != write_pointer && !access) {
    access = gpu_vm_.snapshot(gpu_vm_.gart_address_space());
    if (!access)
      return MesDoorbellDisposition::Faulted;
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
      util::Logger::warn(std::format("{}: cannot read a MES frame at {:#x}", diagnostic_name_,
                                     queue.ring_base + ring_offset));
      return outcome == amdgpu::VmAccessOutcome::Unavailable ? MesDoorbellDisposition::Retry
                                                             : MesDoorbellDisposition::Faulted;
    }

    const uint32_t header = dword(frame, 0);
    const uint32_t type = header & 0xf;
    const uint32_t opcode = (header >> 4) & 0xff;
    const uint32_t frame_dwords = (header >> 12) & 0xff;
    const std::optional<uint32_t> status_at = status_dword(opcode);
    if (type != kMesApiTypeScheduler || frame_dwords != kMesFrameDwords || !status_at) {
      util::Logger::warn(std::format("{}: unsupported MES frame type {}, opcode {}, size {} dwords",
                                     diagnostic_name_, type, opcode, frame_dwords));
      return MesDoorbellDisposition::Faulted;
    }

    const uint64_t completion_address = qword(frame, *status_at);
    const uint64_t completion_value = qword(frame, *status_at + 2);
    if (completion_address == 0) {
      util::Logger::warn(std::format("{}: cannot publish MES opcode {} completion at {:#x}",
                                     diagnostic_name_, opcode, completion_address));
      return MesDoorbellDisposition::Faulted;
    }

    PendingMesFrame pending_frame(queue, *access, read_pointer, read_pointer + kMesFrameDwords,
                                  completion_address, completion_value, context.source_identity(),
                                  context.source_generation());
    auto [pending_it, inserted] =
        pending_mes_frames_.try_emplace(queue.doorbell_offset, std::move(pending_frame));
    if (!inserted)
      return MesDoorbellDisposition::Faulted;

    const amdgpu::VmAccessOutcome semantic = execute_mes_semantic(frame, opcode, context, *access);
    if (semantic == amdgpu::VmAccessOutcome::Unavailable) {
      // Retryable semantic dependencies either precede any state change (for
      // example, an unavailable ADD_QUEUE MQD) or preserve the prior queue
      // generation until its committed consumer progress has been published.
      pending_mes_frames_.erase(pending_it);
      return MesDoorbellDisposition::Retry;
    }
    if (semantic != amdgpu::VmAccessOutcome::Complete) {
      // Semantic helpers either commit completely or roll back before reporting
      // failure. Leave a rejected frame unretired so the guest may correct it;
      // only a failure after a successful semantic commit is terminalized.
      pending_mes_frames_.erase(pending_it);
      return MesDoorbellDisposition::Faulted;
    }
    pending_it->second.phase_ = MesFramePhase::Completion;
    const amdgpu::VmAccessOutcome publication = publish_mes_frame(pending_it->second);
    if (publication == amdgpu::VmAccessOutcome::Unavailable)
      return MesDoorbellDisposition::Retry;
    if (publication != amdgpu::VmAccessOutcome::Complete)
      return MesDoorbellDisposition::Faulted;

    read_pointer = pending_it->second.next_read_pointer_;
    pending_mes_frames_.erase(pending_it);
  }

  callbacks_.publish_queue_pointers(read_pointer, write_pointer);
  return MesDoorbellDisposition::Complete;
}

MesDoorbellDisposition MesEngine::notify_doorbell(uint64_t byte_offset, uint64_t write_pointer,
                                                  const std::optional<MesKernelQueue> &direct_queue,
                                                  const MesDoorbellContext &context) {
  if (!frontend_attached_)
    return MesDoorbellDisposition::Faulted;
  util::Logger::cp([&](auto &os) {
    os << std::format("{}: doorbell {:#x} <- {}", diagnostic_name_, byte_offset, write_pointer);
  });
  const std::vector<Queue>::iterator mapped =
      std::ranges::find(mapped_queues_, byte_offset, &Queue::doorbell_offset);
  if (mapped != mapped_queues_.end()) {
    if (mapped->queue_handle) {
      const amdgpu::QueueSubmissionResult result =
          queue_registry_.submit_producer(mapped->queue_handle, write_pointer);
      if (!result)
        return MesDoorbellDisposition::Faulted;
      switch (result.status) {
      case amdgpu::QueueSubmissionStatus::Accepted:
        return MesDoorbellDisposition::Complete;
      case amdgpu::QueueSubmissionStatus::Retry:
        return MesDoorbellDisposition::Retry;
      case amdgpu::QueueSubmissionStatus::Faulted:
        return MesDoorbellDisposition::Faulted;
      }
    }
    return process_queue(*mapped, write_pointer, context);
  }
  if (direct_queue && direct_queue->doorbell_offset == byte_offset)
    return process_queue(kernel_queue(*direct_queue), write_pointer, context);
  return MesDoorbellDisposition::Ignored;
}

} // namespace rocjitsu::amdgpu
