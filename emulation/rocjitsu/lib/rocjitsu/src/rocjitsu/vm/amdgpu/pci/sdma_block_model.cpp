// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pci/sdma_block_model.h"

#include "rocjitsu/vm/amdgpu/gpu_queue_registry.h"
#include "rocjitsu/vm/amdgpu/sdma_queue_binding_factory.h"
#include "rocjitsu/vm/soc.h"

#include "util/log.h"

#include <algorithm>
#include <format>
#include <utility>

namespace rocjitsu {
namespace {

constexpr uint32_t kSdmaSegment = 0;
constexpr uint32_t kSdmaControlSegment = 1;
constexpr uint32_t kStatus = 0x0024;
constexpr uint32_t kQueueControl = 0x0200;
constexpr uint32_t kQueueBaseLow = 0x0201;
constexpr uint32_t kQueueBaseHigh = 0x0202;
constexpr uint32_t kQueueReadPointerLow = 0x0203;
constexpr uint32_t kQueueReadPointerHigh = 0x0204;
constexpr uint32_t kQueueReadPointerAddressLow = 0x0207;
constexpr uint32_t kQueueReadPointerAddressHigh = 0x0208;
constexpr uint32_t kQueueDoorbell = 0x020f;
constexpr uint32_t kQueueDoorbellOffset = 0x0211;
constexpr uint32_t kInstructionCacheOperation = 0x589d;
constexpr uint32_t kSecondEngineRegisterOffset = 0x0600;
constexpr uint32_t kSecondEngineControlOffset = 0x0030;
constexpr uint32_t kIdle = 0x00000001;
constexpr uint32_t kMicrocodeInitDone = 0x08000000;
constexpr uint32_t kInstructionCachePrimed = 0x00000020;
constexpr uint32_t kQueueEnable = 0x00000001;
constexpr uint32_t kQueueSizeMask = 0x0000003e;
constexpr uint32_t kQueueSizeShift = 1;
constexpr uint32_t kDoorbellEnable = 0x10000000;
constexpr uint32_t kDoorbellOffsetMask = 0x0ffffffc;
uint64_t join(uint32_t low, uint32_t high) {
  return static_cast<uint64_t>(low) | (static_cast<uint64_t>(high) << 32);
}

} // namespace

SdmaBlockModel::SdmaBlockModel(IpRegisterWindow registers)
    : IpBlockModel("the SDMA engine", std::move(registers)) {}

std::unique_ptr<SdmaBlockModel> SdmaBlockModel::create(const IpBlock &block,
                                                       IpRegisterWindow registers) {
  if (block.hardware_id == IpHardwareId::Sdma0 && block.major == 7 && block.minor == 1 &&
      block.revision == 0) {
    return std::unique_ptr<SdmaBlockModel>(new SdmaBlockModel(std::move(registers)));
  }
  util::Logger::warn(std::format(
      "{}: no firmware-free startup state is known for SDMA {}.{}.{}, so SDMA cannot start",
      registers.owner(), block.major, block.minor, block.revision));
  return nullptr;
}

std::vector<RegisterClaim> SdmaBlockModel::claims() const {
  std::vector<RegisterClaim> claimed;
  for (const uint32_t offset : {0u, kSecondEngineRegisterOffset}) {
    claimed.push_back({.segment_index = kSdmaSegment, .first_dword = kStatus + offset, .count = 1});
    for (const uint32_t reg :
         {kQueueControl, kQueueBaseLow, kQueueBaseHigh, kQueueReadPointerLow, kQueueReadPointerHigh,
          kQueueReadPointerAddressLow, kQueueReadPointerAddressHigh, kQueueDoorbell,
          kQueueDoorbellOffset}) {
      claimed.push_back({.segment_index = kSdmaSegment, .first_dword = reg + offset, .count = 1});
    }
  }
  for (const uint32_t offset : {0u, kSecondEngineControlOffset}) {
    claimed.push_back({.segment_index = kSdmaControlSegment,
                       .first_dword = kInstructionCacheOperation + offset,
                       .count = 1});
  }
  return claimed;
}

bool SdmaBlockModel::reset() {
  teardown_queues();

  bool defined = true;
  for (const uint32_t offset : {0u, kSecondEngineRegisterOffset}) {
    defined =
        registers_.define_read_only(kSdmaSegment, kStatus + offset, kIdle | kMicrocodeInitDone) &&
        defined;
    for (const uint32_t reg :
         {kQueueControl, kQueueBaseLow, kQueueBaseHigh, kQueueReadPointerLow, kQueueReadPointerHigh,
          kQueueReadPointerAddressLow, kQueueReadPointerAddressHigh, kQueueDoorbell,
          kQueueDoorbellOffset}) {
      defined = registers_.define(kSdmaSegment, reg + offset, 0) && defined;
    }
  }
  for (const uint32_t offset : {0u, kSecondEngineControlOffset}) {
    defined = registers_.define(kSdmaControlSegment, kInstructionCacheOperation + offset,
                                kInstructionCachePrimed) &&
              defined;
  }
  return defined;
}

void SdmaBlockModel::teardown_queues() {
  if (soc_ != nullptr) {
    for (std::optional<Queue> &queue : register_queues_)
      if (queue && queue->queue_handle)
        (void)soc_->queue_registry().unregister_queue(queue->queue_handle,
                                                      amdgpu::QueueCloseMode::ForceCancel);
  }
  for (std::optional<Queue> &queue : register_queues_)
    queue.reset();
}

SdmaBlockModel::Queue SdmaBlockModel::configured_queue(uint32_t engine) const {
  const uint32_t offset = engine == 0 ? 0 : kSecondEngineRegisterOffset;
  const uint32_t control = registers_.read(kSdmaSegment, kQueueControl + offset);
  const uint32_t size = (control & kQueueSizeMask) >> kQueueSizeShift;
  Queue queue;
  queue.ring_base = join(registers_.read(kSdmaSegment, kQueueBaseLow + offset),
                         registers_.read(kSdmaSegment, kQueueBaseHigh + offset))
                    << 8;
  queue.read_pointer_address =
      join(registers_.read(kSdmaSegment, kQueueReadPointerAddressLow + offset),
           registers_.read(kSdmaSegment, kQueueReadPointerAddressHigh + offset));
  queue.ring_bytes = size < 30 ? (uint64_t{1} << size) * sizeof(uint32_t) : 0;
  queue.read_pointer = join(registers_.read(kSdmaSegment, kQueueReadPointerLow + offset),
                            registers_.read(kSdmaSegment, kQueueReadPointerHigh + offset));
  queue.doorbell_offset =
      registers_.read(kSdmaSegment, kQueueDoorbellOffset + offset) & kDoorbellOffsetMask;
  queue.register_offset = offset;
  queue.engine = engine;
  queue.active = (control & kQueueEnable) != 0 &&
                 (registers_.read(kSdmaSegment, kQueueDoorbell + offset) & kDoorbellEnable) != 0;
  queue.address_space =
      soc_ != nullptr ? soc_->gpu_vm().gart_address_space() : amdgpu::AddressSpaceHandle{};
  queue.initial_read_pointer = queue.read_pointer;
  return queue;
}

amdgpu::QueueSubmissionStatus SdmaBlockModel::process_queue(Queue &queue, uint64_t write_pointer) {
  if (!queue.active || queue.faulted || queue.ring_base == 0 || queue.ring_bytes == 0 ||
      !queue.address_space || soc_ == nullptr || queue_binding_factory_ == nullptr)
    return amdgpu::QueueSubmissionStatus::Faulted;

  if (!queue.queue_handle) {
    queue.queue_handle = soc_->queue_registry().register_queue({
        .identity = {.address_space = queue.address_space,
                     .process_id = queue.process_id,
                     .queue_id = 0xffff0000u | queue.engine},
        .ring = {.base_address = queue.ring_base,
                 .size_bytes = static_cast<uint32_t>(queue.ring_bytes),
                 .consumer_pointer_address = queue.read_pointer_address},
        .doorbell = {.offset = static_cast<uint32_t>(queue.doorbell_offset)},
        .binding_factory = queue_binding_factory_,
        .initial_consumer_cursor = queue.initial_read_pointer,
        .engine_id = queue.engine,
        .type = amdgpu::QueueType::Sdma,
        .packet_format = amdgpu::QueuePacketFormat::Sdma,
    });
    if (!queue.queue_handle)
      return amdgpu::QueueSubmissionStatus::Faulted;
  }

  const amdgpu::QueueSubmissionResult result =
      soc_->queue_registry().submit_producer(queue.queue_handle, write_pointer);
  if (!result.found || result.status == amdgpu::QueueSubmissionStatus::Faulted) {
    queue.faulted = true;
    util::Logger::warn(std::format("{}: SDMA{} queue stopped at cursor {:#x}", registers_.owner(),
                                   queue.engine, queue.read_pointer));
  }
  return result.status;
}

void SdmaBlockModel::update_queue_progress(uint32_t engine, uint64_t consumer_cursor,
                                           bool terminal) {
  if (engine >= register_queues_.size() || !register_queues_[engine])
    return;
  Queue &queue = *register_queues_[engine];
  if (!queue.queue_handle)
    return;
  queue.read_pointer = std::max(queue.read_pointer, consumer_cursor);
  const uint32_t offset = queue.register_offset;
  (void)registers_.write(kSdmaSegment, kQueueReadPointerLow + offset,
                         static_cast<uint32_t>(queue.read_pointer));
  (void)registers_.write(kSdmaSegment, kQueueReadPointerHigh + offset,
                         static_cast<uint32_t>(queue.read_pointer >> 32));
  if (terminal && !queue.faulted) {
    queue.faulted = true;
    util::Logger::warn(std::format("{}: SDMA{} queue stopped at cursor {:#x}", registers_.owner(),
                                   queue.engine, queue.read_pointer));
  }
}

DoorbellDisposition SdmaBlockModel::observe_doorbell_write(uint64_t byte_offset,
                                                           uint64_t write_pointer,
                                                           std::size_t width,
                                                           PciMemoryAccess &memory) {
  (void)memory;
  if (width != sizeof(uint32_t) && width != sizeof(uint64_t))
    return DoorbellDisposition::Ignored;
  if (width != sizeof(uint64_t))
    return DoorbellDisposition::Ignored;
  for (uint32_t engine = 0; engine < register_queues_.size(); ++engine) {
    Queue configured = configured_queue(engine);
    if (configured.doorbell_offset != byte_offset)
      continue;

    std::optional<Queue> &runtime = register_queues_[engine];
    bool replace = !runtime;
    if (runtime) {
      replace = runtime->ring_base != configured.ring_base ||
                runtime->read_pointer_address != configured.read_pointer_address ||
                runtime->ring_bytes != configured.ring_bytes ||
                runtime->doorbell_offset != configured.doorbell_offset ||
                runtime->address_space != configured.address_space;
    }
    if (replace && runtime && runtime->queue_handle) {
      runtime->faulted = true;
      return DoorbellDisposition::Faulted;
    }
    if (replace) {
      runtime.emplace(std::move(configured));
    } else {
      runtime->active = configured.active;
    }

    switch (process_queue(*runtime, write_pointer)) {
    case amdgpu::QueueSubmissionStatus::Accepted:
      return DoorbellDisposition::Complete;
    case amdgpu::QueueSubmissionStatus::Retry:
      return DoorbellDisposition::Retry;
    case amdgpu::QueueSubmissionStatus::Faulted:
      return DoorbellDisposition::Faulted;
    }
  }
  return DoorbellDisposition::Ignored;
}

} // namespace rocjitsu
