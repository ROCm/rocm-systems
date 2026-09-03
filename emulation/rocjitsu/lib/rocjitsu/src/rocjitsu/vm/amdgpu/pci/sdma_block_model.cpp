// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pci/sdma_block_model.h"

#include "rocjitsu/vm/amdgpu/queue_service.h"
#include "rocjitsu/vm/soc.h"

#include "util/log.h"

#include <algorithm>
#include <chrono>
#include <format>
#include <stdexcept>
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
constexpr uint8_t kGraphicsInterruptClient = 0x0a;
constexpr uint8_t kSdmaTrapSource = 49;
constexpr uint8_t kFirstXccInterruptNode = 2;

class SdmaUserQueueBackend final : public amdgpu::QueueBackend {
public:
  SdmaUserQueueBackend(SdmaBlockModel &sdma, PciMemoryAccess &memory, uint32_t engine)
      : sdma_(sdma), memory_(memory), engine_(engine) {}

  void attach(const amdgpu::QueueCreateInfo &info) override {
    if (info.kind != amdgpu::QueueKind::Sdma || info.ring_size == 0 ||
        !sdma_.map_user_queue({.ring_base = info.ring_base_va,
                               .read_pointer_address = info.read_ptr_va,
                               .ring_bytes = info.ring_size,
                               .doorbell_offset = info.doorbell_offset,
                               .process_id = info.process_id,
                               .engine = engine_,
                               .address_space = info.address_space,
                               .initial_read_pointer = info.initial_read_pointer})) {
      throw std::runtime_error("cannot attach MES SDMA user queue");
    }
    doorbell_offset_ = info.doorbell_offset;
    attached_ = true;
  }

  void rollback_attach(const amdgpu::QueueCreateInfo &) noexcept override {
    if (std::exchange(attached_, false))
      (void)sdma_.unmap_user_queue(doorbell_offset_);
  }

  void update(uint32_t, uint32_t, uint64_t ring_base_va, uint32_t ring_size,
              uint32_t queue_percentage) override {
    if (attached_)
      (void)sdma_.update_user_queue(doorbell_offset_, ring_base_va, ring_size, queue_percentage);
  }

  amdgpu::QueueDoorbellDisposition notify_doorbell(uint32_t, uint32_t, uint64_t value) override {
    if (attached_)
      return sdma_.notify_user_queue(doorbell_offset_, value, memory_);
    return amdgpu::QueueDoorbellDisposition::Faulted;
  }

  void detach(uint32_t, uint32_t) noexcept override {
    if (std::exchange(attached_, false))
      (void)sdma_.unmap_user_queue(doorbell_offset_);
  }

private:
  SdmaBlockModel &sdma_;
  PciMemoryAccess &memory_;
  uint32_t engine_ = 0;
  uint64_t doorbell_offset_ = 0;
  bool attached_ = false;
};

uint64_t join(uint32_t low, uint32_t high) {
  return static_cast<uint64_t>(low) | (static_cast<uint64_t>(high) << 32);
}

bool same_queue_configuration(const SdmaBlockModel::UserQueue &lhs,
                              const SdmaBlockModel::UserQueue &rhs) {
  return lhs.ring_base == rhs.ring_base && lhs.read_pointer_address == rhs.read_pointer_address &&
         lhs.ring_bytes == rhs.ring_bytes && lhs.doorbell_offset == rhs.doorbell_offset &&
         lhs.process_id == rhs.process_id && lhs.engine == rhs.engine &&
         lhs.address_space == rhs.address_space;
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
  user_queues_.clear();
  for (auto &queue : register_queues_)
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

std::shared_ptr<amdgpu::QueueBackend>
SdmaBlockModel::make_user_queue_backend(PciMemoryAccess &memory, uint32_t engine) {
  return std::make_shared<SdmaUserQueueBackend>(*this, memory, engine);
}

bool SdmaBlockModel::map_user_queue(UserQueue queue) {
  const auto existing =
      std::ranges::find(user_queues_, queue.doorbell_offset, &Queue::doorbell_offset);
  if (existing != user_queues_.end())
    return false;
  Queue mapped;
  mapped.ring_base = queue.ring_base;
  mapped.read_pointer_address = queue.read_pointer_address;
  mapped.ring_bytes = queue.ring_bytes;
  mapped.doorbell_offset = queue.doorbell_offset;
  mapped.process_id = queue.process_id;
  mapped.engine = queue.engine;
  mapped.active = true;
  mapped.register_backed = false;
  mapped.address_space = queue.address_space;
  mapped.initial_read_pointer = queue.initial_read_pointer;
  mapped.read_pointer = queue.initial_read_pointer.value_or(0);
  user_queues_.push_back(std::move(mapped));
  return true;
}

bool SdmaBlockModel::has_in_flight_state(const Queue &queue) {
  return queue.runner && queue.runner->in_flight();
}

bool SdmaBlockModel::update_user_queue(uint64_t doorbell_offset, uint64_t ring_base,
                                       uint64_t ring_bytes, uint32_t queue_percentage) {
  const auto queue = std::ranges::find(user_queues_, doorbell_offset, &Queue::doorbell_offset);
  if (queue == user_queues_.end())
    return false;
  if (queue->ring_base != ring_base || queue->ring_bytes != ring_bytes) {
    if (has_in_flight_state(*queue)) {
      // Rebinding a queue while a packet or its retirement publication is in
      // flight would discard the exactly-once state and permit the same packet
      // to be fetched from the new configuration. Keep the old configuration
      // intact and halt until teardown establishes a clean lifetime boundary.
      queue->faulted = true;
      return false;
    }
    queue->runner.reset();
    queue->initial_read_pointer.reset();
    queue->read_pointer = 0;
    queue->faulted = false;
  }
  queue->ring_base = ring_base;
  queue->ring_bytes = ring_bytes;
  queue->active = queue_percentage != 0;
  return true;
}

amdgpu::QueueDoorbellDisposition SdmaBlockModel::notify_user_queue(uint64_t doorbell_offset,
                                                                   uint64_t write_pointer,
                                                                   PciMemoryAccess &memory) {
  const auto queue = std::ranges::find(user_queues_, doorbell_offset, &Queue::doorbell_offset);
  return queue != user_queues_.end() ? process_queue(*queue, write_pointer, memory)
                                     : amdgpu::QueueDoorbellDisposition::Faulted;
}

bool SdmaBlockModel::unmap_user_queue(uint64_t doorbell_offset) {
  const auto queue = std::ranges::find(user_queues_, doorbell_offset, &Queue::doorbell_offset);
  if (queue == user_queues_.end())
    return false;
  user_queues_.erase(queue);
  return true;
}

amdgpu::SdmaExecutor SdmaBlockModel::make_executor(const Queue &queue, PciMemoryAccess &memory) {
  const uint32_t process_id = queue.process_id;
  const uint32_t engine = queue.engine;
  amdgpu::SdmaExecutorCallbacks callbacks;
  callbacks.poll_register = [&memory](uint32_t address, uint32_t reference, uint32_t mask,
                                      uint32_t function) {
    uint32_t value = 0;
    if (!memory.read_register(address, value))
      return amdgpu::VmAccessOutcome::Faulted;
    const bool satisfied = function == 0 || (function == 3 && (value & mask) == (reference & mask));
    return satisfied ? amdgpu::VmAccessOutcome::Complete : amdgpu::VmAccessOutcome::Unavailable;
  };
  callbacks.write_register = [&memory](uint32_t address, uint32_t value) {
    return memory.write_register(address, value) ? amdgpu::VmAccessOutcome::Complete
                                                 : amdgpu::VmAccessOutcome::Faulted;
  };
  callbacks.deliver_interrupt = [&memory, process_id, engine](uint32_t data) {
    if (process_id > UINT16_MAX)
      return amdgpu::VmAccessOutcome::Malformed;
    const bool delivered =
        memory.deliver_interrupt({.client_id = kGraphicsInterruptClient,
                                  .source_id = kSdmaTrapSource,
                                  .ring_id = static_cast<uint8_t>(engine << 4),
                                  .vmid = process_id == 0 ? uint8_t{0} : uint8_t{1},
                                  .pasid = static_cast<uint16_t>(process_id),
                                  .node_id = kFirstXccInterruptNode,
                                  .data = {data}});
    return delivered ? amdgpu::VmAccessOutcome::Complete : amdgpu::VmAccessOutcome::Faulted;
  };
  callbacks.maintain_caches = [this](amdgpu::SdmaCacheOperation) {
    // SoC currently exposes one conservative whole-hierarchy operation. It is
    // stronger than invalidate-only but preserves dirty data and coherence.
    if (soc_ != nullptr)
      soc_->flush_all();
  };
  callbacks.timestamp = [] {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
  };
  return amdgpu::SdmaExecutor(amdgpu::SdmaPacketDialect::Gfx1250, std::move(callbacks));
}

amdgpu::QueueDoorbellDisposition SdmaBlockModel::process_queue(Queue &queue, uint64_t write_pointer,
                                                               PciMemoryAccess &memory) {
  if (!queue.active || queue.faulted || queue.ring_base == 0 || queue.ring_bytes == 0 ||
      !queue.address_space || soc_ == nullptr)
    return amdgpu::QueueDoorbellDisposition::Faulted;

  if (!queue.runner) {
    queue.runner = std::make_unique<amdgpu::SdmaQueueRunner>(
        soc_->gpu_vm(),
        amdgpu::SdmaQueueRunnerConfig{
            .address_space = queue.address_space,
            .ring_base = queue.ring_base,
            .ring_bytes = queue.ring_bytes,
            .read_pointer_address = queue.read_pointer_address,
            .initial_cursor = queue.initial_read_pointer,
        },
        make_executor(queue, memory));
  }

  const amdgpu::SdmaQueueServiceOutcome outcome = queue.runner->service(write_pointer);
  queue.read_pointer = queue.runner->cursor();
  if (queue.register_backed) {
    const uint32_t offset = queue.register_offset;
    (void)registers_.write(kSdmaSegment, kQueueReadPointerLow + offset,
                           static_cast<uint32_t>(queue.read_pointer));
    (void)registers_.write(kSdmaSegment, kQueueReadPointerHigh + offset,
                           static_cast<uint32_t>(queue.read_pointer >> 32));
  }

  switch (outcome) {
  case amdgpu::SdmaQueueServiceOutcome::Drained:
    return amdgpu::QueueDoorbellDisposition::Complete;
  case amdgpu::SdmaQueueServiceOutcome::Unavailable:
    return amdgpu::QueueDoorbellDisposition::Retry;
  case amdgpu::SdmaQueueServiceOutcome::Faulted:
  case amdgpu::SdmaQueueServiceOutcome::Malformed:
    queue.faulted = true;
    util::Logger::warn(std::format("{}: SDMA{} queue stopped with outcome {} at cursor {:#x}",
                                   registers_.owner(), queue.engine, static_cast<unsigned>(outcome),
                                   queue.read_pointer));
    return amdgpu::QueueDoorbellDisposition::Faulted;
  }
  return amdgpu::QueueDoorbellDisposition::Faulted;
}

DoorbellDisposition SdmaBlockModel::observe_doorbell_write(uint64_t byte_offset,
                                                           uint64_t write_pointer,
                                                           std::size_t width,
                                                           PciMemoryAccess &memory) {
  if (width != sizeof(uint32_t) && width != sizeof(uint64_t))
    return DoorbellDisposition::Ignored;
  const auto mapped = std::ranges::find(user_queues_, byte_offset, &Queue::doorbell_offset);
  if (mapped != user_queues_.end()) {
    switch (notify_user_queue(byte_offset, write_pointer, memory)) {
    case amdgpu::QueueDoorbellDisposition::Complete:
      return DoorbellDisposition::Complete;
    case amdgpu::QueueDoorbellDisposition::Retry:
      return DoorbellDisposition::Retry;
    case amdgpu::QueueDoorbellDisposition::Faulted:
      return DoorbellDisposition::Faulted;
    }
  }
  if (width != sizeof(uint64_t))
    return DoorbellDisposition::Ignored;
  for (uint32_t engine = 0; engine < register_queues_.size(); ++engine) {
    Queue configured = configured_queue(engine);
    if (configured.doorbell_offset != byte_offset)
      continue;

    auto &runtime = register_queues_[engine];
    UserQueue configured_identity{.ring_base = configured.ring_base,
                                  .read_pointer_address = configured.read_pointer_address,
                                  .ring_bytes = configured.ring_bytes,
                                  .doorbell_offset = configured.doorbell_offset,
                                  .process_id = configured.process_id,
                                  .engine = configured.engine,
                                  .address_space = configured.address_space,
                                  .initial_read_pointer = std::nullopt};
    bool replace = !runtime;
    if (runtime) {
      const UserQueue runtime_identity{.ring_base = runtime->ring_base,
                                       .read_pointer_address = runtime->read_pointer_address,
                                       .ring_bytes = runtime->ring_bytes,
                                       .doorbell_offset = runtime->doorbell_offset,
                                       .process_id = runtime->process_id,
                                       .engine = runtime->engine,
                                       .address_space = runtime->address_space,
                                       .initial_read_pointer = std::nullopt};
      replace = !same_queue_configuration(runtime_identity, configured_identity);
    }
    if (replace && runtime && has_in_flight_state(*runtime)) {
      // MMIO reprogramming is the register-backed form of a user-queue rebind.
      // Do not throw away a partially fetched/executed packet or an unpublished
      // retirement and then replay it through the new configuration.
      runtime->faulted = true;
      return DoorbellDisposition::Complete;
    }
    if (replace)
      runtime.emplace(std::move(configured));
    else
      runtime->active = configured.active;

    switch (process_queue(*runtime, write_pointer, memory)) {
    case amdgpu::QueueDoorbellDisposition::Complete:
      return DoorbellDisposition::Complete;
    case amdgpu::QueueDoorbellDisposition::Retry:
      return DoorbellDisposition::Retry;
    case amdgpu::QueueDoorbellDisposition::Faulted:
      return DoorbellDisposition::Faulted;
    }
  }
  return DoorbellDisposition::Ignored;
}

} // namespace rocjitsu
