// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pci/mes_block_model.h"

#include "rocjitsu/vm/amdgpu/mes_engine.h"
#include "rocjitsu/vm/amdgpu/pci/physical_memory_access.h"
#include "simdojo/components/pci_device.h"
#include "util/log.h"

#include <array>
#include <format>
#include <optional>
#include <utility>

namespace rocjitsu {
namespace {

constexpr uint32_t kGcSegment = 0;
constexpr uint32_t kGcControlSegment = 1;
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
constexpr uint32_t kScratchRegister0 = 0x2040;

constexpr uint32_t kDoorbellEnable = 0x40000000;
constexpr uint32_t kDoorbellOffsetMask = 0x0ffffffc;
constexpr uint32_t kQueueSizeMask = 0x3f;
constexpr uint32_t kQueueActiveBit = 0x1;

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

DoorbellDisposition adapt(amdgpu::MesDoorbellDisposition disposition) {
  switch (disposition) {
  case amdgpu::MesDoorbellDisposition::Ignored:
    return DoorbellDisposition::Ignored;
  case amdgpu::MesDoorbellDisposition::Complete:
    return DoorbellDisposition::Complete;
  case amdgpu::MesDoorbellDisposition::Retry:
    return DoorbellDisposition::Retry;
  case amdgpu::MesDoorbellDisposition::Faulted:
    return DoorbellDisposition::Faulted;
  }
  return DoorbellDisposition::Faulted;
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
  for (const uint32_t reg : kWritableRegisters)
    claimed.push_back({.segment_index = kGcSegment, .first_dword = reg, .count = 1});
  claimed.push_back(
      {.segment_index = kGcSegment, .first_dword = kQueueWritePointerLow, .count = 2});
  claimed.push_back({.segment_index = kGcControlSegment, .first_dword = kMesControl, .count = 1});
  claimed.push_back({.segment_index = kGcControlSegment, .first_dword = kGrbmGfxIndex, .count = 1});
  claimed.push_back(
      {.segment_index = kGcControlSegment, .first_dword = kRlcCpSchedulers, .count = 1});
  return claimed;
}

bool MesBlockModel::attach_engine(
    amdgpu::MesEngine &engine,
    std::shared_ptr<amdgpu::SdmaQueueBindingFactory> sdma_queue_binding_factory,
    amdgpu::InterruptSink interrupt_sink) {
  if (engine_ != nullptr)
    return engine_ == &engine;
  amdgpu::MesFrontendCallbacks callbacks(
      [this](uint64_t register_dword, uint32_t value) {
        return write_pm4_uconfig_register(register_dword, value);
      },
      [this](uint64_t read_pointer, uint64_t write_pointer) {
        update_kernel_queue_pointers(read_pointer, write_pointer);
      },
      [](uint32_t offset, amdgpu::QueuePacketFormat) {
        return amdgpu::QueueDoorbellBinding{.mode = amdgpu::QueueDoorbellMode::Explicit,
                                            .offset = offset};
      });
  if (!engine.attach_frontend(registers_.owner(), std::move(sdma_queue_binding_factory),
                              std::move(interrupt_sink), std::move(callbacks))) {
    return false;
  }
  engine_ = &engine;
  return true;
}

bool MesBlockModel::detach_engine() {
  if (engine_ == nullptr)
    return true;
  if (!engine_->detach_frontend())
    return false;
  engine_ = nullptr;
  return true;
}

bool MesBlockModel::teardown_queues() { return engine_ == nullptr || engine_->teardown_queues(); }

bool MesBlockModel::reset() {
  if (engine_ != nullptr && !engine_->reset())
    return false;
  bool defined = true;
  for (const uint32_t reg : kWritableRegisters)
    defined = registers_.define(kGcSegment, reg, 0) && defined;
  defined = registers_.define(kGcSegment, kQueueWritePointerLow, 0) && defined;
  defined = registers_.define(kGcSegment, kQueueWritePointerHigh, 0) && defined;
  defined = registers_.define(kGcControlSegment, kMesControl, 0) && defined;
  defined = registers_.define(kGcControlSegment, kGrbmGfxIndex, 0) && defined;
  defined = registers_.define(kGcControlSegment, kRlcCpSchedulers, 0) && defined;
  return defined;
}

amdgpu::MesKernelQueue MesBlockModel::kernel_queue() const {
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
  };
}

bool MesBlockModel::write_pm4_uconfig_register(uint64_t register_dword, uint32_t value) {
  const std::optional<uint64_t> scratch = registers_.resolve(kGcControlSegment, kScratchRegister0);
  return scratch && register_dword == *scratch / sizeof(uint32_t) &&
         registers_.write(kGcControlSegment, kScratchRegister0, value);
}

void MesBlockModel::update_kernel_queue_pointers(uint64_t read_pointer, uint64_t write_pointer) {
  (void)registers_.write(kGcSegment, kQueueReadPointer, static_cast<uint32_t>(read_pointer));
  (void)registers_.write(kGcSegment, kQueueWritePointerLow, static_cast<uint32_t>(write_pointer));
  (void)registers_.write(kGcSegment, kQueueWritePointerHigh,
                         static_cast<uint32_t>(write_pointer >> 32));
}

DoorbellDisposition MesBlockModel::observe_doorbell_write(uint64_t byte_offset,
                                                          uint64_t write_pointer, std::size_t width,
                                                          PciMemoryAccess &memory) {
  if (width != sizeof(uint32_t) && width != sizeof(uint64_t))
    return DoorbellDisposition::Ignored;
  if (engine_ == nullptr)
    return DoorbellDisposition::Faulted;

  const std::shared_ptr<simdojo::PciTransportSession> session = memory.capture_transport_session();
  amdgpu::MesDoorbellContext context(
      [&memory] { return std::make_shared<PciPhysicalMemoryAccess>(memory); },
      reinterpret_cast<uintptr_t>(session.get()), session != nullptr ? session->generation() : 0);
  return adapt(engine_->notify_doorbell(byte_offset, write_pointer, kernel_queue(), context));
}

} // namespace rocjitsu
