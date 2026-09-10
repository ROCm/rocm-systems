// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/legacy_gpu_vm.h"
#include "rocjitsu/vm/amdgpu/aql_queue_binding_factory.h"
#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/amdgpu/pci/bar_access_trace.h"
#include "rocjitsu/vm/amdgpu/pci/gpu_pci_device.h"
#include "rocjitsu/vm/amdgpu/pci/gpu_pci_device_spec.h"
#include "rocjitsu/vm/amdgpu/pci/mmio_registers.h"
#include "rocjitsu/vm/amdgpu/pci/physical_memory_access.h"
#include "rocjitsu/vm/amdgpu/pci/register_symbols.h"
#include "rocjitsu/vm/amdgpu/xcd.h"
#include "rocjitsu/vm/soc.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <future>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <thread>

namespace {

constexpr uint64_t kVramBytes = 16ULL * 1024 * 1024;

/// @brief The only part with an IP discovery profile.
/// @details Every device built here is meant to be that part: a configuration
/// naming anything else deliberately has no profile and cannot become usable,
/// which two tests below cover on purpose.
constexpr uint32_t kModelledTarget = 120500;

/// @brief A configuration equivalent to what a config file would supply.
rocjitsu::GpuPciDeviceSpec configured_spec() {
  rocjitsu::config::KfdDeviceConfig device;
  device.gfx_target_version = kModelledTarget;
  device.vendor_id = 0x1002;
  device.device_id = 0x1250;
  device.pci_revision_id = 0x5a;
  device.local_mem_size = kVramBytes;
  device.num_shader_engines = 2;
  device.num_shader_arrays_per_engine = 2;
  device.num_cu_per_sh = 8;
  device.wave_front_size = 32;
  device.max_waves_per_simd = 16;
  device.max_slots_scratch_cu = 32;
  device.lds_size_kb = 320;
  return rocjitsu::gpu_pci_spec_from_config(device, {});
}

class GpuDevice : public ::testing::Test {
protected:
  GpuDevice() {
    xcd_.set_command_processor(&command_processor_);
    soc_.add_xcd(&xcd_);
    command_processor_.set_gpu_vm(&soc_.gpu_vm());
  }

  rocjitsu::RegisterSymbols symbols_;
  rocjitsu::BarAccessTrace trace_{symbols_};
  rocjitsu::amdgpu::GpuMemory memory_{"memory"};
  rocjitsu::amdgpu::CommandProcessor command_processor_{"cp"};
  rocjitsu::amdgpu::Xcd xcd_{"xcd"};
  rocjitsu::SoC soc_{"soc", &memory_, ROCJITSU_CODE_ARCH_CDNA5};
  rocjitsu::GpuPciDevice device_{"gpu", configured_spec(), &trace_, &soc_};

  [[nodiscard]] uint32_t read_register(rocjitsu::MmioRegister reg) {
    std::array<std::byte, 4> raw{};
    EXPECT_EQ(device_.bar_access(rocjitsu::GpuPciDevice::kRegisterBar, raw,
                                 rocjitsu::byte_offset_of(reg), /*write=*/false),
              4);
    return std::bit_cast<uint32_t>(raw);
  }

  /// @brief Value read at @p byte_offset, or kAccessFailed if the read failed.
  /// @details A refused access must not also hand back a plausible value: the
  /// caller would assert on it and report a wrong register content, sending the
  /// reader after the value instead of after the refusal.
  static constexpr uint32_t kAccessFailed = 0xDEADDEADu;
  [[nodiscard]] uint32_t read_register_at(uint64_t byte_offset) {
    std::array<std::byte, 4> raw{};
    if (device_.bar_access(rocjitsu::GpuPciDevice::kRegisterBar, raw, byte_offset,
                           /*write=*/false) != 4) {
      ADD_FAILURE() << "the register at byte " << byte_offset << " could not be read";
      return kAccessFailed;
    }
    return std::bit_cast<uint32_t>(raw);
  }

  void write_register_at(uint64_t byte_offset, uint32_t value) {
    auto raw = std::bit_cast<std::array<std::byte, 4>>(value);
    EXPECT_EQ(device_.bar_access(rocjitsu::GpuPciDevice::kRegisterBar, raw, byte_offset,
                                 /*write=*/true),
              4);
  }

  [[nodiscard]] const simdojo::BarSpec *bar(int index) {
    static std::vector<simdojo::BarSpec> bars;
    bars = device_.bars();
    const auto found = std::ranges::find(bars, index, &simdojo::BarSpec::index);
    return found == bars.end() ? nullptr : &*found;
  }
};

// A flush is issued and then waited for, and nothing else reports it finishing.
// An unanswered acknowledge is therefore not a quiet gap in the register model
// but a stall of the driver's full timeout, once per flush — it cost about
// nine seconds each and eighty-eight seconds of a boot.
//
// The addresses here are written out rather than derived the way the device
// derives them, because a test that recomputed them from the same segment list
// would agree with the device about an address the driver does not use. Each is
// `(segment + register + engine * stride) * 4`, with the register from the
// offset headers and engine 17, which is the one the GART flush picks:
//   GFXHUB  (0x1260 + 0x1669 + 17) * 4 == 0xa368
//   MMHUB   (0x1a000 + 0x0599 + 17) * 4 == 0x696a8
// The second of those is the register a guest boot was seen spinning on ten
// million times, and only that one: a GFXHUB flush returns before touching any
// register while the graphics block is unpowered, which holds for as long as
// this device does not bring that block up, so every stalled flush was MMHUB's.
TEST_F(GpuDevice, PublishesVmidZeroBeforeAcknowledgingItsInvalidation) {
  ASSERT_TRUE(device_.usable());

  EXPECT_EQ(read_register_at(0xa368), 0u);
  EXPECT_EQ(read_register_at(0x696a8), 0u);

  // Engine 0 and the last engine, since the driver reaches engines by stride
  // and modelling only the one it happens to use today would break silently.
  EXPECT_EQ(read_register_at((0x1a000 + 0x0599) * 4), 0u);
  EXPECT_EQ(read_register_at((0x1a000 + 0x0599 + 17) * 4), 0u);

  // The request is written rather than read, so what is checked is that the
  // device models it at all: an unmodelled register drops the write and reads
  // back zero. Reading a request register back is not something the driver
  // does, but this asserts the model, not hardware fidelity.
  constexpr uint64_t kMmHubRequest17 = (0x1a000 + 0x0587 + 17) * 4;
  write_register_at(kMmHubRequest17, 0x1);
  EXPECT_EQ(read_register_at(kMmHubRequest17), 0x1u)
      << "MMHUB engine 17's request was dropped as unmodelled";
  EXPECT_EQ(read_register_at(0x696a8) & 1u, 1u);

  constexpr uint64_t kGcBase = 0x1260;
  const auto absolute = [](uint64_t reg) { return (kGcBase + reg) * 4; };
  constexpr uint64_t kPageTable = 0x10000;
  constexpr uint64_t kGartStart = 0x100000000ULL;
  write_register_at(absolute(0x169f), static_cast<uint32_t>(kPageTable));
  write_register_at(absolute(0x16a0), static_cast<uint32_t>(kPageTable >> 32));
  write_register_at(absolute(0x16bf), static_cast<uint32_t>(kGartStart >> 12));
  write_register_at(absolute(0x16c0), static_cast<uint32_t>(kGartStart >> 44));
  write_register_at(absolute(0x16df), static_cast<uint32_t>((kGartStart + 0xfff) >> 12));
  write_register_at(absolute(0x16e0), static_cast<uint32_t>((kGartStart + 0xfff) >> 44));

  const rocjitsu::amdgpu::AddressSpaceHandle gart = soc_.gpu_vm().gart_address_space();
  ASSERT_TRUE(gart);
  const uint64_t old_epoch = soc_.gpu_vm().lookup(gart)->translation_epoch;
  write_register_at(absolute(0x1657 + 17), 1);

  EXPECT_EQ(read_register_at(0xa368) & 1u, 1u);
  ASSERT_TRUE(soc_.gpu_vm().lookup(gart));
  EXPECT_EQ(soc_.gpu_vm().lookup(gart)->translation_epoch, old_epoch + 1);
  const auto translated =
      soc_.gpu_vm().translate(gart, kGartStart, 1, rocjitsu::amdgpu::VmAccessKind::Read);
  EXPECT_EQ(translated.outcome, rocjitsu::amdgpu::VmAccessOutcome::Unavailable)
      << "the published GART did not preserve the absent transport as a typed outcome";
}

TEST_F(GpuDevice, ReportsAReadOnlyAddressConfigurationMatchingItsTopology) {
  ASSERT_TRUE(device_.usable());
  constexpr uint64_t kGbAddrConfigRead = (0x1260 + 0x13e2) * 4;
  constexpr uint32_t kTwoEnginesAndTwoBackendsPerEngine = 0x04080000;

  ASSERT_EQ(read_register_at(kGbAddrConfigRead), kTwoEnginesAndTwoBackendsPerEngine);
  write_register_at(kGbAddrConfigRead, 0xffffffff);
  EXPECT_EQ(read_register_at(kGbAddrConfigRead), kTwoEnginesAndTwoBackendsPerEngine);
}

TEST_F(GpuDevice, CompletesFirmwareFreeComputeCacheInvalidation) {
  ASSERT_TRUE(device_.usable());
  constexpr uint64_t kComputeDataCacheOperation = (0xA000 + 0x290c) * 4;
  constexpr uint64_t kComputeInstructionCacheOperation = (0xA000 + 0x297a) * 4;

  for (const uint64_t operation : {kComputeDataCacheOperation, kComputeInstructionCacheOperation}) {
    ASSERT_EQ(read_register_at(operation), 0x2u);
    write_register_at(operation, 0x3u);
    EXPECT_EQ(read_register_at(operation), 0x2u)
        << "the trigger write cleared the already-complete status";
  }
}

TEST_F(GpuDevice, ReportsFirmwareFreeSdmaMicrocodeStartupComplete) {
  ASSERT_TRUE(device_.usable());
  constexpr uint64_t kSdmaStatuses[] = {
      (0x1260 + 0x0024) * 4,
      (0x1260 + 0x0024 + 0x0600) * 4,
  };
  constexpr uint64_t kInstructionCacheOperations[] = {
      (0xA000 + 0x589d) * 4,
      (0xA000 + 0x589d + 0x0030) * 4,
  };

  for (std::size_t engine = 0; engine < std::size(kSdmaStatuses); ++engine) {
    const uint64_t status = kSdmaStatuses[engine];
    const uint64_t operation = kInstructionCacheOperations[engine];
    EXPECT_EQ(read_register_at(status) & 0x08000001u, 0x08000001u);
    ASSERT_EQ(read_register_at(operation) & 0x20u, 0x20u);

    // The driver's read-modify-write sets PRIME_ICACHE. Completion remains set
    // in the same register so its following poll observes a primed cache.
    write_register_at(operation, read_register_at(operation) | 0x10u);
    EXPECT_EQ(read_register_at(operation) & 0x30u, 0x30u);

    write_register_at(status, 0);
    EXPECT_EQ(read_register_at(status) & 0x08000001u, 0x08000001u);
  }
}

// Covers the older flush path, which brackets a flush with an acquire of the
// engine's semaphore and *releases it by writing zero*. A register that stored
// that write would grant the first acquire and stall every flush after it, so
// this one has to ignore writes -- which is the whole reason read-only
// registers exist here. The version this device publishes today does not take
// the semaphore, so this is covering the profile rather than the boot.
//   MMHUB semaphore, engine 17: (0x1a000 + 0x0575 + 17) * 4 == 0x69618
TEST_F(GpuDevice, KeepsGrantingTheInvalidationSemaphoreAfterItIsReleased) {
  ASSERT_TRUE(device_.usable());
  constexpr uint64_t kMmHubSemaphore17 = 0x69618;
  ASSERT_EQ(read_register_at(kMmHubSemaphore17) & 0x1u, 0x1u) << "the acquire is never granted";

  write_register_at(kMmHubSemaphore17, 0);

  EXPECT_EQ(read_register_at(kMmHubSemaphore17) & 0x1u, 0x1u)
      << "releasing the semaphore latched it low, so the next flush would stall";
}

// The driver says where it put the interrupt ring only by writing these
// registers, and it says it in pieces: the address arrives shifted down by
// eight across two registers, the size as the logarithm of a dword count, and
// the write-pointer address split across two more. Reading that back wrongly
// would point the device at the wrong guest memory, which is indistinguishable
// from the driver never being interrupted.
//   OSSSYS segment 0x10a0, all _BASE_IDX 0, so byte offset (0x10a0 + reg) * 4:
//     IH_RB_CNTL 0x0080 -> 0x4480      IH_RB_BASE 0x0083 -> 0x448c
//     IH_RB_BASE_HI 0x0084 -> 0x4490   WPTR_ADDR_HI 0x0085 -> 0x4494
//     WPTR_ADDR_LO 0x0086 -> 0x4498
TEST_F(GpuDevice, ReadsBackTheInterruptRingTheDriverProgrammed) {
  ASSERT_TRUE(device_.usable());
  EXPECT_EQ(device_.interrupt_ring().base, 0u) << "nothing has been programmed yet";
  EXPECT_FALSE(device_.interrupt_ring().enabled);

  // A ring at 0x1234_5678_9A00 of 256 KiB, its write pointer at 0xABCD_1000,
  // switched on and asking for an interrupt per entry. 256 KiB is 65536 dwords,
  // so the size field is 16.
  constexpr uint64_t kRingBase = 0x123456789a00;
  // Above four gigabytes on purpose: a guest with that much memory routinely
  // puts the write pointer there, and a 32-bit value would leave the high half
  // of the decode unproven -- deleting it entirely would still pass.
  constexpr uint64_t kWptrAddress = 0x5abcd1000;
  write_register_at(0x448c, static_cast<uint32_t>(kRingBase >> 8));
  write_register_at(0x4490, static_cast<uint32_t>(kRingBase >> 40) & 0xff);
  write_register_at(0x4498, static_cast<uint32_t>(kWptrAddress));
  write_register_at(0x4494, static_cast<uint32_t>(kWptrAddress >> 32) & 0xffff);
  // Size 16, enabled, interrupt per entry, and the address-space field saying
  // these are bus addresses (2 at bit 28).
  write_register_at(0x4480, (16u << 1) | (1u << 0) | (1u << 17) | (2u << 28));

  const rocjitsu::InterruptRing ring = device_.interrupt_ring();
  EXPECT_EQ(ring.base, kRingBase);
  EXPECT_EQ(ring.wptr_address, kWptrAddress);
  EXPECT_EQ(ring.bytes, 256u * 1024) << "the size field is a logarithm of a dword count";
  EXPECT_TRUE(ring.enabled);
  EXPECT_TRUE(ring.raises_messages);
  EXPECT_EQ(ring.space, rocjitsu::InterruptRingSpace::BusAddress);

  // The same registers with the space the driver uses when it loads firmware
  // through the security processor: the addresses are then translated, and
  // acting on them as if they were bus addresses would write somewhere real
  // and wrong.
  write_register_at(0x4480, (16u << 1) | (1u << 0) | (1u << 17) | (4u << 28));
  EXPECT_EQ(device_.interrupt_ring().space, rocjitsu::InterruptRingSpace::GpuVirtual);
}

// programmed() is what decides whether the shutdown diagnostic says anything,
// so the states it must recognise are the partial ones: a driver that sized a
// ring, or named a write-pointer address, and then stopped has said a great
// deal, and reporting that as nothing programmed hides exactly the state worth
// seeing. The test above jumps from reset defaults straight to a fully enabled
// ring and would not notice any of these being dropped.
TEST_F(GpuDevice, ReportsAPartiallyProgrammedInterruptRingAsProgrammed) {
  ASSERT_TRUE(device_.usable());
  ASSERT_FALSE(device_.interrupt_ring().programmed()) << "a reset ring has said nothing";

  // Each of these on its own, from reset, with the base and the enable bit
  // still clear -- the two fields a narrower predicate would have keyed on.
  struct Case {
    const char *what;
    uint64_t reg;
    uint32_t value;
  };
  for (const Case &partial :
       {Case{"a size alone", 0x4480, 16u << 1}, Case{"an address space alone", 0x4480, 2u << 28},
        Case{"a request for messages alone", 0x4480, 1u << 17},
        Case{"a write-pointer address alone", 0x4498, 0xabcd1000u}}) {
    device_.reset(simdojo::ResetKind::FunctionLevel);
    ASSERT_FALSE(device_.interrupt_ring().programmed()) << "reset did not clear the ring";

    write_register_at(partial.reg, partial.value);
    const rocjitsu::InterruptRing ring = device_.interrupt_ring();
    EXPECT_TRUE(ring.programmed()) << partial.what << " was reported as nothing programmed";
    EXPECT_EQ(ring.base, 0u) << partial.what;
    EXPECT_FALSE(ring.enabled) << partial.what;
  }

  // And back to nothing, so the diagnostic does not keep reporting a ring the
  // guest has already taken away.
  device_.reset(simdojo::ResetKind::FunctionLevel);
  EXPECT_FALSE(device_.interrupt_ring().programmed());
}

// A segment that wraps when turned into a byte address would otherwise pass an
// aperture bounds test and land on an unrelated register -- the ring's control
// dword resolving on top of something the driver depends on.
TEST(GpuDeviceFlushes, RefusesASegmentThatWrapsIntoTheAperture) {
  rocjitsu::config::KfdDeviceConfig device;
  device.gfx_target_version = kModelledTarget;
  device.local_mem_size = 8ULL * 1024 * 1024 * 1024;

  rocjitsu::GpuPciDeviceSpec spec = rocjitsu::gpu_pci_spec_from_config(device, {});
  for (rocjitsu::IpBlock &block : spec.discovery.blocks) {
    if (block.hardware_id == rocjitsu::IpHardwareId::OssSys) {
      block.register_bases.front() = std::numeric_limits<uint64_t>::max() - 0x7f;
    }
  }

  rocjitsu::RegisterSymbols symbols;
  rocjitsu::BarAccessTrace trace(symbols);
  rocjitsu::GpuPciDevice wrapped("wrapped", spec, &trace);
  ASSERT_TRUE(wrapped.usable()) << "the hubs are untouched, so the device still comes up";

  // This segment puts the ring's control dword at byte zero: the control
  // register sits 0x80 dwords into the block, and 0x80 past this base is
  // exactly 2^64. Unchecked, the multiply lands it inside the aperture and the
  // device both defines it there and reads the ring back out of it.
  std::array<std::byte, 4> raw{};
  const auto enabled = std::bit_cast<std::array<std::byte, 4>>(uint32_t{1});
  ASSERT_EQ(wrapped.bar_access(rocjitsu::GpuPciDevice::kRegisterBar, raw, 0, /*write=*/false), 4);
  raw = enabled;
  (void)wrapped.bar_access(rocjitsu::GpuPciDevice::kRegisterBar, raw, 0, /*write=*/true);

  EXPECT_FALSE(wrapped.interrupt_ring().programmed())
      << "a wrapped segment resolved the ring's control register onto byte zero";
}

/// @brief Guest memory and an interrupt line, recorded rather than delivered.
class RecordingTransport : public simdojo::DmaEngine, public simdojo::IrqSink {
public:
  [[nodiscard]] bool read(uint64_t guest_phys, std::span<std::byte> dst) override {
    {
      std::unique_lock lock(read_mutex_);
      if (blocked_read_address_.has_value() && *blocked_read_address_ == guest_phys) {
        read_is_blocked_ = true;
        read_state_changed_.notify_all();
        read_state_changed_.wait(lock, [this]() { return release_blocked_read_; });
        blocked_read_address_.reset();
        read_is_blocked_ = false;
        release_blocked_read_ = false;
      }
    }
    for (std::size_t i = 0; i < dst.size(); ++i) {
      const auto found = memory.find(guest_phys + i);
      dst[i] = found == memory.end() ? std::byte{0} : found->second;
    }
    return true;
  }

  simdojo::DmaAccessOutcome read_outcome(uint64_t guest_phys, std::span<std::byte> dst) override {
    if (next_read_outcome && next_read_outcome->first == guest_phys) {
      const simdojo::DmaAccessOutcome outcome = next_read_outcome->second;
      next_read_outcome.reset();
      return outcome;
    }
    return read(guest_phys, dst) ? simdojo::DmaAccessOutcome::Complete
                                 : simdojo::DmaAccessOutcome::Faulted;
  }

  [[nodiscard]] bool write(uint64_t guest_phys, std::span<const std::byte> src) override {
    if (refuse_writes || writes_before_refusing == 0) {
      return false;
    }
    if (writes_before_refusing > 0) {
      --writes_before_refusing;
    }
    writes.emplace_back(guest_phys, src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
      memory[guest_phys + i] = src[i];
    }
    return true;
  }

  simdojo::DmaAccessOutcome write_outcome(uint64_t guest_phys,
                                          std::span<const std::byte> src) override {
    if (next_write_outcome && next_write_outcome->first == guest_phys) {
      const simdojo::DmaAccessOutcome outcome = next_write_outcome->second;
      next_write_outcome.reset();
      return outcome;
    }
    return write(guest_phys, src) ? simdojo::DmaAccessOutcome::Complete
                                  : simdojo::DmaAccessOutcome::Faulted;
  }

  simdojo::DmaAtomicLoadResult atomic_load(uint64_t guest_phys, uint32_t width) override {
    if ((width != sizeof(uint32_t) && width != sizeof(uint64_t)) || guest_phys % width != 0)
      return {.outcome = simdojo::DmaAccessOutcome::Malformed};
    std::array<std::byte, sizeof(uint64_t)> bytes{};
    const simdojo::DmaAccessOutcome outcome =
        read_outcome(guest_phys, std::span(bytes).first(width));
    uint64_t value = 0;
    if (outcome == simdojo::DmaAccessOutcome::Complete)
      std::memcpy(&value, bytes.data(), width);
    return {.outcome = outcome, .value = value};
  }

  simdojo::DmaAccessOutcome atomic_store(uint64_t guest_phys, uint32_t width,
                                         uint64_t value) override {
    if ((width != sizeof(uint32_t) && width != sizeof(uint64_t)) || guest_phys % width != 0)
      return simdojo::DmaAccessOutcome::Malformed;
    const auto bytes = std::bit_cast<std::array<std::byte, sizeof(uint64_t)>>(value);
    return write_outcome(guest_phys, std::span(bytes).first(width));
  }

  [[nodiscard]] bool trigger(uint32_t vector) override {
    triggered.push_back(vector);
    return !refuse_trigger;
  }

  [[nodiscard]] uint32_t dword_at(uint64_t guest_phys) {
    std::array<std::byte, 4> raw{};
    (void)read(guest_phys, raw);
    return std::bit_cast<uint32_t>(raw);
  }

  void block_next_read_at(uint64_t guest_phys) {
    const std::lock_guard lock(read_mutex_);
    blocked_read_address_ = guest_phys;
    read_is_blocked_ = false;
    release_blocked_read_ = false;
  }

  void wait_for_blocked_read() {
    std::unique_lock lock(read_mutex_);
    read_state_changed_.wait(lock, [this]() { return read_is_blocked_; });
  }

  void release_read() {
    const std::lock_guard lock(read_mutex_);
    release_blocked_read_ = true;
    read_state_changed_.notify_all();
  }

  std::map<uint64_t, std::byte> memory;
  /// @brief Every write, as address and length, so a transfer that overran can
  /// be told apart from a legitimate one to the address it overran into.
  std::vector<std::pair<uint64_t, std::size_t>> writes;
  std::vector<uint32_t> triggered;
  bool refuse_writes = false;
  bool refuse_trigger = false;
  /// @brief Writes to accept before refusing, or negative for no limit.
  int writes_before_refusing = -1;
  std::optional<std::pair<uint64_t, simdojo::DmaAccessOutcome>> next_read_outcome;
  std::optional<std::pair<uint64_t, simdojo::DmaAccessOutcome>> next_write_outcome;

private:
  std::mutex read_mutex_;
  std::condition_variable read_state_changed_;
  std::optional<uint64_t> blocked_read_address_;
  bool read_is_blocked_ = false;
  bool release_blocked_read_ = false;
};

class SessionTestDevice final : public simdojo::PciDevice {
public:
  SessionTestDevice() : PciDevice("session-test", {}) {}

  [[nodiscard]] std::vector<simdojo::BarSpec> bars() const override { return {}; }
  [[nodiscard]] int64_t bar_access(int, std::span<std::byte>, uint64_t, bool) override {
    return -1;
  }
  void dma_map(const simdojo::DmaRegion &) override {}
  void dma_unmap(const simdojo::DmaRegion &) override {}

  [[nodiscard]] std::shared_ptr<simdojo::PciTransportSession> capture_session() const {
    return transport_session();
  }
};

class OutcomeTransport final : public simdojo::DmaEngine, public simdojo::IrqSink {
public:
  bool read(uint64_t guest_phys, std::span<std::byte> dst) override {
    return read_outcome(guest_phys, dst) == simdojo::DmaAccessOutcome::Complete;
  }
  bool write(uint64_t guest_phys, std::span<const std::byte> src) override {
    return write_outcome(guest_phys, src) == simdojo::DmaAccessOutcome::Complete;
  }
  simdojo::DmaAccessOutcome read_outcome(uint64_t, std::span<std::byte> dst) override {
    ++reads;
    std::ranges::fill(dst, std::byte{0x5a});
    return outcome;
  }
  simdojo::DmaAccessOutcome write_outcome(uint64_t, std::span<const std::byte>) override {
    ++writes;
    return outcome;
  }
  bool trigger(uint32_t) override { return true; }

  simdojo::DmaAccessOutcome outcome = simdojo::DmaAccessOutcome::Complete;
  uint32_t reads = 0;
  uint32_t writes = 0;
};

class SessionBackedMemory final : public rocjitsu::PciMemoryAccess {
public:
  explicit SessionBackedMemory(std::shared_ptr<simdojo::PciTransportSession> session)
      : session_(std::move(session)) {}

  [[nodiscard]] std::shared_ptr<simdojo::PciTransportSession>
  capture_transport_session() const override {
    return session_;
  }
  bool read_vram(uint64_t offset, std::span<std::byte> bytes) override {
    if (offset > vram.size() || bytes.size() > vram.size() - offset)
      return false;
    std::ranges::copy(std::span(vram).subspan(offset, bytes.size()), bytes.begin());
    return true;
  }
  bool write_vram(uint64_t offset, std::span<const std::byte> bytes) override {
    if (offset > vram.size() || bytes.size() > vram.size() - offset)
      return false;
    std::ranges::copy(bytes, vram.begin() + static_cast<std::ptrdiff_t>(offset));
    return true;
  }
  rocjitsu::amdgpu::AtomicLoadResult atomic_load_vram(uint64_t offset, uint32_t width) override {
    if ((width != sizeof(uint32_t) && width != sizeof(uint64_t)) || offset % width != 0 ||
        offset > vram.size() || width > vram.size() - offset) {
      return {.outcome = rocjitsu::amdgpu::VmAccessOutcome::Faulted};
    }
    uint64_t value = 0;
    std::memcpy(&value, vram.data() + offset, width);
    return {.outcome = rocjitsu::amdgpu::VmAccessOutcome::Complete, .value = value};
  }
  rocjitsu::amdgpu::VmAccessOutcome atomic_store_vram(uint64_t offset, uint32_t width,
                                                      uint64_t value) override {
    if ((width != sizeof(uint32_t) && width != sizeof(uint64_t)) || offset % width != 0 ||
        offset > vram.size() || width > vram.size() - offset) {
      return rocjitsu::amdgpu::VmAccessOutcome::Faulted;
    }
    std::memcpy(vram.data() + offset, &value, width);
    return rocjitsu::amdgpu::VmAccessOutcome::Complete;
  }
  rocjitsu::amdgpu::AtomicCompareExchangeResult compare_exchange_vram(uint64_t offset,
                                                                      uint32_t width,
                                                                      uint64_t expected,
                                                                      uint64_t desired) override {
    if ((width != sizeof(uint32_t) && width != sizeof(uint64_t)) || offset % width != 0 ||
        offset > vram.size() || width > vram.size() - offset) {
      return {.outcome = rocjitsu::amdgpu::VmAccessOutcome::Faulted};
    }
    uint64_t observed = 0;
    std::memcpy(&observed, vram.data() + offset, width);
    const uint64_t mask = width == sizeof(uint64_t) ? UINT64_MAX : UINT32_MAX;
    if ((observed & mask) == (expected & mask))
      std::memcpy(vram.data() + offset, &desired, width);
    return {.outcome = rocjitsu::amdgpu::VmAccessOutcome::Complete,
            .observed = observed,
            .exchanged = (observed & mask) == (expected & mask)};
  }
  bool read_register(uint64_t, uint32_t &) override { return false; }
  bool write_register(uint64_t, uint32_t) override { return false; }
  bool deliver_interrupt(const rocjitsu::InterruptEntry &) override { return false; }

  std::array<std::byte, 16> vram{};

private:
  std::shared_ptr<simdojo::PciTransportSession> session_;
};

// Models the ownership shape of GpuPciDevice: the object that supplies local
// memory also owns the transport session captured by retained VM bindings.
class LifetimeBoundMemory final : public simdojo::PciDevice, public rocjitsu::PciMemoryAccess {
public:
  LifetimeBoundMemory() : PciDevice("lifetime-bound-memory", {}) {}
  ~LifetimeBoundMemory() override { shutdown_transport(); }

  [[nodiscard]] std::vector<simdojo::BarSpec> bars() const override { return {}; }
  [[nodiscard]] int64_t bar_access(int, std::span<std::byte>, uint64_t, bool) override {
    return -1;
  }
  void dma_map(const simdojo::DmaRegion &) override {}
  void dma_unmap(const simdojo::DmaRegion &) override {}

  [[nodiscard]] std::shared_ptr<simdojo::PciTransportSession>
  capture_transport_session() const override {
    return transport_session();
  }
  bool read_vram(uint64_t offset, std::span<std::byte> bytes) override {
    if (offset > vram_.size() || bytes.size() > vram_.size() - offset)
      return false;
    std::ranges::copy(std::span(vram_).subspan(offset, bytes.size()), bytes.begin());
    return true;
  }
  bool write_vram(uint64_t offset, std::span<const std::byte> bytes) override {
    if (offset > vram_.size() || bytes.size() > vram_.size() - offset)
      return false;
    std::ranges::copy(bytes, vram_.begin() + static_cast<std::ptrdiff_t>(offset));
    return true;
  }
  rocjitsu::amdgpu::AtomicLoadResult atomic_load_vram(uint64_t, uint32_t) override {
    return {.outcome = rocjitsu::amdgpu::VmAccessOutcome::Faulted};
  }
  rocjitsu::amdgpu::VmAccessOutcome atomic_store_vram(uint64_t, uint32_t, uint64_t) override {
    return rocjitsu::amdgpu::VmAccessOutcome::Faulted;
  }
  rocjitsu::amdgpu::AtomicCompareExchangeResult compare_exchange_vram(uint64_t, uint32_t, uint64_t,
                                                                      uint64_t) override {
    return {.outcome = rocjitsu::amdgpu::VmAccessOutcome::Faulted};
  }
  bool read_register(uint64_t, uint32_t &) override { return false; }
  bool write_register(uint64_t, uint32_t) override { return false; }
  bool deliver_interrupt(const rocjitsu::InterruptEntry &) override { return false; }

private:
  std::array<std::byte, 16> vram_{};
};

TEST(PciBackingSession, StaleGenerationCannotReachReattachedTransport) {
  SessionTestDevice device;
  OutcomeTransport first;
  OutcomeTransport second;
  simdojo::PciDevice::Transport first_endpoints{.irq = &first, .dma = &first};
  simdojo::PciDevice::Transport second_endpoints{.irq = &second, .dma = &second};

  ASSERT_TRUE(device.attach_transport(&first_endpoints));
  std::shared_ptr<simdojo::PciTransportSession> first_session = device.capture_session();
  ASSERT_NE(first_session, nullptr);
  const uint64_t first_generation = first_session->generation();

  std::shared_ptr<simdojo::PciTransportSession> closing =
      device.revoke_transport_session(&first_endpoints);
  ASSERT_EQ(closing, first_session);
  closing->wait_until_drained();
  ASSERT_TRUE(device.detach_transport(&first_endpoints));

  ASSERT_TRUE(device.attach_transport(&second_endpoints));
  std::shared_ptr<simdojo::PciTransportSession> second_session = device.capture_session();
  ASSERT_NE(second_session, nullptr);
  EXPECT_NE(second_session->generation(), first_generation);

  std::array<std::byte, 4> bytes{};
  EXPECT_FALSE(first_session->acquire());
  EXPECT_EQ(first.reads, 0u);
  EXPECT_EQ(second.reads, 0u);
  auto second_lease = second_session->acquire();
  ASSERT_TRUE(second_lease);
  EXPECT_EQ(second_lease.read(0x1000, bytes), simdojo::DmaAccessOutcome::Complete);
  EXPECT_EQ(second.reads, 1u);
  second_lease = {};
  EXPECT_TRUE(device.detach_transport(&second_endpoints));
}

TEST(PciBackingSession, RevokeRefusesNewOperationsAndDetachWaitsForAdmittedOne) {
  SessionTestDevice device;
  OutcomeTransport transport;
  simdojo::PciDevice::Transport endpoints{.irq = &transport, .dma = &transport};
  ASSERT_TRUE(device.attach_transport(&endpoints));
  std::shared_ptr<simdojo::PciTransportSession> session = device.capture_session();
  auto admitted = session->acquire();
  ASSERT_TRUE(admitted);

  std::future<bool> detached =
      std::async(std::launch::async, [&]() { return device.detach_transport(&endpoints); });
  for (int attempt = 0;
       attempt < 100 && session->state() == simdojo::PciTransportSession::State::Open; ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_EQ(session->state(), simdojo::PciTransportSession::State::Closing);
  EXPECT_FALSE(session->acquire()) << "revocation admitted a new endpoint operation";
  EXPECT_EQ(detached.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout)
      << "detach returned while an admitted operation still held endpoint lifetime";

  admitted = {};
  EXPECT_TRUE(detached.get());
  EXPECT_EQ(session->state(), simdojo::PciTransportSession::State::Revoked);
}

TEST(PciBackingSession, RefusesReattachUntilDetachedSessionDrains) {
  SessionTestDevice device;
  OutcomeTransport first;
  OutcomeTransport second;
  simdojo::PciDevice::Transport first_endpoints{.irq = &first, .dma = &first};
  simdojo::PciDevice::Transport second_endpoints{.irq = &second, .dma = &second};
  ASSERT_TRUE(device.attach_transport(&first_endpoints));
  std::shared_ptr<simdojo::PciTransportSession> first_session = device.capture_session();
  simdojo::PciTransportSession::OperationLease admitted = first_session->acquire();
  ASSERT_TRUE(admitted);

  std::future<bool> detached =
      std::async(std::launch::async, [&]() { return device.detach_transport(&first_endpoints); });
  for (int attempt = 0;
       attempt < 100 && first_session->state() == simdojo::PciTransportSession::State::Open;
       ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  if (first_session->state() != simdojo::PciTransportSession::State::Closing) {
    admitted = {};
    EXPECT_TRUE(detached.get());
    FAIL() << "detach did not begin revoking the session";
  }

  EXPECT_TRUE(device.transport_attached());
  EXPECT_FALSE(device.attach_transport(&second_endpoints))
      << "a new transport was installed while the detached session still used its endpoints";

  admitted = {};
  ASSERT_TRUE(detached.get());
  EXPECT_FALSE(device.transport_attached());
  EXPECT_TRUE(device.attach_transport(&second_endpoints));
  EXPECT_TRUE(device.detach_transport(&second_endpoints));
}

TEST(PciPhysicalMemoryAccess, PreservesTypedOutcomesAndRejectsARevokedGeneration) {
  SessionTestDevice device;
  OutcomeTransport transport;
  simdojo::PciDevice::Transport endpoints{.irq = &transport, .dma = &transport};
  ASSERT_TRUE(device.attach_transport(&endpoints));
  std::shared_ptr<simdojo::PciTransportSession> session = device.capture_session();
  SessionBackedMemory memory(session);
  rocjitsu::PciPhysicalMemoryAccess physical(memory);
  std::array<std::byte, 4> bytes{};

  for (const simdojo::DmaAccessOutcome transport_outcome :
       {simdojo::DmaAccessOutcome::Complete, simdojo::DmaAccessOutcome::Unavailable,
        simdojo::DmaAccessOutcome::Faulted, simdojo::DmaAccessOutcome::Malformed}) {
    transport.outcome = transport_outcome;
    const rocjitsu::amdgpu::VmAccessOutcome expected = [&]() {
      switch (transport_outcome) {
      case simdojo::DmaAccessOutcome::Complete:
        return rocjitsu::amdgpu::VmAccessOutcome::Complete;
      case simdojo::DmaAccessOutcome::Unavailable:
        return rocjitsu::amdgpu::VmAccessOutcome::Unavailable;
      case simdojo::DmaAccessOutcome::Faulted:
        return rocjitsu::amdgpu::VmAccessOutcome::Faulted;
      case simdojo::DmaAccessOutcome::Malformed:
        return rocjitsu::amdgpu::VmAccessOutcome::Malformed;
      }
      return rocjitsu::amdgpu::VmAccessOutcome::Malformed;
    }();
    EXPECT_EQ(physical.read(rocjitsu::amdgpu::VmMemoryDomain::System, 0x2000, bytes), expected);
  }

  EXPECT_EQ(physical.read(rocjitsu::amdgpu::VmMemoryDomain::Local, 0, bytes),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(physical.read(rocjitsu::amdgpu::VmMemoryDomain::Local, memory.vram.size(), bytes),
            rocjitsu::amdgpu::VmAccessOutcome::Faulted);
  EXPECT_EQ(physical.read(rocjitsu::amdgpu::VmMemoryDomain::Compatibility, 0, bytes),
            rocjitsu::amdgpu::VmAccessOutcome::Malformed);

  transport.outcome = simdojo::DmaAccessOutcome::Complete;
  EXPECT_EQ(physical.atomic_store(rocjitsu::amdgpu::VmMemoryDomain::Local, 0, sizeof(uint32_t),
                                  0x01020304),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  const auto local_load =
      physical.atomic_load(rocjitsu::amdgpu::VmMemoryDomain::Local, 0, sizeof(uint32_t));
  EXPECT_EQ(local_load.outcome, rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(local_load.value, 0x01020304u);
  const auto local_cas = physical.compare_exchange(rocjitsu::amdgpu::VmMemoryDomain::Local, 0,
                                                   sizeof(uint32_t), 0x01020304, 0x12345678);
  EXPECT_EQ(local_cas.outcome, rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_TRUE(local_cas.exchanged);
  const auto system_cas = physical.compare_exchange(rocjitsu::amdgpu::VmMemoryDomain::System,
                                                    0x2000, sizeof(uint32_t), 0, 1);
  EXPECT_EQ(system_cas.outcome, rocjitsu::amdgpu::VmAccessOutcome::Faulted)
      << "the fake DMA engine does not claim atomicity via read/write synthesis";

  std::shared_ptr<simdojo::PciTransportSession> closing =
      device.revoke_transport_session(&endpoints);
  ASSERT_NE(closing, nullptr);
  closing->wait_until_drained();
  EXPECT_EQ(physical.read(rocjitsu::amdgpu::VmMemoryDomain::System, 0x2000, bytes),
            rocjitsu::amdgpu::VmAccessOutcome::Unavailable);
  EXPECT_EQ(physical.read(rocjitsu::amdgpu::VmMemoryDomain::Local, 0, bytes),
            rocjitsu::amdgpu::VmAccessOutcome::Unavailable);
  EXPECT_EQ(
      physical.atomic_load(rocjitsu::amdgpu::VmMemoryDomain::Local, 0, sizeof(uint32_t)).outcome,
      rocjitsu::amdgpu::VmAccessOutcome::Unavailable);
  EXPECT_TRUE(device.detach_transport(&endpoints));
}

TEST(PciPhysicalMemoryAccess, DeviceDestructionRevokesRetainedBackingBeforeOwnerDies) {
  OutcomeTransport transport;
  simdojo::PciDevice::Transport endpoints{.irq = &transport, .dma = &transport};
  std::unique_ptr<rocjitsu::PciPhysicalMemoryAccess> physical;
  std::shared_ptr<simdojo::PciTransportSession> session;
  {
    auto memory = std::make_unique<LifetimeBoundMemory>();
    ASSERT_TRUE(memory->attach_transport(&endpoints));
    session = memory->capture_transport_session();
    ASSERT_NE(session, nullptr);
    physical = std::make_unique<rocjitsu::PciPhysicalMemoryAccess>(*memory);

    std::array<std::byte, 4> bytes{};
    EXPECT_EQ(physical->read(rocjitsu::amdgpu::VmMemoryDomain::Local, 0, bytes),
              rocjitsu::amdgpu::VmAccessOutcome::Complete);
  }

  EXPECT_EQ(session->state(), simdojo::PciTransportSession::State::Revoked);
  std::array<std::byte, 4> bytes{};
  EXPECT_EQ(physical->read(rocjitsu::amdgpu::VmMemoryDomain::Local, 0, bytes),
            rocjitsu::amdgpu::VmAccessOutcome::Unavailable);
  EXPECT_EQ(physical->read(rocjitsu::amdgpu::VmMemoryDomain::System, 0, bytes),
            rocjitsu::amdgpu::VmAccessOutcome::Unavailable);
}

TEST(VramStoreAtomics, PerformsStrongAlignedOperationsWithoutReadWriteSynthesis) {
  rocjitsu::VramStore vram("atomic-vram", 0x2000, 0x1000);
  ASSERT_TRUE(vram.usable());

  ASSERT_TRUE(vram.atomic_store(0x100, sizeof(uint64_t), 7));
  const auto loaded = vram.atomic_load(0x100, sizeof(uint64_t));
  ASSERT_TRUE(loaded.has_value());
  EXPECT_EQ(loaded->value, 7u);

  const auto exchanged = vram.compare_exchange(0x100, sizeof(uint64_t), 7, 11);
  ASSERT_TRUE(exchanged.has_value());
  EXPECT_EQ(exchanged->observed, 7u);
  EXPECT_TRUE(exchanged->exchanged);

  const auto mismatch = vram.compare_exchange(0x100, sizeof(uint64_t), 7, 13);
  ASSERT_TRUE(mismatch.has_value());
  EXPECT_EQ(mismatch->observed, 11u);
  EXPECT_FALSE(mismatch->exchanged);
  EXPECT_FALSE(vram.atomic_load(0x101, sizeof(uint64_t)).has_value());
  EXPECT_FALSE(vram.atomic_store(0x2000, sizeof(uint64_t), 1));
}

TEST(VramStore, RejectsInvalidApertureSizesBeforeMapping) {
  const rocjitsu::VramStore zero_aperture("zero-aperture", 0x2000, 0);
  EXPECT_FALSE(zero_aperture.usable());
  EXPECT_TRUE(zero_aperture.aperture().empty());

  const rocjitsu::VramStore oversized_aperture("oversized-aperture", 0x1000, 0x2000);
  EXPECT_FALSE(oversized_aperture.usable());
  EXPECT_TRUE(oversized_aperture.aperture().empty());

  const rocjitsu::VramStore zero_backing("zero-backing", 0, 0);
  EXPECT_FALSE(zero_backing.usable());
}

class AlwaysAvailableAddressSpace final : public rocjitsu::amdgpu::AddressSpaceTranslator,
                                          public rocjitsu::amdgpu::PhysicalMemoryAccess {
public:
  rocjitsu::amdgpu::VmTranslationResult
  translate(uint64_t address, std::size_t size,
            rocjitsu::amdgpu::VmAccessKind /*access*/) const override {
    return {
        .outcome = rocjitsu::amdgpu::VmAccessOutcome::Complete,
        .translation = {.domain = rocjitsu::amdgpu::VmMemoryDomain::System,
                        .address = address,
                        .contiguous_bytes = size,
                        .mtype = rocjitsu::amdgpu::Mtype::RW,
                        .permissions = {.readable = true, .writable = true, .executable = true}}};
  }

  rocjitsu::amdgpu::VmAccessOutcome read(rocjitsu::amdgpu::VmMemoryDomain domain, uint64_t,
                                         std::span<std::byte> bytes) override {
    if (domain != rocjitsu::amdgpu::VmMemoryDomain::System)
      return rocjitsu::amdgpu::VmAccessOutcome::Malformed;
    std::ranges::fill(bytes, std::byte{0});
    return rocjitsu::amdgpu::VmAccessOutcome::Complete;
  }

  rocjitsu::amdgpu::VmAccessOutcome write(rocjitsu::amdgpu::VmMemoryDomain domain, uint64_t,
                                          std::span<const std::byte>) override {
    if (domain != rocjitsu::amdgpu::VmMemoryDomain::System)
      return rocjitsu::amdgpu::VmAccessOutcome::Malformed;
    return rocjitsu::amdgpu::VmAccessOutcome::Complete;
  }
};

// MES submits fixed 64-dword API frames through a VMID-0 ring in GART. The
// device has to walk the GART PTEs, execute both the requested operation and
// the query-status frame appended behind it, and publish all three pieces of
// progress: the API status, the ring fence, and the queue read pointer.
class GpuDeviceMes : public GpuDevice {
protected:
  static constexpr uint64_t kPageTable = 0x10000;
  static constexpr uint64_t kGartStart = 0x100000000ULL;
  static constexpr uint64_t kRingGpu = kGartStart;
  static constexpr uint64_t kReadPointerGpu = kGartStart + 0x1000;
  static constexpr uint64_t kApiStatusGpu = kGartStart + 0x2000;
  static constexpr uint64_t kRingFenceGpu = kGartStart + 0x3000;
  static constexpr uint64_t kSchedulerMqdGpu = kGartStart + 0x4000;
  static constexpr uint64_t kSchedulerRingGpu = kGartStart + 0x5000;
  static constexpr uint64_t kSchedulerReadPointerGpu = kGartStart + 0x6000;
  static constexpr uint64_t kSchedulerApiStatusGpu = kGartStart + 0x7000;
  static constexpr uint64_t kSchedulerRingFenceGpu = kGartStart + 0x8000;
  static constexpr uint64_t kRingPhysical = 0x200000;
  static constexpr uint64_t kReadPointerPhysical = 0x210000;
  static constexpr uint64_t kApiStatusPhysical = 0x220000;
  static constexpr uint64_t kRingFencePhysical = 0x230000;
  static constexpr uint64_t kSchedulerMqdPhysical = 0x240000;
  static constexpr uint64_t kSchedulerRingPhysical = 0x250000;
  static constexpr uint64_t kSchedulerReadPointerPhysical = 0x260000;
  static constexpr uint64_t kSchedulerApiStatusPhysical = 0x270000;
  static constexpr uint64_t kSchedulerRingFencePhysical = 0x280000;
  static constexpr uint64_t kDefaultComputeDoorbell = 0x900;
  static constexpr uint64_t kDefaultComputeRingGpu = 0x10000;
  static constexpr uint64_t kDefaultComputeReadPointerGpu = 0x11ff0;
  static constexpr uint64_t kDefaultProcessPageTable = 0x30000;
  static constexpr uint64_t kDefaultProcessPdb2 = 0x31000;
  static constexpr uint64_t kDefaultProcessPdb1 = 0x32000;
  static constexpr uint64_t kDefaultProcessPdb0 = 0x33000;
  static constexpr uint64_t kDefaultProcessPtb = 0x34000;
  static constexpr uint64_t kDefaultComputeRingPhysical = 0x290000;
  static constexpr uint64_t kDefaultComputeSecondPagePhysical = 0x2a0000;
  static constexpr uint64_t kDefaultComputeReadPointerPhysical =
      kDefaultComputeSecondPagePhysical + 0xff0;

  RecordingTransport transport_;
  simdojo::PciDevice::Transport attached_{.irq = &transport_, .dma = &transport_};

  void SetUp() override {
    ASSERT_TRUE(device_.attach_transport(&attached_));

    configure_gart();
  }

  void configure_gart() {
    constexpr uint64_t kGcBase = 0x1260;
    const auto absolute = [](uint64_t reg) { return (kGcBase + reg) * 4; };
    write_register_at(absolute(0x169f), static_cast<uint32_t>(kPageTable));
    write_register_at(absolute(0x16a0), static_cast<uint32_t>(kPageTable >> 32));
    write_register_at(absolute(0x16bf), static_cast<uint32_t>(kGartStart >> 12));
    write_register_at(absolute(0x16c0), static_cast<uint32_t>(kGartStart >> 44));
    write_register_at(absolute(0x16df), static_cast<uint32_t>((kGartStart + 0x8fff) >> 12));
    write_register_at(absolute(0x16e0), static_cast<uint32_t>((kGartStart + 0x8fff) >> 44));

    for (const auto [page, physical] :
         {std::pair{0u, kRingPhysical}, std::pair{1u, kReadPointerPhysical},
          std::pair{2u, kApiStatusPhysical}, std::pair{3u, kRingFencePhysical},
          std::pair{4u, kSchedulerMqdPhysical}, std::pair{5u, kSchedulerRingPhysical},
          std::pair{6u, kSchedulerReadPointerPhysical}, std::pair{7u, kSchedulerApiStatusPhysical},
          std::pair{8u, kSchedulerRingFencePhysical}}) {
      const uint64_t pte = physical | 0x3;
      auto raw = std::bit_cast<std::array<std::byte, 8>>(pte);
      ASSERT_EQ(device_.bar_access(rocjitsu::GpuPciDevice::kVramBar, raw,
                                   kPageTable + page * sizeof(uint64_t), /*write=*/true),
                8);
    }

    // Configuration writes are pending state. The VMID-0 identity changes
    // translator only when the hub invalidation publishes the snapshot.
    write_register_at(absolute(0x1657 + 17), 1);

    write_register_at(absolute(0x1fb1), static_cast<uint32_t>(kRingGpu >> 8));
    write_register_at(absolute(0x1fb2), static_cast<uint32_t>(kRingGpu >> 40));
    write_register_at(absolute(0x1fb4), static_cast<uint32_t>(kReadPointerGpu));
    write_register_at(absolute(0x1fb5), static_cast<uint32_t>(kReadPointerGpu >> 32));
    write_register_at(absolute(0x1fb8), 0x40000060);
    write_register_at(absolute(0x1fba), 9);
    write_register_at(absolute(0x1fab), 1);
  }

  void TearDown() override {
    if (device_.transport_attached())
      EXPECT_TRUE(device_.detach_transport(&attached_));
  }

  void store_dword(uint64_t physical, uint32_t value) {
    const auto raw = std::bit_cast<std::array<std::byte, 4>>(value);
    for (std::size_t byte_index = 0; byte_index < raw.size(); ++byte_index) {
      transport_.memory[physical + byte_index] = raw[byte_index];
    }
  }

  void store_qword(uint64_t physical, uint64_t value) {
    const auto raw = std::bit_cast<std::array<std::byte, 8>>(value);
    for (std::size_t byte_index = 0; byte_index < raw.size(); ++byte_index) {
      transport_.memory[physical + byte_index] = raw[byte_index];
    }
  }

  void store_vram_qword(uint64_t offset, uint64_t value) {
    auto raw = std::bit_cast<std::array<std::byte, 8>>(value);
    ASSERT_EQ(device_.bar_access(rocjitsu::GpuPciDevice::kVramBar, raw, offset,
                                 /*write=*/true),
              8);
  }

  void store_vram_dword(uint64_t offset, uint32_t value) {
    auto raw = std::bit_cast<std::array<std::byte, 4>>(value);
    ASSERT_EQ(device_.bar_access(rocjitsu::GpuPciDevice::kVramBar, raw, offset,
                                 /*write=*/true),
              4);
  }

  [[nodiscard]] uint32_t vram_dword_at(uint64_t offset) {
    std::array<std::byte, 4> raw{};
    EXPECT_EQ(device_.bar_access(rocjitsu::GpuPciDevice::kVramBar, raw, offset,
                                 /*write=*/false),
              4);
    return std::bit_cast<uint32_t>(raw);
  }

  template <std::size_t N> void ring_doorbell(uint64_t offset, std::array<std::byte, N> &value) {
    ASSERT_EQ(device_.bar_access(rocjitsu::GpuPciDevice::kDoorbellBar, value, offset,
                                 /*write=*/true),
              static_cast<int64_t>(N));
    device_.drain_doorbell_inbox_for_test();
  }

  void submit_default_compute_queue() {
    store_vram_qword(kDefaultProcessPageTable, kDefaultProcessPdb2 | 1);
    store_vram_qword(kDefaultProcessPdb2, kDefaultProcessPdb1 | 1);
    store_vram_qword(kDefaultProcessPdb1, kDefaultProcessPdb0 | 1);
    store_vram_qword(kDefaultProcessPdb0, kDefaultProcessPtb | 1);
    store_vram_qword(kDefaultProcessPtb + 0x10 * sizeof(uint64_t),
                     kDefaultComputeRingPhysical | 0x63);
    store_vram_qword(kDefaultProcessPtb + 0x11 * sizeof(uint64_t),
                     kDefaultComputeSecondPagePhysical | 0x63);

    store_qword(kReadPointerPhysical, 0);
    store_dword(kApiStatusPhysical, 0);
    store_dword(kRingPhysical, 0x00040021);
    store_dword(kRingPhysical + 1 * sizeof(uint32_t), 3);
    store_qword(kRingPhysical + 2 * sizeof(uint32_t), kDefaultProcessPageTable);
    store_qword(kRingPhysical + 20 * sizeof(uint32_t), kSchedulerMqdGpu);
    store_dword(kRingPhysical + 28 * sizeof(uint32_t), 1);
    store_qword(kRingPhysical + 38 * sizeof(uint32_t), kApiStatusGpu);
    store_qword(kRingPhysical + 40 * sizeof(uint32_t), 1);

    store_dword(kSchedulerMqdPhysical + 130 * sizeof(uint32_t), 0);
    store_dword(kSchedulerMqdPhysical + 136 * sizeof(uint32_t),
                static_cast<uint32_t>(kDefaultComputeRingGpu >> 8));
    store_dword(kSchedulerMqdPhysical + 137 * sizeof(uint32_t),
                static_cast<uint32_t>(kDefaultComputeRingGpu >> 40));
    store_dword(kSchedulerMqdPhysical + 139 * sizeof(uint32_t),
                static_cast<uint32_t>(kDefaultComputeReadPointerGpu));
    store_dword(kSchedulerMqdPhysical + 140 * sizeof(uint32_t),
                static_cast<uint32_t>(kDefaultComputeReadPointerGpu >> 32));
    store_dword(kSchedulerMqdPhysical + 143 * sizeof(uint32_t), kDefaultComputeDoorbell);
    store_dword(kSchedulerMqdPhysical + 145 * sizeof(uint32_t), 9);

    auto mes_doorbell = std::bit_cast<std::array<std::byte, sizeof(uint64_t)>>(uint64_t{64});
    ring_doorbell(0x60, mes_doorbell);
    ASSERT_EQ(transport_.dword_at(kApiStatusPhysical), 1u);
    ASSERT_EQ(soc_.mes_engine().active_queues(), 1u);
    ASSERT_EQ(soc_.queue_registry().active_queues(), 1u);
    ASSERT_EQ(command_processor_.registered_pm4_queue_count_for_test(), 1u);
  }

  [[nodiscard]] std::optional<rocjitsu::amdgpu::AddressSpaceHandle>
  submit_sdma_queue(uint32_t process_id, uint64_t doorbell_offset, uint64_t ring_gpu,
                    uint64_t read_pointer_gpu, uint64_t write_pointer_gpu,
                    uint64_t page_table_base) {
    store_qword(kReadPointerPhysical, 0);
    store_dword(kApiStatusPhysical, 0);
    store_dword(kRingPhysical, 0x00040021);
    store_dword(kRingPhysical + 1 * sizeof(uint32_t), process_id);
    store_qword(kRingPhysical + 2 * sizeof(uint32_t), page_table_base);
    store_dword(kRingPhysical + 18 * sizeof(uint32_t),
                static_cast<uint32_t>(doorbell_offset / sizeof(uint32_t)));
    store_qword(kRingPhysical + 20 * sizeof(uint32_t), kSchedulerMqdGpu);
    store_qword(kRingPhysical + 22 * sizeof(uint32_t), write_pointer_gpu);
    store_dword(kRingPhysical + 28 * sizeof(uint32_t), 2);
    store_dword(kRingPhysical + 30 * sizeof(uint32_t), 16);
    store_qword(kRingPhysical + 38 * sizeof(uint32_t), kApiStatusGpu);
    store_qword(kRingPhysical + 40 * sizeof(uint32_t), 1);

    store_dword(kSchedulerMqdPhysical + 1 * sizeof(uint32_t), static_cast<uint32_t>(ring_gpu >> 8));
    store_dword(kSchedulerMqdPhysical + 2 * sizeof(uint32_t),
                static_cast<uint32_t>(ring_gpu >> 40));
    store_dword(kSchedulerMqdPhysical + 7 * sizeof(uint32_t),
                static_cast<uint32_t>(read_pointer_gpu));
    store_dword(kSchedulerMqdPhysical + 8 * sizeof(uint32_t),
                static_cast<uint32_t>(read_pointer_gpu >> 32));
    store_dword(kSchedulerMqdPhysical + 126 * sizeof(uint32_t), 1);
    store_dword(kSchedulerMqdPhysical + 127 * sizeof(uint32_t), 7);

    auto mes_doorbell = std::bit_cast<std::array<std::byte, sizeof(uint64_t)>>(uint64_t{64});
    ring_doorbell(0x60, mes_doorbell);
    return soc_.gpu_vm().find_vmid(process_id);
  }
};

TEST_F(GpuDeviceMes, ExecutesTheKernelQueueAndPublishesBothCompletionFences) {
  // SET_HW_RESOURCES: status begins at dword 50 in its 64-dword frame.
  store_dword(kRingPhysical, 0x00040001);
  store_qword(kRingPhysical + 50 * 4, kApiStatusGpu);
  store_qword(kRingPhysical + 52 * 4, 1);

  // QUERY_SCHEDULER_STATUS follows it and carries the ring fence at dword 2.
  constexpr uint64_t kQuery = kRingPhysical + 64 * 4;
  store_dword(kQuery, 0x000400b1);
  store_qword(kQuery + 2 * 4, kRingFenceGpu);
  store_qword(kQuery + 4 * 4, 7);

  auto doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{128});
  ASSERT_EQ(device_.bar_access(rocjitsu::GpuPciDevice::kDoorbellBar, doorbell, 0x60,
                               /*write=*/true),
            8);

  EXPECT_EQ(transport_.dword_at(kApiStatusPhysical), 0u);
  EXPECT_EQ(transport_.dword_at(kRingFencePhysical), 0u);
  EXPECT_EQ(transport_.dword_at(kReadPointerPhysical), 0u)
      << "a transport callback executed semantic doorbell work inline";

  device_.drain_doorbell_inbox_for_test();

  EXPECT_EQ(transport_.dword_at(kApiStatusPhysical), 1u);
  EXPECT_EQ(transport_.dword_at(kRingFencePhysical), 7u);
  EXPECT_EQ(transport_.dword_at(kReadPointerPhysical), 128u);
}

TEST_F(GpuDeviceMes, DropsADoorbellQueuedByARevokedTransportGeneration) {
  store_dword(kRingPhysical, 0x00040001);
  store_qword(kRingPhysical + 50 * 4, kApiStatusGpu);
  store_qword(kRingPhysical + 52 * 4, 1);

  const uint64_t old_generation = device_.transport_session_generation();
  ASSERT_NE(old_generation, 0u);
  auto doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{64});
  ASSERT_EQ(device_.bar_access(rocjitsu::GpuPciDevice::kDoorbellBar, doorbell, 0x60,
                               /*write=*/true),
            8);

  ASSERT_TRUE(device_.detach_transport(&attached_));
  ASSERT_TRUE(device_.attach_transport(&attached_));
  ASSERT_NE(device_.transport_session_generation(), old_generation);
  device_.drain_doorbell_inbox_for_test();

  EXPECT_EQ(transport_.dword_at(kApiStatusPhysical), 0u);
  EXPECT_EQ(transport_.dword_at(kReadPointerPhysical), 0u)
      << "work captured for the old peer generation reached its replacement";
}

TEST_F(GpuDeviceMes, ResetDropsDoorbellsQueuedInTheSameTransportGeneration) {
  store_dword(kRingPhysical, 0x00040001);
  store_qword(kRingPhysical + 50 * 4, kApiStatusGpu);
  store_qword(kRingPhysical + 52 * 4, 1);

  const uint64_t generation = device_.transport_session_generation();
  auto doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{64});
  ASSERT_EQ(device_.bar_access(rocjitsu::GpuPciDevice::kDoorbellBar, doorbell, 0x60,
                               /*write=*/true),
            8);

  device_.reset(simdojo::ResetKind::FunctionLevel);
  ASSERT_EQ(device_.transport_session_generation(), generation);
  device_.drain_doorbell_inbox_for_test();

  EXPECT_EQ(transport_.dword_at(kApiStatusPhysical), 0u);
  EXPECT_EQ(transport_.dword_at(kReadPointerPhysical), 0u)
      << "pre-reset semantic work crossed the reset epoch";
}

TEST_F(GpuDeviceMes, RetainsADoorbellQueuedWhileAnEarlierNotificationIsDraining) {
  store_dword(kRingPhysical, 0x00040001);
  store_qword(kRingPhysical + 50 * 4, kApiStatusGpu);
  store_qword(kRingPhysical + 52 * 4, 1);

  constexpr uint64_t kQuery = kRingPhysical + 64 * 4;
  store_dword(kQuery, 0x000400b1);
  store_qword(kQuery + 2 * 4, kRingFenceGpu);
  store_qword(kQuery + 4 * 4, 7);

  transport_.block_next_read_at(kRingPhysical);
  auto doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{64});
  ASSERT_EQ(device_.bar_access(rocjitsu::GpuPciDevice::kDoorbellBar, doorbell, 0x60,
                               /*write=*/true),
            8);

  auto draining =
      std::async(std::launch::async, [this]() { device_.drain_doorbell_inbox_for_test(); });
  transport_.wait_for_blocked_read();

  doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{128});
  EXPECT_EQ(device_.bar_access(rocjitsu::GpuPciDevice::kDoorbellBar, doorbell, 0x60,
                               /*write=*/true),
            8);
  transport_.release_read();
  EXPECT_EQ(draining.wait_for(std::chrono::seconds(1)), std::future_status::ready);
  draining.get();

  EXPECT_EQ(transport_.dword_at(kApiStatusPhysical), 1u);
  EXPECT_EQ(transport_.dword_at(kRingFencePhysical), 0u);
  EXPECT_EQ(transport_.dword_at(kReadPointerPhysical), 64u);

  device_.drain_doorbell_inbox_for_test();
  EXPECT_EQ(transport_.dword_at(kRingFencePhysical), 7u);
  EXPECT_EQ(transport_.dword_at(kReadPointerPhysical), 128u)
      << "a notification accepted during a drain was lost";
}

TEST_F(GpuDeviceMes, ResetWaitsForAnAdmittedDoorbellObserver) {
  store_dword(kRingPhysical, 0x00040001);
  store_qword(kRingPhysical + 50 * 4, kApiStatusGpu);
  store_qword(kRingPhysical + 52 * 4, 1);

  transport_.block_next_read_at(kRingPhysical);
  auto doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{64});
  ASSERT_EQ(device_.bar_access(rocjitsu::GpuPciDevice::kDoorbellBar, doorbell, 0x60,
                               /*write=*/true),
            8);

  auto draining =
      std::async(std::launch::async, [this]() { device_.drain_doorbell_inbox_for_test(); });
  transport_.wait_for_blocked_read();
  auto resetting = std::async(std::launch::async,
                              [this]() { device_.reset(simdojo::ResetKind::FunctionLevel); });

  EXPECT_EQ(resetting.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout)
      << "reset did not wait for the admitted observer";
  transport_.release_read();
  EXPECT_EQ(draining.wait_for(std::chrono::seconds(1)), std::future_status::ready);
  EXPECT_EQ(resetting.wait_for(std::chrono::seconds(1)), std::future_status::ready);
  draining.get();
  resetting.get();
}

TEST_F(GpuDeviceMes, HonorsTheAlignedStatusInSetHardwareResourcesOne) {
  // The second-generation resource packet puts its first 64-bit member after
  // a four-byte header, so the ABI inserts one dword of padding before status.
  store_dword(kRingPhysical, 0x00040131);
  store_qword(kRingPhysical + 2 * 4, kApiStatusGpu);
  store_qword(kRingPhysical + 4 * 4, 1);

  constexpr uint64_t kQuery = kRingPhysical + 64 * 4;
  store_dword(kQuery, 0x000400b1);
  store_qword(kQuery + 2 * 4, kRingFenceGpu);
  store_qword(kQuery + 4 * 4, 8);

  auto doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{128});
  ring_doorbell(0x60, doorbell);

  EXPECT_EQ(transport_.dword_at(kApiStatusPhysical), 1u);
  EXPECT_EQ(transport_.dword_at(kRingFencePhysical), 8u);
  EXPECT_EQ(transport_.dword_at(kReadPointerPhysical), 128u);
}

TEST_F(GpuDeviceMes, InvalidatingAPasidAdvancesItsGpuVmTranslationEpoch) {
  constexpr uint32_t kPasid = 5;
  auto access = std::make_shared<AlwaysAvailableAddressSpace>();
  const rocjitsu::amdgpu::AddressSpaceHandle address_space =
      soc_.gpu_vm().register_translated(kPasid, access, access);
  ASSERT_TRUE(address_space);
  const uint64_t old_epoch = soc_.gpu_vm().lookup(address_space)->translation_epoch;

  // INV_TLBS puts its API status at dword 2 and the packed selector at dword 6.
  // inv_sel=0 selects a PASID; inv_sel_id occupies the high 16 bits.
  store_dword(kRingPhysical, 0x00040141);
  store_qword(kRingPhysical + 2 * 4, kApiStatusGpu);
  store_qword(kRingPhysical + 4 * 4, 0x31);
  store_dword(kRingPhysical + 6 * 4, kPasid << 16);

  auto doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{64});
  ring_doorbell(0x60, doorbell);

  ASSERT_TRUE(soc_.gpu_vm().lookup(address_space));
  EXPECT_EQ(soc_.gpu_vm().lookup(address_space)->translation_epoch, old_epoch + 1);
  EXPECT_EQ(transport_.dword_at(kApiStatusPhysical), 0x31u);
}

TEST_F(GpuDeviceMes, UpdatingRootPreservesAddressSpaceIdentityAndRedirectsExistingQueues) {
  constexpr uint32_t kPasid = 3;
  constexpr uint64_t kProcessContext = 0x12345000;
  constexpr uint64_t kVirtualAddress = 0x10000;
  constexpr uint64_t kRootA = 0x30000;
  constexpr uint64_t kPdb2A = 0x31000;
  constexpr uint64_t kPdb1A = 0x32000;
  constexpr uint64_t kPdb0A = 0x33000;
  constexpr uint64_t kPtbA = 0x34000;
  constexpr uint64_t kRootB = 0x35000;
  constexpr uint64_t kPdb2B = 0x36000;
  constexpr uint64_t kPdb1B = 0x37000;
  constexpr uint64_t kPdb0B = 0x38000;
  constexpr uint64_t kPtbB = 0x39000;
  constexpr uint64_t kPhysicalA = 0x2a0000;
  constexpr uint64_t kPhysicalB = 0x2b0000;

  const auto program_root = [&](uint64_t root, uint64_t pdb2, uint64_t pdb1, uint64_t pdb0,
                                uint64_t ptb, uint64_t physical) {
    store_vram_qword(root, pdb2 | 1);
    store_vram_qword(pdb2, pdb1 | 1);
    store_vram_qword(pdb1, pdb0 | 1);
    store_vram_qword(pdb0, ptb | 1);
    store_vram_qword(ptb + 0x10 * sizeof(uint64_t), physical | 0x63);
  };
  program_root(kRootA, kPdb2A, kPdb1A, kPdb0A, kPtbA, kPhysicalA);
  program_root(kRootB, kPdb2B, kPdb1B, kPdb0B, kPtbB, kPhysicalB);
  store_dword(kPhysicalA, 0xaaaaaaaa);
  store_dword(kPhysicalB, 0xbbbbbbbb);

  store_dword(kRingPhysical, 0x00040021);
  store_dword(kRingPhysical + 1 * 4, kPasid);
  store_qword(kRingPhysical + 2 * 4, kRootA);
  store_qword(kRingPhysical + 10 * 4, kProcessContext);
  store_qword(kRingPhysical + 20 * 4, kSchedulerMqdGpu);
  store_dword(kRingPhysical + 28 * 4, 1);
  store_qword(kRingPhysical + 38 * 4, kApiStatusGpu);
  store_qword(kRingPhysical + 40 * 4, 1);

  store_dword(kSchedulerMqdPhysical + 130 * 4, 0);
  store_dword(kSchedulerMqdPhysical + 136 * 4, static_cast<uint32_t>(kVirtualAddress >> 8));
  store_dword(kSchedulerMqdPhysical + 137 * 4, static_cast<uint32_t>(kVirtualAddress >> 40));
  store_dword(kSchedulerMqdPhysical + 139 * 4, static_cast<uint32_t>(kVirtualAddress));
  store_dword(kSchedulerMqdPhysical + 140 * 4, static_cast<uint32_t>(kVirtualAddress >> 32));
  store_dword(kSchedulerMqdPhysical + 143 * 4, 0x900);
  store_dword(kSchedulerMqdPhysical + 145 * 4, 9);

  auto doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{64});
  ring_doorbell(0x60, doorbell);
  const std::optional<rocjitsu::amdgpu::AddressSpaceHandle> address_space =
      soc_.gpu_vm().find_vmid(kPasid);
  ASSERT_TRUE(address_space);
  const uint64_t old_epoch = soc_.gpu_vm().lookup(*address_space)->translation_epoch;
  std::array<std::byte, sizeof(uint32_t)> value{};
  ASSERT_EQ(soc_.gpu_vm().read(*address_space, kVirtualAddress, value),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(std::bit_cast<uint32_t>(value), 0xaaaaaaaau);

  constexpr uint64_t kUpdate = kRingPhysical + 64 * 4;
  store_dword(kUpdate, 0x000400f1);
  store_qword(kUpdate + 2 * 4, kRootB);
  store_qword(kUpdate + 4 * 4, kProcessContext);
  store_qword(kUpdate + 6 * 4, kRingFenceGpu);
  store_qword(kUpdate + 8 * 4, 0x44);

  doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{128});
  ring_doorbell(0x60, doorbell);

  ASSERT_TRUE(soc_.gpu_vm().lookup(*address_space));
  EXPECT_EQ(soc_.gpu_vm().lookup(*address_space)->translation_epoch, old_epoch + 1);
  ASSERT_EQ(soc_.gpu_vm().read(*address_space, kVirtualAddress, value),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(std::bit_cast<uint32_t>(value), 0xbbbbbbbbu);
  EXPECT_EQ(transport_.dword_at(kRingFencePhysical), 0x44u);
}

TEST_F(GpuDeviceMes, RemoveQueueStopsAFormerSchedulerDoorbell) {
  // Map a scheduler queue, then remove the same doorbell in the following MES
  // frame. The remove API carries the doorbell in dword units.
  store_dword(kRingPhysical, 0x00040021);
  store_qword(kRingPhysical + 20 * 4, kSchedulerMqdGpu);
  store_dword(kRingPhysical + 28 * 4, 3);
  store_qword(kRingPhysical + 38 * 4, kApiStatusGpu);
  store_qword(kRingPhysical + 40 * 4, 1);

  store_dword(kSchedulerMqdPhysical + 130 * 4, 1);
  store_dword(kSchedulerMqdPhysical + 136 * 4, static_cast<uint32_t>(kSchedulerRingGpu >> 8));
  store_dword(kSchedulerMqdPhysical + 137 * 4, static_cast<uint32_t>(kSchedulerRingGpu >> 40));
  store_dword(kSchedulerMqdPhysical + 139 * 4, static_cast<uint32_t>(kSchedulerReadPointerGpu));
  store_dword(kSchedulerMqdPhysical + 140 * 4,
              static_cast<uint32_t>(kSchedulerReadPointerGpu >> 32));
  store_dword(kSchedulerMqdPhysical + 143 * 4, 0x40000058);
  store_dword(kSchedulerMqdPhysical + 145 * 4, 9);

  constexpr uint64_t kRemove = kRingPhysical + 64 * 4;
  store_dword(kRemove, 0x00040031);
  store_dword(kRemove + 1 * 4, 0x58 / sizeof(uint32_t));
  store_qword(kRemove + 6 * 4, kRingFenceGpu);
  store_qword(kRemove + 8 * 4, 2);

  auto mes_doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{128});
  ring_doorbell(0x60, mes_doorbell);
  ASSERT_EQ(transport_.dword_at(kApiStatusPhysical), 1u);
  ASSERT_EQ(transport_.dword_at(kRingFencePhysical), 2u);

  EXPECT_EQ(soc_.mes_engine().active_queues(), 0u);

  store_dword(kSchedulerRingPhysical, 0x000400b1);
  store_qword(kSchedulerRingPhysical + 2 * 4, kSchedulerApiStatusGpu);
  store_qword(kSchedulerRingPhysical + 4 * 4, 3);
  auto scheduler_doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{64});
  ring_doorbell(0x58, scheduler_doorbell);
  EXPECT_EQ(transport_.dword_at(kSchedulerApiStatusPhysical), 0u);
  EXPECT_EQ(transport_.dword_at(kSchedulerReadPointerPhysical), 0u);
}

TEST_F(GpuDeviceMes, RetriesRemoveQueueCompletionWithoutReplayingRemoval) {
  store_dword(kRingPhysical, 0x00040021);
  store_qword(kRingPhysical + 20 * 4, kSchedulerMqdGpu);
  store_dword(kRingPhysical + 28 * 4, 3);
  store_qword(kRingPhysical + 38 * 4, kApiStatusGpu);
  store_qword(kRingPhysical + 40 * 4, 1);

  store_dword(kSchedulerMqdPhysical + 130 * 4, 1);
  store_dword(kSchedulerMqdPhysical + 136 * 4, static_cast<uint32_t>(kSchedulerRingGpu >> 8));
  store_dword(kSchedulerMqdPhysical + 137 * 4, static_cast<uint32_t>(kSchedulerRingGpu >> 40));
  store_dword(kSchedulerMqdPhysical + 139 * 4, static_cast<uint32_t>(kSchedulerReadPointerGpu));
  store_dword(kSchedulerMqdPhysical + 140 * 4,
              static_cast<uint32_t>(kSchedulerReadPointerGpu >> 32));
  store_dword(kSchedulerMqdPhysical + 143 * 4, 0x40000058);
  store_dword(kSchedulerMqdPhysical + 145 * 4, 9);

  auto mes_doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{64});
  ring_doorbell(0x60, mes_doorbell);
  ASSERT_EQ(transport_.dword_at(kApiStatusPhysical), 1u);
  ASSERT_EQ(transport_.dword_at(kReadPointerPhysical), 64u);

  constexpr uint64_t kRemove = kRingPhysical + 64 * 4;
  store_dword(kRemove, 0x00040031);
  store_dword(kRemove + 1 * 4, 0x58 / sizeof(uint32_t));
  store_qword(kRemove + 6 * 4, kRingFenceGpu);
  store_qword(kRemove + 8 * 4, 2);

  store_dword(kSchedulerRingPhysical, 0x000400b1);
  store_qword(kSchedulerRingPhysical + 2 * 4, kSchedulerApiStatusGpu);
  store_qword(kSchedulerRingPhysical + 4 * 4, 3);

  transport_.writes.clear();
  transport_.next_write_outcome =
      std::pair{kRingFencePhysical, simdojo::DmaAccessOutcome::Unavailable};
  mes_doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{128});
  ring_doorbell(0x60, mes_doorbell);

  EXPECT_EQ(transport_.dword_at(kRingFencePhysical), 0u);
  EXPECT_EQ(transport_.dword_at(kReadPointerPhysical), 64u);

  device_.drain_doorbell_inbox_for_test();
  EXPECT_EQ(transport_.dword_at(kRingFencePhysical), 2u);
  EXPECT_EQ(transport_.dword_at(kReadPointerPhysical), 128u)
      << "the transient completion failure required another guest doorbell";
  EXPECT_EQ(
      std::ranges::count_if(transport_.writes,
                            [](const auto &write) { return write.first == kRingFencePhysical; }),
      1u);

  auto scheduler_doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{64});
  ring_doorbell(0x58, scheduler_doorbell);
  EXPECT_EQ(transport_.dword_at(kSchedulerApiStatusPhysical), 0u);
  EXPECT_EQ(transport_.dword_at(kSchedulerReadPointerPhysical), 0u)
      << "retry replayed the already-committed REMOVE_QUEUE semantic";
}

TEST_F(GpuDeviceMes, RejectsUnsupportedGraphicsQueueWithoutChangingCoreState) {
  store_dword(kRingPhysical, 0x00040021);
  store_dword(kRingPhysical + 28 * sizeof(uint32_t), 0);
  store_qword(kRingPhysical + 38 * sizeof(uint32_t), kApiStatusGpu);
  store_qword(kRingPhysical + 40 * sizeof(uint32_t), 1);

  auto doorbell = std::bit_cast<std::array<std::byte, sizeof(uint64_t)>>(uint64_t{64});
  ring_doorbell(0x60, doorbell);

  EXPECT_EQ(transport_.dword_at(kApiStatusPhysical), 0u);
  EXPECT_EQ(transport_.dword_at(kReadPointerPhysical), 0u);
  EXPECT_EQ(soc_.mes_engine().active_queues(), 0u);
  EXPECT_EQ(soc_.queue_registry().active_queues(), 0u);
}

TEST_F(GpuDeviceMes, RetriesUnavailableAddQueueMqdBeforeCommittingTheQueue) {
  store_dword(kRingPhysical, 0x00040021);
  store_qword(kRingPhysical + 20 * 4, kSchedulerMqdGpu);
  store_dword(kRingPhysical + 28 * 4, 3);
  store_qword(kRingPhysical + 38 * 4, kApiStatusGpu);
  store_qword(kRingPhysical + 40 * 4, 1);

  store_dword(kSchedulerMqdPhysical + 130 * 4, 1);
  store_dword(kSchedulerMqdPhysical + 136 * 4, static_cast<uint32_t>(kSchedulerRingGpu >> 8));
  store_dword(kSchedulerMqdPhysical + 137 * 4, static_cast<uint32_t>(kSchedulerRingGpu >> 40));
  store_dword(kSchedulerMqdPhysical + 139 * 4, static_cast<uint32_t>(kSchedulerReadPointerGpu));
  store_dword(kSchedulerMqdPhysical + 140 * 4,
              static_cast<uint32_t>(kSchedulerReadPointerGpu >> 32));
  store_dword(kSchedulerMqdPhysical + 143 * 4, 0x40000058);
  store_dword(kSchedulerMqdPhysical + 145 * 4, 9);

  constexpr uint64_t kMqdFirstDword = 130;
  transport_.next_read_outcome =
      std::pair{kSchedulerMqdPhysical + kMqdFirstDword * sizeof(uint32_t),
                simdojo::DmaAccessOutcome::Unavailable};
  auto mes_doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{64});
  ring_doorbell(0x60, mes_doorbell);
  EXPECT_EQ(transport_.dword_at(kApiStatusPhysical), 0u);
  EXPECT_EQ(transport_.dword_at(kReadPointerPhysical), 0u);

  device_.drain_doorbell_inbox_for_test();
  EXPECT_EQ(transport_.dword_at(kApiStatusPhysical), 1u);
  EXPECT_EQ(transport_.dword_at(kReadPointerPhysical), 64u)
      << "the transient MQD read required another guest doorbell";

  store_dword(kSchedulerRingPhysical, 0x000400b1);
  store_qword(kSchedulerRingPhysical + 2 * 4, kSchedulerApiStatusGpu);
  store_qword(kSchedulerRingPhysical + 4 * 4, 3);
  auto scheduler_doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{64});
  ring_doorbell(0x58, scheduler_doorbell);
  EXPECT_EQ(transport_.dword_at(kSchedulerApiStatusPhysical), 3u);
  EXPECT_EQ(transport_.dword_at(kSchedulerReadPointerPhysical), 64u);
}

TEST_F(GpuDeviceMes, RetriesRemoveQueueReadPointerWithoutReplayingRemovalOrCompletion) {
  store_dword(kRingPhysical, 0x00040021);
  store_qword(kRingPhysical + 20 * 4, kSchedulerMqdGpu);
  store_dword(kRingPhysical + 28 * 4, 3);
  store_qword(kRingPhysical + 38 * 4, kApiStatusGpu);
  store_qword(kRingPhysical + 40 * 4, 1);

  store_dword(kSchedulerMqdPhysical + 130 * 4, 1);
  store_dword(kSchedulerMqdPhysical + 136 * 4, static_cast<uint32_t>(kSchedulerRingGpu >> 8));
  store_dword(kSchedulerMqdPhysical + 137 * 4, static_cast<uint32_t>(kSchedulerRingGpu >> 40));
  store_dword(kSchedulerMqdPhysical + 139 * 4, static_cast<uint32_t>(kSchedulerReadPointerGpu));
  store_dword(kSchedulerMqdPhysical + 140 * 4,
              static_cast<uint32_t>(kSchedulerReadPointerGpu >> 32));
  store_dword(kSchedulerMqdPhysical + 143 * 4, 0x40000058);
  store_dword(kSchedulerMqdPhysical + 145 * 4, 9);

  auto mes_doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{64});
  ring_doorbell(0x60, mes_doorbell);
  ASSERT_EQ(transport_.dword_at(kApiStatusPhysical), 1u);
  ASSERT_EQ(transport_.dword_at(kReadPointerPhysical), 64u);

  constexpr uint64_t kRemove = kRingPhysical + 64 * 4;
  store_dword(kRemove, 0x00040031);
  store_dword(kRemove + 1 * 4, 0x58 / sizeof(uint32_t));
  store_qword(kRemove + 6 * 4, kRingFenceGpu);
  store_qword(kRemove + 8 * 4, 2);

  transport_.writes.clear();
  transport_.next_write_outcome =
      std::pair{kReadPointerPhysical, simdojo::DmaAccessOutcome::Unavailable};
  mes_doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{128});
  ring_doorbell(0x60, mes_doorbell);

  EXPECT_EQ(transport_.dword_at(kRingFencePhysical), 2u);
  EXPECT_EQ(transport_.dword_at(kReadPointerPhysical), 64u);
  EXPECT_EQ(
      std::ranges::count_if(transport_.writes,
                            [](const auto &write) { return write.first == kRingFencePhysical; }),
      1u);

  device_.drain_doorbell_inbox_for_test();
  EXPECT_EQ(transport_.dword_at(kRingFencePhysical), 2u);
  EXPECT_EQ(transport_.dword_at(kReadPointerPhysical), 128u)
      << "the transient read-pointer failure required another guest doorbell";
  EXPECT_EQ(
      std::ranges::count_if(transport_.writes,
                            [](const auto &write) { return write.first == kRingFencePhysical; }),
      1u)
      << "retry replayed an already-published completion";
}

TEST_F(GpuDeviceMes, TerminalRemoveQueueCompletionFailureDoesNotRetryOrReplayRemoval) {
  store_dword(kRingPhysical, 0x00040021);
  store_qword(kRingPhysical + 20 * 4, kSchedulerMqdGpu);
  store_dword(kRingPhysical + 28 * 4, 3);
  store_qword(kRingPhysical + 38 * 4, kApiStatusGpu);
  store_qword(kRingPhysical + 40 * 4, 1);

  store_dword(kSchedulerMqdPhysical + 130 * 4, 1);
  store_dword(kSchedulerMqdPhysical + 136 * 4, static_cast<uint32_t>(kSchedulerRingGpu >> 8));
  store_dword(kSchedulerMqdPhysical + 137 * 4, static_cast<uint32_t>(kSchedulerRingGpu >> 40));
  store_dword(kSchedulerMqdPhysical + 139 * 4, static_cast<uint32_t>(kSchedulerReadPointerGpu));
  store_dword(kSchedulerMqdPhysical + 140 * 4,
              static_cast<uint32_t>(kSchedulerReadPointerGpu >> 32));
  store_dword(kSchedulerMqdPhysical + 143 * 4, 0x40000058);
  store_dword(kSchedulerMqdPhysical + 145 * 4, 9);

  auto mes_doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{64});
  ring_doorbell(0x60, mes_doorbell);
  ASSERT_EQ(transport_.dword_at(kApiStatusPhysical), 1u);
  ASSERT_EQ(transport_.dword_at(kReadPointerPhysical), 64u);

  constexpr uint64_t kRemove = kRingPhysical + 64 * 4;
  store_dword(kRemove, 0x00040031);
  store_dword(kRemove + 1 * 4, 0x58 / sizeof(uint32_t));
  store_qword(kRemove + 6 * 4, kRingFenceGpu);
  store_qword(kRemove + 8 * 4, 2);

  transport_.writes.clear();
  transport_.next_write_outcome = std::pair{kRingFencePhysical, simdojo::DmaAccessOutcome::Faulted};
  mes_doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{128});
  ring_doorbell(0x60, mes_doorbell);

  device_.drain_doorbell_inbox_for_test();
  EXPECT_EQ(transport_.dword_at(kRingFencePhysical), 0u);
  EXPECT_EQ(transport_.dword_at(kReadPointerPhysical), 64u)
      << "a terminal completion failure was retried";

  ring_doorbell(0x60, mes_doorbell);
  EXPECT_EQ(transport_.dword_at(kRingFencePhysical), 0u);
  EXPECT_EQ(transport_.dword_at(kReadPointerPhysical), 64u)
      << "a terminal journal replayed the committed REMOVE_QUEUE semantic";

  store_dword(kSchedulerRingPhysical, 0x000400b1);
  store_qword(kSchedulerRingPhysical + 2 * 4, kSchedulerApiStatusGpu);
  store_qword(kSchedulerRingPhysical + 4 * 4, 3);
  auto scheduler_doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{64});
  ring_doorbell(0x58, scheduler_doorbell);
  EXPECT_EQ(transport_.dword_at(kSchedulerApiStatusPhysical), 0u);
  EXPECT_EQ(transport_.dword_at(kSchedulerReadPointerPhysical), 0u)
      << "REMOVE_QUEUE did not commit before its terminal completion failure";
}

TEST_F(GpuDeviceMes, ResetInvalidatesScheduledRemoveQueueRetry) {
  store_dword(kRingPhysical, 0x00040021);
  store_qword(kRingPhysical + 20 * 4, kSchedulerMqdGpu);
  store_dword(kRingPhysical + 28 * 4, 3);
  store_qword(kRingPhysical + 38 * 4, kApiStatusGpu);
  store_qword(kRingPhysical + 40 * 4, 1);

  store_dword(kSchedulerMqdPhysical + 130 * 4, 1);
  store_dword(kSchedulerMqdPhysical + 136 * 4, static_cast<uint32_t>(kSchedulerRingGpu >> 8));
  store_dword(kSchedulerMqdPhysical + 137 * 4, static_cast<uint32_t>(kSchedulerRingGpu >> 40));
  store_dword(kSchedulerMqdPhysical + 139 * 4, static_cast<uint32_t>(kSchedulerReadPointerGpu));
  store_dword(kSchedulerMqdPhysical + 140 * 4,
              static_cast<uint32_t>(kSchedulerReadPointerGpu >> 32));
  store_dword(kSchedulerMqdPhysical + 143 * 4, 0x40000058);
  store_dword(kSchedulerMqdPhysical + 145 * 4, 9);

  auto mes_doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{64});
  ring_doorbell(0x60, mes_doorbell);
  ASSERT_EQ(transport_.dword_at(kApiStatusPhysical), 1u);
  ASSERT_EQ(transport_.dword_at(kReadPointerPhysical), 64u);

  constexpr uint64_t kRemove = kRingPhysical + 64 * 4;
  store_dword(kRemove, 0x00040031);
  store_dword(kRemove + 1 * 4, 0x58 / sizeof(uint32_t));
  store_qword(kRemove + 6 * 4, kRingFenceGpu);
  store_qword(kRemove + 8 * 4, 2);

  transport_.next_write_outcome =
      std::pair{kRingFencePhysical, simdojo::DmaAccessOutcome::Unavailable};
  mes_doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{128});
  ring_doorbell(0x60, mes_doorbell);
  ASSERT_EQ(transport_.dword_at(kRingFencePhysical), 0u);
  ASSERT_EQ(transport_.dword_at(kReadPointerPhysical), 64u);

  device_.reset(simdojo::ResetKind::FunctionLevel);
  device_.drain_doorbell_inbox_for_test();
  EXPECT_EQ(transport_.dword_at(kRingFencePhysical), 0u);
  EXPECT_EQ(transport_.dword_at(kReadPointerPhysical), 64u)
      << "a retry scheduled before reset crossed the reset epoch";
}

TEST_F(GpuDeviceMes, MapsAndExecutesTheSchedulerQueueCreatedByAddQueue) {
  // ADD_QUEUE maps the scheduler queue from its MQD. Its status follows the
  // queue metadata at dword 38.
  store_dword(kRingPhysical, 0x00040021);
  store_qword(kRingPhysical + 20 * 4, kSchedulerMqdGpu);
  store_dword(kRingPhysical + 28 * 4, 3);
  store_qword(kRingPhysical + 38 * 4, kApiStatusGpu);
  store_qword(kRingPhysical + 40 * 4, 1);

  store_dword(kSchedulerMqdPhysical + 130 * 4, 1);
  store_dword(kSchedulerMqdPhysical + 136 * 4, static_cast<uint32_t>(kSchedulerRingGpu >> 8));
  store_dword(kSchedulerMqdPhysical + 137 * 4, static_cast<uint32_t>(kSchedulerRingGpu >> 40));
  store_dword(kSchedulerMqdPhysical + 139 * 4, static_cast<uint32_t>(kSchedulerReadPointerGpu));
  store_dword(kSchedulerMqdPhysical + 140 * 4,
              static_cast<uint32_t>(kSchedulerReadPointerGpu >> 32));
  store_dword(kSchedulerMqdPhysical + 143 * 4, 0x40000058);
  store_dword(kSchedulerMqdPhysical + 145 * 4, 9);

  constexpr uint64_t kQuery = kRingPhysical + 64 * 4;
  store_dword(kQuery, 0x000400b1);
  store_qword(kQuery + 2 * 4, kRingFenceGpu);
  store_qword(kQuery + 4 * 4, 9);

  // Queue the scheduler work before ADD_QUEUE has been observed. FIFO delivery
  // is load-bearing: reversing or coalescing these notifications would ring an
  // unmapped scheduler queue and lose its work.
  store_dword(kSchedulerRingPhysical, 0x00040001);
  store_qword(kSchedulerRingPhysical + 50 * 4, kSchedulerApiStatusGpu);
  store_qword(kSchedulerRingPhysical + 52 * 4, 1);

  constexpr uint64_t kSchedulerQuery = kSchedulerRingPhysical + 64 * 4;
  store_dword(kSchedulerQuery, 0x000400b1);
  store_qword(kSchedulerQuery + 2 * 4, kSchedulerRingFenceGpu);
  store_qword(kSchedulerQuery + 4 * 4, 10);

  auto doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{128});
  ASSERT_EQ(device_.bar_access(rocjitsu::GpuPciDevice::kDoorbellBar, doorbell, 0x60,
                               /*write=*/true),
            8);
  ASSERT_EQ(device_.bar_access(rocjitsu::GpuPciDevice::kDoorbellBar, doorbell, 0x58,
                               /*write=*/true),
            8);

  EXPECT_EQ(transport_.dword_at(kApiStatusPhysical), 0u);
  EXPECT_EQ(transport_.dword_at(kSchedulerApiStatusPhysical), 0u);

  device_.drain_doorbell_inbox_for_test();

  EXPECT_EQ(transport_.dword_at(kApiStatusPhysical), 1u);
  EXPECT_EQ(transport_.dword_at(kRingFencePhysical), 9u);
  EXPECT_EQ(transport_.dword_at(kReadPointerPhysical), 128u);
  EXPECT_EQ(transport_.dword_at(kSchedulerApiStatusPhysical), 1u);
  EXPECT_EQ(transport_.dword_at(kSchedulerRingFencePhysical), 10u);
  EXPECT_EQ(transport_.dword_at(kSchedulerReadPointerPhysical), 128u);
  EXPECT_EQ(soc_.mes_engine().active_queues(), 1u)
      << "MES-created queue state belongs to the SoC engine";
}

TEST_F(GpuDeviceMes, ExecutesTheFirmwareFreePm4ComputeRing) {
  constexpr uint64_t kGcControlBase = 0xa000;
  constexpr uint64_t kScratchRegister = (kGcControlBase + 0x2040) * 4;
  constexpr uint64_t kComputeDoorbell = 0x900;
  constexpr uint64_t kComputeRingGpu = 0x10000;
  constexpr uint64_t kComputeReadPointerGpu = 0x11ff0;
  constexpr uint64_t kProcessPageTable = 0x30000;
  constexpr uint64_t kProcessPdb2 = 0x31000;
  constexpr uint64_t kProcessPdb1 = 0x32000;
  constexpr uint64_t kProcessPdb0 = 0x33000;
  constexpr uint64_t kProcessPtb = 0x34000;
  constexpr uint64_t kComputeRingPhysical = 0x290000;
  constexpr uint64_t kComputeSecondPagePhysical = 0x2a0000;

  write_register_at(kScratchRegister, 0xcafedead);

  // GC 12.1 uses five 9-bit page-table levels. The test queue spans virtual
  // pages 0x10 and 0x11, both backed by guest system memory.
  store_vram_qword(kProcessPageTable, kProcessPdb2 | 1);
  store_vram_qword(kProcessPdb2, kProcessPdb1 | 1);
  store_vram_qword(kProcessPdb1, kProcessPdb0 | 1);
  store_vram_qword(kProcessPdb0, kProcessPtb | 1);
  store_vram_qword(kProcessPtb + 0x10 * sizeof(uint64_t), kComputeRingPhysical | 0x63);
  store_vram_qword(kProcessPtb + 0x11 * sizeof(uint64_t), kComputeSecondPagePhysical | 0x63);

  // ADD_QUEUE classifies this MQD as a compute queue and maps its doorbell.
  store_dword(kRingPhysical, 0x00040021);
  store_dword(kRingPhysical + 1 * 4, 3);
  store_qword(kRingPhysical + 2 * 4, kProcessPageTable);
  store_qword(kRingPhysical + 20 * 4, kSchedulerMqdGpu);
  store_dword(kRingPhysical + 28 * 4, 1);
  store_qword(kRingPhysical + 38 * 4, kApiStatusGpu);
  store_qword(kRingPhysical + 40 * 4, 1);

  store_dword(kSchedulerMqdPhysical + 130 * 4, 0);
  store_dword(kSchedulerMqdPhysical + 136 * 4, static_cast<uint32_t>(kComputeRingGpu >> 8));
  store_dword(kSchedulerMqdPhysical + 137 * 4, static_cast<uint32_t>(kComputeRingGpu >> 40));
  store_dword(kSchedulerMqdPhysical + 139 * 4, static_cast<uint32_t>(kComputeReadPointerGpu));
  store_dword(kSchedulerMqdPhysical + 140 * 4, static_cast<uint32_t>(kComputeReadPointerGpu >> 32));
  store_dword(kSchedulerMqdPhysical + 143 * 4, kComputeDoorbell);
  store_dword(kSchedulerMqdPhysical + 145 * 4, 9);

  constexpr uint64_t kQuery = kRingPhysical + 64 * 4;
  store_dword(kQuery, 0x000400b1);
  store_qword(kQuery + 2 * 4, kRingFenceGpu);
  store_qword(kQuery + 4 * 4, 11);

  auto doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{128});
  ring_doorbell(0x60, doorbell);

  ASSERT_EQ(transport_.dword_at(kApiStatusPhysical), 1u);
  ASSERT_EQ(transport_.dword_at(kRingFencePhysical), 11u);

  // The driver self-test emits SET_UCONFIG_REG followed by special one-dword
  // NOPs until the compute ring reaches its 256-dword alignment.
  store_dword(kComputeRingPhysical, 0xc0017900);
  store_dword(kComputeRingPhysical + 4, 0x40);
  store_dword(kComputeRingPhysical + 8, 0xdeadbeef);
  for (uint32_t dword = 3; dword < 256; ++dword) {
    store_dword(kComputeRingPhysical + dword * sizeof(uint32_t), 0xffff1000);
  }

  auto split_doorbell = std::bit_cast<std::array<std::byte, 4>>(uint32_t{256});
  ring_doorbell(kComputeDoorbell, split_doorbell);

  EXPECT_EQ(read_register_at(kScratchRegister), 0xdeadbeefu);
  EXPECT_EQ(transport_.dword_at(kComputeSecondPagePhysical + 0xff0), 256u);
}

TEST_F(GpuDeviceMes, RetriesComputeRingFetchWithoutReplayingCompletedPackets) {
  constexpr uint64_t kGcControlBase = 0xa000;
  constexpr uint64_t kScratchRegister = (kGcControlBase + 0x2040) * 4;
  constexpr uint64_t kComputeDoorbell = 0x900;
  constexpr uint64_t kComputeRingGpu = 0x10000;
  constexpr uint64_t kComputeReadPointerGpu = 0x11ff0;
  constexpr uint64_t kProcessPageTable = 0x30000;
  constexpr uint64_t kProcessPdb2 = 0x31000;
  constexpr uint64_t kProcessPdb1 = 0x32000;
  constexpr uint64_t kProcessPdb0 = 0x33000;
  constexpr uint64_t kProcessPtb = 0x34000;
  constexpr uint64_t kComputeRingPhysical = 0x290000;
  constexpr uint64_t kComputeSecondPagePhysical = 0x2a0000;

  write_register_at(kScratchRegister, 0xcafedead);
  store_vram_qword(kProcessPageTable, kProcessPdb2 | 1);
  store_vram_qword(kProcessPdb2, kProcessPdb1 | 1);
  store_vram_qword(kProcessPdb1, kProcessPdb0 | 1);
  store_vram_qword(kProcessPdb0, kProcessPtb | 1);
  store_vram_qword(kProcessPtb + 0x10 * sizeof(uint64_t), kComputeRingPhysical | 0x63);
  store_vram_qword(kProcessPtb + 0x11 * sizeof(uint64_t), kComputeSecondPagePhysical | 0x63);

  store_dword(kRingPhysical, 0x00040021);
  store_dword(kRingPhysical + 1 * 4, 3);
  store_qword(kRingPhysical + 2 * 4, kProcessPageTable);
  store_qword(kRingPhysical + 20 * 4, kSchedulerMqdGpu);
  store_dword(kRingPhysical + 28 * 4, 1);
  store_qword(kRingPhysical + 38 * 4, kApiStatusGpu);
  store_qword(kRingPhysical + 40 * 4, 1);
  store_dword(kSchedulerMqdPhysical + 130 * 4, 0);
  store_dword(kSchedulerMqdPhysical + 136 * 4, static_cast<uint32_t>(kComputeRingGpu >> 8));
  store_dword(kSchedulerMqdPhysical + 137 * 4, static_cast<uint32_t>(kComputeRingGpu >> 40));
  store_dword(kSchedulerMqdPhysical + 139 * 4, static_cast<uint32_t>(kComputeReadPointerGpu));
  store_dword(kSchedulerMqdPhysical + 140 * 4, static_cast<uint32_t>(kComputeReadPointerGpu >> 32));
  store_dword(kSchedulerMqdPhysical + 143 * 4, kComputeDoorbell);
  store_dword(kSchedulerMqdPhysical + 145 * 4, 9);
  auto mes_doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{64});
  ring_doorbell(0x60, mes_doorbell);
  ASSERT_EQ(transport_.dword_at(kApiStatusPhysical), 1u);

  store_dword(kComputeRingPhysical, 0xc0017900);
  store_dword(kComputeRingPhysical + 4, 0x40);
  store_dword(kComputeRingPhysical + 8, 0xdeadbeef);
  for (uint32_t dword = 3; dword < 256; ++dword)
    store_dword(kComputeRingPhysical + dword * sizeof(uint32_t), 0xffff1000);

  // Let SET_UCONFIG_REG commit, then make the following NOP fetch unavailable.
  // The retry must resume at dword 3 instead of applying the register write twice.
  transport_.next_read_outcome = std::pair{kComputeRingPhysical + 3 * sizeof(uint32_t),
                                           simdojo::DmaAccessOutcome::Unavailable};
  auto compute_doorbell = std::bit_cast<std::array<std::byte, 4>>(uint32_t{256});
  ring_doorbell(kComputeDoorbell, compute_doorbell);
  EXPECT_EQ(read_register_at(kScratchRegister), 0xdeadbeefu);
  EXPECT_EQ(transport_.dword_at(kComputeSecondPagePhysical + 0xff0), 3u);

  constexpr uint32_t kAfterFirstPacketSentinel = 0x13579bdf;
  write_register_at(kScratchRegister, kAfterFirstPacketSentinel);
  device_.drain_doorbell_inbox_for_test();
  EXPECT_EQ(read_register_at(kScratchRegister), kAfterFirstPacketSentinel)
      << "retry replayed an already-executed PM4 register write";
  EXPECT_EQ(transport_.dword_at(kComputeSecondPagePhysical + 0xff0), 256u);
}

TEST_F(GpuDeviceMes, RetriesComputeReadPointerWithoutReplayingRegisterWrites) {
  constexpr uint64_t kGcControlBase = 0xa000;
  constexpr uint64_t kScratchRegister = (kGcControlBase + 0x2040) * 4;
  constexpr uint64_t kComputeDoorbell = 0x900;
  constexpr uint64_t kComputeRingGpu = 0x10000;
  constexpr uint64_t kComputeReadPointerGpu = 0x11ff0;
  constexpr uint64_t kProcessPageTable = 0x30000;
  constexpr uint64_t kProcessPdb2 = 0x31000;
  constexpr uint64_t kProcessPdb1 = 0x32000;
  constexpr uint64_t kProcessPdb0 = 0x33000;
  constexpr uint64_t kProcessPtb = 0x34000;
  constexpr uint64_t kComputeRingPhysical = 0x290000;
  constexpr uint64_t kComputeSecondPagePhysical = 0x2a0000;
  constexpr uint32_t kAfterExecutionSentinel = 0x13579bdf;

  store_vram_qword(kProcessPageTable, kProcessPdb2 | 1);
  store_vram_qword(kProcessPdb2, kProcessPdb1 | 1);
  store_vram_qword(kProcessPdb1, kProcessPdb0 | 1);
  store_vram_qword(kProcessPdb0, kProcessPtb | 1);
  store_vram_qword(kProcessPtb + 0x10 * sizeof(uint64_t), kComputeRingPhysical | 0x63);
  store_vram_qword(kProcessPtb + 0x11 * sizeof(uint64_t), kComputeSecondPagePhysical | 0x63);

  store_dword(kRingPhysical, 0x00040021);
  store_dword(kRingPhysical + 1 * 4, 3);
  store_qword(kRingPhysical + 2 * 4, kProcessPageTable);
  store_qword(kRingPhysical + 20 * 4, kSchedulerMqdGpu);
  store_dword(kRingPhysical + 28 * 4, 1);
  store_qword(kRingPhysical + 38 * 4, kApiStatusGpu);
  store_qword(kRingPhysical + 40 * 4, 1);
  store_dword(kSchedulerMqdPhysical + 130 * 4, 0);
  store_dword(kSchedulerMqdPhysical + 136 * 4, static_cast<uint32_t>(kComputeRingGpu >> 8));
  store_dword(kSchedulerMqdPhysical + 137 * 4, static_cast<uint32_t>(kComputeRingGpu >> 40));
  store_dword(kSchedulerMqdPhysical + 139 * 4, static_cast<uint32_t>(kComputeReadPointerGpu));
  store_dword(kSchedulerMqdPhysical + 140 * 4, static_cast<uint32_t>(kComputeReadPointerGpu >> 32));
  store_dword(kSchedulerMqdPhysical + 143 * 4, kComputeDoorbell);
  store_dword(kSchedulerMqdPhysical + 145 * 4, 9);
  auto mes_doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{64});
  ring_doorbell(0x60, mes_doorbell);
  ASSERT_EQ(transport_.dword_at(kApiStatusPhysical), 1u);

  store_dword(kComputeRingPhysical, 0xc0017900);
  store_dword(kComputeRingPhysical + 4, 0x40);
  store_dword(kComputeRingPhysical + 8, 0xdeadbeef);
  for (uint32_t dword = 3; dword < 256; ++dword)
    store_dword(kComputeRingPhysical + dword * sizeof(uint32_t), 0xffff1000);

  transport_.next_write_outcome =
      std::pair{kComputeSecondPagePhysical + 0xff0, simdojo::DmaAccessOutcome::Unavailable};
  auto compute_doorbell = std::bit_cast<std::array<std::byte, 4>>(uint32_t{256});
  ring_doorbell(kComputeDoorbell, compute_doorbell);
  ASSERT_EQ(read_register_at(kScratchRegister), 0xdeadbeefu);
  ASSERT_EQ(transport_.dword_at(kComputeSecondPagePhysical + 0xff0), 0u);

  write_register_at(kScratchRegister, kAfterExecutionSentinel);
  device_.drain_doorbell_inbox_for_test();
  EXPECT_EQ(read_register_at(kScratchRegister), kAfterExecutionSentinel)
      << "retry replayed an already-executed PM4 register write";
  EXPECT_EQ(transport_.dword_at(kComputeSecondPagePhysical + 0xff0), 256u);
}

TEST_F(GpuDeviceMes, DefersComputeQueueRemovalUntilCommittedProgressIsPublished) {
  constexpr uint64_t kGcControlBase = 0xa000;
  constexpr uint64_t kScratchRegister = (kGcControlBase + 0x2040) * 4;
  constexpr uint64_t kComputeDoorbell = 0x900;
  constexpr uint64_t kComputeRingGpu = 0x10000;
  constexpr uint64_t kComputeReadPointerGpu = 0x11ff0;
  constexpr uint64_t kProcessPageTable = 0x30000;
  constexpr uint64_t kProcessPdb2 = 0x31000;
  constexpr uint64_t kProcessPdb1 = 0x32000;
  constexpr uint64_t kProcessPdb0 = 0x33000;
  constexpr uint64_t kProcessPtb = 0x34000;
  constexpr uint64_t kComputeRingPhysical = 0x290000;
  constexpr uint64_t kComputeSecondPagePhysical = 0x2a0000;
  constexpr uint64_t kComputeReadPointerPhysical = kComputeSecondPagePhysical + 0xff0;
  constexpr uint32_t kAfterExecutionSentinel = 0x13579bdf;

  store_vram_qword(kProcessPageTable, kProcessPdb2 | 1);
  store_vram_qword(kProcessPdb2, kProcessPdb1 | 1);
  store_vram_qword(kProcessPdb1, kProcessPdb0 | 1);
  store_vram_qword(kProcessPdb0, kProcessPtb | 1);
  store_vram_qword(kProcessPtb + 0x10 * sizeof(uint64_t), kComputeRingPhysical | 0x63);
  store_vram_qword(kProcessPtb + 0x11 * sizeof(uint64_t), kComputeSecondPagePhysical | 0x63);

  store_dword(kRingPhysical, 0x00040021);
  store_dword(kRingPhysical + 1 * 4, 3);
  store_qword(kRingPhysical + 2 * 4, kProcessPageTable);
  store_qword(kRingPhysical + 20 * 4, kSchedulerMqdGpu);
  store_dword(kRingPhysical + 28 * 4, 1);
  store_qword(kRingPhysical + 38 * 4, kApiStatusGpu);
  store_qword(kRingPhysical + 40 * 4, 1);
  store_dword(kSchedulerMqdPhysical + 130 * 4, 0);
  store_dword(kSchedulerMqdPhysical + 136 * 4, static_cast<uint32_t>(kComputeRingGpu >> 8));
  store_dword(kSchedulerMqdPhysical + 137 * 4, static_cast<uint32_t>(kComputeRingGpu >> 40));
  store_dword(kSchedulerMqdPhysical + 139 * 4, static_cast<uint32_t>(kComputeReadPointerGpu));
  store_dword(kSchedulerMqdPhysical + 140 * 4, static_cast<uint32_t>(kComputeReadPointerGpu >> 32));
  store_dword(kSchedulerMqdPhysical + 143 * 4, kComputeDoorbell);
  store_dword(kSchedulerMqdPhysical + 145 * 4, 9);
  auto mes_doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{64});
  ring_doorbell(0x60, mes_doorbell);
  ASSERT_EQ(transport_.dword_at(kApiStatusPhysical), 1u);
  ASSERT_EQ(soc_.mes_engine().active_queues(), 1u);

  store_dword(kComputeRingPhysical, 0xc0017900);
  store_dword(kComputeRingPhysical + 4, 0x40);
  store_dword(kComputeRingPhysical + 8, 0xdeadbeef);
  transport_.next_write_outcome =
      std::pair{kComputeReadPointerPhysical, simdojo::DmaAccessOutcome::Unavailable};
  auto compute_doorbell = std::bit_cast<std::array<std::byte, 4>>(uint32_t{3});
  ring_doorbell(kComputeDoorbell, compute_doorbell);
  ASSERT_EQ(read_register_at(kScratchRegister), 0xdeadbeefu);
  ASSERT_EQ(transport_.dword_at(kComputeReadPointerPhysical), 0u);

  write_register_at(kScratchRegister, kAfterExecutionSentinel);
  constexpr uint64_t kRemove = kRingPhysical + 64 * sizeof(uint32_t);
  store_dword(kRemove, 0x00040031);
  store_dword(kRemove + sizeof(uint32_t), kComputeDoorbell / sizeof(uint32_t));
  store_qword(kRemove + 6 * sizeof(uint32_t), kRingFenceGpu);
  store_qword(kRemove + 8 * sizeof(uint32_t), 2);

  // Keep the PM4 cursor publication blocked while REMOVE_QUEUE is attempted.
  // The registry must leave the same queue handle live until that publication
  // completes, rather than destroying the only state that prevents replay.
  transport_.next_write_outcome =
      std::pair{kComputeReadPointerPhysical, simdojo::DmaAccessOutcome::Unavailable};
  mes_doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{128});
  ring_doorbell(0x60, mes_doorbell);
  EXPECT_EQ(soc_.mes_engine().active_queues(), 1u);
  EXPECT_EQ(transport_.dword_at(kReadPointerPhysical), 64u);
  EXPECT_EQ(transport_.dword_at(kRingFencePhysical), 0u);
  EXPECT_EQ(transport_.dword_at(kComputeReadPointerPhysical), 0u);
  EXPECT_EQ(read_register_at(kScratchRegister), kAfterExecutionSentinel)
      << "queue removal replayed a committed PM4 register write";

  device_.drain_doorbell_inbox_for_test();
  EXPECT_EQ(transport_.dword_at(kComputeReadPointerPhysical), 3u);
  EXPECT_EQ(transport_.dword_at(kReadPointerPhysical), 128u);
  EXPECT_EQ(transport_.dword_at(kRingFencePhysical), 2u);
  EXPECT_EQ(soc_.mes_engine().active_queues(), 0u);
  EXPECT_EQ(read_register_at(kScratchRegister), kAfterExecutionSentinel)
      << "publishing and removing the queue replayed its committed packet";
}

TEST_F(GpuDeviceMes, PermanentPm4CursorFaultMakesRemoveQueueTerminalUntilReset) {
  constexpr uint64_t kGcControlBase = 0xa000;
  constexpr uint64_t kScratchRegister = (kGcControlBase + 0x2040) * sizeof(uint32_t);
  constexpr uint32_t kAfterExecutionSentinel = 0x13579bdf;

  submit_default_compute_queue();

  store_dword(kDefaultComputeRingPhysical, 0xc0017900);
  store_dword(kDefaultComputeRingPhysical + sizeof(uint32_t), 0x40);
  store_dword(kDefaultComputeRingPhysical + 2 * sizeof(uint32_t), 0xdeadbeef);
  transport_.next_write_outcome =
      std::pair{kDefaultComputeReadPointerPhysical, simdojo::DmaAccessOutcome::Faulted};
  auto compute_doorbell = std::bit_cast<std::array<std::byte, sizeof(uint32_t)>>(uint32_t{3});
  ring_doorbell(kDefaultComputeDoorbell, compute_doorbell);
  ASSERT_EQ(read_register_at(kScratchRegister), 0xdeadbeefu);
  ASSERT_EQ(transport_.dword_at(kDefaultComputeReadPointerPhysical), 0u);

  write_register_at(kScratchRegister, kAfterExecutionSentinel);
  constexpr uint64_t kRemove = kRingPhysical + 64 * sizeof(uint32_t);
  store_dword(kRemove, 0x00040031);
  store_dword(kRemove + sizeof(uint32_t), kDefaultComputeDoorbell / sizeof(uint32_t));
  store_qword(kRemove + 6 * sizeof(uint32_t), kRingFenceGpu);
  store_qword(kRemove + 8 * sizeof(uint32_t), 2);

  auto mes_doorbell = std::bit_cast<std::array<std::byte, sizeof(uint64_t)>>(uint64_t{128});
  ring_doorbell(0x60, mes_doorbell);
  EXPECT_EQ(transport_.dword_at(kReadPointerPhysical), 64u);
  EXPECT_EQ(transport_.dword_at(kRingFencePhysical), 0u);
  EXPECT_EQ(soc_.mes_engine().active_queues(), 1u);
  EXPECT_EQ(soc_.queue_registry().active_queues(), 1u);
  EXPECT_EQ(command_processor_.registered_pm4_queue_count_for_test(), 1u);

  device_.drain_doorbell_inbox_for_test();
  EXPECT_EQ(transport_.dword_at(kDefaultComputeReadPointerPhysical), 0u)
      << "a permanently faulted PM4 cursor publication was retried";
  EXPECT_EQ(read_register_at(kScratchRegister), kAfterExecutionSentinel)
      << "failed graceful removal replayed the committed PM4 packet";

  device_.reset(simdojo::ResetKind::FunctionLevel);
  EXPECT_EQ(soc_.mes_engine().active_queues(), 0u);
  EXPECT_EQ(soc_.queue_registry().active_queues(), 0u);
  EXPECT_EQ(command_processor_.registered_pm4_queue_count_for_test(), 0u);
}

TEST_F(GpuDeviceMes, ResetForceCancelsPm4QueueWithPendingCursorPublication) {
  constexpr uint64_t kGcControlBase = 0xa000;
  constexpr uint64_t kScratchRegister = (kGcControlBase + 0x2040) * sizeof(uint32_t);
  constexpr uint32_t kAfterResetSentinel = 0x13579bdf;

  submit_default_compute_queue();

  store_dword(kDefaultComputeRingPhysical, 0xc0017900);
  store_dword(kDefaultComputeRingPhysical + sizeof(uint32_t), 0x40);
  store_dword(kDefaultComputeRingPhysical + 2 * sizeof(uint32_t), 0xdeadbeef);
  transport_.next_write_outcome =
      std::pair{kDefaultComputeReadPointerPhysical, simdojo::DmaAccessOutcome::Unavailable};
  auto compute_doorbell = std::bit_cast<std::array<std::byte, sizeof(uint32_t)>>(uint32_t{3});
  ring_doorbell(kDefaultComputeDoorbell, compute_doorbell);
  ASSERT_EQ(read_register_at(kScratchRegister), 0xdeadbeefu);
  ASSERT_EQ(transport_.dword_at(kDefaultComputeReadPointerPhysical), 0u);

  device_.reset(simdojo::ResetKind::FunctionLevel);
  EXPECT_EQ(soc_.mes_engine().active_queues(), 0u);
  EXPECT_EQ(soc_.queue_registry().active_queues(), 0u);
  EXPECT_EQ(command_processor_.registered_pm4_queue_count_for_test(), 0u);

  write_register_at(kScratchRegister, kAfterResetSentinel);
  device_.drain_doorbell_inbox_for_test();
  EXPECT_EQ(transport_.dword_at(kDefaultComputeReadPointerPhysical), 0u)
      << "reset published a cursor that force cancellation discarded";
  EXPECT_EQ(read_register_at(kScratchRegister), kAfterResetSentinel)
      << "a scheduled PM4 retry invoked a frontend callback after reset";

  configure_gart();
  submit_default_compute_queue();
  EXPECT_EQ(soc_.mes_engine().active_queues(), 1u);
  EXPECT_EQ(soc_.queue_registry().active_queues(), 1u);
  EXPECT_EQ(command_processor_.registered_pm4_queue_count_for_test(), 1u);
}

TEST_F(GpuDeviceMes, ShutdownForceCancelsPm4StateBeforeDetachingFrontendCallbacks) {
  constexpr uint64_t kGcControlBase = 0xa000;
  constexpr uint64_t kScratchRegister = (kGcControlBase + 0x2040) * sizeof(uint32_t);
  constexpr uint32_t kAfterExecutionSentinel = 0x13579bdf;

  submit_default_compute_queue();

  store_dword(kDefaultComputeRingPhysical, 0xc0017900);
  store_dword(kDefaultComputeRingPhysical + sizeof(uint32_t), 0x40);
  store_dword(kDefaultComputeRingPhysical + 2 * sizeof(uint32_t), 0xdeadbeef);
  transport_.next_write_outcome =
      std::pair{kDefaultComputeReadPointerPhysical, simdojo::DmaAccessOutcome::Unavailable};
  auto compute_doorbell = std::bit_cast<std::array<std::byte, sizeof(uint32_t)>>(uint32_t{3});
  ring_doorbell(kDefaultComputeDoorbell, compute_doorbell);
  ASSERT_EQ(read_register_at(kScratchRegister), 0xdeadbeefu);
  ASSERT_EQ(transport_.dword_at(kDefaultComputeReadPointerPhysical), 0u);

  write_register_at(kScratchRegister, kAfterExecutionSentinel);
  ASSERT_TRUE(device_.shutdown_frontend());
  EXPECT_EQ(soc_.mes_engine().active_queues(), 0u);
  EXPECT_EQ(soc_.queue_registry().active_queues(), 0u);
  EXPECT_EQ(command_processor_.registered_pm4_queue_count_for_test(), 0u);

  device_.drain_doorbell_inbox_for_test();
  EXPECT_EQ(transport_.dword_at(kDefaultComputeReadPointerPhysical), 0u)
      << "shutdown published a cursor that force cancellation discarded";
  EXPECT_EQ(read_register_at(kScratchRegister), kAfterExecutionSentinel)
      << "a scheduled PM4 retry invoked a frontend callback after shutdown";

  rocjitsu::GpuPciDevice replacement("replacement", configured_spec(), &trace_, &soc_);
  ASSERT_TRUE(replacement.usable());
  ASSERT_TRUE(replacement.attach_transport(&attached_));
  EXPECT_TRUE(replacement.detach_transport(&attached_));
}

TEST_F(GpuDeviceMes, SdmaToComputeSameDoorbellReplacesThePriorBinding) {
  constexpr uint64_t kSdmaDoorbell = 0x4808;
  constexpr uint64_t kSdmaRingGpu = 0x20000;
  constexpr uint64_t kSdmaReadPointerGpu = 0x21000;
  constexpr uint64_t kSdmaWritePointerGpu = 0x22000;
  constexpr uint64_t kSdmaDestinationGpu = 0x23000;
  constexpr uint64_t kComputeRingGpu = 0x24000;
  constexpr uint64_t kComputeReadPointerGpu = 0x25000;
  constexpr uint64_t kGcControlBase = 0xa000;
  constexpr uint64_t kScratchRegister = (kGcControlBase + 0x2040) * 4;
  constexpr uint64_t kProcessPageTable = 0x30000;
  constexpr uint64_t kProcessPdb2 = 0x31000;
  constexpr uint64_t kProcessPdb1 = 0x32000;
  constexpr uint64_t kProcessPdb0 = 0x33000;
  constexpr uint64_t kProcessPtb = 0x34000;
  constexpr uint64_t kSdmaRingPhysical = 0x2b0000;
  constexpr uint64_t kSdmaReadPointerPhysical = 0x2c0000;
  constexpr uint64_t kSdmaWritePointerPhysical = 0x2d0000;
  constexpr uint64_t kSdmaDestinationPhysical = 0x2e0000;
  constexpr uint64_t kComputeRingPhysical = 0x2f0000;
  constexpr uint64_t kComputeReadPointerPhysical = 0x300000;

  store_vram_qword(kProcessPageTable, kProcessPdb2 | 1);
  store_vram_qword(kProcessPdb2, kProcessPdb1 | 1);
  store_vram_qword(kProcessPdb1, kProcessPdb0 | 1);
  store_vram_qword(kProcessPdb0, kProcessPtb | 1);
  for (const auto [page, physical] :
       {std::pair{0x20u, kSdmaRingPhysical}, std::pair{0x21u, kSdmaReadPointerPhysical},
        std::pair{0x22u, kSdmaWritePointerPhysical}, std::pair{0x23u, kSdmaDestinationPhysical},
        std::pair{0x24u, kComputeRingPhysical}, std::pair{0x25u, kComputeReadPointerPhysical}}) {
    store_vram_qword(kProcessPtb + page * sizeof(uint64_t), physical | 0x63);
  }

  // The v12.1 SDMA MQD starts at dword zero. MES supplies the byte doorbell,
  // write-pointer address, and dword ring size separately in ADD_QUEUE.
  store_dword(kRingPhysical, 0x00040021);
  store_dword(kRingPhysical + 1 * 4, 3);
  store_qword(kRingPhysical + 2 * 4, kProcessPageTable);
  store_dword(kRingPhysical + 18 * 4, kSdmaDoorbell / sizeof(uint32_t));
  store_qword(kRingPhysical + 20 * 4, kSchedulerMqdGpu);
  store_qword(kRingPhysical + 22 * 4, kSdmaWritePointerGpu);
  store_dword(kRingPhysical + 28 * 4, 2);
  // KFD converts queue_size from bytes to dwords before MES sees it. This
  // five-dword ring wraps after byte offset 20 because SDMA pointers are bytes.
  store_dword(kRingPhysical + 30 * 4, 5);
  store_qword(kRingPhysical + 38 * 4, kApiStatusGpu);
  store_qword(kRingPhysical + 40 * 4, 1);

  store_dword(kSchedulerMqdPhysical + 1 * 4, static_cast<uint32_t>(kSdmaRingGpu >> 8));
  store_dword(kSchedulerMqdPhysical + 2 * 4, static_cast<uint32_t>(kSdmaRingGpu >> 40));
  store_dword(kSchedulerMqdPhysical + 7 * 4, static_cast<uint32_t>(kSdmaReadPointerGpu));
  store_dword(kSchedulerMqdPhysical + 8 * 4, static_cast<uint32_t>(kSdmaReadPointerGpu >> 32));
  store_dword(kSchedulerMqdPhysical + 126 * 4, 1);
  store_dword(kSchedulerMqdPhysical + 127 * 4, 7);

  constexpr uint64_t kQuery = kRingPhysical + 64 * 4;
  store_dword(kQuery, 0x000400b1);
  store_qword(kQuery + 2 * 4, kRingFenceGpu);
  store_qword(kQuery + 4 * 4, 12);

  auto mes_doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{128});
  ring_doorbell(0x60, mes_doorbell);
  ASSERT_EQ(transport_.dword_at(kApiStatusPhysical), 1u);
  ASSERT_EQ(transport_.dword_at(kRingFencePhysical), 12u);
  ASSERT_EQ(soc_.queue_registry().active_queues(), 1u);

  store_dword(kSdmaRingPhysical + 0 * 4, 2);
  store_qword(kSdmaRingPhysical + 1 * 4, kSdmaDestinationGpu);
  store_dword(kSdmaRingPhysical + 3 * 4, 0);
  store_dword(kSdmaRingPhysical + 4 * 4, 0xdeadbeef);

  auto sdma_doorbell = std::bit_cast<std::array<std::byte, 4>>(uint32_t{20});
  ring_doorbell(kSdmaDoorbell, sdma_doorbell);

  EXPECT_EQ(transport_.dword_at(kSdmaDestinationPhysical), 0xdeadbeefu);
  EXPECT_EQ(transport_.dword_at(kSdmaReadPointerPhysical), 20u);

  store_dword(kSdmaRingPhysical + 4 * 4, 0xcafebabe);
  sdma_doorbell = std::bit_cast<std::array<std::byte, 4>>(uint32_t{40});
  ring_doorbell(kSdmaDoorbell, sdma_doorbell);
  EXPECT_EQ(transport_.dword_at(kSdmaDestinationPhysical), 0xcafebabeu)
      << "the dword-sized ring did not wrap at its declared byte boundary";
  EXPECT_EQ(transport_.dword_at(kSdmaReadPointerPhysical), 40u);

  const std::optional<rocjitsu::amdgpu::AddressSpaceHandle> original_address_space =
      soc_.gpu_vm().find_vmid(3);
  ASSERT_TRUE(original_address_space);
  const uint64_t original_epoch = soc_.gpu_vm().lookup(*original_address_space)->translation_epoch;

  // ADD_QUEUE does not own an already-live PASID's root transition. Reject a
  // replacement that tries to smuggle one in, and leave the old queue and its
  // translation generation fully operational.
  constexpr uint64_t kRejectedAdd = kRingPhysical + 128 * sizeof(uint32_t);
  store_dword(kRejectedAdd, 0x00040021);
  store_dword(kRejectedAdd + 1 * 4, 3);
  store_qword(kRejectedAdd + 2 * 4, kProcessPageTable + 0x1000);
  store_dword(kRejectedAdd + 18 * 4, kSdmaDoorbell / sizeof(uint32_t));
  store_qword(kRejectedAdd + 20 * 4, kSchedulerMqdGpu);
  store_qword(kRejectedAdd + 22 * 4, kSdmaWritePointerGpu);
  store_dword(kRejectedAdd + 28 * 4, 2);
  store_dword(kRejectedAdd + 30 * 4, 5);
  store_qword(kRejectedAdd + 38 * 4, kApiStatusGpu);
  store_qword(kRejectedAdd + 40 * 4, 2);

  mes_doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{192});
  ring_doorbell(0x60, mes_doorbell);
  EXPECT_EQ(transport_.dword_at(kApiStatusPhysical), 1u)
      << "a rejected ADD_QUEUE published success";
  EXPECT_EQ(transport_.dword_at(kReadPointerPhysical), 128u) << "a rejected ADD_QUEUE was retired";
  EXPECT_EQ(soc_.queue_registry().active_queues(), 1u)
      << "a failed replacement detached the old queue binding";
  EXPECT_EQ(soc_.gpu_vm().find_vmid(3), original_address_space);
  ASSERT_TRUE(soc_.gpu_vm().lookup(*original_address_space));
  EXPECT_EQ(soc_.gpu_vm().lookup(*original_address_space)->translation_epoch, original_epoch)
      << "ADD_QUEUE changed an existing PASID's translation generation";

  store_dword(kSdmaRingPhysical + 4 * 4, 0x87654321);
  sdma_doorbell = std::bit_cast<std::array<std::byte, 4>>(uint32_t{60});
  ring_doorbell(kSdmaDoorbell, sdma_doorbell);
  EXPECT_EQ(transport_.dword_at(kSdmaDestinationPhysical), 0x87654321u)
      << "the old SDMA queue stopped after a failed replacement";
  EXPECT_EQ(transport_.dword_at(kSdmaReadPointerPhysical), 60u);

  // Replace the SDMA queue with a compute queue on the same doorbell.
  // GpuQueueRegistry must detach the old SDMA binding exactly once; otherwise
  // both queue owners keep observing the same write.
  constexpr uint64_t kSecondAdd = kRejectedAdd;
  store_dword(kSecondAdd, 0x00040021);
  store_dword(kSecondAdd + 1 * 4, 3);
  store_qword(kSecondAdd + 2 * 4, kProcessPageTable);
  store_qword(kSecondAdd + 20 * 4, kSchedulerMqdGpu);
  store_dword(kSecondAdd + 28 * 4, 1);
  store_qword(kSecondAdd + 38 * 4, kApiStatusGpu);
  store_qword(kSecondAdd + 40 * 4, 2);

  store_dword(kSchedulerMqdPhysical + 130 * 4, 0);
  store_dword(kSchedulerMqdPhysical + 136 * 4, static_cast<uint32_t>(kComputeRingGpu >> 8));
  store_dword(kSchedulerMqdPhysical + 137 * 4, static_cast<uint32_t>(kComputeRingGpu >> 40));
  store_dword(kSchedulerMqdPhysical + 139 * 4, static_cast<uint32_t>(kComputeReadPointerGpu));
  store_dword(kSchedulerMqdPhysical + 140 * 4, static_cast<uint32_t>(kComputeReadPointerGpu >> 32));
  store_dword(kSchedulerMqdPhysical + 143 * 4, static_cast<uint32_t>(kSdmaDoorbell));
  store_dword(kSchedulerMqdPhysical + 145 * 4, 9);
  store_dword(kSchedulerMqdPhysical + 181 * 4, 0);

  mes_doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{192});
  ring_doorbell(0x60, mes_doorbell);
  EXPECT_EQ(transport_.dword_at(kApiStatusPhysical), 2u);
  EXPECT_EQ(soc_.queue_registry().active_queues(), 1u)
      << "the replacement PM4 queue did not take ownership through the registry";

  write_register_at(kScratchRegister, 0);
  store_dword(kComputeRingPhysical, 0xc0017900);
  store_dword(kComputeRingPhysical + 4, 0x40);
  store_dword(kComputeRingPhysical + 8, 0xfeedface);
  for (uint32_t dword = 3; dword < 60; ++dword)
    store_dword(kComputeRingPhysical + dword * sizeof(uint32_t), 0xffff1000);
  store_dword(kSdmaRingPhysical + 4 * 4, 0x12345678);
  sdma_doorbell = std::bit_cast<std::array<std::byte, 4>>(uint32_t{60});
  ring_doorbell(kSdmaDoorbell, sdma_doorbell);
  EXPECT_EQ(read_register_at(kScratchRegister), 0xfeedfaceu);
  EXPECT_EQ(transport_.dword_at(kSdmaDestinationPhysical), 0x87654321u)
      << "the replaced SDMA binding still observed the shared doorbell";
}

TEST_F(GpuDeviceMes, LostConnectionReleasesOnlyPciQueuesAndAllowsPasidReuse) {
  constexpr uint32_t kPasid = 3;
  constexpr uint64_t kSdmaDoorbell = 0x4808;
  constexpr uint64_t kSdmaRingGpu = 0x20000;
  constexpr uint64_t kSdmaReadPointerGpu = 0x21000;
  constexpr uint64_t kSdmaWritePointerGpu = 0x22000;
  constexpr uint64_t kProcessPageTable = 0x30000;

  rocjitsu::amdgpu::CommandProcessor legacy_cp("legacy-cp");
  legacy_cp.set_gpu_vm(&soc_.gpu_vm());
  rocjitsu::amdgpu::LegacyPageTable legacy_page_table;
  std::shared_mutex legacy_page_table_mutex;
  rocjitsu::amdgpu::LegacyGpuVmAdapter legacy_vm(soc_.gpu_vm(), soc_.memory());
  const rocjitsu::amdgpu::AddressSpaceHandle legacy_address_space =
      legacy_vm.register_address_space(9, &legacy_page_table, &legacy_page_table_mutex);
  ASSERT_TRUE(legacy_address_space);
  const rocjitsu::amdgpu::QueueHandle legacy_queue = soc_.queue_registry().register_queue({
      .identity = {.address_space = legacy_address_space, .process_id = 9, .queue_id = 77},
      .ring = {.base_address = 0x1000,
               .size_bytes = 4096,
               .consumer_pointer_address = 0x2000,
               .producer_pointer_address = 0x3000},
      .doorbell = {},
      .binding_factory = rocjitsu::amdgpu::make_aql_queue_binding_factory(legacy_cp),
      .type = rocjitsu::amdgpu::QueueType::Compute,
      .packet_format = rocjitsu::amdgpu::QueuePacketFormat::Aql,
  });
  ASSERT_TRUE(legacy_queue);

  const std::optional<rocjitsu::amdgpu::AddressSpaceHandle> first_address_space =
      submit_sdma_queue(kPasid, kSdmaDoorbell, kSdmaRingGpu, kSdmaReadPointerGpu,
                        kSdmaWritePointerGpu, kProcessPageTable);
  ASSERT_TRUE(first_address_space);
  ASSERT_EQ(soc_.queue_registry().active_queues(), 2u);
  const rocjitsu::amdgpu::AddressSpaceHandle gart = soc_.gpu_vm().gart_address_space();
  ASSERT_TRUE(gart);
  ASSERT_TRUE(soc_.gpu_vm().lookup(gart)->ready);

  device_.reset(simdojo::ResetKind::LostConnection);

  EXPECT_TRUE(soc_.queue_registry().contains(legacy_queue));
  EXPECT_EQ(soc_.queue_registry().active_queues(), 1u);
  EXPECT_EQ(soc_.gpu_vm().find_vmid(9), legacy_address_space);
  EXPECT_FALSE(soc_.gpu_vm().find_vmid(kPasid));
  ASSERT_TRUE(soc_.gpu_vm().lookup(gart));
  EXPECT_FALSE(soc_.gpu_vm().lookup(gart)->ready);

  ASSERT_TRUE(device_.detach_transport(&attached_));
  ASSERT_TRUE(device_.attach_transport(&attached_));
  configure_gart();
  const std::optional<rocjitsu::amdgpu::AddressSpaceHandle> replacement_address_space =
      submit_sdma_queue(kPasid, kSdmaDoorbell, kSdmaRingGpu, kSdmaReadPointerGpu,
                        kSdmaWritePointerGpu, kProcessPageTable);
  ASSERT_TRUE(replacement_address_space);
  EXPECT_EQ(replacement_address_space->slot, first_address_space->slot);
  EXPECT_NE(replacement_address_space->generation, first_address_space->generation);
  EXPECT_EQ(soc_.queue_registry().active_queues(), 2u);

  device_.reset(simdojo::ResetKind::LostConnection);
  EXPECT_TRUE(soc_.queue_registry().unregister_queue(legacy_queue));
  EXPECT_TRUE(legacy_vm.unregister_address_space(legacy_address_space));
}

TEST_F(GpuDeviceMes, PartialMesTeardownStillDrainsRegisterSdmaCallbacks) {
  constexpr uint32_t kPasid = 3;
  constexpr uint64_t kProcessPageTable = 0x30000;
  constexpr uint64_t kSdmaDoorbell = 0x4808;
  constexpr uint64_t kSdmaRingGpu = 0x20000;
  constexpr uint64_t kSdmaReadPointerGpu = 0x21000;
  constexpr uint64_t kSdmaWritePointerGpu = 0x22000;
  constexpr uint64_t kSdmaBase = 0x1260;
  const auto absolute = [](uint64_t reg) { return (kSdmaBase + reg) * sizeof(uint32_t); };

  write_register_at(absolute(0x0200), (10u << 1) | 0x1001u);
  write_register_at(absolute(0x0201), static_cast<uint32_t>(kSchedulerRingGpu >> 8));
  write_register_at(absolute(0x0202), static_cast<uint32_t>(kSchedulerRingGpu >> 40));
  write_register_at(absolute(0x0207), static_cast<uint32_t>(kSchedulerReadPointerGpu));
  write_register_at(absolute(0x0208), static_cast<uint32_t>(kSchedulerReadPointerGpu >> 32));
  write_register_at(absolute(0x020f), 0x10000000);
  write_register_at(absolute(0x0211), 0x800);
  store_dword(kSchedulerRingPhysical, 0);
  auto register_doorbell = std::bit_cast<std::array<std::byte, sizeof(uint64_t)>>(uint64_t{4});
  ring_doorbell(0x800, register_doorbell);
  ASSERT_EQ(soc_.queue_registry().active_queues(), 1u);

  const std::optional<rocjitsu::amdgpu::AddressSpaceHandle> process_address_space =
      submit_sdma_queue(kPasid, kSdmaDoorbell, kSdmaRingGpu, kSdmaReadPointerGpu,
                        kSdmaWritePointerGpu, kProcessPageTable);
  ASSERT_TRUE(process_address_space);
  ASSERT_EQ(soc_.queue_registry().active_queues(), 2u);

  ASSERT_TRUE(soc_.gpu_vm().retain_queue(*process_address_space));
  EXPECT_FALSE(device_.shutdown_frontend())
      << "the injected process-address-space retention should make MES cleanup partial";
  EXPECT_EQ(soc_.queue_registry().active_queues(), 0u)
      << "partial MES cleanup skipped the independent register-backed SDMA owner";

  EXPECT_TRUE(soc_.gpu_vm().release_queue(*process_address_space));

  rocjitsu::GpuPciDevice replacement("replacement", configured_spec(), &trace_, &soc_);
  EXPECT_TRUE(replacement.usable())
      << "orphaned teardown metadata prevented a replacement frontend from binding MES";
  EXPECT_FALSE(soc_.gpu_vm().lookup(*process_address_space))
      << "replacement attachment did not reap the released orphan binding";
  ASSERT_TRUE(replacement.attach_transport(&attached_));
  EXPECT_TRUE(replacement.detach_transport(&attached_));
}

TEST(GpuDeviceLifetime, ReplacementFrontendReusesPasidAndDoorbellAfterDestruction) {
  constexpr uint32_t kPasid = 3;
  constexpr uint64_t kPageTable = 0x10000;
  constexpr uint64_t kGartStart = 0x100000000ULL;
  constexpr uint64_t kMesRingGpu = kGartStart;
  constexpr uint64_t kMesReadPointerGpu = kGartStart + 0x1000;
  constexpr uint64_t kApiStatusGpu = kGartStart + 0x2000;
  constexpr uint64_t kSdmaMqdGpu = kGartStart + 0x4000;
  constexpr uint64_t kMesRingPhysical = 0x200000;
  constexpr uint64_t kMesReadPointerPhysical = 0x210000;
  constexpr uint64_t kApiStatusPhysical = 0x220000;
  constexpr uint64_t kSdmaMqdPhysical = 0x240000;
  constexpr uint64_t kProcessPageTable = 0x30000;
  constexpr uint64_t kSdmaRingGpu = 0x20000;
  constexpr uint64_t kSdmaReadPointerGpu = 0x21000;
  constexpr uint64_t kSdmaWritePointerGpu = 0x22000;
  constexpr uint64_t kSdmaDoorbell = 0x4808;
  constexpr uint64_t kGcBase = 0x1260;
  const auto absolute = [](uint64_t reg) { return (kGcBase + reg) * sizeof(uint32_t); };

  rocjitsu::RegisterSymbols symbols;
  rocjitsu::BarAccessTrace trace(symbols);
  rocjitsu::amdgpu::GpuMemory memory("memory");
  rocjitsu::SoC soc("soc", &memory);
  RecordingTransport transport;
  simdojo::PciDevice::Transport endpoints{.irq = &transport, .dma = &transport};

  const auto write_register = [](rocjitsu::GpuPciDevice &device, uint64_t offset, uint32_t value) {
    auto raw = std::bit_cast<std::array<std::byte, sizeof(value)>>(value);
    return device.bar_access(rocjitsu::GpuPciDevice::kRegisterBar, raw, offset, true) ==
           static_cast<int64_t>(raw.size());
  };
  const auto write_vram_qword = [](rocjitsu::GpuPciDevice &device, uint64_t offset,
                                   uint64_t value) {
    auto raw = std::bit_cast<std::array<std::byte, sizeof(value)>>(value);
    return device.bar_access(rocjitsu::GpuPciDevice::kVramBar, raw, offset, true) ==
           static_cast<int64_t>(raw.size());
  };
  const auto store_dword = [&transport](uint64_t address, uint32_t value) {
    const auto raw = std::bit_cast<std::array<std::byte, sizeof(value)>>(value);
    for (std::size_t index = 0; index < raw.size(); ++index)
      transport.memory[address + index] = raw[index];
  };
  const auto store_qword = [&transport](uint64_t address, uint64_t value) {
    const auto raw = std::bit_cast<std::array<std::byte, sizeof(value)>>(value);
    for (std::size_t index = 0; index < raw.size(); ++index)
      transport.memory[address + index] = raw[index];
  };
  const auto configure_and_add_queue = [&](rocjitsu::GpuPciDevice &device) {
    if (!write_register(device, absolute(0x169f), static_cast<uint32_t>(kPageTable)) ||
        !write_register(device, absolute(0x16a0), static_cast<uint32_t>(kPageTable >> 32)) ||
        !write_register(device, absolute(0x16bf), static_cast<uint32_t>(kGartStart >> 12)) ||
        !write_register(device, absolute(0x16c0), static_cast<uint32_t>(kGartStart >> 44)) ||
        !write_register(device, absolute(0x16df),
                        static_cast<uint32_t>((kGartStart + 0x4fff) >> 12)) ||
        !write_register(device, absolute(0x16e0),
                        static_cast<uint32_t>((kGartStart + 0x4fff) >> 44))) {
      return std::optional<rocjitsu::amdgpu::AddressSpaceHandle>{};
    }
    for (const auto [page, physical] :
         {std::pair{0u, kMesRingPhysical}, std::pair{1u, kMesReadPointerPhysical},
          std::pair{2u, kApiStatusPhysical}, std::pair{4u, kSdmaMqdPhysical}}) {
      if (!write_vram_qword(device, kPageTable + page * sizeof(uint64_t), physical | 0x3))
        return std::optional<rocjitsu::amdgpu::AddressSpaceHandle>{};
    }
    if (!write_register(device, absolute(0x1657 + 17), 1) ||
        !write_register(device, absolute(0x1fb1), static_cast<uint32_t>(kMesRingGpu >> 8)) ||
        !write_register(device, absolute(0x1fb2), static_cast<uint32_t>(kMesRingGpu >> 40)) ||
        !write_register(device, absolute(0x1fb4), static_cast<uint32_t>(kMesReadPointerGpu)) ||
        !write_register(device, absolute(0x1fb5),
                        static_cast<uint32_t>(kMesReadPointerGpu >> 32)) ||
        !write_register(device, absolute(0x1fb8), 0x40000060) ||
        !write_register(device, absolute(0x1fba), 9) ||
        !write_register(device, absolute(0x1fab), 1)) {
      return std::optional<rocjitsu::amdgpu::AddressSpaceHandle>{};
    }

    store_qword(kMesReadPointerPhysical, 0);
    store_dword(kApiStatusPhysical, 0);
    store_dword(kMesRingPhysical, 0x00040021);
    store_dword(kMesRingPhysical + 1 * 4, kPasid);
    store_qword(kMesRingPhysical + 2 * 4, kProcessPageTable);
    store_dword(kMesRingPhysical + 18 * 4, kSdmaDoorbell / sizeof(uint32_t));
    store_qword(kMesRingPhysical + 20 * 4, kSdmaMqdGpu);
    store_qword(kMesRingPhysical + 22 * 4, kSdmaWritePointerGpu);
    store_dword(kMesRingPhysical + 28 * 4, 2);
    store_dword(kMesRingPhysical + 30 * 4, 16);
    store_qword(kMesRingPhysical + 38 * 4, kApiStatusGpu);
    store_qword(kMesRingPhysical + 40 * 4, 1);
    store_dword(kSdmaMqdPhysical + 1 * 4, static_cast<uint32_t>(kSdmaRingGpu >> 8));
    store_dword(kSdmaMqdPhysical + 2 * 4, static_cast<uint32_t>(kSdmaRingGpu >> 40));
    store_dword(kSdmaMqdPhysical + 7 * 4, static_cast<uint32_t>(kSdmaReadPointerGpu));
    store_dword(kSdmaMqdPhysical + 8 * 4, static_cast<uint32_t>(kSdmaReadPointerGpu >> 32));
    store_dword(kSdmaMqdPhysical + 126 * 4, 1);
    store_dword(kSdmaMqdPhysical + 127 * 4, 7);

    auto doorbell = std::bit_cast<std::array<std::byte, sizeof(uint64_t)>>(uint64_t{64});
    if (device.bar_access(rocjitsu::GpuPciDevice::kDoorbellBar, doorbell, 0x60, true) !=
        static_cast<int64_t>(doorbell.size())) {
      return std::optional<rocjitsu::amdgpu::AddressSpaceHandle>{};
    }
    device.drain_doorbell_inbox_for_test();
    return soc.gpu_vm().find_vmid(kPasid);
  };

  rocjitsu::amdgpu::AddressSpaceHandle first_address_space;
  {
    auto device =
        std::make_unique<rocjitsu::GpuPciDevice>("first", configured_spec(), &trace, &soc);
    ASSERT_TRUE(device->attach_transport(&endpoints));
    const auto address_space = configure_and_add_queue(*device);
    ASSERT_TRUE(address_space);
    first_address_space = *address_space;
    ASSERT_EQ(soc.queue_registry().active_queues(), 1u);
    ASSERT_TRUE(device->detach_transport(&endpoints));
  }
  EXPECT_EQ(soc.queue_registry().active_queues(), 0u);
  EXPECT_FALSE(soc.gpu_vm().find_vmid(kPasid));

  {
    auto device =
        std::make_unique<rocjitsu::GpuPciDevice>("replacement", configured_spec(), &trace, &soc);
    ASSERT_TRUE(device->attach_transport(&endpoints));
    const auto address_space = configure_and_add_queue(*device);
    ASSERT_TRUE(address_space);
    EXPECT_EQ(address_space->slot, first_address_space.slot);
    EXPECT_NE(address_space->generation, first_address_space.generation);
    EXPECT_EQ(soc.queue_registry().active_queues(), 1u);
    ASSERT_TRUE(device->detach_transport(&endpoints));
  }
  EXPECT_EQ(soc.queue_registry().active_queues(), 0u);
  EXPECT_FALSE(soc.gpu_vm().find_vmid(kPasid));
}

TEST(GpuDeviceLifetime, ExplicitShutdownSurvivesCoreOwnerDestruction) {
  rocjitsu::RegisterSymbols symbols;
  rocjitsu::BarAccessTrace trace(symbols);
  rocjitsu::amdgpu::GpuMemory memory("memory");
  auto soc = std::make_unique<rocjitsu::SoC>("soc", &memory);
  auto device =
      std::make_unique<rocjitsu::GpuPciDevice>("gpu", configured_spec(), &trace, soc.get());
  ASSERT_TRUE(device->usable());

  EXPECT_TRUE(device->shutdown_frontend());
  EXPECT_TRUE(device->shutdown_frontend()) << "explicit shutdown must be idempotent";

  soc.reset();
  device.reset();
}

class GpuDeviceSdma : public GpuDeviceMes {};

TEST_F(GpuDeviceSdma, ExecutesTheFirmwareFreeSdmaRingWrite) {
  constexpr uint64_t kSdmaBase = 0x1260;
  const auto absolute = [](uint64_t reg) { return (kSdmaBase + reg) * 4; };
  write_register_at(absolute(0x0200), (10u << 1) | 0x1001u);
  write_register_at(absolute(0x0201), static_cast<uint32_t>(kSchedulerRingGpu >> 8));
  write_register_at(absolute(0x0202), static_cast<uint32_t>(kSchedulerRingGpu >> 40));
  write_register_at(absolute(0x0203), 0);
  write_register_at(absolute(0x0204), 0);
  write_register_at(absolute(0x0207), static_cast<uint32_t>(kSchedulerReadPointerGpu));
  write_register_at(absolute(0x0208), static_cast<uint32_t>(kSchedulerReadPointerGpu >> 32));
  write_register_at(absolute(0x020f), 0x10000000);
  write_register_at(absolute(0x0211), 0x800);

  // The driver ring test emits one linear write followed by NOP padding to
  // the ring's 16-dword alignment.
  store_dword(kSchedulerRingPhysical, 2);
  store_qword(kSchedulerRingPhysical + 4, kSchedulerApiStatusGpu);
  store_dword(kSchedulerRingPhysical + 12, 0);
  store_dword(kSchedulerRingPhysical + 16, 0xdeadbeef);

  auto doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{64});
  ring_doorbell(0x800, doorbell);

  EXPECT_EQ(transport_.dword_at(kSchedulerApiStatusPhysical), 0xdeadbeefu);
  EXPECT_EQ(transport_.dword_at(kSchedulerReadPointerPhysical), 64u);
  EXPECT_EQ(read_register_at(absolute(0x0203)), 64u);
}

TEST_F(GpuDeviceSdma, AppliesRegisterEffectsOnlyOnTheDoorbellOwnerThread) {
  constexpr uint64_t kSdmaBase = 0x1260;
  constexpr uint64_t kGcControlBase = 0xa000;
  constexpr uint64_t kScratchRegister = (kGcControlBase + 0x2040) * sizeof(uint32_t);
  const auto absolute = [](uint64_t reg) { return (kSdmaBase + reg) * sizeof(uint32_t); };
  write_register_at(absolute(0x0200), (10u << 1) | 0x1001u);
  write_register_at(absolute(0x0201), static_cast<uint32_t>(kSchedulerRingGpu >> 8));
  write_register_at(absolute(0x0202), static_cast<uint32_t>(kSchedulerRingGpu >> 40));
  write_register_at(absolute(0x0207), static_cast<uint32_t>(kSchedulerReadPointerGpu));
  write_register_at(absolute(0x0208), static_cast<uint32_t>(kSchedulerReadPointerGpu >> 32));
  write_register_at(absolute(0x020f), 0x10000000);
  write_register_at(absolute(0x0211), 0x800);
  write_register_at(kScratchRegister, 0);

  store_dword(kSchedulerRingPhysical, 14);
  store_dword(kSchedulerRingPhysical + sizeof(uint32_t), static_cast<uint32_t>(kScratchRegister));
  store_dword(kSchedulerRingPhysical + 2 * sizeof(uint32_t), 0xdecafbad);

  const std::thread::id owner_thread = std::this_thread::get_id();
  auto doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{3 * sizeof(uint32_t)});
  ring_doorbell(0x800, doorbell);

  EXPECT_EQ(read_register_at(kScratchRegister), 0xdecafbadu);
  EXPECT_EQ(device_.sdma_pci_effect_count_for_test(), 1u);
  EXPECT_EQ(device_.last_sdma_pci_effect_thread_for_test(), owner_thread);
}

TEST_F(GpuDeviceSdma, ConcurrentResetCancelsQueuedPciEffectsWithoutDeadlock) {
  constexpr uint64_t kSdmaBase = 0x1260;
  constexpr uint64_t kGcControlBase = 0xa000;
  constexpr uint64_t kScratchRegister = (kGcControlBase + 0x2040) * sizeof(uint32_t);
  const auto absolute = [](uint64_t reg) { return (kSdmaBase + reg) * sizeof(uint32_t); };
  write_register_at(absolute(0x0200), (10u << 1) | 0x1001u);
  write_register_at(absolute(0x0201), static_cast<uint32_t>(kSchedulerRingGpu >> 8));
  write_register_at(absolute(0x0202), static_cast<uint32_t>(kSchedulerRingGpu >> 40));
  write_register_at(absolute(0x0207), static_cast<uint32_t>(kSchedulerReadPointerGpu));
  write_register_at(absolute(0x0208), static_cast<uint32_t>(kSchedulerReadPointerGpu >> 32));
  write_register_at(absolute(0x020f), 0x10000000);
  write_register_at(absolute(0x0211), 0x800);

  store_dword(kSchedulerRingPhysical, 14);
  store_dword(kSchedulerRingPhysical + sizeof(uint32_t), static_cast<uint32_t>(kScratchRegister));
  store_dword(kSchedulerRingPhysical + 2 * sizeof(uint32_t), 0xdecafbad);
  auto doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{3 * sizeof(uint32_t)});
  ASSERT_EQ(device_.bar_access(rocjitsu::GpuPciDevice::kDoorbellBar, doorbell, 0x800,
                               /*write=*/true),
            static_cast<int64_t>(doorbell.size()));

  std::jthread drain([this]() { device_.drain_doorbell_inbox_for_test(); });
  device_.reset(simdojo::ResetKind::FunctionLevel);
  drain.join();

  EXPECT_EQ(soc_.queue_registry().active_queues(), 0u);
  EXPECT_EQ(read_register_at(kScratchRegister), 0u);
}

TEST_F(GpuDeviceSdma, ResetWaitsForAdmittedPciEffectBeforeResettingRegisters) {
  constexpr uint64_t kSdmaBase = 0x1260;
  constexpr uint64_t kGcControlBase = 0xa000;
  constexpr uint64_t kScratchRegister = (kGcControlBase + 0x2040) * sizeof(uint32_t);
  const auto absolute = [](uint64_t reg) { return (kSdmaBase + reg) * sizeof(uint32_t); };
  write_register_at(absolute(0x0200), (10u << 1) | 0x1001u);
  write_register_at(absolute(0x0201), static_cast<uint32_t>(kSchedulerRingGpu >> 8));
  write_register_at(absolute(0x0202), static_cast<uint32_t>(kSchedulerRingGpu >> 40));
  write_register_at(absolute(0x0207), static_cast<uint32_t>(kSchedulerReadPointerGpu));
  write_register_at(absolute(0x0208), static_cast<uint32_t>(kSchedulerReadPointerGpu >> 32));
  write_register_at(absolute(0x020f), 0x10000000);
  write_register_at(absolute(0x0211), 0x800);

  store_dword(kSchedulerRingPhysical, 14);
  store_dword(kSchedulerRingPhysical + sizeof(uint32_t), static_cast<uint32_t>(kScratchRegister));
  store_dword(kSchedulerRingPhysical + 2 * sizeof(uint32_t), 0xdecafbad);

  std::mutex gate_mutex;
  std::condition_variable gate_changed;
  bool admitted = false;
  bool release = false;
  device_.set_sdma_pci_effect_admitted_hook_for_test([&] {
    std::unique_lock lock(gate_mutex);
    admitted = true;
    gate_changed.notify_all();
    gate_changed.wait(lock, [&] { return release; });
  });

  auto doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{3 * sizeof(uint32_t)});
  ASSERT_EQ(device_.bar_access(rocjitsu::GpuPciDevice::kDoorbellBar, doorbell, 0x800,
                               /*write=*/true),
            static_cast<int64_t>(doorbell.size()));

  std::jthread drain([this]() { device_.drain_doorbell_inbox_for_test(); });
  {
    std::unique_lock lock(gate_mutex);
    ASSERT_TRUE(gate_changed.wait_for(lock, std::chrono::seconds(1), [&] { return admitted; }));
  }

  std::future<void> reset = std::async(
      std::launch::async, [this]() { device_.reset(simdojo::ResetKind::FunctionLevel); });
  EXPECT_EQ(reset.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout)
      << "reset crossed an admitted PCI effect";

  {
    const std::lock_guard lock(gate_mutex);
    release = true;
  }
  gate_changed.notify_all();
  drain.join();

  ASSERT_EQ(reset.wait_for(std::chrono::seconds(1)), std::future_status::ready);
  reset.get();
  device_.set_sdma_pci_effect_admitted_hook_for_test({});
  EXPECT_EQ(read_register_at(kScratchRegister), 0u)
      << "a pre-reset SDMA effect mutated the reset register state";
}

TEST_F(GpuDeviceSdma, RetriesUnavailableRingFetchWithoutAnotherDoorbell) {
  constexpr uint64_t kSdmaBase = 0x1260;
  const auto absolute = [](uint64_t reg) { return (kSdmaBase + reg) * 4; };
  write_register_at(absolute(0x0200), (10u << 1) | 0x1001u);
  write_register_at(absolute(0x0201), static_cast<uint32_t>(kSchedulerRingGpu >> 8));
  write_register_at(absolute(0x0202), static_cast<uint32_t>(kSchedulerRingGpu >> 40));
  write_register_at(absolute(0x0207), static_cast<uint32_t>(kSchedulerReadPointerGpu));
  write_register_at(absolute(0x0208), static_cast<uint32_t>(kSchedulerReadPointerGpu >> 32));
  write_register_at(absolute(0x020f), 0x10000000);
  write_register_at(absolute(0x0211), 0x800);

  store_dword(kSchedulerRingPhysical, 2);
  store_qword(kSchedulerRingPhysical + 4, kSchedulerApiStatusGpu);
  store_dword(kSchedulerRingPhysical + 12, 0);
  store_dword(kSchedulerRingPhysical + 16, 0xdeadbeef);
  transport_.next_read_outcome =
      std::pair{kSchedulerRingPhysical, simdojo::DmaAccessOutcome::Unavailable};

  auto doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{64});
  ring_doorbell(0x800, doorbell);
  EXPECT_EQ(transport_.dword_at(kSchedulerApiStatusPhysical), 0xdeadbeefu);
  EXPECT_EQ(transport_.dword_at(kSchedulerReadPointerPhysical), 64u)
      << "the transient ring fetch required another guest doorbell";
}

TEST_F(GpuDeviceSdma, RetriesUnsatisfiedMemoryPollWithoutAnotherDoorbell) {
  constexpr uint64_t kSdmaBase = 0x1260;
  const auto absolute = [](uint64_t reg) { return (kSdmaBase + reg) * 4; };
  write_register_at(absolute(0x0200), (10u << 1) | 0x1001u);
  write_register_at(absolute(0x0201), static_cast<uint32_t>(kSchedulerRingGpu >> 8));
  write_register_at(absolute(0x0202), static_cast<uint32_t>(kSchedulerRingGpu >> 40));
  write_register_at(absolute(0x0207), static_cast<uint32_t>(kSchedulerReadPointerGpu));
  write_register_at(absolute(0x0208), static_cast<uint32_t>(kSchedulerReadPointerGpu >> 32));
  write_register_at(absolute(0x020f), 0x10000000);
  write_register_at(absolute(0x0211), 0x800);

  store_dword(kSchedulerApiStatusPhysical, 0);
  store_dword(kSchedulerRingPhysical + 0 * 4, 0xb0000008);
  store_qword(kSchedulerRingPhysical + 1 * 4, kSchedulerApiStatusGpu);
  store_dword(kSchedulerRingPhysical + 3 * 4, 1);
  store_dword(kSchedulerRingPhysical + 4 * 4, UINT32_MAX);

  auto doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{64});
  ring_doorbell(0x800, doorbell);
  EXPECT_EQ(transport_.dword_at(kSchedulerReadPointerPhysical), 0u);

  store_dword(kSchedulerApiStatusPhysical, 1);
  device_.drain_doorbell_inbox_for_test();
  EXPECT_EQ(transport_.dword_at(kSchedulerReadPointerPhysical), 64u)
      << "the transient SDMA poll required another guest doorbell";
}

TEST_F(GpuDeviceSdma, RegisterBackedQueueRejectsANonSystemGartRingPage) {
  constexpr uint64_t kSdmaBase = 0x1260;
  const auto absolute = [](uint64_t reg) { return (kSdmaBase + reg) * 4; };

  write_register_at(absolute(0x0200), (10u << 1) | 0x1001u);
  write_register_at(absolute(0x0201), static_cast<uint32_t>(kSchedulerRingGpu >> 8));
  write_register_at(absolute(0x0202), static_cast<uint32_t>(kSchedulerRingGpu >> 40));
  write_register_at(absolute(0x0207), static_cast<uint32_t>(kSchedulerReadPointerGpu));
  write_register_at(absolute(0x0208), static_cast<uint32_t>(kSchedulerReadPointerGpu >> 32));
  write_register_at(absolute(0x020f), 0x10000000);
  write_register_at(absolute(0x0211), 0x800);

  // VMID-0 GART entries must select system memory. A merely valid local entry
  // is not a different route to the same bytes, so the queue must remain at
  // its old read pointer rather than consume a ring from guest RAM.
  store_vram_qword(kPageTable + 5 * sizeof(uint64_t), kSchedulerRingPhysical | 0x1);
  store_dword(kSchedulerRingPhysical, 2);
  store_qword(kSchedulerRingPhysical + 4, kSchedulerApiStatusGpu);
  store_dword(kSchedulerRingPhysical + 12, 0);
  store_dword(kSchedulerRingPhysical + 16, 0xdeadbeef);

  auto doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{64});
  ring_doorbell(0x800, doorbell);

  EXPECT_EQ(transport_.dword_at(kSchedulerApiStatusPhysical), 0u);
  EXPECT_EQ(transport_.dword_at(kSchedulerReadPointerPhysical), 0u);
  EXPECT_EQ(read_register_at(absolute(0x0203)), 0u);
}

TEST_F(GpuDeviceSdma, RegisterBackedQueueRoutesOutsideGartToLocalVram) {
  constexpr uint64_t kSdmaBase = 0x1260;
  constexpr uint64_t kLocalRing = 0x40000;
  constexpr uint64_t kLocalReadPointer = 0x41000;
  constexpr uint64_t kLocalDestination = 0x42000;
  const auto absolute = [](uint64_t reg) { return (kSdmaBase + reg) * 4; };

  write_register_at(absolute(0x0200), (10u << 1) | 0x1001u);
  write_register_at(absolute(0x0201), static_cast<uint32_t>(kLocalRing >> 8));
  write_register_at(absolute(0x0202), static_cast<uint32_t>(kLocalRing >> 40));
  write_register_at(absolute(0x0207), static_cast<uint32_t>(kLocalReadPointer));
  write_register_at(absolute(0x0208), static_cast<uint32_t>(kLocalReadPointer >> 32));
  write_register_at(absolute(0x020f), 0x10000000);
  write_register_at(absolute(0x0211), 0x800);

  // The shared GART translator sends addresses outside the aperture to the
  // local-memory domain. Exercise both the ring fetch and the packet's write
  // through that route so PCI transport memory cannot accidentally satisfy it.
  store_vram_dword(kLocalRing, 2);
  store_vram_qword(kLocalRing + 4, kLocalDestination);
  store_vram_dword(kLocalRing + 12, 0);
  store_vram_dword(kLocalRing + 16, 0xdeadbeef);

  auto doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{64});
  ring_doorbell(0x800, doorbell);

  EXPECT_EQ(vram_dword_at(kLocalDestination), 0xdeadbeefu);
  EXPECT_EQ(vram_dword_at(kLocalReadPointer), 64u);
  EXPECT_EQ(transport_.dword_at(kLocalDestination), 0u);
  EXPECT_EQ(transport_.dword_at(kLocalReadPointer), 0u);
}

TEST_F(GpuDeviceSdma, TerminalFaultSurvivesUnavailableReadPointerPublication) {
  constexpr uint64_t kSdmaBase = 0x1260;
  constexpr uint64_t kFaultSourceGpu = kGartStart + 0x7000;
  constexpr uint64_t kLaterDestinationGpu = kGartStart + 0x8000;
  const auto absolute = [](uint64_t reg) { return (kSdmaBase + reg) * 4; };

  write_register_at(absolute(0x0200), (10u << 1) | 0x1001u);
  write_register_at(absolute(0x0201), static_cast<uint32_t>(kSchedulerRingGpu >> 8));
  write_register_at(absolute(0x0202), static_cast<uint32_t>(kSchedulerRingGpu >> 40));
  write_register_at(absolute(0x0207), static_cast<uint32_t>(kSchedulerReadPointerGpu));
  write_register_at(absolute(0x0208), static_cast<uint32_t>(kSchedulerReadPointerGpu >> 32));
  write_register_at(absolute(0x020f), 0x10000000);
  write_register_at(absolute(0x0211), 0x800);

  // A terminal copy fault retires the seven-dword packet. Its read-pointer
  // publication stalls once, and the following valid write must remain
  // unexecuted when publication is retried autonomously.
  store_dword(kSchedulerRingPhysical + 0 * 4, 1);
  store_dword(kSchedulerRingPhysical + 1 * 4, 0);
  store_qword(kSchedulerRingPhysical + 3 * 4, kFaultSourceGpu);
  store_qword(kSchedulerRingPhysical + 5 * 4, kLaterDestinationGpu);
  store_dword(kSchedulerRingPhysical + 7 * 4, 2);
  store_qword(kSchedulerRingPhysical + 8 * 4, kLaterDestinationGpu);
  store_dword(kSchedulerRingPhysical + 10 * 4, 0);
  store_dword(kSchedulerRingPhysical + 11 * 4, 0xdeadbeef);

  transport_.next_read_outcome =
      std::pair{kSchedulerApiStatusPhysical, simdojo::DmaAccessOutcome::Faulted};
  transport_.next_write_outcome =
      std::pair{kSchedulerReadPointerPhysical, simdojo::DmaAccessOutcome::Unavailable};

  auto doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{48});
  ring_doorbell(0x800, doorbell);
  EXPECT_EQ(transport_.dword_at(kSchedulerReadPointerPhysical), 28u);
  EXPECT_EQ(transport_.dword_at(kSchedulerRingFencePhysical), 0u);

  device_.drain_doorbell_inbox_for_test();
  EXPECT_EQ(transport_.dword_at(kSchedulerReadPointerPhysical), 28u);
  EXPECT_EQ(transport_.dword_at(kSchedulerRingFencePhysical), 0u)
      << "a terminal packet outcome was forgotten after publication retried";
}

TEST_F(GpuDeviceSdma, RegisterReconfigurationWithPendingPacketFailsClosed) {
  constexpr uint64_t kSdmaBase = 0x1260;
  constexpr uint64_t kPollGpu = kGartStart + 0x7000;
  constexpr uint64_t kReplacementRingGpu = kGartStart + 0x8000;
  const auto absolute = [](uint64_t reg) { return (kSdmaBase + reg) * 4; };

  write_register_at(absolute(0x0200), (10u << 1) | 0x1001u);
  write_register_at(absolute(0x0201), static_cast<uint32_t>(kSchedulerRingGpu >> 8));
  write_register_at(absolute(0x0202), static_cast<uint32_t>(kSchedulerRingGpu >> 40));
  write_register_at(absolute(0x0207), static_cast<uint32_t>(kSchedulerReadPointerGpu));
  write_register_at(absolute(0x0208), static_cast<uint32_t>(kSchedulerReadPointerGpu >> 32));
  write_register_at(absolute(0x020f), 0x10000000);
  write_register_at(absolute(0x0211), 0x800);

  store_dword(kSchedulerApiStatusPhysical, 0);
  store_dword(kSchedulerRingPhysical + 0 * 4, 0xb0000008);
  store_qword(kSchedulerRingPhysical + 1 * 4, kPollGpu);
  store_dword(kSchedulerRingPhysical + 3 * 4, 1);
  store_dword(kSchedulerRingPhysical + 4 * 4, UINT32_MAX);

  auto doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{24});
  ring_doorbell(0x800, doorbell);
  ASSERT_EQ(transport_.dword_at(kSchedulerReadPointerPhysical), 0u);

  // Repoint the MMIO queue while the unsatisfied poll is retained. Replacing
  // that runtime would lose the old packet state and execute this new ring as
  // though the old queue had reached a clean lifetime boundary.
  store_dword(kSchedulerRingFencePhysical + 0 * 4, 2);
  store_qword(kSchedulerRingFencePhysical + 1 * 4, kRingFenceGpu);
  store_dword(kSchedulerRingFencePhysical + 3 * 4, 0);
  store_dword(kSchedulerRingFencePhysical + 4 * 4, 0xdeadbeef);
  write_register_at(absolute(0x0201), static_cast<uint32_t>(kReplacementRingGpu >> 8));
  write_register_at(absolute(0x0202), static_cast<uint32_t>(kReplacementRingGpu >> 40));

  doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{20});
  ring_doorbell(0x800, doorbell);
  EXPECT_EQ(transport_.dword_at(kRingFencePhysical), 0u);
  EXPECT_EQ(transport_.dword_at(kSchedulerReadPointerPhysical), 0u);
}

TEST_F(GpuDeviceSdma, ExecutesAnIndirectBufferAndSignalsItsFence) {
  constexpr uint64_t kSdmaBase = 0x1260;
  constexpr uint64_t kInterruptRing = 0x300000;
  constexpr uint64_t kInterruptWritePointer = 0x301000;
  constexpr uint64_t kHdpFlush = 0x44000;
  const auto absolute = [](uint64_t reg) { return (kSdmaBase + reg) * 4; };

  write_register_at(absolute(0x0200), (10u << 1) | 0x1001u);
  write_register_at(absolute(0x0201), static_cast<uint32_t>(kSchedulerRingGpu >> 8));
  write_register_at(absolute(0x0202), static_cast<uint32_t>(kSchedulerRingGpu >> 40));
  write_register_at(absolute(0x0207), static_cast<uint32_t>(kSchedulerReadPointerGpu));
  write_register_at(absolute(0x0208), static_cast<uint32_t>(kSchedulerReadPointerGpu >> 32));
  write_register_at(absolute(0x020f), 0x10000000);
  write_register_at(absolute(0x0211), 0x800);

  // The interrupt ring is in guest-physical memory. A trap after the fence must
  // publish one SDMA entry and raise the device's sole MSI-X vector.
  write_register_at(0x448c, static_cast<uint32_t>(kInterruptRing >> 8));
  write_register_at(0x4490, static_cast<uint32_t>(kInterruptRing >> 40));
  write_register_at(0x4498, static_cast<uint32_t>(kInterruptWritePointer));
  write_register_at(0x4494, static_cast<uint32_t>(kInterruptWritePointer >> 32));
  write_register_at(0x4480, (10u << 1) | (1u << 0) | (1u << 17) | (2u << 28));
  write_register_at(kHdpFlush, 0xfeedface);

  // The IB writes the self-test sentinel and is padded to its required
  // eight-dword boundary with one burst NOP packet.
  store_dword(kApiStatusPhysical + 0 * 4, 2);
  store_qword(kApiStatusPhysical + 1 * 4, kRingFenceGpu);
  store_dword(kApiStatusPhysical + 3 * 4, 0);
  store_dword(kApiStatusPhysical + 4 * 4, 0xdeadbeef);
  store_dword(kApiStatusPhysical + 5 * 4, 2u << 16);

  // A complete amdgpu IB submission begins with the driver's five-dword
  // conditional envelope. The polling word is initialized to CONTINUE (one),
  // so the following 27 dwords execute: the HDP register write, INDIRECT
  // packet, cache request, fence, trap, and final alignment padding.
  store_dword(kSchedulerRingFencePhysical, 1);
  store_dword(kSchedulerRingPhysical + 0 * 4, 9);
  store_qword(kSchedulerRingPhysical + 1 * 4, kSchedulerRingFenceGpu);
  store_dword(kSchedulerRingPhysical + 3 * 4, 1);
  store_dword(kSchedulerRingPhysical + 4 * 4, 27);
  store_dword(kSchedulerRingPhysical + 5 * 4, 14);
  store_dword(kSchedulerRingPhysical + 6 * 4, kHdpFlush);
  store_dword(kSchedulerRingPhysical + 7 * 4, 0);
  store_dword(kSchedulerRingPhysical + 8 * 4, 1u << 16);
  store_dword(kSchedulerRingPhysical + 10 * 4, 4);
  store_qword(kSchedulerRingPhysical + 11 * 4, kApiStatusGpu);
  store_dword(kSchedulerRingPhysical + 13 * 4, 8);
  store_dword(kSchedulerRingPhysical + 16 * 4, 17);
  store_dword(kSchedulerRingPhysical + 22 * 4, 5);
  store_qword(kSchedulerRingPhysical + 23 * 4, kSchedulerApiStatusGpu);
  store_dword(kSchedulerRingPhysical + 25 * 4, 7);
  store_dword(kSchedulerRingPhysical + 26 * 4, 6);
  store_dword(kSchedulerRingPhysical + 28 * 4, 3u << 16);

  auto doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{128});
  ring_doorbell(0x800, doorbell);

  EXPECT_EQ(transport_.dword_at(kRingFencePhysical), 0xdeadbeefu);
  EXPECT_EQ(transport_.dword_at(kSchedulerApiStatusPhysical), 7u);
  EXPECT_EQ(read_register_at(kHdpFlush), 0u);
  EXPECT_EQ(transport_.dword_at(kSchedulerReadPointerPhysical), 128u);
  EXPECT_EQ(transport_.dword_at(kInterruptRing), 0x0000310au)
      << "client 0x0a and source 49 identify the SDMA trap";
  EXPECT_EQ(transport_.dword_at(kInterruptRing + 3 * sizeof(uint32_t)), 2u << 16)
      << "node two identifies the first XCC to the driver";
  EXPECT_EQ(transport_.dword_at(kInterruptWritePointer), 32u);
  ASSERT_EQ(transport_.triggered.size(), 1u);
  EXPECT_EQ(transport_.triggered.front(), 0u);
}

TEST_F(GpuDeviceSdma, SkipsAConditionalRegionWhenTheReferenceDoesNotMatch) {
  constexpr uint64_t kSdmaBase = 0x1260;
  const auto absolute = [](uint64_t reg) { return (kSdmaBase + reg) * 4; };

  write_register_at(absolute(0x0200), (10u << 1) | 0x1001u);
  write_register_at(absolute(0x0201), static_cast<uint32_t>(kSchedulerRingGpu >> 8));
  write_register_at(absolute(0x0202), static_cast<uint32_t>(kSchedulerRingGpu >> 40));
  write_register_at(absolute(0x0207), static_cast<uint32_t>(kSchedulerReadPointerGpu));
  write_register_at(absolute(0x0208), static_cast<uint32_t>(kSchedulerReadPointerGpu >> 32));
  write_register_at(absolute(0x020f), 0x10000000);
  write_register_at(absolute(0x0211), 0x800);

  store_dword(kSchedulerRingFencePhysical, 0);
  store_dword(kSchedulerRingPhysical + 0 * 4, 9);
  store_qword(kSchedulerRingPhysical + 1 * 4, kSchedulerRingFenceGpu);
  store_dword(kSchedulerRingPhysical + 3 * 4, 1);
  store_dword(kSchedulerRingPhysical + 4 * 4, 5);

  // This first write is the skipped five-dword region. The second write must
  // still execute, proving that COND_EXE resumes at the declared boundary.
  store_dword(kSchedulerRingPhysical + 5 * 4, 2);
  store_qword(kSchedulerRingPhysical + 6 * 4, kRingFenceGpu);
  store_dword(kSchedulerRingPhysical + 8 * 4, 0);
  store_dword(kSchedulerRingPhysical + 9 * 4, 0x11111111);
  store_dword(kSchedulerRingPhysical + 10 * 4, 2);
  store_qword(kSchedulerRingPhysical + 11 * 4, kRingFenceGpu);
  store_dword(kSchedulerRingPhysical + 13 * 4, 0);
  store_dword(kSchedulerRingPhysical + 14 * 4, 0x22222222);
  store_dword(kSchedulerRingPhysical + 15 * 4, 0);

  auto doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{64});
  ring_doorbell(0x800, doorbell);

  EXPECT_EQ(transport_.dword_at(kRingFencePhysical), 0x22222222u);
  EXPECT_EQ(transport_.dword_at(kSchedulerReadPointerPhysical), 64u);
}

TEST_F(GpuDeviceSdma, CompletesSatisfiedRegisterAndMemoryPolls) {
  constexpr uint64_t kSdmaBase = 0x1260;
  constexpr uint64_t kHdpFlush = 0x44000;
  const auto absolute = [](uint64_t reg) { return (kSdmaBase + reg) * 4; };

  write_register_at(absolute(0x0200), (10u << 1) | 0x1001u);
  write_register_at(absolute(0x0201), static_cast<uint32_t>(kSchedulerRingGpu >> 8));
  write_register_at(absolute(0x0202), static_cast<uint32_t>(kSchedulerRingGpu >> 40));
  write_register_at(absolute(0x0207), static_cast<uint32_t>(kSchedulerReadPointerGpu));
  write_register_at(absolute(0x0208), static_cast<uint32_t>(kSchedulerReadPointerGpu >> 32));
  write_register_at(absolute(0x020f), 0x10000000);
  write_register_at(absolute(0x0211), 0x800);

  write_register_at(kHdpFlush, 0x5a);
  store_dword(kRingFencePhysical, 0xa5);

  // Function three is masked equality. The first packet polls MMIO and the
  // second polls GPU memory; both are already satisfied in this synchronous
  // model, as the driver's VM-flush and pipeline-sync packets require.
  store_dword(kSchedulerRingPhysical + 0 * 4, 0x30000008);
  store_dword(kSchedulerRingPhysical + 1 * 4, kHdpFlush);
  store_dword(kSchedulerRingPhysical + 3 * 4, 0x5a);
  store_dword(kSchedulerRingPhysical + 4 * 4, 0xff);
  store_dword(kSchedulerRingPhysical + 6 * 4, 0xb0000008);
  store_qword(kSchedulerRingPhysical + 7 * 4, kRingFenceGpu);
  store_dword(kSchedulerRingPhysical + 9 * 4, 0xa5);
  store_dword(kSchedulerRingPhysical + 10 * 4, 0xff);
  // Hardware accepts writes to registers outside this stage's behavioural
  // model. Like a direct guest MMIO write, the packet is consumed and the
  // absent register continues to read as zero rather than stalling the ring.
  store_dword(kSchedulerRingPhysical + 12 * 4, 14);
  store_dword(kSchedulerRingPhysical + 13 * 4, 0x4280);
  store_dword(kSchedulerRingPhysical + 14 * 4, 0x12345678);
  store_dword(kSchedulerRingPhysical + 15 * 4, 0);

  auto doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{64});
  ring_doorbell(0x800, doorbell);

  EXPECT_EQ(transport_.dword_at(kSchedulerReadPointerPhysical), 64u);
}

TEST_F(GpuDeviceSdma, AppliesEveryRegisterPollComparisonInTheSdmaPacketProcessor) {
  constexpr uint64_t kSdmaBase = 0x1260;
  constexpr uint64_t kHdpFlush = 0x44000;
  constexpr uint32_t kObserved = 5;
  constexpr uint32_t kMask = 0xff;
  constexpr uint32_t kPacketDwords = 6;
  const auto absolute = [](uint64_t reg) { return (kSdmaBase + reg) * 4; };

  write_register_at(absolute(0x0200), (10u << 1) | 0x1001u);
  write_register_at(absolute(0x0201), static_cast<uint32_t>(kSchedulerRingGpu >> 8));
  write_register_at(absolute(0x0202), static_cast<uint32_t>(kSchedulerRingGpu >> 40));
  write_register_at(absolute(0x0207), static_cast<uint32_t>(kSchedulerReadPointerGpu));
  write_register_at(absolute(0x0208), static_cast<uint32_t>(kSchedulerReadPointerGpu >> 32));
  write_register_at(absolute(0x020f), 0x10000000);
  write_register_at(absolute(0x0211), 0x800);
  write_register_at(kHdpFlush, kObserved);

  struct RegisterPollCase {
    uint32_t function;
    uint32_t reference;
  };
  constexpr std::array<RegisterPollCase, 6> kCases = {{
      {.function = 1, .reference = kObserved + 1},
      {.function = 2, .reference = kObserved},
      {.function = 3, .reference = kObserved},
      {.function = 4, .reference = kObserved + 1},
      {.function = 5, .reference = kObserved},
      {.function = 6, .reference = kObserved - 1},
  }};

  for (std::size_t case_index = 0; case_index < kCases.size(); ++case_index) {
    const uint64_t packet_dword = case_index * kPacketDwords;
    store_dword(kSchedulerRingPhysical + (packet_dword + 0) * sizeof(uint32_t),
                8u | (kCases[case_index].function << 28));
    store_dword(kSchedulerRingPhysical + (packet_dword + 1) * sizeof(uint32_t), kHdpFlush);
    store_dword(kSchedulerRingPhysical + (packet_dword + 2) * sizeof(uint32_t), 0);
    store_dword(kSchedulerRingPhysical + (packet_dword + 3) * sizeof(uint32_t),
                kCases[case_index].reference);
    store_dword(kSchedulerRingPhysical + (packet_dword + 4) * sizeof(uint32_t), kMask);
    store_dword(kSchedulerRingPhysical + (packet_dword + 5) * sizeof(uint32_t), 0);
  }

  constexpr uint64_t kWritePointer = kCases.size() * kPacketDwords * sizeof(uint32_t);
  auto doorbell = std::bit_cast<std::array<std::byte, 8>>(kWritePointer);
  ring_doorbell(0x800, doorbell);

  EXPECT_EQ(transport_.dword_at(kSchedulerReadPointerPhysical), kWritePointer);
}

TEST_F(GpuDeviceSdma, FillsGpuMemoryWithARepeatedDword) {
  constexpr uint64_t kSdmaBase = 0x1260;
  const auto absolute = [](uint64_t reg) { return (kSdmaBase + reg) * 4; };

  write_register_at(absolute(0x0200), (10u << 1) | 0x1001u);
  write_register_at(absolute(0x0201), static_cast<uint32_t>(kSchedulerRingGpu >> 8));
  write_register_at(absolute(0x0202), static_cast<uint32_t>(kSchedulerRingGpu >> 40));
  write_register_at(absolute(0x0207), static_cast<uint32_t>(kSchedulerReadPointerGpu));
  write_register_at(absolute(0x0208), static_cast<uint32_t>(kSchedulerReadPointerGpu >> 32));
  write_register_at(absolute(0x020f), 0x10000000);
  write_register_at(absolute(0x0211), 0x800);

  store_dword(kSchedulerRingPhysical + 0 * 4, 11);
  store_qword(kSchedulerRingPhysical + 1 * 4, kRingFenceGpu);
  store_dword(kSchedulerRingPhysical + 3 * 4, 0xa1b2c3d4);
  store_dword(kSchedulerRingPhysical + 4 * 4, 15);
  store_dword(kSchedulerRingPhysical + 5 * 4, 10u << 16);

  auto doorbell = std::bit_cast<std::array<std::byte, 8>>(uint64_t{64});
  ring_doorbell(0x800, doorbell);

  for (uint64_t offset = 0; offset < 16; offset += sizeof(uint32_t)) {
    EXPECT_EQ(transport_.dword_at(kRingFencePhysical + offset), 0xa1b2c3d4u);
  }
  EXPECT_EQ(transport_.dword_at(kSchedulerReadPointerPhysical), 64u);
}

// Delivering an interrupt is three things that mean nothing apart: the entry
// goes into the ring, the write pointer is published so the driver knows it is
// there, and the message is raised so the driver looks. The driver reads that
// pointer out of *guest memory* rather than out of the register, so publishing
// it only to the register would leave the driver waiting forever.
class GpuDeviceDelivery : public GpuDevice {
protected:
  static constexpr uint64_t kRingBase = 0x900000000;
  static constexpr uint64_t kWptrAddress = 0x900001000;
  static constexpr uint64_t kRingBytes = 4096;
  static constexpr uint64_t kInterruptEntryBytes = 32;

  RecordingTransport transport_;
  simdojo::PciDevice::Transport attached_;

  void SetUp() override {
    ASSERT_TRUE(device_.usable());
    attached_ = {.irq = &transport_, .dma = &transport_};
    ASSERT_TRUE(device_.attach_transport(&attached_));
    program_ring();
  }

  void program_ring() {
    // A 4 KiB ring above the 32-bit boundary, its write pointer published just
    // past it, switched on and at a bus address. 4 KiB is 1024 dwords, so the
    // size field is 10.
    write_register_at(0x448c, static_cast<uint32_t>(kRingBase >> 8));
    write_register_at(0x4490, static_cast<uint32_t>(kRingBase >> 40) & 0xff);
    write_register_at(0x4498, static_cast<uint32_t>(kWptrAddress));
    write_register_at(0x4494, static_cast<uint32_t>(kWptrAddress >> 32) & 0xffff);
    write_register_at(0x4480, (10u << 1) | (1u << 0) | (1u << 17) | (2u << 28));
  }

  void TearDown() override { device_.detach_transport(&attached_); }
};

TEST_F(GpuDeviceDelivery, PutsTheEntryInTheRingThenPointsAtItThenRaises) {
  ASSERT_TRUE(device_.deliver_interrupt({.client_id = 0x12, .source_id = 0x34, .data = {0xaaaa}}));

  // The two identifiers the driver looks a handler up by share the first dword.
  EXPECT_EQ(transport_.dword_at(kRingBase) & 0xffff, 0x3412u);
  EXPECT_EQ(transport_.dword_at(kRingBase + 16), 0xaaaau) << "the source's own data";
  // Everything describing work this device does not run stays zero.
  EXPECT_EQ(transport_.dword_at(kRingBase + 4), 0u) << "timestamp";
  EXPECT_EQ(transport_.dword_at(kRingBase + 12), 0u) << "process and node";

  EXPECT_EQ(transport_.dword_at(kWptrAddress), 32u)
      << "the driver reads the write pointer from memory, not from the register";
  ASSERT_EQ(transport_.triggered.size(), 1u);
  EXPECT_EQ(transport_.triggered[0], 0u) << "the one vector the device advertises";
}

// The write pointer says where the next entry goes, so an entry must land at
// it. A device that always wrote to the start of the ring would pass every
// other test here.
TEST_F(GpuDeviceDelivery, PlacesEachEntryAtTheCurrentWritePointer) {
  ASSERT_TRUE(device_.deliver_interrupt({.client_id = 1, .source_id = 1}));
  ASSERT_TRUE(device_.deliver_interrupt({.client_id = 2, .source_id = 2}));

  EXPECT_EQ(transport_.dword_at(kRingBase) & 0xffff, 0x0101u) << "the first entry stays put";
  EXPECT_EQ(transport_.dword_at(kRingBase + 32) & 0xffff, 0x0202u)
      << "the second follows it rather than overwriting it";
  EXPECT_EQ(transport_.dword_at(kWptrAddress), 64u);
  EXPECT_EQ(read_register_at(0x4488), 64u) << "the register shadows the published pointer";
}

// The write pointer is a register the guest can write, and this is where its
// value becomes an address. The driver's own pointer shadows sit immediately
// after the ring, so an entry placed at an unaligned offset would run off the
// end and overwrite exactly the words the interrupt protocol depends on.
TEST_F(GpuDeviceDelivery, NeverWritesPastTheRingHoweverTheGuestSetsThePointer) {
  for (const uint32_t hostile : {static_cast<uint32_t>(kRingBytes - 1),
                                 static_cast<uint32_t>(kRingBytes - 4), 0xffffffffu, 0x7u}) {
    // All three, so a failure names the pointer that actually caused it rather
    // than re-reporting an earlier iteration's write against this one's value.
    transport_.memory.clear();
    transport_.writes.clear();
    transport_.triggered.clear();
    write_register_at(0x4488, hostile);

    ASSERT_TRUE(device_.deliver_interrupt({.client_id = 1, .source_id = 2}))
        << "pointer " << hostile;

    // Checked per write rather than by highest address touched: the driver puts
    // its pointer shadows immediately after the ring, so an entry that overran
    // would land on exactly the address the pointer is legitimately published
    // to, and the two are indistinguishable by address alone.
    for (const auto &[address, length] : transport_.writes) {
      if (length != kInterruptEntryBytes) {
        continue;
      }
      EXPECT_LE(address + length, kRingBase + kRingBytes)
          << "an entry written at " << std::hex << address << " with the pointer set to " << hostile
          << " runs past the ring, onto the driver's own pointer shadows";
    }
  }
}

TEST_F(GpuDeviceDelivery, WrapsTheWritePointerAtTheEndOfTheRing) {
  constexpr std::size_t kEntries = kRingBytes / 32;
  for (std::size_t i = 0; i < kEntries; ++i) {
    ASSERT_TRUE(device_.deliver_interrupt({.client_id = 1, .source_id = 2}));
  }

  EXPECT_EQ(transport_.dword_at(kWptrAddress), 0u) << "a filled ring wraps to its start";
  EXPECT_EQ(transport_.triggered.size(), kEntries);
}

// Nothing partial: a delivery that cannot finish must not leave a pointer
// naming an entry that was never written, which the driver would decode out of
// whatever the ring happened to contain.
// The registers are separate and the driver writes them in whatever order it
// likes, so a ring can be switched on before its base has been given. Zero
// there means unset, not "the ring is at address zero" -- delivering anyway
// would write entries over whatever the guest keeps in its first page.
TEST_F(GpuDeviceDelivery, DeclinesARingEnabledBeforeItsBaseWasProgrammed) {
  write_register_at(0x448c, 0);
  write_register_at(0x4490, 0);
  ASSERT_EQ(device_.interrupt_ring().base, 0u) << "the base must be unset for this to prove it";
  ASSERT_TRUE(device_.interrupt_ring().enabled) << "and the ring still switched on";

  EXPECT_FALSE(device_.deliver_interrupt({.client_id = 0x12, .source_id = 0x34, .data = {0xaaaa}}));
  EXPECT_TRUE(transport_.writes.empty()) << "an entry was written into guest-physical zero";
  EXPECT_TRUE(transport_.triggered.empty()) << "a message was raised for an entry never written";
}

TEST_F(GpuDeviceDelivery, PublishesNothingWhenGuestMemoryCannotBeReached) {
  transport_.refuse_writes = true;

  EXPECT_FALSE(device_.deliver_interrupt({.client_id = 1, .source_id = 2}));

  EXPECT_TRUE(transport_.triggered.empty()) << "a message with nothing behind it";
  EXPECT_EQ(transport_.dword_at(kWptrAddress), 0u);
}

// Publishing the pointer is the second of two writes. If it fails, the entry is
// already in the ring -- harmless, since nothing points at it -- but the message
// must not go out, or the driver would read a pointer that never moved.
TEST_F(GpuDeviceDelivery, RaisesNothingWhenThePointerCannotBePublished) {
  transport_.writes_before_refusing = 1;

  EXPECT_FALSE(device_.deliver_interrupt({.client_id = 1, .source_id = 2}));

  EXPECT_TRUE(transport_.triggered.empty());
  EXPECT_EQ(transport_.dword_at(kWptrAddress), 0u) << "the pointer never moved";
  EXPECT_NE(transport_.dword_at(kRingBase), 0u)
      << "the entry stays in the ring with nothing pointing at it, to be overwritten";
}

// The third outcome the contract describes: everything is published and only
// the message fails. The entry and the pointer must stay, so the next message
// covers them -- rolling them back would lose an entry the driver may already
// have seen.
TEST_F(GpuDeviceDelivery, LeavesTheEntryPublishedWhenTheMessageIsRefused) {
  transport_.refuse_trigger = true;

  EXPECT_FALSE(device_.deliver_interrupt({.client_id = 1, .source_id = 2}));

  EXPECT_EQ(transport_.dword_at(kWptrAddress), 32u) << "the pointer stays advanced";
  EXPECT_NE(transport_.dword_at(kRingBase), 0u) << "and the entry stays in the ring";
}

// A ring may be switched on with messages switched off, in which case entries
// accumulate for the driver to find when it next looks. The device decodes that
// field, so it has to act on it.
TEST_F(GpuDeviceDelivery, FillsTheRingWithoutRaisingWhenMessagesAreOff) {
  write_register_at(0x4480, (10u << 1) | (1u << 0) | (2u << 28));

  EXPECT_TRUE(device_.deliver_interrupt({.client_id = 1, .source_id = 2}));

  EXPECT_NE(transport_.dword_at(kRingBase), 0u) << "the entry is written";
  EXPECT_EQ(transport_.dword_at(kWptrAddress), 32u) << "and pointed at";
  EXPECT_TRUE(transport_.triggered.empty()) << "but no message goes out";
}

// A ring the driver has not switched on, one whose addresses are in a space
// this device cannot resolve, one too small to hold an entry, or one with
// nowhere to publish its pointer, is declined rather than written to: such an
// address still names a real guest page and would quietly corrupt it.
TEST_F(GpuDeviceDelivery, DeclinesARingItCannotSafelyWriteTo) {
  write_register_at(0x4480, (10u << 1) | (1u << 17) | (2u << 28));
  EXPECT_FALSE(device_.deliver_interrupt({.client_id = 1, .source_id = 2})) << "not enabled";

  write_register_at(0x4480, (10u << 1) | (1u << 0) | (1u << 17) | (4u << 28));
  EXPECT_FALSE(device_.deliver_interrupt({.client_id = 1, .source_id = 2})) << "translated";

  write_register_at(0x4480, (2u << 1) | (1u << 0) | (1u << 17) | (2u << 28));
  EXPECT_FALSE(device_.deliver_interrupt({.client_id = 1, .source_id = 2}))
      << "a ring too small to hold one entry";

  // The size field is five bits, so a guest can ask for a ring far larger than
  // the sixteen-bit write-pointer field can name. Accepting one would have the
  // device wrap at the field width while the driver wrapped at the ring size,
  // and the driver would decode the never-written remainder as entries.
  write_register_at(0x4480, (17u << 1) | (1u << 0) | (1u << 17) | (2u << 28));
  EXPECT_FALSE(device_.deliver_interrupt({.client_id = 1, .source_id = 2}))
      << "a ring with entries the write pointer could never name";

  write_register_at(0x4480, (10u << 1) | (1u << 0) | (1u << 17) | (2u << 28));
  write_register_at(0x4498, 0);
  write_register_at(0x4494, 0);
  EXPECT_FALSE(device_.deliver_interrupt({.client_id = 1, .source_id = 2}))
      << "nowhere to publish the pointer";

  EXPECT_TRUE(transport_.triggered.empty());
  EXPECT_TRUE(transport_.memory.empty()) << "nothing was written anywhere";
}

// Without a transport there is no guest to reach and no line to raise, which is
// the state between construction and being served.
TEST_F(GpuDeviceDelivery, DeclinesWithNoTransportAttached) {
  // A transport that can raise a line but cannot reach memory, and then one
  // that can reach memory but cannot raise a line. Neither can deliver, and
  // neither may write a partial entry on the way to finding that out.
  device_.detach_transport(&attached_);
  simdojo::PciDevice::Transport no_engine{.irq = &transport_, .dma = nullptr};
  ASSERT_TRUE(device_.attach_transport(&no_engine));
  EXPECT_FALSE(device_.deliver_interrupt({.client_id = 1, .source_id = 2}));

  ASSERT_TRUE(device_.detach_transport(&no_engine));
  simdojo::PciDevice::Transport no_sink{.irq = nullptr, .dma = &transport_};
  ASSERT_TRUE(device_.attach_transport(&no_sink));
  EXPECT_FALSE(device_.deliver_interrupt({.client_id = 1, .source_id = 2}));
  ASSERT_TRUE(device_.detach_transport(&no_sink));
  ASSERT_TRUE(device_.attach_transport(&attached_));

  EXPECT_TRUE(transport_.memory.empty());
}

// A device serves one transport at a time. An unchecked store let a second one
// take a live device from the first silently: the first kept its pointers and
// its belief that it owned the function, while every interrupt the device
// raised went to the second. The refusal is what makes that a reported error at
// the moment it happens rather than a device that answers reads for a transport
// nobody is behind.
TEST_F(GpuDeviceDelivery, RefusesASecondTransportAndKeepsServingTheFirst) {
  // SetUp already attached transport_, so the device is spoken for.
  RecordingTransport interloper;
  simdojo::PciDevice::Transport rival{.irq = &interloper, .dma = &interloper};
  EXPECT_FALSE(device_.attach_transport(&rival))
      << "a second transport was allowed to take an attached device";
  EXPECT_FALSE(device_.detach_transport(&rival))
      << "a transport that never won the device was allowed to release it";
  EXPECT_TRUE(device_.transport_attached());

  ASSERT_TRUE(device_.deliver_interrupt({.client_id = 0x12, .source_id = 0x34}));
  EXPECT_TRUE(interloper.memory.empty()) << "the refused transport was still reached";
  EXPECT_TRUE(interloper.triggered.empty()) << "the refused transport was still raised";
  EXPECT_FALSE(transport_.memory.empty()) << "the owning transport stopped being served";

  // And the device is attachable again once its owner lets go, so a refusal is
  // not a device permanently locked to a transport that has gone away.
  EXPECT_TRUE(device_.detach_transport(&attached_));
  EXPECT_FALSE(device_.transport_attached());
  EXPECT_TRUE(device_.attach_transport(&rival));
  EXPECT_TRUE(device_.detach_transport(&rival));
}

// The HDP flush is a write to a hole the bus reserves rather than to any block's
// register. An unmodelled register drops writes and reads back zero, so reading
// back what was written is a positive check that the hole is answered -- and
// unlike inspecting the unmodelled report, it cannot pass because the report
// happens to be empty or its formatting changed.
TEST_F(GpuDevice, AcceptsTheHdpFlushTheDriverIssues) {
  ASSERT_TRUE(device_.usable());
  constexpr uint64_t kFlushHole = 0x44000;

  write_register_at(kFlushHole, 0xdeadbeef);

  EXPECT_EQ(read_register_at(kFlushHole), 0xdeadbeefu)
      << "the flush hole is not modelled, so the write was dropped";
}

// The pinned driver binds every NBIF revision from 7.11.0 through 7.11.3 to
// the same implementation and remap. Exact matching therefore needs one row
// for each supported revision rather than treating the .0 profile as a range.
TEST(GpuDeviceHdpFlush, AnswersTheHoleForEverySupportedNbifRevision) {
  for (const uint8_t revision : {uint8_t{0}, uint8_t{1}, uint8_t{2}, uint8_t{3}}) {
    rocjitsu::GpuPciDeviceSpec spec = configured_spec();
    bool changed_revision = false;
    for (rocjitsu::IpBlock &block : spec.discovery.blocks) {
      if (block.hardware_id == rocjitsu::IpHardwareId::Nbif) {
        block.revision = revision;
        changed_revision = true;
      }
    }
    ASSERT_TRUE(changed_revision) << "the profile has no NBIF record to exercise";

    rocjitsu::GpuPciDevice device("revision", spec, nullptr);
    ASSERT_TRUE(device.usable());
    constexpr uint64_t kFlushHole = 0x44000;
    auto written = std::bit_cast<std::array<std::byte, 4>>(uint32_t{0xdeadbeef});
    ASSERT_EQ(device.bar_access(rocjitsu::GpuPciDevice::kRegisterBar, written, kFlushHole,
                                /*write=*/true),
              4);

    std::array<std::byte, 4> read_back{};
    ASSERT_EQ(device.bar_access(rocjitsu::GpuPciDevice::kRegisterBar, read_back, kFlushHole,
                                /*write=*/false),
              4);
    EXPECT_EQ(std::bit_cast<uint32_t>(read_back), 0xdeadbeefu)
        << "NBIF 7.11." << static_cast<uint16_t>(revision);
  }
}

// The driver's NBIF selection uses the complete IP version. It groups 7.11.0
// through 7.11.3 under this remap, but a revision beyond those exact arms must
// not silently inherit their flush hole.
TEST(GpuDeviceHdpFlush, DoesNotAnswerAHoleForAnUnknownNbifRevision) {
  rocjitsu::GpuPciDeviceSpec spec = configured_spec();
  bool changed_revision = false;
  for (rocjitsu::IpBlock &block : spec.discovery.blocks) {
    if (block.hardware_id == rocjitsu::IpHardwareId::Nbif) {
      ASSERT_EQ(block.revision, 0);
      block.revision = 4;
      changed_revision = true;
    }
  }
  ASSERT_TRUE(changed_revision) << "the profile has no NBIF record to exercise";

  rocjitsu::GpuPciDevice mismatched("mismatched", spec, nullptr);
  ASSERT_TRUE(mismatched.usable()) << "an unknown bus hole does not invalidate the whole device";
  constexpr uint64_t kFlushHole = 0x44000;
  auto written = std::bit_cast<std::array<std::byte, 4>>(uint32_t{0xdeadbeef});
  ASSERT_EQ(mismatched.bar_access(rocjitsu::GpuPciDevice::kRegisterBar, written, kFlushHole,
                                  /*write=*/true),
            4);

  std::array<std::byte, 4> read_back{};
  ASSERT_EQ(mismatched.bar_access(rocjitsu::GpuPciDevice::kRegisterBar, read_back, kFlushHole,
                                  /*write=*/false),
            4);
  EXPECT_EQ(std::bit_cast<uint32_t>(read_back), 0u)
      << "NBIF 7.11.4 was answered with the flush hole verified only through 7.11.3";
}

// The driver asks the bus for one vector of any kind and treats not getting one
// as fatal rather than as doing without: `pci_alloc_irq_vectors` returning
// negative fails the interrupt block's initialization, which fails the probe.
// So a function advertising no interrupt capability at all is refused before it
// has raised, or failed to raise, anything.
TEST_F(GpuDevice, AdvertisesAnInterruptTheDriverCanAllocate) {
  const simdojo::InterruptSpec interrupts = device_.interrupts();

  EXPECT_NE(interrupts.kind, simdojo::InterruptKind::None)
      << "a device with no interrupt capability fails the driver's probe";
  // The kind is deliberately not asserted: this stage advertises a pin and a
  // later one advertises MSI-X, and freezing the kind here would only record
  // which stage this is. That the transport can actually advertise whichever
  // kind is asked for is checked next to the transport, in bus_plan_test: this
  // device is modelled whether or not the vfio-user backend is built, so a test
  // here that called into it would leave the whole suite unlinkable without it.
}

// The message table is memory the guest writes to say where an interrupt should
// be delivered, so it needs a BAR of its own that traps. Putting it in a corner
// of the register aperture would let a table entry and a register land on the
// same address, and the two are read by entirely different machinery.
TEST_F(GpuDevice, CarriesTheMessageTableInATrappedBarOfItsOwn) {
  ASSERT_TRUE(device_.usable());
  const simdojo::InterruptSpec interrupts = device_.interrupts();
  ASSERT_EQ(interrupts.kind, simdojo::InterruptKind::MsiX);

  const simdojo::BarSpec *table = bar(interrupts.table_bar);
  ASSERT_NE(table, nullptr) << "the table is advertised in a BAR that does not exist";
  EXPECT_LT(table->backing_fd, 0) << "a mapped table would let the guest change it unobserved";
  EXPECT_TRUE(table->mmap_areas.empty());
  EXPECT_NE(interrupts.table_bar, rocjitsu::GpuPciDevice::kRegisterBar);

  // Both structures have to fit, and the pending bits must not start inside the
  // table: one vector's entry is 16 bytes and its pending bit is in the first 8.
  const uint64_t table_bytes = interrupts.vectors * 16;
  EXPECT_LE(interrupts.table_offset + table_bytes, interrupts.pending_offset);
  EXPECT_LE(interrupts.pending_offset + 8, table->size);
}

// The table is plain storage, but it has to be storage: a write the device
// dropped would leave the guest believing it had programmed a destination.
TEST_F(GpuDevice, RemembersWhatTheGuestWritesIntoTheMessageTable) {
  ASSERT_TRUE(device_.usable());
  const simdojo::InterruptSpec interrupts = device_.interrupts();
  const int table_bar = interrupts.table_bar;
  auto raw = std::bit_cast<std::array<std::byte, 4>>(uint32_t{0xfeedface});

  ASSERT_EQ(device_.bar_access(table_bar, raw, interrupts.table_offset, /*write=*/true), 4);

  std::array<std::byte, 4> read_back{};
  ASSERT_EQ(device_.bar_access(table_bar, read_back, interrupts.table_offset, /*write=*/false), 4);
  EXPECT_EQ(std::bit_cast<uint32_t>(read_back), 0xfeedfaceu);
}

// The driver's PCI table wildcards the device ID and matches on the class, so
// this is the field that decides whether amdgpu attaches at all.
TEST_F(GpuDevice, PresentsTheClassAmdgpuBindsOn) {
  const simdojo::PciId id = device_.pci_id();

  EXPECT_EQ(id.vendor, 0x1002);
  EXPECT_EQ(id.cls, 0x12) << "processing accelerator";
  EXPECT_EQ(id.subcls, 0x00);
}

TEST_F(GpuDevice, ExposesAMappableVideoMemoryAperture) {
  ASSERT_TRUE(device_.usable());
  const simdojo::BarSpec *vram = bar(rocjitsu::GpuPciDevice::kVramBar);

  ASSERT_NE(vram, nullptr);
  EXPECT_EQ(vram->size, kVramBytes);
  EXPECT_TRUE(vram->is_64bit);
  EXPECT_TRUE(vram->prefetch);
  EXPECT_GE(vram->backing_fd, 0) << "the guest must be able to map video memory";
  ASSERT_EQ(vram->mmap_areas.size(), 1u);
  EXPECT_EQ(vram->mmap_areas[0].length, kVramBytes);
}

// Registers must trap even though memory does not: a read has to be answered by
// the model rather than served from a page the guest mapped.
TEST_F(GpuDevice, TrapsEveryRegisterAccess) {
  const simdojo::BarSpec *registers = bar(rocjitsu::GpuPciDevice::kRegisterBar);

  ASSERT_NE(registers, nullptr);
  EXPECT_LT(registers->backing_fd, 0);
  EXPECT_TRUE(registers->mmap_areas.empty());
}

// The aperture has to reach the furthest register the driver reads before
// discovery, or that read silently goes through the indirect window instead.
TEST_F(GpuDevice, SizesTheRegisterApertureToCoverThePreDiscoveryRegisters) {
  const simdojo::BarSpec *registers = bar(rocjitsu::GpuPciDevice::kRegisterBar);

  ASSERT_NE(registers, nullptr);
  EXPECT_GT(registers->size, rocjitsu::byte_offset_of(rocjitsu::MmioRegister::Mp0SmnC2pmsg33));
}

TEST_F(GpuDevice, ReportsItsMemorySizeInMegabytes) {
  EXPECT_EQ(read_register(rocjitsu::MmioRegister::RccConfigMemsize), kVramBytes >> 20);
}

// Reporting no memory, or all-ones, makes the driver abandon discovery before
// it starts.
TEST_F(GpuDevice, NeverReportsAMemorySizeThatAbortsDiscovery) {
  const uint32_t size = read_register(rocjitsu::MmioRegister::RccConfigMemsize);

  EXPECT_NE(size, 0u);
  EXPECT_NE(size, 0xffffffffu);
}

// The driver polls this for up to two seconds waiting for firmware to finish
// starting. An emulated device has nothing to wait for.
TEST_F(GpuDevice, ReportsFirmwareInitialisationAlreadyFinished) {
  EXPECT_NE(read_register(rocjitsu::MmioRegister::Mp0SmnC2pmsg33) & rocjitsu::kFirmwareInitDoneBit,
            0u);
}

// Zero tells the driver the discovery table is not published through these
// registers, sending it to the top of video memory instead.
TEST_F(GpuDevice, PublishesNoDiscoveryTableThroughTheScratchRegisters) {
  EXPECT_EQ(read_register(rocjitsu::MmioRegister::DriverScratch0), 0u);
  EXPECT_EQ(read_register(rocjitsu::MmioRegister::DriverScratch1), 0u);
  EXPECT_EQ(read_register(rocjitsu::MmioRegister::DriverScratch2), 0u);
}

TEST_F(GpuDevice, RecordsARegisterItDoesNotModel) {
  constexpr uint64_t kUnmodelled = 0x28a04;
  std::array<std::byte, 4> raw{};

  ASSERT_EQ(device_.bar_access(rocjitsu::GpuPciDevice::kRegisterBar, raw, kUnmodelled,
                               /*write=*/false),
            4);

  EXPECT_EQ(std::bit_cast<uint32_t>(raw), 0u) << "an unmodelled register reads as absent hardware";
  EXPECT_NE(trace_.unmodeled_report().find("0x00028a04"), std::string::npos);
}

TEST_F(GpuDevice, RejectsARegisterAccessNoHardwareWouldAnswer) {
  std::array<std::byte, 2> narrow{};
  EXPECT_LT(device_.bar_access(rocjitsu::GpuPciDevice::kRegisterBar, narrow, 0, false), 0);

  std::array<std::byte, 4> unaligned{};
  EXPECT_LT(device_.bar_access(rocjitsu::GpuPciDevice::kRegisterBar, unaligned, 2, false), 0);
}

TEST_F(GpuDevice, KeepsDoorbellWritesForTheCommandProcessorToFind) {
  constexpr uint64_t kDoorbell = 0x1000;
  auto written = std::bit_cast<std::array<std::byte, 8>>(uint64_t{0x1234});

  ASSERT_EQ(device_.bar_access(rocjitsu::GpuPciDevice::kDoorbellBar, written, kDoorbell, true), 8);

  std::array<std::byte, 8> read{};
  ASSERT_EQ(device_.bar_access(rocjitsu::GpuPciDevice::kDoorbellBar, read, kDoorbell, false), 8);
  EXPECT_EQ(std::bit_cast<uint64_t>(read), 0x1234u);
}

// Which GPU is presented comes from the config, so a different part is a
// different config file rather than a different class.
TEST(GpuDeviceFromConfig, PresentsTheConfiguredIdentityAndApertures) {
  rocjitsu::config::KfdDeviceConfig device;
  device.gfx_target_version = kModelledTarget;
  device.vendor_id = 0x1002;
  device.device_id = 0x74a1;
  device.pci_revision_id = 0x02;
  device.local_mem_size = 192ULL * 1024 * 1024 * 1024;

  rocjitsu::config::PciDeviceConfig pci;
  pci.class_code = 0x030000;
  pci.subsystem_vendor_id = 0x1028;
  pci.subsystem_id = 0x0c34;
  pci.vram_aperture_bytes = 32ULL * 1024 * 1024;
  pci.doorbell_aperture_bytes = 4ULL * 1024 * 1024;
  pci.register_aperture_bytes = 1024ULL * 1024;

  rocjitsu::GpuPciDevice configured("configured", rocjitsu::gpu_pci_spec_from_config(device, pci),
                                    nullptr);
  ASSERT_TRUE(configured.usable());

  const simdojo::PciId id = configured.pci_id();
  EXPECT_EQ(id.device, 0x74a1);
  EXPECT_EQ(id.cls, 0x03) << "the configured class must reach the bus, not a built-in one";
  EXPECT_EQ(id.subsys_vendor, 0x1028);
  EXPECT_EQ(id.revision, 0x02);

  const std::vector<simdojo::BarSpec> bars = configured.bars();
  const auto aperture = [&bars](int index) {
    return std::ranges::find(bars, index, &simdojo::BarSpec::index)->size;
  };
  EXPECT_EQ(aperture(rocjitsu::GpuPciDevice::kVramBar), 32ULL * 1024 * 1024)
      << "a small window onto large memory is the normal case";
  EXPECT_EQ(aperture(rocjitsu::GpuPciDevice::kDoorbellBar), 4ULL * 1024 * 1024);
  EXPECT_EQ(aperture(rocjitsu::GpuPciDevice::kRegisterBar), 1024ULL * 1024);
}

// An unset subsystem follows the device rather than reading as an unrelated one.
TEST(GpuDeviceFromConfig, DefaultsTheSubsystemToTheDeviceItself) {
  rocjitsu::config::KfdDeviceConfig device;
  device.gfx_target_version = kModelledTarget;
  device.vendor_id = 0x1002;
  device.device_id = 0x1250;
  device.local_mem_size = kVramBytes;

  const rocjitsu::GpuPciDeviceSpec spec = rocjitsu::gpu_pci_spec_from_config(device, {});

  EXPECT_EQ(spec.id.subsys_vendor, 0x1002);
  EXPECT_EQ(spec.id.subsys, 0x1250);
}

// A register aperture too small to reach the pre-discovery registers would make
// the driver read them through a window this device does not model, so the
// device refuses rather than answering wrongly.
TEST(GpuDeviceFromConfig, RefusesARegisterApertureThatCannotReachThoseRegisters) {
  rocjitsu::config::KfdDeviceConfig device;
  device.gfx_target_version = kModelledTarget;
  device.local_mem_size = kVramBytes;
  rocjitsu::config::PciDeviceConfig pci;
  pci.register_aperture_bytes = 4096;

  const rocjitsu::GpuPciDevice tiny("tiny", rocjitsu::gpu_pci_spec_from_config(device, pci),
                                    nullptr);

  EXPECT_FALSE(tiny.usable());
}

// A part whose memory is larger than its window is the normal case, and the
// discovery table the driver looks for sits at the top of memory, outside it.
// Reaching that means the indirect window has to work, and has to use the same
// address encoding the driver does.
class GpuDeviceWindow : public ::testing::Test {
public:
  static constexpr uint64_t kMemoryBytes = 8ULL * 1024 * 1024 * 1024;

protected:
  explicit GpuDeviceWindow(uint64_t capacity = kMemoryBytes)
      : gpu_{"gpu", spec(capacity), nullptr} {}

  rocjitsu::GpuPciDevice gpu_;

  static rocjitsu::GpuPciDeviceSpec spec(uint64_t capacity) {
    rocjitsu::config::KfdDeviceConfig device;
    device.gfx_target_version = kModelledTarget;
    device.local_mem_size = capacity;
    rocjitsu::config::PciDeviceConfig pci;
    pci.vram_aperture_bytes = 1024 * 1024;
    return rocjitsu::gpu_pci_spec_from_config(device, pci);
  }

  void write_register(rocjitsu::MmioRegister reg, uint32_t value) {
    auto raw = std::bit_cast<std::array<std::byte, 4>>(value);
    ASSERT_EQ(gpu_.bar_access(rocjitsu::GpuPciDevice::kRegisterBar, raw,
                              rocjitsu::byte_offset_of(reg), /*write=*/true),
              4);
  }

  uint32_t read_register(rocjitsu::MmioRegister reg) {
    std::array<std::byte, 4> raw{};
    EXPECT_EQ(gpu_.bar_access(rocjitsu::GpuPciDevice::kRegisterBar, raw,
                              rocjitsu::byte_offset_of(reg), /*write=*/false),
              4);
    return std::bit_cast<uint32_t>(raw);
  }

  // The sequence amdgpu_device_mm_access uses: the low register carries address
  // bits 0 to 30 with the memory-select bit set on top, and the high register
  // starts at address bit 31.
  void select(uint64_t address) {
    write_register(rocjitsu::MmioRegister::MmIndex, static_cast<uint32_t>(address) | 0x80000000);
    write_register(rocjitsu::MmioRegister::MmIndexHi, static_cast<uint32_t>(address >> 31));
  }
};

// The scratch registers all read zero, which tells the driver the table is at
// the top of memory rather than somewhere the device names. Nothing else checks
// that anything is actually there: a device that answers every register
// correctly and leaves that address empty looks healthy right up until the
// driver reads a zero signature and refuses it.
TEST_F(GpuDeviceWindow, PublishesADiscoveryTableWhereTheRegistersPromiseOne) {
  ASSERT_TRUE(gpu_.usable());
  ASSERT_EQ(read_register(rocjitsu::MmioRegister::DriverScratch2), 0u)
      << "the device names its own discovery address, so it is not at the top of memory";

  select(kMemoryBytes - rocjitsu::GpuPciDevice::kDiscoveryOffsetFromTopOfVram);

  EXPECT_EQ(read_register(rocjitsu::MmioRegister::MmData), 0x28211407u);
}

// The driver never learns the byte count. It reads a count of megabytes out of
// RCC_CONFIG_MEMSIZE and computes the address itself, so a capacity that is not
// a whole number of megabytes has two candidate addresses and only the rounded
// one is ever read. Publishing at the true top instead misses by the remainder,
// and the driver then reports a bad signature rather than a bad address. Some
// of the shipped configs have exactly such a capacity.
class GpuDeviceUnroundedMemory : public GpuDeviceWindow {
protected:
  /// @brief A capacity whose last megabyte is incomplete, by half of one.
  static constexpr uint64_t kUnroundedBytes = kMemoryBytes + 512 * 1024;

  GpuDeviceUnroundedMemory() : GpuDeviceWindow(kUnroundedBytes) {}
};

// The invalidation registers move between versions of the same block, so the
// hardware ID alone does not identify a layout: GC 12.0 puts SEM/REQ/ACK
// 0x10 below where 12.1 does. Answering a 12.0 profile with 12.1's addresses
// leaves the driver polling registers the device never defined, which presents
// as hardware that never completes a flush.
TEST(GpuDeviceFlushes, RefusesAHubVersionWithNoKnownRegisterLayout) {
  rocjitsu::config::KfdDeviceConfig device;
  device.gfx_target_version = kModelledTarget;
  device.local_mem_size = 8ULL * 1024 * 1024 * 1024;

  rocjitsu::GpuPciDeviceSpec spec = rocjitsu::gpu_pci_spec_from_config(device, {});
  ASSERT_TRUE(rocjitsu::GpuPciDevice("baseline", spec, nullptr).usable())
      << "the unmodified profile must be usable, or this proves nothing";

  for (rocjitsu::IpBlock &block : spec.discovery.blocks) {
    if (block.hardware_id == rocjitsu::IpHardwareId::Gc) {
      block.minor = 0; // GC 12.0: a real version, with a different layout.
    }
  }

  const rocjitsu::GpuPciDevice mismatched("mismatched", spec, nullptr);
  EXPECT_FALSE(mismatched.usable())
      << "a hub version with no known layout was answered with another version's addresses";
}

// A revision is part of IP_VERSION too. The current rows were read from GC
// 12.1.0 and MMHUB 4.1.0 headers, so accepting 12.1.1 or 4.1.1 would extend
// those offsets to a version whose layout has not been established.
TEST(GpuDeviceFlushes, RefusesHubRevisionsWithNoKnownRegisterLayout) {
  for (const rocjitsu::IpHardwareId hardware_id :
       {rocjitsu::IpHardwareId::Gc, rocjitsu::IpHardwareId::MmHub}) {
    rocjitsu::config::KfdDeviceConfig device;
    device.gfx_target_version = kModelledTarget;
    device.local_mem_size = 8ULL * 1024 * 1024 * 1024;

    rocjitsu::GpuPciDeviceSpec spec = rocjitsu::gpu_pci_spec_from_config(device, {});
    ASSERT_TRUE(rocjitsu::GpuPciDevice("baseline", spec, nullptr).usable())
        << "the unmodified profile must be usable, or this proves nothing";

    bool changed_revision = false;
    for (rocjitsu::IpBlock &block : spec.discovery.blocks) {
      if (block.hardware_id == hardware_id) {
        ASSERT_EQ(block.revision, 0);
        block.revision = 1;
        changed_revision = true;
      }
    }
    ASSERT_TRUE(changed_revision) << "the profile has no matching hub record to exercise";

    const rocjitsu::GpuPciDevice mismatched("mismatched", spec, nullptr);
    EXPECT_FALSE(mismatched.usable())
        << "hardware id " << static_cast<uint16_t>(hardware_id)
        << " revision 1 was answered with the layout verified only for revision 0";
  }
}

// A hub whose registers fall outside the register aperture cannot be answered
// at all. Publishing the table and reporting usable anyway makes the device
// look correct right up until the driver waits on a flush.
TEST(GpuDeviceFlushes, RefusesWhenAHubFallsOutsideTheRegisterAperture) {
  rocjitsu::config::KfdDeviceConfig device;
  device.gfx_target_version = kModelledTarget;
  device.local_mem_size = 8ULL * 1024 * 1024 * 1024;

  rocjitsu::GpuPciDeviceSpec spec = rocjitsu::gpu_pci_spec_from_config(device, {});
  // Far enough out that engine 0's acknowledge lands past the aperture.
  for (rocjitsu::IpBlock &block : spec.discovery.blocks) {
    if (block.hardware_id == rocjitsu::IpHardwareId::MmHub) {
      block.register_bases.front() = 0x1fc00;
    }
  }

  const rocjitsu::GpuPciDevice unreachable("unreachable", spec, nullptr);
  EXPECT_FALSE(unreachable.usable())
      << "the device published a table and reported usable while its flushes cannot be answered";
}

// A reset republishes the table. Without that, a guest that resets the device
// and re-reads the signature finds whatever the previous guest left there, and
// the failure surfaces as a driver that will not attach for no visible reason.
TEST_F(GpuDeviceWindow, RestoresTheDiscoveryTableOnReset) {
  const uint64_t table_at =
      (static_cast<uint64_t>(read_register(rocjitsu::MmioRegister::RccConfigMemsize)) << 20) -
      rocjitsu::GpuPciDevice::kDiscoveryOffsetFromTopOfVram;

  select(table_at);
  ASSERT_EQ(read_register(rocjitsu::MmioRegister::MmData), 0x28211407u);

  // Corrupt it the only way a guest can: through the indirect window.
  select(table_at);
  write_register(rocjitsu::MmioRegister::MmData, 0xdeadbeefu);
  select(table_at);
  ASSERT_EQ(read_register(rocjitsu::MmioRegister::MmData), 0xdeadbeefu)
      << "the signature was not actually overwritten, so this proves nothing";

  gpu_.reset(simdojo::ResetKind::FunctionLevel);

  select(table_at);
  EXPECT_EQ(read_register(rocjitsu::MmioRegister::MmData), 0x28211407u)
      << "reset left the previous guest's bytes where the driver looks for the table";
}

TEST_F(GpuDeviceUnroundedMemory, PublishesWhereTheReportedCapacityPointsNotAtTheRealTop) {
  ASSERT_TRUE(gpu_.usable());

  const uint64_t reported_top =
      static_cast<uint64_t>(read_register(rocjitsu::MmioRegister::RccConfigMemsize)) << 20;
  ASSERT_LT(reported_top, kUnroundedBytes) << "this capacity does not round down, so it proves "
                                              "nothing about which address is used";

  select(reported_top - rocjitsu::GpuPciDevice::kDiscoveryOffsetFromTopOfVram);
  EXPECT_EQ(read_register(rocjitsu::MmioRegister::MmData), 0x28211407u)
      << "nothing is where the driver computes the address from the capacity it was given";

  select(kUnroundedBytes - rocjitsu::GpuPciDevice::kDiscoveryOffsetFromTopOfVram);
  EXPECT_NE(read_register(rocjitsu::MmioRegister::MmData), 0x28211407u)
      << "the table is at the true top of memory, which the driver never reads";
}

class GpuDeviceIndirectWindow : public GpuDeviceWindow,
                                public ::testing::WithParamInterface<uint64_t> {};

TEST_P(GpuDeviceIndirectWindow, ReachesMemoryTheApertureCannot) {
  const uint64_t address = GetParam();
  ASSERT_TRUE(gpu_.usable());
  constexpr uint32_t kMarker = 0x5a5a1234;

  select(address);
  write_register(rocjitsu::MmioRegister::MmData, kMarker);

  // Point the window somewhere else first, so a stale address cannot pass.
  select(0);
  EXPECT_EQ(read_register(rocjitsu::MmioRegister::MmData), 0u);

  select(address);
  EXPECT_EQ(read_register(rocjitsu::MmioRegister::MmData), kMarker);
}

// Below, at and above the bit where the address splits between the two index
// registers, plus high memory the aperture cannot reach. The last address stays
// clear of the discovery table at the very top, which this test would otherwise
// overwrite.
INSTANTIATE_TEST_SUITE_P(AcrossTheIndexRegisterBoundary, GpuDeviceIndirectWindow,
                         ::testing::Values(0x1000ULL, 0x7ffffffcULL, 0x80000000ULL, 0x80000004ULL,
                                           0x1'0000'0000ULL,
                                           GpuDeviceIndirectWindow::kMemoryBytes - 0x20000));

// Bit 31 of MM_INDEX chooses the space the indirect window addresses. Only the
// memory side is modelled, and answering a register-side request out of the
// framebuffer would hand the driver bytes from an unrelated address while
// looking like working hardware.
TEST(GpuDeviceIndirectWindow, RefusesAnAccessToTheRegisterSpace) {
  rocjitsu::RegisterSymbols symbols;
  rocjitsu::BarAccessTrace trace(symbols);
  rocjitsu::GpuPciDevice gpu("gpu", configured_spec(), &trace);
  ASSERT_TRUE(gpu.usable());

  const auto write_reg = [&](rocjitsu::MmioRegister reg, uint32_t value) {
    std::array<std::byte, 4> raw{};
    std::memcpy(raw.data(), &value, sizeof(value));
    ASSERT_EQ(gpu.bar_access(rocjitsu::GpuPciDevice::kRegisterBar, raw,
                             rocjitsu::byte_offset_of(reg), /*write=*/true),
              4);
  };

  // Memory select clear: a register-space request this stage does not model.
  write_reg(rocjitsu::MmioRegister::MmIndexHi, 0);
  write_reg(rocjitsu::MmioRegister::MmIndex, 0x1000);
  std::array<std::byte, 4> raw{};
  EXPECT_LT(gpu.bar_access(rocjitsu::GpuPciDevice::kRegisterBar, raw,
                           rocjitsu::byte_offset_of(rocjitsu::MmioRegister::MmData),
                           /*write=*/false),
            0);
  EXPECT_TRUE(trace.unmodeled_report().empty())
      << "a refused access was reported as a register still to be modeled";

  // Memory select set: the modelled path, which answers.
  write_reg(rocjitsu::MmioRegister::MmIndex, 0x80000000u | 0x1000u);
  EXPECT_EQ(gpu.bar_access(rocjitsu::GpuPciDevice::kRegisterBar, raw,
                           rocjitsu::byte_offset_of(rocjitsu::MmioRegister::MmData),
                           /*write=*/false),
            4);
}

// The driver reads the capacity as a count of megabytes, so a finer size rounds
// down. The device stays usable: what matters is that the reported value is the
// one it derives everything else from, so the guest is never pointed at an
// address the device did not use. The remainder is simply unreachable.
TEST(GpuDeviceFromConfig, ReportsACapacityTheDriverCanReadAndStaysUsable) {
  rocjitsu::config::KfdDeviceConfig device;
  device.gfx_target_version = kModelledTarget;
  device.local_mem_size = kVramBytes + 512;

  const rocjitsu::GpuPciDevice odd("odd", rocjitsu::gpu_pci_spec_from_config(device, {}), nullptr);

  EXPECT_TRUE(odd.usable())
      << "an unreportable remainder is not a reason to refuse the whole device";
}

TEST(GpuDeviceFromConfig, RefusesAnApertureThatIsNotALegalBarSize) {
  rocjitsu::config::KfdDeviceConfig device;
  device.gfx_target_version = kModelledTarget;
  device.local_mem_size = kVramBytes;
  rocjitsu::config::PciDeviceConfig pci;
  pci.doorbell_aperture_bytes = 3 * 1024 * 1024;

  const rocjitsu::GpuPciDevice odd("odd", rocjitsu::gpu_pci_spec_from_config(device, pci), nullptr);

  EXPECT_FALSE(odd.usable()) << "a BAR must be a power of two";
}

// The smallest accepted register aperture has to reach every register the
// device claims to answer, and the device has to actually answer them: a
// constant comparison would still pass if reset_registers() stopped defining
// one, which is the way this breaks in practice.
TEST(GpuDeviceFromConfig, AcceptsTheMinimumApertureAndReachesEveryPreDiscoveryRegister) {
  EXPECT_GT(rocjitsu::GpuPciDevice::kMinRegisterApertureBytes,
            rocjitsu::byte_offset_of(rocjitsu::MmioRegister::IpDiscoveryVersion));
  EXPECT_EQ(rocjitsu::GpuPciDevice::kMinRegisterApertureBytes &
                (rocjitsu::GpuPciDevice::kMinRegisterApertureBytes - 1),
            0u)
      << "the advertised minimum must itself be a legal BAR size";

  rocjitsu::config::KfdDeviceConfig device;
  device.gfx_target_version = kModelledTarget;
  device.local_mem_size = kVramBytes;
  rocjitsu::config::PciDeviceConfig pci;
  pci.register_aperture_bytes = rocjitsu::GpuPciDevice::kMinRegisterApertureBytes;

  rocjitsu::RegisterSymbols symbols;
  rocjitsu::BarAccessTrace trace(symbols);
  rocjitsu::GpuPciDevice gpu("gpu", rocjitsu::gpu_pci_spec_from_config(device, pci), &trace);
  ASSERT_TRUE(gpu.usable());

  for (const rocjitsu::MmioRegister reg :
       {rocjitsu::MmioRegister::RccConfigMemsize, rocjitsu::MmioRegister::Mp0SmnC2pmsg33,
        rocjitsu::MmioRegister::DriverScratch0, rocjitsu::MmioRegister::DriverScratch1,
        rocjitsu::MmioRegister::DriverScratch2, rocjitsu::MmioRegister::MmIndex,
        rocjitsu::MmioRegister::MmIndexHi, rocjitsu::MmioRegister::IpDiscoveryVersion}) {
    std::array<std::byte, 4> raw{};
    EXPECT_EQ(gpu.bar_access(rocjitsu::GpuPciDevice::kRegisterBar, raw,
                             rocjitsu::byte_offset_of(reg), /*write=*/false),
              4)
        << "register at " << rocjitsu::byte_offset_of(reg);
  }
  EXPECT_TRUE(trace.unmodeled_report().empty())
      << "a register the device claims to answer before discovery read as absent:\n"
      << trace.unmodeled_report();
}

// Most parts report a capacity that is not a power of two and say nothing about
// the bus, so an omitted section has to yield a BAR that is legal anyway.
TEST(GpuDeviceFromConfig, DerivesALegalApertureForAConfigWithNoBusSection) {
  rocjitsu::config::KfdDeviceConfig device;
  device.gfx_target_version = kModelledTarget;
  device.local_mem_size = 192ULL * 1024 * 1024 * 1024;

  const rocjitsu::GpuPciDeviceSpec spec = rocjitsu::gpu_pci_spec_from_config(device, {});
  const rocjitsu::GpuPciDevice gpu("gpu", spec, nullptr);

  EXPECT_EQ(spec.vram_aperture_bytes, 256ULL * 1024 * 1024);
  EXPECT_TRUE(gpu.usable()) << "a capacity that is not a power of two must still be presentable";
}

// A config with no usable device description must not produce a device that
// claims to be presentable; the transport would reject it moments later.
TEST(GpuDeviceFromConfig, RefusesAConfigThatDescribesNoMemory) {
  const rocjitsu::GpuPciDeviceSpec spec = rocjitsu::gpu_pci_spec_from_config({}, {});

  EXPECT_EQ(spec.vram_aperture_bytes, 0u) << "no memory has no legal aperture";

  const rocjitsu::GpuPciDevice gpu("gpu", spec, nullptr);
  EXPECT_FALSE(gpu.usable());
}

// PCI sets a floor on a memory BAR, and a device that accepts less would be
// refused by the transport instead, after it had already reported itself fine.
class GpuDeviceApertureBoundary : public ::testing::TestWithParam<uint64_t> {};

TEST_P(GpuDeviceApertureBoundary, AgreesWithThePciMinimumForAMemoryBar) {
  const uint64_t aperture = GetParam();
  rocjitsu::config::KfdDeviceConfig device;
  device.gfx_target_version = kModelledTarget;
  device.local_mem_size = 64 * 1024 * 1024;
  rocjitsu::config::PciDeviceConfig pci;
  pci.vram_aperture_bytes = aperture;

  const rocjitsu::GpuPciDevice gpu("gpu", rocjitsu::gpu_pci_spec_from_config(device, pci), nullptr);

  EXPECT_EQ(gpu.usable(), aperture >= rocjitsu::GpuPciDevice::kMinMemoryBarBytes);
}

INSTANTIATE_TEST_SUITE_P(AroundThePciFloor, GpuDeviceApertureBoundary,
                         ::testing::Values(1ULL, 8ULL, 16ULL, 32ULL));

// The driver reads memory size as megabytes in a 32-bit register and rejects
// zero and all-ones, so not every byte capacity can be presented at all.
struct CapacityCase {
  uint64_t bytes;
  bool presentable;
  const char *why;
};

class GpuDeviceCapacityBoundary : public ::testing::TestWithParam<CapacityCase> {};

TEST_P(GpuDeviceCapacityBoundary, OnlyAcceptsACapacityTheDriverCanRead) {
  const CapacityCase &capacity = GetParam();
  rocjitsu::config::KfdDeviceConfig device;
  device.gfx_target_version = kModelledTarget;
  device.local_mem_size = capacity.bytes;

  const rocjitsu::GpuPciDevice gpu("gpu", rocjitsu::gpu_pci_spec_from_config(device, {}), nullptr);

  EXPECT_EQ(gpu.usable(), capacity.presentable) << capacity.why;
}

INSTANTIATE_TEST_SUITE_P(
    AroundTheMegabyteEncoding, GpuDeviceCapacityBoundary,
    ::testing::Values(
        CapacityCase{16, false, "sixteen bytes rounds to zero megabytes"},
        CapacityCase{(1ULL << 20) - 1, false, "just under a megabyte still rounds to zero"},
        CapacityCase{1ULL << 20, true, "exactly one megabyte is the smallest sayable size"},
        CapacityCase{static_cast<uint64_t>(0xffffffffULL) << 20, false,
                     "all-ones megabytes is how the driver spells no memory"},
        CapacityCase{static_cast<uint64_t>(0x100000000ULL) << 20, false,
                     "beyond the register, which would narrow to zero"}));

// A part smaller than the default window gets the largest legal window that
// fits inside it, rather than one larger than its own memory.
TEST(GpuDeviceFromConfig, CapsTheDerivedApertureAtTheMemoryItHas) {
  rocjitsu::config::KfdDeviceConfig device;
  device.gfx_target_version = kModelledTarget;
  device.local_mem_size = 100ULL * 1024 * 1024;

  const rocjitsu::GpuPciDeviceSpec spec = rocjitsu::gpu_pci_spec_from_config(device, {});

  EXPECT_EQ(spec.vram_aperture_bytes, 64ULL * 1024 * 1024);
  EXPECT_LE(spec.vram_aperture_bytes, device.local_mem_size);
}

// Reset returns everything a client could have changed to power-on state.
TEST_F(GpuDevice, RestoresPowerOnRegisterStateOnReset) {
  const rocjitsu::amdgpu::AddressSpaceHandle old_gart = soc_.gpu_vm().gart_address_space();
  ASSERT_TRUE(old_gart);
  const uint64_t old_gart_epoch = soc_.gpu_vm().lookup(old_gart)->translation_epoch;
  auto changed = std::bit_cast<std::array<std::byte, 4>>(uint32_t{0});
  ASSERT_EQ(device_.bar_access(rocjitsu::GpuPciDevice::kRegisterBar, changed,
                               rocjitsu::byte_offset_of(rocjitsu::MmioRegister::DriverScratch0),
                               /*write=*/true),
            4);
  auto marker = std::bit_cast<std::array<std::byte, 4>>(uint32_t{0xdeadbeef});
  ASSERT_EQ(device_.bar_access(rocjitsu::GpuPciDevice::kRegisterBar, marker,
                               rocjitsu::byte_offset_of(rocjitsu::MmioRegister::DriverScratch1),
                               /*write=*/true),
            4);

  device_.reset(simdojo::ResetKind::LostConnection);

  EXPECT_EQ(read_register(rocjitsu::MmioRegister::DriverScratch1), 0u);
  EXPECT_NE(read_register(rocjitsu::MmioRegister::Mp0SmnC2pmsg33) & rocjitsu::kFirmwareInitDoneBit,
            0u);
  const rocjitsu::amdgpu::AddressSpaceHandle new_gart = soc_.gpu_vm().gart_address_space();
  EXPECT_TRUE(new_gart);
  EXPECT_EQ(new_gart, old_gart);
  ASSERT_TRUE(soc_.gpu_vm().lookup(new_gart));
  EXPECT_FALSE(soc_.gpu_vm().lookup(new_gart)->ready);
  EXPECT_EQ(soc_.gpu_vm().lookup(new_gart)->translation_epoch, old_gart_epoch + 1);
}

} // namespace

// A block's registers move between versions of that block, and the driver binds
// an entirely different implementation per version -- OSSSYS 7.1 gets ih_v7_0
// while 4.4 gets vega20_ih. So answering one version's ring addresses for
// another does not model the hardware slightly wrong; it programs registers the
// device never reads, which presents as a GPU that accepts an interrupt ring and
// then never reports anything through it. Refusing is what makes that a message
// instead of a silence.
TEST(GpuDeviceInterruptRing, RefusesAnOsssysVersionWithNoKnownLayout) {
  rocjitsu::config::KfdDeviceConfig device;
  device.gfx_target_version = kModelledTarget;
  device.local_mem_size = 8ULL * 1024 * 1024 * 1024;

  rocjitsu::GpuPciDeviceSpec spec = rocjitsu::gpu_pci_spec_from_config(device, {});
  ASSERT_TRUE(rocjitsu::GpuPciDevice("baseline", spec, nullptr).usable())
      << "the unmodified profile must be usable, or this proves nothing";

  for (rocjitsu::IpBlock &block : spec.discovery.blocks) {
    if (block.hardware_id == rocjitsu::IpHardwareId::OssSys) {
      // What gfx950 publishes, and what binds a different implementation.
      block.major = 4;
      block.minor = 4;
    }
  }

  rocjitsu::GpuPciDevice mismatched("mismatched", spec, nullptr);
  const rocjitsu::InterruptRing ring = mismatched.interrupt_ring();
  EXPECT_FALSE(ring.programmed())
      << "a ring was reported for an OSSSYS version whose registers were never modelled";
  EXPECT_FALSE(mismatched.deliver_interrupt({.client_id = 1, .source_id = 2}))
      << "an interrupt was delivered through a ring this device could not have located";
}

// Revision is part of the driver's OSSSYS implementation selection. A layout
// verified for 7.1.0 must not be reused for 7.1.1 merely because its first two
// version components happen to match.
TEST(GpuDeviceInterruptRing, RefusesAnOsssysRevisionWithNoKnownLayout) {
  rocjitsu::config::KfdDeviceConfig device;
  device.gfx_target_version = kModelledTarget;
  device.local_mem_size = 8ULL * 1024 * 1024 * 1024;

  rocjitsu::GpuPciDeviceSpec spec = rocjitsu::gpu_pci_spec_from_config(device, {});
  ASSERT_TRUE(rocjitsu::GpuPciDevice("baseline", spec, nullptr).usable())
      << "the unmodified profile must be usable, or this proves nothing";

  bool changed_revision = false;
  for (rocjitsu::IpBlock &block : spec.discovery.blocks) {
    if (block.hardware_id == rocjitsu::IpHardwareId::OssSys) {
      ASSERT_EQ(block.revision, 0);
      block.revision = 1;
      changed_revision = true;
    }
  }
  ASSERT_TRUE(changed_revision) << "the profile has no OSSSYS record to exercise";

  rocjitsu::GpuPciDevice mismatched("mismatched", spec, nullptr);
  auto control = std::bit_cast<std::array<std::byte, 4>>(uint32_t{1});
  ASSERT_EQ(mismatched.bar_access(rocjitsu::GpuPciDevice::kRegisterBar, control, 0x4480,
                                  /*write=*/true),
            4);
  const rocjitsu::InterruptRing ring = mismatched.interrupt_ring();
  EXPECT_FALSE(ring.programmed())
      << "OSSSYS 7.1.1 was answered with the ring layout verified only for 7.1.0";
}

// The ring holds a finite number of entries and the driver acknowledges them by
// writing a doorbell. Until this was modelled the device never looked: it wrote
// an entry and advanced the pointer regardless, so a ring filled faster than it
// was drained overwrote entries the driver had not seen and said nothing about
// it. Occasional deliveries survive that; a command processor producing
// completions does not, and the failure would present as interrupts that were
// raised and never handled.
//
// Real hardware reports it in the write pointer itself, and the driver already
// knows how to read that: ih_v7_0_get_wptr checks the overflow bit in the copy
// it reads from memory, confirms against the register, warns, and resumes from
// wptr + 32 having accepted the loss.
TEST_F(GpuDeviceDelivery, ReportsAnOverflowRatherThanSilentlyOverwriting) {
  // Point the doorbell at index 4 of the doorbell page and enable it, which is
  // how the driver says where it will publish what it has consumed.
  constexpr uint32_t kDoorbellIndex = 4;
  constexpr uint32_t kIhDoorbellEnable = 0x10000000;
  constexpr uint32_t kIhRingControl = (10u << 1) | (1u << 0) | (1u << 17) | (2u << 28);
  constexpr uint32_t kIhWritePointerOverflowClear = 1u << 31;
  write_register_at(0x449c, kDoorbellIndex | kIhDoorbellEnable);

  // The driver has read nothing, so the ring is empty at zero and fills after
  // kRingBytes / kInterruptEntryBytes entries.
  const auto acknowledge = [this](uint32_t byte_offset) {
    auto raw = std::bit_cast<std::array<std::byte, 4>>(byte_offset);
    ASSERT_EQ(device_.bar_access(rocjitsu::GpuPciDevice::kDoorbellBar, raw, kDoorbellIndex * 4,
                                 /*write=*/true),
              4);
  };
  acknowledge(0);

  const auto entries = static_cast<uint32_t>(kRingBytes / kInterruptEntryBytes);
  // One short of full: the pointers being equal is how empty is spelled, so a
  // ring of N entries holds N-1 before it must report a loss.
  for (uint32_t entry_index = 0; entry_index + 1 < entries; ++entry_index) {
    ASSERT_TRUE(device_.deliver_interrupt({.client_id = 0x12, .source_id = 0x34}))
        << "entry " << entry_index << " of a ring that is not yet full";
    EXPECT_EQ(transport_.dword_at(kWptrAddress) & 1u, 0u)
        << "an overflow was reported at entry " << entry_index << ", before the ring was full";
  }

  // The one that closes the gap must say so, in the published pointer where the
  // driver looks and in the register it confirms against.
  ASSERT_TRUE(device_.deliver_interrupt({.client_id = 0x12, .source_id = 0x34}));
  EXPECT_EQ(transport_.dword_at(kWptrAddress) & 1u, 1u)
      << "the ring wrapped onto an unread entry and did not report an overflow";

  // The indication is sticky until the driver explicitly clears it. A later
  // delivery must not erase an overflow before the guest has had a chance to
  // observe it, and making room in the ring is not the hardware's clear
  // protocol.
  ASSERT_TRUE(device_.deliver_interrupt({.client_id = 0x12, .source_id = 0x34}));
  EXPECT_EQ(transport_.dword_at(kWptrAddress) & 1u, 1u)
      << "a later delivery erased an overflow the driver had not acknowledged";

  acknowledge(static_cast<uint32_t>(kInterruptEntryBytes * 8));
  ASSERT_TRUE(device_.deliver_interrupt({.client_id = 0x12, .source_id = 0x34}));
  EXPECT_EQ(transport_.dword_at(kWptrAddress) & 1u, 1u)
      << "advancing the read pointer cleared overflow without the control-register handshake";

  // ih_v7_0_get_wptr acknowledges the loss by pulsing
  // IH_RB_CNTL.WPTR_OVERFLOW_CLEAR. The register indication clears immediately,
  // and the next pointer publication no longer carries the bit.
  write_register_at(0x4480, kIhRingControl | kIhWritePointerOverflowClear);
  EXPECT_EQ(read_register_at(0x4488) & 1u, 0u)
      << "the overflow-clear pulse left the write-pointer register latched";
  write_register_at(0x4480, kIhRingControl);

  ASSERT_TRUE(device_.deliver_interrupt({.client_id = 0x12, .source_id = 0x34}));
  EXPECT_EQ(transport_.dword_at(kWptrAddress) & 1u, 0u)
      << "the device kept reporting an overflow after the driver's clear handshake";
}

TEST_F(GpuDeviceDelivery, ResetClearsAnUnacknowledgedOverflow) {
  constexpr uint32_t kDoorbellIndex = 4;
  constexpr uint32_t kIhDoorbellEnable = 0x10000000;
  write_register_at(0x449c, kDoorbellIndex | kIhDoorbellEnable);
  auto read_pointer = std::bit_cast<std::array<std::byte, 4>>(uint32_t{0});
  ASSERT_EQ(device_.bar_access(rocjitsu::GpuPciDevice::kDoorbellBar, read_pointer,
                               kDoorbellIndex * 4, /*write=*/true),
            4);

  const auto entries = static_cast<uint32_t>(kRingBytes / kInterruptEntryBytes);
  for (uint32_t entry_index = 0; entry_index < entries; ++entry_index) {
    ASSERT_TRUE(device_.deliver_interrupt({.client_id = 0x12, .source_id = 0x34}));
  }
  ASSERT_EQ(transport_.dword_at(kWptrAddress) & 1u, 1u)
      << "the test did not establish a latched overflow before reset";

  device_.reset(simdojo::ResetKind::FunctionLevel);
  program_ring();
  ASSERT_TRUE(device_.deliver_interrupt({.client_id = 0x12, .source_id = 0x34}));
  EXPECT_EQ(transport_.dword_at(kWptrAddress) & 1u, 0u)
      << "reset carried the previous guest's overflow into the reprogrammed ring";
}

// Two blocks answering one register is silent damage: whichever model defined
// it last wins, and the one that lost stalls the driver on a register reading
// as somebody else's. It cannot be ruled out by inspection either, because
// blocks legitimately share whole segments on this family -- GC and SDMA0
// publish identical bases, as do the two management processors -- and stay
// apart only in the offsets they claim inside them. So the device lays every
// model's claims into one absolute-dword map at construction and refuses a
// profile whose blocks collide.
//
// MMHUB's invalidation registers sit 0x575 dwords into its first segment and
// GC's sit 0x1645 into its own, so a MMHUB segment of 0x1260 + 0x1645 - 0x575
// puts the two engine-0 semaphores on the same dword.
TEST(GpuDeviceFlushes, RefusesAProfileWhereTwoBlocksClaimTheSameRegister) {
  rocjitsu::config::KfdDeviceConfig device;
  device.gfx_target_version = kModelledTarget;
  device.local_mem_size = 8ULL * 1024 * 1024 * 1024;

  rocjitsu::GpuPciDeviceSpec spec = rocjitsu::gpu_pci_spec_from_config(device, {});
  ASSERT_TRUE(rocjitsu::GpuPciDevice("baseline", spec, nullptr).usable())
      << "the unmodified profile must be usable, or this proves nothing";

  for (rocjitsu::IpBlock &block : spec.discovery.blocks) {
    if (block.hardware_id == rocjitsu::IpHardwareId::MmHub) {
      block.register_bases.front() = 0x1260 + 0x1645 - 0x575;
    }
  }

  const rocjitsu::GpuPciDevice colliding("colliding", spec, nullptr);
  EXPECT_FALSE(colliding.usable())
      << "two blocks were allowed to answer one register, and the later one won silently";
}
