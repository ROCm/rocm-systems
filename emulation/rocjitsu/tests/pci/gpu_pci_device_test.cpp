// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pci/bar_access_trace.h"
#include "rocjitsu/vm/amdgpu/pci/gpu_pci_device.h"
#include "rocjitsu/vm/amdgpu/pci/gpu_pci_device_spec.h"
#include "rocjitsu/vm/amdgpu/pci/mmio_registers.h"
#include "rocjitsu/vm/amdgpu/pci/register_symbols.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>

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
  return rocjitsu::gpu_pci_spec_from_config(device, {});
}

class GpuDevice : public ::testing::Test {
protected:
  rocjitsu::RegisterSymbols symbols_;
  rocjitsu::BarAccessTrace trace_{symbols_};
  rocjitsu::GpuPciDevice device_{"gpu", configured_spec(), &trace_};

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
TEST_F(GpuDevice, ReportsEveryVmInvalidationAlreadyComplete) {
  ASSERT_TRUE(device_.usable());

  EXPECT_EQ(read_register_at(0xa368), 0xffffffffu) << "GFXHUB engine 17 never acknowledges";
  EXPECT_EQ(read_register_at(0x696a8), 0xffffffffu) << "MMHUB engine 17 never acknowledges";

  // Engine 0 and the last engine, since the driver reaches engines by stride
  // and modelling only the one it happens to use today would break silently.
  EXPECT_EQ(read_register_at((0x1a000 + 0x0599) * 4), 0xffffffffu);
  EXPECT_EQ(read_register_at((0x1a000 + 0x0599 + 17) * 4), 0xffffffffu);

  // The request is written rather than read, so what is checked is that the
  // device models it at all: an unmodelled register drops the write and reads
  // back zero. Reading a request register back is not something the driver
  // does, but this asserts the model, not hardware fidelity.
  constexpr uint64_t kMmHubRequest17 = (0x1a000 + 0x0587 + 17) * 4;
  write_register_at(kMmHubRequest17, 0x1);
  EXPECT_EQ(read_register_at(kMmHubRequest17), 0x1u)
      << "MMHUB engine 17's request was dropped as unmodelled";
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

  const rocjitsu::GpuPciDevice::InterruptRing ring = device_.interrupt_ring();
  EXPECT_EQ(ring.base, kRingBase);
  EXPECT_EQ(ring.wptr_address, kWptrAddress);
  EXPECT_EQ(ring.bytes, 256u * 1024) << "the size field is a logarithm of a dword count";
  EXPECT_TRUE(ring.enabled);
  EXPECT_TRUE(ring.raises_messages);
  EXPECT_EQ(ring.space, rocjitsu::GpuPciDevice::InterruptRingSpace::BusAddress);

  // The same registers with the space the driver uses when it loads firmware
  // through the security processor: the addresses are then translated, and
  // acting on them as if they were bus addresses would write somewhere real
  // and wrong.
  write_register_at(0x4480, (16u << 1) | (1u << 0) | (1u << 17) | (4u << 28));
  EXPECT_EQ(device_.interrupt_ring().space, rocjitsu::GpuPciDevice::InterruptRingSpace::GpuVirtual);
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
    const rocjitsu::GpuPciDevice::InterruptRing ring = device_.interrupt_ring();
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
    for (std::size_t i = 0; i < dst.size(); ++i) {
      const auto found = memory.find(guest_phys + i);
      dst[i] = found == memory.end() ? std::byte{0} : found->second;
    }
    return true;
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

  [[nodiscard]] bool trigger(uint32_t vector) override {
    triggered.push_back(vector);
    return !refuse_trigger;
  }

  [[nodiscard]] uint32_t dword_at(uint64_t guest_phys) {
    std::array<std::byte, 4> raw{};
    (void)read(guest_phys, raw);
    return std::bit_cast<uint32_t>(raw);
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
};

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

  void SetUp() override {
    ASSERT_TRUE(device_.usable());
    device_.set_dma_engine(&transport_);
    device_.set_irq_sink(&transport_);
    // A 4 KiB ring above the 32-bit boundary, its write pointer published just
    // past it, switched on and at a bus address. 4 KiB is 1024 dwords, so the
    // size field is 10.
    write_register_at(0x448c, static_cast<uint32_t>(kRingBase >> 8));
    write_register_at(0x4490, static_cast<uint32_t>(kRingBase >> 40) & 0xff);
    write_register_at(0x4498, static_cast<uint32_t>(kWptrAddress));
    write_register_at(0x4494, static_cast<uint32_t>(kWptrAddress >> 32) & 0xffff);
    write_register_at(0x4480, (10u << 1) | (1u << 0) | (1u << 17) | (2u << 28));
  }

  void TearDown() override {
    device_.set_dma_engine(nullptr);
    device_.set_irq_sink(nullptr);
  }
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
  device_.set_dma_engine(nullptr);
  EXPECT_FALSE(device_.deliver_interrupt({.client_id = 1, .source_id = 2}));

  device_.set_dma_engine(&transport_);
  device_.set_irq_sink(nullptr);
  EXPECT_FALSE(device_.deliver_interrupt({.client_id = 1, .source_id = 2}));

  EXPECT_TRUE(transport_.memory.empty());
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
}

} // namespace
