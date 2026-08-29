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

  [[nodiscard]] const simdojo::BarSpec *bar(int index) {
    static std::vector<simdojo::BarSpec> bars;
    bars = device_.bars();
    const auto found = std::ranges::find(bars, index, &simdojo::BarSpec::index);
    return found == bars.end() ? nullptr : &*found;
  }
};

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
