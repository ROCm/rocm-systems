// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pci/bar_access_trace.h"
#include "rocjitsu/vm/amdgpu/pci/register_symbols.h"
#include "rocjitsu/vm/amdgpu/pci/scratch_pci_device.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace {

constexpr simdojo::PciId kTestId = {.vendor = 0x1002,
                                    .device = 0x1250,
                                    .subsys_vendor = 0x1002,
                                    .subsys = 0x0000,
                                    .cls = 0x12,
                                    .subcls = 0x00,
                                    .prog_if = 0x00,
                                    .revision = 0x00};

std::span<std::byte> as_bytes(uint32_t &value) {
  return {reinterpret_cast<std::byte *>(&value), sizeof(value)};
}

TEST(ScratchPciDevice, DeclaresOneTrappedMemoryBar) {
  rocjitsu::ScratchPciDevice device("scratch", kTestId, nullptr);

  const std::vector<simdojo::BarSpec> bars = device.bars();

  ASSERT_EQ(bars.size(), 1u);
  EXPECT_EQ(bars[0].index, rocjitsu::ScratchPciDevice::kBarIndex);
  EXPECT_EQ(bars[0].size, rocjitsu::ScratchPciDevice::kBarSize);
  EXPECT_TRUE(bars[0].mem);
  EXPECT_LT(bars[0].backing_fd, 0) << "the scratch BAR must trap so accesses are observable";
  EXPECT_TRUE(bars[0].mmap_areas.empty());
}

TEST(ScratchPciDevice, ReadsBackWhatWasWritten) {
  rocjitsu::ScratchPciDevice device("scratch", kTestId, nullptr);

  uint32_t written = 0xdeadbeef;
  ASSERT_EQ(device.bar_access(rocjitsu::ScratchPciDevice::kBarIndex, as_bytes(written), 0x10,
                              /*write=*/true),
            4);

  uint32_t read = 0;
  ASSERT_EQ(device.bar_access(rocjitsu::ScratchPciDevice::kBarIndex, as_bytes(read), 0x10,
                              /*write=*/false),
            4);
  EXPECT_EQ(read, 0xdeadbeefu);
}

TEST(ScratchPciDevice, KeepsOffsetsIndependent) {
  rocjitsu::ScratchPciDevice device("scratch", kTestId, nullptr);

  uint32_t first = 0x11111111;
  uint32_t second = 0x22222222;
  ASSERT_GT(device.bar_access(0, as_bytes(first), 0x00, /*write=*/true), 0);
  ASSERT_GT(device.bar_access(0, as_bytes(second), 0x04, /*write=*/true), 0);

  uint32_t read = 0;
  ASSERT_GT(device.bar_access(0, as_bytes(read), 0x00, /*write=*/false), 0);
  EXPECT_EQ(read, 0x11111111u);
}

TEST(ScratchPciDevice, RejectsAccessBeyondTheBar) {
  rocjitsu::ScratchPciDevice device("scratch", kTestId, nullptr);

  uint32_t value = 0;
  EXPECT_LT(device.bar_access(0, as_bytes(value), rocjitsu::ScratchPciDevice::kBarSize,
                              /*write=*/false),
            0);
  EXPECT_LT(device.bar_access(0, as_bytes(value), rocjitsu::ScratchPciDevice::kBarSize - 2,
                              /*write=*/false),
            0);
}

TEST(ScratchPciDevice, RejectsAWidthNoRegisterFileWouldAnswer) {
  rocjitsu::ScratchPciDevice device("scratch", kTestId, nullptr);

  std::array<std::byte, 3> odd{};
  EXPECT_LT(device.bar_access(0, odd, 0x0, /*write=*/false), 0);
}

TEST(ScratchPciDevice, RejectsAccessToABarItDoesNotHave) {
  rocjitsu::ScratchPciDevice device("scratch", kTestId, nullptr);

  uint32_t value = 0;
  EXPECT_LT(device.bar_access(5, as_bytes(value), 0x0, /*write=*/false), 0);
}

// A refused access names no hardware, so it must not appear in the burndown
// list of registers still to be modeled -- that list is a work queue, and an
// out-of-range probe is not work.
TEST(ScratchPciDevice, ReportsRejectedAccessesApartFromUnmodeledOnes) {
  rocjitsu::RegisterSymbols symbols;
  rocjitsu::BarAccessTrace trace(symbols);
  rocjitsu::ScratchPciDevice device("scratch", kTestId, &trace);

  uint32_t value = 0;
  ASSERT_GT(device.bar_access(0, as_bytes(value), 0x0, /*write=*/false), 0);
  EXPECT_TRUE(trace.unmodeled_report().empty());

  ASSERT_LT(device.bar_access(0, as_bytes(value), rocjitsu::ScratchPciDevice::kBarSize,
                              /*write=*/false),
            0);
  EXPECT_TRUE(trace.unmodeled_report().empty())
      << "a refused access was reported as a register still to be modeled";
  EXPECT_FALSE(trace.rejected_report().empty());
}

TEST(ScratchPciDevice, ForgetsItsContentsOnReset) {
  rocjitsu::ScratchPciDevice device("scratch", kTestId, nullptr);

  uint32_t written = 0xa5a5a5a5;
  ASSERT_GT(device.bar_access(0, as_bytes(written), 0x20, /*write=*/true), 0);
  device.reset(simdojo::ResetKind::FunctionLevel);

  uint32_t read = 0xffffffff;
  ASSERT_GT(device.bar_access(0, as_bytes(read), 0x20, /*write=*/false), 0);
  EXPECT_EQ(read, 0u);
}

TEST(ScratchPciDevice, TracksMappedGuestWindows) {
  rocjitsu::ScratchPciDevice device("scratch", kTestId, nullptr);
  const simdojo::DmaRegion region = {
      .guest_phys = 0x100000, .length = 0x1000, .access = simdojo::DmaAccess::None};

  EXPECT_EQ(device.mapped_regions(), 0u);
  device.dma_map(region);
  EXPECT_EQ(device.mapped_regions(), 1u);
  device.dma_unmap(region);
  EXPECT_EQ(device.mapped_regions(), 0u);
}

} // namespace
