// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vmm/vfu/bus_plan.h"

#include "rocjitsu/vm/amdgpu/pci/gpu_pci_device.h"
#include "rocjitsu/vm/amdgpu/pci/gpu_pci_device_spec.h"

#include <gtest/gtest.h>

#include <cstdint>

namespace {

simdojo::BarSpec trapped_bar() {
  simdojo::BarSpec bar;
  bar.index = 0;
  bar.size = 0x1000;
  bar.mem = true;
  return bar;
}

// A device that names no mappable window expects every access to reach it. The
// transport must not hand over the backing descriptor in that case: libvfio-user
// treats a descriptor with no windows as "map the whole region", which would let
// the guest read and write a register aperture without the device ever seeing it.
TEST(BarRegionPlan, WithholdsTheDescriptorWhenNoWindowIsDeclared) {
  simdojo::BarSpec bar = trapped_bar();
  bar.backing_fd = 7;
  bar.fd_offset = 0x800;

  const rocjitsu::BarRegionPlan plan = rocjitsu::plan_bar_region(bar);

  ASSERT_TRUE(plan.valid);
  EXPECT_LT(plan.backing_fd, 0) << "an unshared BAR must trap every access";
  EXPECT_EQ(plan.fd_offset, 0u);
  EXPECT_TRUE(plan.mmap_areas.empty());
}

TEST(BarRegionPlan, SharesTheDescriptorForDeclaredWindows) {
  simdojo::BarSpec bar = trapped_bar();
  bar.backing_fd = 7;
  bar.fd_offset = 0x800;
  bar.mmap_areas = {{.offset = 0, .length = 0x1000}};

  const rocjitsu::BarRegionPlan plan = rocjitsu::plan_bar_region(bar);

  ASSERT_TRUE(plan.valid);
  EXPECT_EQ(plan.backing_fd, 7);
  EXPECT_EQ(plan.fd_offset, 0x800u);
  ASSERT_EQ(plan.mmap_areas.size(), 1u);
  EXPECT_EQ(plan.mmap_areas[0].length, 0x1000u);
}

TEST(BarRegionPlan, RejectsWindowsWithNothingToBackThem) {
  simdojo::BarSpec bar = trapped_bar();
  bar.mmap_areas = {{.offset = 0, .length = 0x1000}};

  EXPECT_FALSE(rocjitsu::plan_bar_region(bar).valid);
}

TEST(BarRegionPlan, RejectsAWindowThatLeavesTheRegion) {
  simdojo::BarSpec bar = trapped_bar();
  bar.backing_fd = 7;
  bar.mmap_areas = {{.offset = 0xf00, .length = 0x200}};

  EXPECT_FALSE(rocjitsu::plan_bar_region(bar).valid);
}

TEST(BarRegionPlan, RejectsAnEmptyWindow) {
  simdojo::BarSpec bar = trapped_bar();
  bar.backing_fd = 7;
  bar.mmap_areas = {{.offset = 0, .length = 0}};

  EXPECT_FALSE(rocjitsu::plan_bar_region(bar).valid);
}

TEST(BarRegionPlan, RejectsABarThatCannotExist) {
  simdojo::BarSpec out_of_range = trapped_bar();
  out_of_range.index = 6;
  EXPECT_FALSE(rocjitsu::plan_bar_region(out_of_range).valid);

  simdojo::BarSpec empty = trapped_bar();
  empty.size = 0;
  EXPECT_FALSE(rocjitsu::plan_bar_region(empty).valid);
}

TEST(InterruptPlan, AdvertisesNothingForADeviceThatRaisesNothing) {
  const rocjitsu::InterruptPlan plan = rocjitsu::plan_interrupts({});

  ASSERT_TRUE(plan.supported);
  EXPECT_EQ(plan.intx_count, 0u) << "a silent device must not be given a pin to wait on";
}

TEST(InterruptPlan, AdvertisesOnePinWhenAsked) {
  const rocjitsu::InterruptPlan plan =
      rocjitsu::plan_interrupts({.kind = simdojo::InterruptKind::IntxPin, .vectors = 0});

  ASSERT_TRUE(plan.supported);
  EXPECT_EQ(plan.intx_count, 1u);
}

// The device asking and the transport advertising are written apart, so nothing
// else pairs them. Asking for a kind this transport cannot advertise is worse
// than asking for none: the function is refused before it is served at all, so
// the guest sees no device rather than a device without interrupts.
TEST(InterruptPlan, AdvertisesWhatTheModelledGpuAsksFor) {
  rocjitsu::config::KfdDeviceConfig configured;
  configured.gfx_target_version = 120500;
  configured.vendor_id = 0x1002;
  configured.device_id = 0x1250;
  configured.local_mem_size = 16ULL * 1024 * 1024;
  rocjitsu::GpuPciDevice device("gpu", rocjitsu::gpu_pci_spec_from_config(configured, {}), nullptr);

  EXPECT_TRUE(rocjitsu::plan_interrupts(device.interrupts()).supported)
      << "the transport cannot advertise the capability this device asks for";
}

// Where the table lives is the device's decision, and the transport carries it
// through unchanged: the guest is told to look at this BAR and these offsets,
// and it looks exactly there. A transport that adjusted them would point the
// guest at bytes the device does not treat as a table.
TEST(InterruptPlan, CarriesTheMessageTableThroughUntouched) {
  const rocjitsu::InterruptPlan plan = rocjitsu::plan_interrupts({
      .kind = simdojo::InterruptKind::MsiX,
      .vectors = 4,
      .table_bar = 4,
      .table_offset = 0,
      .pending_offset = 4096,
  });

  ASSERT_TRUE(plan.supported);
  EXPECT_EQ(plan.msix_count, 4u);
  EXPECT_EQ(plan.table_bar, 4);
  EXPECT_EQ(plan.table_offset, 0u);
  EXPECT_EQ(plan.pending_offset, 4096u);
  EXPECT_EQ(plan.intx_count, 0u) << "a pin as well would be two capabilities for one interrupt";
}

// A capability offering nothing to allocate is worse than none at all: the
// driver asks for a vector, gets nothing, and fails the same probe it would
// have failed anyway -- but now with a published table to explain away.
TEST(InterruptPlan, RefusesMessageInterruptsWithNoVectors) {
  const rocjitsu::InterruptPlan plan =
      rocjitsu::plan_interrupts({.kind = simdojo::InterruptKind::MsiX, .vectors = 0});

  EXPECT_FALSE(plan.supported);
}

// Every field of the capability is packed -- eleven bits of count, three of BAR
// index, twenty-nine of offset in units of eight bytes -- and nothing
// downstream refuses a declaration that does not fit. It would be published
// rounded down or wrapped around, naming bytes the device does not treat as a
// table. These are the boundaries, because a wrong bound would sit on one.
TEST(InterruptPlan, AcceptsTheLargestDeclarationACapabilityCanExpress) {
  constexpr uint64_t kMaxOffset = 0xfffffff8;
  const auto accepted = [](uint32_t vectors, int bar, uint64_t table, uint64_t pending) {
    return rocjitsu::plan_interrupts({.kind = simdojo::InterruptKind::MsiX,
                                      .vectors = vectors,
                                      .table_bar = bar,
                                      .table_offset = table,
                                      .pending_offset = pending})
        .supported;
  };

  EXPECT_TRUE(accepted(2048, 5, kMaxOffset, kMaxOffset)) << "every field at its ceiling";
  EXPECT_TRUE(accepted(1, 0, 8, 16));

  EXPECT_FALSE(accepted(2049, 0, 0, 8)) << "one more vector than eleven bits can say";
  EXPECT_FALSE(accepted(1, 6, 0, 8)) << "a BAR index past the six a function has";
  EXPECT_FALSE(accepted(1, -1, 0, 8)) << "a negative BAR index would wrap into another BAR";
  EXPECT_FALSE(accepted(1, 0, 4, 8)) << "an offset that is not a whole number of units";
  EXPECT_FALSE(accepted(1, 0, 0, 12)) << "likewise for the pending bits";
  EXPECT_FALSE(accepted(1, 0, kMaxOffset + 8, 8)) << "one unit past what the field holds";
  // Both ceilings, not just the table's: the two are separate disjuncts, and a
  // slip duplicating one of them would leave the other unchecked.
  EXPECT_FALSE(accepted(1, 0, 8, kMaxOffset + 8)) << "likewise for the pending bits";
}

} // namespace
