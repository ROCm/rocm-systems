// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vmm/vfu/bus_plan.h"

#include <gtest/gtest.h>

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

TEST(InterruptPlan, RefusesMsiXUntilItIsImplemented) {
  const rocjitsu::InterruptPlan plan =
      rocjitsu::plan_interrupts({.kind = simdojo::InterruptKind::MsiX, .vectors = 4});

  EXPECT_FALSE(plan.supported);
}

} // namespace
