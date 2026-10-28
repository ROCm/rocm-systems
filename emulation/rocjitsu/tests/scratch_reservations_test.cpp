// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/scratch_reservations.h"

#include <gtest/gtest.h>

#include <array>
#include <barrier>
#include <cstdint>
#include <limits>
#include <memory>
#include <thread>

namespace {
using rocjitsu::amdgpu::AddressSpaceHandle;
using rocjitsu::amdgpu::ScratchReservation;
using rocjitsu::amdgpu::ScratchReservations;
constexpr AddressSpaceHandle kSpace{.slot = 1, .generation = 2};

TEST(ScratchReservationsTest, RejectsOverlapsAndReusesOnlyReleasedRanges) {
  ScratchReservations reservations;
  auto middle = reservations.try_reserve(kSpace, 256, 256);
  ASSERT_TRUE(middle);
  EXPECT_EQ(middle->base(), 256u);
  EXPECT_EQ(middle->size(), 256u);
  auto left = reservations.try_reserve(kSpace, 0, 256);
  auto right = reservations.try_reserve(kSpace, 512, 256);
  ASSERT_TRUE(left);
  ASSERT_TRUE(right);
  EXPECT_FALSE(reservations.try_reserve(kSpace, 256, 256));
  EXPECT_FALSE(reservations.try_reserve(kSpace, 255, 2));
  EXPECT_FALSE(reservations.try_reserve(kSpace, 511, 2));
  EXPECT_FALSE(reservations.try_reserve(kSpace, 0, 1024));
  auto retained = middle;
  middle.reset();
  EXPECT_FALSE(reservations.try_reserve(kSpace, 256, 256));
  retained.reset();
  ASSERT_TRUE(reservations.try_reserve(kSpace, 256, 256));
  EXPECT_FALSE(reservations.try_reserve(kSpace, 0, 256));
  EXPECT_FALSE(reservations.try_reserve(kSpace, 512, 256));
}

TEST(ScratchReservationsTest, SeparatesAddressSpacesAndReusedIdentities) {
  ScratchReservations reservations;
  auto first = reservations.try_reserve(kSpace, 0x10000, 4096);
  auto other = reservations.try_reserve({.slot = 2, .generation = 2}, 0x10000, 4096);
  auto rebuilt = reservations.try_reserve({.slot = 1, .generation = 3}, 0x10000, 4096);
  ASSERT_TRUE(first);
  ASSERT_TRUE(other);
  ASSERT_TRUE(rebuilt);
  first.reset();
  EXPECT_FALSE(reservations.try_reserve({.slot = 1, .generation = 3}, 0x10000, 4096));
  EXPECT_TRUE(reservations.try_reserve(kSpace, 0x10000, 4096));
}

TEST(ScratchReservationsTest, ReleasesBackingLeaseAndCanOutliveItsManager) {
  auto backing = std::make_shared<uint32_t>(7);
  std::weak_ptr<uint32_t> released = backing;
  std::shared_ptr<ScratchReservation> lease;
  {
    ScratchReservations reservations;
    lease = reservations.try_reserve(kSpace, 0x1000, 256, backing);
    ASSERT_TRUE(lease);
    backing.reset();
    EXPECT_FALSE(released.expired());
  }
  EXPECT_EQ(lease->base(), 0x1000u);
  lease.reset();
  EXPECT_TRUE(released.expired());
}

TEST(ScratchReservationsTest, RejectsInvalidRangesWithoutChangingOccupancy) {
  ScratchReservations reservations;
  constexpr auto last = std::numeric_limits<uint64_t>::max();
  EXPECT_FALSE(reservations.try_reserve({}, 0, 1));
  EXPECT_FALSE(reservations.try_reserve(kSpace, 0, 0));
  EXPECT_FALSE(reservations.try_reserve(kSpace, last, 2));
  auto valid = reservations.try_reserve(kSpace, last, 1);
  ASSERT_TRUE(valid);
  EXPECT_FALSE(reservations.try_reserve(kSpace, last - 1, 2));
}

TEST(ScratchReservationsTest, ConcurrentRequestsCannotOwnTheSameRange) {
  ScratchReservations reservations;
  std::array<std::shared_ptr<ScratchReservation>, 2> leases;
  std::barrier start(2);
  std::array<std::thread, 2> workers;
  for (size_t i = 0; i < workers.size(); ++i)
    workers[i] = std::thread([&, i] {
      start.arrive_and_wait();
      leases[i] = reservations.try_reserve(kSpace, 0x1000, 256);
    });
  for (auto &worker : workers)
    worker.join();
  EXPECT_EQ(static_cast<unsigned>(bool(leases[0])) + static_cast<unsigned>(bool(leases[1])), 1u);
}
} // namespace
