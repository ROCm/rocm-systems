// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "simdojo/components/register_file.h"
#include "simdojo/components/vector_reg.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(__linux__)
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#endif

namespace {

using simdojo::RegisterFile;
using simdojo::RegisterFileStorage;

using DemandPagedUint32Storage = simdojo::detail::DemandPagedRegisterStorage<uint32_t>;
static_assert(!std::is_copy_constructible_v<DemandPagedUint32Storage>);
static_assert(!std::is_copy_assignable_v<DemandPagedUint32Storage>);
static_assert(!std::is_move_constructible_v<DemandPagedUint32Storage>);
static_assert(!std::is_move_assignable_v<DemandPagedUint32Storage>);

#if defined(__linux__)
size_t resident_page_count(const void *data, size_t bytes) {
  const long page_size_result = sysconf(_SC_PAGESIZE);
  if (page_size_result <= 0) {
    ADD_FAILURE() << "sysconf(_SC_PAGESIZE) failed";
    return std::numeric_limits<size_t>::max();
  }
  const size_t page_size = static_cast<size_t>(page_size_result);
  const uintptr_t first = reinterpret_cast<uintptr_t>(data) & ~(page_size - 1);
  const uintptr_t last =
      (reinterpret_cast<uintptr_t>(data) + bytes + page_size - 1) & ~(page_size - 1);
  std::vector<unsigned char> residency((last - first) / page_size);
  if (mincore(reinterpret_cast<void *>(first), last - first, residency.data()) != 0) {
    ADD_FAILURE() << "mincore failed: " << std::strerror(errno);
    return std::numeric_limits<size_t>::max();
  }
  return static_cast<size_t>(std::count_if(residency.begin(), residency.end(),
                                           [](unsigned char value) { return value & 1; }));
}
#endif

TEST(RegisterFileTest, EagerStorageClearsReusedBlock) {
  RegisterFile<uint32_t> file("eager");
  file.init(/*total_regs=*/16, /*regs_per_block=*/8);

  ASSERT_EQ(file.allocate(4), 0);
  for (uint32_t i = 0; i < 8; ++i)
    file[i] = i + 1;

  file.free(0);
  ASSERT_EQ(file.allocate(1), 0);
  for (uint32_t i = 0; i < 8; ++i)
    EXPECT_EQ(file[i], 0u) << "register " << i;
}

TEST(RegisterFileTest, DemandPagedStorageIsContiguousAndInitiallyZero) {
  using File = RegisterFile<uint32_t, RegisterFileStorage::DEMAND_PAGED>;
  File file("demand_paged");
  file.init(/*total_regs=*/3000, /*regs_per_block=*/1000);

  ASSERT_NE(file.data(), nullptr);
#if defined(__linux__)
  EXPECT_EQ(resident_page_count(file.data(), file.total_regs() * sizeof(uint32_t)), 0u);
  const long page_size_result = sysconf(_SC_PAGESIZE);
  ASSERT_GT(page_size_result, 0);
  const size_t page_size = static_cast<size_t>(page_size_result);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(file.data()) % page_size, alignof(std::max_align_t));
#endif
  ASSERT_EQ(file.allocate(1), 0);
  ASSERT_EQ(file.allocate(1), 1000);
  ASSERT_EQ(file.allocate(1), 2000);
  EXPECT_EQ(&file[0], file.data());
  EXPECT_EQ(&file[1000], file.data() + 1000);
  for (uint32_t i = 0; i < file.total_regs(); ++i)
    EXPECT_EQ(file[i], 0u) << "register " << i;
}

TEST(RegisterFileTest, DemandPagedResetPreservesNeighboringUnalignedBlocks) {
  using File = RegisterFile<uint32_t, RegisterFileStorage::DEMAND_PAGED>;
  File file("demand_paged");
  file.init(/*total_regs=*/3000, /*regs_per_block=*/1000);

  ASSERT_EQ(file.allocate(1), 0);
  ASSERT_EQ(file.allocate(1), 1000);
  ASSERT_EQ(file.allocate(1), 2000);
  for (uint32_t i = 0; i < 1000; ++i) {
    file[i] = 0x11111111u;
    file[1000 + i] = 0x22222222u;
    file[2000 + i] = 0x33333333u;
  }

  file.free(1000);
  ASSERT_EQ(file.allocate(17), 1000);
  for (uint32_t i = 0; i < 1000; ++i) {
    EXPECT_EQ(file[i], 0x11111111u) << "left neighbor register " << i;
    EXPECT_EQ(file[1000 + i], 0u) << "reset register " << i;
    EXPECT_EQ(file[2000 + i], 0x33333333u) << "right neighbor register " << i;
  }

  file.free(0);
  file.free(1000);
  file.free(2000);
#if defined(__linux__)
  EXPECT_LE(resident_page_count(file.data(), file.total_regs() * sizeof(uint32_t)), 2u);
#endif
}

TEST(RegisterFileTest, DemandPagedVectorBlockResetClearsEveryLane) {
  using Vgpr = simdojo::VectorReg<64, uint32_t>;
  using File = RegisterFile<Vgpr, RegisterFileStorage::DEMAND_PAGED>;
  File file("vgpr");
  file.init(/*total_regs=*/1024, /*regs_per_block=*/512);

  ASSERT_EQ(file.allocate(256), 0);
  for (uint32_t reg = 0; reg < 512; ++reg)
    for (uint32_t lane = 0; lane < 64; ++lane)
      file[reg][lane] = reg * 64 + lane + 1;

  file.free(0);
  ASSERT_EQ(file.allocate(1), 0);
  for (uint32_t reg = 0; reg < 512; ++reg)
    for (uint32_t lane = 0; lane < 64; ++lane)
      EXPECT_EQ(file[reg][lane], 0u) << "v" << reg << " lane " << lane;
}

TEST(RegisterFileTest, DemandPagedFreeRecyclesPhysicalPages) {
#if defined(__linux__)
  using Vgpr = simdojo::VectorReg<64, uint32_t>;
  using File = RegisterFile<Vgpr, RegisterFileStorage::DEMAND_PAGED>;
  File file("vgpr");
  file.init(/*total_regs=*/512, /*regs_per_block=*/512);

  ASSERT_EQ(file.allocate(256), 0);
  for (uint32_t reg = 0; reg < 512; ++reg)
    for (uint32_t lane = 0; lane < 64; ++lane)
      file[reg][lane] = reg + lane + 1;

  const size_t block_bytes = 512 * sizeof(Vgpr);
  const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  const uintptr_t first_page = reinterpret_cast<uintptr_t>(file.data()) & ~(page_size - 1);
  const uintptr_t last_page =
      (reinterpret_cast<uintptr_t>(file.data()) + block_bytes + page_size - 1) & ~(page_size - 1);
  const size_t block_pages = (last_page - first_page) / page_size;
  EXPECT_EQ(resident_page_count(file.data(), block_bytes), block_pages);

  file.free(0);
  // The two partial edge pages are zeroed in place because they can overlap
  // neighboring allocation blocks. Every page wholly owned by this block is
  // discarded immediately.
  EXPECT_LE(resident_page_count(file.data(), block_bytes), 2u);
#else
  GTEST_SKIP() << "physical-page residency is only observable on Linux";
#endif
}

} // namespace
