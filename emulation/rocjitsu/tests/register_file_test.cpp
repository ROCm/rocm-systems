// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "simdojo/components/register_file.h"
#include "simdojo/components/vector_reg.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

namespace {

using simdojo::RegisterFile;
using simdojo::RegisterFileStorage;

constexpr size_t SOFTWARE_LAZY_TEST_CAPACITY = 4096;
template <typename RegType>
using SoftwareLazyTestStorage =
    simdojo::detail::SoftwareLazyRegisterStorage<RegType, SOFTWARE_LAZY_TEST_CAPACITY>;
template <typename RegType>
using SoftwareLazyTestFile =
    RegisterFile<RegType, RegisterFileStorage::SOFTWARE_LAZY, SOFTWARE_LAZY_TEST_CAPACITY>;

using SoftwareLazyUint32Storage = SoftwareLazyTestStorage<uint32_t>;
static_assert(!std::is_copy_constructible_v<SoftwareLazyUint32Storage>);
static_assert(!std::is_copy_assignable_v<SoftwareLazyUint32Storage>);
static_assert(!std::is_move_constructible_v<SoftwareLazyUint32Storage>);
static_assert(!std::is_move_assignable_v<SoftwareLazyUint32Storage>);
template <typename File>
concept HasContiguousData = requires(File &file) { file.data(); };
using SoftwareLazyUint32File = SoftwareLazyTestFile<uint32_t>;
static_assert(!HasContiguousData<SoftwareLazyUint32File>);

TEST(RegisterFileTest, ContiguousStorageClearsReusedBlock) {
  RegisterFile<uint32_t> file("contiguous");
  file.init(/*total_regs=*/16, /*regs_per_block=*/8);

  ASSERT_EQ(file.allocate(4), 0);
  for (uint32_t i = 0; i < 8; ++i)
    file[i] = i + 1;

  file.free(0);
  ASSERT_EQ(file.allocate(1), 0);
  for (uint32_t i = 0; i < 8; ++i)
    EXPECT_EQ(file[i], 0u) << "register " << i;
}

TEST(RegisterFileTest, SoftwareLazyStorageMaterializesOnlyMutableChunks) {
  using Vgpr = simdojo::VectorReg<64, uint32_t>;
  using Storage = SoftwareLazyTestStorage<Vgpr>;
  Storage storage;
  constexpr uint32_t regs_per_chunk = Storage::registers_per_chunk();
  static_assert(regs_per_chunk > 1);
  storage.init(2 * regs_per_chunk);

  const Storage &const_storage = storage;
  EXPECT_EQ(storage.materialized_chunk_count(), 0u);
  EXPECT_EQ(const_storage[0][0], 0u);
  EXPECT_EQ(const_storage[regs_per_chunk][63], 0u);
  EXPECT_EQ(storage.materialized_chunk_count(), 0u);

  storage[0][0] = 0x11111111u;
  storage[regs_per_chunk - 1][63] = 0x22222222u;
  EXPECT_EQ(storage.materialized_chunk_count(), 1u);
  storage[regs_per_chunk][0] = 0x33333333u;
  EXPECT_EQ(storage.materialized_chunk_count(), 2u);

  storage.reset(regs_per_chunk / 2, regs_per_chunk);
  EXPECT_EQ(storage.materialized_chunk_count(), 2u);
  EXPECT_EQ(const_storage[0][0], 0x11111111u);
  EXPECT_EQ(const_storage[regs_per_chunk - 1][63], 0u);
  EXPECT_EQ(const_storage[regs_per_chunk][0], 0u);

  storage.reset(0, 2 * regs_per_chunk);
  EXPECT_EQ(storage.materialized_chunk_count(), 0u);
  EXPECT_EQ(const_storage[0][0], 0u);
  EXPECT_EQ(const_storage[regs_per_chunk][0], 0u);
}

TEST(RegisterFileTest, SoftwareLazyStorageSupportsFixedCapacityBoundary) {
  using Vgpr = simdojo::VectorReg<64, uint32_t>;
  using Storage = simdojo::detail::SoftwareLazyRegisterStorage<Vgpr, 32>;
  static_assert(Storage::registers_per_chunk() == 16);

  Storage storage;
  storage.init(32);
  storage[31][63] = 0xA5A5A5A5u;

  const Storage &const_storage = storage;
  EXPECT_EQ(const_storage[31][63], 0xA5A5A5A5u);
  EXPECT_EQ(storage.materialized_chunk_count(), 1u);
}

TEST(RegisterFileTest, SoftwareLazyStorageClearsReusedUnalignedBlock) {
  using Vgpr = simdojo::VectorReg<64, uint32_t>;
  using File = SoftwareLazyTestFile<Vgpr>;
  File file("software_lazy");
  file.init(/*total_regs=*/48, /*regs_per_block=*/24);

  ASSERT_EQ(file.allocate(24), 0);
  ASSERT_EQ(file.allocate(24), 24);
  file[0][0] = 0x11111111u;
  file[23][63] = 0x22222222u;
  file[24][0] = 0x33333333u;
  file[47][63] = 0x44444444u;
  ASSERT_EQ(file.materialized_chunk_count(), 3u);

  file.free(0);
  EXPECT_EQ(file.materialized_chunk_count(), 2u);
  ASSERT_EQ(file.allocate(1), 0);
  const File &const_file = file;
  EXPECT_EQ(const_file[0][0], 0u);
  EXPECT_EQ(const_file[23][63], 0u);
  EXPECT_EQ(const_file[24][0], 0x33333333u);
  EXPECT_EQ(const_file[47][63], 0x44444444u);

  file.free(0);
  file.free(24);
  EXPECT_EQ(file.materialized_chunk_count(), 0u);
  ASSERT_EQ(file.allocate(1), 0);
  ASSERT_EQ(file.allocate(1), 24);
  EXPECT_EQ(const_file[0][0], 0u);
  EXPECT_EQ(const_file[23][63], 0u);
  EXPECT_EQ(const_file[24][0], 0u);
  EXPECT_EQ(const_file[47][63], 0u);
  EXPECT_EQ(file.materialized_chunk_count(), 0u);
}

TEST(RegisterFileTest, SoftwareLazyStorageReclaimsAlignedBlocksDuringChurn) {
  using Vgpr = simdojo::VectorReg<64, uint32_t>;
  using File = SoftwareLazyTestFile<Vgpr>;
  constexpr uint32_t regs_per_block = 512;
  File file("software_lazy");
  file.init(/*total_regs=*/2 * regs_per_block, regs_per_block);

  ASSERT_EQ(file.allocate(regs_per_block), 0);
  ASSERT_EQ(file.allocate(regs_per_block), static_cast<int32_t>(regs_per_block));
  file[0][0] = 0x11111111u;
  file[regs_per_block - 1][63] = 0x22222222u;
  file[regs_per_block][0] = 0x33333333u;
  file[2 * regs_per_block - 1][63] = 0x44444444u;
  ASSERT_EQ(file.materialized_chunk_count(), 4u);

  file.free(0);
  EXPECT_EQ(file.materialized_chunk_count(), 2u);
  const File &const_file = file;
  EXPECT_EQ(const_file[regs_per_block][0], 0x33333333u);
  EXPECT_EQ(const_file[2 * regs_per_block - 1][63], 0x44444444u);

  for (uint32_t value = 1; value <= 8; ++value) {
    ASSERT_EQ(file.allocate(regs_per_block), 0);
    file[regs_per_block / 2][17] = value;
    ASSERT_EQ(file.materialized_chunk_count(), 3u);
    file.free(0);
    EXPECT_EQ(file.materialized_chunk_count(), 2u);
  }

  file.free(regs_per_block);
  EXPECT_EQ(file.materialized_chunk_count(), 0u);
}

TEST(RegisterFileTest, SoftwareLazyLogicalRangeTraversalAndCopyCrossChunks) {
  using Vgpr = simdojo::VectorReg<64, uint32_t>;
  using Storage = SoftwareLazyTestStorage<Vgpr>;
  using File = SoftwareLazyTestFile<Vgpr>;
  constexpr uint32_t regs_per_chunk = Storage::registers_per_chunk();
  constexpr uint32_t reg_count = 2 * regs_per_chunk + 1;
  File file("software_lazy");
  file.init(reg_count, reg_count);
  ASSERT_EQ(file.allocate(reg_count), 0);

  const size_t byte_count = static_cast<size_t>(reg_count) * sizeof(Vgpr) - 3;
  std::vector<std::byte> source(byte_count);
  for (size_t idx = 0; idx < source.size(); ++idx)
    source[idx] = static_cast<std::byte>((idx * 37 + 11) & 0xFF);
  file.copy_from(0, reg_count, source);

  std::vector<std::byte> copied(byte_count);
  const File &const_file = file;
  const_file.copy_to(0, reg_count, copied);
  EXPECT_EQ(copied, source);

  uint32_t visited = 0;
  const_file.for_each(0, reg_count, [&](const Vgpr &reg) {
    EXPECT_EQ(reg[0], file[visited][0]);
    ++visited;
  });
  EXPECT_EQ(visited, reg_count);
}

TEST(RegisterFileTest, SoftwareLazySparseCopyPreservesAbsentZeroChunks) {
  using Vgpr = simdojo::VectorReg<64, uint32_t>;
  using Storage = SoftwareLazyTestStorage<Vgpr>;
  using File = SoftwareLazyTestFile<Vgpr>;
  constexpr uint32_t regs_per_chunk = Storage::registers_per_chunk();
  constexpr uint32_t reg_count = 3 * regs_per_chunk;
  File file("software_lazy");
  file.init(reg_count, reg_count);
  ASSERT_EQ(file.allocate(reg_count), 0);

  std::vector<std::byte> source(static_cast<size_t>(reg_count) * sizeof(Vgpr));
  source[7] = std::byte{0x5A};
  source[2 * regs_per_chunk * sizeof(Vgpr) + 13] = std::byte{0xA5};
  file.copy_nonzero_from(0, reg_count, source);
  EXPECT_EQ(file.materialized_chunk_count(), 2u);

  std::vector<std::byte> copied(source.size());
  const File &const_file = file;
  const_file.copy_to(0, reg_count, copied);
  EXPECT_EQ(copied, source);

  const_file.for_each(regs_per_chunk, regs_per_chunk,
                      [](const Vgpr &reg) { EXPECT_EQ(reg[0], 0u); });
  EXPECT_EQ(file.materialized_chunk_count(), 2u);
}

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG)
TEST(RegisterFileDeathTest, ConstAccessToFreedBlockAsserts) {
  using File = SoftwareLazyTestFile<uint32_t>;
  File file("software_lazy");
  file.init(/*total_regs=*/16, /*regs_per_block=*/8);

  ASSERT_EQ(file.allocate(1), 0);
  file.free(0);
  const File &const_file = file;

  EXPECT_DEATH({ static_cast<void>(const_file[0]); }, "const access to a free register block");
}

TEST(RegisterFileDeathTest, MutableAccessToFreedBlockAsserts) {
  using File = SoftwareLazyTestFile<uint32_t>;
  File file("software_lazy");
  file.init(/*total_regs=*/16, /*regs_per_block=*/8);

  ASSERT_EQ(file.allocate(1), 0);
  file.free(0);

  EXPECT_DEATH({ static_cast<void>(file[0]); }, "mutable access to a free register block");
}
#endif

} // namespace
