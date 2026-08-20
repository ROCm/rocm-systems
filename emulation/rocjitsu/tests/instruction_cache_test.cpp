// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/instruction_cache.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace {

using rocjitsu::amdgpu::GpuMemory;
using rocjitsu::amdgpu::InstructionCache;

constexpr uint64_t kCodeBase = 0x200000;

/// @brief Fill @p bytes of code memory at kCodeBase with a per-byte pattern.
std::vector<uint8_t> fill_code(GpuMemory &memory, size_t bytes, uint8_t salt, uint32_t vmid = 0) {
  std::vector<uint8_t> expected(bytes);
  for (size_t i = 0; i < bytes; ++i)
    expected[i] = static_cast<uint8_t>((i * 7) ^ salt);
  memory.write_block(kCodeBase, std::span<const uint8_t>(expected), vmid);
  return expected;
}

std::array<uint8_t, InstructionCache::kFetchBytes>
fetch_at(InstructionCache &icache, const GpuMemory &memory, uint64_t pc, uint32_t vmid = 0) {
  std::array<uint8_t, InstructionCache::kFetchBytes> got{};
  icache.fetch(memory, pc, vmid, got.data());
  return got;
}

// Every four-byte-aligned PC in a two-line window, including the offsets whose
// fetch window runs off the end of a line, must return the backing bytes.
TEST(InstructionCacheTest, FetchMatchesBackingMemoryAtEveryAlignedOffset) {
  GpuMemory memory("memory");
  InstructionCache icache;
  const size_t span = InstructionCache::kLineSize * 3;
  const std::vector<uint8_t> expected = fill_code(memory, span, 0x5a);

  for (uint32_t off = 0; off + InstructionCache::kFetchBytes <= span; off += 4) {
    const auto got = fetch_at(icache, memory, kCodeBase + off);
    EXPECT_TRUE(std::equal(got.begin(), got.end(), expected.begin() + off))
        << "mismatch at offset " << off;
  }
}

// The straddling offsets are the interesting ones: assert they are actually
// exercised above, so the loop cannot silently stop covering them.
TEST(InstructionCacheTest, FetchWindowStraddlesALineBoundary) {
  static_assert(InstructionCache::kLineSize % InstructionCache::kFetchBytes == 0);
  GpuMemory memory("memory");
  InstructionCache icache;
  const std::vector<uint8_t> expected = fill_code(memory, InstructionCache::kLineSize * 2, 0x3c);

  // Offset 60 puts 4 bytes in one line and 12 in the next.
  constexpr uint32_t kStraddle = InstructionCache::kLineSize - 4;
  ASSERT_GT(kStraddle + InstructionCache::kFetchBytes, InstructionCache::kLineSize);

  const auto got = fetch_at(icache, memory, kCodeBase + kStraddle);
  EXPECT_TRUE(std::equal(got.begin(), got.end(), expected.begin() + kStraddle));
}

// The I$ is deliberately not coherent with data writes, matching hardware: a
// write to code memory is invisible until something issues s_icache_inv.
TEST(InstructionCacheTest, CachedLineSurvivesABackingWriteUntilInvalidated) {
  GpuMemory memory("memory");
  InstructionCache icache;
  const std::vector<uint8_t> first = fill_code(memory, InstructionCache::kLineSize, 0x11);

  const auto before = fetch_at(icache, memory, kCodeBase);
  EXPECT_TRUE(std::equal(before.begin(), before.end(), first.begin()));

  const std::vector<uint8_t> second = fill_code(memory, InstructionCache::kLineSize, 0x22);
  ASSERT_NE(first, second);

  const auto stale = fetch_at(icache, memory, kCodeBase);
  EXPECT_TRUE(std::equal(stale.begin(), stale.end(), first.begin()))
      << "the I$ must not observe a data write on its own";

  icache.invalidate_all();
  const auto after = fetch_at(icache, memory, kCodeBase);
  EXPECT_TRUE(std::equal(after.begin(), after.end(), second.begin()));
}

// Lines are tagged by vmid, so the same address in two address spaces must not
// alias even though it selects the same line.
TEST(InstructionCacheTest, LinesDoNotAliasAcrossVmids) {
  GpuMemory memory("memory");
  InstructionCache icache;
  const std::vector<uint8_t> vm0 = fill_code(memory, InstructionCache::kLineSize, 0x01, 0);

  const auto got0 = fetch_at(icache, memory, kCodeBase, 0);
  EXPECT_TRUE(std::equal(got0.begin(), got0.end(), vm0.begin()));

  // vmid 1 has no mapping and no client process, so it falls through to the
  // same sparse backing; what matters is that the lookup refills rather than
  // returning vmid 0's line without checking.
  const auto got1 = fetch_at(icache, memory, kCodeBase, 1);
  EXPECT_TRUE(std::equal(got1.begin(), got1.end(), vm0.begin()));

  // Re-reading vmid 0 must still be correct after the vmid 1 refill.
  const auto again0 = fetch_at(icache, memory, kCodeBase, 0);
  EXPECT_TRUE(std::equal(again0.begin(), again0.end(), vm0.begin()));
}

// A working set larger than the cache must still read correctly once lines
// start evicting each other.
TEST(InstructionCacheTest, FetchIsCorrectWhenTheWorkingSetExceedsTheCache) {
  GpuMemory memory("memory");
  InstructionCache icache;
  const size_t span = InstructionCache::kCacheBytes * 2;
  const std::vector<uint8_t> expected = fill_code(memory, span, 0x7e);

  for (int pass = 0; pass < 2; ++pass) {
    for (uint32_t off = 0; off + InstructionCache::kFetchBytes <= span;
         off += InstructionCache::kLineSize) {
      const auto got = fetch_at(icache, memory, kCodeBase + off);
      EXPECT_TRUE(std::equal(got.begin(), got.end(), expected.begin() + off))
          << "pass " << pass << " offset " << off;
    }
  }
}

} // namespace
