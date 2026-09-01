// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "rocm_smi/rocm_smi_vram.h"

namespace {

constexpr uint64_t kSampleVramTotal = 128ULL * 1024 * 1024 * 1024;  // 128 GiB
constexpr uint64_t kApuCarveout = 512ULL * 1024 * 1024;             // 512 MiB BIOS carveout
constexpr uint64_t kApuUnified = 110ULL * 1024 * 1024 * 1024;       // 110 GiB unified pool

}  // namespace

// Failed sysfs read (the MI300A path, where mem_info_vram_total is absent)
// always prefers KFD, regardless of partition mode.
TEST(GpuUnit, VramTotalUnusableSysfsPrefersKfd) {
  for (const char* mode : {"", "SPX", "CPX", "DPX", "TPX", "QPX"}) {
    EXPECT_TRUE(amd::smi::vram_total_prefer_kfd(false, kSampleVramTotal, mode, kSampleVramTotal))
        << "Failed sysfs read must fall back to KFD (mode=" << mode << ")";
  }
  // The helper does not guard on kfd_total; the caller substitutes only when
  // kfd_total > 0. A failed read still prefers KFD even when kfd_total is 0.
  EXPECT_TRUE(amd::smi::vram_total_prefer_kfd(false, kSampleVramTotal, "SPX", 0));
}

// A zero sysfs total is unusable and must fall back to the KFD total.
TEST(GpuUnit, VramTotalZeroSysfsPrefersKfd) {
  for (const char* mode : {"", "SPX", "CPX", "DPX", "TPX", "QPX"}) {
    EXPECT_TRUE(amd::smi::vram_total_prefer_kfd(true, 0, mode, kSampleVramTotal))
        << "Zero sysfs total must fall back to KFD (mode=" << mode << ")";
  }
  // A zero sysfs total is unusable regardless of the KFD total.
  EXPECT_TRUE(amd::smi::vram_total_prefer_kfd(true, 0, "SPX", 0));
}

// Usable sysfs on a non-partitioned GPU (SPX or empty) is trusted; KFD is not
// preferred.
TEST(GpuUnit, VramTotalUsableNonPartitionedKeepsSysfs) {
  EXPECT_FALSE(amd::smi::vram_total_prefer_kfd(true, kSampleVramTotal, "SPX", kSampleVramTotal));
  EXPECT_FALSE(amd::smi::vram_total_prefer_kfd(true, kSampleVramTotal, "", kSampleVramTotal));
  // A zero KFD total must never override a usable sysfs value.
  EXPECT_FALSE(amd::smi::vram_total_prefer_kfd(true, kSampleVramTotal, "SPX", 0));
  EXPECT_FALSE(amd::smi::vram_total_prefer_kfd(true, kSampleVramTotal, "", 0));
}

// In a multi-partition mode, sysfs reports the whole device split evenly and is
// misleading, so the per-partition KFD total must be preferred. The KFD value
// here is smaller than sysfs, so the result is driven by the partition clause,
// not the size heuristic.
TEST(GpuUnit, VramTotalUsablePartitionedPrefersKfd) {
  for (const char* mode : {"CPX", "DPX", "TPX", "QPX"}) {
    EXPECT_TRUE(amd::smi::vram_total_prefer_kfd(true, kSampleVramTotal, mode, kSampleVramTotal / 6))
        << "Partition mode " << mode << " must prefer the KFD per-partition total";
  }
}

// APU (e.g. gfx1151 / Strix Halo): sysfs reports only the small BIOS VRAM
// carveout while KFD reports the true, larger unified pool, which must win.
TEST(GpuUnit, VramTotalApuCarveoutPrefersKfd) {
  EXPECT_TRUE(amd::smi::vram_total_prefer_kfd(true, kApuCarveout, "", kApuUnified));
  EXPECT_TRUE(amd::smi::vram_total_prefer_kfd(true, kApuCarveout, "SPX", kApuUnified));
}

// Discrete GPU: the KFD mem_banks total is not larger than sysfs, so the final
// clause stays false and the sysfs value is kept.
TEST(GpuUnit, VramTotalDiscreteKeepsSysfs) {
  const uint64_t kfd_smaller = kSampleVramTotal - 1;
  EXPECT_FALSE(amd::smi::vram_total_prefer_kfd(true, kSampleVramTotal, "SPX", kfd_smaller));
  EXPECT_FALSE(amd::smi::vram_total_prefer_kfd(true, kSampleVramTotal, "", kfd_smaller));
}

namespace {

// KFD heap types (HSA_MEM_HEAP_TYPE_*): only FB_PUBLIC counts as user-visible VRAM.
constexpr uint32_t kHeapFbPublic = 1;
constexpr uint32_t kHeapFbPrivate = 2;
constexpr uint32_t kHeapGpuScratch = 5;
constexpr uint64_t kGiB = 1024ULL * 1024 * 1024;

}  // namespace

// A single FB_PUBLIC bank, the layout amdkfd emits for a large-BAR GPU node.
TEST(GpuUnit, SumPublicVramSingleFbPublicBank) {
  EXPECT_EQ(amd::smi::sum_public_vram_bytes({{kHeapFbPublic, 94ULL * kGiB}}), 94ULL * kGiB);
}

// Non-public heaps are excluded when an FB_PUBLIC bank is present.
TEST(GpuUnit, SumPublicVramExcludesNonPublicHeaps) {
  const std::vector<amd::smi::KfdMemBank> banks = {
      {kHeapFbPublic, 32ULL * kGiB},
      {kHeapFbPrivate, 4ULL * kGiB},
      {kHeapGpuScratch, 1ULL * kGiB},
  };
  EXPECT_EQ(amd::smi::sum_public_vram_bytes(banks), 32ULL * kGiB);
}

// Multiple FB_PUBLIC banks are summed.
TEST(GpuUnit, SumPublicVramSumsMultiplePublicBanks) {
  const std::vector<amd::smi::KfdMemBank> banks = {
      {kHeapFbPublic, 16ULL * kGiB},
      {kHeapFbPublic, 16ULL * kGiB},
  };
  EXPECT_EQ(amd::smi::sum_public_vram_bytes(banks), 32ULL * kGiB);
}

// A node with no FB_PUBLIC heap falls back to summing every bank. This is the
// small-BAR and APU layout, where amdkfd reports the whole framebuffer as a
// single FB_PRIVATE bank; without the fallback those nodes would report zero.
TEST(GpuUnit, SumPublicVramFallsBackWhenNoPublicHeap) {
  const std::vector<amd::smi::KfdMemBank> banks = {{kHeapFbPrivate, 110ULL * kGiB}};
  EXPECT_EQ(amd::smi::sum_public_vram_bytes(banks), 110ULL * kGiB);
}

// No banks at all yields zero.
TEST(GpuUnit, SumPublicVramEmptyIsZero) { EXPECT_EQ(amd::smi::sum_public_vram_bytes({}), 0ULL); }
