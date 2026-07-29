/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

#include "rocm_smi/rocm_smi_vram.h"

namespace {

constexpr uint64_t kSampleVramTotal = 128ULL * 1024 * 1024 * 1024;  // 128 GiB
constexpr uint64_t kApuCarveout = 512ULL * 1024 * 1024;             // 512 MiB BIOS carveout
constexpr uint64_t kApuUnified = 110ULL * 1024 * 1024 * 1024;       // 110 GiB unified pool
// Real MI300X SPX values observed on hardware: sysfs and KFD agree byte-for-byte.
constexpr uint64_t kMi300xSpxTotal = 206141652992ULL;  // ~192 GiB

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

// MI300X SPX: sysfs and KFD report the same value, so the final clause is
// false and the sysfs value is kept. Values confirmed on hardware.
TEST(GpuUnit, VramTotalMi300xSpxKeepsSysfs) {
  EXPECT_FALSE(amd::smi::vram_total_prefer_kfd(true, kMi300xSpxTotal, "SPX", kMi300xSpxTotal));
}

// Discrete GPU: the KFD mem_banks total is not larger than sysfs, so the final
// clause stays false and the sysfs value is kept.
TEST(GpuUnit, VramTotalDiscreteKeepsSysfs) {
  const uint64_t kfd_smaller = kSampleVramTotal - 1;
  EXPECT_FALSE(amd::smi::vram_total_prefer_kfd(true, kSampleVramTotal, "SPX", kfd_smaller));
  EXPECT_FALSE(amd::smi::vram_total_prefer_kfd(true, kSampleVramTotal, "", kfd_smaller));
}
