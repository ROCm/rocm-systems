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

namespace amd::smi {

// Internal helper under test. It decides whether the KFD topology total should
// be preferred over the sysfs mem_info_vram_total value for VRAM (see
// rsmi_dev_memory_total_get in rocm_smi.cc / SWDEV-536184).
bool vram_total_prefer_kfd(bool sysfs_read_ok, uint64_t sysfs_total,
                           const std::string& compute_partition, uint64_t kfd_total);

}  // namespace amd::smi

namespace {

constexpr uint64_t kSampleVramTotal = 128ULL * 1024 * 1024 * 1024;  // 128 GiB
constexpr uint64_t kApuCarveout = 512ULL * 1024 * 1024;             // 512 MiB BIOS carveout
constexpr uint64_t kApuUnified = 110ULL * 1024 * 1024 * 1024;       // 110 GiB unified pool

}  // namespace

// When the sysfs read failed, the KFD total is always preferred, regardless of
// the reported partition mode.
TEST(AmdSmiVramTotalPreferKfdTest, UnusableSysfsAlwaysPrefersKfd) {
  for (const char* mode : {"", "SPX", "CPX", "DPX", "TPX", "QPX"}) {
    EXPECT_TRUE(amd::smi::vram_total_prefer_kfd(false, kSampleVramTotal, mode, kSampleVramTotal))
        << "Failed sysfs read must fall back to KFD (mode=" << mode << ")";
  }
}

// A zero sysfs total is unusable and must fall back to the KFD total.
TEST(AmdSmiVramTotalPreferKfdTest, ZeroSysfsTotalPrefersKfd) {
  for (const char* mode : {"", "SPX", "CPX", "DPX", "TPX", "QPX"}) {
    EXPECT_TRUE(amd::smi::vram_total_prefer_kfd(true, 0, mode, kSampleVramTotal))
        << "Zero sysfs total must fall back to KFD (mode=" << mode << ")";
  }
}

// With a usable sysfs value on a non-partitioned GPU (SPX or no partition
// string), the sysfs value is trusted and the KFD total is not preferred.
TEST(AmdSmiVramTotalPreferKfdTest, UsableSysfsNonPartitionedKeepsSysfs) {
  EXPECT_FALSE(amd::smi::vram_total_prefer_kfd(true, kSampleVramTotal, "SPX", kSampleVramTotal));
  EXPECT_FALSE(amd::smi::vram_total_prefer_kfd(true, kSampleVramTotal, "", kSampleVramTotal));
}

// With a usable sysfs value in a multi-partition compute mode, the sysfs value
// reports the whole device split evenly across partitions and is misleading, so
// the per-partition KFD total must be preferred.
TEST(AmdSmiVramTotalPreferKfdTest, UsableSysfsPartitionedPrefersKfd) {
  for (const char* mode : {"CPX", "DPX", "TPX", "QPX"}) {
    EXPECT_TRUE(amd::smi::vram_total_prefer_kfd(true, kSampleVramTotal, mode, kSampleVramTotal / 6))
        << "Partition mode " << mode << " must prefer the KFD per-partition total";
  }
}

// APU (e.g. gfx1151 / Strix Halo): sysfs reports only the small BIOS VRAM
// carveout while KFD reports the true unified pool, which must win.
TEST(AmdSmiVramTotalPreferKfdTest, ApuCarveoutPrefersKfd) {
  EXPECT_TRUE(amd::smi::vram_total_prefer_kfd(true, kApuCarveout, "", kApuUnified));
  EXPECT_TRUE(amd::smi::vram_total_prefer_kfd(true, kApuCarveout, "SPX", kApuUnified));
}

// Discrete GPU: sysfs and KFD agree, so the sysfs value is kept.
TEST(AmdSmiVramTotalPreferKfdTest, DiscreteAgreeingSourcesKeepsSysfs) {
  EXPECT_FALSE(amd::smi::vram_total_prefer_kfd(true, kSampleVramTotal, "SPX", kSampleVramTotal));
  EXPECT_FALSE(amd::smi::vram_total_prefer_kfd(true, kSampleVramTotal, "", kSampleVramTotal));
}
