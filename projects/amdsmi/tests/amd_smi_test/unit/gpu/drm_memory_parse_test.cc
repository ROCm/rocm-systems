// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>

#include "amd_smi/impl/amd_smi_utils.h"

// The keys amdgpu prints, as amdgpu_fdinfo.c formats them: KiB, and a space
// before the tab for gtt and cpu.
TEST(GpuUnit, DrmMemoryParseAmdgpuKiB) {
  uint64_t bytes = 0;
  EXPECT_TRUE(
      smi_amdgpu_parse_drm_memory("drm-memory-vram:\t16 KiB\n", "drm-memory-vram:", &bytes));
  EXPECT_EQ(bytes, 16384u);
  EXPECT_TRUE(
      smi_amdgpu_parse_drm_memory("drm-memory-gtt: \t2056 KiB\n", "drm-memory-gtt:", &bytes));
  EXPECT_EQ(bytes, 2105344u);
  EXPECT_TRUE(smi_amdgpu_parse_drm_memory("drm-memory-cpu: \t0 KiB\n", "drm-memory-cpu:", &bytes));
  EXPECT_EQ(bytes, 0u);
}

// drm-usage-stats.rst: a value without a unit is in bytes, and MiB is the other unit.
TEST(GpuUnit, DrmMemoryParseBytesAndMiB) {
  uint64_t bytes = 0;
  EXPECT_TRUE(smi_amdgpu_parse_drm_memory("drm-memory-vram:\t4096\n", "drm-memory-vram:", &bytes));
  EXPECT_EQ(bytes, 4096u);
  EXPECT_TRUE(smi_amdgpu_parse_drm_memory("drm-memory-vram:\t3 MiB\n", "drm-memory-vram:", &bytes));
  EXPECT_EQ(bytes, uint64_t{3} << 20);
}

TEST(GpuUnit, DrmMemoryParseRejectsOtherLines) {
  const uint64_t kUntouched = 7;
  uint64_t bytes = kUntouched;
  // Other keys.
  EXPECT_FALSE(smi_amdgpu_parse_drm_memory("drm-memory-gtt: \t1 KiB", "drm-memory-vram:", &bytes));
  EXPECT_FALSE(smi_amdgpu_parse_drm_memory("drm-total-vram:\t1 KiB", "drm-memory-vram:", &bytes));
  // No value, a malformed or negative one, an unknown unit, and too many bytes for 64 bits.
  EXPECT_FALSE(smi_amdgpu_parse_drm_memory("drm-memory-vram:\t\n", "drm-memory-vram:", &bytes));
  EXPECT_FALSE(
      smi_amdgpu_parse_drm_memory("drm-memory-vram:\t12x KiB", "drm-memory-vram:", &bytes));
  EXPECT_FALSE(smi_amdgpu_parse_drm_memory("drm-memory-vram:\t-1 KiB", "drm-memory-vram:", &bytes));
  EXPECT_FALSE(smi_amdgpu_parse_drm_memory("drm-memory-vram:\t1 GiB", "drm-memory-vram:", &bytes));
  EXPECT_FALSE(smi_amdgpu_parse_drm_memory("drm-memory-vram:\t18014398509481984 KiB",
                                           "drm-memory-vram:", &bytes));
  EXPECT_EQ(bytes, kUntouched);
}
