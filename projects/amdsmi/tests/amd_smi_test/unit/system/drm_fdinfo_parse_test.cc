// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Coverage for amd::smi::ParseDrmFdinfo, which amdsmi_get_gpu_process_info_self()
// applies to each /proc/self/fdinfo file.

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

#include "amd_smi/impl/amd_smi_drm_fdinfo_parser.h"

namespace {

// An amdgpu 6.16.6 render node client, trimmed to the fields that matter.
const char* kAmdgpuFdinfo =
    "pos:\t0\n"
    "flags:\t02100002\n"
    "drm-driver:\tamdgpu\n"
    "drm-pdev:\t0000:05:00.0\n"
    "pasid:\t32825\n"
    "drm-total-cpu:\t4132 KiB\n"
    "drm-resident-cpu:\t0\n"
    "drm-total-gtt:\t4168 KiB\n"
    "drm-resident-gtt:\t4136 KiB\n"
    "drm-total-vram:\t65592 KiB\n"
    "drm-resident-vram:\t56 KiB\n"
    "drm-memory-vram:\t56 KiB\n"
    "drm-memory-gtt: \t4136 KiB\n"
    "drm-memory-cpu: \t0 KiB\n"
    "drm-engine-gfx:\t12345 ns\n";

}  // namespace

TEST(SystemUnit, ParseDrmFdinfoReadsAmdgpuClient) {
  const auto usage = amd::smi::ParseDrmFdinfo(kAmdgpuFdinfo);
  EXPECT_EQ(usage.driver, "amdgpu");
  EXPECT_EQ(usage.client, "32825");
  EXPECT_EQ(usage.cpu, 4132ULL << 10);
  EXPECT_EQ(usage.gtt, 4168ULL << 10);
  EXPECT_EQ(usage.gfx_ns, 12345ULL);
  EXPECT_EQ(usage.enc_ns, 0ULL);
}

// drm-memory-vram drops when buffers are evicted; drm-total-vram does not.
TEST(SystemUnit, ParseDrmFdinfoPrefersTotalOverResidentMemory) {
  EXPECT_EQ(amd::smi::ParseDrmFdinfo(kAmdgpuFdinfo).vram, 65592ULL << 10);
}

// Kernels before drm-total-* only print drm-memory-*.
TEST(SystemUnit, ParseDrmFdinfoFallsBackToDrmMemory) {
  const auto usage = amd::smi::ParseDrmFdinfo(
      "drm-driver:\tamdgpu\n"
      "drm-memory-vram:\t2048 KiB\n"
      "drm-memory-gtt: \t4 KiB\n");
  EXPECT_EQ(usage.vram, 2048ULL << 10);
  EXPECT_EQ(usage.gtt, 4ULL << 10);
  EXPECT_EQ(usage.cpu, 0ULL);
}

TEST(SystemUnit, ParseDrmFdinfoPrefersClientIdOverPasid) {
  const auto usage = amd::smi::ParseDrmFdinfo(
      "drm-driver:\tamdgpu\n"
      "drm-client-id:\t7\n"
      "pasid:\t32825\n");
  EXPECT_EQ(usage.client, "7");
}

TEST(SystemUnit, ParseDrmSizeHandlesKernelUnits) {
  uint64_t bytes = 0;
  EXPECT_TRUE(amd::smi::ParseDrmSize("0", &bytes));
  EXPECT_EQ(bytes, 0ULL);
  EXPECT_TRUE(amd::smi::ParseDrmSize("512", &bytes));
  EXPECT_EQ(bytes, 512ULL);
  EXPECT_TRUE(amd::smi::ParseDrmSize("3 KiB", &bytes));
  EXPECT_EQ(bytes, 3ULL << 10);
  EXPECT_TRUE(amd::smi::ParseDrmSize("5 MiB", &bytes));
  EXPECT_EQ(bytes, 5ULL << 20);
  EXPECT_FALSE(amd::smi::ParseDrmSize("", &bytes));
  EXPECT_FALSE(amd::smi::ParseDrmSize("5 PiB", &bytes));
  EXPECT_FALSE(amd::smi::ParseDrmSize("garbage", &bytes));
  EXPECT_FALSE(amd::smi::ParseDrmSize("-5", &bytes));
  EXPECT_FALSE(amd::smi::ParseDrmSize("5 KiB extra", &bytes));
  EXPECT_FALSE(amd::smi::ParseDrmSize("18446744073709551616", &bytes));
  EXPECT_FALSE(amd::smi::ParseDrmSize("17592186044416 MiB", &bytes));
}

// A present counter that does not parse marks the client invalid instead of
// reading as zero, and does not fall back to the resident drm-memory-*.
TEST(SystemUnit, ParseDrmFdinfoFlagsInvalidCounters) {
  const auto bad_total = amd::smi::ParseDrmFdinfo(
      "drm-driver:\tamdgpu\n"
      "drm-total-vram:\tgarbage\n"
      "drm-memory-vram:\t56 KiB\n");
  EXPECT_FALSE(bad_total.valid);
  EXPECT_EQ(bad_total.vram, 0ULL);

  const auto bad_engine = amd::smi::ParseDrmFdinfo(
      "drm-driver:\tamdgpu\n"
      "drm-engine-gfx:\t12 parsecs\n");
  EXPECT_FALSE(bad_engine.valid);

  for (const char* vram :
       {"-5 KiB", "+5 KiB", "18446744073709551616", "17592186044416 MiB", "5 GiB", "5 KiB extra"}) {
    const auto usage = amd::smi::ParseDrmFdinfo(std::string("drm-driver:\tamdgpu\n") +
                                                "drm-total-vram:\t" + vram + "\n");
    EXPECT_FALSE(usage.valid) << vram;
    EXPECT_EQ(usage.vram, 0ULL) << vram;
  }
  const auto largest = amd::smi::ParseDrmFdinfo(
      "drm-driver:\tamdgpu\n"
      "drm-total-vram:\t17592186044415 MiB\n");
  EXPECT_TRUE(largest.valid);
  EXPECT_EQ(largest.vram, 17592186044415ULL << 20);

  EXPECT_TRUE(amd::smi::ParseDrmFdinfo(kAmdgpuFdinfo).valid);
}

// Files that are not DRM clients (pipes, sockets) parse to an empty driver.
TEST(SystemUnit, ParseDrmFdinfoIgnoresNonDrmFiles) {
  const auto usage = amd::smi::ParseDrmFdinfo("pos:\t0\nflags:\t02\nmnt_id:\t15\nino:\t1\n");
  EXPECT_EQ(usage.driver, "");
  EXPECT_EQ(usage.vram, 0ULL);
}
