// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file sysfs_test.cpp
/// @brief Golden tests for the synthetic KFD topology's debug capability bits.
///
/// @details Verifies that Sysfs::generate() advertises the KFD debugger API
/// (HSA_CAP_TRAP_DEBUG_*) capability/debug_prop bits that rocdbgapi's
/// os_driver_kfd.cpp reads to decide whether an agent is debuggable, and that
/// architecture-specific "precise" debug bits are gated correctly.

#include "rocjitsu/kmd/linux/sysfs.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "linux/uapi/kfd_sysfs.h"
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

namespace {

using namespace rocjitsu;

// Reads a KFD sysfs "properties" file (space-separated "key value" lines)
// into a lookup table.
std::unordered_map<std::string, uint64_t> read_properties(const std::string &path) {
  std::unordered_map<std::string, uint64_t> props;
  std::ifstream f(path);
  std::string key;
  uint64_t value = 0;
  while (f >> key >> value)
    props[key] = value;
  return props;
}

Sysfs::GpuInfo make_gpu_info(uint32_t gfx_target_version) {
  Sysfs::GpuInfo gpu{};
  gpu.gpu_id = 1;
  gpu.gfx_target_version = gfx_target_version;
  gpu.marketing_name = "Test GPU";
  gpu.simd_count = 256;
  gpu.num_shader_engines = 8;
  gpu.num_cu_per_sh = 4;
  gpu.local_mem_size = 1ull << 34;
  return gpu;
}

// Golden per-GFXIP expectations. Each row mirrors what
// kfd_topology_set_capabilities() in drivers/gpu/drm/amd/amdkfd/kfd_topology.c
// programs for the corresponding GC hardware IP version. The watch-mask lo/hi
// values are spelled out as literals so the test pins the exact ABI the KFD
// debugger clients (libhsakmt / rocdbgapi) read back.
struct DebugCapExpectation {
  uint32_t gfx_target_version;
  const char *name;
  uint32_t watch_lo;
  uint32_t watch_hi;
  bool dispatch_info_always_valid;
  bool precise_memory;
  bool precise_alu;
  bool per_queue_reset;
  bool lds_out_of_range; // capability2
};

constexpr DebugCapExpectation kDebugCapExpectations[] = {
    // gfx90a: GC 9.4.2 — precise memory, but ttmps are *not* always set up.
    {90010u, "gfx90a", 6, 29, false, true, false, true, false},
    // gfx942: GC 9.4.3 — widened watch mask (lo 7 / hi 30), precise memory.
    {90402u, "gfx942", 7, 30, true, true, false, true, false},
    // gfx950: GC 9.5.0 — precise memory, default (gfx9) watch mask.
    {90500u, "gfx950", 6, 29, true, true, false, true, false},
    // gfx1100: GC 11.0.0 — base debugger only, no precise ops.
    {110000u, "gfx1100", 7, 29, true, false, false, false, false},
    // gfx1200 / gfx1201: GC 12.0.x — precise ALU, not yet precise memory.
    {120000u, "gfx1200", 7, 29, true, false, true, false, false},
    {120001u, "gfx1201", 7, 29, true, false, true, false, false},
    // gfx1250: GC 12.1.0 — precise ALU + memory, per-queue reset, LDS OOR.
    {120500u, "gfx1250", 7, 29, true, true, true, true, true},
};

TEST(SysfsTopologyDebugCapabilityTest, PerGfxipDebugBitsMatchDriver) {
  for (const auto &e : kDebugCapExpectations) {
    SCOPED_TRACE(e.name);

    Sysfs sysfs;
    std::string topology_dir = sysfs.generate(make_gpu_info(e.gfx_target_version));
    ASSERT_FALSE(topology_dir.empty());

    auto props = read_properties(topology_dir + "/nodes/1/properties");
    ASSERT_TRUE(props.count("capability"));
    ASSERT_TRUE(props.count("capability2"));
    ASSERT_TRUE(props.count("debug_prop"));

    const uint32_t cap = static_cast<uint32_t>(props["capability"]);
    const uint32_t cap2 = static_cast<uint32_t>(props["capability2"]);
    const uint64_t dp = props["debug_prop"];

    // Base trap-debugger support is advertised on every simulated GPU.
    EXPECT_TRUE(cap & HSA_CAP_TRAP_DEBUG_SUPPORT);
    EXPECT_TRUE(cap & HSA_CAP_TRAP_DEBUG_WAVE_LAUNCH_TRAP_OVERRIDE_SUPPORTED);
    EXPECT_TRUE(cap & HSA_CAP_TRAP_DEBUG_WAVE_LAUNCH_MODE_SUPPORTED);
    EXPECT_TRUE(cap & HSA_CAP_TRAP_DEBUG_FIRMWARE_SUPPORTED);

    // Address-watch-mask range must match the per-GFXIP driver values exactly.
    const uint32_t lo =
        (dp & HSA_DBG_WATCH_ADDR_MASK_LO_BIT_MASK) >> HSA_DBG_WATCH_ADDR_MASK_LO_BIT_SHIFT;
    const uint32_t hi =
        (dp & HSA_DBG_WATCH_ADDR_MASK_HI_BIT_MASK) >> HSA_DBG_WATCH_ADDR_MASK_HI_BIT_SHIFT;
    EXPECT_EQ(lo, e.watch_lo);
    EXPECT_EQ(hi, e.watch_hi);

    EXPECT_EQ(static_cast<bool>(dp & HSA_DBG_DISPATCH_INFO_ALWAYS_VALID),
              e.dispatch_info_always_valid);
    EXPECT_EQ(static_cast<bool>(cap & HSA_CAP_TRAP_DEBUG_PRECISE_MEMORY_OPERATIONS_SUPPORTED),
              e.precise_memory);
    EXPECT_EQ(static_cast<bool>(cap & HSA_CAP_TRAP_DEBUG_PRECISE_ALU_OPERATIONS_SUPPORTED),
              e.precise_alu);
    EXPECT_EQ(static_cast<bool>(cap & HSA_CAP_PER_QUEUE_RESET_SUPPORTED), e.per_queue_reset);
    EXPECT_EQ(static_cast<bool>(cap2 & HSA_CAP2_TRAP_DEBUG_LDS_OUT_OF_ADDR_RANGE_SUPPORTED),
              e.lds_out_of_range);
  }
}

TEST(SysfsTopologyDebugCapabilityTest, ExplicitCapabilityAndDebugPropArePreserved) {
  Sysfs::GpuInfo gpu = make_gpu_info(110000u /* gfx1100 */);
  gpu.capability = HSA_CAP_TRAP_DEBUG_SUPPORT;
  gpu.capability2 = HSA_CAP2_TRAP_DEBUG_LDS_OUT_OF_ADDR_RANGE_SUPPORTED;
  gpu.debug_prop = HSA_DBG_DISPATCH_INFO_ALWAYS_VALID;

  Sysfs sysfs;
  std::string topology_dir = sysfs.generate(gpu);
  ASSERT_FALSE(topology_dir.empty());

  auto props = read_properties(topology_dir + "/nodes/1/properties");
  EXPECT_EQ(props["capability"], static_cast<uint64_t>(HSA_CAP_TRAP_DEBUG_SUPPORT));
  EXPECT_EQ(props["capability2"],
            static_cast<uint64_t>(HSA_CAP2_TRAP_DEBUG_LDS_OUT_OF_ADDR_RANGE_SUPPORTED));
  EXPECT_EQ(props["debug_prop"], static_cast<uint64_t>(HSA_DBG_DISPATCH_INFO_ALWAYS_VALID));
}

} // namespace
