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

TEST(SysfsTopologyDebugCapabilityTest, Gfx11AdvertisesPreciseDebugBits) {
  Sysfs sysfs;
  std::string topology_dir = sysfs.generate(make_gpu_info(110000u /* gfx1100 */));
  ASSERT_FALSE(topology_dir.empty());

  auto props = read_properties(topology_dir + "/nodes/1/properties");
  ASSERT_TRUE(props.count("capability"));

  const uint32_t capability = static_cast<uint32_t>(props["capability"]);
  EXPECT_TRUE(capability & HSA_CAP_TRAP_DEBUG_SUPPORT);
  EXPECT_TRUE(capability & HSA_CAP_TRAP_DEBUG_PRECISE_ALU_OPERATIONS_SUPPORTED);
  EXPECT_TRUE(capability & HSA_CAP_TRAP_DEBUG_PRECISE_MEMORY_OPERATIONS_SUPPORTED);
}

TEST(SysfsTopologyDebugCapabilityTest, ExplicitCapabilityAndDebugPropArePreserved) {
  Sysfs::GpuInfo gpu = make_gpu_info(110000u /* gfx1100 */);
  gpu.capability = HSA_CAP_TRAP_DEBUG_SUPPORT;
  gpu.debug_prop = HSA_DBG_DISPATCH_INFO_ALWAYS_VALID;

  Sysfs sysfs;
  std::string topology_dir = sysfs.generate(gpu);
  ASSERT_FALSE(topology_dir.empty());

  auto props = read_properties(topology_dir + "/nodes/1/properties");
  EXPECT_EQ(props["capability"], static_cast<uint64_t>(HSA_CAP_TRAP_DEBUG_SUPPORT));
  EXPECT_EQ(props["debug_prop"], static_cast<uint64_t>(HSA_DBG_DISPATCH_INFO_ALWAYS_VALID));
}

} // namespace
