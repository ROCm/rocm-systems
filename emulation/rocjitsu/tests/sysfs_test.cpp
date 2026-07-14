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

#include "rocjitsu/config/config_loader.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "linux/uapi/kfd_sysfs.h"
RJ_DIAGNOSTIC_POP

#include "embedded_schema.h"

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
  // Full real-hardware node-property words captured from the KFD sysfs
  // "properties" of physical GPUs; each pinned row cites its dump below. Values
  // are decimal to match that ABI verbatim. A row with `capability` != 0 is
  // pinned by the exactness tests; rows without a captured reference leave all
  // three at 0 and are skipped.
  uint32_t capability;
  uint32_t capability2;
  uint64_t debug_prop;
};

constexpr DebugCapExpectation kDebugCapExpectations[] = {
    // gfx90a (MI210): ROCm/k8s-device-plugin testdata/topo-mi210-xgmi-pcie node 2
    // (older-kernel dump has no capability2 line, so capability2 == 0).
    {90010u, "gfx90a", 6, 29, false, true, false, true, false, 746037888u, 0u, 470u},
    // gfx942 (MI300X): rocprofiler-sdk tests/data/topology node 4.
    {90402u, "gfx942", 7, 30, true, true, false, true, false, 2893521536u, 1u, 1511u},
    // gfx950 (MI350X): rocprofiler-sdk tests/data/topology node 5.
    {90500u, "gfx950", 6, 29, true, true, false, true, false, 2889327232u, 1u, 1494u},
    // gfx1100: GC 11.0.0 — base debugger only, no precise ops (no dump yet).
    {110000u, "gfx1100", 7, 29, true, false, false, false, false, 0u, 0u, 0u},
    // gfx1200: GC 12.0.0 — precise ALU, not yet precise memory (no dump yet).
    {120000u, "gfx1200", 7, 29, true, false, true, false, false, 0u, 0u, 0u},
    // gfx1201 (R9700): rocprofiler-sdk tests/data/topology node 6.
    {120001u, "gfx1201", 7, 29, true, false, true, false, false, 1745068672u, 0u, 1495u},
    // gfx1250: GC 12.1.0 — precise ALU + memory, per-queue reset, LDS OOR (no dump yet).
    {120500u, "gfx1250", 7, 29, true, true, true, true, true, 0u, 0u, 0u},
};

// Finds the golden expectation row for a gfx_target_version, or nullptr.
const DebugCapExpectation *find_expectation(uint32_t gfx_target_version) {
  for (const auto &e : kDebugCapExpectations)
    if (e.gfx_target_version == gfx_target_version)
      return &e;
  return nullptr;
}

// Builds the subset of Sysfs::GpuInfo that drives the debug capability node
// properties. gfx_target_version and revision_id feed the auto-computed values;
// explicit capability/capability2/debug_prop (when non-zero) override them. All
// other GpuInfo fields are irrelevant to these three properties.
Sysfs::GpuInfo debug_gpu_info(const config::KfdDeviceConfig &dev) {
  Sysfs::GpuInfo gpu{};
  gpu.gpu_id = dev.gpu_id;
  gpu.gfx_target_version = dev.gfx_target_version;
  gpu.revision_id = dev.revision_id;
  gpu.capability = dev.capability;
  gpu.capability2 = dev.capability2;
  gpu.debug_prop = dev.debug_prop;
  return gpu;
}

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

// The synthetic topology's debug_prop is fully driver-derived, so the
// auto-computed value for a default GPU must match the captured real-hardware
// debug_prop exactly for every GFXIP we have a reference for.
TEST(SysfsTopologyDebugCapabilityTest, DefaultDebugPropMatchesHardware) {
  for (const auto &e : kDebugCapExpectations) {
    if (e.capability == 0)
      continue; // no captured real-hardware reference for this GFXIP
    SCOPED_TRACE(e.name);

    Sysfs sysfs;
    std::string topology_dir = sysfs.generate(make_gpu_info(e.gfx_target_version));
    ASSERT_FALSE(topology_dir.empty());

    auto props = read_properties(topology_dir + "/nodes/1/properties");
    ASSERT_TRUE(props.count("debug_prop"));
    EXPECT_EQ(props["debug_prop"], e.debug_prop);
  }
}

// Loading a shipped config for a real GPU must make the synthetic topology
// advertise that GPU's exact capability/capability2/debug_prop, i.e. the values
// captured from its physical KFD sysfs. The configs carry these as explicit
// overrides (see configs/*.json), which sysfs emits verbatim.
TEST(SysfsTopologyDebugCapabilityTest, ConfigTopologyMatchesRealHardware) {
  const std::string config_dir = CONFIG_DIR;
  // Configs whose device matches a captured real-hardware topology dump.
  constexpr const char *kConfigs[] = {
      "gfx942_cdna3.json",  // MI300X
      "gfx950_cdna4.json",  // MI350X
      "gfx1201_r9700.json", // R9700
  };

  for (const char *cfg : kConfigs) {
    SCOPED_TRACE(cfg);

    auto loaded = config::load_config(config_dir + "/" + cfg, rocjitsu::kEmbeddedSchema);
    ASSERT_TRUE(loaded.device.present);

    const DebugCapExpectation *e = find_expectation(loaded.device.gfx_target_version);
    ASSERT_NE(e, nullptr);
    ASSERT_NE(e->capability, 0u) << "expected a pinned real-hardware reference";

    Sysfs sysfs;
    std::string topology_dir = sysfs.generate(debug_gpu_info(loaded.device));
    ASSERT_FALSE(topology_dir.empty());

    auto props = read_properties(topology_dir + "/nodes/1/properties");
    ASSERT_TRUE(props.count("capability"));
    ASSERT_TRUE(props.count("capability2"));
    ASSERT_TRUE(props.count("debug_prop"));

    // The synthetic topology re-derives the HSA_CAP_ASIC_REVISION field from the
    // config's revision_id, so compare the feature bits with that field masked
    // out (the captured dumps carry the physical part's revision there).
    const uint32_t asic_mask = static_cast<uint32_t>(HSA_CAP_ASIC_REVISION_MASK);
    EXPECT_EQ(static_cast<uint32_t>(props["capability"]) & ~asic_mask, e->capability & ~asic_mask);
    EXPECT_EQ(static_cast<uint32_t>(props["capability2"]), e->capability2);
    EXPECT_EQ(props["debug_prop"], e->debug_prop);
  }
}

} // namespace
