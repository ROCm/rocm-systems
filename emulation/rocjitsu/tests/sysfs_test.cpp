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
#include "rocjitsu/kmd/linux/amdgpu_properties.h"
#include "rocjitsu/kmd/linux/cwsr.h"
#include "rocjitsu/kmd/linux/kfd_topology.h"
#include "rocjitsu/vm/soc.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "linux/uapi/kfd_sysfs.h"
RJ_DIAGNOSTIC_POP

#include "embedded_schema.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
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

std::string read_sysfs_file(const std::string &path) {
  std::ifstream f(path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
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

TEST(SysfsTopologyTest, DefaultGpuInfoHasCoherentSdmaCounts) {
  Sysfs sysfs;
  std::string topology_dir = sysfs.generate(make_gpu_info(90500u /* gfx950 */));
  ASSERT_FALSE(topology_dir.empty());

  auto props = read_properties(topology_dir + "/nodes/1/properties");
  ASSERT_TRUE(props.count("num_sdma_engines"));
  ASSERT_TRUE(props.count("num_sdma_queues_per_engine"));
  EXPECT_EQ(props["num_sdma_engines"], 0u);
  EXPECT_EQ(props["num_sdma_queues_per_engine"], 0u);
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
    // TODO(hanchung): Replace the inferred gfx1250 feature row and pin its
    // capability/capability2/debug_prop words after capturing a physical MI455X
    // KFD topology.
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

// KFD's node_show() publishes a "name" sysfs file for every topology node.
// GPU nodes carry the marketing name; the CPU node is present but empty. Clients
// that walk the topology tree expect the file to exist on node 0 as well.
TEST(SysfsTopologyTest, CpuNodePublishesNameFile) {
  Sysfs sysfs;
  std::string topology_dir = sysfs.generate(make_gpu_info(90402u /* gfx942 */));
  ASSERT_FALSE(topology_dir.empty());

  const std::string cpu_name_path = topology_dir + "/nodes/0/name";
  ASSERT_TRUE(std::filesystem::exists(cpu_name_path));
  EXPECT_EQ(read_sysfs_file(cpu_name_path), "\n");

  const std::string gpu_name_path = topology_dir + "/nodes/1/name";
  ASSERT_TRUE(std::filesystem::exists(gpu_name_path));
  EXPECT_EQ(read_sysfs_file(gpu_name_path), "Test GPU\n");
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

    // Base trap-debugger support is advertised on every GPU the CWSR codec can
    // produce a record for, and withheld on the rest: rocdbgapi reads wave state
    // out of the save area, so an agent whose record layout the simulator does
    // not model cannot be serviced and must be declined at attach rather than at
    // every wave stop. The sub-capabilities below stay driver-derived either
    // way; they describe a support that is simply no longer claimed.
    EXPECT_EQ(static_cast<bool>(cap & HSA_CAP_TRAP_DEBUG_SUPPORT),
              rocjitsu::kmd::cwsr_layout_modelled_for_gc_ip_version(
                  rocjitsu::kmd::gc_ip_version_for_gfx_target_version(e.gfx_target_version)));
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
  // gfx942, so the trap-debug bit survives the override and this stays a test
  // about the override rather than about the CWSR gate; the gfx1100 case is
  // CapturedCapabilityCannotReadvertiseUnservicableTrapDebug's.
  Sysfs::GpuInfo gpu = make_gpu_info(90402u /* gfx942 */);
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

TEST(SysfsTopologyDebugCapabilityTest, Mi455xConfigPublishesB0AsicRevision) {
  auto loaded = config::load_config(std::string(CONFIG_DIR) + "/gfx1250_mi455x.json",
                                    rocjitsu::kEmbeddedSchema);
  ASSERT_TRUE(loaded.device.present);
  ASSERT_EQ(loaded.device.revision_id, 1u);

  Sysfs sysfs;
  std::string topology_dir = sysfs.generate(debug_gpu_info(loaded.device));
  ASSERT_FALSE(topology_dir.empty());

  auto props = read_properties(topology_dir + "/nodes/1/properties");
  ASSERT_TRUE(props.count("capability"));
  const uint32_t capability = static_cast<uint32_t>(props["capability"]);
  const uint32_t asic_revision =
      (capability & HSA_CAP_ASIC_REVISION_MASK) >> HSA_CAP_ASIC_REVISION_SHIFT;
  EXPECT_EQ(asic_revision, loaded.device.revision_id);
}

// The address-watch register count is the one capability field a debugger acts
// on numerically rather than as a flag: rocdbgapi recovers it as 1<<TOTALBITS
// (os_driver_kfd.cpp) and refuses to insert more watchpoints than that. Every
// captured dump reads TOTALBITS 2 for the driver's num_of_watch_points = 4, so
// the derived topology must too -- packing the count itself would claim sixteen
// registers the simulated ASIC does not have.
TEST(SysfsTopologyDebugCapabilityTest, DefaultWatchPointCountMatchesHardware) {
  for (const auto &e : kDebugCapExpectations) {
    if (e.capability == 0)
      continue; // no captured real-hardware reference for this GFXIP
    SCOPED_TRACE(e.name);

    Sysfs sysfs;
    std::string topology_dir = sysfs.generate(make_gpu_info(e.gfx_target_version));
    ASSERT_FALSE(topology_dir.empty());

    auto props = read_properties(topology_dir + "/nodes/1/properties");
    ASSERT_TRUE(props.count("capability"));
    const auto cap = static_cast<uint32_t>(props["capability"]);

    EXPECT_TRUE(cap & HSA_CAP_WATCH_POINTS_SUPPORTED);
    EXPECT_TRUE(e.capability & HSA_CAP_WATCH_POINTS_SUPPORTED);

    const uint32_t total_bits =
        (cap & HSA_CAP_WATCH_POINTS_TOTALBITS_MASK) >> HSA_CAP_WATCH_POINTS_TOTALBITS_SHIFT;
    EXPECT_EQ(total_bits, (e.capability & HSA_CAP_WATCH_POINTS_TOTALBITS_MASK) >>
                              HSA_CAP_WATCH_POINTS_TOTALBITS_SHIFT);
    EXPECT_EQ(1u << total_bits, kmd::kNumWatchPoints);
  }
}

// KFD reports array_count per node, not per XCC: node_show() emits
// node_props.array_count * NUM_XCC while simd_arrays_per_engine and
// cu_per_simd_array stay per-XCC. Both gfx942 configs model MI300X, whose
// captured sysfs reads array_count 32 / simd_arrays_per_engine 1 /
// cu_per_simd_array 10 at num_xcc 8, so the generated topology must match it
// exactly — a debugger divides these back out to recover shader engines.
//
// Both are checked, not just the _kmd one. Several representations multiply out
// to the same 40 CUs per XCD (4 engines of two 5-CU arrays reaches it just as
// well as 4 engines of one 10-CU array), so a config can satisfy every product
// this suite checks and still publish node properties no MI300X ever reports.
// Two configs for one part have to describe it the same way, or the debugger's
// view depends on which one the run happened to load.
TEST(SysfsTopologyGeometryTest, ArrayCountIsScaledByNumXcc) {
  const std::string config_dir = CONFIG_DIR;
  constexpr const char *kMi300xConfigs[] = {"gfx942_cdna3_kmd.json", "gfx942_cdna3.json"};

  for (const char *cfg : kMi300xConfigs) {
    SCOPED_TRACE(cfg);
    auto loaded = config::load_config(config_dir + "/" + cfg, rocjitsu::kEmbeddedSchema);
    ASSERT_TRUE(loaded.device.present);
    const uint32_t num_xcc = loaded.soc()->num_xcds();
    ASSERT_EQ(num_xcc, 8u);

    Sysfs sysfs;
    std::string topology_dir = sysfs.generate(gpu_info_from_config(loaded.device, num_xcc));
    ASSERT_FALSE(topology_dir.empty());

    auto props = read_properties(topology_dir + "/nodes/1/properties");
    ASSERT_TRUE(props.count("array_count"));
    // Fatal: the derivation below divides by this, and operator[] would silently
    // insert a zero for a renamed or dropped property.
    ASSERT_TRUE(props.count("simd_arrays_per_engine"));
    ASSERT_NE(props["simd_arrays_per_engine"], 0u);
    EXPECT_EQ(props["array_count"], 32u);
    EXPECT_EQ(props["simd_arrays_per_engine"], 1u);
    EXPECT_EQ(props["cu_per_simd_array"], 10u);
    EXPECT_EQ(props["num_xcc"], num_xcc);

    // The same dump reads simd_count 1216, i.e. 304 of the 320 CUs the array
    // geometry above holds: MI300X is harvested, and KFD publishes the physical
    // arrays with the active SIMD total beside them. Pinned to the capture
    // rather than to the product, because the product is what the part is not.
    // Both configs are checked here for the same reason the geometry is --
    // kfd_debug.c copies this field into the debugger's device entry verbatim,
    // so a pair that disagrees reports one part two ways.
    EXPECT_EQ(props["simd_count"], 1216u);
    EXPECT_EQ(props["simd_per_cu"], 4u);
    EXPECT_EQ(props["num_sdma_queues_per_engine"], 8u);

    // What libhsakmt derives from those: NumShaderBanks = array_count /
    // simd_arrays_per_engine, i.e. the node's total shader engines.
    EXPECT_EQ(props["array_count"] / props["simd_arrays_per_engine"],
              num_xcc * loaded.soc()->xcd(0)->num_shader_engines());
  }
}

// MI350X is harvested too: its captured KFD topology has 32 shader arrays of
// 9 CUs (288 physical CUs), while simd_count exposes 1024 SIMDs, or 256 active
// CUs at four SIMDs per CU. Keep every full-device gfx950 config pinned to that
// same capture so local, KMD, and multi-GPU launch paths cannot drift apart.
TEST(SysfsTopologyGeometryTest, Mi350xMatchesCapturedPhysicalAndActiveCuCounts) {
  const std::string config_dir = CONFIG_DIR;
  constexpr const char *kMi350xConfigs[] = {"gfx950_mi355x.json", "gfx950_mi355x_kmd.json",
                                            "gfx950_mi355x_kmd_2gpu.json"};

  for (const char *cfg : kMi350xConfigs) {
    SCOPED_TRACE(cfg);
    auto loaded = config::load_config(config_dir + "/" + cfg, rocjitsu::kEmbeddedSchema);
    ASSERT_TRUE(loaded.device.present);
    ASSERT_NE(loaded.soc(), nullptr);
    const uint32_t num_xcc = loaded.soc()->num_xcds();
    ASSERT_EQ(num_xcc, 8u);

    Sysfs sysfs;
    std::string topology_dir = sysfs.generate(gpu_info_from_config(loaded.device, num_xcc));
    ASSERT_FALSE(topology_dir.empty());
    auto props = read_properties(topology_dir + "/nodes/1/properties");

    ASSERT_TRUE(props.count("array_count"));
    ASSERT_TRUE(props.count("simd_arrays_per_engine"));
    ASSERT_TRUE(props.count("cu_per_simd_array"));
    ASSERT_TRUE(props.count("simd_count"));
    ASSERT_TRUE(props.count("simd_per_cu"));
    EXPECT_EQ(props["array_count"], 32u);
    EXPECT_EQ(props["simd_arrays_per_engine"], 1u);
    EXPECT_EQ(props["cu_per_simd_array"], 9u);
    EXPECT_EQ(props["simd_count"], 1024u);
    EXPECT_EQ(props["simd_per_cu"], 4u);

    const uint64_t physical_cus = props["array_count"] * props["cu_per_simd_array"];
    const uint64_t active_cus = props["simd_count"] / props["simd_per_cu"];
    EXPECT_EQ(physical_cus, 288u);
    EXPECT_EQ(active_cus, 256u);

    EXPECT_EQ(loaded.device.device_id, 30112u);
    EXPECT_EQ(loaded.device.local_mem_size, 309220868096ULL);
    EXPECT_EQ(loaded.device.mem_clk_max, 1900u);
    EXPECT_EQ(loaded.device.num_sdma_engines, 2u);
    EXPECT_EQ(loaded.device.num_sdma_xgmi_engines, 14u);
    EXPECT_EQ(loaded.device.num_sdma_queues_per_engine, 8u);
    ASSERT_TRUE(props.count("num_sdma_queues_per_engine"));
    EXPECT_EQ(props["num_sdma_queues_per_engine"], 8u);
    EXPECT_EQ(loaded.device.num_cp_queues, 24u);
    EXPECT_EQ(loaded.device.max_engine_clk_fcompute, 2200u);
  }
}

// TODO(hanchung): Pin the physical CU geometry, active-CU mask, memory-bank
// size, clocks, GPU/unique IDs, SDMA engine/queue counts, CP queue count, family
// ID, and ASIC/PCI revisions after capturing a physical MI455X KFD topology.
// Until then, this test covers only the checked-in physical device identity and
// must not be treated as exact rocminfo parity.
TEST(SysfsTopologyIdentityTest, Mi455xUsesPublishedPhysicalDeviceId) {
  const std::string config_dir = CONFIG_DIR;
  auto loaded = config::load_config(config_dir + "/gfx1250_mi455x.json", rocjitsu::kEmbeddedSchema);
  ASSERT_TRUE(loaded.device.present);
  ASSERT_NE(loaded.soc(), nullptr);

  Sysfs sysfs;
  std::string topology_dir =
      sysfs.generate(gpu_info_from_config(loaded.device, loaded.soc()->num_xcds()));
  ASSERT_FALSE(topology_dir.empty());
  auto props = read_properties(topology_dir + "/nodes/1/properties");

  ASSERT_TRUE(props.count("vendor_id"));
  ASSERT_TRUE(props.count("device_id"));
  EXPECT_EQ(props["vendor_id"], 4098u);
  EXPECT_EQ(props["device_id"], 30145u);
}

// Every shipped config must describe the machine it actually simulates.
//
// num_shader_engines is the per-XCC shader-engine count, so it has to equal the
// SoC's own se[] count -- KFD's array_count is derived from it, not stored in
// it. And the CU geometry the topology advertises has to multiply back out to
// the declared simd_count, or a runtime sizing scratch and CWSR from the
// reported CU count provisions for a machine the simulator does not have.
//
// simd_count itself can only be bounded here, not derived -- the shortfall on a
// harvested part is a fact about the silicon. What can be checked is that every
// config modelling the same part carries the same one, which is done below.
//
// \NPI new GPU: a config added to configs/ is picked up here automatically.
TEST(SysfsTopologyGeometryTest, ShippedConfigsMatchTheSimulatedSoC) {
  const std::filesystem::path config_dir = CONFIG_DIR;
  unsigned checked = 0;

  // One part may ship as several configs -- a KMD capture, a sibling that models
  // more of the SoC, a multi-GPU variant -- and they have to advertise the same
  // machine. Keyed by the part, first config seen wins and the rest are compared
  // against it.
  struct PartGeometry {
    std::string config;
    uint32_t simd_count;
    uint32_t num_shader_engines;
    uint32_t arrays_per_engine;
    uint32_t num_cu_per_sh;
    uint32_t simd_per_cu;
    uint32_t num_xcc;
  };
  std::unordered_map<std::string, PartGeometry> parts;

  for (const auto &entry : std::filesystem::directory_iterator(config_dir)) {
    if (entry.path().extension() != ".json")
      continue;
    const std::string name = entry.path().filename().string();
    // DBT guest configs describe a synthetic node with no SoC behind it; their
    // geometry is validated by validate_guest_device_geometry() at load time.
    if (name.rfind("guest_", 0) == 0)
      continue;
    SCOPED_TRACE(name);

    auto loaded = config::load_config(entry.path().string(), rocjitsu::kEmbeddedSchema);
    if (!loaded.device.present || loaded.soc() == nullptr)
      continue;
    ++checked;

    const uint32_t num_xcc = loaded.soc()->num_xcds();
    ASSERT_NE(num_xcc, 0u);
    EXPECT_EQ(loaded.device.num_shader_engines, loaded.soc()->xcd(0)->num_shader_engines())
        << "num_shader_engines must be the SoC's shader-engine count, not the array count";

    // Shader arrays partition an engine's CUs, so the CU total is
    // engines * arrays/engine * CUs/array, per XCC and then across XCCs.
    const uint32_t arrays_per_engine = loaded.device.num_shader_arrays_per_engine;
    ASSERT_NE(arrays_per_engine, 0u);
    const uint64_t cus = static_cast<uint64_t>(loaded.device.num_shader_engines) *
                         arrays_per_engine * loaded.device.num_cu_per_sh * num_xcc;

    // Against the SoC, not against the device block. Deriving the CU total from
    // the config and then checking the config's own simd_count against it only
    // catches a config that contradicts itself -- both sides come from the same
    // declaration, so it passes for any config whose numbers multiply out,
    // however unlike the machine underneath. The whole point of this test is
    // the comparison with what the simulator instantiates.
    uint64_t soc_cus = 0;
    for (uint32_t xcd_index = 0; xcd_index < num_xcc; ++xcd_index) {
      const auto *xcd = loaded.soc()->xcd(xcd_index);
      ASSERT_NE(xcd, nullptr);
      for (uint32_t se_index = 0; se_index < xcd->num_shader_engines(); ++se_index)
        soc_cus += xcd->shader_engine(se_index)->num_compute_units();
    }
    EXPECT_EQ(cus, soc_cus) << "the advertised CU geometry is not the machine the simulator runs: "
                            << cus << " advertised vs " << soc_cus << " instantiated";

    // Harvested parts ship with fewer active CUs than the array geometry holds
    // (MI300X reports 304 of 320), so the advertised simd_count may be lower --
    // never higher, which would mean SIMDs with nowhere to live.
    const uint64_t simds = soc_cus * loaded.device.simd_per_cu;
    EXPECT_LE(loaded.device.simd_count, simds)
        << "simd_count exceeds the simulated CU geometry: " << soc_cus << " CUs * "
        << loaded.device.simd_per_cu << " SIMDs";

    // That bound is all a lone config can be held to -- how far below the array
    // geometry a harvested part sits is a property of the part, not something
    // any product here can derive. So simd_count is carried alongside the
    // geometry instead: every config for one part must agree on all of it.
    // Nothing downstream reconciles a disagreement. sysfs publishes the loaded
    // config's simd_count as the node property and debug_device_snapshot()
    // copies it into the debugger's device entry, so two configs for one GPU
    // that differ here report that GPU's active-SIMD count two ways depending
    // on which one the run happened to load.
    const PartGeometry geometry{name,
                                loaded.device.simd_count,
                                loaded.device.num_shader_engines,
                                arrays_per_engine,
                                loaded.device.num_cu_per_sh,
                                loaded.device.simd_per_cu,
                                num_xcc};
    const std::string part =
        std::to_string(loaded.device.gfx_target_version) + " " + loaded.device.marketing_name;
    const auto [it, inserted] = parts.emplace(part, geometry);
    if (!inserted) {
      const PartGeometry &first = it->second;
      SCOPED_TRACE("same part as " + first.config + " (" + part + ")");
      EXPECT_EQ(geometry.simd_count, first.simd_count)
          << "configs for one part disagree on the active SIMD count";
      EXPECT_EQ(geometry.num_shader_engines, first.num_shader_engines);
      EXPECT_EQ(geometry.arrays_per_engine, first.arrays_per_engine);
      EXPECT_EQ(geometry.num_cu_per_sh, first.num_cu_per_sh);
      EXPECT_EQ(geometry.simd_per_cu, first.simd_per_cu);
      EXPECT_EQ(geometry.num_xcc, first.num_xcc);
    }
  }

  EXPECT_GE(checked, 5u) << "expected the shipped device configs to be discovered";
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
      "gfx950_mi355x.json", // MI350X
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
    //
    // HSA_CAP_TRAP_DEBUG_SUPPORT is masked out for a different reason: it is the
    // one bit the simulator deliberately does not reproduce faithfully. A part
    // whose CWSR record layout the codec cannot produce is not debuggable here
    // however debuggable the physical device is, so the bit is withheld and
    // checked below against the gate instead of against the dump.
    const uint32_t asic_mask = static_cast<uint32_t>(HSA_CAP_ASIC_REVISION_MASK) |
                               static_cast<uint32_t>(HSA_CAP_TRAP_DEBUG_SUPPORT);
    EXPECT_EQ(static_cast<uint32_t>(props["capability"]) & ~asic_mask, e->capability & ~asic_mask);
    EXPECT_TRUE(e->capability & HSA_CAP_TRAP_DEBUG_SUPPORT)
        << "the physical part advertises it, so the simulator's value is a real deviation";
    EXPECT_EQ(
        static_cast<bool>(static_cast<uint32_t>(props["capability"]) & HSA_CAP_TRAP_DEBUG_SUPPORT),
        kmd::cwsr_layout_modelled_for_gc_ip_version(
            kmd::gc_ip_version_for_gfx_target_version(loaded.device.gfx_target_version)));
    EXPECT_EQ(static_cast<uint32_t>(props["capability2"]), e->capability2);
    EXPECT_EQ(props["debug_prop"], e->debug_prop);
  }
}

} // namespace
