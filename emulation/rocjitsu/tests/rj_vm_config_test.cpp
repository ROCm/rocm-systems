// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/api.h"

#include "scoped_temp.h"
#include "test_paths.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <set>
#include <string>
#include <string_view>

namespace {

using Config = std::unique_ptr<rj_vm_config_t, decltype(&rj_vm_config_destroy)>;
using Vm = std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)>;

std::string small_config(bool replicated, std::string_view system_fields = {}) {
  return std::string(R"({
    "max_ticks":100,"num_threads":1,"cpu_dispatch_threads":1,
    "vm":{"arch":"cdna5","gpu":{)") +
         (replicated ? R"("num_gpus":2,)" : "") +
         R"("pci":{"vram_aperture_bytes":2097152},"device":{
      "gpu_id":1250,"gfx_target_version":120500,"unique_id":1250,
      "marketing_name":"Discovery fixture","hive_id":42,"simd_count":32,
      "num_shader_engines":1,"num_shader_arrays_per_engine":1,"num_cu_per_sh":4,
      "num_sdma_engines":0,"num_sdma_xgmi_engines":2,
      "local_mem_size":1048576}}},
    "topology":{"root":{"name":"soc","type":"soc","children":[
      {"name":"vram","type":"gpu_memory"},
      {"name":"xcd[0:2]","type":"xcd","children":[
        {"name":"l2","type":"l2_cache"},
        {"name":"cp","type":"command_processor"},
        {"name":"se0","type":"shader_engine","children":[
          {"name":"cu[0:4]","type":"compute_unit"}]}]}]}})" +
         std::string(system_fields) + "}";
}

constexpr std::string_view kManifest = R"(,
  "system":{"nodes":[{"id":"tray7","gpus":[
    {"id":"g5","gpu_id":505,"unique_id":3005,
     "location_id":1024,"drm_render_minor":129},
    {"id":"g9","gpu_id":909,"unique_id":3009,
     "location_id":1280,"drm_render_minor":130}]}]})";

constexpr std::string_view kTwoNodes = R"(,
  "system":{"nodes":[
    {"id":"tray7","gpus":[{"id":"g5","gpu_id":505,"unique_id":3005,
      "location_id":1024,"drm_render_minor":129}]},
    {"id":"tray9","gpus":[{"id":"g9","gpu_id":909,"unique_id":3009,
      "location_id":1280,"drm_render_minor":130}]}]})";

void expect_same_info(const rj_vm_gpu_info_t &actual, const rj_vm_gpu_info_t &expected) {
  EXPECT_EQ(actual.present, expected.present);
  EXPECT_EQ(actual.gpu_id, expected.gpu_id);
  EXPECT_EQ(actual.gfx_target_version, expected.gfx_target_version);
  EXPECT_EQ(actual.vendor_id, expected.vendor_id);
  EXPECT_EQ(actual.device_id, expected.device_id);
  EXPECT_EQ(actual.family_id, expected.family_id);
  EXPECT_EQ(actual.unique_id, expected.unique_id);
  EXPECT_EQ(actual.location_id, expected.location_id);
  EXPECT_EQ(actual.domain, expected.domain);
  EXPECT_EQ(actual.hive_id, expected.hive_id);
  EXPECT_EQ(actual.drm_render_minor, expected.drm_render_minor);
  EXPECT_EQ(actual.revision_id, expected.revision_id);
  EXPECT_EQ(actual.pci_revision_id, expected.pci_revision_id);
  EXPECT_EQ(actual.simd_count, expected.simd_count);
  EXPECT_EQ(actual.max_waves_per_simd, expected.max_waves_per_simd);
  EXPECT_EQ(actual.num_shader_engines, expected.num_shader_engines);
  EXPECT_EQ(actual.num_shader_arrays_per_engine, expected.num_shader_arrays_per_engine);
  EXPECT_EQ(actual.num_cu_per_sh, expected.num_cu_per_sh);
  EXPECT_EQ(actual.simd_per_cu, expected.simd_per_cu);
  EXPECT_EQ(actual.wave_front_size, expected.wave_front_size);
  EXPECT_EQ(actual.num_xcc, expected.num_xcc);
  EXPECT_EQ(actual.max_slots_scratch_cu, expected.max_slots_scratch_cu);
  EXPECT_EQ(actual.local_mem_size, expected.local_mem_size);
  EXPECT_EQ(actual.visible_vram_size, expected.visible_vram_size);
  EXPECT_EQ(actual.control_stack_size, expected.control_stack_size);
  EXPECT_EQ(actual.context_save_size, expected.context_save_size);
  EXPECT_EQ(actual.vram_type, expected.vram_type);
  EXPECT_EQ(actual.lds_size_kb, expected.lds_size_kb);
  EXPECT_EQ(actual.mem_width, expected.mem_width);
  EXPECT_EQ(actual.mem_clk_max, expected.mem_clk_max);
  EXPECT_EQ(actual.l1_size_kb, expected.l1_size_kb);
  EXPECT_EQ(actual.l1_line_size, expected.l1_line_size);
  EXPECT_EQ(actual.l1_assoc, expected.l1_assoc);
  EXPECT_EQ(actual.l2_size_kb, expected.l2_size_kb);
  EXPECT_EQ(actual.l2_line_size, expected.l2_line_size);
  EXPECT_EQ(actual.l2_assoc, expected.l2_assoc);
  EXPECT_EQ(actual.num_sdma_engines, expected.num_sdma_engines);
  EXPECT_EQ(actual.num_sdma_xgmi_engines, expected.num_sdma_xgmi_engines);
  EXPECT_EQ(actual.num_cp_queues, expected.num_cp_queues);
  EXPECT_EQ(actual.max_engine_clk_fcompute, expected.max_engine_clk_fcompute);
  EXPECT_EQ(actual.capability, expected.capability);
  EXPECT_EQ(actual.capability2, expected.capability2);
  EXPECT_EQ(actual.debug_prop, expected.debug_prop);
  EXPECT_EQ(actual.fw_version, expected.fw_version);
  EXPECT_EQ(actual.sdma_fw_version, expected.sdma_fw_version);
  EXPECT_STREQ(actual.marketing_name, expected.marketing_name);
}

std::set<std::string> directory_entries(const std::filesystem::path &path) {
  std::set<std::string> entries;
  for (const auto &entry : std::filesystem::directory_iterator(path))
    entries.insert(entry.path().filename().string());
  return entries;
}

TEST(VmConfigApi, CanonicalGpuProfilesAreDiscoverableWithoutActivation) {
  rocjitsu::test::ScopedTempDirectory runtime;
  ASSERT_TRUE(runtime.initialize("rocjitsu-passive-config-"));
  const rocjitsu::test::ScopedEnvironmentVariable runtime_dir("ROCJITSU_RUNTIME_DIR",
                                                              runtime.path());
  const auto tasks = directory_entries("/proc/self/task");
  const auto descriptors = directory_entries("/proc/self/fd");
  for (const char *name :
       {"gfx1100_w7900.json", "gfx1151.json", "gfx1201_r9700.json", "gfx1250_mi455x.json",
        "gfx1250_mi455x_kmd_2node_net.json", "gfx1250_mi455x_kmd_4gpu.json",
        "gfx1251_synthetic.json", "gfx90a_mi210_kmd.json", "gfx942_cdna3.json",
        "gfx942_cdna3_kmd.json", "gfx950_mi350x.json", "gfx950_mi350x_kmd.json",
        "gfx950_mi350x_kmd_2gpu.json", "gfx950_mi355x.json", "gfx950_mi355x_kmd.json",
        "gfx950_mi355x_kmd_2gpu.json"}) {
    SCOPED_TRACE(name);
    rj_vm_config_t *raw = nullptr;
    {
      std::ifstream input(rocjitsu::test::config_path(name));
      ASSERT_TRUE(input.is_open());
      const std::string json{std::istreambuf_iterator<char>(input), {}};
      ASSERT_EQ(rj_vm_config_create_from_string_for_node(json.c_str(), "node0", &raw),
                ROCJITSU_STATUS_SUCCESS);
    }
    Config config(raw, rj_vm_config_destroy);
    uint32_t count = 0;
    ASSERT_EQ(rj_vm_config_gpu_count(config.get(), &count), ROCJITSU_STATUS_SUCCESS);
    ASSERT_GT(count, 0u);
    for (uint32_t ordinal = 0; ordinal < count; ++ordinal) {
      SCOPED_TRACE(ordinal);
      rj_vm_gpu_info_t info{};
      ASSERT_EQ(rj_vm_config_gpu_info_at(config.get(), ordinal, &info), ROCJITSU_STATUS_SUCCESS);
      EXPECT_EQ(info.present, 1u);
      EXPECT_NE(info.gfx_target_version, 0u);
      EXPECT_NE(info.num_xcc, 0u);
    }
    EXPECT_EQ(directory_entries("/proc/self/task"), tasks);
    EXPECT_EQ(directory_entries("/proc/self/fd"), descriptors);
    EXPECT_TRUE(std::filesystem::is_empty(runtime.path()));
  }
}

TEST(VmConfigApi, MetadataMatchesIndependentLocalAndDaemonVms) {
  for (const auto mode : {RJ_VM_MODE_LOCAL, RJ_VM_MODE_DAEMON}) {
    SCOPED_TRACE(mode);
    for (const bool manifest : {false, true}) {
      SCOPED_TRACE(manifest);
      rj_vm_config_t *raw = nullptr;
      {
        const std::string json = small_config(!manifest, manifest ? kManifest : "");
        const std::string node = manifest ? "tray7" : "node0";
        ASSERT_EQ(rj_vm_config_create_from_string_for_node(json.c_str(), node.c_str(), &raw),
                  ROCJITSU_STATUS_SUCCESS);
      }
      Config config(raw, rj_vm_config_destroy);
      uint32_t count = 0;
      ASSERT_EQ(rj_vm_config_gpu_count(config.get(), &count), ROCJITSU_STATUS_SUCCESS);
      ASSERT_EQ(count, 2u);
      std::array<rj_vm_gpu_info_t, 2> expected{};
      for (uint32_t ordinal = 0; ordinal < count; ++ordinal) {
        auto &info = expected[ordinal];
        ASSERT_EQ(rj_vm_config_gpu_info_at(config.get(), ordinal, &info), ROCJITSU_STATUS_SUCCESS);
        EXPECT_EQ(info.gpu_id, manifest ? (ordinal == 0 ? 505u : 909u) : 1250u + ordinal);
        EXPECT_EQ(info.drm_render_minor, (manifest ? 129u : 128u) + ordinal);
        EXPECT_EQ(info.num_xcc, 2u);
        EXPECT_EQ(info.visible_vram_size, info.local_mem_size);
        EXPECT_NE(info.control_stack_size, 0u);
        EXPECT_NE(info.context_save_size, 0u);
        EXPECT_EQ(info.hive_id, manifest ? 0u : 42u);
        EXPECT_EQ(info.num_sdma_xgmi_engines, manifest ? 0u : 2u);
      }
      // The same immutable snapshot supports either destruction order.
      for (const bool destroy_config_first : {false, true}) {
        SCOPED_TRACE(destroy_config_first);
        rj_vm_t *raw_vm = nullptr;
        ASSERT_EQ(rj_vm_create_from_config(config.get(), mode, &raw_vm), ROCJITSU_STATUS_SUCCESS);
        Vm vm(raw_vm, rj_vm_destroy);
        if (destroy_config_first)
          config.reset();
        uint32_t active_count = 0;
        ASSERT_EQ(rj_vm_gpu_count(vm.get(), &active_count), ROCJITSU_STATUS_SUCCESS);
        ASSERT_EQ(active_count, count);
        for (uint32_t ordinal = 0; ordinal < count; ++ordinal) {
          rj_vm_gpu_info_t actual{};
          ASSERT_EQ(rj_vm_gpu_info_at(vm.get(), ordinal, &actual), ROCJITSU_STATUS_SUCCESS);
          expect_same_info(actual, expected[ordinal]);
        }
        vm.reset();
        if (config) {
          rj_vm_gpu_info_t actual{};
          ASSERT_EQ(rj_vm_config_gpu_info_at(config.get(), 1, &actual), ROCJITSU_STATUS_SUCCESS);
          expect_same_info(actual, expected[1]);
        }
      }
    }
  }
}

TEST(VmConfigApi, SelectsNodesWithTheVmCreationRules) {
  const std::string ambiguous = small_config(false, kTwoNodes);
  const std::string configured =
      small_config(false, std::string(kTwoNodes) + R"(,"local_node":"tray9")");
  const std::string sole = small_config(false, kManifest);
  const std::string legacy = small_config(true);
  struct Case {
    const std::string &json; ///< Input configuration.
    const char *node;        ///< Optional explicit selector.
    uint32_t gpu_id;         ///< First selected GPU, or zero for a rejected selection.
  };
  for (const auto &test :
       {Case{ambiguous, nullptr, 0}, Case{ambiguous, "", 0}, Case{ambiguous, "tray7", 505},
        Case{ambiguous, "tray9", 909}, Case{ambiguous, "missing", 0},
        Case{configured, nullptr, 909}, Case{configured, "", 909}, Case{configured, "tray9", 909},
        Case{configured, "tray7", 0}, Case{sole, nullptr, 505}, Case{sole, "", 505},
        Case{legacy, nullptr, 1250}, Case{legacy, "node0", 1250}, Case{legacy, "missing", 0}}) {
    SCOPED_TRACE(test.node ? test.node : "<null>");
    rj_vm_config_t *raw = nullptr;
    const auto status =
        rj_vm_config_create_from_string_for_node(test.json.c_str(), test.node, &raw);
    Config config(raw, rj_vm_config_destroy);
    if (test.gpu_id == 0) {
      EXPECT_EQ(status, ROCJITSU_STATUS_INVALID_ARGUMENT);
      EXPECT_EQ(raw, nullptr);
    } else {
      ASSERT_EQ(status, ROCJITSU_STATUS_SUCCESS);
      rj_vm_gpu_info_t info{};
      ASSERT_EQ(rj_vm_config_gpu_info_at(config.get(), 0, &info), ROCJITSU_STATUS_SUCCESS);
      EXPECT_EQ(info.gpu_id, test.gpu_id);
    }
  }
}

TEST(VmConfigApi, FailedConstructionPreservesThePassiveSnapshot) {
  std::string json = small_config(true);
  const auto position = json.find("l2_cache");
  ASSERT_NE(position, std::string::npos);
  json.replace(position, std::strlen("l2_cache"), "unsupported_execution_component");
  rj_vm_config_t *raw = nullptr;
  ASSERT_EQ(rj_vm_config_create_from_string(json.c_str(), &raw), ROCJITSU_STATUS_SUCCESS);
  Config config(raw, rj_vm_config_destroy);
  rj_vm_gpu_info_t before{};
  ASSERT_EQ(rj_vm_config_gpu_info_at(config.get(), 1, &before), ROCJITSU_STATUS_SUCCESS);
  auto *vm = reinterpret_cast<rj_vm_t *>(uintptr_t{1});
  EXPECT_EQ(rj_vm_create_from_config(config.get(), RJ_VM_MODE_LOCAL, &vm),
            ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(vm, reinterpret_cast<rj_vm_t *>(uintptr_t{1}));
  rj_vm_gpu_info_t after{};
  ASSERT_EQ(rj_vm_config_gpu_info_at(config.get(), 1, &after), ROCJITSU_STATUS_SUCCESS);
  expect_same_info(after, before);
}

TEST(VmConfigApi, InvalidArgumentsLeaveOutputsUntouched) {
  const std::string json = small_config(true);
  auto *sentinel = reinterpret_cast<rj_vm_config_t *>(uintptr_t{1});
  EXPECT_EQ(rj_vm_config_create_from_string(nullptr, &sentinel), ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(rj_vm_config_create_from_string(json.c_str(), nullptr),
            ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(rj_vm_config_create_from_string("{", &sentinel), ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(rj_vm_config_create_from_string("{}", &sentinel), ROCJITSU_STATUS_UNSUPPORTED);
  EXPECT_EQ(sentinel, reinterpret_cast<rj_vm_config_t *>(uintptr_t{1}));

  rj_vm_config_t *raw = nullptr;
  ASSERT_EQ(rj_vm_config_create_from_string(json.c_str(), &raw), ROCJITSU_STATUS_SUCCESS);
  Config config(raw, rj_vm_config_destroy);
  uint32_t count = 99;
  EXPECT_EQ(rj_vm_config_gpu_count(nullptr, &count), ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(count, 99u);
  EXPECT_EQ(rj_vm_config_gpu_count(config.get(), nullptr), ROCJITSU_STATUS_INVALID_ARGUMENT);
  rj_vm_gpu_info_t info;
  std::memset(&info, 0xa5, sizeof(info));
  std::array<std::byte, sizeof(info)> bytes;
  std::memcpy(bytes.data(), &info, sizeof(info));
  EXPECT_EQ(rj_vm_config_gpu_info_at(nullptr, 0, &info), ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(rj_vm_config_gpu_info_at(config.get(), 0, nullptr), ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(rj_vm_config_gpu_info_at(config.get(), 2, &info), ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(rj_vm_config_gpu_info_at(config.get(), UINT32_MAX, &info),
            ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(std::memcmp(&info, bytes.data(), sizeof(info)), 0);
  auto *vm = reinterpret_cast<rj_vm_t *>(uintptr_t{1});
  EXPECT_EQ(rj_vm_create_from_config(nullptr, RJ_VM_MODE_LOCAL, &vm),
            ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(vm, reinterpret_cast<rj_vm_t *>(uintptr_t{1}));
  EXPECT_EQ(rj_vm_create_from_config(config.get(), RJ_VM_MODE_LOCAL, nullptr),
            ROCJITSU_STATUS_INVALID_ARGUMENT);
  rj_vm_config_destroy(nullptr);
}

} // namespace
