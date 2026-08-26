// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "embedded_schema.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/vm/amdgpu/partitioning.h"
#include "rocjitsu/vm/rj_vm.h"
#include "rocjitsu/vm/rj_vm_impl.h"
#include "rocjitsu/vm/soc.h"

#include "simdojo/sim/component.h"
#include "simdojo/sim/simulation.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

namespace {

using namespace rocjitsu;

const std::string CONFIG_PATH = std::string(CONFIG_DIR) + "/gfx950_mi355x.json";
const std::string CONFIG_2GPU_PATH = std::string(CONFIG_DIR) + "/gfx950_mi355x_kmd_2gpu.json";
const std::string CONFIG_1XCD_PATH = std::string(CONFIG_DIR) + "/gfx1100_w7900.json";

std::string config_json_with_num_threads(const std::string &path, uint32_t num_threads) {
  std::ifstream input(path);
  if (!input.is_open())
    throw std::runtime_error("Failed to open config: " + path);

  std::string json((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  const std::string configured_threads = "\"num_threads\": 1";
  const auto num_threads_pos = json.find(configured_threads);
  if (num_threads_pos == std::string::npos)
    throw std::runtime_error("Config does not have the expected num_threads field: " + path);

  std::string replacement = "\"num_threads\": " + std::to_string(num_threads);
  json.replace(num_threads_pos, configured_threads.size(), replacement);
  return json;
}

struct PartitionedTopology {
  config::LoadedConfig loaded;
  SoC *soc = nullptr;
  simdojo::Component *memory = nullptr;
  std::unique_ptr<simdojo::SimulationEngine> engine;
  bool partitioned = false;
};

PartitionedTopology build_partitioned_topology(uint32_t num_threads) {
  auto loaded = config::load_config(CONFIG_PATH, rocjitsu::kEmbeddedSchema);
  auto *soc = loaded.soc();
  auto *memory = loaded.memory();
  loaded.engine_config.num_threads = num_threads;

  auto engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
  engine->topology().set_root(loaded.take_root());
  loaded.wire_links(engine->topology());
  bool partitioned = amdgpu::partition_topology_by_xcds(engine->topology(), soc, num_threads);
  engine->create();

  return {std::move(loaded), soc, memory, std::move(engine), partitioned};
}

void expect_subtree_partition(simdojo::Component *component, simdojo::PartitionID expected) {
  ASSERT_NE(component, nullptr);
  EXPECT_EQ(component->partition_id(), expected) << component->full_path();

  auto *composite = dynamic_cast<simdojo::CompositeComponent *>(component);
  if (!composite)
    return;

  for (const auto &child : composite->children())
    expect_subtree_partition(child.get(), expected);
}

TEST(XcdPartitioningTest, EightThreadsMapsEachCdna4XcdToItsOwnPartition) {
  auto topology = build_partitioned_topology(8);

  ASSERT_TRUE(topology.partitioned);
  ASSERT_EQ(topology.engine->topology().partitions().size(), 8u);
  ASSERT_EQ(topology.soc->num_xcds(), 8u);

  for (uint32_t i = 0; i < topology.soc->num_xcds(); ++i)
    expect_subtree_partition(topology.soc->xcd(i), i);
}

TEST(XcdPartitioningTest, FourThreadsDistributesCdna4XcdsRoundRobinWithoutSplits) {
  auto topology = build_partitioned_topology(4);

  ASSERT_TRUE(topology.partitioned);
  ASSERT_EQ(topology.engine->topology().partitions().size(), 4u);
  ASSERT_EQ(topology.soc->num_xcds(), 8u);

  for (uint32_t i = 0; i < topology.soc->num_xcds(); ++i)
    expect_subtree_partition(topology.soc->xcd(i), i % 4);
}

TEST(XcdPartitioningTest, ClampPartitionCountUsesAllNonNullSocs) {
  auto loaded = config::load_config(CONFIG_2GPU_PATH, rocjitsu::kEmbeddedSchema);
  ASSERT_EQ(loaded.extra_gpu_builds.size(), 1u);

  auto *soc0 = loaded.soc();
  auto *soc1 = dynamic_cast<SoC *>(loaded.extra_gpu_builds[0].root.get());
  ASSERT_NE(soc0, nullptr);
  ASSERT_NE(soc1, nullptr);
  ASSERT_EQ(soc0->num_xcds(), 8u);
  ASSERT_EQ(soc1->num_xcds(), 8u);

  std::array<SoC *, 3> socs = {soc0, nullptr, soc1};
  EXPECT_EQ(amdgpu::clamp_xcd_partition_count(std::span<SoC *>(socs), 0), 1u);
  EXPECT_EQ(amdgpu::clamp_xcd_partition_count(std::span<SoC *>(socs), 4), 4u);
  EXPECT_EQ(amdgpu::clamp_xcd_partition_count(std::span<SoC *>(socs), 64), 16u);
  EXPECT_EQ(amdgpu::clamp_xcd_partition_count(soc0, 64), 8u);
  EXPECT_EQ(amdgpu::clamp_xcd_partition_count(static_cast<SoC *>(nullptr), 64), 1u);
}

TEST(XcdPartitioningTest, ZeroPartitionsIsNoopWithoutManualPartitions) {
  auto loaded = config::load_config(CONFIG_PATH, rocjitsu::kEmbeddedSchema);
  auto *soc = loaded.soc();
  simdojo::Topology topology;
  topology.set_root(loaded.take_root());
  loaded.wire_links(topology);

  EXPECT_FALSE(amdgpu::partition_topology_by_xcds(topology, soc, 0));
  EXPECT_TRUE(topology.partitions().empty());
  EXPECT_EQ(soc->partition_id(), simdojo::INVALID_PARTITION_ID);
  for (uint32_t i = 0; i < soc->num_xcds(); ++i)
    EXPECT_EQ(soc->xcd(i)->partition_id(), simdojo::INVALID_PARTITION_ID);
}

TEST(XcdPartitioningTest, NoXcdsIsNoopWithoutManualPartitions) {
  simdojo::Topology topology;

  EXPECT_FALSE(amdgpu::partition_topology_by_xcds(topology, std::span<SoC *>{}, 1));
  EXPECT_FALSE(amdgpu::partition_topology_by_xcds(topology, static_cast<SoC *>(nullptr), 1));
  EXPECT_TRUE(topology.partitions().empty());
}

TEST(XcdPartitioningTest, SinglePartitionAssignsAllComponentsToZero) {
  auto loaded = config::load_config(CONFIG_PATH, rocjitsu::kEmbeddedSchema);
  auto *soc = loaded.soc();
  simdojo::Topology topology;
  topology.set_root(loaded.take_root());
  loaded.wire_links(topology);

  ASSERT_TRUE(amdgpu::partition_topology_by_xcds(topology, soc, 1));
  ASSERT_EQ(topology.partitions().size(), 1u);
  for (auto *component : topology.collect_all_components())
    EXPECT_EQ(component->partition_id(), 0u) << component->full_path();
}

TEST(XcdPartitioningTest, ThreeThreadsDistributesCdna4XcdsRoundRobinWithoutSplits) {
  auto topology = build_partitioned_topology(3);

  ASSERT_TRUE(topology.partitioned);
  ASSERT_EQ(topology.engine->topology().partitions().size(), 3u);
  ASSERT_EQ(topology.soc->num_xcds(), 8u);

  for (uint32_t i = 0; i < topology.soc->num_xcds(); ++i)
    expect_subtree_partition(topology.soc->xcd(i), i % 3);
}

TEST(XcdPartitioningTest, SpanOverMultipleSocsUsesGlobalXcdIndex) {
  auto loaded = config::load_config(CONFIG_2GPU_PATH, rocjitsu::kEmbeddedSchema);
  ASSERT_EQ(loaded.extra_gpu_builds.size(), 1u);

  auto root = std::make_unique<simdojo::CompositeComponent>("system");
  auto *soc0 = dynamic_cast<SoC *>(root->add_child(loaded.take_root()));
  auto *soc1 = dynamic_cast<SoC *>(root->add_child(std::move(loaded.extra_gpu_builds[0].root)));
  ASSERT_NE(soc0, nullptr);
  ASSERT_NE(soc1, nullptr);

  simdojo::Topology topology;
  topology.set_root(std::move(root));
  std::array<SoC *, 2> socs = {soc0, soc1};
  ASSERT_TRUE(amdgpu::partition_topology_by_xcds(topology, std::span<SoC *>(socs), 3));
  ASSERT_EQ(topology.partitions().size(), 3u);

  uint32_t global_xcd_index = 0;
  for (auto *soc : socs) {
    ASSERT_EQ(soc->num_xcds(), 8u);
    for (uint32_t i = 0; i < soc->num_xcds(); ++i, ++global_xcd_index)
      expect_subtree_partition(soc->xcd(i), global_xcd_index % 3);
  }
}

TEST(XcdPartitioningTest, RejectsSocOutsideTopologyWithoutChangingPartitionState) {
  auto loaded = config::load_config(CONFIG_2GPU_PATH, rocjitsu::kEmbeddedSchema);
  ASSERT_EQ(loaded.extra_gpu_builds.size(), 1u);

  auto *soc0 = loaded.soc();
  auto *soc1 = dynamic_cast<SoC *>(loaded.extra_gpu_builds[0].root.get());
  ASSERT_NE(soc0, nullptr);
  ASSERT_NE(soc1, nullptr);

  simdojo::Topology topology;
  topology.set_root(loaded.take_root());
  std::array<SoC *, 2> socs = {soc0, soc1};
  EXPECT_FALSE(amdgpu::partition_topology_by_xcds(topology, std::span<SoC *>(socs), 16));

  EXPECT_TRUE(topology.partitions().empty());
  for (SoC *soc : socs) {
    EXPECT_EQ(soc->partition_id(), simdojo::INVALID_PARTITION_ID);
    for (uint32_t i = 0; i < soc->num_xcds(); ++i)
      EXPECT_EQ(soc->xcd(i)->partition_id(), simdojo::INVALID_PARTITION_ID);
  }
}

TEST(XcdPartitioningTest, NonXcdComponentsStayOnPartitionZero) {
  auto topology = build_partitioned_topology(8);

  ASSERT_TRUE(topology.partitioned);
  EXPECT_EQ(topology.soc->partition_id(), 0u);
  EXPECT_EQ(topology.memory->partition_id(), 0u);

  for (uint32_t i = 0; i < topology.soc->num_iods(); ++i)
    expect_subtree_partition(topology.soc->iod(i), 0);
}

TEST(XcdPartitioningTest, VmStepRejectsMultipleEnginePartitions) {
  auto json = config_json_with_num_threads(CONFIG_PATH, 2);

  rj_vm_t *raw_vm = nullptr;
  ASSERT_EQ(rj_vm_create_from_string(json.c_str(), RJ_VM_MODE_DEFAULT, &raw_vm),
            ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(raw_vm, nullptr);
  std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)> vm(raw_vm, &rj_vm_destroy);

  int active = 7;
  EXPECT_EQ(rj_vm_step(vm.get(), &active), ROCJITSU_STATUS_UNSUPPORTED);
  EXPECT_EQ(active, 7);

  uint64_t ticks_executed = 0;
  EXPECT_EQ(rj_vm_run(vm.get(), &ticks_executed), ROCJITSU_STATUS_SUCCESS);
}

TEST(XcdPartitioningTest, VmStepSucceedsWithSingleEnginePartition) {
  auto json = config_json_with_num_threads(CONFIG_1XCD_PATH, 1);

  rj_vm_t *raw_vm = nullptr;
  ASSERT_EQ(rj_vm_create_from_string(json.c_str(), RJ_VM_MODE_DEFAULT, &raw_vm),
            ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(raw_vm, nullptr);
  std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)> vm(raw_vm, &rj_vm_destroy);

  int active = 7;
  EXPECT_EQ(rj_vm_step(vm.get(), &active), ROCJITSU_STATUS_SUCCESS);
  EXPECT_EQ(active, 0);
}

TEST(XcdPartitioningTest, CApiClampsZeroThreadsToOne) {
  auto json = config_json_with_num_threads(CONFIG_1XCD_PATH, 0);

  rj_vm_t *raw_vm = nullptr;
  ASSERT_EQ(rj_vm_create_from_string(json.c_str(), RJ_VM_MODE_DEFAULT, &raw_vm),
            ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(raw_vm, nullptr);
  std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)> vm(raw_vm, &rj_vm_destroy);

  EXPECT_EQ(vm->engine_config.num_threads, 1u);
  EXPECT_EQ(rj_vm_run(vm.get(), nullptr), ROCJITSU_STATUS_SUCCESS);
}

TEST(XcdPartitioningTest, CApiClampsSingleGpuThreadsToXcdCount) {
  auto json = config_json_with_num_threads(CONFIG_PATH, 64);

  rj_vm_t *raw_vm = nullptr;
  testing::internal::CaptureStderr();
  const rj_status_t status = rj_vm_create_from_string(json.c_str(), RJ_VM_MODE_DEFAULT, &raw_vm);
  const std::string warning = testing::internal::GetCapturedStderr();
  ASSERT_EQ(status, ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(raw_vm, nullptr);
  std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)> vm(raw_vm, &rj_vm_destroy);

  EXPECT_NE(warning.find("num_threads clamped: requested=64, effective=8"), std::string::npos);
  EXPECT_EQ(vm->engine_config.num_threads, 8u);
  EXPECT_EQ(rj_vm_run(vm.get(), nullptr), ROCJITSU_STATUS_SUCCESS);
}

TEST(XcdPartitioningTest, CApiDoesNotWarnWhenThreadCountIsUnchanged) {
  auto json = config_json_with_num_threads(CONFIG_1XCD_PATH, 1);

  rj_vm_t *raw_vm = nullptr;
  testing::internal::CaptureStderr();
  const rj_status_t status = rj_vm_create_from_string(json.c_str(), RJ_VM_MODE_DEFAULT, &raw_vm);
  const std::string warning = testing::internal::GetCapturedStderr();
  ASSERT_EQ(status, ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(raw_vm, nullptr);
  std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)> vm(raw_vm, &rj_vm_destroy);

  EXPECT_EQ(warning.find("num_threads clamped"), std::string::npos);
}

TEST(XcdPartitioningTest, CApiClampsMultiGpuThreadsToTotalXcdCountAndRuns) {
  auto json = config_json_with_num_threads(CONFIG_2GPU_PATH, 64);

  rj_vm_t *raw_vm = nullptr;
  ASSERT_EQ(rj_vm_create_from_string(json.c_str(), RJ_VM_MODE_DEFAULT, &raw_vm),
            ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(raw_vm, nullptr);
  std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)> vm(raw_vm, &rj_vm_destroy);

  EXPECT_EQ(vm->engine_config.num_threads, 16u);
  ASSERT_NE(vm->vm, nullptr);
  ASSERT_EQ(vm->vm->num_socs(), 2u);

  uint32_t global_xcd_index = 0;
  for (uint32_t gpu = 0; gpu < vm->vm->num_socs(); ++gpu) {
    auto *soc = vm->vm->soc(gpu);
    ASSERT_NE(soc, nullptr);
    ASSERT_EQ(soc->num_xcds(), 8u);
    for (uint32_t i = 0; i < soc->num_xcds(); ++i, ++global_xcd_index)
      expect_subtree_partition(soc->xcd(i), global_xcd_index % vm->engine_config.num_threads);
  }

  EXPECT_EQ(rj_vm_run(vm.get(), nullptr), ROCJITSU_STATUS_SUCCESS);
}

} // namespace
