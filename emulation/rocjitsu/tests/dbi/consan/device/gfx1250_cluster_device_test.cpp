// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Host side of the gfx1250 clustered device tests. Unlike the portable HIP
/// fixtures, these tests submit hsa_amd_ext_kernel_dispatch_packet_t directly
/// so RocJITsu observes the real cluster dimensions and command-processor ABI.

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#ifndef RJ_CONSAN_GFX1250_CLUSTER_HSACO
#error "RJ_CONSAN_GFX1250_CLUSTER_HSACO must name the checked-in kernel fixture"
#endif

namespace {

constexpr uint32_t kWorkgroupSize = 64;
constexpr uint32_t kValuesPerGroup = 32;

enum class Workload {
  ClusterSync,
  MultiCluster,
  TdmWait,
  ClusterMulticast,
};

struct WorkloadConfig {
  const char *kernel_stem;
  uint32_t cluster_count;
  uint8_t cluster_size;
  uint32_t marker_base;
};

WorkloadConfig workload_config(Workload workload) {
  switch (workload) {
  case Workload::ClusterSync:
    return {"consan_gfx1250_cluster_sync", 1, 2, 0x51000000u};
  case Workload::MultiCluster:
    return {"consan_gfx1250_multi_cluster", 2, 2, 0x52000000u};
  case Workload::TdmWait:
    return {"consan_gfx1250_tdm_wait", 1, 2, 0x5d000000u};
  case Workload::ClusterMulticast:
    return {"consan_gfx1250_cluster_multicast", 1, 2, 0x5c000000u};
  }
  return {};
}

struct AgentSearch {
  hsa_agent_t gpu{};
  hsa_agent_t cpu{};
  std::vector<std::string> gpu_isas;
};

hsa_status_t find_agents(hsa_agent_t agent, void *data) {
  auto &search = *static_cast<AgentSearch *>(data);
  hsa_device_type_t type{};
  if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type) != HSA_STATUS_SUCCESS)
    return HSA_STATUS_SUCCESS;
  if (type == HSA_DEVICE_TYPE_CPU && search.cpu.handle == 0)
    search.cpu = agent;
  if (type != HSA_DEVICE_TYPE_GPU)
    return HSA_STATUS_SUCCESS;
  hsa_isa_t isa{};
  char name[128]{};
  if (hsa_agent_get_info(agent, HSA_AGENT_INFO_ISA, &isa) == HSA_STATUS_SUCCESS &&
      hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, name) == HSA_STATUS_SUCCESS) {
    search.gpu_isas.emplace_back(name);
    if (std::strstr(name, "gfx1250") != nullptr)
      search.gpu = agent;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_amd_memory_pool_t find_pool(hsa_agent_t agent, hsa_amd_segment_t segment,
                                bool host_accessible = false) {
  struct Context {
    hsa_amd_segment_t segment;
    bool host_accessible;
    hsa_amd_memory_pool_t result{};
  } context{segment, host_accessible};
  hsa_amd_agent_iterate_memory_pools(
      agent,
      [](hsa_amd_memory_pool_t pool, void *opaque) -> hsa_status_t {
        auto &ctx = *static_cast<Context *>(opaque);
        hsa_amd_segment_t found{};
        if (hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &found) !=
                HSA_STATUS_SUCCESS ||
            found != ctx.segment)
          return HSA_STATUS_SUCCESS;
        if (ctx.host_accessible) {
          bool accessible = false;
          if (hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL,
                                           &accessible) != HSA_STATUS_SUCCESS ||
              !accessible)
            return HSA_STATUS_SUCCESS;
        }
        ctx.result = pool;
        return HSA_STATUS_INFO_BREAK;
      },
      &context);
  return context.result;
}

std::vector<uint8_t> read_fixture() {
  std::ifstream file(RJ_CONSAN_GFX1250_CLUSTER_HSACO, std::ios::binary | std::ios::ate);
  EXPECT_TRUE(file.good()) << RJ_CONSAN_GFX1250_CLUSTER_HSACO;
  if (!file)
    return {};
  const std::streamsize size = file.tellg();
  EXPECT_GT(size, 0);
  if (size <= 0)
    return {};
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  file.seekg(0);
  file.read(reinterpret_cast<char *>(bytes.data()), size);
  EXPECT_TRUE(file.good());
  return bytes;
}

class ClusterResources {
public:
  ~ClusterResources() {
    if (signal_.handle != 0)
      hsa_signal_destroy(signal_);
    if (queue_ != nullptr)
      hsa_queue_destroy(queue_);
    for (void *allocation : allocations_)
      hsa_amd_memory_pool_free(allocation);
    if (executable_.handle != 0)
      hsa_executable_destroy(executable_);
    if (reader_.handle != 0)
      hsa_code_object_reader_destroy(reader_);
  }

  void remember(void *allocation) { allocations_.push_back(allocation); }

  hsa_code_object_reader_t reader_{};
  hsa_executable_t executable_{};
  hsa_queue_t *queue_ = nullptr;
  hsa_signal_t signal_{};

private:
  std::vector<void *> allocations_;
};

void run_cluster_workload(Workload workload, bool correct) {
  const WorkloadConfig config = workload_config(workload);
  AgentSearch agents;
  ASSERT_EQ(hsa_iterate_agents(find_agents, &agents), HSA_STATUS_SUCCESS);
  ASSERT_NE(agents.cpu.handle, 0u);
  ASSERT_NE(agents.gpu.handle, 0u) << "No gfx1250 guest agent was exposed";

  const std::vector<uint8_t> image = read_fixture();
  ASSERT_FALSE(image.empty());
  ClusterResources resources;
  ASSERT_EQ(
      hsa_code_object_reader_create_from_memory(image.data(), image.size(), &resources.reader_),
      HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                      nullptr, &resources.executable_),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_executable_load_agent_code_object(resources.executable_, agents.gpu,
                                                  resources.reader_, nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_executable_freeze(resources.executable_, nullptr), HSA_STATUS_SUCCESS);

  const std::string kernel_name =
      std::string(config.kernel_stem) + (correct ? "_correct.kd" : "_incorrect.kd");
  hsa_executable_symbol_t symbol{};
  ASSERT_EQ(hsa_executable_get_symbol_by_name(resources.executable_, kernel_name.c_str(),
                                              &agents.gpu, &symbol),
            HSA_STATUS_SUCCESS);
  uint64_t kernel_object = 0;
  uint32_t private_bytes = 0;
  uint32_t group_bytes = 0;
  uint32_t kernarg_bytes = 0;
  ASSERT_EQ(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                           &kernel_object),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_executable_symbol_get_info(
                symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &private_bytes),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_executable_symbol_get_info(
                symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &group_bytes),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_executable_symbol_get_info(
                symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &kernarg_bytes),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(kernel_object, 0u);
  ASSERT_GE(kernarg_bytes, 90u);

  const uint32_t group_count = config.cluster_count * config.cluster_size;
  const size_t input_bytes = group_count * kValuesPerGroup * sizeof(uint32_t);
  const size_t observed_bytes = group_count * kValuesPerGroup * sizeof(uint32_t);
  const size_t control_bytes = group_count * kWorkgroupSize * sizeof(uint32_t);
  auto gpu_pool = find_pool(agents.gpu, HSA_AMD_SEGMENT_GLOBAL);
  auto kernarg_pool = find_pool(agents.cpu, HSA_AMD_SEGMENT_GLOBAL, true);
  ASSERT_NE(gpu_pool.handle, 0u);
  ASSERT_NE(kernarg_pool.handle, 0u);

  uint32_t *input = nullptr;
  uint32_t *observed = nullptr;
  uint32_t *control = nullptr;
  void *kernarg = nullptr;
  ASSERT_EQ(
      hsa_amd_memory_pool_allocate(gpu_pool, input_bytes, 0, reinterpret_cast<void **>(&input)),
      HSA_STATUS_SUCCESS);
  resources.remember(input);
  ASSERT_EQ(hsa_amd_memory_pool_allocate(gpu_pool, observed_bytes, 0,
                                         reinterpret_cast<void **>(&observed)),
            HSA_STATUS_SUCCESS);
  resources.remember(observed);
  ASSERT_EQ(
      hsa_amd_memory_pool_allocate(gpu_pool, control_bytes, 0, reinterpret_cast<void **>(&control)),
      HSA_STATUS_SUCCESS);
  resources.remember(control);
  ASSERT_EQ(hsa_amd_memory_pool_allocate(kernarg_pool, kernarg_bytes, 0, &kernarg),
            HSA_STATUS_SUCCESS);
  resources.remember(kernarg);

  const hsa_agent_t both[] = {agents.cpu, agents.gpu};
  ASSERT_EQ(hsa_amd_agents_allow_access(2, both, nullptr, input), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_amd_agents_allow_access(2, both, nullptr, observed), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_amd_agents_allow_access(2, both, nullptr, control), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_amd_agents_allow_access(2, both, nullptr, kernarg), HSA_STATUS_SUCCESS);
  std::vector<uint32_t> host_input(group_count * kValuesPerGroup);
  for (uint32_t group = 0; group < group_count; ++group) {
    for (uint32_t lane = 0; lane < kValuesPerGroup; ++lane)
      host_input[group * kValuesPerGroup + lane] = 0xa5000000u | (group << 8u) | lane;
  }
  const std::vector<uint32_t> zero_observed(group_count * kValuesPerGroup, 0);
  const std::vector<uint32_t> zero_control(group_count * kWorkgroupSize, 0);
  ASSERT_EQ(hsa_memory_copy(input, host_input.data(), input_bytes), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_memory_copy(observed, zero_observed.data(), observed_bytes), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_memory_copy(control, zero_control.data(), control_bytes), HSA_STATUS_SUCCESS);
  std::memset(kernarg, 0, kernarg_bytes);
  struct __attribute__((packed)) Kernargs {
    const uint32_t *input;    // 0: user argument
    uint32_t *observed;       // 8: user argument
    uint32_t *control;        // 16: user argument
    uint32_t block_count_x;   // 24: hidden_block_count_x
    uint32_t block_count_y;   // 28: hidden_block_count_y
    uint32_t block_count_z;   // 32: hidden_block_count_z
    uint16_t group_size_x;    // 36: hidden_group_size_x
    uint16_t group_size_y;    // 38: hidden_group_size_y
    uint16_t group_size_z;    // 40: hidden_group_size_z
    uint16_t remainder_x;     // 42: hidden_remainder_x
    uint16_t remainder_y;     // 44: hidden_remainder_y
    uint16_t remainder_z;     // 46: hidden_remainder_z
    uint8_t reserved0[16];    // 48
    uint64_t global_offset_x; // 64: hidden_global_offset_x
    uint64_t global_offset_y; // 72: hidden_global_offset_y
    uint64_t global_offset_z; // 80: hidden_global_offset_z
    uint16_t grid_dims;       // 88: hidden_grid_dims
  };
  static_assert(sizeof(Kernargs) == 90);
  auto &args = *static_cast<Kernargs *>(kernarg);
  args.input = input;
  args.observed = observed;
  args.control = control;
  args.block_count_x = group_count;
  args.block_count_y = 1;
  args.block_count_z = 1;
  args.group_size_x = kWorkgroupSize;
  args.group_size_y = 1;
  args.group_size_z = 1;
  args.grid_dims = 1;

  uint32_t queue_size = 0;
  ASSERT_EQ(hsa_agent_get_info(agents.gpu, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_queue_create(agents.gpu, queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr,
                             UINT32_MAX, UINT32_MAX, &resources.queue_),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(resources.queue_, nullptr);
  ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &resources.signal_), HSA_STATUS_SUCCESS);

  const uint64_t write_index = hsa_queue_add_write_index_relaxed(resources.queue_, 1);
  auto *packet =
      reinterpret_cast<hsa_amd_ext_kernel_dispatch_packet_t *>(resources.queue_->base_address) +
      (write_index & (resources.queue_->size - 1));
  std::memset(packet, 0, sizeof(*packet));
  packet->amd_format = HSA_AMD_PACKET_TYPE_EXT_KERNEL_DISPATCH;
  packet->setup = 1;
  packet->workgroup_size_x = kWorkgroupSize;
  packet->workgroup_size_y = 1;
  packet->workgroup_size_z = 1;
  packet->cluster_count_x = config.cluster_count;
  packet->cluster_count_y = 1;
  packet->cluster_count_z = 1;
  packet->cluster_size_x = config.cluster_size;
  packet->cluster_size_y = 1;
  packet->cluster_size_z = 1;
  packet->private_segment_size = private_bytes;
  packet->group_segment_size = group_bytes;
  packet->kernel_object = kernel_object;
  packet->kernarg_address = kernarg;
  packet->completion_signal = resources.signal_;

  uint16_t header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
  header |= 1u << HSA_PACKET_HEADER_BARRIER;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
  __atomic_store_n(&packet->header, header, __ATOMIC_RELEASE);
  hsa_signal_store_relaxed(resources.queue_->doorbell_signal, write_index);
  const hsa_signal_value_t completion = hsa_signal_wait_scacquire(
      resources.signal_, HSA_SIGNAL_CONDITION_LT, 1, 5'000'000'000ULL, HSA_WAIT_STATE_BLOCKED);
  ASSERT_EQ(completion, 0) << "cluster dispatch timed out or failed";

  std::vector<uint32_t> host_observed(group_count * kValuesPerGroup);
  std::vector<uint32_t> host_control(group_count * kWorkgroupSize);
  ASSERT_EQ(hsa_memory_copy(host_observed.data(), observed, observed_bytes), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_memory_copy(host_control.data(), control, control_bytes), HSA_STATUS_SUCCESS);
  for (uint32_t group = 0; group < group_count; ++group) {
    if (correct && workload == Workload::ClusterSync)
      EXPECT_EQ(host_observed[group], 0x51000000u | group) << "group=" << group;
    if (correct && workload == Workload::MultiCluster)
      EXPECT_EQ(host_observed[group], 0x52000000u | group) << "group=" << group;
    if (correct && workload == Workload::TdmWait) {
      for (uint32_t lane = 0; lane < kValuesPerGroup; ++lane)
        EXPECT_EQ(host_observed[group * kValuesPerGroup + lane],
                  host_input[group * kValuesPerGroup + lane])
            << "group=" << group << " lane=" << lane;
    }
    if (correct && workload == Workload::ClusterMulticast) {
      for (uint32_t lane = 0; lane < kValuesPerGroup; ++lane)
        EXPECT_EQ(host_observed[group * kValuesPerGroup + lane], host_input[lane])
            << "group=" << group << " lane=" << lane;
    }
    for (uint32_t tid = 0; tid < kWorkgroupSize; ++tid)
      EXPECT_EQ(host_control[group * kWorkgroupSize + tid],
                config.marker_base | (group << 8u) | tid)
          << "group=" << group << " tid=" << tid;
  }
}

TEST(ConSanDeviceGfx1250ClusterSyncTest, Correct) {
  run_cluster_workload(Workload::ClusterSync, true);
}
TEST(ConSanDeviceGfx1250ClusterSyncTest, Incorrect) {
  run_cluster_workload(Workload::ClusterSync, false);
}
TEST(ConSanDeviceGfx1250MultiClusterTest, Correct) {
  run_cluster_workload(Workload::MultiCluster, true);
}
TEST(ConSanDeviceGfx1250MultiClusterTest, Incorrect) {
  run_cluster_workload(Workload::MultiCluster, false);
}
TEST(ConSanDeviceGfx1250TdmWaitTest, Correct) { run_cluster_workload(Workload::TdmWait, true); }
TEST(ConSanDeviceGfx1250TdmWaitTest, Incorrect) { run_cluster_workload(Workload::TdmWait, false); }
TEST(ConSanDeviceGfx1250ClusterMulticastTest, Correct) {
  run_cluster_workload(Workload::ClusterMulticast, true);
}
TEST(ConSanDeviceGfx1250ClusterMulticastTest, Incorrect) {
  run_cluster_workload(Workload::ClusterMulticast, false);
}

} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  if (hsa_init() != HSA_STATUS_SUCCESS)
    return 1;
  const int result = RUN_ALL_TESTS();
  (void)hsa_shut_down();
  return result;
}
