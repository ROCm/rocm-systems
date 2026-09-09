// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Direct-HSA regression for queue-local AMDHSA dispatch IDs. Two explicitly
/// distinct queues submit matching queue-local packet IDs with the same kernel
/// object, first reusing and then separating kernarg allocations. A
/// synchronized kernel must not acquire a false cross-dispatch conflict; its
/// paired unsynchronized kernel remains racy.

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#ifndef RJ_CONSAN_GFX950_TWO_QUEUE_HSACO
#error "RJ_CONSAN_GFX950_TWO_QUEUE_HSACO must name the two-queue kernel fixture"
#endif

namespace {

constexpr uint32_t kMaximumThreads = 128u;
constexpr uint32_t kQueueCount = 2u;
constexpr uint32_t kRounds = 2u;
constexpr uint32_t kDispatches = kQueueCount * kRounds;

struct TwoQueueState {
  uint32_t launches;
  uint32_t observed[kDispatches];
  uint32_t control[kDispatches][kMaximumThreads];
};

struct AgentSearch {
  hsa_agent_t gpu{};
  hsa_agent_t cpu{};
};

hsa_status_t find_agents(hsa_agent_t agent, void *data) {
  auto &search = *static_cast<AgentSearch *>(data);
  hsa_device_type_t type{};
  if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type) != HSA_STATUS_SUCCESS)
    return HSA_STATUS_SUCCESS;
  if (type == HSA_DEVICE_TYPE_CPU && search.cpu.handle == 0u)
    search.cpu = agent;
  if (type != HSA_DEVICE_TYPE_GPU)
    return HSA_STATUS_SUCCESS;
  hsa_isa_t isa{};
  char name[128]{};
  if (hsa_agent_get_info(agent, HSA_AGENT_INFO_ISA, &isa) == HSA_STATUS_SUCCESS &&
      hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, name) == HSA_STATUS_SUCCESS &&
      std::strstr(name, "gfx950") != nullptr) {
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
  std::ifstream file(RJ_CONSAN_GFX950_TWO_QUEUE_HSACO, std::ios::binary | std::ios::ate);
  EXPECT_TRUE(file.good()) << RJ_CONSAN_GFX950_TWO_QUEUE_HSACO;
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

class Resources {
public:
  ~Resources() {
    for (hsa_signal_t signal : signals_) {
      if (signal.handle != 0u)
        (void)hsa_signal_destroy(signal);
    }
    for (hsa_queue_t *queue : queues_) {
      if (queue != nullptr)
        (void)hsa_queue_destroy(queue);
    }
    for (void *allocation : allocations_) {
      if (allocation != nullptr)
        (void)hsa_amd_memory_pool_free(allocation);
    }
    if (executable_.handle != 0u)
      (void)hsa_executable_destroy(executable_);
    if (reader_.handle != 0u)
      (void)hsa_code_object_reader_destroy(reader_);
  }

  void remember(void *allocation) { allocations_.push_back(allocation); }

  hsa_code_object_reader_t reader_{};
  hsa_executable_t executable_{};
  std::array<hsa_queue_t *, kQueueCount> queues_{};
  std::array<hsa_signal_t, kQueueCount> signals_{};

private:
  std::vector<void *> allocations_;
};

void run_two_queue_dispatch(bool correct, bool reuse_kernarg) {
  AgentSearch agents;
  ASSERT_EQ(hsa_iterate_agents(find_agents, &agents), HSA_STATUS_SUCCESS);
  ASSERT_NE(agents.cpu.handle, 0u);
  ASSERT_NE(agents.gpu.handle, 0u) << "No gfx950 agent was exposed";

  const std::vector<uint8_t> image = read_fixture();
  ASSERT_FALSE(image.empty());
  Resources resources;
  ASSERT_EQ(hsa_code_object_reader_create_from_memory(image.data(), image.size(), &resources.reader_),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                      nullptr, &resources.executable_),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_executable_load_agent_code_object(resources.executable_, agents.gpu,
                                                  resources.reader_, nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_executable_freeze(resources.executable_, nullptr), HSA_STATUS_SUCCESS);

  const char *kernel_name = correct ? "consan_gfx950_two_queue_dispatch_correct.kd"
                                    : "consan_gfx950_two_queue_dispatch_incorrect.kd";
  hsa_executable_symbol_t symbol{};
  ASSERT_EQ(hsa_executable_get_symbol_by_name(resources.executable_, kernel_name, &agents.gpu,
                                              &symbol),
            HSA_STATUS_SUCCESS);
  uint64_t kernel_object = 0u;
  uint32_t private_bytes = 0u;
  uint32_t group_bytes = 0u;
  uint32_t kernarg_bytes = 0u;
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
  ASSERT_GE(kernarg_bytes, sizeof(TwoQueueState *));

  const hsa_amd_memory_pool_t gpu_pool = find_pool(agents.gpu, HSA_AMD_SEGMENT_GLOBAL);
  const hsa_amd_memory_pool_t kernarg_pool =
      find_pool(agents.cpu, HSA_AMD_SEGMENT_GLOBAL, true);
  ASSERT_NE(gpu_pool.handle, 0u);
  ASSERT_NE(kernarg_pool.handle, 0u);

  TwoQueueState *state = nullptr;
  std::array<void *, kQueueCount> kernargs{};
  void *state_staging = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(gpu_pool, sizeof(TwoQueueState), 0,
                                         reinterpret_cast<void **>(&state)),
            HSA_STATUS_SUCCESS);
  resources.remember(state);
  const uint32_t kernarg_allocation_count = reuse_kernarg ? 1u : kQueueCount;
  for (uint32_t i = 0u; i < kernarg_allocation_count; ++i) {
    ASSERT_EQ(hsa_amd_memory_pool_allocate(kernarg_pool, kernarg_bytes, 0, &kernargs[i]),
              HSA_STATUS_SUCCESS);
    resources.remember(kernargs[i]);
  }
  if (reuse_kernarg)
    kernargs[1] = kernargs[0];
  ASSERT_EQ(hsa_amd_memory_pool_allocate(kernarg_pool, sizeof(TwoQueueState), 0,
                                         &state_staging),
            HSA_STATUS_SUCCESS);
  resources.remember(state_staging);
  const hsa_agent_t both[] = {agents.cpu, agents.gpu};
  ASSERT_EQ(hsa_amd_agents_allow_access(2u, both, nullptr, state), HSA_STATUS_SUCCESS);
  for (uint32_t i = 0u; i < kernarg_allocation_count; ++i)
    ASSERT_EQ(hsa_amd_agents_allow_access(2u, both, nullptr, kernargs[i]), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_amd_agents_allow_access(2u, both, nullptr, state_staging), HSA_STATUS_SUCCESS);
  // Both HSA copy endpoints must be runtime allocations. Ordinary host stack
  // or heap pages are not portable HSA memory and are deliberately not mapped
  // by the gfx950 simulator.
  std::memset(state_staging, 0, sizeof(TwoQueueState));
  ASSERT_EQ(hsa_memory_copy(state, state_staging, sizeof(TwoQueueState)), HSA_STATUS_SUCCESS);
  for (uint32_t i = 0u; i < kernarg_allocation_count; ++i) {
    std::memset(kernargs[i], 0, kernarg_bytes);
    std::memcpy(kernargs[i], &state, sizeof(state));
  }

  uint32_t maximum_queue_size = 0u;
  uint32_t minimum_queue_size = 0u;
  ASSERT_EQ(hsa_agent_get_info(agents.gpu, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &maximum_queue_size),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_agent_get_info(agents.gpu, HSA_AGENT_INFO_QUEUE_MIN_SIZE, &minimum_queue_size),
            HSA_STATUS_SUCCESS);
  const uint32_t queue_size = std::max(minimum_queue_size, std::min(maximum_queue_size, 64u));
  for (uint32_t i = 0u; i < kQueueCount; ++i) {
    ASSERT_EQ(hsa_queue_create(agents.gpu, queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr,
                               UINT32_MAX, UINT32_MAX, &resources.queues_[i]),
              HSA_STATUS_SUCCESS);
    ASSERT_NE(resources.queues_[i], nullptr);
    ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &resources.signals_[i]), HSA_STATUS_SUCCESS);
  }
  ASSERT_NE(resources.queues_[0]->id, resources.queues_[1]->id);

  for (uint32_t round = 0u; round < kRounds; ++round) {
    std::array<uint64_t, kQueueCount> packet_ids{};
    for (uint32_t i = 0u; i < kQueueCount; ++i) {
      if (round != 0u)
        hsa_signal_store_relaxed(resources.signals_[i], 1);
      hsa_queue_t *queue = resources.queues_[i];
      packet_ids[i] = hsa_queue_add_write_index_relaxed(queue, 1u);
      auto *packet = static_cast<hsa_kernel_dispatch_packet_t *>(queue->base_address) +
                     (packet_ids[i] & (queue->size - 1u));
      std::memset(packet, 0, sizeof(*packet));
      packet->setup = 1u;
      packet->workgroup_size_x = kMaximumThreads;
      packet->workgroup_size_y = 1u;
      packet->workgroup_size_z = 1u;
      packet->grid_size_x = kMaximumThreads;
      packet->grid_size_y = 1u;
      packet->grid_size_z = 1u;
      packet->private_segment_size = private_bytes;
      packet->group_segment_size = group_bytes;
      packet->kernel_object = kernel_object;
      packet->kernarg_address = kernargs[i];
      packet->completion_signal = resources.signals_[i];

      uint16_t header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
      header |= 1u << HSA_PACKET_HEADER_BARRIER;
      header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
      header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
      __atomic_store_n(&packet->header, header, __ATOMIC_RELEASE);
    }
    ASSERT_EQ(packet_ids[0], packet_ids[1]);
    ASSERT_EQ(packet_ids[0], round);
    std::fprintf(stderr,
                 "CONSAN_TWO_QUEUE queue_ids=%llu,%llu raw_dispatch_ids=%llu,%llu "
                 "same_kernel=1 same_kernarg=%u round=%u\n",
                 static_cast<unsigned long long>(resources.queues_[0]->id),
                 static_cast<unsigned long long>(resources.queues_[1]->id),
                 static_cast<unsigned long long>(packet_ids[0]),
                 static_cast<unsigned long long>(packet_ids[1]), reuse_kernarg ? 1u : 0u,
                 round);

    for (uint32_t i = 0u; i < kQueueCount; ++i)
      hsa_signal_store_relaxed(resources.queues_[i]->doorbell_signal, packet_ids[i]);
    for (uint32_t i = 0u; i < kQueueCount; ++i) {
      const hsa_signal_value_t completion = hsa_signal_wait_scacquire(
          resources.signals_[i], HSA_SIGNAL_CONDITION_LT, 1, 5'000'000'000ULL,
          HSA_WAIT_STATE_BLOCKED);
      ASSERT_EQ(completion, 0) << "round=" << round << " queue=" << i
                               << " dispatch timed out or failed";
    }
  }

  ASSERT_EQ(hsa_memory_copy(state_staging, state, sizeof(TwoQueueState)), HSA_STATUS_SUCCESS);
  TwoQueueState result{};
  std::memcpy(&result, state_staging, sizeof(result));
  ASSERT_EQ(result.launches, kDispatches);
  for (uint32_t launch = 0u; launch < kDispatches; ++launch) {
    if (correct)
      EXPECT_EQ(result.observed[launch], 0xe9500000u | launch) << "launch=" << launch;
    for (uint32_t tid = 0u; tid < kMaximumThreads; ++tid) {
      EXPECT_EQ(result.control[launch][tid], 0xe9510000u | (launch << 8u) | tid)
          << "launch=" << launch << " tid=" << tid;
    }
  }
}

TEST(ConSanDeviceTwoQueueDispatchIdentityTest, Correct) {
  run_two_queue_dispatch(true, true);
  run_two_queue_dispatch(true, false);
}

TEST(ConSanDeviceTwoQueueDispatchIdentityTest, Incorrect) {
  run_two_queue_dispatch(false, true);
  run_two_queue_dispatch(false, false);
}

} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  const hsa_status_t init = hsa_init();
  if (init != HSA_STATUS_SUCCESS) {
    std::fprintf(stderr, "ConSan two-queue fixture: hsa_init failed status=%u\n",
                 static_cast<unsigned>(init));
    return 1;
  }
  const int result = RUN_ALL_TESTS();
  (void)hsa_shut_down();
  return result;
}
