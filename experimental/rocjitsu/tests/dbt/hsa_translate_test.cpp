// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hsa_translate_test.cpp
/// @brief End-to-end hardware test: translate CDNA4 vector_add → RDNA4,
/// load via HSA, dispatch on the real GPU, and verify against CPU golden.

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
RJ_DIAGNOSTIC_POP

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/dbt/binary_translator.h"
#include "rocjitsu/code/executable.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifdef HAS_HOST_AMDGPU

using namespace rocjitsu;

namespace {

std::string kernel_path(const char *name) { return std::string(KERNEL_DIR) + "/" + name + ".o"; }

hsa_agent_t find_gpu_agent() {
  hsa_agent_t gpu{};
  hsa_iterate_agents(
      [](hsa_agent_t agent, void *data) -> hsa_status_t {
        hsa_device_type_t type;
        hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
        if (type == HSA_DEVICE_TYPE_GPU) {
          *static_cast<hsa_agent_t *>(data) = agent;
          return HSA_STATUS_INFO_BREAK;
        }
        return HSA_STATUS_SUCCESS;
      },
      &gpu);
  return gpu;
}

hsa_agent_t find_cpu_agent() {
  hsa_agent_t cpu{};
  hsa_iterate_agents(
      [](hsa_agent_t agent, void *data) -> hsa_status_t {
        hsa_device_type_t type;
        hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
        if (type == HSA_DEVICE_TYPE_CPU) {
          *static_cast<hsa_agent_t *>(data) = agent;
          return HSA_STATUS_INFO_BREAK;
        }
        return HSA_STATUS_SUCCESS;
      },
      &cpu);
  return cpu;
}

hsa_amd_memory_pool_t find_pool(hsa_agent_t agent, hsa_amd_segment_t segment,
                                bool host_accessible = false) {
  struct Ctx {
    hsa_amd_segment_t seg;
    bool host_acc;
    hsa_amd_memory_pool_t pool;
  } ctx{segment, host_accessible, {}};

  hsa_amd_agent_iterate_memory_pools(
      agent,
      [](hsa_amd_memory_pool_t pool, void *data) -> hsa_status_t {
        auto *c = static_cast<Ctx *>(data);
        hsa_amd_segment_t seg;
        hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg);
        if (seg != c->seg)
          return HSA_STATUS_SUCCESS;
        if (c->host_acc) {
          bool acc = false;
          hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL, &acc);
          if (!acc)
            return HSA_STATUS_SUCCESS;
        }
        c->pool = pool;
        return HSA_STATUS_INFO_BREAK;
      },
      &ctx);
  return ctx.pool;
}

} // namespace

TEST(HsaTranslateTest, TranslateAndDispatchVectorAdd) {
  // 1. Translate CDNA4 → RDNA4.
  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid());
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);
  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  hsa_agent_t gpu = find_gpu_agent();
  ASSERT_NE(gpu.handle, 0u) << "No GPU agent found";

  hsa_isa_t isa{};
  hsa_agent_get_info(gpu, HSA_AGENT_INFO_ISA, &isa);
  char isa_name[128] = {};
  hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, isa_name);
  ASSERT_TRUE(std::strstr(isa_name, "gfx1200") || std::strstr(isa_name, "gfx1201"))
      << "Test requires RDNA4 GPU, found: " << isa_name;

  uint32_t target_mach = std::strstr(isa_name, "gfx1201") ? 0x4E : 0x48;

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4, target_mach);
  auto result = translator.translate(*co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << "Translation warnings: " << result.warnings.front();

  // 2. Load via HSA.
  hsa_agent_t cpu = find_cpu_agent();

  hsa_code_object_reader_t reader{};
  auto st = hsa_code_object_reader_create_from_memory(result.elf_bytes.data(),
                                                      result.elf_bytes.size(), &reader);
  ASSERT_EQ(st, HSA_STATUS_SUCCESS);

  hsa_executable_t executable{};
  st = hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, nullptr,
                                 &executable);
  ASSERT_EQ(st, HSA_STATUS_SUCCESS);
  st = hsa_executable_load_agent_code_object(executable, gpu, reader, nullptr, nullptr);
  ASSERT_EQ(st, HSA_STATUS_SUCCESS);
  st = hsa_executable_freeze(executable, nullptr);
  ASSERT_EQ(st, HSA_STATUS_SUCCESS);

  hsa_executable_symbol_t symbol{};
  st = hsa_executable_get_symbol_by_name(executable, "vector_add.kd", &gpu, &symbol);
  ASSERT_EQ(st, HSA_STATUS_SUCCESS);

  uint64_t kernel_object = 0;
  hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object);
  ASSERT_NE(kernel_object, 0u);

  // 3. Allocate GPU memory and dispatch.
  constexpr uint32_t N = 1024;
  constexpr size_t buf_size = N * sizeof(float);

  auto gpu_pool = find_pool(gpu, HSA_AMD_SEGMENT_GLOBAL);
  float *A_dev = nullptr, *B_dev = nullptr, *C_dev = nullptr;
  hsa_amd_memory_pool_allocate(gpu_pool, buf_size, 0, reinterpret_cast<void **>(&A_dev));
  hsa_amd_memory_pool_allocate(gpu_pool, buf_size, 0, reinterpret_cast<void **>(&B_dev));
  hsa_amd_memory_pool_allocate(gpu_pool, buf_size, 0, reinterpret_cast<void **>(&C_dev));
  ASSERT_NE(A_dev, nullptr);
  ASSERT_NE(B_dev, nullptr);
  ASSERT_NE(C_dev, nullptr);

  hsa_agent_t both[] = {cpu, gpu};
  hsa_amd_agents_allow_access(2, both, nullptr, A_dev);
  hsa_amd_agents_allow_access(2, both, nullptr, B_dev);
  hsa_amd_agents_allow_access(2, both, nullptr, C_dev);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
  std::vector<float> A_host(N), B_host(N), C_golden(N);
  for (uint32_t i = 0; i < N; ++i) {
    A_host[i] = dist(rng);
    B_host[i] = dist(rng);
    C_golden[i] = A_host[i] + B_host[i];
  }

  hsa_memory_copy(A_dev, A_host.data(), buf_size);
  hsa_memory_copy(B_dev, B_host.data(), buf_size);
  std::memset(C_dev, 0, buf_size);

  auto kernarg_pool = find_pool(cpu, HSA_AMD_SEGMENT_GLOBAL, true);
  void *kernarg = nullptr;
  hsa_amd_memory_pool_allocate(kernarg_pool, 256, 0, &kernarg);
  ASSERT_NE(kernarg, nullptr);
  hsa_amd_agents_allow_access(2, both, nullptr, kernarg);
  std::memset(kernarg, 0, 256);

  struct __attribute__((packed)) KernArgs {
    const float *A;
    const float *B;
    float *C;
    uint32_t N;
  };
  auto *args = static_cast<KernArgs *>(kernarg);
  args->A = A_dev;
  args->B = B_dev;
  args->C = C_dev;
  args->N = N;

  hsa_queue_t *queue = nullptr;
  uint32_t queue_size = 0;
  hsa_agent_get_info(gpu, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
  st = hsa_queue_create(gpu, queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, UINT32_MAX,
                        UINT32_MAX, &queue);
  ASSERT_EQ(st, HSA_STATUS_SUCCESS);

  hsa_signal_t signal{};
  hsa_signal_create(1, 0, nullptr, &signal);

  uint64_t write_idx = hsa_queue_add_write_index_relaxed(queue, 1);
  auto *aql = static_cast<hsa_kernel_dispatch_packet_t *>(queue->base_address) +
              (write_idx & (queue->size - 1));

  std::memset(aql, 0, sizeof(*aql));
  aql->setup = 1;
  aql->workgroup_size_x = 64;
  aql->workgroup_size_y = 1;
  aql->workgroup_size_z = 1;
  aql->grid_size_x = N;
  aql->grid_size_y = 1;
  aql->grid_size_z = 1;
  aql->kernel_object = kernel_object;
  aql->kernarg_address = kernarg;
  aql->completion_signal = signal;

  uint16_t header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  header |= 1 << HSA_PACKET_HEADER_BARRIER;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
  __atomic_store_n(reinterpret_cast<uint16_t *>(aql), header, __ATOMIC_RELEASE);

  hsa_signal_store_relaxed(queue->doorbell_signal, write_idx);

  // 4. Wait and verify.
  hsa_signal_value_t val = hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1,
                                                     5'000'000'000ULL, HSA_WAIT_STATE_BLOCKED);
  ASSERT_EQ(val, 0) << "Kernel dispatch timed out or failed";

  std::vector<float> C_result(N);
  hsa_memory_copy(C_result.data(), C_dev, buf_size);

  int mismatches = 0;
  for (uint32_t i = 0; i < N; ++i) {
    if (std::abs(C_result[i] - C_golden[i]) > 1e-5f)
      ++mismatches;
  }
  EXPECT_EQ(mismatches, 0) << mismatches << " element mismatches out of " << N;

  // 5. Cleanup.
  hsa_signal_destroy(signal);
  hsa_queue_destroy(queue);
  hsa_amd_memory_pool_free(kernarg);
  hsa_amd_memory_pool_free(A_dev);
  hsa_amd_memory_pool_free(B_dev);
  hsa_amd_memory_pool_free(C_dev);
  hsa_executable_destroy(executable);
  hsa_code_object_reader_destroy(reader);
  hsa_shut_down();
}

#endif // HAS_HOST_AMDGPU
