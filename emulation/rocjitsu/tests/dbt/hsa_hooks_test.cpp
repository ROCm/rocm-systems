// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hsa_hooks_test.cpp
/// @brief End-to-end HSA_TOOLS_LIB test for load-time DBT translation.
///
/// @details This test intentionally creates an HSA code-object reader from the
/// original gfx950 ELF bytes. It does not call BinaryTranslator directly. The
/// expected execution path is:
///   1. ROCR loads librocjitsu_hooks.so from HSA_TOOLS_LIB during hsa_init().
///   2. The hook records the gfx950 reader bytes.
///   3. hsa_executable_load_agent_code_object() is intercepted and translated
///      to the RJ_DBT_TARGET_ISA selected by the test environment.
///   4. The translated object is loaded and dispatched on the local GPU.

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
RJ_DIAGNOSTIC_POP

#include "rocjitsu/code/executable.h"
#include "rocjitsu/code/rj_code.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifdef HAS_HOST_AMDGPU
using namespace rocjitsu;

namespace {

std::string kernel_path(const char *name) { return std::string(KERNEL_DIR) + "/" + name + ".o"; }

struct GpuTarget {
  hsa_agent_t agent{};
  std::string isa_name;
  std::vector<std::string> seen_gpu_isas;
};

std::string join_seen_isas(const std::vector<std::string> &isas) {
  if (isas.empty())
    return "<none>";
  std::string out = isas.front();
  for (size_t i = 1; i < isas.size(); ++i) {
    out += ", ";
    out += isas[i];
  }
  return out;
}

GpuTarget find_gpu_agent_for_isa(const char *target_isa) {
  struct Ctx {
    const char *target_isa = nullptr;
    GpuTarget result;
  } ctx{target_isa, {}};

  hsa_iterate_agents(
      [](hsa_agent_t agent, void *data) -> hsa_status_t {
        auto *ctx = static_cast<Ctx *>(data);
        hsa_device_type_t type{};
        hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
        if (type != HSA_DEVICE_TYPE_GPU)
          return HSA_STATUS_SUCCESS;

        hsa_isa_t isa{};
        char isa_name[128]{};
        hsa_agent_get_info(agent, HSA_AGENT_INFO_ISA, &isa);
        hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, isa_name);
        ctx->result.seen_gpu_isas.emplace_back(isa_name);

        if (ctx->target_isa != nullptr && std::strstr(isa_name, ctx->target_isa) != nullptr) {
          ctx->result.agent = agent;
          ctx->result.isa_name = isa_name;
          return HSA_STATUS_INFO_BREAK;
        }
        return HSA_STATUS_SUCCESS;
      },
      &ctx);
  return ctx.result;
}

hsa_agent_t find_cpu_agent() {
  hsa_agent_t cpu{};
  hsa_iterate_agents(
      [](hsa_agent_t agent, void *data) -> hsa_status_t {
        hsa_device_type_t type{};
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
        hsa_amd_segment_t seg{};
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

TEST(HsaHooksTest, TranslateGfx950Mfma16x16ThroughToolsLib) {
  Executable exec(kernel_path("matmul_mfma_16x16"));
  ASSERT_TRUE(exec.is_valid());
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);
  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);
  ASSERT_TRUE(co->is_valid());

  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  const char *target_isa = std::getenv("RJ_DBT_TARGET_ISA");
  ASSERT_NE(target_isa, nullptr) << "HSA hook test requires RJ_DBT_TARGET_ISA";
  // The hook target and dispatch agent must agree exactly. On multi-GPU hosts,
  // the first HSA GPU can be a different gfx stepping from the CTest-selected
  // RJ_DBT_TARGET_ISA, which makes ROCR reject the translated code object.
  GpuTarget gpu_target = find_gpu_agent_for_isa(target_isa);
  hsa_agent_t gpu = gpu_target.agent;
  hsa_agent_t cpu = find_cpu_agent();
  ASSERT_NE(gpu.handle, 0u) << "No GPU agent matched RJ_DBT_TARGET_ISA=" << target_isa
                            << "; seen GPU ISAs: " << join_seen_isas(gpu_target.seen_gpu_isas);
  ASSERT_NE(cpu.handle, 0u);

  // Pass the original gfx950 ELF to ROCR. librocjitsu_hooks.so must perform the
  // same gfx950->gfx1201 MFMA lowering that HsaTranslateTest exercises directly.
  hsa_code_object_reader_t reader{};
  auto st = hsa_code_object_reader_create_from_memory(co->image_data(), co->image_size(), &reader);
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
  st = hsa_executable_get_symbol_by_name(executable, "matmul_mfma_16x16.kd", &gpu, &symbol);
  ASSERT_EQ(st, HSA_STATUS_SUCCESS);

  uint64_t kernel_object = 0;
  hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object);
  ASSERT_NE(kernel_object, 0u);

  constexpr uint32_t M = 16;
  constexpr uint32_t N = 16;
  constexpr uint32_t K = 16;
  constexpr size_t ab_size = M * K * sizeof(uint16_t);
  constexpr size_t c_size = M * N * sizeof(float);

  auto gpu_pool = find_pool(gpu, HSA_AMD_SEGMENT_GLOBAL);
  uint16_t *a_dev = nullptr;
  uint16_t *b_dev = nullptr;
  float *c_dev = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(gpu_pool, ab_size, 0, reinterpret_cast<void **>(&a_dev)),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_amd_memory_pool_allocate(gpu_pool, ab_size, 0, reinterpret_cast<void **>(&b_dev)),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_amd_memory_pool_allocate(gpu_pool, c_size, 0, reinterpret_cast<void **>(&c_dev)),
            HSA_STATUS_SUCCESS);

  hsa_agent_t both[] = {cpu, gpu};
  hsa_amd_agents_allow_access(2, both, nullptr, a_dev);
  hsa_amd_agents_allow_access(2, both, nullptr, b_dev);
  hsa_amd_agents_allow_access(2, both, nullptr, c_dev);

  auto f32_to_f16 = [](float val) -> uint16_t {
    uint32_t fbits;
    std::memcpy(&fbits, &val, 4);
    const uint32_t sign = (fbits >> 16) & 0x8000;
    const int32_t exp = static_cast<int32_t>((fbits >> 23) & 0xFF) - 127;
    const uint32_t mant = fbits & 0x7FFFFF;
    if ((fbits & 0x7FFFFFFF) == 0)
      return static_cast<uint16_t>(sign);
    if (exp > 15)
      return static_cast<uint16_t>(sign | 0x7BFF);
    if (exp < -14)
      return static_cast<uint16_t>(sign);
    return static_cast<uint16_t>(sign | ((exp + 15) << 10) | (mant >> 13));
  };
  auto f16_to_f32 = [](uint16_t h) -> float {
    const uint32_t sign = (h >> 15) & 1;
    const uint32_t exp = (h >> 10) & 0x1F;
    const uint32_t mant = h & 0x3FF;
    if (exp == 0)
      return sign ? -0.0f : 0.0f;
    const uint32_t f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    float result;
    std::memcpy(&result, &f, 4);
    return result;
  };

  auto kernarg_pool = find_pool(cpu, HSA_AMD_SEGMENT_GLOBAL, true);
  void *kernarg = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(kernarg_pool, 256, 0, &kernarg), HSA_STATUS_SUCCESS);
  hsa_amd_agents_allow_access(2, both, nullptr, kernarg);
  std::memset(kernarg, 0, 256);

  struct __attribute__((packed)) KernArgs {
    const uint16_t *a;
    const uint16_t *b;
    float *c;
  };
  auto *args = static_cast<KernArgs *>(kernarg);
  args->a = a_dev;
  args->b = b_dev;
  args->c = c_dev;

  hsa_queue_t *queue = nullptr;
  uint32_t queue_size = 0;
  hsa_agent_get_info(gpu, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
  st = hsa_queue_create(gpu, queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, UINT32_MAX,
                        UINT32_MAX, &queue);
  ASSERT_EQ(st, HSA_STATUS_SUCCESS);

  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &signal), HSA_STATUS_SUCCESS);

  constexpr int kNumIterations = 10;
  std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  int total_mismatches = 0;

  for (int iter = 0; iter < kNumIterations; ++iter) {
    std::vector<uint16_t> a_host(M * K), b_host(K * N);
    for (uint16_t &value : a_host)
      value = f32_to_f16(dist(rng));
    for (uint16_t &value : b_host)
      value = f32_to_f16(dist(rng));

    ASSERT_EQ(hsa_memory_copy(a_dev, a_host.data(), ab_size), HSA_STATUS_SUCCESS);
    ASSERT_EQ(hsa_memory_copy(b_dev, b_host.data(), ab_size), HSA_STATUS_SUCCESS);
    const std::vector<std::uint8_t> zero(c_size, 0);
    ASSERT_EQ(hsa_memory_copy(c_dev, zero.data(), c_size), HSA_STATUS_SUCCESS);

    std::vector<float> c_golden(M * N, 0.0f);
    for (uint32_t i = 0; i < M; ++i)
      for (uint32_t j = 0; j < N; ++j)
        for (uint32_t k = 0; k < K; ++k)
          c_golden[i * N + j] += f16_to_f32(a_host[i * K + k]) * f16_to_f32(b_host[k * N + j]);

    uint64_t write_idx = hsa_queue_add_write_index_relaxed(queue, 1);
    auto *aql = static_cast<hsa_kernel_dispatch_packet_t *>(queue->base_address) +
                (write_idx & (queue->size - 1));

    std::memset(aql, 0, sizeof(*aql));
    aql->setup = 1;
    aql->workgroup_size_x = 64;
    aql->workgroup_size_y = 1;
    aql->workgroup_size_z = 1;
    aql->grid_size_x = 64;
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

    hsa_signal_value_t val = hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1,
                                                       5'000'000'000ULL, HSA_WAIT_STATE_BLOCKED);
    ASSERT_EQ(val, 0) << "Kernel dispatch timed out or failed (iter " << iter << ")";

    std::vector<float> c_result(M * N);
    ASSERT_EQ(hsa_memory_copy(c_result.data(), c_dev, c_size), HSA_STATUS_SUCCESS);

    int mismatches = 0;
    for (uint32_t i = 0; i < M * N; ++i) {
      const float expected = c_golden[i];
      const float got = c_result[i];
      const float tol = std::max(0.01f * std::abs(expected), 0.001f);
      if (std::abs(got - expected) > tol)
        ++mismatches;
    }
    total_mismatches += mismatches;

    hsa_signal_store_relaxed(signal, 1);
  }
  EXPECT_EQ(total_mismatches, 0) << total_mismatches << " total mismatches across "
                                 << kNumIterations << " iterations";

  hsa_signal_destroy(signal);
  hsa_queue_destroy(queue);
  hsa_amd_memory_pool_free(kernarg);
  hsa_amd_memory_pool_free(a_dev);
  hsa_amd_memory_pool_free(b_dev);
  hsa_amd_memory_pool_free(c_dev);
  hsa_executable_destroy(executable);
  hsa_code_object_reader_destroy(reader);
  hsa_shut_down();
}

#endif // HAS_HOST_AMDGPU
