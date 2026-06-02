// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hsa_translate_test.cpp
/// @brief End-to-end hardware test: translate CDNA4 kernels to the host GPU target,
/// load via HSA, dispatch on the real GPU, and verify against CPU golden.

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
RJ_DIAGNOSTIC_POP

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
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

struct HostTarget {
  rj_code_arch_t arch;
  uint32_t mach;
  char isa_name[128];
};

// Inspect the running GPU's ISA name and pick the matching translator target
// arch + ELF e_flags MACH constant. Returns mach == 0 for unsupported ISAs;
// callers should ASSERT_NE(target.mach, 0u).
HostTarget select_host_target(hsa_agent_t gpu) {
  HostTarget t{};
  hsa_isa_t isa{};
  hsa_agent_get_info(gpu, HSA_AGENT_INFO_ISA, &isa);
  hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, t.isa_name);
  // GFX940/941/942 are all CDNA3 ISA targets. Keep the DBT arch selection the
  // same for all three, but preserve the exact ELF MACH value that HSA expects
  // for the host agent's code object loader.
  if (std::strstr(t.isa_name, "gfx940")) {
    t.arch = ROCJITSU_CODE_ARCH_CDNA3;
    t.mach = EF_AMDGPU_MACH_AMDGCN_GFX940;
  } else if (std::strstr(t.isa_name, "gfx941")) {
    t.arch = ROCJITSU_CODE_ARCH_CDNA3;
    t.mach = EF_AMDGPU_MACH_AMDGCN_GFX941;
  } else if (std::strstr(t.isa_name, "gfx942")) {
    t.arch = ROCJITSU_CODE_ARCH_CDNA3;
    t.mach = EF_AMDGPU_MACH_AMDGCN_GFX942;
  } else if (std::strstr(t.isa_name, "gfx1100")) {
    t.arch = ROCJITSU_CODE_ARCH_RDNA3;
    t.mach = EF_AMDGPU_MACH_AMDGCN_GFX1100;
  } else if (std::strstr(t.isa_name, "gfx1201")) {
    t.arch = ROCJITSU_CODE_ARCH_RDNA4;
    t.mach = EF_AMDGPU_MACH_AMDGCN_GFX1201;
  } else if (std::strstr(t.isa_name, "gfx1200")) {
    t.arch = ROCJITSU_CODE_ARCH_RDNA4;
    t.mach = EF_AMDGPU_MACH_AMDGCN_GFX1200;
  }
  return t;
}

} // namespace

TEST(HsaTranslateTest, TranslateAndDispatchVectorAdd) {
  // 1. Translate CDNA4 to the selected host target.
  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid());
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);
  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  hsa_agent_t gpu = find_gpu_agent();
  ASSERT_NE(gpu.handle, 0u) << "No GPU agent found";

  auto target = select_host_target(gpu);
  ASSERT_NE(target.mach, 0u) << "Test requires CDNA3 (gfx940/941/942), RDNA3 (gfx1100), or RDNA4 "
                                "(gfx1200/1201) GPU, found: "
                             << target.isa_name;

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, target.arch, target.mach);
  auto result = translator.translate(*co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.ok()) << "Translation diagnostic: " << result.diagnostics.front().message;

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
  // Zero the output buffer via hsa_memory_copy (works for both coarse- and
  // fine-grained pools); a direct std::memset would segfault when the pool
  // isn't CPU-mapped.
  const std::vector<std::uint8_t> c_zero_bytes(buf_size, 0);
  ASSERT_EQ(hsa_memory_copy(C_dev, c_zero_bytes.data(), buf_size), HSA_STATUS_SUCCESS);

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

TEST(HsaTranslateTest, TranslateAndDispatchMfma16x16) {
  // 1. Translate CDNA4 matmul_mfma_16x16 to a host with MFMA lowering support.
  Executable exec(kernel_path("matmul_mfma_16x16"));
  ASSERT_TRUE(exec.is_valid());
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);
  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  hsa_agent_t gpu = find_gpu_agent();
  ASSERT_NE(gpu.handle, 0u);

  auto target = select_host_target(gpu);
  ASSERT_NE(target.mach, 0u) << "Test requires RDNA3 (gfx1100) or RDNA4 (gfx1200/1201) GPU, found: "
                             << target.isa_name;
  // RDNA3 accepts the target selector for vector_add, but this MFMA test still
  // needs RDNA4-only MFMA->WMMA expansion rules. Skip instead of translating
  // unsupported MFMA instructions into NOPs and comparing invalid output.
  if (target.arch != ROCJITSU_CODE_ARCH_RDNA4) {
    hsa_shut_down();
    GTEST_SKIP() << "MFMA→WMMA semantic expansion is currently implemented only for RDNA4; found: "
                 << target.isa_name;
  }

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, target.arch, target.mach);
  auto result = translator.translate(*co);
  ASSERT_FALSE(result.elf_bytes.empty());

  // Dump translated CO for offline inspection.
  {
    auto *f = std::fopen("/tmp/translated_mfma16.co", "wb");
    if (f) {
      std::fwrite(result.elf_bytes.data(), 1, result.elf_bytes.size(), f);
      std::fclose(f);
    }
  }

  // 2. Load translated CO via HSA.
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
  st = hsa_executable_get_symbol_by_name(executable, "matmul_mfma_16x16.kd", &gpu, &symbol);
  ASSERT_EQ(st, HSA_STATUS_SUCCESS);

  uint64_t kernel_object = 0;
  hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object);
  ASSERT_NE(kernel_object, 0u);

  // 3. Allocate 16x16 FP16 inputs and FP32 output.
  constexpr uint32_t M = 16;
  constexpr uint32_t N = 16;
  constexpr uint32_t K = 16;
  constexpr size_t ab_size = M * K * sizeof(uint16_t); // FP16
  constexpr size_t c_size = M * N * sizeof(float);     // FP32

  auto gpu_pool = find_pool(gpu, HSA_AMD_SEGMENT_GLOBAL);
  uint16_t *A_dev = nullptr, *B_dev = nullptr;
  float *C_dev = nullptr;
  hsa_amd_memory_pool_allocate(gpu_pool, ab_size, 0, reinterpret_cast<void **>(&A_dev));
  hsa_amd_memory_pool_allocate(gpu_pool, ab_size, 0, reinterpret_cast<void **>(&B_dev));
  hsa_amd_memory_pool_allocate(gpu_pool, c_size, 0, reinterpret_cast<void **>(&C_dev));
  ASSERT_NE(A_dev, nullptr);
  ASSERT_NE(B_dev, nullptr);
  ASSERT_NE(C_dev, nullptr);

  hsa_agent_t both[] = {cpu, gpu};
  hsa_amd_agents_allow_access(2, both, nullptr, A_dev);
  hsa_amd_agents_allow_access(2, both, nullptr, B_dev);
  hsa_amd_agents_allow_access(2, both, nullptr, C_dev);

  auto f32_to_f16 = [](float val) -> uint16_t {
    uint32_t fbits;
    std::memcpy(&fbits, &val, 4);
    uint32_t sign = (fbits >> 16) & 0x8000;
    int32_t exp = static_cast<int32_t>((fbits >> 23) & 0xFF) - 127;
    uint32_t mant = fbits & 0x7FFFFF;
    if ((fbits & 0x7FFFFFFF) == 0)
      return static_cast<uint16_t>(sign);
    if (exp > 15)
      return static_cast<uint16_t>(sign | 0x7BFF);
    if (exp < -14)
      return static_cast<uint16_t>(sign);
    return static_cast<uint16_t>(sign | ((exp + 15) << 10) | (mant >> 13));
  };
  auto f16_to_f32 = [](uint16_t h) -> float {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0)
      return sign ? -0.0f : 0.0f;
    uint32_t f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    float result;
    std::memcpy(&result, &f, 4);
    return result;
  };

  // Set up queue, signal, and kernarg (reused across iterations).
  auto kernarg_pool = find_pool(cpu, HSA_AMD_SEGMENT_GLOBAL, true);
  void *kernarg = nullptr;
  hsa_amd_memory_pool_allocate(kernarg_pool, 256, 0, &kernarg);
  ASSERT_NE(kernarg, nullptr);
  hsa_amd_agents_allow_access(2, both, nullptr, kernarg);
  std::memset(kernarg, 0, 256);
  struct __attribute__((packed)) KernArgs {
    const uint16_t *A;
    const uint16_t *B;
    float *C;
  };
  auto *args = static_cast<KernArgs *>(kernarg);
  args->A = A_dev;
  args->B = B_dev;
  args->C = C_dev;

  hsa_queue_t *queue = nullptr;
  uint32_t queue_size = 0;
  hsa_agent_get_info(gpu, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
  st = hsa_queue_create(gpu, queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, UINT32_MAX,
                        UINT32_MAX, &queue);
  ASSERT_EQ(st, HSA_STATUS_SUCCESS);

  hsa_signal_t signal{};
  hsa_signal_create(1, 0, nullptr, &signal);

  // Fuzz with random FP16 inputs over multiple iterations.
  constexpr int kNumIterations = 10;
  std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  int total_mismatches = 0;

  for (int iter = 0; iter < kNumIterations; ++iter) {
    std::vector<uint16_t> A_host(M * K), B_host(K * N);
    for (auto &v : A_host)
      v = f32_to_f16(dist(rng));
    for (auto &v : B_host)
      v = f32_to_f16(dist(rng));

    hsa_memory_copy(A_dev, A_host.data(), ab_size);
    hsa_memory_copy(B_dev, B_host.data(), ab_size);
    // Device allocations are not necessarily host-addressable on dGPUs, so
    // zero the output through HSA instead of calling std::memset on C_dev.
    const std::vector<std::uint8_t> c_zero_bytes(c_size, 0);
    ASSERT_EQ(hsa_memory_copy(C_dev, c_zero_bytes.data(), c_size), HSA_STATUS_SUCCESS);

    std::vector<float> C_golden(M * N, 0.0f);
    for (uint32_t i = 0; i < M; ++i)
      for (uint32_t j = 0; j < N; ++j)
        for (uint32_t k = 0; k < K; ++k)
          C_golden[i * N + j] += f16_to_f32(A_host[i * K + k]) * f16_to_f32(B_host[k * N + j]);

    // Dispatch.
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

    std::vector<float> C_result(M * N);
    hsa_memory_copy(C_result.data(), C_dev, c_size);

    int mismatches = 0;
    for (uint32_t i = 0; i < M * N; ++i) {
      float expected = C_golden[i];
      float got = C_result[i];
      float tol = std::max(0.01f * std::abs(expected), 0.001f);
      if (std::abs(got - expected) > tol)
        ++mismatches;
    }

    if (mismatches > 0) {
      std::fprintf(stderr, "\n=== MFMA 16x16 iter %d: %d mismatches ===\n", iter, mismatches);
      for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < N; ++col) {
          uint32_t raw;
          std::memcpy(&raw, &C_result[row * N + col], 4);
          std::fprintf(stderr, "%08X ", raw);
        }
        std::fprintf(stderr, "\n");
      }
    }
    total_mismatches += mismatches;

    hsa_signal_store_relaxed(signal, 1); // reset for next iteration
  } // end fuzzing loop

  std::fprintf(stderr, "\nMFMA 16x16: %d iterations, %d total mismatches\n", kNumIterations,
               total_mismatches);
  EXPECT_EQ(total_mismatches, 0) << total_mismatches << " total mismatches across "
                                 << kNumIterations << " iterations";

  // 6. Cleanup.
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
