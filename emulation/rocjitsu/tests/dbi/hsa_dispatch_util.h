// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hsa_dispatch_util.h
/// @brief Shared HSA agent/pool discovery + vector_add dispatch scaffold for
///        the DBI smoke tests. Both hsa_dbi_nop_asm_test.cpp (inline-nop path)
///        and hsa_dbi_nop_probe_test.cpp (probe-call path) load a patched ELF
///        and dispatch it the same way; this header is the single copy of that
///        boilerplate.
///
/// Functions are `inline` so the header can be included in multiple TUs without
/// ODR violations. They require a live HSA runtime; callers gate on a real
/// gfx90a agent before invoking dispatch_vector_add.

#pragma once

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
RJ_DIAGNOSTIC_POP

#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

namespace rocjitsu::dbi_test {

// Find a GPU agent whose ISA name contains "gfx90a". Returns {handle=0} if
// no such agent is present.
inline hsa_agent_t find_gfx90a_agent() {
  hsa_agent_t result{};
  hsa_iterate_agents(
      [](hsa_agent_t agent, void *data) -> hsa_status_t {
        hsa_device_type_t type;
        hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
        if (type != HSA_DEVICE_TYPE_GPU)
          return HSA_STATUS_SUCCESS;
        hsa_isa_t isa{};
        hsa_agent_get_info(agent, HSA_AGENT_INFO_ISA, &isa);
        char isa_name[128]{};
        hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, isa_name);
        if (std::strstr(isa_name, "gfx90a")) {
          *static_cast<hsa_agent_t *>(data) = agent;
          return HSA_STATUS_INFO_BREAK;
        }
        return HSA_STATUS_SUCCESS;
      },
      &result);
  return result;
}

inline hsa_agent_t find_cpu_agent() {
  hsa_agent_t result{};
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
      &result);
  return result;
}

inline hsa_amd_memory_pool_t find_pool(hsa_agent_t agent, hsa_amd_segment_t segment,
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

// Dispatch the vector_add kernel from @p elf_bytes and return the output
// buffer. Returns an empty vector on any HSA failure (the caller asserts on
// size). Mirrors the dispatch scaffold in tests/dbt/hsa_translate_test.cpp.
inline std::vector<float> dispatch_vector_add(std::span<const uint8_t> elf_bytes, hsa_agent_t gpu,
                                              hsa_agent_t cpu, const std::vector<float> &a_in,
                                              const std::vector<float> &b_in, uint32_t n) {
  constexpr size_t kKernArgsBytes = 256;
  const size_t buf_size = n * sizeof(float);

  hsa_code_object_reader_t reader{};
  if (hsa_code_object_reader_create_from_memory(elf_bytes.data(), elf_bytes.size(), &reader) !=
      HSA_STATUS_SUCCESS)
    return {};

  hsa_executable_t executable{};
  if (hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, nullptr,
                                &executable) != HSA_STATUS_SUCCESS) {
    hsa_code_object_reader_destroy(reader);
    return {};
  }
  if (hsa_executable_load_agent_code_object(executable, gpu, reader, nullptr, nullptr) !=
          HSA_STATUS_SUCCESS ||
      hsa_executable_freeze(executable, nullptr) != HSA_STATUS_SUCCESS) {
    hsa_executable_destroy(executable);
    hsa_code_object_reader_destroy(reader);
    return {};
  }

  hsa_executable_symbol_t symbol{};
  uint64_t kernel_object = 0;
  if (hsa_executable_get_symbol_by_name(executable, "vector_add.kd", &gpu, &symbol) !=
          HSA_STATUS_SUCCESS ||
      hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                     &kernel_object) != HSA_STATUS_SUCCESS ||
      kernel_object == 0) {
    hsa_executable_destroy(executable);
    hsa_code_object_reader_destroy(reader);
    return {};
  }

  // Pool + buffer + queue setup. Each step is checked so a failure on a
  // flaky system returns an empty result cleanly rather than crashing on a
  // null pointer or zero handle later. A single cleanup epilogue runs
  // regardless of success.
  auto gpu_pool = find_pool(gpu, HSA_AMD_SEGMENT_GLOBAL);
  auto kernarg_pool = find_pool(cpu, HSA_AMD_SEGMENT_GLOBAL, /*host_accessible=*/true);
  float *A_dev = nullptr, *B_dev = nullptr, *C_dev = nullptr;
  void *kernarg = nullptr;
  hsa_queue_t *queue = nullptr;
  hsa_signal_t signal{};
  std::vector<float> out;

  bool ok = (gpu_pool.handle != 0) && (kernarg_pool.handle != 0);
  if (ok)
    ok = hsa_amd_memory_pool_allocate(gpu_pool, buf_size, 0, reinterpret_cast<void **>(&A_dev)) ==
         HSA_STATUS_SUCCESS;
  if (ok)
    ok = hsa_amd_memory_pool_allocate(gpu_pool, buf_size, 0, reinterpret_cast<void **>(&B_dev)) ==
         HSA_STATUS_SUCCESS;
  if (ok)
    ok = hsa_amd_memory_pool_allocate(gpu_pool, buf_size, 0, reinterpret_cast<void **>(&C_dev)) ==
         HSA_STATUS_SUCCESS;
  if (ok)
    ok = hsa_amd_memory_pool_allocate(kernarg_pool, kKernArgsBytes, 0, &kernarg) ==
         HSA_STATUS_SUCCESS;

  if (ok) {
    hsa_agent_t both[] = {cpu, gpu};
    hsa_amd_agents_allow_access(2, both, nullptr, A_dev);
    hsa_amd_agents_allow_access(2, both, nullptr, B_dev);
    hsa_amd_agents_allow_access(2, both, nullptr, C_dev);
    hsa_amd_agents_allow_access(2, both, nullptr, kernarg);

    hsa_memory_copy(A_dev, a_in.data(), buf_size);
    hsa_memory_copy(B_dev, b_in.data(), buf_size);
    const std::vector<uint8_t> zero_bytes(buf_size, 0);
    hsa_memory_copy(C_dev, zero_bytes.data(), buf_size);

    std::memset(kernarg, 0, kKernArgsBytes);
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
    args->N = n;

    uint32_t queue_size = 0;
    hsa_agent_get_info(gpu, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
    ok = hsa_queue_create(gpu, queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, UINT32_MAX,
                          UINT32_MAX, &queue) == HSA_STATUS_SUCCESS &&
         queue != nullptr;
  }

  if (ok)
    ok = hsa_signal_create(1, 0, nullptr, &signal) == HSA_STATUS_SUCCESS;

  if (ok) {
    uint64_t write_idx = hsa_queue_add_write_index_relaxed(queue, 1);
    auto *aql = static_cast<hsa_kernel_dispatch_packet_t *>(queue->base_address) +
                (write_idx & (queue->size - 1));
    std::memset(aql, 0, sizeof(*aql));
    aql->setup = 1;
    aql->workgroup_size_x = 64;
    aql->workgroup_size_y = 1;
    aql->workgroup_size_z = 1;
    aql->grid_size_x = n;
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
    if (val == 0) {
      out.resize(n);
      hsa_memory_copy(out.data(), C_dev, buf_size);
    }
  }

  // Cleanup epilogue: each call is guarded so partial-initialization paths
  // don't dereference null handles.
  if (signal.handle != 0)
    hsa_signal_destroy(signal);
  if (queue != nullptr)
    hsa_queue_destroy(queue);
  if (kernarg != nullptr)
    hsa_amd_memory_pool_free(kernarg);
  if (A_dev != nullptr)
    hsa_amd_memory_pool_free(A_dev);
  if (B_dev != nullptr)
    hsa_amd_memory_pool_free(B_dev);
  if (C_dev != nullptr)
    hsa_amd_memory_pool_free(C_dev);
  hsa_executable_destroy(executable);
  hsa_code_object_reader_destroy(reader);
  return out;
}

} // namespace rocjitsu::dbi_test
