// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file cdna4_to_cdna3_dispatch_test.cpp
/// @brief End-to-end dispatch tests for CDNA4-to-CDNA3 DBT.

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
RJ_DIAGNOSTIC_POP

#include "../test_paths.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/dbt/binary_translator.h"
#include "rocjitsu/code/executable.h"
#include "rocjitsu/code/rj_code.h"
#include "util/data_types.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#ifdef HAS_HOST_AMDGPU
using namespace rocjitsu;

namespace {

using test::kernel_hsaco_path;
using test::kernel_path;

// Upper bound on how long to block for a dispatch's completion signal. This is a
// hang guard, NOT a performance budget: the real hang guard is the ctest-level
// TIMEOUT (120-300s). The larger fixtures (e.g. the 1024^3 buffer-async Triton
// matmul) are genuinely ~10s of single-threaded functional emulation; under CI's
// `ctest -j` the emulating thread is CPU-starved and a correct dispatch can take
// well past a tight per-signal budget, so a 5-10s wait produced false "dispatch
// timed out" failures. Keep this generously below the ctest TIMEOUT so a real hang
// still surfaces as a signal-wait failure rather than a hard ctest kill.
constexpr uint64_t kDispatchSignalWaitTimeoutNs = 120'000'000'000ULL;

std::vector<uint8_t> load_kernel_hsaco_bytes(const char *name) {
  std::ifstream file(kernel_hsaco_path(name), std::ios::binary);
  if (!file)
    return {};
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(file),
                              std::istreambuf_iterator<char>());
}

std::vector<uint8_t> load_gfx950_code_object(const char *name) {
  Executable exec(kernel_hsaco_path(name));
  if (!exec.is_valid() || exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950) == 0)
    return {};
  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  if (!co)
    return {};
  std::vector<uint8_t> bytes(co->image_size());
  std::memcpy(bytes.data(), co->image_data(), co->image_size());
  return bytes;
}

struct TritonMatmulCase {
  uint32_t m;
  uint32_t n;
  uint32_t k;
};

constexpr uint32_t kFlashAttentionQ = 1024;
constexpr uint32_t kFlashAttentionKV = 1024;
constexpr uint32_t kFlashAttentionHeadDim = 64;
constexpr uint32_t kFlashAttentionBlockN = 64;

struct FlashAttentionReferenceSlice {
  uint32_t row;
  uint32_t begin;
  uint32_t end;
};

constexpr std::array<FlashAttentionReferenceSlice, 2> kFlashAttentionReferenceSlices = {{
    {0, 0, kFlashAttentionHeadDim / 2},
    {kFlashAttentionQ - 1, kFlashAttentionHeadDim / 2, kFlashAttentionHeadDim},
}};

void expect_float_vectors_near(const std::vector<float> &actual, const std::vector<float> &expected,
                               float tolerance, const char *label) {
  ASSERT_EQ(actual.size(), expected.size()) << label;
  uint32_t mismatches = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    const float diff = std::fabs(actual[i] - expected[i]);
    if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) || diff > tolerance) {
      if (mismatches < 8)
        ADD_FAILURE() << label << " mismatch at i=" << i << ": got=" << actual[i]
                      << " expected=" << expected[i] << " diff=" << diff;
      ++mismatches;
    }
  }
  EXPECT_EQ(mismatches, 0u) << mismatches << " " << label << " mismatches";
}

void fill_seeded_half_inputs(std::vector<uint16_t> &values, uint32_t seed) {
  // Generate finite fp16 values directly in [-1, 1].  Sampling the raw
  // magnitude bits up to fp16 1.0 keeps this reproducible without baking in a
  // small fixed table of hand-picked values.
  std::mt19937 rng(seed);
  std::uniform_int_distribution<uint16_t> sign_dist(0, 1);
  std::uniform_int_distribution<uint16_t> magnitude_dist(0, 0x3c00);
  for (uint16_t &value : values) {
    const uint16_t sign = static_cast<uint16_t>(sign_dist(rng) << 15);
    value = static_cast<uint16_t>(sign | magnitude_dist(rng));
  }
}

std::vector<float> reference_flash_attention_slices(const std::vector<uint16_t> &q,
                                                    const std::vector<uint16_t> &k,
                                                    const std::vector<uint16_t> &v) {
  std::vector<float> result;
  result.reserve(kFlashAttentionHeadDim);

  for (const FlashAttentionReferenceSlice &slice : kFlashAttentionReferenceSlices) {
    float max_score = -1.0e20f;
    float normalization = 0.0f;
    std::vector<float> accumulator(slice.end - slice.begin, 0.0f);

    for (uint32_t block = 0; block < kFlashAttentionKV; block += kFlashAttentionBlockN) {
      std::array<float, kFlashAttentionBlockN> scores{};
      float block_max = -1.0e20f;
      for (uint32_t n = 0; n < kFlashAttentionBlockN; ++n) {
        float score = 0.0f;
        for (uint32_t d = 0; d < kFlashAttentionHeadDim; ++d) {
          const float q_value =
              util::f16_to_f32(q[static_cast<size_t>(slice.row) * kFlashAttentionHeadDim + d]);
          const float k_value =
              util::f16_to_f32(k[static_cast<size_t>(block + n) * kFlashAttentionHeadDim + d]);
          score = std::fma(q_value, k_value, score);
        }
        // The Triton fixture scales by log2(e) and uses exp2 for the online
        // softmax recurrence.
        scores[n] = score * 1.4426950408889634f;
        block_max = std::max(block_max, scores[n]);
      }

      const float new_max = std::max(max_score, block_max);
      const float alpha = std::exp2(max_score - new_max);
      float block_normalization = 0.0f;
      std::array<float, kFlashAttentionBlockN> probabilities{};
      for (uint32_t n = 0; n < kFlashAttentionBlockN; ++n) {
        const float probability = std::exp2(scores[n] - new_max);
        block_normalization += probability;
        probabilities[n] = util::f16_to_f32(util::f32_to_f16(probability));
      }

      for (float &value : accumulator)
        value *= alpha;
      for (uint32_t n = 0; n < kFlashAttentionBlockN; ++n) {
        for (uint32_t d = slice.begin; d < slice.end; ++d) {
          const float v_value =
              util::f16_to_f32(v[static_cast<size_t>(block + n) * kFlashAttentionHeadDim + d]);
          accumulator[d - slice.begin] =
              std::fma(probabilities[n], v_value, accumulator[d - slice.begin]);
        }
      }

      normalization = normalization * alpha + block_normalization;
      max_score = new_max;
    }

    for (float value : accumulator)
      result.push_back(value / normalization);
  }
  return result;
}

void expect_flash_attention_slices_near(const std::vector<float> &actual,
                                        const std::vector<float> &expected, float tolerance,
                                        const char *label) {
  ASSERT_EQ(actual.size(), static_cast<size_t>(kFlashAttentionQ) * kFlashAttentionHeadDim) << label;
  ASSERT_EQ(expected.size(), kFlashAttentionHeadDim) << label;

  uint32_t mismatches = 0;
  size_t reference_index = 0;
  for (const FlashAttentionReferenceSlice &slice : kFlashAttentionReferenceSlices) {
    for (uint32_t d = slice.begin; d < slice.end; ++d, ++reference_index) {
      const float actual_value =
          actual[static_cast<size_t>(slice.row) * kFlashAttentionHeadDim + d];
      const float expected_value = expected[reference_index];
      const float diff = std::fabs(actual_value - expected_value);
      if (!std::isfinite(actual_value) || !std::isfinite(expected_value) || diff > tolerance) {
        if (mismatches < 8)
          ADD_FAILURE() << label << " mismatch at row=" << slice.row << " d=" << d
                        << ": got=" << actual_value << " expected=" << expected_value
                        << " diff=" << diff;
        ++mismatches;
      }
    }
  }
  EXPECT_EQ(mismatches, 0u) << mismatches << " " << label << " mismatches";
}

std::vector<float> reference_triton_matmul(const TritonMatmulCase &test_case) {
  std::vector<uint16_t> a(static_cast<size_t>(test_case.m) * test_case.k);
  std::vector<uint16_t> b(static_cast<size_t>(test_case.k) * test_case.n);
  fill_seeded_half_inputs(a, 0xA0D4'0001u ^ test_case.m ^ (test_case.k << 8));
  fill_seeded_half_inputs(b, 0xB0D4'0001u ^ test_case.n ^ (test_case.k << 8));

  std::vector<float> result(static_cast<size_t>(test_case.m) * test_case.n, 0.0f);
  for (uint32_t m = 0; m < test_case.m; ++m) {
    for (uint32_t n = 0; n < test_case.n; ++n) {
      float sum = 0.0f;
      for (uint32_t k = 0; k < test_case.k; ++k) {
        const float lhs = util::f16_to_f32(a[static_cast<size_t>(m) * test_case.k + k]);
        const float rhs = util::f16_to_f32(b[static_cast<size_t>(k) * test_case.n + n]);
        sum = std::fma(lhs, rhs, sum);
      }
      result[static_cast<size_t>(m) * test_case.n + n] = sum;
    }
  }
  return result;
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

struct DispatchTarget {
  hsa_agent_t agent{};
  uint32_t mach = 0;
  std::string isa_name;
  std::vector<std::string> seen_gpu_isas;
};

uint32_t cdna3_mach_for_isa_name(const char *name) {
  if (std::strstr(name, "gfx940"))
    return EF_AMDGPU_MACH_AMDGCN_GFX940;
  if (std::strstr(name, "gfx941"))
    return EF_AMDGPU_MACH_AMDGCN_GFX941;
  if (std::strstr(name, "gfx942"))
    return EF_AMDGPU_MACH_AMDGCN_GFX942;
  return 0;
}

DispatchTarget find_cdna3_target() {
  DispatchTarget target;
  hsa_iterate_agents(
      [](hsa_agent_t agent, void *data) -> hsa_status_t {
        auto *t = static_cast<DispatchTarget *>(data);
        hsa_device_type_t type;
        hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
        if (type != HSA_DEVICE_TYPE_GPU)
          return HSA_STATUS_SUCCESS;

        hsa_isa_t isa{};
        char isa_name[128]{};
        hsa_agent_get_info(agent, HSA_AGENT_INFO_ISA, &isa);
        hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, isa_name);
        t->seen_gpu_isas.emplace_back(isa_name);

        const uint32_t mach = cdna3_mach_for_isa_name(isa_name);
        if (mach == 0)
          return HSA_STATUS_SUCCESS;

        t->agent = agent;
        t->mach = mach;
        t->isa_name = isa_name;
        return HSA_STATUS_INFO_BREAK;
      },
      &target);
  return target;
}

DispatchTarget find_gpu_target_named(std::string_view expected_isa) {
  struct Context {
    std::string_view expected_isa;
    DispatchTarget target;
  } context{expected_isa, {}};

  hsa_iterate_agents(
      [](hsa_agent_t agent, void *data) -> hsa_status_t {
        auto *ctx = static_cast<Context *>(data);
        hsa_device_type_t type;
        hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
        if (type != HSA_DEVICE_TYPE_GPU)
          return HSA_STATUS_SUCCESS;

        hsa_isa_t isa{};
        char isa_name[128]{};
        hsa_agent_get_info(agent, HSA_AGENT_INFO_ISA, &isa);
        hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, isa_name);
        ctx->target.seen_gpu_isas.emplace_back(isa_name);
        if (std::string_view(isa_name).find(ctx->expected_isa) == std::string_view::npos)
          return HSA_STATUS_SUCCESS;

        ctx->target.agent = agent;
        ctx->target.isa_name = isa_name;
        return HSA_STATUS_INFO_BREAK;
      },
      &context);
  return context.target;
}

std::string join_seen_isas(const std::vector<std::string> &isas) {
  if (isas.empty())
    return "<none>";
  std::string out = isas.front();
  for (size_t i = 1; i < isas.size(); ++i)
    out += ", " + isas[i];
  return out;
}

struct HsaShutdownGuard {
  ~HsaShutdownGuard() { hsa_shut_down(); }
};

struct HsaMatmulResources {
  hsa_code_object_reader_t reader{};
  hsa_executable_t executable{};
  hsa_queue_t *queue = nullptr;
  hsa_signal_t signal{};
  void *kernarg = nullptr;
  void *a = nullptr;
  void *b = nullptr;
  void *c = nullptr;
  bool reader_created = false;
  bool executable_created = false;
  bool signal_created = false;

  ~HsaMatmulResources() {
    // Stop queue execution before releasing anything referenced by an AQL
    // packet. This also makes an assertion after a dispatch timeout safe: the
    // test must not leave a live simulator dispatch for hsa_shut_down().
    if (queue)
      hsa_queue_destroy(queue);
    if (signal_created)
      hsa_signal_destroy(signal);
    if (kernarg)
      hsa_amd_memory_pool_free(kernarg);
    if (a)
      hsa_amd_memory_pool_free(a);
    if (b)
      hsa_amd_memory_pool_free(b);
    if (c)
      hsa_amd_memory_pool_free(c);
    if (executable_created)
      hsa_executable_destroy(executable);
    if (reader_created)
      hsa_code_object_reader_destroy(reader);
  }
};

void run_dynamic_copy_loop(const std::vector<uint8_t> &elf_bytes, const DispatchTarget &target) {
  hsa_agent_t cpu = find_cpu_agent();
  ASSERT_NE(cpu.handle, 0u) << "No CPU agent found";

  hsa_code_object_reader_t reader{};
  auto st = hsa_code_object_reader_create_from_memory(elf_bytes.data(), elf_bytes.size(), &reader);
  ASSERT_EQ(st, HSA_STATUS_SUCCESS);

  hsa_executable_t executable{};
  st = hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, nullptr,
                                 &executable);
  ASSERT_EQ(st, HSA_STATUS_SUCCESS);
  st = hsa_executable_load_agent_code_object(executable, target.agent, reader, nullptr, nullptr);
  ASSERT_EQ(st, HSA_STATUS_SUCCESS);
  st = hsa_executable_freeze(executable, nullptr);
  ASSERT_EQ(st, HSA_STATUS_SUCCESS);

  hsa_executable_symbol_t symbol{};
  st =
      hsa_executable_get_symbol_by_name(executable, "dynamic_copy_loop.kd", &target.agent, &symbol);
  ASSERT_EQ(st, HSA_STATUS_SUCCESS);

  uint64_t kernel_object = 0;
  hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object);
  ASSERT_NE(kernel_object, 0u);

  constexpr uint32_t kMaxN = 8193;
  constexpr uint32_t kWorkgroupSize = 64;
  constexpr uint32_t kDispatchWorkItems = 256;
  constexpr size_t kMaxBytes = kMaxN * sizeof(uint32_t);
  constexpr uint32_t kSentinel = 0xDEADBEEFu;

  auto gpu_pool = find_pool(target.agent, HSA_AMD_SEGMENT_GLOBAL);
  ASSERT_NE(gpu_pool.handle, 0u);
  uint32_t *src_dev = nullptr;
  uint32_t *dst_dev = nullptr;
  ASSERT_EQ(
      hsa_amd_memory_pool_allocate(gpu_pool, kMaxBytes, 0, reinterpret_cast<void **>(&src_dev)),
      HSA_STATUS_SUCCESS);
  ASSERT_EQ(
      hsa_amd_memory_pool_allocate(gpu_pool, kMaxBytes, 0, reinterpret_cast<void **>(&dst_dev)),
      HSA_STATUS_SUCCESS);

  hsa_agent_t both[] = {cpu, target.agent};
  ASSERT_EQ(hsa_amd_agents_allow_access(2, both, nullptr, src_dev), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_amd_agents_allow_access(2, both, nullptr, dst_dev), HSA_STATUS_SUCCESS);

  auto kernarg_pool = find_pool(cpu, HSA_AMD_SEGMENT_GLOBAL, true);
  ASSERT_NE(kernarg_pool.handle, 0u);
  void *kernarg = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(kernarg_pool, 256, 0, &kernarg), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_amd_agents_allow_access(2, both, nullptr, kernarg), HSA_STATUS_SUCCESS);
  std::memset(kernarg, 0, 256);

  struct __attribute__((packed)) KernArgs {
    const uint32_t *src;
    uint32_t *dst;
    uint32_t n;
    uint32_t workgroup_size;
    uint32_t stride;
  };
  auto *args = static_cast<KernArgs *>(kernarg);
  args->src = src_dev;
  args->dst = dst_dev;
  // The kernel is launched through raw HSA rather than the HIP runtime, so pass
  // the packet workgroup size explicitly instead of relying on blockDim.x.
  args->workgroup_size = kWorkgroupSize;
  args->stride = kDispatchWorkItems;

  hsa_queue_t *queue = nullptr;
  uint32_t queue_size = 0;
  hsa_agent_get_info(target.agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
  st = hsa_queue_create(target.agent, queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr,
                        UINT32_MAX, UINT32_MAX, &queue);
  ASSERT_EQ(st, HSA_STATUS_SUCCESS);

  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &signal), HSA_STATUS_SUCCESS);

  const std::array<uint32_t, 12> shapes = {0, 1, 17, 63, 64, 65, 255, 256, 257, 1024, 4097, kMaxN};
  std::vector<uint32_t> src_host(kMaxN);
  std::vector<uint32_t> dst_init(kMaxN, kSentinel);
  std::vector<uint32_t> dst_host(kMaxN);

  for (size_t iter = 0; iter < shapes.size(); ++iter) {
    const uint32_t n = shapes[iter];
    SCOPED_TRACE(::testing::Message()
                 << "shape N=" << n << " iter=" << iter << " host=" << target.isa_name);

    for (uint32_t i = 0; i < kMaxN; ++i)
      src_host[i] = 0x12340000u ^ (static_cast<uint32_t>(iter) << 12) ^ i;
    std::fill(dst_init.begin(), dst_init.end(), kSentinel);

    ASSERT_EQ(hsa_memory_copy(src_dev, src_host.data(), kMaxBytes), HSA_STATUS_SUCCESS);
    ASSERT_EQ(hsa_memory_copy(dst_dev, dst_init.data(), kMaxBytes), HSA_STATUS_SUCCESS);

    args->n = n;
    hsa_signal_store_relaxed(signal, 1);

    const uint64_t write_idx = hsa_queue_add_write_index_relaxed(queue, 1);
    auto *aql = static_cast<hsa_kernel_dispatch_packet_t *>(queue->base_address) +
                (write_idx & (queue->size - 1));
    std::memset(aql, 0, sizeof(*aql));
    aql->setup = 1;
    aql->workgroup_size_x = kWorkgroupSize;
    aql->workgroup_size_y = 1;
    aql->workgroup_size_z = 1;
    aql->grid_size_x = kDispatchWorkItems;
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

    const hsa_signal_value_t val = hsa_signal_wait_scacquire(
        signal, HSA_SIGNAL_CONDITION_LT, 1, kDispatchSignalWaitTimeoutNs, HSA_WAIT_STATE_BLOCKED);
    ASSERT_EQ(val, 0) << "Kernel dispatch timed out or failed";

    ASSERT_EQ(hsa_memory_copy(dst_host.data(), dst_dev, kMaxBytes), HSA_STATUS_SUCCESS);

    uint32_t mismatches = 0;
    for (uint32_t i = 0; i < kMaxN; ++i) {
      const uint32_t expected = (i < n) ? src_host[i] : kSentinel;
      if (dst_host[i] != expected) {
        if (mismatches < 8) {
          ADD_FAILURE() << "mismatch at i=" << i << ": got=0x" << std::hex << dst_host[i]
                        << " expected=0x" << expected << std::dec;
        }
        ++mismatches;
      }
    }
    EXPECT_EQ(mismatches, 0u) << mismatches << " mismatches for N=" << n;
  }

  hsa_signal_destroy(signal);
  hsa_queue_destroy(queue);
  hsa_amd_memory_pool_free(kernarg);
  hsa_amd_memory_pool_free(src_dev);
  hsa_amd_memory_pool_free(dst_dev);
  hsa_executable_destroy(executable);
  hsa_code_object_reader_destroy(reader);
}

void translate_triton_fixture(const char *name, uint32_t mach, std::vector<uint8_t> &elf_bytes,
                              BinaryTranslatorOptions options = {}) {
  Executable exec(kernel_hsaco_path(name));
  ASSERT_TRUE(exec.is_valid());
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);
  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, mach, options);
  auto translated = translator.translate(*co);
  ASSERT_FALSE(translated.elf_bytes.empty());
  ASSERT_TRUE(translated.ok()) << (translated.diagnostics.empty()
                                       ? "Translation failed without diagnostics"
                                       : translated.diagnostics.front().message);
  elf_bytes = std::move(translated.elf_bytes);
}

void translate_dynamic_triton_matmul(uint32_t mach, std::vector<uint8_t> &elf_bytes) {
  translate_triton_fixture("triton_cdna4_matmul_dynamic_32x32x64", mach, elf_bytes);
}

void run_triton_matmul(const std::vector<uint8_t> &elf_bytes, const DispatchTarget &target,
                       const TritonMatmulCase &test_case, uint32_t shared_bytes,
                       std::vector<float> *observed = nullptr) {
  hsa_agent_t cpu = find_cpu_agent();
  ASSERT_NE(cpu.handle, 0u) << "No CPU agent found";

  HsaMatmulResources resources;
  auto st = hsa_code_object_reader_create_from_memory(elf_bytes.data(), elf_bytes.size(),
                                                      &resources.reader);
  ASSERT_EQ(st, HSA_STATUS_SUCCESS);
  resources.reader_created = true;

  st = hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, nullptr,
                                 &resources.executable);
  ASSERT_EQ(st, HSA_STATUS_SUCCESS);
  resources.executable_created = true;
  st = hsa_executable_load_agent_code_object(resources.executable, target.agent, resources.reader,
                                             nullptr, nullptr);
  ASSERT_EQ(st, HSA_STATUS_SUCCESS);
  st = hsa_executable_freeze(resources.executable, nullptr);
  ASSERT_EQ(st, HSA_STATUS_SUCCESS);

  hsa_executable_symbol_t symbol{};
  st = hsa_executable_get_symbol_by_name(resources.executable, "triton_cdna4_matmul_kernel.kd",
                                         &target.agent, &symbol);
  ASSERT_EQ(st, HSA_STATUS_SUCCESS);

  uint64_t kernel_object = 0;
  hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object);
  ASSERT_NE(kernel_object, 0u);

  const uint32_t kM = test_case.m;
  const uint32_t kN = test_case.n;
  const uint32_t kK = test_case.k;
  constexpr uint32_t kBlockM = 32;
  constexpr uint32_t kBlockN = 32;
  constexpr uint32_t kWorkgroupSize = 256;
  const uint32_t kGroupsM = (kM + kBlockM - 1) / kBlockM;
  const uint32_t kGroupsN = (kN + kBlockN - 1) / kBlockN;
  constexpr float kSentinel = -12345.0f;
  const size_t kAElements = static_cast<size_t>(kM) * kK;
  const size_t kBElements = static_cast<size_t>(kK) * kN;
  const size_t kCElements = static_cast<size_t>(kM) * kN;
  const size_t kABytes = kAElements * sizeof(uint16_t);
  const size_t kBBytes = kBElements * sizeof(uint16_t);
  const size_t kCBytes = kCElements * sizeof(float);

  auto gpu_pool = find_pool(target.agent, HSA_AMD_SEGMENT_GLOBAL);
  ASSERT_NE(gpu_pool.handle, 0u);
  ASSERT_EQ(hsa_amd_memory_pool_allocate(gpu_pool, kABytes, 0, &resources.a), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_amd_memory_pool_allocate(gpu_pool, kBBytes, 0, &resources.b), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_amd_memory_pool_allocate(gpu_pool, kCBytes, 0, &resources.c), HSA_STATUS_SUCCESS);
  auto *a_dev = static_cast<uint16_t *>(resources.a);
  auto *b_dev = static_cast<uint16_t *>(resources.b);
  auto *c_dev = static_cast<float *>(resources.c);

  hsa_agent_t both[] = {cpu, target.agent};
  ASSERT_EQ(hsa_amd_agents_allow_access(2, both, nullptr, a_dev), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_amd_agents_allow_access(2, both, nullptr, b_dev), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_amd_agents_allow_access(2, both, nullptr, c_dev), HSA_STATUS_SUCCESS);

  auto kernarg_pool = find_pool(cpu, HSA_AMD_SEGMENT_GLOBAL, true);
  ASSERT_NE(kernarg_pool.handle, 0u);
  ASSERT_EQ(hsa_amd_memory_pool_allocate(kernarg_pool, 256, 0, &resources.kernarg),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_amd_agents_allow_access(2, both, nullptr, resources.kernarg), HSA_STATUS_SUCCESS);
  std::memset(resources.kernarg, 0, 256);

  struct __attribute__((packed)) DynamicKernArgs {
    const uint16_t *a;
    const uint16_t *b;
    float *c;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t padding;
    const void *unused0;
    const void *unused1;
  };
  static_assert(sizeof(DynamicKernArgs) == 56, "Dynamic Triton matmul kernarg layout changed");

  auto *args = static_cast<DynamicKernArgs *>(resources.kernarg);
  args->a = a_dev;
  args->b = b_dev;
  args->c = c_dev;
  args->m = kM;
  args->n = kN;
  args->k = kK;

  std::vector<uint16_t> a_host(kAElements);
  std::vector<uint16_t> b_host(kBElements);
  std::vector<float> c_init(kCElements, kSentinel);
  std::vector<float> c_host(kCElements);
  fill_seeded_half_inputs(a_host, 0xA0D4'0001u ^ kM ^ (kK << 8));
  fill_seeded_half_inputs(b_host, 0xB0D4'0001u ^ kN ^ (kK << 8));
  ASSERT_EQ(hsa_memory_copy(a_dev, a_host.data(), kABytes), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_memory_copy(b_dev, b_host.data(), kBBytes), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_memory_copy(c_dev, c_init.data(), kCBytes), HSA_STATUS_SUCCESS);

  uint32_t queue_size = 0;
  hsa_agent_get_info(target.agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
  st = hsa_queue_create(target.agent, queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr,
                        UINT32_MAX, UINT32_MAX, &resources.queue);
  ASSERT_EQ(st, HSA_STATUS_SUCCESS);

  ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &resources.signal), HSA_STATUS_SUCCESS);
  resources.signal_created = true;
  hsa_signal_store_relaxed(resources.signal, 1);

  const uint64_t write_idx = hsa_queue_add_write_index_relaxed(resources.queue, 1);
  auto *aql = static_cast<hsa_kernel_dispatch_packet_t *>(resources.queue->base_address) +
              (write_idx & (resources.queue->size - 1));
  std::memset(aql, 0, sizeof(*aql));
  aql->setup = 2;
  aql->workgroup_size_x = kWorkgroupSize;
  aql->workgroup_size_y = 1;
  aql->workgroup_size_z = 1;
  aql->grid_size_x = kGroupsM * kWorkgroupSize;
  aql->grid_size_y = kGroupsN;
  aql->grid_size_z = 1;
  aql->group_segment_size = shared_bytes;
  aql->kernel_object = kernel_object;
  aql->kernarg_address = resources.kernarg;
  aql->completion_signal = resources.signal;

  uint16_t header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  header |= 1 << HSA_PACKET_HEADER_BARRIER;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
  __atomic_store_n(reinterpret_cast<uint16_t *>(aql), header, __ATOMIC_RELEASE);
  hsa_signal_store_relaxed(resources.queue->doorbell_signal, write_idx);

  const hsa_signal_value_t val =
      hsa_signal_wait_scacquire(resources.signal, HSA_SIGNAL_CONDITION_LT, 1,
                                kDispatchSignalWaitTimeoutNs, HSA_WAIT_STATE_BLOCKED);
  ASSERT_EQ(val, 0) << "Kernel dispatch timed out or failed";

  ASSERT_EQ(hsa_memory_copy(c_host.data(), c_dev, kCBytes), HSA_STATUS_SUCCESS);

  if (observed)
    *observed = std::move(c_host);
}

template <size_t N>
void compare_dynamic_triton_matmul_cases(const DispatchTarget &target,
                                         const std::array<TritonMatmulCase, N> &cases) {
  std::vector<uint8_t> translated;
  ASSERT_NO_FATAL_FAILURE(translate_dynamic_triton_matmul(target.mach, translated));
  std::vector<uint8_t> native = load_kernel_hsaco_bytes("triton_cdna3_matmul_dynamic_32x32x64");
  ASSERT_FALSE(native.empty());

  for (const TritonMatmulCase &test_case : cases) {
    SCOPED_TRACE(::testing::Message()
                 << "M=" << test_case.m << " N=" << test_case.n << " K=" << test_case.k);
    std::vector<float> translated_out;
    std::vector<float> native_out;
    ASSERT_NO_FATAL_FAILURE(
        run_triton_matmul(translated, target, test_case, 8192, &translated_out));
    ASSERT_NO_FATAL_FAILURE(run_triton_matmul(native, target, test_case, 4096, &native_out));
    expect_float_vectors_near(translated_out, native_out, 0.02f,
                              "translated-vs-native dynamic Triton matmul");
  }
}

void run_buffer_async_triton_matmul(const std::vector<uint8_t> &elf_bytes,
                                    const DispatchTarget &target, uint32_t shared_bytes,
                                    std::vector<float> *observed = nullptr,
                                    std::vector<float> *reference = nullptr) {
  hsa_agent_t cpu = find_cpu_agent();
  ASSERT_NE(cpu.handle, 0u) << "No CPU agent found";

  hsa_code_object_reader_t reader{};
  auto st = hsa_code_object_reader_create_from_memory(elf_bytes.data(), elf_bytes.size(), &reader);
  ASSERT_EQ(st, HSA_STATUS_SUCCESS);

  hsa_executable_t executable{};
  st = hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, nullptr,
                                 &executable);
  ASSERT_EQ(st, HSA_STATUS_SUCCESS);
  st = hsa_executable_load_agent_code_object(executable, target.agent, reader, nullptr, nullptr);
  ASSERT_EQ(st, HSA_STATUS_SUCCESS);
  st = hsa_executable_freeze(executable, nullptr);
  ASSERT_EQ(st, HSA_STATUS_SUCCESS);

  hsa_executable_symbol_t symbol{};
  st = hsa_executable_get_symbol_by_name(executable, "matmul_async_buffer_load_lds.kd",
                                         &target.agent, &symbol);
  ASSERT_EQ(st, HSA_STATUS_SUCCESS);

  uint64_t kernel_object = 0;
  hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object);
  ASSERT_NE(kernel_object, 0u);

  constexpr uint32_t kM = 1024;
  constexpr uint32_t kN = 1024;
  constexpr uint32_t kK = 1024;
  constexpr uint32_t kBlockM = 64;
  constexpr uint32_t kBlockN = 64;
  constexpr uint32_t kWorkgroupSize = 256;
  constexpr uint32_t kGroups = ((kM + kBlockM - 1) / kBlockM) * ((kN + kBlockN - 1) / kBlockN);
  // Triton emits buffer-load-to-LDS instructions in this fixture, but the
  // metadata currently reports zero fixed LDS.  Give the dispatch the target
  // gfx942 maximum LDS footprint so the generated dynamic LDS offsets are
  // legal while still exercising real buffer-load-to-LDS execution.
  constexpr float kSentinel = -12345.0f;
  const size_t kAElements = static_cast<size_t>(kM) * kK;
  const size_t kBElements = static_cast<size_t>(kK) * kN;
  const size_t kCElements = static_cast<size_t>(kM) * kN;
  const size_t kABytes = kAElements * sizeof(uint16_t);
  const size_t kBBytes = kBElements * sizeof(uint16_t);
  const size_t kCBytes = kCElements * sizeof(float);

  auto gpu_pool = find_pool(target.agent, HSA_AMD_SEGMENT_GLOBAL);
  ASSERT_NE(gpu_pool.handle, 0u);
  uint16_t *a_dev = nullptr;
  uint16_t *b_dev = nullptr;
  float *c_dev = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(gpu_pool, kABytes, 0, reinterpret_cast<void **>(&a_dev)),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_amd_memory_pool_allocate(gpu_pool, kBBytes, 0, reinterpret_cast<void **>(&b_dev)),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_amd_memory_pool_allocate(gpu_pool, kCBytes, 0, reinterpret_cast<void **>(&c_dev)),
            HSA_STATUS_SUCCESS);

  hsa_agent_t both[] = {cpu, target.agent};
  ASSERT_EQ(hsa_amd_agents_allow_access(2, both, nullptr, a_dev), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_amd_agents_allow_access(2, both, nullptr, b_dev), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_amd_agents_allow_access(2, both, nullptr, c_dev), HSA_STATUS_SUCCESS);

  auto kernarg_pool = find_pool(cpu, HSA_AMD_SEGMENT_GLOBAL, true);
  ASSERT_NE(kernarg_pool.handle, 0u);
  void *kernarg = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(kernarg_pool, 64, 0, &kernarg), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_amd_agents_allow_access(2, both, nullptr, kernarg), HSA_STATUS_SUCCESS);
  std::memset(kernarg, 0, 64);

  struct __attribute__((packed)) BufferAsyncKernArgs {
    const uint16_t *a;
    const uint16_t *b;
    float *c;
    const void *unused0;
    const void *unused1;
  };
  static_assert(sizeof(BufferAsyncKernArgs) == 40,
                "Buffer async Triton matmul kernarg layout changed");

  auto *args = static_cast<BufferAsyncKernArgs *>(kernarg);
  args->a = a_dev;
  args->b = b_dev;
  args->c = c_dev;

  std::vector<uint16_t> a_host(kAElements);
  std::vector<uint16_t> b_host(kBElements);
  std::vector<float> c_init(kCElements, kSentinel);
  std::vector<float> c_host(kCElements);
  fill_seeded_half_inputs(a_host, 0xA0D4'1001u);
  if (reference) {
    // An identity RHS gives every output element a cheap closed-form CPU
    // reference while still exercising the full translated 1024^3 kernel.
    std::fill(b_host.begin(), b_host.end(), 0);
    for (uint32_t i = 0; i < kK; ++i)
      b_host[static_cast<size_t>(i) * kN + i] = 0x3c00; // fp16 1.0
    reference->resize(kCElements);
    for (size_t i = 0; i < kCElements; ++i)
      (*reference)[i] = util::f16_to_f32(a_host[i]);
  } else {
    fill_seeded_half_inputs(b_host, 0xB0D4'1001u);
  }
  ASSERT_EQ(hsa_memory_copy(a_dev, a_host.data(), kABytes), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_memory_copy(b_dev, b_host.data(), kBBytes), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_memory_copy(c_dev, c_init.data(), kCBytes), HSA_STATUS_SUCCESS);

  hsa_queue_t *queue = nullptr;
  uint32_t queue_size = 0;
  hsa_agent_get_info(target.agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
  st = hsa_queue_create(target.agent, queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr,
                        UINT32_MAX, UINT32_MAX, &queue);
  ASSERT_EQ(st, HSA_STATUS_SUCCESS);

  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &signal), HSA_STATUS_SUCCESS);
  hsa_signal_store_relaxed(signal, 1);

  const uint64_t write_idx = hsa_queue_add_write_index_relaxed(queue, 1);
  auto *aql = static_cast<hsa_kernel_dispatch_packet_t *>(queue->base_address) +
              (write_idx & (queue->size - 1));
  std::memset(aql, 0, sizeof(*aql));
  aql->setup = 1;
  aql->workgroup_size_x = kWorkgroupSize;
  aql->workgroup_size_y = 1;
  aql->workgroup_size_z = 1;
  aql->grid_size_x = kGroups * kWorkgroupSize;
  aql->grid_size_y = 1;
  aql->grid_size_z = 1;
  aql->group_segment_size = shared_bytes;
  aql->kernel_object = kernel_object;
  aql->kernarg_address = kernarg;
  aql->completion_signal = signal;

  uint16_t header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  header |= 1 << HSA_PACKET_HEADER_BARRIER;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
  __atomic_store_n(reinterpret_cast<uint16_t *>(aql), header, __ATOMIC_RELEASE);
  hsa_signal_store_relaxed(queue->doorbell_signal, write_idx);

  const hsa_signal_value_t val = hsa_signal_wait_scacquire(
      signal, HSA_SIGNAL_CONDITION_LT, 1, kDispatchSignalWaitTimeoutNs, HSA_WAIT_STATE_BLOCKED);
  ASSERT_EQ(val, 0) << "Kernel dispatch timed out or failed";

  ASSERT_EQ(hsa_memory_copy(c_host.data(), c_dev, kCBytes), HSA_STATUS_SUCCESS);

  if (observed)
    *observed = std::move(c_host);

  hsa_signal_destroy(signal);
  hsa_queue_destroy(queue);
  hsa_amd_memory_pool_free(kernarg);
  hsa_amd_memory_pool_free(a_dev);
  hsa_amd_memory_pool_free(b_dev);
  hsa_amd_memory_pool_free(c_dev);
  hsa_executable_destroy(executable);
  hsa_code_object_reader_destroy(reader);
}

void run_flash_attention_triton(const std::vector<uint8_t> &elf_bytes, const DispatchTarget &target,
                                const char *symbol_name, uint32_t shared_bytes,
                                std::vector<float> *observed = nullptr,
                                std::vector<float> *reference = nullptr) {
  hsa_agent_t cpu = find_cpu_agent();
  ASSERT_NE(cpu.handle, 0u) << "No CPU agent found";
  ASSERT_LE(shared_bytes, 65536u)
      << "gfx942 dispatch must stay within the 64 KiB LDS limit until LDS virtualization exists";

  hsa_code_object_reader_t reader{};
  auto st = hsa_code_object_reader_create_from_memory(elf_bytes.data(), elf_bytes.size(), &reader);
  ASSERT_EQ(st, HSA_STATUS_SUCCESS);

  hsa_executable_t executable{};
  st = hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, nullptr,
                                 &executable);
  ASSERT_EQ(st, HSA_STATUS_SUCCESS);
  st = hsa_executable_load_agent_code_object(executable, target.agent, reader, nullptr, nullptr);
  ASSERT_EQ(st, HSA_STATUS_SUCCESS);
  st = hsa_executable_freeze(executable, nullptr);
  ASSERT_EQ(st, HSA_STATUS_SUCCESS);

  hsa_executable_symbol_t symbol{};
  st = hsa_executable_get_symbol_by_name(executable, symbol_name, &target.agent, &symbol);
  ASSERT_EQ(st, HSA_STATUS_SUCCESS);

  uint64_t kernel_object = 0;
  hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object);
  ASSERT_NE(kernel_object, 0u);

  constexpr uint32_t kBlockM = 64;
  constexpr uint32_t kWorkgroupSize = 256;
  constexpr uint32_t kGroups = (kFlashAttentionQ + kBlockM - 1) / kBlockM;
  const size_t kQElements = static_cast<size_t>(kFlashAttentionQ) * kFlashAttentionHeadDim;
  const size_t kKVElements = static_cast<size_t>(kFlashAttentionKV) * kFlashAttentionHeadDim;
  const size_t kOElements = static_cast<size_t>(kFlashAttentionQ) * kFlashAttentionHeadDim;
  const size_t kQBytes = kQElements * sizeof(uint16_t);
  const size_t kKVBytes = kKVElements * sizeof(uint16_t);
  const size_t kOBytes = kOElements * sizeof(float);

  auto gpu_pool = find_pool(target.agent, HSA_AMD_SEGMENT_GLOBAL);
  ASSERT_NE(gpu_pool.handle, 0u);
  uint16_t *q_dev = nullptr;
  uint16_t *k_dev = nullptr;
  uint16_t *v_dev = nullptr;
  float *o_dev = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(gpu_pool, kQBytes, 0, reinterpret_cast<void **>(&q_dev)),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_amd_memory_pool_allocate(gpu_pool, kKVBytes, 0, reinterpret_cast<void **>(&k_dev)),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_amd_memory_pool_allocate(gpu_pool, kKVBytes, 0, reinterpret_cast<void **>(&v_dev)),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_amd_memory_pool_allocate(gpu_pool, kOBytes, 0, reinterpret_cast<void **>(&o_dev)),
            HSA_STATUS_SUCCESS);

  hsa_agent_t both[] = {cpu, target.agent};
  ASSERT_EQ(hsa_amd_agents_allow_access(2, both, nullptr, q_dev), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_amd_agents_allow_access(2, both, nullptr, k_dev), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_amd_agents_allow_access(2, both, nullptr, v_dev), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_amd_agents_allow_access(2, both, nullptr, o_dev), HSA_STATUS_SUCCESS);

  auto kernarg_pool = find_pool(cpu, HSA_AMD_SEGMENT_GLOBAL, true);
  ASSERT_NE(kernarg_pool.handle, 0u);
  void *kernarg = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(kernarg_pool, 64, 0, &kernarg), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_amd_agents_allow_access(2, both, nullptr, kernarg), HSA_STATUS_SUCCESS);
  std::memset(kernarg, 0, 64);

  struct __attribute__((packed)) FlashAttentionKernArgs {
    const uint16_t *q;
    const uint16_t *k;
    const uint16_t *v;
    float *o;
    float softmax_scale;
    uint32_t padding;
    const void *unused0;
    const void *unused1;
  };
  static_assert(sizeof(FlashAttentionKernArgs) == 56,
                "Triton flash attention kernarg layout changed");

  auto *args = static_cast<FlashAttentionKernArgs *>(kernarg);
  args->q = q_dev;
  args->k = k_dev;
  args->v = v_dev;
  args->o = o_dev;
  args->softmax_scale = 1.0f;

  std::vector<uint16_t> q_host(kQElements);
  std::vector<uint16_t> k_host(kKVElements);
  std::vector<uint16_t> v_host(kKVElements);
  std::vector<float> o_init(kOElements, -12345.0f);
  std::vector<float> o_host(kOElements);
  fill_seeded_half_inputs(q_host, 0xF1A5'0001u);
  fill_seeded_half_inputs(k_host, 0xF1A5'0002u);
  fill_seeded_half_inputs(v_host, 0xF1A5'0003u);
  if (reference)
    *reference = reference_flash_attention_slices(q_host, k_host, v_host);
  ASSERT_EQ(hsa_memory_copy(q_dev, q_host.data(), kQBytes), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_memory_copy(k_dev, k_host.data(), kKVBytes), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_memory_copy(v_dev, v_host.data(), kKVBytes), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_memory_copy(o_dev, o_init.data(), kOBytes), HSA_STATUS_SUCCESS);

  hsa_queue_t *queue = nullptr;
  uint32_t queue_size = 0;
  hsa_agent_get_info(target.agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
  st = hsa_queue_create(target.agent, queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr,
                        UINT32_MAX, UINT32_MAX, &queue);
  ASSERT_EQ(st, HSA_STATUS_SUCCESS);

  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &signal), HSA_STATUS_SUCCESS);
  hsa_signal_store_relaxed(signal, 1);

  const uint64_t write_idx = hsa_queue_add_write_index_relaxed(queue, 1);
  auto *aql = static_cast<hsa_kernel_dispatch_packet_t *>(queue->base_address) +
              (write_idx & (queue->size - 1));
  std::memset(aql, 0, sizeof(*aql));
  aql->setup = 1;
  aql->workgroup_size_x = kWorkgroupSize;
  aql->workgroup_size_y = 1;
  aql->workgroup_size_z = 1;
  aql->grid_size_x = kGroups * kWorkgroupSize;
  aql->grid_size_y = 1;
  aql->grid_size_z = 1;
  aql->group_segment_size = shared_bytes;
  aql->kernel_object = kernel_object;
  aql->kernarg_address = kernarg;
  aql->completion_signal = signal;

  uint16_t header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  header |= 1 << HSA_PACKET_HEADER_BARRIER;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
  __atomic_store_n(reinterpret_cast<uint16_t *>(aql), header, __ATOMIC_RELEASE);
  hsa_signal_store_relaxed(queue->doorbell_signal, write_idx);

  const hsa_signal_value_t val = hsa_signal_wait_scacquire(
      signal, HSA_SIGNAL_CONDITION_LT, 1, kDispatchSignalWaitTimeoutNs, HSA_WAIT_STATE_BLOCKED);
  ASSERT_EQ(val, 0) << "Kernel dispatch timed out or failed";

  ASSERT_EQ(hsa_memory_copy(o_host.data(), o_dev, kOBytes), HSA_STATUS_SUCCESS);

  if (observed)
    *observed = std::move(o_host);

  hsa_signal_destroy(signal);
  hsa_queue_destroy(queue);
  hsa_amd_memory_pool_free(kernarg);
  hsa_amd_memory_pool_free(q_dev);
  hsa_amd_memory_pool_free(k_dev);
  hsa_amd_memory_pool_free(v_dev);
  hsa_amd_memory_pool_free(o_dev);
  hsa_executable_destroy(executable);
  hsa_code_object_reader_destroy(reader);
}

} // namespace

TEST(Cdna4ToCdna3DispatchTest, DynamicCopyLoopTranslates) {
  Executable exec(kernel_path("dynamic_copy_loop"));
  ASSERT_TRUE(exec.is_valid());
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);
  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3,
                              EF_AMDGPU_MACH_AMDGCN_GFX942);
  auto translated = translator.translate(*co);
  ASSERT_FALSE(translated.elf_bytes.empty());
  ASSERT_TRUE(translated.ok()) << "Translation diagnostic: "
                               << translated.diagnostics.front().message;
}

TEST(Cdna4ToCdna3DispatchTest, TritonDynamicMatmulTranslates) {
  std::vector<uint8_t> translated;
  ASSERT_NO_FATAL_FAILURE(
      translate_dynamic_triton_matmul(EF_AMDGPU_MACH_AMDGCN_GFX942, translated));
}

TEST(Cdna4ToCdna3DispatchTest, TritonBufferAsyncMatmulTranslates) {
  std::vector<uint8_t> translated;
  ASSERT_NO_FATAL_FAILURE(translate_triton_fixture("triton_cdna4_matmul_buffer_async_1024",
                                                   EF_AMDGPU_MACH_AMDGCN_GFX942, translated));
}

TEST(Cdna4ToCdna3DispatchTest, TritonFlashAttentionNoAsyncTranslates) {
  std::vector<uint8_t> translated;
  ASSERT_NO_FATAL_FAILURE(translate_triton_fixture("triton_cdna4_flash_attention_no_async_1024",
                                                   EF_AMDGPU_MACH_AMDGCN_GFX942, translated));
}

TEST(Cdna4ToCdna3DispatchTest, TritonFlashAttentionBufferAsyncTranslates) {
  std::vector<uint8_t> translated;
  ASSERT_NO_FATAL_FAILURE(translate_triton_fixture("triton_cdna4_flash_attention_buffer_async_1024",
                                                   EF_AMDGPU_MACH_AMDGCN_GFX942, translated));
}

TEST(Cdna4ToCdna3DispatchTest, DynamicCopyLoopDispatchAndRun) {
  const hsa_status_t init_status = hsa_init();
  if (init_status != HSA_STATUS_SUCCESS)
    GTEST_SKIP() << "hsa_init failed with status " << init_status
                 << "; CDNA4->CDNA3 dispatch verification requires an HSA-capable host";
  const HsaShutdownGuard shutdown;

  DispatchTarget target = find_cdna3_target();
  if (target.agent.handle == 0)
    GTEST_SKIP() << "Test requires a CDNA3 GPU agent (gfx940/gfx941/gfx942); found: "
                 << join_seen_isas(target.seen_gpu_isas);

  Executable exec(kernel_path("dynamic_copy_loop"));
  ASSERT_TRUE(exec.is_valid());
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);
  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, target.mach);
  auto translated = translator.translate(*co);
  ASSERT_FALSE(translated.elf_bytes.empty());
  ASSERT_TRUE(translated.ok()) << "Translation diagnostic: "
                               << translated.diagnostics.front().message;

  ASSERT_NO_FATAL_FAILURE(run_dynamic_copy_loop(translated.elf_bytes, target));
}

TEST(Cdna4ToCdna3DbtGuestTest, DynamicCopyLoopDispatchAndRun) {
  const hsa_status_t init_status = hsa_init();
  ASSERT_EQ(init_status, HSA_STATUS_SUCCESS)
      << "DBT guest simulator launch must provide an HSA-capable execution backend";
  const HsaShutdownGuard shutdown;

  DispatchTarget guest = find_gpu_target_named("gfx950");
  ASSERT_NE(guest.agent.handle, 0u) << "DBT guest launch must publicly expose gfx950; found: "
                                    << join_seen_isas(guest.seen_gpu_isas);

  Executable exec(kernel_path("dynamic_copy_loop"));
  ASSERT_TRUE(exec.is_valid());
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);
  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);
  std::vector<uint8_t> guest_elf(co->image_size());
  std::memcpy(guest_elf.data(), co->image_data(), co->image_size());

  // Deliberately load the original gfx950 ELF through the public guest agent.
  // The DBT HSA hook owns translation and maps execution to simulated gfx942.
  ASSERT_NO_FATAL_FAILURE(run_dynamic_copy_loop(guest_elf, guest));
}

TEST(Cdna4ToCdna3DbtGuestTest, TritonDynamicMatmulDispatchAndRun) {
  const hsa_status_t init_status = hsa_init();
  ASSERT_EQ(init_status, HSA_STATUS_SUCCESS)
      << "DBT guest simulator launch must provide an HSA-capable execution backend";
  const HsaShutdownGuard shutdown;

  DispatchTarget guest = find_gpu_target_named("gfx950");
  ASSERT_NE(guest.agent.handle, 0u) << "DBT guest launch must publicly expose gfx950; found: "
                                    << join_seen_isas(guest.seen_gpu_isas);

  Executable exec(kernel_hsaco_path("triton_cdna4_matmul_dynamic_32x32x64"));
  ASSERT_TRUE(exec.is_valid());
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);
  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);
  std::vector<uint8_t> guest_elf(co->image_size());
  std::memcpy(guest_elf.data(), co->image_data(), co->image_size());

  constexpr TritonMatmulCase kCase{32, 32, 64};
  std::vector<float> observed;
  ASSERT_NO_FATAL_FAILURE(run_triton_matmul(guest_elf, guest, kCase, 8192, &observed));
  expect_float_vectors_near(observed, reference_triton_matmul(kCase), 0.02f,
                            "guest DBT dynamic Triton matmul");
}

TEST(Cdna4ToCdna3DbtGuestTest, TritonBufferAsyncMatmulDispatchAndRun) {
  const hsa_status_t init_status = hsa_init();
  ASSERT_EQ(init_status, HSA_STATUS_SUCCESS)
      << "DBT guest simulator launch must provide an HSA-capable execution backend";
  const HsaShutdownGuard shutdown;

  DispatchTarget guest = find_gpu_target_named("gfx950");
  ASSERT_NE(guest.agent.handle, 0u) << "DBT guest launch must publicly expose gfx950; found: "
                                    << join_seen_isas(guest.seen_gpu_isas);
  const std::vector<uint8_t> guest_elf =
      load_gfx950_code_object("triton_cdna4_matmul_buffer_async_1024");
  ASSERT_FALSE(guest_elf.empty());

  std::vector<float> observed;
  std::vector<float> expected;
  ASSERT_NO_FATAL_FAILURE(
      run_buffer_async_triton_matmul(guest_elf, guest, 65536, &observed, &expected));
  expect_float_vectors_near(observed, expected, 0.02f, "guest DBT buffer async Triton matmul");
}

TEST(Cdna4ToCdna3DbtGuestTest, TritonFlashAttentionNoAsyncDispatchAndRun) {
  const hsa_status_t init_status = hsa_init();
  ASSERT_EQ(init_status, HSA_STATUS_SUCCESS)
      << "DBT guest simulator launch must provide an HSA-capable execution backend";
  const HsaShutdownGuard shutdown;

  DispatchTarget guest = find_gpu_target_named("gfx950");
  ASSERT_NE(guest.agent.handle, 0u) << "DBT guest launch must publicly expose gfx950; found: "
                                    << join_seen_isas(guest.seen_gpu_isas);
  const std::vector<uint8_t> guest_elf =
      load_gfx950_code_object("triton_cdna4_flash_attention_no_async_1024");
  ASSERT_FALSE(guest_elf.empty());

  std::vector<float> observed;
  std::vector<float> expected;
  ASSERT_NO_FATAL_FAILURE(run_flash_attention_triton(
      guest_elf, guest, "flash_attention_fwd_no_async_kernel.kd", 65536, &observed, &expected));
  expect_flash_attention_slices_near(observed, expected, 0.02f,
                                     "guest DBT flash attention no-async");
}

TEST(Cdna4ToCdna3DbtGuestTest, TritonFlashAttentionBufferAsyncDispatchAndRun) {
  const hsa_status_t init_status = hsa_init();
  ASSERT_EQ(init_status, HSA_STATUS_SUCCESS)
      << "DBT guest simulator launch must provide an HSA-capable execution backend";
  const HsaShutdownGuard shutdown;

  DispatchTarget guest = find_gpu_target_named("gfx950");
  ASSERT_NE(guest.agent.handle, 0u) << "DBT guest launch must publicly expose gfx950; found: "
                                    << join_seen_isas(guest.seen_gpu_isas);
  const std::vector<uint8_t> guest_elf =
      load_gfx950_code_object("triton_cdna4_flash_attention_buffer_async_1024");
  ASSERT_FALSE(guest_elf.empty());

  std::vector<float> observed;
  std::vector<float> expected;
  ASSERT_NO_FATAL_FAILURE(run_flash_attention_triton(
      guest_elf, guest, "flash_attention_fwd_async_buffer_kernel.kd", 65536, &observed, &expected));
  expect_flash_attention_slices_near(observed, expected, 0.02f,
                                     "guest DBT flash attention buffer-async");
}

TEST(Cdna4ToCdna3DispatchTest, TritonDynamicMatmulDispatchAndRun) {
  const std::array<TritonMatmulCase, 5> cases = {{
      {32, 32, 64},
      {31, 31, 63},
      {33, 33, 65},
      {128, 128, 128},
      {65, 33, 130},
  }};

  const hsa_status_t init_status = hsa_init();
  if (init_status != HSA_STATUS_SUCCESS)
    GTEST_SKIP() << "hsa_init failed with status " << init_status
                 << "; CDNA4->CDNA3 dispatch verification requires an HSA-capable host";
  const HsaShutdownGuard shutdown;

  DispatchTarget target = find_cdna3_target();
  if (target.agent.handle == 0)
    GTEST_SKIP() << "Test requires a CDNA3 GPU agent (gfx940/gfx941/gfx942); found: "
                 << join_seen_isas(target.seen_gpu_isas);

  ASSERT_NO_FATAL_FAILURE(compare_dynamic_triton_matmul_cases(target, cases));
}

TEST(Cdna4ToCdna3DispatchTest, TritonBufferAsyncMatmulDispatchAndRun) {
  const hsa_status_t init_status = hsa_init();
  if (init_status != HSA_STATUS_SUCCESS)
    GTEST_SKIP() << "hsa_init failed with status " << init_status
                 << "; CDNA4->CDNA3 dispatch verification requires an HSA-capable host";
  const HsaShutdownGuard shutdown;

  DispatchTarget target = find_cdna3_target();
  if (target.agent.handle == 0)
    GTEST_SKIP() << "Test requires a CDNA3 GPU agent (gfx940/gfx941/gfx942); found: "
                 << join_seen_isas(target.seen_gpu_isas);

  std::vector<uint8_t> translated;
  ASSERT_NO_FATAL_FAILURE(
      translate_triton_fixture("triton_cdna4_matmul_buffer_async_1024", target.mach, translated));
  std::vector<uint8_t> native = load_kernel_hsaco_bytes("triton_cdna3_matmul_buffer_async_1024");
  ASSERT_FALSE(native.empty());

  std::vector<float> translated_out;
  std::vector<float> native_out;
  ASSERT_NO_FATAL_FAILURE(
      run_buffer_async_triton_matmul(translated, target, 65536, &translated_out));
  ASSERT_NO_FATAL_FAILURE(run_buffer_async_triton_matmul(native, target, 8192, &native_out));
  expect_float_vectors_near(translated_out, native_out, 0.02f,
                            "translated-vs-native buffer async Triton matmul");
}

TEST(Cdna4ToCdna3DispatchTest, TritonBufferAsyncMatmulConservativeLivenessDispatchAndRun) {
  const hsa_status_t init_status = hsa_init();
  if (init_status != HSA_STATUS_SUCCESS)
    GTEST_SKIP() << "hsa_init failed with status " << init_status
                 << "; CDNA4->CDNA3 dispatch verification requires an HSA-capable host";
  const HsaShutdownGuard shutdown;

  DispatchTarget target = find_cdna3_target();
  if (target.agent.handle == 0)
    GTEST_SKIP() << "Test requires a CDNA3 GPU agent (gfx940/gfx941/gfx942); found: "
                 << join_seen_isas(target.seen_gpu_isas);

  BinaryTranslatorOptions options;
  // The source descriptor declares 88 ordinary VGPRs and places AccVGPRs above
  // that window. For this debug run, force semantic scratch above the ordinary
  // VGPR range without changing the actual live-before sets.
  options.debug_min_free_vgpr = 88;

  std::vector<uint8_t> translated;
  ASSERT_NO_FATAL_FAILURE(translate_triton_fixture("triton_cdna4_matmul_buffer_async_1024",
                                                   target.mach, translated, options));
  std::vector<uint8_t> native = load_kernel_hsaco_bytes("triton_cdna3_matmul_buffer_async_1024");
  ASSERT_FALSE(native.empty());

  std::vector<float> translated_out;
  std::vector<float> native_out;
  ASSERT_NO_FATAL_FAILURE(
      run_buffer_async_triton_matmul(translated, target, 65536, &translated_out));
  ASSERT_NO_FATAL_FAILURE(run_buffer_async_triton_matmul(native, target, 8192, &native_out));
  expect_float_vectors_near(
      translated_out, native_out, 0.02f,
      "translated-vs-native conservative-liveness buffer async Triton matmul");
}

TEST(Cdna4ToCdna3DispatchTest, TritonFlashAttentionNoAsyncDispatchAndRun) {
  const hsa_status_t init_status = hsa_init();
  if (init_status != HSA_STATUS_SUCCESS)
    GTEST_SKIP() << "hsa_init failed with status " << init_status
                 << "; CDNA4->CDNA3 dispatch verification requires an HSA-capable host";
  const HsaShutdownGuard shutdown;

  DispatchTarget target = find_cdna3_target();
  if (target.agent.handle == 0)
    GTEST_SKIP() << "Test requires a CDNA3 GPU agent (gfx940/gfx941/gfx942); found: "
                 << join_seen_isas(target.seen_gpu_isas);

  std::vector<uint8_t> translated;
  ASSERT_NO_FATAL_FAILURE(translate_triton_fixture("triton_cdna4_flash_attention_no_async_1024",
                                                   target.mach, translated));
  std::vector<uint8_t> native =
      load_kernel_hsaco_bytes("triton_cdna3_flash_attention_no_async_1024");
  ASSERT_FALSE(native.empty());

  std::vector<float> translated_out;
  std::vector<float> native_out;
  ASSERT_NO_FATAL_FAILURE(run_flash_attention_triton(
      translated, target, "flash_attention_fwd_no_async_kernel.kd", 65536, &translated_out));
  ASSERT_NO_FATAL_FAILURE(run_flash_attention_triton(
      native, target, "flash_attention_fwd_no_async_kernel.kd", 8192, &native_out));
  expect_float_vectors_near(translated_out, native_out, 0.02f,
                            "translated-vs-native flash attention no-async");
}

TEST(Cdna4ToCdna3DispatchTest, TritonFlashAttentionBufferAsyncDispatchAndRun) {
  const hsa_status_t init_status = hsa_init();
  if (init_status != HSA_STATUS_SUCCESS)
    GTEST_SKIP() << "hsa_init failed with status " << init_status
                 << "; CDNA4->CDNA3 dispatch verification requires an HSA-capable host";
  const HsaShutdownGuard shutdown;

  DispatchTarget target = find_cdna3_target();
  if (target.agent.handle == 0)
    GTEST_SKIP() << "Test requires a CDNA3 GPU agent (gfx940/gfx941/gfx942); found: "
                 << join_seen_isas(target.seen_gpu_isas);

  std::vector<uint8_t> translated;
  ASSERT_NO_FATAL_FAILURE(translate_triton_fixture("triton_cdna4_flash_attention_buffer_async_1024",
                                                   target.mach, translated));
  std::vector<uint8_t> native =
      load_kernel_hsaco_bytes("triton_cdna3_flash_attention_buffer_async_1024");
  ASSERT_FALSE(native.empty());

  std::vector<float> translated_out;
  std::vector<float> native_out;
  ASSERT_NO_FATAL_FAILURE(run_flash_attention_triton(
      translated, target, "flash_attention_fwd_async_buffer_kernel.kd", 65536, &translated_out));
  ASSERT_NO_FATAL_FAILURE(run_flash_attention_triton(
      native, target, "flash_attention_fwd_async_buffer_kernel.kd", 8192, &native_out));
  expect_float_vectors_near(translated_out, native_out, 0.02f,
                            "translated-vs-native flash attention buffer-async");
}

#endif // HAS_HOST_AMDGPU
