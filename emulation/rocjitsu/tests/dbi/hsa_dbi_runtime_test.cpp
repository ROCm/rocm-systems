// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// The DBI runtime hook end to end: an HSA application loads the uninstrumented
// block_reduce fixture, the hook (installed by ROCr from HSA_TOOLS_LIB)
// instruments it at load, and the dispatch runs the instrumented kernel.
//
// The binary links the hook's test-only library and HSA_TOOLS_LIB names the
// same file, so the dynamic loader maps it once and the rj_test_dbi_* seams
// read the instance ROCr installed. The same cases run on hardware and on the
// simulator; only the CTest registration differs.

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
RJ_DIAGNOSTIC_POP

#include "../test_paths.h"
#include "dbi_test_fixtures.h"
#include "hsa_dispatch_util.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/executable.h"
#include "rocjitsu/code/patch/probe_symbol.h"
#include "rocjitsu/hooks/dbi_runtime_sites.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <optional>
#include <span>
#include <string>
#include <vector>

extern "C" bool rj_test_dbi_set_probe(const void *elf, size_t size, const char *symbol);
extern "C" uint64_t rj_test_dbi_loads();
extern "C" uint64_t rj_test_dbi_decisions(int decision);
extern "C" uint64_t rj_test_dbi_sites();
extern "C" size_t rj_test_dbi_retained_buffer_count();
extern "C" void rj_test_dbi_reset();

namespace rocjitsu::dbi_test {
namespace {

using hooks::DbiLoadDecision;
using test::kernel_hsaco_path;
using test::kernel_path;

constexpr const char *kProbeSymbol = "rj_nop_probe";
constexpr uint32_t kBlockSize = 256; // dbi_block_reduce.hip's kBlockSize.
constexpr uint32_t kBlocks = 4;
// Every input is 1, so the result is the exact integer sum of i % 64.
constexpr uint32_t kElements = 4096;
constexpr float kGolden = (kElements / 64) * (63 * 64 / 2);

constexpr DbiTargetParams kCdna4{ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_TARGET_GFX950,
                                 "dbi_block_reduce_gfx950", "rj_nop_probe_gfx950", "gfx950"};
constexpr DbiTargetParams kRdna4{ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_TARGET_GFX1201,
                                 "dbi_block_reduce_gfx1201", "rj_nop_probe_gfx1201", "gfx1201"};

struct BlockReduceRun {
  std::optional<float> sum;    ///< Empty on any HSA failure.
  size_t retained_buffers = 0; ///< Instrumented objects the hook held while loaded.
};

uint64_t decisions(DbiLoadDecision decision) {
  return rj_test_dbi_decisions(static_cast<int>(decision));
}

std::vector<uint8_t> read_file(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(input),
                              std::istreambuf_iterator<char>());
}

// Loads @p elf through the HSA API, which is where the hook instruments it, and
// runs block_reduce over kElements ones.
BlockReduceRun dispatch_block_reduce(std::span<const uint8_t> elf, hsa_agent_t gpu,
                                     hsa_agent_t cpu) {
  hsa_code_object_reader_t reader{};
  hsa_executable_t executable{};
  hsa_executable_symbol_t symbol{};
  uint64_t kernel_object = 0;
  uint32_t group_segment_size = 0;
  uint32_t private_segment_size = 0;
  uint32_t kernarg_segment_size = 0;
  float *in_dev = nullptr;
  float *out_dev = nullptr;
  void *kernarg = nullptr;
  hsa_queue_t *queue = nullptr;
  hsa_signal_t signal{};
  BlockReduceRun result;

  bool ok =
      hsa_code_object_reader_create_from_memory(elf.data(), elf.size(), &reader) ==
          HSA_STATUS_SUCCESS &&
      hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, nullptr,
                                &executable) == HSA_STATUS_SUCCESS &&
      hsa_executable_load_agent_code_object(executable, gpu, reader, nullptr, nullptr) ==
          HSA_STATUS_SUCCESS &&
      hsa_executable_freeze(executable, nullptr) == HSA_STATUS_SUCCESS &&
      hsa_executable_get_symbol_by_name(executable, "block_reduce.kd", &gpu, &symbol) ==
          HSA_STATUS_SUCCESS &&
      hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                     &kernel_object) == HSA_STATUS_SUCCESS &&
      hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
                                     &group_segment_size) == HSA_STATUS_SUCCESS &&
      hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                                     &private_segment_size) == HSA_STATUS_SUCCESS &&
      hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
                                     &kernarg_segment_size) == HSA_STATUS_SUCCESS;
  result.retained_buffers = rj_test_dbi_retained_buffer_count();

  struct __attribute__((packed)) KernArgs {
    const float *in;
    float *out;
    uint32_t n;
    uint32_t stride;
  };
  const size_t kernarg_bytes = std::max<size_t>(kernarg_segment_size, sizeof(KernArgs));
  const hsa_amd_memory_pool_t gpu_pool = find_pool(gpu, HSA_AMD_SEGMENT_GLOBAL);
  const hsa_amd_memory_pool_t kernarg_pool =
      find_pool(cpu, HSA_AMD_SEGMENT_GLOBAL, /*host_accessible=*/true);
  ok = ok && gpu_pool.handle != 0 && kernarg_pool.handle != 0 &&
       hsa_amd_memory_pool_allocate(gpu_pool, kElements * sizeof(float), 0,
                                    reinterpret_cast<void **>(&in_dev)) == HSA_STATUS_SUCCESS &&
       hsa_amd_memory_pool_allocate(gpu_pool, sizeof(float), 0,
                                    reinterpret_cast<void **>(&out_dev)) == HSA_STATUS_SUCCESS &&
       hsa_amd_memory_pool_allocate(kernarg_pool, kernarg_bytes, 0, &kernarg) == HSA_STATUS_SUCCESS;

  if (ok) {
    hsa_agent_t both[] = {cpu, gpu};
    hsa_amd_agents_allow_access(2, both, nullptr, in_dev);
    hsa_amd_agents_allow_access(2, both, nullptr, out_dev);
    hsa_amd_agents_allow_access(2, both, nullptr, kernarg);
    const std::vector<float> ones(kElements, 1.0f);
    const float zero = 0.0f;
    ok = synchronized_hsa_memory_copy(in_dev, ones.data(), kElements * sizeof(float)) ==
             HSA_STATUS_SUCCESS &&
         synchronized_hsa_memory_copy(out_dev, &zero, sizeof(zero)) == HSA_STATUS_SUCCESS;
  }
  if (ok) {
    std::memset(kernarg, 0, kernarg_bytes);
    auto *args = static_cast<KernArgs *>(kernarg);
    args->in = in_dev;
    args->out = out_dev;
    args->n = kElements;
    args->stride = kBlocks * kBlockSize;
    annotate_hsa_copy_release(kernarg, sizeof(*args));

    uint32_t queue_size = 0;
    hsa_agent_get_info(gpu, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
    ok = hsa_queue_create(gpu, queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, UINT32_MAX,
                          UINT32_MAX, &queue) == HSA_STATUS_SUCCESS &&
         hsa_signal_create(1, 0, nullptr, &signal) == HSA_STATUS_SUCCESS;
  }
  if (ok) {
    const uint64_t write_index = hsa_queue_add_write_index_relaxed(queue, 1);
    auto *packet = static_cast<hsa_kernel_dispatch_packet_t *>(queue->base_address) +
                   (write_index & (queue->size - 1));
    std::memset(packet, 0, sizeof(*packet));
    packet->setup = 1;
    packet->workgroup_size_x = kBlockSize;
    packet->workgroup_size_y = 1;
    packet->workgroup_size_z = 1;
    packet->grid_size_x = kBlocks * kBlockSize;
    packet->grid_size_y = 1;
    packet->grid_size_z = 1;
    packet->group_segment_size = group_segment_size;
    packet->private_segment_size = private_segment_size;
    packet->kernel_object = kernel_object;
    packet->kernarg_address = kernarg;
    packet->completion_signal = signal;

    uint16_t header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
    header |= 1 << HSA_PACKET_HEADER_BARRIER;
    header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
    header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
    __atomic_store_n(reinterpret_cast<uint16_t *>(packet), header, __ATOMIC_RELEASE);
    hsa_signal_store_screlease(queue->doorbell_signal, write_index);

    float out = 0.0f;
    if (wait_for_dispatch(signal, 5'000'000'000ULL) == 0 &&
        synchronized_hsa_memory_copy(&out, out_dev, sizeof(out)) == HSA_STATUS_SUCCESS)
      result.sum = out;
  }

  if (queue != nullptr)
    hsa_queue_destroy(queue);
  if (signal.handle != 0)
    hsa_signal_destroy(signal);
  if (kernarg != nullptr)
    hsa_amd_memory_pool_free(kernarg);
  if (in_dev != nullptr)
    hsa_amd_memory_pool_free(in_dev);
  if (out_dev != nullptr)
    hsa_amd_memory_pool_free(out_dev);
  if (executable.handle != 0)
    hsa_executable_destroy(executable);
  if (reader.handle != 0)
    hsa_code_object_reader_destroy(reader);
  return result;
}

template <const DbiTargetParams &Target> class HsaDbiRuntimeTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    s_init_ok_ = hsa_init() == HSA_STATUS_SUCCESS;
    if (s_init_ok_)
      s_gpu_ = find_gpu_agent(Target.isa_substring);
  }

  static void TearDownTestSuite() {
    if (s_init_ok_)
      hsa_shut_down();
    s_init_ok_ = false;
    s_gpu_ = {};
  }

  void SetUp() override {
    if (!s_init_ok_)
      GTEST_SKIP() << "hsa_init failed (no HSA runtime at runtime)";
    if (s_gpu_.handle == 0)
      GTEST_SKIP() << "No " << Target.isa_substring << " agent present";
    cpu_ = find_cpu_agent();
    ASSERT_NE(cpu_.handle, 0u);
    rj_test_dbi_reset();

    Executable kernels(kernel_path(Target.kernel_fixture));
    ASSERT_TRUE(kernels.is_valid()) << "Failed to load " << Target.kernel_fixture << ".o";
    const AmdGpuCodeObject *kernel = kernels.code_object(Target.target, 0);
    ASSERT_NE(kernel, nullptr);
    const auto *image = reinterpret_cast<const uint8_t *>(kernel->image_data());
    kernel_elf_.assign(image, image + kernel->image_size());
    probe_elf_ = read_file(kernel_hsaco_path(Target.probe_fixture));
    ASSERT_FALSE(probe_elf_.empty()) << "Failed to read " << Target.probe_fixture << ".hsaco";

    // The hook's selection, run directly, so the site count checks the hook's
    // wiring rather than restating the selection rule.
    const AmdGpuCodeObject probe(probe_elf_.data(), probe_elf_.size());
    const hooks::DbiInstrumentation selection =
        hooks::instrument_each_block(kernel_elf_, hooks::DbiProbe{&probe, kProbeSymbol});
    ASSERT_EQ(selection.decision, DbiLoadDecision::Instrumented)
        << (selection.errors.empty() ? std::string{} : selection.errors.front());
    expected_sites_ = selection.sites;
  }

  void TearDown() override { rj_test_dbi_reset(); }

  // The kernel's result through the hook with @p probe set, after checking that
  // the hook saw the load and loaded the instrumented object.
  std::optional<float> run_instrumented(const std::vector<uint8_t> &probe) {
    EXPECT_TRUE(rj_test_dbi_set_probe(probe.data(), probe.size(), kProbeSymbol));
    const BlockReduceRun run = dispatch_block_reduce(kernel_elf_, s_gpu_, cpu_);
    // ROCr drops a tool whose OnLoad fails without failing hsa_init, so a
    // missing hook would otherwise look like a kernel that ran unchanged.
    EXPECT_GE(rj_test_dbi_loads(), 1u)
        << "the DBI runtime hook saw no load; HSA_TOOLS_LIB must name its testing library";
    EXPECT_EQ(decisions(DbiLoadDecision::Instrumented), 1u);
    EXPECT_EQ(rj_test_dbi_sites(), expected_sites_);
    EXPECT_EQ(run.retained_buffers, 1u) << "the instrumented object was not the one loaded";
    return run.sum;
  }

  void run_instrumented_kernel_matches_golden() {
    const std::optional<float> out = run_instrumented(probe_elf_);
    ASSERT_TRUE(out.has_value()) << "dispatch failed";
    EXPECT_EQ(*out, kGolden);
  }

  // Ends the probe at its first instruction. The entry block holds a site that
  // every wave reaches before its atomic add, so the sum stays at zero, which
  // can only happen if the instrumented object is the one that ran.
  void run_probe_that_ends_the_wave_leaves_output_unwritten() {
    std::vector<uint8_t> sabotaged = probe_elf_;
    const AmdGpuCodeObject parsed(sabotaged.data(), sabotaged.size());
    ASSERT_TRUE(parsed.is_valid());
    std::string error;
    const std::optional<ResolvedProbeSymbol> body =
        resolve_probe_symbol(parsed, kProbeSymbol, &error);
    ASSERT_TRUE(body.has_value()) << error;
    const uint32_t s_endpgm = build_s_endpgm(Target.arch);
    std::memcpy(sabotaged.data() + body->body_file_offset, &s_endpgm, sizeof(s_endpgm));

    const std::optional<float> out = run_instrumented(sabotaged);
    ASSERT_TRUE(out.has_value()) << "dispatch failed";
    EXPECT_EQ(*out, 0.0f);
  }

  static inline bool s_init_ok_ = false;
  static inline hsa_agent_t s_gpu_{};
  hsa_agent_t cpu_{};
  std::vector<uint8_t> kernel_elf_;
  std::vector<uint8_t> probe_elf_;
  uint64_t expected_sites_ = 0;
};

using HsaDbiRuntimeCdna4 = HsaDbiRuntimeTest<kCdna4>;
using HsaDbiRuntimeRdna4 = HsaDbiRuntimeTest<kRdna4>;

TEST_F(HsaDbiRuntimeCdna4, InstrumentedKernelMatchesGolden) {
  run_instrumented_kernel_matches_golden();
}
TEST_F(HsaDbiRuntimeCdna4, ProbeThatEndsTheWaveLeavesOutputUnwritten) {
  run_probe_that_ends_the_wave_leaves_output_unwritten();
}
TEST_F(HsaDbiRuntimeRdna4, InstrumentedKernelMatchesGolden) {
  run_instrumented_kernel_matches_golden();
}
TEST_F(HsaDbiRuntimeRdna4, ProbeThatEndsTheWaveLeavesOutputUnwritten) {
  run_probe_that_ends_the_wave_leaves_output_unwritten();
}

} // namespace
} // namespace rocjitsu::dbi_test
