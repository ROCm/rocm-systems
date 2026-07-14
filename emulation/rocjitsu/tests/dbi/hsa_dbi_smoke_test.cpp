// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hsa_dbi_smoke_test.cpp
/// @brief End-to-end DBI smoke: patch a real compiled gfx90a kernel with
///        Instrumentor::patch, then load and eventually dispatch the patched
///        ELF via HSA.

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
RJ_DIAGNOSTIC_POP

#include "../test_paths.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/executable.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/code/patch/instrumentor.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#ifdef HAS_HOST_AMDGPU
using namespace rocjitsu;

namespace {

using test::kernel_path;

// Find a GPU agent whose ISA name contains "gfx90a". Returns {handle=0} if
// no such agent is present.
hsa_agent_t find_gfx90a_agent() {
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

hsa_agent_t find_cpu_agent() {
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

// Dispatch the vector_add kernel from @p elf_bytes and return the output
// buffer. Returns an empty vector on any HSA failure (test will fail with an
// assertion in the caller). Mirrors the dispatch scaffold in
// tests/dbt/hsa_translate_test.cpp:186-298.
std::vector<float> dispatch_vector_add(std::span<const uint8_t> elf_bytes, hsa_agent_t gpu,
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

} // namespace

// Shared fixture: loads vector_add_gfx90a.o, decodes .text, finds the first
// relocatable v_add_f32 anchor, and patches it via Instrumentor. Two empty
// derived classes (HsaDbiSmokeStatic / HsaDbiSmokeHardware) inherit this so
// CMake can register one CTest entry per gating policy:
//   - HsaDbiSmokeStatic.*   - no GPU needed, registered unconditionally
//   - HsaDbiSmokeHardware.* - registered only when HAS_CDNA2_GPU is set
class HsaDbiSmokeFixture : public ::testing::Test {
protected:
  // Load a vector_add_gfx90a.o device ELF and instrument a single instruction
  // with the inlined nop functionality currently available in Instrumentor
  void SetUp() override {
    // Get gfx90a build of vector_add.
    Executable exec(kernel_path("vector_add_gfx90a"));
    ASSERT_TRUE(exec.is_valid()) << "Failed to load vector_add_gfx90a.o";
    ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX90A), 0u);
    const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX90A, 0);
    ASSERT_NE(co, nullptr);

    // Snapshot the original device ELF so we can dispatch it too
    original_elf_bytes_.assign(reinterpret_cast<const uint8_t *>(co->image_data()),
                               reinterpret_cast<const uint8_t *>(co->image_data()) +
                                   co->image_size());

    // Decode .text and find the first v_add_f32-mnemonic anchor that the
    // trampoline machinery considers relocatable. Decode-and-search so the
    // test is stable across compiler revisions.
    // TODO: instrument multiple instructions
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA2);
    ASSERT_NE(decoder, nullptr);
    auto blocks = BasicBlock::build(*co, *decoder, ROCJITSU_CODE_ARCH_CDNA2);

    ASSERT_FALSE(co->text_sections().empty());
    const auto *text = co->text_sections().front();
    const std::span<const uint8_t> text_bytes(reinterpret_cast<const uint8_t *>(text->data()),
                                              text->size());

    for (const auto &block : blocks) {
      uint64_t cur = block->start_offset();
      for (const Instruction &inst : block->instructions()) {
        if (is_relocatable_anchor(inst, cur, text_bytes, ROCJITSU_CODE_ARCH_CDNA2)) {
          anchor_offsets_.push_back(cur); // Instrumentor will need offset
          anchor_mnemonics_.push_back(std::string(inst.mnemonic()));
          ++anchor_count_;
        }
        cur += static_cast<uint64_t>(inst.size());
        if (anchor_count_ == 2)
          break;
      }
      if (anchor_count_ == 2)
        break;
    }
    ASSERT_NE(anchor_count_, 0u) << "No relocatable anchor in vector_add_gfx90a.o; "
                                    "did the compiler change the lowering?";

    // Apply the inline-nop trampoline.
    Instrumentor instrumentor(*co, ROCJITSU_CODE_ARCH_CDNA2);
    for (uint64_t anchor_idx = 0; anchor_idx < anchor_count_; ++anchor_idx) {
      instrumentor.add_point_by_offset(anchor_offsets_[anchor_idx]);
    }
    auto result = instrumentor.patch_with_debug_summaries();
    ASSERT_TRUE(result.errors.empty())
        << "Instrumentor::patch failed: "
        << (result.errors.empty() ? std::string{} : result.errors.front());
    patched_elf_bytes_ = std::move(result.elf_bytes);
    ASSERT_FALSE(patched_elf_bytes_.empty());
    // Keep the per-site summaries so tests can locate each trampoline by its
    // real cave offset rather than guessing the in-section stride.
    patches_ = std::move(result.patches);
  }

  std::vector<uint8_t> original_elf_bytes_;
  std::vector<uint8_t> patched_elf_bytes_;
  std::vector<uint64_t> anchor_offsets_;
  std::vector<std::string> anchor_mnemonics_;
  std::vector<InstrumentationPatch> patches_;
  uint64_t anchor_count_ = 0;
};

// Tests that need only HSA libs (parsing + decoding the patched ELF). Run
// anywhere the binary is built.
class HsaDbiSmokeStatic : public HsaDbiSmokeFixture {};

// Tests that need to load + dispatch on a real gfx90a GPU. Gated at CMake
// time by HAS_CDNA2_GPU; bodies also GTEST_SKIP at runtime if no agent.
//
// hsa_init / hsa_shut_down run once per suite (HSA tolerates per-test
// init/shutdown but it isn't free). The gfx90a agent is enumerated once
// in SetUpTestSuite and cached. Bodies pull the cached agent and skip if
// initialization or enumeration didn't succeed.
class HsaDbiSmokeHardware : public HsaDbiSmokeFixture {
protected:
  static void SetUpTestSuite() {
    s_init_ok_ = (hsa_init() == HSA_STATUS_SUCCESS);
    if (s_init_ok_)
      s_gpu_ = find_gfx90a_agent();
  }
  static void TearDownTestSuite() {
    if (s_init_ok_)
      hsa_shut_down();
    s_init_ok_ = false;
    s_gpu_ = {};
  }

  static inline bool s_init_ok_ = false;
  static inline hsa_agent_t s_gpu_{};
};

// Static verification: prove the patcher actually changed the kernel before
// any HSA / dispatch tests run. No GPU or HSA runtime required. Catches
// failure modes that the byte-equality dispatch check cannot:
//   - patcher silently produced the original bytes (e.g. Instrumentor::patch
//     short-circuited without applying the patch)
//   - patch landed at a different offset than anchor_offset_ records
//   - the trampoline cave was not appended to .text
TEST_F(HsaDbiSmokeStatic, PatchedElfActuallyContainsInstrumentation) {
  // (a) Patcher produced different bytes from the original.
  ASSERT_NE(patched_elf_bytes_, original_elf_bytes_)
      << "Patched ELF is byte-identical to original - patcher silently no-oped?";

  // (b) The patched .text at anchor_offsets_[idx] now decodes as s_branch, not the
  //     original instructions we recorded as anchor_mnemonics_[idx].
  AmdGpuCodeObject patched(patched_elf_bytes_.data(), patched_elf_bytes_.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_FALSE(patched.text_sections().empty());
  const Section *text = patched.text_sections().front();
  ASSERT_GT(text->size(), anchor_offsets_[0] + 4);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA2);
  ASSERT_NE(decoder, nullptr);
  for (uint64_t anchor_idx = 0; anchor_idx < anchor_count_; ++anchor_idx) {
    rj_code_binary_inst_t anchor_word = 0;
    std::memcpy(&anchor_word, text->data() + anchor_offsets_[anchor_idx], sizeof(anchor_word));
    std::unique_ptr<Instruction> decoded(decoder->decode(&anchor_word));
    ASSERT_NE(decoded, nullptr);
    EXPECT_NE(decoded->mnemonic().find("s_branch"), std::string_view::npos)
        << "Anchor at offset " << anchor_offsets_[anchor_idx]
        << " should now decode as s_branch; got: " << decoded->mnemonic();
    EXPECT_NE(decoded->mnemonic(), anchor_mnemonics_[anchor_idx])
        << "Anchor mnemonic unchanged (" << anchor_mnemonics_[anchor_idx]
        << ") - was the patch applied?";
  }

  // (c) The trampoline cave was appended to .text
  AmdGpuCodeObject original(original_elf_bytes_.data(), original_elf_bytes_.size());
  ASSERT_TRUE(original.is_valid());
  ASSERT_FALSE(original.text_sections().empty());
  EXPECT_GT(text->size(), original.text_sections().front()->size())
      << ".text must grow to hold the appended trampoline cave";
}

// Load patched ELF into an HSA executable and validate. No dispatch.
// Gates on a real gfx90a agent because hsa_executable_load_agent_code_object
// requires an agent whose ISA matches the code object.
TEST_F(HsaDbiSmokeHardware, PatchedElfLoadsAndValidatesInHsaExecutable) {
  if (!s_init_ok_)
    GTEST_SKIP() << "hsa_init failed (no HSA runtime at runtime)";
  if (s_gpu_.handle == 0)
    GTEST_SKIP() << "No gfx90a agent present";
  hsa_agent_t gpu = s_gpu_;

  hsa_code_object_reader_t reader{};
  ASSERT_EQ(hsa_code_object_reader_create_from_memory(patched_elf_bytes_.data(),
                                                      patched_elf_bytes_.size(), &reader),
            HSA_STATUS_SUCCESS)
      << "Patched ELF rejected by hsa_code_object_reader_create_from_memory";

  hsa_executable_t executable{};
  ASSERT_EQ(hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                      nullptr, &executable),
            HSA_STATUS_SUCCESS);

  ASSERT_EQ(hsa_executable_load_agent_code_object(executable, gpu, reader, nullptr, nullptr),
            HSA_STATUS_SUCCESS)
      << "hsa_executable_load_agent_code_object rejected the patched ELF";

  ASSERT_EQ(hsa_executable_freeze(executable, nullptr), HSA_STATUS_SUCCESS);

  uint32_t validate_result = 0;
  ASSERT_EQ(hsa_executable_validate(executable, &validate_result), HSA_STATUS_SUCCESS);
  EXPECT_EQ(validate_result, 0u) << "hsa_executable_validate reported error: " << validate_result;

  // Sanity: the kernel symbol is still findable post-patch.
  hsa_executable_symbol_t symbol{};
  EXPECT_EQ(hsa_executable_get_symbol_by_name(executable, "vector_add.kd", &gpu, &symbol),
            HSA_STATUS_SUCCESS)
      << "kernel symbol vector_add.kd missing after patching";

  hsa_executable_destroy(executable);
  hsa_code_object_reader_destroy(reader);
}

// Dispatch both the original and patched kernels with identical
// inputs and confirm bit-identical outputs. The inline-nop placeholder is
// semantically a no-op, so the patched kernel must produce the same buffer
// as the original — anything else means the patch path corrupted execution.
TEST_F(HsaDbiSmokeHardware, PatchedKernelDispatchMatchesOriginal) {
  if (!s_init_ok_)
    GTEST_SKIP() << "hsa_init failed";
  if (s_gpu_.handle == 0)
    GTEST_SKIP() << "No gfx90a agent present";
  hsa_agent_t gpu = s_gpu_;
  hsa_agent_t cpu = find_cpu_agent();
  ASSERT_NE(cpu.handle, 0u) << "No CPU agent found";

  constexpr uint32_t N = 1024;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
  std::vector<float> a(N), b(N), golden(N);
  for (uint32_t i = 0; i < N; ++i) {
    a[i] = dist(rng);
    b[i] = dist(rng);
    golden[i] = a[i] + b[i];
  }

  // Helper: assert the output buffer is non-zero somewhere. The C buffer is
  // zeroed before each dispatch (see dispatch_vector_add), so a non-zero
  // result proves the kernel actually executed and wrote into it \xE2\x80\x94 not just
  // that the surrounding plumbing succeeded.
  auto assert_kernel_wrote_output = [](const std::vector<float> &out, const char *label) {
    bool wrote = false;
    for (float v : out) {
      if (v != 0.0f) {
        wrote = true;
        break;
      }
    }
    ASSERT_TRUE(wrote) << label << " dispatch left output buffer all zeros (kernel didn't run?)";
  };

  // Sanity: original (unpatched) dispatch matches the CPU golden. If this
  // fails the test fixture is bad, not the instrumentation.
  auto orig_out = dispatch_vector_add(original_elf_bytes_, gpu, cpu, a, b, N);
  ASSERT_EQ(orig_out.size(), N) << "original dispatch failed (empty result)";
  assert_kernel_wrote_output(orig_out, "original");
  int orig_mismatches = 0;
  for (uint32_t i = 0; i < N; ++i) {
    if (std::abs(orig_out[i] - golden[i]) > 1e-5f)
      ++orig_mismatches;
  }
  ASSERT_EQ(orig_mismatches, 0) << "original vector_add dispatch produced " << orig_mismatches
                                << "/" << N
                                << " mismatches against CPU golden (test fixture problem)";

  // The real check: patched dispatch must produce the same buffer as the
  // original. Bit-identical because the trampoline body is just s_nop 0
  // around the relocated instruction.
  auto patched_out = dispatch_vector_add(patched_elf_bytes_, gpu, cpu, a, b, N);
  ASSERT_EQ(patched_out.size(), N) << "patched dispatch failed (empty result)";
  assert_kernel_wrote_output(patched_out, "patched");
  EXPECT_EQ(patched_out, orig_out)
      << "patched kernel output differs from original — the inline-nop placeholder "
         "should be semantically a no-op";
}

// "Sabotage" verification: overwrite the s_nop 0 placeholders in the patched
// trampolines with s_endpgm one at a time. If the GPU genuinely takes the trampoline
// path, every wave terminates before reaching the relocated instruction and
// the output stays at the pre-dispatch zero pattern. If that trampoline is
// somehow bypassed (e.g., the forward s_branch didn't take effect), the
// kernel would still produce the golden output.
//
// This is the only test that proves "the trampoline executes on the GPU" -
// the other tests are statically verifiable (correct bytes, correct ELF
// structure, semantically-equivalent dispatch output).
TEST_F(HsaDbiSmokeHardware, TrampolineIsActuallyExecutedByGpu) {
  if (!s_init_ok_)
    GTEST_SKIP() << "hsa_init failed";
  if (s_gpu_.handle == 0)
    GTEST_SKIP() << "No gfx90a agent present";
  hsa_agent_t gpu = s_gpu_;
  hsa_agent_t cpu = find_cpu_agent();
  ASSERT_NE(cpu.handle, 0u);

  // The trampolines live inside .text at .text-relative offset
  // patches_[i].trampoline_offset. Overwrite the first 4 bytes of each
  // trampoline (the s_nop 0 placeholder) with s_endpgm.
  std::vector<uint8_t> sabotaged = patched_elf_bytes_;
  AmdGpuCodeObject parsed(sabotaged.data(), sabotaged.size());
  ASSERT_TRUE(parsed.is_valid());
  ASSERT_FALSE(parsed.text_sections().empty());
  const Section *text = parsed.text_sections().front();

  ASSERT_EQ(anchor_count_, 2u);
  ASSERT_EQ(patches_.size(), 2u);
  ASSERT_GE(text->size(), patches_[1].trampoline_offset + 4);
  // File offset of the first trampoline's placeholder word.
  const uint64_t tramp0_file_off = text->sectionOffset() + patches_[0].trampoline_offset;
  // Before sabotaging: verify the bytes we're about to overwrite are indeed
  // s_nop 0. If this assertion fails, the orchestrator's trampoline layout
  // no longer starts with the placeholder we think it does, and the
  // sabotage premise ("we replaced the no-op with s_endpgm") would be a lie.
  constexpr uint32_t kSNop0 = build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA2);
  uint32_t pre_overwrite = 0;
  std::memcpy(&pre_overwrite, sabotaged.data() + tramp0_file_off, sizeof(pre_overwrite));
  ASSERT_EQ(pre_overwrite, kSNop0) << "Expected s_nop 0 (0x" << std::hex << kSNop0
                                   << ") at start of the trampoline cave but found 0x"
                                   << pre_overwrite << " - trampoline body layout changed?";

  constexpr uint32_t kSEndpgm0 = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA2);
  std::memcpy(sabotaged.data() + tramp0_file_off, &kSEndpgm0, sizeof(kSEndpgm0));

  // Same inputs as the dispatch test so we can compare against its golden.
  constexpr uint32_t N = 1024;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
  std::vector<float> a(N), b(N), golden(N);
  for (uint32_t i = 0; i < N; ++i) {
    a[i] = dist(rng);
    b[i] = dist(rng);
    golden[i] = a[i] + b[i];
  }

  auto sabotaged_out_1 = dispatch_vector_add(sabotaged, gpu, cpu, a, b, N);
  ASSERT_EQ(sabotaged_out_1.size(), N)
      << "Sabotaged dispatch failed (HSA error before s_endpgm could run)";

  // Trampoline-executed path: every thread that enters the trampoline hits
  // s_endpgm and terminates before reaching the relocated instruction or its
  // store-to-C. So C stays at the pre-dispatch zero pattern.
  bool any_nonzero_1 = false;
  for (float v : sabotaged_out_1) {
    if (v != 0.0f) {
      any_nonzero_1 = true;
      break;
    }
  }
  EXPECT_FALSE(any_nonzero_1)
      << "Sabotaged dispatch produced non-zero output - did the GPU bypass the trampoline?";

  // And the output must NOT match the golden (would mean the trampoline
  // wasn't hit and the kernel ran end-to-end normally).
  int matches_golden_1 = 0;
  for (uint32_t i = 0; i < N; ++i) {
    if (std::abs(sabotaged_out_1[i] - golden[i]) < 1e-5f)
      ++matches_golden_1;
  }
  EXPECT_LT(matches_golden_1, N) << "Sabotaged dispatch matched the golden in " << matches_golden_1
                                 << "/" << N << " elements - trampoline appears bypassed";

  // Revert first trampoline in sabotaged
  std::memcpy(sabotaged.data() + tramp0_file_off, &kSNop0, sizeof(kSNop0));
  auto unsabotaged_out = dispatch_vector_add(sabotaged, gpu, cpu, a, b, N);
  ASSERT_EQ(unsabotaged_out.size(), N) << "HSA error before unsabotaged run could finish";

  for (uint32_t i = 0; i < N; ++i) {
    ASSERT_LT(std::abs(unsabotaged_out[i] - golden[i]), 1e-5f)
        << "Unsabotaged code differs from golden";
  }

  // Perform same change and test for second trampoline
  int64_t offset_between_anchors = patches_[1].trampoline_offset - patches_[0].trampoline_offset;
  EXPECT_NE(offset_between_anchors, 0)
      << "Both selected trampolines have the same trampoline offset";
  std::memcpy(&pre_overwrite, sabotaged.data() + tramp0_file_off + offset_between_anchors,
              sizeof(pre_overwrite));
  ASSERT_EQ(pre_overwrite, kSNop0) << "Expected s_nop 0 (0x" << std::hex << kSNop0
                                   << ") at start of the trampoline cave but found 0x"
                                   << pre_overwrite << " - trampoline body layout changed?";

  std::memcpy(sabotaged.data() + tramp0_file_off + offset_between_anchors, &kSEndpgm0,
              sizeof(kSEndpgm0));

  auto sabotaged_out_2 = dispatch_vector_add(sabotaged, gpu, cpu, a, b, N);
  ASSERT_EQ(sabotaged_out_2.size(), N)
      << "Sabotaged dispatch failed (HSA error before s_endpgm could run)";

  // Trampoline-executed path: every thread that enters the trampoline hits
  // s_endpgm and terminates before reaching the relocated v_add_f32 or its
  // store-to-C. So C stays at the pre-dispatch zero pattern.
  bool any_nonzero_2 = false;
  for (float v : sabotaged_out_2) {
    if (v != 0.0f) {
      any_nonzero_2 = true;
      break;
    }
  }
  EXPECT_FALSE(any_nonzero_2)
      << "Sabotaged dispatch produced non-zero output - did the GPU bypass the trampoline?";

  // And the output must NOT match the golden (would mean the trampoline
  // wasn't hit and the kernel ran end-to-end normally).
  int matches_golden_2 = 0;
  for (uint32_t i = 0; i < N; ++i) {
    if (std::abs(sabotaged_out_2[i] - golden[i]) < 1e-5f)
      ++matches_golden_2;
  }
  EXPECT_LT(matches_golden_2, N) << "Sabotaged dispatch matched the golden in " << matches_golden_2
                                 << "/" << N << " elements - trampoline appears bypassed";
}

#endif // HAS_HOST_AMDGPU
