// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hsa_dbi_nop_probe_test.cpp
/// @brief End-to-end DBI smoke for the probe-CALL path (PC01-B): patch a real
///        compiled gfx90a vector_add kernel so a chosen anchor calls the
///        amdclang++-compiled rj_nop_probe body via an s_swappc_b64 trampoline,
///        then load + dispatch the patched ELF via HSA.
///
/// This is the probe-call counterpart to hsa_dbi_nop_asm_test.cpp, which covers
/// the inline-nop path. Both share the dispatch scaffold in
/// hsa_dispatch_util.h.
///
/// Gating (see tests/CMakeLists.txt):
///   - Builds when HAS_DEVICE_KERNELS AND HAS_PROBE_FIXTURES (needs amdclang++
///     and clang-offload-bundler to produce vector_add_gfx90a.o and
///     rj_nop_probe_gfx90a.hsaco at build time).
///   - HsaDbiNopProbeStatic.*   - no GPU; registered whenever the binary builds.
///   - HsaDbiNopProbeHardware.* - registered only when HAS_CDNA2_GPU is set;
///     bodies also GTEST_SKIP at runtime if no gfx90a agent is present.

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
RJ_DIAGNOSTIC_POP

#include "../test_paths.h"
#include "hsa_dispatch_util.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/executable.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/code/patch/instrumentor.h"
#include "rocjitsu/code/patch/probe_callable.h"
#include "rocjitsu/code/patch/probe_symbol.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <gtest/gtest.h>

#include <algorithm>
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
using namespace rocjitsu::dbi_test;

namespace {

using test::kernel_hsaco_path;
using test::kernel_path;

// Decode @p text starting at @p offset and walk forward up to @p max_insts
// instructions, returning true if any decodes to @p mnemonic.
bool decodes_mnemonic_within(Decoder &decoder, std::span<const uint8_t> text, uint64_t offset,
                             std::string_view mnemonic, size_t max_insts) {
  uint64_t cur = offset;
  for (size_t i = 0; i < max_insts && cur + sizeof(uint32_t) <= text.size(); ++i) {
    rj_code_binary_inst_t word_buf[2] = {0, 0};
    const size_t avail = std::min<size_t>(sizeof(word_buf), text.size() - cur);
    std::memcpy(word_buf, text.data() + cur, avail);
    std::unique_ptr<Instruction> inst(decoder.decode(word_buf));
    if (!inst)
      return false;
    if (inst->mnemonic() == mnemonic)
      return true;
    cur += static_cast<uint64_t>(inst->size());
  }
  return false;
}

} // namespace

// Shared fixture: loads vector_add_probe_gfx90a.o and rj_nop_probe_gfx90a.hsaco,
// resolves rj_nop_probe, finds the first relocatable anchor that the probe-call
// resource policy accepts, and patches it via Instrumentor's probe-call path
// (InstrumentationPoint::probe_obj + probe_symbol). Two empty derived classes
// gate the static vs hardware cases at CMake time.
class HsaDbiNopProbeFixture : public ::testing::Test {
protected:
  void SetUp() override {
    // Load the gfx90a vector_add kernel (the instrumentation target). This is
    // the register-padded build (vector_add_probe.hip): the probe's link pair
    // s[30:31] must be granted by the kernel's SGPR allocation, which a normal
    // ~12-SGPR vector_add does not provide. Auto-growing the allocation in the
    // instrumentor is a follow-up; until then the fixture kernel reserves >=32.
    Executable kexec(kernel_path("vector_add_probe_gfx90a"));
    ASSERT_TRUE(kexec.is_valid()) << "Failed to load vector_add_probe_gfx90a.o";
    ASSERT_GT(kexec.num_code_objects(ROCJITSU_CODE_TARGET_GFX90A), 0u);
    const AmdGpuCodeObject *co = kexec.code_object(ROCJITSU_CODE_TARGET_GFX90A, 0);
    ASSERT_NE(co, nullptr);

    // Snapshot the original device ELF so we can dispatch it for comparison.
    original_elf_bytes_.assign(reinterpret_cast<const uint8_t *>(co->image_data()),
                               reinterpret_cast<const uint8_t *>(co->image_data()) +
                                   co->image_size());

    // Load the compiled rj_nop_probe device ELF (the probe to call) and resolve
    // its body so the static test can compare the copied bytes.
    Executable pexec(kernel_hsaco_path("rj_nop_probe_gfx90a"));
    ASSERT_TRUE(pexec.is_valid()) << "Failed to load rj_nop_probe_gfx90a.hsaco";
    ASSERT_GT(pexec.num_code_objects(ROCJITSU_CODE_TARGET_GFX90A), 0u);
    const AmdGpuCodeObject *probe_co = pexec.code_object(ROCJITSU_CODE_TARGET_GFX90A, 0);
    ASSERT_NE(probe_co, nullptr);

    std::string err;
    const auto resolved = resolve_probe_symbol(*probe_co, "rj_nop_probe", &err);
    ASSERT_TRUE(resolved.has_value()) << "resolve_probe_symbol(rj_nop_probe) failed: " << err;
    const auto callable =
        build_probe_callable(*probe_co, *resolved, ROCJITSU_CODE_ARCH_CDNA2, &err);
    ASSERT_TRUE(callable.has_value()) << "build_probe_callable failed: " << err;
    probe_body_words_ = callable->body_words;
    ASSERT_FALSE(probe_body_words_.empty());

    // Decode .text and collect relocatable anchors. Decode-and-search so the
    // test stays stable across compiler revisions.
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA2);
    ASSERT_NE(decoder, nullptr);
    auto blocks = BasicBlock::build(*co, *decoder, ROCJITSU_CODE_ARCH_CDNA2);
    ASSERT_FALSE(co->text_sections().empty());
    const auto *text = co->text_sections().front();
    const std::span<const uint8_t> text_bytes(reinterpret_cast<const uint8_t *>(text->data()),
                                              text->size());

    std::vector<uint64_t> candidates;
    for (const auto &block : blocks) {
      uint64_t cur = block->start_offset();
      for (const Instruction &inst : block->instructions()) {
        if (is_relocatable_anchor(inst, cur, text_bytes, ROCJITSU_CODE_ARCH_CDNA2))
          candidates.push_back(cur);
        cur += static_cast<uint64_t>(inst.size());
      }
    }
    ASSERT_FALSE(candidates.empty()) << "No relocatable anchor in vector_add_probe_gfx90a.o; "
                                        "did the compiler change the lowering?";

    // Pick the first anchor whose probe-call patch the resource/spill policy
    // accepts. A live fixed link pair s[30:31] or an exhausted dead-pair pool
    // would make a given anchor fail closed; trying several keeps the smoke
    // test deterministic without hard-coding a fragile offset.
    for (uint64_t off : candidates) {
      Instrumentor instr(*co, ROCJITSU_CODE_ARCH_CDNA2);
      InstrumentationPoint pt;
      pt.anchor_offset = off;
      pt.probe_obj = probe_co;
      pt.probe_symbol = "rj_nop_probe";
      instr.add_point(pt);
      auto result = instr.patch_with_debug_summaries();
      if (result.errors.empty()) {
        anchor_offset_ = off;
        patched_elf_bytes_ = std::move(result.elf_bytes);
        patches_ = std::move(result.patches);
        break;
      }
      last_patch_error_ = result.errors.front();
    }
    ASSERT_FALSE(patched_elf_bytes_.empty())
        << "No relocatable anchor passed the probe-call resource policy. Last error: "
        << last_patch_error_;
    ASSERT_EQ(patches_.size(), 1u);
    ASSERT_TRUE(patches_[0].is_probe_call);
  }

  std::vector<uint8_t> original_elf_bytes_;
  std::vector<uint8_t> patched_elf_bytes_;
  std::vector<uint32_t> probe_body_words_;
  std::vector<InstrumentationPatch> patches_;
  uint64_t anchor_offset_ = 0;
  std::string last_patch_error_;
};

// Static tests: parse + decode the patched ELF. No GPU or HSA runtime needed.
class HsaDbiNopProbeStatic : public HsaDbiNopProbeFixture {};

// Hardware tests: load + dispatch on a real gfx90a GPU. Gated at CMake by
// HAS_CDNA2_GPU; bodies also GTEST_SKIP if no agent is present. hsa_init /
// hsa_shut_down run once per suite; the gfx90a agent is cached.
class HsaDbiNopProbeHardware : public HsaDbiNopProbeFixture {
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

// Static verification: prove the probe-call patch actually rewrote the kernel,
// copied the probe body in, and wired an s_swappc_b64 call to it. Catches:
//   - patcher silently produced the original bytes
//   - the anchor was not redirected to a branch stub
//   - the copied probe body is missing or differs from rj_nop_probe
//   - the trampoline does not contain the call to the probe
TEST_F(HsaDbiNopProbeStatic, PatchedElfContainsProbeCallInstrumentation) {
  // (a) Patcher produced different bytes from the original.
  ASSERT_NE(patched_elf_bytes_, original_elf_bytes_)
      << "Patched ELF is byte-identical to original - patcher silently no-oped?";

  // (b) The patched ELF still parses and .text grew to hold the copied probe
  //     body plus the trampoline cave.
  AmdGpuCodeObject patched(patched_elf_bytes_.data(), patched_elf_bytes_.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_FALSE(patched.text_sections().empty());
  const Section *text = patched.text_sections().front();
  const std::span<const uint8_t> text_bytes(reinterpret_cast<const uint8_t *>(text->data()),
                                            text->size());

  AmdGpuCodeObject original(original_elf_bytes_.data(), original_elf_bytes_.size());
  ASSERT_TRUE(original.is_valid());
  ASSERT_FALSE(original.text_sections().empty());
  EXPECT_GT(text->size(), original.text_sections().front()->size())
      << ".text must grow to hold the copied probe body and trampoline cave";

  // (c) The anchor now decodes as an s_branch stub (redirected to the cave).
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA2);
  ASSERT_NE(decoder, nullptr);
  ASSERT_GT(text->size(), anchor_offset_ + sizeof(uint32_t));
  rj_code_binary_inst_t anchor_word = 0;
  std::memcpy(&anchor_word, text->data() + anchor_offset_, sizeof(anchor_word));
  std::unique_ptr<Instruction> anchor_inst(decoder->decode(&anchor_word));
  ASSERT_NE(anchor_inst, nullptr);
  EXPECT_NE(anchor_inst->mnemonic().find("s_branch"), std::string_view::npos)
      << "Anchor at offset " << anchor_offset_ << " should decode as s_branch; got "
      << anchor_inst->mnemonic();

  // (d) The per-site summary identifies a probe call to rj_nop_probe.
  const InstrumentationPatch &patch = patches_[0];
  EXPECT_TRUE(patch.is_probe_call);
  EXPECT_EQ(patch.probe_symbol, "rj_nop_probe");

  // (e) The copied probe body sits at probe_target_offset and is byte-identical
  //     to the resolved rj_nop_probe body, ending in s_setpc_b64.
  const uint64_t body_off = patch.probe_target_offset;
  const size_t body_bytes = probe_body_words_.size() * sizeof(uint32_t);
  ASSERT_GE(text->size(), body_off + body_bytes);
  EXPECT_EQ(std::memcmp(text->data() + body_off, probe_body_words_.data(), body_bytes), 0)
      << "Copied probe body at probe_target_offset differs from rj_nop_probe";
  EXPECT_TRUE(decodes_mnemonic_within(*decoder, text_bytes, body_off, "s_setpc_b64",
                                      probe_body_words_.size()))
      << "Copied probe body should return via s_setpc_b64";

  // (f) The trampoline contains the s_swappc_b64 call into the probe. The
  //     call sits within the before-region envelope at the head of the
  //     trampoline; scan a small bounded window from trampoline_offset.
  EXPECT_TRUE(decodes_mnemonic_within(*decoder, text_bytes, patch.trampoline_offset, "s_swappc_b64",
                                      /*max_insts=*/12))
      << "Trampoline at offset " << patch.trampoline_offset
      << " should contain an s_swappc_b64 call to the probe";
}

// Load the patched ELF into an HSA executable and validate. No dispatch.
// Gated on a real gfx90a agent because hsa_executable_load_agent_code_object
// requires an agent whose ISA matches the code object.
TEST_F(HsaDbiNopProbeHardware, PatchedElfLoadsAndValidatesInHsaExecutable) {
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

  hsa_executable_symbol_t symbol{};
  EXPECT_EQ(hsa_executable_get_symbol_by_name(executable, "vector_add.kd", &gpu, &symbol),
            HSA_STATUS_SUCCESS)
      << "kernel symbol vector_add.kd missing after patching";

  hsa_executable_destroy(executable);
  hsa_code_object_reader_destroy(reader);
}

// Dispatch the original and probe-call-patched kernels with identical inputs
// and confirm bit-identical outputs. rj_nop_probe is semantically a no-op, so
// the patched kernel must produce the same buffer as the original. This is a
// regression check; it does NOT prove the probe dynamically executed (see the
// sabotage test below for that).
TEST_F(HsaDbiNopProbeHardware, PatchedKernelDispatchMatchesOriginal) {
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

  // The C buffer is zeroed before each dispatch, so a non-zero result proves
  // the kernel actually ran and wrote into it.
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
  // fails the fixture is bad, not the instrumentation.
  auto orig_out = dispatch_vector_add(original_elf_bytes_, gpu, cpu, a, b, N);
  ASSERT_EQ(orig_out.size(), N) << "original dispatch failed (empty result)";
  assert_kernel_wrote_output(orig_out, "original");
  int orig_mismatches = 0;
  for (uint32_t i = 0; i < N; ++i) {
    if (std::abs(orig_out[i] - golden[i]) > 1e-5f)
      ++orig_mismatches;
  }
  ASSERT_EQ(orig_mismatches, 0) << "original vector_add dispatch produced " << orig_mismatches
                                << "/" << N << " mismatches vs CPU golden (fixture problem)";

  // The real check: probe-call-patched dispatch must match the original.
  auto patched_out = dispatch_vector_add(patched_elf_bytes_, gpu, cpu, a, b, N);
  ASSERT_EQ(patched_out.size(), N) << "patched dispatch failed (empty result)";
  assert_kernel_wrote_output(patched_out, "patched");
  EXPECT_EQ(patched_out, orig_out)
      << "probe-call patched kernel output differs from original — calling the "
         "no-op probe should be semantically transparent";
}

// "Sabotage" verification: overwrite the first word of the COPIED probe body
// with s_endpgm. If the s_swappc_b64 in the trampoline genuinely transfers
// control into the copied body, every wave hits s_endpgm and terminates before
// returning to the relocated instruction and the store-to-C, so C stays at the
// pre-dispatch zero pattern. If the call were somehow bypassed, the kernel
// would run end-to-end and produce the golden output.
//
// This is the test that proves "the GPU actually calls the probe" — the
// equivalence test above only proves the patched program is semantically
// unchanged.
TEST_F(HsaDbiNopProbeHardware, ProbeBodyIsActuallyCalledByGpu) {
  if (!s_init_ok_)
    GTEST_SKIP() << "hsa_init failed";
  if (s_gpu_.handle == 0)
    GTEST_SKIP() << "No gfx90a agent present";
  hsa_agent_t gpu = s_gpu_;
  hsa_agent_t cpu = find_cpu_agent();
  ASSERT_NE(cpu.handle, 0u);

  std::vector<uint8_t> sabotaged = patched_elf_bytes_;
  AmdGpuCodeObject parsed(sabotaged.data(), sabotaged.size());
  ASSERT_TRUE(parsed.is_valid());
  ASSERT_FALSE(parsed.text_sections().empty());
  const Section *text = parsed.text_sections().front();

  const uint64_t body_off = patches_[0].probe_target_offset;
  ASSERT_GE(text->size(), body_off + sizeof(uint32_t));
  const uint64_t body_file_off = text->sectionOffset() + body_off;

  // The first word of the copied body must not already be s_endpgm; otherwise
  // the sabotage proves nothing.
  constexpr uint32_t kSEndpgm0 = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA2);
  uint32_t pre_overwrite = 0;
  std::memcpy(&pre_overwrite, sabotaged.data() + body_file_off, sizeof(pre_overwrite));
  ASSERT_NE(pre_overwrite, kSEndpgm0) << "Probe body already starts with s_endpgm?";
  std::memcpy(sabotaged.data() + body_file_off, &kSEndpgm0, sizeof(kSEndpgm0));

  constexpr uint32_t N = 1024;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
  std::vector<float> a(N), b(N), golden(N);
  for (uint32_t i = 0; i < N; ++i) {
    a[i] = dist(rng);
    b[i] = dist(rng);
    golden[i] = a[i] + b[i];
  }

  auto sabotaged_out = dispatch_vector_add(sabotaged, gpu, cpu, a, b, N);
  ASSERT_EQ(sabotaged_out.size(), N)
      << "sabotaged dispatch failed (HSA error before s_endpgm could run)";

  // Probe-called path: every wave enters the body, hits s_endpgm, and
  // terminates before the store-to-C, so C stays zero.
  bool any_nonzero = false;
  for (float v : sabotaged_out) {
    if (v != 0.0f) {
      any_nonzero = true;
      break;
    }
  }
  EXPECT_FALSE(any_nonzero)
      << "sabotaged dispatch produced non-zero output — did the GPU bypass the probe call?";

  int matches_golden = 0;
  for (uint32_t i = 0; i < N; ++i) {
    if (std::abs(sabotaged_out[i] - golden[i]) < 1e-5f)
      ++matches_golden;
  }
  EXPECT_LT(matches_golden, N) << "sabotaged dispatch matched the golden in " << matches_golden
                               << "/" << N << " elements — probe call appears bypassed";

  // Revert: restore the original first body word and confirm the kernel is
  // back to producing the golden output (proves the sabotage, not some
  // unrelated corruption, caused the zero result).
  std::memcpy(sabotaged.data() + body_file_off, &pre_overwrite, sizeof(pre_overwrite));
  auto reverted_out = dispatch_vector_add(sabotaged, gpu, cpu, a, b, N);
  ASSERT_EQ(reverted_out.size(), N) << "HSA error before reverted run could finish";
  for (uint32_t i = 0; i < N; ++i) {
    ASSERT_LT(std::abs(reverted_out[i] - golden[i]), 1e-5f)
        << "reverted (un-sabotaged) probe-call code differs from golden at " << i;
  }
}

#endif // HAS_HOST_AMDGPU
