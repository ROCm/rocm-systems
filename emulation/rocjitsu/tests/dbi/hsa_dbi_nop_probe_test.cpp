// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hsa_dbi_nop_probe_test.cpp
/// @brief End-to-end DBI smoke for the probe-CALL path (PC01-B): patch a real
///        compiled vector_add kernel so a chosen anchor calls the
///        amdclang++-compiled rj_nop_probe body via an s_swappc_b64 trampoline,
///        then load + dispatch the patched ELF via HSA.
///
/// This is the probe-call counterpart to hsa_dbi_nop_asm_test.cpp, which covers
/// the inline-nop path. Both share dbi_test_fixtures.h: the DbiTargetParams
/// parameterization, the per-target HSA setup, and the two hardware bodies that
/// are identical between the two paths. The concrete suites at the bottom of
/// this file are what CMake registers.
///
/// Gating (see tests/CMakeLists.txt):
///   - Builds when HAS_DEVICE_KERNELS AND HAS_PROBE_FIXTURES (needs amdclang++
///     and clang-offload-bundler to produce the per-target vector_add_probe_*.o
///     and rj_nop_probe_*.hsaco at build time).
///   - HsaDbiNopProbe<Target>Static.*   - no GPU; registered whenever the binary
///     builds.
///   - HsaDbiNopProbe<Target>Hardware.* - registered as a Sim case
///     unconditionally, and bare only when that target's HAS_*_GPU is set;
///     DbiHardwareBase::SetUp() also skips at runtime if no matching agent is
///     present.

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
RJ_DIAGNOSTIC_POP

#include "../test_paths.h"
#include "dbi_test_fixtures.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/executable.h"
#include "rocjitsu/code/patch/instrumentor.h"
#include "rocjitsu/code/patch/probe_callable.h"
#include "rocjitsu/code/patch/probe_symbol.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
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
    auto decoded = decoder.decode(word_buf);
    if (decoded.failed())
      return false;
    std::unique_ptr<Instruction> inst = std::move(decoded).value();
    if (inst->mnemonic() == mnemonic)
      return true;
    cur += static_cast<uint64_t>(inst->size());
  }
  return false;
}

constexpr DbiTargetParams kCdna2Params{ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_TARGET_GFX90A,
                                       "vector_add_probe_gfx90a", "rj_nop_probe_gfx90a", "gfx90a"};
constexpr DbiTargetParams kCdna4Params{ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_TARGET_GFX950,
                                       "vector_add_probe_gfx950", "rj_nop_probe_gfx950", "gfx950"};
constexpr DbiTargetParams kRdna4Params{ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_TARGET_GFX1201,
                                       "vector_add_probe_gfx1201", "rj_nop_probe_gfx1201",
                                       "gfx1201"};

} // namespace

// Shared fixture: loads the target's vector_add_probe and rj_nop_probe builds,
// resolves rj_nop_probe, finds the first relocatable anchor that the probe-call
// resource policy accepts, and patches it via Instrumentor's probe-call path
// (InstrumentationPoint::probe_obj + probe_symbol). The concrete suites at the
// bottom of this file bind a DbiTargetParams and gate static vs hardware.
class HsaDbiNopProbeFixture : public ::testing::Test {
protected:
  // Names this test's instrumentation mechanism in the shared dispatch-
  // equivalence failure message (see DbiHardwareBase).
  static constexpr const char *kPatchDescription = "calling the no-op probe";

  explicit HsaDbiNopProbeFixture(const DbiTargetParams &params) : params_(params) {}

  void SetUp() override {
    // Load the target's vector_add kernel (the instrumentation target). This is
    // the register-padded build (vector_add_probe.hip): the probe's link pair
    // s[30:31] must be granted by the kernel's SGPR allocation, which a normal
    // ~12-SGPR vector_add does not provide. Auto-growing the allocation in the
    // instrumentor is a follow-up; until then the fixture kernel reserves >=32.
    Executable kexec(kernel_path(params_.kernel_fixture));
    ASSERT_TRUE(kexec.is_valid()) << "Failed to load " << params_.kernel_fixture << ".o";
    ASSERT_GT(kexec.num_code_objects(params_.target), 0u);
    const AmdGpuCodeObject *co = kexec.code_object(params_.target, 0);
    ASSERT_NE(co, nullptr);

    // Snapshot the original device ELF so we can dispatch it for comparison.
    original_elf_bytes_.assign(reinterpret_cast<const uint8_t *>(co->image_data()),
                               reinterpret_cast<const uint8_t *>(co->image_data()) +
                                   co->image_size());

    // Load the compiled rj_nop_probe device ELF (the probe to call) and resolve
    // its body so the static test can compare the copied bytes.
    Executable pexec(kernel_hsaco_path(params_.probe_fixture));
    ASSERT_TRUE(pexec.is_valid()) << "Failed to load " << params_.probe_fixture << ".hsaco";
    ASSERT_GT(pexec.num_code_objects(params_.target), 0u);
    const AmdGpuCodeObject *probe_co = pexec.code_object(params_.target, 0);
    ASSERT_NE(probe_co, nullptr);

    std::string err;
    const auto resolved = resolve_probe_symbol(*probe_co, "rj_nop_probe", &err);
    ASSERT_TRUE(resolved.has_value()) << "resolve_probe_symbol(rj_nop_probe) failed: " << err;
    const auto callable = build_probe_callable(*probe_co, *resolved, params_.arch, &err);
    ASSERT_TRUE(callable.has_value()) << "build_probe_callable failed: " << err;
    probe_body_words_ = callable->body_words;
    ASSERT_FALSE(probe_body_words_.empty());

    // Decode .text and collect relocatable anchors. Decode-and-search so the
    // test stays stable across compiler revisions.
    auto decoder = Decoder::create(params_.arch);
    ASSERT_NE(decoder, nullptr);
    auto block_result = BasicBlock::build(*co, *decoder, params_.arch);
    ASSERT_TRUE(block_result.succeeded());
    auto blocks = std::move(block_result).value();
    ASSERT_FALSE(co->text_sections().empty());
    const auto *text = co->text_sections().front();
    const std::span<const uint8_t> text_bytes(reinterpret_cast<const uint8_t *>(text->data()),
                                              text->size());

    std::vector<uint64_t> candidates;
    for (const auto &block : blocks) {
      uint64_t cur = block->start_offset();
      for (const Instruction &inst : block->instructions()) {
        if (is_relocatable_anchor(inst, cur, text_bytes, params_.arch))
          candidates.push_back(cur);
        cur += static_cast<uint64_t>(inst.size());
      }
    }
    ASSERT_FALSE(candidates.empty()) << "No relocatable anchor in " << params_.kernel_fixture
                                     << ".o; did the compiler change the lowering?";

    // Pick the first anchor whose probe-call patch the resource/spill policy
    // accepts. A live fixed link pair s[30:31] or an exhausted dead-pair pool
    // would make a given anchor fail closed; trying several keeps the smoke
    // test deterministic without hard-coding a fragile offset.
    for (uint64_t off : candidates) {
      Instrumentor instr(*co, params_.arch);
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

  // Body of the static case. It lives on the fixture rather than in the TEST_F
  // macro so every target's suite runs the same code; the concrete suites only
  // bind a DbiTargetParams to it.
  void run_patched_elf_contains_probe_call_instrumentation();

  const DbiTargetParams &params_;
  std::vector<uint8_t> original_elf_bytes_;
  std::vector<uint8_t> patched_elf_bytes_;
  std::vector<uint32_t> probe_body_words_;
  std::vector<InstrumentationPatch> patches_;
  uint64_t anchor_offset_ = 0;
  std::string last_patch_error_;
};

// Hardware suites for this test. DbiHardwareBase supplies the per-target HSA
// setup and the two bodies both smoke tests share; the only case unique to the
// probe-call path is the sabotage one below.
template <const DbiTargetParams &Params>
class HsaDbiNopProbeHardwareBase : public DbiHardwareBase<HsaDbiNopProbeFixture, Params> {
protected:
  using Base = DbiHardwareBase<HsaDbiNopProbeFixture, Params>;

  void run_probe_body_is_actually_called_by_gpu();
};

// Static verification: prove the probe-call patch actually rewrote the kernel,
// copied the probe body in, and wired an s_swappc_b64 call to it. Catches:
//   - patcher silently produced the original bytes
//   - the anchor was not redirected to a branch stub
//   - the copied probe body is missing or differs from rj_nop_probe
//   - the trampoline does not contain the call to the probe
void HsaDbiNopProbeFixture::run_patched_elf_contains_probe_call_instrumentation() {
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
  auto decoder = Decoder::create(params_.arch);
  ASSERT_NE(decoder, nullptr);
  ASSERT_GT(text->size(), anchor_offset_ + sizeof(uint32_t));
  rj_code_binary_inst_t anchor_word = 0;
  std::memcpy(&anchor_word, text->data() + anchor_offset_, sizeof(anchor_word));
  auto decoded = decoder->decode(&anchor_word);
  ASSERT_TRUE(decoded.succeeded());
  std::unique_ptr<Instruction> anchor_inst = std::move(decoded).value();
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
template <const DbiTargetParams &Params>
void HsaDbiNopProbeHardwareBase<Params>::run_probe_body_is_actually_called_by_gpu() {
  hsa_agent_t gpu = Base::s_gpu_;
  hsa_agent_t cpu = find_cpu_agent();
  ASSERT_NE(cpu.handle, 0u);

  std::vector<uint8_t> sabotaged = this->patched_elf_bytes_;
  AmdGpuCodeObject parsed(sabotaged.data(), sabotaged.size());
  ASSERT_TRUE(parsed.is_valid());
  ASSERT_FALSE(parsed.text_sections().empty());
  const Section *text = parsed.text_sections().front();

  const uint64_t body_off = this->patches_[0].probe_target_offset;
  ASSERT_GE(text->size(), body_off + sizeof(uint32_t));
  const uint64_t body_file_off = text->sectionOffset() + body_off;

  // The first word of the copied body must not already be s_endpgm; otherwise
  // the sabotage proves nothing.
  const uint32_t s_endpgm_0 = build_s_endpgm(Params.arch);
  uint32_t pre_overwrite = 0;
  std::memcpy(&pre_overwrite, sabotaged.data() + body_file_off, sizeof(pre_overwrite));
  ASSERT_NE(pre_overwrite, s_endpgm_0) << "Probe body already starts with s_endpgm?";
  std::memcpy(sabotaged.data() + body_file_off, &s_endpgm_0, sizeof(s_endpgm_0));

  // Same inputs as the dispatch-equivalence test, so the golden matches.
  const GoldenVectorAddInputs inputs;
  constexpr uint32_t N = GoldenVectorAddInputs::kSize;

  auto sabotaged_out = dispatch_vector_add(sabotaged, gpu, cpu, inputs.a(), inputs.b(), N);
  ASSERT_EQ(sabotaged_out.size(), N)
      << "sabotaged dispatch failed (HSA error before s_endpgm could run)";

  // Probe-called path: every wave enters the body, hits s_endpgm, and
  // terminates before the store-to-C, so C stays zero.
  EXPECT_FALSE(kernel_wrote_output(sabotaged_out))
      << "sabotaged dispatch produced non-zero output — did the GPU bypass the probe call?";

  const uint32_t matches_golden = count_matching_golden(sabotaged_out, inputs.golden());
  EXPECT_LT(matches_golden, N) << "sabotaged dispatch matched the golden in " << matches_golden
                               << "/" << N << " elements — probe call appears bypassed";

  // Revert: restore the original first body word and confirm the kernel is
  // back to producing the golden output (proves the sabotage, not some
  // unrelated corruption, caused the zero result).
  std::memcpy(sabotaged.data() + body_file_off, &pre_overwrite, sizeof(pre_overwrite));
  auto reverted_out = dispatch_vector_add(sabotaged, gpu, cpu, inputs.a(), inputs.b(), N);
  ASSERT_EQ(reverted_out.size(), N) << "HSA error before reverted run could finish";
  ASSERT_EQ(count_matching_golden(reverted_out, inputs.golden()), N)
      << "reverted (un-sabotaged) probe-call code differs from golden";
}

// The concrete suites CMake registers. Availability is handled by
// DbiHardwareBase::SetUp(), so these bodies are ordinary calls.

// gfx90a / CDNA2.

class HsaDbiNopProbeCdna2Static : public HsaDbiNopProbeFixture {
protected:
  HsaDbiNopProbeCdna2Static() : HsaDbiNopProbeFixture(kCdna2Params) {}
};

class HsaDbiNopProbeCdna2Hardware : public HsaDbiNopProbeHardwareBase<kCdna2Params> {};

TEST_F(HsaDbiNopProbeCdna2Static, PatchedElfContainsProbeCallInstrumentation) {
  run_patched_elf_contains_probe_call_instrumentation();
}

TEST_F(HsaDbiNopProbeCdna2Hardware, PatchedElfLoadsAndValidatesInHsaExecutable) {
  run_patched_elf_loads_and_validates();
}

TEST_F(HsaDbiNopProbeCdna2Hardware, PatchedKernelDispatchMatchesOriginal) {
  run_patched_kernel_dispatch_matches_original();
}

TEST_F(HsaDbiNopProbeCdna2Hardware, ProbeBodyIsActuallyCalledByGpu) {
  run_probe_body_is_actually_called_by_gpu();
}

// gfx950 / CDNA4.

class HsaDbiNopProbeCdna4Static : public HsaDbiNopProbeFixture {
protected:
  HsaDbiNopProbeCdna4Static() : HsaDbiNopProbeFixture(kCdna4Params) {}
};

class HsaDbiNopProbeCdna4Hardware : public HsaDbiNopProbeHardwareBase<kCdna4Params> {};

TEST_F(HsaDbiNopProbeCdna4Static, PatchedElfContainsProbeCallInstrumentation) {
  run_patched_elf_contains_probe_call_instrumentation();
}

TEST_F(HsaDbiNopProbeCdna4Hardware, PatchedElfLoadsAndValidatesInHsaExecutable) {
  run_patched_elf_loads_and_validates();
}

TEST_F(HsaDbiNopProbeCdna4Hardware, PatchedKernelDispatchMatchesOriginal) {
  run_patched_kernel_dispatch_matches_original();
}

TEST_F(HsaDbiNopProbeCdna4Hardware, ProbeBodyIsActuallyCalledByGpu) {
  run_probe_body_is_actually_called_by_gpu();
}

// gfx1201 / RDNA4.

class HsaDbiNopProbeRdna4Static : public HsaDbiNopProbeFixture {
protected:
  HsaDbiNopProbeRdna4Static() : HsaDbiNopProbeFixture(kRdna4Params) {}
};

class HsaDbiNopProbeRdna4Hardware : public HsaDbiNopProbeHardwareBase<kRdna4Params> {};

TEST_F(HsaDbiNopProbeRdna4Static, PatchedElfContainsProbeCallInstrumentation) {
  run_patched_elf_contains_probe_call_instrumentation();
}

TEST_F(HsaDbiNopProbeRdna4Hardware, PatchedElfLoadsAndValidatesInHsaExecutable) {
  run_patched_elf_loads_and_validates();
}

TEST_F(HsaDbiNopProbeRdna4Hardware, PatchedKernelDispatchMatchesOriginal) {
  run_patched_kernel_dispatch_matches_original();
}

TEST_F(HsaDbiNopProbeRdna4Hardware, ProbeBodyIsActuallyCalledByGpu) {
  run_probe_body_is_actually_called_by_gpu();
}

#endif // HAS_HOST_AMDGPU
