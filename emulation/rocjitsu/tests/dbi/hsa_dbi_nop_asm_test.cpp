// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hsa_dbi_nop_asm_test.cpp
/// @brief End-to-end DBI smoke: patch a real compiled kernel with
///        Instrumentor::patch, then load and eventually dispatch the patched
///        ELF via HSA.
///
/// The fixture is parameterized by target (DbiTargetParams) so the same bodies
/// cover every ISA the DBI trampoline path supports; the concrete suites at the
/// bottom of this file are what CMake registers.

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
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <gtest/gtest.h>

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

using test::kernel_path;

// probe_fixture is null: the inline-nop path calls no probe.
constexpr DbiTargetParams kCdna2Params{ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_TARGET_GFX90A,
                                       "vector_add_gfx90a", nullptr, "gfx90a"};

} // namespace

// Shared fixture: loads the target's vector_add build, decodes .text, finds the
// first two relocatable anchors, and patches them via Instrumentor. The
// concrete suites at the bottom of this file inherit it so CMake can register
// one CTest entry per (target, gating policy) pair:
//   - HsaDbiNopAsm<Target>Static.*   - no GPU needed, registered unconditionally
//   - HsaDbiNopAsm<Target>Hardware.* - registered as a Sim case unconditionally,
//                                      and bare only when the GPU is present
class HsaDbiNopAsmFixture : public ::testing::Test {
protected:
  // Names this test's instrumentation mechanism in the shared dispatch-
  // equivalence failure message (see DbiHardwareBase).
  static constexpr const char *kPatchDescription = "the inline-nop placeholder";

  explicit HsaDbiNopAsmFixture(const DbiTargetParams &params) : params_(params) {}

  // Load the target's vector_add device ELF and instrument two instructions
  // with the inlined nop functionality currently available in Instrumentor
  void SetUp() override {
    Executable exec(kernel_path(params_.kernel_fixture));
    ASSERT_TRUE(exec.is_valid()) << "Failed to load " << params_.kernel_fixture << ".o";
    ASSERT_GT(exec.num_code_objects(params_.target), 0u);
    const auto *co = exec.code_object(params_.target, 0);
    ASSERT_NE(co, nullptr);

    // Snapshot the original device ELF so we can dispatch it too
    original_elf_bytes_.assign(reinterpret_cast<const uint8_t *>(co->image_data()),
                               reinterpret_cast<const uint8_t *>(co->image_data()) +
                                   co->image_size());

    // Decode .text and find the first v_add_f32-mnemonic anchor that the
    // trampoline machinery considers relocatable. Decode-and-search so the
    // test is stable across compiler revisions.
    // TODO: instrument multiple instructions
    auto decoder = Decoder::create(params_.arch);
    ASSERT_NE(decoder, nullptr);
    auto block_result = BasicBlock::build(*co, *decoder, params_.arch);
    ASSERT_TRUE(block_result.succeeded());
    auto blocks = std::move(block_result).value();

    ASSERT_FALSE(co->text_sections().empty());
    const auto *text = co->text_sections().front();
    const std::span<const uint8_t> text_bytes(reinterpret_cast<const uint8_t *>(text->data()),
                                              text->size());

    for (const auto &block : blocks) {
      uint64_t cur = block->start_offset();
      for (const Instruction &inst : block->instructions()) {
        if (is_relocatable_anchor(inst, cur, text_bytes, params_.arch)) {
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
    ASSERT_NE(anchor_count_, 0u) << "No relocatable anchor in " << params_.kernel_fixture
                                 << ".o; did the compiler change the lowering?";

    // Apply the inline-nop trampoline.
    Instrumentor instrumentor(*co, params_.arch);
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

  // Body of the static case. It lives on the fixture rather than in the TEST_F
  // macro so every target's suite runs the same code; the concrete suites only
  // bind a DbiTargetParams to it.
  void run_patched_elf_actually_contains_instrumentation();

  const DbiTargetParams &params_;
  std::vector<uint8_t> original_elf_bytes_;
  std::vector<uint8_t> patched_elf_bytes_;
  std::vector<uint64_t> anchor_offsets_;
  std::vector<std::string> anchor_mnemonics_;
  std::vector<InstrumentationPatch> patches_;
  uint64_t anchor_count_ = 0;
};

// Hardware suites for this test. DbiHardwareBase supplies the per-target HSA
// setup and the two bodies both smoke tests share; the only case unique to the
// inline-nop path is the sabotage one below.
template <const DbiTargetParams &Params>
class HsaDbiNopAsmHardwareBase : public DbiHardwareBase<HsaDbiNopAsmFixture, Params> {
protected:
  using Base = DbiHardwareBase<HsaDbiNopAsmFixture, Params>;

  void run_trampoline_is_actually_executed_by_gpu();
};

// Static verification: prove the patcher actually changed the kernel before
// any HSA / dispatch tests run. No GPU or HSA runtime required. Catches
// failure modes that the byte-equality dispatch check cannot:
//   - patcher silently produced the original bytes (e.g. Instrumentor::patch
//     short-circuited without applying the patch)
//   - patch landed at a different offset than anchor_offset_ records
//   - the trampoline cave was not appended to .text
void HsaDbiNopAsmFixture::run_patched_elf_actually_contains_instrumentation() {
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

  auto decoder = Decoder::create(params_.arch);
  ASSERT_NE(decoder, nullptr);
  for (uint64_t anchor_idx = 0; anchor_idx < anchor_count_; ++anchor_idx) {
    rj_code_binary_inst_t anchor_word = 0;
    std::memcpy(&anchor_word, text->data() + anchor_offsets_[anchor_idx], sizeof(anchor_word));
    auto decode_result = decoder->decode(&anchor_word);
    ASSERT_TRUE(decode_result.succeeded());
    std::unique_ptr<Instruction> decoded = std::move(decode_result).value();
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
template <const DbiTargetParams &Params>
void HsaDbiNopAsmHardwareBase<Params>::run_trampoline_is_actually_executed_by_gpu() {
  hsa_agent_t gpu = Base::s_gpu_;
  hsa_agent_t cpu = find_cpu_agent();
  ASSERT_NE(cpu.handle, 0u);

  // The trampolines live inside .text at .text-relative offset
  // patches_[i].trampoline_offset. Overwrite the first 4 bytes of each
  // trampoline (the s_nop 0 placeholder) with s_endpgm.
  std::vector<uint8_t> sabotaged = this->patched_elf_bytes_;
  AmdGpuCodeObject parsed(sabotaged.data(), sabotaged.size());
  ASSERT_TRUE(parsed.is_valid());
  ASSERT_FALSE(parsed.text_sections().empty());
  const Section *text = parsed.text_sections().front();

  ASSERT_EQ(this->anchor_count_, 2u);
  ASSERT_EQ(this->patches_.size(), 2u);
  ASSERT_GE(text->size(), this->patches_[1].trampoline_offset + 4);
  // File offset of the first trampoline's placeholder word.
  const uint64_t tramp0_file_off = text->sectionOffset() + this->patches_[0].trampoline_offset;
  // Before sabotaging: verify the bytes we're about to overwrite are indeed
  // s_nop 0. If this assertion fails, the orchestrator's trampoline layout
  // no longer starts with the placeholder we think it does, and the
  // sabotage premise ("we replaced the no-op with s_endpgm") would be a lie.
  const uint32_t s_nop_0 = build_s_nop(0, Params.arch);
  uint32_t pre_overwrite = 0;
  std::memcpy(&pre_overwrite, sabotaged.data() + tramp0_file_off, sizeof(pre_overwrite));
  ASSERT_EQ(pre_overwrite, s_nop_0) << "Expected s_nop 0 (0x" << std::hex << s_nop_0
                                    << ") at start of the trampoline cave but found 0x"
                                    << pre_overwrite << " - trampoline body layout changed?";

  const uint32_t s_endpgm_0 = build_s_endpgm(Params.arch);
  std::memcpy(sabotaged.data() + tramp0_file_off, &s_endpgm_0, sizeof(s_endpgm_0));

  // Same inputs as the dispatch-equivalence test, so the golden matches.
  const GoldenVectorAddInputs inputs;
  constexpr uint32_t N = GoldenVectorAddInputs::kSize;

  auto sabotaged_out_1 = dispatch_vector_add(sabotaged, gpu, cpu, inputs.a(), inputs.b(), N);
  ASSERT_EQ(sabotaged_out_1.size(), N)
      << "Sabotaged dispatch failed (HSA error before s_endpgm could run)";

  // Trampoline-executed path: every thread that enters the trampoline hits
  // s_endpgm and terminates before reaching the relocated instruction or its
  // store-to-C. So C stays at the pre-dispatch zero pattern.
  EXPECT_FALSE(kernel_wrote_output(sabotaged_out_1))
      << "Sabotaged dispatch produced non-zero output - did the GPU bypass the trampoline?";

  // And the output must NOT match the golden (would mean the trampoline
  // wasn't hit and the kernel ran end-to-end normally).
  const uint32_t matches_golden_1 = count_matching_golden(sabotaged_out_1, inputs.golden());
  EXPECT_LT(matches_golden_1, N) << "Sabotaged dispatch matched the golden in " << matches_golden_1
                                 << "/" << N << " elements - trampoline appears bypassed";

  // Revert first trampoline in sabotaged
  std::memcpy(sabotaged.data() + tramp0_file_off, &s_nop_0, sizeof(s_nop_0));
  auto unsabotaged_out = dispatch_vector_add(sabotaged, gpu, cpu, inputs.a(), inputs.b(), N);
  ASSERT_EQ(unsabotaged_out.size(), N) << "HSA error before unsabotaged run could finish";

  ASSERT_EQ(count_matching_golden(unsabotaged_out, inputs.golden()), N)
      << "Unsabotaged code differs from golden";

  // Perform same change and test for second trampoline
  int64_t offset_between_anchors =
      this->patches_[1].trampoline_offset - this->patches_[0].trampoline_offset;
  EXPECT_NE(offset_between_anchors, 0)
      << "Both selected trampolines have the same trampoline offset";
  std::memcpy(&pre_overwrite, sabotaged.data() + tramp0_file_off + offset_between_anchors,
              sizeof(pre_overwrite));
  ASSERT_EQ(pre_overwrite, s_nop_0) << "Expected s_nop 0 (0x" << std::hex << s_nop_0
                                    << ") at start of the trampoline cave but found 0x"
                                    << pre_overwrite << " - trampoline body layout changed?";

  std::memcpy(sabotaged.data() + tramp0_file_off + offset_between_anchors, &s_endpgm_0,
              sizeof(s_endpgm_0));

  auto sabotaged_out_2 = dispatch_vector_add(sabotaged, gpu, cpu, inputs.a(), inputs.b(), N);
  ASSERT_EQ(sabotaged_out_2.size(), N)
      << "Sabotaged dispatch failed (HSA error before s_endpgm could run)";

  // Trampoline-executed path: every thread that enters the trampoline hits
  // s_endpgm and terminates before reaching the relocated v_add_f32 or its
  // store-to-C. So C stays at the pre-dispatch zero pattern.
  EXPECT_FALSE(kernel_wrote_output(sabotaged_out_2))
      << "Sabotaged dispatch produced non-zero output - did the GPU bypass the trampoline?";

  // And the output must NOT match the golden (would mean the trampoline
  // wasn't hit and the kernel ran end-to-end normally).
  const uint32_t matches_golden_2 = count_matching_golden(sabotaged_out_2, inputs.golden());
  EXPECT_LT(matches_golden_2, N) << "Sabotaged dispatch matched the golden in " << matches_golden_2
                                 << "/" << N << " elements - trampoline appears bypassed";
}

// The concrete suites CMake registers. Availability is handled by
// DbiHardwareBase::SetUp(), so these bodies are ordinary calls.

// gfx90a / CDNA2.

class HsaDbiNopAsmCdna2Static : public HsaDbiNopAsmFixture {
protected:
  HsaDbiNopAsmCdna2Static() : HsaDbiNopAsmFixture(kCdna2Params) {}
};

class HsaDbiNopAsmCdna2Hardware : public HsaDbiNopAsmHardwareBase<kCdna2Params> {};

TEST_F(HsaDbiNopAsmCdna2Static, PatchedElfActuallyContainsInstrumentation) {
  run_patched_elf_actually_contains_instrumentation();
}

TEST_F(HsaDbiNopAsmCdna2Hardware, PatchedElfLoadsAndValidatesInHsaExecutable) {
  run_patched_elf_loads_and_validates();
}

TEST_F(HsaDbiNopAsmCdna2Hardware, PatchedKernelDispatchMatchesOriginal) {
  run_patched_kernel_dispatch_matches_original();
}

TEST_F(HsaDbiNopAsmCdna2Hardware, TrampolineIsActuallyExecutedByGpu) {
  run_trampoline_is_actually_executed_by_gpu();
}

#endif // HAS_HOST_AMDGPU
