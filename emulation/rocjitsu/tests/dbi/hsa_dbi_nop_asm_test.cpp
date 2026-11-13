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

#include "../elf_test_support.h"
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

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#ifdef HAS_HOST_AMDGPU
using namespace rocjitsu;
using namespace rocjitsu::dbi_test;

// ROCR's async-event pool is protected by an uninstrumented HybridMutex, so
// TSan can report its allocator reuse as a race during HSA initialization.
// Suppress only accesses originating in that external runtime.
extern "C" RJ_API_EXPORT const char *__tsan_default_suppressions() {
  return "called_from_lib:libhsa-runtime64.so\n"
         "thread:rocr::os::os_thread\n";
}

namespace {

using test::kernel_path;

// probe_fixture is null throughout: the inline-nop path calls no probe.
constexpr DbiTargetParams kCdna2Params{ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_TARGET_GFX90A,
                                       "vector_add_gfx90a", nullptr, "gfx90a"};
// gfx950 reuses the default-arch vector_add build (tests/kernels/CMakeLists.txt
// compiles it for gfx950 already), so this target needs no fixture of its own.
constexpr DbiTargetParams kCdna4Params{ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_TARGET_GFX950,
                                       "vector_add", nullptr, "gfx950"};
constexpr DbiTargetParams kRdna4Params{ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_TARGET_GFX1201,
                                       "vector_add_gfx1201", nullptr, "gfx1201"};

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

    // Select anchors and locate the end of live code from decoded instructions
    // so the fixture follows compiler changes without fixed byte offsets.
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
        if (anchor_count_ < 2 && is_relocatable_anchor(inst, cur, text_bytes, params_.arch)) {
          anchor_offsets_.push_back(cur); // Instrumentor will need offset
          anchor_mnemonics_.push_back(std::string(inst.mnemonic()));
          ++anchor_count_;
        }
        cur += static_cast<uint64_t>(inst.size());
        if (inst.mnemonic() == "s_endpgm")
          original_live_end_ = std::max(original_live_end_, cur);
      }
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
  uint64_t original_live_end_ = 0;
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
  const hsa_agent_t gpu = Base::s_gpu_;
  const hsa_agent_t cpu = find_cpu_agent();
  ASSERT_NE(cpu.handle, 0u);
  std::vector<uint8_t> sabotaged = this->patched_elf_bytes_;
  AmdGpuCodeObject parsed(sabotaged.data(), sabotaged.size());
  ASSERT_TRUE(parsed.is_valid());
  ASSERT_FALSE(parsed.text_sections().empty());
  const Section *text = parsed.text_sections().front();
  ASSERT_EQ(this->anchor_count_, 2u);
  ASSERT_EQ(this->patches_.size(), 2u);
  EXPECT_NE(this->patches_[0].trampoline_offset, this->patches_[1].trampoline_offset)
      << "Both selected trampolines have the same trampoline offset";

  const util::FailureOr<uint32_t> nop = build_s_nop(0, Params.arch);
  const util::FailureOr<uint32_t> endpgm = build_s_endpgm(Params.arch);
  ASSERT_TRUE(nop.succeeded());
  ASSERT_TRUE(endpgm.succeeded());
  const GoldenVectorAddInputs inputs;
  constexpr uint32_t kSize = GoldenVectorAddInputs::kSize;
  for (size_t index = 0; index < this->patches_.size(); ++index) {
    SCOPED_TRACE(index);
    const uint64_t cave = this->patches_[index].trampoline_offset;
    ASSERT_LE(cave, text->size());
    ASSERT_LE(sizeof(uint32_t), text->size() - cave);
    const uint64_t file_offset = text->sectionOffset() + cave;
    // Verify each cave's NOP premise before replacing it with s_endpgm.
    const uint32_t original_word =
        test_support::read_elf_struct_for_test<uint32_t>(sabotaged, file_offset);
    ASSERT_EQ(original_word, nop.value()) << "Trampoline body layout changed";
    test_support::write_value_for_test(sabotaged, file_offset, endpgm.value());

    const std::vector<float> stopped =
        dispatch_vector_add(sabotaged, gpu, cpu, inputs.a(), inputs.b(), kSize);
    ASSERT_EQ(stopped.size(), kSize)
        << "Sabotaged dispatch failed (HSA error before s_endpgm could run)";
    EXPECT_FALSE(kernel_wrote_output(stopped))
        << "Sabotaged dispatch produced non-zero output - did the GPU bypass the trampoline?";
    const uint32_t matches = count_matching_golden(stopped, inputs.golden());
    EXPECT_LT(matches, kSize) << "Sabotaged dispatch matched the golden in " << matches << "/"
                              << kSize << " elements - trampoline appears bypassed";

    // Restore the first cave and verify the golden before sabotaging the second.
    if (index == 0) {
      test_support::write_value_for_test(sabotaged, file_offset, nop.value());
      const std::vector<float> restored =
          dispatch_vector_add(sabotaged, gpu, cpu, inputs.a(), inputs.b(), kSize);
      ASSERT_EQ(restored.size(), kSize) << "HSA error before unsabotaged run could finish";
      ASSERT_EQ(count_matching_golden(restored, inputs.golden()), kSize)
          << "Unsabotaged code differs from golden";
    }
  }
}

// The concrete suites CMake registers. Availability is handled by
// DbiHardwareBase::SetUp(), so these bodies are ordinary calls.

// gfx90a / CDNA2.

class HsaDbiNopAsmCdna2Static : public HsaDbiNopAsmFixture {
protected:
  HsaDbiNopAsmCdna2Static() : HsaDbiNopAsmFixture(kCdna2Params) {}
};

class HsaDbiNopAsmCdna2Hardware : public HsaDbiNopAsmHardwareBase<kCdna2Params> {
protected:
  void run_tail_backing_case(bool file_backed);
};

void HsaDbiNopAsmCdna2Hardware::run_tail_backing_case(bool file_backed) {
  std::vector<uint8_t> image = original_elf_bytes_;
  AmdGpuCodeObject original(image.data(), image.size());
  ASSERT_TRUE(original.is_valid());
  ASSERT_EQ(original.text_sections().size(), 1u);
  const Section &text = *original.text_sections().front();
  std::unique_ptr<Decoder> decoder = Decoder::create(kCdna2Params.arch);
  ASSERT_NE(decoder, nullptr);
  const uint64_t live_end = original_live_end_;
  ASSERT_GT(live_end, anchor_offsets_.front());
  ASSERT_LE(live_end + sizeof(uint32_t), text.size());
  const uint32_t padding_word =
      test_support::read_elf_struct_for_test<uint32_t>(image, text.sectionOffset() + live_end);
  const DecodeResult padding = decoder->decode_window(std::span(&padding_word, 1));
  ASSERT_TRUE(padding.succeeded());
  ASSERT_EQ(padding.value()->mnemonic(), "s_nop");

  const Elf64_Ehdr header = test_support::read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
  ASSERT_TRUE(text.sectionHeaderIndex());
  const size_t text_index = *text.sectionHeaderIndex();
  ASSERT_LT(text_index, header.e_shnum);
  Elf64_Shdr text_header = test_support::read_elf_struct_for_test<Elf64_Shdr>(
      image, header.e_shoff + text_index * header.e_shentsize);
  const std::optional<Elf64_Sym> kernel =
      test_support::find_elf_symbol_for_test(image, "vector_add");
  ASSERT_TRUE(kernel);
  ASSERT_EQ(kernel->st_shndx, text_index);
  ASSERT_GE(kernel->st_value, text.vaddr());
  ASSERT_LE(kernel->st_value - text.vaddr() + kernel->st_size, live_end);
  std::optional<std::vector<Elf64_Phdr>> segments =
      read_elf_table<Elf64_Phdr>(image, header.e_phoff, header.e_phnum, header.e_phentsize);
  ASSERT_TRUE(segments);
  size_t owner = segments->size();
  for (size_t index = 0; index < segments->size(); ++index) {
    const Elf64_Phdr &segment = (*segments)[index];
    if (segment.p_type == PT_LOAD && segment.p_offset == text.sectionOffset() &&
        segment.p_vaddr == text.vaddr() && (segment.p_flags & PF_X) != 0) {
      ASSERT_EQ(owner, segments->size());
      owner = index;
    }
  }
  ASSERT_LT(owner, segments->size());
  ASSERT_EQ((*segments)[owner].p_filesz, text.size());
  ASSERT_EQ((*segments)[owner].p_memsz, text.size());

  // Consume existing unreachable linker padding without moving another section.
  // The only difference between the two cases is this tail word's file backing.
  text_header.sh_size = live_end + sizeof(uint32_t);
  (*segments)[owner].p_filesz = live_end + (file_backed ? sizeof(uint32_t) : 0);
  (*segments)[owner].p_memsz = text_header.sh_size;
  test_support::write_value_for_test<uint32_t>(image, text.sectionOffset() + live_end, 0);
  test_support::write_elf_struct_for_test(image, header.e_shoff + text_index * header.e_shentsize,
                                          text_header);
  test_support::write_elf_struct_for_test(image, header.e_phoff + owner * header.e_phentsize,
                                          (*segments)[owner]);
  const std::vector<uint8_t> pristine = image;
  AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  const GoldenVectorAddInputs inputs;
  const hsa_agent_t cpu = find_cpu_agent();
  ASSERT_NE(cpu.handle, 0u);
  const std::vector<float> original_out =
      dispatch_vector_add(image, s_gpu_, cpu, inputs.a(), inputs.b(), GoldenVectorAddInputs::kSize);
  ASSERT_EQ(original_out.size(), GoldenVectorAddInputs::kSize);
  const uint32_t original_matches = count_matching_golden(original_out, inputs.golden());
  RecordProperty("original_golden_matches", static_cast<int>(original_matches));
  ASSERT_EQ(original_matches, GoldenVectorAddInputs::kSize);

  Instrumentor instrumentor(source, kCdna2Params.arch);
  instrumentor.add_point_by_offset(anchor_offsets_.front());
  const Instrumentor::ValidationResult eligible = instrumentor.validate_points();
  ASSERT_TRUE(eligible.errors.empty());
  ASSERT_EQ(eligible.sites.size(), 1u);
  const InstrumentedCodeObjectDebug result = instrumentor.patch_with_debug_summaries();
  EXPECT_EQ(image, pristine);
  EXPECT_EQ(std::memcmp(source.image_data(), pristine.data(), pristine.size()), 0);
  if (!file_backed) {
    EXPECT_FALSE(result.errors.empty());
    EXPECT_TRUE(result.elf_bytes.empty());
    EXPECT_TRUE(result.sites.empty());
    EXPECT_TRUE(result.patches.empty());
    return;
  }

  ASSERT_TRUE(result.errors.empty())
      << (result.errors.empty() ? std::string{} : result.errors.front());
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_EQ(result.patches.size(), 1u);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const Section &patched_text = *patched.text_sections().front();
  ASSERT_GT(patched_text.size(), text_header.sh_size);
  for (uint64_t offset = text_header.sh_size; offset < patched_text.size(); ++offset) {
    const std::optional<uint64_t> loaded_offset =
        test_support::loaded_vaddr_to_file_offset(result.elf_bytes, patched_text.vaddr() + offset);
    ASSERT_TRUE(loaded_offset) << "Appended byte lacks file backing at text offset " << offset;
    ASSERT_EQ(*loaded_offset, patched_text.sectionOffset() + offset);
  }
  RecordProperty("appended_bytes_checked",
                 static_cast<int>(patched_text.size() - text_header.sh_size));
  ASSERT_NO_FATAL_FAILURE(expect_elf_loads_and_validates(result.elf_bytes, s_gpu_));
  const std::vector<float> patched_out = dispatch_vector_add(
      result.elf_bytes, s_gpu_, cpu, inputs.a(), inputs.b(), GoldenVectorAddInputs::kSize);
  ASSERT_EQ(patched_out.size(), GoldenVectorAddInputs::kSize);
  EXPECT_EQ(patched_out, original_out);
}

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

TEST_F(HsaDbiNopAsmCdna2Hardware, RejectsAppendedCodePastFileBackedLoad) {
  run_tail_backing_case(false);
}

TEST_F(HsaDbiNopAsmCdna2Hardware, AcceptsAppendedCodeAtFileBackedLoadEnd) {
  run_tail_backing_case(true);
}

// gfx950 / CDNA4.

class HsaDbiNopAsmCdna4Static : public HsaDbiNopAsmFixture {
protected:
  HsaDbiNopAsmCdna4Static() : HsaDbiNopAsmFixture(kCdna4Params) {}
};

class HsaDbiNopAsmCdna4Hardware : public HsaDbiNopAsmHardwareBase<kCdna4Params> {};

TEST_F(HsaDbiNopAsmCdna4Static, PatchedElfActuallyContainsInstrumentation) {
  run_patched_elf_actually_contains_instrumentation();
}

TEST_F(HsaDbiNopAsmCdna4Hardware, PatchedElfLoadsAndValidatesInHsaExecutable) {
  run_patched_elf_loads_and_validates();
}

TEST_F(HsaDbiNopAsmCdna4Hardware, PatchedKernelDispatchMatchesOriginal) {
  run_patched_kernel_dispatch_matches_original();
}

TEST_F(HsaDbiNopAsmCdna4Hardware, TrampolineIsActuallyExecutedByGpu) {
  run_trampoline_is_actually_executed_by_gpu();
}

// gfx1201 / RDNA4.

class HsaDbiNopAsmRdna4Static : public HsaDbiNopAsmFixture {
protected:
  HsaDbiNopAsmRdna4Static() : HsaDbiNopAsmFixture(kRdna4Params) {}
};

class HsaDbiNopAsmRdna4Hardware : public HsaDbiNopAsmHardwareBase<kRdna4Params> {};

TEST_F(HsaDbiNopAsmRdna4Static, PatchedElfActuallyContainsInstrumentation) {
  run_patched_elf_actually_contains_instrumentation();
}

TEST_F(HsaDbiNopAsmRdna4Hardware, PatchedElfLoadsAndValidatesInHsaExecutable) {
  run_patched_elf_loads_and_validates();
}

TEST_F(HsaDbiNopAsmRdna4Hardware, PatchedKernelDispatchMatchesOriginal) {
  run_patched_kernel_dispatch_matches_original();
}

TEST_F(HsaDbiNopAsmRdna4Hardware, TrampolineIsActuallyExecutedByGpu) {
  run_trampoline_is_actually_executed_by_gpu();
}

#endif // HAS_HOST_AMDGPU
