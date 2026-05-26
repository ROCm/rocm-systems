#include "instruction_relocator.h"

#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu_fuzzer/afl_dbi_plan.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <array>
#include <span>
#include <string_view>
#include <vector>

namespace {

void check(bool condition, const char *message) {
  if (!condition) {
    fprintf(stderr, "instruction_relocator_unit: %s\n", message);
    std::exit(1);
  }
}

std::vector<uint8_t> bytes(std::span<const uint32_t> words) {
  std::vector<uint8_t> out(words.size() * sizeof(uint32_t));
  memcpy(out.data(), words.data(), out.size());
  return out;
}

uint32_t build_s_or_b64_word(uint8_t sdst, uint16_t src0, uint16_t src1) {
  constexpr uint8_t kSOrB64Sop2Op = 25;
  return rocjitsu::pack_sop2(kSOrB64Sop2Op, sdst, src0, src1);
}

uint32_t build_sopc_word(uint8_t op, uint16_t src0, uint16_t src1) {
  constexpr uint32_t kSopcEncoding = 0x17eu;
  return (kSopcEncoding << 23) | (static_cast<uint32_t>(op) << 16) |
         (static_cast<uint32_t>(src1 & 0xffu) << 8) |
         static_cast<uint32_t>(src0 & 0xffu);
}

void check_failure(std::span<const uint32_t> words, rj_code_arch_t arch,
                   std::string_view expected_reason) {
  const std::vector<uint8_t> text = bytes(words);
  const char *failure = "stale failure";
  const auto relocated = rocjitsu::fuzzer::afl_dbi::relocate_overwritten_instruction(
      text, /*patch_text_offset=*/0, static_cast<uint32_t>(text.size()), arch, &failure);
  if (relocated.has_value()) {
    fprintf(stderr,
            "instruction_relocator_unit: unsafe instruction unexpectedly relocated; "
            "expected failure reason: %.*s\n",
            static_cast<int>(expected_reason.size()), expected_reason.data());
    std::exit(1);
  }
  check(failure != nullptr, "unsafe instruction should report a failure reason");
  if (std::string_view(failure) != expected_reason) {
    fprintf(stderr,
            "instruction_relocator_unit: unexpected relocation failure reason: "
            "actual='%s' expected='%.*s'\n",
            failure, static_cast<int>(expected_reason.size()), expected_reason.data());
    std::exit(1);
  }
}

void check_arch(rj_code_arch_t arch) {
  using namespace rocjitsu::fuzzer::afl_dbi;

  const uint32_t nop = rocjitsu::build_s_nop(0, arch);
  const std::vector<uint8_t> nop_text = bytes(std::span<const uint32_t>(&nop, 1));
  const char *failure = "stale failure";
  auto relocated =
      relocate_overwritten_instruction(nop_text, /*patch_text_offset=*/0, sizeof(nop), arch,
                                       &failure);
  check(relocated.has_value(), "s_nop should be safe to replay");
  check(relocated->words.size() == 1, "s_nop relocation word count changed");
  check(relocated->words[0] == nop, "s_nop relocation should preserve encoding");
  check(failure == nullptr, "safe relocation should clear failure reason");

  const uint32_t ordinary_mov =
      build_s_mov_b32_word(/*sdst=*/4, /*src=*/5, arch);
  const std::vector<uint8_t> ordinary_mov_text =
      bytes(std::span<const uint32_t>(&ordinary_mov, 1));
  failure = "stale failure";
  relocated = relocate_overwritten_instruction(ordinary_mov_text, /*patch_text_offset=*/0,
                                               sizeof(ordinary_mov), arch, &failure);
  check(relocated.has_value(), "ordinary s_mov_b32 should be safe to replay");
  check(relocated->words[0] == ordinary_mov,
        "ordinary s_mov_b32 relocation should preserve encoding");
  check(failure == nullptr, "ordinary s_mov_b32 should clear failure reason");

  const uint32_t ordinary_s_or_b64 =
      build_s_or_b64_word(/*sdst=*/8, /*src0=*/10, /*src1=*/12);
  const std::vector<uint8_t> ordinary_s_or_b64_text =
      bytes(std::span<const uint32_t>(&ordinary_s_or_b64, 1));
  failure = "stale failure";
  relocated = relocate_overwritten_instruction(ordinary_s_or_b64_text, /*patch_text_offset=*/0,
                                               sizeof(ordinary_s_or_b64), arch, &failure);
  check(relocated.has_value(), "ordinary s_or_b64 should be safe to replay");
  check(relocated->words[0] == ordinary_s_or_b64,
        "ordinary s_or_b64 relocation should preserve encoding");
  check(failure == nullptr, "ordinary s_or_b64 should clear failure reason");

  const uint32_t exec_src_mov =
      build_s_mov_b32_word(/*sdst=*/4, kScalarExecLo, arch);
  check_failure(std::span<const uint32_t>(&exec_src_mov, 1), arch,
                "overwritten s_mov_b32 requires operand-sensitive state-preservation checks");

  const uint32_t exec_dst_mov =
      build_s_mov_b32_word(kScalarExecLo, /*src=*/4, arch);
  check_failure(std::span<const uint32_t>(&exec_dst_mov, 1), arch,
                "overwritten s_mov_b32 requires operand-sensitive state-preservation checks");

  const uint32_t branch = rocjitsu::build_s_branch(/*offset_dwords=*/2, arch);
  const std::array<uint32_t, 5> branch_text = {
      rocjitsu::build_s_nop(0, arch),
      branch,
      rocjitsu::build_s_nop(0, arch),
      rocjitsu::build_s_nop(0, arch),
      rocjitsu::build_s_nop(0, arch),
  };
  const std::vector<uint8_t> branch_bytes = bytes(branch_text);
  failure = "stale failure";
  relocated = relocate_overwritten_instruction(branch_bytes, /*patch_text_offset=*/sizeof(uint32_t),
                                               sizeof(branch), arch, &failure);
  check(relocated.has_value(), "unconditional s_branch should be relocatable");
  check(relocated->words.empty(), "direct branch relocation should re-encode from the cave");
  check(relocated->direct_branch_target_text_offset.has_value(),
        "direct branch relocation should preserve target offset");
  check(*relocated->direct_branch_target_text_offset == 4 * sizeof(uint32_t),
        "direct branch relocation target changed");
  check(failure == nullptr, "direct branch relocation should clear failure reason");

  const uint32_t s_getpc_b64 = 0xBE804700u;
  check_failure(std::span<const uint32_t>(&s_getpc_b64, 1), arch,
                "overwritten PC-sensitive instruction requires PC relocation");

  const uint32_t wait_counter =
      build_rdna4_sopp(arch == ROCJITSU_CODE_ARCH_RDNA4 ? 0x47 : 9, /*imm=*/0);
  check_failure(std::span<const uint32_t>(&wait_counter, 1), arch,
                "overwritten wait-counter instruction must execute before injected probe");

  const uint32_t s_cmp_eq_i32 = build_sopc_word(/*op=*/0, /*src0=*/3, /*src1=*/4);
  const std::vector<uint8_t> s_cmp_text = bytes(std::span<const uint32_t>(&s_cmp_eq_i32, 1));
  failure = "stale failure";
  relocated = relocate_overwritten_instruction(s_cmp_text, /*patch_text_offset=*/0,
                                               sizeof(s_cmp_eq_i32), arch, &failure);
  check(relocated.has_value(), "ordinary s_cmp should be safe to replay after the probe");
  check(relocated->words[0] == s_cmp_eq_i32, "ordinary s_cmp relocation should preserve encoding");
  check(failure == nullptr, "ordinary s_cmp should clear failure reason");

  const uint32_t s_bitcmp0_b32 = build_sopc_word(/*op=*/12, /*src0=*/3, /*src1=*/4);
  const std::vector<uint8_t> s_bitcmp_text =
      bytes(std::span<const uint32_t>(&s_bitcmp0_b32, 1));
  failure = "stale failure";
  relocated = relocate_overwritten_instruction(s_bitcmp_text, /*patch_text_offset=*/0,
                                               sizeof(s_bitcmp0_b32), arch, &failure);
  check(relocated.has_value(), "ordinary s_bitcmp should be safe to replay after the probe");
  check(relocated->words[0] == s_bitcmp0_b32,
        "ordinary s_bitcmp relocation should preserve encoding");
  check(failure == nullptr, "ordinary s_bitcmp should clear failure reason");

  const std::array<uint32_t, 2> modeled_vopd = {3391357056u, 36569223u};
  const std::vector<uint8_t> modeled_vopd_text = bytes(modeled_vopd);
  failure = "stale failure";
  relocated = relocate_overwritten_instruction(
      modeled_vopd_text, /*patch_text_offset=*/0,
      static_cast<uint32_t>(modeled_vopd_text.size()), arch, &failure);
  check(relocated.has_value(), "modeled VOPD should be safe to replay after the probe");
  check(relocated->words.size() == modeled_vopd.size() &&
            relocated->words[0] == modeled_vopd[0] &&
            relocated->words[1] == modeled_vopd[1],
        "modeled VOPD relocation should preserve raw words");
  check(failure == nullptr, "modeled VOPD should clear failure reason");

  const uint32_t exec_src_cmp = build_sopc_word(/*op=*/0, kScalarExecLo, /*src1=*/4);
  check_failure(std::span<const uint32_t>(&exec_src_cmp, 1), arch,
                "overwritten SCC-producing instruction requires operand-sensitive relocation support");

  const uint32_t s_and_saveexec_b64 = 0xBE802100u;
  check_failure(std::span<const uint32_t>(&s_and_saveexec_b64, 1), arch,
                "overwritten EXEC transition requires mask relocation support");

  const uint32_t s_or_b64 =
      build_s_or_b64_word(static_cast<uint8_t>(kScalarExecLo), kScalarExecLo, /*src1=*/4);
  check_failure(std::span<const uint32_t>(&s_or_b64, 1), arch,
                "overwritten scalar OR may restore EXEC and needs operand-sensitive relocation support");

  constexpr uint32_t unmodeled_vopd_word0 =
      (0x32u << 26) | (0xeu << 22) | (0x8u << 17) | (7u << 9) | 260u;
  constexpr uint32_t unmodeled_vopd_word1 =
      (10u << 24) | (6u << 17) | (8u << 9) | 261u;
  const std::array<uint32_t, 2> unmodeled_vopd = {unmodeled_vopd_word0,
                                                  unmodeled_vopd_word1};
  check_failure(unmodeled_vopd, arch,
                "overwritten VOPD liveness is unmodeled and replay relocation is not enabled");
}

} // namespace

int main() {
  using namespace rocjitsu::fuzzer::afl_dbi;

  check_arch(ROCJITSU_CODE_ARCH_RDNA3);
  check_arch(ROCJITSU_CODE_ARCH_RDNA4);

  const std::vector<uint8_t> truncated_text = {0, 1, 2};
  const char *failure = nullptr;
  auto relocated = relocate_overwritten_instruction(truncated_text, /*patch_text_offset=*/0,
                                                    static_cast<uint32_t>(truncated_text.size()),
                                                    ROCJITSU_CODE_ARCH_RDNA4, &failure);
  check(!relocated.has_value(), "misaligned instruction size should fail");
  check(std::string_view(failure) == "overwritten instruction size is invalid",
        "misaligned instruction failure reason changed");

  return 0;
}
