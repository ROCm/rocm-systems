#include "instruction_relocator.h"

#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu_fuzzer/afl_dbi_plan.h"
#include "vopd_liveness.h"

#include <stdint.h>
#include <string.h>

#include <exception>
#include <memory>
#include <string_view>

namespace rocjitsu::fuzzer::afl_dbi {
namespace {

bool starts_with(std::string_view text, std::string_view prefix) {
  return text.substr(0, prefix.size()) == prefix;
}

bool is_control_flow(const rocjitsu::Instruction &inst) {
  return (inst.flags() & (rocjitsu::BRANCH | rocjitsu::COND_BRANCH |
                          rocjitsu::INDIRECT_BRANCH | rocjitsu::INDIRECT_CALL |
                          rocjitsu::PROGRAM_TERMINATOR)) != 0;
}

bool is_unconditional_direct_branch(const rocjitsu::Instruction &inst) {
  return (inst.flags() & rocjitsu::BRANCH) != 0 &&
         (inst.flags() & (rocjitsu::COND_BRANCH | rocjitsu::INDIRECT_BRANCH |
                          rocjitsu::INDIRECT_CALL | rocjitsu::PROGRAM_TERMINATOR)) == 0 &&
         inst.branch_offset_bytes().has_value();
}

bool arch_supports_vopd_replay(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
         arch == ROCJITSU_CODE_ARCH_RDNA4;
}

const char *vopd_relocation_blocker(std::span<const uint32_t> words,
                                    uint32_t instruction_size,
                                    rj_code_arch_t arch) {
  if (words.size() < 2 || decode_vopd64_word_count(words[0], words[1]) == 0)
    return nullptr;
  if (!arch_supports_vopd_replay(arch))
    return "overwritten VOPD instruction is unsupported for target";
  const uint32_t word_count = decode_vopd64_word_count(words[0], words[1]);
  if (instruction_size != word_count * sizeof(uint32_t))
    return "overwritten VOPD instruction size does not match encoding";
  if (!decode_vopd64_liveness_model(words[0], words[1]))
    return "overwritten VOPD liveness is unmodeled and replay relocation is not enabled";
  return nullptr;
}

bool register_ref_is_special_state(const rocjitsu::RegisterRef &ref) {
  switch (ref.cls) {
  case rocjitsu::RegClass::SGPR:
    return static_cast<uint32_t>(ref.index) + ref.width > kScalarVccLo;
  case rocjitsu::RegClass::VGPR:
  case rocjitsu::RegClass::ACC_VGPR:
    return false;
  case rocjitsu::RegClass::EXEC:
  case rocjitsu::RegClass::VCC:
  case rocjitsu::RegClass::SCC:
  case rocjitsu::RegClass::M0:
  case rocjitsu::RegClass::FLAT_SCRATCH:
  case rocjitsu::RegClass::TTMP:
  case rocjitsu::RegClass::PC:
    return true;
  }
  return true;
}

bool operand_is_special_state(const rocjitsu::Operand *operand, bool missing_is_special) {
  if (operand == nullptr)
    return missing_is_special;
  std::optional<rocjitsu::RegisterRef> ref = operand->to_register_ref();
  if (!ref)
    return missing_is_special;
  return register_ref_is_special_state(*ref);
}

bool scalar_operand_encoding_is_special_state(uint32_t raw, uint8_t width = 1) {
  return raw <= kScalarExecHi && raw + width > kScalarVccLo;
}

bool s_mov_b32_raw_uses_special_state(const rocjitsu::Instruction &inst) {
  if (inst.raw_encoding() == nullptr || inst.size() < static_cast<int>(sizeof(uint32_t)))
    return true;
  const uint32_t word = inst.raw_encoding()[0];
  const uint32_t ssrc0 = word & 0xffu;
  const uint32_t sdst = (word >> 16) & 0x7fu;
  return scalar_operand_encoding_is_special_state(ssrc0) ||
         scalar_operand_encoding_is_special_state(sdst);
}

bool s_or_b64_raw_uses_special_state(const rocjitsu::Instruction &inst) {
  if (inst.raw_encoding() == nullptr || inst.size() < static_cast<int>(sizeof(uint32_t)))
    return true;
  const uint32_t word = inst.raw_encoding()[0];
  const uint32_t ssrc0 = word & 0xffu;
  const uint32_t ssrc1 = (word >> 8) & 0xffu;
  const uint32_t sdst = (word >> 16) & 0x7fu;
  return scalar_operand_encoding_is_special_state(ssrc0, /*width=*/2) ||
         scalar_operand_encoding_is_special_state(ssrc1, /*width=*/2) ||
         scalar_operand_encoding_is_special_state(sdst, /*width=*/2);
}

bool is_scc_producing_scalar_compare(std::string_view mnemonic) {
  return starts_with(mnemonic, "s_cmp") || starts_with(mnemonic, "s_bitcmp");
}

bool is_wait_counter_instruction(const rocjitsu::Instruction &inst) {
  const std::string_view mnemonic = inst.mnemonic();
  return (inst.flags() & rocjitsu::WAITCNT) != 0 || mnemonic == "s_wait_alu" ||
         mnemonic == "s_wait_kmcnt" || mnemonic == "s_waitcnt";
}

bool sopc_raw_uses_special_state(const rocjitsu::Instruction &inst) {
  if (inst.raw_encoding() == nullptr || inst.size() < static_cast<int>(sizeof(uint32_t)))
    return true;
  const uint32_t word = inst.raw_encoding()[0];
  constexpr uint32_t kSopcEncoding = 0x17eu;
  if ((word >> 23) != kSopcEncoding)
    return false;
  const uint32_t ssrc0 = word & 0xffu;
  const uint32_t ssrc1 = (word >> 8) & 0xffu;
  return scalar_operand_encoding_is_special_state(ssrc0, /*width=*/2) ||
         scalar_operand_encoding_is_special_state(ssrc1, /*width=*/2);
}

const char *relocation_blocker(const rocjitsu::Instruction &inst) {
  if (inst.branch_offset_bytes().has_value() && !is_unconditional_direct_branch(inst))
    return "overwritten direct branch requires PC-relative relocation";
  if (is_control_flow(inst) && !is_unconditional_direct_branch(inst))
    return "overwritten control-flow instruction requires relocation support";

  const std::string_view mnemonic = inst.mnemonic();
  if (mnemonic == "s_getpc_b64")
    return "overwritten PC-sensitive instruction requires PC relocation";
  if (scc_compare_requires_state_preservation(inst))
    return "overwritten SCC-producing instruction requires operand-sensitive relocation support";
  if (is_wait_counter_instruction(inst))
    return "overwritten wait-counter instruction must execute before injected probe";
  // saveexec and common EXEC-restore idioms are not ordinary replayable
  // instructions. A safe relocation has to choose an explicit coverage point:
  // before the mask transition, after the mask transition, or around a probe
  // that preserves the original and transitioned EXEC values exactly.
  if (mnemonic.find("saveexec") != std::string_view::npos)
    return "overwritten EXEC transition requires mask relocation support";
  if (mnemonic == "s_or_b64" && s_or_b64_requires_state_preservation(inst))
    return "overwritten scalar OR may restore EXEC and needs operand-sensitive relocation support";
  if (mnemonic == "s_mov_b32" && s_mov_b32_requires_state_preservation(inst))
    return "overwritten s_mov_b32 requires operand-sensitive state-preservation checks";
  if (mnemonic == "vopd_opaque")
    return "overwritten opaque VOPD instruction is not relocatable";
  if (mnemonic == "unknown_opaque")
    return "overwritten unknown instruction is not relocatable";
  return nullptr;
}

} // namespace

bool s_mov_b32_requires_state_preservation(const rocjitsu::Instruction &inst) {
  if (inst.mnemonic() != "s_mov_b32")
    return false;

  // Ordinary SGPR/literal moves can be replayed from the trampoline once the
  // liveness planner has reserved probe temporaries. Moves that read or write
  // EXEC/VCC/SCC-like architectural state still need explicit preservation.
  if (s_mov_b32_raw_uses_special_state(inst))
    return true;
  for (int i = 0; i < inst.num_dst_operands(); ++i) {
    if (operand_is_special_state(inst.dst_operand(i), /*missing_is_special=*/true))
      return true;
  }
  for (int i = 0; i < inst.num_src_operands(); ++i) {
    if (operand_is_special_state(inst.src_operand(i), /*missing_is_special=*/false))
      return true;
  }
  return false;
}

bool s_or_b64_requires_state_preservation(const rocjitsu::Instruction &inst) {
  if (inst.mnemonic() != "s_or_b64")
    return false;

  if (s_or_b64_raw_uses_special_state(inst))
    return true;
  for (int i = 0; i < inst.num_dst_operands(); ++i) {
    if (operand_is_special_state(inst.dst_operand(i), /*missing_is_special=*/true))
      return true;
  }
  for (int i = 0; i < inst.num_src_operands(); ++i) {
    if (operand_is_special_state(inst.src_operand(i), /*missing_is_special=*/false))
      return true;
  }
  return false;
}

bool scc_compare_requires_state_preservation(const rocjitsu::Instruction &inst) {
  if (!is_scc_producing_scalar_compare(inst.mnemonic()))
    return false;

  // Replaying an ordinary scalar compare after the coverage probe restores the
  // original SCC value before the trampoline returns. Only operands that read
  // special architectural state need a more explicit relocation sequence.
  if (sopc_raw_uses_special_state(inst))
    return true;
  if (inst.num_src_operands() == 0)
    return true;
  for (int i = 0; i < inst.num_src_operands(); ++i) {
    const rocjitsu::Operand *operand = inst.src_operand(i);
    if (operand == nullptr)
      return true;
    std::optional<rocjitsu::RegisterRef> ref = operand->to_register_ref();
    if (ref && register_ref_is_special_state(*ref))
      return true;
  }
  return false;
}

std::optional<RelocatedInstruction>
relocate_overwritten_instruction(std::span<const uint8_t> text, uint64_t patch_text_offset,
                                 uint32_t instruction_size, rj_code_arch_t arch,
                                 const char **failure_reason) {
  if (failure_reason != nullptr)
    *failure_reason = nullptr;

  auto fail = [&](const char *reason) -> std::optional<RelocatedInstruction> {
    if (failure_reason != nullptr)
      *failure_reason = reason;
    return std::nullopt;
  };

  if (patch_text_offset > text.size() || instruction_size > text.size() - patch_text_offset)
    return fail("overwritten instruction is outside .text");
  if (instruction_size == 0 || instruction_size % sizeof(uint32_t) != 0)
    return fail("overwritten instruction size is invalid");

  RelocatedInstruction relocated;
  relocated.words.resize(instruction_size / sizeof(uint32_t));
  memcpy(relocated.words.data(), text.data() + patch_text_offset, instruction_size);

  if (relocated.words.size() >= 2 && decode_vopd64_word_count(relocated.words[0],
                                                              relocated.words[1]) != 0) {
    if (const char *reason = vopd_relocation_blocker(relocated.words, instruction_size, arch))
      return fail(reason);
    return relocated;
  }

  std::unique_ptr<rocjitsu::Decoder> decoder = rocjitsu::Decoder::create(arch);
  if (decoder == nullptr)
    return fail("no decoder for overwritten instruction target");

  std::unique_ptr<rocjitsu::Instruction> inst;
  try {
    inst.reset(decoder->decode(relocated.words.data()));
  } catch (const std::exception &) {
    return fail("overwritten instruction decode failed");
  }
  if (inst == nullptr)
    return fail("overwritten instruction decode returned null");
  if (inst->raw_encoding() == nullptr)
    return fail("overwritten instruction has no raw encoding");
  if (inst->size() != static_cast<int>(instruction_size))
    return fail("overwritten instruction size changed during decode");

  if (const char *reason = relocation_blocker(*inst))
    return fail(reason);

  if (is_unconditional_direct_branch(*inst)) {
    const uint64_t next_pc = patch_text_offset + static_cast<uint64_t>(instruction_size);
    const int64_t target =
        static_cast<int64_t>(next_pc) + static_cast<int64_t>(*inst->branch_offset_bytes());
    if (target < 0)
      return fail("overwritten direct branch target is negative");
    relocated.words.clear();
    relocated.direct_branch_target_text_offset = static_cast<uint64_t>(target);
  }

  return relocated;
}

} // namespace rocjitsu::fuzzer::afl_dbi
