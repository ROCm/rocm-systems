// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/static_pc_recovery.h"

#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/code/patch/kernel_text_layout.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <array>
#include <cstring>

namespace rocjitsu {

namespace {

enum class ScalarPcOp {
  GetPc64,
  SetPc64,
  SwapPc64,
};

enum class ScalarSop2Op {
  AddU32,
  AddI32,
  AddcU32,
};

[[nodiscard]] uint32_t text_word_at(std::span<const uint8_t> text, uint64_t offset) {
  uint32_t word = 0;
  if (offset + sizeof(word) <= text.size())
    std::memcpy(&word, text.data() + offset, sizeof(word));
  return word;
}

/// @brief Return the SOP1 opcode number for a scalar PC operation on @p arch.
///
/// @details DBT recovery describes the operation it needs, then asks this
/// helper for the encoding detail for the current ISA. That keeps architecture
/// differences in one switch instead of spreading per-family constants through
/// the recovery algorithm.
[[nodiscard]] std::optional<uint8_t> scalar_pc_opcode(rj_code_arch_t arch, ScalarPcOp op) {
  auto add_base = [&](uint8_t base) -> uint8_t {
    switch (op) {
    case ScalarPcOp::GetPc64:
      return base;
    case ScalarPcOp::SetPc64:
      return static_cast<uint8_t>(base + 1);
    case ScalarPcOp::SwapPc64:
      return static_cast<uint8_t>(base + 2);
    }
    return base;
  };

  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
  case ROCJITSU_CODE_ARCH_CDNA2:
  case ROCJITSU_CODE_ARCH_CDNA3:
  case ROCJITSU_CODE_ARCH_CDNA4:
    return add_base(0x1c);
  case ROCJITSU_CODE_ARCH_RDNA1:
  case ROCJITSU_CODE_ARCH_RDNA2:
    return add_base(0x1f);
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
  case ROCJITSU_CODE_ARCH_RDNA4:
    return add_base(0x47);
  case ROCJITSU_CODE_ARCH_GFX1250:
  case ROCJITSU_CODE_ARCH_RV32I:
  case ROCJITSU_CODE_ARCH_RV64I:
  case ROCJITSU_CODE_ARCH_NUM_ARCHS:
    return std::nullopt;
  }
  return std::nullopt;
}

/// @brief Return the SOP2 opcode number for scalar arithmetic on @p arch.
///
/// @details The current supported AMDGPU families use the same SOP2 opcode
/// numbers for these arithmetic operations, but recovery still asks through
/// this helper so future ISA differences are isolated in one switch.
[[nodiscard]] std::optional<uint8_t> scalar_sop2_opcode(rj_code_arch_t arch, ScalarSop2Op op) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
  case ROCJITSU_CODE_ARCH_CDNA2:
  case ROCJITSU_CODE_ARCH_CDNA3:
  case ROCJITSU_CODE_ARCH_CDNA4:
  case ROCJITSU_CODE_ARCH_RDNA1:
  case ROCJITSU_CODE_ARCH_RDNA2:
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
  case ROCJITSU_CODE_ARCH_RDNA4:
  case ROCJITSU_CODE_ARCH_GFX1250:
    switch (op) {
    case ScalarSop2Op::AddU32:
      return 0;
    case ScalarSop2Op::AddI32:
      return 2;
    case ScalarSop2Op::AddcU32:
      return 4;
    }
  case ROCJITSU_CODE_ARCH_RV32I:
  case ROCJITSU_CODE_ARCH_RV64I:
  case ROCJITSU_CODE_ARCH_NUM_ARCHS:
    return std::nullopt;
  }
  return std::nullopt;
}

/// @brief Match a scalar PC instruction and return its scalar register pair.
///
/// @details The low register of the PC pair is encoded as SDST for getpc and
/// SSRC0 for setpc/swappc. The matcher checks instruction size, SOP1 encoding
/// class, and the opcode selected by @ref scalar_pc_opcode for @p arch.
[[nodiscard]] std::optional<uint16_t> scalar_pc_sreg(rj_code_arch_t arch, const Instruction &inst,
                                                     uint32_t word, ScalarPcOp op) {
  if (inst.size() != sizeof(uint32_t))
    return std::nullopt;
  if ((word >> 23) != kSop1EncodingPrefix)
    return std::nullopt;
  auto opcode = scalar_pc_opcode(arch, op);
  if (!opcode || ((word >> 8) & 0xffu) != *opcode)
    return std::nullopt;
  if (op == ScalarPcOp::GetPc64)
    return static_cast<uint16_t>((word >> 16) & 0x7fu);
  return static_cast<uint16_t>(word & 0xffu);
}

/// @brief Match a literal-to-scalar SOP2 instruction.
///
/// @details Returns true only when @p inst is a two-word SOP2 instruction with
/// the requested opcode, destination, source register, and a literal second
/// source. The literal payload is returned through @p literal.
[[nodiscard]] bool sop2_literal_to_sreg(const Instruction &inst, uint32_t word,
                                        uint32_t literal_word, uint32_t opcode, uint16_t sdst,
                                        uint16_t ssrc0, uint32_t &literal) {
  if (inst.size() != 2 * sizeof(uint32_t))
    return false;
  if ((word >> 30) != kSop2EncodingPrefix)
    return false;
  if (((word >> 23) & 0x7fu) != opcode)
    return false;
  if (((word >> 16) & 0x7fu) != sdst)
    return false;
  if (((word >> 8) & 0xffu) != 255u)
    return false;
  if ((word & 0xffu) != ssrc0)
    return false;
  literal = literal_word;
  return true;
}

/// @brief Match a literal-plus-inline SOP2 instruction writing an SGPR.
///
/// @details Used for Tensile's temp-register PC builder variant, where the
/// temporary low-half add is encoded as literal plus inline constant.
[[nodiscard]] bool sop2_literal_inline_to_sreg(const Instruction &inst, uint32_t word,
                                               uint32_t literal_word, uint32_t opcode,
                                               uint16_t sdst, uint16_t inline_src1,
                                               uint32_t &literal) {
  if (inst.size() != 2 * sizeof(uint32_t))
    return false;
  if ((word >> 30) != kSop2EncodingPrefix)
    return false;
  if (((word >> 23) & 0x7fu) != opcode)
    return false;
  if (((word >> 16) & 0x7fu) != sdst)
    return false;
  if (((word >> 8) & 0xffu) != inline_src1)
    return false;
  if ((word & 0xffu) != 255u)
    return false;
  literal = literal_word;
  return true;
}

/// @brief Match a scalar-plus-inline SOP2 instruction writing an SGPR.
[[nodiscard]] bool sop2_sreg_inline_to_sreg(const Instruction &inst, uint32_t word, uint32_t opcode,
                                            uint16_t sdst, uint16_t ssrc0, uint16_t inline_src1) {
  if (inst.size() != sizeof(uint32_t))
    return false;
  if ((word >> 30) != kSop2EncodingPrefix)
    return false;
  if (((word >> 23) & 0x7fu) != opcode)
    return false;
  if (((word >> 16) & 0x7fu) != sdst)
    return false;
  if (((word >> 8) & 0xffu) != inline_src1)
    return false;
  return (word & 0xffu) == ssrc0;
}

} // namespace

bool s_setpc_from_sreg(rj_code_arch_t arch, const Instruction &inst, uint32_t word,
                       uint16_t ssrc0) {
  auto actual = scalar_pc_sreg(arch, inst, word, ScalarPcOp::SetPc64);
  return actual && *actual == ssrc0;
}

std::optional<uint16_t> s_call_sdst(const Instruction &inst, uint32_t word) {
  if (inst.size() != sizeof(uint32_t))
    return std::nullopt;
  if ((inst.flags() & INDIRECT_CALL) == 0 || !inst.branch_offset_bytes())
    return std::nullopt;
  return static_cast<uint16_t>((word >> 16) & 0x7fu);
}

std::vector<IndirectCallFixup>
recover_static_indirect_call_targets(std::span<const Instruction *const> insts,
                                     std::span<const uint64_t> offsets,
                                     std::span<const uint8_t> text, rj_code_arch_t arch) {
  std::vector<IndirectCallFixup> fixups;
  const auto add_u32_opcode = scalar_sop2_opcode(arch, ScalarSop2Op::AddU32);
  const auto add_i32_opcode = scalar_sop2_opcode(arch, ScalarSop2Op::AddI32);
  const auto addc_u32_opcode = scalar_sop2_opcode(arch, ScalarSop2Op::AddcU32);
  if (!add_u32_opcode || !add_i32_opcode || !addc_u32_opcode)
    return fixups;

  // Static call sequences materialize a PC in a scalar register pair and then
  // branch through that register later. The old recovery loop rescanned the
  // whole suffix after every matched s_getpc pattern, which becomes quadratic
  // on large Tensile code objects. Build one source-register index up front so
  // each pattern can jump directly to the matching s_setpc/s_swappc sites.
  std::array<std::vector<size_t>, 256> indirect_call_indices_by_sreg;
  for (size_t i = 0; i < insts.size(); ++i) {
    const uint32_t word = text_word_at(text, offsets[i]);
    if (auto ssrc0 = scalar_pc_sreg(arch, *insts[i], word, ScalarPcOp::SwapPc64))
      indirect_call_indices_by_sreg[*ssrc0].push_back(i);
    if (auto ssrc0 = scalar_pc_sreg(arch, *insts[i], word, ScalarPcOp::SetPc64))
      indirect_call_indices_by_sreg[*ssrc0].push_back(i);
  }

  for (size_t i = 0; i + 3 < insts.size(); ++i) {
    auto sdst =
        scalar_pc_sreg(arch, *insts[i], text_word_at(text, offsets[i]), ScalarPcOp::GetPc64);
    if (!sdst || *sdst >= 127)
      continue;

    uint32_t lo = 0;
    uint32_t hi = 0;
    uint64_t recovery_begin_offset = offsets[i + 1];
    uint64_t recovery_end_offset = offsets[i + 2] + insts[i + 2]->size();
    bool matched_static_target = false;
    constexpr uint16_t kInlineInt0 = 128;
    if (sop2_literal_to_sreg(*insts[i + 1], text_word_at(text, offsets[i + 1]),
                             text_word_at(text, offsets[i + 1] + sizeof(uint32_t)), *add_u32_opcode,
                             *sdst, *sdst, lo)) {
      if (sop2_literal_to_sreg(*insts[i + 2], text_word_at(text, offsets[i + 2]),
                               text_word_at(text, offsets[i + 2] + sizeof(uint32_t)),
                               *addc_u32_opcode, static_cast<uint16_t>(*sdst + 1),
                               static_cast<uint16_t>(*sdst + 1), hi)) {
        matched_static_target = true;
      } else if (sop2_sreg_inline_to_sreg(*insts[i + 2], text_word_at(text, offsets[i + 2]),
                                          *addc_u32_opcode, static_cast<uint16_t>(*sdst + 1),
                                          static_cast<uint16_t>(*sdst + 1), kInlineInt0)) {
        hi = 0;
        matched_static_target = true;
      }
    }
    if (!matched_static_target && i + 4 < insts.size()) {
      // Tensile emits static skips as:
      //   s_getpc_b64 s[pc:pc+1]
      //   s_add_i32 temp, literal, 4
      //   s_add_u32 pc, pc, temp
      //   s_addc_u32 pc+1, pc+1, 0
      //   s_setpc_b64 s[pc:pc+1]
      uint32_t temp_literal = 0;
      constexpr uint16_t kInlineInt4 = 132;
      const auto temp_sdst =
          static_cast<uint16_t>((text_word_at(text, offsets[i + 1]) >> 16) & 0x7fu);
      if (!sop2_literal_inline_to_sreg(*insts[i + 1], text_word_at(text, offsets[i + 1]),
                                       text_word_at(text, offsets[i + 1] + sizeof(uint32_t)),
                                       *add_i32_opcode, temp_sdst, kInlineInt4, temp_literal))
        continue;
      if (!sop2_sreg_inline_to_sreg(*insts[i + 2], text_word_at(text, offsets[i + 2]),
                                    *add_u32_opcode, *sdst, *sdst, temp_sdst))
        continue;
      if (!sop2_sreg_inline_to_sreg(*insts[i + 3], text_word_at(text, offsets[i + 3]),
                                    *addc_u32_opcode, static_cast<uint16_t>(*sdst + 1),
                                    static_cast<uint16_t>(*sdst + 1), kInlineInt0))
        continue;

      lo = temp_literal + 4u;
      hi = 0;
      recovery_begin_offset = offsets[i + 1];
      recovery_end_offset = offsets[i + 3] + insts[i + 3]->size();
      matched_static_target = true;
    }
    if (!matched_static_target) {
      continue;
    }

    const auto &call_indices = indirect_call_indices_by_sreg[*sdst];
    auto first_call = std::ranges::lower_bound(call_indices, i + 3, std::less<>{},
                                               [](size_t index) { return index; });
    if (first_call == call_indices.end())
      continue;

    const int64_t addend = static_cast<int64_t>((static_cast<uint64_t>(hi) << 32) | lo);
    const int64_t base = static_cast<int64_t>(offsets[i] + insts[i]->size());
    const int64_t target = base + addend;
    if (target < 0 || static_cast<uint64_t>(target) >= text.size())
      continue;

    for (auto call_it = first_call; call_it != call_indices.end(); ++call_it) {
      const size_t call_index = *call_it;
      fixups.push_back({.source_getpc_offset = offsets[i],
                        .source_recovery_begin_offset = recovery_begin_offset,
                        .source_recovery_end_offset = recovery_end_offset,
                        .source_call_offset = offsets[call_index],
                        .source_target_offset = static_cast<uint64_t>(target),
                        .source_call_sreg = *sdst});
    }
  }
  return fixups;
}

std::optional<IndirectCallFixup>
recover_carried_indirect_call_target(const KernelTextLayout &layout, const Instruction &inst,
                                     uint32_t word, uint64_t source_call_offset,
                                     rj_code_arch_t arch) {
  auto ssrc0 = scalar_pc_sreg(arch, inst, word, ScalarPcOp::SwapPc64);
  if (!ssrc0)
    ssrc0 = scalar_pc_sreg(arch, inst, word, ScalarPcOp::SetPc64);
  if (!ssrc0)
    return std::nullopt;

  const IndirectCallFixup *latest = nullptr;
  for (const IndirectCallFixup &fixup : layout.indirect_call_fixups) {
    if (fixup.source_call_sreg != *ssrc0 || fixup.source_call_offset >= source_call_offset)
      continue;
    if (latest == nullptr || fixup.source_call_offset > latest->source_call_offset)
      latest = &fixup;
  }
  if (!latest)
    return std::nullopt;

  IndirectCallFixup carried = *latest;
  carried.source_call_offset = source_call_offset;
  return carried;
}

bool recovered_direct_call_return(const KernelTextLayout &layout, const Instruction &inst,
                                  uint32_t word, uint64_t source_offset, rj_code_arch_t arch) {
  auto ssrc0 = scalar_pc_sreg(arch, inst, word, ScalarPcOp::SetPc64);
  if (!ssrc0)
    return false;

  return std::ranges::any_of(layout.direct_call_returns, [&](const DirectCallReturn &call) {
    return call.return_sreg == *ssrc0 && call.source_call_offset < source_offset;
  });
}

} // namespace rocjitsu
