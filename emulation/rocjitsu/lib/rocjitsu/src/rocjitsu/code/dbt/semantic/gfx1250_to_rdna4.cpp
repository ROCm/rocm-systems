// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file semantic/gfx1250_to_rdna4.cpp
/// @brief gfx1250-to-RDNA4 handwritten semantic expansion rules.

#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/dbt/generated/encoding_gfx1250_to_rdna4.h"
#include "rocjitsu/code/dbt/hazard_tracker.h"
#include "rocjitsu/code/dbt/semantic/rules.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/machine_insts.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/vgpr_msb.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {

constexpr uint32_t kVop3Encoding = 0x35u;
constexpr uint8_t kNullSgpr = 124;
constexpr uint16_t kGfx1250SrcFlatScratchBaseLo = 230;
constexpr uint16_t kGfx1250SrcFlatScratchBaseHi = 231;
constexpr uint16_t kRdna4SrcPrivateBase = 237;
constexpr uint32_t kK128Fp8BorrowedVgprCount = 5;
constexpr uint32_t kPrivateBorrowedVgprCount = 21;
constexpr uint32_t kPrivateBorrowScratchBytes = kPrivateBorrowedVgprCount * sizeof(uint32_t);

[[nodiscard]] constexpr std::pair<uint32_t, uint32_t>
build_vop3p(uint8_t op, uint8_t vdst, uint16_t src0, uint16_t src1, uint16_t src2, uint8_t neg = 0,
            bool clamp = false, uint8_t neg_hi = 0) {
  const uint32_t w0 = static_cast<uint32_t>(vdst) | ((neg_hi & 0x7u) << 8) | (1u << 14) |
                      (static_cast<uint32_t>(clamp) << 15) |
                      (static_cast<uint32_t>(op & 0x7F) << 16) | (0xCCu << 24);
  const uint32_t w1 = (src0 & 0x1FF) | ((src1 & 0x1FF) << 9) | ((src2 & 0x1FF) << 18) | (3u << 27) |
                      ((neg & 0x7u) << 29);
  return {w0, w1};
}

[[nodiscard]] constexpr uint32_t build_vop2(uint8_t op, uint8_t vdst, uint16_t src0,
                                            uint8_t vsrc1) {
  return (src0 & 0x1FFu) | ((vsrc1 & 0xFFu) << 9) | ((vdst & 0xFFu) << 17) | ((op & 0x3Fu) << 25);
}

[[nodiscard]] constexpr uint32_t build_vop1(uint8_t op, uint8_t vdst, uint16_t src0) {
  return (src0 & 0x1FFu) | ((op & 0x7Fu) << 9) | ((vdst & 0xFFu) << 17) | (0x3Fu << 25);
}

[[nodiscard]] constexpr uint16_t scalar_negative_inline_i32(int16_t value) {
  return static_cast<uint16_t>(192 - value);
}

[[nodiscard]] constexpr uint16_t build_hwreg(uint8_t reg_id, uint8_t offset, uint8_t size) {
  return static_cast<uint16_t>((reg_id & 0x3Fu) | ((offset & 0x1Fu) << 6) |
                               (((size - 1u) & 0x1Fu) << 11));
}

[[nodiscard]] constexpr uint32_t build_sopk(uint8_t op, uint16_t simm16, uint8_t sdst = 0) {
  return 0xB0000000u | (simm16 & 0xFFFFu) | ((sdst & 0x7Fu) << 16) | ((op & 0x1Fu) << 23);
}

[[nodiscard]] constexpr uint32_t build_sopc(uint8_t op, uint16_t ssrc0, uint16_t ssrc1) {
  return 0xBF000000u | ((op & 0x7Fu) << 16) | ((ssrc1 & 0xFFu) << 8) | (ssrc0 & 0xFFu);
}

[[nodiscard]] constexpr std::optional<uint16_t> scalar_inline_i32(int32_t value) {
  if (value >= 0 && value <= 64)
    return scalar_positive_inline_u32(static_cast<uint16_t>(value));
  if (value >= -16 && value <= -1)
    return scalar_negative_inline_i32(static_cast<int16_t>(value));
  return std::nullopt;
}

void append_vop1(std::vector<uint32_t> &words, uint8_t op, uint8_t vdst, uint16_t src0,
                 std::optional<uint32_t> literal = std::nullopt) {
  words.push_back(build_vop1(op, vdst, src0));
  if (literal && src0 == 255)
    words.push_back(*literal);
}

void append_vop2(std::vector<uint32_t> &words, uint8_t op, uint8_t vdst, uint16_t src0,
                 uint8_t vsrc1, std::optional<uint32_t> literal = std::nullopt) {
  words.push_back(build_vop2(op, vdst, src0, vsrc1));
  if (literal && src0 == 255)
    words.push_back(*literal);
}

[[nodiscard]] constexpr std::pair<uint32_t, uint32_t>
build_vop3_mod(uint16_t op, uint8_t vdst, uint16_t src0, uint16_t src1, uint16_t src2 = 0,
               uint8_t abs = 0, uint8_t opsel = 0, bool clamp = false, uint8_t omod = 0,
               uint8_t neg = 0) {
  const uint32_t w0 = (vdst & 0xFFu) | ((abs & 0x7u) << 8) | ((opsel & 0xFu) << 11) |
                      ((clamp ? 1u : 0u) << 15) | ((op & 0x3FFu) << 16) | (kVop3Encoding << 26);
  const uint32_t w1 = (src0 & 0x1FFu) | ((src1 & 0x1FFu) << 9) | ((src2 & 0x1FFu) << 18) |
                      ((omod & 0x3u) << 27) | ((neg & 0x7u) << 29);
  return {w0, w1};
}

[[nodiscard]] constexpr std::pair<uint32_t, uint32_t>
build_vop3(uint16_t op, uint8_t vdst, uint16_t src0, uint16_t src1, uint16_t src2 = 0) {
  return build_vop3_mod(op, vdst, src0, src1, src2);
}

void append_vop3(std::vector<uint32_t> &words, uint16_t op, uint8_t vdst, uint16_t src0,
                 uint16_t src1, uint16_t src2 = 0, std::optional<uint32_t> literal = std::nullopt) {
  auto [w0, w1] = build_vop3(op, vdst, src0, src1, src2);
  words.push_back(w0);
  words.push_back(w1);
  if (literal && (src0 == 255 || src1 == 255 || src2 == 255))
    words.push_back(*literal);
}

void append_vop3_mod(std::vector<uint32_t> &words, uint16_t op, uint8_t vdst, uint16_t src0,
                     uint16_t src1, uint16_t src2, uint8_t opsel,
                     std::optional<uint32_t> literal = std::nullopt) {
  const auto [w0, w1] = build_vop3_mod(op, vdst, src0, src1, src2, 0, opsel);
  words.push_back(w0);
  words.push_back(w1);
  if (literal && (src0 == 255 || src1 == 255 || src2 == 255))
    words.push_back(*literal);
}

void append_v_readlane_b32(std::vector<uint32_t> &words, uint8_t sdst, uint8_t src, uint8_t lane) {
  constexpr uint16_t kOpVReadlaneB32 = 0x360;
  auto [w0, w1] = build_vop3(kOpVReadlaneB32, sdst, static_cast<uint16_t>(256u + src),
                             scalar_positive_inline_u32(lane));
  words.push_back(w0);
  words.push_back(w1);
}

[[nodiscard]] constexpr std::pair<uint32_t, uint32_t> build_vop3_sdst(uint16_t op, uint8_t vdst,
                                                                      uint8_t sdst, uint16_t src0,
                                                                      uint16_t src1,
                                                                      uint16_t src2 = 0) {
  const uint32_t w0 =
      (vdst & 0xFFu) | ((sdst & 0x7Fu) << 8) | ((op & 0x3FFu) << 16) | (kVop3Encoding << 26);
  const uint32_t w1 = (src0 & 0x1FFu) | ((src1 & 0x1FFu) << 9) | ((src2 & 0x1FFu) << 18);
  return {w0, w1};
}

[[nodiscard]] constexpr std::pair<uint32_t, uint32_t> build_ds_bpermute(uint8_t vdst, uint8_t vaddr,
                                                                        uint8_t vdata) {
  constexpr uint32_t kDsW0 = (0xB3u << 18) | (0x36u << 26);
  return {kDsW0, static_cast<uint32_t>(vaddr) | (static_cast<uint32_t>(vdata) << 8) |
                     (static_cast<uint32_t>(vdst) << 24)};
}

[[nodiscard]] constexpr std::pair<uint32_t, uint32_t>
build_ds_bpermute_fi(uint8_t vdst, uint8_t vaddr, uint8_t vdata) {
  constexpr uint32_t kDsW0 = (0xCDu << 18) | (0x36u << 26);
  return {kDsW0, static_cast<uint32_t>(vaddr) | (static_cast<uint32_t>(vdata) << 8) |
                     (static_cast<uint32_t>(vdst) << 24)};
}

[[nodiscard]] constexpr std::array<uint32_t, 3> build_scratch_store_b32(uint8_t vdata,
                                                                        uint32_t offset) {
  return {0xED06807Cu, static_cast<uint32_t>(vdata) << 23, (offset & 0xFFFFFFu) << 8};
}

[[nodiscard]] constexpr std::array<uint32_t, 3> build_scratch_load_b32(uint8_t vdst,
                                                                       uint32_t offset) {
  return {0xED05007Cu, static_cast<uint32_t>(vdst), (offset & 0xFFFFFFu) << 8};
}

[[nodiscard]] constexpr std::pair<uint32_t, uint32_t> build_s_mov_b32_lit(uint8_t sdst,
                                                                          uint32_t literal) {
  gfx1250::Sop1MachineInst s{};
  s.encoding = 0x17D;
  s.op = 0;
  s.sdst = sdst & 0x7F;
  s.ssrc0 = 0xFF;
  return {std::bit_cast<uint32_t>(s), literal};
}

[[nodiscard]] constexpr uint32_t build_sop1_mov_b32(uint8_t sdst, uint16_t ssrc0) {
  gfx1250::Sop1MachineInst s{};
  s.encoding = 0x17D;
  s.op = 0;
  s.sdst = sdst & 0x7F;
  s.ssrc0 = ssrc0 & 0xFF;
  return std::bit_cast<uint32_t>(s);
}

[[nodiscard]] constexpr uint32_t build_s_mov_b64(uint8_t sdst, uint16_t ssrc0) {
  gfx1250::Sop1MachineInst s{};
  s.encoding = 0x17D;
  s.op = 1;
  s.sdst = sdst & 0x7F;
  s.ssrc0 = ssrc0 & 0xFF;
  return std::bit_cast<uint32_t>(s);
}

[[nodiscard]] constexpr uint32_t build_s_or_b32(uint8_t sdst, uint16_t ssrc0, uint16_t ssrc1) {
  constexpr uint8_t kOpSOrB32 = 24;
  return pack_sop2(kOpSOrB32, sdst, ssrc0, ssrc1);
}

void append_sop2_b32_u32(std::vector<uint32_t> &words, uint8_t op, uint8_t sdst, uint16_t ssrc0,
                         uint32_t value) {
  if (auto inline_src = scalar_inline_i32(static_cast<int32_t>(value))) {
    words.push_back(pack_sop2(op, sdst, ssrc0, *inline_src));
    return;
  }
  words.push_back(pack_sop2(op, sdst, ssrc0, 255));
  words.push_back(value);
}

void append_s_and_b32_lit(std::vector<uint32_t> &words, uint8_t sdst, uint16_t ssrc0,
                          uint32_t literal) {
  constexpr uint8_t kOpSAndB32 = 22;
  words.push_back(pack_sop2(kOpSAndB32, sdst, ssrc0, 255));
  words.push_back(literal);
}

void append_s_cmp_eq_u32_lit(std::vector<uint32_t> &words, uint16_t ssrc0, uint32_t literal) {
  constexpr uint8_t kOpSCmpEqU32 = 6;
  words.push_back(build_sopc(kOpSCmpEqU32, ssrc0, 255));
  words.push_back(literal);
}

[[nodiscard]] constexpr uint16_t vgpr_msb_mode_hwreg() {
  return build_hwreg(1, amdgpu::VGPR_MSB_MODE_SHIFT, 8);
}

void append_s_get_vgpr_msb_mode(std::vector<uint32_t> &words, uint8_t sdst) {
  constexpr uint8_t kOpSGetregB32 = 17;
  words.push_back(build_sopk(kOpSGetregB32, vgpr_msb_mode_hwreg(), sdst));
}

void append_s_set_vgpr_msb_mode_from_sgpr(std::vector<uint32_t> &words, uint8_t ssrc) {
  constexpr uint8_t kOpSSetregB32 = 18;
  words.push_back(build_sopk(kOpSSetregB32, vgpr_msb_mode_hwreg(), ssrc));
}

void append_s_set_vgpr_msb_mode(std::vector<uint32_t> &words, uint8_t mode) {
  constexpr uint8_t kOpSSetregImm32B32 = 19;
  const uint32_t mode_literal = amdgpu::set_vgpr_msb_to_mode_layout(mode);
  words.push_back(build_sopk(kOpSSetregImm32B32, vgpr_msb_mode_hwreg()));
  words.push_back(mode_literal);
}

[[nodiscard]] constexpr uint32_t build_s_or_b64(uint8_t sdst, uint16_t ssrc0, uint16_t ssrc1) {
  constexpr uint8_t kOpSOrB64 = 25;
  return pack_sop2(kOpSOrB64, sdst, ssrc0, ssrc1);
}

[[nodiscard]] std::optional<uint16_t> pair_hi_src(uint16_t src0) {
  if (src0 == kGfx1250SrcFlatScratchBaseLo)
    return kGfx1250SrcFlatScratchBaseHi;
  if (src0 < 128 || src0 >= 256)
    return static_cast<uint16_t>(src0 + 1);

  // Inline constants do not naturally represent a 64-bit source pair. Support
  // the common integer-zero case, but leave anything else to an explicit
  // lowering when it appears in real inputs.
  if (src0 == scalar_positive_inline_u32(0))
    return scalar_positive_inline_u32(0);
  return std::nullopt;
}

[[nodiscard]] std::optional<uint16_t> pair_hi_src_sign_extended_inline(uint16_t src0) {
  if (src0 == kGfx1250SrcFlatScratchBaseLo)
    return kGfx1250SrcFlatScratchBaseHi;
  if (src0 < 128 || src0 >= 256)
    return static_cast<uint16_t>(src0 + 1);

  if (src0 >= scalar_positive_inline_u32(0) && src0 <= scalar_positive_inline_u32(64))
    return scalar_positive_inline_u32(0);
  if (src0 >= scalar_negative_inline_i32(-1) && src0 <= scalar_negative_inline_i32(-16))
    return scalar_negative_inline_i32(-1);
  return std::nullopt;
}

[[nodiscard]] std::optional<uint32_t> simm32_literal_word(const Instruction &inst,
                                                          uint8_t operand_index);

[[nodiscard]] std::optional<uint64_t> simm64_literal_value(const Instruction &inst,
                                                           uint8_t operand_index);

struct Vector64SourceParts {
  uint16_t lo = 0;
  uint16_t hi = 0;
  std::optional<uint32_t> lo_literal;
  std::optional<uint32_t> hi_literal;
};

[[nodiscard]] std::optional<Vector64SourceParts>
decode_vector64_source(const Instruction &inst, uint16_t src, uint8_t operand_index) {
  if (src == 255) {
    const auto literal = simm32_literal_word(inst, operand_index);
    if (!literal)
      return std::nullopt;
    return Vector64SourceParts{
        255,
        (*literal & 0x8000'0000u) ? scalar_negative_inline_i32(-1) : scalar_positive_inline_u32(0),
        *literal,
        std::nullopt,
    };
  }

  if (src == 254) {
    const auto literal = simm64_literal_value(inst, operand_index);
    if (!literal)
      return std::nullopt;
    return Vector64SourceParts{
        255,
        255,
        static_cast<uint32_t>(*literal),
        static_cast<uint32_t>(*literal >> 32u),
    };
  }

  const auto hi = pair_hi_src_sign_extended_inline(src);
  if (!hi)
    return std::nullopt;
  return Vector64SourceParts{src, *hi, std::nullopt, std::nullopt};
}

[[nodiscard]] constexpr std::optional<uint8_t> vgpr_index(uint16_t src) {
  if (src < 256 || src >= 512)
    return std::nullopt;
  return static_cast<uint8_t>(src - 256);
}

[[nodiscard]] constexpr bool scalar_register_src(uint16_t src) { return src < 128; }

void append_set_exec_lo_mask(std::vector<uint32_t> &words, uint32_t mask);
void append_restore_exec(std::vector<uint32_t> &words, uint8_t exec_save);
void append_wait_valu_vgpr(std::vector<uint32_t> &words);
void append_scratch_store_b32(std::vector<uint32_t> &words, uint8_t vdata, uint32_t offset);
void append_scratch_load_b32(std::vector<uint32_t> &words, uint8_t vdst, uint32_t offset);
void append_merge_b16_result(std::vector<uint32_t> &words, uint8_t vdst, uint8_t result,
                             bool dst_high);
void append_f32_to_bf16_rne(std::vector<uint32_t> &words, uint8_t result, uint8_t tmp,
                            uint8_t src_vgpr);

[[nodiscard]] bool low_write_clobbers_high_source(uint8_t vdst, uint16_t src_hi) {
  const auto src_hi_vgpr = vgpr_index(src_hi);
  return src_hi_vgpr && vdst == *src_hi_vgpr;
}

[[nodiscard]] bool overlaps_vdst_pair(uint16_t vgpr, uint8_t vdst) {
  return vgpr == vdst || vgpr == static_cast<uint16_t>(vdst + 1u);
}

[[nodiscard]] bool overlaps_vgpr_run(uint16_t run_base, uint16_t count, uint8_t vgpr) {
  return vgpr >= run_base && vgpr < run_base + count;
}

[[nodiscard]] bool overlaps_vgpr_runs(uint16_t lhs_base, uint16_t lhs_count, uint16_t rhs_base,
                                      uint16_t rhs_count) {
  return lhs_base < rhs_base + rhs_count && rhs_base < lhs_base + lhs_count;
}

std::optional<uint16_t> find_free_vgpr_run_away_from_dst(const Instruction &inst,
                                                         const LivenessAnalysis &liveness,
                                                         uint16_t count, uint8_t vdst) {
  uint16_t search_start = 0;
  while (true) {
    auto tmp_base = liveness.find_free_run(&inst, count, search_start);
    if (!tmp_base || *tmp_base + count - 1u > 255u)
      return std::nullopt;
    bool overlaps = false;
    for (uint16_t i = 0; i < count; ++i)
      overlaps |= overlaps_vdst_pair(static_cast<uint16_t>(*tmp_base + i), vdst);
    if (!overlaps)
      return tmp_base;
    search_start = static_cast<uint16_t>(*tmp_base + 1u);
  }
}

std::optional<uint16_t> find_free_vgpr_run_avoiding(const Instruction &inst,
                                                    const LivenessAnalysis &liveness,
                                                    uint16_t count,
                                                    const std::vector<uint8_t> &avoid) {
  uint16_t search_start = 0;
  while (true) {
    auto tmp_base = liveness.find_free_run(&inst, count, search_start);
    if (!tmp_base || *tmp_base + count - 1u > 255u)
      return std::nullopt;
    bool overlaps = false;
    for (uint8_t vgpr : avoid)
      overlaps |= overlaps_vgpr_run(*tmp_base, count, vgpr);
    if (!overlaps)
      return tmp_base;
    search_start = static_cast<uint16_t>(*tmp_base + 1u);
  }
}

std::optional<uint16_t> find_aligned_free_vgpr_run_avoiding(const Instruction &inst,
                                                            const LivenessAnalysis &liveness,
                                                            uint16_t count, uint16_t alignment,
                                                            const std::vector<uint8_t> &avoid,
                                                            uint16_t search_start = 0) {
  while (true) {
    auto tmp_base = liveness.find_free_run(&inst, count, search_start);
    if (!tmp_base || *tmp_base + count - 1u > 255u)
      return std::nullopt;
    bool overlaps = false;
    for (uint8_t vgpr : avoid)
      overlaps |= overlaps_vgpr_run(*tmp_base, count, vgpr);
    if (!overlaps && (*tmp_base % alignment) == 0)
      return tmp_base;
    search_start = static_cast<uint16_t>(*tmp_base + 1u);
  }
}

std::optional<uint8_t> find_borrowable_low_vgpr_run(uint8_t count, uint8_t alignment,
                                                    const std::vector<uint8_t> &avoid) {
  for (uint16_t base = 0; base + count <= 128u; ++base) {
    if ((base % alignment) != 0)
      continue;
    bool overlaps = false;
    for (uint8_t vgpr : avoid)
      overlaps |= overlaps_vgpr_run(base, count, vgpr);
    if (!overlaps)
      return static_cast<uint8_t>(base);
  }
  return std::nullopt;
}

struct PrivateBorrowedVgprRun {
  uint8_t base = 0;
  uint8_t count = 0;
  uint32_t private_base = 0;
};

[[nodiscard]] std::optional<PrivateBorrowedVgprRun>
find_private_borrowed_vgpr_run(const LivenessAnalysis &liveness, uint8_t count, uint8_t alignment,
                               const std::vector<uint8_t> &avoid) {
  const auto private_base = liveness.private_spill_base();
  if (!private_base || liveness.private_spill_bytes() < kPrivateBorrowScratchBytes ||
      count > kPrivateBorrowedVgprCount)
    return std::nullopt;
  const auto base = find_borrowable_low_vgpr_run(count, alignment, avoid);
  if (!base || static_cast<uint16_t>(*base) + count > 256u)
    return std::nullopt;
  return PrivateBorrowedVgprRun{*base, count, *private_base};
}

void append_private_borrow_save(std::vector<uint32_t> &words,
                                const PrivateBorrowedVgprRun &borrow) {
  constexpr uint8_t kOpWaitLoadcnt = 64;
  constexpr uint8_t kOpWaitStorecnt = 65;
  constexpr uint8_t kOpWaitDscnt = 70;

  append_wait_valu_vgpr(words);
  words.push_back(pack_sopp(kOpWaitLoadcnt, 0));
  words.push_back(pack_sopp(kOpWaitDscnt, 0));
  append_set_exec_lo_mask(words, 0xFFFFFFFFu);
  for (uint8_t reg = 0; reg < borrow.count; ++reg) {
    append_scratch_store_b32(words, static_cast<uint8_t>(borrow.base + reg),
                             borrow.private_base + reg * sizeof(uint32_t));
  }
  words.push_back(pack_sopp(kOpWaitStorecnt, 0));
}

void append_private_borrow_restore(std::vector<uint32_t> &words,
                                   const PrivateBorrowedVgprRun &borrow, uint8_t exec_save) {
  constexpr uint8_t kOpWaitLoadcnt = 64;

  append_wait_valu_vgpr(words);
  append_set_exec_lo_mask(words, 0xFFFFFFFFu);
  for (uint8_t reg = 0; reg < borrow.count; ++reg) {
    append_scratch_load_b32(words, static_cast<uint8_t>(borrow.base + reg),
                            borrow.private_base + reg * sizeof(uint32_t));
  }
  words.push_back(pack_sopp(kOpWaitLoadcnt, 0));
  append_restore_exec(words, exec_save);
  append_wait_valu_vgpr(words);
}

std::optional<uint16_t> find_free_sgpr_avoiding(const Instruction &inst,
                                                const LivenessAnalysis &liveness,
                                                const std::vector<uint8_t> &avoid,
                                                uint16_t search_start = 0) {
  while (true) {
    auto sgpr = liveness.find_free_sgpr(&inst, search_start);
    if (!sgpr || *sgpr > 105)
      return std::nullopt;
    if (std::find(avoid.begin(), avoid.end(), static_cast<uint8_t>(*sgpr)) == avoid.end())
      return sgpr;
    search_start = static_cast<uint16_t>(*sgpr + 1u);
  }
}

template <size_t Count>
std::optional<std::array<uint8_t, Count>> find_free_sgprs_avoiding(const Instruction &inst,
                                                                   const LivenessAnalysis &liveness,
                                                                   std::vector<uint8_t> avoid) {
  std::array<uint8_t, Count> result{};
  for (size_t index = 0; index < Count; ++index) {
    const auto sgpr = find_free_sgpr_avoiding(inst, liveness, avoid);
    if (!sgpr)
      return std::nullopt;
    result[index] = static_cast<uint8_t>(*sgpr);
    avoid.push_back(result[index]);
  }
  return result;
}

std::optional<uint16_t> find_free_sgpr_pair_avoiding(const Instruction &inst,
                                                     const LivenessAnalysis &liveness,
                                                     const std::vector<uint8_t> &avoid,
                                                     uint16_t search_start = 0) {
  while (true) {
    auto sgpr = liveness.find_free_sgpr_pair(&inst, search_start);
    if (!sgpr || *sgpr > 124)
      return std::nullopt;
    bool overlaps = false;
    for (uint8_t avoid_sgpr : avoid)
      overlaps |= avoid_sgpr == *sgpr || avoid_sgpr == *sgpr + 1u;
    if (!overlaps)
      return sgpr;
    search_start = static_cast<uint16_t>(*sgpr + 2u);
  }
}

void add_avoid_vgpr(std::vector<uint8_t> &avoid, uint8_t vgpr) {
  if (std::find(avoid.begin(), avoid.end(), vgpr) == avoid.end())
    avoid.push_back(vgpr);
}

void add_avoid_vgpr_run(std::vector<uint8_t> &avoid, uint8_t base, uint8_t count) {
  for (uint8_t i = 0; i < count; ++i)
    add_avoid_vgpr(avoid, static_cast<uint8_t>(base + i));
}

void add_avoid_src_vgpr(std::vector<uint8_t> &avoid, uint16_t src) {
  if (auto vgpr = vgpr_index(src))
    add_avoid_vgpr(avoid, *vgpr);
}

[[nodiscard]] std::optional<uint8_t> src_vgpr_base_for_run(uint16_t src, uint8_t count) {
  const auto base = vgpr_index(src);
  if (!base)
    return std::nullopt;
  if (static_cast<uint16_t>(*base) + count > 256u)
    return std::nullopt;
  return *base;
}

[[nodiscard]] bool scalar_inline_src_as_u32_word(uint16_t src) {
  if (src >= scalar_positive_inline_u32(0) && src <= scalar_positive_inline_u32(64))
    return true;
  return src >= scalar_negative_inline_i32(-1) && src <= scalar_negative_inline_i32(-16);
}

[[nodiscard]] constexpr bool scalar_inline_zero_src(uint16_t src) {
  return src == scalar_positive_inline_u32(0);
}

[[nodiscard]] bool scale_word_source_supported(uint16_t src) {
  return src_vgpr_base_for_run(src, 1).has_value() || scalar_inline_src_as_u32_word(src);
}

[[nodiscard]] bool dst_overlaps_wmma_ab_sources(uint8_t vdst, uint16_t src0, uint16_t src1,
                                                uint8_t src_count = 8) {
  const uint16_t dst_base = vdst;
  const auto src0_base = src_vgpr_base_for_run(src0, src_count);
  const auto src1_base = src_vgpr_base_for_run(src1, src_count);
  return (src0_base && overlaps_vgpr_runs(dst_base, 8, *src0_base, src_count)) ||
         (src1_base && overlaps_vgpr_runs(dst_base, 8, *src1_base, src_count));
}

void add_avoid_src_vgpr_run(std::vector<uint8_t> &avoid, uint16_t src, uint8_t count) {
  if (auto base = src_vgpr_base_for_run(src, count))
    add_avoid_vgpr_run(avoid, *base, count);
}

[[nodiscard]] bool source_pair_reads_vdst_pair(uint8_t vdst, uint16_t src_lo, uint16_t src_hi) {
  const auto low = vgpr_index(src_lo);
  const auto high = vgpr_index(src_hi);
  return (low && overlaps_vdst_pair(*low, vdst)) || (high && overlaps_vdst_pair(*high, vdst));
}

void add_avoid_source_pair_vgprs(std::vector<uint8_t> &avoid, uint16_t src_lo, uint16_t src_hi) {
  add_avoid_src_vgpr(avoid, src_lo);
  add_avoid_src_vgpr(avoid, src_hi);
}

[[nodiscard]] bool append_vop3_from_vector64_parts(std::vector<uint32_t> &words, uint16_t op,
                                                   uint8_t vdst, const Vector64SourceParts &src0,
                                                   bool src0_high, const Vector64SourceParts &src1,
                                                   bool src1_high) {
  const uint16_t encoded_src0 = src0_high ? src0.hi : src0.lo;
  const uint16_t encoded_src1 = src1_high ? src1.hi : src1.lo;
  const auto &src0_literal = src0_high ? src0.hi_literal : src0.lo_literal;
  const auto &src1_literal = src1_high ? src1.hi_literal : src1.lo_literal;

  std::optional<uint32_t> literal;
  if (encoded_src0 == 255) {
    if (!src0_literal)
      return false;
    literal = *src0_literal;
  }
  if (encoded_src1 == 255) {
    if (literal || !src1_literal)
      return false;
    literal = *src1_literal;
  }

  append_vop3(words, op, vdst, encoded_src0, encoded_src1, 0, literal);
  return true;
}

[[nodiscard]] bool append_v_mul_u64_low64(std::vector<uint32_t> &words, uint8_t out_lo,
                                          uint8_t out_hi, const Vector64SourceParts &src0,
                                          const Vector64SourceParts &src1) {
  constexpr uint16_t kOpMulLoU32 = 812;
  constexpr uint16_t kOpMulHiU32 = 813;
  constexpr uint16_t kOpAddNcU32 = 293;
  constexpr uint16_t kVgprSrcBase = 256;

  if (!append_vop3_from_vector64_parts(words, kOpMulHiU32, out_hi, src0, false, src1, false) ||
      !append_vop3_from_vector64_parts(words, kOpMulLoU32, out_lo, src0, true, src1, false))
    return false;
  words.push_back(build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4));
  append_vop3(words, kOpAddNcU32, out_hi, static_cast<uint16_t>(kVgprSrcBase + out_hi),
              static_cast<uint16_t>(kVgprSrcBase + out_lo));
  if (!append_vop3_from_vector64_parts(words, kOpMulLoU32, out_lo, src0, false, src1, true))
    return false;
  words.push_back(build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4));
  append_vop3(words, kOpAddNcU32, out_hi, static_cast<uint16_t>(kVgprSrcBase + out_hi),
              static_cast<uint16_t>(kVgprSrcBase + out_lo));
  return append_vop3_from_vector64_parts(words, kOpMulLoU32, out_lo, src0, false, src1, false);
}

[[nodiscard]] std::vector<uint32_t>
expand_v_mul_u64_high_scratch(uint8_t vdst, const Vector64SourceParts &src0,
                              const Vector64SourceParts &src1, const Instruction &inst,
                              const LivenessAnalysis &liveness) {
  const auto scratch_base_opt = liveness.high_vgpr_scratch_base();
  if (!scratch_base_opt || *scratch_base_opt > 254)
    return {};
  const auto mode_save_opt = liveness.find_free_sgpr(&inst);
  if (!mode_save_opt || *mode_save_opt > 105)
    return {};

  constexpr uint16_t kOpMulLoU32 = 812;
  constexpr uint16_t kOpMulHiU32 = 813;
  constexpr uint16_t kOpAddNcU32 = 293;
  constexpr uint16_t kVgprSrcBase = 256;
  constexpr uint8_t kModeSrc0High = 0x01;
  constexpr uint8_t kModeScratchAdd = 0x45;
  constexpr uint8_t kModeDstHigh = 0x40;
  constexpr uint8_t kOpMovB32 = 1;

  const uint8_t tmp_lo = static_cast<uint8_t>(*scratch_base_opt);
  const uint8_t tmp_hi = static_cast<uint8_t>(tmp_lo + 1u);
  const uint8_t mode_save = static_cast<uint8_t>(*mode_save_opt);

  std::vector<uint32_t> words;
  words.reserve(32);
  append_s_get_vgpr_msb_mode(words, mode_save);

  append_s_set_vgpr_msb_mode(words, kModeDstHigh);
  if (!append_vop3_from_vector64_parts(words, kOpMulHiU32, tmp_hi, src0, false, src1, false) ||
      !append_vop3_from_vector64_parts(words, kOpMulLoU32, tmp_lo, src0, true, src1, false))
    return {};

  append_s_set_vgpr_msb_mode(words, kModeScratchAdd);
  words.push_back(build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4));
  append_vop3(words, kOpAddNcU32, tmp_hi, static_cast<uint16_t>(kVgprSrcBase + tmp_hi),
              static_cast<uint16_t>(kVgprSrcBase + tmp_lo));

  append_s_set_vgpr_msb_mode(words, kModeDstHigh);
  if (!append_vop3_from_vector64_parts(words, kOpMulLoU32, tmp_lo, src0, false, src1, true))
    return {};

  append_s_set_vgpr_msb_mode(words, kModeScratchAdd);
  words.push_back(build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4));
  append_vop3(words, kOpAddNcU32, tmp_hi, static_cast<uint16_t>(kVgprSrcBase + tmp_hi),
              static_cast<uint16_t>(kVgprSrcBase + tmp_lo));

  append_s_set_vgpr_msb_mode(words, kModeDstHigh);
  if (!append_vop3_from_vector64_parts(words, kOpMulLoU32, tmp_lo, src0, false, src1, false))
    return {};

  words.push_back(build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4));
  append_s_set_vgpr_msb_mode(words, kModeSrc0High);
  append_vop1(words, kOpMovB32, vdst, static_cast<uint16_t>(kVgprSrcBase + tmp_lo));
  append_vop1(words, kOpMovB32, static_cast<uint8_t>(vdst + 1u),
              static_cast<uint16_t>(kVgprSrcBase + tmp_hi));
  append_s_set_vgpr_msb_mode_from_sgpr(words, mode_save);
  return words;
}

std::vector<uint32_t> expand_v_mul_u64(uint8_t vdst, const Vector64SourceParts &src0,
                                       const Vector64SourceParts &src1, const Instruction &inst,
                                       const LivenessAnalysis &liveness) {
  if (vdst > 254)
    return {};

  if (src0.lo > 511u || src1.lo > 511u || src0.hi > 511u || src1.hi > 511u)
    return {};

  const bool overlaps_dst_pair = source_pair_reads_vdst_pair(vdst, src0.lo, src0.hi) ||
                                 source_pair_reads_vdst_pair(vdst, src1.lo, src1.hi);

  std::vector<uint32_t> words;
  if (!overlaps_dst_pair) {
    words.reserve(15);
    if (!append_v_mul_u64_low64(words, vdst, static_cast<uint8_t>(vdst + 1u), src0, src1))
      return {};
    return words;
  }

  std::vector<uint8_t> avoid;
  add_avoid_vgpr_run(avoid, vdst, 2);
  add_avoid_source_pair_vgprs(avoid, src0.lo, src0.hi);
  add_avoid_source_pair_vgprs(avoid, src1.lo, src1.hi);
  const auto tmp_opt = find_free_vgpr_run_avoiding(inst, liveness, 2, avoid);
  if (!tmp_opt) {
    return expand_v_mul_u64_high_scratch(vdst, src0, src1, inst, liveness);
  }

  constexpr uint16_t kVgprSrcBase = 256;
  constexpr uint8_t kOpMovB32 = 1;
  const uint8_t tmp = static_cast<uint8_t>(*tmp_opt);
  words.reserve(19);
  if (!append_v_mul_u64_low64(words, tmp, static_cast<uint8_t>(tmp + 1u), src0, src1))
    return {};
  words.push_back(build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4));
  append_vop1(words, kOpMovB32, vdst, static_cast<uint16_t>(kVgprSrcBase + tmp));
  append_vop1(words, kOpMovB32, static_cast<uint8_t>(vdst + 1u),
              static_cast<uint16_t>(kVgprSrcBase + tmp + 1u));
  return words;
}

[[nodiscard]] uint16_t literal_or_inline_u32(uint32_t value, std::optional<uint32_t> &literal);

[[nodiscard]] constexpr uint16_t vgpr_src(uint8_t vgpr);

[[nodiscard]] bool append_materialize_b16_half(std::vector<uint32_t> &words, uint8_t tmp,
                                               uint16_t src, bool high_half,
                                               std::optional<uint32_t> literal_word);

void append_s_mov_b32_lit(std::vector<uint32_t> &words, uint8_t sdst, uint32_t literal);

void append_set_exec_lo_mask(std::vector<uint32_t> &words, uint32_t mask);

void append_set_exec_from_saved_xor16_mask(std::vector<uint32_t> &words, uint8_t exec_save);

void append_lane_id(std::vector<uint32_t> &words, uint8_t vaddr);

void append_v_mov_b32_broadcast(std::vector<uint32_t> &words, uint8_t vdst, uint16_t src,
                                uint8_t count);

void append_wait_valu_vgpr(std::vector<uint32_t> &words) {
  constexpr uint8_t kSoppWaitAlu = 8;
  words.push_back(pack_sopp(kSoppWaitAlu, kWaitAluDepctrVaVdst0));
}

void append_wait_salu_sgpr(std::vector<uint32_t> &words) {
  constexpr uint8_t kSoppWaitAlu = 8;
  words.push_back(pack_sopp(kSoppWaitAlu, kWaitAluDepctrSaSdst0));
}

void append_delay_salu_scc(std::vector<uint32_t> &words) {
  words.push_back(build_s_delay_alu(kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4));
}

[[nodiscard]] constexpr std::optional<uint8_t> scaled_vglobal_access_size(uint16_t op) {
  switch (op) {
  case 16: // global_load_u8
  case 17: // global_load_i8
  case 24: // global_store_b8
  case 30: // global_load_d16_u8
  case 31: // global_load_d16_i8
  case 33: // global_load_d16_hi_u8
  case 34: // global_load_d16_hi_i8
  case 36: // global_store_d16_hi_b8
    return 1;
  case 18: // global_load_u16
  case 19: // global_load_i16
  case 25: // global_store_b16
  case 32: // global_load_d16_b16
  case 35: // global_load_d16_hi_b16
  case 37: // global_store_d16_hi_b16
    return 2;
  case 20: // global_load_b32
  case 26: // global_store_b32
    return 4;
  case 21:  // global_load_b64
  case 27:  // global_store_b64
  case 88:  // global_load_tr8_b64
  case 115: // global_load_tr4_b64
    return 8;
  case 22:  // global_load_b96
  case 28:  // global_store_b96
  case 116: // global_load_tr6_b96
    return 12;
  case 23: // global_load_b128
  case 29: // global_store_b128
  case 87: // global_load_tr16_b128
    return 16;
  default:
    return std::nullopt;
  }
}

[[nodiscard]] constexpr std::optional<uint8_t> scaled_offset_shift_for_vglobal(uint16_t op) {
  const auto access_size = scaled_vglobal_access_size(op);
  if (!access_size || !std::has_single_bit(*access_size))
    return std::nullopt;
  return static_cast<uint8_t>(std::countr_zero(*access_size));
}

void append_set_exec_from_saved_mask(std::vector<uint32_t> &words, uint8_t exec_save,
                                     uint32_t mask) {
  constexpr uint8_t kExecLo = 126;
  append_s_and_b32_lit(words, kExecLo, exec_save, mask);
  append_s_mov_b32_lit(words, static_cast<uint8_t>(kExecLo + 1u), 0);
  append_wait_salu_sgpr(words);
}

void append_restore_exec(std::vector<uint32_t> &words, uint8_t exec_save) {
  constexpr uint8_t kExecLo = 126;
  words.push_back(build_s_mov_b64(kExecLo, exec_save));
  append_wait_salu_sgpr(words);
}

void append_save_exec(std::vector<uint32_t> &words, uint8_t exec_save) {
  constexpr uint8_t kExecLo = 126;
  words.push_back(build_s_mov_b64(exec_save, kExecLo));
  append_wait_salu_sgpr(words);
}

void append_restore_vcc(std::vector<uint32_t> &words, uint8_t vcc_save) {
  constexpr uint8_t kVccLo = 106;
  words.push_back(build_s_mov_b64(kVccLo, vcc_save));
  append_wait_salu_sgpr(words);
}

void append_save_vcc(std::vector<uint32_t> &words, uint8_t vcc_save) {
  constexpr uint8_t kVccLo = 106;
  words.push_back(build_s_mov_b64(vcc_save, kVccLo));
  append_wait_salu_sgpr(words);
}

bool append_rdna4_vglobal_load(std::vector<uint32_t> &words, const VglobalFields &fields,
                               uint16_t rdna4_op) {
  auto mem = gfx1250_to_rdna4::encode_vglobal_rdna4(fields, rdna4_op);
  if (mem.word_count != 3)
    return false;
  words.insert(words.end(), mem.words, mem.words + mem.word_count);
  return true;
}

bool append_rdna4_vds_inst(std::vector<uint32_t> &words, const VdsFields &fields,
                           uint16_t rdna4_op) {
  auto mem = gfx1250_to_rdna4::encode_vds_rdna4(fields, rdna4_op);
  if (mem.word_count != 2)
    return false;
  words.insert(words.end(), mem.words, mem.words + mem.word_count);
  return true;
}

void set_vds_byte_offset(VdsFields &fields, uint32_t byte_offset) {
  fields.offset0 = byte_offset & 0xFFu;
  fields.offset1 = (byte_offset >> 8) & 0xFFu;
}

bool append_rdna4_ds_load_b32_sequence(std::vector<uint32_t> &words, VdsFields fields,
                                       uint8_t raw_load, uint8_t word_count,
                                       uint32_t base_byte_offset) {
  constexpr uint8_t kOpDsLoadB32 = 0x36;
  fields.data0 = 0;
  fields.data1 = 0;
  for (uint8_t word = 0; word < word_count; ++word) {
    fields.vdst = static_cast<uint8_t>(raw_load + word);
    set_vds_byte_offset(fields, base_byte_offset + static_cast<uint32_t>(word) * 4u);
    if (!append_rdna4_vds_inst(words, fields, kOpDsLoadB32))
      return false;
  }
  return true;
}

void append_copy_vgpr_words(std::vector<uint32_t> &words, uint32_t host_arch, uint8_t dst_base,
                            uint8_t src_base, uint8_t word_count) {
  constexpr uint8_t kOpOrB32 = 28;
  constexpr uint16_t kInline0 = 128;
  for (uint8_t word = 0; word < word_count; ++word) {
    append_vop2(words, kOpOrB32, static_cast<uint8_t>(dst_base + word), kInline0,
                static_cast<uint8_t>(src_base + word));
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
  }
}

size_t append_s_branch_placeholder(std::vector<uint32_t> &words) {
  const size_t index = words.size();
  words.push_back(build_s_branch(0, ROCJITSU_CODE_ARCH_RDNA4));
  return index;
}

size_t append_s_cbranch_scc0_placeholder(std::vector<uint32_t> &words) {
  constexpr uint8_t kOpSCbranchScc0 = 33;
  const size_t index = words.size();
  words.push_back(pack_sopp(kOpSCbranchScc0, 0));
  return index;
}

size_t append_s_cbranch_scc1_placeholder(std::vector<uint32_t> &words) {
  constexpr uint8_t kOpSCbranchScc1 = 34;
  const size_t index = words.size();
  words.push_back(pack_sopp(kOpSCbranchScc1, 0));
  return index;
}

void patch_sopp_branch_target(std::vector<uint32_t> &words, size_t branch_index,
                              size_t target_index) {
  const int64_t offset =
      static_cast<int64_t>(target_index) - static_cast<int64_t>(branch_index) - 1;
  const uint32_t op = (words[branch_index] >> 16) & 0x7Fu;
  words[branch_index] = pack_sopp(op, static_cast<uint16_t>(static_cast<int16_t>(offset)));
}

enum class TensorCopyPath {
  Dense,
  Pad1D,
  Iterate2D,
};

void append_tensor_global_base(std::vector<uint32_t> &words, uint8_t base_sgpr, uint8_t d0_sgpr) {
  words.push_back(build_sop1_mov_b32(base_sgpr, static_cast<uint16_t>(d0_sgpr + 2u)));
  append_s_and_b32_lit(words, static_cast<uint8_t>(base_sgpr + 1u),
                       static_cast<uint16_t>(d0_sgpr + 3u), 0x01FFFFFFu);
}

void append_tensor_dense_offsets(std::vector<uint32_t> &words, uint32_t host_arch, uint8_t lane,
                                 uint8_t global_offset, uint8_t ds_addr, uint8_t d0_sgpr) {
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpAddNcU32 = 37;
  constexpr uint16_t kInline2 = 130;

  append_lane_id(words, lane);
  append_vop2(words, kOpLshlrevB32, global_offset, kInline2, lane);
  words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
  append_vop2(words, kOpAddNcU32, ds_addr, static_cast<uint16_t>(d0_sgpr + 1u), global_offset);
  words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
}

void append_tensor_pad_offsets(std::vector<uint32_t> &words, uint32_t host_arch, uint8_t lane,
                               uint8_t global_offset, uint8_t ds_addr, uint8_t tmp,
                               uint8_t d0_sgpr) {
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpLshrrevB32 = 25;
  constexpr uint8_t kOpAddNcU32 = 37;
  constexpr uint16_t kInline1 = 129;
  constexpr uint16_t kInline2 = 130;

  append_lane_id(words, lane);
  append_vop2(words, kOpLshrrevB32, tmp, kInline1, lane);
  words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
  append_vop2(words, kOpAddNcU32, tmp, static_cast<uint16_t>(256u + tmp), lane);
  words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
  append_vop2(words, kOpLshlrevB32, global_offset, kInline2, lane);
  words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
  append_vop2(words, kOpLshlrevB32, ds_addr, kInline2, tmp);
  words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
  append_vop2(words, kOpAddNcU32, ds_addr, static_cast<uint16_t>(d0_sgpr + 1u), ds_addr);
  words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
}

void append_tensor_iterate_offsets(std::vector<uint32_t> &words, uint32_t host_arch, uint8_t lane,
                                   uint8_t global_offset, uint8_t ds_addr, uint8_t tmp,
                                   uint8_t d0_sgpr) {
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpOrB32 = 28;
  constexpr uint8_t kOpAddNcU32 = 37;
  constexpr uint16_t kInline1 = 129;
  constexpr uint16_t kInline2 = 130;

  append_lane_id(words, lane);
  append_vop2(words, kOpAndB32, tmp, kInline1, lane);
  words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
  append_vop2(words, kOpAndB32, ds_addr, kInline2, lane);
  words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
  append_vop2(words, kOpLshlrevB32, ds_addr, kInline1, ds_addr);
  words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
  append_vop2(words, kOpOrB32, ds_addr, static_cast<uint16_t>(256u + tmp), ds_addr);
  words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
  append_vop2(words, kOpLshlrevB32, global_offset, kInline2, lane);
  words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
  append_vop2(words, kOpLshlrevB32, ds_addr, kInline2, ds_addr);
  words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
  append_vop2(words, kOpAddNcU32, ds_addr, static_cast<uint16_t>(d0_sgpr + 1u), ds_addr);
  words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
}

void append_tensor_offsets(std::vector<uint32_t> &words, uint32_t host_arch, TensorCopyPath path,
                           uint8_t lane, uint8_t global_offset, uint8_t ds_addr, uint8_t tmp,
                           uint8_t d0_sgpr) {
  switch (path) {
  case TensorCopyPath::Dense:
    append_tensor_dense_offsets(words, host_arch, lane, global_offset, ds_addr, d0_sgpr);
    break;
  case TensorCopyPath::Pad1D:
    append_tensor_pad_offsets(words, host_arch, lane, global_offset, ds_addr, tmp, d0_sgpr);
    break;
  case TensorCopyPath::Iterate2D:
    append_tensor_iterate_offsets(words, host_arch, lane, global_offset, ds_addr, tmp, d0_sgpr);
    break;
  }
}

bool append_tensor_zero_lds_path(std::vector<uint32_t> &words, uint32_t host_arch,
                                 TensorCopyPath path, uint8_t d0_sgpr, uint8_t tmp_vgpr_base) {
  constexpr uint8_t kOpMovB32 = 1;
  constexpr uint8_t kOpDsStoreB32 = 0x0D;
  constexpr uint8_t kOpWaitDscnt = 70;
  constexpr uint16_t kInline0 = 128;

  const uint8_t lane = tmp_vgpr_base;
  const uint8_t global_offset = static_cast<uint8_t>(tmp_vgpr_base + 1u);
  const uint8_t ds_addr = static_cast<uint8_t>(tmp_vgpr_base + 2u);
  const uint8_t data = static_cast<uint8_t>(tmp_vgpr_base + 3u);
  const uint8_t tmp = static_cast<uint8_t>(tmp_vgpr_base + 4u);

  append_set_exec_lo_mask(words, 0x0000FFFFu);
  append_tensor_offsets(words, host_arch, path, lane, global_offset, ds_addr, tmp, d0_sgpr);
  append_vop1(words, kOpMovB32, data, kInline0);

  VdsFields ds{};
  ds.vaddr = ds_addr;
  ds.data0 = data;
  if (!append_rdna4_vds_inst(words, ds, kOpDsStoreB32))
    return false;
  words.push_back(pack_sopp(kOpWaitDscnt, 0));
  return true;
}

bool append_tensor_copy_path(std::vector<uint32_t> &words, uint32_t host_arch, TensorCopyPath path,
                             bool store_from_lds, uint8_t d0_sgpr, uint8_t base_sgpr,
                             uint8_t tmp_vgpr_base) {
  constexpr uint8_t kOpGlobalLoadB32Rdna4 = 20;
  constexpr uint8_t kOpGlobalStoreB32Rdna4 = 26;
  constexpr uint8_t kOpDsStoreB32 = 0x0D;
  constexpr uint8_t kOpDsLoadB32 = 0x36;
  constexpr uint8_t kOpWaitLoadcnt = 64;
  constexpr uint8_t kOpWaitStorecnt = 65;
  constexpr uint8_t kOpWaitDscnt = 70;

  const uint8_t lane = tmp_vgpr_base;
  const uint8_t global_offset = static_cast<uint8_t>(tmp_vgpr_base + 1u);
  const uint8_t ds_addr = static_cast<uint8_t>(tmp_vgpr_base + 2u);
  const uint8_t data = static_cast<uint8_t>(tmp_vgpr_base + 3u);
  const uint8_t tmp = static_cast<uint8_t>(tmp_vgpr_base + 4u);
  const uint32_t lane_mask =
      store_from_lds || path == TensorCopyPath::Dense ? 0x0000FFFFu : 0x0000000Fu;

  if (!store_from_lds && path != TensorCopyPath::Dense) {
    if (!append_tensor_zero_lds_path(words, host_arch, path, d0_sgpr, tmp_vgpr_base))
      return false;
  }

  append_set_exec_lo_mask(words, lane_mask);
  append_tensor_offsets(words, host_arch, path, lane, global_offset, ds_addr, tmp, d0_sgpr);

  VglobalFields global{};
  global.saddr = base_sgpr;
  global.vaddr = global_offset;
  global.scope = 0;
  global.th = 0;

  VdsFields ds{};
  ds.vaddr = ds_addr;

  if (store_from_lds) {
    ds.vdst = data;
    if (!append_rdna4_vds_inst(words, ds, kOpDsLoadB32))
      return false;
    words.push_back(pack_sopp(kOpWaitDscnt, 0));
    global.vsrc = data;
    if (!append_rdna4_vglobal_load(words, global, kOpGlobalStoreB32Rdna4))
      return false;
    words.push_back(pack_sopp(kOpWaitStorecnt, 0));
  } else {
    global.vdst = data;
    if (!append_rdna4_vglobal_load(words, global, kOpGlobalLoadB32Rdna4))
      return false;
    words.push_back(pack_sopp(kOpWaitLoadcnt, 0));
    ds.data0 = data;
    if (!append_rdna4_vds_inst(words, ds, kOpDsStoreB32))
      return false;
    words.push_back(pack_sopp(kOpWaitDscnt, 0));
  }
  return true;
}

bool append_tensor_tensile_u8_256x16_pad32x4_load(std::vector<uint32_t> &words, uint32_t host_arch,
                                                  uint8_t d0_sgpr, uint8_t d1_sgpr,
                                                  uint8_t base_sgpr, uint8_t tmp_vgpr_base) {
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpLshrrevB32 = 25;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpAddNcU32 = 37;
  constexpr uint8_t kOpGlobalLoadB32Rdna4 = 20;
  constexpr uint8_t kOpDsStoreB32 = 0x0D;
  constexpr uint8_t kOpWaitLoadcnt = 64;
  constexpr uint8_t kOpWaitDscnt = 70;
  constexpr uint16_t kOpMulLoU32 = 812;
  constexpr uint16_t kInline1 = scalar_positive_inline_u32(1);
  constexpr uint16_t kInline4 = scalar_positive_inline_u32(4);
  constexpr uint16_t kInline5 = scalar_positive_inline_u32(5);
  constexpr uint16_t kInline8 = scalar_positive_inline_u32(8);
  constexpr uint16_t kInline15 = scalar_positive_inline_u32(15);

  const uint8_t lane = tmp_vgpr_base;
  const uint8_t global_offset = static_cast<uint8_t>(tmp_vgpr_base + 1u);
  const uint8_t ds_addr = static_cast<uint8_t>(tmp_vgpr_base + 2u);
  const uint8_t data = static_cast<uint8_t>(tmp_vgpr_base + 3u);
  const uint8_t tmp = static_cast<uint8_t>(tmp_vgpr_base + 4u);
  const uint8_t stride0_sgpr = static_cast<uint8_t>(d1_sgpr + 5u);

  VglobalFields global{};
  global.saddr = base_sgpr;
  global.vaddr = global_offset;
  global.vdst = data;
  global.scope = 0;
  global.th = 0;

  VdsFields ds{};
  ds.vaddr = ds_addr;
  ds.data0 = data;

  append_set_exec_lo_mask(words, 0xFFFFFFFFu);
  append_lane_id(words, lane);
  append_wait_valu_vgpr(words);

  for (uint8_t half = 0; half < 2; ++half) {
    for (uint8_t word = 0; word < 16; ++word) {
      const uint32_t k_base = static_cast<uint32_t>(half) * 128u +
                              static_cast<uint32_t>(word / 8u) * 64u +
                              static_cast<uint32_t>(word % 8u) * 4u;
      const uint32_t ds_word_offset = static_cast<uint32_t>(half) * 128u +
                                      static_cast<uint32_t>(word / 4u) * 32u +
                                      static_cast<uint32_t>(word % 4u) * 4u;

      append_vop2(words, kOpAndB32, global_offset, kInline15, lane);
      words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
      append_vop3(words, kOpMulLoU32, global_offset, static_cast<uint16_t>(256u + global_offset),
                  stride0_sgpr);
      words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
      append_vop2(words, kOpLshrrevB32, tmp, kInline4, lane);
      words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
      append_vop2(words, kOpAndB32, tmp, kInline1, tmp);
      words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
      append_vop2(words, kOpLshlrevB32, tmp, kInline5, tmp);
      words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
      append_vop2(words, kOpAddNcU32, global_offset, static_cast<uint16_t>(256u + tmp),
                  global_offset);
      words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
      if (k_base != 0) {
        std::optional<uint32_t> literal;
        append_vop2(words, kOpAddNcU32, global_offset, literal_or_inline_u32(k_base, literal),
                    global_offset, literal);
        words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
      }

      append_vop2(words, kOpAndB32, ds_addr, kInline15, lane);
      words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
      append_vop2(words, kOpLshlrevB32, tmp, kInline8, ds_addr);
      words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
      append_vop2(words, kOpLshlrevB32, ds_addr, kInline4, ds_addr);
      words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
      append_vop2(words, kOpAddNcU32, ds_addr, static_cast<uint16_t>(256u + tmp), ds_addr);
      words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
      append_vop2(words, kOpLshrrevB32, tmp, kInline4, lane);
      words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
      append_vop2(words, kOpAndB32, tmp, kInline1, tmp);
      words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
      append_vop2(words, kOpLshlrevB32, tmp, kInline4, tmp);
      words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
      append_vop2(words, kOpAddNcU32, ds_addr, static_cast<uint16_t>(256u + tmp), ds_addr);
      words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
      if (ds_word_offset != 0) {
        std::optional<uint32_t> literal;
        append_vop2(words, kOpAddNcU32, ds_addr, literal_or_inline_u32(ds_word_offset, literal),
                    ds_addr, literal);
        words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
      }
      append_vop2(words, kOpAddNcU32, ds_addr, static_cast<uint16_t>(d0_sgpr + 1u), ds_addr);
      words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));

      if (!append_rdna4_vglobal_load(words, global, kOpGlobalLoadB32Rdna4))
        return false;
      words.push_back(pack_sopp(kOpWaitLoadcnt, 0));
      if (!append_rdna4_vds_inst(words, ds, kOpDsStoreB32))
        return false;
      words.push_back(pack_sopp(kOpWaitDscnt, 0));
    }
  }
  return true;
}

bool append_tensor_tensile_u8_128x16_pad64x4_load(std::vector<uint32_t> &words, uint32_t host_arch,
                                                  uint8_t d0_sgpr, uint8_t d1_sgpr,
                                                  uint8_t base_sgpr, uint8_t tmp_vgpr_base) {
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpLshrrevB32 = 25;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpAddNcU32 = 37;
  constexpr uint8_t kOpGlobalLoadB32Rdna4 = 20;
  constexpr uint8_t kOpDsStoreB32 = 0x0D;
  constexpr uint8_t kOpWaitLoadcnt = 64;
  constexpr uint8_t kOpWaitDscnt = 70;
  constexpr uint16_t kOpMulLoU32 = 812;
  constexpr uint16_t kInline1 = scalar_positive_inline_u32(1);
  constexpr uint16_t kInline4 = scalar_positive_inline_u32(4);
  constexpr uint16_t kInline5 = scalar_positive_inline_u32(5);
  constexpr uint16_t kInline7 = scalar_positive_inline_u32(7);
  constexpr uint16_t kInline15 = scalar_positive_inline_u32(15);

  const uint8_t lane = tmp_vgpr_base;
  const uint8_t global_offset = static_cast<uint8_t>(tmp_vgpr_base + 1u);
  const uint8_t ds_addr = static_cast<uint8_t>(tmp_vgpr_base + 2u);
  const uint8_t data = static_cast<uint8_t>(tmp_vgpr_base + 3u);
  const uint8_t tmp = static_cast<uint8_t>(tmp_vgpr_base + 4u);
  const uint8_t stride0_sgpr = static_cast<uint8_t>(d1_sgpr + 5u);

  VglobalFields global{};
  global.saddr = base_sgpr;
  global.vaddr = global_offset;
  global.vdst = data;
  global.scope = 0;
  global.th = 0;

  VdsFields ds{};
  ds.vaddr = ds_addr;
  ds.data0 = data;

  append_set_exec_lo_mask(words, 0xFFFFFFFFu);
  append_lane_id(words, lane);
  append_wait_valu_vgpr(words);

  for (uint8_t word = 0; word < 16; ++word) {
    const uint32_t k_base =
        static_cast<uint32_t>(word / 8u) * 64u + static_cast<uint32_t>(word % 8u) * 4u;
    // Preserve the source Tensor DMA K-column order in LDS. The 256x16 fast path uses a
    // different physical swizzle; reusing it here pairs random data with the wrong MX scale.
    const uint32_t ds_word_offset =
        static_cast<uint32_t>(word / 8u) * 64u + static_cast<uint32_t>(word % 8u) * 4u;

    append_vop2(words, kOpAndB32, global_offset, kInline15, lane);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    append_vop3(words, kOpMulLoU32, global_offset, static_cast<uint16_t>(256u + global_offset),
                stride0_sgpr);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    append_vop2(words, kOpLshrrevB32, tmp, kInline4, lane);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    append_vop2(words, kOpAndB32, tmp, kInline1, tmp);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    append_vop2(words, kOpLshlrevB32, tmp, kInline5, tmp);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    append_vop2(words, kOpAddNcU32, global_offset, static_cast<uint16_t>(256u + tmp),
                global_offset);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    if (k_base != 0) {
      std::optional<uint32_t> literal;
      append_vop2(words, kOpAddNcU32, global_offset, literal_or_inline_u32(k_base, literal),
                  global_offset, literal);
      words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    }

    append_vop2(words, kOpAndB32, ds_addr, kInline15, lane);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    append_vop2(words, kOpLshlrevB32, tmp, kInline7, ds_addr);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    append_vop2(words, kOpLshrrevB32, ds_addr, kInline1, ds_addr);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    append_vop2(words, kOpLshlrevB32, ds_addr, kInline4, ds_addr);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    append_vop2(words, kOpAddNcU32, ds_addr, static_cast<uint16_t>(256u + tmp), ds_addr);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    append_vop2(words, kOpLshrrevB32, tmp, kInline4, lane);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    append_vop2(words, kOpAndB32, tmp, kInline1, tmp);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    append_vop2(words, kOpLshlrevB32, tmp, kInline5, tmp);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    append_vop2(words, kOpAddNcU32, ds_addr, static_cast<uint16_t>(256u + tmp), ds_addr);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    if (ds_word_offset != 0) {
      std::optional<uint32_t> literal;
      append_vop2(words, kOpAddNcU32, ds_addr, literal_or_inline_u32(ds_word_offset, literal),
                  ds_addr, literal);
      words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    }
    append_vop2(words, kOpAddNcU32, ds_addr, static_cast<uint16_t>(d0_sgpr + 1u), ds_addr);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));

    if (!append_rdna4_vglobal_load(words, global, kOpGlobalLoadB32Rdna4))
      return false;
    words.push_back(pack_sopp(kOpWaitLoadcnt, 0));
    if (!append_rdna4_vds_inst(words, ds, kOpDsStoreB32))
      return false;
    words.push_back(pack_sopp(kOpWaitDscnt, 0));
  }
  return true;
}

bool append_tensor_tensile_u8_128x1_load(std::vector<uint32_t> &words, uint32_t host_arch,
                                         uint8_t d0_sgpr, uint8_t base_sgpr,
                                         uint8_t tmp_vgpr_base) {
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpAddNcU32 = 37;
  constexpr uint8_t kOpGlobalLoadB32Rdna4 = 20;
  constexpr uint8_t kOpDsStoreB32 = 0x0D;
  constexpr uint8_t kOpWaitLoadcnt = 64;
  constexpr uint8_t kOpWaitDscnt = 70;
  constexpr uint16_t kInline2 = 130;

  const uint8_t lane = tmp_vgpr_base;
  const uint8_t global_offset = static_cast<uint8_t>(tmp_vgpr_base + 1u);
  const uint8_t ds_addr = static_cast<uint8_t>(tmp_vgpr_base + 2u);
  const uint8_t data = static_cast<uint8_t>(tmp_vgpr_base + 3u);

  VglobalFields global{};
  global.saddr = base_sgpr;
  global.vaddr = global_offset;
  global.vdst = data;
  global.scope = 0;
  global.th = 0;

  VdsFields ds{};
  ds.vaddr = ds_addr;
  ds.data0 = data;

  append_set_exec_lo_mask(words, 0xFFFFFFFFu);
  append_lane_id(words, lane);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpLshlrevB32, global_offset, kInline2, lane);
  words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
  append_vop2(words, kOpLshlrevB32, ds_addr, kInline2, lane);
  words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
  append_vop2(words, kOpAddNcU32, ds_addr, static_cast<uint16_t>(d0_sgpr + 1u), ds_addr);
  words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));

  if (!append_rdna4_vglobal_load(words, global, kOpGlobalLoadB32Rdna4))
    return false;
  words.push_back(pack_sopp(kOpWaitLoadcnt, 0));
  if (!append_rdna4_vds_inst(words, ds, kOpDsStoreB32))
    return false;
  words.push_back(pack_sopp(kOpWaitDscnt, 0));
  return true;
}

bool append_tensor_barrier_arrive_zero(std::vector<uint32_t> &words, uint32_t host_arch,
                                       uint8_t d1_sgpr, uint8_t tmp_vgpr_base) {
  constexpr uint8_t kOpMovB32 = 1;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpDsStoreB32 = 0x0D;
  constexpr uint8_t kOpWaitDscnt = 70;
  constexpr uint16_t kInline3 = 131;

  const uint8_t addr = static_cast<uint8_t>(tmp_vgpr_base + 2u);
  const uint8_t data = static_cast<uint8_t>(tmp_vgpr_base + 3u);

  append_set_exec_lo_mask(words, 1u);
  append_vop1(words, kOpMovB32, addr, static_cast<uint16_t>(d1_sgpr + 1u));
  words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
  append_vop2(words, kOpAndB32, addr, 255, addr, 0x0000FFFFu);
  words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
  append_vop2(words, kOpLshlrevB32, addr, kInline3, addr);
  words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
  append_vop1(words, kOpMovB32, data, 255, 0xE0000000u);
  words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));

  VdsFields ds{};
  ds.vaddr = addr;
  ds.data0 = data;
  if (!append_rdna4_vds_inst(words, ds, kOpDsStoreB32))
    return false;
  words.push_back(pack_sopp(kOpWaitDscnt, 0));
  return true;
}

std::vector<uint32_t> expand_tensor_dma_vimage(const Instruction &inst, uint32_t host_arch,
                                               uint64_t, const LivenessAnalysis &liveness,
                                               const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::VimageMachineInst))
    return {};

  gfx1250::VimageMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  constexpr uint8_t kTensorLoadToLds = 196;
  constexpr uint8_t kTensorStoreFromLds = 197;
  if (src.op != kTensorLoadToLds && src.op != kTensorStoreFromLds)
    return {};
  if (src.vaddr0 == kNullSgpr || src.vaddr1 == kNullSgpr || src.vaddr0 > 101 || src.vaddr1 > 97)
    return {};

  const auto tmp_vgpr_opt = liveness.find_free_run(&inst, 5);
  if (!tmp_vgpr_opt || *tmp_vgpr_opt > 251)
    return {};
  const uint8_t tmp_vgpr = static_cast<uint8_t>(*tmp_vgpr_opt);

  auto exec_save_opt = liveness.find_free_sgpr_pair(&inst);
  if (!exec_save_opt || *exec_save_opt > 124)
    return {};
  const uint8_t exec_save = static_cast<uint8_t>(*exec_save_opt);

  std::vector<uint8_t> avoid_sgpr = {exec_save, static_cast<uint8_t>(exec_save + 1u)};
  auto base_pair_opt = find_free_sgpr_pair_avoiding(inst, liveness, avoid_sgpr);
  if (!base_pair_opt || *base_pair_opt > 122)
    return {};
  const uint8_t base_pair = static_cast<uint8_t>(*base_pair_opt);
  avoid_sgpr.push_back(base_pair);
  avoid_sgpr.push_back(static_cast<uint8_t>(base_pair + 1u));

  auto flag_sgpr_opt = find_free_sgpr_avoiding(inst, liveness, avoid_sgpr);
  if (!flag_sgpr_opt)
    return {};
  const uint8_t flag_sgpr = static_cast<uint8_t>(*flag_sgpr_opt);

  constexpr uint32_t kTensorAtomicBarrierFlag = 1u << 18;
  constexpr uint32_t kTensorIterateFlag = 1u << 19;
  constexpr uint32_t kTensorPadFlag = 1u << 20;

  const bool store_from_lds = src.op == kTensorStoreFromLds;
  std::vector<uint32_t> words;
  words.reserve(96);
  append_save_exec(words, exec_save);
  append_tensor_global_base(words, base_pair, static_cast<uint8_t>(src.vaddr0));

  std::optional<size_t> branch_tensile_u8_128x16_data;
  std::optional<size_t> branch_tensile_u8_256x16_data;
  std::optional<size_t> branch_tensile_mxf8_scale;
  if (!store_from_lds && src.vaddr0 == 16 && src.vaddr1 == 20) {
    append_s_and_b32_lit(words, flag_sgpr, static_cast<uint16_t>(src.vaddr1), 0xFFFF0000u);
    append_s_cmp_eq_u32_lit(words, flag_sgpr, 0x07500000u);
    const size_t branch_data_flags_mismatch = append_s_cbranch_scc0_placeholder(words);
    append_s_and_b32_lit(words, flag_sgpr, static_cast<uint16_t>(src.vaddr1 + 3u), 0xFFFF0000u);
    append_s_cmp_eq_u32_lit(words, flag_sgpr, 0x00800000u);
    branch_tensile_u8_128x16_data = append_s_cbranch_scc1_placeholder(words);
    append_s_cmp_eq_u32_lit(words, flag_sgpr, 0x01000000u);
    branch_tensile_u8_256x16_data = append_s_cbranch_scc1_placeholder(words);
    patch_sopp_branch_target(words, branch_data_flags_mismatch, words.size());
  } else if (!store_from_lds && src.vaddr0 == 28 && src.vaddr1 == 80) {
    append_s_and_b32_lit(words, flag_sgpr, static_cast<uint16_t>(src.vaddr1), 0xFFFF0000u);
    append_s_cmp_eq_u32_lit(words, flag_sgpr, 0);
    branch_tensile_mxf8_scale = append_s_cbranch_scc1_placeholder(words);
  }

  append_s_and_b32_lit(words, flag_sgpr, static_cast<uint16_t>(src.vaddr1), kTensorIterateFlag);
  const size_t branch_iterate = append_s_cbranch_scc1_placeholder(words);
  append_s_and_b32_lit(words, flag_sgpr, static_cast<uint16_t>(src.vaddr1), kTensorPadFlag);
  const size_t branch_pad = append_s_cbranch_scc1_placeholder(words);

  if (!append_tensor_copy_path(words, host_arch, TensorCopyPath::Dense, store_from_lds,
                               static_cast<uint8_t>(src.vaddr0), base_pair, tmp_vgpr))
    return {};
  const size_t branch_dense_end = append_s_branch_placeholder(words);

  const size_t pad_start = words.size();
  patch_sopp_branch_target(words, branch_pad, pad_start);
  if (!append_tensor_copy_path(words, host_arch, TensorCopyPath::Pad1D, store_from_lds,
                               static_cast<uint8_t>(src.vaddr0), base_pair, tmp_vgpr))
    return {};
  const size_t branch_pad_end = append_s_branch_placeholder(words);

  const size_t iterate_start = words.size();
  patch_sopp_branch_target(words, branch_iterate, iterate_start);
  if (!append_tensor_copy_path(words, host_arch, TensorCopyPath::Iterate2D, store_from_lds,
                               static_cast<uint8_t>(src.vaddr0), base_pair, tmp_vgpr))
    return {};

  const size_t paths_end = words.size();
  patch_sopp_branch_target(words, branch_dense_end, paths_end);
  patch_sopp_branch_target(words, branch_pad_end, paths_end);

  append_s_and_b32_lit(words, flag_sgpr, static_cast<uint16_t>(src.vaddr1),
                       kTensorAtomicBarrierFlag);
  const size_t branch_no_barrier = append_s_cbranch_scc0_placeholder(words);
  if (!append_tensor_barrier_arrive_zero(words, host_arch, static_cast<uint8_t>(src.vaddr1),
                                         tmp_vgpr))
    return {};
  patch_sopp_branch_target(words, branch_no_barrier, words.size());

  append_restore_exec(words, exec_save);
  std::optional<size_t> branch_generic_end;
  if (branch_tensile_u8_128x16_data || branch_tensile_u8_256x16_data || branch_tensile_mxf8_scale)
    branch_generic_end = append_s_branch_placeholder(words);

  std::vector<size_t> branch_special_end;
  if (branch_tensile_u8_128x16_data) {
    patch_sopp_branch_target(words, *branch_tensile_u8_128x16_data, words.size());
    if (!append_tensor_tensile_u8_128x16_pad64x4_load(
            words, host_arch, static_cast<uint8_t>(src.vaddr0), static_cast<uint8_t>(src.vaddr1),
            base_pair, tmp_vgpr))
      return {};
    append_restore_exec(words, exec_save);
    branch_special_end.push_back(append_s_branch_placeholder(words));
  }
  if (branch_tensile_u8_256x16_data) {
    patch_sopp_branch_target(words, *branch_tensile_u8_256x16_data, words.size());
    if (!append_tensor_tensile_u8_256x16_pad32x4_load(
            words, host_arch, static_cast<uint8_t>(src.vaddr0), static_cast<uint8_t>(src.vaddr1),
            base_pair, tmp_vgpr))
      return {};
    append_restore_exec(words, exec_save);
    branch_special_end.push_back(append_s_branch_placeholder(words));
  }
  if (branch_tensile_mxf8_scale) {
    patch_sopp_branch_target(words, *branch_tensile_mxf8_scale, words.size());
    if (!append_tensor_tensile_u8_128x1_load(words, host_arch, static_cast<uint8_t>(src.vaddr0),
                                             base_pair, tmp_vgpr))
      return {};
    append_restore_exec(words, exec_save);
    branch_special_end.push_back(append_s_branch_placeholder(words));
  }

  if (branch_generic_end)
    patch_sopp_branch_target(words, *branch_generic_end, words.size());
  for (const size_t branch : branch_special_end)
    patch_sopp_branch_target(words, branch, words.size());
  return words;
}

ExpandResult expand_scaled_vglobal(const Instruction &inst, uint32_t, uint64_t,
                                   const LivenessAnalysis &liveness, TranslationContext &context,
                                   const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::VglobalMachineInst))
    return ExpandResult::not_handled();

  gfx1250::VglobalMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (!src.scale_offset)
    return ExpandResult::not_handled();

  const auto access_size = scaled_vglobal_access_size(inst.opcode());
  if (!access_size)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " uses scale_offset with an unsupported access width");

  std::optional<uint8_t> tmp;
  if (*access_size != 1) {
    const auto tmp_opt = liveness.find_free_run(&inst, 1);
    if (!tmp_opt || *tmp_opt > 255)
      return ExpandResult::failed(std::string(inst.mnemonic()) +
                                  " cannot allocate a VGPR for scale_offset lowering");
    tmp = static_cast<uint8_t>(*tmp_opt);
  }

  auto fields = gfx1250_to_rdna4::decode_vglobal_gfx1250(raw[0], raw[1], raw[2]);
  fields.scale_offset = 0;

  std::vector<uint32_t> words;
  words.reserve(6);
  if (tmp) {
    fields.vaddr = *tmp;
    if (std::has_single_bit(*access_size)) {
      constexpr uint8_t kOpLshlrevB32 = 24;
      const auto shift = static_cast<uint8_t>(std::countr_zero(*access_size));
      words.push_back(
          build_vop2(kOpLshlrevB32, *tmp, scalar_positive_inline_u32(shift), src.vaddr));
    } else {
      constexpr uint16_t kOpMulLoU32 = 812;
      append_vop3(words, kOpMulLoU32, *tmp, vgpr_src(static_cast<uint8_t>(src.vaddr)),
                  scalar_positive_inline_u32(*access_size));
    }
    append_wait_valu_vgpr(words);
    context.require_vgprs(static_cast<uint32_t>(*tmp) + 1u);
  }

  auto mem = gfx1250_to_rdna4::encode_vglobal_rdna4(fields, inst.opcode());
  if (mem.word_count != 3)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " could not encode the scaled RDNA4 VMEM replacement");
  words.insert(words.end(), mem.words, mem.words + mem.word_count);
  return ExpandResult::success(std::move(words));
}

ExpandResult expand_scaled_vscratch(const Instruction &inst, uint32_t, uint64_t,
                                    const LivenessAnalysis &liveness, TranslationContext &context,
                                    const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::VscratchMachineInst))
    return ExpandResult::not_handled();

  gfx1250::VscratchMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (!src.scale_offset || !src.sve)
    return ExpandResult::not_handled();

  const auto access_size = scaled_vglobal_access_size(inst.opcode());
  if (!access_size)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " uses scale_offset with an unsupported access width");
  if (*access_size == 1)
    return ExpandResult::not_handled();

  const auto tmp_opt = liveness.find_free_run(&inst, 1);
  if (!tmp_opt || *tmp_opt > 255)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " cannot allocate a VGPR for scale_offset lowering");
  const auto tmp = static_cast<uint8_t>(*tmp_opt);

  auto fields = gfx1250_to_rdna4::decode_vscratch_gfx1250(raw[0], raw[1], raw[2]);
  fields.scale_offset = 0;
  fields.vaddr = tmp;

  std::vector<uint32_t> words;
  words.reserve(6);
  if (std::has_single_bit(*access_size)) {
    constexpr uint8_t kOpLshlrevB32 = 24;
    const auto shift = static_cast<uint8_t>(std::countr_zero(*access_size));
    words.push_back(build_vop2(kOpLshlrevB32, tmp, scalar_positive_inline_u32(shift), src.vaddr));
  } else {
    constexpr uint16_t kOpMulLoU32 = 812;
    append_vop3(words, kOpMulLoU32, tmp, vgpr_src(static_cast<uint8_t>(src.vaddr)),
                scalar_positive_inline_u32(*access_size));
  }
  append_wait_valu_vgpr(words);
  context.require_vgprs(static_cast<uint32_t>(tmp) + 1u);

  auto mem = gfx1250_to_rdna4::encode_vscratch_rdna4(fields, inst.opcode());
  if (mem.word_count != 3)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " could not encode the scaled RDNA4 scratch replacement");
  words.insert(words.end(), mem.words, mem.words + mem.word_count);
  return ExpandResult::success(std::move(words));
}

ExpandResult expand_global_atomic_add_f64(const Instruction &inst, uint32_t, uint64_t,
                                          const LivenessAnalysis &liveness,
                                          TranslationContext &context, const LaneLayout *,
                                          const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::VglobalMachineInst))
    return ExpandResult::not_handled();

  gfx1250::VglobalMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.op != 0x55)
    return ExpandResult::not_handled();
  if (src.scale_offset != 0) {
    return ExpandResult::failed(
        "global_atomic_add_f64 scale_offset needs explicit address scaling before CAS lowering");
  }
  if (src.vsrc > 254 || src.vaddr == 255)
    return ExpandResult::failed("global_atomic_add_f64 uses an out-of-range VGPR operand");
  if (src.th > 1)
    return ExpandResult::failed("global_atomic_add_f64 uses an unsupported temporal hint");

  auto fields = gfx1250_to_rdna4::decode_vglobal_gfx1250(raw[0], raw[1], raw[2]);

  std::vector<uint8_t> avoid_vgpr;
  add_avoid_vgpr(avoid_vgpr, static_cast<uint8_t>(fields.vsrc));
  add_avoid_vgpr(avoid_vgpr, static_cast<uint8_t>(fields.vsrc + 1u));
  add_avoid_vgpr(avoid_vgpr, static_cast<uint8_t>(fields.vaddr));
  if (fields.saddr == 124 && fields.vaddr < 255)
    add_avoid_vgpr(avoid_vgpr, static_cast<uint8_t>(fields.vaddr + 1u));
  if (fields.th == 1) {
    add_avoid_vgpr(avoid_vgpr, static_cast<uint8_t>(fields.vdst));
    if (fields.vdst < 255)
      add_avoid_vgpr(avoid_vgpr, static_cast<uint8_t>(fields.vdst + 1u));
  }

  const auto tmp_base = find_free_vgpr_run_avoiding(inst, liveness, 4, avoid_vgpr);
  if (!tmp_base || *tmp_base > 252u) {
    return ExpandResult::failed(
        "No free VGPR run is available for global_atomic_add_f64 CAS lowering",
        {"Reduce VGPR pressure or add private spilling for atomic lowering."});
  }
  const auto tmp_new = static_cast<uint8_t>(*tmp_base);
  const auto tmp_compare = static_cast<uint8_t>(*tmp_base + 2u);
  context.require_vgprs(static_cast<uint32_t>(*tmp_base) + 4u);

  std::vector<uint8_t> avoid_sgpr;
  if (fields.saddr < 124) {
    avoid_sgpr.push_back(static_cast<uint8_t>(fields.saddr));
    avoid_sgpr.push_back(static_cast<uint8_t>(fields.saddr + 1u));
  }
  const auto exec_save = find_free_sgpr_pair_avoiding(inst, liveness, avoid_sgpr);
  if (!exec_save)
    return ExpandResult::failed("No free SGPR pair is available to save EXEC for atomic lowering");
  avoid_sgpr.push_back(static_cast<uint8_t>(*exec_save));
  avoid_sgpr.push_back(static_cast<uint8_t>(*exec_save + 1u));
  const auto success_pred = find_free_sgpr_pair_avoiding(inst, liveness, avoid_sgpr);
  if (!success_pred) {
    return ExpandResult::failed(
        "No free SGPR pair is available for the global_atomic_add_f64 CAS predicate");
  }
  context.require_sgprs(
      std::max(static_cast<uint32_t>(*exec_save) + 2u, static_cast<uint32_t>(*success_pred) + 2u));

  constexpr uint16_t kOpGlobalLoadB64 = 21;
  constexpr uint16_t kOpGlobalAtomicCmpSwapB64 = 0x42;
  constexpr uint8_t kOpAddF64 = 2;
  constexpr uint16_t kOpCmpEqU64 = 90;
  constexpr uint8_t kOpMovB32 = 1;
  constexpr uint8_t kOpSAndNot1B64 = 35;
  constexpr uint8_t kOpSCbranchExecnz = 0x26;
  constexpr uint8_t kOpWaitLoadcnt = 64;
  constexpr uint8_t kExecLo = 126;

  std::vector<uint32_t> words;
  words.reserve(32);
  append_save_exec(words, static_cast<uint8_t>(*exec_save));

  auto load = fields;
  load.vdst = tmp_compare;
  load.vsrc = 0;
  load.th = 0;
  if (!append_rdna4_vglobal_load(words, load, kOpGlobalLoadB64))
    return ExpandResult::failed("Could not encode the initial F64 atomic load");
  words.push_back(pack_sopp(kOpWaitLoadcnt, 0));

  const size_t loop_start = words.size();
  append_vop2(words, kOpAddF64, tmp_new, vgpr_src(tmp_compare), static_cast<uint8_t>(fields.vsrc));
  append_wait_valu_vgpr(words);

  auto cas = fields;
  cas.vdst = tmp_new;
  cas.vsrc = tmp_new;
  cas.th = 1;
  if (!append_rdna4_vglobal_load(words, cas, kOpGlobalAtomicCmpSwapB64))
    return ExpandResult::failed("Could not encode the F64 atomic CAS replacement");
  words.push_back(pack_sopp(kOpWaitLoadcnt, 0));

  append_vop3(words, kOpCmpEqU64, static_cast<uint8_t>(*success_pred), vgpr_src(tmp_new),
              vgpr_src(tmp_compare));
  words.push_back(build_s_wait_alu(kWaitAluDepctrVaSdst0, ROCJITSU_CODE_ARCH_RDNA4));
  append_vop1(words, kOpMovB32, tmp_compare, vgpr_src(tmp_new));
  append_vop1(words, kOpMovB32, static_cast<uint8_t>(tmp_compare + 1u),
              vgpr_src(static_cast<uint8_t>(tmp_new + 1u)));
  append_wait_valu_vgpr(words);

  words.push_back(pack_sop2(kOpSAndNot1B64, kExecLo, kExecLo, *success_pred));
  append_wait_salu_sgpr(words);
  const size_t branch_index = words.size();
  const auto branch_delta = static_cast<int16_t>(static_cast<ptrdiff_t>(loop_start) -
                                                 static_cast<ptrdiff_t>(branch_index + 1u));
  words.push_back(pack_sopp(kOpSCbranchExecnz, static_cast<uint16_t>(branch_delta)));

  append_restore_exec(words, static_cast<uint8_t>(*exec_save));
  if (fields.th == 1) {
    append_vop1(words, kOpMovB32, static_cast<uint8_t>(fields.vdst), vgpr_src(tmp_compare));
    append_vop1(words, kOpMovB32, static_cast<uint8_t>(fields.vdst + 1u),
                vgpr_src(static_cast<uint8_t>(tmp_compare + 1u)));
    append_wait_valu_vgpr(words);
  }
  return ExpandResult::success(std::move(words));
}

[[nodiscard]] constexpr std::optional<uint32_t> gfx1250_smem_access_size(uint32_t opcode) {
  switch (opcode) {
  case 0:  // s_load_b32
  case 16: // s_buffer_load_b32
    return 4;
  case 1:  // s_load_b64
  case 17: // s_buffer_load_b64
    return 8;
  case 2:  // s_load_b128
  case 18: // s_buffer_load_b128
    return 16;
  case 3:  // s_load_b256
  case 19: // s_buffer_load_b256
    return 32;
  case 4:  // s_load_b512
  case 20: // s_buffer_load_b512
    return 64;
  case 5:  // s_load_b96
  case 21: // s_buffer_load_b96
    return 12;
  case 8:  // s_load_i8
  case 9:  // s_load_u8
  case 24: // s_buffer_load_i8
  case 25: // s_buffer_load_u8
    return 1;
  case 10: // s_load_i16
  case 11: // s_load_u16
  case 26: // s_buffer_load_i16
  case 27: // s_buffer_load_u16
    return 2;
  default:
    return std::nullopt;
  }
}

[[nodiscard]] constexpr uint32_t gfx1250_smem_destination_dwords(uint32_t opcode) {
  switch (opcode) {
  case 1:
  case 17:
    return 2;
  case 2:
  case 18:
    return 4;
  case 3:
  case 19:
    return 8;
  case 4:
  case 20:
    return 16;
  case 5:
  case 21:
    return 3;
  default:
    return 1;
  }
}

ExpandResult expand_scaled_smem(const Instruction &inst, uint32_t, uint64_t,
                                const LivenessAnalysis &liveness, TranslationContext &context,
                                const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::SmemMachineInst))
    return ExpandResult::not_handled();

  gfx1250::SmemMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.scale_offset == 0)
    return ExpandResult::not_handled();

  const auto access_size = gfx1250_smem_access_size(src.op);
  if (!access_size)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " uses scale_offset with an unsupported access width");

  auto fields = gfx1250_to_rdna4::decode_smem_gfx1250(raw[0], raw[1]);
  const int64_t signed_ioffset = static_cast<int32_t>(src.ioffset << 8u) >> 8u;
  const int64_t scaled_ioffset = signed_ioffset * *access_size;
  constexpr int64_t kMinSigned24 = -(int64_t{1} << 23u);
  constexpr int64_t kMaxSigned24 = (int64_t{1} << 23u) - 1;
  if (scaled_ioffset < kMinSigned24 || scaled_ioffset > kMaxSigned24)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " scaled immediate offset does not fit RDNA4 SMEM");

  fields.ioffset = static_cast<uint32_t>(scaled_ioffset) & 0x00FF'FFFFu;
  fields.scale_offset = 0;
  // gfx1250's no-allocate hint has no RDNA4 equivalent in this DBT path.
  fields.nv = 0;

  std::vector<uint32_t> words;
  constexpr uint8_t kSoffsetNull = 124;
  if (src.soffset != kSoffsetNull && *access_size != 1) {
    std::vector<uint8_t> avoid;
    avoid.reserve(3u + gfx1250_smem_destination_dwords(src.op));
    avoid.push_back(static_cast<uint8_t>(src.sbase * 2u));
    avoid.push_back(static_cast<uint8_t>(src.sbase * 2u + 1u));
    if (src.soffset <= 105)
      avoid.push_back(static_cast<uint8_t>(src.soffset));
    for (uint32_t i = 0; i < gfx1250_smem_destination_dwords(src.op); ++i) {
      if (src.sdata + i <= 105)
        avoid.push_back(static_cast<uint8_t>(src.sdata + i));
    }

    const auto scaled_soffset = find_free_sgpr_avoiding(inst, liveness, avoid);
    if (!scaled_soffset)
      return ExpandResult::failed(std::string(inst.mnemonic()) +
                                  " cannot allocate an SGPR for scale_offset lowering");

    constexpr uint8_t kOpSMulI32 = 44;
    words.push_back(pack_sop2(kOpSMulI32, *scaled_soffset, src.soffset,
                              scalar_positive_inline_u32(*access_size)));
    fields.soffset = *scaled_soffset;
    context.require_sgprs(*scaled_soffset + 1u);
  }

  const auto mem = gfx1250_to_rdna4::encode_smem_rdna4(fields, inst.opcode());
  if (mem.word_count != 2)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " could not encode the scaled RDNA4 SMEM replacement");
  words.insert(words.end(), mem.words, mem.words + mem.word_count);
  return ExpandResult::success(std::move(words));
}

void append_tr4_b64_repack(std::vector<uint32_t> &words, uint32_t host_arch, uint8_t dst_base,
                           uint8_t raw_load, uint8_t exec_save, uint8_t lane, uint8_t addr,
                           uint8_t shift, uint8_t value) {
  constexpr uint8_t kOpMovB32 = 1;
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpOrB32 = 28;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpLshrrevB32 = 25;
  constexpr uint8_t kOpAddNcU32 = 37;
  constexpr uint8_t kOpWaitDscnt = 70;
  constexpr uint16_t kInline0 = 128;
  constexpr uint16_t kInline1 = 129;
  constexpr uint16_t kInline2 = 130;
  constexpr uint16_t kInline7 = 135;
  constexpr uint16_t kInline15 = 143;
  constexpr uint16_t kInline16 = 144;

  append_set_exec_lo_mask(words, 0xFFFFFFFFu);
  append_lane_id(words, lane);
  append_vop2(words, kOpAndB32, shift, kInline7, lane);
  words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
  append_vop2(words, kOpLshlrevB32, shift, kInline2, shift);
  words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));

  append_restore_exec(words, exec_save);
  append_vop1(words, kOpMovB32, dst_base, kInline0);
  append_vop1(words, kOpMovB32, static_cast<uint8_t>(dst_base + 1u), kInline0);

  auto append_addr_for_source = [&](uint32_t base_byte_offset, uint8_t lane_in_group) {
    append_set_exec_lo_mask(words, 0xFFFFFFFFu);
    append_vop2(words, kOpAndB32, addr, kInline16, lane);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    append_vop2(words, kOpLshlrevB32, addr, kInline1, addr);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    const uint32_t addend = base_byte_offset + static_cast<uint32_t>(lane_in_group) * 4u;
    if (addend != 0) {
      std::optional<uint32_t> literal;
      append_vop2(words, kOpAddNcU32, addr, literal_or_inline_u32(addend, literal), addr, literal);
      words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    }
  };

  auto append_component = [&](uint8_t dst, uint8_t raw_word, uint32_t lane_mask,
                              uint32_t base_byte_offset, uint8_t lane_in_group) {
    append_addr_for_source(base_byte_offset, lane_in_group);
    auto [bp0, bp1] = build_ds_bpermute(value, addr, static_cast<uint8_t>(raw_load + raw_word));
    words.push_back(bp0);
    words.push_back(bp1);
    words.push_back(pack_sopp(kOpWaitDscnt, 0));

    append_set_exec_from_saved_mask(words, exec_save, lane_mask);
    append_vop2(words, kOpLshrrevB32, value, static_cast<uint16_t>(256u + shift), value);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    append_vop2(words, kOpAndB32, value, kInline15, value);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    if (lane_in_group != 0) {
      append_vop2(words, kOpLshlrevB32, value,
                  scalar_positive_inline_u32(static_cast<uint16_t>(lane_in_group * 4u)), value);
      words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    }
    append_vop2(words, kOpOrB32, dst, static_cast<uint16_t>(256u + value), dst);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
  };

  for (uint8_t dst_word = 0; dst_word < 2; ++dst_word) {
    const uint8_t dst = static_cast<uint8_t>(dst_base + dst_word);
    const uint32_t base_byte_offset = static_cast<uint32_t>(dst_word) * 64u;
    for (uint8_t raw_word = 0; raw_word < 2; ++raw_word) {
      const uint32_t lane_mask = raw_word == 0 ? 0x00FF00FFu : 0xFF00FF00u;
      for (uint8_t lane_in_group = 0; lane_in_group < 8; ++lane_in_group)
        append_component(dst, raw_word, lane_mask, base_byte_offset, lane_in_group);
    }
  }

  append_restore_exec(words, exec_save);
}

void append_tr6_b96_repack(std::vector<uint32_t> &words, uint32_t host_arch, uint8_t dst_base,
                           uint8_t raw_load, uint8_t exec_save, uint8_t lane, uint8_t part,
                           uint8_t addr, uint8_t shift, uint8_t gather0, uint8_t gather1,
                           uint8_t gather2, uint8_t value) {
  constexpr uint8_t kOpMovB32 = 1;
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpOrB32 = 28;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpLshrrevB32 = 25;
  constexpr uint8_t kOpAddNcU32 = 37;
  constexpr uint8_t kOpWaitDscnt = 70;
  constexpr uint16_t kInline0 = 128;
  constexpr uint16_t kInline1 = 129;
  constexpr uint16_t kInline2 = 130;
  constexpr uint16_t kInline3 = 131;
  constexpr uint16_t kInline15 = 143;
  constexpr uint16_t kInline16 = 144;
  constexpr uint16_t kInline31 = 159;
  constexpr uint16_t kInline63 = 191;

  append_set_exec_lo_mask(words, 0xFFFFFFFFu);
  append_lane_id(words, lane);
  append_vop2(words, kOpAndB32, part, kInline15, lane);
  words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
  append_vop2(words, kOpLshlrevB32, shift, kInline2, part);
  append_vop2(words, kOpLshlrevB32, addr, kInline1, part);
  words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
  append_vop2(words, kOpAddNcU32, shift, static_cast<uint16_t>(256u + addr), shift);
  words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
  append_vop2(words, kOpAndB32, shift, kInline31, shift);
  words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));

  append_restore_exec(words, exec_save);
  append_vop1(words, kOpMovB32, dst_base, kInline0);
  append_vop1(words, kOpMovB32, static_cast<uint8_t>(dst_base + 1u), kInline0);
  append_vop1(words, kOpMovB32, static_cast<uint8_t>(dst_base + 2u), kInline0);

  auto append_source_addr = [&](uint8_t input_lane_in_pass) {
    const uint32_t source_lane_base = 32u * static_cast<uint32_t>(input_lane_in_pass / 4u) +
                                      4u * static_cast<uint32_t>(input_lane_in_pass % 4u);
    append_set_exec_lo_mask(words, 0xFFFFFFFFu);
    append_vop2(words, kOpAndB32, addr, kInline16, lane);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    if (source_lane_base != 0) {
      std::optional<uint32_t> literal;
      append_vop2(words, kOpAddNcU32, addr, literal_or_inline_u32(source_lane_base, literal), addr,
                  literal);
      words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    }
  };

  auto append_or_value_bits = [&](uint8_t dst, uint8_t right_shift, uint16_t mask,
                                  uint8_t left_shift) {
    append_vop2(words, kOpOrB32, part, kInline0, value);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    if (right_shift != 0) {
      append_vop2(words, kOpLshrrevB32, part, scalar_positive_inline_u32(right_shift), part);
      words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    }
    if (mask != kInline63) {
      append_vop2(words, kOpAndB32, part, mask, part);
      words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    }
    if (left_shift != 0) {
      append_vop2(words, kOpLshlrevB32, part, scalar_positive_inline_u32(left_shift), part);
      words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    }
    append_vop2(words, kOpOrB32, dst, static_cast<uint16_t>(256u + part), dst);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
  };

  auto append_output_contribution = [&](uint8_t input_lane_in_pass) {
    const uint8_t dst0 = dst_base;
    const uint8_t dst1 = static_cast<uint8_t>(dst_base + 1u);
    const uint8_t dst2 = static_cast<uint8_t>(dst_base + 2u);
    if (input_lane_in_pass <= 4) {
      append_or_value_bits(dst0, 0, kInline63, static_cast<uint8_t>(input_lane_in_pass * 6u));
    } else if (input_lane_in_pass == 5) {
      append_or_value_bits(dst0, 0, kInline3, 30);
      append_or_value_bits(dst1, 2, kInline15, 0);
    } else if (input_lane_in_pass <= 9) {
      append_or_value_bits(
          dst1, 0, kInline63,
          static_cast<uint8_t>(static_cast<uint32_t>(input_lane_in_pass) * 6u - 32u));
    } else if (input_lane_in_pass == 10) {
      append_or_value_bits(dst1, 0, kInline15, 28);
      append_or_value_bits(dst2, 4, kInline3, 0);
    } else {
      append_or_value_bits(
          dst2, 0, kInline63,
          static_cast<uint8_t>(static_cast<uint32_t>(input_lane_in_pass) * 6u - 64u));
    }
  };

  auto append_extract_from_gather = [&](uint32_t lane_mask, uint8_t gather_word, bool cross,
                                        uint8_t cross_shift) {
    append_set_exec_from_saved_mask(words, exec_save, lane_mask);
    append_vop2(words, kOpLshrrevB32, value,
                cross ? scalar_positive_inline_u32(cross_shift)
                      : static_cast<uint16_t>(256u + shift),
                gather_word);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    if (cross) {
      append_vop2(words, kOpLshlrevB32, part, scalar_positive_inline_u32(32u - cross_shift),
                  static_cast<uint8_t>(gather_word + 1u));
      words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
      append_vop2(words, kOpOrB32, value, static_cast<uint16_t>(256u + part), value);
      words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    }
    append_vop2(words, kOpAndB32, value, kInline63, value);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
  };

  constexpr std::array<uint32_t, 5> kLaneMasks = {0x001F001Fu, 0x00200020u, 0x03C003C0u,
                                                  0x04000400u, 0xF800F800u};

  for (uint8_t input_lane = 0; input_lane < 16; ++input_lane) {
    append_source_addr(input_lane);
    auto [bp0, bp1] = build_ds_bpermute(gather0, addr, raw_load);
    words.push_back(bp0);
    words.push_back(bp1);
    auto [bp2, bp3] = build_ds_bpermute(gather1, addr, static_cast<uint8_t>(raw_load + 1u));
    words.push_back(bp2);
    words.push_back(bp3);
    auto [bp4, bp5] = build_ds_bpermute(gather2, addr, static_cast<uint8_t>(raw_load + 2u));
    words.push_back(bp4);
    words.push_back(bp5);
    words.push_back(pack_sopp(kOpWaitDscnt, 0));

    append_extract_from_gather(kLaneMasks[0], gather0, false, 0);
    append_output_contribution(input_lane);
    append_extract_from_gather(kLaneMasks[1], gather0, true, 30);
    append_output_contribution(input_lane);
    append_extract_from_gather(kLaneMasks[2], gather1, false, 0);
    append_output_contribution(input_lane);
    append_extract_from_gather(kLaneMasks[3], gather1, true, 28);
    append_output_contribution(input_lane);
    append_extract_from_gather(kLaneMasks[4], gather2, false, 0);
    append_output_contribution(input_lane);
  }

  append_restore_exec(words, exec_save);
}

void append_tr16_b128_repack(std::vector<uint32_t> &words, uint32_t host_arch, uint8_t dst_base,
                             uint8_t raw_load, uint8_t exec_save, uint8_t lane, uint8_t addr,
                             uint8_t shift, uint8_t value) {
  constexpr uint8_t kOpMovB32 = 1;
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpOrB32 = 28;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpLshrrevB32 = 25;
  constexpr uint8_t kOpAddNcU32 = 37;
  constexpr uint8_t kOpWaitDscnt = 70;
  constexpr uint16_t kInline0 = 128;
  constexpr uint16_t kInline1 = 129;
  constexpr uint16_t kInline2 = 130;
  constexpr uint16_t kInline24 = 152;

  append_set_exec_lo_mask(words, 0xFFFFFFFFu);
  append_lane_id(words, lane);
  append_vop2(words, kOpAndB32, shift, kInline1, lane);
  words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
  append_vop2(words, kOpLshlrevB32, shift, scalar_positive_inline_u32(4), shift);
  words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));

  append_restore_exec(words, exec_save);
  for (uint8_t dst_word = 0; dst_word < 4; ++dst_word)
    append_vop1(words, kOpMovB32, static_cast<uint8_t>(dst_base + dst_word), kInline0);

  auto append_addr_for_source = [&](uint8_t dst_word, uint8_t lane_half) {
    append_set_exec_lo_mask(words, 0xFFFFFFFFu);
    append_vop2(words, kOpAndB32, addr, kInline24, lane);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    append_vop2(words, kOpLshlrevB32, addr, kInline2, addr);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    const uint32_t addend =
        static_cast<uint32_t>(dst_word) * 8u + static_cast<uint32_t>(lane_half) * 4u;
    if (addend != 0) {
      std::optional<uint32_t> literal;
      append_vop2(words, kOpAddNcU32, addr, literal_or_inline_u32(addend, literal), addr, literal);
      words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    }
  };

  auto append_component = [&](uint8_t dst, uint8_t raw_word, uint32_t lane_mask, uint8_t dst_word,
                              uint8_t lane_half) {
    append_addr_for_source(dst_word, lane_half);
    auto [bp0, bp1] = build_ds_bpermute(value, addr, static_cast<uint8_t>(raw_load + raw_word));
    words.push_back(bp0);
    words.push_back(bp1);
    words.push_back(pack_sopp(kOpWaitDscnt, 0));

    append_set_exec_from_saved_mask(words, exec_save, lane_mask);
    append_vop2(words, kOpLshrrevB32, value, static_cast<uint16_t>(256u + shift), value);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    std::optional<uint32_t> mask_literal;
    append_vop2(words, kOpAndB32, value, literal_or_inline_u32(0xFFFFu, mask_literal), value,
                mask_literal);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    if (lane_half != 0) {
      append_vop2(words, kOpLshlrevB32, value, scalar_positive_inline_u32(16), value);
      words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    }
    append_vop2(words, kOpOrB32, dst, static_cast<uint16_t>(256u + value), dst);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
  };

  constexpr std::array<uint32_t, 4> kRawWordMasks = {0x03030303u, 0x0C0C0C0Cu, 0x30303030u,
                                                     0xC0C0C0C0u};
  for (uint8_t dst_word = 0; dst_word < 4; ++dst_word) {
    const uint8_t dst = static_cast<uint8_t>(dst_base + dst_word);
    for (uint8_t raw_word = 0; raw_word < 4; ++raw_word) {
      for (uint8_t lane_half = 0; lane_half < 2; ++lane_half)
        append_component(dst, raw_word, kRawWordMasks[raw_word], dst_word, lane_half);
    }
  }

  append_restore_exec(words, exec_save);
}

bool append_ds_tr4_b64_direct_repack(std::vector<uint32_t> &words, uint32_t host_arch,
                                     VdsFields fields, uint8_t dst_base, uint8_t src_addr,
                                     uint32_t byte_offset, uint8_t exec_save, uint8_t lane,
                                     uint8_t addr, uint8_t byte_index, uint8_t nibble_shift,
                                     uint8_t value) {
  constexpr uint8_t kOpMovB32 = 1;
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpOrB32 = 28;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpLshrrevB32 = 25;
  constexpr uint8_t kOpAddNcU32 = 37;
  constexpr uint8_t kOpDsLoadU8 = 0x3A;
  constexpr uint8_t kOpWaitDscnt = 70;
  constexpr uint8_t kOpWaitLoadcntDscnt = 72;
  constexpr uint16_t kInline0 = 128;
  constexpr uint16_t kInline1 = 129;
  constexpr uint16_t kInline2 = 130;
  constexpr uint16_t kInline7 = 135;
  constexpr uint16_t kInline15 = 143;
  constexpr uint16_t kInline16 = 144;

  append_set_exec_lo_mask(words, 0xFFFFFFFFu);
  append_lane_id(words, lane);
  append_vop2(words, kOpAndB32, nibble_shift, kInline1, lane);
  words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
  append_vop2(words, kOpLshlrevB32, nibble_shift, kInline2, nibble_shift);
  words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
  append_vop2(words, kOpAndB32, byte_index, kInline7, lane);
  words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
  append_vop2(words, kOpLshrrevB32, byte_index, kInline1, byte_index);
  words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));

  append_restore_exec(words, exec_save);
  append_vop1(words, kOpMovB32, dst_base, kInline0);
  append_vop1(words, kOpMovB32, static_cast<uint8_t>(dst_base + 1u), kInline0);

  fields.data0 = 0;
  fields.data1 = 0;
  fields.vaddr = addr;
  fields.vdst = value;

  auto append_addr_for_source = [&](uint32_t base_byte_offset, uint8_t lane_in_group) {
    append_set_exec_lo_mask(words, 0xFFFFFFFFu);
    append_vop2(words, kOpAndB32, addr, kInline16, lane);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    append_vop2(words, kOpLshlrevB32, addr, kInline1, addr);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    const uint32_t source_lane_addend = (base_byte_offset + lane_in_group) * 4u;
    if (source_lane_addend != 0) {
      std::optional<uint32_t> literal;
      append_vop2(words, kOpAddNcU32, addr, literal_or_inline_u32(source_lane_addend, literal),
                  addr, literal);
      words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    }
    auto [bp0, bp1] = build_ds_bpermute(addr, addr, src_addr);
    words.push_back(bp0);
    words.push_back(bp1);
    words.push_back(pack_sopp(kOpWaitDscnt, 0));
    append_vop2(words, kOpAddNcU32, addr, static_cast<uint16_t>(256u + byte_index), addr);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
  };

  auto append_component = [&](uint8_t dst, uint8_t raw_word, uint32_t lane_mask,
                              uint32_t source_lane_base, uint8_t lane_in_group) {
    append_addr_for_source(source_lane_base, lane_in_group);
    append_set_exec_from_saved_mask(words, exec_save, lane_mask);
    set_vds_byte_offset(fields, byte_offset + static_cast<uint32_t>(raw_word) * 4u);
    if (!append_rdna4_vds_inst(words, fields, kOpDsLoadU8))
      return false;
    words.push_back(pack_sopp(kOpWaitLoadcntDscnt, 0));
    append_vop2(words, kOpLshrrevB32, value, static_cast<uint16_t>(256u + nibble_shift), value);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    append_vop2(words, kOpAndB32, value, kInline15, value);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    if (lane_in_group != 0) {
      append_vop2(words, kOpLshlrevB32, value,
                  scalar_positive_inline_u32(static_cast<uint16_t>(lane_in_group * 4u)), value);
      words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    }
    append_vop2(words, kOpOrB32, dst, static_cast<uint16_t>(256u + value), dst);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    return true;
  };

  for (uint8_t dst_word = 0; dst_word < 2; ++dst_word) {
    const uint8_t dst = static_cast<uint8_t>(dst_base + dst_word);
    const uint32_t source_lane_base = static_cast<uint32_t>(dst_word) * 16u;
    for (uint8_t raw_word = 0; raw_word < 2; ++raw_word) {
      const uint32_t lane_mask = raw_word == 0 ? 0x00FF00FFu : 0xFF00FF00u;
      for (uint8_t lane_in_group = 0; lane_in_group < 8; ++lane_in_group) {
        if (!append_component(dst, raw_word, lane_mask, source_lane_base, lane_in_group))
          return false;
      }
    }
  }

  append_restore_exec(words, exec_save);
  return true;
}

bool append_ds_tr8_b64_direct_repack(std::vector<uint32_t> &words, uint32_t host_arch,
                                     VdsFields fields, uint8_t dst_base, uint8_t src_addr,
                                     uint32_t byte_offset, uint8_t exec_save, uint8_t lane,
                                     uint8_t addr, uint8_t byte_index, uint8_t group,
                                     uint8_t value) {
  constexpr uint8_t kOpMovB32 = 1;
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpOrB32 = 28;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpAddNcU32 = 37;
  constexpr uint8_t kOpDsLoadU8 = 0x3A;
  constexpr uint8_t kOpWaitDscnt = 70;
  constexpr uint8_t kOpWaitLoadcntDscnt = 72;
  constexpr uint16_t kInline0 = 128;
  constexpr uint16_t kInline1 = 129;
  constexpr uint16_t kInline3 = 131;
  constexpr uint16_t kInline8 = 136;
  constexpr uint16_t kInline16 = 144;

  append_set_exec_lo_mask(words, 0xFFFFFFFFu);
  append_lane_id(words, lane);
  append_vop2(words, kOpAndB32, byte_index, kInline3, lane);
  words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));

  append_restore_exec(words, exec_save);
  append_vop1(words, kOpMovB32, dst_base, kInline0);
  append_vop1(words, kOpMovB32, static_cast<uint8_t>(dst_base + 1u), kInline0);

  fields.data0 = 0;
  fields.data1 = 0;
  fields.vaddr = addr;
  fields.vdst = value;

  auto append_addr_for_source = [&](uint32_t source_lane_addend) {
    append_set_exec_lo_mask(words, 0xFFFFFFFFu);
    append_vop2(words, kOpAndB32, addr, kInline16, lane);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    append_vop2(words, kOpLshlrevB32, addr, scalar_positive_inline_u32(2), addr);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    append_vop2(words, kOpAndB32, group, kInline8, lane);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    append_vop2(words, kOpLshlrevB32, group, kInline1, group);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    append_vop2(words, kOpAddNcU32, addr, static_cast<uint16_t>(256u + group), addr);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    if (source_lane_addend != 0) {
      std::optional<uint32_t> literal;
      append_vop2(words, kOpAddNcU32, addr, literal_or_inline_u32(source_lane_addend * 4u, literal),
                  addr, literal);
      words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    }
    auto [bp0, bp1] = build_ds_bpermute(addr, addr, src_addr);
    words.push_back(bp0);
    words.push_back(bp1);
    words.push_back(pack_sopp(kOpWaitDscnt, 0));
    append_vop2(words, kOpAddNcU32, addr, static_cast<uint16_t>(256u + byte_index), addr);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
  };

  auto append_component = [&](uint8_t dst, uint8_t raw_word, uint32_t lane_mask,
                              uint32_t source_lane_base, uint8_t lane_in_group) {
    append_addr_for_source(source_lane_base + lane_in_group);
    append_set_exec_from_saved_mask(words, exec_save, lane_mask);
    set_vds_byte_offset(fields, byte_offset + static_cast<uint32_t>(raw_word) * 4u);
    if (!append_rdna4_vds_inst(words, fields, kOpDsLoadU8))
      return false;
    words.push_back(pack_sopp(kOpWaitLoadcntDscnt, 0));
    if (lane_in_group != 0) {
      append_vop2(words, kOpLshlrevB32, value,
                  scalar_positive_inline_u32(static_cast<uint16_t>(lane_in_group * 8u)), value);
      words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    }
    append_vop2(words, kOpOrB32, dst, static_cast<uint16_t>(256u + value), dst);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
    return true;
  };

  for (uint8_t dst_word = 0; dst_word < 2; ++dst_word) {
    const uint8_t dst = static_cast<uint8_t>(dst_base + dst_word);
    const uint32_t source_lane_base = static_cast<uint32_t>(dst_word) * 8u;
    for (uint8_t raw_word = 0; raw_word < 2; ++raw_word) {
      const uint32_t lane_mask = raw_word == 0 ? 0x0F0F0F0Fu : 0xF0F0F0F0u;
      for (uint8_t lane_in_group = 0; lane_in_group < 4; ++lane_in_group) {
        if (!append_component(dst, raw_word, lane_mask, source_lane_base, lane_in_group))
          return false;
      }
    }
  }

  append_restore_exec(words, exec_save);
  return true;
}

std::vector<uint32_t> expand_global_load_tr4_b64(const Instruction &inst, uint32_t host_arch,
                                                 uint64_t, const LivenessAnalysis &liveness,
                                                 const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::VglobalMachineInst))
    return {};

  gfx1250::VglobalMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.op != 115 || src.vdst >= 255)
    return {};

  std::vector<uint8_t> avoid;
  add_avoid_vgpr_run(avoid, static_cast<uint8_t>(src.vdst), 2);
  add_avoid_vgpr_run(avoid, static_cast<uint8_t>(src.vaddr), 2);

  const auto tmp_opt = find_free_vgpr_run_avoiding(inst, liveness, 6, avoid);
  if (!tmp_opt || *tmp_opt > 250)
    return {};
  const auto tmp = static_cast<uint8_t>(*tmp_opt);
  const uint8_t raw_load = tmp;
  const uint8_t lane = static_cast<uint8_t>(tmp + 2u);
  const uint8_t addr = static_cast<uint8_t>(tmp + 3u);
  const uint8_t shift = static_cast<uint8_t>(tmp + 4u);
  const uint8_t value = static_cast<uint8_t>(tmp + 5u);

  auto exec_save_opt = liveness.find_free_sgpr_pair(&inst);
  if (!exec_save_opt || *exec_save_opt > 124)
    return {};
  const uint8_t exec_save = static_cast<uint8_t>(*exec_save_opt);

  constexpr uint8_t kOpGlobalLoadB64 = 21;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpWaitLoadcnt = 64;

  auto fields = gfx1250_to_rdna4::decode_vglobal_gfx1250(raw[0], raw[1], raw[2]);
  fields.vdst = raw_load;
  fields.scale_offset = 0;

  std::vector<uint32_t> words;
  words.reserve(260);
  append_save_exec(words, exec_save);

  if (src.scale_offset != 0) {
    const auto scale_shift = scaled_offset_shift_for_vglobal(src.op);
    if (!scale_shift)
      return {};
    fields.vaddr = addr;
    append_vop2(words, kOpLshlrevB32, addr, scalar_positive_inline_u32(*scale_shift), src.vaddr);
    words.push_back(build_s_delay_alu(1, static_cast<rj_code_arch_t>(host_arch)));
  }
  if (!append_rdna4_vglobal_load(words, fields, kOpGlobalLoadB64))
    return {};

  words.push_back(pack_sopp(kOpWaitLoadcnt, 0));
  append_tr4_b64_repack(words, host_arch, static_cast<uint8_t>(src.vdst), raw_load, exec_save, lane,
                        addr, shift, value);
  return words;
}

std::vector<uint32_t> expand_global_load_tr6_b96(const Instruction &inst, uint32_t host_arch,
                                                 uint64_t, const LivenessAnalysis &liveness,
                                                 const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::VglobalMachineInst))
    return {};

  gfx1250::VglobalMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.op != 116 || src.vdst > 253 || src.scale_offset != 0)
    return {};

  std::vector<uint8_t> avoid;
  add_avoid_vgpr_run(avoid, static_cast<uint8_t>(src.vdst), 3);
  add_avoid_vgpr_run(avoid, static_cast<uint8_t>(src.vaddr), 2);

  const auto tmp_opt = find_free_vgpr_run_avoiding(inst, liveness, 11, avoid);
  if (!tmp_opt || *tmp_opt > 245)
    return {};
  const auto tmp = static_cast<uint8_t>(*tmp_opt);
  const uint8_t raw_load = tmp;
  const uint8_t lane = static_cast<uint8_t>(tmp + 3u);
  const uint8_t part = static_cast<uint8_t>(tmp + 4u);
  const uint8_t addr = static_cast<uint8_t>(tmp + 5u);
  const uint8_t shift = static_cast<uint8_t>(tmp + 6u);
  const uint8_t gather0 = static_cast<uint8_t>(tmp + 7u);
  const uint8_t gather1 = static_cast<uint8_t>(tmp + 8u);
  const uint8_t gather2 = static_cast<uint8_t>(tmp + 9u);
  const uint8_t value = static_cast<uint8_t>(tmp + 10u);

  auto exec_save_opt = liveness.find_free_sgpr_pair(&inst);
  if (!exec_save_opt || *exec_save_opt > 124)
    return {};
  const uint8_t exec_save = static_cast<uint8_t>(*exec_save_opt);

  constexpr uint8_t kOpGlobalLoadB96 = 22;
  constexpr uint8_t kOpWaitLoadcnt = 64;

  auto fields = gfx1250_to_rdna4::decode_vglobal_gfx1250(raw[0], raw[1], raw[2]);
  fields.vdst = raw_load;
  fields.scale_offset = 0;

  std::vector<uint32_t> words;
  words.reserve(620);
  append_save_exec(words, exec_save);

  if (!append_rdna4_vglobal_load(words, fields, kOpGlobalLoadB96))
    return {};
  words.push_back(pack_sopp(kOpWaitLoadcnt, 0));
  append_tr6_b96_repack(words, host_arch, static_cast<uint8_t>(src.vdst), raw_load, exec_save, lane,
                        part, addr, shift, gather0, gather1, gather2, value);
  return words;
}

std::vector<uint32_t> expand_ds_store_2addr_b64(const Instruction &inst, uint32_t, uint64_t,
                                                const LivenessAnalysis &, const LaneLayout *,
                                                const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::VdsMachineInst))
    return {};

  gfx1250::VdsMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.op != 0x4E)
    return {};

  constexpr uint8_t kOpDsStoreB64 = 0x4D;
  constexpr uint8_t kOpWaitStorecnt = 65;
  auto fields = gfx1250_to_rdna4::decode_vds_gfx1250(raw[0], raw[1]);
  fields.data1 = 0;
  fields.vdst = 0;

  std::vector<uint32_t> words;
  words.reserve(4);
  fields.data0 = src.data0;
  set_vds_byte_offset(fields, static_cast<uint32_t>(src.offset0) * 8u);
  if (!append_rdna4_vds_inst(words, fields, kOpDsStoreB64))
    return {};

  fields.data0 = src.data1;
  set_vds_byte_offset(fields, static_cast<uint32_t>(src.offset1) * 8u);
  if (!append_rdna4_vds_inst(words, fields, kOpDsStoreB64))
    return {};

  words.push_back(pack_sopp(kOpWaitStorecnt, 0));
  return words;
}

std::vector<uint32_t> expand_ds_transpose_load(const Instruction &inst, uint32_t host_arch,
                                               uint64_t, const LivenessAnalysis &liveness,
                                               const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::VdsMachineInst))
    return {};

  gfx1250::VdsMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));

  constexpr uint8_t kOpDsLoadTr4B64 = 0xFA;
  constexpr uint8_t kOpDsLoadTr6B96 = 0xFB;
  constexpr uint8_t kOpDsLoadTr16B128 = 0xFC;
  constexpr uint8_t kOpDsLoadTr8B64 = 0xFD;
  constexpr uint8_t kOpDsLoadB96 = 0xFE;

  struct Plan {
    uint8_t dst_words;
    uint8_t tmp_words;
    uint16_t reserve_words;
  };

  Plan plan{};
  switch (src.op) {
  case kOpDsLoadTr4B64:
    plan = {2, 8, 9500};
    break;
  case kOpDsLoadTr6B96:
    plan = {3, 11, 620};
    break;
  case kOpDsLoadTr16B128:
    plan = {4, 12, 5200};
    break;
  case kOpDsLoadTr8B64:
    plan = {2, 8, 5200};
    break;
  default:
    return {};
  }

  if (src.vdst > 256u - plan.dst_words)
    return {};

  std::vector<uint8_t> avoid;
  add_avoid_vgpr_run(avoid, static_cast<uint8_t>(src.vdst), plan.dst_words);
  add_avoid_vgpr(avoid, static_cast<uint8_t>(src.addr));

  const auto tmp_opt = find_free_vgpr_run_avoiding(inst, liveness, plan.tmp_words, avoid);
  if (!tmp_opt || *tmp_opt > 256u - plan.tmp_words)
    return {};
  const uint8_t tmp = static_cast<uint8_t>(*tmp_opt);
  const uint8_t raw_load = tmp;

  auto exec_save_opt = liveness.find_free_sgpr_pair(&inst);
  if (!exec_save_opt || *exec_save_opt > 124)
    return {};
  const uint8_t exec_save = static_cast<uint8_t>(*exec_save_opt);

  constexpr uint8_t kOpWaitLoadcntDscnt = 72;

  auto fields = gfx1250_to_rdna4::decode_vds_gfx1250(raw[0], raw[1]);
  const uint32_t byte_offset =
      static_cast<uint32_t>(fields.offset0) | (static_cast<uint32_t>(fields.offset1) << 8);

  std::vector<uint32_t> words;
  words.reserve(plan.reserve_words);
  append_save_exec(words, exec_save);

  switch (src.op) {
  case kOpDsLoadTr4B64: {
    const uint8_t lane = tmp;
    const uint8_t addr = static_cast<uint8_t>(tmp + 1u);
    const uint8_t byte_index = static_cast<uint8_t>(tmp + 2u);
    const uint8_t nibble_shift = static_cast<uint8_t>(tmp + 3u);
    const uint8_t value = static_cast<uint8_t>(tmp + 4u);
    if (!append_ds_tr4_b64_direct_repack(words, host_arch, fields, static_cast<uint8_t>(src.vdst),
                                         static_cast<uint8_t>(src.addr), byte_offset, exec_save,
                                         lane, addr, byte_index, nibble_shift, value))
      return {};
    break;
  }
  case kOpDsLoadTr6B96: {
    fields.vdst = raw_load;
    fields.data0 = 0;
    fields.data1 = 0;
    if (!append_rdna4_vds_inst(words, fields, kOpDsLoadB96))
      return {};
    words.push_back(pack_sopp(kOpWaitLoadcntDscnt, 0));
    const uint8_t lane = static_cast<uint8_t>(tmp + 3u);
    const uint8_t part = static_cast<uint8_t>(tmp + 4u);
    const uint8_t addr = static_cast<uint8_t>(tmp + 5u);
    const uint8_t shift = static_cast<uint8_t>(tmp + 6u);
    const uint8_t gather0 = static_cast<uint8_t>(tmp + 7u);
    const uint8_t gather1 = static_cast<uint8_t>(tmp + 8u);
    const uint8_t gather2 = static_cast<uint8_t>(tmp + 9u);
    const uint8_t value = static_cast<uint8_t>(tmp + 10u);
    append_tr6_b96_repack(words, host_arch, static_cast<uint8_t>(src.vdst), raw_load, exec_save,
                          lane, part, addr, shift, gather0, gather1, gather2, value);
    break;
  }
  case kOpDsLoadTr16B128: {
    if (!append_rdna4_ds_load_b32_sequence(words, fields, raw_load, 4, byte_offset))
      return {};
    words.push_back(pack_sopp(kOpWaitLoadcntDscnt, 0));
    const uint8_t copy = static_cast<uint8_t>(tmp + 4u);
    append_copy_vgpr_words(words, host_arch, copy, raw_load, 4);
    const uint8_t lane = static_cast<uint8_t>(tmp + 8u);
    const uint8_t addr = static_cast<uint8_t>(tmp + 9u);
    const uint8_t shift = static_cast<uint8_t>(tmp + 10u);
    const uint8_t value = static_cast<uint8_t>(tmp + 11u);
    append_tr16_b128_repack(words, host_arch, static_cast<uint8_t>(src.vdst), copy, exec_save, lane,
                            addr, shift, value);
    break;
  }
  case kOpDsLoadTr8B64: {
    const uint8_t lane = tmp;
    const uint8_t addr = static_cast<uint8_t>(tmp + 1u);
    const uint8_t byte_index = static_cast<uint8_t>(tmp + 2u);
    const uint8_t group = static_cast<uint8_t>(tmp + 3u);
    const uint8_t value = static_cast<uint8_t>(tmp + 4u);
    if (!append_ds_tr8_b64_direct_repack(words, host_arch, fields, static_cast<uint8_t>(src.vdst),
                                         static_cast<uint8_t>(src.addr), byte_offset, exec_save,
                                         lane, addr, byte_index, group, value))
      return {};
    break;
  }
  default:
    return {};
  }

  return words;
}

std::vector<uint32_t> expand_native_global_transpose_load(const Instruction &inst, uint32_t,
                                                          uint64_t,
                                                          const LivenessAnalysis &liveness,
                                                          const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::VglobalMachineInst))
    return {};

  gfx1250::VglobalMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.op != 87 && src.op != 88)
    return {};

  auto fields = gfx1250_to_rdna4::decode_vglobal_gfx1250(raw[0], raw[1], raw[2]);
  fields.scale_offset = 0;

  std::vector<uint32_t> words;
  words.reserve(src.scale_offset ? 5 : 3);

  if (src.scale_offset != 0) {
    const auto shift = scaled_offset_shift_for_vglobal(src.op);
    if (!shift)
      return {};

    std::vector<uint8_t> avoid;
    add_avoid_vgpr_run(avoid, static_cast<uint8_t>(src.vdst), src.op == 87 ? 4 : 2);
    add_avoid_vgpr_run(avoid, static_cast<uint8_t>(src.vaddr), 2);
    const auto tmp_opt = find_free_vgpr_run_avoiding(inst, liveness, 1, avoid);
    if (!tmp_opt || *tmp_opt > 255)
      return {};

    constexpr uint8_t kOpLshlrevB32 = 24;
    const uint8_t tmp = static_cast<uint8_t>(*tmp_opt);
    fields.vaddr = tmp;
    append_vop2(words, kOpLshlrevB32, tmp, scalar_positive_inline_u32(*shift), src.vaddr);
    append_wait_valu_vgpr(words);
  }

  if (!append_rdna4_vglobal_load(words, fields, src.op))
    return {};
  return words;
}

std::vector<uint32_t> lower_s_clause_to_nop(const Instruction &, uint32_t host_arch, uint64_t,
                                            const LivenessAnalysis &, const LaneLayout *,
                                            const LaneLayout *) {
  return {build_s_nop(0, static_cast<rj_code_arch_t>(host_arch))};
}

std::vector<uint32_t> lower_s_set_vgpr_msb_to_setreg(const Instruction &inst, uint32_t, uint64_t,
                                                     const LivenessAnalysis &, const LaneLayout *,
                                                     const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() != sizeof(uint32_t))
    return {};

  const auto sopp = std::bit_cast<gfx1250::SoppMachineInst>(raw[0]);

  constexpr uint8_t kOpSSetregImm32B32 = 19;
  constexpr uint8_t kHwregMode = 1;
  constexpr uint8_t kVgprMsbModeOffset = amdgpu::VGPR_MSB_MODE_SHIFT;
  constexpr uint8_t kVgprMsbModeSize = 8;
  const uint16_t hwreg = build_hwreg(kHwregMode, kVgprMsbModeOffset, kVgprMsbModeSize);
  const uint32_t mode_literal =
      amdgpu::set_vgpr_msb_to_mode_layout(amdgpu::s_set_vgpr_msb_new_mode(sopp.simm16));
  return {build_sopk(kOpSSetregImm32B32, hwreg), mode_literal};
}

std::vector<uint32_t> lower_s_mov_b64_literal64(const Instruction &inst, uint32_t host_arch,
                                                uint64_t, const LivenessAnalysis &,
                                                const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() != 3 * static_cast<int>(sizeof(uint32_t)))
    return {};

  const auto src = std::bit_cast<gfx1250::Sop1MachineInst>(raw[0]);
  if (src.ssrc0 != 254)
    return {};

  const auto literal = simm64_literal_value(inst, 0);
  if (!literal)
    return {};

  const uint32_t lo = static_cast<uint32_t>(*literal);
  const uint32_t hi = static_cast<uint32_t>(*literal >> 32);
  const int64_t sign_extended_lo = static_cast<int64_t>(static_cast<int32_t>(lo));
  if (static_cast<uint64_t>(sign_extended_lo) == *literal) {
    return {pack_sop1(1, src.sdst, 255), lo,
            build_s_nop(0, static_cast<rj_code_arch_t>(host_arch))};
  }

  if (src.sdst == 127)
    return {};

  std::vector<uint32_t> words;
  words.reserve(4);
  words.push_back(pack_sop1(0, src.sdst, 255));
  words.push_back(lo);
  if (auto hi_inline = scalar_inline_i32(static_cast<int32_t>(hi))) {
    words.push_back(pack_sop1(0, src.sdst + 1u, *hi_inline));
  } else {
    words.push_back(pack_sop1(0, src.sdst + 1u, 255));
    words.push_back(hi);
  }
  return words;
}

std::vector<uint32_t> lower_s_and_b64_literal64(const Instruction &inst, uint32_t host_arch,
                                                uint64_t, const LivenessAnalysis &,
                                                const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() != 3 * static_cast<int>(sizeof(uint32_t)))
    return {};

  const auto src = std::bit_cast<gfx1250::Sop2MachineInst>(raw[0]);
  if (src.sdst == 127)
    return {};

  uint16_t non_literal_src = 0;
  std::optional<uint64_t> literal;
  if (src.ssrc0 == 254 && src.ssrc1 != 254) {
    non_literal_src = src.ssrc1;
    literal = simm64_literal_value(inst, 0);
  } else if (src.ssrc1 == 254 && src.ssrc0 != 254) {
    non_literal_src = src.ssrc0;
    literal = simm64_literal_value(inst, 1);
  } else {
    return {};
  }
  if (!literal)
    return {};

  const auto non_literal_hi = pair_hi_src_sign_extended_inline(non_literal_src);
  if (!non_literal_hi)
    return {};

  constexpr uint8_t kOpSAndB32 = 22;
  const uint32_t literal_lo = static_cast<uint32_t>(*literal);
  const uint32_t literal_hi = static_cast<uint32_t>(*literal >> 32);

  const auto is_gfx1250_flat_address_alignment_mask = [](uint64_t mask) {
    const uint32_t lo = static_cast<uint32_t>(mask);
    const uint32_t hi = static_cast<uint32_t>(mask >> 32u);
    return (lo == 0xFFFF'F000u || lo == 0xFFFF'E000u || lo == 0xFFFF'C000u) &&
           (hi == 0x7FFF'FFFFu || hi == 0x1FFF'FFFFu);
  };
  if (is_gfx1250_flat_address_alignment_mask(*literal)) {
    // IREE uses this gfx1250 flat-address alignment idiom before RDNA4 VMEM.
    // Carrying the gfx1250 high aperture tag into the host address high word
    // can hang the dispatch, so canonicalize the host high word.
    std::vector<uint32_t> words;
    words.reserve(3);
    append_sop2_b32_u32(words, kOpSAndB32, src.sdst, non_literal_src, literal_lo);
    words.push_back(build_s_mov_b32(static_cast<uint8_t>(src.sdst + 1u),
                                    scalar_positive_inline_u32(0),
                                    static_cast<rj_code_arch_t>(host_arch)));
    return words;
  }

  std::vector<uint32_t> lo_words;
  std::vector<uint32_t> hi_words;
  append_sop2_b32_u32(lo_words, kOpSAndB32, src.sdst, non_literal_src, literal_lo);
  append_sop2_b32_u32(hi_words, kOpSAndB32, static_cast<uint8_t>(src.sdst + 1u), *non_literal_hi,
                      literal_hi);

  std::vector<uint32_t> words;
  words.reserve(lo_words.size() + hi_words.size());
  if (src.sdst == non_literal_src + 1u) {
    words.insert(words.end(), hi_words.begin(), hi_words.end());
    words.insert(words.end(), lo_words.begin(), lo_words.end());
  } else {
    words.insert(words.end(), lo_words.begin(), lo_words.end());
    words.insert(words.end(), hi_words.begin(), hi_words.end());
  }
  return words;
}

std::vector<uint32_t> lower_s_or_b64_literal64(const Instruction &inst, uint32_t host_arch,
                                               uint64_t, const LivenessAnalysis &,
                                               const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() != 3 * static_cast<int>(sizeof(uint32_t)))
    return {};

  const auto src = std::bit_cast<gfx1250::Sop2MachineInst>(raw[0]);
  if (src.sdst == 127)
    return {};

  uint16_t non_literal_src = 0;
  std::optional<uint64_t> literal;
  if (src.ssrc0 == 254 && src.ssrc1 != 254) {
    non_literal_src = src.ssrc1;
    literal = simm64_literal_value(inst, 0);
  } else if (src.ssrc1 == 254 && src.ssrc0 != 254) {
    non_literal_src = src.ssrc0;
    literal = simm64_literal_value(inst, 1);
  } else {
    return {};
  }
  if (!literal)
    return {};

  const uint32_t literal_lo = static_cast<uint32_t>(*literal);
  const uint32_t literal_hi = static_cast<uint32_t>(*literal >> 32);
  if (literal_lo != 0)
    return {};

  std::vector<uint32_t> words;
  words.reserve(3);
  words.push_back(build_s_or_b64(src.sdst, non_literal_src, scalar_positive_inline_u32(0)));
  if (literal_hi != 0) {
    if (auto hi_inline = scalar_inline_i32(static_cast<int32_t>(literal_hi))) {
      words.push_back(build_s_or_b32(static_cast<uint8_t>(src.sdst + 1u),
                                     static_cast<uint16_t>(src.sdst + 1u), *hi_inline));
    } else {
      words.push_back(build_s_or_b32(static_cast<uint8_t>(src.sdst + 1u),
                                     static_cast<uint16_t>(src.sdst + 1u), 255));
      words.push_back(literal_hi);
    }
  }
  while (words.size() * sizeof(uint32_t) < static_cast<size_t>(inst.size()))
    words.push_back(build_s_nop(0, static_cast<rj_code_arch_t>(host_arch)));
  return words;
}

struct Scalar64Sop2Source {
  uint16_t lo = 0;
  uint16_t hi = 0;
  std::optional<uint32_t> lo_literal;
  std::optional<uint32_t> hi_literal;
};

[[nodiscard]] std::optional<uint16_t> scalar_inline_i32_bitpattern(uint32_t value) {
  if (value <= 64u)
    return scalar_positive_inline_u32(static_cast<uint16_t>(value));
  if (value >= 0xFFFF'FFF0u) {
    const auto signed_value = static_cast<int16_t>(static_cast<int64_t>(value) - (1ll << 32));
    return scalar_negative_inline_i32(signed_value);
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<Scalar64Sop2Source>
sop2_scalar64_source_parts(const Instruction &inst, uint16_t src, uint8_t operand_index) {
  if (src == 255) {
    const auto literal = simm32_literal_word(inst, operand_index);
    if (!literal)
      return std::nullopt;

    Scalar64Sop2Source parts;
    parts.lo = 255;
    parts.hi = (*literal & 0x8000'0000u) != 0u ? scalar_negative_inline_i32(-1)
                                               : scalar_positive_inline_u32(0);
    parts.lo_literal = *literal;
    return parts;
  }

  if (src == 254) {
    const auto literal = simm64_literal_value(inst, operand_index);
    if (!literal)
      return std::nullopt;

    const uint32_t lo = static_cast<uint32_t>(*literal);
    const uint32_t hi = static_cast<uint32_t>(*literal >> 32);
    Scalar64Sop2Source parts;
    if (auto lo_inline = scalar_inline_i32_bitpattern(lo)) {
      parts.lo = *lo_inline;
    } else {
      parts.lo = 255;
      parts.lo_literal = lo;
    }
    if (auto hi_inline = scalar_inline_i32_bitpattern(hi)) {
      parts.hi = *hi_inline;
    } else {
      parts.hi = 255;
      parts.hi_literal = hi;
    }
    return parts;
  }

  const auto src_hi = pair_hi_src_sign_extended_inline(src);
  if (!src_hi)
    return std::nullopt;

  return Scalar64Sop2Source{src, *src_hi, std::nullopt, std::nullopt};
}

ExpandResult lower_s_mul_u64_literal64(const Instruction &inst, uint32_t host_arch, uint64_t,
                                       const LivenessAnalysis &liveness,
                                       TranslationContext &context, const LaneLayout *,
                                       const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() != 3 * static_cast<int>(sizeof(uint32_t)))
    return ExpandResult::not_handled();

  const auto src = std::bit_cast<gfx1250::Sop2MachineInst>(raw[0]);
  const bool src0_is_literal64 = src.ssrc0 == 254;
  const bool src1_is_literal64 = src.ssrc1 == 254;
  if (!src0_is_literal64 && !src1_is_literal64)
    return ExpandResult::not_handled();
  if ((src0_is_literal64 && src.ssrc1 == 255) || (src1_is_literal64 && src.ssrc0 == 255)) {
    return ExpandResult::failed(
        "s_mul_u64 cannot combine a 64-bit literal with a distinct 32-bit literal",
        {"Materialize one literal in an SGPR pair before lowering s_mul_u64."});
  }
  if (src.sdst == 127) {
    return ExpandResult::failed("s_mul_u64 destination pair is out of range");
  }

  const uint8_t literal_operand = src0_is_literal64 ? 0 : 1;
  const auto literal = simm64_literal_value(inst, literal_operand);
  if (!literal) {
    return ExpandResult::failed("Could not decode the gfx1250 s_mul_u64 64-bit literal");
  }

  constexpr uint8_t kOpSMulU64 = 85;
  const uint32_t literal_lo = static_cast<uint32_t>(*literal);
  const uint32_t literal_hi = static_cast<uint32_t>(*literal >> 32u);
  const int64_t sign_extended_lo = static_cast<int64_t>(static_cast<int32_t>(literal_lo));
  if (static_cast<uint64_t>(sign_extended_lo) == *literal) {
    const uint16_t ssrc0 = src0_is_literal64 ? 255 : src.ssrc0;
    const uint16_t ssrc1 = src1_is_literal64 ? 255 : src.ssrc1;
    std::vector<uint32_t> words{pack_sop2(kOpSMulU64, src.sdst, ssrc0, ssrc1), literal_lo};
    append_wait_salu_sgpr(words);
    return ExpandResult::success(std::move(words));
  }

  std::vector<uint8_t> avoid{static_cast<uint8_t>(src.sdst), static_cast<uint8_t>(src.sdst + 1u)};
  const auto literal_pair = find_free_sgpr_pair_avoiding(inst, liveness, avoid);
  if (!literal_pair) {
    return ExpandResult::failed(
        "No free SGPR pair is available to materialize the s_mul_u64 64-bit literal",
        {"Reduce SGPR pressure or add scalar spilling for gfx1250 literal64 lowering."});
  }
  context.require_sgprs(static_cast<uint32_t>(*literal_pair) + 2u);

  std::vector<uint32_t> words;
  words.reserve(8);
  auto append_materialize_word = [&](uint8_t sdst, uint32_t value) {
    if (const auto inline_value = scalar_inline_i32_bitpattern(value)) {
      words.push_back(build_s_mov_b32(sdst, *inline_value, static_cast<rj_code_arch_t>(host_arch)));
    } else {
      append_s_mov_b32_lit(words, sdst, value);
    }
  };
  append_materialize_word(static_cast<uint8_t>(*literal_pair), literal_lo);
  append_materialize_word(static_cast<uint8_t>(*literal_pair + 1u), literal_hi);
  append_wait_salu_sgpr(words);

  const uint16_t ssrc0 = src0_is_literal64 ? *literal_pair : src.ssrc0;
  const uint16_t ssrc1 = src1_is_literal64 ? *literal_pair : src.ssrc1;
  words.push_back(pack_sop2(kOpSMulU64, src.sdst, ssrc0, ssrc1));
  append_wait_salu_sgpr(words);
  return ExpandResult::success(std::move(words));
}

ExpandResult lower_s_cmp_u64_literal64(const Instruction &inst, uint32_t host_arch, uint64_t,
                                       const LivenessAnalysis &liveness,
                                       TranslationContext &context, const LaneLayout *,
                                       const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() != 3 * static_cast<int>(sizeof(uint32_t)))
    return ExpandResult::not_handled();

  const auto src = std::bit_cast<gfx1250::SopcMachineInst>(raw[0]);
  const bool src0_is_literal64 = src.ssrc0 == 254;
  const bool src1_is_literal64 = src.ssrc1 == 254;
  if (!src0_is_literal64 && !src1_is_literal64)
    return ExpandResult::not_handled();
  if ((src0_is_literal64 && src.ssrc1 == 255) || (src1_is_literal64 && src.ssrc0 == 255)) {
    return ExpandResult::failed(
        "A 64-bit scalar compare cannot combine literal64 and literal32 operands");
  }

  const auto literal = simm64_literal_value(inst, src0_is_literal64 ? 0 : 1);
  if (!literal)
    return ExpandResult::failed("Could not decode the gfx1250 scalar-compare literal64");

  std::vector<uint8_t> avoid;
  auto avoid_pair = [&](uint16_t encoded) {
    if (encoded < 128) {
      avoid.push_back(static_cast<uint8_t>(encoded));
      if (encoded < 127)
        avoid.push_back(static_cast<uint8_t>(encoded + 1u));
    }
  };
  if (!src0_is_literal64)
    avoid_pair(static_cast<uint16_t>(src.ssrc0));
  if (!src1_is_literal64)
    avoid_pair(static_cast<uint16_t>(src.ssrc1));

  const auto literal_pair = find_free_sgpr_pair_avoiding(inst, liveness, avoid);
  if (!literal_pair) {
    return ExpandResult::failed(
        "No free SGPR pair is available to materialize a scalar-compare literal64",
        {"Reduce SGPR pressure or add scalar spilling for gfx1250 literal64 lowering."});
  }
  context.require_sgprs(static_cast<uint32_t>(*literal_pair) + 2u);

  std::vector<uint32_t> words;
  words.reserve(7);
  auto append_materialize_word = [&](uint8_t sdst, uint32_t value) {
    if (const auto inline_value = scalar_inline_i32_bitpattern(value)) {
      words.push_back(build_s_mov_b32(sdst, *inline_value, static_cast<rj_code_arch_t>(host_arch)));
    } else {
      append_s_mov_b32_lit(words, sdst, value);
    }
  };
  append_materialize_word(static_cast<uint8_t>(*literal_pair), static_cast<uint32_t>(*literal));
  append_materialize_word(static_cast<uint8_t>(*literal_pair + 1u),
                          static_cast<uint32_t>(*literal >> 32u));
  append_wait_salu_sgpr(words);
  words.push_back(build_sopc(static_cast<uint8_t>(src.op),
                             src0_is_literal64 ? *literal_pair : static_cast<uint16_t>(src.ssrc0),
                             src1_is_literal64 ? *literal_pair : static_cast<uint16_t>(src.ssrc1)));
  return ExpandResult::success(std::move(words));
}

[[nodiscard]] bool append_sop2_from_scalar64_parts(std::vector<uint32_t> &words, uint8_t op,
                                                   uint8_t sdst, const Scalar64Sop2Source &src0,
                                                   const Scalar64Sop2Source &src1, bool high) {
  const uint16_t ssrc0 = high ? src0.hi : src0.lo;
  const uint16_t ssrc1 = high ? src1.hi : src1.lo;
  const std::optional<uint32_t> &literal0 = high ? src0.hi_literal : src0.lo_literal;
  const std::optional<uint32_t> &literal1 = high ? src1.hi_literal : src1.lo_literal;

  std::optional<uint32_t> literal;
  if (ssrc0 == 255) {
    if (!literal0)
      return false;
    literal = *literal0;
  }
  if (ssrc1 == 255) {
    if (literal || !literal1)
      return false;
    literal = *literal1;
  }

  words.push_back(pack_sop2(op, sdst, ssrc0, ssrc1));
  if (literal)
    words.push_back(*literal);
  return true;
}

[[nodiscard]] std::vector<uint32_t> native_sop2_words_with_simm32(const Instruction &inst,
                                                                  gfx1250::Sop2MachineInst src) {
  const auto *raw = inst.raw_encoding();
  if (!raw)
    return {};

  std::vector<uint32_t> words{raw[0]};
  if (src.ssrc0 == 255 || src.ssrc1 == 255) {
    const auto literal = simm32_literal_word(inst, src.ssrc0 == 255 ? 0 : 1);
    if (!literal)
      return {};
    words.push_back(*literal);
  }
  return words;
}

[[nodiscard]] bool scalar_low_write_clobbers_high_source(uint8_t sdst, uint16_t src_hi) {
  return scalar_register_src(src_hi) && sdst == src_hi;
}

void append_scc_save(std::vector<uint32_t> &words, uint8_t tmp_sgpr) {
  constexpr uint8_t kOpSCselectB32 = 48;
  words.push_back(pack_sop2(kOpSCselectB32, tmp_sgpr, scalar_positive_inline_u32(1),
                            scalar_positive_inline_u32(0)));
}

void append_scc_restore(std::vector<uint32_t> &words, uint8_t tmp_sgpr) {
  constexpr uint8_t kOpSCmpLgU32 = 7;
  words.push_back(build_sopc(kOpSCmpLgU32, tmp_sgpr, scalar_positive_inline_u32(0)));
}

[[nodiscard]] std::vector<uint32_t>
native_sop2_words_preserving_scc(const Instruction &inst, const LivenessAnalysis &liveness,
                                 gfx1250::Sop2MachineInst src) {
  std::vector<uint8_t> avoid{static_cast<uint8_t>(src.sdst)};
  if (src.sdst < 127)
    avoid.push_back(static_cast<uint8_t>(src.sdst + 1u));
  const auto scc_save = find_free_sgpr_avoiding(inst, liveness, avoid);
  if (!scc_save)
    return {};

  auto native_words = native_sop2_words_with_simm32(inst, src);
  if (native_words.empty())
    return {};

  std::vector<uint32_t> words;
  words.reserve(native_words.size() + 3u);
  append_scc_save(words, static_cast<uint8_t>(*scc_save));
  words.insert(words.end(), native_words.begin(), native_words.end());
  append_wait_salu_sgpr(words);
  append_scc_restore(words, static_cast<uint8_t>(*scc_save));
  return words;
}

std::vector<uint32_t> lower_s_add_sub_nc_u64_to_carry_chain(const Instruction &inst,
                                                            const LivenessAnalysis &liveness,
                                                            bool subtract) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Sop2MachineInst))
    return {};

  const auto src = std::bit_cast<gfx1250::Sop2MachineInst>(raw[0]);
  if (src.ssrc0 != 254 && src.ssrc1 != 254)
    return native_sop2_words_preserving_scc(inst, liveness, src);

  if (src.sdst == 127)
    return {};

  const auto src0 = sop2_scalar64_source_parts(inst, static_cast<uint16_t>(src.ssrc0), 0);
  const auto src1 = sop2_scalar64_source_parts(inst, static_cast<uint16_t>(src.ssrc1), 1);
  if (!src0 || !src1)
    return {};

  if (scalar_low_write_clobbers_high_source(static_cast<uint8_t>(src.sdst), src0->hi) ||
      scalar_low_write_clobbers_high_source(static_cast<uint8_t>(src.sdst), src1->hi))
    return {};

  std::vector<uint8_t> avoid{static_cast<uint8_t>(src.sdst)};
  if (src.sdst < 127)
    avoid.push_back(static_cast<uint8_t>(src.sdst + 1u));
  const auto scc_save = find_free_sgpr_avoiding(inst, liveness, avoid);
  if (!scc_save)
    return {};

  constexpr uint8_t kOpSAddCoU32 = 0;
  constexpr uint8_t kOpSSubCoU32 = 1;
  constexpr uint8_t kOpSAddCoCiU32 = 4;
  constexpr uint8_t kOpSSubCoCiU32 = 5;

  const uint8_t lo_op = subtract ? kOpSSubCoU32 : kOpSAddCoU32;
  const uint8_t hi_op = subtract ? kOpSSubCoCiU32 : kOpSAddCoCiU32;

  std::vector<uint32_t> words;
  words.reserve(10);
  append_scc_save(words, static_cast<uint8_t>(*scc_save));
  if (!append_sop2_from_scalar64_parts(words, lo_op, static_cast<uint8_t>(src.sdst), *src0, *src1,
                                       false))
    return {};
  append_delay_salu_scc(words);
  append_wait_salu_sgpr(words);
  if (!append_sop2_from_scalar64_parts(words, hi_op, static_cast<uint8_t>(src.sdst + 1u), *src0,
                                       *src1, true))
    return {};
  append_delay_salu_scc(words);
  append_wait_salu_sgpr(words);
  append_scc_restore(words, static_cast<uint8_t>(*scc_save));
  return words;
}

std::vector<uint32_t> lower_s_add_nc_u64_to_carry_chain(const Instruction &inst, uint32_t, uint64_t,
                                                        const LivenessAnalysis &liveness,
                                                        const LaneLayout *, const LaneLayout *) {
  return lower_s_add_sub_nc_u64_to_carry_chain(inst, liveness, false);
}

std::vector<uint32_t> lower_s_sub_nc_u64_to_borrow_chain(const Instruction &inst, uint32_t,
                                                         uint64_t, const LivenessAnalysis &liveness,
                                                         const LaneLayout *, const LaneLayout *) {
  return lower_s_add_sub_nc_u64_to_carry_chain(inst, liveness, true);
}

std::vector<uint32_t> lower_gfx1250_grid_mode_s_getreg_to_zero(const Instruction &inst,
                                                               uint32_t host_arch, uint64_t,
                                                               const LivenessAnalysis &,
                                                               const LaneLayout *,
                                                               const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() != sizeof(uint32_t))
    return {};

  const auto src = std::bit_cast<gfx1250::SopkMachineInst>(raw[0]);
  if (src.simm16 != build_hwreg(/*reg_id=*/28, /*offset=*/6, /*size=*/4))
    return {};

  return {build_s_mov_b32(src.sdst, scalar_positive_inline_u32(0),
                          static_cast<rj_code_arch_t>(host_arch))};
}

std::vector<uint32_t> preserve_gfx1250_replay_mode_s_setreg_imm32(const Instruction &inst, uint32_t,
                                                                  uint64_t,
                                                                  const LivenessAnalysis &,
                                                                  const LaneLayout *,
                                                                  const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() != 2 * static_cast<int>(sizeof(uint32_t)))
    return {};

  const auto src = std::bit_cast<gfx1250::SopkMachineInst>(raw[0]);
  if (src.simm16 != build_hwreg(/*reg_id=*/1, /*offset=*/25, /*size=*/1) || raw[1] != 1)
    return {};

  return {raw[0], raw[1]};
}

std::vector<uint32_t> preserve_same_sop1_encoding(const Instruction &inst, uint32_t, uint64_t,
                                                  const LivenessAnalysis &, const LaneLayout *,
                                                  const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() != static_cast<int>(sizeof(uint32_t)))
    return {};
  return {raw[0]};
}

std::vector<uint32_t> lower_s_sendmsg_rtn_to_rdna4(const Instruction &inst, uint32_t, uint64_t,
                                                   const LivenessAnalysis &, const LaneLayout *,
                                                   const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() != static_cast<int>(sizeof(uint32_t)))
    return {};

  // On GFX12, send-message responses are tracked by kmcnt. Serializing this
  // uncommon operation keeps the returned SGPR value ready even when a guest
  // inline-assembly sequence omitted the counter wait that LLVM normally emits
  // before consuming the response.
  constexpr uint8_t kSoppWaitKmcnt = 0x47;
  return {raw[0], pack_sopp(kSoppWaitKmcnt, 0)};
}

std::vector<uint32_t> lower_gfx1250_resource_word2_movk(const Instruction &inst, uint32_t, uint64_t,
                                                        const LivenessAnalysis &,
                                                        const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() != sizeof(uint32_t))
    return {};

  const auto src = std::bit_cast<gfx1250::SopkMachineInst>(raw[0]);
  const bool descriptor_bound_word = src.sdst == 2 || src.sdst == 6;
  const bool observed_bound = src.simm16 == 512 || src.simm16 == 1024 || src.simm16 == 8192;
  if (!descriptor_bound_word || !observed_bound)
    return {};

  auto [w0, w1] = build_s_mov_b32_lit(src.sdst, static_cast<uint32_t>(src.simm16) << 7);
  if (src.sdst == 2 && src.simm16 == 512) {
    auto [config_w0, config_w1] = build_s_mov_b32_lit(3, 0x31016000u);
    return {w0, w1, config_w0, config_w1};
  }
  return {w0, w1};
}

std::vector<uint32_t> lower_gfx1250_resource_s_mov_b32(const Instruction &inst, uint32_t, uint64_t,
                                                       const LivenessAnalysis &, const LaneLayout *,
                                                       const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() != sizeof(uint32_t))
    return {};

  return {};
}

std::vector<uint32_t> expand_v_mov_b64_sources(uint8_t vdst, uint16_t src0,
                                               std::optional<uint32_t> literal32,
                                               std::optional<uint64_t> literal64) {
  if (vdst == 255)
    return {};

  constexpr uint8_t kOpMovB32 = 1;
  constexpr uint16_t kLiteral32Src = 255;
  constexpr uint16_t kLiteral64Src = 254;

  if (src0 == kLiteral64Src) {
    if (!literal64 || literal32)
      return {};
    return {build_vop1(kOpMovB32, vdst, kLiteral32Src), static_cast<uint32_t>(*literal64),
            build_vop1(kOpMovB32, static_cast<uint8_t>(vdst + 1), kLiteral32Src),
            static_cast<uint32_t>(*literal64 >> 32)};
  }

  if (src0 == kLiteral32Src) {
    if (!literal32 || literal64)
      return {};
    const auto hi = scalar_inline_i32(static_cast<int32_t>(*literal32) < 0 ? -1 : 0);
    if (!hi)
      return {};
    return {build_vop1(kOpMovB32, vdst, kLiteral32Src), *literal32,
            build_vop1(kOpMovB32, static_cast<uint8_t>(vdst + 1), *hi)};
  }

  if (literal32 || literal64)
    return {};
  const auto hi = pair_hi_src_sign_extended_inline(src0);
  if (!hi)
    return {};
  return {build_vop1(kOpMovB32, vdst, src0),
          build_vop1(kOpMovB32, static_cast<uint8_t>(vdst + 1), *hi)};
}

std::vector<uint32_t> expand_v_mov_b64(const Instruction &inst, uint32_t, uint64_t,
                                       const LivenessAnalysis &, const LaneLayout *,
                                       const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() < static_cast<int>(sizeof(uint32_t)) ||
      inst.size() % static_cast<int>(sizeof(uint32_t)) != 0)
    return {};

  const auto src = std::bit_cast<gfx1250::Vop1MachineInst>(raw[0]);
  const auto literal32 = src.src0 == 255 ? simm32_literal_word(inst, 0) : std::nullopt;
  const auto literal64 = src.src0 == 254 ? simm64_literal_value(inst, 0) : std::nullopt;
  return expand_v_mov_b64_sources(static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src0),
                                  literal32, literal64);
}

std::vector<uint32_t> expand_v_mov_b64_vop3(const Instruction &inst, uint32_t, uint64_t,
                                            const LivenessAnalysis &, const LaneLayout *,
                                            const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3MachineInst))
    return {};

  gfx1250::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.abs != 0 || src.opsel != 0 || src.clamp != 0 || src.omod != 0 || src.neg != 0)
    return {};

  const auto literal32 = src.src0 == 255 ? simm32_literal_word(inst, 0) : std::nullopt;
  const auto literal64 = src.src0 == 254 ? simm64_literal_value(inst, 0) : std::nullopt;
  return expand_v_mov_b64_sources(static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src0),
                                  literal32, literal64);
}

ExpandResult lower_vop2_f64_literal64(const Instruction &inst, uint32_t, uint64_t,
                                      const LivenessAnalysis &liveness, TranslationContext &context,
                                      const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() != 3 * static_cast<int>(sizeof(uint32_t)))
    return ExpandResult::not_handled();

  const auto src = std::bit_cast<gfx1250::Vop2MachineInst>(raw[0]);
  if (src.src0 != 254)
    return ExpandResult::not_handled();
  if (src.vdst > 254 || src.vsrc1 > 254)
    return ExpandResult::failed("A packed F64 VOP2 register pair is out of range");

  const auto literal = simm64_literal_value(inst, 0);
  if (!literal)
    return ExpandResult::failed("Could not decode the gfx1250 F64 VOP2 literal64");

  std::vector<uint8_t> avoid{static_cast<uint8_t>(src.vdst), static_cast<uint8_t>(src.vdst + 1u),
                             static_cast<uint8_t>(src.vsrc1), static_cast<uint8_t>(src.vsrc1 + 1u)};
  const auto literal_pair = find_free_vgpr_run_avoiding(inst, liveness, 2, avoid);
  if (!literal_pair || *literal_pair > 254u) {
    return ExpandResult::failed(
        "No free VGPR pair is available to materialize an F64 VOP2 literal64",
        {"Reduce VGPR pressure or add private spilling for gfx1250 literal64 lowering."});
  }
  context.require_vgprs(static_cast<uint32_t>(*literal_pair) + 2u);

  constexpr uint8_t kOpMovB32 = 1;
  const auto tmp = static_cast<uint8_t>(*literal_pair);
  std::vector<uint32_t> words;
  words.reserve(7);
  append_vop1(words, kOpMovB32, tmp, 255, static_cast<uint32_t>(*literal));
  append_vop1(words, kOpMovB32, static_cast<uint8_t>(tmp + 1u), 255,
              static_cast<uint32_t>(*literal >> 32u));
  append_wait_valu_vgpr(words);
  words.push_back(build_vop2(static_cast<uint8_t>(src.op), static_cast<uint8_t>(src.vdst),
                             vgpr_src(tmp), static_cast<uint8_t>(src.vsrc1)));
  return ExpandResult::success(std::move(words));
}

std::vector<uint32_t> expand_v_cvt_f32_f16_e32_high_src(const Instruction &inst, uint32_t, uint64_t,
                                                        const LivenessAnalysis &liveness,
                                                        const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() != static_cast<int>(sizeof(uint32_t)))
    return {};

  auto src = std::bit_cast<gfx1250::Vop1MachineInst>(raw[0]);
  const auto src_vgpr = vgpr_index(static_cast<uint16_t>(src.src0));
  if (!src_vgpr || *src_vgpr < 128u)
    return {};

  std::vector<uint8_t> avoid;
  add_avoid_vgpr(avoid, static_cast<uint8_t>(src.vdst));
  const auto src_phys = static_cast<uint8_t>(*src_vgpr & 0x7Fu);
  add_avoid_vgpr(avoid, src_phys);

  const auto tmp_opt = find_free_vgpr_run_avoiding(inst, liveness, 1, avoid);
  if (!tmp_opt || *tmp_opt > 255u)
    return {};
  const auto tmp = static_cast<uint8_t>(*tmp_opt);

  constexpr uint16_t kRdna4OpVCvtF32F16Vop3 = 0x18B;
  constexpr uint16_t kVgprSrcBase = 256;
  std::vector<uint32_t> words;
  words.reserve(4);
  if (!append_materialize_b16_half(words, tmp, static_cast<uint16_t>(kVgprSrcBase + src_phys),
                                   /*high_half=*/true, std::nullopt))
    return {};
  append_wait_valu_vgpr(words);
  auto [w0, w1] = build_vop3(kRdna4OpVCvtF32F16Vop3, static_cast<uint8_t>(src.vdst),
                             static_cast<uint16_t>(kVgprSrcBase + tmp), 0);
  words.push_back(w0);
  words.push_back(w1);
  return words;
}

std::vector<uint32_t> expand_v_mov_b16(const Instruction &inst, uint32_t, uint64_t,
                                       const LivenessAnalysis &liveness, const LaneLayout *,
                                       const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(uint32_t))
    return {};

  auto src = std::bit_cast<gfx1250::Vop1MachineInst>(raw[0]);

  if (src.src0 == 254)
    return {};

  std::optional<uint32_t> literal_word;
  if (src.src0 == 255) {
    if (inst.size() != sizeof(gfx1250::Vop1InstLiteralMachineInst))
      return {};
    literal_word = simm32_literal_word(inst, 0);
    if (!literal_word)
      return {};
  } else if (inst.size() != sizeof(uint32_t)) {
    return {};
  }

  const uint8_t dst_phys = static_cast<uint8_t>(src.vdst & 0x7Fu);
  const bool dst_high = (src.vdst & 0x80u) != 0;
  uint16_t lowered_src = static_cast<uint16_t>(src.src0);
  bool src_high = false;

  std::vector<uint8_t> avoid;
  add_avoid_vgpr(avoid, dst_phys);
  if (auto src_vgpr = vgpr_index(static_cast<uint16_t>(src.src0))) {
    const uint8_t src_phys = static_cast<uint8_t>(*src_vgpr & 0x7Fu);
    src_high = (*src_vgpr & 0x80u) != 0;
    lowered_src = static_cast<uint16_t>(256u + src_phys);
    add_avoid_vgpr(avoid, src_phys);
  }

  const auto tmp_opt = find_free_vgpr_run_avoiding(inst, liveness, 1, avoid);
  if (!tmp_opt || *tmp_opt > 255)
    return {};
  const auto tmp = static_cast<uint8_t>(*tmp_opt);

  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpLshrrevB32 = 25;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpMovB32 = 1;
  constexpr uint8_t kOpOrB32 = 28;

  std::vector<uint32_t> words;
  words.reserve(16);
  using P = HazardTracker::Pipeline;
  HazardTracker hz;
  auto emit_vop1 = [&](uint8_t op, uint8_t vdst, uint16_t src0,
                       std::optional<uint32_t> literal = std::nullopt) {
    hz.emit(words, build_vop1(op, vdst, src0), P::VALU);
    if (literal && src0 == 255)
      hz.emit_raw(words, *literal);
  };
  auto emit_vop2 = [&](uint8_t op, uint8_t vdst, uint16_t src0, uint8_t vsrc1,
                       std::optional<uint32_t> literal = std::nullopt) {
    hz.emit(words, build_vop2(op, vdst, src0, vsrc1), P::VALU);
    if (literal && src0 == 255)
      hz.emit_raw(words, *literal);
  };

  if (lowered_src == 255) {
    if (!literal_word)
      return {};
    const uint32_t half = src_high ? ((*literal_word >> 16) & 0xFFFFu) : (*literal_word & 0xFFFFu);
    std::optional<uint32_t> half_literal;
    emit_vop1(kOpMovB32, tmp, literal_or_inline_u32(half, half_literal), half_literal);
  } else if (auto src_vgpr = vgpr_index(lowered_src)) {
    if (src_high)
      emit_vop2(kOpLshrrevB32, tmp, scalar_positive_inline_u32(16), *src_vgpr);
    else
      emit_vop2(kOpOrB32, tmp, scalar_positive_inline_u32(0), *src_vgpr);
  } else {
    emit_vop1(kOpMovB32, tmp, lowered_src);
  }

  emit_vop2(kOpAndB32, tmp, 255, tmp, 0x0000FFFFu);
  if (dst_high) {
    emit_vop2(kOpLshlrevB32, tmp, scalar_positive_inline_u32(16), tmp);
    emit_vop2(kOpAndB32, dst_phys, 255, dst_phys, 0x0000FFFFu);
  } else {
    emit_vop2(kOpAndB32, dst_phys, 255, dst_phys, 0xFFFF0000u);
  }
  emit_vop2(kOpOrB32, dst_phys, static_cast<uint16_t>(256u + tmp), dst_phys);
  hz.emit_raw(words, build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4));
  return words;
}

[[nodiscard]] bool append_vop3_sdst_from_vector64_parts(std::vector<uint32_t> &words, uint16_t op,
                                                        uint8_t vdst, uint8_t sdst,
                                                        const Vector64SourceParts &src0,
                                                        const Vector64SourceParts &src1, bool high,
                                                        uint16_t src2 = 0) {
  const uint16_t encoded_src0 = high ? src0.hi : src0.lo;
  const uint16_t encoded_src1 = high ? src1.hi : src1.lo;
  const auto &src0_literal = high ? src0.hi_literal : src0.lo_literal;
  const auto &src1_literal = high ? src1.hi_literal : src1.lo_literal;

  std::optional<uint32_t> literal;
  if (encoded_src0 == 255) {
    if (!src0_literal)
      return false;
    literal = *src0_literal;
  }
  if (encoded_src1 == 255) {
    if (literal || !src1_literal)
      return false;
    literal = *src1_literal;
  }

  auto [w0, w1] = build_vop3_sdst(op, vdst, sdst, encoded_src0, encoded_src1, src2);
  words.push_back(w0);
  words.push_back(w1);
  if (literal)
    words.push_back(*literal);
  return true;
}

std::vector<uint32_t> expand_v_sub_nc_u64(uint8_t vdst, const Vector64SourceParts &src0,
                                          const Vector64SourceParts &src1, const Instruction &inst,
                                          const LivenessAnalysis &liveness) {
  if (vdst == 255)
    return {};

  if (low_write_clobbers_high_source(vdst, src0.hi) ||
      low_write_clobbers_high_source(vdst, src1.hi))
    return {};

  auto carry_sgpr = liveness.find_free_sgpr_pair(&inst);
  if (!carry_sgpr || *carry_sgpr > 105)
    return {};

  constexpr uint16_t kOpSubCoCiU32 = 289;
  constexpr uint16_t kOpSubCoU32 = 769;
  constexpr uint8_t kSoppWaitAlu = 8;
  const auto carry = static_cast<uint8_t>(*carry_sgpr);

  std::vector<uint32_t> words;
  words.reserve(8);
  if (!append_vop3_sdst_from_vector64_parts(words, kOpSubCoU32, vdst, carry, src0, src1, false))
    return {};
  words.push_back(pack_sopp(kSoppWaitAlu, kWaitAluDepctrVaSdst0));
  if (!append_vop3_sdst_from_vector64_parts(words, kOpSubCoCiU32, static_cast<uint8_t>(vdst + 1),
                                            kNullSgpr, src0, src1, true, carry))
    return {};
  words.push_back(pack_sopp(kSoppWaitAlu, kWaitAluDepctrVaVdst0));
  return words;
}

std::vector<uint32_t> expand_v_add_nc_u64(uint8_t vdst, const Vector64SourceParts &src0,
                                          const Vector64SourceParts &src1, const Instruction &inst,
                                          const LivenessAnalysis &liveness) {
  if (vdst == 255)
    return {};

  // gfx1250's globally-addressable scratch pointer is
  //
  //   SRC_FLAT_SCRATCH_BASE + (lane_id << 52) + private_offset.
  //
  // RDNA4 does not implement that addressing mode. Its equivalent flat
  // private pointer is {private_offset, SRC_PRIVATE_BASE.hi}; reading
  // SRC_PRIVATE_BASE as a 64-bit pair is required because 32-bit reads of the
  // special source do not provide the aperture value. Normalize this common
  // address-construction idiom here instead of carrying gfx1250-only special
  // sources into the target instruction stream.
  const auto is_gfx1250_flat_scratch_base = [](const Vector64SourceParts &src) {
    return src.lo == kGfx1250SrcFlatScratchBaseLo && src.hi == kGfx1250SrcFlatScratchBaseHi &&
           !src.lo_literal && !src.hi_literal;
  };
  const Vector64SourceParts *private_offset = nullptr;
  if (is_gfx1250_flat_scratch_base(src0))
    private_offset = &src1;
  else if (is_gfx1250_flat_scratch_base(src1))
    private_offset = &src0;

  if (private_offset) {
    auto aperture_sgpr = liveness.find_free_sgpr_pair(&inst);
    if (!aperture_sgpr || *aperture_sgpr > 105)
      return {};

    constexpr uint8_t kOpMovB32 = 1;
    std::vector<uint32_t> words;
    words.reserve(5);
    words.push_back(build_s_mov_b64(static_cast<uint8_t>(*aperture_sgpr), kRdna4SrcPrivateBase));
    append_wait_salu_sgpr(words);
    append_vop1(words, kOpMovB32, vdst, private_offset->lo, private_offset->lo_literal);
    append_vop1(words, kOpMovB32, static_cast<uint8_t>(vdst + 1u),
                static_cast<uint16_t>(*aperture_sgpr + 1u));
    return words;
  }

  if (low_write_clobbers_high_source(vdst, src0.hi) ||
      low_write_clobbers_high_source(vdst, src1.hi))
    return {};

  auto carry_sgpr = liveness.find_free_sgpr_pair(&inst);
  if (!carry_sgpr || *carry_sgpr > 105)
    return {};

  constexpr uint16_t kOpAddCoCiU32 = 288;
  constexpr uint16_t kOpAddCoU32 = 768;
  constexpr uint8_t kSoppWaitAlu = 8;
  const auto carry = static_cast<uint8_t>(*carry_sgpr);

  std::vector<uint32_t> words;
  words.reserve(8);
  if (!append_vop3_sdst_from_vector64_parts(words, kOpAddCoU32, vdst, carry, src0, src1, false))
    return {};
  words.push_back(pack_sopp(kSoppWaitAlu, kWaitAluDepctrVaSdst0));
  if (!append_vop3_sdst_from_vector64_parts(words, kOpAddCoCiU32, static_cast<uint8_t>(vdst + 1),
                                            kNullSgpr, src0, src1, true, carry))
    return {};
  words.push_back(pack_sopp(kSoppWaitAlu, kWaitAluDepctrVaVdst0));
  return words;
}

std::vector<uint32_t> expand_v_add_nc_u64_e32(const Instruction &inst, uint32_t, uint64_t,
                                              const LivenessAnalysis &liveness, const LaneLayout *,
                                              const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(uint32_t))
    return {};

  auto src = std::bit_cast<gfx1250::Vop2MachineInst>(raw[0]);
  if (src.vsrc1 == 255)
    return {};

  const int expected_size = src.src0 == 254   ? 3 * static_cast<int>(sizeof(uint32_t))
                            : src.src0 == 255 ? 2 * static_cast<int>(sizeof(uint32_t))
                                              : static_cast<int>(sizeof(uint32_t));
  if (inst.size() != expected_size)
    return {};

  const auto src0 = decode_vector64_source(inst, static_cast<uint16_t>(src.src0), 0);
  const auto src1 = decode_vector64_source(inst, static_cast<uint16_t>(256u + src.vsrc1), 1);
  if (!src0 || !src1)
    return {};
  return expand_v_add_nc_u64(static_cast<uint8_t>(src.vdst), *src0, *src1, inst, liveness);
}

std::vector<uint32_t> expand_v_add_nc_u64_vop3(const Instruction &inst, uint32_t, uint64_t,
                                               const LivenessAnalysis &liveness, const LaneLayout *,
                                               const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3MachineInst))
    return {};

  gfx1250::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  const bool has_literal64 = src.src0 == 254 || src.src1 == 254;
  const bool has_literal32 = src.src0 == 255 || src.src1 == 255;
  if (((src.src0 == 254 || src.src0 == 255) && (src.src1 == 254 || src.src1 == 255)) ||
      inst.size() != static_cast<int>(sizeof(gfx1250::Vop3MachineInst)) + (has_literal64   ? 8
                                                                           : has_literal32 ? 4
                                                                                           : 0))
    return {};

  const auto src0 = decode_vector64_source(inst, static_cast<uint16_t>(src.src0), 0);
  const auto src1 = decode_vector64_source(inst, static_cast<uint16_t>(src.src1), 1);
  if (!src0 || !src1)
    return {};
  return expand_v_add_nc_u64(static_cast<uint8_t>(src.vdst), *src0, *src1, inst, liveness);
}

std::vector<uint32_t> expand_v_sub_nc_u64_e32(const Instruction &inst, uint32_t, uint64_t,
                                              const LivenessAnalysis &liveness, const LaneLayout *,
                                              const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(uint32_t))
    return {};

  auto src = std::bit_cast<gfx1250::Vop2MachineInst>(raw[0]);
  if (src.vsrc1 == 255)
    return {};

  const int expected_size = src.src0 == 254   ? 3 * static_cast<int>(sizeof(uint32_t))
                            : src.src0 == 255 ? 2 * static_cast<int>(sizeof(uint32_t))
                                              : static_cast<int>(sizeof(uint32_t));
  if (inst.size() != expected_size)
    return {};

  const auto src0 = decode_vector64_source(inst, static_cast<uint16_t>(src.src0), 0);
  const auto src1 = decode_vector64_source(inst, static_cast<uint16_t>(256u + src.vsrc1), 1);
  if (!src0 || !src1)
    return {};
  return expand_v_sub_nc_u64(static_cast<uint8_t>(src.vdst), *src0, *src1, inst, liveness);
}

std::vector<uint32_t> expand_v_sub_nc_u64_vop3(const Instruction &inst, uint32_t, uint64_t,
                                               const LivenessAnalysis &liveness, const LaneLayout *,
                                               const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3MachineInst))
    return {};

  gfx1250::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  const bool has_literal64 = src.src0 == 254 || src.src1 == 254;
  const bool has_literal32 = src.src0 == 255 || src.src1 == 255;
  if (((src.src0 == 254 || src.src0 == 255) && (src.src1 == 254 || src.src1 == 255)) ||
      inst.size() != static_cast<int>(sizeof(gfx1250::Vop3MachineInst)) + (has_literal64   ? 8
                                                                           : has_literal32 ? 4
                                                                                           : 0))
    return {};

  const auto src0 = decode_vector64_source(inst, static_cast<uint16_t>(src.src0), 0);
  const auto src1 = decode_vector64_source(inst, static_cast<uint16_t>(src.src1), 1);
  if (!src0 || !src1)
    return {};
  return expand_v_sub_nc_u64(static_cast<uint8_t>(src.vdst), *src0, *src1, inst, liveness);
}

std::vector<uint32_t> expand_v_add_f16_e32(const Instruction &inst, uint32_t, uint64_t,
                                           const LivenessAnalysis &, const LaneLayout *,
                                           const LaneLayout *) {
  constexpr uint16_t kOpVAddF16Vop3 = 306;
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop2MachineInst))
    return {};

  auto src = std::bit_cast<gfx1250::Vop2MachineInst>(raw[0]);
  std::optional<uint32_t> literal_word;
  if (src.src0 == 255u) {
    literal_word = simm32_literal_word(inst, 0);
    if (!literal_word)
      return {};
  }

  uint8_t opsel = 0;
  uint16_t src0 = static_cast<uint16_t>(src.src0);
  if (src0 >= 256u) {
    const uint16_t encoded_vgpr = static_cast<uint16_t>(src0 - 256u);
    if ((encoded_vgpr & 0x80u) != 0)
      opsel |= 0x1u;
    src0 = static_cast<uint16_t>(256u + (encoded_vgpr & 0x7Fu));
  }
  if ((src.vsrc1 & 0x80u) != 0)
    opsel |= 0x2u;
  if ((src.vdst & 0x80u) != 0)
    opsel |= 0x8u;

  std::vector<uint32_t> words;
  const auto [w0, w1] =
      build_vop3_mod(kOpVAddF16Vop3, static_cast<uint8_t>(src.vdst & 0x7Fu), src0,
                     static_cast<uint16_t>(256u + (src.vsrc1 & 0x7Fu)), 0, 0, opsel);
  words.push_back(w0);
  words.push_back(w1);
  if (literal_word)
    words.push_back(*literal_word);
  return words;
}

std::vector<uint32_t> expand_v_mad_u32_vop3(const Instruction &inst, uint32_t, uint64_t,
                                            const LivenessAnalysis &liveness, const LaneLayout *,
                                            const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3MachineInst))
    return {};

  gfx1250::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.vdst == 255 || src.abs != 0 || src.opsel != 0 || src.clamp != 0 || src.omod != 0 ||
      src.neg != 0 || src.src0 == 254 || src.src1 == 254 || src.src2 == 254)
    return {};

  std::optional<uint32_t> literal_word;
  if (src.src0 == 255 || src.src1 == 255 || src.src2 == 255) {
    const uint8_t literal_operand = src.src0 == 255 ? 0 : src.src1 == 255 ? 1 : 2;
    literal_word = simm32_literal_word(inst, literal_operand);
    if (!literal_word)
      return {};
  }

  std::vector<uint8_t> avoid{static_cast<uint8_t>(src.vdst)};
  if (auto src0_vgpr = vgpr_index(static_cast<uint16_t>(src.src0)))
    avoid.push_back(*src0_vgpr);
  if (auto src1_vgpr = vgpr_index(static_cast<uint16_t>(src.src1)))
    avoid.push_back(*src1_vgpr);
  if (auto src2_vgpr = vgpr_index(static_cast<uint16_t>(src.src2)))
    avoid.push_back(*src2_vgpr);

  const auto tmp_opt = find_free_vgpr_run_avoiding(inst, liveness, 1, avoid);
  if (!tmp_opt)
    return {};
  const uint8_t tmp = static_cast<uint8_t>(*tmp_opt);

  constexpr uint16_t kOpMulLoU32 = 812;
  constexpr uint16_t kOpAddNcU32 = 293;
  constexpr uint16_t kTmpSrcBase = 256;

  std::vector<uint32_t> words;
  words.reserve(literal_word ? 6 : 5);
  append_vop3(words, kOpMulLoU32, tmp, static_cast<uint16_t>(src.src0),
              static_cast<uint16_t>(src.src1), 0, literal_word);
  words.push_back(build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4));
  append_vop3(words, kOpAddNcU32, static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src2),
              static_cast<uint16_t>(kTmpSrcBase + tmp), 0, literal_word);
  return words;
}

std::vector<uint32_t> expand_v_mad_nc_64_32_vop3(const Instruction &inst, uint32_t, uint64_t,
                                                 const LivenessAnalysis &liveness,
                                                 const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3MachineInst))
    return {};

  gfx1250::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.vdst > 254 || src.abs != 0 || src.opsel != 0 || src.clamp != 0 || src.omod != 0 ||
      src.neg != 0 || src.src0 == 254 || src.src1 == 254 || src.src2 == 254 || src.src2 == 255)
    return {};

  uint16_t src0 = static_cast<uint16_t>(src.src0);
  uint16_t src1 = static_cast<uint16_t>(src.src1);
  std::vector<uint32_t> words;
  if (src.src0 == 255 || src.src1 == 255) {
    if (inst.size() != sizeof(gfx1250::Vop3InstLiteralMachineInst))
      return {};
    const auto literal_word = simm32_literal_word(inst, src.src0 == 255 ? 0 : 1);
    if (!literal_word)
      return {};

    const auto literal_sgpr = liveness.find_free_sgpr(&inst);
    if (!literal_sgpr || *literal_sgpr > 105)
      return {};
    const auto tmp = static_cast<uint8_t>(*literal_sgpr);
    words.reserve(5);
    append_s_mov_b32_lit(words, tmp, *literal_word);
    append_wait_salu_sgpr(words);
    if (src.src0 == 255)
      src0 = tmp;
    if (src.src1 == 255)
      src1 = tmp;
  } else if (inst.size() != sizeof(gfx1250::Vop3MachineInst)) {
    return {};
  } else {
    words.reserve(2);
  }

  constexpr uint16_t kOpMadNcU64U32 = 762;
  constexpr uint16_t kOpMadNcI64I32 = 763;
  constexpr uint16_t kOpMadCoU64U32 = 766;
  constexpr uint16_t kOpMadCoI64I32 = 767;
  if (src.op != kOpMadNcU64U32 && src.op != kOpMadNcI64I32)
    return {};
  const uint16_t target_op = src.op == kOpMadNcI64I32 ? kOpMadCoI64I32 : kOpMadCoU64U32;
  auto [w0, w1] = build_vop3_sdst(target_op, static_cast<uint8_t>(src.vdst), kNullSgpr, src0, src1,
                                  static_cast<uint16_t>(src.src2));
  words.push_back(w0);
  words.push_back(w1);
  return words;
}

std::vector<uint32_t> expand_v_mul_u64_e32(const Instruction &inst, uint32_t, uint64_t,
                                           const LivenessAnalysis &liveness, const LaneLayout *,
                                           const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(uint32_t))
    return {};

  auto src = std::bit_cast<gfx1250::Vop2MachineInst>(raw[0]);
  if (src.vsrc1 == 255)
    return {};

  const int expected_size = src.src0 == 254   ? 3 * static_cast<int>(sizeof(uint32_t))
                            : src.src0 == 255 ? 2 * static_cast<int>(sizeof(uint32_t))
                                              : static_cast<int>(sizeof(uint32_t));
  if (inst.size() != expected_size)
    return {};

  const auto src0 = decode_vector64_source(inst, static_cast<uint16_t>(src.src0), 0);
  const auto src1 = decode_vector64_source(inst, static_cast<uint16_t>(256u + src.vsrc1), 1);
  if (!src0 || !src1)
    return {};
  return expand_v_mul_u64(static_cast<uint8_t>(src.vdst), *src0, *src1, inst, liveness);
}

std::vector<uint32_t> expand_v_mul_u64_vop3(const Instruction &inst, uint32_t, uint64_t,
                                            const LivenessAnalysis &liveness, const LaneLayout *,
                                            const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3MachineInst))
    return {};

  gfx1250::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  const bool has_literal64 = src.src0 == 254 || src.src1 == 254;
  const bool has_literal32 = src.src0 == 255 || src.src1 == 255;
  if (src.vdst > 254 || src.abs != 0 || src.opsel != 0 || src.clamp != 0 || src.omod != 0 ||
      src.neg != 0 ||
      ((src.src0 == 254 || src.src0 == 255) && (src.src1 == 254 || src.src1 == 255)) ||
      inst.size() != static_cast<int>(sizeof(gfx1250::Vop3MachineInst)) + (has_literal64   ? 8
                                                                           : has_literal32 ? 4
                                                                                           : 0))
    return {};

  const auto src0 = decode_vector64_source(inst, static_cast<uint16_t>(src.src0), 0);
  const auto src1 = decode_vector64_source(inst, static_cast<uint16_t>(src.src1), 1);
  if (!src0 || !src1)
    return {};
  return expand_v_mul_u64(static_cast<uint8_t>(src.vdst), *src0, *src1, inst, liveness);
}

std::vector<uint32_t> expand_v_lshl_add_u64_vop3(const Instruction &inst, uint32_t, uint64_t,
                                                 const LivenessAnalysis &liveness,
                                                 const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() != sizeof(gfx1250::Vop3MachineInst))
    return {};

  gfx1250::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.vdst == 255)
    return {};

  constexpr uint16_t kInlineZero = scalar_positive_inline_u32(0);
  constexpr uint16_t kMaxLowerableShift = 31;
  if (src.src1 <= kInlineZero || src.src1 > scalar_positive_inline_u32(kMaxLowerableShift))
    return {};
  const uint16_t shift = static_cast<uint16_t>(src.src1 - kInlineZero);

  const auto src0_hi = pair_hi_src(static_cast<uint16_t>(src.src0));
  const auto src2_hi = pair_hi_src(static_cast<uint16_t>(src.src2));
  if (!src0_hi || !src2_hi)
    return {};

  auto carry_sgpr = liveness.find_free_sgpr_pair(&inst);
  if (!carry_sgpr || *carry_sgpr > 105)
    return {};
  const auto carry = static_cast<uint8_t>(*carry_sgpr);

  const uint16_t vdst_lo_src = static_cast<uint16_t>(256u + src.vdst);
  const uint16_t vdst_hi_src = static_cast<uint16_t>(256u + src.vdst + 1u);
  const auto src0_lo_vgpr = vgpr_index(static_cast<uint16_t>(src.src0));
  const auto src2_lo_vgpr = vgpr_index(static_cast<uint16_t>(src.src2));
  const auto src2_hi_vgpr = vgpr_index(*src2_hi);
  const bool can_shift_into_dst = !(src0_lo_vgpr && *src0_lo_vgpr == src.vdst + 1u) &&
                                  !(src2_lo_vgpr && overlaps_vdst_pair(*src2_lo_vgpr, src.vdst)) &&
                                  !(src2_hi_vgpr && overlaps_vdst_pair(*src2_hi_vgpr, src.vdst));

  uint16_t shifted_lo = vdst_lo_src;
  uint16_t shifted_hi = vdst_hi_src;
  if (!can_shift_into_dst) {
    const auto tmp_base = find_free_vgpr_run_away_from_dst(inst, liveness, 2, src.vdst);
    if (!tmp_base)
      return {};
    shifted_lo = static_cast<uint16_t>(256u + *tmp_base);
    shifted_hi = static_cast<uint16_t>(256u + *tmp_base + 1u);
  }

  constexpr uint16_t kOpAlignbitB32 = 534;
  constexpr uint16_t kOpLshlrevB32 = 280;
  constexpr uint16_t kOpAddCoCiU32 = 288;
  constexpr uint16_t kOpAddCoU32 = 768;
  constexpr uint8_t kSoppWaitAlu = 8;

  std::vector<uint32_t> words;
  words.reserve(9);
  {
    auto [w0, w1] =
        build_vop3(kOpAlignbitB32, static_cast<uint8_t>(shifted_hi - 256u), *src0_hi,
                   static_cast<uint16_t>(src.src0), scalar_positive_inline_u32(32 - shift));
    words.push_back(w0);
    words.push_back(w1);
  }
  {
    auto [w0, w1] = build_vop3(kOpLshlrevB32, static_cast<uint8_t>(shifted_lo - 256u),
                               scalar_positive_inline_u32(shift), static_cast<uint16_t>(src.src0));
    words.push_back(w0);
    words.push_back(w1);
  }
  words.push_back(build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4));
  {
    auto [w0, w1] = build_vop3_sdst(kOpAddCoU32, static_cast<uint8_t>(src.vdst), carry, shifted_lo,
                                    static_cast<uint16_t>(src.src2));
    words.push_back(w0);
    words.push_back(w1);
  }
  words.push_back(pack_sopp(kSoppWaitAlu, kWaitAluDepctrVaSdst0));
  {
    auto [w0, w1] = build_vop3_sdst(kOpAddCoCiU32, static_cast<uint8_t>(src.vdst + 1u), kNullSgpr,
                                    shifted_hi, *src2_hi, carry);
    words.push_back(w0);
    words.push_back(w1);
  }
  return words;
}

std::vector<uint32_t> expand_v_lshl_add_u32_vop3(const Instruction &inst, uint32_t, uint64_t,
                                                 const LivenessAnalysis &liveness,
                                                 const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3MachineInst))
    return {};

  gfx1250::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.vdst == 255 || src.abs != 0 || src.opsel != 0 || src.clamp != 0 || src.omod != 0 ||
      src.neg != 0 || src.src0 == 254 || src.src1 == 254 || src.src2 == 254)
    return {};

  std::optional<uint32_t> shift_literal;
  if (src.src0 == 255 || src.src1 == 255) {
    shift_literal = (src.src0 == 255) ? simm32_literal_word(inst, 0) : simm32_literal_word(inst, 1);
    if (!shift_literal)
      return {};
  }

  std::optional<uint32_t> add_literal;
  if (src.src2 == 255) {
    add_literal = simm32_literal_word(inst, 2);
    if (!add_literal)
      return {};
  }

  uint8_t shift_dst = static_cast<uint8_t>(src.vdst);
  const auto src2_vgpr = vgpr_index(static_cast<uint16_t>(src.src2));
  if (src2_vgpr && *src2_vgpr == src.vdst) {
    std::vector<uint8_t> avoid{static_cast<uint8_t>(src.vdst)};
    if (auto src0_vgpr = vgpr_index(static_cast<uint16_t>(src.src0)))
      avoid.push_back(*src0_vgpr);
    if (auto src1_vgpr = vgpr_index(static_cast<uint16_t>(src.src1)))
      avoid.push_back(*src1_vgpr);
    avoid.push_back(*src2_vgpr);

    const auto tmp_opt = find_free_vgpr_run_avoiding(inst, liveness, 1, avoid);
    if (!tmp_opt)
      return {};
    shift_dst = static_cast<uint8_t>(*tmp_opt);
  }

  constexpr uint16_t kOpAddNcU32 = 293;
  constexpr uint16_t kOpLshlrevB32 = 280;
  constexpr uint16_t kTmpSrcBase = 256;

  std::vector<uint32_t> words;
  words.reserve((shift_literal ? 1u : 0u) + (add_literal ? 1u : 0u) + 5u);
  append_vop3(words, kOpLshlrevB32, shift_dst, static_cast<uint16_t>(src.src1),
              static_cast<uint16_t>(src.src0), 0, shift_literal);
  words.push_back(build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4));
  append_vop3(words, kOpAddNcU32, static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src2),
              static_cast<uint16_t>(kTmpSrcBase + shift_dst), 0, add_literal);
  return words;
}

std::vector<uint32_t> expand_v_lshl_or_b32_vop3(const Instruction &inst, uint32_t, uint64_t,
                                                const LivenessAnalysis &liveness,
                                                const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3MachineInst))
    return {};

  gfx1250::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.vdst == 255 || src.abs != 0 || src.opsel != 0 || src.clamp != 0 || src.omod != 0 ||
      src.neg != 0 || src.src0 == 254 || src.src1 == 254 || src.src2 == 254)
    return {};

  std::optional<uint32_t> shift_literal;
  if (src.src0 == 255 || src.src1 == 255) {
    shift_literal = (src.src0 == 255) ? simm32_literal_word(inst, 0) : simm32_literal_word(inst, 1);
    if (!shift_literal)
      return {};
  }

  std::optional<uint32_t> or_literal;
  if (src.src2 == 255) {
    or_literal = simm32_literal_word(inst, 2);
    if (!or_literal)
      return {};
  }

  uint8_t shift_dst = static_cast<uint8_t>(src.vdst);
  const auto src2_vgpr = vgpr_index(static_cast<uint16_t>(src.src2));
  if (src2_vgpr && *src2_vgpr == src.vdst) {
    std::vector<uint8_t> avoid{static_cast<uint8_t>(src.vdst)};
    if (auto src0_vgpr = vgpr_index(static_cast<uint16_t>(src.src0)))
      avoid.push_back(*src0_vgpr);
    if (auto src1_vgpr = vgpr_index(static_cast<uint16_t>(src.src1)))
      avoid.push_back(*src1_vgpr);
    avoid.push_back(*src2_vgpr);

    const auto tmp_opt = find_free_vgpr_run_avoiding(inst, liveness, 1, avoid);
    if (!tmp_opt)
      return {};
    shift_dst = static_cast<uint8_t>(*tmp_opt);
  }

  constexpr uint16_t kOpLshlrevB32 = 280;
  constexpr uint8_t kOpOrB32 = 28;

  std::vector<uint32_t> words;
  words.reserve((shift_literal ? 1u : 0u) + (or_literal ? 1u : 0u) + 4u);
  append_vop3(words, kOpLshlrevB32, shift_dst, static_cast<uint16_t>(src.src1),
              static_cast<uint16_t>(src.src0), 0, shift_literal);
  words.push_back(build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4));
  append_vop2(words, kOpOrB32, static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src2),
              shift_dst, or_literal);
  return words;
}

std::vector<uint32_t> expand_v_lshlrev_b64_e32(const Instruction &inst, uint32_t, uint64_t,
                                               const LivenessAnalysis &, const LaneLayout *,
                                               const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop2MachineInst))
    return {};

  const auto src = std::bit_cast<gfx1250::Vop2MachineInst>(raw[0]);
  if (src.vdst > 254 || src.vsrc1 > 254 || src.src0 == 254 || src.src0 == 255)
    return {};
  if (src.src0 < scalar_positive_inline_u32(1) || src.src0 > scalar_positive_inline_u32(31))
    return {};

  const auto shift = static_cast<uint16_t>(src.src0 - scalar_positive_inline_u32(0));
  const uint16_t src_lo = static_cast<uint16_t>(256u + src.vsrc1);
  const uint16_t src_hi = static_cast<uint16_t>(src_lo + 1u);
  constexpr uint16_t kOpAlignbitB32 = 534;
  constexpr uint16_t kOpLshlrevB32 = 280;

  auto append_hi = [&](std::vector<uint32_t> &words) {
    append_vop3(words, kOpAlignbitB32, static_cast<uint8_t>(src.vdst + 1u), src_hi, src_lo,
                scalar_positive_inline_u32(static_cast<uint16_t>(32u - shift)));
  };
  auto append_lo = [&](std::vector<uint32_t> &words) {
    append_vop3(words, kOpLshlrevB32, static_cast<uint8_t>(src.vdst),
                scalar_positive_inline_u32(shift), src_lo);
  };

  std::vector<uint32_t> words;
  words.reserve(4);
  if (src.vdst + 1u == src.vsrc1) {
    append_lo(words);
    append_hi(words);
  } else {
    append_hi(words);
    append_lo(words);
  }
  return words;
}

std::vector<uint32_t> lower_gfx1250_vop3_single_src_to_rdna4(const Instruction &inst, uint32_t,
                                                             uint64_t, const LivenessAnalysis &,
                                                             const LaneLayout *,
                                                             const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() < 2 * static_cast<int>(sizeof(uint32_t)) ||
      (inst.size() % static_cast<int>(sizeof(uint32_t))) != 0)
    return {};

  if ((raw[0] >> 26) != kVop3Encoding || inst.opcode() < 640 || inst.opcode() > 649)
    return {};

  const size_t word_count = static_cast<size_t>(inst.size()) / sizeof(uint32_t);
  std::vector<uint32_t> words(raw, raw + word_count);
  rdna4::Vop3MachineInst dst{};
  std::memcpy(&dst, words.data(), sizeof(dst));
  dst.src1 = 0;
  dst.src2 = 0;
  dst.abs &= 0x1u;
  dst.opsel &= 0x9u;
  dst.neg &= 0x1u;
  std::memcpy(words.data(), &dst, sizeof(dst));
  return words;
}

[[nodiscard]] std::optional<uint16_t> minmax_hi_src(uint16_t src, std::optional<int32_t> literal) {
  if (src == 255) {
    if (!literal || *literal < 0)
      return std::nullopt;
    return scalar_positive_inline_u32(0);
  }
  if (src == 254)
    return std::nullopt;
  return pair_hi_src_sign_extended_inline(src);
}

[[nodiscard]] std::optional<uint32_t> simm32_literal_word(const Instruction &inst,
                                                          uint8_t operand_index) {
  const Operand *operand = inst.src_operand(operand_index);
  if (!operand)
    return std::nullopt;
  return static_cast<uint32_t>(operand->encoding_value());
}

[[nodiscard]] std::optional<uint64_t> simm64_literal_value(const Instruction &inst,
                                                           uint8_t operand_index) {
  const Operand *operand = inst.src_operand(operand_index);
  if (!operand)
    return std::nullopt;
  return operand->literal64_value();
}

std::vector<uint32_t> expand_v_minmax_64_vop3(const Instruction &inst,
                                              const LivenessAnalysis &liveness, uint16_t cmp_op) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3MachineInst))
    return {};

  gfx1250::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.vdst == 255 || src.abs != 0 || src.opsel != 0 || src.clamp != 0 || src.omod != 0 ||
      src.neg != 0)
    return {};

  std::optional<uint32_t> literal_word;
  std::optional<int32_t> literal_s32;
  if (src.src0 == 255 || src.src1 == 255) {
    literal_word = (src.src0 == 255) ? simm32_literal_word(inst, 0) : simm32_literal_word(inst, 1);
    if (!literal_word)
      return {};
    literal_s32 = static_cast<int32_t>(*literal_word);
  }

  const auto src0_hi = minmax_hi_src(static_cast<uint16_t>(src.src0), literal_s32);
  const auto src1_hi = minmax_hi_src(static_cast<uint16_t>(src.src1), literal_s32);
  if (!src0_hi || !src1_hi)
    return {};

  auto pred_sgpr = liveness.find_free_sgpr_pair(&inst);
  if (!pred_sgpr || *pred_sgpr > 105)
    return {};
  const auto pred = static_cast<uint8_t>(*pred_sgpr);

  std::vector<uint8_t> avoid_vgprs = {static_cast<uint8_t>(src.vdst),
                                      static_cast<uint8_t>(src.vdst + 1u)};
  auto avoid_vgpr_src = [&avoid_vgprs](uint16_t source) {
    if (auto vgpr = vgpr_index(source))
      avoid_vgprs.push_back(*vgpr);
  };
  avoid_vgpr_src(static_cast<uint16_t>(src.src0));
  avoid_vgpr_src(static_cast<uint16_t>(src.src1));
  avoid_vgpr_src(*src0_hi);
  avoid_vgpr_src(*src1_hi);

  constexpr uint16_t kOpCndmaskB32 = 257;
  constexpr uint8_t kOpMovB32 = 1;
  constexpr uint8_t kSoppWaitAlu = 8;
  const bool has_scalar_source = scalar_register_src(static_cast<uint16_t>(src.src0)) ||
                                 scalar_register_src(static_cast<uint16_t>(src.src1));

  std::vector<uint32_t> words;
  words.reserve((literal_word ? 10 : 8) + (has_scalar_source ? 5u : 0u));

  auto lower_pair = [&](uint16_t lo, uint16_t hi,
                        bool use_destination) -> std::optional<std::pair<uint16_t, uint16_t>> {
    if (!scalar_register_src(lo))
      return std::pair<uint16_t, uint16_t>{lo, hi};

    if (use_destination) {
      const auto dst = static_cast<uint8_t>(src.vdst);
      append_vop1(words, kOpMovB32, dst, lo);
      append_vop1(words, kOpMovB32, static_cast<uint8_t>(dst + 1u), hi);
      return std::pair<uint16_t, uint16_t>{static_cast<uint16_t>(256u + dst),
                                           static_cast<uint16_t>(256u + dst + 1u)};
    }

    const auto tmp_opt = find_free_vgpr_run_avoiding(inst, liveness, 2, avoid_vgprs);
    if (!tmp_opt)
      return std::nullopt;
    const auto tmp = static_cast<uint8_t>(*tmp_opt);
    avoid_vgprs.push_back(tmp);
    avoid_vgprs.push_back(static_cast<uint8_t>(tmp + 1u));
    append_vop1(words, kOpMovB32, tmp, lo);
    append_vop1(words, kOpMovB32, static_cast<uint8_t>(tmp + 1u), hi);
    return std::pair<uint16_t, uint16_t>{static_cast<uint16_t>(256u + tmp),
                                         static_cast<uint16_t>(256u + tmp + 1u)};
  };

  auto pair_overlaps_destination = [&](uint16_t lo, uint16_t hi) {
    for (uint16_t source : {lo, hi}) {
      if (const auto vgpr = vgpr_index(source);
          vgpr && (*vgpr == src.vdst || *vgpr == src.vdst + 1u))
        return true;
    }
    return false;
  };
  const bool src0_scalar = scalar_register_src(static_cast<uint16_t>(src.src0));
  const bool src1_scalar = scalar_register_src(static_cast<uint16_t>(src.src1));
  const bool stage_src0_in_dst =
      src0_scalar && !pair_overlaps_destination(static_cast<uint16_t>(src.src1), *src1_hi);
  const bool stage_src1_in_dst =
      !stage_src0_in_dst && src1_scalar &&
      !pair_overlaps_destination(static_cast<uint16_t>(src.src0), *src0_hi);
  const auto lowered_src0 =
      lower_pair(static_cast<uint16_t>(src.src0), *src0_hi, stage_src0_in_dst);
  const auto lowered_src1 =
      lower_pair(static_cast<uint16_t>(src.src1), *src1_hi, stage_src1_in_dst);
  if (!lowered_src0 || !lowered_src1)
    return {};

  if (has_scalar_source)
    words.push_back(pack_sopp(kSoppWaitAlu, kWaitAluDepctrVaVdst0));

  append_vop3(words, cmp_op, pred, lowered_src0->first, lowered_src1->first, 0, literal_word);
  words.push_back(pack_sopp(kSoppWaitAlu, kWaitAluDepctrVaSdst0));
  append_vop3(words, kOpCndmaskB32, static_cast<uint8_t>(src.vdst), lowered_src1->first,
              lowered_src0->first, pred, literal_word);
  append_vop3(words, kOpCndmaskB32, static_cast<uint8_t>(src.vdst + 1u), lowered_src1->second,
              lowered_src0->second, pred);
  words.push_back(pack_sopp(kSoppWaitAlu, kWaitAluDepctrVaVdst0));
  return words;
}

std::vector<uint32_t> expand_v_max_u64_vop3(const Instruction &inst, uint32_t, uint64_t,
                                            const LivenessAnalysis &liveness, const LaneLayout *,
                                            const LaneLayout *) {
  constexpr uint16_t kOpCmpGtU64 = 92;
  return expand_v_minmax_64_vop3(inst, liveness, kOpCmpGtU64);
}

std::vector<uint32_t> expand_v_min_u64_vop3(const Instruction &inst, uint32_t, uint64_t,
                                            const LivenessAnalysis &liveness, const LaneLayout *,
                                            const LaneLayout *) {
  constexpr uint16_t kOpCmpLtU64 = 89;
  return expand_v_minmax_64_vop3(inst, liveness, kOpCmpLtU64);
}

std::vector<uint32_t> expand_v_min_i64_vop3(const Instruction &inst, uint32_t, uint64_t,
                                            const LivenessAnalysis &liveness, const LaneLayout *,
                                            const LaneLayout *) {
  constexpr uint16_t kOpCmpLtI64 = 81;
  return expand_v_minmax_64_vop3(inst, liveness, kOpCmpLtI64);
}

std::vector<uint32_t> expand_v_max_i64_vop3(const Instruction &inst, uint32_t, uint64_t,
                                            const LivenessAnalysis &liveness, const LaneLayout *,
                                            const LaneLayout *) {
  constexpr uint16_t kOpCmpGtI64 = 84;
  return expand_v_minmax_64_vop3(inst, liveness, kOpCmpGtI64);
}

[[nodiscard]] std::optional<uint16_t> pk_f32_lane_src(uint16_t src, bool select_high) {
  if (src == 254 || src == 255 || src >= 512)
    return std::nullopt;
  if (select_high && src >= 256) {
    if (src == 511)
      return std::nullopt;
    return static_cast<uint16_t>(src + 1u);
  }
  return src;
}

[[nodiscard]] bool
stage_pk_high_lane_vdst_low_source(const Instruction &inst, const LivenessAnalysis &liveness,
                                   uint8_t vdst, std::span<const uint16_t> all_sources,
                                   std::span<uint16_t> high_sources, std::vector<uint32_t> &words) {
  const uint16_t old_vdst_low = static_cast<uint16_t>(256u + vdst);
  const bool needs_stage = std::ranges::any_of(
      high_sources, [old_vdst_low](uint16_t src) { return src == old_vdst_low; });
  if (!needs_stage)
    return true;

  std::vector<uint8_t> avoid;
  add_avoid_vgpr(avoid, vdst);
  add_avoid_vgpr(avoid, static_cast<uint8_t>(vdst + 1u));
  for (const uint16_t src : all_sources)
    add_avoid_src_vgpr(avoid, src);

  const auto tmp_opt = find_free_vgpr_run_avoiding(inst, liveness, 1, avoid);
  if (!tmp_opt)
    return false;

  constexpr uint8_t kOpMovB32 = 1;
  const uint8_t tmp = static_cast<uint8_t>(*tmp_opt);
  append_vop1(words, kOpMovB32, tmp, old_vdst_low);
  words.push_back(build_s_wait_alu(kWaitAluDepctrVaVdst0, ROCJITSU_CODE_ARCH_RDNA4));
  for (uint16_t &src : high_sources) {
    if (src == old_vdst_low)
      src = static_cast<uint16_t>(256u + tmp);
  }
  return true;
}

std::vector<uint32_t> expand_v_pk_add_f32_vop3p(const Instruction &inst, uint32_t, uint64_t,
                                                const LivenessAnalysis &liveness,
                                                const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() != sizeof(gfx1250::Vop3pMachineInst))
    return {};

  gfx1250::Vop3pMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.vdst == 255)
    return {};

  const auto lo_src0 = pk_f32_lane_src(static_cast<uint16_t>(src.src0), (src.opsel & 0x1u) != 0);
  const auto lo_src1 = pk_f32_lane_src(static_cast<uint16_t>(src.src1), (src.opsel & 0x2u) != 0);
  const auto hi_src0 = pk_f32_lane_src(static_cast<uint16_t>(src.src0), (src.opsel_hi & 0x1u) != 0);
  const auto hi_src1 = pk_f32_lane_src(static_cast<uint16_t>(src.src1), (src.opsel_hi & 0x2u) != 0);
  if (!lo_src0 || !lo_src1 || !hi_src0 || !hi_src1)
    return {};

  constexpr uint16_t kOpAddF32 = 259;
  std::array<uint16_t, 2> lo_srcs{*lo_src0, *lo_src1};
  std::array<uint16_t, 2> hi_srcs{*hi_src0, *hi_src1};
  std::array<uint16_t, 4> all_srcs{lo_srcs[0], lo_srcs[1], hi_srcs[0], hi_srcs[1]};
  std::vector<uint32_t> words;
  words.reserve(6);
  if (!stage_pk_high_lane_vdst_low_source(inst, liveness, static_cast<uint8_t>(src.vdst), all_srcs,
                                          hi_srcs, words))
    return {};
  {
    auto [w0, w1] =
        build_vop3_mod(kOpAddF32, static_cast<uint8_t>(src.vdst), lo_srcs[0], lo_srcs[1], 0, 0, 0,
                       src.clamp != 0, 0, static_cast<uint8_t>(src.neg & 0x3u));
    words.push_back(w0);
    words.push_back(w1);
  }
  {
    auto [w0, w1] =
        build_vop3_mod(kOpAddF32, static_cast<uint8_t>(src.vdst + 1u), hi_srcs[0], hi_srcs[1], 0, 0,
                       0, src.clamp != 0, 0, static_cast<uint8_t>(src.neg_hi & 0x3u));
    words.push_back(w0);
    words.push_back(w1);
  }
  return words;
}

std::vector<uint32_t> expand_v_pk_mul_f32_vop3p(const Instruction &inst, uint32_t, uint64_t,
                                                const LivenessAnalysis &liveness,
                                                const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() != sizeof(gfx1250::Vop3pMachineInst))
    return {};

  gfx1250::Vop3pMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.vdst == 255)
    return {};

  const auto lo_src0 = pk_f32_lane_src(static_cast<uint16_t>(src.src0), (src.opsel & 0x1u) != 0);
  const auto lo_src1 = pk_f32_lane_src(static_cast<uint16_t>(src.src1), (src.opsel & 0x2u) != 0);
  const auto hi_src0 = pk_f32_lane_src(static_cast<uint16_t>(src.src0), (src.opsel_hi & 0x1u) != 0);
  const auto hi_src1 = pk_f32_lane_src(static_cast<uint16_t>(src.src1), (src.opsel_hi & 0x2u) != 0);
  if (!lo_src0 || !lo_src1 || !hi_src0 || !hi_src1)
    return {};

  constexpr uint16_t kOpMulF32 = 264;
  std::array<uint16_t, 2> lo_srcs{*lo_src0, *lo_src1};
  std::array<uint16_t, 2> hi_srcs{*hi_src0, *hi_src1};
  std::array<uint16_t, 4> all_srcs{lo_srcs[0], lo_srcs[1], hi_srcs[0], hi_srcs[1]};
  std::vector<uint32_t> words;
  words.reserve(6);
  if (!stage_pk_high_lane_vdst_low_source(inst, liveness, static_cast<uint8_t>(src.vdst), all_srcs,
                                          hi_srcs, words))
    return {};
  {
    auto [w0, w1] =
        build_vop3_mod(kOpMulF32, static_cast<uint8_t>(src.vdst), lo_srcs[0], lo_srcs[1], 0, 0, 0,
                       src.clamp != 0, 0, static_cast<uint8_t>(src.neg & 0x3u));
    words.push_back(w0);
    words.push_back(w1);
  }
  {
    auto [w0, w1] =
        build_vop3_mod(kOpMulF32, static_cast<uint8_t>(src.vdst + 1u), hi_srcs[0], hi_srcs[1], 0, 0,
                       0, src.clamp != 0, 0, static_cast<uint8_t>(src.neg_hi & 0x3u));
    words.push_back(w0);
    words.push_back(w1);
  }
  return words;
}

std::vector<uint32_t> expand_v_pk_fma_f32_vop3p(const Instruction &inst, uint32_t, uint64_t,
                                                const LivenessAnalysis &liveness,
                                                const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() != sizeof(gfx1250::Vop3pMachineInst))
    return {};

  gfx1250::Vop3pMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.vdst == 255)
    return {};

  const auto lo_src0 = pk_f32_lane_src(static_cast<uint16_t>(src.src0), (src.opsel & 0x1u) != 0);
  const auto lo_src1 = pk_f32_lane_src(static_cast<uint16_t>(src.src1), (src.opsel & 0x2u) != 0);
  const auto lo_src2 = pk_f32_lane_src(static_cast<uint16_t>(src.src2), (src.opsel & 0x4u) != 0);
  const auto hi_src0 = pk_f32_lane_src(static_cast<uint16_t>(src.src0), (src.opsel_hi & 0x1u) != 0);
  const auto hi_src1 = pk_f32_lane_src(static_cast<uint16_t>(src.src1), (src.opsel_hi & 0x2u) != 0);
  const auto hi_src2 = pk_f32_lane_src(static_cast<uint16_t>(src.src2), src.pad_14 != 0);
  if (!lo_src0 || !lo_src1 || !lo_src2 || !hi_src0 || !hi_src1 || !hi_src2)
    return {};

  constexpr uint16_t kOpFmaF32 = 531;
  std::array<uint16_t, 3> lo_srcs{*lo_src0, *lo_src1, *lo_src2};
  std::array<uint16_t, 3> hi_srcs{*hi_src0, *hi_src1, *hi_src2};
  std::array<uint16_t, 6> all_srcs{lo_srcs[0], lo_srcs[1], lo_srcs[2],
                                   hi_srcs[0], hi_srcs[1], hi_srcs[2]};
  std::vector<uint32_t> words;
  words.reserve(6);
  if (!stage_pk_high_lane_vdst_low_source(inst, liveness, static_cast<uint8_t>(src.vdst), all_srcs,
                                          hi_srcs, words))
    return {};
  {
    auto [w0, w1] =
        build_vop3_mod(kOpFmaF32, static_cast<uint8_t>(src.vdst), lo_srcs[0], lo_srcs[1],
                       lo_srcs[2], 0, 0, src.clamp != 0, 0, static_cast<uint8_t>(src.neg & 0x7u));
    words.push_back(w0);
    words.push_back(w1);
  }
  {
    auto [w0, w1] = build_vop3_mod(kOpFmaF32, static_cast<uint8_t>(src.vdst + 1u), hi_srcs[0],
                                   hi_srcs[1], hi_srcs[2], 0, 0, src.clamp != 0, 0,
                                   static_cast<uint8_t>(src.neg_hi & 0x7u));
    words.push_back(w0);
    words.push_back(w1);
  }
  return words;
}

[[nodiscard]] constexpr std::optional<uint16_t> fma_mix_f16_inline_bits(uint16_t src) {
  switch (src) {
  case 240:
    return 0x3800u; // 0.5f rounded to f16.
  case 241:
    return 0xB800u; // -0.5f rounded to f16.
  case 242:
    return 0x3C00u; // 1.0f rounded to f16.
  case 243:
    return 0xBC00u; // -1.0f rounded to f16.
  case 244:
    return 0x4000u; // 2.0f rounded to f16.
  case 245:
    return 0xC000u; // -2.0f rounded to f16.
  case 246:
    return 0x4400u; // 4.0f rounded to f16.
  case 247:
    return 0xC400u; // -4.0f rounded to f16.
  case 248:
    return 0x3118u; // 1/(2*pi) rounded to f16.
  default:
    return std::nullopt;
  }
}

[[nodiscard]] bool append_materialize_fma_mix_f16_source(std::vector<uint32_t> &words, uint8_t tmp,
                                                         uint16_t src, bool high_half,
                                                         std::optional<uint32_t> literal_word) {
  constexpr uint8_t kOpMovB32 = 1;
  constexpr uint16_t kOpCvtF32F16 = 0x18B;

  if (const auto inline_bits = fma_mix_f16_inline_bits(src)) {
    std::optional<uint32_t> literal;
    append_vop1(words, kOpMovB32, tmp, literal_or_inline_u32(*inline_bits, literal), literal);
  } else if (!append_materialize_b16_half(words, tmp, src, high_half, literal_word)) {
    return false;
  }

  append_wait_valu_vgpr(words);
  append_vop3(words, kOpCvtF32F16, tmp, vgpr_src(tmp), 0);
  return true;
}

std::vector<uint32_t> expand_v_fma_mix_f32_f16_vop3p(const Instruction &inst, uint32_t, uint64_t,
                                                     const LivenessAnalysis &liveness,
                                                     const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() != sizeof(gfx1250::Vop3pMachineInst))
    return {};

  gfx1250::Vop3pMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.vdst == 255 || src.src0 == 254 || src.src1 == 254 || src.src2 == 254)
    return {};

  const std::array<uint16_t, 3> sources{static_cast<uint16_t>(src.src0),
                                        static_cast<uint16_t>(src.src1),
                                        static_cast<uint16_t>(src.src2)};
  const std::array<bool, 3> source_is_f16{(src.opsel_hi & 0x1u) != 0, (src.opsel_hi & 0x2u) != 0,
                                          src.pad_14 != 0};
  const std::array<bool, 3> high_half{(src.opsel & 0x1u) != 0, (src.opsel & 0x2u) != 0,
                                      (src.opsel & 0x4u) != 0};

  std::array<std::optional<uint32_t>, 3> literals{};
  for (size_t i = 0; i < sources.size(); ++i) {
    if (sources[i] != 255)
      continue;
    literals[i] = simm32_literal_word(inst, static_cast<uint8_t>(i));
    if (!literals[i])
      return {};
  }

  std::array<uint16_t, 3> fma_sources = sources;
  const auto f16_source_count =
      static_cast<size_t>(std::count(source_is_f16.begin(), source_is_f16.end(), true));

  std::vector<uint8_t> avoid{static_cast<uint8_t>(src.vdst)};
  for (const uint16_t source : sources)
    add_avoid_src_vgpr(avoid, source);

  std::optional<uint16_t> tmp_base;
  if (f16_source_count != 0) {
    tmp_base = find_free_vgpr_run_avoiding(inst, liveness, f16_source_count, avoid);
    if (!tmp_base)
      return {};
  }

  std::vector<uint32_t> words;
  words.reserve(6 + f16_source_count * 4u);

  size_t tmp_index = 0;
  for (size_t i = 0; i < sources.size(); ++i) {
    if (!source_is_f16[i])
      continue;
    const auto tmp = static_cast<uint8_t>(*tmp_base + tmp_index);
    if (!append_materialize_fma_mix_f16_source(words, tmp, sources[i], high_half[i], literals[i]))
      return {};
    fma_sources[i] = vgpr_src(tmp);
    ++tmp_index;
  }

  if (f16_source_count != 0)
    append_wait_valu_vgpr(words);

  constexpr uint16_t kOpFmaF32 = 531;
  auto [w0, w1] =
      build_vop3_mod(kOpFmaF32, static_cast<uint8_t>(src.vdst), fma_sources[0], fma_sources[1],
                     fma_sources[2], 0, 0, src.clamp != 0, 0, static_cast<uint8_t>(src.neg & 0x7u));
  words.push_back(w0);
  words.push_back(w1);

  for (size_t i = 0; i < fma_sources.size(); ++i) {
    if (source_is_f16[i] || fma_sources[i] != 255)
      continue;
    words.push_back(*literals[i]);
    break;
  }

  return words;
}

void append_s_mov_b32_lit(std::vector<uint32_t> &words, uint8_t sdst, uint32_t literal) {
  auto [w0, w1] = build_s_mov_b32_lit(sdst, literal);
  words.push_back(w0);
  words.push_back(w1);
}

void append_set_exec_lo_mask(std::vector<uint32_t> &words, uint32_t mask) {
  constexpr uint8_t kExecLo = 126;
  append_s_mov_b32_lit(words, kExecLo, mask);
  append_s_mov_b32_lit(words, kExecLo + 1, 0);
  append_wait_salu_sgpr(words);
}

void append_set_exec_from_saved_xor16_mask(std::vector<uint32_t> &words, uint8_t exec_save) {
  constexpr uint8_t kExecLo = 126;
  constexpr uint8_t kExecHi = 127;
  constexpr uint8_t kOpSOrB32 = 24;
  constexpr uint8_t kOpSLshlB32 = 8;
  constexpr uint8_t kOpSLshrB32 = 10;
  constexpr uint16_t kInline16 = scalar_positive_inline_u32(16);

  // gfx1250 is wave32-only. Use EXEC_HI as a scratch scalar while forming the
  // lane-xor-16 mask, then clear it before any vector instruction observes EXEC.
  words.push_back(pack_sop2(kOpSLshrB32, kExecLo, exec_save, kInline16));
  words.push_back(pack_sop2(kOpSLshlB32, kExecHi, exec_save, kInline16));
  words.push_back(pack_sop2(kOpSOrB32, kExecLo, kExecLo, kExecHi));
  append_s_mov_b32_lit(words, kExecHi, 0);
  append_wait_salu_sgpr(words);
}

void append_scratch_store_b32(std::vector<uint32_t> &words, uint8_t vdata, uint32_t offset) {
  const auto encoded = build_scratch_store_b32(vdata, offset);
  words.insert(words.end(), encoded.begin(), encoded.end());
}

void append_scratch_load_b32(std::vector<uint32_t> &words, uint8_t vdst, uint32_t offset) {
  const auto encoded = build_scratch_load_b32(vdst, offset);
  words.insert(words.end(), encoded.begin(), encoded.end());
}

void append_lane_xor16_byte_addr(std::vector<uint32_t> &words, uint8_t vaddr) {
  constexpr uint16_t kOpMbcntLo = 0x31F;
  constexpr uint16_t kOpMbcntHi = 0x320;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpXorB32 = 29;
  constexpr uint16_t kInlineConst0 = 128;
  constexpr uint16_t kInlineConst2 = 130;
  constexpr uint16_t kInlineConstNeg1 = 193;
  constexpr uint16_t kInlineConst64 = 192;

  using P = HazardTracker::Pipeline;
  HazardTracker hz;
  {
    auto [w0, w1] = build_vop3(kOpMbcntLo, vaddr, kInlineConstNeg1, kInlineConst0);
    hz.emit2(words, w0, w1, P::VALU);
  }
  {
    auto [w0, w1] =
        build_vop3(kOpMbcntHi, vaddr, kInlineConstNeg1, static_cast<uint16_t>(256u + vaddr));
    hz.emit2(words, w0, w1, P::VALU);
  }
  hz.emit(words, build_vop2(kOpLshlrevB32, vaddr, kInlineConst2, vaddr), P::VALU);
  hz.emit(words, build_vop2(kOpXorB32, vaddr, kInlineConst64, vaddr), P::VALU);
}

void append_lane_id(std::vector<uint32_t> &words, uint8_t vaddr) {
  constexpr uint16_t kOpMbcntLo = 0x31F;
  constexpr uint16_t kOpMbcntHi = 0x320;
  constexpr uint16_t kInlineConst0 = 128;
  constexpr uint16_t kInlineConstNeg1 = 193;

  using P = HazardTracker::Pipeline;
  HazardTracker hz;
  {
    auto [w0, w1] = build_vop3(kOpMbcntLo, vaddr, kInlineConstNeg1, kInlineConst0);
    hz.emit2(words, w0, w1, P::VALU);
  }
  {
    auto [w0, w1] =
        build_vop3(kOpMbcntHi, vaddr, kInlineConstNeg1, static_cast<uint16_t>(256u + vaddr));
    hz.emit2(words, w0, w1, P::VALU);
  }
}

void append_wmma_f32_16x16x16_f16(std::vector<uint32_t> &words, uint8_t vdst, uint8_t src0,
                                  uint8_t src1, uint16_t src2,
                                  std::optional<uint32_t> literal_word = std::nullopt,
                                  uint8_t c_modifier = 0) {
  constexpr uint8_t kOpWmmaF32_16x16x16_F16 = 64;
  const uint8_t neg = (c_modifier & 0x1u) != 0 ? 0x4u : 0u;
  const uint8_t neg_hi = (c_modifier & 0x2u) != 0 ? 0x4u : 0u;
  auto [w0, w1] = build_vop3p(kOpWmmaF32_16x16x16_F16, vdst, static_cast<uint16_t>(256u + src0),
                              static_cast<uint16_t>(256u + src1), src2, neg, false, neg_hi);
  words.push_back(w0);
  words.push_back(w1);
  if (literal_word && src2 == 255)
    words.push_back(*literal_word);
}

void append_wmma_f32_16x16x16_bf16(std::vector<uint32_t> &words, uint8_t vdst, uint8_t src0,
                                   uint8_t src1, uint16_t src2,
                                   std::optional<uint32_t> literal_word = std::nullopt,
                                   uint8_t c_modifier = 0) {
  constexpr uint8_t kOpWmmaF32_16x16x16_BF16 = 65;
  const uint8_t neg = (c_modifier & 0x1u) != 0 ? 0x4u : 0u;
  const uint8_t neg_hi = (c_modifier & 0x2u) != 0 ? 0x4u : 0u;
  auto [w0, w1] = build_vop3p(kOpWmmaF32_16x16x16_BF16, vdst, static_cast<uint16_t>(256u + src0),
                              static_cast<uint16_t>(256u + src1), src2, neg, false, neg_hi);
  words.push_back(w0);
  words.push_back(w1);
  if (literal_word && src2 == 255)
    words.push_back(*literal_word);
}

void append_wmma_packed16_16x16x16(std::vector<uint32_t> &words, uint8_t op, uint8_t vdst,
                                   uint8_t src0, uint8_t src1, uint16_t src2,
                                   std::optional<uint32_t> literal_word = std::nullopt) {
  auto [w0, w1] = build_vop3p(op, vdst, static_cast<uint16_t>(256u + src0),
                              static_cast<uint16_t>(256u + src1), src2);
  words.push_back(w0);
  words.push_back(w1);
  if (literal_word && src2 == 255)
    words.push_back(*literal_word);
}

void append_wmma_2word_relayout_chunk(std::vector<uint32_t> &words, uint16_t src_a, uint16_t src_b,
                                      uint8_t tmp_a, uint8_t tmp_b, uint8_t vaddr_xor16,
                                      uint8_t source_words, uint8_t chunks_per_lane_group,
                                      uint8_t chunk, uint8_t exec_save) {
  constexpr uint8_t kOpMovB32 = 1;
  constexpr uint8_t kOpWaitDscnt = 70;
  const uint8_t src_a_vgpr = *src_vgpr_base_for_run(src_a, source_words);
  const uint8_t src_b_vgpr = *src_vgpr_base_for_run(src_b, source_words);

  const uint8_t chunk_in_lane_group = static_cast<uint8_t>(chunk % chunks_per_lane_group);
  const bool high_lane_group = chunk >= chunks_per_lane_group;
  const uint8_t word_base = static_cast<uint8_t>(chunk_in_lane_group * 4u);
  const uint8_t cross_word_base = static_cast<uint8_t>(word_base + (high_lane_group ? 0 : 2));
  const uint8_t local_word_base = static_cast<uint8_t>(word_base + (high_lane_group ? 2 : 0));

  append_set_exec_lo_mask(words, 0xFFFFFFFFu);
  append_v_mov_b32_broadcast(words, tmp_a, scalar_positive_inline_u32(0), 2);
  append_v_mov_b32_broadcast(words, tmp_b, scalar_positive_inline_u32(0), 2);
  append_set_exec_from_saved_xor16_mask(words, exec_save);
  for (uint8_t word = 0; word < 2; ++word) {
    auto [a_w0, a_w1] =
        build_ds_bpermute(static_cast<uint8_t>(tmp_a + word), vaddr_xor16,
                          static_cast<uint8_t>(src_a_vgpr + cross_word_base + word));
    words.push_back(a_w0);
    words.push_back(a_w1);
    auto [b_w0, b_w1] =
        build_ds_bpermute(static_cast<uint8_t>(tmp_b + word), vaddr_xor16,
                          static_cast<uint8_t>(src_b_vgpr + cross_word_base + word));
    words.push_back(b_w0);
    words.push_back(b_w1);
  }
  words.push_back(pack_sopp(kOpWaitDscnt, 0));

  append_set_exec_from_saved_mask(words, exec_save, high_lane_group ? 0xFFFF0000u : 0x0000FFFFu);
  for (uint8_t word = 0; word < 2; ++word) {
    append_vop1(words, kOpMovB32, static_cast<uint8_t>(tmp_a + word),
                static_cast<uint16_t>(src_a + local_word_base + word));
    append_vop1(words, kOpMovB32, static_cast<uint8_t>(tmp_b + word),
                static_cast<uint16_t>(src_b + local_word_base + word));
  }

  append_restore_exec(words, exec_save);
  append_wait_valu_vgpr(words);
}

void append_wmma_single_relayout_chunk(std::vector<uint32_t> &words, uint16_t src, uint8_t tmp,
                                       uint8_t vaddr_xor16, uint8_t source_words,
                                       uint8_t words_per_chunk, uint8_t chunk, uint8_t exec_save) {
  constexpr uint8_t kOpMovB32 = 1;
  constexpr uint8_t kOpWaitDscnt = 70;
  const uint8_t src_vgpr = *src_vgpr_base_for_run(src, source_words);
  const uint8_t chunks_per_lane_group = static_cast<uint8_t>(source_words / (2u * words_per_chunk));

  const uint8_t chunk_in_lane_group = static_cast<uint8_t>(chunk % chunks_per_lane_group);
  const bool high_lane_group = chunk >= chunks_per_lane_group;
  const uint8_t word_base = static_cast<uint8_t>(chunk_in_lane_group * 2u * words_per_chunk);
  const uint8_t cross_word_base =
      static_cast<uint8_t>(word_base + (high_lane_group ? 0u : words_per_chunk));
  const uint8_t local_word_base =
      static_cast<uint8_t>(word_base + (high_lane_group ? words_per_chunk : 0u));

  append_set_exec_lo_mask(words, 0xFFFFFFFFu);
  append_v_mov_b32_broadcast(words, tmp, scalar_positive_inline_u32(0), words_per_chunk);
  append_set_exec_from_saved_xor16_mask(words, exec_save);
  for (uint8_t word = 0; word < words_per_chunk; ++word) {
    auto [w0, w1] = build_ds_bpermute(static_cast<uint8_t>(tmp + word), vaddr_xor16,
                                      static_cast<uint8_t>(src_vgpr + cross_word_base + word));
    words.push_back(w0);
    words.push_back(w1);
  }
  words.push_back(pack_sopp(kOpWaitDscnt, 0));

  append_set_exec_from_saved_mask(words, exec_save, high_lane_group ? 0xFFFF0000u : 0x0000FFFFu);
  for (uint8_t word = 0; word < words_per_chunk; ++word) {
    append_vop1(words, kOpMovB32, static_cast<uint8_t>(tmp + word),
                static_cast<uint16_t>(src + local_word_base + word));
  }

  append_restore_exec(words, exec_save);
  append_wait_valu_vgpr(words);
}

void append_wmma_i32_16x16x16_iu8(std::vector<uint32_t> &words, uint8_t vdst, uint8_t src0,
                                  uint8_t src1, uint16_t src2, uint8_t neg, bool clamp,
                                  std::optional<uint32_t> literal_word = std::nullopt) {
  constexpr uint8_t kOpWmmaI32_16x16x16_IU8 = 68;
  auto [w0, w1] = build_vop3p(kOpWmmaI32_16x16x16_IU8, vdst, static_cast<uint16_t>(256u + src0),
                              static_cast<uint16_t>(256u + src1), src2, neg, clamp);
  words.push_back(w0);
  words.push_back(w1);
  if (literal_word && src2 == 255)
    words.push_back(*literal_word);
}

void append_swmmac_16x16x32(std::vector<uint32_t> &words, uint8_t op, uint8_t vdst, uint8_t src0,
                            uint8_t src1, uint8_t src2) {
  auto [w0, w1] =
      build_vop3p(op, vdst, static_cast<uint16_t>(256u + src0), static_cast<uint16_t>(256u + src1),
                  static_cast<uint16_t>(256u + src2));
  words.push_back(w0);
  words.push_back(w1);
}

void append_swmmac_index_chunk_relayout(std::vector<uint32_t> &words, uint8_t dst_index,
                                        uint8_t vaddr, uint8_t src_index, uint8_t chunk,
                                        bool byte_format, uint8_t exec_save) {
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpOrB32 = 28;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpLshrrevB32 = 25;
  constexpr uint8_t kOpAddNcU32 = 37;
  constexpr uint16_t kInlineConst2 = 130;
  constexpr uint16_t kInlineConst15 = 143;
  constexpr uint16_t kInlineConst16 = 144;
  constexpr uint8_t kOpWaitDscnt = 70;

  // A gfx1250 K64/K128 instruction stores one K32 chunk's 16 selectors on
  // one source row lane. RDNA4 K32 consumes the same selectors from both row
  // half-lanes; byte formats additionally split the low/high eight entries.
  append_set_exec_lo_mask(words, 0xFFFFFFFFu);
  append_lane_id(words, vaddr);
  append_vop2(words, kOpAndB32, vaddr, kInlineConst15, vaddr);
  const uint8_t source_lane_half = byte_format ? static_cast<uint8_t>(chunk / 2u) : chunk;
  if (source_lane_half != 0u)
    append_vop2(words, kOpAddNcU32, vaddr, kInlineConst16, vaddr);
  append_vop2(words, kOpLshlrevB32, vaddr, kInlineConst2, vaddr);
  append_wait_valu_vgpr(words);
  const uint8_t source_word = static_cast<uint8_t>(src_index + (byte_format ? (chunk & 1u) : 0u));
  auto [word0, word1] = build_ds_bpermute(dst_index, vaddr, source_word);
  words.push_back(word0);
  words.push_back(word1);
  words.push_back(pack_sopp(kOpWaitDscnt, 0));

  if (byte_format) {
    append_lane_id(words, vaddr);
    append_vop2(words, kOpAndB32, vaddr, kInlineConst16, vaddr);
    append_vop2(words, kOpLshrrevB32, dst_index, vgpr_src(vaddr), dst_index);
  } else {
    // gfx1250 K64 f16/bf16 metadata interleaves compressed-K bit 1 into
    // bits 16..31 and bit 2 into the source lane. RDNA4 K32 keeps bit 1 in
    // bits 4..7/12..15 and uses bit 2 as its target lane half.
    append_set_exec_from_saved_mask(words, exec_save, 0x0000FFFFu);
    append_vop2(words, kOpAndB32, vaddr, 255, dst_index, 0x00000F0Fu);
    append_vop2(words, kOpAndB32, dst_index, 255, dst_index, 0x0F0F0000u);
    append_wait_valu_vgpr(words);
    append_vop2(words, kOpLshrrevB32, dst_index, scalar_positive_inline_u32(12), dst_index);
    append_wait_valu_vgpr(words);
    append_vop2(words, kOpOrB32, dst_index, vgpr_src(vaddr), dst_index);

    append_set_exec_from_saved_mask(words, exec_save, 0xFFFF0000u);
    append_vop2(words, kOpAndB32, vaddr, 255, dst_index, 0x0000F0F0u);
    append_vop2(words, kOpAndB32, dst_index, 255, dst_index, 0xF0F00000u);
    append_wait_valu_vgpr(words);
    append_vop2(words, kOpLshrrevB32, vaddr, scalar_positive_inline_u32(4), vaddr);
    append_vop2(words, kOpLshrrevB32, dst_index, kInlineConst16, dst_index);
    append_wait_valu_vgpr(words);
    append_vop2(words, kOpOrB32, dst_index, vgpr_src(vaddr), dst_index);
  }
  append_restore_exec(words, exec_save);
  append_wait_valu_vgpr(words);
}

void append_swmmac_fp8_index_chunk_relayout(std::vector<uint32_t> &words, uint8_t dst_index,
                                            uint8_t merge_tmp, uint8_t vaddr, uint8_t src_index,
                                            uint8_t chunk, uint8_t exec_save) {
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpOrB32 = 28;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpLshrrevB32 = 25;
  constexpr uint8_t kOpAddNcU32 = 37;
  constexpr uint16_t kInlineConst2 = 130;
  constexpr uint16_t kInlineConst8 = 136;
  constexpr uint16_t kInlineConst15 = 143;
  constexpr uint16_t kInlineConst16 = 144;
  constexpr uint8_t kOpWaitDscnt = 70;

  // gfx1250 alternates adjacent sparse groups between two index dwords:
  //   dword=(cc/2)&1, bit=4*((cc/4)%8)+2*(cc&1).
  // RDNA4 instead packs each K32 half linearly. Gather the source dwords from
  // the chunk's row lane, select the target lane's byte, then interleave their
  // low/high nibbles into the target 16-bit selector set.
  append_set_exec_lo_mask(words, 0xFFFFFFFFu);
  append_lane_id(words, vaddr);
  append_vop2(words, kOpAndB32, vaddr, kInlineConst15, vaddr);
  if (chunk >= 2u)
    append_vop2(words, kOpAddNcU32, vaddr, kInlineConst16, vaddr);
  append_vop2(words, kOpLshlrevB32, vaddr, kInlineConst2, vaddr);
  append_wait_valu_vgpr(words);
  auto [lo0, lo1] = build_ds_bpermute(dst_index, vaddr, src_index);
  words.push_back(lo0);
  words.push_back(lo1);
  auto [hi0, hi1] = build_ds_bpermute(merge_tmp, vaddr, static_cast<uint8_t>(src_index + 1u));
  words.push_back(hi0);
  words.push_back(hi1);
  words.push_back(pack_sopp(kOpWaitDscnt, 0));

  if ((chunk & 1u) != 0u) {
    append_vop2(words, kOpLshrrevB32, dst_index, kInlineConst16, dst_index);
    append_vop2(words, kOpLshrrevB32, merge_tmp, kInlineConst16, merge_tmp);
    append_wait_valu_vgpr(words);
  }
  append_set_exec_from_saved_mask(words, exec_save, 0xFFFF0000u);
  append_vop2(words, kOpLshrrevB32, dst_index, kInlineConst8, dst_index);
  append_vop2(words, kOpLshrrevB32, merge_tmp, kInlineConst8, merge_tmp);
  append_wait_valu_vgpr(words);
  append_restore_exec(words, exec_save);

  std::optional<uint32_t> mask_literal;
  append_vop2(words, kOpAndB32, vaddr, literal_or_inline_u32(0xF0u, mask_literal), dst_index,
              mask_literal);
  append_vop2(words, kOpLshlrevB32, vaddr, scalar_positive_inline_u32(4), vaddr);
  append_vop2(words, kOpAndB32, dst_index, scalar_positive_inline_u32(15), dst_index);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpOrB32, dst_index, vgpr_src(vaddr), dst_index);
  append_wait_valu_vgpr(words);

  mask_literal.reset();
  append_vop2(words, kOpAndB32, vaddr, scalar_positive_inline_u32(15), merge_tmp);
  append_vop2(words, kOpLshlrevB32, vaddr, scalar_positive_inline_u32(4), vaddr);
  append_vop2(words, kOpAndB32, merge_tmp, literal_or_inline_u32(0xF0u, mask_literal), merge_tmp,
              mask_literal);
  append_vop2(words, kOpLshlrevB32, merge_tmp, scalar_positive_inline_u32(8), merge_tmp);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpOrB32, dst_index, vgpr_src(vaddr), dst_index);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpOrB32, dst_index, vgpr_src(merge_tmp), dst_index);
  append_wait_valu_vgpr(words);
}

void append_swmmac_select_word_by_target_half(std::vector<uint32_t> &words, uint8_t dst,
                                              uint8_t vaddr, uint8_t src_base,
                                              uint8_t source_word_low, uint8_t source_word_high,
                                              uint8_t source_lane_half, uint8_t exec_save) {
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpAddNcU32 = 37;
  constexpr uint16_t kInlineConst2 = 130;
  constexpr uint16_t kInlineConst15 = 143;
  constexpr uint16_t kInlineConst16 = 144;
  constexpr uint8_t kOpWaitDscnt = 70;

  append_set_exec_lo_mask(words, 0xFFFFFFFFu);
  append_lane_id(words, vaddr);
  append_vop2(words, kOpAndB32, vaddr, kInlineConst15, vaddr);
  if (source_lane_half != 0u)
    append_vop2(words, kOpAddNcU32, vaddr, kInlineConst16, vaddr);
  append_vop2(words, kOpLshlrevB32, vaddr, kInlineConst2, vaddr);
  append_wait_valu_vgpr(words);

  // ds_bpermute only reads from lanes that are active in EXEC. Seed both
  // target halves while the selected source half is active, then overwrite
  // the target half that shares that source half. Ordering the half-wave
  // gather last also preserves the other half of dst.
  const uint8_t first_word = source_lane_half == 0u ? source_word_high : source_word_low;
  const uint8_t second_word = source_lane_half == 0u ? source_word_low : source_word_high;
  const uint32_t second_mask = source_lane_half == 0u ? 0x0000FFFFu : 0xFFFF0000u;
  auto [first0, first1] =
      build_ds_bpermute(dst, vaddr, static_cast<uint8_t>(src_base + first_word));
  words.push_back(first0);
  words.push_back(first1);
  words.push_back(pack_sopp(kOpWaitDscnt, 0));
  append_set_exec_from_saved_mask(words, exec_save, second_mask);
  auto [second0, second1] =
      build_ds_bpermute(dst, vaddr, static_cast<uint8_t>(src_base + second_word));
  words.push_back(second0);
  words.push_back(second1);
  words.push_back(pack_sopp(kOpWaitDscnt, 0));
  append_restore_exec(words, exec_save);
  append_wait_valu_vgpr(words);
}

void append_swmmac_f16_a_chunk_relayout(std::vector<uint32_t> &words, uint8_t dst_base,
                                        uint8_t vaddr, uint8_t src_base, uint8_t chunk,
                                        uint8_t exec_save) {
  for (uint8_t word = 0; word < 4; ++word) {
    const uint8_t source_word_low = static_cast<uint8_t>(4u * chunk + 2u * (word >> 1u));
    append_swmmac_select_word_by_target_half(
        words, static_cast<uint8_t>(dst_base + word), vaddr, src_base, source_word_low,
        static_cast<uint8_t>(source_word_low + 1u), static_cast<uint8_t>(word & 1u), exec_save);
  }
}

void append_swmmac_f16_b_chunk_relayout(std::vector<uint32_t> &words, uint8_t dst_base,
                                        uint8_t vaddr, uint8_t src_base, uint8_t chunk,
                                        uint8_t exec_save) {
  for (uint8_t word = 0; word < 8; ++word) {
    const uint8_t source_word_low =
        static_cast<uint8_t>(8u * chunk + (word & 1u) + 4u * (word >> 2u));
    append_swmmac_select_word_by_target_half(words, static_cast<uint8_t>(dst_base + word), vaddr,
                                             src_base, source_word_low,
                                             static_cast<uint8_t>(source_word_low + 2u),
                                             static_cast<uint8_t>((word >> 1u) & 1u), exec_save);
  }
}

void append_swmmac_fp8_a_chunk_relayout(std::vector<uint32_t> &words, uint8_t dst_base,
                                        uint8_t merge_tmp, uint8_t vaddr, uint8_t src_base,
                                        uint8_t chunk, uint8_t exec_save) {
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpOrB32 = 28;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpLshrrevB32 = 25;

  // gfx1250 stores the two compressed values in each 2:4 group on sibling
  // lanes. RDNA4 stores them beside each other in one dword. Gather both
  // sibling words, then merge the selected 16-bit pair for this target word.
  for (uint8_t word = 0; word < 2; ++word) {
    const uint8_t source_word_low = static_cast<uint8_t>(2u * chunk);
    const uint8_t source_word_high = static_cast<uint8_t>(source_word_low + 1u);
    append_swmmac_select_word_by_target_half(words, static_cast<uint8_t>(dst_base + word), vaddr,
                                             src_base, source_word_low, source_word_high, 0,
                                             exec_save);
    append_swmmac_select_word_by_target_half(words, merge_tmp, vaddr, src_base, source_word_low,
                                             source_word_high, 1, exec_save);

    if (word == 0) {
      std::optional<uint32_t> mask_literal;
      append_vop2(words, kOpAndB32, static_cast<uint8_t>(dst_base + word),
                  literal_or_inline_u32(0xFFFFu, mask_literal),
                  static_cast<uint8_t>(dst_base + word), mask_literal);
      append_vop2(words, kOpLshlrevB32, merge_tmp, scalar_positive_inline_u32(16), merge_tmp);
    } else {
      append_vop2(words, kOpLshrrevB32, static_cast<uint8_t>(dst_base + word),
                  scalar_positive_inline_u32(16), static_cast<uint8_t>(dst_base + word));
      std::optional<uint32_t> mask_literal;
      append_vop2(words, kOpAndB32, merge_tmp, literal_or_inline_u32(0xFFFF0000u, mask_literal),
                  merge_tmp, mask_literal);
    }
    append_wait_valu_vgpr(words);
    append_vop2(words, kOpOrB32, static_cast<uint8_t>(dst_base + word), vgpr_src(merge_tmp),
                static_cast<uint8_t>(dst_base + word));
    append_wait_valu_vgpr(words);
  }
}

void append_swmmac_fp8_b_chunk_relayout(std::vector<uint32_t> &words, uint8_t dst_base,
                                        uint8_t vaddr, uint8_t src_base, uint8_t chunk,
                                        uint8_t exec_save) {
  for (uint8_t word = 0; word < 4; ++word) {
    const uint8_t source_word_low = static_cast<uint8_t>(4u * chunk + (word >> 1u));
    append_swmmac_select_word_by_target_half(
        words, static_cast<uint8_t>(dst_base + word), vaddr, src_base, source_word_low,
        static_cast<uint8_t>(source_word_low + 2u), static_cast<uint8_t>(word & 1u), exec_save);
  }
}

void append_swmmac_i32_16x16x32_iu8(std::vector<uint32_t> &words, uint8_t vdst, uint8_t src0,
                                    uint8_t src1, uint8_t src2, uint8_t neg, bool clamp) {
  constexpr uint8_t kOpSwmmacI32_16x16x32_IU8 = 0x54;
  auto [w0, w1] = build_vop3p(kOpSwmmacI32_16x16x32_IU8, vdst, static_cast<uint16_t>(256u + src0),
                              static_cast<uint16_t>(256u + src1),
                              static_cast<uint16_t>(256u + src2), neg, clamp);
  words.push_back(w0);
  words.push_back(w1);
}

void append_wmma_f32_16x16x16_f8(std::vector<uint32_t> &words, uint8_t op, uint8_t vdst,
                                 uint8_t src0, uint8_t src1, uint16_t src2,
                                 std::optional<uint32_t> literal_word = std::nullopt) {
  auto [w0, w1] = build_vop3p(op, vdst, static_cast<uint16_t>(256u + src0),
                              static_cast<uint16_t>(256u + src1), src2);
  words.push_back(w0);
  words.push_back(w1);
  if (literal_word && src2 == 255)
    words.push_back(*literal_word);
}

void append_v_dot4_f32_f8(std::vector<uint32_t> &words, uint8_t op, uint8_t vdst, uint8_t src0,
                          uint8_t src1, uint16_t src2) {
  auto [w0, w1] = build_vop3p(op, vdst, static_cast<uint16_t>(256u + src0),
                              static_cast<uint16_t>(256u + src1), src2);
  words.push_back(w0);
  words.push_back(w1);
}

void append_v_dot4_f32_f8_encoded(std::vector<uint32_t> &words, uint8_t op, uint8_t vdst,
                                  uint16_t src0, uint16_t src1, uint16_t src2) {
  auto [w0, w1] = build_vop3p(op, vdst, src0, src1, src2);
  words.push_back(w0);
  words.push_back(w1);
}

void append_v_dot8_u32_u4(std::vector<uint32_t> &words, uint8_t vdst, uint8_t src0, uint8_t src1,
                          uint16_t src2) {
  constexpr uint8_t kOpVDot8U32U4 = 0x19;
  auto [w0, w1] = build_vop3p(kOpVDot8U32U4, vdst, static_cast<uint16_t>(256u + src0),
                              static_cast<uint16_t>(256u + src1), src2);
  words.push_back(w0);
  words.push_back(w1);
}

void append_fp4_scaled_mag2(std::vector<uint32_t> &words, uint8_t dst, uint8_t raw,
                            uint8_t scratch0, uint8_t scratch1, uint8_t scratch2) {
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpOrB32 = 28;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpLshrrevB32 = 25;
  constexpr uint8_t kOpNotB32 = 55;
  constexpr uint16_t kInlineConst1 = 129;
  constexpr uint16_t kInlineConst2 = 130;
  constexpr uint32_t kNibbleBit0 = 0x11111111u;
  constexpr uint32_t kNibbleBit1 = 0x22222222u;
  constexpr uint32_t kNibbleBit2 = 0x44444444u;

  // OCP MX FP4 E2M1 magnitudes scaled by two:
  // low3 0..7 -> 0, 1, 2, 3, 4, 6, 8, 12.
  append_vop2(words, kOpAndB32, dst, 255, raw, kNibbleBit0);
  append_vop2(words, kOpAndB32, scratch0, 255, raw, kNibbleBit2);
  append_vop2(words, kOpLshrrevB32, scratch1, kInlineConst2, scratch0);
  append_vop1(words, kOpNotB32, scratch1, static_cast<uint16_t>(256u + scratch1));
  append_vop2(words, kOpAndB32, dst, static_cast<uint16_t>(256u + scratch1), dst);

  append_vop2(words, kOpAndB32, scratch0, 255, raw, kNibbleBit1);
  append_vop2(words, kOpAndB32, scratch1, 255, raw, kNibbleBit2);
  append_vop2(words, kOpLshrrevB32, scratch1, kInlineConst1, scratch1);
  append_vop1(words, kOpNotB32, scratch1, static_cast<uint16_t>(256u + scratch1));
  append_vop2(words, kOpAndB32, scratch1, static_cast<uint16_t>(256u + scratch1), scratch0);
  append_vop2(words, kOpAndB32, scratch2, 255, raw, kNibbleBit0);
  append_vop2(words, kOpLshlrevB32, scratch2, kInlineConst1, scratch2);
  append_vop2(words, kOpAndB32, scratch0, 255, raw, kNibbleBit2);
  append_vop2(words, kOpLshrrevB32, scratch0, kInlineConst1, scratch0);
  append_vop2(words, kOpAndB32, scratch2, static_cast<uint16_t>(256u + scratch0), scratch2);
  append_vop2(words, kOpAndB32, scratch0, 255, raw, kNibbleBit1);
  append_vop1(words, kOpNotB32, scratch0, static_cast<uint16_t>(256u + scratch0));
  append_vop2(words, kOpAndB32, scratch2, static_cast<uint16_t>(256u + scratch0), scratch2);
  append_vop2(words, kOpOrB32, scratch1, static_cast<uint16_t>(256u + scratch2), scratch1);
  append_vop2(words, kOpOrB32, dst, static_cast<uint16_t>(256u + scratch1), dst);

  append_vop2(words, kOpAndB32, scratch0, 255, raw, kNibbleBit2);
  append_vop2(words, kOpAndB32, scratch1, 255, raw, kNibbleBit1);
  append_vop2(words, kOpLshlrevB32, scratch1, kInlineConst1, scratch1);
  append_vop1(words, kOpNotB32, scratch1, static_cast<uint16_t>(256u + scratch1));
  append_vop2(words, kOpAndB32, scratch2, 255, raw, kNibbleBit0);
  append_vop2(words, kOpLshlrevB32, scratch2, kInlineConst2, scratch2);
  append_vop2(words, kOpOrB32, scratch1, static_cast<uint16_t>(256u + scratch2), scratch1);
  append_vop2(words, kOpAndB32, scratch1, static_cast<uint16_t>(256u + scratch0), scratch1);
  append_vop2(words, kOpOrB32, dst, static_cast<uint16_t>(256u + scratch1), dst);

  append_vop2(words, kOpAndB32, scratch0, 255, raw, kNibbleBit2);
  append_vop2(words, kOpLshlrevB32, scratch0, kInlineConst1, scratch0);
  append_vop2(words, kOpAndB32, scratch1, 255, raw, kNibbleBit1);
  append_vop2(words, kOpLshlrevB32, scratch1, kInlineConst2, scratch1);
  append_vop2(words, kOpAndB32, scratch0, static_cast<uint16_t>(256u + scratch1), scratch0);
  append_vop2(words, kOpOrB32, dst, static_cast<uint16_t>(256u + scratch0), dst);
}

void append_fp4_scaled_mag2_pos_neg(std::vector<uint32_t> &words, uint8_t raw, uint8_t pos,
                                    uint8_t neg, uint8_t scratch0, uint8_t scratch1,
                                    uint8_t scratch2) {
  constexpr uint8_t kOpMovB32 = 1;
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpOrB32 = 28;
  constexpr uint8_t kOpLshrrevB32 = 25;
  constexpr uint8_t kOpNotB32 = 55;
  constexpr uint16_t kInlineConst1 = 129;
  constexpr uint16_t kInlineConst2 = 130;
  constexpr uint32_t kNibbleSign = 0x88888888u;

  append_fp4_scaled_mag2(words, pos, raw, scratch0, scratch1, scratch2);
  append_vop2(words, kOpAndB32, neg, 255, raw, kNibbleSign);
  append_vop2(words, kOpLshrrevB32, scratch0, kInlineConst1, neg);
  append_vop2(words, kOpOrB32, neg, static_cast<uint16_t>(256u + scratch0), neg);
  append_vop2(words, kOpLshrrevB32, scratch0, kInlineConst2, neg);
  append_vop2(words, kOpOrB32, neg, static_cast<uint16_t>(256u + scratch0), neg);
  append_vop2(words, kOpAndB32, scratch0, static_cast<uint16_t>(256u + pos), neg);
  append_vop1(words, kOpNotB32, scratch1, static_cast<uint16_t>(256u + neg));
  append_vop2(words, kOpAndB32, pos, static_cast<uint16_t>(256u + scratch1), pos);
  append_vop1(words, kOpMovB32, neg, static_cast<uint16_t>(256u + scratch0));
}

struct WmmaPackedFieldLoc {
  uint8_t word;
  uint8_t bit_offset;
  bool high_lane_group;
};

[[nodiscard]] constexpr uint8_t wmma_f8f6f4_format_bits(uint8_t format) {
  return format <= 1u ? 8u : format <= 3u ? 6u : 4u;
}

[[nodiscard]] WmmaPackedFieldLoc wmma_f8f6f4_field_loc(uint8_t format, uint8_t other_format,
                                                       uint8_t k) {
  const uint8_t bits = wmma_f8f6f4_format_bits(format);
  const bool mixed_subbyte = bits < 8u && wmma_f8f6f4_format_bits(other_format) == 8u;
  uint8_t slot = 0;
  bool high_lane_group = false;
  if (mixed_subbyte) {
    high_lane_group = ((k >> 5u) & 1u) != 0;
    slot = static_cast<uint8_t>(32u * ((k >> 6u) & 1u) + 16u * ((k >> 2u) & 1u) +
                                8u * ((k >> 4u) & 1u) + 4u * ((k >> 3u) & 1u) +
                                2u * ((k >> 1u) & 1u) + (k & 1u));
  } else {
    high_lane_group = ((k >> 2u) & 1u) != 0;
    const uint8_t reg =
        static_cast<uint8_t>(((k >> 1u) & 1u) + 2u * ((k >> 3u) & 1u) + 4u * ((k >> 4u) & 1u) +
                             8u * ((k >> 5u) & 1u) + 16u * ((k >> 6u) & 1u));
    slot = static_cast<uint8_t>(2u * reg + (k & 1u));
  }
  const uint16_t bit = static_cast<uint16_t>(slot) * bits;
  return {static_cast<uint8_t>(bit / 32u), static_cast<uint8_t>(bit % 32u), high_lane_group};
}

void append_wmma_gather_packed_field(std::vector<uint32_t> &words, uint8_t result,
                                     uint8_t next_word, uint8_t vaddr, uint8_t src_base,
                                     uint8_t output_reg, bool matrix_a, uint8_t bits,
                                     WmmaPackedFieldLoc loc, uint8_t exec_save) {
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpOrB32 = 28;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpLshrrevB32 = 25;
  constexpr uint8_t kOpAddNcU32 = 37;
  constexpr uint8_t kOpXorB32 = 29;
  constexpr uint16_t kInlineConst1 = 129;
  constexpr uint16_t kInlineConst2 = 130;
  constexpr uint16_t kInlineConst15 = 143;
  constexpr uint16_t kInlineConst16 = 144;
  constexpr uint16_t kInlineConst64 = 192;
  constexpr uint8_t kOpWaitDscnt = 70;

  append_set_exec_lo_mask(words, 0xFFFFFFFFu);
  append_lane_id(words, vaddr);
  if (matrix_a) {
    append_vop2(words, kOpAndB32, vaddr, kInlineConst16, vaddr);
    append_vop2(words, kOpLshlrevB32, vaddr, kInlineConst1, vaddr);
    if (output_reg != 0) {
      append_vop2(words, kOpAddNcU32, vaddr,
                  scalar_positive_inline_u32(static_cast<uint16_t>(output_reg * 4u)), vaddr);
    }
  } else {
    append_vop2(words, kOpAndB32, vaddr, kInlineConst15, vaddr);
    append_vop2(words, kOpLshlrevB32, vaddr, kInlineConst2, vaddr);
  }
  if (loc.high_lane_group)
    append_vop2(words, kOpXorB32, vaddr, kInlineConst64, vaddr);
  append_wait_valu_vgpr(words);

  auto [word0, word1] = build_ds_bpermute(result, vaddr, static_cast<uint8_t>(src_base + loc.word));
  words.push_back(word0);
  words.push_back(word1);
  const bool crosses_word = static_cast<uint16_t>(loc.bit_offset) + bits > 32u;
  if (crosses_word) {
    auto [next0, next1] =
        build_ds_bpermute(next_word, vaddr, static_cast<uint8_t>(src_base + loc.word + 1u));
    words.push_back(next0);
    words.push_back(next1);
  }
  words.push_back(pack_sopp(kOpWaitDscnt, 0));

  if (loc.bit_offset != 0) {
    append_vop2(words, kOpLshrrevB32, result, scalar_positive_inline_u32(loc.bit_offset), result);
  }
  if (crosses_word) {
    append_vop2(words, kOpLshlrevB32, next_word,
                scalar_positive_inline_u32(static_cast<uint16_t>(32u - loc.bit_offset)), next_word);
    append_wait_valu_vgpr(words);
    append_vop2(words, kOpOrB32, result, vgpr_src(next_word), result);
  }
  append_wait_valu_vgpr(words);
  std::optional<uint32_t> mask_literal;
  const uint16_t mask_src = literal_or_inline_u32((1u << bits) - 1u, mask_literal);
  append_vop2(words, kOpAndB32, result, mask_src, result, mask_literal);
  append_restore_exec(words, exec_save);
  append_wait_valu_vgpr(words);
}

void append_wmma_f4_32x16_gather_a(std::vector<uint32_t> &words, uint8_t result, uint8_t vaddr,
                                   uint8_t src_base, uint8_t output_reg, uint8_t k,
                                   uint8_t exec_save) {
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpLshrrevB32 = 25;
  constexpr uint8_t kOpAddNcU32 = 37;
  constexpr uint8_t kOpXorB32 = 29;
  constexpr uint16_t kInlineConst1 = 129;
  constexpr uint16_t kInlineConst2 = 130;
  constexpr uint16_t kInlineConst15 = 143;
  constexpr uint16_t kInlineConst16 = 144;
  constexpr uint8_t kOpWaitDscnt = 70;

  // For row = output_reg + 16*(lane/16), the silicon-grounded source layout is:
  //   lane = (output_reg&7) + 8*(lane/16) + 16*((k>>2)&1)
  //   slot = 64*(output_reg>>3) + (k&3) + 4*(k>>3)
  // The output row group therefore selects the high eight source words while
  // the current half-wave selects the middle lane bit.
  const uint16_t slot = static_cast<uint16_t>(64u * (output_reg >> 3u) + (k & 3u) + 4u * (k >> 3u));
  const uint8_t word = static_cast<uint8_t>(slot / 8u);
  const uint8_t bit_offset = static_cast<uint8_t>(4u * (slot % 8u));

  append_set_exec_lo_mask(words, 0xFFFFFFFFu);
  append_lane_id(words, vaddr);
  append_vop2(words, kOpAndB32, vaddr, kInlineConst16, vaddr);
  append_vop2(words, kOpLshrrevB32, vaddr, kInlineConst1, vaddr);
  if ((output_reg & 7u) != 0u) {
    append_vop2(words, kOpAddNcU32, vaddr, scalar_positive_inline_u32(output_reg & 7u), vaddr);
  }
  if ((k >> 2u) & 1u)
    append_vop2(words, kOpXorB32, vaddr, kInlineConst16, vaddr);
  append_vop2(words, kOpLshlrevB32, vaddr, kInlineConst2, vaddr);
  append_wait_valu_vgpr(words);

  auto [word0, word1] = build_ds_bpermute(result, vaddr, static_cast<uint8_t>(src_base + word));
  words.push_back(word0);
  words.push_back(word1);
  words.push_back(pack_sopp(kOpWaitDscnt, 0));
  if (bit_offset != 0u)
    append_vop2(words, kOpLshrrevB32, result, scalar_positive_inline_u32(bit_offset), result);
  append_vop2(words, kOpAndB32, result, kInlineConst15, result);
  append_restore_exec(words, exec_save);
  append_wait_valu_vgpr(words);
}

void append_wmma_lowp_to_f32(std::vector<uint32_t> &words, uint8_t result, uint8_t raw,
                             uint8_t sign, uint8_t exponent, uint8_t mantissa, uint8_t subnormal,
                             uint8_t predicate, uint8_t format) {
  constexpr uint16_t kOpCvtF32Fp8 = 492;
  constexpr uint16_t kOpCvtF32Bf8 = 493;
  if (format <= 1u) {
    append_vop3_mod(words, format == 1u ? kOpCvtF32Bf8 : kOpCvtF32Fp8, result, vgpr_src(raw), 0, 0,
                    0);
    append_wait_valu_vgpr(words);
    return;
  }

  constexpr uint8_t kOpMovB32 = 1;
  constexpr uint8_t kOpCvtF32U32 = 6;
  constexpr uint8_t kOpMulF32 = 8;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpLshrrevB32 = 25;
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpOrB32 = 28;
  constexpr uint8_t kOpAddNcU32 = 37;
  constexpr uint16_t kOpCmpEqU32 = 74;
  constexpr uint16_t kOpCndmaskB32 = 257;
  constexpr uint8_t kSoppWaitAlu = 8;

  const uint8_t total_bits = format == 4u ? 4u : 6u;
  const uint8_t exponent_bits = format == 3u ? 3u : 2u;
  const uint8_t mantissa_bits = format == 2u ? 3u : format == 3u ? 2u : 1u;
  const uint8_t bias = format == 3u ? 3u : 1u;
  const uint32_t sign_mask = 1u << (total_bits - 1u);
  const uint32_t exponent_mask = (1u << exponent_bits) - 1u;
  const uint32_t mantissa_mask = (1u << mantissa_bits) - 1u;
  const uint32_t subnormal_scale_bits = (127u + 1u - bias - mantissa_bits) << 23u;

  append_vop2(words, kOpAndB32, sign, scalar_positive_inline_u32(sign_mask), raw);
  append_vop2(words, kOpLshrrevB32, exponent, scalar_positive_inline_u32(mantissa_bits), raw);
  append_vop2(words, kOpAndB32, mantissa, scalar_positive_inline_u32(mantissa_mask), raw);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpAndB32, exponent, scalar_positive_inline_u32(exponent_mask), exponent);
  append_vop2(words, kOpLshlrevB32, sign,
              scalar_positive_inline_u32(static_cast<uint16_t>(32u - total_bits)), sign);
  append_vop1(words, kOpMovB32, subnormal, vgpr_src(mantissa));

  append_vop2(words, kOpAddNcU32, result, 255, exponent, 127u - bias);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpLshlrevB32, result, scalar_positive_inline_u32(23), result);
  append_vop2(words, kOpLshlrevB32, mantissa,
              scalar_positive_inline_u32(static_cast<uint16_t>(23u - mantissa_bits)), mantissa);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpOrB32, result, vgpr_src(mantissa), result);
  append_vop2(words, kOpOrB32, result, vgpr_src(sign), result);

  append_vop1(words, kOpCvtF32U32, subnormal, vgpr_src(subnormal));
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpMulF32, subnormal, 255, subnormal, subnormal_scale_bits);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpOrB32, subnormal, vgpr_src(sign), subnormal);
  append_vop3(words, kOpCmpEqU32, predicate, vgpr_src(exponent), scalar_positive_inline_u32(0));
  words.push_back(pack_sopp(kSoppWaitAlu, kWaitAluDepctrVaSdst0));
  append_vop3(words, kOpCndmaskB32, result, vgpr_src(result), vgpr_src(subnormal), predicate);
  append_wait_valu_vgpr(words);
}

void append_wmma_f32_k4_fmac_term(std::vector<uint32_t> &words, uint8_t dst_reg, uint8_t src_a,
                                  uint8_t src_b, uint8_t tmp_a, uint8_t tmp_b, uint8_t vaddr,
                                  uint8_t output_reg, uint8_t k, uint8_t exec_save) {
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpAddNcU32 = 37;
  constexpr uint8_t kOpXorB32 = 29;
  constexpr uint16_t kOpFmacF32 = 299;
  constexpr uint16_t kInlineConst1 = 129;
  constexpr uint16_t kInlineConst2 = 130;
  constexpr uint16_t kInlineConst15 = 143;
  constexpr uint16_t kInlineConst16 = 144;
  constexpr uint16_t kInlineConst64 = 192;
  constexpr uint8_t kOpWaitDscnt = 70;

  const uint8_t src_word = static_cast<uint8_t>(k & 1u);
  const bool high_k_group = k >= 2;

  append_set_exec_lo_mask(words, 0xFFFFFFFFu);
  append_lane_id(words, vaddr);
  append_vop2(words, kOpAndB32, vaddr, kInlineConst16, vaddr);
  append_vop2(words, kOpLshlrevB32, vaddr, kInlineConst1, vaddr);
  if (output_reg != 0)
    append_vop2(words, kOpAddNcU32, vaddr, scalar_positive_inline_u32(output_reg * 4u), vaddr);
  if (high_k_group)
    append_vop2(words, kOpXorB32, vaddr, kInlineConst64, vaddr);
  append_wait_valu_vgpr(words);
  auto [a_w0, a_w1] = build_ds_bpermute(tmp_a, vaddr, static_cast<uint8_t>(src_a + src_word));
  words.push_back(a_w0);
  words.push_back(a_w1);

  append_lane_id(words, vaddr);
  append_vop2(words, kOpAndB32, vaddr, kInlineConst15, vaddr);
  append_vop2(words, kOpLshlrevB32, vaddr, kInlineConst2, vaddr);
  if (high_k_group)
    append_vop2(words, kOpXorB32, vaddr, kInlineConst64, vaddr);
  append_wait_valu_vgpr(words);
  auto [b_w0, b_w1] = build_ds_bpermute(tmp_b, vaddr, static_cast<uint8_t>(src_b + src_word));
  words.push_back(b_w0);
  words.push_back(b_w1);
  words.push_back(pack_sopp(kOpWaitDscnt, 0));

  append_restore_exec(words, exec_save);
  append_wait_valu_vgpr(words);
  append_vop3(words, kOpFmacF32, dst_reg, static_cast<uint16_t>(256u + tmp_a),
              static_cast<uint16_t>(256u + tmp_b));
}

void append_wmma_f32_k128_fp8_dot4_term(std::vector<uint32_t> &words, uint8_t dot_op,
                                        uint8_t dst_reg, uint8_t src_a, uint8_t src_b,
                                        uint8_t tmp_a, uint8_t tmp_b, uint8_t vaddr,
                                        uint8_t output_reg, uint8_t k_group, uint8_t exec_save) {
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpAddNcU32 = 37;
  constexpr uint8_t kOpXorB32 = 29;
  constexpr uint16_t kInlineConst1 = 129;
  constexpr uint16_t kInlineConst2 = 130;
  constexpr uint16_t kInlineConst15 = 143;
  constexpr uint16_t kInlineConst16 = 144;
  constexpr uint16_t kInlineConst64 = 192;
  constexpr uint8_t kOpWaitDscnt = 70;

  const uint8_t src_word = static_cast<uint8_t>((k_group & 0x7u) + ((k_group & 0x10u) ? 8u : 0u));
  const bool high_k_group = (k_group & 0x8u) != 0;

  append_set_exec_lo_mask(words, 0xFFFFFFFFu);
  append_lane_id(words, vaddr);
  append_vop2(words, kOpAndB32, vaddr, kInlineConst16, vaddr);
  append_vop2(words, kOpLshlrevB32, vaddr, kInlineConst1, vaddr);
  if (output_reg != 0)
    append_vop2(words, kOpAddNcU32, vaddr, scalar_positive_inline_u32(output_reg * 4u), vaddr);
  if (high_k_group)
    append_vop2(words, kOpXorB32, vaddr, kInlineConst64, vaddr);
  append_wait_valu_vgpr(words);
  auto [a_w0, a_w1] = build_ds_bpermute(tmp_a, vaddr, static_cast<uint8_t>(src_a + src_word));
  words.push_back(a_w0);
  words.push_back(a_w1);

  append_lane_id(words, vaddr);
  append_vop2(words, kOpAndB32, vaddr, kInlineConst15, vaddr);
  append_vop2(words, kOpLshlrevB32, vaddr, kInlineConst2, vaddr);
  if (high_k_group)
    append_vop2(words, kOpXorB32, vaddr, kInlineConst64, vaddr);
  append_wait_valu_vgpr(words);
  auto [b_w0, b_w1] = build_ds_bpermute(tmp_b, vaddr, static_cast<uint8_t>(src_b + src_word));
  words.push_back(b_w0);
  words.push_back(b_w1);
  words.push_back(pack_sopp(kOpWaitDscnt, 0));

  append_restore_exec(words, exec_save);
  append_wait_valu_vgpr(words);
  append_v_dot4_f32_f8(words, dot_op, dst_reg, tmp_a, tmp_b, static_cast<uint16_t>(256u + dst_reg));
}

void append_wmma_f32_k128_fp8_dot4_group(std::vector<uint32_t> &words, uint8_t dot_op,
                                         uint8_t dst_base, uint8_t src_a, uint8_t src_b,
                                         uint8_t tmp_a_base, uint8_t tmp_b, uint8_t vaddr,
                                         uint8_t k_group, uint8_t exec_save) {
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpAddNcU32 = 37;
  constexpr uint8_t kOpXorB32 = 29;
  constexpr uint16_t kInlineConst1 = 129;
  constexpr uint16_t kInlineConst2 = 130;
  constexpr uint16_t kInlineConst15 = 143;
  constexpr uint16_t kInlineConst16 = 144;
  constexpr uint16_t kInlineConst64 = 192;
  constexpr uint8_t kOpWaitDscnt = 70;

  const uint8_t src_word = static_cast<uint8_t>((k_group & 0x7u) + ((k_group & 0x10u) ? 8u : 0u));
  const bool high_k_group = (k_group & 0x8u) != 0;

  // All eight output words use the same B permutation for a given K group.
  // Keep EXEC widened while issuing that B permutation and the eight distinct
  // A permutations, then wait and restore EXEC once for the eight dot4s.  This
  // preserves each accumulator's K-group order while avoiding eight serialized
  // DS waits and EXEC transitions per group.
  append_set_exec_lo_mask(words, 0xFFFFFFFFu);
  append_lane_id(words, vaddr);
  append_vop2(words, kOpAndB32, vaddr, kInlineConst15, vaddr);
  append_vop2(words, kOpLshlrevB32, vaddr, kInlineConst2, vaddr);
  if (high_k_group)
    append_vop2(words, kOpXorB32, vaddr, kInlineConst64, vaddr);
  append_wait_valu_vgpr(words);
  auto [b_w0, b_w1] = build_ds_bpermute(tmp_b, vaddr, static_cast<uint8_t>(src_b + src_word));
  words.push_back(b_w0);
  words.push_back(b_w1);

  for (uint8_t output_reg = 0; output_reg < 8; ++output_reg) {
    append_lane_id(words, vaddr);
    append_vop2(words, kOpAndB32, vaddr, kInlineConst16, vaddr);
    append_vop2(words, kOpLshlrevB32, vaddr, kInlineConst1, vaddr);
    if (output_reg != 0)
      append_vop2(words, kOpAddNcU32, vaddr, scalar_positive_inline_u32(output_reg * 4u), vaddr);
    if (high_k_group)
      append_vop2(words, kOpXorB32, vaddr, kInlineConst64, vaddr);
    append_wait_valu_vgpr(words);
    auto [a_w0, a_w1] = build_ds_bpermute(static_cast<uint8_t>(tmp_a_base + output_reg), vaddr,
                                          static_cast<uint8_t>(src_a + src_word));
    words.push_back(a_w0);
    words.push_back(a_w1);
  }
  words.push_back(pack_sopp(kOpWaitDscnt, 0));

  append_restore_exec(words, exec_save);
  append_wait_valu_vgpr(words);
  for (uint8_t output_reg = 0; output_reg < 8; ++output_reg) {
    const uint8_t dst_reg = static_cast<uint8_t>(dst_base + output_reg);
    append_v_dot4_f32_f8(words, dot_op, dst_reg, static_cast<uint8_t>(tmp_a_base + output_reg),
                         tmp_b, static_cast<uint16_t>(256u + dst_reg));
  }
}

void append_wmma_f32_k128_fp8_scalar_a_group(std::vector<uint32_t> &words, uint8_t dot_op,
                                             uint8_t dst_base, uint8_t src_a, uint8_t src_b,
                                             uint8_t tmp_b, uint8_t vaddr,
                                             const std::array<uint8_t, 8> &scalar_a,
                                             uint8_t k_group, uint8_t exec_save) {
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpXorB32 = 29;
  constexpr uint16_t kInlineConst2 = 130;
  constexpr uint16_t kInlineConst15 = 143;
  constexpr uint16_t kInlineConst64 = 192;
  constexpr uint8_t kOpWaitDscnt = 70;
  constexpr uint8_t kSoppWaitAlu = 8;

  const uint8_t src_word = static_cast<uint8_t>((k_group & 0x7u) + ((k_group & 0x10u) ? 8u : 0u));
  const bool high_k_group = (k_group & 0x8u) != 0;

  // B varies by output column but is common to all eight output registers.
  // Broadcast A through an SGPR separately for the low and high half waves;
  // this reduces the exact dot4 fallback to one DS permutation per K group.
  append_set_exec_lo_mask(words, 0xFFFFFFFFu);
  append_lane_id(words, vaddr);
  append_vop2(words, kOpAndB32, vaddr, kInlineConst15, vaddr);
  append_vop2(words, kOpLshlrevB32, vaddr, kInlineConst2, vaddr);
  if (high_k_group)
    append_vop2(words, kOpXorB32, vaddr, kInlineConst64, vaddr);
  append_wait_valu_vgpr(words);
  auto [b_w0, b_w1] = build_ds_bpermute(tmp_b, vaddr, static_cast<uint8_t>(src_b + src_word));
  words.push_back(b_w0);
  words.push_back(b_w1);
  words.push_back(pack_sopp(kOpWaitDscnt, 0));

  for (uint8_t output_reg = 0; output_reg < 8; ++output_reg) {
    const uint8_t low_a_lane = static_cast<uint8_t>((high_k_group ? 16u : 0u) + output_reg);
    append_v_readlane_b32(words, scalar_a[output_reg], static_cast<uint8_t>(src_a + src_word),
                          low_a_lane);
  }
  words.push_back(pack_sopp(kSoppWaitAlu, kWaitAluDepctrVaSdst0));
  append_set_exec_from_saved_mask(words, exec_save, 0x0000FFFFu);
  for (uint8_t output_reg = 0; output_reg < 8; ++output_reg) {
    const uint8_t dst_reg = static_cast<uint8_t>(dst_base + output_reg);
    append_v_dot4_f32_f8_encoded(words, dot_op, dst_reg, scalar_a[output_reg],
                                 static_cast<uint16_t>(256u + tmp_b),
                                 static_cast<uint16_t>(256u + dst_reg));
  }

  for (uint8_t output_reg = 0; output_reg < 8; ++output_reg) {
    const uint8_t high_a_lane = static_cast<uint8_t>((high_k_group ? 24u : 8u) + output_reg);
    append_v_readlane_b32(words, scalar_a[output_reg], static_cast<uint8_t>(src_a + src_word),
                          high_a_lane);
  }
  words.push_back(pack_sopp(kSoppWaitAlu, kWaitAluDepctrVaSdst0));
  append_set_exec_from_saved_mask(words, exec_save, 0xFFFF0000u);
  for (uint8_t output_reg = 0; output_reg < 8; ++output_reg) {
    const uint8_t dst_reg = static_cast<uint8_t>(dst_base + output_reg);
    append_v_dot4_f32_f8_encoded(words, dot_op, dst_reg, scalar_a[output_reg],
                                 static_cast<uint16_t>(256u + tmp_b),
                                 static_cast<uint16_t>(256u + dst_reg));
  }
  append_restore_exec(words, exec_save);
}

void append_wmma_scale_f32_from_word(std::vector<uint32_t> &words, uint8_t scale_f32,
                                     uint8_t scale_word, uint8_t byte_index, uint8_t scale_format,
                                     uint8_t pred_sgpr, bool inline_zero_is_unity = false) {
  constexpr uint8_t kOpMovB32 = 1;
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpLshrrevB32 = 25;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint16_t kOpCndmaskB32 = 257;
  constexpr uint16_t kOpCmpEqU32 = 74;
  constexpr uint16_t kInlineConst23 = 151;
  constexpr uint8_t kSoppWaitAlu = 8;
  constexpr uint32_t kByteMask = 0xFFu;
  constexpr uint32_t kF32One = 0x3F800000u;
  constexpr uint32_t kF32QuietNaN = 0x7FC00000u;

  if (inline_zero_is_unity) {
    append_vop1(words, kOpMovB32, scale_f32, 255, kF32One);
    return;
  }

  if (byte_index == 0) {
    append_vop1(words, kOpMovB32, scale_f32, static_cast<uint16_t>(256u + scale_word));
  } else {
    append_vop2(words, kOpLshrrevB32, scale_f32,
                scalar_positive_inline_u32(static_cast<uint16_t>(byte_index * 8u)), scale_word);
  }
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpAndB32, scale_f32, 255, scale_f32, kByteMask);
  append_wait_valu_vgpr(words);
  if (scale_format == 2u) {
    constexpr uint16_t kOpCvtF32Fp8 = 492;
    append_vop3_mod(words, kOpCvtF32Fp8, scale_f32, vgpr_src(scale_f32), 0, 0, 0);
    append_wait_valu_vgpr(words);
    return;
  }
  append_vop3(words, kOpCmpEqU32, pred_sgpr, static_cast<uint16_t>(256u + scale_f32), 255, 0,
              kByteMask);
  append_vop2(words, kOpLshlrevB32, scale_f32, kInlineConst23, scale_f32);
  words.push_back(pack_sopp(kSoppWaitAlu, kWaitAluDepctrVaSdst0));
  append_wait_valu_vgpr(words);
  append_vop3(words, kOpCndmaskB32, scale_f32, static_cast<uint16_t>(256u + scale_f32), 255,
              pred_sgpr, kF32QuietNaN);
}

void append_wmma_f32_apply_scales_and_accumulate(
    std::vector<uint32_t> &words, uint8_t dst_reg, uint8_t contribution, uint16_t scale_src_a,
    uint16_t scale_src_b, uint8_t scale_a_tmp, uint8_t scale_b_tmp, uint8_t vaddr,
    uint8_t output_reg, uint8_t scale_byte_a, uint8_t scale_byte_b, uint8_t matrix_a_scale_select,
    uint8_t matrix_b_scale_select, uint8_t matrix_a_scale_fmt, uint8_t matrix_b_scale_fmt,
    uint8_t pred_sgpr, uint8_t exec_save, bool f4_32x16 = false);

uint8_t fp4_k128_src_word(uint8_t k_group) {
  return static_cast<uint8_t>(((k_group >> 2u) * 2u) + (k_group & 0x1u));
}

bool fp4_k128_high_lane_group(uint8_t k_group) { return ((k_group >> 1u) & 0x1u) != 0; }

void append_wmma_f32_k128_fp8_scaled_dot4_term(std::vector<uint32_t> &words, uint8_t dst_reg,
                                               uint8_t src_a, uint8_t src_b, uint16_t scale_src_a,
                                               uint16_t scale_src_b, uint8_t tmp_a, uint8_t tmp_b,
                                               uint8_t tmp_dot, uint8_t vaddr, uint8_t output_reg,
                                               uint8_t k_group, uint8_t matrix_a_scale_select,
                                               uint8_t matrix_b_scale_select, uint8_t pred_sgpr,
                                               uint8_t exec_save) {
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpAddNcU32 = 37;
  constexpr uint8_t kOpXorB32 = 29;
  constexpr uint16_t kInlineConst1 = 129;
  constexpr uint16_t kInlineConst2 = 130;
  constexpr uint16_t kInlineConst15 = 143;
  constexpr uint16_t kInlineConst16 = 144;
  constexpr uint16_t kInlineConst64 = 192;
  constexpr uint8_t kOpWaitDscnt = 70;

  const uint8_t src_word = static_cast<uint8_t>((k_group & 0x7u) + ((k_group & 0x10u) ? 8u : 0u));
  const bool high_k_group = (k_group & 0x8u) != 0;
  const uint8_t scale_byte = static_cast<uint8_t>(k_group / 8u);

  append_set_exec_lo_mask(words, 0xFFFFFFFFu);
  append_lane_id(words, vaddr);
  append_vop2(words, kOpAndB32, vaddr, kInlineConst16, vaddr);
  append_vop2(words, kOpLshlrevB32, vaddr, kInlineConst1, vaddr);
  if (output_reg != 0)
    append_vop2(words, kOpAddNcU32, vaddr, scalar_positive_inline_u32(output_reg * 4u), vaddr);
  if (high_k_group)
    append_vop2(words, kOpXorB32, vaddr, kInlineConst64, vaddr);
  append_wait_valu_vgpr(words);
  auto [a_w0, a_w1] = build_ds_bpermute(tmp_a, vaddr, static_cast<uint8_t>(src_a + src_word));
  words.push_back(a_w0);
  words.push_back(a_w1);

  append_lane_id(words, vaddr);
  append_vop2(words, kOpAndB32, vaddr, kInlineConst15, vaddr);
  append_vop2(words, kOpLshlrevB32, vaddr, kInlineConst2, vaddr);
  if (high_k_group)
    append_vop2(words, kOpXorB32, vaddr, kInlineConst64, vaddr);
  append_wait_valu_vgpr(words);
  auto [b_w0, b_w1] = build_ds_bpermute(tmp_b, vaddr, static_cast<uint8_t>(src_b + src_word));
  words.push_back(b_w0);
  words.push_back(b_w1);
  words.push_back(pack_sopp(kOpWaitDscnt, 0));

  append_restore_exec(words, exec_save);
  append_wait_valu_vgpr(words);
  append_v_dot4_f32_f8(words, /*v_dot4_f32_fp8_fp8=*/38, tmp_dot, tmp_a, tmp_b,
                       scalar_positive_inline_u32(0));

  append_wmma_f32_apply_scales_and_accumulate(words, dst_reg, tmp_dot, scale_src_a, scale_src_b,
                                              tmp_a, tmp_b, vaddr, output_reg, scale_byte,
                                              scale_byte, matrix_a_scale_select,
                                              matrix_b_scale_select, 0, 0, pred_sgpr, exec_save);
}

void append_wmma_f32_apply_scales_and_accumulate(
    std::vector<uint32_t> &words, uint8_t dst_reg, uint8_t contribution, uint16_t scale_src_a,
    uint16_t scale_src_b, uint8_t scale_a_tmp, uint8_t scale_b_tmp, uint8_t vaddr,
    uint8_t output_reg, uint8_t scale_byte_a, uint8_t scale_byte_b, uint8_t matrix_a_scale_select,
    uint8_t matrix_b_scale_select, uint8_t matrix_a_scale_fmt, uint8_t matrix_b_scale_fmt,
    uint8_t pred_sgpr, uint8_t exec_save, bool f4_32x16) {
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpLshrrevB32 = 25;
  constexpr uint8_t kOpAddNcU32 = 37;
  constexpr uint16_t kOpAddF32 = 259;
  constexpr uint16_t kOpMulF32 = 264;
  constexpr uint16_t kInlineConst1 = 129;
  constexpr uint16_t kInlineConst2 = 130;
  constexpr uint16_t kInlineConst15 = 143;
  constexpr uint16_t kInlineConst16 = 144;
  constexpr uint16_t kInlineConst64 = 192;
  constexpr uint8_t kOpWaitDscnt = 70;
  constexpr uint8_t kOpMovB32 = 1;

  append_wait_valu_vgpr(words);
  append_set_exec_lo_mask(words, 0xFFFFFFFFu);
  const bool scale_src_a_inline_zero = scalar_inline_zero_src(scale_src_a);
  const bool scale_src_b_inline_zero = scalar_inline_zero_src(scale_src_b);
  bool waits_on_ds = false;
  if (auto scale_src_a_vgpr = src_vgpr_base_for_run(scale_src_a, 1)) {
    append_lane_id(words, vaddr);
    if (f4_32x16) {
      // row = output_reg + 16*(lane/16). The 32x16 FP4 scale ABI maps row
      // groups 0,1,2,3 to lane groups 0,2,1,3 respectively.
      append_vop2(words, kOpAndB32, vaddr, kInlineConst16, vaddr);
      append_vop2(words, kOpLshrrevB32, vaddr, kInlineConst1, vaddr);
      if ((output_reg & 7u) != 0u) {
        append_vop2(words, kOpAddNcU32, vaddr, scalar_positive_inline_u32(output_reg & 7u), vaddr);
      }
      if (output_reg >= 8u)
        append_vop2(words, kOpAddNcU32, vaddr, kInlineConst16, vaddr);
      append_vop2(words, kOpLshlrevB32, vaddr, kInlineConst2, vaddr);
    } else {
      append_vop2(words, kOpAndB32, vaddr, kInlineConst16, vaddr);
      append_vop2(words, kOpLshlrevB32, vaddr, kInlineConst1, vaddr);
      if (output_reg != 0) {
        append_vop2(words, kOpAddNcU32, vaddr, scalar_positive_inline_u32(output_reg * 4u), vaddr);
      }
      if (matrix_a_scale_select & 0x1u)
        append_vop2(words, kOpAddNcU32, vaddr, kInlineConst64, vaddr);
    }
    append_wait_valu_vgpr(words);
    auto [scale_a_w0, scale_a_w1] = build_ds_bpermute(scale_a_tmp, vaddr, *scale_src_a_vgpr);
    words.push_back(scale_a_w0);
    words.push_back(scale_a_w1);
    waits_on_ds = true;
  } else {
    append_vop1(words, kOpMovB32, scale_a_tmp, scale_src_a);
  }

  if (auto scale_src_b_vgpr = src_vgpr_base_for_run(scale_src_b, 1)) {
    append_lane_id(words, vaddr);
    append_vop2(words, kOpAndB32, vaddr, kInlineConst15, vaddr);
    append_vop2(words, kOpLshlrevB32, vaddr, kInlineConst2, vaddr);
    if (matrix_b_scale_select & 0x1u)
      append_vop2(words, kOpAddNcU32, vaddr, kInlineConst64, vaddr);
    append_wait_valu_vgpr(words);
    auto [scale_b_w0, scale_b_w1] = build_ds_bpermute(scale_b_tmp, vaddr, *scale_src_b_vgpr);
    words.push_back(scale_b_w0);
    words.push_back(scale_b_w1);
    waits_on_ds = true;
  } else {
    append_vop1(words, kOpMovB32, scale_b_tmp, scale_src_b);
  }
  if (waits_on_ds)
    words.push_back(pack_sopp(kOpWaitDscnt, 0));

  append_restore_exec(words, exec_save);
  append_wait_valu_vgpr(words);
  append_wmma_scale_f32_from_word(words, scale_a_tmp, scale_a_tmp, scale_byte_a, matrix_a_scale_fmt,
                                  pred_sgpr, scale_src_a_inline_zero);
  append_wmma_scale_f32_from_word(words, scale_b_tmp, scale_b_tmp, scale_byte_b, matrix_b_scale_fmt,
                                  pred_sgpr, scale_src_b_inline_zero);
  append_vop3(words, kOpMulF32, vaddr, static_cast<uint16_t>(256u + contribution),
              static_cast<uint16_t>(256u + scale_a_tmp));
  append_wait_valu_vgpr(words);
  append_vop3(words, kOpMulF32, vaddr, static_cast<uint16_t>(256u + vaddr),
              static_cast<uint16_t>(256u + scale_b_tmp));
  append_wait_valu_vgpr(words);
  append_vop3(words, kOpAddF32, dst_reg, static_cast<uint16_t>(256u + dst_reg),
              static_cast<uint16_t>(256u + vaddr));
}

void append_wmma_f32_k128_fp4_dot8_term(std::vector<uint32_t> &words, uint8_t dst_reg,
                                        uint8_t src_a, uint8_t src_b, uint8_t tmp_a_raw,
                                        uint8_t tmp_b_raw, uint8_t tmp_a_pos, uint8_t tmp_a_neg,
                                        uint8_t tmp_b_pos, uint8_t tmp_b_neg, uint8_t tmp_dot,
                                        uint8_t tmp_cross, uint8_t tmp_f32, uint8_t vaddr,
                                        uint8_t output_reg, uint8_t k_group, uint8_t exec_save) {
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpAddNcU32 = 37;
  constexpr uint8_t kOpSubNcU32 = 38;
  constexpr uint8_t kOpXorB32 = 29;
  constexpr uint8_t kOpCvtF32I32 = 5;
  constexpr uint16_t kOpFmacF32 = 299;
  constexpr uint16_t kInlineConst1 = 129;
  constexpr uint16_t kInlineConst2 = 130;
  constexpr uint16_t kInlineConst15 = 143;
  constexpr uint16_t kInlineConst16 = 144;
  constexpr uint16_t kInlineConst64 = 192;
  constexpr uint8_t kOpWaitDscnt = 70;
  constexpr uint32_t kF32OneQuarter = 0x3E800000u;

  const uint8_t src_word = fp4_k128_src_word(k_group);
  const bool high_k_group = fp4_k128_high_lane_group(k_group);

  append_set_exec_lo_mask(words, 0xFFFFFFFFu);
  append_lane_id(words, vaddr);
  append_vop2(words, kOpAndB32, vaddr, kInlineConst16, vaddr);
  append_vop2(words, kOpLshlrevB32, vaddr, kInlineConst1, vaddr);
  if (output_reg != 0)
    append_vop2(words, kOpAddNcU32, vaddr, scalar_positive_inline_u32(output_reg * 4u), vaddr);
  if (high_k_group)
    append_vop2(words, kOpXorB32, vaddr, kInlineConst64, vaddr);
  append_wait_valu_vgpr(words);
  auto [a_w0, a_w1] = build_ds_bpermute(tmp_a_raw, vaddr, static_cast<uint8_t>(src_a + src_word));
  words.push_back(a_w0);
  words.push_back(a_w1);

  append_lane_id(words, vaddr);
  append_vop2(words, kOpAndB32, vaddr, kInlineConst15, vaddr);
  append_vop2(words, kOpLshlrevB32, vaddr, kInlineConst2, vaddr);
  if (high_k_group)
    append_vop2(words, kOpXorB32, vaddr, kInlineConst64, vaddr);
  append_wait_valu_vgpr(words);
  auto [b_w0, b_w1] = build_ds_bpermute(tmp_b_raw, vaddr, static_cast<uint8_t>(src_b + src_word));
  words.push_back(b_w0);
  words.push_back(b_w1);
  words.push_back(pack_sopp(kOpWaitDscnt, 0));

  append_restore_exec(words, exec_save);
  append_wait_valu_vgpr(words);

  append_fp4_scaled_mag2_pos_neg(words, tmp_a_raw, tmp_a_pos, tmp_a_neg, tmp_dot, tmp_cross,
                                 tmp_f32);
  append_fp4_scaled_mag2_pos_neg(words, tmp_b_raw, tmp_b_pos, tmp_b_neg, tmp_dot, tmp_cross,
                                 tmp_f32);

  append_v_dot8_u32_u4(words, tmp_dot, tmp_a_pos, tmp_b_pos, scalar_positive_inline_u32(0));
  append_v_dot8_u32_u4(words, tmp_dot, tmp_a_neg, tmp_b_neg, static_cast<uint16_t>(256u + tmp_dot));
  append_v_dot8_u32_u4(words, tmp_cross, tmp_a_pos, tmp_b_neg, scalar_positive_inline_u32(0));
  append_vop2(words, kOpSubNcU32, tmp_dot, static_cast<uint16_t>(256u + tmp_dot), tmp_cross);
  append_v_dot8_u32_u4(words, tmp_cross, tmp_a_neg, tmp_b_pos, scalar_positive_inline_u32(0));
  append_vop2(words, kOpSubNcU32, tmp_dot, static_cast<uint16_t>(256u + tmp_dot), tmp_cross);
  append_vop1(words, kOpCvtF32I32, tmp_f32, static_cast<uint16_t>(256u + tmp_dot));
  append_vop3(words, kOpFmacF32, dst_reg, static_cast<uint16_t>(256u + tmp_f32), 255, 0,
              kF32OneQuarter);
}

void append_wmma_f32_k128_fp4_scaled_dot8_term(
    std::vector<uint32_t> &words, uint8_t dst_reg, uint8_t src_a, uint8_t src_b,
    uint16_t scale_src_a, uint16_t scale_src_b, uint8_t tmp_a_raw, uint8_t tmp_b_raw,
    uint8_t tmp_a_pos, uint8_t tmp_a_neg, uint8_t tmp_b_pos, uint8_t tmp_b_neg, uint8_t tmp_dot,
    uint8_t tmp_cross, uint8_t tmp_f32, uint8_t vaddr, uint8_t output_reg, uint8_t k_group,
    uint8_t matrix_a_scale_select, uint8_t matrix_b_scale_select, uint8_t pred_sgpr,
    uint8_t exec_save) {
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpAddNcU32 = 37;
  constexpr uint8_t kOpSubNcU32 = 38;
  constexpr uint8_t kOpXorB32 = 29;
  constexpr uint8_t kOpCvtF32I32 = 5;
  constexpr uint16_t kOpMulF32 = 264;
  constexpr uint16_t kInlineConst1 = 129;
  constexpr uint16_t kInlineConst2 = 130;
  constexpr uint16_t kInlineConst15 = 143;
  constexpr uint16_t kInlineConst16 = 144;
  constexpr uint16_t kInlineConst64 = 192;
  constexpr uint8_t kOpWaitDscnt = 70;
  constexpr uint32_t kF32OneQuarter = 0x3E800000u;

  const uint8_t src_word = fp4_k128_src_word(k_group);
  const bool high_k_group = fp4_k128_high_lane_group(k_group);
  const uint8_t scale_byte_a = static_cast<uint8_t>(k_group / 4u);
  const uint8_t scale_byte_b =
      static_cast<uint8_t>(((k_group / 2u) & 0x1u) | (((k_group / 8u) & 0x1u) << 1u));

  append_set_exec_lo_mask(words, 0xFFFFFFFFu);
  append_lane_id(words, vaddr);
  append_vop2(words, kOpAndB32, vaddr, kInlineConst16, vaddr);
  append_vop2(words, kOpLshlrevB32, vaddr, kInlineConst1, vaddr);
  if (output_reg != 0)
    append_vop2(words, kOpAddNcU32, vaddr, scalar_positive_inline_u32(output_reg * 4u), vaddr);
  if (high_k_group)
    append_vop2(words, kOpXorB32, vaddr, kInlineConst64, vaddr);
  append_wait_valu_vgpr(words);
  auto [a_w0, a_w1] = build_ds_bpermute(tmp_a_raw, vaddr, static_cast<uint8_t>(src_a + src_word));
  words.push_back(a_w0);
  words.push_back(a_w1);

  append_lane_id(words, vaddr);
  append_vop2(words, kOpAndB32, vaddr, kInlineConst15, vaddr);
  append_vop2(words, kOpLshlrevB32, vaddr, kInlineConst2, vaddr);
  if (high_k_group)
    append_vop2(words, kOpXorB32, vaddr, kInlineConst64, vaddr);
  append_wait_valu_vgpr(words);
  auto [b_w0, b_w1] = build_ds_bpermute(tmp_b_raw, vaddr, static_cast<uint8_t>(src_b + src_word));
  words.push_back(b_w0);
  words.push_back(b_w1);
  words.push_back(pack_sopp(kOpWaitDscnt, 0));

  append_restore_exec(words, exec_save);
  append_wait_valu_vgpr(words);

  append_fp4_scaled_mag2_pos_neg(words, tmp_a_raw, tmp_a_pos, tmp_a_neg, tmp_dot, tmp_cross,
                                 tmp_f32);
  append_fp4_scaled_mag2_pos_neg(words, tmp_b_raw, tmp_b_pos, tmp_b_neg, tmp_dot, tmp_cross,
                                 tmp_f32);

  append_v_dot8_u32_u4(words, tmp_dot, tmp_a_pos, tmp_b_pos, scalar_positive_inline_u32(0));
  append_v_dot8_u32_u4(words, tmp_dot, tmp_a_neg, tmp_b_neg, static_cast<uint16_t>(256u + tmp_dot));
  append_v_dot8_u32_u4(words, tmp_cross, tmp_a_pos, tmp_b_neg, scalar_positive_inline_u32(0));
  append_vop2(words, kOpSubNcU32, tmp_dot, static_cast<uint16_t>(256u + tmp_dot), tmp_cross);
  append_v_dot8_u32_u4(words, tmp_cross, tmp_a_neg, tmp_b_pos, scalar_positive_inline_u32(0));
  append_vop2(words, kOpSubNcU32, tmp_dot, static_cast<uint16_t>(256u + tmp_dot), tmp_cross);
  append_vop1(words, kOpCvtF32I32, tmp_f32, static_cast<uint16_t>(256u + tmp_dot));
  append_vop3(words, kOpMulF32, tmp_f32, static_cast<uint16_t>(256u + tmp_f32), 255, 0,
              kF32OneQuarter);

  append_wmma_f32_apply_scales_and_accumulate(words, dst_reg, tmp_f32, scale_src_a, scale_src_b,
                                              tmp_a_pos, tmp_b_pos, vaddr, output_reg, scale_byte_a,
                                              scale_byte_b, matrix_a_scale_select,
                                              matrix_b_scale_select, 0, 0, pred_sgpr, exec_save);
}

void append_v_mov_b32_run(std::vector<uint32_t> &words, uint8_t vdst, uint16_t src, uint8_t count) {
  constexpr uint8_t kOpMovB32 = 1;
  for (uint8_t word = 0; word < count; ++word)
    append_vop1(words, kOpMovB32, static_cast<uint8_t>(vdst + word),
                static_cast<uint16_t>(src + word));
}

void append_v_mov_b32_broadcast(std::vector<uint32_t> &words, uint8_t vdst, uint16_t src,
                                uint8_t count) {
  constexpr uint8_t kOpMovB32 = 1;
  for (uint8_t word = 0; word < count; ++word)
    append_vop1(words, kOpMovB32, static_cast<uint8_t>(vdst + word), src);
}

constexpr uint16_t k16BitK32WmmaScratchSearchBase = 208;

std::vector<uint32_t> expand_v_wmma_f32_16x16x32_f16(const Instruction &inst, uint32_t, uint64_t,
                                                     const LivenessAnalysis &liveness,
                                                     const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3pMachineInst))
    return {};

  gfx1250::Vop3pMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.vdst > 248 || (src.neg_hi & ~0x4u) != 0 || (src.opsel & ~0x4u) != 0 || src.clamp != 0 ||
      (src.neg & ~0x4u) != 0)
    return {};
  const uint8_t c_modifier =
      static_cast<uint8_t>(((src.neg >> 2u) & 0x1u) | (((src.neg_hi >> 2u) & 0x1u) << 1u));

  const auto src0_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src0), 8);
  const auto src1_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src1), 8);
  if (!src0_base || !src1_base)
    return {};

  std::optional<uint32_t> literal_word;
  if (src.src2 == 255) {
    literal_word = simm32_literal_word(inst, 2);
    if (!literal_word)
      return {};
  } else if (src.src2 == 254) {
    return {};
  } else if (src.src2 >= 256 && !src_vgpr_base_for_run(static_cast<uint16_t>(src.src2), 8)) {
    return {};
  }

  std::vector<uint8_t> avoid;
  add_avoid_vgpr_run(avoid, static_cast<uint8_t>(src.vdst), 8);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src0), 8);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src1), 8);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src2), 8);

  const bool accumulate_in_vdst =
      (src.vdst % 8) == 0 &&
      !dst_overlaps_wmma_ab_sources(static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src0),
                                    static_cast<uint16_t>(src.src1));
  uint8_t tmp_acc = static_cast<uint8_t>(src.vdst);
  if (!accumulate_in_vdst) {
    auto tmp_base_opt = find_aligned_free_vgpr_run_avoiding(inst, liveness, 8, 8, avoid,
                                                            k16BitK32WmmaScratchSearchBase);
    if (!tmp_base_opt || *tmp_base_opt > 248)
      return {};
    tmp_acc = static_cast<uint8_t>(*tmp_base_opt);
  }

  constexpr uint8_t kSoppWaitKmcnt = 0x47;

  std::vector<uint32_t> words;
  words.reserve(32);

  uint16_t first_acc = static_cast<uint16_t>(src.src2);
  bool wait_for_acc_copy = false;
  if (src.src2 == scalar_positive_inline_u32(0)) {
    append_v_mov_b32_broadcast(words, tmp_acc, scalar_positive_inline_u32(0), 8);
    first_acc = static_cast<uint16_t>(256u + tmp_acc);
    wait_for_acc_copy = true;
  } else if (src.src2 >= 256 && src.src2 != static_cast<uint16_t>(256u + tmp_acc)) {
    append_v_mov_b32_run(words, tmp_acc, static_cast<uint16_t>(src.src2), 8);
    first_acc = static_cast<uint16_t>(256u + tmp_acc);
    wait_for_acc_copy = true;
  }
  if (wait_for_acc_copy)
    append_wait_valu_vgpr(words);

  append_wmma_f32_16x16x16_f16(words, tmp_acc, *src0_base, *src1_base, first_acc, literal_word,
                               c_modifier);
  words.push_back(pack_sopp(kSoppWaitKmcnt, 0));

  append_wmma_f32_16x16x16_f16(words, tmp_acc, static_cast<uint8_t>(*src0_base + 4u),
                               static_cast<uint8_t>(*src1_base + 4u),
                               static_cast<uint16_t>(256u + tmp_acc));
  words.push_back(pack_sopp(kSoppWaitKmcnt, 0));
  if (!accumulate_in_vdst) {
    append_wait_valu_vgpr(words);
    append_v_mov_b32_run(words, static_cast<uint8_t>(src.vdst),
                         static_cast<uint16_t>(256u + tmp_acc), 8);
  }
  return words;
}

std::vector<uint32_t> expand_v_wmma_f32_16x16x32_bf16(const Instruction &inst, uint32_t, uint64_t,
                                                      const LivenessAnalysis &liveness,
                                                      const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3pMachineInst))
    return {};

  gfx1250::Vop3pMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.vdst > 248 || (src.neg_hi & ~0x4u) != 0 || (src.opsel & ~0x4u) != 0 || src.clamp != 0 ||
      (src.neg & ~0x4u) != 0)
    return {};
  const uint8_t c_modifier =
      static_cast<uint8_t>(((src.neg >> 2u) & 0x1u) | (((src.neg_hi >> 2u) & 0x1u) << 1u));

  const auto src0_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src0), 8);
  const auto src1_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src1), 8);
  if (!src0_base || !src1_base)
    return {};

  std::optional<uint32_t> literal_word;
  if (src.src2 == 255) {
    literal_word = simm32_literal_word(inst, 2);
    if (!literal_word)
      return {};
  } else if (src.src2 == 254) {
    return {};
  } else if (src.src2 >= 256 && !src_vgpr_base_for_run(static_cast<uint16_t>(src.src2), 8)) {
    return {};
  }

  std::vector<uint8_t> avoid;
  add_avoid_vgpr_run(avoid, static_cast<uint8_t>(src.vdst), 8);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src0), 8);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src1), 8);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src2), 8);

  const bool accumulate_in_vdst =
      (src.vdst % 8) == 0 &&
      !dst_overlaps_wmma_ab_sources(static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src0),
                                    static_cast<uint16_t>(src.src1));
  uint8_t tmp_acc = static_cast<uint8_t>(src.vdst);
  if (!accumulate_in_vdst) {
    auto tmp_base_opt = find_aligned_free_vgpr_run_avoiding(inst, liveness, 8, 8, avoid,
                                                            k16BitK32WmmaScratchSearchBase);
    if (!tmp_base_opt || *tmp_base_opt > 248)
      return {};
    tmp_acc = static_cast<uint8_t>(*tmp_base_opt);
  }

  constexpr uint8_t kSoppWaitKmcnt = 0x47;

  std::vector<uint32_t> words;
  words.reserve(32);

  uint16_t first_acc = static_cast<uint16_t>(src.src2);
  bool wait_for_acc_copy = false;
  if (src.src2 == scalar_positive_inline_u32(0)) {
    append_v_mov_b32_broadcast(words, tmp_acc, scalar_positive_inline_u32(0), 8);
    first_acc = static_cast<uint16_t>(256u + tmp_acc);
    wait_for_acc_copy = true;
  } else if (src.src2 >= 256 && src.src2 != static_cast<uint16_t>(256u + tmp_acc)) {
    append_v_mov_b32_run(words, tmp_acc, static_cast<uint16_t>(src.src2), 8);
    first_acc = static_cast<uint16_t>(256u + tmp_acc);
    wait_for_acc_copy = true;
  }
  if (wait_for_acc_copy)
    append_wait_valu_vgpr(words);

  append_wmma_f32_16x16x16_bf16(words, tmp_acc, *src0_base, *src1_base, first_acc, literal_word,
                                c_modifier);
  words.push_back(pack_sopp(kSoppWaitKmcnt, 0));

  append_wmma_f32_16x16x16_bf16(words, tmp_acc, static_cast<uint8_t>(*src0_base + 4u),
                                static_cast<uint8_t>(*src1_base + 4u),
                                static_cast<uint16_t>(256u + tmp_acc));
  words.push_back(pack_sopp(kSoppWaitKmcnt, 0));
  if (!accumulate_in_vdst) {
    append_wait_valu_vgpr(words);
    append_v_mov_b32_run(words, static_cast<uint8_t>(src.vdst),
                         static_cast<uint16_t>(256u + tmp_acc), 8);
  }
  return words;
}

std::vector<uint32_t> expand_v_wmma_packed16_16x16x32(const Instruction &inst,
                                                      const LivenessAnalysis &liveness,
                                                      uint8_t target_op) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3pMachineInst))
    return {};

  gfx1250::Vop3pMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  // RDNA4's packed-output K16 forms do not expose gfx1250's C modifier.
  if (src.vdst > 252 || src.neg_hi != 0 || (src.opsel & ~0x4u) != 0 || src.clamp != 0 ||
      src.neg != 0)
    return {};

  const auto src0_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src0), 8);
  const auto src1_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src1), 8);
  if (!src0_base || !src1_base)
    return {};

  std::optional<uint32_t> literal_word;
  if (src.src2 == 255) {
    literal_word = simm32_literal_word(inst, 2);
    if (!literal_word)
      return {};
  } else if (src.src2 == 254) {
    return {};
  } else if (src.src2 >= 256 && !src_vgpr_base_for_run(static_cast<uint16_t>(src.src2), 4)) {
    return {};
  }

  std::vector<uint8_t> avoid;
  add_avoid_vgpr_run(avoid, static_cast<uint8_t>(src.vdst), 4);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src0), 8);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src1), 8);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src2), 4);

  const auto src2_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src2), 4);
  const bool c_can_be_staged_in_vdst =
      !src2_base || *src2_base == src.vdst ||
      !overlaps_vgpr_runs(static_cast<uint16_t>(src.vdst), 4, *src2_base, 4);
  const bool accumulate_in_vdst =
      (src.vdst % 4) == 0 && c_can_be_staged_in_vdst &&
      !overlaps_vgpr_runs(static_cast<uint16_t>(src.vdst), 4, *src0_base, 8) &&
      !overlaps_vgpr_runs(static_cast<uint16_t>(src.vdst), 4, *src1_base, 8);
  uint8_t tmp_acc = static_cast<uint8_t>(src.vdst);
  if (!accumulate_in_vdst) {
    const auto tmp_base = find_aligned_free_vgpr_run_avoiding(inst, liveness, 4, 4, avoid,
                                                              k16BitK32WmmaScratchSearchBase);
    if (!tmp_base || *tmp_base > 252)
      return {};
    tmp_acc = static_cast<uint8_t>(*tmp_base);
  }

  constexpr uint8_t kSoppWaitKmcnt = 0x47;
  std::vector<uint32_t> words;
  words.reserve(24);

  uint16_t first_acc = static_cast<uint16_t>(src.src2);
  bool wait_for_acc_copy = false;
  if (src.src2 == scalar_positive_inline_u32(0)) {
    append_v_mov_b32_broadcast(words, tmp_acc, scalar_positive_inline_u32(0), 4);
    first_acc = static_cast<uint16_t>(256u + tmp_acc);
    wait_for_acc_copy = true;
  } else if (src2_base && *src2_base != tmp_acc) {
    append_v_mov_b32_run(words, tmp_acc, static_cast<uint16_t>(src.src2), 4);
    first_acc = static_cast<uint16_t>(256u + tmp_acc);
    wait_for_acc_copy = true;
  }
  if (wait_for_acc_copy)
    append_wait_valu_vgpr(words);

  append_wmma_packed16_16x16x16(words, target_op, tmp_acc, *src0_base, *src1_base, first_acc,
                                literal_word);
  words.push_back(pack_sopp(kSoppWaitKmcnt, 0));
  append_wmma_packed16_16x16x16(words, target_op, tmp_acc, static_cast<uint8_t>(*src0_base + 4u),
                                static_cast<uint8_t>(*src1_base + 4u),
                                static_cast<uint16_t>(256u + tmp_acc));
  words.push_back(pack_sopp(kSoppWaitKmcnt, 0));
  if (!accumulate_in_vdst) {
    append_wait_valu_vgpr(words);
    append_v_mov_b32_run(words, static_cast<uint8_t>(src.vdst),
                         static_cast<uint16_t>(256u + tmp_acc), 4);
  }
  return words;
}

std::vector<uint32_t> expand_v_wmma_f16_16x16x32_f16(const Instruction &inst, uint32_t, uint64_t,
                                                     const LivenessAnalysis &liveness,
                                                     const LaneLayout *, const LaneLayout *) {
  constexpr uint8_t kOpWmmaF16_16x16x16_F16 = 66;
  return expand_v_wmma_packed16_16x16x32(inst, liveness, kOpWmmaF16_16x16x16_F16);
}

std::vector<uint32_t> expand_v_wmma_bf16_16x16x32_bf16(const Instruction &inst, uint32_t, uint64_t,
                                                       const LivenessAnalysis &liveness,
                                                       const LaneLayout *, const LaneLayout *) {
  constexpr uint8_t kOpWmmaBf16_16x16x16_Bf16 = 67;
  return expand_v_wmma_packed16_16x16x32(inst, liveness, kOpWmmaBf16_16x16x16_Bf16);
}

std::vector<uint32_t> expand_v_wmma_bf16f32_16x16x32_bf16(const Instruction &inst, uint32_t,
                                                          uint64_t,
                                                          const LivenessAnalysis &liveness,
                                                          const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3pMachineInst))
    return {};

  gfx1250::Vop3pMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.vdst > 252 || (src.neg_hi & ~0x4u) != 0 || (src.opsel & ~0x4u) != 0 || src.clamp != 0 ||
      (src.neg & ~0x4u) != 0)
    return {};
  const uint8_t c_modifier =
      static_cast<uint8_t>(((src.neg >> 2u) & 0x1u) | (((src.neg_hi >> 2u) & 0x1u) << 1u));

  const auto src0_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src0), 8);
  const auto src1_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src1), 8);
  if (!src0_base || !src1_base)
    return {};

  std::optional<uint32_t> literal_word;
  if (src.src2 == 255) {
    literal_word = simm32_literal_word(inst, 2);
    if (!literal_word)
      return {};
  } else if (src.src2 == 254) {
    return {};
  } else if (src.src2 >= 256 && !src_vgpr_base_for_run(static_cast<uint16_t>(src.src2), 8)) {
    return {};
  }

  std::vector<uint8_t> avoid;
  add_avoid_vgpr_run(avoid, static_cast<uint8_t>(src.vdst), 4);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src0), 8);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src1), 8);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src2), 8);

  constexpr uint16_t kTmpCount = 10;
  const auto tmp_base = find_aligned_free_vgpr_run_avoiding(inst, liveness, kTmpCount, 8, avoid,
                                                            k16BitK32WmmaScratchSearchBase);
  if (!tmp_base || *tmp_base > 246)
    return {};
  const uint8_t tmp_acc = static_cast<uint8_t>(*tmp_base);
  const uint8_t round_tmp = static_cast<uint8_t>(*tmp_base + 8u);
  const uint8_t pack_hi = static_cast<uint8_t>(*tmp_base + 9u);

  constexpr uint8_t kSoppWaitKmcnt = 0x47;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpOrB32 = 28;
  std::vector<uint32_t> words;
  words.reserve(80);

  uint16_t first_acc = static_cast<uint16_t>(src.src2);
  bool wait_for_acc_copy = false;
  if (src.src2 == scalar_positive_inline_u32(0)) {
    append_v_mov_b32_broadcast(words, tmp_acc, scalar_positive_inline_u32(0), 8);
    first_acc = static_cast<uint16_t>(256u + tmp_acc);
    wait_for_acc_copy = true;
  } else if (src.src2 >= 256) {
    append_v_mov_b32_run(words, tmp_acc, static_cast<uint16_t>(src.src2), 8);
    first_acc = static_cast<uint16_t>(256u + tmp_acc);
    wait_for_acc_copy = true;
  }
  if (wait_for_acc_copy)
    append_wait_valu_vgpr(words);

  append_wmma_f32_16x16x16_bf16(words, tmp_acc, *src0_base, *src1_base, first_acc, literal_word,
                                c_modifier);
  words.push_back(pack_sopp(kSoppWaitKmcnt, 0));
  append_wmma_f32_16x16x16_bf16(words, tmp_acc, static_cast<uint8_t>(*src0_base + 4u),
                                static_cast<uint8_t>(*src1_base + 4u),
                                static_cast<uint16_t>(256u + tmp_acc));
  words.push_back(pack_sopp(kSoppWaitKmcnt, 0));
  append_wait_valu_vgpr(words);

  for (uint8_t word = 0; word < 4; ++word) {
    const uint8_t dst = static_cast<uint8_t>(src.vdst + word);
    append_f32_to_bf16_rne(words, dst, round_tmp, static_cast<uint8_t>(tmp_acc + 2u * word));
    append_f32_to_bf16_rne(words, pack_hi, round_tmp,
                           static_cast<uint8_t>(tmp_acc + 2u * word + 1u));
    append_vop2(words, kOpLshlrevB32, pack_hi, scalar_positive_inline_u32(16), pack_hi);
    append_vop2(words, kOpOrB32, dst, vgpr_src(dst), pack_hi);
  }
  return words;
}

std::vector<uint32_t> expand_v_wmma_f32_16x16x4_f32(const Instruction &inst, uint32_t, uint64_t,
                                                    const LivenessAnalysis &liveness,
                                                    const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3pMachineInst))
    return {};

  gfx1250::Vop3pMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.vdst > 248 || src.neg_hi != 0 || (src.opsel & ~0x4u) != 0 || src.clamp != 0 ||
      src.neg != 0)
    return {};

  const auto src0_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src0), 2);
  const auto src1_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src1), 2);
  if (!src0_base || !src1_base)
    return {};

  if (src.src2 == 255 || src.src2 == 254)
    return {};
  const bool zero_acc = src.src2 == scalar_positive_inline_u32(0);
  const bool accumulate_in_vdst =
      src.src2 == static_cast<uint16_t>(256u + static_cast<uint16_t>(src.vdst));
  const auto src2_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src2), 8);
  if (!zero_acc && !accumulate_in_vdst && !src2_base)
    return {};

  std::vector<uint8_t> avoid;
  add_avoid_vgpr_run(avoid, static_cast<uint8_t>(src.vdst), 8);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src0), 2);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src1), 2);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src2), 8);

  auto tmp_a_opt = find_free_vgpr_run_avoiding(inst, liveness, 1, avoid);
  if (!tmp_a_opt || *tmp_a_opt > 255)
    return {};
  const uint8_t tmp_a = static_cast<uint8_t>(*tmp_a_opt);
  add_avoid_vgpr(avoid, tmp_a);

  auto tmp_b_opt = find_free_vgpr_run_avoiding(inst, liveness, 1, avoid);
  if (!tmp_b_opt || *tmp_b_opt > 255)
    return {};
  const uint8_t tmp_b = static_cast<uint8_t>(*tmp_b_opt);
  add_avoid_vgpr(avoid, tmp_b);

  auto vaddr_opt = find_free_vgpr_run_avoiding(inst, liveness, 1, avoid);
  if (!vaddr_opt || *vaddr_opt > 255)
    return {};
  const uint8_t vaddr = static_cast<uint8_t>(*vaddr_opt);
  add_avoid_vgpr(avoid, vaddr);

  auto exec_save_opt = liveness.find_free_sgpr_pair(&inst);
  if (!exec_save_opt || *exec_save_opt > 124)
    return {};
  const uint8_t exec_save = static_cast<uint8_t>(*exec_save_opt);

  constexpr uint8_t kOpMovB32 = 1;

  std::vector<uint32_t> words;
  words.reserve(520);
  append_save_exec(words, exec_save);

  if (zero_acc) {
    for (uint8_t reg = 0; reg < 8; ++reg) {
      append_vop1(words, kOpMovB32, static_cast<uint8_t>(src.vdst + reg),
                  scalar_positive_inline_u32(0));
    }
  } else if (!accumulate_in_vdst) {
    append_v_mov_b32_run(words, static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src2), 8);
  }

  for (uint8_t reg = 0; reg < 8; ++reg) {
    const uint8_t dst_reg = static_cast<uint8_t>(src.vdst + reg);
    for (uint8_t k = 0; k < 4; ++k) {
      append_wmma_f32_k4_fmac_term(words, dst_reg, *src0_base, *src1_base, tmp_a, tmp_b, vaddr, reg,
                                   k, exec_save);
    }
  }

  return words;
}

std::vector<uint32_t> expand_v_wmma_f32_16x16x128_f8f6f4_fp4_fp4(const Instruction &inst, uint32_t,
                                                                 uint64_t,
                                                                 const LivenessAnalysis &liveness,
                                                                 const LaneLayout *,
                                                                 const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3pMachineInst))
    return {};

  gfx1250::Vop3pMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  const uint32_t matrix_a_fmt = src.opsel;
  const uint32_t matrix_b_fmt = (src.pad_14 << 2u) | src.opsel_hi;
  if (src.vdst > 248 || src.neg_hi != 0 || src.clamp != 0 || src.neg != 0 || matrix_a_fmt != 4 ||
      matrix_b_fmt != 4)
    return {};

  const auto src0_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src0), 8);
  const auto src1_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src1), 8);
  if (!src0_base || !src1_base)
    return {};

  if (dst_overlaps_wmma_ab_sources(static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src0),
                                   static_cast<uint16_t>(src.src1), 8))
    return {};

  constexpr uint8_t kOpMovB32 = 1;
  const bool zero_acc = src.src2 == scalar_positive_inline_u32(0);
  const bool accumulate_in_vdst =
      src.src2 == static_cast<uint16_t>(256u + static_cast<uint16_t>(src.vdst));
  std::optional<uint32_t> literal_word;
  std::optional<uint8_t> src2_base;
  if (src.src2 == 255) {
    literal_word = simm32_literal_word(inst, 2);
    if (!literal_word)
      return {};
  } else if (src.src2 == 254) {
    return {};
  } else if (!zero_acc && !accumulate_in_vdst) {
    src2_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src2), 8);
    if (!src2_base || overlaps_vgpr_runs(static_cast<uint16_t>(src.vdst), 8, *src2_base, 8))
      return {};
  }

  std::vector<uint8_t> avoid;
  add_avoid_vgpr_run(avoid, static_cast<uint8_t>(src.vdst), 8);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src0), 8);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src1), 8);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src2), 8);

  auto tmp_base_opt = find_free_vgpr_run_avoiding(inst, liveness, 10, avoid);
  if (!tmp_base_opt || *tmp_base_opt > 246)
    return {};
  const uint8_t tmp_a_raw = static_cast<uint8_t>(*tmp_base_opt);
  const uint8_t tmp_b_raw = static_cast<uint8_t>(tmp_a_raw + 1u);
  const uint8_t tmp_a_pos = static_cast<uint8_t>(tmp_a_raw + 2u);
  const uint8_t tmp_a_neg = static_cast<uint8_t>(tmp_a_raw + 3u);
  const uint8_t tmp_b_pos = static_cast<uint8_t>(tmp_a_raw + 4u);
  const uint8_t tmp_b_neg = static_cast<uint8_t>(tmp_a_raw + 5u);
  const uint8_t tmp_dot = static_cast<uint8_t>(tmp_a_raw + 6u);
  const uint8_t tmp_cross = static_cast<uint8_t>(tmp_a_raw + 7u);
  const uint8_t tmp_f32 = static_cast<uint8_t>(tmp_a_raw + 8u);
  const uint8_t vaddr = static_cast<uint8_t>(tmp_a_raw + 9u);

  auto exec_save_opt = liveness.find_free_sgpr_pair(&inst);
  if (!exec_save_opt || *exec_save_opt > 124)
    return {};
  const uint8_t exec_save = static_cast<uint8_t>(*exec_save_opt);

  std::vector<uint32_t> words;
  words.reserve(16384);
  append_save_exec(words, exec_save);

  if (zero_acc) {
    for (uint8_t reg = 0; reg < 8; ++reg) {
      append_vop1(words, kOpMovB32, static_cast<uint8_t>(src.vdst + reg),
                  scalar_positive_inline_u32(0));
    }
  } else if (literal_word) {
    for (uint8_t reg = 0; reg < 8; ++reg) {
      append_vop1(words, kOpMovB32, static_cast<uint8_t>(src.vdst + reg), 255, literal_word);
    }
  } else if (!accumulate_in_vdst) {
    append_v_mov_b32_run(words, static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src2), 8);
  }

  for (uint8_t reg = 0; reg < 8; ++reg) {
    const uint8_t dst_reg = static_cast<uint8_t>(src.vdst + reg);
    for (uint8_t k_group = 0; k_group < 16; ++k_group) {
      append_wmma_f32_k128_fp4_dot8_term(
          words, dst_reg, *src0_base, *src1_base, tmp_a_raw, tmp_b_raw, tmp_a_pos, tmp_a_neg,
          tmp_b_pos, tmp_b_neg, tmp_dot, tmp_cross, tmp_f32, vaddr, reg, k_group, exec_save);
    }
  }

  append_restore_exec(words, exec_save);
  append_wait_valu_vgpr(words);
  return words;
}

std::vector<uint32_t>
lower_v_wmma_scale_f32_16x16x128_f8f6f4_fp4_fp4(const Instruction &inst,
                                                const LivenessAnalysis &liveness) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < 2 * sizeof(gfx1250::Vop3pMachineInst))
    return {};

  gfx1250::Vop3pMachineInst scale{};
  gfx1250::Vop3pMachineInst src{};
  std::memcpy(&scale, raw, sizeof(scale));
  std::memcpy(&src, raw + 2, sizeof(src));

  const uint32_t matrix_a_fmt = src.opsel;
  const uint32_t matrix_b_fmt = (src.pad_14 << 2u) | src.opsel_hi;
  const uint32_t matrix_a_scale_fmt = scale.neg & 0x3u;
  const uint32_t matrix_b_scale_fmt = scale.neg_hi & 0x3u;
  if (scale.encoding != 0xCC || scale.op != 0x35 || src.encoding != 0xCC || src.op != 0x33 ||
      src.vdst > 248 || src.neg_hi != 0 || src.clamp != 0 || src.neg != 0 || matrix_a_fmt != 4 ||
      matrix_b_fmt != 4 || matrix_a_scale_fmt != 0 || matrix_b_scale_fmt != 0)
    return {};

  const auto src0_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src0), 8);
  const auto src1_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src1), 8);
  if (!src0_base || !src1_base || !scale_word_source_supported(static_cast<uint16_t>(scale.src0)) ||
      !scale_word_source_supported(static_cast<uint16_t>(scale.src1)))
    return {};

  if (dst_overlaps_wmma_ab_sources(static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src0),
                                   static_cast<uint16_t>(src.src1), 8))
    return {};

  constexpr uint8_t kOpMovB32 = 1;
  const bool zero_acc = src.src2 == scalar_positive_inline_u32(0);
  const bool accumulate_in_vdst =
      src.src2 == static_cast<uint16_t>(256u + static_cast<uint16_t>(src.vdst));
  std::optional<uint32_t> literal_word;
  std::optional<uint8_t> src2_base;
  if (src.src2 == 255) {
    literal_word = simm32_literal_word(inst, 2);
    if (!literal_word)
      return {};
  } else if (src.src2 == 254) {
    return {};
  } else if (!zero_acc && !accumulate_in_vdst) {
    src2_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src2), 8);
    if (!src2_base || overlaps_vgpr_runs(static_cast<uint16_t>(src.vdst), 8, *src2_base, 8))
      return {};
  }

  std::vector<uint8_t> avoid;
  add_avoid_vgpr_run(avoid, static_cast<uint8_t>(src.vdst), 8);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src0), 8);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src1), 8);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src2), 8);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(scale.src0), 1);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(scale.src1), 1);

  auto tmp_base_opt = find_free_vgpr_run_avoiding(inst, liveness, 10, avoid);
  if (!tmp_base_opt || *tmp_base_opt > 246)
    return {};
  const uint8_t tmp_a_raw = static_cast<uint8_t>(*tmp_base_opt);
  const uint8_t tmp_b_raw = static_cast<uint8_t>(tmp_a_raw + 1u);
  const uint8_t tmp_a_pos = static_cast<uint8_t>(tmp_a_raw + 2u);
  const uint8_t tmp_a_neg = static_cast<uint8_t>(tmp_a_raw + 3u);
  const uint8_t tmp_b_pos = static_cast<uint8_t>(tmp_a_raw + 4u);
  const uint8_t tmp_b_neg = static_cast<uint8_t>(tmp_a_raw + 5u);
  const uint8_t tmp_dot = static_cast<uint8_t>(tmp_a_raw + 6u);
  const uint8_t tmp_cross = static_cast<uint8_t>(tmp_a_raw + 7u);
  const uint8_t tmp_f32 = static_cast<uint8_t>(tmp_a_raw + 8u);
  const uint8_t vaddr = static_cast<uint8_t>(tmp_a_raw + 9u);

  auto exec_save_opt = liveness.find_free_sgpr_pair(&inst);
  if (!exec_save_opt || *exec_save_opt > 124)
    return {};
  const uint8_t exec_save = static_cast<uint8_t>(*exec_save_opt);
  const std::vector<uint8_t> avoid_sgpr = {exec_save, static_cast<uint8_t>(exec_save + 1u)};
  auto scale_pred_opt = find_free_sgpr_pair_avoiding(inst, liveness, avoid_sgpr);
  if (!scale_pred_opt || *scale_pred_opt > 105)
    return {};
  const uint8_t scale_pred = static_cast<uint8_t>(*scale_pred_opt);

  std::vector<uint32_t> words;
  words.reserve(24576);
  append_save_exec(words, exec_save);

  if (zero_acc) {
    for (uint8_t reg = 0; reg < 8; ++reg) {
      append_vop1(words, kOpMovB32, static_cast<uint8_t>(src.vdst + reg),
                  scalar_positive_inline_u32(0));
    }
  } else if (literal_word) {
    for (uint8_t reg = 0; reg < 8; ++reg) {
      append_vop1(words, kOpMovB32, static_cast<uint8_t>(src.vdst + reg), 255, literal_word);
    }
  } else if (!accumulate_in_vdst) {
    append_v_mov_b32_run(words, static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src2), 8);
  }

  const uint8_t matrix_a_scale_select =
      static_cast<uint8_t>((scale.opsel & 0x1u) | (((scale.opsel >> 2u) & 0x1u) << 1u));
  const uint8_t matrix_b_scale_select =
      static_cast<uint8_t>((scale.opsel_hi & 0x1u) | ((scale.pad_14 & 0x1u) << 1u));
  for (uint8_t reg = 0; reg < 8; ++reg) {
    const uint8_t dst_reg = static_cast<uint8_t>(src.vdst + reg);
    for (uint8_t k_group = 0; k_group < 16; ++k_group) {
      append_wmma_f32_k128_fp4_scaled_dot8_term(
          words, dst_reg, *src0_base, *src1_base, static_cast<uint16_t>(scale.src0),
          static_cast<uint16_t>(scale.src1), tmp_a_raw, tmp_b_raw, tmp_a_pos, tmp_a_neg, tmp_b_pos,
          tmp_b_neg, tmp_dot, tmp_cross, tmp_f32, vaddr, reg, k_group, matrix_a_scale_select,
          matrix_b_scale_select, scale_pred, exec_save);
    }
  }

  append_restore_exec(words, exec_save);
  append_wait_valu_vgpr(words);
  return words;
}

std::vector<uint32_t>
lower_v_wmma_scale_f32_16x16x128_f8f6f4_fp8_fp8(const Instruction &inst,
                                                const LivenessAnalysis &liveness) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < 2 * sizeof(gfx1250::Vop3pMachineInst))
    return {};

  gfx1250::Vop3pMachineInst scale{};
  gfx1250::Vop3pMachineInst src{};
  std::memcpy(&scale, raw, sizeof(scale));
  std::memcpy(&src, raw + 2, sizeof(src));

  const uint32_t matrix_a_fmt = src.opsel;
  const uint32_t matrix_b_fmt = (src.pad_14 << 2u) | src.opsel_hi;
  const uint32_t matrix_a_scale_fmt = scale.neg & 0x3u;
  const uint32_t matrix_b_scale_fmt = scale.neg_hi & 0x3u;
  if (scale.encoding != 0xCC || scale.op != 0x35 || src.encoding != 0xCC || src.op != 0x33 ||
      src.vdst > 248 || src.neg_hi != 0 || src.clamp != 0 || src.neg != 0 || matrix_a_fmt != 0 ||
      matrix_b_fmt != 0 || matrix_a_scale_fmt != 0 || matrix_b_scale_fmt != 0)
    return {};

  const auto src0_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src0), 16);
  const auto src1_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src1), 16);
  if (!src0_base || !src1_base || !scale_word_source_supported(static_cast<uint16_t>(scale.src0)) ||
      !scale_word_source_supported(static_cast<uint16_t>(scale.src1)))
    return {};

  if (dst_overlaps_wmma_ab_sources(static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src0),
                                   static_cast<uint16_t>(src.src1), 16))
    return {};

  constexpr uint8_t kOpMovB32 = 1;
  const bool zero_acc = src.src2 == scalar_positive_inline_u32(0);
  const bool accumulate_in_vdst =
      src.src2 == static_cast<uint16_t>(256u + static_cast<uint16_t>(src.vdst));
  std::optional<uint32_t> literal_word;
  std::optional<uint8_t> src2_base;
  if (src.src2 == 255) {
    literal_word = simm32_literal_word(inst, 2);
    if (!literal_word)
      return {};
  } else if (src.src2 == 254) {
    return {};
  } else if (!zero_acc && !accumulate_in_vdst) {
    src2_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src2), 8);
    if (!src2_base || overlaps_vgpr_runs(static_cast<uint16_t>(src.vdst), 8, *src2_base, 8))
      return {};
  }

  std::vector<uint8_t> avoid;
  add_avoid_vgpr_run(avoid, static_cast<uint8_t>(src.vdst), 8);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src0), 16);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src1), 16);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src2), 8);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(scale.src0), 1);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(scale.src1), 1);

  auto tmp_base_opt = find_free_vgpr_run_avoiding(inst, liveness, 4, avoid);
  if (!tmp_base_opt || *tmp_base_opt > 252)
    return {};
  const uint8_t tmp_a = static_cast<uint8_t>(*tmp_base_opt);
  const uint8_t tmp_b = static_cast<uint8_t>(tmp_a + 1u);
  const uint8_t tmp_dot = static_cast<uint8_t>(tmp_a + 2u);
  const uint8_t vaddr = static_cast<uint8_t>(tmp_a + 3u);

  auto exec_save_opt = liveness.find_free_sgpr_pair(&inst);
  if (!exec_save_opt || *exec_save_opt > 124)
    return {};
  const uint8_t exec_save = static_cast<uint8_t>(*exec_save_opt);
  const std::vector<uint8_t> avoid_sgpr = {exec_save, static_cast<uint8_t>(exec_save + 1u)};
  auto scale_pred_opt = find_free_sgpr_pair_avoiding(inst, liveness, avoid_sgpr);
  if (!scale_pred_opt || *scale_pred_opt > 105)
    return {};
  const uint8_t scale_pred = static_cast<uint8_t>(*scale_pred_opt);

  std::vector<uint32_t> words;
  words.reserve(16384);
  append_save_exec(words, exec_save);

  if (zero_acc) {
    for (uint8_t reg = 0; reg < 8; ++reg) {
      append_vop1(words, kOpMovB32, static_cast<uint8_t>(src.vdst + reg),
                  scalar_positive_inline_u32(0));
    }
  } else if (literal_word) {
    for (uint8_t reg = 0; reg < 8; ++reg) {
      append_vop1(words, kOpMovB32, static_cast<uint8_t>(src.vdst + reg), 255, literal_word);
    }
  } else if (!accumulate_in_vdst) {
    append_v_mov_b32_run(words, static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src2), 8);
  }

  const uint8_t matrix_a_scale_select =
      static_cast<uint8_t>((scale.opsel & 0x1u) | (((scale.opsel >> 2u) & 0x1u) << 1u));
  const uint8_t matrix_b_scale_select =
      static_cast<uint8_t>((scale.opsel_hi & 0x1u) | ((scale.pad_14 & 0x1u) << 1u));
  for (uint8_t reg = 0; reg < 8; ++reg) {
    const uint8_t dst_reg = static_cast<uint8_t>(src.vdst + reg);
    for (uint8_t k_group = 0; k_group < 32; ++k_group) {
      append_wmma_f32_k128_fp8_scaled_dot4_term(
          words, dst_reg, *src0_base, *src1_base, static_cast<uint16_t>(scale.src0),
          static_cast<uint16_t>(scale.src1), tmp_a, tmp_b, tmp_dot, vaddr, reg, k_group,
          matrix_a_scale_select, matrix_b_scale_select, scale_pred, exec_save);
    }
  }

  append_restore_exec(words, exec_save);
  append_wait_valu_vgpr(words);
  return words;
}

ExpandResult lower_v_wmma_scale_f32_16x16x128_f8f6f4_scalar(const Instruction &inst,
                                                            const LivenessAnalysis &liveness);
ExpandResult lower_v_wmma_f32_32x16x128_f4_scalar(const Instruction &inst,
                                                  const LivenessAnalysis &liveness, bool scaled);

ExpandResult expand_v_wmma_scale_f32_16x16x128_f8f6f4(const Instruction &inst, uint32_t, uint64_t,
                                                      const LivenessAnalysis &liveness,
                                                      TranslationContext &, const LaneLayout *,
                                                      const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (raw && static_cast<size_t>(inst.size()) >= 4u * sizeof(uint32_t)) {
    gfx1250::Vop3pMachineInst matrix{};
    std::memcpy(&matrix, raw + 2, sizeof(matrix));
    if (matrix.op == 0x88)
      return lower_v_wmma_f32_32x16x128_f4_scalar(inst, liveness, true);
  }

  auto scalar = lower_v_wmma_scale_f32_16x16x128_f8f6f4_scalar(inst, liveness);
  if (scalar.status != ExpandStatus::NotHandled)
    return scalar;

  auto fp4_words = lower_v_wmma_scale_f32_16x16x128_f8f6f4_fp4_fp4(inst, liveness);
  if (!fp4_words.empty())
    return ExpandResult::success(std::move(fp4_words));

  auto words = lower_v_wmma_scale_f32_16x16x128_f8f6f4_fp8_fp8(inst, liveness);
  if (!words.empty())
    return ExpandResult::success(std::move(words));

  return lower_v_wmma_scale_f32_16x16x128_f8f6f4_scalar(inst, liveness);
}

std::vector<uint32_t> expand_v_wmma_i32_16x16x64_iu8(const Instruction &inst, uint32_t, uint64_t,
                                                     const LivenessAnalysis &liveness,
                                                     const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3pMachineInst))
    return {};

  gfx1250::Vop3pMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.vdst > 248 || src.neg_hi != 0 || src.opsel != 0)
    return {};

  if (!src_vgpr_base_for_run(static_cast<uint16_t>(src.src0), 8) ||
      !src_vgpr_base_for_run(static_cast<uint16_t>(src.src1), 8))
    return {};

  std::optional<uint32_t> literal_word;
  if (src.src2 == 255) {
    literal_word = simm32_literal_word(inst, 2);
    if (!literal_word)
      return {};
  } else if (src.src2 == 254) {
    return {};
  } else if (src.src2 >= 256 && !src_vgpr_base_for_run(static_cast<uint16_t>(src.src2), 8)) {
    return {};
  }

  std::vector<uint8_t> avoid;
  add_avoid_vgpr_run(avoid, static_cast<uint8_t>(src.vdst), 8);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src0), 8);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src1), 8);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src2), 8);

  const bool accumulate_in_vdst =
      (src.vdst % 8) == 0 &&
      !dst_overlaps_wmma_ab_sources(static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src0),
                                    static_cast<uint16_t>(src.src1));

  uint8_t tmp_a = 0;
  uint8_t tmp_b = 0;
  uint8_t tmp_acc = 0;
  uint8_t vaddr_xor16 = 0;
  if (accumulate_in_vdst) {
    auto tmp_base_opt = find_aligned_free_vgpr_run_avoiding(inst, liveness, 5, 2, avoid);
    if (!tmp_base_opt || *tmp_base_opt > 251)
      return {};
    tmp_a = static_cast<uint8_t>(*tmp_base_opt);
    tmp_b = static_cast<uint8_t>(tmp_a + 2u);
    tmp_acc = static_cast<uint8_t>(src.vdst);
    vaddr_xor16 = static_cast<uint8_t>(tmp_a + 4u);
  } else {
    auto tmp_base_opt = find_aligned_free_vgpr_run_avoiding(inst, liveness, 13, 8, avoid);
    if (!tmp_base_opt || *tmp_base_opt > 243)
      return {};
    tmp_acc = static_cast<uint8_t>(*tmp_base_opt);
    tmp_a = static_cast<uint8_t>(tmp_acc + 8u);
    tmp_b = static_cast<uint8_t>(tmp_acc + 10u);
    vaddr_xor16 = static_cast<uint8_t>(tmp_acc + 12u);
  }

  auto exec_save_opt = liveness.find_free_sgpr_pair(&inst);
  if (!exec_save_opt || *exec_save_opt > 124)
    return {};
  const uint8_t exec_save = static_cast<uint8_t>(*exec_save_opt);
  const std::vector<uint8_t> avoid_sgpr = {exec_save, static_cast<uint8_t>(exec_save + 1u)};
  auto vcc_save_opt = find_free_sgpr_pair_avoiding(inst, liveness, avoid_sgpr);
  if (!vcc_save_opt || *vcc_save_opt > 124)
    return {};
  const uint8_t vcc_save = static_cast<uint8_t>(*vcc_save_opt);

  constexpr uint8_t kSoppWaitIdle = 0x0A;
  constexpr uint8_t kSoppWaitKmcnt = 0x47;

  std::vector<uint32_t> words;
  words.reserve(160);
  append_save_exec(words, exec_save);
  append_save_vcc(words, vcc_save);
  append_set_exec_lo_mask(words, 0xFFFFFFFFu);
  append_lane_xor16_byte_addr(words, vaddr_xor16);
  append_restore_exec(words, exec_save);
  append_wait_valu_vgpr(words);

  uint16_t acc_src = static_cast<uint16_t>(src.src2);
  if (src.src2 >= 256 && src.src2 != static_cast<uint16_t>(256u + tmp_acc)) {
    append_v_mov_b32_run(words, tmp_acc, static_cast<uint16_t>(src.src2), 8);
    acc_src = static_cast<uint16_t>(256u + tmp_acc);
  }

  for (uint8_t chunk = 0; chunk < 4; ++chunk) {
    append_wmma_2word_relayout_chunk(words, static_cast<uint16_t>(src.src0),
                                     static_cast<uint16_t>(src.src1), tmp_a, tmp_b, vaddr_xor16, 8,
                                     2, chunk, exec_save);
    append_wmma_i32_16x16x16_iu8(words, tmp_acc, tmp_a, tmp_b, acc_src,
                                 static_cast<uint8_t>(src.neg), src.clamp != 0, literal_word);
    words.push_back(pack_sopp(kSoppWaitKmcnt, 0));
    // Chained split i8 WMMA accumulation can see stale reads with kmcnt alone.
    words.push_back(pack_sopp(kSoppWaitIdle, 0));
    acc_src = static_cast<uint16_t>(256u + tmp_acc);
    literal_word = std::nullopt;
  }

  if (!accumulate_in_vdst) {
    append_v_mov_b32_run(words, static_cast<uint8_t>(src.vdst),
                         static_cast<uint16_t>(256u + tmp_acc), 8);
    append_wait_valu_vgpr(words);
  }
  append_restore_vcc(words, vcc_save);
  return words;
}

std::vector<uint32_t> expand_v_swmmac_f32_16x16x64_f16(const Instruction &inst, uint32_t, uint64_t,
                                                       const LivenessAnalysis &liveness,
                                                       const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3pMachineInst))
    return {};

  gfx1250::Vop3pMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  uint8_t target_op = 0;
  uint8_t output_words = 8;
  switch (src.op) {
  case 0x65:
    target_op = 0x50; // f32, f16 inputs
    break;
  case 0x66:
  case 0x69:
    target_op = 0x51; // f32, bf16 inputs (including bf16f32 accumulator spelling)
    break;
  case 0x67:
    target_op = 0x52; // packed f16 result
    output_words = 4;
    break;
  case 0x68:
    target_op = 0x53; // packed bf16 result
    output_words = 4;
    break;
  default:
    return {};
  }
  if (src.vdst > static_cast<uint8_t>(256u - output_words) || src.neg_hi != 0 || src.clamp != 0 ||
      src.neg != 0 || (src.opsel & ~0x4u) != 0)
    return {};

  const auto src0_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src0), 8);
  const auto src1_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src1), 16);
  const auto index_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src2), 1);
  if (!src0_base || !src1_base || !index_base)
    return {};

  std::vector<uint8_t> avoid;
  add_avoid_vgpr_run(avoid, static_cast<uint8_t>(src.vdst), output_words);
  add_avoid_vgpr_run(avoid, *src0_base, 8);
  add_avoid_vgpr_run(avoid, *src1_base, 16);
  add_avoid_vgpr(avoid, *index_base);

  auto exec_save_opt = liveness.find_free_sgpr_pair(&inst);
  if (!exec_save_opt || *exec_save_opt > 124)
    return {};
  const uint8_t exec_save = static_cast<uint8_t>(*exec_save_opt);

  constexpr uint8_t kTmpCount = 22;
  auto tmp_base_opt = find_aligned_free_vgpr_run_avoiding(inst, liveness, kTmpCount, 8, avoid);
  std::optional<PrivateBorrowedVgprRun> borrowed;
  if (!tmp_base_opt || *tmp_base_opt > 234) {
    borrowed = find_private_borrowed_vgpr_run(liveness, kTmpCount, 8, avoid);
    if (!borrowed || borrowed->base > 234)
      return {};
    tmp_base_opt = borrowed->base;
  }
  const uint8_t tmp_acc = static_cast<uint8_t>(*tmp_base_opt);
  const uint8_t tmp_a = static_cast<uint8_t>(tmp_acc + 8u);
  const uint8_t tmp_b = static_cast<uint8_t>(tmp_acc + 12u);
  const uint8_t vaddr_xor16 = static_cast<uint8_t>(tmp_acc + 20u);
  const uint8_t tmp_index = static_cast<uint8_t>(tmp_acc + 21u);

  constexpr uint8_t kSoppWaitKmcnt = 0x47;

  std::vector<uint32_t> words;
  words.reserve(192 + (borrowed ? borrowed->count * 6u + 16u : 0u));
  append_save_exec(words, exec_save);
  if (borrowed)
    append_private_borrow_save(words, *borrowed);
  append_set_exec_lo_mask(words, 0xFFFFFFFFu);
  append_lane_xor16_byte_addr(words, vaddr_xor16);
  append_restore_exec(words, exec_save);
  append_wait_valu_vgpr(words);

  append_v_mov_b32_run(words, tmp_acc, static_cast<uint16_t>(256u + src.vdst), output_words);

  for (uint8_t chunk = 0; chunk < 2; ++chunk) {
    append_swmmac_f16_a_chunk_relayout(words, tmp_a, vaddr_xor16, *src0_base, chunk, exec_save);
    append_swmmac_f16_b_chunk_relayout(words, tmp_b, vaddr_xor16, *src1_base, chunk, exec_save);
    append_swmmac_index_chunk_relayout(words, tmp_index, vaddr_xor16, *index_base, chunk, false,
                                       exec_save);
    append_swmmac_16x16x32(words, target_op, tmp_acc, tmp_a, tmp_b, tmp_index);
    words.push_back(pack_sopp(kSoppWaitKmcnt, 0));
  }

  append_v_mov_b32_run(words, static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(256u + tmp_acc),
                       output_words);
  if (borrowed)
    append_private_borrow_restore(words, *borrowed, exec_save);
  return words;
}

[[nodiscard]] std::optional<uint8_t> rdna4_swmmac_f32_f8_opcode(uint16_t gfx1250_op) {
  if (gfx1250_op >= 0x73u && gfx1250_op <= 0x76u)
    return static_cast<uint8_t>(0x57u + gfx1250_op - 0x73u);
  if (gfx1250_op >= 0x77u && gfx1250_op <= 0x7Au)
    return static_cast<uint8_t>(0x57u + gfx1250_op - 0x77u);
  return std::nullopt;
}

std::vector<uint32_t> expand_v_swmmac_16x16x128_f8(const Instruction &inst, uint32_t, uint64_t,
                                                   const LivenessAnalysis &liveness,
                                                   const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3pMachineInst))
    return {};

  gfx1250::Vop3pMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  const auto target_op = rdna4_swmmac_f32_f8_opcode(src.op);
  if (!target_op)
    return {};
  const bool packed_f16_output = src.op >= 0x77u;
  const uint8_t output_words = packed_f16_output ? 4u : 8u;
  if (src.vdst > static_cast<uint8_t>(256u - output_words) || src.neg_hi != 0 ||
      (src.opsel & ~0x4u) != 0 || src.clamp != 0 || src.neg != 0)
    return {};

  const auto src0_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src0), 8);
  const auto src1_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src1), 16);
  const auto index_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src2), 2);
  if (!src0_base || !src1_base || !index_base)
    return {};

  std::vector<uint8_t> avoid;
  add_avoid_vgpr_run(avoid, static_cast<uint8_t>(src.vdst), output_words);
  add_avoid_vgpr_run(avoid, *src0_base, 8);
  add_avoid_vgpr_run(avoid, *src1_base, 16);
  add_avoid_vgpr_run(avoid, *index_base, 2);

  auto exec_save_opt = liveness.find_free_sgpr_pair(&inst);
  if (!exec_save_opt || *exec_save_opt > 124)
    return {};
  const uint8_t exec_save = static_cast<uint8_t>(*exec_save_opt);

  constexpr uint8_t kTmpCount = 17;
  auto tmp_base_opt = find_aligned_free_vgpr_run_avoiding(inst, liveness, kTmpCount, 8, avoid);
  std::optional<PrivateBorrowedVgprRun> borrowed;
  if (!tmp_base_opt || *tmp_base_opt > 239) {
    borrowed = find_private_borrowed_vgpr_run(liveness, kTmpCount, 8, avoid);
    if (!borrowed || borrowed->base > 239)
      return {};
    tmp_base_opt = borrowed->base;
  }
  const uint8_t tmp_acc = static_cast<uint8_t>(*tmp_base_opt);
  const uint8_t tmp_a = static_cast<uint8_t>(tmp_acc + 8u);
  const uint8_t tmp_b = static_cast<uint8_t>(tmp_acc + 10u);
  const uint8_t vaddr = static_cast<uint8_t>(tmp_acc + 14u);
  const uint8_t tmp_index = static_cast<uint8_t>(tmp_acc + 15u);
  const uint8_t tmp_index_merge = static_cast<uint8_t>(tmp_acc + 16u);

  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpOrB32 = 28;
  constexpr uint16_t kOpCvtF16F32 = 0x18A;
  constexpr uint16_t kOpCvtF32F16 = 0x18B;
  constexpr uint8_t kSoppWaitKmcnt = 0x47;

  std::vector<uint32_t> words;
  words.reserve(512 + (borrowed ? borrowed->count * 6u + 16u : 0u));
  append_save_exec(words, exec_save);
  if (borrowed)
    append_private_borrow_save(words, *borrowed);

  if (packed_f16_output) {
    for (uint8_t word = 0; word < 4; ++word) {
      const uint16_t packed = static_cast<uint16_t>(256u + src.vdst + word);
      append_vop3_mod(words, kOpCvtF32F16, static_cast<uint8_t>(tmp_acc + 2u * word), packed, 0, 0,
                      0);
      append_vop3_mod(words, kOpCvtF32F16, static_cast<uint8_t>(tmp_acc + 2u * word + 1u), packed,
                      0, 0, 1);
    }
    append_wait_valu_vgpr(words);
  } else {
    append_v_mov_b32_run(words, tmp_acc, static_cast<uint16_t>(256u + src.vdst), 8);
  }

  for (uint8_t chunk = 0; chunk < 4; ++chunk) {
    append_swmmac_fp8_a_chunk_relayout(words, tmp_a, tmp_index, vaddr, *src0_base, chunk,
                                       exec_save);
    append_swmmac_fp8_b_chunk_relayout(words, tmp_b, vaddr, *src1_base, chunk, exec_save);
    append_swmmac_fp8_index_chunk_relayout(words, tmp_index, tmp_index_merge, vaddr, *index_base,
                                           chunk, exec_save);
    append_swmmac_16x16x32(words, *target_op, tmp_acc, tmp_a, tmp_b, tmp_index);
    words.push_back(pack_sopp(kSoppWaitKmcnt, 0));
  }

  if (packed_f16_output) {
    for (uint8_t word = 0; word < 4; ++word) {
      append_vop3(words, kOpCvtF16F32, tmp_a, static_cast<uint16_t>(256u + tmp_acc + 2u * word), 0);
      append_vop3(words, kOpCvtF16F32, tmp_b,
                  static_cast<uint16_t>(256u + tmp_acc + 2u * word + 1u), 0);
      append_wait_valu_vgpr(words);
      std::optional<uint32_t> mask_literal;
      append_vop2(words, kOpAndB32, tmp_a, literal_or_inline_u32(0xFFFFu, mask_literal), tmp_a,
                  mask_literal);
      append_vop2(words, kOpLshlrevB32, tmp_b, scalar_positive_inline_u32(16), tmp_b);
      append_wait_valu_vgpr(words);
      append_vop2(words, kOpOrB32, static_cast<uint8_t>(src.vdst + word), vgpr_src(tmp_a), tmp_b);
    }
  } else {
    append_v_mov_b32_run(words, static_cast<uint8_t>(src.vdst),
                         static_cast<uint16_t>(256u + tmp_acc), 8);
  }
  if (borrowed)
    append_private_borrow_restore(words, *borrowed, exec_save);
  return words;
}

std::vector<uint32_t> expand_v_swmmac_i32_16x16x128_iu8(const Instruction &inst, uint32_t, uint64_t,
                                                        const LivenessAnalysis &liveness,
                                                        const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3pMachineInst))
    return {};

  gfx1250::Vop3pMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.vdst > 248 || src.neg_hi != 0 || (src.opsel & ~0x4u) != 0)
    return {};

  const auto src0_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src0), 8);
  const auto src1_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src1), 16);
  const auto index_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src2), 2);
  if (!src0_base || !src1_base || !index_base)
    return {};

  std::vector<uint8_t> avoid;
  add_avoid_vgpr_run(avoid, static_cast<uint8_t>(src.vdst), 8);
  add_avoid_vgpr_run(avoid, *src0_base, 8);
  add_avoid_vgpr_run(avoid, *src1_base, 16);
  add_avoid_vgpr_run(avoid, *index_base, 2);

  auto tmp_base_opt = find_aligned_free_vgpr_run_avoiding(inst, liveness, 15, 8, avoid);
  if (!tmp_base_opt || *tmp_base_opt > 241)
    return {};
  const uint8_t tmp_acc = static_cast<uint8_t>(*tmp_base_opt);
  const uint8_t tmp_a = static_cast<uint8_t>(tmp_acc + 8u);
  const uint8_t tmp_b = static_cast<uint8_t>(tmp_acc + 10u);
  const uint8_t vaddr_xor16 = static_cast<uint8_t>(tmp_acc + 14u);

  auto exec_save_opt = liveness.find_free_sgpr_pair(&inst);
  if (!exec_save_opt || *exec_save_opt > 124)
    return {};
  const uint8_t exec_save = static_cast<uint8_t>(*exec_save_opt);

  constexpr uint8_t kSoppWaitKmcnt = 0x47;

  std::vector<uint32_t> words;
  words.reserve(240);
  append_save_exec(words, exec_save);
  append_set_exec_lo_mask(words, 0xFFFFFFFFu);
  append_lane_xor16_byte_addr(words, vaddr_xor16);
  append_restore_exec(words, exec_save);
  append_wait_valu_vgpr(words);

  append_v_mov_b32_run(words, tmp_acc, static_cast<uint16_t>(256u + src.vdst), 8);

  for (uint8_t chunk = 0; chunk < 4; ++chunk) {
    append_wmma_single_relayout_chunk(words, static_cast<uint16_t>(src.src0), tmp_a, vaddr_xor16, 8,
                                      2, chunk, exec_save);
    append_wmma_single_relayout_chunk(words, static_cast<uint16_t>(src.src1), tmp_b, vaddr_xor16,
                                      16, 4, chunk, exec_save);
    const uint8_t chunk_index = static_cast<uint8_t>(*index_base + (chunk / 2u));
    append_swmmac_i32_16x16x32_iu8(words, tmp_acc, tmp_a, tmp_b, chunk_index,
                                   static_cast<uint8_t>(src.neg), src.clamp != 0);
    words.push_back(pack_sopp(kSoppWaitKmcnt, 0));
  }

  append_v_mov_b32_run(words, static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(256u + tmp_acc),
                       8);
  return words;
}

[[nodiscard]] std::optional<uint8_t> gfx1250_wmma_f8_format_index(uint16_t op) {
  for (const uint16_t base : {uint16_t{0x6A}, uint16_t{0x6E}, uint16_t{0x80}, uint16_t{0x84}}) {
    if (op >= base && op < base + 4u)
      return static_cast<uint8_t>(op - base);
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<uint8_t> rdna4_wmma_f32_f8_opcode(uint16_t gfx1250_op) {
  const auto format = gfx1250_wmma_f8_format_index(gfx1250_op);
  if (!format)
    return std::nullopt;
  return static_cast<uint8_t>(0x46u + *format);
}

[[nodiscard]] std::optional<uint8_t> rdna4_dot4_f32_f8_opcode(uint16_t gfx1250_op) {
  const auto format = gfx1250_wmma_f8_format_index(gfx1250_op);
  if (!format)
    return std::nullopt;
  // RDNA4 orders the mixed dot products before the same-format forms.
  constexpr std::array<uint8_t, 4> kOpcodes{0x26, 0x24, 0x25, 0x27};
  return kOpcodes[*format];
}

std::vector<uint32_t> expand_v_wmma_f32_16x16xk_f8(const Instruction &inst,
                                                   const LivenessAnalysis &liveness,
                                                   uint8_t src_words, uint8_t chunks) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3pMachineInst))
    return {};

  gfx1250::Vop3pMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  const auto target_op = rdna4_wmma_f32_f8_opcode(inst.opcode());
  if (!target_op)
    return {};
  constexpr uint8_t kWmmaMatrixAReuse = 0x4;
  if (src.vdst > 248 || src.neg_hi != 0 || (src.opsel & ~kWmmaMatrixAReuse) != 0 ||
      src.clamp != 0 || src.neg != 0)
    return {};

  if (!src_vgpr_base_for_run(static_cast<uint16_t>(src.src0), src_words) ||
      !src_vgpr_base_for_run(static_cast<uint16_t>(src.src1), src_words))
    return {};

  std::optional<uint32_t> literal_word;
  if (src.src2 == 255) {
    literal_word = simm32_literal_word(inst, 2);
    if (!literal_word)
      return {};
  } else if (src.src2 == 254) {
    return {};
  } else if (src.src2 >= 256 && !src_vgpr_base_for_run(static_cast<uint16_t>(src.src2), 8)) {
    return {};
  }

  std::vector<uint8_t> avoid;
  add_avoid_vgpr_run(avoid, static_cast<uint8_t>(src.vdst), 8);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src0), src_words);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src1), src_words);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src2), 8);

  const bool accumulate_in_vdst =
      (src.vdst % 8) == 0 &&
      !dst_overlaps_wmma_ab_sources(static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src0),
                                    static_cast<uint16_t>(src.src1), src_words);

  uint8_t tmp_a = 0;
  uint8_t tmp_b = 0;
  uint8_t tmp_acc = 0;
  uint8_t vaddr_xor16 = 0;
  if (accumulate_in_vdst) {
    tmp_acc = static_cast<uint8_t>(src.vdst);
    auto tmp_a_opt = find_aligned_free_vgpr_run_avoiding(inst, liveness, 2, 2, avoid);
    if (!tmp_a_opt || *tmp_a_opt > 254)
      return {};
    tmp_a = static_cast<uint8_t>(*tmp_a_opt);
    add_avoid_vgpr_run(avoid, tmp_a, 2);

    auto tmp_b_opt = find_aligned_free_vgpr_run_avoiding(inst, liveness, 2, 2, avoid);
    if (!tmp_b_opt || *tmp_b_opt > 254)
      return {};
    tmp_b = static_cast<uint8_t>(*tmp_b_opt);
    add_avoid_vgpr_run(avoid, tmp_b, 2);

    auto vaddr_opt = find_free_vgpr_run_avoiding(inst, liveness, 1, avoid);
    if (!vaddr_opt || *vaddr_opt > 255)
      return {};
    vaddr_xor16 = static_cast<uint8_t>(*vaddr_opt);
  } else {
    auto tmp_acc_opt = find_aligned_free_vgpr_run_avoiding(inst, liveness, 8, 8, avoid);
    if (!tmp_acc_opt || *tmp_acc_opt > 248)
      return {};
    tmp_acc = static_cast<uint8_t>(*tmp_acc_opt);
    add_avoid_vgpr_run(avoid, tmp_acc, 8);

    auto tmp_a_opt = find_aligned_free_vgpr_run_avoiding(inst, liveness, 2, 2, avoid);
    if (!tmp_a_opt || *tmp_a_opt > 254)
      return {};
    tmp_a = static_cast<uint8_t>(*tmp_a_opt);
    add_avoid_vgpr_run(avoid, tmp_a, 2);

    auto tmp_b_opt = find_aligned_free_vgpr_run_avoiding(inst, liveness, 2, 2, avoid);
    if (!tmp_b_opt || *tmp_b_opt > 254)
      return {};
    tmp_b = static_cast<uint8_t>(*tmp_b_opt);
    add_avoid_vgpr_run(avoid, tmp_b, 2);

    auto vaddr_opt = find_free_vgpr_run_avoiding(inst, liveness, 1, avoid);
    if (!vaddr_opt || *vaddr_opt > 255)
      return {};
    vaddr_xor16 = static_cast<uint8_t>(*vaddr_opt);
  }

  auto exec_save_opt = liveness.find_free_sgpr_pair(&inst);
  if (!exec_save_opt || *exec_save_opt > 124)
    return {};
  const uint8_t exec_save = static_cast<uint8_t>(*exec_save_opt);

  constexpr uint8_t kSoppWaitKmcnt = 0x47;

  std::vector<uint32_t> words;
  words.reserve(static_cast<size_t>(32u + chunks * 34u));
  append_save_exec(words, exec_save);
  append_set_exec_lo_mask(words, 0xFFFFFFFFu);
  append_lane_xor16_byte_addr(words, vaddr_xor16);
  append_restore_exec(words, exec_save);
  append_wait_valu_vgpr(words);

  uint16_t acc_src = static_cast<uint16_t>(src.src2);
  if (src.src2 >= 256 && src.src2 != static_cast<uint16_t>(256u + tmp_acc)) {
    append_v_mov_b32_run(words, tmp_acc, static_cast<uint16_t>(src.src2), 8);
    acc_src = static_cast<uint16_t>(256u + tmp_acc);
  }

  for (uint8_t chunk = 0; chunk < chunks; ++chunk) {
    append_wmma_2word_relayout_chunk(
        words, static_cast<uint16_t>(src.src0), static_cast<uint16_t>(src.src1), tmp_a, tmp_b,
        vaddr_xor16, src_words, static_cast<uint8_t>(chunks / 2u), chunk, exec_save);
    append_wmma_f32_16x16x16_f8(words, *target_op, tmp_acc, tmp_a, tmp_b, acc_src, literal_word);
    words.push_back(pack_sopp(kSoppWaitKmcnt, 0));
    acc_src = static_cast<uint16_t>(256u + tmp_acc);
    literal_word = std::nullopt;
  }

  if (!accumulate_in_vdst) {
    append_v_mov_b32_run(words, static_cast<uint8_t>(src.vdst),
                         static_cast<uint16_t>(256u + tmp_acc), 8);
  }
  return words;
}

std::vector<uint32_t> expand_v_wmma_f32_16x16x64_fp8_fp8(const Instruction &inst, uint32_t,
                                                         uint64_t, const LivenessAnalysis &liveness,
                                                         const LaneLayout *, const LaneLayout *) {
  return expand_v_wmma_f32_16x16xk_f8(inst, liveness, 8, 4);
}

std::vector<uint32_t> expand_v_wmma_f16_16x16xk_f8(const Instruction &inst,
                                                   const LivenessAnalysis &liveness,
                                                   uint8_t src_words, uint8_t chunks) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3pMachineInst))
    return {};

  gfx1250::Vop3pMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  const auto target_op = rdna4_wmma_f32_f8_opcode(inst.opcode());
  if (!target_op || src.vdst > 252 || src.neg_hi != 0 || (src.opsel & ~0x4u) != 0 ||
      src.clamp != 0 || src.neg != 0)
    return {};

  if (!src_vgpr_base_for_run(static_cast<uint16_t>(src.src0), src_words) ||
      !src_vgpr_base_for_run(static_cast<uint16_t>(src.src1), src_words))
    return {};

  const bool zero_acc = src.src2 == scalar_positive_inline_u32(0);
  std::optional<uint32_t> literal_word;
  std::optional<uint8_t> src2_base;
  if (src.src2 == 255) {
    literal_word = simm32_literal_word(inst, 2);
    if (!literal_word)
      return {};
  } else if (src.src2 == 254) {
    return {};
  } else if (!zero_acc) {
    src2_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src2), 4);
    if (!src2_base)
      return {};
  }

  std::vector<uint8_t> avoid;
  add_avoid_vgpr_run(avoid, static_cast<uint8_t>(src.vdst), 4);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src0), src_words);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src1), src_words);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src2), 4);

  constexpr uint16_t kScratchCount = 13;
  const auto scratch = find_aligned_free_vgpr_run_avoiding(inst, liveness, kScratchCount, 8, avoid);
  if (!scratch || *scratch > 243)
    return {};
  const uint8_t tmp_acc = static_cast<uint8_t>(*scratch);
  const uint8_t tmp_a = static_cast<uint8_t>(*scratch + 8u);
  const uint8_t tmp_b = static_cast<uint8_t>(*scratch + 10u);
  const uint8_t vaddr_xor16 = static_cast<uint8_t>(*scratch + 12u);

  const auto exec_save_opt = liveness.find_free_sgpr_pair(&inst);
  if (!exec_save_opt || *exec_save_opt > 124)
    return {};
  const uint8_t exec_save = static_cast<uint8_t>(*exec_save_opt);

  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpOrB32 = 28;
  constexpr uint16_t kOpCvtF16F32 = 0x18A;
  constexpr uint16_t kOpCvtF32F16 = 0x18B;
  constexpr uint8_t kSoppWaitKmcnt = 0x47;

  std::vector<uint32_t> words;
  words.reserve(static_cast<size_t>(64u + chunks * 36u));
  append_save_exec(words, exec_save);
  append_set_exec_lo_mask(words, 0xFFFFFFFFu);
  append_lane_xor16_byte_addr(words, vaddr_xor16);
  append_restore_exec(words, exec_save);
  append_wait_valu_vgpr(words);

  if (zero_acc) {
    append_v_mov_b32_broadcast(words, tmp_acc, scalar_positive_inline_u32(0), 8);
  } else {
    for (uint8_t word = 0; word < 4; ++word) {
      const uint16_t packed = literal_word ? 255u : static_cast<uint16_t>(256u + *src2_base + word);
      append_vop3_mod(words, kOpCvtF32F16, static_cast<uint8_t>(tmp_acc + 2u * word), packed, 0, 0,
                      0, literal_word);
      append_vop3_mod(words, kOpCvtF32F16, static_cast<uint8_t>(tmp_acc + 2u * word + 1u), packed,
                      0, 0, 1, literal_word);
    }
  }
  append_wait_valu_vgpr(words);

  for (uint8_t chunk = 0; chunk < chunks; ++chunk) {
    append_wmma_2word_relayout_chunk(
        words, static_cast<uint16_t>(src.src0), static_cast<uint16_t>(src.src1), tmp_a, tmp_b,
        vaddr_xor16, src_words, static_cast<uint8_t>(chunks / 2u), chunk, exec_save);
    append_wmma_f32_16x16x16_f8(words, *target_op, tmp_acc, tmp_a, tmp_b,
                                static_cast<uint16_t>(256u + tmp_acc));
    words.push_back(pack_sopp(kSoppWaitKmcnt, 0));
  }

  for (uint8_t word = 0; word < 4; ++word) {
    append_vop3(words, kOpCvtF16F32, tmp_a, static_cast<uint16_t>(256u + tmp_acc + 2u * word), 0);
    append_vop3(words, kOpCvtF16F32, tmp_b, static_cast<uint16_t>(256u + tmp_acc + 2u * word + 1u),
                0);
    append_wait_valu_vgpr(words);
    std::optional<uint32_t> mask_literal;
    append_vop2(words, kOpAndB32, tmp_a, literal_or_inline_u32(0xFFFFu, mask_literal), tmp_a,
                mask_literal);
    append_vop2(words, kOpLshlrevB32, tmp_b, scalar_positive_inline_u32(16), tmp_b);
    append_wait_valu_vgpr(words);
    append_vop2(words, kOpOrB32, static_cast<uint8_t>(src.vdst + word),
                static_cast<uint16_t>(256u + tmp_a), tmp_b);
  }
  return words;
}

std::vector<uint32_t> expand_v_wmma_f16_16x16x64_f8(const Instruction &inst, uint32_t, uint64_t,
                                                    const LivenessAnalysis &liveness,
                                                    const LaneLayout *, const LaneLayout *) {
  return expand_v_wmma_f16_16x16xk_f8(inst, liveness, 8, 4);
}

std::vector<uint32_t> expand_v_wmma_f32_16x16x128_fp8_fp8_dot4_fallback(
    const Instruction &inst, const LivenessAnalysis &liveness,
    std::optional<uint8_t> explicit_dot_op = std::nullopt) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3pMachineInst))
    return {};

  gfx1250::Vop3pMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  const auto dot_op = explicit_dot_op ? explicit_dot_op : rdna4_dot4_f32_f8_opcode(inst.opcode());
  if (!dot_op)
    return {};
  if (src.vdst > 248 || src.neg_hi != 0 || (!explicit_dot_op && src.opsel != 0) || src.clamp != 0 ||
      src.neg != 0)
    return {};

  const auto src0_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src0), 16);
  const auto src1_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src1), 16);
  if (!src0_base || !src1_base)
    return {};

  if (dst_overlaps_wmma_ab_sources(static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src0),
                                   static_cast<uint16_t>(src.src1), 16))
    return {};

  constexpr uint8_t kOpMovB32 = 1;
  const bool zero_acc = src.src2 == scalar_positive_inline_u32(0);
  const bool accumulate_in_vdst =
      src.src2 == static_cast<uint16_t>(256u + static_cast<uint16_t>(src.vdst));
  std::optional<uint32_t> literal_word;
  std::optional<uint8_t> src2_base;
  if (src.src2 == 255) {
    literal_word = simm32_literal_word(inst, 2);
    if (!literal_word)
      return {};
  } else if (src.src2 == 254) {
    return {};
  } else if (!zero_acc && !accumulate_in_vdst) {
    src2_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src2), 8);
    if (!src2_base || overlaps_vgpr_runs(static_cast<uint16_t>(src.vdst), 8, *src2_base, 8))
      return {};
  }

  std::vector<uint8_t> avoid;
  add_avoid_vgpr_run(avoid, static_cast<uint8_t>(src.vdst), 8);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src0), 16);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src1), 16);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src2), 8);

  auto tmp_b_opt = find_free_vgpr_run_avoiding(inst, liveness, 1, avoid);
  if (!tmp_b_opt || *tmp_b_opt > 255)
    return {};
  const uint8_t tmp_b = static_cast<uint8_t>(*tmp_b_opt);
  add_avoid_vgpr(avoid, tmp_b);

  auto vaddr_opt = find_free_vgpr_run_avoiding(inst, liveness, 1, avoid);
  if (!vaddr_opt || *vaddr_opt > 255)
    return {};
  const uint8_t vaddr = static_cast<uint8_t>(*vaddr_opt);
  add_avoid_vgpr(avoid, vaddr);

  auto exec_save_opt = liveness.find_free_sgpr_pair(&inst);
  if (!exec_save_opt || *exec_save_opt > 124)
    return {};
  const uint8_t exec_save = static_cast<uint8_t>(*exec_save_opt);

  const std::vector<uint8_t> avoid_sgpr{exec_save, static_cast<uint8_t>(exec_save + 1u)};
  const auto scalar_a_opt = find_free_sgprs_avoiding<8>(inst, liveness, avoid_sgpr);

  bool grouped_dot4 = true;
  std::optional<uint16_t> tmp_a_opt;
  if (!scalar_a_opt) {
    tmp_a_opt = find_free_vgpr_run_avoiding(inst, liveness, 8, avoid);
    if (!tmp_a_opt) {
      grouped_dot4 = false;
      tmp_a_opt = find_free_vgpr_run_avoiding(inst, liveness, 1, avoid);
    }
    if (!tmp_a_opt || *tmp_a_opt > 255)
      return {};
  }
  const uint8_t tmp_a = static_cast<uint8_t>(tmp_a_opt.value_or(0));

  std::vector<uint32_t> words;
  words.reserve(8192);
  append_save_exec(words, exec_save);

  if (zero_acc) {
    for (uint8_t reg = 0; reg < 8; ++reg) {
      append_vop1(words, kOpMovB32, static_cast<uint8_t>(src.vdst + reg),
                  scalar_positive_inline_u32(0));
    }
  } else if (literal_word) {
    for (uint8_t reg = 0; reg < 8; ++reg) {
      append_vop1(words, kOpMovB32, static_cast<uint8_t>(src.vdst + reg), 255, literal_word);
    }
  } else if (!accumulate_in_vdst) {
    append_v_mov_b32_run(words, static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src2), 8);
  }

  if (scalar_a_opt) {
    for (uint8_t k_group = 0; k_group < 32; ++k_group) {
      append_wmma_f32_k128_fp8_scalar_a_group(words, *dot_op, static_cast<uint8_t>(src.vdst),
                                              *src0_base, *src1_base, tmp_b, vaddr, *scalar_a_opt,
                                              k_group, exec_save);
    }
  } else if (grouped_dot4) {
    for (uint8_t k_group = 0; k_group < 32; ++k_group) {
      append_wmma_f32_k128_fp8_dot4_group(words, *dot_op, static_cast<uint8_t>(src.vdst),
                                          *src0_base, *src1_base, tmp_a, tmp_b, vaddr, k_group,
                                          exec_save);
    }
  } else {
    for (uint8_t reg = 0; reg < 8; ++reg) {
      const uint8_t dst_reg = static_cast<uint8_t>(src.vdst + reg);
      for (uint8_t k_group = 0; k_group < 32; ++k_group) {
        append_wmma_f32_k128_fp8_dot4_term(words, *dot_op, dst_reg, *src0_base, *src1_base, tmp_a,
                                           tmp_b, vaddr, reg, k_group, exec_save);
      }
    }
  }

  return words;
}

std::vector<uint32_t>
expand_v_wmma_f16_16x16x128_f8_dot4_fallback(const Instruction &inst,
                                             const LivenessAnalysis &liveness) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3pMachineInst))
    return {};

  gfx1250::Vop3pMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  const auto dot_op = rdna4_dot4_f32_f8_opcode(inst.opcode());
  if (!dot_op || src.vdst > 252 || src.neg_hi != 0 || src.opsel != 0 || src.clamp != 0 ||
      src.neg != 0)
    return {};

  const auto src0_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src0), 16);
  const auto src1_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src1), 16);
  if (!src0_base || !src1_base)
    return {};

  const bool zero_acc = src.src2 == scalar_positive_inline_u32(0);
  std::optional<uint32_t> literal_word;
  std::optional<uint8_t> src2_base;
  if (src.src2 == 255) {
    literal_word = simm32_literal_word(inst, 2);
    if (!literal_word)
      return {};
  } else if (src.src2 == 254) {
    return {};
  } else if (!zero_acc) {
    src2_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src2), 4);
    if (!src2_base)
      return {};
  }

  std::vector<uint8_t> avoid;
  add_avoid_vgpr_run(avoid, static_cast<uint8_t>(src.vdst), 4);
  add_avoid_vgpr_run(avoid, *src0_base, 16);
  add_avoid_vgpr_run(avoid, *src1_base, 16);
  if (src2_base)
    add_avoid_vgpr_run(avoid, *src2_base, 4);

  const auto tmp_acc_opt = find_aligned_free_vgpr_run_avoiding(inst, liveness, 8, 8, avoid);
  if (!tmp_acc_opt || *tmp_acc_opt > 248)
    return {};
  const uint8_t tmp_acc = static_cast<uint8_t>(*tmp_acc_opt);
  add_avoid_vgpr_run(avoid, tmp_acc, 8);

  const auto tmp_b_opt = find_free_vgpr_run_avoiding(inst, liveness, 1, avoid);
  if (!tmp_b_opt || *tmp_b_opt > 255)
    return {};
  const uint8_t tmp_b = static_cast<uint8_t>(*tmp_b_opt);
  add_avoid_vgpr(avoid, tmp_b);

  const auto vaddr_opt = find_free_vgpr_run_avoiding(inst, liveness, 1, avoid);
  if (!vaddr_opt || *vaddr_opt > 255)
    return {};
  const uint8_t vaddr = static_cast<uint8_t>(*vaddr_opt);
  add_avoid_vgpr(avoid, vaddr);

  const auto exec_save_opt = liveness.find_free_sgpr_pair(&inst);
  if (!exec_save_opt || *exec_save_opt > 124)
    return {};
  const uint8_t exec_save = static_cast<uint8_t>(*exec_save_opt);

  const std::vector<uint8_t> avoid_sgpr{exec_save, static_cast<uint8_t>(exec_save + 1u)};
  const auto scalar_a_opt = find_free_sgprs_avoiding<8>(inst, liveness, avoid_sgpr);

  bool grouped_dot4 = true;
  std::optional<uint16_t> tmp_a_opt;
  if (scalar_a_opt) {
    // The f16 result pack still needs one temporary VGPR after the dot4s.
    grouped_dot4 = false;
    tmp_a_opt = find_free_vgpr_run_avoiding(inst, liveness, 1, avoid);
  } else {
    tmp_a_opt = find_free_vgpr_run_avoiding(inst, liveness, 8, avoid);
    if (!tmp_a_opt) {
      grouped_dot4 = false;
      tmp_a_opt = find_free_vgpr_run_avoiding(inst, liveness, 1, avoid);
    }
  }
  if (!tmp_a_opt || *tmp_a_opt > 255)
    return {};
  const uint8_t tmp_a = static_cast<uint8_t>(*tmp_a_opt);

  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpOrB32 = 28;
  constexpr uint16_t kOpCvtF16F32 = 0x18A;
  constexpr uint16_t kOpCvtF32F16 = 0x18B;

  std::vector<uint32_t> words;
  words.reserve(9000);
  append_save_exec(words, exec_save);

  if (zero_acc) {
    append_v_mov_b32_broadcast(words, tmp_acc, scalar_positive_inline_u32(0), 8);
  } else {
    for (uint8_t word = 0; word < 4; ++word) {
      const uint16_t packed = literal_word ? 255u : static_cast<uint16_t>(256u + *src2_base + word);
      append_vop3_mod(words, kOpCvtF32F16, static_cast<uint8_t>(tmp_acc + 2u * word), packed, 0, 0,
                      0, literal_word);
      append_vop3_mod(words, kOpCvtF32F16, static_cast<uint8_t>(tmp_acc + 2u * word + 1u), packed,
                      0, 0, 1, literal_word);
    }
  }
  append_wait_valu_vgpr(words);

  if (scalar_a_opt) {
    for (uint8_t k_group = 0; k_group < 32; ++k_group) {
      append_wmma_f32_k128_fp8_scalar_a_group(words, *dot_op, tmp_acc, *src0_base, *src1_base,
                                              tmp_b, vaddr, *scalar_a_opt, k_group, exec_save);
    }
  } else if (grouped_dot4) {
    for (uint8_t k_group = 0; k_group < 32; ++k_group) {
      append_wmma_f32_k128_fp8_dot4_group(words, *dot_op, tmp_acc, *src0_base, *src1_base, tmp_a,
                                          tmp_b, vaddr, k_group, exec_save);
    }
  } else {
    for (uint8_t reg = 0; reg < 8; ++reg) {
      const uint8_t dst_reg = static_cast<uint8_t>(tmp_acc + reg);
      for (uint8_t k_group = 0; k_group < 32; ++k_group) {
        append_wmma_f32_k128_fp8_dot4_term(words, *dot_op, dst_reg, *src0_base, *src1_base, tmp_a,
                                           tmp_b, vaddr, reg, k_group, exec_save);
      }
    }
  }

  append_restore_exec(words, exec_save);
  for (uint8_t word = 0; word < 4; ++word) {
    append_vop3(words, kOpCvtF16F32, tmp_a, static_cast<uint16_t>(256u + tmp_acc + 2u * word), 0);
    append_vop3(words, kOpCvtF16F32, tmp_b, static_cast<uint16_t>(256u + tmp_acc + 2u * word + 1u),
                0);
    append_wait_valu_vgpr(words);
    std::optional<uint32_t> mask_literal;
    append_vop2(words, kOpAndB32, tmp_a, literal_or_inline_u32(0xFFFFu, mask_literal), tmp_a,
                mask_literal);
    append_vop2(words, kOpLshlrevB32, tmp_b, scalar_positive_inline_u32(16), tmp_b);
    append_wait_valu_vgpr(words);
    append_vop2(words, kOpOrB32, static_cast<uint8_t>(src.vdst + word),
                static_cast<uint16_t>(256u + tmp_a), tmp_b);
  }
  return words;
}

std::vector<uint32_t>
expand_v_wmma_f32_16x16x128_fp8_fp8_private_spill_fallback(const Instruction &inst,
                                                           const LivenessAnalysis &liveness) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3pMachineInst))
    return {};

  gfx1250::Vop3pMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  const auto target_op = rdna4_wmma_f32_f8_opcode(inst.opcode());
  if (!target_op)
    return {};
  if (src.vdst > 248 || src.neg_hi != 0 || src.opsel != 0 || src.clamp != 0 || src.neg != 0)
    return {};

  if (!src_vgpr_base_for_run(static_cast<uint16_t>(src.src0), 16) ||
      !src_vgpr_base_for_run(static_cast<uint16_t>(src.src1), 16))
    return {};
  if (dst_overlaps_wmma_ab_sources(static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src0),
                                   static_cast<uint16_t>(src.src1), 16))
    return {};
  if (src.src2 != static_cast<uint16_t>(256u + static_cast<uint16_t>(src.vdst)))
    return {};

  const auto private_base_opt = liveness.private_spill_base();
  if (!private_base_opt || liveness.private_spill_bytes() < kPrivateBorrowScratchBytes)
    return {};

  std::vector<uint8_t> avoid_vgpr;
  add_avoid_vgpr_run(avoid_vgpr, static_cast<uint8_t>(src.vdst), 8);
  add_avoid_src_vgpr_run(avoid_vgpr, static_cast<uint16_t>(src.src0), 16);
  add_avoid_src_vgpr_run(avoid_vgpr, static_cast<uint16_t>(src.src1), 16);
  const auto scratch_base_opt =
      find_borrowable_low_vgpr_run(kK128Fp8BorrowedVgprCount, 2, avoid_vgpr);
  if (!scratch_base_opt || *scratch_base_opt > 251)
    return {};

  auto exec_save_opt = liveness.find_free_sgpr_pair(&inst);
  if (!exec_save_opt || *exec_save_opt > 124)
    return {};
  const uint8_t exec_save = static_cast<uint8_t>(*exec_save_opt);
  const uint8_t kTmpA = static_cast<uint8_t>(*scratch_base_opt);
  const uint8_t kTmpB = static_cast<uint8_t>(kTmpA + 2u);
  const uint8_t kVaddrXor16 = static_cast<uint8_t>(kTmpA + 4u);
  const uint32_t private_base = *private_base_opt;
  constexpr uint8_t kOpWaitLoadcnt = 64;
  constexpr uint8_t kOpWaitDscnt = 70;
  constexpr uint8_t kOpWaitStorecnt = 65;
  constexpr uint8_t kSoppWaitKmcnt = 0x47;

  std::vector<uint32_t> words;
  words.reserve(360);
  append_save_exec(words, exec_save);
  words.push_back(pack_sopp(kOpWaitLoadcnt, 0));
  words.push_back(pack_sopp(kOpWaitDscnt, 0));
  append_set_exec_lo_mask(words, 0xFFFFFFFFu);
  for (uint8_t reg = 0; reg < kK128Fp8BorrowedVgprCount; ++reg) {
    append_scratch_store_b32(words, static_cast<uint8_t>(kTmpA + reg),
                             private_base + reg * sizeof(uint32_t));
  }
  words.push_back(pack_sopp(kOpWaitStorecnt, 0));

  append_lane_xor16_byte_addr(words, kVaddrXor16);
  append_restore_exec(words, exec_save);
  append_wait_valu_vgpr(words);

  uint16_t acc_src = static_cast<uint16_t>(src.src2);
  for (uint8_t chunk = 0; chunk < 8; ++chunk) {
    append_wmma_2word_relayout_chunk(words, static_cast<uint16_t>(src.src0),
                                     static_cast<uint16_t>(src.src1), kTmpA, kTmpB, kVaddrXor16, 16,
                                     4, chunk, exec_save);
    append_wmma_f32_16x16x16_f8(words, *target_op, static_cast<uint8_t>(src.vdst), kTmpA, kTmpB,
                                acc_src);
    words.push_back(pack_sopp(kSoppWaitKmcnt, 0));
    acc_src = static_cast<uint16_t>(256u + static_cast<uint16_t>(src.vdst));
  }

  append_set_exec_lo_mask(words, 0xFFFFFFFFu);
  for (uint8_t reg = 0; reg < kK128Fp8BorrowedVgprCount; ++reg) {
    append_scratch_load_b32(words, static_cast<uint8_t>(kTmpA + reg),
                            private_base + reg * sizeof(uint32_t));
  }
  words.push_back(pack_sopp(kOpWaitLoadcnt, 0));
  append_restore_exec(words, exec_save);
  append_wait_valu_vgpr(words);
  return words;
}

void append_exec_lane_mask_from_saved(std::vector<uint32_t> &words, uint8_t exec_save,
                                      uint8_t lane) {
  constexpr uint8_t kExecLo = 126;
  append_s_and_b32_lit(words, kExecLo, exec_save, 1u << lane);
  append_wait_salu_sgpr(words);
}

std::vector<uint32_t>
expand_v_wmma_f32_16x16x128_fp8_fp8_scalar_fallback(const Instruction &inst,
                                                    const LivenessAnalysis &liveness) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3pMachineInst))
    return {};

  gfx1250::Vop3pMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  const auto dot_op = rdna4_dot4_f32_f8_opcode(inst.opcode());
  if (!dot_op)
    return {};
  if (src.vdst > 248 || src.neg_hi != 0 || src.opsel != 0 || src.clamp != 0 || src.neg != 0)
    return {};

  const auto src0_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src0), 16);
  const auto src1_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src1), 16);
  if (!src0_base || !src1_base)
    return {};

  if (dst_overlaps_wmma_ab_sources(static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src0),
                                   static_cast<uint16_t>(src.src1), 16))
    return {};

  constexpr uint8_t kOpMovB32 = 1;
  const bool zero_acc = src.src2 == scalar_positive_inline_u32(0);
  const bool accumulate_in_vdst =
      src.src2 == static_cast<uint16_t>(256u + static_cast<uint16_t>(src.vdst));
  std::optional<uint32_t> literal_word;
  std::optional<uint8_t> src2_base;
  if (src.src2 == 255) {
    literal_word = simm32_literal_word(inst, 2);
    if (!literal_word)
      return {};
  } else if (src.src2 == 254) {
    return {};
  } else if (!zero_acc && !accumulate_in_vdst) {
    src2_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src2), 8);
    if (!src2_base || overlaps_vgpr_runs(static_cast<uint16_t>(src.vdst), 8, *src2_base, 8))
      return {};
  }

  auto exec_save_opt = liveness.find_free_sgpr_pair(&inst);
  if (!exec_save_opt || *exec_save_opt > 124)
    return {};
  const uint8_t exec_save = static_cast<uint8_t>(*exec_save_opt);

  std::vector<uint8_t> avoid_sgpr;
  auto add_avoid_sgpr = [&](uint8_t sgpr) {
    if (std::find(avoid_sgpr.begin(), avoid_sgpr.end(), sgpr) == avoid_sgpr.end())
      avoid_sgpr.push_back(sgpr);
  };
  add_avoid_sgpr(exec_save);
  add_avoid_sgpr(static_cast<uint8_t>(exec_save + 1u));

  auto scalar_a_opt = find_free_sgpr_avoiding(inst, liveness, avoid_sgpr);
  if (!scalar_a_opt)
    return {};
  const uint8_t scalar_a = static_cast<uint8_t>(*scalar_a_opt);
  add_avoid_sgpr(scalar_a);

  auto scalar_b_opt = find_free_sgpr_avoiding(inst, liveness, avoid_sgpr);
  if (!scalar_b_opt)
    return {};
  const uint8_t scalar_b = static_cast<uint8_t>(*scalar_b_opt);

  constexpr uint8_t kExecLo = 126;
  std::vector<uint32_t> words;
  words.reserve(32768);
  append_save_exec(words, exec_save);
  append_s_mov_b32_lit(words, kExecLo + 1, 0);
  append_wait_salu_sgpr(words);

  if (zero_acc) {
    for (uint8_t reg = 0; reg < 8; ++reg) {
      append_vop1(words, kOpMovB32, static_cast<uint8_t>(src.vdst + reg),
                  scalar_positive_inline_u32(0));
    }
  } else if (literal_word) {
    for (uint8_t reg = 0; reg < 8; ++reg) {
      append_vop1(words, kOpMovB32, static_cast<uint8_t>(src.vdst + reg), 255, literal_word);
    }
  } else if (!accumulate_in_vdst) {
    append_v_mov_b32_run(words, static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src2), 8);
  }

  for (uint8_t reg = 0; reg < 8; ++reg) {
    const uint8_t dst_reg = static_cast<uint8_t>(src.vdst + reg);
    for (uint8_t k_group = 0; k_group < 32; ++k_group) {
      const uint8_t src_word = static_cast<uint8_t>(k_group & 15u);
      const bool high_k_group = k_group >= 16;

      const uint8_t direct_half_base = high_k_group ? 16 : 0;
      const uint32_t direct_half_mask = high_k_group ? 0xFFFF0000u : 0x0000FFFFu;
      const uint8_t direct_a_lane =
          static_cast<uint8_t>((high_k_group ? 16u : 0u) + direct_half_base + reg);
      append_v_readlane_b32(words, scalar_a, static_cast<uint8_t>(*src0_base + src_word),
                            direct_a_lane);
      append_set_exec_from_saved_mask(words, exec_save, direct_half_mask);
      append_v_dot4_f32_f8(words, *dot_op, dst_reg, scalar_a,
                           static_cast<uint8_t>(*src1_base + src_word),
                           static_cast<uint16_t>(256u + dst_reg));

      const uint8_t scalar_half_base = high_k_group ? 0 : 16;
      const uint8_t scalar_a_lane =
          static_cast<uint8_t>((high_k_group ? 16u : 0u) + scalar_half_base + reg);
      append_v_readlane_b32(words, scalar_a, static_cast<uint8_t>(*src0_base + src_word),
                            scalar_a_lane);
      for (uint8_t col = 0; col < 16; ++col) {
        const uint8_t lane = static_cast<uint8_t>(scalar_half_base + col);
        const uint8_t b_lane = static_cast<uint8_t>((high_k_group ? 16u : 0u) + col);
        append_v_readlane_b32(words, scalar_b, static_cast<uint8_t>(*src1_base + src_word), b_lane);
        append_exec_lane_mask_from_saved(words, exec_save, lane);
        append_v_dot4_f32_f8(words, *dot_op, dst_reg, scalar_a, scalar_b,
                             static_cast<uint16_t>(256u + dst_reg));
      }
    }
  }

  append_restore_exec(words, exec_save);
  return words;
}

std::vector<uint32_t> expand_v_wmma_f32_16x16x128_fp8_fp8(const Instruction &inst, uint32_t,
                                                          uint64_t,
                                                          const LivenessAnalysis &liveness,
                                                          const LaneLayout *, const LaneLayout *) {
  auto words = expand_v_wmma_f32_16x16x128_fp8_fp8_dot4_fallback(inst, liveness);
  if (!words.empty())
    return words;
  words = expand_v_wmma_f32_16x16x128_fp8_fp8_private_spill_fallback(inst, liveness);
  if (!words.empty())
    return words;
  return expand_v_wmma_f32_16x16x128_fp8_fp8_scalar_fallback(inst, liveness);
}

std::vector<uint32_t> expand_v_wmma_f16_16x16x128_f8(const Instruction &inst, uint32_t, uint64_t,
                                                     const LivenessAnalysis &liveness,
                                                     const LaneLayout *, const LaneLayout *) {
  return expand_v_wmma_f16_16x16x128_f8_dot4_fallback(inst, liveness);
}

ExpandResult lower_v_wmma_f32_32x16x128_f4_scalar(const Instruction &inst,
                                                  const LivenessAnalysis &liveness, bool scaled) {
  const auto *raw = inst.raw_encoding();
  const size_t required_words = scaled ? 4u : 2u;
  if (!raw || static_cast<size_t>(inst.size()) < required_words * sizeof(uint32_t))
    return ExpandResult::failed("32x16 FP4 WMMA has a malformed encoding");

  gfx1250::Vop3pMachineInst scale{};
  gfx1250::Vop3pMachineInst src{};
  if (scaled) {
    std::memcpy(&scale, raw, sizeof(scale));
    std::memcpy(&src, raw + 2, sizeof(src));
  } else {
    std::memcpy(&src, raw, sizeof(src));
  }
  const bool scale16 = scaled && scale.op == 0x3Au;
  if (src.encoding != 0xCC || src.op != 0x88 || src.vdst > 240 || src.clamp != 0 ||
      (src.neg & ~0x4u) != 0 || (src.neg_hi & ~0x4u) != 0)
    return ExpandResult::failed("32x16 FP4 WMMA uses unsupported modifiers");
  if (scaled && (scale.encoding != 0xCC || (scale.op != 0x35 && !scale16) ||
                 ((scale.neg & 0x3u) != 0u && (scale.neg & 0x3u) != 2u) ||
                 ((scale.neg_hi & 0x3u) != 0u && (scale.neg_hi & 0x3u) != 2u))) {
    return ExpandResult::failed("scaled 32x16 FP4 WMMA uses unsupported scale formats");
  }

  const auto src0_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src0), 16);
  const auto src1_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src1), 8);
  if (!src0_base || !src1_base)
    return ExpandResult::failed("32x16 FP4 WMMA source fragments are not valid VGPR runs");
  if (overlaps_vgpr_runs(static_cast<uint16_t>(src.vdst), 16, *src0_base, 16) ||
      overlaps_vgpr_runs(static_cast<uint16_t>(src.vdst), 16, *src1_base, 8)) {
    return ExpandResult::failed("32x16 FP4 WMMA destination overlaps a source fragment");
  }

  const auto scale_source_supported = [&](uint16_t encoded) {
    if (!scaled)
      return true;
    return scale16
               ? scalar_inline_zero_src(encoded) || src_vgpr_base_for_run(encoded, 2).has_value()
               : scale_word_source_supported(encoded);
  };
  if (scaled && (!scale_source_supported(static_cast<uint16_t>(scale.src0)) ||
                 !scale_source_supported(static_cast<uint16_t>(scale.src1)))) {
    return ExpandResult::failed("scaled 32x16 FP4 WMMA has unsupported scale operands");
  }

  constexpr uint8_t kOpMovB32 = 1;
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpXorB32 = 29;
  constexpr uint16_t kOpFmacF32 = 299;
  constexpr uint16_t kOpMulF32 = 264;
  const bool zero_acc = src.src2 == scalar_positive_inline_u32(0);
  const bool accumulate_in_vdst =
      src.src2 == static_cast<uint16_t>(256u + static_cast<uint16_t>(src.vdst));
  std::optional<uint32_t> literal_word;
  std::optional<uint8_t> src2_base;
  if (src.src2 == 255) {
    if (scaled)
      return ExpandResult::failed("scaled 32x16 FP4 WMMA does not support a literal accumulator");
    literal_word = simm32_literal_word(inst, 2);
    if (!literal_word)
      return ExpandResult::failed("32x16 FP4 WMMA accumulator literal is missing");
  } else if (src.src2 == 254) {
    return ExpandResult::failed("32x16 FP4 WMMA does not support an m0 accumulator");
  } else if (!zero_acc && !accumulate_in_vdst) {
    src2_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src2), 16);
    if (!src2_base || overlaps_vgpr_runs(static_cast<uint16_t>(src.vdst), 16, *src2_base, 16)) {
      return ExpandResult::failed(
          "32x16 FP4 WMMA accumulator is invalid or overlaps its destination");
    }
  }

  std::vector<uint8_t> avoid;
  add_avoid_vgpr_run(avoid, static_cast<uint8_t>(src.vdst), 16);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src0), 16);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src1), 8);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src2), 16);
  if (scaled) {
    add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(scale.src0), scale16 ? 2 : 1);
    add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(scale.src1), scale16 ? 2 : 1);
  }
  auto scratch_opt = find_free_vgpr_run_avoiding(inst, liveness, 10, avoid);
  if (!scratch_opt || *scratch_opt > 246)
    return ExpandResult::failed("32x16 FP4 WMMA lowering needs ten contiguous temporary VGPRs");
  const uint8_t raw_a = static_cast<uint8_t>(*scratch_opt);
  const uint8_t raw_b = static_cast<uint8_t>(raw_a + 1u);
  const uint8_t contribution = static_cast<uint8_t>(raw_a + 2u);
  const uint8_t vaddr = static_cast<uint8_t>(raw_a + 3u);
  const uint8_t value_a = static_cast<uint8_t>(raw_a + 4u);
  const uint8_t value_b = static_cast<uint8_t>(raw_a + 5u);
  const uint8_t sign = static_cast<uint8_t>(raw_a + 6u);
  const uint8_t exponent = static_cast<uint8_t>(raw_a + 7u);
  const uint8_t mantissa = static_cast<uint8_t>(raw_a + 8u);
  const uint8_t subnormal = static_cast<uint8_t>(raw_a + 9u);

  const auto exec_save_opt = liveness.find_free_sgpr_pair(&inst);
  if (!exec_save_opt || *exec_save_opt > 124)
    return ExpandResult::failed("32x16 FP4 WMMA lowering needs a free SGPR pair to save EXEC");
  const uint8_t exec_save = static_cast<uint8_t>(*exec_save_opt);
  const std::vector<uint8_t> avoid_sgpr{exec_save, static_cast<uint8_t>(exec_save + 1u)};
  const auto predicate_opt = find_free_sgpr_pair_avoiding(inst, liveness, avoid_sgpr);
  if (!predicate_opt || *predicate_opt > 105)
    return ExpandResult::failed("32x16 FP4 WMMA lowering needs a free SGPR predicate pair");
  const uint8_t predicate = static_cast<uint8_t>(*predicate_opt);

  std::vector<uint32_t> words;
  words.reserve(scaled ? 196608u : 131072u);
  append_save_exec(words, exec_save);
  if (zero_acc) {
    for (uint8_t reg = 0; reg < 16; ++reg)
      append_vop1(words, kOpMovB32, static_cast<uint8_t>(src.vdst + reg),
                  scalar_positive_inline_u32(0));
  } else if (literal_word) {
    for (uint8_t reg = 0; reg < 16; ++reg)
      append_vop1(words, kOpMovB32, static_cast<uint8_t>(src.vdst + reg), 255, literal_word);
  } else if (!accumulate_in_vdst) {
    append_v_mov_b32_run(words, static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src2),
                         16);
  }

  const uint8_t c_modifier =
      static_cast<uint8_t>(((src.neg >> 2u) & 1u) | (((src.neg_hi >> 2u) & 1u) << 1u));
  for (uint8_t reg = 0; reg < 16; ++reg) {
    const uint8_t dst_reg = static_cast<uint8_t>(src.vdst + reg);
    if (c_modifier & 2u)
      append_vop2(words, kOpAndB32, dst_reg, 255, dst_reg, 0x7FFFFFFFu);
    if (c_modifier & 1u)
      append_vop2(words, kOpXorB32, dst_reg, 255, dst_reg, 0x80000000u);
  }

  const uint8_t matrix_a_scale_select =
      scaled ? static_cast<uint8_t>((scale.opsel & 1u) | (((scale.opsel >> 2u) & 1u) << 1u)) : 0;
  const uint8_t matrix_b_scale_select =
      scaled ? static_cast<uint8_t>((scale.opsel_hi & 1u) | ((scale.pad_14 & 1u) << 1u)) : 0;
  const uint8_t matrix_a_scale_fmt = scaled ? static_cast<uint8_t>(scale.neg & 3u) : 0;
  const uint8_t matrix_b_scale_fmt = scaled ? static_cast<uint8_t>(scale.neg_hi & 3u) : 0;
  const auto scale_word_for_byte = [&](uint16_t encoded, uint8_t byte_index) {
    if (!scale16 || scalar_inline_zero_src(encoded))
      return encoded;
    const auto base = src_vgpr_base_for_run(encoded, 2);
    return static_cast<uint16_t>(256u + *base + byte_index / 4u);
  };

  for (uint8_t k = 0; k < 128; ++k) {
    append_wmma_gather_packed_field(words, raw_b, contribution, vaddr, *src1_base, 0, false, 4,
                                    wmma_f8f6f4_field_loc(4, 4, k), exec_save);
    append_wmma_lowp_to_f32(words, value_b, raw_b, sign, exponent, mantissa, subnormal, predicate,
                            4);
    for (uint8_t reg = 0; reg < 16; ++reg) {
      append_wmma_f4_32x16_gather_a(words, raw_a, vaddr, *src0_base, reg, k, exec_save);
      append_wmma_lowp_to_f32(words, value_a, raw_a, sign, exponent, mantissa, subnormal, predicate,
                              4);
      const uint8_t dst_reg = static_cast<uint8_t>(src.vdst + reg);
      if (!scaled) {
        append_vop3(words, kOpFmacF32, dst_reg, vgpr_src(value_a), vgpr_src(value_b));
        append_wait_valu_vgpr(words);
        continue;
      }

      append_vop3(words, kOpMulF32, contribution, vgpr_src(value_a), vgpr_src(value_b));
      const uint8_t scale_byte =
          scale16 ? static_cast<uint8_t>(4u * (k >> 6u) + 2u * ((k >> 2u) & 1u) + ((k >> 5u) & 1u))
                  : static_cast<uint8_t>(2u * (k >> 6u) + ((k >> 2u) & 1u));
      append_wmma_f32_apply_scales_and_accumulate(
          words, dst_reg, contribution,
          scale_word_for_byte(static_cast<uint16_t>(scale.src0), scale_byte),
          scale_word_for_byte(static_cast<uint16_t>(scale.src1), scale_byte), raw_a, raw_b, vaddr,
          reg, static_cast<uint8_t>(scale_byte % 4u), static_cast<uint8_t>(scale_byte % 4u),
          matrix_a_scale_select, matrix_b_scale_select, matrix_a_scale_fmt, matrix_b_scale_fmt,
          predicate, exec_save, true);
    }
  }

  append_restore_exec(words, exec_save);
  append_wait_valu_vgpr(words);
  return ExpandResult::success(std::move(words));
}

ExpandResult expand_v_wmma_f32_32x16x128_f4(const Instruction &inst, uint32_t, uint64_t,
                                            const LivenessAnalysis &liveness, TranslationContext &,
                                            const LaneLayout *, const LaneLayout *) {
  return lower_v_wmma_f32_32x16x128_f4_scalar(inst, liveness, false);
}

ExpandResult expand_v_wmma_f32_16x16x128_f8f6f4_scalar(const Instruction &inst,
                                                       const LivenessAnalysis &liveness) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3pMachineInst))
    return ExpandResult::failed("f8f6f4 WMMA has a malformed VOP3P encoding");

  gfx1250::Vop3pMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  const uint8_t matrix_a_fmt = static_cast<uint8_t>(src.opsel);
  const uint8_t matrix_b_fmt = static_cast<uint8_t>((src.pad_14 << 2u) | src.opsel_hi);
  if (matrix_a_fmt > 4u || matrix_b_fmt > 4u || src.vdst > 248 || src.neg_hi != 0 ||
      src.clamp != 0 || src.neg != 0)
    return ExpandResult::failed("f8f6f4 WMMA uses unsupported formats or modifiers");

  const uint8_t bits_a = wmma_f8f6f4_format_bits(matrix_a_fmt);
  const uint8_t bits_b = wmma_f8f6f4_format_bits(matrix_b_fmt);
  const uint8_t src0_count = static_cast<uint8_t>(2u * bits_a);
  const uint8_t src1_count = static_cast<uint8_t>(2u * bits_b);
  const auto src0_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src0), src0_count);
  const auto src1_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src1), src1_count);
  if (!src0_base || !src1_base)
    return ExpandResult::failed("f8f6f4 WMMA source fragments are not valid VGPR runs");
  if (overlaps_vgpr_runs(static_cast<uint16_t>(src.vdst), 8, *src0_base, src0_count) ||
      overlaps_vgpr_runs(static_cast<uint16_t>(src.vdst), 8, *src1_base, src1_count))
    return ExpandResult::failed("f8f6f4 WMMA destination overlaps a source fragment");

  constexpr uint8_t kOpMovB32 = 1;
  const bool zero_acc = src.src2 == scalar_positive_inline_u32(0);
  const bool accumulate_in_vdst =
      src.src2 == static_cast<uint16_t>(256u + static_cast<uint16_t>(src.vdst));
  std::optional<uint32_t> literal_word;
  std::optional<uint8_t> src2_base;
  if (src.src2 == 255) {
    literal_word = simm32_literal_word(inst, 2);
    if (!literal_word)
      return ExpandResult::failed("f8f6f4 WMMA accumulator literal is missing");
  } else if (src.src2 == 254) {
    return ExpandResult::failed("f8f6f4 WMMA does not support an m0 accumulator");
  } else if (!zero_acc && !accumulate_in_vdst) {
    src2_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src2), 8);
    if (!src2_base || overlaps_vgpr_runs(static_cast<uint16_t>(src.vdst), 8, *src2_base, 8))
      return ExpandResult::failed("f8f6f4 WMMA accumulator is invalid or overlaps its destination");
  }

  std::vector<uint8_t> avoid;
  add_avoid_vgpr_run(avoid, static_cast<uint8_t>(src.vdst), 8);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src0), src0_count);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src1), src1_count);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src2), 8);
  auto scratch_opt = find_free_vgpr_run_avoiding(inst, liveness, 10, avoid);
  if (!scratch_opt || *scratch_opt > 246)
    return ExpandResult::failed("f8f6f4 WMMA lowering needs ten contiguous temporary VGPRs");
  const uint8_t raw_a = static_cast<uint8_t>(*scratch_opt);
  const uint8_t raw_b = static_cast<uint8_t>(raw_a + 1u);
  const uint8_t next_word = static_cast<uint8_t>(raw_a + 2u);
  const uint8_t vaddr = static_cast<uint8_t>(raw_a + 3u);
  const uint8_t value_a = static_cast<uint8_t>(raw_a + 4u);
  const uint8_t value_b = static_cast<uint8_t>(raw_a + 5u);
  const uint8_t sign = static_cast<uint8_t>(raw_a + 6u);
  const uint8_t exponent = static_cast<uint8_t>(raw_a + 7u);
  const uint8_t mantissa = static_cast<uint8_t>(raw_a + 8u);
  const uint8_t subnormal = static_cast<uint8_t>(raw_a + 9u);

  const auto exec_save_opt = liveness.find_free_sgpr_pair(&inst);
  if (!exec_save_opt || *exec_save_opt > 124)
    return ExpandResult::failed("f8f6f4 WMMA lowering needs a free SGPR pair to save EXEC");
  const uint8_t exec_save = static_cast<uint8_t>(*exec_save_opt);
  const std::vector<uint8_t> avoid_sgpr{exec_save, static_cast<uint8_t>(exec_save + 1u)};
  const auto predicate_opt = find_free_sgpr_pair_avoiding(inst, liveness, avoid_sgpr);
  if (!predicate_opt || *predicate_opt > 105)
    return ExpandResult::failed("f8f6f4 WMMA lowering needs a free SGPR predicate pair");
  const uint8_t predicate = static_cast<uint8_t>(*predicate_opt);

  constexpr uint16_t kOpFmacF32 = 299;
  std::vector<uint32_t> words;
  words.reserve(65536);
  append_save_exec(words, exec_save);

  if (zero_acc) {
    for (uint8_t reg = 0; reg < 8; ++reg) {
      append_vop1(words, kOpMovB32, static_cast<uint8_t>(src.vdst + reg),
                  scalar_positive_inline_u32(0));
    }
  } else if (literal_word) {
    for (uint8_t reg = 0; reg < 8; ++reg)
      append_vop1(words, kOpMovB32, static_cast<uint8_t>(src.vdst + reg), 255, literal_word);
  } else if (!accumulate_in_vdst) {
    append_v_mov_b32_run(words, static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src2), 8);
  }

  for (uint8_t k = 0; k < 128; ++k) {
    append_wmma_gather_packed_field(words, raw_b, next_word, vaddr, *src1_base, 0, false, bits_b,
                                    wmma_f8f6f4_field_loc(matrix_b_fmt, matrix_a_fmt, k),
                                    exec_save);
    append_wmma_lowp_to_f32(words, value_b, raw_b, sign, exponent, mantissa, subnormal, predicate,
                            matrix_b_fmt);

    for (uint8_t reg = 0; reg < 8; ++reg) {
      append_wmma_gather_packed_field(words, raw_a, next_word, vaddr, *src0_base, reg, true, bits_a,
                                      wmma_f8f6f4_field_loc(matrix_a_fmt, matrix_b_fmt, k),
                                      exec_save);
      append_wmma_lowp_to_f32(words, value_a, raw_a, sign, exponent, mantissa, subnormal, predicate,
                              matrix_a_fmt);
      append_vop3(words, kOpFmacF32, static_cast<uint8_t>(src.vdst + reg), vgpr_src(value_a),
                  vgpr_src(value_b));
      append_wait_valu_vgpr(words);
    }
  }

  append_restore_exec(words, exec_save);
  append_wait_valu_vgpr(words);
  return ExpandResult::success(std::move(words));
}

ExpandResult lower_v_wmma_scale_f32_16x16x128_f8f6f4_scalar(const Instruction &inst,
                                                            const LivenessAnalysis &liveness) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < 2 * sizeof(gfx1250::Vop3pMachineInst))
    return ExpandResult::failed("scaled f8f6f4 WMMA has a malformed VOP3PX2 encoding");

  gfx1250::Vop3pMachineInst scale{};
  gfx1250::Vop3pMachineInst src{};
  std::memcpy(&scale, raw, sizeof(scale));
  std::memcpy(&src, raw + 2, sizeof(src));
  const uint8_t matrix_a_fmt = static_cast<uint8_t>(src.opsel);
  const uint8_t matrix_b_fmt = static_cast<uint8_t>((src.pad_14 << 2u) | src.opsel_hi);
  const uint8_t matrix_a_scale_fmt = static_cast<uint8_t>(scale.neg & 0x3u);
  const uint8_t matrix_b_scale_fmt = static_cast<uint8_t>(scale.neg_hi & 0x3u);
  const bool scale16 = scale.op == 0x3A;
  if (scale.encoding != 0xCC || (scale.op != 0x35 && !scale16) || src.encoding != 0xCC ||
      src.op != 0x33 || matrix_a_fmt > 4u || matrix_b_fmt > 4u || src.vdst > 248 ||
      src.neg_hi != 0 || src.clamp != 0 || src.neg != 0)
    return ExpandResult::failed("scaled f8f6f4 WMMA uses unsupported formats or modifiers");
  if ((matrix_a_scale_fmt != 0 && matrix_a_scale_fmt != 2) ||
      (matrix_b_scale_fmt != 0 && matrix_b_scale_fmt != 2))
    return ExpandResult::failed("scaled f8f6f4 WMMA uses an unsupported scale format");
  const auto scale_source_supported = [&](uint16_t encoded) {
    return scale16
               ? scalar_inline_zero_src(encoded) || src_vgpr_base_for_run(encoded, 2).has_value()
               : scale_word_source_supported(encoded);
  };
  if (!scale_source_supported(static_cast<uint16_t>(scale.src0)) ||
      !scale_source_supported(static_cast<uint16_t>(scale.src1)))
    return ExpandResult::failed("scaled f8f6f4 WMMA has unsupported scale-word operands");

  const uint8_t bits_a = wmma_f8f6f4_format_bits(matrix_a_fmt);
  const uint8_t bits_b = wmma_f8f6f4_format_bits(matrix_b_fmt);
  const uint8_t src0_count = static_cast<uint8_t>(2u * bits_a);
  const uint8_t src1_count = static_cast<uint8_t>(2u * bits_b);
  const auto src0_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src0), src0_count);
  const auto src1_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src1), src1_count);
  if (!src0_base || !src1_base)
    return ExpandResult::failed("scaled f8f6f4 WMMA source fragments are not valid VGPR runs");
  if (overlaps_vgpr_runs(static_cast<uint16_t>(src.vdst), 8, *src0_base, src0_count) ||
      overlaps_vgpr_runs(static_cast<uint16_t>(src.vdst), 8, *src1_base, src1_count))
    return ExpandResult::failed("scaled f8f6f4 WMMA destination overlaps a source fragment");

  constexpr uint8_t kOpMovB32 = 1;
  constexpr uint16_t kOpMulF32 = 264;
  const bool zero_acc = src.src2 == scalar_positive_inline_u32(0);
  const bool accumulate_in_vdst =
      src.src2 == static_cast<uint16_t>(256u + static_cast<uint16_t>(src.vdst));
  std::optional<uint8_t> src2_base;
  if (src.src2 == 255 || src.src2 == 254) {
    return ExpandResult::failed("scaled f8f6f4 WMMA requires a VGPR or zero accumulator");
  } else if (!zero_acc && !accumulate_in_vdst) {
    src2_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src2), 8);
    if (!src2_base || overlaps_vgpr_runs(static_cast<uint16_t>(src.vdst), 8, *src2_base, 8))
      return ExpandResult::failed(
          "scaled f8f6f4 WMMA accumulator is invalid or overlaps its destination");
  }

  std::vector<uint8_t> avoid;
  add_avoid_vgpr_run(avoid, static_cast<uint8_t>(src.vdst), 8);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src0), src0_count);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src1), src1_count);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src2), 8);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(scale.src0), scale16 ? 2 : 1);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(scale.src1), scale16 ? 2 : 1);
  auto scratch_opt = find_free_vgpr_run_avoiding(inst, liveness, 10, avoid);
  if (!scratch_opt || *scratch_opt > 246)
    return ExpandResult::failed("scaled f8f6f4 WMMA lowering needs ten contiguous temporary VGPRs");
  const uint8_t raw_a = static_cast<uint8_t>(*scratch_opt);
  const uint8_t raw_b = static_cast<uint8_t>(raw_a + 1u);
  const uint8_t contribution = static_cast<uint8_t>(raw_a + 2u);
  const uint8_t vaddr = static_cast<uint8_t>(raw_a + 3u);
  const uint8_t value_a = static_cast<uint8_t>(raw_a + 4u);
  const uint8_t value_b = static_cast<uint8_t>(raw_a + 5u);
  const uint8_t sign = static_cast<uint8_t>(raw_a + 6u);
  const uint8_t exponent = static_cast<uint8_t>(raw_a + 7u);
  const uint8_t mantissa = static_cast<uint8_t>(raw_a + 8u);
  const uint8_t subnormal = static_cast<uint8_t>(raw_a + 9u);

  const auto exec_save_opt = liveness.find_free_sgpr_pair(&inst);
  if (!exec_save_opt || *exec_save_opt > 124)
    return ExpandResult::failed("scaled f8f6f4 WMMA lowering needs a free SGPR pair to save EXEC");
  const uint8_t exec_save = static_cast<uint8_t>(*exec_save_opt);
  const std::vector<uint8_t> avoid_sgpr{exec_save, static_cast<uint8_t>(exec_save + 1u)};
  const auto predicate_opt = find_free_sgpr_pair_avoiding(inst, liveness, avoid_sgpr);
  if (!predicate_opt || *predicate_opt > 105)
    return ExpandResult::failed("scaled f8f6f4 WMMA lowering needs a free SGPR predicate pair");
  const uint8_t predicate = static_cast<uint8_t>(*predicate_opt);

  const uint8_t matrix_a_scale_select =
      static_cast<uint8_t>((scale.opsel & 0x1u) | (((scale.opsel >> 2u) & 0x1u) << 1u));
  const uint8_t matrix_b_scale_select =
      static_cast<uint8_t>((scale.opsel_hi & 0x1u) | ((scale.pad_14 & 0x1u) << 1u));
  const auto scale_word_for_byte = [&](uint16_t encoded, uint8_t byte_index) {
    if (!scale16 || scalar_inline_zero_src(encoded))
      return encoded;
    const auto base = src_vgpr_base_for_run(encoded, 2);
    return static_cast<uint16_t>(256u + *base + byte_index / 4u);
  };

  std::vector<uint32_t> words;
  words.reserve(98304);
  append_save_exec(words, exec_save);
  if (zero_acc) {
    for (uint8_t reg = 0; reg < 8; ++reg)
      append_vop1(words, kOpMovB32, static_cast<uint8_t>(src.vdst + reg),
                  scalar_positive_inline_u32(0));
  } else if (!accumulate_in_vdst) {
    append_v_mov_b32_run(words, static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src2), 8);
  }

  for (uint8_t k = 0; k < 128; ++k) {
    append_wmma_gather_packed_field(words, raw_b, contribution, vaddr, *src1_base, 0, false, bits_b,
                                    wmma_f8f6f4_field_loc(matrix_b_fmt, matrix_a_fmt, k),
                                    exec_save);
    append_wmma_lowp_to_f32(words, value_b, raw_b, sign, exponent, mantissa, subnormal, predicate,
                            matrix_b_fmt);

    for (uint8_t reg = 0; reg < 8; ++reg) {
      append_wmma_gather_packed_field(words, raw_a, contribution, vaddr, *src0_base, reg, true,
                                      bits_a, wmma_f8f6f4_field_loc(matrix_a_fmt, matrix_b_fmt, k),
                                      exec_save);
      append_wmma_lowp_to_f32(words, value_a, raw_a, sign, exponent, mantissa, subnormal, predicate,
                              matrix_a_fmt);
      append_vop3(words, kOpMulF32, contribution, vgpr_src(value_a), vgpr_src(value_b));
      uint8_t scale_byte = 0;
      if (scale16) {
        if (matrix_a_fmt > 1u && matrix_b_fmt > 1u) {
          scale_byte =
              static_cast<uint8_t>(4u * (k >> 6u) + 2u * ((k >> 2u) & 1u) + ((k >> 5u) & 1u));
        } else {
          scale_byte = static_cast<uint8_t>(2u * (k >> 5u) + ((k >> 2u) & 1u));
        }
      } else if (matrix_a_fmt > 1u && matrix_b_fmt > 1u) {
        scale_byte = static_cast<uint8_t>(2u * (k >> 6u) + ((k >> 2u) & 1u));
      } else {
        scale_byte = static_cast<uint8_t>(k >> 5u);
      }
      append_wmma_f32_apply_scales_and_accumulate(
          words, static_cast<uint8_t>(src.vdst + reg), contribution,
          scale_word_for_byte(static_cast<uint16_t>(scale.src0), scale_byte),
          scale_word_for_byte(static_cast<uint16_t>(scale.src1), scale_byte), raw_a, raw_b, vaddr,
          reg, static_cast<uint8_t>(scale_byte % 4u), static_cast<uint8_t>(scale_byte % 4u),
          matrix_a_scale_select, matrix_b_scale_select, matrix_a_scale_fmt, matrix_b_scale_fmt,
          predicate, exec_save);
    }
  }

  append_restore_exec(words, exec_save);
  append_wait_valu_vgpr(words);
  return ExpandResult::success(std::move(words));
}

ExpandResult expand_v_wmma_f32_16x16x128_f8f6f4(const Instruction &inst, uint32_t pc,
                                                uint64_t guest_pc, const LivenessAnalysis &liveness,
                                                TranslationContext &,
                                                const LaneLayout *input_layout,
                                                const LaneLayout *output_layout) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3pMachineInst))
    return ExpandResult::failed("f8f6f4 WMMA has a malformed VOP3P encoding");

  gfx1250::Vop3pMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  const uint8_t matrix_a_fmt = static_cast<uint8_t>(src.opsel);
  const uint8_t matrix_b_fmt = static_cast<uint8_t>((src.pad_14 << 2u) | src.opsel_hi);
  if (matrix_a_fmt <= 1u && matrix_b_fmt <= 1u) {
    // The fixed 512-bit f8f6f4 fragments use the same physical layout as the
    // dedicated K128 FP8/BF8 instructions for byte-wide format pairs. Reuse
    // the exact dot4 fallback while selecting the target opcode from the two
    // explicit format fields carried by this instruction.
    constexpr std::array<uint8_t, 4> kDotOps{0x26, 0x24, 0x25, 0x27};
    auto words = expand_v_wmma_f32_16x16x128_fp8_fp8_dot4_fallback(
        inst, liveness, kDotOps[(matrix_a_fmt << 1u) | matrix_b_fmt]);
    if (words.empty())
      return ExpandResult::failed("byte-format f8f6f4 WMMA could not allocate its dot4 lowering");
    return ExpandResult::success(std::move(words));
  }

  auto words = expand_v_wmma_f32_16x16x128_f8f6f4_fp4_fp4(inst, pc, guest_pc, liveness,
                                                          input_layout, output_layout);
  if (!words.empty())
    return ExpandResult::success(std::move(words));
  return expand_v_wmma_f32_16x16x128_f8f6f4_scalar(inst, liveness);
}

enum class PermlaneFamilyOp : uint8_t { Bcast, Up, Down, Xor };

[[nodiscard]] std::optional<PermlaneFamilyOp> permlane_family_op(uint16_t op) {
  switch (op) {
  case 0x270:
    return PermlaneFamilyOp::Bcast;
  case 0x271:
    return PermlaneFamilyOp::Up;
  case 0x272:
    return PermlaneFamilyOp::Down;
  case 0x273:
    return PermlaneFamilyOp::Xor;
  default:
    return std::nullopt;
  }
}

ExpandResult expand_v_permlane_family_b32(const Instruction &inst, uint32_t, uint64_t,
                                          const LivenessAnalysis &liveness,
                                          TranslationContext &context, const LaneLayout *,
                                          const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3MachineInst))
    return ExpandResult::not_handled();

  gfx1250::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  const auto operation = permlane_family_op(src.op);
  if (!operation)
    return ExpandResult::not_handled();
  if (src.abs != 0 || src.neg != 0 || src.opsel != 0 || src.clamp != 0 || src.omod != 0)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " uses unsupported source or result modifiers");

  const auto data_vgpr = vgpr_index(static_cast<uint16_t>(src.src0));
  if (!data_vgpr)
    return ExpandResult::failed(std::string(inst.mnemonic()) + " requires a VGPR data source");
  if (src.src1 == 254 || src.src2 == 254)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " does not support literal64 selector operands");

  const bool selector_uses_literal = src.src1 == 255;
  const bool width_uses_literal = src.src2 == 255;
  const auto selector_literal = selector_uses_literal ? simm32_literal_word(inst, 1) : std::nullopt;
  const auto width_literal = width_uses_literal ? simm32_literal_word(inst, 2) : std::nullopt;
  if ((selector_uses_literal && !selector_literal) || (width_uses_literal && !width_literal))
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " is missing its literal32 selector operand");

  std::vector<uint8_t> avoid;
  add_avoid_vgpr(avoid, static_cast<uint8_t>(src.vdst));
  add_avoid_src_vgpr(avoid, static_cast<uint16_t>(src.src0));
  const auto scratch_base = find_free_vgpr_run_avoiding(inst, liveness, 4, avoid);
  if (!scratch_base || *scratch_base > 252)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " cannot allocate four VGPRs for lane-address lowering");

  const uint8_t lane = static_cast<uint8_t>(*scratch_base);
  const uint8_t group_mask = static_cast<uint8_t>(lane + 1u);
  const uint8_t group_base = static_cast<uint8_t>(lane + 2u);
  const uint8_t valid_mask = static_cast<uint8_t>(lane + 3u);

  constexpr uint8_t kOpMovB32 = 1;
  constexpr uint8_t kOpNotB32 = 55;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpAshrrevI32 = 26;
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpOrB32 = 28;
  constexpr uint8_t kOpXorB32 = 29;
  constexpr uint8_t kOpAddNcU32 = 37;
  constexpr uint8_t kOpSubrevNcU32 = 39;
  constexpr uint16_t kOpMbcntLoU32B32 = 0x31F;
  constexpr uint8_t kOpWaitDscnt = 70;
  constexpr uint16_t kInline0 = 128;
  constexpr uint16_t kInline1 = 129;
  constexpr uint16_t kInline2 = 130;
  constexpr uint16_t kInline31 = 159;
  constexpr uint16_t kInlineMinus1 = 193;

  std::vector<uint32_t> words;
  words.reserve(40);

  // A wave32 lane id is mbcnt(-1, 0). The family ISA contract requires a
  // non-zero power-of-two lane-group width, so width-1 is both the within-group
  // mask and the basis for the group base.
  append_vop3(words, kOpMbcntLoU32B32, lane, kInlineMinus1, kInline0);
  append_vop1(words, kOpMovB32, group_mask, static_cast<uint16_t>(src.src2), width_literal);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpAddNcU32, group_mask, kInlineMinus1, group_mask);
  append_wait_valu_vgpr(words);
  append_vop1(words, kOpNotB32, group_base, vgpr_src(group_mask));
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpAndB32, group_base, vgpr_src(lane), group_base);
  append_vop2(words, kOpAndB32, lane, vgpr_src(group_mask), lane);
  append_wait_valu_vgpr(words);

  switch (*operation) {
  case PermlaneFamilyOp::Bcast:
    // selector modulo lane_group_width
    append_vop2(words, kOpAndB32, lane, static_cast<uint16_t>(src.src1), group_mask,
                selector_literal);
    break;
  case PermlaneFamilyOp::Down:
    append_vop2(words, kOpAddNcU32, lane, static_cast<uint16_t>(src.src1), lane, selector_literal);
    break;
  case PermlaneFamilyOp::Up:
    append_vop2(words, kOpSubrevNcU32, lane, static_cast<uint16_t>(src.src1), lane,
                selector_literal);
    break;
  case PermlaneFamilyOp::Xor:
    append_vop2(words, kOpXorB32, lane, static_cast<uint16_t>(src.src1), lane, selector_literal);
    break;
  }
  append_wait_valu_vgpr(words);

  const bool can_leave_group = *operation != PermlaneFamilyOp::Bcast;
  if (can_leave_group) {
    // Remember whether the unmasked source offset leaves its lane group. Turn
    // that non-zero bit-set into an all-zero/all-one result mask without
    // borrowing or clobbering VCC.
    append_vop1(words, kOpNotB32, valid_mask, vgpr_src(group_mask));
    append_wait_valu_vgpr(words);
    append_vop2(words, kOpAndB32, valid_mask, vgpr_src(lane), valid_mask);
  }

  append_vop2(words, kOpAndB32, lane, vgpr_src(group_mask), lane);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpOrB32, lane, vgpr_src(group_base), lane);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpLshlrevB32, lane, kInline2, lane);
  append_wait_valu_vgpr(words);

  const auto [ds0, ds1] = build_ds_bpermute_fi(static_cast<uint8_t>(src.vdst), lane, *data_vgpr);
  words.push_back(ds0);
  words.push_back(ds1);

  if (can_leave_group) {
    append_vop1(words, kOpNotB32, group_mask, vgpr_src(valid_mask));
    append_wait_valu_vgpr(words);
    append_vop2(words, kOpAddNcU32, group_mask, kInline1, group_mask);
    append_wait_valu_vgpr(words);
    append_vop2(words, kOpOrB32, valid_mask, vgpr_src(group_mask), valid_mask);
    append_wait_valu_vgpr(words);
    append_vop2(words, kOpAshrrevI32, valid_mask, kInline31, valid_mask);
    append_wait_valu_vgpr(words);
    append_vop1(words, kOpNotB32, valid_mask, vgpr_src(valid_mask));
  }

  words.push_back(pack_sopp(kOpWaitDscnt, 0));
  if (can_leave_group) {
    append_vop2(words, kOpAndB32, static_cast<uint8_t>(src.vdst), vgpr_src(valid_mask),
                static_cast<uint8_t>(src.vdst));
  }

  context.require_vgprs(static_cast<uint32_t>(lane) + 4u);
  return ExpandResult::success(std::move(words));
}

ExpandResult expand_v_permlane_idx_gen_b32(const Instruction &inst, uint32_t, uint64_t,
                                           const LivenessAnalysis &, TranslationContext &,
                                           const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3MachineInst))
    return ExpandResult::not_handled();

  gfx1250::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.op != 0x314)
    return ExpandResult::not_handled();
  if (src.abs != 0 || src.neg != 0 || src.opsel != 0 || src.clamp != 0 || src.omod != 0)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " uses unsupported source or result modifiers");
  const auto data_vgpr = vgpr_index(static_cast<uint16_t>(src.src0));
  if (!data_vgpr)
    return ExpandResult::failed(std::string(inst.mnemonic()) + " requires a VGPR data source");
  if (src.src1 == 254)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " does not support a literal64 selector");
  const auto literal = src.src1 == 255 ? simm32_literal_word(inst, 1) : std::nullopt;
  if (src.src1 == 255 && !literal)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " is missing its literal32 selector");

  constexpr uint8_t kOpXorB32 = 29;
  std::vector<uint32_t> words;
  append_vop2(words, kOpXorB32, static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src1),
              *data_vgpr, literal);
  return ExpandResult::success(std::move(words));
}

ExpandResult expand_v_bitop3_b32_vop3(const Instruction &inst, uint32_t, uint64_t,
                                      const LivenessAnalysis &liveness, TranslationContext &context,
                                      const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3MachineInst))
    return ExpandResult::not_handled();

  gfx1250::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  const uint8_t truth_table = static_cast<uint8_t>((src.omod << 6) | (src.abs << 3) | src.neg);
  if ((truth_table != 0xC8 && truth_table != 0x6C && truth_table != 0x78) || src.opsel != 0 ||
      src.clamp != 0)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " uses an unsupported truth table or modifier combination");
  if (src.src0 == 254 || src.src1 == 254 || src.src2 == 254)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " does not support literal64 operands");

  if (truth_table == 0x6C || truth_table == 0x78) {
    // ttbl 0x6c is S1 ^ (S0 & S2), while 0x78 is S0 ^ (S1 & S2). Compute the
    // product in liveness-proven scratch so either XOR input may alias VDST.
    const uint16_t product_src0 =
        truth_table == 0x6C ? static_cast<uint16_t>(src.src0) : static_cast<uint16_t>(src.src1);
    const uint16_t product_src1 = static_cast<uint16_t>(src.src2);
    const uint16_t xor_src =
        truth_table == 0x6C ? static_cast<uint16_t>(src.src1) : static_cast<uint16_t>(src.src0);
    const uint8_t product_src0_operand = truth_table == 0x6C ? 0 : 1;
    const uint8_t product_src1_operand = 2;
    const uint8_t xor_src_operand = truth_table == 0x6C ? 1 : 0;
    uint16_t and_src0 = 0;
    uint8_t and_vsrc1 = 0;
    uint8_t and_src0_operand = 0;
    if (const auto src0_vgpr = vgpr_index(product_src0)) {
      and_src0 = product_src1;
      and_vsrc1 = *src0_vgpr;
      and_src0_operand = product_src1_operand;
    } else if (const auto src1_vgpr = vgpr_index(product_src1)) {
      and_src0 = product_src0;
      and_vsrc1 = *src1_vgpr;
      and_src0_operand = product_src0_operand;
    } else {
      return ExpandResult::failed(std::string(inst.mnemonic()) +
                                  " needs a VGPR product operand for XOR-AND lowering");
    }

    std::vector<uint8_t> avoid;
    add_avoid_vgpr(avoid, static_cast<uint8_t>(src.vdst));
    add_avoid_src_vgpr(avoid, static_cast<uint16_t>(src.src0));
    add_avoid_src_vgpr(avoid, static_cast<uint16_t>(src.src1));
    add_avoid_src_vgpr(avoid, static_cast<uint16_t>(src.src2));
    const auto tmp_opt = find_free_vgpr_run_avoiding(inst, liveness, 1, avoid);
    if (!tmp_opt || *tmp_opt > 255)
      return ExpandResult::failed(std::string(inst.mnemonic()) +
                                  " cannot allocate a VGPR for XOR-AND lowering");
    const uint8_t tmp = static_cast<uint8_t>(*tmp_opt);

    const bool and_uses_literal = and_src0 == 255;
    const bool xor_uses_literal = xor_src == 255;
    const auto literal_word =
        and_uses_literal
            ? simm32_literal_word(inst, and_src0_operand)
            : (xor_uses_literal ? simm32_literal_word(inst, xor_src_operand) : std::nullopt);
    if ((and_uses_literal || xor_uses_literal) && !literal_word)
      return ExpandResult::failed(std::string(inst.mnemonic()) +
                                  " is missing its literal32 operand");

    constexpr uint8_t kOpAndB32 = 27;
    constexpr uint8_t kOpXorB32 = 29;
    std::vector<uint32_t> words;
    words.reserve(5);
    append_vop2(words, kOpAndB32, tmp, and_src0, and_vsrc1,
                and_uses_literal ? literal_word : std::nullopt);
    append_wait_valu_vgpr(words);
    append_vop2(words, kOpXorB32, static_cast<uint8_t>(src.vdst), xor_src, tmp,
                xor_uses_literal ? literal_word : std::nullopt);
    context.require_vgprs(static_cast<uint32_t>(tmp) + 1u);
    return ExpandResult::success(std::move(words));
  }

  uint16_t or_src0 = 0;
  uint8_t or_vsrc1 = 0;
  if (auto src2_vgpr = vgpr_index(static_cast<uint16_t>(src.src2))) {
    or_src0 = static_cast<uint16_t>(src.src0);
    or_vsrc1 = *src2_vgpr;
  } else if (auto src0_vgpr = vgpr_index(static_cast<uint16_t>(src.src0))) {
    or_src0 = static_cast<uint16_t>(src.src2);
    or_vsrc1 = *src0_vgpr;
  } else {
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " needs a VGPR S0 or S2 operand for ttbl 0xc8 lowering");
  }

  const bool or_uses_literal = or_src0 == 255;
  const bool and_uses_literal = src.src1 == 255;
  if (or_uses_literal && and_uses_literal)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " cannot encode two distinct literal operands");
  const auto literal_word = or_uses_literal
                                ? simm32_literal_word(inst, src.src0 == 255 ? 0 : 2)
                                : (and_uses_literal ? simm32_literal_word(inst, 1) : std::nullopt);
  if ((or_uses_literal || and_uses_literal) && !literal_word)
    return ExpandResult::failed(std::string(inst.mnemonic()) + " is missing its literal32 operand");

  const auto src1_vgpr = vgpr_index(static_cast<uint16_t>(src.src1));
  if (src1_vgpr && *src1_vgpr == src.vdst)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " ttbl 0xc8 lowering would clobber aliased S1");

  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpOrB32 = 28;
  std::vector<uint32_t> words;
  words.reserve(literal_word ? 3 : 2);
  words.push_back(build_vop2(kOpOrB32, static_cast<uint8_t>(src.vdst), or_src0, or_vsrc1));
  if (or_uses_literal)
    words.push_back(*literal_word);
  words.push_back(build_vop2(kOpAndB32, static_cast<uint8_t>(src.vdst),
                             static_cast<uint16_t>(src.src1), static_cast<uint8_t>(src.vdst)));
  if (and_uses_literal)
    words.push_back(*literal_word);
  return ExpandResult::success(std::move(words));
}

[[nodiscard]] constexpr uint16_t vgpr_src(uint8_t vgpr) {
  return static_cast<uint16_t>(256u + vgpr);
}

[[nodiscard]] uint16_t literal_or_inline_u32(uint32_t value, std::optional<uint32_t> &literal) {
  if (value <= 64)
    return scalar_positive_inline_u32(static_cast<uint8_t>(value));
  literal = value;
  return 255;
}

[[nodiscard]] bool append_materialize_b16_half(std::vector<uint32_t> &words, uint8_t tmp,
                                               uint16_t src, bool high_half,
                                               std::optional<uint32_t> literal_word) {
  constexpr uint8_t kOpMovB32 = 1;
  constexpr uint8_t kOpLshrrevB32 = 25;
  constexpr uint8_t kOpOrB32 = 28;

  if (src == 254)
    return false;

  if (src == 255) {
    if (!literal_word)
      return false;
    const uint32_t half = high_half ? ((*literal_word >> 16) & 0xFFFFu) : (*literal_word & 0xFFFFu);
    std::optional<uint32_t> half_literal;
    append_vop1(words, kOpMovB32, tmp, literal_or_inline_u32(half, half_literal), half_literal);
    return true;
  }

  if (auto src_vgpr = vgpr_index(src)) {
    if (high_half) {
      append_vop2(words, kOpLshrrevB32, tmp, scalar_positive_inline_u32(16), *src_vgpr);
    } else {
      append_vop2(words, kOpOrB32, tmp, scalar_positive_inline_u32(0), *src_vgpr);
    }
    return true;
  }

  append_vop1(words, kOpMovB32, tmp, src);
  if (high_half)
    append_vop2(words, kOpLshrrevB32, tmp, scalar_positive_inline_u32(16), tmp);
  return true;
}

[[nodiscard]] constexpr uint16_t f32_bits_to_bf16_bits(uint32_t bits) {
  const uint32_t lsb = (bits >> 16) & 1u;
  return static_cast<uint16_t>((bits + 0x7FFFu + lsb) >> 16);
}

[[nodiscard]] constexpr std::optional<uint32_t> fma_mix_bf16_inline_f32_bits(uint16_t src) {
  switch (src) {
  case 240:
    return static_cast<uint32_t>(f32_bits_to_bf16_bits(0x3F000000u)) << 16; // 0.5f
  case 241:
    return static_cast<uint32_t>(f32_bits_to_bf16_bits(0xBF000000u)) << 16; // -0.5f
  case 242:
    return static_cast<uint32_t>(f32_bits_to_bf16_bits(0x3F800000u)) << 16; // 1.0f
  case 243:
    return static_cast<uint32_t>(f32_bits_to_bf16_bits(0xBF800000u)) << 16; // -1.0f
  case 244:
    return static_cast<uint32_t>(f32_bits_to_bf16_bits(0x40000000u)) << 16; // 2.0f
  case 245:
    return static_cast<uint32_t>(f32_bits_to_bf16_bits(0xC0000000u)) << 16; // -2.0f
  case 246:
    return static_cast<uint32_t>(f32_bits_to_bf16_bits(0x40800000u)) << 16; // 4.0f
  case 247:
    return static_cast<uint32_t>(f32_bits_to_bf16_bits(0xC0800000u)) << 16; // -4.0f
  case 248:
    return static_cast<uint32_t>(f32_bits_to_bf16_bits(0x3E22F983u)) << 16; // 1/(2*pi)
  default:
    return std::nullopt;
  }
}

[[nodiscard]] bool append_materialize_fma_mix_bf16_source(std::vector<uint32_t> &words, uint8_t tmp,
                                                          uint16_t src, bool high_half,
                                                          std::optional<uint32_t> literal_word) {
  constexpr uint8_t kOpMovB32 = 1;
  constexpr uint8_t kOpLshlrevB32 = 24;

  if (const auto inline_bits = fma_mix_bf16_inline_f32_bits(src)) {
    std::optional<uint32_t> literal;
    append_vop1(words, kOpMovB32, tmp, literal_or_inline_u32(*inline_bits, literal), literal);
    return true;
  }

  if (!append_materialize_b16_half(words, tmp, src, high_half, literal_word))
    return false;
  append_vop2(words, kOpLshlrevB32, tmp, scalar_positive_inline_u32(16), tmp);
  return true;
}

std::vector<uint32_t> lower_v_cvt_f32_bf16_to_shift(uint8_t vdst, uint16_t src0, bool high_half,
                                                    std::optional<uint32_t> literal_word) {
  if (vdst == 255 || src0 == 254)
    return {};

  constexpr uint8_t kOpLshlrevB32 = 24;
  std::vector<uint32_t> words;
  words.reserve(high_half ? 3 : 2);
  if (!append_materialize_b16_half(words, vdst, src0, high_half, literal_word))
    return {};
  append_vop2(words, kOpLshlrevB32, vdst, scalar_positive_inline_u32(16), vdst);
  return words;
}

std::vector<uint32_t> expand_v_cvt_f32_bf16_vop1(const Instruction &inst, uint32_t, uint64_t,
                                                 const LivenessAnalysis &, const LaneLayout *,
                                                 const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() != static_cast<int>(sizeof(uint32_t)))
    return {};

  const auto src = std::bit_cast<gfx1250::Vop1MachineInst>(raw[0]);
  std::optional<uint32_t> literal_word;
  if (src.src0 == 255) {
    literal_word = simm32_literal_word(inst, 0);
    if (!literal_word)
      return {};
  }

  return lower_v_cvt_f32_bf16_to_shift(static_cast<uint8_t>(src.vdst),
                                       static_cast<uint16_t>(src.src0),
                                       /*high_half=*/false, literal_word);
}

std::vector<uint32_t> expand_v_cvt_f32_bf16_vop3(const Instruction &inst, uint32_t, uint64_t,
                                                 const LivenessAnalysis &, const LaneLayout *,
                                                 const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3MachineInst))
    return {};

  gfx1250::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.abs != 0 || (src.opsel & ~0x1u) != 0 || src.clamp != 0 || src.omod != 0 || src.neg != 0)
    return {};

  std::optional<uint32_t> literal_word;
  if (src.src0 == 255) {
    literal_word = simm32_literal_word(inst, 0);
    if (!literal_word)
      return {};
  }

  return lower_v_cvt_f32_bf16_to_shift(static_cast<uint8_t>(src.vdst),
                                       static_cast<uint16_t>(src.src0),
                                       /*high_half=*/(src.opsel & 0x1u) != 0, literal_word);
}

std::vector<uint32_t> expand_v_cvt_f32_fp8_e5m3_vop3(const Instruction &inst, uint32_t, uint64_t,
                                                     const LivenessAnalysis &liveness,
                                                     const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3MachineInst))
    return {};

  gfx1250::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  // Gfx1250 overloads CLAMP as the unsigned E5M3 format selector. RDNA4 does
  // not have that format, so only this selected form needs software lowering.
  if (src.clamp == 0)
    return {};
  if (src.vdst == 255 || src.src0 == 254 || src.abs != 0 || (src.opsel & ~0x3u) != 0 ||
      src.omod != 0 || src.neg != 0)
    return {};

  std::optional<uint32_t> literal_word;
  if (src.src0 == 255) {
    literal_word = simm32_literal_word(inst, 0);
    if (!literal_word)
      return {};
  }

  std::vector<uint8_t> avoid{static_cast<uint8_t>(src.vdst)};
  add_avoid_src_vgpr(avoid, static_cast<uint16_t>(src.src0));
  const auto tmp_base = find_free_vgpr_run_avoiding(inst, liveness, 4, avoid);
  const auto pred_sgpr = liveness.find_free_sgpr_pair(&inst);
  if (!tmp_base || *tmp_base + 3u > 255u || !pred_sgpr || *pred_sgpr > 105u)
    return {};

  const auto byte = static_cast<uint8_t>(*tmp_base);
  const auto exponent = static_cast<uint8_t>(*tmp_base + 1u);
  const auto normal = static_cast<uint8_t>(*tmp_base + 2u);
  const auto subnormal = static_cast<uint8_t>(*tmp_base + 3u);
  const auto predicate = static_cast<uint8_t>(*pred_sgpr);
  const uint8_t byte_index =
      static_cast<uint8_t>(((src.opsel & 0x1u) << 1u) | ((src.opsel & 0x2u) >> 1u));

  constexpr uint8_t kOpMovB32 = 1;
  constexpr uint8_t kOpCvtF32U32 = 6;
  constexpr uint8_t kOpMulF32 = 8;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpLshrrevB32 = 25;
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpAddNcU32 = 37;
  constexpr uint16_t kOpCmpEqU32 = 74;
  constexpr uint16_t kOpCndmaskB32 = 257;
  constexpr uint8_t kSoppWaitAlu = 8;
  constexpr uint32_t kByteMask = 0xFFu;
  constexpr uint32_t kMantissaMask = 0x7u;
  constexpr uint32_t kNormalBias = 0x38000000u;
  constexpr uint32_t kSubnormalScale = 0x37000000u; // 2^-17
  constexpr uint32_t kQuietNaN = 0x7FC00000u;

  std::vector<uint32_t> words;
  words.reserve(32);
  append_vop1(words, kOpMovB32, byte, static_cast<uint16_t>(src.src0), literal_word);
  if (byte_index != 0) {
    append_wait_valu_vgpr(words);
    append_vop2(words, kOpLshrrevB32, byte,
                scalar_positive_inline_u32(static_cast<uint16_t>(byte_index * 8u)), byte);
  }
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpAndB32, byte, 255, byte, kByteMask);

  append_wait_valu_vgpr(words);
  append_vop2(words, kOpLshlrevB32, normal, scalar_positive_inline_u32(20), byte);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpAddNcU32, normal, 255, normal, kNormalBias);

  append_vop2(words, kOpLshrrevB32, exponent, scalar_positive_inline_u32(3), byte);
  append_vop2(words, kOpAndB32, subnormal, 255, byte, kMantissaMask);
  append_wait_valu_vgpr(words);
  append_vop1(words, kOpCvtF32U32, subnormal, static_cast<uint16_t>(256u + subnormal));
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpMulF32, subnormal, 255, subnormal, kSubnormalScale);

  append_wait_valu_vgpr(words);
  append_vop3(words, kOpCmpEqU32, predicate, static_cast<uint16_t>(256u + exponent),
              scalar_positive_inline_u32(0));
  words.push_back(pack_sopp(kSoppWaitAlu, kWaitAluDepctrVaSdst0));
  append_wait_valu_vgpr(words);
  append_vop3(words, kOpCndmaskB32, normal, static_cast<uint16_t>(256u + normal),
              static_cast<uint16_t>(256u + subnormal), predicate);

  append_wait_valu_vgpr(words);
  append_vop3(words, kOpCmpEqU32, predicate, static_cast<uint16_t>(256u + byte), 255, 0, kByteMask);
  words.push_back(pack_sopp(kSoppWaitAlu, kWaitAluDepctrVaSdst0));
  append_wait_valu_vgpr(words);
  append_vop3(words, kOpCndmaskB32, static_cast<uint8_t>(src.vdst),
              static_cast<uint16_t>(256u + normal), 255, predicate, kQuietNaN);
  return words;
}

void append_f32_to_e5m3_rne(std::vector<uint32_t> &words, uint8_t result, uint8_t value,
                            uint8_t abs_bits, uint8_t normal, uint8_t subnormal, uint8_t tmp,
                            uint8_t predicate) {
  constexpr uint8_t kOpCvtU32F32 = 7;
  constexpr uint8_t kOpMulF32 = 8;
  constexpr uint8_t kOpLshrrevB32 = 25;
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpAddNcU32 = 37;
  constexpr uint16_t kOpCmpGtU32 = 76;
  constexpr uint16_t kOpCmpGeU32 = 78;
  constexpr uint16_t kOpCndmaskB32 = 257;
  constexpr uint8_t kSoppWaitAlu = 8;
  constexpr uint32_t kAbsMask = 0x7FFFFFFFu;
  constexpr uint32_t kMantissaRoundBias = 0x0007FFFFu;
  constexpr uint32_t kMinNormal = 0x38800000u; // 2^-14
  constexpr uint32_t kInf = 0x7F800000u;
  constexpr uint32_t kSubnormalScale = 0x48000000u; // 2^17

  append_vop2(words, kOpAndB32, abs_bits, 255, value, kAbsMask);
  append_wait_valu_vgpr(words);

  // Normal RNE: retain exponent plus three mantissa bits, using the retained
  // LSB to implement ties-to-even before rebiasing from f32 (127) to E5M3 (15).
  append_vop2(words, kOpLshrrevB32, tmp, scalar_positive_inline_u32(20), abs_bits);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpAndB32, tmp, scalar_positive_inline_u32(1), tmp);
  append_vop2(words, kOpAddNcU32, normal, 255, abs_bits, kMantissaRoundBias);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpAddNcU32, normal, vgpr_src(tmp), normal);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpLshrrevB32, normal, scalar_positive_inline_u32(20), normal);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpAddNcU32, normal, 255, normal, 0u - 0x380u);

  // E5M3 subnormals are integral multiples of 2^-17.
  append_vop2(words, kOpMulF32, subnormal, 255, abs_bits, kSubnormalScale);
  append_wait_valu_vgpr(words);
  append_vop1(words, kOpCvtU32F32, subnormal, vgpr_src(subnormal));

  append_wait_valu_vgpr(words);
  append_vop3(words, kOpCmpGeU32, predicate, vgpr_src(abs_bits), 255, 0, kMinNormal);
  words.push_back(pack_sopp(kSoppWaitAlu, kWaitAluDepctrVaSdst0));
  append_wait_valu_vgpr(words);
  append_vop3(words, kOpCndmaskB32, result, vgpr_src(subnormal), vgpr_src(normal), predicate);

  // E5M3 has no infinity: all overflow and IEEE NaN/Inf inputs map to 0xFF.
  append_wait_valu_vgpr(words);
  append_vop3(words, kOpCmpGtU32, predicate, vgpr_src(result), 255, 0, 0xFEu);
  words.push_back(pack_sopp(kSoppWaitAlu, kWaitAluDepctrVaSdst0));
  append_wait_valu_vgpr(words);
  append_vop3(words, kOpCndmaskB32, result, vgpr_src(result), 255, predicate, 0xFFu);
  append_wait_valu_vgpr(words);
  append_vop3(words, kOpCmpGeU32, predicate, vgpr_src(abs_bits), 255, 0, kInf);
  words.push_back(pack_sopp(kSoppWaitAlu, kWaitAluDepctrVaSdst0));
  append_wait_valu_vgpr(words);
  append_vop3(words, kOpCndmaskB32, result, vgpr_src(result), 255, predicate, 0xFFu);
}

void append_f32_to_e5m3_sr(std::vector<uint32_t> &words, uint8_t result, uint8_t value,
                           uint8_t seed, uint8_t abs_bits, uint8_t normal, uint8_t subnormal,
                           uint8_t shift, uint8_t full_mant, uint8_t work, uint8_t predicate) {
  constexpr uint8_t kOpMinU32 = 19;
  constexpr uint8_t kOpLshrrevB32 = 25;
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpOrB32 = 28;
  constexpr uint8_t kOpAddNcU32 = 37;
  constexpr uint8_t kOpSubNcU32 = 38;
  constexpr uint16_t kOpLshlrevB32Vop3 = 280;
  constexpr uint16_t kOpLshrrevB32Vop3 = 281;
  constexpr uint16_t kOpCmpGtU32 = 76;
  constexpr uint16_t kOpCmpGeU32 = 78;
  constexpr uint16_t kOpCndmaskB32 = 257;
  constexpr uint8_t kSoppWaitAlu = 8;
  constexpr uint32_t kAbsMask = 0x7FFFFFFFu;
  constexpr uint32_t kMantissaMask = 0x000FFFFFu;
  constexpr uint32_t kFullMantissaMask = 0x007FFFFFu;
  constexpr uint32_t kImplicitOne = 0x00800000u;
  constexpr uint32_t kMinNormal = 0x38800000u;
  constexpr uint32_t kInf = 0x7F800000u;

  append_vop2(words, kOpAndB32, abs_bits, 255, value, kAbsMask);
  append_wait_valu_vgpr(words);

  // Normal stochastic rounding: add the high 20 seed bits to the discarded
  // 20-bit mantissa suffix and carry into the retained E5M3 code.
  append_vop2(words, kOpLshrrevB32, normal, scalar_positive_inline_u32(20), abs_bits);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpAddNcU32, normal, 255, normal, 0u - 0x380u);
  append_vop2(words, kOpAndB32, full_mant, 255, abs_bits, kMantissaMask);
  append_vop2(words, kOpLshrrevB32, work, scalar_positive_inline_u32(12), seed);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpAddNcU32, full_mant, vgpr_src(work), full_mant);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpLshrrevB32, full_mant, scalar_positive_inline_u32(20), full_mant);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpAddNcU32, normal, vgpr_src(full_mant), normal);

  // Subnormal stochastic rounding follows the ISA definition with the
  // exponent-dependent discarded-bit width (shift = 133 - f32_exp_field).
  append_vop2(words, kOpLshrrevB32, shift, scalar_positive_inline_u32(23), abs_bits);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpSubNcU32, shift, 255, shift, 133u);
  append_vop2(words, kOpAndB32, full_mant, 255, abs_bits, kFullMantissaMask);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpOrB32, full_mant, 255, full_mant, kImplicitOne);
  append_wait_valu_vgpr(words);
  append_vop3(words, kOpLshrrevB32Vop3, subnormal, vgpr_src(shift), vgpr_src(full_mant));
  append_wait_valu_vgpr(words);
  append_vop3(words, kOpLshlrevB32Vop3, work, vgpr_src(shift), vgpr_src(subnormal));
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpSubNcU32, full_mant, vgpr_src(full_mant), work);
  append_vop2(words, kOpSubNcU32, work, scalar_positive_inline_u32(32), shift);
  append_wait_valu_vgpr(words);
  append_vop3(words, kOpLshrrevB32Vop3, work, vgpr_src(work), vgpr_src(seed));
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpAddNcU32, full_mant, vgpr_src(work), full_mant);
  append_wait_valu_vgpr(words);
  append_vop3(words, kOpLshrrevB32Vop3, full_mant, vgpr_src(shift), vgpr_src(full_mant));
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpAddNcU32, subnormal, vgpr_src(full_mant), subnormal);
  append_vop2(words, kOpMinU32, subnormal, scalar_positive_inline_u32(8), subnormal);

  append_vop3(words, kOpCmpGtU32, predicate, vgpr_src(shift), scalar_positive_inline_u32(24));
  words.push_back(pack_sopp(kSoppWaitAlu, kWaitAluDepctrVaSdst0));
  append_vop3(words, kOpCndmaskB32, subnormal, vgpr_src(subnormal), scalar_positive_inline_u32(0),
              predicate);
  append_vop3(words, kOpCmpGeU32, predicate, vgpr_src(abs_bits), 255, 0, kMinNormal);
  words.push_back(pack_sopp(kSoppWaitAlu, kWaitAluDepctrVaSdst0));
  append_vop3(words, kOpCndmaskB32, result, vgpr_src(subnormal), vgpr_src(normal), predicate);

  append_wait_valu_vgpr(words);
  append_vop3(words, kOpCmpGtU32, predicate, vgpr_src(result), 255, 0, 0xFEu);
  words.push_back(pack_sopp(kSoppWaitAlu, kWaitAluDepctrVaSdst0));
  append_vop3(words, kOpCndmaskB32, result, vgpr_src(result), 255, predicate, 0xFFu);
  append_vop3(words, kOpCmpGeU32, predicate, vgpr_src(abs_bits), 255, 0, kInf);
  words.push_back(pack_sopp(kSoppWaitAlu, kWaitAluDepctrVaSdst0));
  append_vop3(words, kOpCndmaskB32, result, vgpr_src(result), 255, predicate, 0xFFu);
}

ExpandResult expand_v_cvt_pk_fp8_f32_e5m3_vop3(const Instruction &inst, uint32_t, uint64_t,
                                               const LivenessAnalysis &liveness,
                                               TranslationContext &context, const LaneLayout *,
                                               const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3MachineInst))
    return ExpandResult::not_handled();
  gfx1250::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.op != 873 || src.clamp == 0)
    return ExpandResult::not_handled();
  if (src.vdst == 255 || src.abs != 0 || (src.opsel & ~0x8u) != 0 || src.omod != 0 ||
      src.neg != 0 || src.src0 == 254 || src.src1 == 254)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " uses unsupported E5M3 pack operands or modifiers");

  std::optional<uint32_t> literal_word;
  if (src.src0 == 255)
    literal_word = simm32_literal_word(inst, 0);
  else if (src.src1 == 255)
    literal_word = simm32_literal_word(inst, 1);
  if ((src.src0 == 255 || src.src1 == 255) && !literal_word)
    return ExpandResult::failed(std::string(inst.mnemonic()) + " is missing its literal32 source");

  std::vector<uint8_t> avoid{static_cast<uint8_t>(src.vdst)};
  add_avoid_src_vgpr(avoid, static_cast<uint16_t>(src.src0));
  add_avoid_src_vgpr(avoid, static_cast<uint16_t>(src.src1));
  constexpr uint16_t kScratchCount = 6;
  const auto scratch = find_free_vgpr_run_avoiding(inst, liveness, kScratchCount, avoid);
  const auto predicate = liveness.find_free_sgpr(&inst);
  if (!scratch || *scratch > 250 || !predicate || *predicate > 105)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " cannot allocate E5M3 pack scratch registers");

  const uint8_t lo = static_cast<uint8_t>(*scratch);
  const uint8_t hi = static_cast<uint8_t>(*scratch + 1u);
  const uint8_t abs_bits = static_cast<uint8_t>(*scratch + 2u);
  const uint8_t normal = static_cast<uint8_t>(*scratch + 3u);
  const uint8_t subnormal = static_cast<uint8_t>(*scratch + 4u);
  const uint8_t tmp = static_cast<uint8_t>(*scratch + 5u);
  constexpr uint8_t kOpMovB32 = 1;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpOrB32 = 28;

  std::vector<uint32_t> words;
  words.reserve(80);
  append_vop1(words, kOpMovB32, lo, static_cast<uint16_t>(src.src0), literal_word);
  append_vop1(words, kOpMovB32, hi, static_cast<uint16_t>(src.src1), literal_word);
  append_f32_to_e5m3_rne(words, lo, lo, abs_bits, normal, subnormal, tmp,
                         static_cast<uint8_t>(*predicate));
  append_f32_to_e5m3_rne(words, hi, hi, abs_bits, normal, subnormal, tmp,
                         static_cast<uint8_t>(*predicate));
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpLshlrevB32, hi, scalar_positive_inline_u32(8), hi);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpOrB32, lo, vgpr_src(lo), hi);
  append_wait_valu_vgpr(words);
  append_merge_b16_result(words, static_cast<uint8_t>(src.vdst), lo, (src.opsel & 0x8u) != 0);

  context.require_vgprs(static_cast<uint32_t>(*scratch) + kScratchCount);
  context.require_sgprs(static_cast<uint32_t>(*predicate) + 1u);
  return ExpandResult::success(std::move(words));
}

ExpandResult expand_v_cvt_sr_fp8_f32_e5m3_vop3(const Instruction &inst, uint32_t, uint64_t,
                                               const LivenessAnalysis &liveness,
                                               TranslationContext &context, const LaneLayout *,
                                               const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3MachineInst))
    return ExpandResult::not_handled();
  gfx1250::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.op != 875 || src.clamp == 0)
    return ExpandResult::not_handled();
  if (src.vdst == 255 || src.abs != 0 || (src.opsel & ~0xCu) != 0 || src.omod != 0 ||
      src.neg != 0 || src.src0 == 254 || src.src1 == 254)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " uses unsupported E5M3 stochastic operands or modifiers");

  std::optional<uint32_t> literal_word;
  if (src.src0 == 255)
    literal_word = simm32_literal_word(inst, 0);
  else if (src.src1 == 255)
    literal_word = simm32_literal_word(inst, 1);
  if ((src.src0 == 255 || src.src1 == 255) && !literal_word)
    return ExpandResult::failed(std::string(inst.mnemonic()) + " is missing its literal32 source");

  std::vector<uint8_t> avoid{static_cast<uint8_t>(src.vdst)};
  add_avoid_src_vgpr(avoid, static_cast<uint16_t>(src.src0));
  add_avoid_src_vgpr(avoid, static_cast<uint16_t>(src.src1));
  constexpr uint16_t kScratchCount = 9;
  const auto scratch = find_free_vgpr_run_avoiding(inst, liveness, kScratchCount, avoid);
  const auto predicate = liveness.find_free_sgpr(&inst);
  if (!scratch || *scratch > 247 || !predicate || *predicate > 105)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " cannot allocate E5M3 stochastic scratch registers");

  const uint8_t result = static_cast<uint8_t>(*scratch);
  const uint8_t seed = static_cast<uint8_t>(*scratch + 1u);
  const uint8_t abs_bits = static_cast<uint8_t>(*scratch + 2u);
  const uint8_t normal = static_cast<uint8_t>(*scratch + 3u);
  const uint8_t subnormal = static_cast<uint8_t>(*scratch + 4u);
  const uint8_t shift = static_cast<uint8_t>(*scratch + 5u);
  const uint8_t full_mant = static_cast<uint8_t>(*scratch + 6u);
  const uint8_t work = static_cast<uint8_t>(*scratch + 7u);
  const uint8_t shifted = static_cast<uint8_t>(*scratch + 8u);
  constexpr uint8_t kOpMovB32 = 1;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpOrB32 = 28;

  std::vector<uint32_t> words;
  words.reserve(100);
  append_vop1(words, kOpMovB32, result, static_cast<uint16_t>(src.src0), literal_word);
  append_vop1(words, kOpMovB32, seed, static_cast<uint16_t>(src.src1), literal_word);
  append_f32_to_e5m3_sr(words, result, result, seed, abs_bits, normal, subnormal, shift, full_mant,
                        work, static_cast<uint8_t>(*predicate));

  const uint8_t byte_index = static_cast<uint8_t>((src.opsel >> 2u) & 0x3u);
  const uint8_t byte_shift = static_cast<uint8_t>(byte_index * 8u);
  if (byte_shift != 0) {
    append_wait_valu_vgpr(words);
    append_vop2(words, kOpLshlrevB32, shifted, scalar_positive_inline_u32(byte_shift), result);
  } else {
    append_vop1(words, kOpMovB32, shifted, vgpr_src(result));
  }
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpAndB32, static_cast<uint8_t>(src.vdst), 255, static_cast<uint8_t>(src.vdst),
              ~(0xFFu << byte_shift));
  append_vop2(words, kOpOrB32, static_cast<uint8_t>(src.vdst), vgpr_src(shifted),
              static_cast<uint8_t>(src.vdst));

  context.require_vgprs(static_cast<uint32_t>(*scratch) + kScratchCount);
  context.require_sgprs(static_cast<uint32_t>(*predicate) + 1u);
  return ExpandResult::success(std::move(words));
}

std::vector<uint32_t> lower_v_cvt_f16_f8(uint8_t vdst, uint16_t src0, uint8_t opsel,
                                         std::optional<uint32_t> literal_word, bool bf8,
                                         const Instruction &inst,
                                         const LivenessAnalysis &liveness) {
  if (vdst == 255 || src0 == 254 || (opsel & ~0x3u) != 0)
    return {};

  std::vector<uint8_t> avoid{vdst};
  add_avoid_src_vgpr(avoid, src0);
  const auto tmp_opt = find_free_vgpr_run_avoiding(inst, liveness, 1, avoid);
  if (!tmp_opt || *tmp_opt > 255u)
    return {};
  const auto tmp = static_cast<uint8_t>(*tmp_opt);

  constexpr uint16_t kOpCvtF32Fp8 = 492;
  constexpr uint16_t kOpCvtF32Bf8 = 493;
  constexpr uint8_t kOpCvtF16F32 = 10;
  std::vector<uint32_t> words;
  words.reserve(literal_word ? 5 : 4);
  const auto [decode_w0, decode_w1] =
      build_vop3_mod(bf8 ? kOpCvtF32Bf8 : kOpCvtF32Fp8, tmp, src0, 0, 0, 0, opsel);
  words.push_back(decode_w0);
  words.push_back(decode_w1);
  if (literal_word && src0 == 255)
    words.push_back(*literal_word);
  append_wait_valu_vgpr(words);
  append_vop1(words, kOpCvtF16F32, vdst, static_cast<uint16_t>(256u + tmp));
  return words;
}

std::vector<uint32_t> expand_v_cvt_f16_fp8_vop1(const Instruction &inst, uint32_t, uint64_t,
                                                const LivenessAnalysis &liveness,
                                                const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop1MachineInst))
    return {};
  const auto src = std::bit_cast<gfx1250::Vop1MachineInst>(raw[0]);
  std::optional<uint32_t> literal_word;
  if (src.src0 == 255) {
    literal_word = simm32_literal_word(inst, 0);
    if (!literal_word)
      return {};
  }
  return lower_v_cvt_f16_f8(static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src0), 0,
                            literal_word, /*bf8=*/false, inst, liveness);
}

std::vector<uint32_t> expand_v_cvt_f16_bf8_vop1(const Instruction &inst, uint32_t, uint64_t,
                                                const LivenessAnalysis &liveness,
                                                const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop1MachineInst))
    return {};
  const auto src = std::bit_cast<gfx1250::Vop1MachineInst>(raw[0]);
  std::optional<uint32_t> literal_word;
  if (src.src0 == 255) {
    literal_word = simm32_literal_word(inst, 0);
    if (!literal_word)
      return {};
  }
  return lower_v_cvt_f16_f8(static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src0), 0,
                            literal_word, /*bf8=*/true, inst, liveness);
}

std::vector<uint32_t> expand_v_cvt_f16_f8_vop3(const Instruction &inst, uint32_t, uint64_t,
                                               const LivenessAnalysis &liveness, bool bf8) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3MachineInst))
    return {};
  gfx1250::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.abs != 0 || src.clamp != 0 || src.omod != 0 || src.neg != 0)
    return {};
  std::optional<uint32_t> literal_word;
  if (src.src0 == 255) {
    literal_word = simm32_literal_word(inst, 0);
    if (!literal_word)
      return {};
  }
  return lower_v_cvt_f16_f8(static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src0),
                            static_cast<uint8_t>(src.opsel), literal_word, bf8, inst, liveness);
}

std::vector<uint32_t> expand_v_cvt_f16_fp8_vop3(const Instruction &inst, uint32_t host_arch,
                                                uint64_t offset, const LivenessAnalysis &liveness,
                                                const LaneLayout *, const LaneLayout *) {
  return expand_v_cvt_f16_f8_vop3(inst, host_arch, offset, liveness, /*bf8=*/false);
}

std::vector<uint32_t> expand_v_cvt_f16_bf8_vop3(const Instruction &inst, uint32_t host_arch,
                                                uint64_t offset, const LivenessAnalysis &liveness,
                                                const LaneLayout *, const LaneLayout *) {
  return expand_v_cvt_f16_f8_vop3(inst, host_arch, offset, liveness, /*bf8=*/true);
}

std::vector<uint32_t> lower_v_cvt_pk_f8_f16(uint8_t vdst, uint16_t src0, uint8_t opsel,
                                            std::optional<uint32_t> literal_word, bool bf8,
                                            const Instruction &inst,
                                            const LivenessAnalysis &liveness) {
  if (src0 == 254 || (opsel & ~0x8u) != 0)
    return {};

  std::vector<uint8_t> avoid{vdst};
  add_avoid_src_vgpr(avoid, src0);
  const auto tmp_base = find_free_vgpr_run_avoiding(inst, liveness, 2, avoid);
  if (!tmp_base || *tmp_base > 254u)
    return {};
  const auto lo = static_cast<uint8_t>(*tmp_base);
  const auto hi = static_cast<uint8_t>(*tmp_base + 1u);

  constexpr uint16_t kOpCvtF32F16 = 0x18B;
  constexpr uint16_t kOpCvtPkFp8F32 = 873;
  constexpr uint16_t kOpCvtPkBf8F32 = 874;
  std::vector<uint32_t> words;
  words.reserve(literal_word ? 9 : 7);
  append_vop3_mod(words, kOpCvtF32F16, lo, src0, 0, 0, 0, literal_word);
  append_vop3_mod(words, kOpCvtF32F16, hi, src0, 0, 0, 1, literal_word);
  append_wait_valu_vgpr(words);
  append_vop3_mod(words, bf8 ? kOpCvtPkBf8F32 : kOpCvtPkFp8F32, vdst, vgpr_src(lo), vgpr_src(hi), 0,
                  static_cast<uint8_t>(opsel & 0x8u));
  return words;
}

std::vector<uint32_t> expand_v_cvt_pk_f8_f16_vop3(const Instruction &inst,
                                                  const LivenessAnalysis &liveness, bool bf8) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3MachineInst))
    return {};
  gfx1250::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.abs != 0 || src.clamp != 0 || src.omod != 0 || src.neg != 0)
    return {};
  std::optional<uint32_t> literal_word;
  if (src.src0 == 255) {
    literal_word = simm32_literal_word(inst, 0);
    if (!literal_word)
      return {};
  }
  return lower_v_cvt_pk_f8_f16(static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src0),
                               static_cast<uint8_t>(src.opsel), literal_word, bf8, inst, liveness);
}

std::vector<uint32_t> expand_v_cvt_pk_fp8_f16_vop3(const Instruction &inst, uint32_t, uint64_t,
                                                   const LivenessAnalysis &liveness,
                                                   const LaneLayout *, const LaneLayout *) {
  return expand_v_cvt_pk_f8_f16_vop3(inst, liveness, /*bf8=*/false);
}

std::vector<uint32_t> expand_v_cvt_pk_bf8_f16_vop3(const Instruction &inst, uint32_t, uint64_t,
                                                   const LivenessAnalysis &liveness,
                                                   const LaneLayout *, const LaneLayout *) {
  return expand_v_cvt_pk_f8_f16_vop3(inst, liveness, /*bf8=*/true);
}

std::vector<uint32_t> lower_v_cvt_pk_f16_f8(uint8_t vdst, uint16_t src0, bool src_high,
                                            std::optional<uint32_t> literal_word, bool bf8,
                                            const Instruction &inst,
                                            const LivenessAnalysis &liveness) {
  if (src0 == 254)
    return {};

  std::vector<uint8_t> avoid{vdst};
  add_avoid_src_vgpr(avoid, src0);
  const auto tmp_base = find_free_vgpr_run_avoiding(inst, liveness, 2, avoid);
  if (!tmp_base || *tmp_base > 254u)
    return {};
  const auto lo = static_cast<uint8_t>(*tmp_base);
  const auto hi = static_cast<uint8_t>(*tmp_base + 1u);

  constexpr uint16_t kOpCvtF32Fp8 = 492;
  constexpr uint16_t kOpCvtF32Bf8 = 493;
  constexpr uint8_t kOpCvtF16F32 = 10;
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpOrB32 = 28;
  // VOP3 OPSEL encodes byte selection as {half, byte-within-half}: byte 1 is
  // 0b10 and byte 2 is 0b01, rather than the linear byte index.
  const uint8_t first_opsel = src_high ? 1u : 0u;
  const uint8_t second_opsel = src_high ? 3u : 2u;

  std::vector<uint32_t> words;
  words.reserve(literal_word ? 14 : 12);
  append_vop3_mod(words, bf8 ? kOpCvtF32Bf8 : kOpCvtF32Fp8, lo, src0, 0, 0, first_opsel,
                  literal_word);
  append_vop3_mod(words, bf8 ? kOpCvtF32Bf8 : kOpCvtF32Fp8, hi, src0, 0, 0, second_opsel,
                  literal_word);
  append_wait_valu_vgpr(words);
  append_vop1(words, kOpCvtF16F32, lo, vgpr_src(lo));
  append_vop1(words, kOpCvtF16F32, hi, vgpr_src(hi));
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpAndB32, lo, 255, lo, 0x0000FFFFu);
  append_vop2(words, kOpLshlrevB32, hi, scalar_positive_inline_u32(16), hi);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpOrB32, vdst, vgpr_src(lo), hi);
  return words;
}

std::vector<uint32_t> expand_v_cvt_pk_f16_fp8_vop1(const Instruction &inst, uint32_t, uint64_t,
                                                   const LivenessAnalysis &liveness,
                                                   const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop1MachineInst))
    return {};
  const auto src = std::bit_cast<gfx1250::Vop1MachineInst>(raw[0]);
  std::optional<uint32_t> literal_word;
  if (src.src0 == 255) {
    literal_word = simm32_literal_word(inst, 0);
    if (!literal_word)
      return {};
  }
  return lower_v_cvt_pk_f16_f8(static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src0),
                               false, literal_word,
                               /*bf8=*/false, inst, liveness);
}

std::vector<uint32_t> expand_v_cvt_pk_f16_bf8_vop1(const Instruction &inst, uint32_t, uint64_t,
                                                   const LivenessAnalysis &liveness,
                                                   const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop1MachineInst))
    return {};
  const auto src = std::bit_cast<gfx1250::Vop1MachineInst>(raw[0]);
  std::optional<uint32_t> literal_word;
  if (src.src0 == 255) {
    literal_word = simm32_literal_word(inst, 0);
    if (!literal_word)
      return {};
  }
  return lower_v_cvt_pk_f16_f8(static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src0),
                               false, literal_word,
                               /*bf8=*/true, inst, liveness);
}

std::vector<uint32_t> expand_v_cvt_pk_f16_f8_vop3(const Instruction &inst,
                                                  const LivenessAnalysis &liveness, bool bf8) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3MachineInst))
    return {};
  gfx1250::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.abs != 0 || src.clamp != 0 || src.omod != 0 || src.neg != 0 || (src.opsel & ~0x1u) != 0)
    return {};
  std::optional<uint32_t> literal_word;
  if (src.src0 == 255) {
    literal_word = simm32_literal_word(inst, 0);
    if (!literal_word)
      return {};
  }
  return lower_v_cvt_pk_f16_f8(static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src0),
                               (src.opsel & 0x1u) != 0, literal_word, bf8, inst, liveness);
}

std::vector<uint32_t> expand_v_cvt_pk_f16_fp8_vop3(const Instruction &inst, uint32_t, uint64_t,
                                                   const LivenessAnalysis &liveness,
                                                   const LaneLayout *, const LaneLayout *) {
  return expand_v_cvt_pk_f16_f8_vop3(inst, liveness, /*bf8=*/false);
}

std::vector<uint32_t> expand_v_cvt_pk_f16_bf8_vop3(const Instruction &inst, uint32_t, uint64_t,
                                                   const LivenessAnalysis &liveness,
                                                   const LaneLayout *, const LaneLayout *) {
  return expand_v_cvt_pk_f16_f8_vop3(inst, liveness, /*bf8=*/true);
}

std::vector<uint32_t> expand_v_cvt_sr_f8_f16_vop3(const Instruction &inst,
                                                  const LivenessAnalysis &liveness, bool bf8) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3MachineInst))
    return {};
  gfx1250::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.abs != 0 || src.clamp != 0 || src.omod != 0 || src.neg != 0 || (src.opsel & ~0xDu) != 0 ||
      src.src0 == 254 || src.src1 == 254)
    return {};

  std::optional<uint32_t> src0_literal;
  std::optional<uint32_t> src1_literal;
  if (src.src0 == 255) {
    src0_literal = simm32_literal_word(inst, 0);
    if (!src0_literal)
      return {};
  }
  if (src.src1 == 255) {
    src1_literal = simm32_literal_word(inst, 1);
    if (!src1_literal)
      return {};
  }

  std::vector<uint8_t> avoid{static_cast<uint8_t>(src.vdst)};
  add_avoid_src_vgpr(avoid, static_cast<uint16_t>(src.src0));
  add_avoid_src_vgpr(avoid, static_cast<uint16_t>(src.src1));
  const auto tmp_opt = find_free_vgpr_run_avoiding(inst, liveness, 1, avoid);
  if (!tmp_opt || *tmp_opt > 255u)
    return {};
  const auto tmp = static_cast<uint8_t>(*tmp_opt);

  constexpr uint16_t kOpCvtF32F16 = 0x18B;
  constexpr uint16_t kOpCvtSrFp8F32 = 875;
  constexpr uint16_t kOpCvtSrBf8F32 = 876;
  std::vector<uint32_t> words;
  words.reserve((src0_literal || src1_literal) ? 8 : 7);
  append_vop3_mod(words, kOpCvtF32F16, tmp, static_cast<uint16_t>(src.src0), 0, 0,
                  static_cast<uint8_t>(src.opsel & 0x1u), src0_literal);
  append_wait_valu_vgpr(words);
  append_vop3_mod(words, bf8 ? kOpCvtSrBf8F32 : kOpCvtSrFp8F32, static_cast<uint8_t>(src.vdst),
                  vgpr_src(tmp), static_cast<uint16_t>(src.src1), 0,
                  static_cast<uint8_t>(src.opsel & 0xCu), src1_literal);
  return words;
}

std::vector<uint32_t> expand_v_cvt_sr_fp8_f16_vop3(const Instruction &inst, uint32_t, uint64_t,
                                                   const LivenessAnalysis &liveness,
                                                   const LaneLayout *, const LaneLayout *) {
  return expand_v_cvt_sr_f8_f16_vop3(inst, liveness, /*bf8=*/false);
}

std::vector<uint32_t> expand_v_cvt_sr_bf8_f16_vop3(const Instruction &inst, uint32_t, uint64_t,
                                                   const LivenessAnalysis &liveness,
                                                   const LaneLayout *, const LaneLayout *) {
  return expand_v_cvt_sr_f8_f16_vop3(inst, liveness, /*bf8=*/true);
}

std::vector<uint32_t> expand_v_fma_mix_f32_bf16_vop3p(const Instruction &inst, uint32_t, uint64_t,
                                                      const LivenessAnalysis &liveness,
                                                      const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() != sizeof(gfx1250::Vop3pMachineInst))
    return {};

  gfx1250::Vop3pMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.vdst == 255 || src.src0 == 254 || src.src1 == 254 || src.src2 == 254)
    return {};

  const std::array<uint16_t, 3> sources{static_cast<uint16_t>(src.src0),
                                        static_cast<uint16_t>(src.src1),
                                        static_cast<uint16_t>(src.src2)};
  const std::array<bool, 3> source_is_bf16{(src.opsel_hi & 0x1u) != 0, (src.opsel_hi & 0x2u) != 0,
                                           src.pad_14 != 0};
  const std::array<bool, 3> high_half{(src.opsel & 0x1u) != 0, (src.opsel & 0x2u) != 0,
                                      (src.opsel & 0x4u) != 0};

  std::array<std::optional<uint32_t>, 3> literals{};
  for (size_t i = 0; i < sources.size(); ++i) {
    if (sources[i] != 255)
      continue;
    literals[i] = simm32_literal_word(inst, static_cast<uint8_t>(i));
    if (!literals[i])
      return {};
  }

  std::array<uint16_t, 3> fma_sources = sources;
  const auto bf16_source_count =
      static_cast<size_t>(std::count(source_is_bf16.begin(), source_is_bf16.end(), true));

  std::vector<uint8_t> avoid{static_cast<uint8_t>(src.vdst)};
  for (const uint16_t source : sources)
    add_avoid_src_vgpr(avoid, source);

  std::optional<uint16_t> tmp_base;
  if (bf16_source_count != 0) {
    tmp_base = find_free_vgpr_run_avoiding(inst, liveness, bf16_source_count, avoid);
    if (!tmp_base)
      return {};
  }

  std::vector<uint32_t> words;
  words.reserve(8 + bf16_source_count * 3u);

  size_t tmp_index = 0;
  for (size_t i = 0; i < sources.size(); ++i) {
    if (!source_is_bf16[i])
      continue;
    const auto tmp = static_cast<uint8_t>(*tmp_base + tmp_index);
    if (!append_materialize_fma_mix_bf16_source(words, tmp, sources[i], high_half[i], literals[i]))
      return {};
    fma_sources[i] = vgpr_src(tmp);
    ++tmp_index;
  }

  if (bf16_source_count != 0)
    append_wait_valu_vgpr(words);

  constexpr uint16_t kOpFmaF32 = 531;
  auto [w0, w1] =
      build_vop3_mod(kOpFmaF32, static_cast<uint8_t>(src.vdst), fma_sources[0], fma_sources[1],
                     fma_sources[2], 0, 0, src.clamp != 0, 0, static_cast<uint8_t>(src.neg & 0x7u));
  words.push_back(w0);
  words.push_back(w1);

  for (size_t i = 0; i < fma_sources.size(); ++i) {
    if (source_is_bf16[i] || fma_sources[i] != 255)
      continue;
    words.push_back(*literals[i]);
    break;
  }

  return words;
}

std::vector<uint32_t> expand_v_fma_mixlo_bf16_vop3p(const Instruction &inst, uint32_t, uint64_t,
                                                    const LivenessAnalysis &liveness,
                                                    const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() != sizeof(gfx1250::Vop3pMachineInst))
    return {};

  gfx1250::Vop3pMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.vdst == 255 || src.src0 == 254 || src.src1 == 254 || src.src2 == 254)
    return {};

  const std::array<uint16_t, 3> sources{static_cast<uint16_t>(src.src0),
                                        static_cast<uint16_t>(src.src1),
                                        static_cast<uint16_t>(src.src2)};
  const std::array<bool, 3> source_is_bf16{(src.opsel_hi & 0x1u) != 0, (src.opsel_hi & 0x2u) != 0,
                                           src.pad_14 != 0};
  const std::array<bool, 3> high_half{(src.opsel & 0x1u) != 0, (src.opsel & 0x2u) != 0,
                                      (src.opsel & 0x4u) != 0};

  std::array<std::optional<uint32_t>, 3> literals{};
  for (size_t i = 0; i < sources.size(); ++i) {
    if (sources[i] != 255)
      continue;
    literals[i] = simm32_literal_word(inst, static_cast<uint8_t>(i));
    if (!literals[i])
      return {};
  }

  const auto bf16_source_count =
      static_cast<uint16_t>(std::count(source_is_bf16.begin(), source_is_bf16.end(), true));
  constexpr uint16_t kResultTmpCount = 2;
  const uint16_t tmp_count = static_cast<uint16_t>(bf16_source_count + kResultTmpCount);
  std::vector<uint8_t> avoid{static_cast<uint8_t>(src.vdst)};
  for (const uint16_t source : sources)
    add_avoid_src_vgpr(avoid, source);
  const auto tmp_base = find_free_vgpr_run_avoiding(inst, liveness, tmp_count, avoid);
  if (!tmp_base || *tmp_base + tmp_count > 256u)
    return {};

  std::array<uint16_t, 3> fma_sources = sources;
  std::vector<uint32_t> words;
  words.reserve(24);
  uint16_t tmp_index = 0;
  for (size_t i = 0; i < sources.size(); ++i) {
    if (!source_is_bf16[i])
      continue;
    const uint8_t tmp = static_cast<uint8_t>(*tmp_base + tmp_index++);
    if (!append_materialize_fma_mix_bf16_source(words, tmp, sources[i], high_half[i], literals[i]))
      return {};
    fma_sources[i] = vgpr_src(tmp);
  }
  if (bf16_source_count != 0)
    append_wait_valu_vgpr(words);

  const uint8_t result = static_cast<uint8_t>(*tmp_base + bf16_source_count);
  const uint8_t round_tmp = static_cast<uint8_t>(result + 1u);
  constexpr uint16_t kOpFmaF32 = 531;
  const auto [w0, w1] = build_vop3_mod(kOpFmaF32, result, fma_sources[0], fma_sources[1],
                                       fma_sources[2], static_cast<uint8_t>(src.neg_hi & 0x7u), 0,
                                       src.clamp != 0, 0, static_cast<uint8_t>(src.neg & 0x7u));
  words.push_back(w0);
  words.push_back(w1);
  for (size_t i = 0; i < fma_sources.size(); ++i) {
    if (source_is_bf16[i] || fma_sources[i] != 255)
      continue;
    words.push_back(*literals[i]);
    break;
  }

  append_wait_valu_vgpr(words);
  append_f32_to_bf16_rne(words, result, round_tmp, result);
  append_wait_valu_vgpr(words);
  append_merge_b16_result(words, static_cast<uint8_t>(src.vdst), result, false);
  return words;
}

void append_merge_b16_result(std::vector<uint32_t> &words, uint8_t vdst, uint8_t result,
                             bool dst_high) {
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpOrB32 = 28;

  append_vop2(words, kOpAndB32, result, 255, result, 0x0000FFFFu);
  if (dst_high) {
    append_vop2(words, kOpLshlrevB32, result, scalar_positive_inline_u32(16), result);
    append_vop2(words, kOpAndB32, vdst, 255, vdst, 0x0000FFFFu);
  } else {
    append_vop2(words, kOpAndB32, vdst, 255, vdst, 0xFFFF0000u);
  }
  append_vop2(words, kOpOrB32, vdst, vgpr_src(result), vdst);
}

std::vector<uint32_t> expand_v_bitop3_b16_vop3(const Instruction &inst, uint32_t, uint64_t,
                                               const LivenessAnalysis &liveness, const LaneLayout *,
                                               const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3MachineInst))
    return {};

  gfx1250::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  const uint8_t truth_table = static_cast<uint8_t>((src.omod << 6) | (src.abs << 3) | src.neg);
  if ((truth_table != 0xEC && truth_table != 0xF8 && truth_table != 0xFE) || src.clamp != 0 ||
      src.vdst == 255)
    return {};
  if (src.src0 == 254 || src.src1 == 254 || src.src2 == 254)
    return {};

  std::optional<uint32_t> literal_word;
  if (src.src0 == 255)
    literal_word = simm32_literal_word(inst, 0);
  else if (src.src1 == 255)
    literal_word = simm32_literal_word(inst, 1);
  else if (src.src2 == 255)
    literal_word = simm32_literal_word(inst, 2);
  if ((src.src0 == 255 || src.src1 == 255 || src.src2 == 255) && !literal_word)
    return {};

  std::vector<uint8_t> avoid;
  add_avoid_vgpr(avoid, static_cast<uint8_t>(src.vdst));
  add_avoid_src_vgpr(avoid, static_cast<uint16_t>(src.src0));
  add_avoid_src_vgpr(avoid, static_cast<uint16_t>(src.src1));
  add_avoid_src_vgpr(avoid, static_cast<uint16_t>(src.src2));

  constexpr uint16_t kTmpCount = 4;
  const auto tmp_base = find_free_vgpr_run_avoiding(inst, liveness, kTmpCount, avoid);
  if (!tmp_base)
    return {};
  const auto tmp0 = static_cast<uint8_t>(*tmp_base);
  const auto tmp1 = static_cast<uint8_t>(*tmp_base + 1u);
  const auto tmp2 = static_cast<uint8_t>(*tmp_base + 2u);
  const auto result = static_cast<uint8_t>(*tmp_base + 3u);

  const bool src0_high = (src.opsel & 0x1u) != 0;
  const bool src1_high = (src.opsel & 0x2u) != 0;
  const bool src2_high = (src.opsel & 0x4u) != 0;
  const bool dst_high = (src.opsel & 0x8u) != 0;

  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpOrB32 = 28;

  std::vector<uint32_t> words;
  words.reserve(14);
  if (!append_materialize_b16_half(words, tmp0, static_cast<uint16_t>(src.src0), src0_high,
                                   literal_word) ||
      !append_materialize_b16_half(words, tmp1, static_cast<uint16_t>(src.src1), src1_high,
                                   literal_word) ||
      !append_materialize_b16_half(words, tmp2, static_cast<uint16_t>(src.src2), src2_high,
                                   literal_word))
    return {};

  if (truth_table == 0xEC) {
    append_vop2(words, kOpAndB32, result, vgpr_src(tmp0), tmp2);
    append_vop2(words, kOpOrB32, result, vgpr_src(tmp1), result);
  } else if (truth_table == 0xF8) {
    append_vop2(words, kOpAndB32, result, vgpr_src(tmp1), tmp2);
    append_vop2(words, kOpOrB32, result, vgpr_src(tmp0), result);
  } else {
    append_vop2(words, kOpOrB32, result, vgpr_src(tmp0), tmp1);
    append_vop2(words, kOpOrB32, result, vgpr_src(tmp2), result);
  }

  append_merge_b16_result(words, static_cast<uint8_t>(src.vdst), result, dst_high);
  return words;
}

std::vector<uint32_t> expand_v_lshlrev_b16_vop3(const Instruction &inst, uint32_t, uint64_t,
                                                const LivenessAnalysis &liveness,
                                                const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3MachineInst))
    return {};

  gfx1250::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.abs != 0 || src.clamp != 0 || src.omod != 0 || src.neg != 0 || src.vdst == 255)
    return {};
  if (src.src0 == 254 || src.src1 == 254)
    return {};

  std::optional<uint32_t> literal_word;
  if (src.src0 == 255)
    literal_word = simm32_literal_word(inst, 0);
  else if (src.src1 == 255)
    literal_word = simm32_literal_word(inst, 1);
  if ((src.src0 == 255 || src.src1 == 255) && !literal_word)
    return {};

  std::vector<uint8_t> avoid;
  add_avoid_vgpr(avoid, static_cast<uint8_t>(src.vdst));
  add_avoid_src_vgpr(avoid, static_cast<uint16_t>(src.src0));
  add_avoid_src_vgpr(avoid, static_cast<uint16_t>(src.src1));

  constexpr uint16_t kTmpCount = 3;
  const auto tmp_base = find_free_vgpr_run_avoiding(inst, liveness, kTmpCount, avoid);
  if (!tmp_base)
    return {};
  const auto shift = static_cast<uint8_t>(*tmp_base);
  const auto value = static_cast<uint8_t>(*tmp_base + 1u);
  const auto result = static_cast<uint8_t>(*tmp_base + 2u);

  const bool src0_high = (src.opsel & 0x1u) != 0;
  const bool src1_high = (src.opsel & 0x2u) != 0;
  const bool dst_high = (src.opsel & 0x8u) != 0;

  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpAndB32 = 27;

  std::vector<uint32_t> words;
  words.reserve(12);
  if (!append_materialize_b16_half(words, shift, static_cast<uint16_t>(src.src0), src0_high,
                                   literal_word) ||
      !append_materialize_b16_half(words, value, static_cast<uint16_t>(src.src1), src1_high,
                                   literal_word))
    return {};

  append_vop2(words, kOpAndB32, shift, 255, shift, 0x0000000Fu);
  append_vop2(words, kOpLshlrevB32, result, vgpr_src(shift), value);
  append_merge_b16_result(words, static_cast<uint8_t>(src.vdst), result, dst_high);
  return words;
}

std::vector<uint32_t> expand_v_or_b16_vop3(const Instruction &inst, uint32_t, uint64_t,
                                           const LivenessAnalysis &liveness, const LaneLayout *,
                                           const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3MachineInst))
    return {};

  gfx1250::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.abs != 0 || src.clamp != 0 || src.omod != 0 || src.neg != 0 || src.vdst == 255)
    return {};
  if (src.src0 == 254 || src.src1 == 254)
    return {};

  std::optional<uint32_t> literal_word;
  if (src.src0 == 255)
    literal_word = simm32_literal_word(inst, 0);
  else if (src.src1 == 255)
    literal_word = simm32_literal_word(inst, 1);
  if ((src.src0 == 255 || src.src1 == 255) && !literal_word)
    return {};

  std::vector<uint8_t> avoid;
  add_avoid_vgpr(avoid, static_cast<uint8_t>(src.vdst));
  add_avoid_src_vgpr(avoid, static_cast<uint16_t>(src.src0));
  add_avoid_src_vgpr(avoid, static_cast<uint16_t>(src.src1));

  constexpr uint16_t kTmpCount = 3;
  const auto tmp_base = find_free_vgpr_run_avoiding(inst, liveness, kTmpCount, avoid);
  if (!tmp_base)
    return {};
  const auto tmp0 = static_cast<uint8_t>(*tmp_base);
  const auto tmp1 = static_cast<uint8_t>(*tmp_base + 1u);
  const auto result = static_cast<uint8_t>(*tmp_base + 2u);

  const bool src0_high = (src.opsel & 0x1u) != 0;
  const bool src1_high = (src.opsel & 0x2u) != 0;
  const bool dst_high = (src.opsel & 0x8u) != 0;

  constexpr uint8_t kOpOrB32 = 28;

  std::vector<uint32_t> words;
  words.reserve(10);
  if (!append_materialize_b16_half(words, tmp0, static_cast<uint16_t>(src.src0), src0_high,
                                   literal_word) ||
      !append_materialize_b16_half(words, tmp1, static_cast<uint16_t>(src.src1), src1_high,
                                   literal_word))
    return {};

  append_vop2(words, kOpOrB32, result, vgpr_src(tmp0), tmp1);
  append_merge_b16_result(words, static_cast<uint8_t>(src.vdst), result, dst_high);
  return words;
}

void append_f32_to_bf16_rne(std::vector<uint32_t> &words, uint8_t result, uint8_t tmp,
                            uint8_t src_vgpr) {
  constexpr uint8_t kOpLshrrevB32 = 25;
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpAddNcU32 = 37;

  append_vop2(words, kOpLshrrevB32, tmp, scalar_positive_inline_u32(16), src_vgpr);
  append_vop2(words, kOpAndB32, tmp, scalar_positive_inline_u32(1), tmp);
  append_vop2(words, kOpAddNcU32, result, 255, src_vgpr, 0x00007FFFu);
  append_vop2(words, kOpAddNcU32, result, vgpr_src(tmp), result);
  append_vop2(words, kOpLshrrevB32, result, scalar_positive_inline_u32(16), result);
}

ExpandResult expand_v_cvt_sr_pk_f16_bf16_f32_vop3(const Instruction &inst, uint32_t, uint64_t,
                                                  const LivenessAnalysis &liveness,
                                                  TranslationContext &context, const LaneLayout *,
                                                  const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3MachineInst))
    return ExpandResult::not_handled();

  gfx1250::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  constexpr uint16_t kOpCvtSrPkBf16F32 = 878;
  constexpr uint16_t kOpCvtSrPkF16F32 = 880;
  if (src.op != kOpCvtSrPkBf16F32 && src.op != kOpCvtSrPkF16F32)
    return ExpandResult::not_handled();
  if (src.vdst == 255 || src.abs != 0 || src.opsel != 0 || src.clamp != 0 || src.omod != 0 ||
      src.neg != 0)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " uses unsupported source or result modifiers");
  if (src.src0 == 254 || src.src1 == 254 || src.src2 == 254)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " does not support literal64 operands");

  std::optional<uint32_t> literal_word;
  if (src.src0 == 255)
    literal_word = simm32_literal_word(inst, 0);
  else if (src.src1 == 255)
    literal_word = simm32_literal_word(inst, 1);
  else if (src.src2 == 255)
    literal_word = simm32_literal_word(inst, 2);
  if ((src.src0 == 255 || src.src1 == 255 || src.src2 == 255) && !literal_word)
    return ExpandResult::failed(std::string(inst.mnemonic()) + " is missing its literal32 operand");

  std::vector<uint8_t> avoid;
  add_avoid_vgpr(avoid, static_cast<uint8_t>(src.vdst));
  add_avoid_src_vgpr(avoid, static_cast<uint16_t>(src.src0));
  add_avoid_src_vgpr(avoid, static_cast<uint16_t>(src.src1));
  add_avoid_src_vgpr(avoid, static_cast<uint16_t>(src.src2));
  constexpr uint16_t kScratchCount = 5;
  const auto scratch_base = find_free_vgpr_run_avoiding(inst, liveness, kScratchCount, avoid);
  if (!scratch_base || *scratch_base > 251)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " cannot allocate five VGPRs for stochastic narrowing");
  const auto predicate = liveness.find_free_sgpr(&inst);
  if (!predicate || *predicate > 105)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " cannot allocate an SGPR predicate for zero preservation");

  const uint8_t lo = static_cast<uint8_t>(*scratch_base);
  const uint8_t hi = static_cast<uint8_t>(*scratch_base + 1u);
  const uint8_t seed = static_cast<uint8_t>(*scratch_base + 2u);
  const uint8_t random = static_cast<uint8_t>(*scratch_base + 3u);
  const uint8_t convert_tmp = static_cast<uint8_t>(*scratch_base + 4u);
  const bool is_f16 = src.op == kOpCvtSrPkF16F32;

  constexpr uint8_t kOpMovB32 = 1;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpLshrrevB32 = 25;
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpOrB32 = 28;
  constexpr uint8_t kOpXorB32 = 29;
  constexpr uint8_t kOpAddNcU32 = 37;
  constexpr uint16_t kOpCmpEqU32 = 74;
  constexpr uint16_t kOpCndmaskB32 = 257;
  constexpr uint16_t kOpCvtF16F32 = 0x18A;
  constexpr uint16_t kOpMulLoU32 = 812;
  constexpr uint8_t kSoppWaitAlu = 8;

  const uint16_t random_shift = scalar_positive_inline_u32(is_f16 ? 19 : 16);
  const uint32_t half_ulp = is_f16 ? 0x1000u : 0x8000u;

  std::vector<uint32_t> words;
  words.reserve(45);
  append_vop1(words, kOpMovB32, lo, static_cast<uint16_t>(src.src0), literal_word);
  append_vop1(words, kOpMovB32, hi, static_cast<uint16_t>(src.src1), literal_word);
  append_vop1(words, kOpMovB32, seed, static_cast<uint16_t>(src.src2), literal_word);
  append_wait_valu_vgpr(words);

  auto append_centered_dither = [&](uint8_t value) {
    append_vop2(words, kOpLshrrevB32, random, random_shift, seed);
    append_wait_valu_vgpr(words);
    append_vop2(words, kOpAddNcU32, random, 255, random, 0u - half_ulp);
    append_vop2(words, kOpLshlrevB32, convert_tmp, scalar_positive_inline_u32(1), value);
    append_wait_valu_vgpr(words);
    append_vop3(words, kOpCmpEqU32, static_cast<uint8_t>(*predicate), vgpr_src(convert_tmp),
                scalar_positive_inline_u32(0));
    words.push_back(pack_sopp(kSoppWaitAlu, kWaitAluDepctrVaSdst0));
    append_vop3(words, kOpCndmaskB32, random, vgpr_src(random), scalar_positive_inline_u32(0),
                static_cast<uint16_t>(*predicate));
    append_wait_valu_vgpr(words);
    append_vop2(words, kOpAddNcU32, value, vgpr_src(random), value);
    append_wait_valu_vgpr(words);
  };

  // The gfx1250 definition adds uniformly distributed discarded mantissa bits
  // before truncation. RDNA4 provides RNE narrow conversions instead, so use
  // the equivalent centered dither interval before RNE. Exact values remain
  // exact, while inexact values choose between their adjacent narrow values.
  append_centered_dither(lo);

  // seed_hi = (seed << 1) ^ ((seed >> 31) ? 197 : 0)
  append_vop2(words, kOpLshrrevB32, random, scalar_positive_inline_u32(31), seed);
  append_vop2(words, kOpLshlrevB32, seed, scalar_positive_inline_u32(1), seed);
  append_wait_valu_vgpr(words);
  append_vop3(words, kOpMulLoU32, random, vgpr_src(random), 255, 0, 197u);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpXorB32, seed, vgpr_src(random), seed);
  append_wait_valu_vgpr(words);
  append_centered_dither(hi);

  if (is_f16) {
    append_vop3(words, kOpCvtF16F32, lo, vgpr_src(lo), 0);
    append_vop3(words, kOpCvtF16F32, hi, vgpr_src(hi), 0);
    append_wait_valu_vgpr(words);
    append_vop2(words, kOpAndB32, lo, 255, lo, 0xFFFFu);
  } else {
    append_f32_to_bf16_rne(words, lo, convert_tmp, lo);
    append_f32_to_bf16_rne(words, hi, convert_tmp, hi);
  }
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpLshlrevB32, hi, scalar_positive_inline_u32(16), hi);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpOrB32, static_cast<uint8_t>(src.vdst), vgpr_src(lo), hi);

  context.require_vgprs(static_cast<uint32_t>(*scratch_base) + kScratchCount);
  context.require_sgprs(static_cast<uint32_t>(*predicate) + 1u);
  return ExpandResult::success(std::move(words));
}

std::vector<uint32_t> expand_v_pk_add_bf16_vop3p(const Instruction &inst, uint32_t, uint64_t,
                                                 const LivenessAnalysis &liveness,
                                                 const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() != sizeof(gfx1250::Vop3pMachineInst))
    return {};

  gfx1250::Vop3pMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.vdst == 255 || src.src0 == 254 || src.src1 == 254)
    return {};

  std::optional<uint32_t> literal_word;
  if (src.src0 == 255)
    literal_word = simm32_literal_word(inst, 0);
  else if (src.src1 == 255)
    literal_word = simm32_literal_word(inst, 1);
  if ((src.src0 == 255 || src.src1 == 255) && !literal_word)
    return {};

  std::vector<uint8_t> avoid{static_cast<uint8_t>(src.vdst)};
  add_avoid_src_vgpr(avoid, static_cast<uint16_t>(src.src0));
  add_avoid_src_vgpr(avoid, static_cast<uint16_t>(src.src1));
  constexpr uint16_t kTmpCount = 7;
  const auto tmp_base = find_free_vgpr_run_avoiding(inst, liveness, kTmpCount, avoid);
  if (!tmp_base || *tmp_base > 249u)
    return {};

  const auto a_lo = static_cast<uint8_t>(*tmp_base);
  const auto b_lo = static_cast<uint8_t>(*tmp_base + 1u);
  const auto a_hi = static_cast<uint8_t>(*tmp_base + 2u);
  const auto b_hi = static_cast<uint8_t>(*tmp_base + 3u);
  const auto packed_lo = static_cast<uint8_t>(*tmp_base + 4u);
  const auto packed_hi = static_cast<uint8_t>(*tmp_base + 5u);
  const auto round_tmp = static_cast<uint8_t>(*tmp_base + 6u);

  std::vector<uint32_t> words;
  words.reserve(32);
  if (!append_materialize_fma_mix_bf16_source(words, a_lo, static_cast<uint16_t>(src.src0),
                                              (src.opsel & 0x1u) != 0, literal_word) ||
      !append_materialize_fma_mix_bf16_source(words, b_lo, static_cast<uint16_t>(src.src1),
                                              (src.opsel & 0x2u) != 0, literal_word) ||
      !append_materialize_fma_mix_bf16_source(words, a_hi, static_cast<uint16_t>(src.src0),
                                              (src.opsel_hi & 0x1u) != 0, literal_word) ||
      !append_materialize_fma_mix_bf16_source(words, b_hi, static_cast<uint16_t>(src.src1),
                                              (src.opsel_hi & 0x2u) != 0, literal_word))
    return {};
  append_wait_valu_vgpr(words);

  constexpr uint16_t kOpAddF32 = 259;
  {
    const auto [w0, w1] = build_vop3_mod(kOpAddF32, a_lo, vgpr_src(a_lo), vgpr_src(b_lo), 0, 0, 0,
                                         src.clamp != 0, 0, static_cast<uint8_t>(src.neg & 0x3u));
    words.push_back(w0);
    words.push_back(w1);
  }
  {
    const auto [w0, w1] =
        build_vop3_mod(kOpAddF32, a_hi, vgpr_src(a_hi), vgpr_src(b_hi), 0, 0, 0, src.clamp != 0, 0,
                       static_cast<uint8_t>(src.neg_hi & 0x3u));
    words.push_back(w0);
    words.push_back(w1);
  }
  append_wait_valu_vgpr(words);

  append_f32_to_bf16_rne(words, packed_lo, round_tmp, a_lo);
  append_f32_to_bf16_rne(words, packed_hi, round_tmp, a_hi);
  append_wait_valu_vgpr(words);

  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpOrB32 = 28;
  append_vop2(words, kOpLshlrevB32, packed_hi, scalar_positive_inline_u32(16), packed_hi);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpOrB32, static_cast<uint8_t>(src.vdst), vgpr_src(packed_lo), packed_hi);
  return words;
}

std::vector<uint32_t> expand_v_pk_fma_bf16_vop3p(const Instruction &inst, uint32_t, uint64_t,
                                                 const LivenessAnalysis &liveness,
                                                 const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() != sizeof(gfx1250::Vop3pMachineInst))
    return {};

  gfx1250::Vop3pMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.vdst == 255 || src.src0 == 254 || src.src1 == 254 || src.src2 == 254)
    return {};

  std::optional<uint32_t> literal_word;
  if (src.src0 == 255)
    literal_word = simm32_literal_word(inst, 0);
  else if (src.src1 == 255)
    literal_word = simm32_literal_word(inst, 1);
  else if (src.src2 == 255)
    literal_word = simm32_literal_word(inst, 2);
  if ((src.src0 == 255 || src.src1 == 255 || src.src2 == 255) && !literal_word)
    return {};

  std::vector<uint8_t> avoid{static_cast<uint8_t>(src.vdst)};
  add_avoid_src_vgpr(avoid, static_cast<uint16_t>(src.src0));
  add_avoid_src_vgpr(avoid, static_cast<uint16_t>(src.src1));
  add_avoid_src_vgpr(avoid, static_cast<uint16_t>(src.src2));
  constexpr uint16_t kTmpCount = 7;
  const auto tmp_base = find_free_vgpr_run_avoiding(inst, liveness, kTmpCount, avoid);
  if (!tmp_base || *tmp_base > 249u)
    return {};

  const uint8_t a_lo = static_cast<uint8_t>(*tmp_base);
  const uint8_t b_lo = static_cast<uint8_t>(*tmp_base + 1u);
  const uint8_t c_lo = static_cast<uint8_t>(*tmp_base + 2u);
  const uint8_t a_hi = static_cast<uint8_t>(*tmp_base + 3u);
  const uint8_t b_hi = static_cast<uint8_t>(*tmp_base + 4u);
  const uint8_t c_hi = static_cast<uint8_t>(*tmp_base + 5u);
  const uint8_t round_tmp = static_cast<uint8_t>(*tmp_base + 6u);

  std::vector<uint32_t> words;
  words.reserve(48);
  if (!append_materialize_fma_mix_bf16_source(words, a_lo, static_cast<uint16_t>(src.src0),
                                              (src.opsel & 0x1u) != 0, literal_word) ||
      !append_materialize_fma_mix_bf16_source(words, b_lo, static_cast<uint16_t>(src.src1),
                                              (src.opsel & 0x2u) != 0, literal_word) ||
      !append_materialize_fma_mix_bf16_source(words, c_lo, static_cast<uint16_t>(src.src2),
                                              (src.opsel & 0x4u) != 0, literal_word) ||
      !append_materialize_fma_mix_bf16_source(words, a_hi, static_cast<uint16_t>(src.src0),
                                              (src.opsel_hi & 0x1u) != 0, literal_word) ||
      !append_materialize_fma_mix_bf16_source(words, b_hi, static_cast<uint16_t>(src.src1),
                                              (src.opsel_hi & 0x2u) != 0, literal_word) ||
      !append_materialize_fma_mix_bf16_source(words, c_hi, static_cast<uint16_t>(src.src2),
                                              src.pad_14 != 0, literal_word))
    return {};
  append_wait_valu_vgpr(words);

  constexpr uint16_t kOpFmaF32 = 531;
  {
    const auto [w0, w1] =
        build_vop3_mod(kOpFmaF32, a_lo, vgpr_src(a_lo), vgpr_src(b_lo), vgpr_src(c_lo), 0, 0,
                       src.clamp != 0, 0, static_cast<uint8_t>(src.neg & 0x7u));
    words.push_back(w0);
    words.push_back(w1);
  }
  {
    const auto [w0, w1] =
        build_vop3_mod(kOpFmaF32, a_hi, vgpr_src(a_hi), vgpr_src(b_hi), vgpr_src(c_hi), 0, 0,
                       src.clamp != 0, 0, static_cast<uint8_t>(src.neg_hi & 0x7u));
    words.push_back(w0);
    words.push_back(w1);
  }
  append_wait_valu_vgpr(words);

  append_f32_to_bf16_rne(words, a_lo, round_tmp, a_lo);
  append_f32_to_bf16_rne(words, a_hi, round_tmp, a_hi);
  append_wait_valu_vgpr(words);

  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpOrB32 = 28;
  append_vop2(words, kOpLshlrevB32, a_hi, scalar_positive_inline_u32(16), a_hi);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpOrB32, static_cast<uint8_t>(src.vdst), vgpr_src(a_lo), a_hi);
  return words;
}

std::vector<uint32_t> expand_v_cvt_pk_bf16_f32_vop3(const Instruction &inst, uint32_t, uint64_t,
                                                    const LivenessAnalysis &liveness,
                                                    const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3MachineInst))
    return {};

  gfx1250::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.vdst == 255 || src.abs != 0 || src.opsel != 0 || src.clamp != 0 || src.omod != 0 ||
      src.neg != 0 || src.src0 == 254 || src.src1 == 254)
    return {};

  std::optional<uint32_t> literal_word;
  if (src.src0 == 255)
    literal_word = simm32_literal_word(inst, 0);
  else if (src.src1 == 255)
    literal_word = simm32_literal_word(inst, 1);
  if ((src.src0 == 255 || src.src1 == 255) && !literal_word)
    return {};

  const auto src0_vgpr = vgpr_index(static_cast<uint16_t>(src.src0));
  const auto src1_vgpr = vgpr_index(static_cast<uint16_t>(src.src1));
  const bool materialize_source = !src0_vgpr || !src1_vgpr;

  std::vector<uint8_t> avoid;
  add_avoid_vgpr(avoid, static_cast<uint8_t>(src.vdst));
  add_avoid_src_vgpr(avoid, static_cast<uint16_t>(src.src0));
  add_avoid_src_vgpr(avoid, static_cast<uint16_t>(src.src1));

  const uint16_t tmp_count = materialize_source ? 4u : 3u;
  const auto tmp_base = find_free_vgpr_run_avoiding(inst, liveness, tmp_count, avoid);
  if (!tmp_base)
    return {};
  const auto lo = static_cast<uint8_t>(*tmp_base);
  const auto hi = static_cast<uint8_t>(*tmp_base + 1u);
  const auto tmp = static_cast<uint8_t>(*tmp_base + 2u);
  const auto src_tmp = static_cast<uint8_t>(*tmp_base + 3u);

  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpMovB32 = 1;
  constexpr uint8_t kOpOrB32 = 28;

  std::vector<uint32_t> words;
  words.reserve(18);
  const uint8_t src0 = src0_vgpr.value_or(src_tmp);
  if (!src0_vgpr)
    append_vop1(words, kOpMovB32, src_tmp, static_cast<uint16_t>(src.src0), literal_word);
  append_f32_to_bf16_rne(words, lo, tmp, src0);

  const uint8_t src1 = src1_vgpr.value_or(src_tmp);
  if (!src1_vgpr)
    append_vop1(words, kOpMovB32, src_tmp, static_cast<uint16_t>(src.src1), literal_word);
  append_f32_to_bf16_rne(words, hi, tmp, src1);
  append_vop2(words, kOpLshlrevB32, hi, scalar_positive_inline_u32(16), hi);
  append_vop2(words, kOpOrB32, static_cast<uint8_t>(src.vdst), vgpr_src(lo), hi);
  return words;
}

std::vector<uint32_t> expand_v_cvt_pk_f16_f32_vop3(const Instruction &inst, uint32_t, uint64_t,
                                                   const LivenessAnalysis &liveness,
                                                   const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3MachineInst))
    return {};

  gfx1250::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.vdst == 255 || src.abs != 0 || src.opsel != 0 || src.clamp != 0 || src.omod != 0 ||
      src.neg != 0)
    return {};

  if (src.src0 == 254 || src.src1 == 254)
    return {};

  std::optional<uint32_t> literal_word;
  if (src.src0 == 255)
    literal_word = simm32_literal_word(inst, 0);
  else if (src.src1 == 255)
    literal_word = simm32_literal_word(inst, 1);
  if ((src.src0 == 255 || src.src1 == 255) && !literal_word)
    return {};

  std::vector<uint8_t> avoid;
  add_avoid_vgpr(avoid, static_cast<uint8_t>(src.vdst));
  add_avoid_src_vgpr(avoid, static_cast<uint16_t>(src.src0));
  add_avoid_src_vgpr(avoid, static_cast<uint16_t>(src.src1));

  constexpr uint8_t kTmpCount = 2;
  auto tmp_base = find_free_vgpr_run_avoiding(inst, liveness, kTmpCount, avoid);
  std::optional<PrivateBorrowedVgprRun> borrowed;
  std::optional<uint16_t> exec_save;
  if (!tmp_base) {
    borrowed = find_private_borrowed_vgpr_run(liveness, kTmpCount, 2, avoid);
    if (!borrowed)
      return {};
    exec_save = liveness.find_free_sgpr_pair(&inst);
    if (!exec_save || *exec_save > 124)
      return {};
    tmp_base = borrowed->base;
  }
  const auto lo = static_cast<uint8_t>(*tmp_base);
  const auto hi = static_cast<uint8_t>(*tmp_base + 1u);

  constexpr uint16_t kOpCvtF16F32 = 0x18A;
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpOrB32 = 28;

  std::vector<uint32_t> words;
  words.reserve(10 + (borrowed ? borrowed->count * 6u + 16u : 0u));
  if (borrowed) {
    append_save_exec(words, static_cast<uint8_t>(*exec_save));
    append_private_borrow_save(words, *borrowed);
    append_restore_exec(words, static_cast<uint8_t>(*exec_save));
    append_wait_valu_vgpr(words);
  }
  append_vop3(words, kOpCvtF16F32, lo, static_cast<uint16_t>(src.src0), 0, 0,
              src.src0 == 255 ? literal_word : std::nullopt);
  append_vop3(words, kOpCvtF16F32, hi, static_cast<uint16_t>(src.src1), 0, 0,
              src.src1 == 255 ? literal_word : std::nullopt);
  // This expansion is emitted as already-translated RDNA4 code, so the outer
  // copied-instruction hazard pass does not see the RAW dependencies below.
  words.push_back(build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4));
  std::optional<uint32_t> lo_mask_literal;
  append_vop2(words, kOpAndB32, lo, literal_or_inline_u32(0xFFFFu, lo_mask_literal), lo,
              lo_mask_literal);
  append_vop2(words, kOpLshlrevB32, hi, scalar_positive_inline_u32(16), hi);
  words.push_back(build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4));
  append_vop2(words, kOpOrB32, static_cast<uint8_t>(src.vdst), vgpr_src(lo), hi);
  if (borrowed)
    append_private_borrow_restore(words, *borrowed, static_cast<uint8_t>(*exec_save));
  return words;
}

enum class ScaledLowpFormat : uint8_t { Fp4, Fp6, Bf6, Fp8, Bf8, F16, Bf16, F32 };

struct ScaledLowpCvtSpec {
  bool pack = false;
  bool stochastic = false;
  uint8_t count = 0;
  ScaledLowpFormat lowp = ScaledLowpFormat::Fp8;
  ScaledLowpFormat wide = ScaledLowpFormat::F32;
};

[[nodiscard]] std::optional<ScaledLowpCvtSpec> scaled_lowp_cvt_spec(uint16_t op) {
  using F = ScaledLowpFormat;
  switch (op) {
  case 0x297:
    return ScaledLowpCvtSpec{true, true, 8, F::Fp4, F::F32};
  case 0x298:
    return ScaledLowpCvtSpec{true, true, 8, F::Fp8, F::F32};
  case 0x299:
    return ScaledLowpCvtSpec{true, true, 8, F::Bf8, F::F32};
  case 0x29F:
    return ScaledLowpCvtSpec{false, false, 8, F::Fp4, F::F16};
  case 0x2A0:
    return ScaledLowpCvtSpec{false, false, 8, F::Fp4, F::Bf16};
  case 0x2A1:
    return ScaledLowpCvtSpec{false, false, 8, F::Fp4, F::F32};
  case 0x2A8:
    return ScaledLowpCvtSpec{false, false, 8, F::Fp8, F::F16};
  case 0x2A9:
    return ScaledLowpCvtSpec{false, false, 8, F::Fp8, F::Bf16};
  case 0x2AA:
    return ScaledLowpCvtSpec{false, false, 8, F::Fp8, F::F32};
  case 0x2AB:
    return ScaledLowpCvtSpec{false, false, 8, F::Bf8, F::F16};
  case 0x2AC:
    return ScaledLowpCvtSpec{false, false, 8, F::Bf8, F::Bf16};
  case 0x2AD:
    return ScaledLowpCvtSpec{false, false, 8, F::Bf8, F::F32};
  case 0x2B0:
    return ScaledLowpCvtSpec{true, false, 8, F::Fp4, F::F32};
  case 0x2B3:
    return ScaledLowpCvtSpec{true, false, 8, F::Fp4, F::F16};
  case 0x2B4:
    return ScaledLowpCvtSpec{true, false, 8, F::Fp8, F::Bf16};
  case 0x2B5:
    return ScaledLowpCvtSpec{true, false, 8, F::Bf8, F::Bf16};
  case 0x2B8:
    return ScaledLowpCvtSpec{true, false, 8, F::Fp4, F::Bf16};
  case 0x2B9:
    return ScaledLowpCvtSpec{true, true, 8, F::Fp4, F::F16};
  case 0x2BC:
    return ScaledLowpCvtSpec{true, true, 8, F::Fp4, F::Bf16};
  case 0x2BF:
    return ScaledLowpCvtSpec{true, true, 8, F::Fp8, F::F16};
  case 0x2C0:
    return ScaledLowpCvtSpec{true, true, 8, F::Fp8, F::Bf16};
  case 0x2C1:
    return ScaledLowpCvtSpec{true, true, 8, F::Bf8, F::F16};
  case 0x2C2:
    return ScaledLowpCvtSpec{true, true, 8, F::Bf8, F::Bf16};
  case 0x2C3:
    return ScaledLowpCvtSpec{true, false, 8, F::Fp8, F::F32};
  case 0x2C4:
    return ScaledLowpCvtSpec{true, false, 8, F::Fp8, F::F16};
  case 0x2C5:
    return ScaledLowpCvtSpec{true, false, 8, F::Bf8, F::F32};
  case 0x2C6:
    return ScaledLowpCvtSpec{true, false, 8, F::Bf8, F::F16};
  case 0x2C7:
    return ScaledLowpCvtSpec{false, false, 16, F::Fp6, F::F16};
  case 0x2C8:
    return ScaledLowpCvtSpec{false, false, 16, F::Fp6, F::Bf16};
  case 0x2C9:
    return ScaledLowpCvtSpec{false, false, 16, F::Fp6, F::F32};
  case 0x2CA:
    return ScaledLowpCvtSpec{false, false, 16, F::Bf6, F::F16};
  case 0x2CB:
    return ScaledLowpCvtSpec{false, false, 16, F::Bf6, F::Bf16};
  case 0x2CC:
    return ScaledLowpCvtSpec{false, false, 16, F::Bf6, F::F32};
  case 0x2CD:
    return ScaledLowpCvtSpec{true, false, 16, F::Fp6, F::F32};
  case 0x2CE:
    return ScaledLowpCvtSpec{true, false, 16, F::Bf6, F::F32};
  case 0x2CF:
    return ScaledLowpCvtSpec{true, false, 16, F::Fp6, F::F16};
  case 0x2D0:
    return ScaledLowpCvtSpec{true, false, 16, F::Bf6, F::F16};
  case 0x2D1:
    return ScaledLowpCvtSpec{true, false, 16, F::Fp6, F::Bf16};
  case 0x2D2:
    return ScaledLowpCvtSpec{true, false, 16, F::Bf6, F::Bf16};
  case 0x2D3:
    return ScaledLowpCvtSpec{true, true, 16, F::Fp6, F::F32};
  case 0x2D4:
    return ScaledLowpCvtSpec{true, true, 16, F::Bf6, F::F32};
  case 0x2D5:
    return ScaledLowpCvtSpec{true, true, 16, F::Fp6, F::F16};
  case 0x2D6:
    return ScaledLowpCvtSpec{true, true, 16, F::Bf6, F::F16};
  case 0x2D7:
    return ScaledLowpCvtSpec{true, true, 16, F::Fp6, F::Bf16};
  case 0x2D8:
    return ScaledLowpCvtSpec{true, true, 16, F::Bf6, F::Bf16};
  default:
    return std::nullopt;
  }
}

[[nodiscard]] constexpr uint8_t lowp_bits(ScaledLowpFormat format) {
  switch (format) {
  case ScaledLowpFormat::Fp4:
    return 4;
  case ScaledLowpFormat::Fp6:
  case ScaledLowpFormat::Bf6:
    return 6;
  case ScaledLowpFormat::Fp8:
  case ScaledLowpFormat::Bf8:
    return 8;
  default:
    return 0;
  }
}

[[nodiscard]] constexpr uint8_t wide_words(ScaledLowpFormat format, uint8_t count) {
  return format == ScaledLowpFormat::F32 ? count : static_cast<uint8_t>(count / 2u);
}

struct FiniteLowpSpec {
  uint8_t total_bits;
  uint8_t exponent_bits;
  uint8_t mantissa_bits;
  uint8_t bias;
};

[[nodiscard]] constexpr FiniteLowpSpec finite_lowp_spec(ScaledLowpFormat format) {
  switch (format) {
  case ScaledLowpFormat::Fp4:
    return {4, 2, 1, 1};
  case ScaledLowpFormat::Fp6:
    return {6, 2, 3, 1};
  case ScaledLowpFormat::Bf6:
    return {6, 3, 2, 3};
  default:
    return {};
  }
}

void append_f32_to_finite_lowp_rne(std::vector<uint32_t> &words, uint8_t result, uint8_t value,
                                   uint8_t abs_bits, uint8_t sign, uint8_t normal,
                                   uint8_t subnormal, uint8_t tmp, uint8_t predicate,
                                   FiniteLowpSpec spec) {
  constexpr uint8_t kOpCvtU32F32 = 7;
  constexpr uint8_t kOpMulF32 = 8;
  constexpr uint8_t kOpLshrrevB32 = 25;
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpOrB32 = 28;
  constexpr uint8_t kOpAddNcU32 = 37;
  constexpr uint16_t kOpCmpGtU32 = 76;
  constexpr uint16_t kOpCmpGeU32 = 78;
  constexpr uint16_t kOpCndmaskB32 = 257;
  constexpr uint8_t kSoppWaitAlu = 8;

  const uint32_t sign_mask = 1u << (spec.total_bits - 1u);
  const uint32_t max_magnitude = sign_mask - 1u;
  const uint8_t discarded = static_cast<uint8_t>(23u - spec.mantissa_bits);
  const uint32_t round_bias = (1u << (discarded - 1u)) - 1u;
  const uint32_t rebias = (127u - spec.bias) << spec.mantissa_bits;
  const uint32_t min_normal_bits = (127u + 1u - spec.bias) << 23u;
  const uint32_t subnormal_scale_bits = (127u + spec.bias + spec.mantissa_bits - 1u) << 23u;

  append_vop2(words, kOpAndB32, abs_bits, 255, value, 0x7FFFFFFFu);
  append_vop2(words, kOpLshrrevB32, sign,
              scalar_positive_inline_u32(static_cast<uint16_t>(32u - spec.total_bits)), value);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpAndB32, sign, 255, sign, sign_mask);

  append_vop2(words, kOpLshrrevB32, tmp, scalar_positive_inline_u32(discarded), abs_bits);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpAndB32, tmp, scalar_positive_inline_u32(1), tmp);
  append_vop2(words, kOpAddNcU32, normal, 255, abs_bits, round_bias);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpAddNcU32, normal, vgpr_src(tmp), normal);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpLshrrevB32, normal, scalar_positive_inline_u32(discarded), normal);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpAddNcU32, normal, 255, normal, 0u - rebias);

  append_vop2(words, kOpMulF32, subnormal, 255, abs_bits, subnormal_scale_bits);
  append_wait_valu_vgpr(words);
  append_vop1(words, kOpCvtU32F32, subnormal, vgpr_src(subnormal));
  append_wait_valu_vgpr(words);
  append_vop3(words, kOpCmpGeU32, predicate, vgpr_src(abs_bits), 255, 0, min_normal_bits);
  words.push_back(pack_sopp(kSoppWaitAlu, kWaitAluDepctrVaSdst0));
  append_vop3(words, kOpCndmaskB32, result, vgpr_src(subnormal), vgpr_src(normal), predicate);

  append_wait_valu_vgpr(words);
  append_vop3(words, kOpCmpGtU32, predicate, vgpr_src(result), 255, 0, max_magnitude);
  words.push_back(pack_sopp(kSoppWaitAlu, kWaitAluDepctrVaSdst0));
  append_vop3(words, kOpCndmaskB32, result, vgpr_src(result), 255, predicate, max_magnitude);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpOrB32, result, vgpr_src(sign), result);

  // Finite overflow and infinity already clamp to the signed max-finite code.
  // IEEE NaNs sort above the infinity bit pattern and map to canonical zero.
  append_wait_valu_vgpr(words);
  append_vop3(words, kOpCmpGtU32, predicate, vgpr_src(abs_bits), 255, 0, 0x7F800000u);
  words.push_back(pack_sopp(kSoppWaitAlu, kWaitAluDepctrVaSdst0));
  append_vop3(words, kOpCndmaskB32, result, vgpr_src(result), scalar_positive_inline_u32(0),
              predicate);
}

void append_f32_to_finite_lowp_sr(std::vector<uint32_t> &words, uint8_t result, uint8_t value,
                                  uint8_t seed, uint8_t abs_bits, uint8_t sign, uint8_t normal,
                                  uint8_t subnormal, uint8_t shift, uint8_t full_mant, uint8_t work,
                                  uint8_t predicate, FiniteLowpSpec spec) {
  constexpr uint8_t kOpMinU32 = 19;
  constexpr uint8_t kOpLshrrevB32 = 25;
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpOrB32 = 28;
  constexpr uint8_t kOpAddNcU32 = 37;
  constexpr uint8_t kOpSubNcU32 = 38;
  constexpr uint16_t kOpLshlrevB32Vop3 = 280;
  constexpr uint16_t kOpLshrrevB32Vop3 = 281;
  constexpr uint16_t kOpCmpGtU32 = 76;
  constexpr uint16_t kOpCmpGeU32 = 78;
  constexpr uint16_t kOpCndmaskB32 = 257;
  constexpr uint8_t kSoppWaitAlu = 8;

  const uint32_t sign_mask = 1u << (spec.total_bits - 1u);
  const uint32_t max_magnitude = sign_mask - 1u;
  const uint8_t discarded = static_cast<uint8_t>(23u - spec.mantissa_bits);
  const uint32_t discarded_mask = (1u << discarded) - 1u;
  const uint32_t rebias = (127u - spec.bias) << spec.mantissa_bits;
  const uint32_t min_normal_bits = (127u + 1u - spec.bias) << 23u;
  const uint32_t subnormal_shift_base = 151u - spec.bias - spec.mantissa_bits;

  append_vop2(words, kOpAndB32, abs_bits, 255, value, 0x7FFFFFFFu);
  append_vop2(words, kOpLshrrevB32, sign,
              scalar_positive_inline_u32(static_cast<uint16_t>(32u - spec.total_bits)), value);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpAndB32, sign, 255, sign, sign_mask);

  // Normal values retain exponent plus mantissa.  The high `discarded` seed
  // bits form a uniform addend over the truncated suffix, matching the gfx1250
  // scalar model's carry test exactly.
  append_vop2(words, kOpLshrrevB32, normal, scalar_positive_inline_u32(discarded), abs_bits);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpAddNcU32, normal, 255, normal, 0u - rebias);
  append_vop2(words, kOpAndB32, full_mant, 255, abs_bits, discarded_mask);
  append_vop2(words, kOpLshrrevB32, work,
              scalar_positive_inline_u32(static_cast<uint16_t>(32u - discarded)), seed);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpAddNcU32, full_mant, vgpr_src(work), full_mant);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpLshrrevB32, full_mant, scalar_positive_inline_u32(discarded), full_mant);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpAddNcU32, normal, vgpr_src(full_mant), normal);

  // Subnormal codes are integer multiples of 2^(1-bias-mantissa_bits).  Form
  // the integer quotient and remainder from the explicit f32 significand, add
  // the exponent-dependent high seed bits, then carry into the code.  The
  // gfx1250 model flushes shifts wider than the 24-bit significand.
  append_vop2(words, kOpLshrrevB32, shift, scalar_positive_inline_u32(23), abs_bits);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpSubNcU32, shift, 255, shift, subnormal_shift_base);
  append_vop2(words, kOpAndB32, full_mant, 255, abs_bits, 0x007FFFFFu);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpOrB32, full_mant, 255, full_mant, 0x00800000u);
  append_wait_valu_vgpr(words);
  append_vop3(words, kOpLshrrevB32Vop3, subnormal, vgpr_src(shift), vgpr_src(full_mant));
  append_wait_valu_vgpr(words);
  append_vop3(words, kOpLshlrevB32Vop3, work, vgpr_src(shift), vgpr_src(subnormal));
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpSubNcU32, full_mant, vgpr_src(full_mant), work);
  append_vop2(words, kOpSubNcU32, work, scalar_positive_inline_u32(32), shift);
  append_wait_valu_vgpr(words);
  append_vop3(words, kOpLshrrevB32Vop3, work, vgpr_src(work), vgpr_src(seed));
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpAddNcU32, full_mant, vgpr_src(work), full_mant);
  append_wait_valu_vgpr(words);
  append_vop3(words, kOpLshrrevB32Vop3, full_mant, vgpr_src(shift), vgpr_src(full_mant));
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpAddNcU32, subnormal, vgpr_src(full_mant), subnormal);
  append_vop2(words, kOpMinU32, subnormal,
              scalar_positive_inline_u32(static_cast<uint16_t>(1u << spec.mantissa_bits)),
              subnormal);

  append_vop3(words, kOpCmpGtU32, predicate, vgpr_src(shift), scalar_positive_inline_u32(24));
  words.push_back(pack_sopp(kSoppWaitAlu, kWaitAluDepctrVaSdst0));
  append_vop3(words, kOpCndmaskB32, subnormal, vgpr_src(subnormal), scalar_positive_inline_u32(0),
              predicate);
  append_vop3(words, kOpCmpGeU32, predicate, vgpr_src(abs_bits), 255, 0, min_normal_bits);
  words.push_back(pack_sopp(kSoppWaitAlu, kWaitAluDepctrVaSdst0));
  append_vop3(words, kOpCndmaskB32, result, vgpr_src(subnormal), vgpr_src(normal), predicate);

  append_wait_valu_vgpr(words);
  append_vop3(words, kOpCmpGtU32, predicate, vgpr_src(result), 255, 0, max_magnitude);
  words.push_back(pack_sopp(kSoppWaitAlu, kWaitAluDepctrVaSdst0));
  append_vop3(words, kOpCndmaskB32, result, vgpr_src(result), 255, predicate, max_magnitude);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpOrB32, result, vgpr_src(sign), result);

  append_wait_valu_vgpr(words);
  append_vop3(words, kOpCmpGtU32, predicate, vgpr_src(abs_bits), 255, 0, 0x7F800000u);
  words.push_back(pack_sopp(kSoppWaitAlu, kWaitAluDepctrVaSdst0));
  append_vop3(words, kOpCndmaskB32, result, vgpr_src(result), scalar_positive_inline_u32(0),
              predicate);
}

void append_finite_lowp_to_f32(std::vector<uint32_t> &words, uint8_t result, uint8_t raw,
                               uint8_t sign, uint8_t exponent, uint8_t mantissa, uint8_t subnormal,
                               uint8_t predicate, FiniteLowpSpec spec) {
  constexpr uint8_t kOpMovB32 = 1;
  constexpr uint8_t kOpCvtF32U32 = 6;
  constexpr uint8_t kOpMulF32 = 8;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpLshrrevB32 = 25;
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpOrB32 = 28;
  constexpr uint8_t kOpAddNcU32 = 37;
  constexpr uint16_t kOpCmpEqU32 = 74;
  constexpr uint16_t kOpCndmaskB32 = 257;
  constexpr uint8_t kSoppWaitAlu = 8;

  const uint32_t sign_mask = 1u << (spec.total_bits - 1u);
  const uint32_t exponent_mask = (1u << spec.exponent_bits) - 1u;
  const uint32_t mantissa_mask = (1u << spec.mantissa_bits) - 1u;
  const uint32_t subnormal_scale_bits = (127u + 1u - spec.bias - spec.mantissa_bits) << 23u;

  append_vop2(words, kOpAndB32, sign, 255, raw, sign_mask);
  append_vop2(words, kOpLshrrevB32, exponent, scalar_positive_inline_u32(spec.mantissa_bits), raw);
  append_vop2(words, kOpAndB32, mantissa, 255, raw, mantissa_mask);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpAndB32, exponent, 255, exponent, exponent_mask);
  append_vop2(words, kOpLshlrevB32, sign,
              scalar_positive_inline_u32(static_cast<uint16_t>(32u - spec.total_bits)), sign);
  append_vop1(words, kOpMovB32, subnormal, vgpr_src(mantissa));

  append_vop2(words, kOpAddNcU32, result, 255, exponent, 127u - spec.bias);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpLshlrevB32, result, scalar_positive_inline_u32(23), result);
  append_vop2(words, kOpLshlrevB32, mantissa,
              scalar_positive_inline_u32(static_cast<uint16_t>(23u - spec.mantissa_bits)),
              mantissa);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpOrB32, result, vgpr_src(mantissa), result);
  append_vop2(words, kOpOrB32, result, vgpr_src(sign), result);

  append_vop1(words, kOpCvtF32U32, subnormal, vgpr_src(subnormal));
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpMulF32, subnormal, 255, subnormal, subnormal_scale_bits);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpOrB32, subnormal, vgpr_src(sign), subnormal);
  append_vop3(words, kOpCmpEqU32, predicate, vgpr_src(exponent), scalar_positive_inline_u32(0));
  words.push_back(pack_sopp(kSoppWaitAlu, kWaitAluDepctrVaSdst0));
  append_vop3(words, kOpCndmaskB32, result, vgpr_src(result), vgpr_src(subnormal), predicate);
}

void append_advance_lfsr(std::vector<uint32_t> &words, uint8_t seed, uint8_t tmp) {
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpLshrrevB32 = 25;
  constexpr uint8_t kOpXorB32 = 29;
  constexpr uint16_t kOpMulLoU32 = 812;
  append_vop2(words, kOpLshrrevB32, tmp, scalar_positive_inline_u32(31), seed);
  append_vop2(words, kOpLshlrevB32, seed, scalar_positive_inline_u32(1), seed);
  append_wait_valu_vgpr(words);
  append_vop3(words, kOpMulLoU32, tmp, vgpr_src(tmp), 255, 0, 197u);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpXorB32, seed, vgpr_src(tmp), seed);
  append_wait_valu_vgpr(words);
}

void append_materialize_pack_input(std::vector<uint32_t> &words, uint8_t result, uint8_t src_base,
                                   uint8_t index, ScaledLowpFormat format) {
  constexpr uint8_t kOpMovB32 = 1;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint16_t kOpCvtF32F16 = 0x18B;
  if (format == ScaledLowpFormat::F32) {
    append_vop1(words, kOpMovB32, result, vgpr_src(static_cast<uint8_t>(src_base + index)));
  } else if (format == ScaledLowpFormat::F16) {
    append_vop3_mod(words, kOpCvtF32F16, result,
                    vgpr_src(static_cast<uint8_t>(src_base + index / 2u)), 0, 0,
                    static_cast<uint8_t>(index & 1u));
  } else {
    const uint8_t src_word = static_cast<uint8_t>(src_base + index / 2u);
    if ((index & 1u) == 0)
      append_vop2(words, kOpLshlrevB32, result, scalar_positive_inline_u32(16), src_word);
    else
      append_vop1(words, kOpMovB32, result, vgpr_src(src_word));
  }
  append_wait_valu_vgpr(words);
}

void append_insert_packed_code(std::vector<uint32_t> &words, uint8_t accum_base, uint8_t code,
                               uint8_t tmp, uint8_t index, uint8_t bits) {
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpLshrrevB32 = 25;
  constexpr uint8_t kOpOrB32 = 28;
  const uint16_t bit = static_cast<uint16_t>(index) * bits;
  const uint8_t word = static_cast<uint8_t>(bit / 32u);
  const uint8_t shift = static_cast<uint8_t>(bit & 31u);
  if (shift == 0) {
    append_vop2(words, kOpOrB32, static_cast<uint8_t>(accum_base + word), vgpr_src(code),
                static_cast<uint8_t>(accum_base + word));
  } else {
    append_vop2(words, kOpLshlrevB32, tmp, scalar_positive_inline_u32(shift), code);
    append_wait_valu_vgpr(words);
    append_vop2(words, kOpOrB32, static_cast<uint8_t>(accum_base + word), vgpr_src(tmp),
                static_cast<uint8_t>(accum_base + word));
  }
  if (static_cast<uint16_t>(shift) + bits > 32u) {
    append_vop2(words, kOpLshrrevB32, tmp,
                scalar_positive_inline_u32(static_cast<uint16_t>(32u - shift)), code);
    append_wait_valu_vgpr(words);
    append_vop2(words, kOpOrB32, static_cast<uint8_t>(accum_base + word + 1u), vgpr_src(tmp),
                static_cast<uint8_t>(accum_base + word + 1u));
  }
  append_wait_valu_vgpr(words);
}

void append_extract_packed_code(std::vector<uint32_t> &words, uint8_t result, uint8_t input_base,
                                uint8_t tmp, uint8_t index, uint8_t bits) {
  constexpr uint8_t kOpMovB32 = 1;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpLshrrevB32 = 25;
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpOrB32 = 28;
  const uint16_t bit = static_cast<uint16_t>(index) * bits;
  const uint8_t word = static_cast<uint8_t>(bit / 32u);
  const uint8_t shift = static_cast<uint8_t>(bit & 31u);
  if (shift == 0)
    append_vop1(words, kOpMovB32, result, vgpr_src(static_cast<uint8_t>(input_base + word)));
  else
    append_vop2(words, kOpLshrrevB32, result, scalar_positive_inline_u32(shift),
                static_cast<uint8_t>(input_base + word));
  if (static_cast<uint16_t>(shift) + bits > 32u) {
    append_vop2(words, kOpLshlrevB32, tmp,
                scalar_positive_inline_u32(static_cast<uint16_t>(32u - shift)),
                static_cast<uint8_t>(input_base + word + 1u));
    append_wait_valu_vgpr(words);
    append_vop2(words, kOpOrB32, result, vgpr_src(tmp), result);
  }
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpAndB32, result, 255, result, (1u << bits) - 1u);
  append_wait_valu_vgpr(words);
}

void append_materialize_e8m0_scale(std::vector<uint32_t> &words, uint8_t scale, uint8_t tmp,
                                   uint16_t encoded_src, uint8_t byte_index, uint8_t predicate,
                                   std::optional<uint32_t> literal) {
  constexpr uint8_t kOpMovB32 = 1;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpLshrrevB32 = 25;
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint16_t kOpCmpEqU32 = 74;
  constexpr uint16_t kOpCndmaskB32 = 257;
  constexpr uint8_t kSoppWaitAlu = 8;
  append_vop1(words, kOpMovB32, tmp, encoded_src, literal);
  if (byte_index != 0) {
    append_wait_valu_vgpr(words);
    append_vop2(words, kOpLshrrevB32, tmp,
                scalar_positive_inline_u32(static_cast<uint16_t>(byte_index * 8u)), tmp);
  }
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpAndB32, tmp, 255, tmp, 0xFFu);
  append_wait_valu_vgpr(words);
  append_vop2(words, kOpLshlrevB32, scale, scalar_positive_inline_u32(23), tmp);
  append_vop3(words, kOpCmpEqU32, predicate, vgpr_src(tmp), scalar_positive_inline_u32(0));
  words.push_back(pack_sopp(kSoppWaitAlu, kWaitAluDepctrVaSdst0));
  append_vop3(words, kOpCndmaskB32, scale, vgpr_src(scale), 255, predicate, 0x00400000u);
  append_wait_valu_vgpr(words);
  append_vop3(words, kOpCmpEqU32, predicate, vgpr_src(tmp), 255, 0, 0xFFu);
  words.push_back(pack_sopp(kSoppWaitAlu, kWaitAluDepctrVaSdst0));
  append_vop3(words, kOpCndmaskB32, scale, vgpr_src(scale), 255, predicate, 0x7FC00000u);
  append_wait_valu_vgpr(words);
}

ExpandResult expand_v_cvt_scaled_lowp_vop3(const Instruction &inst, uint32_t, uint64_t,
                                           const LivenessAnalysis &liveness,
                                           TranslationContext &context, const LaneLayout *,
                                           const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3MachineInst))
    return ExpandResult::not_handled();
  gfx1250::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  const auto spec = scaled_lowp_cvt_spec(src.op);
  if (!spec)
    return ExpandResult::not_handled();
  const uint8_t bits = lowp_bits(spec->lowp);
  const uint8_t packed_words = static_cast<uint8_t>((spec->count * bits + 31u) / 32u);
  const uint8_t wide_word_count = wide_words(spec->wide, spec->count);
  const uint8_t source_words = spec->pack ? wide_word_count : packed_words;
  const uint8_t destination_words = spec->pack ? packed_words : wide_word_count;

  if (src.vdst == 255 || static_cast<uint16_t>(src.vdst) + destination_words > 256u ||
      src.abs != 0 || src.clamp != 0 || src.omod != 0 || src.neg != 0 ||
      (spec->pack ? src.opsel != 0 : (src.opsel & ~0x3u) != 0))
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " uses unsupported scaled-conversion modifiers");
  const auto source_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src0), source_words);
  if (!source_base)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " requires a contiguous VGPR source tuple");

  const uint16_t seed_src = static_cast<uint16_t>(src.src1);
  const uint16_t scale_src = static_cast<uint16_t>(spec->stochastic ? src.src2 : src.src1);
  if (seed_src == 254 || scale_src == 254)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " does not support a literal64 scale or seed");
  std::optional<uint32_t> literal;
  const uint8_t literal_operand = spec->stochastic && seed_src == 255 ? 1
                                  : scale_src == 255                  ? (spec->stochastic ? 2 : 1)
                                                                      : 0;
  if (literal_operand != 0) {
    literal = simm32_literal_word(inst, literal_operand);
    if (!literal)
      return ExpandResult::failed(std::string(inst.mnemonic()) +
                                  " is missing its literal32 operand");
  }

  std::vector<uint8_t> avoid;
  add_avoid_vgpr_run(avoid, static_cast<uint8_t>(src.vdst), destination_words);
  add_avoid_vgpr_run(avoid, *source_base, source_words);
  add_avoid_src_vgpr(avoid, seed_src);
  add_avoid_src_vgpr(avoid, scale_src);
  constexpr uint16_t kScratchCount = 16;
  const auto scratch = find_free_vgpr_run_avoiding(inst, liveness, kScratchCount, avoid);
  const auto predicate = liveness.find_free_sgpr(&inst);
  if (!scratch || *scratch > 240 || !predicate || *predicate > 105)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " cannot allocate scaled-conversion scratch registers");

  const uint8_t state = static_cast<uint8_t>(*scratch);
  const uint8_t value = static_cast<uint8_t>(*scratch + 4u);
  const uint8_t code = static_cast<uint8_t>(*scratch + 5u);
  const uint8_t scale = static_cast<uint8_t>(*scratch + 6u);
  const uint8_t seed = static_cast<uint8_t>(*scratch + 7u);
  const uint8_t t0 = static_cast<uint8_t>(*scratch + 8u);
  const uint8_t t1 = static_cast<uint8_t>(*scratch + 9u);
  const uint8_t t2 = static_cast<uint8_t>(*scratch + 10u);
  const uint8_t t3 = static_cast<uint8_t>(*scratch + 11u);
  const uint8_t t4 = static_cast<uint8_t>(*scratch + 12u);
  const uint8_t t5 = static_cast<uint8_t>(*scratch + 13u);
  const uint8_t t6 = static_cast<uint8_t>(*scratch + 14u);
  const uint8_t out_word = static_cast<uint8_t>(*scratch + 15u);
  const uint8_t pred = static_cast<uint8_t>(*predicate);

  constexpr uint8_t kOpMovB32 = 1;
  constexpr uint8_t kOpRcpF32 = 42;
  constexpr uint8_t kOpMulF32 = 8;
  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpOrB32 = 28;
  constexpr uint16_t kOpCvtF16F32 = 0x18A;
  constexpr uint16_t kOpCvtF32Fp8 = 492;
  constexpr uint16_t kOpCvtF32Bf8 = 493;
  constexpr uint16_t kOpCvtPkFp8F32 = 873;
  constexpr uint16_t kOpCvtPkBf8F32 = 874;
  constexpr uint16_t kOpCvtSrFp8F32 = 875;
  constexpr uint16_t kOpCvtSrBf8F32 = 876;

  std::vector<uint32_t> words;
  words.reserve(spec->count * 48u + 64u);
  if (spec->pack) {
    append_vop1(words, kOpMovB32, scale, scale_src, scale_src == 255 ? literal : std::nullopt);
    append_wait_valu_vgpr(words);
    append_vop1(words, kOpRcpF32, scale, vgpr_src(scale));
    if (spec->stochastic)
      append_vop1(words, kOpMovB32, seed, seed_src, seed_src == 255 ? literal : std::nullopt);
    for (uint8_t word = 0; word < packed_words; ++word)
      append_vop1(words, kOpMovB32, static_cast<uint8_t>(state + word),
                  scalar_positive_inline_u32(0));
    append_wait_valu_vgpr(words);

    for (uint8_t index = 0; index < spec->count; ++index) {
      append_materialize_pack_input(words, value, *source_base, index, spec->wide);
      append_vop2(words, kOpMulF32, value, vgpr_src(scale), value);
      append_wait_valu_vgpr(words);
      if (spec->lowp == ScaledLowpFormat::Fp8 || spec->lowp == ScaledLowpFormat::Bf8) {
        const bool bf8 = spec->lowp == ScaledLowpFormat::Bf8;
        if (spec->stochastic) {
          append_vop3(words, bf8 ? kOpCvtSrBf8F32 : kOpCvtSrFp8F32, code, vgpr_src(value),
                      vgpr_src(seed));
          append_advance_lfsr(words, seed, t6);
        } else {
          append_vop3(words, bf8 ? kOpCvtPkBf8F32 : kOpCvtPkFp8F32, code, vgpr_src(value),
                      vgpr_src(value));
        }
        append_wait_valu_vgpr(words);
        append_vop2(words, kOpAndB32, code, 255, code, 0xFFu);
      } else {
        if (spec->stochastic) {
          append_f32_to_finite_lowp_sr(words, code, value, seed, t0, t1, t2, t3, t4, t5, t6, pred,
                                       finite_lowp_spec(spec->lowp));
          append_advance_lfsr(words, seed, t6);
        } else {
          append_f32_to_finite_lowp_rne(words, code, value, t0, t1, t2, t3, t4, pred,
                                        finite_lowp_spec(spec->lowp));
        }
      }
      append_wait_valu_vgpr(words);
      append_insert_packed_code(words, state, code, t6, index, bits);
    }
    for (uint8_t word = 0; word < packed_words; ++word)
      append_vop1(words, kOpMovB32, static_cast<uint8_t>(src.vdst + word),
                  vgpr_src(static_cast<uint8_t>(state + word)));
  } else {
    for (uint8_t word = 0; word < packed_words; ++word)
      append_vop1(words, kOpMovB32, static_cast<uint8_t>(state + word),
                  vgpr_src(static_cast<uint8_t>(*source_base + word)));
    append_materialize_e8m0_scale(words, scale, t6, scale_src,
                                  static_cast<uint8_t>(src.opsel & 0x3u), pred,
                                  scale_src == 255 ? literal : std::nullopt);

    for (uint8_t index = 0; index < spec->count; ++index) {
      if (spec->lowp == ScaledLowpFormat::Fp8 || spec->lowp == ScaledLowpFormat::Bf8) {
        const uint8_t src_word = static_cast<uint8_t>(state + index / 4u);
        const uint8_t byte = static_cast<uint8_t>(index & 3u);
        const uint8_t opsel = static_cast<uint8_t>(((byte & 1u) << 1u) | ((byte & 2u) >> 1u));
        append_vop3_mod(words, spec->lowp == ScaledLowpFormat::Bf8 ? kOpCvtF32Bf8 : kOpCvtF32Fp8,
                        value, vgpr_src(src_word), 0, 0, opsel);
      } else {
        append_extract_packed_code(words, code, state, t6, index, bits);
        append_finite_lowp_to_f32(words, value, code, t0, t1, t2, t3, pred,
                                  finite_lowp_spec(spec->lowp));
      }
      append_wait_valu_vgpr(words);
      append_vop2(words, kOpMulF32, value, vgpr_src(scale), value);
      append_wait_valu_vgpr(words);

      if (spec->wide == ScaledLowpFormat::F32) {
        append_vop1(words, kOpMovB32, static_cast<uint8_t>(src.vdst + index), vgpr_src(value));
        continue;
      }
      if ((index & 1u) == 0)
        append_vop1(words, kOpMovB32, out_word, scalar_positive_inline_u32(0));
      if (spec->wide == ScaledLowpFormat::F16) {
        append_vop3(words, kOpCvtF16F32, code, vgpr_src(value), 0);
        append_wait_valu_vgpr(words);
        append_vop2(words, kOpAndB32, code, 255, code, 0xFFFFu);
      } else {
        append_f32_to_bf16_rne(words, code, t6, value);
      }
      append_wait_valu_vgpr(words);
      if ((index & 1u) != 0) {
        append_vop2(words, kOpLshlrevB32, code, scalar_positive_inline_u32(16), code);
        append_wait_valu_vgpr(words);
      }
      append_vop2(words, kOpOrB32, out_word, vgpr_src(code), out_word);
      if ((index & 1u) != 0) {
        append_wait_valu_vgpr(words);
        append_vop1(words, kOpMovB32, static_cast<uint8_t>(src.vdst + index / 2u),
                    vgpr_src(out_word));
      }
    }
  }

  context.require_vgprs(static_cast<uint32_t>(*scratch) + kScratchCount);
  context.require_sgprs(static_cast<uint32_t>(*predicate) + 1u);
  return ExpandResult::success(std::move(words));
}

struct Vopd3Slot {
  uint16_t op = 0;
  uint8_t vdst = 0;
  uint16_t src0 = 0;
  uint8_t vsrc1 = 0;
  uint8_t vsrc2 = 0;
  uint8_t neg = 0;
  bool uses_vcc = false;
};

struct Vopd3Fields {
  Vopd3Slot x;
  Vopd3Slot y;
};

struct VopdXyFields {
  Vopd3Slot x;
  Vopd3Slot y;
  std::optional<uint32_t> literal_word;
};

[[nodiscard]] std::optional<Vopd3Fields> decode_vopd3(const Instruction &inst) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() != 3 * sizeof(uint32_t) || (raw[0] >> 24) != 0xCF)
    return std::nullopt;

  Vopd3Fields fields;
  fields.x.op = static_cast<uint16_t>((raw[0] >> 18) & 0x3Fu);
  fields.y.op = static_cast<uint16_t>((raw[0] >> 12) & 0x3Fu);
  fields.x.src0 = static_cast<uint16_t>(raw[0] & 0x1FFu);
  fields.y.src0 = static_cast<uint16_t>(raw[1] & 0x1FFu);
  fields.x.neg = static_cast<uint8_t>((raw[1] >> 9) & 0x7u);
  fields.y.neg = static_cast<uint8_t>((raw[1] >> 12) & 0x7u);
  fields.x.vsrc1 = static_cast<uint8_t>((raw[1] >> 16) & 0xFFu);
  fields.x.vsrc2 = static_cast<uint8_t>((raw[1] >> 24) & 0xFFu);
  fields.x.vdst = static_cast<uint8_t>(raw[2] & 0xFFu);
  fields.y.vsrc1 = static_cast<uint8_t>((raw[2] >> 8) & 0xFFu);
  fields.y.vsrc2 = static_cast<uint8_t>((raw[2] >> 16) & 0xFFu);
  fields.y.vdst = static_cast<uint8_t>((raw[2] >> 24) & 0xFFu);
  return fields;
}

[[nodiscard]] std::optional<VopdXyFields> decode_vopd_xy(const Instruction &inst) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() < 2 * static_cast<int>(sizeof(uint32_t)) || (raw[0] >> 26) != 0x32)
    return std::nullopt;

  VopdXyFields fields;
  fields.x.op = static_cast<uint16_t>((raw[0] >> 22) & 0xFu);
  fields.y.op = static_cast<uint16_t>((raw[0] >> 17) & 0x1Fu);
  fields.x.src0 = static_cast<uint16_t>(raw[0] & 0x1FFu);
  fields.x.vsrc1 = static_cast<uint8_t>((raw[0] >> 9) & 0xFFu);
  fields.y.src0 = static_cast<uint16_t>(raw[1] & 0x1FFu);
  fields.y.vsrc1 = static_cast<uint8_t>((raw[1] >> 9) & 0xFFu);
  fields.x.vdst = static_cast<uint8_t>((raw[1] >> 24) & 0xFFu);
  const uint16_t y_vdst_hi = static_cast<uint16_t>((raw[1] >> 17) & 0x7Fu);
  fields.y.vdst = static_cast<uint8_t>((y_vdst_hi << 1) | ((~fields.x.vdst) & 1u));
  fields.x.uses_vcc = fields.x.op == 9;
  fields.y.uses_vcc = fields.y.op == 9;

  const bool has_literal = fields.x.src0 == 255 || fields.y.src0 == 255 || fields.x.op == 1 ||
                           fields.x.op == 2 || fields.y.op == 1 || fields.y.op == 2;
  const int expected_size =
      has_literal ? 3 * static_cast<int>(sizeof(uint32_t)) : 2 * static_cast<int>(sizeof(uint32_t));
  if (inst.size() != expected_size)
    return std::nullopt;
  if (has_literal)
    fields.literal_word = raw[2];
  return fields;
}

[[nodiscard]] constexpr uint16_t vopd_vgpr_src(uint8_t vsrc) {
  return static_cast<uint16_t>(256u + vsrc);
}

[[nodiscard]] constexpr bool vopd_slot_reads_src1(uint16_t op) {
  return op != 8; // v_dual_mov_b32 reads src0 only.
}

[[nodiscard]] constexpr bool vopd_slot_reads_vsrc2(uint16_t op) {
  return op == 19; // v_dual_fma_f32
}

[[nodiscard]] constexpr bool vopd_slot_reads_old_dst(uint16_t op) {
  return op == 0; // v_dual_fmac_f32 accumulates into the old destination value.
}

[[nodiscard]] constexpr bool vopd_is_float32_op(uint16_t op) {
  switch (op) {
  case 0:
  case 3:
  case 4:
  case 5:
  case 6:
  case 7:
  case 10:
  case 11:
  case 19:
    return true;
  default:
    return false;
  }
}

[[nodiscard]] constexpr uint8_t vopd_neg_mask(const Vopd3Slot &slot) {
  if (!vopd_is_float32_op(slot.op))
    return 0;
  return static_cast<uint8_t>(slot.neg & (slot.op == 19 ? 0x7u : 0x3u));
}

[[nodiscard]] std::optional<uint16_t> rdna4_vop3_op_for_vopd_slot(const Vopd3Slot &slot) {
  switch (slot.op) {
  case 0: // v_dual_fmac_f32
    return 299;
  case 3: // v_dual_mul_f32
    return 264;
  case 4: // v_dual_add_f32
    return 259;
  case 5: // v_dual_sub_f32
    return 260;
  case 6: // v_dual_subrev_f32
    return 261;
  case 7: // v_dual_mul_dx9_zero_f32
    return 263;
  case 9: // v_dual_cndmask_b32
    return 257;
  case 10: // v_dual_max_num_f32
    return 278;
  case 11: // v_dual_min_num_f32
    return 277;
  case 16: // v_dual_add_nc_u32
    return 293;
  case 17: // v_dual_lshlrev_b32
    return 280;
  case 20: // v_dual_sub_nc_u32
    return 294;
  case 21: // v_dual_lshrrev_b32
    return 281;
  case 22: // v_dual_ashrrev_i32
    return 282;
  case 23: // v_dual_max_i32
    return 274;
  case 24: // v_dual_min_i32
    return 273;
  case 19: // v_dual_fma_f32
    return 531;
  default:
    return std::nullopt;
  }
}

[[nodiscard]] std::optional<uint8_t> rdna4_vop2_op_for_vopd_slot(const Vopd3Slot &slot) {
  if (vopd_neg_mask(slot) != 0)
    return std::nullopt;

  switch (slot.op) {
  case 9: // v_dual_cndmask_b32 with VCC
    return slot.uses_vcc ? std::optional<uint8_t>(1) : std::nullopt;
  case 0: // v_dual_fmac_f32
    return 43;
  case 1: // v_dual_fmaak_f32
    return 45;
  case 2: // v_dual_fmamk_f32
    return 44;
  case 3: // v_dual_mul_f32
    return 8;
  case 4: // v_dual_add_f32
    return 3;
  case 5: // v_dual_sub_f32
    return 4;
  case 6: // v_dual_subrev_f32
    return 5;
  case 7: // v_dual_mul_dx9_zero_f32
    return 7;
  case 10: // v_dual_max_num_f32
    return 22;
  case 11: // v_dual_min_num_f32
    return 21;
  case 16: // v_dual_add_nc_u32
    return 37;
  case 17: // v_dual_lshlrev_b32
    return 24;
  case 20: // v_dual_sub_nc_u32
    return 38;
  case 21: // v_dual_lshrrev_b32
    return 25;
  case 22: // v_dual_ashrrev_i32
    return 26;
  case 23: // v_dual_max_i32
    return 18;
  case 24: // v_dual_min_i32
    return 17;
  default:
    return std::nullopt;
  }
}

[[nodiscard]] std::optional<uint8_t> rdna4_vop2_op_for_vopd_bitop2(uint8_t truth_table) {
  switch (truth_table) {
  case 0x40:
    return 27; // v_and_b32
  case 0x54:
    return 28; // v_or_b32
  case 0x14:
    return 29; // v_xor_b32
  default:
    return std::nullopt;
  }
}

[[nodiscard]] std::vector<uint8_t> vopd_slot_vgpr_uses(const Vopd3Slot &slot) {
  std::vector<uint8_t> uses;
  if (auto src0_vgpr = vgpr_index(slot.src0))
    uses.push_back(*src0_vgpr);
  if (vopd_slot_reads_src1(slot.op))
    uses.push_back(slot.vsrc1);
  if (vopd_slot_reads_vsrc2(slot.op))
    uses.push_back(slot.vsrc2);
  if (vopd_slot_reads_old_dst(slot.op))
    uses.push_back(slot.vdst);
  return uses;
}

[[nodiscard]] bool vopd_slot_uses_vgpr(const Vopd3Slot &slot, uint8_t vgpr) {
  const auto uses = vopd_slot_vgpr_uses(slot);
  return std::ranges::find(uses, vgpr) != uses.end();
}

[[nodiscard]] std::vector<uint32_t> lower_vopd_slot(const Vopd3Slot &slot,
                                                    std::optional<uint32_t> literal_word) {
  if (slot.src0 == 254 || (slot.src0 == 255 && !literal_word))
    return {};

  if (slot.op == 8) {
    constexpr uint8_t kOpMovB32 = 1;
    std::vector<uint32_t> words;
    words.reserve(literal_word && slot.src0 == 255 ? 2 : 1);
    append_vop1(words, kOpMovB32, slot.vdst, slot.src0, literal_word);
    return words;
  }

  if (slot.op == 18) {
    const auto rdna4_op = rdna4_vop2_op_for_vopd_bitop2(slot.vsrc2);
    if (!rdna4_op)
      return {};
    std::vector<uint32_t> words;
    words.reserve(literal_word && slot.src0 == 255 ? 2 : 1);
    append_vop2(words, *rdna4_op, slot.vdst, slot.src0, slot.vsrc1,
                slot.src0 == 255 ? literal_word : std::nullopt);
    return words;
  }

  if ((slot.op == 1 || slot.op == 2) && (!literal_word || slot.src0 == 255))
    return {};

  if (const auto rdna4_op = rdna4_vop2_op_for_vopd_slot(slot)) {
    std::vector<uint32_t> words;
    words.reserve(literal_word && (slot.src0 == 255 || slot.op == 1 || slot.op == 2) ? 2 : 1);
    append_vop2(words, *rdna4_op, slot.vdst, slot.src0, slot.vsrc1,
                (slot.src0 == 255 || slot.op == 1 || slot.op == 2) ? literal_word : std::nullopt);
    return words;
  }

  if (slot.op == 19) {
    if (slot.src0 == 255)
      return {};
    if (vopd_neg_mask(slot) == 0) {
      constexpr uint8_t kOpMulF32 = 8;
      constexpr uint8_t kOpAddF32 = 3;
      std::vector<uint32_t> words;
      words.reserve(2);
      append_vop2(words, kOpMulF32, slot.vdst, slot.src0, slot.vsrc1);
      append_vop2(words, kOpAddF32, slot.vdst, vopd_vgpr_src(slot.vsrc2), slot.vdst);
      return words;
    }
    auto [w0, w1] = build_vop3_mod(531, slot.vdst, slot.src0, vopd_vgpr_src(slot.vsrc1),
                                   vopd_vgpr_src(slot.vsrc2), 0, 0, false, 0, vopd_neg_mask(slot));
    return {w0, w1};
  }

  const auto rdna4_op = rdna4_vop3_op_for_vopd_slot(slot);
  if (!rdna4_op)
    return {};

  if (slot.src0 == 255)
    return {};
  const uint16_t src2 = (slot.op == 9)
                            ? static_cast<uint16_t>(slot.vsrc2)
                            : (vopd_slot_reads_vsrc2(slot.op) ? vopd_vgpr_src(slot.vsrc2) : 0);
  auto [w0, w1] = build_vop3_mod(*rdna4_op, slot.vdst, slot.src0, vopd_vgpr_src(slot.vsrc1), src2,
                                 0, 0, false, 0, vopd_neg_mask(slot));
  return {w0, w1};
}

std::vector<uint32_t> expand_vopd3(const Instruction &inst, uint32_t host_arch, uint64_t,
                                   const LivenessAnalysis &, const LaneLayout *,
                                   const LaneLayout *) {
  auto fields = decode_vopd3(inst);
  if (!fields || fields->x.vdst == fields->y.vdst)
    return {};

  const bool xclobbers_y = vopd_slot_uses_vgpr(fields->y, fields->x.vdst);
  const bool yclobbers_x = vopd_slot_uses_vgpr(fields->x, fields->y.vdst);
  if (xclobbers_y && yclobbers_x)
    return {};

  const auto &first = xclobbers_y ? fields->y : fields->x;
  const auto &second = xclobbers_y ? fields->x : fields->y;
  auto first_words = lower_vopd_slot(first, std::nullopt);
  auto second_words = lower_vopd_slot(second, std::nullopt);
  if (first_words.empty() || second_words.empty())
    return {};

  std::vector<uint32_t> words;
  words.reserve(first_words.size() + second_words.size());
  words.insert(words.end(), first_words.begin(), first_words.end());
  words.insert(words.end(), second_words.begin(), second_words.end());
  while (words.size() < 3)
    words.push_back(build_s_nop(0, static_cast<rj_code_arch_t>(host_arch)));
  return words;
}

std::vector<uint32_t> expand_vopd_xy(const Instruction &inst, uint32_t host_arch, uint64_t,
                                     const LivenessAnalysis &, const LaneLayout *,
                                     const LaneLayout *) {
  auto fields = decode_vopd_xy(inst);
  if (!fields || fields->x.vdst == fields->y.vdst)
    return {};

  const bool xclobbers_y = vopd_slot_uses_vgpr(fields->y, fields->x.vdst);
  const bool yclobbers_x = vopd_slot_uses_vgpr(fields->x, fields->y.vdst);
  if (xclobbers_y && yclobbers_x)
    return {};

  const auto &first = xclobbers_y ? fields->y : fields->x;
  const auto &second = xclobbers_y ? fields->x : fields->y;
  auto first_words = lower_vopd_slot(first, fields->literal_word);
  auto second_words = lower_vopd_slot(second, fields->literal_word);
  if (first_words.empty() || second_words.empty())
    return {};

  std::vector<uint32_t> words;
  words.reserve(first_words.size() + second_words.size());
  words.insert(words.end(), first_words.begin(), first_words.end());
  words.insert(words.end(), second_words.begin(), second_words.end());
  while (words.size() * sizeof(uint32_t) < static_cast<size_t>(inst.size()))
    words.push_back(build_s_nop(0, static_cast<rj_code_arch_t>(host_arch)));
  return words;
}

std::vector<uint32_t> expand_s_wait_kmcnt(const Instruction &inst, uint32_t, uint64_t,
                                          const LivenessAnalysis &, const LaneLayout *,
                                          const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::SoppMachineInst))
    return {};

  gfx1250::SoppMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  constexpr uint8_t kOpWaitLoadcnt = 64;
  constexpr uint8_t kOpWaitKmcnt = 71;
  return {pack_sopp(kOpWaitLoadcnt, src.simm16), pack_sopp(kOpWaitKmcnt, src.simm16)};
}

template <auto Fn>
ExpandResult legacy_expand_adapter(const Instruction &inst, uint32_t host_arch, uint64_t offset,
                                   const LivenessAnalysis &liveness, TranslationContext &,
                                   const LaneLayout *guest_layout, const LaneLayout *host_layout) {
  auto words = Fn(inst, host_arch, offset, liveness, guest_layout, host_layout);
  if (words.empty())
    return ExpandResult::not_handled();
  return ExpandResult::success(std::move(words));
}

constexpr uint16_t kEncVopd = 0x032;
constexpr uint16_t kEncVop2MinNumF64 = 0x034;
constexpr uint16_t kEncVop2MaxNumF64 = 0x038;
constexpr uint16_t kEncVop1_0 = 0x0FC;
constexpr uint16_t kEncVop1_1 = 0x0FD;
constexpr uint16_t kEncVop1_2 = 0x0FE;
constexpr uint16_t kEncVop1_3 = 0x0FF;
constexpr uint16_t kEncVop3p = 0x198;
constexpr uint16_t kEncVop3p1 = 0x199;
constexpr uint16_t kEncVopd3 = 0x0CF;
constexpr uint16_t kEncSop2SAndB64 = 0x117;
constexpr uint16_t kEncSop2SOrB64 = 0x119;
constexpr uint16_t kEncSop2SAddNcU64 = 0x153;
constexpr uint16_t kEncSop2SSubNcU64 = 0x154;
constexpr uint16_t kEncSop2SMulU64 = 0x155;
constexpr uint16_t kEncSopkMovk = 0x160;
constexpr uint16_t kEncSopkGetreg = 0x171;
constexpr uint16_t kEncSopkSetregImm32 = 0x173;
constexpr uint16_t kEncSop1 = 0x17D;
constexpr uint16_t kEncSopc = 0x17E;
constexpr uint16_t kEncSopp = 0x17F;
constexpr uint16_t kEncSmem0 = 0x1E8;
constexpr uint16_t kEncSmem1 = 0x1E9;
constexpr uint16_t kEncSmem2 = 0x1EA;
constexpr uint16_t kEncSmem3 = 0x1EB;
constexpr uint16_t kEncVop2AddNcU64_0 = 0x0A0;
constexpr uint16_t kEncVop2AddNcU64_1 = 0x0A1;
constexpr uint16_t kEncVop2AddNcU64_2 = 0x0A2;
constexpr uint16_t kEncVop2AddNcU64_3 = 0x0A3;
constexpr uint16_t kEncVop2_0 = 0x0A4;
constexpr uint16_t kEncVop2_1 = 0x0A5;
constexpr uint16_t kEncVop2_2 = 0x0A6;
constexpr uint16_t kEncVop2_3 = 0x0A7;
constexpr uint16_t kEncVop2MulU64_0 = 0x0A8;
constexpr uint16_t kEncVop2MulU64_1 = 0x0A9;
constexpr uint16_t kEncVop2MulU64_2 = 0x0AA;
constexpr uint16_t kEncVop2MulU64_3 = 0x0AB;
constexpr uint16_t kEncVop2AddF16_0 = 0x0C8;
constexpr uint16_t kEncVop2AddF16_1 = 0x0C9;
constexpr uint16_t kEncVop2AddF16_2 = 0x0CA;
constexpr uint16_t kEncVop2AddF16_3 = 0x0CB;
constexpr uint16_t kEncVop2LshlrevB64 = 0x07C;
constexpr uint16_t kEncVop3_0 = 0x1A8;
constexpr uint16_t kEncVop3_1 = 0x1A9;
constexpr uint16_t kEncVop3_2 = 0x1AA;
constexpr uint16_t kEncVop3_3 = 0x1AB;
constexpr uint16_t kEncVop3_4 = 0x1AC;
constexpr uint16_t kEncVop3_5 = 0x1AD;
constexpr uint16_t kEncVop3_6 = 0x1AE;
constexpr uint16_t kEncVop3_7 = 0x1AF;
constexpr uint16_t kEncVdsStore2AddrB64 = 0x1B2;
constexpr uint16_t kEncVdsTranspose = 0x1B7;
constexpr uint16_t kEncVglobal = 0x1DC;
constexpr uint16_t kEncVglobal1 = 0x1DD;
constexpr uint16_t kEncVscratch = 0x1DA;
constexpr uint16_t kEncVscratch1 = 0x1DB;
constexpr uint16_t kEncVimage = 0x1A0;
constexpr uint16_t kOpSMovB64 = 1;
constexpr uint16_t kOpSMovkI32 = 0;
constexpr uint16_t kOpSMovB32 = 0;
constexpr uint16_t kOpSGetPcI64 = 71;
constexpr uint16_t kOpSSetPcI64 = 72;
constexpr uint16_t kOpSSwapPcI64 = 73;
constexpr uint16_t kOpSSendmsgRtnB32 = 0x4C;
constexpr uint16_t kOpSSendmsgRtnB64 = 0x4D;
constexpr uint16_t kOpSCmpLgU64 = 17;
constexpr uint16_t kOpSGetregB32 = 17;
constexpr uint16_t kOpSSetregImm32B32 = 19;
constexpr uint16_t kOpSAndB64 = 23;
constexpr uint16_t kOpSOrB64 = 25;
constexpr uint16_t kOpSAddNcU64 = 83;
constexpr uint16_t kOpSSubNcU64 = 84;
constexpr uint16_t kOpSClause = 5;
constexpr uint16_t kOpSSetVgprMsb = 6;
constexpr uint16_t kOpSWaitKmcnt = 0x47;
constexpr uint16_t kOpSWaitTensorcnt = 0x4B;
constexpr uint16_t kOpVCvtF32F16E32 = 11;
constexpr uint16_t kOpVMinNumF64E32 = 13;
constexpr uint16_t kOpVMaxNumF64E32 = 14;
constexpr uint16_t kOpVMovB16Vop1 = 28;
constexpr uint16_t kOpVMovB64Vop1 = 29;
constexpr uint16_t kOpVMovB64Vop3 = 413;
constexpr uint16_t kOpVMulU64Vop3 = 0;
constexpr uint16_t kOpVLshlAddU32Vop3 = 582;
constexpr uint16_t kOpVLshlAddU64Vop3 = 594;
constexpr uint16_t kOpVLshlOrB32Vop3 = 598;
constexpr uint16_t kOpVMaxU64Vop3 = 793;
constexpr uint16_t kOpVMinU64Vop3 = 792;
constexpr uint16_t kOpVMinI64Vop3 = 794;
constexpr uint16_t kOpVMaxI64Vop3 = 795;
constexpr uint16_t kOpVCvtPkFp8F32Vop3 = 873;
constexpr uint16_t kOpVCvtSrFp8F32Vop3 = 875;
constexpr uint16_t kOpVCvtPkBf16F32Vop3 = 877;
constexpr uint16_t kOpVCvtSrPkBf16F32Vop3 = 878;
constexpr uint16_t kOpVCvtPkF16F32Vop3 = 879;
constexpr uint16_t kOpVCvtSrPkF16F32Vop3 = 880;
constexpr uint16_t kOpVFmaMixF32Vop3p = 32;
constexpr uint16_t kOpVFmaMixF32Bf16Vop3p = 0x3D;
constexpr uint16_t kOpVFmaMixloBf16Vop3p = 0x3E;
constexpr uint16_t kOpVPkFmaF32Vop3p = 31;
constexpr uint16_t kOpVPkFmaBf16Vop3p = 17;
constexpr uint16_t kOpVPkAddBf16Vop3p = 35;
constexpr uint16_t kOpVPkMulF32Vop3p = 40;
constexpr uint16_t kOpVPkAddF32Vop3p = 41;
constexpr uint16_t kOpVAddNcU64Vop3 = 296;
constexpr uint16_t kOpVAddNcU64E32 = 40;
constexpr uint16_t kOpVSubNcU64Vop3 = 297;
constexpr uint16_t kOpVSubNcU64E32 = 41;
constexpr uint16_t kOpVAddF16E32 = 50;
constexpr uint16_t kOpVCvtF32Bf16E32 = 114;
constexpr uint16_t kOpVCvtPkF16Fp8E32 = 117;
constexpr uint16_t kOpVCvtPkF16Bf8E32 = 118;
constexpr uint16_t kOpVCvtF16Fp8E32 = 119;
constexpr uint16_t kOpVCvtF16Bf8E32 = 120;
constexpr uint16_t kOpVLshlrevB64E32 = 31;
constexpr uint16_t kOpVMulU64E32 = 42;
constexpr uint16_t kOpVPermlaneBcastB32Vop3 = 0x270;
constexpr uint16_t kOpVPermlaneUpB32Vop3 = 0x271;
constexpr uint16_t kOpVPermlaneDownB32Vop3 = 0x272;
constexpr uint16_t kOpVPermlaneXorB32Vop3 = 0x273;
constexpr uint16_t kOpVPermlaneIdxGenB32Vop3 = 0x314;
constexpr uint16_t kOpVBitop3B16Vop3 = 563;
constexpr uint16_t kOpVBitop3B32Vop3 = 564;
constexpr uint16_t kOpVMadU32Vop3 = 565;
constexpr uint16_t kOpVMadNcU64U32Vop3 = 762;
constexpr uint16_t kOpVMadNcI64I32Vop3 = 763;
constexpr uint16_t kOpVWmmaScaleF32_16x16x128F8f6f4 = 0x35;
constexpr uint16_t kOpVWmmaScale16F32_16x16x128F8f6f4 = 0x3A;
constexpr uint16_t kOpVCvtF32Bf16Vop3 = 498;
constexpr uint16_t kOpVCvtF32Fp8Vop3 = 492;
constexpr uint16_t kOpVCvtPkF16Fp8Vop3 = 501;
constexpr uint16_t kOpVCvtPkF16Bf8Vop3 = 502;
constexpr uint16_t kOpVCvtF16Fp8Vop3 = 503;
constexpr uint16_t kOpVCvtF16Bf8Vop3 = 504;
constexpr uint16_t kOpVCvtPkFp8F16Vop3 = 882;
constexpr uint16_t kOpVCvtPkBf8F16Vop3 = 883;
constexpr uint16_t kOpVCvtSrFp8F16Vop3 = 884;
constexpr uint16_t kOpVCvtSrBf8F16Vop3 = 885;
constexpr uint16_t kOpVWmmaF32_16x16x128F8f6f4 = 0x33;
constexpr uint16_t kOpVWmmaF32_16x16x4F32 = 0x5D;
constexpr uint16_t kOpVWmmaF32_16x16x32F16 = 0x60;
constexpr uint16_t kOpVWmmaF16_16x16x32F16 = 0x61;
constexpr uint16_t kOpVWmmaF32_16x16x32Bf16 = 0x62;
constexpr uint16_t kOpVWmmaBf16_16x16x32Bf16 = 0x63;
constexpr uint16_t kOpVWmmaBf16f32_16x16x32Bf16 = 0x64;
constexpr uint16_t kOpVSwmmacF32_16x16x64F16 = 0x65;
constexpr uint16_t kOpVSwmmacF32_16x16x64Bf16 = 0x66;
constexpr uint16_t kOpVSwmmacF16_16x16x64F16 = 0x67;
constexpr uint16_t kOpVSwmmacBf16_16x16x64Bf16 = 0x68;
constexpr uint16_t kOpVSwmmacBf16f32_16x16x64Bf16 = 0x69;
constexpr uint16_t kOpVWmmaF32_16x16x64Fp8Fp8 = 0x6A;
constexpr uint16_t kOpVWmmaF32_16x16x64Fp8Bf8 = 0x6B;
constexpr uint16_t kOpVWmmaF32_16x16x64Bf8Fp8 = 0x6C;
constexpr uint16_t kOpVWmmaF32_16x16x64Bf8Bf8 = 0x6D;
constexpr uint16_t kOpVWmmaF16_16x16x64Fp8Fp8 = 0x6E;
constexpr uint16_t kOpVWmmaF16_16x16x64Fp8Bf8 = 0x6F;
constexpr uint16_t kOpVWmmaF16_16x16x64Bf8Fp8 = 0x70;
constexpr uint16_t kOpVWmmaF16_16x16x64Bf8Bf8 = 0x71;
constexpr uint16_t kOpVWmmaI32_16x16x64Iu8 = 0x72;
constexpr uint16_t kOpVSwmmacF32_16x16x128Fp8Fp8 = 0x73;
constexpr uint16_t kOpVSwmmacF32_16x16x128Fp8Bf8 = 0x74;
constexpr uint16_t kOpVSwmmacF32_16x16x128Bf8Fp8 = 0x75;
constexpr uint16_t kOpVSwmmacF32_16x16x128Bf8Bf8 = 0x76;
constexpr uint16_t kOpVSwmmacF16_16x16x128Fp8Fp8 = 0x77;
constexpr uint16_t kOpVSwmmacF16_16x16x128Fp8Bf8 = 0x78;
constexpr uint16_t kOpVSwmmacF16_16x16x128Bf8Fp8 = 0x79;
constexpr uint16_t kOpVSwmmacF16_16x16x128Bf8Bf8 = 0x7A;
constexpr uint16_t kOpVSwmmacI32_16x16x128Iu8 = 0x7B;
constexpr uint16_t kOpVWmmaF32_16x16x128Fp8Fp8 = 0x80;
constexpr uint16_t kOpVWmmaF32_16x16x128Fp8Bf8 = 0x81;
constexpr uint16_t kOpVWmmaF32_16x16x128Bf8Fp8 = 0x82;
constexpr uint16_t kOpVWmmaF32_16x16x128Bf8Bf8 = 0x83;
constexpr uint16_t kOpVWmmaF32_32x16x128F4 = 0x88;
constexpr uint16_t kOpVWmmaF16_16x16x128Fp8Fp8 = 0x84;
constexpr uint16_t kOpVWmmaF16_16x16x128Fp8Bf8 = 0x85;
constexpr uint16_t kOpVWmmaF16_16x16x128Bf8Fp8 = 0x86;
constexpr uint16_t kOpVWmmaF16_16x16x128Bf8Bf8 = 0x87;
constexpr uint16_t kOpVLshlrevB16Vop3 = 824;
constexpr uint16_t kOpVOrB16Vop3 = 867;
constexpr uint16_t kOpGlobalAtomicAddF64 = 0x55;
constexpr uint16_t kOpGlobalLoadTr16B128 = 87;
constexpr uint16_t kOpGlobalLoadTr8B64 = 88;
constexpr uint16_t kOpGlobalLoadTr4B64 = 115;
constexpr uint16_t kOpGlobalLoadTr6B96 = 116;
constexpr uint16_t kOpTensorLoadToLds = 196;
constexpr uint16_t kOpTensorStoreFromLds = 197;
constexpr uint16_t kOpDsStore2AddrB64 = 0x4E;
constexpr uint16_t kOpDsLoadTr4B64 = 0xFA;
constexpr uint16_t kOpDsLoadTr6B96 = 0xFB;
constexpr uint16_t kOpDsLoadTr16B128 = 0xFC;
constexpr uint16_t kOpDsLoadTr8B64 = 0xFD;

#define RJ_GFX1250_EXPAND(FN) legacy_expand_adapter<FN>

#define RJ_VOPD3_RULE(OP_PAIR)                                                                     \
  {                                                                                                \
    kEncVopd3, OP_PAIR, RuleAction::Expand, 0, 0, nullptr, RJ_GFX1250_EXPAND(expand_vopd3),        \
        nullptr, nullptr                                                                           \
  }

#define RJ_VOPD_RULE(OP_PAIR)                                                                      \
  {                                                                                                \
    kEncVopd, OP_PAIR, RuleAction::Expand, 0, 0, nullptr, RJ_GFX1250_EXPAND(expand_vopd_xy),       \
        nullptr, nullptr                                                                           \
  }

#define RJ_VOP3_SINGLE_SRC_RULE(ENC)                                                               \
  {                                                                                                \
    ENC, kAnyTranslationOpcode, RuleAction::Expand, 0, 0, nullptr,                                 \
        RJ_GFX1250_EXPAND(lower_gfx1250_vop3_single_src_to_rdna4), nullptr, nullptr                \
  }

#define RJ_VOP3_EXPAND_RULE(ENC, OP, FN)                                                           \
  { ENC, OP, RuleAction::Expand, 0, 0, nullptr, RJ_GFX1250_EXPAND(FN), nullptr, nullptr }

#define RJ_VOP3_SCALED_CVT_RULE(ENC, OP)                                                           \
  { ENC, OP, RuleAction::Expand, 0, 0, nullptr, expand_v_cvt_scaled_lowp_vop3, nullptr, nullptr }

#define RJ_VOP3_SCALED_CVT_RULES(ENC)                                                              \
  RJ_VOP3_SCALED_CVT_RULE(ENC, 0x297), RJ_VOP3_SCALED_CVT_RULE(ENC, 0x298),                        \
      RJ_VOP3_SCALED_CVT_RULE(ENC, 0x299), RJ_VOP3_SCALED_CVT_RULE(ENC, 0x29F),                    \
      RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2A0), RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2A1),                    \
      RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2A8), RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2A9),                    \
      RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2AA), RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2AB),                    \
      RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2AC), RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2AD),                    \
      RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2B0), RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2B3),                    \
      RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2B4), RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2B5),                    \
      RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2B8), RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2B9),                    \
      RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2BC), RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2BF),                    \
      RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2C0), RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2C1),                    \
      RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2C2), RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2C3),                    \
      RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2C4), RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2C5),                    \
      RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2C6), RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2C7),                    \
      RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2C8), RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2C9),                    \
      RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2CA), RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2CB),                    \
      RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2CC), RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2CD),                    \
      RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2CE), RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2CF),                    \
      RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2D0), RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2D1),                    \
      RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2D2), RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2D3),                    \
      RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2D4), RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2D5),                    \
      RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2D6), RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2D7),                    \
      RJ_VOP3_SCALED_CVT_RULE(ENC, 0x2D8)

#define RJ_VOP3_PERMLANE_FAMILY_RULES(ENC)                                                         \
  {ENC,                                                                                            \
   kOpVPermlaneBcastB32Vop3,                                                                       \
   RuleAction::Expand,                                                                             \
   0,                                                                                              \
   0,                                                                                              \
   nullptr,                                                                                        \
   expand_v_permlane_family_b32,                                                                   \
   nullptr,                                                                                        \
   nullptr},                                                                                       \
      {ENC,                                                                                        \
       kOpVPermlaneUpB32Vop3,                                                                      \
       RuleAction::Expand,                                                                         \
       0,                                                                                          \
       0,                                                                                          \
       nullptr,                                                                                    \
       expand_v_permlane_family_b32,                                                               \
       nullptr,                                                                                    \
       nullptr},                                                                                   \
      {ENC,                                                                                        \
       kOpVPermlaneDownB32Vop3,                                                                    \
       RuleAction::Expand,                                                                         \
       0,                                                                                          \
       0,                                                                                          \
       nullptr,                                                                                    \
       expand_v_permlane_family_b32,                                                               \
       nullptr,                                                                                    \
       nullptr},                                                                                   \
  {                                                                                                \
    ENC, kOpVPermlaneXorB32Vop3, RuleAction::Expand, 0, 0, nullptr, expand_v_permlane_family_b32,  \
        nullptr, nullptr                                                                           \
  }

#define RJ_VOP3_CVT_PK_BF16_F32_RULE(ENC)                                                          \
  {                                                                                                \
    ENC, kOpVCvtPkBf16F32Vop3, RuleAction::Expand, 0, 0, nullptr,                                  \
        RJ_GFX1250_EXPAND(expand_v_cvt_pk_bf16_f32_vop3), nullptr, nullptr                         \
  }

#define RJ_VOP3_CVT_F32_BF16_RULE(ENC)                                                             \
  {                                                                                                \
    ENC, kOpVCvtF32Bf16Vop3, RuleAction::Expand, 0, 0, nullptr,                                    \
        RJ_GFX1250_EXPAND(expand_v_cvt_f32_bf16_vop3), nullptr, nullptr                            \
  }

#define RJ_VOP3_CVT_F32_FP8_RULE(ENC)                                                              \
  {                                                                                                \
    ENC, kOpVCvtF32Fp8Vop3, RuleAction::Expand, 0, 0, nullptr,                                     \
        RJ_GFX1250_EXPAND(expand_v_cvt_f32_fp8_e5m3_vop3), nullptr, nullptr                        \
  }

#define RJ_VOP3_CVT_F16_FP8_RULE(ENC)                                                              \
  {                                                                                                \
    ENC, kOpVCvtF16Fp8Vop3, RuleAction::Expand, 0, 0, nullptr,                                     \
        RJ_GFX1250_EXPAND(expand_v_cvt_f16_fp8_vop3), nullptr, nullptr                             \
  }

#define RJ_VOP3_CVT_F16_BF8_RULE(ENC)                                                              \
  {                                                                                                \
    ENC, kOpVCvtF16Bf8Vop3, RuleAction::Expand, 0, 0, nullptr,                                     \
        RJ_GFX1250_EXPAND(expand_v_cvt_f16_bf8_vop3), nullptr, nullptr                             \
  }

#define RJ_VOP3_CVT_F16_F8_DECODE_RULES(ENC)                                                       \
  RJ_VOP3_EXPAND_RULE(ENC, kOpVCvtPkF16Fp8Vop3, expand_v_cvt_pk_f16_fp8_vop3),                     \
      RJ_VOP3_EXPAND_RULE(ENC, kOpVCvtPkF16Bf8Vop3, expand_v_cvt_pk_f16_bf8_vop3),                 \
      RJ_VOP3_CVT_F16_FP8_RULE(ENC), RJ_VOP3_CVT_F16_BF8_RULE(ENC)

#define RJ_VOP3_CVT_F16_F8_ENCODE_RULES(ENC)                                                       \
  RJ_VOP3_EXPAND_RULE(ENC, kOpVCvtPkFp8F16Vop3, expand_v_cvt_pk_fp8_f16_vop3),                     \
      RJ_VOP3_EXPAND_RULE(ENC, kOpVCvtPkBf8F16Vop3, expand_v_cvt_pk_bf8_f16_vop3),                 \
      RJ_VOP3_EXPAND_RULE(ENC, kOpVCvtSrFp8F16Vop3, expand_v_cvt_sr_fp8_f16_vop3),                 \
      RJ_VOP3_EXPAND_RULE(ENC, kOpVCvtSrBf8F16Vop3, expand_v_cvt_sr_bf8_f16_vop3)

#define RJ_VOP3_CVT_PK_F16_F32_RULE(ENC)                                                           \
  {                                                                                                \
    ENC, kOpVCvtPkF16F32Vop3, RuleAction::Expand, 0, 0, nullptr,                                   \
        RJ_GFX1250_EXPAND(expand_v_cvt_pk_f16_f32_vop3), nullptr, nullptr                          \
  }

const TranslationRule kExpandRules_gfx1250_to_rdna4[] = {
    RJ_VOPD_RULE(0x0303),
    RJ_VOPD_RULE(0x0400),
    RJ_VOPD_RULE(0x0404),
    RJ_VOPD_RULE(0x0809),
    RJ_VOPD_RULE(0x0811),
    RJ_VOPD_RULE(0x0A09),
    RJ_VOPD_RULE(0x0A0A),
    RJ_VOPD_RULE(kAnyTranslationOpcode),
    {kEncVop2MinNumF64, kOpVMinNumF64E32, RuleAction::Expand, 0, 0, nullptr,
     lower_vop2_f64_literal64, nullptr, nullptr},
    {kEncVop2MaxNumF64, kOpVMaxNumF64E32, RuleAction::Expand, 0, 0, nullptr,
     lower_vop2_f64_literal64, nullptr, nullptr},
    {kEncVop2LshlrevB64, kOpVLshlrevB64E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshlrev_b64_e32), nullptr, nullptr},
    {kEncVop2AddNcU64_0, kOpVAddNcU64E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_add_nc_u64_e32), nullptr, nullptr},
    {kEncVop2AddNcU64_1, kOpVAddNcU64E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_add_nc_u64_e32), nullptr, nullptr},
    {kEncVop2AddNcU64_2, kOpVAddNcU64E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_add_nc_u64_e32), nullptr, nullptr},
    {kEncVop2AddNcU64_3, kOpVAddNcU64E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_add_nc_u64_e32), nullptr, nullptr},
    {kEncVop2_0, kOpVSubNcU64E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_sub_nc_u64_e32), nullptr, nullptr},
    {kEncVop2_1, kOpVSubNcU64E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_sub_nc_u64_e32), nullptr, nullptr},
    {kEncVop2_2, kOpVSubNcU64E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_sub_nc_u64_e32), nullptr, nullptr},
    {kEncVop2_3, kOpVSubNcU64E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_sub_nc_u64_e32), nullptr, nullptr},
    {kEncVop2MulU64_0, kOpVMulU64E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mul_u64_e32), nullptr, nullptr},
    {kEncVop2MulU64_1, kOpVMulU64E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mul_u64_e32), nullptr, nullptr},
    {kEncVop2MulU64_2, kOpVMulU64E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mul_u64_e32), nullptr, nullptr},
    {kEncVop2MulU64_3, kOpVMulU64E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mul_u64_e32), nullptr, nullptr},
    {kEncVop2AddF16_0, kOpVAddF16E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_add_f16_e32), nullptr, nullptr},
    {kEncVop2AddF16_1, kOpVAddF16E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_add_f16_e32), nullptr, nullptr},
    {kEncVop2AddF16_2, kOpVAddF16E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_add_f16_e32), nullptr, nullptr},
    {kEncVop2AddF16_3, kOpVAddF16E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_add_f16_e32), nullptr, nullptr},
    RJ_VOPD3_RULE(0x0013),
    RJ_VOPD3_RULE(0x0313),
    RJ_VOPD3_RULE(0x0408),
    RJ_VOPD3_RULE(0x0808),
    RJ_VOPD3_RULE(0x0809),
    RJ_VOPD3_RULE(0x0810),
    RJ_VOPD3_RULE(0x0811),
    RJ_VOPD3_RULE(0x0812),
    RJ_VOPD3_RULE(0x0908),
    RJ_VOPD3_RULE(0x0909),
    RJ_VOPD3_RULE(0x0910),
    RJ_VOPD3_RULE(0x0912),
    RJ_VOPD3_RULE(0x0A0A),
    RJ_VOPD3_RULE(0x0A12),
    RJ_VOPD3_RULE(0x1008),
    RJ_VOPD3_RULE(0x1009),
    RJ_VOPD3_RULE(0x1010),
    RJ_VOPD3_RULE(0x1011),
    RJ_VOPD3_RULE(0x1012),
    RJ_VOPD3_RULE(0x1014),
    RJ_VOPD3_RULE(0x1015),
    RJ_VOPD3_RULE(0x1108),
    RJ_VOPD3_RULE(0x1110),
    RJ_VOPD3_RULE(0x1111),
    RJ_VOPD3_RULE(0x1112),
    RJ_VOPD3_RULE(0x1115),
    RJ_VOPD3_RULE(0x1116),
    RJ_VOPD3_RULE(0x1303),
    RJ_VOPD3_RULE(0x1410),
    RJ_VOPD3_RULE(0x1411),
    RJ_VOPD3_RULE(0x1415),
    RJ_VOPD3_RULE(0x1510),
    RJ_VOPD3_RULE(0x1511),
    RJ_VOPD3_RULE(0x1512),
    RJ_VOPD3_RULE(0x1515),
    RJ_VOPD3_RULE(0x1516),
    RJ_VOPD3_RULE(0x1608),
    RJ_VOPD3_RULE(0x1612),
    RJ_VOPD3_RULE(0x1615),
    RJ_VOPD3_RULE(0x1616),
    RJ_VOPD3_RULE(kAnyTranslationOpcode),
    {kEncVop1_0, kOpVCvtF32F16E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_cvt_f32_f16_e32_high_src), nullptr, nullptr},
    {kEncVop1_0, kOpVMovB16Vop1, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mov_b16), nullptr, nullptr},
    {kEncVop1_0, kOpVMovB64Vop1, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mov_b64), nullptr, nullptr},
    {kEncVop1_0, kOpVCvtF32Bf16E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_cvt_f32_bf16_vop1), nullptr, nullptr},
    {kEncVop1_0, kOpVCvtPkF16Fp8E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_cvt_pk_f16_fp8_vop1), nullptr, nullptr},
    {kEncVop1_0, kOpVCvtPkF16Bf8E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_cvt_pk_f16_bf8_vop1), nullptr, nullptr},
    {kEncVop1_0, kOpVCvtF16Fp8E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_cvt_f16_fp8_vop1), nullptr, nullptr},
    {kEncVop1_0, kOpVCvtF16Bf8E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_cvt_f16_bf8_vop1), nullptr, nullptr},
    {kEncVop1_1, kOpVCvtF32F16E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_cvt_f32_f16_e32_high_src), nullptr, nullptr},
    {kEncVop1_1, kOpVMovB16Vop1, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mov_b16), nullptr, nullptr},
    {kEncVop1_1, kOpVMovB64Vop1, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mov_b64), nullptr, nullptr},
    {kEncVop1_1, kOpVCvtF32Bf16E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_cvt_f32_bf16_vop1), nullptr, nullptr},
    {kEncVop1_1, kOpVCvtPkF16Fp8E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_cvt_pk_f16_fp8_vop1), nullptr, nullptr},
    {kEncVop1_1, kOpVCvtPkF16Bf8E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_cvt_pk_f16_bf8_vop1), nullptr, nullptr},
    {kEncVop1_1, kOpVCvtF16Fp8E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_cvt_f16_fp8_vop1), nullptr, nullptr},
    {kEncVop1_1, kOpVCvtF16Bf8E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_cvt_f16_bf8_vop1), nullptr, nullptr},
    {kEncVop1_2, kOpVCvtF32F16E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_cvt_f32_f16_e32_high_src), nullptr, nullptr},
    {kEncVop1_2, kOpVMovB16Vop1, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mov_b16), nullptr, nullptr},
    {kEncVop1_2, kOpVMovB64Vop1, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mov_b64), nullptr, nullptr},
    {kEncVop1_2, kOpVCvtF32Bf16E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_cvt_f32_bf16_vop1), nullptr, nullptr},
    {kEncVop1_2, kOpVCvtPkF16Fp8E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_cvt_pk_f16_fp8_vop1), nullptr, nullptr},
    {kEncVop1_2, kOpVCvtPkF16Bf8E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_cvt_pk_f16_bf8_vop1), nullptr, nullptr},
    {kEncVop1_2, kOpVCvtF16Fp8E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_cvt_f16_fp8_vop1), nullptr, nullptr},
    {kEncVop1_2, kOpVCvtF16Bf8E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_cvt_f16_bf8_vop1), nullptr, nullptr},
    {kEncVop1_3, kOpVCvtF32F16E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_cvt_f32_f16_e32_high_src), nullptr, nullptr},
    {kEncVop1_3, kOpVMovB16Vop1, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mov_b16), nullptr, nullptr},
    {kEncVop1_3, kOpVMovB64Vop1, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mov_b64), nullptr, nullptr},
    {kEncVop1_3, kOpVCvtF32Bf16E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_cvt_f32_bf16_vop1), nullptr, nullptr},
    {kEncVop1_3, kOpVCvtPkF16Fp8E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_cvt_pk_f16_fp8_vop1), nullptr, nullptr},
    {kEncVop1_3, kOpVCvtPkF16Bf8E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_cvt_pk_f16_bf8_vop1), nullptr, nullptr},
    {kEncVop1_3, kOpVCvtF16Fp8E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_cvt_f16_fp8_vop1), nullptr, nullptr},
    {kEncVop1_3, kOpVCvtF16Bf8E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_cvt_f16_bf8_vop1), nullptr, nullptr},
    {kEncSop2SAndB64, kOpSAndB64, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(lower_s_and_b64_literal64), nullptr, nullptr},
    {kEncSop2SOrB64, kOpSOrB64, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(lower_s_or_b64_literal64), nullptr, nullptr},
    {kEncSop2SAddNcU64, kOpSAddNcU64, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(lower_s_add_nc_u64_to_carry_chain), nullptr, nullptr},
    {kEncSop2SSubNcU64, kOpSSubNcU64, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(lower_s_sub_nc_u64_to_borrow_chain), nullptr, nullptr},
    {kEncSop2SMulU64, 85, RuleAction::Expand, 0, 0, nullptr, lower_s_mul_u64_literal64, nullptr,
     nullptr},
    {kEncSopkMovk, kOpSMovkI32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(lower_gfx1250_resource_word2_movk), nullptr, nullptr},
    {kEncSopkGetreg, kOpSGetregB32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(lower_gfx1250_grid_mode_s_getreg_to_zero), nullptr, nullptr},
    {kEncSopkSetregImm32, kOpSSetregImm32B32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(preserve_gfx1250_replay_mode_s_setreg_imm32), nullptr, nullptr},
    {kEncSop1, kOpSMovB32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(lower_gfx1250_resource_s_mov_b32), nullptr, nullptr},
    {kEncSop1, kOpSMovB64, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(lower_s_mov_b64_literal64), nullptr, nullptr},
    {kEncSop1, kOpSGetPcI64, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(preserve_same_sop1_encoding), nullptr, nullptr},
    {kEncSop1, kOpSSetPcI64, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(preserve_same_sop1_encoding), nullptr, nullptr},
    {kEncSop1, kOpSSwapPcI64, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(preserve_same_sop1_encoding), nullptr, nullptr},
    {kEncSop1, kOpSSendmsgRtnB32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(lower_s_sendmsg_rtn_to_rdna4), nullptr, nullptr},
    {kEncSop1, kOpSSendmsgRtnB64, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(lower_s_sendmsg_rtn_to_rdna4), nullptr, nullptr},
    {kEncSopc, kOpSCmpLgU64, RuleAction::Expand, 0, 0, nullptr, lower_s_cmp_u64_literal64, nullptr,
     nullptr},
    {kEncSopp, kOpSClause, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(lower_s_clause_to_nop), nullptr, nullptr},
    {kEncSopp, kOpSSetVgprMsb, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(lower_s_set_vgpr_msb_to_setreg), nullptr, nullptr},
    {kEncSopp, kOpSWaitKmcnt, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_s_wait_kmcnt), nullptr, nullptr},
    {kEncSopp, kOpSWaitTensorcnt, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(lower_s_clause_to_nop), nullptr, nullptr},
    {kEncVop3p, kOpVPkFmaBf16Vop3p, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_pk_fma_bf16_vop3p), nullptr, nullptr},
    {kEncVop3p, kOpVPkFmaF32Vop3p, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_pk_fma_f32_vop3p), nullptr, nullptr},
    {kEncVop3p, kOpVFmaMixF32Vop3p, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_fma_mix_f32_f16_vop3p), nullptr, nullptr},
    {kEncVop3p, kOpVPkAddBf16Vop3p, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_pk_add_bf16_vop3p), nullptr, nullptr},
    {kEncVop3p, kOpVPkMulF32Vop3p, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_pk_mul_f32_vop3p), nullptr, nullptr},
    {kEncVop3p, kOpVPkAddF32Vop3p, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_pk_add_f32_vop3p), nullptr, nullptr},
    {kEncVop3p, kOpVWmmaF32_16x16x128F8f6f4, RuleAction::Expand, 0, 0, nullptr,
     expand_v_wmma_f32_16x16x128_f8f6f4, nullptr, nullptr},
    {kEncVop3p, kOpVWmmaScaleF32_16x16x128F8f6f4, RuleAction::Expand, 0, 0, nullptr,
     expand_v_wmma_scale_f32_16x16x128_f8f6f4, nullptr, nullptr},
    {kEncVop3p, kOpVWmmaScale16F32_16x16x128F8f6f4, RuleAction::Expand, 0, 0, nullptr,
     expand_v_wmma_scale_f32_16x16x128_f8f6f4, nullptr, nullptr},
    {kEncVop3p, kOpVFmaMixF32Bf16Vop3p, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_fma_mix_f32_bf16_vop3p), nullptr, nullptr},
    {kEncVop3p, kOpVFmaMixloBf16Vop3p, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_fma_mixlo_bf16_vop3p), nullptr, nullptr},
    {kEncVop3p, kOpVWmmaF32_16x16x4F32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_wmma_f32_16x16x4_f32), nullptr, nullptr},
    {kEncVop3p, kOpVWmmaF32_16x16x32F16, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_wmma_f32_16x16x32_f16), nullptr, nullptr},
    {kEncVop3p, kOpVWmmaF16_16x16x32F16, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_wmma_f16_16x16x32_f16), nullptr, nullptr},
    {kEncVop3p, kOpVWmmaF32_16x16x32Bf16, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_wmma_f32_16x16x32_bf16), nullptr, nullptr},
    {kEncVop3p, kOpVWmmaBf16_16x16x32Bf16, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_wmma_bf16_16x16x32_bf16), nullptr, nullptr},
    {kEncVop3p, kOpVWmmaBf16f32_16x16x32Bf16, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_wmma_bf16f32_16x16x32_bf16), nullptr, nullptr},
    {kEncVop3p, kOpVSwmmacF32_16x16x64F16, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_swmmac_f32_16x16x64_f16), nullptr, nullptr},
    {kEncVop3p, kOpVSwmmacF32_16x16x64Bf16, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_swmmac_f32_16x16x64_f16), nullptr, nullptr},
    {kEncVop3p, kOpVSwmmacF16_16x16x64F16, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_swmmac_f32_16x16x64_f16), nullptr, nullptr},
    {kEncVop3p, kOpVSwmmacBf16_16x16x64Bf16, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_swmmac_f32_16x16x64_f16), nullptr, nullptr},
    {kEncVop3p, kOpVSwmmacBf16f32_16x16x64Bf16, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_swmmac_f32_16x16x64_f16), nullptr, nullptr},
    {kEncVop3p, kOpVWmmaF32_16x16x64Fp8Fp8, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_wmma_f32_16x16x64_fp8_fp8), nullptr, nullptr},
    {kEncVop3p, kOpVWmmaF32_16x16x64Fp8Bf8, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_wmma_f32_16x16x64_fp8_fp8), nullptr, nullptr},
    {kEncVop3p, kOpVWmmaF32_16x16x64Bf8Fp8, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_wmma_f32_16x16x64_fp8_fp8), nullptr, nullptr},
    {kEncVop3p, kOpVWmmaF32_16x16x64Bf8Bf8, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_wmma_f32_16x16x64_fp8_fp8), nullptr, nullptr},
    {kEncVop3p, kOpVWmmaF16_16x16x64Fp8Fp8, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_wmma_f16_16x16x64_f8), nullptr, nullptr},
    {kEncVop3p, kOpVWmmaF16_16x16x64Fp8Bf8, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_wmma_f16_16x16x64_f8), nullptr, nullptr},
    {kEncVop3p, kOpVWmmaF16_16x16x64Bf8Fp8, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_wmma_f16_16x16x64_f8), nullptr, nullptr},
    {kEncVop3p, kOpVWmmaF16_16x16x64Bf8Bf8, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_wmma_f16_16x16x64_f8), nullptr, nullptr},
    {kEncVop3p, kOpVWmmaI32_16x16x64Iu8, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_wmma_i32_16x16x64_iu8), nullptr, nullptr},
    {kEncVop3p, kOpVSwmmacF32_16x16x128Fp8Fp8, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_swmmac_16x16x128_f8), nullptr, nullptr},
    {kEncVop3p, kOpVSwmmacF32_16x16x128Fp8Bf8, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_swmmac_16x16x128_f8), nullptr, nullptr},
    {kEncVop3p, kOpVSwmmacF32_16x16x128Bf8Fp8, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_swmmac_16x16x128_f8), nullptr, nullptr},
    {kEncVop3p, kOpVSwmmacF32_16x16x128Bf8Bf8, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_swmmac_16x16x128_f8), nullptr, nullptr},
    {kEncVop3p, kOpVSwmmacF16_16x16x128Fp8Fp8, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_swmmac_16x16x128_f8), nullptr, nullptr},
    {kEncVop3p, kOpVSwmmacF16_16x16x128Fp8Bf8, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_swmmac_16x16x128_f8), nullptr, nullptr},
    {kEncVop3p, kOpVSwmmacF16_16x16x128Bf8Fp8, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_swmmac_16x16x128_f8), nullptr, nullptr},
    {kEncVop3p, kOpVSwmmacF16_16x16x128Bf8Bf8, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_swmmac_16x16x128_f8), nullptr, nullptr},
    {kEncVop3p, kOpVSwmmacI32_16x16x128Iu8, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_swmmac_i32_16x16x128_iu8), nullptr, nullptr},
    {kEncVop3p1, kOpVWmmaF32_16x16x128Fp8Fp8, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_wmma_f32_16x16x128_fp8_fp8), nullptr, nullptr},
    {kEncVop3p1, kOpVWmmaF32_16x16x128Fp8Bf8, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_wmma_f32_16x16x128_fp8_fp8), nullptr, nullptr},
    {kEncVop3p1, kOpVWmmaF32_16x16x128Bf8Fp8, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_wmma_f32_16x16x128_fp8_fp8), nullptr, nullptr},
    {kEncVop3p1, kOpVWmmaF32_16x16x128Bf8Bf8, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_wmma_f32_16x16x128_fp8_fp8), nullptr, nullptr},
    {kEncVop3p1, kOpVWmmaF16_16x16x128Fp8Fp8, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_wmma_f16_16x16x128_f8), nullptr, nullptr},
    {kEncVop3p1, kOpVWmmaF16_16x16x128Fp8Bf8, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_wmma_f16_16x16x128_f8), nullptr, nullptr},
    {kEncVop3p1, kOpVWmmaF16_16x16x128Bf8Fp8, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_wmma_f16_16x16x128_f8), nullptr, nullptr},
    {kEncVop3p1, kOpVWmmaF16_16x16x128Bf8Bf8, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_wmma_f16_16x16x128_f8), nullptr, nullptr},
    {kEncVop3p1, kOpVWmmaF32_32x16x128F4, RuleAction::Expand, 0, 0, nullptr,
     expand_v_wmma_f32_32x16x128_f4, nullptr, nullptr},
    {kEncVimage, kOpTensorLoadToLds, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_tensor_dma_vimage), nullptr, nullptr},
    {kEncVimage, kOpTensorStoreFromLds, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_tensor_dma_vimage), nullptr, nullptr},
    {kEncVop3_0, kOpVMulU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mul_u64_vop3), nullptr, nullptr},
    {kEncVop3_0, kOpVAddNcU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_add_nc_u64_vop3), nullptr, nullptr},
    {kEncVop3_0, kOpVSubNcU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_sub_nc_u64_vop3), nullptr, nullptr},
    {kEncVop3_0, kOpVMovB64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mov_b64_vop3), nullptr, nullptr},
    RJ_VOP3_CVT_F32_FP8_RULE(kEncVop3_0),
    RJ_VOP3_CVT_F32_BF16_RULE(kEncVop3_0),
    RJ_VOP3_CVT_F16_F8_DECODE_RULES(kEncVop3_0),
    {kEncVop3_0, kOpVMadU32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_u32_vop3), nullptr, nullptr},
    {kEncVop3_0, kOpVLshlAddU32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_add_u32_vop3), nullptr, nullptr},
    {kEncVop3_0, kOpVLshlAddU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_add_u64_vop3), nullptr, nullptr},
    {kEncVop3_0, kOpVLshlOrB32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_or_b32_vop3), nullptr, nullptr},
    {kEncVop3_0, kOpVMadNcU64U32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_nc_64_32_vop3), nullptr, nullptr},
    {kEncVop3_0, kOpVMadNcI64I32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_nc_64_32_vop3), nullptr, nullptr},
    {kEncVop3_0, kOpVMinU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_min_u64_vop3), nullptr, nullptr},
    {kEncVop3_0, kOpVMaxU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_max_u64_vop3), nullptr, nullptr},
    {kEncVop3_0, kOpVMinI64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_min_i64_vop3), nullptr, nullptr},
    {kEncVop3_0, kOpVMaxI64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_max_i64_vop3), nullptr, nullptr},
    RJ_VOP3_CVT_PK_BF16_F32_RULE(kEncVop3_0),
    RJ_VOP3_CVT_PK_F16_F32_RULE(kEncVop3_0),
    RJ_VOP3_CVT_F16_F8_ENCODE_RULES(kEncVop3_0),
    RJ_VOP3_SINGLE_SRC_RULE(kEncVop3_0),
    {kEncVop3_1, kOpVMulU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mul_u64_vop3), nullptr, nullptr},
    {kEncVop3_1, kOpVAddNcU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_add_nc_u64_vop3), nullptr, nullptr},
    {kEncVop3_1, kOpVSubNcU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_sub_nc_u64_vop3), nullptr, nullptr},
    {kEncVop3_1, kOpVMovB64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mov_b64_vop3), nullptr, nullptr},
    RJ_VOP3_CVT_F32_FP8_RULE(kEncVop3_1),
    RJ_VOP3_CVT_F32_BF16_RULE(kEncVop3_1),
    RJ_VOP3_CVT_F16_F8_DECODE_RULES(kEncVop3_1),
    {kEncVop3_1, kOpVMadU32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_u32_vop3), nullptr, nullptr},
    {kEncVop3_1, kOpVLshlAddU32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_add_u32_vop3), nullptr, nullptr},
    {kEncVop3_1, kOpVLshlAddU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_add_u64_vop3), nullptr, nullptr},
    {kEncVop3_1, kOpVLshlOrB32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_or_b32_vop3), nullptr, nullptr},
    {kEncVop3_1, kOpVMadNcU64U32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_nc_64_32_vop3), nullptr, nullptr},
    {kEncVop3_1, kOpVMadNcI64I32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_nc_64_32_vop3), nullptr, nullptr},
    {kEncVop3_1, kOpVMinU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_min_u64_vop3), nullptr, nullptr},
    {kEncVop3_1, kOpVMaxU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_max_u64_vop3), nullptr, nullptr},
    {kEncVop3_1, kOpVMinI64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_min_i64_vop3), nullptr, nullptr},
    {kEncVop3_1, kOpVMaxI64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_max_i64_vop3), nullptr, nullptr},
    RJ_VOP3_CVT_PK_BF16_F32_RULE(kEncVop3_1),
    RJ_VOP3_CVT_PK_F16_F32_RULE(kEncVop3_1),
    RJ_VOP3_CVT_F16_F8_ENCODE_RULES(kEncVop3_1),
    RJ_VOP3_SINGLE_SRC_RULE(kEncVop3_1),
    {kEncVop3_2, kOpVMulU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mul_u64_vop3), nullptr, nullptr},
    {kEncVop3_2, kOpVAddNcU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_add_nc_u64_vop3), nullptr, nullptr},
    {kEncVop3_2, kOpVSubNcU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_sub_nc_u64_vop3), nullptr, nullptr},
    {kEncVop3_2, kOpVMovB64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mov_b64_vop3), nullptr, nullptr},
    RJ_VOP3_CVT_F32_FP8_RULE(kEncVop3_2),
    RJ_VOP3_CVT_F32_BF16_RULE(kEncVop3_2),
    RJ_VOP3_CVT_F16_F8_DECODE_RULES(kEncVop3_2),
    {kEncVop3_2, kOpVMadU32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_u32_vop3), nullptr, nullptr},
    {kEncVop3_2, kOpVLshlAddU32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_add_u32_vop3), nullptr, nullptr},
    {kEncVop3_2, kOpVLshlAddU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_add_u64_vop3), nullptr, nullptr},
    {kEncVop3_2, kOpVLshlOrB32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_or_b32_vop3), nullptr, nullptr},
    {kEncVop3_2, kOpVMadNcU64U32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_nc_64_32_vop3), nullptr, nullptr},
    {kEncVop3_2, kOpVMadNcI64I32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_nc_64_32_vop3), nullptr, nullptr},
    {kEncVop3_2, kOpVMinU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_min_u64_vop3), nullptr, nullptr},
    {kEncVop3_2, kOpVMaxU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_max_u64_vop3), nullptr, nullptr},
    {kEncVop3_2, kOpVMinI64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_min_i64_vop3), nullptr, nullptr},
    {kEncVop3_2, kOpVMaxI64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_max_i64_vop3), nullptr, nullptr},
    RJ_VOP3_CVT_PK_BF16_F32_RULE(kEncVop3_2),
    RJ_VOP3_CVT_PK_F16_F32_RULE(kEncVop3_2),
    RJ_VOP3_CVT_F16_F8_ENCODE_RULES(kEncVop3_2),
    RJ_VOP3_SINGLE_SRC_RULE(kEncVop3_2),
    {kEncVop3_3, kOpVMulU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mul_u64_vop3), nullptr, nullptr},
    {kEncVop3_3, kOpVAddNcU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_add_nc_u64_vop3), nullptr, nullptr},
    {kEncVop3_3, kOpVSubNcU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_sub_nc_u64_vop3), nullptr, nullptr},
    {kEncVop3_3, kOpVMovB64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mov_b64_vop3), nullptr, nullptr},
    RJ_VOP3_CVT_F32_FP8_RULE(kEncVop3_3),
    RJ_VOP3_CVT_F32_BF16_RULE(kEncVop3_3),
    RJ_VOP3_CVT_F16_F8_DECODE_RULES(kEncVop3_3),
    {kEncVop3_3, kOpVMadU32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_u32_vop3), nullptr, nullptr},
    {kEncVop3_3, kOpVLshlAddU32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_add_u32_vop3), nullptr, nullptr},
    {kEncVop3_3, kOpVLshlAddU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_add_u64_vop3), nullptr, nullptr},
    {kEncVop3_3, kOpVLshlOrB32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_or_b32_vop3), nullptr, nullptr},
    {kEncVop3_3, kOpVMadNcU64U32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_nc_64_32_vop3), nullptr, nullptr},
    {kEncVop3_3, kOpVMadNcI64I32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_nc_64_32_vop3), nullptr, nullptr},
    {kEncVop3_3, kOpVMinU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_min_u64_vop3), nullptr, nullptr},
    {kEncVop3_3, kOpVMaxU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_max_u64_vop3), nullptr, nullptr},
    {kEncVop3_3, kOpVMinI64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_min_i64_vop3), nullptr, nullptr},
    {kEncVop3_3, kOpVMaxI64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_max_i64_vop3), nullptr, nullptr},
    RJ_VOP3_CVT_PK_BF16_F32_RULE(kEncVop3_3),
    RJ_VOP3_CVT_PK_F16_F32_RULE(kEncVop3_3),
    RJ_VOP3_CVT_F16_F8_ENCODE_RULES(kEncVop3_3),
    RJ_VOP3_SINGLE_SRC_RULE(kEncVop3_3),
    {kEncVop3_4, kOpVMulU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mul_u64_vop3), nullptr, nullptr},
    {kEncVop3_4, kOpVAddNcU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_add_nc_u64_vop3), nullptr, nullptr},
    {kEncVop3_4, kOpVSubNcU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_sub_nc_u64_vop3), nullptr, nullptr},
    {kEncVop3_4, kOpVMovB64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mov_b64_vop3), nullptr, nullptr},
    RJ_VOP3_CVT_F32_FP8_RULE(kEncVop3_4),
    RJ_VOP3_CVT_F32_BF16_RULE(kEncVop3_4),
    RJ_VOP3_CVT_F16_F8_DECODE_RULES(kEncVop3_4),
    {kEncVop3_4, kOpVBitop3B16Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_bitop3_b16_vop3), nullptr, nullptr},
    {kEncVop3_4, kOpVBitop3B32Vop3, RuleAction::Expand, 0, 0, nullptr, expand_v_bitop3_b32_vop3,
     nullptr, nullptr},
    {kEncVop3_4, kOpVMadU32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_u32_vop3), nullptr, nullptr},
    {kEncVop3_4, kOpVLshlAddU32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_add_u32_vop3), nullptr, nullptr},
    {kEncVop3_4, kOpVLshlAddU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_add_u64_vop3), nullptr, nullptr},
    {kEncVop3_4, kOpVLshlOrB32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_or_b32_vop3), nullptr, nullptr},
    RJ_VOP3_PERMLANE_FAMILY_RULES(kEncVop3_4),
    {kEncVop3_4, kOpVMadNcU64U32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_nc_64_32_vop3), nullptr, nullptr},
    {kEncVop3_4, kOpVMadNcI64I32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_nc_64_32_vop3), nullptr, nullptr},
    {kEncVop3_4, kOpVMinU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_min_u64_vop3), nullptr, nullptr},
    {kEncVop3_4, kOpVMaxU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_max_u64_vop3), nullptr, nullptr},
    {kEncVop3_4, kOpVMinI64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_min_i64_vop3), nullptr, nullptr},
    {kEncVop3_4, kOpVMaxI64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_max_i64_vop3), nullptr, nullptr},
    RJ_VOP3_CVT_PK_BF16_F32_RULE(kEncVop3_4),
    RJ_VOP3_CVT_PK_F16_F32_RULE(kEncVop3_4),
    RJ_VOP3_CVT_F16_F8_ENCODE_RULES(kEncVop3_4),
    RJ_VOP3_SINGLE_SRC_RULE(kEncVop3_4),
    {kEncVop3_5, kOpVMulU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mul_u64_vop3), nullptr, nullptr},
    {kEncVop3_5, kOpVAddNcU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_add_nc_u64_vop3), nullptr, nullptr},
    {kEncVop3_5, kOpVSubNcU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_sub_nc_u64_vop3), nullptr, nullptr},
    {kEncVop3_5, kOpVMovB64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mov_b64_vop3), nullptr, nullptr},
    RJ_VOP3_CVT_F32_FP8_RULE(kEncVop3_5),
    RJ_VOP3_CVT_F32_BF16_RULE(kEncVop3_5),
    RJ_VOP3_CVT_F16_F8_DECODE_RULES(kEncVop3_5),
    {kEncVop3_5, kOpVMadU32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_u32_vop3), nullptr, nullptr},
    {kEncVop3_5, kOpVLshlAddU32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_add_u32_vop3), nullptr, nullptr},
    {kEncVop3_5, kOpVLshlAddU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_add_u64_vop3), nullptr, nullptr},
    {kEncVop3_5, kOpVLshlOrB32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_or_b32_vop3), nullptr, nullptr},
    RJ_VOP3_SCALED_CVT_RULES(kEncVop3_5),
    {kEncVop3_5, kOpVMadNcU64U32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_nc_64_32_vop3), nullptr, nullptr},
    {kEncVop3_5, kOpVMadNcI64I32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_nc_64_32_vop3), nullptr, nullptr},
    {kEncVop3_5, kOpVMinU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_min_u64_vop3), nullptr, nullptr},
    {kEncVop3_5, kOpVMaxU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_max_u64_vop3), nullptr, nullptr},
    {kEncVop3_5, kOpVMinI64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_min_i64_vop3), nullptr, nullptr},
    {kEncVop3_5, kOpVMaxI64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_max_i64_vop3), nullptr, nullptr},
    RJ_VOP3_CVT_PK_BF16_F32_RULE(kEncVop3_5),
    RJ_VOP3_CVT_PK_F16_F32_RULE(kEncVop3_5),
    RJ_VOP3_CVT_F16_F8_ENCODE_RULES(kEncVop3_5),
    RJ_VOP3_SINGLE_SRC_RULE(kEncVop3_5),
    {kEncVop3_6, kOpVMulU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mul_u64_vop3), nullptr, nullptr},
    {kEncVop3_6, kOpVAddNcU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_add_nc_u64_vop3), nullptr, nullptr},
    {kEncVop3_6, kOpVSubNcU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_sub_nc_u64_vop3), nullptr, nullptr},
    {kEncVop3_6, kOpVMovB64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mov_b64_vop3), nullptr, nullptr},
    RJ_VOP3_CVT_F32_FP8_RULE(kEncVop3_6),
    RJ_VOP3_CVT_F32_BF16_RULE(kEncVop3_6),
    RJ_VOP3_CVT_F16_F8_DECODE_RULES(kEncVop3_6),
    {kEncVop3_6, kOpVMadU32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_u32_vop3), nullptr, nullptr},
    {kEncVop3_6, kOpVLshlAddU32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_add_u32_vop3), nullptr, nullptr},
    {kEncVop3_6, kOpVLshlAddU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_add_u64_vop3), nullptr, nullptr},
    {kEncVop3_6, kOpVLshlOrB32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_or_b32_vop3), nullptr, nullptr},
    {kEncVop3_6, kOpVMadNcU64U32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_nc_64_32_vop3), nullptr, nullptr},
    {kEncVop3_6, kOpVMadNcI64I32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_nc_64_32_vop3), nullptr, nullptr},
    {kEncVop3_6, kOpVPermlaneIdxGenB32Vop3, RuleAction::Expand, 0, 0, nullptr,
     expand_v_permlane_idx_gen_b32, nullptr, nullptr},
    {kEncVop3_6, kOpVMinU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_min_u64_vop3), nullptr, nullptr},
    {kEncVop3_6, kOpVMaxU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_max_u64_vop3), nullptr, nullptr},
    {kEncVop3_6, kOpVMinI64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_min_i64_vop3), nullptr, nullptr},
    {kEncVop3_6, kOpVMaxI64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_max_i64_vop3), nullptr, nullptr},
    {kEncVop3_6, kOpVLshlrevB16Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshlrev_b16_vop3), nullptr, nullptr},
    {kEncVop3_6, kOpVOrB16Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_or_b16_vop3), nullptr, nullptr},
    {kEncVop3_6, kOpVCvtPkFp8F32Vop3, RuleAction::Expand, 0, 0, nullptr,
     expand_v_cvt_pk_fp8_f32_e5m3_vop3, nullptr, nullptr},
    {kEncVop3_6, kOpVCvtSrFp8F32Vop3, RuleAction::Expand, 0, 0, nullptr,
     expand_v_cvt_sr_fp8_f32_e5m3_vop3, nullptr, nullptr},
    RJ_VOP3_CVT_PK_BF16_F32_RULE(kEncVop3_6),
    {kEncVop3_6, kOpVCvtSrPkBf16F32Vop3, RuleAction::Expand, 0, 0, nullptr,
     expand_v_cvt_sr_pk_f16_bf16_f32_vop3, nullptr, nullptr},
    RJ_VOP3_CVT_PK_F16_F32_RULE(kEncVop3_6),
    {kEncVop3_6, kOpVCvtSrPkF16F32Vop3, RuleAction::Expand, 0, 0, nullptr,
     expand_v_cvt_sr_pk_f16_bf16_f32_vop3, nullptr, nullptr},
    RJ_VOP3_CVT_F16_F8_ENCODE_RULES(kEncVop3_6),
    RJ_VOP3_SINGLE_SRC_RULE(kEncVop3_6),
    {kEncVop3_7, kOpVMulU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mul_u64_vop3), nullptr, nullptr},
    {kEncVop3_7, kOpVAddNcU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_add_nc_u64_vop3), nullptr, nullptr},
    {kEncVop3_7, kOpVSubNcU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_sub_nc_u64_vop3), nullptr, nullptr},
    {kEncVop3_7, kOpVMovB64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mov_b64_vop3), nullptr, nullptr},
    RJ_VOP3_CVT_F32_FP8_RULE(kEncVop3_7),
    RJ_VOP3_CVT_F32_BF16_RULE(kEncVop3_7),
    RJ_VOP3_CVT_F16_F8_DECODE_RULES(kEncVop3_7),
    {kEncVop3_7, kOpVMadU32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_u32_vop3), nullptr, nullptr},
    {kEncVop3_7, kOpVLshlAddU32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_add_u32_vop3), nullptr, nullptr},
    {kEncVop3_7, kOpVLshlAddU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_add_u64_vop3), nullptr, nullptr},
    {kEncVop3_7, kOpVLshlOrB32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_or_b32_vop3), nullptr, nullptr},
    {kEncVop3_7, kOpVMadNcU64U32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_nc_64_32_vop3), nullptr, nullptr},
    {kEncVop3_7, kOpVMadNcI64I32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_nc_64_32_vop3), nullptr, nullptr},
    {kEncVop3_7, kOpVMinU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_min_u64_vop3), nullptr, nullptr},
    {kEncVop3_7, kOpVMaxU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_max_u64_vop3), nullptr, nullptr},
    {kEncVop3_7, kOpVMinI64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_min_i64_vop3), nullptr, nullptr},
    {kEncVop3_7, kOpVMaxI64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_max_i64_vop3), nullptr, nullptr},
    RJ_VOP3_CVT_PK_BF16_F32_RULE(kEncVop3_7),
    RJ_VOP3_CVT_PK_F16_F32_RULE(kEncVop3_7),
    RJ_VOP3_CVT_F16_F8_ENCODE_RULES(kEncVop3_7),
    RJ_VOP3_SINGLE_SRC_RULE(kEncVop3_7),
    {kEncVdsStore2AddrB64, kOpDsStore2AddrB64, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_ds_store_2addr_b64), nullptr, nullptr},
    {kEncVdsTranspose, kOpDsLoadTr4B64, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_ds_transpose_load), nullptr, nullptr},
    {kEncVdsTranspose, kOpDsLoadTr6B96, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_ds_transpose_load), nullptr, nullptr},
    {kEncVdsTranspose, kOpDsLoadTr16B128, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_ds_transpose_load), nullptr, nullptr},
    {kEncVdsTranspose, kOpDsLoadTr8B64, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_ds_transpose_load), nullptr, nullptr},
    {kEncVscratch, kAnyTranslationOpcode, RuleAction::Expand, 0, 0, nullptr, expand_scaled_vscratch,
     nullptr, nullptr},
    {kEncVscratch1, kAnyTranslationOpcode, RuleAction::Expand, 0, 0, nullptr,
     expand_scaled_vscratch, nullptr, nullptr},
    {kEncVglobal, kOpGlobalAtomicAddF64, RuleAction::Expand, 0, 0, nullptr,
     expand_global_atomic_add_f64, nullptr, nullptr},
    {kEncVglobal, kOpGlobalLoadTr16B128, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_native_global_transpose_load), nullptr, nullptr},
    {kEncVglobal, kOpGlobalLoadTr8B64, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_native_global_transpose_load), nullptr, nullptr},
    {kEncVglobal, kOpGlobalLoadTr4B64, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_global_load_tr4_b64), nullptr, nullptr},
    {kEncVglobal, kOpGlobalLoadTr6B96, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_global_load_tr6_b96), nullptr, nullptr},
    {kEncVglobal, kAnyTranslationOpcode, RuleAction::Expand, 0, 0, nullptr, expand_scaled_vglobal,
     nullptr, nullptr},
    {kEncVglobal1, kOpGlobalLoadTr16B128, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_native_global_transpose_load), nullptr, nullptr},
    {kEncVglobal1, kOpGlobalLoadTr8B64, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_native_global_transpose_load), nullptr, nullptr},
    {kEncVglobal1, kAnyTranslationOpcode, RuleAction::Expand, 0, 0, nullptr, expand_scaled_vglobal,
     nullptr, nullptr},
    {kEncSmem0, kAnyTranslationOpcode, RuleAction::Expand, 0, 0, nullptr, expand_scaled_smem,
     nullptr, nullptr},
    {kEncSmem1, kAnyTranslationOpcode, RuleAction::Expand, 0, 0, nullptr, expand_scaled_smem,
     nullptr, nullptr},
    {kEncSmem2, kAnyTranslationOpcode, RuleAction::Expand, 0, 0, nullptr, expand_scaled_smem,
     nullptr, nullptr},
    {kEncSmem3, kAnyTranslationOpcode, RuleAction::Expand, 0, 0, nullptr, expand_scaled_smem,
     nullptr, nullptr},
};

#undef RJ_VOPD3_RULE
#undef RJ_VOPD_RULE
#undef RJ_VOP3_SINGLE_SRC_RULE
#undef RJ_VOP3_EXPAND_RULE
#undef RJ_VOP3_PERMLANE_FAMILY_RULES
#undef RJ_VOP3_CVT_PK_BF16_F32_RULE
#undef RJ_VOP3_CVT_PK_F16_F32_RULE
#undef RJ_VOP3_CVT_F32_BF16_RULE
#undef RJ_VOP3_CVT_F32_FP8_RULE
#undef RJ_VOP3_CVT_F16_FP8_RULE
#undef RJ_VOP3_CVT_F16_BF8_RULE
#undef RJ_VOP3_CVT_F16_F8_DECODE_RULES
#undef RJ_VOP3_CVT_F16_F8_ENCODE_RULES
#undef RJ_GFX1250_EXPAND

} // namespace

std::span<const TranslationRule> semantic_expand_rules_gfx1250_to_rdna4() {
  return std::span<const TranslationRule>(kExpandRules_gfx1250_to_rdna4);
}

} // namespace rocjitsu
