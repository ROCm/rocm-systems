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
constexpr uint32_t kK128Fp8BorrowedVgprCount = 5;
constexpr uint32_t kPrivateBorrowedVgprCount = 21;
constexpr uint32_t kPrivateBorrowScratchBytes = kPrivateBorrowedVgprCount * sizeof(uint32_t);

[[nodiscard]] constexpr std::pair<uint32_t, uint32_t> build_vop3p(uint8_t op, uint8_t vdst,
                                                                  uint16_t src0, uint16_t src1,
                                                                  uint16_t src2, uint8_t neg = 0,
                                                                  bool clamp = false) {
  const uint32_t w0 = static_cast<uint32_t>(vdst) | (1u << 14) |
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
  if (src0 < 128 || src0 >= 256)
    return static_cast<uint16_t>(src0 + 1);

  if (src0 >= scalar_positive_inline_u32(0) && src0 <= scalar_positive_inline_u32(64))
    return scalar_positive_inline_u32(0);
  if (src0 >= scalar_negative_inline_i32(-1) && src0 <= scalar_negative_inline_i32(-16))
    return scalar_negative_inline_i32(-1);
  return std::nullopt;
}

[[nodiscard]] std::optional<uint16_t> pair_hi_src_with_literal(uint16_t src0,
                                                               std::optional<int32_t> literal) {
  if (src0 == 255) {
    if (!literal)
      return std::nullopt;
    return scalar_inline_i32(*literal < 0 ? -1 : 0);
  }
  if (src0 == 254)
    return std::nullopt;
  return pair_hi_src_sign_extended_inline(src0);
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

void append_v_mul_u64_low64(std::vector<uint32_t> &words, uint8_t out_lo, uint8_t out_hi,
                            uint16_t src0_lo, uint16_t src0_hi, uint16_t src1_lo, uint16_t src1_hi,
                            std::optional<uint32_t> literal = std::nullopt) {
  constexpr uint16_t kOpMulLoU32 = 812;
  constexpr uint16_t kOpMulHiU32 = 813;
  constexpr uint16_t kOpAddNcU32 = 293;
  constexpr uint16_t kVgprSrcBase = 256;

  append_vop3(words, kOpMulHiU32, out_hi, src0_lo, src1_lo, 0, literal);
  append_vop3(words, kOpMulLoU32, out_lo, src0_hi, src1_lo, 0, literal);
  words.push_back(build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4));
  append_vop3(words, kOpAddNcU32, out_hi, static_cast<uint16_t>(kVgprSrcBase + out_hi),
              static_cast<uint16_t>(kVgprSrcBase + out_lo));
  append_vop3(words, kOpMulLoU32, out_lo, src0_lo, src1_hi, 0, literal);
  words.push_back(build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4));
  append_vop3(words, kOpAddNcU32, out_hi, static_cast<uint16_t>(kVgprSrcBase + out_hi),
              static_cast<uint16_t>(kVgprSrcBase + out_lo));
  append_vop3(words, kOpMulLoU32, out_lo, src0_lo, src1_lo, 0, literal);
}

[[nodiscard]] std::vector<uint32_t>
expand_v_mul_u64_high_scratch(uint8_t vdst, uint16_t src0_lo, uint16_t src0_hi, uint16_t src1_lo,
                              uint16_t src1_hi, const Instruction &inst,
                              std::optional<uint32_t> literal_word,
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
  words.reserve(literal_word ? 32 : 28);
  append_s_get_vgpr_msb_mode(words, mode_save);

  append_s_set_vgpr_msb_mode(words, kModeDstHigh);
  append_vop3(words, kOpMulHiU32, tmp_hi, src0_lo, src1_lo, 0, literal_word);
  append_vop3(words, kOpMulLoU32, tmp_lo, src0_hi, src1_lo, 0, literal_word);

  append_s_set_vgpr_msb_mode(words, kModeScratchAdd);
  words.push_back(build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4));
  append_vop3(words, kOpAddNcU32, tmp_hi, static_cast<uint16_t>(kVgprSrcBase + tmp_hi),
              static_cast<uint16_t>(kVgprSrcBase + tmp_lo));

  append_s_set_vgpr_msb_mode(words, kModeDstHigh);
  append_vop3(words, kOpMulLoU32, tmp_lo, src0_lo, src1_hi, 0, literal_word);

  append_s_set_vgpr_msb_mode(words, kModeScratchAdd);
  words.push_back(build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4));
  append_vop3(words, kOpAddNcU32, tmp_hi, static_cast<uint16_t>(kVgprSrcBase + tmp_hi),
              static_cast<uint16_t>(kVgprSrcBase + tmp_lo));

  append_s_set_vgpr_msb_mode(words, kModeDstHigh);
  append_vop3(words, kOpMulLoU32, tmp_lo, src0_lo, src1_lo, 0, literal_word);

  words.push_back(build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4));
  append_s_set_vgpr_msb_mode(words, kModeSrc0High);
  append_vop1(words, kOpMovB32, vdst, static_cast<uint16_t>(kVgprSrcBase + tmp_lo));
  append_vop1(words, kOpMovB32, static_cast<uint8_t>(vdst + 1u),
              static_cast<uint16_t>(kVgprSrcBase + tmp_hi));
  append_s_set_vgpr_msb_mode_from_sgpr(words, mode_save);
  return words;
}

std::vector<uint32_t> expand_v_mul_u64(uint8_t vdst, uint16_t src0, uint16_t src1,
                                       const Instruction &inst,
                                       std::optional<uint32_t> literal_word,
                                       const LivenessAnalysis &liveness) {
  if (vdst > 254)
    return {};

  const std::optional<int32_t> literal_s32 =
      literal_word ? std::optional<int32_t>(static_cast<int32_t>(*literal_word)) : std::nullopt;
  const auto src0_hi = pair_hi_src_with_literal(src0, literal_s32);
  const auto src1_hi = pair_hi_src_with_literal(src1, literal_s32);
  if (!src0_hi || !src1_hi)
    return {};
  if (src0 > 511u || src1 > 511u || *src0_hi > 511u || *src1_hi > 511u)
    return {};

  const bool overlaps_dst_pair = source_pair_reads_vdst_pair(vdst, src0, *src0_hi) ||
                                 source_pair_reads_vdst_pair(vdst, src1, *src1_hi);

  std::vector<uint32_t> words;
  if (!overlaps_dst_pair) {
    words.reserve(literal_word ? 15 : 11);
    append_v_mul_u64_low64(words, vdst, static_cast<uint8_t>(vdst + 1u), src0, *src0_hi, src1,
                           *src1_hi, literal_word);
    return words;
  }

  std::vector<uint8_t> avoid;
  add_avoid_vgpr_run(avoid, vdst, 2);
  add_avoid_source_pair_vgprs(avoid, src0, *src0_hi);
  add_avoid_source_pair_vgprs(avoid, src1, *src1_hi);
  const auto tmp_opt = find_free_vgpr_run_avoiding(inst, liveness, 2, avoid);
  if (!tmp_opt) {
    return expand_v_mul_u64_high_scratch(vdst, src0, *src0_hi, src1, *src1_hi, inst, literal_word,
                                         liveness);
  }

  constexpr uint16_t kVgprSrcBase = 256;
  constexpr uint8_t kOpMovB32 = 1;
  const uint8_t tmp = static_cast<uint8_t>(*tmp_opt);
  words.reserve(literal_word ? 19 : 15);
  append_v_mul_u64_low64(words, tmp, static_cast<uint8_t>(tmp + 1u), src0, *src0_hi, src1, *src1_hi,
                         literal_word);
  words.push_back(build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4));
  append_vop1(words, kOpMovB32, vdst, static_cast<uint16_t>(kVgprSrcBase + tmp));
  append_vop1(words, kOpMovB32, static_cast<uint8_t>(vdst + 1u),
              static_cast<uint16_t>(kVgprSrcBase + tmp + 1u));
  return words;
}

[[nodiscard]] std::optional<uint32_t> simm32_literal_word(const Instruction &inst,
                                                          uint8_t operand_index);

[[nodiscard]] std::optional<uint64_t> simm64_literal_value(const Instruction &inst,
                                                           uint8_t operand_index);

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

[[nodiscard]] uint8_t scaled_offset_shift_for_vglobal(uint16_t op) {
  switch (op) {
  case 20: // global_load_b32
  case 26: // global_store_b32
    return 2;
  case 21:  // global_load_b64
  case 88:  // global_load_tr8_b64
  case 115: // global_load_tr4_b64
    return 3;
  case 23: // global_load_b128
  case 87: // global_load_tr16_b128
    return 4;
  default:
    return 0;
  }
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

std::vector<uint32_t> expand_scaled_vglobal_b32(const Instruction &inst, uint32_t, uint64_t,
                                                const LivenessAnalysis &liveness,
                                                const LaneLayout *, const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::VglobalMachineInst))
    return {};

  gfx1250::VglobalMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (!src.scale_offset)
    return {};

  const uint8_t shift = scaled_offset_shift_for_vglobal(inst.opcode());
  if (shift == 0)
    return {};

  auto tmp_opt = liveness.find_free_run(&inst, 1);
  if (!tmp_opt || *tmp_opt > 255)
    return {};
  const auto tmp = static_cast<uint8_t>(*tmp_opt);

  auto fields = gfx1250_to_rdna4::decode_vglobal_gfx1250(raw[0], raw[1], raw[2]);
  fields.vaddr = tmp;
  fields.scale_offset = 0;
  auto mem = gfx1250_to_rdna4::encode_vglobal_rdna4(fields, inst.opcode());
  if (mem.word_count != 3)
    return {};

  constexpr uint8_t kOpLshlrevB32 = 24;
  std::vector<uint32_t> words;
  words.reserve(6);
  words.push_back(build_vop2(kOpLshlrevB32, tmp, scalar_positive_inline_u32(shift), src.vaddr));
  append_wait_valu_vgpr(words);
  words.insert(words.end(), mem.words, mem.words + mem.word_count);
  return words;
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

  const uint8_t scale_shift = scaled_offset_shift_for_vglobal(src.op);
  if (src.scale_offset != 0) {
    if (scale_shift == 0)
      return {};
    fields.vaddr = addr;
    append_vop2(words, kOpLshlrevB32, addr, scalar_positive_inline_u32(scale_shift), src.vaddr);
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
    const uint8_t shift = scaled_offset_shift_for_vglobal(src.op);
    if (shift == 0)
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
    append_vop2(words, kOpLshlrevB32, tmp, scalar_positive_inline_u32(shift), src.vaddr);
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

std::vector<uint32_t> expand_v_mov_b64(const Instruction &inst, uint32_t, uint64_t,
                                       const LivenessAnalysis &, const LaneLayout *,
                                       const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() != sizeof(uint32_t))
    return {};

  auto src = std::bit_cast<gfx1250::Vop1MachineInst>(raw[0]);
  auto hi = pair_hi_src(static_cast<uint16_t>(src.src0));
  if (!hi || src.vdst == 255)
    return {};

  constexpr uint8_t kOpMovB32 = 1;
  return {build_vop1(kOpMovB32, static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src0)),
          build_vop1(kOpMovB32, static_cast<uint8_t>(src.vdst + 1), *hi)};
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

  auto hi = pair_hi_src(static_cast<uint16_t>(src.src0));
  if (!hi || src.vdst == 255)
    return {};

  constexpr uint8_t kOpMovB32 = 1;
  return {build_vop1(kOpMovB32, static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src0)),
          build_vop1(kOpMovB32, static_cast<uint8_t>(src.vdst + 1), *hi)};
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

std::vector<uint32_t> expand_v_sub_nc_u64(uint8_t vdst, uint16_t src0, uint16_t src1,
                                          const Instruction &inst,
                                          const LivenessAnalysis &liveness) {
  if (vdst == 255)
    return {};

  const auto src0_hi = pair_hi_src_sign_extended_inline(src0);
  const auto src1_hi = pair_hi_src_sign_extended_inline(src1);
  if (!src0_hi || !src1_hi)
    return {};

  if (low_write_clobbers_high_source(vdst, *src0_hi) ||
      low_write_clobbers_high_source(vdst, *src1_hi))
    return {};

  auto carry_sgpr = liveness.find_free_sgpr_pair(&inst);
  if (!carry_sgpr || *carry_sgpr > 105)
    return {};

  constexpr uint16_t kOpSubCoCiU32 = 289;
  constexpr uint16_t kOpSubCoU32 = 769;
  constexpr uint8_t kSoppWaitAlu = 8;
  const auto carry = static_cast<uint8_t>(*carry_sgpr);

  std::vector<uint32_t> words;
  words.reserve(6);
  {
    auto [w0, w1] = build_vop3_sdst(kOpSubCoU32, vdst, carry, src0, src1);
    words.push_back(w0);
    words.push_back(w1);
  }
  words.push_back(pack_sopp(kSoppWaitAlu, kWaitAluDepctrVaSdst0));
  {
    auto [w0, w1] = build_vop3_sdst(kOpSubCoCiU32, static_cast<uint8_t>(vdst + 1), kNullSgpr,
                                    *src0_hi, *src1_hi, carry);
    words.push_back(w0);
    words.push_back(w1);
  }
  words.push_back(pack_sopp(kSoppWaitAlu, kWaitAluDepctrVaVdst0));
  return words;
}

std::vector<uint32_t> expand_v_add_nc_u64(uint8_t vdst, uint16_t src0, uint16_t src1,
                                          const Instruction &inst,
                                          std::optional<uint32_t> literal_word,
                                          const LivenessAnalysis &liveness) {
  if (vdst == 255)
    return {};

  const std::optional<int32_t> literal_s32 =
      literal_word ? std::optional<int32_t>(static_cast<int32_t>(*literal_word)) : std::nullopt;
  const auto src0_hi = pair_hi_src_with_literal(src0, literal_s32);
  const auto src1_hi = pair_hi_src_sign_extended_inline(src1);
  if (!src0_hi || !src1_hi)
    return {};

  if (low_write_clobbers_high_source(vdst, *src0_hi) ||
      low_write_clobbers_high_source(vdst, *src1_hi))
    return {};

  auto carry_sgpr = liveness.find_free_sgpr_pair(&inst);
  if (!carry_sgpr || *carry_sgpr > 105)
    return {};

  constexpr uint16_t kOpAddCoCiU32 = 288;
  constexpr uint16_t kOpAddCoU32 = 768;
  constexpr uint8_t kSoppWaitAlu = 8;
  const auto carry = static_cast<uint8_t>(*carry_sgpr);

  std::vector<uint32_t> words;
  words.reserve(literal_word ? 7 : 6);
  {
    auto [w0, w1] = build_vop3_sdst(kOpAddCoU32, vdst, carry, src0, src1);
    words.push_back(w0);
    words.push_back(w1);
    if (literal_word && (src0 == 255 || src1 == 255))
      words.push_back(*literal_word);
  }
  words.push_back(pack_sopp(kSoppWaitAlu, kWaitAluDepctrVaSdst0));
  {
    auto [w0, w1] = build_vop3_sdst(kOpAddCoCiU32, static_cast<uint8_t>(vdst + 1), kNullSgpr,
                                    *src0_hi, *src1_hi, carry);
    words.push_back(w0);
    words.push_back(w1);
  }
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

  std::optional<uint32_t> literal_word;
  if (src.src0 == 255) {
    if (inst.size() != sizeof(gfx1250::Vop2InstLiteralMachineInst))
      return {};
    literal_word = simm32_literal_word(inst, 0);
    if (!literal_word)
      return {};
  } else if (inst.size() != sizeof(uint32_t)) {
    return {};
  }

  return expand_v_add_nc_u64(static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src0),
                             static_cast<uint16_t>(256u + src.vsrc1), inst, literal_word, liveness);
}

std::vector<uint32_t> expand_v_add_nc_u64_vop3(const Instruction &inst, uint32_t, uint64_t,
                                               const LivenessAnalysis &liveness, const LaneLayout *,
                                               const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() != sizeof(gfx1250::Vop3MachineInst))
    return {};

  gfx1250::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.src0 == 255 || src.src0 == 254 || src.src1 == 255 || src.src1 == 254)
    return {};

  return expand_v_add_nc_u64(static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src0),
                             static_cast<uint16_t>(src.src1), inst, std::nullopt, liveness);
}

std::vector<uint32_t> expand_v_sub_nc_u64_e32(const Instruction &inst, uint32_t, uint64_t,
                                              const LivenessAnalysis &liveness, const LaneLayout *,
                                              const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() != sizeof(uint32_t))
    return {};

  auto src = std::bit_cast<gfx1250::Vop2MachineInst>(raw[0]);
  if (src.vsrc1 == 255)
    return {};

  return expand_v_sub_nc_u64(static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src0),
                             static_cast<uint16_t>(256u + src.vsrc1), inst, liveness);
}

std::vector<uint32_t> expand_v_sub_nc_u64_vop3(const Instruction &inst, uint32_t, uint64_t,
                                               const LivenessAnalysis &liveness, const LaneLayout *,
                                               const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() != sizeof(gfx1250::Vop3MachineInst))
    return {};

  gfx1250::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.src0 == 255 || src.src0 == 254 || src.src1 == 255 || src.src1 == 254)
    return {};

  return expand_v_sub_nc_u64(static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src0),
                             static_cast<uint16_t>(src.src1), inst, liveness);
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

  std::vector<uint32_t> words;
  append_vop3(words, kOpVAddF16Vop3, static_cast<uint8_t>(src.vdst),
              static_cast<uint16_t>(src.src0), static_cast<uint16_t>(256u + src.vsrc1), 0,
              literal_word);
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
      src.neg != 0 || src.src0 == 254 || src.src1 == 254 || src.src2 == 254 || src.src2 == 255)
    return {};

  std::optional<uint32_t> literal_word;
  if (src.src0 == 255 || src.src1 == 255) {
    literal_word = (src.src0 == 255) ? simm32_literal_word(inst, 0) : simm32_literal_word(inst, 1);
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
              static_cast<uint16_t>(kTmpSrcBase + tmp));
  return words;
}

std::vector<uint32_t> expand_v_mad_nc_u64_u32_vop3(const Instruction &inst, uint32_t, uint64_t,
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

  constexpr uint16_t kOpMadCoU64U32 = 766;
  auto [w0, w1] = build_vop3_sdst(kOpMadCoU64U32, static_cast<uint8_t>(src.vdst), kNullSgpr, src0,
                                  src1, static_cast<uint16_t>(src.src2));
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
  if (src.vsrc1 == 255 || src.src0 == 254)
    return {};

  std::optional<uint32_t> literal_word;
  if (src.src0 == 255) {
    if (inst.size() != sizeof(gfx1250::Vop2InstLiteralMachineInst))
      return {};
    literal_word = simm32_literal_word(inst, 0);
    if (!literal_word)
      return {};
  } else if (inst.size() != sizeof(uint32_t)) {
    return {};
  }

  return expand_v_mul_u64(static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src0),
                          static_cast<uint16_t>(256u + src.vsrc1), inst, literal_word, liveness);
}

std::vector<uint32_t> expand_v_mul_u64_vop3(const Instruction &inst, uint32_t, uint64_t,
                                            const LivenessAnalysis &liveness, const LaneLayout *,
                                            const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3MachineInst))
    return {};

  gfx1250::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.vdst > 254 || src.abs != 0 || src.opsel != 0 || src.clamp != 0 || src.omod != 0 ||
      src.neg != 0 || src.src0 == 254 || src.src1 == 254)
    return {};

  std::optional<uint32_t> literal_word;
  if (src.src0 == 255 || src.src1 == 255) {
    literal_word = (src.src0 == 255) ? simm32_literal_word(inst, 0) : simm32_literal_word(inst, 1);
    if (!literal_word)
      return {};
  }

  return expand_v_mul_u64(static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(src.src0),
                          static_cast<uint16_t>(src.src1), inst, literal_word, liveness);
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

  uint16_t shift = 0;
  if (src.src1 == scalar_positive_inline_u32(1))
    shift = 1;
  else if (src.src1 == scalar_positive_inline_u32(2))
    shift = 2;
  else
    return {};

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

  const bool stage_src0_in_dst = scalar_register_src(static_cast<uint16_t>(src.src0));
  const bool stage_src1_in_dst =
      !stage_src0_in_dst && scalar_register_src(static_cast<uint16_t>(src.src1));
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

std::vector<uint32_t> expand_v_min_i64_vop3(const Instruction &inst, uint32_t, uint64_t,
                                            const LivenessAnalysis &liveness, const LaneLayout *,
                                            const LaneLayout *) {
  constexpr uint16_t kOpCmpLtI64 = 81;
  return expand_v_minmax_64_vop3(inst, liveness, kOpCmpLtI64);
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
                                  std::optional<uint32_t> literal_word = std::nullopt) {
  constexpr uint8_t kOpWmmaF32_16x16x16_F16 = 64;
  auto [w0, w1] = build_vop3p(kOpWmmaF32_16x16x16_F16, vdst, static_cast<uint16_t>(256u + src0),
                              static_cast<uint16_t>(256u + src1), src2);
  words.push_back(w0);
  words.push_back(w1);
  if (literal_word && src2 == 255)
    words.push_back(*literal_word);
}

void append_wmma_f32_16x16x16_bf16(std::vector<uint32_t> &words, uint8_t vdst, uint8_t src0,
                                   uint8_t src1, uint16_t src2,
                                   std::optional<uint32_t> literal_word = std::nullopt) {
  constexpr uint8_t kOpWmmaF32_16x16x16_BF16 = 65;
  auto [w0, w1] = build_vop3p(kOpWmmaF32_16x16x16_BF16, vdst, static_cast<uint16_t>(256u + src0),
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

void append_swmmac_f32_16x16x32_f16(std::vector<uint32_t> &words, uint8_t vdst, uint8_t src0,
                                    uint8_t src1, uint8_t src2) {
  constexpr uint8_t kOpSwmmacF32_16x16x32_F16 = 0x50;
  auto [w0, w1] =
      build_vop3p(kOpSwmmacF32_16x16x32_F16, vdst, static_cast<uint16_t>(256u + src0),
                  static_cast<uint16_t>(256u + src1), static_cast<uint16_t>(256u + src2));
  words.push_back(w0);
  words.push_back(w1);
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

void append_wmma_f32_16x16x16_fp8_fp8(std::vector<uint32_t> &words, uint8_t vdst, uint8_t src0,
                                      uint8_t src1, uint16_t src2,
                                      std::optional<uint32_t> literal_word = std::nullopt) {
  constexpr uint8_t kOpWmmaF32_16x16x16_Fp8Fp8 = 70;
  auto [w0, w1] = build_vop3p(kOpWmmaF32_16x16x16_Fp8Fp8, vdst, static_cast<uint16_t>(256u + src0),
                              static_cast<uint16_t>(256u + src1), src2);
  words.push_back(w0);
  words.push_back(w1);
  if (literal_word && src2 == 255)
    words.push_back(*literal_word);
}

void append_v_dot4_f32_fp8_fp8(std::vector<uint32_t> &words, uint8_t vdst, uint8_t src0,
                               uint8_t src1, uint16_t src2) {
  constexpr uint8_t kOpVDot4F32Fp8Fp8 = 38;
  auto [w0, w1] = build_vop3p(kOpVDot4F32Fp8Fp8, vdst, static_cast<uint16_t>(256u + src0),
                              static_cast<uint16_t>(256u + src1), src2);
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

void append_wmma_f32_k128_fp8_dot4_term(std::vector<uint32_t> &words, uint8_t dst_reg,
                                        uint8_t src_a, uint8_t src_b, uint8_t tmp_a, uint8_t tmp_b,
                                        uint8_t vaddr, uint8_t output_reg, uint8_t k_group,
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
  append_v_dot4_f32_fp8_fp8(words, dst_reg, tmp_a, tmp_b, static_cast<uint16_t>(256u + dst_reg));
}

void append_e8m0_scale_f32_from_word(std::vector<uint32_t> &words, uint8_t scale_f32,
                                     uint8_t scale_word, uint8_t byte_index, uint8_t pred_sgpr,
                                     bool inline_zero_is_unity = false) {
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
  append_vop3(words, kOpCmpEqU32, pred_sgpr, static_cast<uint16_t>(256u + scale_f32), 255, 0,
              kByteMask);
  append_vop2(words, kOpLshlrevB32, scale_f32, kInlineConst23, scale_f32);
  words.push_back(pack_sopp(kSoppWaitAlu, kWaitAluDepctrVaSdst0));
  append_wait_valu_vgpr(words);
  append_vop3(words, kOpCndmaskB32, scale_f32, static_cast<uint16_t>(256u + scale_f32), 255,
              pred_sgpr, kF32QuietNaN);
}

void append_wmma_f32_apply_e8m0_scales_and_accumulate(
    std::vector<uint32_t> &words, uint8_t dst_reg, uint8_t contribution, uint16_t scale_src_a,
    uint16_t scale_src_b, uint8_t scale_a_tmp, uint8_t scale_b_tmp, uint8_t vaddr,
    uint8_t output_reg, uint8_t scale_byte_a, uint8_t scale_byte_b, uint8_t matrix_a_scale_select,
    uint8_t matrix_b_scale_select, uint8_t pred_sgpr, uint8_t exec_save);

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
  append_v_dot4_f32_fp8_fp8(words, tmp_dot, tmp_a, tmp_b, scalar_positive_inline_u32(0));

  append_wmma_f32_apply_e8m0_scales_and_accumulate(
      words, dst_reg, tmp_dot, scale_src_a, scale_src_b, tmp_a, tmp_b, vaddr, output_reg,
      scale_byte, scale_byte, matrix_a_scale_select, matrix_b_scale_select, pred_sgpr, exec_save);
}

void append_wmma_f32_apply_e8m0_scales_and_accumulate(
    std::vector<uint32_t> &words, uint8_t dst_reg, uint8_t contribution, uint16_t scale_src_a,
    uint16_t scale_src_b, uint8_t scale_a_tmp, uint8_t scale_b_tmp, uint8_t vaddr,
    uint8_t output_reg, uint8_t scale_byte_a, uint8_t scale_byte_b, uint8_t matrix_a_scale_select,
    uint8_t matrix_b_scale_select, uint8_t pred_sgpr, uint8_t exec_save) {
  constexpr uint8_t kOpAndB32 = 27;
  constexpr uint8_t kOpLshlrevB32 = 24;
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
    append_vop2(words, kOpAndB32, vaddr, kInlineConst16, vaddr);
    append_vop2(words, kOpLshlrevB32, vaddr, kInlineConst1, vaddr);
    if (output_reg != 0)
      append_vop2(words, kOpAddNcU32, vaddr, scalar_positive_inline_u32(output_reg * 4u), vaddr);
    if (matrix_a_scale_select & 0x1u)
      append_vop2(words, kOpAddNcU32, vaddr, kInlineConst64, vaddr);
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
  append_e8m0_scale_f32_from_word(words, scale_a_tmp, scale_a_tmp, scale_byte_a, pred_sgpr,
                                  scale_src_a_inline_zero);
  append_e8m0_scale_f32_from_word(words, scale_b_tmp, scale_b_tmp, scale_byte_b, pred_sgpr,
                                  scale_src_b_inline_zero);
  append_vop3(words, kOpMulF32, vaddr, static_cast<uint16_t>(256u + contribution),
              static_cast<uint16_t>(256u + scale_a_tmp));
  append_vop3(words, kOpMulF32, vaddr, static_cast<uint16_t>(256u + vaddr),
              static_cast<uint16_t>(256u + scale_b_tmp));
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

  append_wmma_f32_apply_e8m0_scales_and_accumulate(
      words, dst_reg, tmp_f32, scale_src_a, scale_src_b, tmp_a_pos, tmp_b_pos, vaddr, output_reg,
      scale_byte_a, scale_byte_b, matrix_a_scale_select, matrix_b_scale_select, pred_sgpr,
      exec_save);
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
  if (src.vdst > 248 || src.neg_hi != 0 || (src.opsel & ~0x4u) != 0 || src.clamp != 0 ||
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

  append_wmma_f32_16x16x16_f16(words, tmp_acc, *src0_base, *src1_base, first_acc, literal_word);
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
  if (src.vdst > 248 || src.neg_hi != 0 || (src.opsel & ~0x4u) != 0 || src.clamp != 0 ||
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

  append_wmma_f32_16x16x16_bf16(words, tmp_acc, *src0_base, *src1_base, first_acc, literal_word);
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

ExpandResult expand_v_wmma_scale_f32_16x16x128_f8f6f4(const Instruction &inst, uint32_t, uint64_t,
                                                      const LivenessAnalysis &liveness,
                                                      TranslationContext &, const LaneLayout *,
                                                      const LaneLayout *) {
  auto fp4_words = lower_v_wmma_scale_f32_16x16x128_f8f6f4_fp4_fp4(inst, liveness);
  if (!fp4_words.empty())
    return ExpandResult::success(std::move(fp4_words));

  auto words = lower_v_wmma_scale_f32_16x16x128_f8f6f4_fp8_fp8(inst, liveness);
  if (!words.empty())
    return ExpandResult::success(std::move(words));

  return ExpandResult::failed(
      "v_wmma_scale_f32_16x16x128_f8f6f4 requires supported E8M0 scale lowering for "
      "gfx1250-to-RDNA4 DBT");
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
  if (src.vdst > 248 || src.neg_hi != 0 || src.clamp != 0 || src.neg != 0 ||
      (src.opsel & ~0x4u) != 0)
    return {};

  const auto src0_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src0), 8);
  const auto src1_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src1), 16);
  const auto index_base = src_vgpr_base_for_run(static_cast<uint16_t>(src.src2), 1);
  if (!src0_base || !src1_base || !index_base)
    return {};

  std::vector<uint8_t> avoid;
  add_avoid_vgpr_run(avoid, static_cast<uint8_t>(src.vdst), 8);
  add_avoid_vgpr_run(avoid, *src0_base, 8);
  add_avoid_vgpr_run(avoid, *src1_base, 16);
  add_avoid_vgpr(avoid, *index_base);

  auto exec_save_opt = liveness.find_free_sgpr_pair(&inst);
  if (!exec_save_opt || *exec_save_opt > 124)
    return {};
  const uint8_t exec_save = static_cast<uint8_t>(*exec_save_opt);

  constexpr uint8_t kTmpCount = 21;
  auto tmp_base_opt = find_aligned_free_vgpr_run_avoiding(inst, liveness, kTmpCount, 8, avoid);
  std::optional<PrivateBorrowedVgprRun> borrowed;
  if (!tmp_base_opt || *tmp_base_opt > 235) {
    borrowed = find_private_borrowed_vgpr_run(liveness, kTmpCount, 8, avoid);
    if (!borrowed || borrowed->base > 235)
      return {};
    tmp_base_opt = borrowed->base;
  }
  const uint8_t tmp_acc = static_cast<uint8_t>(*tmp_base_opt);
  const uint8_t tmp_a = static_cast<uint8_t>(tmp_acc + 8u);
  const uint8_t tmp_b = static_cast<uint8_t>(tmp_acc + 12u);
  const uint8_t vaddr_xor16 = static_cast<uint8_t>(tmp_acc + 20u);

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

  append_v_mov_b32_run(words, tmp_acc, static_cast<uint16_t>(256u + src.vdst), 8);

  for (uint8_t chunk = 0; chunk < 2; ++chunk) {
    append_wmma_single_relayout_chunk(words, static_cast<uint16_t>(src.src0), tmp_a, vaddr_xor16, 8,
                                      4, chunk, exec_save);
    append_wmma_single_relayout_chunk(words, static_cast<uint16_t>(src.src1), tmp_b, vaddr_xor16,
                                      16, 8, chunk, exec_save);
    append_swmmac_f32_16x16x32_f16(words, tmp_acc, tmp_a, tmp_b, *index_base);
    words.push_back(pack_sopp(kSoppWaitKmcnt, 0));
  }

  append_v_mov_b32_run(words, static_cast<uint8_t>(src.vdst), static_cast<uint16_t>(256u + tmp_acc),
                       8);
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

std::vector<uint32_t> expand_v_wmma_f32_16x16xk_fp8_fp8(const Instruction &inst,
                                                        const LivenessAnalysis &liveness,
                                                        uint8_t src_words, uint8_t chunks) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3pMachineInst))
    return {};

  gfx1250::Vop3pMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
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
    append_wmma_f32_16x16x16_fp8_fp8(words, tmp_acc, tmp_a, tmp_b, acc_src, literal_word);
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
  return expand_v_wmma_f32_16x16xk_fp8_fp8(inst, liveness, 8, 4);
}

std::vector<uint32_t>
expand_v_wmma_f32_16x16x128_fp8_fp8_dot4_fallback(const Instruction &inst,
                                                  const LivenessAnalysis &liveness) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3pMachineInst))
    return {};

  gfx1250::Vop3pMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
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

  std::vector<uint8_t> avoid;
  add_avoid_vgpr_run(avoid, static_cast<uint8_t>(src.vdst), 8);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src0), 16);
  add_avoid_src_vgpr_run(avoid, static_cast<uint16_t>(src.src1), 16);
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

  auto exec_save_opt = liveness.find_free_sgpr_pair(&inst);
  if (!exec_save_opt || *exec_save_opt > 124)
    return {};
  const uint8_t exec_save = static_cast<uint8_t>(*exec_save_opt);

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

  for (uint8_t reg = 0; reg < 8; ++reg) {
    const uint8_t dst_reg = static_cast<uint8_t>(src.vdst + reg);
    for (uint8_t k_group = 0; k_group < 32; ++k_group) {
      append_wmma_f32_k128_fp8_dot4_term(words, dst_reg, *src0_base, *src1_base, tmp_a, tmp_b,
                                         vaddr, reg, k_group, exec_save);
    }
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
    append_wmma_f32_16x16x16_fp8_fp8(words, static_cast<uint8_t>(src.vdst), kTmpA, kTmpB, acc_src);
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
      append_v_dot4_f32_fp8_fp8(words, dst_reg, scalar_a,
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
        append_v_dot4_f32_fp8_fp8(words, dst_reg, scalar_a, scalar_b,
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

std::vector<uint32_t> expand_v_bitop3_b32_vop3(const Instruction &inst, uint32_t, uint64_t,
                                               const LivenessAnalysis &, const LaneLayout *,
                                               const LaneLayout *) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(gfx1250::Vop3MachineInst))
    return {};

  gfx1250::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  const uint8_t truth_table = static_cast<uint8_t>((src.omod << 6) | (src.abs << 3) | src.neg);
  if (truth_table != 0xC8 || src.opsel != 0 || src.clamp != 0)
    return {};
  if (src.src0 == 254 || src.src1 == 254 || src.src2 == 254)
    return {};

  uint16_t or_src0 = 0;
  uint8_t or_vsrc1 = 0;
  if (auto src2_vgpr = vgpr_index(static_cast<uint16_t>(src.src2))) {
    or_src0 = static_cast<uint16_t>(src.src0);
    or_vsrc1 = *src2_vgpr;
  } else if (auto src0_vgpr = vgpr_index(static_cast<uint16_t>(src.src0))) {
    or_src0 = static_cast<uint16_t>(src.src2);
    or_vsrc1 = *src0_vgpr;
  } else {
    return {};
  }

  const bool or_uses_literal = or_src0 == 255;
  const bool and_uses_literal = src.src1 == 255;
  if (or_uses_literal && and_uses_literal)
    return {};
  const auto literal_word = or_uses_literal
                                ? simm32_literal_word(inst, src.src0 == 255 ? 0 : 2)
                                : (and_uses_literal ? simm32_literal_word(inst, 1) : std::nullopt);
  if ((or_uses_literal || and_uses_literal) && !literal_word)
    return {};

  const auto src1_vgpr = vgpr_index(static_cast<uint16_t>(src.src1));
  if (src1_vgpr && *src1_vgpr == src.vdst)
    return {};

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
  return words;
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

std::vector<uint32_t> expand_v_cvt_pk_bf16_f32_vop3(const Instruction &inst, uint32_t, uint64_t,
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

  const auto src0 = vgpr_index(static_cast<uint16_t>(src.src0));
  const auto src1 = vgpr_index(static_cast<uint16_t>(src.src1));
  if (!src0 || !src1)
    return {};

  std::vector<uint8_t> avoid;
  add_avoid_vgpr(avoid, static_cast<uint8_t>(src.vdst));
  add_avoid_vgpr(avoid, *src0);
  add_avoid_vgpr(avoid, *src1);

  constexpr uint16_t kTmpCount = 3;
  const auto tmp_base = find_free_vgpr_run_avoiding(inst, liveness, kTmpCount, avoid);
  if (!tmp_base)
    return {};
  const auto lo = static_cast<uint8_t>(*tmp_base);
  const auto hi = static_cast<uint8_t>(*tmp_base + 1u);
  const auto tmp = static_cast<uint8_t>(*tmp_base + 2u);

  constexpr uint8_t kOpLshlrevB32 = 24;
  constexpr uint8_t kOpOrB32 = 28;

  std::vector<uint32_t> words;
  words.reserve(14);
  append_f32_to_bf16_rne(words, lo, tmp, *src0);
  append_f32_to_bf16_rne(words, hi, tmp, *src1);
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

  const auto src0 = vgpr_index(static_cast<uint16_t>(src.src0));
  const auto src1 = vgpr_index(static_cast<uint16_t>(src.src1));
  if (!src0 || !src1)
    return {};

  std::vector<uint8_t> avoid;
  add_avoid_vgpr(avoid, static_cast<uint8_t>(src.vdst));
  add_avoid_vgpr(avoid, *src0);
  add_avoid_vgpr(avoid, *src1);

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
  append_vop3(words, kOpCvtF16F32, lo, static_cast<uint16_t>(src.src0), 0);
  append_vop3(words, kOpCvtF16F32, hi, static_cast<uint16_t>(src.src1), 0);
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
constexpr uint16_t kEncSopkMovk = 0x160;
constexpr uint16_t kEncSopkGetreg = 0x171;
constexpr uint16_t kEncSopkSetregImm32 = 0x173;
constexpr uint16_t kEncSop1 = 0x17D;
constexpr uint16_t kEncSopp = 0x17F;
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
constexpr uint16_t kEncVimage = 0x1A0;
constexpr uint16_t kOpSMovB64 = 1;
constexpr uint16_t kOpSMovkI32 = 0;
constexpr uint16_t kOpSMovB32 = 0;
constexpr uint16_t kOpSGetPcI64 = 71;
constexpr uint16_t kOpSSetPcI64 = 72;
constexpr uint16_t kOpSSwapPcI64 = 73;
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
constexpr uint16_t kOpVMovB16Vop1 = 28;
constexpr uint16_t kOpVMovB64Vop1 = 29;
constexpr uint16_t kOpVMovB64Vop3 = 413;
constexpr uint16_t kOpVMulU64Vop3 = 0;
constexpr uint16_t kOpVLshlAddU32Vop3 = 582;
constexpr uint16_t kOpVLshlAddU64Vop3 = 594;
constexpr uint16_t kOpVLshlOrB32Vop3 = 598;
constexpr uint16_t kOpVMaxU64Vop3 = 793;
constexpr uint16_t kOpVMinI64Vop3 = 794;
constexpr uint16_t kOpVCvtPkBf16F32Vop3 = 877;
constexpr uint16_t kOpVCvtPkF16F32Vop3 = 879;
constexpr uint16_t kOpVFmaMixF32Vop3p = 32;
constexpr uint16_t kOpVFmaMixF32Bf16Vop3p = 0x3D;
constexpr uint16_t kOpVPkFmaF32Vop3p = 31;
constexpr uint16_t kOpVPkMulF32Vop3p = 40;
constexpr uint16_t kOpVPkAddF32Vop3p = 41;
constexpr uint16_t kOpVAddNcU64Vop3 = 296;
constexpr uint16_t kOpVAddNcU64E32 = 40;
constexpr uint16_t kOpVSubNcU64Vop3 = 297;
constexpr uint16_t kOpVSubNcU64E32 = 41;
constexpr uint16_t kOpVAddF16E32 = 50;
constexpr uint16_t kOpVCvtF32Bf16E32 = 114;
constexpr uint16_t kOpVLshlrevB64E32 = 31;
constexpr uint16_t kOpVMulU64E32 = 42;
constexpr uint16_t kOpVBitop3B16Vop3 = 563;
constexpr uint16_t kOpVBitop3B32Vop3 = 564;
constexpr uint16_t kOpVMadU32Vop3 = 565;
constexpr uint16_t kOpVMadNcU64U32Vop3 = 762;
constexpr uint16_t kOpVWmmaScaleF32_16x16x128F8f6f4 = 0x35;
constexpr uint16_t kOpVCvtF32Bf16Vop3 = 498;
constexpr uint16_t kOpVWmmaF32_16x16x128F8f6f4 = 0x33;
constexpr uint16_t kOpVWmmaF32_16x16x4F32 = 0x5D;
constexpr uint16_t kOpVWmmaF32_16x16x32F16 = 0x60;
constexpr uint16_t kOpVWmmaF32_16x16x32Bf16 = 0x62;
constexpr uint16_t kOpVSwmmacF32_16x16x64F16 = 0x65;
constexpr uint16_t kOpVWmmaF32_16x16x64Fp8Fp8 = 0x6A;
constexpr uint16_t kOpVWmmaI32_16x16x64Iu8 = 0x72;
constexpr uint16_t kOpVSwmmacI32_16x16x128Iu8 = 0x7B;
constexpr uint16_t kOpVWmmaF32_16x16x128Fp8Fp8 = 0x80;
constexpr uint16_t kOpVLshlrevB16Vop3 = 824;
constexpr uint16_t kOpVOrB16Vop3 = 867;
constexpr uint16_t kOpGlobalLoadB32 = 20;
constexpr uint16_t kOpGlobalLoadB64 = 21;
constexpr uint16_t kOpGlobalLoadB128 = 23;
constexpr uint16_t kOpGlobalStoreB32 = 26;
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
    {kEncVop1_1, kOpVCvtF32F16E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_cvt_f32_f16_e32_high_src), nullptr, nullptr},
    {kEncVop1_1, kOpVMovB16Vop1, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mov_b16), nullptr, nullptr},
    {kEncVop1_1, kOpVMovB64Vop1, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mov_b64), nullptr, nullptr},
    {kEncVop1_1, kOpVCvtF32Bf16E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_cvt_f32_bf16_vop1), nullptr, nullptr},
    {kEncVop1_2, kOpVCvtF32F16E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_cvt_f32_f16_e32_high_src), nullptr, nullptr},
    {kEncVop1_2, kOpVMovB16Vop1, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mov_b16), nullptr, nullptr},
    {kEncVop1_2, kOpVMovB64Vop1, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mov_b64), nullptr, nullptr},
    {kEncVop1_2, kOpVCvtF32Bf16E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_cvt_f32_bf16_vop1), nullptr, nullptr},
    {kEncVop1_3, kOpVCvtF32F16E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_cvt_f32_f16_e32_high_src), nullptr, nullptr},
    {kEncVop1_3, kOpVMovB16Vop1, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mov_b16), nullptr, nullptr},
    {kEncVop1_3, kOpVMovB64Vop1, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mov_b64), nullptr, nullptr},
    {kEncVop1_3, kOpVCvtF32Bf16E32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_cvt_f32_bf16_vop1), nullptr, nullptr},
    {kEncSop2SAndB64, kOpSAndB64, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(lower_s_and_b64_literal64), nullptr, nullptr},
    {kEncSop2SOrB64, kOpSOrB64, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(lower_s_or_b64_literal64), nullptr, nullptr},
    {kEncSop2SAddNcU64, kOpSAddNcU64, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(lower_s_add_nc_u64_to_carry_chain), nullptr, nullptr},
    {kEncSop2SSubNcU64, kOpSSubNcU64, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(lower_s_sub_nc_u64_to_borrow_chain), nullptr, nullptr},
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
    {kEncSopp, kOpSClause, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(lower_s_clause_to_nop), nullptr, nullptr},
    {kEncSopp, kOpSSetVgprMsb, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(lower_s_set_vgpr_msb_to_setreg), nullptr, nullptr},
    {kEncSopp, kOpSWaitKmcnt, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_s_wait_kmcnt), nullptr, nullptr},
    {kEncSopp, kOpSWaitTensorcnt, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(lower_s_clause_to_nop), nullptr, nullptr},
    {kEncVop3p, kOpVPkFmaF32Vop3p, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_pk_fma_f32_vop3p), nullptr, nullptr},
    {kEncVop3p, kOpVFmaMixF32Vop3p, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_fma_mix_f32_f16_vop3p), nullptr, nullptr},
    {kEncVop3p, kOpVPkMulF32Vop3p, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_pk_mul_f32_vop3p), nullptr, nullptr},
    {kEncVop3p, kOpVPkAddF32Vop3p, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_pk_add_f32_vop3p), nullptr, nullptr},
    {kEncVop3p, kOpVWmmaF32_16x16x128F8f6f4, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_wmma_f32_16x16x128_f8f6f4_fp4_fp4), nullptr, nullptr},
    {kEncVop3p, kOpVWmmaScaleF32_16x16x128F8f6f4, RuleAction::Expand, 0, 0, nullptr,
     expand_v_wmma_scale_f32_16x16x128_f8f6f4, nullptr, nullptr},
    {kEncVop3p, kOpVFmaMixF32Bf16Vop3p, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_fma_mix_f32_bf16_vop3p), nullptr, nullptr},
    {kEncVop3p, kOpVWmmaF32_16x16x4F32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_wmma_f32_16x16x4_f32), nullptr, nullptr},
    {kEncVop3p, kOpVWmmaF32_16x16x32F16, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_wmma_f32_16x16x32_f16), nullptr, nullptr},
    {kEncVop3p, kOpVWmmaF32_16x16x32Bf16, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_wmma_f32_16x16x32_bf16), nullptr, nullptr},
    {kEncVop3p, kOpVSwmmacF32_16x16x64F16, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_swmmac_f32_16x16x64_f16), nullptr, nullptr},
    {kEncVop3p, kOpVWmmaF32_16x16x64Fp8Fp8, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_wmma_f32_16x16x64_fp8_fp8), nullptr, nullptr},
    {kEncVop3p, kOpVWmmaI32_16x16x64Iu8, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_wmma_i32_16x16x64_iu8), nullptr, nullptr},
    {kEncVop3p, kOpVSwmmacI32_16x16x128Iu8, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_swmmac_i32_16x16x128_iu8), nullptr, nullptr},
    {kEncVop3p1, kOpVWmmaF32_16x16x128Fp8Fp8, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_wmma_f32_16x16x128_fp8_fp8), nullptr, nullptr},
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
    RJ_VOP3_CVT_F32_BF16_RULE(kEncVop3_0),
    {kEncVop3_0, kOpVMadU32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_u32_vop3), nullptr, nullptr},
    {kEncVop3_0, kOpVLshlAddU32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_add_u32_vop3), nullptr, nullptr},
    {kEncVop3_0, kOpVLshlAddU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_add_u64_vop3), nullptr, nullptr},
    {kEncVop3_0, kOpVLshlOrB32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_or_b32_vop3), nullptr, nullptr},
    {kEncVop3_0, kOpVMadNcU64U32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_nc_u64_u32_vop3), nullptr, nullptr},
    {kEncVop3_0, kOpVMaxU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_max_u64_vop3), nullptr, nullptr},
    {kEncVop3_0, kOpVMinI64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_min_i64_vop3), nullptr, nullptr},
    RJ_VOP3_CVT_PK_BF16_F32_RULE(kEncVop3_0),
    RJ_VOP3_CVT_PK_F16_F32_RULE(kEncVop3_0),
    RJ_VOP3_SINGLE_SRC_RULE(kEncVop3_0),
    {kEncVop3_1, kOpVMulU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mul_u64_vop3), nullptr, nullptr},
    {kEncVop3_1, kOpVAddNcU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_add_nc_u64_vop3), nullptr, nullptr},
    {kEncVop3_1, kOpVSubNcU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_sub_nc_u64_vop3), nullptr, nullptr},
    {kEncVop3_1, kOpVMovB64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mov_b64_vop3), nullptr, nullptr},
    RJ_VOP3_CVT_F32_BF16_RULE(kEncVop3_1),
    {kEncVop3_1, kOpVMadU32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_u32_vop3), nullptr, nullptr},
    {kEncVop3_1, kOpVLshlAddU32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_add_u32_vop3), nullptr, nullptr},
    {kEncVop3_1, kOpVLshlAddU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_add_u64_vop3), nullptr, nullptr},
    {kEncVop3_1, kOpVLshlOrB32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_or_b32_vop3), nullptr, nullptr},
    {kEncVop3_1, kOpVMadNcU64U32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_nc_u64_u32_vop3), nullptr, nullptr},
    {kEncVop3_1, kOpVMaxU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_max_u64_vop3), nullptr, nullptr},
    {kEncVop3_1, kOpVMinI64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_min_i64_vop3), nullptr, nullptr},
    RJ_VOP3_CVT_PK_BF16_F32_RULE(kEncVop3_1),
    RJ_VOP3_CVT_PK_F16_F32_RULE(kEncVop3_1),
    RJ_VOP3_SINGLE_SRC_RULE(kEncVop3_1),
    {kEncVop3_2, kOpVMulU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mul_u64_vop3), nullptr, nullptr},
    {kEncVop3_2, kOpVAddNcU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_add_nc_u64_vop3), nullptr, nullptr},
    {kEncVop3_2, kOpVSubNcU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_sub_nc_u64_vop3), nullptr, nullptr},
    {kEncVop3_2, kOpVMovB64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mov_b64_vop3), nullptr, nullptr},
    RJ_VOP3_CVT_F32_BF16_RULE(kEncVop3_2),
    {kEncVop3_2, kOpVMadU32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_u32_vop3), nullptr, nullptr},
    {kEncVop3_2, kOpVLshlAddU32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_add_u32_vop3), nullptr, nullptr},
    {kEncVop3_2, kOpVLshlAddU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_add_u64_vop3), nullptr, nullptr},
    {kEncVop3_2, kOpVLshlOrB32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_or_b32_vop3), nullptr, nullptr},
    {kEncVop3_2, kOpVMadNcU64U32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_nc_u64_u32_vop3), nullptr, nullptr},
    {kEncVop3_2, kOpVMaxU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_max_u64_vop3), nullptr, nullptr},
    {kEncVop3_2, kOpVMinI64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_min_i64_vop3), nullptr, nullptr},
    RJ_VOP3_CVT_PK_BF16_F32_RULE(kEncVop3_2),
    RJ_VOP3_CVT_PK_F16_F32_RULE(kEncVop3_2),
    RJ_VOP3_SINGLE_SRC_RULE(kEncVop3_2),
    {kEncVop3_3, kOpVMulU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mul_u64_vop3), nullptr, nullptr},
    {kEncVop3_3, kOpVAddNcU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_add_nc_u64_vop3), nullptr, nullptr},
    {kEncVop3_3, kOpVSubNcU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_sub_nc_u64_vop3), nullptr, nullptr},
    {kEncVop3_3, kOpVMovB64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mov_b64_vop3), nullptr, nullptr},
    RJ_VOP3_CVT_F32_BF16_RULE(kEncVop3_3),
    {kEncVop3_3, kOpVMadU32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_u32_vop3), nullptr, nullptr},
    {kEncVop3_3, kOpVLshlAddU32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_add_u32_vop3), nullptr, nullptr},
    {kEncVop3_3, kOpVLshlAddU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_add_u64_vop3), nullptr, nullptr},
    {kEncVop3_3, kOpVLshlOrB32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_or_b32_vop3), nullptr, nullptr},
    {kEncVop3_3, kOpVMadNcU64U32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_nc_u64_u32_vop3), nullptr, nullptr},
    {kEncVop3_3, kOpVMaxU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_max_u64_vop3), nullptr, nullptr},
    {kEncVop3_3, kOpVMinI64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_min_i64_vop3), nullptr, nullptr},
    RJ_VOP3_CVT_PK_BF16_F32_RULE(kEncVop3_3),
    RJ_VOP3_CVT_PK_F16_F32_RULE(kEncVop3_3),
    RJ_VOP3_SINGLE_SRC_RULE(kEncVop3_3),
    {kEncVop3_4, kOpVMulU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mul_u64_vop3), nullptr, nullptr},
    {kEncVop3_4, kOpVAddNcU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_add_nc_u64_vop3), nullptr, nullptr},
    {kEncVop3_4, kOpVSubNcU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_sub_nc_u64_vop3), nullptr, nullptr},
    {kEncVop3_4, kOpVMovB64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mov_b64_vop3), nullptr, nullptr},
    RJ_VOP3_CVT_F32_BF16_RULE(kEncVop3_4),
    {kEncVop3_4, kOpVBitop3B16Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_bitop3_b16_vop3), nullptr, nullptr},
    {kEncVop3_4, kOpVBitop3B32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_bitop3_b32_vop3), nullptr, nullptr},
    {kEncVop3_4, kOpVMadU32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_u32_vop3), nullptr, nullptr},
    {kEncVop3_4, kOpVLshlAddU32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_add_u32_vop3), nullptr, nullptr},
    {kEncVop3_4, kOpVLshlAddU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_add_u64_vop3), nullptr, nullptr},
    {kEncVop3_4, kOpVLshlOrB32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_or_b32_vop3), nullptr, nullptr},
    {kEncVop3_4, kOpVMadNcU64U32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_nc_u64_u32_vop3), nullptr, nullptr},
    {kEncVop3_4, kOpVMaxU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_max_u64_vop3), nullptr, nullptr},
    {kEncVop3_4, kOpVMinI64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_min_i64_vop3), nullptr, nullptr},
    RJ_VOP3_CVT_PK_BF16_F32_RULE(kEncVop3_4),
    RJ_VOP3_CVT_PK_F16_F32_RULE(kEncVop3_4),
    RJ_VOP3_SINGLE_SRC_RULE(kEncVop3_4),
    {kEncVop3_5, kOpVMulU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mul_u64_vop3), nullptr, nullptr},
    {kEncVop3_5, kOpVAddNcU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_add_nc_u64_vop3), nullptr, nullptr},
    {kEncVop3_5, kOpVSubNcU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_sub_nc_u64_vop3), nullptr, nullptr},
    {kEncVop3_5, kOpVMovB64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mov_b64_vop3), nullptr, nullptr},
    RJ_VOP3_CVT_F32_BF16_RULE(kEncVop3_5),
    {kEncVop3_5, kOpVMadU32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_u32_vop3), nullptr, nullptr},
    {kEncVop3_5, kOpVLshlAddU32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_add_u32_vop3), nullptr, nullptr},
    {kEncVop3_5, kOpVLshlAddU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_add_u64_vop3), nullptr, nullptr},
    {kEncVop3_5, kOpVLshlOrB32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_or_b32_vop3), nullptr, nullptr},
    {kEncVop3_5, kOpVMadNcU64U32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_nc_u64_u32_vop3), nullptr, nullptr},
    {kEncVop3_5, kOpVMaxU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_max_u64_vop3), nullptr, nullptr},
    {kEncVop3_5, kOpVMinI64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_min_i64_vop3), nullptr, nullptr},
    RJ_VOP3_CVT_PK_BF16_F32_RULE(kEncVop3_5),
    RJ_VOP3_CVT_PK_F16_F32_RULE(kEncVop3_5),
    RJ_VOP3_SINGLE_SRC_RULE(kEncVop3_5),
    {kEncVop3_6, kOpVMulU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mul_u64_vop3), nullptr, nullptr},
    {kEncVop3_6, kOpVAddNcU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_add_nc_u64_vop3), nullptr, nullptr},
    {kEncVop3_6, kOpVSubNcU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_sub_nc_u64_vop3), nullptr, nullptr},
    {kEncVop3_6, kOpVMovB64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mov_b64_vop3), nullptr, nullptr},
    RJ_VOP3_CVT_F32_BF16_RULE(kEncVop3_6),
    {kEncVop3_6, kOpVMadU32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_u32_vop3), nullptr, nullptr},
    {kEncVop3_6, kOpVLshlAddU32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_add_u32_vop3), nullptr, nullptr},
    {kEncVop3_6, kOpVLshlAddU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_add_u64_vop3), nullptr, nullptr},
    {kEncVop3_6, kOpVLshlOrB32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_or_b32_vop3), nullptr, nullptr},
    {kEncVop3_6, kOpVMadNcU64U32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_nc_u64_u32_vop3), nullptr, nullptr},
    {kEncVop3_6, kOpVMaxU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_max_u64_vop3), nullptr, nullptr},
    {kEncVop3_6, kOpVMinI64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_min_i64_vop3), nullptr, nullptr},
    {kEncVop3_6, kOpVLshlrevB16Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshlrev_b16_vop3), nullptr, nullptr},
    {kEncVop3_6, kOpVOrB16Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_or_b16_vop3), nullptr, nullptr},
    RJ_VOP3_CVT_PK_BF16_F32_RULE(kEncVop3_6),
    RJ_VOP3_CVT_PK_F16_F32_RULE(kEncVop3_6),
    RJ_VOP3_SINGLE_SRC_RULE(kEncVop3_6),
    {kEncVop3_7, kOpVMulU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mul_u64_vop3), nullptr, nullptr},
    {kEncVop3_7, kOpVAddNcU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_add_nc_u64_vop3), nullptr, nullptr},
    {kEncVop3_7, kOpVSubNcU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_sub_nc_u64_vop3), nullptr, nullptr},
    {kEncVop3_7, kOpVMovB64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mov_b64_vop3), nullptr, nullptr},
    RJ_VOP3_CVT_F32_BF16_RULE(kEncVop3_7),
    {kEncVop3_7, kOpVMadU32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_u32_vop3), nullptr, nullptr},
    {kEncVop3_7, kOpVLshlAddU32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_add_u32_vop3), nullptr, nullptr},
    {kEncVop3_7, kOpVLshlAddU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_add_u64_vop3), nullptr, nullptr},
    {kEncVop3_7, kOpVLshlOrB32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_lshl_or_b32_vop3), nullptr, nullptr},
    {kEncVop3_7, kOpVMadNcU64U32Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_mad_nc_u64_u32_vop3), nullptr, nullptr},
    {kEncVop3_7, kOpVMaxU64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_max_u64_vop3), nullptr, nullptr},
    {kEncVop3_7, kOpVMinI64Vop3, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_v_min_i64_vop3), nullptr, nullptr},
    RJ_VOP3_CVT_PK_BF16_F32_RULE(kEncVop3_7),
    RJ_VOP3_CVT_PK_F16_F32_RULE(kEncVop3_7),
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
    {kEncVglobal, kOpGlobalLoadB32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_scaled_vglobal_b32), nullptr, nullptr},
    {kEncVglobal, kOpGlobalLoadB64, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_scaled_vglobal_b32), nullptr, nullptr},
    {kEncVglobal, kOpGlobalLoadB128, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_scaled_vglobal_b32), nullptr, nullptr},
    {kEncVglobal, kOpGlobalStoreB32, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_scaled_vglobal_b32), nullptr, nullptr},
    {kEncVglobal, kOpGlobalLoadTr16B128, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_native_global_transpose_load), nullptr, nullptr},
    {kEncVglobal, kOpGlobalLoadTr8B64, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_native_global_transpose_load), nullptr, nullptr},
    {kEncVglobal, kOpGlobalLoadTr4B64, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_global_load_tr4_b64), nullptr, nullptr},
    {kEncVglobal, kOpGlobalLoadTr6B96, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_global_load_tr6_b96), nullptr, nullptr},
    {kEncVglobal1, kOpGlobalLoadTr16B128, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_native_global_transpose_load), nullptr, nullptr},
    {kEncVglobal1, kOpGlobalLoadTr8B64, RuleAction::Expand, 0, 0, nullptr,
     RJ_GFX1250_EXPAND(expand_native_global_transpose_load), nullptr, nullptr},
};

#undef RJ_VOPD3_RULE
#undef RJ_VOPD_RULE
#undef RJ_VOP3_SINGLE_SRC_RULE
#undef RJ_VOP3_CVT_PK_BF16_F32_RULE
#undef RJ_VOP3_CVT_PK_F16_F32_RULE
#undef RJ_GFX1250_EXPAND

} // namespace

std::span<const TranslationRule> semantic_expand_rules_gfx1250_to_rdna4() {
  return std::span<const TranslationRule>(kExpandRules_gfx1250_to_rdna4);
}

} // namespace rocjitsu
