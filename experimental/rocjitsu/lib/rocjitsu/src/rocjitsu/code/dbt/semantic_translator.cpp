// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file semantic_translator.cpp
/// @brief Semantic translator implementation and per-pair rule tables.

#include "rocjitsu/code/dbt/semantic_translator.h"

#include "rocjitsu/analysis/def_use_chain.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/dbt/hazard_tracker.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/machine_insts.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstring>
#include <optional>
#include <string_view>
#include <utility>

namespace rocjitsu {

// --- Waitcnt decode/encode ---

namespace {

[[nodiscard]] uint32_t make_gfx12_sopp(uint8_t op, uint16_t simm16) {
  rdna4::SoppMachineInst s{};
  s.encoding = 0x17F;
  s.op = op;
  s.simm16 = simm16;
  return std::bit_cast<uint32_t>(s);
}

[[nodiscard]] uint32_t make_gfx12_sop1(uint8_t op, uint8_t ssrc0) {
  rdna4::Sop1MachineInst s{};
  s.encoding = 0x17D;
  s.op = op;
  s.ssrc0 = ssrc0;
  return std::bit_cast<uint32_t>(s);
}

constexpr uint8_t kOpWaitLoadcnt = 64;
constexpr uint8_t kOpWaitStorecntDscnt = 73;
constexpr uint8_t kOpWaitKmcnt = 71;
constexpr uint8_t kOpWaitExpcnt = 68;
constexpr uint16_t kTtmp9Encoding = 117; // RDNA4 OPR_*_TTMP_MIN + 9.

[[nodiscard]] bool is_sgpr_ref(const Operand *op, uint16_t sgpr) {
  if (op == nullptr)
    return false;
  auto reg = op->to_register_ref(64);
  return reg && reg->cls == RegClass::SGPR && reg->index <= sgpr && sgpr < reg->index + reg->width;
}

[[nodiscard]] bool replace_field(uint32_t &word, uint32_t shift, uint32_t width, uint16_t expected,
                                 uint16_t replacement) {
  const uint32_t mask = ((1u << width) - 1u) << shift;
  const uint32_t old_value = (word & mask) >> shift;
  if (old_value != expected)
    return false;
  word = (word & ~mask) | (static_cast<uint32_t>(replacement) << shift);
  return true;
}

[[nodiscard]] bool patch_workgroup_src_operand(const Instruction &inst, uint32_t &w0, uint32_t &w1,
                                               uint16_t old_sgpr, uint16_t ttmp_encoding) {
  bool changed = false;

  // This helper deliberately only patches source register fields in encodings
  // with known layouts. Def/use chooses the semantic source operands; the raw
  // bit edits below are just the current replacement mechanism until generated
  // operand-field rewrite metadata exists.
  if (dynamic_cast<const cdna4::Sop2 *>(&inst) != nullptr) { // ssrc0[7:0], ssrc1[15:8].
    if (is_sgpr_ref(inst.src_operand(0), old_sgpr))
      changed |= replace_field(w0, 0, 8, old_sgpr, ttmp_encoding);
    if (is_sgpr_ref(inst.src_operand(1), old_sgpr))
      changed |= replace_field(w0, 8, 8, old_sgpr, ttmp_encoding);
  } else if (dynamic_cast<const cdna4::Sop1 *>(&inst) != nullptr) { // ssrc0[7:0].
    if (is_sgpr_ref(inst.src_operand(0), old_sgpr))
      changed |= replace_field(w0, 0, 8, old_sgpr, ttmp_encoding);
  } else if (dynamic_cast<const cdna4::Sopc *>(&inst) != nullptr) { // ssrc0[7:0], ssrc1[15:8].
    if (is_sgpr_ref(inst.src_operand(0), old_sgpr))
      changed |= replace_field(w0, 0, 8, old_sgpr, ttmp_encoding);
    if (is_sgpr_ref(inst.src_operand(1), old_sgpr))
      changed |= replace_field(w0, 8, 8, old_sgpr, ttmp_encoding);
  } else if (dynamic_cast<const cdna4::Vop2 *>(&inst) != nullptr ||
             dynamic_cast<const cdna4::Vopc *>(&inst) != nullptr ||
             dynamic_cast<const cdna4::Vop1 *>(&inst) != nullptr) {
    // Scalar-capable src0 is in word0[8:0].
    if (is_sgpr_ref(inst.src_operand(0), old_sgpr))
      changed |= replace_field(w0, 0, 9, old_sgpr, ttmp_encoding);
  } else if (dynamic_cast<const cdna4::Vop3 *>(&inst) != nullptr ||
             dynamic_cast<const cdna4::Vop3p *>(&inst) != nullptr) {
    // src0/src1/src2 are in word1[8:0], [17:9], [26:18].
    for (int i = 0; i < inst.num_src_operands(); ++i) {
      if (is_sgpr_ref(inst.src_operand(i), old_sgpr))
        changed |= replace_field(w1, static_cast<uint32_t>(i * 9), 9, old_sgpr, ttmp_encoding);
    }
  }

  return changed;
}

} // namespace

WaitcntValues decode_waitcnt_gfx9(uint16_t simm16) {
  WaitcntValues v;
  v.vmcnt = (simm16 & 0xF) | (static_cast<uint8_t>((simm16 >> 14) & 0x3) << 4);
  v.expcnt = static_cast<uint8_t>((simm16 >> 4) & 0x7);
  v.lgkmcnt = static_cast<uint8_t>((simm16 >> 8) & 0x0F);
  return v;
}

std::vector<uint32_t> encode_waitcnt_gfx12(const WaitcntValues &vals) {
  std::vector<uint32_t> words;

  const bool need_loadcnt = (vals.vmcnt != 0x3F);
  const bool need_storecnt = (vals.vmcnt != 0x3F);
  const bool need_kmcnt = (vals.lgkmcnt != 0x0F);
  const bool need_dscnt = (vals.lgkmcnt != 0x0F);
  const bool need_expcnt = (vals.expcnt != 0x07);

  if (need_loadcnt)
    words.push_back(make_gfx12_sopp(kOpWaitLoadcnt, std::min<uint16_t>(vals.vmcnt, 63)));

  if (need_storecnt || need_dscnt) {
    // GFX12 packs STORECNT in SIMM16[13:8] and DSCNT in SIMM16[5:0].
    // Use the relaxed max value for the counter that the original GFX9 wait
    // did not constrain; otherwise a VM-only wait would accidentally wait DS,
    // or an LGKM-only wait would accidentally wait vector stores.
    const uint8_t sc = need_storecnt ? std::min<uint8_t>(vals.vmcnt, 63) : 63;
    const uint8_t dc = need_dscnt ? std::min<uint8_t>(vals.lgkmcnt, 63) : 63;
    words.push_back(make_gfx12_sopp(kOpWaitStorecntDscnt, (sc << 8) | dc));
  }

  if (need_kmcnt)
    words.push_back(make_gfx12_sopp(kOpWaitKmcnt, std::min<uint16_t>(vals.lgkmcnt, 15)));

  if (need_expcnt)
    words.push_back(make_gfx12_sopp(kOpWaitExpcnt, vals.expcnt));

  if (words.empty())
    words.push_back(build_s_nop());

  return words;
}

// --- Instruction lowering (Action::Expand) ---

namespace {

std::vector<uint32_t> lower_v_lshl_add_u64(const Instruction &inst,
                                           [[maybe_unused]] rj_code_arch_t host_arch,
                                           uint64_t offset, const LivenessAnalysis &liveness) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() < 8)
    return {};

  cdna4::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  const uint16_t vdst = src.vdst;
  const uint16_t src0 = src.src0;
  const uint16_t src1 = src.src1;
  const uint16_t src2 = src.src2;

  auto src_pair_hi = [](uint16_t low) -> std::optional<uint16_t> {
    if (low <= 126 || (low >= 256 && low < 511))
      return static_cast<uint16_t>(low + 1);
    if (low >= 128 && low <= 208)
      return 128;
    return std::nullopt;
  };
  auto inline_u32 = [](uint16_t op) -> std::optional<uint32_t> {
    if (op >= 128 && op <= 192)
      return static_cast<uint32_t>(op - 128);
    if (op >= 193 && op <= 208)
      return static_cast<uint32_t>(-static_cast<int32_t>(op - 192));
    return std::nullopt;
  };

  auto src0_hi = src_pair_hi(src0);
  auto src2_hi = src_pair_hi(src2);
  auto shift = inline_u32(src1);
  if (!src0_hi || !src2_hi || !shift || *shift >= 64)
    return {};

  auto carry_sgpr = liveness.find_free_sgpr_pair(offset);
  if (!carry_sgpr)
    return {};

  constexpr uint16_t kOpLshlrevB64 = 0x11F;
  constexpr uint16_t kOpAddCoCiU32 = 288;
  constexpr uint16_t kOpAddCoU32 = 768;
  auto build_vop3_sdst = [](uint16_t op, uint8_t vdst, uint8_t sdst, uint16_t src0, uint16_t src1,
                            uint16_t src2 = 0) {
    rdna4::Vop3SdstEncMachineInst out{};
    out.encoding = 0x35;
    out.op = op;
    out.vdst = vdst;
    out.sdst = sdst;
    out.src0 = src0;
    out.src1 = src1;
    out.src2 = src2;
    return std::bit_cast<uint64_t>(out);
  };
  auto build_vop3_local = [](uint16_t op, uint8_t vdst, uint16_t src0, uint16_t src1,
                             uint16_t src2 = 0) {
    rdna4::Vop3MachineInst out{};
    out.encoding = 0x35;
    out.op = op;
    out.vdst = vdst;
    out.src0 = src0;
    out.src1 = src1;
    out.src2 = src2;
    return std::bit_cast<uint64_t>(out);
  };

  auto low_word = [](uint64_t inst) { return static_cast<uint32_t>(inst); };
  auto high_word = [](uint64_t inst) { return static_cast<uint32_t>(inst >> 32); };

  uint16_t add_src0_lo = src0;
  uint16_t add_src0_hi = *src0_hi;
  std::vector<uint32_t> words;
  HazardTracker hz;

  if (*shift != 0) {
    auto shifted = liveness.find_free_run(offset, 2);
    if (!shifted)
      return {};

    const uint64_t shift_inst =
        build_vop3_local(kOpLshlrevB64, static_cast<uint8_t>(*shifted), src1, src0);
    hz.emit2(words, low_word(shift_inst), high_word(shift_inst), HazardTracker::Pipeline::VALU);
    add_src0_lo = static_cast<uint16_t>(256u + *shifted);
    add_src0_hi = static_cast<uint16_t>(add_src0_lo + 1);
  }

  // v_add_co_u32 vdst_lo, carry_sgpr, src0_lo, src2_lo
  const uint64_t add_lo = build_vop3_sdst(kOpAddCoU32, static_cast<uint8_t>(vdst),
                                          static_cast<uint8_t>(*carry_sgpr), add_src0_lo, src2);
  hz.emit2(words, low_word(add_lo), high_word(add_lo), HazardTracker::Pipeline::VALU);

  // The carry-out is a VALU-produced scalar destination consumed by the next
  // VALU as a scalar source. Use a private SGPR carry instead of VCC: the guest
  // v_lshl_add_u64 does not clobber VCC, and compare masks are often live
  // across address calculation in loop/control-flow code.
  constexpr uint16_t kWaitVaSdst = 0xF1FF;
  hz.emit_raw(words, pack_sopp(8, kWaitVaSdst));

  // v_add_co_ci_u32 vdst_hi, carry_sgpr, src0_hi, src2_hi, carry_sgpr
  const uint64_t add_hi =
      build_vop3_sdst(kOpAddCoCiU32, static_cast<uint8_t>(vdst + 1),
                      static_cast<uint8_t>(*carry_sgpr), add_src0_hi, *src2_hi, *carry_sgpr);
  hz.emit2(words, low_word(add_hi), high_word(add_hi), HazardTracker::Pipeline::VALU);

  return words;
}

/// @brief Build a VOP1 v_mov_b32 instruction for RDNA4.
/// @param vdst  Destination VGPR index (0-255).
/// @param src0  Source operand (9-bit encoding: 256-511 for VGPR, 0-255 for SGPR/const).
[[nodiscard]] constexpr uint32_t build_v_mov_b32(uint8_t vdst, uint16_t src0) {
  return (0x3Fu << 25) | (static_cast<uint32_t>(vdst) << 17) | (1u << 9) | (src0 & 0x1FF);
}

/// @brief Lower v_accvgpr_read_b32 to v_mov_b32 or NOP on RDNA4.
/// @details acc[N] = v[N+256] on the unified file. If dst aliases the unified
/// src, emit NOP. Otherwise emit v_mov_b32.
std::vector<uint32_t> lower_accvgpr_read(const Instruction &inst,
                                         [[maybe_unused]] rj_code_arch_t host_arch,
                                         uint16_t accum_vgpr_base) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() < 8)
    return {};

  cdna4::Vop3pMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));

  const uint16_t dst_vgpr = src.vdst;
  const uint16_t src_acc = src.src0;
  assert(src_acc >= 256 && src_acc <= 511);
  const uint16_t src_unified =
      static_cast<uint16_t>((accum_vgpr_base ? accum_vgpr_base : 0) + src_acc - 256);

  if (dst_vgpr == src_unified)
    return {build_s_nop()};

  return {build_v_mov_b32(static_cast<uint8_t>(dst_vgpr), 256 + src_unified)};
}

// ---------------------------------------------------------------------------
// RDNA4 instruction builders
// ---------------------------------------------------------------------------

/// @brief Build VOP3P instruction word pair (packed math: WMMA, dot products).
[[nodiscard]] constexpr std::pair<uint32_t, uint32_t>
build_vop3p(uint8_t op, uint8_t vdst, uint16_t src0, uint16_t src1, uint16_t src2) {
  const uint32_t w0 = static_cast<uint32_t>(vdst) | (1u << 14) |
                      (static_cast<uint32_t>(op & 0x7F) << 16) | (0xCCu << 24);
  const uint32_t w1 = (src0 & 0x1FF) | ((src1 & 0x1FF) << 9) | ((src2 & 0x1FF) << 18) | (3u << 27);
  return {w0, w1};
}

/// @brief Build VOP3 instruction word pair (non-packed: mbcnt, permlane, add_co).
[[nodiscard]] constexpr std::pair<uint32_t, uint32_t>
build_vop3(uint16_t op, uint8_t vdst, uint16_t src0, uint16_t src1 = 0, uint16_t src2 = 0) {
  const uint32_t w0 = (vdst & 0xFFu) | ((op & 0x3FFu) << 16) | (0x35u << 26);
  const uint32_t w1 = (src0 & 0x1FFu) | ((src1 & 0x1FFu) << 9) | ((src2 & 0x1FFu) << 18);
  return {w0, w1};
}

/// @brief Build VOP2 instruction word (xor, lshlrev, add_nc, etc.).
[[nodiscard]] constexpr uint32_t build_vop2(uint8_t op, uint8_t vdst, uint16_t src0,
                                            uint8_t vsrc1) {
  return (src0 & 0x1FFu) | ((vsrc1 & 0xFFu) << 9) | ((vdst & 0xFFu) << 17) | ((op & 0x3Fu) << 25);
}

/// @brief Build s_mov_b64 sdst, ssrc0.
[[nodiscard]] constexpr uint32_t build_s_mov_b64(uint8_t sdst, uint16_t ssrc0) {
  rdna4::Sop1MachineInst s{};
  s.encoding = 0x17D;
  s.op = 1;
  s.sdst = sdst & 0x7F;
  s.ssrc0 = ssrc0 & 0xFF;
  return std::bit_cast<uint32_t>(s);
}

/// @brief Build s_mov_b32 sdst, literal (two-word instruction).
[[nodiscard]] constexpr std::pair<uint32_t, uint32_t> build_s_mov_b32_lit(uint8_t sdst,
                                                                          uint32_t literal) {
  rdna4::Sop1MachineInst s{};
  s.encoding = 0x17D;
  s.op = 0;
  s.sdst = sdst & 0x7F;
  s.ssrc0 = 0xFF;
  return {std::bit_cast<uint32_t>(s), literal};
}

/// @brief Build ds_bpermute_b32 vdst, vaddr, vdata.
[[nodiscard]] constexpr std::pair<uint32_t, uint32_t> build_ds_bpermute(uint8_t vdst, uint8_t vaddr,
                                                                        uint8_t vdata) {
  constexpr uint32_t kDsW0 = (0xB3u << 18) | (0x36u << 26);
  return {kDsW0, static_cast<uint32_t>(vaddr) | (static_cast<uint32_t>(vdata) << 8) |
                     (static_cast<uint32_t>(vdst) << 24)};
}

// ---------------------------------------------------------------------------
// GFX12 SOPP opcodes and s_delay_alu constants
// ---------------------------------------------------------------------------

constexpr uint8_t kSoppWaitIdle = 0x0A;
constexpr uint8_t kOpWaitDscnt = 70;
constexpr uint8_t kOpBarrierSignal = 78;
constexpr uint8_t kOpBarrierWait = 20;

// ---------------------------------------------------------------------------
// RDNA4 operand encoding constants
// ---------------------------------------------------------------------------

constexpr uint8_t kExecLo = 126;
constexpr uint16_t kInlineConst0 = 128;
constexpr uint16_t kInlineConst2 = 130;
constexpr uint16_t kInlineConstNeg1 = 193;

// VOP3 opcodes (GFX12)
constexpr uint16_t kOpMbcntLo = 0x31F;
constexpr uint16_t kOpMbcntHi = 0x320;

// VOP2 opcodes (GFX12)
constexpr uint8_t kOpAndB32 = 27;
constexpr uint8_t kOpLshlrevB32 = 24;
constexpr uint8_t kOpXorB32 = 29;

// VOP3P opcodes (GFX12)
constexpr uint8_t kOpWmmaF32_16x16x16_F16 = 64;

// VGLOBAL opcodes (GFX12)
constexpr uint8_t kOpGlobalStoreB128 = 29;

/// @brief Lower v_mfma_f32_16x16x16_f16 to v_wmma_f32_16x16x16_f16 on RDNA4.
///
/// WMMA Wave64 writes all 64 lanes but swaps rows 4-7 and 8-11 vs MFMA
/// layout (lanes 16-31 ↔ lanes 32-47). Fix via ds_bpermute with a
/// pre-computed address VGPR that encodes identity for lanes 0-15,48-63
/// and XOR-48 for lanes 16-47.
std::vector<uint32_t> lower_mfma_f32_16x16x16_f16(const Instruction &inst,
                                                  [[maybe_unused]] rj_code_arch_t host_arch,
                                                  uint64_t offset, const LivenessAnalysis &liveness,
                                                  uint16_t accum_vgpr_base,
                                                  const LaneLayout *guest_layout,
                                                  const LaneLayout *host_layout) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() < 8)
    return {};

  cdna4::Vop3pMfmaMachineInst mfma{};
  std::memcpy(&mfma, raw, sizeof(mfma));

  const uint16_t vdst = mfma.vdst;
  const uint16_t src0 = mfma.src0;
  const uint16_t src1 = mfma.src1;
  const uint16_t src2 = mfma.src2;
  const uint16_t wmma_vdst = accum_vgpr_base ? static_cast<uint16_t>(accum_vgpr_base + vdst) : vdst;

  if ((src2 >= 512 && src2 < 768) || src2 > 1023)
    return {};
  const bool has_vgpr_accumulator = src2 >= 256;
  const uint16_t src2_index =
      has_vgpr_accumulator
          ? (src2 >= 768 ? static_cast<uint16_t>(src2 - 768) : static_cast<uint16_t>(src2 - 256))
          : 0;
  const uint16_t src2_vgpr =
      has_vgpr_accumulator
          ? static_cast<uint16_t>((accum_vgpr_base ? accum_vgpr_base : 0) + src2_index)
          : 0;

  assert(src0 >= 256 && src1 >= 256 && "MFMA VGPR sources expected");

  // Prefer non-ABI SGPRs for injected lowering temporaries. Low SGPRs commonly
  // carry kernarg/user pointers across matrix instructions, and generated
  // operand metadata for some memory forms can under-report scalar address
  // bases. Liveness still decides whether the chosen high SGPRs are dead.
  constexpr uint16_t kInjectedSgprSearchStart = 8;
  auto exec_save_opt = liveness.find_free_sgpr_pair(offset, kInjectedSgprSearchStart);
  if (!exec_save_opt)
    return {};
  const uint8_t kExecSave = static_cast<uint8_t>(*exec_save_opt);

  auto tmp_sgpr_opt = liveness.find_free_sgpr(
      offset, std::max<uint16_t>(static_cast<uint16_t>(kExecSave + 2), kInjectedSgprSearchStart));
  if (!tmp_sgpr_opt)
    return {};
  const uint8_t kTmpSgpr = static_cast<uint8_t>(*tmp_sgpr_opt);

  const uint16_t temp_search_start =
      has_vgpr_accumulator ? static_cast<uint16_t>(std::max<uint16_t>(wmma_vdst + 8, src2_vgpr + 4))
                           : static_cast<uint16_t>(wmma_vdst + 8);
  auto vaddr_reg = liveness.find_free_run(offset, 1, temp_search_start);
  if (!vaddr_reg)
    return {};
  const uint8_t vaddr = static_cast<uint8_t>(*vaddr_reg);

  std::optional<uint16_t> acc_tmp_reg;
  if (has_vgpr_accumulator) {
    const uint16_t acc_search_start = static_cast<uint16_t>(vaddr + 1);
    acc_tmp_reg = liveness.find_free_run(offset, 4, acc_search_start);
    if (!acc_tmp_reg)
      return {};
  }

  std::vector<uint32_t> words;

  words.push_back(make_gfx12_sopp(kOpWaitLoadcnt, 0));
  words.push_back(build_s_mov_b64(kExecSave, kExecLo));

  // Compute bpermute byte address: vaddr = lane_id * 4.
  // HazardTracker auto-inserts s_delay_alu between dependent instructions.
  using P = HazardTracker::Pipeline;
  HazardTracker hz;

  {
    auto [w0, w1] = build_vop3(kOpMbcntLo, vaddr, kInlineConstNeg1, kInlineConst0);
    hz.emit2(words, w0, w1, P::VALU);
  }
  {
    auto [w0, w1] = build_vop3(kOpMbcntHi, vaddr, kInlineConstNeg1, 256 + vaddr);
    hz.emit2(words, w0, w1, P::VALU);
  }
  hz.emit(words, build_vop2(kOpLshlrevB32, vaddr, kInlineConst2, vaddr), P::VALU);

  // XOR byte address at the lanes that differ between guest and host layout.
  auto perm = (guest_layout && host_layout) ? compute_lane_permutation(*guest_layout, *host_layout)
                                            : LanePermutation{192, 16, 48};
  if (perm.xor_byte_mask != 0) {
    auto [sw0, sw1] = build_s_mov_b32_lit(kTmpSgpr, perm.xor_byte_mask);
    hz.emit2(words, sw0, sw1, P::SALU);
    uint64_t exec_mask = 0;
    for (uint8_t lane = perm.range_start; lane < perm.range_end; ++lane)
      exec_mask |= (1ULL << lane);
    auto [el, lit] = build_s_mov_b32_lit(kExecLo, static_cast<uint32_t>(exec_mask));
    hz.emit2(words, el, lit, P::None); // EXEC writes excluded from hazard tracking
    auto [eh, lith] = build_s_mov_b32_lit(kExecLo + 1, static_cast<uint32_t>(exec_mask >> 32));
    hz.emit2(words, eh, lith, P::None);
    hz.emit(words, build_vop2(kOpXorB32, vaddr, kTmpSgpr, vaddr), P::VALU);
  }

  words.push_back(build_s_mov_b64(kExecLo, kExecSave));

  uint16_t wmma_src2 = src2;
  if (has_vgpr_accumulator) {
    // Each translated MFMA preserves the CDNA/MFMA architectural layout at its
    // instruction boundary. WMMA expects its accumulator operand in WMMA lane
    // layout, so remap a chained accumulator source into temporaries before
    // issuing this instruction-local WMMA lowering.
    const uint8_t acc_tmp = static_cast<uint8_t>(*acc_tmp_reg);
    for (int r = 0; r < 4; ++r) {
      auto [w0, w1] = build_ds_bpermute(acc_tmp + r, vaddr, src2_vgpr + r);
      words.push_back(w0);
      words.push_back(w1);
    }
    words.push_back(pack_sopp(kOpWaitDscnt, 0));
    wmma_src2 = 256 + acc_tmp;
  }

  // WMMA: single pass, writes all 64 lanes in WMMA layout.
  {
    auto [w0, w1] = build_vop3p(kOpWmmaF32_16x16x16_F16, static_cast<uint8_t>(wmma_vdst), src0,
                                src1, wmma_src2);
    words.push_back(w0);
    words.push_back(w1);
  }

  // Drain pipelines so ds_bpermute can read WMMA output from VGPR file.
  words.push_back(pack_sopp(kSoppWaitIdle, 0));

  // ds_bpermute: remap WMMA output lanes 16-31 ↔ 32-47 to match MFMA layout.
  for (int r = 0; r < 4; ++r) {
    auto [w0, w1] = build_ds_bpermute(wmma_vdst + r, vaddr, wmma_vdst + r);
    words.push_back(w0);
    words.push_back(w1);
  }
  words.push_back(pack_sopp(kOpWaitDscnt, 0));

  words.push_back(build_s_mov_b64(kExecLo, kExecSave));

  return words;
}

/// @brief Lower the Triton-emitted v_bitop3_b32 LUT 0x6c to RDNA4 VALU.
/// @details LUT 0x6c is src1 ^ (src0 & src2). If vdst does not alias src1,
/// clobber vdst with the AND result first. If vdst aliases src1, preserve src1
/// by computing the AND into a liveness-selected temporary VGPR.
std::vector<uint32_t> lower_v_bitop3_b32(const Instruction &inst, uint64_t offset,
                                         const LivenessAnalysis &liveness) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() < 8)
    return {};

  cdna4::Vop3MachineInst bitop{};
  std::memcpy(&bitop, raw, sizeof(bitop));

  const uint8_t vdst = static_cast<uint8_t>(bitop.vdst);
  const uint16_t truth_table =
      static_cast<uint16_t>((bitop.omod << 6) | (bitop.abs << 3) | bitop.neg);
  if (truth_table != 0x6Cu)
    return {};

  if (bitop.src0 < 256u || bitop.src0 >= 512u || bitop.src1 >= 512u || bitop.src2 >= 512u)
    return {};

  using P = HazardTracker::Pipeline;
  HazardTracker hz;
  std::vector<uint32_t> words;
  const uint8_t src0_vgpr = static_cast<uint8_t>(bitop.src0 - 256u);
  if (bitop.src1 == 256u + vdst) {
    auto tmp = liveness.find_free_run(offset, 1);
    if (!tmp)
      return {};
    const auto tmp_vgpr = static_cast<uint8_t>(*tmp);
    hz.emit(words, build_vop2(kOpAndB32, tmp_vgpr, bitop.src2, src0_vgpr), P::VALU);
    hz.emit(words, build_vop2(kOpXorB32, vdst, 256u + tmp_vgpr, vdst), P::VALU);
  } else {
    hz.emit(words, build_vop2(kOpAndB32, vdst, bitop.src2, src0_vgpr), P::VALU);
    hz.emit(words, build_vop2(kOpXorB32, vdst, bitop.src1, vdst), P::VALU);
  }
  return words;
}

/// @brief Lower a CDNA4 global_store_dwordx4 whose data source is AccVGPR.
/// @details CDNA4 encodes the data VGPR index plus an acc bit. RDNA4 has only
/// the unified VGPR field, so data must be shifted by the kernel's accumulator
/// offset before encoding the VGLOBAL store.
std::vector<uint32_t> lower_flat_glbl_acc_store_b128(const Instruction &inst,
                                                     uint16_t accum_vgpr_base) {
  if (accum_vgpr_base == 0)
    return {};

  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() < 8)
    return {};

  cdna4::FlatGlblMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.acc == 0)
    return {};

  rdna4::VglobalMachineInst dst{};
  dst.encoding = 0xEE;
  dst.op = kOpGlobalStoreB128;
  dst.ioffset = src.offset;
  dst.sve = src.sve;
  dst.vaddr = src.addr;
  dst.vsrc = static_cast<uint8_t>(accum_vgpr_base + src.data);
  dst.saddr = src.saddr == 0x7F ? 0x7C : src.saddr;
  dst.vdst = src.vdst;
  dst.nv = 0;

  std::vector<uint32_t> words(3);
  std::memcpy(words.data(), &dst, sizeof(dst));
  return words;
}

/// @brief Lower v_accvgpr_write_b32 to NOP on RDNA4.
/// @details On the unified file the producer already writes to the correct
/// physical register. The MFMA that consumes the AccVGPR will be remapped
/// to read from the unified VGPR index.
std::vector<uint32_t> lower_accvgpr_write([[maybe_unused]] const Instruction &inst,
                                          [[maybe_unused]] rj_code_arch_t host_arch) {
  return {build_s_nop()};
}

/// @brief Lower CDNA4 s_cmpk_lt_u32 to RDNA4 s_cmp_lt_u32 with a literal.
/// @details RDNA4 removed the SOPK compare-with-immediate form, but SOPC
/// compare against a 32-bit literal has the same SCC result for the zero-
/// extended 16-bit immediate used by s_cmpk_lt_u32.
std::vector<uint32_t> lower_s_cmpk_lt_u32(const Instruction &inst) {
  if (std::string_view(inst.mnemonic()) != "s_cmpk_lt_u32")
    return {};

  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() < 4)
    return {};

  cdna4::SopkMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));

  constexpr uint8_t kRdna4OpSCmpLtU32 = 10;
  constexpr uint8_t kLiteralConstant = 255;
  rdna4::SopcMachineInst dst{};
  dst.encoding = 0x17E;
  dst.op = kRdna4OpSCmpLtU32;
  dst.ssrc0 = static_cast<uint8_t>(src.sdst);
  dst.ssrc1 = kLiteralConstant;

  return {std::bit_cast<uint32_t>(dst), static_cast<uint32_t>(src.simm16)};
}

// ---------------------------------------------------------------------------
// ExpandFn adapters — conform each lowering function to the unified signature
// ---------------------------------------------------------------------------

std::vector<uint32_t> expand_waitcnt(const Instruction &inst, uint32_t, uint64_t,
                                     const LivenessAnalysis &, const LaneLayout *,
                                     const LaneLayout *) {
  if (!inst.raw_encoding())
    return {};
  const auto &sopp = *reinterpret_cast<const cdna4::SoppMachineInst *>(inst.raw_encoding());
  return encode_waitcnt_gfx12(decode_waitcnt_gfx9(sopp.simm16));
}

std::vector<uint32_t> expand_s_barrier(const Instruction &inst, uint32_t, uint64_t,
                                       const LivenessAnalysis &, const LaneLayout *,
                                       const LaneLayout *) {
  if (std::string_view(inst.mnemonic()) != "s_barrier")
    return {};

  // CDNA4 has a single workgroup barrier instruction. RDNA4 split it into an
  // arrival signal and a wait. Barrier id -1 is the architectural "normal"
  // workgroup barrier used by LLVM when lowering a monolithic s_barrier.
  constexpr uint8_t kInlineConstNeg1 = 193;
  constexpr uint16_t kBarrierIdNeg1 = 0xFFFF;
  return {make_gfx12_sop1(kOpBarrierSignal, kInlineConstNeg1),
          make_gfx12_sopp(kOpBarrierWait, kBarrierIdNeg1)};
}

std::vector<uint32_t> expand_v_lshl_add_u64(const Instruction &inst, uint32_t, uint64_t offset,
                                            const LivenessAnalysis &liveness, const LaneLayout *,
                                            const LaneLayout *) {
  return lower_v_lshl_add_u64(inst, ROCJITSU_CODE_ARCH_RDNA4, offset, liveness);
}

std::vector<uint32_t> expand_v_bitop3_b32(const Instruction &inst, uint32_t, uint64_t offset,
                                          const LivenessAnalysis &liveness, const LaneLayout *,
                                          const LaneLayout *) {
  return lower_v_bitop3_b32(inst, offset, liveness);
}

std::vector<uint32_t> expand_accvgpr_read(const Instruction &inst, uint32_t, uint64_t,
                                          const LivenessAnalysis &, const LaneLayout *,
                                          const LaneLayout *) {
  return lower_accvgpr_read(inst, ROCJITSU_CODE_ARCH_RDNA4, 0);
}

std::vector<uint32_t> expand_accvgpr_write(const Instruction &inst, uint32_t, uint64_t,
                                           const LivenessAnalysis &, const LaneLayout *,
                                           const LaneLayout *) {
  return lower_accvgpr_write(inst, ROCJITSU_CODE_ARCH_RDNA4);
}

std::vector<uint32_t> expand_mfma_f32_16x16x16_f16(const Instruction &inst, uint32_t arch,
                                                   uint64_t offset,
                                                   const LivenessAnalysis &liveness,
                                                   const LaneLayout *guest,
                                                   const LaneLayout *host) {
  return lower_mfma_f32_16x16x16_f16(inst, static_cast<rj_code_arch_t>(arch), offset, liveness, 0,
                                     guest, host);
}

std::vector<uint32_t> expand_s_cmpk_lt_u32(const Instruction &inst, uint32_t, uint64_t,
                                           const LivenessAnalysis &, const LaneLayout *,
                                           const LaneLayout *) {
  return lower_s_cmpk_lt_u32(inst);
}

// ---------------------------------------------------------------------------
// Expand rules table — sorted by (src_encoding_id, src_opcode) for binary search
// ---------------------------------------------------------------------------

// CDNA4 encoding prefixes as reported by Instruction::encoding_id() (word0 >> 23).
// Opcodes are only unique within one encoding. For example, SOPP:s_waitcnt and
// SOP2:s_and_b32 both use opcode 12, so semantic rules must include the
// encoding prefix to avoid expanding an unrelated instruction.
constexpr uint16_t kCdna4Enc_SOPK_s_cmpk_lt_u32 = 0x16C;
constexpr uint16_t kCdna4Enc_SOPP = 0x17F;
constexpr uint16_t kCdna4Enc_FLAT_GLBL = 0x1B8;
constexpr uint16_t kCdna4Enc_VOP3P = 0x1A7;
// For VOP3, the current decoder key is still the raw top nine bits of word0,
// so it includes the high opcode bits rather than only the six-bit VOP3
// encoding field. Keep this constant tied to the concrete source instruction.
constexpr uint16_t kCdna4Enc_VOP3_v_lshl_add_u64 = 0x1A4;
constexpr uint16_t kCdna4Enc_VOP3_v_bitop3_b32 = 0x1A4;

// CDNA4 opcodes (from decoder: opcode_ = inst_.op)
constexpr uint16_t kCdna4Op_s_cmpk_lt_u32 = 12;
constexpr uint16_t kCdna4Op_s_barrier = 10;
constexpr uint16_t kCdna4Op_s_waitcnt = 12;
constexpr uint16_t kCdna4Op_global_store_dwordx4 = 31;
constexpr uint16_t kCdna4Op_v_mfma_f32_16x16x16_f16 = 77;
constexpr uint16_t kCdna4Op_v_accvgpr_read = 88;
constexpr uint16_t kCdna4Op_v_accvgpr_write = 89;
constexpr uint16_t kCdna4Op_v_lshl_add_u64 = 520;
constexpr uint16_t kCdna4Op_v_bitop3_b32 = 564;

const TranslationRule kExpandRules_cdna4_to_rdna4[] = {
    {kCdna4Enc_SOPK_s_cmpk_lt_u32, kCdna4Op_s_cmpk_lt_u32, RuleAction::Expand, 0, 0, nullptr,
     expand_s_cmpk_lt_u32, nullptr, nullptr},
    {kCdna4Enc_SOPP, kCdna4Op_s_barrier, RuleAction::Expand, 0, 0, nullptr, expand_s_barrier,
     nullptr, nullptr},
    {kCdna4Enc_SOPP, kCdna4Op_s_waitcnt, RuleAction::Expand, 0, 0, nullptr, expand_waitcnt, nullptr,
     nullptr},
    {kCdna4Enc_VOP3_v_lshl_add_u64, kCdna4Op_v_lshl_add_u64, RuleAction::Expand, 0, 0, nullptr,
     expand_v_lshl_add_u64, nullptr, nullptr},
    {kCdna4Enc_VOP3_v_bitop3_b32, kCdna4Op_v_bitop3_b32, RuleAction::Expand, 0, 0, nullptr,
     expand_v_bitop3_b32, nullptr, nullptr},
    {kCdna4Enc_VOP3P, kCdna4Op_v_mfma_f32_16x16x16_f16, RuleAction::Expand, 0, 0, nullptr,
     expand_mfma_f32_16x16x16_f16, &kMfmaF32_16x16x16_F16_Cdna4, &kWmmaF32_16x16x16_F16_Rdna4},
    {kCdna4Enc_VOP3P, kCdna4Op_v_accvgpr_read, RuleAction::Expand, 0, 0, nullptr,
     expand_accvgpr_read, nullptr, nullptr},
    {kCdna4Enc_VOP3P, kCdna4Op_v_accvgpr_write, RuleAction::Expand, 0, 0, nullptr,
     expand_accvgpr_write, nullptr, nullptr},
};

} // namespace

// ---------------------------------------------------------------------------
// SemanticTranslator implementation
// ---------------------------------------------------------------------------

SemanticTranslator::SemanticTranslator(rj_code_arch_t guest, rj_code_arch_t host)
    : host_arch_(host) {
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA4)
    expand_rules_ = kExpandRules_cdna4_to_rdna4;
}

void SemanticTranslator::set_accum_offset_info(std::vector<AccumOffsetInfo> infos) {
  std::ranges::sort(infos, {}, &AccumOffsetInfo::entry_text_offset);
  accum_offsets_ = std::move(infos);
}

uint16_t SemanticTranslator::accum_vgpr_base_for_offset(uint64_t offset) const {
  uint16_t base = 0;
  for (const auto &info : accum_offsets_) {
    if (info.entry_text_offset > offset)
      break;
    base = info.vgpr_base;
  }
  return base;
}

std::vector<uint32_t> SemanticTranslator::try_lower_expand(const Instruction &inst, uint64_t offset,
                                                           const LivenessAnalysis &liveness) const {
  const uint16_t enc = inst.encoding_id();
  const uint16_t op = inst.opcode();

  if (enc == kCdna4Enc_VOP3P && op == kCdna4Op_v_mfma_f32_16x16x16_f16) {
    return lower_mfma_f32_16x16x16_f16(inst, host_arch_, offset, liveness,
                                       accum_vgpr_base_for_offset(offset),
                                       &kMfmaF32_16x16x16_F16_Cdna4, &kWmmaF32_16x16x16_F16_Rdna4);
  }

  if (enc == kCdna4Enc_VOP3P && op == kCdna4Op_v_accvgpr_read) {
    return lower_accvgpr_read(inst, host_arch_, accum_vgpr_base_for_offset(offset));
  }

  if (enc == kCdna4Enc_FLAT_GLBL && op == kCdna4Op_global_store_dwordx4) {
    auto lowered = lower_flat_glbl_acc_store_b128(inst, accum_vgpr_base_for_offset(offset));
    if (!lowered.empty())
      return lowered;
  }

  TranslationRule key{enc, op, RuleAction::Expand, 0, 0, nullptr, nullptr, nullptr, nullptr};
  auto it = std::lower_bound(expand_rules_.begin(), expand_rules_.end(), key);
  if (it != expand_rules_.end() && it->src_encoding_id == enc && it->src_opcode == op &&
      it->expand_fn)
    return it->expand_fn(inst, static_cast<uint32_t>(host_arch_), offset, liveness,
                         it->guest_layout, it->host_layout);
  return {};
}

// --- Workgroup ID rewrite ---

std::vector<SemanticReplacement>
SemanticTranslator::rewrite_workgroup_ids(BasicBlock &block,
                                          std::span<WorkGroupRewriteState> wg_states,
                                          std::span<const uint8_t> translated_text) const {
  if (host_arch_ != ROCJITSU_CODE_ARCH_RDNA4 || wg_states.empty())
    return {};

  // RDNA4 delivers workgroup IDs via TTMP registers, not SGPRs.
  [[maybe_unused]] constexpr uint8_t kTTMP7 = 115; // workgroup_id_y (low16), _z (high16)

  std::vector<SemanticReplacement> result;

  for (auto &state : wg_states) {
    const auto &info = state.info;
    if (info.sgpr_wg_id_x < 0)
      continue;
    const auto old_sgpr = static_cast<uint16_t>(info.sgpr_wg_id_x);

    uint64_t offset = block.start_offset();
    for (const auto &inst : block.instructions()) {
      if (offset < info.entry_text_offset ||
          offset + static_cast<uint64_t>(inst.size()) > translated_text.size()) {
        offset += inst.size();
        continue;
      }

      // Read the already-translated instruction words from translated_text.
      uint32_t w0 = 0;
      uint32_t w1 = 0;
      std::memcpy(&w0, translated_text.data() + offset, 4);
      if (inst.size() >= 8)
        std::memcpy(&w1, translated_text.data() + offset + 4, 4);

      if (state.wg_id_x_live) {
        bool changed = patch_workgroup_src_operand(inst, w0, w1, old_sgpr, kTtmp9Encoding);
        if (changed) {
          std::vector<uint32_t> words{w0};
          if (inst.size() >= 8)
            words.push_back(w1);
          result.push_back({offset, offset + static_cast<uint64_t>(inst.size()), words});
        }
      }

      InstDefUse def_use(inst, 64);
      if (def_use.defs.contains({RegClass::SGPR, old_sgpr, 1}))
        state.wg_id_x_live = false;

      offset += inst.size();
    }
  }
  return result;
}

} // namespace rocjitsu
