// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file semantic_translator.cpp
/// @brief Semantic translator implementation and per-pair rule tables.

#include "rocjitsu/code/dbt/semantic_translator.h"

#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/dbt/hazard_tracker.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/machine_insts.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstring>
#include <string_view>

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

constexpr uint8_t kOpWaitLoadcnt = 64;
constexpr uint8_t kOpWaitStorecntDscnt = 73;
constexpr uint8_t kOpWaitKmcnt = 71;
constexpr uint8_t kOpWaitExpcnt = 68;

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
  // GFX9 vmcnt is a unified counter for both loads and stores; GFX12 splits
  // them. Conservative: emit storecnt whenever vmcnt is active, since we
  // cannot distinguish load-only from store-only waits in the source.
  const bool need_storecnt = (vals.vmcnt != 0x3F);
  const bool need_kmcnt = (vals.lgkmcnt != 0x0F);
  const bool need_dscnt = (vals.lgkmcnt != 0x0F);
  const bool need_expcnt = (vals.expcnt != 0x07);

  if (need_loadcnt)
    words.push_back(make_gfx12_sopp(kOpWaitLoadcnt, std::min<uint16_t>(vals.vmcnt, 63)));

  if (need_storecnt || need_dscnt) {
    const uint8_t sc = need_storecnt ? std::min<uint8_t>(vals.vmcnt, 15) : 15;
    const uint8_t dc = need_dscnt ? std::min<uint8_t>(vals.lgkmcnt, 15) : 15;
    words.push_back(make_gfx12_sopp(kOpWaitStorecntDscnt, (sc << 4) | dc));
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

// ---------------------------------------------------------------------------
// RDNA4 instruction builders (used by lowering functions below)
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

/// @brief Build a VOP1 v_mov_b32 instruction for RDNA4.
[[nodiscard]] constexpr uint32_t build_v_mov_b32(uint8_t vdst, uint16_t src0) {
  return (0x3Fu << 25) | (static_cast<uint32_t>(vdst) << 17) | (1u << 9) | (src0 & 0x1FF);
}

// ---------------------------------------------------------------------------
// Instruction lowering functions
// ---------------------------------------------------------------------------

std::vector<uint32_t> lower_v_lshl_add_u64(const Instruction &inst, rj_code_arch_t host_arch) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() < 8)
    return {};

  cdna4::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  const uint16_t vdst = src.vdst;
  const uint16_t src0 = src.src0;
  const uint16_t src2 = src.src2;

  constexpr uint16_t kVccLo = 106;
  constexpr uint16_t kOpAddCoU32 = 768;
  constexpr uint16_t kOpAddCoCiU32 = 288;
  constexpr uint8_t kSoppWaitAlu = 8;

  std::vector<uint32_t> words;

  // v_add_co_u32 vdst_lo, vcc_lo, src0_lo, src2_lo
  {
    auto [w0, w1] = build_vop3(kOpAddCoU32, static_cast<uint8_t>(vdst), src0, src2);
    w0 |= (kVccLo << 8); // sdst = vcc_lo
    words.push_back(w0);
    words.push_back(w1);
  }

  // VCC writeback dependency. GFX12 (RDNA4) requires an explicit s_wait_alu
  // before reading VCC as carry-in. GFX11 (RDNA3) handles back-to-back
  // VCC-write/VCC-read VALU ops via hardware scoreboarding, so no wait is
  // required there. SOPP opcode 8 has different meanings between gens
  // (s_wait_alu on GFX12 vs s_waitcnt on GFX11) so emitting it
  // unconditionally would corrupt RDNA3 output.
  if (host_arch == ROCJITSU_CODE_ARCH_RDNA4)
    words.push_back(pack_sopp(kSoppWaitAlu, 0xFFFD));

  // v_add_co_ci_u32 vdst_hi, vcc_lo, src0_hi, src2_hi, vcc_lo
  {
    auto [w0, w1] =
        build_vop3(kOpAddCoCiU32, static_cast<uint8_t>(vdst + 1), static_cast<uint16_t>(src0 + 1),
                   static_cast<uint16_t>(src2 + 1), kVccLo);
    w0 |= (kVccLo << 8); // sdst = vcc_lo
    words.push_back(w0);
    words.push_back(w1);
  }

  return words;
}

/// @brief Lower v_accvgpr_read_b32 to v_mov_b32 or NOP on RDNA4.
/// @details acc[N] = v[N+256] on the unified file. If dst aliases the unified
/// src, emit NOP. Otherwise emit v_mov_b32.
std::vector<uint32_t> lower_accvgpr_read(const Instruction &inst,
                                         [[maybe_unused]] rj_code_arch_t host_arch) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() < 8)
    return {};

  cdna4::Vop3pMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));

  const uint16_t dst_vgpr = src.vdst;
  const uint16_t src_acc = src.src0;
  assert(src_acc >= 768 && src_acc <= 1023);
  const uint16_t src_unified = src_acc - 512;

  if (dst_vgpr == src_unified)
    return {build_s_nop()};

  return {build_v_mov_b32(static_cast<uint8_t>(dst_vgpr), 256 + src_unified)};
}

// ---------------------------------------------------------------------------
// GFX12 SOPP opcodes and s_delay_alu constants
// ---------------------------------------------------------------------------

constexpr uint8_t kSoppWaitIdle = 0x0A;
constexpr uint8_t kOpWaitDscnt = 70;

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
constexpr uint8_t kOpLshlrevB32 = 24;
constexpr uint8_t kOpXorB32 = 29;

// VOP3P opcodes (GFX12)
constexpr uint8_t kOpWmmaF32_16x16x16_F16 = 64;

/// @brief Lower v_mfma_f32_16x16x16_f16 to v_wmma_f32_16x16x16_f16 on RDNA4.
///
/// WMMA Wave64 writes all 64 lanes but swaps rows 4-7 and 8-11 vs MFMA
/// layout (lanes 16-31 ↔ lanes 32-47). Fix via ds_bpermute with a
/// pre-computed address VGPR that encodes identity for lanes 0-15,48-63
/// and XOR-48 for lanes 16-47.
std::vector<uint32_t> lower_mfma_f32_16x16x16_f16(const Instruction &inst,
                                                  [[maybe_unused]] rj_code_arch_t host_arch,
                                                  uint64_t offset, const RegisterLiveness &liveness,
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

  if (src2 >= 256)
    return {};

  assert(src0 >= 256 && src1 >= 256 && "MFMA VGPR sources expected");

  auto exec_save_opt = liveness.find_free_sgpr_pair();
  if (!exec_save_opt)
    return {};
  const uint8_t kExecSave = static_cast<uint8_t>(*exec_save_opt);

  auto tmp_sgpr_opt = liveness.find_free_sgpr(kExecSave + 2);
  if (!tmp_sgpr_opt)
    return {};
  const uint8_t kTmpSgpr = static_cast<uint8_t>(*tmp_sgpr_opt);

  auto free_reg = liveness.find_free_run(offset, 1, vdst + 4);
  if (!free_reg)
    return {};
  const uint8_t vaddr = static_cast<uint8_t>(*free_reg);

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

  // WMMA: single pass, writes all 64 lanes in WMMA layout.
  {
    auto [w0, w1] = build_vop3p(kOpWmmaF32_16x16x16_F16, vdst, src0, src1, src2);
    words.push_back(w0);
    words.push_back(w1);
  }

  // Drain pipelines so ds_bpermute can read WMMA output from VGPR file.
  words.push_back(pack_sopp(kSoppWaitIdle, 0));

  // ds_bpermute: remap WMMA output lanes 16-31 ↔ 32-47 to match MFMA layout.
  for (int r = 0; r < 4; ++r) {
    auto [w0, w1] = build_ds_bpermute(vdst + r, vaddr, vdst + r);
    words.push_back(w0);
    words.push_back(w1);
  }
  words.push_back(pack_sopp(kOpWaitDscnt, 0));

  words.push_back(build_s_mov_b64(kExecLo, kExecSave));

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

// ---------------------------------------------------------------------------
// ExpandFn adapters — conform each lowering function to the unified signature
// ---------------------------------------------------------------------------

std::vector<uint32_t> expand_waitcnt(const Instruction &inst, uint32_t, uint64_t,
                                     const RegisterLiveness &, const LaneLayout *,
                                     const LaneLayout *) {
  if (!inst.raw_encoding())
    return {};
  const auto &sopp = *reinterpret_cast<const cdna4::SoppMachineInst *>(inst.raw_encoding());
  return encode_waitcnt_gfx12(decode_waitcnt_gfx9(sopp.simm16));
}

std::vector<uint32_t> expand_v_lshl_add_u64(const Instruction &inst, uint32_t arch, uint64_t,
                                            const RegisterLiveness &, const LaneLayout *,
                                            const LaneLayout *) {
  return lower_v_lshl_add_u64(inst, static_cast<rj_code_arch_t>(arch));
}

/// @brief Lower CDNA4 (GFX9) s_waitcnt to RDNA3 (GFX11) s_waitcnt.
/// @details The simm16 counter-bit layout differs between GFX9 and GFX11
/// (different bit positions for vmcnt/expcnt/lgkmcnt), so a verbatim
/// simm16 copy by the auto-encoder gives incorrect waits. Rather than
/// re-encode the counters precisely, emit a conservative full-drain
/// s_waitcnt 0 which waits for every counter to reach 0. Slower than the
/// original (which only waited on the specified counter classes) but
/// always correct. A future change can add encode_waitcnt_gfx11 for a
/// precise mapping, mirroring encode_waitcnt_gfx12 for the RDNA4 path.
std::vector<uint32_t> expand_waitcnt_gfx9_to_gfx11(const Instruction &inst, uint32_t, uint64_t,
                                                   const RegisterLiveness &, const LaneLayout *,
                                                   const LaneLayout *) {
  // Defensive guard: the rule table is keyed by encoding and opcode, but only
  // SOPP s_waitcnt has the GFX9 waitcnt simm16 layout this lowering expects.
  constexpr uint16_t kEnc_SOPP_value = 0x17F;
  if (inst.encoding_id() != kEnc_SOPP_value)
    return {};
  constexpr uint32_t kRdna3SoppOp_s_waitcnt = 9;
  return {pack_sopp(kRdna3SoppOp_s_waitcnt, 0)};
}

std::vector<uint32_t> expand_accvgpr_read(const Instruction &inst, uint32_t, uint64_t,
                                          const RegisterLiveness &, const LaneLayout *,
                                          const LaneLayout *) {
  return lower_accvgpr_read(inst, ROCJITSU_CODE_ARCH_RDNA4);
}

std::vector<uint32_t> expand_accvgpr_write(const Instruction &inst, uint32_t, uint64_t,
                                           const RegisterLiveness &, const LaneLayout *,
                                           const LaneLayout *) {
  return lower_accvgpr_write(inst, ROCJITSU_CODE_ARCH_RDNA4);
}

std::vector<uint32_t> expand_mfma_f32_16x16x16_f16(const Instruction &inst, uint32_t arch,
                                                   uint64_t offset,
                                                   const RegisterLiveness &liveness,
                                                   const LaneLayout *guest,
                                                   const LaneLayout *host) {
  return lower_mfma_f32_16x16x16_f16(inst, static_cast<rj_code_arch_t>(arch), offset, liveness,
                                     guest, host);
}

// ---------------------------------------------------------------------------
// Expand rules table — sorted by (src_encoding_id, src_opcode) for binary search
// ---------------------------------------------------------------------------

// CDNA4 encoding IDs (encoding_id = w0 >> 23, from CDNA4 instruction words)
constexpr uint16_t kEncSopp = 0x17F;      // SOPP (0xBF8x → 0x17F)
constexpr uint16_t kEncVop3 = 0x1A4;      // VOP3A (0xD2xx → 0x1A4)
constexpr uint16_t kEncVop3pMfma = 0x1A7; // VOP3P-MFMA (0xD3xx → 0x1A7, also AccVGPR)

// CDNA4 opcodes (from decoder: opcode_ = inst_.op)
constexpr uint16_t kCdna4Op_s_waitcnt = 12;
constexpr uint16_t kCdna4Op_v_mfma_f32_16x16x16_f16 = 77;
constexpr uint16_t kCdna4Op_v_accvgpr_read = 88;
constexpr uint16_t kCdna4Op_v_accvgpr_write = 89;
constexpr uint16_t kCdna4Op_v_lshl_add_u64 = 520;

// Table MUST be sorted by (src_encoding_id, src_opcode) for binary search.
const TranslationRule kExpandRules_cdna4_to_rdna4[] = {
    {kEncSopp, kCdna4Op_s_waitcnt, RuleAction::Expand, 0, 0, nullptr, expand_waitcnt, nullptr,
     nullptr},
    {kEncVop3, kCdna4Op_v_lshl_add_u64, RuleAction::Expand, 0, 0, nullptr, expand_v_lshl_add_u64,
     nullptr, nullptr},
    {kEncVop3pMfma, kCdna4Op_v_mfma_f32_16x16x16_f16, RuleAction::Expand, 0, 0, nullptr,
     expand_mfma_f32_16x16x16_f16, &kMfmaF32_16x16x16_F16_Cdna4, &kWmmaF32_16x16x16_F16_Rdna4},
    {kEncVop3pMfma, kCdna4Op_v_accvgpr_read, RuleAction::Expand, 0, 0, nullptr, expand_accvgpr_read,
     nullptr, nullptr},
    {kEncVop3pMfma, kCdna4Op_v_accvgpr_write, RuleAction::Expand, 0, 0, nullptr,
     expand_accvgpr_write, nullptr, nullptr},
};

// CDNA4 -> RDNA3 expand rules. With the codegen fixes (per-pair enc_map for
// the FLAT family, FLAT_LOAD_/FLAT_STORE_ DWORD->B32 in the rename map,
// target_opcode preserved through domain-rule overrides), the auto-generated
// encoding translator handles the FLAT_GLBL family at the encoding level.
// Two genuine Expand cases remain:
//   - s_waitcnt: simm16 counter-bit layout differs between GFX9 and GFX11,
//     so an opcode swap alone is not enough.
//   - v_lshl_add_u64: no carry-propagating fused 64-bit add on RDNA3, expands
//     to v_add_co_u32 + v_add_co_ci_u32.
// Rule table must stay sorted by (src_encoding_id, src_opcode).
const TranslationRule kExpandRules_cdna4_to_rdna3[] = {
    {kEncSopp, kCdna4Op_s_waitcnt, RuleAction::Expand, 0, 0, nullptr, expand_waitcnt_gfx9_to_gfx11,
     nullptr, nullptr},
    {kEncVop3, kCdna4Op_v_lshl_add_u64, RuleAction::Expand, 0, 0, nullptr, expand_v_lshl_add_u64,
     nullptr, nullptr},
};

} // namespace

// ---------------------------------------------------------------------------
// SemanticTranslator implementation
// ---------------------------------------------------------------------------

SemanticTranslator::SemanticTranslator(rj_code_arch_t guest, rj_code_arch_t host)
    : host_arch_(host) {
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA4)
    expand_rules_ = kExpandRules_cdna4_to_rdna4;
  else if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA3)
    expand_rules_ = kExpandRules_cdna4_to_rdna3;
}

std::vector<uint32_t> SemanticTranslator::try_lower_expand(const Instruction &inst, uint64_t offset,
                                                           const RegisterLiveness &liveness) const {
  const uint16_t eid = inst.encoding_id();
  const uint16_t op = inst.opcode();
  TranslationRule key{eid, op, RuleAction::Expand, 0, 0, nullptr, nullptr, nullptr, nullptr};
  auto it = std::lower_bound(expand_rules_.begin(), expand_rules_.end(), key);
  if (it != expand_rules_.end() && it->src_encoding_id == eid && it->src_opcode == op &&
      it->expand_fn)
    return it->expand_fn(inst, static_cast<uint32_t>(host_arch_), offset, liveness,
                         it->guest_layout, it->host_layout);
  return {};
}

// --- Workgroup ID rewrite ---

std::vector<SemanticReplacement> SemanticTranslator::rewrite_workgroup_ids(
    BasicBlock &block, std::span<const CodeObjectPatcher::WorkGroupIdInfo> wg_info,
    std::span<const uint8_t> translated_text) const {
  if (host_arch_ != ROCJITSU_CODE_ARCH_RDNA4 || wg_info.empty())
    return {};

  // RDNA4 delivers workgroup IDs via TTMP registers, not SGPRs.
  // TTMP9 = workgroup_id_x (full 32 bits).
  // TTMP7 = workgroup_id_y in bits[15:0], workgroup_id_z in bits[31:16].
  constexpr uint8_t kTTMP9 = 117;
  constexpr uint8_t kTTMP7 = 115;

  std::vector<SemanticReplacement> result;

  for (const auto &info : wg_info) {
    uint64_t offset = block.start_offset();
    for (const auto &inst : block.instructions()) {
      if (offset < info.entry_text_offset || offset + 8 > translated_text.size()) {
        offset += inst.size();
        continue;
      }

      if (inst.size() < 8) {
        offset += inst.size();
        continue;
      }

      uint32_t w0, w1;
      std::memcpy(&w0, translated_text.data() + offset, 4);
      std::memcpy(&w1, translated_text.data() + offset + 4, 4);

      // VOP3 word1 has src0 in bits[8:0]. Check if it matches any
      // workgroup_id SGPR and substitute the corresponding TTMP.
      // Only rewrite src0 — the workgroup_id SGPRs are consumed in
      // s_mul/v_mul patterns where the ID appears as src0. Rewriting
      // other operand slots risks false matches with unrelated SGPRs
      // that happen to share the same encoding value.
      uint16_t src0 = w1 & 0x1FF;
      if (info.sgpr_wg_id_x >= 0 && src0 == static_cast<uint16_t>(info.sgpr_wg_id_x)) {
        uint32_t new_w1 = (w1 & ~0x1FFu) | kTTMP9;
        result.push_back({offset, offset + static_cast<uint64_t>(inst.size()), {w0, new_w1}});
      } else if (info.sgpr_wg_id_y >= 0 && src0 == static_cast<uint16_t>(info.sgpr_wg_id_y)) {
        uint32_t new_w1 = (w1 & ~0x1FFu) | kTTMP7;
        result.push_back({offset, offset + static_cast<uint64_t>(inst.size()), {w0, new_w1}});
      } else if (info.sgpr_wg_id_z >= 0 && src0 == static_cast<uint16_t>(info.sgpr_wg_id_z)) {
        uint32_t new_w1 = (w1 & ~0x1FFu) | kTTMP7;
        result.push_back({offset, offset + static_cast<uint64_t>(inst.size()), {w0, new_w1}});
      }

      offset += inst.size();
    }
  }
  return result;
}

} // namespace rocjitsu
