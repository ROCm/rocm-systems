// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "effects.h"

#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/isa/register_set.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>

namespace rocjitsu::isa_diff {
namespace {

// SALU instructions whose execution RocJITsu implements and whose entire effect is on
// SGPRs, VCC, EXEC, M0 and SCC (the float forms of GFX11.5+ also read MODE, which is
// equal on both sides). Names of every generation: an instruction is only ever decoded
// by its own target's decoder. Every operand is still checked;
// membership here is necessary, not sufficient. Branches, PC access, hwreg access,
// indirect SGPR access, waits, messages and traps are deliberately absent.
constexpr std::array kP0Mnemonics = std::to_array<std::string_view>({
    // SOP2
    "s_add_u32",
    "s_sub_u32",
    "s_add_i32",
    "s_sub_i32",
    "s_addc_u32",
    "s_subb_u32",
    "s_min_i32",
    "s_min_u32",
    "s_max_i32",
    "s_max_u32",
    "s_cselect_b32",
    "s_cselect_b64",
    "s_and_b32",
    "s_and_b64",
    "s_or_b32",
    "s_or_b64",
    "s_xor_b32",
    "s_xor_b64",
    "s_andn2_b32",
    "s_andn2_b64",
    "s_orn2_b32",
    "s_orn2_b64",
    "s_nand_b32",
    "s_nand_b64",
    "s_nor_b32",
    "s_nor_b64",
    "s_xnor_b32",
    "s_xnor_b64",
    "s_lshl_b32",
    "s_lshl_b64",
    "s_lshr_b32",
    "s_lshr_b64",
    "s_ashr_i32",
    "s_ashr_i64",
    "s_bfm_b32",
    "s_bfm_b64",
    "s_mul_i32",
    "s_bfe_u32",
    "s_bfe_i32",
    "s_bfe_u64",
    "s_bfe_i64",
    "s_absdiff_i32",
    "s_mul_hi_u32",
    "s_mul_hi_i32",
    "s_lshl1_add_u32",
    "s_lshl2_add_u32",
    "s_lshl3_add_u32",
    "s_lshl4_add_u32",
    "s_pack_ll_b32_b16",
    "s_pack_lh_b32_b16",
    "s_pack_hh_b32_b16",
    // SOP1
    "s_mov_b32",
    "s_mov_b64",
    "s_cmov_b32",
    "s_cmov_b64",
    "s_not_b32",
    "s_not_b64",
    "s_wqm_b32",
    "s_wqm_b64",
    "s_brev_b32",
    "s_brev_b64",
    "s_bcnt0_i32_b32",
    "s_bcnt0_i32_b64",
    "s_bcnt1_i32_b32",
    "s_bcnt1_i32_b64",
    "s_ff0_i32_b32",
    "s_ff0_i32_b64",
    "s_ff1_i32_b32",
    "s_ff1_i32_b64",
    "s_flbit_i32_b32",
    "s_flbit_i32_b64",
    "s_flbit_i32",
    "s_flbit_i32_i64",
    "s_sext_i32_i8",
    "s_sext_i32_i16",
    "s_bitset0_b32",
    "s_bitset0_b64",
    "s_bitset1_b32",
    "s_bitset1_b64",
    "s_and_saveexec_b64",
    "s_or_saveexec_b64",
    "s_xor_saveexec_b64",
    "s_andn2_saveexec_b64",
    "s_orn2_saveexec_b64",
    "s_nand_saveexec_b64",
    "s_nor_saveexec_b64",
    "s_xnor_saveexec_b64",
    "s_quadmask_b32",
    "s_quadmask_b64",
    "s_abs_i32",
    "s_andn1_saveexec_b64",
    "s_orn1_saveexec_b64",
    "s_andn1_wrexec_b64",
    "s_andn2_wrexec_b64",
    "s_bitreplicate_b64_b32",
    // Wave32 exec-mask forms (GFX10+).
    "s_and_saveexec_b32",
    "s_or_saveexec_b32",
    "s_xor_saveexec_b32",
    "s_andn2_saveexec_b32",
    "s_orn2_saveexec_b32",
    "s_nand_saveexec_b32",
    "s_nor_saveexec_b32",
    "s_xnor_saveexec_b32",
    "s_andn1_saveexec_b32",
    "s_orn1_saveexec_b32",
    "s_andn1_wrexec_b32",
    "s_andn2_wrexec_b32",
    // GFX11 names.
    "s_and_not1_b32",
    "s_and_not1_b64",
    "s_or_not1_b32",
    "s_or_not1_b64",
    "s_and_not1_saveexec_b32",
    "s_and_not1_saveexec_b64",
    "s_or_not1_saveexec_b32",
    "s_or_not1_saveexec_b64",
    "s_and_not0_saveexec_b32",
    "s_and_not0_saveexec_b64",
    "s_or_not0_saveexec_b32",
    "s_or_not0_saveexec_b64",
    "s_and_not1_wrexec_b32",
    "s_and_not1_wrexec_b64",
    "s_and_not0_wrexec_b32",
    "s_and_not0_wrexec_b64",
    "s_ctz_i32_b32",
    "s_ctz_i32_b64",
    "s_clz_i32_u32",
    "s_clz_i32_u64",
    "s_cls_i32",
    "s_cls_i32_i64",
    "s_pack_hl_b32_b16",
    // GFX12 names and GFX12.5 64-bit arithmetic.
    "s_add_co_u32",
    "s_add_co_i32",
    "s_add_co_ci_u32",
    "s_sub_co_u32",
    "s_sub_co_i32",
    "s_sub_co_ci_u32",
    "s_addk_co_i32",
    "s_add_nc_u64",
    "s_sub_nc_u64",
    "s_mul_u64",
    // SALU floating point (GFX11.5+).
    "s_add_f32",
    "s_sub_f32",
    "s_mul_f32",
    "s_fmac_f32",
    "s_fmaak_f32",
    "s_fmamk_f32",
    "s_min_f32",
    "s_max_f32",
    "s_min_num_f32",
    "s_max_num_f32",
    "s_minimum_f32",
    "s_maximum_f32",
    "s_add_f16",
    "s_sub_f16",
    "s_mul_f16",
    "s_fmac_f16",
    "s_min_f16",
    "s_max_f16",
    "s_min_num_f16",
    "s_max_num_f16",
    "s_minimum_f16",
    "s_maximum_f16",
    "s_ceil_f32",
    "s_floor_f32",
    "s_trunc_f32",
    "s_rndne_f32",
    "s_ceil_f16",
    "s_floor_f16",
    "s_trunc_f16",
    "s_rndne_f16",
    "s_cvt_f32_i32",
    "s_cvt_f32_u32",
    "s_cvt_i32_f32",
    "s_cvt_u32_f32",
    "s_cvt_f16_f32",
    "s_cvt_f32_f16",
    "s_cvt_hi_f32_f16",
    "s_cvt_pk_rtz_f16_f32",
    // SOPC
    "s_cmp_eq_i32",
    "s_cmp_lg_i32",
    "s_cmp_gt_i32",
    "s_cmp_ge_i32",
    "s_cmp_lt_i32",
    "s_cmp_le_i32",
    "s_cmp_eq_u32",
    "s_cmp_lg_u32",
    "s_cmp_gt_u32",
    "s_cmp_ge_u32",
    "s_cmp_lt_u32",
    "s_cmp_le_u32",
    "s_bitcmp0_b32",
    "s_bitcmp1_b32",
    "s_bitcmp0_b64",
    "s_bitcmp1_b64",
    "s_cmp_eq_u64",
    "s_cmp_lg_u64",
    "s_cmp_eq_f32",
    "s_cmp_lg_f32",
    "s_cmp_gt_f32",
    "s_cmp_ge_f32",
    "s_cmp_lt_f32",
    "s_cmp_le_f32",
    "s_cmp_o_f32",
    "s_cmp_u_f32",
    "s_cmp_neq_f32",
    "s_cmp_nlg_f32",
    "s_cmp_ngt_f32",
    "s_cmp_nge_f32",
    "s_cmp_nlt_f32",
    "s_cmp_nle_f32",
    "s_cmp_eq_f16",
    "s_cmp_lg_f16",
    "s_cmp_gt_f16",
    "s_cmp_ge_f16",
    "s_cmp_lt_f16",
    "s_cmp_le_f16",
    "s_cmp_o_f16",
    "s_cmp_u_f16",
    "s_cmp_neq_f16",
    "s_cmp_nlg_f16",
    "s_cmp_ngt_f16",
    "s_cmp_nge_f16",
    "s_cmp_nlt_f16",
    "s_cmp_nle_f16",
    // SOPK
    "s_movk_i32",
    "s_cmovk_i32",
    "s_cmpk_eq_i32",
    "s_cmpk_lg_i32",
    "s_cmpk_gt_i32",
    "s_cmpk_ge_i32",
    "s_cmpk_lt_i32",
    "s_cmpk_le_i32",
    "s_cmpk_eq_u32",
    "s_cmpk_lg_u32",
    "s_cmpk_gt_u32",
    "s_cmpk_ge_u32",
    "s_cmpk_lt_u32",
    "s_cmpk_le_u32",
    "s_addk_i32",
    "s_mulk_i32",
    // SOPP with no architectural effect.
    "s_nop",
    "s_version",
});

// Scheduling hints and dependency waits: no functional effect on an in-order wave.
constexpr std::array kRegionNoops = std::to_array<std::string_view>({
    "s_nop",
    "s_delay_alu",
    "s_clause",
    "s_wait_alu",
    "s_waitcnt_depctr",
    "s_set_inst_prefetch_distance",
    "s_inst_prefetch",
    "s_version",
});

// SALU instructions outside P0 that read no tracked state beyond their classified
// operands (the branch conditions SCC/VCC/EXEC are fieldless operands; see tests).
constexpr std::array kNoImplicitReads = std::to_array<std::string_view>({
    "s_endpgm",
    "s_waitcnt",
    "s_barrier",
    "s_branch",
    "s_cbranch_scc0",
    "s_cbranch_scc1",
    "s_cbranch_vccz",
    "s_cbranch_vccnz",
    "s_cbranch_execz",
    "s_cbranch_execnz",
    "s_sleep",
    "s_setprio",
    "s_setpc_b64", // reads its target only; a return's live-out is the calling convention's
    "s_set_pc_i64",
    "s_delay_alu",
    "s_clause",
    "s_wait_alu",
    "s_waitcnt_depctr",
    "s_set_inst_prefetch_distance",
    "s_inst_prefetch",
    "s_version",
    "s_nop",
    "s_waitcnt_vscnt",
    "s_waitcnt_vmcnt",
    "s_waitcnt_lgkmcnt",
    "s_waitcnt_expcnt",
    "s_wait_loadcnt",
    "s_wait_storecnt",
    "s_wait_dscnt",
    "s_wait_kmcnt",
    "s_wait_samplecnt",
    "s_wait_bvhcnt",
    "s_wait_expcnt",
    "s_wait_loadcnt_dscnt",
    "s_wait_storecnt_dscnt",
    "s_wait_asynccnt",
    "s_wait_tensorcnt",
    "s_wait_xcnt",
});

// Scalar operand selector values (SSRC/SDST fields) shared by every generation. M0 and
// NULL move (Target::m0_selector/null_selector); 102..105 are FLAT_SCRATCH/XNACK_MASK on
// GFX9 and ordinary SGPRs from GFX10.
constexpr int kSelFlatScratchLo = 102;
constexpr int kSelXnackMaskHi = 105;
constexpr int kSelVccLo = 106;
constexpr int kSelVccHi = 107;
constexpr int kSelSharedBase = 235;        ///< src_shared_base
constexpr int kSelPopsExitingWaveId = 239; ///< src_pops_exiting_wave_id
constexpr int kSelVccz = 251;
constexpr int kSelExecz = 252;
constexpr int kSelScc = 253;

enum class Space {
  Tracked,  ///< Names tracked locations (added to the LocSet).
  Vector,   ///< Names VGPRs (added to the VSet).
  Outside,  ///< Names state outside the tracked universe (VGPR, TTMP, memory, ...).
  Constant, ///< Inline constant or literal.
  Pc,       ///< The program counter.
  Unknown,  ///< Cannot be named; may alias tracked state.
};

struct OperandClass {
  Space space = Space::Unknown;
  LocSet locs;
  VSet vlocs;
  std::string what;
};

void add_pair(LocSet &locs, unsigned lo, unsigned hi, int bits) {
  locs.set(lo);
  if (bits > 32)
    locs.set(hi);
}

OperandClass classify_special(RegClass cls, const Target &t) {
  OperandClass out;
  out.space = Space::Tracked;
  // A fieldless VCC or EXEC operand is the wave-sized mask: the pair on wave64, the low
  // half on wave32.
  const int mask_bits = t.wave32() ? 32 : 64;
  switch (cls) {
  case RegClass::SCC:
    out.locs.set(kLocScc);
    return out;
  case RegClass::VCC:
    add_pair(out.locs, kLocVccLo, kLocVccHi, mask_bits);
    return out;
  case RegClass::EXEC:
    add_pair(out.locs, kLocExecLo, kLocExecHi, mask_bits);
    return out;
  case RegClass::M0:
    out.locs.set(kLocM0);
    return out;
  case RegClass::PC:
    out.space = Space::Pc;
    out.what = "pc";
    return out;
  default:
    out.space = Space::Outside;
    out.what = "special register outside the tracked universe";
    return out;
  }
}

OperandClass classify_operand(const Operand &op, const Target &t) {
  OperandClass out;
  if (op.is_fieldless()) {
    if (auto cls = op.to_special_reg_class())
      return classify_special(*cls, t);
    out.space = Space::Outside;
    out.what = "fieldless operand '" + op.name() + "'";
    return out;
  }
  // A wave-mask operand is decoded at wave64 width; wave32 uses its first register.
  const bool narrow = t.wave32() && op.is_wave_mask();
  if (auto ref = op.to_register_ref()) {
    if (narrow)
      ref->width = 1;
    switch (ref->cls) {
    case RegClass::SGPR:
      if (ref->index + ref->width > t.sgprs()) {
        out.space = Space::Unknown;
        out.what = "SGPR beyond s" + std::to_string(t.sgprs() - 1);
        return out;
      }
      out.space = Space::Tracked;
      for (unsigned i = 0; i < ref->width; ++i)
        out.locs.set(ref->index + i);
      return out;
    case RegClass::EXEC:
      out.space = Space::Tracked;
      for (unsigned i = 0; i < ref->width; ++i)
        out.locs.set(kLocExecLo + ((ref->index + i) & 1u));
      return out;
    case RegClass::VCC:
      out.space = Space::Tracked;
      for (unsigned i = 0; i < ref->width; ++i)
        out.locs.set(kLocVccLo + ((ref->index + i) & 1u));
      return out;
    case RegClass::M0:
    case RegClass::SCC:
      return classify_special(ref->cls, t);
    case RegClass::PC:
      out.space = Space::Pc;
      out.what = "pc";
      return out;
    case RegClass::VGPR:
    case RegClass::ACC_VGPR: {
      // AGPRs are vector locations kAgprBase + n (RocJITsu's logical layout).
      const bool acc = ref->cls == RegClass::ACC_VGPR;
      if (ref->index + std::max<unsigned>(ref->width, 1) > kNumArchVgprs) {
        out.space = Space::Unknown;
        out.what = acc ? "AGPR beyond a255" : "VGPR beyond v255";
        return out;
      }
      out.space = Space::Vector;
      for (unsigned i = 0; i < std::max<unsigned>(ref->width, 1); ++i)
        out.vlocs.set((acc ? kAgprBase : 0u) + ref->index + i);
      return out;
    }
    case RegClass::TTMP:
      out.space = Space::Outside;
      out.what = "trap temporary";
      return out;
    case RegClass::FLAT_SCRATCH:
      out.space = Space::Outside;
      out.what = "flat_scratch";
      return out;
    }
  }
  if (op.has_register_selector()) {
    const int sel = op.encoding_value();
    const int bits = narrow ? 32 : op.size_bits();
    out.space = Space::Tracked;
    if (sel == t.m0_selector()) {
      out.locs.set(kLocM0);
      return out;
    }
    if (sel == t.null_selector()) {
      // NULL reads as zero and discards writes.
      out.space = Space::Constant;
      return out;
    }
    switch (sel) {
    case kSelVccLo:
      add_pair(out.locs, kLocVccLo, kLocVccHi, bits);
      return out;
    case kSelVccHi:
      out.locs.set(kLocVccHi);
      return out;
    case kSelScc:
      out.locs.set(kLocScc);
      return out;
    case kSelVccz:
      add_pair(out.locs, kLocVccLo, kLocVccHi, static_cast<int>(t.wave_size));
      return out;
    case kSelExecz:
      add_pair(out.locs, kLocExecLo, kLocExecHi, static_cast<int>(t.wave_size));
      return out;
    default:
      break;
    }
    if (t.major == 9 && sel >= kSelFlatScratchLo && sel <= kSelXnackMaskHi) {
      out.space = Space::Outside;
      out.what = "flat_scratch/xnack_mask selector";
      return out;
    }
    if (sel >= kSelSharedBase && sel <= kSelPopsExitingWaveId) {
      // src_shared_base/limit, src_private_base/limit, src_pops_exiting_wave_id: read-only
      // dispatch constants, the same for both sides; named state, not an unknown effect.
      out.space = Space::Outside;
      out.what = "aperture register";
      return out;
    }
    if (op.const_value()) {
      out.space = Space::Constant;
      return out;
    }
    out.space = Space::Unknown;
    out.what = "scalar selector " + std::to_string(sel) + " ('" + op.name() + "')";
    return out;
  }
  if (op.const_value()) {
    out.space = Space::Constant;
    return out;
  }
  out.space = Space::Outside;
  out.what = "operand '" + op.name() + "' without a register";
  return out;
}

LocSet special_locs() {
  LocSet s;
  for (unsigned l : {kLocExecLo, kLocExecHi, kLocVccLo, kLocVccHi, kLocM0})
    s.set(l);
  return s;
}

bool has_mnemonic_prefix(std::string_view mnemonic, std::string_view prefix) {
  return mnemonic.substr(0, prefix.size()) == prefix;
}

/// @p text names @p name as an operand token.
bool text_names(std::string_view text, std::string_view name) {
  for (size_t p = text.find(name); p != std::string_view::npos; p = text.find(name, p + 1)) {
    const bool start = p == 0 || text[p - 1] == ' ' || text[p - 1] == ',' || text[p - 1] == '[';
    const size_t e = p + name.size();
    const bool end = e == text.size() || text[e] == ',' || text[e] == ' ' || text[e] == ']';
    if (start && end)
      return true;
  }
  return false;
}

} // namespace

std::vector<SourceOperand> source_operands(const Instruction &inst, const Target &target) {
  std::vector<SourceOperand> out;
  for (int i = 0; i < inst.num_src_operands(); ++i) {
    const Operand &op = *inst.src_operand(i);
    SourceOperand s;
    s.fieldless = op.is_fieldless();
    const OperandClass cls = classify_operand(op, target);
    if (cls.space == Space::Tracked)
      s.locs = cls.locs;
    else if (cls.space == Space::Vector)
      s.vlocs = cls.vlocs;
    else if (cls.space == Space::Constant)
      if (auto v = op.const_value())
        s.constant = static_cast<uint32_t>(*v);
    out.push_back(s);
  }
  return out;
}

bool is_p0_mnemonic(std::string_view mnemonic) {
  return std::ranges::find(kP0Mnemonics, mnemonic) != kP0Mnemonics.end();
}

bool is_region_memory(std::string_view m) {
  auto starts = [&](std::string_view p) { return m.substr(0, p.size()) == p; };
  auto has = [&](std::string_view p) { return m.find(p) != std::string_view::npos; };
  // Counter waits: s_waitcnt (all), s_waitcnt_{vm,lgkm,exp,vs}cnt (GFX10-11) and the split
  // s_wait_*cnt forms (GFX12). s_wait_alu/s_wait_idle/s_wait_event are not counter waits.
  if (m == "s_waitcnt" || m == "s_waitcnt_vmcnt" || m == "s_waitcnt_lgkmcnt" ||
      m == "s_waitcnt_expcnt" || m == "s_waitcnt_vscnt")
    return true;
  if (starts("s_wait_") && (m.ends_with("cnt") || has("cnt_")))
    return true; // s_wait_loadcnt, ..., s_wait_loadcnt_dscnt, s_wait_storecnt_dscnt
  return region_memory_access(m).has_value();
}

std::optional<MemoryAccess> region_memory_access(std::string_view m) {
  auto starts = [&](std::string_view p) { return m.substr(0, p.size()) == p; };
  auto has = [&](std::string_view p) { return m.find(p) != std::string_view::npos; };
  const MemoryAccess load{true, false, false}, store{false, true, false}, atomic{true, true, true};
  // LDS-DMA, block, add-tid, typed (format) and global transpose forms stay outside. The
  // LDS transpose loads (ds_read_b64_tr_b16, ...) are loads: each lane reads at its own
  // address and the LDS unit redistributes the elements across the lanes (lane-crossing,
  // see KernelModel's crosses_lanes).
  if (has("addtid") || has("block") || has("lds") || has("format") ||
      (has("_tr") && !(starts("ds_read_") && has("_tr_b"))))
    return std::nullopt;
  if (starts("s_load_dword") || starts("s_load_b") || starts("s_load_u") || starts("s_load_i"))
    return load;
  for (std::string_view p : {"global_", "flat_", "scratch_", "buffer_"}) {
    if (!starts(p))
      continue;
    const std::string_view rest = m.substr(p.size());
    if (rest.substr(0, 5) == "load_")
      return load;
    if (rest.substr(0, 6) == "store_")
      return store;
    if (rest.substr(0, 7) == "atomic_" && p != "scratch_")
      return atomic;
    return std::nullopt;
  }
  if (!starts("ds_") || has("src2") || has("gws") || has("ordered") || has("swizzle") ||
      has("permute") || has("append") || has("consume") || has("bvh") || has("nop") ||
      has("condxchg") || has("barrier"))
    return std::nullopt;
  if (starts("ds_read") || starts("ds_load"))
    return load;
  if (starts("ds_write_") || starts("ds_write2") || starts("ds_store"))
    return store;
  for (std::string_view p :
       {"ds_add_", "ds_sub_", "ds_rsub_", "ds_inc_", "ds_dec_", "ds_min_", "ds_max_", "ds_and_",
        "ds_or_", "ds_xor_", "ds_mskor_", "ds_wrxchg", "ds_cmpst_", "ds_cmpstore_", "ds_pk_add_"})
    if (starts(p))
      return atomic;
  return std::nullopt;
}

bool is_frame_access(std::string_view m, std::string_view text) {
  const bool buffer = m.substr(0, 7) == "buffer_", scratch = m.substr(0, 8) == "scratch_";
  if (!buffer && !scratch)
    return false;
  bool sgpr = false, vaddr = false;
  unsigned vgprs = 0;
  for (size_t p = text.find(' '); p != std::string_view::npos && p < text.size();) {
    size_t q = text.find_first_of(" ,", p + 1);
    if (q == std::string_view::npos)
      q = text.size();
    const std::string_view tok = text.substr(p + 1, q - p - 1);
    vaddr = vaddr || tok == "offen" || tok == "idxen";
    sgpr = sgpr || (tok.size() > 1 && tok[0] == 's' && tok[1] >= '0' && tok[1] <= '9');
    vgprs += tok.size() > 1 && tok[0] == 'v' && (tok[1] == '[' || (tok[1] >= '0' && tok[1] <= '9'));
    p = q;
  }
  // MUBUF: no offen/idxen means no VGPR address (LLVM prints `off`). Scratch: an SGPR
  // address and a single VGPR operand, the data or destination (LLVM prints `off` for the
  // missing VGPR address).
  return buffer ? !vaddr : sgpr && vgprs == 1;
}

bool may_read_private(std::string_view m) {
  for (std::string_view p :
       {"buffer_load", "buffer_atomic", "scratch_load", "flat_load", "flat_atomic"})
    if (m.substr(0, p.size()) == p)
      return true;
  return false;
}

bool is_region_branch(std::string_view m) {
  for (std::string_view b : {"s_branch", "s_cbranch_scc0", "s_cbranch_scc1", "s_cbranch_vccz",
                             "s_cbranch_vccnz", "s_cbranch_execz", "s_cbranch_execnz"})
    if (m == b)
      return true;
  return false;
}

bool is_region_noop(std::string_view m) {
  return std::ranges::find(kRegionNoops, m) != kRegionNoops.end();
}

bool is_region_barrier(std::string_view m) { return m == "s_barrier"; }

bool is_region_lane_permute(std::string_view m) {
  return m == "ds_bpermute_b32" || m == "ds_permute_b32";
}

bool is_lds_transpose(std::string_view m) {
  return m.substr(0, 8) == "ds_read_" && m.find("_tr_b") != std::string_view::npos;
}

bool is_region_valu(std::string_view mnemonic, uint64_t flags) {
  if (!(flags & VALU) ||
      (flags & (MEMORY_OP | SMEM | VMEM | LDS | WAITCNT | BARRIER | SUSPENDS_WAVE)))
    return false;
  // MFMA/SMFMAC and AGPR moves run (AGPRs are vector locations); interpolation (LDS
  // parameters via M0), indirect VGPR access and exception-state instructions stay outside.
  for (std::string_view p : {"v_interp", "v_movrel", "v_clrexcp", "v_swaprel"})
    if (mnemonic.substr(0, p.size()) == p)
      return false;
  return true;
}

Effects classify(const Instruction &inst, const Target &target) {
  Effects fx;
  const uint64_t flags = inst.flags();
  const std::string_view mnemonic = inst.mnemonic();
  fx.salu = (flags & SALU) != 0;
  fx.control_flow = (flags & (BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL |
                              PROGRAM_TERMINATOR | WRITES_PC | SOFTWARE_TRAP)) != 0;

  std::string first_outside;
  const bool memory_inst = is_region_memory(mnemonic);
  const std::string inst_text = memory_inst ? inst.disassemble() : std::string();
  bool unknown = false;
  std::string unknown_what;
  auto absorb = [&](const Operand &op, bool is_dst) {
    OperandClass cls = classify_operand(op, target);
    switch (cls.space) {
    case Space::Tracked:
      (is_dst ? fx.writes : fx.reads) |= cls.locs;
      if (!is_dst && cls.locs.count() == 2)
        for (unsigned loc = 0; loc < kNumLocs; ++loc)
          if (cls.locs.test(loc)) {
            fx.wide_reads.set(loc);
            break;
          }
      break;
    case Space::Vector:
      (is_dst ? fx.vwrites : fx.vreads) |= cls.vlocs;
      if (!is_dst && cls.vlocs.count() == 2)
        for (unsigned r = 0; r < kNumVgprs; ++r)
          if (cls.vlocs.test(r)) {
            fx.vwide_reads.set(r);
            break;
          }
      break;
    case Space::Pc:
      if (!is_dst)
        fx.reads_pc = true;
      if (first_outside.empty())
        first_outside = "accesses the program counter";
      break;
    case Space::Outside:
      // The memory pseudo-operand of a supported memory instruction is the memory the
      // region runner models (initial image in, stored bytes out).
      if (memory_inst && op.is_fieldless())
        break;
      // A disabled address field (MUBUF without offen/idxen, scratch without a VGPR
      // address) decodes as a register-less operand the disassembly does not show.
      if (memory_inst && cls.what.ends_with("without a register") &&
          !text_names(inst_text, op.name()))
        break;
      if (first_outside.empty())
        first_outside = "operand outside the P0 state: " + cls.what;
      break;
    case Space::Constant:
      break;
    case Space::Unknown:
      unknown = true;
      if (unknown_what.empty())
        unknown_what = cls.what;
      break;
    }
  };
  for (int i = 0; i < inst.num_src_operands(); ++i)
    absorb(*inst.src_operand(i), /*is_dst=*/false);
  for (int i = 0; i < inst.num_dst_operands(); ++i)
    absorb(*inst.dst_operand(i), /*is_dst=*/true);

  // A conditional definition preserves the old value on some paths.
  if (flags & PREDICATED_DEF)
    fx.reads |= fx.writes;

  // Hidden effects that the operand list does not show. Outside P0 they are translated
  // into tracked locations where possible (a VALU's implicit EXEC read, a carry-out VCC
  // write); VGPR/AGPR effects are not tracked state. Without this, every VALU instruction
  // would make every location live before it, and renamed dead SALU temporaries would
  // look observable. Classes that cannot be translated (TTMP, FLAT_SCRATCH, PC) stay an
  // unknown effect. P0 instructions keep the strict rule: any hidden effect rejects them.
  RegisterSet implicit_uses;
  RegisterSet implicit_defs;
  inst.implicit_uses(implicit_uses);
  inst.implicit_defs(implicit_defs);
  const bool has_implicit = !implicit_uses.none() || !implicit_defs.none();
  bool implicit_unmapped = false;
  auto translate = [&](const RegisterSet &set, LocSet &into, VSet &vinto) {
    set.for_each([&](const RegisterRef &r) {
      switch (r.cls) {
      case RegClass::SGPR:
        for (unsigned w = 0; w < std::max<unsigned>(r.width, 1); ++w) {
          const unsigned n = r.index + w;
          if (n < target.sgprs())
            into.set(n);
          else
            implicit_unmapped = true;
        }
        break;
      case RegClass::VGPR:
      case RegClass::ACC_VGPR:
        for (unsigned w = 0; w < std::max<unsigned>(r.width, 1); ++w) {
          const unsigned n = r.index + w;
          if (n < kNumArchVgprs)
            vinto.set((r.cls == RegClass::ACC_VGPR ? kAgprBase : 0u) + n);
          else
            implicit_unmapped = true;
        }
        break;
      case RegClass::EXEC: // the wave-sized mask
        add_pair(into, kLocExecLo, kLocExecHi, static_cast<int>(target.wave_size));
        break;
      case RegClass::VCC:
        add_pair(into, kLocVccLo, kLocVccHi, static_cast<int>(target.wave_size));
        break;
      case RegClass::SCC:
        into.set(kLocScc);
        break;
      case RegClass::M0:
        into.set(kLocM0);
        break;
      default:
        implicit_unmapped = true;
        break;
      }
    });
  };
  LocSet hidden_reads, hidden_writes;
  VSet hidden_vreads, hidden_vwrites;
  translate(implicit_uses, hidden_reads, hidden_vreads);
  translate(implicit_defs, hidden_writes, hidden_vwrites);
  const bool hidden_unknown = has_implicit && (is_p0_mnemonic(mnemonic) || implicit_unmapped);
  if (has_implicit && !hidden_unknown) {
    fx.reads |= hidden_reads;
    fx.writes |= hidden_writes;
    fx.vreads |= hidden_vreads;
    fx.vwrites |= hidden_vwrites;
  }
  const bool indirect_sgpr = has_mnemonic_prefix(mnemonic, "s_movrel") ||
                             has_mnemonic_prefix(mnemonic, "s_set_gpr_idx") ||
                             (flags & BRANCH_STACK) != 0;
  if (hidden_unknown || indirect_sgpr || unknown) {
    fx.unknown_effect = true;
    if (unknown_what.empty())
      unknown_what = indirect_sgpr ? "indirect SGPR access" : "implicit register effects";
  }

  // A VALU instruction reads EXEC (the lanes it writes). v_readlane/v_writelane ignore
  // it; counting it as read is conservative.
  fx.valu = (flags & VALU) != 0;
  if (fx.valu)
    add_pair(fx.reads, kLocExecLo, kLocExecHi, static_cast<int>(target.wave_size));

  // Region executability: the SALU allowlist, a plain VALU instruction, or a supported
  // memory instruction.
  const bool noop_ok = is_region_noop(mnemonic);
  const bool branch_ok = is_region_branch(mnemonic);
  const bool barrier_ok = is_region_barrier(mnemonic);
  // s_endpgm: an early exit, inside a region only when it reaches the final s_endpgm
  // (region_flow); the runner leaves through the region's end there.
  const bool endpgm_ok = mnemonic == "s_endpgm";
  const bool salu_ok = is_p0_mnemonic(mnemonic) || noop_ok || branch_ok || barrier_ok || endpgm_ok;
  const bool valu_ok = is_region_valu(mnemonic, flags);
  const bool mem_ok = is_region_memory(mnemonic);
  // Lane permutes through the LDS crossbar (no LDS memory): only active lanes write, and a
  // backward permute reads only active source lanes.
  const bool permute_ok = is_region_lane_permute(mnemonic);
  if (permute_ok)
    add_pair(fx.reads, kLocExecLo, kLocExecHi, static_cast<int>(target.wave_size));
  if (mem_ok) {
    const auto access = region_memory_access(mnemonic);
    fx.mem_read = access && access->reads;
    fx.mem_write = access && access->writes;
    // Vector memory and LDS move the lanes in EXEC.
    if ((flags & (VMEM | LDS)) != 0 || mnemonic.substr(0, 3) == "ds_")
      add_pair(fx.reads, kLocExecLo, kLocExecHi, static_cast<int>(target.wave_size));
  }
  if (!salu_ok && !valu_ok && !mem_ok && !permute_ok) {
    fx.reject_reason =
        "opcode not in the region allowlist (SALU allowlist, plain VALU, memory: global, "
        "flat, scratch, buffer, LDS, scalar loads, atomics): " +
        std::string(mnemonic);
  } else if (fx.unknown_effect) {
    fx.reject_reason = "unmodelled effect: " + unknown_what;
  } else if (!first_outside.empty() && !noop_ok) { // a hint's operands are not state
    fx.reject_reason = first_outside;
  } else if ((fx.control_flow && !branch_ok && !endpgm_ok) ||
             (!mem_ok && !noop_ok && !permute_ok &&
              (flags & (MEMORY_OP | WAITCNT | SMEM | VMEM | LDS)) != 0) ||
             ((flags & (BARRIER | SUSPENDS_WAVE)) != 0 && !barrier_ok) || (salu_ok && fx.valu)) {
    fx.reject_reason = "instruction class outside the executable region subset";
  } else if (inst.data() != nullptr) {
    fx.reject_reason = "instruction carries dynamic state";
  } else if (inst.execute == nullptr) {
    fx.reject_reason = "no execution callback (model-only provider)";
  } else {
    fx.p0 = true;
    fx.branch = branch_ok;
    fx.barrier = barrier_ok;
  }
  if (!fx.p0) {
    const bool lds = (flags & LDS) != 0 || mnemonic.find("lds") != std::string_view::npos;
    if (fx.unknown_effect) {
      fx.implicit_reads = special_locs();
    } else if (fx.salu && std::ranges::find(kNoImplicitReads, mnemonic) != kNoImplicitReads.end()) {
      // Operands name everything it reads.
    } else if ((flags & SMEM) != 0) {
      // Scalar memory: address and offset operands only.
    } else if ((flags & VMEM) != 0 || (flags & LDS) != 0) {
      add_pair(fx.implicit_reads, kLocExecLo, kLocExecHi, static_cast<int>(target.wave_size));
      if (lds)
        fx.implicit_reads.set(kLocM0);
    } else {
      fx.implicit_reads = special_locs();
    }
  }
  fx.implicit_special_reads = fx.implicit_reads.any();
  return fx;
}

namespace {

std::string_view trim(std::string_view s) {
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front())))
    s.remove_prefix(1);
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))
    s.remove_suffix(1);
  return s;
}

std::optional<unsigned> parse_uint(std::string_view s) {
  unsigned v = 0;
  const auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
  if (ec != std::errc() || p != s.data() + s.size() || s.empty())
    return std::nullopt;
  return v;
}

/// Register range "x5" or "x[4:7]" with prefix @p x; returns [first, last].
std::optional<std::pair<unsigned, unsigned>> parse_range(std::string_view op,
                                                         std::string_view prefix) {
  if (op.substr(0, prefix.size()) != prefix)
    return std::nullopt;
  std::string_view rest = op.substr(prefix.size());
  if (!rest.empty() && rest.front() == '[' && rest.back() == ']') {
    rest = rest.substr(1, rest.size() - 2);
    const size_t colon = rest.find(':');
    if (colon == std::string_view::npos)
      return std::nullopt;
    auto a = parse_uint(rest.substr(0, colon));
    auto b = parse_uint(rest.substr(colon + 1));
    if (!a || !b || *b < *a)
      return std::nullopt;
    return std::pair{*a, *b};
  }
  auto a = parse_uint(rest);
  if (!a)
    return std::nullopt;
  return std::pair{*a, *a};
}

/// One operand into slots. Returns false if not understood; sets @p untracked for AGPRs.
bool operand_slots(std::string_view op, std::vector<AsmSlot> &out, bool &untracked) {
  if (op == "vcc") {
    out.push_back({false, kLocVccLo});
    out.push_back({false, kLocVccHi});
    return true;
  }
  if (op == "exec") {
    out.push_back({false, kLocExecLo});
    out.push_back({false, kLocExecHi});
    return true;
  }
  if (auto loc = parse_loc(op); loc && *loc >= kNumSgprs) {
    out.push_back({false, *loc}); // vcc_lo, vcc_hi, exec_lo, exec_hi, m0, scc
    return true;
  }
  if (auto r = parse_range(op, "v")) {
    if (r->second >= kNumVgprs)
      return false;
    for (unsigned i = r->first; i <= r->second; ++i)
      out.push_back({true, i});
    return true;
  }
  if (auto r = parse_range(op, "s")) {
    if (r->second >= kNumSgprs)
      return false;
    for (unsigned i = r->first; i <= r->second; ++i)
      out.push_back({false, i});
    return true;
  }
  if (parse_range(op, "a") || parse_range(op, "agpr")) {
    untracked = true;
    return true;
  }
  return false;
}

/// True if @p line names something that looks like a register.
bool names_register(std::string_view line) {
  for (size_t i = 0; i < line.size(); ++i) {
    const bool start = i == 0 || !std::isalnum(static_cast<unsigned char>(line[i - 1]));
    if (!start)
      continue;
    const std::string_view rest = line.substr(i);
    for (std::string_view w : {"vcc", "exec", "m0", "scc"})
      if (rest.substr(0, w.size()) == w)
        return true;
    if ((rest[0] == 'v' || rest[0] == 's' || rest[0] == 'a') && rest.size() > 1 &&
        (std::isdigit(static_cast<unsigned char>(rest[1])) || rest[1] == '['))
      return true;
  }
  return false;
}

} // namespace

InlineAsmEffect parse_inline_asm(std::string_view text) {
  InlineAsmEffect e;
  size_t pos = 0;
  while (pos <= text.size()) {
    size_t nl = text.find('\n', pos);
    if (nl == std::string_view::npos)
      nl = text.size();
    std::string_view line = trim(text.substr(pos, nl - pos));
    pos = nl + 1;
    if (line.empty())
      continue;
    if (line.front() != ';') {
      e.opaque = true;
      e.reason = "inline asm line is not a comment: '" + std::string(line) + "'";
      return e;
    }
    while (!line.empty() && line.front() == ';')
      line.remove_prefix(1);
    line = trim(line);
    const size_t sp = line.find_first_of(" \t");
    const std::string_view verb = line.substr(0, sp);
    if (verb != "def" && verb != "use") {
      if (names_register(line)) {
        e.opaque = true;
        e.reason = "inline asm comment not understood: '" + std::string(line) + "'";
        return e;
      }
      continue;
    }
    std::string_view ops = sp == std::string_view::npos ? std::string_view{} : line.substr(sp);
    auto &slots = verb == "def" ? e.defs : e.uses;
    while (!trim(ops).empty()) {
      const size_t comma = ops.find(',');
      const std::string_view op = trim(ops.substr(0, comma));
      ops = comma == std::string_view::npos ? std::string_view{} : ops.substr(comma + 1);
      if (!operand_slots(op, slots, e.untracked)) {
        e.opaque = true;
        e.reason = "inline asm operand not understood: '" + std::string(op) + "'";
        return e;
      }
    }
  }
  if (e.untracked)
    e.reason = "inline asm names AGPRs, which are not tracked";
  return e;
}

Effects classify_inline_asm(std::string_view text) {
  Effects fx;
  fx.implicit_special_reads = false;
  const InlineAsmEffect e = parse_inline_asm(text);
  if (e.opaque) {
    fx.unknown_effect = true;
    fx.reject_reason = e.reason;
    return fx;
  }
  for (const AsmSlot &s : e.uses) {
    if (s.vgpr) {
      fx.vreads.set(s.index);
      fx.reads.set(kLocExecLo);
      fx.reads.set(kLocExecHi);
    } else {
      fx.reads.set(s.index);
    }
  }
  for (const AsmSlot &s : e.defs) {
    if (s.vgpr) {
      fx.vwrites.set(s.index); // the lanes in EXEC
      fx.reads.set(kLocExecLo);
      fx.reads.set(kLocExecHi);
    } else {
      fx.writes.set(s.index);
    }
  }
  fx.p0 = !e.untracked;
  if (e.untracked)
    fx.reject_reason = e.reason;
  return fx;
}

std::string inline_asm_shape(std::string_view text) {
  const InlineAsmEffect e = parse_inline_asm(text);
  if (e.opaque)
    return "opaque:" + std::string(text);
  auto shape = [](const std::vector<AsmSlot> &slots) {
    std::string s;
    for (const AsmSlot &a : slots)
      s += a.vgpr ? 'v' : (a.index < kNumSgprs ? 's' : 'x');
    return s;
  };
  return "use " + shape(e.uses) + ";def " + shape(e.defs) + (e.untracked ? ";agpr" : "");
}

// ---------------------------------------------------------------------------
// Floating-point provenance and poison analogues.

namespace {

std::string_view strip_encoding(std::string_view m) {
  for (std::string_view suffix : {"_e64_dpp", "_e32", "_e64", "_sdwa", "_dpp"})
    if (m.size() > suffix.size() && m.substr(m.size() - suffix.size()) == suffix)
      return m.substr(0, m.size() - suffix.size());
  return m;
}

bool starts(std::string_view s, std::string_view p) { return s.substr(0, p.size()) == p; }
bool ends(std::string_view s, std::string_view p) {
  return s.size() >= p.size() && s.substr(s.size() - p.size()) == p;
}

/// The last element of "op_sel:[a,b,...]" (VOP3: the destination half), or -1.
int op_sel_dst(std::string_view text) {
  const size_t at = text.find("op_sel:[");
  if (at == std::string_view::npos)
    return -1;
  const size_t close = text.find(']', at);
  if (close == std::string_view::npos || close == 0)
    return -1;
  const char c = text[close - 1];
  return c == '1' ? 1 : c == '0' ? 0 : -1;
}

/// The destination is the high half of a true16 register ("v_add_f16 v0.h, ..." on GFX11+).
bool true16_dst_hi(std::string_view text) {
  const size_t sp = text.find(' ');
  if (sp == std::string_view::npos)
    return false;
  std::string_view dst = text.substr(sp + 1);
  dst = dst.substr(0, dst.find(','));
  while (!dst.empty() && dst.back() == ' ')
    dst.remove_suffix(1);
  return dst.size() > 2 && dst.substr(dst.size() - 2) == ".h";
}

FpResult half_result(std::string_view base, std::string_view text, bool bf16) {
  const bool hi = text.find("dst_sel:WORD_1") != std::string_view::npos ||
                  (!starts(base, "v_pk_") && op_sel_dst(text) == 1) || true16_dst_hi(text);
  if (bf16)
    return hi ? FpResult::BF16Hi : FpResult::BF16Lo;
  return hi ? FpResult::F16Hi : FpResult::F16Lo;
}

FpResult format_suffix(std::string_view base, std::string_view text) {
  if (ends(base, "_bf16"))
    return half_result(base, text, true);
  if (ends(base, "_f16"))
    return half_result(base, text, false);
  if (ends(base, "_f32"))
    return FpResult::F32;
  if (ends(base, "_f64"))
    return FpResult::F64;
  return FpResult::None;
}

} // namespace

FpResult fp_result_kind(std::string_view mnemonic, std::string_view text) {
  const std::string_view base = strip_encoding(mnemonic);
  if (!starts(base, "v_"))
    return FpResult::None;
  const bool lane_shuffle = base != mnemonic && (ends(mnemonic, "_sdwa") || ends(mnemonic, "_dpp"));
  if (base == "v_mov_b32" || base == "v_mov_b64")
    return lane_shuffle || text.find("row_") != std::string_view::npos ? FpResult::None
                                                                       : FpResult::Move;
  if (base == "v_cndmask_b32")
    return lane_shuffle ? FpResult::None : FpResult::CndMask;
  if (base == "v_pk_mov_b32")
    return FpResult::PkMov;
  if (starts(base, "v_cmp") || starts(base, "v_frexp_exp") || base.find("mfma") != base.npos ||
      base.find("smfmac") != base.npos)
    return FpResult::None;
  if (starts(base, "v_cvt_")) {
    const std::string_view rest = base.substr(6);
    if (starts(rest, "pkrtz_f16") || starts(rest, "pk_f16_f32"))
      return FpResult::F16x2;
    if (starts(rest, "pk_bf16_f32"))
      return FpResult::BF16x2;
    if (starts(rest, "f32_"))
      return FpResult::F32;
    if (starts(rest, "f64_"))
      return FpResult::F64;
    if (starts(rest, "f16_"))
      return half_result(base, text, false);
    if (starts(rest, "bf16_"))
      return half_result(base, text, true);
    return FpResult::None; // to integers, packed normalised/fp8 forms
  }
  if (starts(base, "v_pk_")) {
    if (ends(base, "_bf16"))
      return FpResult::BF16x2;
    if (ends(base, "_f16"))
      return FpResult::F16x2;
    if (ends(base, "_f32"))
      return FpResult::F32; // a pair of f32 results
    return FpResult::None;
  }
  if (base == "v_fma_mix_f32" || base == "v_mad_mix_f32")
    return FpResult::F32;
  if (base == "v_fma_mixlo_f16" || base == "v_mad_mixlo_f16")
    return FpResult::F16Lo;
  if (base == "v_fma_mixhi_f16" || base == "v_mad_mixhi_f16")
    return FpResult::F16Hi;
  if (starts(base, "v_dot"))
    return starts(base, "v_dot2_f32_") || starts(base, "v_dot2c_f32_") ? FpResult::F32
                                                                       : FpResult::None;
  return format_suffix(base, text);
}

FpResult fp_scalar_result_kind(std::string_view mnemonic) {
  const std::string_view b = strip_encoding(mnemonic);
  if (b == "s_mov_b32" || b == "s_mov_b64")
    return FpResult::Move;
  if (b == "s_cselect_b32" || b == "s_cselect_b64")
    return FpResult::Select;
  if (b == "s_cmov_b32" || b == "s_cmov_b64")
    return FpResult::CMov;
  if (b == "v_readfirstlane_b32")
    return FpResult::ReadFirstLane;
  if (b == "v_readlane_b32")
    return FpResult::ReadLane;
  if (!starts(b, "s_") || starts(b, "s_cmp"))
    return FpResult::None;
  if (b == "s_cvt_pk_rtz_f16_f32")
    return FpResult::F16x2;
  if (b == "s_cvt_f16_f32")
    return FpResult::F16Lo;
  if (starts(b, "s_cvt_"))
    return starts(b, "s_cvt_f32_") || starts(b, "s_cvt_hi_f32_") ? FpResult::F32 : FpResult::None;
  if (ends(b, "_f32"))
    return FpResult::F32;
  if (ends(b, "_f16"))
    return FpResult::F16Lo;
  return FpResult::None;
}

FpResult fp_source_kind(std::string_view mnemonic) {
  const std::string_view base = strip_encoding(mnemonic);
  if (starts(base, "s_") && !starts(base, "s_cmp")) { // SALU float (GFX11.5+)
    if (base == "s_cvt_f32_f16")
      return FpResult::F16Lo;
    if (base == "s_cvt_hi_f32_f16")
      return FpResult::F16Hi;
    if (starts(base, "s_cvt_"))
      return ends(base, "_f32") ? FpResult::F32 : FpResult::None;
    if (ends(base, "_f32"))
      return FpResult::F32;
    if (ends(base, "_f16"))
      return FpResult::F16Lo;
    return FpResult::None;
  }
  if (starts(base, "s_cmp_") && (ends(base, "_f32") || ends(base, "_f16")))
    return ends(base, "_f32") ? FpResult::F32 : FpResult::F16Lo;
  if (!starts(base, "v_") || base.find("mfma") != base.npos || starts(base, "v_fma_mix") ||
      starts(base, "v_mad_mix"))
    return FpResult::None;
  if (starts(base, "v_cvt_")) {
    if (starts(base, "v_cvt_pkrtz_f16_f32") || starts(base, "v_cvt_pk_f16_f32") ||
        starts(base, "v_cvt_pk_bf16_f32"))
      return FpResult::F32;
    if (ends(base, "_f32"))
      return FpResult::F32;
    if (ends(base, "_f16"))
      return FpResult::F16Lo;
    if (ends(base, "_bf16"))
      return FpResult::BF16Lo;
    if (ends(base, "_f64"))
      return FpResult::F64;
    return FpResult::None;
  }
  if (starts(base, "v_pk_")) {
    if (ends(base, "_bf16"))
      return FpResult::BF16x2;
    if (ends(base, "_f16"))
      return FpResult::F16x2;
    if (ends(base, "_f32"))
      return FpResult::F32;
    return FpResult::None;
  }
  if (starts(base, "v_dot2_f32_f16") || starts(base, "v_dot2c_f32_f16"))
    return FpResult::F16x2;
  if (starts(base, "v_dot2_f32_bf16") || starts(base, "v_dot2c_f32_bf16"))
    return FpResult::BF16x2;
  if (ends(base, "_bf16"))
    return FpResult::BF16Lo;
  if (ends(base, "_f16"))
    return FpResult::F16Lo;
  if (ends(base, "_f32"))
    return FpResult::F32;
  if (ends(base, "_f64"))
    return FpResult::F64;
  return FpResult::None;
}

namespace {

constexpr std::pair<std::string_view, uint32_t> kPoisonNames[] = {
    {"shift", kPoisonShift}, {"fptoint", kPoisonFpToInt}, {"ctz", kPoisonCtz},
    {"nsw", kPoisonNsw},     {"nuw", kPoisonNuw},         {"disjoint", kPoisonDisjoint},
    {"exact", kPoisonExact}, {"nnan", kPoisonNnan},       {"ninf", kPoisonNinf},
};

} // namespace

std::optional<uint32_t> parse_poison_kinds(std::string_view text) {
  uint32_t out = 0;
  while (!text.empty()) {
    const size_t comma = text.find(',');
    const std::string_view name = trim(text.substr(0, comma));
    text = comma == std::string_view::npos ? std::string_view{} : text.substr(comma + 1);
    if (name.empty())
      continue;
    if (name == "all") {
      for (const auto &[n, k] : kPoisonNames)
        out |= k;
      continue;
    }
    bool found = false;
    for (const auto &[n, k] : kPoisonNames)
      if (n == name) {
        out |= k;
        found = true;
      }
    if (!found)
      return std::nullopt;
  }
  return out;
}

std::string format_poison_kinds(uint32_t kinds) {
  std::string s;
  for (const auto &[n, k] : kPoisonNames)
    if (kinds & k)
      s += (s.empty() ? "" : ",") + std::string(n);
  return s;
}

uint32_t poison_check_kind(std::string_view mnemonic) {
  const std::string_view b = strip_encoding(mnemonic);
  for (std::string_view s : {"v_lshlrev_b32", "v_lshlrev_b64", "v_lshlrev_b16", "v_pk_lshlrev_b16",
                             "s_lshl_b32", "s_lshl_b64"})
    if (b == s)
      return kPoisonShift;
  for (std::string_view s :
       {"v_lshrrev_b32", "v_ashrrev_i32", "v_lshrrev_b64", "v_ashrrev_i64", "v_lshrrev_b16",
        "v_ashrrev_i16", "v_pk_lshrrev_b16", "v_pk_ashrrev_i16", "s_lshr_b32", "s_ashr_i32",
        "s_lshr_b64", "s_ashr_i64"})
    if (b == s)
      return kPoisonShift | kPoisonExact;
  for (std::string_view s :
       {"v_cvt_i32_f32", "v_cvt_u32_f32", "v_cvt_i32_f64", "v_cvt_u32_f64", "v_cvt_i16_f16",
        "v_cvt_u16_f16", "v_cvt_flr_i32_f32", "v_cvt_rpi_i32_f32", "v_cvt_floor_i32_f32",
        "v_cvt_nearest_i32_f32", "s_cvt_i32_f32", "s_cvt_u32_f32"})
    if (b == s)
      return kPoisonFpToInt | kPoisonNnan | kPoisonNinf;
  for (std::string_view s : {"v_ffbh_u32", "v_ffbl_b32", "s_flbit_i32_b32", "s_flbit_i32_b64",
                             "s_ff1_i32_b32", "s_ff1_i32_b64", "v_clz_i32_u32", "v_ctz_i32_b32",
                             "s_clz_i32_u32", "s_clz_i32_u64", "s_ctz_i32_b32", "s_ctz_i32_b64"})
    if (b == s)
      return kPoisonCtz;
  for (std::string_view s : {"v_add_u32", "v_sub_u32", "v_subrev_u32", "v_mul_lo_u32", "s_add_i32",
                             "s_sub_i32", "s_mul_i32", "v_add_nc_u32", "v_sub_nc_u32",
                             "v_subrev_nc_u32", "s_add_co_i32", "s_sub_co_i32"})
    if (b == s)
      return kPoisonNsw | kPoisonNuw;
  if (b == "v_or_b32" || b == "s_or_b32")
    return kPoisonDisjoint;
  if (fp_source_kind(mnemonic) != FpResult::None)
    return kPoisonNnan | kPoisonNinf;
  return 0;
}

uint32_t nan_residual(uint32_t a, uint32_t b, uint8_t tag_a, uint8_t tag_b) {
  uint32_t diff = a ^ b;
  const uint8_t t = tag_a & tag_b;
  if (!diff || !t)
    return diff;
  if (t & kFpF32)
    return is_nan_f32(a) && is_nan_f32(b) ? 0u : diff;
  for (unsigned h = 0; h < 2; ++h) {
    const uint32_t m = h ? 0xffff0000u : 0x0000ffffu;
    if (!(diff & m))
      continue;
    const uint32_t ha = (a & m) >> (16 * h), hb = (b & m) >> (16 * h);
    const uint8_t f16 = h ? kFpF16Hi : kFpF16Lo, bf16 = h ? kFpBF16Hi : kFpBF16Lo;
    if (((t & f16) && is_nan_f16(ha) && is_nan_f16(hb)) ||
        ((t & bf16) && is_nan_bf16(ha) && is_nan_bf16(hb)))
      diff &= ~m;
  }
  return diff;
}

} // namespace rocjitsu::isa_diff
