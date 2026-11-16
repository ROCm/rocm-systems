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

// Integer SALU instructions whose execution RocJITsu implements on gfx950 and whose
// entire effect is on SGPRs, VCC, EXEC, M0 and SCC. Every operand is still checked;
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
});

// SALU instructions outside P0 that read no tracked state beyond their classified
// operands (the branch conditions SCC/VCC/EXEC are fieldless operands; see tests).
constexpr std::array kNoImplicitReads = std::to_array<std::string_view>({
    "s_endpgm", "s_waitcnt", "s_barrier", "s_branch", "s_cbranch_scc0", "s_cbranch_scc1",
    "s_cbranch_vccz", "s_cbranch_vccnz", "s_cbranch_execz", "s_cbranch_execnz", "s_sleep",
    "s_setprio",
    "s_setpc_b64", // reads its target only; a return's live-out is the calling convention's
});

// Generic scalar operand selector values (gfx9 encoding of SSRC/SDST fields).
constexpr int kSelFlatScratchLo = 102;
constexpr int kSelXnackMaskHi = 105;
constexpr int kSelVccLo = 106;
constexpr int kSelVccHi = 107;
constexpr int kSelM0 = 124;
constexpr int kSelSharedBase = 235;        ///< src_shared_base (gfx9)
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

OperandClass classify_special(RegClass cls, int bits) {
  OperandClass out;
  out.space = Space::Tracked;
  switch (cls) {
  case RegClass::SCC:
    out.locs.set(kLocScc);
    return out;
  case RegClass::VCC:
    // A fieldless VCC operand is the wave-sized mask; on wave64 that is the pair.
    add_pair(out.locs, kLocVccLo, kLocVccHi, std::max(bits, 64));
    return out;
  case RegClass::EXEC:
    add_pair(out.locs, kLocExecLo, kLocExecHi, std::max(bits, 64));
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

OperandClass classify_operand(const Operand &op) {
  OperandClass out;
  if (op.is_fieldless()) {
    if (auto cls = op.to_special_reg_class())
      return classify_special(*cls, op.size_bits());
    out.space = Space::Outside;
    out.what = "fieldless operand '" + op.name() + "'";
    return out;
  }
  if (auto ref = op.to_register_ref()) {
    switch (ref->cls) {
    case RegClass::SGPR:
      if (ref->index + ref->width > kNumSgprs) {
        out.space = Space::Unknown;
        out.what = "SGPR beyond s101";
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
      return classify_special(ref->cls, op.size_bits());
    case RegClass::PC:
      out.space = Space::Pc;
      out.what = "pc";
      return out;
    case RegClass::VGPR:
      if (ref->index + std::max<unsigned>(ref->width, 1) > kNumVgprs) {
        out.space = Space::Unknown;
        out.what = "VGPR beyond v255";
        return out;
      }
      out.space = Space::Vector;
      for (unsigned i = 0; i < std::max<unsigned>(ref->width, 1); ++i)
        out.vlocs.set(ref->index + i);
      return out;
    case RegClass::ACC_VGPR:
      out.space = Space::Outside;
      out.what = "accumulation register (AGPR)";
      return out;
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
    const int bits = op.size_bits();
    out.space = Space::Tracked;
    switch (sel) {
    case kSelVccLo:
      add_pair(out.locs, kLocVccLo, kLocVccHi, bits);
      return out;
    case kSelVccHi:
      out.locs.set(kLocVccHi);
      return out;
    case kSelM0:
      out.locs.set(kLocM0);
      return out;
    case kSelScc:
      out.locs.set(kLocScc);
      return out;
    case kSelVccz:
      out.locs.set(kLocVccLo);
      out.locs.set(kLocVccHi);
      return out;
    case kSelExecz:
      out.locs.set(kLocExecLo);
      out.locs.set(kLocExecHi);
      return out;
    default:
      break;
    }
    if (sel >= kSelFlatScratchLo && sel <= kSelXnackMaskHi) {
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

} // namespace

std::vector<SourceOperand> source_operands(const Instruction &inst) {
  std::vector<SourceOperand> out;
  for (int i = 0; i < inst.num_src_operands(); ++i) {
    const Operand &op = *inst.src_operand(i);
    SourceOperand s;
    s.fieldless = op.is_fieldless();
    const OperandClass cls = classify_operand(op);
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
  if (m == "s_waitcnt")
    return true;
  if (starts("global_load_"))
    return m.find("lds") == std::string_view::npos && m.find("_tr") == std::string_view::npos;
  return starts("global_store_") || starts("s_load_dword");
}

bool is_region_valu(std::string_view mnemonic, uint64_t flags) {
  if (!(flags & VALU) ||
      (flags & (MFMA | MEMORY_OP | SMEM | VMEM | LDS | WAITCNT | BARRIER | SUSPENDS_WAVE)))
    return false;
  // Interpolation (LDS parameters via M0), indirect VGPR access, MFMA-family and
  // exception-state instructions stay outside.
  for (std::string_view p :
       {"v_interp", "v_movrel", "v_mfma", "v_smfmac", "v_clrexcp", "v_accvgpr", "v_swaprel"})
    if (mnemonic.substr(0, p.size()) == p)
      return false;
  return true;
}

Effects classify(const Instruction &inst) {
  Effects fx;
  const uint64_t flags = inst.flags();
  const std::string_view mnemonic = inst.mnemonic();
  fx.salu = (flags & SALU) != 0;
  fx.control_flow = (flags & (BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL |
                              PROGRAM_TERMINATOR | WRITES_PC | SOFTWARE_TRAP)) != 0;

  std::string first_outside;
  const bool memory_inst = is_region_memory(mnemonic);
  bool unknown = false;
  std::string unknown_what;
  auto absorb = [&](const Operand &op, bool is_dst) {
    OperandClass cls = classify_operand(op);
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
          if (n < kNumSgprs)
            into.set(n);
          else
            implicit_unmapped = true;
        }
        break;
      case RegClass::VGPR:
        for (unsigned w = 0; w < std::max<unsigned>(r.width, 1); ++w) {
          const unsigned n = r.index + w;
          if (n < kNumVgprs)
            vinto.set(n);
          else
            implicit_unmapped = true;
        }
        break;
      case RegClass::ACC_VGPR:
        break;
      case RegClass::EXEC:
        into.set(kLocExecLo);
        into.set(kLocExecHi);
        break;
      case RegClass::VCC:
        into.set(kLocVccLo);
        into.set(kLocVccHi);
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
  if (fx.valu) {
    fx.reads.set(kLocExecLo);
    fx.reads.set(kLocExecHi);
  }

  // Region executability: the SALU allowlist, a plain VALU instruction, or a supported
  // memory instruction.
  const bool salu_ok = is_p0_mnemonic(mnemonic);
  const bool valu_ok = is_region_valu(mnemonic, flags);
  const bool mem_ok = is_region_memory(mnemonic);
  if (mem_ok) {
    fx.mem_read = mnemonic.substr(0, 12) == "global_load_" || mnemonic.substr(0, 6) == "s_load";
    fx.mem_write = mnemonic.substr(0, 13) == "global_store_";
    if ((flags & VMEM) != 0) { // vector memory moves the lanes in EXEC
      fx.reads.set(kLocExecLo);
      fx.reads.set(kLocExecHi);
    }
  }
  if (!salu_ok && !valu_ok && !mem_ok) {
    fx.reject_reason =
        "opcode not in the region allowlist (SALU allowlist, plain VALU, global/scalar memory): " +
        std::string(mnemonic);
  } else if (fx.unknown_effect) {
    fx.reject_reason = "unmodelled effect: " + unknown_what;
  } else if (!first_outside.empty()) {
    fx.reject_reason = first_outside;
  } else if (fx.control_flow ||
             (!mem_ok && (flags & (MEMORY_OP | WAITCNT | SMEM | VMEM | LDS)) != 0) ||
             (flags & (BARRIER | SUSPENDS_WAVE | MFMA)) != 0 || (salu_ok && fx.valu)) {
    fx.reject_reason = "instruction class outside the executable region subset";
  } else if (inst.data() != nullptr) {
    fx.reject_reason = "instruction carries dynamic state";
  } else if (inst.execute == nullptr) {
    fx.reject_reason = "no execution callback (model-only provider)";
  } else {
    fx.p0 = true;
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
      fx.implicit_reads.set(kLocExecLo);
      fx.implicit_reads.set(kLocExecHi);
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

} // namespace rocjitsu::isa_diff
