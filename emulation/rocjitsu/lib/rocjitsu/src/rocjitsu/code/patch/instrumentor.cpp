// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/instrumentor.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/analysis/exec_state.h"
#include "rocjitsu/code/analysis/liveness.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/code_object.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/error_report.h"
#include "rocjitsu/code/patch/kernel_text_layout.h"
#include "rocjitsu/code/patch/probe_callable.h"
#include "rocjitsu/code/patch/probe_clobber.h"
#include "rocjitsu/code/patch/probe_live_in.h"
#include "rocjitsu/code/patch/probe_symbol.h"
#include "rocjitsu/code/patch/trampoline_builder.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"
#include "util/except.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocjitsu {

namespace {

// PC-reading / PC-relative instructions to reject as anchors: relocating any of
// them into a trampoline changes the PC value they read (s_getpc) or the target
// they branch to. s_getpc and the s_rfe_ family carry no control-flow flag or
// branch_offset_bytes, so validate_anchor's flag/offset checks miss them and the
// denylist is the only thing that catches them. The s_call / s_setpc / s_swappc
// entries are already rejected upstream by their INDIRECT_BRANCH / INDIRECT_CALL
// flags and are listed only as defense-in-depth. gfx1250 renames the whole
// family from *_b64 to *_i64 (e.g. s_getpc_b64 -> s_get_pc_i64), so both
// spellings are listed; the s_rfe_ prefix match below covers s_rfe_b64,
// s_rfe_i64, and s_rfe_restore_b64.
constexpr std::array<std::string_view, 8> kPcRelativeDenylist = {
    "s_getpc_b64",  "s_call_b64", "s_setpc_b64",  "s_swappc_b64",
    "s_get_pc_i64", "s_call_i64", "s_set_pc_i64", "s_swap_pc_i64",
};

constexpr std::string_view kRfePrefix = "s_rfe_";

[[nodiscard]] bool is_denylisted_mnemonic(std::string_view mnemonic) {
  for (auto m : kPcRelativeDenylist)
    if (mnemonic == m)
      return true;
  if (mnemonic.size() >= kRfePrefix.size() && mnemonic.substr(0, kRfePrefix.size()) == kRfePrefix)
    return true;
  return false;
}

[[nodiscard]] bool is_s_clause(const Instruction &inst) { return inst.mnemonic() == "s_clause"; }

[[nodiscard]] uint32_t s_clause_following_instruction_count(const Instruction &inst) {
  if (!is_s_clause(inst) || inst.size() != sizeof(uint32_t) || inst.num_src_operands() != 1)
    return 0;

  const Operand *clause = inst.src_operand(0);
  if (clause == nullptr)
    return 0;

  // The decoded OPR_CLAUSE retains the packed SIMM16 value. SIMM16[5:0]
  // encodes one less than the number of following instructions; bits 8:11 are
  // BREAK_SPAN and are not part of the count.
  return (static_cast<uint32_t>(clause->encoding_value()) & 0x3fu) + 1u;
}

// Per-site result of Instrumentor::patch's preflight: the chosen trampoline
// offset and the concrete bytes we'll splice in once all preflights succeed.
struct AppliedSite {
  const ResolvedInstrumentationSite *site;
  uint64_t trampoline_offset;
  TrampolineBytes bytes;

  // Probe-call facts for the patch record; defaulted for the inline-nop site.
  bool is_probe_call = false;
  uint64_t probe_target_offset = 0;
  uint16_t link_pair_base = 0;
  uint16_t target_pair_base = 0;
};

// Human-readable single-lane register name for spill diagnostics.
std::string reg_name(RegisterRef ref) {
  const char *prefix = "?";
  switch (ref.cls) {
  case RegClass::SGPR:
    prefix = "s";
    break;
  case RegClass::VGPR:
    prefix = "v";
    break;
  case RegClass::ACC_VGPR:
    prefix = "acc";
    break;
  default:
    break;
  }
  return std::string(prefix) + std::to_string(ref.index);
}

// Largest positive byte offset encodable in the scratch store/load offset field,
// per arch. 0 means the arch has no scratch spill emitter (spilling unsupported).
// The fields are signed, so the cap is the positive half: the CDNA FLAT path
// writes the 12-bit offset (pad_12 left 0), RDNA4 has a signed 24-bit VSCRATCH
// ioffset.
uint32_t max_scratch_offset_bytes(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA3:
  case ROCJITSU_CODE_ARCH_CDNA4:
    return 0xFFF; // 12-bit, pad_12 = 0
  case ROCJITSU_CODE_ARCH_RDNA4:
    return 0x7FFFFF; // positive half of signed 24-bit
  default:
    return 0;
  }
}

// TODO: these two arch predicates duplicate logic that also lives in DBT
// (kernel_descriptor_translator.cpp's arch_has_accvgpr / uses_gfx90a_accum_offset)
// and code_object_patcher.cpp (target_uses_gfx90a_accum_offset). They should be
// consolidated into isa/isa_traits.h alongside arch_is_cdna_4_or_lower/arch_is_rdna,
// but the
// arch_has_accvgpr copies disagree on CDNA1 (this one follows the physical AGPR file;
// DBT's follows the HasAccVgpr trait, which models CDNA1 as zero), so unifying needs a
// deliberate CDNA1-semantics decision. Kept DBI-local until then.

// All CDNA generations (gfx908/gfx90a/gfx942/gfx950) have an AccVGPR file.
// RDNA/gfx1250 have none.
bool arch_has_accvgpr(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
  case ROCJITSU_CODE_ARCH_CDNA2:
  case ROCJITSU_CODE_ARCH_CDNA3:
  case ROCJITSU_CODE_ARCH_CDNA4:
    return true;
  default:
    return false;
  }
}

// CDNA2-4 (gfx90a/gfx942/gfx950) carve the AccVGPR file out of a single unified VGPR
// allocation, split at the descriptor's ACCUM_OFFSET field. CDNA1/gfx908 allocates
// AGPRs separately and has no such field; RDNA and gfx1250 have no AccVGPR file at
// all.
bool arch_has_unified_vgpr_allocation(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA2:
  case ROCJITSU_CODE_ARCH_CDNA3:
  case ROCJITSU_CODE_ARCH_CDNA4:
    return true;
  case ROCJITSU_CODE_ARCH_CDNA1:
  case ROCJITSU_CODE_ARCH_RDNA1:
  case ROCJITSU_CODE_ARCH_RDNA2:
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
  case ROCJITSU_CODE_ARCH_RDNA4:
  case ROCJITSU_CODE_ARCH_CDNA5:
    return false;
  default:
    throw util::UnimplementedInst("unified VGPR allocation for target architecture");
  }
}

// How one kernel's VGPR allocation divides into an ordinary prefix and an
// AccVGPR window.
struct KernelVgprBounds {
  uint32_t total = 0;          // Allocated VGPRs: ordinary prefix plus AccVGPR window.
  uint32_t ordinary_bound = 0; // One past the last ordinary VGPR.
  uint32_t acc_count = 0;      // AccVGPRs in the window; 0 when there is none.
};

// Decode @p desc's VGPR allocation for @p arch.
KernelVgprBounds kernel_vgpr_bounds(rj_code_arch_t arch,
                                    const rocr::llvm::amdhsa::kernel_descriptor_t &desc) {
  const uint32_t granulated = AMDHSA_BITS_GET(
      desc.compute_pgm_rsrc1, rocr::llvm::amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  // The descriptor encoding granule is wave-size dependent on RDNA (8 for
  // Wave32, 4 for Wave64); using the Wave32 granule for a Wave64 kernel would
  // overcount the allocation and let the SGPR bridge scan pick an unallocated
  // VGPR. Share the wave-aware decoder with DBT so the two cannot diverge.
  const uint32_t total = (granulated + 1) * descriptor_vgpr_granularity_for_wavefront(
                                                arch, kernel_wavefront_size(arch, desc));
  // On a unified-allocation arch the VGPR allocation splits at the ACCUM_OFFSET
  // base ((encoded+1)*4) into an ordinary-VGPR prefix and the AccVGPR window.
  // Arches without that split (non-CDNA, and CDNA1/gfx908 whose AGPRs allocate
  // separately) have no ACCUM_OFFSET field, so the whole allocation is ordinary.
  const uint32_t accum_base =
      arch_has_unified_vgpr_allocation(arch)
          ? (AMDHSA_BITS_GET(desc.compute_pgm_rsrc3,
                             rocr::llvm::amdhsa::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET) +
             1) *
                4
          : total;
  return KernelVgprBounds{
      .total = total,
      // An ordinary VGPR is one below the accumulator window: an index inside it
      // would alias an AGPR.
      .ordinary_bound = std::min(total, accum_base),
      .acc_count = total > accum_base ? total - accum_base : 0,
  };
}

// Special machine state preserved across a probe call: SCC (via the trampoline
// envelope), EXEC/VCC/M0 (saved to a dead SGPR temp; the orchestrator sets the
// plan.preserve_* flags below), and ordinary GPRs (via the spill policy).
// FLAT_SCRATCH stays rejected: the spill store/load depend on it, so a probe
// that clobbers it entangles with the spill mechanism. Fail closed there rather
// than let it silently corrupt the host kernel's state.
bool check_probe_special_state(const ProbeClobberSummary &summary, std::string *error_out) {
  if (summary.touches_flat_scratch) {
    report(error_out, "probe body writes FLAT_SCRATCH, which the spill store/load depend on; "
                      "not preservable across a probe call");
    return false;
  }
  return true;
}

// Check that the probe does not clobber the link pair.
bool check_probe_link_pair(const ProbeClobberSummary &summary, const ProbeAbi &abi,
                           std::string *error_out) {
  if (!is_valid_probe_abi(abi))
    return true; // plan_probe_call rejects an unusable ABI with its own error.
  if (!summary.ordinary_clobbers.intersects(probe_link_pair(abi)))
    return true;
  if (error_out != nullptr) {
    const uint16_t hi = static_cast<uint16_t>(abi.link_pair_base + 1);
    *error_out = "probe body overwrites its own return-link pair s[" +
                 std::to_string(abi.link_pair_base) + ":" + std::to_string(hi) +
                 "] before returning; it would return through a corrupted PC";
  }
  return false;
}

} // namespace

bool is_relocatable_anchor(const Instruction &anchor, uint64_t anchor_offset,
                           std::span<const uint8_t> text_bytes,
                           [[maybe_unused]] rj_code_arch_t arch, std::string *error_out) {
  if (anchor_offset % sizeof(uint32_t) != 0) {
    report(error_out, "anchor_offset must be dword aligned");
    return false;
  }
  // Instruction::size() returns int by convention; the `!= 4 && != 8` check
  // also rejects negative values (which decoders never produce in practice).
  const int size = anchor.size();
  if (size != 4 && size != 8) {
    report(error_out, "anchor instruction size must be 4 or 8 bytes");
    return false;
  }
  // Subtraction-based bounds check: a huge anchor_offset would otherwise wrap
  // the addition and silently pass.
  const uint64_t size_u = static_cast<uint64_t>(size);
  if (anchor_offset > text_bytes.size() || size_u > text_bytes.size() - anchor_offset) {
    report(error_out, "anchor extends past end of .text");
    return false;
  }
  if (anchor.raw_encoding() == nullptr) {
    report(error_out, "anchor instruction has no raw encoding bytes");
    return false;
  }
  constexpr uint64_t kControlFlowFlags =
      BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL | PROGRAM_TERMINATOR;
  if (anchor.flags() & kControlFlowFlags) {
    report(error_out, "anchor instruction is a branch / indirect / program terminator");
    return false;
  }
  if (anchor.branch_offset_bytes().has_value()) {
    report(error_out, "anchor instruction has a PC-relative branch offset");
    return false;
  }
  if (is_denylisted_mnemonic(anchor.mnemonic())) {
    report(error_out, "anchor mnemonic is in the PC-relative denylist");
    return false;
  }
  if (is_s_clause(anchor)) {
    report(error_out, "anchor mnemonic is s_clause");
    return false;
  }
  return true;
}

std::optional<ResolvedInstrumentationSite>
validate_anchor(const Instruction &anchor, uint64_t anchor_offset,
                std::span<const uint8_t> text_bytes, const InstrumentationPoint &pt,
                rj_code_arch_t arch, std::string *error_out) {
  auto fail = [&](const char *msg) {
    report(error_out, (msg + (", anchor_offset = " + std::to_string(anchor_offset))).c_str());
  };
  // TODO: consume filter_flags to filter anchors based on InstFlags.
  if (pt.filter_flags != 0) {
    fail("InstrumentationPoint::filter_flags must be 0 temporarily");
    return std::nullopt;
  }
  // TODO: support AfterInst / BlockEntry / BlockExit.
  if (pt.kind != InstrumentationKind::BeforeInst) {
    fail("InstrumentationPoint::kind must be BeforeInst temporarily");
    return std::nullopt;
  }
  // A probe call needs both halves of the request: the object to resolve the
  // symbol in, and the symbol name
  if ((pt.probe_obj != nullptr) != (!pt.probe_symbol.empty())) {
    report(error_out, "InstrumentationPoint probe_obj and probe_symbol must both be set "
                      "(a probe call) or both be empty (the inline nop)");
    return std::nullopt;
  }
  // TODO: consume force_full_exec when EXEC policy management is implemented
  if (pt.force_full_exec) {
    fail("InstrumentationPoint::force_full_exec must be false temporarily");
    return std::nullopt;
  }

  if (!is_relocatable_anchor(anchor, anchor_offset, text_bytes, arch, error_out))
    return std::nullopt;

  const auto size = static_cast<uint32_t>(anchor.size());
  ResolvedInstrumentationSite site;
  site.kind = pt.kind;
  site.anchor_offset = anchor_offset;
  site.original_size = size;
  site.original_bytes.assign(text_bytes.begin() + anchor_offset,
                             text_bytes.begin() + anchor_offset + size);
  site.mnemonic = std::string(anchor.mnemonic());
  return site;
}

bool validate_inline_nop_plan(const TrampolinePlan &plan, std::string *error_out) {
  if (!plan.emit_original) {
    report(error_out, "trampoline plan: emit_original must be true for the inlined nop");
    return false;
  }
  if (!plan.after_items.empty()) {
    report(error_out, "trampoline plan: after_items must be empty for the inline nop");
    return false;
  }
  if (plan.before_items.size() != 1 || plan.before_items[0].words.size() != 1 ||
      plan.before_items[0].words[0] != build_s_nop(0, plan.arch)) {
    report(error_out, "trampoline plan: before_items must be exactly { { s_nop 0 } } "
                      "for the inlined nop");
    return false;
  }
  return true;
}

// Only ordinary GPR clobbers feed the spill set. Special machine state in the
// summary (EXEC/VCC/M0/FLAT_SCRATCH) is rejected up front by
// check_probe_special_state
RegisterSet compute_instrumentation_clobbers(const ProbeClobberSummary &probe_summary,
                                             const RegisterSet &builder_clobbers) {
  return probe_summary.ordinary_clobbers | builder_clobbers;
}

RegisterSet compute_spill_set(const RegisterSet &live_at_anchor,
                              const RegisterSet &instrumentation_clobbers) {
  return live_at_anchor & instrumentation_clobbers;
}

bool plan_vgpr_spills(const RegisterSet &spill_set, SpillManager &spills, rj_code_arch_t arch,
                      std::vector<SpillSlot> &out, std::string *error_out) {
  out.clear();

  // Plans VGPR spills only; the orchestrator routes SGPRs to plan_sgpr_spills.
  // Reject any non-VGPR here defensively before reserving anything.
  std::string non_vgpr;
  spill_set.for_each([&](RegisterRef ref) {
    if (ref.cls != RegClass::VGPR) {
      non_vgpr += non_vgpr.empty() ? " " : ", ";
      non_vgpr += reg_name(ref);
    }
  });
  if (!non_vgpr.empty()) {
    report(error_out,
           ("probe-call spill of non-VGPR registers not yet supported:" + non_vgpr).c_str());
    return false;
  }

  const uint32_t max_offset = max_scratch_offset_bytes(arch);
  if (max_offset == 0) {
    report(error_out, "probe-call spilling not supported for target architecture");
    return false;
  }

  bool ok = true;
  std::string fail;
  spill_set.for_each([&](RegisterRef ref) {
    if (!ok)
      return;
    const std::optional<uint32_t> off = spills.allocate_slot(ref);
    if (!off) {
      fail = "probe-call spill of " + reg_name(ref) + " exceeds the per-lane scratch limit";
      ok = false;
    } else if (*off > max_offset) {
      fail = "probe-call spill offset for " + reg_name(ref) +
             " exceeds the scratch instruction offset field";
      ok = false;
    } else {
      out.push_back(SpillSlot{RegClass::VGPR, ref.index, *off});
    }
  });
  if (!ok) {
    out.clear();
    report(error_out, fail.c_str());
    return false;
  }
  return true;
}

bool plan_sgpr_spills(const RegisterSet &spill_set, const RegisterSet &bridge_unavailable,
                      const std::vector<SpillSlot> &vgpr_spills, uint32_t kernel_vgpr_count,
                      SpillManager &spills, rj_code_arch_t arch, std::vector<SpillSlot> &out,
                      uint16_t &out_bridge, std::string *error_out) {
  out.clear();

  // Only SGPRs bridge through a VGPR here; other classes (AccVGPR) are unhandled.
  std::string non_sgpr;
  spill_set.for_each([&](RegisterRef ref) {
    if (ref.cls != RegClass::SGPR) {
      non_sgpr += non_sgpr.empty() ? " " : ", ";
      non_sgpr += reg_name(ref);
    }
  });
  if (!non_sgpr.empty()) {
    report(error_out,
           ("probe-call spill of non-SGPR registers not yet supported:" + non_sgpr).c_str());
    return false;
  }
  if (spill_set.none())
    return true;

  const uint32_t max_offset = max_scratch_offset_bytes(arch);
  if (max_offset == 0) {
    report(error_out, "probe-call spilling not supported for target architecture");
    return false;
  }

  // Bridge, within the kernel's allocated VGPR count (never an unallocated index).
  // Prefer a VGPR the site is not already using; else reuse a spilled VGPR, whose
  // own value is already on scratch and gets reloaded after the bridge's last use
  // (build_spill_bracket orders the VGPR fills after the SGPR ones). Those are the
  // only two safe choices: the prologue's writelane destroys whatever the bridge
  // held, so anything live that is not spilled would be lost. bridge_unavailable
  // covers the live set, so the first scan never returns one of those.
  const uint16_t vgpr_bound =
      static_cast<uint16_t>(std::min<uint32_t>(kernel_vgpr_count, REGISTER_SET_MAX_VGPRS));
  std::optional<uint16_t> bridge;
  for (uint16_t v = 0; v < vgpr_bound; ++v) {
    if (!bridge_unavailable.contains(RegisterRef{RegClass::VGPR, v, 1})) {
      bridge = v;
      break;
    }
  }
  if (!bridge && !vgpr_spills.empty())
    bridge = vgpr_spills.front().reg; // allocated by construction, so in bounds
  if (!bridge) {
    report(error_out, "probe-call SGPR spill needs a bridge VGPR, but none is dead or already "
                      "spilled within the kernel's allocated VGPRs");
    return false;
  }

  bool ok = true;
  std::string fail;
  spill_set.for_each([&](RegisterRef ref) {
    if (!ok)
      return;
    const std::optional<uint32_t> off = spills.allocate_slot(ref);
    if (!off) {
      fail = "probe-call spill of " + reg_name(ref) + " exceeds the per-lane scratch limit";
      ok = false;
    } else if (*off > max_offset) {
      fail = "probe-call spill offset for " + reg_name(ref) +
             " exceeds the scratch instruction offset field";
      ok = false;
    } else {
      out.push_back(SpillSlot{RegClass::SGPR, ref.index, *off});
    }
  });
  if (!ok) {
    out.clear();
    report(error_out, fail.c_str());
    return false;
  }
  out_bridge = *bridge;
  return true;
}

bool plan_acc_spills(const RegisterSet &spill_set, uint32_t acc_count, SpillManager &spills,
                     rj_code_arch_t arch, std::vector<SpillSlot> &out, std::string *error_out) {
  out.clear();

  // Plans AccVGPR spills only; the orchestrator routes VGPRs/SGPRs elsewhere.
  std::string non_acc;
  spill_set.for_each([&](RegisterRef ref) {
    if (ref.cls != RegClass::ACC_VGPR) {
      non_acc += non_acc.empty() ? " " : ", ";
      non_acc += reg_name(ref);
    }
  });
  if (!non_acc.empty()) {
    report(error_out, ("AccVGPR spill planning received non-AccVGPR registers:" + non_acc).c_str());
    return false;
  }
  if (spill_set.none())
    return true;

  if (!arch_has_accvgpr(arch)) {
    report(error_out, "AccVGPR spilling is only supported on CDNA1-4 targets");
    return false;
  }

  const uint32_t max_offset = max_scratch_offset_bytes(arch);
  if (max_offset == 0) {
    report(error_out, "probe-call spilling not supported for target architecture");
    return false;
  }

  // Never spill an AccVGPR the kernel did not allocate: acc_count is the AGPR
  // window size (unified VGPR budget minus the ACCUM_OFFSET base). Reject out of
  // band before reserving anything.
  std::string oob;
  spill_set.for_each([&](RegisterRef ref) {
    if (ref.index >= acc_count) {
      oob += oob.empty() ? " " : ", ";
      oob += reg_name(ref);
    }
  });
  if (!oob.empty()) {
    report(error_out,
           ("probe-call spill of AccVGPR past the kernel's allocated count:" + oob).c_str());
    return false;
  }

  bool ok = true;
  std::string fail;
  spill_set.for_each([&](RegisterRef ref) {
    if (!ok)
      return;
    const std::optional<uint32_t> off = spills.allocate_slot(ref);
    if (!off) {
      fail = "probe-call spill of " + reg_name(ref) + " exceeds the per-lane scratch limit";
      ok = false;
    } else if (*off > max_offset) {
      fail = "probe-call spill offset for " + reg_name(ref) +
             " exceeds the scratch instruction offset field";
      ok = false;
    } else {
      out.push_back(SpillSlot{RegClass::ACC_VGPR, ref.index, *off});
    }
  });
  if (!ok) {
    out.clear();
    report(error_out, fail.c_str());
    return false;
  }
  return true;
}

namespace {

// Fill the layout/identity fields shared by every trampoline plan
TrampolinePlan make_base_plan(const ResolvedInstrumentationSite &site, rj_code_arch_t arch,
                              uint64_t trampoline_offset) {
  TrampolinePlan plan;
  plan.arch = arch;
  plan.anchor_offset = site.anchor_offset;
  plan.original_size = site.original_size;
  plan.trampoline_offset = trampoline_offset;
  plan.return_target = site.anchor_offset + site.original_size;

  // Little-endian host assumption (consistent with DBT and the rest of the
  // codebase): host byte order matches AMDGPU's little-endian encoding, so
  // memcpy of the raw bytes into uint32_t words preserves semantics.
  const size_t num_words = site.original_size / sizeof(uint32_t);
  plan.original_words.resize(num_words);
  std::memcpy(plan.original_words.data(), site.original_bytes.data(), site.original_size);
  return plan;
}

} // namespace

TrampolinePlan make_trampoline_plan(const ResolvedInstrumentationSite &site, rj_code_arch_t arch,
                                    uint64_t trampoline_offset) {
  TrampolinePlan plan = make_base_plan(site, arch, trampoline_offset);
  plan.before_items = {InlineAsmItem{{build_s_nop(0, arch)}}};
  plan.after_items = {};
  plan.emit_original = true;
  return plan;
}

Instrumentor::Instrumentor(const AmdGpuCodeObject &obj, rj_code_arch_t arch)
    : obj_(obj), arch_(arch) {}

Instrumentor::~Instrumentor() = default;

void Instrumentor::add_point(InstrumentationPoint pt) { points_.push_back(std::move(pt)); }

void Instrumentor::add_point_by_offset(uint64_t anchor_offset, InstrumentationKind kind) {
  InstrumentationPoint pt;
  pt.anchor_offset = anchor_offset;
  pt.kind = kind;
  points_.push_back(std::move(pt));
}

bool Instrumentor::ensure_blocks_built(std::string *error_out) {
  if (blocks_built_)
    return true;
  if (arch_ == ROCJITSU_CODE_ARCH_RV32I || arch_ == ROCJITSU_CODE_ARCH_RV64I) {
    report(error_out, "AMDGPU instrumentation does not support RISC-V architectures");
    return false;
  }
  const auto &registry = default_isa_target_registry();
  const rj_code_target_id_t target = obj_.target_id();
  if (target != ROCJITSU_CODE_TARGET_INVALID) {
    const IsaTargetDescriptor *descriptor = registry.find(target);
    if (descriptor == nullptr || descriptor->architecture_id != arch_) {
      report(error_out, "code-object target does not match the requested architecture");
      return false;
    }
  }
  auto decoder = target == ROCJITSU_CODE_TARGET_INVALID ? Decoder::create(arch_)
                                                        : Decoder::create(registry, target);
  if (!decoder) {
    report(error_out, "no decoder available for the requested architecture");
    return false;
  }
  decoder_ = std::move(decoder);
  util::StringDiagnostic decode_error;
  auto blocks = BasicBlock::build(obj_, *decoder_, arch_, decode_error.emitter());
  if (blocks.failed()) {
    report(error_out, decode_error.message().c_str());
    return false;
  }
  blocks_ = std::move(blocks).value();
  // BasicBlock::build returns blocks in .text order. Keep clause state across
  // block boundaries because a branch target may split the linear instruction
  // stream in the middle of a clause.
  uint32_t clause_remaining = 0;
  for (const auto &block : blocks_) {
    uint64_t cur = block->start_offset();
    for (const Instruction &inst : block->instructions()) {
      if (clause_remaining > 0) {
        clause_blocked_offsets_.insert(cur);
        --clause_remaining;
      }
      offset_to_inst_.emplace(cur, &inst);
      if (is_s_clause(inst))
        clause_remaining = std::max(clause_remaining, s_clause_following_instruction_count(inst));
      cur += static_cast<uint64_t>(inst.size());
    }
  }
  blocks_built_ = true;
  return true;
}

const Instruction *Instrumentor::find_instruction_at_offset(uint64_t anchor_offset) const {
  auto it = offset_to_inst_.find(anchor_offset);
  return it == offset_to_inst_.end() ? nullptr : it->second;
}

Instrumentor::ResolvedPoints Instrumentor::resolve_points() {
  ResolvedPoints out;

  std::string err;
  if (!ensure_blocks_built(&err)) {
    out.errors.push_back(std::move(err));
    return out;
  }
  if (obj_.text_sections().empty()) {
    out.errors.emplace_back("code object has no .text section");
    return out;
  }
  // TODO: support multi-text code objects. Anchor offsets would need to identify
  // which .text section they belong to.
  if (obj_.text_sections().size() > 1) {
    out.errors.emplace_back("code object has multiple .text sections; currently supports only one");
    return out;
  }
  const Section *text = obj_.text_sections().front();
  const std::span<const uint8_t> text_bytes(reinterpret_cast<const uint8_t *>(text->data()),
                                            text->size());

  const auto &registry = default_isa_target_registry();
  const IsaTargetDescriptor *arch_descriptor = registry.find(arch_);
  auto effective_target = [&](const AmdGpuCodeObject &object) {
    if (object.target_id() != ROCJITSU_CODE_TARGET_INVALID)
      return object.target_id();
    if (arch_descriptor == nullptr)
      return ROCJITSU_CODE_TARGET_INVALID;
    const IsaGpuTargetDescription *gpu_target = registry.find_default_gpu_target(*arch_descriptor);
    return gpu_target == nullptr ? ROCJITSU_CODE_TARGET_INVALID : gpu_target->public_id;
  };
  const rj_code_target_id_t destination_target = effective_target(obj_);

  // Store probe objects and symbols together in probe_keys (object, symbol).
  std::vector<std::pair<const AmdGpuCodeObject *, std::string>> probe_keys;
  // Helper function to get a probe index for a given InstrumentationPoint
  // If the probe is new, then resolve it and get probe info; add it to
  // probe_keys and out.probes.
  auto resolve_probe_index = [&](const InstrumentationPoint &pt,
                                 std::string &perr) -> std::optional<size_t> {
    for (size_t i = 0; i < probe_keys.size(); ++i) {
      if (probe_keys[i].first == pt.probe_obj && probe_keys[i].second == pt.probe_symbol)
        return i;
    }
    const rj_code_target_id_t probe_target = effective_target(*pt.probe_obj);
    if (destination_target != ROCJITSU_CODE_TARGET_INVALID &&
        probe_target != ROCJITSU_CODE_TARGET_INVALID && destination_target != probe_target) {
      perr = "probe concrete target does not match the destination code-object target";
      return std::nullopt;
    }
    auto sym = resolve_probe_symbol(*pt.probe_obj, pt.probe_symbol, &perr);
    if (!sym)
      return std::nullopt;
    auto callable = build_probe_callable(*pt.probe_obj, *sym, arch_, &perr);
    if (!callable)
      return std::nullopt;
    // Inputs the probe reads that its convention does not supply. Typically a
    // value only the kernel prologue produces -- workitem_id_x in v31, say --
    // which a trampoline at an arbitrary site cannot reproduce, so the probe
    // would read whatever the instrumented kernel left behind. An ordinary
    // uninitialized read lands here too, and is equally unusable.
    auto live_ins = analyze_probe_live_ins(*pt.probe_obj, *sym, arch_, callable->abi, &perr);
    if (!live_ins)
      return std::nullopt;
    if (!live_ins->none()) {
      perr = "probe '" + pt.probe_symbol + "' reads " + format_register_set(*live_ins) +
             " before defining it, and the calling convention does not supply it";
      return std::nullopt;
    }
    out.probes.push_back(std::move(*callable));
    probe_keys.emplace_back(pt.probe_obj, pt.probe_symbol);
    return out.probes.size() - 1;
  };

  // All-or-nothing: per-point errors accumulate
  std::vector<ResolvedInstrumentationSite> sites;
  std::unordered_set<uint64_t> site_offsets;
  sites.reserve(points_.size());
  for (const auto &pt : points_) {
    const Instruction *anchor = find_instruction_at_offset(pt.anchor_offset);
    if (anchor == nullptr) {
      out.errors.emplace_back("no decoded instruction starts at the requested anchor_offset = " +
                              std::to_string(pt.anchor_offset));
      continue;
    }

    if (site_offsets.find(pt.anchor_offset) != site_offsets.end()) {
      out.errors.emplace_back("multiple points requested the same anchor_offset = " +
                              std::to_string(pt.anchor_offset));
      continue;
    }

    std::string perr;
    auto site = validate_anchor(*anchor, pt.anchor_offset, text_bytes, pt, arch_, &perr);
    if (!site) {
      out.errors.push_back(std::move(perr));
      continue;
    }

    if (clause_blocked_offsets_.contains(pt.anchor_offset)) {
      out.errors.emplace_back("anchor is inside an s_clause run, anchor_offset = " +
                              std::to_string(pt.anchor_offset));
      continue;
    }

    // Resolve the probe request. An unresolvable or non-relocatable probe is
    // a fatal, all-or-nothing validation error.
    if (pt.probe_obj != nullptr) {
      auto index = resolve_probe_index(pt, perr);
      if (!index) {
        out.errors.push_back(std::move(perr));
        continue;
      }
      site->probe_index = *index;
    }

    sites.push_back(std::move(*site));
    site_offsets.insert(site->anchor_offset);
  }

  if (out.errors.empty())
    out.sites = std::move(sites);
  else
    out.probes.clear(); // all-or-nothing: no partial registry on failure.
  return out;
}

Instrumentor::ValidationResult Instrumentor::validate_points() {
  ResolvedPoints resolved = resolve_points();
  ValidationResult result;
  result.errors = std::move(resolved.errors);
  result.sites = std::move(resolved.sites);
  return result;
}

InstrumentedCodeObject Instrumentor::patch() {
  // Slice off the debug summaries; move the base subobject into the return.
  auto debug = patch_with_debug_summaries();
  return std::move(static_cast<InstrumentedCodeObject &>(debug));
}

InstrumentedCodeObjectDebug Instrumentor::patch_with_debug_summaries() {
  InstrumentedCodeObjectDebug result;

  if (patched_) {
    result.errors.emplace_back(
        "Instrumentor::patch / patch_with_debug_summaries has already been called");
    return result;
  }
  patched_ = true;

  if (points_.empty()) {
    result.errors.emplace_back("Instrumentor::patch requires at least one queued point; got zero");
    return result;
  }

  ResolvedPoints resolved = resolve_points();
  if (!resolved.errors.empty()) {
    result.errors = std::move(resolved.errors);
    return result;
  }

  // Construct the patcher and preflight builder output before mutating it.
  // Each trampoline is appended directly after the original .text bytes as a
  // local code cave (as in the current DBT design).
  CodeObjectPatcher patcher(obj_);
  const uint64_t cave_start = patcher.text_size();

  // Liveness over the decoded blocks, built once and reused by every probe-call
  // site. validate_points() already built blocks_, and obj_ owns the
  // Instructions live_before() is keyed on.
  // TODO: scope this to the kernel containing each anchor, like the DBT path,
  // rather than treating every decoded block as one CFG. Safe here: kernels end
  // in s_endpgm, so there are no false cross-kernel fallthrough edges.
  std::vector<BasicBlock *> liveness_scope;
  liveness_scope.reserve(blocks_.size());
  for (const auto &block : blocks_)
    liveness_scope.push_back(block.get());
  const LivenessAnalysis liveness{KernelBlockScope(liveness_scope)};

  // Lay out the appended region as [probe bodies][trampolines]. Each distinct
  // probe body is copied once, ahead of the trampolines that call into it, so a
  // trampoline's target address is known before it is emitted and sites sharing
  // a probe share its single body.
  const auto &sites = resolved.sites;
  // Allocate offsets for each probe body, starting at the local cave (the first
  // byte after the original .text). The site loop below continues advancing this
  // same cursor so trampolines follow the probe bodies.
  uint64_t cave_cursor = cave_start;
  for (ProbeCallable &probe : resolved.probes) {
    probe.output_text_offset = cave_cursor;
    cave_cursor += probe.body_words.size() * sizeof(uint32_t);
  }

  // Register spilling targets a single-kernel code object for now: any site that
  // must spill grows that one kernel's per-lane scratch. The SpillManager is
  // created lazily on the first spilling site and shared by the rest; its final
  // size is written back to the descriptor after all sites succeed.
  const std::vector<KernelDescriptorInfo> kernels =
      scan_kernel_descriptors(patcher.image_bytes(), patcher.text_offset(), patcher.text_size());

  // Temporary SGPR-allocation bound. The probe-call return-link pair is fixed
  // by the calling convention, so the kernel must allocate up to it. Derived from
  // the descriptors already scanned above.
  const std::optional<uint32_t> kernel_sgpr_count =
      AmdGpuCodeObject::min_kernel_sgpr_count(arch_, kernels);

  // The kernel's VGPR allocation, decoded once: it depends only on the
  // descriptor and the arch, both loop-invariant. Absent unless exactly one
  // kernel was discovered, since with several there is no single allocation to
  // name.
  const std::optional<KernelVgprBounds> vgpr_bounds =
      kernels.size() == 1
          ? std::optional<KernelVgprBounds>(kernel_vgpr_bounds(arch_, kernels.front().descriptor))
          : std::nullopt;
  std::optional<SpillManager> spills;
  uint64_t spill_descriptor_file_offset = 0;

  std::vector<AppliedSite> applied;
  applied.reserve(sites.size());
  // Allocate offsets for each site. Trampoline size is highly dependent on the
  // plan (inlined assembly? spills? arguments passed?) so set up the plan first.
  // Then build the trampoline based on the plan.
  for (const auto &site : sites) {
    const uint64_t trampoline_offset = cave_cursor;
    std::string err;
    std::optional<TrampolineBytes> bytes;
    AppliedSite record{&site, trampoline_offset, {}};

    // If this site calls a probe, then we need to know what registers it will
    // clobber
    if (site.is_probe_call()) {
      const ProbeCallable &probe = resolved.probes[*site.probe_index];

      // A probe call selects temp SGPRs (link/target/SCC/special-state) bounded by
      // the kernel's own allocation. Without a discovered descriptor that bound is
      // unknown, so a temp could land past the kernel's .sgpr_count; fail closed
      // rather than fall back to the device-wide default (growing the allocation is
      // deferred).
      if (!kernel_sgpr_count) {
        result.errors.push_back("probe call at anchor_offset " +
                                std::to_string(site.anchor_offset) +
                                " requires a discovered kernel descriptor to bound SGPR "
                                "selection, but none was found");
        continue;
      }

      // The kernel must own the fixed return-link pair.
      if (const uint16_t link_base = probe.abi.link_pair_base;
          is_valid_probe_abi(probe.abi) &&
          !probe_link_pair_fits_in_kernel(*kernel_sgpr_count, link_base)) {
        result.errors.push_back(
            "probe call needs the return-link pair s[" + std::to_string(link_base) + ":" +
            std::to_string(link_base + 1) + "] but the kernel allocates only " +
            std::to_string(*kernel_sgpr_count) + " SGPRs; rebuild the kernel with at least " +
            std::to_string(link_base + 2) + " SGPRs");
        continue;
      }

      // Callee clobbers (probe body) + liveness at the anchor feed envelope
      // resource selection and the no-spill policy gate.
      auto summary = build_probe_clobber_summary(probe, &err);
      if (!summary) {
        result.errors.push_back(std::move(err));
        continue;
      }

      // Check for special machine state that has no save/restore path yet
      if (!check_probe_special_state(*summary, &err)) {
        result.errors.push_back(std::move(err));
        continue;
      }

      // The probe must not overwrite its own return-link pair before returning.
      if (!check_probe_link_pair(*summary, probe.abi, &err)) {
        result.errors.push_back(std::move(err));
        continue;
      }

      // Get liveness for this anchor
      const Instruction *anchor = find_instruction_at_offset(site.anchor_offset);
      if (anchor == nullptr) {
        result.errors.emplace_back("internal: anchor instruction vanished after validation");
        continue;
      }
      const RegisterSet &live = liveness.live_before(*anchor);

      // Set the probe's offset
      TrampolinePlan plan = make_base_plan(site, arch_, trampoline_offset);
      plan.probe_target_offset = probe.output_text_offset;
      // Preserve special state the probe clobbers by saving it to a dead SGPR
      // around the call. EXEC/VCC are saved unconditionally as a conservative
      // policy: the summary detects the special-state writes the decoder exposes as
      // operands, but always saving keeps correctness independent of per-opcode
      // implicit-def coverage. M0 stays clobber-gated.
      plan.preserve_exec = true;
      plan.preserve_vcc = true;
      plan.preserve_m0 = summary->touches_m0;
      // Cap envelope/temp SGPR selection at the kernel's own allocation so a temp
      // never lands past its .sgpr_count
      plan.kernel_sgpr_count = *kernel_sgpr_count;
      // Given liveness, clobbers, and calling convention, select registers
      // for trampoline and determine how big the trampoline will be
      if (!TrampolineBuilder::plan_probe_call(plan, probe.abi, live, summary->ordinary_clobbers,
                                              &err)) {
        result.errors.push_back(std::move(err));
        continue;
      }

      // Spill any register that is both live at the anchor and clobbered by the
      // instrumentation envelope/probe. VGPRs spill to the kernel's per-lane
      // scratch; SGPRs bridge through a dead VGPR; AccVGPRs and over-cap sites
      // fail closed inside the planners.
      const RegisterSet clobbers =
          compute_instrumentation_clobbers(*summary, plan.builder_clobbers);
      const RegisterSet spill = compute_spill_set(live, clobbers);
      if (!spill.none()) {
        // Single-kernel assumption: spilling needs exactly one kernel descriptor
        // with non-zero fixed scratch to grow. vgpr_bounds is present on exactly
        // that condition, so testing it here is what licenses the dereferences
        // below.
        if (!vgpr_bounds || kernels.front().descriptor.private_segment_fixed_size == 0) {
          result.errors.push_back(
              "probe call at anchor_offset " + std::to_string(site.anchor_offset) +
              " must spill live registers, but the code object does not have a single kernel with "
              "fixed scratch to spill into");
          continue;
        }
        const KernelDescriptorInfo &kernel = kernels.front();
        if (!spills) {
          // Cap total per-lane scratch at the widest slot the scratch offset field
          // can encode, so the SpillManager limit and the per-instruction offset
          // guards in the planners agree.
          const uint32_t scratch_limit = max_scratch_offset_bytes(arch_) + SpillManager::kSlotBytes;
          spills.emplace(kernel.descriptor.private_segment_fixed_size, scratch_limit);
          spill_descriptor_file_offset = kernel.descriptor_file_offset;
        }
        // Split by class: VGPRs go straight to scratch; SGPRs bridge through a dead
        // VGPR; AccVGPRs go straight to scratch via the CDNA `acc` bit (no bridge).
        RegisterSet vgpr_spill = spill;
        vgpr_spill.clear_class(RegClass::SGPR);
        vgpr_spill.clear_class(RegClass::ACC_VGPR);
        RegisterSet sgpr_spill = spill;
        sgpr_spill.clear_class(RegClass::VGPR);
        sgpr_spill.clear_class(RegClass::ACC_VGPR);
        RegisterSet acc_spill = spill;
        acc_spill.clear_class(RegClass::VGPR);
        acc_spill.clear_class(RegClass::SGPR);
        if (!plan_vgpr_spills(vgpr_spill, *spills, arch_, plan.vgpr_spills, &err)) {
          result.errors.push_back(std::move(err));
          continue;
        }
        // The SGPR bridge must be an ordinary VGPR: an index in the accumulator
        // window would alias an AGPR that is not part of acc_spills.
        if (!sgpr_spill.none() &&
            !plan_sgpr_spills(sgpr_spill, live, plan.vgpr_spills, vgpr_bounds->ordinary_bound,
                              *spills, arch_, plan.sgpr_spills, plan.spill_bridge_vgpr, &err)) {
          result.errors.push_back(std::move(err));
          continue;
        }
        // AccVGPRs (CDNA only): reject an index past the allocated AGPR window.
        if (!acc_spill.none() && !plan_acc_spills(acc_spill, vgpr_bounds->acc_count, *spills, arch_,
                                                  plan.acc_spills, &err)) {
          result.errors.push_back(std::move(err));
          continue;
        }
      }

      // Get trampoline from TrampolineBuilder
      bytes = TrampolineBuilder::emit_probe_call(plan, &err);
      record.is_probe_call = true;
      record.probe_target_offset = probe.output_text_offset;
      record.link_pair_base = plan.link_pair_base;
      record.target_pair_base = plan.target_pair_base;
      // If this site does not call a probe
      // Currently, this means that the plan is to put a nop which means we do
      // not need to mess with spills
    } else {
      TrampolinePlan plan = make_trampoline_plan(site, arch_, trampoline_offset);
      // The only assembly we currently allow is an inlined nop
      if (!validate_inline_nop_plan(plan, &err)) {
        result.errors.push_back(std::move(err));
        continue;
      }
      bytes = TrampolineBuilder::build(plan, &err);
    }

    if (!bytes) {
      result.errors.push_back(std::move(err));
      continue;
    }
    // Update the the current cave offset based on the size of the trampoline
    // and record the built trampoline
    cave_cursor += bytes->trampoline_words.size() * sizeof(uint32_t);
    record.bytes = std::move(*bytes);
    applied.push_back(std::move(record));
  }

  // All-or-nothing: bail before mutating the patcher if any site failed.
  if (!result.errors.empty())
    return result;

  // Grow the spilled kernel's per-lane scratch to cover its DBI spill zone. Done
  // before replace_text so the edit lands at the descriptor's original file offset;
  // replace_text then relocates the descriptor with the rest of the ELF.
  if (spills) {
    if (!patcher.set_private_segment_fixed_size(spill_descriptor_file_offset,
                                                spills->total_private_bytes())) {
      result.errors.emplace_back(
          "failed to grow private_segment_fixed_size for the spilled kernel descriptor");
      return result;
    }
  }

  // Every per-site validation, branch-range check, and trampoline-byte
  // construction has succeeded up to this point. Assemble the new .text in one
  // buffer: the original bytes with each anchor spliced to its forward branch,
  // followed by every trampoline appended as the local cave. replace_text()
  // grows .text in place and fixes up the surrounding ELF (section/segment
  // sizes, moved symbols, descriptor entries).
  const auto text_span = patcher.text_bytes();
  std::vector<uint8_t> new_text(text_span.begin(), text_span.end());
  for (const auto &a : applied) {
    std::memcpy(new_text.data() + a.site->anchor_offset, a.bytes.patched_anchor_bytes.data(),
                a.site->original_size);
  }
  // Append in the laid-out order: probe bodies first (one per distinct probe),
  // then trampolines.
  for (const ProbeCallable &probe : resolved.probes)
    append_words(new_text, probe.body_words);
  for (const auto &a : applied)
    append_words(new_text, a.bytes.trampoline_words);
  if (!patcher.replace_text(new_text)) {
    result.errors.emplace_back("failed to replace .text with the instrumented code");
    return result;
  }

  // Emit and build patch summaries.
  result.elf_bytes = patcher.emit();
  for (const auto &a : applied) {
    InstrumentationPatch patch;
    patch.anchor_offset = a.site->anchor_offset;
    patch.original_size = a.site->original_size;
    patch.trampoline_offset = a.trampoline_offset;
    patch.return_target = a.site->anchor_offset + a.site->original_size;
    patch.original_bytes = a.site->original_bytes;
    patch.patched_anchor_bytes = a.bytes.patched_anchor_bytes;
    patch.is_probe_call = a.is_probe_call;
    if (a.is_probe_call) {
      patch.probe_symbol = resolved.probes[*a.site->probe_index].symbol;
      patch.probe_target_offset = a.probe_target_offset;
      patch.link_pair_base = a.link_pair_base;
      patch.target_pair_base = a.target_pair_base;
    }
    result.patches.push_back(std::move(patch));
  }
  return result;
}

} // namespace rocjitsu
