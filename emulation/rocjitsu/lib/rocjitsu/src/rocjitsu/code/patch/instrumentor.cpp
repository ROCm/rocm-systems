// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/instrumentor.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/analysis/def_use_chain.h"
#include "rocjitsu/code/analysis/exec_state.h"
#include "rocjitsu/code/analysis/gfx1250_vgpr_msb.h"
#include "rocjitsu/code/analysis/liveness.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/builders/spill_builders.h"
#include "rocjitsu/code/code_object.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/entry_prologue.h"
#include "rocjitsu/code/patch/error_report.h"
#include "rocjitsu/code/patch/kernarg_extension.h"
#include "rocjitsu/code/patch/kernel_text_layout.h"
#include "rocjitsu/code/patch/probe_callable.h"
#include "rocjitsu/code/patch/probe_clobber.h"
#include "rocjitsu/code/patch/probe_live_in.h"
#include "rocjitsu/code/patch/probe_symbol.h"
#include "rocjitsu/code/patch/trampoline_builder.h"
#include "rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/isa_traits.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/isa/target_registry.h"
#include "util/diagnostic.h"
#include "util/result.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocjitsu {

namespace {

rj_code_target_id_t effective_target(const AmdGpuCodeObject &object, rj_code_arch_t arch) {
  if (object.target_id() != ROCJITSU_CODE_TARGET_INVALID)
    return object.target_id();
  const IsaTargetRegistry &registry = default_isa_target_registry();
  const IsaTargetDescriptor *descriptor = registry.find(arch);
  if (descriptor == nullptr)
    return ROCJITSU_CODE_TARGET_INVALID;
  const IsaGpuTargetDescription *target = registry.find_default_gpu_target(*descriptor);
  return target == nullptr ? ROCJITSU_CODE_TARGET_INVALID : target->public_id;
}

// These ISA message tables assign MSG_DEALLOC_VGPRS to message 3. Older
// architectures give that encoding different semantics.
bool deallocates_vgprs(const Instruction &instruction, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA5 && arch != ROCJITSU_CODE_ARCH_RDNA3 &&
      arch != ROCJITSU_CODE_ARCH_RDNA3_5 && arch != ROCJITSU_CODE_ARCH_RDNA4)
    return false;
  if (instruction.mnemonic() != "s_sendmsg" && instruction.mnemonic() != "s_sendmsghalt")
    return false;
  const Operand *message = instruction.src_operand(0);
  constexpr uint32_t kDeallocVgprsMessage = 3;
  // In-tree OPR_SENDMSG.MSG is 9/4/7/7 bits for RDNA3/3.5/4/CDNA5.
  // LLVM SendMsg::ID_MASK_GFX11Plus_ (SIDefines.h) uses the low byte. Until these descriptions
  // agree, reject deallocation under either interpretation (including 0x103).
  const uint32_t message_mask = arch == ROCJITSU_CODE_ARCH_RDNA3     ? 0x1ffu
                                : arch == ROCJITSU_CODE_ARCH_RDNA3_5 ? 0xfu
                                                                     : 0x7fu;
  return message == nullptr || (message->encoding_value() & message_mask) == kDeallocVgprsMessage ||
         (message->encoding_value() & 0xffu) == kDeallocVgprsMessage;
}

enum class CallableState { GprIndexing, Vskip, UnavailableStorage };

enum class StateWrite { Unchanged, Safe, Unsafe };

bool has_callable_state(rj_code_arch_t arch, CallableState state) {
  switch (state) {
  case CallableState::GprIndexing:
    return isa_properties(arch).mode_has_gpr_idx_en;
  case CallableState::Vskip:
    return arch_is_cdna_4_or_lower(arch);
  case CallableState::UnavailableStorage:
    return arch == ROCJITSU_CODE_ARCH_CDNA5 || arch == ROCJITSU_CODE_ARCH_RDNA3 ||
           arch == ROCJITSU_CODE_ARCH_RDNA3_5 || arch == ROCJITSU_CODE_ARCH_RDNA4;
  }
  return false;
}

StateWrite
callable_state_write(const Instruction &inst, std::span<const uint8_t> text, rj_code_arch_t arch,
                     CallableState state,
                     const std::unordered_set<const Instruction *> *preserving_calls = nullptr) {
  if (!has_callable_state(arch, state))
    return StateWrite::Unchanged;
  if (inst.flags() & INDIRECT_CALL)
    return preserving_calls != nullptr && preserving_calls->contains(&inst) ? StateWrite::Unchanged
                                                                            : StateWrite::Unsafe;
  // Descriptor counts prove only dispatch-time storage. No later allocation
  // currently proves that the callback envelope has sufficient VGPRs/scratch.
  if (state == CallableState::UnavailableStorage)
    return inst.mnemonic() == "s_alloc_vgpr" || deallocates_vgprs(inst, arch)
               ? StateWrite::Unsafe
               : StateWrite::Unchanged;
  const bool enables = state == CallableState::GprIndexing
                           ? may_enable_gpr_indexing(inst, text, arch)
                           : may_enable_vskip(inst, text, arch);
  if (enables)
    return StateWrite::Unsafe;
  if ((state == CallableState::GprIndexing && inst.mnemonic() == "s_set_gpr_idx_off") ||
      (state == CallableState::Vskip && inst.mnemonic() == "s_setvskip"))
    return StateWrite::Safe;
  if (inst.mnemonic() == "s_setreg_b32" || inst.mnemonic() == "s_setreg_imm32_b32") {
    const Operand *hwreg = inst.dst_operand(0);
    if (hwreg != nullptr &&
        amdgpu::hwreg_slice_overlaps_mode_bit(static_cast<uint16_t>(hwreg->encoding_value()),
                                              state == CallableState::GprIndexing ? 27 : 28))
      return StateWrite::Safe;
  }
  return StateWrite::Unchanged;
}

// A resolved straight-line helper can preserve the required state without an
// ABI assumption. More complex or unresolved callees remain unknown here.
bool callee_preserves_state(const BasicBlock::CallEdge &edge, std::span<const uint8_t> text,
                            rj_code_arch_t arch, CallableState state) {
  if (edge.callee == nullptr || !edge.callee->successors().empty() ||
      !edge.callee->predecessors().empty())
    return false;
  const Instruction *ret = edge.callee->terminator();
  if (ret == nullptr || (ret->mnemonic() != "s_setpc_b64" && ret->mnemonic() != "s_set_pc_i64"))
    return false;
  const Operand *link = ret->src_operand(0);
  if (link == nullptr || link->encoding_value() != edge.return_sreg)
    return false;
  for (const Instruction &inst : edge.callee->instructions()) {
    if (&inst == ret)
      break;
    if (inst.mnemonic().starts_with("s_movrel") ||
        callable_state_write(inst, text, arch, state) != StateWrite::Unchanged)
      return false;
    const InstDefUse effects(inst, 64);
    if (effects.defs.contains({RegClass::SGPR, edge.return_sreg, 1}) ||
        effects.defs.contains({RegClass::SGPR, static_cast<uint16_t>(edge.return_sreg + 1), 1}))
      return false;
  }
  return true;
}

// Recognize a contiguous negative 64-bit PC adjustment with its carry intact.
// Restrict hidden edges to dispatch roots so they cannot enter another helper
// with a return-link context supplied only by that helper's own callers.
bool adjusted_pc_returns_to_entry(const std::array<const Instruction *, 3> &tail, uint64_t base,
                                  uint16_t link, std::span<const KernelDescriptorInfo> kernels) {
  if (tail[0] == nullptr || tail[0]->src_loc() + tail[0]->size() != tail[1]->src_loc() ||
      tail[1]->src_loc() + tail[1]->size() != tail[2]->src_loc())
    return false;
  const std::string_view low_name = tail[0]->mnemonic();
  const std::string_view high_name = tail[1]->mnemonic();
  const std::string_view ret_name = tail[2]->mnemonic();
  if ((low_name != "s_add_u32" && low_name != "s_add_co_u32") ||
      (high_name != "s_addc_u32" && high_name != "s_add_co_ci_u32") ||
      (ret_name != "s_setpc_b64" && ret_name != "s_set_pc_i64"))
    return false;
  const auto is_link = [](const Operand *operand, uint16_t index, uint8_t width = 1) {
    return operand != nullptr &&
           operand->to_register_ref() == RegisterRef{RegClass::SGPR, index, width};
  };
  const uint16_t high = link + 1;
  if (!is_link(tail[0]->dst_operand(0), link) || !is_link(tail[0]->src_operand(0), link) ||
      !is_link(tail[1]->dst_operand(0), high) || !is_link(tail[1]->src_operand(0), high) ||
      !is_link(tail[2]->src_operand(0), link, 2))
    return false;
  const Operand *low_value = tail[0]->src_operand(1);
  const Operand *high_value = tail[1]->src_operand(1);
  if (low_value == nullptr || high_value == nullptr)
    return false;
  const std::optional<uint64_t> low_constant = low_value->const_value();
  const std::optional<uint64_t> high_constant = high_value->const_value();
  if (!low_constant || !high_constant || static_cast<uint32_t>(*high_constant) != 0xffffffffu)
    return false;
  const int64_t delta = static_cast<int32_t>(static_cast<uint32_t>(*low_constant));
  if (delta >= 0 || base < static_cast<uint64_t>(-delta))
    return false;
  const uint64_t target = base - static_cast<uint64_t>(-delta);
  if (!std::ranges::any_of(kernels, [&](const KernelDescriptorInfo &kernel) {
        return kernel.entry_text_offset == target;
      }))
    return false;
  return true;
}

bool callee_returns_to_entry(const BasicBlock::CallEdge &edge,
                             std::span<const KernelDescriptorInfo> kernels) {
  if (edge.callee == nullptr || edge.continuation == nullptr ||
      !edge.callee->successors().empty() || !edge.callee->predecessors().empty())
    return false;
  std::array<const Instruction *, 3> tail = {};
  for (const Instruction &inst : edge.callee->instructions())
    tail = {tail[1], tail[2], &inst};
  if (!adjusted_pc_returns_to_entry(tail, edge.continuation->start_offset(), edge.return_sreg,
                                    kernels))
    return false;
  const uint16_t high = edge.return_sreg + 1;
  for (const Instruction &inst : edge.callee->instructions()) {
    if (&inst == tail[0])
      break;
    const InstDefUse effects(inst, 64);
    if (inst.mnemonic().starts_with("s_movrel") ||
        effects.defs.contains({RegClass::SGPR, edge.return_sreg, 1}) ||
        effects.defs.contains({RegClass::SGPR, high, 1}))
      return false;
  }
  return true;
}

bool static_pc_returns_to_entry(const BasicBlock &block,
                                std::span<const KernelDescriptorInfo> kernels) {
  std::array<const Instruction *, 4> tail = {};
  for (const Instruction &inst : block.instructions())
    tail = {tail[1], tail[2], tail[3], &inst};
  if (tail[0] == nullptr ||
      (tail[0]->mnemonic() != "s_getpc_b64" && tail[0]->mnemonic() != "s_get_pc_i64") ||
      tail[0]->src_loc() + tail[0]->size() != tail[1]->src_loc())
    return false;
  const Operand *destination = tail[0]->dst_operand(0);
  if (destination == nullptr)
    return false;
  const std::optional<RegisterRef> pair = destination->to_register_ref();
  return pair && pair->cls == RegClass::SGPR && pair->width == 2 &&
         adjusted_pc_returns_to_entry({tail[1], tail[2], tail[3]},
                                      tail[0]->src_loc() + tail[0]->size(), pair->index, kernels);
}

// Hidden but bounded return edges preserve initial state if no instruction in
// the closed text changes it. Unknown/external transfers still fail.
bool callable_state_is_preserved_in_closed_text(std::span<const std::unique_ptr<BasicBlock>> blocks,
                                                std::span<const KernelDescriptorInfo> kernels,
                                                std::span<const uint8_t> text, rj_code_arch_t arch,
                                                CallableState state) {
  std::unordered_set<const Instruction *> known_returns;
  for (const std::unique_ptr<BasicBlock> &block : blocks) {
    for (const Instruction &inst : block->instructions()) {
      if (inst.mnemonic().starts_with("s_rfe"))
        return false;
      if (inst.flags() & INDIRECT_CALL) {
        if (&inst != block->terminator() || block->call_edges().empty())
          return false;
      } else if (callable_state_write(inst, text, arch, state) != StateWrite::Unchanged) {
        return false;
      }
    }
    for (const BasicBlock::CallEdge &edge : block->call_edges()) {
      if (edge.callee == nullptr ||
          std::ranges::any_of(kernels,
                              [&](const KernelDescriptorInfo &kernel) {
                                return kernel.entry_text_offset == edge.callee->start_offset();
                              }) ||
          (!callee_preserves_state(edge, text, arch, state) &&
           !callee_returns_to_entry(edge, kernels)))
        return false;
      known_returns.insert(edge.callee->terminator());
    }
  }
  for (const std::unique_ptr<BasicBlock> &block : blocks) {
    const BasicBlock::SuccessorIssue issue = block->successor_issue();
    if (issue == BasicBlock::SuccessorIssue::None)
      continue;
    if (issue != BasicBlock::SuccessorIssue::IndirectControlFlow)
      return false;
    if (known_returns.contains(block->terminator()) || static_pc_returns_to_entry(*block, kernels))
      continue;
    const std::vector<IndirectCallFixup> &fixups = block->static_indirect_call_fixups();
    if (fixups.empty() || !std::ranges::all_of(fixups, [&](const IndirectCallFixup &fixup) {
          return !fixup.source_incomplete && block->terminator() != nullptr &&
                 fixup.source_call_offset == block->terminator()->src_loc() &&
                 std::ranges::any_of(blocks, [&](const std::unique_ptr<BasicBlock> &target) {
                   return target->start_offset() == fixup.source_target_offset;
                 });
        }))
      return false;
  }
  return true;
}

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
  std::optional<uint64_t> return_target = std::nullopt;

  // Probe-call facts for the patch record; defaulted for the inline-nop site.
  bool is_probe_call = false;
  uint64_t probe_target_offset = 0;
  uint16_t link_pair_base = 0;
  uint16_t target_pair_base = 0;
};

// Human-readable register name for spill diagnostics. Ordinary registers are
// named by prefix and index (s5, v3, acc2); special singletons carry no index.
std::string reg_name(RegisterRef ref) {
  switch (ref.cls) {
  case RegClass::SGPR:
    return "s" + std::to_string(ref.index);
  case RegClass::VGPR:
    return "v" + std::to_string(ref.index);
  case RegClass::ACC_VGPR:
    return "acc" + std::to_string(ref.index);
  case RegClass::TTMP:
    return "ttmp" + std::to_string(ref.index);
  case RegClass::EXEC:
    return "exec";
  case RegClass::VCC:
    return "vcc";
  case RegClass::SCC:
    return "scc";
  case RegClass::M0:
    return "m0";
  case RegClass::FLAT_SCRATCH:
    return "flat_scratch";
  case RegClass::PC:
    return "pc";
  }
  return "?" + std::to_string(ref.index);
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
// preserving the separate gfx908 allocation and the gfx90a unified allocation.

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
  case ROCJITSU_CODE_ARCH_RV32I:
  case ROCJITSU_CODE_ARCH_RV64I:
  case ROCJITSU_CODE_ARCH_INVALID:
    break;
  }
  assert(false && "unified VGPR allocation for an unadmitted architecture");
  return false;
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
                                    const rocjitsu::amdhsa::kernel_descriptor_t &desc) {
  const uint32_t granulated = RJ_AMDHSA_BITS_GET(
      desc.compute_pgm_rsrc1, rocjitsu::amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
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
          ? (RJ_AMDHSA_BITS_GET(desc.compute_pgm_rsrc3,
                                rocjitsu::amdhsa::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET) +
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

// The wave size every kernel in @p kernels runs at, or nullopt when there are
// none or they disagree.
std::optional<uint8_t> common_wavefront_size(rj_code_arch_t arch,
                                             std::span<const KernelDescriptorInfo> kernels) {
  std::optional<uint8_t> common;
  for (const KernelDescriptorInfo &kernel : kernels) {
    const uint8_t wave_size = kernel_wavefront_size(arch, kernel.descriptor);
    if (common && *common != wave_size)
      return std::nullopt;
    common = wave_size;
  }
  return common;
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

namespace {
bool validate_anchor_bytes(const Instruction &anchor, uint64_t anchor_offset,
                           std::span<const uint8_t> text_bytes, std::string *error_out,
                           bool relocated = true) {
  if (anchor_offset % sizeof(uint32_t) != 0) {
    report(error_out, "anchor_offset must be dword aligned");
    return false;
  }
  // Instruction::size() returns int by convention; the `!= 4 && != 8` check
  // also rejects negative values (which decoders never produce in practice).
  const int size = anchor.size();
  if (size <= 0 || size % 4 != 0 || (relocated && size != 4 && size != 8)) {
    report(error_out, relocated ? "anchor instruction size must be 4 or 8 bytes"
                                : "anchor instruction size must contain complete dwords");
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
  return true;
}
} // namespace

bool is_relocatable_anchor(const Instruction &anchor, uint64_t anchor_offset,
                           std::span<const uint8_t> text_bytes,
                           [[maybe_unused]] rj_code_arch_t arch, std::string *error_out) {
  if (!validate_anchor_bytes(anchor, anchor_offset, text_bytes, error_out))
    return false;
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
                rj_code_arch_t arch, std::string *error_out, bool setreg_vgpr_msb_fixup) {
  auto fail = [&](const char *msg) {
    report(error_out, (msg + (", anchor_offset = " + std::to_string(anchor_offset))).c_str());
  };
  // TODO: consume filter_flags to filter anchors based on InstFlags.
  if (pt.filter_flags != 0) {
    fail("InstrumentationPoint::filter_flags must be 0 temporarily");
    return std::nullopt;
  }
  // TODO: support BlockEntry / BlockExit.
  if (pt.kind != InstrumentationKind::BeforeInst && pt.kind != InstrumentationKind::AfterInst &&
      pt.kind != InstrumentationKind::KernelEntry && pt.kind != InstrumentationKind::KernelExit) {
    fail("InstrumentationPoint::kind must be BeforeInst, AfterInst, KernelEntry, or KernelExit");
    return std::nullopt;
  }
  // A probe call needs both halves of the request: the object to resolve the
  // symbol in, and the symbol name
  if ((pt.probe_obj != nullptr) != (!pt.probe_symbol.empty())) {
    report(error_out, "InstrumentationPoint probe_obj and probe_symbol must both be set "
                      "(a probe call) or both be empty (the inline nop)");
    return std::nullopt;
  }
  // The inline nop has nowhere to put arguments. Rejected rather than ignored,
  // so a caller that meant to request a probe call finds out.
  if (pt.probe_obj == nullptr && !pt.probe_args.empty()) {
    fail("InstrumentationPoint::probe_args requires a probe_obj / probe_symbol");
    return std::nullopt;
  }
  // Likewise, the inline nop has no envelope whose mask could be widened.
  if (pt.probe_obj == nullptr && pt.force_full_exec) {
    fail("InstrumentationPoint::force_full_exec requires a probe_obj / probe_symbol");
    return std::nullopt;
  }

  // The bank analysis proves the state before the anchor. On CDNA5, a bank-changing
  // anchor can make the envelope's encoded v0 name v256 instead, invalidating
  // that addressing proof before an AfterInst callback runs.
  if (pt.kind == InstrumentationKind::AfterInst &&
      (pt.probe_obj == nullptr || may_change_vgpr_msb(anchor, arch, setreg_vgpr_msb_fixup))) {
    fail("AfterInst requires a callable probe and an anchor preserving VGPR bank selection");
    return std::nullopt;
  }
  if (pt.kind == InstrumentationKind::AfterInst &&
      may_enable_gpr_indexing(anchor, text_bytes, arch, anchor_offset)) {
    fail("AfterInst anchor may enable GPR indexing");
    return std::nullopt;
  }
  if (pt.kind == InstrumentationKind::AfterInst &&
      may_enable_vskip(anchor, text_bytes, arch, anchor_offset)) {
    fail("AfterInst anchor may enable VSKIP");
    return std::nullopt;
  }
  // No allocation analysis proves that the post-anchor register file can hold
  // the envelope and callback, including SGPR-supplied shrinking allocations.
  if (pt.kind == InstrumentationKind::AfterInst && anchor.mnemonic() == "s_alloc_vgpr") {
    fail("AfterInst cannot prove the VGPR allocation after s_alloc_vgpr");
    return std::nullopt;
  }
  if (pt.kind == InstrumentationKind::AfterInst && deallocates_vgprs(anchor, arch)) {
    fail("AfterInst anchor deallocates VGPRs and scratch");
    return std::nullopt;
  }
  if (pt.kind == InstrumentationKind::AfterInst && anchor.is_mfma()) {
    fail("AfterInst cannot satisfy MFMA software-managed register hazards");
    return std::nullopt;
  }
  // CDNA5 ISA 5.7.2 / 7.12.1 require independent VALUs for co-execution hazards;
  // scalar waits do not provide that spacing before argument copies or spills.
  // Reject whole families conservatively, without inferring a latency class.
  if (pt.kind == InstrumentationKind::AfterInst && arch == ROCJITSU_CODE_ARCH_CDNA5 &&
      may_require_cdna5_valu_spacing(anchor)) {
    fail("AfterInst requires proven VALU spacing for CDNA5 matrix or transcendental anchors");
    return std::nullopt;
  }
  if (pt.kind == InstrumentationKind::BeforeInst || pt.kind == InstrumentationKind::AfterInst) {
    if (!is_relocatable_anchor(anchor, anchor_offset, text_bytes, arch, error_out))
      return std::nullopt;
  } else {
    if (pt.probe_obj == nullptr ||
        !validate_anchor_bytes(anchor, anchor_offset, text_bytes, error_out,
                               pt.kind != InstrumentationKind::KernelEntry)) {
      fail("kernel boundary instrumentation requires a callable probe and valid anchor bytes");
      return std::nullopt;
    }
    if (pt.kind == InstrumentationKind::KernelExit && anchor.mnemonic() != "s_endpgm") {
      fail("KernelExit must name s_endpgm");
      return std::nullopt;
    }
  }

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
  OptionalStringDiagnosticSink record_error(error_out);
  const util::DiagnosticEmitter error(record_error);
  util::FailureOr<uint32_t> nop = build_s_nop(0, plan.arch, &error);
  if (nop.failed())
    return false;
  if (plan.before_items.size() != 1 || plan.before_items[0].words.size() != 1 ||
      plan.before_items[0].words[0] != nop.value()) {
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

bool compute_probe_reserved_registers(std::span<const ProbeCallable> probes,
                                      std::span<const ProbeClobberSummary> summaries,
                                      RegisterSet &out, std::string *error_out) {
  if (probes.size() != summaries.size()) {
    report(error_out, "internal: probe registry and clobber summaries have different sizes");
    return false;
  }
  RegisterSet reserved;
  for (size_t i = 0; i < probes.size(); ++i) {
    reserved |= summaries[i].ordinary_clobbers;
    // The link pair holds the return address across the call, but it is written
    // by the envelope's s_swappc rather than by the body, so no summary reports
    // it. Skipping an ABI that names no usable pair would under-reserve silently.
    if (!is_valid_probe_abi(probes[i].abi)) {
      report(error_out, ("probe '" + probes[i].symbol +
                         "' has an unusable ABI; cannot name the pair it returns through")
                            .c_str());
      return false;
    }
    reserved.expand(probe_link_pair(probes[i].abi));
  }
  out = std::move(reserved);
  return true;
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

// Declare the kernarg wrapper the entry prologue reads from, so a runtime can
// build one without knowing how the prologue was encoded.
//
// The record carries the wrapper's inputs (the guest's kernarg size and the
// payload's size and alignment) rather than the offsets themselves. A consumer
// runs make_kernarg_extension_layout over them and lands on the offsets the
// prologue encoded, because the prologue derived those from the same helper and
// the same descriptor.
//
// An object that already carries this section is refused: the loader-side
// reader stops at the first match, so a second one would be unreachable. The
// check reads the input object rather than the patcher, which is sound only
// because this is the instrumentor's one append_nonalloc_section call.
bool append_dbi_kernarg_record(CodeObjectPatcher &patcher, const AmdGpuCodeObject &obj,
                               const KernelDescriptorInfo &kernel, std::string *error_out) {
  for (const auto &section : obj.all_sections()) {
    if (section->name() != kKernargExtensionMetadataSectionName)
      continue;
    report(error_out, ("the code object already carries a " +
                       std::string(kKernargExtensionMetadataSectionName) +
                       " section, and a second one would be unreachable to the loader; "
                       "instrumenting a code object that already declares a kernarg extension "
                       "is not supported")
                          .c_str());
    return false;
  }

  const KernargExtensionMetadata extension{
      .kernel_name = kernel.kernel_name,
      .variant_name = std::string(kDbiKernargVariantName),
      .original_kernarg_size = kernel.descriptor.kernarg_size,
      .payloads = {{
          .size = kDbiEntryPayloadLayout.size,
          .alignment = kDbiEntryPayloadLayout.alignment,
          .name = std::string(kDbiEntryPayloadName),
      }},
  };
  const std::vector<uint8_t> bytes = serialize_kernarg_extension_metadata(std::span{&extension, 1});
  if (!patcher.append_nonalloc_section(kKernargExtensionMetadataSectionName, bytes)) {
    report(error_out, "failed to append the .rocjitsu.kernarg section");
    return false;
  }
  return true;
}

} // namespace

std::optional<TrampolinePlan> make_trampoline_plan(const ResolvedInstrumentationSite &site,
                                                   rj_code_arch_t arch, uint64_t trampoline_offset,
                                                   std::string *error_out) {
  OptionalStringDiagnosticSink record_error(error_out);
  const util::DiagnosticEmitter error(record_error);
  util::FailureOr<uint32_t> nop = build_s_nop(0, arch, &error);
  if (nop.failed())
    return std::nullopt;
  TrampolinePlan plan = make_base_plan(site, arch, trampoline_offset);
  plan.before_items = {InlineAsmItem{{nop.value()}}};
  plan.after_items = {};
  plan.emit_original = true;
  return plan;
}

namespace {

// Does any probe declare an argument sourced from the framework's entry storage?
// The declaration lives on the probe rather than the point, so this is the whole
// question of whether the patch needs an entry prologue.
[[nodiscard]] bool probes_read_entry_storage(const std::vector<ProbeCallable> &probes) {
  return std::any_of(probes.begin(), probes.end(), [](const ProbeCallable &probe) {
    return std::any_of(probe.arg_sources.begin(), probe.arg_sources.end(), reads_entry_storage);
  });
}

// The probes whose arguments made a prologue necessary, named so a rejection
// does not read as an unprompted failure on an otherwise ordinary kernel.
[[nodiscard]] std::string entry_storage_reader_list(const std::vector<ProbeCallable> &probes) {
  std::string names;
  for (const ProbeCallable &probe : probes) {
    if (!std::any_of(probe.arg_sources.begin(), probe.arg_sources.end(), reads_entry_storage))
      continue;
    if (!names.empty())
      names += ", ";
    names += "'" + probe.symbol + "'";
  }
  return names;
}

} // namespace

std::optional<Instrumentor::EntryProloguePatch> Instrumentor::plan_entry_prologue(
    const std::vector<KernelDescriptorInfo> &kernels, std::optional<uint32_t> kernel_sgpr_count,
    const std::vector<BasicBlock *> &scope, const std::vector<ProbeCallable> &probes,
    const std::vector<ProbeClobberSummary> &summaries, uint64_t cave_offset,
    std::string *error_out) {
  // DBI does not support multiple kernels, so neither does the prologue. Every
  // site that can ask for the storage passes an argument, and that path already
  // requires a single kernel to bound VGPR selection; this gate reports it
  // earlier and names the reason.
  if (kernels.size() != 1) {
    report(error_out, ("a probe reads the framework's entry storage, which instrumentation does "
                       "not yet support on a multi-kernel code object; this one has " +
                       std::to_string(kernels.size()) + " kernels")
                          .c_str());
    return std::nullopt;
  }
  if (!kernel_sgpr_count) {
    report(error_out, "a probe reads the framework's entry storage, but no kernel SGPR "
                      "allocation was discovered to reserve it from");
    return std::nullopt;
  }

  const KernelDescriptorInfo &kernel = kernels.front();
  const uint64_t entry_offset = kernel.entry_text_offset;
  if (find_instruction_at_offset(entry_offset) == nullptr) {
    report(error_out, ("no decoded instruction starts at the kernel entry, .text offset " +
                       std::to_string(entry_offset))
                          .c_str());
    return std::nullopt;
  }

  std::string err;
  RegisterSet reserved;
  if (!compute_probe_reserved_registers(probes, summaries, reserved, error_out))
    return std::nullopt;
  if (options_.register_growth)
    for (uint16_t reg = 0; reg < options_.register_growth->sgprs; ++reg)
      reserved.expand(RegisterRef{RegClass::SGPR, reg, 1});

  std::optional<DbiEntryProloguePlan> planned;
  if (probes_read_entry_storage(probes))
    planned = plan_dbi_entry_prologue(KernelBlockScope(scope), kernel.descriptor, arch_,
                                      *kernel_sgpr_count, reserved, &err);
  if (probes_read_entry_storage(probes) && !planned) {
    // Every rejection the planner reports is a property of the kernel, so its
    // message alone does not say why this kernel was asked to carry a prologue
    // at all. Naming the readers is what makes the failure actionable, and it
    // holds for the storage-exhaustion case and the descriptor ones alike.
    report(
        error_out,
        (err + "; the entry prologue is required by " + entry_storage_reader_list(probes)).c_str());
    return std::nullopt;
  }

  // Dispatch enters at the stub, and nothing else in the kernel can: the
  // original entry keeps its bytes, so a later transfer to it runs guest code
  // only. The prologue is not idempotent, which is why it must not sit on the
  // entry itself.
  EntryProloguePatch patch;
  patch.padding_bytes =
      padding_for_residue(cave_offset, entry_offset % kKernelEntryAlignment, kKernelEntryAlignment);
  patch.stub_offset = cave_offset + patch.padding_bytes;
  if (planned) {
    patch.storage_base = planned->storage.persistent_base;
    patch.words = planned->prologue.words;
    const uint32_t bound = kernel_vgpr_bounds(arch_, kernel.descriptor).ordinary_bound;
    std::vector<uint16_t> sources;
    for (const InstrumentationPoint &point : points_)
      for (const ProbeArgValue &arg : point.probe_args)
        if (arg.source == ProbeArgSource::EntryVgprLane0 &&
            std::ranges::find(sources, arg.register_index) == sources.end())
          sources.push_back(arg.register_index);
    if (sources.size() > 2) {
      report(error_out, "the entry snapshot has room for at most two lane-zero VGPR values");
      return std::nullopt;
    }
    for (size_t index = 0; index < sources.size(); ++index) {
      if (sources[index] >= std::min(bound, 256u)) {
        report(error_out, "entry VGPR capture exceeds the kernel's ordinary VGPR allocation");
        return std::nullopt;
      }
      const uint16_t destination = planned->storage.entry_temp_base + index;
      util::StringDiagnostic lane_error;
      const util::DiagnosticEmitter emit_lane_error = lane_error.emitter();
      const util::FailureOr<std::array<uint32_t, 2>> maybe_words =
          build_v_readlane_b32(destination, sources[index], 0, arch_, &emit_lane_error);
      if (maybe_words.failed()) {
        report(error_out, lane_error.message().c_str());
        return std::nullopt;
      }
      const std::array<uint32_t, 2> &words = maybe_words.value();
      patch.words.insert(patch.words.end(), words.begin(), words.end());
      patch.captured_vgprs.push_back({sources[index], destination});
    }
  }
  const uint64_t branch_pc = patch.stub_offset + patch.words.size() * sizeof(uint32_t);
  const auto simm16 = compute_sopp_branch_simm16(branch_pc, entry_offset);
  if (!simm16) {
    report(error_out, ("the entry-prologue stub is out of branch range of the kernel entry; the "
                       "entry prologue is required by " +
                       entry_storage_reader_list(probes))
                          .c_str());
    return std::nullopt;
  }
  OptionalStringDiagnosticSink record_error(error_out);
  const util::DiagnosticEmitter error(record_error);
  util::FailureOr<uint32_t> branch = build_s_branch(*simm16, arch_, &error);
  if (branch.failed())
    return std::nullopt;
  patch.words.push_back(branch.value());
  return patch;
}

// The object and decoded blocks stay unchanged while points are queued. Compute
// their observation states once, independently of the requested instrumentation.
class Instrumentor::CallableStateAnalysis {
public:
  CallableStateAnalysis(std::span<const std::unique_ptr<BasicBlock>> blocks,
                        std::span<const KernelDescriptorInfo> kernels,
                        std::span<const uint8_t> text, rj_code_arch_t arch) {
    std::unordered_set<uint64_t> entries;
    for (const KernelDescriptorInfo &kernel : kernels)
      entries.insert(kernel.entry_text_offset);
    std::unordered_set<const BasicBlock *> reachable;
    std::vector<const BasicBlock *> work;
    for (const std::unique_ptr<BasicBlock> &block : blocks) {
      if (entries.contains(block->start_offset()))
        work.push_back(block.get());
      for (const Instruction &inst : block->instructions()) {
        states_.try_emplace(&inst);
        has_mfma_producer_ |= inst.is_mfma();
        has_cdna5_producer_ |=
            arch == ROCJITSU_CODE_ARCH_CDNA5 && may_require_cdna5_valu_spacing(inst);
      }
    }
    while (!work.empty()) {
      const BasicBlock *block = work.back();
      work.pop_back();
      if (!reachable.insert(block).second)
        continue;
      work.insert(work.end(), block->successors().begin(), block->successors().end());
      for (const BasicBlock::CallEdge &edge : block->call_edges())
        if (edge.callee != nullptr)
          work.push_back(edge.callee);
    }
    for (const CallableState state :
         {CallableState::GprIndexing, CallableState::Vskip, CallableState::UnavailableStorage}) {
      if (has_callable_state(arch, state))
        analyze_state(blocks, kernels, text, arch, state, entries, reachable);
    }
  }

  [[nodiscard]] bool has_mfma_producer() const { return has_mfma_producer_; }
  [[nodiscard]] bool has_cdna5_producer() const { return has_cdna5_producer_; }

  [[nodiscard]] bool may_be_unsafe(const Instruction &anchor, bool after,
                                   CallableState state) const {
    const std::unordered_map<const Instruction *, PointState>::const_iterator found =
        states_.find(&anchor);
    return found == states_.end() ||
           ((after ? found->second.after : found->second.before) & state_bit(state)) != 0;
  }

private:
  struct PointState {
    uint8_t before = 0;
    uint8_t after = 0;
  };
  struct BlockState {
    StateWrite last_write = StateWrite::Unchanged;
    uint8_t incoming = 0;
    uint8_t outgoing = 0;
  };
  std::unordered_map<const Instruction *, PointState> states_;
  bool has_mfma_producer_ = false;
  bool has_cdna5_producer_ = false;

  static uint8_t state_bit(CallableState state) { return 1u << static_cast<unsigned>(state); }

  void analyze_state(std::span<const std::unique_ptr<BasicBlock>> blocks,
                     std::span<const KernelDescriptorInfo> kernels, std::span<const uint8_t> text,
                     rj_code_arch_t arch, CallableState state,
                     const std::unordered_set<uint64_t> &entries,
                     const std::unordered_set<const BasicBlock *> &reachable) {
    std::unordered_set<const Instruction *> preserving_calls;
    std::unordered_set<const Instruction *> known_returns;
    for (const std::unique_ptr<BasicBlock> &block : blocks) {
      if (block->successor_issue() != BasicBlock::SuccessorIssue::None ||
          block->call_edges().empty())
        continue;
      if (std::ranges::all_of(block->call_edges(), [&](const BasicBlock::CallEdge &edge) {
            return callee_preserves_state(edge, text, arch, state);
          })) {
        preserving_calls.insert(block->terminator());
        if (reachable.contains(block.get()))
          for (const BasicBlock::CallEdge &edge : block->call_edges())
            known_returns.insert(edge.callee->terminator());
      }
    }
    // Unknown transfers can introduce incoming MODE/storage state at every
    // ordinary observation. Bounded closed text can preserve its initial state.
    const bool storage = state == CallableState::UnavailableStorage;
    bool incomplete = false;
    bool missing_fallthrough = false;
    for (const BasicBlock *block : reachable) {
      const BasicBlock::SuccessorIssue issue = block->successor_issue();
      if (issue == BasicBlock::SuccessorIssue::MissingFallthrough) {
        missing_fallthrough = true;
        continue;
      }
      incomplete |=
          issue != BasicBlock::SuccessorIssue::None && !known_returns.contains(block->terminator());
    }
    const bool closed_state = incomplete && callable_state_is_preserved_in_closed_text(
                                                blocks, kernels, text, arch, state);

    // Each bit denotes a possible incoming state; zero has no proven source.
    // Constant writes seed the worklist even inside cycles. Ordinary edges join
    // possibilities, including backedges into descriptor entries.
    constexpr uint8_t kSafe = 1;
    constexpr uint8_t kUnsafe = 2;
    const auto transfer = [](uint8_t incoming, StateWrite write) -> uint8_t {
      if (write == StateWrite::Safe)
        return kSafe;
      if (write == StateWrite::Unsafe)
        return kUnsafe;
      return incoming;
    };
    std::unordered_map<const BasicBlock *, BlockState> summaries;
    std::vector<const BasicBlock *> work(reachable.begin(), reachable.end());
    std::unordered_set<const BasicBlock *> queued = reachable;
    for (const BasicBlock *block : reachable) {
      BlockState &summary = summaries[block];
      for (const Instruction &inst : block->instructions()) {
        const StateWrite write = callable_state_write(inst, text, arch, state, &preserving_calls);
        if (write != StateWrite::Unchanged)
          summary.last_write = write;
      }
    }
    while (!work.empty()) {
      const BasicBlock *block = work.back();
      work.pop_back();
      queued.erase(block);
      BlockState &summary = summaries.at(block);
      summary.incoming = entries.contains(block->start_offset())
                             ? kSafe
                             : (block->predecessors().empty() ? kUnsafe : 0);
      for (const BasicBlock *predecessor : block->predecessors())
        if (reachable.contains(predecessor))
          summary.incoming |= summaries.at(predecessor).outgoing;
      const uint8_t outgoing = transfer(summary.incoming, summary.last_write);
      if (outgoing == summary.outgoing)
        continue;
      summary.outgoing = outgoing;
      for (const BasicBlock *successor : block->successors())
        if (reachable.contains(successor) && queued.insert(successor).second)
          work.push_back(successor);
    }
    const uint8_t bit = state_bit(state);
    for (const std::unique_ptr<BasicBlock> &block : blocks) {
      const bool is_reachable = reachable.contains(block.get());
      uint8_t current = is_reachable ? summaries.at(block.get()).incoming : 0;
      const auto unsafe = [&](uint8_t value, bool after) {
        if (!is_reachable)
          return true;
        // Falling off the original text cannot enter an earlier observation.
        // Retain the existing stricter AfterInst MODE completeness boundary.
        if (incomplete || (!storage && after && missing_fallthrough))
          return !closed_state || (!storage && after);
        return value == 0 || (value & kUnsafe) != 0;
      };
      for (const Instruction &inst : block->instructions()) {
        PointState &point = states_.at(&inst);
        if (unsafe(current, false))
          point.before |= bit;
        current =
            transfer(current, callable_state_write(inst, text, arch, state, &preserving_calls));
        if (unsafe(current, true))
          point.after |= bit;
      }
    }
  }
};

Instrumentor::Instrumentor(const AmdGpuCodeObject &obj, rj_code_arch_t arch,
                           InstrumentationOptions options)
    : obj_(obj), arch_(arch), options_(options) {}

Instrumentor::~Instrumentor() = default;

void Instrumentor::add_point(InstrumentationPoint pt) { points_.push_back(std::move(pt)); }

void Instrumentor::add_point_by_offset(uint64_t anchor_offset, InstrumentationKind kind) {
  InstrumentationPoint pt;
  pt.anchor_offset = anchor_offset;
  pt.kind = kind;
  points_.push_back(std::move(pt));
}

bool Instrumentor::ensure_blocks_built(std::string *error_out) {
  if (!obj_.is_valid()) {
    report(error_out, "cannot instrument an invalid code object");
    return false;
  }
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
  std::vector<uint64_t> kernel_entries;
  if (obj_.text_sections().size() == 1) {
    const Section *text = obj_.text_sections().front();
    const std::span<const uint8_t> image(reinterpret_cast<const uint8_t *>(obj_.image_data()),
                                         obj_.image_size());
    const std::vector<KernelDescriptorInfo> kernels =
        scan_kernel_descriptors(image, text->sectionOffset(), text->size());
    for (const KernelDescriptorInfo &kernel : kernels)
      kernel_entries.push_back(kernel.entry_text_offset);
  }
  auto blocks = BasicBlock::build(obj_, *decoder_, arch_, decode_error.emitter(), kernel_entries);
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

bool Instrumentor::ensure_liveness_built(uint8_t wave_size,
                                         std::span<const KernelDescriptorInfo> kernels,
                                         std::string *error_out) {
  if (liveness_)
    return true;
  if (!ensure_blocks_built(error_out) || obj_.text_sections().size() != 1) {
    report(error_out, "liveness requires one decodable text section");
    return false;
  }
  const Section *text = obj_.text_sections().front();
  std::vector<BasicBlock *> scope;
  LivenessAnalysisOptions options;
  if (arch_ == ROCJITSU_CODE_ARCH_CDNA5) {
    options.arch = arch_;
    options.target = effective_target(obj_, arch_);
    options.text = {reinterpret_cast<const uint8_t *>(text->data()), text->size()};
  }
  for (const std::unique_ptr<BasicBlock> &block : blocks_) {
    scope.push_back(block.get());
    if (kernels.size() == 1 && block->start_offset() == kernels.front().entry_text_offset)
      options.entry_block = block.get();
  }
  liveness_ =
      std::make_unique<LivenessAnalysis>(KernelBlockScope(scope), wave_size, nullptr, options);
  return true;
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
  // A join keeps its fallthrough edge, but its SGPR-held resume edge to the waiting side
  // is unknown. G_FORK also has an unknown SGPR-held target. Liveness can miss registers
  // needed on those paths, allowing scratch allocation to clobber them.
  for (const auto &block : blocks_) {
    for (const Instruction &inst : block->instructions()) {
      if (inst.flags() & BRANCH_STACK) {
        out.errors.push_back("code object uses the conditional branch stack (" +
                             std::string(inst.mnemonic()) + " at .text offset " +
                             std::to_string(inst.src_loc()) +
                             "); liveness across fork/join is not modelled");
        return out;
      }
    }
  }
  // M0 can displace SGPR reads and writes, which ordinary liveness does not
  // model. Callable probes allocate envelope registers from those facts.
  if (std::ranges::any_of(
          points_, [](const InstrumentationPoint &point) { return point.probe_obj != nullptr; })) {
    std::vector<BasicBlock *> probe_scope;
    probe_scope.reserve(blocks_.size());
    for (const std::unique_ptr<BasicBlock> &block : blocks_)
      probe_scope.push_back(block.get());
    if (accesses_sgprs_indirectly_via_movrel(KernelBlockScope(probe_scope))) {
      out.errors.emplace_back("probe calls do not support relative SGPR access (s_movrel*)");
      return out;
    }
  }
  const Section *text = obj_.text_sections().front();
  const std::span<const uint8_t> text_bytes(reinterpret_cast<const uint8_t *>(text->data()),
                                            text->size());
  // A probe call is analyzed and planned at the wave size the object's kernels
  // share, with temps bounded by their SGPR allocation. The point loop refuses a
  // probe call before any of that when either is missing.
  const std::span<const uint8_t> image(reinterpret_cast<const uint8_t *>(obj_.image_data()),
                                       obj_.image_size());
  out.kernels = scan_kernel_descriptors(image, text->sectionOffset(), text->size());
  out.wave_size = common_wavefront_size(arch_, out.kernels);

  const auto &registry = default_isa_target_registry();
  const rj_code_target_id_t destination_target = effective_target(obj_, arch_);
  const IsaGpuTargetDescription *gpu_target = registry.find_gpu_target(destination_target);
  const bool setreg_vgpr_msb_fixup =
      gpu_target == nullptr || gpu_target->capabilities.setreg_vgpr_msb_fixup;
  ProbeEntryState probe_entry_state = ProbeEntryState::Unknown;
  if (arch_ == ROCJITSU_CODE_ARCH_CDNA5 && out.wave_size) {
    if (!ensure_liveness_built(*out.wave_size, out.kernels, &err)) {
      out.errors.push_back(err);
      return out;
    }
    if (out.kernels.size() == 1) {
      bool has_call = false;
      for (const std::unique_ptr<BasicBlock> &block : blocks_) {
        for (const Instruction &instruction : block->instructions())
          has_call |= (instruction.flags() & INDIRECT_CALL) != 0;
      }
      if (!has_call) {
        bool bank_analysis_failed = false;
        const bool zero = std::ranges::all_of(points_, [&](const auto &point) {
          const Instruction *anchor = find_instruction_at_offset(point.anchor_offset);
          if (!anchor)
            return false;
          for (amdgpu::VgprMsbRole role : {amdgpu::VgprMsbRole::Src0, amdgpu::VgprMsbRole::Src1,
                                           amdgpu::VgprMsbRole::Src2, amdgpu::VgprMsbRole::Dst}) {
            const util::FailureOr<std::optional<uint8_t>> bank =
                liveness_->vgpr_msb_bank_before(*anchor, role);
            if (bank.failed()) {
              bank_analysis_failed = true;
              return false;
            }
            if (bank.value() != 0)
              return false;
          }
          return true;
        });
        if (bank_analysis_failed) {
          out.errors.emplace_back("probe entry VGPR bank liveness is unavailable");
          return out;
        }
        if (zero)
          probe_entry_state = ProbeEntryState::ZeroVgprBanks;
      }
    }
  }

  // A probe is identified by the object it came from and the symbol inside it,
  // and its body is copied into the cave once per identity. The rest of the call
  // shape (how many argument dwords, where each one comes from, which lanes the
  // body runs on) describes that one body, so it is recorded on the
  // ProbeCallable rather than keyed here. Two points naming one probe and
  // declaring it differently are not two probes; they are one probe declared
  // twice, and the second declaration is rejected. Immediate values and guest
  // register indices may vary per site.
  //
  // Whichever point resolves first supplies the declaration. Nothing here can do
  // better: a body reveals neither its arity nor its mask policy, so the two
  // declarations are equally credible and the diagnostic names the conflict
  // instead of blaming the later point.
  struct ProbeKey {
    const AmdGpuCodeObject *obj;
    std::string symbol;
  };
  // The shape half of a point's argument list, which belongs to the probe.
  auto arg_sources_of = [](const InstrumentationPoint &pt) {
    std::vector<ProbeArgSource> sources;
    sources.reserve(pt.probe_args.size());
    for (const ProbeArgValue &arg : pt.probe_args)
      sources.push_back(arg.source);
    return sources;
  };
  std::vector<ProbeKey> probe_keys;
  // Helper function to get a probe index for a given InstrumentationPoint
  // If the probe is new, then resolve it and get probe info; add it to
  // probe_keys and out.probes.
  auto resolve_probe_index = [&](const InstrumentationPoint &pt,
                                 std::string &perr) -> std::optional<size_t> {
    const std::vector<ProbeArgSource> sources = arg_sources_of(pt);
    for (size_t i = 0; i < probe_keys.size(); ++i) {
      if (probe_keys[i].obj != pt.probe_obj || probe_keys[i].symbol != pt.probe_symbol)
        continue;
      const ProbeCallable &declared = out.probes[i];
      if (static_cast<size_t>(declared.abi.num_arg_vgprs) != pt.probe_args.size()) {
        perr = "probe '" + pt.probe_symbol + "' was already declared with " +
               std::to_string(declared.abi.num_arg_vgprs) +
               " argument dwords; this point declares " + std::to_string(pt.probe_args.size());
        return std::nullopt;
      }
      if (declared.arg_sources != sources) {
        perr =
            "probe '" + pt.probe_symbol + "' was already declared with different argument sources";
        return std::nullopt;
      }
      if (declared.force_full_exec != pt.force_full_exec) {
        perr = std::string("probe '") + pt.probe_symbol + "' was already declared with " +
               (declared.force_full_exec ? "force_full_exec" : "the anchor mask") +
               "; this point declares " +
               (pt.force_full_exec ? "force_full_exec" : "the anchor mask");
        return std::nullopt;
      }
      return i;
    }
    // Bounded before the narrowing cast below, which would wrap a large count
    // into a small in-range one.
    if (pt.probe_args.size() > kMaxProbeArgVgprs) {
      perr = "probe '" + pt.probe_symbol + "' was given " + std::to_string(pt.probe_args.size()) +
             " arguments; the limit is " + std::to_string(kMaxProbeArgVgprs);
      return std::nullopt;
    }
    const rj_code_target_id_t probe_target = effective_target(*pt.probe_obj, arch_);
    if (destination_target != ROCJITSU_CODE_TARGET_INVALID &&
        probe_target != ROCJITSU_CODE_TARGET_INVALID && destination_target != probe_target) {
      perr = "probe concrete target does not match the destination code-object target";
      return std::nullopt;
    }
    auto sym = resolve_probe_symbol(*pt.probe_obj, pt.probe_symbol, &perr);
    if (!sym)
      return std::nullopt;
    auto callable = build_probe_callable(*pt.probe_obj, *sym, arch_,
                                         static_cast<uint8_t>(pt.probe_args.size()), &perr);
    if (!callable)
      return std::nullopt;
    // Inputs the probe reads that its convention does not supply. Typically a
    // value only the kernel prologue produces -- workitem_id_x in v31, say --
    // which a trampoline at an arbitrary site cannot reproduce, so the probe
    // would read whatever the instrumented kernel left behind. An ordinary
    // uninitialized read lands here too, and is equally unusable.
    std::optional<RegisterSet> live_ins = analyze_probe_live_ins(
        *pt.probe_obj, *sym, arch_, callable->abi, *out.wave_size, &perr, probe_entry_state);
    if (!live_ins)
      return std::nullopt;
    if (!live_ins->none()) {
      perr = "probe '" + pt.probe_symbol + "' reads " + format_register_set(*live_ins) +
             " before defining it, and its ABI (" + std::to_string(pt.probe_args.size()) +
             " argument dwords) does not supply it";
      return std::nullopt;
    }
    callable->arg_sources = sources;
    callable->force_full_exec = pt.force_full_exec;
    out.probes.push_back(std::move(*callable));
    probe_keys.push_back({pt.probe_obj, pt.probe_symbol});
    return out.probes.size() - 1;
  };

  // All-or-nothing: per-point errors accumulate
  std::vector<ResolvedInstrumentationSite> sites;
  std::unordered_set<uint64_t> site_offsets;
  std::unordered_set<uint64_t> entry_offsets;
  sites.reserve(points_.size());
  for (const auto &pt : points_) {
    const Instruction *anchor = find_instruction_at_offset(pt.anchor_offset);
    if (anchor == nullptr) {
      out.errors.emplace_back("no decoded instruction starts at the requested anchor_offset = " +
                              std::to_string(pt.anchor_offset));
      continue;
    }

    std::unordered_set<uint64_t> &offsets =
        pt.kind == InstrumentationKind::KernelEntry ? entry_offsets : site_offsets;
    if (offsets.contains(pt.anchor_offset)) {
      out.errors.emplace_back("multiple points requested the same anchor_offset = " +
                              std::to_string(pt.anchor_offset));
      continue;
    }

    std::string perr;
    std::optional<ResolvedInstrumentationSite> site = validate_anchor(
        *anchor, pt.anchor_offset, text_bytes, pt, arch_, &perr, setreg_vgpr_msb_fixup);
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
      // A temp past the kernel's .sgpr_count would be unallocated, so probe
      // calls always require a kernel descriptor to bound both SGPR selection
      // and register growth.
      if (out.kernels.empty()) {
        out.errors.push_back("probe call at anchor_offset " + std::to_string(pt.anchor_offset) +
                             " requires a discovered kernel descriptor to bound SGPR "
                             "selection, but none was found");
        continue;
      }
      if (!out.wave_size) {
        out.errors.push_back("probe call at anchor_offset " + std::to_string(pt.anchor_offset) +
                             " is in a code object whose kernels mix Wave32 and Wave64; "
                             "DBI currently analyzes kernels and probes at one wave size "
                             "per code object");
        continue;
      }
      bool unsafe_state = false;
      // Dispatch enters a separate stub once. Guest backedges to the original
      // entry do not revisit this callback and cannot change its initial state.
      if (pt.kind != InstrumentationKind::KernelEntry) {
        if (!callable_state_analysis_)
          callable_state_analysis_ =
              std::make_unique<CallableStateAnalysis>(blocks_, out.kernels, text_bytes, arch_);
        // A scalar anchor does not discharge dependencies left by earlier VALUs.
        // Until spacing is proven, exclude the entire decoded object, including
        // unreachable producers and helper bodies rather than guessing ownership.
        if (pt.kind == InstrumentationKind::AfterInst &&
            callable_state_analysis_->has_mfma_producer()) {
          out.errors.emplace_back("AfterInst requires proven MFMA register spacing in a code "
                                  "object containing MFMA or SMFMAC producers");
          continue;
        }
        if (pt.kind == InstrumentationKind::AfterInst &&
            callable_state_analysis_->has_cdna5_producer()) {
          out.errors.emplace_back("AfterInst requires proven VALU spacing in a CDNA5 code object "
                                  "containing matrix or transcendental producers");
          continue;
        }
        for (const CallableState state : {CallableState::GprIndexing, CallableState::Vskip,
                                          CallableState::UnavailableStorage}) {
          if (callable_state_analysis_->may_be_unsafe(
                  *anchor, pt.kind == InstrumentationKind::AfterInst, state)) {
            const std::string reason =
                state == CallableState::UnavailableStorage
                    ? "cannot prove the VGPR allocation and scratch storage"
                    : std::string("may inherit enabled ") +
                          (state == CallableState::GprIndexing ? "GPR indexing" : "VSKIP");
            out.errors.emplace_back("probe observation " + reason +
                                    ", anchor_offset = " + std::to_string(pt.anchor_offset));
            unsafe_state = true;
            break;
          }
        }
      }
      if (unsafe_state)
        continue;
      auto index = resolve_probe_index(pt, perr);
      if (!index) {
        out.errors.push_back(std::move(perr));
        continue;
      }
      site->probe_index = *index;
      site->probe_args = pt.probe_args;
    }

    sites.push_back(std::move(*site));
    offsets.insert(pt.anchor_offset);
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

  if (options_.preservation == GuestStatePreservation::DeclaredRegisters &&
      !options_.register_growth) {
    result.errors.emplace_back("declared-register preservation requires original metadata counts");
    return result;
  }

  if (points_.empty()) {
    result.errors.emplace_back("Instrumentor::patch requires at least one queued point; got zero");
    return result;
  }

  // Entry storage needs a runtime-owned kernarg wrapper. Both patch entry points
  // require the caller to bind it explicitly. Tests may skip this gate to inspect
  // or dispatch a wrapper themselves; the prologue planner's checks still run.
  if (!skip_patch_gates_ && options_.kernarg_binding == DbiKernargBinding::None) {
    std::string readers;
    for (const InstrumentationPoint &pt : points_) {
      if (std::none_of(pt.probe_args.begin(), pt.probe_args.end(),
                       [](const ProbeArgValue &arg) { return reads_entry_storage(arg.source); }))
        continue;
      if (readers.find("'" + pt.probe_symbol + "'") != std::string::npos)
        continue;
      if (!readers.empty())
        readers += ", ";
      readers += "'" + pt.probe_symbol + "'";
    }
    if (!readers.empty()) {
      result.errors.emplace_back(
          "a probe reads the framework's entry storage, which needs a "
          "kernarg wrapper and CallerSuppliedWrapper binding; requested by " +
          readers);
      return result;
    }
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
  const std::vector<KernelDescriptorInfo> &kernels = resolved.kernels;

  // Reuse the bank proof from resolution when present, and otherwise build
  // liveness only for probe calls or register growth.
  // resolve_points() built blocks_ and resolved those probes only because the
  // kernels share a wave size. obj_ owns the Instructions live_before() is keyed
  // on.
  // TODO: scope this to the kernel containing each anchor, like the DBT path,
  // rather than treating every decoded block as one CFG. Safe here: kernels end
  // in s_endpgm, so there are no false cross-kernel fallthrough edges.
  std::vector<BasicBlock *> liveness_scope;
  liveness_scope.reserve(blocks_.size());
  for (const auto &block : blocks_)
    liveness_scope.push_back(block.get());
  if (options_.register_growth && kernels.size() != 1) {
    result.errors.emplace_back("register growth needs one kernel with enumerable register uses");
    return result;
  }
  if (!resolved.probes.empty() || options_.register_growth) {
    std::string analysis_error;
    if (!ensure_liveness_built(*resolved.wave_size, kernels, &analysis_error)) {
      result.errors.push_back(std::move(analysis_error));
      return result;
    }
  }

  // Lay out the appended region as [probe bodies][s_nop padding][entry stub]
  // [trampolines], the padding and stub only when the kernel has an entry
  // prologue. Each distinct probe body is copied once, ahead of the trampolines
  // that call into it, so a trampoline's target address is known before it is
  // emitted and sites sharing a probe share its single body.
  const auto &sites = resolved.sites;
  // Allocate offsets for each probe body, starting at the local cave (the first
  // byte after the original .text). The site loop below continues advancing this
  // same cursor so trampolines follow the probe bodies.
  uint64_t cave_cursor = cave_start;
  for (ProbeCallable &probe : resolved.probes) {
    probe.output_text_offset = cave_cursor;
    cave_cursor += probe.body_words.size() * sizeof(uint32_t);
  }

  // Temporary SGPR-allocation bound. The probe-call return-link pair is fixed
  // by the calling convention, so the kernel must allocate up to it. Derived from
  // the descriptors already scanned above.
  std::optional<uint32_t> kernel_sgpr_count =
      AmdGpuCodeObject::min_kernel_sgpr_count(arch_, kernels);

  // The kernel's VGPR allocation, decoded once: it depends only on the
  // descriptor and the arch, both loop-invariant. Stays all-zero unless exactly
  // one kernel was discovered, since with several there is no single allocation
  // to name; `kernels.size() != 1` is the guard every use tests. The zero state
  // fails closed rather than silently widening a bound: `ordinary_bound == 0`
  // leaves the SGPR bridge scan with nothing to pick, and `acc_count == 0`
  // rejects every AccVGPR index.
  KernelVgprBounds vgpr_bounds = kernels.size() == 1
                                     ? kernel_vgpr_bounds(arch_, kernels.front().descriptor)
                                     : KernelVgprBounds{};
  // What each probe body overwrites. Keyed on the probe, not the site, so it is
  // decoded once per distinct body and indexed alongside resolved.probes.
  std::vector<ProbeClobberSummary> probe_summaries;
  probe_summaries.reserve(resolved.probes.size());
  for (const ProbeCallable &probe : resolved.probes) {
    std::string probe_err;
    std::optional<ProbeClobberSummary> summary =
        build_probe_clobber_summary(probe, *resolved.wave_size, &probe_err);
    if (!summary) {
      result.errors.push_back(std::move(probe_err));
      continue;
    }
    probe_summaries.push_back(std::move(*summary));
  }
  // The site loop indexes this by probe_index, so a short vector would
  // misattribute one probe's clobbers to another.
  if (!result.errors.empty())
    return result;

  uint32_t sgpr_save_base = 0;
  uint32_t vgpr_save_base = 0;
  auto include_registers = [&](const RegisterSet &registers) {
    registers.for_each_ordinary([&](RegisterRef reg) {
      const uint32_t end = uint32_t{reg.index} + reg.width;
      if (reg.cls == RegClass::SGPR)
        result.resources.sgprs = std::max(result.resources.sgprs, end);
      if (reg.cls == RegClass::VGPR)
        result.resources.vgprs = std::max(result.resources.vgprs, end);
    });
  };
  if (options_.register_growth) {
    const util::FailureOr<bool> complete = liveness_->global_vgpr_usage_is_complete();
    if (complete.failed()) {
      result.errors.emplace_back("register growth's liveness is unavailable");
      return result;
    }
    if (!complete.value() || accesses_sgprs_indirectly_via_movrel(liveness_scope)) {
      result.errors.emplace_back("register growth needs one kernel with enumerable register uses");
      return result;
    }
    result.resources = {.sgprs = options_.register_growth->sgprs,
                        .vgprs = options_.register_growth->vgprs};
    RegisterSet guest_registers;
    for (const std::unique_ptr<BasicBlock> &block : blocks_)
      for (const Instruction &instruction : block->instructions()) {
        const InstDefUse effects(instruction, *resolved.wave_size, liveness_->gfx1250_vgpr_msb(),
                                 UnknownVgprDefPolicy::ExpandAll);
        guest_registers |= effects.uses | effects.defs;
      }
    bool invalid = false;
    guest_registers.for_each_ordinary([&](RegisterRef reg) {
      const uint32_t end = uint32_t{reg.index} + reg.width;
      invalid |= reg.cls == RegClass::ACC_VGPR ||
                 (reg.cls == RegClass::SGPR && end > result.resources.sgprs) ||
                 (reg.cls == RegClass::VGPR && end > result.resources.vgprs);
    });
    const uint32_t granule = descriptor_vgpr_granularity_for_wavefront(
        arch_, kernel_wavefront_size(arch_, kernels.front().descriptor));
    invalid |= result.resources.sgprs == 0 ||
               result.resources.sgprs > REGISTER_SET_ALLOCATABLE_SGPRS ||
               result.resources.vgprs == 0 || result.resources.vgprs > 256 ||
               (result.resources.vgprs + granule - 1) / granule !=
                   (kernels.front().descriptor.compute_pgm_rsrc1 & 63) + 1;
    if (invalid) {
      result.errors.emplace_back("register growth's original allocation disagrees with the guest");
      return result;
    }
    // Entry inputs may include ABI padding beyond the registers the kernel uses
    // (notably gfx1100 wave32). Growth must retain that physical allocation;
    // metadata's used-register count need not include the padding.
    result.resources.sgprs =
        std::max(result.resources.sgprs,
                 kernel_descriptor_initial_sgpr_count(arch_, kernels.front().descriptor));
    for (size_t index = 0; index < resolved.probes.size(); ++index) {
      include_registers(probe_summaries[index].ordinary_clobbers);
      include_registers(arg_registers(resolved.probes[index].abi));
      RegisterSet link;
      link.expand(probe_link_pair(resolved.probes[index].abi));
      include_registers(link);
    }
    sgpr_save_base = result.resources.sgprs + kDbiEntryStorageRegisters;
    vgpr_save_base = result.resources.vgprs;
    kernel_sgpr_count = REGISTER_SET_ALLOCATABLE_SGPRS +
                        (arch_is_cdna_4_or_lower(arch_) ? kCdnaSpecialSgprTailReserve : 0);
    vgpr_bounds = {.ordinary_bound = 256, .acc_count = 0};
  }

  // Dispatch-only storage/bootstrap or boundary callback. A production client
  // using entry storage must explicitly accept the runtime wrapper obligation.
  std::optional<EntryProloguePatch> entry_patch;
  std::optional<uint16_t> entry_storage_base;
  const bool has_entry_callback = std::ranges::any_of(
      sites, [](const auto &site) { return site.kind == InstrumentationKind::KernelEntry; });
  for (const ResolvedInstrumentationSite &site : sites) {
    if (site.kind == InstrumentationKind::KernelEntry &&
        (kernels.size() != 1 || site.anchor_offset != kernels.front().entry_text_offset)) {
      result.errors.emplace_back("KernelEntry must name the sole kernel's descriptor entry");
      return result;
    }
  }
  if (probes_read_entry_storage(resolved.probes) || has_entry_callback) {
    std::string err;
    entry_patch = plan_entry_prologue(kernels, kernel_sgpr_count, liveness_scope, resolved.probes,
                                      probe_summaries, cave_cursor, &err);
    if (!entry_patch) {
      result.errors.push_back(std::move(err));
      return result;
    }
    entry_storage_base = entry_patch->storage_base;
    if (entry_storage_base && options_.register_growth) {
      result.resources.sgprs = std::max(result.resources.sgprs,
                                        uint32_t{*entry_storage_base} + kDbiEntryStorageRegisters);
      // Aligning entry storage can advance its snapshots past the initial
      // backup base when the original SGPR count is odd.
      for (const EntryVgprCapture &capture : entry_patch->captured_vgprs)
        sgpr_save_base = std::max(sgpr_save_base, uint32_t{capture.sgpr} + 1);
    }
    cave_cursor = entry_patch->stub_offset + entry_patch->words.size() * sizeof(uint32_t);
  }

  // Register spilling targets a single-kernel code object for now: any site that
  // must spill grows that one kernel's per-lane scratch. The SpillManager is
  // created lazily on the first spilling site and shared by the rest; its final
  // size is written back to the descriptor after all sites succeed.
  std::optional<SpillManager> spills;

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

      // The kernel must likewise own the argument VGPRs. Only asked of a call
      // that passes arguments, so a zero-argument probe call stays independent
      // of the VGPR allocation entirely.
      if (probe.abi.num_arg_vgprs != 0) {
        // vgpr_bounds stays all-zero when no single kernel descriptor was
        // discovered, so `kernels.size() != 1` is the guard, matching every other
        // use. Same fail-closed case as the SGPR bound above: the allocation the
        // argument VGPRs must fit inside is unknown. The zero state would reject
        // any non-zero count anyway; asking here is what makes the diagnostic say
        // which of the two things went wrong.
        if (kernels.size() != 1) {
          result.errors.push_back("probe call at anchor_offset " +
                                  std::to_string(site.anchor_offset) +
                                  " passes arguments, which requires a discovered kernel "
                                  "descriptor to bound VGPR selection, but none was found");
          continue;
        }
        if (!probe_args_fit_in_kernel(vgpr_bounds.ordinary_bound, probe.abi)) {
          const uint16_t last =
              static_cast<uint16_t>(probe.abi.arg_vgpr_base + probe.abi.num_arg_vgprs - 1);
          result.errors.push_back("probe call at anchor_offset " +
                                  std::to_string(site.anchor_offset) + " needs argument VGPRs v" +
                                  std::to_string(probe.abi.arg_vgpr_base) + "..v" +
                                  std::to_string(last) + " but the kernel allocates only " +
                                  std::to_string(vgpr_bounds.ordinary_bound) + " ordinary VGPRs");
          continue;
        }
        // A Wave32 kernel's EXEC is one dword; exec_hi is not part of the mask
        // the guest ran under, so handing it to a probe would deliver whatever
        // the register happens to hold. The probe's own signature is already
        // wave-size specific (uint32_t or uint64_t at compile time), so this is
        // the caller declaring the wrong one, not a gap to paper over.
        if (kernel_wavefront_size(arch_, kernels.front().descriptor) == 32 &&
            std::ranges::any_of(site.probe_args, [](const ProbeArgValue &a) {
              return a.source == ProbeArgSource::AnchorExecHi;
            })) {
          result.errors.push_back(
              "probe call at anchor_offset " + std::to_string(site.anchor_offset) +
              " passes the high dword of the anchor EXEC mask, but the kernel is Wave32 and "
              "has no such dword");
          continue;
        }
      }

      // Callee clobbers (probe body) + liveness at the anchor feed envelope
      // resource selection and the no-spill policy gate.
      const ProbeClobberSummary *summary = &probe_summaries[*site.probe_index];

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
      const util::FailureOr<const RegisterSet *> maybe_live = liveness_->live_before(*anchor);
      if (maybe_live.failed()) {
        result.errors.emplace_back("internal: anchor liveness is unavailable");
        continue;
      }
      RegisterSet live = *maybe_live.value() | probe_argument_registers(site.probe_args);
      // Including original definitions conservatively covers live-after state:
      // the callback must observe and preserve the relocated instruction's result.
      if (site.kind == InstrumentationKind::AfterInst)
        live |= InstDefUse(*anchor, *resolved.wave_size, liveness_->gfx1250_vgpr_msb(),
                           UnknownVgprDefPolicy::ExpandAll)
                    .defs;
      if (options_.register_growth &&
          options_.preservation == GuestStatePreservation::DeclaredRegisters) {
        for (uint16_t reg = 0; reg < options_.register_growth->sgprs; ++reg)
          live.expand(RegisterRef{RegClass::SGPR, reg, 1});
        for (uint16_t reg = 0; reg < options_.register_growth->vgprs; ++reg)
          live.expand(RegisterRef{RegClass::VGPR, reg, 1});
      }

      // Set the probe's offset
      TrampolinePlan plan = make_base_plan(site, arch_, trampoline_offset);
      plan.probe_target_offset = probe.output_text_offset;
      plan.probe_after_original = site.kind == InstrumentationKind::AfterInst;
      if (site.kind == InstrumentationKind::KernelEntry) {
        plan.emit_original = false;
        plan.patch_anchor = false;
        plan.return_target = site.anchor_offset;
      }
      if (site.kind == InstrumentationKind::KernelExit)
        plan.returns_to_kernel = false;
      // Orchestrator policy: save EXEC/VCC unconditionally so correctness does
      // not depend on complete per-opcode implicit-def coverage. The summary
      // sees decoder-exposed operands; M0 follows touches_m0, which includes
      // exposed reads as well as writes.
      plan.preserve_exec = true;
      plan.preserve_vcc = true;
      plan.preserve_m0 = summary->touches_m0;
      // Cap envelope/temp SGPR selection at the kernel's own allocation so a temp
      // never lands past its .sgpr_count
      plan.kernel_sgpr_count = *kernel_sgpr_count;
      plan.kernel_vgpr_count = vgpr_bounds.ordinary_bound;
      plan.probe_args = site.probe_args;
      plan.force_full_exec = probe.force_full_exec;
      // Set on every site of a kernel that has a prologue, not only the ones
      // passing the pointer on: the pair has to reach them all, so the envelope
      // has to stay off it everywhere.
      plan.entry_storage_base = entry_storage_base;
      if (entry_patch)
        plan.entry_vgpr_captures = entry_patch->captured_vgprs;
      if (options_.register_growth) {
        const RegisterSet overwritten =
            (summary->ordinary_clobbers | arg_registers(probe.abi)) & live;
        RegisterSet fixed_link;
        fixed_link.expand(probe_link_pair(probe.abi));
        const RegisterSet saved = overwritten | (fixed_link & live);
        uint32_t next_sgpr = sgpr_save_base;
        uint32_t next_vgpr = vgpr_save_base;
        bool unavailable = false;
        saved.for_each_ordinary([&](RegisterRef reg) {
          for (uint32_t lane = 0; lane < reg.width; ++lane) {
            uint32_t &destination = reg.cls == RegClass::SGPR ? next_sgpr : next_vgpr;
            if ((reg.cls != RegClass::SGPR && reg.cls != RegClass::VGPR) ||
                destination >=
                    (reg.cls == RegClass::SGPR ? REGISTER_SET_ALLOCATABLE_SGPRS : 256u)) {
              unavailable = true;
              continue;
            }
            plan.register_saves.push_back({reg.cls, static_cast<uint16_t>(reg.index + lane),
                                           static_cast<uint16_t>(destination++)});
          }
        });
        if (unavailable) {
          result.errors.emplace_back("register preservation exceeds the bounded register file");
          continue;
        }
      }
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
      if (options_.register_growth) {
        include_registers(plan.builder_clobbers);
      } else if (!spill.none()) {
        // Single-kernel assumption: spilling needs exactly one kernel descriptor
        // with non-zero fixed scratch to grow. That is the same condition under
        // which vgpr_bounds was decoded, so testing it here is what licenses the
        // reads below.
        if (kernels.size() != 1 || kernels.front().descriptor.private_segment_fixed_size == 0) {
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
        // Defensively exclude the argument VGPRs from bridge selection as well
        // as the live set.
        const RegisterSet bridge_unavailable = live | arg_registers(probe.abi);
        if (!sgpr_spill.none() &&
            !plan_sgpr_spills(sgpr_spill, bridge_unavailable, plan.vgpr_spills,
                              vgpr_bounds.ordinary_bound, *spills, arch_, plan.sgpr_spills,
                              plan.spill_bridge_vgpr, &err)) {
          result.errors.push_back(std::move(err));
          continue;
        }
        // AccVGPRs (CDNA only): reject an index past the allocated AGPR window.
        if (!acc_spill.none() && !plan_acc_spills(acc_spill, vgpr_bounds.acc_count, *spills, arch_,
                                                  plan.acc_spills, &err)) {
          result.errors.push_back(std::move(err));
          continue;
        }
      }

      // Get trampoline from TrampolineBuilder
      bytes = TrampolineBuilder::emit_probe_call(plan, &err);
      if (plan.returns_to_kernel)
        record.return_target = plan.return_target;
      record.is_probe_call = true;
      record.probe_target_offset = probe.output_text_offset;
      record.link_pair_base = plan.link_pair_base;
      record.target_pair_base = plan.target_pair_base;
      // If this site does not call a probe
      // Currently, this means that the plan is to put a nop which means we do
      // not need to mess with spills
    } else {
      std::optional<TrampolinePlan> maybe_plan =
          make_trampoline_plan(site, arch_, trampoline_offset, &err);
      if (!maybe_plan) {
        result.errors.push_back(std::move(err));
        continue;
      }
      const TrampolinePlan &plan = *maybe_plan;
      // The only assembly we currently allow is an inlined nop
      if (!validate_inline_nop_plan(plan, &err)) {
        result.errors.push_back(std::move(err));
        continue;
      }
      bytes = TrampolineBuilder::build(plan, &err);
      if (plan.returns_to_kernel)
        record.return_target = plan.return_target;
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

  // Redirect at the original descriptor offset; replace_text then shifts the
  // entry field with everything else. Resource sizes are committed together
  // with their metadata after text replacement.
  if (entry_patch &&
      !patcher.redirect_kernel_entry(kernels.front().descriptor_file_offset,
                                     kernels.front().entry_text_offset, entry_patch->stub_offset)) {
    result.errors.emplace_back("failed to redirect the kernel entry to the entry-prologue stub");
    return result;
  }

  // Every per-site validation, branch-range check, and trampoline-byte
  // construction has succeeded up to this point. Assemble the new .text in one
  // buffer: the original bytes with each anchor spliced to its forward branch,
  // followed by the local cave. replace_text() grows .text in place and fixes
  // up the surrounding ELF (section/segment sizes, moved symbols, descriptor
  // entries).
  const auto text_span = patcher.text_bytes();
  std::vector<uint8_t> new_text(text_span.begin(), text_span.end());
  for (const auto &a : applied) {
    if (a.site->kind == InstrumentationKind::KernelEntry) {
      const uint64_t branch_pc =
          entry_patch->stub_offset + (entry_patch->words.size() - 1) * sizeof(uint32_t);
      const std::optional<int16_t> target =
          compute_sopp_branch_simm16(branch_pc, a.trampoline_offset);
      if (!target) {
        result.errors.emplace_back("kernel-entry callback is outside the stub's branch range");
        return result;
      }
      auto record_error = [&](std::string_view message) { result.errors.emplace_back(message); };
      const util::DiagnosticEmitter error(record_error);
      const util::FailureOr<uint32_t> branch = build_s_branch(*target, arch_, &error);
      if (branch.failed())
        return result;
      entry_patch->words.back() = branch.value();
    } else {
      std::memcpy(new_text.data() + a.site->anchor_offset, a.bytes.patched_anchor_bytes.data(),
                  a.site->original_size);
    }
  }
  // Append in the laid-out order: probe bodies first (one per distinct probe),
  // then the padding and the entry stub, then the per-site trampolines.
  for (const ProbeCallable &probe : resolved.probes)
    append_words(new_text, probe.body_words);
  if (entry_patch) {
    auto record_error = [&](std::string_view message) { result.errors.emplace_back(message); };
    const util::DiagnosticEmitter error(record_error);
    if (append_nop_padding(new_text, entry_patch->padding_bytes, arch_, &error).failed()) {
      return result;
    }
    append_words(new_text, entry_patch->words);
  }
  for (const auto &a : applied)
    append_words(new_text, a.bytes.trampoline_words);
  if (!patcher.replace_text(new_text)) {
    result.errors.emplace_back("failed to replace .text with the instrumented code");
    return result;
  }

  if (entry_patch && entry_patch->storage_base) {
    std::string err;
    if (!append_dbi_kernarg_record(patcher, obj_, kernels.front(), &err)) {
      result.errors.push_back(std::move(err));
      return result;
    }
  }

  // Emit and build patch summaries.
  std::vector<uint8_t> image = patcher.emit();
  if (spills || options_.register_growth ||
      options_.kernarg_binding == DbiKernargBinding::CallerSuppliedWrapper) {
    AmdGpuCodeObject updated(image.data(), image.size());
    if (!updated.is_valid() || updated.text_sections().size() != 1) {
      result.errors.emplace_back("resource update cannot reparse the instrumented image");
      return result;
    }
    const Section *updated_text = updated.text_sections().front();
    const std::vector<KernelDescriptorInfo> updated_kernels =
        scan_kernel_descriptors(image, updated_text->sectionOffset(), updated_text->size());
    if (updated_kernels.size() != 1) {
      result.errors.emplace_back("resource update requires the instrumented kernel descriptor");
      return result;
    }
    const KernelDescriptorInfo &kernel = updated_kernels.front();
    amdhsa::kernel_descriptor_t descriptor = kernel.descriptor;
    if (spills)
      descriptor.private_segment_fixed_size = spills->total_private_bytes();
    result.resources.kernarg_bytes = descriptor.kernarg_size;
    // A binding authorizes the wrapper needed by entry-storage readers. Calls
    // without that prologue must retain the guest's original kernarg ABI.
    if (entry_storage_base &&
        options_.kernarg_binding == DbiKernargBinding::CallerSuppliedWrapper) {
      const std::array payloads{kDbiEntryPayloadLayout};
      const std::optional<KernargExtensionLayout> wrapper =
          make_kernarg_extension_layout(descriptor.kernarg_size, payloads);
      if (!wrapper) {
        result.errors.emplace_back("runtime kernarg wrapper size overflows");
        return result;
      }
      result.resources.kernarg_bytes = wrapper->wrapper_size;
      descriptor.kernarg_size = wrapper->wrapper_size;
    }
    if (options_.register_growth) {
      const uint32_t granule = descriptor_vgpr_granularity_for_wavefront(
          arch_, kernel_wavefront_size(arch_, descriptor));
      result.resources.vgprs = (result.resources.vgprs + granule - 1) / granule * granule;
      if (result.resources.sgprs > REGISTER_SET_ALLOCATABLE_SGPRS || result.resources.vgprs > 256) {
        result.errors.emplace_back("instrumentation exceeds the bounded register allocation");
        return result;
      }
      RJ_AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                         amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                         result.resources.vgprs / granule - 1);
      if (arch_ == ROCJITSU_CODE_ARCH_CDNA3 || arch_ == ROCJITSU_CODE_ARCH_CDNA4) {
        RJ_AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                           amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                           (result.resources.sgprs + kCdnaSpecialSgprTailReserve + 7) / 8 - 1);
        RJ_AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3,
                           amdhsa::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET,
                           result.resources.vgprs / 4 - 1);
      }
    }
    if (spills || options_.register_growth || entry_storage_base) {
      CodeObjectPatcher resource_patcher{AmdGpuCodeObject(image.data(), image.size())};
      std::string error;
      const std::optional<uint32_t> sgprs =
          options_.register_growth ? std::optional<uint32_t>{result.resources.sgprs} : std::nullopt;
      const std::optional<uint32_t> vgprs =
          options_.register_growth ? std::optional<uint32_t>{result.resources.vgprs} : std::nullopt;
      if (!resource_patcher.patch_kernel_resources(kernel, descriptor, sgprs, vgprs, &error)) {
        result.errors.push_back("resource metadata/descriptor update failed: " + error);
        return result;
      }
      image = resource_patcher.emit();
    }
  }
  result.elf_bytes = std::move(image);
  for (const auto &a : applied) {
    InstrumentationPatch patch;
    patch.anchor_offset = a.site->anchor_offset;
    patch.original_size = a.site->original_size;
    patch.trampoline_offset = a.trampoline_offset;
    patch.return_target = a.return_target;
    patch.original_bytes = a.site->original_bytes;
    patch.patched_anchor_bytes = a.bytes.patched_anchor_bytes;
    patch.is_probe_call = a.is_probe_call;
    if (a.is_probe_call) {
      patch.probe_symbol = resolved.probes[*a.site->probe_index].symbol;
      patch.probe_target_offset = a.probe_target_offset;
      patch.link_pair_base = a.link_pair_base;
      patch.target_pair_base = a.target_pair_base;
    }
    // The builder owns the layout, including the call envelope's attribution.
    result.sites.push_back(
        {.kind = a.site->kind,
         .original_offset = a.site->anchor_offset,
         .guest_offset = a.trampoline_offset + a.bytes.original_byte_offset,
         .guest_size =
             a.site->kind == InstrumentationKind::KernelEntry ? 0u : a.site->original_size,
         .instrumentation_offset =
             a.trampoline_offset + (a.is_probe_call ? a.bytes.call_envelope_byte_offset : 0),
         .instrumentation_size =
             a.is_probe_call ? a.bytes.call_envelope_byte_size : a.bytes.original_byte_offset});
    result.patches.push_back(std::move(patch));
  }
  return result;
}

} // namespace rocjitsu
