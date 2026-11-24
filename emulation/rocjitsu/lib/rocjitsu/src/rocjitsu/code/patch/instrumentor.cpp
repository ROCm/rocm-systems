// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/instrumentor.h"
#include "rocjitsu/isa/isa_family.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/analysis/control_flow.h"
#include "rocjitsu/code/analysis/def_use_chain.h"
#include "rocjitsu/code/analysis/exec_state.h"
#include "rocjitsu/code/analysis/gfx1250_vgpr_msb.h"
#include "rocjitsu/code/analysis/liveness.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/builders/spill_builders.h"
#include "rocjitsu/code/code_object.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/code/kernel_scope.h"
#include "rocjitsu/code/patch/checked_text_edits.h"
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
#include "rocjitsu/code/scoped_cfg_edges.h"
#include "rocjitsu/code/text_padding.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/sgpr_layout.h"
#include "rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_selectors.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/isa_traits.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/isa/target_registry.h"
#include "util/diagnostic.h"
#include "util/result.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstring>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocjitsu {

// The immutable source/profile binding is enforced by Instrumentor: only its
// own source initializes these facts, and fork() retains that exact binding.
// Storage owns the index/references, never the borrowed ELF, decoder or CFG.
struct SourceTextFacts {
  struct SourceLocation {
    uint64_t address;
    std::optional<uint64_t> text_offset;
  };
  struct ControlFlowEdge {
    SourceLocation source;
    uint64_t destination;
    bool fallthrough = false;
  };
  InstructionBoundaries boundaries;
  TextEditReferences references;
  std::vector<SourceLocation> pc_reads;
  std::vector<ControlFlowEdge> flow_edges;
  std::vector<uint64_t> foreign_kernel_entries;
};

struct Instrumentor::ResolvedPoints {
  std::vector<ResolvedInstrumentationSite> sites;
  std::vector<ProbeCallable> probes;
  std::vector<std::string> errors;
  std::optional<TextEditError> edit_error;
  // The object's kernel descriptors and the wave size they share. A probe-call
  // site is resolved only when there is at least one kernel and one wave size.
  std::vector<KernelDescriptorInfo> kernels;
  std::optional<uint8_t> wave_size;
};

namespace {

rj_code_target_id_t effective_target(const AmdGpuCodeObject &object, rocjitsu::IsaFamily arch) {
  if (object.target_id() != ROCJITSU_CODE_TARGET_INVALID)
    return object.target_id();
  const IsaTargetRegistry &registry = default_isa_target_registry();
  const IsaTargetDescriptor *descriptor = registry.find(arch);
  if (descriptor == nullptr)
    return ROCJITSU_CODE_TARGET_INVALID;
  const IsaTargetBinding *target = registry.find_default_target(*descriptor);
  return target == nullptr ? ROCJITSU_CODE_TARGET_INVALID : target->public_id;
}

// CDNA3/CDNA4 retain a second dispatch entry for compatible preload firmware.
// Hardware preloading on gfx1250 uses only the descriptor entry.
bool has_kernarg_preload_firmware_entry(const KernelDescriptorInfo &kernel,
                                        rocjitsu::IsaFamily arch) {
  return (arch == rocjitsu::IsaFamily::Cdna3 || arch == rocjitsu::IsaFamily::Cdna4) &&
         kernarg_preload_length(kernel.descriptor) != 0;
}

std::optional<uint64_t> kernarg_preload_firmware_entry(const KernelDescriptorInfo &kernel,
                                                       rocjitsu::IsaFamily arch,
                                                       uint64_t text_size) {
  constexpr uint64_t kFirmwareSkip = 256;
  if (!has_kernarg_preload_firmware_entry(kernel, arch) || kernel.entry_text_offset >= text_size ||
      text_size - kernel.entry_text_offset <= kFirmwareSkip)
    return std::nullopt;
  return kernel.entry_text_offset + kFirmwareSkip;
}

bool requires_cdna5_valu_follower(const Instruction &inst) {
  // These gfx1250/1251 permutes need an immediately issuing safe VALU.
  // https://github.com/llvm/llvm-project/commit/0c8e5777423c2e657d7e5c61b2cc812253f9b3ec
  const std::string_view mnemonic = inst.mnemonic();
  return mnemonic == "v_perm_pk16_b4_u4" || mnemonic == "v_perm_pk16_b6_u4" ||
         mnemonic == "v_perm_pk16_b8_u4";
}

// Ownership requires a complete descriptor inventory. The shared scanner and
// descriptor rebasing currently recognize only global symbols and deduplicate
// by location, so check exported definitions before those distinctions are lost.
bool validate_entry_descriptor_symbols(std::span<const uint8_t> image,
                                       std::span<const KernelDescriptorInfo> kernels,
                                       std::string *error_out) {
  const auto fail = [error_out](const char *message) {
    report(error_out, message);
    return false;
  };
  if (image.size() < sizeof(Elf64_Ehdr))
    return fail("entry requests require a complete descriptor symbol inventory");
  Elf64_Ehdr header = {};
  std::memcpy(&header, image.data(), sizeof(header));
  const std::optional<std::vector<Elf64_Shdr>> sections =
      read_elf_table<Elf64_Shdr>(image, header.e_shoff, header.e_shnum, header.e_shentsize);
  if (!sections)
    return fail("entry requests require a complete descriptor symbol inventory");
  for (const Elf64_Shdr &table : *sections) {
    if (table.sh_type != SHT_SYMTAB && table.sh_type != SHT_DYNSYM)
      continue;
    if (table.sh_entsize == 0 || table.sh_size % table.sh_entsize != 0 ||
        table.sh_link >= sections->size())
      return fail("entry requests require a complete descriptor symbol inventory");
    const std::optional<std::vector<Elf64_Sym>> symbols = read_elf_table<Elf64_Sym>(
        image, table.sh_offset, table.sh_size / table.sh_entsize, table.sh_entsize);
    const Elf64_Shdr &strings = (*sections)[table.sh_link];
    if (!symbols || strings.sh_offset > image.size() ||
        strings.sh_size > image.size() - strings.sh_offset)
      return fail("entry requests require a complete descriptor symbol inventory");
    for (const Elf64_Sym &symbol : *symbols) {
      const uint8_t binding = elf_symbol_bind(symbol.st_info);
      if (elf_symbol_type(symbol.st_info) != kElfSymbolTypeObject ||
          (binding != kElfSymbolBindGlobal && binding != kElfSymbolBindWeak) ||
          symbol.st_shndx == SHN_UNDEF)
        continue;
      if (symbol.st_name >= strings.sh_size)
        return fail("entry requests require a complete descriptor symbol inventory");
      const char *name =
          reinterpret_cast<const char *>(image.data() + strings.sh_offset + symbol.st_name);
      const size_t available = strings.sh_size - symbol.st_name;
      const size_t length = strnlen(name, available);
      if (length == available)
        return fail("entry requests require a complete descriptor symbol inventory");
      const std::string_view symbol_name(name, length);
      if (!symbol_name.ends_with(".kd"))
        continue;
      if (binding == kElfSymbolBindWeak)
        return fail("entry requests do not admit weak kernel descriptor symbols");
      if (symbol.st_shndx >= sections->size() ||
          symbol.st_size != sizeof(amdhsa::kernel_descriptor_t))
        return fail("entry requests require a complete descriptor symbol inventory");
      const Elf64_Shdr &owner = (*sections)[symbol.st_shndx];
      if (owner.sh_type == SHT_NOBITS || symbol.st_value < owner.sh_addr ||
          owner.sh_offset > image.size() || owner.sh_size > image.size() - owner.sh_offset)
        return fail("entry requests require a complete descriptor symbol inventory");
      const uint64_t delta = symbol.st_value - owner.sh_addr;
      if (delta > owner.sh_size || symbol.st_size > owner.sh_size - delta)
        return fail("entry requests require a complete descriptor symbol inventory");
      const uint64_t descriptor_offset = owner.sh_offset + delta;
      const std::span<const KernelDescriptorInfo>::iterator found = std::ranges::find(
          kernels, descriptor_offset, &KernelDescriptorInfo::descriptor_file_offset);
      if (found == kernels.end())
        return fail("entry requests require a complete descriptor symbol inventory");
      if (found->kernel_name != symbol_name.substr(0, length - 3))
        return fail("entry requests do not admit descriptor aliases with distinct names");
    }
  }
  return true;
}

// These ISA message tables assign MSG_DEALLOC_VGPRS to message 3. Older
// architectures give that encoding different semantics.
bool deallocates_vgprs(const Instruction &instruction, rocjitsu::IsaFamily arch) {
  if (arch != rocjitsu::IsaFamily::Cdna5 && arch != rocjitsu::IsaFamily::Rdna3 &&
      arch != rocjitsu::IsaFamily::Rdna3_5 && arch != rocjitsu::IsaFamily::Rdna4)
    return false;
  if (instruction.mnemonic() != "s_sendmsg" && instruction.mnemonic() != "s_sendmsghalt")
    return false;
  const Operand *message = instruction.src_operand(0);
  constexpr uint32_t kDeallocVgprsMessage = 3;
  // In-tree OPR_SENDMSG.MSG is 9/4/7/7 bits for RDNA3/3.5/4/CDNA5.
  // LLVM SendMsg::ID_MASK_GFX11Plus_ (SIDefines.h) uses the low byte. Until these descriptions
  // agree, reject deallocation under either interpretation (including 0x103).
  const uint32_t message_mask = arch == rocjitsu::IsaFamily::Rdna3     ? 0x1ffu
                                : arch == rocjitsu::IsaFamily::Rdna3_5 ? 0xfu
                                                                       : 0x7fu;
  return message == nullptr || (message->encoding_value() & message_mask) == kDeallocVgprsMessage ||
         (message->encoding_value() & 0xffu) == kDeallocVgprsMessage;
}

bool uses_dynamic_vgprs(rocjitsu::IsaFamily arch, const amdhsa::kernel_descriptor_t &descriptor) {
  if (arch == rocjitsu::IsaFamily::Rdna4)
    return (descriptor.compute_pgm_rsrc2 & amdhsa::COMPUTE_PGM_RSRC2_GFX120_ENABLE_DYNAMIC_VGPR) !=
           0;
  return arch == rocjitsu::IsaFamily::Cdna5 &&
         (descriptor.compute_pgm_rsrc3 & amdhsa::COMPUTE_PGM_RSRC3_GFX125_ENABLE_DYNAMIC_VGPR) != 0;
}

enum class CallableState { GprIndexing, Vskip, UnavailableStorage };

enum class StateWrite { Unchanged, Safe, Unsafe };

bool has_callable_state(rocjitsu::IsaFamily arch, CallableState state) {
  switch (state) {
  case CallableState::GprIndexing:
    return isa_properties(arch).mode_has_gpr_idx_en;
  case CallableState::Vskip:
    return arch_is_cdna_4_or_lower(arch);
  case CallableState::UnavailableStorage:
    return arch == rocjitsu::IsaFamily::Cdna5 || arch == rocjitsu::IsaFamily::Rdna3 ||
           arch == rocjitsu::IsaFamily::Rdna3_5 || arch == rocjitsu::IsaFamily::Rdna4;
  }
  return false;
}

bool has_unavailable_branch_reference(const Instruction &instruction) {
  const amdgpu::Operation operation = instruction.amdgpu_operation();
  return !instruction.branch_offset_bytes() &&
         (operation == amdgpu::Operation::S_CBRANCH_CDBGSYS ||
          operation == amdgpu::Operation::S_CBRANCH_CDBGUSER ||
          operation == amdgpu::Operation::S_CBRANCH_CDBGSYS_AND_USER ||
          operation == amdgpu::Operation::S_CBRANCH_CDBGSYS_OR_USER ||
          operation == amdgpu::Operation::S_SUBVECTOR_LOOP_BEGIN ||
          operation == amdgpu::Operation::S_SUBVECTOR_LOOP_END);
}

bool reject_unavailable_branch_reference(const Instruction &instruction, TextEditError &error) {
  if (!has_unavailable_branch_reference(instruction))
    return false;
  error = {.code = TextEditErrorCode::UnsupportedReferences,
           .offset = instruction.src_loc(),
           .message = "branch destination metadata is unavailable for " +
                      std::string(instruction.mnemonic())};
  return true;
}

StateWrite
callable_state_write(const Instruction &inst, std::span<const uint8_t> text,
                     rocjitsu::IsaFamily arch, CallableState state,
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

using HelperEntryLinks = std::unordered_map<uint64_t, std::optional<uint16_t>>;
using CalleeReturns = std::unordered_map<const BasicBlock *, std::vector<const Instruction *>>;

bool preserves_helper_link(const Instruction &inst, RegisterRef link, uint32_t ordinary_bound) {
  // A numeric SGPR in the special-register tail can also be defined through a
  // named or implicit operand. A nop preserves either spelling of the link.
  if (uint32_t{link.index} + link.width > ordinary_bound)
    return inst.mnemonic() == "s_nop";
  return !inst.mnemonic().starts_with("s_movrel") && !InstDefUse(inst, 64).defs.intersects(link);
}

// A call supplies the link only at the helper entry. Visit a join once all its
// predecessors have been proven, so external branches and cycles fail closed.
std::vector<const Instruction *>
preserving_callee_returns(const BasicBlock &callee, uint16_t return_sreg, uint32_t ordinary_bound,
                          const HelperEntryLinks &entry_links, std::span<const uint8_t> text,
                          rocjitsu::IsaFamily arch, CallableState state) {
  if (!callee.predecessors().empty())
    return {};
  const RegisterRef link = {RegClass::SGPR, return_sreg, 2};
  std::vector<const Instruction *> returns;
  std::vector<const BasicBlock *> ready = {&callee};
  std::unordered_map<const BasicBlock *, size_t> pending;
  while (!ready.empty()) {
    const BasicBlock *block = ready.back();
    ready.pop_back();
    const Instruction *term = block->terminator();
    if (term == nullptr || !block->call_edges().empty())
      return {};
    const bool is_return = term->mnemonic() == "s_setpc_b64" || term->mnemonic() == "s_set_pc_i64";
    for (const Instruction &inst : block->instructions()) {
      // A descriptor or another callee entry can reach this instruction with
      // a different return link, bypassing the ordinary predecessor list.
      if (inst.src_loc() != callee.start_offset() && entry_links.contains(inst.src_loc()))
        return {};
      if (inst.mnemonic().starts_with("s_rfe") ||
          callable_state_write(inst, text, arch, state) != StateWrite::Unchanged ||
          ((!is_return || &inst != term) && !preserves_helper_link(inst, link, ordinary_bound)))
        return {};
    }
    if (is_return) {
      const Operand *source = term->src_operand(0);
      if (!block->successors().empty() || source == nullptr || source->to_register_ref() != link)
        return {};
      returns.push_back(term);
      continue;
    }
    if (block->successor_issue() != BasicBlock::SuccessorIssue::None || block->successors().empty())
      return {};
    for (const BasicBlock *successor : block->successors()) {
      if (++pending[successor] == successor->predecessors().size()) {
        pending.erase(successor);
        ready.push_back(successor);
      }
    }
  }
  if (!pending.empty())
    return {};
  return returns;
}

// Recognize a contiguous negative 64-bit PC adjustment with its carry intact.
// Restrict hidden edges to dispatch roots so they cannot enter another helper
// with a return-link context supplied only by that helper's own callers.
std::optional<uint64_t> adjusted_pc_entry_target(const std::array<const Instruction *, 3> &tail,
                                                 uint64_t base, uint16_t link,
                                                 std::span<const KernelDescriptorInfo> kernels) {
  if (tail[0] == nullptr || tail[0]->src_loc() + tail[0]->size() != tail[1]->src_loc() ||
      tail[1]->src_loc() + tail[1]->size() != tail[2]->src_loc())
    return std::nullopt;
  const std::string_view low_name = tail[0]->mnemonic();
  const std::string_view high_name = tail[1]->mnemonic();
  const std::string_view ret_name = tail[2]->mnemonic();
  if ((low_name != "s_add_u32" && low_name != "s_add_co_u32") ||
      (high_name != "s_addc_u32" && high_name != "s_add_co_ci_u32") ||
      (ret_name != "s_setpc_b64" && ret_name != "s_set_pc_i64"))
    return std::nullopt;
  const auto is_link = [](const Operand *operand, uint16_t index, uint8_t width = 1) {
    return operand != nullptr &&
           operand->to_register_ref() == RegisterRef{RegClass::SGPR, index, width};
  };
  const uint16_t high = link + 1;
  if (!is_link(tail[0]->dst_operand(0), link) || !is_link(tail[0]->src_operand(0), link) ||
      !is_link(tail[1]->dst_operand(0), high) || !is_link(tail[1]->src_operand(0), high) ||
      !is_link(tail[2]->src_operand(0), link, 2))
    return std::nullopt;
  const Operand *low_value = tail[0]->src_operand(1);
  const Operand *high_value = tail[1]->src_operand(1);
  if (low_value == nullptr || high_value == nullptr)
    return std::nullopt;
  const std::optional<uint64_t> low_constant = low_value->const_value();
  const std::optional<uint64_t> high_constant = high_value->const_value();
  if (!low_constant || !high_constant || static_cast<uint32_t>(*high_constant) != 0xffffffffu)
    return std::nullopt;
  const int64_t delta = static_cast<int32_t>(static_cast<uint32_t>(*low_constant));
  if (delta >= 0 || base < static_cast<uint64_t>(-delta))
    return std::nullopt;
  const uint64_t target = base - static_cast<uint64_t>(-delta);
  if (!std::ranges::any_of(kernels, [&](const KernelDescriptorInfo &kernel) {
        return kernel.entry_text_offset == target;
      }))
    return std::nullopt;
  return target;
}

std::optional<uint64_t> callee_reentry_target(const BasicBlock::CallEdge &edge,
                                              std::span<const KernelDescriptorInfo> kernels,
                                              rocjitsu::IsaFamily arch,
                                              const HelperEntryLinks *entry_links = nullptr) {
  if (edge.callee == nullptr || edge.continuation == nullptr ||
      !edge.callee->successors().empty() || !edge.callee->predecessors().empty())
    return std::nullopt;
  std::array<const Instruction *, 3> tail = {};
  for (const Instruction &inst : edge.callee->instructions()) {
    if (entry_links != nullptr) {
      const HelperEntryLinks::const_iterator entry = entry_links->find(inst.src_loc());
      if (entry != entry_links->end() &&
          (inst.src_loc() != edge.callee->start_offset() || entry->second != edge.return_sreg))
        return std::nullopt;
    }
    tail = {tail[1], tail[2], &inst};
  }
  const std::optional<uint64_t> target =
      adjusted_pc_entry_target(tail, edge.continuation->start_offset(), edge.return_sreg, kernels);
  if (!target)
    return std::nullopt;
  const uint32_t sgpr_count = AmdGpuCodeObject::min_kernel_sgpr_count(arch, kernels).value_or(0);
  const RegisterRef link = {RegClass::SGPR, edge.return_sreg, 2};
  if (uint32_t{link.index} + link.width > sgpr_count)
    return std::nullopt;
  const uint32_t ordinary_bound = ordinary_sgpr_bound(arch, sgpr_count);
  for (const Instruction &inst : edge.callee->instructions()) {
    if (&inst == tail[0])
      break;
    if (!preserves_helper_link(inst, link, ordinary_bound))
      return std::nullopt;
  }
  return target;
}

std::optional<uint64_t> static_pc_reentry_target(const BasicBlock &block,
                                                 std::span<const KernelDescriptorInfo> kernels) {
  std::array<const Instruction *, 4> tail = {};
  for (const Instruction &inst : block.instructions())
    tail = {tail[1], tail[2], tail[3], &inst};
  if (tail[0] == nullptr ||
      (tail[0]->mnemonic() != "s_getpc_b64" && tail[0]->mnemonic() != "s_get_pc_i64") ||
      tail[0]->src_loc() + tail[0]->size() != tail[1]->src_loc())
    return std::nullopt;
  const Operand *destination = tail[0]->dst_operand(0);
  if (destination == nullptr)
    return std::nullopt;
  const std::optional<RegisterRef> pair = destination->to_register_ref();
  if (!pair || pair->cls != RegClass::SGPR || pair->width != 2)
    return std::nullopt;
  return adjusted_pc_entry_target({tail[1], tail[2], tail[3]}, tail[0]->src_loc() + tail[0]->size(),
                                  pair->index, kernels);
}

// Hidden but bounded return edges preserve initial state if no instruction in
// the closed text changes it. Unknown/external transfers still fail.
bool callable_state_is_preserved_in_closed_text(std::span<const std::unique_ptr<BasicBlock>> blocks,
                                                std::span<const KernelDescriptorInfo> kernels,
                                                std::span<const uint8_t> text,
                                                rocjitsu::IsaFamily arch, CallableState state,
                                                const CalleeReturns &preserving_returns,
                                                const HelperEntryLinks &entry_links) {
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
      if (edge.callee == nullptr)
        return false;
      const CalleeReturns::const_iterator found = preserving_returns.find(edge.callee);
      if (found != preserving_returns.end()) {
        known_returns.insert(found->second.begin(), found->second.end());
      } else if (callee_reentry_target(edge, kernels, arch, &entry_links)) {
        known_returns.insert(edge.callee->terminator());
      } else {
        return false;
      }
    }
  }
  for (const std::unique_ptr<BasicBlock> &block : blocks) {
    const BasicBlock::SuccessorIssue issue = block->successor_issue();
    if (issue == BasicBlock::SuccessorIssue::None)
      continue;
    if (issue != BasicBlock::SuccessorIssue::IndirectControlFlow)
      return false;
    if (known_returns.contains(block->terminator()) || static_pc_reentry_target(*block, kernels))
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

// Relocation changes the value observed by PC readers. PC restore and set
// operations remain disallowed separately from ordinary branch metadata.
[[nodiscard]] bool is_pc_relative_anchor(const Instruction &inst) {
  if (inst.reads_pc())
    return true;
  switch (inst.amdgpu_operation()) {
  case amdgpu::Operation::S_SETPC_B64:
  case amdgpu::Operation::S_SET_PC_I64:
  case amdgpu::Operation::S_RFE_B64:
  case amdgpu::Operation::S_RFE_I64:
  case amdgpu::Operation::S_RFE_RESTORE_B64:
    return true;
  default:
    return false;
  }
}

[[nodiscard]] bool is_s_clause(const Instruction &inst) {
  return inst.amdgpu_operation() == amdgpu::Operation::S_CLAUSE;
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
uint32_t max_scratch_offset_bytes(rocjitsu::IsaFamily arch) {
  switch (arch) {
  case rocjitsu::IsaFamily::Cdna3:
  case rocjitsu::IsaFamily::Cdna4:
    return 0xFFF; // 12-bit, pad_12 = 0
  case rocjitsu::IsaFamily::Rdna4:
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
bool arch_has_accvgpr(rocjitsu::IsaFamily arch) {
  switch (arch) {
  case rocjitsu::IsaFamily::Cdna1:
  case rocjitsu::IsaFamily::Cdna2:
  case rocjitsu::IsaFamily::Cdna3:
  case rocjitsu::IsaFamily::Cdna4:
    return true;
  default:
    return false;
  }
}

// CDNA2-4 (gfx90a/gfx942/gfx950) carve the AccVGPR file out of a single unified VGPR
// allocation, split at the descriptor's ACCUM_OFFSET field. CDNA1/gfx908 allocates
// AGPRs separately and has no such field; RDNA and gfx1250 have no AccVGPR file at
// all.
bool arch_has_unified_vgpr_allocation(rocjitsu::IsaFamily arch) {
  switch (arch) {
  case rocjitsu::IsaFamily::Cdna2:
  case rocjitsu::IsaFamily::Cdna3:
  case rocjitsu::IsaFamily::Cdna4:
    return true;
  case rocjitsu::IsaFamily::Cdna1:
  case rocjitsu::IsaFamily::Rdna1:
  case rocjitsu::IsaFamily::Rdna2:
  case rocjitsu::IsaFamily::Rdna3:
  case rocjitsu::IsaFamily::Rdna3_5:
  case rocjitsu::IsaFamily::Rdna4:
  case rocjitsu::IsaFamily::Cdna5:
    return false;
  case rocjitsu::IsaFamily::Rv32i:
  case rocjitsu::IsaFamily::Rv64i:
  case rocjitsu::IsaFamily::Invalid:
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
KernelVgprBounds kernel_vgpr_bounds(rocjitsu::IsaFamily arch,
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
std::optional<uint8_t> common_wavefront_size(rocjitsu::IsaFamily arch,
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
bool check_probe_special_state(const ProbeClobberSummary &summary, rocjitsu::IsaFamily arch,
                               std::span<const uint32_t> allocations, std::string *error_out) {
  const bool writes_backing =
      arch == rocjitsu::IsaFamily::Cdna3 && std::ranges::any_of(allocations, [&](uint32_t count) {
        const auto base = cdna3::flat_scratch_base(cdna3::sgpr_alias_end(count));
        return summary.ordinary_clobbers.intersects(RegisterRef{RegClass::SGPR, base, 2});
      });
  if (summary.touches_flat_scratch || writes_backing) {
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
                           [[maybe_unused]] rocjitsu::IsaFamily arch, std::string *error_out) {
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
  if (is_pc_relative_anchor(anchor)) {
    report(error_out, "anchor instruction has PC-relative semantics");
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
                rocjitsu::IsaFamily arch, std::string *error_out, bool setreg_vgpr_msb_fixup) {
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
  if (pt.replacement) {
    fail("replacement payload requires ReplaceWithInst");
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
  if (pt.kind == InstrumentationKind::AfterInst && arch == rocjitsu::IsaFamily::Cdna5 &&
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
    if (pt.kind == InstrumentationKind::KernelExit &&
        anchor.amdgpu_operation() != amdgpu::Operation::S_ENDPGM) {
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

bool plan_vgpr_spills(const RegisterSet &spill_set, SpillManager &spills, rocjitsu::IsaFamily arch,
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
                      SpillManager &spills, rocjitsu::IsaFamily arch, std::vector<SpillSlot> &out,
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
                     rocjitsu::IsaFamily arch, std::vector<SpillSlot> &out,
                     std::string *error_out) {
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
TrampolinePlan make_base_plan(const ResolvedInstrumentationSite &site, rocjitsu::IsaFamily arch,
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
  if (site.kind == InstrumentationKind::ReplaceWithInst) {
    plan.emit_original = false;
    plan.returns_to_kernel = !site.replacement_terminates;
    if (!site.replacement_words.empty())
      plan.after_items.push_back({.words = site.replacement_words});
  }
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
                                                   rocjitsu::IsaFamily arch,
                                                   uint64_t trampoline_offset,
                                                   std::string *error_out) {
  OptionalStringDiagnosticSink record_error(error_out);
  const util::DiagnosticEmitter error(record_error);
  util::FailureOr<uint32_t> nop = build_s_nop(0, arch, &error);
  if (nop.failed())
    return std::nullopt;
  TrampolinePlan plan = make_base_plan(site, arch, trampoline_offset);
  if (site.kind == InstrumentationKind::ReplaceWithInst)
    return plan;
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

// A direct call initializes its even SGPR pair to one of this kernel's decoded
// continuations. A SETPC is bounded only if every path still carries such a link.
bool kernel_has_unknown_target(BasicBlock &entry, const KernelScopeSpec &spec,
                               const BlockOffsetIndex &offsets,
                               const std::unordered_set<BasicBlock *> &reachable,
                               uint32_t ordinary_sgpr_count,
                               const std::unordered_set<const BasicBlock *> &proven_reentries = {},
                               bool allow_implicit_exit = false) {
  if (std::ranges::none_of(reachable, [](const BasicBlock *block) {
        return block->successor_issue() != BasicBlock::SuccessorIssue::None;
      }))
    return false;

  std::unordered_map<BasicBlock *, uint64_t> incoming;
  std::vector<BasicBlock *> worklist;
  auto merge = [&](BasicBlock *block, uint64_t links) {
    if (!reachable.contains(block))
      return;
    auto [it, inserted] = incoming.emplace(block, links);
    if (!inserted) {
      links &= it->second;
      if (it->second == links)
        return;
      it->second = links;
    }
    worklist.push_back(block);
  };
  merge(&entry, 0);
  for (uint64_t offset : spec.own_entries)
    merge(block_for_offset(offsets, offset), 0);
  const auto pair_bit = [ordinary_sgpr_count](unsigned reg) -> uint64_t {
    // Firmware's aliased scalar tail can also be written by implicit operands.
    if ((reg & 1u) || reg >= 128 || reg + 1 >= ordinary_sgpr_count)
      return 0;
    return uint64_t{1} << (reg / 2);
  };
  while (!worklist.empty()) {
    BasicBlock *block = worklist.back();
    worklist.pop_back();
    uint64_t links = incoming.at(block);
    for (const Instruction &inst : block->instructions()) {
      if (!links)
        break;
      if (inst.mnemonic().starts_with("s_movrel")) {
        links = 0;
        break;
      }
      const InstDefUse effects(inst, 64);
      for (uint64_t pairs = links; pairs; pairs &= pairs - 1) {
        const uint16_t first = 2 * std::countr_zero(pairs);
        if (effects.defs.intersects(RegisterRef{RegClass::SGPR, first, 2}))
          links &= ~pair_bit(first);
      }
    }
    const BasicBlock::SuccessorIssue issue = block->successor_issue();
    const bool implicit_exit =
        allow_implicit_exit && issue == BasicBlock::SuccessorIssue::MissingFallthrough;
    if (issue != BasicBlock::SuccessorIssue::None && !implicit_exit &&
        !proven_reentries.contains(block)) {
      const Instruction *term = block->terminator();
      if (block->successor_issue() != BasicBlock::SuccessorIssue::IndirectControlFlow || !term ||
          term->mnemonic() != "s_setpc_b64" || !term->num_src_operands())
        return true;
      const auto src = term->src_operand(0)->to_register_ref();
      if (!src || src->cls != RegClass::SGPR || src->width != 2 || !(links & pair_bit(src->index)))
        return true;
    }

    const auto &calls = block->call_edges();
    for (BasicBlock *next : block->successors())
      merge(next, calls.empty() ? links : 0);
    for (const BasicBlock::CallEdge &call : calls) {
      // The callee may clobber any older saved link before it returns.
      merge(call.continuation, 0);
      const bool local = call.kind == BasicBlock::CallEdgeKind::DirectCall &&
                         reachable.contains(call.continuation);
      merge(call.callee, links | (local ? pair_bit(call.return_sreg) : 0));
    }
  }
  return false;
}

// Return provenance for the closed-text MODE proof is separate from register
// liveness. Summarized callees may destroy any register they actually define;
// lane saves survive a nested call only when its inspected callees leave them alone.
class ClosedCallFlow {
public:
  ClosedCallFlow(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                 const BlockOffsetIndex &offsets,
                 const std::unordered_set<const BasicBlock *> &proven_reentries,
                 uint32_t ordinary_sgprs)
      : offsets_(offsets), proven_reentries_(proven_reentries), ordinary_sgprs_(ordinary_sgprs) {
    for (const std::unique_ptr<BasicBlock> &block : blocks)
      decoded_.insert(block.get());
  }

  bool contains_flow(BasicBlock &entry, const KernelScopeSpec &spec) {
    if (!analyze(entry, std::nullopt))
      return false;
    for (uint64_t offset : spec.own_entries) {
      BasicBlock *extra = block_for_offset(offsets_, offset);
      if (extra == nullptr || extra->start_offset() != offset || !analyze(*extra, std::nullopt))
        return false;
    }
    return true;
  }

private:
  struct State {
    std::array<uint8_t, REGISTER_SET_MAX_SGPRS> scalar = {};
    std::unordered_map<uint32_t, uint8_t> lanes;
  };

  const BlockOffsetIndex &offsets_;
  const std::unordered_set<const BasicBlock *> &proven_reentries_;
  uint32_t ordinary_sgprs_;
  unsigned active_summaries_ = 0;
  std::unordered_set<const BasicBlock *> decoded_;
  std::unordered_map<const BasicBlock *, std::unordered_map<uint16_t, std::optional<RegisterSet>>>
      summaries_;

  static std::optional<RegisterRef> reg(const Operand *operand, RegClass cls) {
    const std::optional<RegisterRef> ref =
        operand == nullptr ? std::nullopt : operand->to_register_ref();
    return ref && ref->cls == cls ? ref : std::nullopt;
  }

  static std::optional<uint16_t> lane(const Operand *operand) {
    const std::optional<uint64_t> value =
        operand == nullptr ? std::nullopt : operand->const_value();
    return value && *value < 64 ? std::optional<uint16_t>(static_cast<uint16_t>(*value))
                                : std::nullopt;
  }

  static uint32_t slot(RegisterRef vector, uint16_t lane_index) {
    return uint32_t{vector.index} * 64 + lane_index;
  }

  static void kill(State &state, const RegisterSet &defs,
                   std::optional<uint16_t> preserved_vector = std::nullopt) {
    for (uint16_t index = 0; index < state.scalar.size(); ++index)
      if (state.scalar[index] && defs.contains({RegClass::SGPR, index, 1}))
        state.scalar[index] = 0;
    std::erase_if(state.lanes, [&](const std::pair<const uint32_t, uint8_t> &item) {
      const uint16_t vector = item.first / 64;
      return vector != preserved_vector && defs.contains({RegClass::VGPR, vector, 1});
    });
  }

  bool complete_calls(const BasicBlock &block) const {
    const Instruction *term = block.terminator();
    if (term == nullptr || block.call_edges().empty())
      return false;
    for (const BasicBlock::CallEdge &edge : block.call_edges()) {
      if (edge.callee == nullptr || edge.continuation == nullptr ||
          !decoded_.contains(edge.callee) || !decoded_.contains(edge.continuation) ||
          edge.source_call_offset != term->src_loc() || edge.return_sreg % 2 ||
          edge.return_sreg + 1u >= ordinary_sgprs_)
        return false;
      if (edge.kind == BasicBlock::CallEdgeKind::DirectCall) {
        const std::optional<int64_t> delta = term->branch_offset_bytes();
        if (!delta || static_cast<int64_t>(block.end_offset()) + *delta < 0 ||
            static_cast<uint64_t>(static_cast<int64_t>(block.end_offset()) + *delta) !=
                edge.callee->start_offset())
          return false;
      } else {
        const std::vector<IndirectCallFixup> &fixups = block.static_indirect_call_fixups();
        if (fixups.empty() ||
            !std::ranges::any_of(fixups,
                                 [&](const IndirectCallFixup &fixup) {
                                   return fixup.source_call_offset == term->src_loc() &&
                                          fixup.source_target_offset ==
                                              edge.callee->start_offset() &&
                                          fixup.source_return_sreg == edge.return_sreg;
                                 }) ||
            !std::ranges::all_of(fixups, [&](const IndirectCallFixup &fixup) {
              return !fixup.source_incomplete && fixup.source_is_call &&
                     fixup.source_call_offset == term->src_loc() &&
                     std::ranges::any_of(
                         block.call_edges(), [&](const BasicBlock::CallEdge &target) {
                           return target.callee != nullptr &&
                                  target.callee->start_offset() == fixup.source_target_offset &&
                                  target.return_sreg == fixup.source_return_sreg;
                         });
            }))
          return false;
      }
    }
    return true;
  }

  std::optional<RegisterSet> callee(const BasicBlock::CallEdge &edge) {
    std::unordered_map<uint16_t, std::optional<RegisterSet>> &by_link = summaries_[edge.callee];
    const std::unordered_map<uint16_t, std::optional<RegisterSet>>::const_iterator found =
        by_link.find(edge.return_sreg);
    if (found != by_link.end())
      return found->second;
    // Bound recursion as well as cycles; an incomplete summary never grants admission.
    by_link.emplace(edge.return_sreg, std::nullopt);
    if (active_summaries_ == 64)
      return std::nullopt;
    ++active_summaries_;
    const std::optional<RegisterSet> result = analyze(*edge.callee, edge.return_sreg);
    --active_summaries_;
    summaries_[edge.callee][edge.return_sreg] = result;
    return result;
  }

  std::optional<RegisterSet> analyze(BasicBlock &entry, std::optional<uint16_t> link) {
    State initial;
    if (link) {
      initial.scalar[*link] = 1;
      initial.scalar[*link + 1] = 2;
    }
    std::unordered_map<const BasicBlock *, State> incoming;
    std::vector<const BasicBlock *> work;
    auto merge = [&](const BasicBlock *block, const State &state) {
      if (block == nullptr || !decoded_.contains(block))
        return false;
      auto [position, inserted] = incoming.emplace(block, state);
      bool changed = inserted;
      if (!inserted) {
        State &old = position->second;
        for (size_t index = 0; index < old.scalar.size(); ++index) {
          if (old.scalar[index] && old.scalar[index] != state.scalar[index]) {
            old.scalar[index] = 0;
            changed = true;
          }
        }
        std::erase_if(old.lanes, [&](const std::pair<const uint32_t, uint8_t> &item) {
          const std::unordered_map<uint32_t, uint8_t>::const_iterator other =
              state.lanes.find(item.first);
          const bool remove = other == state.lanes.end() || other->second != item.second;
          changed |= remove;
          return remove;
        });
      }
      if (changed)
        work.push_back(block);
      return true;
    };
    merge(&entry, initial);
    RegisterSet clobbers;
    while (!work.empty()) {
      const BasicBlock *block = work.back();
      work.pop_back();
      State state = incoming.at(block);
      for (const Instruction &inst : block->instructions()) {
        const std::string_view name = inst.mnemonic();
        if (is_control_flow_transfer(inst) && &inst != block->terminator())
          return std::nullopt;
        // Dynamic register addressing is not represented by these fixed slots.
        if (name.starts_with("s_movrel") || name.starts_with("s_set_gpr_idx") ||
            name.starts_with("s_setreg"))
          return std::nullopt;
        const InstDefUse effects(inst, 64);
        RegisterSet defs = effects.defs;
        if (effects.conditional_vgpr_def)
          defs.expand(*effects.conditional_vgpr_def);
        clobbers |= defs;
        const std::optional<RegisterRef> scalar_dst = reg(inst.dst_operand(0), RegClass::SGPR);
        // WRITELANE's read-modify-write destination is its first decoded source.
        const bool writes_lane = name == "v_writelane_b32";
        const std::optional<RegisterRef> scalar_src =
            reg(inst.src_operand(writes_lane ? 1 : 0), RegClass::SGPR);
        const std::optional<RegisterRef> vector_dst = reg(inst.dst_operand(0), RegClass::VGPR);
        const std::optional<RegisterRef> vector_src = reg(inst.src_operand(0), RegClass::VGPR);
        const std::optional<uint16_t> lane_index = lane(inst.src_operand(writes_lane ? 2 : 1));
        std::array<uint8_t, 2> copied = {};
        unsigned copied_width = 0;
        std::optional<uint32_t> written_lane;
        uint8_t lane_value = 0;
        if ((name == "s_mov_b32" || name == "s_mov_b64") && scalar_dst && scalar_src &&
            scalar_dst->width == scalar_src->width && scalar_dst->width <= copied.size() &&
            scalar_dst->index + scalar_dst->width <= ordinary_sgprs_ &&
            scalar_src->index + scalar_src->width <= ordinary_sgprs_) {
          copied_width = scalar_dst->width;
          for (unsigned half = 0; half < copied_width; ++half)
            copied[half] = state.scalar[scalar_src->index + half];
        } else if (name == "v_writelane_b32" && vector_dst && vector_dst->width == 1 &&
                   lane_index) {
          written_lane = slot(*vector_dst, *lane_index);
          if (scalar_src && scalar_src->width == 1 && scalar_src->index < ordinary_sgprs_)
            lane_value = state.scalar[scalar_src->index];
        } else if (name == "v_readlane_b32" && scalar_dst && scalar_dst->width == 1 &&
                   scalar_dst->index < ordinary_sgprs_ && vector_src && vector_src->width == 1 &&
                   lane_index) {
          copied_width = 1;
          const std::unordered_map<uint32_t, uint8_t>::const_iterator saved =
              state.lanes.find(slot(*vector_src, *lane_index));
          copied[0] = saved == state.lanes.end() ? 0 : saved->second;
        }
        kill(state, defs, written_lane ? std::optional<uint16_t>(vector_dst->index) : std::nullopt);
        if (written_lane) {
          state.lanes.erase(*written_lane);
          if (lane_value)
            state.lanes.emplace(*written_lane, lane_value);
        }
        for (unsigned half = 0; half < copied_width; ++half)
          state.scalar[scalar_dst->index + half] = copied[half];
      }
      const Instruction *term = block->terminator();
      if (term == nullptr)
        return std::nullopt;
      if (term->mnemonic() == "s_setpc_b64") {
        const std::optional<RegisterRef> source = reg(term->src_operand(0), RegClass::SGPR);
        if (link && source && source->width == 2 && source->index + 1u < ordinary_sgprs_ &&
            state.scalar[source->index] == 1 && state.scalar[source->index + 1] == 2)
          continue;
      }
      const bool has_calls = !block->call_edges().empty();
      if (is_control_flow_transfer(*term) && !is_program_path_terminator(*term) &&
          !term->branch_offset_bytes() && term->mnemonic() != "s_setpc_b64" && !has_calls &&
          !proven_reentries_.contains(block))
        return std::nullopt;
      if (has_calls) {
        if (!complete_calls(*block))
          return std::nullopt;
        for (const BasicBlock::CallEdge &edge : block->call_edges()) {
          const std::optional<RegisterSet> summary = callee(edge);
          if (!summary)
            return std::nullopt;
          clobbers |= *summary;
          kill(state, *summary);
        }
      }
      const BasicBlock::SuccessorIssue issue = block->successor_issue();
      if (issue != BasicBlock::SuccessorIssue::None &&
          issue != BasicBlock::SuccessorIssue::MissingFallthrough &&
          !(issue == BasicBlock::SuccessorIssue::IndirectControlFlow && has_calls) &&
          !proven_reentries_.contains(block))
        return std::nullopt;
      for (const BasicBlock *next : block->successors())
        if (!merge(next, state))
          return std::nullopt;
    }
    return clobbers;
  }
};

// Bounded call/return edges preserve a MODE bit only when its own writer scan
// proves it stays clear. This establishes neither EXEC nor register liveness.
bool mode_state_is_preserved_in_closed_text(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                                            std::span<const KernelDescriptorInfo> kernels,
                                            std::span<const uint8_t> text, rocjitsu::IsaFamily arch,
                                            CallableState state, uint64_t entry_text_size,
                                            const HelperEntryLinks &entry_links) {
  if (!arch_is_cdna_4_or_lower(arch) ||
      (state != CallableState::GprIndexing && state != CallableState::Vskip))
    return false;
  std::unordered_set<const BasicBlock *> proven_reentries;
  std::unordered_map<const BasicBlock *, bool> bounded_callees;
  for (const std::unique_ptr<BasicBlock> &block : blocks) {
    for (const Instruction &inst : block->instructions()) {
      const bool enables = state == CallableState::GprIndexing
                               ? may_enable_gpr_indexing(inst, text, arch)
                               : may_enable_vskip(inst, text, arch);
      // Return-link tags in VGPR lanes require their save/restore instructions
      // to issue, even when this invocation proves GPR indexing rather than VSKIP.
      const bool skips_lane_saves =
          state == CallableState::GprIndexing && may_enable_vskip(inst, text, arch);
      if (enables || skips_lane_saves || inst.mnemonic().starts_with("s_rfe"))
        return false;
      if ((inst.flags() & INDIRECT_CALL) &&
          (&inst != block->terminator() || block->call_edges().empty()))
        return false;
    }
    if (static_pc_reentry_target(*block, kernels))
      proven_reentries.insert(block.get());
    for (const BasicBlock::CallEdge &edge : block->call_edges()) {
      if (edge.callee == nullptr)
        continue;
      bool &bounded = bounded_callees.try_emplace(edge.callee, true).first->second;
      bounded = bounded && callee_reentry_target(edge, kernels, arch, &entry_links).has_value();
    }
  }
  // A computed return can depend on the caller's continuation. Every call
  // into the block must establish a bounded destination.
  for (const auto &[callee, bounded] : bounded_callees)
    if (bounded)
      proven_reentries.insert(callee);
  const BlockOffsetIndex offsets = build_block_offset_index(blocks);
  const BlockPositionIndex positions = build_block_position_index(blocks);
  for (const KernelDescriptorInfo &kernel : kernels) {
    BasicBlock *entry = block_for_offset(offsets, kernel.entry_text_offset);
    const std::optional<uint32_t> count =
        AmdGpuCodeObject::min_kernel_sgpr_count(arch, std::span(&kernel, 1));
    if (entry == nullptr || !count)
      return false;
    KernelScopeSpec spec;
    if (has_kernarg_preload_firmware_entry(kernel, arch)) {
      const std::optional<uint64_t> firmware =
          kernarg_preload_firmware_entry(kernel, arch, entry_text_size);
      BasicBlock *extra = firmware ? block_for_offset(offsets, *firmware) : nullptr;
      if (extra == nullptr || extra->start_offset() != *firmware)
        return false;
      spec.own_entries.insert(*firmware);
    }
    const std::vector<BasicBlock *> reached =
        reachable_kernel_blocks(blocks, offsets, positions, *entry, spec);
    const std::unordered_set<BasicBlock *> reachable(reached.begin(), reached.end());
    const uint32_t ordinary = arch == rocjitsu::IsaFamily::Cdna3
                                  ? cdna3::flat_scratch_base(cdna3::sgpr_alias_end(*count))
                                  : ordinary_sgpr_bound(arch, *count);
    if (kernel_has_unknown_target(*entry, spec, offsets, reachable, ordinary, proven_reentries,
                                  true)) {
      ClosedCallFlow calls(blocks, offsets, proven_reentries, ordinary);
      if (!calls.contains_flow(*entry, spec))
        return false;
    }
  }
  return !kernels.empty();
}

// Shared code must satisfy every kernel that can execute a probe. Keep an
// unresolved caller among the possible owners of every site.
std::unordered_map<uint64_t, std::vector<uint32_t>>
cdna3_probe_sgpr_counts(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                        std::span<const KernelDescriptorInfo> kernels,
                        std::span<const ResolvedInstrumentationSite> sites,
                        uint64_t source_text_size) {
  std::unordered_map<uint64_t, std::vector<uint32_t>> owners;
  const auto offsets = build_block_offset_index(blocks);
  const auto positions = build_block_position_index(blocks);
  std::vector<uint32_t> all_counts;
  for (const auto &kernel : kernels) {
    const uint32_t count =
        *AmdGpuCodeObject::min_kernel_sgpr_count(rocjitsu::IsaFamily::Cdna3, std::span(&kernel, 1));
    all_counts.push_back(count);
    auto *entry = block_for_offset(offsets, kernel.entry_text_offset);
    KernelScopeSpec spec;
    bool missing_firmware_entry = false;
    if (has_kernarg_preload_firmware_entry(kernel, rocjitsu::IsaFamily::Cdna3)) {
      const std::optional<uint64_t> firmware =
          kernarg_preload_firmware_entry(kernel, rocjitsu::IsaFamily::Cdna3, source_text_size);
      BasicBlock *extra = firmware ? block_for_offset(offsets, *firmware) : nullptr;
      missing_firmware_entry = extra == nullptr || extra->start_offset() != *firmware;
      if (!missing_firmware_entry)
        spec.own_entries.insert(*firmware);
    }
    std::vector<BasicBlock *> reached;
    if (entry)
      reached = reachable_kernel_blocks(blocks, offsets, positions, *entry, spec);
    const std::unordered_set<BasicBlock *> reachable(reached.begin(), reached.end());
    const bool incomplete =
        !entry || missing_firmware_entry ||
        kernel_has_unknown_target(*entry, spec, offsets, reachable,
                                  cdna3::flat_scratch_base(cdna3::sgpr_alias_end(count)));
    for (const auto &site : sites) {
      if (site.is_probe_call() &&
          (incomplete || reachable.contains(block_for_offset(offsets, site.anchor_offset))))
        owners[site.anchor_offset].push_back(count);
    }
  }
  for (const auto &site : sites)
    if (site.is_probe_call() && !owners.contains(site.anchor_offset))
      owners.emplace(site.anchor_offset, all_counts);
  return owners;
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
  const std::optional<uint32_t> original_sgpr_count =
      AmdGpuCodeObject::min_kernel_sgpr_count(arch_, kernels);
  if (!kernel_sgpr_count || !original_sgpr_count) {
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

  // Reserve scalar sources before the wrapper prologue chooses its scratch:
  // a requested entry input can otherwise be dead to the original guest CFG.
  // CP inputs remain valid even when metadata counts only used registers.
  // The planning allocation may already be enlarged, but entry sources must
  // remain within the original descriptor's ordinary register ranges.
  const uint32_t original_scalar_bound = ordinary_sgpr_bound(arch_, *original_sgpr_count);
  const uint32_t scalar_bound =
      options_.register_growth
          ? std::max(std::min(options_.register_growth->sgprs, original_scalar_bound),
                     kernel_descriptor_initial_sgpr_count(arch_, kernel.descriptor))
          : original_scalar_bound;
  for (const InstrumentationPoint &point : points_)
    for (const ProbeArgValue &arg : point.probe_args)
      if (arg.source == ProbeArgSource::EntrySgpr) {
        if (arg.register_index >= scalar_bound) {
          report(error_out, "entry SGPR capture exceeds the original guest allocation");
          return std::nullopt;
        }
        reserved.expand(RegisterRef{RegClass::SGPR, arg.register_index, 1});
      }

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
    // The prologue has restored the guest kernarg pointer. Capture EntrySgpr
    // sources afterward so a kernarg-pointer word cannot expose the wrapper.
    // The prologue's temporary pair is now available for persistent snapshots.
    const uint32_t original_vector_bound =
        kernel_vgpr_bounds(arch_, kernel.descriptor).ordinary_bound;
    const uint32_t vector_bound =
        options_.register_growth ? std::min(options_.register_growth->vgprs, original_vector_bound)
                                 : original_vector_bound;
    std::vector<std::pair<RegClass, uint16_t>> sources;
    for (const InstrumentationPoint &point : points_)
      for (const ProbeArgValue &arg : point.probe_args) {
        const std::optional<RegClass> source_class = entry_register_class(arg.source);
        if (!source_class)
          continue;
        if (arg.source == ProbeArgSource::EntryTtmp &&
            arg.register_index >= REGISTER_SET_MAX_TTMPS) {
          report(error_out, "entry TTMP capture exceeds the trap-temporary register file");
          return std::nullopt;
        }
        if (*source_class == RegClass::TTMP && !supports_entry_ttmp_capture(arch_)) {
          report(error_out, "entry TTMP capture requires unprivileged TTMP reads, which this "
                            "target does not support");
          return std::nullopt;
        }
        const std::pair<RegClass, uint16_t> source = {*source_class, arg.register_index};
        if (std::ranges::find(sources, source) == sources.end())
          sources.push_back(source);
      }
    if (sources.size() > 2) {
      report(error_out, "the entry snapshot has room for at most two register values");
      return std::nullopt;
    }
    for (size_t index = 0; index < sources.size(); ++index) {
      const auto [source_class, source_index] = sources[index];
      if (source_class == RegClass::VGPR && source_index >= std::min(vector_bound, 256u)) {
        report(error_out, "entry VGPR capture exceeds the kernel's ordinary VGPR allocation");
        return std::nullopt;
      }
      const uint16_t destination = planned->storage.entry_temp_base + index;
      if (source_class == RegClass::SGPR || source_class == RegClass::TTMP) {
        OptionalStringDiagnosticSink record_error(error_out);
        const util::DiagnosticEmitter error(record_error);
        const util::FailureOr<uint32_t> word = build_s_mov_b32(
            destination,
            source_class == RegClass::TTMP ? amdgpu::kTtmpSelectorFirst + source_index
                                           : source_index,
            arch_, &error);
        if (word.failed())
          return std::nullopt;
        patch.words.push_back(word.value());
      } else {
        util::StringDiagnostic lane_error;
        const util::DiagnosticEmitter emit_lane_error = lane_error.emitter();
        const util::FailureOr<std::array<uint32_t, 2>> maybe_words =
            build_v_readlane_b32(destination, source_index, 0, arch_, &emit_lane_error);
        if (maybe_words.failed()) {
          report(error_out, lane_error.message().c_str());
          return std::nullopt;
        }
        const std::array<uint32_t, 2> &words = maybe_words.value();
        patch.words.insert(patch.words.end(), words.begin(), words.end());
      }
      patch.captured_registers.push_back(
          {.source_index = source_index, .sgpr = destination, .source_class = source_class});
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
  CallableStateAnalysis(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                        std::span<const KernelDescriptorInfo> kernels,
                        std::span<const uint8_t> text, rocjitsu::IsaFamily arch,
                        uint64_t source_text_size) {
    const uint64_t entry_text_size = std::min<uint64_t>(text.size(), source_text_size);
    std::unordered_set<uint64_t> entries;
    // A return is bounded only if every entry into its helper supplies the
    // same link. Hardware dispatch and conflicting calls supply no such proof.
    HelperEntryLinks entry_links;
    std::unordered_set<uint64_t> dynamic_entries;
    for (const KernelDescriptorInfo &kernel : kernels) {
      entries.insert(kernel.entry_text_offset);
      entry_links.emplace(kernel.entry_text_offset, std::nullopt);
      if (uses_dynamic_vgprs(arch, kernel.descriptor))
        dynamic_entries.insert(kernel.entry_text_offset);
      if (!has_kernarg_preload_firmware_entry(kernel, arch))
        continue;
      const std::optional<uint64_t> firmware =
          kernarg_preload_firmware_entry(kernel, arch, entry_text_size);
      // An appended replacement cave cannot supply a missing source entry.
      if (!firmware || !std::ranges::any_of(blocks, [&](const std::unique_ptr<BasicBlock> &block) {
            return block->start_offset() == *firmware;
          })) {
        missing_firmware_entry_ = true;
        continue;
      }
      entries.insert(*firmware);
      entry_links.emplace(*firmware, std::nullopt);
    }
    std::vector<const BasicBlock *> dynamic_roots;
    std::vector<const BasicBlock *> static_roots;
    for (const std::unique_ptr<BasicBlock> &block : blocks) {
      if (dynamic_entries.contains(block->start_offset()))
        dynamic_roots.push_back(block.get());
      else if (entries.contains(block->start_offset()))
        static_roots.push_back(block.get());
      for (const BasicBlock::CallEdge &edge : block->call_edges()) {
        if (edge.callee == nullptr)
          continue;
        const auto [entry, inserted] =
            entry_links.try_emplace(edge.callee->start_offset(), edge.return_sreg);
        if (!inserted && entry->second != edge.return_sreg)
          entry->second.reset();
      }
      for (const Instruction &inst : block->instructions()) {
        states_.try_emplace(&inst);
        has_mfma_producer_ |= inst.is_mfma();
        has_cdna5_producer_ |=
            arch == rocjitsu::IsaFamily::Cdna5 && may_require_cdna5_valu_spacing(inst);
        has_cdna5_packed_permute_ |=
            arch == rocjitsu::IsaFamily::Cdna5 && requires_cdna5_valu_follower(inst);
      }
    }
    std::unordered_set<const BasicBlock *> reachable;
    const auto visit = [&](std::vector<const BasicBlock *> &work) {
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
    };
    // A dynamic wave's initial allocation can be smaller than the descriptor
    // count. Growing that count does not allocate the envelope's save VGPRs.
    // Walk dynamic roots first so a shared block inherits their restriction.
    const uint8_t storage = state_bit(CallableState::UnavailableStorage);
    const PointState dynamic_storage = {.before = storage, .after = storage};
    bool unknown_dynamic_target = dynamic_roots.size() != dynamic_entries.size();
    visit(dynamic_roots);
    for (const BasicBlock *block : reachable) {
      unknown_dynamic_target |= block->successor_issue() != BasicBlock::SuccessorIssue::None;
      for (const Instruction &inst : block->instructions())
        states_.at(&inst) = dynamic_storage;
    }
    if (unknown_dynamic_target)
      for (auto &[inst, point] : states_)
        point = dynamic_storage;
    visit(static_roots);
    for (const CallableState state :
         {CallableState::GprIndexing, CallableState::Vskip, CallableState::UnavailableStorage}) {
      if (has_callable_state(arch, state))
        analyze_state(blocks, kernels, text, arch, state, entries, reachable, entry_links,
                      entry_text_size);
    }
  }

  [[nodiscard]] bool has_mfma_producer() const { return has_mfma_producer_; }
  [[nodiscard]] bool has_cdna5_producer() const { return has_cdna5_producer_; }
  [[nodiscard]] bool has_cdna5_packed_permute() const { return has_cdna5_packed_permute_; }

  [[nodiscard]] bool may_be_unsafe(const Instruction &anchor, bool after,
                                   CallableState state) const {
    const std::unordered_map<const Instruction *, PointState>::const_iterator found =
        states_.find(&anchor);
    return missing_firmware_entry_ || found == states_.end() ||
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
  bool missing_firmware_entry_ = false;
  bool has_mfma_producer_ = false;
  bool has_cdna5_producer_ = false;
  bool has_cdna5_packed_permute_ = false;

  static uint8_t state_bit(CallableState state) { return 1u << static_cast<unsigned>(state); }

  void analyze_state(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                     std::span<const KernelDescriptorInfo> kernels, std::span<const uint8_t> text,
                     rocjitsu::IsaFamily arch, CallableState state,
                     const std::unordered_set<uint64_t> &entries,
                     const std::unordered_set<const BasicBlock *> &reachable,
                     const HelperEntryLinks &entry_links, uint64_t entry_text_size) {
    CalleeReturns preserving_returns;
    const uint32_t sgpr_count = AmdGpuCodeObject::min_kernel_sgpr_count(arch, kernels).value_or(0);
    const uint32_t ordinary_bound = ordinary_sgpr_bound(arch, sgpr_count);
    for (const std::unique_ptr<BasicBlock> &block : blocks) {
      const HelperEntryLinks::const_iterator entry = entry_links.find(block->start_offset());
      if (entry == entry_links.end() || !entry->second || uint32_t{*entry->second} + 2 > sgpr_count)
        continue;
      std::vector<const Instruction *> returns = preserving_callee_returns(
          *block, *entry->second, ordinary_bound, entry_links, text, arch, state);
      if (!returns.empty())
        preserving_returns.emplace(block.get(), std::move(returns));
    }
    std::unordered_set<const Instruction *> preserving_calls;
    std::unordered_set<const Instruction *> known_returns;
    for (const std::unique_ptr<BasicBlock> &block : blocks) {
      if (block->successor_issue() != BasicBlock::SuccessorIssue::None ||
          block->call_edges().empty())
        continue;
      if (std::ranges::all_of(block->call_edges(), [&](const BasicBlock::CallEdge &edge) {
            return preserving_returns.contains(edge.callee);
          })) {
        preserving_calls.insert(block->terminator());
        if (reachable.contains(block.get()))
          for (const BasicBlock::CallEdge &edge : block->call_edges()) {
            const std::vector<const Instruction *> &returns = preserving_returns.at(edge.callee);
            known_returns.insert(returns.begin(), returns.end());
          }
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
    const bool closed_state =
        incomplete && (callable_state_is_preserved_in_closed_text(
                           blocks, kernels, text, arch, state, preserving_returns, entry_links) ||
                       mode_state_is_preserved_in_closed_text(blocks, kernels, text, arch, state,
                                                              entry_text_size, entry_links));

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

struct Instrumentor::EntryPatchPlan {
  KernelDescriptorInfo kernel;
  ResolvedPoints resolved;
  std::vector<AppliedSite> applied;
  std::optional<EntryProloguePatch> prologue;
  std::optional<uint32_t> private_bytes;
  InstrumentationResources resources;
  RegisterSet guest_registers;
  uint64_t cave_end = 0;
};

Instrumentor::Instrumentor(const AmdGpuCodeObject &obj, rocjitsu::IsaFamily arch,
                           InstrumentationOptions options)
    : obj_(obj), arch_(arch), options_(std::move(options)) {}

Instrumentor::~Instrumentor() = default;

void Instrumentor::add_point(InstrumentationPoint pt) {
  points_.push_back(std::move(pt));
  // Probe arguments contribute reads to liveness; a changed queue needs new facts.
  liveness_.reset();
}

void Instrumentor::add_point_by_offset(uint64_t anchor_offset, InstrumentationKind kind) {
  InstrumentationPoint pt;
  pt.anchor_offset = anchor_offset;
  pt.kind = kind;
  add_point(std::move(pt));
}

bool Instrumentor::ensure_source_facts(TextEditError &error) {
  error = {};
  if (source_facts_)
    return true;
  if (CheckedTextEdits::validate_source_extent(obj_, error).failed())
    return false;
  std::string decoder_error;
  if (!ensure_decoder(&decoder_error)) {
    error = {.code = TextEditErrorCode::UnsupportedTarget,
             .offset = 0,
             .message = std::move(decoder_error)};
    return false;
  }
  auto facts = std::make_shared<SourceTextFacts>();
  const Section &text = *obj_.text_sections().front();
  const TextAlignmentGaps alignment_gaps(obj_, arch_);
  Elf64_Ehdr header = {};
  std::memcpy(&header, obj_.image_data(), sizeof(header));
  const auto record_instruction = [&](const Instruction &instruction, const Section &section,
                                      InstructionBoundaries &boundaries, bool in_text) {
    const size_t reference_count = boundaries.reference_addresses().size();
    boundaries.record(instruction, section.vaddr());
    const SourceTextFacts::SourceLocation source = {
        .address = section.vaddr() + instruction.src_loc(),
        .text_offset = in_text ? std::optional<uint64_t>(instruction.src_loc()) : std::nullopt};
    // Reuse the boundary index's checked branch math; source admission below
    // rejects its overflow flag before these facts can be used.
    const std::span<const uint64_t> references = boundaries.reference_addresses();
    if (references.size() != reference_count)
      facts->flow_edges.push_back({.source = source, .destination = references.back()});
    const bool unconditional_branch =
        (instruction.flags() & BRANCH) != 0 && (instruction.flags() & COND_BRANCH) == 0;
    const bool can_fall_through = !is_program_path_terminator(instruction) &&
                                  !unconditional_branch &&
                                  ((instruction.flags() & INDIRECT_BRANCH) == 0 ||
                                   indirect_branch_falls_through(instruction));
    const uint64_t size = instruction.size();
    if (can_fall_through && size <= section.size() &&
        instruction.src_loc() == section.size() - size)
      facts->flow_edges.push_back(
          {.source = source, .destination = section.vaddr() + section.size(), .fallthrough = true});
    if (instruction.mnemonic() == "s_getpc_b64" || instruction.mnemonic() == "s_get_pc_i64")
      facts->pc_reads.push_back(source);
  };
  const auto decode_section = [&](const Section &section, InstructionBoundaries &boundaries,
                                  bool in_text) {
    const auto fail = [&](TextEditErrorCode code, uint64_t offset, std::string message) {
      if (!in_text)
        message = "executable section " + section.name() + ": " + message;
      error = {.code = code, .offset = in_text ? offset : 0, .message = std::move(message)};
      return false;
    };
    if (section.data() == nullptr || section.size() % sizeof(uint32_t) != 0 ||
        section.vaddr() > std::numeric_limits<uint64_t>::max() - section.size())
      return fail(TextEditErrorCode::InvalidImage, 0, "invalid executable extent");
    const std::span<const uint32_t> words(reinterpret_cast<const uint32_t *>(section.data()),
                                          section.size() / sizeof(uint32_t));
    size_t cursor = 0;
    while (cursor < words.size()) {
      if (is_text_padding_word(arch_, words[cursor])) {
        ++cursor;
        continue;
      }
      util::StringDiagnostic diagnostic;
      DecodeResult decoded =
          decoder_->decode_window(words.subspan(cursor), cursor * 4, diagnostic.emitter());
      if (decoded.failed() || decoded.value()->size() == 0 || decoded.value()->size() % 4 != 0 ||
          static_cast<size_t>(decoded.value()->size()) / 4 > words.size() - cursor)
        return fail(TextEditErrorCode::InvalidInstruction, cursor * 4,
                    "invalid source instruction: " + diagnostic.message());
      if (reject_unavailable_branch_reference(*decoded.value(), error))
        return fail(error.code, error.offset, std::move(error.message));
      record_instruction(*decoded.value(), section, boundaries, in_text);
      cursor += static_cast<size_t>(decoded.value()->size()) / sizeof(uint32_t);
    }
    if (boundaries.reference_overflow())
      return fail(TextEditErrorCode::UnsupportedReferences, 0, "branch reference overflows");
    return true;
  };
  const std::span<const uint32_t> text_words(reinterpret_cast<const uint32_t *>(text.data()),
                                             text.size() / sizeof(uint32_t));
  for (size_t cursor = 0; cursor < text_words.size(); ++cursor) {
    const uint64_t byte_offset = cursor * sizeof(uint32_t);
    if (text_words[cursor] != 0 || !alignment_gaps.contains(byte_offset))
      continue;
    // Qualify alignment with the original CFG, including calls that may return
    // into it. Sources without such gaps retain the linear replacement path.
    std::string cfg_error;
    if (!ensure_blocks_built(&cfg_error)) {
      error = {.code = TextEditErrorCode::InvalidInstruction,
               .offset = byte_offset,
               .message = std::move(cfg_error)};
      return false;
    }
    break;
  }
  if (blocks_built_ && !analysis_object_) {
    // Own the index rather than borrowing storage from the parent Instrumentor.
    for (const std::unique_ptr<BasicBlock> &block : blocks_) {
      for (const Instruction &instruction : block->instructions()) {
        if (reject_unavailable_branch_reference(instruction, error))
          return false;
        record_instruction(instruction, text, facts->boundaries, true);
      }
    }
  } else if (!decode_section(text, facts->boundaries, true)) {
    return false;
  }
  // Replacement bypasses observation resolution. Collect descriptor entries
  // explicitly, without imposing probe, owner, liveness or loader policy.
  const std::span<const uint8_t> image(reinterpret_cast<const uint8_t *>(obj_.image_data()),
                                       obj_.image_size());
  const std::vector<KernelDescriptorInfo> kernels =
      scan_kernel_descriptors(image, text.sectionOffset(), text.size(), text.sectionHeaderIndex());
  std::vector<uint64_t> kernel_entries;
  kernel_entries.reserve(kernels.size());
  for (const KernelDescriptorInfo &kernel : kernels)
    kernel_entries.push_back(kernel.entry_text_offset);
  const CodeObjectPatcher patcher(obj_);
  FailureOr<TextEditReferences> references = CheckedTextEdits::collect_references(
      obj_, arch_, facts->boundaries, patcher, kernel_entries, error);
  if (references.failed())
    return false;
  facts->references = std::move(references).value();
  // Foreign offsets must not enter the public .text index or be mistaken for
  // replacement anchors. Their references still participate in the shared proof.
  for (const Section *section : obj_.allocated_executable_sections()) {
    // Relocatable sections have independent section-relative coordinates.
    if (header.e_type == ET_REL || section == &text || section->size() == 0)
      continue;
    InstructionBoundaries boundaries;
    if (!decode_section(*section, boundaries, false))
      return false;
    const std::vector<KernelDescriptorInfo> foreign_kernels = scan_kernel_descriptors(
        image, section->sectionOffset(), section->size(), section->sectionHeaderIndex());
    for (const KernelDescriptorInfo &kernel : foreign_kernels)
      facts->foreign_kernel_entries.push_back(section->vaddr() + kernel.entry_text_offset);
  }
  for (const SourceTextFacts::ControlFlowEdge &edge : facts->flow_edges) {
    if (edge.destination >= text.vaddr() &&
        alignment_gaps.contains(edge.destination - text.vaddr())) {
      error = {.code = TextEditErrorCode::UnsupportedReferences,
               .offset = edge.source.text_offset.value_or(0),
               .message = "control flow targets .text alignment"};
      if (!edge.source.text_offset)
        error.message += " (source VA " + std::to_string(edge.source.address) + ")";
      return false;
    }
    if (!edge.fallthrough && !edge.source.text_offset)
      facts->references.branch_addresses.push_back(edge.destination);
  }
  source_facts_ = std::move(facts);
  return true;
}

const InstructionBoundaries *Instrumentor::instruction_boundaries(TextEditError &error) {
  return ensure_source_facts(error) ? &source_facts_->boundaries : nullptr;
}

FailureOr<std::unique_ptr<Instrumentor>> Instrumentor::fork(TextEditError &error) {
  if (!ensure_source_facts(error))
    return Result::failure();
  std::unique_ptr<Instrumentor> child = std::make_unique<Instrumentor>(obj_, arch_, options_);
  child->source_facts_ = source_facts_;
  return child;
}

bool Instrumentor::ensure_decoder(std::string *error_out) {
  if (!obj_.is_valid()) {
    report(error_out, "cannot instrument an invalid code object");
    return false;
  }
  if (decoder_)
    return true;
  if (arch_ == rocjitsu::IsaFamily::Rv32i || arch_ == rocjitsu::IsaFamily::Rv64i) {
    report(error_out, "AMDGPU instrumentation does not support RISC-V architectures");
    return false;
  }
  const IsaTargetRegistry &registry = default_isa_target_registry();
  const rj_code_target_id_t target = obj_.target_id();
  if (target != ROCJITSU_CODE_TARGET_INVALID) {
    const IsaTargetDescriptor *descriptor = registry.find(target);
    if (descriptor == nullptr || descriptor->family != arch_) {
      report(error_out, "code-object target does not match the requested architecture");
      return false;
    }
  } else if (obj_.image_size() >= sizeof(Elf64_Ehdr)) {
    Elf64_Ehdr header = {};
    std::memcpy(&header, obj_.image_data(), sizeof(header));
    const rocjitsu::IsaFamily source_arch = arch_for_elf_mach(header.e_flags);
    if (source_arch != rocjitsu::IsaFamily::Invalid && source_arch != arch_) {
      report(error_out, "code-object ELF architecture does not match the requested architecture");
      return false;
    }
  }
  std::unique_ptr<Decoder> decoder = target == ROCJITSU_CODE_TARGET_INVALID
                                         ? Decoder::create(arch_)
                                         : Decoder::create(registry, target);
  if (!decoder) {
    report(error_out, "no decoder available for the requested architecture");
    return false;
  }
  decoder_ = std::move(decoder);
  return true;
}

bool Instrumentor::ensure_blocks_built(std::string *error_out) {
  if (blocks_built_)
    return true;
  if (!ensure_decoder(error_out))
    return false;
  util::StringDiagnostic decode_error;
  const AmdGpuCodeObject &analysis = analysis_object_ ? *analysis_object_ : obj_;
  std::vector<uint64_t> kernel_entries;
  if (analysis.text_sections().size() == 1) {
    const Section *text = analysis.text_sections().front();
    const std::span<const uint8_t> image(reinterpret_cast<const uint8_t *>(analysis.image_data()),
                                         analysis.image_size());
    const std::vector<KernelDescriptorInfo> kernels =
        scan_kernel_descriptors(image, text->sectionOffset(), text->size());
    const uint64_t entry_text_size =
        obj_.text_sections().size() == 1
            ? std::min(text->size(), obj_.text_sections().front()->size())
            : 0;
    for (const KernelDescriptorInfo &kernel : kernels) {
      kernel_entries.push_back(kernel.entry_text_offset);
      const std::optional<uint64_t> firmware =
          kernarg_preload_firmware_entry(kernel, arch_, entry_text_size);
      if (firmware)
        kernel_entries.push_back(*firmware);
    }
    std::ranges::sort(kernel_entries);
    kernel_entries.erase(std::ranges::unique(kernel_entries).begin(), kernel_entries.end());
  }
  // Both firmware paths are external roots: a split point alone would retain
  // indirect-target facts from a prefix the command processor can skip.
  FailureOr<std::vector<std::unique_ptr<BasicBlock>>> blocks =
      BasicBlock::build(analysis, *decoder_, arch_, decode_error.emitter(), kernel_entries);
  if (blocks.failed()) {
    report(error_out, decode_error.message().c_str());
    return false;
  }
  blocks_ = std::move(blocks).value();
  for (const auto &block : blocks_) {
    for (const Instruction &inst : block->instructions()) {
      offset_to_inst_.emplace(inst.src_loc(), &inst);
      instruction_boundaries_.record(inst, analysis.text_sections().front()->vaddr());
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
                                         std::string *error_out,
                                         std::span<const InstrumentationPoint> observations) {
  if (liveness_)
    return true;
  if (!ensure_blocks_built(error_out) || obj_.text_sections().size() != 1) {
    report(error_out, "liveness requires one decodable text section");
    return false;
  }
  const AmdGpuCodeObject &analysis = analysis_object_ ? *analysis_object_ : obj_;
  const Section *text = analysis.text_sections().front();
  std::vector<BasicBlock *> scope;
  LivenessAnalysisOptions options = {
      .arch = arch_,
      .target = effective_target(analysis, arch_),
      .additional_entry_blocks = {},
      .text = {reinterpret_cast<const uint8_t *>(text->data()), text->size()}};
  if (planning_descriptor_offset_ && kernels.size() == 1) {
    const BlockOffsetIndex offsets = build_block_offset_index(blocks_);
    const BlockPositionIndex positions = build_block_position_index(blocks_);
    BasicBlock *entry = block_for_offset(offsets, kernels.front().entry_text_offset);
    if (entry == nullptr || entry->start_offset() != kernels.front().entry_text_offset) {
      report(error_out, "selected entry has no decoded block");
      return false;
    }
    KernelScopeSpec spec;
    spec.own_entries.insert(kernels.front().entry_text_offset);
    scope = reachable_kernel_blocks(blocks_, offsets, positions, *entry, spec);
    options.entry_block = entry;
  } else {
    for (const std::unique_ptr<BasicBlock> &block : blocks_) {
      scope.push_back(block.get());
      if (kernels.size() == 1 && block->start_offset() == kernels.front().entry_text_offset)
        options.entry_block = block.get();
    }
  }
  std::vector<AdditionalRegisterUse> uses;
  for (const InstrumentationPoint &point : observations) {
    // KernelEntry probes run only on dispatch, before this CFG. Their arguments
    // must not become reads on guest backedges to the original entry.
    if (point.probe_obj == nullptr || point.kind == InstrumentationKind::KernelEntry)
      continue;
    const Instruction *anchor = find_instruction_at_offset(point.anchor_offset);
    if (anchor != nullptr)
      uses.push_back({.instruction = anchor,
                      .registers = probe_argument_registers(point.probe_args),
                      .after_instruction = point.kind == InstrumentationKind::AfterInst});
  }
  options.additional_uses = uses;

  // Ordinary successors omit callees; their live-ins must reach the caller.
  std::vector<ScopedCfgEdge> edges =
      scoped_call_liveness_edges(KernelBlockScope(scope), options.text);
  // State admission also recognizes computed re-entry. Preserve its actual
  // destination; the call's continuation alone omits values read on re-entry.
  const auto add_entry_edge = [&](BasicBlock *from, std::optional<uint64_t> target) {
    if (!target)
      return true;
    const std::vector<BasicBlock *>::const_iterator entry =
        std::ranges::find(scope, *target, &BasicBlock::start_offset);
    if (entry == scope.end()) {
      report(error_out, "computed kernel re-entry has no decoded entry block");
      return false;
    }
    edges.push_back({.from = from, .to = *entry});
    return true;
  };
  for (BasicBlock *block : scope) {
    if (!add_entry_edge(block, static_pc_reentry_target(*block, kernels)))
      return false;
    for (const BasicBlock::CallEdge &call : block->call_edges())
      if (!add_entry_edge(call.callee, callee_reentry_target(call, kernels, arch_)))
        return false;
  }

  // A full-lane definition ends an observed value's live range, but only when
  // EXEC is proven over this same graph. Calls and opaque transfers lack an
  // interprocedural EXEC proof. A descriptor entry inside a block is another
  // unmodeled input, since preceding instructions need not run on dispatch.
  bool can_analyze_exec = !kernels.empty();
  std::vector<const BasicBlock *> entries;
  const uint64_t entry_text_size = std::min(text->size(), obj_.text_sections().front()->size());
  for (const KernelDescriptorInfo &kernel : kernels) {
    const std::vector<BasicBlock *>::const_iterator entry =
        std::ranges::find(scope, kernel.entry_text_offset, &BasicBlock::start_offset);
    if (entry == scope.end())
      can_analyze_exec = false;
    else
      entries.push_back(*entry);
    if (!has_kernarg_preload_firmware_entry(kernel, arch_))
      continue;
    // Firmware dispatch has clear VSKIP but may have partial EXEC, even though
    // the skipped-prefix boundary is represented in this graph.
    can_analyze_exec = false;
    const std::optional<uint64_t> firmware =
        kernarg_preload_firmware_entry(kernel, arch_, entry_text_size);
    if (firmware) {
      const std::vector<BasicBlock *>::const_iterator extra =
          std::ranges::find(scope, *firmware, &BasicBlock::start_offset);
      if (extra != scope.end())
        entries.push_back(*extra);
    }
  }
  for (const BasicBlock *block : scope) {
    if (!block->call_edges().empty() || block->has_implicit_terminator() ||
        block->successor_issue() != BasicBlock::SuccessorIssue::None)
      can_analyze_exec = false;
    for (const Instruction &inst : block->instructions()) {
      if (is_control_flow_transfer(inst) && !is_program_path_terminator(inst) &&
          !inst.branch_offset_bytes())
        can_analyze_exec = false;
    }
  }
  // VSKIP is clear on dispatch. Keep this entry proof separate from EXEC,
  // whose dispatch mask can be partial, and merge it with guest backedges.
  options.vskip_clear_entry_blocks = entries;
  std::unique_ptr<ExecMaskAnalysis> exec;
  if (can_analyze_exec)
    exec = std::make_unique<ExecMaskAnalysis>(KernelBlockScope(scope), wave_size, edges, entries);
  liveness_ = std::make_unique<LivenessAnalysis>(KernelBlockScope(scope), wave_size,
                                                 std::move(exec), options, edges);
  return true;
}

bool Instrumentor::prepare_mutation_analysis(
    std::span<const ResolvedInstrumentationSite> replacements, TextEditError &error) {
  CodeObjectPatcher patcher(obj_);
  const size_t original_size = patcher.text_bytes().size();
  std::vector<uint8_t> text(patcher.text_bytes().begin(), patcher.text_bytes().end());
  std::vector<CheckedTextEdit> edits;
  for (const ResolvedInstrumentationSite &site : replacements) {
    const TrampolinePlan plan = make_base_plan(site, arch_, text.size());
    std::string message;
    const std::optional<TrampolineBytes> bytes = TrampolineBuilder::build(plan, &message);
    if (!bytes) {
      error = {.code = TextEditErrorCode::InvalidRange,
               .offset = site.anchor_offset,
               .message = std::move(message)};
      return false;
    }
    edits.push_back({.offset = site.anchor_offset,
                     .expected = site.original_bytes,
                     .replacement = bytes->patched_anchor_bytes});
    append_words(text, bytes->trampoline_words);
  }
  const CheckedTextEdits transaction(obj_, *decoder_, source_facts_->boundaries,
                                     source_facts_->references);
  if (transaction.apply(edits, std::span(text).first(original_size), error).failed())
    return false;
  if (!patcher.replace_text(text)) {
    error = {.code = TextEditErrorCode::InvalidImage,
             .offset = 0,
             .message = "cannot construct the private replacement analysis layout"};
    return false;
  }
  const std::span<const uint8_t> image = patcher.image_bytes();
  auto object = std::make_unique<AmdGpuCodeObject>(image.data(), image.size());
  if (!object->is_valid()) {
    error = {.code = TextEditErrorCode::InvalidImage,
             .offset = 0,
             .message = "cannot parse the private replacement analysis layout"};
    return false;
  }
  // Rebuild dataflow from all effective replacement bodies, including deleted
  // definitions at other sites. PC-independent bodies make this layout
  // semantically equivalent to the final probe-bearing layout. Pristine
  // live-before sets are not a safe union.
  liveness_.reset();
  callable_state_analysis_.reset();
  blocks_.clear();
  offset_to_inst_.clear();
  instruction_boundaries_ = {};
  blocks_built_ = false;
  analysis_object_ = std::move(object);
  return true;
}

Instrumentor::ResolvedPoints Instrumentor::resolve_points() {
  ResolvedPoints out;
  const bool has_replacements = std::ranges::any_of(points_, [](const InstrumentationPoint &point) {
    return point.kind == InstrumentationKind::ReplaceWithInst;
  });
  std::vector<InstrumentationPoint> requests = points_;
  std::vector<ResolvedInstrumentationSite> replacements;
  std::unordered_map<uint64_t, size_t> replacement_indices;
  std::string err;
  if (has_replacements) {
    TextEditError error;
    if (!ensure_source_facts(error)) {
      out.errors.push_back(error.message);
      out.edit_error = std::move(error);
      return out;
    }
    if (!ensure_decoder(&err)) {
      out.errors.push_back(std::move(err));
      return out;
    }
    std::vector<CheckedTextEdit> source_edits;
    for (const InstrumentationPoint &point : requests) {
      if (point.kind != InstrumentationKind::ReplaceWithInst)
        continue;
      if (!point.replacement) {
        out.errors.emplace_back("ReplaceWithInst requires expected and replacement bytes");
        return out;
      }
      if (point.filter_flags != 0 || (point.probe_obj == nullptr) != point.probe_symbol.empty() ||
          (!point.probe_obj && (!point.probe_args.empty() || point.force_full_exec))) {
        out.errors.emplace_back(
            "replacement probe fields are inconsistent or filters are unsupported");
        return out;
      }
      const std::vector<uint8_t> &expected = point.replacement->expected_bytes;
      if (expected.size() != 4 && expected.size() != 8 && expected.size() != 12) {
        out.errors.emplace_back("replacement requires one complete 4/8/12-byte source instruction");
        out.edit_error = TextEditError{.code = TextEditErrorCode::InvalidRange,
                                       .offset = point.anchor_offset,
                                       .message = out.errors.back()};
        return out;
      }
      source_edits.push_back(
          {.offset = point.anchor_offset, .expected = expected, .replacement = {}});
      replacement_indices.emplace(point.anchor_offset, replacements.size());
      replacements.push_back({.kind = InstrumentationKind::ReplaceWithInst,
                              .anchor_offset = point.anchor_offset,
                              .original_size = static_cast<uint32_t>(expected.size()),
                              .original_bytes = expected,
                              .mnemonic = {},
                              .probe_index = std::nullopt,
                              .probe_args = {}});
    }
    const CheckedTextEdits transaction(obj_, *decoder_, source_facts_->boundaries,
                                       source_facts_->references);
    if (transaction.validate_source_edits(source_edits, error).failed()) {
      out.errors.push_back(error.message);
      out.edit_error = std::move(error);
      return out;
    }
    for (const InstrumentationPoint &point : requests) {
      if (point.kind != InstrumentationKind::ReplaceWithInst)
        continue;
      ResolvedInstrumentationSite &site = replacements[replacement_indices.at(point.anchor_offset)];
      const std::vector<uint8_t> &bytes = point.replacement->replacement_bytes;
      if (bytes.size() % sizeof(uint32_t) != 0 ||
          bytes.size() > std::numeric_limits<uint32_t>::max()) {
        out.errors.emplace_back("replacement body must contain complete instruction dwords");
        out.edit_error = TextEditError{.code = TextEditErrorCode::InvalidInstruction,
                                       .offset = point.anchor_offset,
                                       .message = out.errors.back()};
        return out;
      }
      site.replacement_words.resize(bytes.size() / sizeof(uint32_t));
      if (!bytes.empty())
        std::memcpy(site.replacement_words.data(), bytes.data(), bytes.size());
      size_t cursor = 0;
      while (cursor < site.replacement_words.size()) {
        const DecodeResult decoded = decoder_->decode_window(
            std::span(site.replacement_words).subspan(cursor), point.anchor_offset + cursor * 4);
        if (decoded.failed() || decoded.value()->size() == 0) {
          out.errors.emplace_back("replacement contains an invalid or truncated instruction");
          out.edit_error = TextEditError{.code = TextEditErrorCode::InvalidInstruction,
                                         .offset = point.anchor_offset,
                                         .message = out.errors.back()};
          return out;
        }
        const Instruction &instruction = *decoded.value();
        if (is_s_clause(instruction)) {
          out.errors.emplace_back("replacement may not introduce a clause");
          out.edit_error = TextEditError{.code = TextEditErrorCode::Clause,
                                         .offset = point.anchor_offset,
                                         .message = out.errors.back()};
          return out;
        }
        if ((instruction.flags() &
             (BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL | BRANCH_STACK)) != 0 ||
            instruction.branch_offset_bytes() || is_pc_relative_anchor(instruction) ||
            has_unavailable_branch_reference(instruction)) {
          out.errors.emplace_back(
              "replacement body contains unsupported PC-dependent control flow: " +
              std::string(instruction.mnemonic()));
          out.edit_error = TextEditError{.code = TextEditErrorCode::InvalidInstruction,
                                         .offset = point.anchor_offset,
                                         .message = out.errors.back()};
          return out;
        }
        cursor += static_cast<size_t>(instruction.size()) / sizeof(uint32_t);
        if ((instruction.flags() & PROGRAM_TERMINATOR) != 0) {
          if (instruction.mnemonic() != "s_endpgm" || cursor != site.replacement_words.size()) {
            out.errors.emplace_back("only a final s_endpgm may terminate a replacement body");
            out.edit_error = TextEditError{.code = TextEditErrorCode::InvalidInstruction,
                                           .offset = point.anchor_offset,
                                           .message = out.errors.back()};
            return out;
          }
          site.replacement_terminates = true;
        }
      }
      if (!site.replacement_terminates &&
          !source_facts_->boundaries.find(site.anchor_offset + site.original_size)) {
        out.errors.emplace_back(
            "replacement has no instruction to continue to after the source extent");
        out.edit_error = TextEditError{.code = TextEditErrorCode::InvalidInstruction,
                                       .offset = point.anchor_offset,
                                       .message = out.errors.back()};
        return out;
      }
    }
    // A single replacement point can carry a probe directly. Also coalesce the
    // explicit BeforeInst+Replace pair independently of request order. Other
    // duplicates retain their existing all-or-nothing rejection.
    for (const InstrumentationPoint &point : requests) {
      const std::unordered_map<uint64_t, size_t>::const_iterator replacement =
          replacement_indices.find(point.anchor_offset);
      if (point.kind != InstrumentationKind::BeforeInst || replacement == replacement_indices.end())
        continue;
      if (point.replacement || !point.probe_obj || point.probe_symbol.empty() ||
          point.filter_flags) {
        out.errors.emplace_back("same-site observation composition requires a callable BeforeInst");
        return out;
      }
      InstrumentationPoint *destination = nullptr;
      for (InstrumentationPoint &candidate : requests) {
        if (candidate.kind == InstrumentationKind::ReplaceWithInst &&
            candidate.anchor_offset == point.anchor_offset) {
          destination = &candidate;
          break;
        }
      }
      if (!destination || destination->probe_obj || !destination->probe_symbol.empty() ||
          !destination->probe_args.empty() || destination->force_full_exec) {
        out.errors.emplace_back("multiple probes requested the same replacement site");
        return out;
      }
      destination->probe_obj = point.probe_obj;
      destination->probe_symbol = point.probe_symbol;
      destination->probe_args = point.probe_args;
      destination->force_full_exec = point.force_full_exec;
    }
    // Copy from requests after coalescing, so a replacement preceding its probe
    // sees the same merged fields as the opposite queue order.
    std::vector<InstrumentationPoint> merged;
    for (const InstrumentationPoint &point : requests) {
      if (point.kind == InstrumentationKind::BeforeInst &&
          replacement_indices.contains(point.anchor_offset))
        continue;
      merged.push_back(point);
    }
    requests = std::move(merged);

    // Linked getpc/add constants carry no ELF relocation to adjust when data
    // moves. Keep those layouts unsupported until the complete builder can be
    // relocated; a getpc removed by this batch no longer produces a stale VA.
    const auto is_removed = [&](const SourceTextFacts::SourceLocation &source) {
      return source.text_offset && replacement_indices.contains(*source.text_offset);
    };
    const std::vector<SourceTextFacts::SourceLocation>::const_iterator retained_pc =
        std::ranges::find_if(
            source_facts_->pc_reads,
            [&](const SourceTextFacts::SourceLocation &source) { return !is_removed(source); });
    const bool has_retained_edge = std::ranges::any_of(
        source_facts_->flow_edges,
        [&](const SourceTextFacts::ControlFlowEdge &edge) { return !is_removed(edge.source); });
    if (retained_pc != source_facts_->pc_reads.end() || has_retained_edge ||
        !source_facts_->foreign_kernel_entries.empty()) {
      const Section &text = *obj_.text_sections().front();
      const uint64_t text_end = text.vaddr() + text.size();
      const uint64_t file_end = text.sectionOffset() + text.size();
      const CodeObjectPatcher source_patcher(obj_);
      const std::vector<Elf64_Shdr> sections = source_patcher.section_headers();
      const std::span<const uint8_t> image(reinterpret_cast<const uint8_t *>(obj_.image_data()),
                                           obj_.image_size());
      Elf64_Ehdr header = {};
      std::memcpy(&header, image.data(), sizeof(header));
      const std::optional<std::vector<Elf64_Phdr>> segments =
          read_elf_table<Elf64_Phdr>(image, header.e_phoff, header.e_phnum, header.e_phentsize);
      if (!segments) {
        out.errors.emplace_back("invalid program headers for replacement text growth");
        return out;
      }
      const auto storage_moves = [&](std::optional<uint64_t> destination) {
        const auto contains = [&](uint64_t base, uint64_t size, uint64_t begin = 0) {
          return size > begin &&
                 (!destination || (*destination >= base && *destination - base >= begin &&
                                   *destination - base < size));
        };
        for (const Elf64_Shdr &section : sections) {
          if ((section.sh_flags & SHF_ALLOC) != 0 && section.sh_addr >= text_end &&
              contains(section.sh_addr, section.sh_size))
            return true;
        }
        for (const Elf64_Phdr &segment : *segments) {
          if (segment.p_type != PT_LOAD || segment.p_memsz == 0)
            continue;
          if (segment.p_offset >= file_end) {
            if (segment.p_vaddr >= text_end && contains(segment.p_vaddr, segment.p_memsz))
              return true;
          } else {
            // Include unsectioned file-backed data and zero-fill after .text.
            const uint64_t insertion_offset = file_end - segment.p_offset;
            if (insertion_offset <= segment.p_filesz &&
                contains(segment.p_vaddr, segment.p_memsz, insertion_offset))
              return true;
          }
        }
        return false;
      };
      if (retained_pc != source_facts_->pc_reads.end() && storage_moves(std::nullopt)) {
        out.errors.emplace_back(
            "replacement growth cannot preserve retained PC-relative addresses to moved storage");
        if (!retained_pc->text_offset)
          out.errors.back() += " (source VA " + std::to_string(retained_pc->address) + ")";
        out.edit_error = TextEditError{.code = TextEditErrorCode::UnsupportedReferences,
                                       .offset = retained_pc->text_offset.value_or(0),
                                       .message = out.errors.back()};
        return out;
      }
      for (uint64_t entry : source_facts_->foreign_kernel_entries) {
        if (!storage_moves(entry))
          continue;
        out.errors.emplace_back(
            "replacement growth cannot preserve a kernel entry in another executable section "
            "(entry VA " +
            std::to_string(entry) + ")");
        out.edit_error = TextEditError{.code = TextEditErrorCode::UnsupportedReferences,
                                       .offset = 0,
                                       .message = out.errors.back()};
        return out;
      }
      for (const SourceTextFacts::ControlFlowEdge &edge : source_facts_->flow_edges) {
        if (is_removed(edge.source) ||
            storage_moves(edge.source.address) == storage_moves(edge.destination))
          continue;
        out.errors.emplace_back(
            edge.fallthrough
                ? "replacement growth cannot preserve a retained fallthrough edge"
                : "replacement growth cannot preserve a retained direct branch displacement");
        if (!edge.source.text_offset)
          out.errors.back() += " (source VA " + std::to_string(edge.source.address) + ")";
        out.edit_error = TextEditError{.code = TextEditErrorCode::UnsupportedReferences,
                                       .offset = edge.source.text_offset.value_or(0),
                                       .message = out.errors.back()};
        return out;
      }
    }
  }
  const bool needs_graph =
      std::ranges::any_of(requests,
                          [](const InstrumentationPoint &point) {
                            return point.kind != InstrumentationKind::ReplaceWithInst ||
                                   point.probe_obj != nullptr;
                          }) ||
      options_.register_growth.has_value();
  if (has_replacements && needs_graph) {
    TextEditError error;
    if (!prepare_mutation_analysis(replacements, error)) {
      out.errors.push_back(error.message);
      out.edit_error = std::move(error);
      return out;
    }
  }
  if (needs_graph && !ensure_blocks_built(&err)) {
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
  if (std::ranges::any_of(
          points_, [](const InstrumentationPoint &point) { return point.probe_obj != nullptr; })) {
    if (has_replacements) {
      for (const std::unique_ptr<BasicBlock> &block : blocks_) {
        // The local liveness scope has no external or call/return edge overlay.
        if (block->successor_issue() == BasicBlock::SuccessorIssue::None &&
            block->call_edges().empty())
          continue;
        const Instruction *terminator = block->terminator();
        const uint64_t offset =
            terminator && terminator->src_loc() < obj_.text_sections().front()->size()
                ? terminator->src_loc()
                : 0;
        out.errors.emplace_back(
            !block->call_edges().empty()
                ? "replacement probe does not support call/return liveness in the effective CFG"
                : "replacement probe requires an effective CFG with all guest successors analyzed");
        out.edit_error = TextEditError{.code = TextEditErrorCode::UnsupportedReferences,
                                       .offset = offset,
                                       .message = out.errors.back()};
        return out;
      }
    }
    // M0-displaced SGPR accesses are absent from ordinary def/use facts.
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
  const Section *analysis_text =
      analysis_object_ ? analysis_object_->text_sections().front() : text;
  const std::span<const uint8_t> analysis_text_bytes(
      reinterpret_cast<const uint8_t *>(analysis_text->data()), analysis_text->size());
  // A probe call is analyzed and planned at the wave size the object's kernels
  // share, with temps bounded by their SGPR allocation. The point loop refuses a
  // probe call before any of that when either is missing.
  const std::span<const uint8_t> image(reinterpret_cast<const uint8_t *>(obj_.image_data()),
                                       obj_.image_size());
  out.kernels = scan_kernel_descriptors(image, text->sectionOffset(), text->size());
  if (planning_descriptor_offset_) {
    std::erase_if(out.kernels, [&](const KernelDescriptorInfo &kernel) {
      return kernel.descriptor_file_offset != *planning_descriptor_offset_;
    });
    if (out.kernels.size() != 1) {
      out.errors.emplace_back("selected descriptor is not unique in the original image");
      return out;
    }
  }
  out.wave_size = common_wavefront_size(arch_, out.kernels);

  const IsaTargetRegistry &registry = default_isa_target_registry();
  const rj_code_target_id_t destination_target = effective_target(obj_, arch_);
  const IsaTargetBinding *gpu_target = registry.find_target(destination_target);
  const bool setreg_vgpr_msb_fixup =
      gpu_target == nullptr || gpu_target->capabilities.setreg_vgpr_msb_fixup;
  ProbeEntryState probe_entry_state = ProbeEntryState::Unknown;
  const bool has_probe_requests = std::ranges::any_of(
      requests, [](const InstrumentationPoint &point) { return point.probe_obj != nullptr; });
  // Replacement liveness and physical VGPR observations require enumerable uses.
  // Ordinary observations retain per-point MODE admission; AfterInst also checks
  // completeness before preservation planning.
  const bool requires_complete_vgpr_usage =
      has_replacements || std::ranges::any_of(requests, [](const InstrumentationPoint &point) {
        return point.probe_obj != nullptr &&
               std::ranges::any_of(point.probe_args, [](const ProbeArgValue &arg) {
                 return arg.source == ProbeArgSource::AnchorVgpr;
               });
      });
  if (has_probe_requests && !out.kernels.empty() && out.wave_size) {
    if (!ensure_liveness_built(*out.wave_size, out.kernels, &err, requests)) {
      out.errors.push_back(std::move(err));
      return out;
    }
    const util::FailureOr<bool> complete = liveness_->global_vgpr_usage_is_complete();
    if (complete.failed()) {
      out.errors.emplace_back("probe calls require available guest liveness");
      return out;
    }
    if (requires_complete_vgpr_usage && !complete.value()) {
      out.errors.emplace_back("probe calls do not support relative or GPR-indexed VGPR access");
      return out;
    }
  }
  if (has_probe_requests && arch_ == rocjitsu::IsaFamily::Cdna5 && out.wave_size) {
    if (out.kernels.size() == 1) {
      bool has_call = false;
      for (const std::unique_ptr<BasicBlock> &block : blocks_) {
        for (const Instruction &instruction : block->instructions())
          has_call |= (instruction.flags() & INDIRECT_CALL) != 0;
      }
      if (!has_call) {
        bool bank_analysis_failed = false;
        const bool zero = std::ranges::all_of(requests, [&](const InstrumentationPoint &point) {
          if (!point.probe_obj)
            return true;
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
  sites.reserve(requests.size());
  for (const InstrumentationPoint &pt : requests) {
    const Instruction *anchor = find_instruction_at_offset(pt.anchor_offset);
    std::unique_ptr<Instruction> source_anchor;
    if (analysis_object_ && pt.kind != InstrumentationKind::ReplaceWithInst) {
      // Validate and capture the pristine instruction. The effective CFG has a
      // replacement branch at an entry shared with a replacement request.
      anchor = nullptr;
      const InstructionBoundaries::Entry *boundary =
          source_facts_->boundaries.find(pt.anchor_offset);
      if (boundary != nullptr) {
        if (pt.anchor_offset > text_bytes.size() ||
            boundary->size > text_bytes.size() - pt.anchor_offset) {
          out.errors.emplace_back("source anchor extends past end of .text");
          continue;
        }
        const std::span<const uint32_t> words(
            reinterpret_cast<const uint32_t *>(text_bytes.data() + pt.anchor_offset),
            boundary->size / sizeof(uint32_t));
        util::StringDiagnostic diagnostic;
        DecodeResult decoded =
            decoder_->decode_window(words, pt.anchor_offset, diagnostic.emitter());
        if (decoded.failed() || static_cast<uint32_t>(decoded.value()->size()) != boundary->size) {
          out.errors.push_back("cannot decode source anchor at .text offset " +
                               std::to_string(pt.anchor_offset) + ": " + diagnostic.message());
          continue;
        }
        source_anchor = std::move(decoded).value();
        anchor = source_anchor.get();
      }
    }
    if (anchor == nullptr && pt.kind != InstrumentationKind::ReplaceWithInst) {
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
    std::optional<ResolvedInstrumentationSite> site;
    if (pt.kind == InstrumentationKind::ReplaceWithInst)
      site = replacements[replacement_indices.at(pt.anchor_offset)];
    else
      site = validate_anchor(*anchor, pt.anchor_offset, text_bytes, pt, arch_, &perr,
                             setreg_vgpr_msb_fixup);
    if (!site) {
      out.errors.push_back(std::move(perr));
      continue;
    }

    const InstructionBoundaries &source_boundaries =
        source_facts_ ? source_facts_->boundaries : instruction_boundaries_;
    const InstructionBoundaries::Entry *boundary = source_boundaries.find(pt.anchor_offset);
    if (boundary && boundary->clause_member) {
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
      if (pt.kind == InstrumentationKind::KernelEntry) {
        // Only dispatch executes the entry stub. Guest backedges to the original
        // entry do not change the storage available to this callback.
        if (out.kernels.size() != 1 || pt.anchor_offset != out.kernels.front().entry_text_offset) {
          out.errors.emplace_back("KernelEntry must name the sole kernel's descriptor entry");
          continue;
        }
        if (uses_dynamic_vgprs(arch_, out.kernels.front().descriptor)) {
          out.errors.emplace_back("KernelEntry requires a static VGPR allocation at dispatch");
          continue;
        }
      } else {
        const Instruction *state_anchor = find_instruction_at_offset(pt.anchor_offset);
        if (state_anchor == nullptr) {
          out.errors.emplace_back("effective anchor is unavailable for callable-state analysis");
          continue;
        }
        if (!callable_state_analysis_)
          callable_state_analysis_ = std::make_unique<CallableStateAnalysis>(
              blocks_, out.kernels, analysis_text_bytes, arch_, text->size());
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
        // Replacing the required follower with a branch is unsafe too. Until
        // adjacency is proven, exclude the family throughout the decoded object.
        if (pt.kind == InstrumentationKind::AfterInst &&
            callable_state_analysis_->has_cdna5_packed_permute()) {
          out.errors.emplace_back("AfterInst cannot preserve the required VALU adjacency in a "
                                  "CDNA5 code object containing packed permutes");
          continue;
        }
        bool unsafe_state = false;
        for (const CallableState state : {CallableState::GprIndexing, CallableState::Vskip,
                                          CallableState::UnavailableStorage}) {
          if (callable_state_analysis_->may_be_unsafe(
                  *state_anchor, pt.kind == InstrumentationKind::AfterInst, state)) {
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
        if (unsafe_state)
          continue;
      }
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

InstrumentedCodeObjectDebug Instrumentor::patch_entries() {
  InstrumentedCodeObjectDebug result;
  if (options_.register_growth || options_.kernarg_binding != DbiKernargBinding::None ||
      options_.preservation != GuestStatePreservation::LiveRegisters) {
    result.errors.emplace_back("explicit entry requests cannot be combined with legacy options");
    return result;
  }
  if (points_.empty()) {
    result.errors.emplace_back("Instrumentor::patch requires at least one queued point; got zero");
    return result;
  }
  if (std::ranges::any_of(points_, [](const InstrumentationPoint &point) {
        return point.kind == InstrumentationKind::ReplaceWithInst;
      })) {
    result.errors.emplace_back("explicit entry requests do not admit instruction replacement");
    return result;
  }
  std::string error;
  if (!ensure_blocks_built(&error) || obj_.text_sections().size() != 1) {
    result.errors.push_back(error.empty() ? "entry requests require one text section" : error);
    return result;
  }
  const Section *text = obj_.text_sections().front();
  const std::vector<const Section *> &executable_sections = obj_.allocated_executable_sections();
  if (executable_sections.size() != 1 || executable_sections.front() != text) {
    result.errors.emplace_back(
        "entry requests require .text to be the sole allocated executable section");
    return result;
  }
  const std::span<const uint8_t> original(reinterpret_cast<const uint8_t *>(obj_.image_data()),
                                          obj_.image_size());
  const std::vector<KernelDescriptorInfo> kernels =
      scan_kernel_descriptors(original, text->sectionOffset(), text->size());
  if (!validate_entry_descriptor_symbols(original, kernels, &error)) {
    result.errors.push_back(std::move(error));
    return result;
  }
  if (kernels.empty() || !common_wavefront_size(arch_, kernels)) {
    result.errors.emplace_back("entry requests require descriptors with one common wave size");
    return result;
  }

  const BlockOffsetIndex offsets = build_block_offset_index(blocks_);
  const BlockPositionIndex positions = build_block_position_index(blocks_);
  std::unordered_map<const BasicBlock *, uint64_t> owners;
  std::unordered_set<uint64_t> descriptor_offsets;
  std::unordered_set<uint64_t> entry_offsets;
  std::unordered_set<std::string> kernel_names;
  for (const KernelDescriptorInfo &kernel : kernels) {
    if (kernarg_preload_length(kernel.descriptor) != 0) {
      result.errors.emplace_back("entry requests do not admit kernarg preload hardware entries");
      return result;
    }
    if (!descriptor_offsets.insert(kernel.descriptor_file_offset).second ||
        !entry_offsets.insert(kernel.entry_text_offset).second ||
        !kernel_names.insert(kernel.kernel_name).second) {
      result.errors.emplace_back("entry requests require distinct descriptors, entries and names");
      return result;
    }
  }
  // Establish every owner's complete reachable scope before selecting any sites.
  // The bounded profile has no calls or unknown transfers; stopping at another
  // descriptor is not evidence that a cross-entry edge is safe.
  for (const KernelDescriptorInfo &kernel : kernels) {
    BasicBlock *entry = block_for_offset(offsets, kernel.entry_text_offset);
    if (entry == nullptr || entry->start_offset() != kernel.entry_text_offset) {
      result.errors.emplace_back("kernel descriptor does not name a decoded block entry");
      return result;
    }
    KernelScopeSpec spec;
    spec.own_entries.insert(kernel.entry_text_offset);
    const std::vector<BasicBlock *> scope =
        reachable_kernel_blocks(blocks_, offsets, positions, *entry, spec);
    for (const BasicBlock *block : scope) {
      if (block->has_implicit_terminator() ||
          block->successor_issue() != BasicBlock::SuccessorIssue::None ||
          !block->call_edges().empty() ||
          (block != entry && entry_offsets.contains(block->start_offset())) ||
          !owners.emplace(block, kernel.descriptor_file_offset).second) {
        result.errors.emplace_back("entry requests require complete disjoint kernel scopes");
        return result;
      }
      for (const Instruction &instruction : block->instructions()) {
        if ((instruction.flags() & (INDIRECT_CALL | INDIRECT_BRANCH)) != 0) {
          result.errors.emplace_back("disjoint entry requests do not admit calls or indirect flow");
          return result;
        }
        if ((instruction.flags() & WRITES_PC) != 0 && !is_program_path_terminator(instruction) &&
            !instruction.branch_offset_bytes()) {
          result.errors.emplace_back(
              "disjoint entry requests do not admit an unknown branch target");
          return result;
        }
      }
    }
  }

  std::vector<EntryInstrumentationOptions> requests = options_.entries;
  std::ranges::sort(requests, {}, &EntryInstrumentationOptions::descriptor_file_offset);
  std::unordered_map<uint64_t, size_t> request_positions;
  for (size_t index = 0; index < requests.size(); ++index) {
    const uint64_t key = requests[index].descriptor_file_offset;
    if (!descriptor_offsets.contains(key) || !request_positions.emplace(key, index).second) {
      result.errors.emplace_back("entry request has an unknown or duplicate original descriptor");
      return result;
    }
  }
  std::vector<std::vector<InstrumentationPoint>> entry_points(requests.size());
  for (const InstrumentationPoint &point : points_) {
    BasicBlock *block = block_for_offset(offsets, point.anchor_offset);
    const std::unordered_map<const BasicBlock *, uint64_t>::const_iterator owner =
        owners.find(block);
    if (owner == owners.end() || !request_positions.contains(owner->second)) {
      result.errors.emplace_back("instrumentation point has no unique selected entry owner");
      return result;
    }
    entry_points[request_positions.at(owner->second)].push_back(point);
  }

  if (entry_validation_ != nullptr) {
    std::vector<ResolvedInstrumentationSite> sites;
    for (size_t index = 0; index < requests.size(); ++index) {
      if (entry_points[index].empty()) {
        result.errors.emplace_back("selected entry request has no instrumentation points");
        return result;
      }
      Instrumentor validator(obj_, arch_);
      validator.planning_descriptor_offset_ = requests[index].descriptor_file_offset;
      for (InstrumentationPoint &point : entry_points[index])
        validator.add_point(std::move(point));
      ValidationResult validation = validator.validate_points();
      if (!validation.errors.empty()) {
        result.errors = std::move(validation.errors);
        return result;
      }
      sites.insert(sites.end(), std::make_move_iterator(validation.sites.begin()),
                   std::make_move_iterator(validation.sites.end()));
    }
    std::map<std::pair<uint64_t, InstrumentationKind>, size_t> point_positions;
    for (size_t index = 0; index < points_.size(); ++index)
      point_positions.emplace(std::pair{points_[index].anchor_offset, points_[index].kind}, index);
    std::ranges::sort(sites, [&](const ResolvedInstrumentationSite &first,
                                 const ResolvedInstrumentationSite &second) {
      return point_positions.at({first.anchor_offset, first.kind}) <
             point_positions.at({second.anchor_offset, second.kind});
    });
    entry_validation_->sites = std::move(sites);
    return result;
  }

  CodeObjectPatcher patcher(obj_);
  uint64_t cursor = patcher.text_size();
  std::vector<EntryPatchPlan> plans(requests.size());
  for (size_t index = 0; index < requests.size(); ++index) {
    const EntryInstrumentationOptions &request = requests[index];
    if (entry_points[index].empty()) {
      result.errors.emplace_back("selected entry request has no instrumentation points");
      return result;
    }
    Instrumentor planner(obj_, arch_,
                         {.kernarg_binding = request.kernarg_binding,
                          .preservation = request.preservation,
                          .register_growth = request.register_growth});
    planner.skip_patch_gates_ = skip_patch_gates_;
    planner.planning_descriptor_offset_ = request.descriptor_file_offset;
    planner.planning_cave_offset_ = cursor;
    planner.entry_plan_ = &plans[index];
    for (InstrumentationPoint &point : entry_points[index])
      planner.add_point(std::move(point));
    InstrumentedCodeObjectDebug planned = planner.patch_with_debug_summaries();
    if (!planned.errors.empty()) {
      result.errors = std::move(planned.errors);
      result.edit_error = std::move(planned.edit_error);
      return result;
    }
    cursor = plans[index].cave_end;
  }

  std::vector<CheckedTextEdit> edits;
  std::vector<KernargExtensionMetadata> extensions;
  for (EntryPatchPlan &plan : plans) {
    if (plan.prologue &&
        !patcher.redirect_kernel_entry(plan.kernel.descriptor_file_offset,
                                       plan.kernel.entry_text_offset, plan.prologue->stub_offset)) {
      result.errors.emplace_back("failed to redirect a selected kernel entry");
      return result;
    }
    for (const AppliedSite &site : plan.applied) {
      if (site.site->kind == InstrumentationKind::KernelEntry) {
        if (!plan.prologue) {
          result.errors.emplace_back("kernel entry callback has no entry stub");
          return result;
        }
        const uint64_t branch_pc =
            plan.prologue->stub_offset + (plan.prologue->words.size() - 1) * sizeof(uint32_t);
        const std::optional<int16_t> displacement =
            compute_sopp_branch_simm16(branch_pc, site.trampoline_offset);
        if (!displacement) {
          result.errors.emplace_back("kernel entry callback is outside its stub's branch range");
          return result;
        }
        const util::FailureOr<uint32_t> branch = build_s_branch(*displacement, arch_);
        if (branch.failed()) {
          result.errors.emplace_back("cannot encode selected entry callback branch");
          return result;
        }
        plan.prologue->words.back() = branch.value();
      } else {
        edits.push_back(
            {.offset = site.site->anchor_offset,
             .expected = site.site->original_bytes,
             .replacement = {site.bytes.patched_anchor_bytes.begin(),
                             site.bytes.patched_anchor_bytes.begin() + site.site->original_size}});
      }
    }
    if (plan.prologue && plan.prologue->storage_base) {
      extensions.push_back({.kernel_name = plan.kernel.kernel_name,
                            .variant_name = std::string(kDbiKernargVariantName),
                            .original_kernarg_size = plan.kernel.descriptor.kernarg_size,
                            .payloads = {{.size = kDbiEntryPayloadLayout.size,
                                          .alignment = kDbiEntryPayloadLayout.alignment,
                                          .name = std::string(kDbiEntryPayloadName)}}});
    }
  }

  const std::span<const uint8_t> original_text = patcher.text_bytes();
  std::vector<uint8_t> new_text(original_text.begin(), original_text.end());
  if (!edits.empty()) {
    for (const std::unique_ptr<BasicBlock> &block : blocks_)
      for (const Instruction &instruction : block->instructions()) {
        const std::string_view mnemonic = instruction.mnemonic();
        if ((mnemonic.starts_with("s_cbranch_cdbg") || mnemonic == "s_subvector_loop_begin" ||
             mnemonic == "s_subvector_loop_end") &&
            !instruction.branch_offset_bytes()) {
          result.errors.emplace_back("anchor edit transaction has unsupported branch references");
          return result;
        }
      }
    std::vector<uint64_t> entries(entry_offsets.begin(), entry_offsets.end());
    TextEditError edit_error;
    const InstructionBoundaries &source_boundaries =
        source_facts_ ? source_facts_->boundaries : instruction_boundaries_;
    std::optional<TextEditReferences> local_references;
    if (!source_facts_) {
      FailureOr<TextEditReferences> collected = CheckedTextEdits::collect_references(
          obj_, arch_, source_boundaries, patcher, entries, edit_error);
      if (collected.failed()) {
        result.errors.push_back("anchor edit transaction: " + edit_error.message);
        result.edit_error = std::move(edit_error);
        return result;
      }
      local_references = std::move(collected).value();
    }
    const TextEditReferences &references =
        source_facts_ ? source_facts_->references : *local_references;
    const CheckedTextEdits transaction(obj_, *decoder_, source_boundaries, references);
    if (transaction.apply(edits, new_text, edit_error).failed()) {
      result.errors.push_back("anchor edit transaction: " + edit_error.message);
      result.edit_error = std::move(edit_error);
      return result;
    }
  }
  for (const EntryPatchPlan &plan : plans) {
    for (const ProbeCallable &probe : plan.resolved.probes)
      append_words(new_text, probe.body_words);
    if (plan.prologue) {
      if (append_nop_padding(new_text, plan.prologue->padding_bytes, arch_).failed()) {
        result.errors.emplace_back("cannot encode entry alignment padding");
        return result;
      }
      append_words(new_text, plan.prologue->words);
    }
    for (const AppliedSite &site : plan.applied)
      append_words(new_text, site.bytes.trampoline_words);
  }
  if (!patcher.replace_text(new_text)) {
    result.errors.emplace_back("failed to replace whole-image instrumented text");
    return result;
  }
  if (!extensions.empty()) {
    for (const std::unique_ptr<Section> &section : obj_.all_sections())
      if (section->name() == kKernargExtensionMetadataSectionName) {
        result.errors.emplace_back("input already declares a kernarg extension");
        return result;
      }
    const std::vector<uint8_t> bytes = serialize_kernarg_extension_metadata(extensions);
    if (!patcher.append_nonalloc_section(kKernargExtensionMetadataSectionName, bytes)) {
      result.errors.emplace_back("failed to append entry kernarg extension records");
      return result;
    }
  }

  std::vector<uint8_t> image = patcher.emit();
  for (size_t index = 0; index < plans.size(); ++index) {
    EntryPatchPlan &plan = plans[index];
    const EntryInstrumentationOptions &request = requests[index];
    AmdGpuCodeObject updated(image.data(), image.size());
    if (!updated.is_valid() || updated.text_sections().size() != 1) {
      result.errors.emplace_back("cannot reparse whole-image resource update");
      return result;
    }
    const Section *updated_text = updated.text_sections().front();
    const std::vector<KernelDescriptorInfo> updated_kernels =
        scan_kernel_descriptors(image, updated_text->sectionOffset(), updated_text->size());
    const std::vector<KernelDescriptorInfo>::const_iterator found = std::ranges::find(
        updated_kernels, plan.kernel.kernel_name, &KernelDescriptorInfo::kernel_name);
    if (updated_kernels.size() != kernels.size() || found == updated_kernels.end()) {
      result.errors.emplace_back("whole-image resource update lost a kernel descriptor");
      return result;
    }
    const KernelDescriptorInfo &kernel = *found;
    amdhsa::kernel_descriptor_t descriptor = kernel.descriptor;
    if (plan.private_bytes)
      descriptor.private_segment_fixed_size = *plan.private_bytes;
    plan.resources.kernarg_bytes = descriptor.kernarg_size;
    const bool has_storage = plan.prologue && plan.prologue->storage_base;
    if (has_storage && request.kernarg_binding == DbiKernargBinding::CallerSuppliedWrapper) {
      const std::array payloads{kDbiEntryPayloadLayout};
      const std::optional<KernargExtensionLayout> wrapper =
          make_kernarg_extension_layout(descriptor.kernarg_size, payloads);
      if (!wrapper) {
        result.errors.emplace_back("selected entry kernarg wrapper overflows");
        return result;
      }
      plan.resources.kernarg_bytes = wrapper->wrapper_size;
      descriptor.kernarg_size = wrapper->wrapper_size;
    }
    if (request.register_growth) {
      const uint32_t granule = descriptor_vgpr_granularity_for_wavefront(
          arch_, kernel_wavefront_size(arch_, descriptor));
      plan.resources.vgprs = (plan.resources.vgprs + granule - 1) / granule * granule;
      if (plan.resources.sgprs > REGISTER_SET_ALLOCATABLE_SGPRS || plan.resources.vgprs > 256) {
        result.errors.emplace_back("selected entry exceeds bounded register allocation");
        return result;
      }
      if (arch_ == rocjitsu::IsaFamily::Cdna3 || arch_ == rocjitsu::IsaFamily::Cdna4) {
        const std::array source_kernel{plan.kernel};
        // gfx942/gfx950 round physical SGPR storage to 16 dwords. The
        // FLAT_SCRATCH, XNACK_MASK and VCC pairs alias its final six slots.
        const auto alias_end = [](uint32_t requested) {
          return std::min(amdgpu::kVccSelectorLast + 1, std::max(16u, (requested + 15u) & ~15u));
        };
        const uint32_t old_end =
            alias_end(*AmdGpuCodeObject::min_kernel_sgpr_count(arch_, source_kernel));
        const uint32_t new_end = alias_end(plan.resources.sgprs + kCdnaSpecialSgprTailReserve);
        const auto uses_alias = [&](uint32_t end) {
          return plan.guest_registers.intersects(
              {RegClass::SGPR, static_cast<uint16_t>(end - kCdnaSpecialSgprTailReserve),
               kCdnaSpecialSgprTailReserve});
        };
        if (old_end != new_end && (uses_alias(old_end) || uses_alias(new_end))) {
          result.errors.emplace_back("selected entry growth changes a guest numeric SGPR alias");
          return result;
        }
      }
      RJ_AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                         amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                         plan.resources.vgprs / granule - 1);
      if (arch_ == rocjitsu::IsaFamily::Cdna3 || arch_ == rocjitsu::IsaFamily::Cdna4) {
        RJ_AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                           amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                           (plan.resources.sgprs + kCdnaSpecialSgprTailReserve + 7) / 8 - 1);
        RJ_AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3,
                           amdhsa::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET,
                           plan.resources.vgprs / 4 - 1);
      }
    }
    if (plan.private_bytes || request.register_growth || has_storage) {
      CodeObjectPatcher resources(updated);
      const std::optional<uint32_t> sgprs =
          request.register_growth ? std::optional<uint32_t>{plan.resources.sgprs} : std::nullopt;
      const std::optional<uint32_t> vgprs =
          request.register_growth ? std::optional<uint32_t>{plan.resources.vgprs} : std::nullopt;
      if (!resources.patch_kernel_resources(kernel, descriptor, sgprs, vgprs, &error)) {
        result.errors.push_back("entry resource metadata/descriptor update failed: " + error);
        return result;
      }
      image = resources.emit();
    }
  }

  const AmdGpuCodeObject final_object(image.data(), image.size());
  if (!final_object.is_valid() || final_object.text_sections().size() != 1) {
    result.errors.emplace_back("cannot reparse final whole-image resource result");
    return result;
  }
  const Section *final_text = final_object.text_sections().front();
  const std::vector<KernelDescriptorInfo> final_kernels =
      scan_kernel_descriptors(image, final_text->sectionOffset(), final_text->size());
  for (const EntryPatchPlan &plan : plans) {
    const std::vector<KernelDescriptorInfo>::const_iterator kernel = std::ranges::find(
        final_kernels, plan.kernel.kernel_name, &KernelDescriptorInfo::kernel_name);
    if (kernel == final_kernels.end()) {
      result.errors.emplace_back("final image lost a selected kernel descriptor");
      return result;
    }
  }
  result.elf_bytes = std::move(image);
  for (const EntryPatchPlan &plan : plans) {
    const KernelDescriptorInfo &kernel = *std::ranges::find(final_kernels, plan.kernel.kernel_name,
                                                            &KernelDescriptorInfo::kernel_name);
    InstrumentedEntry entry = {.original_descriptor_offset = plan.kernel.descriptor_file_offset,
                               .prepared_descriptor_offset = kernel.descriptor_file_offset,
                               .original_entry_offset = plan.kernel.entry_text_offset,
                               .prepared_entry_offset = kernel.entry_text_offset,
                               .resources = plan.resources};
    for (const AppliedSite &site : plan.applied) {
      const InstrumentedSite production = {
          .kind = site.site->kind,
          .original_offset = site.site->anchor_offset,
          .original_size = site.site->original_size,
          .guest_offset = site.trampoline_offset + site.bytes.original_byte_offset,
          .guest_size =
              site.site->kind == InstrumentationKind::KernelEntry ? 0u : site.site->original_size,
          .instrumentation_offset = site.trampoline_offset +
                                    (site.is_probe_call ? site.bytes.call_envelope_byte_offset : 0),
          .instrumentation_size = site.is_probe_call ? site.bytes.call_envelope_byte_size
                                                     : site.bytes.original_byte_offset};
      entry.sites.push_back(production);
      result.sites.push_back(production);
      InstrumentationPatch patch = {.anchor_offset = site.site->anchor_offset,
                                    .original_size = site.site->original_size,
                                    .trampoline_offset = site.trampoline_offset,
                                    .return_target = site.return_target,
                                    .original_bytes = site.site->original_bytes,
                                    .patched_anchor_bytes = site.bytes.patched_anchor_bytes,
                                    .is_probe_call = site.is_probe_call,
                                    .probe_symbol = {}};
      if (site.is_probe_call) {
        patch.probe_symbol = plan.resolved.probes[*site.site->probe_index].symbol;
        patch.probe_target_offset = site.probe_target_offset;
        patch.link_pair_base = site.link_pair_base;
        patch.target_pair_base = site.target_pair_base;
      }
      result.patches.push_back(std::move(patch));
    }
    result.entries.push_back(std::move(entry));
  }
  return result;
}

Instrumentor::ValidationResult Instrumentor::validate_points() {
  if (!options_.entries.empty()) {
    ValidationResult result;
    entry_validation_ = &result;
    const InstrumentedCodeObjectDebug validation = patch_entries();
    entry_validation_ = nullptr;
    result.errors = validation.errors;
    return result;
  }
  ResolvedPoints resolved = resolve_points();
  ValidationResult result;
  result.errors = std::move(resolved.errors);
  result.edit_error = std::move(resolved.edit_error);
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
  if (!options_.entries.empty())
    return patch_entries();

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
    result.edit_error = std::move(resolved.edit_error);
    return result;
  }

  // Construct the patcher and preflight builder output before mutating it.
  // Each trampoline is appended directly after the original .text bytes as a
  // local code cave (as in the current DBT design).
  CodeObjectPatcher patcher(obj_);
  const uint64_t cave_start = planning_cave_offset_.value_or(patcher.text_size());
  const std::vector<KernelDescriptorInfo> &kernels = resolved.kernels;

  // Reuse the bank proof from resolution when present, and otherwise build
  // liveness only for probe calls or register growth.
  // resolve_points() built blocks_ and resolved those probes only because the
  // kernels share a wave size. blocks_ owns the Instructions live_before() is
  // keyed on; their encoding bytes belong to obj_ or the owned analysis_object_.
  // Explicit-entry planning scopes analysis to the selected kernel's reachable
  // blocks. Legacy requests retain the existing whole-image analysis scope,
  // including new edges introduced by replacement of a terminator.
  std::vector<BasicBlock *> liveness_scope;
  if (planning_descriptor_offset_ && kernels.size() == 1) {
    const BlockOffsetIndex offsets = build_block_offset_index(blocks_);
    const BlockPositionIndex positions = build_block_position_index(blocks_);
    BasicBlock *entry = block_for_offset(offsets, kernels.front().entry_text_offset);
    if (entry == nullptr || entry->start_offset() != kernels.front().entry_text_offset) {
      result.errors.emplace_back("selected entry has no decoded block");
      return result;
    }
    KernelScopeSpec spec;
    spec.own_entries.insert(kernels.front().entry_text_offset);
    liveness_scope = reachable_kernel_blocks(blocks_, offsets, positions, *entry, spec);
  } else {
    liveness_scope.reserve(blocks_.size());
    for (const std::unique_ptr<BasicBlock> &block : blocks_)
      liveness_scope.push_back(block.get());
  }
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
  // MODE safety at the callback does not enumerate later indexed guest reads.
  // Require that independent proof before AfterInst relies on physical liveness.
  if (std::ranges::any_of(resolved.sites, [](const ResolvedInstrumentationSite &site) {
        return site.kind == InstrumentationKind::AfterInst;
      })) {
    const util::FailureOr<bool> complete = liveness_->global_vgpr_usage_is_complete();
    if (complete.failed() || !complete.value()) {
      result.errors.emplace_back("AfterInst requires complete physical VGPR usage; relative or "
                                 "GPR-indexed accesses are unsupported");
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
  // Keep source bounds and growth reservations tied to the original allocation.
  // Per-site preservation instead follows observation reads in the effective CFG.
  RegisterSet observed_vgprs;
  const uint32_t source_vgpr_bound = std::min(vgpr_bounds.ordinary_bound, 256u);
  for (const ResolvedInstrumentationSite &site : sites) {
    if (!site.is_probe_call())
      continue;
    for (const ProbeArgValue &arg : site.probe_args) {
      if (arg.source == ProbeArgSource::AnchorVgpr && arg.register_index >= source_vgpr_bound) {
        result.errors.emplace_back(
            "guest VGPR argument exceeds the kernel's ordinary VGPR allocation");
        return result;
      }
    }
    observed_vgprs |= probe_argument_registers(site.probe_args);
  }
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
  RegisterSet guest_registers;
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
    for (const BasicBlock *block : liveness_scope)
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
    // Inputs may occupy physical padding beyond the original metadata counts.
    // Keep growth's backup destinations above those observed registers too.
    include_registers(observed_vgprs);
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
      for (const EntryRegisterCapture &capture : entry_patch->captured_registers)
        sgpr_save_base = std::max(sgpr_save_base, uint32_t{capture.sgpr} + 1);
    }
    cave_cursor = entry_patch->stub_offset + entry_patch->words.size() * sizeof(uint32_t);
  }

  // Register spilling targets a single-kernel code object for now: any site that
  // must spill grows that one kernel's per-lane scratch. The SpillManager is
  // created lazily on the first spilling site and shared by the rest; its final
  // size is written back to the descriptor after all sites succeed.
  std::optional<SpillManager> spills;

  std::unordered_map<uint64_t, std::vector<uint32_t>> cdna3_owners;
  if (arch_ == rocjitsu::IsaFamily::Cdna3 && kernels.size() > 1 && !resolved.probes.empty())
    cdna3_owners =
        cdna3_probe_sgpr_counts(blocks_, kernels, sites, obj_.text_sections().front()->size());

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

      const uint32_t default_count = kernel_sgpr_count.value_or(0);
      std::span<const uint32_t> allocations(&default_count, 1);
      if (!cdna3_owners.empty())
        allocations = cdna3_owners.at(site.anchor_offset);
      const uint32_t site_sgpr_count = std::ranges::min(allocations);
      // The kernel must own an ordinary pair throughout the call. On gfx942 an
      // in-bounds pair can still be the physical backing of VCC or FLAT_SCRATCH.
      if (is_valid_probe_abi(probe.abi)) {
        const uint16_t link_base = probe.abi.link_pair_base;
        if (!probe_link_pair_fits_in_kernel(site_sgpr_count, link_base)) {
          result.errors.push_back(
              "probe call needs the return-link pair s[" + std::to_string(link_base) + ":" +
              std::to_string(link_base + 1) + "] but the kernel allocates only " +
              std::to_string(site_sgpr_count) + " SGPRs; rebuild the kernel with at least " +
              std::to_string(link_base + 2) + " SGPRs");
          continue;
        }
        const auto alias = std::ranges::find_if(allocations, [&](uint32_t count) {
          return arch_ == rocjitsu::IsaFamily::Cdna3 &&
                 cdna3::sgpr_pair_overlaps_vcc_or_flat_scratch(count, link_base);
        });
        if (alias != allocations.end()) {
          result.errors.push_back("probe return-link pair s[" + std::to_string(link_base) + ":" +
                                  std::to_string(link_base + 1) +
                                  "] overlaps VCC or FLAT_SCRATCH in the kernel's " +
                                  std::to_string(*alias) + "-SGPR allocation");
          continue;
        }
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
      if (!check_probe_special_state(*summary, arch_, allocations, &err)) {
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
      plan.kernel_sgpr_count = site_sgpr_count;
      plan.kernel_vgpr_count = vgpr_bounds.ordinary_bound;
      plan.probe_args = site.probe_args;
      plan.force_full_exec = probe.force_full_exec;
      // Set on every site of a kernel that has a prologue, not only the ones
      // passing the pointer on: the pair has to reach them all, so the envelope
      // has to stay off it everywhere.
      plan.entry_storage_base = entry_storage_base;
      if (entry_patch)
        plan.entry_register_captures = entry_patch->captured_registers;
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
      // Sites without a probe need no call-envelope registers or spills.
    } else {
      std::optional<TrampolinePlan> maybe_plan =
          make_trampoline_plan(site, arch_, trampoline_offset, &err);
      if (!maybe_plan) {
        result.errors.push_back(std::move(err));
        continue;
      }
      const TrampolinePlan &plan = *maybe_plan;
      // Observation retains its inline-nop guard. Replacement bodies have
      // already passed concrete decoding and PC-independence admission.
      if (site.kind != InstrumentationKind::ReplaceWithInst &&
          !validate_inline_nop_plan(plan, &err)) {
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

  if (entry_plan_ != nullptr) {
    entry_plan_->kernel = kernels.front();
    entry_plan_->resources = result.resources;
    entry_plan_->guest_registers = guest_registers;
    if (spills)
      entry_plan_->private_bytes = spills->total_private_bytes();
    entry_plan_->prologue = std::move(entry_patch);
    entry_plan_->cave_end = cave_cursor;
    entry_plan_->resolved = std::move(resolved);
    entry_plan_->applied = std::move(applied);
    return result;
  }

  // Redirect at the original descriptor offset; replace_text then shifts the
  // entry field with everything else. Resource sizes are committed together
  // with their metadata after text replacement.
  if (entry_patch &&
      !patcher.redirect_kernel_entry(kernels.front().descriptor_file_offset,
                                     kernels.front().entry_text_offset, entry_patch->stub_offset)) {
    result.errors.emplace_back("failed to redirect the kernel entry to the entry-prologue stub");
    return result;
  }

  // Per-site plans and trampoline bytes are ready. Collect original-text
  // anchor replacements for one checked-edit transaction; KernelEntry callbacks
  // retarget only the entry-stub branch. Apply checked edits before appending
  // the local cave, then let replace_text() grow .text and fix up the surrounding
  // ELF (section/segment sizes, moved symbols, descriptor entries).
  std::vector<CheckedTextEdit> anchor_edits;
  anchor_edits.reserve(applied.size());
  for (const AppliedSite &applied_site : applied) {
    if (applied_site.site->kind == InstrumentationKind::KernelEntry) {
      const uint64_t branch_pc =
          entry_patch->stub_offset + (entry_patch->words.size() - 1) * sizeof(uint32_t);
      const std::optional<int16_t> target =
          compute_sopp_branch_simm16(branch_pc, applied_site.trampoline_offset);
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
      anchor_edits.push_back({.offset = applied_site.site->anchor_offset,
                              .expected = applied_site.site->original_bytes,
                              .replacement = {applied_site.bytes.patched_anchor_bytes.begin(),
                                              applied_site.bytes.patched_anchor_bytes.begin() +
                                                  applied_site.site->original_size}});
    }
  }
  // Entry-only callbacks patch the appended stub, leaving original text intact.
  const std::span<const uint8_t> original_text = patcher.text_bytes();
  std::vector<uint8_t> new_text(original_text.begin(), original_text.end());
  if (!anchor_edits.empty()) {
    TextEditError edit_error;
    // Retain B1's guard for known conditional-debug and subvector-loop forms
    // whose missing destination metadata cannot establish reference safety.
    for (const std::unique_ptr<BasicBlock> &block : blocks_) {
      for (const Instruction &instruction : block->instructions()) {
        if (reject_unavailable_branch_reference(instruction, edit_error)) {
          result.errors.push_back("anchor edit transaction: " + edit_error.message);
          result.edit_error = std::move(edit_error);
          return result;
        }
      }
    }
    std::vector<uint64_t> kernel_entries;
    kernel_entries.reserve(kernels.size());
    for (const KernelDescriptorInfo &kernel : kernels)
      kernel_entries.push_back(kernel.entry_text_offset);
    const InstructionBoundaries &source_boundaries =
        source_facts_ ? source_facts_->boundaries : instruction_boundaries_;
    std::optional<TextEditReferences> local_references;
    if (!source_facts_) {
      FailureOr<TextEditReferences> collected = CheckedTextEdits::collect_references(
          obj_, arch_, source_boundaries, patcher, kernel_entries, edit_error);
      if (collected.failed()) {
        result.errors.push_back("anchor edit transaction: " + edit_error.message);
        result.edit_error = std::move(edit_error);
        return result;
      }
      local_references = std::move(collected).value();
    }
    const TextEditReferences &references =
        source_facts_ ? source_facts_->references : *local_references;
    const CheckedTextEdits transaction(obj_, *decoder_, source_boundaries, references);
    if (transaction.apply(anchor_edits, new_text, edit_error).failed()) {
      result.errors.emplace_back("anchor edit transaction: " + edit_error.message);
      result.edit_error = std::move(edit_error);
      return result;
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
      if (arch_ == rocjitsu::IsaFamily::Cdna3) {
        const uint32_t old_end =
            cdna3::sgpr_alias_end(*AmdGpuCodeObject::min_kernel_sgpr_count(arch_, kernels));
        const uint32_t new_end =
            cdna3::sgpr_alias_end(result.resources.sgprs + kCdnaSpecialSgprTailReserve);
        const auto uses_numeric_alias = [&](uint32_t end) {
          return guest_registers.intersects({RegClass::SGPR, cdna3::flat_scratch_base(end), 2}) ||
                 guest_registers.intersects({RegClass::SGPR, cdna3::vcc_base(end), 2});
        };
        if (old_end != new_end && (uses_numeric_alias(old_end) || uses_numeric_alias(new_end))) {
          result.errors.emplace_back(
              "register growth would change a guest numeric SGPR alias for VCC or FLAT_SCRATCH");
          return result;
        }
      }
      RJ_AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                         amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                         result.resources.vgprs / granule - 1);
      if (arch_ == rocjitsu::IsaFamily::Cdna3 || arch_ == rocjitsu::IsaFamily::Cdna4) {
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
         .original_size = a.site->original_size,
         .guest_offset = a.trampoline_offset + a.bytes.original_byte_offset,
         .guest_size = a.site->kind == InstrumentationKind::KernelEntry ? 0u
                       : a.site->kind == InstrumentationKind::ReplaceWithInst
                           ? static_cast<uint32_t>(a.site->replacement_words.size() * 4)
                           : a.site->original_size,
         .instrumentation_offset =
             a.trampoline_offset + (a.is_probe_call ? a.bytes.call_envelope_byte_offset : 0),
         .instrumentation_size =
             a.is_probe_call ? a.bytes.call_envelope_byte_size : a.bytes.original_byte_offset});
    result.patches.push_back(std::move(patch));
  }
  return result;
}

} // namespace rocjitsu
