// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/probe_live_in.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/analysis/exec_state.h" // complete type for the null ExecMaskAnalysis
#include "rocjitsu/code/analysis/liveness.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/code/kernel_scope.h"
#include "rocjitsu/code/patch/error_report.h"
#include "rocjitsu/code/patch/probe_symbol.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/target_registry.h"
#include "util/diagnostic.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace rocjitsu {

namespace {

const char *class_prefix(RegClass cls) {
  switch (cls) {
  case RegClass::SGPR:
    return "s";
  case RegClass::VGPR:
    return "v";
  case RegClass::ACC_VGPR:
    return "a";
  default:
    return "?";
  }
}

// M0 displaces the encoded SGPR index at runtime, so the operand names one
// register and the hardware reads another. `LivenessAnalysis` tracks the
// indirect *VGPR* forms (global_vgpr_usage_is_complete()) but has no scalar
// equivalent, so this scan stands in for one.
//
// Prefix rather than an enumeration: it covers the load, the store
// (s_movreld_*, whose indirect def is likewise absent from the clobber summary)
// and the RDNA s_movrelsd_2_b32 pair form. Enumerating exact mnemonics is how
// the scalar family slipped past the v_movrel* gate to begin with.
//
// Named for what it matches, not for the property it stands in for: any other
// scalar indirection would need its own scan. Decoded instruction metadata is
// the mechanism that would subsume both, per the TODO in
// may_access_vgprs_indirectly().
[[nodiscard]] bool accesses_sgprs_indirectly_via_movrel(const std::vector<BasicBlock *> &scope) {
  for (BasicBlock *block : scope) {
    if (block == nullptr)
      continue;
    for (const Instruction &inst : block->instructions()) {
      if (inst.mnemonic().starts_with("s_movrel"))
        return true;
    }
  }
  return false;
}

} // namespace

std::string format_register_set(const RegisterSet &regs) {
  std::string out;
  regs.for_each([&](RegisterRef ref) {
    if (!out.empty())
      out += ", ";
    out += class_prefix(ref.cls);
    out += std::to_string(ref.index);
  });
  return out;
}

std::optional<RegisterSet> analyze_probe_live_ins(const AmdGpuCodeObject &probe_obj,
                                                  const ResolvedProbeSymbol &sym,
                                                  rj_code_arch_t arch, ProbeCallingConvention cc,
                                                  std::string *error_out) {
  const std::optional<uint16_t> link_pair = link_pair_for(cc);
  if (!link_pair) {
    report(error_out, "probe calling convention is not recognized, cannot analyze live-ins");
    return std::nullopt;
  }

  // Resolving a gfx1250 vector operand needs a block where MODE.VGPR_MSB is
  // known zero (LivenessAnalysisOptions::entry_block). A trampoline anchor
  // carries no such guarantee and the call envelope sets no mode.
  if (arch == ROCJITSU_CODE_ARCH_CDNA5) {
    report(error_out, "gfx1250 probe bodies are not analyzable: MODE.VGPR_MSB at the "
                      "instrumentation site is unknown, so encoded vector operands do not "
                      "identify physical VGPRs");
    return std::nullopt;
  }

  // BasicBlock::build() decodes only text_sections() and restarts its byte
  // offsets per section, so a body outside ".text" is never decoded and with
  // more than one ".text" a block offset names no unique body.
  if (probe_obj.text_sections().size() != 1) {
    report(error_out, "probe code object must have exactly one .text section for CFG analysis");
    return std::nullopt;
  }
  const Section *text = probe_obj.text_sections().front();
  const std::optional<size_t> text_index = text->sectionHeaderIndex();
  if (!text_index || *text_index != sym.section_index) {
    report(error_out, "probe body is not in the code object's .text section, so it is not decoded "
                      "for CFG analysis");
    return std::nullopt;
  }
  if (sym.body_file_offset < text->sectionOffset()) {
    report(error_out, "probe body starts before its .text section");
    return std::nullopt;
  }
  const uint64_t entry_text_offset = sym.body_file_offset - text->sectionOffset();

  const auto &registry = default_isa_target_registry();
  const rj_code_target_id_t target = probe_obj.target_id();
  auto decoder = target == ROCJITSU_CODE_TARGET_INVALID ? Decoder::create(arch)
                                                        : Decoder::create(registry, target);
  if (decoder == nullptr) {
    report(error_out, "no decoder for probe architecture");
    return std::nullopt;
  }

  // Split point, not extra leader: extra_leaders only carries external-entry
  // meaning under ExplicitOnly, which would oblige us to enumerate every
  // externally reachable entry in the object.
  const uint64_t split_points[] = {entry_text_offset};
  util::StringDiagnostic decode_error;
  auto built = BasicBlock::build(probe_obj, *decoder, arch, decode_error.emitter(), {},
                                 ExternalEntryPolicy::InferPredecessorless, split_points);
  if (built.failed()) {
    report(error_out, ("probe code object failed to decode: " + decode_error.message()).c_str());
    return std::nullopt;
  }
  const std::vector<std::unique_ptr<BasicBlock>> blocks = std::move(built).value();

  const BlockOffsetIndex block_index = build_block_offset_index(blocks);
  const BlockPositionIndex block_positions = build_block_position_index(blocks);
  BasicBlock *entry = block_for_offset(block_index, entry_text_offset);
  if (entry == nullptr || entry->start_offset() != entry_text_offset) {
    report(error_out, "probe entry offset does not start a decoded basic block");
    return std::nullopt;
  }

  // Stop the walk at any hardware kernel entry so a probe that happens to sit
  // in an object alongside kernels never absorbs one of their bodies.
  KernelScopeSpec spec;
  const std::span<const uint8_t> image(reinterpret_cast<const uint8_t *>(probe_obj.image_data()),
                                       probe_obj.image_size());
  for (const KernelDescriptorInfo &kd :
       scan_kernel_descriptors(image, text->sectionOffset(), text->size(), text_index))
    spec.kernel_entries.insert(kd.entry_text_offset);
  spec.own_entries.insert(entry_text_offset);

  const std::vector<BasicBlock *> scope =
      reachable_kernel_blocks(blocks, block_index, block_positions, *entry, spec);

  LivenessAnalysisOptions options;
  options.arch = arch;
  options.entry_block = entry;
  options.text =
      std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(text->data()), text->size());
  // Otherwise no EXEC-masked def is ever a kill and every VGPR the probe reads
  // stays live-in, including ones it defined itself. Obligation in the option's doc.
  options.exec_masked_defs_kill = true;
  // Only the entry block's live-in is needed, so skip materializing
  // per-instruction snapshots for the whole scope.
  options.restrict_live_before_to_instructions = true;
  const LivenessAnalysis liveness{KernelBlockScope(scope), nullptr, options};

  // Relative and GPR-indexed access displace an encoded VGPR index at runtime,
  // so the decoded operands no longer name every register the body reads. That
  // loses live-ins rather than inventing them -- the direction that lets a bad
  // probe through.
  if (!liveness.global_vgpr_usage_is_complete()) {
    report(error_out, "probe body has relative or GPR-indexed VGPR access, so its register reads "
                      "cannot be enumerated");
    return std::nullopt;
  }
  if (accesses_sgprs_indirectly_via_movrel(scope)) {
    report(error_out, "probe body has relative SGPR access, so its register reads cannot be "
                      "enumerated");
    return std::nullopt;
  }

  // The link pair is a genuine live-in -- the closing s_setpc_b64 reads it and
  // nothing writes it -- but the trampoline supplies it.
  RegisterSet live_in = liveness.block_liveness(*entry).live_in;
  live_in.erase(RegisterRef{RegClass::SGPR, *link_pair, 2});
  return live_in;
}

} // namespace rocjitsu
