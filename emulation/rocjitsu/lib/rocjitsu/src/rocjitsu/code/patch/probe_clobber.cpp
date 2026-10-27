// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/probe_clobber.h"

#include "rocjitsu/code/analysis/def_use_chain.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/error_report.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/isa/target_registry.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace rocjitsu {

namespace {

// Record an explicit operand's special-state class on the summary. Ordinary
// classes (SGPR/VGPR/AccVGPR) are handled precisely via InstDefUse and ignored
// here.
void note_special_state(ProbeClobberSummary &summary, RegClass cls) {
  switch (cls) {
  case RegClass::EXEC:
    summary.touches_exec = true;
    break;
  case RegClass::VCC:
    summary.touches_vcc = true;
    break;
  case RegClass::SCC:
    summary.touches_scc = true;
    break;
  case RegClass::M0:
    summary.touches_m0 = true;
    break;
  case RegClass::FLAT_SCRATCH:
    summary.touches_flat_scratch = true;
    break;
  default:
    break;
  }
}

// Generated operands expose both selector-encoded references and fieldless
// architectural effects. An unclassified destination invalidates no-write
// proofs; an absence in the best-effort access summary never grants one.
bool note_operand(const Operand &op, ProbeClobberSummary &summary, bool destination) {
  const std::optional<RegisterRef> ref = op.to_register_ref();
  const std::optional<RegClass> cls = ref ? std::optional(ref->cls) : op.to_special_reg_class();
  if (!cls)
    return !op.is_writable();
  note_special_state(summary, *cls);
  if (destination && *cls == RegClass::EXEC)
    summary.exec_write = ProbeWriteProof::MayWrite;
  if (destination && *cls == RegClass::VCC)
    summary.vcc_write = ProbeWriteProof::MayWrite;
  return true;
}

void require_preservation(ProbeWriteProof &proof) {
  if (proof == ProbeWriteProof::NoWrite)
    proof = ProbeWriteProof::Unknown;
}

// Scan every explicit operand and flag special-state register classes. This is
// best-effort: it sees operands the decoder exposes, not implicit side effects.
void scan_special_state(const Instruction &inst, rj_code_arch_t arch,
                        ProbeClobberSummary &summary) {
  for (int i = 0; i < inst.num_dst_operands(); ++i) {
    const Operand *op = inst.dst_operand(i);
    if (op == nullptr) {
      require_preservation(summary.exec_write);
      require_preservation(summary.vcc_write);
      continue;
    }
    // A tuple beginning in ordinary SGPRs can extend across a special selector.
    // Decoder alignment alone does not prove the whole extent is ordinary.
    const std::optional<RegisterRef> ref = op->to_register_ref();
    if (ref && ref->cls == RegClass::SGPR && op->has_register_selector()) {
      const uint32_t begin = static_cast<uint32_t>(op->encoding_value());
      const uint32_t end = begin + ref->width;
      const uint32_t exec = scalar_operand_exec_lo(arch);
      const uint32_t vcc = scalar_operand_vcc_lo(arch);
      if (begin < exec + 2 && end > exec)
        summary.exec_write = ProbeWriteProof::MayWrite;
      if (begin < vcc + 2 && end > vcc)
        summary.vcc_write = ProbeWriteProof::MayWrite;
    }
    if (!note_operand(*op, summary, /*destination=*/true)) {
      require_preservation(summary.exec_write);
      require_preservation(summary.vcc_write);
    }
  }
  for (int i = 0; i < inst.num_src_operands(); ++i) {
    const Operand *op = inst.src_operand(i);
    if (op == nullptr)
      continue;
    note_operand(*op, summary, /*destination=*/false);
  }
}

} // namespace

std::optional<ProbeClobberSummary> build_probe_clobber_summary(const ProbeCallable &callable,
                                                               std::string *error_out) {
  const auto &registry = default_isa_target_registry();
  if (callable.target != ROCJITSU_CODE_TARGET_INVALID) {
    const IsaTargetDescriptor *descriptor = registry.find(callable.target);
    if (descriptor == nullptr || descriptor->architecture_id != callable.arch) {
      report(error_out, "probe target does not match the probe architecture");
      return std::nullopt;
    }
  }
  auto decoder = callable.target == ROCJITSU_CODE_TARGET_INVALID
                     ? Decoder::create(callable.arch)
                     : Decoder::create(registry, callable.target);
  if (!decoder) {
    report(error_out, "no decoder available for the probe architecture");
    return std::nullopt;
  }

  const std::span<const uint32_t> body(callable.body_words);

  ProbeClobberSummary summary;
  // probe_callable rejects private/scratch access, so this stays false. Kept as
  // a field for summary completeness; a future analysis-derived probe may set it.
  summary.uses_private_segment = false;

  bool self_contained = callable.target != ROCJITSU_CODE_TARGET_INVALID && !body.empty() &&
                        is_valid_probe_abi(callable.abi);
  summary.exec_write = ProbeWriteProof::NoWrite;
  summary.vcc_write = ProbeWriteProof::NoWrite;
  std::vector<int64_t> instruction_offsets;
  std::vector<int64_t> branch_targets;

  size_t w = 0;
  while (w < body.size()) {
    util::StringDiagnostic decode_error;
    // Bounded by the words left in the body, so a truncated trailing
    // instruction fails to decode rather than reading past the copy. See
    // build_probe_callable, which applies the same bound.
    DecodeResult decoded =
        decoder->decode_window(body.subspan(w), w * sizeof(uint32_t), decode_error.emitter());
    if (decoded.failed()) {
      const std::string message = "failed to decode probe body at word " + std::to_string(w) +
                                  " while summarizing clobbers: " + decode_error.message();
      report(error_out, message.c_str());
      return std::nullopt;
    }
    std::unique_ptr<Instruction> inst = std::move(decoded).value();
    const size_t inst_words = static_cast<size_t>(inst->size()) / sizeof(uint32_t);

    // Ordinary clobbers (SGPR/VGPR/AccVGPR), including implicit defs. Special
    // singletons in du.defs are projected out — they are summarized separately
    // via scan_special_state below and must not enter ordinary_clobbers.
    const InstDefUse du(*inst);
    summary.ordinary_clobbers |= du.defs.ordinary_only();

    scan_special_state(*inst, callable.arch, summary);

    if ((inst->flags() & EXEC_WRITE_EFFECTS_KNOWN) == 0)
      require_preservation(summary.exec_write);
    if ((inst->flags() & VCC_WRITE_EFFECTS_KNOWN) == 0)
      require_preservation(summary.vcc_write);
    if ((inst->flags() & WRITES_EXEC) != 0)
      summary.exec_write = ProbeWriteProof::MayWrite;
    if ((inst->flags() & WRITES_VCC) != 0)
      summary.vcc_write = ProbeWriteProof::MayWrite;

    const int64_t offset = static_cast<int64_t>(w * sizeof(uint32_t));
    instruction_offsets.push_back(offset);
    const bool final_instruction = w + inst_words == body.size();
    const Operand *source = inst->src_operand(0);
    const std::optional<RegisterRef> link = source ? source->to_register_ref() : std::nullopt;
    const bool verified_return = (inst->flags() & ABSOLUTE_INDIRECT_BRANCH) != 0 && link &&
                                 link->cls == RegClass::SGPR &&
                                 link->index == callable.abi.link_pair_base && link->width == 2;
    if (final_instruction) {
      self_contained &= verified_return;
    } else if (inst->is_branch()) {
      const std::optional<int64_t> delta = inst->branch_offset_bytes();
      if (delta)
        branch_targets.push_back(offset + inst->size() + *delta);
      else
        self_contained = false;
    } else if ((inst->flags() & (INDIRECT_BRANCH | INDIRECT_CALL | PROGRAM_TERMINATOR)) != 0) {
      self_contained = false;
    }

    w += inst_words;
  }

  // Direct branches must stay within the copied body and land on instruction
  // boundaries, never on a literal/extension word. Calls/unknown indirect edges
  // remain conservative. Inspect even unreachable instructions; no CFG reachability
  // or special-register liveness assumption is needed to prove absence of writes.
  for (int64_t target : branch_targets)
    self_contained &= std::ranges::binary_search(instruction_offsets, target);
  if (summary.ordinary_clobbers.contains(
          RegisterRef{RegClass::SGPR, callable.abi.link_pair_base, 1}) ||
      summary.ordinary_clobbers.contains(
          RegisterRef{RegClass::SGPR, static_cast<uint16_t>(callable.abi.link_pair_base + 1), 1}))
    self_contained = false;
  if (!self_contained) {
    require_preservation(summary.exec_write);
    require_preservation(summary.vcc_write);
  }

  return summary;
}

} // namespace rocjitsu
