// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/probe_clobber.h"

#include "rocjitsu/analysis/def_use_chain.h"
#include "rocjitsu/code/patch/error_report.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/isa/target_registry.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
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

// Special-state detection by operand display name.
//
// The generated Operand::to_register_ref() currently returns nullopt for the
// special-state operand forms this gate exists to catch (OPR_SDST_EXEC,
// OPR_SDST_M0, OPR_FLAT_SCRATCH, OPR_VCC, OPR_SSRC_SPECIAL_SCC), so the
// ref-based path below is blind to them even though the operand itself is
// present. We map the display name ("exec_lo", "m0", "vcc", ...), until
// operands expose their special RegClass structurally.
//
// NOTE: this sees any write the decoder exposes as a named operand, including
// special-state results some ops write to a named "vcc"/"exec" dst. A truly
// operand-less def (e.g. SCC from scalar ALU) stays invisible; SCC is covered by
// the envelope's save/restore, and EXEC/VCC by the trampoline's unconditional
// preserve.
//
// Matching is case-insensitive: operand display names are lowercase on some
// arches (CDNA2) but uppercase on others (CDNA4: "EXEC_LO", "M0", ...).
// TODO: drop this name fallback once the operand-type modeling work lands.
std::optional<RegClass> special_class_from_name(std::string_view name) {
  std::string lower(name);
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  const std::string_view n = lower;
  auto starts_with = [&](std::string_view prefix) { return n.substr(0, prefix.size()) == prefix; };
  if (starts_with("exec"))
    return RegClass::EXEC; // exec, exec_lo, exec_hi
  if (starts_with("vcc"))
    return RegClass::VCC; // vcc, vcc_lo, vcc_hi
  if (starts_with("flat_scratch"))
    return RegClass::FLAT_SCRATCH; // flat_scratch_lo/hi/all
  if (n == "m0")
    return RegClass::M0;
  if (n == "scc" || n == "src_scc")
    return RegClass::SCC;
  return std::nullopt;
}

// Flag the special-state class an operand names. Prefers the precise ref-based
// class; falls back to the display name for the special forms to_register_ref()
// does not map (see special_class_from_name).
void note_operand(const Operand &op, ProbeClobberSummary &summary) {
  if (auto ref = op.to_register_ref()) {
    note_special_state(summary, ref->cls);
    return;
  }
  if (auto cls = special_class_from_name(op.name()))
    note_special_state(summary, *cls);
}

// Scan every explicit operand and flag special-state register classes. This is
// best-effort: it sees operands the decoder exposes, not implicit side effects.
void scan_special_state(const Instruction &inst, ProbeClobberSummary &summary) {
  for (int i = 0; i < inst.num_dst_operands(); ++i) {
    const Operand *op = inst.dst_operand(i);
    if (op == nullptr)
      continue;
    note_operand(*op, summary);
  }
  for (int i = 0; i < inst.num_src_operands(); ++i) {
    const Operand *op = inst.src_operand(i);
    if (op == nullptr)
      continue;
    note_operand(*op, summary);
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

  // One extra zero word of slack so the decoder can always read a trailing
  // second word for an 8-byte instruction whose first word is the last body
  // word. build_probe_callable already verified the body decodes cleanly and is
  // not truncated; this guards the independent decode here. See probe_callable.
  const size_t num_words = callable.body_words.size();
  std::vector<uint32_t> words(num_words + 1, 0);
  std::copy(callable.body_words.begin(), callable.body_words.end(), words.begin());

  ProbeClobberSummary summary;
  // probe_callable rejects private/scratch access, so this stays false. Kept as
  // a field for summary completeness; a future analysis-derived probe may set it.
  summary.uses_private_segment = false;

  size_t w = 0;
  while (w < num_words) {
    util::StringDiagnostic decode_error;
    DecodeResult decoded = decoder->decode(&words[w], decode_error.emitter());
    if (decoded.failed()) {
      const std::string message = "failed to decode probe body at word " + std::to_string(w) +
                                  " while summarizing clobbers: " + decode_error.message();
      report(error_out, message.c_str());
      return std::nullopt;
    }
    std::unique_ptr<Instruction> inst = std::move(decoded).value();
    const int size = inst->size();
    if (size != 4 && size != 8) {
      report(error_out, "probe body instruction has an unexpected size");
      return std::nullopt;
    }
    const size_t inst_words = static_cast<size_t>(size) / sizeof(uint32_t);
    if (w + inst_words > num_words) {
      report(error_out, "probe body instruction extends past the copied body");
      return std::nullopt;
    }

    // Ordinary clobbers (SGPR/VGPR/AccVGPR), including implicit defs.
    const InstDefUse du(*inst);
    summary.ordinary_clobbers |= du.defs;

    scan_special_state(*inst, summary);

    w += inst_words;
  }

  return summary;
}

} // namespace rocjitsu
