// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mutation.h"

#include "effects.h"
#include "json.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "util/diagnostic.h"

namespace rocjitsu::isa_diff {

std::optional<RegionPair> make_synthetic_mutation(const RegionPair &base, const MutationSpec &spec,
                                                  std::string &error) {
  RegionPair r = base;
  RegionSide &side = spec.new_side ? r.new_side : r.old_side;
  if (spec.inst >= side.insts.size() || spec.word >= side.insts[spec.inst].words.size()) {
    error = "mutation target out of range";
    return std::nullopt;
  }
  if (spec.xor_mask == 0) {
    error = "mutation mask is zero";
    return std::nullopt;
  }
  side.insts[spec.inst].words[spec.word] ^= spec.xor_mask;

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  if (!decoder) {
    error = "no gfx950 decoder";
    return std::nullopt;
  }
  LocSet writes, wide;
  for (auto &inst : side.insts) {
    if (inst.inline_asm >= 0) {
      const Effects fx = classify_inline_asm(inst.asm_text);
      writes |= fx.writes;
      continue;
    }
    std::vector<uint32_t> window = inst.words;
    window.resize(window.size() + 3, 0);
    util::StringDiagnostic diag;
    auto decoded = decoder->decode_window(window, inst.pc, diag.emitter());
    if (decoded.failed()) {
      error = "mutated region does not decode: " + diag.message();
      return std::nullopt;
    }
    const Instruction &d = *decoded.value();
    if (static_cast<size_t>(d.size()) != inst.words.size() * 4) {
      error = "mutation changes the instruction size";
      return std::nullopt;
    }
    const Effects fx = classify(d);
    if (!fx.p0) {
      error = "mutated instruction is outside P0: " + fx.reject_reason;
      return std::nullopt;
    }
    inst.text = d.disassemble();
    writes |= fx.writes;
    wide |= fx.wide_reads;
  }
  (spec.new_side ? r.contract.declared_writes_new : r.contract.declared_writes_old) = writes;
  r.contract.wide_reads |= wide;
  r.synthetic = true;
  r.id = base.id + "-syn-" + spec.label;
  r.contract.sources.push_back(
      {std::string("synthetic mutation on ") + (spec.new_side ? "new" : "old") + " inst " +
           std::to_string(spec.inst) + " word " + std::to_string(spec.word) + " xor " +
           hex32(spec.xor_mask) + " -> '" + side.insts[spec.inst].text + "'",
       "synthetic_injection"});
  r.prepared_key = compute_prepared_key(r);
  return r;
}

} // namespace rocjitsu::isa_diff
