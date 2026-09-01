// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/dbt/binary_translator.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/analysis/def_use_chain.h"
#include "rocjitsu/code/analysis/exec_state.h"
#include "rocjitsu/code/analysis/gfx1250_vgpr_msb.h"
#include "rocjitsu/code/analysis/liveness.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/dbt/binary_translator_internal.h"
#include "rocjitsu/code/dbt/generated/encoding_cdna4_to_cdna3.h"
#include "rocjitsu/code/dbt/generated/encoding_cdna4_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/encoding_cdna4_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna4_to_cdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna4_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna4_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_types.h"
#include "rocjitsu/code/dbt/kernel_descriptor_translator.h"
#include "rocjitsu/code/dbt/lds_virtualization.h"
#include "rocjitsu/code/dbt/legalization/gfx1250_b0_to_a0.h"
#include "rocjitsu/code/dbt/scoped_cfg_edges.h"
#include "rocjitsu/code/dbt/semantic_translator.h"
#include "rocjitsu/code/dbt/virtual_lds.h"
#include "rocjitsu/code/kernel_scope.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/kernarg_extension.h"
#include "rocjitsu/code/patch/kernel_text_layout.h"
#include "rocjitsu/code/patch/sidecar_metadata.h"
#include "rocjitsu/code/relocation_function_table.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/isa_traits.h"
#include "rocjitsu/isa/target_registry.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <deque>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocjitsu {

namespace {

/// @brief Normalize legacy scaled-WMMA prefixes for the common decoder.
///
/// @details Legacy source objects may populate the architecturally unused prefix SRC2 field
/// with a noncanonical selector (the offline oracle uses inline zero, 0x080), while the ISA
/// requires that field to encode VGPR0 (0x100). Normalize it inside the revision-specific
/// translation path so translated objects remain ISA-canonical; the architecture decoder
/// separately accepts LLVM's exact compatibility encoding without weakening other fixed-field
/// checks.
class Gfx1250B0ToA0CanonicalizingDecoder final : public Decoder {
public:
  explicit Gfx1250B0ToA0CanonicalizingDecoder(std::unique_ptr<Decoder> decoder)
      : decoder_(std::move(decoder)) {
    assert(decoder_ != nullptr);
    assert(decoder_->max_instruction_words() <= kLookaheadWords);
  }

  DecodeResult decode(const rj_code_binary_inst_t *inst,
                      const DecodeErrorEmitter &emit_error) override {
    const uint32_t prefix_op = (inst[0] >> 16) & 0xffu;
    if ((inst[0] >> 24) != 0xccu || (prefix_op != 0x35u && prefix_op != 0x3au))
      return decoder_->decode(inst, emit_error);

    std::array<rj_code_binary_inst_t, kLookaheadWords> canonical{};
    std::copy_n(inst, kLookaheadWords, canonical.begin());
    constexpr uint32_t kSrc2Mask = 0x1ffu << 18;
    canonical[1] = (canonical[1] & ~kSrc2Mask) | (0x100u << 18);
    return decoder_->decode(canonical.data(), emit_error);
  }

  std::size_t max_instruction_words() const override { return decoder_->max_instruction_words(); }

private:
  static constexpr std::size_t kLookaheadWords = 4;
  std::unique_ptr<Decoder> decoder_;
};

EncodingTranslateFn select_encoding_translator(rj_code_arch_t guest, rj_code_arch_t host) {
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA4)
    return cdna4_to_rdna4::translate_encoding_cdna4_to_rdna4;
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_CDNA3)
    return cdna4_to_cdna3::translate_encoding_cdna4_to_cdna3;
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA3)
    return cdna4_to_rdna3::translate_encoding_cdna4_to_rdna3;
  return nullptr;
}

LegalizationLookupFn select_legalization(rj_code_arch_t guest, rj_code_arch_t host) {
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA4) {
    return [](uint16_t enc_id, uint16_t opcode) -> const InstructionLegalization * {
      return lookup(kLegalization_cdna4_to_rdna4, enc_id, opcode);
    };
  }
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_CDNA3) {
    return [](uint16_t enc_id, uint16_t opcode) -> const InstructionLegalization * {
      return lookup(kLegalization_cdna4_to_cdna3, enc_id, opcode);
    };
  }
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA3) {
    return [](uint16_t enc_id, uint16_t opcode) -> const InstructionLegalization * {
      return lookup(kLegalization_cdna4_to_rdna3, enc_id, opcode);
    };
  }
  return nullptr;
}

[[nodiscard]] std::vector<uint32_t> raw_words_for_inst(const Instruction &inst) {
  const uint32_t *raw = inst.raw_encoding();
  if (!raw)
    return {};
  return {raw, raw + inst.size() / sizeof(uint32_t)};
}

[[nodiscard]] uint32_t text_word_at(std::span<const uint8_t> text, uint64_t offset) {
  uint32_t word = 0;
  if (offset + sizeof(word) <= text.size())
    std::memcpy(&word, text.data() + offset, sizeof(word));
  return word;
}

/// @brief Whether only `s_nop 0` padding separates a builder's last word from its consumer.
///
/// @details patch_recovered_builder_fixups() regenerates a builder into the canonical getpc-delta
/// form and NOP-fills the remainder of the source range, and that form is often one word shorter
/// than the range it replaces. So a builder that ran flush into its consumer before translation
/// runs into a nop after it. Nothing reads those words and the recovery chain crosses them
/// unchanged, but an exact-adjacency test does not -- which would make the next pass classify the
/// same site differently and wrap a consumer this pass deliberately left dynamic, re-growing the
/// body it had just kept flat. Treat the padding as part of the run.
[[nodiscard]] bool builder_runs_into_consumer(std::span<const uint8_t> text, uint64_t recovery_end,
                                              uint64_t call_offset, rj_code_arch_t arch) {
  if (recovery_end > call_offset || (call_offset - recovery_end) % sizeof(uint32_t) != 0)
    return false;
  const uint32_t nop = build_s_nop(0, arch);
  for (uint64_t offset = recovery_end; offset < call_offset; offset += sizeof(uint32_t)) {
    if (text_word_at(text, offset) != nop)
      return false;
  }
  return true;
}

/// @brief Indirect transfers in one scope whose target pair was loaded from the kernarg segment.
///
/// @details A target this object never computed and never stored still has a provenance when it
/// arrives through kernarg: the host wrote it, and the only `.text` address a host can hold is one
/// it read from a symbol, which relocate_text_symbols() rewrites. That is a positive fact about
/// this operand. It is deliberately not the weaker claim that the object produces no code address
/// at all -- an undefined live-in also satisfies that, and it has no supplier whatsoever, so
/// admitting it would accept on absence of evidence.
///
/// A meet-over-predecessors fixpoint, not a linear scan. A block reached from a preheader where
/// the pair is kernarg-loaded and from a latch where it was clobbered must not inherit the
/// preheader's answer, and the transfers this admits sit inside exactly such loops.
[[nodiscard]] std::unordered_set<uint64_t>
kernarg_supplied_indirect_targets(std::span<BasicBlock *const> blocks, const BasicBlock *entry,
                                  const KdTranslation &kd) {
  std::unordered_set<uint64_t> offsets;
  // reachable_kernel_blocks() orders by offset, so blocks.front() can be a backward-reachable
  // helper or an adopted body rather than the scope entry. Seeding that block with the ABI's
  // kernarg fact would fabricate provenance for a pair the entry never established.
  if (!kd.source_has_kernarg_segment_ptr || blocks.empty() || entry == nullptr ||
      std::ranges::find(blocks, entry) == blocks.end())
    return offsets;

  // The only fact this function can produce is attached to an indirect transfer, so a scope with
  // none has nothing to say. Checking first avoids running the fixpoint -- an InstDefUse and two
  // RegisterSet expansions per instruction, per predecessor edge, per sweep -- to build an answer
  // that is empty by construction. The consumer asks per instruction and its cheaper disjunct is
  // usually false, so without this the fixpoint runs for effectively every scope of every object;
  // it measured as the whole of a 22% translation-time regression. An indirect transfer ends its
  // block, so the terminator is the only place it can be.
  if (std::ranges::none_of(blocks, [](const BasicBlock *block) {
        if (block == nullptr)
          return false;
        const Instruction *term = block->terminator();
        return term != nullptr && (term->flags() & (INDIRECT_BRANCH | INDIRECT_CALL)) != 0;
      }))
    return offsets;

  // Pair-low registers holding the kernarg segment pointer, and those holding a value loaded
  // through it. Tracked separately: only the first may be a load base, only the second may be a
  // transfer target.
  struct PairState {
    std::set<uint16_t> pointer;
    std::set<uint16_t> loaded;
    bool operator==(const PairState &other) const {
      return pointer == other.pointer && loaded == other.loaded;
    }
  };

  const auto pair_operand = [](const Operand *operand, uint16_t &lo, uint16_t &width) {
    if (operand == nullptr)
      return false;
    const auto ref = operand->to_register_ref();
    if (!ref || ref->cls != RegClass::SGPR)
      return false;
    lo = ref->index;
    width = ref->width;
    return true;
  };

  // Every even-aligned pair start whose whole pair the instruction defines. Driven from the
  // instruction's full def set rather than a destination operand, so a wide load clears the pairs
  // above it and a def of one half clears the pair it belongs to.
  const auto kill_defs = [&](const Instruction &inst, PairState &state) {
    if (state.pointer.empty() && state.loaded.empty())
      return;
    const InstDefUse def_use(inst);
    // Only the pairs actually being tracked can be killed, and there are a handful at most.
    // Sweeping the whole SGPR file instead cost two RegisterSet::contains() calls per register per
    // instruction per sweep, which profiled as half of all translation time.
    const auto pair_is_defined = [&](uint16_t lo) {
      const RegisterRef low{.cls = RegClass::SGPR, .index = lo, .width = 1};
      const RegisterRef high{
          .cls = RegClass::SGPR, .index = static_cast<uint16_t>(lo + 1), .width = 1};
      return def_use.defs.contains(low) || def_use.defs.contains(high);
    };
    std::erase_if(state.pointer, pair_is_defined);
    std::erase_if(state.loaded, pair_is_defined);
  };

  const auto transfer = [&](BasicBlock *block, PairState state, bool record) {
    for (const Instruction &inst : block->instructions()) {
      const uint64_t flags = inst.flags();
      const bool is_transfer = (flags & (INDIRECT_BRANCH | INDIRECT_CALL)) != 0;
      if (is_transfer && record) {
        // s_setpc takes the target as its only source; s_swappc takes the return pair first.
        for (int i = 0; i < inst.num_src_operands(); ++i) {
          uint16_t lo = 0;
          uint16_t width = 0;
          if (pair_operand(inst.src_operand(i), lo, width) && width == 2 &&
              state.loaded.contains(lo)) {
            offsets.insert(inst.src_loc());
            break;
          }
        }
      }

      uint16_t base_lo = 0;
      uint16_t base_width = 0;
      const bool loads_through_kernarg =
          inst.mnemonic().starts_with("s_load_b") && inst.num_src_operands() >= 1 &&
          pair_operand(inst.src_operand(0), base_lo, base_width) && state.pointer.contains(base_lo);
      uint16_t copy_lo = 0;
      uint16_t copy_width = 0;
      const bool copies_pointer = inst.mnemonic() == "s_mov_b64" && inst.num_src_operands() == 1 &&
                                  pair_operand(inst.src_operand(0), copy_lo, copy_width) &&
                                  copy_width == 2 && state.pointer.contains(copy_lo);

      uint16_t dst_lo = 0;
      uint16_t dst_width = 0;
      const bool has_pair_dst =
          inst.num_dst_operands() == 1 && pair_operand(inst.dst_operand(0), dst_lo, dst_width);

      kill_defs(inst, state);

      // A call clobbers everything the ABI does not preserve, direct or indirect alike.
      if ((flags & INDIRECT_CALL) != 0 || inst.mnemonic().starts_with("s_call")) {
        std::erase_if(state.pointer, [](uint16_t lo) { return !is_callee_saved_sgpr(lo); });
        std::erase_if(state.loaded, [](uint16_t lo) { return !is_callee_saved_sgpr(lo); });
      }

      if (has_pair_dst && loads_through_kernarg && dst_width >= 2) {
        // Only an even-aligned pair wholly inside the loaded range is addressable as a 64-bit
        // transfer operand, so s_load_b96 s[80:82] yields (80,81) and b128 s[80:83] adds (82,83).
        for (unsigned lo = dst_lo; lo + 1u < unsigned{dst_lo} + dst_width; lo += 2)
          state.loaded.insert(static_cast<uint16_t>(lo));
      }
      if (has_pair_dst && copies_pointer && dst_width == 2)
        state.pointer.insert(dst_lo);
    }
    return state;
  };

  const std::unordered_set<const BasicBlock *> in_scope(blocks.begin(), blocks.end());
  PairState seed;
  seed.pointer.insert(kd.kernarg_segment_ptr_sgpr);

  // A must-analysis has to start every non-entry block at TOP, not at the empty set. Starting
  // empty and intersecting is starting at BOTTOM: the meet can never grow a fact back, so a block
  // whose predecessors had not been visited yet would be pinned empty for the rest of the run and
  // the transfer it guards would never be admitted. `computed` marks which blocks hold a real
  // answer; a predecessor without one is treated as TOP and simply skipped by the meet.
  std::unordered_map<const BasicBlock *, PairState> entry_state;
  std::unordered_set<const BasicBlock *> computed;
  entry_state[entry] = seed;
  computed.insert(entry);

  // Sweep to a fixpoint rather than draining a change-driven worklist: a block popped before its
  // predecessors were known would otherwise be dropped and never requeued. The lattice is finite
  // and every step is monotone, so the sweep count is a guard, not a semantic bound.
  //
  // The guard is not a convergence bound -- this is a descending product lattice and a state can
  // keep losing facts for more sweeps than the block count -- so exhausting it means the answer is
  // still an over-approximation. Treating that as proof would admit a transfer the true fixpoint
  // would have refused, so give up and claim nothing instead.
  const size_t sweep_cap = blocks.size() + 2;
  bool converged = false;
  for (size_t sweep = 0; sweep < sweep_cap; ++sweep) {
    bool changed = false;
    for (BasicBlock *block : blocks) {
      if (block == nullptr || block == entry)
        continue;
      PairState in;
      bool seeded = false;
      for (const BasicBlock *predecessor : block->predecessors()) {
        if (!in_scope.contains(predecessor) || !computed.contains(predecessor))
          continue;
        PairState exit =
            transfer(const_cast<BasicBlock *>(predecessor), entry_state[predecessor], false);
        if (!seeded) {
          in = std::move(exit);
          seeded = true;
          continue;
        }
        std::erase_if(in.pointer, [&](uint16_t lo) { return !exit.pointer.contains(lo); });
        std::erase_if(in.loaded, [&](uint16_t lo) { return !exit.loaded.contains(lo); });
      }
      if (!seeded)
        continue;
      const auto it = entry_state.find(block);
      if (it == entry_state.end() || !(it->second == in)) {
        entry_state[block] = std::move(in);
        computed.insert(block);
        changed = true;
      }
    }
    if (!changed) {
      converged = true;
      break;
    }
  }
  if (!converged)
    return offsets;

  for (BasicBlock *block : blocks) {
    const auto it = block == nullptr ? entry_state.end() : entry_state.find(block);
    if (it != entry_state.end())
      transfer(block, it->second, true);
  }
  return offsets;
}

[[nodiscard]] std::unordered_set<uint64_t>
generated_branch_island_pool_offsets(std::span<const uint8_t> text, rj_code_arch_t arch) {
  std::unordered_set<uint64_t> offsets;
  auto decoder = Decoder::create(arch);
  if (!decoder)
    return offsets;
  const std::size_t lookahead_words = decoder->max_instruction_words();
  if (lookahead_words == 0)
    return offsets;
  std::vector<uint32_t> slot_words(lookahead_words, 0);

  const uint32_t marker = build_s_nop(kBranchIslandPoolMarkerNopImmediate, arch);
  const uint32_t skip_pool =
      build_s_branch(static_cast<int16_t>(kDirectBranchIslandPoolSlots), arch);
  constexpr uint64_t kPoolBytes = (kGeneratedIslandPoolHeaderWords + kDirectBranchIslandPoolSlots) *
                                  static_cast<uint64_t>(sizeof(uint32_t));
  for (uint64_t offset = 0;
       offset + kGeneratedIslandPoolHeaderWords * sizeof(uint32_t) <= text.size();
       offset += sizeof(uint32_t)) {
    if (text_word_at(text, offset) != marker ||
        text_word_at(text, offset + sizeof(uint32_t)) != skip_pool || text.size() < kPoolBytes ||
        offset > text.size() - kPoolBytes) {
      continue;
    }

    bool has_canonical_slots = true;
    for (uint16_t slot = 0; slot < kDirectBranchIslandPoolSlots; ++slot) {
      const uint64_t slot_offset =
          offset + (kGeneratedIslandPoolHeaderWords + slot) * sizeof(uint32_t);
      // A malformed slot can select an extension-bearing format. Pad the
      // speculative decode so pool recognition never reads beyond its input.
      slot_words[0] = text_word_at(text, slot_offset);
      DecodeResult decoded = decoder->decode(slot_words.data());
      if (decoded.succeeded()) {
        const std::unique_ptr<Instruction> &slot_inst = decoded.value();
        if (slot_inst->size() == static_cast<int>(sizeof(uint32_t)) &&
            slot_inst->mnemonic() == "s_branch" && slot_inst->branch_offset_bytes()) {
          continue;
        }
      }
      has_canonical_slots = false;
      break;
    }
    if (has_canonical_slots)
      offsets.insert(offset);
  }
  return offsets;
}

[[nodiscard]] bool words_changed(std::span<const uint32_t> before,
                                 std::span<const uint32_t> after) {
  if (before.size() != after.size())
    return true;
  return !std::ranges::equal(before, after);
}

void append_diagnostic(std::vector<TranslationDiagnostic> &diagnostics, DiagnosticSeverity severity,
                       DiagnosticKind kind, std::string message,
                       std::optional<uint64_t> guest_offset = std::nullopt,
                       std::string mnemonic = {}, std::vector<std::string> required_work = {}) {
  diagnostics.push_back({.severity = severity,
                         .kind = kind,
                         .guest_offset = guest_offset,
                         .output_offset = std::nullopt,
                         .mnemonic = std::move(mnemonic),
                         .message = std::move(message),
                         .required_work = std::move(required_work)});
}

void append_error(std::vector<TranslationDiagnostic> &diagnostics, DiagnosticKind kind,
                  std::string message, std::optional<uint64_t> guest_offset = std::nullopt,
                  std::string mnemonic = {}, std::vector<std::string> required_work = {}) {
  append_diagnostic(diagnostics, DiagnosticSeverity::Error, kind, std::move(message), guest_offset,
                    std::move(mnemonic), std::move(required_work));
}

void append_warning(std::vector<TranslationDiagnostic> &diagnostics, DiagnosticKind kind,
                    std::string message, std::optional<uint64_t> guest_offset = std::nullopt,
                    std::string mnemonic = {}, std::vector<std::string> required_work = {}) {
  append_diagnostic(diagnostics, DiagnosticSeverity::Warning, kind, std::move(message),
                    guest_offset, std::move(mnemonic), std::move(required_work));
}

void append_rewrite_discharge_error(std::vector<TranslationDiagnostic> &diagnostics,
                                    std::string message,
                                    std::optional<uint64_t> output_offset = std::nullopt,
                                    std::string mnemonic = {}) {
  diagnostics.push_back({.severity = DiagnosticSeverity::Error,
                         .kind = DiagnosticKind::ResidualRewrite,
                         .guest_offset = std::nullopt,
                         .output_offset = output_offset,
                         .mnemonic = std::move(mnemonic),
                         .message = std::move(message),
                         .required_work = {}});
}

void append_diagnostics(std::vector<TranslationDiagnostic> &dst,
                        const std::vector<TranslationDiagnostic> &src) {
  dst.insert(dst.end(), src.begin(), src.end());
}

/// @brief First condition that stops this object from being translated, if any.
///
/// @details Every result leaves the object unchanged, but not every result is a failure: an
/// object with no executable text and a matching target is a successful no-op, so the caller
/// must preserve the returned severity rather than assuming an error.
///
/// Check order is observable, because only the first match is reported. Existing checks must not
/// be permuted, and a new one should be appended unless it is deliberately more specific than a
/// check already present.
///
/// Everything this needs comes from @p patcher, including the text span, so there is no second
/// view of the same bytes to keep consistent. The ELF-header read below bounds itself rather than
/// trusting the caller's earlier size check: that check exists to gate constructing the patcher,
/// and a bound enforced hundreds of lines away is not one this function can rely on.
///
/// The decoder check stays with the caller because it gates constructing the decoder that the
/// rest of translation needs.
[[nodiscard]] std::optional<TranslationDiagnostic>
translation_refusal(const CodeObjectPatcher &patcher, rj_code_arch_t guest_arch,
                    rj_code_arch_t host_arch, uint32_t target_mach,
                    const BinaryTranslatorOptions &options) {
  const auto error = [](DiagnosticKind kind, std::string message) {
    return TranslationDiagnostic{.severity = DiagnosticSeverity::Error,
                                 .kind = kind,
                                 .guest_offset = std::nullopt,
                                 .output_offset = std::nullopt,
                                 .mnemonic = {},
                                 .message = std::move(message),
                                 .required_work = {}};
  };

  uint32_t source_mach = EF_AMDGPU_MACH_NONE;
  if (patcher.image_bytes().size() >= sizeof(Elf64_Ehdr)) {
    Elf64_Ehdr header{};
    std::memcpy(&header, patcher.image_bytes().data(), sizeof(header));
    source_mach = header.e_flags & EF_AMDGPU_MACH;
  }

  // A same-architecture gfx1250 translation is direction-specific: A0 and B0 share an ELF machine
  // ID, so both revisions must be given. Enforce this here as well as in the C API.
  if (source_mach == EF_AMDGPU_MACH_AMDGCN_GFX1250 &&
      (target_mach & EF_AMDGPU_MACH) == EF_AMDGPU_MACH_AMDGCN_GFX1250) {
    if (options.input_revision == ProcessorRevision::Unspecified ||
        options.output_revision == ProcessorRevision::Unspecified) {
      return error(DiagnosticKind::Legalization,
                   "gfx1250 same-target translation requires both input and output silicon "
                   "revisions");
    }
    // Only the B0-to-A0 direction is implemented.
    if (options.input_revision == ProcessorRevision::Gfx1250A0 &&
        options.output_revision == ProcessorRevision::Gfx1250B0)
      return error(DiagnosticKind::Legalization, "gfx1250 A0-to-B0 translation is not supported");
  }

  const uint32_t target_machine = target_mach & EF_AMDGPU_MACH;
  const IsaTargetRegistry &registry = default_isa_target_registry();
  const auto is_concrete_cdna5_target = [&](uint32_t machine) {
    const IsaGpuTargetDescription *target = registry.find_gpu_target_by_elf_machine(machine);
    const IsaTargetDescriptor *descriptor =
        target != nullptr ? registry.find(target->public_id) : nullptr;
    return descriptor != nullptr && descriptor->architecture_id == ROCJITSU_CODE_ARCH_CDNA5;
  };
  if (guest_arch == ROCJITSU_CODE_ARCH_CDNA5 && host_arch == ROCJITSU_CODE_ARCH_CDNA5 &&
      source_mach != target_machine && is_concrete_cdna5_target(source_mach) &&
      is_concrete_cdna5_target(target_machine)) {
    // Same-architecture translation otherwise copies raw encodings. Different concrete CDNA5
    // targets may have different instruction legality, and no cross-target legalization contract
    // exists.
    return error(DiagnosticKind::Legalization, "cross-target CDNA5 translation is unsupported");
  }

  if (patcher.text_bytes().empty()) {
    const std::span<const uint8_t> image = patcher.image_bytes();
    if (image.size() < sizeof(Elf64_Ehdr))
      return error(DiagnosticKind::ResourceLimit, "code object is too small to contain an ELF "
                                                  "header");
    Elf64_Ehdr header{};
    std::memcpy(&header, image.data(), sizeof(header));
    source_mach = header.e_flags & EF_AMDGPU_MACH;
    if (guest_arch == host_arch && source_mach == (target_mach & EF_AMDGPU_MACH)) {
      return TranslationDiagnostic{
          .severity = DiagnosticSeverity::Warning,
          .kind = DiagnosticKind::DataOnly,
          .guest_offset = std::nullopt,
          .output_offset = std::nullopt,
          .mnemonic = {},
          .message = "code object has no executable sections, segments, or callable symbols; "
                     "leaving unchanged",
          .required_work = {}};
    }
    return error(DiagnosticKind::ResourceLimit,
                 "code object does not expose a non-empty .text section for translation");
  }

  // DBT relocates instructions within .text (compaction, expansion, per-kernel block placement)
  // but does not rewrite relocation places that land inside .text. An in-.text relocation would
  // therefore be applied to the wrong translated bytes. Fail closed rather than silently
  // miscompile.
  if (patcher.has_relocations_within_text()) {
    return error(DiagnosticKind::Legalization,
                 "code object has a relocation place inside .text; relocating instructions would "
                 "apply it to the wrong bytes and is not supported");
  }

  // The patcher can retarget ordinary zero-addend symbol references and symbol-less RELATIVE64
  // addends through the final source-to-target offset map. Keep less explicit forms fail-closed
  // until their relocation-specific addend semantics are modeled.
  if (patcher.has_unsupported_relocation_to_text()) {
    return error(DiagnosticKind::Legalization,
                 "code object has an unsupported relocation referencing .text; section symbols, "
                 "implicit addends, and named-symbol addends cannot be remapped safely");
  }

  return std::nullopt;
}

/// @brief Return a human-readable kernel label for diagnostics.
///
/// @details Some code objects carry empty kernel symbol names. Falling back to
/// the source .text entry offset keeps skip/failure diagnostics useful for
/// debugging because the user can still identify which code-object entry failed.
[[nodiscard]] std::string kernel_label(const KdTranslation &translation) {
  if (!translation.kernel_name.empty())
    return translation.kernel_name;

  std::ostringstream os;
  os << ".text+0x" << std::hex << translation.entry_text_offset;
  return os.str();
}

[[nodiscard]] uint32_t max_descriptor_sgpr_allocation_for_long_branch(rj_code_arch_t arch) {
  // Long direct branches consume their scratch pair at the final
  // s_setpc_b64/s_swappc_b64 transfer, so DBT may only use a pair that can be
  // made descriptor-backed for the destination kernel.
  return arch_descriptor_sgpr_allocation_limit(arch);
}

/// @brief Find the next even SGPR pair that can be descriptor-backed for a branch thunk.
[[nodiscard]] std::optional<uint16_t> next_long_branch_sgpr_pair(const TranslationContext &context,
                                                                 rj_code_arch_t arch) {
  const uint32_t current = std::max(context.num_sgprs, context.required_sgpr_count);
  const uint32_t base = (current + 1u) & ~1u;
  if (base > 126)
    return std::nullopt;

  const uint32_t max_descriptor_sgprs = max_descriptor_sgpr_allocation_for_long_branch(arch);
  if (max_descriptor_sgprs != 0 && base + 2 > max_descriptor_sgprs)
    return std::nullopt;
  return static_cast<uint16_t>(base);
}

[[nodiscard]] std::vector<uint64_t> kernel_entry_offsets(std::span<const KdTranslation> kernels) {
  std::vector<uint64_t> offsets;
  offsets.reserve(kernels.size());
  for (const KdTranslation &kernel : kernels)
    offsets.push_back(kernel.entry_text_offset);

  std::ranges::sort(offsets);
  offsets.erase(std::ranges::unique(offsets).begin(), offsets.end());
  return offsets;
}

[[nodiscard]] std::vector<uint64_t>
kernel_hardware_entry_offsets(std::span<const KdTranslation> kernels) {
  std::vector<uint64_t> offsets;
  offsets.reserve(kernels.size() * 2);
  for (const KdTranslation &kernel : kernels) {
    offsets.push_back(kernel.entry_text_offset);
    if (kernel.has_kernarg_preload_firmware_skip)
      offsets.push_back(kernel.kernarg_preload_firmware_entry_text_offset);
  }

  std::ranges::sort(offsets);
  offsets.erase(std::ranges::unique(offsets).begin(), offsets.end());
  return offsets;
}

[[nodiscard]] std::vector<uint64_t> kernel_block_leaders(std::span<const KdTranslation> kernels,
                                                         std::span<const uint8_t> text) {
  std::vector<uint64_t> offsets;
  offsets.reserve(kernels.size() * 2);
  for (const KdTranslation &kernel : kernels) {
    offsets.push_back(kernel.entry_text_offset);
    // AMDHSA kernarg preloading is descriptor-controlled. When
    // kernarg_preload_spec_length is non-zero, compatible CP firmware starts at
    // KERNEL_CODE_ENTRY_BYTE_OFFSET + 256. That address is a real hardware entry,
    // not merely padding, so split a block there and seed reachability from it.
    if (kernel.has_kernarg_preload_firmware_skip &&
        kernel.kernarg_preload_firmware_entry_text_offset < text.size())
      offsets.push_back(kernel.kernarg_preload_firmware_entry_text_offset);
  }

  std::ranges::sort(offsets);
  offsets.erase(std::ranges::unique(offsets).begin(), offsets.end());
  return offsets;
}

[[nodiscard]] bool elf_image_contains(std::span<const uint8_t> image, uint64_t offset,
                                      uint64_t size) {
  const uint64_t image_size = image.size();
  return offset <= image_size && size <= image_size - offset;
}

/// @brief Collect relocation-backed entries within one executable section.
///
/// @details Kernel descriptors and relocation-backed function tables are
/// collected by their owning subsystems. This helper adds ordinary text symbols
/// demonstrably consumed through an allocated relocation and anonymous
/// R_AMDGPU_RELATIVE64 addends into .text. Symbol visibility alone is not an
/// executable-entry contract: ROCr does not promote an ordinary STT_FUNC merely
/// because it is visible and resolves STT_NOTYPE externally rather than from its
/// text st_value. The patcher intentionally does not require unreferenced
/// tooling/debug symbols to have output mappings. Keeping those symbols out of
/// this set makes reachability, relocation, and final verification agree on
/// which entries DBT preserves.
[[nodiscard]] std::optional<std::vector<uint64_t>>
text_relocation_block_leaders(std::span<const uint8_t> image, const Section &text) {
  if (!elf_image_contains(image, 0, sizeof(Elf64_Ehdr)))
    return std::nullopt;

  Elf64_Ehdr ehdr{};
  std::memcpy(&ehdr, image.data(), sizeof(ehdr));
  if (ehdr.e_shentsize != sizeof(Elf64_Shdr) ||
      !elf_image_contains(image, ehdr.e_shoff,
                          static_cast<uint64_t>(ehdr.e_shnum) * sizeof(Elf64_Shdr))) {
    return std::nullopt;
  }

  std::vector<Elf64_Shdr> shdrs(ehdr.e_shnum);
  if (!shdrs.empty())
    std::memcpy(shdrs.data(), image.data() + ehdr.e_shoff, shdrs.size() * sizeof(Elf64_Shdr));

  const auto text_index = text.sectionHeaderIndex();
  if (!text_index || *text_index >= shdrs.size())
    return std::nullopt;
  const Elf64_Shdr &text_header = shdrs[*text_index];
  if (text_header.sh_type == SHT_NOBITS || text_header.sh_name != text.sectionHeaderNameIdx() ||
      text_header.sh_flags != text.flags() || text_header.sh_addr != text.vaddr() ||
      text_header.sh_offset != text.sectionOffset() || text_header.sh_size != text.size()) {
    return std::nullopt;
  }

  std::vector<uint64_t> leaders;

  for (size_t relocation_section_index = 0; relocation_section_index < shdrs.size();
       ++relocation_section_index) {
    const Elf64_Shdr &relocs = shdrs[relocation_section_index];
    if (relocs.sh_type != SHT_RELA)
      continue;
    const RocrRelocationSectionMode section_mode =
        classify_rocr_relocation_section(ehdr, shdrs, relocs);
    if (section_mode == RocrRelocationSectionMode::Malformed)
      return std::nullopt;
    if (section_mode != RocrRelocationSectionMode::Dynamic && relocs.sh_info != SHN_UNDEF) {
      if (relocs.sh_info >= shdrs.size())
        return std::nullopt;
      if ((shdrs[relocs.sh_info].sh_flags & SHF_ALLOC) == 0) {
        continue;
      }
    }
    if (relocs.sh_entsize != sizeof(Elf64_Rela) || relocs.sh_size % sizeof(Elf64_Rela) != 0 ||
        !elf_image_contains(image, relocs.sh_offset, relocs.sh_size)) {
      return std::nullopt;
    }

    const size_t count = relocs.sh_size / sizeof(Elf64_Rela);
    for (size_t relocation_index = 0; relocation_index < count; ++relocation_index) {
      Elf64_Rela relocation{};
      std::memcpy(&relocation,
                  image.data() + relocs.sh_offset + relocation_index * sizeof(Elf64_Rela),
                  sizeof(relocation));
      const uint32_t relocation_type = elf_reloc_type(relocation.r_info);
      const uint32_t symbol_index = elf_reloc_sym(relocation.r_info);
      const RocrNoneRelocationAction none_action =
          classify_rocr_none_relocation(section_mode, relocation.r_info);
      if (none_action == RocrNoneRelocationAction::Rejected)
        return std::nullopt;
      if (none_action == RocrNoneRelocationAction::Ignored)
        continue;
      if (!elf_relocation_place_is_allocated(ehdr, shdrs, relocs, relocation.r_offset))
        continue;

      const bool rocr_dynamic = section_mode == RocrRelocationSectionMode::Dynamic;
      const bool may_require_text_symbol_classification =
          symbol_index != 0 &&
          (!rocr_dynamic || rocr_dynamic_relocation_uses_symbol_value(relocation_type));
      if (may_require_text_symbol_classification) {
        if (relocs.sh_link >= shdrs.size())
          return std::nullopt;
        const Elf64_Shdr &symtab = shdrs[relocs.sh_link];
        if ((symtab.sh_type != SHT_SYMTAB && symtab.sh_type != SHT_DYNSYM) ||
            symtab.sh_entsize != sizeof(Elf64_Sym) || symtab.sh_size % sizeof(Elf64_Sym) != 0 ||
            !elf_image_contains(image, symtab.sh_offset, symtab.sh_size)) {
          return std::nullopt;
        }
        const size_t symbol_count = symtab.sh_size / sizeof(Elf64_Sym);
        if (symbol_index >= symbol_count)
          return std::nullopt;

        Elf64_Sym symbol{};
        std::memcpy(&symbol, image.data() + symtab.sh_offset + symbol_index * sizeof(Elf64_Sym),
                    sizeof(symbol));
        if (symbol.st_shndx == *text_index &&
            classify_text_symbol_relocation(section_mode, relocation.r_info,
                                            /*has_explicit_addend=*/true, relocation.r_addend,
                                            symbol) ==
                TextSymbolRelocationAction::RequiresExecutableEntry) {
          uint64_t offset = symbol.st_value;
          if (ehdr.e_type != ET_REL) {
            if (symbol.st_value < text.vaddr())
              return std::nullopt;
            offset = symbol.st_value - text.vaddr();
          }
          if (offset > text.size())
            return std::nullopt;
          if (offset < text.size())
            leaders.push_back(offset);
        }
      }

      if (ehdr.e_type != ET_DYN || relocation_type != R_AMDGPU_RELATIVE64 ||
          relocation.r_addend < 0) {
        continue;
      }
      const uint64_t target = static_cast<uint64_t>(relocation.r_addend);
      if (target < text.vaddr())
        continue;
      const uint64_t offset = target - text.vaddr();
      if (offset < text.size())
        leaders.push_back(offset);
    }
  }

  std::ranges::sort(leaders);
  leaders.erase(std::ranges::unique(leaders).begin(), leaders.end());
  return leaders;
}

[[nodiscard]] bool has_invalid_block_leader(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                                            std::span<const uint64_t> block_leaders,
                                            uint64_t text_size) {
  std::unordered_set<uint64_t> decoded_block_starts;
  decoded_block_starts.reserve(blocks.size());
  for (const auto &block : blocks) {
    if (block != nullptr)
      decoded_block_starts.insert(block->start_offset());
  }
  return std::ranges::any_of(block_leaders, [&](uint64_t leader) {
    return leader >= text_size || !decoded_block_starts.contains(leader);
  });
}

[[nodiscard]] bool has_supported_executable_section_layout(const AmdGpuCodeObject &object) {
  const auto &text = object.text_sections();
  const auto &executable = object.allocated_executable_sections();
  if (text.empty() && executable.empty())
    return true;
  return text.size() == 1 && executable.size() == 1 && text.front() == executable.front();
}

struct KernelTranslationScope {
  KdTranslation *translation = nullptr;
  BasicBlock *entry = nullptr;
  std::vector<BasicBlock *> blocks;
};

/// @brief Descriptor state mutated by one kernel-scope translation transaction.
struct DescriptorVariantCheckpoint {
  size_t index = 0;
  KdTranslation translation;
};

[[nodiscard]] uint64_t kernel_scope_key(const KdTranslation &kernel) {
  assert(kernel.entry_text_offset <= (std::numeric_limits<uint64_t>::max() >> 1) &&
         "kernel entry offset is too large to pack with variant bit");
  return (kernel.entry_text_offset << 1) | (kernel.needs_lds_overflow_buf ? 1u : 0u);
}

[[nodiscard]] bool same_kernel_scope_variant(const KdTranslation &lhs, const KdTranslation &rhs) {
  return lhs.entry_text_offset == rhs.entry_text_offset &&
         lhs.needs_lds_overflow_buf == rhs.needs_lds_overflow_buf;
}

[[nodiscard]] std::vector<DescriptorVariantCheckpoint>
checkpoint_scope_descriptors(std::span<const KdTranslation> translations,
                             const KdTranslation &scope_translation) {
  std::vector<DescriptorVariantCheckpoint> checkpoint;
  for (size_t i = 0; i < translations.size(); ++i) {
    if (same_kernel_scope_variant(translations[i], scope_translation))
      checkpoint.push_back({.index = i, .translation = translations[i]});
  }
  return checkpoint;
}

[[nodiscard]] size_t kernel_translation_scope_count(std::span<const KdTranslation> kernels) {
  std::unordered_set<uint64_t> keys;
  for (const KdTranslation &kernel : kernels)
    keys.insert(kernel_scope_key(kernel));
  return keys.size();
}

[[nodiscard]] bool scope_uses_virtualizable_lds(const KernelTranslationScope &scope,
                                                rj_code_arch_t guest_arch,
                                                rj_code_arch_t host_arch) {
  if (scope.translation == nullptr)
    return false;
  if (scope.translation->target_lds_size != 0)
    return true;

  for (BasicBlock *block : scope.blocks) {
    if (block == nullptr)
      continue;
    for (const Instruction &inst : block->instructions()) {
      if (source_instruction_uses_virtualizable_lds(inst, guest_arch, host_arch))
        return true;
    }
  }
  return false;
}

/// @brief Assemble a scope's hardware-entry offsets and run the external-entry
///        soundness gate (internal::scope_roots_are_entry_state).
///
/// @details Thin wrapper over the pure gate so translate() can pass a
/// KernelTranslationScope; the full soundness argument lives at the pure
/// function's definition below.
[[nodiscard]] bool
scope_incomplete_roots_are_entry_state(const KernelTranslationScope &scope,
                                       const std::unordered_set<uint64_t> &table_callee_offsets) {
  // Only the ordinary kernel scope entry is a safe root: hardware/ABI initializes
  // its SGPRs (dispatch pointer, kernarg pointer, workgroup ids), never a caller-
  // chosen code address.
  //
  // The kernarg-preload firmware entry (+256) is deliberately NOT a safe root.
  // Before control reaches it the command processor copies caller-controlled
  // kernarg words straight into user SGPRs (see command_processor.cpp,
  // KERNARG_PRELOAD_SPEC_LENGTH handling), so a preloaded user SGPR can hold an
  // original, unrelocated .text pointer that no in-scope builder or relocation
  // rewrites. An incomplete consumer rooted at that entry could therefore read a
  // stale code pointer, so it must fail closed.
  const std::unordered_set<uint64_t> hardware_entry_offsets{scope.entry->start_offset()};

  return internal::scope_roots_are_entry_state(scope.blocks, hardware_entry_offsets,
                                               table_callee_offsets);
}

/// @brief Prove that no stale PC-derived value can exist in one kernel scope.
///
/// @details The translator's usual model is "prove the target of every dynamic
/// transfer, or refuse". This helper establishes the complementary — and
/// strictly stronger — property: every value in this scope that was derived from
/// an `s_getpc_b64` is rewritten to hold its RELOCATED address. A consumer whose
/// dataflow fact is incomplete is then still safe, because whatever the
/// unconstrained path delivers can only be one of:
///   * a value this scope built from a getpc, which is now relocation-correct;
///   * an architectural return PC from s_call/s_swap_pc, which hardware writes
///     from the already-relocated program counter;
///   * a code address loaded from data, whose ELF relocation the code-object
///     patcher rewrites through the same final offset map (and refuses to
///     translate when it cannot).
/// No path can therefore carry an original, unrelocated `.text` address.
///
/// The proof obligation is discharged per producer and fails closed:
///   * a producer the analysis could not follow leaves an unknown value;
///   * a producer whose value is not a block start emitted by this scope cannot
///     be rewritten to a relocated address (it points at data, into the middle
///     of an instruction, or outside the emitted scope);
///   * a bare producer that is the last instruction of its block is the shape
///     whose delta add lives in a successor, so its chain is not proven closed.
/// Any of those returns nullopt, which keeps the caller's existing refusal.
///
/// The recorded value is the one the pair holds at its block's exit, so a later
/// unmodeled write inside the SAME block already leaves the producer unresolved.
/// The residual modeling assumption is that a closed chain is not extended by
/// unmodeled PC arithmetic in a SUCCESSOR block. Within one function that case
/// is refused elsewhere: the extension is a KILL transfer, a killed lattice fact
/// yields no fixup at all, and an indirect consumer with no fixup fails closed
/// as unrecovered. Escaping it would need an interprocedural chain (partial
/// build in a caller, completion after a call boundary) whose intermediate value
/// also lands exactly on an emitted block start. AMDGPU materializes a function
/// address with one indivisible getpc+add expansion, so no such chain exists.
///
/// @returns Builder rewrites that must all be applied, or nullopt when the
///          scope cannot be made free of stale PC-derived values.
[[nodiscard]] std::optional<std::vector<IndirectCallFixup>>
scope_relocatable_pc_builders(std::span<BasicBlock *const> blocks) {
  std::unordered_set<uint64_t> block_starts;
  block_starts.reserve(blocks.size());
  for (BasicBlock *block : blocks) {
    if (block == nullptr)
      return std::nullopt;
    block_starts.insert(block->start_offset());
  }

  std::vector<IndirectCallFixup> builder_fixups;
  // The instruction-start set is rebuilt per owning block rather than pooled
  // across the whole scope. patch_recovered_builder_fixups NOPs the entire
  // [begin, end) interval of a builder as one contiguous run, so that interval
  // must lie inside a single block. Discovery may add a recovered leader in a
  // later round that splits the analysis block a builder was recorded on; a
  // scope-wide instruction-start pool would still accept a range that now
  // straddles that split, and the patcher would overwrite the bytes inserted
  // between the final blocks. Bounding each builder to its owning block's
  // [start_offset, end_offset) and validating its range against only that
  // block's instruction starts fails the proof closed for any cross-block range.
  for (BasicBlock *block : blocks) {
    if (block->static_pc_address_builders().empty())
      continue;

    std::unordered_set<uint64_t> block_instruction_starts;
    block_instruction_starts.insert(block->end_offset());
    for (const Instruction &inst : block->instructions())
      block_instruction_starts.insert(inst.src_loc());

    const auto in_owning_block = [&](uint64_t offset) {
      return offset >= block->start_offset() && offset <= block->end_offset() &&
             block_instruction_starts.contains(offset);
    };

    for (const PcAddressBuilder &builder : block->static_pc_address_builders()) {
      if (!builder.resolved)
        return std::nullopt;
      // A non-contiguous range holds an unrelated instruction between builder
      // steps. patch_recovered_builder_fixups NOPs the whole range, so rewriting
      // it would erase that instruction. Fail the proof closed instead.
      if (!builder.contiguous)
        return std::nullopt;
      if (builder.source_target_offset < 0)
        return std::nullopt;
      const auto target = static_cast<uint64_t>(builder.source_target_offset);
      // patch_recovered_builder_fixups resolves the relocated target through
      // block placements, so only a block start has a defined new address.
      if (!block_starts.contains(target))
        return std::nullopt;
      // The getpc and its whole recovery range must be instruction starts inside
      // the block that owns the getpc, so the NOP-and-rewrite stays contiguous.
      if (!in_owning_block(builder.source_getpc_offset) ||
          !in_owning_block(builder.source_recovery_begin_offset) ||
          !in_owning_block(builder.source_recovery_end_offset)) {
        return std::nullopt;
      }
      if (builder.source_recovery_begin_offset == builder.source_recovery_end_offset) {
        // A bare getpc has no delta to rewrite: hardware already supplies the
        // relocated PC. Accept it only when its recorded value really is "the
        // instruction after the getpc" and at least one more instruction of the
        // same block follows without consuming it into a delta. A getpc that is
        // the last instruction of its block is the shape whose add lives in a
        // successor, where an unmodeled write would leave the original delta.
        if (target != builder.source_recovery_begin_offset)
          return std::nullopt;
        if (builder.source_recovery_end_offset >= block->end_offset())
          return std::nullopt;
        continue;
      }

      builder_fixups.push_back(
          IndirectCallFixup{.source_getpc_offset = builder.source_getpc_offset,
                            .source_recovery_begin_offset = builder.source_recovery_begin_offset,
                            .source_recovery_end_offset = builder.source_recovery_end_offset,
                            .source_call_offset = builder.source_getpc_offset,
                            .source_target_offset = target,
                            // A whole-scope builder has no consumer at all, so the two registers
                            // are necessarily the producer's pair.
                            .source_call_sreg = builder.source_sreg,
                            .source_builder_sreg = builder.source_sreg});
    }
  }
  return builder_fixups;
}

[[nodiscard]] std::unordered_set<uint64_t>
attach_relocation_table_call_edges(const BlockOffsetIndex &block_index,
                                   std::span<const RelocationFunctionTable> tables,
                                   std::span<const RelocationTableDispatch> dispatches) {
  std::unordered_set<uint64_t> accepted_calls;
  for (const RelocationTableDispatch &dispatch : dispatches) {
    if (dispatch.table_index >= tables.size())
      continue;
    BasicBlock *source = block_for_offset(block_index, dispatch.source_call_offset);
    if (source == nullptr || source->terminator() == nullptr ||
        source->terminator()->src_loc() != dispatch.source_call_offset)
      continue;
    BasicBlock *continuation = block_for_offset(block_index, source->end_offset());
    if (continuation == nullptr || continuation->start_offset() != source->end_offset())
      continue;

    std::vector<BasicBlock *> callees;
    callees.reserve(tables[dispatch.table_index].entries.size());
    bool complete = true;
    for (const RelocationFunctionPointer &entry : tables[dispatch.table_index].entries) {
      BasicBlock *callee = block_for_offset(block_index, entry.target_text_offset);
      if (callee == nullptr || callee->start_offset() != entry.target_text_offset) {
        complete = false;
        break;
      }
      callees.push_back(callee);
    }
    if (!complete || callees.empty())
      continue;

    for (BasicBlock *callee : callees) {
      source->add_call_edge({.kind = BasicBlock::CallEdgeKind::IndirectSwapPc,
                             .callee = callee,
                             .continuation = continuation,
                             .source_call_offset = dispatch.source_call_offset,
                             .return_sreg = dispatch.return_sreg});
    }
    accepted_calls.insert(dispatch.source_call_offset);
  }
  return accepted_calls;
}

/// @brief Build one translation scope per kernel descriptor variant.
///
/// @param adopted_roots Device-function entries that no kernel scope reaches on its own. A body
/// whose address is only ever produced by a data relocation has no decoded edge leading to it, so
/// the reachability walk cannot find it and the relocated `.text` would drop it. Such a body is
/// adopted as an additional root of the lowest-offset scope. One scope rather than every scope is
/// deliberate: a helper reached by several kernels is cloned per scope so each clone resolves
/// through its own placement map, but a runtime-dereferenced pointer has exactly one value, and
/// cloning would leave `relocate_relative_text_addends()` choosing between placements. Attributing
/// the body to a single scope is sound because it is entered through an absolute pointer and
/// returns through a caller-saved PC, so it is position-independent with respect to its callers.
[[nodiscard]] std::vector<KernelTranslationScope>
kernel_translation_scopes(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                          const BlockOffsetIndex &block_index, std::span<KdTranslation> kernels,
                          std::span<const uint64_t> adopted_roots = {},
                          const std::unordered_set<uint64_t> *address_taken_entries = nullptr) {
  std::vector<KernelTranslationScope> scopes;
  const auto entries = kernel_entry_offsets(kernels);
  if (entries.empty())
    return scopes;

  const BlockPositionIndex block_positions = build_block_position_index(blocks);
  static const std::unordered_set<uint64_t> kNoAddressTakenEntries;
  const std::unordered_set<uint64_t> &address_taken =
      address_taken_entries != nullptr ? *address_taken_entries : kNoAddressTakenEntries;
  const auto hardware_entries = kernel_hardware_entry_offsets(kernels);
  // Only own_entries varies per kernel, so the spec is built once and its
  // own_entries replaced each iteration rather than copying the other two sets
  // for every scope.
  KernelScopeSpec spec{
      .kernel_entries = {hardware_entries.begin(), hardware_entries.end()},
      .own_entries = {},
      .address_taken_entries = address_taken,
  };
  std::vector<KdTranslation *> ordered_kernels;
  ordered_kernels.reserve(kernels.size());
  std::unordered_set<uint64_t> seen_scopes;
  for (KdTranslation &kernel : kernels) {
    if (seen_scopes.insert(kernel_scope_key(kernel)).second)
      ordered_kernels.push_back(&kernel);
  }

  std::ranges::sort(ordered_kernels, [](const auto *lhs, const auto *rhs) {
    if (lhs->entry_text_offset != rhs->entry_text_offset)
      return lhs->entry_text_offset < rhs->entry_text_offset;
    return lhs->needs_lds_overflow_buf < rhs->needs_lds_overflow_buf;
  });

  scopes.reserve(ordered_kernels.size());
  for (KdTranslation *kernel : ordered_kernels) {
    BasicBlock *entry = block_for_offset(block_index, kernel->entry_text_offset);
    if (entry == nullptr)
      continue;
    std::unordered_set<uint64_t> own_entries{kernel->entry_text_offset};
    if (kernel->has_kernarg_preload_firmware_skip) {
      if (block_for_offset(block_index, kernel->kernarg_preload_firmware_entry_text_offset) ==
          nullptr)
        continue;
      own_entries.insert(kernel->kernarg_preload_firmware_entry_text_offset);
    }
    // ordered_kernels is sorted by entry offset, so "the first scope built" names the same scope on
    // every run and on a second pass over already-translated text.
    if (scopes.empty())
      own_entries.insert(adopted_roots.begin(), adopted_roots.end());

    spec.own_entries = std::move(own_entries);
    scopes.push_back({kernel, entry,
                      reachable_kernel_blocks(blocks, block_index, block_positions, *entry, spec)});
  }
  return scopes;
}

/// @brief Return offsets of the `s_setpc_b64` returns belonging to adopted device-function roots.
///
/// @details An adopted root has no call edge, so scoped_call_return_offsets() -- which derives the
/// return register from the call site that saved it -- reports nothing for it and the body's own
/// return reads as an unrecoverable indirect branch. The register is recovered from the body
/// instead: a terminator that jumps through an SGPR pair the body never redefines can only be
/// returning to an address its caller supplied. Such an address is absolute and produced outside
/// this code object's relocated text, so moving the body does not change it and there is nothing to
/// recover.
///
/// A lane restore is not a redefinition for this purpose. A non-leaf device function stashes the
/// incoming pair in a VGPR lane, lets its own `s_swap_pc_i64` calls overwrite the architectural
/// pair, and reads the original back with `v_readlane_b32` before returning. The question is
/// therefore which definition *reaches* the terminator, not whether one exists anywhere in the
/// body: the call's write is real but dead by then. The search walks back from the terminator to
/// the nearest definition of each half on every path, and accepts only a lane restore or the
/// function entry itself. Any other reaching definition disqualifies the pair, since a computed
/// value could be a relocated PC that needs the recovery this bypasses. A PC-relative builder
/// feeding the terminator is not reached here at all: the caller's recovered-indirect and
/// direct-branch tests run first and claim that shape.
[[nodiscard]] std::unordered_set<uint64_t>
adopted_root_return_offsets(const BlockOffsetIndex &block_index,
                            std::span<const uint64_t> adopted_roots,
                            std::span<const uint8_t> text) {
  std::unordered_set<uint64_t> returns;
  // AMDGPU pads between functions rather than ending one with a terminator, so a forward walk from
  // one root falls straight through into the next. Several pointer-only functions can share a
  // scope, and crossing that boundary would classify the next root's s_setpc against this root's
  // entry state. Every other root is therefore a wall for this walk.
  const std::unordered_set<uint64_t> root_boundaries(adopted_roots.begin(), adopted_roots.end());
  for (const uint64_t root : adopted_roots) {
    BasicBlock *entry = block_for_offset(block_index, root);
    if (entry == nullptr)
      continue;

    // Forward pass: collect the body's blocks and its candidate return terminators.
    std::vector<BasicBlock *> stack{entry};
    std::unordered_set<BasicBlock *> body;
    std::vector<std::pair<BasicBlock *, uint16_t>> candidates;
    while (!stack.empty()) {
      BasicBlock *block = stack.back();
      stack.pop_back();
      if (block == nullptr || !body.insert(block).second)
        continue;
      const Instruction *term = block->terminator();
      if (term != nullptr && term->size() == sizeof(uint32_t)) {
        const std::string_view mnemonic = term->mnemonic();
        if (mnemonic == "s_setpc_b64" || mnemonic == "s_set_pc_i64")
          candidates.emplace_back(
              block, static_cast<uint16_t>(text_word_at(text, term->src_loc()) & 0xffu));
      }
      for (BasicBlock *succ : block->successors()) {
        // A self-recursive root legitimately re-enters its own entry, so only OTHER roots stop it.
        if (succ != nullptr && succ != entry && root_boundaries.contains(succ->start_offset()))
          continue;
        stack.push_back(succ);
      }
    }

    // Whether (vgpr, lane) still holds what `v_writelane_b32` put there from `sgpr`, asked at one
    // program point rather than of the body as a whole. A body-wide set of saves cannot answer
    // this: a save that appears after the read, or on a path that does not join it, or one a later
    // write to the same VGPR has already destroyed, would all still be in the set and would let an
    // arbitrary lane read pass as a return-address restore.
    // Declared ahead of the lane query so that query can prove the SGPR a save captured really
    // was the caller's incoming value. The bool disables lane-restore classification inside the
    // nested query, which bounds the recursion at one level and keeps the nested question strictly
    // simpler than the outer one.
    std::function<bool(BasicBlock *, const Instruction *, uint16_t, bool)> caller_value_reaches;

    auto lane_holds_saved_sgpr = [&](BasicBlock *start, const Instruction *from, uint16_t vgpr,
                                     uint16_t lane, uint16_t sgpr) {
      std::vector<std::pair<BasicBlock *, const Instruction *>> work{{start, from}};
      // Keyed by the whole work item, not the block. Processing one is deterministic, so meeting
      // the same pair twice adds nothing and is skipped; the same block reached with a different
      // scan start is a different query and still has to run.
      std::set<std::pair<const BasicBlock *, const Instruction *>> seen;
      while (!work.empty()) {
        const auto [block, before] = work.back();
        work.pop_back();
        if (block == nullptr || !body.contains(block))
          return false;
        if (!seen.insert({block, before}).second)
          continue;
        std::vector<const Instruction *> ordered;
        ordered.reserve(block->num_instructions());
        for (const Instruction &inst : block->instructions())
          ordered.push_back(&inst);
        size_t cursor = ordered.size();
        if (before != nullptr) {
          const auto found = std::ranges::find(ordered, before);
          if (found == ordered.end())
            return false;
          cursor = static_cast<size_t>(found - ordered.begin());
        }

        bool resolved = false;
        while (cursor > 0) {
          --cursor;
          const Instruction &inst = *ordered[cursor];
          const std::string_view lane_mnemonic = inst.mnemonic();
          // The lane identity here is a low selector only. Anything that moves the VGPR bank makes
          // that selector name a different physical register, so the comparison below stops meaning
          // what it says and the proof has to give up.
          if (lane_mnemonic == "s_set_vgpr_msb" || lane_mnemonic == "s_setreg_b32" ||
              lane_mnemonic == "s_setreg_imm32_b32")
            return false;
          // A call only endangers the stash if the callee is allowed to write the lane's register.
          // The ABI keeps callee-saved VGPRs across a call, which is exactly what makes the
          // non-leaf save/call/restore shape legitimate, so those survive; a caller-saved register
          // may be overwritten inside the callee and fails closed.
          const bool is_call = (inst.flags() & INDIRECT_CALL) != 0 ||
                               lane_mnemonic == "s_call_i64" || lane_mnemonic == "s_swap_pc_i64" ||
                               lane_mnemonic == "s_swappc_b64";
          if (is_call && !is_callee_saved_vgpr(vgpr))
            return false;
          const Operand *vdst = inst.dst_operand(0);
          const bool writes_lane =
              lane_mnemonic == "v_writelane_b32" && vdst != nullptr && vdst->encoding_value() >= 0;
          if (writes_lane && static_cast<uint16_t>(vdst->encoding_value()) == vgpr) {
            const Operand *ssrc = inst.src_operand(0);
            const Operand *written_lane = inst.src_operand(1);
            if (ssrc == nullptr || written_lane == nullptr || ssrc->encoding_value() < 0 ||
                written_lane->encoding_value() < 0)
              return false;
            // A write to a different lane of the same VGPR leaves this lane alone.
            if (static_cast<uint16_t>(written_lane->encoding_value()) != lane)
              continue;
            if (static_cast<uint16_t>(ssrc->encoding_value()) != sgpr)
              return false;
            // Finding the save is not enough: it captured whatever the SGPR held at that point, so
            // the value is the caller's return PC only if the caller's value still reached here.
            if (!caller_value_reaches(block, &inst, sgpr, /*allow_lane_restore=*/false))
              return false;
            resolved = true;
            break;
          }
          // Any other definition of the VGPR rewrites every lane, including this one.
          bool clobbers = false;
          for (int i = 0; i < inst.num_dst_operands(); ++i) {
            const Operand *dst = inst.dst_operand(i);
            if (dst == nullptr || dst->encoding_value() < 0)
              continue;
            const auto base = static_cast<uint16_t>(dst->encoding_value());
            const int halves = dst->size_bits() > 32 ? dst->size_bits() / 32 : 1;
            if (vgpr >= base && vgpr < base + halves)
              clobbers = true;
          }
          // A destination operand is not the only way to write a VGPR. Ask the instruction for the
          // registers it writes without naming them, so a producer that touches this lane through a
          // hidden definition cannot slip past the operand walk above.
          if (!clobbers) {
            RegisterSet implicit;
            inst.implicit_defs(implicit);
            if (implicit.contains(RegisterRef{RegClass::VGPR, vgpr, 1}) ||
                implicit.contains(RegisterRef{RegClass::ACC_VGPR, vgpr, 1}))
              clobbers = true;
          }
          if (clobbers)
            return false;
        }
        if (resolved)
          continue;
        // Reaching the entry without meeting the save means the lane was never given the caller's
        // value on this path, which is the opposite of the SGPR search: there the entry is where
        // the caller's value comes from, here the save has to be inside the body.
        if (block == entry)
          return false;
        // Reconverging control flow -- a diamond, or a loop back edge -- reaches a block that is
        // already queued or done. That is ordinary, not a reason to give up on the whole query; the
        // dedup above absorbs it. Only a block with no predecessor inside the body is a path this
        // analysis cannot see, and that still fails closed.
        if (block->predecessors().empty())
          return false;
        for (BasicBlock *pred : block->predecessors())
          work.emplace_back(pred, nullptr);
      }
      return true;
    };

    // Classify one instruction's effect on the tracked half: a lane restore reinstates the
    // caller's value, any other definition replaces it with something this analysis cannot vouch
    // for, and everything else leaves the search running.
    enum class Effect { kNone, kRestores, kRedefines };
    auto effect_on = [&](const Instruction &inst, uint16_t sgpr, BasicBlock *block,
                         bool allow_lane_restore) {
      // A call writes the tracked pair without naming it whenever the callee does: the scalar ABI
      // has no callee-saved SGPRs that a translated body may rely on here, so any call between the
      // definition and the use invalidates the value. The call instruction's own destination list
      // does not mention it, which is why this is checked before the operand walk.
      const std::string_view sgpr_mnemonic = inst.mnemonic();
      if ((inst.flags() & INDIRECT_CALL) != 0 || sgpr_mnemonic == "s_call_i64" ||
          sgpr_mnemonic == "s_swap_pc_i64" || sgpr_mnemonic == "s_swappc_b64")
        return Effect::kRedefines;
      bool defines = false;
      for (int i = 0; i < inst.num_dst_operands(); ++i) {
        const Operand *dst = inst.dst_operand(i);
        if (dst == nullptr || dst->encoding_value() < 0)
          continue;
        const auto base = static_cast<uint16_t>(dst->encoding_value());
        const int halves = dst->size_bits() > 32 ? dst->size_bits() / 32 : 1;
        if (sgpr >= base && sgpr < base + halves)
          defines = true;
      }
      // Destination operands are not the only way to write an SGPR; ask the instruction for the
      // ones it writes without naming them.
      if (!defines) {
        RegisterSet implicit;
        inst.implicit_defs(implicit);
        if (implicit.contains(RegisterRef{RegClass::SGPR, sgpr, 1}))
          defines = true;
      }
      if (!defines)
        return Effect::kNone;
      // The nested provenance query runs with this off: a lane restore is exactly what it is
      // trying to justify, so letting it count one would be circular.
      if (!allow_lane_restore || inst.mnemonic() != "v_readlane_b32")
        return Effect::kRedefines;
      // Only a read of the exact lane this body saved the same register into restores the caller's
      // value. Reading some other lane, or another register's lane, produces a value this analysis
      // cannot vouch for and must not be mistaken for a return address.
      const Operand *vsrc = inst.src_operand(0);
      const Operand *lane = inst.src_operand(1);
      if (vsrc == nullptr || lane == nullptr || vsrc->encoding_value() < 0 ||
          lane->encoding_value() < 0) {
        return Effect::kRedefines;
      }
      // Ask at this instruction, not of the body: the save has to reach this read on every path.
      if (!lane_holds_saved_sgpr(block, &inst, static_cast<uint16_t>(vsrc->encoding_value()),
                                 static_cast<uint16_t>(lane->encoding_value()), sgpr))
        return Effect::kRedefines;
      return Effect::kRestores;
    };

    // Backward reaching-definition search for one half, starting just above `from`.
    caller_value_reaches = [&](BasicBlock *start, const Instruction *from, uint16_t sgpr,
                               bool allow_lane_restore) {
      std::vector<std::pair<BasicBlock *, const Instruction *>> work{{start, from}};
      // Keyed by the whole work item, not the block. Processing one is deterministic, so meeting
      // the same pair twice adds nothing and is skipped; the same block reached with a different
      // scan start is a different query and still has to run.
      std::set<std::pair<const BasicBlock *, const Instruction *>> seen;
      while (!work.empty()) {
        const auto [block, before] = work.back();
        work.pop_back();
        if (block == nullptr || !body.contains(block))
          return false;
        if (!seen.insert({block, before}).second)
          continue;
        // The instruction list is forward-only, so materialize it to scan upwards from `before`.
        std::vector<const Instruction *> ordered;
        ordered.reserve(block->num_instructions());
        for (const Instruction &inst : block->instructions())
          ordered.push_back(&inst);
        size_t cursor = ordered.size();
        if (before != nullptr) {
          const auto found = std::ranges::find(ordered, before);
          if (found == ordered.end())
            return false;
          cursor = static_cast<size_t>(found - ordered.begin());
        }

        bool resolved = false;
        while (cursor > 0) {
          --cursor;
          const Effect effect = effect_on(*ordered[cursor], sgpr, block, allow_lane_restore);
          if (effect == Effect::kNone)
            continue;
          if (effect == Effect::kRedefines)
            return false;
          resolved = true;
          break;
        }
        if (resolved)
          continue;
        // No definition in this block. The function entry means the value is the caller's; any
        // other block defers to its predecessors, and a predecessor outside the body is a path
        // this analysis cannot see, so it fails closed above.
        if (block == entry)
          continue;
        // Reconverging control flow -- a diamond, or a loop back edge -- reaches a block that is
        // already queued or done. That is ordinary, not a reason to give up on the whole query; the
        // dedup above absorbs it. Only a block with no predecessor inside the body is a path this
        // analysis cannot see, and that still fails closed.
        if (block->predecessors().empty())
          return false;
        for (BasicBlock *pred : block->predecessors())
          work.emplace_back(pred, nullptr);
      }
      return true;
    };

    for (const auto &[block, ssrc0] : candidates) {
      const Instruction *term = block->terminator();
      if (caller_value_reaches(block, term, ssrc0, /*allow_lane_restore=*/true) &&
          caller_value_reaches(block, term, static_cast<uint16_t>(ssrc0 + 1),
                               /*allow_lane_restore=*/true))
        returns.insert(term->src_loc());
    }
  }
  return returns;
}

/// @brief Commit translated text, descriptor plans, and runtime metadata to one ELF image.
///
/// @details BinaryTranslator owns analysis and per-kernel lowering. This helper
/// owns the separate commit responsibility: applying completed descriptor plans,
/// replacing `.text`, appending sidecar descriptors, resolving their final virtual
/// addresses, and serializing runtime metadata. It mutates only its private patcher
/// copy, so any failure leaves the caller free to return the original code object.
[[nodiscard]] std::optional<std::vector<uint8_t>> materialize_translated_code_object(
    CodeObjectPatcher patcher, std::vector<uint8_t> translated_text, uint64_t original_text_size,
    std::span<const TextOffsetRelocation> text_relocations,
    std::span<const PcRelativeDataRelocation> data_relocations,
    std::span<const PcRelativeTextRelocation> code_relocations,
    std::span<const KdTranslation> translations, rj_code_arch_t host_arch, uint32_t target_mach,
    bool require_every_text_symbol_mapped,
    const std::unordered_map<uint64_t, uint64_t> &canonical_code_pointer_placement,
    std::vector<TranslationDiagnostic> &diagnostics) {
  if (translated_text.size() < original_text_size)
    append_nop_padding(translated_text, original_text_size - translated_text.size(), host_arch);

  std::unordered_set<uint64_t> applied_descriptors;
  for (const KdTranslation &translation : translations) {
    if (translation.sidecar_descriptor)
      continue;
    if (!applied_descriptors.insert(translation.descriptor_file_offset).second)
      continue;
    if (!patcher.apply_kernel_descriptor_translation(translation, host_arch)) {
      append_error(
          diagnostics, DiagnosticKind::KernelDescriptor,
          translation.skipped
              ? "skipped kernel descriptor could not be patched to a target stub safely; leaving "
                "code object unchanged"
              : "kernel descriptor translation could not be applied safely; leaving code object "
                "unchanged");
      return std::nullopt;
    }
  }

  if (!patcher.replace_text(translated_text, text_relocations, data_relocations, code_relocations,
                            require_every_text_symbol_mapped, &canonical_code_pointer_placement)) {
    append_error(diagnostics, DiagnosticKind::ResourceLimit,
                 "relocated .text could not be materialized safely; leaving code object unchanged");
    return std::nullopt;
  }

  std::vector<uint64_t> sidecar_descriptor_vaddrs(translations.size(), 0);
  std::vector<KdTranslation> sidecar_descriptors;
  std::vector<size_t> sidecar_indices;
  for (size_t i = 0; i < translations.size(); ++i) {
    const KdTranslation &translation = translations[i];
    if (!translation.sidecar_descriptor || !translation.needs_lds_overflow_buf ||
        translation.skipped)
      continue;
    sidecar_descriptors.push_back(translation);
    sidecar_indices.push_back(i);
  }
  if (!sidecar_descriptors.empty()) {
    auto appended =
        patcher.append_sidecar_descriptor_translations(sidecar_descriptors, host_arch, 64);
    if (!appended || appended->size() != sidecar_descriptors.size()) {
      append_error(diagnostics, DiagnosticKind::ResourceLimit,
                   "virtual LDS sidecar descriptors could not be materialized safely; leaving code "
                   "object unchanged");
      return std::nullopt;
    }
    for (size_t i = 0; i < appended->size(); ++i)
      sidecar_descriptor_vaddrs[sidecar_indices[i]] = (*appended)[i].vaddr;
  }

  const auto patched_image = patcher.image_bytes();
  AmdGpuCodeObject patched_layout(patched_image.data(), patched_image.size());
  if (!patched_layout.is_valid()) {
    append_error(diagnostics, DiagnosticKind::ResourceLimit,
                 "relocated ELF could not be reparsed for runtime metadata; leaving code object "
                 "unchanged");
    return std::nullopt;
  }

  std::vector<SidecarVariantMetadata> sidecar_metadata;
  std::vector<KernargExtensionMetadata> kernarg_extension_metadata;
  std::vector<VirtualLdsKernelMetadata> virtual_lds_metadata;
  for (size_t i = 0; i < translations.size(); ++i) {
    const KdTranslation &translation = translations[i];
    if (!translation.sidecar_descriptor || !translation.needs_lds_overflow_buf ||
        translation.skipped) {
      continue;
    }
    const auto normal_translation =
        std::ranges::find_if(translations, [&](const KdTranslation &candidate) {
          return !candidate.sidecar_descriptor && !candidate.skipped &&
                 candidate.kernel_name == translation.kernel_name;
        });
    if (normal_translation == translations.end()) {
      append_error(diagnostics, DiagnosticKind::KernelDescriptor,
                   "virtual LDS metadata could not find the normal descriptor translation; "
                   "leaving code object unchanged");
      return std::nullopt;
    }
    const uint64_t descriptor_vaddr =
        patched_layout.kernel_descriptor_offset(translation.kernel_name);
    if (descriptor_vaddr == 0) {
      append_error(diagnostics, DiagnosticKind::KernelDescriptor,
                   "virtual LDS metadata could not find the translated kernel descriptor symbol; "
                   "leaving code object unchanged");
      return std::nullopt;
    }

    const uint64_t sidecar_descriptor_vaddr = sidecar_descriptor_vaddrs[i];
    if (sidecar_descriptor_vaddr == 0) {
      append_error(diagnostics, DiagnosticKind::KernelDescriptor,
                   "virtual LDS metadata could not find the appended sidecar descriptor; leaving "
                   "code object unchanged");
      return std::nullopt;
    }

    // Sidecar identity, kernarg extension layout, and virtual-LDS policy are
    // serialized independently. Their stable kernel/variant names are the join
    // key; no generic mechanism embeds another feature's fields.
    sidecar_metadata.push_back(SidecarVariantMetadata{
        .kernel_name = translation.kernel_name,
        .variant_name = std::string(kVirtualLdsSidecarVariantName),
        .normal_descriptor_vaddr = descriptor_vaddr,
        .variant_descriptor_vaddr = sidecar_descriptor_vaddr,
    });

    KernargExtensionMetadata kernarg_extension{
        .kernel_name = translation.kernel_name,
        .variant_name = std::string(kVirtualLdsSidecarVariantName),
        .original_kernarg_size = translation.kernarg_size,
        .payloads = {{
            .size = kVirtualLdsRuntimeStateBytes,
            .alignment = alignof(uint64_t),
            .name = std::string(kVirtualLdsRuntimeStatePayloadName),
        }},
    };
    const KernargExtensionPayloadLayout payload_layout{
        .size = kernarg_extension.payloads.front().size,
        .alignment = kernarg_extension.payloads.front().alignment,
    };
    const auto wrapper_layout = make_kernarg_extension_layout(
        kernarg_extension.original_kernarg_size, std::span{&payload_layout, 1});
    if (!wrapper_layout || wrapper_layout->payload_offsets.empty() ||
        wrapper_layout->payload_offsets.front() !=
            translation.lds_overflow_kernarg_pointer_offset) {
      append_error(diagnostics, DiagnosticKind::KernelDescriptor,
                   "virtual LDS kernarg extension layout disagrees with the translated entry "
                   "prologue; leaving code object unchanged");
      return std::nullopt;
    }
    kernarg_extension_metadata.push_back(std::move(kernarg_extension));

    VirtualLdsKernelMetadata record{};
    record.kernel_name = translation.kernel_name;
    record.sidecar_variant_name = std::string(kVirtualLdsSidecarVariantName);
    record.static_lds_bytes = translation.lds_overflow_size;
    // AQL private_segment_size can include a dynamic call-stack request above
    // the normal descriptor's fixed allocation. Dispatch rewriting needs both
    // fixed sizes to preserve that dynamic portion while switching variants;
    // neither loaded descriptor address is guaranteed to be CPU-readable.
    record.normal_private_segment_size = normal_translation->target_private_size;
    record.virtual_private_segment_size = translation.target_private_size;
    record.virtual_lds_base_sgpr = translation.virtual_lds_lowering.base_sgpr;
    record.flags |= kVirtualLdsFlagRuntimeStateBlock;
    if (translation.workgroup_id_sgpr_x >= 0)
      record.flags |= kVirtualLdsFlagWorkgroupIdX;
    if (translation.workgroup_id_sgpr_y >= 0)
      record.flags |= kVirtualLdsFlagWorkgroupIdY;
    if (translation.workgroup_id_sgpr_z >= 0)
      record.flags |= kVirtualLdsFlagWorkgroupIdZ;
    virtual_lds_metadata.push_back(std::move(record));
  }

  if (!sidecar_metadata.empty()) {
    const auto metadata_bytes = serialize_sidecar_metadata(sidecar_metadata);
    if (metadata_bytes.empty() ||
        !patcher.append_nonalloc_section(kSidecarMetadataSectionName, metadata_bytes, 8)) {
      append_error(diagnostics, DiagnosticKind::ResourceLimit,
                   "sidecar metadata could not be materialized safely; leaving code object "
                   "unchanged");
      return std::nullopt;
    }
  }

  if (!kernarg_extension_metadata.empty()) {
    const auto metadata_bytes = serialize_kernarg_extension_metadata(kernarg_extension_metadata);
    if (metadata_bytes.empty() ||
        !patcher.append_nonalloc_section(kKernargExtensionMetadataSectionName, metadata_bytes, 8)) {
      append_error(diagnostics, DiagnosticKind::ResourceLimit,
                   "kernarg extension metadata could not be materialized safely; leaving code "
                   "object unchanged");
      return std::nullopt;
    }
  }

  if (!virtual_lds_metadata.empty()) {
    const auto metadata_bytes = serialize_virtual_lds_metadata(virtual_lds_metadata);
    if (metadata_bytes.empty() ||
        !patcher.append_nonalloc_section(kVirtualLdsMetadataSectionName, metadata_bytes, 8)) {
      append_error(diagnostics, DiagnosticKind::ResourceLimit,
                   "virtual LDS metadata could not be materialized safely; leaving code object "
                   "unchanged");
      return std::nullopt;
    }
  }

  if (target_mach)
    patcher.update_elf_flags(target_mach);
  return std::move(patcher).emit();
}

/// @brief One instruction's source and translated form, held until its target offset is known.
///
/// @details Traces cannot be emitted during lowering because a body's final placement is only
/// fixed after relocation, so each is queued with its source-relative offset and rebased later.
struct PendingTrace {
  uint64_t source_offset = 0;
  uint32_t source_size = 0;
  std::vector<uint32_t> source_words;
  const InstructionLegalization *legalization = nullptr;
  bool copied_original = false;
  bool semantic_lowering = false;
  bool changed = false;
  uint64_t target_offset = 0;
  std::vector<uint32_t> target_words;
};

/// @brief A per-kernel lowering failure, before it is turned into a diagnostic.
///
/// @details Carried rather than reported immediately because skip_failed_kernels decides whether
/// it becomes an object-level error or the prose of a KernelSkipped warning.
struct KernelFailure {
  DiagnosticKind kind = DiagnosticKind::Legalization;
  std::string message;
  std::optional<uint64_t> guest_offset;
  std::string mnemonic;
  std::vector<std::string> required_work;
};

/// @brief Build a per-kernel failure record.
[[nodiscard]] KernelFailure make_kernel_failure(DiagnosticKind kind, std::string message,
                                                std::optional<uint64_t> guest_offset = std::nullopt,
                                                std::string mnemonic = {},
                                                std::vector<std::string> required_work = {}) {
  return KernelFailure{kind, std::move(message), guest_offset, std::move(mnemonic),
                       std::move(required_work)};
}

/// @brief Map a control-flow relocation failure onto the diagnostic kind that reports it.
[[nodiscard]] DiagnosticKind relocation_diagnostic_kind(const TextRelocationResult &relocation) {
  if (relocation.failure == TextLayoutFailureCategory::ResourceLimit)
    return DiagnosticKind::ResourceLimit;
  return DiagnosticKind::Legalization;
}

/// @brief Map a body-materialization failure onto the diagnostic kind that reports it.
[[nodiscard]] DiagnosticKind
materialization_diagnostic_kind(const KernelTextAppendResult &materialization) {
  if (materialization.failure == TextLayoutFailureCategory::ResourceLimit)
    return DiagnosticKind::ResourceLimit;
  return DiagnosticKind::KernelDescriptor;
}

/// @brief A code-address builder awaiting the final placement of the body it names.
struct PendingCodeRelocation {
  uint64_t target_getpc_offset = 0;
  uint64_t target_literal_offset = 0;
  uint64_t source_target_text_offset = 0;
};

/// @brief Point every deferred code-address builder at the placement its target received.
///
/// @details Runs after every scope is placed, which is the first moment a target's final offset
/// is known. A source offset emitted more than once has no single answer -- a runtime-dereferenced
/// code address cannot choose between clones -- so that fails closed rather than picking one,
/// matching how relocate_relative_text_addends treats a conflicting relocation addend.
///
/// @returns false when the object must be left unchanged.
[[nodiscard]] bool resolve_pending_code_relocations(
    const std::vector<PendingCodeRelocation> &pending,
    const std::vector<TextOffsetRelocation> &text_relocations,
    const std::unordered_map<uint64_t, uint64_t> &canonical_placement,
    const std::unordered_set<uint64_t> &canonical_placement_variant_conflict,
    std::vector<PcRelativeTextRelocation> &code_relocations,
    std::vector<TranslationDiagnostic> &diagnostics) {
  if (pending.empty())
    return true;

  std::unordered_map<uint64_t, uint64_t> placement;
  std::unordered_set<uint64_t> conflicting;
  for (const TextOffsetRelocation &relocation : text_relocations) {
    auto [it, inserted] = placement.try_emplace(relocation.source_offset, relocation.target_offset);
    if (!inserted && it->second != relocation.target_offset)
      conflicting.insert(relocation.source_offset);
  }

  for (const PendingCodeRelocation &entry : pending) {
    // A canonical copy only answers for its clones when they are interchangeable. Offsets whose
    // clones disagreed on kernel-scope variant are not, so they fall through to the placement map
    // below and fail closed there rather than silently naming whichever clone was nominated first.
    if (const auto canonical = canonical_placement.find(entry.source_target_text_offset);
        canonical != canonical_placement.end() &&
        !canonical_placement_variant_conflict.contains(entry.source_target_text_offset)) {
      code_relocations.push_back({.target_getpc_offset = entry.target_getpc_offset,
                                  .target_literal_offset = entry.target_literal_offset,
                                  .target_text_offset = canonical->second});
      continue;
    }
    const auto placed = placement.find(entry.source_target_text_offset);
    if (placed == placement.end() || conflicting.contains(entry.source_target_text_offset)) {
      append_error(diagnostics, DiagnosticKind::Legalization,
                   conflicting.contains(entry.source_target_text_offset)
                       ? "PC-relative code address names a body emitted at more than one "
                         "placement; leaving code object unchanged"
                       : "PC-relative code address names a body this translation did not emit; "
                         "leaving code object unchanged",
                   entry.source_target_text_offset);
      return false;
    }
    code_relocations.push_back({.target_getpc_offset = entry.target_getpc_offset,
                                .target_literal_offset = entry.target_literal_offset,
                                .target_text_offset = placed->second});
  }
  return true;
}

/// @brief What one scope added to the three code-address containers, so a skipped scope can take
/// it back.
///
/// @details The two vectors only ever grow within a scope, so a length restores them;
/// canonical_placement is keyed by source offset and needs the inserted keys named.
struct ScopeRelocationCheckpoint {
  size_t code_relocations_begin = 0;
  size_t pending_code_relocations_begin = 0;
  std::vector<uint64_t> canonical_placements_added;
  // Variant conflicts this scope raised. A skipped scope's text is discarded, so the clone it
  // disagreed about no longer exists; leaving the conflict behind would withhold a canonical
  // placement whose surviving clones actually agree and refuse the object over a scope that
  // contributes nothing. The companion is_sidecar entry is restored with it so a later scope
  // compares against the variant that really nominated the canonical copy.
  std::vector<uint64_t> canonical_variant_conflicts_added;
  // The resource verdict is set before descriptor recomputation, which can still skip the scope.
  // Rolling the placements back without also restoring this would leave a skipped host's verdict
  // standing and refuse the object over a scope that no longer contributes anything.
  bool canonical_host_outgrew_its_descriptor = false;
};

/// @brief The object-level state a scope appends to, gathered so capture and rollback name one
/// set rather than two open-coded lists that can drift apart.
struct ObjectRollbackState {
  std::vector<uint8_t> &translated_text;
  std::vector<TextOffsetRelocation> &text_relocations;
  std::vector<PcRelativeDataRelocation> &data_relocations;
  std::vector<PcRelativeTextRelocation> &code_relocations;
  std::vector<PendingCodeRelocation> &pending_code_relocations;
  std::unordered_map<uint64_t, uint64_t> &canonical_placement;
  std::unordered_map<uint64_t, bool> &canonical_placement_is_sidecar;
  std::unordered_set<uint64_t> &canonical_placement_variant_conflict;
  bool &canonical_host_outgrew_its_descriptor;
  std::vector<KdTranslation> &descriptor_translations;
};

/// @brief Everything one scope must be able to give back if it fails.
///
/// @details capture() and rollback() are the only two places that name the rollback set, so the
/// two halves sit next to each other and can be read against one another. That is a legibility
/// guarantee, not a compiler-enforced one: adding a member to ObjectRollbackState and binding it
/// still compiles without a matching capture or restore, so both methods must be updated by hand.
/// This is also not a RAII guard: rollback is conditional on skip_failed_kernels, and a failing
/// scope still contributes a stub body, so unwinding is an explicit call.
struct ScopeTransaction {
  size_t output_begin = 0;
  size_t text_relocations_begin = 0;
  size_t data_relocations_begin = 0;
  std::vector<DescriptorVariantCheckpoint> descriptors;
  ScopeRelocationCheckpoint relocations;

  /// @brief Record where every object-level container stands before a scope emits into it.
  [[nodiscard]] static ScopeTransaction capture(const ObjectRollbackState &state,
                                                const KdTranslation &translation) {
    return ScopeTransaction{
        .output_begin = state.translated_text.size(),
        .text_relocations_begin = state.text_relocations.size(),
        .data_relocations_begin = state.data_relocations.size(),
        .descriptors = checkpoint_scope_descriptors(state.descriptor_translations, translation),
        .relocations = {.code_relocations_begin = state.code_relocations.size(),
                        .pending_code_relocations_begin = state.pending_code_relocations.size(),
                        .canonical_placements_added = {},
                        .canonical_variant_conflicts_added = {},
                        .canonical_host_outgrew_its_descriptor =
                            state.canonical_host_outgrew_its_descriptor}};
  }

  /// @brief Return every container to its captured state.
  ///
  /// @details The code-address records and canonical placements name offsets inside the text this
  /// discards. A later scope reuses those offsets for its own body, so keeping the records would
  /// write a 64-bit literal into instructions that never asked for one.
  ///
  /// @returns false if a descriptor checkpoint no longer indexes its translation, which the caller
  /// must report rather than silently restore the wrong entry.
  [[nodiscard]] bool rollback(const ObjectRollbackState &state) const {
    state.translated_text.resize(output_begin);
    state.text_relocations.resize(text_relocations_begin);
    state.data_relocations.resize(data_relocations_begin);
    state.code_relocations.resize(relocations.code_relocations_begin);
    state.pending_code_relocations.resize(relocations.pending_code_relocations_begin);
    for (const uint64_t placed : relocations.canonical_placements_added) {
      state.canonical_placement.erase(placed);
      state.canonical_placement_is_sidecar.erase(placed);
    }
    for (const uint64_t conflicted : relocations.canonical_variant_conflicts_added)
      state.canonical_placement_variant_conflict.erase(conflicted);
    state.canonical_host_outgrew_its_descriptor = relocations.canonical_host_outgrew_its_descriptor;
    for (const DescriptorVariantCheckpoint &saved : descriptors) {
      if (saved.index >= state.descriptor_translations.size())
        return false;
      state.descriptor_translations[saved.index] = saved.translation;
    }
    return true;
  }
};

} // namespace

namespace internal {

RewriteDischargeInstructionDecoder::RewriteDischargeInstructionDecoder(Decoder &decoder)
    : decoder_(decoder), lookahead_words_(decoder.max_instruction_words()) {}

RewriteDischargeDecodeStatus RewriteDischargeInstructionDecoder::decode(
    std::span<const uint8_t> remaining_bytes, uint64_t source_offset,
    std::unique_ptr<Instruction> &instruction, const DecodeErrorEmitter &emit_error) {
  instruction.reset();
  if (lookahead_words_.empty())
    return RewriteDischargeDecodeStatus::InvalidLookaheadBound;

  std::ranges::fill(lookahead_words_, 0u);
  const size_t remaining_words = remaining_bytes.size() / sizeof(uint32_t);
  const size_t copied_words = std::min(remaining_words, lookahead_words_.size());
  std::memcpy(lookahead_words_.data(), remaining_bytes.data(), copied_words * sizeof(uint32_t));
  DecodeResult decoded = decoder_.decode(lookahead_words_.data(), source_offset, emit_error);
  if (decoded.failed())
    return RewriteDischargeDecodeStatus::InvalidEncoding;
  instruction = std::move(decoded).value();
  if (instruction == nullptr || instruction->size() <= 0 ||
      instruction->size() % static_cast<int>(sizeof(uint32_t)) != 0) {
    return RewriteDischargeDecodeStatus::InvalidInstructionSize;
  }

  const size_t decoded_words = static_cast<size_t>(instruction->size()) / sizeof(uint32_t);
  if (decoded_words > lookahead_words_.size() || decoded_words > remaining_words)
    return RewriteDischargeDecodeStatus::TruncatedInstruction;
  return RewriteDischargeDecodeStatus::Success;
}

/// @brief Prove that every external entry into an incomplete-consumer scope is
///        an entry-state root that cannot carry an original `.text` pointer.
///
/// @details scope_relocatable_pc_builders proves every getpc-derived value in a
/// scope is relocated, but that proof only covers values this scope PRODUCES. An
/// incomplete consumer is also reachable along a path that enters the scope
/// carrying an SGPR value from OUTSIDE it. The whole-scope proof is sound only
/// when every such external entry is a root whose incoming SGPRs are
/// architecturally defined, never a raw original code address:
///   * a hardware kernel entry passed in @p hardware_entry_offsets — the caller
///     supplies only entries whose live-in SGPRs are ABI-initialized (dispatch
///     pointer, kernarg pointer, workgroup ids). The kernarg-preload firmware
///     entry is deliberately excluded there, because caller-controlled kernarg
///     words are copied into user SGPRs before it runs;
///   * a getpc-recovered in-scope call target (callee of a proven direct or
///     swappc call edge) — it is entered only through that call, so its live-in
///     PC pair is the architected return PC hardware wrote from the already-
///     relocated program counter, or a value the caller built in-scope from a
///     getpc (now relocated).
///
/// A relocation-table-dispatched callee is NOT such a safe root, even though it
/// has an in-scope CallEdge: the dispatch selects a callee dynamically and its
/// live-in scalar registers are arbitrary caller-supplied arguments, which can
/// include an original, unrelocated `.text` pointer. A call edge constrains
/// control flow, not the SGPR arguments delivered along it, so a table-dispatched
/// callee that itself holds an incomplete consumer could still receive a stale
/// code pointer on one path. Those callees are treated as unconstrained roots.
///
/// A block reachable within the scope has an in-scope predecessor (an ordinary
/// CFG edge) or is a non-table call-edge callee; any other block — one with no
/// in-scope predecessor and no proven getpc-recovered call edge — is entered from
/// outside the scope. Such an external-entry block is an unconstrained root:
/// control can arrive there holding a caller-supplied function pointer that is an
/// original, unrelocated `.text` address. The producer scan cannot rewrite that
/// value, so the incomplete consumer downstream could jump to stale bytes. This
/// gate fails closed for such a scope, which keeps the caller's original refusal.
/// Empirically every incomplete-consumer scope in the gfx1250 hotswap corpus
/// roots only at the kernel entry and at getpc-recovered call targets.
bool scope_roots_are_entry_state(std::span<BasicBlock *const> blocks,
                                 const std::unordered_set<uint64_t> &hardware_entry_offsets,
                                 const std::unordered_set<uint64_t> &table_callee_offsets) {
  std::unordered_set<const BasicBlock *> in_scope(blocks.begin(), blocks.end());
  std::unordered_set<const BasicBlock *> call_targets;
  for (const BasicBlock *block : blocks) {
    if (block == nullptr)
      return false;
    for (const BasicBlock::CallEdge &edge : block->call_edges()) {
      if (in_scope.contains(edge.callee))
        call_targets.insert(edge.callee);
    }
  }

  for (const BasicBlock *block : blocks) {
    const bool has_in_scope_predecessor = std::ranges::any_of(
        block->predecessors(), [&](const BasicBlock *pred) { return in_scope.contains(pred); });
    if (has_in_scope_predecessor)
      continue;
    // A block with no in-scope predecessor is an external entry (it has no
    // ordinary CFG edge from within the scope, even if it is reached by a call
    // edge or has predecessors outside the scope).
    //
    // A relocation-table-dispatched callee delivers unconstrained caller-supplied
    // SGPR arguments, so it is never a safe root regardless of its CallEdge; fail
    // closed for it even if it is also a getpc-recovered call target.
    if (table_callee_offsets.contains(block->start_offset()))
      return false;
    // Otherwise accept a hardware kernel entry or a getpc-recovered in-scope call
    // target; any other root is an unconstrained external entry that may deliver a
    // stale code pointer.
    if (hardware_entry_offsets.contains(block->start_offset()) || call_targets.contains(block))
      continue;
    return false;
  }
  return true;
}

} // namespace internal

BinaryTranslator::~BinaryTranslator() = default;

BinaryTranslator::BinaryTranslator(rj_code_arch_t guest_arch, rj_code_arch_t host_arch,
                                   uint32_t target_mach, BinaryTranslatorOptions options)
    : guest_arch_(guest_arch), host_arch_(host_arch),
      target_mach_(target_mach ? target_mach : elf_mach_for_arch(host_arch)), options_(options),
      encoding_translate_(select_encoding_translator(guest_arch, host_arch)),
      legalization_lookup_(select_legalization(guest_arch, host_arch)),
      semantic_translator_(std::make_unique<SemanticTranslator>(
          guest_arch, host_arch, options.input_revision, options.output_revision)) {}

bool BinaryTranslator::is_gfx1250_b0_to_a0() const {
  return guest_arch_ == ROCJITSU_CODE_ARCH_CDNA5 && host_arch_ == ROCJITSU_CODE_ARCH_CDNA5 &&
         (target_mach_ & EF_AMDGPU_MACH) == EF_AMDGPU_MACH_AMDGCN_GFX1250 &&
         options_.input_revision == ProcessorRevision::Gfx1250B0 &&
         options_.output_revision == ProcessorRevision::Gfx1250A0;
}

const InstructionLegalization *
BinaryTranslator::lookup_legalization(const Instruction &inst) const {
  // gfx1250 B0 and A0 have the same structural ISA, so the generated cross-ISA
  // tables cannot express their revision-specific behavior. Instructions in
  // the B0-to-A0 profile use handwritten legalization; everything else follows
  // the raw same-ISA copy path.
  const InstructionLegalization *legalization = nullptr;
  if (is_gfx1250_b0_to_a0())
    legalization = gfx1250_b0_to_a0_legalization(inst);
  else if (legalization_lookup_)
    legalization = legalization_lookup_(inst.encoding_id(), inst.opcode());
  return legalization;
}

void BinaryTranslator::set_trace_callback(TranslationTraceCallback callback) {
  trace_callback_ = std::move(callback);
}

void BinaryTranslator::verify_rewrite_discharge(TranslatedCodeObject &result) const {
  result.rewrite_discharge_checked = true;
  if (!semantic_translator_->supports_rewrite_discharge()) {
    append_rewrite_discharge_error(
        result.diagnostics,
        "rewrite-discharge verification is unavailable for this translation profile");
    return;
  }

  try {
    AmdGpuCodeObject output(result.elf_bytes.data(), result.elf_bytes.size());
    if (!output.is_valid()) {
      append_rewrite_discharge_error(
          result.diagnostics,
          "rewrite-discharge verification could not parse the final code object");
      return;
    }
    if (!has_supported_executable_section_layout(output)) {
      append_rewrite_discharge_error(
          result.diagnostics,
          "rewrite-discharge verification requires exactly one allocated executable section "
          "named .text");
      return;
    }
    if (output.allocated_executable_sections().empty()) {
      result.rewrite_discharge_verified = true;
      return;
    }

    const IsaGpuTargetDescription *host_target =
        default_isa_target_registry().find_gpu_target_by_elf_machine(target_mach_ & EF_AMDGPU_MACH);
    auto decoder = host_target == nullptr
                       ? Decoder::create(host_arch_)
                       : Decoder::create(default_isa_target_registry(), host_target->public_id);
    if (!decoder) {
      append_rewrite_discharge_error(
          result.diagnostics,
          "rewrite-discharge verification has no decoder for the final architecture");
      return;
    }

    const Section *text = output.text_sections().front();
    if (text->size() == 0) {
      result.rewrite_discharge_verified = true;
      return;
    }
    const auto image = std::span<const uint8_t>(result.elf_bytes);
    const auto text_bytes =
        std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(text->data()), text->size());
    if (text_bytes.size() % sizeof(uint32_t) != 0) {
      append_rewrite_discharge_error(
          result.diagnostics,
          "rewrite-discharge verification found a partial instruction word in final output",
          text_bytes.size() - text_bytes.size() % sizeof(uint32_t));
      return;
    }

    KernelDescriptorTranslator descriptor_parser(host_arch_, host_arch_);
    const auto descriptor_translations = descriptor_parser.translate_image(
        image, text->sectionOffset(), text->size(), KernelDescriptorTranslationOptions{},
        text->sectionHeaderIndex());
    auto block_leaders = kernel_block_leaders(descriptor_translations, text_bytes);

    for (const RelocationFunctionTable &table : discover_relocation_function_tables(output)) {
      for (const RelocationFunctionPointer &entry : table.entries)
        block_leaders.push_back(entry.target_text_offset);
    }

    const auto relocation_leaders = text_relocation_block_leaders(image, *text);
    if (!relocation_leaders) {
      append_rewrite_discharge_error(
          result.diagnostics,
          "rewrite-discharge verification could not recover final executable entries");
      return;
    }
    block_leaders.insert(block_leaders.end(), relocation_leaders->begin(),
                         relocation_leaders->end());
    std::ranges::sort(block_leaders);
    block_leaders.erase(std::ranges::unique(block_leaders).begin(), block_leaders.end());

    // Most residual predicates inspect only one decoded instruction. Scan those
    // with constant live memory and retain the full CFG path only when a rule
    // explicitly requires basic-block neighbors. Besides reducing peak RSS,
    // delaying CFG construction preserves the existing contextual behavior for
    // tensor, cluster-load, and IU8-WMMA sequence checks.
    std::vector<TranslationDiagnostic> streaming_diagnostics;
    const size_t instruction_word_count = text->size() / sizeof(uint32_t);
    internal::RewriteDischargeInstructionDecoder instruction_decoder(*decoder);
    if (!instruction_decoder.has_valid_lookahead_bound()) {
      append_rewrite_discharge_error(
          result.diagnostics,
          "rewrite-discharge verification decoder reported an invalid lookahead bound");
      return;
    }
    size_t instruction_word_index = 0;
    uint64_t instruction_offset = 0;
    size_t next_block_leader = 0;
    bool invalid_block_leader = false;
    bool needs_basic_block = false;
    while (instruction_word_index < instruction_word_count) {
      if (next_block_leader < block_leaders.size() &&
          block_leaders[next_block_leader] < instruction_offset) {
        invalid_block_leader = true;
        break;
      }

      uint32_t first_word = 0;
      std::memcpy(&first_word, text_bytes.data() + instruction_offset, sizeof(first_word));
      if (host_arch_ == ROCJITSU_CODE_ARCH_CDNA5 && first_word == 0) {
        ++instruction_word_index;
        instruction_offset += sizeof(uint32_t);
        continue;
      }

      if (next_block_leader < block_leaders.size() &&
          block_leaders[next_block_leader] == instruction_offset) {
        ++next_block_leader;
      }

      std::unique_ptr<Instruction> inst;
      util::StringDiagnostic decode_error;
      const auto decode_status = instruction_decoder.decode(
          text_bytes.subspan(instruction_offset), instruction_offset, inst, decode_error.emitter());
      if (decode_status == internal::RewriteDischargeDecodeStatus::InvalidLookaheadBound) {
        append_rewrite_discharge_error(
            result.diagnostics,
            "rewrite-discharge verification decoder reported an invalid lookahead bound");
        return;
      }
      if (decode_status == internal::RewriteDischargeDecodeStatus::InvalidEncoding) {
        append_rewrite_discharge_error(
            result.diagnostics,
            "rewrite-discharge verification failed to decode final output: " +
                decode_error.message(),
            instruction_offset);
        return;
      }
      if (decode_status == internal::RewriteDischargeDecodeStatus::InvalidInstructionSize) {
        append_rewrite_discharge_error(
            result.diagnostics,
            "rewrite-discharge verification decoded an invalid instruction size in final output",
            instruction_offset);
        return;
      }
      const size_t instruction_size = static_cast<size_t>(inst->size());
      const size_t decoded_words = instruction_size / sizeof(uint32_t);
      if (decode_status == internal::RewriteDischargeDecodeStatus::TruncatedInstruction) {
        append_rewrite_discharge_error(
            result.diagnostics,
            "rewrite-discharge verification found a truncated instruction in final output",
            instruction_offset, std::string(inst->mnemonic()));
        return;
      }
      if (semantic_translator_->residual_rewrite_needs_basic_block(*inst))
        needs_basic_block = true;
      if (semantic_translator_->instruction_local_residual_rewrite_applies(*inst)) {
        append_rewrite_discharge_error(streaming_diagnostics,
                                       "registered rewrite remains actionable in final output",
                                       inst->src_loc(), std::string(inst->mnemonic()));
      }
      instruction_word_index += decoded_words;
      instruction_offset += instruction_size;
    }

    invalid_block_leader |= next_block_leader != block_leaders.size();
    invalid_block_leader |= instruction_offset != text_bytes.size();
    if (invalid_block_leader) {
      append_rewrite_discharge_error(
          result.diagnostics,
          "rewrite-discharge verification found an invalid final executable entry");
      return;
    }

    if (!needs_basic_block) {
      const bool found_residual = !streaming_diagnostics.empty();
      result.diagnostics.insert(result.diagnostics.end(),
                                std::make_move_iterator(streaming_diagnostics.begin()),
                                std::make_move_iterator(streaming_diagnostics.end()));
      result.rewrite_discharge_verified = !found_residual;
      return;
    }

    util::StringDiagnostic decode_error;
    FailureOr<std::vector<std::unique_ptr<BasicBlock>>> block_result =
        BasicBlock::build(output, *decoder, host_arch_, decode_error.emitter(), block_leaders,
                          ExternalEntryPolicy::ExplicitOnly);
    if (block_result.failed()) {
      append_rewrite_discharge_error(
          result.diagnostics, "rewrite-discharge verification failed to decode final output: " +
                                  decode_error.message());
      return;
    }
    std::vector<std::unique_ptr<BasicBlock>> blocks = std::move(block_result).value();
    if (has_invalid_block_leader(blocks, block_leaders, text->size())) {
      append_rewrite_discharge_error(
          result.diagnostics,
          "rewrite-discharge verification found an invalid final executable entry");
      return;
    }

    bool found_residual = false;
    for (const auto &block : blocks) {
      for (const Instruction &inst : block->instructions()) {
        if (!semantic_translator_->residual_rewrite_applies(inst))
          continue;
        found_residual = true;
        append_rewrite_discharge_error(result.diagnostics,
                                       "registered rewrite remains actionable in final output",
                                       inst.src_loc(), std::string(inst.mnemonic()));
      }
    }

    result.rewrite_discharge_verified = !found_residual;
  } catch (const std::exception &error) {
    append_rewrite_discharge_error(
        result.diagnostics,
        std::string("rewrite-discharge verification failed to decode final output: ") +
            error.what());
  }
}

TranslatedCodeObject BinaryTranslator::translate(const AmdGpuCodeObject &obj) {
  TranslatedCodeObject result = translate_impl(obj);
  if (options_.verify_rewrite_discharge && result.ok())
    verify_rewrite_discharge(result);
  return result;
}

TranslatedCodeObject BinaryTranslator::translate_impl(const AmdGpuCodeObject &obj) {
  TranslatedCodeObject result;
  result.host_arch = host_arch_;

  // Deferred-family warnings are suppressed per translation, not per translator
  // instance, so a reused translator still reports each code object's gaps.
  reported_deferred_families_.clear();

  auto leave_unchanged = [&]() {
    const auto *image = reinterpret_cast<const uint8_t *>(obj.image_data());
    if (obj.image_size() != 0)
      result.elf_bytes.assign(image, image + obj.image_size());
    return result;
  };

  if (options_.verify_rewrite_discharge && options_.skip_failed_kernels) {
    append_error(result.diagnostics, DiagnosticKind::Legalization,
                 "rewrite-discharge verification cannot be combined with skip-failed-kernels");
    return leave_unchanged();
  }

  if (obj.image_size() < sizeof(Elf64_Ehdr)) {
    append_error(result.diagnostics, DiagnosticKind::ResourceLimit,
                 "code object is too small to contain an ELF header");
    return leave_unchanged();
  }
  if (!obj.is_valid()) {
    append_error(result.diagnostics, DiagnosticKind::ResourceLimit,
                 "translation could not parse input code object");
    return leave_unchanged();
  }

  const auto &registry = default_isa_target_registry();
  const IsaTargetDescriptor *object_target = registry.find(obj.target_id());
  rj_code_target_id_t effective_guest_target = ROCJITSU_CODE_TARGET_INVALID;
  if (object_target != nullptr && object_target->architecture_id == guest_arch_) {
    effective_guest_target = obj.target_id();
  } else if (const IsaTargetDescriptor *guest_descriptor = registry.find(guest_arch_)) {
    if (const IsaGpuTargetDescription *default_target =
            registry.find_default_gpu_target(*guest_descriptor))
      effective_guest_target = default_target->public_id;
  }

  // A same-target gfx1250 translation is direction-specific: A0 and B0 share
  // an ELF machine ID, so both revisions must be given. Do not apply this
  // revision contract to other concrete CDNA5 targets.
  if (guest_arch_ == ROCJITSU_CODE_ARCH_CDNA5 && host_arch_ == ROCJITSU_CODE_ARCH_CDNA5 &&
      effective_guest_target == ROCJITSU_CODE_TARGET_GFX1250 &&
      (target_mach_ & EF_AMDGPU_MACH) == EF_AMDGPU_MACH_AMDGCN_GFX1250) {
    if (options_.input_revision == ProcessorRevision::Unspecified ||
        options_.output_revision == ProcessorRevision::Unspecified) {
      append_error(result.diagnostics, DiagnosticKind::Legalization,
                   "gfx1250 same-target translation requires both input and output silicon "
                   "revisions");
      return leave_unchanged();
    }
    // Only the B0-to-A0 direction is implemented.
    if (options_.input_revision == ProcessorRevision::Gfx1250A0 &&
        options_.output_revision == ProcessorRevision::Gfx1250B0) {
      append_error(result.diagnostics, DiagnosticKind::Legalization,
                   "gfx1250 A0-to-B0 translation is not supported");
      return leave_unchanged();
    }
  }

  if (!has_supported_executable_section_layout(obj)) {
    append_error(result.diagnostics, DiagnosticKind::Legalization,
                 "translation requires exactly one allocated executable section named .text");
    return leave_unchanged();
  }

  CodeObjectPatcher patcher(obj);

  // Reject relocation-section metadata ROCr cannot materialize before examining
  // individual records. ROCr then processes target-less SHT_RELA sections as
  // dynamic relocations and rejects R_AMDGPU_NONE there; reject that form by
  // record identity before place, symbol, executable-entry, or data-only handling.
  if (patcher.has_malformed_rocr_relocation_section()) {
    append_error(result.diagnostics, DiagnosticKind::Legalization,
                 "code object has malformed ROCr relocation-section metadata; a nonzero "
                 "SHT_NULL or out-of-range sh_info cannot identify a relocation target");
    return leave_unchanged();
  }

  if (patcher.has_rocr_rejected_none_relocation()) {
    append_error(result.diagnostics, DiagnosticKind::Legalization,
                 "code object has R_AMDGPU_NONE outside a valid explicit-target relocation "
                 "section; ROCr rejects this dynamic relocation form");
    return leave_unchanged();
  }

  auto text = patcher.text_bytes();
  if (auto refusal =
          translation_refusal(patcher, guest_arch_, host_arch_, target_mach_, options_)) {
    result.diagnostics.push_back(std::move(*refusal));
    return leave_unchanged();
  }

  std::unordered_set<uint64_t> generated_island_pool_candidates;
  if (guest_arch_ == host_arch_)
    generated_island_pool_candidates = generated_branch_island_pool_offsets(text, guest_arch_);

  // Per-kernel text relocation strategy:
  // 1. Translate descriptors first so their source entries and ABI state define
  //    the normal and sidecar kernel scopes.
  // 2. Decode .text, recover bounded static indirect targets, and form each
  //    scope from ordinary CFG successors plus validated call edges.
  // 3. Emit each scope into a source-ordered local body. Semantic expansions
  //    grow that body and control transfers reserve explicit patch windows.
  // 4. Place entry stubs and bodies in final .text coordinates, then repair
  //    direct transfers, recovered indirect transfers, and their PC builders.
  // 5. Feed discovered register/private-memory requirements back into each
  //    descriptor and commit .text, descriptors, sidecars, metadata, and flags.
  // BinaryTranslator's explicit guest_arch remains authoritative when legacy
  // or synthetic code-object metadata names a different ISA family. Preserve
  // concrete identity whenever the ELF target belongs to that family; otherwise
  // use the family's architecture default, which is fail-closed for CDNA5.
  auto decoder = effective_guest_target == ROCJITSU_CODE_TARGET_INVALID
                     ? Decoder::create(guest_arch_)
                     : Decoder::create(registry, effective_guest_target);
  if (!decoder) {
    append_error(result.diagnostics, DiagnosticKind::UnsupportedGuestArch,
                 "unsupported guest_arch: no decoder available");
    return leave_unchanged();
  }
  if (is_gfx1250_b0_to_a0())
    decoder = std::make_unique<Gfx1250B0ToA0CanonicalizingDecoder>(std::move(decoder));

  // Phase 1: descriptor translation gives DBT the source kernel roots and any
  // target descriptor/prologue bytes that must be materialized with the body.
  const bool skip_failed_kernels = options_.skip_failed_kernels;
  KernelDescriptorTranslator descriptor_translator(guest_arch_, host_arch_);
  const bool can_emit_sidecar_descriptors = supports_virtual_lds_sidecars(guest_arch_, host_arch_);
  KernelDescriptorTranslationOptions initial_descriptor_options;
  initial_descriptor_options.allow_oversized_lds = can_emit_sidecar_descriptors;
  auto descriptor_translations = descriptor_translator.translate_image(
      patcher.image_bytes(), patcher.text_offset(), patcher.text_size(), initial_descriptor_options,
      obj.text_sections().front()->sectionHeaderIndex());
  bool descriptors_supported = true;
  for (const auto &translation : descriptor_translations) {
    if (translation.supported || !skip_failed_kernels)
      append_diagnostics(result.diagnostics, translation.diagnostics);
    descriptors_supported &= translation.supported;
  }
  if (!descriptors_supported && !skip_failed_kernels) {
    append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                 "kernel descriptor translation requires unsupported resource or ABI "
                 "virtualization; leaving code object unchanged");
    return leave_unchanged();
  }

  if (descriptor_translations.empty()) {
    const bool descriptorless_gfx1250_b0_to_a0 =
        guest_arch_ == ROCJITSU_CODE_ARCH_CDNA5 && host_arch_ == ROCJITSU_CODE_ARCH_CDNA5 &&
        options_.input_revision == ProcessorRevision::Gfx1250B0 &&
        options_.output_revision == ProcessorRevision::Gfx1250A0;
    if (!descriptorless_gfx1250_b0_to_a0) {
      append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                   "kernel descriptors are required for kernel-level translation");
      return leave_unchanged();
    }
    append_warning(result.diagnostics, DiagnosticKind::NothingToTranslate,
                   "code object has no kernel descriptors; leaving executable text unchanged");
    return leave_unchanged();
  }

  const auto relocation_function_tables = discover_relocation_function_tables(obj);
  auto block_leaders = kernel_block_leaders(descriptor_translations, text);
  std::vector<uint64_t> required_external_entries;
  for (const RelocationFunctionTable &table : relocation_function_tables) {
    for (const RelocationFunctionPointer &entry : table.entries)
      required_external_entries.push_back(entry.target_text_offset);
  }
  const auto relocation_leaders =
      text_relocation_block_leaders(patcher.image_bytes(), *obj.text_sections().front());
  if (!relocation_leaders) {
    append_error(result.diagnostics, DiagnosticKind::Legalization,
                 "translation could not recover relocation-backed executable entries");
    return leave_unchanged();
  }
  required_external_entries.insert(required_external_entries.end(), relocation_leaders->begin(),
                                   relocation_leaders->end());
  std::ranges::sort(required_external_entries);
  required_external_entries.erase(std::ranges::unique(required_external_entries).begin(),
                                  required_external_entries.end());
  block_leaders.insert(block_leaders.end(), required_external_entries.begin(),
                       required_external_entries.end());
  std::ranges::sort(block_leaders);
  block_leaders.erase(std::ranges::unique(block_leaders).begin(), block_leaders.end());

  // A device function reached only through a pointer is named by no decoded edge, and AMDGPU pads
  // between functions with s_nop rather than ending the previous one with a terminator. Without
  // its symbol as a boundary that padding falls through into the function, putting the entry
  // inside a block -- which is not something a root can be adopted at or an entry state seeded
  // for.
  //
  // These are SPLIT POINTS, not external entries. Nearly all of them are ordinary helpers their
  // callers reach by a decoded call, and declaring those externally entered would discard the
  // incoming SGPR-pair facts that call establishes, turning otherwise recoverable getpc flows
  // unresolved.
  const auto text_function_symbol_offsets = discover_text_function_symbol_offsets(obj);
  // One of the two ways the kernarg admission below is satisfied. An object that defines no device
  // function has nothing a kernarg pointer could name locally, so the admission carries no
  // obligation at all. An object that DOES define them is no longer fenced off: the obligation --
  // that those bodies are adopted as roots rather than dropped -- is instead discharged directly,
  // by adopting every exported body whenever object_admits_kernarg_supplied_transfer holds. See
  // that predicate for why triggering on the admission's own fact keeps the adopted set stable
  // across passes.
  // Both walk the whole symbol table, and most objects never reach the code that needs them, so
  // pay for them on first use. Doing it eagerly cost about 2x the translation time across the
  // packaged-HSACO corpus -- enough on its own to exhaust the sanitizer job's budget.
  std::optional<bool> object_defines_no_device_function_cache;
  const auto object_defines_no_device_function = [&] {
    if (!object_defines_no_device_function_cache)
      object_defines_no_device_function_cache = object_defines_only_kernels(obj);
    return *object_defines_no_device_function_cache;
  };
  std::optional<std::vector<uint64_t>> externally_resolvable_cache;
  const auto externally_resolvable_text_function_offsets = [&]() -> const std::vector<uint64_t> & {
    if (!externally_resolvable_cache)
      externally_resolvable_cache = discover_externally_resolvable_text_function_offsets(obj);
    return *externally_resolvable_cache;
  };
  const auto relative_text_addend_targets =
      discover_relative_text_addend_targets(obj, text_function_symbol_offsets);
  std::vector<uint64_t> block_split_points = text_function_symbol_offsets;
  block_split_points.insert(block_split_points.end(), relative_text_addend_targets.begin(),
                            relative_text_addend_targets.end());
  std::ranges::sort(block_split_points);
  block_split_points.erase(std::ranges::unique(block_split_points).begin(),
                           block_split_points.end());

  // Phase 2: build a CFG over .text, including recovered indirect targets as
  // block leaders, then compute one source-reachable block set per descriptor
  // root. These sets are intentionally kernel-local: if two roots reach the same
  // helper block, Phase 3 emits that helper into both relocated bodies so every
  // branch or call target can be resolved through the current kernel's placement
  // map without borrowing another kernel's return continuation.
  std::vector<std::unique_ptr<BasicBlock>> blocks;
  util::StringDiagnostic decode_error;
  auto block_result =
      BasicBlock::build(obj, *decoder, guest_arch_, decode_error.emitter(), block_leaders,
                        ExternalEntryPolicy::ExplicitOnly, block_split_points);
  if (block_result.failed()) {
    append_error(result.diagnostics, DiagnosticKind::Legalization, decode_error.message());
    return leave_unchanged();
  }
  blocks = std::move(block_result).value();
  if (has_invalid_block_leader(blocks, block_leaders, text.size())) {
    append_error(result.diagnostics, DiagnosticKind::Legalization,
                 "translation found an invalid executable entry");
    return leave_unchanged();
  }
  const BlockOffsetIndex block_index = build_block_offset_index(blocks);
  const uint64_t text_vaddr = obj.text_sections().front()->vaddr();
  const auto relocation_pair_analysis =
      analyze_relocation_pairs(blocks, relocation_function_tables, text_vaddr);
  const auto &relocation_table_dispatches = relocation_pair_analysis.dispatches;
  const auto relocation_table_calls = attach_relocation_table_call_edges(
      block_index, relocation_function_tables, relocation_table_dispatches);

  // Address materializations that must survive relocation. See the use sites for how a data target
  // and a code target are each rewritten.
  const auto &pc_relative_address_builders = relocation_pair_analysis.address_builders;

  // Whether every code address this object can produce is one this translation will relocate.
  //
  // The translator's usual discipline is consumer-side: prove an indirect transfer's target or
  // refuse. That cannot see a target loaded from memory, because the value was placed there by a
  // relocation or by a store the transfer's scope never executed. The complementary producer-side
  // question is answerable: a code address can only enter this object as an ELF relocation addend
  // or as a getpc computation, so if every one of those is rewritten, then whatever a load yields
  // is relocation-correct by construction and there is nothing left for the consumer to prove.
  //
  // Relocation addends are discharged by relocate_relative_text_addends(), which already fails
  // closed on an addend it cannot map. What remains is the getpc side, and the denominator has to
  // be the discovery pass rather than the lattice: the lattice only models the 64-bit-literal add,
  // while a long-branch expansion builds its address from a split 32-bit pair the lattice never
  // sees. Requiring the lattice to have matched every discovered builder is what keeps that form
  // from silently falling outside the claim.
  // The claim below is "every code address THIS OBJECT PRODUCES is relocated", so it is worth
  // nothing unless the object produces some. An object holding no function-pointer table and
  // computing no code address can still reach an indirect transfer -- through a kernarg, or a
  // pointer handed over by a separately translated object -- and about those the claim is silent.
  // Requiring a producer keeps the permission from resting on a vacuous truth.
  const bool object_produces_code_addresses =
      std::ranges::any_of(
          relocation_function_tables,
          [](const RelocationFunctionTable &table) { return !table.entries.empty(); }) ||
      // A stored pointer produces a code address whether or not a symbol names the array holding
      // it. Omitting the anonymous ones made an object whose only producers are those slots look
      // producerless, so an unresolved indirect consumer took the rejection path even though this
      // pass discovers, adopts and relocates every one of those addends.
      !relative_text_addend_targets.empty();
  // Deliberately NOT counting in-text getpc builders. Admitting a transfer on this permission also
  // promises that every externally resolvable `.text` symbol still names its body, which is kept by
  // adopting those bodies as roots -- and adoption has to rest on facts read from the image, or the
  // adopted set differs between passes and `.text` moves. The discovery pass finds a different
  // number of getpc builders each time (it folds the ones it recovers), so counting them here would
  // let the promise be made where the adoption backing it cannot be, and materialization would then
  // refuse the object for a symbol nothing emitted.

  // Getpc producers whose builder range the recovered-indirect path rewrites.
  //
  // The rewrite lattice is not the only thing that relocates a PC-derived value. A long-branch
  // expansion -- getpc, split 32-bit add pair, setpc -- is recovered as an indirect transfer, and
  // the translator rewrites that consumer's whole builder range through pending_builder_fixups and
  // patch_recovered_builder_fixups. The lattice models only the 64-bit-literal add and deliberately
  // does not see the split pair, so crediting the lattice alone reports such a producer unaccounted
  // even though the recovery path leaves nothing stale in it. Teaching the lattice the split form
  // instead would put two rewriters on one range, which patch_recovered_builder_fixups forbids.
  std::unordered_set<uint64_t> recovery_rewritten_getpc_offsets;
  for (const auto &block : blocks) {
    if (block == nullptr)
      continue;
    for (const IndirectCallFixup &fixup : block->static_indirect_call_fixups())
      recovery_rewritten_getpc_offsets.insert(fixup.source_getpc_offset);
  }

  const bool code_addresses_fully_accounted = [&] {
    // Keyed by the value each add produces, not merely by the getpc that seeded it. One getpc can
    // feed several adds -- distinct branches materializing distinct addresses -- so treating the
    // getpc as covered because any one add was tracked would let a sibling add keep a stale code
    // literal while this predicate still reported everything accounted for.
    std::unordered_map<uint64_t, std::unordered_set<uint64_t>> lattice_targets_by_getpc;
    for (const PcRelativeAddressBuilder &builder : pc_relative_address_builders)
      lattice_targets_by_getpc[builder.source_getpc_offset].insert(builder.target_vaddr);

    for (const auto &block : blocks) {
      if (block == nullptr)
        continue;
      for (const PcAddressBuilder &builder : block->static_pc_address_builders()) {
        // A producer whose value this pass could not pin down, or whose arithmetic is not one
        // contiguous run, cannot be rewritten by anybody -- accept it only if the rewrite lattice
        // resolved the very same value, which is the evidence that a relocation was emitted for it.
        const auto targets = lattice_targets_by_getpc.find(builder.source_getpc_offset);
        const bool lattice_saw_getpc = targets != lattice_targets_by_getpc.end();
        // The recovery path rewrites this producer's whole range, so whatever it leaves behind is
        // already relocated and the lattice does not have to have tracked it. A poisoned producer
        // is still excluded: two irreconcilable values under one getpc mean a sibling add the
        // recovered consumer does not cover can keep a stale literal.
        const bool recovery_rewrites_builder =
            !builder.poisoned &&
            recovery_rewritten_getpc_offsets.contains(builder.source_getpc_offset);
        if (builder.resolved && builder.contiguous && builder.source_target_offset >= 0) {
          // Both analyses pinned a value, so demand they agree on it. Accepting the getpc merely
          // because some add under it was tracked is what would let a sibling add keep a stale
          // literal while this predicate still reported the object accounted for.
          if (lattice_saw_getpc &&
              targets->second.contains(text_vaddr +
                                       static_cast<uint64_t>(builder.source_target_offset))) {
            continue;
          }
          // A bare getpc carries no delta, so the two analyses are entitled to disagree about it:
          // this pass reports the value at the producer's block exit, which for a getpc whose add
          // lives in a successor is just the raw PC, while the lattice follows the chain to the
          // address that add finally produces. Neither reading names something stale -- hardware
          // supplies the relocated PC for the getpc itself, and the add consuming it is the one
          // the lattice tracked and will rewrite, which is what lattice_saw_getpc attests.
          // scope_relocatable_pc_builders makes the same carve-out for a bare producer.
          const bool bare_getpc_value =
              builder.source_recovery_begin_offset == builder.source_recovery_end_offset &&
              static_cast<uint64_t>(builder.source_target_offset) ==
                  builder.source_recovery_begin_offset;
          if (bare_getpc_value && lattice_saw_getpc)
            continue;
          if (recovery_rewrites_builder)
            continue;
        } else if (lattice_saw_getpc && !builder.poisoned) {
          // This pass gave up -- it only follows producers that reach a setpc -- but the rewrite
          // lattice did track the getpc, and the lattice is what emits the relocation. Its coverage
          // is the operative one.
          //
          // A poisoned producer is excluded because it is not the same situation. There the pass
          // did follow the getpc and saw two irreconcilable values, so a sibling add under it can
          // still carry a stale literal that the one value the lattice tracked does not cover.
          continue;
        } else {
          return false;
        }
        // A known non-code target needs nothing: it cannot be a stale jump destination.
        if (builder.source_target_offset < 0 ||
            static_cast<uint64_t>(builder.source_target_offset) >= text.size())
          continue;
        // A code target whose exact value the lattice never produced cannot be rewritten.
        return false;
      }
    }
    return true;
  }();
  // Read the section headers straight from the image rather than through all_sections(), which
  // drops SHT_NOBITS. A zero-initialized device global lives in .bss, so that omission would hide
  // the most ordinary target there is. Both classification and replacement consume the patcher's
  // validated table and shared resolver, so a reported data builder is always actionable.
  const auto source_section_headers = patcher.section_headers();

  // Callees reached through a relocation-table dispatch are explicit analysis
  // roots whose live-in SGPRs are caller-supplied, not architected: a dispatched
  // callee can be entered with an original .text pointer in a scalar argument.
  // The whole-scope stale-PC proof must therefore treat such a callee as an
  // unconstrained external entry rather than a safe entry-state root, even though
  // it has an in-scope CallEdge. Collect their block-start offsets so the gate
  // can fail closed for them (see scope_incomplete_roots_are_entry_state).
  std::unordered_set<uint64_t> relocation_table_callee_offsets;
  for (const RelocationTableDispatch &dispatch : relocation_table_dispatches) {
    if (dispatch.table_index >= relocation_function_tables.size())
      continue;
    if (!relocation_table_calls.contains(dispatch.source_call_offset))
      continue;
    for (const RelocationFunctionPointer &entry :
         relocation_function_tables[dispatch.table_index].entries)
      relocation_table_callee_offsets.insert(entry.target_text_offset);
  }
  // Bodies whose address a pointer can hold. Computed before the scopes because it decides which
  // callees a scope may clone, and derived only from input-fixed data -- relocation-table slots and
  // RELATIVE64 addends -- so the partition it produces is the same on every pass.
  std::unordered_set<uint64_t> address_taken_entries(relocation_table_callee_offsets.begin(),
                                                     relocation_table_callee_offsets.end());
  for (const uint64_t target : relative_text_addend_targets)
    address_taken_entries.insert(target);

  auto scopes = kernel_translation_scopes(blocks, block_index, descriptor_translations, {},
                                          &address_taken_entries);

  if (can_emit_sidecar_descriptors) {
    std::vector<KdTranslation> sidecar_variants;
    for (const KernelTranslationScope &scope : scopes) {
      if (scope.translation == nullptr)
        continue;
      const uint32_t host_lds_bytes = arch_lds_bytes(host_arch_);
      const bool static_lds_exceeds_host =
          host_lds_bytes != 0 && scope.translation->target_lds_size > host_lds_bytes;
      // Dynamic LDS is only known at dispatch time, so every LDS-using kernel
      // needs a virtual sidecar. The sidecar descriptor owns the wrapper ABI
      // and may enable a target-only kernarg segment pointer when the source
      // descriptor left room in the 16 initialized User SGPRs.
      if (!static_lds_exceeds_host &&
          !scope_uses_virtualizable_lds(scope, guest_arch_, host_arch_)) {
        continue;
      }

      KernelDescriptorTranslationOptions virtual_descriptor_options;
      virtual_descriptor_options.virtualize_lds = true;
      auto virtual_translation = descriptor_translator.translate_descriptor(
          patcher.image_bytes(), scope.translation->descriptor_file_offset,
          scope.translation->entry_text_offset, virtual_descriptor_options);
      if (!virtual_translation) {
        append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                     "virtual LDS sidecar descriptor could not be computed; leaving code object "
                     "unchanged",
                     scope.translation->entry_text_offset);
        return leave_unchanged();
      }
      virtual_translation->kernel_name = scope.translation->kernel_name;
      virtual_translation->sidecar_descriptor = true;
      sidecar_variants.push_back(std::move(*virtual_translation));
    }
    if (!sidecar_variants.empty()) {
      descriptor_translations.insert(descriptor_translations.end(),
                                     std::make_move_iterator(sidecar_variants.begin()),
                                     std::make_move_iterator(sidecar_variants.end()));
      scopes = kernel_translation_scopes(blocks, block_index, descriptor_translations, {},
                                         &address_taken_entries);
    }
  }

  // A device function whose address is produced only by a data relocation -- a C++ vtable slot is
  // the common case -- is named by no decoded edge, so no kernel scope reaches it. Translated text
  // replaces .text wholesale, so leaving it unreached drops the body and strands the addend that
  // points at it. Adopt those bodies as roots and rebuild the scopes so the ordinary lowering path
  // emits them and records their placement, which is what lets the addend be rewritten.
  //
  // The candidate set is the populated slots of the discovered function tables: a C++ vtable is an
  // STT_OBJECT holding symbol-less RELATIVE64 addends into .text, which is exactly what discovery
  // already qualifies. A compiler-anonymous pointer array carries no such symbol and so is not
  // covered here; that case still refuses, which is the pre-existing behavior.
  std::vector<uint64_t> adopted_roots;
  std::unordered_set<uint64_t> address_taken_offsets;
  // Set inside the adoption block below and read again by the kernarg admission further
  // down, which must rest on exactly the fact that adopted the roots satisfying it.
  bool object_admits_kernarg_supplied_transfer = false;
  {
    // How many scopes would emit each block on their own. A body reached by one scope already has
    // a single placement, so its address is unambiguous and nothing needs adopting. A body reached
    // by none has no placement at all, and one reached by several has no single placement -- a
    // runtime-dereferenced pointer holds one value and cannot choose between clones. Both of those
    // get a canonical copy, emitted once by the adopting scope, that every code address names.
    std::unordered_map<const BasicBlock *, unsigned> emitting_scopes;
    for (const KernelTranslationScope &scope : scopes) {
      for (const BasicBlock *block : scope.blocks)
        ++emitting_scopes[block];
    }

    const auto note_address_taken = [&](uint64_t text_offset) -> const BasicBlock * {
      const BasicBlock *entry = block_for_offset(block_index, text_offset);
      if (entry != nullptr)
        address_taken_offsets.insert(text_offset);
      return entry;
    };

    const auto adopt = [&](uint64_t text_offset) {
      const BasicBlock *entry = note_address_taken(text_offset);
      if (entry == nullptr)
        return;
      // block_for_offset() answers with the block CONTAINING the offset, so a target naming a
      // mid-function label -- or an offset inside an instruction -- would otherwise be adopted as
      // a root and seeded with the ABI's function-entry VGPR_MSB state it never actually has.
      // Require the target to be exactly where a block starts. Every adopt() caller shares this
      // guard so relocation slots, relative addends and getpc targets agree on what an entry is.
      // A rejected target stays noted as address-taken, so the code-address audit still fails
      // closed on it rather than silently dropping the reference.
      if (entry->start_offset() != text_offset)
        return;
      // Only a body no scope reaches needs adopting. One that several scopes clone is left where it
      // is: lowering it through an unrelated scope would deny it the context its real caller
      // supplies -- DS2 mode inference and liveness are scope-relative -- so the ambiguity between
      // clones is settled by naming one of them canonical instead of by making a fresh copy.
      if (emitting_scopes.contains(entry))
        return;
      adopted_roots.push_back(text_offset);
    };

    for (const RelocationFunctionTable &table : relocation_function_tables) {
      for (const RelocationFunctionPointer &slot : table.entries)
        adopt(slot.target_text_offset);
    }
    // Every stored function pointer names a body that has to exist after translation, whether or
    // not its holder was recognized as a table. relocate_relative_text_addends() rewrites each of
    // these addends to its target's new placement and refuses the object when a target was never
    // emitted, so adopting them is what turns that refusal into a translation.
    for (const uint64_t target : relative_text_addend_targets)
      adopt(target);
    // Admitting an unrecovered transfer promises that every externally resolvable `.text` symbol
    // still names its body afterwards -- that promise is what lets an outside holder of a code
    // address keep it. replace_text() enforces it and refuses when such a symbol has no mapping,
    // so a body no kernel reaches has to be adopted here or the promise cannot be kept. Symbols
    // are a fixed property of the input, so this leaves the partition a fixed point.
    // Gated on the two producers that are pure ELF facts -- a relocation function table and a
    // RELATIVE64 addend are read from the image, not recovered -- so the adopted set is the same on
    // every pass. code_addresses_fully_accounted must NOT appear here: it is analysis-derived, so
    // gating on it adopts a different set the second time round and grows `.text` between passes.
    // An object with neither producer can never claim the permission, so it never turns the strict
    // symbol requirement on and needs none of these roots.
    // And only when the object can actually reach the promise: the strict symbol requirement is
    // turned on by admitting an UNRECOVERED indirect transfer, so an object with no indirect
    // transfer at all can never turn it on. Skipping those keeps their scopes the size they were,
    // which matters because every later analysis is proportional to blocks emitted.
    const bool object_has_indirect_transfer =
        std::ranges::any_of(blocks, [](const std::unique_ptr<BasicBlock> &block) {
          if (block == nullptr)
            return false;
          const Instruction *term = block->terminator();
          return term != nullptr && (term->flags() & (INDIRECT_BRANCH | INDIRECT_CALL)) != 0;
        });
    // Same predicate the promise uses, so a permission is never granted without the roots
    // that satisfy it being adopted.
    // The kernarg admission below grants the same promise the producer permission does -- that
    // every externally resolvable `.text` symbol still names its body -- so it needs the same
    // roots, or replace_text() refuses the object for a body no scope reaches. rocPRIM's
    // trampoline_kernel is the case: it calls a device function pointer the host wrote into
    // kernarg, and its object also defines device-function bodies, one of which is an unreferenced
    // __cxa_pure_virtual stub that nothing emits.
    //
    // Trigger on the admission's own fact rather than a proxy. A kernarg-supplied target cannot be
    // recovered -- no dataflow fact names a value the host wrote -- so translation cannot fold such
    // a transfer into a direct call, and it survives verbatim into the output. The second pass
    // therefore sees the identical set and adopts identically. object_has_indirect_transfer is NOT
    // a substitute: it counts recovered transfers too, and those DO become direct calls, so gating
    // on it adopts less the second time round and moves `.text` (measured: 36 idempotence failures
    // across the packaged-kpack corpus).
    //
    // The three cheap tests run first because the fixpoint is expensive -- an InstDefUse and two
    // RegisterSet expansions per instruction, per predecessor edge, per sweep -- and running it for
    // every scope of every object measured as a 22% translation-time regression.
    object_admits_kernarg_supplied_transfer = [&] {
      if (externally_resolvable_text_function_offsets().empty() || !object_has_indirect_transfer)
        return false;
      if (std::ranges::none_of(descriptor_translations, [](const KdTranslation &translation) {
            return translation.source_has_kernarg_segment_ptr;
          }))
        return false;
      return std::ranges::any_of(scopes, [&](const KernelTranslationScope &scope) {
        return scope.translation != nullptr &&
               !kernarg_supplied_indirect_targets(scope.blocks, scope.entry, *scope.translation)
                    .empty();
      });
    }();
    if ((object_produces_code_addresses && object_has_indirect_transfer) ||
        object_admits_kernarg_supplied_transfer) {
      for (const uint64_t target : externally_resolvable_text_function_offsets())
        adopt(target);
    }

    // Everything adopted so far is input-fixed: relocation-table slots and RELATIVE64 addends are
    // discovered identically whether or not the object has been translated before. What they reach
    // is therefore knowable before any analysis-derived root is considered, and rebuilding the
    // scopes here is what lets the getpc loop below ask a meaningful question. Asked of the
    // pre-adoption scopes it is inert: in a device-linked object those are the kernel trampolines
    // and reach none of these bodies, so every candidate the old filter let through was adopted
    // whether it needed a root of its own or not.
    std::ranges::sort(adopted_roots);
    adopted_roots.erase(std::ranges::unique(adopted_roots).begin(), adopted_roots.end());
    const size_t pointer_root_count = adopted_roots.size();
    // The rebuild exists solely so the getpc loop below can ask "does the pointer closure already
    // emit this?". With no getpc naming an in-text address that loop adopts nothing, so the walk --
    // a full reachability pass per kernel over every block -- would be paid for an answer nobody
    // reads. Most objects are in that case, and under a sanitizer the difference is the corpus
    // job's time budget.
    const bool has_in_text_getpc_target = std::ranges::any_of(
        pc_relative_address_builders, [&](const PcRelativeAddressBuilder &builder) {
          return builder.target_vaddr >= text_vaddr &&
                 builder.target_vaddr - text_vaddr < text.size();
        });
    const bool rebuilt_for_pointer_roots = pointer_root_count != 0 && has_in_text_getpc_target;
    if (rebuilt_for_pointer_roots) {
      scopes = kernel_translation_scopes(blocks, block_index, descriptor_translations,
                                         adopted_roots, &address_taken_entries);
      emitting_scopes.clear();
      for (const KernelTranslationScope &scope : scopes)
        for (const BasicBlock *block : scope.blocks)
          ++emitting_scopes[block];
    }

    // A getpc that computes a code address makes that body address-taken just as surely as a
    // relocation slot does, so its target has to be accounted for before the rewrite is permitted.
    //
    // Whether it may also be ADOPTED turns on whether anything already emits it. A target reached
    // by some decoded edge from the input-fixed closure above is emitted by that closure and needs
    // no root of its own; one reached only through a pointer is not, and is adopted. That question
    // survives folding, because folding only ever replaces a recovered edge into a body with a
    // direct edge into the same body -- the closure is unchanged either way.
    //
    // Keying on whether recovery still had a consumer for the builder, as this once did, does not
    // survive it: pass one folds those consumers into one-word direct transfers, so the next pass
    // finds the same builders with no consumer, stops excluding them, and adopts thousands of
    // bodies the first pass had merely noted -- moving the scope partition and with it the whole
    // `.text` layout. This is the shape a separately compiled device library produces: an address
    // built with getpc, spilled to a VGPR lane, and later called through s_swap_pc_i64, so no
    // decoded edge names the body. Refusing those would refuse every device-linked object.
    for (const PcRelativeAddressBuilder &builder : pc_relative_address_builders) {
      if (builder.target_vaddr < text_vaddr || builder.target_vaddr - text_vaddr >= text.size())
        continue;
      const uint64_t target_text_offset = builder.target_vaddr - text_vaddr;
      // The addend path already intersects its targets with sized STT_FUNC symbols, because a
      // stored pointer is only known to name a function when a symbol says so. A recovered getpc
      // target carries no such evidence on its own -- the same construct materializes branch-table
      // entries and other intra-function labels -- so require a declared entry here too. Without
      // it, only note the address: that keeps the audit honest and leaves the conservative refusal
      // in place rather than adopting something that is not a function.
      if (!std::ranges::binary_search(text_function_symbol_offsets, target_text_offset)) {
        note_address_taken(target_text_offset);
        continue;
      }
      adopt(target_text_offset);
    }
    std::ranges::sort(adopted_roots);
    adopted_roots.erase(std::ranges::unique(adopted_roots).begin(), adopted_roots.end());
    // Phase A builds the scopes for the pointer roots only when it had a reason to. Repeat the
    // walk when it did not, or when the getpc loop added roots on top of what it built.
    if (!adopted_roots.empty() &&
        (!rebuilt_for_pointer_roots || adopted_roots.size() != pointer_root_count)) {
      scopes = kernel_translation_scopes(blocks, block_index, descriptor_translations,
                                         adopted_roots, &address_taken_entries);
    }
  }
  const std::unordered_set<uint64_t> adopted_root_offsets(adopted_roots.begin(),
                                                          adopted_roots.end());
  const std::unordered_set<uint64_t> adopted_return_offsets =
      adopted_root_return_offsets(block_index, adopted_roots, text);

  // An address consumed by loaded data must survive translation even when no direct branch reaches
  // it. Function-table targets may have been adopted into the emitted scopes above; fail before
  // lowering only when an entry remains outside every scope. The patcher would also reject its
  // missing offset mapping during commit, but enforcing the contract here keeps reachability,
  // emission, and verification in agreement and avoids doing a complete translation that cannot
  // commit.
  if (!required_external_entries.empty()) {
    std::unordered_set<uint64_t> uncovered_external_entries(required_external_entries.begin(),
                                                            required_external_entries.end());
    for (const KernelTranslationScope &scope : scopes) {
      for (const BasicBlock *block : scope.blocks) {
        if (block != nullptr)
          uncovered_external_entries.erase(block->start_offset());
      }
    }
    if (!uncovered_external_entries.empty()) {
      const uint64_t uncovered_entry = *std::ranges::min_element(uncovered_external_entries);
      append_error(result.diagnostics, DiagnosticKind::Legalization,
                   "relocation-backed executable entry is outside every kernel translation scope",
                   uncovered_entry);
      return leave_unchanged();
    }
  }

  const size_t expected_scope_count = kernel_translation_scope_count(descriptor_translations);
  if (scopes.size() != expected_scope_count) {
    append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                 "kernel descriptor entry offsets are required to map to decoded text blocks");
    return leave_unchanged();
  }

  std::vector<uint8_t> translated_text;
  translated_text.reserve(text.size());
  const bool continue_after_failure = options_.debug_continue_after_failure;

  auto queue_trace = [&](std::vector<PendingTrace> &pending, const Instruction &inst,
                         uint64_t offset, const InstructionLegalization *leg, bool copied_original,
                         bool semantic_lowering, bool changed, uint64_t target_offset,
                         std::vector<uint32_t> target_words) {
    if (!trace_callback_)
      return;
    pending.push_back({.source_offset = offset,
                       .source_size = static_cast<uint32_t>(inst.size()),
                       .source_words = raw_words_for_inst(inst),
                       .legalization = leg,
                       .copied_original = copied_original,
                       .semantic_lowering = semantic_lowering,
                       .changed = changed,
                       .target_offset = target_offset,
                       .target_words = std::move(target_words)});
  };

  auto flush_traces = [&](std::vector<PendingTrace> &pending, uint64_t target_delta) {
    if (!trace_callback_)
      return;
    for (PendingTrace &trace : pending) {
      trace_callback_({.source_offset = trace.source_offset,
                       .source_size = trace.source_size,
                       .source_words = trace.source_words,
                       .legalization = trace.legalization,
                       .copied_original = trace.copied_original,
                       .semantic_lowering = trace.semantic_lowering,
                       .changed = trace.changed,
                       .emitted_in_cave = false,
                       .target_offset = trace.target_offset + target_delta,
                       .target_words = trace.target_words});
    }
  };

  auto copy_original_instruction =
      [&](const Instruction &inst, uint64_t offset, std::vector<uint8_t> &kernel_text,
          std::vector<PendingTrace> &pending_traces, std::span<const uint32_t> suffix_words = {}) {
        const uint32_t inst_size = inst.size();
        const uint64_t target_offset = kernel_text.size();
        const auto *words = reinterpret_cast<const uint32_t *>(text.data() + offset);
        std::vector<uint32_t> copied_words(words, words + inst_size / sizeof(uint32_t));
        copied_words.insert(copied_words.end(), suffix_words.begin(), suffix_words.end());
        append_words(kernel_text, copied_words);
        // Continued-failure mode is diagnostic-only. Emit an explicit copy event so
        // diff reports make it clear which failed source instruction was preserved.
        queue_trace(pending_traces, inst, offset, nullptr, true, false, !suffix_words.empty(),
                    target_offset, std::move(copied_words));
      };

  auto continue_after_instruction_error = [&](const Instruction &inst, uint64_t offset,
                                              std::vector<uint8_t> &kernel_text,
                                              std::vector<PendingTrace> &pending_traces) {
    if (!continue_after_failure)
      return false;
    copy_original_instruction(inst, offset, kernel_text, pending_traces);
    return true;
  };

  auto emit_skipped_kernel = [&](const KernelTranslationScope &scope,
                                 KernelFailure failure) -> bool {
    assert(scope.translation != nullptr && "kernel scope should have descriptor translation");
    if (scope.blocks.empty()) {
      append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                   "cannot skip failed kernel without a decoded source block",
                   scope.translation->entry_text_offset);
      return false;
    }

    const uint64_t source_entry = scope.translation->entry_text_offset;
    const bool skipped_uses_virtual_lds = scope.translation->needs_lds_overflow_buf;
    auto skipped_text = append_skipped_kernel_stub(
        translated_text,
        {.source_entry = scope.translation->entry_text_offset,
         .has_kernarg_preload_firmware_skip = scope.translation->has_kernarg_preload_firmware_skip},
        host_arch_);
    if (!skipped_text.ok) {
      append_error(result.diagnostics, materialization_diagnostic_kind(skipped_text),
                   skipped_text.message, skipped_text.source_offset);
      return false;
    }

    for (KdTranslation &translation : descriptor_translations) {
      if (translation.entry_text_offset != source_entry)
        continue;
      // Normal hardware-LDS and virtual-LDS sidecar descriptors share the same
      // source entry, but a sidecar translation failure must not turn a normal
      // descriptor that can still launch on hardware LDS into a no-op. Skip
      // together any descriptor of the failing variant. Additionally, when the
      // sidecar variant is the one failing, a normal descriptor whose static LDS
      // only fit because a sidecar was promised is itself undispatchable (its
      // advertised LDS would fault on the host), so it must be stubbed too.
      const bool same_variant = translation.needs_lds_overflow_buf == skipped_uses_virtual_lds;
      const bool orphaned_by_sidecar_failure = skipped_uses_virtual_lds &&
                                               !translation.needs_lds_overflow_buf &&
                                               translation.static_lds_requires_sidecar;
      if (!same_variant && !orphaned_by_sidecar_failure)
        continue;
      translation.target_entry_text_offset = skipped_text.target_entry;
      translation.target_body_entry_text_offset = skipped_text.target_body_entry;
      // A skipped descriptor must describe the target stub, not the failed
      // guest kernel. Leaving oversized SGPR/LDS/private requirements in place
      // can make HIP fail during launch even though the entry points at safe
      // target code. Granulated zero encodes the minimum allocation bucket.
      translation.configure_skipped_stub();
    }

    std::string message =
        "*** SKIPPED KERNEL " + kernel_label(*scope.translation) +
        " REPLACED WITH S_ENDPGM; DISPATCHING IT WILL SILENTLY PRODUCE INVALID OUTPUTS *** "
        "Translation error: " +
        std::move(failure.message);
    append_warning(result.diagnostics, DiagnosticKind::KernelSkipped, std::move(message),
                   failure.guest_offset ? failure.guest_offset
                                        : std::optional<uint64_t>(source_entry),
                   std::move(failure.mnemonic), std::move(failure.required_work));
    return true;
  };

  // Per-scope relocation output. Declared before fail_or_skip_kernel so a skip can
  // truncate them back to their pre-scope sizes together with translated_text —
  // otherwise a scope that fails AFTER appending relocations (e.g. during branch
  // fixup or descriptor recompute) would leave stale entries whose source mapping
  // then applies to the replacement stub or a later kernel.
  std::vector<TextOffsetRelocation> text_relocations;
  std::vector<PcRelativeDataRelocation> data_relocations;
  // A code-target builder cannot be finished inside the scope loop: its target may belong to a
  // scope not yet emitted, and a body reached from several kernels is cloned once per scope, so the
  // final offset is only knowable once every placement is recorded. Hold the source-side answer and
  // resolve against the completed map below.
  std::vector<PendingCodeRelocation> pending_code_relocations;
  std::vector<PcRelativeTextRelocation> code_relocations;
  // Final offset of the one copy of each adopted body. Code addresses name this copy, so a body a
  // caller also clones still has exactly one address, and that address is stable no matter which
  // scope happens to hold a clone.
  std::unordered_map<uint64_t, uint64_t> canonical_placement;
  // Which kernel-scope variant supplied each canonical copy, and the offsets whose clones disagree
  // on that variant. See the nomination site for why a disagreement disqualifies the offset.
  std::unordered_map<uint64_t, bool> canonical_placement_is_sidecar;
  std::unordered_set<uint64_t> canonical_placement_variant_conflict;

  // Set when a scope hosting a canonical copy had to grow its own descriptor to lower it. See the
  // assignment for why that combination cannot be made safe from inside the scope loop.
  bool canonical_host_outgrew_its_descriptor = false;
  // Set when an unresolved indirect transfer was admitted only because every code address this
  // object produces is relocated. That argument also covers addresses arriving from outside, which
  // reach `.text` through a symbol, so while it is in force no `.text` symbol may keep its source
  // value -- hence the tightening handed to replace_text() below.
  bool relied_on_relocated_by_construction = false;

  // Reports the failure, or rolls the scope back and emits a stub in its place, depending on
  // skip_failed_kernels. Returns true when the caller should continue with the next scope.
  // Bound once so capture and rollback cannot be handed different views of the object state.
  const ObjectRollbackState rollback_state{
      .translated_text = translated_text,
      .text_relocations = text_relocations,
      .data_relocations = data_relocations,
      .code_relocations = code_relocations,
      .pending_code_relocations = pending_code_relocations,
      .canonical_placement = canonical_placement,
      .canonical_placement_is_sidecar = canonical_placement_is_sidecar,
      .canonical_placement_variant_conflict = canonical_placement_variant_conflict,
      .canonical_host_outgrew_its_descriptor = canonical_host_outgrew_its_descriptor,
      .descriptor_translations = descriptor_translations};

  auto fail_or_skip_kernel = [&](const KernelTranslationScope &scope, KernelFailure failure,
                                 const ScopeTransaction &transaction) -> bool {
    if (!skip_failed_kernels) {
      append_error(result.diagnostics, failure.kind, std::move(failure.message),
                   failure.guest_offset, std::move(failure.mnemonic),
                   std::move(failure.required_work));
      return false;
    }

    const uint64_t source_entry = scope.translation->entry_text_offset;
    if (!transaction.rollback(rollback_state)) {
      append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                   "descriptor checkpoint index changed during skip rollback", source_entry);
      return false;
    }

    KernelTranslationScope restored_scope = scope;
    restored_scope.translation = nullptr;
    for (KdTranslation &translation : descriptor_translations) {
      if (!same_kernel_scope_variant(translation, *scope.translation))
        continue;
      restored_scope.translation = &translation;
      break;
    }
    if (restored_scope.translation == nullptr) {
      append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                   "failed kernel descriptor was lost during skip rollback", source_entry);
      return false;
    }

    return emit_skipped_kernel(restored_scope, std::move(failure));
  };

  auto reserve_long_branch_sgpr_pair = [&](TranslationContext &context) -> std::optional<uint16_t> {
    auto base = next_long_branch_sgpr_pair(context, host_arch_);
    if (!base)
      return std::nullopt;
    context.require_sgprs(static_cast<uint32_t>(*base) + 2);
    return base;
  };

  for (const KernelTranslationScope &scope : scopes) {
    if (scope.blocks.empty())
      continue;
    assert(scope.translation != nullptr && "kernel scope should have descriptor translation");
    if (scope.translation->skipped)
      continue;

    ScopeTransaction transaction = ScopeTransaction::capture(rollback_state, *scope.translation);
    bool skip_scope = false;

    if (!scope.translation->supported) {
      auto failure = make_kernel_failure(
          DiagnosticKind::KernelDescriptor,
          "kernel descriptor translation requires unsupported resource or ABI virtualization");
      for (const TranslationDiagnostic &diagnostic : scope.translation->diagnostics) {
        if (diagnostic.severity != DiagnosticSeverity::Error)
          continue;
        failure.kind = diagnostic.kind;
        failure.message = diagnostic.message;
        failure.guest_offset = diagnostic.guest_offset;
        failure.mnemonic = diagnostic.mnemonic;
        failure.required_work = diagnostic.required_work;
        break;
      }
      if (fail_or_skip_kernel(scope, std::move(failure), transaction))
        continue;
      return leave_unchanged();
    }

    // Phase 3: translate this kernel into a temporary, source-ordered body. The
    // body starts at offset zero while it is being built; after final padding and
    // any launch window are chosen, every recorded target offset is rebased into
    // the output .text. This lets instruction expansions change block sizes
    // without precomputing speculative side-region offsets.
    KernelTextLayout layout;
    layout.source_entry = scope.translation->entry_text_offset;

    TranslationContext kernel_context(
        scope.translation->target_vgpr_count, scope.translation->target_agpr_count,
        scope.translation->target_accvgpr_base, scope.translation->target_sgpr_count,
        scope.translation->target_private_size, scope.translation->uses_dynamic_stack);
    if (scope.translation->needs_lds_overflow_buf) {
      auto virtual_lds_base =
          reserve_virtual_lds_base_sgpr_pair(kernel_context, KernelBlockScope(scope.blocks),
                                             *scope.translation, guest_arch_, host_arch_);
      if (!virtual_lds_base) {
        auto failure = make_kernel_failure(
            DiagnosticKind::ResourceLimit,
            "virtual LDS lowering cannot reserve a backing-buffer SGPR pair", layout.source_entry);
        if (fail_or_skip_kernel(scope, std::move(failure), transaction))
          continue;
        return leave_unchanged();
      }
      kernel_context.virtualize_lds = true;
      kernel_context.virtual_lds_base_sgpr = virtual_lds_base->base;
      kernel_context.virtual_lds_base_sgpr_spill_per_use = virtual_lds_base->spill_per_use;
      kernel_context.virtual_lds_kernarg_segment_ptr_sgpr =
          scope.translation->kernarg_segment_ptr_sgpr;
      kernel_context.virtual_lds_kernarg_pointer_offset =
          scope.translation->lds_overflow_kernarg_pointer_offset;
      scope.translation->virtual_lds_lowering.base_sgpr = virtual_lds_base->base;
      scope.translation->virtual_lds_lowering.prologue_temp_sgpr = virtual_lds_base->prologue_temp;
      scope.translation->virtual_lds_lowering.base_sgpr_spill_per_use =
          virtual_lds_base->spill_per_use;
      if (virtual_lds_base->spill_per_use) {
        const auto pointer_spill = kernel_context.reserve_persistent_semantic_spill_dwords(2);
        if (!pointer_spill) {
          auto failure = make_kernel_failure(
              DiagnosticKind::ResourceLimit,
              "virtual LDS backing-pointer spill offset overflows the 32-bit private segment",
              layout.source_entry);
          if (fail_or_skip_kernel(scope, std::move(failure), transaction))
            continue;
          return leave_unchanged();
        }
        kernel_context.virtual_lds_base_pointer_spilled = true;
        kernel_context.virtual_lds_base_pointer_spill_offset = *pointer_spill;
        scope.translation->virtual_lds_lowering.base_pointer_spilled = true;
        scope.translation->virtual_lds_lowering.base_pointer_spill_offset = *pointer_spill;
      }
      if (!append_virtual_lds_entry_prologue(*scope.translation, guest_arch_, host_arch_)) {
        auto failure = make_kernel_failure(
            DiagnosticKind::KernelDescriptor,
            "virtual LDS lowering cannot materialize backing-buffer pointer entry prologue",
            layout.source_entry);
        if (fail_or_skip_kernel(scope, std::move(failure), transaction))
          continue;
        return leave_unchanged();
      }
    }

    layout.entry_plan = {
        .has_kernarg_preload_firmware_skip = scope.translation->has_kernarg_preload_firmware_skip,
        .kernarg_preload_firmware_entry_text_offset =
            scope.translation->kernarg_preload_firmware_entry_text_offset,
        .prologue_words = scope.translation->prologue_words,
    };
    if (!kernarg_preload_launch_window_fits(layout.entry_plan)) {
      auto failure = make_kernel_failure(
          DiagnosticKind::KernelDescriptor,
          "kernel descriptor prologue does not fit in the 256-byte kernarg preload compatibility "
          "window",
          layout.source_entry);
      if (fail_or_skip_kernel(scope, std::move(failure), transaction))
        continue;
      return leave_unchanged();
    }
    const bool can_use_long_direct_branches =
        next_long_branch_sgpr_pair(kernel_context, host_arch_).has_value();
    const bool is_gfx1250_b0_to_a0_profile = is_gfx1250_b0_to_a0();
    std::unordered_map<uint64_t, const Instruction *> source_instruction_by_offset;
    std::unordered_map<uint64_t, const BasicBlock *> source_block_by_end_offset;
    for (BasicBlock *block : scope.blocks) {
      for (const Instruction &inst : block->instructions())
        source_instruction_by_offset.emplace(inst.src_loc(), &inst);
      source_block_by_end_offset.emplace(block->end_offset(), block);
    }

    // An adopted body is entered by a call through its address rather than by an edge from this
    // scope, so the architectural analyses have to be told it is an entry. Without that its blocks
    // are never reached by the forward fixed point and every VGPR_MSB query over them is
    // unanswerable, which fails any lowering that has to prove the mode.
    std::vector<BasicBlock *> scope_adopted_entry_blocks;
    if (!adopted_root_offsets.empty()) {
      for (BasicBlock *block : scope.blocks) {
        if (block != nullptr && adopted_root_offsets.contains(block->start_offset()))
          scope_adopted_entry_blocks.push_back(block);
      }
    }

    LivenessAnalysisOptions liveness_options;
    liveness_options.max_free_vgpr =
        static_cast<uint16_t>(isa_properties(host_arch_).max_addressable_vgprs_per_wf);
    liveness_options.arch = guest_arch_;
    liveness_options.entry_block = scope.entry;
    liveness_options.additional_entry_blocks = scope_adopted_entry_blocks;
    liveness_options.text = text;
    if (options_.debug_min_free_vgpr)
      liveness_options.min_free_vgpr = *options_.debug_min_free_vgpr;
    std::vector<const Instruction *> live_before_instructions;
    bool scope_requires_liveness = false;
    if (semantic_translator_ && semantic_translator_->has_rules()) {
      // Semantic expansion rules are the only BinaryTranslator path that queries
      // LivenessAnalysis. Other rewrites use separate resource strategies: virtual
      // LDS grows descriptor-backed registers or explicitly saves/restores borrowed
      // registers. Collect live-before snapshots only for rules that can query them,
      // and skip the kernel dataflow entirely when no such rule is present.
      for (BasicBlock *block : scope.blocks) {
        if (block == nullptr)
          continue;
        for (const Instruction &inst : block->instructions()) {
          if (semantic_translator_->rewrite_requires_liveness(inst)) {
            scope_requires_liveness = true;
            live_before_instructions.push_back(&inst);
          }
        }
      }
      liveness_options.restrict_live_before_to_instructions = true;
      liveness_options.live_before_instructions = std::span<const Instruction *const>(
          live_before_instructions.data(), live_before_instructions.size());
    }
    LivenessAnalysis liveness = LivenessAnalysis::unavailable();
    std::vector<ScopedCfgEdge> scope_analysis_edges;
    if (scope_requires_liveness) {
      scope_analysis_edges = scoped_call_liveness_edges(KernelBlockScope(scope.blocks), text);
      // DBT runs liveness without EXEC-state analysis: with a null ExecMaskAnalysis
      // every EXEC-masked vector def is treated as `Unknown` and so is never
      // promoted to a kill, keeping scratch allocation conservative.
      //
      // TODO: Once ExecMaskAnalysis is performance-optimized and ensures that it
      // doesn't have any waitalu/hazard analysis interference, build one here (seeded
      // with the descriptor entry plus, when present, the kernarg-preload firmware
      // entry at +256, which hardware enters with unknown EXEC) and pass it to
      // LivenessAnalysis so EXEC-masked vector defs can be promoted to kills where
      // EXEC is provably full, freeing more scratch registers. Reference issue #9733.
      liveness = LivenessAnalysis(KernelBlockScope(scope.blocks), /*exec=*/nullptr,
                                  liveness_options, scope_analysis_edges);
    }

    std::unique_ptr<Gfx1250VgprMsbAnalysis> wmma_completion_wait_vgpr_msb;
    auto needs_profile_wmma_completion_wait = [&](const Instruction &inst) {
      return is_gfx1250_b0_to_a0_profile && gfx1250_b0_to_a0_requires_wmma_completion_wait(inst);
    };
    auto append_profile_wmma_completion_wait_if_needed = [&](const Instruction &inst,
                                                             std::vector<uint32_t> &words) {
      // Liveness already built this exact analysis -- same scope, entry, edges, text and adopted
      // entries -- so reuse it rather than running the fixpoint a second time.
      const Gfx1250VgprMsbAnalysis *vgpr_msb = liveness.gfx1250_vgpr_msb();
      if (vgpr_msb == nullptr) {
        if (wmma_completion_wait_vgpr_msb == nullptr) {
          if (!scope_requires_liveness) {
            scope_analysis_edges = scoped_call_liveness_edges(KernelBlockScope(scope.blocks), text);
          }
          wmma_completion_wait_vgpr_msb = std::make_unique<Gfx1250VgprMsbAnalysis>(
              KernelBlockScope(scope.blocks), scope.entry, scope_analysis_edges, text,
              scope_adopted_entry_blocks);
        }
        vgpr_msb = wmma_completion_wait_vgpr_msb.get();
      }
      gfx1250_b0_to_a0_append_wmma_completion_wait_if_needed(inst, source_instruction_by_offset,
                                                             *vgpr_msb, words);
    };

    // Raw marker bytes are only candidates. Preserve a pool for this kernel
    // when either its marker and skip are adjacent decoded instructions in this
    // CFG scope, a reachable private slot proves that the kernel uses it, or a
    // reachable direct branch immediately before the pool skips exactly to the
    // first instruction after it. The latter two forms cover used and unused
    // pools whose headers became unreachable after relocation. Candidate
    // discovery already proved that every private slot is a one-word direct
    // branch; requiring reachable code here rejects marker-shaped literal data
    // and keeps preservation local to the kernel that owns the generated pool.
    constexpr uint64_t kGeneratedIslandPoolWords =
        kGeneratedIslandPoolHeaderWords + kDirectBranchIslandPoolSlots;
    constexpr uint64_t kGeneratedIslandPoolBytes =
        kGeneratedIslandPoolWords * static_cast<uint64_t>(sizeof(uint32_t));
    std::unordered_set<uint64_t> generated_island_pool_offsets;
    std::unordered_map<uint64_t, uint64_t> generated_island_pool_by_source_offset;
    if (guest_arch_ == host_arch_) {
      for (const uint64_t pool_offset : generated_island_pool_candidates) {
        const auto marker_it = source_instruction_by_offset.find(pool_offset);
        const auto skip_it = source_instruction_by_offset.find(pool_offset + sizeof(uint32_t));
        bool reachable_header = false;
        if (marker_it != source_instruction_by_offset.end() &&
            skip_it != source_instruction_by_offset.end()) {
          const Instruction *marker = marker_it->second;
          const Instruction *skip = skip_it->second;
          const auto skip_delta = skip->branch_offset_bytes();
          reachable_header =
              marker->size() == static_cast<int>(sizeof(uint32_t)) &&
              marker->raw_encoding() != nullptr &&
              marker->raw_encoding()[0] ==
                  build_s_nop(kBranchIslandPoolMarkerNopImmediate, guest_arch_) &&
              marker->next_instruction() == skip &&
              skip->size() == static_cast<int>(sizeof(uint32_t)) &&
              skip->raw_encoding() != nullptr &&
              skip->raw_encoding()[0] ==
                  build_s_branch(static_cast<int16_t>(kDirectBranchIslandPoolSlots), guest_arch_) &&
              skip_delta &&
              pool_offset + kGeneratedIslandPoolHeaderWords * sizeof(uint32_t) +
                      static_cast<uint64_t>(*skip_delta) ==
                  pool_offset + kGeneratedIslandPoolBytes;
        }

        bool reachable_slot = false;
        for (uint64_t word_index = kGeneratedIslandPoolHeaderWords;
             word_index < kGeneratedIslandPoolWords; ++word_index) {
          const uint64_t source_offset = pool_offset + word_index * sizeof(uint32_t);
          if (source_instruction_by_offset.contains(source_offset)) {
            reachable_slot = true;
            break;
          }
        }
        const uint64_t pool_end = pool_offset + kGeneratedIslandPoolBytes;
        const auto preceding_block_it = source_block_by_end_offset.find(pool_offset);
        const BasicBlock *preceding_block = preceding_block_it == source_block_by_end_offset.end()
                                                ? nullptr
                                                : preceding_block_it->second;
        const Instruction *preceding_terminator =
            preceding_block == nullptr ? nullptr : preceding_block->terminator();
        int64_t preceding_branch_target = -1;
        if (preceding_terminator != nullptr) {
          if (const auto delta = preceding_terminator->branch_offset_bytes()) {
            preceding_branch_target = static_cast<int64_t>(preceding_terminator->src_loc() +
                                                           preceding_terminator->size()) +
                                      static_cast<int64_t>(*delta);
          }
        }
        const bool reachable_skip_over_pool =
            preceding_terminator != nullptr && preceding_terminator->mnemonic() == "s_branch" &&
            source_instruction_by_offset.contains(pool_end) && preceding_branch_target >= 0 &&
            static_cast<uint64_t>(preceding_branch_target) == pool_end;
        if (!reachable_header && !reachable_slot && !reachable_skip_over_pool)
          continue;

        generated_island_pool_offsets.insert(pool_offset);
        for (uint64_t word_index = 0; word_index < kGeneratedIslandPoolWords; ++word_index) {
          generated_island_pool_by_source_offset.emplace(
              pool_offset + word_index * sizeof(uint32_t), pool_offset);
        }
        if (!reachable_header && !reachable_slot)
          generated_island_pool_by_source_offset.emplace(pool_end, pool_offset);
      }
    }
    // A recognized pool comes from an already translated body. Reuse that
    // deterministic grid instead of appending another set on the verification
    // pass. If preserved capacity is insufficient after an unexpected body
    // change, relocation fails closed; fixed-size pool slots are not widened.
    const bool preserve_generated_branch_island_pools = !generated_island_pool_offsets.empty();

    // Phase 4: translate each relocated body instruction at the current cursor.
    // Return-like s_setpc_b64 instructions are accepted only when they are the
    // terminator of a block reached from a validated call edge in this
    // kernel-local scope. Recovered indirect setpc/swappc consumers reserve a
    // compact window when recovery proves one effective target. A
    // marked long direct transfer generated by an earlier pass is instead
    // preserved as one exact getpc-through-consumer window. When one dynamic
    // consumer has multiple recovered targets, no single direct window can
    // preserve semantics; DBT keeps the original indirect consumer and asks the
    // patch layer to rewrite each source-side PC builder once.
    std::unordered_set<uint64_t> valid_call_return_offsets =
        scoped_call_return_offsets(KernelBlockScope(scope.blocks), text);
    // An adopted root's return is validated from its own body rather than from a call site, so it
    // is added here for whichever scope ended up carrying that body.
    valid_call_return_offsets.insert(adopted_return_offsets.begin(), adopted_return_offsets.end());
    // Only an unrecovered indirect transfer ever asks this, and most scopes have none, so run the
    // fixpoint on first question rather than for every scope.
    std::optional<std::unordered_set<uint64_t>> kernarg_supplied_cache;
    const auto kernarg_supplied_targets = [&]() -> const std::unordered_set<uint64_t> & {
      if (!kernarg_supplied_cache) {
        // Either the object defines no body a kernarg pointer could name, or the bodies it does
        // export were adopted above under the very same fact this admission rests on.
        kernarg_supplied_cache =
            scope.translation != nullptr &&
                    (object_defines_no_device_function() || object_admits_kernarg_supplied_transfer)
                ? kernarg_supplied_indirect_targets(scope.blocks, scope.entry, *scope.translation)
                : std::unordered_set<uint64_t>{};
      }
      return *kernarg_supplied_cache;
    };
    struct RecoveredConsumer {
      std::vector<IndirectCallFixup> fixups;
      bool use_transfer_window = false;
      bool preserve_marked_long_transfer = false;
      /// @brief The builder runs contiguously into this consumer and both are in this scope.
      ///
      /// @details Such a window may degenerate to the original dynamic transfer when the relocated
      /// target is out of direct-branch range, so it is pinned to the consumer's single word.
      bool builder_is_contiguous = false;
      uint64_t marked_window_begin = 0;
      uint64_t marked_window_end = 0;
      IndirectCallFixup window_fixup;
    };
    std::unordered_map<uint64_t, RecoveredConsumer> recovered_indirect_by_call;
    for (BasicBlock *block : scope.blocks) {
      for (const IndirectCallFixup &source_fixup : block->static_indirect_call_fixups()) {
        recovered_indirect_by_call[source_fixup.source_call_offset].fixups.push_back(source_fixup);
      }
    }

    // An incomplete consumer is only translatable when this scope can be proven
    // free of stale PC-derived values (see scope_relocatable_pc_builders). That
    // proof is expensive and only ever needed when such a consumer exists, so
    // establish it lazily; scopes without one keep byte-identical output.
    const bool has_incomplete_consumer =
        std::ranges::any_of(recovered_indirect_by_call, [](const auto &entry) {
          return std::ranges::any_of(entry.second.fixups, [](const IndirectCallFixup &fixup) {
            return fixup.source_incomplete;
          });
        });
    std::optional<std::vector<IndirectCallFixup>> whole_scope_builder_fixups;
    if (has_incomplete_consumer &&
        scope_incomplete_roots_are_entry_state(scope, relocation_table_callee_offsets))
      whole_scope_builder_fixups = scope_relocatable_pc_builders(scope.blocks);
    const bool no_stale_pc_values_in_scope = whole_scope_builder_fixups.has_value();

    std::vector<IndirectCallFixup> pending_builder_fixups;
    // Builders whose consumer became a transfer window. Rewritten when this scope emitted them,
    // skipped when it did not; see the window-conversion site for why absence is benign here.
    std::vector<IndirectCallFixup> pending_window_builder_fixups;
    // Indices over this scope's blocks, built once. The window recognizer below asks three
    // questions per recovered consumer -- is there a block starting strictly between the getpc and
    // the call, which block ends at the call, and which block starts at it -- and each was a linear
    // scan of scope.blocks. A device library gives one scope on the order of a million blocks and
    // tens of thousands of recovered consumers, so those scans dominated translation: they were
    // ~57% of samples in a perf profile of the 745 MB RCCL image.
    std::unordered_map<uint64_t, BasicBlock *> scope_block_by_start;
    std::unordered_map<uint64_t, BasicBlock *> scope_block_by_end;
    std::vector<uint64_t> scope_block_starts;
    scope_block_by_start.reserve(scope.blocks.size());
    scope_block_by_end.reserve(scope.blocks.size());
    scope_block_starts.reserve(scope.blocks.size());
    for (BasicBlock *candidate : scope.blocks) {
      if (candidate == nullptr)
        continue;
      scope_block_by_start.emplace(candidate->start_offset(), candidate);
      scope_block_by_end.emplace(candidate->end_offset(), candidate);
      scope_block_starts.push_back(candidate->start_offset());
    }
    std::ranges::sort(scope_block_starts);

    for (auto &[source_call_offset, consumer] : recovered_indirect_by_call) {
      if (consumer.fixups.empty())
        continue;

      const IndirectCallFixup &first = consumer.fixups.front();
      bool single_effective_target = true;
      for (const IndirectCallFixup &fixup : consumer.fixups) {
        if (fixup.source_call_sreg != first.source_call_sreg ||
            fixup.source_is_call != first.source_is_call ||
            fixup.source_return_sreg != first.source_return_sreg) {
          auto failure =
              make_kernel_failure(DiagnosticKind::Legalization,
                                  "recovered indirect branch has inconsistent consumer metadata",
                                  source_call_offset, "indirect branch");
          if (fail_or_skip_kernel(scope, std::move(failure), transaction)) {
            skip_scope = true;
            break;
          }
          return leave_unchanged();
        }
        if (fixup.source_target_offset != first.source_target_offset)
          single_effective_target = false;
      }
      if (skip_scope)
        break;

      // An incomplete fact means at least one predecessor left the PC pair
      // unconstrained. That path has no recovered builder or target to relocate,
      // yet its runtime SGPR pair may hold an original .text address; after DBT
      // relocates .text, the retained dynamic transfer would jump to stale or moved
      // bytes, and the unknown target block may be absent from the emitted scope.
      // Unless this scope was proven to contain no stale PC-derived value at all,
      // we cannot rule that out, so fail closed for the whole consumer rather
      // than relocate only the known builders.
      const bool any_incomplete = std::ranges::any_of(
          consumer.fixups, [](const IndirectCallFixup &fixup) { return fixup.source_incomplete; });
      if (any_incomplete && !no_stale_pc_values_in_scope) {
        auto failure = make_kernel_failure(
            DiagnosticKind::Legalization,
            "recovered indirect branch has an unconstrained predecessor path that cannot be "
            "relocated",
            source_call_offset, "indirect branch");
        if (fail_or_skip_kernel(scope, std::move(failure), transaction)) {
          skip_scope = true;
          break;
        }
        return leave_unchanged();
      }

      if (any_incomplete) {
        // The unconstrained path can still deliver a value this consumer never
        // reaches through a recovered builder, so a direct transfer window would
        // redirect it. Keep the dynamic transfer; every PC value it can observe
        // is relocation-correct under the whole-scope proof.
        continue;
      }

      // A complete consumer with one effective target can become a direct
      // window. Preserve a translator-generated long direct transfer as one
      // marked source window so a repeated pass cannot wrap its setpc/swappc
      // consumer a second time. A complete multi-target consumer keeps the
      // dynamic consumer and rewrites each source-side builder
      // (relocation/liveness).
      if (single_effective_target) {
        consumer.window_fixup = first;
        const bool contiguous_builder =
            consumer.fixups.size() == 1 && first.source_getpc_offset >= sizeof(uint32_t) &&
            first.source_recovery_begin_offset == first.source_getpc_offset + sizeof(uint32_t) &&
            builder_runs_into_consumer(text, first.source_recovery_end_offset,
                                       first.source_call_offset, guest_arch_) &&
            first.source_call_offset <= text.size() &&
            sizeof(uint32_t) <= text.size() - first.source_call_offset;
        const auto getpc_it = source_instruction_by_offset.find(first.source_getpc_offset);
        const Instruction *getpc =
            getpc_it == source_instruction_by_offset.end() ? nullptr : getpc_it->second;
        const Instruction *marker = getpc == nullptr ? nullptr : getpc->previous_instruction();
        const auto call_it = source_instruction_by_offset.find(first.source_call_offset);
        const Instruction *call =
            call_it == source_instruction_by_offset.end() ? nullptr : call_it->second;
        // Same three predicates as the scans they replace: first block start strictly inside
        // (getpc, call); the block ending exactly at the call whose start is at or before the
        // getpc; and the block starting exactly at the call.
        const auto interior_it =
            std::ranges::upper_bound(scope_block_starts, first.source_getpc_offset);
        const bool has_interior_block_entry =
            interior_it != scope_block_starts.end() && *interior_it < first.source_call_offset;
        const auto builder_entry = scope_block_by_end.find(first.source_call_offset);
        BasicBlock *builder_block =
            builder_entry != scope_block_by_end.end() &&
                    builder_entry->second->start_offset() <= first.source_getpc_offset
                ? builder_entry->second
                : nullptr;
        const auto call_entry = scope_block_by_start.find(first.source_call_offset);
        BasicBlock *call_block =
            call_entry != scope_block_by_start.end() ? call_entry->second : nullptr;
        // Recovered consumers are block leaders by construction, so their
        // canonical fallthrough block is expected. Reject only an additional
        // CFG predecessor that can enter at the consumer and bypass the builder.
        const bool has_external_call_entry =
            builder_block == nullptr || call_block == nullptr ||
            std::ranges::any_of(call_block->predecessors(), [&](const BasicBlock *predecessor) {
              return predecessor != builder_block;
            });
        const bool canonical_marked_window =
            contiguous_builder && marker != nullptr &&
            marker->size() == static_cast<int>(sizeof(uint32_t)) &&
            marker->src_loc() + sizeof(uint32_t) == first.source_getpc_offset &&
            marker->raw_encoding() != nullptr &&
            marker->raw_encoding()[0] ==
                build_s_nop(kLongDirectBranchMarkerNopImmediate, guest_arch_) &&
            call != nullptr && call->size() == static_cast<int>(sizeof(uint32_t)) &&
            !has_interior_block_entry && !has_external_call_entry;
        if (canonical_marked_window) {
          consumer.preserve_marked_long_transfer = true;
          consumer.marked_window_begin = marker->src_loc();
          consumer.marked_window_end = first.source_call_offset + sizeof(uint32_t);
        } else {
          consumer.use_transfer_window = true;
          // Both blocks come from this scope's own index, so the scope necessarily emits the
          // builder bytes and the consumer -- which is what makes the fail-closed routing below
          // safe rather than merely tidy.
          consumer.builder_is_contiguous =
              contiguous_builder && !has_interior_block_entry && !has_external_call_entry;
          // The window recomputes the address itself, so the builder feeding this consumer is now
          // dead -- but it is still emitted, and skipping the rewrite below would leave it holding
          // a delta measured against the body's old position. Harmless to run, since the window
          // supersedes it, yet indistinguishable on a later pass from a live producer of an
          // unrelocated code address, which is what made translation non-idempotent.
          //
          // Rewriting is the fix rather than erasing: patch_recovered_builder_fixups replaces the
          // range with the canonical getpc-delta form, which both names the relocated target and
          // is the shape the rewrite lattice models, so a later pass accounts for it instead of
          // reporting an unrelocated producer. It cannot collide with the window, which occupies
          // the consumer rather than the builder range, and owned_by_recovered_builder() keeps the
          // lattice from writing the same range twice.
          //
          // Kept apart from pending_builder_fixups because absence means something different here.
          // A consumer's own builder must be present or the rewrite it depends on is missing, but
          // recovery runs over all of `.text` while a scope emits only its own blocks, so a window
          // consumer can name a builder this scope never emitted. There is nothing stale in bytes
          // that were not written, and the scope that does emit them rewrites them itself.
          if (consumer.builder_is_contiguous) {
            // This window can degenerate to the original dynamic transfer, so its builder is
            // load-bearing rather than dead. pending_builder_fixups fails closed on a range it
            // cannot map; the window list silently skips, which would leave the transfer jumping
            // to a stale address with no diagnostic.
            pending_builder_fixups.insert(pending_builder_fixups.end(), consumer.fixups.begin(),
                                          consumer.fixups.end());
          } else {
            for (IndirectCallFixup window_builder : consumer.fixups) {
              window_builder.consumer_replaced_by_window = true;
              pending_window_builder_fixups.push_back(window_builder);
            }
          }
        }
      } else {
        pending_builder_fixups.insert(pending_builder_fixups.end(), consumer.fixups.begin(),
                                      consumer.fixups.end());
      }
    }
    if (skip_scope)
      continue;

    // Discharging the proof requires actually performing every rewrite it
    // assumes. Append it after the consumer-driven fixups so a builder that both
    // paths cover keeps the bytes the consumer path already produced today;
    // patch_recovered_builder_fixups collapses the duplicate range.
    if (no_stale_pc_values_in_scope) {
      pending_builder_fixups.insert(pending_builder_fixups.end(),
                                    whole_scope_builder_fixups->begin(),
                                    whole_scope_builder_fixups->end());
    }

    std::unordered_map<uint64_t, const RecoveredConsumer *> marked_long_transfer_by_start;
    for (const auto &[source_call_offset, consumer] : recovered_indirect_by_call) {
      (void)source_call_offset;
      if (consumer.preserve_marked_long_transfer)
        marked_long_transfer_by_start.emplace(consumer.marked_window_begin, &consumer);
    }

    std::vector<uint8_t> kernel_text;
    std::vector<PendingTrace> pending_traces;
    uint64_t source_body_size = 0;
    for (BasicBlock *block : scope.blocks)
      source_body_size += block->size();
    const uint64_t recovered_window_growth =
        recovered_indirect_by_call.size() * kMaxRecoveredIndirectTransferWords * sizeof(uint32_t);
    kernel_text.reserve(static_cast<size_t>(std::min<uint64_t>(
        source_body_size + recovered_window_growth, std::numeric_limits<size_t>::max())));

    std::unordered_map<uint64_t, uint64_t> target_offset_by_source_offset;
    target_offset_by_source_offset.reserve(
        static_cast<size_t>(source_body_size / sizeof(uint32_t)) + scope.blocks.size());
    layout.body_begin = 0;
    layout.blocks.reserve(scope.blocks.size());
    uint64_t next_branch_island_pool_offset = first_direct_branch_island_pool_offset();
    struct ActiveMarkedLongTransfer {
      uint64_t source_end = 0;
      // The window is reserved at a fixed size and regenerated into exactly those bytes, so the
      // source-to-target map is a pure translation of slope 1 across it. Keeping both ends lets a
      // block boundary that lands INSIDE the window be mapped affinely instead of falling back to
      // the running output size, which overshoots to the window's end.
      uint64_t source_begin = 0;
      uint64_t target_begin = 0;
    };
    std::optional<ActiveMarkedLongTransfer> active_marked_long_transfer;
    struct ActiveGeneratedIslandPool {
      uint64_t source_begin = 0;
      uint64_t source_end = 0;
      uint64_t target_begin = 0;
    };
    std::optional<ActiveGeneratedIslandPool> active_generated_island_pool;
    // reachable_kernel_blocks() materializes reached indices in source order.
    // Pool preservation relies on that ordering while one copied pool spans
    // several reachable slot blocks.
    assert(std::ranges::is_sorted(scope.blocks, [](const BasicBlock *lhs, const BasicBlock *rhs) {
      return lhs->start_offset() < rhs->start_offset();
    }));
    for (BasicBlock *block : scope.blocks) {
      std::optional<ActiveGeneratedIslandPool> block_generated_island_pool;
      if (active_generated_island_pool &&
          block->start_offset() < active_generated_island_pool->source_end) {
        block_generated_island_pool = active_generated_island_pool;
      }
      const uint64_t block_target_start =
          block_generated_island_pool
              ? block_generated_island_pool->target_begin +
                    (block->start_offset() - block_generated_island_pool->source_begin)
              : kernel_text.size();
      BlockPlacement placement{.block = block,
                               .source_start = block->start_offset(),
                               .source_end = block->end_offset(),
                               .target_start = block_target_start,
                               .target_end = block_target_start};

      for (auto it = block->instructions().begin(); it != block->instructions().end(); ++it) {
        const auto &inst = *it;
        const uint64_t offset = inst.src_loc();
        uint64_t target_offset = kernel_text.size();
        const uint32_t inst_size = inst.size();

        if (active_generated_island_pool && offset < active_generated_island_pool->source_end)
          continue;
        active_generated_island_pool.reset();

        if (const auto pool_it = generated_island_pool_by_source_offset.find(offset);
            pool_it != generated_island_pool_by_source_offset.end()) {
          const uint64_t pool_offset = pool_it->second;
          const uint64_t source_end = pool_offset + kGeneratedIslandPoolBytes;
          const bool source_instruction_in_pool = offset < source_end;
          // Candidate qualification already bounds the complete pool.
          assert(source_end <= text.size());
          for (uint64_t word_index = kGeneratedIslandPoolHeaderWords;
               word_index < kGeneratedIslandPoolWords; ++word_index) {
            const uint64_t source_branch_offset = pool_offset + word_index * sizeof(uint32_t);
            const auto source_branch_it = source_instruction_by_offset.find(source_branch_offset);
            // Unused placeholder slots are unreachable and absent from this
            // scope. Make those slots available to repair any new layout drift;
            // an unused slot remains s_branch 0 when no fixup allocates it.
            if (source_branch_it == source_instruction_by_offset.end()) {
              layout.branch_island_slots.push_back(target_offset + word_index * sizeof(uint32_t));
              continue;
            }
            const Instruction *source_branch = source_branch_it->second;
            const auto source_branch_delta = source_branch->branch_offset_bytes();
            assert(source_branch->raw_encoding() != nullptr && source_branch_delta &&
                   "candidate qualification proved every generated pool slot");
            if (source_branch->raw_encoding() == nullptr || !source_branch_delta) {
              auto failure = make_kernel_failure(
                  DiagnosticKind::Legalization,
                  "generated direct branch island pool contains malformed live slot",
                  source_branch_offset);
              if (fail_or_skip_kernel(scope, std::move(failure), transaction)) {
                skip_scope = true;
                break;
              }
              return leave_unchanged();
            }
            const int64_t source_target =
                static_cast<int64_t>(source_branch_offset + sizeof(uint32_t)) +
                static_cast<int64_t>(*source_branch_delta);
            if (source_target < 0 || static_cast<uint64_t>(source_target) > text.size()) {
              auto failure = make_kernel_failure(
                  DiagnosticKind::Legalization,
                  "generated direct branch island pool targets outside source .text",
                  source_branch_offset);
              if (fail_or_skip_kernel(scope, std::move(failure), transaction)) {
                skip_scope = true;
                break;
              }
              return leave_unchanged();
            }
            layout.branch_fixups.push_back(
                {.inst = source_branch,
                 .source_inst_offset = source_branch_offset,
                 .source_target_offset = static_cast<uint64_t>(source_target),
                 .target_inst_offset = target_offset + word_index * sizeof(uint32_t),
                 .target_window_bytes = sizeof(uint32_t),
                 .allow_window_growth = false,
                 .translated_words = {source_branch->raw_encoding()[0]}});
          }
          if (skip_scope)
            break;
          for (uint64_t source_word = pool_offset; source_word < source_end;
               source_word += sizeof(uint32_t)) {
            target_offset_by_source_offset.emplace(source_word,
                                                   target_offset + (source_word - pool_offset));
          }
          kernel_text.insert(kernel_text.end(),
                             text.begin() + static_cast<std::ptrdiff_t>(pool_offset),
                             text.begin() + static_cast<std::ptrdiff_t>(source_end));
          if (source_instruction_in_pool) {
            active_generated_island_pool = {
                .source_begin = pool_offset,
                .source_end = source_end,
                .target_begin = target_offset,
            };
            block_generated_island_pool = active_generated_island_pool;
            if (block->start_offset() >= pool_offset) {
              placement.target_start = target_offset + (block->start_offset() - pool_offset);
            }
            continue;
          }
          placement.target_start = kernel_text.size();
          target_offset = kernel_text.size();
        }

        if (active_marked_long_transfer && offset < active_marked_long_transfer->source_end) {
          // The patch layer regenerates the marked window, so its interior
          // source instructions have no stable one-to-one target offsets.
          if (offset + inst_size >= active_marked_long_transfer->source_end)
            active_marked_long_transfer.reset();
          continue;
        }
        active_marked_long_transfer.reset();

        if (const auto marked = marked_long_transfer_by_start.find(offset);
            marked != marked_long_transfer_by_start.end()) {
          const RecoveredConsumer &consumer = *marked->second;
          const IndirectCallFixup &source_fixup = consumer.window_fixup;
          const uint64_t window_bytes = consumer.marked_window_end - consumer.marked_window_begin;
          target_offset_by_source_offset.emplace(offset, target_offset);
          layout.recovered_indirect_fixups.push_back(
              {.source_call_offset = source_fixup.source_call_offset,
               .source_target_offset = source_fixup.source_target_offset,
               .target_window_offset = target_offset,
               .target_window_bytes = window_bytes,
               .target_sreg = source_fixup.source_call_sreg,
               .return_sreg = source_fixup.source_return_sreg,
               .is_call = source_fixup.source_is_call,
               .preserve_marked_long_transfer = true});
          append_nop_padding(kernel_text, window_bytes, host_arch_);
          active_marked_long_transfer = {
              .source_end = consumer.marked_window_end,
              .source_begin = consumer.marked_window_begin,
              .target_begin = target_offset,
          };
          continue;
        }

        // Ask the semantic translator directly whether this instruction has an
        // expand rule. The previous positional cursor into live_before_instructions
        // silently depended on that vector being built in the exact same block/
        // instruction iteration order as this loop; querying by encoding/opcode
        // removes that hidden coupling.
        const bool has_semantic_expand_rule =
            semantic_translator_ != nullptr &&
            semantic_translator_->has_expand_rule(inst.encoding_id(), inst.opcode());
        // Record every instruction start, not just recovered-PC builder
        // boundaries. The same final map keeps ELF labels attached to the
        // relocated instruction stream after semantic expansions change sizes.
        target_offset_by_source_offset.emplace(offset, target_offset);

        const auto recovered_it = recovered_indirect_by_call.find(offset);
        const bool has_recovered_indirect_call = recovered_it != recovered_indirect_by_call.end();
        const bool has_relocation_table_call = relocation_table_calls.contains(offset);
        const bool recovered_indirect_return = valid_call_return_offsets.contains(offset);
        const auto direct_branch_delta = inst.branch_offset_bytes();
        // A transfer through a value this object loaded from memory needs no per-target proof once
        // every code address the object can produce is relocated: the loaded word is either a
        // rewritten relocation addend or a rewritten getpc result, so it already names the body's
        // new home. This is what makes a C++ virtual call translatable -- its target comes out of a
        // vtable slot that no dataflow fact can name, and no amount of consumer-side analysis will
        // ever name it.
        //
        // The permission is object-wide because the question it answers is. A consumer cannot tell
        // where a loaded 64-bit word came from, so tying it to this transfer's provenance is not
        // available; what makes it sound is that both routes into `.text` are covered. Addresses
        // the object computes for itself are the ones code_addresses_fully_accounted ranges over.
        // An address supplied from outside -- a kernarg holding a device function pointer, or a
        // pointer another code object resolved -- can only have been obtained from a `.text`
        // symbol, and relocate_text_symbols() rewrites those and refuses the whole object when a
        // referenced text symbol has no exact offset map. Translation happens at load, so there is
        // no window in which a caller could have captured a pre-translation value.
        //
        // What that leaves is narrow and worth stating: a host that derives an entry from a symbol
        // plus a byte offset, and a symbol that is present but unreferenced, which
        // relocate_text_symbols() tolerates so debug labels in padding do not refuse the object.
        const bool target_is_relocated_by_construction =
            (object_produces_code_addresses && code_addresses_fully_accounted) ||
            kernarg_supplied_targets().contains(offset);
        if ((inst.flags() & (INDIRECT_BRANCH | INDIRECT_CALL)) != 0 &&
            !has_recovered_indirect_call && !has_relocation_table_call &&
            !recovered_indirect_return && !direct_branch_delta &&
            target_is_relocated_by_construction) {
          relied_on_relocated_by_construction = true;
        }
        if ((inst.flags() & (INDIRECT_BRANCH | INDIRECT_CALL)) != 0 &&
            !has_recovered_indirect_call && !has_relocation_table_call &&
            !recovered_indirect_return && !direct_branch_delta &&
            !target_is_relocated_by_construction) {
          auto failure = make_kernel_failure(
              DiagnosticKind::Legalization,
              "indirect branch or call target recovery is not implemented for relocated kernel "
              "text",
              offset, std::string(inst.mnemonic()));
          if (continue_after_failure && !skip_failed_kernels) {
            append_error(result.diagnostics, failure.kind, failure.message, failure.guest_offset,
                         failure.mnemonic, failure.required_work);
            if (continue_after_instruction_error(inst, offset, kernel_text, pending_traces)) {
              continue;
            }
          }
          if (fail_or_skip_kernel(scope, std::move(failure), transaction)) {
            skip_scope = true;
            break;
          }
          return leave_unchanged();
        }

        if (direct_branch_delta) {
          // Record direct branches while emitting the body, but patch only after
          // every block has a final target placement. This keeps fallthrough
          // implicit and limits fixups to explicit PC-relative edges. Emit the
          // branch into an initially compact patch window. Kernels with a legal
          // descriptor-backed SGPR pair can grow that window later if the final
          // target moves out of range; kernels already at the SGPR allocation
          // limit grow only an out-of-range conditional into the two-word
          // SGPR-free island form.
          const int64_t source_target =
              static_cast<int64_t>(offset + inst_size) + static_cast<int64_t>(*direct_branch_delta);
          if (source_target < 0) {
            auto failure =
                make_kernel_failure(DiagnosticKind::Legalization,
                                    "direct branch target is outside the source .text range",
                                    offset, std::string(inst.mnemonic()));
            if (continue_after_failure && !skip_failed_kernels) {
              append_error(result.diagnostics, failure.kind, failure.message, failure.guest_offset,
                           failure.mnemonic, failure.required_work);
              if (continue_after_instruction_error(inst, offset, kernel_text, pending_traces)) {
                continue;
              }
            }
            if (fail_or_skip_kernel(scope, std::move(failure), transaction)) {
              skip_scope = true;
              break;
            }
            return leave_unchanged();
          }
          const uint64_t branch_window_bytes = initial_direct_branch_patch_window_bytes(inst);

          if (!inst.raw_encoding()) {
            auto failure = make_kernel_failure(DiagnosticKind::Legalization,
                                               "direct branch is missing raw encoding", offset,
                                               std::string(inst.mnemonic()));
            if (fail_or_skip_kernel(scope, std::move(failure), transaction)) {
              skip_scope = true;
              break;
            }
            return leave_unchanged();
          }

          const InstructionLegalization *branch_leg = lookup_legalization(inst);
          const uint16_t branch_dst_opcode = branch_leg ? branch_leg->target_opcode : inst.opcode();

          bool copied_original = false;
          bool changed = false;
          std::vector<uint32_t> target_words;
          if (!handle_encoding(inst, offset, kernel_text, branch_dst_opcode, text, true,
                               copied_original, changed, target_words)) {
            if (continue_after_instruction_error(inst, offset, kernel_text, pending_traces))
              continue;
            return leave_unchanged();
          }
          layout.branch_fixups.push_back(
              {.inst = &inst,
               .source_inst_offset = offset,
               .source_target_offset = static_cast<uint64_t>(source_target),
               .target_inst_offset = target_offset,
               .target_window_bytes = branch_window_bytes,
               .translated_words = target_words});
          append_nop_padding(kernel_text, branch_window_bytes - inst.size(), host_arch_);
          queue_trace(pending_traces, inst, offset, branch_leg, copied_original, false, changed,
                      target_offset, std::move(target_words));
          continue;
        }

        if (has_recovered_indirect_call && recovered_it->second.use_transfer_window) {
          const IndirectCallFixup &source_fixup = recovered_it->second.window_fixup;
          layout.recovered_indirect_fixups.push_back(
              {.source_call_offset = source_fixup.source_call_offset,
               .source_target_offset = source_fixup.source_target_offset,
               .target_window_offset = target_offset,
               .target_sreg = source_fixup.source_call_sreg,
               .return_sreg = source_fixup.source_return_sreg,
               .is_call = source_fixup.source_is_call,
               .keep_dynamic_when_long = recovered_it->second.builder_is_contiguous});
          append_nop_padding(kernel_text, sizeof(uint32_t), host_arch_);
          continue;
        }

        const uint32_t *raw = inst.raw_encoding();
        if (!raw) {
          copy_original_instruction(inst, offset, kernel_text, pending_traces);
          continue;
        }

        const InstructionLegalization *leg = lookup_legalization(inst);

        // A deferred gfx1250 family has no A0 handling yet and stays on the copy
        // path, so the omission is reported rather than left silent. The report
        // describes the mnemonic, not the site, so one per mnemonic per
        // translation says everything a second copy would: an earlier RCCL
        // all_reduce run emitted 104,831 identical lines, burying the
        // diagnostics that do name a specific instruction. The record is
        // per-translation, so each code object still reports its own gaps.
        if (leg == nullptr && is_gfx1250_b0_to_a0() &&
            gfx1250_b0_to_a0_is_deferred_family(inst.mnemonic()) &&
            reported_deferred_families_.emplace(inst.mnemonic()).second) {
          append_warning(result.diagnostics, DiagnosticKind::Legalization,
                         "gfx1250 translation passes through '" + std::string(inst.mnemonic()) +
                             "' unchanged; target-specific handling is not yet implemented",
                         offset, std::string(inst.mnemonic()));
        }

        const uint16_t dst_opcode = leg ? leg->target_opcode : inst.opcode();

        // Try semantic lowering before raw encoding translation. A matched
        // semantic rule that cannot safely emit code is a translation error:
        // falling through would silently preserve guest semantics on the wrong
        // host ISA.
        if (has_semantic_expand_rule) {
          auto expansion =
              semantic_translator_->try_lower_expand(inst, offset, text, liveness, kernel_context);
          if (expansion.status == ExpandStatus::Failed) {
            auto failure = make_kernel_failure(
                DiagnosticKind::ExpandFailed,
                expansion.message.empty()
                    ? "semantic EXPAND rule matched, but could not safely lower"
                    : expansion.message,
                offset, std::string(inst.mnemonic()), std::move(expansion.required_work));
            if (continue_after_failure && !skip_failed_kernels) {
              append_error(result.diagnostics, failure.kind, failure.message, failure.guest_offset,
                           failure.mnemonic, failure.required_work);
              if (continue_after_instruction_error(inst, offset, kernel_text, pending_traces)) {
                continue;
              }
            }
            if (fail_or_skip_kernel(scope, std::move(failure), transaction)) {
              skip_scope = true;
              break;
            }
            return leave_unchanged();
          }

          if (expansion.status == ExpandStatus::Success) {
            std::vector<uint32_t> target_words = std::move(expansion.words);
            if (needs_profile_wmma_completion_wait(inst))
              append_profile_wmma_completion_wait_if_needed(inst, target_words);
            append_words(kernel_text, target_words);
            queue_trace(pending_traces, inst, offset, leg, false, true, true, target_offset,
                        std::move(target_words));
            continue;
          }
        }

        {
          auto virtual_lds_expansion =
              lower_virtual_lds_instruction(inst, kernel_context, guest_arch_, host_arch_);
          if (virtual_lds_expansion.status == ExpandStatus::Failed) {
            auto failure = make_kernel_failure(DiagnosticKind::ExpandFailed,
                                               virtual_lds_expansion.message.empty()
                                                   ? "virtual LDS lowering failed"
                                                   : virtual_lds_expansion.message,
                                               offset, std::string(inst.mnemonic()),
                                               std::move(virtual_lds_expansion.required_work));
            if (continue_after_failure && !skip_failed_kernels) {
              append_error(result.diagnostics, failure.kind, failure.message, failure.guest_offset,
                           failure.mnemonic, failure.required_work);
              if (continue_after_instruction_error(inst, offset, kernel_text, pending_traces)) {
                continue;
              }
            }
            if (fail_or_skip_kernel(scope, std::move(failure), transaction)) {
              skip_scope = true;
              break;
            }
            return leave_unchanged();
          }

          if (virtual_lds_expansion.status == ExpandStatus::Success) {
            std::vector<uint32_t> target_words = std::move(virtual_lds_expansion.words);
            append_words(kernel_text, target_words);
            queue_trace(pending_traces, inst, offset, leg, false, true, true, target_offset,
                        std::move(target_words));
            continue;
          }
        }

        // Non-opcode-keyed rewrites retain their historical position after
        // virtual-LDS lowering. Their registry supplies the same applicability
        // test to liveness collection and the corresponding final-stream check.
        if (semantic_translator_ != nullptr) {
          auto registered_expansion = semantic_translator_->try_lower_instruction_rewrite(
              inst, offset, text, liveness, kernel_context);
          if (registered_expansion.status == ExpandStatus::Failed) {
            auto failure = make_kernel_failure(
                DiagnosticKind::ExpandFailed, registered_expansion.message, offset,
                std::string(inst.mnemonic()), std::move(registered_expansion.required_work));
            if (continue_after_failure && !skip_failed_kernels) {
              append_error(result.diagnostics, failure.kind, failure.message, failure.guest_offset,
                           failure.mnemonic, failure.required_work);
              if (continue_after_instruction_error(inst, offset, kernel_text, pending_traces)) {
                continue;
              }
            }
            if (fail_or_skip_kernel(scope, std::move(failure), transaction)) {
              skip_scope = true;
              break;
            }
            return leave_unchanged();
          }
          if (registered_expansion.status == ExpandStatus::Success) {
            std::vector<uint32_t> target_words = std::move(registered_expansion.words);
            append_words(kernel_text, target_words);
            queue_trace(pending_traces, inst, offset, leg, false, true, true, target_offset,
                        std::move(target_words));
            continue;
          }
        }

        if (leg && leg->action == Action::Expand) {
          auto failure = make_kernel_failure(
              DiagnosticKind::ExpandMissing,
              "legalization requires EXPAND, but no expansion rule is implemented", offset,
              std::string(inst.mnemonic()), {"Add a semantic expansion rule for this mnemonic."});
          if (continue_after_failure && !skip_failed_kernels) {
            append_error(result.diagnostics, failure.kind, failure.message, failure.guest_offset,
                         failure.mnemonic, failure.required_work);
            if (continue_after_instruction_error(inst, offset, kernel_text, pending_traces)) {
              continue;
            }
          }
          if (fail_or_skip_kernel(scope, std::move(failure), transaction)) {
            skip_scope = true;
            break;
          }
          return leave_unchanged();
        }

        // A stepping-only translation has identical encodings for ordinary
        // instructions. Copy the authoritative source text, including literal
        // and modifier suffix words, instead of reconstructing bytes from the
        // decoder's base-format raw encoding. Direct branches and recovered
        // indirect transfers have already taken their relocation paths above;
        // explicit profile expansions have already continued or failed closed.
        if (guest_arch_ == host_arch_ && leg == nullptr) {
          if (needs_profile_wmma_completion_wait(inst)) {
            std::vector<uint32_t> suffix_words;
            append_profile_wmma_completion_wait_if_needed(inst, suffix_words);
            if (!suffix_words.empty()) {
              copy_original_instruction(inst, offset, kernel_text, pending_traces, suffix_words);
              continue;
            }
          }
          copy_original_instruction(inst, offset, kernel_text, pending_traces);
          continue;
        }

        // Cross-arch translation must have a legalization decision for every
        // opcode. When a lookup function exists (i.e. this is not a same-arch
        // identity pass) but the opcode is absent from the table, or the table
        // marks it Illegal, re-encoding it verbatim with the guest opcode number
        // would silently produce a different — possibly valid but wrong — host
        // instruction. Fail loudly instead of that silent passthrough. A null
        // lookup function means same-arch identity translation, where verbatim
        // copy is correct, so that path is intentionally not gated here.
        if (legalization_lookup_ != nullptr && (leg == nullptr || leg->action == Action::Illegal)) {
          auto failure = make_kernel_failure(
              DiagnosticKind::Legalization,
              leg == nullptr
                  ? "no legalization entry for this opcode on the target ISA; refusing to emit the "
                    "guest encoding verbatim"
                  : "legalization marks this opcode illegal on the target ISA",
              offset, std::string(inst.mnemonic()),
              {"Add a legalization/substitution/expansion entry for this mnemonic in the amdisa "
               "codegen pipeline."});
          if (continue_after_failure && !skip_failed_kernels) {
            append_error(result.diagnostics, failure.kind, failure.message, failure.guest_offset,
                         failure.mnemonic, failure.required_work);
            if (continue_after_instruction_error(inst, offset, kernel_text, pending_traces)) {
              continue;
            }
          }
          if (fail_or_skip_kernel(scope, std::move(failure), transaction)) {
            skip_scope = true;
            break;
          }
          return leave_unchanged();
        }

        bool copied_original = false;
        bool changed = false;
        std::vector<uint32_t> target_words;
        if (!handle_encoding(inst, offset, kernel_text, dst_opcode, text,
                             trace_callback_ != nullptr, copied_original, changed, target_words)) {
          if (continue_after_instruction_error(inst, offset, kernel_text, pending_traces)) {
            continue;
          }
          return leave_unchanged();
        }
        queue_trace(pending_traces, inst, offset, leg, copied_original, false, changed,
                    target_offset, std::move(target_words));
      }
      if (skip_scope)
        break;
      if (block->has_implicit_terminator()) {
        // Materialize the CFG boundary as part of the translated block. Like
        // any other target-side expansion, the terminator belongs in relocated
        // function extents. Without an architectural terminator, ordinary text
        // materialization can turn this unreachable stub into a fallthrough.
        const uint32_t endpgm = build_s_endpgm(host_arch_);
        append_words(kernel_text, std::span<const uint32_t>(&endpgm, 1));
        // The boundary is inferred from what follows the block, not from anything the block itself
        // says, so a body that really did run on past here is cut short instead of translated.
        // Name the offset: an execution that stops early is otherwise indistinguishable from one
        // that was always meant to, and nothing else in the output records that a decision was
        // made here.
        append_warning(result.diagnostics, DiagnosticKind::Legalization,
                       "fallthrough past this block reaches no decodable instruction; translated "
                       "with a synthesized s_endpgm boundary",
                       block->end_offset());
      }
      // A block can END inside a preserved marked long-transfer window: the recognizer only
      // accepts the window when some block's end_offset() is exactly the consumer offset, which is
      // window-interior. The window's own instructions are skipped during emission, so
      // kernel_text.size() has already advanced past the whole reserved window and would place
      // that boundary one consumer word too far. Map it affinely instead -- the window is
      // reserved and regenerated at a fixed size, so slope 1 is exact, and it agrees with
      // kernel_text.size() at the window end where that answer was already right.
      //
      // The stale entry was observable: a FUNC symbol ending at the consumer had its st_size
      // rounded up by one word on re-translation, and the same overshoot would misplace an
      // st_value or a RELATIVE64 addend naming that offset.
      const bool ends_inside_marked_window =
          active_marked_long_transfer &&
          block->end_offset() >= active_marked_long_transfer->source_begin &&
          block->end_offset() <= active_marked_long_transfer->source_end;
      placement.target_end =
          ends_inside_marked_window
              ? active_marked_long_transfer->target_begin +
                    (block->end_offset() - active_marked_long_transfer->source_begin)
          : block_generated_island_pool &&
                  block->end_offset() <= block_generated_island_pool->source_end
              ? block_generated_island_pool->target_begin +
                    (block->end_offset() - block_generated_island_pool->source_begin)
              : kernel_text.size();
      layout.blocks.push_back(placement);
      target_offset_by_source_offset.emplace(block->end_offset(), placement.target_end);
      if (!can_use_long_direct_branches && !preserve_generated_branch_island_pools &&
          block != scope.blocks.back() && kernel_text.size() >= next_branch_island_pool_offset) {
        append_direct_branch_island_pool(kernel_text, layout, host_arch_);
        next_branch_island_pool_offset = next_direct_branch_island_pool_offset(kernel_text.size());
      }
    }
    if (skip_scope)
      continue;
    layout.body_end = kernel_text.size();

    if (continue_after_failure && has_error_diagnostic(result.diagnostics))
      continue;

    // A preserved marked window is itself a getpc-plus-delta sequence, so re-translating our own
    // output finds a textbook builder inside one. Whoever claims it -- a multi-target consumer's
    // recovered fixup or the whole-scope proof -- the claim cannot be honored and must not be:
    // emission skips the window's interior precisely because the patch layer regenerates the whole
    // window from recovered_indirect_fixups with an already-relocated delta, so those source
    // offsets have no one-to-one target. Rewriting the range would fight the regeneration; there
    // is no stale value in bytes that were never emitted. Claiming them produced a spurious
    // "builder is not fully present in the relocated body" refusal on a second pass.
    // Collected once and binary-searched, not rescanned per fixup: a scope can hold tens of
    // thousands of recovered consumers, and the membership test below runs for every fixup.
    // Windows are disjoint, so sorting by start makes the containing candidate the last one that
    // begins at or before the query.
    std::vector<std::pair<uint64_t, uint64_t>> preserved_marked_windows;
    for (const auto &entry : recovered_indirect_by_call) {
      const RecoveredConsumer &consumer = entry.second;
      if (consumer.preserve_marked_long_transfer)
        preserved_marked_windows.emplace_back(consumer.marked_window_begin,
                                              consumer.marked_window_end);
    }
    std::ranges::sort(preserved_marked_windows);
    const auto inside_preserved_marked_window = [&](uint64_t getpc_offset) {
      const auto it = std::ranges::upper_bound(preserved_marked_windows, getpc_offset, {},
                                               &std::pair<uint64_t, uint64_t>::first);
      return it != preserved_marked_windows.begin() && getpc_offset < std::prev(it)->second;
    };
    for (IndirectCallFixup fixup : pending_builder_fixups) {
      if (inside_preserved_marked_window(fixup.source_getpc_offset))
        continue;
      const auto getpc_it = target_offset_by_source_offset.find(fixup.source_getpc_offset);
      const auto begin_it = target_offset_by_source_offset.find(fixup.source_recovery_begin_offset);
      const auto end_it = target_offset_by_source_offset.find(fixup.source_recovery_end_offset);
      if (getpc_it == target_offset_by_source_offset.end() ||
          begin_it == target_offset_by_source_offset.end() ||
          end_it == target_offset_by_source_offset.end()) {
        auto failure = make_kernel_failure(
            DiagnosticKind::Legalization,
            "recovered indirect branch builder is not fully present in the relocated body",
            fixup.source_call_offset, "indirect branch");
        if (fail_or_skip_kernel(scope, std::move(failure), transaction)) {
          skip_scope = true;
          break;
        }
        return leave_unchanged();
      }

      fixup.target_getpc_offset = getpc_it->second;
      fixup.target_recovery_begin_offset = begin_it->second;
      fixup.target_recovery_end_offset = end_it->second;
      layout.recovered_builder_fixups.push_back(fixup);
    }
    if (skip_scope)
      continue;

    for (IndirectCallFixup fixup : pending_window_builder_fixups) {
      if (inside_preserved_marked_window(fixup.source_getpc_offset))
        continue;
      const auto getpc_it = target_offset_by_source_offset.find(fixup.source_getpc_offset);
      const auto begin_it = target_offset_by_source_offset.find(fixup.source_recovery_begin_offset);
      const auto end_it = target_offset_by_source_offset.find(fixup.source_recovery_end_offset);
      if (getpc_it == target_offset_by_source_offset.end() ||
          begin_it == target_offset_by_source_offset.end() ||
          end_it == target_offset_by_source_offset.end()) {
        continue;
      }
      fixup.target_getpc_offset = getpc_it->second;
      fixup.target_recovery_begin_offset = begin_it->second;
      fixup.target_recovery_end_offset = end_it->second;
      layout.recovered_builder_fixups.push_back(fixup);
    }

    // Resolve control flow against the kernel-local body before materializing
    // it in the output section. Direct and recovered-indirect transfers both
    // start compact. If final placement proves a long transfer is required,
    // grow all requested windows in one rebuild and retry. Insertions can move a
    // later transfer across the range boundary, so repeat until the monotonic
    // layout is fixed.
    const auto patch_control_flow = [&]() {
      TextRelocationResult patched = patch_direct_branch_fixups(kernel_text, layout, host_arch_);
      if (!patched.ok)
        return patched;
      return patch_recovered_indirect_fixups(kernel_text, layout, host_arch_);
    };
    TextRelocationResult patched_control_flow = patch_control_flow();
    constexpr uint64_t kDirectGrowthWords = kMaxDirectBranchTransferWords - 1;
    constexpr uint64_t kRecoveredGrowthWords = kMaxRecoveredIndirectTransferWords - 1;
    // Each fixup grows monotonically and is capped at its format-specific
    // maximum. Its cumulative growth is therefore bounded by the maximum minus
    // the one-word initial window, no matter how many rounds request it. The sum
    // below is a proof-sized convergence budget. Exhaustion or an invalid
    // request is not constructible from a valid code object; keep the runtime
    // checks fail-closed against future patcher changes.
    uint64_t remaining_growth_words = 0;
    if (layout.branch_fixups.size() > std::numeric_limits<uint64_t>::max() / kDirectGrowthWords ||
        layout.recovered_indirect_fixups.size() >
            std::numeric_limits<uint64_t>::max() / kRecoveredGrowthWords) {
      remaining_growth_words = std::numeric_limits<uint64_t>::max();
    } else {
      const uint64_t direct_words = layout.branch_fixups.size() * kDirectGrowthWords;
      const uint64_t recovered_words =
          layout.recovered_indirect_fixups.size() * kRecoveredGrowthWords;
      remaining_growth_words = direct_words > std::numeric_limits<uint64_t>::max() - recovered_words
                                   ? std::numeric_limits<uint64_t>::max()
                                   : direct_words + recovered_words;
    }
    while (!patched_control_flow.ok) {
      if (patched_control_flow.reason != TextLayoutFailureReason::BranchOutOfRange)
        break;

      if (!patched_control_flow.required_windows.empty()) {
        uint64_t requested_growth_words = 0;
        bool valid_growth_budget = true;
        for (const ControlFlowWindowRequirement &requirement :
             patched_control_flow.required_windows) {
          uint64_t current_window_bytes = 0;
          uint64_t maximum_window_bytes = 0;
          if (requirement.kind == ControlFlowWindowKind::DirectBranch) {
            const auto fixup =
                std::ranges::find_if(layout.branch_fixups, [&](const BranchFixup &candidate) {
                  return candidate.source_inst_offset == requirement.source_inst_offset;
                });
            if (fixup == layout.branch_fixups.end()) {
              valid_growth_budget = false;
              break;
            }
            current_window_bytes = fixup->target_window_bytes;
            maximum_window_bytes = kMaxDirectBranchTransferWords * sizeof(uint32_t);
          } else {
            const auto fixup = std::ranges::find_if(
                layout.recovered_indirect_fixups, [&](const RecoveredIndirectFixup &candidate) {
                  return candidate.source_call_offset == requirement.source_inst_offset;
                });
            if (fixup == layout.recovered_indirect_fixups.end()) {
              valid_growth_budget = false;
              break;
            }
            current_window_bytes = fixup->target_window_bytes;
            maximum_window_bytes = kMaxRecoveredIndirectTransferWords * sizeof(uint32_t);
          }
          if (requirement.required_window_bytes <= current_window_bytes ||
              requirement.required_window_bytes > maximum_window_bytes) {
            valid_growth_budget = false;
            break;
          }
          requested_growth_words +=
              (requirement.required_window_bytes - current_window_bytes) / sizeof(uint32_t);
        }
        if (!valid_growth_budget || requested_growth_words > remaining_growth_words) {
          patched_control_flow = {
              .ok = false,
              .failure = TextLayoutFailureCategory::InvalidLayout,
              .source_offset = patched_control_flow.source_offset,
              .required_windows = {},
              .message = "control-flow layout growth did not converge",
              .compact_builder_fallbacks = {},
          };
          break;
        }

        const auto insertions = grow_control_flow_windows(
            kernel_text, layout, patched_control_flow.required_windows, host_arch_);
        if (!insertions) {
          patched_control_flow = {
              .ok = false,
              .failure = TextLayoutFailureCategory::InvalidLayout,
              .source_offset = patched_control_flow.source_offset,
              .required_windows = {},
              .message = "control-flow layout returned an invalid growth request",
              .compact_builder_fallbacks = {},
          };
          break;
        }
        // One entry per instruction in the scope, so rescanning the insertion list for each is
        // quadratic; share one prefix-summed rebaser across the whole sweep instead.
        const TextOffsetRebaser rebaser(*insertions);
        for (auto &[source_offset, target_offset] : target_offset_by_source_offset) {
          (void)source_offset;
          rebaser.rebase(target_offset);
        }
        for (PendingTrace &trace : pending_traces)
          rebaser.rebase(trace.target_offset);
        remaining_growth_words -= requested_growth_words;
      } else if (!layout.long_branch_sgpr) {
        auto sgpr = reserve_long_branch_sgpr_pair(kernel_context);
        if (!sgpr) {
          patched_control_flow = {
              .ok = false,
              .failure = TextLayoutFailureCategory::ResourceLimit,
              .source_offset = patched_control_flow.source_offset,
              .required_windows = {},
              .message =
                  "long direct branch requires an additional descriptor-backed SGPR pair after "
                  "semantic expansion",
              .compact_builder_fallbacks = {},
          };
          break;
        }
        layout.long_branch_sgpr = *sgpr;
      } else {
        break;
      }

      patched_control_flow = patch_control_flow();
    }
    if (!patched_control_flow.ok) {
      auto failure =
          make_kernel_failure(relocation_diagnostic_kind(patched_control_flow),
                              patched_control_flow.message, patched_control_flow.source_offset);
      if (fail_or_skip_kernel(scope, std::move(failure), transaction))
        continue;
      return leave_unchanged();
    }

    if (auto patched = patch_recovered_builder_fixups(kernel_text, layout, host_arch_);
        !patched.ok) {
      auto failure = make_kernel_failure(relocation_diagnostic_kind(patched), patched.message,
                                         patched.source_offset, "indirect branch");
      if (fail_or_skip_kernel(scope, std::move(failure), transaction))
        continue;
      return leave_unchanged();
    } else {
      // Say so here rather than let a later pass refuse the object with nothing naming the cause.
      // The object emitted now is correct; what it has lost is a builder the relocation lattice
      // can see, which only bites if an unrecovered indirect transfer also survives.
      for (const uint64_t source_call_offset : patched.compact_builder_fallbacks) {
        append_warning(result.diagnostics, DiagnosticKind::ResourceLimit,
                       "recovered indirect branch builder did not fit the canonical literal64 form "
                       "and fell back to the compact add; it is invisible to relocation analysis "
                       "on a later translation",
                       source_call_offset);
      }
    }

    auto materialized =
        append_relocated_kernel_text(translated_text, layout, kernel_text, host_arch_);
    if (!materialized.ok) {
      auto failure = make_kernel_failure(materialization_diagnostic_kind(materialized),
                                         materialized.message, materialized.source_offset);
      if (fail_or_skip_kernel(scope, std::move(failure), transaction))
        continue;
      return leave_unchanged();
    }
    const uint64_t target_delta = materialized.target_delta;
    // kernel_translation_scopes() gives the adopted roots to the first scope it builds, so that is
    // the scope whose placements are canonical.
    const bool scope_is_sidecar =
        scope.translation != nullptr && scope.translation->sidecar_descriptor;
    for (const uint64_t taken : address_taken_offsets) {
      const auto placed = target_offset_by_source_offset.find(taken);
      if (placed == target_offset_by_source_offset.end())
        continue;
      if (canonical_placement.try_emplace(taken, placed->second + target_delta).second) {
        transaction.relocations.canonical_placements_added.push_back(taken);
        canonical_placement_is_sidecar[taken] = scope_is_sidecar;
        continue;
      }
      // Nominating one clone canonical only works when the clones are interchangeable. A
      // virtual-LDS sidecar clone is lowered against a different LDS model than its hardware-LDS
      // twin, so a pointer that reached the wrong one would run the wrong lowering. Record the
      // disagreement and keep those offsets out of the canonical map so the addend path stays
      // fail-closed for exactly the case the clones are not equivalent.
      const auto variant = canonical_placement_is_sidecar.find(taken);
      if (variant != canonical_placement_is_sidecar.end() && variant->second != scope_is_sidecar &&
          canonical_placement_variant_conflict.insert(taken).second) {
        transaction.relocations.canonical_variant_conflicts_added.push_back(taken);
      }
    }
    for (const auto &[source_offset, target_offset] : target_offset_by_source_offset) {
      text_relocations.push_back(
          {.source_offset = source_offset, .target_offset = target_offset + target_delta});
    }
    // patch_recovered_builder_fixups NOPs and regenerates a builder's whole source range, so a
    // literal it owns must not also be written here -- the later write would land in a range the
    // other model has already rebuilt.
    // Only the fixups actually queued for patching own a literal. A consumer's candidate fixups are
    // not the same set: one whose target could not be resolved, or whose consumer was handled by a
    // direct-window conversion instead, never reaches patch_recovered_builder_fixups. Excluding
    // those here would leave their builders written by nobody.
    // Keyed by the byte range each queued recovery rebuilds, not by its getpc. One getpc can seed
    // several adds -- distinct branches materializing distinct addresses -- and only the add inside
    // a queued range is rewritten by that model. Keying on the shared getpc would suppress the
    // sibling's relocation too, leaving its literal measuring the distance the body used to be at.
    std::vector<std::pair<uint64_t, uint64_t>> recovered_builder_ranges;
    recovered_builder_ranges.reserve(layout.recovered_builder_fixups.size());
    for (const IndirectCallFixup &fixup : layout.recovered_builder_fixups) {
      recovered_builder_ranges.emplace_back(fixup.source_recovery_begin_offset,
                                            fixup.source_recovery_end_offset);
    }
    const auto owned_by_recovered_builder = [&](uint64_t source_add_offset) {
      return std::ranges::any_of(recovered_builder_ranges, [&](const auto &range) {
        return source_add_offset >= range.first && source_add_offset < range.second;
      });
    };

    std::unordered_set<uint64_t> patched_address_add_offsets;
    for (const RelocationTableDispatch &dispatch : relocation_table_dispatches) {
      if (!target_offset_by_source_offset.contains(dispatch.source_call_offset))
        continue;
      const auto getpc = target_offset_by_source_offset.find(dispatch.source_getpc_offset);
      const auto add = target_offset_by_source_offset.find(dispatch.source_address_add_offset);
      if (getpc == target_offset_by_source_offset.end() ||
          add == target_offset_by_source_offset.end()) {
        append_error(result.diagnostics, DiagnosticKind::Legalization,
                     "relocation-table GOT address builder is not fully present in the relocated "
                     "body",
                     dispatch.source_call_offset, "s_swap_pc_i64");
        return leave_unchanged();
      }
      data_relocations.push_back(
          {.target_getpc_offset = getpc->second + target_delta,
           .target_literal_offset = add->second + target_delta + sizeof(uint32_t),
           .source_target_vaddr = dispatch.source_table_address_vaddr});
      patched_address_add_offsets.insert(dispatch.source_address_add_offset);
    }

    // Every other getpc-plus-literal that names data needs the same treatment. The literal is the
    // distance from the getpc to its target, so relocating the body silently retargets it: a
    // device-library routine that reads its control block this way reaches into whatever now sits
    // at the old distance. The instructions are copied verbatim, so nothing else in the pipeline
    // notices. Recompute the literal from the getpc's final placement instead.
    //
    // A `.text` target is a code address and is handled below rather than here, because it moves
    // with the body that holds it and so cannot be named by a fixed virtual address.
    for (const PcRelativeAddressBuilder &builder : pc_relative_address_builders) {
      if (patched_address_add_offsets.contains(builder.source_address_add_offset))
        continue;
      const auto getpc = target_offset_by_source_offset.find(builder.source_getpc_offset);
      const auto add = target_offset_by_source_offset.find(builder.source_address_add_offset);
      // A builder outside this scope's emitted blocks belongs to another scope's copy and is
      // rewritten when that scope emits it.
      if (getpc == target_offset_by_source_offset.end() ||
          add == target_offset_by_source_offset.end()) {
        continue;
      }
      const bool target_is_in_text =
          builder.target_vaddr >= text_vaddr && builder.target_vaddr - text_vaddr < text.size();
      // When a data section ends exactly where .text begins, preserve the pre-endpoint-widening
      // classification: an address inside source text remains a code target.
      if (resolve_pc_relative_data_section_address(source_section_headers, builder.target_vaddr,
                                                   text_vaddr, text.size())) {
        data_relocations.push_back(
            {.target_getpc_offset = getpc->second + target_delta,
             .target_literal_offset = add->second + target_delta + sizeof(uint32_t),
             .source_target_vaddr = builder.target_vaddr});
        continue;
      }
      // A code address the builder merely computes -- stored to a function pointer, passed as an
      // argument -- reaches no indirect transfer here, so the recovered-builder path never claims
      // it and it would otherwise be copied verbatim with a literal measuring the distance the body
      // used to be at. Record it and resolve after every scope has been placed: the target may be
      // emitted by a different scope, so this scope's placement map cannot answer for it yet.
      if (!target_is_in_text)
        continue;
      if (owned_by_recovered_builder(builder.source_address_add_offset))
        continue;
      const uint64_t source_target = builder.target_vaddr - text_vaddr;
      // An adopted body has one canonical copy; every code address names it, whatever scope the
      // builder lives in. Resolving that here rather than after the loop keeps the address stable
      // even when a caller also clones the body. Offsets whose clones disagreed on kernel-scope
      // variant are excluded: those clones are lowered against different LDS models, so no single
      // one of them can answer for the rest.
      if (const auto canonical = canonical_placement.find(source_target);
          canonical != canonical_placement.end() &&
          !canonical_placement_variant_conflict.contains(source_target)) {
        code_relocations.push_back(
            {.target_getpc_offset = getpc->second + target_delta,
             .target_literal_offset = add->second + target_delta + sizeof(uint32_t),
             .target_text_offset = canonical->second});
        continue;
      }
      // Otherwise prefer this scope's own copy of the target. A body reached by several kernels is
      // cloned once per scope so each scope's branches resolve through its own placement map, and
      // patch_recovered_builder_fixups already points a recovered builder at its scope's clone.
      // Following the same convention keeps a computed address consistent with the code that
      // computed it, and avoids inventing a conflict between clones that are only distinguishable
      // by which scope emitted them.
      if (const auto local = target_offset_by_source_offset.find(source_target);
          local != target_offset_by_source_offset.end()) {
        code_relocations.push_back(
            {.target_getpc_offset = getpc->second + target_delta,
             .target_literal_offset = add->second + target_delta + sizeof(uint32_t),
             .target_text_offset = local->second + target_delta});
        continue;
      }
      // Otherwise the target belongs to another scope -- an adopted body, say -- and can only be
      // resolved once every placement is known.
      pending_code_relocations.push_back(
          {.target_getpc_offset = getpc->second + target_delta,
           .target_literal_offset = add->second + target_delta + sizeof(uint32_t),
           .source_target_text_offset = source_target});
    }

    if (kernel_context.required_vgpr_count > kernel_context.num_vgprs)
      scope.translation->target_vgpr_count = kernel_context.required_vgpr_count;
    if (kernel_context.required_sgpr_count > kernel_context.num_sgprs)
      scope.translation->target_sgpr_count = kernel_context.required_sgpr_count;
    if (kernel_context.required_private_segment_fixed_size >
        kernel_context.private_segment_fixed_size)
      scope.translation->target_private_size = kernel_context.required_private_segment_fixed_size;

    // Only this scope's descriptor is grown from the values above, but a canonical copy this scope
    // hosts is entered through a pointer any kernel can dereference. If lowering that copy needed
    // resources beyond what this descriptor started with, another kernel would enter it under a
    // budget that was never raised. Nothing here can raise that kernel's descriptor -- it may
    // already be translated, and its own recompute happens inside its own scope -- so record the
    // condition and refuse below rather than emit a descriptor that under-provisions a real caller.
    //
    // Only a resource the target descriptor actually encodes can under-provision a caller. On
    // GFX10+ the wavefront SGPR count is a reserved field, so the decoded 8 is the granule-0
    // artifact rather than a budget, and a long-branch thunk needing an SGPR pair above it raises
    // nothing and starves nobody. VGPR and private are encoded on every target and still count.
    if (!transaction.relocations.canonical_placements_added.empty() &&
        (kernel_context.required_vgpr_count > kernel_context.num_vgprs ||
         (arch_descriptor_encodes_sgpr_allocation(host_arch_) &&
          kernel_context.required_sgpr_count > kernel_context.num_sgprs) ||
         kernel_context.required_private_segment_fixed_size >
             kernel_context.private_segment_fixed_size)) {
      canonical_host_outgrew_its_descriptor = true;
    }

    if (scope.translation->target_vgpr_count != kernel_context.num_vgprs ||
        scope.translation->target_sgpr_count != kernel_context.num_sgprs ||
        scope.translation->target_private_size != kernel_context.private_segment_fixed_size) {
      // Semantic rules may allocate descriptor-backed scratch registers or
      // per-lane private spill slots beyond the kernel's original resources.
      // Recompute the descriptor with those larger minimums before patching it
      // into the output image.
      KernelDescriptorTranslationOptions descriptor_options;
      descriptor_options.minimum_vgprs = scope.translation->target_vgpr_count;
      descriptor_options.minimum_sgprs = scope.translation->target_sgpr_count;
      descriptor_options.private_segment_fixed_size_addend =
          scope.translation->target_private_size - kernel_context.private_segment_fixed_size;
      descriptor_options.virtualize_lds = scope.translation->needs_lds_overflow_buf;
      descriptor_options.allow_oversized_lds =
          can_emit_sidecar_descriptors && !scope.translation->needs_lds_overflow_buf;

      // Descriptor growth is intentionally done after instruction lowering so
      // each kernel is translated once. Only descriptors that enter this code
      // scope need the larger register counts; rescanning the whole image would
      // also recompute unrelated kernels and risks mixing diagnostics across
      // scopes.
      bool recomputed_descriptor = false;
      for (KdTranslation &translation : descriptor_translations) {
        if (!same_kernel_scope_variant(translation, *scope.translation))
          continue;

        auto updated = descriptor_translator.translate_descriptor(
            patcher.image_bytes(), translation.descriptor_file_offset,
            translation.entry_text_offset, descriptor_options);
        if (!updated) {
          auto failure =
              make_kernel_failure(DiagnosticKind::KernelDescriptor,
                                  "kernel descriptor translation could not be recomputed");
          if (fail_or_skip_kernel(scope, std::move(failure), transaction)) {
            skip_scope = true;
            break;
          }
          return leave_unchanged();
        }
        updated->kernel_name = translation.kernel_name;
        updated->sidecar_descriptor = translation.sidecar_descriptor;
        updated->virtual_lds_lowering = translation.virtual_lds_lowering;
        // Register feedback also recomputes ordinary, non-virtual descriptors.
        // Only a virtual-LDS variant owns a backing-pointer entry prologue;
        // asking an unrelated ISA pair to materialize one makes otherwise
        // valid SGPR/VGPR growth fail after semantic lowering.
        if (updated->needs_lds_overflow_buf &&
            !append_virtual_lds_entry_prologue(*updated, guest_arch_, host_arch_)) {
          auto failure = make_kernel_failure(
              DiagnosticKind::KernelDescriptor,
              "virtual LDS lowering cannot materialize backing-buffer pointer entry prologue");
          if (fail_or_skip_kernel(scope, std::move(failure), transaction)) {
            skip_scope = true;
            break;
          }
          return leave_unchanged();
        }

        if (!updated->supported) {
          if (skip_failed_kernels) {
            auto failure = make_kernel_failure(DiagnosticKind::KernelDescriptor,
                                               "kernel descriptor translation requires unsupported "
                                               "resource or ABI virtualization");
            for (const TranslationDiagnostic &diagnostic : updated->diagnostics) {
              if (diagnostic.severity != DiagnosticSeverity::Error)
                continue;
              failure.kind = diagnostic.kind;
              failure.message = diagnostic.message;
              failure.guest_offset = diagnostic.guest_offset;
              failure.mnemonic = diagnostic.mnemonic;
              failure.required_work = diagnostic.required_work;
              break;
            }
            if (fail_or_skip_kernel(scope, std::move(failure), transaction)) {
              skip_scope = true;
              break;
            }
          }
          append_diagnostics(result.diagnostics, updated->diagnostics);
          append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                       "kernel descriptor translation requires unsupported resource or ABI "
                       "virtualization; leaving code object unchanged");
          return leave_unchanged();
        }
        append_diagnostics(result.diagnostics, updated->diagnostics);

        if (updated->prologue_words != translation.prologue_words) {
          auto failure = make_kernel_failure(
              DiagnosticKind::KernelDescriptor,
              "kernel descriptor prologue changed after relocated text was emitted");
          if (fail_or_skip_kernel(scope, std::move(failure), transaction)) {
            skip_scope = true;
            break;
          }
          return leave_unchanged();
        }

        updated->target_entry_text_offset = layout.target_entry;
        updated->target_body_entry_text_offset = layout.target_body_entry;
        translation = std::move(*updated);
        recomputed_descriptor = true;
      }
      if (skip_scope)
        continue;

      if (!recomputed_descriptor) {
        auto failure = make_kernel_failure(DiagnosticKind::KernelDescriptor,
                                           "kernel descriptor translation could not be recomputed");
        if (fail_or_skip_kernel(scope, std::move(failure), transaction))
          continue;
        return leave_unchanged();
      }
    }

    flush_traces(pending_traces, target_delta);

    for (KdTranslation &translation : descriptor_translations) {
      if (!same_kernel_scope_variant(translation, *scope.translation))
        continue;
      translation.target_entry_text_offset = layout.target_entry;
      translation.target_body_entry_text_offset = layout.target_body_entry;
    }
  }

  // A canonical copy is reached through a pointer, so any kernel in the object is a possible
  // caller, but only the hosting scope's descriptor was grown to cover it. One kernel scope means
  // the host is the only caller and the grown descriptor already covers it.
  if (canonical_host_outgrew_its_descriptor && scopes.size() > 1) {
    append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                 "a body reached through a code address needs resources beyond its hosting "
                 "kernel's descriptor, which would under-provision every other kernel that can "
                 "reach it");
    return leave_unchanged();
  }

  if (!resolve_pending_code_relocations(pending_code_relocations, text_relocations,
                                        canonical_placement, canonical_placement_variant_conflict,
                                        code_relocations, result.diagnostics))
    return leave_unchanged();

  if (continue_after_failure && has_error_diagnostic(result.diagnostics))
    return leave_unchanged();

  // Phase 6 commits the completed translation plan without mixing ELF mutation
  // and sidecar metadata construction into the per-kernel lowering transaction.
  // A stored function pointer must name the canonical copy of its target. Offsets whose clones
  // disagreed on kernel-scope variant are withheld, which leaves the addend path fail-closed for
  // them exactly as before.
  std::unordered_map<uint64_t, uint64_t> canonical_code_pointer_placement;
  for (const auto &[source_offset, target_offset] : canonical_placement) {
    if (!canonical_placement_variant_conflict.contains(source_offset))
      canonical_code_pointer_placement.emplace(source_offset, target_offset);
  }

  auto materialized = materialize_translated_code_object(
      std::move(patcher), std::move(translated_text), text.size(), text_relocations,
      data_relocations, code_relocations, descriptor_translations, host_arch_, target_mach_,
      relied_on_relocated_by_construction, canonical_code_pointer_placement, result.diagnostics);
  if (!materialized)
    return leave_unchanged();
  result.elf_bytes = std::move(*materialized);
  return result;
}

bool BinaryTranslator::handle_encoding(const Instruction &inst, uint64_t offset,
                                       std::vector<uint8_t> &text, uint16_t dst_opcode,
                                       std::span<const uint8_t> orig_text,
                                       bool collect_target_words, bool &copied_original,
                                       bool &changed, std::vector<uint32_t> &target_words) {
  const uint32_t *raw = inst.raw_encoding();
  assert(raw && "handle_encoding called without raw encoding");
  copied_original = false;
  changed = false;
  if (collect_target_words)
    target_words.clear();

  if (!encoding_translate_) {
    copied_original = true;
    const size_t word_count = inst.size() / sizeof(uint32_t);
    if (collect_target_words)
      target_words.assign(raw, raw + word_count);
    append_words(text, std::span<const uint32_t>(raw, word_count));
    return true;
  }

  const uint32_t w0 = raw[0];
  const uint32_t w1 = inst.size() > 4 ? raw[1] : 0;
  const uint32_t w2 = inst.size() > 8 ? raw[2] : 0;

  auto tr = encoding_translate_(inst.encoding_id(), w0, w1, w2, dst_opcode);

  if (tr.word_count == 0) {
    copied_original = true;
    const size_t word_count = inst.size() / sizeof(uint32_t);
    if (collect_target_words)
      target_words.assign(raw, raw + word_count);
    append_words(text, std::span<const uint32_t>(raw, word_count));
    return true;
  }

  // Append trailing literal constant when the source instruction is larger
  // than the translated encoding. This handles single-word formats (SOP1,
  // SOP2, VOP1, VOP2, etc.) with a 32-bit literal appended when a source
  // operand is 0xFF. The encoding translator returns the format's native
  // word count; the literal is always one extra word beyond that.
  // Guard: only append if the gap is exactly one word (the literal). Larger
  // gaps would indicate a format mismatch, not a trailing literal.
  const uint32_t translated_bytes = tr.word_count * 4u;
  const uint32_t orig_bytes = inst.size();
  if (orig_bytes - translated_bytes == 4 && tr.word_count < 3) {
    uint32_t lit_word;
    std::memcpy(&lit_word, orig_text.data() + offset + translated_bytes, 4);
    tr.words[tr.word_count++] = lit_word;
  }

  append_words(text, std::span<const uint32_t>(tr.words, tr.word_count));
  if (collect_target_words) {
    target_words.assign(tr.words, tr.words + tr.word_count);
    changed = words_changed(raw_words_for_inst(inst), target_words);
  }
  return true;
}

} // namespace rocjitsu
