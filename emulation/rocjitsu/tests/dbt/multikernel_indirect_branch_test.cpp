// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file multikernel_indirect_branch_test.cpp
/// @brief DBT coverage for a real HIP code object with shared swappc targets.
///
/// @details "Real" describes the container and the surroundings, not every
/// instruction. The object is a genuine seven-kernel amdclang++ build. The
/// `s_swappc_b64` shared-helper calls and the helper's own `s_setpc_b64` return
/// are compiler-emitted; the three `s_setpc_b64` static-skip islands and the
/// three `s_call_b64` sites are planted by inline asm. An "island" here is one
/// such planted sequence; see `RJ_STATIC_SETPC_ISLAND` and
/// `RJ_STATIC_SCALL_ISLAND` in tests/kernels/multikernel_indirect_branch_common.h.
/// A setpc island builds its own target with `s_getpc_b64` plus a literal addend
/// and then jumps past a run of padding, which is the shape DBT's static
/// recovery is written to see through.
///
/// The fixture is compiled at build time, so its layout tracks whatever
/// toolchain built it; nothing here may pin an observed offset or an observed
/// instruction count.

#ifndef HAS_DEVICE_KERNELS
#error "multikernel_indirect_branch_test.cpp requires HAS_DEVICE_KERNELS"
#endif

#include "../test_paths.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/dbt/binary_translator.h"
#include "rocjitsu/code/dbt/kernel_descriptor_translator.h"
#include "rocjitsu/code/dbt/translation_diagnostic.h"
#include "rocjitsu/code/executable.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

using rocjitsu::test::kernel_path;

constexpr const char *kKernelA = "multikernel_indirect_branch_a";
constexpr const char *kKernelB = "multikernel_indirect_branch_b";
constexpr const char *kKernelC = "multikernel_indirect_branch_c";
constexpr const char *kKernelD = "multikernel_indirect_branch_d";
constexpr const char *kKernelE = "multikernel_indirect_branch_e";
constexpr const char *kKernelSetpc = "multikernel_indirect_branch_setpc";
constexpr const char *kKernelScall = "multikernel_indirect_branch_scall";

/// @brief Kernels that call the shared helper, per multikernel_indirect_branch.hip.
constexpr const char *kHelperCallers[] = {kKernelA, kKernelB, kKernelC, kKernelD, kKernelE};
constexpr const char *kAllKernels[] = {kKernelA, kKernelB,     kKernelC,    kKernelD,
                                       kKernelE, kKernelSetpc, kKernelScall};

/// @brief First literal of the shared helper body, from
/// `mixed = value ^ (salt * 0x45d9f3b)` in multikernel_indirect_branch_common.h.
///
/// @details A convenient marker we can use to count helper copies in a
/// translated image. DBT duplicates a shared callee into every kernel scope
/// that reaches it, and the helper has no symbol of its own after relocation,
/// so the literal is the available marker.
constexpr uint32_t kHelperMarkerWord = 0x045d9f3bu;

/// @brief Occurrences of a 32-bit value anywhere in `.text`.
///
/// @details A raw word scan, not a decode: it matches the value wherever it
/// lands, whether as an instruction literal or by coincidence. That is
/// sufficient for counting helper copies by their marker constant, and it is why
/// the untranslated baseline is pinned before the count is trusted.
size_t count_text_word(const rocjitsu::AmdGpuCodeObject &co, uint32_t word) {
  size_t count = 0;
  for (const auto *section : co.text_sections()) {
    const auto *words = reinterpret_cast<const uint32_t *>(section->data());
    const size_t word_count = section->size() / sizeof(uint32_t);
    count += static_cast<size_t>(std::count(words, words + word_count, word));
  }
  return count;
}

/// @brief Number of instructions in `.text` decoding to @p mnemonic.
///
/// @details Reports a failure rather than resynchronizing on an undecodable
/// word, matching the fail-closed policy of `BasicBlock::build`. Counts from
/// this sweep are compared against counts derived from the CFG, so the two must
/// agree on what an undecodable word means: guessing an instruction boundary
/// here would decode operand data as phantom instructions and silently skew one
/// side. The failure is non-fatal, so the returned count is partial and any
/// comparison against it is meaningless, including one that happens to pass,
/// such as a comparison against zero. The `ADD_FAILURE` is the real result.
size_t count_text_mnemonic(const rocjitsu::AmdGpuCodeObject &co, rj_code_arch_t arch,
                           std::string_view mnemonic) {
  auto decoder = rocjitsu::Decoder::create(arch);
  if (!decoder) {
    ADD_FAILURE() << "no decoder for arch " << static_cast<int>(arch);
    return 0;
  }

  size_t count = 0;
  for (const auto *section : co.text_sections()) {
    const auto *words = reinterpret_cast<const rj_code_binary_inst_t *>(section->data());
    const size_t word_count = section->size() / sizeof(rj_code_binary_inst_t);
    size_t word_offset = 0;
    while (word_offset < word_count) {
      const uint64_t byte_offset = word_offset * sizeof(rj_code_binary_inst_t);
      std::unique_ptr<rocjitsu::Instruction> inst(
          decoder->decode(words + word_offset, byte_offset));
      if (!inst) {
        ADD_FAILURE() << "undecodable word at .text offset " << byte_offset;
        return count;
      }
      if (inst->mnemonic() == mnemonic)
        ++count;
      // A zero-width decode would spin forever rather than fail.
      const size_t step = static_cast<size_t>(inst->size()) / sizeof(rj_code_binary_inst_t);
      if (step == 0) {
        ADD_FAILURE() << "zero-width decode at .text offset " << byte_offset;
        return count;
      }
      word_offset += step;
    }
  }
  return count;
}

/// @brief Messages of the error-severity diagnostics only.
///
/// @details `ok()` is false when *some* diagnostic is an error, but the first
/// entry may be an unrelated warning.
std::string error_diagnostics(const std::vector<rocjitsu::TranslationDiagnostic> &diagnostics) {
  std::string joined;
  for (const auto &diagnostic : diagnostics) {
    if (diagnostic.severity != rocjitsu::DiagnosticSeverity::Error)
      continue;
    if (!joined.empty())
      joined += "; ";
    joined += diagnostic.message;
  }
  return joined;
}

const rocjitsu::BasicBlock *
block_starting_at(const std::vector<std::unique_ptr<rocjitsu::BasicBlock>> &blocks,
                  uint64_t offset) {
  auto it = std::ranges::find_if(
      blocks, [offset](const auto &block) { return block->start_offset() == offset; });
  return it == blocks.end() ? nullptr : it->get();
}

bool has_successor_start(const rocjitsu::BasicBlock &block, uint64_t offset) {
  return std::ranges::any_of(block.successors(), [offset](const rocjitsu::BasicBlock *succ) {
    return succ->start_offset() == offset;
  });
}

/// @brief Blocks reachable from @p entry without crossing into another kernel.
///
/// @details Traversal stops at another kernel's hardware entry, matching the
/// scope rule in docs/dbt-design.md. Those stops are counted into
/// @p pruned_cross_kernel_edges rather than silently dropped: this fixture has
/// no kernel that legitimately transfers to another kernel's entry, so a nonzero
/// count is itself a containment failure that the reachable set cannot show.
std::vector<const rocjitsu::BasicBlock *>
reachable_kernel_blocks(const std::vector<std::unique_ptr<rocjitsu::BasicBlock>> &blocks,
                        const rocjitsu::BasicBlock &entry,
                        const std::unordered_set<uint64_t> &kernel_entries,
                        size_t *pruned_cross_kernel_edges = nullptr) {
  std::unordered_set<const rocjitsu::BasicBlock *> reachable;
  std::vector<const rocjitsu::BasicBlock *> stack{&entry};

  const auto crosses_into_other_kernel = [&](const rocjitsu::BasicBlock *target) {
    if (target->start_offset() == entry.start_offset() ||
        !kernel_entries.contains(target->start_offset()))
      return false;
    if (pruned_cross_kernel_edges != nullptr)
      ++*pruned_cross_kernel_edges;
    return true;
  };

  while (!stack.empty()) {
    const rocjitsu::BasicBlock *block = stack.back();
    stack.pop_back();
    if (block == nullptr || !reachable.insert(block).second)
      continue;

    for (const rocjitsu::BasicBlock *succ : block->successors()) {
      if (succ == nullptr || crosses_into_other_kernel(succ))
        continue;
      stack.push_back(succ);
    }
    for (const rocjitsu::BasicBlock::CallEdge &call : block->call_edges()) {
      const rocjitsu::BasicBlock *callee = call.callee;
      if (callee == nullptr || crosses_into_other_kernel(callee))
        continue;
      stack.push_back(callee);
    }
  }

  std::vector<const rocjitsu::BasicBlock *> ordered;
  ordered.reserve(reachable.size());
  for (const auto &block : blocks) {
    if (block && reachable.contains(block.get()))
      ordered.push_back(block.get());
  }
  return ordered;
}

/// @brief The parsed descriptor for kernel @p name, or nullptr if absent.
///
/// @details Matches on the name the parser recorded. Matching on a descriptor
/// offset instead would compare two different coordinate spaces:
/// `kernel_descriptor_offset` reports a symbol's vaddr, while
/// `descriptor_file_offset` is a file offset.
const rocjitsu::KdTranslation *
kernel_translation_by_name(std::span<const rocjitsu::KdTranslation> kernels, const char *name) {
  auto it =
      std::ranges::find_if(kernels, [name](const auto &info) { return info.kernel_name == name; });
  return it == kernels.end() ? nullptr : &*it;
}

/// @brief A fixture object loaded, parsed, and turned into a CFG.
struct LoadedObject {
  std::unique_ptr<rocjitsu::Executable> exec;
  const rocjitsu::AmdGpuCodeObject *co = nullptr;
  std::vector<rocjitsu::KdTranslation> kernels;
  std::unordered_set<uint64_t> entries;
  std::unique_ptr<rocjitsu::Decoder> decoder;
  std::vector<std::unique_ptr<rocjitsu::BasicBlock>> blocks;
};

LoadedObject load_fixture(const char *name) {
  LoadedObject loaded;
  loaded.exec = std::make_unique<rocjitsu::Executable>(kernel_path(name));
  if (!loaded.exec->is_valid() || loaded.exec->num_code_objects(ROCJITSU_CODE_TARGET_GFX950) == 0)
    return loaded;

  loaded.co = loaded.exec->code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  if (loaded.co == nullptr || loaded.co->text_sections().empty())
    return loaded;

  const auto *text = loaded.co->text_sections()[0];
  rocjitsu::KernelDescriptorTranslator parser(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  loaded.kernels = parser.translate_image(
      {reinterpret_cast<const uint8_t *>(loaded.co->image_data()), loaded.co->image_size()},
      text->sectionOffset(), text->size(), rocjitsu::KernelDescriptorTranslationOptions{});

  std::vector<uint64_t> entry_leaders;
  entry_leaders.reserve(loaded.kernels.size());
  for (const auto &kernel : loaded.kernels) {
    entry_leaders.push_back(kernel.entry_text_offset);
    loaded.entries.insert(kernel.entry_text_offset);
  }

  loaded.decoder = rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  if (!loaded.decoder)
    return loaded;

  loaded.blocks = rocjitsu::BasicBlock::build(*loaded.co, *loaded.decoder, ROCJITSU_CODE_ARCH_CDNA4,
                                              entry_leaders);
  return loaded;
}

/// @brief One kernel's scope, in both absolute and layout-independent form.
///
/// @details `blocks` and `external_blocks` are absolute `.text` offsets and are
/// therefore only comparable within one object. `own_offsets` are relative to
/// the kernel's entry, so they survive being compiled into a different object.
/// Blocks outside the kernel's own extent (the shared callee) are deliberately
/// kept out of `own_offsets`: their distance from the entry depends on how many
/// kernels happen to precede them, which changes between the full fixture and a
/// split one.
struct KernelScope {
  std::set<uint64_t> blocks;
  std::set<uint64_t> external_blocks;
  std::vector<int64_t> own_offsets;
};

KernelScope kernel_scope(const LoadedObject &loaded, const char *name,
                         size_t *pruned_cross_kernel_edges = nullptr) {
  KernelScope scope;
  const auto *kernel = kernel_translation_by_name(loaded.kernels, name);
  if (kernel == nullptr)
    return scope;
  const auto *entry = block_starting_at(loaded.blocks, kernel->entry_text_offset);
  if (entry == nullptr)
    return scope;

  const uint64_t own_begin = kernel->entry_text_offset;
  uint64_t own_end = std::numeric_limits<uint64_t>::max();
  for (uint64_t other_entry : loaded.entries) {
    if (other_entry > own_begin && other_entry < own_end)
      own_end = other_entry;
  }

  for (const auto *block :
       reachable_kernel_blocks(loaded.blocks, *entry, loaded.entries, pruned_cross_kernel_edges)) {
    const uint64_t start = block->start_offset();
    scope.blocks.insert(start);
    if (start >= own_begin && start < own_end)
      scope.own_offsets.push_back(static_cast<int64_t>(start) - static_cast<int64_t>(own_begin));
    else
      scope.external_blocks.insert(start);
  }
  std::ranges::sort(scope.own_offsets);
  return scope;
}

/// @brief Printable shape of a scope, comparable across separately built objects.
std::string scope_shape(const KernelScope &scope) {
  std::string shape;
  for (int64_t offset : scope.own_offsets)
    shape += std::to_string(offset) + " ";
  shape += "| blocks outside the kernel: " + std::to_string(scope.external_blocks.size());
  return shape;
}

std::set<uint64_t> intersect(const std::set<uint64_t> &lhs, const std::set<uint64_t> &rhs) {
  std::set<uint64_t> common;
  std::ranges::set_intersection(lhs, rhs, std::inserter(common, common.end()));
  return common;
}

/// @brief CFG shape required of a recovered `s_setpc_b64` block.
///
/// @details A setpc is an indirect *branch*, so its recovered target becomes an
/// ordinary successor. Contrast expect_recovered_target_call_edges below.
void expect_recovered_target_successors(
    const std::vector<std::unique_ptr<rocjitsu::BasicBlock>> &blocks,
    const rocjitsu::BasicBlock &block) {
  for (const auto &fixup : block.static_indirect_call_fixups()) {
    SCOPED_TRACE("recovered setpc target at .text offset " +
                 std::to_string(fixup.source_target_offset));
    EXPECT_FALSE(fixup.source_is_call) << "a setpc consumer is a branch, not a call";
    EXPECT_FALSE(fixup.source_incomplete)
        << "an incomplete fact must not be lowered to a direct transfer window";
    if (block_starting_at(blocks, fixup.source_target_offset) == nullptr) {
      ADD_FAILURE() << "no block begins at the recovered target";
      continue;
    }
    EXPECT_TRUE(has_successor_start(block, fixup.source_target_offset));
  }
}

/// @brief CFG shape required of a recovered `s_swappc_b64` block.
///
/// @details A swappc is an indirect *call*, so its recovered target becomes a
/// call edge carrying both the callee and this call's own continuation, rather
/// than an ordinary successor. Contrast expect_recovered_target_successors above.
void expect_recovered_target_call_edges(
    const std::vector<std::unique_ptr<rocjitsu::BasicBlock>> &blocks,
    const rocjitsu::BasicBlock &block) {
  EXPECT_EQ(block.call_edges().size(), 1u) << "single-target call site should have one call edge";
  for (const auto &fixup : block.static_indirect_call_fixups()) {
    SCOPED_TRACE("recovered swappc target at .text offset " +
                 std::to_string(fixup.source_target_offset));
    EXPECT_TRUE(fixup.source_is_call);
    EXPECT_FALSE(fixup.source_incomplete)
        << "an incomplete fact must not be lowered to a direct transfer window";
    if (block_starting_at(blocks, fixup.source_target_offset) == nullptr) {
      ADD_FAILURE() << "no block begins at the recovered target";
      continue;
    }
    EXPECT_TRUE(std::ranges::any_of(block.call_edges(), [&](const auto &edge) {
      return edge.kind == rocjitsu::BasicBlock::CallEdgeKind::IndirectSwapPc &&
             edge.callee != nullptr && edge.callee->start_offset() == fixup.source_target_offset &&
             edge.continuation != nullptr &&
             edge.continuation->start_offset() == block.end_offset();
    }));
  }
}

} // namespace

/// @brief Goal: the fixture properties whose source of truth is the .hip.
///
/// @details Only properties with a source of truth in
/// tests/kernels/multikernel_indirect_branch.hip are pinned here: the kernels it
/// declares, the three `s_call_b64` islands it plants, which the compiler emits
/// verbatim, and its single definition of the shared helper — the baseline that
/// makes a helper-copy count meaningful after translation.
/// Deliberately absent is the `s_swappc_b64` count: the .hip has six
/// call sites into the helper, but reaching them by an indirect swappc rather
/// than a direct call is the compiler's choice, so no count of them is an
/// invariant. That the helper is genuinely shared is asserted in the CFG test
/// instead, as a property rather than a tally.
///
/// Not everything the other tests rely on can be pinned here. The three
/// `RJ_STATIC_SETPC_ISLAND` sites are only distinguishable from the helper's own
/// return once the CFG exists, so their count is asserted in the CFG test
/// instead; a codegen change that lost them shows up there rather than here.
TEST(BinaryTranslatorE2E, MultiKernelIndirectBranchFixtureHasExpectedShape) {
  const auto loaded = load_fixture("multikernel_indirect_branch");
  ASSERT_NE(loaded.co, nullptr) << "Failed to load multikernel_indirect_branch.o";

  ASSERT_EQ(loaded.kernels.size(), std::size(kAllKernels))
      << "fixture should carry seven real kernel descriptors in one code object";
  for (const char *name : kAllKernels)
    EXPECT_NE(kernel_translation_by_name(loaded.kernels, name), nullptr) << name;

  // Three RJ_STATIC_SCALL_ISLAND sites, emitted verbatim from inline asm.
  EXPECT_EQ(count_text_mnemonic(*loaded.co, ROCJITSU_CODE_ARCH_CDNA4, "s_call_b64"), 3u);
  EXPECT_EQ(count_text_word(*loaded.co, kHelperMarkerWord), 1u)
      << "the untranslated object holds exactly one occurrence of the helper's mixing constant";
}

/// @brief Goal: a real seven-kernel object must translate CDNA4 -> CDNA3 into a
/// usable gfx942 object.
///
/// @details Translation is not refused; all seven kernels survive the rewrite as
/// real kernels rather than skipped stubs; the shared helper is duplicated once
/// per calling kernel; recovered indirect transfers are lowered to direct ones,
/// conserving the call population; and the emitted ELF really targets gfx942.
/// `ASSERT_TRUE(result.ok())` is not merely a smoke check: DBT fails closed on
/// an unrecovered indirect transfer reachable in a relocated kernel, so a
/// successful translation is evidence recovery resolved these sites. The CFG
/// test asserts that directly. The one exemption is a validated call return: a
/// `s_setpc_b64` reached from a call edge whose callee and continuation are both
/// in this kernel scope is deliberately left indirect (see
/// `scoped_call_return_offsets` in binary_translator.cpp), which is how the
/// helper's own return survives untranslated and is duplicated below.
///
/// Semantic equivalence of the translated code is out of scope. Nothing here
/// executes; that belongs to the HIP and corpus test tiers.
TEST(BinaryTranslatorE2E, TranslatesRealMultiKernelSharedSwappcCodeObject) {
  const auto loaded = load_fixture("multikernel_indirect_branch");
  ASSERT_NE(loaded.co, nullptr) << "Failed to load multikernel_indirect_branch.o";
  const auto &co = *loaded.co;

  const size_t source_swappc = count_text_mnemonic(co, ROCJITSU_CODE_ARCH_CDNA4, "s_swappc_b64");
  const size_t source_call = count_text_mnemonic(co, ROCJITSU_CODE_ARCH_CDNA4, "s_call_b64");

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(co);

  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << error_diagnostics(result.diagnostics);
  // A skipped kernel keeps its descriptor symbol, so the per-kernel checks below
  // cannot by themselves tell a translated kernel from an `s_endpgm` stub.
  ASSERT_TRUE(result.dispatchable()) << "translation must not silently stub out a kernel";
  EXPECT_EQ(result.host_arch, ROCJITSU_CODE_ARCH_CDNA3);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  for (const char *name : kAllKernels)
    EXPECT_NE(translated.kernel_descriptor_offset(name), 0u) << name;

  // DBT duplicates a shared callee into every kernel scope that reaches it
  // (docs/dbt-design.md, "Form Kernel Scopes"), so the five kernels calling the
  // helper must each get their own copy.
  const size_t helper_copies = count_text_word(translated, kHelperMarkerWord);
  EXPECT_EQ(helper_copies, std::size(kHelperCallers));

  EXPECT_EQ(count_text_mnemonic(translated, ROCJITSU_CODE_ARCH_CDNA3, "s_swappc_b64"), 0u)
      << "in-range recovered swappc targets should patch to direct s_call_b64 windows";
  // Recovered `s_setpc_b64` islands lower to direct branches exactly as recovered
  // swappc lower to direct calls, so every setpc left in the output is one
  // duplicated copy of the helper's ordinary return.
  EXPECT_EQ(count_text_mnemonic(translated, ROCJITSU_CODE_ARCH_CDNA3, "s_setpc_b64"),
            helper_copies);
  // Conservation, and the counterweight to the zero above: without it, deleting
  // the calls outright would satisfy `s_swappc_b64 == 0`. Every source direct
  // call survives and every lowered swappc becomes one more, with nothing else in
  // the output introducing an `s_call_b64`.
  EXPECT_EQ(count_text_mnemonic(translated, ROCJITSU_CODE_ARCH_CDNA3, "s_call_b64"),
            source_call + source_swappc);

  ASSERT_GE(result.elf_bytes.size(), sizeof(rocjitsu::Elf64_Ehdr));
  const auto *ehdr = reinterpret_cast<const rocjitsu::Elf64_Ehdr *>(result.elf_bytes.data());
  EXPECT_EQ(ehdr->e_flags & rocjitsu::EF_AMDGPU_MACH, rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX942);
}

/// @brief Goal: `BasicBlock::build` must give each scalar transfer form the
/// correct kind of CFG edge, on real compiler output.
///
/// @details docs/dbt-design.md models calls apart from ordinary control flow so
/// that a callee's return binds to its own continuation rather than to every
/// possible one:
///   - recovered `s_swappc_b64`: a call edge carrying callee and continuation;
///     the callee is not an ordinary successor, and the return edge is kept;
///   - recovered `s_setpc_b64`: an ordinary successor, and no fallthrough;
///   - direct `s_call_b64` whose target does not return: an ordinary successor
///     at the PC-relative target, and no call edge. A *returning* direct call
///     instead produces a `CallEdgeKind::DirectCall`; this fixture's three island
///     targets all jump to the join, so that form is not exercised here.
///     tests/analysis/liveness_test.cpp covers it synthetically.
///
/// Alongside those per-site shapes it pins two things about recovery itself:
/// that it reached every indirect transfer the compiler emitted, and that all
/// shared-helper calls converge on one callee.
///
/// One limit worth stating: the recovered target offsets are taken from the
/// fixups and checked against the CFG built from those same fixups, so a
/// recovery bug that resolved every call to the same wrong address would satisfy
/// everything here. Target correctness is established by execution in the corpus
/// tier, not statically in this file.
///
/// What this fixture adds over the synthetic CfgAnalysis suite in
/// tests/analysis/liveness_test.cpp is that the wiring holds on real toolchain
/// output, where five kernels share one helper body and a recovered island sits
/// inside compiler-generated divergent control flow.
TEST(BinaryTranslatorE2E, BuildsCfgForRealMultiKernelIndirectBranches) {
  const auto loaded = load_fixture("multikernel_indirect_branch");
  ASSERT_NE(loaded.co, nullptr) << "Failed to load multikernel_indirect_branch.o";
  ASSERT_FALSE(loaded.blocks.empty());
  const auto &blocks = loaded.blocks;

  size_t recovered_swappc_blocks = 0;
  size_t recovered_setpc_blocks = 0;
  size_t direct_scall_blocks = 0;
  std::set<uint64_t> unrecovered_setpc_blocks;
  std::set<uint64_t> swappc_target_offsets;

  for (const auto &block : blocks) {
    const auto *term = block->terminator();
    if (term == nullptr)
      continue;

    SCOPED_TRACE("block at .text offset " + std::to_string(block->start_offset()));
    const std::string_view mnemonic = term->mnemonic();
    if (mnemonic == "s_swappc_b64" && !block->static_indirect_call_fixups().empty()) {
      ++recovered_swappc_blocks;
      expect_recovered_target_call_edges(blocks, *block);
      for (const auto &fixup : block->static_indirect_call_fixups()) {
        EXPECT_FALSE(has_successor_start(*block, fixup.source_target_offset))
            << "swappc callees must be call edges, not ordinary CFG successors";
        swappc_target_offsets.insert(fixup.source_target_offset);
      }
      EXPECT_TRUE(has_successor_start(*block, block->end_offset()))
          << "indirect calls must retain the return/fallthrough edge";
    } else if (mnemonic == "s_setpc_b64") {
      if (block->static_indirect_call_fixups().empty()) {
        unrecovered_setpc_blocks.insert(block->start_offset());
      } else {
        ++recovered_setpc_blocks;
        expect_recovered_target_successors(blocks, *block);
        EXPECT_FALSE(has_successor_start(*block, block->end_offset()))
            << "indirect branches must not keep a fallthrough edge";
      }
    } else if (mnemonic == "s_call_b64") {
      ++direct_scall_blocks;
      // RJ_STATIC_SCALL_ISLAND places the call target immediately after the
      // island's dead `s_branch 2f`, so the destination is always four bytes
      // past the end of the call block. Checking against that rather than
      // recomputing `end_offset() + delta` keeps the oracle independent of the
      // arithmetic in BasicBlock::build.
      const auto branch_delta = term->branch_offset_bytes();
      ASSERT_TRUE(branch_delta.has_value());
      EXPECT_EQ(*branch_delta, int64_t{4});
      EXPECT_TRUE(has_successor_start(*block, block->end_offset() + 4));
      EXPECT_TRUE(block->call_edges().empty()) << "fixture s_call sites are tail transfers";
      EXPECT_FALSE(has_successor_start(*block, block->end_offset()))
          << "tail transfers must drop their unreachable fallthrough edge";
    }
  }

  // The fixture exists to put several kernel entries in front of one reachable
  // helper body (tests/kernels/multikernel_indirect_branch_common.h). Every
  // recovered swappc must therefore resolve to the same callee.
  ASSERT_FALSE(swappc_target_offsets.empty()) << "no recovered swappc call sites were found";
  EXPECT_EQ(swappc_target_offsets.size(), 1u) << "all shared-helper calls must reach one callee";

  // Recovery must reach every indirect call the compiler emitted, not just the
  // count the current codegen happens to produce.
  EXPECT_EQ(recovered_swappc_blocks,
            count_text_mnemonic(*loaded.co, ROCJITSU_CODE_ARCH_CDNA4, "s_swappc_b64"));
  // Three RJ_STATIC_SETPC_ISLAND sites are recoverable.
  EXPECT_EQ(recovered_setpc_blocks, 3u);
  // The helper's return is the one indirect transfer recovery must leave alone:
  // where it goes depends on the caller, so there is no single static answer. It
  // terminates the helper body, which is the block every shared-helper call
  // reaches, and asserting that identity is stronger than counting the leftover.
  ASSERT_EQ(unrecovered_setpc_blocks.size(), 1u);
  EXPECT_EQ(*unrecovered_setpc_blocks.begin(), *swappc_target_offsets.begin())
      << "the unrecovered setpc should terminate the block the swappc calls reach";
  // Three RJ_STATIC_SCALL_ISLAND sites, each a block terminator.
  EXPECT_EQ(direct_scall_blocks, 3u);
}

/// @brief Goal: each kernel's CFG must contain exactly its own blocks, plus the
/// shared helper it calls.
///
/// @details A kernel scope is the set of blocks reachable from one descriptor
/// entry (docs/dbt-design.md, "Form Kernel Scopes"). Recovered indirect targets
/// are the hazard: a wrong target can pull a neighboring kernel's body into this
/// scope.
///
/// The properties asserted are relations between scopes rather than block
/// counts, so a benign codegen change does not move them: the two kernels that
/// call no helper share nothing with anyone, and the five that do share exactly
/// the helper and nothing else.
///
/// It also asserts the scope walk never had to stop at another kernel's entry.
/// The walk suppresses exactly the edge a leak of that shape would take, so
/// without counting the suppressions the reachable sets could not reveal it.
TEST(BinaryTranslatorE2E, MultiKernelIndirectBranchKernelScopesShareOnlyTheHelper) {
  const auto loaded = load_fixture("multikernel_indirect_branch");
  ASSERT_NE(loaded.co, nullptr) << "Failed to load multikernel_indirect_branch.o";
  ASSERT_FALSE(loaded.blocks.empty());

  size_t pruned_cross_kernel_edges = 0;
  std::unordered_map<std::string, KernelScope> scopes;
  for (const char *name : kAllKernels) {
    auto scope = kernel_scope(loaded, name, &pruned_cross_kernel_edges);
    ASSERT_FALSE(scope.blocks.empty()) << name;
    scopes.emplace(name, std::move(scope));
  }

  EXPECT_EQ(pruned_cross_kernel_edges, 0u)
      << "a CFG edge reached another kernel's entry and was pruned by the scope walk";

  // The setpc and scall kernels call nothing, so they share no block with anyone.
  for (const char *isolated : {kKernelSetpc, kKernelScall}) {
    for (const char *other : kAllKernels) {
      if (std::string_view(other) == isolated)
        continue;
      EXPECT_TRUE(intersect(scopes.at(isolated).blocks, scopes.at(other).blocks).empty())
          << isolated << " overlaps " << other;
    }
  }

  // The five helper callers overlap in exactly the helper body, identically for
  // every pair. Anchoring `shared` to kernel A's blocks-outside-itself is what
  // makes this say "the callee"; pairwise equality alone would also hold if all
  // five wrongly absorbed one common block of somebody's kernel body.
  const auto shared = intersect(scopes.at(kKernelA).blocks, scopes.at(kKernelB).blocks);
  EXPECT_FALSE(shared.empty()) << "helper callers should share the helper body";
  EXPECT_EQ(shared, scopes.at(kKernelA).external_blocks)
      << "what the callers share should be exactly the callee, not part of a kernel body";
  for (const char *lhs : kHelperCallers) {
    for (const char *rhs : kHelperCallers) {
      if (std::string_view(lhs) == rhs)
        continue;
      EXPECT_EQ(intersect(scopes.at(lhs).blocks, scopes.at(rhs).blocks), shared)
          << lhs << " vs " << rhs;
    }
  }
}

/// @brief Goal: control for the kernel-scope containment above.
///
/// @details The same kernel bodies compiled into two smaller objects have fewer
/// neighbors to leak into. With cross-kernel overlap and entry-edge leaks now
/// asserted directly, the case this control uniquely covers is a kernel
/// absorbing inter-kernel padding that only exists in the full seven-kernel
/// object: that inflates the full scope while leaving every scope disjoint.
///
/// The split objects are first checked to hold exactly their own kernels;
/// without that a mis-generated part would make this compare the full object
/// against a near-copy of itself and pass.
///
/// Each scope is compared as the entry-relative offsets of the kernel's own
/// blocks, plus a count of the blocks outside it. Absolute offsets could not be
/// used (the parts are compiled independently and share no addresses with the
/// full object), and terminator mnemonics alone are too coarse: kernels b, c, d
/// and e produce identical mnemonic multisets, so a scope swapped between them
/// would go unnoticed. The shared callee is counted rather than placed, because
/// its distance from a kernel entry depends on how many kernels precede it.
TEST(BinaryTranslatorE2E, SplitFixturesMatchFullFixtureKernelScopes) {
  constexpr const char *kPart0Kernels[] = {kKernelA, kKernelB, kKernelC};
  constexpr const char *kPart1Kernels[] = {kKernelD, kKernelE, kKernelSetpc, kKernelScall};

  const auto full = load_fixture("multikernel_indirect_branch");
  ASSERT_NE(full.co, nullptr);
  ASSERT_EQ(full.kernels.size(), std::size(kAllKernels));

  const auto part0 = load_fixture("multikernel_indirect_branch_part0");
  ASSERT_NE(part0.co, nullptr);
  ASSERT_EQ(part0.kernels.size(), std::size(kPart0Kernels));
  for (const char *name : kPart0Kernels)
    ASSERT_NE(kernel_translation_by_name(part0.kernels, name), nullptr) << name;

  const auto part1 = load_fixture("multikernel_indirect_branch_part1");
  ASSERT_NE(part1.co, nullptr);
  ASSERT_EQ(part1.kernels.size(), std::size(kPart1Kernels));
  for (const char *name : kPart1Kernels)
    ASSERT_NE(kernel_translation_by_name(part1.kernels, name), nullptr) << name;

  for (const char *name : kPart0Kernels) {
    const auto split_scope = kernel_scope(part0, name);
    ASSERT_FALSE(split_scope.blocks.empty()) << name;
    EXPECT_EQ(scope_shape(kernel_scope(full, name)), scope_shape(split_scope)) << name;
  }
  for (const char *name : kPart1Kernels) {
    const auto split_scope = kernel_scope(part1, name);
    ASSERT_FALSE(split_scope.blocks.empty()) << name;
    EXPECT_EQ(scope_shape(kernel_scope(full, name)), scope_shape(split_scope)) << name;
  }
}
