// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file multikernel_indirect_branch_test.cpp
/// @brief DBT coverage for a real HIP code object with shared swappc targets.

#ifndef HAS_DEVICE_KERNELS
#error "multikernel_indirect_branch_test.cpp requires HAS_DEVICE_KERNELS"
#endif

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/dbt/binary_translator.h"
#include "rocjitsu/code/dbt/kernel_descriptor_translator.h"
#include "rocjitsu/code/executable.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

std::string kernel_path(const char *name) { return std::string(KERNEL_DIR) + "/" + name + ".o"; }

size_t count_text_mnemonic(const rocjitsu::AmdGpuCodeObject &co, rj_code_arch_t arch,
                           std::string_view mnemonic) {
  auto decoder = rocjitsu::Decoder::create(arch);
  if (!decoder)
    return 0;

  size_t count = 0;
  for (const auto *section : co.text_sections()) {
    const auto *words = reinterpret_cast<const rj_code_binary_inst_t *>(section->data());
    const size_t word_count = section->size() / sizeof(rj_code_binary_inst_t);
    size_t word_offset = 0;
    while (word_offset < word_count) {
      std::unique_ptr<rocjitsu::Instruction> inst(
          decoder->decode(words + word_offset, word_offset * sizeof(rj_code_binary_inst_t)));
      if (!inst) {
        ++word_offset;
        continue;
      }
      if (inst->mnemonic() == mnemonic)
        ++count;
      word_offset += static_cast<size_t>(inst->size()) / sizeof(rj_code_binary_inst_t);
    }
  }
  return count;
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

std::vector<const rocjitsu::BasicBlock *>
reachable_kernel_blocks(const std::vector<std::unique_ptr<rocjitsu::BasicBlock>> &blocks,
                        const rocjitsu::BasicBlock &entry,
                        const std::unordered_set<uint64_t> &kernel_entries) {
  std::unordered_set<const rocjitsu::BasicBlock *> reachable;
  std::vector<const rocjitsu::BasicBlock *> stack{&entry};

  while (!stack.empty()) {
    const rocjitsu::BasicBlock *block = stack.back();
    stack.pop_back();
    if (block == nullptr || !reachable.insert(block).second)
      continue;

    for (const rocjitsu::BasicBlock *succ : block->successors()) {
      if (succ == nullptr)
        continue;
      if (succ->start_offset() != entry.start_offset() &&
          kernel_entries.contains(succ->start_offset()))
        continue;
      stack.push_back(succ);
    }
    for (const rocjitsu::BasicBlock::CallEdge &call : block->call_edges()) {
      const rocjitsu::BasicBlock *callee = call.callee;
      if (callee == nullptr)
        continue;
      if (callee->start_offset() != entry.start_offset() &&
          kernel_entries.contains(callee->start_offset()))
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

const rocjitsu::KdTranslation *
kernel_translation_by_name(std::span<const rocjitsu::KdTranslation> kernels,
                           const rocjitsu::AmdGpuCodeObject &co, const char *name) {
  const uint64_t descriptor_offset = co.kernel_descriptor_offset(name);
  auto it = std::ranges::find_if(kernels, [descriptor_offset](const auto &info) {
    return info.descriptor_file_offset == descriptor_offset;
  });
  return it == kernels.end() ? nullptr : &*it;
}

void expect_recovered_target_successors(
    const std::vector<std::unique_ptr<rocjitsu::BasicBlock>> &blocks,
    const rocjitsu::BasicBlock &block) {
  for (const auto &fixup : block.static_indirect_call_fixups()) {
    EXPECT_EQ(fixup.source_call_offset, block.start_offset());
    ASSERT_NE(block_starting_at(blocks, fixup.source_target_offset), nullptr);
    EXPECT_TRUE(has_successor_start(block, fixup.source_target_offset));
  }
}

void expect_recovered_target_call_edges(
    const std::vector<std::unique_ptr<rocjitsu::BasicBlock>> &blocks,
    const rocjitsu::BasicBlock &block) {
  for (const auto &fixup : block.static_indirect_call_fixups()) {
    EXPECT_TRUE(fixup.source_is_call);
    EXPECT_EQ(fixup.source_call_offset, block.start_offset());
    ASSERT_NE(block_starting_at(blocks, fixup.source_target_offset), nullptr);
    EXPECT_TRUE(std::ranges::any_of(block.call_edges(), [&](const auto &edge) {
      return edge.callee != nullptr && edge.callee->start_offset() == fixup.source_target_offset &&
             edge.continuation != nullptr &&
             edge.continuation->start_offset() == block.end_offset();
    }));
  }
}

std::unordered_map<std::string, size_t>
cfg_block_counts_by_kernel(const rocjitsu::AmdGpuCodeObject &co,
                           std::span<const char *const> kernel_names) {
  std::unordered_map<std::string, size_t> counts;
  if (co.text_sections().empty())
    return counts;

  const auto *text = co.text_sections()[0];
  rocjitsu::KernelDescriptorTranslator source_parser(ROCJITSU_CODE_ARCH_CDNA4,
                                                     ROCJITSU_CODE_ARCH_CDNA3);
  const auto source_kernels = source_parser.translate_image(
      {reinterpret_cast<const uint8_t *>(co.image_data()), co.image_size()}, text->sectionOffset(),
      text->size(), rocjitsu::KernelDescriptorTranslationOptions{});

  std::vector<uint64_t> kernel_entry_leaders;
  kernel_entry_leaders.reserve(source_kernels.size());
  std::unordered_set<uint64_t> entries;
  for (const auto &kernel : source_kernels) {
    kernel_entry_leaders.push_back(kernel.entry_text_offset);
    entries.insert(kernel.entry_text_offset);
  }

  auto decoder = rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  if (!decoder)
    return counts;

  auto blocks =
      rocjitsu::BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4, kernel_entry_leaders);

  for (const char *name : kernel_names) {
    const auto *kernel = kernel_translation_by_name(source_kernels, co, name);
    if (kernel == nullptr)
      continue;
    const auto *entry = block_starting_at(blocks, kernel->entry_text_offset);
    if (entry == nullptr)
      continue;
    const auto reachable = reachable_kernel_blocks(blocks, *entry, entries);
    counts.emplace(name, reachable.size());
  }
  return counts;
}

} // namespace

TEST(BinaryTranslatorE2E, TranslatesRealMultiKernelSharedSwappcCodeObject) {
  rocjitsu::Executable exec(kernel_path("multikernel_indirect_branch"));
  ASSERT_TRUE(exec.is_valid()) << "Failed to load multikernel_indirect_branch.o";
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);

  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);
  ASSERT_FALSE(co->text_sections().empty());

  EXPECT_NE(co->kernel_descriptor_offset("multikernel_indirect_branch_a"), 0u);
  EXPECT_NE(co->kernel_descriptor_offset("multikernel_indirect_branch_b"), 0u);
  EXPECT_NE(co->kernel_descriptor_offset("multikernel_indirect_branch_c"), 0u);
  EXPECT_NE(co->kernel_descriptor_offset("multikernel_indirect_branch_d"), 0u);
  EXPECT_NE(co->kernel_descriptor_offset("multikernel_indirect_branch_e"), 0u);
  EXPECT_NE(co->kernel_descriptor_offset("multikernel_indirect_branch_setpc"), 0u);
  EXPECT_NE(co->kernel_descriptor_offset("multikernel_indirect_branch_scall"), 0u);

  const auto *text = co->text_sections()[0];
  rocjitsu::KernelDescriptorTranslator source_parser(ROCJITSU_CODE_ARCH_CDNA4,
                                                     ROCJITSU_CODE_ARCH_CDNA3);
  const auto source_kernels = source_parser.translate_image(
      {reinterpret_cast<const uint8_t *>(co->image_data()), co->image_size()},
      text->sectionOffset(), text->size(), rocjitsu::KernelDescriptorTranslationOptions{});
  ASSERT_EQ(source_kernels.size(), 7u)
      << "fixture should carry seven real kernel descriptors in one code object";

  // Kernels A-E all call the same noinline helper, and kernel A calls it twice.
  // Those shared-helper edges are compiler-emitted `s_swappc_b64` instructions.
  EXPECT_GE(count_text_mnemonic(*co, ROCJITSU_CODE_ARCH_CDNA4, "s_swappc_b64"), 6u);
  EXPECT_GE(count_text_mnemonic(*co, ROCJITSU_CODE_ARCH_CDNA4, "s_setpc_b64"), 3u);
  EXPECT_GE(count_text_mnemonic(*co, ROCJITSU_CODE_ARCH_CDNA4, "s_call_b64"), 3u);

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(*co);

  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  EXPECT_EQ(result.host_arch, ROCJITSU_CODE_ARCH_CDNA3);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  EXPECT_NE(translated.kernel_descriptor_offset("multikernel_indirect_branch_a"), 0u);
  EXPECT_NE(translated.kernel_descriptor_offset("multikernel_indirect_branch_b"), 0u);
  EXPECT_NE(translated.kernel_descriptor_offset("multikernel_indirect_branch_c"), 0u);
  EXPECT_NE(translated.kernel_descriptor_offset("multikernel_indirect_branch_d"), 0u);
  EXPECT_NE(translated.kernel_descriptor_offset("multikernel_indirect_branch_e"), 0u);
  EXPECT_NE(translated.kernel_descriptor_offset("multikernel_indirect_branch_setpc"), 0u);
  EXPECT_NE(translated.kernel_descriptor_offset("multikernel_indirect_branch_scall"), 0u);
  EXPECT_GE(count_text_mnemonic(translated, ROCJITSU_CODE_ARCH_CDNA3, "s_swappc_b64"), 6u);
  EXPECT_GE(count_text_mnemonic(translated, ROCJITSU_CODE_ARCH_CDNA3, "s_setpc_b64"), 3u);
  EXPECT_GE(count_text_mnemonic(translated, ROCJITSU_CODE_ARCH_CDNA3, "s_call_b64"), 3u);

  ASSERT_GE(result.elf_bytes.size(), sizeof(rocjitsu::Elf64_Ehdr));
  const auto *ehdr = reinterpret_cast<const rocjitsu::Elf64_Ehdr *>(result.elf_bytes.data());
  EXPECT_EQ(ehdr->e_flags & rocjitsu::EF_AMDGPU_MACH, rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX942);
}

TEST(BinaryTranslatorE2E, BuildsCfgForRealMultiKernelIndirectBranches) {
  rocjitsu::Executable exec(kernel_path("multikernel_indirect_branch"));
  ASSERT_TRUE(exec.is_valid()) << "Failed to load multikernel_indirect_branch.o";
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);

  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);
  ASSERT_FALSE(co->text_sections().empty());

  const auto *text = co->text_sections()[0];
  rocjitsu::KernelDescriptorTranslator source_parser(ROCJITSU_CODE_ARCH_CDNA4,
                                                     ROCJITSU_CODE_ARCH_CDNA3);
  const auto source_kernels = source_parser.translate_image(
      {reinterpret_cast<const uint8_t *>(co->image_data()), co->image_size()},
      text->sectionOffset(), text->size(), rocjitsu::KernelDescriptorTranslationOptions{});
  ASSERT_EQ(source_kernels.size(), 7u);

  std::vector<uint64_t> kernel_entries;
  kernel_entries.reserve(source_kernels.size());
  for (const auto &kernel : source_kernels)
    kernel_entries.push_back(kernel.entry_text_offset);

  auto decoder = rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  auto blocks =
      rocjitsu::BasicBlock::build(*co, *decoder, ROCJITSU_CODE_ARCH_CDNA4, kernel_entries);
  ASSERT_FALSE(blocks.empty());

  size_t recovered_swappc_blocks = 0;
  size_t recovered_setpc_blocks = 0;
  size_t direct_scall_blocks = 0;
  for (const auto &block : blocks) {
    const auto *term = block->terminator();
    if (term == nullptr)
      continue;

    const std::string_view mnemonic = term->mnemonic();
    if (mnemonic == "s_swappc_b64" && !block->static_indirect_call_fixups().empty()) {
      ++recovered_swappc_blocks;
      expect_recovered_target_call_edges(blocks, *block);
      for (const auto &fixup : block->static_indirect_call_fixups())
        EXPECT_FALSE(has_successor_start(*block, fixup.source_target_offset))
            << "swappc callees must be call edges, not ordinary CFG successors";
      EXPECT_TRUE(has_successor_start(*block, block->end_offset()))
          << "indirect calls must retain the return/fallthrough edge";
    } else if (mnemonic == "s_setpc_b64" && !block->static_indirect_call_fixups().empty()) {
      ++recovered_setpc_blocks;
      expect_recovered_target_successors(blocks, *block);
      EXPECT_FALSE(has_successor_start(*block, block->end_offset()))
          << "indirect branches must not keep a fallthrough edge";
    } else if (mnemonic == "s_call_b64") {
      ++direct_scall_blocks;
      const auto branch_delta = term->branch_offset_bytes();
      ASSERT_TRUE(branch_delta.has_value());
      const int64_t target =
          static_cast<int64_t>(block->end_offset()) + static_cast<int64_t>(*branch_delta);
      ASSERT_GE(target, 0);
      EXPECT_TRUE(has_successor_start(*block, static_cast<uint64_t>(target)));
      EXPECT_TRUE(has_successor_start(*block, block->end_offset()))
          << "direct calls must retain the return/fallthrough edge";
    }
  }

  EXPECT_GE(recovered_swappc_blocks, 6u);
  EXPECT_GE(recovered_setpc_blocks, 3u);
  EXPECT_GE(direct_scall_blocks, 3u);
}

TEST(BinaryTranslatorE2E, CountsRealMultiKernelIndirectBranchCfgBlocksPerKernel) {
  rocjitsu::Executable exec(kernel_path("multikernel_indirect_branch"));
  ASSERT_TRUE(exec.is_valid()) << "Failed to load multikernel_indirect_branch.o";
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);

  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);
  ASSERT_FALSE(co->text_sections().empty());

  const auto *text = co->text_sections()[0];
  rocjitsu::KernelDescriptorTranslator source_parser(ROCJITSU_CODE_ARCH_CDNA4,
                                                     ROCJITSU_CODE_ARCH_CDNA3);
  const auto source_kernels = source_parser.translate_image(
      {reinterpret_cast<const uint8_t *>(co->image_data()), co->image_size()},
      text->sectionOffset(), text->size(), rocjitsu::KernelDescriptorTranslationOptions{});
  ASSERT_EQ(source_kernels.size(), 7u);

  std::vector<uint64_t> kernel_entry_leaders;
  kernel_entry_leaders.reserve(source_kernels.size());
  std::unordered_set<uint64_t> entries;
  for (const auto &kernel : source_kernels) {
    kernel_entry_leaders.push_back(kernel.entry_text_offset);
    entries.insert(kernel.entry_text_offset);
  }

  auto decoder = rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  auto blocks =
      rocjitsu::BasicBlock::build(*co, *decoder, ROCJITSU_CODE_ARCH_CDNA4, kernel_entry_leaders);
  ASSERT_FALSE(blocks.empty());

  struct ExpectedKernelCfg {
    const char *name;
    size_t reachable_blocks;
  };

  const ExpectedKernelCfg expected[] = {
      // A reaches the shared helper block and has two shared-helper swappc
      // calls. Its kernel-local CFG is: range-check entry, in-range call setup,
      // first swappc, between-call value prep, second swappc, store/end,
      // out-of-range end, plus the shared helper return block.
      {"multikernel_indirect_branch_a", 8},
      // B-E each reach the same shared helper through one swappc. Each kernel
      // contributes range-check entry, in-range call setup, swappc, store/end,
      // and out-of-range end, plus the shared helper return block.
      {"multikernel_indirect_branch_b", 6},
      {"multikernel_indirect_branch_c", 6},
      {"multikernel_indirect_branch_d", 6},
      {"multikernel_indirect_branch_e", 6},
      // The setpc kernel has three independent static-skip islands. Each island
      // splits into a compare/branch block, a recovered s_setpc block, and the
      // recovered target block, with the surrounding range-check and final
      // store/end blocks making the exact reachable count below.
      {"multikernel_indirect_branch_setpc", 20},
      // The s_call kernel has three independent direct-call islands. Each call
      // block has both the direct call-target edge and the return/fallthrough
      // edge; the exact count catches accidental extra CFG edges between those
      // islands or into another kernel entry.
      {"multikernel_indirect_branch_scall", 23},
  };

  for (const auto &test_case : expected) {
    const auto *kernel = kernel_translation_by_name(source_kernels, *co, test_case.name);
    ASSERT_NE(kernel, nullptr) << test_case.name;
    const auto *entry = block_starting_at(blocks, kernel->entry_text_offset);
    ASSERT_NE(entry, nullptr) << test_case.name;
    const auto reachable = reachable_kernel_blocks(blocks, *entry, entries);
    EXPECT_EQ(reachable.size(), test_case.reachable_blocks) << test_case.name;
  }
}

TEST(BinaryTranslatorE2E, SplitFixturesMatchFullFixtureCfgBlockCounts) {
  constexpr const char *kPart0Kernels[] = {
      "multikernel_indirect_branch_a",
      "multikernel_indirect_branch_b",
      "multikernel_indirect_branch_c",
  };
  constexpr const char *kPart1Kernels[] = {
      "multikernel_indirect_branch_d",
      "multikernel_indirect_branch_e",
      "multikernel_indirect_branch_setpc",
      "multikernel_indirect_branch_scall",
  };
  constexpr const char *kAllKernels[] = {
      "multikernel_indirect_branch_a",     "multikernel_indirect_branch_b",
      "multikernel_indirect_branch_c",     "multikernel_indirect_branch_d",
      "multikernel_indirect_branch_e",     "multikernel_indirect_branch_setpc",
      "multikernel_indirect_branch_scall",
  };

  rocjitsu::Executable full_exec(kernel_path("multikernel_indirect_branch"));
  ASSERT_TRUE(full_exec.is_valid());
  ASSERT_GT(full_exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);
  const auto *full = full_exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(full, nullptr);
  const auto full_counts = cfg_block_counts_by_kernel(*full, kAllKernels);

  rocjitsu::Executable part0_exec(kernel_path("multikernel_indirect_branch_part0"));
  ASSERT_TRUE(part0_exec.is_valid());
  ASSERT_GT(part0_exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);
  const auto *part0 = part0_exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(part0, nullptr);
  const auto part0_counts = cfg_block_counts_by_kernel(*part0, kPart0Kernels);

  rocjitsu::Executable part1_exec(kernel_path("multikernel_indirect_branch_part1"));
  ASSERT_TRUE(part1_exec.is_valid());
  ASSERT_GT(part1_exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);
  const auto *part1 = part1_exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(part1, nullptr);
  const auto part1_counts = cfg_block_counts_by_kernel(*part1, kPart1Kernels);

  // Splitting the same kernel bodies into two smaller real code objects should
  // not change per-kernel CFG reachability. If the full fixture accidentally
  // gains an edge into a neighboring kernel body, that kernel's count diverges
  // from its count in the corresponding split fixture.
  for (const char *name : kPart0Kernels) {
    ASSERT_TRUE(full_counts.contains(name)) << name;
    ASSERT_TRUE(part0_counts.contains(name)) << name;
    EXPECT_EQ(full_counts.at(name), part0_counts.at(name)) << name;
  }
  for (const char *name : kPart1Kernels) {
    ASSERT_TRUE(full_counts.contains(name)) << name;
    ASSERT_TRUE(part1_counts.contains(name)) << name;
    EXPECT_EQ(full_counts.at(name), part1_counts.at(name)) << name;
  }
}
