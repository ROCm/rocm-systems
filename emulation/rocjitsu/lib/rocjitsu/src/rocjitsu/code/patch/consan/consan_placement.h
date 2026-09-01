// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_placement.h
/// @brief Shared placement, range reservation, and scratch-selection contracts.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace rocjitsu {

class AmdGpuCodeObject;
class BasicBlock;
class Instruction;
class LivenessAnalysis;

struct ByteRange {
  uint64_t begin = 0;
  uint64_t end = 0;
};

struct InPlaceNopSite {
  uint64_t text_offset = 0;
  uint64_t file_offset = 0;
};

struct InPlaceInstructionSite {
  uint64_t text_offset = 0;
  uint64_t file_offset = 0;
  uint32_t size = 0;
};

struct BarrierSite {
  uint64_t text_offset = 0;
  uint64_t file_offset = 0;
  uint32_t size = 0;
  std::string range_name;
  std::string mnemonic;
};

struct LocalNopCave {
  uint64_t text_offset = 0;
  uint64_t file_offset = 0;
  uint32_t word_count = 0;
  uint64_t owner_entry_text_offset = 0;
};

struct KernelMaxRegisterRefs {
  std::optional<uint16_t> sgpr;
  std::optional<uint16_t> vgpr;
};

[[nodiscard]] bool ranges_overlap(ByteRange lhs, ByteRange rhs);
[[nodiscard]] bool overlaps_preapplied_reserved_range(const ProgramInventory &inventory,
                                                      ByteRange range);
[[nodiscard]] bool overlaps_reserved_range(std::span<const ByteRange> ranges, ByteRange range);
[[nodiscard]] std::optional<std::vector<ByteRange>>
reserved_ranges_for_existing_patches(const AmdGpuCodeObject &code_object,
                                     const ConSanTransformArtifacts &result);

[[nodiscard]] std::vector<LocalNopCave>
find_uncovered_nop_caves(const AmdGpuCodeObject &code_object, const ProgramInventory &inventory,
                         rj_code_arch_t arch);
[[nodiscard]] std::optional<InPlaceNopSite>
find_existing_nop_site(const AmdGpuCodeObject &code_object, rj_code_arch_t arch);
[[nodiscard]] std::optional<InPlaceInstructionSite>
find_preferred_in_place_instruction_site(const AmdGpuCodeObject &code_object, rj_code_arch_t arch);
[[nodiscard]] bool rewrite_word_in_place(const AmdGpuCodeObject &code_object, uint64_t file_offset,
                                         uint32_t replacement, ConSanTransformArtifacts &result);
[[nodiscard]] std::optional<uint64_t>
find_first_relocatable_anchor(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                              std::string *error_out);
[[nodiscard]] bool is_relocatable_consan_barrier_destination(const Instruction &instruction,
                                                             uint64_t offset,
                                                             std::span<const uint8_t> text,
                                                             rj_code_arch_t arch,
                                                             std::string *error_out);

[[nodiscard]] std::optional<uint16_t> lds_dword_count(const ConSanAccessInventorySite &access);
[[nodiscard]] bool vgpr_ranges_overlap(uint16_t lhs_base, uint16_t lhs_count, uint16_t rhs_base,
                                       uint16_t rhs_count);
[[nodiscard]] bool lds_scratch_tuple_base_is_valid(const ConSanAccessInventorySite &access,
                                                   uint16_t candidate);
[[nodiscard]] bool lds_load_clobbers_address(const ConSanAccessInventorySite &access);
[[nodiscard]] std::optional<uint16_t>
required_descriptor_vgpr_allocation_for_scratch(const ConSanAccessInventorySite &access,
                                                uint16_t scratch_vgpr, uint16_t required_vgprs);
[[nodiscard]] std::optional<uint16_t>
choose_scratch_vgpr(const ConSanAccessInventorySite &access, const ConSanOptions &options,
                    const Instruction *instruction, const LivenessAnalysis *liveness,
                    std::optional<uint16_t> min_auto_scratch_vgpr,
                    std::optional<uint16_t> max_auto_scratch_vgpr, uint16_t required_vgprs);
[[nodiscard]] std::optional<uint16_t>
choose_spill_scratch_vgpr(const ConSanAccessInventorySite &access, uint16_t allocation_count,
                          uint16_t required_vgprs);

[[nodiscard]] std::vector<BasicBlock *>
block_ptrs_for(const std::vector<std::unique_ptr<BasicBlock>> &blocks);
[[nodiscard]] const Instruction *
find_instruction_at_text_offset(std::span<BasicBlock *const> blocks, uint64_t text_offset);
[[nodiscard]] bool text_offset_is_inside_s_clause(std::span<BasicBlock *const> blocks,
                                                  uint64_t text_offset);
[[nodiscard]] KernelMaxRegisterRefs
max_register_refs_in_kernel(const ConSanKernelInfo &kernel, std::span<BasicBlock *const> blocks);

[[nodiscard]] bool read_words_at(std::span<const uint8_t> bytes, uint64_t offset,
                                 std::span<uint32_t> words);
void append_consan_patch_words(std::vector<uint8_t> &bytes, std::span<const uint32_t> words);
[[nodiscard]] uint32_t count_nop_padding(std::span<const uint8_t> bytes, uint64_t offset,
                                         uint32_t max_word_count, rj_code_arch_t arch);
[[nodiscard]] bool has_only_rocclr_runtime_kernels(const ProgramInventory &program_inventory);

} // namespace rocjitsu
