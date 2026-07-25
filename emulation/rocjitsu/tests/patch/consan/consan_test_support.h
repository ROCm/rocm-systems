// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/patch/cdna3_instrumentation_builder.h"
#include "rocjitsu/code/patch/cdna4_instrumentation_builder.h"
#include "rocjitsu/code/patch/consan/consan_moi.h"
#include "rocjitsu/code/patch/consan/consan_resource.h"
#include "rocjitsu/code/patch/gfx1250_instrumentation_builder.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/code/patch/rdna4_instrumentation_builder.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_sampled_sync.h"
#include "util/bit.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

namespace rocjitsu {
namespace {

inline constexpr uint8_t kElfSymbolBindLocal = 0;
inline constexpr uint8_t kElfSymbolTypeNotype = 0;
inline constexpr uint16_t kRdna4VccLo = 106;
inline constexpr uint16_t kRdna4ExecLo = 126;
inline constexpr uint16_t kScalarOperandTtmpBase = 108;
inline constexpr uint16_t kTtmpRdna4GridYz = 7;
inline constexpr uint16_t kTtmpRdna4GridX = 9;
inline constexpr uint32_t kRdna4Wave64AllVgprsGranulated = 63;

[[nodiscard]] constexpr uint64_t direct_sampled_report_bytes(uint32_t slot_count) {
  return sizeof(ConSanMoiReportHeader) +
         static_cast<uint64_t>(slot_count) *
             (sizeof(uint64_t) + sizeof(ConSanMoiSampledCausalWindow) +
              sizeof(ConSanMoiSampledSyncMetadataPacked) +
              sizeof(ConSanMoiSampledPendingAcquireSlot));
}

using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
namespace kd = rocr::llvm::amdhsa;

testing::AssertionResult consan_patch_succeeded(const ConSanResult &result) {
  if (result.errors.empty())
    return testing::AssertionSuccess();
  return testing::AssertionFailure() << testing::PrintToString(result.errors);
}

template <size_t WordCount>
std::array<uint32_t, WordCount> patched_words_at_file_offset(const ConSanResult &result,
                                                             size_t file_offset) {
  std::array<uint32_t, WordCount> words{};
  constexpr size_t byte_count = WordCount * sizeof(uint32_t);
  if (file_offset > result.elf_bytes.size() || byte_count > result.elf_bytes.size() - file_offset) {
    ADD_FAILURE() << "patched word range exceeds the emitted ELF image";
    return words;
  }
  std::memcpy(words.data(), result.elf_bytes.data() + file_offset, byte_count);
  return words;
}

std::vector<uint32_t> patched_words_at_file_offset(const ConSanResult &result, size_t file_offset,
                                                   size_t byte_count) {
  if (byte_count % sizeof(uint32_t) != 0 || file_offset > result.elf_bytes.size() ||
      byte_count > result.elf_bytes.size() - file_offset) {
    ADD_FAILURE() << "patched word range is unaligned or exceeds the emitted ELF image";
    return {};
  }
  std::vector<uint32_t> words(byte_count / sizeof(uint32_t));
  std::memcpy(words.data(), result.elf_bytes.data() + file_offset, byte_count);
  return words;
}

std::vector<uint32_t> text_words_at_offset(const AmdGpuCodeObject &code_object, size_t text_offset,
                                           size_t byte_count) {
  if (byte_count % sizeof(uint32_t) != 0 || code_object.text_sections().size() != 1u) {
    ADD_FAILURE() << "patched text is missing, ambiguous, or unaligned";
    return {};
  }
  const Section *text = code_object.text_sections().front();
  if (text_offset > text->size() || byte_count > text->size() - text_offset) {
    ADD_FAILURE() << "patched word range exceeds the emitted text section";
    return {};
  }
  std::vector<uint32_t> words(byte_count / sizeof(uint32_t));
  std::memcpy(words.data(), text->data() + text_offset, byte_count);
  return words;
}

ConSanOptions moi_options(ConSanMoiEngine engine = ConSanMoiEngine::RecordReplay) {
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = engine;
  return options;
}

ConSanRegisterRequest vgpr_request(uint16_t count, uint16_t current_allocation_count,
                                   uint16_t max_referenced_count) {
  ConSanRegisterRequest request;
  request.reg_class = RegClass::VGPR;
  request.count = count;
  request.current_allocation_count = current_allocation_count;
  request.max_referenced_count = max_referenced_count;
  request.architecture_limit = 256;
  return request;
}

void expand_all_vgprs(RegisterSet &set) {
  set.expand({RegClass::VGPR, 0, 255});
  set.expand({RegClass::VGPR, 255, 1});
}

uint32_t add_elf_name(std::vector<uint8_t> &names, std::string_view name) {
  const uint32_t offset = static_cast<uint32_t>(names.size());
  names.insert(names.end(), name.begin(), name.end());
  names.push_back('\0');
  return offset;
}

uint64_t align_up(uint64_t value, uint64_t alignment) {
  const uint64_t remainder = value % alignment;
  return remainder == 0 ? value : value + alignment - remainder;
}

template <typename Mutator>
void mutate_first_kernel_descriptor(std::vector<uint8_t> &bytes, Mutator mutator) {
  AmdGpuCodeObject code_object(bytes.data(), bytes.size());
  ASSERT_TRUE(code_object.is_valid());
  ASSERT_EQ(code_object.kernels().size(), 1u);

  const uint64_t descriptor_file_offset = code_object.kernels().front().descriptor_file_offset;
  ASSERT_LE(descriptor_file_offset, bytes.size());
  ASSERT_LE(sizeof(KD), bytes.size() - descriptor_file_offset);

  KD descriptor{};
  std::memcpy(&descriptor, bytes.data() + descriptor_file_offset, sizeof(descriptor));
  mutator(descriptor);
  std::memcpy(bytes.data() + descriptor_file_offset, &descriptor, sizeof(descriptor));
}

template <typename Mutator>
void mutate_kernel_descriptor(std::vector<uint8_t> &bytes, std::string_view kernel_name,
                              Mutator mutator) {
  AmdGpuCodeObject code_object(bytes.data(), bytes.size());
  ASSERT_TRUE(code_object.is_valid());
  const auto kernel =
      std::ranges::find(code_object.kernels(), kernel_name, &AmdGpuKernelInfo::name);
  ASSERT_NE(kernel, code_object.kernels().end());
  const uint64_t descriptor_file_offset = kernel->descriptor_file_offset;
  ASSERT_LE(descriptor_file_offset, bytes.size());
  ASSERT_LE(sizeof(KD), bytes.size() - descriptor_file_offset);
  KD descriptor{};
  std::memcpy(&descriptor, bytes.data() + descriptor_file_offset, sizeof(descriptor));
  mutator(descriptor);
  std::memcpy(bytes.data() + descriptor_file_offset, &descriptor, sizeof(descriptor));
}

template <typename Mutator> void mutate_elf_header(std::vector<uint8_t> &bytes, Mutator mutator) {
  ASSERT_GE(bytes.size(), sizeof(Elf64_Ehdr));
  Elf64_Ehdr header{};
  std::memcpy(&header, bytes.data(), sizeof(header));
  mutator(header);
  std::memcpy(bytes.data(), &header, sizeof(header));
}

template <typename Mutator>
void mutate_elf_section(std::vector<uint8_t> &bytes, uint16_t section_index, Mutator mutator) {
  ASSERT_GE(bytes.size(), sizeof(Elf64_Ehdr));
  Elf64_Ehdr header{};
  std::memcpy(&header, bytes.data(), sizeof(header));
  ASSERT_LT(section_index, header.e_shnum);
  const uint64_t offset = header.e_shoff + section_index * sizeof(Elf64_Shdr);
  ASSERT_LE(offset, bytes.size());
  ASSERT_LE(sizeof(Elf64_Shdr), bytes.size() - offset);
  Elf64_Shdr section{};
  std::memcpy(&section, bytes.data() + offset, sizeof(section));
  mutator(section);
  std::memcpy(bytes.data() + offset, &section, sizeof(section));
}

template <typename Mutator>
void mutate_elf_symbol(std::vector<uint8_t> &bytes, uint32_t symbol_index, Mutator mutator) {
  ASSERT_GE(bytes.size(), sizeof(Elf64_Ehdr));
  Elf64_Ehdr header{};
  std::memcpy(&header, bytes.data(), sizeof(header));
  ASSERT_GT(header.e_shnum, 3u);
  Elf64_Shdr symtab{};
  const uint64_t symtab_header_offset = header.e_shoff + 3u * sizeof(Elf64_Shdr);
  ASSERT_LE(symtab_header_offset, bytes.size());
  ASSERT_LE(sizeof(Elf64_Shdr), bytes.size() - symtab_header_offset);
  std::memcpy(&symtab, bytes.data() + symtab_header_offset, sizeof(symtab));
  const uint64_t offset = symtab.sh_offset + symbol_index * sizeof(Elf64_Sym);
  ASSERT_LE(offset, bytes.size());
  ASSERT_LE(sizeof(Elf64_Sym), bytes.size() - offset);
  Elf64_Sym symbol{};
  std::memcpy(&symbol, bytes.data() + offset, sizeof(symbol));
  mutator(symbol);
  std::memcpy(bytes.data() + offset, &symbol, sizeof(symbol));
}

bool contains_subsequence(std::span<const uint32_t> haystack, std::span<const uint32_t> needle) {
  return std::search(haystack.begin(), haystack.end(), needle.begin(), needle.end()) !=
         haystack.end();
}

size_t count_subsequence(std::span<const uint32_t> haystack, std::span<const uint32_t> needle) {
  size_t count = 0;
  auto it = haystack.begin();
  while (it != haystack.end()) {
    it = std::search(it, haystack.end(), needle.begin(), needle.end());
    if (it == haystack.end())
      break;
    ++count;
    ++it;
  }
  return count;
}

std::vector<uint32_t> expected_vgpr_spill_words(uint16_t base, uint16_t count, bool restore,
                                                uint32_t slot_base = 0) {
  std::vector<uint32_t> words;
  if (!restore) {
    const auto wait = build_s_wait_loadcnt0(ROCJITSU_CODE_ARCH_RDNA4);
    if (!wait)
      return {};
    words.push_back(*wait);
  }
  for (uint16_t i = 0; i < count; ++i) {
    const auto instruction =
        restore
            ? build_address_free_scratch_load_b32(static_cast<uint16_t>(base + i),
                                                  slot_base + 4u * i, ROCJITSU_CODE_ARCH_RDNA4)
            : build_address_free_scratch_store_b32(static_cast<uint16_t>(base + i),
                                                   slot_base + 4u * i, ROCJITSU_CODE_ARCH_RDNA4);
    if (!instruction)
      return {};
    words.insert(words.end(), instruction->begin(), instruction->end());
  }
  const auto wait = restore ? build_s_wait_loadcnt0(ROCJITSU_CODE_ARCH_RDNA4)
                            : build_s_wait_storecnt0(ROCJITSU_CODE_ARCH_RDNA4);
  if (!wait)
    return {};
  words.push_back(*wait);
  return words;
}

std::vector<uint32_t> make_expected_vgpr_store_words(uint64_t address, uint16_t value_vgpr,
                                                     uint16_t scratch_vgpr) {
  std::vector<uint32_t> words;
  const auto mov_address_lo = build_v_mov_b32_e64_literal(
      scratch_vgpr, static_cast<uint32_t>(address), ROCJITSU_CODE_ARCH_RDNA4);
  const auto mov_address_hi =
      build_v_mov_b32_e64_literal(static_cast<uint16_t>(scratch_vgpr + 1u),
                                  static_cast<uint32_t>(address >> 32u), ROCJITSU_CODE_ARCH_RDNA4);
  const auto store =
      build_flat_store_b32_vaddr_vsrc(scratch_vgpr, value_vgpr, ROCJITSU_CODE_ARCH_RDNA4);
  if (!mov_address_lo || !mov_address_hi || !store)
    return words;
  words.insert(words.end(), mov_address_lo->begin(), mov_address_lo->end());
  words.insert(words.end(), mov_address_hi->begin(), mov_address_hi->end());
  words.insert(words.end(), store->begin(), store->end());
  return words;
}

std::vector<uint32_t> make_expected_literal_store_words(uint64_t address, uint32_t value,
                                                        uint16_t scratch_vgpr) {
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + 2u);
  std::vector<uint32_t> words;
  const auto mov_address_lo = build_v_mov_b32_e64_literal(
      scratch_vgpr, static_cast<uint32_t>(address), ROCJITSU_CODE_ARCH_RDNA4);
  const auto mov_address_hi =
      build_v_mov_b32_e64_literal(static_cast<uint16_t>(scratch_vgpr + 1u),
                                  static_cast<uint32_t>(address >> 32u), ROCJITSU_CODE_ARCH_RDNA4);
  const auto mov_value = build_v_mov_b32_e64_literal(value_vgpr, value, ROCJITSU_CODE_ARCH_RDNA4);
  const auto store =
      build_flat_store_b32_vaddr_vsrc(scratch_vgpr, value_vgpr, ROCJITSU_CODE_ARCH_RDNA4);
  if (!mov_address_lo || !mov_address_hi || !mov_value || !store)
    return words;
  words.insert(words.end(), mov_address_lo->begin(), mov_address_lo->end());
  words.insert(words.end(), mov_address_hi->begin(), mov_address_hi->end());
  words.insert(words.end(), mov_value->begin(), mov_value->end());
  words.insert(words.end(), store->begin(), store->end());
  return words;
}

std::vector<uint32_t> make_expected_offset_store_words(uint32_t byte_offset, uint16_t value_vgpr,
                                                       uint16_t address_vgpr) {
  const auto store = build_flat_store_b32_vaddr_vsrc(address_vgpr, value_vgpr,
                                                     ROCJITSU_CODE_ARCH_RDNA4, byte_offset);
  return store ? std::vector<uint32_t>(store->begin(), store->end()) : std::vector<uint32_t>{};
}

std::vector<uint32_t> make_expected_literal_offset_store_words(uint32_t byte_offset, uint32_t value,
                                                               uint16_t address_vgpr,
                                                               uint16_t value_vgpr) {
  const auto materialize = build_v_mov_b32_e64_literal(value_vgpr, value, ROCJITSU_CODE_ARCH_RDNA4);
  if (!materialize)
    return {};
  std::vector<uint32_t> words(materialize->begin(), materialize->end());
  const std::vector<uint32_t> store =
      make_expected_offset_store_words(byte_offset, value_vgpr, address_vgpr);
  words.insert(words.end(), store.begin(), store.end());
  return words;
}

std::vector<uint32_t> make_expected_offset_load_words(uint32_t byte_offset, uint16_t value_vgpr,
                                                      uint16_t address_vgpr) {
  const auto load = build_flat_load_b32_vaddr_vdst(address_vgpr, value_vgpr,
                                                   ROCJITSU_CODE_ARCH_RDNA4, byte_offset);
  if (!load)
    return {};
  std::vector<uint32_t> words(load->begin(), load->end());
  words.push_back(0xBFC00000u);
  return words;
}

std::vector<uint32_t> make_expected_scalar_store_words(uint64_t address, uint16_t scalar_src,
                                                       uint16_t scratch_vgpr,
                                                       bool shift_right_16 = false,
                                                       bool mask_low_16 = false) {
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + 2u);
  std::vector<uint32_t> words;
  words.push_back(build_v_mov_b32_e32(value_vgpr, scalar_src, ROCJITSU_CODE_ARCH_RDNA4));
  if (mask_low_16) {
    const auto shift_left = build_v_lshlrev_b32_e32(value_vgpr, scalar_positive_inline_u32(16),
                                                    value_vgpr, ROCJITSU_CODE_ARCH_RDNA4);
    const auto shift_right = build_v_lshrrev_b32_e32(value_vgpr, scalar_positive_inline_u32(16),
                                                     value_vgpr, ROCJITSU_CODE_ARCH_RDNA4);
    if (!shift_left || !shift_right)
      return words;
    words.push_back(*shift_left);
    words.push_back(*shift_right);
  }
  if (shift_right_16) {
    const auto shift = build_v_lshrrev_b32_e32(value_vgpr, scalar_positive_inline_u32(16),
                                               value_vgpr, ROCJITSU_CODE_ARCH_RDNA4);
    if (!shift)
      return words;
    words.push_back(*shift);
  }
  const auto mov_address_lo = build_v_mov_b32_e64_literal(
      scratch_vgpr, static_cast<uint32_t>(address), ROCJITSU_CODE_ARCH_RDNA4);
  const auto mov_address_hi =
      build_v_mov_b32_e64_literal(static_cast<uint16_t>(scratch_vgpr + 1u),
                                  static_cast<uint32_t>(address >> 32u), ROCJITSU_CODE_ARCH_RDNA4);
  const auto store =
      build_flat_store_b32_vaddr_vsrc(scratch_vgpr, value_vgpr, ROCJITSU_CODE_ARCH_RDNA4);
  if (!mov_address_lo || !mov_address_hi || !store)
    return words;
  words.insert(words.end(), mov_address_lo->begin(), mov_address_lo->end());
  words.insert(words.end(), mov_address_hi->begin(), mov_address_hi->end());
  words.insert(words.end(), store->begin(), store->end());
  return words;
}

std::vector<uint32_t> make_expected_scalar_offset_store_words(uint32_t byte_offset,
                                                              uint16_t scalar_src,
                                                              uint16_t address_vgpr,
                                                              bool shift_right_16 = false,
                                                              bool mask_low_16 = false) {
  const uint16_t value_vgpr = static_cast<uint16_t>(address_vgpr + 2u);
  std::vector<uint32_t> words;
  words.push_back(build_v_mov_b32_e32(value_vgpr, scalar_src, ROCJITSU_CODE_ARCH_RDNA4));
  if (mask_low_16) {
    const auto shift_left = build_v_lshlrev_b32_e32(value_vgpr, scalar_positive_inline_u32(16),
                                                    value_vgpr, ROCJITSU_CODE_ARCH_RDNA4);
    const auto shift_right = build_v_lshrrev_b32_e32(value_vgpr, scalar_positive_inline_u32(16),
                                                     value_vgpr, ROCJITSU_CODE_ARCH_RDNA4);
    if (!shift_left || !shift_right)
      return {};
    words.push_back(*shift_left);
    words.push_back(*shift_right);
  }
  if (shift_right_16) {
    const auto shift = build_v_lshrrev_b32_e32(value_vgpr, scalar_positive_inline_u32(16),
                                               value_vgpr, ROCJITSU_CODE_ARCH_RDNA4);
    if (!shift)
      return {};
    words.push_back(*shift);
  }
  const std::vector<uint32_t> store =
      make_expected_offset_store_words(byte_offset, value_vgpr, address_vgpr);
  words.insert(words.end(), store.begin(), store.end());
  return words;
}

constexpr uint16_t ttmp_scalar_operand(uint16_t ttmp) {
  return static_cast<uint16_t>(kScalarOperandTtmpBase + ttmp);
}

std::vector<uint8_t> make_rdna4_lds_code_object(
    std::span<const uint32_t> text_words, std::string_view kernel_name = "lds_probe",
    uint32_t vgpr_granulated = kRdna4Wave64AllVgprsGranulated, bool wave32 = false,
    bool uses_dynamic_stack = false, uint32_t workgroup_id_dimension_mask = 0,
    uint32_t group_segment_fixed_size = 0) {
  constexpr uint64_t text_offset = 0x100;
  constexpr uint64_t text_vaddr = 0x1100;
  constexpr uint64_t rodata_vaddr = 0x2100;
  constexpr uint64_t kernel_descriptor_size = 64;

  const uint64_t text_size = text_words.size() * sizeof(uint32_t);

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_elf_name(shstrtab, ".text");
  const uint32_t rodata_name = add_elf_name(shstrtab, ".rodata");
  const uint32_t symtab_name = add_elf_name(shstrtab, ".symtab");
  const uint32_t strtab_name = add_elf_name(shstrtab, ".strtab");
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");

  std::vector<uint8_t> strtab{'\0'};
  const uint32_t kernel_symbol_name = add_elf_name(strtab, kernel_name);
  const std::string kd_name = std::string(kernel_name) + ".kd";
  const uint32_t kd_symbol_name = add_elf_name(strtab, kd_name);
  const std::string dynamic_stack_name = std::string(kernel_name) + ".has_dyn_sized_stack";
  const uint32_t dynamic_stack_symbol_name = add_elf_name(strtab, dynamic_stack_name);

  const uint64_t rodata_offset = text_offset + text_size;
  const uint64_t strtab_offset = rodata_offset + kernel_descriptor_size;
  const uint64_t symtab_offset = align_up(strtab_offset + strtab.size(), 8);
  constexpr size_t sym_count = 4;
  const uint64_t shstrtab_offset = symtab_offset + sym_count * sizeof(Elf64_Sym);
  const uint64_t shoff = align_up(shstrtab_offset + shstrtab.size(), 8);
  constexpr uint16_t section_count = 6;

  std::vector<uint8_t> image(shoff + section_count * sizeof(Elf64_Shdr), 0);

  Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr.e_ident[EI_DATA] = 1;
  ehdr.e_ident[EI_VERSION] = 1;
  ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  ehdr.e_ident[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  ehdr.e_type = ET_DYN;
  ehdr.e_machine = EM_AMDGPU;
  ehdr.e_version = 1;
  ehdr.e_shoff = shoff;
  ehdr.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX1201;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = section_count;
  ehdr.e_shstrndx = 5;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  std::memcpy(image.data() + text_offset, text_words.data(), text_size);

  static_assert(sizeof(KD) == kernel_descriptor_size);
  KD kernel_descriptor{};
  kernel_descriptor.group_segment_fixed_size = group_segment_fixed_size;
  const int64_t entry_offset =
      static_cast<int64_t>(text_vaddr) - static_cast<int64_t>(rodata_vaddr);
  kernel_descriptor.kernel_code_entry_byte_offset = entry_offset;
  AMDHSA_BITS_SET(kernel_descriptor.compute_pgm_rsrc1,
                  kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, vgpr_granulated);
  const uint32_t wave32_value = wave32 ? 1u : 0u;
  AMDHSA_BITS_SET(kernel_descriptor.kernel_code_properties,
                  kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32, wave32_value);
  if (workgroup_id_dimension_mask & 1u) {
    AMDHSA_BITS_SET(kernel_descriptor.compute_pgm_rsrc2,
                    kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X, 1);
  }
  if (workgroup_id_dimension_mask & 2u) {
    AMDHSA_BITS_SET(kernel_descriptor.compute_pgm_rsrc2,
                    kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y, 1);
  }
  if (workgroup_id_dimension_mask & 4u) {
    AMDHSA_BITS_SET(kernel_descriptor.compute_pgm_rsrc2,
                    kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z, 1);
  }
  std::memcpy(image.data() + rodata_offset, &kernel_descriptor, sizeof(kernel_descriptor));
  std::memcpy(image.data() + strtab_offset, strtab.data(), strtab.size());
  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());

  std::array<Elf64_Sym, sym_count> symbols{};
  symbols[1].st_name = kernel_symbol_name;
  symbols[1].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeFunc);
  symbols[1].st_shndx = 1;
  symbols[1].st_value = text_vaddr;
  symbols[1].st_size = text_size;
  symbols[2].st_name = kd_symbol_name;
  symbols[2].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeObject);
  symbols[2].st_shndx = 2;
  symbols[2].st_value = rodata_vaddr;
  symbols[2].st_size = kernel_descriptor_size;
  symbols[3].st_name = dynamic_stack_symbol_name;
  symbols[3].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeNotype);
  symbols[3].st_shndx = SHN_ABS;
  symbols[3].st_value = uses_dynamic_stack ? 1u : 0u;
  std::memcpy(image.data() + symtab_offset, symbols.data(), symbols.size() * sizeof(Elf64_Sym));

  std::array<Elf64_Shdr, section_count> sections{};
  sections[1].sh_name = text_name;
  sections[1].sh_type = SHT_PROGBITS;
  sections[1].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  sections[1].sh_addr = text_vaddr;
  sections[1].sh_offset = text_offset;
  sections[1].sh_size = text_size;
  sections[1].sh_addralign = sizeof(uint32_t);
  sections[2].sh_name = rodata_name;
  sections[2].sh_type = SHT_PROGBITS;
  sections[2].sh_flags = SHF_ALLOC;
  sections[2].sh_addr = rodata_vaddr;
  sections[2].sh_offset = rodata_offset;
  sections[2].sh_size = kernel_descriptor_size;
  sections[2].sh_addralign = 64;
  sections[3].sh_name = symtab_name;
  sections[3].sh_type = SHT_SYMTAB;
  sections[3].sh_offset = symtab_offset;
  sections[3].sh_size = sym_count * sizeof(Elf64_Sym);
  sections[3].sh_link = 4;
  sections[3].sh_info = 1;
  sections[3].sh_addralign = 8;
  sections[3].sh_entsize = sizeof(Elf64_Sym);
  sections[4].sh_name = strtab_name;
  sections[4].sh_type = SHT_STRTAB;
  sections[4].sh_offset = strtab_offset;
  sections[4].sh_size = strtab.size();
  sections[4].sh_addralign = 1;
  sections[5].sh_name = shstrtab_name;
  sections[5].sh_type = SHT_STRTAB;
  sections[5].sh_offset = shstrtab_offset;
  sections[5].sh_size = shstrtab.size();
  sections[5].sh_addralign = 1;
  std::memcpy(image.data() + shoff, sections.data(), sections.size() * sizeof(Elf64_Shdr));

  return image;
}

void append_kernel_metadata_note(
    std::vector<uint8_t> &image, std::string_view kernel_name, bool uses_dynamic_stack,
    uint8_t sgpr_count, std::optional<uint8_t> private_segment_fixed_size = std::nullopt,
    std::optional<std::array<uint8_t, 3>> required_workgroup_size = std::nullopt,
    bool has_dynamic_lds = false, std::span<const std::string_view> additional_kernel_names = {}) {
  const auto append_string = [](std::vector<uint8_t> &bytes, std::string_view value) {
    ASSERT_LE(value.size(), 255u);
    if (value.size() <= 31u) {
      bytes.push_back(static_cast<uint8_t>(0xa0u | value.size()));
    } else {
      bytes.push_back(0xd9u);
      bytes.push_back(static_cast<uint8_t>(value.size()));
    }
    bytes.insert(bytes.end(), value.begin(), value.end());
  };

  std::vector<uint8_t> payload;
  payload.push_back(0x81u); // root map(1)
  append_string(payload, "amdhsa.kernels");
  ASSERT_LT(additional_kernel_names.size(), 15u);
  payload.push_back(static_cast<uint8_t>(0x91u + additional_kernel_names.size()));
  const auto append_kernel = [&](std::string_view name) {
    payload.push_back(static_cast<uint8_t>(0x80u + 3u + (private_segment_fixed_size ? 1u : 0u) +
                                           (required_workgroup_size ? 1u : 0u) +
                                           (has_dynamic_lds ? 1u : 0u)));
    append_string(payload, ".name");
    append_string(payload, name);
    append_string(payload, ".uses_dynamic_stack");
    payload.push_back(uses_dynamic_stack ? 0xc3u : 0xc2u);
    append_string(payload, ".sgpr_count");
    payload.push_back(sgpr_count);
    if (private_segment_fixed_size) {
      append_string(payload, ".private_segment_fixed_size");
      payload.push_back(*private_segment_fixed_size);
    }
    if (required_workgroup_size) {
      append_string(payload, ".reqd_workgroup_size");
      payload.push_back(0x93u); // array(3)
      payload.insert(payload.end(), required_workgroup_size->begin(),
                     required_workgroup_size->end());
    }
    if (has_dynamic_lds) {
      append_string(payload, ".args");
      payload.push_back(0x91u); // array(1)
      payload.push_back(0x81u); // map(1)
      append_string(payload, ".value_kind");
      append_string(payload, "hidden_dynamic_lds_size");
    }
  };
  append_kernel(kernel_name);
  for (std::string_view name : additional_kernel_names)
    append_kernel(name);

  Elf64_Ehdr header{};
  ASSERT_GE(image.size(), sizeof(header));
  std::memcpy(&header, image.data(), sizeof(header));
  const uint64_t phoff = align_up(image.size(), 8u);
  const uint64_t note_offset = phoff + sizeof(Elf64_Phdr);
  constexpr std::array<uint8_t, 8> note_name = {'A', 'M', 'D', 'G', 'P', 'U', 0, 0};
  const uint64_t padded_payload_size = align_up(payload.size(), 4u);
  image.resize(note_offset + sizeof(Elf64_Nhdr) + note_name.size() + padded_payload_size, 0u);

  header.e_phoff = phoff;
  header.e_phentsize = sizeof(Elf64_Phdr);
  header.e_phnum = 1u;
  std::memcpy(image.data(), &header, sizeof(header));

  Elf64_Phdr program_header{};
  program_header.p_type = PT_NOTE;
  program_header.p_offset = note_offset;
  program_header.p_filesz = image.size() - note_offset;
  program_header.p_memsz = program_header.p_filesz;
  program_header.p_align = 4u;
  std::memcpy(image.data() + phoff, &program_header, sizeof(program_header));

  Elf64_Nhdr note{};
  note.n_namesz = 7u;
  note.n_descsz = static_cast<uint32_t>(payload.size());
  note.n_type = NT_AMDGPU_METADATA;
  std::memcpy(image.data() + note_offset, &note, sizeof(note));
  const uint64_t name_offset = note_offset + sizeof(note);
  std::memcpy(image.data() + name_offset, note_name.data(), note_name.size());
  std::memcpy(image.data() + name_offset + note_name.size(), payload.data(), payload.size());
}

std::vector<uint8_t> first_note_segment_bytes(std::span<const uint8_t> image) {
  if (image.size() < sizeof(Elf64_Ehdr))
    return {};
  Elf64_Ehdr header{};
  std::memcpy(&header, image.data(), sizeof(header));
  if (header.e_phentsize != sizeof(Elf64_Phdr) || header.e_phoff > image.size() ||
      static_cast<uint64_t>(header.e_phnum) * sizeof(Elf64_Phdr) > image.size() - header.e_phoff) {
    return {};
  }
  for (uint16_t index = 0; index < header.e_phnum; ++index) {
    Elf64_Phdr program_header{};
    std::memcpy(&program_header,
                image.data() + header.e_phoff + static_cast<uint64_t>(index) * sizeof(Elf64_Phdr),
                sizeof(program_header));
    if (program_header.p_type != PT_NOTE || program_header.p_offset > image.size() ||
        program_header.p_filesz > image.size() - program_header.p_offset) {
      continue;
    }
    return {image.begin() + static_cast<size_t>(program_header.p_offset),
            image.begin() + static_cast<size_t>(program_header.p_offset + program_header.p_filesz)};
  }
  return {};
}

std::vector<uint8_t> make_gfx1250_code_object(
    std::span<const uint32_t> text_words, std::string_view kernel_name = "barrier_lifecycle",
    uint32_t vgpr_granulated = kRdna4Wave64AllVgprsGranulated, bool wave32 = true) {
  std::vector<uint8_t> image =
      make_rdna4_lds_code_object(text_words, kernel_name, vgpr_granulated, wave32);
  mutate_elf_header(image,
                    [](Elf64_Ehdr &header) { header.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX1250; });
  return image;
}

std::vector<uint8_t> make_cdna4_lds_code_object(std::span<const uint32_t> text_words,
                                                std::string_view kernel_name = "lds_probe",
                                                uint32_t vgpr_granulated = 0u,
                                                bool uses_dynamic_stack = false,
                                                uint32_t workgroup_id_dimension_mask = 0u,
                                                uint32_t group_segment_fixed_size = 0u) {
  std::vector<uint8_t> image = make_rdna4_lds_code_object(
      text_words, kernel_name, vgpr_granulated, /*wave32=*/false, uses_dynamic_stack,
      workgroup_id_dimension_mask, group_segment_fixed_size);
  mutate_elf_header(image,
                    [](Elf64_Ehdr &header) { header.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX950; });
  return image;
}

std::vector<uint8_t> make_cdna3_lds_code_object(std::span<const uint32_t> text_words,
                                                std::string_view kernel_name = "lds_probe",
                                                uint32_t vgpr_granulated = 0u,
                                                bool uses_dynamic_stack = false,
                                                uint32_t workgroup_id_dimension_mask = 0u,
                                                uint32_t group_segment_fixed_size = 0u) {
  std::vector<uint8_t> image = make_rdna4_lds_code_object(
      text_words, kernel_name, vgpr_granulated, /*wave32=*/false, uses_dynamic_stack,
      workgroup_id_dimension_mask, group_segment_fixed_size);
  mutate_elf_header(image,
                    [](Elf64_Ehdr &header) { header.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX942; });
  return image;
}

std::vector<uint8_t> make_rdna4_code_object_with_local_function(
    std::span<const uint32_t> kernel_words, std::span<const uint32_t> function_words,
    std::span<const uint32_t> tail_words = {},
    uint32_t vgpr_granulated = kRdna4Wave64AllVgprsGranulated, bool function_is_kernel = false,
    bool wave32 = false) {
  constexpr uint64_t text_offset = 0x100;
  constexpr uint64_t text_vaddr = 0x1100;
  constexpr uint64_t rodata_vaddr = 0x2100;
  constexpr uint64_t kernel_descriptor_size = 64;

  const uint64_t kernel_size = kernel_words.size() * sizeof(uint32_t);
  const uint64_t function_size = function_words.size() * sizeof(uint32_t);
  const uint64_t tail_size = tail_words.size() * sizeof(uint32_t);
  const uint64_t text_size = kernel_size + function_size + tail_size;

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_elf_name(shstrtab, ".text");
  const uint32_t rodata_name = add_elf_name(shstrtab, ".rodata");
  const uint32_t symtab_name = add_elf_name(shstrtab, ".symtab");
  const uint32_t strtab_name = add_elf_name(shstrtab, ".strtab");
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");

  std::vector<uint8_t> strtab{'\0'};
  const uint32_t kernel_symbol_name = add_elf_name(strtab, "lds_probe");
  const uint32_t function_symbol_name = add_elf_name(strtab, "lds_helper");
  const uint32_t kd_symbol_name = add_elf_name(strtab, "lds_probe.kd");
  const uint32_t function_kd_symbol_name = add_elf_name(strtab, "lds_helper.kd");

  const uint64_t rodata_offset = text_offset + text_size;
  const uint64_t descriptor_count = function_is_kernel ? 2u : 1u;
  const uint64_t strtab_offset = rodata_offset + descriptor_count * kernel_descriptor_size;
  const uint64_t symtab_offset = align_up(strtab_offset + strtab.size(), 8);
  const size_t sym_count = function_is_kernel ? 5u : 4u;
  const uint64_t shstrtab_offset = symtab_offset + sym_count * sizeof(Elf64_Sym);
  const uint64_t shoff = align_up(shstrtab_offset + shstrtab.size(), 8);
  constexpr uint16_t section_count = 6;

  std::vector<uint8_t> image(shoff + section_count * sizeof(Elf64_Shdr), 0);

  Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr.e_ident[EI_DATA] = 1;
  ehdr.e_ident[EI_VERSION] = 1;
  ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  ehdr.e_ident[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  ehdr.e_type = ET_DYN;
  ehdr.e_machine = EM_AMDGPU;
  ehdr.e_version = 1;
  ehdr.e_shoff = shoff;
  ehdr.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX1201;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = section_count;
  ehdr.e_shstrndx = 5;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  std::memcpy(image.data() + text_offset, kernel_words.data(), kernel_size);
  std::memcpy(image.data() + text_offset + kernel_size, function_words.data(), function_size);
  if (tail_size != 0)
    std::memcpy(image.data() + text_offset + kernel_size + function_size, tail_words.data(),
                tail_size);

  static_assert(sizeof(KD) == kernel_descriptor_size);
  KD kernel_descriptor{};
  const int64_t entry_offset =
      static_cast<int64_t>(text_vaddr) - static_cast<int64_t>(rodata_vaddr);
  kernel_descriptor.kernel_code_entry_byte_offset = entry_offset;
  AMDHSA_BITS_SET(kernel_descriptor.compute_pgm_rsrc1,
                  kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, vgpr_granulated);
  AMDHSA_BITS_SET(kernel_descriptor.kernel_code_properties,
                  kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32, wave32 ? 1u : 0u);
  std::memcpy(image.data() + rodata_offset, &kernel_descriptor, sizeof(kernel_descriptor));
  if (function_is_kernel) {
    KD function_descriptor = kernel_descriptor;
    function_descriptor.kernel_code_entry_byte_offset =
        static_cast<int64_t>(text_vaddr + kernel_size) -
        static_cast<int64_t>(rodata_vaddr + kernel_descriptor_size);
    std::memcpy(image.data() + rodata_offset + kernel_descriptor_size, &function_descriptor,
                sizeof(function_descriptor));
  }
  std::memcpy(image.data() + strtab_offset, strtab.data(), strtab.size());
  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());

  std::vector<Elf64_Sym> symbols(sym_count);
  symbols[1].st_name = kernel_symbol_name;
  symbols[1].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeFunc);
  symbols[1].st_shndx = 1;
  symbols[1].st_value = text_vaddr;
  symbols[1].st_size = kernel_size;
  symbols[2].st_name = function_symbol_name;
  symbols[2].st_info = elf_symbol_info(
      function_is_kernel ? kElfSymbolBindGlobal : kElfSymbolBindLocal, kElfSymbolTypeFunc);
  symbols[2].st_shndx = 1;
  symbols[2].st_value = text_vaddr + kernel_size;
  symbols[2].st_size = function_size;
  symbols[3].st_name = kd_symbol_name;
  symbols[3].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeObject);
  symbols[3].st_shndx = 2;
  symbols[3].st_value = rodata_vaddr;
  symbols[3].st_size = kernel_descriptor_size;
  if (function_is_kernel) {
    symbols[4].st_name = function_kd_symbol_name;
    symbols[4].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeObject);
    symbols[4].st_shndx = 2;
    symbols[4].st_value = rodata_vaddr + kernel_descriptor_size;
    symbols[4].st_size = kernel_descriptor_size;
  }
  std::memcpy(image.data() + symtab_offset, symbols.data(), symbols.size() * sizeof(Elf64_Sym));

  std::array<Elf64_Shdr, section_count> sections{};
  sections[1].sh_name = text_name;
  sections[1].sh_type = SHT_PROGBITS;
  sections[1].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  sections[1].sh_addr = text_vaddr;
  sections[1].sh_offset = text_offset;
  sections[1].sh_size = text_size;
  sections[1].sh_addralign = sizeof(uint32_t);
  sections[2].sh_name = rodata_name;
  sections[2].sh_type = SHT_PROGBITS;
  sections[2].sh_flags = SHF_ALLOC;
  sections[2].sh_addr = rodata_vaddr;
  sections[2].sh_offset = rodata_offset;
  sections[2].sh_size = descriptor_count * kernel_descriptor_size;
  sections[2].sh_addralign = 64;
  sections[3].sh_name = symtab_name;
  sections[3].sh_type = SHT_SYMTAB;
  sections[3].sh_offset = symtab_offset;
  sections[3].sh_size = sym_count * sizeof(Elf64_Sym);
  sections[3].sh_link = 4;
  sections[3].sh_info = function_is_kernel ? 1 : 2;
  sections[3].sh_addralign = 8;
  sections[3].sh_entsize = sizeof(Elf64_Sym);
  sections[4].sh_name = strtab_name;
  sections[4].sh_type = SHT_STRTAB;
  sections[4].sh_offset = strtab_offset;
  sections[4].sh_size = strtab.size();
  sections[4].sh_addralign = 1;
  sections[5].sh_name = shstrtab_name;
  sections[5].sh_type = SHT_STRTAB;
  sections[5].sh_offset = shstrtab_offset;
  sections[5].sh_size = shstrtab.size();
  sections[5].sh_addralign = 1;
  std::memcpy(image.data() + shoff, sections.data(), sections.size() * sizeof(Elf64_Shdr));

  return image;
}

std::vector<uint8_t> make_gfx1250_code_object_with_local_function(
    std::span<const uint32_t> kernel_words, std::span<const uint32_t> function_words,
    std::span<const uint32_t> tail_words = {},
    uint32_t vgpr_granulated = kRdna4Wave64AllVgprsGranulated, bool function_is_kernel = false) {
  std::vector<uint8_t> image = make_rdna4_code_object_with_local_function(
      kernel_words, function_words, tail_words, vgpr_granulated, function_is_kernel,
      /*wave32=*/true);
  mutate_elf_header(image,
                    [](Elf64_Ehdr &header) { header.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX1250; });
  return image;
}

std::vector<uint8_t> make_cdna4_code_object_with_local_function(
    std::span<const uint32_t> kernel_words, std::span<const uint32_t> function_words,
    std::span<const uint32_t> tail_words = {}, uint32_t vgpr_granulated = 0u,
    bool function_is_kernel = false) {
  std::vector<uint8_t> image = make_rdna4_code_object_with_local_function(
      kernel_words, function_words, tail_words, vgpr_granulated, function_is_kernel,
      /*wave32=*/false);
  mutate_elf_header(image,
                    [](Elf64_Ehdr &header) { header.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX950; });
  return image;
}

std::vector<uint8_t>
make_cdna4_disconnected_scalar_pressure_code_object(uint16_t live_sgpr_count = 96u) {
  EXPECT_TRUE(live_sgpr_count == 96u || live_sgpr_count == 98u);
  const auto guest = build_cdna4_ds_store_b32(
      /*vaddr=*/0, /*vdata=*/1, /*byte_offset=*/0, ROCJITSU_CODE_ARCH_CDNA4);
  EXPECT_TRUE(guest);
  if (!guest)
    return {};

  std::vector<uint32_t> low_pressure(320u, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  std::copy(guest->begin(), guest->end(), low_pressure.begin());
  low_pressure.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);

  std::vector<uint32_t> high_pressure;
  high_pressure.insert(high_pressure.end(), guest->begin(), guest->end());
  // Keep the ordinary scalar prefix live across the access. With 96 live
  // registers the remaining pair can carry dispatch identity but not the
  // transient EXEC window; with 98, even dispatch identity must exclude this
  // disconnected owner.
  for (uint16_t sgpr = 0u; sgpr < live_sgpr_count; ++sgpr)
    high_pressure.push_back(build_s_mov_b32(/*sdst=*/0u, sgpr, ROCJITSU_CODE_ARCH_CDNA4));
  high_pressure.resize(320u, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  high_pressure.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);

  std::vector<uint8_t> image = make_cdna4_code_object_with_local_function(
      low_pressure, high_pressure, {}, /*vgpr_granulated=*/0u, /*function_is_kernel=*/true);
  AmdGpuCodeObject object(image.data(), image.size());
  EXPECT_TRUE(object.is_valid());
  EXPECT_EQ(object.kernels().size(), 2u);
  if (!object.is_valid() || object.kernels().size() != 2u)
    return {};
  KD descriptor{};
  const uint64_t low_descriptor_offset = object.kernels()[0].descriptor_file_offset;
  std::memcpy(&descriptor, image.data() + low_descriptor_offset, sizeof(descriptor));
  AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                  kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 7u);
  std::memcpy(image.data() + low_descriptor_offset, &descriptor, sizeof(descriptor));
  const uint64_t high_descriptor_offset = object.kernels()[1].descriptor_file_offset;
  std::memcpy(&descriptor, image.data() + high_descriptor_offset, sizeof(descriptor));
  // Thirteen allocation granules give 104 decoded SGPRs and physical VCC at
  // s98:s99, leaving no legal fresh pair at the top of the ordinary file.
  AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                  kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 12u);
  std::memcpy(image.data() + high_descriptor_offset, &descriptor, sizeof(descriptor));
  append_kernel_metadata_note(image, "lds_helper", /*uses_dynamic_stack=*/true,
                              /*sgpr_count=*/static_cast<uint8_t>(live_sgpr_count));
  return image;
}

struct TwoKernelSharedFixtureOptions {
  uint32_t first_vgpr_granulated = kRdna4Wave64AllVgprsGranulated;
  uint32_t second_vgpr_granulated = kRdna4Wave64AllVgprsGranulated;
  uint32_t first_private_bytes = 0;
  uint32_t second_private_bytes = 0;
  bool first_wave32 = false;
  bool second_wave32 = false;
  bool first_continuation_uses_v1 = false;
  std::vector<uint16_t> first_continuation_live_vgprs;
  std::vector<uint16_t> second_continuation_live_vgprs;
  std::vector<uint16_t> first_continuation_live_sgprs;
  std::vector<uint16_t> second_continuation_live_sgprs;
  bool helper_keeps_v1_v3_live = false;
  bool helper_has_ordinary_memory = false;
  bool helper_has_ordered_atomic = false;
  bool helper_atomic_acquire_release = false;
  bool helper_has_barrier = false;
  bool unrelated_has_lds = false;
  bool unrelated_has_barrier = false;
  bool use_indirect_calls = false;
  uint32_t entry_nop_words = 0;
  uint32_t group_bytes = 0;
};

std::vector<uint8_t> make_two_kernel_shared_helper_code_object(
    const TwoKernelSharedFixtureOptions &options, rj_code_arch_t arch,
    std::span<const uint32_t> architecture_specific_helper = {}) {
  if (arch != ROCJITSU_CODE_ARCH_RDNA4 && arch != ROCJITSU_CODE_ARCH_CDNA4) {
    ADD_FAILURE() << "shared-helper fixture supports only RDNA4 and CDNA4";
    return {};
  }
  if (arch == ROCJITSU_CODE_ARCH_CDNA4 &&
      (architecture_specific_helper.empty() || options.first_wave32 || options.second_wave32 ||
       options.first_continuation_uses_v1 || !options.first_continuation_live_vgprs.empty() ||
       !options.second_continuation_live_vgprs.empty() ||
       !options.first_continuation_live_sgprs.empty() ||
       !options.second_continuation_live_sgprs.empty() || options.helper_keeps_v1_v3_live ||
       options.helper_has_ordinary_memory || options.helper_has_ordered_atomic ||
       options.helper_atomic_acquire_release || options.helper_has_barrier ||
       options.unrelated_has_lds || options.unrelated_has_barrier || options.use_indirect_calls ||
       options.group_bytes != 0u)) {
    ADD_FAILURE() << "CDNA4 shared-helper fixture accepts only an architecture-specific helper, "
                     "VGPR counts, private sizes, and an entry NOP prefix";
    return {};
  }
  constexpr uint16_t kReturnSgpr = 30;
  constexpr uint16_t kPcSgpr = 2;
  constexpr uint32_t kLiteralOperand = 255;
  const size_t entry_call_words = options.use_indirect_calls ? 7u : 1u;
  std::vector<uint32_t> first_kernel(options.entry_nop_words + entry_call_words,
                                     build_s_nop(0u, arch));
  if (options.first_continuation_uses_v1) {
    first_kernel.push_back(build_v_mov_b32_e32(/*vdst=*/1, vector_source_vgpr(1), arch));
  }
  for (uint16_t vgpr : options.first_continuation_live_vgprs)
    first_kernel.push_back(build_v_mov_b32_e32(vgpr, vector_source_vgpr(vgpr), arch));
  for (uint16_t sgpr : options.first_continuation_live_sgprs) {
    first_kernel.push_back(build_s_mov_b32(/*sdst=*/0, sgpr, ROCJITSU_CODE_ARCH_RDNA4));
  }
  first_kernel.push_back(build_s_endpgm(arch));
  std::vector<uint32_t> second_kernel(options.entry_nop_words + entry_call_words,
                                      build_s_nop(0u, arch));
  for (uint16_t vgpr : options.second_continuation_live_vgprs)
    second_kernel.push_back(build_v_mov_b32_e32(vgpr, vector_source_vgpr(vgpr), arch));
  for (uint16_t sgpr : options.second_continuation_live_sgprs) {
    second_kernel.push_back(build_s_mov_b32(/*sdst=*/105, sgpr, ROCJITSU_CODE_ARCH_RDNA4));
  }
  second_kernel.push_back(build_s_endpgm(arch));
  std::vector<uint32_t> unrelated_kernel;
  if (options.unrelated_has_lds) {
    unrelated_kernel.push_back(0xD8340000u);
    unrelated_kernel.push_back(0x00000000u); // ds_store_b32 v0, v0
  }
  if (options.unrelated_has_barrier)
    unrelated_kernel.push_back(0xBF940000u); // s_barrier_wait -1
  unrelated_kernel.push_back(build_s_endpgm(arch));
  std::vector<uint32_t> helper;
  if (!architecture_specific_helper.empty()) {
    helper.assign(architecture_specific_helper.begin(), architecture_specific_helper.end());
  } else if (options.helper_has_barrier) {
    helper = {
        0xD8340000u,
        0x00000000u, // ds_store_b32 v0, v0
        0xBE804EC1u, // s_barrier_signal -1
        0xBF94FFFFu, // s_barrier_wait -1
        0xD8D80000u,
        0x00000000u, // ds_load_b32 v0, v0
    };
  } else if (options.helper_has_ordinary_memory) {
    helper = {
        0xEE050004u, 7u | (2u << 18u) | (1u << 20u),
        10u | (0xfffff0u << 8u), // global_load_b32 v7, v10, s[4:5] offset:-16
    };
  } else if (options.helper_has_ordered_atomic) {
    const auto atomic = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
        /*vaddr=*/0, /*vsrc=*/2, /*vdst=*/3, /*return_old_value=*/false, /*scope=*/2,
        ROCJITSU_CODE_ARCH_RDNA4);
    if (!atomic)
      return {};
    helper = {
        0xEE0B0000u,  0x00000000u,  0x00000000u, // global_wb
        (*atomic)[0], (*atomic)[1], (*atomic)[2],
    };
    if (options.helper_atomic_acquire_release)
      helper.insert(helper.end(), {0xEE0AC000u, 0x00000000u, 0x00000000u}); // global_inv
  } else {
    helper = {
        0xD8340000u,
        0x00000000u, // ds_store_b32 v0, v0
    };
  }
  if (options.helper_keeps_v1_v3_live) {
    helper.push_back(build_v_mov_b32_e32(/*vdst=*/1, vector_source_vgpr(1), arch));
    helper.push_back(build_v_mov_b32_e32(/*vdst=*/2, vector_source_vgpr(2), arch));
    helper.push_back(build_v_mov_b32_e32(/*vdst=*/3, vector_source_vgpr(3), arch));
  }
  helper.push_back(build_s_setpc_b64(kReturnSgpr, arch));

  const uint64_t first_entry = 0;
  const uint64_t second_entry = first_kernel.size() * sizeof(uint32_t);
  const uint64_t unrelated_entry = (first_kernel.size() + second_kernel.size()) * sizeof(uint32_t);
  const uint64_t helper_entry =
      (first_kernel.size() + second_kernel.size() + unrelated_kernel.size()) * sizeof(uint32_t);
  const auto encode_call = [&](std::vector<uint32_t> &kernel, uint64_t entry) {
    const size_t call_index = options.entry_nop_words;
    const uint64_t call_pc = entry + call_index * sizeof(uint32_t);
    if (!options.use_indirect_calls) {
      kernel[call_index] = build_s_call_b64(
          kReturnSgpr,
          static_cast<int16_t>((helper_entry - (call_pc + sizeof(uint32_t))) / sizeof(uint32_t)),
          arch);
      return;
    }
    const int64_t delta =
        static_cast<int64_t>(helper_entry) - static_cast<int64_t>(call_pc + sizeof(uint32_t));
    kernel[call_index] = pack_sop1(/*s_getpc_b64=*/0x47, kPcSgpr, 0);
    kernel[call_index + 1u] = pack_sop1(/*s_sext_i32_i16=*/0x0f, kPcSgpr + 1, kPcSgpr + 1);
    kernel[call_index + 2u] = pack_sop2(/*s_add_co_u32=*/0, kPcSgpr, kPcSgpr, kLiteralOperand);
    kernel[call_index + 3u] = static_cast<uint32_t>(delta);
    kernel[call_index + 4u] =
        pack_sop2(/*s_add_co_ci_u32=*/4, kPcSgpr + 1, kPcSgpr + 1, kLiteralOperand);
    kernel[call_index + 5u] = static_cast<uint32_t>(delta >> 32);
    kernel[call_index + 6u] = pack_sop1(/*s_swappc_b64=*/0x49, kReturnSgpr, kPcSgpr);
  };
  encode_call(first_kernel, first_entry);
  encode_call(second_kernel, second_entry);

  constexpr uint64_t text_offset = 0x100;
  constexpr uint64_t text_vaddr = 0x1100;
  constexpr uint64_t rodata_vaddr = 0x2100;
  constexpr uint64_t descriptor_size = sizeof(KD);
  const uint64_t first_kernel_size = first_kernel.size() * sizeof(uint32_t);
  const uint64_t second_kernel_size = second_kernel.size() * sizeof(uint32_t);
  const uint64_t unrelated_kernel_size = unrelated_kernel.size() * sizeof(uint32_t);
  const uint64_t helper_size = helper.size() * sizeof(uint32_t);
  const uint64_t text_size =
      first_kernel_size + second_kernel_size + unrelated_kernel_size + helper_size;

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_elf_name(shstrtab, ".text");
  const uint32_t rodata_name = add_elf_name(shstrtab, ".rodata");
  const uint32_t symtab_name = add_elf_name(shstrtab, ".symtab");
  const uint32_t strtab_name = add_elf_name(shstrtab, ".strtab");
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");

  std::vector<uint8_t> strtab{'\0'};
  const uint32_t helper_symbol_name = add_elf_name(strtab, "shared_lds_helper");
  const uint32_t first_kernel_name = add_elf_name(strtab, "shared_owner_0");
  const uint32_t second_kernel_name = add_elf_name(strtab, "shared_owner_1");
  const uint32_t unrelated_kernel_name = add_elf_name(strtab, "unrelated_kernel");
  const uint32_t first_kd_name = add_elf_name(strtab, "shared_owner_0.kd");
  const uint32_t second_kd_name = add_elf_name(strtab, "shared_owner_1.kd");
  const uint32_t unrelated_kd_name = add_elf_name(strtab, "unrelated_kernel.kd");

  const uint64_t rodata_offset = text_offset + text_size;
  const uint64_t strtab_offset = rodata_offset + 3u * descriptor_size;
  const uint64_t symtab_offset = align_up(strtab_offset + strtab.size(), 8);
  constexpr size_t symbol_count = 8;
  const uint64_t shstrtab_offset = symtab_offset + symbol_count * sizeof(Elf64_Sym);
  const uint64_t shoff = align_up(shstrtab_offset + shstrtab.size(), 8);
  constexpr uint16_t section_count = 6;
  std::vector<uint8_t> image(shoff + section_count * sizeof(Elf64_Shdr), 0);

  Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr.e_ident[EI_DATA] = 1;
  ehdr.e_ident[EI_VERSION] = 1;
  ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  ehdr.e_ident[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  ehdr.e_type = ET_DYN;
  ehdr.e_machine = EM_AMDGPU;
  ehdr.e_version = 1;
  ehdr.e_shoff = shoff;
  ehdr.e_flags = arch == ROCJITSU_CODE_ARCH_CDNA4 ? EF_AMDGPU_MACH_AMDGCN_GFX950
                                                  : EF_AMDGPU_MACH_AMDGCN_GFX1201;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = section_count;
  ehdr.e_shstrndx = 5;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  std::memcpy(image.data() + text_offset, first_kernel.data(), first_kernel_size);
  std::memcpy(image.data() + text_offset + first_kernel_size, second_kernel.data(),
              second_kernel_size);
  std::memcpy(image.data() + text_offset + first_kernel_size + second_kernel_size,
              unrelated_kernel.data(), unrelated_kernel_size);
  std::memcpy(image.data() + text_offset + first_kernel_size + second_kernel_size +
                  unrelated_kernel_size,
              helper.data(), helper_size);

  const auto write_descriptor = [&](uint64_t offset, uint64_t descriptor_vaddr,
                                    uint64_t entry_text_offset, uint32_t vgpr_granulated,
                                    uint32_t private_bytes, bool wave32) {
    KD descriptor{};
    descriptor.kernel_code_entry_byte_offset =
        static_cast<int64_t>(text_vaddr + entry_text_offset) -
        static_cast<int64_t>(descriptor_vaddr);
    descriptor.private_segment_fixed_size = private_bytes;
    descriptor.group_segment_fixed_size = options.group_bytes;
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, vgpr_granulated);
    if (arch == ROCJITSU_CODE_ARCH_CDNA4) {
      AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 2u);
    }
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT,
                    private_bytes != 0 ? 1u : 0u);
    const uint32_t wave32_value = wave32 ? 1u : 0u;
    AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                    kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32, wave32_value);
    std::memcpy(image.data() + offset, &descriptor, sizeof(descriptor));
  };
  write_descriptor(rodata_offset, rodata_vaddr, first_entry, options.first_vgpr_granulated,
                   options.first_private_bytes, options.first_wave32);
  write_descriptor(rodata_offset + descriptor_size, rodata_vaddr + descriptor_size, second_entry,
                   options.second_vgpr_granulated, options.second_private_bytes,
                   options.second_wave32);
  write_descriptor(rodata_offset + 2u * descriptor_size, rodata_vaddr + 2u * descriptor_size,
                   unrelated_entry,
                   arch == ROCJITSU_CODE_ARCH_CDNA4 ? options.first_vgpr_granulated
                                                    : kRdna4Wave64AllVgprsGranulated,
                   0, false);
  std::memcpy(image.data() + strtab_offset, strtab.data(), strtab.size());
  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());

  std::array<Elf64_Sym, symbol_count> symbols{};
  symbols[1].st_name = helper_symbol_name;
  symbols[1].st_info = elf_symbol_info(kElfSymbolBindLocal, kElfSymbolTypeFunc);
  symbols[1].st_shndx = 1;
  symbols[1].st_value = text_vaddr + helper_entry;
  symbols[1].st_size = helper_size;
  symbols[2].st_name = first_kernel_name;
  symbols[2].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeFunc);
  symbols[2].st_shndx = 1;
  symbols[2].st_value = text_vaddr + first_entry;
  symbols[2].st_size = first_kernel_size;
  symbols[3].st_name = second_kernel_name;
  symbols[3].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeFunc);
  symbols[3].st_shndx = 1;
  symbols[3].st_value = text_vaddr + second_entry;
  symbols[3].st_size = second_kernel_size;
  symbols[4].st_name = unrelated_kernel_name;
  symbols[4].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeFunc);
  symbols[4].st_shndx = 1;
  symbols[4].st_value = text_vaddr + unrelated_entry;
  symbols[4].st_size = unrelated_kernel_size;
  symbols[5].st_name = first_kd_name;
  symbols[5].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeObject);
  symbols[5].st_shndx = 2;
  symbols[5].st_value = rodata_vaddr;
  symbols[5].st_size = descriptor_size;
  symbols[6].st_name = second_kd_name;
  symbols[6].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeObject);
  symbols[6].st_shndx = 2;
  symbols[6].st_value = rodata_vaddr + descriptor_size;
  symbols[6].st_size = descriptor_size;
  symbols[7].st_name = unrelated_kd_name;
  symbols[7].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeObject);
  symbols[7].st_shndx = 2;
  symbols[7].st_value = rodata_vaddr + 2u * descriptor_size;
  symbols[7].st_size = descriptor_size;
  std::memcpy(image.data() + symtab_offset, symbols.data(), symbols.size() * sizeof(Elf64_Sym));

  std::array<Elf64_Shdr, section_count> sections{};
  sections[1].sh_name = text_name;
  sections[1].sh_type = SHT_PROGBITS;
  sections[1].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  sections[1].sh_addr = text_vaddr;
  sections[1].sh_offset = text_offset;
  sections[1].sh_size = text_size;
  sections[1].sh_addralign = sizeof(uint32_t);
  sections[2].sh_name = rodata_name;
  sections[2].sh_type = SHT_PROGBITS;
  sections[2].sh_flags = SHF_ALLOC;
  sections[2].sh_addr = rodata_vaddr;
  sections[2].sh_offset = rodata_offset;
  sections[2].sh_size = 3u * descriptor_size;
  sections[2].sh_addralign = 64;
  sections[3].sh_name = symtab_name;
  sections[3].sh_type = SHT_SYMTAB;
  sections[3].sh_offset = symtab_offset;
  sections[3].sh_size = symbol_count * sizeof(Elf64_Sym);
  sections[3].sh_link = 4;
  sections[3].sh_info = 2;
  sections[3].sh_addralign = 8;
  sections[3].sh_entsize = sizeof(Elf64_Sym);
  sections[4].sh_name = strtab_name;
  sections[4].sh_type = SHT_STRTAB;
  sections[4].sh_offset = strtab_offset;
  sections[4].sh_size = strtab.size();
  sections[4].sh_addralign = 1;
  sections[5].sh_name = shstrtab_name;
  sections[5].sh_type = SHT_STRTAB;
  sections[5].sh_offset = shstrtab_offset;
  sections[5].sh_size = shstrtab.size();
  sections[5].sh_addralign = 1;
  std::memcpy(image.data() + shoff, sections.data(), sections.size() * sizeof(Elf64_Shdr));
  return image;
}

std::vector<uint8_t>
make_rdna4_two_kernel_shared_helper_code_object(const TwoKernelSharedFixtureOptions &options = {}) {
  return make_two_kernel_shared_helper_code_object(options, ROCJITSU_CODE_ARCH_RDNA4);
}

std::vector<uint8_t> make_cdna4_two_kernel_shared_sampled_atomic_code_object(
    uint32_t first_vgpr_granulated, uint32_t second_vgpr_granulated, uint32_t first_private_bytes,
    uint32_t second_private_bytes) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  const auto access =
      build_cdna4_ds_store_b32(/*vaddr=*/10, /*vdata=*/11, /*byte_offset=*/0, kArch);
  const auto release = cdna4::build_mubuf(cdna4::kBufferWbl2Mubuf, {.sc1 = 1});
  const auto atomic = build_cdna4_flat_atomic_add_u32(
      /*vaddr=*/4, /*vsrc=*/6, /*vdst=*/7, /*return_old_value=*/true,
      /*scope=*/2, kArch);
  const auto wait = build_cdna4_s_wait_flat0(kArch);
  const auto acquire = cdna4::build_mubuf(cdna4::kBufferInvMubuf, {.sc1 = 1});
  if (!access || !atomic || !wait)
    return {};
  std::vector<uint32_t> helper;
  helper.insert(helper.end(), access->begin(), access->end());
  helper.insert(helper.end(), release.begin(), release.end());
  helper.push_back(*wait);
  helper.insert(helper.end(), atomic->begin(), atomic->end());
  helper.push_back(*wait);
  helper.insert(helper.end(), acquire.begin(), acquire.end());

  TwoKernelSharedFixtureOptions options;
  options.first_vgpr_granulated = first_vgpr_granulated;
  options.second_vgpr_granulated = second_vgpr_granulated;
  options.first_private_bytes = first_private_bytes;
  options.second_private_bytes = second_private_bytes;
  return make_two_kernel_shared_helper_code_object(options, kArch, helper);
}

std::vector<uint8_t> make_rdna4_supported_lds_code_object() {
  const std::array<uint32_t, 6> text_words = {
      0xD8340000u, 0x00000000u, // ds_store_b32
      0xD8D80000u, 0x00000000u, // ds_load_b32
      0xBFC60000u,              // s_wait_dscnt
      0xBFB00000u,              // s_endpgm
  };
  return make_rdna4_lds_code_object(text_words);
}

std::vector<uint8_t> make_rdna4_unsupported_lds_code_object() {
  const std::array<uint32_t, 11> text_words = {
      0xD8340000u, 0x00000000u, // ds_store_b32
      0xD8D80000u, 0x00000000u, // ds_load_b32
      0xD8000000u, 0x00000000u, // ds_add_u32
      0xBF940000u,              // s_barrier_wait
      0xBFC60000u,              // s_wait_dscnt
      0xF4042000u, 0x00000000u, // s_dcache_inv
      0xBFB00000u,              // s_endpgm
  };
  return make_rdna4_lds_code_object(text_words);
}

constexpr uint64_t kInlineShadowFullLdsReportBufferSize =
    sizeof(ConSanMoiReportHeader) +
    kConSanMoiInlineShadowDefaultDiagnosticCapacity * sizeof(ConSanMoiDiagnosticRecord) +
    kConSanMoiInlineShadowAtomicReleaseSlotCapacity * sizeof(ConSanMoiInlineAtomicReleaseSlot) +
    kConSanMoiInlineShadowAtomicReleaseSlotCapacity * sizeof(ConSanMoiInlineCausalSnapshot) +
    kConSanMoiInlineShadowAcquiredEpochTokenSlotCapacity *
        sizeof(ConSanMoiInlineAcquiredEpochTokenSlot) +
    kConSanMoiInlineShadowConservativeExactShadowEntries * sizeof(ConSanMoiInlineExactShadowSlot);

std::vector<uint8_t> make_rdna4_flat_memory_code_object() {
  const std::array<uint32_t, 13> text_words = {
      0xEC050000u, 0x00000000u, 0x00000000u, // flat_load_b32
      0xEC068000u, 0x00000000u, 0x00000000u, // flat_store_b32
      0xEE050000u, 0x00000000u, 0x00000000u, // global_load_b32
      0xED050000u, 0x00000000u, 0x00000000u, // scratch_load_b32
      0xBFB00000u,                           // s_endpgm
  };
  return make_rdna4_lds_code_object(text_words);
}

std::vector<uint8_t> make_rdna4_global_atomic_code_object() {
  const std::array<uint32_t, 4> text_words = {
      0xEE158004u, 0x00980000u,
      0x00000002u, // global_atomic_add_f32 v0, v2, v1, s[4:5] th:return scope:device
      0xBFB00000u, // s_endpgm
  };
  return make_rdna4_lds_code_object(text_words);
}

std::vector<uint8_t> make_rdna4_ordered_global_atomic_code_object() {
  const std::array<uint32_t, 7> text_words = {
      0xEE0B0000u, // global_wb
      0x00000000u, 0x00000000u, 0xEE158004u, 0x00980000u,
      0x00000002u, // global_atomic_add_f32 v0, v2, v1, s[4:5] th:return scope:device
      0xBFB00000u, // s_endpgm
  };
  return make_rdna4_lds_code_object(text_words);
}

std::vector<uint8_t>
make_rdna4_release_wait_no_return_bitwise_code_object(uint32_t wait_word = 0xbfc90000u) {
  const std::array<uint32_t, 5> text_words = {
      wait_word,   0xEE0F0006u, 0x00880000u,
      0x00000000u, // global_atomic_and_b32 v0, v1, s[6:7], no-return, device
      0xBFB00000u, // s_endpgm
  };
  return make_rdna4_lds_code_object(text_words);
}

std::vector<uint8_t> make_rdna4_lds_store_and_release_wait_no_return_bitwise_code_object() {
  const std::array<uint32_t, 7> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32
      0xBFC90000u, 0xEE0F0006u, 0x00880000u,
      0x00000000u, // global_atomic_and_b32 v0, v1, s[6:7], no-return, device
      0xBFB00000u, // s_endpgm
  };
  return make_rdna4_lds_code_object(text_words, "lds_store_and_release_and");
}

std::vector<uint8_t> make_rdna4_ordered_global_cas_code_object(bool return_old_value = true,
                                                               bool vector_only_address = false) {
  const auto wait_store = build_s_wait_storecnt0(ROCJITSU_CODE_ARCH_RDNA4);
  if (!wait_store)
    return {};
  // global_atomic_cmpswap_b32 v0, v2, v[4:5], s[4:5] (or v[2:3] with
  // saddr=off), th:return/no-return, scope:device.
  const uint32_t first =
      0xEE0D0000u | (vector_only_address ? static_cast<uint32_t>(rdna4::OPR_SREG_NULL) : 4u);
  const uint32_t second = return_old_value ? 0x02180000u : 0x02080000u;
  const std::array<uint32_t, 12> text_words = {
      0xEE0B0000u, 0x00000000u, 0x00000000u, // global_wb
      *wait_store, first,       second,      0x00000002u,
      *wait_store, 0xEE0AC000u, 0x00000000u, 0x00000000u, // global_inv
      0xBFB00000u,                                        // s_endpgm
  };
  return make_rdna4_lds_code_object(text_words);
}

std::vector<uint8_t> make_rdna4_buffer_atomic_code_object() {
  // buffer_atomic_add_u32 v1, v2, s[4:7], 0 th:return scope:device
  const std::array<uint32_t, 4> text_words = {
      0xC40D4000u, 1u | (4u << 9u) | (2u << 18u) | (1u << 20u), 2u,
      0xBFB00000u, // s_endpgm
  };
  return make_rdna4_lds_code_object(text_words, "buffer_atomic_probe");
}

std::vector<uint8_t> make_rdna4_ordered_buffer_atomic_code_object() {
  const std::array<uint32_t, 7> text_words = {
      0xEE0B0000u, // global_wb
      0x00000000u, 0x00000000u, 0xC40D4000u, 1u | (4u << 9u) | (2u << 18u) | (1u << 20u), 2u,
      0xBFB00000u, // s_endpgm
  };
  return make_rdna4_lds_code_object(text_words, "ordered_buffer_atomic_probe");
}

std::vector<uint8_t> make_rdna4_ds_atomic_code_object() {
  const std::array<uint32_t, 3> text_words = {
      0xD8000000u, // ds_add_u32 v0, v0 offset:0
      0x00000000u,
      0xBFB00000u, // s_endpgm
  };
  return make_rdna4_lds_code_object(text_words, "ds_atomic_probe");
}

std::vector<uint8_t> make_rdna4_flat_atomic_code_object() {
  const auto atomic = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/2, /*vsrc=*/1, /*vdst=*/0, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  if (!atomic)
    return {};
  const std::array<uint32_t, 4> text_words = {
      (*atomic)[0], (*atomic)[1], (*atomic)[2],
      0xBFB00000u, // s_endpgm
  };
  return make_rdna4_lds_code_object(text_words);
}

std::vector<uint8_t> make_rdna4_flat_atomic_release_acquire_code_object() {
  const auto release = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/2, /*vsrc=*/1, /*vdst=*/0, /*return_old_value=*/false, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto acquire = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/4, /*vsrc=*/1, /*vdst=*/0, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  if (!release || !acquire)
    return {};
  const std::array<uint32_t, 7> text_words = {
      (*release)[0], (*release)[1], (*release)[2], (*acquire)[0], (*acquire)[1], (*acquire)[2],
      0xBFB00000u, // s_endpgm
  };
  return make_rdna4_lds_code_object(text_words);
}

std::vector<uint8_t> make_rdna4_ordered_flat_atomic_code_object(bool return_old_value = false) {
  const auto atomic = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/2, /*vsrc=*/1, /*vdst=*/0, return_old_value, /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  if (!atomic)
    return {};
  const std::array<uint32_t, 7> text_words = {
      0xEE0B0000u,  0x00000000u,  0x00000000u, // global_wb
      (*atomic)[0], (*atomic)[1], (*atomic)[2],
      0xBFB00000u, // s_endpgm
  };
  return make_rdna4_lds_code_object(text_words);
}

std::vector<uint8_t> make_rdna4_atomic_fence_sequence_code_object() {
  const auto atomic = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/2, /*vsrc=*/1, /*vdst=*/0, /*return_old_value=*/false, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto wait_store = build_s_wait_storecnt0(ROCJITSU_CODE_ARCH_RDNA4);
  if (!atomic || !wait_store)
    return {};
  const std::array<uint32_t, 12> text_words = {
      0xEE0B0000u, 0x00000000u,  0x00000000u, // global_wb
      *wait_store, (*atomic)[0], (*atomic)[1], (*atomic)[2],
      *wait_store, 0xEE0AC000u,  0x00000000u,  0x00000000u, // global_inv
      0xBFB00000u,                                          // s_endpgm
  };
  return make_rdna4_lds_code_object(text_words);
}

std::vector<uint8_t> make_rdna4_ordered_flat_cas_code_object(bool return_old_value = true) {
  const auto atomic = build_flat_atomic_cmpswap_b32_vaddr_vsrc_vdst(
      /*vaddr=*/4, /*vsrc=*/1, /*vdst=*/0, return_old_value, /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  if (!atomic)
    return {};
  const std::array<uint32_t, 7> text_words = {
      0xEE0B0000u,  0x00000000u,  0x00000000u, // global_wb
      (*atomic)[0], (*atomic)[1], (*atomic)[2],
      0xBFB00000u, // s_endpgm
  };
  return make_rdna4_lds_code_object(text_words);
}

std::vector<uint8_t>
make_rdna4_sampled_lds_and_ordered_flat_atomic_code_object(bool compare_exchange = false,
                                                           uint32_t scope = 2u) {
  const auto atomic = compare_exchange
                          ? build_flat_atomic_cmpswap_b32_vaddr_vsrc_vdst(
                                /*vaddr=*/4, /*vsrc=*/1, /*vdst=*/0,
                                /*return_old_value=*/true, scope, ROCJITSU_CODE_ARCH_RDNA4)
                          : build_flat_atomic_add_u32_vaddr_vsrc_vdst(
                                /*vaddr=*/2, /*vsrc=*/1, /*vdst=*/0,
                                /*return_old_value=*/false, scope, ROCJITSU_CODE_ARCH_RDNA4);
  if (!atomic)
    return {};
  std::array<uint32_t, 520> words{};
  words[0] = 0xD8340000u;
  words[1] = 0; // ds_store_b32 v0, v0
  words[2] = 0xEE0B0000u;
  words[3] = 0;
  words[4] = 0; // global_wb
  words[5] = (*atomic)[0];
  words[6] = (*atomic)[1];
  words[7] = (*atomic)[2];
  for (size_t i = 8; i + 1u < words.size(); ++i)
    words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
  return make_rdna4_lds_code_object(words);
}

std::vector<uint8_t> make_rdna4_ordered_flat_atomic_release_acquire_code_object() {
  const auto release = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/2, /*vsrc=*/1, /*vdst=*/0, /*return_old_value=*/false, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto acquire = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/4, /*vsrc=*/1, /*vdst=*/0, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  if (!release || !acquire)
    return {};
  const std::array<uint32_t, 13> text_words = {
      0xEE0B0000u,   0x00000000u,   0x00000000u, // global_wb
      (*release)[0], (*release)[1], (*release)[2], (*acquire)[0], (*acquire)[1],
      (*acquire)[2], 0xEE0AC000u,   0x00000000u,   0x00000000u, // global_inv
      0xBFB00000u,                                              // s_endpgm
  };
  return make_rdna4_lds_code_object(text_words);
}

std::vector<uint8_t> make_rdna4_ordered_flat_atomic_high_sgpr_pressure_code_object() {
  const auto release = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/2, /*vsrc=*/1, /*vdst=*/0, /*return_old_value=*/false, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto acquire = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/4, /*vsrc=*/1, /*vdst=*/0, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  if (!release || !acquire)
    return {};
  std::vector<uint32_t> text_words = {
      0xEE0B0000u,   0x00000000u,   0x00000000u, // global_wb
      (*release)[0], (*release)[1], (*release)[2], (*acquire)[0], (*acquire)[1],
      (*acquire)[2], 0xEE0AC000u,   0x00000000u,   0x00000000u, // global_inv
  };
  // Keep every lower scalar register live across both atomic sites. Dispatch
  // identity must therefore occupy s82:s83 and the atomic-only 20-SGPR probe
  // window must fit in s84:s103; an unnecessary 24-SGPR request has
  // no legal fresh or liveness-dead placement.
  for (uint16_t sgpr = 0; sgpr <= 81u; ++sgpr) {
    const auto use =
        build_s_cmp_eq_u32(sgpr, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
    if (!use)
      return {};
    text_words.push_back(*use);
  }
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4));
  return make_rdna4_lds_code_object(text_words, "atomic_high_sgpr_pressure");
}

std::vector<uint8_t> make_rdna4_lds_and_ordered_flat_atomic_handoff_code_object() {
  const auto release = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/2, /*vsrc=*/1, /*vdst=*/0, /*return_old_value=*/false, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto acquire = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/4, /*vsrc=*/1, /*vdst=*/0, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  if (!release || !acquire)
    return {};
  const std::array<uint32_t, 17> text_words = {
      0xD8340000u,   0x00000000u,                  // ds_store_b32 v0, v0
      0xEE0B0000u,   0x00000000u,   0x00000000u,   // global_wb
      (*release)[0], (*release)[1], (*release)[2], // release RMW
      (*acquire)[0], (*acquire)[1], (*acquire)[2], // acquire RMW
      0xEE0AC000u,   0x00000000u,   0x00000000u,   // global_inv
      0xD8D80000u,   0x06000007u,                  // ds_load_b32 v6, v7
      0xBFB00000u,                                 // s_endpgm
  };
  return make_rdna4_lds_code_object(text_words, "lds_atomic_handoff");
}

std::vector<uint8_t> make_rdna4_ordered_global_atomic_release_acquire_code_object() {
  const std::array<uint32_t, 13> text_words = {
      0xEE0B0000u, 0x00000000u, 0x00000000u, // global_wb
      0xEE158004u, 0x00880000u,
      0x00000002u, // global_atomic_add_f32 v0, v2, v1, s[4:5], no-return, device
      0xEE158004u, 0x00980000u,
      0x00000002u, // global_atomic_add_f32 v0, v2, v1, s[4:5], return, device
      0xEE0AC000u, 0x00000000u, 0x00000000u, // global_inv
      0xBFB00000u,                           // s_endpgm
  };
  return make_rdna4_lds_code_object(text_words, "global_release_acquire");
}

std::vector<uint8_t> make_rdna4_displaced_vglobal_atomic_release_acquire_code_object() {
  const std::array<uint32_t, 11> text_words = {
      0xBFC90000u, // s_wait_storecnt_dscnt 0
      0xEE0F0006u, 0x00880000u,
      0x00000000u, // global_atomic_and_b32 v0, v1, s[6:7], no-return, device
      0xEE0EC07Cu, 0x05080000u,
      0x00001408u, // global_atomic_max_u32 v[8:9], v10, off offset:20, device
      0xEE0AC000u, 0x00000000u,
      0x00000000u, // global_inv scope:device
      0xBFB00000u, // s_endpgm
  };
  return make_rdna4_lds_code_object(text_words, "displaced_vglobal_release_acquire");
}

std::vector<uint8_t> make_rdna4_ordinary_acquire_code_object(uint32_t scope = 2u,
                                                             bool include_cache = true,
                                                             bool store = false,
                                                             int32_t ioffset = 0) {
  const auto wait = build_s_wait_loadcnt0(ROCJITSU_CODE_ARCH_RDNA4);
  if (!wait)
    return {};
  std::vector<uint32_t> words = {
      store ? 0xEE068004u : 0xEE050004u,
      (scope << 18u) | (store ? 7u << 23u : 7u),
      (static_cast<uint32_t>(ioffset) & 0xffffffu) << 8u | 10u,
      *wait,
  };
  if (include_cache)
    words.insert(words.end(), {0xEE0AC000u, 0x00000000u, 0x00000000u});
  words.push_back(0xBFB00000u);
  return make_rdna4_lds_code_object(words, "ordinary_acquire_fault");
}

std::vector<uint8_t>
make_rdna4_bounded_atomic_acquire_code_object(std::span<const uint32_t> atomic,
                                              std::span<const uint32_t> bookkeeping,
                                              std::string_view name = "bounded_atomic_acquire") {
  std::vector<uint32_t> words(atomic.begin(), atomic.end());
  words.insert(words.end(), bookkeeping.begin(), bookkeeping.end());
  words.insert(words.end(), {0xEE0AC07Cu, 0x00080000u, 0x00000000u}); // global_inv dev
  words.push_back(0xBFB00000u);                                       // s_endpgm
  return make_rdna4_lds_code_object(words, name);
}

std::vector<uint8_t> make_rdna4_atomic_acquire_fallthrough_join_code_object() {
  constexpr std::array<uint32_t, 3> atomic = {0xEE0D400Cu, 0x01980002u, 0x00000002u};
  constexpr std::array<uint32_t, 8> bookkeeping = {
      0xBF88FF9Eu, // s_wait_alu
      0x8C7E007Eu, // s_or_b32 exec restore
      0xBF118008u, // s_cmp_lg_u64
      0xBFC00000u, // s_wait_loadcnt 0
      0x7E000500u, // v_readfirstlane_b32
      0x980080C1u, // s_cselect_b32
      0xBF028016u, // s_cmp_gt_i32
      0xBFC90000u, // s_wait_storecnt_dscnt 0
  };
  // The conditional edge makes global_inv a join-block leader. Its fallthrough
  // path still executes the RMW followed by the exact bounded acquire
  // bookkeeping emitted by the retained hip-moi workload.
  const auto skip_rmw = build_s_cbranch_vccz(
      static_cast<int16_t>(atomic.size() + bookkeeping.size()), ROCJITSU_CODE_ARCH_RDNA4);
  if (!skip_rmw)
    return {};
  std::vector<uint32_t> words{*skip_rmw};
  words.insert(words.end(), atomic.begin(), atomic.end());
  words.insert(words.end(), bookkeeping.begin(), bookkeeping.end());
  words.insert(words.end(), {0xEE0AC07Cu, 0x00080000u, 0x00000000u}); // global_inv dev
  words.push_back(0xBFB00000u);                                       // s_endpgm
  return make_rdna4_lds_code_object(words, "atomic_acquire_fallthrough_join");
}

std::vector<uint8_t> make_rdna4_flag_self_loop_acquire_code_object() {
  const std::array<uint32_t, 14> words = {
      0xEE050006u, 0x00080001u, 0x00000000u, // global_load_b32, scope dev
      0xBFC00000u,                           // s_wait_loadcnt 0
      0x7C9A0280u,                           // v_cmp_ne_u32
      0xBFA4FFFAu,                           // s_cbranch_vccnz to the load
      0xBF118004u,                           // s_cmp_lg_u64
      0xBEAD0080u,                           // s_mov_b32
      0x980080C1u,                           // s_cselect_b32
      0xBF028016u,                           // s_cmp_gt_i32
      0xEE0AC07Cu, 0x00080000u, 0x00000000u, // global_inv dev
      0xBFB00000u,                           // s_endpgm
  };
  return make_rdna4_lds_code_object(words, "flag_self_loop_acquire");
}

std::vector<uint8_t> make_rdna4_successful_cas_self_loop_acquire_code_object() {
  const std::array<uint32_t, 18> words = {
      0xEE0D0006u, 0x00180002u, 0x00000001u, // global_atomic_cmpswap_b32 return
      0xBFC00000u,                           // s_wait_loadcnt 0
      0x7C940480u,                           // v_cmp_eq_u32
      0x8C00006Au,                           // s_or_b32
      0xBF870009u,                           // s_delay_alu
      0x917E007Eu,                           // s_and_not1_b32 exec
      0xBFA6FFF7u,                           // s_cbranch_execnz to the atomic
      0x8C7E007Eu,                           // s_or_b32 exec restore
      0xBF118004u,                           // s_cmp_lg_u64
      0xBEAD0080u,                           // s_mov_b32
      0x980080C1u,                           // s_cselect_b32
      0xBF028016u,                           // s_cmp_gt_i32
      0xEE0AC07Cu, 0x00080000u, 0x00000000u, // global_inv dev
      0xBFB00000u,                           // s_endpgm
  };
  return make_rdna4_lds_code_object(words, "successful_cas_self_loop_acquire");
}

struct ConSanTransformProfile {
  std::string_view name;
  ConSanOptions options;
};

std::array<ConSanTransformProfile, 4> all_consan_transform_profiles() {
  ConSanOptions supercollider;
  supercollider.flavor = ConSanFlavor::SuperCollider;
  supercollider.probe_lds_check_trap = true;
  supercollider.probe_flat_check_trap = true;
  supercollider.probe_trampoline_nop = true;
  supercollider.max_patches = 8;

  return {{{"supercollider", std::move(supercollider)},
           {"record_replay", moi_options(ConSanMoiEngine::RecordReplay)},
           {"inline_shadow", moi_options(ConSanMoiEngine::InlineShadow)},
           {"sampled", moi_options(ConSanMoiEngine::Sampled)}}};
}

std::array<ConSanTransformProfile, 4> all_consan_replacement_profiles() {
  auto profiles = all_consan_transform_profiles();
  for (auto &profile : profiles) {
    ConSanOptions &options = profile.options;
    if (options.flavor != ConSanFlavor::Moi)
      continue;
    options.moi_report_buffer_address = 0x123456780000ull;
    switch (options.moi_engine) {
    case ConSanMoiEngine::RecordReplay:
      options.moi_dynamic_access_records = true;
      options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(8, 0, 0, 0);
      break;
    case ConSanMoiEngine::InlineShadow:
      options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
      break;
    case ConSanMoiEngine::Sampled:
      options.moi_sampled_check = true;
      options.moi_report_buffer_size = direct_sampled_report_bytes(8);
      break;
    }
  }
  return profiles;
}

static_assert(consan_moi_exact_shadow::granule_bytes == 4);
static_assert(consan_moi_exact_shadow::instruction_offset_shift +
                  consan_moi_exact_shadow::instruction_offset_bits ==
              64);
static_assert(consan_moi_sampled_watchpoint::granule_bytes == 4);
static_assert(consan_moi_sampled_watchpoint::count_shift +
                  consan_moi_sampled_watchpoint::count_bits ==
              64);
static_assert(consan_moi_shadow_kind_from_access_kind(ConSanLdsAccessKind::Read) ==
              ConSanMoiShadowAccessKind::Read);
static_assert(consan_moi_shadow_kind_from_access_kind(ConSanLdsAccessKind::Write) ==
              ConSanMoiShadowAccessKind::Write);
static_assert(consan_moi_shadow_kind_from_access_kind(ConSanLdsAccessKind::Atomic) ==
              ConSanMoiShadowAccessKind::Atomic);
static_assert(consan_moi_shadow_kind_from_access_kind(ConSanLdsAccessKind::Other) ==
              ConSanMoiShadowAccessKind::Empty);
static_assert(consan_moi_shadow_kind_conflicts(ConSanMoiShadowAccessKind::Write,
                                               ConSanMoiShadowAccessKind::Read));
static_assert(!consan_moi_shadow_kind_conflicts(ConSanMoiShadowAccessKind::Read,
                                                ConSanMoiShadowAccessKind::Read));
static_assert(!consan_moi_shadow_kind_conflicts(ConSanMoiShadowAccessKind::Atomic,
                                                ConSanMoiShadowAccessKind::Atomic));
static_assert(alignof(ConSanMoiReportHeader) == 8);
static_assert(alignof(ConSanMoiAccessRecord) == 8);
static_assert(alignof(ConSanMoiBarrierRecord) == 8);
static_assert(alignof(ConSanMoiAtomicRecord) == 8);
static_assert(alignof(ConSanMoiInlineAtomicReleaseSlot) == 8);
static_assert(alignof(ConSanMoiDiagnosticRecord) == 8);
static_assert(sizeof(ConSanMoiAccessRecord) == 64);
static_assert(sizeof(ConSanMoiBarrierRecord) == 40);
static_assert(sizeof(ConSanMoiAtomicRecord) == 80);
static_assert(sizeof(ConSanMoiFenceRecord) == 56);
static_assert(sizeof(ConSanMoiInlineAtomicReleaseSlot) == 32);
static_assert(sizeof(ConSanMoiDiagnosticRecord) == 80);
static_assert(sizeof(ConSanMoiRecordReplayTraceHeader) == 72);
static_assert(sizeof(ConSanMoiRecordReplayPcEntry) == 16);
static_assert(sizeof(ConSanMoiRecordReplayWorkgroupRun) == 24);
static_assert(sizeof(ConSanMoiRecordReplayCompactEvent) == 32);
static_assert(sizeof(ConSanMoiRecordReplayCaptureWindow) == 40);

std::vector<uint32_t> make_padded_moi_first_light_text(uint32_t word0, uint32_t word1) {
  std::vector<uint32_t> text_words(170, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words[0] = word0;
  text_words[1] = word1;
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
  return text_words;
}

void expect_moi_first_light_width(uint32_t word0, uint32_t word1, uint32_t expected_width_bits,
                                  ConSanLdsAccessKind expected_kind) {
  const std::vector<uint32_t> text_words = make_padded_moi_first_light_text(word0, word1);
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options();
  options.scratch_vgpr = 20;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.moi_candidates.size(), 1u);
  EXPECT_EQ(result.moi_candidates.front().kind, expected_kind);
  EXPECT_EQ(result.moi_candidates.front().width_bits, expected_width_bits);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineMoiAccessRecordStore);
  EXPECT_EQ(result.patches.front().original_size, 116u * sizeof(uint32_t));
}

std::vector<uint32_t> make_padded_moi_flat_first_light_function_words() {
  std::vector<uint32_t> function_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000000u,              // v_mov_b32_e64 v0, s0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC05007Cu, 0x00000002u, 0x00000000u, // flat_load_b32 v2, v[0:1]
  };
  function_words.resize(180, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  function_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
  return function_words;
}

} // namespace
} // namespace rocjitsu

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
