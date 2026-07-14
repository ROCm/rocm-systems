// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/patch/consan_moi.h"
#include "rocjitsu/code/patch/consan_resource.h"
#include "rocjitsu/code/patch/instruction_builder.h"
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
#include <vector>

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

using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
namespace kd = rocr::llvm::amdhsa;

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

TEST(ConSanResourcePlan, PrefersDeadWindowInsideCurrentAllocation) {
  RegisterSet live;
  live.expand({RegClass::VGPR, 0, 4});
  const ConSanRegisterRequest request = vgpr_request(3, 8, 8);

  const ConSanRegisterPlan plan = plan_consan_registers(request, live);
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::LivenessDead);
  EXPECT_EQ(plan.base, 4);
  EXPECT_EQ(plan.required_descriptor_count, 8);
}

TEST(ConSanResourcePlan, GrowsAboveGuestReferencesAfterDeadSearchFails) {
  RegisterSet live;
  live.expand({RegClass::VGPR, 0, 8});
  const ConSanRegisterRequest request = vgpr_request(3, 8, 8);

  const ConSanRegisterPlan plan = plan_consan_registers(request, live);
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::DescriptorGrowth);
  EXPECT_EQ(plan.base, 8);
  EXPECT_EQ(plan.required_descriptor_count, 11);
}

TEST(ConSanResourcePlan, FullRegisterFileSelectsAllowedLiveVictim) {
  RegisterSet live;
  expand_all_vgprs(live);
  ConSanRegisterRequest request = vgpr_request(3, 256, 256);
  request.forbidden.expand({RegClass::VGPR, 0, 2});

  const ConSanRegisterPlan plan = plan_consan_registers(request, live);
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::SpillRequired);
  EXPECT_EQ(plan.base, 2);
  EXPECT_EQ(plan.required_descriptor_count, 256);
}

TEST(ConSanResourcePlan, ForceSpillBypassesDeadAndGrowthWindows) {
  RegisterSet live;
  ConSanRegisterRequest request = vgpr_request(3, 8, 8);
  request.force_spill = true;
  request.forbidden.expand({RegClass::VGPR, 0, 1});

  const ConSanRegisterPlan plan = plan_consan_registers(request, live);
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::SpillRequired);
  EXPECT_EQ(plan.base, 1);
  EXPECT_EQ(plan.required_descriptor_count, 8);
}

TEST(ConSanResourcePlan, ExplicitOverrideCannotClobberLiveGuestValue) {
  RegisterSet live;
  live.expand({RegClass::VGPR, 4, 1});
  ConSanRegisterRequest request = vgpr_request(1, 8, 8);
  request.explicit_base = 4;

  const ConSanRegisterPlan plan = plan_consan_registers(request, live);
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::Unsupported);
  EXPECT_EQ(plan.reason, ConSanRegisterPlanReason::ExplicitLive);
}

TEST(ConSanResourcePlan, ExplicitFreshOverrideCarriesDescriptorRequirement) {
  RegisterSet live;
  ConSanRegisterRequest request = vgpr_request(3, 8, 8);
  request.alignment = 2;
  request.explicit_base = 10;

  const ConSanRegisterPlan plan = plan_consan_registers(request, live);
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::Explicit);
  EXPECT_EQ(plan.base, 10);
  EXPECT_EQ(plan.required_descriptor_count, 13);
}

TEST(ConSanResourcePlan, ForbiddenFullFileHasTypedFailure) {
  RegisterSet live;
  expand_all_vgprs(live);
  ConSanRegisterRequest request = vgpr_request(3, 256, 256);
  expand_all_vgprs(request.forbidden);

  const ConSanRegisterPlan plan = plan_consan_registers(request, live);
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::Unsupported);
  EXPECT_EQ(plan.reason, ConSanRegisterPlanReason::NoLegalWindow);
}

TEST(ConSanCapabilities, DistinguishesGfx950InventoryFromNativeEmission) {
  const ConSanTargetCapabilities gfx950 = consan_target_capabilities(ROCJITSU_CODE_ARCH_CDNA4);
  EXPECT_EQ(gfx950.lds_access, ConSanNativeSupport::NativeEmission);
  EXPECT_EQ(gfx950.scratch_spill, ConSanNativeSupport::NativeEmission);
  EXPECT_EQ(gfx950.non_scratch_wait, ConSanNativeSupport::NativeEmission);
  EXPECT_EQ(gfx950.report_publication, ConSanNativeSupport::NativeEmission);
  EXPECT_EQ(gfx950.record_replay, ConSanNativeSupport::NativeEmission);
  EXPECT_EQ(gfx950.group_flat_access, ConSanNativeSupport::NativeEmission);
  EXPECT_EQ(gfx950.barrier, ConSanNativeSupport::InventoryOnly);
  EXPECT_EQ(gfx950.atomic, ConSanNativeSupport::NativeEmission);
  EXPECT_EQ(gfx950.workgroup_identity, ConSanNativeSupport::NativeEmission);
  EXPECT_EQ(gfx950.stable_wave_owner, ConSanNativeSupport::NativeEmission);
  EXPECT_EQ(gfx950.supercollider, ConSanNativeSupport::NativeEmission);
  EXPECT_EQ(gfx950.sampled, ConSanNativeSupport::NativeEmission);
  EXPECT_EQ(gfx950.inline_shadow, ConSanNativeSupport::NativeEmission);
  EXPECT_EQ(gfx950.hw_id_owner, ConSanNativeSupport::Unavailable);

  const ConSanTargetCapabilities unknown = consan_target_capabilities(ROCJITSU_CODE_ARCH_INVALID);
  EXPECT_EQ(consan_native_feature_support(unknown, ConSanNativeFeature::LdsAccess),
            ConSanNativeSupport::Unavailable);
  EXPECT_STREQ(consan_native_feature_name(ConSanNativeFeature::GroupFlatAccess),
               "group-flat-access");
  EXPECT_STREQ(consan_native_support_name(ConSanNativeSupport::InventoryOnly), "inventory-only");
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
                                                uint32_t slot_base = 0,
                                                rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4) {
  std::vector<uint32_t> words;
  for (uint16_t i = 0; i < count; ++i) {
    const uint16_t vgpr = static_cast<uint16_t>(base + i);
    const uint32_t offset = slot_base + 4u * i;
    const auto append_instruction = [&](const auto &instruction) {
      if (!instruction)
        return false;
      words.insert(words.end(), instruction->begin(), instruction->end());
      return true;
    };
    if (arch == ROCJITSU_CODE_ARCH_CDNA4) {
      if (!append_instruction(restore
                                  ? build_cdna4_address_free_scratch_load_b32(vgpr, offset, arch)
                                  : build_cdna4_address_free_scratch_store_b32(vgpr, offset, arch)))
        return {};
    } else {
      if (!append_instruction(restore ? build_address_free_scratch_load_b32(vgpr, offset, arch)
                                      : build_address_free_scratch_store_b32(vgpr, offset, arch)))
        return {};
    }
  }
  const auto wait = arch == ROCJITSU_CODE_ARCH_CDNA4
                        ? build_cdna4_s_wait_vmcnt0(arch)
                        : (restore ? build_s_wait_loadcnt0(arch) : build_s_wait_storecnt0(arch));
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

std::vector<uint32_t> make_expected_vgpr_load_words(uint64_t address, uint16_t value_vgpr,
                                                    uint16_t scratch_vgpr) {
  std::vector<uint32_t> words;
  const auto mov_address_lo = build_v_mov_b32_e64_literal(
      scratch_vgpr, static_cast<uint32_t>(address), ROCJITSU_CODE_ARCH_RDNA4);
  const auto mov_address_hi =
      build_v_mov_b32_e64_literal(static_cast<uint16_t>(scratch_vgpr + 1u),
                                  static_cast<uint32_t>(address >> 32u), ROCJITSU_CODE_ARCH_RDNA4);
  const auto load =
      build_flat_load_b32_vaddr_vdst(scratch_vgpr, value_vgpr, ROCJITSU_CODE_ARCH_RDNA4);
  if (!mov_address_lo || !mov_address_hi || !load)
    return words;
  words.insert(words.end(), mov_address_lo->begin(), mov_address_lo->end());
  words.insert(words.end(), mov_address_hi->begin(), mov_address_hi->end());
  words.insert(words.end(), load->begin(), load->end());
  words.push_back(0xBFC00000u);
  return words;
}

std::vector<uint32_t>
make_expected_scalar_store_words(uint64_t address, uint16_t scalar_src, uint16_t scratch_vgpr,
                                 bool shift_right_16 = false, bool mask_low_16 = false,
                                 rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4) {
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + 2u);
  std::vector<uint32_t> words;
  words.push_back(build_v_mov_b32_e32(value_vgpr, scalar_src, arch));
  if (mask_low_16) {
    const auto shift_left =
        build_v_lshlrev_b32_e32(value_vgpr, scalar_positive_inline_u32(16), value_vgpr, arch);
    const auto shift_right =
        build_v_lshrrev_b32_e32(value_vgpr, scalar_positive_inline_u32(16), value_vgpr, arch);
    if (!shift_left || !shift_right)
      return words;
    words.push_back(*shift_left);
    words.push_back(*shift_right);
  }
  if (shift_right_16) {
    const auto shift =
        build_v_lshrrev_b32_e32(value_vgpr, scalar_positive_inline_u32(16), value_vgpr, arch);
    if (!shift)
      return words;
    words.push_back(*shift);
  }
  const auto mov_address_lo =
      build_v_mov_b32_e64_literal(scratch_vgpr, static_cast<uint32_t>(address), arch);
  const auto mov_address_hi = build_v_mov_b32_e64_literal(
      static_cast<uint16_t>(scratch_vgpr + 1u), static_cast<uint32_t>(address >> 32u), arch);
  const auto store = build_flat_store_b32_vaddr_vsrc(scratch_vgpr, value_vgpr, arch);
  if (!mov_address_lo || !mov_address_hi || !store)
    return words;
  words.insert(words.end(), mov_address_lo->begin(), mov_address_lo->end());
  words.insert(words.end(), mov_address_hi->begin(), mov_address_hi->end());
  words.insert(words.end(), store->begin(), store->end());
  return words;
}

constexpr uint16_t ttmp_scalar_operand(uint16_t ttmp) {
  return static_cast<uint16_t>(kScalarOperandTtmpBase + ttmp);
}

std::vector<uint8_t> make_rdna4_lds_code_object(
    std::span<const uint32_t> text_words, std::string_view kernel_name = "lds_probe",
    uint32_t vgpr_granulated = kRdna4Wave64AllVgprsGranulated, bool wave32 = false,
    bool uses_dynamic_stack = false, uint32_t machine = EF_AMDGPU_MACH_AMDGCN_GFX1201) {
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
  ehdr.e_flags = machine;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = section_count;
  ehdr.e_shstrndx = 5;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  std::memcpy(image.data() + text_offset, text_words.data(), text_size);

  static_assert(sizeof(KD) == kernel_descriptor_size);
  KD kernel_descriptor{};
  const int64_t entry_offset =
      static_cast<int64_t>(text_vaddr) - static_cast<int64_t>(rodata_vaddr);
  kernel_descriptor.kernel_code_entry_byte_offset = entry_offset;
  AMDHSA_BITS_SET(kernel_descriptor.compute_pgm_rsrc1,
                  kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, vgpr_granulated);
  if (machine == EF_AMDGPU_MACH_AMDGCN_GFX950) {
    AMDHSA_BITS_SET(kernel_descriptor.compute_pgm_rsrc2,
                    kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO, 1u);
  }
  const uint32_t wave32_value = wave32 ? 1u : 0u;
  AMDHSA_BITS_SET(kernel_descriptor.kernel_code_properties,
                  kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32, wave32_value);
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

std::vector<uint8_t> make_rdna4_code_object_with_local_function(
    std::span<const uint32_t> kernel_words, std::span<const uint32_t> function_words,
    std::span<const uint32_t> tail_words = {},
    uint32_t vgpr_granulated = kRdna4Wave64AllVgprsGranulated) {
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
  std::memcpy(image.data() + rodata_offset, &kernel_descriptor, sizeof(kernel_descriptor));
  std::memcpy(image.data() + strtab_offset, strtab.data(), strtab.size());
  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());

  std::array<Elf64_Sym, sym_count> symbols{};
  symbols[1].st_name = kernel_symbol_name;
  symbols[1].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeFunc);
  symbols[1].st_shndx = 1;
  symbols[1].st_value = text_vaddr;
  symbols[1].st_size = kernel_size;
  symbols[2].st_name = function_symbol_name;
  symbols[2].st_info = elf_symbol_info(kElfSymbolBindLocal, kElfSymbolTypeFunc);
  symbols[2].st_shndx = 1;
  symbols[2].st_value = text_vaddr + kernel_size;
  symbols[2].st_size = function_size;
  symbols[3].st_name = kd_symbol_name;
  symbols[3].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeObject);
  symbols[3].st_shndx = 2;
  symbols[3].st_value = rodata_vaddr;
  symbols[3].st_size = kernel_descriptor_size;
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

struct TwoKernelSharedFixtureOptions {
  rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4;
  uint32_t first_vgpr_granulated = kRdna4Wave64AllVgprsGranulated;
  uint32_t second_vgpr_granulated = kRdna4Wave64AllVgprsGranulated;
  uint32_t first_private_bytes = 0;
  uint32_t second_private_bytes = 0;
  bool first_wave32 = false;
  bool second_wave32 = false;
  bool first_uses_dynamic_stack = false;
  bool second_uses_dynamic_stack = false;
  bool first_continuation_uses_v1 = false;
  bool helper_keeps_v1_v3_live = false;
};

std::vector<uint8_t>
make_two_kernel_shared_helper_code_object(const TwoKernelSharedFixtureOptions &options = {}) {
  constexpr uint16_t kReturnSgpr = 30;
  const uint32_t call_op = options.arch == ROCJITSU_CODE_ARCH_CDNA4 ? 0x15u : 0x14u;
  const uint32_t setpc_op = options.arch == ROCJITSU_CODE_ARCH_CDNA4 ? 0x1du : 0x48u;
  const uint32_t all_vgprs_granulated =
      options.arch == ROCJITSU_CODE_ARCH_CDNA4 ? 31u : kRdna4Wave64AllVgprsGranulated;
  std::vector<uint32_t> first_kernel{0};
  if (options.first_continuation_uses_v1) {
    first_kernel.push_back(build_v_mov_b32_e32(/*vdst=*/1, vector_source_vgpr(1), options.arch));
  }
  first_kernel.push_back(build_s_endpgm(options.arch));
  std::vector<uint32_t> second_kernel{0, build_s_endpgm(options.arch)};
  const std::vector<uint32_t> unrelated_kernel{build_s_endpgm(options.arch)};
  std::vector<uint32_t> helper{
      options.arch == ROCJITSU_CODE_ARCH_CDNA4 ? 0xD81A0000u : 0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
  };
  if (options.helper_keeps_v1_v3_live) {
    helper.push_back(build_v_mov_b32_e32(/*vdst=*/1, vector_source_vgpr(1), options.arch));
    helper.push_back(build_v_mov_b32_e32(/*vdst=*/2, vector_source_vgpr(2), options.arch));
    helper.push_back(build_v_mov_b32_e32(/*vdst=*/3, vector_source_vgpr(3), options.arch));
  }
  helper.push_back(pack_sop1(setpc_op, 0, kReturnSgpr));

  const uint64_t first_entry = 0;
  const uint64_t second_entry = first_kernel.size() * sizeof(uint32_t);
  const uint64_t unrelated_entry = (first_kernel.size() + second_kernel.size()) * sizeof(uint32_t);
  const uint64_t helper_entry =
      (first_kernel.size() + second_kernel.size() + unrelated_kernel.size()) * sizeof(uint32_t);
  first_kernel[0] = pack_sopk(
      call_op, kReturnSgpr,
      static_cast<uint16_t>((helper_entry - (first_entry + sizeof(uint32_t))) / sizeof(uint32_t)));
  second_kernel[0] = pack_sopk(
      call_op, kReturnSgpr,
      static_cast<uint16_t>((helper_entry - (second_entry + sizeof(uint32_t))) / sizeof(uint32_t)));

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
  const uint32_t first_dynamic_stack_name =
      add_elf_name(strtab, "shared_owner_0.has_dyn_sized_stack");
  const uint32_t second_dynamic_stack_name =
      add_elf_name(strtab, "shared_owner_1.has_dyn_sized_stack");
  const uint32_t unrelated_dynamic_stack_name =
      add_elf_name(strtab, "unrelated_kernel.has_dyn_sized_stack");

  const uint64_t rodata_offset = text_offset + text_size;
  const uint64_t strtab_offset = rodata_offset + 3u * descriptor_size;
  const uint64_t symtab_offset = align_up(strtab_offset + strtab.size(), 8);
  constexpr size_t symbol_count = 11;
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
  ehdr.e_flags = options.arch == ROCJITSU_CODE_ARCH_CDNA4 ? EF_AMDGPU_MACH_AMDGCN_GFX950
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
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, vgpr_granulated);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT,
                    private_bytes != 0 ? 1u : 0u);
    if (options.arch == ROCJITSU_CODE_ARCH_CDNA4) {
      AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2,
                      kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO, 1u);
    }
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
                   unrelated_entry, all_vgprs_granulated, 0, false);
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
  symbols[8].st_name = first_dynamic_stack_name;
  symbols[8].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeNotype);
  symbols[8].st_shndx = SHN_ABS;
  symbols[8].st_value = options.first_uses_dynamic_stack ? 1u : 0u;
  symbols[9].st_name = second_dynamic_stack_name;
  symbols[9].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeNotype);
  symbols[9].st_shndx = SHN_ABS;
  symbols[9].st_value = options.second_uses_dynamic_stack ? 1u : 0u;
  symbols[10].st_name = unrelated_dynamic_stack_name;
  symbols[10].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeNotype);
  symbols[10].st_shndx = SHN_ABS;
  symbols[10].st_value = 0u;
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
    kConSanMoiInlineShadowConservativeExactShadowEntries * sizeof(uint64_t);

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
      0xEE158004u,
      0x00980000u,
      0x00000002u, // global_atomic_add_f32 v0, v2, v1, s[4:5] th:return scope:device
      0xBFB00000u, // s_endpgm
  };
  return make_rdna4_lds_code_object(text_words);
}

std::vector<uint8_t> make_rdna4_flat_atomic_code_object() {
  const auto atomic = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/2, /*vsrc=*/1, /*vdst=*/0, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  if (!atomic)
    return {};
  const std::array<uint32_t, 4> text_words = {
      (*atomic)[0],
      (*atomic)[1],
      (*atomic)[2],
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

std::vector<uint8_t> make_gfx950_flat_atomic_release_acquire_code_object() {
  const auto release = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/2, /*vsrc=*/1, /*vdst=*/0, /*return_old_value=*/false, /*scope=*/0,
      ROCJITSU_CODE_ARCH_CDNA4);
  const auto acquire = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/4, /*vsrc=*/1, /*vdst=*/0, /*return_old_value=*/true, /*scope=*/0,
      ROCJITSU_CODE_ARCH_CDNA4);
  if (!release || !acquire)
    return {};
  const std::array<uint32_t, 5> text_words = {
      (*release)[0],
      (*release)[1],
      (*acquire)[0],
      (*acquire)[1],
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  return make_rdna4_lds_code_object(text_words, "gfx950_atomic_handoff",
                                    kRdna4Wave64AllVgprsGranulated, /*wave32=*/false,
                                    /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
}

TEST(ConSan, DisabledModeDoesNotParseCodeObject) {
  const std::vector<uint8_t> bytes = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  ConSanOptions options;
  options.flavor = ConSanFlavor::None;
  options.delay_nops = 32;

  const auto result = try_patch_consan(bytes, options);

  EXPECT_TRUE(result.visited_code_object);
  EXPECT_FALSE(result.modified);
  EXPECT_EQ(result.input_size, bytes.size());
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_TRUE(result.errors.empty());
  EXPECT_TRUE(result.warnings.empty());
  EXPECT_TRUE(result.target_name.empty());
  EXPECT_TRUE(result.kernels.empty());
}

TEST(ConSan, ParsesFlavorNames) {
  EXPECT_EQ(parse_consan_flavor("supercollider"), ConSanFlavor::SuperCollider);
  EXPECT_EQ(parse_consan_flavor("SUPERCOLLIDER"), ConSanFlavor::SuperCollider);
  EXPECT_EQ(parse_consan_flavor("moi"), ConSanFlavor::Moi);
  EXPECT_EQ(parse_consan_flavor("MOI"), ConSanFlavor::Moi);
  EXPECT_EQ(consan_flavor_name(ConSanFlavor::None), std::string_view("none"));
  EXPECT_EQ(consan_flavor_name(ConSanFlavor::SuperCollider), std::string_view("supercollider"));
  EXPECT_EQ(consan_flavor_name(ConSanFlavor::Moi), std::string_view("moi"));
  EXPECT_FALSE(parse_consan_flavor(""));
  EXPECT_FALSE(parse_consan_flavor("context"));
}

TEST(ConSan, ParsesMoiEngineNamesAndAliases) {
  EXPECT_EQ(parse_consan_moi_engine("record_replay"), ConSanMoiEngine::RecordReplay);
  EXPECT_EQ(parse_consan_moi_engine("record-replay"), ConSanMoiEngine::RecordReplay);
  EXPECT_EQ(parse_consan_moi_engine("context"), ConSanMoiEngine::RecordReplay);
  EXPECT_EQ(parse_consan_moi_engine("CONTEXT"), ConSanMoiEngine::RecordReplay);
  EXPECT_EQ(parse_consan_moi_engine("inline_shadow"), ConSanMoiEngine::InlineShadow);
  EXPECT_EQ(parse_consan_moi_engine("inline-shadow"), ConSanMoiEngine::InlineShadow);
  EXPECT_EQ(parse_consan_moi_engine("sampled_watchpoint"), ConSanMoiEngine::Sampled);
  EXPECT_EQ(parse_consan_moi_engine("sampled-watchpoint"), ConSanMoiEngine::Sampled);
  EXPECT_EQ(parse_consan_moi_engine("sampled"), ConSanMoiEngine::Sampled);
  EXPECT_EQ(consan_moi_engine_name(ConSanMoiEngine::RecordReplay),
            std::string_view("record_replay"));
  EXPECT_EQ(consan_moi_engine_name(ConSanMoiEngine::InlineShadow),
            std::string_view("inline_shadow"));
  EXPECT_EQ(consan_moi_engine_name(ConSanMoiEngine::Sampled), std::string_view("sampled"));
  EXPECT_FALSE(parse_consan_moi_engine(""));
  EXPECT_FALSE(parse_consan_moi_engine("moi"));
}

TEST(ConSan, EnabledModeRejectsInvalidCodeObject) {
  const std::vector<uint8_t> bytes = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const auto result = try_patch_consan(bytes, options);

  EXPECT_TRUE(result.visited_code_object);
  EXPECT_FALSE(result.modified);
  EXPECT_EQ(result.input_size, bytes.size());
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_FALSE(result.errors.empty());
  EXPECT_TRUE(result.target_name.empty());
  EXPECT_TRUE(result.kernels.empty());
}

TEST(ConSan, StubRejectsEmptyCodeObject) {
  const std::vector<uint8_t> bytes;
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const auto result = try_patch_consan(bytes, options);

  EXPECT_TRUE(result.visited_code_object);
  EXPECT_FALSE(result.modified);
  EXPECT_EQ(result.input_size, 0u);
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_FALSE(result.errors.empty());
}

TEST(ConSanMoi, RecordReplayEngineInventoriesCodeObjectWithoutModification) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::RecordReplay;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.visited_code_object);
  EXPECT_FALSE(result.modified);
  EXPECT_EQ(result.flavor, ConSanFlavor::Moi);
  EXPECT_EQ(result.moi_engine, ConSanMoiEngine::RecordReplay);
  EXPECT_TRUE(result.elf_bytes.empty());
  ASSERT_EQ(result.kernels.size(), 1u);
  const ConSanKernelInfo &kernel = result.kernels.front();
  ASSERT_TRUE(kernel.uses_dynamic_stack.has_value());
  EXPECT_FALSE(*kernel.uses_dynamic_stack);
  EXPECT_TRUE(kernel.decoded);
  EXPECT_EQ(kernel.preflight_action, ConSanPreflightAction::NotRun);
  EXPECT_EQ(kernel.stats.lds_read_count, 1u);
  EXPECT_EQ(kernel.stats.lds_write_count, 1u);
  ASSERT_EQ(kernel.lds_sites.size(), 2u);
  ASSERT_EQ(result.moi_candidates.size(), 2u);
  EXPECT_EQ(result.moi_candidates[0].source, ConSanMoiCandidateSource::NativeLds);
  EXPECT_EQ(result.moi_candidates[0].kind, ConSanLdsAccessKind::Write);
  EXPECT_TRUE(result.moi_candidates[0].in_kernel);
  EXPECT_EQ(result.moi_candidates[0].container_name, "lds_probe");
  EXPECT_EQ(result.moi_candidates[0].mnemonic, "ds_store_b32");
  EXPECT_EQ(result.moi_candidates[0].text_offset, 0u);
  EXPECT_EQ(result.moi_candidates[0].file_offset, 0x100u);
  ASSERT_TRUE(result.moi_candidates[0].addr_vgpr);
  EXPECT_EQ(*result.moi_candidates[0].addr_vgpr, 0u);
  ASSERT_TRUE(result.moi_candidates[0].data_vgpr);
  EXPECT_EQ(*result.moi_candidates[0].data_vgpr, 0u);
  EXPECT_EQ(result.moi_candidates[1].kind, ConSanLdsAccessKind::Read);
  EXPECT_EQ(result.moi_candidates[1].mnemonic, "ds_load_b32");
  ASSERT_TRUE(result.moi_candidates[1].dst_vgpr);
  EXPECT_EQ(*result.moi_candidates[1].dst_vgpr, 0u);
  ASSERT_EQ(result.resource_plans.size(), 2u);
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
    ASSERT_EQ(plan.owner_descriptor_file_offsets.size(), 1u);
    EXPECT_EQ(plan.owner_descriptor_file_offsets.front(), kernel.descriptor_file_offset);
    EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::LivenessDead);
    EXPECT_EQ(plan.reason, ConSanRegisterPlanReason::None);
    EXPECT_EQ(plan.scratch_vgpr, 1);
    EXPECT_EQ(plan.scratch_vgpr_count, 3);
    EXPECT_EQ(plan.current_vgpr_count, 256);
    EXPECT_EQ(plan.max_referenced_vgpr_count, 1);
    EXPECT_EQ(plan.required_vgpr_count, 256);
    EXPECT_EQ(plan.original_private_segment_size, 0u);
  }
  EXPECT_FALSE(result.warnings.empty());
}

TEST(ConSanMoi, InventoriesDynamicStackMarker) {
  const std::array<uint32_t, 1> text_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "dynamic_stack_kernel", kRdna4Wave64AllVgprsGranulated,
                                 /*wave32=*/false, /*uses_dynamic_stack=*/true);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_TRUE(result.kernels.front().uses_dynamic_stack.has_value());
  EXPECT_TRUE(*result.kernels.front().uses_dynamic_stack);
}

TEST(ConSanMoi, SampledEngineInventoriesCodeObjectWithoutModification) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::Sampled;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_FALSE(result.modified);
  EXPECT_EQ(result.flavor, ConSanFlavor::Moi);
  EXPECT_EQ(result.moi_engine, ConSanMoiEngine::Sampled);
  EXPECT_TRUE(result.elf_bytes.empty());
  ASSERT_EQ(result.kernels.size(), 1u);
  EXPECT_TRUE(result.kernels.front().decoded);
  ASSERT_EQ(result.resource_plans.size(), 2u);
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
    EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::LivenessDead);
    EXPECT_EQ(plan.scratch_vgpr, 1);
    EXPECT_EQ(plan.scratch_vgpr_count, 5);
  }
  EXPECT_EQ(result.kernels.front().preflight_action, ConSanPreflightAction::NotRun);
  bool saw_stub_warning = false;
  for (const std::string &warning : result.warnings)
    saw_stub_warning |= warning.find("sampled") != std::string::npos;
  EXPECT_TRUE(saw_stub_warning);
}

TEST(ConSanMoi, InlineShadowProbePublishesNativeLdsStoreToExactShadow) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::InlineShadow;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 20;
  options.moi_epoch_vgpr = 21;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  EXPECT_EQ(result.flavor, ConSanFlavor::Moi);
  EXPECT_EQ(result.moi_engine, ConSanMoiEngine::InlineShadow);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::TrampolineMoiExactShadowStore);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 8u);
  ASSERT_EQ(result.kernels.size(), 1u);
  EXPECT_TRUE(result.kernels.front().decoded);
  bool saw_inline_shadow_warning = false;
  for (const std::string &warning : result.warnings)
    saw_inline_shadow_warning |= warning.find("exact-shadow publish probe") != std::string::npos;
  EXPECT_TRUE(saw_inline_shadow_warning);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const auto *text_section = patched.text_sections().front();
  ASSERT_EQ(text_section->size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> text_words(text_section->size() / sizeof(uint32_t));
  std::memcpy(text_words.data(), text_section->data(), text_section->size());

  const uint32_t low_literal = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write) |
                               (1u << consan_moi_exact_shadow::generation_shift);
  const std::array<uint32_t, 3> expected_original_access = {
      0xD8340000u,
      0x00000000u,
      0xBFC60000u,
  };
  const auto mov_low = build_v_mov_b32_e64_literal(10, low_literal, ROCJITSU_CODE_ARCH_RDNA4);
  const auto owner_mask = build_v_and_b32_e32_literal(14, consan_moi_exact_shadow::max_owner, 20,
                                                      ROCJITSU_CODE_ARCH_RDNA4);
  const auto owner_shift =
      build_v_lshlrev_b32_e32(14, scalar_positive_inline_u32(consan_moi_exact_shadow::owner_shift),
                              14, ROCJITSU_CODE_ARCH_RDNA4);
  const auto owner_add =
      build_v_add_nc_u32_e32(10, vector_source_vgpr(10), 14, ROCJITSU_CODE_ARCH_RDNA4);
  const auto epoch_mask = build_v_and_b32_e32_literal(14, consan_moi_exact_shadow::max_epoch, 21,
                                                      ROCJITSU_CODE_ARCH_RDNA4);
  const auto epoch_shift =
      build_v_lshlrev_b32_e32(14, scalar_positive_inline_u32(consan_moi_exact_shadow::epoch_shift),
                              14, ROCJITSU_CODE_ARCH_RDNA4);
  const auto epoch_add =
      build_v_add_nc_u32_e32(10, vector_source_vgpr(10), 14, ROCJITSU_CODE_ARCH_RDNA4);
  const auto mov_high = build_v_mov_b32_e64_literal(11, 0, ROCJITSU_CODE_ARCH_RDNA4);
  const auto start_cell_shift = build_v_lshrrev_b32_e32(
      14, scalar_positive_inline_u32(consan_moi_exact_shadow::granule_shift), 0,
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto atomic_swap = build_flat_atomic_swap_b64_vaddr_vsrc_vdst(
      8, 10, 12, /*return_old_value=*/true, /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(mov_low);
  ASSERT_TRUE(owner_mask);
  ASSERT_TRUE(owner_shift);
  ASSERT_TRUE(owner_add);
  ASSERT_TRUE(epoch_mask);
  ASSERT_TRUE(epoch_shift);
  ASSERT_TRUE(epoch_add);
  ASSERT_TRUE(mov_high);
  ASSERT_TRUE(start_cell_shift);
  ASSERT_TRUE(atomic_swap);
  EXPECT_TRUE(contains_subsequence(text_words, expected_original_access));
  EXPECT_NE(std::find(text_words.begin(), text_words.end(), *start_cell_shift), text_words.end());
  std::vector<uint32_t> expected_publish_prefix;
  expected_publish_prefix.insert(expected_publish_prefix.end(), mov_low->begin(), mov_low->end());
  expected_publish_prefix.insert(expected_publish_prefix.end(), owner_mask->begin(),
                                 owner_mask->end());
  expected_publish_prefix.push_back(*owner_shift);
  expected_publish_prefix.push_back(*owner_add);
  expected_publish_prefix.insert(expected_publish_prefix.end(), epoch_mask->begin(),
                                 epoch_mask->end());
  expected_publish_prefix.push_back(*epoch_shift);
  expected_publish_prefix.push_back(*epoch_add);
  expected_publish_prefix.insert(expected_publish_prefix.end(), mov_high->begin(), mov_high->end());
  expected_publish_prefix.insert(expected_publish_prefix.end(), atomic_swap->begin(),
                                 atomic_swap->end());
  EXPECT_TRUE(contains_subsequence(text_words, expected_publish_prefix));
}

TEST(ConSanMoi, InlineShadowAutomaticallyAllocatesPersistentOwnerEpochVgprs) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::InlineShadow;
  options.scratch_vgpr = 8;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.modified) << "candidates=" << result.moi_candidates.size()
                               << " plans=" << result.resource_plans.size()
                               << " patches=" << result.patches.size()
                               << " warnings=" << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.moi_persistent_vgprs_automatic);
  EXPECT_EQ(result.resolved_moi_owner_vgpr, 15);
  EXPECT_EQ(result.resolved_moi_epoch_vgpr, 16);
  ASSERT_EQ(result.patches.size(), 2u);
  const auto prologue = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
  });
  ASSERT_NE(prologue, result.patches.end());
  EXPECT_EQ(std::ranges::count_if(result.patches,
                                  [](const ConSanPatchInfo &patch) {
                                    return patch.kind ==
                                           ConSanPatchKind::TrampolineMoiExactShadowStore;
                                  }),
            1);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  ASSERT_EQ(patched.text_sections().size(), 1u);
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  EXPECT_GE(AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc1,
                            kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT),
            4u);

  std::vector<uint32_t> prologue_words(prologue->trampoline_size / sizeof(uint32_t));
  std::memcpy(prologue_words.data(),
              patched.text_sections().front()->data() + prologue->trampoline_offset,
              prologue->trampoline_size);
  const auto owner_init =
      build_v_lshrrev_b32_e32(15, scalar_positive_inline_u32(6), 0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(owner_init);
  ASSERT_GE(prologue_words.size(), 2u);
  EXPECT_EQ(prologue_words[0], *owner_init);
  EXPECT_EQ(prologue_words[1],
            build_v_mov_b32_e32(16, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(ConSanMoi, InlineShadowAutomaticallyAllocatesScratchAndPersistentVgprs) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::InlineShadow;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.modified);
  EXPECT_TRUE(result.moi_persistent_vgprs_automatic);
  EXPECT_EQ(result.resolved_moi_owner_vgpr, 9);
  EXPECT_EQ(result.resolved_moi_epoch_vgpr, 10);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  ASSERT_NE(access, result.patches.end());
  EXPECT_EQ(access->scratch_vgpr, 2);
  EXPECT_EQ(access->spilled_vgpr_count, 0u);
}

TEST(ConSanMoi, InlineShadowAutomaticallyAllocatesHwIdOwnerAndSpecialStateSgprs) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::InlineShadow;
  options.moi_owner_source = ConSanMoiOwnerSource::HwId;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.modified);
  EXPECT_TRUE(result.moi_owner_sgpr_automatic);
  EXPECT_TRUE(result.moi_exec_save_sgprs_automatic);
  EXPECT_EQ(result.resolved_moi_owner_sgpr, 0);
  EXPECT_EQ(result.resolved_moi_exec_save_sgpr, 2);

  const auto prologue = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
  });
  ASSERT_NE(prologue, result.patches.end());
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  std::vector<uint32_t> prologue_words(prologue->trampoline_size / sizeof(uint32_t));
  std::memcpy(prologue_words.data(),
              patched.text_sections().front()->data() + prologue->trampoline_offset,
              prologue->trampoline_size);
  const auto hwreg = build_hwreg_imm(/*reg_id=*/23, /*offset=*/0, /*size_bits=*/10);
  ASSERT_TRUE(hwreg);
  const auto get_hw_id = build_s_getreg_b32(/*sdst=*/0, *hwreg, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(get_hw_id);
  EXPECT_TRUE(std::find(prologue_words.begin(), prologue_words.end(), *get_hw_id) !=
              prologue_words.end());

  ASSERT_EQ(patched.kernels().size(), 1u);
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  const uint32_t sgpr_granulated = AMDHSA_BITS_GET(
      descriptor.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);
  EXPECT_GE(sgpr_granulated, 1u);
}

TEST(ConSanMoi, InlineShadowPrivateEpochUsesWave32OwnerShift) {
  const std::array<uint32_t, 3> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "wave32_private_epoch", kRdna4Wave64AllVgprsGranulated, /*wave32=*/true);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::InlineShadow;
  options.force_private_epoch = true;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.modified);
  ASSERT_TRUE(result.moi_private_epoch_automatic);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  ASSERT_NE(access, result.patches.end());
  ASSERT_EQ(access->scratch_vgpr, 2);
  EXPECT_EQ(access->persistent_epoch_private_offset, 0u);
  EXPECT_EQ(access->persistent_owner_private_offset, 4u);

  const auto prologue = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue;
  });
  ASSERT_NE(prologue, result.patches.end());

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  std::vector<uint32_t> words(prologue->trampoline_size / sizeof(uint32_t));
  std::memcpy(words.data(), patched.text_sections().front()->data() + prologue->trampoline_offset,
              prologue->trampoline_size);
  const auto owner = build_v_lshrrev_b32_e32(
      /*vdst=*/2, scalar_positive_inline_u32(5), /*workitem_id_x=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(owner);
  EXPECT_TRUE(contains_subsequence(words, std::span<const uint32_t>(&*owner, 1)));
}

TEST(ConSanMoi, InlineShadowDescriptorFullUsesPrivateEpochWithoutSpillOverlap) {
  constexpr uint32_t kBarrierWait = 0xBF940000u;
  const std::array<uint32_t, 5> text_words = {
      build_v_mov_b32_e32(/*vdst=*/255, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4),
      0xD8340000u,
      0x00000000u, // ds_store_b32
      kBarrierWait,
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "descriptor_full_private_epoch", kRdna4Wave64AllVgprsGranulated);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::InlineShadow;
  options.moi_track_barriers = true;
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.moi_private_epoch_automatic);
  EXPECT_FALSE(result.moi_persistent_vgprs_automatic);
  EXPECT_FALSE(result.resolved_moi_owner_vgpr);
  EXPECT_FALSE(result.resolved_moi_epoch_vgpr);

  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  const auto barrier = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiInlineEpochBarrier;
  });
  const auto prologue = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue;
  });
  ASSERT_NE(access, result.patches.end());
  ASSERT_NE(barrier, result.patches.end());
  ASSERT_NE(prologue, result.patches.end());
  ASSERT_TRUE(access->scratch_vgpr);
  EXPECT_EQ(access->scratch_vgpr, 2);
  EXPECT_EQ(access->spilled_vgpr_count, 7u);
  EXPECT_EQ(access->persistent_epoch_private_offset, 0u);
  EXPECT_EQ(access->required_private_segment_size, 44u);
  EXPECT_EQ(barrier->scratch_vgpr, access->scratch_vgpr);
  EXPECT_EQ(barrier->spilled_vgpr_count, 1u);
  EXPECT_EQ(barrier->persistent_epoch_private_offset, 0u);
  EXPECT_EQ(barrier->required_private_segment_size, 44u);
  EXPECT_EQ(prologue->scratch_vgpr, access->scratch_vgpr);
  EXPECT_EQ(prologue->spilled_vgpr_count, 1u);
  EXPECT_EQ(prologue->persistent_epoch_private_offset, 0u);
  EXPECT_EQ(prologue->required_private_segment_size, 44u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  ASSERT_EQ(patched.text_sections().size(), 1u);
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  EXPECT_EQ(descriptor.private_segment_fixed_size, 44u);
  EXPECT_EQ(
      AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT),
      1u);

  const auto patch_words = [&](const ConSanPatchInfo &patch) {
    std::vector<uint32_t> words(patch.trampoline_size / sizeof(uint32_t));
    std::memcpy(words.data(), patched.text_sections().front()->data() + patch.trampoline_offset,
                patch.trampoline_size);
    return words;
  };
  const std::vector<uint32_t> access_words = patch_words(*access);
  const std::vector<uint32_t> barrier_words = patch_words(*barrier);
  const std::vector<uint32_t> prologue_words = patch_words(*prologue);
  const auto epoch_load = build_address_free_scratch_load_b32(
      /*vdst=*/8, /*byte_offset=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  const auto epoch_barrier_load = build_address_free_scratch_load_b32(
      /*vdst=*/2, /*byte_offset=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  const auto epoch_store = build_address_free_scratch_store_b32(
      /*vsrc=*/2, /*byte_offset=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  const auto spill_store = build_address_free_scratch_store_b32(
      /*vsrc=*/2, /*byte_offset=*/16, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(epoch_load);
  ASSERT_TRUE(epoch_barrier_load);
  ASSERT_TRUE(epoch_store);
  ASSERT_TRUE(spill_store);
  EXPECT_TRUE(contains_subsequence(access_words, *spill_store));
  EXPECT_TRUE(contains_subsequence(access_words, *epoch_load));
  EXPECT_TRUE(contains_subsequence(barrier_words, *spill_store));
  EXPECT_TRUE(contains_subsequence(barrier_words, *epoch_barrier_load));
  EXPECT_TRUE(contains_subsequence(barrier_words, *epoch_store));
  EXPECT_TRUE(contains_subsequence(prologue_words, *spill_store));
  EXPECT_TRUE(contains_subsequence(prologue_words, *epoch_store));
}

TEST(ConSanMoi, InlineShadowProbePublishesMultiCellNativeLdsStore) {
  const std::array<uint32_t, 3> input_words = {
      0xDB7C0000u,
      0x00000109u, // ds_store_b128 v0, v[1:4]
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(input_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::InlineShadow;
  options.scratch_vgpr = 16;
  options.moi_owner_vgpr = 24;
  options.moi_epoch_vgpr = 25;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified) << "errors=" << result.errors.size()
                               << " warnings=" << result.warnings.size()
                               << " candidates=" << result.moi_candidates.size()
                               << " first_warning="
                               << (result.warnings.empty() ? "" : result.warnings.front());
  ASSERT_EQ(result.moi_candidates.size(), 1u);
  EXPECT_EQ(result.moi_candidates.front().width_bits, 128u);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::TrampolineMoiExactShadowStore);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const auto *text_section = patched.text_sections().front();
  ASSERT_EQ(text_section->size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> text_words(text_section->size() / sizeof(uint32_t));
  std::memcpy(text_words.data(), text_section->data(), text_section->size());

  const auto atomic_swap = build_flat_atomic_swap_b64_vaddr_vsrc_vdst(
      /*vaddr=*/16, /*vsrc=*/18, /*vdst=*/20, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(atomic_swap);
  EXPECT_EQ(count_subsequence(text_words, *atomic_swap), 4u);

  const auto mov_shadow_low = build_v_mov_b32_e64_literal(
      /*vdst=*/18,
      static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write) |
          (1u << consan_moi_exact_shadow::generation_shift),
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto mov_shadow_high = build_v_mov_b32_e64_literal(
      /*vdst=*/19, /*literal=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(mov_shadow_low);
  ASSERT_TRUE(mov_shadow_high);
  EXPECT_EQ(count_subsequence(text_words, *mov_shadow_low), 4u);
  // The zero literal also initializes the default workgroup partition index.
  EXPECT_EQ(count_subsequence(text_words, *mov_shadow_high), 8u);

  const auto mov_cell_offset =
      build_v_mov_b32_e64_literal(/*vdst=*/16, /*literal=*/3, ROCJITSU_CODE_ARCH_RDNA4);
  const auto add_cell_offset = build_v_add_nc_u32_e32(/*vdst=*/22, vector_source_vgpr(22),
                                                      /*vsrc1=*/16, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(mov_cell_offset);
  ASSERT_TRUE(add_cell_offset);
  std::vector<uint32_t> expected_final_cell_offset;
  expected_final_cell_offset.insert(expected_final_cell_offset.end(), mov_cell_offset->begin(),
                                    mov_cell_offset->end());
  expected_final_cell_offset.push_back(*add_cell_offset);
  EXPECT_TRUE(contains_subsequence(text_words, expected_final_cell_offset));
}

TEST(ConSanMoi, InlineShadowProbeCoversNativeWidthAndTwoAddressFamilies) {
  const auto expect_cell_publications = [](uint32_t word0, uint32_t word1,
                                           std::string_view expected_mnemonic,
                                           uint32_t expected_cells) {
    const std::array<uint32_t, 3> input_words = {
        word0,
        word1,
        build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
    };
    const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(input_words);
    ConSanOptions options;
    options.flavor = ConSanFlavor::Moi;
    options.moi_engine = ConSanMoiEngine::InlineShadow;
    options.scratch_vgpr = 16;
    options.moi_owner_vgpr = 24;
    options.moi_epoch_vgpr = 25;
    options.moi_report_buffer_address = 0x100000000ull;
    options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

    const auto result = try_patch_consan(bytes, options);

    ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
    ASSERT_TRUE(result.modified);
    ASSERT_EQ(result.moi_candidates.size(), 1u);
    EXPECT_EQ(result.moi_candidates.front().mnemonic, expected_mnemonic);

    AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(patched.is_valid());
    ASSERT_EQ(patched.text_sections().size(), 1u);
    const auto *text_section = patched.text_sections().front();
    ASSERT_EQ(text_section->size() % sizeof(uint32_t), 0u);
    std::vector<uint32_t> text_words(text_section->size() / sizeof(uint32_t));
    std::memcpy(text_words.data(), text_section->data(), text_section->size());

    const auto atomic_swap = build_flat_atomic_swap_b64_vaddr_vsrc_vdst(
        /*vaddr=*/16, /*vsrc=*/18, /*vdst=*/20, /*return_old_value=*/true, /*scope=*/2,
        ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(atomic_swap);
    EXPECT_EQ(count_subsequence(text_words, *atomic_swap), expected_cells);
  };

  expect_cell_publications(0xD9D80000u, 0x01000009u, "ds_load_b64", 2u);
  expect_cell_publications(0xDA980000u, 0x01000002u, "ds_load_u16_d16", 1u);
  expect_cell_publications(0xD8380201u, 0x00000000u, "ds_store_2addr_b32", 2u);
  expect_cell_publications(0xD9DC0201u, 0x01000009u, "ds_load_2addr_b64", 4u);
}

TEST(ConSanMoi, InlineShadowProbeCanEmitGpuConflictDiagnostic) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::InlineShadow;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 20;
  options.moi_epoch_vgpr = 21;
  options.moi_exec_save_sgpr = 30;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::TrampolineMoiExactShadowStore);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const auto *text_section = patched.text_sections().front();
  ASSERT_EQ(text_section->size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> text_words(text_section->size() / sizeof(uint32_t));
  std::memcpy(text_words.data(), text_section->data(), text_section->size());

  const uint64_t report_base = *options.moi_report_buffer_address;

  const auto save_scc = build_s_cselect_b32(
      /*sdst=*/40, scalar_positive_inline_u32(1), scalar_positive_inline_u32(0),
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto save_vcc = build_s_mov_b64(/*sdst=*/38, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(save_scc);
  ASSERT_TRUE(save_vcc);
  EXPECT_TRUE(contains_subsequence(text_words, std::array<uint32_t, 2>{*save_scc, *save_vcc}));
  std::vector<uint32_t> expected_conflict_predicate;
  const auto zero = build_v_mov_b32_e64_literal(/*vdst=*/14, 0, ROCJITSU_CODE_ARCH_RDNA4);
  const auto nonempty =
      build_v_cmp_gt_u32_e32_vcc(vector_source_vgpr(12), /*vsrc1=*/14, ROCJITSU_CODE_ARCH_RDNA4);
  const auto narrow_nonempty =
      build_s_and_saveexec_b64(/*sdst=*/30, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto prior_owner = build_v_lshrrev_b32_e32(
      /*vdst=*/14, scalar_positive_inline_u32(consan_moi_exact_shadow::owner_shift), /*vsrc1=*/12,
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto owner_mask = build_v_and_b32_e32_literal(
      /*vdst=*/14, consan_moi_exact_shadow::max_owner, /*vsrc1=*/14, ROCJITSU_CODE_ARCH_RDNA4);
  const auto current_owner = build_v_lshrrev_b32_e32(
      /*vdst=*/11, scalar_positive_inline_u32(consan_moi_exact_shadow::owner_shift), /*vsrc1=*/10,
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto current_owner_mask = build_v_and_b32_e32_literal(
      /*vdst=*/11, consan_moi_exact_shadow::max_owner, /*vsrc1=*/11, ROCJITSU_CODE_ARCH_RDNA4);
  const auto owner_ne =
      build_v_cmp_ne_u32_e32_vcc(vector_source_vgpr(11), /*vsrc1=*/14, ROCJITSU_CODE_ARCH_RDNA4);
  const auto narrow_conflict =
      build_s_and_saveexec_b64(/*sdst=*/32, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto prior_epoch = build_v_lshrrev_b32_e32(
      /*vdst=*/14, scalar_positive_inline_u32(consan_moi_exact_shadow::epoch_shift),
      /*vsrc1=*/12, ROCJITSU_CODE_ARCH_RDNA4);
  const auto epoch_mask = build_v_and_b32_e32_literal(
      /*vdst=*/14, consan_moi_exact_shadow::max_epoch, /*vsrc1=*/14, ROCJITSU_CODE_ARCH_RDNA4);
  const auto current_epoch = build_v_lshrrev_b32_e32(
      /*vdst=*/11, scalar_positive_inline_u32(consan_moi_exact_shadow::epoch_shift), /*vsrc1=*/10,
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto current_epoch_mask = build_v_and_b32_e32_literal(
      /*vdst=*/11, consan_moi_exact_shadow::max_epoch, /*vsrc1=*/11, ROCJITSU_CODE_ARCH_RDNA4);
  const auto epoch_eq =
      build_v_cmp_eq_u32_e32_vcc(vector_source_vgpr(11), /*vsrc1=*/14, ROCJITSU_CODE_ARCH_RDNA4);
  const auto narrow_same_epoch =
      build_s_and_saveexec_b64(/*sdst=*/34, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(zero);
  ASSERT_TRUE(nonempty);
  ASSERT_TRUE(narrow_nonempty);
  ASSERT_TRUE(prior_owner);
  ASSERT_TRUE(owner_mask);
  ASSERT_TRUE(current_owner);
  ASSERT_TRUE(current_owner_mask);
  ASSERT_TRUE(owner_ne);
  ASSERT_TRUE(narrow_conflict);
  ASSERT_TRUE(prior_epoch);
  ASSERT_TRUE(epoch_mask);
  ASSERT_TRUE(current_epoch);
  ASSERT_TRUE(current_epoch_mask);
  ASSERT_TRUE(epoch_eq);
  ASSERT_TRUE(narrow_same_epoch);
  expected_conflict_predicate.insert(expected_conflict_predicate.end(), zero->begin(), zero->end());
  expected_conflict_predicate.push_back(*nonempty);
  expected_conflict_predicate.push_back(*narrow_nonempty);
  expected_conflict_predicate.push_back(*prior_owner);
  expected_conflict_predicate.insert(expected_conflict_predicate.end(), owner_mask->begin(),
                                     owner_mask->end());
  expected_conflict_predicate.push_back(*current_owner);
  expected_conflict_predicate.insert(expected_conflict_predicate.end(), current_owner_mask->begin(),
                                     current_owner_mask->end());
  expected_conflict_predicate.push_back(*owner_ne);
  expected_conflict_predicate.push_back(*narrow_conflict);
  expected_conflict_predicate.push_back(*prior_epoch);
  expected_conflict_predicate.insert(expected_conflict_predicate.end(), epoch_mask->begin(),
                                     epoch_mask->end());
  expected_conflict_predicate.push_back(*current_epoch);
  expected_conflict_predicate.insert(expected_conflict_predicate.end(), current_epoch_mask->begin(),
                                     current_epoch_mask->end());
  expected_conflict_predicate.push_back(*epoch_eq);
  expected_conflict_predicate.push_back(*narrow_same_epoch);
  EXPECT_TRUE(contains_subsequence(text_words, expected_conflict_predicate));

  const auto count_address_lo = build_v_mov_b32_e64_literal(
      /*vdst=*/8,
      static_cast<uint32_t>(report_base + offsetof(ConSanMoiReportHeader, diagnostic_count)),
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto count_address_hi = build_v_mov_b32_e64_literal(
      /*vdst=*/9,
      static_cast<uint32_t>((report_base + offsetof(ConSanMoiReportHeader, diagnostic_count)) >>
                            32u),
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto count_one = build_v_mov_b32_e64_literal(/*vdst=*/11, 1u, ROCJITSU_CODE_ARCH_RDNA4);
  const auto count_add = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/8, /*vsrc=*/11, /*vdst=*/11, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto count_wait = build_s_wait_loadcnt0(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(count_address_lo);
  ASSERT_TRUE(count_address_hi);
  ASSERT_TRUE(count_one);
  ASSERT_TRUE(count_add);
  ASSERT_TRUE(count_wait);
  std::vector<uint32_t> expected_slot_reservation;
  expected_slot_reservation.insert(expected_slot_reservation.end(), count_address_lo->begin(),
                                   count_address_lo->end());
  expected_slot_reservation.insert(expected_slot_reservation.end(), count_address_hi->begin(),
                                   count_address_hi->end());
  expected_slot_reservation.insert(expected_slot_reservation.end(), count_one->begin(),
                                   count_one->end());
  expected_slot_reservation.insert(expected_slot_reservation.end(), count_add->begin(),
                                   count_add->end());
  expected_slot_reservation.push_back(*count_wait);
  EXPECT_TRUE(contains_subsequence(text_words, expected_slot_reservation));

  const auto slot_times_16 = build_v_lshlrev_b32_e32(
      /*vdst=*/9, scalar_positive_inline_u32(4), /*vsrc1=*/11, ROCJITSU_CODE_ARCH_RDNA4);
  const auto slot_times_64 = build_v_lshlrev_b32_e32(
      /*vdst=*/8, scalar_positive_inline_u32(6), /*vsrc1=*/11, ROCJITSU_CODE_ARCH_RDNA4);
  const auto slot_times_80 = build_v_add_nc_u32_e32(
      /*vdst=*/9, vector_source_vgpr(8), /*vsrc1=*/9, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(slot_times_16);
  ASSERT_TRUE(slot_times_64);
  ASSERT_TRUE(slot_times_80);
  const std::array<uint32_t, 3> expected_slot_stride = {
      *slot_times_16,
      *slot_times_64,
      *slot_times_80,
  };
  EXPECT_TRUE(contains_subsequence(text_words, expected_slot_stride));

  const auto restore_exec = build_s_mov_b64(kRdna4ExecLo, /*ssrc0=*/30, ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_vcc = build_s_mov_b64(kRdna4VccLo, /*ssrc0=*/38, ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_scc = build_s_cmp_lg_u32(
      /*ssrc0=*/40, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(restore_exec);
  ASSERT_TRUE(restore_vcc);
  ASSERT_TRUE(restore_scc);
  EXPECT_NE(std::find(text_words.begin(), text_words.end(), *restore_exec), text_words.end());
  EXPECT_TRUE(
      contains_subsequence(text_words, std::array<uint32_t, 2>{*restore_vcc, *restore_scc}));
}

TEST(ConSanMoi, InlineShadowProbeCanPatchTwoAppendedCaveSites) {
  const std::array<uint32_t, 5> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
      0xD8D80000u,
      0x01000000u, // ds_load_b32 v1, v0
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::InlineShadow;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 20;
  options.moi_epoch_vgpr = 21;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 2u);
  EXPECT_EQ(result.patches[0].kind, ConSanPatchKind::TrampolineMoiExactShadowStore);
  EXPECT_EQ(result.patches[0].anchor_offset, 0u);
  EXPECT_EQ(result.patches[0].original_size, 2u * sizeof(uint32_t));
  EXPECT_EQ(result.patches[1].kind, ConSanPatchKind::TrampolineMoiExactShadowStore);
  EXPECT_EQ(result.patches[1].anchor_offset, 2u * sizeof(uint32_t));
  EXPECT_EQ(result.patches[1].original_size, 2u * sizeof(uint32_t));

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  EXPECT_GT(patched.text_sections().front()->size(), text_words.size() * sizeof(uint32_t));
}

TEST(ConSanMoi, InlineShadowProbePublishesNativeLdsLoadAndSuppressesReadRead) {
  const std::array<uint32_t, 4> input_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBFC60000u, // s_wait_dscnt
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(input_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::InlineShadow;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 20;
  options.moi_epoch_vgpr = 21;
  options.moi_exec_save_sgpr = 30;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const auto *text_section = patched.text_sections().front();
  ASSERT_EQ(text_section->size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> text_words(text_section->size() / sizeof(uint32_t));
  std::memcpy(text_words.data(), text_section->data(), text_section->size());

  const uint32_t read_low_literal = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read) |
                                    (1u << consan_moi_exact_shadow::generation_shift);
  const auto mov_read_low =
      build_v_mov_b32_e64_literal(10, read_low_literal, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(mov_read_low);
  EXPECT_TRUE(contains_subsequence(text_words, *mov_read_low));

  const auto prior_kind = build_v_and_b32_e32_literal(
      /*vdst=*/14, static_cast<uint32_t>(consan_moi_exact_shadow::access_kind_mask), /*vsrc1=*/12,
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto kind_ne = build_v_cmp_ne_u32_e32_vcc(
      scalar_positive_inline_u32(static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read)),
      /*vsrc1=*/14, ROCJITSU_CODE_ARCH_RDNA4);
  const auto narrow_kind_conflict =
      build_s_and_saveexec_b64(/*sdst=*/36, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(prior_kind);
  ASSERT_TRUE(kind_ne);
  ASSERT_TRUE(narrow_kind_conflict);
  std::vector<uint32_t> expected_read_kind_filter;
  expected_read_kind_filter.insert(expected_read_kind_filter.end(), prior_kind->begin(),
                                   prior_kind->end());
  expected_read_kind_filter.push_back(*kind_ne);
  expected_read_kind_filter.push_back(*narrow_kind_conflict);
  EXPECT_TRUE(contains_subsequence(text_words, expected_read_kind_filter));
}

TEST(ConSanMoi, InlineShadowProbeRejectsSmallExactShadowCapacity) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::InlineShadow;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 20;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = sizeof(ConSanMoiReportHeader) +
                                   4u * sizeof(ConSanMoiDiagnosticRecord) + 64u * sizeof(uint64_t);

  const auto result = try_patch_consan(bytes, options);

  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  bool saw_capacity_warning = false;
  for (const std::string &warning : result.warnings)
    saw_capacity_warning |= warning.find("full 64 KiB LDS address range") != std::string::npos;
  EXPECT_TRUE(saw_capacity_warning);
}

TEST(ConSanMoi, InventoryIncludesLikelyGroupFlatSitesFromLocalFunctions) {
  const std::array<uint32_t, 1> kernel_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 9> function_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000000u,              // v_mov_b32_e64 v0, s0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC05007Cu, 0x00000002u, 0x00000000u, // flat_load_b32 v2, v[0:1]
      0xBFB00000u,                           // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_EQ(result.functions.size(), 1u);
  ASSERT_EQ(result.functions.front().flat_sites.size(), 1u);
  ASSERT_EQ(result.moi_candidates.size(), 1u);
  const ConSanMoiCandidate &candidate = result.moi_candidates.front();
  EXPECT_EQ(candidate.source, ConSanMoiCandidateSource::FlatGroup);
  EXPECT_EQ(candidate.kind, ConSanLdsAccessKind::Read);
  EXPECT_EQ(candidate.flat_address_space_hint, ConSanFlatAddressSpaceHint::Group);
  EXPECT_FALSE(candidate.in_kernel);
  EXPECT_EQ(candidate.container_name, "lds_helper");
  EXPECT_EQ(candidate.mnemonic, "flat_load_b32");
  EXPECT_EQ(candidate.text_offset, 24u);
  EXPECT_EQ(candidate.file_offset, 0x118u);
  EXPECT_EQ(candidate.size, 3u * sizeof(uint32_t));
  EXPECT_EQ(candidate.width_bits, 32u);
  ASSERT_TRUE(candidate.dst_vgpr);
  EXPECT_EQ(*candidate.dst_vgpr, 2u);
  ASSERT_TRUE(candidate.addr_vgpr);
  EXPECT_EQ(*candidate.addr_vgpr, 0u);
  ASSERT_TRUE(candidate.raw_vaddr);
  EXPECT_EQ(*candidate.raw_vaddr, 0u);
  ASSERT_TRUE(candidate.raw_vdst);
  EXPECT_EQ(*candidate.raw_vdst, 2u);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_TRUE(result.resource_plans.front().owner_descriptor_file_offsets.empty());
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::Unsupported);
  EXPECT_EQ(result.resource_plans.front().reason, ConSanRegisterPlanReason::MissingOwner);
  EXPECT_EQ(result.resource_plan_summary.unsupported_plans, 1u);
}

TEST(ConSanMoi, SharedHelperPlanUsesCommonDeadWindowAcrossTwoOwners) {
  const std::vector<uint8_t> bytes = make_two_kernel_shared_helper_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_EQ(result.kernels.size(), 3u);
  ASSERT_EQ(result.functions.size(), 1u);
  ASSERT_EQ(result.moi_candidates.size(), 1u);
  EXPECT_FALSE(result.moi_candidates.front().in_kernel);
  EXPECT_EQ(result.moi_candidates.front().container_name, "shared_lds_helper");
  ASSERT_EQ(result.resource_plans.size(), 1u);
  const ConSanCandidateResourcePlan &plan = result.resource_plans.front();
  ASSERT_EQ(plan.owner_descriptor_file_offsets.size(), 2u);
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::LivenessDead);
  EXPECT_EQ(plan.reason, ConSanRegisterPlanReason::None);
  EXPECT_EQ(plan.scratch_vgpr, 1);
  EXPECT_EQ(plan.scratch_vgpr_count, 3u);
}

TEST(ConSanMoi, Gfx950SharedHelperSpillUsesEveryOwnerResourceAndPrivateLayout) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.arch = ROCJITSU_CODE_ARCH_CDNA4;
  fixture.first_vgpr_granulated = 0;
  fixture.second_vgpr_granulated = 1;
  fixture.first_private_bytes = 0;
  fixture.second_private_bytes = 20;
  fixture.helper_keeps_v1_v3_live = true;
  const std::vector<uint8_t> bytes = make_two_kernel_shared_helper_code_object(fixture);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.functions.size(), 1u);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  const ConSanCandidateResourcePlan &plan = result.resource_plans.front();
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::SpillRequired);
  EXPECT_EQ(plan.scratch_vgpr, 1u);
  EXPECT_EQ(plan.original_private_segment_size, 20u);
  ASSERT_EQ(plan.owner_descriptor_file_offsets.size(), 2u);
  const auto access_patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
  });
  ASSERT_NE(access_patch, result.patches.end());
  EXPECT_EQ(std::count_if(result.patches.begin(), result.patches.end(),
                          [](const ConSanPatchInfo &patch) {
                            return patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
                          }),
            2);
  const ConSanPatchInfo &patch = *access_patch;
  EXPECT_EQ(patch.spilled_vgpr_count, 3u);
  EXPECT_EQ(patch.required_private_segment_size, 44u);
  EXPECT_EQ(patch.owner_descriptor_file_offsets, plan.owner_descriptor_file_offsets);

  const std::vector<uint32_t> expected_save = expected_vgpr_spill_words(
      /*base=*/1, /*count=*/3, /*restore=*/false, /*slot_base=*/32, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_FALSE(expected_save.empty());
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  std::vector<uint32_t> trampoline(patch.trampoline_size / sizeof(uint32_t));
  std::memcpy(trampoline.data(), patched.text_sections().front()->data() + patch.trampoline_offset,
              patch.trampoline_size);
  EXPECT_TRUE(std::equal(expected_save.begin(), expected_save.end(), trampoline.begin()));

  for (const AmdGpuKernelInfo &kernel : patched.kernels()) {
    KD descriptor{};
    std::memcpy(&descriptor, result.elf_bytes.data() + kernel.descriptor_file_offset,
                sizeof(descriptor));
    if (kernel.name == "shared_owner_0" || kernel.name == "shared_owner_1")
      EXPECT_EQ(descriptor.private_segment_fixed_size, 44u);
    else if (kernel.name == "unrelated_kernel")
      EXPECT_EQ(descriptor.private_segment_fixed_size, 0u);
  }
}

TEST(ConSanMoi, Gfx950SharedHelperSampledSpillUsesEveryOwnerPrivateLayout) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.arch = ROCJITSU_CODE_ARCH_CDNA4;
  fixture.first_vgpr_granulated = 0;
  fixture.second_vgpr_granulated = 1;
  fixture.first_private_bytes = 0;
  fixture.second_private_bytes = 20;
  fixture.helper_keeps_v1_v3_live = true;
  const std::vector<uint8_t> bytes = make_two_kernel_shared_helper_code_object(fixture);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::Sampled;
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = sizeof(ConSanMoiReportHeader) + 2u * sizeof(uint64_t);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.moi_persistent_vgprs_automatic) << testing::PrintToString(result.warnings);
  for (const auto workgroup_vgpr : result.resolved_moi_workgroup_vgprs)
    ASSERT_TRUE(workgroup_vgpr);
  const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiSampledWatchpointStore;
  });
  ASSERT_NE(patch, result.patches.end()) << testing::PrintToString(result.warnings);
  EXPECT_EQ(patch->owner_descriptor_file_offsets.size(), 2u);
  EXPECT_EQ(patch->spilled_vgpr_count, 5u);
  EXPECT_EQ(patch->required_private_segment_size, 52u);
  EXPECT_EQ(result.resource_plan_summary.emitted_spill_patches, 1u);
  EXPECT_EQ(std::ranges::count_if(result.patches,
                                  [](const ConSanPatchInfo &candidate) {
                                    return candidate.kind ==
                                           ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
                                  }),
            2u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  for (const AmdGpuKernelInfo &kernel : patched.kernels()) {
    KD descriptor{};
    std::memcpy(&descriptor, result.elf_bytes.data() + kernel.descriptor_file_offset,
                sizeof(descriptor));
    if (kernel.name == "shared_owner_0" || kernel.name == "shared_owner_1")
      EXPECT_EQ(descriptor.private_segment_fixed_size, 52u);
    else if (kernel.name == "unrelated_kernel")
      EXPECT_EQ(descriptor.private_segment_fixed_size, 0u);
  }
}

TEST(ConSanMoi, Gfx950SharedHelperInlineSpillUsesEveryOwnerPrivateLayout) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.arch = ROCJITSU_CODE_ARCH_CDNA4;
  fixture.first_vgpr_granulated = 2;
  fixture.second_vgpr_granulated = 2;
  fixture.first_private_bytes = 0;
  fixture.second_private_bytes = 20;
  fixture.helper_keeps_v1_v3_live = true;
  const std::vector<uint8_t> bytes = make_two_kernel_shared_helper_code_object(fixture);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::InlineShadow;
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  ASSERT_NE(patch, result.patches.end());
  EXPECT_EQ(patch->owner_descriptor_file_offsets.size(), 2u);
  EXPECT_EQ(patch->spilled_vgpr_count, 7u);
  EXPECT_EQ(patch->required_private_segment_size, 60u);
  EXPECT_EQ(result.resource_plan_summary.emitted_spill_patches, 1u);
  EXPECT_EQ(std::ranges::count_if(result.patches,
                                  [](const ConSanPatchInfo &patch) {
                                    return patch.kind ==
                                           ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
                                  }),
            2u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  for (const AmdGpuKernelInfo &kernel : patched.kernels()) {
    KD descriptor{};
    std::memcpy(&descriptor, result.elf_bytes.data() + kernel.descriptor_file_offset,
                sizeof(descriptor));
    if (kernel.name == "shared_owner_0" || kernel.name == "shared_owner_1")
      EXPECT_EQ(descriptor.private_segment_fixed_size, 60u);
    else if (kernel.name == "unrelated_kernel")
      EXPECT_EQ(descriptor.private_segment_fixed_size, 0u);
  }
}

TEST(ConSanMoi, Gfx950SharedHelperSpillRollsBackWhenOneOwnerUsesDynamicStack) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.arch = ROCJITSU_CODE_ARCH_CDNA4;
  fixture.first_vgpr_granulated = 0;
  fixture.second_vgpr_granulated = 1;
  fixture.first_private_bytes = 0;
  fixture.second_private_bytes = 20;
  fixture.second_uses_dynamic_stack = true;
  fixture.helper_keeps_v1_v3_live = true;
  const std::vector<uint8_t> bytes = make_two_kernel_shared_helper_code_object(fixture);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  EXPECT_TRUE(result.errors.empty());
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::SpillRequired);
  EXPECT_EQ(result.resource_plans.front().owner_descriptor_file_offsets.size(), 2u);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("dynamic-stack") != std::string::npos;
  }));
}

TEST(ConSanMoi, SharedHelperPatchNamesEveryOwnerAndLeavesUnrelatedDescriptorUnchanged) {
  const std::vector<uint8_t> bytes = make_two_kernel_shared_helper_code_object();
  AmdGpuCodeObject original(bytes.data(), bytes.size());
  ASSERT_TRUE(original.is_valid());
  const auto original_unrelated =
      std::ranges::find_if(original.kernels(), [](const AmdGpuKernelInfo &kernel) {
        return kernel.name == "unrelated_kernel";
      });
  ASSERT_NE(original_unrelated, original.kernels().end());
  KD original_unrelated_descriptor{};
  std::memcpy(&original_unrelated_descriptor,
              bytes.data() + original_unrelated->descriptor_file_offset,
              sizeof(original_unrelated_descriptor));

  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);
  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.modified);
  const auto access_patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
  });
  ASSERT_NE(access_patch, result.patches.end());
  const ConSanPatchInfo &patch = *access_patch;
  EXPECT_EQ(patch.kind, ConSanPatchKind::TrampolineMoiAccessRecordStore);
  EXPECT_EQ(patch.anchor_offset, 20u);
  ASSERT_EQ(patch.owner_descriptor_file_offsets.size(), 2u);
  EXPECT_EQ(patch.owner_descriptor_file_offsets,
            result.resource_plans.front().owner_descriptor_file_offsets);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const auto patched_unrelated =
      std::ranges::find_if(patched.kernels(), [](const AmdGpuKernelInfo &kernel) {
        return kernel.name == "unrelated_kernel";
      });
  ASSERT_NE(patched_unrelated, patched.kernels().end());
  KD patched_unrelated_descriptor{};
  std::memcpy(&patched_unrelated_descriptor,
              result.elf_bytes.data() + patched_unrelated->descriptor_file_offset,
              sizeof(patched_unrelated_descriptor));
  // Text growth legitimately adjusts KD-relative entry offsets. Resource and
  // ABI fields for a kernel that cannot reach the helper stay unchanged.
  EXPECT_EQ(patched_unrelated_descriptor.compute_pgm_rsrc1,
            original_unrelated_descriptor.compute_pgm_rsrc1);
  EXPECT_EQ(patched_unrelated_descriptor.compute_pgm_rsrc2,
            original_unrelated_descriptor.compute_pgm_rsrc2);
  EXPECT_EQ(patched_unrelated_descriptor.private_segment_fixed_size,
            original_unrelated_descriptor.private_segment_fixed_size);
  EXPECT_EQ(patched_unrelated_descriptor.kernel_code_properties,
            original_unrelated_descriptor.kernel_code_properties);
}

TEST(ConSanMoi, SharedHelperPlanGrowsEveryOwnerForOneFreshWindow) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.first_vgpr_granulated = 0;
  fixture.second_vgpr_granulated = 0;
  fixture.helper_keeps_v1_v3_live = true;
  const std::vector<uint8_t> bytes = make_two_kernel_shared_helper_code_object(fixture);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  const ConSanCandidateResourcePlan &plan = result.resource_plans.front();
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::DescriptorGrowth);
  EXPECT_EQ(plan.scratch_vgpr, 4);
  EXPECT_EQ(plan.required_vgpr_count, 7u);
  ASSERT_EQ(plan.owner_descriptor_file_offsets.size(), 2u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  for (const AmdGpuKernelInfo &kernel : patched.kernels()) {
    KD descriptor{};
    std::memcpy(&descriptor, result.elf_bytes.data() + kernel.descriptor_file_offset,
                sizeof(descriptor));
    const uint32_t granulated = AMDHSA_BITS_GET(
        descriptor.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
    if (kernel.name == "shared_owner_0" || kernel.name == "shared_owner_1")
      EXPECT_EQ(granulated, 1u);
    else if (kernel.name == "unrelated_kernel")
      EXPECT_EQ(granulated, kRdna4Wave64AllVgprsGranulated);
  }
}

TEST(ConSanMoi, SharedHelperSpillUsesOneLayoutAndGrowsEveryOwner) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.first_private_bytes = 0;
  fixture.second_private_bytes = 20;
  const std::vector<uint8_t> bytes = make_two_kernel_shared_helper_code_object(fixture);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::SpillRequired);
  EXPECT_EQ(result.resource_plans.front().original_private_segment_size, 20u);
  ASSERT_EQ(result.patches.size(), 1u);
  const ConSanPatchInfo &patch = result.patches.front();
  EXPECT_EQ(patch.spilled_vgpr_count, 3u);
  EXPECT_EQ(patch.required_private_segment_size, 44u);
  ASSERT_EQ(patch.owner_descriptor_file_offsets.size(), 2u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const std::vector<uint32_t> expected_save =
      expected_vgpr_spill_words(/*base=*/1, /*count=*/3, /*restore=*/false, /*slot_base=*/32);
  ASSERT_FALSE(expected_save.empty());
  std::vector<uint32_t> trampoline_words(patch.trampoline_size / sizeof(uint32_t));
  std::memcpy(trampoline_words.data(),
              patched.text_sections().front()->data() + patch.trampoline_offset,
              patch.trampoline_size);
  ASSERT_GE(trampoline_words.size(), expected_save.size());
  EXPECT_TRUE(std::equal(expected_save.begin(), expected_save.end(), trampoline_words.begin()));

  for (const AmdGpuKernelInfo &kernel : patched.kernels()) {
    KD descriptor{};
    std::memcpy(&descriptor, result.elf_bytes.data() + kernel.descriptor_file_offset,
                sizeof(descriptor));
    if (kernel.name == "shared_owner_0" || kernel.name == "shared_owner_1")
      EXPECT_EQ(descriptor.private_segment_fixed_size, 44u);
    else if (kernel.name == "unrelated_kernel")
      EXPECT_EQ(descriptor.private_segment_fixed_size, 0u);
  }
}

TEST(ConSanMoi, SharedHelperRejectsAssignmentLiveInAnyOwnerScope) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.first_continuation_uses_v1 = true;
  const std::vector<uint8_t> bytes = make_two_kernel_shared_helper_code_object(fixture);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.scratch_vgpr = 1;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  EXPECT_TRUE(result.errors.empty());
  EXPECT_FALSE(result.modified);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::Unsupported);
  EXPECT_EQ(result.resource_plans.front().reason, ConSanRegisterPlanReason::ExplicitLive);
  EXPECT_EQ(result.resource_plans.front().owner_descriptor_file_offsets.size(), 2u);
}

TEST(ConSanMoi, SharedInlineShadowUsesOnePersistentPairForEveryOwner) {
  const std::vector<uint8_t> bytes = make_two_kernel_shared_helper_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::InlineShadow;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.modified);
  EXPECT_TRUE(result.moi_persistent_vgprs_automatic);
  ASSERT_TRUE(result.resolved_moi_owner_vgpr);
  ASSERT_TRUE(result.resolved_moi_epoch_vgpr);
  EXPECT_EQ(std::count_if(result.patches.begin(), result.patches.end(),
                          [](const ConSanPatchInfo &patch) {
                            return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
                          }),
            1);
  EXPECT_EQ(std::count_if(result.patches.begin(), result.patches.end(),
                          [](const ConSanPatchInfo &patch) {
                            return patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
                          }),
            2);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  ASSERT_NE(access, result.patches.end());
  EXPECT_EQ(access->owner_descriptor_file_offsets.size(), 2u);
  std::vector<uint64_t> prologue_anchors;
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue)
      prologue_anchors.push_back(patch.anchor_offset);
  }
  std::ranges::sort(prologue_anchors);
  EXPECT_EQ(prologue_anchors, (std::vector<uint64_t>{0u, 8u}));
}

TEST(ConSanMoi, SharedInlineShadowUsesOnePrivateEpochLayoutForEveryOwner) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.first_private_bytes = 0;
  fixture.second_private_bytes = 20;
  const std::vector<uint8_t> bytes = make_two_kernel_shared_helper_code_object(fixture);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::InlineShadow;
  options.force_private_epoch = true;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.modified) << "candidates=" << result.moi_candidates.size()
                               << " plans=" << result.resource_plans.size()
                               << " patches=" << result.patches.size()
                               << " warnings=" << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.moi_private_epoch_automatic);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  ASSERT_NE(access, result.patches.end());
  EXPECT_EQ(access->persistent_epoch_private_offset, 32u);
  EXPECT_EQ(access->owner_descriptor_file_offsets.size(), 2u);
  EXPECT_EQ(std::count_if(result.patches.begin(), result.patches.end(),
                          [](const ConSanPatchInfo &patch) {
                            return patch.kind ==
                                   ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue;
                          }),
            2);
  std::vector<uint64_t> prologue_anchors;
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind == ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue) {
      prologue_anchors.push_back(patch.anchor_offset);
      EXPECT_EQ(patch.persistent_epoch_private_offset, 32u);
      EXPECT_EQ(patch.required_private_segment_size, 52u);
    }
  }
  std::ranges::sort(prologue_anchors);
  EXPECT_EQ(prologue_anchors, (std::vector<uint64_t>{0u, 8u}));

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  for (const AmdGpuKernelInfo &kernel : patched.kernels()) {
    KD descriptor{};
    std::memcpy(&descriptor, result.elf_bytes.data() + kernel.descriptor_file_offset,
                sizeof(descriptor));
    if (kernel.name == "shared_owner_0" || kernel.name == "shared_owner_1")
      EXPECT_EQ(descriptor.private_segment_fixed_size, 52u);
    else if (kernel.name == "unrelated_kernel")
      EXPECT_EQ(descriptor.private_segment_fixed_size, 0u);
  }
}

TEST(ConSanMoi, SharedPrivateOwnerInitializesEachDescriptorWaveSize) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.first_wave32 = true;
  fixture.second_wave32 = false;
  const std::vector<uint8_t> bytes = make_two_kernel_shared_helper_code_object(fixture);
  AmdGpuCodeObject original(bytes.data(), bytes.size());
  ASSERT_TRUE(original.is_valid());
  for (const AmdGpuKernelInfo &kernel : original.kernels()) {
    KD descriptor{};
    std::memcpy(&descriptor, bytes.data() + kernel.descriptor_file_offset, sizeof(descriptor));
    if (kernel.name == "shared_owner_0") {
      EXPECT_EQ(AMDHSA_BITS_GET(descriptor.kernel_code_properties,
                                kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32),
                1u);
    } else if (kernel.name == "shared_owner_1") {
      EXPECT_EQ(AMDHSA_BITS_GET(descriptor.kernel_code_properties,
                                kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32),
                0u);
    }
  }
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::InlineShadow;
  options.force_private_epoch = true;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const auto result = try_patch_consan(bytes, options);

  EXPECT_TRUE(result.errors.empty());
  EXPECT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().owner_descriptor_file_offsets.size(), 2u);
  EXPECT_EQ(std::ranges::count_if(result.patches,
                                  [](const ConSanPatchInfo &patch) {
                                    return patch.kind ==
                                           ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue;
                                  }),
            2u);
}

TEST(ConSanMoi, Gfx950SharedOwnerUsesScalarStateWhenAnyAccumWindowOverlaps) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.arch = ROCJITSU_CODE_ARCH_CDNA4;
  fixture.first_vgpr_granulated = 3;
  fixture.second_vgpr_granulated = 3;
  std::vector<uint8_t> bytes = make_two_kernel_shared_helper_code_object(fixture);
  AmdGpuCodeObject code_object(bytes.data(), bytes.size());
  ASSERT_TRUE(code_object.is_valid());
  for (const AmdGpuKernelInfo &kernel : code_object.kernels()) {
    if (kernel.name != "shared_owner_0" && kernel.name != "shared_owner_1")
      continue;
    KD descriptor{};
    std::memcpy(&descriptor, bytes.data() + kernel.descriptor_file_offset, sizeof(descriptor));
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET,
                    kernel.name == "shared_owner_0" ? 5u : 4u);
    std::memcpy(bytes.data() + kernel.descriptor_file_offset, &descriptor, sizeof(descriptor));
  }
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::InlineShadow;
  options.scratch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.moi_scalar_identity_automatic) << testing::PrintToString(result.warnings);
  EXPECT_FALSE(result.moi_private_epoch_automatic);
  EXPECT_FALSE(result.moi_persistent_vgprs_automatic);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  ASSERT_NE(access, result.patches.end());
  EXPECT_EQ(access->owner_descriptor_file_offsets.size(), 2u);
  EXPECT_FALSE(access->persistent_epoch_private_offset);
  EXPECT_FALSE(access->persistent_owner_private_offset);
  EXPECT_EQ(std::ranges::count_if(result.patches,
                                  [](const ConSanPatchInfo &patch) {
                                    return patch.kind ==
                                           ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
                                  }),
            2u);
}

TEST(ConSanMoi, Gfx950SharedRecordReplayUsesScalarWorkgroupState) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.arch = ROCJITSU_CODE_ARCH_CDNA4;
  fixture.first_vgpr_granulated = 3;
  fixture.second_vgpr_granulated = 3;
  std::vector<uint8_t> bytes = make_two_kernel_shared_helper_code_object(fixture);
  AmdGpuCodeObject code_object(bytes.data(), bytes.size());
  ASSERT_TRUE(code_object.is_valid());
  for (const AmdGpuKernelInfo &kernel : code_object.kernels()) {
    KD descriptor{};
    std::memcpy(&descriptor, bytes.data() + kernel.descriptor_file_offset, sizeof(descriptor));
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 3u);
    std::memcpy(bytes.data() + kernel.descriptor_file_offset, &descriptor, sizeof(descriptor));
  }
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::RecordReplay;
  options.scratch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.moi_scalar_identity_automatic);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
  });
  ASSERT_NE(access, result.patches.end());
  EXPECT_EQ(access->owner_descriptor_file_offsets.size(), 2u);
  ASSERT_TRUE(access->scratch_vgpr);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  std::vector<uint32_t> words(access->trampoline_size / sizeof(uint32_t));
  std::memcpy(words.data(), patched.text_sections().front()->data() + access->trampoline_offset,
              access->trampoline_size);
  const uint16_t value_vgpr = static_cast<uint16_t>(*access->scratch_vgpr + 2u);
  ASSERT_TRUE(result.resolved_moi_state_owner_sgpr);
  ASSERT_TRUE(result.resolved_moi_state_epoch_sgpr);
  EXPECT_NE(std::ranges::find(words,
                              build_v_mov_b32_e32(value_vgpr, *result.resolved_moi_state_owner_sgpr,
                                                  ROCJITSU_CODE_ARCH_CDNA4)),
            words.end());
  EXPECT_NE(std::ranges::find(words,
                              build_v_mov_b32_e32(value_vgpr, *result.resolved_moi_state_epoch_sgpr,
                                                  ROCJITSU_CODE_ARCH_CDNA4)),
            words.end());
  for (std::optional<uint16_t> workgroup_sgpr : result.resolved_moi_workgroup_sgprs) {
    ASSERT_TRUE(workgroup_sgpr);
    EXPECT_NE(std::ranges::find(words, build_v_mov_b32_e32(value_vgpr, *workgroup_sgpr,
                                                           ROCJITSU_CODE_ARCH_CDNA4)),
              words.end());
  }
}

TEST(ConSanMoi, InventorySkipsUnknownFlatSites) {
  const std::vector<uint8_t> bytes = make_rdna4_flat_memory_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels.front().flat_sites.size(), 2u);
  EXPECT_TRUE(result.moi_candidates.empty());

  bool saw_skip_warning = false;
  for (const std::string &warning : result.warnings) {
    saw_skip_warning |= warning.find("skipped flat sites") != std::string::npos &&
                        warning.find("unknown=2") != std::string::npos;
  }
  EXPECT_TRUE(saw_skip_warning);
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
static_assert(sizeof(ConSanMoiAtomicRecord) == 56);
static_assert(sizeof(ConSanMoiInlineAtomicReleaseSlot) == 24);
static_assert(sizeof(ConSanMoiDiagnosticRecord) == 80);

TEST(ConSanMoi, SampledWorkgroupPartitionLayoutMapsOneTwoAndThreeDimensions) {
  constexpr auto one_d = consan_moi_workgroup_partition_layout(40, 4, 1, 1);
  static_assert(one_d);
  EXPECT_EQ(one_d->partition_count, 4u);
  EXPECT_EQ(one_d->slots_per_partition, 10u);
  EXPECT_EQ(consan_moi_workgroup_partition_index(3, 0, 0, 4, 1, 1), 3u);

  constexpr auto two_d = consan_moi_workgroup_partition_layout(60, 3, 2, 1);
  static_assert(two_d);
  EXPECT_EQ(two_d->partition_count, 6u);
  EXPECT_EQ(two_d->slots_per_partition, 10u);
  EXPECT_EQ(consan_moi_workgroup_partition_index(2, 1, 0, 3, 2, 1), 5u);

  constexpr auto three_d = consan_moi_workgroup_partition_layout(80, 2, 2, 2);
  static_assert(three_d);
  EXPECT_EQ(three_d->partition_count, 8u);
  EXPECT_EQ(three_d->slots_per_partition, 10u);
  EXPECT_EQ(consan_moi_workgroup_partition_index(1, 1, 1, 2, 2, 2), 7u);
}

TEST(ConSanMoi, SampledWorkgroupPartitionLayoutRejectsInvalidAndBoundsCoordinates) {
  EXPECT_FALSE(consan_moi_workgroup_partition_layout(8, 0, 1, 1));
  EXPECT_FALSE(consan_moi_workgroup_partition_layout(8, 0x01000000u, 1, 1));
  EXPECT_FALSE(consan_moi_workgroup_partition_layout(8, std::numeric_limits<uint32_t>::max(),
                                                     std::numeric_limits<uint32_t>::max(),
                                                     std::numeric_limits<uint32_t>::max()));
  const auto insufficient = consan_moi_workgroup_partition_layout(7, 2, 2, 2);
  ASSERT_TRUE(insufficient);
  EXPECT_EQ(insufficient->slots_per_partition, 0u);
  EXPECT_FALSE(consan_moi_workgroup_partition_index(2, 0, 0, 2, 2, 2));
  EXPECT_FALSE(consan_moi_workgroup_partition_index(0, 2, 0, 2, 2, 2));
  EXPECT_FALSE(consan_moi_workgroup_partition_index(0, 0, 2, 2, 2, 2));
}

TEST(ConSanMoi, InlineWorkgroupLayoutReservesExactAndAtomicStatePerPartition) {
  constexpr uint32_t kPartitions = 8;
  constexpr uint64_t kBytes =
      sizeof(ConSanMoiReportHeader) +
      kConSanMoiInlineShadowDefaultDiagnosticCapacity * sizeof(ConSanMoiDiagnosticRecord) +
      kPartitions * sizeof(ConSanMoiInlineAtomicReleaseSlot) +
      static_cast<uint64_t>(kPartitions) * kConSanMoiInlineShadowConservativeExactShadowEntries *
          sizeof(uint64_t);
  constexpr auto layout = consan_moi_inline_shadow_report_buffer_layout_for_bytes(
      kBytes, kConSanMoiInlineShadowDefaultDiagnosticCapacity, kPartitions);
  EXPECT_EQ(layout.inline_atomic_release_slot_capacity, kPartitions);
  EXPECT_EQ(layout.exact_shadow_entry_capacity,
            kPartitions * kConSanMoiInlineShadowConservativeExactShadowEntries);
  EXPECT_EQ(layout.sampled_watchpoints_offset,
            layout.inline_atomic_release_slots_offset +
                kPartitions * sizeof(ConSanMoiInlineAtomicReleaseSlot));
}

TEST(ConSanMoi, Gfx950InlineShadowPartitionsExactStateWithChecked64BitAddress) {
  std::array<uint32_t, 300> text_words{};
  text_words[0] = 0xD81A0000u;
  text_words[1] = 0x00000000u; // ds_write_b32 v0, v0
  for (size_t i = 2; i + 1 < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);
  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "gfx950_inline_partition", /*vgpr_granulated=*/3, /*wave32=*/false,
      /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);

  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::InlineShadow;
  options.max_patches = 1;
  options.moi_report_buffer_address = 0x1234fffffff0ull;
  options.moi_report_buffer_size = 2u * 1024u * 1024u;
  options.moi_workgroup_extent_x = 2;
  options.moi_workgroup_extent_y = 2;
  options.moi_workgroup_extent_z = 2;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::InlineMoiExactShadowStore ||
           patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  ASSERT_NE(access, result.patches.end());
  ASSERT_TRUE(access->scratch_vgpr);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const auto *text = patched.text_sections().front();
  const uint64_t rewritten_offset =
      access->trampoline_size != 0 ? access->trampoline_offset : access->anchor_offset;
  const uint32_t rewritten_size =
      access->trampoline_size != 0 ? access->trampoline_size : access->original_size;
  std::vector<uint32_t> rewritten(rewritten_size / sizeof(uint32_t));
  std::memcpy(rewritten.data(), text->data() + rewritten_offset,
              rewritten.size() * sizeof(uint32_t));
  const uint16_t scratch = *access->scratch_vgpr;
  const uint16_t old_value = static_cast<uint16_t>((scratch + 5u) & ~uint16_t{1});
  const uint16_t tmp = old_value == scratch + 4u ? static_cast<uint16_t>(scratch + 6u)
                                                 : static_cast<uint16_t>(scratch + 4u);
  const auto partition =
      build_v_mad_u32_u24(static_cast<uint16_t>(scratch + 3u), vector_source_vgpr(scratch),
                          static_cast<uint16_t>(scratch + 3u), static_cast<uint16_t>(scratch + 1u),
                          ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(partition);
  EXPECT_TRUE(contains_subsequence(rewritten, *partition));
  EXPECT_NE(std::find(rewritten.begin(), rewritten.end(),
                      *build_v_lshlrev_b32_e32(scratch, scalar_positive_inline_u32(17),
                                               static_cast<uint16_t>(scratch + 3u),
                                               ROCJITSU_CODE_ARCH_CDNA4)),
            rewritten.end());
  EXPECT_NE(std::find(rewritten.begin(), rewritten.end(),
                      *build_v_lshrrev_b32_e32(
                          static_cast<uint16_t>(scratch + 1u), scalar_positive_inline_u32(15),
                          static_cast<uint16_t>(scratch + 3u), ROCJITSU_CODE_ARCH_CDNA4)),
            rewritten.end());
  const auto layout = consan_moi_inline_shadow_report_buffer_layout_for_bytes(
      options.moi_report_buffer_size, kConSanMoiInlineShadowDefaultDiagnosticCapacity,
      /*requested_atomic_release_slot_capacity=*/8);
  const uint64_t exact_base =
      *options.moi_report_buffer_address + layout.exact_shadow_entries_offset;
  const uint32_t base_low = static_cast<uint32_t>(exact_base);
  ASSERT_NE(base_low, 0u);
  const auto no_carry_limit = build_v_mov_b32_e64_literal(
      tmp, std::numeric_limits<uint32_t>::max() - base_low, ROCJITSU_CODE_ARCH_CDNA4);
  const auto carry =
      build_v_cmp_gt_u32_e32_vcc(vector_source_vgpr(scratch), tmp, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(no_carry_limit && carry);
  std::vector<uint32_t> carry_sequence(no_carry_limit->begin(), no_carry_limit->end());
  carry_sequence.push_back(*carry);
  EXPECT_TRUE(contains_subsequence(rewritten, carry_sequence));
  const auto overflow_atomic = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      scratch, tmp, tmp,
      /*return_old_value=*/true, /*scope=*/2, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(overflow_atomic);
  EXPECT_TRUE(contains_subsequence(rewritten, *overflow_atomic));
}

TEST(ConSanMoi, Gfx950InlineShadowDefersLoadThatOverwritesItsLdsAddress) {
  std::array<uint32_t, 300> text_words{};
  text_words[0] = 0xD86C0180u;
  text_words[1] = 0x03000003u; // ds_read_b32 v3, v3 offset:384
  for (size_t i = 2; i + 1 < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "gfx950_inline_overlapping_load", /*vgpr_granulated=*/3, /*wave32=*/false,
      /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);

  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::InlineShadow;
  options.max_patches = 1;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::InlineMoiExactShadowStore ||
           patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  ASSERT_NE(access, result.patches.end());
  ASSERT_TRUE(access->scratch_vgpr);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const auto *text = patched.text_sections().front();
  const uint64_t rewritten_offset =
      access->trampoline_size != 0 ? access->trampoline_offset : access->anchor_offset;
  const uint32_t rewritten_size =
      access->trampoline_size != 0 ? access->trampoline_size : access->original_size;
  std::vector<uint32_t> rewritten(rewritten_size / sizeof(uint32_t));
  std::memcpy(rewritten.data(), text->data() + rewritten_offset,
              rewritten.size() * sizeof(uint32_t));

  const uint16_t scratch = *access->scratch_vgpr;
  const uint16_t old_value = static_cast<uint16_t>((scratch + 5u) & ~uint16_t{1});
  const auto atomic = build_flat_atomic_swap_b64_vaddr_vsrc_vdst(
      scratch, static_cast<uint16_t>(scratch + 2u), old_value,
      /*return_old_value=*/true, /*scope=*/2, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(atomic);
  const std::array<uint32_t, 2> original = {text_words[0], text_words[1]};
  const auto atomic_it =
      std::search(rewritten.begin(), rewritten.end(), atomic->begin(), atomic->end());
  const auto original_it =
      std::search(rewritten.begin(), rewritten.end(), original.begin(), original.end());
  ASSERT_NE(atomic_it, rewritten.end());
  ASSERT_NE(original_it, rewritten.end());
  EXPECT_LT(atomic_it, original_it);
}

TEST(ConSanMoi, Gfx950InlineShadowRejectsInsufficientPerWorkgroupExactCapacity) {
  const std::array<uint32_t, 3> text_words = {0xD81A0000u, 0x00000000u,
                                              build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4)};
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "gfx950_inline_partition_capacity", /*vgpr_granulated=*/3,
      /*wave32=*/false, /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::InlineShadow;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.moi_workgroup_extent_x = 2;

  const ConSanResult result = try_patch_consan(bytes, options);

  EXPECT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("per workgroup partition") != std::string::npos;
  }));
}

TEST(ConSanMoi, ReportAbiHeaderCarriesVersionedLayout) {
  constexpr ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7,
      /*dispatch_id=*/9,
      /*access_record_capacity=*/11,
      /*diagnostic_capacity=*/13,
      /*exact_shadow_entry_capacity=*/17,
      /*sampled_watchpoint_capacity=*/19,
      /*barrier_record_capacity=*/23,
      /*atomic_record_capacity=*/29);

  EXPECT_EQ(header.magic, kConSanMoiReportMagic);
  EXPECT_EQ(header.abi_version, kConSanMoiReportAbiVersion);
  EXPECT_EQ(header.header_size, sizeof(ConSanMoiReportHeader));
  EXPECT_EQ(header.generation, 7u);
  EXPECT_EQ(header.dispatch_id, 9u);
  EXPECT_EQ(header.access_record_capacity, 11u);
  EXPECT_EQ(header.barrier_record_capacity, 23u);
  EXPECT_EQ(header.atomic_record_capacity, 29u);
  EXPECT_EQ(header.diagnostic_capacity, 13u);
  EXPECT_EQ(header.exact_shadow_entry_capacity, 17u);
  EXPECT_EQ(header.sampled_watchpoint_capacity, 19u);
  EXPECT_EQ(header.access_record_count, 0u);
  EXPECT_EQ(header.barrier_record_count, 0u);
  EXPECT_EQ(header.atomic_record_count, 0u);
  EXPECT_EQ(header.diagnostic_count, 0u);
  EXPECT_EQ(header.event_counter, 0u);
  EXPECT_EQ(header.partition_overflow_count, 0u);

  constexpr size_t expected_bytes =
      sizeof(ConSanMoiReportHeader) + 11u * sizeof(ConSanMoiAccessRecord) +
      23u * sizeof(ConSanMoiBarrierRecord) + 29u * sizeof(ConSanMoiAtomicRecord) +
      13u * sizeof(ConSanMoiDiagnosticRecord) + 17u * sizeof(uint64_t) + 19u * sizeof(uint64_t);
  EXPECT_EQ(consan_moi_report_buffer_min_bytes(11, 13, 17, 19, 23, 29), expected_bytes);

  EXPECT_EQ(consan_moi_default_auto_report_buffer_size(ConSanMoiEngine::RecordReplay), 64u * 1024u);
  EXPECT_EQ(consan_moi_default_auto_report_buffer_size(ConSanMoiEngine::Sampled), 64u * 1024u);
  EXPECT_EQ(consan_moi_default_auto_report_buffer_size(ConSanMoiEngine::InlineShadow),
            256u * 1024u);

  constexpr ConSanMoiReportBufferLayout default_record_layout =
      consan_moi_report_buffer_layout_for_bytes(
          consan_moi_default_auto_report_buffer_size(ConSanMoiEngine::RecordReplay), true, true);
  EXPECT_GT(default_record_layout.access_record_capacity, 0u);
  EXPECT_EQ(default_record_layout.access_record_capacity,
            default_record_layout.barrier_record_capacity);
  EXPECT_EQ(default_record_layout.access_record_capacity,
            default_record_layout.atomic_record_capacity);

  constexpr ConSanMoiReportBufferLayout default_sampled_layout =
      consan_moi_direct_sampled_report_buffer_layout_for_bytes(
          consan_moi_default_auto_report_buffer_size(ConSanMoiEngine::Sampled));
  EXPECT_GT(default_sampled_layout.sampled_watchpoint_capacity, 0u);

  constexpr ConSanMoiReportBufferLayout default_inline_layout =
      consan_moi_inline_shadow_report_buffer_layout_for_bytes(
          consan_moi_default_auto_report_buffer_size(ConSanMoiEngine::InlineShadow));
  EXPECT_EQ(default_inline_layout.diagnostic_capacity,
            kConSanMoiInlineShadowDefaultDiagnosticCapacity);
  EXPECT_GE(default_inline_layout.exact_shadow_entry_capacity,
            kConSanMoiInlineShadowConservativeExactShadowEntries);

  constexpr ConSanMoiReportBufferLayout access_only_layout =
      consan_moi_report_buffer_layout_for_bytes(consan_moi_report_buffer_min_bytes(5, 0, 0, 0),
                                                /*include_barriers=*/false);
  EXPECT_EQ(access_only_layout.access_record_capacity, 5u);
  EXPECT_EQ(access_only_layout.barrier_record_capacity, 0u);
  EXPECT_EQ(access_only_layout.atomic_record_capacity, 0u);
  EXPECT_EQ(access_only_layout.diagnostic_capacity, 0u);
  EXPECT_EQ(access_only_layout.exact_shadow_entry_capacity, 0u);
  EXPECT_EQ(access_only_layout.sampled_watchpoint_capacity, 0u);
  EXPECT_EQ(access_only_layout.access_records_offset, sizeof(ConSanMoiReportHeader));
  EXPECT_EQ(access_only_layout.barrier_records_offset,
            sizeof(ConSanMoiReportHeader) + 5u * sizeof(ConSanMoiAccessRecord));
  EXPECT_EQ(access_only_layout.atomic_records_offset, access_only_layout.barrier_records_offset);
  EXPECT_EQ(access_only_layout.diagnostic_records_offset, access_only_layout.atomic_records_offset);
  EXPECT_EQ(access_only_layout.exact_shadow_entries_offset,
            access_only_layout.diagnostic_records_offset);
  EXPECT_EQ(access_only_layout.inline_atomic_release_slots_offset,
            access_only_layout.exact_shadow_entries_offset);
  EXPECT_EQ(access_only_layout.sampled_watchpoints_offset,
            access_only_layout.exact_shadow_entries_offset);

  constexpr ConSanMoiReportBufferLayout barrier_layout = consan_moi_report_buffer_layout_for_bytes(
      consan_moi_report_buffer_min_bytes(3, 0, 0, 0, 3), /*include_barriers=*/true);
  EXPECT_EQ(barrier_layout.access_record_capacity, 3u);
  EXPECT_EQ(barrier_layout.barrier_record_capacity, 3u);
  EXPECT_EQ(barrier_layout.atomic_record_capacity, 0u);
  EXPECT_EQ(barrier_layout.diagnostic_capacity, 0u);
  EXPECT_EQ(barrier_layout.exact_shadow_entry_capacity, 0u);
  EXPECT_EQ(barrier_layout.sampled_watchpoint_capacity, 0u);
  EXPECT_EQ(barrier_layout.access_records_offset, sizeof(ConSanMoiReportHeader));
  EXPECT_EQ(barrier_layout.barrier_records_offset,
            sizeof(ConSanMoiReportHeader) + 3u * sizeof(ConSanMoiAccessRecord));
  EXPECT_EQ(barrier_layout.atomic_records_offset,
            barrier_layout.barrier_records_offset + 3u * sizeof(ConSanMoiBarrierRecord));
  EXPECT_EQ(barrier_layout.diagnostic_records_offset, barrier_layout.atomic_records_offset);
  EXPECT_EQ(barrier_layout.exact_shadow_entries_offset, barrier_layout.diagnostic_records_offset);
  EXPECT_EQ(barrier_layout.inline_atomic_release_slots_offset,
            barrier_layout.exact_shadow_entries_offset);
  EXPECT_EQ(barrier_layout.sampled_watchpoints_offset, barrier_layout.exact_shadow_entries_offset);

  constexpr ConSanMoiReportBufferLayout atomic_layout = consan_moi_report_buffer_layout_for_bytes(
      consan_moi_report_buffer_min_bytes(2, 0, 0, 0, 0, 2),
      /*include_barriers=*/false,
      /*include_atomics=*/true);
  EXPECT_EQ(atomic_layout.access_record_capacity, 2u);
  EXPECT_EQ(atomic_layout.barrier_record_capacity, 0u);
  EXPECT_EQ(atomic_layout.atomic_record_capacity, 2u);
  EXPECT_EQ(atomic_layout.diagnostic_capacity, 0u);
  EXPECT_EQ(atomic_layout.exact_shadow_entry_capacity, 0u);
  EXPECT_EQ(atomic_layout.sampled_watchpoint_capacity, 0u);
  EXPECT_EQ(atomic_layout.access_records_offset, sizeof(ConSanMoiReportHeader));
  EXPECT_EQ(atomic_layout.barrier_records_offset,
            sizeof(ConSanMoiReportHeader) + 2u * sizeof(ConSanMoiAccessRecord));
  EXPECT_EQ(atomic_layout.atomic_records_offset, atomic_layout.barrier_records_offset);
  EXPECT_EQ(atomic_layout.diagnostic_records_offset,
            atomic_layout.atomic_records_offset + 2u * sizeof(ConSanMoiAtomicRecord));
  EXPECT_EQ(atomic_layout.exact_shadow_entries_offset, atomic_layout.diagnostic_records_offset);
  EXPECT_EQ(atomic_layout.inline_atomic_release_slots_offset,
            atomic_layout.exact_shadow_entries_offset);
  EXPECT_EQ(atomic_layout.sampled_watchpoints_offset, atomic_layout.exact_shadow_entries_offset);

  constexpr ConSanMoiReportBufferLayout combined_layout = consan_moi_report_buffer_layout_for_bytes(
      consan_moi_report_buffer_min_bytes(4, 0, 0, 0, 4, 4),
      /*include_barriers=*/true,
      /*include_atomics=*/true);
  EXPECT_EQ(combined_layout.access_record_capacity, 4u);
  EXPECT_EQ(combined_layout.barrier_record_capacity, 4u);
  EXPECT_EQ(combined_layout.atomic_record_capacity, 4u);
  EXPECT_EQ(combined_layout.diagnostic_capacity, 0u);
  EXPECT_EQ(combined_layout.exact_shadow_entry_capacity, 0u);
  EXPECT_EQ(combined_layout.sampled_watchpoint_capacity, 0u);
  EXPECT_EQ(combined_layout.access_records_offset, sizeof(ConSanMoiReportHeader));
  EXPECT_EQ(combined_layout.barrier_records_offset,
            sizeof(ConSanMoiReportHeader) + 4u * sizeof(ConSanMoiAccessRecord));
  EXPECT_EQ(combined_layout.atomic_records_offset,
            combined_layout.barrier_records_offset + 4u * sizeof(ConSanMoiBarrierRecord));
  EXPECT_EQ(combined_layout.diagnostic_records_offset,
            combined_layout.atomic_records_offset + 4u * sizeof(ConSanMoiAtomicRecord));
  EXPECT_EQ(combined_layout.exact_shadow_entries_offset, combined_layout.diagnostic_records_offset);
  EXPECT_EQ(combined_layout.inline_atomic_release_slots_offset,
            combined_layout.exact_shadow_entries_offset);
  EXPECT_EQ(combined_layout.sampled_watchpoints_offset,
            combined_layout.exact_shadow_entries_offset);

  constexpr ConSanMoiReportBufferLayout direct_sampled_layout =
      consan_moi_direct_sampled_report_buffer_layout_for_bytes(sizeof(ConSanMoiReportHeader) +
                                                               6u * sizeof(uint64_t));
  EXPECT_EQ(direct_sampled_layout.access_record_capacity, 0u);
  EXPECT_EQ(direct_sampled_layout.barrier_record_capacity, 0u);
  EXPECT_EQ(direct_sampled_layout.atomic_record_capacity, 0u);
  EXPECT_EQ(direct_sampled_layout.diagnostic_capacity, 0u);
  EXPECT_EQ(direct_sampled_layout.exact_shadow_entry_capacity, 0u);
  EXPECT_EQ(direct_sampled_layout.sampled_watchpoint_capacity, 6u);
  EXPECT_EQ(direct_sampled_layout.diagnostic_records_offset, sizeof(ConSanMoiReportHeader));
  EXPECT_EQ(direct_sampled_layout.exact_shadow_entries_offset, sizeof(ConSanMoiReportHeader));
  EXPECT_EQ(direct_sampled_layout.inline_atomic_release_slots_offset,
            sizeof(ConSanMoiReportHeader));
  EXPECT_EQ(direct_sampled_layout.sampled_watchpoints_offset, sizeof(ConSanMoiReportHeader));

  constexpr ConSanMoiReportBufferLayout inline_shadow_layout =
      consan_moi_inline_shadow_report_buffer_layout_for_bytes(
          sizeof(ConSanMoiReportHeader) + 4u * sizeof(ConSanMoiDiagnosticRecord) +
          sizeof(ConSanMoiInlineAtomicReleaseSlot) + 32u * sizeof(uint64_t));
  EXPECT_EQ(inline_shadow_layout.access_record_capacity, 0u);
  EXPECT_EQ(inline_shadow_layout.barrier_record_capacity, 0u);
  EXPECT_EQ(inline_shadow_layout.atomic_record_capacity, 0u);
  EXPECT_EQ(inline_shadow_layout.diagnostic_capacity, 4u);
  EXPECT_EQ(inline_shadow_layout.exact_shadow_entry_capacity, 32u);
  EXPECT_EQ(inline_shadow_layout.sampled_watchpoint_capacity, 0u);
  EXPECT_EQ(inline_shadow_layout.diagnostic_records_offset, sizeof(ConSanMoiReportHeader));
  EXPECT_EQ(inline_shadow_layout.exact_shadow_entries_offset,
            sizeof(ConSanMoiReportHeader) + 4u * sizeof(ConSanMoiDiagnosticRecord));
  EXPECT_EQ(inline_shadow_layout.inline_atomic_release_slots_offset,
            inline_shadow_layout.exact_shadow_entries_offset + 32u * sizeof(uint64_t));
  EXPECT_EQ(inline_shadow_layout.sampled_watchpoints_offset,
            inline_shadow_layout.inline_atomic_release_slots_offset +
                sizeof(ConSanMoiInlineAtomicReleaseSlot));

  constexpr ConSanMoiReportBufferLayout small_inline_shadow_layout =
      consan_moi_inline_shadow_report_buffer_layout_for_bytes(sizeof(ConSanMoiReportHeader) +
                                                              sizeof(ConSanMoiDiagnosticRecord));
  EXPECT_EQ(small_inline_shadow_layout.diagnostic_capacity, 0u);
  EXPECT_EQ(small_inline_shadow_layout.exact_shadow_entry_capacity,
            sizeof(ConSanMoiDiagnosticRecord) / sizeof(uint64_t));
}

TEST(ConSanMoi, WarnsWhenReportBufferIsSmallerThanHeader) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_report_buffer_address = 0x1000;
  options.moi_report_buffer_size = sizeof(ConSanMoiReportHeader) - 1;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  bool saw_small_buffer_warning = false;
  for (const std::string &warning : result.warnings)
    saw_small_buffer_warning |=
        warning.find("smaller than the report ABI header") != std::string::npos;
  EXPECT_TRUE(saw_small_buffer_warning);
}

TEST(ConSanMoi, Gfx950DescriptorRegisterGeometryUsesCdna4Encoding) {
  const ConSanMoiDescriptorRegisterGeometry geometry =
      consan_moi_descriptor_register_geometry(ROCJITSU_CODE_ARCH_CDNA4,
                                              /*descriptor_wave32=*/true);
  EXPECT_EQ(geometry.wavefront_size, 64u);
  EXPECT_EQ(geometry.max_vgpr_count, 256u);
  EXPECT_EQ(geometry.max_sgpr_count, 102u);
  EXPECT_EQ(geometry.vgpr_encoding_granularity, 8u);
  EXPECT_EQ(geometry.sgpr_encoding_granularity, 8u);

  EXPECT_EQ(consan_moi_decode_descriptor_register_count(0, geometry.max_vgpr_count,
                                                        geometry.vgpr_encoding_granularity),
            8u);
  EXPECT_EQ(consan_moi_decode_descriptor_register_count(31, geometry.max_vgpr_count,
                                                        geometry.vgpr_encoding_granularity),
            256u);
  EXPECT_EQ(consan_moi_encode_descriptor_register_count(256, geometry.max_vgpr_count,
                                                        geometry.vgpr_encoding_granularity),
            31u);
  EXPECT_FALSE(consan_moi_encode_descriptor_register_count(257, geometry.max_vgpr_count,
                                                           geometry.vgpr_encoding_granularity)
                   .has_value());

  EXPECT_EQ(consan_moi_decode_descriptor_register_count(12, geometry.max_sgpr_count,
                                                        geometry.sgpr_encoding_granularity),
            102u);
  EXPECT_EQ(consan_moi_encode_descriptor_register_count(102, geometry.max_sgpr_count,
                                                        geometry.sgpr_encoding_granularity),
            12u);
  EXPECT_FALSE(consan_moi_encode_descriptor_register_count(103, geometry.max_sgpr_count,
                                                           geometry.sgpr_encoding_granularity)
                   .has_value());
}

TEST(ConSanMoi, RdnaDescriptorRegisterGeometryRetainsWaveSizeEncoding) {
  const auto wave64 = consan_moi_descriptor_register_geometry(ROCJITSU_CODE_ARCH_RDNA4, false);
  const auto wave32 = consan_moi_descriptor_register_geometry(ROCJITSU_CODE_ARCH_RDNA4, true);
  EXPECT_EQ(wave64.wavefront_size, 64u);
  EXPECT_EQ(wave64.vgpr_encoding_granularity, 4u);
  EXPECT_EQ(wave32.wavefront_size, 32u);
  EXPECT_EQ(wave32.vgpr_encoding_granularity, 8u);
}

TEST(ConSanMoi, Gfx950ResourcePlanDecodesEightVgprsFromZeroDescriptorField) {
  std::array<uint32_t, 170> text_words{};
  text_words[0] = 0xD81A0000u;
  text_words[1] = 0x00000000u; // ds_write_b32 v0, v0
  for (size_t i = 2; i + 1 < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);

  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "gfx950_descriptor_geometry",
                                 /*vgpr_granulated=*/0, /*wave32=*/true,
                                 /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().current_vgpr_count, 8u);
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::LivenessDead);
}

[[nodiscard]] std::vector<uint8_t>
make_gfx950_accum_pressure_code_object(std::optional<uint16_t> dead_window_base,
                                       uint32_t encoded_accum_offset) {
  std::vector<uint32_t> text_words = {
      0xD81A0000u,
      0x00000000u, // ds_write_b32 v0, v0
  };
  for (uint16_t vgpr = 1; vgpr < 132; ++vgpr) {
    if (dead_window_base && vgpr >= *dead_window_base && vgpr < *dead_window_base + 3u)
      continue;
    text_words.push_back(
        build_v_mov_b32_e32(vgpr, vector_source_vgpr(vgpr), ROCJITSU_CODE_ARCH_CDNA4));
  }
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4));
  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "gfx950_accum_pressure", /*vgpr_granulated=*/16, /*wave32=*/false,
      /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  mutate_first_kernel_descriptor(bytes, [&](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET,
                    encoded_accum_offset);
  });
  return bytes;
}

[[nodiscard]] ConSanOptions gfx950_accum_pressure_options() {
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);
  return options;
}

TEST(ConSanMoi, Gfx950ResourcePlanSpillsBelowAccumBoundary) {
  const std::vector<uint8_t> bytes =
      make_gfx950_accum_pressure_code_object(std::nullopt, /*encoded_accum_offset=*/32u);

  const ConSanResult result = try_patch_consan(bytes, gfx950_accum_pressure_options());

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  const ConSanCandidateResourcePlan &plan = result.resource_plans.front();
  EXPECT_EQ(plan.current_vgpr_count, 136u);
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::SpillRequired);
  ASSERT_TRUE(plan.scratch_vgpr);
  EXPECT_LE(static_cast<uint32_t>(*plan.scratch_vgpr) + plan.scratch_vgpr_count, 132u);
  EXPECT_EQ(plan.required_vgpr_count, 136u);
}

TEST(ConSanMoi, Gfx950ResourcePlanPrefersDeadScratchBelowAccumBoundary) {
  const std::vector<uint8_t> bytes = make_gfx950_accum_pressure_code_object(
      /*dead_window_base=*/120u, /*encoded_accum_offset=*/32u);

  const ConSanResult result = try_patch_consan(bytes, gfx950_accum_pressure_options());

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  const ConSanCandidateResourcePlan &plan = result.resource_plans.front();
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::LivenessDead);
  EXPECT_EQ(plan.scratch_vgpr, 120u);
}

TEST(ConSanMoi, Gfx950ResourcePlanAllowsScratchEndingAtAccumBoundary) {
  const std::vector<uint8_t> bytes = make_gfx950_accum_pressure_code_object(
      /*dead_window_base=*/129u, /*encoded_accum_offset=*/32u);

  const ConSanResult result = try_patch_consan(bytes, gfx950_accum_pressure_options());

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::LivenessDead);
  EXPECT_EQ(result.resource_plans.front().scratch_vgpr, 129u);
}

TEST(ConSanMoi, Gfx950ResourcePlanRejectsExplicitScratchAtAccumBoundary) {
  const std::vector<uint8_t> bytes =
      make_gfx950_accum_pressure_code_object(std::nullopt, /*encoded_accum_offset=*/32u);
  ConSanOptions options = gfx950_accum_pressure_options();
  options.scratch_vgpr = 132u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::Unsupported);
  EXPECT_EQ(result.resource_plans.front().reason, ConSanRegisterPlanReason::ExplicitOutOfRange);
  EXPECT_FALSE(result.modified);
}

TEST(ConSanMoi, Gfx950ResourcePlanDoesNotBoundEncodedZeroAccumOffset) {
  const std::vector<uint8_t> bytes =
      make_gfx950_accum_pressure_code_object(std::nullopt, /*encoded_accum_offset=*/0u);

  const ConSanResult result = try_patch_consan(bytes, gfx950_accum_pressure_options());

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::LivenessDead);
  EXPECT_EQ(result.resource_plans.front().scratch_vgpr, 132u);
}

[[nodiscard]] std::vector<uint8_t> make_gfx950_inline_candidate_ranking_code_object() {
  std::vector<uint32_t> text_words = {
      0xD81A0000u,
      0x00000000u, // ds_write_b32 v0, v0: all other VGPRs are live afterward
  };
  for (uint16_t vgpr = 1; vgpr < 256; ++vgpr) {
    text_words.push_back(
        build_v_mov_b32_e32(vgpr, vector_source_vgpr(vgpr), ROCJITSU_CODE_ARCH_CDNA4));
  }
  text_words.push_back(0xD81A0000u);
  text_words.push_back(0x00000000u); // ds_write_b32 v0, v0: v[1:255] are dead here
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4));
  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "gfx950_inline_candidate_ranking", /*vgpr_granulated=*/31,
      /*wave32=*/false, /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 63u);
  });
  return bytes;
}

TEST(ConSanMoi, Gfx950InlineMaxPatchesPrefersLaterDeadCandidateOverEarlierSpill) {
  const std::vector<uint8_t> bytes = make_gfx950_inline_candidate_ranking_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::InlineShadow;
  options.max_patches = 1;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.moi_scalar_identity_automatic);
  ASSERT_EQ(result.resource_plans.size(), 2u);
  EXPECT_EQ(result.resource_plans[0].source, ConSanRegisterAllocationSource::SpillRequired);
  EXPECT_EQ(result.resource_plans[1].source, ConSanRegisterAllocationSource::LivenessDead);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::InlineMoiExactShadowStore ||
           patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  ASSERT_NE(access, result.patches.end());
  EXPECT_EQ(access->anchor_offset, 257u * sizeof(uint32_t));
  EXPECT_EQ(access->spilled_vgpr_count, 0u);
  EXPECT_EQ(access->required_private_segment_size, 0u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  EXPECT_EQ(descriptor.private_segment_fixed_size, 0u);
}

TEST(ConSanMoi, Gfx950InlineForcedSpillKeepsStableCandidateOrder) {
  const std::vector<uint8_t> bytes = make_gfx950_inline_candidate_ranking_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::InlineShadow;
  options.max_patches = 1;
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.resource_plans.size(), 2u);
  EXPECT_EQ(result.resource_plans[0].source, ConSanRegisterAllocationSource::SpillRequired);
  EXPECT_EQ(result.resource_plans[1].source, ConSanRegisterAllocationSource::SpillRequired);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  ASSERT_NE(access, result.patches.end());
  EXPECT_EQ(access->anchor_offset, 0u);
  EXPECT_EQ(access->spilled_vgpr_count, 7u);
  EXPECT_GT(access->required_private_segment_size, 0u);
}

TEST(ConSanMoi, Gfx950PressureFixtureGrowsDescriptorAfterLiveWindow) {
  std::vector<uint32_t> text_words = {
      0xD81A0000u,
      0x00000000u, // ds_write_b32 v0, v0
  };
  for (uint16_t vgpr = 1; vgpr < 8; ++vgpr)
    text_words.push_back(
        build_v_mov_b32_e32(vgpr, vector_source_vgpr(vgpr), ROCJITSU_CODE_ARCH_CDNA4));
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4));
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "gfx950_descriptor_pressure",
                                 /*vgpr_granulated=*/0, /*wave32=*/false,
                                 /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  const ConSanCandidateResourcePlan &plan = result.resource_plans.front();
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::DescriptorGrowth);
  EXPECT_EQ(plan.current_vgpr_count, 8u);
  EXPECT_EQ(plan.max_referenced_vgpr_count, 8u);
  EXPECT_EQ(plan.scratch_vgpr, 8u);
  EXPECT_EQ(plan.required_vgpr_count, 11u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc1,
                            kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT),
            1u);
}

TEST(ConSanMoi, Gfx950InventoriesBasicLdsReadWriteAndExcludesAtomic) {
  const std::array<uint32_t, 7> text_words = {
      0xD81A000Cu,
      0x00000100u, // ds_write_b32 v0, v1 offset:12
      0xD86C0000u,
      0x01000000u, // ds_read_b32 v1, v0
      0xD8000000u,
      0x00000100u, // ds_add_u32 v0, v1
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "gfx950_basic_lds",
                                 /*vgpr_granulated=*/0, /*wave32=*/false,
                                 /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_EQ(result.kernels.size(), 1u);
  EXPECT_EQ(result.kernels.front().stats.lds_write_count, 1u);
  EXPECT_EQ(result.kernels.front().stats.lds_read_count, 1u);
  ASSERT_EQ(result.moi_candidates.size(), 2u);
  EXPECT_EQ(result.moi_candidates[0].mnemonic, "ds_write_b32");
  EXPECT_EQ(result.moi_candidates[0].kind, ConSanLdsAccessKind::Write);
  EXPECT_EQ(result.moi_candidates[0].addr_vgpr, 0u);
  EXPECT_EQ(result.moi_candidates[0].data_vgpr, 1u);
  EXPECT_EQ(result.moi_candidates[0].width_bits, 32u);
  EXPECT_EQ(result.moi_candidates[1].mnemonic, "ds_read_b32");
  EXPECT_EQ(result.moi_candidates[1].kind, ConSanLdsAccessKind::Read);
  EXPECT_EQ(result.moi_candidates[1].addr_vgpr, 0u);
  EXPECT_EQ(result.moi_candidates[1].dst_vgpr, 1u);
  EXPECT_EQ(result.moi_candidates[1].width_bits, 32u);
}

TEST(ConSan, Gfx950SuperColliderRewritesPaddedStoreWithNativeReadback) {
  const std::array<uint32_t, 12> text_words = {
      0xD81A000Cu,
      0x00000100u, // ds_write_b32 v0, v1 offset:12
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "gfx950_supercollider_store",
                                 /*vgpr_granulated=*/0, /*wave32=*/false,
                                 /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsStoreCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  std::array<uint32_t, 11> words{};
  std::memcpy(words.data(), result.elf_bytes.data() + 0x100, sizeof(words));
  EXPECT_EQ(words[0], 0xD81A000Cu);
  EXPECT_EQ(words[1], 0x00000100u);
  EXPECT_EQ(words[2], *build_s_wait_lds0(ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_EQ(words[3], 0xD86C000Cu);
  EXPECT_EQ(words[4], 0x03000000u);
  EXPECT_EQ(words[5], *build_s_wait_lds0(ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_EQ(words[6], *build_s_mov_b64(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_EQ(words[7],
            *build_v_cmp_ne_u32_e32_vcc(vector_source_vgpr(1), 3, ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_EQ(words[8], *build_s_cbranch_vccz(1, ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_EQ(words[9], *build_s_trap(0, ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_EQ(words[10], *build_s_mov_b64(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_CDNA4));
}

TEST(ConSan, Gfx950SuperColliderUsesAppendedCaveWithoutPadding) {
  const std::array<uint32_t, 3> text_words = {
      0xD81A0000u,
      0x00000100u, // ds_write_b32 v0, v1
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "gfx950_supercollider_appended",
                                 /*vgpr_granulated=*/0, /*wave32=*/false,
                                 /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::LocalCaveLdsStoreCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, text_words.size() * sizeof(uint32_t));
  EXPECT_GT(result.elf_bytes.size(), bytes.size());
}

TEST(ConSan, Gfx950SuperColliderAllowsScratchEndingAtAccumBoundary) {
  const std::array<uint32_t, 4> text_words = {
      0xD89A0000u,
      0x00000109u, // ds_write_b64 v9, v[1:2]
      build_v_mov_b32_e32(129, vector_source_vgpr(129), ROCJITSU_CODE_ARCH_CDNA4),
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "gfx950_supercollider_accum_boundary", /*vgpr_granulated=*/16,
      /*wave32=*/false, /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    // (32 + 1) * 4 = v132, so scratch v130:v131 ends at the boundary.
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 32u);
  });
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.patches.size(), 1u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 130u);
}

TEST(ConSan, Gfx950SuperColliderPrefersDeadScratchBelowAccumBoundary) {
  const std::array<uint32_t, 4> text_words = {
      0xD89A0000u,
      0x00000109u, // ds_write_b64 v9, v[1:2]
      build_v_mov_b32_e32(132, vector_source_vgpr(132), ROCJITSU_CODE_ARCH_CDNA4),
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "gfx950_supercollider_accum_dead", /*vgpr_granulated=*/16,
      /*wave32=*/false, /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 32u);
  });
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.patches.size(), 1u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_LT(static_cast<uint32_t>(*result.patches.front().scratch_vgpr) + 2u, 133u);
}

TEST(ConSan, Gfx950SuperColliderRejectsExplicitScratchOverlappingAccumBoundary) {
  const std::array<uint32_t, 3> text_words = {
      0xD89A0000u,
      0x00000109u, // ds_write_b64 v9, v[1:2]
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "gfx950_supercollider_accum_explicit", /*vgpr_granulated=*/16,
      /*wave32=*/false, /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 32u);
  });
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 132;

  const ConSanResult result = try_patch_consan(bytes, options);

  EXPECT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("scratchable_candidates=0") != std::string::npos &&
           warning.find("accum_offset_blocked_candidates=1") != std::string::npos;
  })) << testing::PrintToString(result.warnings);
}

TEST(ConSan, Gfx950SuperColliderDoesNotBoundEncodedZeroAccumOffset) {
  const std::array<uint32_t, 3> text_words = {
      0xD89A0000u,
      0x00000109u, // ds_write_b64 v9, v[1:2]
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "gfx950_supercollider_zero_accum", /*vgpr_granulated=*/16,
      /*wave32=*/false, /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 132;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().scratch_vgpr, 132u);
}

TEST(ConSanMoi, Gfx950BasicFixtureInventoriesNativeInstructionFamilies) {
  // LLVM gfx950 encodings retained as a deterministic CPU-only A3A fixture.
  const std::array<uint32_t, 12> text_words = {
      0xD81A000Cu,
      0x00000100u, // ds_write_b32 v0, v1 offset:12
      0xD86C0000u,
      0x02000000u, // ds_read_b32 v2, v0
      0xBF8A0000u, // s_barrier
      0xDC500000u,
      0x03000004u, // flat_load_dword v3, v[4:5]
      0xDC700000u,
      0x00000304u, // flat_store_dword v[4:5], v3
      0xD8000000u,
      0x00000100u, // ds_add_u32 v0, v1
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "gfx950_basic_families",
                                 /*vgpr_granulated=*/0, /*wave32=*/false,
                                 /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_EQ(result.kernels.size(), 1u);
  const ConSanKernelInfo &kernel = result.kernels.front();
  EXPECT_EQ(kernel.stats.lds_write_count, 1u);
  EXPECT_EQ(kernel.stats.lds_read_count, 1u);
  EXPECT_EQ(kernel.stats.lds_atomic_count, 1u);
  EXPECT_EQ(kernel.stats.flat_read_count, 1u);
  EXPECT_EQ(kernel.stats.flat_write_count, 1u);
  EXPECT_EQ(kernel.barrier_sites.size(), 1u);
  EXPECT_EQ(kernel.atomic_sites.size(), 1u);
}

TEST(ConSanMoi, Gfx950BarrierFixtureSeparatesWaitAndBarrierInventory) {
  const auto barrier_sequence = build_cdna4_s_barrier_with_memory_wait(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(barrier_sequence);
  const std::array<uint32_t, 3> text_words = {
      (*barrier_sequence)[0],
      (*barrier_sequence)[1],
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "gfx950_barrier_contract", /*vgpr_granulated=*/0, /*wave32=*/false,
      /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_EQ(result.kernels.size(), 1u);
  const ConSanKernelInfo &kernel = result.kernels.front();
  EXPECT_EQ(kernel.stats.wait_count, 1u);
  EXPECT_EQ(kernel.stats.barrier_count, 1u);
  ASSERT_EQ(kernel.barrier_sites.size(), 1u);
  EXPECT_EQ(kernel.barrier_sites.front().mnemonic, "s_barrier");
  EXPECT_EQ(kernel.barrier_sites.front().text_offset, sizeof(uint32_t));
}

TEST(ConSanMoi, Gfx950BarrierRecordForcedSpillUsesPlannedPrivateWindow) {
  const auto barrier_sequence = build_cdna4_s_barrier_with_memory_wait(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(barrier_sequence);
  const std::array<uint32_t, 3> text_words = {
      (*barrier_sequence)[0],
      (*barrier_sequence)[1],
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "gfx950_barrier_forced_spill", kRdna4Wave64AllVgprsGranulated,
      /*wave32=*/false, /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_track_barriers = true;
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 1);
  options.max_patches = 1;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &item) {
    return item.kind == ConSanPatchKind::TrampolineMoiBarrierRecord;
  });
  ASSERT_NE(patch, result.patches.end());
  EXPECT_EQ(patch->spilled_vgpr_count, 6u);
  EXPECT_EQ(patch->required_private_segment_size, 24u);
  EXPECT_EQ(result.resource_plan_summary.emitted_spill_patches, 1u);
  EXPECT_EQ(result.resource_plan_summary.emitted_spill_slot_bytes, 24u);
}

TEST(ConSanMoi, Gfx950InlineBarrierForcedSpillKeepsPrivateEpochWindowDisjoint) {
  const auto barrier_sequence = build_cdna4_s_barrier_with_memory_wait(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(barrier_sequence);
  const std::array<uint32_t, 6> text_words = {
      build_v_mov_b32_e32(/*vdst=*/10, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_CDNA4),
      0xD81A0000u,
      0x00000000u, // ds_store_b32
      (*barrier_sequence)[0],
      (*barrier_sequence)[1],
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "gfx950_inline_barrier_forced_spill", /*vgpr_granulated=*/1,
      /*wave32=*/false, /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::InlineShadow;
  options.moi_track_barriers = true;
  options.force_private_epoch = true;
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 2;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  const auto barrier = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiInlineEpochBarrier;
  });
  ASSERT_NE(access, result.patches.end());
  ASSERT_NE(barrier, result.patches.end());
  EXPECT_EQ(access->spilled_vgpr_count, 7u);
  EXPECT_EQ(barrier->spilled_vgpr_count, 1u);
  EXPECT_EQ(access->persistent_epoch_private_offset, 0u);
  EXPECT_EQ(barrier->persistent_epoch_private_offset, 0u);
  EXPECT_EQ(access->required_private_segment_size, 44u);
  EXPECT_EQ(barrier->required_private_segment_size, 44u);
}

TEST(ConSanMoi, Gfx950ExtendedLdsFormsNormalizeRegistersAndScaledRanges) {
  struct Expected {
    const char *mnemonic;
    ConSanLdsAccessKind kind;
    uint32_t width_bits;
    uint16_t addr;
    std::optional<uint16_t> dst;
    std::optional<uint16_t> data;
    std::optional<uint16_t> secondary_data;
    std::vector<ConSanLdsStaticRange> ranges;
  };
  const std::array<uint32_t, 37> text_words = {
      0xD83C0003u,
      0x00000A04u, // ds_write_b8 v4, v10 offset:3
      0xD83E0005u,
      0x00000B05u, // ds_write_b16 v5, v11 offset:5
      0xD8AC0007u,
      0x14000006u, // ds_read_u8_d16 v20, v6 offset:7
      0xD8B60009u,
      0x15000007u, // ds_read_u16_d16_hi v21, v7 offset:9
      0xD89A0010u,
      0x00000C08u, // ds_write_b64 v8, v[12:13] offset:16
      0xD8EC0018u,
      0x16000009u, // ds_read_b64 v[22:23], v9 offset:24
      0xD9BC0020u,
      0x00000E0Au, // ds_write_b96 v10, v[14:16] offset:32
      0xD9FC0028u,
      0x1800000Bu, // ds_read_b96 v[24:26], v11 offset:40
      0xD9BE0030u,
      0x0000100Cu, // ds_write_b128 v12, v[16:19] offset:48
      0xD9FE0040u,
      0x1C00000Du, // ds_read_b128 v[28:31], v13 offset:64
      0xD81C0703u,
      0x0015140Eu, // ds_write2_b32 v14, v20, v21 offset0:3 offset1:7
      0xD86E0904u,
      0x2000000Fu, // ds_read2_b32 v[32:33], v15 offset0:4 offset1:9
      0xD89C0502u,
      0x00181610u, // ds_write2_b64 v16, v[22:23], v[24:25]
      0xD8EE0603u,
      0x22000011u, // ds_read2_b64 v[34:37], v17
      0xD81E0301u,
      0x001B1A12u, // ds_write2st64_b32 v18, v26, v27
      0xD8700402u,
      0x26000013u, // ds_read2st64_b32 v[38:39], v19
      0xD89E0201u,
      0x001E1C14u, // ds_write2st64_b64 v20, v[28:29], v[30:31]
      0xD8F00302u,
      0x28000015u, // ds_read2st64_b64 v[40:43], v21
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  const std::vector<Expected> expected = {
      {"ds_write_b8", ConSanLdsAccessKind::Write, 8, 4, {}, 10, {}, {{3, 1}}},
      {"ds_write_b16", ConSanLdsAccessKind::Write, 16, 5, {}, 11, {}, {{5, 2}}},
      {"ds_read_u8_d16", ConSanLdsAccessKind::Read, 8, 6, 20, {}, {}, {{7, 1}}},
      {"ds_read_u16_d16_hi", ConSanLdsAccessKind::Read, 16, 7, 21, {}, {}, {{9, 2}}},
      {"ds_write_b64", ConSanLdsAccessKind::Write, 64, 8, {}, 12, {}, {{16, 8}}},
      {"ds_read_b64", ConSanLdsAccessKind::Read, 64, 9, 22, {}, {}, {{24, 8}}},
      {"ds_write_b96", ConSanLdsAccessKind::Write, 96, 10, {}, 14, {}, {{32, 12}}},
      {"ds_read_b96", ConSanLdsAccessKind::Read, 96, 11, 24, {}, {}, {{40, 12}}},
      {"ds_write_b128", ConSanLdsAccessKind::Write, 128, 12, {}, 16, {}, {{48, 16}}},
      {"ds_read_b128", ConSanLdsAccessKind::Read, 128, 13, 28, {}, {}, {{64, 16}}},
      {"ds_write2_b32", ConSanLdsAccessKind::Write, 32, 14, {}, 20, 21, {{12, 4}, {28, 4}}},
      {"ds_read2_b32", ConSanLdsAccessKind::Read, 32, 15, 32, {}, {}, {{16, 4}, {36, 4}}},
      {"ds_write2_b64", ConSanLdsAccessKind::Write, 64, 16, {}, 22, 24, {{16, 8}, {40, 8}}},
      {"ds_read2_b64", ConSanLdsAccessKind::Read, 64, 17, 34, {}, {}, {{24, 8}, {48, 8}}},
      {"ds_write2st64_b32", ConSanLdsAccessKind::Write, 32, 18, {}, 26, 27, {{256, 4}, {768, 4}}},
      {"ds_read2st64_b32", ConSanLdsAccessKind::Read, 32, 19, 38, {}, {}, {{512, 4}, {1024, 4}}},
      {"ds_write2st64_b64", ConSanLdsAccessKind::Write, 64, 20, {}, 28, 30, {{512, 8}, {1024, 8}}},
      {"ds_read2st64_b64", ConSanLdsAccessKind::Read, 64, 21, 40, {}, {}, {{1024, 8}, {1536, 8}}},
  };

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "gfx950_extended_lds", /*vgpr_granulated=*/7, /*wave32=*/false,
      /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::RecordReplay;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_EQ(result.moi_candidates.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    SCOPED_TRACE(expected[i].mnemonic);
    const ConSanMoiCandidate &candidate = result.moi_candidates[i];
    EXPECT_EQ(candidate.mnemonic, expected[i].mnemonic);
    EXPECT_EQ(candidate.kind, expected[i].kind);
    EXPECT_EQ(candidate.width_bits, expected[i].width_bits);
    EXPECT_EQ(candidate.addr_vgpr, expected[i].addr);
    EXPECT_EQ(candidate.dst_vgpr, expected[i].dst);
    EXPECT_EQ(candidate.data_vgpr, expected[i].data);
    EXPECT_EQ(candidate.secondary_data_vgpr, expected[i].secondary_data);
    ASSERT_TRUE(candidate.raw_gds);
    EXPECT_FALSE(*candidate.raw_gds);
    ASSERT_EQ(candidate.native_static_ranges.size(), expected[i].ranges.size());
    for (size_t range_index = 0; range_index < expected[i].ranges.size(); ++range_index) {
      const ConSanLdsStaticRange &actual = candidate.native_static_ranges[range_index];
      const ConSanLdsStaticRange &wanted = expected[i].ranges[range_index];
      EXPECT_EQ(actual.byte_offset, wanted.byte_offset);
      EXPECT_EQ(actual.byte_count, wanted.byte_count);
      const ConSanMoiLdsCellRange actual_cells =
          consan_moi_lds_cell_range_for_bytes(actual.byte_offset, actual.byte_count);
      const ConSanMoiLdsCellRange expected_cells =
          consan_moi_lds_cell_range_for_bytes(wanted.byte_offset, wanted.byte_count);
      EXPECT_EQ(actual_cells.start_cell, expected_cells.start_cell);
      EXPECT_EQ(actual_cells.cell_count, expected_cells.cell_count);
    }
  }
}

TEST(ConSanMoi, Gfx950UnsupportedDsInventoryHasTypedDispositionPerSite) {
  const std::array<uint32_t, 19> text_words = {
      0xD87A1234u,
      0x1E000004u, // ds_swizzle_b32 v30, v4
      0xD87C0000u,
      0x1F000605u, // ds_permute_b32 v31, v5, v6
      0xD87E0000u,
      0x20000807u, // ds_bpermute_b32 v32, v7, v8
      0xD9C0000Cu,
      0x22000009u, // ds_read_b64_tr_b4 v[34:35], v9 offset:12
      0xD8000010u,
      0x00000B0Au, // ds_add_u32 v10, v11 offset:16
      0xD8400014u,
      0x24000D0Cu, // ds_add_rtn_u32 v36, v12, v13 offset:20
      0xD8280000u,
      0x00000000u, // ds_nop
      0xD81B0018u,
      0x00000F0Eu, // reserved GDS bit on ds_write_b32 encoding
      0xD96C001Cu,
      0x25000000u, // ds_read_addtid_b32 v37 offset:28
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  const std::array<ConSanMoiLdsExclusionReason, 9> expected_reasons = {
      ConSanMoiLdsExclusionReason::PermuteOrSwizzle,
      ConSanMoiLdsExclusionReason::PermuteOrSwizzle,
      ConSanMoiLdsExclusionReason::PermuteOrSwizzle,
      ConSanMoiLdsExclusionReason::Transpose,
      ConSanMoiLdsExclusionReason::AtomicReserved,
      ConSanMoiLdsExclusionReason::AtomicReserved,
      ConSanMoiLdsExclusionReason::OtherDs,
      ConSanMoiLdsExclusionReason::GdsReserved,
      ConSanMoiLdsExclusionReason::UnsupportedAccessForm,
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "gfx950_unsupported_ds", /*vgpr_granulated=*/7, /*wave32=*/false,
      /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::RecordReplay;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_EQ(result.kernels.size(), 1u);
  EXPECT_EQ(result.kernels.front().stats.decode_error_count, 0u);
  EXPECT_TRUE(result.moi_candidates.empty());
  ASSERT_EQ(result.kernels.front().lds_sites.size(), expected_reasons.size());
  ASSERT_EQ(result.moi_lds_exclusions.size(), expected_reasons.size());
  for (size_t i = 0; i < expected_reasons.size(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(result.moi_lds_exclusions[i].text_offset,
              result.kernels.front().lds_sites[i].text_offset);
    EXPECT_EQ(result.moi_lds_exclusions[i].mnemonic, result.kernels.front().lds_sites[i].mnemonic);
    EXPECT_EQ(result.moi_lds_exclusions[i].reason, expected_reasons[i]);
  }
  ASSERT_TRUE(result.kernels.front().lds_sites[7].raw_gds);
  EXPECT_TRUE(*result.kernels.front().lds_sites[7].raw_gds);
}

TEST(ConSanMoi, Gfx950StaticSampledCapabilityEmitsNativePatch) {
  const std::array<uint32_t, 3> text_words = {
      0xD81A0000u,
      0x00000000u, // ds_write_b32 v0, v0
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "gfx950_sampled_inventory",
                                 /*vgpr_granulated=*/0, /*wave32=*/false,
                                 /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::Sampled;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = sizeof(ConSanMoiReportHeader) + 2u * sizeof(uint64_t);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.patches.size(), 2u);
  EXPECT_TRUE(std::ranges::any_of(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::InlineMoiSampledWatchpointStore ||
           patch.kind == ConSanPatchKind::TrampolineMoiSampledWatchpointStore;
  }));
  EXPECT_TRUE(std::ranges::any_of(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
  }));
}

TEST(ConSanMoi, Gfx950ScalarPlanningReservesDescriptorAbiInputsWithoutEmission) {
  std::array<uint32_t, 4> text_words = {
      0xD81A0000u,
      0x00000000u, // ds_write_b32 v0, v0
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "gfx950_system_sgprs",
                                 /*vgpr_granulated=*/0, /*wave32=*/false,
                                 /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 31u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT, 1u);
  });
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_dynamic_access_records = true;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_FALSE(result.modified);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().max_referenced_sgpr_count, 36u);
  ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
  // The patched ABI inserts the dispatch pointer after the original 36 SGPR
  // inputs, then the EXEC-save pair starts at the next even SGPR.
  EXPECT_EQ(*result.resolved_moi_exec_save_sgpr, 38u);
}

TEST(ConSanMoi, Gfx950ScalarFullPlanningFailsClosedWithoutEmission) {
  std::array<uint32_t, 4> text_words = {
      build_s_mov_b32(101, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_CDNA4),
      0xD81A0000u,
      0x00000000u, // ds_write_b32 v0, v0
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "gfx950_scalar_full",
                                 /*vgpr_granulated=*/0, /*wave32=*/false,
                                 /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_dynamic_access_records = true;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_FALSE(result.modified);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().max_referenced_sgpr_count, 102u);
  EXPECT_FALSE(result.resolved_moi_exec_save_sgpr);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("could not place a fresh automatic EXEC-save SGPR window") !=
           std::string::npos;
  }));
}

TEST(ConSanMoi, Gfx950SpecialRegisterGeometryMatchesWave64Abi) {
  const auto special = consan_moi_special_register_geometry(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(special);
  EXPECT_EQ(special->vcc_lo, 106u);
  EXPECT_EQ(special->vcc_hi, 107u);
  EXPECT_EQ(special->exec_lo, 126u);
  EXPECT_EQ(special->exec_hi, 127u);
}

TEST(ConSanMoi, FirstLightProbeAutomaticallyUsesDeadVgprs) {
  std::array<uint32_t, 170> text_words{};
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  for (size_t i = 2; i + 1 < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  text_words.back() = 0xBFB00000u; // s_endpgm

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::LivenessDead);
  EXPECT_EQ(result.resource_plans.front().scratch_vgpr, 1);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineMoiAccessRecordStore);
  EXPECT_EQ(result.patches.front().scratch_vgpr, 1);
}

TEST(ConSanMoi, FirstLightProbeAutomaticallyGrowsOwningDescriptor) {
  std::array<uint32_t, 170> text_words{};
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000102u; // ds_store_b32 v2, v1
  for (size_t i = 2; i + 1 < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  text_words.back() = 0xBFB00000u; // s_endpgm

  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, 0);
  });
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  const ConSanCandidateResourcePlan &plan = result.resource_plans.front();
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::DescriptorGrowth);
  EXPECT_EQ(plan.current_vgpr_count, 4);
  EXPECT_EQ(plan.max_referenced_vgpr_count, 3);
  EXPECT_EQ(plan.scratch_vgpr, 4);
  EXPECT_EQ(plan.required_vgpr_count, 7);
  EXPECT_EQ(result.resource_plan_summary.descriptor_growth_plans, 1u);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().scratch_vgpr, 4);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  const uint64_t descriptor_offset = patched.kernels().front().descriptor_file_offset;
  KD descriptor{};
  std::memcpy(&descriptor, result.elf_bytes.data() + descriptor_offset, sizeof(descriptor));
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc1,
                            kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT),
            1u);
}

TEST(ConSanMoi, FirstLightProbeSpillsVictimWindowInAppendedCave) {
  const std::array<uint32_t, 3> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    descriptor.private_segment_fixed_size = 32;
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT, 1u);
  });
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::SpillRequired);
  EXPECT_EQ(result.resource_plans.front().scratch_vgpr, 1);
  ASSERT_EQ(result.patches.size(), 1u);
  const ConSanPatchInfo &patch = result.patches.front();
  EXPECT_EQ(patch.kind, ConSanPatchKind::TrampolineMoiAccessRecordStore);
  EXPECT_EQ(patch.scratch_vgpr, 1);
  EXPECT_EQ(patch.spilled_vgpr_count, 3u);
  EXPECT_EQ(patch.required_private_segment_size, 44u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const auto *text = patched.text_sections().front();
  std::vector<uint32_t> actual(text->size() / sizeof(uint32_t));
  std::memcpy(actual.data(), text->data(), text->size());
  const std::vector<uint32_t> save =
      expected_vgpr_spill_words(1, 3, /*restore=*/false, /*slot_base=*/32);
  const std::vector<uint32_t> restore =
      expected_vgpr_spill_words(1, 3, /*restore=*/true, /*slot_base=*/32);
  ASSERT_FALSE(save.empty());
  ASSERT_FALSE(restore.empty());
  const size_t cave = patch.trampoline_offset / sizeof(uint32_t);
  const auto owner_init =
      build_v_lshrrev_b32_e32(3, scalar_positive_inline_u32(6), 0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(owner_init);
  ASSERT_LE(cave + save.size() + restore.size() + 4u, actual.size());
  EXPECT_TRUE(std::equal(save.begin(), save.end(), actual.begin() + cave));
  EXPECT_EQ(actual[cave + save.size()], *owner_init);
  EXPECT_EQ(actual[cave + save.size() + 1u], text_words[0]);
  EXPECT_EQ(actual[cave + save.size() + 2u], text_words[1]);
  EXPECT_TRUE(std::equal(restore.begin(), restore.end(), actual.end() - 1u - restore.size()));

  ASSERT_EQ(patched.kernels().size(), 1u);
  KD descriptor{};
  const uint64_t descriptor_offset = patched.kernels().front().descriptor_file_offset;
  std::memcpy(&descriptor, result.elf_bytes.data() + descriptor_offset, sizeof(descriptor));
  EXPECT_EQ(descriptor.private_segment_fixed_size, 44u);
  EXPECT_EQ(
      AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT),
      1u);
}

TEST(ConSanMoi, Gfx950ForcedSpillPatchHasTransactionalSaveProbeRestoreShape) {
  const std::array<uint32_t, 3> text_words = {
      0xD81A0000u,
      0x00000000u, // ds_write_b32 v0, v0
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "gfx950_forced_spill",
                                 /*vgpr_granulated=*/0, /*wave32=*/false,
                                 /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    descriptor.private_segment_fixed_size = 32;
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT, 1u);
  });
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::SpillRequired);
  const auto access_patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
  });
  ASSERT_NE(access_patch, result.patches.end());
  EXPECT_EQ(std::count_if(result.patches.begin(), result.patches.end(),
                          [](const ConSanPatchInfo &patch) {
                            return patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
                          }),
            1);
  const ConSanPatchInfo &patch = *access_patch;
  EXPECT_EQ(patch.kind, ConSanPatchKind::TrampolineMoiAccessRecordStore);
  EXPECT_EQ(patch.scratch_vgpr, 1u);
  EXPECT_EQ(patch.spilled_vgpr_count, 3u);
  EXPECT_EQ(patch.required_private_segment_size, 44u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  std::vector<uint32_t> trampoline(patch.trampoline_size / sizeof(uint32_t));
  std::memcpy(trampoline.data(), patched.text_sections().front()->data() + patch.trampoline_offset,
              patch.trampoline_size);
  const std::vector<uint32_t> save = expected_vgpr_spill_words(
      1, 3, /*restore=*/false, /*slot_base=*/32, ROCJITSU_CODE_ARCH_CDNA4);
  const std::vector<uint32_t> restore =
      expected_vgpr_spill_words(1, 3, /*restore=*/true, /*slot_base=*/32, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_FALSE(save.empty());
  ASSERT_FALSE(restore.empty());
  ASSERT_GT(trampoline.size(), save.size() + text_words.size() + restore.size());
  EXPECT_TRUE(std::equal(save.begin(), save.end(), trampoline.begin()));
  const auto access = std::search(trampoline.begin() + save.size(), trampoline.end(),
                                  text_words.begin(), text_words.begin() + 2);
  ASSERT_NE(access, trampoline.end());
  const size_t restore_offset = trampoline.size() - restore.size() - 1u;
  EXPECT_LT(static_cast<size_t>(access - trampoline.begin()) + 2u, restore_offset);
  EXPECT_TRUE(std::equal(restore.begin(), restore.end(), trampoline.begin() + restore_offset));
  EXPECT_EQ(trampoline.back() & 0xffff0000u,
            build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA4) & 0xffff0000u);
}

TEST(ConSanMoi, FirstLightProbeSupportsZeroToNonzeroDispatchScratch) {
  const std::array<uint32_t, 3> text_words = {
      0xD8340000u,
      0x00000000u,
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().spilled_vgpr_count, 3u);
  EXPECT_EQ(result.patches.front().required_private_segment_size, 12u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  EXPECT_EQ(descriptor.private_segment_fixed_size, 12u);
  EXPECT_EQ(
      AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT),
      1u);
}

TEST(ConSanMoi, FirstLightProbeRejectsSpillingDynamicStackKernel) {
  const std::array<uint32_t, 3> text_words = {
      0xD8340000u,
      0x00000000u,
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "dynamic_spill", kRdna4Wave64AllVgprsGranulated, /*wave32=*/false,
      /*uses_dynamic_stack=*/true);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  EXPECT_TRUE(result.errors.empty());
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("dynamic-stack") != std::string::npos;
  }));
}

TEST(ConSanMoi, Gfx950DynamicStackPressureFixtureRollsBackSpillTransaction) {
  const std::array<uint32_t, 3> text_words = {
      0xD81A0000u,
      0x00000000u, // ds_write_b32 v0, v0
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "gfx950_dynamic_spill",
                                 /*vgpr_granulated=*/0, /*wave32=*/false,
                                 /*uses_dynamic_stack=*/true, EF_AMDGPU_MACH_AMDGCN_GFX950);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  EXPECT_TRUE(result.errors.empty());
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::SpillRequired);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("dynamic-stack") != std::string::npos;
  }));
}

TEST(ConSanMoi, FirstLightProbeWritesOneNativeLdsAccessRecord) {
  std::array<uint32_t, 170> text_words{};
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  for (size_t i = 2; i + 1 < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  text_words.back() = 0xBFB00000u; // s_endpgm

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineMoiAccessRecordStore);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 8u);
  EXPECT_EQ(result.patches.front().original_size, 142u * sizeof(uint32_t));
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 8u);
  EXPECT_EQ(result.resource_plan_summary.explicit_plans, 1u);

  std::vector<uint32_t> expected_words = {
      0xD8340000u,
      0x00000000u, // original ds_store_b32 v0, v0
      0xBFC60000u, // s_wait_dscnt 0
  };
  auto append_expected_store = [&](uint64_t address, uint32_t value) {
    const auto mov_address_lo =
        build_v_mov_b32_e64_literal(8, static_cast<uint32_t>(address), ROCJITSU_CODE_ARCH_RDNA4);
    const auto mov_address_hi = build_v_mov_b32_e64_literal(
        9, static_cast<uint32_t>(address >> 32u), ROCJITSU_CODE_ARCH_RDNA4);
    const auto mov_value = build_v_mov_b32_e64_literal(10, value, ROCJITSU_CODE_ARCH_RDNA4);
    const auto store = build_flat_store_b32_vaddr_vsrc(8, 10, ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(mov_address_lo);
    ASSERT_TRUE(mov_address_hi);
    ASSERT_TRUE(mov_value);
    ASSERT_TRUE(store);
    expected_words.insert(expected_words.end(), mov_address_lo->begin(), mov_address_lo->end());
    expected_words.insert(expected_words.end(), mov_address_hi->begin(), mov_address_hi->end());
    expected_words.insert(expected_words.end(), mov_value->begin(), mov_value->end());
    expected_words.insert(expected_words.end(), store->begin(), store->end());
  };
  auto append_expected_store_vgpr = [&](uint64_t address, uint16_t value_vgpr) {
    const auto mov_address_lo =
        build_v_mov_b32_e64_literal(8, static_cast<uint32_t>(address), ROCJITSU_CODE_ARCH_RDNA4);
    const auto mov_address_hi = build_v_mov_b32_e64_literal(
        9, static_cast<uint32_t>(address >> 32u), ROCJITSU_CODE_ARCH_RDNA4);
    const auto store = build_flat_store_b32_vaddr_vsrc(8, value_vgpr, ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(mov_address_lo);
    ASSERT_TRUE(mov_address_hi);
    ASSERT_TRUE(store);
    expected_words.insert(expected_words.end(), mov_address_lo->begin(), mov_address_lo->end());
    expected_words.insert(expected_words.end(), mov_address_hi->begin(), mov_address_hi->end());
    expected_words.insert(expected_words.end(), store->begin(), store->end());
  };
  auto append_expected_start_cell_store = [&](uint64_t address, uint16_t lds_byte_offset_vgpr) {
    const auto shift = build_v_lshrrev_b32_e32(
        10, scalar_positive_inline_u32(consan_moi_exact_shadow::granule_shift),
        lds_byte_offset_vgpr, ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(shift);
    expected_words.push_back(*shift);
    append_expected_store_vgpr(address, 10);
  };
  auto append_expected_store_scalar_src = [&](uint64_t address, uint16_t scalar_src) {
    expected_words.push_back(build_v_mov_b32_e32(10, scalar_src, ROCJITSU_CODE_ARCH_RDNA4));
    append_expected_store_vgpr(address, 10);
  };
  auto append_expected_event_index = [&](uint64_t counter_address, uint64_t event_index_address) {
    const auto mov_address_lo = build_v_mov_b32_e64_literal(
        8, static_cast<uint32_t>(counter_address), ROCJITSU_CODE_ARCH_RDNA4);
    const auto mov_address_hi = build_v_mov_b32_e64_literal(
        9, static_cast<uint32_t>(counter_address >> 32u), ROCJITSU_CODE_ARCH_RDNA4);
    const auto mov_one = build_v_mov_b32_e64_literal(10, 1u, ROCJITSU_CODE_ARCH_RDNA4);
    const auto atomic = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
        8, 10, 10, /*return_old_value=*/true, /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(mov_address_lo);
    ASSERT_TRUE(mov_address_hi);
    ASSERT_TRUE(mov_one);
    ASSERT_TRUE(atomic);
    expected_words.insert(expected_words.end(), mov_address_lo->begin(), mov_address_lo->end());
    expected_words.insert(expected_words.end(), mov_address_hi->begin(), mov_address_hi->end());
    expected_words.insert(expected_words.end(), mov_one->begin(), mov_one->end());
    expected_words.insert(expected_words.end(), atomic->begin(), atomic->end());
    expected_words.push_back(0xBFC00000u);
    append_expected_store_vgpr(event_index_address, 10);
  };

  const uint64_t base = *options.moi_report_buffer_address;
  const uint64_t access_record_base = base + sizeof(ConSanMoiReportHeader);
  append_expected_event_index(base + offsetof(ConSanMoiReportHeader, event_counter),
                              access_record_base + offsetof(ConSanMoiAccessRecord, event_index));
  append_expected_store(base + offsetof(ConSanMoiReportHeader, access_record_count), 1u);
  append_expected_store_scalar_src(access_record_base + offsetof(ConSanMoiAccessRecord, lane_mask),
                                   /*scalar_src=*/126);
  append_expected_store_scalar_src(access_record_base + offsetof(ConSanMoiAccessRecord, lane_mask) +
                                       sizeof(uint32_t),
                                   /*scalar_src=*/127);
  append_expected_store_vgpr(access_record_base + offsetof(ConSanMoiAccessRecord, wave_id), 11);
  append_expected_store_vgpr(access_record_base + offsetof(ConSanMoiAccessRecord, epoch), 12);
  append_expected_store(access_record_base + offsetof(ConSanMoiAccessRecord, instruction_offset),
                        0u);
  append_expected_store(access_record_base + offsetof(ConSanMoiAccessRecord, access_kind),
                        static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write));
  append_expected_store_vgpr(access_record_base + offsetof(ConSanMoiAccessRecord, lds_byte_offset),
                             0);
  append_expected_start_cell_store(access_record_base + offsetof(ConSanMoiAccessRecord, start_cell),
                                   0);
  append_expected_store(access_record_base + offsetof(ConSanMoiAccessRecord, lds_byte_count), 4u);
  append_expected_store(access_record_base + offsetof(ConSanMoiAccessRecord, cell_count), 1u);

  ASSERT_EQ(expected_words.size(), 142u);
  std::vector<uint32_t> rewritten_words(expected_words.size());
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + 0x100,
              rewritten_words.size() * sizeof(uint32_t));
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(ConSanMoi, DynamicAccessRecordProbeAppendsPerLaneRecords) {
  std::array<uint32_t, 260> text_words{};
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  for (size_t i = 2; i + 1 < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.scratch_vgpr = 16;
  options.moi_exec_save_sgpr = 30;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_dynamic_access_records = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(8, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineMoiAccessRecordStore);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 16u);

  std::vector<uint32_t> rewritten_words(result.patches.front().original_size / sizeof(uint32_t));
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + 0x100,
              rewritten_words.size() * sizeof(uint32_t));

  auto make_fetch_add_one = [](uint64_t address, uint16_t result_vgpr, uint16_t scratch_vgpr) {
    std::vector<uint32_t> words;
    const auto mov_address_lo = build_v_mov_b32_e64_literal(
        scratch_vgpr, static_cast<uint32_t>(address), ROCJITSU_CODE_ARCH_RDNA4);
    const auto mov_address_hi = build_v_mov_b32_e64_literal(
        static_cast<uint16_t>(scratch_vgpr + 1u), static_cast<uint32_t>(address >> 32u),
        ROCJITSU_CODE_ARCH_RDNA4);
    const auto mov_one = build_v_mov_b32_e64_literal(result_vgpr, 1u, ROCJITSU_CODE_ARCH_RDNA4);
    const auto atomic_add = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
        scratch_vgpr, result_vgpr, result_vgpr, /*return_old_value=*/true, /*scope=*/2,
        ROCJITSU_CODE_ARCH_RDNA4);
    if (!mov_address_lo || !mov_address_hi || !mov_one || !atomic_add)
      return words;
    words.insert(words.end(), mov_address_lo->begin(), mov_address_lo->end());
    words.insert(words.end(), mov_address_hi->begin(), mov_address_hi->end());
    words.insert(words.end(), mov_one->begin(), mov_one->end());
    words.insert(words.end(), atomic_add->begin(), atomic_add->end());
    words.push_back(0xBFC00000u);
    return words;
  };

  const uint64_t base = *options.moi_report_buffer_address;
  EXPECT_TRUE(contains_subsequence(
      rewritten_words,
      make_fetch_add_one(base + offsetof(ConSanMoiReportHeader, access_record_count),
                         /*result_vgpr=*/18, /*scratch_vgpr=*/16)));
  EXPECT_TRUE(contains_subsequence(
      rewritten_words, make_fetch_add_one(base + offsetof(ConSanMoiReportHeader, event_counter),
                                          /*result_vgpr=*/21, /*scratch_vgpr=*/16)));

  const auto compare_capacity =
      build_v_cmp_gt_u32_e32_vcc(vector_source_vgpr(21), /*vsrc1=*/18, ROCJITSU_CODE_ARCH_RDNA4);
  const auto save_exec =
      build_s_and_saveexec_b64(/*sdst=*/30, /*ssrc0=*/kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_exec = build_s_mov_b64(/*sdst=*/126, /*ssrc0=*/30, ROCJITSU_CODE_ARCH_RDNA4);
  const auto save_scc = build_s_cselect_b32(
      /*sdst=*/34, scalar_positive_inline_u32(1), scalar_positive_inline_u32(0),
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto save_vcc = build_s_mov_b64(/*sdst=*/32, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_vcc = build_s_mov_b64(kRdna4VccLo, /*ssrc0=*/32, ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_scc = build_s_cmp_lg_u32(
      /*ssrc0=*/34, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(compare_capacity);
  ASSERT_TRUE(save_exec);
  ASSERT_TRUE(restore_exec);
  ASSERT_TRUE(save_scc);
  ASSERT_TRUE(save_vcc);
  ASSERT_TRUE(restore_vcc);
  ASSERT_TRUE(restore_scc);
  EXPECT_TRUE(std::find(rewritten_words.begin(), rewritten_words.end(), *save_scc) !=
              rewritten_words.end());
  EXPECT_TRUE(std::find(rewritten_words.begin(), rewritten_words.end(), *save_vcc) !=
              rewritten_words.end());
  EXPECT_TRUE(std::find(rewritten_words.begin(), rewritten_words.end(), *compare_capacity) !=
              rewritten_words.end());
  EXPECT_TRUE(std::find(rewritten_words.begin(), rewritten_words.end(), *save_exec) !=
              rewritten_words.end());
  EXPECT_TRUE(std::find(rewritten_words.begin(), rewritten_words.end(), *restore_exec) !=
              rewritten_words.end());
  EXPECT_TRUE(std::find(rewritten_words.begin(), rewritten_words.end(), *restore_vcc) !=
              rewritten_words.end());
  EXPECT_TRUE(std::find(rewritten_words.begin(), rewritten_words.end(), *restore_scc) !=
              rewritten_words.end());
  const auto save_scc_it = std::find(rewritten_words.begin(), rewritten_words.end(), *save_scc);
  const auto save_vcc_it = std::find(rewritten_words.begin(), rewritten_words.end(), *save_vcc);
  const auto save_exec_it = std::find(rewritten_words.begin(), rewritten_words.end(), *save_exec);
  const auto restore_exec_it =
      std::find(rewritten_words.begin(), rewritten_words.end(), *restore_exec);
  const auto restore_vcc_it =
      std::find(rewritten_words.begin(), rewritten_words.end(), *restore_vcc);
  const auto restore_scc_it =
      std::find(rewritten_words.begin(), rewritten_words.end(), *restore_scc);
  // VCC and SCC use scalar snapshots, so this sequence remains valid even if
  // the incoming EXEC mask has no active lane.
  EXPECT_LT(save_scc_it, save_vcc_it);
  EXPECT_LT(save_vcc_it, save_exec_it);
  EXPECT_LT(restore_exec_it, restore_vcc_it);
  EXPECT_LT(restore_vcc_it, restore_scc_it);
}

TEST(ConSanMoi, DynamicAccessRecordReportsBoundedFullSgprFileFailure) {
  std::vector<uint32_t> text_words(360, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words[0] =
      build_s_mov_b32(/*sdst=*/105, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  text_words[1] = 0xD8340000u;
  text_words[2] = 0x00000000u; // ds_store_b32 v0, v0
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 15u);
  });
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_dynamic_access_records = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(8, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  EXPECT_TRUE(result.errors.empty());
  EXPECT_FALSE(result.modified);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().max_referenced_sgpr_count, 106u);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("could not place a fresh automatic EXEC-save SGPR window") !=
           std::string::npos;
  }));
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("requires RJ_CONSAN_MOI_EXEC_SAVE_SGPR") != std::string::npos;
  }));
}

TEST(ConSanMoi, DynamicAccessRecordPreservesWave32AndWave64SpecialState) {
  for (bool wave32 : {false, true}) {
    std::vector<uint32_t> text_words(360, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
    text_words[0] = 0xD8340000u;
    text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
    text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
    const std::vector<uint8_t> bytes =
        make_rdna4_lds_code_object(text_words, wave32 ? "dynamic_wave32" : "dynamic_wave64",
                                   kRdna4Wave64AllVgprsGranulated, wave32);
    ConSanOptions options;
    options.flavor = ConSanFlavor::Moi;
    options.moi_dynamic_access_records = true;
    options.moi_report_buffer_address = 0x123456780000ull;
    options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(8, 0, 0, 0);

    const auto result = try_patch_consan(bytes, options);

    ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
    ASSERT_TRUE(result.modified);
    ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
    ASSERT_EQ(result.patches.size(), 1u);
    AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(patched.is_valid());
    ASSERT_EQ(patched.kernels().size(), 1u);
    KD descriptor{};
    std::memcpy(&descriptor,
                result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
                sizeof(descriptor));
    EXPECT_EQ(AMDHSA_BITS_GET(descriptor.kernel_code_properties,
                              kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32),
              wave32 ? 1u : 0u);

    std::vector<uint32_t> words(result.patches.front().original_size / sizeof(uint32_t));
    std::memcpy(words.data(), result.elf_bytes.data() + 0x100, words.size() * sizeof(uint32_t));
    const uint16_t base = *result.resolved_moi_exec_save_sgpr;
    const auto save_exec = build_s_and_saveexec_b64(base, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
    const auto save_vcc =
        build_s_mov_b64(static_cast<uint16_t>(base + 2u), kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
    const auto save_scc =
        build_s_cselect_b32(static_cast<uint16_t>(base + 4u), scalar_positive_inline_u32(1),
                            scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(save_exec);
    ASSERT_TRUE(save_vcc);
    ASSERT_TRUE(save_scc);
    EXPECT_TRUE(std::find(words.begin(), words.end(), *save_exec) != words.end());
    EXPECT_TRUE(contains_subsequence(words, std::array<uint32_t, 2>{*save_scc, *save_vcc}));
  }
}

TEST(ConSanMoi, DirectSampledProbeWritesPackedWatchpointEntry) {
  std::array<uint32_t, 170> text_words{};
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  for (size_t i = 2; i + 1 < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  text_words.back() = 0xBFB00000u; // s_endpgm

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::Sampled;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 13;
  options.moi_epoch_vgpr = 14;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = sizeof(ConSanMoiReportHeader) + 2u * sizeof(uint64_t);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineMoiSampledWatchpointStore);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 8u);

  bool saw_sampled_warning = false;
  bool saw_access_warning = false;
  for (const std::string &warning : result.warnings) {
    saw_sampled_warning |= warning.find("direct sampled watchpoint") != std::string::npos;
    saw_access_warning |= warning.find("access record") != std::string::npos;
  }
  EXPECT_TRUE(saw_sampled_warning);
  EXPECT_FALSE(saw_access_warning);

  std::vector<uint32_t> rewritten_words(result.patches.front().original_size / sizeof(uint32_t));
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + 0x100,
              rewritten_words.size() * sizeof(uint32_t));
  const auto default_bound =
      build_v_cmp_gt_u32_e32_vcc(vector_source_vgpr(8), /*vsrc1=*/9, ROCJITSU_CODE_ARCH_RDNA4);
  const auto overflow_branch = build_s_cbranch_vccz(0, ROCJITSU_CODE_ARCH_RDNA4);
  const auto overflow_atomic = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/8, /*vsrc=*/12, /*vdst=*/12, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(default_bound);
  ASSERT_TRUE(overflow_branch);
  ASSERT_TRUE(overflow_atomic);
  EXPECT_GE(std::count(rewritten_words.begin(), rewritten_words.end(), *default_bound), 3);
  EXPECT_GE(std::count_if(rewritten_words.begin(), rewritten_words.end(),
                          [&](uint32_t word) {
                            return (word & 0xffff0000u) == (*overflow_branch & 0xffff0000u);
                          }),
            3);
  EXPECT_TRUE(contains_subsequence(rewritten_words, *overflow_atomic));
}

TEST(ConSanMoi, DirectSampledProbeAutomaticallyUsesDeadVgprs) {
  std::array<uint32_t, 170> text_words{};
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  for (size_t i = 2; i + 1 < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  text_words.back() = 0xBFB00000u; // s_endpgm

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::Sampled;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = sizeof(ConSanMoiReportHeader) + 2u * sizeof(uint64_t);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::LivenessDead);
  EXPECT_EQ(result.resource_plans.front().scratch_vgpr, 1);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineMoiSampledWatchpointStore);
  EXPECT_EQ(result.patches.front().scratch_vgpr, 1);
}

TEST(ConSanMoi, DirectSampledProbeSpillsFiveVgprsInAppendedCave) {
  const std::array<uint32_t, 3> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    descriptor.private_segment_fixed_size = 32;
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT, 1u);
  });
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::Sampled;
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = sizeof(ConSanMoiReportHeader) + 2u * sizeof(uint64_t);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::SpillRequired);
  EXPECT_EQ(result.resource_plans.front().scratch_vgpr, 1);
  ASSERT_EQ(result.patches.size(), 1u);
  const ConSanPatchInfo &patch = result.patches.front();
  EXPECT_EQ(patch.kind, ConSanPatchKind::TrampolineMoiSampledWatchpointStore);
  EXPECT_EQ(patch.scratch_vgpr, 1);
  EXPECT_EQ(patch.spilled_vgpr_count, 5u);
  EXPECT_EQ(patch.required_private_segment_size, 52u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const auto *text = patched.text_sections().front();
  std::vector<uint32_t> actual(text->size() / sizeof(uint32_t));
  std::memcpy(actual.data(), text->data(), text->size());
  const std::vector<uint32_t> save =
      expected_vgpr_spill_words(1, 5, /*restore=*/false, /*slot_base=*/32);
  const std::vector<uint32_t> restore =
      expected_vgpr_spill_words(1, 5, /*restore=*/true, /*slot_base=*/32);
  ASSERT_FALSE(save.empty());
  ASSERT_FALSE(restore.empty());
  const size_t cave = patch.trampoline_offset / sizeof(uint32_t);
  const auto owner_init =
      build_v_lshrrev_b32_e32(5, scalar_positive_inline_u32(6), 0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
  const auto save_vcc =
      build_s_mov_b64(*result.resolved_moi_exec_save_sgpr, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(owner_init);
  ASSERT_TRUE(save_vcc);
  ASSERT_LE(cave + save.size() + restore.size() + 4u, actual.size());
  EXPECT_TRUE(std::equal(save.begin(), save.end(), actual.begin() + cave));
  EXPECT_EQ(actual[cave + save.size()], *save_vcc);
  EXPECT_EQ(actual[cave + save.size() + 1u], *owner_init);
  EXPECT_EQ(actual[cave + save.size() + 2u], text_words[0]);
  EXPECT_EQ(actual[cave + save.size() + 3u], text_words[1]);
  EXPECT_TRUE(std::equal(restore.begin(), restore.end(), actual.end() - 1u - restore.size()));

  ASSERT_EQ(patched.kernels().size(), 1u);
  KD descriptor{};
  const uint64_t descriptor_offset = patched.kernels().front().descriptor_file_offset;
  std::memcpy(&descriptor, result.elf_bytes.data() + descriptor_offset, sizeof(descriptor));
  EXPECT_EQ(descriptor.private_segment_fixed_size, 52u);
  EXPECT_EQ(
      AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT),
      1u);
}

TEST(ConSanMoi, DirectSampledProbeCanUseSleepDelay) {
  std::array<uint32_t, 170> text_words{};
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  for (size_t i = 2; i + 1 < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::Sampled;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 13;
  options.moi_epoch_vgpr = 14;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = sizeof(ConSanMoiReportHeader) + 2u * sizeof(uint64_t);
  options.delay_mode = ConSanDelayMode::Sleep;
  options.delay_nops = 7;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineMoiSampledWatchpointStore);

  std::vector<uint32_t> rewritten_words(result.patches.front().original_size / sizeof(uint32_t));
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + 0x100,
              rewritten_words.size() * sizeof(uint32_t));
  EXPECT_NE(std::find(rewritten_words.begin(), rewritten_words.end(),
                      build_s_sleep(7, ROCJITSU_CODE_ARCH_RDNA4)),
            rewritten_words.end());
}

TEST(ConSanMoi, DirectSampledProbeCanUseSleepVarDelay) {
  std::array<uint32_t, 170> text_words{};
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  for (size_t i = 2; i + 1 < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::Sampled;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 13;
  options.moi_epoch_vgpr = 14;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = sizeof(ConSanMoiReportHeader) + 2u * sizeof(uint64_t);
  options.delay_mode = ConSanDelayMode::SleepVar;
  options.delay_nops = 1;
  options.delay_var_ssrc = kRdna4VccLo;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineMoiSampledWatchpointStore);

  std::vector<uint32_t> rewritten_words(result.patches.front().original_size / sizeof(uint32_t));
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + 0x100,
              rewritten_words.size() * sizeof(uint32_t));
  EXPECT_NE(std::find(rewritten_words.begin(), rewritten_words.end(),
                      build_s_sleep_var(kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4)),
            rewritten_words.end());
}

TEST(ConSanMoi, DirectSampledProbeRejectsOversizedSleepVarSource) {
  std::array<uint32_t, 170> text_words{};
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  for (size_t i = 2; i + 1 < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::Sampled;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 13;
  options.moi_epoch_vgpr = 14;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = sizeof(ConSanMoiReportHeader) + 2u * sizeof(uint64_t);
  options.delay_mode = ConSanDelayMode::SleepVar;
  options.delay_nops = 1;
  options.delay_var_ssrc = 300;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_FALSE(result.modified);

  bool saw_sleep_var_warning = false;
  for (const std::string &warning : result.warnings)
    saw_sleep_var_warning |=
        warning.find("sleep_var source exceeds the 8-bit scalar source field") != std::string::npos;
  EXPECT_TRUE(saw_sleep_var_warning);
}

TEST(ConSanMoi, DirectSampledProbeCanStrideCandidateSelection) {
  constexpr uint32_t kSecondSiteWord = 170;
  std::vector<uint32_t> text_words(360, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  text_words[kSecondSiteWord] = 0xD8D80000u;
  text_words[kSecondSiteWord + 1] = 0x01000000u; // ds_load_b32 v1, v0
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::Sampled;
  options.scratch_vgpr = 20;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = sizeof(ConSanMoiReportHeader) + 2u * sizeof(uint64_t);
  options.max_patches = 2;
  options.moi_sample_stride = 2;
  options.moi_sample_offset = 1;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineMoiSampledWatchpointStore);
  EXPECT_EQ(result.patches.front().anchor_offset, kSecondSiteWord * sizeof(uint32_t));
}

TEST(ConSanMoi, DirectSampledProbeRuntimeWaveSelectionKeepsAllSitesPatchable) {
  constexpr uint32_t kSecondSiteWord = 170;
  std::vector<uint32_t> text_words(420, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  text_words[kSecondSiteWord] = 0xD8D80000u;
  text_words[kSecondSiteWord + 1] = 0x01000000u; // ds_load_b32 v1, v0
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::Sampled;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = sizeof(ConSanMoiReportHeader) + 2u * sizeof(uint64_t);
  options.max_patches = 2;
  options.moi_runtime_sample_stride = 4;
  options.moi_runtime_sample_offset = 1;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 2u);
  EXPECT_EQ(result.patches[0].anchor_offset, 0u);
  EXPECT_EQ(result.patches[1].anchor_offset, kSecondSiteWord * sizeof(uint32_t));
  EXPECT_TRUE(result.moi_exec_save_sgprs_automatic);
  ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const auto *text_section = patched.text_sections().front();
  ASSERT_EQ(text_section->size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> patched_words(text_section->size() / sizeof(uint32_t));
  std::memcpy(patched_words.data(), text_section->data(), text_section->size());

  const uint16_t save_sgpr = *result.resolved_moi_exec_save_sgpr;
  const auto save_vcc = build_s_mov_b64(save_sgpr, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_vcc = build_s_mov_b64(kRdna4VccLo, save_sgpr, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(save_vcc);
  ASSERT_TRUE(restore_vcc);
  EXPECT_EQ(std::count(patched_words.begin(), patched_words.end(), *save_vcc), 2);
  EXPECT_EQ(std::count(patched_words.begin(), patched_words.end(), *restore_vcc), 2);
  for (const ConSanPatchInfo &patch : result.patches) {
    ASSERT_TRUE(patch.scratch_vgpr);
    const uint16_t low_vgpr = static_cast<uint16_t>(*patch.scratch_vgpr + 2u);
    const uint16_t owner_vgpr = static_cast<uint16_t>(*patch.scratch_vgpr + 4u);
    const auto owner_mask =
        build_v_and_b32_e32_literal(low_vgpr, /*literal=*/3, owner_vgpr, ROCJITSU_CODE_ARCH_RDNA4);
    const auto selected = build_v_cmp_eq_u32_e32_vcc(scalar_positive_inline_u32(1), low_vgpr,
                                                     ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(owner_mask);
    ASSERT_TRUE(selected);
    EXPECT_TRUE(contains_subsequence(patched_words, *owner_mask));
    EXPECT_NE(std::find(patched_words.begin(), patched_words.end(), *selected),
              patched_words.end());
  }
}

TEST(ConSanMoi, DirectSampledProbeCanCheckPriorSlotInKernel) {
  constexpr uint32_t kSecondSiteWord = 240;
  std::vector<uint32_t> text_words(560, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  text_words[kSecondSiteWord] = 0xD8D80000u;
  text_words[kSecondSiteWord + 1] = 0x01000000u; // ds_load_b32 v1, v0
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::Sampled;
  options.moi_sampled_check = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = sizeof(ConSanMoiReportHeader) + 2u * sizeof(uint64_t);
  options.max_patches = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 2u);
  EXPECT_TRUE(result.moi_exec_save_sgprs_automatic);
  ASSERT_TRUE(result.patches[1].scratch_vgpr);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const auto *text_section = patched.text_sections().front();
  ASSERT_EQ(text_section->size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> patched_words(text_section->size() / sizeof(uint32_t));
  std::memcpy(patched_words.data(), text_section->data(), text_section->size());

  const uint16_t scratch = *result.patches[1].scratch_vgpr;
  const auto diagnostic_increment = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      scratch, static_cast<uint16_t>(scratch + 4u), static_cast<uint16_t>(scratch + 4u),
      /*return_old_value=*/true, /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(diagnostic_increment);
  // The immediate-conflict and out-of-partition counters deliberately use the
  // same atomic instruction shape at different report-header addresses.
  EXPECT_EQ(count_subsequence(patched_words, *diagnostic_increment), 2u);
}

TEST(ConSanMoi, DirectSampledProbeWarnsWhenReportCapacityLimitsPatches) {
  std::vector<uint32_t> text_words(360, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  text_words[170] = 0xD8D80000u;
  text_words[171] = 0x01000000u; // ds_load_b32 v1, v0
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::Sampled;
  options.scratch_vgpr = 20;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = sizeof(ConSanMoiReportHeader) + sizeof(uint64_t);
  options.max_patches = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineMoiSampledWatchpointStore);

  bool saw_limit_warning = false;
  for (const std::string &warning : result.warnings) {
    saw_limit_warning |= warning.find("limited sampled patches to 1 of 2") != std::string::npos;
  }
  EXPECT_TRUE(saw_limit_warning);
}

TEST(ConSanMoi, Gfx950SampledProbePartitionsBySnapshotWorkgroupIdentity) {
  std::array<uint32_t, 260> text_words{};
  text_words[0] = 0xD81A0000u;
  text_words[1] = 0x00000000u; // ds_write_b32 v0, v0
  for (size_t i = 2; i + 1 < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);

  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "gfx950_sampled_partition", /*vgpr_granulated=*/3, /*wave32=*/false,
      /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 5u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z,
                    1u);
  });

  constexpr uint32_t kSampledCapacity = 16;
  const uint64_t report_size = sizeof(ConSanMoiReportHeader) + kSampledCapacity * sizeof(uint64_t);
  const ConSanMoiReportBufferLayout layout =
      consan_moi_direct_sampled_report_buffer_layout_for_bytes(report_size);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::Sampled;
  options.max_patches = 1;
  // Make the sampled payload itself straddle a 4 GiB low-word boundary.
  options.moi_report_buffer_address =
      0x123400000000ull + (uint64_t{1} << 32u) - layout.sampled_watchpoints_offset - 32u;
  options.moi_report_buffer_size = report_size;
  options.moi_workgroup_extent_x = 2;
  options.moi_workgroup_extent_y = 2;
  options.moi_workgroup_extent_z = 2;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_FALSE(result.moi_scalar_identity_automatic);
  for (const auto vgpr : result.resolved_moi_workgroup_vgprs)
    ASSERT_TRUE(vgpr);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::InlineMoiSampledWatchpointStore ||
           patch.kind == ConSanPatchKind::TrampolineMoiSampledWatchpointStore;
  });
  ASSERT_NE(access, result.patches.end());
  ASSERT_TRUE(access->scratch_vgpr);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const auto *text = patched.text_sections().front();
  const uint64_t rewritten_offset =
      access->trampoline_size != 0 ? access->trampoline_offset : access->anchor_offset;
  const uint32_t rewritten_size =
      access->trampoline_size != 0 ? access->trampoline_size : access->original_size;
  std::vector<uint32_t> rewritten_words(rewritten_size / sizeof(uint32_t));
  std::memcpy(rewritten_words.data(), text->data() + rewritten_offset,
              rewritten_words.size() * sizeof(uint32_t));

  const uint16_t scratch = *access->scratch_vgpr;
  const uint16_t low = static_cast<uint16_t>(scratch + 2u);
  const uint16_t high = static_cast<uint16_t>(scratch + 3u);
  const uint16_t tmp = static_cast<uint16_t>(scratch + 4u);
  const auto store_low = build_flat_store_b32_vaddr_vsrc(scratch, low, ROCJITSU_CODE_ARCH_CDNA4);
  const auto address_increment =
      build_v_mov_b32_e64_literal(tmp, sizeof(uint32_t), ROCJITSU_CODE_ARCH_CDNA4);
  const auto add_address =
      build_v_add_nc_u32_words(scratch, vector_source_vgpr(scratch), tmp, ROCJITSU_CODE_ARCH_CDNA4);
  const auto store_high = build_flat_store_b32_vaddr_vsrc(scratch, high, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(store_low && address_increment && add_address && store_high);
  std::vector<uint32_t> ordered_stores;
  ordered_stores.insert(ordered_stores.end(), store_low->begin(), store_low->end());
  ordered_stores.insert(ordered_stores.end(), address_increment->begin(), address_increment->end());
  ordered_stores.insert(ordered_stores.end(), add_address->begin(), add_address->end());
  ordered_stores.insert(ordered_stores.end(), store_high->begin(), store_high->end());
  EXPECT_TRUE(contains_subsequence(rewritten_words, ordered_stores))
      << "the sampled entry low word must be stored before advancing to its high word";
  for (const auto vgpr : result.resolved_moi_workgroup_vgprs) {
    EXPECT_NE(std::find(rewritten_words.begin(), rewritten_words.end(),
                        build_v_mov_b32_e32(static_cast<uint16_t>(scratch + 1u),
                                            vector_source_vgpr(*vgpr), ROCJITSU_CODE_ARCH_CDNA4)),
              rewritten_words.end());
  }
  const auto partition =
      build_v_mad_u32_u24(static_cast<uint16_t>(scratch + 4u), vector_source_vgpr(scratch),
                          static_cast<uint16_t>(scratch + 1u), static_cast<uint16_t>(scratch + 4u),
                          ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(partition);
  EXPECT_TRUE(contains_subsequence(rewritten_words, *partition));
  const auto overflow_atomic = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      scratch, static_cast<uint16_t>(scratch + 4u), static_cast<uint16_t>(scratch + 4u),
      /*return_old_value=*/true, /*scope=*/2, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(overflow_atomic);
  EXPECT_TRUE(contains_subsequence(rewritten_words, *overflow_atomic));
  EXPECT_NE(std::find(rewritten_words.begin(), rewritten_words.end(),
                      build_v_cmp_gt_u32_e32_vcc(vector_source_vgpr(scratch + 4u), scratch + 1u,
                                                 ROCJITSU_CODE_ARCH_CDNA4)),
            rewritten_words.end())
      << "the cross-4GiB address path must compare the byte offset for carry";
}

TEST(ConSanMoi, Gfx950SampledProbeRejectsPartitionWithoutOneSlotPerWorkgroup) {
  const std::array<uint32_t, 3> text_words = {
      0xD81A0000u,
      0x00000000u, // ds_write_b32 v0, v0
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "gfx950_sampled_partition_capacity", /*vgpr_granulated=*/3,
      /*wave32=*/false, /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::Sampled;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = sizeof(ConSanMoiReportHeader) + 7u * sizeof(uint64_t);
  options.moi_workgroup_extent_x = 2;
  options.moi_workgroup_extent_y = 2;
  options.moi_workgroup_extent_z = 2;

  const ConSanResult result = try_patch_consan(bytes, options);

  EXPECT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("one report slot per workgroup partition") != std::string::npos;
  }));
}

TEST(ConSanMoi, FirstLightProbeDerivesOwnerFromWorkitemIdWhenOwnerVgprIsUnset) {
  std::array<uint32_t, 170> text_words{};
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  for (size_t i = 2; i + 1 < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "lds_probe", kRdna4Wave64AllVgprsGranulated, /*wave32=*/true);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.scratch_vgpr = 8;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineMoiAccessRecordStore);
  EXPECT_EQ(result.patches.front().original_size, 134u * sizeof(uint32_t));

  const uint64_t access_record_base =
      *options.moi_report_buffer_address + sizeof(ConSanMoiReportHeader);
  std::vector<uint32_t> expected_prefix;
  const auto owner_init =
      build_v_lshrrev_b32_e32(10, scalar_positive_inline_u32(5), 0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(owner_init);
  expected_prefix.push_back(*owner_init);
  expected_prefix.push_back(0xD8340000u);
  expected_prefix.push_back(0x00000000u);
  expected_prefix.push_back(0xBFC60000u);
  const auto mov_address_lo = build_v_mov_b32_e64_literal(
      8, static_cast<uint32_t>(access_record_base + offsetof(ConSanMoiAccessRecord, wave_id)),
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto mov_address_hi = build_v_mov_b32_e64_literal(
      9,
      static_cast<uint32_t>((access_record_base + offsetof(ConSanMoiAccessRecord, wave_id)) >> 32u),
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto store_owner = build_flat_store_b32_vaddr_vsrc(8, 10, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(mov_address_lo);
  ASSERT_TRUE(mov_address_hi);
  ASSERT_TRUE(store_owner);
  expected_prefix.insert(expected_prefix.end(), mov_address_lo->begin(), mov_address_lo->end());
  expected_prefix.insert(expected_prefix.end(), mov_address_hi->begin(), mov_address_hi->end());
  expected_prefix.insert(expected_prefix.end(), store_owner->begin(), store_owner->end());

  std::vector<uint32_t> rewritten_prefix(expected_prefix.size());
  std::memcpy(rewritten_prefix.data(), result.elf_bytes.data() + 0x100,
              rewritten_prefix.size() * sizeof(uint32_t));
  EXPECT_EQ(rewritten_prefix, expected_prefix);
}

TEST(ConSanMoi, FirstLightProbeStoresDescriptorWorkgroupIds) {
  std::array<uint32_t, 220> text_words{};
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  for (size_t i = 2; i + 1 < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);

  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 5u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z,
                    1u);
  });

  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineMoiAccessRecordStore);

  const ConSanPatchInfo &patch = result.patches.front();
  const uint64_t rewritten_offset =
      patch.trampoline_size != 0 ? patch.trampoline_offset : patch.anchor_offset;
  const uint32_t rewritten_size =
      patch.trampoline_size != 0 ? patch.trampoline_size : patch.original_size;
  std::vector<uint32_t> rewritten_words(rewritten_size / sizeof(uint32_t));
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + rewritten_offset,
              rewritten_words.size() * sizeof(uint32_t));

  const uint64_t access_record_base =
      *options.moi_report_buffer_address + sizeof(ConSanMoiReportHeader);
  const std::vector<uint32_t> expected_x = make_expected_scalar_store_words(
      access_record_base + offsetof(ConSanMoiAccessRecord, workgroup_x),
      ttmp_scalar_operand(kTtmpRdna4GridX), *options.scratch_vgpr);
  const std::vector<uint32_t> expected_y = make_expected_scalar_store_words(
      access_record_base + offsetof(ConSanMoiAccessRecord, workgroup_y),
      ttmp_scalar_operand(kTtmpRdna4GridYz), *options.scratch_vgpr,
      /*shift_right_16=*/false, /*mask_low_16=*/true);
  const std::vector<uint32_t> expected_z = make_expected_scalar_store_words(
      access_record_base + offsetof(ConSanMoiAccessRecord, workgroup_z),
      ttmp_scalar_operand(kTtmpRdna4GridYz), *options.scratch_vgpr,
      /*shift_right_16=*/true);
  EXPECT_TRUE(contains_subsequence(rewritten_words, expected_x));
  EXPECT_TRUE(contains_subsequence(rewritten_words, expected_y));
  EXPECT_TRUE(contains_subsequence(rewritten_words, expected_z))
      << "expected=" << testing::PrintToString(expected_z)
      << " actual=" << testing::PrintToString(rewritten_words);
}

TEST(ConSanMoi, Gfx950FirstLightUsesStableEntrySnapshotsForWorkgroupAndWaveIdentity) {
  std::array<uint32_t, 220> text_words{};
  text_words[0] = 0xD81A0000u;
  text_words[1] = 0x00000000u; // ds_write_b32 v0, v0
  for (size_t i = 2; i + 1 < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);

  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "gfx950_identity", /*vgpr_granulated=*/3, /*wave32=*/false,
      /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 5u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO,
                    1u);
  });

  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.scratch_vgpr = 8;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto access_patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
           patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
  });
  const auto prologue_patch =
      std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
        return patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
      });
  ASSERT_NE(access_patch, result.patches.end());
  ASSERT_NE(prologue_patch, result.patches.end());
  ASSERT_TRUE(result.resolved_moi_owner_vgpr);
  ASSERT_TRUE(result.resolved_moi_epoch_vgpr);
  ASSERT_TRUE(result.resolved_moi_identity_sgpr);
  for (const auto vgpr : result.resolved_moi_workgroup_vgprs)
    ASSERT_TRUE(vgpr);

  const uint32_t rewritten_size = access_patch->trampoline_size != 0 ? access_patch->trampoline_size
                                                                     : access_patch->original_size;
  std::vector<uint32_t> rewritten_words(rewritten_size / sizeof(uint32_t));
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const uint64_t rewritten_offset = access_patch->trampoline_size != 0
                                        ? access_patch->trampoline_offset
                                        : access_patch->anchor_offset;
  std::memcpy(rewritten_words.data(), patched.text_sections().front()->data() + rewritten_offset,
              rewritten_words.size() * sizeof(uint32_t));

  const uint64_t access_record_base =
      *options.moi_report_buffer_address + sizeof(ConSanMoiReportHeader);
  const auto expected_x = make_expected_scalar_store_words(
      access_record_base + offsetof(ConSanMoiAccessRecord, workgroup_x),
      vector_source_vgpr(*result.resolved_moi_workgroup_vgprs[0]), *options.scratch_vgpr, false,
      false, ROCJITSU_CODE_ARCH_CDNA4);
  const auto expected_y = make_expected_scalar_store_words(
      access_record_base + offsetof(ConSanMoiAccessRecord, workgroup_y),
      vector_source_vgpr(*result.resolved_moi_workgroup_vgprs[1]), *options.scratch_vgpr, false,
      false, ROCJITSU_CODE_ARCH_CDNA4);
  const auto expected_z = make_expected_scalar_store_words(
      access_record_base + offsetof(ConSanMoiAccessRecord, workgroup_z),
      vector_source_vgpr(*result.resolved_moi_workgroup_vgprs[2]), *options.scratch_vgpr, false,
      false, ROCJITSU_CODE_ARCH_CDNA4);
  EXPECT_TRUE(contains_subsequence(rewritten_words, expected_x));
  EXPECT_TRUE(contains_subsequence(rewritten_words, expected_y));
  EXPECT_TRUE(contains_subsequence(rewritten_words, expected_z))
      << "expected=" << testing::PrintToString(expected_z)
      << " actual=" << testing::PrintToString(rewritten_words);

  std::vector<uint32_t> prologue_words(prologue_patch->trampoline_size / sizeof(uint32_t));
  std::memcpy(prologue_words.data(),
              patched.text_sections().front()->data() + prologue_patch->trampoline_offset,
              prologue_patch->trampoline_size);
  const auto dispatch_load =
      build_s_load_dword(*result.resolved_moi_identity_sgpr,
                         /*dispatch_sgpr=*/0, /*byte_offset=*/4, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(dispatch_load);
  EXPECT_TRUE(contains_subsequence(
      prologue_words, std::vector<uint32_t>(dispatch_load->begin(), dispatch_load->end())));
  const std::vector<uint32_t> expected_snapshots = {
      build_v_mov_b32_e32(*result.resolved_moi_workgroup_vgprs[0], /*shifted s5=*/7,
                          ROCJITSU_CODE_ARCH_CDNA4),
      build_v_mov_b32_e32(*result.resolved_moi_workgroup_vgprs[1], /*shifted s6=*/8,
                          ROCJITSU_CODE_ARCH_CDNA4),
      build_v_mov_b32_e32(*result.resolved_moi_workgroup_vgprs[2], /*shifted s7=*/9,
                          ROCJITSU_CODE_ARCH_CDNA4),
  };
  EXPECT_TRUE(contains_subsequence(prologue_words, expected_snapshots));

  const uint16_t owner_vgpr = *result.resolved_moi_owner_vgpr;
  const uint16_t epoch_vgpr = *result.resolved_moi_epoch_vgpr;
  const uint16_t size_x_sgpr = *result.resolved_moi_identity_sgpr;
  const uint16_t size_y_sgpr = static_cast<uint16_t>(size_x_sgpr + 1u);
  const auto z = build_v_lshrrev_b32_e32(owner_vgpr, scalar_positive_inline_u32(20), 0,
                                         ROCJITSU_CODE_ARCH_CDNA4);
  const auto y = build_v_lshrrev_b32_e32(epoch_vgpr, scalar_positive_inline_u32(10), 0,
                                         ROCJITSU_CODE_ARCH_CDNA4);
  const auto mask_y =
      build_v_and_b32_e32_literal(epoch_vgpr, 0x3ffu, epoch_vgpr, ROCJITSU_CODE_ARCH_CDNA4);
  const auto yz = build_v_mad_u32_u24(owner_vgpr, size_y_sgpr, owner_vgpr, epoch_vgpr,
                                      ROCJITSU_CODE_ARCH_CDNA4);
  const auto x = build_v_and_b32_e32_literal(epoch_vgpr, 0x3ffu, 0, ROCJITSU_CODE_ARCH_CDNA4);
  const auto flat = build_v_mad_u32_u24(owner_vgpr, size_x_sgpr, owner_vgpr, epoch_vgpr,
                                        ROCJITSU_CODE_ARCH_CDNA4);
  const auto wave = build_v_lshrrev_b32_e32(owner_vgpr, scalar_positive_inline_u32(6), owner_vgpr,
                                            ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(z && y && mask_y && yz && x && flat && wave);
  std::vector<uint32_t> expected_owner = {*z, *y};
  expected_owner.insert(expected_owner.end(), mask_y->begin(), mask_y->end());
  expected_owner.insert(expected_owner.end(), yz->begin(), yz->end());
  expected_owner.insert(expected_owner.end(), x->begin(), x->end());
  expected_owner.insert(expected_owner.end(), flat->begin(), flat->end());
  expected_owner.push_back(*wave);
  expected_owner.push_back(
      build_v_mov_b32_e32(epoch_vgpr, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_TRUE(contains_subsequence(prologue_words, expected_owner));

  std::vector<uint32_t> expected_restore;
  for (uint16_t destination = 0; destination < 9; ++destination) {
    expected_restore.push_back(build_s_mov_b32(destination, static_cast<uint16_t>(destination + 2u),
                                               ROCJITSU_CODE_ARCH_CDNA4));
  }
  EXPECT_TRUE(contains_subsequence(prologue_words, expected_restore));

  ASSERT_EQ(patched.kernels().size(), 1u);
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.kernel_code_properties,
                            kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR),
            1u);
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT),
            7u);
}

TEST(ConSanMoi, Gfx950IdentityTransactionSuppliesOmittedCoordinatesAndRestoresGuestAbi) {
  struct Case {
    const char *name;
    std::array<bool, 3> original_workgroup_ids;
    bool original_dispatch_ptr = false;
    bool original_private_segment = false;
  };
  const std::array cases = {
      Case{"none", {false, false, false}},
      Case{"only_x", {true, false, false}},
      Case{"only_y", {false, true, false}},
      Case{"only_z", {false, false, true}},
      Case{"x_y", {true, true, false}},
      Case{"x_z_private", {true, false, true}, false, true},
      Case{"y_z", {false, true, true}},
      Case{"x_y_z", {true, true, true}},
      Case{"existing_dispatch_only_z", {false, false, true}, true, false},
  };

  for (const Case &test_case : cases) {
    SCOPED_TRACE(test_case.name);
    std::array<uint32_t, 220> text_words{};
    text_words[0] = 0xD81A0000u;
    text_words[1] = 0x00000000u; // ds_write_b32 v0, v0
    for (size_t i = 2; i + 1 < text_words.size(); ++i)
      text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4);
    text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);
    std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
        text_words, std::string("gfx950_omitted_coordinates_") + test_case.name,
        /*vgpr_granulated=*/3, /*wave32=*/false, /*uses_dynamic_stack=*/false,
        EF_AMDGPU_MACH_AMDGCN_GFX950);
    mutate_first_kernel_descriptor(bytes, [&](KD &descriptor) {
      AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 5u);
      AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2,
                      kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X,
                      test_case.original_workgroup_ids[0]);
      AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2,
                      kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y,
                      test_case.original_workgroup_ids[1]);
      AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2,
                      kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z,
                      test_case.original_workgroup_ids[2]);
      AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                      kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR,
                      test_case.original_dispatch_ptr);
      AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT,
                      test_case.original_private_segment);
      AMDHSA_BITS_SET(descriptor.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH, 1u);
    });

    ConSanOptions options;
    options.flavor = ConSanFlavor::Moi;
    options.scratch_vgpr = 8;
    options.moi_report_buffer_address = 0x123456780000ull;
    options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);
    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
    ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
    const auto prologue = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
      return patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
    });
    ASSERT_NE(prologue, result.patches.end());
    ASSERT_GT(prologue->trampoline_size, 256u);
    for (const auto vgpr : result.resolved_moi_workgroup_vgprs)
      ASSERT_TRUE(vgpr);

    AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(patched.is_valid());
    const char *text = patched.text_sections().front()->data();
    const size_t body_size = prologue->trampoline_size - 256u;
    EXPECT_EQ(std::memcmp(text + prologue->trampoline_offset,
                          text + prologue->trampoline_offset + 256u, body_size),
              0);
    std::vector<uint32_t> words(body_size / sizeof(uint32_t));
    std::memcpy(words.data(), text + prologue->trampoline_offset, body_size);
    const uint16_t patched_user_count = test_case.original_dispatch_ptr ? 5u : 7u;
    const std::vector<uint32_t> snapshots = {
        build_v_mov_b32_e32(*result.resolved_moi_workgroup_vgprs[0], patched_user_count,
                            ROCJITSU_CODE_ARCH_CDNA4),
        build_v_mov_b32_e32(*result.resolved_moi_workgroup_vgprs[1], patched_user_count + 1u,
                            ROCJITSU_CODE_ARCH_CDNA4),
        build_v_mov_b32_e32(*result.resolved_moi_workgroup_vgprs[2], patched_user_count + 2u,
                            ROCJITSU_CODE_ARCH_CDNA4),
    };
    EXPECT_TRUE(contains_subsequence(words, snapshots));

    std::vector<uint32_t> restore;
    if (!test_case.original_dispatch_ptr) {
      for (uint16_t destination = 0; destination < 5; ++destination) {
        restore.push_back(build_s_mov_b32(destination, static_cast<uint16_t>(destination + 2u),
                                          ROCJITSU_CODE_ARCH_CDNA4));
      }
    }
    uint16_t original_workgroup_destination = 5;
    for (uint16_t dimension = 0; dimension < 3; ++dimension) {
      if (test_case.original_workgroup_ids[dimension]) {
        restore.push_back(build_s_mov_b32(original_workgroup_destination++,
                                          static_cast<uint16_t>(patched_user_count + dimension),
                                          ROCJITSU_CODE_ARCH_CDNA4));
      }
    }
    // WORKGROUP_INFO follows all three coordinates in the patched firmware ABI.
    restore.push_back(build_s_mov_b32(original_workgroup_destination++,
                                      static_cast<uint16_t>(patched_user_count + 3u),
                                      ROCJITSU_CODE_ARCH_CDNA4));
    if (test_case.original_private_segment) {
      restore.push_back(build_s_mov_b32(original_workgroup_destination,
                                        static_cast<uint16_t>(patched_user_count + 4u),
                                        ROCJITSU_CODE_ARCH_CDNA4));
    }
    EXPECT_TRUE(contains_subsequence(words, restore));

    ASSERT_EQ(patched.kernels().size(), 1u);
    KD descriptor{};
    std::memcpy(&descriptor,
                result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
                sizeof(descriptor));
    EXPECT_EQ(AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2,
                              kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X),
              1u);
    EXPECT_EQ(AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2,
                              kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y),
              1u);
    EXPECT_EQ(AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2,
                              kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z),
              1u);
  }
}

TEST(ConSanMoi, Gfx950ScalarIdentityProloguePreservesPairedEntryAbi) {
  std::array<uint32_t, 220> text_words{};
  text_words[0] = 0xD81A0000u;
  text_words[1] = 0x00000000u; // ds_write_b32 v0, v0
  for (size_t i = 2; i + 1 < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);
  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "gfx950_scalar_preload", /*vgpr_granulated=*/3, /*wave32=*/false,
      /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 5u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y,
                    0u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z,
                    0u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 3u);
    AMDHSA_BITS_SET(descriptor.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH, 1u);
  });
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.scratch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.moi_scalar_identity_automatic);
  ASSERT_TRUE(result.resolved_moi_state_owner_sgpr);
  ASSERT_TRUE(result.resolved_moi_state_epoch_sgpr);
  const auto prologue = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
  });
  ASSERT_NE(prologue, result.patches.end());
  ASSERT_GT(prologue->trampoline_size, 256u);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const char *text = patched.text_sections().front()->data();
  const size_t body_size = prologue->trampoline_size - 256u;
  EXPECT_EQ(std::memcmp(text + prologue->trampoline_offset,
                        text + prologue->trampoline_offset + 256u, body_size),
            0);
  std::vector<uint32_t> words(body_size / sizeof(uint32_t));
  std::memcpy(words.data(), text + prologue->trampoline_offset, body_size);
  const std::vector<uint32_t> snapshots = {
      build_s_mov_b32(*result.resolved_moi_workgroup_sgprs[0], /*shifted s5=*/7,
                      ROCJITSU_CODE_ARCH_CDNA4),
      build_s_mov_b32(*result.resolved_moi_workgroup_sgprs[1], /*shifted s6=*/8,
                      ROCJITSU_CODE_ARCH_CDNA4),
      build_s_mov_b32(*result.resolved_moi_workgroup_sgprs[2], /*shifted s7=*/9,
                      ROCJITSU_CODE_ARCH_CDNA4),
  };
  EXPECT_TRUE(contains_subsequence(words, snapshots));
  const auto read_owner = build_v_readfirstlane_b32(
      *result.resolved_moi_state_owner_sgpr, *prologue->scratch_vgpr, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(read_owner);
  EXPECT_NE(std::ranges::find(words, *read_owner), words.end());
  EXPECT_NE(std::ranges::find(words, build_s_mov_b32(*result.resolved_moi_state_epoch_sgpr,
                                                     scalar_positive_inline_u32(0),
                                                     ROCJITSU_CODE_ARCH_CDNA4)),
            words.end());
  std::vector<uint32_t> restore;
  for (uint16_t destination = 0; destination < 5; ++destination) {
    restore.push_back(build_s_mov_b32(destination, static_cast<uint16_t>(destination + 2u),
                                      ROCJITSU_CODE_ARCH_CDNA4));
  }
  restore.push_back(
      build_s_mov_b32(/*original workgroup x=*/5, /*patched x=*/7, ROCJITSU_CODE_ARCH_CDNA4));
  restore.push_back(build_s_mov_b32(/*original workgroup info=*/6, /*patched info=*/10,
                                    ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_TRUE(contains_subsequence(words, restore));
}

TEST(ConSanMoi, Gfx950StableOwnerTransactionEnablesDispatchPtrWithoutWorkgroupInfo) {
  const std::array<uint32_t, 3> text_words = {
      0xD81A0000u,
      0x00000000u, // ds_write_b32 v0, v0
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "gfx950_missing_owner", /*vgpr_granulated=*/3, /*wave32=*/false,
      /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO,
                    0u);
  });
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.scratch_vgpr = 8;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2,
                            kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO),
            0u);
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.kernel_code_properties,
                            kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR),
            1u);
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT),
            2u);
}

TEST(ConSanMoi, Gfx950OwnerEpochProloguePreservesKernargPreloadEntryPair) {
  std::array<uint32_t, 220> text_words{};
  text_words[0] = 0xD81A0000u;
  text_words[1] = 0x00000000u; // ds_write_b32 v0, v0
  for (size_t i = 2; i + 1 < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);

  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "gfx950_kernarg_preload", /*vgpr_granulated=*/3, /*wave32=*/false,
      /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH, 1u);
  });

  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.scratch_vgpr = 8;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto prologue_patch =
      std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
        return patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
      });
  ASSERT_NE(prologue_patch, result.patches.end());
  ASSERT_GT(prologue_patch->trampoline_size, 256u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const char *text = patched.text_sections().front()->data();
  const uint64_t first_entry = prologue_patch->trampoline_offset;
  const uint64_t preload_entry = first_entry + 256u;
  const size_t prologue_size = prologue_patch->trampoline_size - 256u;
  EXPECT_EQ(std::memcmp(text + first_entry, text + preload_entry, prologue_size), 0)
      << "the compatibility and kernarg-preloaded firmware entries must execute equivalent "
         "identity prologues";

  ASSERT_EQ(patched.kernels().size(), 1u);
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  const uint64_t descriptor_vaddr = patched.kernel_descriptor_offset("gfx950_kernarg_preload");
  ASSERT_NE(descriptor_vaddr, 0u);
  EXPECT_EQ(static_cast<int64_t>(descriptor_vaddr) + descriptor.kernel_code_entry_byte_offset,
            static_cast<int64_t>(patched.text_sections().front()->vaddr() + first_entry));
}

TEST(ConSanMoi, Gfx950AccumOffsetBoundarySelectsScalarOwnerConservatively) {
  const std::array<uint32_t, 3> text_words = {
      0xD81A0000u,
      0x00000000u, // ds_write_b32 v0, v0
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  struct Case {
    uint32_t encoded_accum_offset;
    bool expect_scalar;
  };
  // Explicit scratch v12:v18 makes the persistent identity start at v19.
  // ACCUM base v24 is the exact non-overlap boundary; v20 overlaps it.
  for (const Case test_case : std::array{Case{0u, false}, Case{5u, false}, Case{4u, true}}) {
    std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
        text_words, "gfx950_accum_boundary", /*vgpr_granulated=*/3, /*wave32=*/false,
        /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
    mutate_first_kernel_descriptor(bytes, [&](KD &descriptor) {
      AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET,
                      test_case.encoded_accum_offset);
    });
    ConSanOptions options;
    options.flavor = ConSanFlavor::Moi;
    options.moi_engine = ConSanMoiEngine::InlineShadow;
    options.scratch_vgpr = 12;
    options.moi_report_buffer_address = 0x123456780000ull;
    options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
    ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
    EXPECT_FALSE(result.moi_private_epoch_automatic);
    EXPECT_EQ(result.moi_scalar_identity_automatic, test_case.expect_scalar);
    EXPECT_EQ(result.moi_persistent_vgprs_automatic, !test_case.expect_scalar);
    const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
      return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
    });
    ASSERT_NE(access, result.patches.end());
    EXPECT_FALSE(access->persistent_epoch_private_offset);
    EXPECT_FALSE(access->persistent_owner_private_offset);
  }
}

TEST(ConSanMoi, Gfx950AccumFallbackUsesGloballyUnreferencedInAllocationWindow) {
  const std::array<uint32_t, 3> text_words = {
      0xD81A0000u,
      0x00000000u, // ds_write_b32 v0, v0
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "gfx950_scalar_window_full", /*vgpr_granulated=*/3, /*wave32=*/false,
      /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 4u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 12u);
  });
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::InlineShadow;
  options.scratch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  EXPECT_TRUE(result.modified);
  EXPECT_TRUE(result.moi_scalar_identity_automatic);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("globally unreferenced in-allocation five-SGPR identity window") !=
           std::string::npos;
  }));
}

TEST(ConSanMoi, Gfx950AccumFallbackRejectsUnselectedSitesWithoutGlobalScalarProof) {
  const std::array<uint32_t, 5> text_words = {
      0xD81A0000u,
      0x00000000u, // ds_write_b32 v0, v0
      0xD81A0000u,
      0x00000000u, // ds_write_b32 v0, v0
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "gfx950_scalar_window_unselected", /*vgpr_granulated=*/3,
      /*wave32=*/false, /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 1u);
  });
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::RecordReplay;
  options.scratch_vgpr = 4;
  options.max_patches = 1;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size =
      sizeof(ConSanMoiReportHeader) + 4u * sizeof(ConSanMoiAccessRecord);

  const ConSanResult result = try_patch_consan(bytes, options);

  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(std::ranges::any_of(result.errors, [](const std::string &error) {
    return error.find("cannot prove a five-SGPR identity window across unselected access sites") !=
           std::string::npos;
  }));
}

TEST(ConSanMoi, Gfx950ForcedPrivateIdentityRejectsMissingFreshFiveSgprWindow) {
  const std::array<uint32_t, 3> text_words = {
      0xD81A0000u,
      0x00000000u, // ds_write_b32 v0, v0
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "gfx950_private_window_full", /*vgpr_granulated=*/3, /*wave32=*/false,
      /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 12u);
  });
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.force_private_epoch = true;
  options.scratch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_FALSE(result.errors.empty());
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(std::ranges::any_of(result.errors, [](const std::string &error) {
    return error.find("fresh five-SGPR private workgroup identity window") != std::string::npos;
  }));
}

TEST(ConSanMoi, Gfx950AccumFallbackPreservesHwIdOwnerSemanticsWithPrivateState) {
  const std::array<uint32_t, 3> text_words = {
      0xD81A0000u,
      0x00000000u, // ds_write_b32 v0, v0
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "gfx950_hw_id_accum", /*vgpr_granulated=*/3, /*wave32=*/false,
      /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 4u);
  });
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::InlineShadow;
  options.moi_owner_source = ConSanMoiOwnerSource::HwId;
  options.scratch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_FALSE(result.errors.empty());
  EXPECT_TRUE(result.moi_private_epoch_automatic);
  EXPECT_FALSE(result.moi_scalar_identity_automatic);
  EXPECT_TRUE(std::ranges::any_of(result.errors, [](const std::string &error) {
    return error.find("HW_ID") != std::string::npos;
  }));
}

TEST(ConSanMoi, Gfx950AccumOverlapUsesScalarStateForRecordReplayAndSampled) {
  const std::array<uint32_t, 3> text_words = {
      0xD81A0000u,
      0x00000000u, // ds_write_b32 v0, v0
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  struct Case {
    ConSanMoiEngine engine;
    uint32_t runtime_stride;
  };
  for (const Case test_case :
       std::array{Case{ConSanMoiEngine::RecordReplay, 1u}, Case{ConSanMoiEngine::Sampled, 1u},
                  Case{ConSanMoiEngine::Sampled, 2u}}) {
    std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
        text_words, "gfx950_private_record", /*vgpr_granulated=*/3, /*wave32=*/false,
        /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
    mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
      AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 4u);
    });
    ConSanOptions options;
    options.flavor = ConSanFlavor::Moi;
    options.moi_engine = test_case.engine;
    options.scratch_vgpr = 15;
    options.moi_runtime_sample_stride = test_case.runtime_stride;
    if (test_case.engine == ConSanMoiEngine::Sampled) {
      options.moi_workgroup_extent_x = 2;
      options.moi_workgroup_extent_y = 2;
      options.moi_workgroup_extent_z = 2;
    }
    options.moi_report_buffer_address = 0x123456780000ull;
    options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
    ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
    EXPECT_TRUE(result.moi_scalar_identity_automatic);
    EXPECT_FALSE(result.moi_private_epoch_automatic);
    EXPECT_FALSE(result.moi_persistent_vgprs_automatic);
    const auto access = std::ranges::find_if(result.patches, [&](const ConSanPatchInfo &patch) {
      if (test_case.engine == ConSanMoiEngine::RecordReplay)
        return patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
      return patch.kind == ConSanPatchKind::TrampolineMoiSampledWatchpointStore;
    });
    ASSERT_NE(access, result.patches.end());
    EXPECT_FALSE(access->persistent_epoch_private_offset);
    EXPECT_FALSE(access->persistent_owner_private_offset);
    const auto prologue = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
      return patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
    });
    ASSERT_NE(prologue, result.patches.end());
    ASSERT_TRUE(prologue->scratch_vgpr);
    EXPECT_GT(*prologue->scratch_vgpr, 0u)
        << "the packed workitem-ID ABI input v0 cannot be an entry temporary";

    EXPECT_TRUE(result.resolved_moi_state_owner_sgpr);
    EXPECT_TRUE(result.resolved_moi_state_epoch_sgpr);
    ASSERT_FALSE(result.resource_plans.empty());
    EXPECT_GE(*result.resolved_moi_state_owner_sgpr,
              result.resource_plans.front().current_sgpr_count)
        << "persistent scalar state must grow above the guest SGPR allocation";
  }
}

TEST(ConSanMoi, Gfx950RecordReplayPrivateStateKeepsSpillsDisjoint) {
  const std::array<uint32_t, 4> text_words = {
      build_v_mov_b32_e32(/*vdst=*/255, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_CDNA4),
      0xD81A0000u,
      0x00000000u, // ds_write_b32 v0, v0
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "gfx950_private_record_spill", /*vgpr_granulated=*/31, /*wave32=*/false,
      /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 63u);
  });
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::RecordReplay;
  options.force_private_epoch = true;
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
  });
  ASSERT_NE(access, result.patches.end());
  EXPECT_EQ(access->persistent_epoch_private_offset, 0u);
  EXPECT_EQ(access->persistent_owner_private_offset, 4u);
  EXPECT_EQ(access->spilled_vgpr_count, 3u);
  EXPECT_GE(access->required_private_segment_size, 28u);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  EXPECT_GE(descriptor.private_segment_fixed_size, 28u);
}

TEST(ConSanMoi, Gfx950PrivateOwnerProloguePreservesKernargPreloadEntryPair) {
  std::array<uint32_t, 220> text_words{};
  text_words[0] = 0xD81A0000u;
  text_words[1] = 0x00000000u; // ds_write_b32 v0, v0
  for (size_t i = 2; i + 1 < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);
  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "gfx950_private_kernarg_preload", /*vgpr_granulated=*/3, /*wave32=*/false,
      /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH, 1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 4u);
  });
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::InlineShadow;
  options.force_private_epoch = true;
  options.scratch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto prologue = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue;
  });
  ASSERT_NE(prologue, result.patches.end());
  EXPECT_EQ(prologue->persistent_epoch_private_offset, 0u);
  EXPECT_EQ(prologue->persistent_owner_private_offset, 4u);
  EXPECT_EQ(prologue->spilled_vgpr_count, 0u);
  EXPECT_EQ(prologue->required_private_segment_size, 16u);
  ASSERT_GT(prologue->trampoline_size, 256u);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const char *text = patched.text_sections().front()->data();
  const size_t prologue_size = prologue->trampoline_size - 256u;
  EXPECT_EQ(std::memcmp(text + prologue->trampoline_offset,
                        text + prologue->trampoline_offset + 256u, prologue_size),
            0);
  ASSERT_TRUE(result.resolved_moi_identity_sgpr);
  for (const auto sgpr : result.resolved_moi_workgroup_sgprs)
    ASSERT_TRUE(sgpr);
  std::vector<uint32_t> prologue_words(prologue_size / sizeof(uint32_t));
  std::memcpy(prologue_words.data(), text + prologue->trampoline_offset, prologue_size);
  const std::vector<uint32_t> workgroup_snapshots = {
      build_s_mov_b32(*result.resolved_moi_workgroup_sgprs[0], 2, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_mov_b32(*result.resolved_moi_workgroup_sgprs[1], 3, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_mov_b32(*result.resolved_moi_workgroup_sgprs[2], 4, ROCJITSU_CODE_ARCH_CDNA4),
  };
  EXPECT_TRUE(contains_subsequence(prologue_words, workgroup_snapshots));
  EXPECT_NE(std::ranges::find(prologue_words, build_s_mov_b32(/*original workgroup info=*/0,
                                                              /*patched workgroup info=*/5,
                                                              ROCJITSU_CODE_ARCH_CDNA4)),
            prologue_words.end());
  const auto unpack_z = build_v_lshrrev_b32_e32(
      /*vdst=*/12, scalar_positive_inline_u32(20), /*packed_workitem_id=*/0,
      ROCJITSU_CODE_ARCH_CDNA4);
  const auto flatten_yz = build_v_mad_u32_u24(
      /*vdst=*/12, static_cast<uint16_t>(*result.resolved_moi_identity_sgpr + 1u),
      /*z=*/12, /*y=*/13, ROCJITSU_CODE_ARCH_CDNA4);
  const auto flatten_x = build_v_mad_u32_u24(
      /*vdst=*/12, *result.resolved_moi_identity_sgpr, /*yz=*/12, /*x=*/13,
      ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(unpack_z);
  ASSERT_TRUE(flatten_yz);
  ASSERT_TRUE(flatten_x);
  EXPECT_TRUE(contains_subsequence(prologue_words, std::span<const uint32_t>(&*unpack_z, 1)));
  EXPECT_TRUE(contains_subsequence(prologue_words, *flatten_yz));
  EXPECT_TRUE(contains_subsequence(prologue_words, *flatten_x));

  const auto owner_store = build_cdna4_address_free_scratch_store_b32(
      /*vsrc=*/12, /*owner_offset=*/4u, ROCJITSU_CODE_ARCH_CDNA4);
  const auto epoch_store = build_cdna4_address_free_scratch_store_b32(
      /*vsrc=*/12, /*epoch_offset=*/0u, ROCJITSU_CODE_ARCH_CDNA4);
  const auto stale_entry_load0 = build_cdna4_address_free_scratch_load_b32(
      /*vdst=*/12, /*old_spill_offset=*/16u, ROCJITSU_CODE_ARCH_CDNA4);
  const auto stale_entry_load1 = build_cdna4_address_free_scratch_load_b32(
      /*vdst=*/13, /*old_spill_offset=*/20u, ROCJITSU_CODE_ARCH_CDNA4);
  const auto stale_entry_store0 = build_cdna4_address_free_scratch_store_b32(
      /*vsrc=*/12, /*old_spill_offset=*/16u, ROCJITSU_CODE_ARCH_CDNA4);
  const auto stale_entry_store1 = build_cdna4_address_free_scratch_store_b32(
      /*vsrc=*/13, /*old_spill_offset=*/20u, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(owner_store && epoch_store && stale_entry_load0 && stale_entry_load1 &&
              stale_entry_store0 && stale_entry_store1);
  EXPECT_TRUE(contains_subsequence(prologue_words, *owner_store));
  EXPECT_TRUE(contains_subsequence(prologue_words, *epoch_store));
  EXPECT_FALSE(contains_subsequence(prologue_words, *stale_entry_load0));
  EXPECT_FALSE(contains_subsequence(prologue_words, *stale_entry_load1));
  EXPECT_FALSE(contains_subsequence(prologue_words, *stale_entry_store0));
  EXPECT_FALSE(contains_subsequence(prologue_words, *stale_entry_store1));
}

TEST(ConSanMoi, Gfx950StableOwnerTransactionPreservesExistingDispatchPtrAbi) {
  const std::array<uint32_t, 3> text_words = {
      0xD81A0000u,
      0x00000000u, // ds_write_b32 v0, v0
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "gfx950_existing_dispatch", /*vgpr_granulated=*/3, /*wave32=*/false,
      /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                    kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR, 1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 5u);
  });
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.scratch_vgpr = 8;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.kernel_code_properties,
                            kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR),
            1u);
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT),
            5u);
}

TEST(ConSanMoi, Gfx950StableOwnerTransactionRejectsSaturatedUserSgprAbi) {
  const std::array<uint32_t, 3> text_words = {
      0xD81A0000u,
      0x00000000u, // ds_write_b32 v0, v0
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "gfx950_saturated_user_sgprs", /*vgpr_granulated=*/3, /*wave32=*/false,
      /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 30u);
  });
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.scratch_vgpr = 8;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_FALSE(result.errors.empty());
  EXPECT_TRUE(std::ranges::any_of(result.errors, [](const std::string &error) {
    return error.find("cannot insert a dispatch pointer into the user-SGPR ABI") !=
           std::string::npos;
  })) << testing::PrintToString(result.errors);
}

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
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.scratch_vgpr = 20;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.moi_candidates.size(), 1u);
  EXPECT_EQ(result.moi_candidates.front().kind, expected_kind);
  EXPECT_EQ(result.moi_candidates.front().width_bits, expected_width_bits);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineMoiAccessRecordStore);
  EXPECT_EQ(result.patches.front().original_size, 142u * sizeof(uint32_t));
}

TEST(ConSanMoi, FirstLightProbeSupportsMultiWidthNativeLdsSites) {
  {
    SCOPED_TRACE("ds_load_u8");
    expect_moi_first_light_width(0xD8E80000u, 0x01000009u, 8u, ConSanLdsAccessKind::Read);
  }
  {
    SCOPED_TRACE("ds_load_u8_d16");
    expect_moi_first_light_width(0xDA880000u, 0x01000009u, 8u, ConSanLdsAccessKind::Read);
  }
  {
    SCOPED_TRACE("ds_load_u16");
    expect_moi_first_light_width(0xD8F00000u, 0x01000009u, 16u, ConSanLdsAccessKind::Read);
  }
  {
    SCOPED_TRACE("ds_load_b64");
    expect_moi_first_light_width(0xD9D80000u, 0x01000009u, 64u, ConSanLdsAccessKind::Read);
  }
  {
    SCOPED_TRACE("ds_store_b8");
    expect_moi_first_light_width(0xD8780000u, 0x00000109u, 8u, ConSanLdsAccessKind::Write);
  }
  {
    SCOPED_TRACE("ds_store_b16");
    expect_moi_first_light_width(0xD87C0000u, 0x00000109u, 16u, ConSanLdsAccessKind::Write);
  }
  {
    SCOPED_TRACE("ds_store_b128");
    expect_moi_first_light_width(0xDB7C0000u, 0x00000109u, 128u, ConSanLdsAccessKind::Write);
  }
  {
    SCOPED_TRACE("ds_load_u16_d16");
    expect_moi_first_light_width(0xDA980000u, 0x01000002u, 16u, ConSanLdsAccessKind::Read);
  }
}

TEST(ConSanMoi, InventorySkipsUnsupportedNativeLdsSites) {
  const std::vector<uint8_t> bytes = make_rdna4_unsupported_lds_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::RecordReplay;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_EQ(result.moi_candidates.size(), 2u);
  EXPECT_EQ(result.moi_candidates[0].mnemonic, "ds_store_b32");
  EXPECT_EQ(result.moi_candidates[1].mnemonic, "ds_load_b32");
  bool saw_skipped_lds_warning = false;
  for (const std::string &warning : result.warnings) {
    saw_skipped_lds_warning |= warning.find("skipped native LDS sites") != std::string::npos &&
                               warning.find("unsupported_kind=1") != std::string::npos &&
                               warning.find("unsupported_mnemonic=0") != std::string::npos;
  }
  EXPECT_TRUE(saw_skipped_lds_warning);
}

TEST(ConSanMoi, FirstLightProbeAddsNativeLdsImmediateOffset) {
  std::array<uint32_t, 180> text_words{};
  text_words[0] = 0xDA980480u;
  text_words[1] = 0x01000002u; // ds_load_u16_d16 v1, v2 offset:1152
  for (size_t i = 2; i + 1 < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.scratch_vgpr = 20;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().original_size, 146u * sizeof(uint32_t));

  std::vector<uint32_t> rewritten_words(result.patches.front().original_size / sizeof(uint32_t));
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + 0x100,
              rewritten_words.size() * sizeof(uint32_t));

  const auto mov_offset = build_v_mov_b32_e64_literal(22, 1152u, ROCJITSU_CODE_ARCH_RDNA4);
  const auto add_offset =
      build_v_add_nc_u32_e32(22, vector_source_vgpr(2), 22, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(mov_offset);
  ASSERT_TRUE(add_offset);
  std::vector<uint32_t> expected_offset = {mov_offset->at(0), mov_offset->at(1), mov_offset->at(2),
                                           *add_offset};
  EXPECT_TRUE(contains_subsequence(rewritten_words, expected_offset));

  const uint64_t access_record_base =
      *options.moi_report_buffer_address + sizeof(ConSanMoiReportHeader);
  EXPECT_TRUE(contains_subsequence(
      rewritten_words, make_expected_vgpr_store_words(
                           access_record_base + offsetof(ConSanMoiAccessRecord, lds_byte_offset),
                           /*value_vgpr=*/22, *options.scratch_vgpr)));
  EXPECT_TRUE(contains_subsequence(
      rewritten_words, make_expected_vgpr_store_words(
                           access_record_base + offsetof(ConSanMoiAccessRecord, start_cell),
                           /*value_vgpr=*/22, *options.scratch_vgpr)));
}

TEST(ConSanMoi, FirstLightProbeLowersTwoAddressNativeLdsSitesToTwoRecords) {
  std::vector<uint32_t> text_words(420, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words[0] = 0xD8380201u;
  text_words[1] = 0x00000000u; // ds_store_2addr_b32 v0, v0, v0 offset0:1 offset1:2
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.scratch_vgpr = 20;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(2, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_EQ(result.moi_candidates.size(), 1u);
  EXPECT_EQ(result.moi_candidates.front().mnemonic, "ds_store_2addr_b32");
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);

  std::vector<uint32_t> rewritten_words(result.patches.front().original_size / sizeof(uint32_t));
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + 0x100,
              rewritten_words.size() * sizeof(uint32_t));

  const uint64_t base = *options.moi_report_buffer_address;
  const uint64_t first_record_base = base + sizeof(ConSanMoiReportHeader);
  const uint64_t second_record_base = first_record_base + sizeof(ConSanMoiAccessRecord);
  EXPECT_TRUE(contains_subsequence(
      rewritten_words, make_expected_vgpr_store_words(
                           first_record_base + offsetof(ConSanMoiAccessRecord, lds_byte_offset),
                           /*value_vgpr=*/22, *options.scratch_vgpr)));
  EXPECT_TRUE(contains_subsequence(
      rewritten_words, make_expected_vgpr_store_words(
                           second_record_base + offsetof(ConSanMoiAccessRecord, lds_byte_offset),
                           /*value_vgpr=*/22, *options.scratch_vgpr)));

  const auto mov_offset0 = build_v_mov_b32_e64_literal(22, 4u, ROCJITSU_CODE_ARCH_RDNA4);
  const auto mov_offset1 = build_v_mov_b32_e64_literal(22, 8u, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(mov_offset0);
  ASSERT_TRUE(mov_offset1);
  EXPECT_TRUE(contains_subsequence(rewritten_words, *mov_offset0));
  EXPECT_TRUE(contains_subsequence(rewritten_words, *mov_offset1));
}

TEST(ConSanMoi, DynamicAccessRecordProbeLowersTwoAddressNativeLdsSites) {
  std::vector<uint32_t> text_words(760, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words[0] = 0xD8380201u;
  text_words[1] = 0x00000000u; // ds_store_2addr_b32 v0, v0, v0 offset0:1 offset1:2
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.scratch_vgpr = 20;
  options.moi_exec_save_sgpr = 30;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_dynamic_access_records = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(8, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineMoiAccessRecordStore);
}

TEST(ConSanMoi, DynamicAccessRecordProbeSkipsImmediateSaveexecRegion) {
  std::vector<uint32_t> text_words(360, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  const auto save_exec =
      build_s_and_saveexec_b64(/*sdst=*/4, /*ssrc0=*/8, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(save_exec);
  text_words[2] = *save_exec;
  text_words[3] = build_v_mov_b32_e32(/*vdst=*/1, /*src=*/scalar_positive_inline_u32(0),
                                      ROCJITSU_CODE_ARCH_RDNA4);
  text_words[4] = 0xD8340020u;
  text_words[5] = 0x00000901u; // ds_store_b32 v1, v9 offset:32
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.scratch_vgpr = 20;
  options.moi_exec_save_sgpr = 30;
  options.moi_dynamic_access_records = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(8, 0, 0, 0);
  options.max_patches = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  bool saw_saveexec_warning = false;
  for (const std::string &warning : result.warnings)
    saw_saveexec_warning |= warning.find("immediately after s_*_saveexec") != std::string::npos;
  EXPECT_TRUE(saw_saveexec_warning);
}

TEST(ConSanMoi, FirstLightProbeCanPatchTwoNativeLdsAccessRecords) {
  constexpr uint32_t kSecondSiteWord = 170;
  std::vector<uint32_t> text_words(360, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  text_words[kSecondSiteWord] = 0xD8D80000u;
  text_words[kSecondSiteWord + 1] = 0x01000000u; // ds_load_b32 v1, v0
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.scratch_vgpr = 20;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(2, 0, 0, 0);
  options.max_patches = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 2u);
  EXPECT_EQ(result.patches[0].kind, ConSanPatchKind::InlineMoiAccessRecordStore);
  EXPECT_EQ(result.patches[0].anchor_offset, 0u);
  EXPECT_EQ(result.patches[0].original_size, 142u * sizeof(uint32_t));
  EXPECT_EQ(result.patches[1].kind, ConSanPatchKind::InlineMoiAccessRecordStore);
  EXPECT_EQ(result.patches[1].anchor_offset, kSecondSiteWord * sizeof(uint32_t));
  EXPECT_EQ(result.patches[1].original_size, 142u * sizeof(uint32_t));

  std::vector<uint32_t> first_words(142);
  std::vector<uint32_t> second_words(142);
  std::memcpy(first_words.data(), result.elf_bytes.data() + 0x100,
              first_words.size() * sizeof(uint32_t));
  std::memcpy(second_words.data(),
              result.elf_bytes.data() + 0x100 +
                  static_cast<uint64_t>(kSecondSiteWord) * sizeof(uint32_t),
              second_words.size() * sizeof(uint32_t));
  EXPECT_EQ(first_words[0], 0xD8340000u);
  EXPECT_EQ(first_words[1], 0x00000000u);
  EXPECT_EQ(first_words[2], 0xBFC60000u);
  EXPECT_EQ(second_words[0], 0xD8D80000u);
  EXPECT_EQ(second_words[1], 0x01000000u);
  EXPECT_EQ(second_words[2], 0xBFC60000u);
}

TEST(ConSanMoi, FirstLightProbeUsesAppendedCaveWhenInlinePaddingIsUnavailable) {
  const std::array<uint32_t, 3> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  const ConSanPatchInfo &patch = result.patches.front();
  EXPECT_EQ(patch.kind, ConSanPatchKind::TrampolineMoiAccessRecordStore);
  EXPECT_EQ(patch.anchor_offset, 0u);
  EXPECT_EQ(patch.trampoline_offset, text_words.size() * sizeof(uint32_t));
  EXPECT_EQ(patch.original_size, 2u * sizeof(uint32_t));
  EXPECT_GT(patch.trampoline_size, 2u * sizeof(uint32_t));

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  EXPECT_EQ(patched.text_sections().front()->size(),
            text_words.size() * sizeof(uint32_t) + patch.trampoline_size);

  std::vector<uint32_t> actual_words(patched.text_sections().front()->size() / sizeof(uint32_t));
  std::memcpy(actual_words.data(), patched.text_sections().front()->data(),
              patched.text_sections().front()->size());
  const auto fwd =
      compute_sopp_branch_simm16(/*branch_pc=*/0, text_words.size() * sizeof(uint32_t));
  ASSERT_TRUE(fwd);
  EXPECT_EQ(actual_words[0], build_s_branch(*fwd, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(actual_words[1], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(actual_words[2], build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4));
  const size_t cave_word_offset = text_words.size();
  EXPECT_EQ(actual_words[cave_word_offset], 0xD8340000u);
  EXPECT_EQ(actual_words[cave_word_offset + 1], 0x00000000u);
  EXPECT_EQ(actual_words[cave_word_offset + 2], 0xBFC60000u);
  const uint64_t return_branch_pc =
      patch.trampoline_offset + patch.trampoline_size - sizeof(uint32_t);
  const auto ret = compute_sopp_branch_simm16(return_branch_pc, 2u * sizeof(uint32_t));
  ASSERT_TRUE(ret);
  EXPECT_EQ(actual_words.back(), build_s_branch(*ret, ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(ConSanMoi, FirstLightProbeCanPatchTwoAppendedCaveAccessRecords) {
  const std::array<uint32_t, 5> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
      0xD8D80000u,
      0x01000000u, // ds_load_b32 v1, v0
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(2, 0, 0, 0);
  options.max_patches = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 2u);
  EXPECT_EQ(result.patches[0].kind, ConSanPatchKind::TrampolineMoiAccessRecordStore);
  EXPECT_EQ(result.patches[0].anchor_offset, 0u);
  EXPECT_EQ(result.patches[0].original_size, 2u * sizeof(uint32_t));
  EXPECT_EQ(result.patches[1].kind, ConSanPatchKind::TrampolineMoiAccessRecordStore);
  EXPECT_EQ(result.patches[1].anchor_offset, 2u * sizeof(uint32_t));
  EXPECT_EQ(result.patches[1].original_size, 2u * sizeof(uint32_t));

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  EXPECT_GT(patched.text_sections().front()->size(), text_words.size() * sizeof(uint32_t));
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

TEST(ConSanMoi, InlineShadowPublishesStronglyClassifiedFlatLdsCell) {
  const std::array<uint32_t, 2> kernel_words = {
      pack_sopk(/*s_call_b64=*/0x14, /*sdst=*/30, /*simm16=*/1),
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint32_t> function_words = make_padded_moi_flat_first_light_function_words();
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::InlineShadow;
  options.flat_provenance_mode = ConSanFlatProvenanceMode::Strict;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 24;
  options.moi_epoch_vgpr = 25;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.moi_candidates.size(), 1u);
  EXPECT_EQ(result.moi_candidates.front().source, ConSanMoiCandidateSource::FlatGroup);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::TrampolineMoiExactShadowStore);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  std::vector<uint32_t> text_words(patched.text_sections().front()->size() / sizeof(uint32_t));
  std::memcpy(text_words.data(), patched.text_sections().front()->data(),
              patched.text_sections().front()->size());
  const auto start_cell_shift = build_v_lshrrev_b32_e32(
      /*vdst=*/14, scalar_positive_inline_u32(consan_moi_exact_shadow::granule_shift),
      /*vsrc=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  const auto atomic_swap = build_flat_atomic_swap_b64_vaddr_vsrc_vdst(
      /*vaddr=*/8, /*vsrc=*/10, /*vdst=*/12, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(start_cell_shift);
  ASSERT_TRUE(atomic_swap);
  EXPECT_TRUE(contains_subsequence(text_words, std::span<const uint32_t>(&*start_cell_shift, 1)));
  EXPECT_EQ(count_subsequence(text_words, *atomic_swap), 1u);
}

TEST(ConSanMoi, FirstLightProbeWritesOneLikelyGroupFlatAccessRecord) {
  const std::array<uint32_t, 2> kernel_words = {
      pack_sopk(/*s_call_b64=*/0x14, /*sdst=*/30, /*simm16=*/1),
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint32_t> function_words = make_padded_moi_flat_first_light_function_words();
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.moi_candidates.size(), 1u);
  EXPECT_EQ(result.moi_candidates.front().source, ConSanMoiCandidateSource::FlatGroup);
  EXPECT_EQ(result.moi_candidates.front().kind, ConSanLdsAccessKind::Read);
  EXPECT_EQ(result.moi_candidates.front().mnemonic, "flat_load_b32");
  EXPECT_EQ(result.moi_candidates.front().text_offset, 28u);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineMoiAccessRecordStore);
  EXPECT_EQ(result.patches.front().anchor_offset, 28u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 40u);
  EXPECT_EQ(result.patches.front().original_size, 143u * sizeof(uint32_t));

  std::array<uint32_t, 4> rewritten_prefix{};
  std::memcpy(rewritten_prefix.data(), result.elf_bytes.data() + 0x11c,
              rewritten_prefix.size() * sizeof(uint32_t));
  EXPECT_EQ(rewritten_prefix[0], 0xEC05007Cu);
  EXPECT_EQ(rewritten_prefix[1], 0x00000002u);
  EXPECT_EQ(rewritten_prefix[2], 0x00000000u);
  EXPECT_EQ(rewritten_prefix[3], 0xBFC60000u);
}

TEST(ConSanMoi, FirstLightProbeRejectsScratchVgprsOverlappingFlatAddressPair) {
  const std::array<uint32_t, 2> kernel_words = {
      pack_sopk(/*s_call_b64=*/0x14, /*sdst=*/30, /*simm16=*/1),
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint32_t> function_words = make_padded_moi_flat_first_light_function_words();
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.scratch_vgpr = 1;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  EXPECT_TRUE(result.errors.empty());
  EXPECT_FALSE(result.modified);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().reason, ConSanRegisterPlanReason::ForbiddenOverlap);
}

TEST(ConSanMoi, OwnerEpochPrologueRedirectsKernelDescriptorEntry) {
  constexpr uint64_t kExpectedPrologueOffset = 256;
  const std::array<uint32_t, 2> text_words = {
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_init_owner_epoch = true;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, kExpectedPrologueOffset);
  EXPECT_EQ(result.patches.front().original_size, 0u);
  EXPECT_EQ(result.patches.front().trampoline_size, 3u * sizeof(uint32_t));

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  ASSERT_EQ(patched.text_sections().size(), 1u);
  EXPECT_EQ(patched.text_sections().front()->size(),
            kExpectedPrologueOffset + 3u * sizeof(uint32_t));

  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  const uint64_t descriptor_vaddr = patched.kernel_descriptor_offset("lds_probe");
  ASSERT_NE(descriptor_vaddr, 0u);
  const int64_t descriptor_entry_vaddr =
      static_cast<int64_t>(descriptor_vaddr) + descriptor.kernel_code_entry_byte_offset;
  const int64_t expected_entry_vaddr =
      static_cast<int64_t>(patched.text_sections().front()->vaddr() + kExpectedPrologueOffset);
  EXPECT_EQ(descriptor_entry_vaddr, expected_entry_vaddr);

  ASSERT_EQ(kExpectedPrologueOffset % sizeof(uint32_t), 0u);
  const auto text_word_count = patched.text_sections().front()->size() / sizeof(uint32_t);
  std::vector<uint32_t> actual_words(text_word_count);
  std::memcpy(actual_words.data(), patched.text_sections().front()->data(),
              actual_words.size() * sizeof(uint32_t));

  ASSERT_GE(actual_words.size(), text_words.size());
  EXPECT_TRUE(std::equal(text_words.begin(), text_words.end(), actual_words.begin()));
  const size_t prologue_word_offset = kExpectedPrologueOffset / sizeof(uint32_t);
  ASSERT_GE(actual_words.size(), prologue_word_offset + 3u);
  for (size_t i = text_words.size(); i < prologue_word_offset; ++i)
    EXPECT_EQ(actual_words[i], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));

  std::vector<uint32_t> expected_prologue_words;
  const auto owner_shift =
      build_v_lshrrev_b32_e32(11, scalar_positive_inline_u32(6), 0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(owner_shift);
  expected_prologue_words.push_back(*owner_shift);
  expected_prologue_words.push_back(
      build_v_mov_b32_e32(12, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4));
  const auto branch =
      compute_sopp_branch_simm16(kExpectedPrologueOffset + 2u * sizeof(uint32_t), 0);
  ASSERT_TRUE(branch);
  expected_prologue_words.push_back(build_s_branch(*branch, ROCJITSU_CODE_ARCH_RDNA4));

  const std::span<const uint32_t> actual_prologue_words(actual_words.data() + prologue_word_offset,
                                                        expected_prologue_words.size());
  EXPECT_TRUE(std::equal(expected_prologue_words.begin(), expected_prologue_words.end(),
                         actual_prologue_words.begin()));
}

TEST(ConSanMoi, OwnerEpochPrologueUsesWave32DescriptorForOwnerShift) {
  constexpr uint64_t kExpectedPrologueOffset = 256;
  const std::array<uint32_t, 2> text_words = {
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "lds_probe", kRdna4Wave64AllVgprsGranulated, /*wave32=*/true);
  AmdGpuCodeObject input(bytes.data(), bytes.size());
  ASSERT_TRUE(input.is_valid());
  ASSERT_EQ(input.kernels().size(), 1u);
  KD input_descriptor{};
  std::memcpy(&input_descriptor, bytes.data() + input.kernels().front().descriptor_file_offset,
              sizeof(input_descriptor));
  ASSERT_NE(AMDHSA_BITS_GET(input_descriptor.kernel_code_properties,
                            kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32),
            0u);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_init_owner_epoch = true;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);

  uint32_t owner_init = 0;
  std::memcpy(&owner_init, patched.text_sections().front()->data() + kExpectedPrologueOffset,
              sizeof(owner_init));
  const auto expected_owner_init =
      build_v_lshrrev_b32_e32(11, scalar_positive_inline_u32(5), 0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(expected_owner_init);
  EXPECT_EQ(owner_init, *expected_owner_init);
}

TEST(ConSanMoi, OwnerEpochPrologueGrowsKernelDescriptorVgprAllocation) {
  const std::array<uint32_t, 2> text_words = {
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };

  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "lds_probe", /*vgpr_granulated=*/0);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_init_owner_epoch = true;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  const uint32_t granulated = AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc1,
                                              kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  EXPECT_EQ(granulated, 3u);
}

TEST(ConSanMoi, OwnerEpochPrologueHwIdOwnerSourceRequiresOwnerSgpr) {
  const std::array<uint32_t, 2> text_words = {
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_init_owner_epoch = true;
  options.moi_owner_source = ConSanMoiOwnerSource::HwId;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;

  const auto result = try_patch_consan(bytes, options);

  EXPECT_FALSE(result.errors.empty());
  EXPECT_FALSE(result.modified);
  bool saw_missing_sgpr = false;
  for (const std::string &error : result.errors)
    saw_missing_sgpr |= error.find("RJ_CONSAN_MOI_OWNER_SGPR") != std::string::npos;
  EXPECT_TRUE(saw_missing_sgpr);
}

TEST(ConSanMoi, OwnerEpochPrologueCanUseHwIdOwnerSource) {
  constexpr uint64_t kExpectedPrologueOffset = 256;
  const std::array<uint32_t, 2> text_words = {
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };

  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "lds_probe", /*vgpr_granulated=*/0);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_init_owner_epoch = true;
  options.moi_owner_source = ConSanMoiOwnerSource::HwId;
  options.moi_owner_sgpr = 20;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue);
  EXPECT_EQ(result.patches.front().trampoline_size, 4u * sizeof(uint32_t));

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  ASSERT_EQ(patched.text_sections().size(), 1u);

  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  const uint32_t vgpr_granulated = AMDHSA_BITS_GET(
      descriptor.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  EXPECT_EQ(vgpr_granulated, 3u);
  const uint32_t sgpr_granulated = AMDHSA_BITS_GET(
      descriptor.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);
  EXPECT_EQ(sgpr_granulated, 2u);

  const auto hwreg = build_hwreg_imm(/*reg_id=*/23, /*offset=*/0, /*size_bits=*/10);
  ASSERT_TRUE(hwreg);
  const auto get_hw_id = build_s_getreg_b32(/*sdst=*/20, *hwreg, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(get_hw_id);
  std::vector<uint32_t> expected_prologue_words;
  expected_prologue_words.push_back(*get_hw_id);
  expected_prologue_words.push_back(build_v_mov_b32_e32(11, /*src0=*/20, ROCJITSU_CODE_ARCH_RDNA4));
  expected_prologue_words.push_back(
      build_v_mov_b32_e32(12, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4));
  const auto branch =
      compute_sopp_branch_simm16(kExpectedPrologueOffset + 3u * sizeof(uint32_t), 0);
  ASSERT_TRUE(branch);
  expected_prologue_words.push_back(build_s_branch(*branch, ROCJITSU_CODE_ARCH_RDNA4));

  ASSERT_EQ(kExpectedPrologueOffset % sizeof(uint32_t), 0u);
  const auto text_word_count = patched.text_sections().front()->size() / sizeof(uint32_t);
  std::vector<uint32_t> actual_words(text_word_count);
  std::memcpy(actual_words.data(), patched.text_sections().front()->data(),
              actual_words.size() * sizeof(uint32_t));
  const size_t prologue_word_offset = kExpectedPrologueOffset / sizeof(uint32_t);
  ASSERT_GE(actual_words.size(), prologue_word_offset + expected_prologue_words.size());
  const std::span<const uint32_t> actual_prologue_words(actual_words.data() + prologue_word_offset,
                                                        expected_prologue_words.size());
  EXPECT_TRUE(std::equal(expected_prologue_words.begin(), expected_prologue_words.end(),
                         actual_prologue_words.begin()));
}

TEST(ConSanMoi, BarrierRecordPatchTrampolinesBarrierAndWritesRecord) {
  constexpr uint32_t kBarrierWait = 0xBF940000u;
  const std::array<uint32_t, 2> text_words = {
      kBarrierWait,
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_track_barriers = true;
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 30;
  options.moi_owner_vgpr = 14;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 1);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels.front().barrier_sites.size(), 1u);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::TrampolineMoiBarrierRecord);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, text_words.size() * sizeof(uint32_t));
  EXPECT_EQ(result.patches.front().original_size, sizeof(uint32_t));

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);

  std::vector<uint32_t> expected_prefix;
  const auto fwd = compute_sopp_branch_simm16(0, text_words.size() * sizeof(uint32_t));
  ASSERT_TRUE(fwd);
  expected_prefix.push_back(build_s_branch(*fwd, ROCJITSU_CODE_ARCH_RDNA4));
  expected_prefix.push_back(text_words[1]);
  std::vector<uint32_t> actual_words(patched.text_sections().front()->size() / sizeof(uint32_t));
  std::memcpy(actual_words.data(), patched.text_sections().front()->data(),
              actual_words.size() * sizeof(uint32_t));
  ASSERT_GE(actual_words.size(), expected_prefix.size());
  EXPECT_TRUE(std::equal(expected_prefix.begin(), expected_prefix.end(), actual_words.begin()));

  const ConSanPatchInfo &patch = result.patches.front();
  std::vector<uint32_t> trampoline_words(patch.trampoline_size / sizeof(uint32_t));
  std::memcpy(trampoline_words.data(),
              patched.text_sections().front()->data() + patch.trampoline_offset,
              trampoline_words.size() * sizeof(uint32_t));

  const auto mbcnt_lo = build_v_mbcnt_lo_u32_b32(
      /*vdst=*/13, /*src0=*/0xC1, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  const auto mbcnt_hi = build_v_mbcnt_hi_u32_b32(
      /*vdst=*/13, /*src0=*/0xC1, vector_source_vgpr(13), ROCJITSU_CODE_ARCH_RDNA4);
  const auto first_active_lane = build_v_cmp_eq_u32_e32_vcc(scalar_positive_inline_u32(0),
                                                            /*vsrc1=*/13, ROCJITSU_CODE_ARCH_RDNA4);
  const auto save_exec =
      build_s_and_saveexec_b64(/*sdst=*/30, /*ssrc0=*/kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_exec = build_s_mov_b64(/*sdst=*/126, /*ssrc0=*/30, ROCJITSU_CODE_ARCH_RDNA4);
  const auto save_scc = build_s_cselect_b32(
      /*sdst=*/34, scalar_positive_inline_u32(1), scalar_positive_inline_u32(0),
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto save_vcc = build_s_mov_b64(/*sdst=*/32, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_vcc = build_s_mov_b64(kRdna4VccLo, /*ssrc0=*/32, ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_scc = build_s_cmp_lg_u32(
      /*ssrc0=*/34, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  const auto skip_overflow = build_s_cbranch_vccz(/*offset_dwords=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(mbcnt_lo);
  ASSERT_TRUE(mbcnt_hi);
  ASSERT_TRUE(first_active_lane);
  ASSERT_TRUE(save_exec);
  ASSERT_TRUE(restore_exec);
  ASSERT_TRUE(save_scc);
  ASSERT_TRUE(save_vcc);
  ASSERT_TRUE(restore_vcc);
  ASSERT_TRUE(restore_scc);
  ASSERT_TRUE(skip_overflow);
  EXPECT_TRUE(contains_subsequence(trampoline_words, *mbcnt_lo));
  EXPECT_TRUE(contains_subsequence(trampoline_words, *mbcnt_hi));
  EXPECT_TRUE(contains_subsequence(trampoline_words,
                                   std::array<uint32_t, 2>{*first_active_lane, *save_exec}));
  EXPECT_TRUE(std::find(trampoline_words.begin(), trampoline_words.end(),
                        build_v_mov_b32_e32(/*vdst=*/13, /*src0=*/30, ROCJITSU_CODE_ARCH_RDNA4)) !=
              trampoline_words.end());
  EXPECT_TRUE(std::find(trampoline_words.begin(), trampoline_words.end(),
                        build_v_mov_b32_e32(/*vdst=*/13, /*src0=*/31, ROCJITSU_CODE_ARCH_RDNA4)) !=
              trampoline_words.end());
  EXPECT_TRUE(std::find(trampoline_words.begin(), trampoline_words.end(), *restore_exec) !=
              trampoline_words.end());
  EXPECT_TRUE(
      contains_subsequence(trampoline_words, std::array<uint32_t, 2>{*save_scc, *save_vcc}));
  EXPECT_TRUE(contains_subsequence(
      trampoline_words, std::array<uint32_t, 3>{*restore_exec, *restore_vcc, *restore_scc}));
  EXPECT_TRUE(std::find(trampoline_words.begin(), trampoline_words.end(), kBarrierWait) !=
              trampoline_words.end());
  EXPECT_TRUE(std::any_of(trampoline_words.begin(), trampoline_words.end(),
                          [](uint32_t word) { return (word & 0xFFFF0000u) == 0xBFA30000u; }));

  const uint64_t base = *options.moi_report_buffer_address;
  const auto mov_barrier_count_lo = build_v_mov_b32_e64_literal(
      8, static_cast<uint32_t>(base + offsetof(ConSanMoiReportHeader, barrier_record_count)),
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(mov_barrier_count_lo);
  EXPECT_TRUE(contains_subsequence(trampoline_words, *mov_barrier_count_lo));
  EXPECT_GT(result.patches.front().trampoline_size, 0u);
}

TEST(ConSanMoi, Gfx950BarrierOnlyScalarStateGetsEntryPrologue) {
  const auto barrier_sequence = build_cdna4_s_barrier_with_memory_wait(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(barrier_sequence);
  const std::array<uint32_t, 3> text_words = {
      (*barrier_sequence)[0],
      (*barrier_sequence)[1],
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "gfx950_barrier_only", /*vgpr_granulated=*/3, /*wave32=*/false,
      /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 3u);
  });
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_track_barriers = true;
  options.scratch_vgpr = 6;
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 1);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  EXPECT_TRUE(result.moi_scalar_identity_automatic) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(std::ranges::any_of(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiBarrierRecord;
  }));
  EXPECT_TRUE(std::ranges::any_of(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
  }));
}

TEST(ConSanMoi, BarrierRecordPatchStoresDescriptorWorkgroupIds) {
  constexpr uint32_t kBarrierWait = 0xBF940000u;
  const std::array<uint32_t, 2> text_words = {
      kBarrierWait,
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };

  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 5u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z,
                    1u);
  });

  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_track_barriers = true;
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 30;
  options.moi_owner_vgpr = 14;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 1);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::TrampolineMoiBarrierRecord);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);

  const ConSanPatchInfo &patch = result.patches.front();
  std::vector<uint32_t> trampoline_words(patch.trampoline_size / sizeof(uint32_t));
  std::memcpy(trampoline_words.data(),
              patched.text_sections().front()->data() + patch.trampoline_offset,
              trampoline_words.size() * sizeof(uint32_t));

  const std::vector<uint32_t> expected_x = {
      build_v_mov_b32_e32(/*vdst=*/13, ttmp_scalar_operand(kTtmpRdna4GridX),
                          ROCJITSU_CODE_ARCH_RDNA4),
  };
  const auto y_shift_left = build_v_lshlrev_b32_e32(
      /*vdst=*/13, scalar_positive_inline_u32(16), /*vsrc1=*/13, ROCJITSU_CODE_ARCH_RDNA4);
  const auto y_shift_right = build_v_lshrrev_b32_e32(
      /*vdst=*/13, scalar_positive_inline_u32(16), /*vsrc1=*/13, ROCJITSU_CODE_ARCH_RDNA4);
  const auto z_shift = build_v_lshrrev_b32_e32(
      /*vdst=*/13, scalar_positive_inline_u32(16), /*vsrc1=*/13, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(y_shift_left);
  ASSERT_TRUE(y_shift_right);
  ASSERT_TRUE(z_shift);
  const std::vector<uint32_t> expected_y = {
      build_v_mov_b32_e32(/*vdst=*/13, ttmp_scalar_operand(kTtmpRdna4GridYz),
                          ROCJITSU_CODE_ARCH_RDNA4),
      *y_shift_left,
      *y_shift_right,
  };
  const std::vector<uint32_t> expected_z = {
      build_v_mov_b32_e32(/*vdst=*/13, ttmp_scalar_operand(kTtmpRdna4GridYz),
                          ROCJITSU_CODE_ARCH_RDNA4),
      *z_shift,
  };
  EXPECT_TRUE(contains_subsequence(trampoline_words, expected_x));
  EXPECT_TRUE(contains_subsequence(trampoline_words, expected_y));
  EXPECT_TRUE(contains_subsequence(trampoline_words, expected_z));
}

TEST(ConSanMoi, BarrierRecordAutomaticallyPlansScratchAndScalarState) {
  constexpr uint32_t kBarrierWait = 0xBF940000u;
  const std::array<uint32_t, 2> text_words = {
      kBarrierWait,
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_track_barriers = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 1);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.moi_exec_save_sgprs_automatic);
  const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &item) {
    return item.kind == ConSanPatchKind::TrampolineMoiBarrierRecord;
  });
  ASSERT_NE(patch, result.patches.end());
  ASSERT_TRUE(patch->scratch_vgpr);
  EXPECT_GT(*patch->scratch_vgpr, 0u);
  const auto plan = std::ranges::find_if(result.resource_plans, [](const auto &item) {
    return item.site_kind == ConSanResourceSiteKind::Barrier;
  });
  ASSERT_NE(plan, result.resource_plans.end());
  EXPECT_EQ(plan->scratch_vgpr, patch->scratch_vgpr);
  EXPECT_EQ(result.resource_plan_summary.dead_plans, 1u);
}

TEST(ConSanMoi, BarrierRecordForcedSpillUsesPlannedPrivateWindow) {
  constexpr uint32_t kBarrierWait = 0xBF940000u;
  const std::array<uint32_t, 2> text_words = {
      kBarrierWait,
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_track_barriers = true;
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 1);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &item) {
    return item.kind == ConSanPatchKind::TrampolineMoiBarrierRecord;
  });
  ASSERT_NE(patch, result.patches.end());
  EXPECT_EQ(patch->spilled_vgpr_count, 6u);
  EXPECT_GT(patch->required_private_segment_size, 0u);
  EXPECT_EQ(result.resource_plan_summary.spill_plans, 1u);
  EXPECT_EQ(result.resource_plan_summary.emitted_spill_patches, 1u);
  EXPECT_EQ(result.resource_plan_summary.emitted_spill_slot_bytes, 24u);
}

TEST(ConSanMoi, InlineShadowBarrierEpochPatchTrampolinesBarrierAndIncrementsEpoch) {
  constexpr uint32_t kBarrierWait = 0xBF940000u;
  const std::array<uint32_t, 4> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32
      kBarrierWait,
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::InlineShadow;
  options.moi_track_barriers = true;
  options.moi_init_owner_epoch = true;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 20;
  options.moi_epoch_vgpr = 21;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  const auto epoch_patch_it =
      std::find_if(result.patches.begin(), result.patches.end(), [](const ConSanPatchInfo &patch) {
        return patch.kind == ConSanPatchKind::TrampolineMoiInlineEpochBarrier;
      });
  ASSERT_NE(epoch_patch_it, result.patches.end());
  EXPECT_EQ(epoch_patch_it->anchor_offset, 2u * sizeof(uint32_t));
  EXPECT_EQ(epoch_patch_it->original_size, sizeof(uint32_t));
  EXPECT_EQ(epoch_patch_it->trampoline_size, 3u * sizeof(uint32_t));

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const auto *text_section = patched.text_sections().front();
  ASSERT_GE(text_section->size(),
            epoch_patch_it->trampoline_offset + epoch_patch_it->trampoline_size);

  uint32_t rewritten_barrier = 0;
  std::memcpy(&rewritten_barrier, text_section->data() + epoch_patch_it->anchor_offset,
              sizeof(rewritten_barrier));
  const auto fwd =
      compute_sopp_branch_simm16(epoch_patch_it->anchor_offset, epoch_patch_it->trampoline_offset);
  ASSERT_TRUE(fwd);
  EXPECT_EQ(rewritten_barrier, build_s_branch(*fwd, ROCJITSU_CODE_ARCH_RDNA4));

  std::array<uint32_t, 3> trampoline_words{};
  std::memcpy(trampoline_words.data(), text_section->data() + epoch_patch_it->trampoline_offset,
              epoch_patch_it->trampoline_size);
  const auto increment_epoch = build_v_add_nc_u32_e32(
      /*vdst=*/21, scalar_positive_inline_u32(1), /*vsrc1=*/21, ROCJITSU_CODE_ARCH_RDNA4);
  const auto ret =
      compute_sopp_branch_simm16(epoch_patch_it->trampoline_offset + 2u * sizeof(uint32_t),
                                 epoch_patch_it->anchor_offset + sizeof(uint32_t));
  ASSERT_TRUE(increment_epoch);
  ASSERT_TRUE(ret);
  EXPECT_EQ(trampoline_words[0], kBarrierWait);
  EXPECT_EQ(trampoline_words[1], *increment_epoch);
  EXPECT_EQ(trampoline_words[2], build_s_branch(*ret, ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(ConSanMoi, Gfx950ScalarBarrierPreservesSpecialState) {
  const auto barrier_sequence = build_cdna4_s_barrier_with_memory_wait(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(barrier_sequence);
  const std::array<uint32_t, 5> text_words = {
      0xD81A0000u,
      0x00000000u, // ds_write_b32
      (*barrier_sequence)[0],
      (*barrier_sequence)[1],
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "gfx950_scalar_barrier", /*vgpr_granulated=*/3, /*wave32=*/false,
      /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 3u);
  });
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::InlineShadow;
  options.moi_track_barriers = true;
  options.scratch_vgpr = 8;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.moi_scalar_identity_automatic);
  ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
  ASSERT_TRUE(result.resolved_moi_state_owner_sgpr);
  ASSERT_TRUE(result.resolved_moi_state_epoch_sgpr);
  ASSERT_TRUE(result.resolved_moi_workgroup_sgprs[2]);
  EXPECT_LT(static_cast<uint32_t>(*result.resolved_moi_workgroup_sgprs[2]),
            static_cast<uint32_t>(*result.resolved_moi_exec_save_sgpr));
  const auto barrier = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiInlineEpochBarrier;
  });
  ASSERT_NE(barrier, result.patches.end());
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  std::vector<uint32_t> words(barrier->trampoline_size / sizeof(uint32_t));
  std::memcpy(words.data(), patched.text_sections().front()->data() + barrier->trampoline_offset,
              barrier->trampoline_size);
  const uint16_t save_base = *result.resolved_moi_exec_save_sgpr;
  const uint16_t vcc_save = static_cast<uint16_t>(save_base + 8u);
  const uint16_t scc_save = static_cast<uint16_t>(save_base + 10u);
  const auto save_scc =
      build_s_cselect_b32(scc_save, scalar_positive_inline_u32(1), scalar_positive_inline_u32(0),
                          ROCJITSU_CODE_ARCH_CDNA4);
  const auto save_vcc = build_s_mov_b64(vcc_save, kRdna4VccLo, ROCJITSU_CODE_ARCH_CDNA4);
  const auto restore_vcc = build_s_mov_b64(kRdna4VccLo, vcc_save, ROCJITSU_CODE_ARCH_CDNA4);
  const auto restore_scc =
      build_s_cmp_lg_u32(scc_save, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(save_scc && save_vcc && restore_vcc && restore_scc);
  const std::vector<uint32_t> expected = {
      *save_scc,
      *save_vcc,
      pack_sop2(/*s_add_u32=*/0, *result.resolved_moi_state_epoch_sgpr,
                *result.resolved_moi_state_epoch_sgpr, scalar_positive_inline_u32(1)),
      *restore_vcc,
      *restore_scc,
  };
  EXPECT_TRUE(contains_subsequence(words, expected));
}

TEST(ConSanMoi, AtomicRecordPatchTrampolinesFlatAtomicAndWritesRecord) {
  const std::vector<uint8_t> bytes = make_rdna4_flat_atomic_code_object();
  ASSERT_FALSE(bytes.empty());
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_track_atomics = true;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 0, 1);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels.front().atomic_sites.size(), 1u);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::TrampolineMoiAtomicRecord);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().original_size, 3u * sizeof(uint32_t));

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);

  const ConSanPatchInfo &patch = result.patches.front();
  std::vector<uint32_t> anchor_words(patch.original_size / sizeof(uint32_t));
  std::memcpy(anchor_words.data(), patched.text_sections().front()->data() + patch.anchor_offset,
              anchor_words.size() * sizeof(uint32_t));
  EXPECT_EQ(anchor_words[0] >> 23u, kSoppEncodingPrefix);
  EXPECT_EQ(anchor_words[1], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(anchor_words[2], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));

  std::vector<uint32_t> trampoline_words(patch.trampoline_size / sizeof(uint32_t));
  std::memcpy(trampoline_words.data(),
              patched.text_sections().front()->data() + patch.trampoline_offset,
              trampoline_words.size() * sizeof(uint32_t));

  const auto original_atomic = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/2, /*vsrc=*/1, /*vdst=*/0, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(original_atomic);
  ASSERT_GE(trampoline_words.size(), original_atomic->size() + 1u);
  EXPECT_TRUE(
      std::equal(original_atomic->begin(), original_atomic->end(), trampoline_words.begin()));
  EXPECT_EQ(trampoline_words[original_atomic->size()], 0xBFC00000u);

  const uint64_t base = *options.moi_report_buffer_address;
  const ConSanMoiReportBufferLayout layout = consan_moi_report_buffer_layout_for_bytes(
      options.moi_report_buffer_size, /*include_barriers=*/false, /*include_atomics=*/true);
  const uint64_t atomic_record_base = base + layout.atomic_records_offset;
  EXPECT_TRUE(contains_subsequence(
      trampoline_words,
      make_expected_literal_store_words(base + offsetof(ConSanMoiReportHeader, atomic_record_count),
                                        1u, *options.scratch_vgpr)));
  EXPECT_TRUE(contains_subsequence(
      trampoline_words,
      make_expected_vgpr_store_words(atomic_record_base + offsetof(ConSanMoiAtomicRecord, owner_id),
                                     *options.moi_owner_vgpr, *options.scratch_vgpr)));
  EXPECT_TRUE(contains_subsequence(
      trampoline_words,
      make_expected_vgpr_store_words(atomic_record_base + offsetof(ConSanMoiAtomicRecord, epoch),
                                     *options.moi_epoch_vgpr, *options.scratch_vgpr)));
  EXPECT_TRUE(contains_subsequence(
      trampoline_words, make_expected_vgpr_store_words(
                            atomic_record_base + offsetof(ConSanMoiAtomicRecord, atomic_address),
                            /*value_vgpr=*/2, *options.scratch_vgpr)));
  EXPECT_TRUE(contains_subsequence(
      trampoline_words,
      make_expected_vgpr_store_words(
          atomic_record_base + offsetof(ConSanMoiAtomicRecord, atomic_address) + sizeof(uint32_t),
          /*value_vgpr=*/3, *options.scratch_vgpr)));
  EXPECT_TRUE(contains_subsequence(
      trampoline_words,
      make_expected_literal_store_words(atomic_record_base + offsetof(ConSanMoiAtomicRecord, kind),
                                        static_cast<uint32_t>(ConSanMoiAtomicEventKind::Acquire),
                                        *options.scratch_vgpr)));
  EXPECT_TRUE(contains_subsequence(
      trampoline_words,
      make_expected_literal_store_words(atomic_record_base + offsetof(ConSanMoiAtomicRecord, scope),
                                        2u, *options.scratch_vgpr)));
}

TEST(ConSanMoi, Gfx950AtomicRecordOnlyScalarStateGetsEntryPrologue) {
  std::vector<uint8_t> bytes = make_gfx950_flat_atomic_release_acquire_code_object();
  ASSERT_FALSE(bytes.empty());
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 3u);
  });
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_track_atomics = true;
  options.scratch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(2, 0, 0, 0, 0, 2);
  options.max_patches = 2;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  EXPECT_TRUE(result.moi_scalar_identity_automatic);
  EXPECT_TRUE(std::ranges::any_of(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiAtomicRecord;
  }));
  EXPECT_TRUE(std::ranges::any_of(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
  }));
}

TEST(ConSanMoi, Gfx950InventoriesFlatAtomicScOrderingFields) {
  const std::vector<uint8_t> bytes = make_gfx950_flat_atomic_release_acquire_code_object();
  ASSERT_FALSE(bytes.empty());
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_EQ(result.kernels.size(), 1u);
  const ConSanKernelInfo &kernel = result.kernels.front();
  ASSERT_EQ(kernel.atomic_sites.size(), 2u);
  const ConSanAtomicSite &release = kernel.atomic_sites[0];
  const ConSanAtomicSite &acquire = kernel.atomic_sites[1];
  EXPECT_EQ(release.mnemonic, "flat_atomic_add");
  EXPECT_EQ(acquire.mnemonic, "flat_atomic_add");
  EXPECT_EQ(release.size, 2u * sizeof(uint32_t));
  EXPECT_EQ(acquire.size, 2u * sizeof(uint32_t));
  ASSERT_TRUE(release.addr_vgpr && acquire.addr_vgpr);
  EXPECT_EQ(*release.addr_vgpr, 2u);
  EXPECT_EQ(*acquire.addr_vgpr, 4u);
  ASSERT_TRUE(release.raw_saddr && acquire.raw_saddr);
  EXPECT_EQ(*release.raw_saddr, 0u);
  EXPECT_EQ(*acquire.raw_saddr, 0u);
  ASSERT_TRUE(release.raw_segment && acquire.raw_segment);
  EXPECT_EQ(*release.raw_segment, 0u);
  EXPECT_EQ(*acquire.raw_segment, 0u);
  ASSERT_TRUE(release.raw_ioffset && acquire.raw_ioffset);
  EXPECT_EQ(*release.raw_ioffset, 0);
  EXPECT_EQ(*acquire.raw_ioffset, 0);
  ASSERT_TRUE(release.raw_scope && acquire.raw_scope);
  EXPECT_EQ(*release.raw_scope, 0u);
  EXPECT_EQ(*acquire.raw_scope, 0u);
  ASSERT_TRUE(release.raw_th && acquire.raw_th);
  EXPECT_EQ(*release.raw_th, 0u);
  EXPECT_EQ(*acquire.raw_th, 1u);
  ASSERT_TRUE(release.returns_old_value && acquire.returns_old_value);
  EXPECT_FALSE(*release.returns_old_value);
  EXPECT_TRUE(*acquire.returns_old_value);
}

TEST(ConSanMoi, Gfx950InventoriesFlatLoadStoreRawFieldsWithoutAdmittingUnknownPointers) {
  const auto load =
      build_flat_load_b32_vaddr_vdst(/*vaddr=*/2, /*vdst=*/4, ROCJITSU_CODE_ARCH_CDNA4);
  const auto store =
      build_flat_store_b32_vaddr_vsrc(/*vaddr=*/2, /*vsrc=*/5, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(load && store);
  const std::array<uint32_t, 5> text_words = {
      (*load)[0], (*load)[1], (*store)[0], (*store)[1], build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "gfx950_flat_unknown", /*vgpr_granulated=*/0, /*wave32=*/false,
      /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_EQ(result.kernels.size(), 1u);
  const ConSanKernelInfo &kernel = result.kernels.front();
  ASSERT_EQ(kernel.flat_sites.size(), 2u);
  EXPECT_EQ(kernel.stats.flat_unknown_hint_count, 2u);
  EXPECT_TRUE(result.moi_candidates.empty());
  for (const ConSanFlatSite &site : kernel.flat_sites) {
    EXPECT_EQ(site.size, 2u * sizeof(uint32_t));
    EXPECT_EQ(site.raw_vaddr, 2u);
    EXPECT_EQ(site.raw_ioffset, 0);
    EXPECT_EQ(site.raw_segment, 0u);
    EXPECT_EQ(site.raw_scope, 0u);
    EXPECT_TRUE(site.raw_op.has_value());
    EXPECT_TRUE(site.raw_saddr.has_value());
    EXPECT_TRUE(site.raw_th.has_value());
  }
  EXPECT_EQ(kernel.flat_sites[0].raw_vdst, 4u);
  EXPECT_EQ(kernel.flat_sites[1].raw_vsrc, 5u);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("unknown=2") != std::string::npos;
  }));
}

TEST(ConSanMoi, Gfx950ClassifiesExplicitSharedBaseFlatLoadAsGroup) {
  const auto load =
      build_flat_load_b32_vaddr_vdst(/*vaddr=*/0, /*vdst=*/2, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(load);
  const std::array<uint32_t, 6> text_words = {
      0xBE8001EBu, // s_mov_b64 s[0:1], src_shared_base
      build_v_mov_b32_e32(/*vdst=*/0, /*scalar s0=*/0, ROCJITSU_CODE_ARCH_CDNA4),
      build_v_mov_b32_e32(/*vdst=*/1, /*scalar s1=*/1, ROCJITSU_CODE_ARCH_CDNA4),
      (*load)[0],
      (*load)[1],
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "gfx950_flat_group", /*vgpr_granulated=*/0, /*wave32=*/false,
      /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(decoder);
  std::unique_ptr<Instruction> base_move(decoder->decode(text_words.data()));
  ASSERT_TRUE(base_move);
  ASSERT_NE(base_move->src_operand(0), nullptr);
  EXPECT_EQ(base_move->src_operand(0)->name(), "SRC_SHARED_BASE");
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_EQ(result.kernels.size(), 1u);
  const ConSanKernelInfo &kernel = result.kernels.front();
  ASSERT_EQ(kernel.flat_sites.size(), 1u);
  EXPECT_EQ(kernel.flat_sites.front().address_space_hint, ConSanFlatAddressSpaceHint::Group);
  EXPECT_EQ(kernel.stats.flat_group_hint_count, 1u);
  ASSERT_EQ(result.moi_candidates.size(), 1u);
  EXPECT_EQ(result.moi_candidates.front().source, ConSanMoiCandidateSource::FlatGroup);
  EXPECT_EQ(result.moi_candidates.front().raw_segment, 0u);
}

std::vector<uint8_t> make_gfx950_padded_group_flat_code_object(uint32_t raw_segment = 0u) {
  const auto load =
      build_flat_load_b32_vaddr_vdst(/*vaddr=*/0, /*vdst=*/2, ROCJITSU_CODE_ARCH_CDNA4);
  if (!load)
    return {};
  std::vector<uint32_t> text_words = {
      0xBE8001EBu, // s_mov_b64 s[0:1], src_shared_base
      build_v_mov_b32_e32(/*vdst=*/0, /*scalar s0=*/0, ROCJITSU_CODE_ARCH_CDNA4),
      build_v_mov_b32_e32(/*vdst=*/1, /*scalar s1=*/1, ROCJITSU_CODE_ARCH_CDNA4),
      (*load)[0],
      (*load)[1],
  };
  text_words[3] = (text_words[3] & ~(3u << 14u)) | ((raw_segment & 3u) << 14u);
  text_words.resize(180, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);
  return make_rdna4_lds_code_object(text_words, "gfx950_flat_group_emission",
                                    /*vgpr_granulated=*/0, /*wave32=*/false,
                                    /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
}

TEST(ConSanMoi, Gfx950RecordReplayEmitsStronglyClassifiedGroupFlatAccess) {
  const std::vector<uint8_t> bytes = make_gfx950_padded_group_flat_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_EQ(result.moi_candidates.size(), 1u);
  EXPECT_EQ(result.moi_candidates.front().source, ConSanMoiCandidateSource::FlatGroup);
  EXPECT_EQ(result.moi_candidates.front().size, 2u * sizeof(uint32_t));
  EXPECT_EQ(result.moi_candidates.front().mnemonic, "flat_load_dword");
  EXPECT_EQ(result.moi_candidates.front().raw_segment, 0u);
  EXPECT_EQ(result.moi_candidates.front().raw_ioffset, 0);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineMoiAccessRecordStore);
  EXPECT_GE(result.patches.front().original_size, 2u * sizeof(uint32_t));
  EXPECT_EQ(result.patches.front().original_size % sizeof(uint32_t), 0u);
}

TEST(ConSanMoi, Gfx950InlineShadowEmitsStronglyClassifiedGroupFlatAccess) {
  const std::vector<uint8_t> bytes = make_gfx950_padded_group_flat_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::InlineShadow;
  options.flat_provenance_mode = ConSanFlatProvenanceMode::Strict;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 24;
  options.moi_epoch_vgpr = 25;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_EQ(result.moi_candidates.size(), 1u);
  EXPECT_EQ(result.moi_candidates.front().source, ConSanMoiCandidateSource::FlatGroup);
  EXPECT_EQ(result.moi_candidates.front().size, 2u * sizeof(uint32_t));
  EXPECT_EQ(result.moi_candidates.front().mnemonic, "flat_load_dword");
  EXPECT_EQ(result.moi_candidates.front().raw_segment, 0u);
  EXPECT_EQ(result.moi_candidates.front().raw_ioffset, 0);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::TrampolineMoiExactShadowStore);
  EXPECT_GE(result.patches.front().original_size, 2u * sizeof(uint32_t));
  EXPECT_EQ(result.patches.front().original_size % sizeof(uint32_t), 0u);
}

TEST(ConSanMoi, Gfx950GroupFlatEmissionExcludesTypedPrivateAndGlobalSegments) {
  for (const uint32_t raw_segment : {1u, 2u}) {
    const std::vector<uint8_t> bytes = make_gfx950_padded_group_flat_code_object(raw_segment);
    ConSanOptions options;
    options.flavor = ConSanFlavor::Moi;
    options.scratch_vgpr = 8;
    options.moi_report_buffer_address = 0x123456780000ull;
    options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
    ASSERT_EQ(result.kernels.size(), 1u);
    // SEG=1 and SEG=2 decode as their typed scratch/global instruction
    // families, not as generic FLAT candidates.
    EXPECT_TRUE(result.kernels.front().flat_sites.empty());
    EXPECT_TRUE(result.moi_candidates.empty());
    EXPECT_FALSE(result.modified);
  }
}

TEST(ConSanMoi, Gfx950AtomicRecordPatchEmitsReleaseAndAcquireRecords) {
  const std::vector<uint8_t> bytes = make_gfx950_flat_atomic_release_acquire_code_object();
  ASSERT_FALSE(bytes.empty());
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_track_atomics = true;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(2, 0, 0, 0, 0, 2);
  options.max_patches = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels.front().atomic_sites.size(), 2u);
  ASSERT_EQ(std::ranges::count_if(result.patches,
                                  [](const ConSanPatchInfo &patch) {
                                    return patch.kind == ConSanPatchKind::TrampolineMoiAtomicRecord;
                                  }),
            2u);
  const auto release_patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiAtomicRecord && patch.anchor_offset == 0u;
  });
  const auto acquire_patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiAtomicRecord &&
           patch.anchor_offset == 2u * sizeof(uint32_t);
  });
  ASSERT_NE(release_patch, result.patches.end());
  ASSERT_NE(acquire_patch, result.patches.end());
  EXPECT_EQ(release_patch->original_size, 2u * sizeof(uint32_t));
  EXPECT_EQ(acquire_patch->original_size, 2u * sizeof(uint32_t));
}

TEST(ConSanMoi, AtomicRecordAutomaticallyPlansScratch) {
  const std::vector<uint8_t> bytes = make_rdna4_flat_atomic_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_track_atomics = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 0, 1);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &item) {
    return item.kind == ConSanPatchKind::TrampolineMoiAtomicRecord;
  });
  ASSERT_NE(patch, result.patches.end());
  ASSERT_TRUE(patch->scratch_vgpr);
  EXPECT_GE(*patch->scratch_vgpr, 4u);
  const auto plan = std::ranges::find_if(result.resource_plans, [](const auto &item) {
    return item.site_kind == ConSanResourceSiteKind::Atomic;
  });
  ASSERT_NE(plan, result.resource_plans.end());
  EXPECT_EQ(plan->scratch_vgpr, patch->scratch_vgpr);
  EXPECT_EQ(result.resource_plan_summary.dead_plans, 1u);
}

TEST(ConSanMoi, AtomicRecordForcedSpillUsesPlannedPrivateWindow) {
  const std::vector<uint8_t> bytes = make_rdna4_flat_atomic_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_track_atomics = true;
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 0, 1);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &item) {
    return item.kind == ConSanPatchKind::TrampolineMoiAtomicRecord;
  });
  ASSERT_NE(patch, result.patches.end());
  EXPECT_EQ(patch->spilled_vgpr_count, 3u);
  EXPECT_GT(patch->required_private_segment_size, 0u);
  EXPECT_EQ(result.resource_plan_summary.spill_plans, 1u);
  EXPECT_EQ(result.resource_plan_summary.emitted_spill_patches, 1u);
  EXPECT_EQ(result.resource_plan_summary.emitted_spill_slot_bytes, 12u);
}

TEST(ConSanMoi, InlineAtomicOrderingPatchPublishesAndImportsReleaseSlot) {
  const std::vector<uint8_t> bytes = make_rdna4_flat_atomic_release_acquire_code_object();
  ASSERT_FALSE(bytes.empty());
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::InlineShadow;
  options.moi_track_atomics = true;
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 13;
  options.moi_epoch_vgpr = 14;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels.front().atomic_sites.size(), 2u);
  EXPECT_EQ(std::count_if(result.patches.begin(), result.patches.end(),
                          [](const ConSanPatchInfo &patch) {
                            return patch.kind == ConSanPatchKind::TrampolineMoiAtomicRecord;
                          }),
            0);
  ASSERT_EQ(std::count_if(result.patches.begin(), result.patches.end(),
                          [](const ConSanPatchInfo &patch) {
                            return patch.kind == ConSanPatchKind::TrampolineMoiInlineAtomicOrdering;
                          }),
            2);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const auto *text_section = patched.text_sections().front();

  const auto release_patch_it =
      std::find_if(result.patches.begin(), result.patches.end(), [](const ConSanPatchInfo &patch) {
        return patch.kind == ConSanPatchKind::TrampolineMoiInlineAtomicOrdering &&
               patch.anchor_offset == 0u;
      });
  const auto acquire_patch_it =
      std::find_if(result.patches.begin(), result.patches.end(), [](const ConSanPatchInfo &patch) {
        return patch.kind == ConSanPatchKind::TrampolineMoiInlineAtomicOrdering &&
               patch.anchor_offset == 3u * sizeof(uint32_t);
      });
  ASSERT_NE(release_patch_it, result.patches.end());
  ASSERT_NE(acquire_patch_it, result.patches.end());

  std::vector<uint32_t> release_words(release_patch_it->trampoline_size / sizeof(uint32_t));
  std::memcpy(release_words.data(), text_section->data() + release_patch_it->trampoline_offset,
              release_patch_it->trampoline_size);
  std::vector<uint32_t> acquire_words(acquire_patch_it->trampoline_size / sizeof(uint32_t));
  std::memcpy(acquire_words.data(), text_section->data() + acquire_patch_it->trampoline_offset,
              acquire_patch_it->trampoline_size);

  const auto original_release = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/2, /*vsrc=*/1, /*vdst=*/0, /*return_old_value=*/false, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto original_acquire = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/4, /*vsrc=*/1, /*vdst=*/0, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(original_release);
  ASSERT_TRUE(original_acquire);
  ASSERT_GE(release_words.size(), original_release->size() + 1u);
  ASSERT_GE(acquire_words.size(), original_acquire->size() + 1u);
  EXPECT_TRUE(
      std::equal(original_release->begin(), original_release->end(), release_words.begin()));
  EXPECT_TRUE(
      std::equal(original_acquire->begin(), original_acquire->end(), acquire_words.begin()));
  EXPECT_EQ(release_words[original_release->size()], 0xBFC00000u);
  EXPECT_EQ(acquire_words[original_acquire->size()], 0xBFC00000u);

  const ConSanMoiReportBufferLayout layout =
      consan_moi_inline_shadow_report_buffer_layout_for_bytes(options.moi_report_buffer_size);
  const uint64_t slot_base =
      *options.moi_report_buffer_address + layout.inline_atomic_release_slots_offset;
  EXPECT_TRUE(contains_subsequence(
      release_words, make_expected_vgpr_store_words(
                         slot_base + offsetof(ConSanMoiInlineAtomicReleaseSlot, owner_id),
                         *options.moi_owner_vgpr, *options.scratch_vgpr)));
  EXPECT_TRUE(contains_subsequence(
      release_words,
      make_expected_vgpr_store_words(slot_base + offsetof(ConSanMoiInlineAtomicReleaseSlot, epoch),
                                     *options.moi_epoch_vgpr, *options.scratch_vgpr)));
  EXPECT_TRUE(contains_subsequence(
      release_words, make_expected_vgpr_store_words(
                         slot_base + offsetof(ConSanMoiInlineAtomicReleaseSlot, atomic_address),
                         /*value_vgpr=*/2, *options.scratch_vgpr)));
  EXPECT_TRUE(contains_subsequence(
      release_words,
      make_expected_vgpr_store_words(
          slot_base + offsetof(ConSanMoiInlineAtomicReleaseSlot, atomic_address) + sizeof(uint32_t),
          /*value_vgpr=*/3, *options.scratch_vgpr)));
  EXPECT_TRUE(contains_subsequence(
      release_words, make_expected_literal_store_words(
                         slot_base + offsetof(ConSanMoiInlineAtomicReleaseSlot, valid), 1u,
                         *options.scratch_vgpr)));

  EXPECT_TRUE(contains_subsequence(
      acquire_words,
      make_expected_vgpr_load_words(slot_base + offsetof(ConSanMoiInlineAtomicReleaseSlot, valid),
                                    /*value_vgpr=*/10, *options.scratch_vgpr)));
  EXPECT_TRUE(contains_subsequence(
      acquire_words, make_expected_vgpr_load_words(
                         slot_base + offsetof(ConSanMoiInlineAtomicReleaseSlot, atomic_address),
                         /*value_vgpr=*/10, *options.scratch_vgpr)));
  EXPECT_TRUE(contains_subsequence(
      acquire_words,
      make_expected_vgpr_load_words(
          slot_base + offsetof(ConSanMoiInlineAtomicReleaseSlot, atomic_address) + sizeof(uint32_t),
          /*value_vgpr=*/10, *options.scratch_vgpr)));
  EXPECT_TRUE(contains_subsequence(
      acquire_words, make_expected_vgpr_load_words(
                         slot_base + offsetof(ConSanMoiInlineAtomicReleaseSlot, owner_id),
                         /*value_vgpr=*/10, *options.scratch_vgpr)));
  EXPECT_TRUE(contains_subsequence(
      acquire_words,
      make_expected_vgpr_load_words(slot_base + offsetof(ConSanMoiInlineAtomicReleaseSlot, epoch),
                                    /*value_vgpr=*/10, *options.scratch_vgpr)));

  const auto import_epoch =
      build_v_add_nc_u32_e32(*options.moi_epoch_vgpr, scalar_positive_inline_u32(1), /*vsrc1=*/10,
                             ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_exec =
      build_s_mov_b64(kRdna4ExecLo, *options.moi_exec_save_sgpr, ROCJITSU_CODE_ARCH_RDNA4);
  const auto save_scc = build_s_cselect_b32(
      /*sdst=*/static_cast<uint16_t>(*options.moi_exec_save_sgpr + 10u),
      scalar_positive_inline_u32(1), scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  const auto save_vcc = build_s_mov_b64(
      /*sdst=*/static_cast<uint16_t>(*options.moi_exec_save_sgpr + 8u), kRdna4VccLo,
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_vcc = build_s_mov_b64(
      kRdna4VccLo, /*ssrc0=*/static_cast<uint16_t>(*options.moi_exec_save_sgpr + 8u),
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_scc = build_s_cmp_lg_u32(
      /*ssrc0=*/static_cast<uint16_t>(*options.moi_exec_save_sgpr + 10u),
      scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(import_epoch);
  ASSERT_TRUE(restore_exec);
  ASSERT_TRUE(save_scc);
  ASSERT_TRUE(save_vcc);
  ASSERT_TRUE(restore_vcc);
  ASSERT_TRUE(restore_scc);
  EXPECT_TRUE(std::find(acquire_words.begin(), acquire_words.end(), *import_epoch) !=
              acquire_words.end());
  EXPECT_TRUE(std::find(acquire_words.begin(), acquire_words.end(), *restore_exec) !=
              acquire_words.end());
  EXPECT_TRUE(contains_subsequence(acquire_words, std::array<uint32_t, 2>{*save_scc, *save_vcc}));
  EXPECT_TRUE(
      contains_subsequence(acquire_words, std::array<uint32_t, 2>{*restore_vcc, *restore_scc}));
}

TEST(ConSanMoi, Gfx950InlineAtomicOrderingEmitsReleaseAndAcquirePatches) {
  const std::vector<uint8_t> bytes = make_gfx950_flat_atomic_release_acquire_code_object();
  ASSERT_FALSE(bytes.empty());
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::InlineShadow;
  options.moi_track_atomics = true;
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 13;
  options.moi_epoch_vgpr = 14;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels.front().atomic_sites.size(), 2u);
  EXPECT_EQ(std::ranges::count_if(result.patches,
                                  [](const ConSanPatchInfo &patch) {
                                    return patch.kind ==
                                           ConSanPatchKind::TrampolineMoiInlineAtomicOrdering;
                                  }),
            2u);
  EXPECT_EQ(std::ranges::count_if(result.patches,
                                  [](const ConSanPatchInfo &patch) {
                                    return patch.kind == ConSanPatchKind::TrampolineMoiAtomicRecord;
                                  }),
            0u);
}

TEST(ConSanMoi, Gfx950InlineAtomicOrderingPartitionsReleaseStateByWorkgroup) {
  const std::vector<uint8_t> bytes = make_gfx950_flat_atomic_release_acquire_code_object();
  ASSERT_FALSE(bytes.empty());
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::InlineShadow;
  options.moi_track_atomics = true;
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 13;
  options.moi_epoch_vgpr = 14;
  options.moi_report_buffer_address = 0x1234fffffff0ull;
  options.moi_report_buffer_size = 2u * 1024u * 1024u;
  options.moi_workgroup_extent_x = 2;
  options.moi_workgroup_extent_y = 2;
  options.moi_workgroup_extent_z = 2;
  options.max_patches = 2;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const char *text = patched.text_sections().front()->data();
  const auto partition = build_v_mad_u32_u24(
      /*vdst=*/10, vector_source_vgpr(8), /*vsrc1=*/10, /*vsrc2=*/9, ROCJITSU_CODE_ARCH_CDNA4);
  const auto slot_stride = build_v_mov_b32_e64_literal(
      /*vdst=*/8, sizeof(ConSanMoiInlineAtomicReleaseSlot), ROCJITSU_CODE_ARCH_CDNA4);
  const auto overflow_atomic = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/8, /*vsrc=*/10, /*vdst=*/10, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(partition && slot_stride && overflow_atomic);
  size_t atomic_patch_count = 0;
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind != ConSanPatchKind::TrampolineMoiInlineAtomicOrdering)
      continue;
    ++atomic_patch_count;
    std::vector<uint32_t> words(patch.trampoline_size / sizeof(uint32_t));
    std::memcpy(words.data(), text + patch.trampoline_offset, patch.trampoline_size);
    EXPECT_TRUE(contains_subsequence(words, *partition));
    EXPECT_TRUE(contains_subsequence(words, *slot_stride));
    EXPECT_TRUE(contains_subsequence(words, *overflow_atomic));
  }
  EXPECT_EQ(atomic_patch_count, 2u);
}

TEST(ConSanMoi, Gfx950AccumOverlapInlineAtomicsUseScalarOwnerEpochState) {
  std::vector<uint8_t> bytes = make_gfx950_flat_atomic_release_acquire_code_object();
  ASSERT_FALSE(bytes.empty());
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 3u);
  });
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::InlineShadow;
  options.moi_track_atomics = true;
  options.scratch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 2;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.moi_scalar_identity_automatic);
  EXPECT_FALSE(result.moi_private_epoch_automatic);
  EXPECT_FALSE(result.moi_persistent_vgprs_automatic);
  const auto prologue = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
  });
  ASSERT_NE(prologue, result.patches.end());

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const char *text = patched.text_sections().front()->data();
  ASSERT_TRUE(result.resolved_moi_state_owner_sgpr);
  ASSERT_TRUE(result.resolved_moi_state_epoch_sgpr);
  const uint32_t materialize_owner = build_v_mov_b32_e32(
      /*vdst=*/14, *result.resolved_moi_state_owner_sgpr, ROCJITSU_CODE_ARCH_CDNA4);
  const uint32_t materialize_epoch = build_v_mov_b32_e32(
      /*vdst=*/14, *result.resolved_moi_state_epoch_sgpr, ROCJITSU_CODE_ARCH_CDNA4);

  size_t atomic_patch_count = 0;
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind != ConSanPatchKind::TrampolineMoiInlineAtomicOrdering)
      continue;
    ++atomic_patch_count;
    EXPECT_FALSE(patch.persistent_epoch_private_offset);
    EXPECT_FALSE(patch.persistent_owner_private_offset);
    std::vector<uint32_t> words(patch.trampoline_size / sizeof(uint32_t));
    std::memcpy(words.data(), text + patch.trampoline_offset, patch.trampoline_size);
    if (patch.anchor_offset == 0u) {
      EXPECT_TRUE(std::ranges::find(words, materialize_owner) != words.end());
      EXPECT_TRUE(std::ranges::find(words, materialize_epoch) != words.end());
    } else {
      const uint32_t materialize_acquire_owner = build_v_mov_b32_e32(
          /*vdst=*/12, *result.resolved_moi_state_owner_sgpr, ROCJITSU_CODE_ARCH_CDNA4);
      EXPECT_TRUE(std::ranges::find(words, materialize_acquire_owner) != words.end());
      const auto execz = build_s_cbranch_execz(0, ROCJITSU_CODE_ARCH_CDNA4);
      const auto import_epoch = build_v_readfirstlane_b32(*result.resolved_moi_state_epoch_sgpr,
                                                          /*vsrc=*/14, ROCJITSU_CODE_ARCH_CDNA4);
      ASSERT_TRUE(execz && import_epoch);
      EXPECT_TRUE(std::ranges::any_of(
          words, [&](uint32_t word) { return (word & 0xffff0000u) == (*execz & 0xffff0000u); }));
      EXPECT_NE(std::ranges::find(words, *import_epoch), words.end());
    }
  }
  EXPECT_EQ(atomic_patch_count, 2u);
}

TEST(ConSanMoi, Gfx950PrivateInlineAtomicSpillsStayBeyondPersistentState) {
  std::vector<uint8_t> bytes = make_gfx950_flat_atomic_release_acquire_code_object();
  ASSERT_FALSE(bytes.empty());
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 2u);
  });
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::InlineShadow;
  options.moi_track_atomics = true;
  options.force_private_epoch = true;
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 2;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.moi_private_epoch_automatic);
  size_t atomic_patch_count = 0;
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind != ConSanPatchKind::TrampolineMoiInlineAtomicOrdering)
      continue;
    ++atomic_patch_count;
    EXPECT_EQ(patch.persistent_epoch_private_offset, 0u);
    EXPECT_EQ(patch.persistent_owner_private_offset, 4u);
    EXPECT_EQ(patch.spilled_vgpr_count, 3u);
    EXPECT_GE(patch.required_private_segment_size, 28u);
  }
  EXPECT_EQ(atomic_patch_count, 2u);
}

TEST(ConSanMoi, Gfx950AtomicRecordForcedSpillUsesPlannedPrivateWindows) {
  const std::vector<uint8_t> bytes = make_gfx950_flat_atomic_release_acquire_code_object();
  ASSERT_FALSE(bytes.empty());
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_track_atomics = true;
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(2, 0, 0, 0, 0, 2);
  options.max_patches = 2;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_EQ(std::ranges::count_if(result.patches,
                                  [](const ConSanPatchInfo &patch) {
                                    return patch.kind == ConSanPatchKind::TrampolineMoiAtomicRecord;
                                  }),
            2u);
  for (const ConSanPatchInfo &patch : result.patches)
    if (patch.kind == ConSanPatchKind::TrampolineMoiAtomicRecord) {
      EXPECT_EQ(patch.spilled_vgpr_count, 3u);
      EXPECT_GT(patch.required_private_segment_size, 0u);
    }
  EXPECT_EQ(result.resource_plan_summary.emitted_spill_patches, 2u);
  EXPECT_EQ(result.resource_plan_summary.emitted_spill_slot_bytes, 24u);
}

TEST(ConSanMoi, Gfx950InlineAtomicForcedSpillUsesPlannedPrivateWindows) {
  const std::vector<uint8_t> bytes = make_gfx950_flat_atomic_release_acquire_code_object();
  ASSERT_FALSE(bytes.empty());
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::InlineShadow;
  options.moi_track_atomics = true;
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 2;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_EQ(std::ranges::count_if(result.patches,
                                  [](const ConSanPatchInfo &patch) {
                                    return patch.kind ==
                                           ConSanPatchKind::TrampolineMoiInlineAtomicOrdering;
                                  }),
            2u);
  for (const ConSanPatchInfo &patch : result.patches)
    if (patch.kind == ConSanPatchKind::TrampolineMoiInlineAtomicOrdering) {
      EXPECT_EQ(patch.spilled_vgpr_count, 3u);
      EXPECT_GT(patch.required_private_segment_size, 0u);
    }
  EXPECT_EQ(result.resource_plan_summary.emitted_spill_patches, 2u);
  EXPECT_EQ(result.resource_plan_summary.emitted_spill_slot_bytes, 24u);
}

TEST(ConSanMoi, InlineAtomicOrderingAutomaticallyPlansAllRegisterState) {
  const std::vector<uint8_t> bytes = make_rdna4_flat_atomic_release_acquire_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::InlineShadow;
  options.moi_track_atomics = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.moi_exec_save_sgprs_automatic);
  EXPECT_TRUE(result.moi_persistent_vgprs_automatic);
  ASSERT_TRUE(result.resolved_moi_owner_vgpr);
  ASSERT_TRUE(result.resolved_moi_epoch_vgpr);
  EXPECT_EQ(std::ranges::count_if(result.patches,
                                  [](const ConSanPatchInfo &patch) {
                                    return patch.kind ==
                                           ConSanPatchKind::TrampolineMoiInlineAtomicOrdering;
                                  }),
            2u);
  EXPECT_EQ(std::ranges::count_if(result.resource_plans,
                                  [](const auto &plan) {
                                    return plan.site_kind == ConSanResourceSiteKind::Atomic &&
                                           plan.scratch_vgpr;
                                  }),
            2u);
}

TEST(ConSanMoi, FirstLightProbeRejectsScratchVgprsOverlappingLdsAddress) {
  std::array<uint32_t, 76> text_words{};
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  for (size_t i = 2; i + 1 < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  text_words.back() = 0xBFB00000u; // s_endpgm

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.scratch_vgpr = 0;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.errors.empty());
  bool saw_overlap_warning = false;
  for (const std::string &warning : result.warnings)
    saw_overlap_warning |= warning.find("scratch VGPRs overlap") != std::string::npos;
  EXPECT_TRUE(saw_overlap_warning);
}

TEST(ConSanMoi, ExactShadowEntryRoundTripsMaskedFields) {
  constexpr uint64_t packed = pack_consan_moi_exact_shadow_entry(ConSanMoiShadowAccessKind::Write,
                                                                 /*owner_id=*/0x413,
                                                                 /*epoch=*/0x477,
                                                                 /*generation=*/0x1f1234,
                                                                 /*instruction_offset=*/0x9abcdef);
  constexpr ConSanMoiExactShadowEntry decoded = decode_consan_moi_exact_shadow_entry(packed);

  EXPECT_EQ(decoded.kind, ConSanMoiShadowAccessKind::Write);
  EXPECT_EQ(decoded.owner_id, 0x13u);
  EXPECT_EQ(decoded.epoch, 0x77u);
  EXPECT_EQ(decoded.generation, 0xf1234u);
  EXPECT_EQ(decoded.instruction_offset, 0xbcdefu);
}

TEST(ConSanMoi, ExactShadowConflictPredicateMatchesSubgroupContract) {
  constexpr ConSanMoiExactShadowEntry current{
      ConSanMoiShadowAccessKind::Read,
      /*owner_id=*/2,
      /*epoch=*/17,
      /*generation=*/99,
      /*instruction_offset=*/0x100,
  };
  constexpr ConSanMoiExactShadowEntry prior_write{
      ConSanMoiShadowAccessKind::Write,
      /*owner_id=*/1,
      /*epoch=*/17,
      /*generation=*/99,
      /*instruction_offset=*/0x120,
  };
  constexpr ConSanMoiExactShadowEntry same_owner{
      ConSanMoiShadowAccessKind::Write,
      /*owner_id=*/2,
      /*epoch=*/17,
      /*generation=*/99,
      /*instruction_offset=*/0x120,
  };
  constexpr ConSanMoiExactShadowEntry old_epoch{
      ConSanMoiShadowAccessKind::Write,
      /*owner_id=*/1,
      /*epoch=*/16,
      /*generation=*/99,
      /*instruction_offset=*/0x120,
  };
  constexpr ConSanMoiExactShadowEntry prior_read{
      ConSanMoiShadowAccessKind::Read,
      /*owner_id=*/1,
      /*epoch=*/17,
      /*generation=*/99,
      /*instruction_offset=*/0x120,
  };

  EXPECT_TRUE(consan_moi_exact_shadow_entries_conflict(current, prior_write));
  EXPECT_FALSE(consan_moi_exact_shadow_entries_conflict(current, same_owner));
  EXPECT_FALSE(consan_moi_exact_shadow_entries_conflict(current, old_epoch));
  EXPECT_FALSE(consan_moi_exact_shadow_entries_conflict(current, prior_read));
}

TEST(ConSanMoi, RecordReplayExactShadowReportsSameEpochConflicts) {
  std::array<uint64_t, 1> shadow{};
  const ConSanMoiRecordReplayAccess writer{
      /*generation=*/7,
      /*owner_id=*/1,
      /*epoch=*/3,
      ConSanMoiShadowAccessKind::Write,
      /*lds_byte_offset=*/0,
      /*lds_byte_count=*/4,
      /*start_cell=*/0,
      /*cell_count=*/1,
      /*instruction_offset=*/0x10,
      /*lane_mask=*/0x1,
  };
  const ConSanMoiRecordReplayAccess reader{
      /*generation=*/7,
      /*owner_id=*/2,
      /*epoch=*/3,
      ConSanMoiShadowAccessKind::Read,
      /*lds_byte_offset=*/0,
      /*lds_byte_count=*/4,
      /*start_cell=*/0,
      /*cell_count=*/1,
      /*instruction_offset=*/0x20,
      /*lane_mask=*/0x2,
  };

  const auto first = consan_moi_record_replay_access(shadow, writer);
  EXPECT_FALSE(first.conflict);
  EXPECT_NE(shadow[0], 0u);

  const auto second = consan_moi_record_replay_access(shadow, reader);
  EXPECT_TRUE(second.conflict);
  EXPECT_FALSE(second.metadata_full);
  EXPECT_EQ(second.diagnostic.kind, static_cast<uint32_t>(ConSanMoiDiagnosticKind::AccessConflict));
  EXPECT_EQ(second.diagnostic.backend, static_cast<uint32_t>(ConSanMoiEngine::RecordReplay));
  EXPECT_EQ(second.diagnostic.generation, 7u);
  EXPECT_EQ(second.diagnostic.epoch, 3u);
  EXPECT_EQ(second.diagnostic.first_owner_id, 1u);
  EXPECT_EQ(second.diagnostic.second_owner_id, 2u);
  EXPECT_EQ(second.diagnostic.second_lane_mask, 0x2u);
  EXPECT_EQ(second.diagnostic.first_instruction_offset, 0x10u);
  EXPECT_EQ(second.diagnostic.second_instruction_offset, 0x20u);
  EXPECT_EQ(second.diagnostic.second_access_kind,
            static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read));
}

TEST(ConSanMoi, RecordReplayExactShadowTreatsDifferentEpochAsOrdered) {
  std::array<uint64_t, 1> shadow{};
  ConSanMoiRecordReplayAccess writer{
      /*generation=*/7,
      /*owner_id=*/1,
      /*epoch=*/3,
      ConSanMoiShadowAccessKind::Write,
      /*lds_byte_offset=*/0,
      /*lds_byte_count=*/4,
      /*start_cell=*/0,
      /*cell_count=*/1,
      /*instruction_offset=*/0x10,
      /*lane_mask=*/0x1,
  };
  ConSanMoiRecordReplayAccess reader = writer;
  reader.owner_id = 2;
  reader.epoch = 4;
  reader.kind = ConSanMoiShadowAccessKind::Read;
  reader.instruction_offset = 0x20;
  reader.lane_mask = 0x2;

  EXPECT_FALSE(consan_moi_record_replay_access(shadow, writer).conflict);
  const auto second = consan_moi_record_replay_access(shadow, reader);
  EXPECT_FALSE(second.conflict);
  const ConSanMoiExactShadowEntry updated = decode_consan_moi_exact_shadow_entry(shadow[0]);
  EXPECT_EQ(updated.kind, ConSanMoiShadowAccessKind::Read);
  EXPECT_EQ(updated.owner_id, 2u);
  EXPECT_EQ(updated.epoch, 4u);
}

TEST(ConSanMoi, RecordReplayAtomicAcquireSuppressesSameEpochConflict) {
  std::array<uint64_t, 1> shadow{};
  std::array<ConSanMoiAtomicReleaseRecord, 1> releases{};
  std::array<ConSanMoiAcquiredEpochToken, 1> tokens{};

  const ConSanMoiRecordReplayAccess writer{
      /*generation=*/7,
      /*owner_id=*/0,
      /*epoch=*/0,
      ConSanMoiShadowAccessKind::Write,
      /*lds_byte_offset=*/0,
      /*lds_byte_count=*/4,
      /*start_cell=*/0,
      /*cell_count=*/1,
      /*instruction_offset=*/0x10,
      /*lane_mask=*/0x1,
  };
  const ConSanMoiRecordReplayAccess reader{
      /*generation=*/7,
      /*owner_id=*/1,
      /*epoch=*/0,
      ConSanMoiShadowAccessKind::Read,
      /*lds_byte_offset=*/0,
      /*lds_byte_count=*/4,
      /*start_cell=*/0,
      /*cell_count=*/1,
      /*instruction_offset=*/0x20,
      /*lane_mask=*/0x2,
  };

  EXPECT_FALSE(consan_moi_record_replay_access(shadow, writer).conflict);
  const ConSanMoiAtomicSyncResult release = consan_moi_record_replay_atomic_release(
      releases, /*generation=*/7, /*atomic_address=*/0x4000, /*producer_owner_id=*/0,
      /*producer_epoch=*/0, /*release_instruction_offset=*/0x100);
  EXPECT_FALSE(release.metadata_full);
  EXPECT_EQ(release.updated_record_count, 1u);

  const ConSanMoiAtomicSyncResult acquire = consan_moi_record_replay_atomic_acquire(
      releases, tokens, /*generation=*/7, /*atomic_address=*/0x4000, /*consumer_owner_id=*/1,
      /*acquire_instruction_offset=*/0x200);
  EXPECT_FALSE(acquire.metadata_full);
  EXPECT_EQ(acquire.updated_record_count, 1u);
  ASSERT_TRUE(tokens[0].valid);
  EXPECT_EQ(tokens[0].consumer_owner_id, 1u);
  EXPECT_EQ(tokens[0].producer_owner_id, 0u);
  EXPECT_EQ(tokens[0].producer_epoch_plus_one, 1u);

  const auto second = consan_moi_record_replay_access(shadow, reader, tokens);
  EXPECT_FALSE(second.conflict);
  const ConSanMoiExactShadowEntry updated = decode_consan_moi_exact_shadow_entry(shadow[0]);
  EXPECT_EQ(updated.kind, ConSanMoiShadowAccessKind::Read);
  EXPECT_EQ(updated.owner_id, 1u);
  EXPECT_EQ(updated.epoch, 0u);
}

TEST(ConSanMoi, RecordReplayAtomicOrderingRequiresMatchingAcquireAddress) {
  const auto make_writer = [] {
    return ConSanMoiRecordReplayAccess{
        /*generation=*/7,
        /*owner_id=*/0,
        /*epoch=*/0,
        ConSanMoiShadowAccessKind::Write,
        /*lds_byte_offset=*/0,
        /*lds_byte_count=*/4,
        /*start_cell=*/0,
        /*cell_count=*/1,
        /*instruction_offset=*/0x10,
        /*lane_mask=*/0x1,
    };
  };
  const auto make_reader = [] {
    return ConSanMoiRecordReplayAccess{
        /*generation=*/7,
        /*owner_id=*/1,
        /*epoch=*/0,
        ConSanMoiShadowAccessKind::Read,
        /*lds_byte_offset=*/0,
        /*lds_byte_count=*/4,
        /*start_cell=*/0,
        /*cell_count=*/1,
        /*instruction_offset=*/0x20,
        /*lane_mask=*/0x2,
    };
  };

  {
    SCOPED_TRACE("release without acquire behaves like a relaxed atomic for diagnostics");
    std::array<uint64_t, 1> shadow{};
    std::array<ConSanMoiAtomicReleaseRecord, 1> releases{};
    std::array<ConSanMoiAcquiredEpochToken, 1> tokens{};

    EXPECT_FALSE(consan_moi_record_replay_access(shadow, make_writer()).conflict);
    EXPECT_FALSE(consan_moi_record_replay_atomic_release(
                     releases, /*generation=*/7, /*atomic_address=*/0x4000,
                     /*producer_owner_id=*/0, /*producer_epoch=*/0,
                     /*release_instruction_offset=*/0x100)
                     .metadata_full);

    const auto second = consan_moi_record_replay_access(shadow, make_reader(), tokens);
    EXPECT_TRUE(second.conflict);
    EXPECT_FALSE(second.metadata_full);
  }

  {
    SCOPED_TRACE("acquire of a different atomic address does not import the producer");
    std::array<uint64_t, 1> shadow{};
    std::array<ConSanMoiAtomicReleaseRecord, 1> releases{};
    std::array<ConSanMoiAcquiredEpochToken, 1> tokens{};

    EXPECT_FALSE(consan_moi_record_replay_access(shadow, make_writer()).conflict);
    EXPECT_FALSE(consan_moi_record_replay_atomic_release(
                     releases, /*generation=*/7, /*atomic_address=*/0x4000,
                     /*producer_owner_id=*/0, /*producer_epoch=*/0,
                     /*release_instruction_offset=*/0x100)
                     .metadata_full);
    EXPECT_EQ(consan_moi_record_replay_atomic_acquire(
                  releases, tokens, /*generation=*/7, /*atomic_address=*/0x5000,
                  /*consumer_owner_id=*/1, /*acquire_instruction_offset=*/0x200)
                  .updated_record_count,
              0u);

    const auto second = consan_moi_record_replay_access(shadow, make_reader(), tokens);
    EXPECT_TRUE(second.conflict);
    EXPECT_FALSE(second.metadata_full);
  }
}

TEST(ConSanMoi, RecordReplayAtomicReleaseKeepsMaxEpochPerProducerAddress) {
  std::array<ConSanMoiAtomicReleaseRecord, 1> releases{};

  EXPECT_EQ(consan_moi_record_replay_atomic_release(releases, /*generation=*/7,
                                                    /*atomic_address=*/0x4000,
                                                    /*producer_owner_id=*/0, /*producer_epoch=*/2,
                                                    /*release_instruction_offset=*/0x100)
                .updated_record_count,
            1u);
  ASSERT_TRUE(releases[0].valid);
  EXPECT_EQ(releases[0].producer_epoch, 2u);
  EXPECT_EQ(releases[0].release_instruction_offset, 0x100u);

  EXPECT_EQ(consan_moi_record_replay_atomic_release(releases, /*generation=*/7,
                                                    /*atomic_address=*/0x4000,
                                                    /*producer_owner_id=*/0, /*producer_epoch=*/1,
                                                    /*release_instruction_offset=*/0x110)
                .updated_record_count,
            0u);
  EXPECT_EQ(releases[0].producer_epoch, 2u);
  EXPECT_EQ(releases[0].release_instruction_offset, 0x100u);

  EXPECT_EQ(consan_moi_record_replay_atomic_release(releases, /*generation=*/7,
                                                    /*atomic_address=*/0x4000,
                                                    /*producer_owner_id=*/0, /*producer_epoch=*/5,
                                                    /*release_instruction_offset=*/0x120)
                .updated_record_count,
            1u);
  EXPECT_EQ(releases[0].producer_epoch, 5u);
  EXPECT_EQ(releases[0].release_instruction_offset, 0x120u);

  const ConSanMoiAtomicSyncResult full = consan_moi_record_replay_atomic_release(
      releases, /*generation=*/7, /*atomic_address=*/0x4000, /*producer_owner_id=*/1,
      /*producer_epoch=*/0, /*release_instruction_offset=*/0x130);
  EXPECT_TRUE(full.metadata_full);
  EXPECT_EQ(full.updated_record_count, 0u);
}

TEST(ConSanMoi, RecordReplayAtomicEventsSuppressOrderedConflict) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/2,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
      /*sampled_watchpoint_capacity=*/0);
  header.access_record_count = 2;

  std::array<ConSanMoiAccessRecord, 2> records{};
  records[0].wave_id = 0;
  records[0].event_index = 0;
  records[0].instruction_offset = 0x10;
  records[0].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
  records[0].lds_byte_count = 4;
  records[0].cell_count = 1;

  records[1].wave_id = 1;
  records[1].event_index = 3;
  records[1].instruction_offset = 0x20;
  records[1].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read);
  records[1].lds_byte_count = 4;
  records[1].cell_count = 1;

  std::array<ConSanMoiRecordReplayAtomicEvent, 2> atomics{};
  atomics[0].owner_id = 0;
  atomics[0].atomic_address = 0x4000;
  atomics[0].instruction_offset = 0x100;
  atomics[0].event_index = 1;
  atomics[0].kind = ConSanMoiAtomicEventKind::Release;

  atomics[1].owner_id = 1;
  atomics[1].atomic_address = 0x4000;
  atomics[1].instruction_offset = 0x200;
  atomics[1].event_index = 2;
  atomics[1].kind = ConSanMoiAtomicEventKind::Acquire;

  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  std::array<uint64_t, 1> shadow{};

  const ConSanMoiRecordReplayResult replay = consan_moi_record_replay_access_records(
      header, records, std::span<const ConSanMoiBarrierRecord>{}, atomics, diagnostics, shadow);

  EXPECT_EQ(replay.processed_access_count, 2u);
  EXPECT_EQ(replay.processed_atomic_count, 2u);
  EXPECT_EQ(replay.unsupported_atomic_count, 0u);
  EXPECT_FALSE(replay.metadata_full);
  EXPECT_FALSE(replay.conflict);
  EXPECT_EQ(header.diagnostic_count, 0u);

  const ConSanMoiExactShadowEntry final = decode_consan_moi_exact_shadow_entry(shadow[0]);
  EXPECT_EQ(final.kind, ConSanMoiShadowAccessKind::Read);
  EXPECT_EQ(final.owner_id, 1u);
  EXPECT_EQ(final.epoch, 0u);
  EXPECT_EQ(final.instruction_offset, 0x20u);
}

TEST(ConSanMoi, RecordReplayAtomicEventsRequireMatchingAddress) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/2,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
      /*sampled_watchpoint_capacity=*/0);
  header.access_record_count = 2;

  std::array<ConSanMoiAccessRecord, 2> records{};
  records[0].wave_id = 0;
  records[0].event_index = 0;
  records[0].instruction_offset = 0x10;
  records[0].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
  records[0].lds_byte_count = 4;
  records[0].cell_count = 1;

  records[1].wave_id = 1;
  records[1].event_index = 3;
  records[1].instruction_offset = 0x20;
  records[1].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read);
  records[1].lds_byte_count = 4;
  records[1].cell_count = 1;

  std::array<ConSanMoiRecordReplayAtomicEvent, 2> atomics{};
  atomics[0].owner_id = 0;
  atomics[0].atomic_address = 0x4000;
  atomics[0].instruction_offset = 0x100;
  atomics[0].event_index = 1;
  atomics[0].kind = ConSanMoiAtomicEventKind::Release;

  atomics[1].owner_id = 1;
  atomics[1].atomic_address = 0x5000;
  atomics[1].instruction_offset = 0x200;
  atomics[1].event_index = 2;
  atomics[1].kind = ConSanMoiAtomicEventKind::Acquire;

  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  std::array<uint64_t, 1> shadow{};

  const ConSanMoiRecordReplayResult replay = consan_moi_record_replay_access_records(
      header, records, std::span<const ConSanMoiBarrierRecord>{}, atomics, diagnostics, shadow);

  EXPECT_EQ(replay.processed_access_count, 2u);
  EXPECT_EQ(replay.processed_atomic_count, 2u);
  EXPECT_TRUE(replay.conflict);
  ASSERT_EQ(header.diagnostic_count, 1u);
  EXPECT_EQ(diagnostics[0].first_instruction_offset, 0x10u);
  EXPECT_EQ(diagnostics[0].second_instruction_offset, 0x20u);
}

TEST(ConSanMoi, RecordReplayAtomicEventsAreWorkgroupLocalForLdsOrdering) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/2,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
      /*sampled_watchpoint_capacity=*/0);
  header.access_record_count = 2;

  std::array<ConSanMoiAccessRecord, 2> records{};
  records[0].workgroup_x = 1;
  records[0].wave_id = 0;
  records[0].event_index = 0;
  records[0].instruction_offset = 0x10;
  records[0].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
  records[0].lds_byte_count = 4;
  records[0].cell_count = 1;

  records[1].workgroup_x = 1;
  records[1].wave_id = 1;
  records[1].event_index = 3;
  records[1].instruction_offset = 0x20;
  records[1].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read);
  records[1].lds_byte_count = 4;
  records[1].cell_count = 1;

  std::array<ConSanMoiRecordReplayAtomicEvent, 2> atomics{};
  atomics[0].workgroup_x = 0;
  atomics[0].owner_id = 0;
  atomics[0].atomic_address = 0x4000;
  atomics[0].instruction_offset = 0x100;
  atomics[0].event_index = 1;
  atomics[0].kind = ConSanMoiAtomicEventKind::Release;

  atomics[1].workgroup_x = 1;
  atomics[1].owner_id = 1;
  atomics[1].atomic_address = 0x4000;
  atomics[1].instruction_offset = 0x200;
  atomics[1].event_index = 2;
  atomics[1].kind = ConSanMoiAtomicEventKind::Acquire;

  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  std::array<uint64_t, 1> shadow{};

  const ConSanMoiRecordReplayResult replay = consan_moi_record_replay_access_records(
      header, records, std::span<const ConSanMoiBarrierRecord>{}, atomics, diagnostics, shadow);

  EXPECT_EQ(replay.processed_atomic_count, 2u);
  EXPECT_TRUE(replay.conflict);
  ASSERT_EQ(header.diagnostic_count, 1u);
  EXPECT_EQ(diagnostics[0].first_owner_id, 0u);
  EXPECT_EQ(diagnostics[0].second_owner_id, 1u);
}

TEST(ConSanMoi, RecordReplayExactShadowReportsMetadataFullForOutOfRangeAccess) {
  std::array<uint64_t, 1> shadow{};
  const ConSanMoiRecordReplayAccess access{
      /*generation=*/9,
      /*owner_id=*/3,
      /*epoch=*/5,
      ConSanMoiShadowAccessKind::Write,
      /*lds_byte_offset=*/8,
      /*lds_byte_count=*/4,
      /*start_cell=*/2,
      /*cell_count=*/1,
      /*instruction_offset=*/0x30,
      /*lane_mask=*/0x4,
  };

  const auto result = consan_moi_record_replay_access(shadow, access);

  EXPECT_TRUE(result.conflict);
  EXPECT_TRUE(result.metadata_full);
  EXPECT_EQ(result.diagnostic.kind, static_cast<uint32_t>(ConSanMoiDiagnosticKind::MetadataFull));
  EXPECT_EQ(result.diagnostic.second_owner_id, 3u);
  EXPECT_EQ(result.diagnostic.second_instruction_offset, 0x30u);
  EXPECT_EQ(shadow[0], 0u);
}

TEST(ConSanMoi, RecordReplayAccessRecordsEmitsConflictDiagnostic) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/2,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/3,
      /*sampled_watchpoint_capacity=*/0);
  header.access_record_count = 2;

  std::array<ConSanMoiAccessRecord, 2> records{};
  records[0].wave_id = 1;
  records[0].lane_mask = 0x1;
  records[0].instruction_offset = 0x10;
  records[0].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
  records[0].lds_byte_offset = 8;
  records[0].lds_byte_count = 4;
  records[0].epoch = 3;

  records[1].wave_id = 2;
  records[1].lane_mask = 0x2;
  records[1].instruction_offset = 0x20;
  records[1].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read);
  records[1].lds_byte_offset = 8;
  records[1].lds_byte_count = 4;
  records[1].epoch = 3;

  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  std::array<uint64_t, 3> shadow{};

  const ConSanMoiRecordReplayResult replay =
      consan_moi_record_replay_access_records(header, records, diagnostics, shadow);

  EXPECT_EQ(replay.processed_access_count, 2u);
  EXPECT_EQ(replay.dropped_access_count, 0u);
  EXPECT_EQ(replay.unsupported_access_count, 0u);
  EXPECT_EQ(replay.emitted_diagnostic_count, 1u);
  EXPECT_FALSE(replay.diagnostic_capacity_exhausted);
  EXPECT_FALSE(replay.metadata_full);
  EXPECT_TRUE(replay.conflict);
  EXPECT_EQ(header.diagnostic_count, 1u);
  EXPECT_EQ(shadow[0], 0u);
  EXPECT_NE(shadow[2], 0u);

  const ConSanMoiDiagnosticRecord &diagnostic = diagnostics[0];
  EXPECT_EQ(diagnostic.kind, static_cast<uint32_t>(ConSanMoiDiagnosticKind::AccessConflict));
  EXPECT_EQ(diagnostic.backend, static_cast<uint32_t>(ConSanMoiEngine::RecordReplay));
  EXPECT_EQ(diagnostic.generation, 7u);
  EXPECT_EQ(diagnostic.epoch, 3u);
  EXPECT_EQ(diagnostic.first_owner_id, 1u);
  EXPECT_EQ(diagnostic.second_owner_id, 2u);
  EXPECT_EQ(diagnostic.second_lane_mask, 0x2u);
  EXPECT_EQ(diagnostic.first_instruction_offset, 0x10u);
  EXPECT_EQ(diagnostic.second_instruction_offset, 0x20u);
  EXPECT_EQ(diagnostic.second_lds_byte_offset, 8u);
  EXPECT_EQ(diagnostic.second_lds_byte_count, 4u);
  EXPECT_EQ(diagnostic.first_access_kind, static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write));
  EXPECT_EQ(diagnostic.second_access_kind, static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read));
}

TEST(ConSanMoi, RecordReplaySeparatesExactShadowByWorkgroup) {
  auto make_record = [](uint32_t workgroup_x, uint32_t wave_id, ConSanMoiShadowAccessKind kind,
                        uint32_t instruction_offset) {
    ConSanMoiAccessRecord record{};
    record.workgroup_x = workgroup_x;
    record.wave_id = wave_id;
    record.lane_mask = uint64_t{1} << wave_id;
    record.instruction_offset = instruction_offset;
    record.access_kind = static_cast<uint32_t>(kind);
    record.lds_byte_count = 4;
    record.cell_count = 1;
    return record;
  };

  {
    SCOPED_TRACE("same workgroup accesses conflict");
    ConSanMoiReportHeader header = make_consan_moi_report_header(
        /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/2,
        /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
        /*sampled_watchpoint_capacity=*/0);
    header.access_record_count = 2;
    std::array<ConSanMoiAccessRecord, 2> records = {
        make_record(/*workgroup_x=*/3, /*wave_id=*/1, ConSanMoiShadowAccessKind::Write,
                    /*instruction_offset=*/0x10),
        make_record(/*workgroup_x=*/3, /*wave_id=*/2, ConSanMoiShadowAccessKind::Read,
                    /*instruction_offset=*/0x20),
    };
    std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
    std::array<uint64_t, 1> shadow{};

    const ConSanMoiRecordReplayResult replay =
        consan_moi_record_replay_access_records(header, records, diagnostics, shadow);

    EXPECT_TRUE(replay.conflict);
    ASSERT_EQ(header.diagnostic_count, 1u);
    EXPECT_EQ(diagnostics[0].first_instruction_offset, 0x10u);
    EXPECT_EQ(diagnostics[0].second_instruction_offset, 0x20u);
  }

  {
    SCOPED_TRACE("different workgroup accesses do not conflict");
    ConSanMoiReportHeader header = make_consan_moi_report_header(
        /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/2,
        /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
        /*sampled_watchpoint_capacity=*/0);
    header.access_record_count = 2;
    std::array<ConSanMoiAccessRecord, 2> records = {
        make_record(/*workgroup_x=*/3, /*wave_id=*/1, ConSanMoiShadowAccessKind::Write,
                    /*instruction_offset=*/0x10),
        make_record(/*workgroup_x=*/4, /*wave_id=*/2, ConSanMoiShadowAccessKind::Read,
                    /*instruction_offset=*/0x20),
    };
    std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
    std::array<uint64_t, 1> shadow{};

    const ConSanMoiRecordReplayResult replay =
        consan_moi_record_replay_access_records(header, records, diagnostics, shadow);

    EXPECT_FALSE(replay.conflict);
    EXPECT_EQ(header.diagnostic_count, 0u);
  }
}

TEST(ConSanMoi, RecordReplayBarrierEventsAdvanceEpochsInEventOrder) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/2,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
      /*sampled_watchpoint_capacity=*/0, /*barrier_record_capacity=*/1);
  header.access_record_count = 2;
  header.barrier_record_count = 1;

  std::array<ConSanMoiAccessRecord, 2> records{};
  records[0].wave_id = 0;
  records[0].event_index = 0;
  records[0].instruction_offset = 0x10;
  records[0].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
  records[0].lds_byte_count = 4;
  records[0].cell_count = 1;

  records[1].wave_id = 1;
  records[1].event_index = 3;
  records[1].instruction_offset = 0x20;
  records[1].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read);
  records[1].lds_byte_count = 4;
  records[1].cell_count = 1;

  std::array<ConSanMoiBarrierRecord, 1> barriers{};
  barriers[0].wave_id = 0;
  barriers[0].event_index = 1;
  barriers[0].instruction_offset = 0x18;

  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  std::array<uint64_t, 1> shadow{};

  ConSanMoiReportHeader no_barrier_header = header;
  no_barrier_header.barrier_record_count = 0;
  const ConSanMoiRecordReplayResult no_barrier =
      consan_moi_record_replay_access_records(no_barrier_header, records, diagnostics, shadow);
  EXPECT_TRUE(no_barrier.conflict);
  EXPECT_EQ(no_barrier.processed_barrier_count, 0u);
  EXPECT_EQ(no_barrier_header.diagnostic_count, 1u);

  diagnostics = {};
  shadow = {};
  const ConSanMoiRecordReplayResult with_barrier =
      consan_moi_record_replay_access_records(header, records, barriers, diagnostics, shadow);
  EXPECT_FALSE(with_barrier.conflict);
  EXPECT_EQ(with_barrier.processed_access_count, 2u);
  EXPECT_EQ(with_barrier.processed_barrier_count, 1u);
  EXPECT_EQ(with_barrier.dropped_barrier_count, 0u);
  EXPECT_EQ(header.diagnostic_count, 0u);

  const ConSanMoiExactShadowEntry final = decode_consan_moi_exact_shadow_entry(shadow[0]);
  EXPECT_EQ(final.kind, ConSanMoiShadowAccessKind::Read);
  EXPECT_EQ(final.owner_id, 1u);
  EXPECT_EQ(final.epoch, 1u);
  EXPECT_EQ(final.instruction_offset, 0x20u);
}

TEST(ConSanMoi, RecordReplayBarrierEventsAdvanceOnlyTheirWorkgroup) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/4,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
      /*sampled_watchpoint_capacity=*/0, /*barrier_record_capacity=*/1);
  header.access_record_count = 4;
  header.barrier_record_count = 1;

  std::array<ConSanMoiAccessRecord, 4> records{};
  records[0].workgroup_x = 0;
  records[0].wave_id = 0;
  records[0].event_index = 0;
  records[0].instruction_offset = 0x10;
  records[0].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
  records[0].lds_byte_count = 4;
  records[0].cell_count = 1;

  records[1].workgroup_x = 1;
  records[1].wave_id = 0;
  records[1].event_index = 0;
  records[1].instruction_offset = 0x30;
  records[1].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
  records[1].lds_byte_count = 4;
  records[1].cell_count = 1;

  records[2].workgroup_x = 0;
  records[2].wave_id = 1;
  records[2].event_index = 2;
  records[2].instruction_offset = 0x20;
  records[2].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read);
  records[2].lds_byte_count = 4;
  records[2].cell_count = 1;

  records[3].workgroup_x = 1;
  records[3].wave_id = 1;
  records[3].event_index = 3;
  records[3].instruction_offset = 0x40;
  records[3].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read);
  records[3].lds_byte_count = 4;
  records[3].cell_count = 1;

  std::array<ConSanMoiBarrierRecord, 1> barriers{};
  barriers[0].workgroup_x = 0;
  barriers[0].wave_id = 0;
  barriers[0].event_index = 1;
  barriers[0].instruction_offset = 0x18;

  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  std::array<uint64_t, 1> shadow{};

  const ConSanMoiRecordReplayResult replay =
      consan_moi_record_replay_access_records(header, records, barriers, diagnostics, shadow);

  EXPECT_TRUE(replay.conflict);
  ASSERT_EQ(header.diagnostic_count, 1u);
  EXPECT_EQ(diagnostics[0].first_instruction_offset, 0x30u);
  EXPECT_EQ(diagnostics[0].second_instruction_offset, 0x40u);
}

TEST(ConSanMoi, RecordReplayCoalescesBarrierRuns) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/2,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
      /*sampled_watchpoint_capacity=*/0, /*barrier_record_capacity=*/2);
  header.access_record_count = 2;
  header.barrier_record_count = 2;

  std::array<ConSanMoiAccessRecord, 2> records{};
  records[0].wave_id = 0;
  records[0].event_index = 0;
  records[0].instruction_offset = 0x10;
  records[0].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
  records[0].lds_byte_count = 4;
  records[0].cell_count = 1;

  records[1].wave_id = 1;
  records[1].event_index = 3;
  records[1].instruction_offset = 0x20;
  records[1].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read);
  records[1].lds_byte_count = 4;
  records[1].cell_count = 1;

  std::array<ConSanMoiBarrierRecord, 2> barriers{};
  barriers[0].wave_id = 0;
  barriers[0].event_index = 1;
  barriers[0].instruction_offset = 0x18;

  barriers[1].wave_id = 0;
  barriers[1].event_index = 2;
  barriers[1].instruction_offset = 0x18;

  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  std::array<uint64_t, 1> shadow{};

  const ConSanMoiRecordReplayResult replay =
      consan_moi_record_replay_access_records(header, records, barriers, diagnostics, shadow);

  EXPECT_FALSE(replay.conflict);
  EXPECT_EQ(replay.processed_access_count, 2u);
  EXPECT_EQ(replay.processed_barrier_count, 2u);
  EXPECT_EQ(header.diagnostic_count, 0u);

  const ConSanMoiExactShadowEntry final = decode_consan_moi_exact_shadow_entry(shadow[0]);
  EXPECT_EQ(final.kind, ConSanMoiShadowAccessKind::Read);
  EXPECT_EQ(final.owner_id, 1u);
  EXPECT_EQ(final.epoch, 1u);
  EXPECT_EQ(final.instruction_offset, 0x20u);
}

TEST(ConSanMoi, RecordReplayMatchesHipMoiExactShadowSeeds) {
  constexpr uint32_t kProducerSite = 0x101;
  constexpr uint32_t kConsumerSite = 0x202;
  constexpr uint32_t kOverflowSite = 0x303;

  auto make_record = [](uint32_t owner, uint32_t epoch, ConSanMoiShadowAccessKind kind,
                        uint32_t lds_byte_offset, uint32_t instruction_offset) {
    ConSanMoiAccessRecord record{};
    record.generation = 7;
    record.wave_id = owner;
    record.epoch = epoch;
    record.lane_mask = uint64_t{1} << owner;
    record.instruction_offset = instruction_offset;
    record.access_kind = static_cast<uint32_t>(kind);
    record.lds_byte_offset = lds_byte_offset;
    record.lds_byte_count = 4;
    return record;
  };

  {
    SCOPED_TRACE("synchronized write/read is ordered by an epoch advance");
    ConSanMoiReportHeader header = make_consan_moi_report_header(
        /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/2,
        /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
        /*sampled_watchpoint_capacity=*/0);
    header.access_record_count = 2;
    std::array<ConSanMoiAccessRecord, 2> records = {
        make_record(/*owner=*/0, /*epoch=*/0, ConSanMoiShadowAccessKind::Write,
                    /*lds_byte_offset=*/0, kProducerSite),
        make_record(/*owner=*/1, /*epoch=*/1, ConSanMoiShadowAccessKind::Read,
                    /*lds_byte_offset=*/0, kConsumerSite),
    };
    std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
    std::array<uint64_t, 1> shadow{};

    const ConSanMoiRecordReplayResult replay =
        consan_moi_record_replay_access_records(header, records, diagnostics, shadow);

    EXPECT_EQ(replay.processed_access_count, 2u);
    EXPECT_FALSE(replay.conflict);
    EXPECT_EQ(header.diagnostic_count, 0u);
    const ConSanMoiExactShadowEntry final = decode_consan_moi_exact_shadow_entry(shadow[0]);
    EXPECT_EQ(final.kind, ConSanMoiShadowAccessKind::Read);
    EXPECT_EQ(final.owner_id, 1u);
    EXPECT_EQ(final.epoch, 1u);
    EXPECT_EQ(final.instruction_offset, kConsumerSite);
  }

  {
    SCOPED_TRACE("same-epoch write/read reports an access conflict");
    ConSanMoiReportHeader header = make_consan_moi_report_header(
        /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/2,
        /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
        /*sampled_watchpoint_capacity=*/0);
    header.access_record_count = 2;
    std::array<ConSanMoiAccessRecord, 2> records = {
        make_record(/*owner=*/0, /*epoch=*/0, ConSanMoiShadowAccessKind::Write,
                    /*lds_byte_offset=*/0, kProducerSite),
        make_record(/*owner=*/1, /*epoch=*/0, ConSanMoiShadowAccessKind::Read,
                    /*lds_byte_offset=*/0, kConsumerSite),
    };
    std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
    std::array<uint64_t, 1> shadow{};

    const ConSanMoiRecordReplayResult replay =
        consan_moi_record_replay_access_records(header, records, diagnostics, shadow);

    EXPECT_TRUE(replay.conflict);
    EXPECT_FALSE(replay.metadata_full);
    ASSERT_EQ(header.diagnostic_count, 1u);
    EXPECT_EQ(diagnostics[0].kind, static_cast<uint32_t>(ConSanMoiDiagnosticKind::AccessConflict));
    EXPECT_EQ(diagnostics[0].first_instruction_offset, kProducerSite);
    EXPECT_EQ(diagnostics[0].second_instruction_offset, kConsumerSite);
    EXPECT_EQ(diagnostics[0].first_owner_id, 0u);
    EXPECT_EQ(diagnostics[0].second_owner_id, 1u);
  }

  {
    SCOPED_TRACE("out-of-range offset reports metadata saturation");
    ConSanMoiReportHeader header = make_consan_moi_report_header(
        /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/1,
        /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
        /*sampled_watchpoint_capacity=*/0);
    header.access_record_count = 1;
    std::array<ConSanMoiAccessRecord, 1> records = {
        make_record(/*owner=*/0, /*epoch=*/0, ConSanMoiShadowAccessKind::Write,
                    /*lds_byte_offset=*/8, kOverflowSite),
    };
    std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
    std::array<uint64_t, 1> shadow{};

    const ConSanMoiRecordReplayResult replay =
        consan_moi_record_replay_access_records(header, records, diagnostics, shadow);

    EXPECT_TRUE(replay.conflict);
    EXPECT_TRUE(replay.metadata_full);
    ASSERT_EQ(header.diagnostic_count, 1u);
    EXPECT_EQ(diagnostics[0].kind, static_cast<uint32_t>(ConSanMoiDiagnosticKind::MetadataFull));
    EXPECT_EQ(diagnostics[0].second_instruction_offset, kOverflowSite);
    EXPECT_EQ(diagnostics[0].second_lds_byte_offset, 8u);
  }
}

TEST(ConSanMoi, RecordReplayReportsDiagnosticCapacityExhaustion) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/2,
      /*diagnostic_capacity=*/0, /*exact_shadow_entry_capacity=*/1,
      /*sampled_watchpoint_capacity=*/0);
  header.access_record_count = 2;

  std::array<ConSanMoiAccessRecord, 2> records{};
  records[0].wave_id = 1;
  records[0].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
  records[0].lds_byte_count = 4;
  records[0].cell_count = 1;

  records[1].wave_id = 2;
  records[1].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read);
  records[1].lds_byte_count = 4;
  records[1].cell_count = 1;

  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  std::array<uint64_t, 1> shadow{};

  const ConSanMoiRecordReplayResult replay =
      consan_moi_record_replay_access_records(header, records, diagnostics, shadow);

  EXPECT_EQ(replay.processed_access_count, 2u);
  EXPECT_EQ(replay.dropped_access_count, 0u);
  EXPECT_EQ(replay.unsupported_access_count, 0u);
  EXPECT_EQ(replay.emitted_diagnostic_count, 0u);
  EXPECT_TRUE(replay.diagnostic_capacity_exhausted);
  EXPECT_TRUE(replay.conflict);
  EXPECT_EQ(header.diagnostic_count, 0u);
}

TEST(ConSanMoi, RecordReplayReportsDroppedAndUnsupportedAccessRecords) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/3,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
      /*sampled_watchpoint_capacity=*/0);
  header.access_record_count = 3;

  std::array<ConSanMoiAccessRecord, 2> records{};
  records[0].wave_id = 1;
  records[0].access_kind = 0xffffffffu;
  records[0].lds_byte_count = 4;
  records[0].cell_count = 1;

  records[1].wave_id = 2;
  records[1].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
  records[1].lds_byte_count = 4;
  records[1].cell_count = 1;

  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  std::array<uint64_t, 1> shadow{};

  const ConSanMoiRecordReplayResult replay =
      consan_moi_record_replay_access_records(header, records, diagnostics, shadow);

  EXPECT_EQ(replay.processed_access_count, 2u);
  EXPECT_EQ(replay.dropped_access_count, 1u);
  EXPECT_EQ(replay.unsupported_access_count, 1u);
  EXPECT_EQ(replay.emitted_diagnostic_count, 0u);
  EXPECT_FALSE(replay.conflict);
  EXPECT_EQ(header.diagnostic_count, 0u);
  EXPECT_NE(shadow[0], 0u);
}

TEST(ConSanMoi, RecordReplayReportsDroppedBarrierRecords) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/2,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
      /*sampled_watchpoint_capacity=*/0, /*barrier_record_capacity=*/2);
  header.access_record_count = 2;
  header.barrier_record_count = 2;

  std::array<ConSanMoiAccessRecord, 2> records{};
  records[0].wave_id = 1;
  records[0].event_index = 0;
  records[0].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
  records[0].lds_byte_count = 4;
  records[0].cell_count = 1;

  records[1].wave_id = 2;
  records[1].event_index = 3;
  records[1].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read);
  records[1].lds_byte_count = 4;
  records[1].cell_count = 1;

  std::array<ConSanMoiBarrierRecord, 1> visible_barriers{};
  visible_barriers[0].event_index = 1;

  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  std::array<uint64_t, 1> shadow{};

  const ConSanMoiRecordReplayResult replay = consan_moi_record_replay_access_records(
      header, records, visible_barriers, diagnostics, shadow);

  EXPECT_EQ(replay.processed_access_count, 2u);
  EXPECT_EQ(replay.processed_barrier_count, 1u);
  EXPECT_EQ(replay.dropped_access_count, 0u);
  EXPECT_EQ(replay.dropped_barrier_count, 1u);
  EXPECT_FALSE(replay.conflict);
  EXPECT_EQ(header.diagnostic_count, 0u);
}

TEST(ConSanMoi, LdsCellRangesRoundUnalignedBytesToFourByteGranules) {
  constexpr ConSanMoiLdsCellRange byte_0_to_3 = consan_moi_lds_cell_range_for_bytes(0, 4);
  constexpr ConSanMoiLdsCellRange byte_3_to_4 = consan_moi_lds_cell_range_for_bytes(3, 2);
  constexpr ConSanMoiLdsCellRange byte_8_to_11 = consan_moi_lds_cell_range_for_bytes(8, 4);
  constexpr ConSanMoiLdsCellRange adjacent = consan_moi_lds_cell_range_for_bytes(12, 4);

  EXPECT_EQ(byte_0_to_3.start_cell, 0u);
  EXPECT_EQ(byte_0_to_3.cell_count, 1u);
  EXPECT_EQ(byte_3_to_4.start_cell, 0u);
  EXPECT_EQ(byte_3_to_4.cell_count, 2u);
  EXPECT_EQ(byte_8_to_11.start_cell, 2u);
  EXPECT_EQ(byte_8_to_11.cell_count, 1u);
  EXPECT_TRUE(consan_moi_cell_ranges_overlap(byte_3_to_4, byte_0_to_3));
  EXPECT_FALSE(consan_moi_cell_ranges_overlap(byte_8_to_11, adjacent));
}

TEST(ConSanMoi, SampledWatchpointRoundTripsRangeFields) {
  constexpr uint64_t packed =
      pack_consan_moi_sampled_watchpoint_entry(ConSanMoiShadowAccessKind::Read,
                                               /*owner_id=*/0x512,
                                               /*epoch=*/0x4aa,
                                               /*generation=*/0x1abcde,
                                               /*start_cell=*/0x4567,
                                               /*cell_count=*/32,
                                               /*consumed=*/true);
  constexpr ConSanMoiSampledWatchpointEntry decoded =
      decode_consan_moi_sampled_watchpoint_entry(packed);

  EXPECT_TRUE(decoded.valid);
  EXPECT_TRUE(decoded.consumed);
  EXPECT_EQ(decoded.kind, ConSanMoiShadowAccessKind::Read);
  EXPECT_EQ(decoded.owner_id, 0x112u);
  EXPECT_EQ(decoded.epoch, 0xaau);
  EXPECT_EQ(decoded.generation, 0xabcdeu);
  EXPECT_EQ(decoded.start_cell, 0x567u);
  EXPECT_EQ(decoded.cell_count, 32u);
}

TEST(ConSanMoi, SampledWatchpointPublishesPackedAccessRecords) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/2,
      /*diagnostic_capacity=*/0, /*exact_shadow_entry_capacity=*/0,
      /*sampled_watchpoint_capacity=*/2);
  header.access_record_count = 2;

  std::array<ConSanMoiAccessRecord, 2> records{};
  records[0].wave_id = 1;
  records[0].epoch = 3;
  records[0].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
  records[0].lds_byte_offset = 8;
  records[0].lds_byte_count = 4;

  records[1].generation = 9;
  records[1].wave_id = 2;
  records[1].epoch = 4;
  records[1].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read);
  records[1].start_cell = 5;
  records[1].cell_count = 2;

  std::array<uint64_t, 2> sampled{};

  const ConSanMoiSampledPublishResult result =
      consan_moi_sampled_publish_access_records(header, records, sampled);

  EXPECT_EQ(result.processed_access_count, 2u);
  EXPECT_EQ(result.published_entry_count, 2u);
  EXPECT_FALSE(result.sampled_capacity_exhausted);

  const ConSanMoiSampledWatchpointEntry first =
      decode_consan_moi_sampled_watchpoint_entry(sampled[0]);
  EXPECT_TRUE(first.valid);
  EXPECT_EQ(first.kind, ConSanMoiShadowAccessKind::Write);
  EXPECT_EQ(first.owner_id, 1u);
  EXPECT_EQ(first.epoch, 3u);
  EXPECT_EQ(first.generation, 7u);
  EXPECT_EQ(first.start_cell, 2u);
  EXPECT_EQ(first.cell_count, 1u);

  const ConSanMoiSampledWatchpointEntry second =
      decode_consan_moi_sampled_watchpoint_entry(sampled[1]);
  EXPECT_TRUE(second.valid);
  EXPECT_EQ(second.kind, ConSanMoiShadowAccessKind::Read);
  EXPECT_EQ(second.owner_id, 2u);
  EXPECT_EQ(second.epoch, 4u);
  EXPECT_EQ(second.generation, 9u);
  EXPECT_EQ(second.start_cell, 5u);
  EXPECT_EQ(second.cell_count, 2u);
}

TEST(ConSanMoi, SampledWatchpointPublishReportsCapacityExhaustion) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/2,
      /*diagnostic_capacity=*/0, /*exact_shadow_entry_capacity=*/0,
      /*sampled_watchpoint_capacity=*/1);
  header.access_record_count = 2;

  std::array<ConSanMoiAccessRecord, 2> records{};
  records[0].wave_id = 1;
  records[0].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
  records[0].cell_count = 1;
  records[1].wave_id = 2;
  records[1].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read);
  records[1].cell_count = 1;

  std::array<uint64_t, 1> sampled{};

  const ConSanMoiSampledPublishResult result =
      consan_moi_sampled_publish_access_records(header, records, sampled);

  EXPECT_EQ(result.processed_access_count, 2u);
  EXPECT_EQ(result.published_entry_count, 1u);
  EXPECT_TRUE(result.sampled_capacity_exhausted);
}

TEST(ConSanMoi, SampledWatchpointReplayEmitsLowerFidelityConflictDiagnostic) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/0,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/0,
      /*sampled_watchpoint_capacity=*/2);
  std::array<uint64_t, 2> sampled = {
      pack_consan_moi_sampled_watchpoint_entry(ConSanMoiShadowAccessKind::Write,
                                               /*owner_id=*/1, /*epoch=*/3, /*generation=*/7,
                                               /*start_cell=*/2, /*cell_count=*/1),
      pack_consan_moi_sampled_watchpoint_entry(ConSanMoiShadowAccessKind::Read,
                                               /*owner_id=*/2, /*epoch=*/3, /*generation=*/7,
                                               /*start_cell=*/2, /*cell_count=*/1),
  };
  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};

  const ConSanMoiSampledReplayResult replay =
      consan_moi_sampled_replay_entries(header, sampled, diagnostics);

  EXPECT_EQ(replay.processed_entry_count, 2u);
  EXPECT_EQ(replay.emitted_diagnostic_count, 1u);
  EXPECT_FALSE(replay.diagnostic_capacity_exhausted);
  EXPECT_TRUE(replay.conflict);
  ASSERT_EQ(header.diagnostic_count, 1u);
  EXPECT_EQ(diagnostics[0].kind, static_cast<uint32_t>(ConSanMoiDiagnosticKind::AccessConflict));
  EXPECT_EQ(diagnostics[0].backend, static_cast<uint32_t>(ConSanMoiEngine::Sampled));
  EXPECT_EQ(diagnostics[0].generation, 7u);
  EXPECT_EQ(diagnostics[0].epoch, 3u);
  EXPECT_EQ(diagnostics[0].first_owner_id, 1u);
  EXPECT_EQ(diagnostics[0].second_owner_id, 2u);
  EXPECT_EQ(diagnostics[0].first_access_kind,
            static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write));
  EXPECT_EQ(diagnostics[0].second_access_kind,
            static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read));
}

TEST(ConSanMoi, SampledWatchpointReplayDoesNotTreatCleanSnapshotAsProof) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/0,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/0,
      /*sampled_watchpoint_capacity=*/2);
  std::array<uint64_t, 2> sampled = {
      pack_consan_moi_sampled_watchpoint_entry(ConSanMoiShadowAccessKind::Write,
                                               /*owner_id=*/1, /*epoch=*/3, /*generation=*/7,
                                               /*start_cell=*/2, /*cell_count=*/1),
      pack_consan_moi_sampled_watchpoint_entry(ConSanMoiShadowAccessKind::Read,
                                               /*owner_id=*/2, /*epoch=*/3, /*generation=*/7,
                                               /*start_cell=*/3, /*cell_count=*/1),
  };
  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};

  const ConSanMoiSampledReplayResult replay =
      consan_moi_sampled_replay_entries(header, sampled, diagnostics);

  EXPECT_EQ(replay.processed_entry_count, 2u);
  EXPECT_EQ(replay.emitted_diagnostic_count, 0u);
  EXPECT_FALSE(replay.conflict);
  EXPECT_EQ(header.diagnostic_count, 0u);
}

TEST(ConSanMoi, SampledWatchpointReplayDoesNotCompareDifferentWorkgroupPartitions) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/0,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/0,
      /*sampled_watchpoint_capacity=*/2, /*barrier_record_capacity=*/0,
      /*atomic_record_capacity=*/0, /*sampled_slots_per_partition=*/1);
  std::array<uint64_t, 2> sampled = {
      pack_consan_moi_sampled_watchpoint_entry(ConSanMoiShadowAccessKind::Write,
                                               /*owner_id=*/1, /*epoch=*/3, /*generation=*/7,
                                               /*start_cell=*/2, /*cell_count=*/1),
      pack_consan_moi_sampled_watchpoint_entry(ConSanMoiShadowAccessKind::Read,
                                               /*owner_id=*/2, /*epoch=*/3, /*generation=*/7,
                                               /*start_cell=*/2, /*cell_count=*/1),
  };
  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};

  const ConSanMoiSampledReplayResult replay =
      consan_moi_sampled_replay_entries(header, sampled, diagnostics);

  EXPECT_EQ(replay.processed_entry_count, 2u);
  EXPECT_EQ(replay.emitted_diagnostic_count, 0u);
  EXPECT_FALSE(replay.conflict);
  EXPECT_EQ(header.diagnostic_count, 0u);
}

TEST(ConSanMoi, SampledWatchpointReplayIgnoresStaleGenerations) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/8, /*dispatch_id=*/11, /*access_record_capacity=*/0,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/0,
      /*sampled_watchpoint_capacity=*/2);
  std::array<uint64_t, 2> sampled = {
      pack_consan_moi_sampled_watchpoint_entry(ConSanMoiShadowAccessKind::Write,
                                               /*owner_id=*/1, /*epoch=*/3, /*generation=*/7,
                                               /*start_cell=*/2, /*cell_count=*/1),
      pack_consan_moi_sampled_watchpoint_entry(ConSanMoiShadowAccessKind::Read,
                                               /*owner_id=*/2, /*epoch=*/3, /*generation=*/7,
                                               /*start_cell=*/2, /*cell_count=*/1),
  };
  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};

  const ConSanMoiSampledReplayResult replay =
      consan_moi_sampled_replay_entries(header, sampled, diagnostics);

  EXPECT_EQ(replay.processed_entry_count, 2u);
  EXPECT_EQ(replay.emitted_diagnostic_count, 0u);
  EXPECT_FALSE(replay.conflict);
  EXPECT_EQ(header.diagnostic_count, 0u);
}

TEST(ConSanMoi, SampledWatchpointConflictRequiresExactRangeOverlap) {
  constexpr ConSanMoiSampledWatchpointEntry current{
      /*valid=*/true,
      /*consumed=*/false, ConSanMoiShadowAccessKind::Write,
      /*owner_id=*/3,
      /*epoch=*/11,
      /*generation=*/42,
      /*start_cell=*/20,
      /*cell_count=*/4,
  };
  constexpr ConSanMoiSampledWatchpointEntry overlapping_read{
      /*valid=*/true,
      /*consumed=*/false, ConSanMoiShadowAccessKind::Read,
      /*owner_id=*/1,
      /*epoch=*/11,
      /*generation=*/42,
      /*start_cell=*/23,
      /*cell_count=*/3,
  };
  constexpr ConSanMoiSampledWatchpointEntry adjacent_read{
      /*valid=*/true,
      /*consumed=*/false, ConSanMoiShadowAccessKind::Read,
      /*owner_id=*/1,
      /*epoch=*/11,
      /*generation=*/42,
      /*start_cell=*/24,
      /*cell_count=*/2,
  };
  constexpr ConSanMoiSampledWatchpointEntry consumed_read{
      /*valid=*/true,
      /*consumed=*/true, ConSanMoiShadowAccessKind::Read,
      /*owner_id=*/1,
      /*epoch=*/11,
      /*generation=*/42,
      /*start_cell=*/23,
      /*cell_count=*/3,
  };
  constexpr ConSanMoiSampledWatchpointEntry old_generation{
      /*valid=*/true,
      /*consumed=*/false, ConSanMoiShadowAccessKind::Read,
      /*owner_id=*/1,
      /*epoch=*/11,
      /*generation=*/41,
      /*start_cell=*/23,
      /*cell_count=*/3,
  };

  EXPECT_TRUE(consan_moi_sampled_watchpoints_conflict(current, overlapping_read));
  EXPECT_FALSE(consan_moi_sampled_watchpoints_conflict(current, adjacent_read));
  EXPECT_FALSE(consan_moi_sampled_watchpoints_conflict(current, consumed_read));
  EXPECT_FALSE(consan_moi_sampled_watchpoints_conflict(current, old_generation));
}

TEST(ConSan, CountsFlatGlobalAndScratchMemoryInstructions) {
  const std::vector<uint8_t> bytes = make_rdna4_flat_memory_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_FALSE(result.warnings.empty());
  ASSERT_EQ(result.kernels.size(), 1u);

  const ConSanKernelInfo &kernel = result.kernels.front();
  EXPECT_TRUE(kernel.decoded);
  EXPECT_EQ(kernel.code_size, 52u);
  EXPECT_EQ(kernel.stats.instruction_count, 5u);
  EXPECT_EQ(kernel.stats.lds_read_count, 0u);
  EXPECT_EQ(kernel.stats.lds_write_count, 0u);
  EXPECT_EQ(kernel.stats.lds_atomic_count, 0u);
  EXPECT_EQ(kernel.stats.ds_other_count, 0u);
  EXPECT_EQ(kernel.stats.flat_read_count, 1u);
  EXPECT_EQ(kernel.stats.flat_write_count, 1u);
  EXPECT_EQ(kernel.stats.flat_atomic_count, 0u);
  EXPECT_EQ(kernel.stats.flat_group_hint_count, 0u);
  EXPECT_EQ(kernel.stats.flat_private_hint_count, 0u);
  EXPECT_EQ(kernel.stats.flat_maybe_group_hint_count, 0u);
  EXPECT_EQ(kernel.stats.flat_maybe_private_hint_count, 0u);
  EXPECT_EQ(kernel.stats.flat_global_hint_count, 0u);
  EXPECT_EQ(kernel.stats.flat_unknown_hint_count, 2u);
  EXPECT_EQ(kernel.stats.global_memory_count, 1u);
  EXPECT_EQ(kernel.stats.scratch_memory_count, 1u);
  EXPECT_EQ(kernel.preflight_action, ConSanPreflightAction::Skip);
  ASSERT_GE(kernel.preflight_reasons.size(), 4u);
  EXPECT_EQ(kernel.preflight_reasons[0], "no supported non-atomic LDS reads or writes");
  EXPECT_EQ(kernel.preflight_reasons[1], "flat/generic memory instructions observed: 2");
  EXPECT_EQ(kernel.preflight_reasons[2], "global memory instructions observed: 1");
  EXPECT_EQ(kernel.preflight_reasons[3], "scratch memory instructions observed: 1");
  EXPECT_TRUE(kernel.lds_sites.empty());
  ASSERT_EQ(kernel.flat_sites.size(), 2u);
  EXPECT_EQ(kernel.flat_sites[0].kind, ConSanLdsAccessKind::Read);
  EXPECT_EQ(kernel.flat_sites[0].mnemonic, "flat_load_b32");
  EXPECT_EQ(kernel.flat_sites[0].text_offset, 0u);
  EXPECT_EQ(kernel.flat_sites[0].file_offset, 0x100u);
  EXPECT_EQ(kernel.flat_sites[0].size, 12u);
  EXPECT_EQ(kernel.flat_sites[0].width_bits, 32u);
  EXPECT_EQ(kernel.flat_sites[0].address_space_hint, ConSanFlatAddressSpaceHint::Unknown);
  ASSERT_TRUE(kernel.flat_sites[0].dst_vgpr);
  ASSERT_TRUE(kernel.flat_sites[0].addr_vgpr);
  EXPECT_FALSE(kernel.flat_sites[0].data_vgpr);
  EXPECT_EQ(*kernel.flat_sites[0].dst_vgpr, 0u);
  EXPECT_LT(*kernel.flat_sites[0].addr_vgpr, 256u);
  ASSERT_TRUE(kernel.flat_sites[0].raw_saddr);
  ASSERT_TRUE(kernel.flat_sites[0].raw_vaddr);
  ASSERT_TRUE(kernel.flat_sites[0].raw_vdst);
  ASSERT_TRUE(kernel.flat_sites[0].raw_ioffset);
  EXPECT_EQ(*kernel.flat_sites[0].raw_saddr, 0u);
  EXPECT_EQ(*kernel.flat_sites[0].raw_vdst, 0u);
  EXPECT_EQ(*kernel.flat_sites[0].raw_ioffset, 0);
  EXPECT_EQ(kernel.flat_sites[1].kind, ConSanLdsAccessKind::Write);
  EXPECT_EQ(kernel.flat_sites[1].mnemonic, "flat_store_b32");
  EXPECT_EQ(kernel.flat_sites[1].text_offset, 12u);
  EXPECT_EQ(kernel.flat_sites[1].file_offset, 0x10cu);
  EXPECT_EQ(kernel.flat_sites[1].size, 12u);
  EXPECT_EQ(kernel.flat_sites[1].width_bits, 32u);
  EXPECT_EQ(kernel.flat_sites[1].address_space_hint, ConSanFlatAddressSpaceHint::Unknown);
  EXPECT_FALSE(kernel.flat_sites[1].dst_vgpr);
  ASSERT_TRUE(kernel.flat_sites[1].addr_vgpr);
  ASSERT_TRUE(kernel.flat_sites[1].data_vgpr);
  EXPECT_LT(*kernel.flat_sites[1].addr_vgpr, 256u);
  EXPECT_EQ(*kernel.flat_sites[1].data_vgpr, 0u);
  ASSERT_TRUE(kernel.flat_sites[1].raw_saddr);
  ASSERT_TRUE(kernel.flat_sites[1].raw_vaddr);
  ASSERT_TRUE(kernel.flat_sites[1].raw_vsrc);
  ASSERT_TRUE(kernel.flat_sites[1].raw_ioffset);
  EXPECT_EQ(*kernel.flat_sites[1].raw_saddr, 0u);
  EXPECT_EQ(*kernel.flat_sites[1].raw_vsrc, 0u);
  EXPECT_EQ(*kernel.flat_sites[1].raw_ioffset, 0);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
}

TEST(ConSan, ClassifiesObviousSharedBaseFlatLoad) {
  const std::array<uint32_t, 9> text_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000000u,              // v_mov_b32_e64 v0, s0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC05007Cu, 0x00000002u, 0x00000000u, // flat_load_b32 v2, v[0:1]
      0xBFB00000u,                           // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_FALSE(result.warnings.empty());
  ASSERT_EQ(result.kernels.size(), 1u);

  const ConSanKernelInfo &kernel = result.kernels.front();
  EXPECT_TRUE(kernel.decoded);
  EXPECT_EQ(kernel.code_size, 36u);
  EXPECT_EQ(kernel.stats.instruction_count, 5u);
  EXPECT_EQ(kernel.stats.flat_read_count, 1u);
  EXPECT_EQ(kernel.stats.flat_write_count, 0u);
  EXPECT_EQ(kernel.stats.flat_group_hint_count, 1u);
  EXPECT_EQ(kernel.stats.flat_maybe_group_hint_count, 0u);
  EXPECT_EQ(kernel.stats.flat_private_hint_count, 0u);
  EXPECT_EQ(kernel.stats.flat_unknown_hint_count, 0u);
  ASSERT_EQ(kernel.flat_sites.size(), 1u);
  EXPECT_EQ(kernel.flat_sites.front().kind, ConSanLdsAccessKind::Read);
  EXPECT_EQ(kernel.flat_sites.front().mnemonic, "flat_load_b32");
  EXPECT_EQ(kernel.flat_sites.front().address_space_hint, ConSanFlatAddressSpaceHint::Group);
  ASSERT_TRUE(kernel.flat_sites.front().addr_vgpr);
  EXPECT_EQ(*kernel.flat_sites.front().addr_vgpr, 0u);
  ASSERT_TRUE(kernel.flat_sites.front().dst_vgpr);
  EXPECT_EQ(*kernel.flat_sites.front().dst_vgpr, 2u);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
}

TEST(ConSan, ClassifiesHighHalfSharedBaseFlatLoadAsMaybeGroup) {
  const std::array<uint32_t, 9> text_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000080u,              // v_mov_b32_e64 v0, 0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC05007Cu, 0x00000002u, 0x00000000u, // flat_load_b32 v2, v[0:1]
      0xBFB00000u,                           // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_FALSE(result.warnings.empty());
  ASSERT_EQ(result.kernels.size(), 1u);

  const ConSanKernelInfo &kernel = result.kernels.front();
  EXPECT_EQ(kernel.stats.flat_read_count, 1u);
  EXPECT_EQ(kernel.stats.flat_group_hint_count, 0u);
  EXPECT_EQ(kernel.stats.flat_maybe_group_hint_count, 1u);
  EXPECT_EQ(kernel.stats.flat_unknown_hint_count, 0u);
  ASSERT_EQ(kernel.flat_sites.size(), 1u);
  EXPECT_EQ(kernel.flat_sites.front().address_space_hint, ConSanFlatAddressSpaceHint::MaybeGroup);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
}

TEST(ConSanMoi, StrictFlatProvenanceExcludesMaybeGroupCandidates) {
  const std::array<uint32_t, 9> text_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000080u,              // v_mov_b32_e64 v0, 0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC05007Cu, 0x00000002u, 0x00000000u, // flat_load_b32 v2, v[0:1]
      0xBFB00000u,                           // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);

  ConSanOptions likely_options;
  likely_options.flavor = ConSanFlavor::Moi;
  const auto likely_result = try_patch_consan(bytes, likely_options);
  ASSERT_TRUE(likely_result.errors.empty());
  ASSERT_EQ(likely_result.moi_candidates.size(), 1u);
  EXPECT_EQ(likely_result.moi_candidates.front().source, ConSanMoiCandidateSource::FlatMaybeGroup);

  ConSanOptions strict_options = likely_options;
  strict_options.flat_provenance_mode = ConSanFlatProvenanceMode::Strict;
  const auto strict_result = try_patch_consan(bytes, strict_options);
  ASSERT_TRUE(strict_result.errors.empty());
  EXPECT_TRUE(strict_result.moi_candidates.empty());
  EXPECT_TRUE(std::ranges::any_of(strict_result.warnings, [](const std::string &warning) {
    return warning.find("excluded_maybe_group=1") != std::string::npos;
  }));
}

TEST(ConSan, PropagatesSharedBaseThroughVectorAddCarryAddressConstruction) {
  const std::array<uint32_t, 11> text_words = {
      0xBE8E01EBu,                           // s_mov_b64 s[14:15], src_shared_base
      0xBE81000Fu,                           // s_mov_b32 s1, s15
      0xD5810000u, 0x00000080u,              // v_mov_b32_e64 v0, 0
      0xD5200100u, 0x00220001u,              // v_add_co_ci_u32_e64 v0, s1, s1, v0, s8
      0x7E020300u,                           // v_mov_b32_e32 v1, v0
      0xEC05007Cu, 0x00000002u, 0x00000000u, // flat_load_b32 v2, v[0:1]
      0xBFB00000u,                           // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_EQ(result.kernels.size(), 1u);

  const ConSanKernelInfo &kernel = result.kernels.front();
  EXPECT_TRUE(kernel.decoded);
  EXPECT_EQ(kernel.stats.flat_read_count, 1u);
  EXPECT_EQ(kernel.stats.flat_group_hint_count, 0u);
  EXPECT_EQ(kernel.stats.flat_maybe_group_hint_count, 1u);
  EXPECT_EQ(kernel.stats.flat_unknown_hint_count, 0u);
  ASSERT_EQ(kernel.flat_sites.size(), 1u);
  EXPECT_EQ(kernel.flat_sites.front().address_space_hint, ConSanFlatAddressSpaceHint::MaybeGroup);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
}

TEST(ConSan, InventoriesLocalFunctionFlatSharedAccesses) {
  const std::array<uint32_t, 1> kernel_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 9> function_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000000u,              // v_mov_b32_e64 v0, s0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC05007Cu, 0x00000002u, 0x00000000u, // flat_load_b32 v2, v[0:1]
      0xBFB00000u,                           // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_EQ(result.kernels.size(), 1u);
  EXPECT_EQ(result.kernels.front().name, "lds_probe");
  EXPECT_EQ(result.kernels.front().code_size, 4u);
  EXPECT_EQ(result.kernels.front().stats.flat_group_hint_count, 0u);
  ASSERT_EQ(result.functions.size(), 1u);

  const ConSanFunctionInfo &function = result.functions.front();
  EXPECT_EQ(function.name, "lds_helper");
  EXPECT_TRUE(function.decoded);
  EXPECT_EQ(function.entry_text_offset, 4u);
  EXPECT_EQ(function.text_file_offset, 0x100u);
  EXPECT_EQ(function.code_size, 36u);
  EXPECT_EQ(function.stats.instruction_count, 5u);
  EXPECT_EQ(function.stats.flat_read_count, 1u);
  EXPECT_EQ(function.stats.flat_group_hint_count, 1u);
  EXPECT_EQ(function.stats.flat_unknown_hint_count, 0u);
  ASSERT_EQ(function.flat_sites.size(), 1u);
  EXPECT_EQ(function.flat_sites.front().address_space_hint, ConSanFlatAddressSpaceHint::Group);
  EXPECT_EQ(function.flat_sites.front().text_offset, 24u);
  EXPECT_EQ(function.flat_sites.front().file_offset, 0x118u);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
}

TEST(ConSan, FlatTrapProofRewritesLikelyGroupLocalFunctionSite) {
  const std::array<uint32_t, 1> kernel_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 9> function_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000000u,              // v_mov_b32_e64 v0, s0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC05007Cu, 0x00000002u, 0x00000000u, // flat_load_b32 v2, v[0:1]
      0xBFB00000u,                           // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_flat_trap = true;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineFlatTrapRewrite);
  EXPECT_EQ(result.patches.front().anchor_offset, 24u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 36u);
  EXPECT_EQ(result.patches.front().original_size, 12u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());
  EXPECT_NE(result.elf_bytes, bytes);

  std::array<uint32_t, 3> rewritten_words{};
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + 0x118,
              rewritten_words.size() * sizeof(uint32_t));
  EXPECT_EQ(rewritten_words[0], 0xBF900000u); // s_trap 0
  EXPECT_EQ(rewritten_words[1], 0xBF800000u); // s_nop 0
  EXPECT_EQ(rewritten_words[2], 0xBF800000u); // s_nop 0
}

TEST(ConSan, FlatCheckTrapProofReportsUnpaddedCandidateCounts) {
  const std::array<uint32_t, 1> kernel_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 9> function_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000000u,              // v_mov_b32_e64 v0, s0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC05007Cu, 0x00000002u, 0x00000000u, // flat_load_b32 v2, v[0:1]
      0xBFB00000u,                           // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_flat_check_trap = true;
  options.delay_nops = 1;
  options.scratch_vgpr = 5;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.patches.empty());

  bool saw_padding_counts = false;
  for (const std::string &warning : result.warnings) {
    saw_padding_counts |= warning.find("supported_candidates=1") != std::string::npos &&
                          warning.find("scratchable_candidates=1") != std::string::npos &&
                          warning.find("max_observed_padding_words=0") != std::string::npos &&
                          warning.find("append_cave_reachable_candidates=1") != std::string::npos;
  }
  EXPECT_TRUE(saw_padding_counts);
}

TEST(ConSan, FlatCheckTrapProofUsesReachableUncoveredNopCave) {
  const std::array<uint32_t, 1> kernel_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 9> function_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000000u,              // v_mov_b32_e64 v0, s0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC05007Cu, 0x00000002u, 0x00000000u, // flat_load_b32 v2, v[0:1]
      0xBFB00000u,                           // s_endpgm
  };
  const std::array<uint32_t, 12> tail_words = {
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words, tail_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_flat_check_trap = true;
  options.delay_nops = 1;
  options.scratch_vgpr = 5;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::LocalCaveFlatLoadCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 24u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 40u);
  EXPECT_EQ(result.patches.front().original_size, 12u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 5u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());
  EXPECT_NE(result.elf_bytes, bytes);

  const std::array<uint32_t, 3> expected_anchor = {
      build_s_branch(3, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
  };
  std::array<uint32_t, expected_anchor.size()> anchor_words{};
  std::memcpy(anchor_words.data(), result.elf_bytes.data() + 0x118,
              anchor_words.size() * sizeof(uint32_t));
  EXPECT_EQ(anchor_words, expected_anchor);

  const std::array<uint32_t, 12> expected_cave = {
      0xEC05007Cu,
      0x00000002u,
      0x00000000u, // original flat_load_b32 v2, v[0:1]
      0xBF800000u, // delay
      0xEC05007Cu,
      0x00000005u,
      0x00000000u, // duplicate flat_load_b32 v5, v[0:1]
      0xBFC60000u, // s_wait_dscnt 0
      0x7C9A0B02u, // v_cmp_ne_u32_e32 v2, v5
      0xBFA30001u, // s_cbranch_vccz +1
      0xBF900000u, // s_trap 0
      build_s_branch(-13, ROCJITSU_CODE_ARCH_RDNA4),
  };
  std::array<uint32_t, expected_cave.size()> cave_words{};
  std::memcpy(cave_words.data(), result.elf_bytes.data() + 0x128,
              cave_words.size() * sizeof(uint32_t));
  EXPECT_EQ(cave_words, expected_cave);
}

TEST(ConSan, FlatLoadCheckTrapProofRewritesPaddedLocalFunctionSite) {
  const std::array<uint32_t, 1> kernel_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 17> function_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000000u,              // v_mov_b32_e64 v0, s0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC05007Cu, 0x00000002u, 0x00000000u, // flat_load_b32 v2, v[0:1]
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_flat_check_trap = true;
  options.delay_nops = 1;
  options.scratch_vgpr = 5;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineFlatLoadCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 24u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 36u);
  EXPECT_EQ(result.patches.front().original_size, 44u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 5u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());
  EXPECT_NE(result.elf_bytes, bytes);

  const std::array<uint32_t, 11> expected_words = {
      0xEC05007Cu, 0x00000002u, 0x00000000u, // original flat_load_b32 v2, v[0:1]
      0xBF800000u,                           // delay
      0xEC05007Cu, 0x00000005u, 0x00000000u, // duplicate flat_load_b32 v5, v[0:1]
      0xBFC60000u,                           // s_wait_dscnt 0
      0x7C9A0B02u,                           // v_cmp_ne_u32_e32 v2, v5
      0xBFA30001u,                           // s_cbranch_vccz +1
      0xBF900000u,                           // s_trap 0
  };
  std::array<uint32_t, expected_words.size()> rewritten_words{};
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + 0x118,
              rewritten_words.size() * sizeof(uint32_t));
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(ConSan, CombinedCheckTrapFallsBackToFlatWhenNoNativeLdsPatchApplies) {
  const std::array<uint32_t, 1> kernel_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 17> function_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000000u,              // v_mov_b32_e64 v0, s0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC05007Cu, 0x00000002u, 0x00000000u, // flat_load_b32 v2, v[0:1]
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.probe_flat_check_trap = true;
  options.delay_nops = 1;
  options.scratch_vgpr = 5;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineFlatLoadCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 24u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 36u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 5u);
}

TEST(ConSan, CombinedCheckTrapCanPatchNativeLdsAndFlatInSameCodeObject) {
  const std::array<uint32_t, 13> kernel_words = {
      0xD8340000u,
      0x00000102u, // ds_store_b32 v2, v1
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 17> function_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000000u,              // v_mov_b32_e64 v0, s0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC05007Cu, 0x00000002u, 0x00000000u, // flat_load_b32 v2, v[0:1]
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.probe_flat_check_trap = true;
  options.delay_nops = 1;
  options.scratch_vgpr = 3;
  options.max_patches = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 2u);
  EXPECT_EQ(result.patches[0].kind, ConSanPatchKind::InlineLdsStoreCheckTrap);
  EXPECT_EQ(result.patches[0].anchor_offset, 0u);
  EXPECT_EQ(result.patches[0].trampoline_offset, 8u);
  EXPECT_EQ(result.patches[0].original_size, 44u);
  EXPECT_EQ(result.patches[0].trampoline_size, 0u);
  ASSERT_TRUE(result.patches[0].scratch_vgpr);
  EXPECT_EQ(*result.patches[0].scratch_vgpr, 3u);
  EXPECT_EQ(result.patches[1].kind, ConSanPatchKind::InlineFlatLoadCheckTrap);
  EXPECT_EQ(result.patches[1].anchor_offset, 72u);
  EXPECT_EQ(result.patches[1].trampoline_offset, 84u);
  EXPECT_EQ(result.patches[1].original_size, 44u);
  EXPECT_EQ(result.patches[1].trampoline_size, 0u);
  ASSERT_TRUE(result.patches[1].scratch_vgpr);
  EXPECT_EQ(*result.patches[1].scratch_vgpr, 3u);
}

TEST(ConSan, FlatCheckTrapProofCanPatchMultiplePaddedLoads) {
  const std::array<uint32_t, 1> kernel_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 28> function_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000000u,              // v_mov_b32_e64 v0, s0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC05007Cu, 0x00000002u, 0x00000000u, // flat_load_b32 v2, v[0:1]
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xEC05007Cu, 0x00000006u, 0x00000000u, // flat_load_b32 v6, v[0:1]
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_flat_check_trap = true;
  options.delay_nops = 1;
  options.scratch_vgpr = 5;
  options.max_patches = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 2u);
  EXPECT_EQ(result.patches[0].kind, ConSanPatchKind::InlineFlatLoadCheckTrap);
  EXPECT_EQ(result.patches[0].anchor_offset, 24u);
  EXPECT_EQ(result.patches[0].trampoline_offset, 36u);
  EXPECT_EQ(result.patches[0].original_size, 44u);
  ASSERT_TRUE(result.patches[0].scratch_vgpr);
  EXPECT_EQ(*result.patches[0].scratch_vgpr, 5u);
  EXPECT_EQ(result.patches[1].kind, ConSanPatchKind::InlineFlatLoadCheckTrap);
  EXPECT_EQ(result.patches[1].anchor_offset, 68u);
  EXPECT_EQ(result.patches[1].trampoline_offset, 80u);
  EXPECT_EQ(result.patches[1].original_size, 44u);
  ASSERT_TRUE(result.patches[1].scratch_vgpr);
  EXPECT_EQ(*result.patches[1].scratch_vgpr, 5u);
}

TEST(ConSan, FlatStoreCheckTrapProofRewritesPaddedLocalFunctionSite) {
  const std::array<uint32_t, 1> kernel_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 17> function_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000000u,              // v_mov_b32_e64 v0, s0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC06807Cu, 0x01000000u, 0x00000000u, // flat_store_b32 v[0:1], v2
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_flat_check_trap = true;
  options.delay_nops = 1;
  options.scratch_vgpr = 5;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineFlatStoreCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 24u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 36u);
  EXPECT_EQ(result.patches.front().original_size, 44u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 5u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());
  EXPECT_NE(result.elf_bytes, bytes);

  const std::array<uint32_t, 11> expected_words = {
      0xEC06807Cu, 0x01000000u, 0x00000000u, // original flat_store_b32 v[0:1], v2
      0xBF800000u,                           // delay
      0xEC05007Cu, 0x00000005u, 0x00000000u, // readback flat_load_b32 v5, v[0:1]
      0xBFC60000u,                           // s_wait_dscnt 0
      0x7C9A0B02u,                           // v_cmp_ne_u32_e32 v2, v5
      0xBFA30001u,                           // s_cbranch_vccz +1
      0xBF900000u,                           // s_trap 0
  };
  std::array<uint32_t, expected_words.size()> rewritten_words{};
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + 0x118,
              rewritten_words.size() * sizeof(uint32_t));
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(ConSan, FlatStoreCheckTrapProofCanUseSleepDelay) {
  const std::array<uint32_t, 1> kernel_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 17> function_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000000u,              // v_mov_b32_e64 v0, s0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC06807Cu, 0x01000000u, 0x00000000u, // flat_store_b32 v[0:1], v2
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_flat_check_trap = true;
  options.delay_mode = ConSanDelayMode::Sleep;
  options.delay_nops = 9;
  options.scratch_vgpr = 5;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineFlatStoreCheckTrap);
  EXPECT_EQ(result.patches.front().original_size, 44u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const std::array<uint32_t, 11> expected_words = {
      0xEC06807Cu,
      0x01000000u,
      0x00000000u, // original flat_store_b32 v[0:1], v2
      build_s_sleep(9, ROCJITSU_CODE_ARCH_RDNA4),
      0xEC05007Cu,
      0x00000005u,
      0x00000000u, // readback flat_load_b32 v5, v[0:1]
      0xBFC60000u, // s_wait_dscnt 0
      0x7C9A0B02u, // v_cmp_ne_u32_e32 v2, v5
      0xBFA30001u, // s_cbranch_vccz +1
      0xBF900000u, // s_trap 0
  };
  std::array<uint32_t, expected_words.size()> rewritten_words{};
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + 0x118,
              rewritten_words.size() * sizeof(uint32_t));
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(ConSan, FlatStoreCheckTrapProofCanUseSleepVarDelay) {
  const std::array<uint32_t, 1> kernel_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 17> function_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000000u,              // v_mov_b32_e64 v0, s0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC06807Cu, 0x01000000u, 0x00000000u, // flat_store_b32 v[0:1], v2
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_flat_check_trap = true;
  options.delay_mode = ConSanDelayMode::SleepVar;
  options.delay_nops = 1;
  options.delay_var_ssrc = kRdna4VccLo;
  options.scratch_vgpr = 5;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineFlatStoreCheckTrap);
  EXPECT_EQ(result.patches.front().original_size, 44u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const std::array<uint32_t, 11> expected_words = {
      0xEC06807Cu,
      0x01000000u,
      0x00000000u, // original flat_store_b32 v[0:1], v2
      build_s_sleep_var(kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0xEC05007Cu,
      0x00000005u,
      0x00000000u, // readback flat_load_b32 v5, v[0:1]
      0xBFC60000u, // s_wait_dscnt 0
      0x7C9A0B02u, // v_cmp_ne_u32_e32 v2, v5
      0xBFA30001u, // s_cbranch_vccz +1
      0xBF900000u, // s_trap 0
  };
  std::array<uint32_t, expected_words.size()> rewritten_words{};
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + 0x118,
              rewritten_words.size() * sizeof(uint32_t));
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(ConSan, CountsRdna4LdsAndSynchronizationInstructions) {
  const std::vector<uint8_t> bytes = make_rdna4_unsupported_lds_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_FALSE(result.warnings.empty());
  ASSERT_EQ(result.kernels.size(), 1u);
  EXPECT_EQ(result.target_name, "gfx1201");
  EXPECT_EQ(result.arch_name, "rdna4");

  const ConSanKernelInfo &kernel = result.kernels.front();
  EXPECT_EQ(kernel.name, "lds_probe");
  EXPECT_TRUE(kernel.has_text_range);
  EXPECT_TRUE(kernel.decoded);
  EXPECT_EQ(kernel.code_size, 44u);
  EXPECT_EQ(kernel.stats.instruction_count, 7u);
  EXPECT_EQ(kernel.stats.lds_read_count, 1u);
  EXPECT_EQ(kernel.stats.lds_write_count, 1u);
  EXPECT_EQ(kernel.stats.lds_atomic_count, 1u);
  EXPECT_EQ(kernel.stats.ds_other_count, 0u);
  EXPECT_EQ(kernel.stats.barrier_count, 1u);
  EXPECT_EQ(kernel.stats.wait_count, 1u);
  EXPECT_EQ(kernel.stats.fence_like_count, 1u);
  EXPECT_EQ(kernel.stats.decode_error_count, 0u);
  ASSERT_EQ(kernel.lds_sites.size(), 3u);
  EXPECT_EQ(kernel.lds_sites[0].kind, ConSanLdsAccessKind::Write);
  EXPECT_TRUE(kernel.lds_sites[0].supported_mvp);
  EXPECT_EQ(kernel.lds_sites[0].mnemonic, "ds_store_b32");
  EXPECT_EQ(kernel.lds_sites[0].text_offset, 0u);
  EXPECT_EQ(kernel.lds_sites[0].file_offset, 0x100u);
  EXPECT_EQ(kernel.lds_sites[0].size, 8u);
  EXPECT_EQ(kernel.lds_sites[0].width_bits, 32u);
  ASSERT_TRUE(kernel.lds_sites[0].addr_vgpr);
  ASSERT_TRUE(kernel.lds_sites[0].data_vgpr);
  EXPECT_FALSE(kernel.lds_sites[0].dst_vgpr);
  EXPECT_EQ(*kernel.lds_sites[0].addr_vgpr, 0u);
  EXPECT_EQ(*kernel.lds_sites[0].data_vgpr, 0u);
  EXPECT_EQ(kernel.lds_sites[1].kind, ConSanLdsAccessKind::Read);
  EXPECT_TRUE(kernel.lds_sites[1].supported_mvp);
  EXPECT_EQ(kernel.lds_sites[1].mnemonic, "ds_load_b32");
  EXPECT_EQ(kernel.lds_sites[1].text_offset, 8u);
  EXPECT_EQ(kernel.lds_sites[1].file_offset, 0x108u);
  EXPECT_EQ(kernel.lds_sites[1].size, 8u);
  EXPECT_EQ(kernel.lds_sites[1].width_bits, 32u);
  ASSERT_TRUE(kernel.lds_sites[1].dst_vgpr);
  ASSERT_TRUE(kernel.lds_sites[1].addr_vgpr);
  EXPECT_FALSE(kernel.lds_sites[1].data_vgpr);
  EXPECT_EQ(*kernel.lds_sites[1].dst_vgpr, 0u);
  EXPECT_EQ(*kernel.lds_sites[1].addr_vgpr, 0u);
  EXPECT_EQ(kernel.lds_sites[2].kind, ConSanLdsAccessKind::Atomic);
  EXPECT_FALSE(kernel.lds_sites[2].supported_mvp);
  EXPECT_EQ(kernel.lds_sites[2].mnemonic, "ds_add_u32");
  EXPECT_EQ(kernel.lds_sites[2].text_offset, 16u);
  EXPECT_EQ(kernel.lds_sites[2].file_offset, 0x110u);
  EXPECT_EQ(kernel.lds_sites[2].size, 8u);
  EXPECT_EQ(kernel.lds_sites[2].width_bits, 32u);
  ASSERT_TRUE(kernel.lds_sites[2].addr_vgpr);
  ASSERT_TRUE(kernel.lds_sites[2].data_vgpr);
  EXPECT_EQ(*kernel.lds_sites[2].addr_vgpr, 0u);
  EXPECT_EQ(*kernel.lds_sites[2].data_vgpr, 0u);
  ASSERT_EQ(kernel.barrier_sites.size(), 1u);
  EXPECT_EQ(kernel.barrier_sites[0].mnemonic, "s_barrier_wait");
  EXPECT_EQ(kernel.barrier_sites[0].text_offset, 24u);
  ASSERT_EQ(kernel.fence_sites.size(), 1u);
  EXPECT_EQ(kernel.fence_sites[0].mnemonic, "s_dcache_inv");
  EXPECT_EQ(kernel.fence_sites[0].text_offset, 32u);
  EXPECT_EQ(kernel.fence_sites[0].file_offset, 0x120u);
  EXPECT_EQ(kernel.fence_sites[0].size, 8u);
  ASSERT_EQ(kernel.atomic_sites.size(), 1u);
  const ConSanAtomicSite &atomic = kernel.atomic_sites.front();
  EXPECT_EQ(atomic.address_space_hint, ConSanAtomicAddressSpaceHint::Lds);
  EXPECT_EQ(atomic.mnemonic, "ds_add_u32");
  EXPECT_EQ(atomic.text_offset, 16u);
  EXPECT_EQ(atomic.file_offset, 0x110u);
  EXPECT_EQ(atomic.size, 8u);
  EXPECT_EQ(atomic.width_bits, 32u);
  ASSERT_TRUE(atomic.addr_vgpr);
  ASSERT_TRUE(atomic.data_vgpr);
  EXPECT_EQ(*atomic.addr_vgpr, 0u);
  EXPECT_EQ(*atomic.data_vgpr, 0u);
  ASSERT_TRUE(atomic.raw_op);
  ASSERT_TRUE(atomic.raw_addr);
  ASSERT_TRUE(atomic.raw_data0);
  ASSERT_TRUE(atomic.raw_data1);
  ASSERT_TRUE(atomic.raw_vdst);
  EXPECT_EQ(*atomic.raw_op, 0u);
  EXPECT_EQ(*atomic.raw_addr, 0u);
  EXPECT_EQ(*atomic.raw_data0, 0u);
  EXPECT_EQ(*atomic.raw_data1, 0u);
  EXPECT_EQ(*atomic.raw_vdst, 0u);
  EXPECT_FALSE(atomic.raw_scope);
  EXPECT_FALSE(atomic.raw_th);
  ASSERT_TRUE(atomic.returns_old_value);
  EXPECT_FALSE(*atomic.returns_old_value);
  EXPECT_EQ(kernel.preflight_action, ConSanPreflightAction::Skip);
  ASSERT_GE(kernel.preflight_reasons.size(), 2u);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
}

TEST(ConSan, InventoriesRdna4GlobalAtomicScopeAndReturnBits) {
  const std::vector<uint8_t> bytes = make_rdna4_global_atomic_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_EQ(result.kernels.size(), 1u);

  const ConSanKernelInfo &kernel = result.kernels.front();
  EXPECT_TRUE(kernel.decoded);
  EXPECT_EQ(kernel.code_size, 16u);
  EXPECT_EQ(kernel.stats.instruction_count, 2u);
  EXPECT_EQ(kernel.stats.global_memory_count, 1u);
  EXPECT_EQ(kernel.stats.lds_atomic_count, 0u);
  EXPECT_TRUE(kernel.lds_sites.empty());
  EXPECT_TRUE(kernel.fence_sites.empty());
  ASSERT_EQ(kernel.atomic_sites.size(), 1u);

  const ConSanAtomicSite &atomic = kernel.atomic_sites.front();
  EXPECT_EQ(atomic.address_space_hint, ConSanAtomicAddressSpaceHint::Global);
  EXPECT_EQ(atomic.mnemonic, "global_atomic_add_f32");
  EXPECT_EQ(atomic.text_offset, 0u);
  EXPECT_EQ(atomic.file_offset, 0x100u);
  EXPECT_EQ(atomic.size, 12u);
  EXPECT_EQ(atomic.width_bits, 32u);
  ASSERT_TRUE(atomic.dst_vgpr);
  ASSERT_TRUE(atomic.addr_vgpr);
  ASSERT_TRUE(atomic.data_vgpr);
  ASSERT_TRUE(atomic.saddr_sgpr);
  EXPECT_EQ(*atomic.dst_vgpr, 0u);
  EXPECT_EQ(*atomic.addr_vgpr, 2u);
  EXPECT_EQ(*atomic.data_vgpr, 1u);
  EXPECT_EQ(*atomic.saddr_sgpr, 4u);
  ASSERT_TRUE(atomic.raw_saddr);
  ASSERT_TRUE(atomic.raw_vaddr);
  ASSERT_TRUE(atomic.raw_vsrc);
  ASSERT_TRUE(atomic.raw_vdst);
  ASSERT_TRUE(atomic.raw_ioffset);
  ASSERT_TRUE(atomic.raw_scope);
  ASSERT_TRUE(atomic.raw_th);
  ASSERT_TRUE(atomic.returns_old_value);
  EXPECT_EQ(*atomic.raw_saddr, 4u);
  EXPECT_EQ(*atomic.raw_vaddr, 2u);
  EXPECT_EQ(*atomic.raw_vsrc, 1u);
  EXPECT_EQ(*atomic.raw_vdst, 0u);
  EXPECT_EQ(*atomic.raw_ioffset, 0);
  EXPECT_EQ(*atomic.raw_scope, 2u);
  EXPECT_EQ(*atomic.raw_th, 1u);
  EXPECT_TRUE(*atomic.returns_old_value);
}

TEST(ConSan, FaultDropBarrierModeRewritesSelectedBarrier) {
  const std::array<uint32_t, 5> text_words = {
      0xBF940000u,              // s_barrier_wait
      0xD8340000u, 0x00000000u, // ds_store_b32
      0xBF940000u,              // s_barrier_wait
      0xBFB00000u,              // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_drop_barrier = true;
  options.fault_barrier_index = 1;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineBarrierNopRewrite);
  EXPECT_EQ(result.patches.front().anchor_offset, 12u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 12u);
  EXPECT_EQ(result.patches.front().original_size, 4u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  uint32_t first_barrier = 0;
  uint32_t rewritten_barrier = 0;
  std::memcpy(&first_barrier, result.elf_bytes.data() + 0x100, sizeof(first_barrier));
  std::memcpy(&rewritten_barrier, result.elf_bytes.data() + 0x100 + 12, sizeof(rewritten_barrier));
  EXPECT_EQ(first_barrier, 0xBF940000u);
  EXPECT_EQ(rewritten_barrier, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(ConSan, FaultDropBarrierModeSkipsRocclrRuntimeHelpers) {
  const std::array<uint32_t, 3> text_words = {
      0xBF940000u, // s_barrier_wait
      0x06040F06u, // v_add_f32_e32 v2, v6, v7
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "__amd_rocclr_fillBufferAligned");
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_drop_barrier = true;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_TRUE(result.patches.empty());

  bool saw_skip_warning = false;
  for (const std::string &warning : result.warnings)
    saw_skip_warning |= warning.find("skipped ROCclr runtime helper") != std::string::npos;
  EXPECT_TRUE(saw_skip_warning);
}

TEST(ConSan, FaultDropBarrierModeComposesWithLdsCheckTrapPatch) {
  const std::array<uint32_t, 12> text_words = {
      0xD8D80000u, 0x00000000u, // ds_load_b32
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF940000u, // s_barrier_wait
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.fault_drop_barrier = true;
  options.scratch_vgpr = 5;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 2u);
  EXPECT_EQ(result.patches[0].kind, ConSanPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_EQ(result.patches[1].kind, ConSanPatchKind::InlineBarrierNopRewrite);
  EXPECT_EQ(result.patches[1].anchor_offset, 40u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  uint32_t rewritten_barrier = 0;
  std::memcpy(&rewritten_barrier, result.elf_bytes.data() + 0x100 + 40, sizeof(rewritten_barrier));
  EXPECT_EQ(rewritten_barrier, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));

  uint32_t original_access = 0;
  uint32_t duplicated_access = 0;
  std::memcpy(&original_access, result.elf_bytes.data() + 0x100, sizeof(original_access));
  std::memcpy(&duplicated_access, result.elf_bytes.data() + 0x100 + 8, sizeof(duplicated_access));
  EXPECT_EQ(original_access, 0xD8D80000u);
  EXPECT_EQ(duplicated_access, 0xD8D80000u);
}

TEST(ConSan, MarksSupportedLdsKernelAsPreflightCandidate) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  ASSERT_EQ(result.kernels.size(), 1u);

  const ConSanKernelInfo &kernel = result.kernels.front();
  EXPECT_TRUE(kernel.decoded);
  EXPECT_EQ(kernel.code_size, 24u);
  EXPECT_EQ(kernel.stats.instruction_count, 4u);
  EXPECT_EQ(kernel.stats.lds_read_count, 1u);
  EXPECT_EQ(kernel.stats.lds_write_count, 1u);
  EXPECT_EQ(kernel.stats.lds_atomic_count, 0u);
  EXPECT_EQ(kernel.stats.fence_like_count, 0u);
  ASSERT_EQ(kernel.lds_sites.size(), 2u);
  EXPECT_EQ(kernel.lds_sites[0].kind, ConSanLdsAccessKind::Write);
  EXPECT_TRUE(kernel.lds_sites[0].supported_mvp);
  EXPECT_EQ(kernel.lds_sites[0].text_offset, 0u);
  EXPECT_EQ(kernel.lds_sites[0].width_bits, 32u);
  EXPECT_EQ(kernel.lds_sites[1].kind, ConSanLdsAccessKind::Read);
  EXPECT_TRUE(kernel.lds_sites[1].supported_mvp);
  EXPECT_EQ(kernel.lds_sites[1].text_offset, 8u);
  EXPECT_EQ(kernel.lds_sites[1].width_bits, 32u);
  EXPECT_EQ(kernel.preflight_action, ConSanPreflightAction::Candidate);
  EXPECT_GE(kernel.preflight_reasons.size(), 2u);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
}

TEST(ConSan, ProbeNopModeEmitsPatchedElfForCandidate) {
  const std::array<uint32_t, 7> text_words = {
      0xD8340000u, 0x00000000u, // ds_store_b32
      0xD8D80000u, 0x00000000u, // ds_load_b32
      0xBFC60000u,              // s_wait_dscnt
      0x06040F06u,              // v_add_f32_e32 v2, v6, v7
      0xBFB00000u,              // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_nop = true;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.text_sections.size(), 1u);
  EXPECT_EQ(result.kernels.front().preflight_action, ConSanPreflightAction::Candidate);
  EXPECT_TRUE(result.modified);
  EXPECT_FALSE(result.elf_bytes.empty());
  EXPECT_NE(result.elf_bytes, bytes);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::TrampolineNop);
  EXPECT_EQ(result.patches.front().anchor_offset, 20u);
  EXPECT_EQ(result.patches.front().original_size, 4u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  EXPECT_GT(patched.text_sections().front()->size(), result.text_sections.front().size);
}

TEST(ConSan, ProbeNopModeRewritesExistingNopInPlace) {
  const std::array<uint32_t, 4> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32
      0xBF800000u, // s_nop 0
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_nop = true;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineNopRewrite);
  EXPECT_EQ(result.patches.front().anchor_offset, 8u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 8u);
  EXPECT_EQ(result.patches.front().original_size, 4u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());
  EXPECT_NE(result.elf_bytes, bytes);

  uint32_t rewritten_word = 0;
  std::memcpy(&rewritten_word, result.elf_bytes.data() + 0x100 + 8, sizeof(rewritten_word));
  EXPECT_EQ(rewritten_word, 0xBF800001u);
}

TEST(ConSan, ProbeTrampolineNopModeSkipsExistingNopRewrite) {
  const std::array<uint32_t, 5> text_words = {
      0xBF800000u, // s_nop 0
      0xD8340000u,
      0x00000000u, // ds_store_b32
      0x06040F06u, // v_add_f32_e32 v2, v6, v7
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_trampoline_nop = true;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::TrampolineNop);
  EXPECT_EQ(result.patches.front().anchor_offset, 12u);
  EXPECT_EQ(result.patches.front().original_size, 4u);
  ASSERT_GT(result.elf_bytes.size(), bytes.size());

  uint32_t original_nop = 0;
  std::memcpy(&original_nop, result.elf_bytes.data() + 0x100, sizeof(original_nop));
  EXPECT_EQ(original_nop, 0xBF800000u);
}

TEST(ConSan, ProbeTrampolineNopModeSkipsSClauseRun) {
  const std::array<uint32_t, 10> text_words = {
      0xD8340000u, 0x00000000u, // ds_store_b32
      0xBF850001u,              // s_clause 1
      0xF4002200u, 0xF8000020u, // s_load_b64 s[8:9], s[0:1], 0x20
      0xF4006000u, 0xF8000000u, // s_load_b256 s[0:7], s[0:1], 0x0
      0xBFC70000u,              // s_wait_kmcnt 0
      0x06040F06u,              // v_add_f32_e32 v2, v6, v7
      0xBFB00000u,              // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_trampoline_nop = true;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::TrampolineNop);
  EXPECT_EQ(result.patches.front().anchor_offset, 32u);
  EXPECT_EQ(result.patches.front().original_size, 4u);
  ASSERT_GT(result.elf_bytes.size(), bytes.size());

  std::array<uint32_t, 8> prefix_words{};
  std::memcpy(prefix_words.data(), result.elf_bytes.data() + 0x100,
              prefix_words.size() * sizeof(uint32_t));
  EXPECT_EQ(prefix_words[0], 0xD8340000u);
  EXPECT_EQ(prefix_words[1], 0x00000000u);
  EXPECT_EQ(prefix_words[2], 0xBF850001u);
  EXPECT_EQ(prefix_words[3], 0xF4002200u);
  EXPECT_EQ(prefix_words[4], 0xF8000020u);
  EXPECT_EQ(prefix_words[5], 0xF4006000u);
  EXPECT_EQ(prefix_words[6], 0xF8000000u);
  EXPECT_EQ(prefix_words[7], 0xBFC70000u);
}

TEST(ConSan, ProbeTrampolineNopModeSkipsRocclrRuntimeHelpers) {
  const std::array<uint32_t, 4> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32
      0x06040F06u, // v_add_f32_e32 v2, v6, v7
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "__amd_rocclr_fillBufferAligned");
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_trampoline_nop = true;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_TRUE(result.patches.empty());

  bool saw_skip_warning = false;
  for (const std::string &warning : result.warnings)
    saw_skip_warning |= warning.find("skipped ROCclr runtime helper") != std::string::npos;
  EXPECT_TRUE(saw_skip_warning);
}

TEST(ConSan, ProbeTrampolineNopModeSkipsCodeObjectWithoutCandidateSites) {
  const std::vector<uint8_t> bytes = make_rdna4_unsupported_lds_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_trampoline_nop = true;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_TRUE(result.patches.empty());

  bool saw_skip_warning = false;
  for (const std::string &warning : result.warnings)
    saw_skip_warning |= warning.find("without supported DBI candidate") != std::string::npos;
  EXPECT_TRUE(saw_skip_warning);
}

TEST(ConSan, ProbeNopModePrefersVectorAluAnchorOverMemoryAnchor) {
  const std::array<uint32_t, 4> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32
      0x06040F06u, // v_add_f32_e32 v2, v6, v7
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_nop = true;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::TrampolineNop);
  EXPECT_EQ(result.patches.front().anchor_offset, 8u);
  EXPECT_EQ(result.patches.front().original_size, 4u);
}

TEST(ConSan, ProbeEndpgmModeRewritesVectorAluInPlace) {
  const std::array<uint32_t, 4> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32
      0x06040F06u, // v_add_f32_e32 v2, v6, v7
      0xBF800000u, // s_nop 0
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_nop = true;
  options.probe_endpgm = true;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineEndpgmRewrite);
  EXPECT_EQ(result.patches.front().anchor_offset, 8u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 8u);
  EXPECT_EQ(result.patches.front().original_size, 4u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());
  EXPECT_NE(result.elf_bytes, bytes);

  uint32_t rewritten_word = 0;
  std::memcpy(&rewritten_word, result.elf_bytes.data() + 0x100 + 8, sizeof(rewritten_word));
  EXPECT_EQ(rewritten_word, 0xBFB00000u);
}

TEST(ConSan, ProbeLdsEndpgmModeRewritesFirstSupportedReadInPlace) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_endpgm = true;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsEndpgmRewrite);
  EXPECT_EQ(result.patches.front().anchor_offset, 8u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 8u);
  EXPECT_EQ(result.patches.front().original_size, 8u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());
  EXPECT_NE(result.elf_bytes, bytes);

  uint32_t rewritten_word = 0;
  std::memcpy(&rewritten_word, result.elf_bytes.data() + 0x100 + 8, sizeof(rewritten_word));
  EXPECT_EQ(rewritten_word, 0xBFB00000u);
}

TEST(ConSan, ProbeLdsCheckTrapModeRewritesPaddedLoadInPlace) {
  const std::array<uint32_t, 13> text_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;
  options.delay_nops = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 8u);
  EXPECT_EQ(result.patches.front().original_size, 48u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 3u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const std::array<uint32_t, 13> expected_words = {
      0xD8D80000u,
      0x01000002u, // original ds_load_b32 v1, v2
      0xBF800000u,
      0xBF800000u, // delay
      0xD8D80000u,
      0x03000002u, // duplicate ds_load_b32 v3, v2
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0701u, // v_cmp_ne_u32_e32 vcc_lo, v1, v3
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
      0xBFB00000u, // original s_endpgm after padding
  };
  std::array<uint32_t, expected_words.size()> rewritten_words{};
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + 0x100,
              rewritten_words.size() * sizeof(uint32_t));
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(ConSan, ProbeLdsCheckTrapModeCanPatchMultiplePaddedLoads) {
  const std::array<uint32_t, 23> text_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xD8D80000u,
      0x04000005u, // ds_load_b32 v4, v5
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.delay_nops = 1;
  options.max_patches = 2;
  options.scratch_vgpr = 3;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 2u);
  EXPECT_EQ(result.patches[0].kind, ConSanPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_EQ(result.patches[0].anchor_offset, 0u);
  EXPECT_EQ(result.patches[0].trampoline_offset, 8u);
  EXPECT_EQ(result.patches[0].original_size, 44u);
  EXPECT_EQ(result.patches[1].kind, ConSanPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_EQ(result.patches[1].anchor_offset, 44u);
  EXPECT_EQ(result.patches[1].trampoline_offset, 52u);
  EXPECT_EQ(result.patches[1].original_size, 44u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const std::array<uint32_t, 22> expected_words = {
      0xD8D80000u,
      0x01000002u, // original ds_load_b32 v1, v2
      0xBF800000u, // delay
      0xD8D80000u,
      0x03000002u, // duplicate ds_load_b32 v3, v2
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0701u, // v_cmp_ne_u32_e32 vcc_lo, v1, v3
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
      0xD8D80000u,
      0x04000005u, // original ds_load_b32 v4, v5
      0xBF800000u, // delay
      0xD8D80000u,
      0x03000005u, // duplicate ds_load_b32 v3, v5
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0704u, // v_cmp_ne_u32_e32 vcc_lo, v4, v3
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
  };
  std::array<uint32_t, expected_words.size()> rewritten_words{};
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + 0x100,
              rewritten_words.size() * sizeof(uint32_t));
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(ConSan, ProbeLdsCheckTrapModeRewritesPaddedU16D16LoadInPlace) {
  const std::array<uint32_t, 14> text_words = {
      0xDA980000u,
      0x01000002u, // ds_load_u16_d16 v1, v2
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.delay_nops = 1;
  options.scratch_vgpr = 3;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 8u);
  EXPECT_EQ(result.patches.front().original_size, 52u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const std::array<uint32_t, 14> expected_words = {
      0xDA980000u,
      0x01000002u, // original ds_load_u16_d16 v1, v2
      0xBFC60000u, // s_wait_dscnt 0 for original d16 load
      build_v_mov_b32_e32(3, vector_source_vgpr(1), ROCJITSU_CODE_ARCH_RDNA4),
      0xBF800000u, // delay
      0xDA980000u,
      0x03000002u, // duplicate ds_load_u16_d16 v3, v2
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0701u, // v_cmp_ne_u32_e32 vcc_lo, v1, v3
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
      0xBFB00000u, // original s_endpgm after padding
  };
  std::array<uint32_t, expected_words.size()> rewritten_words{};
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + 0x100,
              rewritten_words.size() * sizeof(uint32_t));
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(ConSan, ProbeLdsCheckTrapModeRewritesPaddedU16D16HiLoadInPlace) {
  const std::array<uint32_t, 14> text_words = {
      0xDA9C0000u,
      0x01000002u, // ds_load_u16_d16_hi v1, v2
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.delay_nops = 1;
  options.scratch_vgpr = 3;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_EQ(result.patches.front().original_size, 52u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const std::array<uint32_t, 14> expected_words = {
      0xDA9C0000u,
      0x01000002u, // original ds_load_u16_d16_hi v1, v2
      0xBFC60000u, // s_wait_dscnt 0 for original d16 load
      build_v_mov_b32_e32(3, vector_source_vgpr(1), ROCJITSU_CODE_ARCH_RDNA4),
      0xBF800000u, // delay
      0xDA9C0000u,
      0x03000002u, // duplicate ds_load_u16_d16_hi v3, v2
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0701u, // v_cmp_ne_u32_e32 vcc_lo, v1, v3
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
      0xBFB00000u, // original s_endpgm after padding
  };
  std::array<uint32_t, expected_words.size()> rewritten_words{};
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + 0x100,
              rewritten_words.size() * sizeof(uint32_t));
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(ConSan, ProbeLdsCheckTrapModeCanUseSleepDelay) {
  const std::array<uint32_t, 12> text_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;
  options.delay_mode = ConSanDelayMode::Sleep;
  options.delay_nops = 7;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_EQ(result.patches.front().original_size, 44u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const std::array<uint32_t, 12> expected_words = {
      0xD8D80000u,
      0x01000002u, // original ds_load_b32 v1, v2
      build_s_sleep(7, ROCJITSU_CODE_ARCH_RDNA4),
      0xD8D80000u,
      0x03000002u, // duplicate ds_load_b32 v3, v2
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0701u, // v_cmp_ne_u32_e32 vcc_lo, v1, v3
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
      0xBFB00000u, // original s_endpgm after padding
  };
  std::array<uint32_t, expected_words.size()> rewritten_words{};
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + 0x100,
              rewritten_words.size() * sizeof(uint32_t));
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(ConSan, ProbeLdsCheckTrapModeCanUseSleepVarDelay) {
  const std::array<uint32_t, 12> text_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;
  options.delay_mode = ConSanDelayMode::SleepVar;
  options.delay_nops = 1;
  options.delay_var_ssrc = kRdna4VccLo;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_EQ(result.patches.front().original_size, 44u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const std::array<uint32_t, 12> expected_words = {
      0xD8D80000u,
      0x01000002u, // original ds_load_b32 v1, v2
      build_s_sleep_var(kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0xD8D80000u,
      0x03000002u, // duplicate ds_load_b32 v3, v2
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0701u, // v_cmp_ne_u32_e32 vcc_lo, v1, v3
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
      0xBFB00000u, // original s_endpgm after padding
  };
  std::array<uint32_t, expected_words.size()> rewritten_words{};
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + 0x100,
              rewritten_words.size() * sizeof(uint32_t));
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(ConSan, ProbeLdsCheckTrapModeRejectsOversizedSleepDelay) {
  const std::array<uint32_t, 4> text_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.delay_mode = ConSanDelayMode::Sleep;
  options.delay_nops = 65536;

  const auto result = try_patch_consan(bytes, options);

  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
  ASSERT_FALSE(result.errors.empty());
  EXPECT_NE(result.errors.front().find("16-bit s_sleep"), std::string::npos);
}

TEST(ConSan, ProbeLdsCheckTrapModeRewritesPaddedStoreInPlace) {
  const std::array<uint32_t, 13> text_words = {
      0xD8340000u,
      0x00000102u, // ds_store_b32 v2, v1
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;
  options.delay_nops = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsStoreCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 8u);
  EXPECT_EQ(result.patches.front().original_size, 48u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 3u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const std::array<uint32_t, 13> expected_words = {
      0xD8340000u,
      0x00000102u, // original ds_store_b32 v2, v1
      0xBF800000u,
      0xBF800000u, // delay
      0xD8D80000u,
      0x03000002u, // readback ds_load_b32 v3, v2
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0701u, // v_cmp_ne_u32_e32 vcc_lo, v1, v3
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
      0xBFB00000u, // original s_endpgm after padding
  };
  std::array<uint32_t, expected_words.size()> rewritten_words{};
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + 0x100,
              rewritten_words.size() * sizeof(uint32_t));
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(ConSan, ProbeLdsCheckTrapModeCanReportMismatchToMarkerBuffer) {
  const std::array<uint32_t, 24> text_words = {
      0xD8340000u,
      0x00000102u, // ds_store_b32 v2, v1
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;
  options.report_buffer_address = 0x1234567887654321ull;
  options.report_marker = 0xABCDEF01u;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsStoreCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 8u);
  EXPECT_EQ(result.patches.front().original_size, 84u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 3u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const auto mov_report_lo = build_v_mov_b32_e64_literal(4, 0x87654321u, ROCJITSU_CODE_ARCH_RDNA4);
  const auto mov_report_hi = build_v_mov_b32_e64_literal(5, 0x12345678u, ROCJITSU_CODE_ARCH_RDNA4);
  const auto mov_marker = build_v_mov_b32_e64_literal(6, 0xABCDEF01u, ROCJITSU_CODE_ARCH_RDNA4);
  const auto store_marker = build_flat_store_b32_vaddr_vsrc(4, 6, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(mov_report_lo);
  ASSERT_TRUE(mov_report_hi);
  ASSERT_TRUE(mov_marker);
  ASSERT_TRUE(store_marker);

  const std::array<uint32_t, 24> expected_words = {
      0xD8340000u,
      0x00000102u, // original ds_store_b32 v2, v1
      0xD8D80000u,
      0x03000002u, // readback ds_load_b32 v3, v2
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0701u, // v_cmp_ne_u32_e32 vcc_lo, v1, v3
      0xBFA3000Cu, // s_cbranch_vccz +12, skipping marker store when equal
      (*mov_report_lo)[0],
      (*mov_report_lo)[1],
      (*mov_report_lo)[2],
      (*mov_report_hi)[0],
      (*mov_report_hi)[1],
      (*mov_report_hi)[2],
      (*mov_marker)[0],
      (*mov_marker)[1],
      (*mov_marker)[2],
      (*store_marker)[0],
      (*store_marker)[1],
      (*store_marker)[2],
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
      0xBF800000u,
      0xBF800000u,
      0xBFB00000u, // original s_endpgm after padding
  };
  std::array<uint32_t, expected_words.size()> rewritten_words{};
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + 0x100,
              rewritten_words.size() * sizeof(uint32_t));
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(ConSan, ProbeLdsCheckTrapModeRewritesPaddedB64LoadInPlace) {
  const std::array<uint32_t, 16> text_words = {
      0xD9D80000u,
      0x01000009u, // ds_load_b64 v[1:2], v9
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 5;
  options.delay_nops = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 8u);
  EXPECT_EQ(result.patches.front().original_size, 60u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 5u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const std::array<uint32_t, 16> expected_words = {
      0xD9D80000u,
      0x01000009u, // original ds_load_b64 v[1:2], v9
      0xBF800000u,
      0xBF800000u, // delay
      0xD9D80000u,
      0x05000009u, // duplicate ds_load_b64 v[5:6], v9
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0B01u, // v_cmp_ne_u32_e32 vcc_lo, v1, v5
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      0x7C9A0D02u, // v_cmp_ne_u32_e32 vcc_lo, v2, v6
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
      0xBFB00000u, // original s_endpgm after padding
  };
  std::array<uint32_t, expected_words.size()> rewritten_words{};
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + 0x100,
              rewritten_words.size() * sizeof(uint32_t));
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(ConSan, ProbeLdsCheckTrapModeRewritesPaddedB128StoreInPlace) {
  const std::array<uint32_t, 21> text_words = {
      0xDB7C0000u,
      0x00000109u, // ds_store_b128 v9, v[1:4]
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 5;
  options.delay_nops = 1;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsStoreCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 8u);
  EXPECT_EQ(result.patches.front().original_size, 80u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 5u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const std::array<uint32_t, 21> expected_words = {
      0xDB7C0000u,
      0x00000109u, // original ds_store_b128 v9, v[1:4]
      0xBF800000u, // delay
      0xDBFC0000u,
      0x05000009u, // readback ds_load_b128 v[5:8], v9
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0B01u, // v_cmp_ne_u32_e32 vcc_lo, v1, v5
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      0x7C9A0D02u, // v_cmp_ne_u32_e32 vcc_lo, v2, v6
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      0x7C9A0F03u, // v_cmp_ne_u32_e32 vcc_lo, v3, v7
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      0x7C9A1104u, // v_cmp_ne_u32_e32 vcc_lo, v4, v8
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
      0xBFB00000u, // original s_endpgm after padding
  };
  std::array<uint32_t, expected_words.size()> rewritten_words{};
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + 0x100,
              rewritten_words.size() * sizeof(uint32_t));
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(ConSan, ProbeLdsCheckTrapModeAutoScratchUsesLiveness) {
  const std::array<uint32_t, 14> text_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0x06080603u, // v_add_f32_e32 v4, v3, v3
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.delay_nops = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsLoadCheckTrap);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 5u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const std::array<uint32_t, 14> expected_words = {
      0xD8D80000u,
      0x01000002u, // original ds_load_b32 v1, v2
      0xBF800000u,
      0xBF800000u, // delay
      0xD8D80000u,
      0x05000002u, // duplicate ds_load_b32 v5, v2
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(4, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0B01u, // v_cmp_ne_u32_e32 vcc_lo, v1, v5
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 4, ROCJITSU_CODE_ARCH_RDNA4),
      0x06080603u, // original v_add_f32_e32 after padding
      0xBFB00000u, // original s_endpgm
  };
  std::array<uint32_t, expected_words.size()> rewritten_words{};
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + 0x100,
              rewritten_words.size() * sizeof(uint32_t));
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(ConSan, ProbeLdsCheckTrapModeCanGrowDescriptorForAutoScratch) {
  const std::array<uint32_t, 14> text_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xD8340000u,
      0x00000302u, // ds_store_b32 v2, v3
      0xBFB00000u, // s_endpgm
  };
  const uint32_t four_vgprs_granulated = 0;
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "lds_probe", four_vgprs_granulated);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.delay_nops = 1;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsLoadCheckTrap);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 4u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const uint64_t descriptor_offset = 0x100 + text_words.size() * sizeof(uint32_t);
  KD descriptor{};
  std::memcpy(&descriptor, result.elf_bytes.data() + descriptor_offset, sizeof(descriptor));
  const uint32_t granulated = AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc1,
                                              kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  EXPECT_EQ(granulated, 1u);
}

TEST(ConSan, ProbeLdsCheckTrapModeCanGrowDescriptorForB64ScratchHeadroom) {
  const std::array<uint32_t, 4> text_words = {
      0xD9D80000u,
      0x01000009u, // ds_load_b64 v[1:2], v9
      build_v_mov_b32_e32(13, vector_source_vgpr(13), ROCJITSU_CODE_ARCH_RDNA4),
      0xBFB00000u, // s_endpgm
  };
  const uint32_t sixteen_vgprs_granulated = 3;
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "lds_probe", sixteen_vgprs_granulated);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.delay_nops = 1;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::LocalCaveLdsLoadCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, text_words.size() * sizeof(uint32_t));
  EXPECT_EQ(result.patches.front().original_size, 8u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 14u);
  EXPECT_GT(result.elf_bytes.size(), bytes.size());

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  const uint32_t granulated = AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc1,
                                              kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  EXPECT_EQ(granulated, 4u);
}

TEST(ConSan, ProbeLdsCheckTrapModeGrowsDescriptorForB64StoreReadbackHeadroom) {
  const std::array<uint32_t, 4> text_words = {
      0xD89A0000u,
      0x00000109u, // ds_write_b64 v9, v[1:2]
      build_v_mov_b32_e32(69, vector_source_vgpr(69), ROCJITSU_CODE_ARCH_CDNA4),
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  const uint32_t seventy_two_vgprs_granulated = 8;
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "lds_probe", seventy_two_vgprs_granulated, /*wave32=*/false,
      /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.delay_nops = 1;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::LocalCaveLdsStoreCheckTrap);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 70u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  const uint32_t granulated = AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc1,
                                              kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  EXPECT_EQ(granulated, 9u);
}

TEST(ConSan, ProbeLdsCheckTrapModePrefersDescriptorCoveredCandidate) {
  const std::array<uint32_t, 15> text_words = {
      0xD9D80000u,
      0x01000009u, // ds_load_b64 v[1:2], v9; would need descriptor growth
      build_v_mov_b32_e32(13, vector_source_vgpr(13), ROCJITSU_CODE_ARCH_RDNA4),
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2; can use descriptor-covered v14
      0xBF800000u,
      0xBF800000u,
      0xBF800000u,
      0xBF800000u,
      0xBF800000u,
      0xBF800000u,
      0xBF800000u,
      0xBF800000u,
      0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const uint32_t sixteen_vgprs_granulated = 3;
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "lds_probe", sixteen_vgprs_granulated);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.delay_nops = 1;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 3u * sizeof(uint32_t));
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 14u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const uint64_t descriptor_offset = 0x100 + text_words.size() * sizeof(uint32_t);
  KD descriptor{};
  std::memcpy(&descriptor, result.elf_bytes.data() + descriptor_offset, sizeof(descriptor));
  const uint32_t granulated = AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc1,
                                              kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  EXPECT_EQ(granulated, sixteen_vgprs_granulated);
}

TEST(ConSan, ProbeLdsCheckTrapModeUsesReachableUncoveredNopCave) {
  const std::array<uint32_t, 3> kernel_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 1> function_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 12> tail_words = {
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words, tail_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;
  options.delay_nops = 1;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::LocalCaveLdsLoadCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 16u);
  EXPECT_EQ(result.patches.front().original_size, 8u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 3u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());
  EXPECT_NE(result.elf_bytes, bytes);

  const std::array<uint32_t, 2> expected_anchor = {
      build_s_branch(3, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
  };
  std::array<uint32_t, expected_anchor.size()> anchor_words{};
  std::memcpy(anchor_words.data(), result.elf_bytes.data() + 0x100,
              anchor_words.size() * sizeof(uint32_t));
  EXPECT_EQ(anchor_words, expected_anchor);

  const std::array<uint32_t, 12> expected_cave = {
      0xD8D80000u,
      0x01000002u, // original ds_load_b32 v1, v2
      0xBF800000u, // delay
      0xD8D80000u,
      0x03000002u, // duplicate ds_load_b32 v3, v2
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0701u, // v_cmp_ne_u32_e32 vcc_lo, v1, v3
      0xBFA30001u, // s_cbranch_vccz +1
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_branch(-14, ROCJITSU_CODE_ARCH_RDNA4),
  };
  std::array<uint32_t, expected_cave.size()> cave_words{};
  std::memcpy(cave_words.data(), result.elf_bytes.data() + 0x110,
              cave_words.size() * sizeof(uint32_t));
  EXPECT_EQ(cave_words, expected_cave);
}

TEST(ConSan, ProbeLdsCheckTrapModeSelectsOneLocalCavePerKernel) {
  const std::array<uint32_t, 5> kernel_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xD8D80000u,
      0x04000005u, // ds_load_b32 v4, v5
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 1> function_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 25> tail_words = {
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBFB00000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words, tail_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.max_patches = 2;
  options.scratch_vgpr = 6;
  options.delay_nops = 1;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::LocalCaveLdsLoadCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 24u);
  EXPECT_EQ(result.patches.front().original_size, 8u);
}

TEST(ConSan, ProbeLdsCheckTrapModeUsesReachableUncoveredNopCaveFor2addrB64Load) {
  const std::array<uint32_t, 3> kernel_words = {
      0xD9DC0000u,
      0x01000009u, // ds_load_2addr_b64 v[1:4], v9
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 1> function_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 21> tail_words = {
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words, tail_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 5;
  options.delay_nops = 1;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::LocalCaveLdsLoadCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 16u);
  EXPECT_EQ(result.patches.front().original_size, 8u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 5u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());
  EXPECT_NE(result.elf_bytes, bytes);

  const std::array<uint32_t, 2> expected_anchor = {
      build_s_branch(3, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
  };
  std::array<uint32_t, expected_anchor.size()> anchor_words{};
  std::memcpy(anchor_words.data(), result.elf_bytes.data() + 0x100,
              anchor_words.size() * sizeof(uint32_t));
  EXPECT_EQ(anchor_words, expected_anchor);

  const std::array<uint32_t, 21> expected_cave = {
      0xD9DC0000u,
      0x01000009u, // original ds_load_2addr_b64 v[1:4], v9
      0xBF800000u, // delay
      0xD9DC0000u,
      0x05000009u, // duplicate ds_load_2addr_b64 v[5:8], v9
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0B01u, // v_cmp_ne_u32_e32 vcc_lo, v1, v5
      0xBFA30001u, // s_cbranch_vccz +1
      0xBF900000u, // s_trap 0
      0x7C9A0D02u, // v_cmp_ne_u32_e32 vcc_lo, v2, v6
      0xBFA30001u, // s_cbranch_vccz +1
      0xBF900000u, // s_trap 0
      0x7C9A0F03u, // v_cmp_ne_u32_e32 vcc_lo, v3, v7
      0xBFA30001u, // s_cbranch_vccz +1
      0xBF900000u, // s_trap 0
      0x7C9A1104u, // v_cmp_ne_u32_e32 vcc_lo, v4, v8
      0xBFA30001u, // s_cbranch_vccz +1
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_branch(-23, ROCJITSU_CODE_ARCH_RDNA4),
  };
  std::array<uint32_t, expected_cave.size()> cave_words{};
  std::memcpy(cave_words.data(), result.elf_bytes.data() + 0x110,
              cave_words.size() * sizeof(uint32_t));
  EXPECT_EQ(cave_words, expected_cave);
}

TEST(ConSan, ProbeLdsCheckTrapModeUsesReachableUncoveredNopCaveForB128Store) {
  const std::array<uint32_t, 3> kernel_words = {
      0xDB7C0000u,
      0x00000109u, // ds_store_b128 v9, v[1:4]
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 1> function_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 21> tail_words = {
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words, tail_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 5;
  options.delay_nops = 1;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::LocalCaveLdsStoreCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 16u);
  EXPECT_EQ(result.patches.front().original_size, 8u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 5u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());
  EXPECT_NE(result.elf_bytes, bytes);

  const std::array<uint32_t, 2> expected_anchor = {
      build_s_branch(3, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
  };
  std::array<uint32_t, expected_anchor.size()> anchor_words{};
  std::memcpy(anchor_words.data(), result.elf_bytes.data() + 0x100,
              anchor_words.size() * sizeof(uint32_t));
  EXPECT_EQ(anchor_words, expected_anchor);

  const std::array<uint32_t, 21> expected_cave = {
      0xDB7C0000u,
      0x00000109u, // original ds_store_b128 v9, v[1:4]
      0xBF800000u, // delay
      0xDBFC0000u,
      0x05000009u, // readback ds_load_b128 v[5:8], v9
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0B01u, // v_cmp_ne_u32_e32 vcc_lo, v1, v5
      0xBFA30001u, // s_cbranch_vccz +1
      0xBF900000u, // s_trap 0
      0x7C9A0D02u, // v_cmp_ne_u32_e32 vcc_lo, v2, v6
      0xBFA30001u, // s_cbranch_vccz +1
      0xBF900000u, // s_trap 0
      0x7C9A0F03u, // v_cmp_ne_u32_e32 vcc_lo, v3, v7
      0xBFA30001u, // s_cbranch_vccz +1
      0xBF900000u, // s_trap 0
      0x7C9A1104u, // v_cmp_ne_u32_e32 vcc_lo, v4, v8
      0xBFA30001u, // s_cbranch_vccz +1
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_branch(-23, ROCJITSU_CODE_ARCH_RDNA4),
  };
  std::array<uint32_t, expected_cave.size()> cave_words{};
  std::memcpy(cave_words.data(), result.elf_bytes.data() + 0x110,
              cave_words.size() * sizeof(uint32_t));
  EXPECT_EQ(cave_words, expected_cave);
}

TEST(ConSan, ProbeLdsCheckTrapModeUsesAppendedTextCaveWhenNoLocalCaveFits) {
  const std::array<uint32_t, 3> text_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::LocalCaveLdsLoadCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 12u);
  EXPECT_EQ(result.patches.front().original_size, 8u);
  EXPECT_GT(result.elf_bytes.size(), bytes.size());

  const std::array<uint32_t, 2> expected_anchor = {
      build_s_branch(2, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
  };
  std::array<uint32_t, expected_anchor.size()> anchor_words{};
  std::memcpy(anchor_words.data(), result.elf_bytes.data() + 0x100,
              anchor_words.size() * sizeof(uint32_t));
  EXPECT_EQ(anchor_words, expected_anchor);
}

TEST(ConSan, ProbeLdsCheckTrapModeReportsExcessiveDelay) {
  const std::array<uint32_t, 3> text_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;
  options.delay_nops = 300;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_FALSE(result.warnings.empty());
  EXPECT_NE(result.warnings.back().find("requested delay needs too much padding"),
            std::string::npos);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
}

TEST(ConSan, ProbeLdsEndpgmModeCanRewritePreflightSkippedKernel) {
  const std::vector<uint8_t> bytes = make_rdna4_unsupported_lds_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_endpgm = true;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_EQ(result.kernels.size(), 1u);
  EXPECT_EQ(result.kernels.front().preflight_action, ConSanPreflightAction::Skip);
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsEndpgmRewrite);
  EXPECT_EQ(result.patches.front().anchor_offset, 8u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 8u);
  EXPECT_EQ(result.patches.front().original_size, 8u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());
  EXPECT_NE(result.elf_bytes, bytes);

  uint32_t rewritten_word = 0;
  std::memcpy(&rewritten_word, result.elf_bytes.data() + 0x100 + 8, sizeof(rewritten_word));
  EXPECT_EQ(rewritten_word, 0xBFB00000u);
}

TEST(ConSan, FailClosedRejectsUnsupportedLdsKernel) {
  const std::vector<uint8_t> bytes = make_rdna4_unsupported_lds_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fail_closed = true;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_FALSE(result.errors.empty());
  ASSERT_EQ(result.kernels.size(), 1u);
  const ConSanKernelInfo &kernel = result.kernels.front();
  EXPECT_EQ(kernel.preflight_action, ConSanPreflightAction::Reject);
  ASSERT_GE(kernel.preflight_reasons.size(), 2u);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
}

TEST(ConSan, InvalidCdna4InstructionFailsOpenWithTypedPreflightSkip) {
  const std::array<uint32_t, 2> text_words = {
      0xD3AD0000u, // unsupported standalone CDNA4 VOP3P op 45
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "invalid_vop3p", /*vgpr_granulated=*/0,
      /*wave32=*/false, /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_EQ(result.kernels.size(), 1u);
  const ConSanKernelInfo &kernel = result.kernels.front();
  EXPECT_EQ(kernel.stats.decode_error_count, 1u);
  EXPECT_EQ(kernel.preflight_action, ConSanPreflightAction::Skip);
  EXPECT_FALSE(result.modified);
  ASSERT_GE(result.warnings.size(), 2u);
  EXPECT_NE(result.warnings[0].find("kernel 'invalid_vop3p'"), std::string::npos);
  EXPECT_NE(result.warnings[0].find("text offset 0"), std::string::npos);
  EXPECT_NE(result.warnings[0].find("Invalid instruction opcode"), std::string::npos);
  EXPECT_NE(result.warnings[1].find("decode errors: 1"), std::string::npos);
}

TEST(ConSan, InvalidCdna4InstructionFailsClosedWithTypedPreflightReject) {
  const std::array<uint32_t, 2> text_words = {
      0xD3AD0000u, // unsupported standalone CDNA4 VOP3P op 45
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "invalid_vop3p", /*vgpr_granulated=*/0,
      /*wave32=*/false, /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fail_closed = true;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_EQ(result.kernels.size(), 1u);
  const ConSanKernelInfo &kernel = result.kernels.front();
  EXPECT_EQ(kernel.stats.decode_error_count, 1u);
  EXPECT_EQ(kernel.preflight_action, ConSanPreflightAction::Reject);
  ASSERT_FALSE(result.errors.empty());
  EXPECT_NE(result.errors.front().find("decode errors: 1"), std::string::npos);
  EXPECT_FALSE(result.modified);
}

TEST(ConSan, InvalidCdna4SuffixCannotReenterWholeTextBasicBlockDecode) {
  const std::array<uint32_t, 4> text_words = {
      0xD81A0000u,
      0x00000000u, // ds_store_b32 v0, v0
      0xD3AD0000u, // unsupported standalone CDNA4 VOP3P op 45
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "invalid_suffix", /*vgpr_granulated=*/0,
      /*wave32=*/false, /*uses_dynamic_stack=*/false, EF_AMDGPU_MACH_AMDGCN_GFX950);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_EQ(result.kernels.size(), 1u);
  const ConSanKernelInfo &kernel = result.kernels.front();
  EXPECT_EQ(kernel.stats.lds_write_count, 1u);
  EXPECT_EQ(kernel.stats.decode_error_count, 1u);
  EXPECT_EQ(kernel.preflight_action, ConSanPreflightAction::Skip);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_NE(result.warnings.back().find("found no supported"), std::string::npos);
}

TEST(ConSan, InvalidCdna4KernelDoesNotContainValidCandidatePatching) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.arch = ROCJITSU_CODE_ARCH_CDNA4;
  std::vector<uint8_t> bytes = make_two_kernel_shared_helper_code_object(fixture);
  constexpr uint64_t kTextFileOffset = 0x100;
  const std::array<uint32_t, 2> invalid_kernel = {
      0xD3AD0000u, // unsupported standalone CDNA4 VOP3P op 45
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  const std::array<uint32_t, 2> valid_kernel = {
      0xD81A0000u,
      0x00000000u, // ds_store_b32 v0, v0
  };
  std::memcpy(bytes.data() + kTextFileOffset, invalid_kernel.data(), sizeof(invalid_kernel));
  std::memcpy(bytes.data() + kTextFileOffset + sizeof(invalid_kernel), valid_kernel.data(),
              sizeof(valid_kernel));

  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 8;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  const auto invalid = std::ranges::find(result.kernels, "shared_owner_0", &ConSanKernelInfo::name);
  const auto valid = std::ranges::find(result.kernels, "shared_owner_1", &ConSanKernelInfo::name);
  ASSERT_NE(invalid, result.kernels.end());
  ASSERT_NE(valid, result.kernels.end());
  EXPECT_EQ(invalid->stats.decode_error_count, 1u);
  EXPECT_EQ(invalid->preflight_action, ConSanPreflightAction::Skip);
  EXPECT_EQ(valid->preflight_action, ConSanPreflightAction::Candidate);
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().anchor_offset, sizeof(invalid_kernel));
}

} // namespace
} // namespace rocjitsu
