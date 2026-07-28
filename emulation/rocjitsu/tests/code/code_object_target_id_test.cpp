// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file code_object_target_id_test.cpp
/// @brief Tests for AMDGPU code-object target identification: that the ELF
///        machine-flag field (e_flags & EF_AMDGPU_MACH) maps to the right
///        rj_code_target_id_t value, and that the corresponding C API path
///        (rj_code_executable_create -> get_code_object ->
///        basic_block_list_create) accepts each target by exercising the
///        provider-selected target registry in rj_code.cpp.
///
/// Covers the only currently supported targets (gfx90a, gfx942, gfx950,
/// gfx1200, gfx1201, gfx1250) plus an unknown-machine-flag case to guard the
/// INVALID sentinel and prevent a future edit from silently aliasing one target
/// onto another.

// \NPI new GPU: extend these tests with its provider-owned MACH/triple binding.
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/kernel_symbol.h"
#include "rocjitsu/code/rj_code.h"
#include "scoped_temp.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {
namespace {

// Helpers duplicated from tests/dbt/translate_test.cpp /
// tests/patch/instrumentor_test.cpp.
// TODO: extract into a shared fixture header

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

// Minimal AMDGPU ELF tagged with @p mach_flag. .text contains two `s_nop 0`
// words so it can be decoded by any AMDGPU ISA (SOPP encoding is shared).
std::vector<uint8_t> make_minimal_amdgpu_elf(uint32_t mach_flag) {
  constexpr uint64_t text_offset = 0x100;
  constexpr uint64_t text_size = 8;
  constexpr uint64_t rodata_size = 4;

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_elf_name(shstrtab, ".text");
  const uint32_t rodata_name = add_elf_name(shstrtab, ".rodata");
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");

  const uint64_t rodata_offset = text_offset + text_size;
  const uint64_t shstrtab_offset = rodata_offset + rodata_size;
  const uint64_t shoff = align_up(shstrtab_offset + shstrtab.size(), 8);
  constexpr uint16_t section_count = 4;

  std::vector<uint8_t> image(shoff + section_count * sizeof(Elf64_Shdr), 0);

  Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  ehdr.e_type = ET_REL;
  ehdr.e_machine = EM_AMDGPU;
  ehdr.e_version = 1;
  ehdr.e_shoff = shoff;
  ehdr.e_flags = mach_flag;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = section_count;
  ehdr.e_shstrndx = 3;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  const std::array<uint32_t, 2> text_words = {0xBF800000u, 0xBF800000u};
  std::memcpy(image.data() + text_offset, text_words.data(), text_size);

  const uint32_t rodata_word = 0xA5A55A5Au;
  std::memcpy(image.data() + rodata_offset, &rodata_word, sizeof(rodata_word));
  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());

  std::array<Elf64_Shdr, section_count> shdrs{};
  shdrs[1].sh_name = text_name;
  shdrs[1].sh_type = SHT_PROGBITS;
  shdrs[1].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  shdrs[1].sh_offset = text_offset;
  shdrs[1].sh_size = text_size;
  shdrs[1].sh_addralign = sizeof(uint32_t);

  shdrs[2].sh_name = rodata_name;
  shdrs[2].sh_type = SHT_PROGBITS;
  shdrs[2].sh_flags = SHF_ALLOC;
  shdrs[2].sh_offset = rodata_offset;
  shdrs[2].sh_size = rodata_size;
  shdrs[2].sh_addralign = sizeof(uint32_t);

  shdrs[3].sh_name = shstrtab_name;
  shdrs[3].sh_type = SHT_STRTAB;
  shdrs[3].sh_offset = shstrtab_offset;
  shdrs[3].sh_size = shstrtab.size();
  shdrs[3].sh_addralign = 1;

  std::memcpy(image.data() + shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));
  return image;
}

// Build a minimal ELF tagged with @p mach_flag, load it as an
// AmdGpuCodeObject, and assert target_id() resolves to @p expected.
void expect_machine_flag_maps_to_target(uint32_t mach_flag, rj_code_target_id_t expected) {
  auto image = make_minimal_amdgpu_elf(mach_flag);
  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());
  EXPECT_EQ(obj.target_id(), expected);
}

// Drive the C API path that internally calls create_decoder_for_target:
// write a minimal ELF for @p mach_flag to a temp file, open it as an
// executable, fetch the code object for @p target, and build a basic-block
// list. Cleans up after.
void expect_c_api_accepts_target(uint32_t mach_flag, rj_code_target_id_t target) {
  auto image = make_minimal_amdgpu_elf(mach_flag);

  test::ScopedTempFile file("rj-code-object-target-id-");
  file.write(std::string_view(reinterpret_cast<const char *>(image.data()), image.size()));

  rj_code_executable_t *exec = nullptr;
  ASSERT_EQ(rj_code_executable_create(file.path().c_str(), &exec), ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(exec, nullptr);

  ASSERT_GT(rj_code_executable_num_code_objects(exec, target), 0u)
      << "executable must expose at least one code object for the requested target";

  rj_code_object_t *obj = nullptr;
  ASSERT_EQ(rj_code_executable_get_code_object(exec, target, 0, &obj), ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(obj, nullptr);

  // The returned code-object handle must keep its executable storage alive.
  rj_code_executable_destroy(exec);

  rj_code_basic_block_list_t *blocks = nullptr;
  EXPECT_EQ(rj_code_basic_block_list_create(obj, target, &blocks), ROCJITSU_STATUS_SUCCESS)
      << "rj_code_basic_block_list_create must succeed for a target whose decoder is wired in";
  ASSERT_NE(blocks, nullptr);

  rj_code_basic_block_t *block = nullptr;
  ASSERT_EQ(rj_code_basic_block_list_get(blocks, 0, &block), ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(block, nullptr);

  // Likewise, a returned block must keep its list and decoded instructions alive.
  rj_code_basic_block_list_destroy(blocks);
  EXPECT_EQ(rj_code_basic_block_start_offset(block), 0u);
  EXPECT_EQ(rj_code_basic_block_num_instructions(block), 2u);

  // Cleanup follows the refcount discipline in refcount.h:
  //   - block  came from _get()      -> refcount 1 -> destroy + release.
  //   - obj    came from _get_code_object() -> refcount 1 -> destroy + release.
  // Their destroyed parents are released automatically with the child handles.
  rj_code_basic_block_destroy(block);
  rj_code_basic_block_release(block);
  rj_code_object_destroy(obj);
  rj_code_object_release(obj);
}

//==============================================================================
// Machine flag -> target_id (one test per supported target)
//==============================================================================

TEST(GfxCodeObjectTargets, LoadsGfx90aFromMachineFlags) {
  expect_machine_flag_maps_to_target(EF_AMDGPU_MACH_AMDGCN_GFX90A, ROCJITSU_CODE_TARGET_GFX90A);
}

TEST(GfxCodeObjectTargets, LoadsGfx942FromMachineFlags) {
  expect_machine_flag_maps_to_target(EF_AMDGPU_MACH_AMDGCN_GFX942, ROCJITSU_CODE_TARGET_GFX942);
}

TEST(GfxCodeObjectTargets, LoadsGfx950FromMachineFlags) {
  expect_machine_flag_maps_to_target(EF_AMDGPU_MACH_AMDGCN_GFX950, ROCJITSU_CODE_TARGET_GFX950);
}

TEST(GfxCodeObjectTargets, LoadsGfx1200FromMachineFlags) {
  expect_machine_flag_maps_to_target(EF_AMDGPU_MACH_AMDGCN_GFX1200, ROCJITSU_CODE_TARGET_GFX1200);
}

TEST(GfxCodeObjectTargets, LoadsGfx1201FromMachineFlags) {
  expect_machine_flag_maps_to_target(EF_AMDGPU_MACH_AMDGCN_GFX1201, ROCJITSU_CODE_TARGET_GFX1201);
}

TEST(GfxCodeObjectTargets, LoadsGfx1250FromMachineFlags) {
  expect_machine_flag_maps_to_target(EF_AMDGPU_MACH_AMDGCN_GFX1250, ROCJITSU_CODE_TARGET_GFX1250);
}

// Machine flags outside the supported set must surface as
// ROCJITSU_CODE_TARGET_INVALID rather than silently aliasing onto a real
// target (which would happen if someone accidentally made a real target the
// default arm of the switch).
TEST(GfxCodeObjectTargets, UnknownMachineFlagMapsToInvalid) {
  // 0x1234 is not any defined EF_AMDGPU_MACH_AMDGCN_* value.
  auto image = make_minimal_amdgpu_elf(/*mach_flag=*/0x1234);
  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid()) << "ELF should still parse; only target_id is unknown";
  EXPECT_EQ(obj.target_id(), ROCJITSU_CODE_TARGET_INVALID);
}

//==============================================================================
// C API path (rj_code_executable_create + ... + basic_block_list_create)
// exercises create_decoder_for_target for each supported target.
//==============================================================================

TEST(GfxCodeObjectTargets, CApiAcceptsGfx90aForBasicBlockList) {
  expect_c_api_accepts_target(EF_AMDGPU_MACH_AMDGCN_GFX90A, ROCJITSU_CODE_TARGET_GFX90A);
}

TEST(GfxCodeObjectTargets, CApiAcceptsGfx942ForBasicBlockList) {
  expect_c_api_accepts_target(EF_AMDGPU_MACH_AMDGCN_GFX942, ROCJITSU_CODE_TARGET_GFX942);
}

TEST(GfxCodeObjectTargets, CApiAcceptsGfx950ForBasicBlockList) {
  expect_c_api_accepts_target(EF_AMDGPU_MACH_AMDGCN_GFX950, ROCJITSU_CODE_TARGET_GFX950);
}

TEST(GfxCodeObjectTargets, CApiAcceptsGfx1200ForBasicBlockList) {
  expect_c_api_accepts_target(EF_AMDGPU_MACH_AMDGCN_GFX1200, ROCJITSU_CODE_TARGET_GFX1200);
}

TEST(GfxCodeObjectTargets, CApiAcceptsGfx1201ForBasicBlockList) {
  expect_c_api_accepts_target(EF_AMDGPU_MACH_AMDGCN_GFX1201, ROCJITSU_CODE_TARGET_GFX1201);
}

TEST(GfxCodeObjectTargets, CApiAcceptsGfx1250ForBasicBlockList) {
  expect_c_api_accepts_target(EF_AMDGPU_MACH_AMDGCN_GFX1250, ROCJITSU_CODE_TARGET_GFX1250);
}

TEST(KernelSymbolTest, DemanglesMangledKernelSymbol) {
  EXPECT_EQ(demangle_kernel_symbol("_Z11racy_kernelPKfPf"), "racy_kernel(float const*, float*)");
}

TEST(KernelSymbolTest, DisplayNameIsHeaderSafe) {
  constexpr std::string_view tensile_symbol =
      "Cijk_Ailk_Bjlk_S_B_UserArgs_MT8x8x8_SN_LDSB0_ISA1151_WG8_8_1_WGMXCC1";

  EXPECT_EQ(kernel_display_name("_Z11racy_kernelPKfPf"), "racy_kernel");

  // This is a fixture for a real-world HIP/Clang-style nested template symbol.
  // The important behavior is that display names keep useful template context
  // while stripping the argument list and whitespace that would break the
  // race-detector's space-delimited log headers.
  EXPECT_EQ(kernel_display_name("_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_"
                                "15CUDAFunctor_addIfEESt5arrayIPcLm3EEEEviT0_T1_"),
            "at::native::vectorized_elementwise_kernel<4,at::native::CUDAFunctor_add<float>,std::"
            "array<char*,3ul>>");
  EXPECT_EQ(kernel_display_name("_ZN12_GLOBAL__N_16kernelEPf"), "(anonymousnamespace)::kernel");
  EXPECT_EQ(kernel_display_name("_Z3fooIPFviEEvv"), "foo<void(*)(int)>");
  EXPECT_EQ(kernel_display_name("__amd_rocclr_copyBuffer"), "__amd_rocclr_copyBuffer");
  EXPECT_EQ(kernel_display_name(tensile_symbol), tensile_symbol);
}

} // namespace
} // namespace rocjitsu
