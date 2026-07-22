// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gfx950_test_fixtures.h
/// @brief Self-contained, header-only builders for minimal gfx950 (CDNA4) ELFs
///        used by DBI probe/spill tests: a single-kernel target ELF (with a
///        discoverable `.kd` descriptor) and a probe ELF exporting one function
///        symbol, plus small readback helpers.
///
/// These mirror the in-file helpers in tests/patch/instrumentor_test.cpp (which
/// documents that duplicating self-contained ELF builders across test slices is
/// the accepted pattern here). They live in namespace rocjitsu::test so a test
/// TU that needs to both patch (Instrumentor) and execute (simulator) a code
/// object can share one copy without disturbing the large existing test files.

#pragma once

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/code_object.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <array>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <vector>

namespace rocjitsu::test {

//==============================================================================
// Instruction word constants (gfx950)
//==============================================================================

// VOP1 v_mov_b32 encodings (gfx950: [31:25]=0x3F, vdst[24:17], op=1<<9,
// src0[8:0]; VGPR src = 256 + index, SGPR src = index, inline 0 = 128,
// inline 1..64 = 129..192).
inline constexpr uint32_t kMovV3V2 = 0x7E060302u;   // v_mov_b32 v3, v2 -> reads v2.
inline constexpr uint32_t kMovV2Zero = 0x7E040280u; // v_mov_b32 v2, 0  -> clobbers v2.
inline constexpr uint32_t kMovV3S8 = 0x7E060208u;   // v_mov_b32 v3, s8 -> reads s8 (s8 live).
inline constexpr uint32_t kMovS8Zero = 0xbe880080u; // s_mov_b32 s8, 0  -> clobbers s8.

// s_mov_b32 <special>, 0: probe bodies that clobber special machine state the
// trampoline preserves across the call (exec_lo=126, vcc_lo=106, m0=124).
inline constexpr uint32_t kMovExecLoZero = 0xbefe0080u;    // s_mov_b32 exec_lo, 0 -> clobbers EXEC.
inline constexpr uint32_t kMovVccLoZero = 0xbeea0080u;     // s_mov_b32 vcc_lo, 0  -> clobbers VCC.
inline constexpr uint32_t kMovM0Zero = 0xbefc0080u;        // s_mov_b32 m0, 0      -> clobbers M0.
inline constexpr uint32_t kMovFlatScrLoZero = 0xbee60080u; // s_mov_b32 flat_scratch_lo, 0.

// v_mov_b32 v2, <inline const K> for K in [0, 64]. Inline constant 0 is encoded
// as 128, and 1..64 as 129..192, in the src0 field (bits [8:0]).
[[nodiscard]] inline constexpr uint32_t make_mov_v2_inline(uint32_t k) {
  const uint32_t src0 = (k == 0) ? 128u : (128u + k); // 129..192 for 1..64.
  return 0x7E040200u | (src0 & 0x1FFu);
}

// s_setpc_b64 s[30:31] (GFX9 family): a minimal probe body tail that returns
// through the link pair, so build_probe_callable accepts it.
inline constexpr uint32_t kProbeSetpcS30S31 = 0xbe801d1eu;

// s_mov_b32 s30, 0 (GFX9 family): overwrites the low half of the return-link
// pair. A body running this before the closing s_setpc still passes
// build_probe_callable (which only inspects the final instruction) but must be
// rejected because it would return through a corrupted PC.
inline constexpr uint32_t kProbeMovS30_0 = 0xbe9e0080u;

// Distinguishable leading marker words for multi-probe layout tests. Each is a
// harmless, self-contained op the probe verifier accepts (not a call, scratch
// access, nor a write to the link pair). They must not collide with the anchor
// instruction (s_nop) nor any envelope opcode, so a test can tell one copied
// probe body from another in the appended cave. s5/s6 are dead in the fixtures
// and are not the low registers the planner picks for target/scc.
inline constexpr uint32_t kProbeMarkerMovS5 = 0xbe850080u; // s_mov_b32 s5, 0
inline constexpr uint32_t kProbeMarkerMovS6 = 0xbe860080u; // s_mov_b32 s6, 0

//==============================================================================
// ELF-image string/alignment helpers
//==============================================================================

inline uint32_t add_elf_name(std::vector<uint8_t> &names, std::string_view name) {
  const uint32_t offset = static_cast<uint32_t>(names.size());
  names.insert(names.end(), name.begin(), name.end());
  names.push_back('\0');
  return offset;
}

inline uint64_t align_up_for_test(uint64_t value, uint64_t alignment) {
  const uint64_t remainder = value % alignment;
  return remainder == 0 ? value : value + alignment - remainder;
}

//==============================================================================
// Target ELF: one kernel with a discoverable `.kd` descriptor
//==============================================================================

// gfx950 ET_DYN ELF with one kernel: a .kd descriptor (scratch = private_bytes,
// SGPR granulation big enough for the s[30:31] link pair) plus a .text holding
// `text_words`, entry at .text offset 0. The `.kd` symbol is a global
// STT_OBJECT of descriptor size so AmdGpuCodeObject::kernel_descriptors() (and
// replace_text) can find and keep it coherent. Modeled on the in-file
// make_gfx950_kernel_elf in instrumentor_test.cpp.
inline std::vector<uint8_t> make_gfx950_kernel_elf(const std::vector<uint32_t> &text_words,
                                                   uint32_t private_bytes,
                                                   uint32_t granulated_sgpr_count = 3) {
  namespace kd = rocr::llvm::amdhsa;
  using KD = kd::kernel_descriptor_t;

  constexpr uint64_t text_offset = 0x100;
  constexpr uint64_t text_vaddr = 0x1100;
  const uint64_t text_size = text_words.size() * sizeof(uint32_t);
  constexpr uint64_t load_align = 0x1000;
  constexpr uint64_t rodata_size = sizeof(KD);

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_elf_name(shstrtab, ".text");
  const uint32_t rodata_name = add_elf_name(shstrtab, ".rodata");
  const uint32_t symtab_name = add_elf_name(shstrtab, ".symtab");
  const uint32_t strtab_name = add_elf_name(shstrtab, ".strtab");
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");

  std::vector<uint8_t> strtab{'\0'};
  const uint32_t kd_symbol_name = add_elf_name(strtab, "test_kernel.kd");

  const uint64_t rodata_offset = text_offset + text_size;
  const uint64_t rodata_vaddr = text_vaddr + text_size + load_align;
  const uint64_t strtab_offset = rodata_offset + rodata_size;
  const uint64_t symtab_offset = align_up_for_test(strtab_offset + strtab.size(), 8);
  constexpr size_t sym_count = 2;
  const uint64_t shstrtab_offset = symtab_offset + sym_count * sizeof(Elf64_Sym);
  const uint64_t shoff = align_up_for_test(shstrtab_offset + shstrtab.size(), 8);
  constexpr uint16_t section_count = 6;
  constexpr uint16_t phdr_count = 2;

  std::vector<uint8_t> image(shoff + section_count * sizeof(Elf64_Shdr), 0);

  Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  ehdr.e_type = ET_DYN;
  ehdr.e_machine = EM_AMDGPU;
  ehdr.e_version = 1;
  ehdr.e_phoff = sizeof(Elf64_Ehdr);
  ehdr.e_shoff = shoff;
  ehdr.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX950;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_phentsize = sizeof(Elf64_Phdr);
  ehdr.e_phnum = phdr_count;
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = section_count;
  ehdr.e_shstrndx = 5;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  std::array<Elf64_Phdr, phdr_count> phdrs{};
  phdrs[0].p_type = PT_LOAD;
  phdrs[0].p_flags = 0x5; // PF_R | PF_X
  phdrs[0].p_offset = text_offset;
  phdrs[0].p_vaddr = text_vaddr;
  phdrs[0].p_paddr = text_vaddr;
  phdrs[0].p_filesz = text_size;
  phdrs[0].p_memsz = text_size;
  phdrs[0].p_align = load_align;

  phdrs[1].p_type = PT_LOAD;
  phdrs[1].p_flags = 0x4; // PF_R
  phdrs[1].p_offset = rodata_offset;
  phdrs[1].p_vaddr = rodata_vaddr;
  phdrs[1].p_paddr = rodata_vaddr;
  phdrs[1].p_filesz = rodata_size;
  phdrs[1].p_memsz = rodata_size;
  phdrs[1].p_align = load_align;
  std::memcpy(image.data() + ehdr.e_phoff, phdrs.data(), phdrs.size() * sizeof(Elf64_Phdr));

  std::memcpy(image.data() + text_offset, text_words.data(), text_size);

  // Entry at .text offset 0; scratch and SGPR granulation set for spilling.
  KD desc{};
  desc.private_segment_fixed_size = private_bytes;
  desc.kernel_code_entry_byte_offset =
      static_cast<int64_t>(text_vaddr) - static_cast<int64_t>(rodata_vaddr);
  AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  granulated_sgpr_count);
  std::memcpy(image.data() + rodata_offset, &desc, sizeof(desc));
  std::memcpy(image.data() + strtab_offset, strtab.data(), strtab.size());

  std::array<Elf64_Sym, sym_count> syms{};
  syms[1].st_name = kd_symbol_name;
  syms[1].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeObject);
  syms[1].st_shndx = 2; // .rodata
  syms[1].st_value = rodata_vaddr;
  syms[1].st_size = sizeof(KD);
  std::memcpy(image.data() + symtab_offset, syms.data(), syms.size() * sizeof(Elf64_Sym));

  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());

  std::array<Elf64_Shdr, section_count> shdrs{};
  shdrs[1].sh_name = text_name;
  shdrs[1].sh_type = SHT_PROGBITS;
  shdrs[1].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  shdrs[1].sh_addr = text_vaddr;
  shdrs[1].sh_offset = text_offset;
  shdrs[1].sh_size = text_size;
  shdrs[1].sh_addralign = sizeof(uint32_t);

  shdrs[2].sh_name = rodata_name;
  shdrs[2].sh_type = SHT_PROGBITS;
  shdrs[2].sh_flags = SHF_ALLOC;
  shdrs[2].sh_addr = rodata_vaddr;
  shdrs[2].sh_offset = rodata_offset;
  shdrs[2].sh_size = rodata_size;
  shdrs[2].sh_addralign = 64;

  shdrs[3].sh_name = symtab_name;
  shdrs[3].sh_type = SHT_SYMTAB;
  shdrs[3].sh_offset = symtab_offset;
  shdrs[3].sh_size = syms.size() * sizeof(Elf64_Sym);
  shdrs[3].sh_link = 4;
  shdrs[3].sh_info = 1;
  shdrs[3].sh_addralign = 8;
  shdrs[3].sh_entsize = sizeof(Elf64_Sym);

  shdrs[4].sh_name = strtab_name;
  shdrs[4].sh_type = SHT_STRTAB;
  shdrs[4].sh_offset = strtab_offset;
  shdrs[4].sh_size = strtab.size();
  shdrs[4].sh_addralign = 1;

  shdrs[5].sh_name = shstrtab_name;
  shdrs[5].sh_type = SHT_STRTAB;
  shdrs[5].sh_offset = shstrtab_offset;
  shdrs[5].sh_size = shstrtab.size();
  shdrs[5].sh_addralign = 1;

  std::memcpy(image.data() + shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));
  return image;
}

//==============================================================================
// Probe ELF: one exported STT_FUNC symbol
//==============================================================================

// gfx950 ELF exporting one STT_FUNC probe symbol whose body is `body_words`, in
// an executable .text. Sections: [1]=.text, [2]=.strtab, [3]=.symtab,
// [4]=.shstrtab. Mirrors make_gfx950_probe_elf in instrumentor_test.cpp.
inline std::vector<uint8_t> make_gfx950_probe_elf(std::string_view symbol,
                                                  const std::vector<uint32_t> &body_words) {
  const uint64_t text_offset = 0x100;
  const uint64_t text_size = body_words.size() * sizeof(uint32_t);

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_elf_name(shstrtab, ".text");
  const uint32_t strtab_name = add_elf_name(shstrtab, ".strtab");
  const uint32_t symtab_name = add_elf_name(shstrtab, ".symtab");
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");

  std::vector<uint8_t> strtab{'\0'};
  const uint32_t sym_name = add_elf_name(strtab, symbol);

  std::array<Elf64_Sym, 2> syms{};
  syms[1].st_name = sym_name;
  syms[1].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeFunc);
  syms[1].st_shndx = 1; // .text
  syms[1].st_value = 0;
  syms[1].st_size = text_size;

  const uint64_t strtab_offset = text_offset + text_size;
  const uint64_t symtab_offset = align_up_for_test(strtab_offset + strtab.size(), 8);
  const uint64_t shstrtab_offset = symtab_offset + syms.size() * sizeof(Elf64_Sym);
  const uint64_t shoff = align_up_for_test(shstrtab_offset + shstrtab.size(), 8);
  constexpr uint16_t section_count = 5;

  std::vector<uint8_t> image(shoff + section_count * sizeof(Elf64_Shdr), 0);

  Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  ehdr.e_type = ET_REL;
  ehdr.e_machine = EM_AMDGPU;
  ehdr.e_version = 1;
  ehdr.e_shoff = shoff;
  ehdr.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX950;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = section_count;
  ehdr.e_shstrndx = 4;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  std::memcpy(image.data() + text_offset, body_words.data(), text_size);
  std::memcpy(image.data() + strtab_offset, strtab.data(), strtab.size());
  std::memcpy(image.data() + symtab_offset, syms.data(), syms.size() * sizeof(Elf64_Sym));
  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());

  std::array<Elf64_Shdr, section_count> shdrs{};
  shdrs[1].sh_name = text_name;
  shdrs[1].sh_type = SHT_PROGBITS;
  shdrs[1].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  shdrs[1].sh_offset = text_offset;
  shdrs[1].sh_size = text_size;
  shdrs[1].sh_addralign = sizeof(uint32_t);

  shdrs[2].sh_name = strtab_name;
  shdrs[2].sh_type = SHT_STRTAB;
  shdrs[2].sh_offset = strtab_offset;
  shdrs[2].sh_size = strtab.size();
  shdrs[2].sh_addralign = 1;

  shdrs[3].sh_name = symtab_name;
  shdrs[3].sh_type = SHT_SYMTAB;
  shdrs[3].sh_offset = symtab_offset;
  shdrs[3].sh_size = syms.size() * sizeof(Elf64_Sym);
  shdrs[3].sh_link = 2; // .strtab
  shdrs[3].sh_info = 1; // index of first global symbol
  shdrs[3].sh_entsize = sizeof(Elf64_Sym);
  shdrs[3].sh_addralign = 8;

  shdrs[4].sh_name = shstrtab_name;
  shdrs[4].sh_type = SHT_STRTAB;
  shdrs[4].sh_offset = shstrtab_offset;
  shdrs[4].sh_size = shstrtab.size();
  shdrs[4].sh_addralign = 1;

  std::memcpy(image.data() + shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));
  return image;
}

//==============================================================================
// Readback helpers
//==============================================================================

// Copy a named section's bytes out of a reparsed code object as 32-bit words.
inline std::vector<uint32_t> section_words(const AmdGpuCodeObject &obj, std::string_view name) {
  for (const auto &sec : obj.all_sections()) {
    if (sec->name() != name)
      continue;
    std::vector<uint32_t> words(sec->size() / sizeof(uint32_t));
    std::memcpy(words.data(), sec->data(), words.size() * sizeof(uint32_t));
    return words;
  }
  return {};
}

// Read back the (single) kernel's scratch size from a patched ELF.
inline uint32_t patched_private_segment_size(const AmdGpuCodeObject &obj) {
  const auto kernels = obj.kernel_descriptors();
  return kernels.size() == 1 ? kernels.front().private_segment_fixed_size : 0xFFFFFFFFu;
}

} // namespace rocjitsu::test
