// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file dbi_test_util.h
/// @brief Self-contained, header-only builders for minimal AMDGPU ELFs used by DBI
///        probe/spill tests: a single-kernel target ELF (with a discoverable `.kd`
///        descriptor) and a probe ELF exporting one function symbol, plus small
///        readback helpers. The generic `make_amdgpu_*` cores take an ELF machine
///        flag; `make_gfx942_*` (CDNA3), `make_gfx950_*` (CDNA4), and
///        `make_gfx1200_*` (RDNA4) are thin wrappers over them.
///
/// Lives at the tests/ root (not a test slice) because it is shared across
/// tests/patch, tests/dbi, and tests/code. Everything is in namespace
/// rocjitsu::test so a TU that needs to both patch (Instrumentor) and execute
/// (simulator) a code object can share one copy.

#pragma once

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/code_object.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"

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

// Instruction word constants (gfx950)

// VOP1 v_mov_b32 encodings (gfx950: [31:25]=0x3F, vdst[24:17], op=1<<9,
// src0[8:0]; VGPR src = 256 + index, SGPR src = index, inline 0 = 128,
// inline 1..64 = 129..192).
inline constexpr uint32_t kMovV3V2 = 0x7E060302u;   // v_mov_b32 v3, v2 -> reads v2.
inline constexpr uint32_t kMovV2Zero = 0x7E040280u; // v_mov_b32 v2, 0  -> clobbers v2.
inline constexpr uint32_t kMovV3S8 = 0x7E060208u;   // v_mov_b32 v3, s8 -> reads s8 (s8 live).
inline constexpr uint32_t kMovS8Zero = 0xbe880080u; // s_mov_b32 s8, 0  -> clobbers s8.
inline constexpr uint32_t kMovV4S9 = 0x7E080209u;   // v_mov_b32 v4, s9 -> reads s9 (s9 live).
inline constexpr uint32_t kMovS9Zero = 0xbe890080u; // s_mov_b32 s9, 0  -> clobbers s9.
inline constexpr uint32_t kMovV5V0 = 0x7E0A0300u;   // v_mov_b32 v5, v0 -> reads v0 into v5.
inline constexpr uint32_t kMovV5V1 = 0x7E0A0301u;   // v_mov_b32 v5, v1 -> reads v1 into v5.
inline constexpr uint32_t kMovV5V2 = 0x7E0A0302u;   // v_mov_b32 v5, v2 -> reads v2 into v5.
inline constexpr uint32_t kMovV5V3 = 0x7E0A0303u;   // v_mov_b32 v5, v3 -> reads v3 into v5.
inline constexpr uint32_t kMovV6S8 = 0x7E0C0208u;   // v_mov_b32 v6, s8 -> reads s8 into v6.

// v_mov_b32 v{dst}, <inline const K> for K in [0, 64]. vdst occupies bits [24:17];
// inline constant 0 is encoded as 128, and 1..64 as 129..192, in the src0 field (bits [8:0]).
[[nodiscard]] inline constexpr uint32_t make_mov_vgpr_inline(uint16_t dst, uint32_t k) {
  const uint32_t src0 = (k == 0) ? 128u : (128u + k); // 129..192 for 1..64.
  return 0x7E000200u | (static_cast<uint32_t>(dst) << 17) | (src0 & 0x1FFu);
}

// AccVGPR VOP3 (gfx942/gfx950, identical encodings), two words each.
// v_accvgpr_read_b32 v3, a0  -> reads acc0 (acc0 live before the anchor).
inline constexpr uint32_t kAccReadV3A0Lo = 0xD3D84003u;
inline constexpr uint32_t kAccReadV3A0Hi = 0x18000100u;
// v_accvgpr_read_b32 v7, a0  -> reads acc0 into v7.
inline constexpr uint32_t kAccReadV7A0Lo = 0xD3D84007u;
inline constexpr uint32_t kAccReadV7A0Hi = 0x18000100u;
// v_accvgpr_write_b32 a0, 0  -> clobbers acc0.
inline constexpr uint32_t kAccWriteA0ZeroLo = 0xD3D94000u;
inline constexpr uint32_t kAccWriteA0ZeroHi = 0x18000080u;

// v_accvgpr_write_b32 a0, <inline const K> for K in [0, 64]: the lo word is
// kAccWriteA0ZeroLo; the hi word carries the inline src0 (128 for 0, 128+K for
// 1..64), same src0 field encoding as make_mov_vgpr_inline.
[[nodiscard]] inline constexpr uint32_t make_accvgpr_write_a0_inline_hi(uint32_t k) {
  const uint32_t src0 = (k == 0) ? 128u : (128u + k); // 129..192 for 1..64.
  return 0x18000000u | (src0 & 0x1FFu);
}

// v_accvgpr_write_b32 a{acc}, 0 -> clobbers acc{acc}. The lo word carries the
// accumulator index in vdst bits [7:0] (kAccWriteA0ZeroLo is the acc0 case); the
// hi word is kAccWriteA0ZeroHi (inline src0 = 0). acc in [0, 255]. Lets a probe
// clobber a whole accumulator tuple with one write per lane.
[[nodiscard]] inline constexpr uint32_t make_accvgpr_write_lo(uint16_t acc) {
  return kAccWriteA0ZeroLo | (acc & 0xFFu);
}

// v_mfma_f32_16x16x16_f16 a[0:3], v[0:1], v[2:3], a[0:3] (gfx942/gfx950, identical
// encodings), two words. acc_cd=1 makes both the C source and D destination the acc[0:3]
// tuple, so the instruction reads and writes all four accumulator lanes (A=v[0:1],
// B=v[2:3]); it gives a test AccVGPR liveness through an MFMA acc_cd operand rather than
// one-lane v_accvgpr_read/write words.
inline constexpr uint32_t kMfmaF32_16x16x16F16_A0to3_Lo = 0xD3CD8000u;
inline constexpr uint32_t kMfmaF32_16x16x16F16_A0to3_Hi = 0x04020500u;

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

// ELF-image string/alignment helpers

inline uint32_t add_elf_name(std::vector<uint8_t> &names, std::string_view name) {
  const uint32_t offset = static_cast<uint32_t>(names.size());
  names.resize(offset + name.size() + 1);
  if (!name.empty()) {
    std::memcpy(names.data() + offset, name.data(), name.size());
  }
  return offset;
}

inline uint64_t align_up_for_test(uint64_t value, uint64_t alignment) {
  const uint64_t remainder = value % alignment;
  return remainder == 0 ? value : value + alignment - remainder;
}

// Target ELF: one kernel with a discoverable `.kd` descriptor

// ET_DYN ELF with one kernel: a .kd descriptor (scratch = private_bytes, SGPR
// granulation big enough for the s[30:31] link pair) plus a .text holding
// `text_words`, entry at .text offset 0. The `.kd` symbol is a global STT_OBJECT
// of descriptor size so scan_kernel_descriptors() (and replace_text) can find and
// keep it coherent. `e_flags` selects the target ISA (e.g. GFX950 / GFX1200).
// granulated_vgpr_count / accum_offset encode the descriptor's
// GRANULATED_WORKITEM_VGPR_COUNT (VGPR count = (field + 1) * granule) and gfx90a
// ACCUM_OFFSET (AccVGPR base = (field + 1) * 4) fields. The defaults reproduce the
// historical shape: 8 unified VGPRs with the AGPR window starting at v4.
// The trailing bools inject malformed headers to exercise the scanner's rejection
// paths (each keeps the rest of the image well-formed): `unterminated_kd_name` trims
// the .strtab size so the `.kd` name runs off the table end with no NUL;
// `wrap_section_header_table` sets e_shoff so e_shoff + e_shnum*sizeof(Shdr) overflows;
// `wrap_symtab_range` sets the .symtab sh_offset so sh_offset + sh_size overflows;
// `kd_crosses_section` shrinks the .rodata sh_size below sizeof(KD) so the 64-byte `.kd`
// descriptor extends past its owning section into the adjacent one.
inline std::vector<uint8_t>
make_amdgpu_kernel_elf(const std::vector<uint32_t> &text_words, uint32_t private_bytes,
                       uint32_t granulated_sgpr_count, uint32_t e_flags,
                       uint32_t granulated_vgpr_count = 0, uint32_t accum_offset = 0,
                       bool unterminated_kd_name = false, bool wrap_section_header_table = false,
                       bool wrap_symtab_range = false, bool kd_crosses_section = false) {
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
  ehdr.e_flags = e_flags;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_phentsize = sizeof(Elf64_Phdr);
  ehdr.e_phnum = phdr_count;
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = section_count;
  ehdr.e_shstrndx = 5;
  if (wrap_section_header_table)
    ehdr.e_shoff = UINT64_MAX - 8; // e_shoff + e_shnum * sizeof(Elf64_Shdr) overflows
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
  AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  granulated_vgpr_count);
  AMDHSA_BITS_SET(desc.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, accum_offset);
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
  // Declared half-size so the 64-byte `.kd` (still physically present) extends past this
  // section's end, letting the scanner's owning-section bound reject the crossing symbol.
  shdrs[2].sh_size = kd_crosses_section ? rodata_size / 2 : rodata_size;
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
  // Declared size excludes the `.kd` name's NUL, so the scanner sees it unterminated.
  shdrs[4].sh_size =
      unterminated_kd_name ? kd_symbol_name + std::strlen("test_kernel.kd") : strtab.size();
  shdrs[4].sh_addralign = 1;

  shdrs[5].sh_name = shstrtab_name;
  shdrs[5].sh_type = SHT_STRTAB;
  shdrs[5].sh_offset = shstrtab_offset;
  shdrs[5].sh_size = shstrtab.size();
  shdrs[5].sh_addralign = 1;

  if (wrap_symtab_range)
    shdrs[3].sh_offset = UINT64_MAX - 4; // sh_offset + sh_size overflows

  std::memcpy(image.data() + shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));
  return image;
}

// CDNA3 (gfx942) single-kernel target ELF.
inline std::vector<uint8_t> make_gfx942_kernel_elf(const std::vector<uint32_t> &text_words,
                                                   uint32_t private_bytes,
                                                   uint32_t granulated_sgpr_count = 3,
                                                   uint32_t granulated_vgpr_count = 0,
                                                   uint32_t accum_offset = 0) {
  return make_amdgpu_kernel_elf(text_words, private_bytes, granulated_sgpr_count,
                                EF_AMDGPU_MACH_AMDGCN_GFX942, granulated_vgpr_count, accum_offset);
}

// CDNA4 (gfx950) single-kernel target ELF.
inline std::vector<uint8_t> make_gfx950_kernel_elf(const std::vector<uint32_t> &text_words,
                                                   uint32_t private_bytes,
                                                   uint32_t granulated_sgpr_count = 3,
                                                   uint32_t granulated_vgpr_count = 0,
                                                   uint32_t accum_offset = 0) {
  return make_amdgpu_kernel_elf(text_words, private_bytes, granulated_sgpr_count,
                                EF_AMDGPU_MACH_AMDGCN_GFX950, granulated_vgpr_count, accum_offset);
}

// RDNA4 (gfx1200) single-kernel target ELF.
inline std::vector<uint8_t> make_gfx1200_kernel_elf(const std::vector<uint32_t> &text_words,
                                                    uint32_t private_bytes,
                                                    uint32_t granulated_sgpr_count = 3,
                                                    uint32_t granulated_vgpr_count = 0,
                                                    uint32_t accum_offset = 0) {
  return make_amdgpu_kernel_elf(text_words, private_bytes, granulated_sgpr_count,
                                EF_AMDGPU_MACH_AMDGCN_GFX1200, granulated_vgpr_count, accum_offset);
}

// gfx950 target ELF whose `.kd` symbol name runs to the end of its string table
// with no in-bounds NUL terminator, for exercising the scanner's rejection of an
// unterminated descriptor name.
inline std::vector<uint8_t>
make_gfx950_unterminated_kd_name_elf(const std::vector<uint32_t> &text_words,
                                     uint32_t private_bytes) {
  return make_amdgpu_kernel_elf(text_words, private_bytes, /*granulated_sgpr_count=*/3,
                                EF_AMDGPU_MACH_AMDGCN_GFX950, /*granulated_vgpr_count=*/0,
                                /*accum_offset=*/0, /*unterminated_kd_name=*/true);
}

// gfx950 target ELF whose e_shoff + e_shnum*sizeof(Elf64_Shdr) overflows, for
// exercising the scanner's overflow-safe section-header-table bounds check.
inline std::vector<uint8_t> make_gfx950_wrapping_shoff_elf(const std::vector<uint32_t> &text_words,
                                                           uint32_t private_bytes) {
  return make_amdgpu_kernel_elf(text_words, private_bytes, /*granulated_sgpr_count=*/3,
                                EF_AMDGPU_MACH_AMDGCN_GFX950, /*granulated_vgpr_count=*/0,
                                /*accum_offset=*/0, /*unterminated_kd_name=*/false,
                                /*wrap_section_header_table=*/true);
}

// gfx950 target ELF whose .symtab sh_offset + sh_size overflows, for exercising the
// scanner's overflow-safe section-range bounds check (the .text section stays intact
// so discovery still resolves the text base before rejecting the bad symtab).
inline std::vector<uint8_t> make_gfx950_wrapping_symtab_elf(const std::vector<uint32_t> &text_words,
                                                            uint32_t private_bytes) {
  return make_amdgpu_kernel_elf(text_words, private_bytes, /*granulated_sgpr_count=*/3,
                                EF_AMDGPU_MACH_AMDGCN_GFX950, /*granulated_vgpr_count=*/0,
                                /*accum_offset=*/0, /*unterminated_kd_name=*/false,
                                /*wrap_section_header_table=*/false, /*wrap_symtab_range=*/true);
}

// gfx950 target ELF whose `.kd` symbol's 64-byte descriptor extends past its owning
// (.rodata) section into the adjacent section, for exercising the scanner's owning-section
// bound: the descriptor must fit within sh_size, not merely the image.
inline std::vector<uint8_t>
make_gfx950_kd_crossing_section_elf(const std::vector<uint32_t> &text_words,
                                    uint32_t private_bytes) {
  return make_amdgpu_kernel_elf(text_words, private_bytes, /*granulated_sgpr_count=*/3,
                                EF_AMDGPU_MACH_AMDGCN_GFX950, /*granulated_vgpr_count=*/0,
                                /*accum_offset=*/0, /*unterminated_kd_name=*/false,
                                /*wrap_section_header_table=*/false, /*wrap_symtab_range=*/false,
                                /*kd_crosses_section=*/true);
}

// Like make_gfx950_kernel_elf but exports *two* `.kd` descriptors so
// kernel_descriptors() returns two kernels, exercising the orchestrator's
// single-kernel spill guard -- where only the count matters, which is why both
// entries default to .text offset 0.
//
// @p second_entry_text_offset moves the second kernel's entry, giving the two
// descriptors distinct entries and therefore distinct kernel scopes. Pass it
// when the test needs per-kernel analysis to differ between them.
inline std::vector<uint8_t> make_gfx950_two_kernel_elf(const std::vector<uint32_t> &text_words,
                                                       uint32_t private_bytes,
                                                       uint32_t granulated_sgpr_count = 3,
                                                       uint64_t second_entry_text_offset = 0) {
  namespace kd = rocr::llvm::amdhsa;
  using KD = kd::kernel_descriptor_t;

  constexpr uint64_t text_offset = 0x100;
  constexpr uint64_t text_vaddr = 0x1100;
  const uint64_t text_size = text_words.size() * sizeof(uint32_t);
  constexpr uint64_t load_align = 0x1000;
  constexpr size_t kKernelCount = 2;
  const uint64_t rodata_size = kKernelCount * sizeof(KD);

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_elf_name(shstrtab, ".text");
  const uint32_t rodata_name = add_elf_name(shstrtab, ".rodata");
  const uint32_t symtab_name = add_elf_name(shstrtab, ".symtab");
  const uint32_t strtab_name = add_elf_name(shstrtab, ".strtab");
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");

  std::vector<uint8_t> strtab{'\0'};
  const uint32_t kd0_symbol_name = add_elf_name(strtab, "test_kernel0.kd");
  const uint32_t kd1_symbol_name = add_elf_name(strtab, "test_kernel1.kd");

  const uint64_t rodata_offset = text_offset + text_size;
  const uint64_t rodata_vaddr = text_vaddr + text_size + load_align;
  const uint64_t strtab_offset = rodata_offset + rodata_size;
  const uint64_t symtab_offset = align_up_for_test(strtab_offset + strtab.size(), 8);
  constexpr size_t sym_count = 1 + kKernelCount; // null + one per kernel
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

  // Two descriptors; entry offset is signed and relative to each descriptor's vaddr.
  for (size_t i = 0; i < kKernelCount; ++i) {
    const uint64_t kd_vaddr = rodata_vaddr + i * sizeof(KD);
    const uint64_t entry_vaddr = text_vaddr + (i == 1 ? second_entry_text_offset : 0);
    KD desc{};
    desc.private_segment_fixed_size = private_bytes;
    desc.kernel_code_entry_byte_offset =
        static_cast<int64_t>(entry_vaddr) - static_cast<int64_t>(kd_vaddr);
    AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                    granulated_sgpr_count);
    std::memcpy(image.data() + rodata_offset + i * sizeof(KD), &desc, sizeof(desc));
  }
  std::memcpy(image.data() + strtab_offset, strtab.data(), strtab.size());

  std::array<Elf64_Sym, sym_count> syms{};
  syms[1].st_name = kd0_symbol_name;
  syms[1].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeObject);
  syms[1].st_shndx = 2; // .rodata
  syms[1].st_value = rodata_vaddr;
  syms[1].st_size = sizeof(KD);
  syms[2].st_name = kd1_symbol_name;
  syms[2].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeObject);
  syms[2].st_shndx = 2; // .rodata
  syms[2].st_value = rodata_vaddr + sizeof(KD);
  syms[2].st_size = sizeof(KD);
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

// Probe ELF: one exported STT_FUNC symbol

// ELF exporting one STT_FUNC probe symbol whose body is `body_words`, in an
// executable .text. Sections: [1]=.text, [2]=.strtab, [3]=.symtab, [4]=.shstrtab.
// `e_flags` selects the target ISA (e.g. GFX950 / GFX1200).
inline std::vector<uint8_t> make_amdgpu_probe_elf(std::string_view symbol,
                                                  const std::vector<uint32_t> &body_words,
                                                  uint32_t e_flags) {
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
  ehdr.e_flags = e_flags;
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

// CDNA3 (gfx942) probe ELF.
inline std::vector<uint8_t> make_gfx942_probe_elf(std::string_view symbol,
                                                  const std::vector<uint32_t> &body_words) {
  return make_amdgpu_probe_elf(symbol, body_words, EF_AMDGPU_MACH_AMDGCN_GFX942);
}

// CDNA4 (gfx950) probe ELF.
inline std::vector<uint8_t> make_gfx950_probe_elf(std::string_view symbol,
                                                  const std::vector<uint32_t> &body_words) {
  return make_amdgpu_probe_elf(symbol, body_words, EF_AMDGPU_MACH_AMDGCN_GFX950);
}

// RDNA4 (gfx1200) probe ELF.
inline std::vector<uint8_t> make_gfx1200_probe_elf(std::string_view symbol,
                                                   const std::vector<uint32_t> &body_words) {
  return make_amdgpu_probe_elf(symbol, body_words, EF_AMDGPU_MACH_AMDGCN_GFX1200);
}

// Readback helpers

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
  if (obj.text_sections().empty())
    return 0xFFFFFFFFu;
  const Section *text = obj.text_sections().front();
  const auto kernels = scan_kernel_descriptors(
      {reinterpret_cast<const uint8_t *>(obj.image_data()), obj.image_size()},
      text->sectionOffset(), text->size());
  return kernels.size() == 1 ? kernels.front().descriptor.private_segment_fixed_size : 0xFFFFFFFFu;
}

} // namespace rocjitsu::test
