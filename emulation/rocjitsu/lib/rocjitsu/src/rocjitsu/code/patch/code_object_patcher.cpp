// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/code_object_patcher.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/dbt/kernel_descriptor_translator.h"
#include "rocjitsu/isa/arch/amdgpu/isa_properties.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <limits>
#include <new>
#include <numeric>
// Standard library
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {

static_assert(sizeof(size_t) >= sizeof(uint64_t),
              "CodeObjectPatcher requires a 64-bit host address space");

using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
namespace kd = rocr::llvm::amdhsa;

[[nodiscard]] std::optional<std::vector<Elf64_Shdr>>
read_section_headers(const std::vector<uint8_t> &image, const Elf64_Ehdr &ehdr) {
  if ((ehdr.e_shoff == 0) != (ehdr.e_shnum == 0))
    return std::nullopt;
  if (ehdr.e_shoff == 0)
    return std::vector<Elf64_Shdr>{};

  if (ehdr.e_shentsize != sizeof(Elf64_Shdr) || ehdr.e_shoff > image.size() ||
      static_cast<uint64_t>(ehdr.e_shnum) >
          (static_cast<uint64_t>(image.size()) - ehdr.e_shoff) / sizeof(Elf64_Shdr))
    return std::nullopt;

  std::vector<Elf64_Shdr> shdrs(ehdr.e_shnum);
  std::memcpy(shdrs.data(), image.data() + ehdr.e_shoff, shdrs.size() * sizeof(Elf64_Shdr));
  return shdrs;
}

[[nodiscard]] std::optional<std::vector<Elf64_Phdr>>
read_program_headers(const std::vector<uint8_t> &image, const Elf64_Ehdr &ehdr) {
  if ((ehdr.e_phoff == 0) != (ehdr.e_phnum == 0))
    return std::nullopt;
  if (ehdr.e_phoff == 0)
    return std::vector<Elf64_Phdr>{};

  if (ehdr.e_phentsize != sizeof(Elf64_Phdr) || ehdr.e_phoff > image.size() ||
      static_cast<uint64_t>(ehdr.e_phnum) >
          (static_cast<uint64_t>(image.size()) - ehdr.e_phoff) / sizeof(Elf64_Phdr))
    return std::nullopt;

  std::vector<Elf64_Phdr> phdrs(ehdr.e_phnum);
  std::memcpy(phdrs.data(), image.data() + ehdr.e_phoff, phdrs.size() * sizeof(Elf64_Phdr));
  return phdrs;
}

void write_elf_tables(std::vector<uint8_t> &image, const Elf64_Ehdr &ehdr,
                      std::span<const Elf64_Shdr> shdrs, std::span<const Elf64_Phdr> phdrs) {
  assert(image.size() >= sizeof(Elf64_Ehdr) && "ELF header out of bounds");
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  assert(ehdr.e_shoff + shdrs.size() * sizeof(Elf64_Shdr) <= image.size() &&
         "section header table write out of bounds");
  std::memcpy(image.data() + ehdr.e_shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));

  if (!phdrs.empty()) {
    assert(ehdr.e_phoff + phdrs.size() * sizeof(Elf64_Phdr) <= image.size() &&
           "program header table write out of bounds");
    std::memcpy(image.data() + ehdr.e_phoff, phdrs.data(), phdrs.size() * sizeof(Elf64_Phdr));
  }
}

[[nodiscard]] bool can_add_u64(uint64_t value, uint64_t delta) {
  return value <= std::numeric_limits<uint64_t>::max() - delta;
}

enum class FileInsertionOutcome : uint8_t {
  Success,
  MalformedInput,
  AllocationFailure,
};

[[nodiscard]] FileInsertionOutcome
insert_file_bytes(std::vector<uint8_t> &image, Elf64_Ehdr &ehdr, std::vector<Elf64_Shdr> &shdrs,
                  std::vector<Elf64_Phdr> &phdrs, uint64_t file_offset,
                  std::span<const uint8_t> bytes, std::optional<size_t> grown_section_index,
                  bool grow_load_at_segment_end) {
  if (file_offset > image.size())
    return FileInsertionOutcome::MalformedInput;
  if (bytes.size() > image.max_size() - image.size())
    return FileInsertionOutcome::AllocationFailure;
  if (bytes.empty())
    return FileInsertionOutcome::Success;

  const uint64_t delta = bytes.size();
  if ((ehdr.e_shoff >= file_offset && !can_add_u64(ehdr.e_shoff, delta)) ||
      (ehdr.e_phoff != 0 && ehdr.e_phoff >= file_offset && !can_add_u64(ehdr.e_phoff, delta)))
    return FileInsertionOutcome::MalformedInput;
  for (size_t i = 0; i < shdrs.size(); ++i) {
    if (grown_section_index && i == *grown_section_index)
      continue;
    if (shdrs[i].sh_type != SHT_NULL && shdrs[i].sh_offset >= file_offset &&
        !can_add_u64(shdrs[i].sh_offset, delta))
      return FileInsertionOutcome::MalformedInput;
  }
  for (const Elf64_Phdr &phdr : phdrs) {
    if (!can_add_u64(phdr.p_offset, phdr.p_filesz))
      return FileInsertionOutcome::MalformedInput;
    const uint64_t old_end = phdr.p_offset + phdr.p_filesz;
    if (phdr.p_offset >= file_offset) {
      if (!can_add_u64(phdr.p_offset, delta))
        return FileInsertionOutcome::MalformedInput;
      continue;
    }
    const bool inside_segment = file_offset < old_end;
    const bool at_segment_end = grow_load_at_segment_end && file_offset == old_end;
    if (phdr.p_type == PT_LOAD && (inside_segment || at_segment_end) &&
        (!can_add_u64(phdr.p_filesz, delta) || !can_add_u64(phdr.p_memsz, delta)))
      return FileInsertionOutcome::MalformedInput;
  }

  try {
    image.insert(image.begin() + static_cast<std::ptrdiff_t>(file_offset), bytes.begin(),
                 bytes.end());
  } catch (const std::bad_alloc &) {
    return FileInsertionOutcome::AllocationFailure;
  } catch (const std::length_error &) {
    return FileInsertionOutcome::AllocationFailure;
  }

  if (ehdr.e_shoff >= file_offset)
    ehdr.e_shoff += delta;
  if (ehdr.e_phoff != 0 && ehdr.e_phoff >= file_offset)
    ehdr.e_phoff += delta;

  // Sections at or after the insertion point move forward, except for the one
  // section that the caller is explicitly extending in place.
  for (size_t i = 0; i < shdrs.size(); ++i) {
    if (grown_section_index && i == *grown_section_index)
      continue;
    if (shdrs[i].sh_type == SHT_NULL)
      continue;
    if (shdrs[i].sh_offset >= file_offset)
      shdrs[i].sh_offset += delta;
  }

  for (Elf64_Phdr &phdr : phdrs) {
    const uint64_t old_end = phdr.p_offset + phdr.p_filesz;
    if (phdr.p_offset >= file_offset) {
      phdr.p_offset += delta;
      continue;
    }

    // If bytes are inserted inside a loaded segment, keep the segment covering
    // the shifted contents. When inserting exactly at the end, only the caller
    // that is adding executable cave bytes should grow the segment.
    const bool inside_segment = file_offset < old_end;
    const bool at_segment_end = grow_load_at_segment_end && file_offset == old_end;
    if (phdr.p_type == PT_LOAD && (inside_segment || at_segment_end)) {
      phdr.p_filesz += delta;
      phdr.p_memsz += delta;
    }
  }
  return FileInsertionOutcome::Success;
}

[[nodiscard]] std::optional<uint64_t> checked_lcm_u64(uint64_t lhs, uint64_t rhs) {
  const uint64_t gcd = std::gcd(lhs, rhs);
  if (gcd == 0)
    return std::nullopt;
  if (lhs / gcd > std::numeric_limits<uint64_t>::max() / rhs)
    return std::nullopt;
  return (lhs / gcd) * rhs;
}

[[nodiscard]] std::optional<uint64_t> checked_align_up(uint64_t value, uint64_t alignment) {
  if (alignment <= 1)
    return value;
  const uint64_t remainder = value % alignment;
  const uint64_t padding = remainder == 0 ? 0 : alignment - remainder;
  if (value > std::numeric_limits<uint64_t>::max() - padding)
    return std::nullopt;
  return value + padding;
}

[[nodiscard]] std::optional<uint64_t>
shifted_load_delta_alignment(std::span<const Elf64_Phdr> phdrs, uint64_t file_offset) {
  uint64_t alignment = 1;
  // Later LOAD segments remain loader-valid only when their file shift keeps
  // p_offset % p_align congruent with p_vaddr % p_align.
  for (const Elf64_Phdr &phdr : phdrs) {
    if (phdr.p_type != PT_LOAD || phdr.p_align <= 1)
      continue;
    if (phdr.p_offset >= file_offset) {
      const auto next = checked_lcm_u64(alignment, phdr.p_align);
      if (!next)
        return std::nullopt;
      alignment = *next;
    }
  }
  return alignment;
}

[[nodiscard]] std::optional<size_t> find_text_section(std::span<const Elf64_Shdr> shdrs,
                                                      uint64_t text_offset, uint64_t text_size) {
  for (size_t i = 0; i < shdrs.size(); ++i) {
    if (shdrs[i].sh_offset == text_offset && shdrs[i].sh_size == text_size)
      return i;
  }
  return std::nullopt;
}

[[nodiscard]] bool target_supports_wave32(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2 ||
         arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
         arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_GFX1250;
}

[[nodiscard]] bool target_uses_gfx10_plus_mode_bits(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2 ||
         arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
         arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_GFX1250;
}

[[nodiscard]] bool target_uses_wgp_mode(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2 ||
         arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
         arch == ROCJITSU_CODE_ARCH_RDNA4;
}

[[nodiscard]] bool target_uses_gfx90a_accum_offset(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA2 || arch == ROCJITSU_CODE_ARCH_CDNA3 ||
         arch == ROCJITSU_CODE_ARCH_CDNA4;
}

[[nodiscard]] bool target_clears_rsrc1_mode_bits(rj_code_arch_t arch) {
  // DX10_CLAMP and IEEE_MODE are deprecated on GFX12. Preserve them for GFX10
  // and GFX11 targets where they still affect floating-point behavior.
  return arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_GFX1250;
}

[[nodiscard]] uint32_t target_default_inst_pref_size(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
                 arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_GFX1250
             ? 2
             : 0;
}

[[nodiscard]] bool image_contains_range(size_t image_size, uint64_t file_offset, uint64_t size) {
  const uint64_t limit = static_cast<uint64_t>(image_size);
  return file_offset <= limit && size <= limit - file_offset;
}

void apply_kernel_descriptor_resource_translation(KD &desc, const KdTranslation &translation,
                                                  rj_code_arch_t target_arch) {
  AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  translation.target_vgpr_granulated);
  AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  translation.target_sgpr_granulated);

  if (target_uses_gfx90a_accum_offset(target_arch) && translation.target_accvgpr_base != 0) {
    // GFX90A-style descriptors encode the first AccVGPR as (field + 1) * 4.
    // KernelDescriptorTranslator may move this base upward when semantic
    // lowering needs ordinary VGPR scratch above the source AccVGPR window, so
    // the patcher must write the recomputed base alongside the VGPR allocation.
    const uint32_t encoded_accum_offset = (translation.target_accvgpr_base / 4) - 1;
    AMDHSA_BITS_SET(desc.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET,
                    encoded_accum_offset);
  }

  if (target_uses_gfx10_plus_mode_bits(target_arch)) {
    if (target_clears_rsrc1_mode_bits(target_arch)) {
      AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_ENABLE_DX10_CLAMP, 0);
      AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_ENABLE_IEEE_MODE, 0);
    }
    const uint32_t wgp_mode = target_uses_wgp_mode(target_arch) ? 1u : 0u;
    AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_WGP_MODE, wgp_mode);
    AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_MEM_ORDERED, 1);
    AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_FWD_PROGRESS, 1);
  }

  if (target_supports_wave32(target_arch)) {
    const uint32_t wave32 = translation.target_wave_size == 32 ? 1u : 0u;
    AMDHSA_BITS_SET(desc.kernel_code_properties, kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32,
                    wave32);
  } else {
    AMDHSA_BITS_SET(desc.kernel_code_properties, kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32,
                    0);
  }

  if (target_uses_gfx10_plus_mode_bits(target_arch)) {
    desc.compute_pgm_rsrc3 = 0;
    if (const uint32_t inst_pref = target_default_inst_pref_size(target_arch); inst_pref != 0) {
      AMDHSA_BITS_SET(desc.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX10_PLUS_INST_PREF_SIZE,
                      inst_pref);
    }
  } else if (target_uses_gfx90a_accum_offset(target_arch) && translation.target_accvgpr_base != 0) {
    // On GFX90A/GFX942/GFX950, AccVGPRs are placed by ACCUM_OFFSET rather than
    // by the ordinary VGPR count. KernelDescriptorTranslator decides whether the
    // base must move up to make room for semantic-lowering scratch; the patcher
    // only materializes that already-translated target base.
    assert(translation.target_accvgpr_base >= 4 &&
           "ACCUM_OFFSET base must encode at least 4 VGPRs");
    assert(translation.target_accvgpr_base % 4 == 0 && "ACCUM_OFFSET base must be 4-VGPR aligned");
    AMDHSA_BITS_SET(desc.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET,
                    (translation.target_accvgpr_base / 4 - 1));
  }

  desc.private_segment_fixed_size = translation.target_private_size;
  desc.group_segment_fixed_size = translation.target_lds_size;
  desc.kernarg_size = translation.target_kernarg_size;
  // `kernel_code_properties` is a 16-bit descriptor field. Keep this target ABI
  // bit update explicit so adding the DBT-only kernarg segment pointer cannot
  // be lost through integer-promotion surprises in the AMDHSA field macro.
  constexpr uint16_t kKernargSegmentPtrMask =
      static_cast<uint16_t>(kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR);
  if (translation.has_kernarg_segment_ptr) {
    desc.kernel_code_properties =
        static_cast<uint16_t>(desc.kernel_code_properties | kKernargSegmentPtrMask);
  } else {
    desc.kernel_code_properties =
        static_cast<uint16_t>(desc.kernel_code_properties & ~kKernargSegmentPtrMask);
  }
  // The AMDHSA descriptor must not carry a preprogrammed LDS_SIZE in RSRC2.
  // CP derives COMPUTE_PGM_RSRC2.LDS_SIZE from the dispatch packet's group
  // segment size, which already includes descriptor fixed LDS plus dispatch-time
  // dynamic LDS. Leaving a guest value here is especially bad for virtual-LDS
  // sidecars: their packet LDS is zero, but stale descriptor bits can still be
  // ORed into hardware command streams on some runtime paths.
  AMDHSA_BITS_SET(desc.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_GRANULATED_LDS_SIZE, 0);
  AMDHSA_BITS_SET(desc.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT,
                  translation.target_user_sgpr_count);
  // Fixed private size can be zero for a kernel that requests its call stack
  // dynamically through the AQL packet. Preserve an existing scratch-enable
  // requirement and also enable it whenever DBT introduces fixed spill space.
  // The skipped-kernel path below intentionally clears the failed guest ABI.
  const uint32_t enable_private_segment =
      AMDHSA_BITS_GET(desc.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT) != 0 ||
              translation.target_private_size != 0
          ? 1u
          : 0u;
  AMDHSA_BITS_SET(desc.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT,
                  enable_private_segment);
  if (translation.skipped) {
    // Skipped kernels are target-ISA trap stubs. Do not preserve the
    // failed guest kernel's descriptor ABI inputs: ROCR validates the resource
    // request before the stub can run, and stale kernarg/LDS/private settings
    // can turn an intentional trap into an earlier invalid-allocation failure.
    //
    // Keep the same non-resource defaults that amdclang emits for an empty
    // gfx942 kernel: IEEE/DX10 mode with both FP denorm modes enabled, plus
    // SGPR_WORKGROUP_ID_X. A completely zeroed RSRC1/RSRC2 looks smaller, but
    // it is not a reliable dispatchable descriptor on the runtime path used by
    // skipped-kernel stubs.
    desc.kernarg_size = 0;
    desc.compute_pgm_rsrc1 = 0;
    AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_32, 3);
    AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_16_64, 3);
    AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_ENABLE_DX10_CLAMP, 1);
    AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_ENABLE_IEEE_MODE, 1);
    desc.compute_pgm_rsrc3 = 0;
    desc.compute_pgm_rsrc2 = 0;
    AMDHSA_BITS_SET(desc.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X, 1);
    desc.kernel_code_properties = 0;
    desc.kernarg_preload = 0;
  }
}

[[nodiscard]] bool set_kernel_entry_from_vaddr(KD &desc, uint64_t descriptor_vaddr,
                                               uint64_t text_vaddr,
                                               uint64_t new_entry_text_offset) {
  if (descriptor_vaddr > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    return false;
  if (text_vaddr > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      new_entry_text_offset >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) - text_vaddr)
    return false;

  const int64_t entry_vaddr = static_cast<int64_t>(text_vaddr + new_entry_text_offset);
  const int64_t descriptor = static_cast<int64_t>(descriptor_vaddr);
  desc.kernel_code_entry_byte_offset = entry_vaddr - descriptor;
  return true;
}

[[nodiscard]] bool shift_symbols_in_moved_sections(std::vector<uint8_t> &image,
                                                   const Elf64_Ehdr &ehdr,
                                                   std::span<const Elf64_Shdr> shdrs,
                                                   const std::vector<bool> &shift_section_vaddr,
                                                   uint64_t delta) {
  assert(shdrs.size() == shift_section_vaddr.size() && "section shift map size mismatch");
  if (delta == 0 || ehdr.e_type == ET_REL)
    return true;

  // ET_DYN symbol values are virtual addresses. Symbols bound to moved
  // allocated sections must track those sections' new virtual addresses.
  for (const Elf64_Shdr &symtab : shdrs) {
    if (symtab.sh_type != SHT_SYMTAB && symtab.sh_type != SHT_DYNSYM)
      continue;
    if (symtab.sh_entsize != sizeof(Elf64_Sym))
      continue;
    if (!image_contains_range(image.size(), symtab.sh_offset, symtab.sh_size))
      continue;

    const size_t count = symtab.sh_size / sizeof(Elf64_Sym);
    for (size_t i = 0; i < count; ++i) {
      const uint64_t symbol_offset = symtab.sh_offset + i * sizeof(Elf64_Sym);
      Elf64_Sym symbol{};
      std::memcpy(&symbol, image.data() + symbol_offset, sizeof(symbol));

      if (symbol.st_shndx == SHN_UNDEF || symbol.st_shndx == SHN_ABS ||
          symbol.st_shndx >= shdrs.size())
        continue;
      if (!shift_section_vaddr[symbol.st_shndx])
        continue;

      if (!can_add_u64(symbol.st_value, delta))
        return false;
      symbol.st_value += delta;
      std::memcpy(image.data() + symbol_offset, &symbol, sizeof(symbol));
    }
  }
  return true;
}

void grow_text_function_symbols(std::vector<uint8_t> &image, const Elf64_Ehdr &ehdr,
                                std::span<const Elf64_Shdr> shdrs, size_t text_index,
                                uint64_t old_text_size, uint64_t new_text_size) {
  if (new_text_size <= old_text_size)
    return;

  const Elf64_Shdr &text = shdrs[text_index];
  for (const Elf64_Shdr &symtab : shdrs) {
    if (symtab.sh_type != SHT_SYMTAB && symtab.sh_type != SHT_DYNSYM)
      continue;
    if (symtab.sh_entsize != sizeof(Elf64_Sym))
      continue;
    if (!image_contains_range(image.size(), symtab.sh_offset, symtab.sh_size))
      continue;

    const size_t count = symtab.sh_size / sizeof(Elf64_Sym);
    for (size_t i = 0; i < count; ++i) {
      const uint64_t symbol_offset = symtab.sh_offset + i * sizeof(Elf64_Sym);
      Elf64_Sym symbol{};
      std::memcpy(&symbol, image.data() + symbol_offset, sizeof(symbol));

      if (symbol.st_shndx != text_index || elf_symbol_type(symbol.st_info) != kElfSymbolTypeFunc)
        continue;

      // ET_DYN symbols use virtual addresses; ET_REL symbols use section
      // offsets.  Normalize both forms so the range test below is independent
      // of the input code-object kind.
      uint64_t symbol_text_offset = symbol.st_value;
      if (ehdr.e_type != ET_REL) {
        if (symbol.st_value < text.sh_addr)
          continue;
        symbol_text_offset = symbol.st_value - text.sh_addr;
      }
      // Appended executable cave code is contiguous only with a function that
      // already reaches the old end of .text. Growing every function through
      // the cave makes their ranges overlap inter-function alignment padding;
      // a subsequent instrumentation pass then tries to decode that padding as
      // instructions. The CFG builder sees all function ranges together, so a
      // branch from an earlier function can still reach the cave block covered
      // by the tail function.
      if (symbol.st_size == 0)
        continue;

      if (symbol_text_offset > old_text_size ||
          symbol.st_size != old_text_size - symbol_text_offset)
        continue;

      const uint64_t grown_size = new_text_size - symbol_text_offset;
      if (symbol.st_size >= grown_size)
        continue;
      symbol.st_size = grown_size;
      std::memcpy(image.data() + symbol_offset, &symbol, sizeof(symbol));
    }
  }
}

[[nodiscard]] bool relocation_place_was_moved(uint64_t place, std::span<const Elf64_Shdr> shdrs,
                                              const std::vector<bool> &shift_section_vaddr,
                                              uint64_t delta) {
  // shdrs already contains the new addresses, so recover the previous range
  // before deciding whether an ET_DYN relocation place moved.
  for (size_t i = 0; i < shdrs.size(); ++i) {
    if (!shift_section_vaddr[i])
      continue;

    // Shifted sections were moved forward by exactly delta, so their new
    // address must be large enough to recover the pre-insert range.
    assert(shdrs[i].sh_addr >= delta && "moved section address underflow");
    const uint64_t old_addr = shdrs[i].sh_addr - delta;
    if (place >= old_addr && place - old_addr < shdrs[i].sh_size)
      return true;
  }
  return false;
}

[[nodiscard]] bool shift_relocation_offsets_in_moved_sections(
    std::vector<uint8_t> &image, const Elf64_Ehdr &ehdr, std::span<const Elf64_Shdr> shdrs,
    const std::vector<bool> &shift_section_vaddr, uint64_t delta) {
  assert(shdrs.size() == shift_section_vaddr.size() && "section shift map size mismatch");
  if (delta == 0 || ehdr.e_type != ET_DYN)
    return true;

  // AMDHSA uses Elf64_Rela records, and ET_DYN relocation r_offset is the
  // relocated storage address. Any relocation whose place is inside a section we
  // moved must track that section's new virtual address.
  for (const Elf64_Shdr &relocs : shdrs) {
    if (relocs.sh_type != SHT_RELA && relocs.sh_type != SHT_REL)
      continue;
    if (!image_contains_range(image.size(), relocs.sh_offset, relocs.sh_size))
      continue;

    if (relocs.sh_type == SHT_RELA) {
      if (relocs.sh_entsize != sizeof(Elf64_Rela))
        continue;
      const size_t count = relocs.sh_size / sizeof(Elf64_Rela);
      for (size_t i = 0; i < count; ++i) {
        const uint64_t offset = relocs.sh_offset + i * sizeof(Elf64_Rela);
        Elf64_Rela rela{};
        std::memcpy(&rela, image.data() + offset, sizeof(rela));
        if (!relocation_place_was_moved(rela.r_offset, shdrs, shift_section_vaddr, delta))
          continue;
        if (!can_add_u64(rela.r_offset, delta))
          return false;
        rela.r_offset += delta;
        std::memcpy(image.data() + offset, &rela, sizeof(rela));
      }
      continue;
    }

    if (relocs.sh_entsize != sizeof(Elf64_Rel))
      continue;
    const size_t count = relocs.sh_size / sizeof(Elf64_Rel);
    for (size_t i = 0; i < count; ++i) {
      const uint64_t offset = relocs.sh_offset + i * sizeof(Elf64_Rel);
      Elf64_Rel rel{};
      std::memcpy(&rel, image.data() + offset, sizeof(rel));
      if (!relocation_place_was_moved(rel.r_offset, shdrs, shift_section_vaddr, delta))
        continue;
      if (!can_add_u64(rel.r_offset, delta))
        return false;
      rel.r_offset += delta;
      std::memcpy(image.data() + offset, &rel, sizeof(rel));
    }
  }
  return true;
}

[[nodiscard]] bool kernel_descriptor_symbol(const Elf64_Sym &symbol, const char *strtab,
                                            size_t strtab_size) {
  if (symbol.st_size != sizeof(KD))
    return false;

  // AMDHSA kernel descriptors are global object symbols. This keeps other
  // sizeof(KD) data objects from being treated as descriptors by accident.
  if (elf_symbol_type(symbol.st_info) != kElfSymbolTypeObject ||
      elf_symbol_bind(symbol.st_info) != kElfSymbolBindGlobal)
    return false;

  // AMDHSA descriptors are named "<kernel>.kd". An unnamed 64-byte global
  // object is ambiguous, so require the ABI suffix instead of treating stripped
  // or minimized symbol records as descriptors.
  if (strtab == nullptr || strtab_size == 0 || symbol.st_name == 0)
    return false;
  if (symbol.st_name >= strtab_size)
    return false;

  const char *name = strtab + symbol.st_name;
  const size_t len = strnlen(name, strtab_size - symbol.st_name);
  return len > 3 && std::strcmp(name + len - 3, ".kd") == 0;
}

[[nodiscard]] bool adjust_kernel_descriptor_entry_offsets_in_moved_sections(
    std::vector<uint8_t> &image, std::span<const Elf64_Shdr> shdrs,
    const std::vector<bool> &shift_section_vaddr, uint64_t delta) {
  assert(shdrs.size() == shift_section_vaddr.size() && "section shift map size mismatch");
  if (delta == 0)
    return true;

  // KERNEL_CODE_ENTRY_BYTE_OFFSET is relative to the descriptor address, not to
  // .text. When a .kd object lives in a shifted allocated section, preserve the
  // same absolute entry PC by subtracting the descriptor section's VA delta.
  std::unordered_set<uint64_t> adjusted_file_offsets;
  for (const Elf64_Shdr &symtab : shdrs) {
    if (symtab.sh_type != SHT_SYMTAB && symtab.sh_type != SHT_DYNSYM)
      continue;
    if (symtab.sh_entsize != sizeof(Elf64_Sym))
      continue;
    if (!image_contains_range(image.size(), symtab.sh_offset, symtab.sh_size))
      continue;

    const char *strtab = nullptr;
    size_t strtab_size = 0;
    if (symtab.sh_link < shdrs.size()) {
      const Elf64_Shdr &strtab_shdr = shdrs[symtab.sh_link];
      if (image_contains_range(image.size(), strtab_shdr.sh_offset, strtab_shdr.sh_size)) {
        strtab = reinterpret_cast<const char *>(image.data() + strtab_shdr.sh_offset);
        strtab_size = strtab_shdr.sh_size;
      }
    }

    const size_t count = symtab.sh_size / sizeof(Elf64_Sym);
    for (size_t i = 0; i < count; ++i) {
      const uint64_t symbol_offset = symtab.sh_offset + i * sizeof(Elf64_Sym);
      Elf64_Sym symbol{};
      std::memcpy(&symbol, image.data() + symbol_offset, sizeof(symbol));

      if (!kernel_descriptor_symbol(symbol, strtab, strtab_size))
        continue;
      if (symbol.st_shndx == SHN_UNDEF || symbol.st_shndx == SHN_ABS ||
          symbol.st_shndx >= shdrs.size())
        continue;
      if (!shift_section_vaddr[symbol.st_shndx])
        continue;

      const Elf64_Shdr &section = shdrs[symbol.st_shndx];
      if (symbol.st_value < section.sh_addr)
        continue;

      const uint64_t section_offset = symbol.st_value - section.sh_addr;
      if (!can_add_u64(section.sh_offset, section_offset))
        return false;
      const uint64_t file_offset = section.sh_offset + section_offset;
      if (!image_contains_range(image.size(), file_offset, sizeof(KD)))
        continue;
      if (!adjusted_file_offsets.insert(file_offset).second)
        continue;

      KD desc{};
      std::memcpy(&desc, image.data() + file_offset, sizeof(desc));
      if (delta > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
          desc.kernel_code_entry_byte_offset <
              std::numeric_limits<int64_t>::min() + static_cast<int64_t>(delta))
        return false;
      desc.kernel_code_entry_byte_offset -= static_cast<int64_t>(delta);
      std::memcpy(image.data() + file_offset, &desc, sizeof(desc));
    }
  }
  return true;
}

} // namespace

CodeObjectPatcher::CodeObjectPatcher(const AmdGpuCodeObject &obj)
    : image_(obj.image_data(), obj.image_data() + obj.image_size()), text_offset_(0), text_size_(0),
      text_vaddr_(0), text_tail_size_(0) {
  auto &text_secs = obj.text_sections();
  if (!text_secs.empty()) {
    text_offset_ = text_secs[0]->sectionOffset();
    text_size_ = text_secs[0]->size();
    text_vaddr_ = text_secs[0]->vaddr();
  }
}

std::span<uint8_t> CodeObjectPatcher::text_bytes() {
  return {image_.data() + text_offset_, text_size_};
}

std::span<const uint8_t> CodeObjectPatcher::text_bytes() const {
  return {image_.data() + text_offset_, text_size_};
}

bool CodeObjectPatcher::has_relocations_within_text() const {
  if (text_size_ == 0)
    return false;
  if (image_.size() < sizeof(Elf64_Ehdr))
    return false;

  auto header = *reinterpret_cast<const Elf64_Ehdr *>(image_.data());
  const auto maybe_shdrs = read_section_headers(image_, header);
  if (!maybe_shdrs)
    return false;
  const auto &shdrs = *maybe_shdrs;
  const auto text_index = find_text_section(shdrs, text_offset_, text_size_);
  if (!text_index)
    return false;

  // A relocation's place (r_offset) is interpreted differently by ELF type:
  //   - ET_DYN/ET_EXEC: r_offset is a virtual address. An in-.text place is one
  //     inside [text.sh_addr, text.sh_addr + size). The reloc section applies
  //     across the whole image, so every record must be range-checked.
  //   - ET_REL: r_offset is relative to the section the reloc section applies to,
  //     named by the reloc section's sh_info. Only records whose sh_info is the
  //     text section can land in .text, and there any r_offset < text.sh_size is
  //     in-text. Comparing an ET_REL r_offset against a file offset (text.sh_offset)
  //     is wrong and would miss e.g. an .rela.text record with r_offset == 0.
  const Elf64_Shdr &text = shdrs[*text_index];
  const bool is_rel_object = header.e_type == ET_REL;

  for (const Elf64_Shdr &relocs : shdrs) {
    if (relocs.sh_type != SHT_RELA && relocs.sh_type != SHT_REL)
      continue;
    if (!image_contains_range(image_.size(), relocs.sh_offset, relocs.sh_size))
      continue;
    if (is_rel_object && relocs.sh_info != *text_index)
      continue; // ET_REL: only relocations that apply to .text matter.
    const bool is_rela = relocs.sh_type == SHT_RELA;
    const size_t entsize = is_rela ? sizeof(Elf64_Rela) : sizeof(Elf64_Rel);
    if (relocs.sh_entsize != entsize)
      continue;
    const size_t count = relocs.sh_size / entsize;
    for (size_t i = 0; i < count; ++i) {
      const uint64_t offset = relocs.sh_offset + i * entsize;
      uint64_t r_offset = 0;
      if (is_rela) {
        Elf64_Rela rela{};
        std::memcpy(&rela, image_.data() + offset, sizeof(rela));
        r_offset = rela.r_offset;
      } else {
        Elf64_Rel rel{};
        std::memcpy(&rel, image_.data() + offset, sizeof(rel));
        r_offset = rel.r_offset;
      }
      const bool in_text = is_rel_object
                               ? r_offset < text_size_
                               : (r_offset >= text.sh_addr && r_offset < text.sh_addr + text_size_);
      if (in_text)
        return true;
    }
  }
  return false;
}

bool CodeObjectPatcher::has_relocation_to_text_symbol() const {
  if (text_size_ == 0)
    return false;
  if (image_.size() < sizeof(Elf64_Ehdr))
    return false;

  auto header = *reinterpret_cast<const Elf64_Ehdr *>(image_.data());
  const auto maybe_shdrs = read_section_headers(image_, header);
  if (!maybe_shdrs)
    return false;
  const auto &shdrs = *maybe_shdrs;
  const auto text_index = find_text_section(shdrs, text_offset_, text_size_);
  if (!text_index)
    return false;

  // The virtual-address interval of the source .text, used to catch symbol-less
  // relocations whose addend names an in-.text address (see the RELATIVE64 check
  // below). find_text_section only guarantees offset/size, so read sh_addr here.
  const Elf64_Shdr &text = shdrs[*text_index];
  const uint64_t text_addr_lo = text.sh_addr;
  const uint64_t text_addr_hi = text.sh_addr + text_size_;

  // True if the referenced symbol is *defined in .text*, regardless of type. DBT
  // does not remap text-defined st_value, so any such reference resolves to a
  // stale pre-move PC. This deliberately matches all types: STT_FUNC helpers,
  // STT_NOTYPE labels, and an STT_SECTION symbol for .text (whose reloc addend
  // names an in-.text offset) all alias moved code. Keying on st_shndx (not the
  // symbol type) is what makes the section-symbol and untyped-label cases fail
  // closed. STN_UNDEF (index 0) has st_shndx == SHN_UNDEF, so it never matches.
  auto symbol_is_defined_in_text = [&](const Elf64_Shdr &symtab, uint32_t sym_index) {
    if (symtab.sh_entsize != sizeof(Elf64_Sym))
      return false;
    if (!image_contains_range(image_.size(), symtab.sh_offset, symtab.sh_size))
      return false;
    if (static_cast<uint64_t>(sym_index) * sizeof(Elf64_Sym) + sizeof(Elf64_Sym) > symtab.sh_size)
      return false;
    Elf64_Sym symbol{};
    std::memcpy(&symbol, image_.data() + symtab.sh_offset + sym_index * sizeof(Elf64_Sym),
                sizeof(symbol));
    return symbol.st_shndx == *text_index;
  };

  for (const Elf64_Shdr &relocs : shdrs) {
    if (relocs.sh_type != SHT_RELA && relocs.sh_type != SHT_REL)
      continue;
    if (!image_contains_range(image_.size(), relocs.sh_offset, relocs.sh_size))
      continue;
    // sh_link names the symbol table this relocation section indexes into.
    if (relocs.sh_link >= shdrs.size())
      continue;
    const Elf64_Shdr &symtab = shdrs[relocs.sh_link];
    const bool is_rela = relocs.sh_type == SHT_RELA;
    const size_t entsize = is_rela ? sizeof(Elf64_Rela) : sizeof(Elf64_Rel);
    if (relocs.sh_entsize != entsize)
      continue;
    const size_t count = relocs.sh_size / entsize;
    for (size_t i = 0; i < count; ++i) {
      const uint64_t offset = relocs.sh_offset + i * entsize;
      uint64_t r_info = 0;
      int64_t r_addend = 0;
      if (is_rela) {
        Elf64_Rela rela{};
        std::memcpy(&rela, image_.data() + offset, sizeof(rela));
        r_info = rela.r_info;
        r_addend = rela.r_addend;
      } else {
        Elf64_Rel rel{};
        std::memcpy(&rel, image_.data() + offset, sizeof(rel));
        r_info = rel.r_info;
      }
      const uint32_t sym_index = elf_reloc_sym(r_info);
      if (sym_index != 0) {
        if (symbol_is_defined_in_text(symtab, sym_index))
          return true;
        continue;
      }
      // Symbol index 0: no owning symbol, so the st_shndx check above cannot see
      // it. R_AMDGPU_RELATIVE64 forms its stored value from the load bias plus
      // r_addend, so an addend inside the source .text virtual-address interval
      // aliases code DBT moves without remapping the addend. Fail closed on that
      // form; other symbol-zero types have no addend->text derivation to check.
      if (is_rela && elf_reloc_type(r_info) == R_AMDGPU_RELATIVE64) {
        const uint64_t target = static_cast<uint64_t>(r_addend);
        if (target >= text_addr_lo && target < text_addr_hi)
          return true;
      }
    }
  }
  return false;
}

TextReplacementResult CodeObjectPatcher::replace_text(std::span<const uint8_t> new_text,
                                                      size_t max_file_growth) noexcept {
  try {
    return replace_text_impl(new_text, max_file_growth);
  } catch (const std::bad_alloc &) {
    return TextReplacementResult::allocation_failure();
  } catch (const std::length_error &) {
    return TextReplacementResult::allocation_failure();
  }
}

TextReplacementResult CodeObjectPatcher::replace_text_impl(std::span<const uint8_t> new_text,
                                                           size_t max_file_growth) {
  std::optional<size_t> required_file_growth;
  const auto malformed = [&] {
    return TextReplacementResult::malformed_input(required_file_growth);
  };

  // Keep fail-closed behavior for callers that assume word-aligned executable
  // sections; accepting a non-word-aligned replacement can break downstream
  // PC-relative patching and branch-distance checks.
  if ((new_text.size() % sizeof(uint32_t)) != 0)
    return malformed();

  if (text_size_ == 0)
    return malformed();
  if (new_text.size() < text_size_)
    return malformed();
  if (!image_contains_range(image_.size(), text_offset_, text_size_))
    return malformed();

  if (image_.size() < sizeof(Elf64_Ehdr))
    return malformed();
  // Parse immutable metadata before allocating the transactional image.
  // Malformed metadata discovered late in relocation or descriptor adjustment
  // must not leave a partially shifted code object behind.
  Elf64_Ehdr header{};
  std::memcpy(&header, image_.data(), sizeof(header));
  auto maybe_shdrs = read_section_headers(image_, header);
  auto maybe_phdrs = read_program_headers(image_, header);
  if (!maybe_shdrs || !maybe_phdrs)
    return malformed();
  auto shdrs = std::move(*maybe_shdrs);
  auto phdrs = std::move(*maybe_phdrs);
  std::vector<uint8_t> image;

  const auto text_index = find_text_section(shdrs, text_offset_, text_size_);
  if (!text_index) {
    assert(false && "text section header not found");
    return malformed();
  }

  const auto text_header = shdrs[*text_index];
  const uint64_t old_text_end_file = text_offset_ + text_size_;
  if (!can_add_u64(text_header.sh_addr, text_size_))
    return malformed();
  const uint64_t old_text_end_vaddr = text_header.sh_addr + text_size_;
  const uint64_t growth = new_text.size() - text_size_;

  if (growth != 0) {
    std::vector<bool> shift_section_vaddr(shdrs.size(), false);
    uint64_t shifted_section_alignment = 1;
    for (size_t i = 0; i < shdrs.size(); ++i) {
      if (i == *text_index || shdrs[i].sh_type == SHT_NULL)
        continue;
      const bool shifts_in_file = shdrs[i].sh_offset >= old_text_end_file;
      const bool shifts_in_memory =
          (shdrs[i].sh_flags & SHF_ALLOC) != 0 && shdrs[i].sh_addr >= old_text_end_vaddr;
      shift_section_vaddr[i] = shifts_in_memory;
      if (shifts_in_memory && !can_add_u64(shdrs[i].sh_addr, growth))
        return malformed();
      if (shifts_in_file || shifts_in_memory) {
        const auto next = checked_lcm_u64(shifted_section_alignment,
                                          std::max<uint64_t>(shdrs[i].sh_addralign, 1));
        if (!next)
          return malformed();
        shifted_section_alignment = *next;
      }
    }
    // Growing .text can shift later LOAD segments. Pad the inserted file range
    // so every shifted LOAD keeps p_offset % p_align congruent with p_vaddr %
    // p_align and every shifted allocated section keeps its address-alignment
    // residue. The padding is executable segment filler, not part of .text.
    const auto load_alignment = shifted_load_delta_alignment(phdrs, old_text_end_file);
    if (!load_alignment)
      return malformed();
    const auto file_delta_alignment = checked_lcm_u64(*load_alignment, shifted_section_alignment);
    if (!file_delta_alignment)
      return malformed();
    const auto padded_file_delta = checked_align_up(growth, *file_delta_alignment);
    if (!padded_file_delta || *padded_file_delta < growth)
      return malformed();
    required_file_growth = static_cast<size_t>(*padded_file_delta);
    if (*padded_file_delta > max_file_growth)
      return TextReplacementResult::file_growth_limit_exceeded(*required_file_growth);
    if (*padded_file_delta % sizeof(uint32_t) != 0)
      return malformed();
    if (*padded_file_delta > image_.max_size() - image_.size())
      return TextReplacementResult::allocation_failure(required_file_growth);

    try {
      // Allocate the exact final image capacity before copying the pristine
      // bytes. This avoids both a temporary full-size transactional buffer and
      // vector's geometric reallocation during the insertion below.
      image.reserve(image_.size() + *padded_file_delta);
      image.assign(image_.begin(), image_.end());
      std::vector<uint8_t> inserted;
      inserted.resize(*padded_file_delta, 0);
      std::vector<bool> shift_segment_vaddr(phdrs.size(), false);
      for (size_t i = 0; i < phdrs.size(); ++i) {
        if (phdrs[i].p_vaddr >= old_text_end_vaddr && phdrs[i].p_offset >= old_text_end_file) {
          if (!can_add_u64(phdrs[i].p_vaddr, *padded_file_delta) ||
              !can_add_u64(phdrs[i].p_paddr, *padded_file_delta))
            return malformed();
          shift_segment_vaddr[i] = true;
        }
      }

      const FileInsertionOutcome insertion = insert_file_bytes(
          image, header, shdrs, phdrs, old_text_end_file, inserted, *text_index, true);
      if (insertion == FileInsertionOutcome::AllocationFailure)
        return TextReplacementResult::allocation_failure(required_file_growth);
      if (insertion != FileInsertionOutcome::Success)
        return malformed();

      for (size_t i = 0; i < shdrs.size(); ++i) {
        if (!shift_section_vaddr[i])
          continue;
        [[maybe_unused]] const uint64_t old_addr = shdrs[i].sh_addr;
        if (!can_add_u64(shdrs[i].sh_addr, *padded_file_delta))
          return malformed();
        shdrs[i].sh_addr += *padded_file_delta;
        assert((shdrs[i].sh_addralign <= 1 ||
                shdrs[i].sh_addr % shdrs[i].sh_addralign == old_addr % shdrs[i].sh_addralign) &&
               "shifted allocated section lost its address alignment residue");
      }

      if (!shift_symbols_in_moved_sections(image, header, shdrs, shift_section_vaddr,
                                           *padded_file_delta) ||
          !shift_relocation_offsets_in_moved_sections(image, header, shdrs, shift_section_vaddr,
                                                      *padded_file_delta) ||
          !adjust_kernel_descriptor_entry_offsets_in_moved_sections(
              image, shdrs, shift_section_vaddr, *padded_file_delta))
        return malformed();

      for (size_t i = 0; i < phdrs.size(); ++i) {
        if (!shift_segment_vaddr[i])
          continue;
        assert((phdrs[i].p_align <= 1 || *padded_file_delta % phdrs[i].p_align == 0) &&
               "text padding does not preserve shifted LOAD alignment");
        phdrs[i].p_vaddr += *padded_file_delta;
        phdrs[i].p_paddr += *padded_file_delta;
        assert((phdrs[i].p_align <= 1 ||
                phdrs[i].p_offset % phdrs[i].p_align == phdrs[i].p_vaddr % phdrs[i].p_align) &&
               "shifted LOAD lost file/virtual address congruence");
      }
    } catch (const std::bad_alloc &) {
      return TextReplacementResult::allocation_failure(required_file_growth);
    } catch (const std::length_error &) {
      return TextReplacementResult::allocation_failure(required_file_growth);
    }
  } else {
    required_file_growth = 0;
    image = image_;
  }

  std::memcpy(image.data() + text_offset_, new_text.data(), new_text.size());
  shdrs[*text_index].sh_size = new_text.size();
  grow_text_function_symbols(image, header, shdrs, *text_index, text_size_, new_text.size());

  for (const Elf64_Phdr &phdr : phdrs) {
    if (phdr.p_type != PT_LOAD || phdr.p_align <= 1)
      continue;
    assert(phdr.p_offset % phdr.p_align == phdr.p_vaddr % phdr.p_align &&
           "patched LOAD lost file/virtual address congruence");
  }
  write_elf_tables(image, header, shdrs, phdrs);
  image_ = std::move(image);
  text_size_ = new_text.size();
  return TextReplacementResult::success(required_file_growth.value_or(0));
}

void CodeObjectPatcher::update_elf_flags(uint32_t new_mach) {
  auto *ehdr = reinterpret_cast<Elf64_Ehdr *>(image_.data());
  // Preserve upper bits (XNACK, SRAMECC feature flags); only replace EF_AMDGPU_MACH in low byte.
  ehdr->e_flags = (ehdr->e_flags & ~0xFFu) | (new_mach & 0xFFu);
}

bool CodeObjectPatcher::patch_kernel_descriptor(uint64_t file_offset,
                                                std::span<const uint8_t> descriptor) {
  if (!image_contains_range(image_.size(), file_offset, descriptor.size()))
    return false;

  std::memcpy(image_.data() + file_offset, descriptor.data(), descriptor.size());
  return true;
}

bool CodeObjectPatcher::apply_kernel_descriptor_translation(const KdTranslation &translation,
                                                            rj_code_arch_t target_arch) {
  if (!image_contains_range(image_.size(), translation.descriptor_file_offset, sizeof(KD)))
    return false;

  KD desc;
  std::memcpy(&desc, image_.data() + translation.descriptor_file_offset, sizeof(desc));
  apply_kernel_descriptor_resource_translation(desc, translation, target_arch);

  std::memcpy(image_.data() + translation.descriptor_file_offset, &desc, sizeof(desc));
  if (!redirect_kernel_entry(translation.descriptor_file_offset, translation.entry_text_offset,
                             translation.target_entry_text_offset))
    return false;
  return true;
}

std::optional<std::vector<AppendedSidecarDescriptor>>
CodeObjectPatcher::append_sidecar_descriptor_translations(
    std::span<const KdTranslation> translations, rj_code_arch_t target_arch, uint64_t alignment) {
  if (translations.empty())
    return std::vector<AppendedSidecarDescriptor>{};
  if (alignment == 0 || (alignment & (alignment - 1)) != 0)
    return std::nullopt;
  if (text_size_ == 0 || text_vaddr_ == 0)
    return std::nullopt;
  if (image_.size() < sizeof(Elf64_Ehdr))
    return std::nullopt;

  auto header = *reinterpret_cast<const Elf64_Ehdr *>(image_.data());
  auto maybe_shdrs = read_section_headers(image_, header);
  auto maybe_phdrs = read_program_headers(image_, header);
  if (!maybe_shdrs || !maybe_phdrs)
    return std::nullopt;
  auto shdrs = std::move(*maybe_shdrs);
  auto phdrs = std::move(*maybe_phdrs);

  const auto text_index = find_text_section(shdrs, text_offset_, text_size_);
  if (!text_index)
    return std::nullopt;

  const uint64_t insertion_file_offset = text_offset_ + text_size_ + text_tail_size_;
  const uint64_t insertion_vaddr = text_vaddr_ + text_size_ + text_tail_size_;
  if (insertion_file_offset > image_.size())
    return std::nullopt;

  const bool insertion_is_loaded = std::ranges::any_of(phdrs, [&](const Elf64_Phdr &phdr) {
    if (phdr.p_type != PT_LOAD)
      return false;
    const uint64_t segment_end = phdr.p_offset + phdr.p_filesz;
    return phdr.p_offset <= insertion_file_offset && insertion_file_offset <= segment_end;
  });
  if (!insertion_is_loaded)
    return std::nullopt;

  std::vector<uint8_t> inserted;
  std::vector<AppendedSidecarDescriptor> appended;
  appended.reserve(translations.size());
  uint64_t cursor_vaddr = insertion_vaddr;
  for (const KdTranslation &translation : translations) {
    const auto maybe_descriptor_vaddr = checked_align_up(cursor_vaddr, alignment);
    if (!maybe_descriptor_vaddr)
      return std::nullopt;
    const uint64_t descriptor_vaddr = *maybe_descriptor_vaddr;
    const uint64_t padding = descriptor_vaddr - cursor_vaddr;
    if (padding > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) - inserted.size())
      return std::nullopt;
    inserted.resize(inserted.size() + static_cast<size_t>(padding), 0);

    // Copy the sidecar template from the source-descriptor snapshot, NOT from
    // translation.descriptor_file_offset: this runs after .text growth, which
    // shifts a descriptor section that follows .text, leaving that offset
    // pointing at relocated instruction bytes.
    static_assert(sizeof(KD) == 64, "sidecar descriptor snapshot size mismatch");
    KD desc{};
    std::memcpy(&desc, translation.source_descriptor_bytes.data(), sizeof(desc));
    apply_kernel_descriptor_resource_translation(desc, translation, target_arch);
    if (!set_kernel_entry_from_vaddr(desc, descriptor_vaddr, text_vaddr_,
                                     translation.target_entry_text_offset))
      return std::nullopt;

    const uint64_t descriptor_file_offset = insertion_file_offset + inserted.size();
    const auto *descriptor_bytes = reinterpret_cast<const uint8_t *>(&desc);
    inserted.insert(inserted.end(), descriptor_bytes, descriptor_bytes + sizeof(desc));
    appended.push_back({.file_offset = descriptor_file_offset, .vaddr = descriptor_vaddr});
    cursor_vaddr = descriptor_vaddr + sizeof(desc);
  }

  // Later LOAD segments remain valid only if the file shift preserves their
  // p_offset/p_vaddr congruence. Any padding after the descriptor payload is
  // intentionally unsectioned loaded data; runtime metadata points only at the
  // descriptor starts recorded above.
  const auto file_delta_alignment = shifted_load_delta_alignment(phdrs, insertion_file_offset);
  if (!file_delta_alignment)
    return std::nullopt;
  const auto maybe_padded_size = checked_align_up(inserted.size(), *file_delta_alignment);
  if (!maybe_padded_size ||
      *maybe_padded_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
    return std::nullopt;
  const uint64_t padded_size = *maybe_padded_size;
  inserted.resize(static_cast<size_t>(padded_size), 0);

  std::vector<bool> shift_section_vaddr(shdrs.size(), false);
  for (size_t i = 0; i < shdrs.size(); ++i) {
    if (i == *text_index || shdrs[i].sh_type == SHT_NULL)
      continue;
    if ((shdrs[i].sh_flags & SHF_ALLOC) != 0 && shdrs[i].sh_addr >= insertion_vaddr)
      shift_section_vaddr[i] = true;
  }

  std::vector<bool> shift_segment_vaddr(phdrs.size(), false);
  for (size_t i = 0; i < phdrs.size(); ++i) {
    if (phdrs[i].p_vaddr >= insertion_vaddr && phdrs[i].p_offset >= insertion_file_offset)
      shift_segment_vaddr[i] = true;
  }

  if (insert_file_bytes(image_, header, shdrs, phdrs, insertion_file_offset, inserted, std::nullopt,
                        true) != FileInsertionOutcome::Success)
    return std::nullopt;

  const uint64_t delta = inserted.size();
  for (size_t i = 0; i < shdrs.size(); ++i) {
    if (!shift_section_vaddr[i])
      continue;
    [[maybe_unused]] const uint64_t old_addr = shdrs[i].sh_addr;
    shdrs[i].sh_addr += delta;
    assert((shdrs[i].sh_addralign <= 1 ||
            shdrs[i].sh_addr % shdrs[i].sh_addralign == old_addr % shdrs[i].sh_addralign) &&
           "shifted allocated section lost its address alignment residue");
  }

  if (!shift_symbols_in_moved_sections(image_, header, shdrs, shift_section_vaddr, delta))
    return std::nullopt;
  if (!shift_relocation_offsets_in_moved_sections(image_, header, shdrs, shift_section_vaddr,
                                                  delta) ||
      !adjust_kernel_descriptor_entry_offsets_in_moved_sections(image_, shdrs, shift_section_vaddr,
                                                                delta))
    return std::nullopt;

  for (size_t i = 0; i < phdrs.size(); ++i) {
    if (!shift_segment_vaddr[i])
      continue;
    assert((phdrs[i].p_align <= 1 || delta % phdrs[i].p_align == 0) &&
           "descriptor tail padding does not preserve shifted LOAD alignment");
    phdrs[i].p_vaddr += delta;
    phdrs[i].p_paddr += delta;
    assert((phdrs[i].p_align <= 1 ||
            phdrs[i].p_offset % phdrs[i].p_align == phdrs[i].p_vaddr % phdrs[i].p_align) &&
           "shifted LOAD lost file/virtual address congruence");
  }

  for (const Elf64_Phdr &phdr : phdrs) {
    if (phdr.p_type != PT_LOAD || phdr.p_align <= 1)
      continue;
    assert(phdr.p_offset % phdr.p_align == phdr.p_vaddr % phdr.p_align &&
           "patched LOAD lost file/virtual address congruence");
  }

  text_tail_size_ += delta;
  write_elf_tables(image_, header, shdrs, phdrs);
  return appended;
}

bool CodeObjectPatcher::append_nonalloc_section(std::string_view name,
                                                std::span<const uint8_t> contents,
                                                uint64_t alignment) {
  if (name.empty() || contents.empty())
    return false;
  if (alignment == 0 || (alignment & (alignment - 1)) != 0)
    return false;
  if (image_.size() < sizeof(Elf64_Ehdr))
    return false;

  auto header = *reinterpret_cast<const Elf64_Ehdr *>(image_.data());
  if (header.e_shoff == 0 || header.e_shentsize != sizeof(Elf64_Shdr) || header.e_shnum == 0)
    return false;
  if (header.e_shstrndx == SHN_UNDEF || header.e_shstrndx >= header.e_shnum)
    return false;
  if (!image_contains_range(image_.size(), header.e_shoff,
                            static_cast<uint64_t>(header.e_shnum) * sizeof(Elf64_Shdr)))
    return false;

  auto maybe_shdrs = read_section_headers(image_, header);
  auto maybe_phdrs = read_program_headers(image_, header);
  if (!maybe_shdrs || !maybe_phdrs)
    return false;
  auto shdrs = std::move(*maybe_shdrs);
  auto phdrs = std::move(*maybe_phdrs);
  Elf64_Shdr &shstrtab = shdrs[header.e_shstrndx];
  if (shstrtab.sh_type != SHT_STRTAB)
    return false;
  if (!image_contains_range(image_.size(), shstrtab.sh_offset, shstrtab.sh_size))
    return false;
  if (shstrtab.sh_size > std::numeric_limits<uint32_t>::max())
    return false;
  if (name.size() > std::numeric_limits<uint32_t>::max() - shstrtab.sh_size - 1)
    return false;
  if (shdrs.size() >= std::numeric_limits<uint16_t>::max())
    return false;

  std::vector<uint8_t> new_shstrtab(
      image_.begin() + static_cast<std::ptrdiff_t>(shstrtab.sh_offset),
      image_.begin() + static_cast<std::ptrdiff_t>(shstrtab.sh_offset + shstrtab.sh_size));
  const uint32_t section_name_offset = static_cast<uint32_t>(new_shstrtab.size());
  new_shstrtab.insert(new_shstrtab.end(), name.begin(), name.end());
  new_shstrtab.push_back('\0');

  const auto maybe_payload_offset = checked_align_up(image_.size(), alignment);
  if (!maybe_payload_offset ||
      *maybe_payload_offset > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
    return false;
  const uint64_t payload_offset = *maybe_payload_offset;
  image_.resize(static_cast<size_t>(payload_offset), 0);
  image_.insert(image_.end(), contents.begin(), contents.end());

  const uint64_t shstrtab_offset = image_.size();
  image_.insert(image_.end(), new_shstrtab.begin(), new_shstrtab.end());

  constexpr uint64_t kSectionHeaderAlignment = 8;
  const auto maybe_new_shoff = checked_align_up(image_.size(), kSectionHeaderAlignment);
  if (!maybe_new_shoff ||
      *maybe_new_shoff > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
    return false;
  const uint64_t new_shoff = *maybe_new_shoff;
  image_.resize(static_cast<size_t>(new_shoff), 0);

  shstrtab.sh_offset = shstrtab_offset;
  shstrtab.sh_size = new_shstrtab.size();
  shstrtab.sh_addralign = 1;

  Elf64_Shdr metadata_section{};
  metadata_section.sh_name = section_name_offset;
  metadata_section.sh_type = SHT_PROGBITS;
  metadata_section.sh_flags = 0;
  metadata_section.sh_addr = 0;
  metadata_section.sh_offset = payload_offset;
  metadata_section.sh_size = contents.size();
  metadata_section.sh_addralign = alignment;
  shdrs.push_back(metadata_section);

  header.e_shoff = new_shoff;
  header.e_shnum = static_cast<uint16_t>(shdrs.size());
  const uint64_t table_size = shdrs.size() * sizeof(Elf64_Shdr);
  if (new_shoff > std::numeric_limits<uint64_t>::max() - table_size ||
      new_shoff + table_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
    return false;
  image_.resize(static_cast<size_t>(new_shoff + table_size), 0);
  write_elf_tables(image_, header, shdrs, phdrs);
  return true;
}

bool CodeObjectPatcher::redirect_kernel_entry(uint64_t descriptor_file_offset,
                                              uint64_t old_entry_text_offset,
                                              uint64_t new_entry_text_offset) {
  if (!image_contains_range(image_.size(), descriptor_file_offset, sizeof(KD)))
    return false;

  KD desc;
  std::memcpy(&desc, image_.data() + descriptor_file_offset, sizeof(desc));
  const int64_t delta =
      static_cast<int64_t>(new_entry_text_offset) - static_cast<int64_t>(old_entry_text_offset);
  const int64_t redirected = static_cast<int64_t>(desc.kernel_code_entry_byte_offset) + delta;
  // The descriptor field is signed because the entry point may be before or
  // after the descriptor in virtual address order. Preserve that signed value
  // when applying the text-relative delta.
  desc.kernel_code_entry_byte_offset = redirected;
  std::memcpy(image_.data() + descriptor_file_offset, &desc, sizeof(desc));
  return true;
}

std::vector<uint8_t> CodeObjectPatcher::emit() const & { return image_; }

std::vector<uint8_t> CodeObjectPatcher::emit() && { return std::move(image_); }

} // namespace rocjitsu
