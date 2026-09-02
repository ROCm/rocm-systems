// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/kernel_descriptor_scan.h"

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/isa/isa_traits.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <optional>
#include <unordered_set>
#include <utility>

namespace rocjitsu {

namespace {

using KD = rocr::llvm::amdhsa::kernel_descriptor_t;

static_assert(sizeof(KD) == 64, "AMDHSA kernel descriptor size changed");

// True if [offset, offset + length) fits within [0, size) without overflow (the
// image is an arbitrary span, so offset + length may wrap). Use before forming any
// pointer into the image.
[[nodiscard]] bool range_in_bounds(uint64_t offset, uint64_t length, uint64_t size) {
  return offset <= size && length <= size - offset;
}

[[nodiscard]] std::optional<std::string>
kernel_descriptor_symbol_name(const Elf64_Sym &sym, const char *strtab, size_t strtab_size) {
  if (sym.st_size != sizeof(KD))
    return std::nullopt;

  // AMDHSA kernel descriptors are global object symbols. Size alone is not a
  // durable signal because unrelated data objects can also be 64 bytes.
  if (elf_symbol_type(sym.st_info) != kElfSymbolTypeObject ||
      elf_symbol_bind(sym.st_info) != kElfSymbolBindGlobal)
    return std::nullopt;

  // AMDHSA descriptors are named "<kernel>.kd". An unnamed 64-byte global object
  // is ambiguous, so require the ABI suffix instead of treating stripped or
  // minimized symbol records as descriptors.
  if (strtab == nullptr || strtab_size == 0 || sym.st_name == 0)
    return std::nullopt;
  if (sym.st_name >= strtab_size)
    return std::nullopt;

  const char *name = strtab + sym.st_name;
  const size_t avail = strtab_size - sym.st_name;
  const size_t len = strnlen(name, avail);
  // No NUL within bounds: the name is unterminated, so the strcmp below would read
  // past the table (the image span is arbitrary). Reject before the suffix check.
  if (len == avail)
    return std::nullopt;
  if (len <= 3 || std::strcmp(name + len - 3, ".kd") != 0)
    return std::nullopt;
  return std::string(name, len - 3);
}

[[nodiscard]] std::optional<uint64_t>
text_vaddr_for_section(uint64_t text_offset, uint64_t text_size, const Elf64_Ehdr &ehdr,
                       const Elf64_Shdr *shdr, std::optional<size_t> text_section_index) {
  if (text_section_index) {
    if (*text_section_index >= ehdr.e_shnum)
      return std::nullopt;
    const Elf64_Shdr &text = shdr[*text_section_index];
    if (text.sh_type == SHT_NOBITS || text.sh_offset != text_offset || text.sh_size != text_size)
      return std::nullopt;
    return text.sh_addr;
  }
  for (int i = 0; i < ehdr.e_shnum; ++i) {
    if (shdr[i].sh_offset == text_offset && shdr[i].sh_size == text_size)
      return shdr[i].sh_addr;
  }
  return std::nullopt;
}

} // namespace

KernelNameIndexMatch match_kernel_name_index(std::span<const uint8_t> image,
                                             std::span<const std::string> exact_kernel_names) {
  if (exact_kernel_names.empty())
    return KernelNameIndexMatch::Matched;
  if (image.size() < sizeof(Elf64_Ehdr))
    return KernelNameIndexMatch::Indeterminate;

  Elf64_Ehdr header{};
  std::memcpy(&header, image.data(), sizeof(header));
  if (std::memcmp(header.e_ident, EI_MAGIC, EI_MAGIC_SIZE) != 0 ||
      header.e_ident[EI_CLASS] != ELFCLASS64 || header.e_ident[EI_OSABI] != ELFOSABI_AMDGPU_HSA ||
      header.e_shentsize != sizeof(Elf64_Shdr) || header.e_shnum == 0 ||
      !range_in_bounds(header.e_shoff, static_cast<uint64_t>(header.e_shnum) * sizeof(Elf64_Shdr),
                       image.size())) {
    return KernelNameIndexMatch::Indeterminate;
  }

  const auto section = [&](size_t index) -> std::optional<Elf64_Shdr> {
    if (index >= header.e_shnum)
      return std::nullopt;
    Elf64_Shdr value{};
    std::memcpy(&value, image.data() + header.e_shoff + index * sizeof(value), sizeof(value));
    return value;
  };

  bool indeterminate = false;
  for (size_t section_index = 0; section_index < header.e_shnum; ++section_index) {
    const Elf64_Shdr symbols = *section(section_index);
    if (symbols.sh_type != SHT_SYMTAB && symbols.sh_type != SHT_DYNSYM)
      continue;
    if (symbols.sh_entsize != sizeof(Elf64_Sym) || symbols.sh_size % sizeof(Elf64_Sym) != 0 ||
        !range_in_bounds(symbols.sh_offset, symbols.sh_size, image.size()) ||
        symbols.sh_link >= header.e_shnum) {
      indeterminate = true;
      continue;
    }
    const Elf64_Shdr strings = *section(symbols.sh_link);
    if (strings.sh_type != SHT_STRTAB ||
        !range_in_bounds(strings.sh_offset, strings.sh_size, image.size())) {
      indeterminate = true;
      continue;
    }

    for (uint64_t symbol_offset = 0; symbol_offset < symbols.sh_size;
         symbol_offset += sizeof(Elf64_Sym)) {
      Elf64_Sym symbol{};
      std::memcpy(&symbol, image.data() + symbols.sh_offset + symbol_offset, sizeof(symbol));
      if (symbol.st_name >= strings.sh_size) {
        indeterminate = true;
        continue;
      }
      const char *name_begin =
          reinterpret_cast<const char *>(image.data() + strings.sh_offset + symbol.st_name);
      const size_t maximum_name_size = static_cast<size_t>(strings.sh_size - symbol.st_name);
      const size_t name_size = strnlen(name_begin, maximum_name_size);
      if (name_size == maximum_name_size) {
        indeterminate = true;
        continue;
      }
      std::string_view name(name_begin, name_size);
      if (!name.ends_with(".kd"))
        continue;
      name.remove_suffix(3);
      if (std::ranges::any_of(exact_kernel_names,
                              [&](const std::string &expected) { return name == expected; })) {
        return KernelNameIndexMatch::Matched;
      }
    }
  }
  return indeterminate ? KernelNameIndexMatch::Indeterminate : KernelNameIndexMatch::NoMatch;
}

std::vector<KernelDescriptorInfo>
scan_kernel_descriptors(std::span<const uint8_t> image, uint64_t text_offset, uint64_t text_size,
                        std::optional<size_t> text_section_index) {
  std::vector<KernelDescriptorInfo> out;
  if (image.size() < sizeof(Elf64_Ehdr))
    return out;

  const auto *ehdr = reinterpret_cast<const Elf64_Ehdr *>(image.data());
  if (!range_in_bounds(ehdr->e_shoff, static_cast<uint64_t>(ehdr->e_shnum) * sizeof(Elf64_Shdr),
                       image.size()))
    return out;

  const auto *shdr = reinterpret_cast<const Elf64_Shdr *>(image.data() + ehdr->e_shoff);
  auto text_vaddr = text_vaddr_for_section(text_offset, text_size, *ehdr, shdr, text_section_index);
  if (!text_vaddr)
    return out;
  constexpr uint64_t max_u64 = std::numeric_limits<uint64_t>::max();
  if (*text_vaddr > max_u64 - text_size)
    return out;
  const uint64_t text_end = *text_vaddr + text_size;

  // .symtab and .dynsym may both describe the same descriptor. Discovery is keyed
  // by descriptor bytes, so visit each file offset once.
  std::unordered_set<uint64_t> seen_descriptor_offsets;
  for (int i = 0; i < ehdr->e_shnum; ++i) {
    if (shdr[i].sh_type != SHT_SYMTAB && shdr[i].sh_type != SHT_DYNSYM)
      continue;
    if (!range_in_bounds(shdr[i].sh_offset, shdr[i].sh_size, image.size()) ||
        shdr[i].sh_entsize == 0)
      continue;
    if (shdr[i].sh_entsize != sizeof(Elf64_Sym))
      continue;

    const char *strtab = nullptr;
    size_t strtab_size = 0;
    if (shdr[i].sh_link < ehdr->e_shnum) {
      const auto &strtab_shdr = shdr[shdr[i].sh_link];
      if (range_in_bounds(strtab_shdr.sh_offset, strtab_shdr.sh_size, image.size())) {
        strtab = reinterpret_cast<const char *>(image.data() + strtab_shdr.sh_offset);
        strtab_size = strtab_shdr.sh_size;
      }
    }

    const auto *symtab = reinterpret_cast<const Elf64_Sym *>(image.data() + shdr[i].sh_offset);
    const size_t nsyms = shdr[i].sh_size / shdr[i].sh_entsize;
    for (size_t j = 0; j < nsyms; ++j) {
      auto kernel_name = kernel_descriptor_symbol_name(symtab[j], strtab, strtab_size);
      if (!kernel_name)
        continue;

      // Map the descriptor symbol to the file bytes it points at, validating every
      // step so a crafted symbol cannot form an out-of-bounds pointer: its section
      // index must exist; its value must sit at or above that section's vaddr (so the
      // vaddr->file delta does not underflow); the owning section must fit within the
      // image; and the full 64-byte descriptor must fit within *that section* -- not
      // merely the image -- so a .kd near the section's end cannot extend past sh_size
      // into an adjacent section and later be mutated across the boundary. Any failure
      // skips this symbol rather than reading past the section.
      const uint16_t sec_idx = symtab[j].st_shndx;
      if (sec_idx >= ehdr->e_shnum || symtab[j].st_value < shdr[sec_idx].sh_addr)
        continue;

      const Elf64_Shdr &owner = shdr[sec_idx];
      const uint64_t delta = symtab[j].st_value - owner.sh_addr;
      if (!range_in_bounds(owner.sh_offset, owner.sh_size, image.size()) ||
          !range_in_bounds(delta, sizeof(KD), owner.sh_size))
        continue;
      const uint64_t file_off = owner.sh_offset + delta;
      if (!seen_descriptor_offsets.insert(file_off).second)
        continue;

      KD desc;
      std::memcpy(&desc, image.data() + file_off, sizeof(desc));
      const int64_t entry_vaddr_signed =
          static_cast<int64_t>(symtab[j].st_value) + desc.kernel_code_entry_byte_offset;

      if (entry_vaddr_signed < 0)
        continue;
      const uint64_t entry_vaddr = static_cast<uint64_t>(entry_vaddr_signed);
      if (entry_vaddr < *text_vaddr || entry_vaddr >= text_end)
        continue;

      out.push_back(KernelDescriptorInfo{
          .descriptor_file_offset = file_off,
          .kernel_name = std::move(*kernel_name),
          .entry_text_offset = entry_vaddr - *text_vaddr,
          .descriptor = desc,
      });
    }
  }
  return out;
}

uint8_t kernel_wavefront_size(rj_code_arch_t arch, const KD &desc) {
  // CDNA kernels are Wave64 in the code objects currently handled here.
  if (arch_is_cdna_4_or_lower(arch))
    return 64;

  // gfx1250 is Wave32-only. Do not interpret a missing legacy descriptor bit
  // as Wave64: older producers may omit the bit even though the hardware has
  // no Wave64 launch mode.
  if (arch == ROCJITSU_CODE_ARCH_CDNA5)
    return 32;

  // RDNA descriptors opt into Wave32 with ENABLE_WAVEFRONT_SIZE32. If the bit is
  // clear, launch hardware interprets the descriptor as Wave64.
  if (arch_is_rdna(arch)) {
    const bool wave32 =
        AMDHSA_BITS_GET(desc.kernel_code_properties,
                        rocr::llvm::amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32);
    return wave32 ? 32 : 64;
  }

  return 64;
}

uint32_t descriptor_vgpr_granularity_for_wavefront(rj_code_arch_t arch, uint32_t wavefront_size) {
  // This is the AMDHSA kernel-descriptor encoding granularity for
  // COMPUTE_PGM_RSRC1.GRANULATED_WORKITEM_VGPR_COUNT, not the physical VGPR
  // allocation block from the ISA manuals. For example, RDNA3/RDNA4 manuals
  // describe Wave64 physical allocation in blocks of 8 VGPRs (or 12 on
  // 1536-VGPR/SIMD parts), while the AMDHSA descriptor table encodes
  // GFX10-GFX12 Wave64 as max(0, ceil(vgprs_used / 4) - 1).
  //
  // If/when occupancy modeling needs the physical allocation block size, add a
  // separate helper for that policy. Reusing this descriptor helper for
  // occupancy would mix two different hardware contracts.
  if (arch == ROCJITSU_CODE_ARCH_CDNA1)
    return 4;
  if (arch_is_cdna_4_or_lower(arch))
    return 8;
  // gfx1250 exposes four 256-VGPR banks selected by WAVE_MODE.VGPR_MSB. Its
  // AMDHSA descriptor allocates that combined Wave32 namespace in blocks of
  // 16 VGPRs, unlike the 8-VGPR Wave32 granule used by generic RDNA targets.
  if (arch == ROCJITSU_CODE_ARCH_CDNA5)
    return 16;
  if (arch_is_rdna(arch))
    return wavefront_size == 32 ? 8 : 4;
  return 1;
}

} // namespace rocjitsu
