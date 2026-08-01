// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/kernel_descriptor_scan.h"

#include "rocjitsu/code/amdgpu_elf.h"

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

[[nodiscard]] std::optional<uint64_t> text_vaddr_for_section(uint64_t text_offset,
                                                             uint64_t text_size,
                                                             const Elf64_Ehdr &ehdr,
                                                             const Elf64_Shdr *shdr) {
  for (int i = 0; i < ehdr.e_shnum; ++i) {
    if (shdr[i].sh_offset == text_offset && shdr[i].sh_size == text_size)
      return shdr[i].sh_addr;
  }
  return std::nullopt;
}

} // namespace

std::vector<KernelDescriptorInfo>
scan_kernel_descriptors(std::span<const uint8_t> image, uint64_t text_offset, uint64_t text_size) {
  std::vector<KernelDescriptorInfo> out;
  if (image.size() < sizeof(Elf64_Ehdr))
    return out;

  const auto *ehdr = reinterpret_cast<const Elf64_Ehdr *>(image.data());
  if (!range_in_bounds(ehdr->e_shoff, static_cast<uint64_t>(ehdr->e_shnum) * sizeof(Elf64_Shdr),
                       image.size()))
    return out;

  const auto *shdr = reinterpret_cast<const Elf64_Shdr *>(image.data() + ehdr->e_shoff);
  auto text_vaddr = text_vaddr_for_section(text_offset, text_size, *ehdr, shdr);
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
      // vaddr->file delta does not underflow); the delta must land the file offset
      // within the image (no wrap); and the full 64-byte descriptor must fit from
      // there. Any failure skips this symbol rather than reading past the image.
      const uint16_t sec_idx = symtab[j].st_shndx;
      if (sec_idx >= ehdr->e_shnum || symtab[j].st_value < shdr[sec_idx].sh_addr)
        continue;

      const uint64_t delta = symtab[j].st_value - shdr[sec_idx].sh_addr;
      if (!range_in_bounds(shdr[sec_idx].sh_offset, delta, image.size()))
        continue;
      const uint64_t file_off = shdr[sec_idx].sh_offset + delta;
      if (!range_in_bounds(file_off, sizeof(KD), image.size()))
        continue;
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

} // namespace rocjitsu
