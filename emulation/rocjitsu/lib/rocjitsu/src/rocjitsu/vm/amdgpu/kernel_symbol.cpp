// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/kernel_symbol.h"

#include <elf.h>
#include <string_view>

namespace rocjitsu {
namespace amdgpu {
namespace {

bool in_range(const uint8_t *ptr, uint64_t size, const uint8_t *base, uint64_t range_size) {
  return ptr >= base && size <= range_size &&
         static_cast<uint64_t>(ptr - base) <= range_size - size;
}

} // namespace

std::string find_kernel_symbol(const uint8_t *kernel_object_ptr, const uint8_t *elf_base,
                               uint64_t elf_accessible) {
  if (!kernel_object_ptr || !elf_base || kernel_object_ptr < elf_base)
    return {};
  if (elf_accessible < sizeof(Elf64_Ehdr))
    return {};
  if (elf_base[0] != 0x7f || elf_base[1] != 'E' || elf_base[2] != 'L' || elf_base[3] != 'F')
    return {};

  uint64_t kd_offset = static_cast<uint64_t>(kernel_object_ptr - elf_base);

  auto *ehdr = reinterpret_cast<const Elf64_Ehdr *>(elf_base);
  if (ehdr->e_phnum == 0 || ehdr->e_phoff == 0 || ehdr->e_phnum > 64)
    return {};
  if (!in_range(elf_base + ehdr->e_phoff, uint64_t{ehdr->e_phnum} * sizeof(Elf64_Phdr), elf_base,
                elf_accessible))
    return {};

  auto *phdrs = reinterpret_cast<const Elf64_Phdr *>(elf_base + ehdr->e_phoff);

  // Use .dynsym via PT_DYNAMIC for exact symbol lookup. DT_SYMTAB and
  // DT_STRTAB point to .dynsym/.dynstr which are in LOAD segments and always
  // host-accessible. Use p_vaddr (not p_offset) because ROCR loads code
  // object segments at their virtual addresses.
  for (uint16_t i = 0; i < ehdr->e_phnum; ++i) {
    if (phdrs[i].p_type != PT_DYNAMIC)
      continue;
    if (!in_range(elf_base + phdrs[i].p_vaddr, phdrs[i].p_memsz, elf_base, elf_accessible))
      break;

    auto *dyn = reinterpret_cast<const Elf64_Dyn *>(elf_base + phdrs[i].p_vaddr);
    uint64_t symtab_off = 0, strtab_off = 0, strsz = 0;
    uint64_t syment = sizeof(Elf64_Sym);
    uint64_t hash_off = 0;
    for (size_t d = 0; d < phdrs[i].p_memsz / sizeof(Elf64_Dyn); ++d) {
      if (dyn[d].d_tag == DT_SYMTAB)
        symtab_off = dyn[d].d_un.d_val;
      else if (dyn[d].d_tag == DT_STRTAB)
        strtab_off = dyn[d].d_un.d_val;
      else if (dyn[d].d_tag == DT_STRSZ)
        strsz = dyn[d].d_un.d_val;
      else if (dyn[d].d_tag == DT_SYMENT)
        syment = dyn[d].d_un.d_val;
      else if (dyn[d].d_tag == DT_HASH)
        hash_off = dyn[d].d_un.d_val;
      else if (dyn[d].d_tag == DT_NULL)
        break;
    }
    if (symtab_off == 0 || strtab_off == 0 || strsz == 0)
      break;

    uint32_t nsyms = 0;
    if (hash_off != 0 && in_range(elf_base + hash_off, 8, elf_base, elf_accessible)) {
      auto *hash = reinterpret_cast<const uint32_t *>(elf_base + hash_off);
      nsyms = hash[1];
    }
    if (nsyms == 0 ||
        !in_range(elf_base + symtab_off, uint64_t{nsyms} * syment, elf_base, elf_accessible))
      break;
    if (!in_range(elf_base + strtab_off, strsz, elf_base, elf_accessible))
      break;

    auto *syms = reinterpret_cast<const Elf64_Sym *>(elf_base + symtab_off);
    auto *strtab = reinterpret_cast<const char *>(elf_base + strtab_off);

    for (uint32_t s = 0; s < nsyms; ++s) {
      if (syms[s].st_value != kd_offset)
        continue;
      if (syms[s].st_name >= strsz)
        continue;
      std::string_view name(strtab + syms[s].st_name);
      if (name.size() > 3 && name.substr(name.size() - 3) == ".kd")
        name = name.substr(0, name.size() - 3);
      if (!name.empty())
        return std::string(name);
    }
    break;
  }

  // Fallback: scan msgpack in PT_NOTE for .kd symbol strings.
  // Returns the first match (no per-kernel matching).
  for (uint16_t i = 0; i < ehdr->e_phnum; ++i) {
    if (phdrs[i].p_type != PT_NOTE || phdrs[i].p_filesz < sizeof(Elf64_Nhdr))
      continue;
    if (!in_range(elf_base + phdrs[i].p_offset, phdrs[i].p_filesz, elf_base, elf_accessible))
      continue;
    auto *nhdr = reinterpret_cast<const Elf64_Nhdr *>(elf_base + phdrs[i].p_offset);
    constexpr uint32_t NT_AMDGPU_METADATA = 32;
    if (nhdr->n_type != NT_AMDGPU_METADATA)
      continue;

    uint32_t name_aligned = (nhdr->n_namesz + 3) & ~3u;
    uint64_t desc_off = phdrs[i].p_offset + sizeof(Elf64_Nhdr) + name_aligned;
    uint32_t desc_sz = nhdr->n_descsz;
    if (!in_range(elf_base + desc_off, desc_sz, elf_base, elf_accessible))
      continue;
    auto *note = elf_base + desc_off;

    std::string best_name;
    for (size_t pos = 2; pos + 2 < desc_sz; ++pos) {
      if (note[pos] != '.' || note[pos + 1] != 'k' || note[pos + 2] != 'd')
        continue;
      size_t end = pos + 3;
      if (end < desc_sz && note[end] >= 0x20 && note[end] < 0x7f)
        continue;
      size_t start = pos;
      while (start > 0 && note[start - 1] >= 0x20 && note[start - 1] < 0x7f)
        --start;
      if (start == pos)
        continue;
      std::string_view sym(reinterpret_cast<const char *>(note + start), pos - start);
      if (best_name.empty())
        best_name = std::string(sym);
    }
    if (!best_name.empty())
      return best_name;
  }
  return {};
}

} // namespace amdgpu
} // namespace rocjitsu
