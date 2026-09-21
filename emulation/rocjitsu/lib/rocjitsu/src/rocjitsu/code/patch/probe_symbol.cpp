// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/probe_symbol.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/file_io.h"
#include "rocjitsu/code/patch/error_report.h"

#include <cstring>
#include <span>
#include <vector>

namespace rocjitsu {

namespace {

// Bounds-checked read out of the raw image. memcpy (not reinterpret_cast)
// because the image buffer is byte-aligned and T may have stricter alignment.
// Shares the bounds primitive with the rest of the code loader (file_io.h).
template <typename T>
[[nodiscard]] bool read_pod(std::span<const uint8_t> image, uint64_t offset, T *out) {
  if (!detail::fits_in_bounds(offset, sizeof(T), image.size()))
    return false;
  std::memcpy(out, image.data() + offset, sizeof(T));
  return true;
}

// Read a section header by index, honoring e_shentsize as the stride.
[[nodiscard]] bool read_shdr(std::span<const uint8_t> image, const Elf64_Ehdr &ehdr, uint64_t index,
                             Elf64_Shdr *out) {
  const uint64_t off = ehdr.e_shoff + index * ehdr.e_shentsize;
  return read_pod(image, off, out);
}

// Extract the NUL-terminated name at @p name_off within the string table
// [strtab_off, strtab_off + strtab_size). Returns false if name_off is out of
// range or the string is not terminated inside the table.
[[nodiscard]] bool read_string(std::span<const uint8_t> image, uint64_t strtab_off,
                               uint64_t strtab_size, uint32_t name_off, std::string *out) {
  if (name_off >= strtab_size)
    return false;
  if (!detail::fits_in_bounds(strtab_off, strtab_size, image.size()))
    return false;
  const char *base = reinterpret_cast<const char *>(image.data() + strtab_off);
  const size_t max_len = strtab_size - name_off;
  const size_t len = strnlen(base + name_off, max_len);
  if (len == max_len)
    return false; // Not terminated inside the table.
  out->assign(base + name_off, len);
  return true;
}

// TODO: this scan duplicates AmdGpuCodeObject::load_sections. Could extract a
// code in to a shared ElfSymbol.
//
// Collect every symbol named @p symbol_name in the (single) section of type
// @p sym_type. Returns false on a malformed table; on success @p out_matches
// holds zero or more entries. Search is intentionally confined to one table
// type so that a symbol mirrored in both .symtab and .dynsym is not double
// counted as a duplicate.
[[nodiscard]] bool collect_matches(std::span<const uint8_t> image, const Elf64_Ehdr &ehdr,
                                   uint32_t sym_type, std::string_view symbol_name,
                                   std::vector<Elf64_Sym> *out_matches, std::string *error_out) {
  for (uint64_t i = 0; i < ehdr.e_shnum; ++i) {
    Elf64_Shdr shdr;
    if (!read_shdr(image, ehdr, i, &shdr)) {
      report(error_out, "malformed ELF: section header out of range");
      return false;
    }
    if (shdr.sh_type != sym_type)
      continue;

    if (shdr.sh_entsize < sizeof(Elf64_Sym) || shdr.sh_entsize == 0) {
      report(error_out, "malformed ELF: symbol table entry size too small");
      return false;
    }
    if (shdr.sh_offset > image.size() || shdr.sh_size > image.size() - shdr.sh_offset) {
      report(error_out, "malformed ELF: symbol table extends past end of image");
      return false;
    }
    // The linked string table for this symbol table.
    if (shdr.sh_link >= ehdr.e_shnum) {
      report(error_out, "malformed ELF: symbol table has invalid string-table link");
      return false;
    }
    Elf64_Shdr strtab;
    if (!read_shdr(image, ehdr, shdr.sh_link, &strtab)) {
      report(error_out, "malformed ELF: string-table section header out of range");
      return false;
    }
    // sh_link must name an actual string table; otherwise a symbol table could
    // resolve names against arbitrary (e.g. .text) bytes and match by accident.
    if (strtab.sh_type != SHT_STRTAB) {
      report(error_out, "malformed ELF: symbol table string-table link is not a string table");
      return false;
    }

    const uint64_t count = shdr.sh_size / shdr.sh_entsize;
    for (uint64_t s = 0; s < count; ++s) {
      Elf64_Sym sym;
      if (!read_pod(image, shdr.sh_offset + s * shdr.sh_entsize, &sym)) {
        report(error_out, "malformed ELF: symbol entry out of range");
        return false;
      }
      std::string name;
      if (!read_string(image, strtab.sh_offset, strtab.sh_size, sym.st_name, &name))
        continue; // Skip unreadable names rather than failing the whole table.
      if (name != symbol_name)
        continue;
      // Dedup byte-identical definitions so the same symbol appearing in more
      // than one table of this type (or repeated within one) is treated as a
      // single definition, not a false duplicate. Genuinely conflicting
      // definitions still survive as multiple matches and are rejected upstream.
      bool already_seen = false;
      for (const auto &m : *out_matches) {
        if (std::memcmp(&m, &sym, sizeof(Elf64_Sym)) == 0) {
          already_seen = true;
          break;
        }
      }
      if (!already_seen)
        out_matches->push_back(sym);
    }
  }
  return true;
}

} // namespace

std::optional<ResolvedProbeSymbol> resolve_probe_symbol(const AmdGpuCodeObject &obj,
                                                        std::string_view symbol_name,
                                                        std::string *error_out) {
  // Reject an empty query up front: it would otherwise match the reserved
  // index-0 null symbol (and any unnamed symbols), whose name decodes to "".
  if (symbol_name.empty()) {
    report(error_out, "probe symbol name is empty");
    return std::nullopt;
  }

  const std::span<const uint8_t> image(reinterpret_cast<const uint8_t *>(obj.image_data()),
                                       obj.image_size());

  Elf64_Ehdr ehdr;
  if (!read_pod(image, 0, &ehdr)) {
    report(error_out, "ELF image too small for an ELF header");
    return std::nullopt;
  }
  if (std::memcmp(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE) != 0) {
    report(error_out, "not an ELF image (bad magic)");
    return std::nullopt;
  }
  if (ehdr.e_ident[EI_CLASS] != ELFCLASS64) {
    report(error_out, "unsupported ELF class (expected ELFCLASS64)");
    return std::nullopt;
  }
  if (ehdr.e_shentsize < sizeof(Elf64_Shdr) || ehdr.e_shnum == 0) {
    report(error_out, "ELF has no usable section header table");
    return std::nullopt;
  }
  // Validate the whole section header table up front so per-index reads
  // (read_shdr) cannot wrap a huge e_shoff into an aliased in-bounds offset.
  // Every later read_shdr index is < e_shnum, hence inside this range.
  if (!detail::fits_in_bounds(ehdr.e_shoff, static_cast<uint64_t>(ehdr.e_shnum) * ehdr.e_shentsize,
                              image.size())) {
    report(error_out, "malformed ELF: section header table out of range");
    return std::nullopt;
  }

  // SHT_SYMTAB first, then SHT_DYNSYM. The first type with any name match wins.
  std::vector<Elf64_Sym> matches;
  for (const uint32_t sym_type : {SHT_SYMTAB, SHT_DYNSYM}) {
    matches.clear();
    if (!collect_matches(image, ehdr, sym_type, symbol_name, &matches, error_out))
      return std::nullopt;
    if (!matches.empty())
      break;
  }

  if (matches.empty()) {
    report(error_out, "probe symbol not found");
    return std::nullopt;
  }
  if (matches.size() > 1) {
    report(error_out, "probe symbol is defined more than once");
    return std::nullopt;
  }

  const Elf64_Sym &sym = matches.front();

  if (sym.st_shndx == SHN_UNDEF) {
    report(error_out, "probe symbol is undefined");
    return std::nullopt;
  }
  // Reserved indices (SHN_ABS and everything in the reserved range) are not
  // backed by a normal, file-resident section, so there is no body to copy.
  constexpr uint16_t kShnLoReserve = 0xff00;
  if (sym.st_shndx >= kShnLoReserve) {
    report(error_out, "probe symbol is not backed by a normal section");
    return std::nullopt;
  }
  if (sym.st_shndx >= ehdr.e_shnum) {
    report(error_out, "malformed ELF: probe symbol section index out of range");
    return std::nullopt;
  }
  if (elf_symbol_type(sym.st_info) != kElfSymbolTypeFunc) {
    report(error_out, "probe symbol is not a function (STT_FUNC)");
    return std::nullopt;
  }
  if (sym.st_size == 0) {
    report(error_out, "probe symbol has zero size");
    return std::nullopt;
  }
  // Probe bodies are sequences of instruction words.
  if (sym.st_value % sizeof(uint32_t) != 0 || sym.st_size % sizeof(uint32_t) != 0) {
    report(error_out, "probe symbol value or size is not dword aligned");
    return std::nullopt;
  }

  Elf64_Shdr owner;
  if (!read_shdr(image, ehdr, sym.st_shndx, &owner)) {
    report(error_out, "malformed ELF: probe symbol owning section out of range");
    return std::nullopt;
  }
  if ((owner.sh_flags & SHF_EXECINSTR) == 0) {
    report(error_out, "probe symbol is not in an executable section");
    return std::nullopt;
  }
  if (owner.sh_type == SHT_NOBITS) {
    report(error_out, "probe symbol section occupies no file space");
    return std::nullopt;
  }
  // Validate the owning section's file extent before deriving body_file_offset.
  // With sh_offset + sh_size <= image.size() and the in-section range check
  // below, sh_offset + (st_value - sh_addr) + st_size cannot overflow.
  if (!detail::fits_in_bounds(owner.sh_offset, owner.sh_size, image.size())) {
    report(error_out, "probe symbol owning section extends past end of image");
    return std::nullopt;
  }
  // The symbol value must lie within its owning section's address range.
  if (sym.st_value < owner.sh_addr || sym.st_value - owner.sh_addr > owner.sh_size ||
      sym.st_size > owner.sh_size - (sym.st_value - owner.sh_addr)) {
    report(error_out, "probe symbol body extends past its owning section");
    return std::nullopt;
  }
  const uint64_t body_file_offset = owner.sh_offset + (sym.st_value - owner.sh_addr);
  if (body_file_offset > image.size() || sym.st_size > image.size() - body_file_offset) {
    report(error_out, "probe symbol body extends past end of image");
    return std::nullopt;
  }

  ResolvedProbeSymbol resolved;
  resolved.name = std::string(symbol_name);
  resolved.section_index = sym.st_shndx;
  resolved.st_value = sym.st_value;
  resolved.body_file_offset = body_file_offset;
  resolved.body_size = sym.st_size;
  return resolved;
}

} // namespace rocjitsu
