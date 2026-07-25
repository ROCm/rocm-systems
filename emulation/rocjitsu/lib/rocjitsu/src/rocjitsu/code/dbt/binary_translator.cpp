// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/dbt/binary_translator.h"

#include "rocjitsu/analysis/def_use_chain.h"
#include "rocjitsu/analysis/gfx1250_vgpr_msb.h"
#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/dbt/binary_translator_internal.h"
#include "rocjitsu/code/dbt/generated/encoding_cdna4_to_cdna3.h"
#include "rocjitsu/code/dbt/generated/encoding_cdna4_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/encoding_cdna4_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna4_to_cdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna4_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna4_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_types.h"
#include "rocjitsu/code/dbt/kernel_descriptor_translator.h"
#include "rocjitsu/code/dbt/lds_virtualization.h"
#include "rocjitsu/code/dbt/legalization/gfx1250_b0_to_a0.h"
#include "rocjitsu/code/dbt/semantic_translator.h"
#include "rocjitsu/code/dbt/virtual_lds.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/code/patch/kernarg_extension.h"
#include "rocjitsu/code/patch/kernel_text_layout.h"
#include "rocjitsu/code/patch/sidecar_metadata.h"
#include "rocjitsu/code/relocation_function_table.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/isa_properties.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/isa_traits.h"
#include "util/except.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocjitsu {

namespace {

EncodingTranslateFn select_encoding_translator(rj_code_arch_t guest, rj_code_arch_t host) {
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA4)
    return cdna4_to_rdna4::translate_encoding_cdna4_to_rdna4;
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_CDNA3)
    return cdna4_to_cdna3::translate_encoding_cdna4_to_cdna3;
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA3)
    return cdna4_to_rdna3::translate_encoding_cdna4_to_rdna3;
  return nullptr;
}

LegalizationLookupFn select_legalization(rj_code_arch_t guest, rj_code_arch_t host) {
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA4) {
    return [](uint16_t enc_id, uint16_t opcode) -> const InstructionLegalization * {
      return lookup(kLegalization_cdna4_to_rdna4, enc_id, opcode);
    };
  }
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_CDNA3) {
    return [](uint16_t enc_id, uint16_t opcode) -> const InstructionLegalization * {
      return lookup(kLegalization_cdna4_to_cdna3, enc_id, opcode);
    };
  }
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA3) {
    return [](uint16_t enc_id, uint16_t opcode) -> const InstructionLegalization * {
      return lookup(kLegalization_cdna4_to_rdna3, enc_id, opcode);
    };
  }
  return nullptr;
}

[[nodiscard]] std::vector<uint32_t> raw_words_for_inst(const Instruction &inst) {
  const uint32_t *raw = inst.raw_encoding();
  if (!raw)
    return {};
  return {raw, raw + inst.size() / sizeof(uint32_t)};
}

[[nodiscard]] uint32_t text_word_at(std::span<const uint8_t> text, uint64_t offset) {
  uint32_t word = 0;
  if (offset + sizeof(word) <= text.size())
    std::memcpy(&word, text.data() + offset, sizeof(word));
  return word;
}

[[nodiscard]] bool words_changed(std::span<const uint32_t> before,
                                 std::span<const uint32_t> after) {
  if (before.size() != after.size())
    return true;
  return !std::ranges::equal(before, after);
}

void append_diagnostic(std::vector<TranslationDiagnostic> &diagnostics, DiagnosticSeverity severity,
                       DiagnosticKind kind, std::string message,
                       std::optional<uint64_t> guest_offset = std::nullopt,
                       std::string mnemonic = {}, std::vector<std::string> required_work = {}) {
  diagnostics.push_back({.severity = severity,
                         .kind = kind,
                         .guest_offset = guest_offset,
                         .mnemonic = std::move(mnemonic),
                         .message = std::move(message),
                         .required_work = std::move(required_work)});
}

void append_error(std::vector<TranslationDiagnostic> &diagnostics, DiagnosticKind kind,
                  std::string message, std::optional<uint64_t> guest_offset = std::nullopt,
                  std::string mnemonic = {}, std::vector<std::string> required_work = {}) {
  append_diagnostic(diagnostics, DiagnosticSeverity::Error, kind, std::move(message), guest_offset,
                    std::move(mnemonic), std::move(required_work));
}

void append_warning(std::vector<TranslationDiagnostic> &diagnostics, DiagnosticKind kind,
                    std::string message, std::optional<uint64_t> guest_offset = std::nullopt,
                    std::string mnemonic = {}, std::vector<std::string> required_work = {}) {
  append_diagnostic(diagnostics, DiagnosticSeverity::Warning, kind, std::move(message),
                    guest_offset, std::move(mnemonic), std::move(required_work));
}

void append_diagnostics(std::vector<TranslationDiagnostic> &dst,
                        const std::vector<TranslationDiagnostic> &src) {
  dst.insert(dst.end(), src.begin(), src.end());
}

[[nodiscard]] bool image_contains_range(size_t image_size, uint64_t offset, uint64_t size) {
  return offset <= image_size && size <= image_size - offset;
}

[[nodiscard]] bool interval_contains(uint64_t outer_begin, uint64_t outer_size,
                                     uint64_t inner_begin, uint64_t inner_size) {
  return inner_begin >= outer_begin && inner_begin - outer_begin <= outer_size &&
         inner_size <= outer_size - (inner_begin - outer_begin);
}

/// @brief One decoded, non-empty ELF function-symbol extent in `.text`.
struct CallableFunctionRange {
  uint64_t begin = 0;
  uint64_t end = 0;
  std::string name;
  uint8_t binding = kElfSymbolBindLocal;
  uint8_t visibility = kElfSymbolVisibilityDefault;

  [[nodiscard]] bool externally_visible() const {
    return (binding == kElfSymbolBindGlobal || binding == kElfSymbolBindWeak) &&
           (visibility == kElfSymbolVisibilityDefault ||
            visibility == kElfSymbolVisibilityProtected);
  }
};

/// @brief Exhaustive ELF facts used to distinguish data-only objects from code.
///
/// The translator itself currently rewrites one canonical `.text` section, but
/// deciding that an object has no executable work cannot use that implementation
/// detail. This analysis independently inspects every executable section,
/// executable PT_LOAD, executable symbol, descriptor-shaped symbol, and
/// relocation target. Unsupported executable layouts fail closed instead of
/// being mistaken for a successful no-op.
struct ExecutableElfAnalysis {
  bool valid = false;
  bool data_only = false;
  bool canonical_text_supported = false;
  bool assume_llvm_amdhsa_callable_abi = false;
  bool has_kernel_or_descriptor_symbol = false;
  size_t canonical_text_index = 0;
  uint64_t canonical_text_offset = 0;
  uint64_t canonical_text_size = 0;
  uint64_t canonical_text_vaddr = 0;
  std::vector<CallableFunctionRange> callable_functions;
  std::vector<uint64_t> zero_sized_function_entries;
  std::string error;
};

[[nodiscard]] ExecutableElfAnalysis analyze_executable_elf(std::span<const uint8_t> image,
                                                           uint64_t selected_text_offset,
                                                           uint64_t selected_text_size) {
  ExecutableElfAnalysis result;
  auto fail = [&](std::string message) {
    result.error = std::move(message);
    return std::move(result);
  };

  if (image.size() < sizeof(Elf64_Ehdr))
    return fail("code object is too small to contain an ELF header");

  Elf64_Ehdr header{};
  std::memcpy(&header, image.data(), sizeof(header));
  if (std::memcmp(header.e_ident, EI_MAGIC, EI_MAGIC_SIZE) != 0 ||
      header.e_ident[EI_CLASS] != ELFCLASS64 || header.e_machine != EM_AMDGPU) {
    return fail("code object does not contain a supported AMDGPU ELF64 image");
  }
  // Inferring a callable ABI from these ELF fields alone is not generally
  // safe: they establish the linked AMDHSA shape, but not the producer or
  // calling convention. All descriptorless STT_FUNC inputs known to us are
  // LLVM-produced, so assume its C/Fast/Cold callable ABI for this shape.
  // TODO: Generalize this policy when another descriptorless producer is
  // identified or a reliable producer/convention marker becomes available.
  result.assume_llvm_amdhsa_callable_abi =
      header.e_ident[EI_OSABI] == ELFOSABI_AMDGPU_HSA && header.e_type == ET_DYN &&
      header.e_ident[EI_ABIVERSION] >= ELFABIVERSION_AMDGPU_HSA_V4 &&
      header.e_ident[EI_ABIVERSION] <= ELFABIVERSION_AMDGPU_HSA_V6;
  if (header.e_shnum == 0 || header.e_shentsize != sizeof(Elf64_Shdr) ||
      !image_contains_range(image.size(), header.e_shoff,
                            static_cast<uint64_t>(header.e_shnum) * sizeof(Elf64_Shdr))) {
    return fail("code object has an invalid section-header table");
  }

  std::vector<Elf64_Shdr> sections(header.e_shnum);
  std::memcpy(sections.data(), image.data() + header.e_shoff, sections.size() * sizeof(Elf64_Shdr));
  if (header.e_shstrndx >= sections.size()) {
    return fail("code object has an invalid section-name string table index");
  }
  const Elf64_Shdr &shstrtab = sections[header.e_shstrndx];
  if (shstrtab.sh_type != SHT_STRTAB ||
      !image_contains_range(image.size(), shstrtab.sh_offset, shstrtab.sh_size)) {
    return fail("code object has an invalid section-name string table");
  }
  const char *section_names = reinterpret_cast<const char *>(image.data() + shstrtab.sh_offset);
  auto section_name = [&](size_t index) -> std::optional<std::string_view> {
    if (index >= sections.size() || sections[index].sh_name >= shstrtab.sh_size)
      return std::nullopt;
    const uint64_t offset = sections[index].sh_name;
    const size_t available = static_cast<size_t>(shstrtab.sh_size - offset);
    const char *name = section_names + offset;
    const size_t length = strnlen(name, available);
    if (length == available)
      return std::nullopt;
    return std::string_view(name, length);
  };

  std::vector<size_t> nonempty_executable_sections;
  std::vector<std::pair<uint64_t, uint64_t>> executable_vaddr_ranges;
  for (size_t i = 0; i < sections.size(); ++i) {
    const Elf64_Shdr &section = sections[i];
    if (section.sh_type != SHT_NOBITS &&
        !image_contains_range(image.size(), section.sh_offset, section.sh_size)) {
      return fail("code object has a section payload outside the ELF image");
    }
    const auto name = section_name(i);
    if (!name)
      return fail("code object has an unterminated or invalid section name");
    if ((section.sh_flags & SHF_EXECINSTR) == 0 || section.sh_size == 0)
      continue;
    nonempty_executable_sections.push_back(i);
    if (section.sh_addr > std::numeric_limits<uint64_t>::max() - section.sh_size)
      return fail("code object has an overflowing executable section address range");
    executable_vaddr_ranges.emplace_back(section.sh_addr, section.sh_size);
  }

  std::vector<Elf64_Phdr> program_headers;
  bool has_nonempty_executable_segment = false;
  if (header.e_phnum != 0) {
    if (header.e_phentsize != sizeof(Elf64_Phdr) ||
        !image_contains_range(image.size(), header.e_phoff,
                              static_cast<uint64_t>(header.e_phnum) * sizeof(Elf64_Phdr))) {
      return fail("code object has an invalid program-header table");
    }
    program_headers.resize(header.e_phnum);
    std::memcpy(program_headers.data(), image.data() + header.e_phoff,
                program_headers.size() * sizeof(Elf64_Phdr));
    for (const Elf64_Phdr &program : program_headers) {
      if (program.p_type != PT_LOAD)
        continue;
      if (!image_contains_range(image.size(), program.p_offset, program.p_filesz))
        return fail("code object has a PT_LOAD payload outside the ELF image");
      if ((program.p_flags & PF_X) == 0 || (program.p_filesz == 0 && program.p_memsz == 0))
        continue;
      has_nonempty_executable_segment = true;
      if (program.p_vaddr > std::numeric_limits<uint64_t>::max() - program.p_memsz)
        return fail("code object has an overflowing executable PT_LOAD address range");
      executable_vaddr_ranges.emplace_back(program.p_vaddr, program.p_memsz);
    }
  }

  if (nonempty_executable_sections.size() == 1) {
    const size_t index = nonempty_executable_sections.front();
    const Elf64_Shdr &text = sections[index];
    const auto name = section_name(index);
    if (name && *name == ".text" && text.sh_type == SHT_PROGBITS &&
        text.sh_offset == selected_text_offset && text.sh_size == selected_text_size) {
      result.canonical_text_supported = true;
      result.canonical_text_index = index;
      result.canonical_text_offset = text.sh_offset;
      result.canonical_text_size = text.sh_size;
      result.canonical_text_vaddr = text.sh_addr;
    }
  }

  // A linked executable object must not hide non-zero instructions in an
  // executable segment outside the one section DBT rewrites. Zero file padding
  // is harmless (zero is not a gfx1250 instruction), but any other byte is an
  // executable payload that would otherwise bypass translation.
  if (result.canonical_text_supported) {
    const uint64_t text_file_begin = result.canonical_text_offset;
    const uint64_t text_file_size = result.canonical_text_size;
    const uint64_t text_vaddr_begin = result.canonical_text_vaddr;
    bool text_is_loaded_executable = header.e_type == ET_REL;
    for (const Elf64_Phdr &program : program_headers) {
      if (program.p_type != PT_LOAD || (program.p_flags & PF_X) == 0 ||
          (program.p_filesz == 0 && program.p_memsz == 0)) {
        continue;
      }
      const bool file_contains_text =
          interval_contains(program.p_offset, program.p_filesz, text_file_begin, text_file_size);
      const bool text_contains_file =
          interval_contains(text_file_begin, text_file_size, program.p_offset, program.p_filesz);
      const bool memory_contains_text =
          interval_contains(program.p_vaddr, program.p_memsz, text_vaddr_begin, text_file_size);
      const bool text_contains_memory =
          interval_contains(text_vaddr_begin, text_file_size, program.p_vaddr, program.p_memsz);
      if (file_contains_text && memory_contains_text)
        text_is_loaded_executable = true;

      const uint64_t segment_begin = program.p_offset;
      const uint64_t segment_end = program.p_offset + program.p_filesz;
      const uint64_t text_file_end = text_file_begin + text_file_size;
      const auto contains_nonzero_bytes = [&](uint64_t begin, uint64_t end) {
        if (begin >= end)
          return false;
        return std::ranges::any_of(
            image.subspan(static_cast<size_t>(begin), static_cast<size_t>(end - begin)),
            [](uint8_t byte) { return byte != 0; });
      };
      // Inspect only the segment gaps around .text. Scanning through the text
      // body itself made this structural guard needlessly O(code size).
      const uint64_t prefix_end = std::min(segment_end, text_file_begin);
      const uint64_t suffix_begin = std::max(segment_begin, text_file_end);
      if (contains_nonzero_bytes(segment_begin, prefix_end) ||
          contains_nonzero_bytes(suffix_begin, segment_end)) {
        result.canonical_text_supported = false;
      }
      if (!result.canonical_text_supported)
        break;
      // Every executable mapping must describe this .text allocation. A
      // disjoint or only-partially-overlapping segment is another executable
      // payload even when its current file bytes happen to be zero.
      if ((!file_contains_text && !text_contains_file) ||
          (!memory_contains_text && !text_contains_memory)) {
        result.canonical_text_supported = false;
        break;
      }
    }
    if (!text_is_loaded_executable)
      result.canonical_text_supported = false;
  }

  bool has_executable_symbol = false;
  bool has_executable_relocation_target = false;
  std::unordered_map<uint64_t, CallableFunctionRange> functions_by_start;
  std::vector<std::vector<Elf64_Sym>> symbol_tables(sections.size());
  for (size_t i = 0; i < sections.size(); ++i) {
    const Elf64_Shdr &symtab = sections[i];
    if (symtab.sh_type != SHT_SYMTAB && symtab.sh_type != SHT_DYNSYM)
      continue;
    if (symtab.sh_entsize != sizeof(Elf64_Sym) || symtab.sh_size % sizeof(Elf64_Sym) != 0 ||
        !image_contains_range(image.size(), symtab.sh_offset, symtab.sh_size) ||
        symtab.sh_link >= sections.size()) {
      return fail("code object has an invalid symbol table");
    }
    const Elf64_Shdr &strtab = sections[symtab.sh_link];
    if (strtab.sh_type != SHT_STRTAB ||
        !image_contains_range(image.size(), strtab.sh_offset, strtab.sh_size)) {
      return fail("code object has an invalid symbol string table");
    }
    const char *strings = reinterpret_cast<const char *>(image.data() + strtab.sh_offset);
    const size_t count = static_cast<size_t>(symtab.sh_size / sizeof(Elf64_Sym));
    auto &symbols = symbol_tables[i];
    symbols.resize(count);
    std::memcpy(symbols.data(), image.data() + symtab.sh_offset, symtab.sh_size);
    for (const Elf64_Sym &symbol : symbols) {
      if (symbol.st_name >= strtab.sh_size)
        return fail("code object has a symbol name outside its string table");
      const size_t available = static_cast<size_t>(strtab.sh_size - symbol.st_name);
      const char *raw_name = strings + symbol.st_name;
      const size_t name_size = strnlen(raw_name, available);
      if (name_size == available)
        return fail("code object has an unterminated symbol name");
      const std::string_view name(raw_name, name_size);
      const uint8_t type = elf_symbol_type(symbol.st_info);
      const bool descriptor_symbol = name.size() > 3 && name.ends_with(".kd");
      if (descriptor_symbol || type == kElfSymbolTypeAmdGpuHsaKernel)
        result.has_kernel_or_descriptor_symbol = true;

      const bool defined_in_section = symbol.st_shndx != SHN_UNDEF && symbol.st_shndx != SHN_ABS &&
                                      symbol.st_shndx < sections.size();
      const bool defined_in_executable_section =
          defined_in_section && (sections[symbol.st_shndx].sh_flags & SHF_EXECINSTR) != 0;
      if ((type == kElfSymbolTypeFunc && defined_in_executable_section) ||
          type == kElfSymbolTypeAmdGpuHsaKernel) {
        has_executable_symbol = true;
      }
      if (type != kElfSymbolTypeFunc || !defined_in_executable_section)
        continue;
      if (!result.canonical_text_supported || symbol.st_shndx != result.canonical_text_index) {
        continue;
      }
      const Elf64_Shdr &text = sections[result.canonical_text_index];
      uint64_t begin = symbol.st_value;
      if (header.e_type != ET_REL) {
        if (begin < text.sh_addr)
          return fail("code object has an STT_FUNC value before .text");
        begin -= text.sh_addr;
      }
      if (symbol.st_size == 0) {
        if (begin % sizeof(uint32_t) != 0 || begin > text.sh_size)
          return fail("code object has an invalid zero-sized STT_FUNC entry");
        // Linked AMDGPU kernel symbols commonly remain zero-sized STT_FUNCs.
        // Retain the entry for later descriptor corroboration; without a .kd at
        // the same address, its callable extent and ownership are unknowable.
        result.zero_sized_function_entries.push_back(begin);
        continue;
      }
      if (begin % sizeof(uint32_t) != 0 || symbol.st_size % sizeof(uint32_t) != 0 ||
          begin > text.sh_size || symbol.st_size > text.sh_size - begin) {
        return fail("code object has an STT_FUNC range that cannot be translated safely");
      }
      CallableFunctionRange function{.begin = begin,
                                     .end = begin + symbol.st_size,
                                     .name = std::string(name),
                                     .binding = elf_symbol_bind(symbol.st_info),
                                     .visibility = elf_symbol_visibility(symbol.st_other)};
      auto [it, inserted] = functions_by_start.try_emplace(begin, function);
      if (!inserted && it->second.end != function.end)
        return fail("code object has conflicting STT_FUNC ranges for one entry");
      if (!inserted && (it->second.binding != function.binding ||
                        it->second.visibility != function.visibility)) {
        return fail("code object has conflicting STT_FUNC linkage for one entry");
      }
      if (!inserted && it->second.name.empty() && !function.name.empty())
        it->second.name = std::move(function.name);
    }
  }

  for (size_t i = 0; i < sections.size(); ++i) {
    const Elf64_Shdr &relocs = sections[i];
    if (relocs.sh_type != SHT_RELA && relocs.sh_type != SHT_REL)
      continue;
    const bool is_rela = relocs.sh_type == SHT_RELA;
    const size_t entry_size = is_rela ? sizeof(Elf64_Rela) : sizeof(Elf64_Rel);
    if (relocs.sh_entsize != entry_size || relocs.sh_size % entry_size != 0 ||
        !image_contains_range(image.size(), relocs.sh_offset, relocs.sh_size)) {
      return fail("code object has an invalid relocation section");
    }
    const std::vector<Elf64_Sym> *symbols = nullptr;
    if (relocs.sh_link < symbol_tables.size() && (sections[relocs.sh_link].sh_type == SHT_SYMTAB ||
                                                  sections[relocs.sh_link].sh_type == SHT_DYNSYM)) {
      symbols = &symbol_tables[relocs.sh_link];
    } else if (relocs.sh_link != SHN_UNDEF) {
      return fail("code object has a relocation section with an invalid symbol table link");
    }
    const size_t count = static_cast<size_t>(relocs.sh_size / entry_size);
    for (size_t j = 0; j < count; ++j) {
      uint64_t info = 0;
      int64_t addend = 0;
      if (is_rela) {
        Elf64_Rela rela{};
        std::memcpy(&rela, image.data() + relocs.sh_offset + j * entry_size, sizeof(rela));
        info = rela.r_info;
        addend = rela.r_addend;
      } else {
        Elf64_Rel rel{};
        std::memcpy(&rel, image.data() + relocs.sh_offset + j * entry_size, sizeof(rel));
        info = rel.r_info;
      }
      const uint32_t symbol_index = elf_reloc_sym(info);
      if (symbol_index != 0) {
        if (symbols == nullptr)
          return fail("code object has a symbolic relocation without a symbol table");
        if (symbol_index >= symbols->size())
          return fail("code object has a relocation with an invalid symbol index");
        const Elf64_Sym &symbol = (*symbols)[symbol_index];
        if (symbol.st_shndx < sections.size() &&
            (sections[symbol.st_shndx].sh_flags & SHF_EXECINSTR) != 0) {
          has_executable_relocation_target = true;
        }
        continue;
      }
      if (!is_rela || elf_reloc_type(info) != R_AMDGPU_RELATIVE64 || addend < 0)
        continue;
      const uint64_t target = static_cast<uint64_t>(addend);
      if (std::ranges::any_of(executable_vaddr_ranges, [&](const auto &range) {
            return target >= range.first && target - range.first < range.second;
          })) {
        has_executable_relocation_target = true;
      }
    }
  }

  result.callable_functions.reserve(functions_by_start.size());
  for (auto &[_, function] : functions_by_start)
    result.callable_functions.push_back(std::move(function));
  std::ranges::sort(result.callable_functions, {}, &CallableFunctionRange::begin);
  std::ranges::sort(result.zero_sized_function_entries);
  result.zero_sized_function_entries.erase(
      std::ranges::unique(result.zero_sized_function_entries).begin(),
      result.zero_sized_function_entries.end());
  for (size_t i = 1; i < result.callable_functions.size(); ++i) {
    if (result.callable_functions[i].begin < result.callable_functions[i - 1].end)
      return fail("code object has overlapping STT_FUNC ranges");
  }

  const bool has_executable_section = !nonempty_executable_sections.empty();
  result.data_only = !has_executable_section && !has_nonempty_executable_segment &&
                     !has_executable_symbol && !result.has_kernel_or_descriptor_symbol &&
                     !has_executable_relocation_target;
  if (!result.data_only && !result.canonical_text_supported && result.error.empty()) {
    result.error =
        "code object executable content is not confined to one supported non-empty .text section";
  }
  result.valid = true;
  return result;
}

/// @brief Return a human-readable kernel label for diagnostics.
///
/// @details Some code objects carry empty kernel symbol names. Falling back to
/// the source .text entry offset keeps skip/failure diagnostics useful for
/// debugging because the user can still identify which code-object entry failed.
[[nodiscard]] std::string kernel_label(const KdTranslation &translation) {
  if (!translation.kernel_name.empty())
    return translation.kernel_name;

  std::ostringstream os;
  os << ".text+0x" << std::hex << translation.entry_text_offset;
  return os.str();
}

[[nodiscard]] uint32_t max_descriptor_sgpr_allocation_for_long_branch(rj_code_arch_t arch) {
  // Long direct branches consume their scratch pair at the final
  // s_setpc_b64/s_swappc_b64 transfer, so DBT may only use a pair that can be
  // made descriptor-backed for the destination kernel.
  return arch_descriptor_sgpr_allocation_limit(arch);
}

/// @brief Find the next even SGPR pair that can be descriptor-backed for a branch thunk.
[[nodiscard]] std::optional<uint16_t> next_long_branch_sgpr_pair(const TranslationContext &context,
                                                                 rj_code_arch_t arch) {
  const uint32_t current = std::max(context.num_sgprs, context.required_sgpr_count);
  const uint32_t base = (current + 1u) & ~1u;
  if (base > 126)
    return std::nullopt;

  const uint32_t max_descriptor_sgprs = max_descriptor_sgpr_allocation_for_long_branch(arch);
  if (max_descriptor_sgprs != 0 && base + 2 > max_descriptor_sgprs)
    return std::nullopt;
  return static_cast<uint16_t>(base);
}

[[nodiscard]] std::vector<uint64_t> kernel_entry_offsets(std::span<const KdTranslation> kernels) {
  std::vector<uint64_t> offsets;
  for (const KdTranslation &kernel : kernels) {
    offsets.push_back(kernel.entry_text_offset);
    for (const CallableRange &range : kernel.callable_ranges)
      offsets.push_back(range.begin);
  }

  std::ranges::sort(offsets);
  offsets.erase(std::ranges::unique(offsets).begin(), offsets.end());
  return offsets;
}

[[nodiscard]] std::vector<uint64_t>
kernel_hardware_entry_offsets(std::span<const KdTranslation> kernels) {
  std::vector<uint64_t> offsets;
  offsets.reserve(kernels.size() * 2);
  for (const KdTranslation &kernel : kernels) {
    if (kernel.descriptorless_callable)
      continue;
    offsets.push_back(kernel.entry_text_offset);
    if (kernel.has_kernarg_preload_firmware_skip)
      offsets.push_back(kernel.kernarg_preload_firmware_entry_text_offset);
  }

  std::ranges::sort(offsets);
  offsets.erase(std::ranges::unique(offsets).begin(), offsets.end());
  return offsets;
}

[[nodiscard]] std::vector<uint64_t> kernel_block_leaders(std::span<const KdTranslation> kernels,
                                                         std::span<const uint8_t> text) {
  std::vector<uint64_t> offsets;
  offsets.reserve(kernels.size() * 2);
  for (const KdTranslation &kernel : kernels) {
    offsets.push_back(kernel.entry_text_offset);
    for (const CallableRange &range : kernel.callable_ranges) {
      offsets.push_back(range.begin);
      offsets.push_back(range.end);
    }
    // AMDHSA kernarg preloading is descriptor-controlled. When
    // kernarg_preload_spec_length is non-zero, compatible CP firmware starts at
    // KERNEL_CODE_ENTRY_BYTE_OFFSET + 256. That address is a real hardware entry,
    // not merely padding, so split a block there and seed reachability from it.
    if (kernel.has_kernarg_preload_firmware_skip &&
        kernel.kernarg_preload_firmware_entry_text_offset < text.size())
      offsets.push_back(kernel.kernarg_preload_firmware_entry_text_offset);
  }

  std::ranges::sort(offsets);
  offsets.erase(std::ranges::unique(offsets).begin(), offsets.end());
  return offsets;
}

/// @brief Return real external entries, excluding boundary-only block leaders.
///
/// Function ends split blocks so sized ELF symbols stay exact, but treating
/// those ends as callable roots poisons indirect-control-flow dataflow and makes
/// its cost scale with every symbol boundary. Keep only descriptor, callable,
/// and firmware entry points in the conservative external-entry set.
[[nodiscard]] std::vector<uint64_t>
kernel_analysis_entry_offsets(std::span<const KdTranslation> kernels,
                              std::span<const uint8_t> text) {
  std::vector<uint64_t> offsets;
  offsets.reserve(kernels.size() * 2);
  for (const KdTranslation &kernel : kernels) {
    offsets.push_back(kernel.entry_text_offset);
    for (const CallableRange &range : kernel.callable_ranges)
      offsets.push_back(range.begin);
    if (kernel.has_kernarg_preload_firmware_skip &&
        kernel.kernarg_preload_firmware_entry_text_offset < text.size()) {
      offsets.push_back(kernel.kernarg_preload_firmware_entry_text_offset);
    }
  }

  std::ranges::sort(offsets);
  offsets.erase(std::ranges::unique(offsets).begin(), offsets.end());
  return offsets;
}

struct KernelTranslationScope {
  KdTranslation *translation = nullptr;
  BasicBlock *entry = nullptr;
  std::vector<BasicBlock *> blocks;
};

/// @brief Descriptor state mutated by one kernel-scope translation transaction.
struct DescriptorVariantCheckpoint {
  size_t index = 0;
  KdTranslation translation;
};

[[nodiscard]] uint64_t kernel_scope_key(const KdTranslation &kernel) {
  assert(kernel.entry_text_offset <= (std::numeric_limits<uint64_t>::max() >> 2) &&
         "kernel entry offset is too large to pack with scope bits");
  return (kernel.entry_text_offset << 2) | (kernel.descriptorless_callable ? 2u : 0u) |
         (kernel.needs_lds_overflow_buf ? 1u : 0u);
}

[[nodiscard]] bool same_kernel_scope_variant(const KdTranslation &lhs, const KdTranslation &rhs) {
  return lhs.entry_text_offset == rhs.entry_text_offset &&
         lhs.descriptorless_callable == rhs.descriptorless_callable &&
         lhs.needs_lds_overflow_buf == rhs.needs_lds_overflow_buf;
}

[[nodiscard]] std::vector<DescriptorVariantCheckpoint>
checkpoint_scope_descriptors(std::span<const KdTranslation> translations,
                             const KdTranslation &scope_translation) {
  std::vector<DescriptorVariantCheckpoint> checkpoint;
  for (size_t i = 0; i < translations.size(); ++i) {
    if (same_kernel_scope_variant(translations[i], scope_translation))
      checkpoint.push_back({.index = i, .translation = translations[i]});
  }
  return checkpoint;
}

[[nodiscard]] size_t kernel_translation_scope_count(std::span<const KdTranslation> kernels) {
  std::unordered_set<uint64_t> keys;
  for (const KdTranslation &kernel : kernels)
    keys.insert(kernel_scope_key(kernel));
  return keys.size();
}

[[nodiscard]] bool scope_uses_virtualizable_lds(const KernelTranslationScope &scope,
                                                rj_code_arch_t guest_arch,
                                                rj_code_arch_t host_arch) {
  if (scope.translation == nullptr)
    return false;
  if (scope.translation->target_lds_size != 0)
    return true;

  for (BasicBlock *block : scope.blocks) {
    if (block == nullptr)
      continue;
    for (const Instruction &inst : block->instructions()) {
      if (source_instruction_uses_virtualizable_lds(inst, guest_arch, host_arch))
        return true;
    }
  }
  return false;
}

/// @brief Sorted index from source .text byte offsets to decoded blocks.
///
/// @details DBT relocation repeatedly maps descriptor entries, branch targets,
/// and recovered indirect targets back to the BasicBlock that owns a source
/// offset. Keeping this compact sorted index avoids rebuilding that lookup while
/// preserving BasicBlock ownership in the vector returned by BasicBlock::build().
using BlockOffsetIndex = std::vector<std::pair<uint64_t, BasicBlock *>>;
using BlockPositionIndex = std::unordered_map<const BasicBlock *, size_t>;

[[nodiscard]] BlockOffsetIndex
build_block_offset_index(const std::vector<std::unique_ptr<BasicBlock>> &blocks) {
  BlockOffsetIndex index;
  index.reserve(blocks.size());
  for (const auto &block : blocks) {
    if (block != nullptr)
      index.emplace_back(block->start_offset(), block.get());
  }
  std::ranges::sort(index, {}, &std::pair<uint64_t, BasicBlock *>::first);
  return index;
}

[[nodiscard]] BlockPositionIndex
build_block_position_index(const std::vector<std::unique_ptr<BasicBlock>> &blocks) {
  BlockPositionIndex index;
  index.reserve(blocks.size());
  for (size_t i = 0; i < blocks.size(); ++i) {
    if (blocks[i] != nullptr)
      index.emplace(blocks[i].get(), i);
  }
  return index;
}

[[nodiscard]] BasicBlock *block_for_offset(const BlockOffsetIndex &index, uint64_t offset) {
  auto it = std::ranges::upper_bound(index, offset, std::less<>{},
                                     &std::pair<uint64_t, BasicBlock *>::first);
  if (it == index.begin())
    return nullptr;
  --it;

  BasicBlock *block = it->second;
  if (block == nullptr || offset >= block->end_offset())
    return nullptr;
  return block;
}

/// @brief Assemble a scope's hardware-entry offsets and run the external-entry
///        soundness gate (internal::scope_roots_are_entry_state).
///
/// @details Thin wrapper over the pure gate so translate() can pass a
/// KernelTranslationScope; the full soundness argument lives at the pure
/// function's definition below.
[[nodiscard]] bool
scope_incomplete_roots_are_entry_state(const KernelTranslationScope &scope,
                                       const std::unordered_set<uint64_t> &table_callee_offsets) {
  // Only the ordinary kernel scope entry is a safe root: hardware/ABI initializes
  // its SGPRs (dispatch pointer, kernarg pointer, workgroup ids), never a caller-
  // chosen code address.
  //
  // The kernarg-preload firmware entry (+256) is deliberately NOT a safe root.
  // Before control reaches it the command processor copies caller-controlled
  // kernarg words straight into user SGPRs (see command_processor.cpp,
  // KERNARG_PRELOAD_SPEC_LENGTH handling), so a preloaded user SGPR can hold an
  // original, unrelocated .text pointer that no in-scope builder or relocation
  // rewrites. An incomplete consumer rooted at that entry could therefore read a
  // stale code pointer, so it must fail closed.
  const std::unordered_set<uint64_t> hardware_entry_offsets{scope.entry->start_offset()};

  return internal::scope_roots_are_entry_state(scope.blocks, hardware_entry_offsets,
                                               table_callee_offsets);
}

/// @brief Prove that no stale PC-derived value can exist in one kernel scope.
///
/// @details The translator's usual model is "prove the target of every dynamic
/// transfer, or refuse". This helper establishes the complementary — and
/// strictly stronger — property: every value in this scope that was derived from
/// an `s_getpc_b64` is rewritten to hold its RELOCATED address. A consumer whose
/// dataflow fact is incomplete is then still safe, because whatever the
/// unconstrained path delivers can only be one of:
///   * a value this scope built from a getpc, which is now relocation-correct;
///   * an architectural return PC from s_call/s_swap_pc, which hardware writes
///     from the already-relocated program counter;
///   * a code address loaded from data, whose ELF relocation the code-object
///     patcher rewrites through the same final offset map (and refuses to
///     translate when it cannot).
/// No path can therefore carry an original, unrelocated `.text` address.
///
/// The proof obligation is discharged per producer and fails closed:
///   * a producer the analysis could not follow leaves an unknown value;
///   * a producer whose value is not a block start emitted by this scope cannot
///     be rewritten to a relocated address (it points at data, into the middle
///     of an instruction, or outside the emitted scope);
///   * a bare producer that is the last instruction of its block is the shape
///     whose delta add lives in a successor, so its chain is not proven closed.
/// Any of those returns nullopt, which keeps the caller's existing refusal.
///
/// The recorded value is the one the pair holds at its block's exit, so a later
/// unmodeled write inside the SAME block already leaves the producer unresolved.
/// The residual modeling assumption is that a closed chain is not extended by
/// unmodeled PC arithmetic in a SUCCESSOR block. Within one function that case
/// is refused elsewhere: the extension is a KILL transfer, a killed lattice fact
/// yields no fixup at all, and an indirect consumer with no fixup fails closed
/// as unrecovered. Escaping it would need an interprocedural chain (partial
/// build in a caller, completion after a call boundary) whose intermediate value
/// also lands exactly on an emitted block start. AMDGPU materializes a function
/// address with one indivisible getpc+add expansion, so no such chain exists.
///
/// @returns Builder rewrites that must all be applied, or nullopt when the
///          scope cannot be made free of stale PC-derived values.
[[nodiscard]] std::optional<std::vector<IndirectCallFixup>>
scope_relocatable_pc_builders(std::span<BasicBlock *const> blocks) {
  std::unordered_set<uint64_t> block_starts;
  block_starts.reserve(blocks.size());
  for (BasicBlock *block : blocks) {
    if (block == nullptr)
      return std::nullopt;
    block_starts.insert(block->start_offset());
  }

  std::vector<IndirectCallFixup> builder_fixups;
  // The instruction-start set is rebuilt per owning block rather than pooled
  // across the whole scope. patch_recovered_builder_fixups NOPs the entire
  // [begin, end) interval of a builder as one contiguous run, so that interval
  // must lie inside a single block. Discovery may add a recovered leader in a
  // later round that splits the analysis block a builder was recorded on; a
  // scope-wide instruction-start pool would still accept a range that now
  // straddles that split, and the patcher would overwrite the bytes inserted
  // between the final blocks. Bounding each builder to its owning block's
  // [start_offset, end_offset) and validating its range against only that
  // block's instruction starts fails the proof closed for any cross-block range.
  for (BasicBlock *block : blocks) {
    if (block->static_pc_address_builders().empty())
      continue;

    std::unordered_set<uint64_t> block_instruction_starts;
    block_instruction_starts.insert(block->end_offset());
    for (const Instruction &inst : block->instructions())
      block_instruction_starts.insert(inst.src_loc());

    const auto in_owning_block = [&](uint64_t offset) {
      return offset >= block->start_offset() && offset <= block->end_offset() &&
             block_instruction_starts.contains(offset);
    };

    for (const PcAddressBuilder &builder : block->static_pc_address_builders()) {
      if (!builder.resolved)
        return std::nullopt;
      // A non-contiguous range holds an unrelated instruction between builder
      // steps. patch_recovered_builder_fixups NOPs the whole range, so rewriting
      // it would erase that instruction. Fail the proof closed instead.
      if (!builder.contiguous)
        return std::nullopt;
      if (builder.source_target_offset < 0)
        return std::nullopt;
      const auto target = static_cast<uint64_t>(builder.source_target_offset);
      // patch_recovered_builder_fixups resolves the relocated target through
      // block placements, so only a block start has a defined new address.
      if (!block_starts.contains(target))
        return std::nullopt;
      // The getpc and its whole recovery range must be instruction starts inside
      // the block that owns the getpc, so the NOP-and-rewrite stays contiguous.
      if (!in_owning_block(builder.source_getpc_offset) ||
          !in_owning_block(builder.source_recovery_begin_offset) ||
          !in_owning_block(builder.source_recovery_end_offset)) {
        return std::nullopt;
      }
      if (builder.source_recovery_begin_offset == builder.source_recovery_end_offset) {
        // A bare getpc has no delta to rewrite: hardware already supplies the
        // relocated PC. Accept it only when its recorded value really is "the
        // instruction after the getpc" and at least one more instruction of the
        // same block follows without consuming it into a delta. A getpc that is
        // the last instruction of its block is the shape whose add lives in a
        // successor, where an unmodeled write would leave the original delta.
        if (target != builder.source_recovery_begin_offset)
          return std::nullopt;
        if (builder.source_recovery_end_offset >= block->end_offset())
          return std::nullopt;
        continue;
      }

      builder_fixups.push_back(
          IndirectCallFixup{.source_getpc_offset = builder.source_getpc_offset,
                            .source_recovery_begin_offset = builder.source_recovery_begin_offset,
                            .source_recovery_end_offset = builder.source_recovery_end_offset,
                            .source_call_offset = builder.source_getpc_offset,
                            .source_target_offset = target,
                            .source_call_sreg = builder.source_sreg});
    }
  }
  return builder_fixups;
}

[[nodiscard]] std::unordered_set<uint64_t>
attach_relocation_table_call_edges(const BlockOffsetIndex &block_index,
                                   std::span<const RelocationFunctionTable> tables,
                                   std::span<const RelocationTableDispatch> dispatches) {
  std::unordered_set<uint64_t> accepted_calls;
  for (const RelocationTableDispatch &dispatch : dispatches) {
    if (dispatch.table_index >= tables.size())
      continue;
    BasicBlock *source = block_for_offset(block_index, dispatch.source_call_offset);
    if (source == nullptr || source->terminator() == nullptr ||
        source->terminator()->src_loc() != dispatch.source_call_offset)
      continue;
    BasicBlock *continuation = block_for_offset(block_index, source->end_offset());
    if (continuation == nullptr || continuation->start_offset() != source->end_offset())
      continue;

    std::vector<BasicBlock *> callees;
    callees.reserve(tables[dispatch.table_index].entries.size());
    bool complete = true;
    for (const RelocationFunctionPointer &entry : tables[dispatch.table_index].entries) {
      BasicBlock *callee = block_for_offset(block_index, entry.target_text_offset);
      if (callee == nullptr || callee->start_offset() != entry.target_text_offset) {
        complete = false;
        break;
      }
      callees.push_back(callee);
    }
    if (!complete || callees.empty())
      continue;

    for (BasicBlock *callee : callees) {
      source->add_call_edge({.kind = BasicBlock::CallEdgeKind::IndirectSwapPc,
                             .callee = callee,
                             .continuation = continuation,
                             .source_call_offset = dispatch.source_call_offset,
                             .return_sreg = dispatch.return_sreg});
    }
    accepted_calls.insert(dispatch.source_call_offset);
  }
  return accepted_calls;
}

[[nodiscard]] std::vector<BasicBlock *>
reachable_kernel_blocks(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                        const BlockOffsetIndex &block_index,
                        const BlockPositionIndex &block_positions, BasicBlock &entry,
                        const std::unordered_set<uint64_t> &kernel_entries,
                        const std::unordered_set<uint64_t> &own_entries) {
  std::vector<uint8_t> reachable(blocks.size(), 0);
  std::vector<size_t> reached_indices;
  std::vector<size_t> stack;
  auto push_block = [&](BasicBlock *block) {
    auto it = block_positions.find(block);
    if (it != block_positions.end())
      stack.push_back(it->second);
  };
  push_block(&entry);
  for (const uint64_t own_entry : own_entries) {
    if (own_entry == entry.start_offset())
      continue;
    if (BasicBlock *extra_entry = block_for_offset(block_index, own_entry);
        extra_entry != nullptr && extra_entry != &entry) {
      push_block(extra_entry);
    }
  }

  while (!stack.empty()) {
    const size_t block_idx = stack.back();
    stack.pop_back();
    if (block_idx >= blocks.size() || reachable[block_idx])
      continue;
    reachable[block_idx] = 1;
    reached_indices.push_back(block_idx);
    BasicBlock *block = blocks[block_idx].get();
    assert(block != nullptr && "reachable walk stack should contain only decoded blocks");

    for (BasicBlock *succ : block->successors()) {
      assert(succ != nullptr && "BasicBlock successors should never be null");
      if (!own_entries.contains(succ->start_offset()) &&
          kernel_entries.contains(succ->start_offset()))
        continue;
      push_block(succ);
    }
    // Ordinary CFG successors describe control that always follows from the
    // current program counter: fallthroughs, conditional targets, direct branch
    // targets, and recovered non-returning setpc targets. Call edges are tracked
    // separately because a shared callee block can return to different
    // continuations depending on which call site entered it. Reachability for
    // translation still has to include the callee body, but later liveness gets
    // explicit call/return edges rather than treating every possible return as a
    // global CFG successor.
    for (const BasicBlock::CallEdge &call : block->call_edges()) {
      BasicBlock *callee = call.callee;
      assert(callee != nullptr && "BasicBlock call edges should always have a callee");
      if (!own_entries.contains(callee->start_offset()) &&
          kernel_entries.contains(callee->start_offset()))
        continue;
      push_block(callee);
    }
  }

  std::ranges::sort(reached_indices);
  std::vector<BasicBlock *> ordered;
  ordered.reserve(reached_indices.size());
  for (size_t block_idx : reached_indices) {
    if (blocks[block_idx])
      ordered.push_back(blocks[block_idx].get());
  }
  return ordered;
}

[[nodiscard]] bool valid_callable_ranges(std::span<const CallableRange> ranges) {
  if (ranges.empty())
    return false;
  for (size_t i = 0; i < ranges.size(); ++i) {
    if (ranges[i].end <= ranges[i].begin)
      return false;
    if (i != 0 && ranges[i].begin < ranges[i - 1].end)
      return false;
  }
  return true;
}

[[nodiscard]] std::vector<KernelTranslationScope>
kernel_translation_scopes(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                          const BlockOffsetIndex &block_index, std::span<KdTranslation> kernels) {
  std::vector<KernelTranslationScope> scopes;
  const auto entries = kernel_entry_offsets(kernels);
  if (entries.empty())
    return scopes;

  const BlockPositionIndex block_positions = build_block_position_index(blocks);
  const auto hardware_entries = kernel_hardware_entry_offsets(kernels);
  std::unordered_set<uint64_t> descriptor_entry_set(hardware_entries.begin(),
                                                    hardware_entries.end());
  std::vector<KdTranslation *> ordered_kernels;
  ordered_kernels.reserve(kernels.size());
  std::unordered_set<uint64_t> seen_scopes;
  for (KdTranslation &kernel : kernels) {
    if (seen_scopes.insert(kernel_scope_key(kernel)).second)
      ordered_kernels.push_back(&kernel);
  }

  std::ranges::sort(ordered_kernels, [](const auto *lhs, const auto *rhs) {
    if (lhs->entry_text_offset != rhs->entry_text_offset)
      return lhs->entry_text_offset < rhs->entry_text_offset;
    return lhs->needs_lds_overflow_buf < rhs->needs_lds_overflow_buf;
  });

  scopes.reserve(ordered_kernels.size());
  for (KdTranslation *kernel : ordered_kernels) {
    BasicBlock *entry = block_for_offset(block_index, kernel->entry_text_offset);
    if (entry == nullptr || entry->start_offset() != kernel->entry_text_offset)
      continue;
    if (kernel->descriptorless_callable) {
      if (!valid_callable_ranges(kernel->callable_ranges) ||
          kernel->callable_ranges.front().begin != kernel->entry_text_offset) {
        continue;
      }

      std::vector<BasicBlock *> callable_blocks;
      size_t range_index = 0;
      for (const auto &block : blocks) {
        if (block == nullptr)
          continue;
        while (range_index < kernel->callable_ranges.size() &&
               kernel->callable_ranges[range_index].end <= block->start_offset()) {
          ++range_index;
        }
        if (range_index == kernel->callable_ranges.size())
          break;
        const CallableRange &range = kernel->callable_ranges[range_index];
        if (block->start_offset() >= range.begin && block->end_offset() <= range.end)
          callable_blocks.push_back(block.get());
      }
      scopes.push_back({kernel, entry, std::move(callable_blocks)});
      continue;
    }

    std::unordered_set<uint64_t> own_entries{kernel->entry_text_offset};
    if (kernel->has_kernarg_preload_firmware_skip) {
      if (block_for_offset(block_index, kernel->kernarg_preload_firmware_entry_text_offset) ==
          nullptr)
        continue;
      own_entries.insert(kernel->kernarg_preload_firmware_entry_text_offset);
    }

    scopes.push_back({kernel, entry,
                      reachable_kernel_blocks(blocks, block_index, block_positions, *entry,
                                              descriptor_entry_set, own_entries)});
  }
  return scopes;
}

struct DecodedCallableFunction {
  CallableFunctionRange function;
  std::vector<BasicBlock *> blocks;
};

struct DecodedCallableFunctions {
  bool valid = false;
  std::vector<DecodedCallableFunction> functions;
  std::string error;
};

/// @brief Prove that every STT_FUNC range is a complete sequence of decoded blocks.
///
/// Function begin/end offsets are forced into the BasicBlock leader set before
/// this check. A gap or block crossing a symbol boundary therefore means that
/// final-HSACO metadata is not precise enough to relocate the callable safely.
[[nodiscard]] DecodedCallableFunctions
decode_callable_function_ranges(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                                std::span<const CallableFunctionRange> functions) {
  DecodedCallableFunctions result;
  result.functions.reserve(functions.size());
  size_t block_index = 0;
  for (const CallableFunctionRange &function : functions) {
    while (block_index < blocks.size() && (blocks[block_index] == nullptr ||
                                           blocks[block_index]->end_offset() <= function.begin)) {
      ++block_index;
    }

    uint64_t cursor = function.begin;
    std::vector<BasicBlock *> function_blocks;
    size_t scan_index = block_index;
    for (; scan_index < blocks.size(); ++scan_index) {
      const auto &block = blocks[scan_index];
      if (block == nullptr)
        continue;
      if (block->start_offset() >= function.end)
        break;
      if (block->start_offset() != cursor || block->end_offset() > function.end) {
        result.error = "STT_FUNC range does not describe a complete decoded instruction sequence";
        return result;
      }
      function_blocks.push_back(block.get());
      cursor = block->end_offset();
    }
    if (function_blocks.empty() || cursor != function.end) {
      result.error = "STT_FUNC range contains undecodable or unaccounted executable bytes";
      return result;
    }
    result.functions.push_back({function, std::move(function_blocks)});
    block_index = scan_index;
  }
  result.valid = true;
  return result;
}

[[nodiscard]] bool scope_intersects_function(const KernelTranslationScope &scope,
                                             const DecodedCallableFunction &function) {
  const auto first = std::ranges::lower_bound(scope.blocks, function.function.begin, {},
                                              &BasicBlock::start_offset);
  return first != scope.blocks.end() && *first != nullptr &&
         (*first)->start_offset() < function.function.end;
}

[[nodiscard]] bool scopes_intersect(const KernelTranslationScope &lhs,
                                    const KernelTranslationScope &rhs) {
  size_t lhs_index = 0;
  size_t rhs_index = 0;
  while (lhs_index < lhs.blocks.size() && rhs_index < rhs.blocks.size()) {
    const BasicBlock *lhs_block = lhs.blocks[lhs_index];
    const BasicBlock *rhs_block = rhs.blocks[rhs_index];
    if (lhs_block == nullptr) {
      ++lhs_index;
      continue;
    }
    if (rhs_block == nullptr) {
      ++rhs_index;
      continue;
    }
    if (lhs_block == rhs_block)
      return true;
    if (lhs_block->end_offset() <= rhs_block->start_offset())
      ++lhs_index;
    else if (rhs_block->end_offset() <= lhs_block->start_offset())
      ++rhs_index;
    else
      return true;
  }
  return false;
}

/// @brief Keep complete STT_FUNC bodies in every descriptor scope that reaches them.
///
/// Static CFG recovery can miss an address-taken internal block even though the
/// linker retained it inside a sized function symbol. Once any part of that
/// function is owned by a descriptor, its whole declared body inherits that
/// descriptor's resource contract and must move with the same kernel-local clone.
void expand_descriptor_scopes_to_complete_functions(
    std::span<KernelTranslationScope> scopes, std::span<const DecodedCallableFunction> functions) {
  for (KernelTranslationScope &scope : scopes) {
    if (scope.translation == nullptr || scope.translation->descriptorless_callable)
      continue;

    std::vector<BasicBlock *> additions;
    for (const DecodedCallableFunction &function : functions) {
      if (!scope_intersects_function(scope, function))
        continue;
      additions.insert(additions.end(), function.blocks.begin(), function.blocks.end());
    }
    if (additions.empty())
      continue;

    // Both inputs are source ordered. Merge them instead of re-sorting a
    // potentially multi-million-block kernel after every function expansion.
    std::vector<BasicBlock *> expanded;
    expanded.reserve(scope.blocks.size() + additions.size());
    std::ranges::set_union(scope.blocks, additions, std::back_inserter(expanded), {},
                           &BasicBlock::start_offset, &BasicBlock::start_offset);
    scope.blocks = std::move(expanded);
  }
}

/// @brief Conservative source VGPR allocation envelope for callable-only code.
///
/// Final linked HSACO does not retain per-function resource records. For gfx1250
/// low-bank code, however, any caller capable of executing the source function
/// must allocate the descriptor granule containing its highest referenced VGPR.
/// Restricting new dead-register scratch to that same granule preserves the
/// caller-visible resource contract. Explicit VGPR-MSB changes are rejected by
/// the caller because their physical-bank dataflow needs a retained linker
/// resource contract rather than this low-bank inference.
[[nodiscard]] std::vector<ScratchVgprRangeLimit>
descriptorless_vgpr_envelopes(const KernelTranslationScope &scope) {
  const auto &ranges = scope.translation->callable_ranges;
  if (!valid_callable_ranges(ranges))
    return {};

  const uint32_t descriptor_granule =
      descriptor_vgpr_granularity_for_wavefront(ROCJITSU_CODE_ARCH_GFX1250, 32);
  std::vector<ScratchVgprRangeLimit> envelopes;
  envelopes.reserve(ranges.size());
  size_t block_index = 0;
  for (const CallableRange &range : ranges) {
    const uint64_t begin = range.begin;
    const uint64_t end = range.end;

    while (block_index < scope.blocks.size() &&
           (scope.blocks[block_index] == nullptr ||
            scope.blocks[block_index]->end_offset() <= begin)) {
      ++block_index;
    }

    uint32_t referenced = 0;
    size_t scan_index = block_index;
    for (; scan_index < scope.blocks.size(); ++scan_index) {
      BasicBlock *block = scope.blocks[scan_index];
      if (block == nullptr)
        continue;
      if (block->start_offset() >= end)
        break;
      for (const Instruction &inst : block->instructions()) {
        if (inst.src_loc() < begin || inst.src_loc() >= end)
          continue;
        const InstDefUse def_use(inst);
        const auto account = [&](const RegisterSet &set) {
          set.for_each([&](RegisterRef ref) {
            if (ref.cls == RegClass::VGPR)
              referenced = std::max(referenced, static_cast<uint32_t>(ref.index) + 1u);
          });
        };
        account(def_use.defs);
        account(def_use.uses);
      }
    }
    block_index = scan_index;
    referenced = std::max(referenced, descriptor_granule);
    const uint32_t envelope =
        ((referenced + descriptor_granule - 1u) / descriptor_granule) * descriptor_granule;
    envelopes.push_back({.begin_offset = begin,
                         .end_offset = end,
                         .max_free_vgpr = static_cast<uint16_t>(envelope)});
  }
  return envelopes;
}

/// @brief Registers unavailable to scratch code in an LLVM AMDHSA C callable.
///
/// Final linked HSACO does not retain a function signature. The LLVM
/// C/Fast/Cold convention can return values in v0-v31 and s4-s29, so all of
/// those registers are possible external live-outs. LLVM's striped
/// callee-saved registers must also be preserved even when the original
/// function never references them and local liveness cannot see their
/// caller-owned values. Other LLVM calling conventions are outside the
/// descriptorless translation contract.
[[nodiscard]] const RegisterSet &llvm_amdhsa_callable_scratch_reserved_registers() {
  static const RegisterSet reserved = [] {
    RegisterSet value;
    value.expand({RegClass::VGPR, 0, 32});
    for (uint16_t base = 40; base <= 248; base = static_cast<uint16_t>(base + 16))
      value.expand({RegClass::VGPR, base, 8});

    value.expand({RegClass::SGPR, 0, 40});
    for (uint16_t base : {48, 64, 80})
      value.expand({RegClass::SGPR, base, 8});
    value.expand({RegClass::SGPR, 96, 10});
    return value;
  }();
  return reserved;
}

/// @brief Find return blocks inside one context-sensitive call target.
///
/// @details Call-like scalar control flow is not represented as a normal CFG
/// edge from the callee back to every possible continuation. The same helper
/// block can be entered by multiple kernels or multiple call sites, and the
/// correct continuation is the one selected by the return SGPR written at that
/// call site. This walk therefore stays inside @p allowed_blocks and mirrors
/// call-return classification: nested callees are visited with their own return
/// SGPR as a stopping condition, while their continuations retain the enclosing
/// condition. This exposes paths that return directly through an enclosing pair
/// without mistaking a nested callee's normal return for the enclosing return.
/// The caller then pairs each reported return with the specific continuation
/// from the call edge being analyzed.
[[nodiscard]] std::vector<BasicBlock *>
function_return_blocks(BasicBlock &callee, uint16_t return_sreg, std::span<const uint8_t> text,
                       const std::unordered_set<BasicBlock *> &allowed_blocks) {
  struct WalkPoint {
    BasicBlock *block = nullptr;
    std::optional<uint16_t> terminal_return_sreg;
  };

  std::vector<BasicBlock *> returns;
  std::unordered_set<BasicBlock *> return_set;
  std::vector<WalkPoint> stack{{.block = &callee, .terminal_return_sreg = std::nullopt}};
  std::set<std::pair<BasicBlock *, std::optional<uint16_t>>> visited;

  while (!stack.empty()) {
    const WalkPoint point = stack.back();
    stack.pop_back();
    BasicBlock *block = point.block;
    assert(block != nullptr && "return-block walk stack should contain only decoded blocks");
    if (!allowed_blocks.contains(block) ||
        !visited.insert({block, point.terminal_return_sreg}).second)
      continue;

    const Instruction *term = block->terminator();
    assert(term != nullptr && "decoded BasicBlock should contain at least one instruction");
    if (point.terminal_return_sreg && s_setpc_from_sreg(*term, text_word_at(text, term->src_loc()),
                                                        *point.terminal_return_sreg)) {
      continue;
    }
    if (s_setpc_from_sreg(*term, text_word_at(text, term->src_loc()), return_sreg)) {
      if (return_set.insert(block).second)
        returns.push_back(block);
      continue;
    }

    for (BasicBlock *succ : block->successors()) {
      assert(succ != nullptr && "BasicBlock successors should never be null");
      stack.push_back({.block = succ, .terminal_return_sreg = point.terminal_return_sreg});
    }
    for (const BasicBlock::CallEdge &call : block->call_edges()) {
      assert(call.callee != nullptr && "BasicBlock call edges should always have a callee");
      assert(call.continuation != nullptr &&
             "BasicBlock call edges should always have a continuation");
      stack.push_back({.block = call.callee, .terminal_return_sreg = call.return_sreg});
      stack.push_back(
          {.block = call.continuation, .terminal_return_sreg = point.terminal_return_sreg});
    }
  }

  return returns;
}

/// @brief Collect validated return-like terminators for one kernel scope.
///
/// @details Binary translation rejects unresolved indirect branches after CFG
/// construction, but a call-return `s_setpc_b64` is intentionally left as an
/// indirect instruction in the emitted code: its dynamic target is the return PC
/// saved by the matching `s_call_b64` or `s_swappc_b64`. To avoid accepting an
/// arbitrary `s_setpc_b64`, this helper only marks return offsets that are
/// reachable from a `BasicBlock::CallEdge` whose callee and continuation both
/// belong to the current kernel-local scope.
[[nodiscard]] std::unordered_set<uint64_t>
scoped_call_return_offsets(std::span<BasicBlock *const> blocks, std::span<const uint8_t> text) {
  std::unordered_set<BasicBlock *> allowed_blocks;
  allowed_blocks.reserve(blocks.size());
  for (BasicBlock *block : blocks) {
    assert(block != nullptr && "kernel scope should contain only decoded blocks");
    allowed_blocks.insert(block);
  }

  std::unordered_set<uint64_t> returns;
  for (BasicBlock *block : blocks) {
    assert(block != nullptr && "kernel scope should contain only decoded blocks");
    for (const BasicBlock::CallEdge &call : block->call_edges()) {
      assert(call.callee != nullptr && "BasicBlock call edges should always have a callee");
      assert(call.continuation != nullptr &&
             "BasicBlock call edges should always have a continuation");
      if (!allowed_blocks.contains(call.callee) || !allowed_blocks.contains(call.continuation))
        continue;

      for (BasicBlock *return_block :
           function_return_blocks(*call.callee, call.return_sreg, text, allowed_blocks)) {
        const Instruction *term = return_block->terminator();
        assert(term != nullptr && "function_return_blocks returns non-empty decoded blocks");
        returns.insert(term->src_loc());
      }
    }
  }
  return returns;
}

/// @brief Recognize ABI returns at externally entered descriptorless functions.
///
/// A callable object has no in-object call edge for a function invoked through
/// a relocated pointer. Accepting every s_setpc would be unsafe, so require an
/// exact block terminator that reads the AMDGPU ABI return-address pair
/// SGPR30:31 and is wholly contained in a complete, rooted STT_FUNC range.
/// Multiple such terminators are valid: optimized functions need not merge all
/// returns into one final epilogue.
[[nodiscard]] std::unordered_set<uint64_t>
descriptorless_callable_return_offsets(const KdTranslation &translation, KernelBlockScope blocks,
                                       std::span<const uint8_t> text) {
  std::unordered_set<uint64_t> returns;
  if (!translation.descriptorless_callable)
    return returns;

  if (!valid_callable_ranges(translation.callable_ranges))
    return returns;

  constexpr uint16_t kCallableReturnSgpr = 30;
  size_t block_index = 0;
  for (const CallableRange &range : translation.callable_ranges) {
    const uint64_t begin = range.begin;
    const uint64_t end = range.end;

    while (block_index < blocks.size() &&
           (blocks[block_index] == nullptr || blocks[block_index]->end_offset() <= begin)) {
      ++block_index;
    }
    size_t scan_index = block_index;
    for (; scan_index < blocks.size(); ++scan_index) {
      BasicBlock *block = blocks[scan_index];
      if (block == nullptr)
        continue;
      if (block->start_offset() >= end)
        break;
      if (block->start_offset() < begin || block->end_offset() > end)
        continue;
      const Instruction *term = block->terminator();
      if (term != nullptr &&
          s_setpc_from_sreg(*term, text_word_at(text, term->src_loc()), kCallableReturnSgpr)) {
        returns.insert(term->src_loc());
      }
    }
    block_index = scan_index;
  }
  return returns;
}

/// @brief Materialize context-sensitive call edges for liveness.
///
/// @details `BasicBlock` deliberately separates call edges from ordinary CFG
/// successors. The translator still needs liveness to see the effects of a
/// call: values live into the callee are used by the callee, and values live
/// after the call continuation must be live at each validated return block.
/// This helper converts each scoped call edge into temporary analysis edges
/// `caller -> callee` and `return -> continuation` without mutating the CFG or
/// creating cross-kernel return edges.
[[nodiscard]] std::vector<ScopedCfgEdge>
scoped_call_liveness_edges(std::span<BasicBlock *const> blocks, std::span<const uint8_t> text) {
  std::unordered_set<BasicBlock *> allowed_blocks;
  allowed_blocks.reserve(blocks.size());
  for (BasicBlock *block : blocks) {
    assert(block != nullptr && "kernel scope should contain only decoded blocks");
    allowed_blocks.insert(block);
  }

  std::vector<ScopedCfgEdge> edges;
  for (BasicBlock *block : blocks) {
    assert(block != nullptr && "kernel scope should contain only decoded blocks");
    for (const BasicBlock::CallEdge &call : block->call_edges()) {
      assert(call.callee != nullptr && "BasicBlock call edges should always have a callee");
      assert(call.continuation != nullptr &&
             "BasicBlock call edges should always have a continuation");
      if (!allowed_blocks.contains(call.callee) || !allowed_blocks.contains(call.continuation))
        continue;

      edges.push_back({.from = block, .to = call.callee});
      for (BasicBlock *return_block :
           function_return_blocks(*call.callee, call.return_sreg, text, allowed_blocks)) {
        edges.push_back({.from = return_block, .to = call.continuation});
      }
    }
  }

  return edges;
}

/// @brief Commit translated text, descriptor plans, and runtime metadata to one ELF image.
///
/// @details BinaryTranslator owns analysis and per-kernel lowering. This helper
/// owns the separate commit responsibility: applying completed descriptor plans,
/// replacing `.text`, appending sidecar descriptors, resolving their final virtual
/// addresses, and serializing runtime metadata. It mutates only its private patcher
/// copy, so any failure leaves the caller free to return the original code object.
[[nodiscard]] std::optional<std::vector<uint8_t>> materialize_translated_code_object(
    CodeObjectPatcher patcher, std::vector<uint8_t> translated_text, uint64_t original_text_size,
    std::span<const TextOffsetRelocation> text_relocations,
    std::span<const PcRelativeDataRelocation> data_relocations,
    std::span<const KdTranslation> translations, rj_code_arch_t host_arch, uint32_t target_mach,
    bool require_exact_function_ranges, std::vector<TranslationDiagnostic> &diagnostics) {
  if (translated_text.size() < original_text_size)
    append_nop_padding(translated_text, original_text_size - translated_text.size(), host_arch);

  std::unordered_set<uint64_t> applied_descriptors;
  for (const KdTranslation &translation : translations) {
    if (translation.descriptorless_callable || translation.sidecar_descriptor)
      continue;
    if (!applied_descriptors.insert(translation.descriptor_file_offset).second)
      continue;
    if (!patcher.apply_kernel_descriptor_translation(translation, host_arch)) {
      append_error(
          diagnostics, DiagnosticKind::KernelDescriptor,
          translation.skipped
              ? "skipped kernel descriptor could not be patched to a target stub safely; leaving "
                "code object unchanged"
              : "kernel descriptor translation could not be applied safely; leaving code object "
                "unchanged");
      return std::nullopt;
    }
  }

  std::string replace_text_error;
  std::vector<std::string> patch_warnings;
  if (!patcher.replace_text(translated_text, text_relocations, data_relocations,
                            &replace_text_error, &patch_warnings, require_exact_function_ranges)) {
    std::string message = "relocated .text could not be materialized safely";
    if (!replace_text_error.empty())
      message += ": " + replace_text_error;
    message += "; leaving code object unchanged";
    append_error(diagnostics, DiagnosticKind::ResourceLimit, std::move(message));
    return std::nullopt;
  }
  for (std::string &warning : patch_warnings)
    append_warning(diagnostics, DiagnosticKind::Legalization, std::move(warning));

  std::vector<uint64_t> sidecar_descriptor_vaddrs(translations.size(), 0);
  std::vector<KdTranslation> sidecar_descriptors;
  std::vector<size_t> sidecar_indices;
  for (size_t i = 0; i < translations.size(); ++i) {
    const KdTranslation &translation = translations[i];
    if (!translation.sidecar_descriptor || !translation.needs_lds_overflow_buf ||
        translation.skipped)
      continue;
    sidecar_descriptors.push_back(translation);
    sidecar_indices.push_back(i);
  }
  if (!sidecar_descriptors.empty()) {
    std::string sidecar_error;
    auto appended = patcher.append_sidecar_descriptor_translations(sidecar_descriptors, host_arch,
                                                                   64, &sidecar_error);
    if (!appended || appended->size() != sidecar_descriptors.size()) {
      std::string message = "virtual LDS sidecar descriptors could not be materialized safely";
      if (!sidecar_error.empty())
        message += ": " + sidecar_error;
      message += "; leaving code object unchanged";
      append_error(diagnostics, DiagnosticKind::ResourceLimit, std::move(message));
      return std::nullopt;
    }
    for (size_t i = 0; i < appended->size(); ++i)
      sidecar_descriptor_vaddrs[sidecar_indices[i]] = (*appended)[i].vaddr;
  }

  const auto patched_image = patcher.image_bytes();
  AmdGpuCodeObject patched_layout(patched_image.data(), patched_image.size());
  if (!patched_layout.is_valid()) {
    append_error(diagnostics, DiagnosticKind::ResourceLimit,
                 "relocated ELF could not be reparsed for runtime metadata; leaving code object "
                 "unchanged");
    return std::nullopt;
  }

  std::vector<SidecarVariantMetadata> sidecar_metadata;
  std::vector<KernargExtensionMetadata> kernarg_extension_metadata;
  std::vector<VirtualLdsKernelMetadata> virtual_lds_metadata;
  for (size_t i = 0; i < translations.size(); ++i) {
    const KdTranslation &translation = translations[i];
    if (!translation.sidecar_descriptor || !translation.needs_lds_overflow_buf ||
        translation.skipped) {
      continue;
    }
    const auto normal_translation =
        std::ranges::find_if(translations, [&](const KdTranslation &candidate) {
          return !candidate.sidecar_descriptor && !candidate.skipped &&
                 candidate.kernel_name == translation.kernel_name;
        });
    if (normal_translation == translations.end()) {
      append_error(diagnostics, DiagnosticKind::KernelDescriptor,
                   "virtual LDS metadata could not find the normal descriptor translation; "
                   "leaving code object unchanged");
      return std::nullopt;
    }
    const uint64_t descriptor_vaddr =
        patched_layout.kernel_descriptor_offset(translation.kernel_name);
    if (descriptor_vaddr == 0) {
      append_error(diagnostics, DiagnosticKind::KernelDescriptor,
                   "virtual LDS metadata could not find the translated kernel descriptor symbol; "
                   "leaving code object unchanged");
      return std::nullopt;
    }

    const uint64_t sidecar_descriptor_vaddr = sidecar_descriptor_vaddrs[i];
    if (sidecar_descriptor_vaddr == 0) {
      append_error(diagnostics, DiagnosticKind::KernelDescriptor,
                   "virtual LDS metadata could not find the appended sidecar descriptor; leaving "
                   "code object unchanged");
      return std::nullopt;
    }

    // Sidecar identity, kernarg extension layout, and virtual-LDS policy are
    // serialized independently. Their stable kernel/variant names are the join
    // key; no generic mechanism embeds another feature's fields.
    sidecar_metadata.push_back(SidecarVariantMetadata{
        .kernel_name = translation.kernel_name,
        .variant_name = std::string(kVirtualLdsSidecarVariantName),
        .normal_descriptor_vaddr = descriptor_vaddr,
        .variant_descriptor_vaddr = sidecar_descriptor_vaddr,
    });

    KernargExtensionMetadata kernarg_extension{
        .kernel_name = translation.kernel_name,
        .variant_name = std::string(kVirtualLdsSidecarVariantName),
        .original_kernarg_size = translation.kernarg_size,
        .payloads = {{
            .size = kVirtualLdsRuntimeStateBytes,
            .alignment = alignof(uint64_t),
            .name = std::string(kVirtualLdsRuntimeStatePayloadName),
        }},
    };
    const KernargExtensionPayloadLayout payload_layout{
        .size = kernarg_extension.payloads.front().size,
        .alignment = kernarg_extension.payloads.front().alignment,
    };
    const auto wrapper_layout = make_kernarg_extension_layout(
        kernarg_extension.original_kernarg_size, std::span{&payload_layout, 1});
    if (!wrapper_layout || wrapper_layout->payload_offsets.empty() ||
        wrapper_layout->payload_offsets.front() !=
            translation.lds_overflow_kernarg_pointer_offset) {
      append_error(diagnostics, DiagnosticKind::KernelDescriptor,
                   "virtual LDS kernarg extension layout disagrees with the translated entry "
                   "prologue; leaving code object unchanged");
      return std::nullopt;
    }
    kernarg_extension_metadata.push_back(std::move(kernarg_extension));

    VirtualLdsKernelMetadata record{};
    record.kernel_name = translation.kernel_name;
    record.sidecar_variant_name = std::string(kVirtualLdsSidecarVariantName);
    record.static_lds_bytes = translation.lds_overflow_size;
    // AQL private_segment_size can include a dynamic call-stack request above
    // the normal descriptor's fixed allocation. Dispatch rewriting needs both
    // fixed sizes to preserve that dynamic portion while switching variants;
    // neither loaded descriptor address is guaranteed to be CPU-readable.
    record.normal_private_segment_size = normal_translation->target_private_size;
    record.virtual_private_segment_size = translation.target_private_size;
    record.virtual_lds_base_sgpr = translation.virtual_lds_lowering.base_sgpr;
    record.flags |= kVirtualLdsFlagRuntimeStateBlock;
    if (translation.workgroup_id_sgpr_x >= 0)
      record.flags |= kVirtualLdsFlagWorkgroupIdX;
    if (translation.workgroup_id_sgpr_y >= 0)
      record.flags |= kVirtualLdsFlagWorkgroupIdY;
    if (translation.workgroup_id_sgpr_z >= 0)
      record.flags |= kVirtualLdsFlagWorkgroupIdZ;
    virtual_lds_metadata.push_back(std::move(record));
  }

  if (!sidecar_metadata.empty()) {
    const auto metadata_bytes = serialize_sidecar_metadata(sidecar_metadata);
    if (metadata_bytes.empty() ||
        !patcher.append_nonalloc_section(kSidecarMetadataSectionName, metadata_bytes, 8)) {
      append_error(diagnostics, DiagnosticKind::ResourceLimit,
                   "sidecar metadata could not be materialized safely; leaving code object "
                   "unchanged");
      return std::nullopt;
    }
  }

  if (!kernarg_extension_metadata.empty()) {
    const auto metadata_bytes = serialize_kernarg_extension_metadata(kernarg_extension_metadata);
    if (metadata_bytes.empty() ||
        !patcher.append_nonalloc_section(kKernargExtensionMetadataSectionName, metadata_bytes, 8)) {
      append_error(diagnostics, DiagnosticKind::ResourceLimit,
                   "kernarg extension metadata could not be materialized safely; leaving code "
                   "object unchanged");
      return std::nullopt;
    }
  }

  if (!virtual_lds_metadata.empty()) {
    const auto metadata_bytes = serialize_virtual_lds_metadata(virtual_lds_metadata);
    if (metadata_bytes.empty() ||
        !patcher.append_nonalloc_section(kVirtualLdsMetadataSectionName, metadata_bytes, 8)) {
      append_error(diagnostics, DiagnosticKind::ResourceLimit,
                   "virtual LDS metadata could not be materialized safely; leaving code object "
                   "unchanged");
      return std::nullopt;
    }
  }

  if (target_mach)
    patcher.update_elf_flags(target_mach);
  return std::move(patcher).emit();
}

} // namespace

namespace internal {

/// @brief Prove that every external entry into an incomplete-consumer scope is
///        an entry-state root that cannot carry an original `.text` pointer.
///
/// @details scope_relocatable_pc_builders proves every getpc-derived value in a
/// scope is relocated, but that proof only covers values this scope PRODUCES. An
/// incomplete consumer is also reachable along a path that enters the scope
/// carrying an SGPR value from OUTSIDE it. The whole-scope proof is sound only
/// when every such external entry is a root whose incoming SGPRs are
/// architecturally defined, never a raw original code address:
///   * a hardware kernel entry passed in @p hardware_entry_offsets — the caller
///     supplies only entries whose live-in SGPRs are ABI-initialized (dispatch
///     pointer, kernarg pointer, workgroup ids). The kernarg-preload firmware
///     entry is deliberately excluded there, because caller-controlled kernarg
///     words are copied into user SGPRs before it runs;
///   * a getpc-recovered in-scope call target (callee of a proven direct or
///     swappc call edge) — it is entered only through that call, so its live-in
///     PC pair is the architected return PC hardware wrote from the already-
///     relocated program counter, or a value the caller built in-scope from a
///     getpc (now relocated).
///
/// A relocation-table-dispatched callee is NOT such a safe root, even though it
/// has an in-scope CallEdge: the dispatch selects a callee dynamically and its
/// live-in scalar registers are arbitrary caller-supplied arguments, which can
/// include an original, unrelocated `.text` pointer. A call edge constrains
/// control flow, not the SGPR arguments delivered along it, so a table-dispatched
/// callee that itself holds an incomplete consumer could still receive a stale
/// code pointer on one path. Those callees are treated as unconstrained roots.
///
/// A block reachable within the scope has an in-scope predecessor (an ordinary
/// CFG edge) or is a non-table call-edge callee; any other block — one with no
/// in-scope predecessor and no proven getpc-recovered call edge — is entered from
/// outside the scope. Such an external-entry block is an unconstrained root:
/// control can arrive there holding a caller-supplied function pointer that is an
/// original, unrelocated `.text` address. The producer scan cannot rewrite that
/// value, so the incomplete consumer downstream could jump to stale bytes. This
/// gate fails closed for such a scope, which keeps the caller's original refusal.
/// Empirically every incomplete-consumer scope in the gfx1250 hotswap corpus
/// roots only at the kernel entry and at getpc-recovered call targets.
bool scope_roots_are_entry_state(std::span<BasicBlock *const> blocks,
                                 const std::unordered_set<uint64_t> &hardware_entry_offsets,
                                 const std::unordered_set<uint64_t> &table_callee_offsets) {
  std::unordered_set<const BasicBlock *> in_scope(blocks.begin(), blocks.end());
  std::unordered_set<const BasicBlock *> call_targets;
  for (const BasicBlock *block : blocks) {
    if (block == nullptr)
      return false;
    for (const BasicBlock::CallEdge &edge : block->call_edges()) {
      if (in_scope.contains(edge.callee))
        call_targets.insert(edge.callee);
    }
  }

  for (const BasicBlock *block : blocks) {
    const bool has_in_scope_predecessor = std::ranges::any_of(
        block->predecessors(), [&](const BasicBlock *pred) { return in_scope.contains(pred); });
    if (has_in_scope_predecessor)
      continue;
    // A block with no in-scope predecessor is an external entry (it has no
    // ordinary CFG edge from within the scope, even if it is reached by a call
    // edge or has predecessors outside the scope).
    //
    // A relocation-table-dispatched callee delivers unconstrained caller-supplied
    // SGPR arguments, so it is never a safe root regardless of its CallEdge; fail
    // closed for it even if it is also a getpc-recovered call target.
    if (table_callee_offsets.contains(block->start_offset()))
      return false;
    // Otherwise accept a hardware kernel entry or a getpc-recovered in-scope call
    // target; any other root is an unconstrained external entry that may deliver a
    // stale code pointer.
    if (hardware_entry_offsets.contains(block->start_offset()) || call_targets.contains(block))
      continue;
    return false;
  }
  return true;
}

} // namespace internal

BinaryTranslator::~BinaryTranslator() = default;

BinaryTranslator::BinaryTranslator(rj_code_arch_t guest_arch, rj_code_arch_t host_arch,
                                   uint32_t target_mach, BinaryTranslatorOptions options)
    : guest_arch_(guest_arch), host_arch_(host_arch),
      target_mach_(target_mach ? target_mach : elf_mach_for_arch(host_arch)), options_(options),
      encoding_translate_(select_encoding_translator(guest_arch, host_arch)),
      legalization_lookup_(select_legalization(guest_arch, host_arch)),
      semantic_translator_(std::make_unique<SemanticTranslator>(
          guest_arch, host_arch, options.input_revision, options.output_revision)) {}

const InstructionLegalization *
BinaryTranslator::lookup_legalization(const Instruction &inst) const {
  // gfx1250 B0 and A0 have the same structural ISA, so the generated cross-ISA
  // tables cannot express their revision-specific behavior. Instructions in
  // the B0-to-A0 profile use handwritten legalization; everything else follows
  // the raw same-ISA copy path.
  if (guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 && host_arch_ == ROCJITSU_CODE_ARCH_GFX1250 &&
      options_.input_revision == ProcessorRevision::Gfx1250B0 &&
      options_.output_revision == ProcessorRevision::Gfx1250A0)
    return gfx1250_b0_to_a0_legalization(inst);

  return legalization_lookup_ ? legalization_lookup_(inst.encoding_id(), inst.opcode()) : nullptr;
}

void BinaryTranslator::set_trace_callback(TranslationTraceCallback callback) {
  trace_callback_ = std::move(callback);
}

TranslatedCodeObject BinaryTranslator::translate(const AmdGpuCodeObject &obj) {
  TranslatedCodeObject result;
  result.host_arch = host_arch_;

  auto leave_unchanged = [&]() {
    const auto *image = reinterpret_cast<const uint8_t *>(obj.image_data());
    if (obj.image_size() != 0)
      result.elf_bytes.assign(image, image + obj.image_size());
    return result;
  };

  if (obj.image_size() < sizeof(Elf64_Ehdr)) {
    append_error(result.diagnostics, DiagnosticKind::ResourceLimit,
                 "code object is too small to contain an ELF header");
    return leave_unchanged();
  }

  CodeObjectPatcher patcher(obj);

  // A same-architecture gfx1250 translation is direction-specific: A0 and B0
  // share an ELF machine ID, so both revisions must be given. Enforce this here
  // as well as in the C API.
  if (guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 && host_arch_ == ROCJITSU_CODE_ARCH_GFX1250) {
    if (options_.input_revision == ProcessorRevision::Unspecified ||
        options_.output_revision == ProcessorRevision::Unspecified) {
      append_error(result.diagnostics, DiagnosticKind::Legalization,
                   "gfx1250 same-target translation requires both input and output silicon "
                   "revisions");
      return leave_unchanged();
    }
    // Only the B0-to-A0 direction is implemented.
    if (options_.input_revision == ProcessorRevision::Gfx1250A0 &&
        options_.output_revision == ProcessorRevision::Gfx1250B0) {
      append_error(result.diagnostics, DiagnosticKind::Legalization,
                   "gfx1250 A0-to-B0 translation is not supported");
      return leave_unchanged();
    }
  }

  const auto image = std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(obj.image_data()),
                                              obj.image_size());
  const ExecutableElfAnalysis executable =
      analyze_executable_elf(image, patcher.text_offset(), patcher.text_size());
  if (!executable.valid) {
    append_error(result.diagnostics, DiagnosticKind::ResourceLimit,
                 executable.error.empty() ? "code object executable layout is malformed"
                                          : executable.error);
    return leave_unchanged();
  }

  Elf64_Ehdr header{};
  std::memcpy(&header, obj.image_data(), sizeof(header));
  const uint32_t source_mach = header.e_flags & EF_AMDGPU_MACH;
  if (executable.data_only) {
    if (guest_arch_ == host_arch_ && source_mach == (target_mach_ & EF_AMDGPU_MACH)) {
      append_warning(result.diagnostics, DiagnosticKind::DataOnly,
                     "code object has no executable sections, segments, or callable symbols; "
                     "leaving unchanged");
      return leave_unchanged();
    }
    append_error(result.diagnostics, DiagnosticKind::ResourceLimit,
                 "data-only code object cannot be translated when input and output targets differ");
    return leave_unchanged();
  }
  if (!executable.canonical_text_supported) {
    append_error(result.diagnostics, DiagnosticKind::ResourceLimit,
                 executable.error.empty()
                     ? "code object executable content is not confined to one supported non-empty "
                       ".text section"
                     : executable.error);
    return leave_unchanged();
  }

  auto text = patcher.text_bytes();
  assert(!text.empty() && "supported executable layout must have non-empty .text");

  // DBT relocates instructions within .text (compaction, expansion, per-kernel
  // block placement) but does not rewrite relocation places that land inside
  // .text. An in-.text relocation would therefore be applied to the wrong
  // translated bytes. Fail closed rather than silently miscompile.
  if (patcher.has_relocations_within_text()) {
    append_error(result.diagnostics, DiagnosticKind::Legalization,
                 "code object has a relocation place inside .text; relocating instructions would "
                 "apply it to the wrong bytes and is not supported");
    return leave_unchanged();
  }

  // The patcher can retarget ordinary zero-addend symbol references and
  // symbol-less RELATIVE64 addends through the final source-to-target offset
  // map. Keep less explicit forms fail-closed until their relocation-specific
  // addend semantics are modeled.
  if (patcher.has_unsupported_relocation_to_text()) {
    append_error(result.diagnostics, DiagnosticKind::Legalization,
                 "code object has an unsupported relocation referencing .text; section symbols, "
                 "implicit addends, and named-symbol addends cannot be remapped safely");
    return leave_unchanged();
  }

  // Per-kernel text relocation strategy:
  // 1. Translate descriptors first so their source entries and ABI state define
  //    the normal and sidecar kernel scopes.
  // 2. Decode .text, recover bounded static indirect targets, and form each
  //    scope from ordinary CFG successors plus validated call edges.
  // 3. Emit each scope into a source-ordered local body. Semantic expansions
  //    grow that body and control transfers reserve explicit patch windows.
  // 4. Place entry stubs and bodies in final .text coordinates, then repair
  //    direct transfers, recovered indirect transfers, and their PC builders.
  // 5. Feed discovered register/private-memory requirements back into each
  //    descriptor and commit .text, descriptors, sidecars, metadata, and flags.
  auto decoder = Decoder::create(guest_arch_);
  if (!decoder) {
    append_error(result.diagnostics, DiagnosticKind::UnsupportedGuestArch,
                 "unsupported guest_arch: no decoder available");
    return leave_unchanged();
  }

  // Phase 1: descriptor translation gives DBT the source kernel roots and any
  // target descriptor/prologue bytes that must be materialized with the body.
  const bool skip_failed_kernels = options_.skip_failed_kernels;
  KernelDescriptorTranslator descriptor_translator(guest_arch_, host_arch_);
  const bool can_emit_sidecar_descriptors = supports_virtual_lds_sidecars(guest_arch_, host_arch_);
  KernelDescriptorTranslationOptions initial_descriptor_options;
  initial_descriptor_options.allow_oversized_lds = can_emit_sidecar_descriptors;
  auto descriptor_translations =
      descriptor_translator.translate_image(patcher.image_bytes(), patcher.text_offset(),
                                            patcher.text_size(), initial_descriptor_options);
  bool descriptors_supported = true;
  for (const auto &translation : descriptor_translations) {
    if (translation.supported || !skip_failed_kernels)
      append_diagnostics(result.diagnostics, translation.diagnostics);
    descriptors_supported &= translation.supported;
  }
  if (!descriptors_supported && !skip_failed_kernels) {
    append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                 "kernel descriptor translation requires unsupported resource or ABI "
                 "virtualization; leaving code object unchanged");
    return leave_unchanged();
  }

  const bool is_gfx1250_b0_to_a0 = guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 &&
                                   host_arch_ == ROCJITSU_CODE_ARCH_GFX1250 &&
                                   options_.input_revision == ProcessorRevision::Gfx1250B0 &&
                                   options_.output_revision == ProcessorRevision::Gfx1250A0;
  const bool has_kernel_descriptors = !descriptor_translations.empty();
  std::unordered_set<uint64_t> descriptor_entries;
  for (const KdTranslation &translation : descriptor_translations)
    descriptor_entries.insert(translation.entry_text_offset);
  if (is_gfx1250_b0_to_a0) {
    const auto uncorroborated_zero_sized_function =
        std::ranges::find_if(executable.zero_sized_function_entries,
                             [&](uint64_t entry) { return !descriptor_entries.contains(entry); });
    if (uncorroborated_zero_sized_function != executable.zero_sized_function_entries.end()) {
      append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                   "zero-sized STT_FUNC entry is not corroborated by a kernel descriptor",
                   *uncorroborated_zero_sized_function);
      return leave_unchanged();
    }
  }

  if (!has_kernel_descriptors) {
    if (executable.callable_functions.empty()) {
      append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                   "executable .text has neither kernel descriptors nor complete STT_FUNC ranges");
      return leave_unchanged();
    }
    if (!is_gfx1250_b0_to_a0) {
      append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                   "descriptorless callable translation is only supported for gfx1250 B0-to-A0");
      return leave_unchanged();
    }
  }

  const auto relocation_function_tables = discover_relocation_function_tables(obj);
  auto block_leaders = kernel_block_leaders(descriptor_translations, text);
  auto analysis_entries = kernel_analysis_entry_offsets(descriptor_translations, text);
  if (is_gfx1250_b0_to_a0) {
    for (const CallableFunctionRange &function : executable.callable_functions) {
      block_leaders.push_back(function.begin);
      block_leaders.push_back(function.end);
      analysis_entries.push_back(function.begin);
    }
  }
  for (const RelocationFunctionTable &table : relocation_function_tables) {
    for (const RelocationFunctionPointer &entry : table.entries) {
      block_leaders.push_back(entry.target_text_offset);
      analysis_entries.push_back(entry.target_text_offset);
    }
  }
  std::ranges::sort(block_leaders);
  block_leaders.erase(std::ranges::unique(block_leaders).begin(), block_leaders.end());
  std::ranges::sort(analysis_entries);
  analysis_entries.erase(std::ranges::unique(analysis_entries).begin(), analysis_entries.end());

  // Phase 2: build a CFG over .text, including recovered indirect targets as
  // block leaders, then compute one source-reachable block set per descriptor
  // root. These sets are intentionally kernel-local: if two roots reach the same
  // helper block, Phase 3 emits that helper into both relocated bodies so every
  // branch or call target can be resolved through the current kernel's placement
  // map without borrowing another kernel's return continuation.
  std::vector<std::unique_ptr<BasicBlock>> blocks;
  try {
    blocks = BasicBlock::build(obj, *decoder, guest_arch_, block_leaders, analysis_entries,
                               ExternalEntryPolicy::ExplicitOnly);
  } catch (const util::InvalidInst &error) {
    append_error(result.diagnostics, DiagnosticKind::Legalization,
                 std::string("failed to decode executable .text: ") + error.what());
    return leave_unchanged();
  }
  const BlockOffsetIndex block_index = build_block_offset_index(blocks);
  const uint64_t text_vaddr = obj.text_sections().front()->vaddr();
  const auto relocation_table_dispatches =
      discover_relocation_table_dispatches(blocks, relocation_function_tables, text_vaddr);
  const auto relocation_table_calls = attach_relocation_table_call_edges(
      block_index, relocation_function_tables, relocation_table_dispatches);

  // Callees reached through a relocation-table dispatch are explicit analysis
  // roots whose live-in SGPRs are caller-supplied, not architected: a dispatched
  // callee can be entered with an original .text pointer in a scalar argument.
  // The whole-scope stale-PC proof must therefore treat such a callee as an
  // unconstrained external entry rather than a safe entry-state root, even though
  // it has an in-scope CallEdge. Collect their block-start offsets so the gate
  // can fail closed for them (see scope_incomplete_roots_are_entry_state).
  std::unordered_set<uint64_t> relocation_table_callee_offsets;
  for (const RelocationTableDispatch &dispatch : relocation_table_dispatches) {
    if (dispatch.table_index >= relocation_function_tables.size())
      continue;
    if (!relocation_table_calls.contains(dispatch.source_call_offset))
      continue;
    for (const RelocationFunctionPointer &entry :
         relocation_function_tables[dispatch.table_index].entries)
      relocation_table_callee_offsets.insert(entry.target_text_offset);
  }

  if (!has_kernel_descriptors) {
    const bool needs_instruction_rewrite = std::ranges::any_of(blocks, [&](const auto &block) {
      if (block == nullptr)
        return false;
      return std::ranges::any_of(block->instructions(), [&](const Instruction &inst) {
        return lookup_legalization(inst) != nullptr ||
               (semantic_translator_ != nullptr && semantic_translator_->has_expand_rule(inst));
      });
    });

    // A descriptorless callable library whose decoded instructions have no B0
    // rewrite candidates needs no address, symbol, unwind, or resource rewrite.
    // Scan the entire decoded .text before taking this path; the CLI still
    // performs normal host-decode validation for this executable no-op.
    if (!needs_instruction_rewrite) {
      append_warning(result.diagnostics, DiagnosticKind::NothingToTranslate,
                     "descriptorless executable functions contain no instructions that require "
                     "rewriting; leaving unchanged");
      return leave_unchanged();
    }
  }

  DecodedCallableFunctions decoded_callables;
  decoded_callables.valid = true;
  if (is_gfx1250_b0_to_a0) {
    decoded_callables = decode_callable_function_ranges(blocks, executable.callable_functions);
    if (!decoded_callables.valid) {
      append_error(result.diagnostics, DiagnosticKind::Legalization,
                   decoded_callables.error + "; leaving code object unchanged");
      return leave_unchanged();
    }
  }

  auto scopes = kernel_translation_scopes(blocks, block_index, descriptor_translations);
  if (is_gfx1250_b0_to_a0)
    expand_descriptor_scopes_to_complete_functions(scopes, decoded_callables.functions);

  if (can_emit_sidecar_descriptors) {
    std::vector<KdTranslation> sidecar_variants;
    for (const KernelTranslationScope &scope : scopes) {
      if (scope.translation == nullptr || scope.translation->descriptorless_callable)
        continue;
      const uint32_t host_lds_bytes = arch_lds_bytes(host_arch_);
      const bool static_lds_exceeds_host =
          host_lds_bytes != 0 && scope.translation->target_lds_size > host_lds_bytes;
      // Dynamic LDS is only known at dispatch time, so every LDS-using kernel
      // needs a virtual sidecar. The sidecar descriptor owns the wrapper ABI
      // and may enable a target-only kernarg segment pointer when the source
      // descriptor left room in the 16 initialized User SGPRs.
      if (!static_lds_exceeds_host &&
          !scope_uses_virtualizable_lds(scope, guest_arch_, host_arch_)) {
        continue;
      }

      KernelDescriptorTranslationOptions virtual_descriptor_options;
      virtual_descriptor_options.virtualize_lds = true;
      auto virtual_translation = descriptor_translator.translate_descriptor(
          patcher.image_bytes(), scope.translation->descriptor_file_offset,
          scope.translation->entry_text_offset, virtual_descriptor_options);
      if (!virtual_translation) {
        append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                     "virtual LDS sidecar descriptor could not be computed; leaving code object "
                     "unchanged",
                     scope.translation->entry_text_offset);
        return leave_unchanged();
      }
      virtual_translation->kernel_name = scope.translation->kernel_name;
      virtual_translation->sidecar_descriptor = true;
      sidecar_variants.push_back(std::move(*virtual_translation));
    }
    if (!sidecar_variants.empty()) {
      descriptor_translations.insert(descriptor_translations.end(),
                                     std::make_move_iterator(sidecar_variants.begin()),
                                     std::make_move_iterator(sidecar_variants.end()));
      scopes = kernel_translation_scopes(blocks, block_index, descriptor_translations);
      if (is_gfx1250_b0_to_a0)
        expand_descriptor_scopes_to_complete_functions(scopes, decoded_callables.functions);
    }
  }

  // Any complete STT_FUNC body not owned by a descriptor scope remains
  // independently callable in the linked image. Translate all such functions
  // together under the conservative descriptorless ABI/resource contract,
  // including in mixed kernel-plus-library code objects.
  std::vector<CallableFunctionRange> independent_callables;
  if (is_gfx1250_b0_to_a0) {
    for (const DecodedCallableFunction &function : decoded_callables.functions) {
      const bool descriptor_owned = std::ranges::any_of(scopes, [&](const auto &scope) {
        return scope.translation != nullptr && !scope.translation->descriptorless_callable &&
               scope_intersects_function(scope, function);
      });
      if (descriptor_owned && function.function.externally_visible() &&
          !descriptor_entries.contains(function.function.begin)) {
        append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                     "externally visible STT_FUNC is reachable from a kernel descriptor and "
                     "cannot safely inherit that kernel's private resource contract",
                     function.function.begin);
        return leave_unchanged();
      }
      if (!descriptor_owned && !descriptor_entries.contains(function.function.begin))
        independent_callables.push_back(function.function);
    }
  }

  if (!independent_callables.empty()) {
    if (!is_gfx1250_b0_to_a0) {
      append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                   "independent STT_FUNC callable translation is only supported for gfx1250 "
                   "B0-to-A0");
      return leave_unchanged();
    }
    if (!executable.assume_llvm_amdhsa_callable_abi) {
      append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                   "descriptorless STT_FUNC translation supports only linked LLVM AMDHSA code "
                   "objects V4-V6");
      return leave_unchanged();
    }

    KdTranslation callable;
    callable.descriptorless_callable = true;
    callable.kernel_name = "descriptorless callable functions";
    callable.entry_text_offset = independent_callables.front().begin;
    callable.callable_ranges.reserve(independent_callables.size());
    for (const CallableFunctionRange &function : independent_callables)
      callable.callable_ranges.push_back({function.begin, function.end});
    assert(valid_callable_ranges(callable.callable_ranges) &&
           callable.callable_ranges.front().begin == callable.entry_text_offset);
    // gfx1250 allocates ordinary VGPRs in 16-register descriptor granules. The
    // exact source envelope is tightened after decode; starting at the minimum
    // granule prevents an empty/argument-free callable from manufacturing a
    // zero-register resource contract.
    callable.target_vgpr_count = 16;
    callable.target_vgpr_allocation_count = 16;
    callable.target_sgpr_count = 106;
    callable.target_wave_size = 32;
    callable.guest_wavefront_size = 32;
    callable.host_wavefront_size = 32;
    descriptor_translations.push_back(std::move(callable));

    scopes = kernel_translation_scopes(blocks, block_index, descriptor_translations);
    expand_descriptor_scopes_to_complete_functions(scopes, decoded_callables.functions);
  }

  if (descriptor_translations.empty()) {
    append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                 "executable .text has no translatable descriptor or STT_FUNC scope");
    return leave_unchanged();
  }

  const size_t expected_scope_count = kernel_translation_scope_count(descriptor_translations);
  if (scopes.size() != expected_scope_count) {
    append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                 "kernel descriptor entry offsets are required to map to decoded text blocks");
    return leave_unchanged();
  }

  // A callable-only object has no descriptor roots that define which code may
  // be discarded, so every decoded block must belong to a complete STT_FUNC
  // range. Descriptor-backed objects may still compact genuinely unreachable,
  // unreferenced blocks: replace_text() removes those bytes rather than leaving
  // stale B0 instructions executable, while every retained STT_FUNC range is
  // proven and owned above.
  if (!has_kernel_descriptors) {
    std::unordered_set<const BasicBlock *> covered_blocks;
    for (const KernelTranslationScope &scope : scopes) {
      for (const BasicBlock *block : scope.blocks) {
        if (block != nullptr)
          covered_blocks.insert(block);
      }
    }
    const auto uncovered = std::ranges::find_if(blocks, [&](const auto &block) {
      return block != nullptr && !covered_blocks.contains(block.get());
    });
    if (uncovered != blocks.end()) {
      append_error(result.diagnostics, DiagnosticKind::Legalization,
                   "descriptorless executable .text contains decoded instructions outside "
                   "complete STT_FUNC ranges",
                   (*uncovered)->start_offset());
      return leave_unchanged();
    }
  }

  // No descriptor scope may silently absorb an independently callable range.
  // Such overlap would give one source address two target placements while ELF
  // symbols, RELATIVE64 addends, and unwind FDEs can name only one. Cross-scope
  // callable transfers need a dedicated global fixup before that layout is safe.
  const auto descriptorless_scope =
      std::ranges::find_if(scopes, [](const KernelTranslationScope &scope) {
        return scope.translation != nullptr && scope.translation->descriptorless_callable;
      });
  if (descriptorless_scope != scopes.end()) {
    for (const KernelTranslationScope &scope : scopes) {
      if (scope.translation == nullptr || scope.translation->descriptorless_callable)
        continue;
      const bool overlaps = scopes_intersect(scope, *descriptorless_scope);
      // Independent-callable discovery excludes every function intersecting a
      // descriptor scope using the same block identity, so overlap is an
      // internal construction invariant today. Keep the release-mode check
      // below as a fail-closed backstop for future scope-building changes.
      assert(!overlaps && "independent callable scope must be disjoint from descriptor scopes");
      if (overlaps) {
        append_error(result.diagnostics, DiagnosticKind::Legalization,
                     "independently callable STT_FUNC code overlaps a kernel-local translation "
                     "scope");
        return leave_unchanged();
      }
    }
  }

  std::vector<uint8_t> translated_text;
  translated_text.reserve(text.size());
  const bool continue_after_failure = options_.debug_continue_after_failure;

  struct PendingTrace {
    uint64_t source_offset = 0;
    uint32_t source_size = 0;
    std::vector<uint32_t> source_words;
    const InstructionLegalization *legalization = nullptr;
    bool copied_original = false;
    bool semantic_lowering = false;
    bool changed = false;
    uint64_t target_offset = 0;
    std::vector<uint32_t> target_words;
  };

  auto queue_trace = [&](std::vector<PendingTrace> &pending, const Instruction &inst,
                         uint64_t offset, const InstructionLegalization *leg, bool copied_original,
                         bool semantic_lowering, bool changed, uint64_t target_offset,
                         std::vector<uint32_t> target_words) {
    if (!trace_callback_)
      return;
    pending.push_back({.source_offset = offset,
                       .source_size = static_cast<uint32_t>(inst.size()),
                       .source_words = raw_words_for_inst(inst),
                       .legalization = leg,
                       .copied_original = copied_original,
                       .semantic_lowering = semantic_lowering,
                       .changed = changed,
                       .target_offset = target_offset,
                       .target_words = std::move(target_words)});
  };

  auto flush_traces = [&](std::vector<PendingTrace> &pending, uint64_t target_delta) {
    if (!trace_callback_)
      return;
    for (PendingTrace &trace : pending) {
      trace_callback_({.source_offset = trace.source_offset,
                       .source_size = trace.source_size,
                       .source_words = trace.source_words,
                       .legalization = trace.legalization,
                       .copied_original = trace.copied_original,
                       .semantic_lowering = trace.semantic_lowering,
                       .changed = trace.changed,
                       .emitted_in_cave = false,
                       .target_offset = trace.target_offset + target_delta,
                       .target_words = trace.target_words});
    }
  };

  auto copy_original_instruction = [&](const Instruction &inst, uint64_t offset,
                                       std::vector<uint8_t> &kernel_text,
                                       std::vector<PendingTrace> &pending_traces) {
    const uint32_t inst_size = inst.size();
    const uint64_t target_offset = kernel_text.size();
    const auto *words = reinterpret_cast<const uint32_t *>(text.data() + offset);
    std::vector<uint32_t> copied_words(words, words + inst_size / sizeof(uint32_t));
    append_words(kernel_text, copied_words);
    // Continued-failure mode is diagnostic-only. Emit an explicit copy event so
    // diff reports make it clear which failed source instruction was preserved.
    queue_trace(pending_traces, inst, offset, nullptr, true, false, false, target_offset,
                std::move(copied_words));
  };

  auto continue_after_instruction_error = [&](const Instruction &inst, uint64_t offset,
                                              std::vector<uint8_t> &kernel_text,
                                              std::vector<PendingTrace> &pending_traces) {
    if (!continue_after_failure)
      return false;
    copy_original_instruction(inst, offset, kernel_text, pending_traces);
    return true;
  };

  auto relocation_diagnostic_kind = [&](const TextRelocationResult &relocation) {
    if (relocation.failure == TextLayoutFailureCategory::ResourceLimit)
      return DiagnosticKind::ResourceLimit;
    return DiagnosticKind::Legalization;
  };

  auto materialization_diagnostic_kind = [&](const KernelTextAppendResult &materialization) {
    if (materialization.failure == TextLayoutFailureCategory::ResourceLimit)
      return DiagnosticKind::ResourceLimit;
    return DiagnosticKind::KernelDescriptor;
  };

  struct KernelFailure {
    DiagnosticKind kind = DiagnosticKind::Legalization;
    std::string message;
    std::optional<uint64_t> guest_offset;
    std::string mnemonic;
    std::vector<std::string> required_work;
  };

  auto make_kernel_failure = [](DiagnosticKind kind, std::string message,
                                std::optional<uint64_t> guest_offset = std::nullopt,
                                std::string mnemonic = {},
                                std::vector<std::string> required_work = {}) {
    return KernelFailure{kind, std::move(message), guest_offset, std::move(mnemonic),
                         std::move(required_work)};
  };

  auto emit_skipped_kernel = [&](const KernelTranslationScope &scope,
                                 KernelFailure failure) -> bool {
    assert(scope.translation != nullptr && "kernel scope should have descriptor translation");
    if (scope.blocks.empty()) {
      append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                   "cannot skip failed kernel without a decoded source block",
                   scope.translation->entry_text_offset);
      return false;
    }

    const uint64_t source_entry = scope.translation->entry_text_offset;
    const bool skipped_uses_virtual_lds = scope.translation->needs_lds_overflow_buf;
    auto skipped_text = append_skipped_kernel_stub(
        translated_text,
        {.source_entry = scope.translation->entry_text_offset,
         .has_kernarg_preload_firmware_skip = scope.translation->has_kernarg_preload_firmware_skip},
        host_arch_);
    if (!skipped_text.ok) {
      append_error(result.diagnostics, materialization_diagnostic_kind(skipped_text),
                   skipped_text.message, skipped_text.source_offset);
      return false;
    }

    for (KdTranslation &translation : descriptor_translations) {
      if (translation.descriptorless_callable || translation.entry_text_offset != source_entry)
        continue;
      // Normal hardware-LDS and virtual-LDS sidecar descriptors share the same
      // source entry, but a sidecar translation failure must not turn a normal
      // descriptor that can still launch on hardware LDS into a no-op. Skip
      // together any descriptor of the failing variant. Additionally, when the
      // sidecar variant is the one failing, a normal descriptor whose static LDS
      // only fit because a sidecar was promised is itself undispatchable (its
      // advertised LDS would fault on the host), so it must be stubbed too.
      const bool same_variant = translation.needs_lds_overflow_buf == skipped_uses_virtual_lds;
      const bool orphaned_by_sidecar_failure = skipped_uses_virtual_lds &&
                                               !translation.needs_lds_overflow_buf &&
                                               translation.static_lds_requires_sidecar;
      if (!same_variant && !orphaned_by_sidecar_failure)
        continue;
      translation.target_entry_text_offset = skipped_text.target_entry;
      translation.target_body_entry_text_offset = skipped_text.target_body_entry;
      // A skipped descriptor must describe the target stub, not the failed
      // guest kernel. Leaving oversized SGPR/LDS/private requirements in place
      // can make HIP fail during launch even though the entry points at safe
      // target code. Granulated zero encodes the minimum allocation bucket.
      translation.configure_skipped_stub();
    }

    std::string message =
        "*** SKIPPED KERNEL " + kernel_label(*scope.translation) +
        " REPLACED WITH S_ENDPGM; DISPATCHING IT WILL SILENTLY PRODUCE INVALID OUTPUTS *** "
        "Translation error: " +
        std::move(failure.message);
    append_warning(result.diagnostics, DiagnosticKind::KernelSkipped, std::move(message),
                   failure.guest_offset ? failure.guest_offset
                                        : std::optional<uint64_t>(source_entry),
                   std::move(failure.mnemonic), std::move(failure.required_work));
    return true;
  };

  // Per-scope relocation output. Declared before fail_or_skip_kernel so a skip can
  // truncate them back to their pre-scope sizes together with translated_text —
  // otherwise a scope that fails AFTER appending relocations (e.g. during branch
  // fixup or descriptor recompute) would leave stale entries whose source mapping
  // then applies to the replacement stub or a later kernel.
  std::vector<TextOffsetRelocation> text_relocations;
  std::vector<PcRelativeDataRelocation> data_relocations;
  uint64_t next_text_relocation_placement = 0;

  auto fail_or_skip_kernel =
      [&](const KernelTranslationScope &scope, KernelFailure failure, size_t output_begin,
          const std::vector<DescriptorVariantCheckpoint> &descriptor_snapshot,
          size_t text_relocations_begin, size_t data_relocations_begin) -> bool {
    if (!skip_failed_kernels || scope.translation->descriptorless_callable) {
      append_error(result.diagnostics, failure.kind, std::move(failure.message),
                   failure.guest_offset, std::move(failure.mnemonic),
                   std::move(failure.required_work));
      return false;
    }

    const uint64_t source_entry = scope.translation->entry_text_offset;
    translated_text.resize(output_begin);
    // Discard any relocation records this scope committed before failing.
    text_relocations.resize(text_relocations_begin);
    data_relocations.resize(data_relocations_begin);
    for (const DescriptorVariantCheckpoint &saved : descriptor_snapshot) {
      if (saved.index >= descriptor_translations.size()) {
        append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                     "descriptor checkpoint index changed during skip rollback", source_entry);
        return false;
      }
      descriptor_translations[saved.index] = saved.translation;
    }

    KernelTranslationScope restored_scope = scope;
    restored_scope.translation = nullptr;
    for (KdTranslation &translation : descriptor_translations) {
      if (!same_kernel_scope_variant(translation, *scope.translation))
        continue;
      restored_scope.translation = &translation;
      break;
    }
    if (restored_scope.translation == nullptr) {
      append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                   "failed kernel descriptor was lost during skip rollback", source_entry);
      return false;
    }

    return emit_skipped_kernel(restored_scope, std::move(failure));
  };

  auto reserve_long_branch_sgpr_pair = [&](TranslationContext &context) -> std::optional<uint16_t> {
    auto base = next_long_branch_sgpr_pair(context, host_arch_);
    if (!base)
      return std::nullopt;
    context.require_sgprs(static_cast<uint32_t>(*base) + 2);
    return base;
  };

  for (const KernelTranslationScope &scope : scopes) {
    if (scope.blocks.empty())
      continue;
    assert(scope.translation != nullptr && "kernel scope should have descriptor translation");
    if (scope.translation->skipped)
      continue;

    const size_t output_begin = translated_text.size();
    const size_t text_relocations_begin = text_relocations.size();
    const size_t data_relocations_begin = data_relocations.size();
    const auto descriptor_snapshot =
        checkpoint_scope_descriptors(descriptor_translations, *scope.translation);
    bool skip_scope = false;

    if (!scope.translation->supported) {
      auto failure = make_kernel_failure(
          DiagnosticKind::KernelDescriptor,
          "kernel descriptor translation requires unsupported resource or ABI virtualization");
      for (const TranslationDiagnostic &diagnostic : scope.translation->diagnostics) {
        if (diagnostic.severity != DiagnosticSeverity::Error)
          continue;
        failure.kind = diagnostic.kind;
        failure.message = diagnostic.message;
        failure.guest_offset = diagnostic.guest_offset;
        failure.mnemonic = diagnostic.mnemonic;
        failure.required_work = diagnostic.required_work;
        break;
      }
      if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                              text_relocations_begin, data_relocations_begin))
        continue;
      return leave_unchanged();
    }

    const auto opaque_fallthrough =
        std::ranges::find_if(scope.blocks, [&](const BasicBlock *block) {
          return block != nullptr && block->falls_through_to_undecodable_text();
        });
    if (opaque_fallthrough != scope.blocks.end()) {
      auto failure =
          make_kernel_failure(DiagnosticKind::Legalization,
                              "reachable kernel code falls through into undecodable .text bytes",
                              (*opaque_fallthrough)->end_offset());
      if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                              text_relocations_begin, data_relocations_begin))
        continue;
      return leave_unchanged();
    }

    // Phase 3: translate this kernel into a temporary, source-ordered body. The
    // body starts at offset zero while it is being built; after final padding and
    // any launch window are chosen, every recorded target offset is rebased into
    // the output .text. This lets instruction expansions change block sizes
    // without precomputing speculative side-region offsets.
    KernelTextLayout layout;
    layout.source_entry = scope.translation->entry_text_offset;
    std::vector<ScratchVgprRangeLimit> descriptorless_vgpr_limits;

    if (scope.translation->descriptorless_callable) {
      const auto mode_write = std::ranges::find_if(scope.blocks, [](BasicBlock *block) {
        if (block == nullptr)
          return false;
        return std::ranges::any_of(block->instructions(), changes_gfx1250_vgpr_msb_state);
      });
      if (mode_write != scope.blocks.end()) {
        auto failure = make_kernel_failure(
            DiagnosticKind::ResourceLimit,
            "descriptorless callable code changes VGPR-MSB/MODE state, so its physical register "
            "envelope cannot be proven from final-HSACO metadata");
        if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                text_relocations_begin, data_relocations_begin))
          continue;
        return leave_unchanged();
      }

      descriptorless_vgpr_limits = descriptorless_vgpr_envelopes(scope);
      if (descriptorless_vgpr_limits.empty()) {
        auto failure = make_kernel_failure(
            DiagnosticKind::ResourceLimit,
            "descriptorless callable function ranges do not provide complete VGPR resource "
            "envelopes");
        if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                text_relocations_begin, data_relocations_begin))
          continue;
        return leave_unchanged();
      }
      const uint32_t source_vgpr_envelope =
          std::ranges::max(descriptorless_vgpr_limits, {}, &ScratchVgprRangeLimit::max_free_vgpr)
              .max_free_vgpr;
      if (source_vgpr_envelope > 256) {
        auto failure = make_kernel_failure(
            DiagnosticKind::ResourceLimit,
            "descriptorless callable low-bank VGPR envelope exceeds the supported resource "
            "contract");
        if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                text_relocations_begin, data_relocations_begin))
          continue;
        return leave_unchanged();
      }
      scope.translation->target_vgpr_count = source_vgpr_envelope;
      scope.translation->target_vgpr_allocation_count = source_vgpr_envelope;
    }

    TranslationContext kernel_context(
        scope.translation->target_vgpr_count, scope.translation->target_agpr_count,
        scope.translation->target_accvgpr_base, scope.translation->target_sgpr_count,
        scope.translation->target_private_size);
    kernel_context.private_spills_allowed = !scope.translation->descriptorless_callable;
    if (scope.translation->needs_lds_overflow_buf) {
      auto virtual_lds_base =
          reserve_virtual_lds_base_sgpr_pair(kernel_context, KernelBlockScope(scope.blocks),
                                             *scope.translation, guest_arch_, host_arch_);
      if (!virtual_lds_base) {
        auto failure = make_kernel_failure(
            DiagnosticKind::ResourceLimit,
            "virtual LDS lowering cannot reserve a backing-buffer SGPR pair", layout.source_entry);
        if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                text_relocations_begin, data_relocations_begin))
          continue;
        return leave_unchanged();
      }
      kernel_context.virtualize_lds = true;
      kernel_context.virtual_lds_base_sgpr = virtual_lds_base->base;
      kernel_context.virtual_lds_base_sgpr_spill_per_use = virtual_lds_base->spill_per_use;
      kernel_context.virtual_lds_kernarg_segment_ptr_sgpr =
          scope.translation->kernarg_segment_ptr_sgpr;
      kernel_context.virtual_lds_kernarg_pointer_offset =
          scope.translation->lds_overflow_kernarg_pointer_offset;
      scope.translation->virtual_lds_lowering.base_sgpr = virtual_lds_base->base;
      scope.translation->virtual_lds_lowering.prologue_temp_sgpr = virtual_lds_base->prologue_temp;
      scope.translation->virtual_lds_lowering.base_sgpr_spill_per_use =
          virtual_lds_base->spill_per_use;
      if (virtual_lds_base->spill_per_use) {
        const auto pointer_spill = kernel_context.reserve_persistent_semantic_spill_dwords(2);
        if (!pointer_spill) {
          auto failure = make_kernel_failure(
              DiagnosticKind::ResourceLimit,
              "virtual LDS backing-pointer spill offset overflows the 32-bit private segment",
              layout.source_entry);
          if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                  text_relocations_begin, data_relocations_begin))
            continue;
          return leave_unchanged();
        }
        kernel_context.virtual_lds_base_pointer_spilled = true;
        kernel_context.virtual_lds_base_pointer_spill_offset = *pointer_spill;
        scope.translation->virtual_lds_lowering.base_pointer_spilled = true;
        scope.translation->virtual_lds_lowering.base_pointer_spill_offset = *pointer_spill;
      }
      if (!append_virtual_lds_entry_prologue(*scope.translation, guest_arch_, host_arch_)) {
        auto failure = make_kernel_failure(
            DiagnosticKind::KernelDescriptor,
            "virtual LDS lowering cannot materialize backing-buffer pointer entry prologue",
            layout.source_entry);
        if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                text_relocations_begin, data_relocations_begin))
          continue;
        return leave_unchanged();
      }
    }

    layout.entry_plan = {
        .has_kernarg_preload_firmware_skip = scope.translation->has_kernarg_preload_firmware_skip,
        .kernarg_preload_firmware_entry_text_offset =
            scope.translation->kernarg_preload_firmware_entry_text_offset,
        .prologue_words = scope.translation->prologue_words,
    };
    if (!kernarg_preload_launch_window_fits(layout.entry_plan)) {
      auto failure = make_kernel_failure(
          DiagnosticKind::KernelDescriptor,
          "kernel descriptor prologue does not fit in the 256-byte kernarg preload compatibility "
          "window",
          layout.source_entry);
      if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                              text_relocations_begin, data_relocations_begin))
        continue;
      return leave_unchanged();
    }
    const bool can_use_long_direct_branches =
        !scope.translation->descriptorless_callable &&
        next_long_branch_sgpr_pair(kernel_context, host_arch_).has_value();
    LivenessAnalysisOptions liveness_options;
    liveness_options.max_free_vgpr =
        static_cast<uint16_t>(scope.translation->descriptorless_callable
                                  ? scope.translation->target_vgpr_count
                                  : isa_properties(host_arch_).max_addressable_vgprs_per_wf);
    liveness_options.arch = guest_arch_;
    if (scope.translation->descriptorless_callable) {
      liveness_options.scratch_reserved = llvm_amdhsa_callable_scratch_reserved_registers();
      liveness_options.scratch_vgpr_range_limits = descriptorless_vgpr_limits;
    }
    std::vector<BasicBlock *> scope_analysis_entries{scope.entry};
    const auto add_analysis_entry = [&](uint64_t offset) {
      BasicBlock *entry = block_for_offset(block_index, offset);
      if (entry != nullptr && entry->start_offset() == offset)
        scope_analysis_entries.push_back(entry);
    };
    if (scope.translation->descriptorless_callable) {
      for (const CallableRange &range : scope.translation->callable_ranges)
        add_analysis_entry(range.begin);
    } else if (scope.translation->has_kernarg_preload_firmware_skip) {
      add_analysis_entry(scope.translation->kernarg_preload_firmware_entry_text_offset);
    }
    std::ranges::sort(scope_analysis_entries);
    scope_analysis_entries.erase(std::ranges::unique(scope_analysis_entries).begin(),
                                 scope_analysis_entries.end());
    liveness_options.entry_blocks = scope_analysis_entries;
    liveness_options.text = text;
    if (options_.debug_min_free_vgpr)
      liveness_options.min_free_vgpr = *options_.debug_min_free_vgpr;
    std::vector<const Instruction *> live_before_instructions;
    bool scope_requires_liveness = false;
    if (semantic_translator_ && semantic_translator_->has_rules()) {
      // Semantic expansion rules are the only BinaryTranslator path that queries
      // LivenessAnalysis. Other rewrites use separate resource strategies: virtual
      // LDS grows descriptor-backed registers or explicitly saves/restores borrowed
      // registers. Collect live-before snapshots only for rules that can query them,
      // and skip the kernel dataflow entirely when no such rule is present.
      for (BasicBlock *block : scope.blocks) {
        if (block == nullptr)
          continue;
        for (const Instruction &inst : block->instructions()) {
          if (semantic_translator_->expand_rule_requires_liveness(inst)) {
            scope_requires_liveness = true;
            live_before_instructions.push_back(&inst);
          }
        }
      }
      liveness_options.restrict_live_before_to_instructions = true;
      liveness_options.live_before_instructions = std::span<const Instruction *const>(
          live_before_instructions.data(), live_before_instructions.size());
    }
    LivenessAnalysis liveness = LivenessAnalysis::unavailable();
    if (scope_requires_liveness) {
      const auto liveness_edges = scoped_call_liveness_edges(KernelBlockScope(scope.blocks), text);
      liveness = LivenessAnalysis(KernelBlockScope(scope.blocks), liveness_options, liveness_edges);
    }

    // Phase 4: translate each relocated body instruction at the current cursor.
    // Return-like s_setpc_b64 instructions are accepted only when they are the
    // terminator of a block reached from a validated call edge in this
    // kernel-local scope. Recovered indirect setpc/swappc consumers reserve a
    // fixed maximum-size window when recovery proves one effective target. When
    // one dynamic consumer has multiple recovered targets, no single direct
    // window can preserve semantics; DBT keeps the original indirect consumer
    // and asks the patch layer to rewrite each source-side PC builder once.
    std::unordered_set<uint64_t> valid_call_return_offsets =
        scoped_call_return_offsets(KernelBlockScope(scope.blocks), text);
    const auto callable_returns = descriptorless_callable_return_offsets(
        *scope.translation, KernelBlockScope(scope.blocks), text);
    valid_call_return_offsets.insert(callable_returns.begin(), callable_returns.end());
    struct RecoveredConsumer {
      std::vector<IndirectCallFixup> fixups;
      bool use_transfer_window = false;
      IndirectCallFixup window_fixup;
    };
    std::unordered_map<uint64_t, RecoveredConsumer> recovered_indirect_by_call;
    for (BasicBlock *block : scope.blocks) {
      for (const IndirectCallFixup &source_fixup : block->static_indirect_call_fixups()) {
        recovered_indirect_by_call[source_fixup.source_call_offset].fixups.push_back(source_fixup);
      }
    }

    // An incomplete consumer is only translatable when this scope can be proven
    // free of stale PC-derived values (see scope_relocatable_pc_builders). That
    // proof is expensive and only ever needed when such a consumer exists, so
    // establish it lazily; scopes without one keep byte-identical output.
    const bool has_incomplete_consumer =
        std::ranges::any_of(recovered_indirect_by_call, [](const auto &entry) {
          return std::ranges::any_of(entry.second.fixups, [](const IndirectCallFixup &fixup) {
            return fixup.source_incomplete;
          });
        });
    std::optional<std::vector<IndirectCallFixup>> whole_scope_builder_fixups;
    if (has_incomplete_consumer &&
        scope_incomplete_roots_are_entry_state(scope, relocation_table_callee_offsets))
      whole_scope_builder_fixups = scope_relocatable_pc_builders(scope.blocks);
    const bool no_stale_pc_values_in_scope = whole_scope_builder_fixups.has_value();

    std::vector<IndirectCallFixup> pending_builder_fixups;
    for (auto &[source_call_offset, consumer] : recovered_indirect_by_call) {
      if (consumer.fixups.empty())
        continue;

      const IndirectCallFixup &first = consumer.fixups.front();
      bool single_effective_target = true;
      for (const IndirectCallFixup &fixup : consumer.fixups) {
        if (fixup.source_call_sreg != first.source_call_sreg ||
            fixup.source_is_call != first.source_is_call ||
            fixup.source_return_sreg != first.source_return_sreg) {
          auto failure =
              make_kernel_failure(DiagnosticKind::Legalization,
                                  "recovered indirect branch has inconsistent consumer metadata",
                                  source_call_offset, "indirect branch");
          if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                  text_relocations_begin, data_relocations_begin)) {
            skip_scope = true;
            break;
          }
          return leave_unchanged();
        }
        if (fixup.source_target_offset != first.source_target_offset)
          single_effective_target = false;
      }
      if (skip_scope)
        break;

      // An incomplete fact means at least one predecessor left the PC pair
      // unconstrained. That path has no recovered builder or target to relocate,
      // yet its runtime SGPR pair may hold an original .text address; after DBT
      // relocates .text, the retained dynamic transfer would jump to stale or moved
      // bytes, and the unknown target block may be absent from the emitted scope.
      // Unless this scope was proven to contain no stale PC-derived value at all,
      // we cannot rule that out, so fail closed for the whole consumer rather
      // than relocate only the known builders.
      const bool any_incomplete = std::ranges::any_of(
          consumer.fixups, [](const IndirectCallFixup &fixup) { return fixup.source_incomplete; });
      if (any_incomplete && !no_stale_pc_values_in_scope) {
        auto failure = make_kernel_failure(
            DiagnosticKind::Legalization,
            "recovered indirect branch has an unconstrained predecessor path that cannot be "
            "relocated",
            source_call_offset, "indirect branch");
        if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                text_relocations_begin, data_relocations_begin)) {
          skip_scope = true;
          break;
        }
        return leave_unchanged();
      }

      if (any_incomplete) {
        // The unconstrained path can still deliver a value this consumer never
        // reaches through a recovered builder, so a direct transfer window would
        // redirect it. Keep the dynamic transfer; every PC value it can observe
        // is relocation-correct under the whole-scope proof.
        continue;
      }

      // A complete consumer with one effective target can become a direct window.
      // A complete multi-target consumer keeps the dynamic consumer and rewrites
      // each source-side builder (relocation/liveness).
      if (single_effective_target) {
        consumer.use_transfer_window = true;
        consumer.window_fixup = first;
      } else {
        pending_builder_fixups.insert(pending_builder_fixups.end(), consumer.fixups.begin(),
                                      consumer.fixups.end());
      }
    }
    if (skip_scope)
      continue;

    // Discharging the proof requires actually performing every rewrite it
    // assumes. Append it after the consumer-driven fixups so a builder that both
    // paths cover keeps the bytes the consumer path already produced today;
    // patch_recovered_builder_fixups collapses the duplicate range.
    if (no_stale_pc_values_in_scope) {
      pending_builder_fixups.insert(pending_builder_fixups.end(),
                                    whole_scope_builder_fixups->begin(),
                                    whole_scope_builder_fixups->end());
    }

    std::vector<uint8_t> kernel_text;
    std::vector<PendingTrace> pending_traces;
    uint64_t source_body_size = 0;
    for (BasicBlock *block : scope.blocks)
      source_body_size += block->size();
    uint64_t source_allocation_size = source_body_size;
    if (scope.translation->descriptorless_callable) {
      source_allocation_size = 0;
      for (size_t range_index = 0; range_index < scope.translation->callable_ranges.size();
           ++range_index) {
        const CallableRange &range = scope.translation->callable_ranges[range_index];
        uint64_t allocation_end = range_index + 1 < scope.translation->callable_ranges.size()
                                      ? scope.translation->callable_ranges[range_index + 1].begin
                                      : text.size();
        for (const KernelTranslationScope &other : scopes) {
          if (other.translation == nullptr || other.translation->descriptorless_callable)
            continue;
          for (const BasicBlock *block : other.blocks) {
            if (block != nullptr && block->start_offset() >= range.end)
              allocation_end = std::min(allocation_end, block->start_offset());
          }
        }
        if (allocation_end < range.end ||
            allocation_end - range.begin >
                std::numeric_limits<uint64_t>::max() - source_allocation_size) {
          source_allocation_size = 0;
          break;
        }
        source_allocation_size += allocation_end - range.begin;
      }
    }
    const uint64_t recovered_window_growth =
        recovered_indirect_by_call.size() * kMaxRecoveredIndirectTransferWords * sizeof(uint32_t);
    kernel_text.reserve(static_cast<size_t>(std::min<uint64_t>(
        source_body_size + recovered_window_growth, std::numeric_limits<size_t>::max())));

    std::unordered_map<uint64_t, uint64_t> target_offset_by_source_offset;
    target_offset_by_source_offset.reserve(
        static_cast<size_t>(source_body_size / sizeof(uint32_t)) + scope.blocks.size());
    std::unordered_set<uint64_t> instruction_start_offsets;
    instruction_start_offsets.reserve(static_cast<size_t>(source_body_size / sizeof(uint32_t)));
    layout.body_begin = 0;
    layout.blocks.reserve(scope.blocks.size());
    uint64_t next_branch_island_pool_offset = first_direct_branch_island_pool_offset();
    for (BasicBlock *block : scope.blocks) {
      BlockPlacement placement{.block = block,
                               .source_start = block->start_offset(),
                               .source_end = block->end_offset(),
                               .target_start = kernel_text.size(),
                               .target_end = kernel_text.size()};

      for (auto it = block->instructions().begin(); it != block->instructions().end(); ++it) {
        const auto &inst = *it;
        const uint64_t offset = inst.src_loc();
        const uint64_t target_offset = kernel_text.size();
        const uint32_t inst_size = inst.size();
        // Ask the semantic translator directly whether this instruction has an
        // expand rule. The previous positional cursor into live_before_instructions
        // silently depended on that vector being built in the exact same block/
        // instruction iteration order as this loop; querying by encoding/opcode
        // removes that hidden coupling.
        const bool has_semantic_expand_rule =
            semantic_translator_ != nullptr &&
            semantic_translator_->has_expand_rule(inst.encoding_id(), inst.opcode());
        // Record every instruction start, not just recovered-PC builder
        // boundaries. The same final map keeps ELF labels attached to the
        // relocated instruction stream after semantic expansions change sizes.
        // A block start can equal the previous block's end. If an island pool
        // was inserted at that boundary, replace the pre-pool end mapping so
        // branches and symbols land on the instruction.
        target_offset_by_source_offset.insert_or_assign(offset, target_offset);
        instruction_start_offsets.insert(offset);

        const auto recovered_it = recovered_indirect_by_call.find(offset);
        const bool has_recovered_indirect_call = recovered_it != recovered_indirect_by_call.end();
        const bool has_relocation_table_call = relocation_table_calls.contains(offset);
        const bool recovered_indirect_return = valid_call_return_offsets.contains(offset);
        const auto direct_branch_delta = inst.branch_offset_bytes();
        if ((inst.flags() & (INDIRECT_BRANCH | INDIRECT_CALL)) != 0 &&
            !has_recovered_indirect_call && !has_relocation_table_call &&
            !recovered_indirect_return && !direct_branch_delta) {
          auto failure = make_kernel_failure(
              DiagnosticKind::Legalization,
              "indirect branch or call target recovery is not implemented for relocated kernel "
              "text",
              offset, std::string(inst.mnemonic()));
          if (continue_after_failure && !skip_failed_kernels) {
            append_error(result.diagnostics, failure.kind, failure.message, failure.guest_offset,
                         failure.mnemonic, failure.required_work);
            if (continue_after_instruction_error(inst, offset, kernel_text, pending_traces)) {
              continue;
            }
          }
          if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                  text_relocations_begin, data_relocations_begin)) {
            skip_scope = true;
            break;
          }
          return leave_unchanged();
        }

        if (direct_branch_delta) {
          // Record direct branches while emitting the body, but patch only after
          // every block has a final target placement. This keeps fallthrough
          // implicit and limits fixups to explicit PC-relative edges. Emit the
          // branch into a fixed-size patch window. Kernels with a legal
          // descriptor-backed SGPR pair reserve the long form up front; kernels
          // already at the SGPR allocation limit keep compact branch slots so
          // DBT does not create artificial range pressure it cannot repair.
          const int64_t source_target =
              static_cast<int64_t>(offset + inst_size) + static_cast<int64_t>(*direct_branch_delta);
          if (source_target < 0) {
            auto failure =
                make_kernel_failure(DiagnosticKind::Legalization,
                                    "direct branch target is outside the source .text range",
                                    offset, std::string(inst.mnemonic()));
            if (continue_after_failure && !skip_failed_kernels) {
              append_error(result.diagnostics, failure.kind, failure.message, failure.guest_offset,
                           failure.mnemonic, failure.required_work);
              if (continue_after_instruction_error(inst, offset, kernel_text, pending_traces)) {
                continue;
              }
            }
            if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                    text_relocations_begin, data_relocations_begin)) {
              skip_scope = true;
              break;
            }
            return leave_unchanged();
          }
          const uint64_t branch_window_bytes = direct_branch_patch_window_bytes(
              inst, offset, static_cast<uint64_t>(source_target), can_use_long_direct_branches);
          layout.branch_fixups.push_back(
              {.inst = &inst,
               .source_inst_offset = offset,
               .source_target_offset = static_cast<uint64_t>(source_target),
               .target_inst_offset = target_offset,
               .target_window_bytes = branch_window_bytes});

          if (!inst.raw_encoding()) {
            auto failure = make_kernel_failure(DiagnosticKind::Legalization,
                                               "direct branch is missing raw encoding", offset,
                                               std::string(inst.mnemonic()));
            if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                    text_relocations_begin, data_relocations_begin)) {
              skip_scope = true;
              break;
            }
            return leave_unchanged();
          }

          const InstructionLegalization *branch_leg = lookup_legalization(inst);
          const uint16_t branch_dst_opcode = branch_leg ? branch_leg->target_opcode : inst.opcode();

          bool copied_original = false;
          bool changed = false;
          std::vector<uint32_t> target_words;
          if (!handle_encoding(inst, offset, kernel_text, branch_dst_opcode, text,
                               trace_callback_ != nullptr, copied_original, changed,
                               target_words)) {
            if (continue_after_instruction_error(inst, offset, kernel_text, pending_traces))
              continue;
            return leave_unchanged();
          }
          append_nop_padding(kernel_text, branch_window_bytes - inst.size(), host_arch_);
          queue_trace(pending_traces, inst, offset, branch_leg, copied_original, false, changed,
                      target_offset, std::move(target_words));
          continue;
        }

        if (has_recovered_indirect_call && recovered_it->second.use_transfer_window) {
          const IndirectCallFixup &source_fixup = recovered_it->second.window_fixup;
          layout.recovered_indirect_fixups.push_back(
              {.source_call_offset = source_fixup.source_call_offset,
               .source_target_offset = source_fixup.source_target_offset,
               .target_window_offset = target_offset,
               .target_sreg = source_fixup.source_call_sreg,
               .return_sreg = source_fixup.source_return_sreg,
               .is_call = source_fixup.source_is_call});
          append_nop_padding(kernel_text, kMaxRecoveredIndirectTransferWords * sizeof(uint32_t),
                             host_arch_);
          continue;
        }

        const uint32_t *raw = inst.raw_encoding();
        if (!raw) {
          copy_original_instruction(inst, offset, kernel_text, pending_traces);
          continue;
        }

        const InstructionLegalization *leg = lookup_legalization(inst);

        const uint16_t dst_opcode = leg ? leg->target_opcode : inst.opcode();

        // Try semantic lowering before raw encoding translation. A matched
        // semantic rule that cannot safely emit code is a translation error:
        // falling through would silently preserve guest semantics on the wrong
        // host ISA.
        if (has_semantic_expand_rule) {
          auto expansion =
              semantic_translator_->try_lower_expand(inst, offset, text, liveness, kernel_context);
          if (expansion.status == ExpandStatus::Failed) {
            auto failure = make_kernel_failure(
                DiagnosticKind::ExpandFailed,
                expansion.message.empty()
                    ? "semantic EXPAND rule matched, but could not safely lower"
                    : expansion.message,
                offset, std::string(inst.mnemonic()), std::move(expansion.required_work));
            if (continue_after_failure && !skip_failed_kernels) {
              append_error(result.diagnostics, failure.kind, failure.message, failure.guest_offset,
                           failure.mnemonic, failure.required_work);
              if (continue_after_instruction_error(inst, offset, kernel_text, pending_traces)) {
                continue;
              }
            }
            if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                    text_relocations_begin, data_relocations_begin)) {
              skip_scope = true;
              break;
            }
            return leave_unchanged();
          }

          if (expansion.status == ExpandStatus::Success) {
            std::vector<uint32_t> target_words = std::move(expansion.words);
            append_words(kernel_text, target_words);
            queue_trace(pending_traces, inst, offset, leg, false, true, true, target_offset,
                        std::move(target_words));
            continue;
          }
        }

        {
          auto virtual_lds_expansion =
              lower_virtual_lds_instruction(inst, kernel_context, guest_arch_, host_arch_);
          if (virtual_lds_expansion.status == ExpandStatus::Failed) {
            auto failure = make_kernel_failure(DiagnosticKind::ExpandFailed,
                                               virtual_lds_expansion.message.empty()
                                                   ? "virtual LDS lowering failed"
                                                   : virtual_lds_expansion.message,
                                               offset, std::string(inst.mnemonic()),
                                               std::move(virtual_lds_expansion.required_work));
            if (continue_after_failure && !skip_failed_kernels) {
              append_error(result.diagnostics, failure.kind, failure.message, failure.guest_offset,
                           failure.mnemonic, failure.required_work);
              if (continue_after_instruction_error(inst, offset, kernel_text, pending_traces)) {
                continue;
              }
            }
            if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                    text_relocations_begin, data_relocations_begin)) {
              skip_scope = true;
              break;
            }
            return leave_unchanged();
          }

          if (virtual_lds_expansion.status == ExpandStatus::Success) {
            std::vector<uint32_t> target_words = std::move(virtual_lds_expansion.words);
            append_words(kernel_text, target_words);
            queue_trace(pending_traces, inst, offset, leg, false, true, true, target_offset,
                        std::move(target_words));
            continue;
          }
        }

        if (leg && leg->action == Action::Expand) {
          auto failure = make_kernel_failure(
              DiagnosticKind::ExpandMissing,
              "legalization requires EXPAND, but no expansion rule is implemented", offset,
              std::string(inst.mnemonic()), {"Add a semantic expansion rule for this mnemonic."});
          if (continue_after_failure && !skip_failed_kernels) {
            append_error(result.diagnostics, failure.kind, failure.message, failure.guest_offset,
                         failure.mnemonic, failure.required_work);
            if (continue_after_instruction_error(inst, offset, kernel_text, pending_traces)) {
              continue;
            }
          }
          if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                  text_relocations_begin, data_relocations_begin)) {
            skip_scope = true;
            break;
          }
          return leave_unchanged();
        }

        // A stepping-only translation has identical encodings for ordinary
        // instructions. Copy the authoritative source text, including literal
        // and modifier suffix words, instead of reconstructing bytes from the
        // decoder's base-format raw encoding. Direct branches and recovered
        // indirect transfers have already taken their relocation paths above;
        // explicit profile expansions have already continued or failed closed.
        if (guest_arch_ == host_arch_ && leg == nullptr) {
          copy_original_instruction(inst, offset, kernel_text, pending_traces);
          continue;
        }

        // Cross-arch translation must have a legalization decision for every
        // opcode. When a lookup function exists (i.e. this is not a same-arch
        // identity pass) but the opcode is absent from the table, or the table
        // marks it Illegal, re-encoding it verbatim with the guest opcode number
        // would silently produce a different — possibly valid but wrong — host
        // instruction. Fail loudly instead of that silent passthrough. A null
        // lookup function means same-arch identity translation, where verbatim
        // copy is correct, so that path is intentionally not gated here.
        if (legalization_lookup_ != nullptr && (leg == nullptr || leg->action == Action::Illegal)) {
          auto failure = make_kernel_failure(
              DiagnosticKind::Legalization,
              leg == nullptr
                  ? "no legalization entry for this opcode on the target ISA; refusing to emit the "
                    "guest encoding verbatim"
                  : "legalization marks this opcode illegal on the target ISA",
              offset, std::string(inst.mnemonic()),
              {"Add a legalization/substitution/expansion entry for this mnemonic in the amdisa "
               "codegen pipeline."});
          if (continue_after_failure && !skip_failed_kernels) {
            append_error(result.diagnostics, failure.kind, failure.message, failure.guest_offset,
                         failure.mnemonic, failure.required_work);
            if (continue_after_instruction_error(inst, offset, kernel_text, pending_traces)) {
              continue;
            }
          }
          if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                  text_relocations_begin, data_relocations_begin)) {
            skip_scope = true;
            break;
          }
          return leave_unchanged();
        }

        bool copied_original = false;
        bool changed = false;
        std::vector<uint32_t> target_words;
        if (!handle_encoding(inst, offset, kernel_text, dst_opcode, text,
                             trace_callback_ != nullptr, copied_original, changed, target_words)) {
          if (continue_after_instruction_error(inst, offset, kernel_text, pending_traces)) {
            continue;
          }
          return leave_unchanged();
        }
        queue_trace(pending_traces, inst, offset, leg, copied_original, false, changed,
                    target_offset, std::move(target_words));
      }
      if (skip_scope)
        break;
      if (block->has_implicit_terminator()) {
        // Materialize the CFG boundary as part of the translated block. Like
        // any other target-side expansion, the terminator belongs in relocated
        // function extents. Without an architectural terminator, ordinary text
        // materialization can turn this unreachable stub into a fallthrough.
        const uint32_t endpgm = build_s_endpgm(host_arch_);
        append_words(kernel_text, std::span<const uint32_t>(&endpgm, 1));
      }
      placement.target_end = kernel_text.size();
      layout.blocks.push_back(placement);
      target_offset_by_source_offset.emplace(block->end_offset(), kernel_text.size());
      if (!scope.translation->descriptorless_callable && !can_use_long_direct_branches &&
          block != scope.blocks.back() && kernel_text.size() >= next_branch_island_pool_offset) {
        append_direct_branch_island_pool(kernel_text, layout, host_arch_);
        next_branch_island_pool_offset = next_direct_branch_island_pool_offset(kernel_text.size());
      }
    }
    if (skip_scope)
      continue;
    layout.body_end = kernel_text.size();

    if (continue_after_failure && has_error_diagnostic(result.diagnostics))
      continue;

    for (IndirectCallFixup fixup : pending_builder_fixups) {
      const auto getpc_it = target_offset_by_source_offset.find(fixup.source_getpc_offset);
      const auto begin_it = target_offset_by_source_offset.find(fixup.source_recovery_begin_offset);
      const auto end_it = target_offset_by_source_offset.find(fixup.source_recovery_end_offset);
      if (getpc_it == target_offset_by_source_offset.end() ||
          begin_it == target_offset_by_source_offset.end() ||
          end_it == target_offset_by_source_offset.end()) {
        auto failure = make_kernel_failure(
            DiagnosticKind::Legalization,
            "recovered indirect branch builder is not fully present in the relocated body",
            fixup.source_call_offset, "indirect branch");
        if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                text_relocations_begin, data_relocations_begin)) {
          skip_scope = true;
          break;
        }
        return leave_unchanged();
      }

      fixup.target_getpc_offset = getpc_it->second;
      fixup.target_recovery_begin_offset = begin_it->second;
      fixup.target_recovery_end_offset = end_it->second;
      layout.recovered_builder_fixups.push_back(fixup);
    }
    if (skip_scope)
      continue;

    auto materialized =
        append_relocated_kernel_text(translated_text, layout, kernel_text, host_arch_);
    if (!materialized.ok) {
      auto failure = make_kernel_failure(materialization_diagnostic_kind(materialized),
                                         materialized.message, materialized.source_offset);
      if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                              text_relocations_begin, data_relocations_begin))
        continue;
      return leave_unchanged();
    }
    if (scope.translation->descriptorless_callable &&
        translated_text.size() - output_begin > source_allocation_size) {
      auto failure = make_kernel_failure(
          DiagnosticKind::ResourceLimit,
          "descriptorless callable translation exceeds its original executable allocation");
      if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                              text_relocations_begin, data_relocations_begin))
        continue;
      return leave_unchanged();
    }
    const uint64_t target_delta = materialized.target_delta;
    const uint64_t placement_id = next_text_relocation_placement++;
    for (const auto &[source_offset, target_offset] : target_offset_by_source_offset) {
      text_relocations.push_back({
          .source_offset = source_offset,
          .target_offset = target_offset + target_delta,
          .placement_id = placement_id,
          .source_is_instruction_start = instruction_start_offsets.contains(source_offset),
      });
    }
    for (const RelocationTableDispatch &dispatch : relocation_table_dispatches) {
      if (!target_offset_by_source_offset.contains(dispatch.source_call_offset))
        continue;
      const auto getpc = target_offset_by_source_offset.find(dispatch.source_getpc_offset);
      const auto add = target_offset_by_source_offset.find(dispatch.source_address_add_offset);
      if (getpc == target_offset_by_source_offset.end() ||
          add == target_offset_by_source_offset.end()) {
        append_error(result.diagnostics, DiagnosticKind::Legalization,
                     "relocation-table GOT address builder is not fully present in the relocated "
                     "body",
                     dispatch.source_call_offset, "s_swap_pc_i64");
        return leave_unchanged();
      }
      data_relocations.push_back(
          {.target_getpc_offset = getpc->second + target_delta,
           .target_literal_offset = add->second + target_delta + sizeof(uint32_t),
           .source_target_vaddr = dispatch.source_table_address_vaddr});
    }

    // Phase 5: now that every emitted source block has a final target offset,
    // patch explicit direct branches, recovered source-side builders, and
    // recovered indirect transfer windows.
    auto patched_direct_branches = patch_direct_branch_fixups(translated_text, layout, host_arch_);
    if (!patched_direct_branches.ok &&
        patched_direct_branches.reason == TextLayoutFailureReason::BranchOutOfRange) {
      if (auto sgpr = reserve_long_branch_sgpr_pair(kernel_context)) {
        layout.long_branch_sgpr = *sgpr;
        patched_direct_branches = patch_direct_branch_fixups(translated_text, layout, host_arch_);
      }
    }
    if (!patched_direct_branches.ok) {
      auto failure = make_kernel_failure(relocation_diagnostic_kind(patched_direct_branches),
                                         patched_direct_branches.message,
                                         patched_direct_branches.source_offset);
      if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                              text_relocations_begin, data_relocations_begin))
        continue;
      return leave_unchanged();
    }

    if (auto patched = patch_recovered_builder_fixups(translated_text, layout, host_arch_);
        !patched.ok) {
      auto failure = make_kernel_failure(relocation_diagnostic_kind(patched), patched.message,
                                         patched.source_offset, "indirect branch");
      if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                              text_relocations_begin, data_relocations_begin))
        continue;
      return leave_unchanged();
    }

    if (auto patched = patch_recovered_indirect_fixups(translated_text, layout, host_arch_);
        !patched.ok) {
      auto failure = make_kernel_failure(relocation_diagnostic_kind(patched), patched.message,
                                         patched.source_offset, "indirect branch");
      if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                              text_relocations_begin, data_relocations_begin))
        continue;
      return leave_unchanged();
    }

    if (scope.translation->descriptorless_callable) {
      // There is no descriptor to advertise a larger resource request to the
      // caller. The source callable's decoded low-bank VGPR envelope and the
      // architecture's fixed callable SGPR set are therefore hard limits, and
      // descriptorless lowering may not manufacture private spill storage.
      if (kernel_context.required_vgpr_count > kernel_context.num_vgprs ||
          kernel_context.required_sgpr_count > kernel_context.num_sgprs ||
          kernel_context.required_private_segment_fixed_size >
              kernel_context.private_segment_fixed_size) {
        auto failure = make_kernel_failure(
            DiagnosticKind::ResourceLimit,
            "descriptorless callable lowering requires resources that cannot be advertised "
            "without a kernel descriptor");
        if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                text_relocations_begin, data_relocations_begin))
          continue;
        return leave_unchanged();
      }
    } else {
      if (kernel_context.required_vgpr_count > kernel_context.num_vgprs)
        scope.translation->target_vgpr_count = kernel_context.required_vgpr_count;
      if (kernel_context.required_sgpr_count > kernel_context.num_sgprs)
        scope.translation->target_sgpr_count = kernel_context.required_sgpr_count;
      if (kernel_context.required_private_segment_fixed_size >
          kernel_context.private_segment_fixed_size) {
        scope.translation->target_private_size = kernel_context.required_private_segment_fixed_size;
      }
    }

    if (!scope.translation->descriptorless_callable &&
        (scope.translation->target_vgpr_count != kernel_context.num_vgprs ||
         scope.translation->target_sgpr_count != kernel_context.num_sgprs ||
         scope.translation->target_private_size != kernel_context.private_segment_fixed_size)) {
      // Semantic rules may allocate descriptor-backed scratch registers or
      // per-lane private spill slots beyond the kernel's original resources.
      // Recompute the descriptor with those larger minimums before patching it
      // into the output image.
      KernelDescriptorTranslationOptions descriptor_options;
      descriptor_options.minimum_vgprs = scope.translation->target_vgpr_count;
      descriptor_options.minimum_sgprs = scope.translation->target_sgpr_count;
      descriptor_options.private_segment_fixed_size_addend =
          scope.translation->target_private_size - kernel_context.private_segment_fixed_size;
      descriptor_options.virtualize_lds = scope.translation->needs_lds_overflow_buf;
      descriptor_options.allow_oversized_lds =
          can_emit_sidecar_descriptors && !scope.translation->needs_lds_overflow_buf;

      // Descriptor growth is intentionally done after instruction lowering so
      // each kernel is translated once. Only descriptors that enter this code
      // scope need the larger register counts; rescanning the whole image would
      // also recompute unrelated kernels and risks mixing diagnostics across
      // scopes.
      bool recomputed_descriptor = false;
      for (KdTranslation &translation : descriptor_translations) {
        if (!same_kernel_scope_variant(translation, *scope.translation))
          continue;

        auto updated = descriptor_translator.translate_descriptor(
            patcher.image_bytes(), translation.descriptor_file_offset,
            translation.entry_text_offset, descriptor_options);
        if (!updated) {
          auto failure =
              make_kernel_failure(DiagnosticKind::KernelDescriptor,
                                  "kernel descriptor translation could not be recomputed");
          if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                  text_relocations_begin, data_relocations_begin)) {
            skip_scope = true;
            break;
          }
          return leave_unchanged();
        }
        updated->kernel_name = translation.kernel_name;
        updated->sidecar_descriptor = translation.sidecar_descriptor;
        updated->virtual_lds_lowering = translation.virtual_lds_lowering;
        // Register feedback also recomputes ordinary, non-virtual descriptors.
        // Only a virtual-LDS variant owns a backing-pointer entry prologue;
        // asking an unrelated ISA pair to materialize one makes otherwise
        // valid SGPR/VGPR growth fail after semantic lowering.
        if (updated->needs_lds_overflow_buf &&
            !append_virtual_lds_entry_prologue(*updated, guest_arch_, host_arch_)) {
          auto failure = make_kernel_failure(
              DiagnosticKind::KernelDescriptor,
              "virtual LDS lowering cannot materialize backing-buffer pointer entry prologue");
          if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                  text_relocations_begin, data_relocations_begin)) {
            skip_scope = true;
            break;
          }
          return leave_unchanged();
        }

        if (!updated->supported) {
          if (skip_failed_kernels) {
            auto failure = make_kernel_failure(DiagnosticKind::KernelDescriptor,
                                               "kernel descriptor translation requires unsupported "
                                               "resource or ABI virtualization");
            for (const TranslationDiagnostic &diagnostic : updated->diagnostics) {
              if (diagnostic.severity != DiagnosticSeverity::Error)
                continue;
              failure.kind = diagnostic.kind;
              failure.message = diagnostic.message;
              failure.guest_offset = diagnostic.guest_offset;
              failure.mnemonic = diagnostic.mnemonic;
              failure.required_work = diagnostic.required_work;
              break;
            }
            if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                    text_relocations_begin, data_relocations_begin)) {
              skip_scope = true;
              break;
            }
          }
          append_diagnostics(result.diagnostics, updated->diagnostics);
          append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                       "kernel descriptor translation requires unsupported resource or ABI "
                       "virtualization; leaving code object unchanged");
          return leave_unchanged();
        }
        append_diagnostics(result.diagnostics, updated->diagnostics);

        if (updated->prologue_words != translation.prologue_words) {
          auto failure = make_kernel_failure(
              DiagnosticKind::KernelDescriptor,
              "kernel descriptor prologue changed after relocated text was emitted");
          if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                  text_relocations_begin, data_relocations_begin)) {
            skip_scope = true;
            break;
          }
          return leave_unchanged();
        }

        updated->target_entry_text_offset = layout.target_entry;
        updated->target_body_entry_text_offset = layout.target_body_entry;
        translation = std::move(*updated);
        recomputed_descriptor = true;
      }
      if (skip_scope)
        continue;

      if (!recomputed_descriptor) {
        auto failure = make_kernel_failure(DiagnosticKind::KernelDescriptor,
                                           "kernel descriptor translation could not be recomputed");
        if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                text_relocations_begin, data_relocations_begin))
          continue;
        return leave_unchanged();
      }
    }

    flush_traces(pending_traces, target_delta);

    for (KdTranslation &translation : descriptor_translations) {
      if (!same_kernel_scope_variant(translation, *scope.translation))
        continue;
      translation.target_entry_text_offset = layout.target_entry;
      translation.target_body_entry_text_offset = layout.target_body_entry;
    }
  }

  if (continue_after_failure && has_error_diagnostic(result.diagnostics))
    return leave_unchanged();

  if (!has_kernel_descriptors && translated_text.size() > text.size()) {
    append_error(result.diagnostics, DiagnosticKind::ResourceLimit,
                 "descriptorless callable translation exceeds the original executable .text "
                 "allocation");
    return leave_unchanged();
  }

  // Phase 6 commits the completed translation plan without mixing ELF mutation
  // and sidecar metadata construction into the per-kernel lowering transaction.
  auto materialized = materialize_translated_code_object(
      std::move(patcher), std::move(translated_text), text.size(), text_relocations,
      data_relocations, descriptor_translations, host_arch_, target_mach_, is_gfx1250_b0_to_a0,
      result.diagnostics);
  if (!materialized)
    return leave_unchanged();
  result.elf_bytes = std::move(*materialized);
  return result;
}

bool BinaryTranslator::handle_encoding(const Instruction &inst, uint64_t offset,
                                       std::vector<uint8_t> &text, uint16_t dst_opcode,
                                       std::span<const uint8_t> orig_text, bool collect_trace_words,
                                       bool &copied_original, bool &changed,
                                       std::vector<uint32_t> &target_words) {
  const uint32_t *raw = inst.raw_encoding();
  assert(raw && "handle_encoding called without raw encoding");
  copied_original = false;
  changed = false;
  if (collect_trace_words)
    target_words.clear();

  if (!encoding_translate_) {
    copied_original = true;
    const size_t word_count = inst.size() / sizeof(uint32_t);
    if (collect_trace_words)
      target_words.assign(raw, raw + word_count);
    append_words(text, std::span<const uint32_t>(raw, word_count));
    return true;
  }

  const uint32_t w0 = raw[0];
  const uint32_t w1 = inst.size() > 4 ? raw[1] : 0;
  const uint32_t w2 = inst.size() > 8 ? raw[2] : 0;

  auto tr = encoding_translate_(inst.encoding_id(), w0, w1, w2, dst_opcode);

  if (tr.word_count == 0) {
    copied_original = true;
    const size_t word_count = inst.size() / sizeof(uint32_t);
    if (collect_trace_words)
      target_words.assign(raw, raw + word_count);
    append_words(text, std::span<const uint32_t>(raw, word_count));
    return true;
  }

  // Append trailing literal constant when the source instruction is larger
  // than the translated encoding. This handles single-word formats (SOP1,
  // SOP2, VOP1, VOP2, etc.) with a 32-bit literal appended when a source
  // operand is 0xFF. The encoding translator returns the format's native
  // word count; the literal is always one extra word beyond that.
  // Guard: only append if the gap is exactly one word (the literal). Larger
  // gaps would indicate a format mismatch, not a trailing literal.
  const uint32_t translated_bytes = tr.word_count * 4u;
  const uint32_t orig_bytes = inst.size();
  if (orig_bytes - translated_bytes == 4 && tr.word_count < 3) {
    uint32_t lit_word;
    std::memcpy(&lit_word, orig_text.data() + offset + translated_bytes, 4);
    tr.words[tr.word_count++] = lit_word;
  }

  append_words(text, std::span<const uint32_t>(tr.words, tr.word_count));
  if (collect_trace_words) {
    target_words.assign(tr.words, tr.words + tr.word_count);
    changed = words_changed(raw_words_for_inst(inst), target_words);
  }
  return true;
}

} // namespace rocjitsu
