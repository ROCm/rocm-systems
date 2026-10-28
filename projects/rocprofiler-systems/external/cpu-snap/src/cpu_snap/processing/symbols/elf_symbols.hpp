#pragma once

/**
 * @file elf_symbols.hpp
 * @brief Read FUNC/GNU_IFUNC symbols from an ELF's .symtab/.dynsym and
 *        look up the symbol covering a given virtual address.
 *
 * Used by symbolizer.hpp to detect addresses that fall outside every
 * known symbol's [st_value, st_value + st_size) range: without a size
 * check, addr2line names such an address after whatever symbol
 * precedes it, which is wrong for stripped libraries that only ship
 * .dynsym.
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <elf.h>

// The nested detail namespace below is kept split (not concatenated into
// this one) to match the existing cpu_snap::processing::symbols::detail pattern in
// proc_maps.hpp.
// NOLINTNEXTLINE(modernize-concat-nested-namespaces)
namespace cpu_snap::processing::symbols
{

// NOLINTBEGIN(readability-identifier-naming): codebase uses k_*-prefixed
//   lower_case for global constants; the .clang-tidy GlobalConstantCase
//   rule defaults to UPPER_CASE but the entire project ships with k_*.
namespace detail
{

/** @brief fclose deleter for FILE* handles. */
struct file_closer
{
    void operator()(FILE* file) const noexcept
    {
        [[maybe_unused]] const auto close_result = std::fclose(file);
    }
};

/** @brief An open 64-bit little-endian ELF file and its validated header. */
struct elf64_file
{
    std::unique_ptr<FILE, file_closer> file;
    Elf64_Ehdr                         ehdr{};
};

/**
 * @brief Open @p path and validate its ELF header.
 *
 * x86_64 only: rejects anything that is not a 64-bit little-endian ELF.
 * @param path Path to the ELF file.
 * @return The open file positioned after the header and the header, or
 *         @c std::nullopt if the file cannot be opened, the header
 *         cannot be read, or it is not a 64-bit LE ELF.
 */
[[nodiscard]] inline std::optional<elf64_file> open_elf64(const std::string& path)
{
    elf64_file elf;
    elf.file.reset(std::fopen(path.c_str(), "rb"));
    if(elf.file == nullptr) return std::nullopt;

    const Elf64_Ehdr& ehdr = elf.ehdr;
    const bool header_ok =
        std::fread(&elf.ehdr, sizeof(elf.ehdr), 1, elf.file.get()) == 1
        && ehdr.e_ident[EI_MAG0] == ELFMAG0 && ehdr.e_ident[EI_MAG1] == ELFMAG1
        && ehdr.e_ident[EI_MAG2] == ELFMAG2 && ehdr.e_ident[EI_MAG3] == ELFMAG3
        && ehdr.e_ident[EI_CLASS] == ELFCLASS64
        && ehdr.e_ident[EI_DATA] == ELFDATA2LSB;
    if(!header_ok) return std::nullopt;
    return elf;
}

/**
 * @brief One FUNC/GNU_IFUNC/sized-NOTYPE symbol read from .symtab or
 *        .dynsym.
 *
 * @c name is a view into the owning elf_symbol_table's string buffers;
 * it is valid as long as that elf_symbol_table is alive.
 */
struct elf_symbol
{
    std::uint64_t    st_value = 0;
    std::uint64_t    st_size  = 0;
    std::string_view name;
};

/**
 * @brief FUNC-like symbols of one ELF file, sorted by address, plus
 *        whether the file carries DWARF debug info.
 */
struct elf_symbol_table
{
    /** Symbols with st_size > 0, sorted ascending by st_value. */
    std::vector<elf_symbol> symbols;
    /** True if the ELF has a .debug_info section (DWARF present). */
    bool                     has_debug_info = false;

    // Backing storage for elf_symbol::name views; never touched after
    // read_elf_symbol_table() returns (moves preserve the heap buffer
    // address, so views stay valid).
    std::vector<char> symtab_strs;
    std::vector<char> dynsym_strs;
};

/**
 * @brief Find the symbol whose [st_value, st_value + st_size) range
 *        covers @p vaddr.
 *
 * Pure binary search, no I/O; suitable for compile-time testing.
 *
 * @param vaddr             ELF virtual address to look up.
 * @param symbols_by_value  Symbols sorted ascending by st_value (as
 *                          produced by read_elf_symbol_table()).
 * @return Copy of the covering symbol, or @c std::nullopt if @p vaddr
 *         lies in a gap between symbols, before the first symbol, or
 *         past the last symbol's range.
 */
[[nodiscard]] constexpr std::optional<elf_symbol> find_symbol(
    std::uint64_t vaddr, std::span<const elf_symbol> symbols_by_value) noexcept
{
    const auto past_vaddr =
        std::ranges::upper_bound(symbols_by_value, vaddr, std::less{}, &elf_symbol::st_value);
    if(past_vaddr == symbols_by_value.begin())
    {
        return std::nullopt;
    }
    const auto& candidate = *(past_vaddr - 1);
    if(vaddr < candidate.st_value + candidate.st_size)
    {
        return candidate;
    }
    return std::nullopt;
}

/**
 * @brief Parse one SHT_SYMTAB/SHT_DYNSYM section's FUNC-like symbols
 *        into @p out_symbols, backed by @p out_str_storage.
 *
 * @param sym_shdr        The symbol-table section header.
 * @param shdrs           All section headers (for the linked strtab).
 * @param read_section    Callable reading a section's raw bytes.
 * @param out_symbols     Appended to with the parsed FUNC-like symbols.
 * @param out_str_storage Receives the linked string-table bytes; the
 *                        backing storage for @c out_symbols' name views.
 */
template<typename ReadSectionFn>
inline void add_symbols_from(const Elf64_Shdr& sym_shdr, const std::vector<Elf64_Shdr>& shdrs,
                              const ReadSectionFn&      read_section,
                              std::vector<elf_symbol>& out_symbols,
                              std::vector<char>&        out_str_storage)
{
    if(sym_shdr.sh_entsize == 0 || sym_shdr.sh_link >= shdrs.size()) return;
    out_str_storage = read_section(shdrs.at(sym_shdr.sh_link));
    if(out_str_storage.empty()) return;

    const std::vector<char> sym_data = read_section(sym_shdr);
    const std::size_t       count    = sym_data.size() / sizeof(Elf64_Sym);
    const auto*             syms     = reinterpret_cast<const Elf64_Sym*>(sym_data.data());

    for(std::size_t i = 0; i < count; ++i)
    {
        const Elf64_Sym& sym  = syms[i];
        const auto        type = ELF64_ST_TYPE(sym.st_info);
        const bool         is_func_like =
            type == STT_FUNC || type == STT_GNU_IFUNC || type == STT_NOTYPE;
        if(!is_func_like || sym.st_size == 0 || sym.st_shndx == SHN_UNDEF
           || sym.st_shndx == SHN_ABS)
        {
            continue;
        }
        if(sym.st_name >= out_str_storage.size()) continue;

        out_symbols.push_back(elf_symbol{
            .st_value = sym.st_value,
            .st_size  = sym.st_size,
            .name     = std::string_view{ out_str_storage.data() + sym.st_name },
        });
    }
}

/**
 * @brief Read the FUNC/GNU_IFUNC/sized-NOTYPE symbols and debug-info
 *        presence of an ELF file.
 *
 * x86_64 only: rejects anything that is not a 64-bit little-endian
 * ELF, mirroring read_elf_load_segments().
 *
 * @param path Path to the ELF file.
 * @return Symbol table; empty (no symbols, has_debug_info=false) if
 *         @p path cannot be opened, is not a 64-bit LE ELF, or has no
 *         section headers.
 */
[[nodiscard]] inline elf_symbol_table read_elf_symbol_table(const std::string& path)
{
    elf_symbol_table table;

    const auto elf = open_elf64(path);
    if(!elf) return table;

    FILE*             file = elf->file.get();
    const Elf64_Ehdr& ehdr = elf->ehdr;
    if(ehdr.e_shoff == 0 || ehdr.e_shnum == 0) return table;

    std::vector<Elf64_Shdr> shdrs(ehdr.e_shnum);
    if(std::fseek(file, static_cast<std::int64_t>(ehdr.e_shoff), SEEK_SET) != 0
       || std::fread(shdrs.data(), sizeof(Elf64_Shdr), shdrs.size(), file) != shdrs.size())
    {
        return table;
    }

    const auto read_section = [file](const Elf64_Shdr& shdr) -> std::vector<char>
    {
        std::vector<char> data(shdr.sh_size);
        if(shdr.sh_size == 0) return data;
        if(std::fseek(file, static_cast<std::int64_t>(shdr.sh_offset), SEEK_SET) != 0
           || std::fread(data.data(), 1, data.size(), file) != data.size())
        {
            return {};
        }
        return data;
    };

    if(ehdr.e_shstrndx < shdrs.size())
    {
        const std::vector<char> shstrtab = read_section(shdrs.at(ehdr.e_shstrndx));
        const auto section_name = [&shstrtab](const Elf64_Shdr& shdr) -> std::string_view
        {
            if(shdr.sh_name >= shstrtab.size()) return {};
            return { shstrtab.data() + shdr.sh_name };
        };
        table.has_debug_info = std::ranges::any_of(
            shdrs, [&](const Elf64_Shdr& shdr) { return section_name(shdr) == ".debug_info"; });
    }

    for(const Elf64_Shdr& shdr : shdrs)
    {
        if(shdr.sh_type == SHT_SYMTAB)
        {
            add_symbols_from(shdr, shdrs, read_section, table.symbols, table.symtab_strs);
        }
        else if(shdr.sh_type == SHT_DYNSYM)
        {
            add_symbols_from(shdr, shdrs, read_section, table.symbols, table.dynsym_strs);
        }
    }

    std::ranges::sort(table.symbols, {}, &elf_symbol::st_value);

    return table;
}

}  // namespace detail
// NOLINTEND(readability-identifier-naming)

}  // namespace cpu_snap::processing::symbols
