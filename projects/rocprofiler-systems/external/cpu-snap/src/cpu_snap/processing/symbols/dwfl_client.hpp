#pragma once

/**
 * @file dwfl_client.hpp
 * @brief In-process replacement for `addr2line -ifCe`: resolves ELF addresses
 *        of one file to inline chains and source lines with libdwfl
 *        (elfutils), without starting a process.
 *
 * Produces what the symbolizer used to parse from addr2line's output:
 *   - functions: the innermost DWARF function scope containing the address
 *     and its physical parents (inlined subroutines, subprograms, entry
 *     points), innermost first, as elfutils' addr2line -i walks them. Each
 *     is named by its linkage name (through abstract origin /
 *     specification); without one, like bfd, by the fallback symbol below
 *     when that symbol starts at the function's lowest address, else by
 *     DW_AT_name. Demangled with parameters like `addr2line -C`. Without a
 *     named DWARF function: the fallback symbol (bfd's
 *     _bfd_elf_find_function and better_fit, binutils bfd/dwarf2.c): over
 *     the main file's .symtab (else .dynsym), in table order, the
 *     function-like symbol of the address's section starting nearest below
 *     the address, even if it ends before the address.
 *   - file_line: the line-table row covering the address, as addr2line
 *     prints it: "<file>:<line>", " (discriminator N)" when non-zero,
 *     "<file>:?" for line 0; empty without a row.
 * A file that cannot be opened yields empty chains (addr2line printed
 * nothing for it).
 *
 * Known differences from addr2line (binutils 2.42), see
 * planning/cpu-snap/symbolization.md in rocprofiler-systems:
 *   - bfd names the innermost function without a linkage name, the first
 *     time it looks it up, after the fallback symbol even when that symbol
 *     does not start at the function (an inlined lambda shows as its
 *     caller); here the name does not depend on lookup order.
 *   - With DWARF in a separate debug file, bfd takes fallback symbols from
 *     that file's .symtab; here always from the main file.
 *   - A line-table sequence that never sets the file uses file 1 here, as
 *     DWARF 5 specifies; addr2line 2.42 uses file 0 (fixed in later
 *     binutils, PR 30783).
 *   - DWARF 5 directory 0 is not prefixed with the compilation directory a
 *     second time ("./misc/err.c", not "./misc/./misc/err.c").
 *   - No file name from an STT_FILE symbol when there is no line row.
 *   - .dynsym fallback names carry no "@VERSION" suffix; extended section
 *     indices (SHN_XINDEX) are not resolved.
 *   - A sequence's zero-length last row at the address where another
 *     sequence of the same CU starts, when that sequence comes first in the
 *     line program, wins over it.
 */

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <dwarf.h>
#include <elf.h>
#include <elfutils/libdw.h>
#include <elfutils/libdwfl.h>
#include <gelf.h>
#include <libelf.h>

#include "cpu_snap/processing/symbols/symbol_name.hpp"

namespace cpu_snap::processing::symbols
{

/**
 * @brief Resolution of one address: what addr2line printed for it.
 */
struct addr2line_chain
{
    /** Function names, innermost inline frame first; empty when nothing was
     *  found for the address. */
    std::vector<std::string> functions;
    /** "file:line" of the address; empty when unavailable. */
    std::string file_line;
};

/**
 * @brief Resolves (ip, ELF address) pairs of one ELF file with libdwfl.
 *
 * Keeps one libdwfl session per file, opened on first use and reused for
 * later batches. Not thread-safe; one instance per symbolizer.
 */
class dwfl_client
{
public:
    /** (ip, ELF address) pair; only the ELF address is resolved. */
    using address_pair = std::pair<std::uint64_t, std::uint64_t>;

    /**
     * @brief Resolve every pair's ELF address in @p path.
     * @return One entry per pair, in order.
     */
    [[nodiscard]] std::vector<addr2line_chain> resolve(
        std::string_view path, std::span<const address_pair> pairs)
    {
        file_session&                file = session_for(path);
        std::vector<addr2line_chain> results;
        results.reserve(pairs.size());
        for(const auto& pair : pairs)
        {
            results.push_back(file.resolve(pair.second));
        }
        return results;
    }

private:
    struct dwfl_deleter
    {
        void operator()(Dwfl* dwfl) const noexcept { dwfl_end(dwfl); }
    };

    /** One symbol of the fallback table, as bfd sees it. */
    struct fallback_symbol
    {
        std::uint64_t value      = 0;
        std::uint64_t size       = 0;
        std::size_t   shndx      = 0;
        unsigned char type       = STT_NOTYPE;
        unsigned char bind       = STB_LOCAL;
        unsigned char visibility = STV_DEFAULT;
        std::string   name;
    };

    /** An SHF_ALLOC section, in section-header order. */
    struct alloc_section
    {
        std::size_t   index = 0;
        std::uint64_t addr  = 0;
        std::uint64_t size  = 0;
    };

    class file_session
    {
    public:
        explicit file_session(const std::string& path)
        {
            // dwfl_standard_find_debuginfo reads the char* this points to;
            // nullptr selects the default search path (".debug", /usr/lib/debug).
            static char*                s_debuginfo_path = nullptr;
            static const Dwfl_Callbacks k_callbacks{
                .find_elf        = dwfl_build_id_find_elf,
                .find_debuginfo  = dwfl_standard_find_debuginfo,
                .section_address = dwfl_offline_section_address,
                .debuginfo_path  = &s_debuginfo_path,
            };
            m_dwfl.reset(dwfl_begin(&k_callbacks));
            if(m_dwfl == nullptr) return;
            dwfl_report_begin(m_dwfl.get());
            // base 0 + add_p_vaddr: bias 0 for ET_DYN (also prelinked) and
            // ET_EXEC, so module addresses are the file's own ELF addresses.
            m_module =
                dwfl_report_elf(m_dwfl.get(), path.c_str(), path.c_str(), -1, 0, true);
            if(dwfl_report_end(m_dwfl.get(), nullptr, nullptr) != 0) m_module = nullptr;
        }

        [[nodiscard]] addr2line_chain resolve(std::uint64_t vaddr)
        {
            addr2line_chain chain;
            if(m_module == nullptr) return chain;

            // The symbol addr2line falls back to for this address; it also
            // names DWARF functions that have no linkage name.
            const fallback_symbol* fallback = fallback_symbol_for(vaddr);

            Dwarf_Addr bias  = 0;
            Dwarf_Die* cudie = dwfl_module_addrdie(m_module, vaddr, &bias);
            if(cudie != nullptr)
            {
                chain.functions = dwarf_functions(cudie, vaddr - bias, bias, fallback);
                chain.file_line = source_line(cudie, vaddr - bias);
            }
            if(chain.functions.empty() && fallback != nullptr)
            {
                chain.functions.push_back(symbol_name::demangle(fallback->name));
            }
            return chain;
        }

    private:
        /** Whether @p cudie's own ranges contain @p dwarf_addr, or it has none. */
        [[nodiscard]] static bool cu_covers(Dwarf_Die* cudie, Dwarf_Addr dwarf_addr)
        {
            Dwarf_Addr base  = 0;
            Dwarf_Addr start = 0;
            Dwarf_Addr end   = 0;
            if(dwarf_ranges(cudie, 0, &base, &start, &end) <= 0) return true;
            return dwarf_haspc(cudie, dwarf_addr) == 1;
        }

        /** Function scopes containing @p dwarf_addr, innermost first: the
         *  innermost scope, then its physical parents (callers of inlined
         *  code), as elfutils' own addr2line -i walks them. */
        [[nodiscard]] std::vector<std::string> dwarf_functions(
            Dwarf_Die* cudie, Dwarf_Addr dwarf_addr, Dwarf_Addr bias,
            const fallback_symbol* fallback) const
        {
            std::vector<std::string> names;
            Dwarf_Die                innermost{};
            if(!innermost_scope(cudie, dwarf_addr, innermost)) return names;

            Dwarf_Die*        raw   = nullptr;
            const int         count = dwarf_getscopes_die(&innermost, &raw);
            const scope_array scopes{ raw, &std::free };
            if(count <= 0) return names;
            for(Dwarf_Die& scope :
                std::span{ scopes.get(), static_cast<std::size_t>(count) })
            {
                const int tag = dwarf_tag(&scope);
                if(tag != DW_TAG_subprogram && tag != DW_TAG_inlined_subroutine &&
                   tag != DW_TAG_entry_point)
                {
                    continue;
                }
                std::string name = function_name(&scope, bias, fallback);
                if(!name.empty()) names.push_back(std::move(name));
            }
            return names;
        }

        /** Scope array from dwarf_getscopes / dwarf_getscopes_die, which
         *  allocate it with malloc. */
        using scope_array = std::unique_ptr<Dwarf_Die, decltype(&std::free)>;

        /** The innermost scope containing @p dwarf_addr, read again by its
         *  offset as elfutils' addr2line does: dwarf_getscopes continues
         *  from an inlined instance into its abstract origin's scopes,
         *  while dwarf_getscopes_die on this DIE gives its physical parents.
         *  False when there is none. */
        [[nodiscard]] bool innermost_scope(Dwarf_Die* cudie, Dwarf_Addr dwarf_addr,
                                           Dwarf_Die& innermost) const
        {
            Dwarf_Die*        raw   = nullptr;
            const int         count = dwarf_getscopes(cudie, dwarf_addr, &raw);
            const scope_array scopes{ raw, &std::free };
            Dwarf_Addr        dwarf_bias = 0;
            Dwarf*            dwarf      = dwfl_module_getdwarf(m_module, &dwarf_bias);
            return count > 0 && dwarf != nullptr &&
                   dwarf_offdie(dwarf, dwarf_dieoffset(scopes.get()), &innermost) !=
                       nullptr;
        }

        /** Linkage name (through abstract origin / specification). Without
         *  one, like bfd: the fallback symbol's name if that symbol starts at
         *  the function's entry, else DW_AT_name. Demangled; empty when the
         *  DIE has no name at all. */
        [[nodiscard]] static std::string function_name(Dwarf_Die* die, Dwarf_Addr bias,
                                                       const fallback_symbol* fallback)
        {
            Dwarf_Attribute attr{};
            if(dwarf_attr_integrate(die, DW_AT_linkage_name, &attr) != nullptr ||
               dwarf_attr_integrate(die, DW_AT_MIPS_linkage_name, &attr) != nullptr)
            {
                if(const char* linkage = dwarf_formstring(&attr);
                   linkage != nullptr && *linkage != '\0')
                {
                    return symbol_name::demangle(linkage);
                }
            }
            if(fallback != nullptr)
            {
                if(const auto entry = entry_address(die);
                   entry && *entry + bias == fallback->value)
                {
                    return symbol_name::demangle(fallback->name);
                }
            }
            if(dwarf_attr_integrate(die, DW_AT_name, &attr) != nullptr)
            {
                if(const char* name = dwarf_formstring(&attr);
                   name != nullptr && *name != '\0')
                {
                    return symbol_name::demangle(name);
                }
            }
            return {};
        }

        /** Lowest address of the function: DW_AT_low_pc, else the first
         *  DW_AT_ranges entry (bfd's arange.low). */
        [[nodiscard]] static std::optional<Dwarf_Addr> entry_address(Dwarf_Die* die)
        {
            Dwarf_Addr low = 0;
            if(dwarf_lowpc(die, &low) == 0) return low;
            Dwarf_Addr base  = 0;
            Dwarf_Addr start = 0;
            Dwarf_Addr end   = 0;
            if(dwarf_ranges(die, 0, &base, &start, &end) > 0) return start;
            return std::nullopt;
        }

        /** The line-table row of @p cudie covering @p dwarf_addr, formatted
         *  like addr2line. */
        [[nodiscard]] static std::string source_line(Dwarf_Die* cudie,
                                                     Dwarf_Addr dwarf_addr)
        {
            Dwarf_Line* row = covering_row(cudie, dwarf_addr);
            if(row == nullptr) return {};
            int         lineno = 0;
            const char* file   = dwarf_linesrc(row, nullptr, nullptr);
            if(file == nullptr || dwarf_lineno(row, &lineno) != 0) return {};

            std::string out{ file };
            out += ':';
            if(lineno == 0)
            {
                out += '?';
                return out;
            }
            out += std::to_string(lineno);
            unsigned int discriminator = 0;
            if(dwarf_linediscriminator(row, &discriminator) == 0 && discriminator != 0)
            {
                out += " (discriminator ";
                out += std::to_string(discriminator);
                out += ')';
            }
            return out;
        }

        /** bfd's row for @p dwarf_addr: the last row at or below it, inside
         *  a sequence. libdw sorts rows by address with an end_sequence row
         *  first among equal addresses, so a sequence's zero-length last row
         *  (a normal row at its own end address) sorts after its end and
         *  seems to cover the addresses that follow. A row sharing its
         *  address with an end_sequence row is used only where the CU's
         *  ranges contain the address, i.e. where it can start the next
         *  sequence. nullptr when no row covers the address. */
        [[nodiscard]] static Dwarf_Line* covering_row(Dwarf_Die* cudie,
                                                      Dwarf_Addr dwarf_addr)
        {
            Dwarf_Lines* lines = nullptr;
            std::size_t  count = 0;
            if(dwarf_getsrclines(cudie, &lines, &count) != 0) return nullptr;

            const auto row_address = [lines](std::size_t index) {
                Dwarf_Addr address = 0;
                dwarf_lineaddr(dwarf_onesrcline(lines, index), &address);
                return address;
            };
            const auto is_end = [lines](std::size_t index) {
                bool end = false;
                dwarf_lineendsequence(dwarf_onesrcline(lines, index), &end);
                return end;
            };

            // First row above the address; the row before it is the candidate.
            std::size_t low  = 0;
            std::size_t high = count;
            while(low < high)
            {
                const std::size_t mid = low + ((high - low) / 2);
                if(row_address(mid) <= dwarf_addr)
                {
                    low = mid + 1;
                }
                else
                {
                    high = mid;
                }
            }
            if(low == 0) return nullptr;
            const std::size_t found = low - 1;
            if(is_end(found)) return nullptr;

            const Dwarf_Addr found_address = row_address(found);
            for(std::size_t i = found; i > 0 && row_address(i - 1) == found_address; --i)
            {
                if(is_end(i - 1))
                {
                    return cu_covers(cudie, dwarf_addr) ? dwarf_onesrcline(lines, found)
                                                        : nullptr;
                }
            }
            return dwarf_onesrcline(lines, found);
        }

        /** bfd's no-DWARF function lookup (_bfd_elf_find_function); nullptr
         *  when no symbol qualifies. */
        [[nodiscard]] const fallback_symbol* fallback_symbol_for(std::uint64_t vaddr)
        {
            load_fallback_tables();

            const alloc_section* section = nullptr;
            for(const auto& sec : m_sections)
            {
                if(vaddr >= sec.addr && vaddr < sec.addr + sec.size)
                {
                    section = &sec;
                    break;
                }
            }
            if(section == nullptr) return nullptr;

            const fallback_symbol* best      = nullptr;
            std::uint64_t          best_off  = 0;
            std::uint64_t          best_size = 0;
            for(const auto& sym : m_symbols)
            {
                // bfd's _bfd_elf_maybe_function_sym.
                if(sym.type == STT_SECTION || sym.type == STT_FILE ||
                   sym.type == STT_OBJECT || sym.type == STT_TLS ||
                   sym.shndx != section->index)
                {
                    continue;
                }
                if(sym.size == 0 && sym.bind == STB_LOCAL && sym.type == STT_NOTYPE &&
                   sym.visibility == STV_HIDDEN)
                {
                    continue;
                }
                const std::uint64_t code_off  = sym.value;
                const std::uint64_t code_size = sym.size != 0 ? sym.size : 1;
                if(better_fit(best, best_off, best_size, sym, code_off, code_size, vaddr))
                {
                    best      = &sym;
                    best_off  = code_off;
                    best_size = code_size;
                }
                else if(code_off > vaddr && code_off > best_off &&
                        code_off < best_off + best_size)
                {
                    best_size = code_off - best_off;
                }
            }
            return best;
        }

        [[nodiscard]] static bool is_function(const fallback_symbol& sym) noexcept
        {
            return sym.type == STT_FUNC || sym.type == STT_GNU_IFUNC;
        }

        /** bfd's better_fit (bfd/dwarf2.c). */
        [[nodiscard]] static bool better_fit(
            const fallback_symbol* best, std::uint64_t best_off, std::uint64_t best_size,
            const fallback_symbol& sym, std::uint64_t code_off, std::uint64_t code_size,
            std::uint64_t offset) noexcept
        {
            if(code_off > offset) return false;
            if(code_off < best_off) return false;
            if(code_off > best_off) return true;
            if(best_off + best_size <= offset) return code_size > best_size;
            if(code_off + code_size <= offset) return false;
            // Both cover the offset (best is set: its size is non-zero here).
            if(is_function(*best) && !is_function(sym)) return false;
            if(is_function(sym) && !is_function(*best)) return true;
            if(best->type == STT_NOTYPE && sym.type != STT_NOTYPE) return true;
            if(best->type != STT_NOTYPE && sym.type == STT_NOTYPE) return false;
            return code_size < best_size;
        }

        /** Sections and the symbol table addr2line would use (.symtab, else
         *  .dynsym) of the main file, read once. */
        void load_fallback_tables()
        {
            if(m_tables_loaded) return;
            m_tables_loaded = true;

            GElf_Addr bias = 0;
            Elf*      elf  = dwfl_module_getelf(m_module, &bias);
            if(elf == nullptr) return;

            Elf_Scn* symtab = nullptr;
            Elf_Scn* dynsym = nullptr;
            for(Elf_Scn* scn = elf_nextscn(elf, nullptr); scn != nullptr;
                scn          = elf_nextscn(elf, scn))
            {
                GElf_Shdr shdr{};
                if(gelf_getshdr(scn, &shdr) == nullptr) continue;
                if((shdr.sh_flags & SHF_ALLOC) != 0)
                {
                    m_sections.push_back(alloc_section{
                        .index = elf_ndxscn(scn),
                        .addr  = shdr.sh_addr,
                        .size  = shdr.sh_size,
                    });
                }
                if(shdr.sh_type == SHT_SYMTAB) symtab = scn;
                if(shdr.sh_type == SHT_DYNSYM) dynsym = scn;
            }
            read_symbols(elf, symtab);
            if(m_symbols.empty()) read_symbols(elf, dynsym);
        }

        void read_symbols(Elf* elf, Elf_Scn* scn)
        {
            if(scn == nullptr) return;
            GElf_Shdr shdr{};
            if(gelf_getshdr(scn, &shdr) == nullptr || shdr.sh_entsize == 0) return;
            Elf_Data* data = elf_getdata(scn, nullptr);
            if(data == nullptr) return;
            const std::size_t count = shdr.sh_size / shdr.sh_entsize;
            // Index 0 is the null symbol; bfd skips it too.
            for(std::size_t i = 1; i < count; ++i)
            {
                GElf_Sym sym{};
                if(gelf_getsym(data, static_cast<int>(i), &sym) == nullptr) continue;
                const char* name = elf_strptr(elf, shdr.sh_link, sym.st_name);
                m_symbols.push_back(fallback_symbol{
                    .value = sym.st_value,
                    .size  = sym.st_size,
                    .shndx = sym.st_shndx,
                    .type  = static_cast<unsigned char>(GELF_ST_TYPE(sym.st_info)),
                    .bind  = static_cast<unsigned char>(GELF_ST_BIND(sym.st_info)),
                    .visibility =
                        static_cast<unsigned char>(GELF_ST_VISIBILITY(sym.st_other)),
                    .name = name != nullptr ? name : "",
                });
            }
        }

        std::unique_ptr<Dwfl, dwfl_deleter> m_dwfl;
        Dwfl_Module*                        m_module        = nullptr;
        bool                                m_tables_loaded = false;
        std::vector<alloc_section>          m_sections;
        std::vector<fallback_symbol>        m_symbols;
    };

    file_session& session_for(std::string_view path)
    {
        const std::string key{ path };
        auto              found = m_sessions.find(key);
        if(found == m_sessions.end())
        {
            found = m_sessions.emplace(key, std::make_unique<file_session>(key)).first;
        }
        return *found->second;
    }

    std::unordered_map<std::string, std::unique_ptr<file_session>> m_sessions;
};

}  // namespace cpu_snap::processing::symbols
