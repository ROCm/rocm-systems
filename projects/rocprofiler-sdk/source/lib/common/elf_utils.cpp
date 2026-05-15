// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/common/elf_utils.hpp"

#include <rocprofiler-sdk/cxx/utility.hpp>

#include <fmt/format.h>
#include <elfio/elfio.hpp>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>
#include "lib/common/logging.hpp"

namespace rocprofiler
{
namespace common
{
namespace elf_utils
{
namespace
{
const ELFIO::Elf_Xword PAGE_SIZE = sysconf(_SC_PAGESIZE);

using ::rocprofiler::sdk::utility::as_hex;
}  // namespace

SymbolEntry::SymbolEntry(unsigned int _idx, const accessor_type& _accessor)
: index{_idx}
{
    if(!_accessor.get_symbol(index, name, value, size, bind, type, section_index, other))
        ROCP_WARNING << "ELFIO::symbol_section_accessor::get_symbol failed of symbol " << _idx;
}

DynamicEntry::DynamicEntry(unsigned int _idx, const accessor_type& _accessor)
: index{_idx}
{
    if(!_accessor.get_entry(_idx, tag, value, name)) return;
}

RelocationEntry::RelocationEntry(unsigned int _idx, const accessor_type& _accessor)
: index{_idx}
{
    if(!_accessor.get_entry(_idx, offset, symbol, type, addend))
        ROCP_WARNING << "ELFIO::relocation_section_accessor::get_entry failed for symbol " << _idx;
}

ElfInfo::ElfInfo(std::string _fname)
: filename{std::move(_fname)}
{}

bool
ElfInfo::has_symbol(const std::function<bool(std::string_view)>& _checker) const
{
    ROCP_INFO << fmt::format("{} has {} symbols ({} .dynsym + {} .symtab)",
                             filename,
                             dynamic_symbol_entries.size() + symbol_entries.size(),
                             dynamic_symbol_entries.size(),
                             symbol_entries.size());

    // Search publicly visible symbols first
    for(const auto& itr : dynamic_symbol_entries)
    {
        if(!itr.name.empty() && _checker(itr.name)) return true;
    }

    // Search all other symbols (may be empty if optimize_for_visible_symbols + non-empty .dynsym)
    for(const auto& itr : symbol_entries)
    {
        if(!itr.name.empty() && _checker(itr.name)) return true;
    }

    return false;
}

std::string
ElfInfo::as_string() const
{
    auto _oss = std::ostringstream{};

    _oss << "ELF File: " << filename << "\n";
    if(reader.get_class() == ELFIO::ELFCLASS32)
        _oss << "  - ELF 32-bit\n";
    else
        _oss << "  - ELF 64-bit\n";

    _oss << "  - ELF file encoding: "
         << ((reader.get_encoding() == ELFIO::ELFDATA2LSB) ? std::string_view{"Little endian"}
                                                           : std::string_view{"Big endian"})
         << "\n";

    _oss << "  - ELF version: " << reader.get_elf_version() << "\n";
    _oss << "  - ELF header size: " << reader.get_header_size() << "\n";
    _oss << "  - ELF OS ABI: " << reader.get_os_abi() << "\n";
    _oss << "  - Number of sections: " << reader.sections.size() << "\n";

    _oss << fmt::format("  - Symbols ({}):\n", symbol_entries.size());
    for(size_t k = 0; k < symbol_entries.size(); ++k)
    {
        if(!symbol_entries.at(k).name.empty())
            _oss << "          [" << k << "] " << symbol_entries.at(k).name << "\n";
    }

    _oss << fmt::format("  - Dynamic Symbols ({}):\n", dynamic_symbol_entries.size());
    for(size_t k = 0; k < dynamic_symbol_entries.size(); ++k)
    {
        if(!dynamic_symbol_entries.at(k).name.empty())
            _oss << "          [" << k << "] " << dynamic_symbol_entries.at(k).name << "\n";
    }

    _oss << fmt::format("  - Dynamic entries ({}):\n", dynamic_entries.size());
    for(size_t k = 0; k < dynamic_entries.size(); ++k)
    {
        if(!dynamic_entries.at(k).name.empty())
            _oss << "          [" << k << "] " << dynamic_entries.at(k).name << "\n";
    }

    _oss << fmt::format("  - Relocation entries ({}):\n", reloc_entries.size());
    for(size_t k = 0; k < reloc_entries.size(); ++k)
    {
        auto _sym_idx = reloc_entries.at(k).symbol;
        auto _name    = std::string{};
        if(_sym_idx < symbol_entries.size()) _name = symbol_entries.at(_sym_idx).name;
        if(!_name.empty()) _oss << "          [" << k << "] " << _name << "\n";
    }

    // Print ELF file segments info
    ELFIO::Elf_Half seg_num = reader.segments.size();
    _oss << fmt::format("  - Number of segments ({}):\n", seg_num);
    for(ELFIO::Elf_Half j = 0; j < seg_num; ++j)
    {
        const ELFIO::segment* pseg = reader.segments[j];
        _oss << "      [" << std::setw(2) << j << "] flags: " << as_hex(pseg->get_flags(), 16)
             << "   offset: " << as_hex(pseg->get_offset(), 16)
             << "   align: " << as_hex(pseg->get_align(), 16)
             << "   virt: " << as_hex(pseg->get_virtual_address(), 16)
             << "   phys: " << as_hex(pseg->get_physical_address(), 16)
             << "  fsize: " << std::setw(8) << pseg->get_file_size() << "  msize: " << std::setw(8)
             << pseg->get_memory_size() << "\n";
    }

    return _oss.str();
}

ElfInfo
read(const std::string& _inp, bool optimize_for_visible_symbols)
{
    auto  _info                  = ElfInfo{_inp};
    auto& reader                 = _info.reader;
    auto& sections               = _info.sections;
    auto& symbol_entries         = _info.symbol_entries;
    auto& dynamic_symbol_entries = _info.dynamic_symbol_entries;
    auto& dynamic_entries        = _info.dynamic_entries;
    auto& reloc_entries          = _info.reloc_entries;

    ROCP_TRACE << "\nReading " << _inp;

    constexpr auto lazy_parsing = true;
    if(!reader.load(_inp, lazy_parsing))
        ROCP_WARNING << fmt::format("ELF parsing for '{}' did not succeed", _inp);

    if(reader.get_class() == ELFIO::ELFCLASS32)
        ROCP_TRACE << "  - ELF 32-bit";
    else
        ROCP_TRACE << "  - ELF 64-bit";

    ROCP_TRACE << "  - ELF file encoding: "
               << ((reader.get_encoding() == ELFIO::ELFDATA2LSB) ? std::string_view{"Little endian"}
                                                                 : std::string_view{"Big endian"});

    ROCP_TRACE << "  - ELF version: " << reader.get_elf_version();
    ROCP_TRACE << "  - ELF header size: " << reader.get_header_size();
    ROCP_TRACE << "  - ELF OS ABI: " << reader.get_os_abi();

    // Print ELF file sections info
    ELFIO::Elf_Half sec_num = reader.sections.size();
    ROCP_TRACE << "  - Number of sections: " << sec_num;

    for(ELFIO::Elf_Half j = 0; j < sec_num; ++j)
    {
        ELFIO::section* psec = reader.sections[j];
        sections.emplace_back(psec);
    }

    std::sort(sections.begin(), sections.end(), [](const Section* lhs, const Section* rhs) {
        return std::string_view{lhs->get_name()} < std::string_view{rhs->get_name()};
    });

    bool has_dsymtab = false;
    for(auto* itr : sections)
    {
        if(itr->get_size() > 0 && itr->get_type() == ELFIO::SHT_DYNSYM)
        {
            has_dsymtab = true;
            break;
        }
    }

    for(ELFIO::Elf_Half j = 0; j < sec_num; ++j)
    {
        Section* psec = sections.at(j);
        ROCP_TRACE << "      [" << j << "] \t" << std::setw(20) << psec->get_name() << "\t : \t"
                   << "size / entry-size = " << std::setw(6) << psec->get_size() << " / "
                   << std::setw(3) << psec->get_entry_size()
                   << " | addr: " << as_hex(psec->get_address(), 16)
                   << " | offset: " << as_hex(psec->get_offset(), 16);

        if(psec->get_size() == 0) continue;

        if(psec->get_type() == ELFIO::SHT_SYMTAB)
        {
            if(optimize_for_visible_symbols && has_dsymtab)
            {
                ROCP_INFO << fmt::format("[{}] Skipping loading of .symtab since .dynsym is "
                                         "present and optimize_for_visible_symbols=true",
                                         _inp);
                continue;
            }
            const ELFIO::symbol_section_accessor _symbols(reader, psec);
            ROCP_TRACE << "  - Number of symbol entries: " << _symbols.get_symbols_num();
            for(ELFIO::Elf_Xword k = 0; k < _symbols.get_symbols_num(); ++k)
                symbol_entries.emplace_back(k, _symbols);
        }
        else if(psec->get_type() == ELFIO::SHT_DYNSYM)
        {
            const ELFIO::symbol_section_accessor _symbols(reader, psec);
            ROCP_TRACE << "  - Number of dynamic symbol entries: " << _symbols.get_symbols_num();
            for(ELFIO::Elf_Xword k = 0; k < _symbols.get_symbols_num(); ++k)
                dynamic_symbol_entries.emplace_back(k, _symbols);
        }
        else if(psec->get_type() == ELFIO::SHT_DYNAMIC)
        {
            const ELFIO::dynamic_section_accessor dynamic{reader, psec};
            ROCP_TRACE << "  - Number of dynamic entries: " << dynamic.get_entries_num();
            for(ELFIO::Elf_Xword k = 0; k < dynamic.get_entries_num(); ++k)
                dynamic_entries.emplace_back(k, dynamic);
        }
        else if(psec->get_type() == ELFIO::SHT_REL || psec->get_type() == ELFIO::SHT_RELA)
        {
            if(optimize_for_visible_symbols)
            {
                ROCP_INFO_IF(psec->get_type() == ELFIO::SHT_REL) << fmt::format(
                    "[{}] Skipping loading of .rel since optimize_for_visible_symbols=true", _inp);
                ROCP_INFO_IF(psec->get_type() == ELFIO::SHT_RELA) << fmt::format(
                    "[{}] Skipping loading of .rela since optimize_for_visible_symbols=true", _inp);
                continue;
            }
            const ELFIO::relocation_section_accessor reloc{reader, psec};
            ROCP_TRACE << "  - Number of relocation entries: " << reloc.get_entries_num();
            for(ELFIO::Elf_Xword k = 0; k < reloc.get_entries_num(); ++k)
                reloc_entries.emplace_back(k, reloc);
        }
    }

    ROCP_TRACE << _info.as_string();

    return _info;
}
}  // namespace elf_utils
}  // namespace common
}  // namespace rocprofiler
