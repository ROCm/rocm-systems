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

#pragma once

#include <elfio/elfio.hpp>

#include <cstdint>
#include <functional>
#include <map>
#include <ostream>
#include <set>
#include <sstream>
#include <string>

namespace rocprofiler
{
namespace common
{
namespace elf_utils
{
using Section = ELFIO::section;
using Segment = ELFIO::segment;

struct SymbolEntry
{
    using accessor_type = ELFIO::symbol_section_accessor;

    SymbolEntry(unsigned int _idx, const accessor_type& _accessor);

    unsigned int      index         = 0;
    std::string       name          = {};
    ELFIO::Elf64_Addr value         = {};
    ELFIO::Elf_Xword  size          = {};
    unsigned char     bind          = {};
    unsigned char     type          = {};
    ELFIO::Elf_Half   section_index = {};
    unsigned char     other         = {};
};

struct DynamicEntry
{
    using accessor_type = ELFIO::dynamic_section_accessor;

    DynamicEntry(unsigned int _idx, const accessor_type& _accessor);

    unsigned int     index = 0;
    std::string      name  = {};
    ELFIO::Elf_Xword tag   = {};
    ELFIO::Elf_Xword value = {};
};

struct RelocationEntry
{
    using accessor_type = ELFIO::relocation_section_accessor;

    RelocationEntry(unsigned int _idx, const accessor_type& _accessor);

    unsigned int      index  = 0;
    ELFIO::Elf64_Addr offset = {};
    ELFIO::Elf_Word   symbol = {};
    ELFIO::Elf_Word   type   = {};
    ELFIO::Elf_Sxword addend = {};
};

struct ElfInfo
{
    explicit ElfInfo(std::string);

    std::string                  filename               = {};
    ELFIO::elfio                 reader                 = {};
    std::vector<Section*>        sections               = {};
    std::vector<SymbolEntry>     symbol_entries         = {};
    std::vector<SymbolEntry>     dynamic_symbol_entries = {};
    std::vector<DynamicEntry>    dynamic_entries        = {};
    std::vector<RelocationEntry> reloc_entries          = {};

    bool has_symbol(const std::function<bool(std::string_view)>&) const;

    friend bool operator==(const ElfInfo& lhs, const ElfInfo& rhs)
    {
        return (lhs.filename == rhs.filename);
    }

    friend bool operator<(const ElfInfo& lhs, const ElfInfo& rhs)
    {
        return (lhs.filename < rhs.filename);
    }

    friend bool operator>(const ElfInfo& lhs, const ElfInfo& rhs)
    {
        return !(lhs == rhs || lhs < rhs);
    }

    friend bool operator<=(const ElfInfo& lhs, const ElfInfo& rhs) { return !(lhs > rhs); }
    friend bool operator>=(const ElfInfo& lhs, const ElfInfo& rhs) { return !(lhs < rhs); }
};

ElfInfo
read(const std::string& _inp);
}  // namespace elf_utils
}  // namespace common
}  // namespace rocprofiler
