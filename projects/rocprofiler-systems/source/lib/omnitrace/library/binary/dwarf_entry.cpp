// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "library/binary/dwarf_entry.hpp"
#include "library/binary/fwd.hpp"
#include "library/timemory.hpp"
#include "library/utility.hpp"

#include <dwarf.h>
#include <elfutils/libdw.h>

namespace omnitrace
{
namespace binary
{
namespace
{
using utility::combine;

auto
get_dwarf_address_ranges(Dwarf_Die* _die)
{
    auto _ranges = std::vector<address_range>{};

    if(dwarf_tag(_die) != DW_TAG_compile_unit) return _ranges;

    Dwarf_Addr _low_pc;
    Dwarf_Addr _high_pc;
    dwarf_lowpc(_die, &_low_pc);
    dwarf_highpc(_die, &_high_pc);

    _ranges.emplace_back(address_range{ _low_pc, _high_pc });

    Dwarf_Addr _base_addr;
    ptrdiff_t  _offset = 0;
    do
    {
        _ranges.emplace_back(address_range{ 0, 0 });
    } while((_offset = dwarf_ranges(_die, _offset, &_base_addr, &_ranges.back().low,
                                    &_ranges.back().high)) > 0);
    // will always have one extra
    _ranges.pop_back();

    return _ranges;
}

auto
get_dwarf_entry(Dwarf_Die* _die)
{
    auto _line_info = std::deque<dwarf_entry>{};

    if(dwarf_tag(_die) != DW_TAG_compile_unit) return _line_info;

    Dwarf_Lines* _lines     = nullptr;
    size_t       _num_lines = 0;
    if(dwarf_getsrclines(_die, &_lines, &_num_lines) == 0)
    {
        _line_info.resize(_num_lines);
        for(size_t j = 0; j < _num_lines; ++j)
        {
            auto& itr   = _line_info.at(j);
            auto* _line = dwarf_onesrcline(_lines, j);
            if(_line)
            {
                int       _lineno  = 0;
                uintptr_t _address = 0;
                dwarf_lineno(_line, &_lineno);
                dwarf_linecol(_line, &itr.col);
                dwarf_linebeginstatement(_line, &itr.begin_statement);
                dwarf_lineendsequence(_line, &itr.end_sequence);
                dwarf_lineblock(_line, &itr.line_block);
                dwarf_lineepiloguebegin(_line, &itr.epilogue_begin);
                dwarf_lineprologueend(_line, &itr.prologue_end);
                dwarf_lineisa(_line, &itr.isa);
                dwarf_linediscriminator(_line, &itr.discriminator);
                dwarf_lineaddr(_line, &_address);
                itr.address = address_range{ _address };
                if(_lineno > 0) itr.line = _lineno;
                const auto* _file = dwarf_linesrc(_line, nullptr, nullptr);
                if(!_file) _file = dwarf_diename(_die);
                itr.file = filepath::realpath(_file, nullptr, false);
            }
        }
    }

    return _line_info;
}
}  // namespace

bool
dwarf_entry::operator<(const dwarf_entry& _rhs) const
{
    return std::tie(address, line, col, discriminator) <
           std::tie(_rhs.address, _rhs.line, _rhs.col, _rhs.discriminator);
}

bool
dwarf_entry::operator==(const dwarf_entry& _rhs) const
{
    return std::tie(address, line, col, discriminator, vliw_op_index, isa, file) ==
           std::tie(_rhs.address, _rhs.line, _rhs.col, _rhs.discriminator,
                    _rhs.vliw_op_index, _rhs.isa, _rhs.file);
}

bool
dwarf_entry::operator!=(const dwarf_entry& _rhs) const
{
    return !(*this == _rhs);
}

bool
dwarf_entry::is_valid() const
{
    return (*this != dwarf_entry{} && !file.empty());
}

std::deque<dwarf_entry>
dwarf_entry::process_dwarf(int _fd, std::vector<address_range>& _ranges)
{
    auto* _dwarf_v   = dwarf_begin(_fd, DWARF_C_READ);
    auto  _line_info = std::deque<dwarf_entry>{};

    size_t    cu_header_size = 0;
    Dwarf_Off cu_off         = 0;
    Dwarf_Off next_cu_off    = 0;
    for(; dwarf_nextcu(_dwarf_v, cu_off, &next_cu_off, &cu_header_size, nullptr, nullptr,
                       nullptr) == 0;
        cu_off = next_cu_off)
    {
        Dwarf_Off cu_die_off = cu_off + cu_header_size;
        Dwarf_Die cu_die;
        if(dwarf_offdie(_dwarf_v, cu_die_off, &cu_die) != nullptr)
        {
            Dwarf_Die* _die = &cu_die;
            if(dwarf_tag(_die) == DW_TAG_compile_unit)
            {
                combine(_line_info, get_dwarf_entry(_die));
                combine(_ranges, get_dwarf_address_ranges(_die));
            }
        }
    }

    dwarf_end(_dwarf_v);
    utility::filter_sort_unique(_line_info);
    utility::filter_sort_unique(_ranges);

    return _line_info;
}

template <typename ArchiveT>
void
dwarf_entry::serialize(ArchiveT& ar, const unsigned int)
{
#define OMNITRACE_SERIALIZE_MEMBER(MEMBER) ar(::tim::cereal::make_nvp(#MEMBER, MEMBER));

    OMNITRACE_SERIALIZE_MEMBER(file)
    OMNITRACE_SERIALIZE_MEMBER(line)
    OMNITRACE_SERIALIZE_MEMBER(col)
    OMNITRACE_SERIALIZE_MEMBER(address)
    OMNITRACE_SERIALIZE_MEMBER(discriminator)
    // OMNITRACE_SERIALIZE_MEMBER(begin_statement)
    // OMNITRACE_SERIALIZE_MEMBER(end_sequence)
    // OMNITRACE_SERIALIZE_MEMBER(line_block)
    // OMNITRACE_SERIALIZE_MEMBER(prologue_end)
    // OMNITRACE_SERIALIZE_MEMBER(epilogue_begin)
    // OMNITRACE_SERIALIZE_MEMBER(vliw_op_index)
    // OMNITRACE_SERIALIZE_MEMBER(isa)
}

template void
dwarf_entry::serialize<cereal::JSONInputArchive>(cereal::JSONInputArchive&,
                                                 const unsigned int);

template void
dwarf_entry::serialize<cereal::MinimalJSONOutputArchive>(
    cereal::MinimalJSONOutputArchive&, const unsigned int);

template void
dwarf_entry::serialize<cereal::PrettyJSONOutputArchive>(cereal::PrettyJSONOutputArchive&,
                                                        const unsigned int);
}  // namespace binary
}  // namespace omnitrace
