// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "core/config.hpp"
#include "core/debug.hpp"

#if !defined(TIMEMORY_USE_BFD)
#    error "BFD support not enabled"
#endif

#define PACKAGE     "rocprofiler-systems"
#define L_LNNO_SIZE 4

#include <bfd.h>
#include <coff/external.h>
#include <coff/internal.h>
#include <cstddef>
#include <cstdio>
#include <dwarf.h>
#include <elf-bfd.h>
#include <elfutils/libdw.h>
#include <libcoff.h>

#include "core/binary/fwd.hpp"
#include "core/timemory.hpp"
#include "core/utility.hpp"
#include "dwarf_entry.hpp"
#include "scope_filter.hpp"
#include "symbol.hpp"

#include <timemory/mpl/concepts.hpp>

namespace rocprofsys
{
namespace binary
{
namespace
{
std::vector<inlined_symbol>
read_inliner_info(bfd* _inp)
{
    auto _data = std::vector<inlined_symbol>{};
    while(true)
    {
        const char*  _file = nullptr;
        const char*  _func = nullptr;
        unsigned int _line = 0;
        if(bfd_find_inliner_info(_inp, &_file, &_func, &_line) != 0)
        {
            if(_file && _func && _line > 0)
                _data.emplace_back(inlined_symbol{
                    _line, filepath::realpath(_file, nullptr, false), _func });
        }
        else
        {
            break;
        }
    }

    return _data;
}
}  // namespace

symbol::symbol(const base_type& _v)
: base_type{ _v }
, address{ _v.address, _v.address + _v.symsize + 1 }
, func{ std::string{ base_type::name } }
{}

bool
symbol::operator==(const symbol& _rhs) const
{
    return std::tie(address, base_type::name) ==
           std::tie(_rhs.address, _rhs.base_type::name);
}

bool
symbol::operator<(const symbol& _rhs) const
{
    // if both have non-zero load addresses that are not equal, compare based on load
    // addresses
    if(load_address > 0 && _rhs.load_address > 0 && load_address != _rhs.load_address)
        return (load_address < _rhs.load_address);

    // if address is same and name is same, return true if load_address is higher
    if(address == _rhs.address && base_type::name == _rhs.base_type::name)
        return load_address > _rhs.load_address;

    return std::tie(address, base_type::binding, base_type::visibility, base_type::name) <
           std::tie(_rhs.address, _rhs.base_type::binding, base_type::visibility,
                    base_type::name);
}

bool
symbol::operator()(const std::vector<scope_filter>& _filters) const
{
    using sf = scope_filter;

    // apply filters to the main symbol
    return (sf::satisfies_filter(_filters, sf::FUNCTION_FILTER, demangle(func)) &&
            (sf::satisfies_filter(_filters, sf::SOURCE_FILTER, file) ||
             sf::satisfies_filter(_filters, sf::SOURCE_FILTER, join(':', file, line))));
}

symbol&
symbol::operator+=(const symbol& _rhs)
{
    if(address.contiguous_with(_rhs.address) &&
       std::tie(line, load_address, func, file) ==
           std::tie(_rhs.line, _rhs.load_address, _rhs.func, _rhs.file))
    {
        address += _rhs.address;
        utility::combine(inlines, _rhs.inlines);
        utility::combine(dwarf_info, _rhs.dwarf_info);
        if(_rhs.binding < binding) binding = _rhs.binding;
        if(_rhs.visibility < visibility) visibility = _rhs.visibility;
        if(load_address == 0 && _rhs.load_address > load_address)
            load_address = _rhs.load_address;
    }
    else
    {
        throw exception<std::runtime_error>("incompatible symbol+=");
    }

    return *this;
}

symbol::operator bool() const
{
    return address.is_valid() && (file.length() + func.length() + line) > 0;
}

size_t
symbol::read_dwarf_entries(const std::deque<dwarf_entry>& _info)
{
    for(const auto& itr : _info)
    {
        if(address.contains(itr.address)) dwarf_info.emplace_back(itr);
    }

    // make sure the dwarf info is sorted by address (low to high)
    std::sort(dwarf_info.begin(), dwarf_info.end(),
              [](const dwarf_entry& _lhs, const dwarf_entry& _rhs) {
                  return _lhs.address < _rhs.address;
              });

    // helper for getting the end address
    auto _get_next_address = [&](auto nitr, uintptr_t _low) {
        while(++nitr != dwarf_info.end())
        {
            if(nitr->address.low > _low)
            {
                return nitr->address.low;
            }
        }
        // return the end address of the symbol
        return address.high;
    };
    // convert the single addresses into ranges
    for(auto itr = dwarf_info.begin(); itr != dwarf_info.end(); ++itr)
    {
        // if address is already a range, do not update it
        if(!itr->address.is_range())
            itr->address = address_range{ itr->address.low,
                                          _get_next_address(itr, itr->address.low) };
    }

    std::sort(dwarf_info.begin(), dwarf_info.end(),
              [](const auto& _lhs, const auto& _rhs) {
                  return std::tie(_lhs.address, _lhs.file, _lhs.line, _lhs.col) <
                         std::tie(_rhs.address, _rhs.file, _rhs.line, _rhs.col);
              });

    dwarf_info.erase(std::unique(dwarf_info.begin(), dwarf_info.end(),
                                 [](const auto& _lhs, const auto& _rhs) {
                                     return std::tie(_lhs.address, _lhs.file,
                                                     _lhs.line) ==
                                            std::tie(_rhs.address, _rhs.file, _rhs.line);
                                 }),
                     dwarf_info.end());

    return dwarf_info.size();
}

size_t
symbol::read_dwarf_breakpoints(const std::vector<uintptr_t>& _bkpts)
{
    for(const auto& itr : _bkpts)
    {
        if(address.contains(itr)) breakpoints.emplace_back(itr);
    }

    // make sure the breakpoints are sorted low to high
    std::sort(breakpoints.begin(), breakpoints.end());

    return breakpoints.size();
}

bool
symbol::read_bfd_line_info(bfd_file& _bfd)
{
    auto*         _section = static_cast<asection*>(section);
    bfd_vma       _vma     = bfd_section_vma(_section);
    bfd_size_type _size    = bfd_section_size(_section);

    auto& _pc     = address.low;
    auto& _pc_end = address.high;

    if(_pc < _vma || _pc >= _vma + _size) return false;
    // add one to vma + size because address range is exclusive of last address
    if(_pc_end > _vma + _size) _pc_end = (_vma + _size);

    auto* _inp  = static_cast<bfd*>(_bfd.data);
    auto* _syms = reinterpret_cast<asymbol**>(_bfd.syms);

    {
        const char*  _file = nullptr;
        const char*  _func = nullptr;
        unsigned int _line = 0;

        if(bfd_find_nearest_line(_inp, _section, _syms, _pc - _vma, &_file, &_func,
                                 &_line) != 0)
        {
            if(_file) file = _file;
            if(_func) func = _func;
            if(_file && strnlen(_file, 1) > 0)
                file = _file;
            else if(!_file || strnlen(_file, 1) == 0)
                file = bfd_get_filename(_inp);
            if(!func.empty())
            {
                file    = filepath::realpath(file, nullptr, false);
                line    = _line;
                inlines = read_inliner_info(_inp);
                return true;
            }
        }
    }

    return false;
}

symbol
symbol::clone() const
{
    auto _sym         = symbol{ static_cast<base_type>(*this) };
    _sym.line         = line;
    _sym.load_address = load_address;
    _sym.address      = address;
    _sym.func         = func;
    _sym.file         = file;

    return _sym;
}

template <typename Tp>
Tp
symbol::get_inline_symbols(const std::vector<scope_filter>& _filters) const
{
    using sf         = scope_filter;
    using value_type = typename Tp::value_type;

    auto _data = Tp{};

    for(const auto& itr : inlines)
    {
        if(sf::satisfies_filter(_filters, sf::FUNCTION_FILTER, demangle(itr.func)) &&
           (sf::satisfies_filter(_filters, sf::SOURCE_FILTER, itr.file) ||
            sf::satisfies_filter(_filters, sf::SOURCE_FILTER,
                                 join(':', itr.file, itr.line))))
        {
            if constexpr(concepts::is_unqualified_same<value_type, symbol>::value)
            {
                auto _sym = clone();
                _sym.func = itr.func;
                _sym.line = itr.line;
                _sym.file = itr.file;
                _data.emplace_back(_sym);
            }
            else if constexpr(concepts::is_unqualified_same<value_type,
                                                            inlined_symbol>::value)
            {
                _data.emplace_back(itr);
            }
        }
    }

    return _data;
}

template <typename Tp>
Tp
symbol::get_debug_line_info(const std::vector<scope_filter>& _filters) const
{
    using sf         = scope_filter;
    using value_type = typename Tp::value_type;

    auto _data = Tp{};

    if(sf::satisfies_filter(_filters, sf::FUNCTION_FILTER, demangle(func)))
    {
        for(const auto& itr : dwarf_info)
        {
            if(sf::satisfies_filter(_filters, sf::SOURCE_FILTER, itr.file) ||
               sf::satisfies_filter(_filters, sf::SOURCE_FILTER,
                                    join(':', itr.file, itr.line)))
            {
                if constexpr(concepts::is_unqualified_same<value_type, symbol>::value)
                {
                    auto _sym    = clone();
                    _sym.address = itr.address;
                    _sym.file    = itr.file;
                    _sym.line    = itr.line;
                    _data.emplace_back(_sym);
                }
                else if constexpr(concepts::is_unqualified_same<value_type,
                                                                dwarf_entry>::value)
                {
                    _data.emplace_back(itr);
                }
            }
        }
    }

    return _data;
}

template <typename ArchiveT>
void
inlined_symbol::serialize(ArchiveT& ar, const unsigned int)
{
    using ::tim::cereal::make_nvp;
    ar(make_nvp("func", func), make_nvp("file", file), make_nvp("line", line));
}

template <typename ArchiveT>
void
symbol::serialize(ArchiveT& ar, const unsigned int)
{
    using ::tim::cereal::make_nvp;
    ar(make_nvp("address", address), make_nvp("load_address", load_address),
       make_nvp("line", line), make_nvp("func", func), make_nvp("file", file),
       make_nvp("inlines", inlines), make_nvp("dwarf_info", dwarf_info));
    if constexpr(concepts::is_output_archive<ArchiveT>::value)
        ar(cereal::make_nvp("dfunc", demangle(func)));
}

template void
inlined_symbol::serialize<cereal::JSONInputArchive>(cereal::JSONInputArchive&,
                                                    const unsigned int);

template void
inlined_symbol::serialize<cereal::MinimalJSONOutputArchive>(
    cereal::MinimalJSONOutputArchive&, const unsigned int);

template void
inlined_symbol::serialize<cereal::PrettyJSONOutputArchive>(
    cereal::PrettyJSONOutputArchive&, const unsigned int);

template void
symbol::serialize<cereal::JSONInputArchive>(cereal::JSONInputArchive&,
                                            const unsigned int);

template void
symbol::serialize<cereal::MinimalJSONOutputArchive>(cereal::MinimalJSONOutputArchive&,
                                                    const unsigned int);

template void
symbol::serialize<cereal::PrettyJSONOutputArchive>(cereal::PrettyJSONOutputArchive&,
                                                   const unsigned int);

template std::deque<symbol>
symbol::get_inline_symbols<std::deque<symbol>>(
    const std::vector<scope_filter>& _filters) const;

template std::vector<inlined_symbol>
symbol::get_inline_symbols<std::vector<inlined_symbol>>(
    const std::vector<scope_filter>& _filters) const;

template std::deque<symbol>
symbol::get_debug_line_info<std::deque<symbol>>(
    const std::vector<scope_filter>& _filters) const;

template std::vector<dwarf_entry>
symbol::get_debug_line_info<std::vector<dwarf_entry>>(
    const std::vector<scope_filter>& _filters) const;
}  // namespace binary
}  // namespace rocprofsys
