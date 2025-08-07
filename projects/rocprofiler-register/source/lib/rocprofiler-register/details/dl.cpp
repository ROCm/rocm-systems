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

#define GNU_SOURCE 1

#include "dl.hpp"
#include "filesystem.hpp"
#include "utility.hpp"

#include <fstream>
#include <optional>
#include <string>
#include <string_view>

#include <dlfcn.h>
#include <elf.h>
#include <fmt/core.h>
#include <link.h>
#include <sys/types.h>
#include <unistd.h>

namespace rocprofiler_register
{
namespace binary
{
namespace
{
const open_modes_vec_t default_link_open_modes = { (RTLD_LAZY | RTLD_NOLOAD) };
}  // namespace

std::vector<segment_address_ranges>
get_segment_addresses(pid_t _pid)
{
    auto _data  = std::vector<segment_address_ranges>{};
    auto _fname = fmt::format("/{}/{}/{}", "proc", _pid, "maps");
    auto ifs    = std::ifstream{ _fname };
    if(!ifs)
    {
        fprintf(stderr, "Failure opening %s\n", _fname.c_str());
    }
    else
    {
        auto _get_entry = [&_data](std::string_view _name) -> segment_address_ranges& {
            for(auto& itr : _data)
            {
                if(itr.filepath == _name) return itr;
            }
            return _data.emplace_back(
                segment_address_ranges{ .filepath = std::string{ _name } });
        };

        while(ifs)
        {
            std::string _line = {};
            if(std::getline(ifs, _line) && !_line.empty())
            {
                auto _delim = utility::delimit(_line, " \t\n\r");
                if(_delim.size() > 5 && fs::exists(fs::path{ _delim.back() }))
                {
                    auto& _entry       = _get_entry(_delim.back());
                    auto  _addr        = utility::delimit(_delim.front(), "-");
                    auto  load_address = std::stoull(_addr.front(), nullptr, 16);
                    auto  last_address = std::stoull(_addr.back(), nullptr, 16);
                    _entry.ranges.emplace_back(
                        address_range{ load_address, last_address });
                }
            }
        }
    }
    return _data;
}

std::optional<std::string>
get_linked_path(std::string_view _name, open_modes_vec_t&& _open_modes)
{
    if(_name.empty()) return fs::current_path().string();

    if(_open_modes.empty()) _open_modes = default_link_open_modes;

    void* _handle = nullptr;
    bool  _noload = false;
    for(auto _mode : _open_modes)
    {
        _handle = dlopen(_name.data(), _mode);
        _noload = (_mode & RTLD_NOLOAD) == RTLD_NOLOAD;
        if(_handle) break;
    }

    if(_handle)
    {
        struct link_map* _link_map = nullptr;
        dlinfo(_handle, RTLD_DI_LINKMAP, &_link_map);
        if(_link_map != nullptr && !std::string_view{ _link_map->l_name }.empty())
        {
            return fs::absolute(fs::path{ _link_map->l_name }).string();
        }
        if(_noload == false) dlclose(_handle);
    }

    return std::nullopt;
}
}  // namespace binary
}  // namespace rocprofiler_register
