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

#include "link_map.hpp"
#include "core/common.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "core/timemory.hpp"

#include <timemory/utility/filepath.hpp>

#include <cstdint>
#include <dlfcn.h>
#include <link.h>
#include <set>
#include <string>
#include <string_view>

namespace rocprofsys
{
namespace binary
{
namespace
{
const open_modes_vec_t default_link_open_modes = { (RTLD_LAZY | RTLD_NOLOAD),
                                                   (RTLD_LAZY | RTLD_LOCAL) };
}

std::optional<std::string>
get_linked_path(const char* _name, open_modes_vec_t&& _open_modes)
{
    if(_name == nullptr) return config::get_exe_realpath();

    if(_open_modes.empty()) _open_modes = default_link_open_modes;

    void* _handle = nullptr;
    bool  _noload = false;
    for(auto _mode : _open_modes)
    {
        _handle = dlopen(_name, _mode);
        _noload = (_mode & RTLD_NOLOAD) == RTLD_NOLOAD;
        if(_handle) break;
    }

    if(_handle)
    {
        struct link_map* _link_map = nullptr;
        dlinfo(_handle, RTLD_DI_LINKMAP, &_link_map);
        if(_link_map != nullptr && !std::string_view{ _link_map->l_name }.empty())
        {
            return filepath::realpath(_link_map->l_name, nullptr, false);
        }
        if(_noload == false) dlclose(_handle);
    }

    return std::optional<std::string>{};
}

std::set<link_file>
get_link_map(const char* _lib, const std::string& _exclude_linked_by,
             const std::string& _exclude_re, open_modes_vec_t&& _open_modes)
{
    if(_open_modes.empty()) _open_modes = default_link_open_modes;

    auto _get_chain = [&_open_modes](const char* _name) {
        void* _handle = nullptr;
        bool  _noload = false;
        for(auto _mode : _open_modes)
        {
            _handle = dlopen(_name, _mode);
            _noload = (_mode & RTLD_NOLOAD) == RTLD_NOLOAD;
            if(_handle) break;
        }

        auto _chain = std::set<std::string>{};
        if(_handle)
        {
            struct link_map* _link_map = nullptr;
            dlinfo(_handle, RTLD_DI_LINKMAP, &_link_map);
            struct link_map* _next = _link_map;
            while(_next)
            {
                if(_name == nullptr && _next == _link_map &&
                   std::string_view{ _next->l_name }.empty())
                {
                    // only insert exe name if dlopened the exe and
                    // empty name is first entry
                    _chain.emplace(config::get_exe_realpath());
                }
                else if(!std::string_view{ _next->l_name }.empty())
                {
                    _chain.emplace(_next->l_name);
                }
                _next = _next->l_next;
            }

            if(_noload == false) dlclose(_handle);
        }
        return _chain;
    };

    auto _full_chain = _get_chain(_lib);
    auto _excl_chain = (_exclude_linked_by.empty())
                           ? std::set<std::string>{}
                           : _get_chain(_exclude_linked_by.c_str());
    auto _fini_chain = std::set<link_file>{};

    for(const auto& itr : _full_chain)
    {
        if(_excl_chain.find(itr) == _excl_chain.end())
        {
            if(_exclude_re.empty() || !std::regex_search(itr, std::regex{ _exclude_re }))
                _fini_chain.emplace(itr);
            else
                _excl_chain.emplace(itr);
        }
    }

    auto _name = (!_lib) ? config::get_exe_realpath() : std::string{ _lib };
    for(const auto& itr : _fini_chain)
    {
        ROCPROFSYS_BASIC_VERBOSE(2, "[linkmap][%s]: %s\n", filepath::basename(_name),
                                 itr.real().c_str());
    }

    for(const auto& itr : _excl_chain)
    {
        ROCPROFSYS_BASIC_VERBOSE(3, "[linkmap][%s]: %s\n", _exclude_linked_by.c_str(),
                                 link_file{ itr }.real().c_str());
    }

    return _fini_chain;
}

bool
link_file::operator<(const link_file& _rhs) const
{
    if(name == _rhs.name) return false;

    auto _lhs_base = base();
    auto _lhs_real = real();
    auto _rhs_base = _rhs.base();
    auto _rhs_real = _rhs.real();

    if(_lhs_base == _rhs_base || _lhs_real == _rhs_real) return false;

    return (_lhs_real < _rhs_real);
}

std::string_view
link_file::base() const
{
    return std::string_view{ filepath::basename(name) };
}

std::string
link_file::real() const
{
    return filepath::realpath(name, nullptr, false);
}
}  // namespace binary
}  // namespace rocprofsys
