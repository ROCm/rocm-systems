// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "details/platform/loader.hpp"

#include "details/filesystem.hpp"
#include "details/utility.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include <dlfcn.h>
#include <elf.h>
#include <fmt/core.h>
#include <link.h>
#include <sys/types.h>
#include <unistd.h>

namespace rocprofiler_register
{
namespace platform
{
module_handle_t
module_open(const char* name) noexcept
{
    if(name == nullptr) return nullptr;
    return ::dlopen(name, RTLD_NOW | RTLD_LOCAL);
}

module_handle_t
module_open_already_loaded(const char* name) noexcept
{
    if(name == nullptr) return nullptr;
    return ::dlopen(name, RTLD_LAZY | RTLD_NOLOAD);
}

module_handle_t
module_open_with_fallback(const char* name) noexcept
{
    if(name == nullptr) return nullptr;

    // Mirror prior behavior: try noload first to avoid bringing in a new copy,
    // then RTLD_NOW (local), then RTLD_NOW | RTLD_GLOBAL as a final attempt.
    auto* handle = ::dlopen(name, RTLD_LAZY | RTLD_NOLOAD);
    if(handle != nullptr) return handle;

    handle = ::dlopen(name, RTLD_NOW);
    if(handle != nullptr) return handle;

    handle = ::dlopen(name, RTLD_NOW | RTLD_GLOBAL);
    return handle;
}

void*
module_sym(module_handle_t handle, const char* sym) noexcept
{
    if(handle == nullptr || sym == nullptr) return nullptr;
    return ::dlsym(handle, sym);
}

void*
module_sym_default(const char* sym) noexcept
{
    if(sym == nullptr) return nullptr;
    return ::dlsym(RTLD_DEFAULT, sym);
}

void
module_close(module_handle_t handle) noexcept
{
    if(handle == nullptr) return;
    ::dlclose(handle);
}

std::string
module_path(module_handle_t handle) noexcept
{
    if(handle == nullptr) return std::string{};
    auto* link_map = static_cast<struct link_map*>(nullptr);
    if(::dlinfo(handle, RTLD_DI_LINKMAP, &link_map) != 0 || link_map == nullptr)
        return std::string{};
    if(std::string_view{ link_map->l_name }.empty()) return std::string{};
    auto absolute = fs::absolute(fs::path{ link_map->l_name });
    return absolute.string();
}

std::vector<module_segments>
get_segment_addresses(std::uint32_t pid)
{
    auto data  = std::vector<module_segments>{};
    auto fname = fmt::format("/{}/{}/{}", "proc", pid, "maps");
    auto ifs   = std::ifstream{ fname };
    if(!ifs)
    {
        std::fprintf(stderr, "Failure opening %s\n", fname.c_str());
        return data;
    }

    auto get_entry = [&data](std::string_view name) -> module_segments& {
        for(auto& itr : data)
        {
            if(itr.filepath == name) return itr;
        }
        return data.emplace_back(module_segments{ std::string{ name }, {} });
    };

    while(ifs)
    {
        auto line = std::string{};
        if(std::getline(ifs, line) && !line.empty())
        {
            auto delim = utility::delimit(line, " \t\n\r");
            if(delim.size() > 5 && fs::exists(fs::path{ delim.back() }))
            {
                auto& entry        = get_entry(delim.back());
                auto  addr         = utility::delimit(delim.front(), "-");
                auto  load_address = std::stoull(addr.front(), nullptr, 16);
                auto  last_address = std::stoull(addr.back(), nullptr, 16);
                entry.ranges.emplace_back(
                    module_address_range{ load_address, last_address });
            }
        }
    }
    return data;
}

std::vector<module_segments>
get_segment_addresses()
{
    return get_segment_addresses(static_cast<std::uint32_t>(::getpid()));
}

}  // namespace platform
}  // namespace rocprofiler_register
