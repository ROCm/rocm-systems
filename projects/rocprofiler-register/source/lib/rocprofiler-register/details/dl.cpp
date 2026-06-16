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

#include "dl.hpp"

#include "filesystem.hpp"
#include "platform/loader.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace rocprofiler_register
{
namespace binary
{
std::vector<segment_address_ranges>
get_segment_addresses()
{
    auto modules = platform::get_segment_addresses();
    auto out     = std::vector<segment_address_ranges>{};
    out.reserve(modules.size());
    for(auto& mod : modules)
    {
        auto entry     = segment_address_ranges{};
        entry.filepath = std::move(mod.filepath);
        entry.ranges.reserve(mod.ranges.size());
        for(const auto& range : mod.ranges)
        {
            entry.ranges.emplace_back(address_range{ range.start, range.last });
        }
        out.emplace_back(std::move(entry));
    }
    return out;
}

std::optional<std::string>
get_linked_path(std::string_view name, open_modes_vec_t&& open_modes)
{
    if(name.empty()) return fs::current_path().string();

    auto name_str = std::string{ name };

    // Prefer the already-loaded handle (no refcount bump).
    //
    // open_modes carries POSIX dlopen flags (e.g. RTLD_NOLOAD | RTLD_LAZY).
    // On Linux those flags are forwarded to dlopen; on Windows the Win32
    // loader has no equivalent flag set, so we use open_modes only as a
    // boolean: non-empty == "caller wants noload-only" (skip the fallback
    // transient open), empty == "any open is acceptable".
    //
    // The canonical noload-only caller is get_this_library_path, which passes
    // { RTLD_NOLOAD | RTLD_LAZY } to avoid loading a second copy of the
    // rocprofiler-register DSO when its soname is missing or renamed.
    auto* handle = platform::module_open_already_loaded(name_str.c_str());
    auto  opened = false;
    if(handle == nullptr && open_modes.empty())
    {
        handle = platform::module_open(name_str.c_str());
        opened = (handle != nullptr);
    }

    if(handle == nullptr) return std::nullopt;

    auto path = platform::module_path(handle);
    if(opened) platform::module_close(handle);
    if(path.empty()) return std::nullopt;
    return path;
}

}  // namespace binary
}  // namespace rocprofiler_register
