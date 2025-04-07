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

#pragma once

#include <cstdint>
#include <dlfcn.h>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys
{
namespace binary
{
using open_modes_vec_t = std::vector<int>;

struct link_file
{
    link_file(std::string_view&& _v)
    : name{ _v }
    {}

    std::string_view base() const;
    std::string      real() const;
    bool             operator<(const link_file&) const;

    std::string name = {};
};

// helper function for translating generic lib name to resolved path
std::optional<std::string>
get_linked_path(const char*, open_modes_vec_t&& = {});

// default parameters: get the linked binaries for the exe but exclude the linked binaries
// from librocprof-sys
std::set<link_file>
get_link_map(const char*        _lib               = nullptr,
             const std::string& _exclude_linked_by = "librocprof-sys.so",
             const std::string& _exclude_re        = "librocprof-sys-([a-zA-Z]+)\\.so",
             open_modes_vec_t&& _open_modes        = {});
}  // namespace binary
}  // namespace rocprofsys
