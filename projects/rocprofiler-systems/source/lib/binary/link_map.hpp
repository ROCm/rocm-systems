// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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
