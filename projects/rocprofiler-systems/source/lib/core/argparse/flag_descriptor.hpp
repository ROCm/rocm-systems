// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "argparse.hpp"
#include "common/environment.hpp"

#include <string_view>
#include <vector>

namespace rocprofsys
{
namespace argparse
{

enum class value_kind
{
    flag,
    scalar,
    list,
};

enum class join_with
{
    none,
    space,
    comma,
    colon,
};

struct count_spec
{
    int exact = -1;
    int min   = -1;
    int max   = -1;

    static constexpr count_spec exactly(int n) noexcept { return { n, -1, -1 }; }
    static constexpr count_spec range(int lo, int hi) noexcept { return { -1, lo, hi }; }
    static constexpr count_spec at_least(int lo) noexcept { return { -1, lo, -1 }; }
    static constexpr count_spec at_most(int hi) noexcept { return { -1, -1, hi }; }
};

using custom_action_t = void (*)(parser_t&, parser_data&);

struct flag_descriptor
{
    std::vector<std::string_view> names;
    std::string_view              help;
    std::string_view              dtype       = {};
    count_spec                    count       = {};
    value_kind                    kind        = value_kind::flag;
    join_with                     join        = join_with::none;
    std::vector<std::string_view> env_vars    = {};
    common::update_mode           mode        = common::update_mode::REPLACE;
    std::vector<std::string_view> aliased_env = {};
    std::vector<std::string_view> choices     = {};
    std::vector<std::string_view> conflicts   = {};
    std::vector<std::string_view> requires_   = {};
    custom_action_t               custom      = nullptr;
};

struct flag_group
{
    std::string_view             title;
    std::string_view             subtitle = {};
    std::vector<flag_descriptor> flags;
};

}  // namespace argparse
}  // namespace rocprofsys
