// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "argparse.hpp"
#include "common/environment.hpp"
#include "parsed_values.hpp"

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
    scalar_int,
    scalar_double,
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
    static constexpr count_spec any() noexcept { return {}; }
};

// Engine-agnostic action callback. Receives a parsed_values accessor that
// hides the underlying parser library; descriptors never name the engine.
using custom_action_t = void (*)(parsed_values&, parser_data&);

// Lifetime contract for the embedded string_views:
// `names`, `help`, `dtype`, `env_vars`, `dedup_keys`, `choices`, `conflicts`,
// and `requires_` must reference storage that outlives the descriptor's use
// by the parser. In practice every site that builds a flag_descriptor
// references either string literals or `constexpr std::string_view` constants
// in core_flags.cpp (static storage), and the flag_group factories return
// `static const flag_group&`. Do not assign view fields from temporaries.
//
// Convention: when multiple names are provided, the LONG name must be last.
// The interpreter derives the parser lookup key from `names.back()`.
struct flag_descriptor
{
    std::vector<std::string_view> names;
    std::string_view              help;
    std::string_view              dtype      = {};
    count_spec                    count      = count_spec::any();
    value_kind                    kind       = value_kind::flag;
    join_with                     join       = join_with::none;
    std::vector<std::string_view> env_vars   = {};
    common::update_mode           mode       = common::update_mode::REPLACE;
    std::vector<std::string_view> dedup_keys = {};
    std::vector<std::string_view> choices    = {};
    std::vector<std::string_view> conflicts  = {};
    std::vector<std::string_view> requires_  = {};
    custom_action_t               custom     = nullptr;
};

struct flag_group
{
    std::string_view             title;
    std::string_view             subtitle = {};
    std::vector<flag_descriptor> flags;
};

}  // namespace argparse
}  // namespace rocprofsys
