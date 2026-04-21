// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/environment.hpp"

#include <timemory/settings/vsettings.hpp>
#include <timemory/utility/argparse.hpp>

#include <functional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocprofsys
{
namespace argparse
{
struct parser_data;

using parser_t          = ::tim::argparse::argument_parser;
using vsetting_t        = ::tim::vsettings;
using vsettings_set_t   = std::unordered_set<vsetting_t*>;
using strset_t          = std::set<std::string>;
using strvec_t          = std::vector<std::string>;
using setting_filter_t  = std::function<bool(vsetting_t*, const parser_data&)>;
using environ_filter_t  = std::function<bool(std::string_view, const parser_data&)>;
using grouping_filter_t = std::function<bool(std::string_view, const parser_data&)>;

bool
default_setting_filter(vsetting_t*, const parser_data&);

bool
default_environ_filter(std::string_view, const parser_data&);

bool
default_grouping_filter(std::string_view, const parser_data&);

struct env_snapshot
{
    std::unordered_set<std::string> initial = {};
    std::vector<std::string>        current = {};
    // Owns its keys: callers may pass temporaries (e.g. std::string{key}) into
    // update_env, so storing string_view here would dangle once they die.
    std::unordered_set<std::string> updated      = {};
    std::string                     dl_libpath   = {};
    std::string                     omni_libpath = {};

    // Convenience wrapper: hides the (current, updated, initial) plumbing
    // and the join delimiter, so callers stop reaching into three fields.
    template <typename Tp>
    void set(
        std::string_view key, Tp&& value,
        rocprofsys::common::update_mode mode = rocprofsys::common::update_mode::REPLACE,
        std::string_view                join_delim = ":")
    {
        rocprofsys::common::update_env(current, key, std::forward<Tp>(value), mode,
                                       join_delim, updated, initial);
    }
};

struct parse_outcome
{
    std::vector<std::string> command    = {};
    std::string              launcher   = {};
    bool                     monochrome = false;
    bool                     debug      = false;
    bool                     fork_exec  = false;
    int                      verbose    = 0;
};

struct registration_config
{
    vsettings_set_t                 processed_settings = {};
    std::unordered_set<std::string> processed_environs = {};
    std::unordered_set<std::string> processed_groups   = {};
    grouping_filter_t               grouping_filter    = default_grouping_filter;
    setting_filter_t                setting_filter     = default_setting_filter;
    environ_filter_t                environ_filter     = default_environ_filter;
};

struct parser_data
{
    env_snapshot        env = {};
    parse_outcome       out = {};
    registration_config reg = {};
};

parser_data&
init_parser(parser_data&);

parser_data&
add_ld_preload(parser_data&);

parser_data&
add_ld_library_path(parser_data&);

parser_data&
add_torch_library_path(parser_data&, bool verbose = false);

parser_data&
add_core_arguments(parser_t&, parser_data&);

parser_data&
add_group_arguments(parser_t&, const std::string&, parser_data&, bool _add_group = false);

parser_data&
add_extended_arguments(parser_t&, parser_data&);
}  // namespace argparse
}  // namespace rocprofsys
