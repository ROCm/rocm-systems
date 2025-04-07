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

#include "defines.hpp"

#include <timemory/settings/vsettings.hpp>
#include <timemory/utility/argparse.hpp>

#include <functional>
#include <set>
#include <vector>

namespace rocprofsys
{
namespace argparse
{
struct parser_data;

using parser_t          = ::tim::argparse::argument_parser;
using vsetting_t        = ::tim::vsettings;
using vsettings_set_t   = std::set<vsetting_t*>;
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

struct parser_data
{
    bool                       monochrome         = false;
    bool                       debug              = false;
    int                        verbose            = 0;
    std::string                dl_libpath         = {};
    std::string                omni_libpath       = {};
    std::string                launcher           = {};
    vsettings_set_t            processed_settings = {};
    std::set<std::string>      processed_environs = {};
    std::set<std::string>      processed_groups   = {};
    std::vector<char*>         current            = {};
    std::vector<char*>         command            = {};
    std::set<std::string_view> updated            = {};
    std::set<std::string>      initial            = {};
    grouping_filter_t          grouping_filter    = default_grouping_filter;
    setting_filter_t           setting_filter     = default_setting_filter;
    environ_filter_t           environ_filter     = default_environ_filter;
};

parser_data&
init_parser(parser_data&);

parser_data&
add_ld_preload(parser_data&);

parser_data&
add_ld_library_path(parser_data&);

parser_data&
add_core_arguments(parser_t&, parser_data&);

parser_data&
add_group_arguments(parser_t&, const std::string&, parser_data&, bool _add_group = false);

parser_data&
add_extended_arguments(parser_t&, parser_data&);
}  // namespace argparse
}  // namespace rocprofsys
