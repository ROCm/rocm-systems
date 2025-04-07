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

#include <timemory/components/types.hpp>
#include <timemory/mpl/concepts.hpp>
#include <timemory/settings/types.hpp>
#include <timemory/utility/argparse.hpp>
#include <timemory/utility/demangle.hpp>
#include <timemory/utility/type_list.hpp>

#include <array>
#include <cstddef>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

//--------------------------------------------------------------------------------------//
// namespaces

namespace regex_const = ::std::regex_constants;  // NOLINT
namespace comp        = ::tim::component;        // NOLINT
using settings        = ::tim::settings;         // NOLINT
using tim::demangle;                             // NOLINT
using tim::type_list;                            // NOLINT

//--------------------------------------------------------------------------------------//
// aliases

template <typename Tp, size_t N>
using array_t        = ::std::array<Tp, N>;
using string_t       = ::std::string;
using stringstream_t = ::std::stringstream;
using str_vec_t      = ::std::vector<string_t>;
using str_set_t      = ::std::set<string_t>;
using info_type_base = ::std::tuple<string_t, bool, str_vec_t>;
using parser_t       = ::tim::argparse::argument_parser;

//--------------------------------------------------------------------------------------//
// enums

enum : int
{
    VAL      = 0,
    ENUM     = 1,
    LANG     = 2,
    CID      = 3,
    FNAME    = 4,
    DESC     = 5,
    CATEGORY = 6,
    TOTAL    = 7
};

//--------------------------------------------------------------------------------------//
// variables

constexpr size_t num_component_options   = 7;
constexpr size_t num_settings_options    = 4;
constexpr size_t num_hw_counter_options  = 5;
constexpr size_t num_dump_config_options = TOTAL;

extern std::string       global_delim;
extern bool              csv;
extern bool              markdown;
extern bool              alphabetical;
extern bool              available_only;
extern bool              all_info;
extern bool              force_brief;
extern bool              debug_msg;
extern bool              case_insensitive;
extern bool              regex_hl;
extern bool              expand_keys;
extern bool              force_config;
extern bool              print_advanced;
extern int32_t           max_width;
extern int32_t           num_cols;
extern int32_t           min_width;
extern int32_t           padding;
extern int32_t           verbose_level;
extern str_vec_t         regex_keys;
extern str_vec_t         category_regex_keys;
extern str_set_t         category_view;
extern std::stringstream lerr;

// explicit setting names to exclude
extern std::set<std::string> settings_exclude;

// exclude some timemory settings which are not relevant to rocprof-sys
//  exact matches, e.g. ROCPROFSYS_BANNER
extern std::string settings_rexclude_exact;

//  leading matches, e.g. ROCPROFSYS_MPI_[A-Z_]+
extern std::string settings_rexclude_begin;

constexpr size_t max_error_message_buffer_length = 4096;

//--------------------------------------------------------------------------------------//
// functions

bool
is_selected(const std::string& line);

bool
is_category_selected(const std::string& _line);

std::string
hl_selected(const std::string& line);

void
process_categories(parser_t&, const str_set_t&);

bool
exclude_setting(const std::string&);

void
dump_log();

void
dump_log_abort(int _v);

std::string
remove(std::string inp, const std::set<std::string>& entries);

bool
file_exists(const std::string&);

// control debug printf statements
#define errprintf(LEVEL, ...)                                                            \
    {                                                                                    \
        if(LEVEL < verbose_level)                                                        \
        {                                                                                \
            if(debug_msg || verbose_level >= LEVEL)                                      \
            {                                                                            \
                fprintf(stderr, "%s", tim::log::color::fatal());                         \
                fprintf(stderr, "[rocprof-sys][avail] Error! " __VA_ARGS__);             \
                fprintf(stderr, "%s", tim::log::color::end());                           \
            }                                                                            \
            char _buff[max_error_message_buffer_length];                                 \
            snprintf(_buff, max_error_message_buffer_length,                             \
                     "[rocprof-sys][avail] Error! " __VA_ARGS__);                        \
            throw std::runtime_error(std::string{ _buff });                              \
        }                                                                                \
        else                                                                             \
        {                                                                                \
            if(debug_msg || verbose_level >= LEVEL)                                      \
            {                                                                            \
                fprintf(stderr, "%s", tim::log::color::warning());                       \
                fprintf(stderr, "[rocprof-sys][avail] Warning! " __VA_ARGS__);           \
                fprintf(stderr, "%s", tim::log::color::end());                           \
            }                                                                            \
        }                                                                                \
        fflush(stderr);                                                                  \
    }

// control verbose printf statements
#define verbprintf(LEVEL, ...)                                                           \
    {                                                                                    \
        if(debug_msg || verbose_level >= LEVEL)                                          \
        {                                                                                \
            fprintf(stderr, "%s", tim::log::color::info());                              \
            fprintf(stderr, "[rocprof-sys][avail] " __VA_ARGS__);                        \
            fprintf(stderr, "%s", tim::log::color::end());                               \
        }                                                                                \
        fflush(stderr);                                                                  \
    }

#define verbprintf_bare(LEVEL, ...)                                                      \
    {                                                                                    \
        if(debug_msg || verbose_level >= LEVEL)                                          \
        {                                                                                \
            fprintf(stderr, __VA_ARGS__);                                                \
        }                                                                                \
        fflush(stderr);                                                                  \
    }
