// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "defines.hpp"
#include <cstdint>

#include <timemory/components/types.hpp>
#include <timemory/mpl/concepts.hpp>
#include <timemory/settings/types.hpp>
#include <timemory/utility/argparse.hpp>
#include <timemory/utility/type_list.hpp>

#include <array>
#include <cstddef>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

//--------------------------------------------------------------------------------------//
// namespaces

namespace regex_const = ::std::regex_constants;  // NOLINT
namespace comp        = ::tim::component;        // NOLINT
using settings        = ::tim::settings;         // NOLINT
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

extern bool              debug_msg;
extern bool              csv;
extern bool              markdown;
extern bool              case_insensitive;
extern bool              regex_hl;
extern std::int32_t      verbose_level;
extern str_vec_t         regex_keys;
extern str_vec_t         category_regex_keys;
extern str_set_t         category_view;
extern std::stringstream lerr;

// explicit setting names to exclude
extern std::set<std::string> settings_exclude;

struct format_options
{
    std::string  delim          = "|";
    bool         csv            = false;
    bool         markdown       = false;
    bool         alphabetical   = false;
    bool         available_only = false;
    bool         all_info       = false;
    bool         force_brief    = false;
    bool         expand_keys    = false;
    bool         force_config   = false;
    bool         print_advanced = false;
    std::int32_t max_width      = 0;
    std::int32_t num_cols       = 0;
    std::int32_t min_width      = 40;
    std::int32_t padding        = 4;
    // Preset export metadata (used with -F json)
    std::string preset_name;
    std::string preset_description;
};

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

// ROCm operation-list settings follow the env-var shape
// ROCPROFSYS_ROCM_<DOMAIN>_OPERATIONS. These helpers are the single source of
// truth for that mapping; do not reconstruct the prefix/suffix elsewhere.

// Returns the lowercased <DOMAIN> if _env_var_name matches the shape exactly,
// or std::nullopt otherwise (e.g. companion settings such as
// _OPERATIONS_EXCLUDE return nullopt).
std::optional<std::string>
rocm_domain_from_setting_name(std::string_view _env_var_name);

// Builds ROCPROFSYS_ROCM_<DOMAIN>_OPERATIONS from any-case domain name.
std::string
rocm_setting_name_for_domain(std::string_view _domain);

void
filter_operations(const std::string& env_var_name, std::vector<std::string>& choices);

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
