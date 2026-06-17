// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common.hpp"
#include "common/env_vars.hpp"
#include "common/environment.hpp"
#include <cstdint>

#include <timemory/mpl/apply.hpp>
#include <timemory/settings/settings.hpp>
#include <timemory/variadic/macros.hpp>

#include <algorithm>
#include <array>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unordered_map>

using settings = ::tim::settings;

bool              csv                 = false;
bool              markdown            = false;
bool              case_insensitive    = false;
bool              regex_hl            = false;
str_vec_t         regex_keys          = {};
str_vec_t         category_regex_keys = {};
str_set_t         category_view       = {};
std::stringstream lerr{};

bool debug_msg =
    rocprofsys::get_env(rocprofsys::env_vars::DEBUG_AVAIL, settings::debug());
std::int32_t verbose_level =
    rocprofsys::get_env(rocprofsys::env_vars::VERBOSE_AVAIL, settings::verbose());

// explicit setting names to exclude
std::set<std::string> settings_exclude = {
    std::string{ rocprofsys::env_vars::ENVIRONMENT },
    std::string{ rocprofsys::env_vars::COMMAND_LINE },
    "cereal_class_version",
    "settings",
};

//--------------------------------------------------------------------------------------//

namespace
{
const auto&
get_regex_constants()
{
    static auto _constants = []() {
        auto _v = regex_const::egrep | regex_const::optimize;
        if(case_insensitive) _v |= regex_const::icase;
        return _v;
    }();
    return _constants;
}

const auto&
get_regex_pattern()
{
    static auto _pattern = []() {
        std::array<std::string, 2> _v{};
        for(const auto& itr : regex_keys)
        {
            if(itr.empty()) continue;
            std::string _local_pattern = {};
            if(itr.at(0) == '~')
            {
                _local_pattern = itr.substr(1);
                _v.at(1) += "|" + _local_pattern;
            }
            else
            {
                _local_pattern = itr;
                _v.at(0) += "|" + _local_pattern;
            }
            lerr << "Adding regex key: '" << _local_pattern << "'...\n";
        }
        for(auto& itr : _v)
            if(!itr.empty()) itr = itr.substr(1);

        return _v;
    }();
    return _pattern;
}

auto
get_regex()
{
    static auto _rc = std::array<std::regex, 2>{
        std::regex(get_regex_pattern().at(0), get_regex_constants()),
        std::regex(get_regex_pattern().at(1), get_regex_constants())
    };
    return _rc;
}

bool
regex_match(const std::string& _line)
{
    if(get_regex_pattern().at(0).empty() && get_regex_pattern().at(1).empty())
        return true;

    static size_t lerr_width = 0;
    lerr_width               = std::max<size_t>(lerr_width, _line.length());
    std::stringstream _line_ss;
    _line_ss << "'" << _line << "'";

    if(!get_regex_pattern().at(1).empty())
    {
        if(std::regex_match(_line, get_regex().at(1)))
        {
            lerr << std::left << std::setw(lerr_width) << _line_ss.str()
                 << " matched negating pattern '" << get_regex_pattern().at(1)
                 << "'...\n";
            return false;
        }

        if(std::regex_search(_line, get_regex().at(1)))
        {
            lerr << std::left << std::setw(lerr_width) << _line_ss.str()
                 << " found negating pattern '" << get_regex_pattern().at(1) << "'...\n";
            return false;
        }
    }

    if(!get_regex_pattern().at(0).empty())
    {
        if(std::regex_match(_line, get_regex().at(0)))
        {
            lerr << std::left << std::setw(lerr_width) << _line_ss.str()
                 << " matched pattern '" << get_regex_pattern().at(0) << "'...\n";
            return true;
        }

        if(std::regex_search(_line, get_regex().at(0)))
        {
            lerr << std::left << std::setw(lerr_width) << _line_ss.str()
                 << " found pattern '" << get_regex_pattern().at(0) << "'...\n";
            return true;
        }
    }

    lerr << std::left << std::setw(lerr_width) << _line_ss.str() << " missing pattern '"
         << get_regex_pattern().at(0) << "'...\n";
    return false;
}

std::string
regex_replace(const std::string& _line)
{
#if defined(TIMEMORY_UNIX)
    if(get_regex_pattern().empty()) return _line;
    if(regex_match(_line))
        return std::regex_replace(_line, get_regex().at(0), "\33[01;04;36;40m$&\33[0m");
#endif
    return _line;
}

const auto&
get_category_regex_pattern()
{
    static auto _pattern = []() {
        std::array<std::string, 2> _v{};
        for(const auto& itr : category_regex_keys)
        {
            if(itr.empty()) continue;
            std::string _local_pattern = {};
            if(itr.at(0) == '~')
            {
                _local_pattern = itr.substr(1);
                _v.at(1) += "|" + _local_pattern;
            }
            else
            {
                _local_pattern = itr;
                _v.at(0) += "|" + _local_pattern;
            }
            lerr << "Adding category regex key: '" << _local_pattern << "'...\n";
        }
        for(auto& itr : _v)
            if(!itr.empty()) itr = itr.substr(1);

        return _v;
    }();
    return _pattern;
}

auto
get_category_regex()
{
    static auto _rc = std::array<std::regex, 2>{
        std::regex(get_category_regex_pattern().at(0), get_regex_constants()),
        std::regex(get_category_regex_pattern().at(1), get_regex_constants())
    };
    return _rc;
}

bool
category_regex_match(const std::string& _line)
{
    if(get_category_regex_pattern().at(0).empty() &&
       get_category_regex_pattern().at(1).empty())
        return true;

    static size_t lerr_width = 0;
    lerr_width               = std::max<size_t>(lerr_width, _line.length());
    std::stringstream _line_ss;
    _line_ss << "'" << _line << "'";

    if(!get_category_regex_pattern().at(1).empty())
    {
        if(std::regex_match(_line, get_category_regex().at(1)))
        {
            lerr << std::left << std::setw(lerr_width) << _line_ss.str()
                 << " matched negating category pattern '"
                 << get_category_regex_pattern().at(1) << "'...\n";
            return false;
        }

        if(std::regex_search(_line, get_category_regex().at(1)))
        {
            lerr << std::left << std::setw(lerr_width) << _line_ss.str()
                 << " found negating category pattern '"
                 << get_category_regex_pattern().at(1) << "'...\n";
            return false;
        }
    }

    if(!get_category_regex_pattern().at(0).empty())
    {
        if(std::regex_match(_line, get_category_regex().at(0)))
        {
            lerr << std::left << std::setw(lerr_width) << _line_ss.str()
                 << " matched category pattern '" << get_category_regex_pattern().at(0)
                 << "'...\n";
            return true;
        }

        if(std::regex_search(_line, get_category_regex().at(0)))
        {
            lerr << std::left << std::setw(lerr_width) << _line_ss.str()
                 << " found category pattern '" << get_category_regex_pattern().at(0)
                 << "'...\n";
            return true;
        }
    }

    lerr << std::left << std::setw(lerr_width) << _line_ss.str()
         << " missing category pattern '" << get_category_regex_pattern().at(0)
         << "'...\n";
    return false;
}
}  // namespace

//--------------------------------------------------------------------------------------//

bool
is_selected(const std::string& _line)
{
    return regex_match(_line);
}

//--------------------------------------------------------------------------------------//

bool
is_category_selected(const std::string& _line)
{
    return category_regex_match(_line);
}

//--------------------------------------------------------------------------------------//

std::string
hl_selected(const std::string& _line)
{
    return (regex_hl) ? regex_replace(_line) : _line;
}

//--------------------------------------------------------------------------------------//

void
process_categories(parser_t& p, const str_set_t& _category_options)
{
    category_view = p.get<str_set_t>("categories");
    std::vector<std::function<void()>> _shorthand_patches{};

    // Helper to do case-insensitive string comparison
    auto _tolower = [](std::string_view in) {
        std::string out(in);
        std::transform(out.begin(), out.end(), out.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        return out;
    };

    // Cache lowercase -> original category mapping to avoid repeated string conversions
    // Also pre-compute shorthand mappings (e.g., "wallclock" -> "component::WallClock")
    std::unordered_map<std::string, std::string> _category_map;
    constexpr std::array<std::string_view, 3> _prefixes = { "component::", "settings::",
                                                            "hw_counters::" };

    for(const auto& opt : _category_options)
    {
        auto opt_lower           = _tolower(opt);
        _category_map[opt_lower] = opt;

        // Add shorthand mappings if the option starts with a known prefix
        for(auto prefix : _prefixes)
        {
            if(opt_lower.size() > prefix.size() &&
               opt_lower.compare(0, prefix.size(), _tolower(prefix)) == 0)
            {
                // Map the shorthand (without prefix) to the full canonical form
                auto shorthand           = opt_lower.substr(prefix.size());
                _category_map[shorthand] = opt;
                break;
            }
        }
    }

    // Helper to find case-insensitive match in category options
    auto _find_category = [&_category_map,
                           &_tolower](std::string_view input) -> std::string_view {
        auto input_lower = _tolower(input);
        auto it          = _category_map.find(input_lower);
        if(it != _category_map.end()) return it->second;
        return "";
    };

    // Process categories - now handles both full names and shorthands via the pre-built
    // map
    for(const auto& itr : category_view)
    {
        auto _matched = _find_category(itr);
        if(!_matched.empty())
        {
            // Only create patch if the matched form differs from input (normalization
            // needed)
            if(_matched != itr)
            {
                // Explicitly convert string_view to string for safe capture
                std::string _matched_str(_matched);
                _shorthand_patches.emplace_back([itr, _matched_str]() {
                    category_view.erase(itr);
                    category_view.emplace(_matched_str);
                });
            }
        }
        else
        {
            throw std::runtime_error(
                itr + " is not a valid category. Use --list-categories to view "
                      "valid categories");
        }
    }
    for(auto&& itr : _shorthand_patches)
        itr();
}

//--------------------------------------------------------------------------------------//

bool
exclude_setting(const std::string& _v)
{
    if(settings_exclude.find(_v) != settings_exclude.end()) return true;
    auto itr = settings::instance()->find(_v, false);
    if(itr == settings::instance()->end()) return true;
    return itr->second->get_hidden();
}

//--------------------------------------------------------------------------------------//

void
dump_log()
{
    if(debug_msg)
    {
        std::cerr << lerr.str() << std::flush;
        lerr = std::stringstream{};
    }
}

void
dump_log_abort(int _v)
{
    fprintf(stderr, "\n[rocprof-sys-avail] Exiting with signal %i...\n", _v);
    debug_msg = true;
    dump_log();
}

//--------------------------------------------------------------------------------------//

std::string
remove(std::string inp, const std::set<std::string>& entries)
{
    for(const auto& itr : entries)
    {
        auto idx = inp.find(itr);
        while(idx != std::string::npos)
        {
            inp.erase(idx, itr.length());
            idx = inp.find(itr);
        }
    }
    return inp;
}

//--------------------------------------------------------------------------------------//

bool
file_exists(const std::string& _fname)
{
    struct stat _buffer;
    if(stat(_fname.c_str(), &_buffer) == 0)
        return (S_ISREG(_buffer.st_mode) != 0 || S_ISLNK(_buffer.st_mode) != 0);
    return false;
}

//--------------------------------------------------------------------------------------//

namespace
{
constexpr std::string_view _rocm_op_prefix = "ROCPROFSYS_ROCM_";
constexpr std::string_view _rocm_op_suffix = "_OPERATIONS";
}  // namespace

std::optional<std::string>
rocm_domain_from_setting_name(std::string_view _env_var_name)
{
    // Check if the environment variable name matches the expected shape
    // ROCPROFSYS_ROCM_<DOMAIN>_OPERATIONS.
    if(_env_var_name.size() <= _rocm_op_prefix.size() + _rocm_op_suffix.size() ||
       _env_var_name.compare(0, _rocm_op_prefix.size(), _rocm_op_prefix) != 0 ||
       _env_var_name.compare(_env_var_name.size() - _rocm_op_suffix.size(),
                             _rocm_op_suffix.size(), _rocm_op_suffix) != 0)
        return std::nullopt;

    // Extract the domain name from the environment variable name, then convert it to
    // lowercase.
    std::string _domain{ _env_var_name.substr(
        _rocm_op_prefix.size(),
        _env_var_name.size() - _rocm_op_prefix.size() - _rocm_op_suffix.size()) };
    std::transform(_domain.begin(), _domain.end(), _domain.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return _domain;
}

std::string
rocm_setting_name_for_domain(std::string_view _domain)
{
    std::string _result;
    _result.reserve(_rocm_op_prefix.size() + _domain.size() + _rocm_op_suffix.size());
    _result.append(_rocm_op_prefix);
    _result.append(_domain);
    _result.append(_rocm_op_suffix);
    std::transform(_result.begin() + _rocm_op_prefix.size(),
                   _result.end() - _rocm_op_suffix.size(),
                   _result.begin() + _rocm_op_prefix.size(),
                   [](unsigned char c) { return std::toupper(c); });
    return _result;
}

void
filter_operations(const std::string& env_var_name, std::vector<std::string>& choices)
{
    auto _domain = rocm_domain_from_setting_name(env_var_name);
    if(!_domain) return;

    // Filter out unsupported operations for the OMPT domain.
    if(*_domain == "ompt")
    {
        choices.erase(
            std::remove_if(choices.begin(), choices.end(),
                           [](const std::string& op) {
                               return op == "omp_callback_functions" ||  // internal
                                      op == "omp_thread_end";            // unsupported
                           }),
            choices.end());
    }
}

//--------------------------------------------------------------------------------------//
