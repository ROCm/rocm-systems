// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

#include "format_path.hpp"

#include "lib/common/defines.hpp"
#include "lib/common/demangle.hpp"
#include "lib/common/environment.hpp"
#include "lib/common/filesystem.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/units.hpp"
#include "lib/common/utility.hpp"
#include "lib/output/output_key.hpp"

#include <rocprofiler-sdk/cxx/details/tokenize.hpp>

#include <fmt/core.h>

#include <linux/limits.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <ctime>
#include <fstream>
#include <limits>
#include <locale>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofiler
{
namespace tool
{
namespace
{
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=77704
// NOLINTBEGIN
[[maybe_unused]] volatile bool _initLocale = []() {
    const std::ctype<char>& ct(std::use_facet<std::ctype<char>>(std::locale()));
    for(size_t i = 0; i <= std::numeric_limits<unsigned char>::max(); i++)
        ct.narrow(static_cast<char>(i), '\0');
    ct.narrow(0, 0, 0, 0);
    return true;
}();
// NOLINTEND

std::string
format_path_impl(std::string _fpath, const std::vector<output_key>& _keys)
{
    if(_fpath.find('%') == std::string::npos && _fpath.find('{') == std::string::npos &&
       _fpath.find('$') == std::string::npos)
        return _fpath;

    auto _replace = [](auto& _v, const output_key& pitr) {
        auto pos = std::string::npos;
        while((pos = _v.find(pitr.key)) != std::string::npos)
            _v.replace(pos, pitr.key.length(), pitr.value);
    };

    for(auto&& itr : _keys)
        _replace(_fpath, itr);

    // environment and configuration variables
    try
    {
        auto strip_leading_and_replace =
            [](std::string_view inp_v, std::initializer_list<char> keys, const char* val) {
                auto inp = std::string{inp_v};
                for(auto key : keys)
                {
                    auto pos = std::string::npos;
                    while((pos = inp.find(key)) == 0)
                        inp = inp.substr(pos + 1);

                    while((pos = inp.find(key)) != std::string::npos)
                        inp = inp.replace(pos, 1, val);
                }
                return inp;
            };

        if(_fpath.find("$env{") != std::string::npos)
        {
            const auto env_regexes =
                new std::array<std::regex, 3>{std::regex{"(.*)%(env|ENV)\\{([A-Z0-9_]+)\\}%(.*)"},
                                              std::regex{"(.*)\\$(env|ENV)\\{([A-Z0-9_]+)\\}(.*)"},
                                              std::regex{"(.*)%q\\{([A-Z0-9_]+)\\}(.*)"}};
            // env regex examples:
            //  - %env{USER}%       Consistent with other output key formats (start+end with %)
            //  - $ENV{USER}        Similar to CMake
            //  - %q{USER}          Compatibility with NVIDIA
            //
            for(const auto& _re : *env_regexes)
            {
                while(std::regex_search(_fpath, _re))
                {
                    auto        _var = std::regex_replace(_fpath, _re, "$3");
                    std::string _val = common::get_env<std::string>(_var, "");
                    _val             = strip_leading_and_replace(_val, {'\t', ' ', '/'}, "_");
                    auto _beg        = std::regex_replace(_fpath, _re, "$1");
                    auto _end        = std::regex_replace(_fpath, _re, "$4");
                    _fpath           = fmt::format("{}{}{}", _beg, _val, _end);
                }
            }
        }
    } catch(std::exception& _e)
    {
        ROCP_WARNING << "[rocprofiler] " << __FUNCTION__ << " threw an exception :: " << _e.what()
                     << "\n";
    }

    // remove %arg<N>% where N >= argc
    try
    {
        auto remove_arg_patterns = [](std::string& path) {
            bool changed = true;
            while(changed)
            {
                changed = false;

                // Look for %argN% or {argN} patterns
                for(size_t i = 0; i < path.length() - 4; ++i)
                {
                    if((path[i] == '%' || path[i] == '{') && path.substr(i + 1, 3) == "arg")
                    {
                        size_t j = i + 4;
                        // Skip digits
                        while(j < path.length() && std::isdigit(path[j]))
                        {
                            j++;
                        }

                        if(j < path.length() && (path[j] == '%' || path[j] == '}'))
                        {
                            j++;  // Include closing delimiter

                            // Skip trailing separators
                            while(j < path.length() &&
                                  (path[j] == '-' || path[j] == '/' || path[j] == '_'))
                            {
                                j++;
                            }

                            path.erase(i, j - i);
                            changed = true;
                            break;
                        }
                    }
                }
            }
        };
        remove_arg_patterns(_fpath);
    } catch(std::exception& _e)
    {
        ROCP_WARNING << "[rocprofiler] " << __FUNCTION__ << " threw an exception :: " << _e.what()
                     << "\n";
    }

    return _fpath;
}

std::string
format_path(std::string&& _fpath, const std::vector<output_key>& _keys)
{
    if(_fpath.find('%') == std::string::npos && _fpath.find('{') == std::string::npos &&
       _fpath.find('$') == std::string::npos)
        return _fpath;

    auto _ref = _fpath;
    _fpath    = format_path_impl(std::move(_fpath), _keys);

    return (_fpath == _ref) ? _fpath : format_path(std::move(_fpath), _keys);
}

template <typename Tp>
Tp
get_variable_env(Tp _default_v, std::initializer_list<std::string_view>&& _options)
{
    // set env variables towards end override preceding environment variables
    auto _val = _default_v;
    for(auto itr : _options)
        _val = common::get_env<Tp>(itr, std::move(_val));
    return _val;
}
}  // namespace

int
get_mpi_size()
{
    static int _v = get_variable_env<int>(0,
                                          {"MPI_SIZE",  // most generic to most runtime-specific
                                           "MPI_LOCALNRANKS",
                                           "MPI_NRANKS",
                                           "MV2_COMM_WORLD_SIZE",
                                           "OMPI_COMM_WORLD_SIZE"});
    return _v;
}

int
get_mpi_rank()
{
    static int _v = get_variable_env<int>(0,
                                          {"MPI_RANK",  // most generic to most runtime-specific
                                           "MPI_LOCALRANKID",
                                           "MPI_RANKID",
                                           "MV2_COMM_WORLD_RANK",
                                           "OMPI_COMM_WORLD_RANK"});
    return _v;
}

std::string
format_path(std::string _fpath, const std::string& _tag)
{
    return format_path(std::move(_fpath), output_keys(_tag));
}
}  // namespace tool
}  // namespace rocprofiler
