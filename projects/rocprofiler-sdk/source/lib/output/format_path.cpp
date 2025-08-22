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
#include "lib/common/regex.hpp"
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
const auto env_regexes =
    new std::array<std::string, 3>{std::string{"(.*)%(env|ENV)\\{([A-Z0-9_]+)\\}%(.*)"},
                                   std::string{"(.*)\\$(env|ENV)\\{([A-Z0-9_]+)\\}(.*)"},
                                   std::string{"(.*)%q\\{([A-Z0-9_]+)\\}(.*)"}};
// env regex examples:
//  - %env{USER}%       Consistent with other output key formats (start+end with %)
//  - $ENV{USER}        Similar to CMake
//  - %q{USER}          Compatibility with NVIDIA
//

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

        for(const auto& _re : *env_regexes)
        {
            while(rocprofiler::common::regex::regex_search(_fpath, _re))
            {
                auto        _var = rocprofiler::common::regex::regex_replace(_fpath, _re, "$3");
                std::string _val = common::get_env<std::string>(_var, "");
                _val             = strip_leading_and_replace(_val, {'\t', ' ', '/'}, "_");
                auto _beg        = rocprofiler::common::regex::regex_replace(_fpath, _re, "$1");
                auto _end        = rocprofiler::common::regex::regex_replace(_fpath, _re, "$4");
                _fpath           = fmt::format("{}{}{}", _beg, _val, _end);
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
        auto _re = std::string{"(.*)(%|\\{)(arg[0-9]+)(%|\\})([-/_]*)(.*)"};
        while(rocprofiler::common::regex::regex_search(_fpath, _re))
            _fpath = rocprofiler::common::regex::regex_replace(_fpath, _re, "$1$6");
    } catch(std::exception& _e)
    {
        ROCP_WARNING << "[rocprofiler] " << __FUNCTION__ << " threw an exception :: " << _e.what()
                     << "\n";
    }

    return _fpath;
}

std::string
format_path(std::string&& _fpath, const std::vector<output_key>& _keys, bool require_mp_stable)
{
    if(_fpath.find('%') == std::string::npos && _fpath.find('{') == std::string::npos &&
       _fpath.find('$') == std::string::npos)
        return _fpath;

    // save the original path
    auto _ref = _fpath;

    // removes multiprocess unstable keys
    if(require_mp_stable)
    {
        for(const auto& kitr : _keys)
        {
            if(!kitr.is_multiprocess_stable)
            {
                auto pos = std::string::npos;
                while((pos = _fpath.find(kitr.key)) != std::string::npos)
                {
                    constexpr auto join_chars = std::string_view{".-_/:;'"};

                    auto _replace = [&_fpath](
                        std::string_view _key, auto _offset, std::string_view _log_label) -> auto&
                    {
                        // verify key is found in correct position before removing
                        if(auto _pos = _fpath.find(_key, _offset);
                           _pos != std::string::npos && _pos < _offset + _key.length())
                        {
                            // save for logging
                            auto _key_ref = std::string{_key};
                            _fpath        = _fpath.replace(_pos, _key.length(), std::string{});
                            ROCP_INFO << fmt::format("[{:<16}] replace(\"{}\", {}) -> \"{}\"",
                                                     _log_label,
                                                     _key_ref,
                                                     _offset,
                                                     _fpath);
                        }
                        return _fpath;
                    };

                    // preceding character
                    auto _prefix = std::string{};
                    if(pos > 0) _prefix = _fpath.substr(pos - 1, 1);

                    // succeeding character
                    auto _suffix = std::string{};
                    if(pos + kitr.key.length() + 1 < _fpath.length())
                        _suffix = _fpath.substr(pos + kitr.key.length(), 1);

                    ROCP_TRACE << fmt::format(
                        "key = '{}', input = '{}', prefix = '{}', suffix = '{}'",
                        kitr.key,
                        _fpath,
                        _prefix,
                        _suffix);

                    auto _orig_fpath = _fpath;
                    _fpath           = _replace(kitr.key, pos, "key remove");

                    // if the preceding and succeeding characters are the same, delete one of them,
                    // e.g. out-{rank}-{size} should become out-{size}, not out--{size}
                    //
                    // if the preceding and succeeding characters are both join characters, delete
                    // trailing character, e.g. out.{rank}_{size} should become out.{size}, not
                    // out._{size}
                    if(!_prefix.empty() && !_suffix.empty())
                    {
                        if(_prefix == _suffix)
                        {
                            _fpath = _replace(_prefix, pos - 1, "equal remove");
                        }
                        else if(_prefix.find_first_of(join_chars) != std::string::npos ||
                                _suffix.find_first_of(join_chars) != std::string::npos)
                        {
                            _fpath = _replace(_suffix, pos, "either remove");
                        }
                    }
                    // if the preceding character is now the last character in the string and a join
                    // character, remove it
                    else if(!_prefix.empty() && _suffix.empty() && _fpath.length() > 1)
                    {
                        if(_prefix.find_first_of(join_chars) != std::string::npos)
                        {
                            _fpath = _replace(_prefix, _fpath.length() - 1, "preceding remove");
                        }
                    }
                    // if the succeeding character is now the first character in the string and a
                    // join character, remove it
                    else if(_prefix.empty() && !_suffix.empty() && _fpath.length() > 1)
                    {
                        if(_suffix.find_first_of(join_chars) != std::string::npos)
                        {
                            _fpath = _replace(_suffix, 0, "succeeding remove");
                        }
                    }

                    ROCP_INFO << fmt::format("format_path remove non-multi-process stable key '{}' "
                                             "at position {} :: {} -> {}",
                                             kitr.key,
                                             pos,
                                             _orig_fpath,
                                             _fpath);
                }
            }
        }
    }

    // substitute any output keys
    _fpath = format_path_impl(std::move(_fpath), _keys);

    return (_fpath == _ref) ? _fpath : format_path(std::move(_fpath), _keys, require_mp_stable);
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
                                           "OMPI_COMM_WORLD_SIZE",
                                           "SLURM_NPROCS"});
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
                                           "OMPI_COMM_WORLD_RANK",
                                           "SLURM_PROCID"});
    return _v;
}

std::string
format_path(std::string _fpath, const std::string& _tag)
{
    return format_path(std::move(_fpath), output_keys(_tag), false);
}

std::string
format_mp_stable_path(std::string _fpath, const std::string& _tag)
{
    return format_path(std::move(_fpath), output_keys(_tag), true);
}
}  // namespace tool
}  // namespace rocprofiler
