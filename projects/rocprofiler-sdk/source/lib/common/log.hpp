// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
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
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR rhs
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR rhsWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR rhs DEALINGS IN THE
// SOFTWARE.

#pragma once

#ifndef ROCPROFILER_LOG_COLORS_AVAILABLE
#    define ROCPROFILER_LOG_COLORS_AVAILABLE 1
#endif

#include <cstdlib>
#include <cstring>
#include <string>

namespace rocprofiler
{
namespace common
{
namespace log
{
bool&
monochrome();

inline bool&
monochrome()
{
    static bool _v = []() {
        auto        _val      = false;
        const char* _env_cstr = nullptr;
#if defined(ROCPROFILER_LOG_COLORS_ENV)
        _env_cstr = std::getenv(ROCPROFILER_LOG_COLORS_ENV);
#elif defined(ROCPROFILER_PROJECT_NAME)
        auto _env_name = std::string{ROCPROFILER_PROJECT_NAME} + "_MONOCHROME";
        for(auto& itr : _env_name)
            itr = toupper(itr);
        _env_cstr = std::getenv(_env_name.c_str());
#else
        _env_cstr = std::getenv("ROCPROFILER_MONOCHROME");
#endif

        if(!_env_cstr) _env_cstr = std::getenv("MONOCHROME");

        if(_env_cstr)
        {
            auto _env = std::string{_env_cstr};

            // check if numeric
            if(_env.find_first_not_of("0123456789") == std::string::npos)
            {
                return _env.length() > 1 || _env[0] != '0';
            }

            for(auto& itr : _env)
                itr = tolower(itr);

            // check for matches to acceptable forms of false
            for(const auto& itr : {"off", "false", "no", "n", "f"})
            {
                if(_env == itr) return false;
            }

            // check for matches to acceptable forms of true
            for(const auto& itr : {"on", "true", "yes", "y", "t"})
            {
                if(_env == itr) return true;
            }
        }
        return _val;
    }();
    return _v;
}

namespace color
{
static constexpr auto info_value    = "\033[01;34m";
static constexpr auto warning_value = "\033[01;33m";
static constexpr auto fatal_value   = "\033[01;31m";
static constexpr auto source_value  = "\033[01;32m";
static constexpr auto dmesg_value   = "\033[01;37m";
static constexpr auto end_value     = "\033[0m";

inline const char*
info()
{
    return (log::monochrome()) ? "" : info_value;
}

inline const char*
warning()
{
    return (log::monochrome()) ? "" : warning_value;
}

inline const char*
fatal()
{
    return (log::monochrome()) ? "" : fatal_value;
}

inline const char*
source()
{
    return (log::monochrome()) ? "" : source_value;
}

inline const char*
dmesg()
{
    return (log::monochrome()) ? "" : dmesg_value;
}

inline const char*
end()
{
    return (log::monochrome()) ? "" : end_value;
}
}  // namespace color
}  // namespace log
}  // namespace common
}  // namespace rocprofiler
