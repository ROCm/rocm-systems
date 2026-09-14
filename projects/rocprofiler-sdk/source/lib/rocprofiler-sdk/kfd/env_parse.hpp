// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include "lib/common/environment.hpp"
#include "lib/common/logging.hpp"

#include <cerrno>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>

// Shared env-parse contract for every dispatch-log knob (D8's quiesce, D5's two
// budgets, D9's cap). common::get_env<int> cannot implement reject-and-warn --
// std::stol silently accepts trailing junk and a narrowing cast hides overflow --
// so accessors build on common::get_env_optional plus this helper instead.
//
// Policy, identical everywhere: reject-and-default, NEVER clamp. A value that is
// SET but malformed or out of range yields nullopt (the caller substitutes its
// default) and emits exactly one warning naming the variable and the value.

namespace rocprofiler
{
namespace kfd
{
// nullopt = unset OR rejected. long is the parse type, so representability is
// decided by strtol + ERANGE and the explicit [lo, hi] range, never a cast.
inline std::optional<long>
env_long_in_range(std::string_view name, long lo, long hi)
{
    auto _raw = common::get_env_optional(name);
    if(!_raw) return std::nullopt;  // unset: caller's default, no warning

    const std::string& _s = *_raw;
    errno                 = 0;
    char*      _end       = nullptr;
    const long _val       = std::strtol(_s.c_str(), &_end, 10);
    const bool _no_digit  = (_end == _s.c_str());
    const bool _trailing  = (*_end != '\0');  // any trailing char, incl whitespace
    const bool _range     = (errno == ERANGE) || (_val < lo) || (_val > hi);
    if(_no_digit || _trailing || _range)
    {
        ROCP_WARNING << "KFD dispatch-log: " << name << "='" << _s << "' is not an integer in ["
                     << lo << ", " << hi << "]; using default";
        return std::nullopt;
    }
    return _val;
}

// Same reject-and-default policy for a STRICT OPT-IN switch: only an explicit
// true value turns the feature on. Unset, empty, a recognized false value, or
// anything unrecognized all yield false -- the unrecognized case with exactly one
// warning naming the variable and the value.
//
// common::get_env<bool> cannot serve here: it maps an unrecognized word ("flase")
// to TRUE, fatals on empty, and throws on an out-of-range number. For an
// LD_PRELOAD switch that changes completion semantics, every one of those is the
// wrong direction -- a typo must never silently enable it and a bad value must
// never throw out of a static initializer.
inline bool
env_bool_opt_in(std::string_view name)
{
    auto _raw = common::get_env_optional(name);
    if(!_raw) return false;  // unset: off, no warning

    const std::string& _s = *_raw;
    if(_s == "1" || _s == "true" || _s == "yes" || _s == "on") return true;
    if(_s == "0" || _s == "false" || _s == "no" || _s == "off" || _s.empty()) return false;

    ROCP_WARNING << "KFD dispatch-log: " << name << "='" << _s
                 << "' is not one of 1/true/yes/on or 0/false/no/off; treating as DISABLED";
    return false;
}
}  // namespace kfd
}  // namespace rocprofiler
