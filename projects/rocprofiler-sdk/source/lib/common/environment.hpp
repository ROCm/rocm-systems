// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#pragma once

#include <glog/logging.h>

#include <unistd.h>
#include <string>
#include <string_view>
#include <type_traits>

namespace rocprofiler
{
namespace common
{
namespace impl
{
std::string get_env(std::string_view, std::string_view);

std::string
get_env(std::string_view, const char*);

int
get_env(std::string_view, int);

bool
get_env(std::string_view, bool);
}  // namespace impl

template <typename Tp>
inline auto
get_env(std::string_view env_id, Tp&& _default)
{
    if constexpr(std::is_enum<Tp>::value)
    {
        using Up = std::underlying_type_t<Tp>;
        // cast to underlying type -> get_env -> cast to enum type
        return static_cast<Tp>(impl::get_env(env_id, static_cast<Up>(_default)));
    }
    else
    {
        return impl::get_env(env_id, std::forward<Tp>(_default));
    }
}

struct env_config
{
    std::string env_name  = {};
    std::string env_value = {};
    int         overwrite = 0;

    auto operator()(bool _verbose = false) const
    {
        if(env_name.empty()) return -1;
        LOG_IF(INFO, _verbose) << "[rocprofiler][set_env] setenv(\"" << env_name << "\", \""
                               << env_value << "\", " << overwrite << ")\n";
        return setenv(env_name.c_str(), env_value.c_str(), overwrite);
    }
};
}  // namespace common
}  // namespace rocprofiler
