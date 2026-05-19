// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprofiler-sdk/experimental/registration.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/cxx/version.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <optional>
#include <vector>

namespace rocprofiler
{
namespace registration
{
namespace iterate
{
using runtime_registration_vec_t = std::vector<rocprofiler_runtime_registration_info_t>;

// invokes `rocprofiler_register_iterate_registration_info` and converts
// `rocprofiler_register_registration_info_t` to `rocprofiler_runtime_registration_info_t`.
std::optional<runtime_registration_vec_t>
get_runtime_registrations();
}  // namespace iterate
}  // namespace registration
}  // namespace rocprofiler

namespace fmt
{
// fmt::format support for rocprofiler_runtime_registration_info_t
template <>
struct formatter<rocprofiler_runtime_registration_info_t>
{
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename Ctx>
    auto format(const rocprofiler_runtime_registration_info_t& val, Ctx& ctx) const
    {
        auto _version = rocprofiler::sdk::version::compute_version_triplet(val.lib_version);
        return fmt::format_to(ctx.out(),
                              "{} (v{}.{}.{})",
                              val.common_name,
                              _version.major,
                              _version.minor,
                              _version.patch);
    }
};
}  // namespace fmt
