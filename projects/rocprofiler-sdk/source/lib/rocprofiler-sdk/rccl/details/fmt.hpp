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

#include <rocprofiler-sdk/rccl.h>
#include <rocprofiler-sdk/version.h>

#include "lib/common/stringize_arg.hpp"

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <cstdint>
#include <sstream>
#include <string_view>
#include <type_traits>

namespace rocprofiler
{
namespace rccl
{
namespace utils
{
template <typename Tp>
struct handle_formatter
{
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename Ctx>
    auto format(const Tp& v, Ctx& ctx) const
    {
        return fmt::format_to(ctx.out(), "handle={}", v.handle);
    }
};

template <typename Tp>
struct handle_formatter<const Tp> : handle_formatter<Tp>
{};
}  // namespace utils
}  // namespace rccl
}  // namespace rocprofiler

namespace fmt
{
template <>
struct formatter<ncclUniqueId>
{
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename Ctx>
    auto format(const ncclUniqueId v, Ctx& ctx) const
    {
        static_assert(sizeof(v) == 128 * sizeof(char), "NCCL ID type changed. Expected char[128]");

        return fmt::format_to(ctx.out(), "0x{:0x}", fmt::join(v.internal, ""));
    }
};

template <>
struct formatter<ncclConfig_t>
{
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename Ctx>
    auto format(const ncclConfig_t& v, Ctx& ctx) const
    {
        return fmt::format_to(ctx.out(),
                              "blocking={}, cgaClusterSize={}, minCTAs={}, maxCTAs={}, "
                              "netName={}, splitShare={}",
                              v.blocking,
                              v.cgaClusterSize,
                              v.minCTAs,
                              v.maxCTAs,
                              v.netName,
                              v.splitShare);
    }
};

template <>
struct formatter<ncclComm_t>
{
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename Ctx>
    auto format(const ncclComm_t& v, Ctx& ctx) const
    {
        return fmt::format_to(ctx.out(), "0x{:0x}", v);
    }
};

}  // namespace fmt
