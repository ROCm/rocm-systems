// Copyright (c) 2018-2023 Advanced Micro Devices, Inc.
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

#include <rocprofiler/version.h>

#include <hsa/hsa_ext_amd.h>

#include "fmt/core.h"
#include "fmt/ranges.h"

#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#if !defined(ROCPROFILER_HSA_RUNTIME_EXT_AMD_VERSION)
#    define ROCPROFILER_HSA_RUNTIME_EXT_AMD_VERSION                                                \
        ((10000 * HSA_AMD_INTERFACE_VERSION_MAJOR) + (100 * HSA_AMD_INTERFACE_VERSION_MINOR))
#endif

namespace rocprofiler
{
namespace hsa
{
namespace utils
{
template <typename Tp, typename Up = Tp, std::enable_if_t<fmt::is_formattable<Tp>::value, int> = 0>
std::string
stringize_impl(Tp _v, int)
{
    return fmt::format("{}", _v);
}

template <typename Tp>
std::string
stringize_impl(Tp _v, long)
{
    auto _ss = std::stringstream{};
    _ss << _v;
    return _ss.str();
}

template <typename LhsT, typename RhsT>
auto
stringize_impl(const std::pair<LhsT, RhsT>& _v, int)
{
    return std::make_pair(stringize_impl(_v.first, 0), stringize_impl(_v.second, 0));
}

struct join_args
{
    std::string_view prefix    = {};
    std::string_view suffix    = {};
    std::string_view separator = {};
};

template <typename Tp>
std::string
join_impl(const Tp& _v)
{
    return stringize_impl(_v, 0);
}

template <typename LhsT, typename RhsT>
std::string
join_impl(const std::pair<LhsT, RhsT>& _v)
{
    return fmt::format("{}={}", join_impl(_v.first), join_impl(_v.second));
}

template <typename... Args>
auto
join(join_args ja, Args... args)
{
    auto _content = std::string{};
    {
        auto _ss = std::stringstream{};
        ((_ss << ja.separator << join_impl(args)), ...);
        auto _v = _ss.str();
        if(_v.length() > ja.separator.length()) _content = _v.substr(2);
    }

    return (std::stringstream{} << ja.prefix << _content << ja.suffix).str();
}

template <typename... Args>
auto
stringize(Args... args)
{
    return std::vector<std::pair<std::string, std::string>>{stringize_impl(args, 0)...};
}

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
}  // namespace hsa
}  // namespace rocprofiler

#if ROCPROFILER_HSA_RUNTIME_EXT_AMD_VERSION >= 10300
namespace fmt
{
template <>
struct formatter<hsa_amd_memory_pool_t>
: rocprofiler::hsa::utils::handle_formatter<hsa_amd_memory_pool_t>
{};

template <>
struct formatter<hsa_amd_vmem_alloc_handle_t>
: rocprofiler::hsa::utils::handle_formatter<hsa_amd_vmem_alloc_handle_t>
{};

template <>
struct formatter<hsa_amd_memory_access_desc_t>
{
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename Ctx>
    auto format(const hsa_amd_memory_access_desc_t& v, Ctx& ctx) const
    {
        return fmt::format_to(
            ctx.out(), "permissions={}, agent_handle={}", v.permissions, v.agent_handle);
    }
};
}  // namespace fmt
#endif
