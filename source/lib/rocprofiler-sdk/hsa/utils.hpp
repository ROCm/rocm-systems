// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprofiler-sdk/ext_version.h>

#include "lib/common/stringize_arg.hpp"

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ext_finalize.h>
#include <hsa/hsa_ext_image.h>

#include <cstdint>
#include <sstream>
#include <string_view>
#include <type_traits>

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
template <typename Tp>
auto
stringize_impl(const Tp& _v)
{
    using value_type = std::decay_t<Tp>;

    if constexpr(fmt::is_formattable<value_type>::value && !std::is_pointer<value_type>::value)
    {
        return fmt::format("{}", _v);
    }
    else
    {
        auto _ss = std::stringstream{};
        _ss << _v;
        return _ss.str();
    }
}

template <typename... Args>
auto
stringize(int32_t max_deref, Args... args)
{
    using array_type = common::stringified_argument_array_t<sizeof...(Args)>;
    return array_type{common::stringize_arg(
        max_deref, args, [](const auto& _v) { return stringize_impl(_v); })...};
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
struct formatter<hsa_agent_t> : rocprofiler::hsa::utils::handle_formatter<hsa_agent_t>
{};

template <>
struct formatter<hsa_amd_memory_pool_t>
: rocprofiler::hsa::utils::handle_formatter<hsa_amd_memory_pool_t>
{};

template <>
struct formatter<hsa_amd_vmem_alloc_handle_t>
: rocprofiler::hsa::utils::handle_formatter<hsa_amd_vmem_alloc_handle_t>
{};

template <>
struct formatter<hsa_access_permission_t>
{
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename Ctx>
    auto format(hsa_access_permission_t v, Ctx& ctx) const
    {
        auto label = [v]() -> std::string_view {
            switch(v)
            {
                case HSA_ACCESS_PERMISSION_NONE: return "NONE";
                case HSA_ACCESS_PERMISSION_RO: return "READ_ONLY";
                case HSA_ACCESS_PERMISSION_WO: return "WRITE_ONLY";
                case HSA_ACCESS_PERMISSION_RW: return "READ_WRITE";
            }
            return "NONE";
        }();
        return fmt::format_to(ctx.out(), "{}", label);
    }
};

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

template <>
struct formatter<hsa_ext_program_t> : rocprofiler::hsa::utils::handle_formatter<hsa_ext_program_t>
{};

template <>
struct formatter<hsa_ext_control_directives_t>
{
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename Ctx>
    auto format(const hsa_ext_control_directives_t& v, Ctx& ctx) const
    {
        return fmt::format_to(
            ctx.out(),
            "control_directives_mask={}, break_exceptions_mask={}, detect_exceptions_mask={}, "
            "max_dynamic_group_size={}, max_flat_grid_size={}, max_flat_workgroup_size={}, "
            "required_grid_size=({},{},{}), required_workgroup_size=({},{},{}), required_dim={}",
            v.control_directives_mask,
            v.break_exceptions_mask,
            v.detect_exceptions_mask,
            v.max_dynamic_group_size,
            v.max_flat_grid_size,
            v.max_flat_workgroup_size,
            v.required_grid_size[0],
            v.required_grid_size[1],
            v.required_grid_size[2],
            v.required_workgroup_size.x,
            v.required_workgroup_size.y,
            v.required_workgroup_size.z,
            v.required_dim);
    }
};
}  // namespace fmt
#endif
