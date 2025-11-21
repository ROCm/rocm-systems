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

#include <rocprofiler-sdk/ompt/omp-tools.h>

#include "fmt/core.h"

#define ROCP_SDK_OMPT_FORMATTER(TYPE, ...)                                                         \
    template <>                                                                                    \
    struct formatter<TYPE> : rocprofiler::ompt::details::base_formatter                            \
    {                                                                                              \
        template <typename Ctx>                                                                    \
        auto format(const TYPE& v, Ctx& ctx) const                                                 \
        {                                                                                          \
            return fmt::format_to(ctx.out(), __VA_ARGS__);                                         \
        }                                                                                          \
    };

#define ROCP_SDK_OMPT_FORMAT_CASE_STMT(PREFIX, SUFFIX)                                             \
    case PREFIX##_##SUFFIX: return fmt::format_to(ctx.out(), #SUFFIX)

namespace rocprofiler
{
namespace ompt
{
namespace details
{
struct base_formatter
{
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin();
    }
};
}  // namespace details
}  // namespace ompt
}  // namespace rocprofiler

namespace fmt
{
template <>
struct formatter<ompt_set_result_t> : rocprofiler::ompt::details::base_formatter
{
    template <typename Ctx>
    auto format(ompt_set_result_t v, Ctx& ctx) const
    {
        switch(v)
        {
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt_set, error);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt_set, never);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt_set, impossible);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt_set, sometimes);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt_set, sometimes_paired);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt_set, always);
        }
        return fmt::format_to(ctx.out(), "Unknown");
    }
};

template <>
struct formatter<ompt_thread_t> : rocprofiler::ompt::details::base_formatter
{
    template <typename Ctx>
    auto format(ompt_thread_t v, Ctx& ctx) const
    {
        switch(v)
        {
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt_thread, initial);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt_thread, worker);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt_thread, other);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt_thread, unknown);
        }
        return fmt::format_to(ctx.out(), "Unknown");
    }
};

template <>
struct formatter<ompt_scope_endpoint_t> : rocprofiler::ompt::details::base_formatter
{
    template <typename Ctx>
    auto format(ompt_scope_endpoint_t v, Ctx& ctx) const
    {
        switch(v)
        {
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, scope_begin);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, scope_end);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, scope_beginend);
        }
        return fmt::format_to(ctx.out(), "Unknown");
    }
};

template <>
struct formatter<ompt_dispatch_t> : rocprofiler::ompt::details::base_formatter
{
    template <typename Ctx>
    auto format(ompt_dispatch_t v, Ctx& ctx) const
    {
        switch(v)
        {
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, dispatch_iteration);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, dispatch_section);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, dispatch_ws_loop_chunk);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, dispatch_taskloop_chunk);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, dispatch_distribute_chunk);
        }
        return fmt::format_to(ctx.out(), "Unknown");
    }
};

template <>
struct formatter<ompt_sync_region_t> : rocprofiler::ompt::details::base_formatter
{
    template <typename Ctx>
    auto format(ompt_sync_region_t v, Ctx& ctx) const
    {
        switch(v)
        {
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, sync_region_barrier);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, sync_region_barrier_implicit);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, sync_region_barrier_explicit);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, sync_region_barrier_implementation);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, sync_region_taskwait);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, sync_region_taskgroup);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, sync_region_reduction);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, sync_region_barrier_implicit_workshare);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, sync_region_barrier_implicit_parallel);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, sync_region_barrier_teams);
        }
        return fmt::format_to(ctx.out(), "Unknown");
    }
};

template <>
struct formatter<ompt_target_data_op_t> : rocprofiler::ompt::details::base_formatter
{
    template <typename Ctx>
    auto format(ompt_target_data_op_t v, Ctx& ctx) const
    {
        switch(v)
        {
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, target_data_alloc);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, target_data_transfer_to_device);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, target_data_transfer_from_device);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, target_data_delete);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, target_data_associate);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, target_data_disassociate);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, target_data_alloc_async);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, target_data_transfer_to_device_async);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, target_data_transfer_from_device_async);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, target_data_delete_async);
        }
        return fmt::format_to(ctx.out(), "Unknown");
    }
};

template <>
struct formatter<ompt_data_t> : rocprofiler::ompt::details::base_formatter
{
    template <typename Ctx>
    auto format(const ompt_data_t& v, Ctx& ctx) const
    {
        return fmt::format_to(ctx.out(), "{}", v.value);
    }
};

template <>
struct formatter<ompt_work_t> : rocprofiler::ompt::details::base_formatter
{
    template <typename Ctx>
    auto format(const ompt_work_t& v, Ctx& ctx) const
    {
        switch(v)
        {
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, work_loop);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, work_sections);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, work_single_executor);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, work_single_other);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, work_workshare);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, work_distribute);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, work_taskloop);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, work_scope);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, work_loop_static);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, work_loop_dynamic);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, work_loop_guided);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, work_loop_other);
        }
        return fmt::format_to(ctx.out(), "Unknown");
    }
};

template <>
struct formatter<ompt_task_status_t> : rocprofiler::ompt::details::base_formatter
{
    template <typename Ctx>
    auto format(const ompt_task_status_t& v, Ctx& ctx) const
    {
        switch(v)
        {
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, task_complete);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, task_yield);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, task_cancel);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, task_detach);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, task_early_fulfill);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, task_late_fulfill);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, task_switch);
            ROCP_SDK_OMPT_FORMAT_CASE_STMT(ompt, taskwait_complete);
        }
        return fmt::format_to(ctx.out(), "Unknown");
    }
};

template <>
struct formatter<ompt_frame_t> : rocprofiler::ompt::details::base_formatter
{
    template <typename Ctx>
    auto format(const ompt_frame_t& v, Ctx& ctx) const
    {
        return fmt::format_to(
            ctx.out(),
            "{}exit_frame={}, enter_frame={}, exit_frame_flags={}, enter_frame_flags={}{}",
            '{',
            v.exit_frame,
            v.enter_frame,
            v.exit_frame_flags,
            v.enter_frame_flags,
            '}');
    }
};

template <>
struct formatter<ompt_dependence_t> : rocprofiler::ompt::details::base_formatter
{
    template <typename Ctx>
    auto format(const ompt_dependence_t& v, Ctx& ctx) const
    {
        // stub
        return fmt::format_to(ctx.out(), "(dependence)");
        (void) v;
    }
};

template <>
struct formatter<ompt_dispatch_chunk_t> : rocprofiler::ompt::details::base_formatter
{
    template <typename Ctx>
    auto format(const ompt_dispatch_chunk_t& v, Ctx& ctx) const
    {
        return fmt::format_to(
            ctx.out(), "{}start={}, iterations={}{}", '{', v.start, v.iterations, '}');
    }
};
}  // namespace fmt

#undef ROCP_SDK_OMPT_FORMATTER
#undef ROCP_SDK_OMPT_OSTREAM_FORMATTER
#undef ROCP_SDK_OMPT_FORMAT_CASE_STMT
