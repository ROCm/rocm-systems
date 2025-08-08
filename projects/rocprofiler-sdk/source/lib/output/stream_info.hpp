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

#include "lib/common/logging.hpp"

#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/cxx/hash.hpp>
#include <rocprofiler-sdk/cxx/name_info.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>
#include <rocprofiler-sdk/cxx/serialization.hpp>

namespace rocprofiler
{
namespace tool
{
struct tool_buffer_tracing_kernel_dispatch_ext_record_t
: rocprofiler_buffer_tracing_kernel_dispatch_record_t
{
    using base_type = rocprofiler_buffer_tracing_kernel_dispatch_record_t;

    tool_buffer_tracing_kernel_dispatch_ext_record_t(const base_type&              _base,
                                                     const rocprofiler_stream_id_t _stream_id)
    : base_type{_base}
    , stream_id{_stream_id}
    {}

    tool_buffer_tracing_kernel_dispatch_ext_record_t()  = delete;
    ~tool_buffer_tracing_kernel_dispatch_ext_record_t() = default;
    tool_buffer_tracing_kernel_dispatch_ext_record_t(
        const tool_buffer_tracing_kernel_dispatch_ext_record_t&) = default;
    tool_buffer_tracing_kernel_dispatch_ext_record_t(
        tool_buffer_tracing_kernel_dispatch_ext_record_t&&) noexcept = default;
    tool_buffer_tracing_kernel_dispatch_ext_record_t& operator       =(
        const tool_buffer_tracing_kernel_dispatch_ext_record_t&) = default;
    tool_buffer_tracing_kernel_dispatch_ext_record_t& operator       =(
        tool_buffer_tracing_kernel_dispatch_ext_record_t&&) noexcept = default;

    rocprofiler_stream_id_t stream_id = {.handle = 0};
};

struct tool_buffer_tracing_memory_copy_ext_record_t
: rocprofiler_buffer_tracing_memory_copy_record_t
{
    using base_type = rocprofiler_buffer_tracing_memory_copy_record_t;

    tool_buffer_tracing_memory_copy_ext_record_t(const base_type&              _base,
                                                 const rocprofiler_stream_id_t _stream_id)
    : base_type{_base}
    , stream_id{_stream_id}
    {}

    tool_buffer_tracing_memory_copy_ext_record_t()  = delete;
    ~tool_buffer_tracing_memory_copy_ext_record_t() = default;
    tool_buffer_tracing_memory_copy_ext_record_t(
        const tool_buffer_tracing_memory_copy_ext_record_t&) = default;
    tool_buffer_tracing_memory_copy_ext_record_t(
        tool_buffer_tracing_memory_copy_ext_record_t&&) noexcept = default;
    tool_buffer_tracing_memory_copy_ext_record_t& operator       =(
        const tool_buffer_tracing_memory_copy_ext_record_t&) = default;
    tool_buffer_tracing_memory_copy_ext_record_t& operator       =(
        tool_buffer_tracing_memory_copy_ext_record_t&&) noexcept = default;

    rocprofiler_stream_id_t stream_id = {};
};

struct tool_buffer_tracing_memory_allocation_ext_record_t
: rocprofiler_buffer_tracing_memory_allocation_record_t
{
    using base_type = rocprofiler_buffer_tracing_memory_allocation_record_t;

    tool_buffer_tracing_memory_allocation_ext_record_t(const base_type&              _base,
                                                       const rocprofiler_stream_id_t _stream_id)
    : base_type{_base}
    , stream_id{_stream_id}
    {}

    tool_buffer_tracing_memory_allocation_ext_record_t()  = delete;
    ~tool_buffer_tracing_memory_allocation_ext_record_t() = default;
    tool_buffer_tracing_memory_allocation_ext_record_t(
        const tool_buffer_tracing_memory_allocation_ext_record_t&) = default;
    tool_buffer_tracing_memory_allocation_ext_record_t(
        tool_buffer_tracing_memory_allocation_ext_record_t&&) noexcept = default;
    tool_buffer_tracing_memory_allocation_ext_record_t& operator       =(
        const tool_buffer_tracing_memory_allocation_ext_record_t&) = default;
    tool_buffer_tracing_memory_allocation_ext_record_t& operator       =(
        tool_buffer_tracing_memory_allocation_ext_record_t&&) noexcept = default;

    rocprofiler_stream_id_t stream_id = {};
};

struct tool_buffer_tracing_hip_api_ext_record_t : rocprofiler_buffer_tracing_hip_api_ext_record_t
{
    using base_type = rocprofiler_buffer_tracing_hip_api_ext_record_t;

    tool_buffer_tracing_hip_api_ext_record_t(const base_type&              _base,
                                             const rocprofiler_stream_id_t _stream_id)
    : base_type{_base}
    , stream_id{_stream_id}
    {}

    tool_buffer_tracing_hip_api_ext_record_t()  = delete;
    ~tool_buffer_tracing_hip_api_ext_record_t() = default;
    tool_buffer_tracing_hip_api_ext_record_t(const tool_buffer_tracing_hip_api_ext_record_t&) =
        default;
    tool_buffer_tracing_hip_api_ext_record_t(tool_buffer_tracing_hip_api_ext_record_t&&) noexcept =
        default;
    tool_buffer_tracing_hip_api_ext_record_t& operator       =(
        const tool_buffer_tracing_hip_api_ext_record_t&) = default;
    tool_buffer_tracing_hip_api_ext_record_t& operator       =(
        tool_buffer_tracing_hip_api_ext_record_t&&) noexcept = default;

    rocprofiler_stream_id_t stream_id = {};
};

}  // namespace tool
}  // namespace rocprofiler

namespace cereal
{
#define SAVE_DATA_FIELD(FIELD) ar(make_nvp(#FIELD, data.FIELD))

template <typename ArchiveT>
void
save(ArchiveT&                                                                    ar,
     const ::rocprofiler::tool::tool_buffer_tracing_kernel_dispatch_ext_record_t& data)
{
    cereal::save(ar, static_cast<const rocprofiler_buffer_tracing_kernel_dispatch_record_t&>(data));
    SAVE_DATA_FIELD(stream_id);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, const ::rocprofiler::tool::tool_buffer_tracing_memory_copy_ext_record_t& data)
{
    cereal::save(ar, static_cast<const rocprofiler_buffer_tracing_memory_copy_record_t&>(data));
    SAVE_DATA_FIELD(stream_id);
}

template <typename ArchiveT>
void
save(ArchiveT&                                                                      ar,
     const ::rocprofiler::tool::tool_buffer_tracing_memory_allocation_ext_record_t& data)
{
    cereal::save(ar,
                 static_cast<const rocprofiler_buffer_tracing_memory_allocation_record_t&>(data));
    SAVE_DATA_FIELD(stream_id);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, const ::rocprofiler::tool::tool_buffer_tracing_hip_api_ext_record_t& data)
{
    cereal::save(ar, static_cast<const rocprofiler_buffer_tracing_hip_api_ext_record_t&>(data));
    SAVE_DATA_FIELD(stream_id);
}

#undef SAVE_DATA_FIELD
}  // namespace cereal
