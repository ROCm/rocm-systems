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
//

#pragma once

#include "lib/common/environment.hpp"
#include "lib/common/filesystem.hpp"
#include "lib/common/mpl.hpp"
#include "lib/common/units.hpp"
#include "lib/output/format_path.hpp"
#include "lib/output/output_config.hpp"

#include <rocprofiler-sdk/cxx/serialization.hpp>

#include <fmt/format.h>

#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace rocprofiler
{
namespace tool
{
using common::get_env;

struct config;

enum class config_context
{
    global = 0,
};

void
initialize();

template <config_context ContextT = config_context::global>
config&
get_config();

std::string
format_name(std::string_view _name, const config& = get_config<>());

struct config : output_config
{
    using base_type = output_config;

    config();

    ~config()                 = default;
    config(const config&)     = default;
    config(config&&) noexcept = default;
    config& operator=(const config&) = default;
    config& operator=(config&&) noexcept = default;

    bool demangle                    = get_env("ROCPROF_DEMANGLE_KERNELS", true);
    bool truncate                    = get_env("ROCPROF_TRUNCATE_KERNELS", false);
    bool kernel_trace                = get_env("ROCPROF_KERNEL_TRACE", false);
    bool hsa_core_api_trace          = get_env("ROCPROF_HSA_CORE_API_TRACE", false);
    bool hsa_amd_ext_api_trace       = get_env("ROCPROF_HSA_AMD_EXT_API_TRACE", false);
    bool hsa_image_ext_api_trace     = get_env("ROCPROF_HSA_IMAGE_EXT_API_TRACE", false);
    bool hsa_finalizer_ext_api_trace = get_env("ROCPROF_HSA_FINALIZER_EXT_API_TRACE", false);
    bool marker_api_trace            = get_env("ROCPROF_MARKER_API_TRACE", false);
    bool memory_copy_trace           = get_env("ROCPROF_MEMORY_COPY_TRACE", false);
    bool memory_allocation_trace     = get_env("ROCPROF_MEMORY_ALLOCATION_TRACE", false);
    bool scratch_memory_trace        = get_env("ROCPROF_SCRATCH_MEMORY_TRACE", false);
    bool counter_collection          = get_env("ROCPROF_COUNTER_COLLECTION", false);
    bool hip_runtime_api_trace       = get_env("ROCPROF_HIP_RUNTIME_API_TRACE", false);
    bool hip_compiler_api_trace      = get_env("ROCPROF_HIP_COMPILER_API_TRACE", false);
    bool rccl_api_trace              = get_env("ROCPROF_RCCL_API_TRACE", false);
    bool list_metrics                = get_env("ROCPROF_LIST_METRICS", false);
    bool list_metrics_output_file    = get_env("ROCPROF_OUTPUT_LIST_METRICS_FILE", false);

    int mpi_size = get_mpi_size();
    int mpi_rank = get_mpi_rank();

    std::string kernel_filter_include   = get_env("ROCPROF_KERNEL_FILTER_INCLUDE_REGEX", ".*");
    std::string kernel_filter_exclude   = get_env("ROCPROF_KERNEL_FILTER_EXCLUDE_REGEX", "");
    std::string extra_counters_contents = get_env("ROCPROF_EXTRA_COUNTERS_CONTENTS", "");

    std::unordered_set<uint32_t> kernel_filter_range = {};
    std::set<std::string>        counters            = {};

    template <typename ArchiveT>
    void save(ArchiveT&) const;

    template <typename ArchiveT>
    void load(ArchiveT&)
    {}
};

template <typename ArchiveT>
void
config::save(ArchiveT& ar) const
{
#define CFG_SERIALIZE_MEMBER(VAR)             ar(cereal::make_nvp(#VAR, VAR))
#define CFG_SERIALIZE_NAMED_MEMBER(NAME, VAR) ar(cereal::make_nvp(NAME, VAR))

    CFG_SERIALIZE_MEMBER(kernel_trace);
    CFG_SERIALIZE_MEMBER(hsa_core_api_trace);
    CFG_SERIALIZE_MEMBER(hsa_amd_ext_api_trace);
    CFG_SERIALIZE_MEMBER(hsa_image_ext_api_trace);
    CFG_SERIALIZE_MEMBER(hsa_finalizer_ext_api_trace);
    CFG_SERIALIZE_MEMBER(marker_api_trace);
    CFG_SERIALIZE_MEMBER(memory_copy_trace);
    CFG_SERIALIZE_MEMBER(memory_allocation_trace);
    CFG_SERIALIZE_MEMBER(scratch_memory_trace);
    CFG_SERIALIZE_MEMBER(counter_collection);
    CFG_SERIALIZE_MEMBER(hip_runtime_api_trace);
    CFG_SERIALIZE_MEMBER(hip_compiler_api_trace);
    CFG_SERIALIZE_MEMBER(kernel_rename);
    CFG_SERIALIZE_MEMBER(counters);
    CFG_SERIALIZE_MEMBER(kernel_filter_include);
    CFG_SERIALIZE_MEMBER(kernel_filter_exclude);
    CFG_SERIALIZE_MEMBER(kernel_filter_range);
    CFG_SERIALIZE_MEMBER(demangle);
    CFG_SERIALIZE_MEMBER(truncate);

    static_cast<const base_type&>(*this).save(ar);

#undef CFG_SERIALIZE_MEMBER
#undef CFG_SERIALIZE_NAMED_MEMBER
}

template <config_context ContextT>
config&
get_config()
{
    if constexpr(ContextT == config_context::global)
    {
        static auto* _v = new config{};
        return *_v;
    }
    else
    {
        // context specific config copied from global config
        static auto* _v = new config{get_config<config_context::global>()};
        return *_v;
    }
}
}  // namespace tool
}  // namespace rocprofiler
