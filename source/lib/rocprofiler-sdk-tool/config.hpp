// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
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

#include "lib/att-tool/att_lib_wrapper.hpp"
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

struct att_perfcounter
{
    std::string counter_name = {};
    uint32_t    simd_mask    = 0xf;

    template <typename ArchiveT>
    void save(ArchiveT&) const;
};

struct config : output_config
{
    using base_type = output_config;

    struct CollectionPeriod
    {
        uint64_t delay    = 0;
        uint64_t duration = 0;
        uint64_t repeat   = 0;

        template <typename ArchiveT>
        void save(ArchiveT& ar) const;
    };

    enum class benchmark
    {
        none = 0,
        disabled_contexts_overhead,
        sdk_callback_overhead,
        sdk_buffered_overhead,
        tool_runtime_overhead,
        execution_profile,
    };

    config();

    ~config()                 = default;
    config(const config&)     = default;
    config(config&&) noexcept = default;
    config& operator=(const config&) = default;
    config& operator=(config&&) noexcept = default;

    bool   demangle                    = get_env("ROCPROF_DEMANGLE_KERNELS", true);
    bool   truncate                    = get_env("ROCPROF_TRUNCATE_KERNELS", false);
    bool   kernel_trace                = get_env("ROCPROF_KERNEL_TRACE", false);
    bool   hsa_core_api_trace          = get_env("ROCPROF_HSA_CORE_API_TRACE", false);
    bool   hsa_amd_ext_api_trace       = get_env("ROCPROF_HSA_AMD_EXT_API_TRACE", false);
    bool   hsa_image_ext_api_trace     = get_env("ROCPROF_HSA_IMAGE_EXT_API_TRACE", false);
    bool   hsa_finalizer_ext_api_trace = get_env("ROCPROF_HSA_FINALIZER_EXT_API_TRACE", false);
    bool   marker_api_trace            = get_env("ROCPROF_MARKER_API_TRACE", false);
    bool   memory_copy_trace           = get_env("ROCPROF_MEMORY_COPY_TRACE", false);
    bool   memory_allocation_trace     = get_env("ROCPROF_MEMORY_ALLOCATION_TRACE", false);
    bool   scratch_memory_trace        = get_env("ROCPROF_SCRATCH_MEMORY_TRACE", false);
    bool   counter_collection          = get_env("ROCPROF_COUNTER_COLLECTION", false);
    bool   hip_runtime_api_trace       = get_env("ROCPROF_HIP_RUNTIME_API_TRACE", false);
    bool   hip_compiler_api_trace      = get_env("ROCPROF_HIP_COMPILER_API_TRACE", false);
    bool   rccl_api_trace              = get_env("ROCPROF_RCCL_API_TRACE", false);
    bool   rocdecode_api_trace         = get_env("ROCPROF_ROCDECODE_API_TRACE", false);
    bool   rocjpeg_api_trace           = get_env("ROCPROF_ROCJPEG_API_TRACE", false);
    bool   list_metrics                = get_env("ROCPROF_LIST_METRICS", false);
    bool   list_metrics_output_file    = get_env("ROCPROF_OUTPUT_LIST_METRICS_FILE", false);
    bool   advanced_thread_trace       = get_env("ROCPROF_ADVANCED_THREAD_TRACE", false);
    bool   att_serialize_all           = get_env("ROCPROF_ATT_PARAM_SERIALIZE_ALL", false);
    bool   enable_signal_handlers      = get_env("ROCPROF_SIGNAL_HANDLERS", true);
    bool   selected_regions            = get_env("ROCPROF_SELECTED_REGIONS", false);
    bool   output_config_file          = get_env("ROCPROF_OUTPUT_CONFIG_FILE", false);
    bool   pc_sampling_host_trap       = false;
    bool   pc_sampling_stochastic      = false;
    size_t pc_sampling_interval        = get_env("ROCPROF_PC_SAMPLING_INTERVAL", 1);
    rocprofiler_pc_sampling_method_t pc_sampling_method_value = ROCPROFILER_PC_SAMPLING_METHOD_NONE;
    rocprofiler_pc_sampling_unit_t   pc_sampling_unit_value   = ROCPROFILER_PC_SAMPLING_UNIT_NONE;

    int      mpi_size = get_mpi_size();
    int      mpi_rank = get_mpi_rank();
    uint64_t att_param_shader_engine_mask =
        get_env<uint64_t>("ROCPROF_ATT_PARAM_SHADER_ENGINE_MASK", 0x1);
    // 256MB
    uint64_t att_param_buffer_size = get_env<uint64_t>("ROCPROF_ATT_PARAM_BUFFER_SIZE", 0x10000000);
    uint64_t att_param_simd_select = get_env<uint64_t>("ROCPROF_ATT_PARAM_SIMD_SELECT", 0xF);
    uint64_t att_param_target_cu   = get_env<uint64_t>("ROCPROF_ATT_PARAM_TARGET_CU", 1);
    uint64_t att_param_perf_ctrl   = get_env<uint64_t>("ROCPROF_ATT_PARAM_PERFCOUNTER_CTRL", 0);

    std::string kernel_filter_include   = get_env("ROCPROF_KERNEL_FILTER_INCLUDE_REGEX", ".*");
    std::string kernel_filter_exclude   = get_env("ROCPROF_KERNEL_FILTER_EXCLUDE_REGEX", "");
    std::string pc_sampling_method      = get_env("ROCPROF_PC_SAMPLING_METHOD", "none");
    std::string pc_sampling_unit        = get_env("ROCPROF_PC_SAMPLING_UNIT", "none");
    std::string extra_counters_contents = get_env("ROCPROF_EXTRA_COUNTERS_CONTENTS", "");
    std::string att_library_path        = get_env("ROCPROF_ATT_LIBRARY_PATH", "");

    std::unordered_set<size_t>         kernel_filter_range    = {};
    std::vector<std::set<std::string>> counters               = {};
    std::vector<att_perfcounter>       att_param_perfcounters = {};

    std::queue<CollectionPeriod> collection_periods = {};
    uint64_t counter_groups_random_seed = get_env("ROCPROF_COUNTER_GROUPS_RANDOM_SEED", 0);
    uint64_t counter_groups_interval    = get_env("ROCPROF_COUNTER_GROUPS_INTERVAL", 1);
    uint64_t minimum_output_bytes       = get_env("ROCPROF_MINIMUM_OUTPUT_BYTES", 0);

    std::string benchmark_mode_env = get_env("ROCPROF_BENCHMARK_MODE", "");
    benchmark   benchmark_mode     = benchmark::none;

    template <typename ArchiveT>
    void save(ArchiveT&) const;

    template <typename ArchiveT>
    void load(ArchiveT&)
    {}
};

#define CFG_SERIALIZE_MEMBER(VAR)             ar(cereal::make_nvp(#VAR, VAR))
#define CFG_SERIALIZE_NAMED_MEMBER(NAME, VAR) ar(cereal::make_nvp(NAME, VAR))

template <typename ArchiveT>
void
att_perfcounter::save(ArchiveT& ar) const
{
    CFG_SERIALIZE_MEMBER(counter_name);
    CFG_SERIALIZE_MEMBER(simd_mask);
}

template <typename ArchiveT>
void
config::CollectionPeriod::save(ArchiveT& ar) const
{
    CFG_SERIALIZE_MEMBER(delay);
    CFG_SERIALIZE_MEMBER(duration);
    CFG_SERIALIZE_MEMBER(repeat);
}

template <typename ArchiveT>
void
config::save(ArchiveT& ar) const
{
    CFG_SERIALIZE_NAMED_MEMBER("benchmark_mode", benchmark_mode_env);

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
    CFG_SERIALIZE_MEMBER(rccl_api_trace);
    CFG_SERIALIZE_MEMBER(rocdecode_api_trace);
    CFG_SERIALIZE_MEMBER(rocjpeg_api_trace);

    CFG_SERIALIZE_MEMBER(mpi_rank);
    CFG_SERIALIZE_MEMBER(mpi_size);
    CFG_SERIALIZE_MEMBER(collection_periods);
    CFG_SERIALIZE_MEMBER(counters);
    CFG_SERIALIZE_MEMBER(extra_counters_contents);
    CFG_SERIALIZE_MEMBER(kernel_filter_include);
    CFG_SERIALIZE_MEMBER(kernel_filter_exclude);
    CFG_SERIALIZE_MEMBER(kernel_filter_range);
    CFG_SERIALIZE_MEMBER(demangle);
    CFG_SERIALIZE_MEMBER(truncate);
    CFG_SERIALIZE_MEMBER(minimum_output_bytes);
    CFG_SERIALIZE_MEMBER(enable_signal_handlers);
    CFG_SERIALIZE_MEMBER(selected_regions);

    CFG_SERIALIZE_MEMBER(counter_groups_random_seed);
    CFG_SERIALIZE_MEMBER(counter_groups_interval);

    CFG_SERIALIZE_MEMBER(pc_sampling_host_trap);
    CFG_SERIALIZE_MEMBER(pc_sampling_stochastic);
    CFG_SERIALIZE_MEMBER(pc_sampling_method);
    CFG_SERIALIZE_MEMBER(pc_sampling_unit);
    CFG_SERIALIZE_MEMBER(pc_sampling_interval);
    CFG_SERIALIZE_MEMBER(pc_sampling_method_value);
    CFG_SERIALIZE_MEMBER(pc_sampling_unit_value);

    CFG_SERIALIZE_MEMBER(advanced_thread_trace);
    CFG_SERIALIZE_MEMBER(att_serialize_all);
    CFG_SERIALIZE_MEMBER(att_param_shader_engine_mask);
    CFG_SERIALIZE_MEMBER(att_param_buffer_size);
    CFG_SERIALIZE_MEMBER(att_param_simd_select);
    CFG_SERIALIZE_MEMBER(att_param_target_cu);
    CFG_SERIALIZE_MEMBER(att_library_path);
    CFG_SERIALIZE_MEMBER(att_param_perfcounters);
    CFG_SERIALIZE_MEMBER(att_param_perf_ctrl);

    // serialize the base class
    static_cast<const base_type*>(this)->save(ar);
}

#undef CFG_SERIALIZE_MEMBER
#undef CFG_SERIALIZE_NAMED_MEMBER

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
