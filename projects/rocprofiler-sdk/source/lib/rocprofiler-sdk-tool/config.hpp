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
namespace fs = common::filesystem;
using common::get_env;

struct config;

enum class config_context
{
    global = 0,
    att_plugin,
    cli_plugin,
    ctf_plugin,
    file_plugin,
    perfetto_plugin,
};

void
initialize();

template <config_context ContextT = config_context::global>
config&
get_config();

std::string
format_name(std::string_view _name, const config& = get_config<>());

std::string
format(std::string _fpath, const std::string& _tag = {});

int
get_mpi_size();

int
get_mpi_rank();

struct config
{
    config();

    bool        demangle                    = get_env("ROCPROF_DEMANGLE_KERNELS", true);
    bool        truncate                    = get_env("ROCPROF_TRUNCATE_KERNELS", false);
    bool        kernel_trace                = get_env("ROCPROF_KERNEL_TRACE", false);
    bool        hsa_core_api_trace          = get_env("ROCPROF_HSA_CORE_API_TRACE", false);
    bool        hsa_amd_ext_api_trace       = get_env("ROCPROF_HSA_AMD_EXT_API_TRACE", false);
    bool        hsa_image_ext_api_trace     = get_env("ROCPROF_HSA_IMAGE_EXT_API_TRACE", false);
    bool        hsa_finalizer_ext_api_trace = get_env("ROCPROF_HSA_FINALIZER_EXT_API_TRACE", false);
    bool        marker_api_trace            = get_env("ROCPROF_MARKER_API_TRACE", false);
    bool        memory_copy_trace           = get_env("ROCPROF_MEMORY_COPY_TRACE", false);
    bool        scratch_memory_trace        = get_env("ROCPROF_SCRATCH_MEMORY_TRACE", false);
    bool        counter_collection          = get_env("ROCPROF_COUNTER_COLLECTION", false);
    bool        hip_runtime_api_trace       = get_env("ROCPROF_HIP_RUNTIME_API_TRACE", false);
    bool        hip_compiler_api_trace      = get_env("ROCPROF_HIP_COMPILER_API_TRACE", false);
    bool        rccl_api_trace              = get_env("ROCPROF_RCCL_API_TRACE", false);
    bool        list_metrics                = get_env("ROCPROF_LIST_METRICS", false);
    bool        list_metrics_output_file    = get_env("ROCPROF_OUTPUT_LIST_METRICS_FILE", false);
    bool        stats                       = get_env("ROCPROF_STATS", false);
    bool        stats_summary               = get_env("ROCPROF_STATS_SUMMARY", false);
    bool        stats_summary_per_domain    = get_env("ROCPROF_STATS_SUMMARY_PER_DOMAIN", false);
    bool        csv_output                  = false;
    bool        json_output                 = false;
    bool        pftrace_output              = false;
    bool        otf2_output                 = false;
    bool        summary_output              = false;
    bool        kernel_rename               = get_env("ROCPROF_KERNEL_RENAME", false);
    int         mpi_size                    = get_mpi_size();
    int         mpi_rank                    = get_mpi_rank();
    size_t      perfetto_shmem_size_hint    = get_env("ROCPROF_PERFETTO_SHMEM_SIZE_HINT_KB", 64);
    size_t      perfetto_buffer_size        = get_env("ROCPROF_PERFETTO_BUFFER_SIZE_KB", 1024000);
    uint64_t    stats_summary_unit_value    = 1;
    std::string stats_summary_unit          = get_env("ROCPROF_STATS_SUMMARY_UNITS", "nsec");
    std::string output_path = get_env("ROCPROF_OUTPUT_PATH", fs::current_path().string());
    std::string output_file =
        get_env("ROCPROF_OUTPUT_FILE_NAME", fmt::format("%hostname%/{}", getpid()));
    std::string tmp_directory      = get_env("ROCPROF_TMPDIR", output_path);
    std::string stats_summary_file = get_env("ROCPROF_STATS_SUMMARY_OUTPUT", "stderr");

    std::string kernel_filter_include =
        get_env("ROCPROF_KERNEL_FILTER_INCLUDE_REGEX", std::string{".*"});
    std::string kernel_filter_exclude =
        get_env("ROCPROF_KERNEL_FILTER_EXCLUDE_REGEX", std::string{});
    std::string perfetto_buffer_fill_policy =
        get_env("ROCPROF_PERFETTO_BUFFER_FILL_POLICY", std::string{"discard"});
    std::string perfetto_backend = get_env("ROCPROF_PERFETTO_BACKEND", std::string{"inprocess"});
    std::unordered_set<uint32_t> kernel_filter_range  = {};
    std::set<std::string>        counters             = {};
    std::vector<std::string>     stats_summary_groups = {};

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

    CFG_SERIALIZE_MEMBER(demangle);
    CFG_SERIALIZE_MEMBER(truncate);
    CFG_SERIALIZE_MEMBER(kernel_trace);
    CFG_SERIALIZE_MEMBER(hsa_core_api_trace);
    CFG_SERIALIZE_MEMBER(hsa_amd_ext_api_trace);
    CFG_SERIALIZE_MEMBER(hsa_image_ext_api_trace);
    CFG_SERIALIZE_MEMBER(hsa_finalizer_ext_api_trace);
    CFG_SERIALIZE_MEMBER(marker_api_trace);
    CFG_SERIALIZE_MEMBER(memory_copy_trace);
    CFG_SERIALIZE_MEMBER(scratch_memory_trace);
    CFG_SERIALIZE_MEMBER(counter_collection);
    CFG_SERIALIZE_MEMBER(hip_runtime_api_trace);
    CFG_SERIALIZE_MEMBER(hip_compiler_api_trace);
    CFG_SERIALIZE_MEMBER(kernel_rename);

    CFG_SERIALIZE_NAMED_MEMBER("summary", stats_summary);
    CFG_SERIALIZE_NAMED_MEMBER("summary_per_domain", stats_summary_per_domain);
    CFG_SERIALIZE_NAMED_MEMBER("summary_groups", stats_summary_groups);
    CFG_SERIALIZE_NAMED_MEMBER("summary_unit", stats_summary_unit);
    CFG_SERIALIZE_NAMED_MEMBER("summary_file", stats_summary_file);

    CFG_SERIALIZE_MEMBER(perfetto_shmem_size_hint);
    CFG_SERIALIZE_MEMBER(perfetto_buffer_size);
    CFG_SERIALIZE_MEMBER(perfetto_buffer_fill_policy);
    CFG_SERIALIZE_MEMBER(perfetto_backend);

    CFG_SERIALIZE_NAMED_MEMBER("raw_tmp_directory", tmp_directory);
    CFG_SERIALIZE_NAMED_MEMBER("raw_output_path", output_path);
    CFG_SERIALIZE_NAMED_MEMBER("raw_output_file", output_file);
    CFG_SERIALIZE_NAMED_MEMBER("tmp_directory", format(tmp_directory));
    CFG_SERIALIZE_NAMED_MEMBER("output_path", format(output_path));
    CFG_SERIALIZE_NAMED_MEMBER("output_file", format(output_file));

    CFG_SERIALIZE_MEMBER(counters);
    CFG_SERIALIZE_MEMBER(kernel_filter_include);
    CFG_SERIALIZE_MEMBER(kernel_filter_exclude);
    CFG_SERIALIZE_MEMBER(kernel_filter_range);

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

struct output_key
{
    output_key(std::string _key, std::string _val, std::string _desc = {});

    template <typename Tp,
              typename Up                                                    = Tp,
              std::enable_if_t<!common::mpl::is_string_type<Up>::value, int> = 0>
    output_key(std::string _key, Tp&& _val, std::string _desc = {});

    operator std::pair<std::string, std::string>() const;

    std::string key         = {};
    std::string value       = {};
    std::string description = {};
};

template <typename Tp, typename Up, std::enable_if_t<!common::mpl::is_string_type<Up>::value, int>>
output_key::output_key(std::string _key, Tp&& _val, std::string _desc)
: key{std::move(_key)}
, value{fmt::format("{}", std::forward<Tp>(_val))}
, description{std::move(_desc)}
{}

std::vector<output_key>
output_keys(std::string _tag = {});
}  // namespace tool
}  // namespace rocprofiler
