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

#include "format_path.hpp"
#include "metadata.hpp"

#include "lib/common/environment.hpp"
#include "lib/common/filesystem.hpp"
#include "lib/common/mpl.hpp"
#include "lib/common/units.hpp"

#include <rocprofiler-sdk/cxx/details/tokenize.hpp>
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
namespace defaults
{
constexpr auto perfetto_buffer_size_kb     = (1 * common::units::GiB) / common::units::KiB;
constexpr auto perfetto_shmem_size_hint_kb = 64;
}  // namespace defaults

struct output_config
{
    output_config()                         = default;
    ~output_config()                        = default;
    output_config(const output_config&)     = default;
    output_config(output_config&&) noexcept = default;
    output_config& operator=(const output_config&) = default;
    output_config& operator=(output_config&&) noexcept = default;

    bool                     stats                       = false;
    bool                     stats_summary               = false;
    bool                     stats_summary_per_domain    = false;
    bool                     csv_output                  = false;
    bool                     json_output                 = false;
    bool                     pftrace_output              = false;
    bool                     otf2_output                 = false;
    bool                     summary_output              = false;
    bool                     kernel_rename               = false;
    bool                     group_by_queue              = false;
    uint64_t                 stats_summary_unit_value    = 1;
    size_t                   perfetto_shmem_size_hint    = defaults::perfetto_shmem_size_hint_kb;
    size_t                   perfetto_buffer_size        = defaults::perfetto_buffer_size_kb;
    agent_indexing           agent_index_value           = agent_indexing::logical_node;
    std::string              stats_summary_unit          = "nsec";
    std::string              output_path                 = "%cwd%";
    std::string              output_file                 = "%hostname%/%pid%";
    std::string              tmp_directory               = output_path;
    std::string              stats_summary_file          = "stderr";
    std::string              perfetto_backend            = "inprocess";
    std::string              perfetto_buffer_fill_policy = "discard";
    std::vector<std::string> stats_summary_groups        = {};

    template <typename ArchiveT>
    void save(ArchiveT&) const;

    template <typename ArchiveT>
    void load(ArchiveT&)
    {}

    static output_config load_from_env();
    static output_config load_from_env(output_config&&);

private:
    void parse_env();

    std::string output_format = "ROCPD";
};

template <typename ArchiveT>
void
output_config::save(ArchiveT& ar) const
{
#define CFG_SERIALIZE_MEMBER(VAR)             ar(cereal::make_nvp(#VAR, VAR))
#define CFG_SERIALIZE_NAMED_MEMBER(NAME, VAR) ar(cereal::make_nvp(NAME, VAR))

    CFG_SERIALIZE_NAMED_MEMBER("output_path", format_path(output_path));
    CFG_SERIALIZE_NAMED_MEMBER("output_file", format_path(output_file));
    CFG_SERIALIZE_NAMED_MEMBER("tmp_directory", format_path(tmp_directory));
    CFG_SERIALIZE_NAMED_MEMBER("raw_output_path", output_path);
    CFG_SERIALIZE_NAMED_MEMBER("raw_output_file", output_file);
    CFG_SERIALIZE_NAMED_MEMBER("raw_tmp_directory", tmp_directory);

    CFG_SERIALIZE_MEMBER(perfetto_shmem_size_hint);
    CFG_SERIALIZE_MEMBER(perfetto_buffer_size);
    CFG_SERIALIZE_MEMBER(perfetto_buffer_fill_policy);
    CFG_SERIALIZE_MEMBER(perfetto_backend);

    CFG_SERIALIZE_NAMED_MEMBER("summary", stats_summary);
    CFG_SERIALIZE_NAMED_MEMBER("summary_per_domain", stats_summary_per_domain);
    CFG_SERIALIZE_NAMED_MEMBER("summary_groups", stats_summary_groups);
    CFG_SERIALIZE_NAMED_MEMBER("summary_unit", stats_summary_unit);
    CFG_SERIALIZE_NAMED_MEMBER("summary_file", stats_summary_file);

    CFG_SERIALIZE_MEMBER(csv_output);
    CFG_SERIALIZE_MEMBER(json_output);
    CFG_SERIALIZE_MEMBER(pftrace_output);
    CFG_SERIALIZE_MEMBER(otf2_output);
    CFG_SERIALIZE_MEMBER(summary_output);
    CFG_SERIALIZE_MEMBER(kernel_rename);
    CFG_SERIALIZE_MEMBER(group_by_queue);

#undef CFG_SERIALIZE_MEMBER
#undef CFG_SERIALIZE_NAMED_MEMBER
}
}  // namespace tool
}  // namespace rocprofiler
