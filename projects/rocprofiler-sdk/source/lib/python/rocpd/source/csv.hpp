// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "lib/python/rocpd/source/types.hpp"

#include "lib/common/defines.hpp"
#include "lib/output/generateStats.hpp"
#include "lib/output/generator.hpp"
#include "lib/output/metadata.hpp"
#include "lib/output/node_info.hpp"
#include "lib/output/output_config.hpp"
#include "lib/output/sql/common.hpp"
#include "lib/output/stream_info.hpp"
#include "lib/rocprofiler-sdk-tool/config.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <filesystem>
#include <fstream>
#include <map>
#include <vector>

namespace rocpd
{
namespace output
{
using rocprofiler::tool::float_type;

struct CsvFileConfig
{
    std::string filename;
    std::string header;
};

enum class CsvType
{
    KERNEL_DISPATCH,
    MEMORY_COPY,
    MEMORY_ALLOCATION,
    SCRATCH_MEMORY,
    HIP_API,
    HSA_CSV_API,
    MARKER,
    COUNTER,
    RCCL_API,
    ROCDECODE_API,
    ROCJPEG_API,
};

class CsvManager
{
public:
    CsvManager(rocprofiler::tool::output_config output_cfg);
    ~CsvManager();

    rocprofiler::tool::output_config config;
    std::map<CsvType, CsvFileConfig> csv_configs;

    std::ofstream& get_stream(CsvType type);

    bool has_stream(CsvType type) const;
    bool initialize_csv_file(CsvType type);

    template <typename... Args>
    void write_line(CsvType type, Args&&... args)
    {
        auto& stream = get_stream(type);
        if(!stream.is_open()) return;

        std::vector<std::string> items;
        (items.push_back(fmt::format("{}", std::forward<Args>(args))), ...);
        stream << fmt::format("{}\n", fmt::join(items, ","));
    }

private:
    std::map<CsvType, std::ofstream> streams;
    std::map<CsvType, std::string>   file_paths;

    bool ensure_output_directory() const;
};

void
write_agent_info_csv(CsvManager& csv_manager, const std::vector<rocpd::types::agent>& agents);

void
write_csvs(CsvManager&                                                          csv_manager,
           const rocprofiler::tool::generator<rocpd::types::kernel_dispatch>&   kernel_dispatch,
           const rocprofiler::tool::generator<rocpd::types::memory_copies>&     memory_copies,
           const rocprofiler::tool::generator<rocpd::types::memory_allocation>& memory_allocations,
           const rocprofiler::tool::generator<rocpd::types::region>&            hip_api_calls,
           const rocprofiler::tool::generator<rocpd::types::region>&            hsa_api_calls,
           const rocprofiler::tool::generator<rocpd::types::region>&            marker_api_calls,
           const rocprofiler::tool::generator<rocpd::types::counter>&           counters_calls,
           const rocprofiler::tool::generator<rocpd::types::scratch_memory>& scratch_memory_calls,
           const rocprofiler::tool::generator<rocpd::types::region>&         rccl_calls,
           const rocprofiler::tool::generator<rocpd::types::region>&         rocdecode_calls,
           const rocprofiler::tool::generator<rocpd::types::region>&         rocjpeg_calls);
}  // namespace output
}  // namespace rocpd
