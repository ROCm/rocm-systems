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

#include "lib/python/rocpd/source/types.hpp"

#include "lib/common/defines.hpp"
#include "lib/output/generator.hpp"
#include "lib/output/metadata.hpp"
#include "lib/output/node_info.hpp"
#include "lib/output/output_config.hpp"
#include "lib/output/sql/common.hpp"
#include "lib/output/stream_info.hpp"
#include "lib/rocprofiler-sdk-tool/config.hpp"

#include <rocprofiler-sdk/cxx/perfetto.hpp>

#include <vector>

namespace rocpd
{
namespace output
{
namespace tool = ::rocprofiler::tool;

struct PerfettoSession
{
    PerfettoSession(const tool::output_config&);
    ~PerfettoSession();

    std::unique_ptr<::perfetto::TracingSession> tracing_session = {};
    const tool::output_config&                  config;
};

void
write_perfetto(const PerfettoSession&                       perfetto_session,
               const types::process&                        process,
               const std::vector<types::agent>&             agent_data,
               const std::vector<types::thread>&            thread_data,
               const std::vector<types::region>&            region_data,
               const std::vector<types::sample>&            sample_data,
               const std::vector<types::kernel_dispatch>&   kernel_dispatch_data,
               const std::vector<types::memory_copies>&     memory_copy_data,
               const std::vector<types::scratch_memory>&    scratch_memory_data,
               const std::vector<types::memory_allocation>& memory_allocation_datan,
               const std::vector<types::counter>&           counter_collection_data);
}  // namespace output
}  // namespace rocpd
