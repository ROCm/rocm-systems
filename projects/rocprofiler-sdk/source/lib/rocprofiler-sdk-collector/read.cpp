// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk-collector/read.hpp"

#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk-collector/ctf/reader.hpp"
#include "lib/rocprofiler-sdk-collector/join.hpp"
#include "lib/rocprofiler-sdk-collector/schema.hpp"

#include <dlfcn.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <vector>

namespace rocprofiler
{
namespace collector
{
namespace
{
namespace fs = std::filesystem;

constexpr auto hip_provider = "rocm_hip";
}  // namespace

std::string
resolve_schema_dir(const char* schema_dir)
{
    if(schema_dir != nullptr && schema_dir[0] != '\0') return schema_dir;
    if(const auto* env = std::getenv("ROCPROFILER_COLLECTOR_SCHEMA_DIR"); env && env[0] != '\0')
        return env;
    // <prefix>/lib/librocprofiler-sdk-collector.so -> <prefix>/share/rocm-trace
    auto info = Dl_info{};
    if(::dladdr(reinterpret_cast<void*>(&resolve_schema_dir), &info) != 0 &&  // NOLINT
       info.dli_fname != nullptr)
    {
        auto library = fs::path{info.dli_fname};
        return (library.parent_path().parent_path() / "share" / "rocm-trace").string();
    }
    return {};
}

rocprofiler_collector_status_t
read_hip_api_calls(const read_request&                      request,
                   string_store*                            strings,
                   rocprofiler_collector_read_statistics_t* statistics)
{
    auto stats = rocprofiler_collector_read_statistics_t{};
    std::memset(&stats, 0, sizeof(stats));
    stats.size         = sizeof(stats);
    stats.clock_stable = request.clocks.stable ? 1 : 0;

    auto schema      = probe_schema{};
    auto error       = std::string{};
    auto schema_path = (fs::path{request.schema_dir} / (std::string{hip_provider} + ".json"));
    if(!load_probe_schema(schema_path.string(), &schema, &error))
    {
        ROCP_ERROR << "[rocprofiler-sdk-collector] " << error;
        return ROCPROFILER_COLLECTOR_STATUS_ERROR_PROBE_SCHEMA;
    }

    auto traces = std::vector<ctf::trace_ptr>{};
    for(const auto& dir : ctf::find_traces(request.trace_path))
    {
        auto source = std::make_shared<ctf::trace>();
        if(!ctf::load_trace(dir, source.get(), &error))
        {
            ROCP_ERROR << "[rocprofiler-sdk-collector] " << error;
            return ROCPROFILER_COLLECTOR_STATUS_ERROR_TRACE;
        }
        traces.push_back(source);
    }

    auto on_call = [&](const api_call& call) {
        if(call.provider != hip_provider) return;
        auto enter = schema.events.find(call.name + "_enter");
        auto exit  = schema.events.find(call.name + "_exit");
        if(enter == schema.events.end())
        {
            ++stats.unknown_calls;
            return;
        }
        auto record = rocprofiler_buffer_tracing_hip_api_ext_record_t{};
        auto status = make_hip_api_record(call,
                                          enter->second,
                                          (exit == schema.events.end()) ? nullptr : &exit->second,
                                          request.clocks,
                                          strings,
                                          &record);
        if(!status.known_api)
        {
            ++stats.unknown_calls;
            return;
        }

        auto info = rocprofiler_collector_call_info_t{};
        std::memset(&info, 0, sizeof(info));
        info.size = sizeof(info);
        info.pid  = static_cast<uint32_t>(call.pid);
        if(call.end == no_end)
        {
            info.flags |= ROCPROFILER_COLLECTOR_CALL_FLAG_NO_END;
            ++stats.calls_no_end;
        }
        if(status.missing_args > 0)
        {
            info.flags |= ROCPROFILER_COLLECTOR_CALL_FLAG_ARGS_INCOMPLETE;
            ++stats.calls_args_incomplete;
        }
        ++stats.calls;
        if(request.hip_api_callback) request.hip_api_callback(&record, &info, request.user_data);
    };

    auto joined = join_statistics{};
    if(!join_api_calls(traces, on_call, nullptr, &joined, &error))
    {
        ROCP_ERROR << "[rocprofiler-sdk-collector] " << error;
        return ROCPROFILER_COLLECTOR_STATUS_ERROR_TRACE;
    }
    stats.events           = joined.read.events;
    stats.discarded_events = joined.read.discarded;
    stats.unmatched_exits  = joined.unmatched_exits;
    stats.ambiguous_exits  = joined.ambiguous_exits;
    if(statistics != nullptr) *statistics = stats;
    return ROCPROFILER_COLLECTOR_STATUS_SUCCESS;
}
}  // namespace collector
}  // namespace rocprofiler
