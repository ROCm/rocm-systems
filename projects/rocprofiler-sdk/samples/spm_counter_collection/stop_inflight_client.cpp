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

#include "client.hpp"
#include "common.hpp"

#include <rocprofiler-sdk/registration.h>

#include <atomic>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <unordered_map>

namespace
{
std::atomic<size_t> dispatch_count{0};

void
record_callback(const rocprofiler_spm_dispatch_counting_service_data_t*,
                const rocprofiler_spm_counter_record_t**,
                size_t,
                rocprofiler_spm_record_flag_t,
                rocprofiler_user_data_t,
                void*)
{
    dispatch_count.fetch_add(1, std::memory_order_relaxed);
}

void
dispatch_callback(const rocprofiler_spm_dispatch_counting_service_data_t* dispatch_data,
                  rocprofiler_counter_config_id_t*                        config,
                  rocprofiler_user_data_t*,
                  void*)
{
    static std::shared_mutex                                             m_mutex       = {};
    static std::unordered_map<uint64_t, rocprofiler_counter_config_id_t> profile_cache = {};

    auto search_cache = [&]() {
        if(auto pos = profile_cache.find(dispatch_data->dispatch_info.agent_id.handle);
           pos != profile_cache.end())
        {
            *config = pos->second;
            return true;
        }
        return false;
    };

    {
        auto rlock = std::shared_lock{m_mutex};
        if(search_cache()) return;
    }

    auto wlock = std::unique_lock{m_mutex};
    if(search_cache()) return;

    auto collect_counters =
        get_matched_spm_counters(dispatch_data->dispatch_info.agent_id, {"SQ_WAVES"});

    std::vector<rocprofiler_spm_parameters_t> input_params{};
    input_params.push_back(rocprofiler_spm_parameters_t{
        .size  = sizeof(rocprofiler_spm_parameters_t),
        .type  = ROCPROFILER_SPM_PARAMETER_TYPE_SAMPLE_INTERVAL_SCLK_CYCLES,
        .value = 1200});

    auto profile = create_spm_counter_config(
        dispatch_data->dispatch_info.agent_id, collect_counters, input_params);

    profile_cache.emplace(dispatch_data->dispatch_info.agent_id.handle, profile);
    *config = profile;
}

int
tool_init(rocprofiler_client_finalize_t, void*)
{
    ROCPROFILER_CALL(rocprofiler_create_context(&get_client_ctx()), "context creation failed");

    ROCPROFILER_CALL(rocprofiler_spm_configure_callback_dispatch_service(
                         get_client_ctx(), dispatch_callback, nullptr, record_callback, nullptr),
                     "Could not setup counting service");
    ROCPROFILER_CALL(rocprofiler_start_context(get_client_ctx()), "start context");

    return 0;
}

void
tool_fini(void*)
{
    std::clog << "In tool fini, dispatches collected: " << dispatch_count.load() << "\n";
}
}  // namespace

CLIENT_API void
stop_profiling()
{
    rocprofiler_stop_context(get_client_ctx());
}

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
    id->name = "SPMStopInflightTest";

    uint32_t major = version / 10000;
    uint32_t minor = (version % 10000) / 100;
    uint32_t patch = version % 100;

    auto info = std::stringstream{};
    info << id->name << " (priority=" << priority << ") is using rocprofiler-sdk v" << major << "."
         << minor << "." << patch << " (" << runtime_version << ")";

    std::clog << info.str() << std::endl;

    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}
