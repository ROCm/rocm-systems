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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

// The counter configuration the dispatch counting service asks for, looked up by name and cached
// per agent: the service asks once per dispatch, and building one means iterating every counter
// the agent supports and querying each by name.
//
// The test tag is an argument rather than the kTestTag each test defines, which is declared after
// the include and so is not in scope here.

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/counter_config.h>
#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace sync_test
{
// Setup rather than the thing under test, so a failure takes the process down where it happened
// instead of being reported as a failed expectation.
inline void
abort_unless_success(rocprofiler_status_t status, const char* what, const char* test_tag)
{
    if(status == ROCPROFILER_STATUS_SUCCESS) return;

    fprintf(stderr, "[%s] %s failed: %s\n", test_tag, what, rocprofiler_get_status_string(status));
    exit(EXIT_FAILURE);
}

inline rocprofiler_counter_config_id_t
counter_config_for_agent(rocprofiler_agent_id_t agent,
                         const char*            counter_name,
                         const char*            test_tag)
{
    static auto _mutex = std::mutex{};
    static auto _cache = std::unordered_map<uint64_t, rocprofiler_counter_config_id_t>{};

    {
        auto _lk = std::lock_guard<std::mutex>{_mutex};
        if(auto itr = _cache.find(agent.handle); itr != _cache.end()) return itr->second;
    }

    auto supported = std::vector<rocprofiler_counter_id_t>{};
    abort_unless_success(
        rocprofiler_iterate_agent_supported_counters(
            agent,
            [](rocprofiler_agent_id_t, rocprofiler_counter_id_t* counters, size_t n, void* out) {
                auto* _out = static_cast<std::vector<rocprofiler_counter_id_t>*>(out);
                for(size_t i = 0; i < n; ++i)
                    _out->push_back(counters[i]);
                return ROCPROFILER_STATUS_SUCCESS;
            },
            &supported),
        "rocprofiler_iterate_agent_supported_counters",
        test_tag);

    auto selected = std::vector<rocprofiler_counter_id_t>{};
    for(auto counter : supported)
    {
        auto info = rocprofiler_counter_info_v0_t{};
        abort_unless_success(
            rocprofiler_query_counter_info(counter, ROCPROFILER_COUNTER_INFO_VERSION_0, &info),
            "rocprofiler_query_counter_info",
            test_tag);
        if(info.name != nullptr && std::string{info.name} == counter_name)
            selected.push_back(counter);
    }

    if(selected.empty())
    {
        fprintf(stderr, "[%s] %s is not available on this agent\n", test_tag, counter_name);
        exit(EXIT_FAILURE);
    }

    auto config = rocprofiler_counter_config_id_t{.handle = 0};
    abort_unless_success(
        rocprofiler_create_counter_config(agent, selected.data(), selected.size(), &config),
        "rocprofiler_create_counter_config",
        test_tag);

    auto _lk = std::lock_guard<std::mutex>{_mutex};
    _cache.emplace(agent.handle, config);
    return config;
}
}  // namespace sync_test
