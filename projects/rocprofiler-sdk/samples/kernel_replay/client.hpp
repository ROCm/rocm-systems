// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#define KR_CHECK(call)                                                                             \
    do                                                                                             \
    {                                                                                              \
        rocprofiler_status_t _s = (call);                                                          \
        if(_s != ROCPROFILER_STATUS_SUCCESS)                                                       \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "[kernel-replay] %s failed: %s\n",                                             \
                    #call,                                                                         \
                    rocprofiler_get_status_string(_s));                                            \
            std::abort();                                                                          \
        }                                                                                          \
    } while(0)

// Distinctive workgroup size used by main.cpp for the replayed kernel.
constexpr uint32_t kReplayBlockX = 67;
constexpr uint32_t kOptOutBlockX = 64;

inline std::vector<rocprofiler_agent_id_t>
gpu_agents()
{
    std::vector<rocprofiler_agent_id_t> agents{};
    KR_CHECK(rocprofiler_query_available_agents(
        ROCPROFILER_AGENT_INFO_VERSION_0,
        [](rocprofiler_agent_version_t, const void** _agents, size_t n, void* data) {
            auto* out = static_cast<std::vector<rocprofiler_agent_id_t>*>(data);
            for(size_t i = 0; i < n; ++i)
            {
                auto* agent = static_cast<const rocprofiler_agent_v0_t*>(_agents[i]);
                if(agent->type == ROCPROFILER_AGENT_TYPE_GPU) out->push_back(agent->id);
            }
            return ROCPROFILER_STATUS_SUCCESS;
        },
        sizeof(rocprofiler_agent_v0_t),
        &agents));
    return agents;
}

inline rocprofiler_counter_config_id_t
sq_waves_config(rocprofiler_agent_id_t agent)
{
    static std::mutex                                                    mutex{};
    static std::unordered_map<uint64_t, rocprofiler_counter_config_id_t> cache{};
    {
        std::lock_guard<std::mutex> lock{mutex};
        if(auto it = cache.find(agent.handle); it != cache.end()) return it->second;
    }

    std::vector<rocprofiler_counter_id_t> all{};
    KR_CHECK(rocprofiler_iterate_agent_supported_counters(
        agent,
        [](rocprofiler_agent_id_t, rocprofiler_counter_id_t* cs, size_t n, void* ud) {
            auto* v = static_cast<std::vector<rocprofiler_counter_id_t>*>(ud);
            v->insert(v->end(), cs, cs + n);
            return ROCPROFILER_STATUS_SUCCESS;
        },
        &all));

    std::vector<rocprofiler_counter_id_t> want{};
    for(auto id : all)
    {
        rocprofiler_counter_info_v0_t info{};
        KR_CHECK(rocprofiler_query_counter_info(id, ROCPROFILER_COUNTER_INFO_VERSION_0, &info));
        if(info.name && std::string{info.name} == "SQ_WAVES") want.push_back(id);
    }
    if(want.empty())
    {
        fprintf(stderr, "[kernel-replay] SQ_WAVES is not available on this agent\n");
        std::abort();
    }

    rocprofiler_counter_config_id_t cfg{.handle = 0};
    KR_CHECK(rocprofiler_create_counter_config(agent, want.data(), want.size(), &cfg));
    {
        std::lock_guard<std::mutex> lock{mutex};
        cache.emplace(agent.handle, cfg);
    }
    return cfg;
}
