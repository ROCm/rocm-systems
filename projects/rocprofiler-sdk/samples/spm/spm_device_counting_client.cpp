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

#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/experimental/spm.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <atomic>
#include <fstream>
#include <iostream>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

#define ROCPROFILER_CALL(result, msg)                                                              \
    {                                                                                              \
        rocprofiler_status_t CHECKSTATUS = result;                                                 \
        if(CHECKSTATUS != ROCPROFILER_STATUS_SUCCESS)                                              \
        {                                                                                          \
            std::string status_msg = rocprofiler_get_status_string(CHECKSTATUS);                   \
            std::cerr << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg            \
                      << " failed with error code " << CHECKSTATUS << ": " << status_msg           \
                      << std::endl;                                                                \
            std::stringstream errmsg{};                                                            \
            errmsg << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg " failure ("  \
                   << status_msg << ")";                                                           \
            throw std::runtime_error(errmsg.str());                                                \
        }                                                                                          \
    }

int
start()
{
    return 1;
}

namespace
{
rocprofiler_agent_id_t&
expected_agent()
{
    static rocprofiler_agent_id_t expected_agent = {.handle = 0};
    return expected_agent;
}

rocprofiler_context_id_t&
get_client_ctx()
{
    static rocprofiler_context_id_t ctx{0};
    return ctx;
}

rocprofiler_buffer_id_t&
get_buffer()
{
    static rocprofiler_buffer_id_t buf = {};
    return buf;
}

struct record_store
{
    std::mutex                                    mtx;
    std::vector<rocprofiler_spm_counter_record_t> records;
    std::atomic<size_t>                           total{0};
};

record_store&
get_store()
{
    static record_store s;
    return s;
}

void
buffered_callback(rocprofiler_context_id_t,
                  rocprofiler_buffer_id_t,
                  rocprofiler_record_header_t** headers,
                  size_t                        num_headers,
                  void*,
                  uint64_t)
{
    auto& store = get_store();
    for(size_t i = 0; i < num_headers; ++i)
    {
        auto* header = headers[i];
        if(!header) continue;
        if(header->category == ROCPROFILER_BUFFER_CATEGORY_COUNTERS &&
           header->kind == ROCPROFILER_COUNTER_RECORD_VALUE)
        {
            auto* record = static_cast<rocprofiler_spm_counter_record_t*>(header->payload);
            store.total.fetch_add(1, std::memory_order_relaxed);
            if(record->value > 0)
            {
                auto lk = std::lock_guard{store.mtx};
                store.records.push_back(*record);
            }
        }
    }
}

std::unordered_map<uint64_t, rocprofiler_counter_config_id_t>&
get_spm_profile_cache()
{
    static std::unordered_map<uint64_t, rocprofiler_counter_config_id_t> profile_cache;
    return profile_cache;
}

rocprofiler_counter_config_id_t
build_spm_profile_for_agent(rocprofiler_agent_id_t agent)
{
    std::set<std::string> counters_to_collect;
    counters_to_collect.insert("SQ_WAVES");
    counters_to_collect.insert("SQC_DCACHE_REQ");

    std::vector<rocprofiler_counter_id_t> gpu_counters;

    ROCPROFILER_CALL(rocprofiler_spm_iterate_agent_supported_counters(
                         agent,
                         [](rocprofiler_agent_id_t,
                            rocprofiler_counter_id_t* counters,
                            size_t                    num_counters,
                            void*                     user_data) {
                             auto* vec =
                                 static_cast<std::vector<rocprofiler_counter_id_t>*>(user_data);
                             for(size_t i = 0; i < num_counters; i++)
                             {
                                 vec->push_back(counters[i]);
                             }
                             return ROCPROFILER_STATUS_SUCCESS;
                         },
                         static_cast<void*>(&gpu_counters)),
                     "Could not fetch SPM supported counters");

    std::vector<rocprofiler_counter_id_t> collect_counters;
    for(auto& counter : gpu_counters)
    {
        rocprofiler_counter_info_v0_t info;
        ROCPROFILER_CALL(
            rocprofiler_query_counter_info(
                counter, ROCPROFILER_COUNTER_INFO_VERSION_0, static_cast<void*>(&info)),
            "Could not query info for counter");
        if(counters_to_collect.count(std::string(info.name)) > 0)
        {
            collect_counters.push_back(counter);
        }
    }

    if(collect_counters.empty())
    {
        throw std::runtime_error("No SPM-supported counters found matching requested set");
    }

    rocprofiler_spm_parameters_t interval_param = {
        .size  = sizeof(rocprofiler_spm_parameters_t),
        .type  = ROCPROFILER_SPM_PARAMETER_TYPE_SAMPLE_INTERVAL_SCLK_CYCLES,
        .value = 100};

    rocprofiler_spm_parameters_t* params[] = {&interval_param};

    rocprofiler_counter_config_id_t config = {.handle = 0};
    ROCPROFILER_CALL(
        rocprofiler_spm_create_counter_config(
            agent, collect_counters.data(), collect_counters.size(), params, 1, &config),
        "Could not construct SPM counter config");

    return config;
}

void
set_spm_profile(rocprofiler_context_id_t               context_id,
                rocprofiler_agent_id_t                 agent,
                rocprofiler_device_counting_agent_cb_t set_config,
                void*)
{
    auto search_cache = [&]() {
        if(auto pos = get_spm_profile_cache().find(agent.handle);
           pos != get_spm_profile_cache().end())
        {
            set_config(context_id, pos->second);
            return true;
        }
        return false;
    };

    if(!search_cache())
    {
        std::cerr << "No SPM profile for agent found in cache\n";
        exit(-1);
    }
}

int
tool_init(rocprofiler_client_finalize_t, void*)
{
    ROCPROFILER_CALL(rocprofiler_create_context(&get_client_ctx()), "context creation failed");

    ROCPROFILER_CALL(rocprofiler_create_buffer(get_client_ctx(),
                                               8 * 1024 * 1024,
                                               4 * 1024 * 1024,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               buffered_callback,
                                               nullptr,
                                               &get_buffer()),
                     "buffer creation failed");

    std::vector<rocprofiler_agent_v0_t>     agents;
    rocprofiler_query_available_agents_cb_t iterate_cb = [](rocprofiler_agent_version_t agents_ver,
                                                            const void**                agents_arr,
                                                            size_t                      num_agents,
                                                            void*                       udata) {
        if(agents_ver != ROCPROFILER_AGENT_INFO_VERSION_0)
            throw std::runtime_error{"unexpected rocprofiler agent version"};
        auto* agents_v = static_cast<std::vector<rocprofiler_agent_v0_t>*>(udata);
        for(size_t i = 0; i < num_agents; ++i)
            agents_v->emplace_back(*static_cast<const rocprofiler_agent_v0_t*>(agents_arr[i]));
        return ROCPROFILER_STATUS_SUCCESS;
    };

    ROCPROFILER_CALL(
        rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                                           iterate_cb,
                                           sizeof(rocprofiler_agent_t),
                                           const_cast<void*>(static_cast<const void*>(&agents))),
        "query available agents");

    auto client_thread = rocprofiler_callback_thread_t{};
    ROCPROFILER_CALL(rocprofiler_create_callback_thread(&client_thread),
                     "failure creating callback thread");
    ROCPROFILER_CALL(rocprofiler_assign_callback_thread(get_buffer(), client_thread),
                     "failed to assign thread for buffer");

    for(const auto& agent : agents)
    {
        if(agent.type == ROCPROFILER_AGENT_TYPE_GPU)
        {
            get_spm_profile_cache().emplace(agent.id.handle, build_spm_profile_for_agent(agent.id));
            expected_agent() = agent.id;
            break;
        }
    }

    if(expected_agent().handle == 0)
    {
        std::cerr << "No GPU agents found" << std::endl;
        return 1;
    }

    ROCPROFILER_CALL(
        rocprofiler_configure_spm_device_counting_service(
            get_client_ctx(), get_buffer(), expected_agent(), set_spm_profile, nullptr),
        "Could not setup SPM device counting service");

    auto start_status = rocprofiler_start_context(get_client_ctx());
    if(start_status != ROCPROFILER_STATUS_SUCCESS &&
       start_status != ROCPROFILER_STATUS_ERROR_HSA_NOT_LOADED)
    {
        ROCPROFILER_CALL(start_status, "start context failed");
    }

    return 0;
}

void
tool_fini(void*)
{
    std::clog << "In tool fini\n" << std::flush;

    rocprofiler_stop_context(get_client_ctx());
    rocprofiler_flush_buffer(get_buffer());

    auto&       store    = get_store();
    std::string filename = "spm_device_counting.log";
    if(auto* outfile = getenv("ROCPROFILER_SAMPLE_OUTPUT_FILE"); outfile) filename = outfile;

    std::ofstream out{filename};
    out << "Total records: " << store.total.load() << "\n";
    out << "Non-zero records: " << store.records.size() << "\n";
    for(const auto& r : store.records)
    {
        out << "timestamp=" << r.timestamp << " id=0x" << std::hex << r.id << std::dec
            << " value=" << r.value << "\n";
    }
    out << std::flush;

    std::clog << "Wrote " << store.records.size() << " records to " << filename << "\n";
    std::clog << "Completed tool fini\n" << std::flush;
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
    id->name = "SPMDeviceCountingSample";

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
