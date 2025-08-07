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

#include "client.hpp"

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "common/secure_filename.hpp"

#include <atomic>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
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

/**
 * Buffer callback called when the buffer is full. rocprofiler_record_header_t
 * can contain counter records as well as other records (such as tracing). These
 * records need to be filtered based on the category type.
 */
void
buffered_callback(rocprofiler_context_id_t,
                  rocprofiler_buffer_id_t,
                  rocprofiler_record_header_t** headers,
                  size_t                        num_headers,
                  void*                         user_data,
                  uint64_t)
{
    std::stringstream ss;
    // Iterate through the returned records
    for(size_t i = 0; i < num_headers; ++i)
    {
        auto* header = headers[i];
        if(header->category == ROCPROFILER_BUFFER_CATEGORY_COUNTERS &&
           header->kind == ROCPROFILER_COUNTER_RECORD_PROFILE_COUNTING_DISPATCH_HEADER)
        {}
        else if(header->category == ROCPROFILER_BUFFER_CATEGORY_COUNTERS &&
                header->kind == ROCPROFILER_COUNTER_RECORD_VALUE)
        {
            // Print the returned counter data.
            auto* record = static_cast<rocprofiler_counter_record_t*>(header->payload);
            ss << "  (Id: " << record->id << " Value [D]: " << record->counter_value << ","
               << " user_data: " << record->user_data.value << "),";

            // Check that the agent is what we expect
            if(record->agent_id.handle != expected_agent().handle)
            {
                throw std::runtime_error("Unexpected agent - " +
                                         std::to_string(record->agent_id.handle) + " " +
                                         std::to_string(expected_agent().handle));
            }
        }
    }

    auto* output_stream = static_cast<std::ostream*>(user_data);
    if(!output_stream) throw std::runtime_error{"nullptr to output stream"};

    *output_stream << "[" << __FUNCTION__ << "] " << ss.str() << "\n";
}

std::unordered_map<uint64_t, rocprofiler_counter_config_id_t>&
get_profile_cache()
{
    static std::unordered_map<uint64_t, rocprofiler_counter_config_id_t> profile_cache;
    return profile_cache;
}

/**
 * Callback from rocprofiler when an kernel dispatch is enqueued into the HSA queue.
 * rocprofiler_counter_config_id_t* is a return to specify what counters to collect
 * for this dispatch (dispatch_packet). This example function creates a profile
 * to collect the counter SQ_WAVES for all kernel dispatch packets.
 */
void
set_profile(rocprofiler_context_id_t               context_id,
            rocprofiler_agent_id_t                 agent,
            rocprofiler_device_counting_agent_cb_t set_config,
            void*)
{
    /**
     * This simple example uses the same profile counter set for all agents.
     * We store this in a cache to prevent constructing many identical profile counter
     * sets. We first check the cache to see if we have already constructed a counter"
     * set for the agent. If we have, return it. Otherwise, construct a new profile counter
     * set.
     */
    auto search_cache = [&]() {
        if(auto pos = get_profile_cache().find(agent.handle); pos != get_profile_cache().end())
        {
            set_config(context_id, pos->second);
            return true;
        }
        return false;
    };

    if(!search_cache())
    {
        std::cerr << "No profile for agent found in cache\n";
        exit(-1);
    }
}

rocprofiler_counter_config_id_t
build_profile_for_agent(rocprofiler_agent_id_t agent)
{
    std::set<std::string>                 counters_to_collect = {"SQ_WAVES"};
    std::vector<rocprofiler_counter_id_t> gpu_counters;

    ROCPROFILER_CALL(rocprofiler_iterate_agent_supported_counters(
                         agent,
                         [](rocprofiler_agent_id_t,
                            rocprofiler_counter_id_t* counters,
                            size_t                    num_counters,
                            void*                     user_data) {
                             std::vector<rocprofiler_counter_id_t>* vec =
                                 static_cast<std::vector<rocprofiler_counter_id_t>*>(user_data);
                             for(size_t i = 0; i < num_counters; i++)
                             {
                                 vec->push_back(counters[i]);
                             }
                             return ROCPROFILER_STATUS_SUCCESS;
                         },
                         static_cast<void*>(&gpu_counters)),
                     "Could not fetch supported counters");

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
            std::clog << "Counter: " << counter.handle << " " << info.name << "\n";
            collect_counters.push_back(counter);
        }
    }

    rocprofiler_counter_config_id_t profile = {.handle = 0};
    ROCPROFILER_CALL(rocprofiler_create_counter_config(
                         agent, collect_counters.data(), collect_counters.size(), &profile),
                     "Could not construct profile cfg");

    return profile;
}

std::atomic<bool>&
exit_toggle()
{
    static std::atomic<bool> exit_toggle = false;
    return exit_toggle;
}

int
tool_init(rocprofiler_client_finalize_t, void* user_data)
{
    ROCPROFILER_CALL(rocprofiler_create_context(&get_client_ctx()), "context creation failed");

    ROCPROFILER_CALL(rocprofiler_create_buffer(get_client_ctx(),
                                               4096,
                                               2048,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               buffered_callback,
                                               user_data,
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

    // Construct the profiles in advance for each agent that is a GPU
    for(const auto& agent : agents)
    {
        if(agent.type == ROCPROFILER_AGENT_TYPE_GPU)
        {
            get_profile_cache().emplace(agent.id.handle, build_profile_for_agent(agent.id));
            expected_agent() = agent.id;
            break;
        }
    }

    if(agents.empty())
    {
        std::cerr << "No agents found" << std::endl;
        return 1;
    }

    ROCPROFILER_CALL(rocprofiler_configure_device_counting_service(
                         get_client_ctx(), get_buffer(), expected_agent(), set_profile, nullptr),
                     "Could not setup buffered service");

    std::thread([=]() {
        size_t count = 1;
        rocprofiler_start_context(get_client_ctx());
        while(exit_toggle().load() == false)
        {
            rocprofiler_sample_device_counting_service(get_client_ctx(),
                                                       {.value = count},
                                                       ROCPROFILER_COUNTER_FLAG_NONE,
                                                       nullptr,
                                                       nullptr);
            count++;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        exit_toggle().store(false);
    }).detach();

    // no errors
    return 0;
}

void
tool_fini(void* user_data)
{
    std::clog << "In tool fini\n" << std::flush;

    exit_toggle().store(true);
    while(exit_toggle().load() == true)
    {};

    rocprofiler_stop_context(get_client_ctx());
    ROCPROFILER_CALL(rocprofiler_flush_buffer(get_buffer()), "buffer flush");

    auto* output_stream = static_cast<std::ostream*>(user_data);
    *output_stream << std::flush;
    if(output_stream != &std::cout && output_stream != &std::cerr) delete output_stream;

    std::clog << "Completed tool fini\n" << std::flush;
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
    // set the client name
    id->name = "CounterClientSample";

    // compute major/minor/patch version info
    uint32_t major = version / 10000;
    uint32_t minor = (version % 10000) / 100;
    uint32_t patch = version % 100;

    // generate info string
    auto info = std::stringstream{};
    info << id->name << " (priority=" << priority << ") is using rocprofiler-sdk v" << major << "."
         << minor << "." << patch << " (" << runtime_version << ")";

    std::clog << info.str() << std::endl;

    std::ostream* output_stream = nullptr;
    std::string   filename      = "counter_collection.log";
    filename = common::secure::get_safe_output_filename(filename);
    if(filename == "stdout")
        output_stream = &std::cout;
    else if(filename == "stderr")
        output_stream = &std::cerr;
    else
        output_stream = new std::ofstream{filename};

    // create configure data
    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &tool_init,
                                            &tool_fini,
                                            static_cast<void*>(output_stream)};

    // return pointer to configure data
    return &cfg;
}
