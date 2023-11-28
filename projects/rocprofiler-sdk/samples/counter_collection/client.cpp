// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <fstream>
#include <iostream>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <rocprofiler/registration.h>
#include <rocprofiler/rocprofiler.h>

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
rocprofiler_context_id_t&
get_client_ctx()
{
    static rocprofiler_context_id_t ctx;
    return ctx;
}

rocprofiler_buffer_id_t&
get_buffer()
{
    static rocprofiler_buffer_id_t buf = {};
    return buf;
}

std::ostream*
get_output_stream()
{
    static std::ostream* isTerm = []() -> std::ostream* {
        if(auto* outfile = getenv("ROCPROFILER_SAMPLE_OUTPUT_FILE"))
        {
            if(std::string_view{outfile} == "stdout")
                return static_cast<std::ostream*>(&std::cout);
            else if(std::string_view{outfile} == "stderr")
                return &std::cerr;
        }
        return nullptr;
    }();
    static std::unique_ptr<std::ofstream> stream;

    if(isTerm) return isTerm;
    if(stream) return stream.get();
    std::string filename = "counter_collection.log";
    if(auto* outfile = getenv("ROCPROFILER_SAMPLE_OUTPUT_FILE"))
    {
        filename = outfile;
    }
    stream = std::make_unique<std::ofstream>(filename);
    return stream.get();
}

void
buffered_callback(rocprofiler_context_id_t,
                  rocprofiler_buffer_id_t,
                  rocprofiler_record_header_t** headers,
                  size_t                        num_headers,
                  void*,
                  uint64_t)
{
    static int enter_count = 0;
    enter_count++;
    if(enter_count % 100 != 0) return;
    std::stringstream ss;
    for(size_t i = 0; i < num_headers; ++i)
    {
        auto* header = headers[i];
        if(header->category == ROCPROFILER_BUFFER_CATEGORY_COUNTERS && header->kind == 0)
        {
            auto* record = static_cast<rocprofiler_record_counter_t*>(header->payload);
            ss << "(Id: " << record->id << " Value [D]: " << record->counter_value
               << " Corr_Id: " << record->corr_id.internal << "),";
        }
    }

    *get_output_stream() << "[" << __FUNCTION__ << "] " << ss.str() << "\n";
}

void
dispatch_callback(rocprofiler_queue_id_t /*queue_id*/,
                  const rocprofiler_agent_t* agent,
                  rocprofiler_correlation_id_t /*correlation_id*/,
                  const hsa_kernel_dispatch_packet_t* /*dispatch_packet*/,
                  void* /*callback_data_args*/,
                  rocprofiler_profile_config_id_t* config)
{
    /**
     * This simple example uses the same profile counter set for all agents.
     * We store this in a cache to prevent constructing many identical profile counter
     * sets. We first check the cache to see if we have already constructed a counter"
     * set for the agent. If we have, return it. Otherwise, construct a new profile counter
     * set.
     */
    static std::shared_mutex                                             m_mutex       = {};
    static std::unordered_map<uint64_t, rocprofiler_profile_config_id_t> profile_cache = {};

    auto search_cache = [&]() {
        if(auto pos = profile_cache.find(agent->id.handle); pos != profile_cache.end())
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

    std::set<std::string>                 counters_to_collect = {"SQ_WAVES"};
    std::vector<rocprofiler_counter_id_t> gpu_counters;
    ROCPROFILER_CALL(
        rocprofiler_iterate_agent_supported_counters(
            *agent,
            [](rocprofiler_counter_id_t* counters, size_t num_counters, void* user_data) {
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
        const char* name;
        size_t      size;
        ROCPROFILER_CALL(rocprofiler_query_counter_name(counter, &name, &size),
                         "Could not query name");
        if(counters_to_collect.count(std::string(name)) > 0)
        {
            std::clog << "Counter: " << counter.handle << " " << name << "\n";
            collect_counters.push_back(counter);
        }
    }

    rocprofiler_profile_config_id_t profile;
    ROCPROFILER_CALL(rocprofiler_create_profile_config(
                         *agent, collect_counters.data(), collect_counters.size(), &profile),
                     "Could not construct profile cfg");

    profile_cache.emplace(agent->id.handle, profile);
    *config = profile;
}

int
tool_init(rocprofiler_client_finalize_t, void*)
{
    ROCPROFILER_CALL(rocprofiler_create_context(&get_client_ctx()), "context creation failed");

    ROCPROFILER_CALL(rocprofiler_create_buffer(get_client_ctx(),
                                               4096,
                                               2048,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               buffered_callback,
                                               nullptr,
                                               &get_buffer()),
                     "buffer creation failed");

    auto client_thread = rocprofiler_callback_thread_t{};
    ROCPROFILER_CALL(rocprofiler_create_callback_thread(&client_thread),
                     "failure creating callback thread");
    get_output_stream();
    ROCPROFILER_CALL(rocprofiler_assign_callback_thread(get_buffer(), client_thread),
                     "failed to assign thread for buffer");
    ROCPROFILER_CALL(rocprofiler_configure_buffered_dispatch_profile_counting_service(
                         get_client_ctx(), get_buffer(), dispatch_callback, nullptr),
                     "Could not setup buffered service");
    rocprofiler_start_context(get_client_ctx());

    // no errors
    return 0;
}

void
tool_fini(void*)
{
    rocprofiler_stop_context(get_client_ctx());
    std::clog << "In tool fini\n";
}

}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t    version,
                      const char* runtime_version,
                      uint32_t,
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
    info << id->name << " is using rocprofiler v" << major << "." << minor << "." << patch << " ("
         << runtime_version << ")";

    std::clog << info.str() << std::endl;

    // create configure data
    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &tool_init,
                                            &tool_fini,
                                            static_cast<void*>(nullptr)};

    // return pointer to configure data
    return &cfg;
}
