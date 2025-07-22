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

#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <sstream>
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

std::unordered_map<uint64_t, std::vector<rocprofiler_counter_record_dimension_info_t>>**
dimension_cache()
{
    static std::unordered_map<uint64_t, std::vector<rocprofiler_counter_record_dimension_info_t>>*
        cache;
    return &cache;
}

/**
 * For a given counter, query the dimensions that it has. Typically you will
 * want to call this function once to get the dimensions and cache them.
 */
std::vector<rocprofiler_counter_record_dimension_info_t>
counter_dimensions(rocprofiler_counter_id_t counter)
{
    if(*dimension_cache() == nullptr) return {};

    if((*dimension_cache())->count(counter.handle) > 0)
    {
        return (*dimension_cache())->at(counter.handle);
    }

    return {};
}

void
fill_dimension_cache(rocprofiler_counter_id_t counter)
{
    assert(*dimension_cache() != nullptr);
    std::vector<rocprofiler_counter_record_dimension_info_t> dims;
    rocprofiler_counter_info_v1_t                            info;
    ROCPROFILER_CALL(rocprofiler_query_counter_info(
                         counter, ROCPROFILER_COUNTER_INFO_VERSION_1, static_cast<void*>(&info)),
                     "Could not query info for counter");

    (*dimension_cache())
        ->emplace(counter.handle,
                  std::vector<rocprofiler_counter_record_dimension_info_t>{
                      *info.dimensions, *info.dimensions + info.dimensions_count});
}

/**
 * buffered_callback (set in rocprofiler_create_buffer in tool_init) is called when the
 * buffer is full (or when the buffer is flushed). The callback is responsible for processing
 * the records in the buffer. The records are returned in the headers array. The headers
 * can contain counter records as well as other records (such as tracing). These
 * records need to be filtered based on the category type. For counter collection,
 * they should be filtered by category == ROCPROFILER_BUFFER_CATEGORY_COUNTERS.
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
        {
            // Print the returned counter data.
            auto* record =
                static_cast<rocprofiler_dispatch_counting_service_record_t*>(header->payload);
            ss << "[Dispatch_Id: " << record->dispatch_info.dispatch_id
               << " Kernel_ID: " << record->dispatch_info.kernel_id
               << " Corr_Id: " << record->correlation_id.internal << ")]\n";
        }
        else if(header->category == ROCPROFILER_BUFFER_CATEGORY_COUNTERS &&
                header->kind == ROCPROFILER_COUNTER_RECORD_VALUE)
        {
            // Print the returned counter data.
            auto* record = static_cast<rocprofiler_counter_record_t*>(header->payload);
            rocprofiler_counter_id_t counter_id = {.handle = 0};

            rocprofiler_query_record_counter_id(record->id, &counter_id);

            ss << "  (Dispatch_Id: " << record->dispatch_id << " Counter_Id: " << counter_id.handle
               << " Record_Id: " << record->id << " Dimensions: [";

            for(auto& dim : counter_dimensions(counter_id))
            {
                size_t pos = 0;
                rocprofiler_query_record_dimension_position(record->id, dim.id, &pos);
                ss << "{" << dim.name << ": " << pos << "},";
            }
            ss << "] Value [D]: " << record->counter_value << "),";
        }
    }

    auto* output_stream = static_cast<std::ostream*>(user_data);
    if(!output_stream) throw std::runtime_error{"nullptr to output stream"};

    *output_stream << "[" << __FUNCTION__ << "] " << ss.str() << "\n";
}

/**
 * Cache to store the profile configs for each agent. This is used to prevent
 * constructing the same profile config multiple times. Used by dispatch_callback
 * to select the profile config (and in turn counters) to use when a kernel dispatch
 * is received.
 */
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
dispatch_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                  rocprofiler_counter_config_id_t*             config,
                  rocprofiler_user_data_t* /*user_data*/,
                  void* /*callback_data_args*/)
{
    /**
     * This simple example uses the same profile counter set for all agents.
     * We store this in a cache to prevent constructing many identical profile counter
     * sets.
     */
    auto search_cache = [&]() {
        if(auto pos = get_profile_cache().find(dispatch_data.dispatch_info.agent_id.handle);
           pos != get_profile_cache().end())
        {
            *config = pos->second;
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

/**
 * Construct a profile config for an agent. This function takes an agent (obtained from
 * get_gpu_device_agents()) and a set of counter names to collect. It returns a profile
 * that can be used when a dispatch is received for the agent to collect the specified
 * counters. Note: while you can dynamically create these profiles, it is more efficient
 * to consturct them once in advance (i.e. in tool_init()) since there are non-trivial
 * costs associated with constructing the profile.
 */
rocprofiler_counter_config_id_t
build_profile_for_agent(rocprofiler_agent_id_t       agent,
                        const std::set<std::string>& counters_to_collect)
{
    std::vector<rocprofiler_counter_id_t> gpu_counters;

    // Iterate all the counters on the agent and store them in gpu_counters.
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

    // Find the counters we actually want to collect (i.e. those in counters_to_collect)
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
            fill_dimension_cache(counter);
        }
    }

    // Create and return the profile
    rocprofiler_counter_config_id_t profile = {.handle = 0};
    ROCPROFILER_CALL(rocprofiler_create_counter_config(
                         agent, collect_counters.data(), collect_counters.size(), &profile),
                     "Could not construct profile cfg");

    return profile;
}

/**
 * Returns all GPU agents visible to rocprofiler on the system
 */
std::vector<rocprofiler_agent_v0_t>
get_gpu_device_agents()
{
    std::vector<rocprofiler_agent_v0_t> agents;

    // Callback used by rocprofiler_query_available_agents to return
    // agents on the device. This can include CPU agents as well. We
    // select GPU agents only (i.e. type == ROCPROFILER_AGENT_TYPE_GPU)
    rocprofiler_query_available_agents_cb_t iterate_cb = [](rocprofiler_agent_version_t agents_ver,
                                                            const void**                agents_arr,
                                                            size_t                      num_agents,
                                                            void*                       udata) {
        if(agents_ver != ROCPROFILER_AGENT_INFO_VERSION_0)
            throw std::runtime_error{"unexpected rocprofiler agent version"};
        auto* agents_v = static_cast<std::vector<rocprofiler_agent_v0_t>*>(udata);
        for(size_t i = 0; i < num_agents; ++i)
        {
            const auto* agent = static_cast<const rocprofiler_agent_v0_t*>(agents_arr[i]);
            if(agent->type == ROCPROFILER_AGENT_TYPE_GPU) agents_v->emplace_back(*agent);
        }
        return ROCPROFILER_STATUS_SUCCESS;
    };

    // Query the agents, only a single callback is made that contains a vector
    // of all agents.
    ROCPROFILER_CALL(
        rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                                           iterate_cb,
                                           sizeof(rocprofiler_agent_t),
                                           const_cast<void*>(static_cast<const void*>(&agents))),
        "query available agents");
    return agents;
}

/**
 * Initialize the tool. This function is called once when the tool is loaded.
 * The function is responsible for creating the context, buffer, profile configs
 * (details counters to collect on each agent), configuring the dispatch profile
 * counting service, and starting the context.
 */
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

    // Get a vector of all GPU devices on the system.
    auto agents = get_gpu_device_agents();

    if(agents.empty())
    {
        std::cerr << "No agents found" << std::endl;
        return 1;
    }

    // Construct the profiles in advance for each agent that is a GPU
    for(const auto& agent : agents)
    {
        // get_profile_cache() is a map that can be accessed by dispatch_callback
        // below to select the profile config to use when a kernel dispatch is
        // recieved.
        get_profile_cache().emplace(
            agent.id.handle, build_profile_for_agent(agent.id, std::set<std::string>{"TCC_HIT"}));
    }

    auto client_thread = rocprofiler_callback_thread_t{};
    // Create the callback thread
    ROCPROFILER_CALL(rocprofiler_create_callback_thread(&client_thread),
                     "failure creating callback thread");
    // Create the buffer and assign the callback thread to the buffer, when the buffer is full
    // a callback will be issued (to client_thread)
    ROCPROFILER_CALL(rocprofiler_assign_callback_thread(get_buffer(), client_thread),
                     "failed to assign thread for buffer");

    // Setup the dispatch profile counting service. This service will trigger the dispatch_callback
    // when a kernel dispatch is enqueued into the HSA queue. The callback will specify what
    // counters to collect by returning a profile config id. In this example, we create the profile
    // configs above and store them in the map get_profile_cache() so we can look them up at
    // dispatch.
    ROCPROFILER_CALL(rocprofiler_configure_buffer_dispatch_counting_service(
                         get_client_ctx(), get_buffer(), dispatch_callback, nullptr),
                     "Could not setup buffered service");

    // Start the context (start intercepting kernel dispatches).
    ROCPROFILER_CALL(rocprofiler_start_context(get_client_ctx()), "start context");

    // no errors
    return 0;
}

void
tool_fini(void* user_data)
{
    std::clog << "In tool fini\n";

    // Flush the buffer and stop the context
    ROCPROFILER_CALL(rocprofiler_flush_buffer(get_buffer()), "buffer flush");
    rocprofiler_stop_context(get_client_ctx());

    auto* output_stream = static_cast<std::ostream*>(user_data);
    *output_stream << std::flush;
    if(output_stream != &std::cout && output_stream != &std::cerr) delete output_stream;

    auto* tmp_ptr      = *dimension_cache();
    *dimension_cache() = nullptr;
    delete tmp_ptr;
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
    if(auto* outfile = getenv("ROCPROFILER_SAMPLE_OUTPUT_FILE"); outfile) filename = outfile;
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

    *dimension_cache() =
        new std::unordered_map<uint64_t,
                               std::vector<rocprofiler_counter_record_dimension_info_t>>();

    // return pointer to configure data
    return &cfg;
}
