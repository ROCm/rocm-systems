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

#include "counter_collection.hpp"
#include "tmp_file.hpp"
#include "generateCSV.hpp"

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

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

namespace{
struct tool_data_t
{
    std::mutex    mut{};
    std::ostream* output_stream{nullptr};
};

enum class cleanup_mode
{
    destroy,
    reset,
};

struct output_data
{
    uint64_t num_output = 0;
    uint64_t num_bytes  = 0;
};

using cleanup_vec_t      = std::vector<std::function<void(cleanup_mode)>>;

rocprofiler_context_id_t&
get_client_ctx()
{
    static rocprofiler_context_id_t ctx{0};
    return ctx;
}

// rocprofiler_buffer_id_t&
// get_buffer()
// {
//     static rocprofiler_buffer_id_t buf = {};
//     return buf;
// }

tmp_file&
get_tmp_file()
{
    static tmp_file file{std::string("/home/amdtest/abhinab/iteration_multiplexing/projects/rocprofiler-compute/proof_of_concept/tmp/counters.tmp")};
    return file;
}

// agent_profiles
// generate_agent_profiles()
// {
//     std::unordered_map<rocprofiler_agent_id_t, std::vector<rocprofiler_counter_config_id_t>>
//                                                                       profiles;
//     std::unordered_map<rocprofiler_agent_id_t, std::atomic<uint64_t>> pos;
//     for(const auto& agent : get_gpu_agents())
//     {
//         for(const auto& counter_set : tool::get_config().counters)
//         {
//             if(agent->type != ROCPROFILER_AGENT_TYPE_GPU) continue;
//             auto profile = construct_counter_collection_profile(agent->id, counter_set);
//             if(profile.has_value())
//             {
//                 profiles[agent->id].push_back(profile.value());
//             }
//         }
//         pos[agent->id] = 0;
//     }
//     return agent_profiles{std::move(pos), tool::get_config().counter_groups_interval, profiles};
// }

// // this function creates a rocprofiler profile config on the first entry
// std::optional<rocprofiler_counter_config_id_t>
// get_device_counting_service(rocprofiler_agent_id_t agent_id)
// {
//     static auto agent_profiles = generate_agent_profiles();

//     auto agent_iter = agent_profiles.current_iter.find(agent_id);
//     if(agent_iter == agent_profiles.current_iter.end())
//     {
//         return std::nullopt;
//     }

//     auto my_iter = agent_iter->second.fetch_add(1);

//     const auto profiles = agent_profiles.profiles.find(agent_id);
//     if(profiles == agent_profiles.profiles.end())
//     {
//         return std::nullopt;
//     }

//     if(profiles->second.empty()) return std::nullopt;

//     uint64_t profile_pos = my_iter / agent_profiles.rotation;
//     return profiles->second[profile_pos % profiles->second.size()];
// }

void
record_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                rocprofiler_counter_record_t*                record_data,
                size_t                                       record_count,
                rocprofiler_user_data_t /* user_data */,
                void* callback_data_args)
{
    std::stringstream ss;
    ss << "Dispatch_Id=" << dispatch_data.dispatch_info.dispatch_id
       << ", Kernel_id=" << dispatch_data.dispatch_info.kernel_id
       << ", Corr_Id=" << dispatch_data.correlation_id.internal << ": ";
    for(size_t i = 0; i < record_count; ++i)
        ss << "(Id: " << record_data[i].id << " Value [D]: " << record_data[i].counter_value
           << "),";

    auto* tool = static_cast<tool_data_t*>(callback_data_args);
    if(!tool || !tool->output_stream) throw std::runtime_error{"nullptr to output stream"};

    auto _lk = std::unique_lock{tool->mut};
    // std::cerr << "[" << __FUNCTION__ << "] " << ss.str() << "\n";
    // *tool->output_stream << "[" << __FUNCTION__ << "] " << ss.str() << "\n";

    auto serialized_records      = std::vector<std::pair<rocprofiler_counter_id_t, double>>{};
    serialized_records.reserve(record_count);

    for(size_t count = 0; count < record_count; ++count)
    {
        auto _counter_id = rocprofiler_counter_id_t{};
        ROCPROFILER_CALL(rocprofiler_query_record_counter_id(record_data[count].id, &_counter_id),
                         "query record counter id");
        serialized_records.emplace_back(
            std::make_pair(_counter_id, record_data[count].counter_value));
    }

    if(!serialized_records.empty())
    {
        get_tmp_file().write(serialized_records);
        get_tmp_file().flush();
    }
}

/**
 * Callback from rocprofiler when an kernel dispatch is enqueued into the HSA queue.
 * rocprofiler_counter_config_id_t* is a return to specify what counters to collect
 * for this dispatch (dispatch_packet). This function creates a profile
 * to collect the counters for all kernel dispatch packets.
 */
void
dispatch_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                rocprofiler_counter_config_id_t*             config,
                rocprofiler_user_data_t* user_data,
                void* /*callback_data_args*/)
{
    static std::shared_mutex                                             m_mutex       = {};
    static std::unordered_map<uint64_t, rocprofiler_counter_config_id_t> profile_cache = {};

    auto search_cache = [&]() {
        if(auto pos = profile_cache.find(dispatch_data.dispatch_info.agent_id.handle);
           pos != profile_cache.end())
        {
            std::cout << "Found profile in cache for agent "
                      << dispatch_data.dispatch_info.agent_id.handle << "\n";
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

    // Counters we want to collect (here its SQ_WAVES)
    std::string counters_env_var = getenv("ROCPROF_COUNTERS");
    char delimiter = ' ';
    std::stringstream sstream_counters(counters_env_var);
    std::set<std::string> counters_to_collect = {};
    std::string counter;
    while (std::getline(sstream_counters, counter, delimiter)) {
        counters_to_collect.insert(counter);
    }

    // GPU Counter IDs
    std::vector<rocprofiler_counter_id_t> gpu_counters;

    // Iterate through the agents and get the counters available on that agent
    ROCPROFILER_CALL(rocprofiler_iterate_agent_supported_counters(
                         dispatch_data.dispatch_info.agent_id,
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
    // Look for the counters contained in counters_to_collect in gpu_counters
    for(auto& counter : gpu_counters)
    {
        rocprofiler_counter_info_v0_t info;
        ROCPROFILER_CALL(
            rocprofiler_query_counter_info(
                counter, ROCPROFILER_COUNTER_INFO_VERSION_0, static_cast<void*>(&info)),
            "Could not query info");
        if(counters_to_collect.count(std::string(info.name)) > 0)
        {
            std::clog << "Counter: " << counter.handle << " " << info.name << "\n";
            collect_counters.push_back(counter);
        }
    }

    // Create a colleciton profile for the counters
    rocprofiler_counter_config_id_t profile = {.handle = 0};
    ROCPROFILER_CALL(rocprofiler_create_counter_config(dispatch_data.dispatch_info.agent_id,
                                                       collect_counters.data(),
                                                       collect_counters.size(),
                                                       &profile),
                     "Could not construct profile cfg");

    profile_cache.emplace(dispatch_data.dispatch_info.agent_id.handle, profile);
    // Return the profile to collect those counters for this dispatch
    *config = profile;
    std::cout << "Created profile " << profile.handle << " for agent "
              << dispatch_data.dispatch_info.agent_id.handle << "\n";
}

void
generate_output(cleanup_mode _cleanup_mode){
    std::cerr << "Generating output\n";
    auto&       file = get_tmp_file();

    rocprofiler::tool::generate_csv(file);
}

/**
 * Initialize the tool. This function is called once when the tool is loaded.
 * The function is responsible for creating the context, buffer, profile configs
 * (details counters to collect on each agent), configuring the dispatch profile
 * counting service, and starting the context.
 */
int
tool_init(rocprofiler_client_finalize_t, void* user_data){
    ROCPROFILER_CALL(rocprofiler_create_context(&get_client_ctx()), "context creation failed");

    ROCPROFILER_CALL(rocprofiler_configure_callback_dispatch_counting_service(
                         get_client_ctx(), dispatch_callback, nullptr, record_callback, user_data),
                     "Could not setup counting service");
    ROCPROFILER_CALL(rocprofiler_start_context(get_client_ctx()), "start context");

    // no errors
    return 0;
}

void
tool_fini(void* user_data)
{
    assert(user_data);
    std::clog << "In tool fini\n";
    rocprofiler_stop_context(get_client_ctx());
    get_tmp_file().flush();
    generate_output(cleanup_mode::destroy);
    auto* tool_data = static_cast<tool_data_t*>(user_data);

    {
        auto  _lk           = std::unique_lock{tool_data->mut};
        auto* output_stream = tool_data->output_stream;

        *output_stream << std::flush;
        if(output_stream != &std::cout && output_stream != &std::cerr) delete output_stream;
    }

    delete tool_data;
}
} // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
    // set the client name
    id->name = "RocprofComputeCounterCollection";

    // compute major/minor/patch version info
    uint32_t major = version / 10000;
    uint32_t minor = (version % 10000) / 100;
    uint32_t patch = version % 100;

    // generate info string
    auto info = std::stringstream{};
    info << id->name << " (priority=" << priority << ") is using rocprofiler-sdk v" << major << "."
         << minor << "." << patch << " (" << runtime_version << ")";

    std::clog << info.str() << std::endl;

    auto* tool_data = new tool_data_t{};

    tool_data->output_stream = &std::cout;

    // create configure data
    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &tool_init,
                                            &tool_fini,
                                            static_cast<void*>(tool_data)};

    // return pointer to configure data
    return &cfg;
}

