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

#include <atomic>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

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
// Class to sample counter values from the ROCProfiler API
// This class is not thread safe and should not be shared between threads.
// Only a single instance of this class should be created per agent.
class counter_sampler
{
public:
    // Setup system profiling for an agent
    counter_sampler(rocprofiler_agent_id_t agent);

    // Decode the counter name of a record
    const std::string& decode_record_name(const rocprofiler_record_counter_t& rec) const;

    // Get the dimensions of a record (what CU/SE/etc the counter is for). High cost operation
    // should be cached if possible.
    std::unordered_map<std::string, size_t> get_record_dimensions(
        const rocprofiler_record_counter_t& rec);

    // Sample the counter values for a set of counters, returns the records in the out parameter.
    void sample_counter_values(const std::vector<std::string>&            counters,
                               std::vector<rocprofiler_record_counter_t>& out);

    // Get the available agents on the system
    static std::vector<rocprofiler_agent_v0_t> get_available_agents();

private:
    rocprofiler_agent_id_t          agent_   = {};
    rocprofiler_context_id_t        ctx_     = {};
    rocprofiler_buffer_id_t         buf_     = {};
    rocprofiler_profile_config_id_t profile_ = {.handle = 0};

    std::map<std::vector<std::string>, rocprofiler_profile_config_id_t> cached_profiles_;
    std::map<uint64_t, uint64_t>                                        profile_sizes_;

    // Internal function used to set the profile for the agent when start_context is called
    void set_profile(rocprofiler_context_id_t                 ctx,
                     rocprofiler_agent_set_profile_callback_t cb) const;

    // Get the size of a counter in number of records
    size_t get_counter_size(rocprofiler_counter_id_t counter);

    // Get the supported counters for an agent
    static std::unordered_map<std::string, rocprofiler_counter_id_t> get_supported_counters(
        rocprofiler_agent_id_t agent);

    // Get the dimensions of a counter
    std::vector<rocprofiler_record_dimension_info_t> get_counter_dimensions(
        rocprofiler_counter_id_t counter);
};

counter_sampler::counter_sampler(rocprofiler_agent_id_t agent)
: agent_(agent)
{
    // Setup context (should only be done once per agent)
    auto client_thread = rocprofiler_callback_thread_t{};
    ROCPROFILER_CALL(rocprofiler_create_context(&ctx_), "context creation failed");

    ROCPROFILER_CALL(rocprofiler_create_buffer(
                         ctx_,
                         4096,
                         2048,
                         ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                         [](rocprofiler_context_id_t,
                            rocprofiler_buffer_id_t,
                            rocprofiler_record_header_t**,
                            size_t,
                            void*,
                            uint64_t) {},
                         nullptr,
                         &buf_),
                     "buffer creation failed");
    ROCPROFILER_CALL(rocprofiler_create_callback_thread(&client_thread),
                     "failure creating callback thread");
    ROCPROFILER_CALL(rocprofiler_assign_callback_thread(buf_, client_thread),
                     "failed to assign thread for buffer");

    ROCPROFILER_CALL(rocprofiler_configure_device_counting_service(
                         ctx_,
                         buf_,
                         agent,
                         [](rocprofiler_context_id_t context_id,
                            rocprofiler_agent_id_t,
                            rocprofiler_agent_set_profile_callback_t set_config,
                            void*                                    user_data) {
                             if(user_data)
                             {
                                 auto* sampler = static_cast<counter_sampler*>(user_data);
                                 sampler->set_profile(context_id, set_config);
                             }
                         },
                         this),
                     "Could not setup buffered service");
}

const std::string&
counter_sampler::decode_record_name(const rocprofiler_record_counter_t& rec) const
{
    static auto roc_counters = [this]() {
        auto name_to_id = counter_sampler::get_supported_counters(agent_);
        std::map<uint64_t, std::string> id_to_name;
        for(const auto& [name, id] : name_to_id)
        {
            id_to_name.emplace(id.handle, name);
        }
        return id_to_name;
    }();
    rocprofiler_counter_id_t counter_id = {.handle = 0};
    rocprofiler_query_record_counter_id(rec.id, &counter_id);
    return roc_counters.at(counter_id.handle);
}

std::unordered_map<std::string, size_t>
counter_sampler::get_record_dimensions(const rocprofiler_record_counter_t& rec)
{
    std::unordered_map<std::string, size_t> out;
    rocprofiler_counter_id_t                counter_id = {.handle = 0};
    rocprofiler_query_record_counter_id(rec.id, &counter_id);
    auto dims = get_counter_dimensions(counter_id);

    for(auto& dim : dims)
    {
        size_t pos = 0;
        rocprofiler_query_record_dimension_position(rec.id, dim.id, &pos);
        out.emplace(dim.name, pos);
    }
    return out;
}

void
counter_sampler::sample_counter_values(const std::vector<std::string>&            counters,
                                       std::vector<rocprofiler_record_counter_t>& out)
{
    auto profile_cached = cached_profiles_.find(counters);
    if(profile_cached == cached_profiles_.end())
    {
        size_t                                expected_size = 0;
        rocprofiler_profile_config_id_t       profile       = {};
        std::vector<rocprofiler_counter_id_t> gpu_counters;
        auto                                  roc_counters = get_supported_counters(agent_);
        for(const auto& counter : counters)
        {
            auto it = roc_counters.find(counter);
            if(it == roc_counters.end())
            {
                std::cerr << "Counter " << counter << " not found\n";
                continue;
            }
            gpu_counters.push_back(it->second);
            expected_size += get_counter_size(it->second);
        }
        ROCPROFILER_CALL(rocprofiler_create_profile_config(
                             agent_, gpu_counters.data(), gpu_counters.size(), &profile),
                         "Could not create profile");
        cached_profiles_.emplace(counters, profile);
        profile_sizes_.emplace(profile.handle, expected_size);
        profile_cached = cached_profiles_.find(counters);
    }

    out.resize(profile_sizes_.at(profile_cached->second.handle));
    profile_ = profile_cached->second;
    rocprofiler_start_context(ctx_);
    size_t out_size = out.size();
    rocprofiler_sample_device_counting_service(
        ctx_, {}, ROCPROFILER_COUNTER_FLAG_NONE, out.data(), &out_size);
    rocprofiler_stop_context(ctx_);
    out.resize(out_size);
}

std::vector<rocprofiler_agent_v0_t>
counter_sampler::get_available_agents()
{
    std::vector<rocprofiler_agent_v0_t>     agents;
    rocprofiler_query_available_agents_cb_t iterate_cb = [](rocprofiler_agent_version_t agents_ver,
                                                            const void**                agents_arr,
                                                            size_t                      num_agents,
                                                            void*                       udata) {
        if(agents_ver != ROCPROFILER_AGENT_INFO_VERSION_0)
            throw std::runtime_error{"unexpected rocprofiler agent version"};
        auto* agents_v = static_cast<std::vector<rocprofiler_agent_v0_t>*>(udata);
        for(size_t i = 0; i < num_agents; ++i)
        {
            const auto* rocp_agent = static_cast<const rocprofiler_agent_v0_t*>(agents_arr[i]);
            if(rocp_agent->type == ROCPROFILER_AGENT_TYPE_GPU) agents_v->emplace_back(*rocp_agent);
        }
        return ROCPROFILER_STATUS_SUCCESS;
    };

    ROCPROFILER_CALL(
        rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                                           iterate_cb,
                                           sizeof(rocprofiler_agent_t),
                                           const_cast<void*>(static_cast<const void*>(&agents))),
        "query available agents");
    return agents;
}

void
counter_sampler::set_profile(rocprofiler_context_id_t                 ctx,
                             rocprofiler_agent_set_profile_callback_t cb) const
{
    if(profile_.handle != 0)
    {
        cb(ctx, profile_);
    }
}

size_t
counter_sampler::get_counter_size(rocprofiler_counter_id_t counter)
{
    size_t size = 1;
    rocprofiler_iterate_counter_dimensions(
        counter,
        [](rocprofiler_counter_id_t,
           const rocprofiler_record_dimension_info_t* dim_info,
           size_t                                     num_dims,
           void*                                      user_data) {
            size_t* s = static_cast<size_t*>(user_data);
            for(size_t i = 0; i < num_dims; i++)
            {
                *s *= dim_info[i].instance_size;
            }
            return ROCPROFILER_STATUS_SUCCESS;
        },
        static_cast<void*>(&size));
    return size;
}

std::unordered_map<std::string, rocprofiler_counter_id_t>
counter_sampler::get_supported_counters(rocprofiler_agent_id_t agent)
{
    std::unordered_map<std::string, rocprofiler_counter_id_t> out;
    std::vector<rocprofiler_counter_id_t>                     gpu_counters;

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
    for(auto& counter : gpu_counters)
    {
        rocprofiler_counter_info_v0_t version;
        ROCPROFILER_CALL(
            rocprofiler_query_counter_info(
                counter, ROCPROFILER_COUNTER_INFO_VERSION_0, static_cast<void*>(&version)),
            "Could not query info for counter");
        out.emplace(version.name, counter);
    }
    return out;
}

std::vector<rocprofiler_record_dimension_info_t>
counter_sampler::get_counter_dimensions(rocprofiler_counter_id_t counter)
{
    std::vector<rocprofiler_record_dimension_info_t> dims;
    rocprofiler_available_dimensions_cb_t            cb =
        [](rocprofiler_counter_id_t,
           const rocprofiler_record_dimension_info_t* dim_info,
           size_t                                     num_dims,
           void*                                      user_data) {
            std::vector<rocprofiler_record_dimension_info_t>* vec =
                static_cast<std::vector<rocprofiler_record_dimension_info_t>*>(user_data);
            for(size_t i = 0; i < num_dims; i++)
            {
                vec->push_back(dim_info[i]);
            }
            return ROCPROFILER_STATUS_SUCCESS;
        };
    ROCPROFILER_CALL(rocprofiler_iterate_counter_dimensions(counter, cb, &dims),
                     "Could not iterate counter dimensions");
    return dims;
}

std::atomic<bool>&
exit_toggle()
{
    static std::atomic<bool> exit_toggle = false;
    return exit_toggle;
}
}  // namespace

int
tool_init(rocprofiler_client_finalize_t, void*)
{
    // Get the agents available on the device
    auto agents = counter_sampler::get_available_agents();
    if(agents.empty())
    {
        std::cerr << "No agents found\n";
        return -1;
    }

    // Use the first agent found
    std::shared_ptr<counter_sampler> sampler = std::make_shared<counter_sampler>(agents[0].id);

    std::thread([=]() {
        size_t                                    count = 1;
        std::vector<rocprofiler_record_counter_t> records;
        while(exit_toggle().load() == false)
        {
            sampler->sample_counter_values({"SQ_WAVES"}, records);
            std::clog << "Sample " << count << ":\n";
            for(const auto& record : records)
            {
                std::clog << "\tCounter: " << record.id
                          << " Name: " << sampler->decode_record_name(record)
                          << " Value: " << record.counter_value
                          << " User data: " << record.user_data.value << "\n";
                if(count == 1)
                {
                    auto dims = sampler->get_record_dimensions(record);
                    for(const auto& [name, pos] : dims)
                    {
                        std::clog << "\t\tDimension Name: " << name << ": " << pos << "\n";
                    }
                }
            }
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
    exit_toggle().store(true);
    while(exit_toggle().load() == true)
    {};

    auto* output_stream = static_cast<std::ostream*>(user_data);
    *output_stream << std::flush;
    if(output_stream != &std::cout && output_stream != &std::cerr) delete output_stream;
}

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

    // return pointer to configure data
    return &cfg;
}
