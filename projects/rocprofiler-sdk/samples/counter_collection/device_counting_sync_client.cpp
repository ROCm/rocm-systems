// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <atomic>
#include <chrono>
#include <cstdlib>
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
#include <thread>
#include <unordered_map>
#include <utility>
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
// Class to sample counter values from the ROCProfiler API
// This class is not thread safe and should not be shared between threads.
// Only a single instance of this class should be created per agent.
class counter_sampler
{
public:
    // Setup system profiling for an agent
    counter_sampler(rocprofiler_agent_id_t agent);

    // Decode the counter name of a record
    std::string decode_record_name(const rocprofiler_counter_record_t& rec) const;

    // Get the dimensions of a record (what CU/SE/etc the counter is for) paired with the
    // position of the record within each dimension. High cost operation, should be cached
    // if possible.
    static std::vector<std::pair<rocprofiler_counter_record_dimension_info_t, size_t>>
    get_record_dimensions(const rocprofiler_counter_record_t& rec);

    // Sample the counter values for a set of counters, returns the records in the out parameter.
    rocprofiler_status_t sample_counter_values(const std::vector<std::string>&            counters,
                                               std::vector<rocprofiler_counter_record_t>& out,
                                               rocprofiler_user_data_t user_data);

    // Get the available agents on the system
    static std::vector<rocprofiler_agent_v0_t> get_available_agents();

    void flush() const { rocprofiler_flush_buffer(buf_); }
    void stop() const { rocprofiler_stop_context(ctx_); }

private:
    rocprofiler_agent_id_t          agent_   = {};
    rocprofiler_context_id_t        ctx_     = {};
    rocprofiler_buffer_id_t         buf_     = {};
    rocprofiler_counter_config_id_t profile_ = {.handle = 0};

    std::map<std::vector<std::string>, rocprofiler_counter_config_id_t> cached_profiles_;
    std::map<uint64_t, uint64_t>                                        profile_sizes_;
    mutable std::map<uint64_t, std::string>                             id_to_name_;

    // Internal function used to set the profile for the agent when start_context is called
    void set_profile(rocprofiler_context_id_t ctx, rocprofiler_device_counting_agent_cb_t cb) const;

    // Get the size of a counter in number of records
    static size_t get_counter_size(rocprofiler_counter_id_t counter);

    // Get the supported counters for an agent
    static std::unordered_map<std::string, rocprofiler_counter_id_t> get_supported_counters(
        rocprofiler_agent_id_t agent);

    // Get the dimensions of a counter
    static std::vector<rocprofiler_counter_record_dimension_info_t> get_counter_dimensions(
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
                            rocprofiler_device_counting_agent_cb_t set_config,
                            void*                                  user_data) {
                             if(user_data)
                             {
                                 auto* sampler = static_cast<counter_sampler*>(user_data);
                                 sampler->set_profile(context_id, set_config);
                             }
                         },
                         this),
                     "Could not setup buffered service");
}

std::string
counter_sampler::decode_record_name(const rocprofiler_counter_record_t& rec) const
{
    if(id_to_name_.empty())
    {
        auto name_to_id = counter_sampler::get_supported_counters(agent_);
        for(const auto& [name, id] : name_to_id)
        {
            id_to_name_.emplace(id.handle, name);
        }
    }

    rocprofiler_counter_id_t counter_id = {.handle = 0};
    rocprofiler_query_record_counter_id(rec.id, &counter_id);
    if(id_to_name_.find(counter_id.handle) == id_to_name_.end())
    {
        std::clog << "Unknown counter id = " << counter_id.handle << "\n";
        return "UNKNOWN";
    }
    return id_to_name_.at(counter_id.handle);
}

std::vector<std::pair<rocprofiler_counter_record_dimension_info_t, size_t>>
counter_sampler::get_record_dimensions(const rocprofiler_counter_record_t& rec)
{
    std::vector<std::pair<rocprofiler_counter_record_dimension_info_t, size_t>> out;
    rocprofiler_counter_id_t counter_id = {.handle = 0};
    rocprofiler_query_record_counter_id(rec.id, &counter_id);

    for(auto& dim : get_counter_dimensions(counter_id))
    {
        size_t pos = 0;
        rocprofiler_query_record_dimension_position(rec.id, dim.id, &pos);
        out.emplace_back(dim, pos);
    }
    return out;
}

rocprofiler_status_t
counter_sampler::sample_counter_values(const std::vector<std::string>&            counters,
                                       std::vector<rocprofiler_counter_record_t>& out,
                                       rocprofiler_user_data_t                    user_data)
{
    auto profile_cached = cached_profiles_.find(counters);
    if(profile_cached == cached_profiles_.end())
    {
        size_t                                expected_size = 0;
        rocprofiler_counter_config_id_t       profile       = {};
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
        // A partial profile would silently drop a requested counter, so treat any missing
        // counter as unavailable.
        if(gpu_counters.size() < counters.size())
        {
            out.clear();
            return ROCPROFILER_STATUS_ERROR_NO_HARDWARE_COUNTERS;
        }
        ROCPROFILER_CALL(rocprofiler_create_counter_config(
                             agent_, gpu_counters.data(), gpu_counters.size(), &profile),
                         "Could not create profile");
        cached_profiles_.emplace(counters, profile);
        profile_sizes_.emplace(profile.handle, expected_size);
        profile_cached = cached_profiles_.find(counters);
    }
    try
    {
        out.resize(profile_sizes_.at(profile_cached->second.handle));
    } catch(const std::exception& e)
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return ROCPROFILER_STATUS_ERROR;
    }
    profile_          = profile_cached->second;
    auto start_status = rocprofiler_start_context(ctx_);
    if(start_status != ROCPROFILER_STATUS_SUCCESS)
    {
        // A failed start still leaves the context active, and starting an active
        // context reports success without configuring the service. Leaving it
        // there also lets the library start the service itself once HSA
        // registers, racing this loop, so undo the attempt before returning.
        rocprofiler_stop_context(ctx_);
        out.clear();
        return start_status;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    size_t out_size = out.size();
    auto   status   = rocprofiler_sample_device_counting_service(
        ctx_, user_data, ROCPROFILER_COUNTER_FLAG_NONE, out.data(), &out_size);
    auto stop_status = rocprofiler_stop_context(ctx_);
    out.resize(out_size);
    if(status == ROCPROFILER_STATUS_SUCCESS && stop_status != ROCPROFILER_STATUS_SUCCESS)
        return stop_status;
    return status;
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
            if(rocp_agent->type == ROCPROFILER_AGENT_TYPE_GPU &&
               rocp_agent->runtime_visibility.hsa && rocp_agent->runtime_visibility.hip)
                agents_v->emplace_back(*rocp_agent);
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
counter_sampler::set_profile(rocprofiler_context_id_t               ctx,
                             rocprofiler_device_counting_agent_cb_t cb) const
{
    if(profile_.handle != 0)
    {
        cb(ctx, profile_);
    }
}

size_t
counter_sampler::get_counter_size(rocprofiler_counter_id_t counter)
{
    rocprofiler_counter_info_v1_t info;
    ROCPROFILER_CALL(rocprofiler_query_counter_info(
                         counter, ROCPROFILER_COUNTER_INFO_VERSION_1, static_cast<void*>(&info)),
                     "Could not query info for counter");
    return info.dimensions_instances_count;
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
        rocprofiler_counter_info_v0_t info;
        ROCPROFILER_CALL(
            rocprofiler_query_counter_info(
                counter, ROCPROFILER_COUNTER_INFO_VERSION_0, static_cast<void*>(&info)),
            "Could not query info for counter");
        out.emplace(info.name, counter);
    }
    return out;
}

std::vector<rocprofiler_counter_record_dimension_info_t>
counter_sampler::get_counter_dimensions(rocprofiler_counter_id_t counter)
{
    rocprofiler_counter_info_v1_t info;
    ROCPROFILER_CALL(rocprofiler_query_counter_info(
                         counter, ROCPROFILER_COUNTER_INFO_VERSION_1, static_cast<void*>(&info)),
                     "Could not query info for counter");
    return std::vector<rocprofiler_counter_record_dimension_info_t>{
        *info.dimensions, *info.dimensions + info.dimensions_count};
}

std::atomic<bool>&
exit_toggle()
{
    static std::atomic<bool> exit_toggle = false;
    return exit_toggle;
}

bool
is_terminal_status(rocprofiler_status_t status)
{
    return status == ROCPROFILER_STATUS_ERROR_FINALIZED ||
           status == ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED;
}

rocprofiler_client_finalize_t    finalize       = nullptr;
rocprofiler_client_id_t*         client_id      = nullptr;
std::shared_ptr<counter_sampler> sampler        = {};
std::thread*                     sampler_thread = nullptr;
}  // namespace

int
tool_init(rocprofiler_client_finalize_t fini_func, void* user_data)
{
    finalize            = fini_func;
    auto* output_stream = static_cast<std::ostream*>(user_data);
    if(!output_stream) throw std::runtime_error{"nullptr to output stream"};

    std::atexit([]() {
        if(client_id) finalize(*client_id);
    });

    // Get the agents available on the device
    auto agents = counter_sampler::get_available_agents();
    if(agents.empty())
    {
        std::cerr << "No agents found\n";
        return -1;
    }

    // Use the first agent found
    sampler = std::make_shared<counter_sampler>(agents[0].id);

    sampler_thread = new std::thread{[output_stream]() {
        // An exception escaping this thread would terminate the profiled application.
        try
        {
            size_t                                    count              = 1;
            bool                                      printed_dimensions = false;
            std::vector<rocprofiler_counter_record_t> records;
            while(sampler && exit_toggle().load() == false)
            {
                auto status = sampler->sample_counter_values(
                    {"SQ_WAVES", "GRBM_COUNT"}, records, {.value = count});
                if(exit_toggle().load()) break;
                if(status == ROCPROFILER_STATUS_ERROR_HSA_NOT_LOADED ||
                   status == ROCPROFILER_STATUS_ERROR_CONTEXT_ERROR ||
                   status == ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED)
                {
                    std::clog << "Device counting service unavailable, retrying....\n";
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    continue;
                }
                if(is_terminal_status(status)) break;
                if(status == ROCPROFILER_STATUS_ERROR_NO_HARDWARE_COUNTERS)
                {
                    *output_stream << "Device counting unavailable: no hardware counters\n";
                    break;
                }
                ROCPROFILER_CALL(status, "Could not sample");
                *output_stream << "Sample " << count << ":\n";
                for(const auto& record : records)
                {
                    if(!sampler) break;
                    auto recname = sampler->decode_record_name(record);
                    *output_stream << "\tCounter: " << record.id << " Name: " << recname
                                   << " Value: " << record.counter_value
                                   << " User data: " << record.user_data.value << "\n";
                    if(!printed_dimensions)
                    {
                        if(!sampler) break;
                        for(const auto& [dim, pos] : sampler->get_record_dimensions(record))
                        {
                            *output_stream << "\t\tDimension Name: " << dim.name << ": " << pos
                                           << "/" << dim.instance_size << "\n";
                        }
                    }
                }
                if(!records.empty()) printed_dimensions = true;
                count++;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        } catch(const std::exception& e)
        {
            std::cerr << "Sampler thread threw an exception: " << e.what() << "\n";
        }
        exit_toggle().store(false);
    }};

    // no errors
    return 0;
}

void
tool_fini(void* user_data)
{
    std::clog << "In tool fini\n" << std::flush;

    client_id = nullptr;

    exit_toggle().store(true);
    if(sampler_thread && sampler_thread->joinable()) sampler_thread->join();
    if(sampler)
    {
        sampler->stop();
        sampler->flush();
    }

    auto* output_stream = static_cast<std::ostream*>(user_data);
    *output_stream << std::flush;
    if(output_stream != &std::cout && output_stream != &std::cerr) delete output_stream;

    sampler.reset();
    delete sampler_thread;
    sampler_thread = nullptr;

    std::clog << "Completed tool fini\n" << std::flush;
}

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
    // set the client name
    id->name  = "CounterClientSample";
    client_id = id;

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
