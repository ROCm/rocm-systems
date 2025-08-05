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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "client.hpp"

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <unistd.h>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <sstream>
#include <vector>

#define PRINT_ONLY_FAILING false

/**
 * Tests the collection of all counters on the agent the test is run on.
 */

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

// Struct to validate that all dimension values are present. Does
// so by creating a tree of dimension values expected. If all are marked as
// having values, then all values are present in the output.
struct validate_dim_presence
{
    validate_dim_presence() {}

    void maybe_forward(const rocprofiler_counter_record_dimension_info_t& dim)
    {
        if(sub_vectors.empty())
        {
            for(size_t i = 0; i < dim.instance_size; i++)
            {
                sub_vectors.emplace_back(std::make_unique<validate_dim_presence>());
                sub_vectors.back()->vector_pos = std::make_pair(dim, i);
            }
        }
        else
        {
            for(auto& vec : sub_vectors)
            {
                vec->maybe_forward(dim);
            }
        }
    }

    void mark_seen(const rocprofiler_counter_instance_id_t& id)
    {
        if(sub_vectors.empty())
        {
            has_value = true;
            return;
        }
        size_t pos = 0;
        ROCPROFILER_CALL(rocprofiler_query_record_dimension_position(
                             id, sub_vectors.at(0)->vector_pos.first.id, &pos),
                         "Could not query position");
        sub_vectors.at(pos)->mark_seen(id);
    }

    bool check_seen(
        std::stringstream&                                                           out,
        std::vector<std::pair<rocprofiler_counter_record_dimension_info_t, size_t>>& pos_stack)
    {
        bool ret = true;
        if(sub_vectors.empty())
        {
            if(!has_value)
            {
                ret = false;
                out << "\tMissing Value at [";
            }
            else
            {
                out << "\tHas Value at [";
            }
            for(const auto& [dim, pos] : pos_stack)
            {
                out << dim.name << ":" << pos << ",";
            }
            out << "]\n";
            return ret;
        }

        for(size_t i = 0; i < sub_vectors.size(); i++)
        {
            pos_stack.push_back(sub_vectors[i]->vector_pos);
            if(!sub_vectors[i]->check_seen(out, pos_stack)) ret = false;
            pos_stack.pop_back();
        }
        return ret;
    }

    std::pair<rocprofiler_counter_record_dimension_info_t, size_t> vector_pos;
    std::vector<std::unique_ptr<validate_dim_presence>>            sub_vectors;
    bool                                                           has_value{false};
};

struct CaptureRecords
{
    std::shared_mutex m_mutex{};
    // <counter id handle, expected instances>
    std::map<uint64_t, size_t> expected{};
    // expected dims that we should see data for
    std::map<uint64_t, validate_dim_presence> expected_data_dims{};
    std::map<uint64_t, std::string>           expected_counter_names{};
    std::vector<rocprofiler_counter_id_t>     remaining{};
    // <counter_id handle, instances seen>
    std::map<uint64_t, size_t> captured{};
};

CaptureRecords* REC = new CaptureRecords;

CaptureRecords*
get_capture()
{
    return REC;
}

void
buffered_callback(rocprofiler_context_id_t,
                  rocprofiler_buffer_id_t,
                  rocprofiler_record_header_t** headers,
                  size_t                        num_headers,
                  void*,
                  uint64_t)
{
    auto& cap   = *get_capture();
    auto  wlock = std::unique_lock{cap.m_mutex};

    std::map<uint64_t, size_t> seen_counters;
    for(size_t i = 0; i < num_headers; ++i)
    {
        auto* header = headers[i];
        if(header->category == ROCPROFILER_BUFFER_CATEGORY_COUNTERS &&
           header->kind == ROCPROFILER_COUNTER_RECORD_VALUE)
        {
            // Record the counters we have in the buffer and the number of instances of
            // the counter we have seen.
            rocprofiler_counter_id_t counter;
            auto* record = static_cast<rocprofiler_counter_record_t*>(header->payload);
            rocprofiler_query_record_counter_id(record->id, &counter);
            cap.expected_data_dims.at(counter.handle).mark_seen(record->id);
            seen_counters.emplace(counter.handle, 0).first->second++;
        }
    }

    // Store these counts for post execution comparison
    for(const auto& [counter_id, instances] : seen_counters)
    {
        cap.captured.emplace(counter_id, 0).first->second += instances;
    }
}

using agent_map_t = std::map<uint64_t, const rocprofiler_agent_v0_t*>;

agent_map_t
get_agent_info()
{
    auto iterate_cb = [](rocprofiler_agent_version_t agents_ver,
                         const void**                agents_arr,
                         size_t                      num_agents,
                         void*                       user_data) {
        if(agents_ver != ROCPROFILER_AGENT_INFO_VERSION_0)
            throw std::runtime_error{"unexpected rocprofiler agent version"};

        auto* agents_v = static_cast<agent_map_t*>(user_data);
        for(size_t i = 0; i < num_agents; ++i)
        {
            const auto* itr = static_cast<const rocprofiler_agent_v0_t*>(agents_arr[i]);
            agents_v->emplace(itr->id.handle, itr);
        }
        return ROCPROFILER_STATUS_SUCCESS;
    };

    auto _agents = agent_map_t{};
    ROCPROFILER_CALL(
        rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                                           iterate_cb,
                                           sizeof(rocprofiler_agent_t),
                                           const_cast<void*>(static_cast<const void*>(&_agents))),
        "query available agents");

    return _agents;
}

void
dispatch_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                  rocprofiler_counter_config_id_t*             config,
                  rocprofiler_user_data_t* /*user_data*/,
                  void* /*callback_data_args*/)
{
    static auto agents = get_agent_info();

    auto& cap   = *get_capture();
    auto  wlock = std::unique_lock{cap.m_mutex};

    /**
     * Fetch all counters that are available for this agent if we haven't already.
     * Each of these counters will be collected 1 by 1 for each dispatch until we
     * have tried all counters. This requires the program to have at least counters
     * number of kernel launches to test all counters.
     */
    if(cap.expected.empty())
    {
        std::vector<rocprofiler_counter_id_t> counters_needed;
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
                             static_cast<void*>(&counters_needed)),
                         "Could not fetch supported counters");

        for(auto& found_counter : counters_needed)
        {
            rocprofiler_counter_info_v1_t info;

            ROCPROFILER_CALL(
                rocprofiler_query_counter_info(
                    found_counter, ROCPROFILER_COUNTER_INFO_VERSION_1, static_cast<void*>(&info)),
                "Could not query counter_id");
            cap.expected_counter_names.emplace(found_counter.handle, std::string(info.name));
            cap.remaining.push_back(found_counter);
            cap.expected.emplace(found_counter.handle, info.dimensions_instances_count);

            auto& info_vector =
                cap.expected_data_dims.emplace(found_counter.handle, validate_dim_presence{})
                    .first->second;

            for(uint64_t i = 0; i < info.dimensions_count; i++)
            {
                info_vector.maybe_forward(*info.dimensions[i]);
            }
        }
        if(cap.expected.empty())
        {
            std::clog << "No counters found for agent "
                      << dispatch_data.dispatch_info.agent_id.handle << " ("
                      << agents.at(dispatch_data.dispatch_info.agent_id.handle)->name << ")";
        }
    }
    if(cap.remaining.empty()) return;

    rocprofiler_counter_config_id_t profile = {.handle = 0};

    // Select the next counter to collect.
    if(rocprofiler_create_counter_config(
           dispatch_data.dispatch_info.agent_id, &(cap.remaining.back()), 1, &profile) ==
       ROCPROFILER_STATUS_SUCCESS)
    {
        *config = profile;
        std::clog << "Attempting to read counter "
                  << cap.expected_counter_names.at(cap.remaining.back().handle) << "\n";
    }

    cap.remaining.pop_back();
}

int
tool_init(rocprofiler_client_finalize_t, void*)
{
    get_capture();
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

    ROCPROFILER_CALL(rocprofiler_assign_callback_thread(get_buffer(), client_thread),
                     "failed to assign thread for buffer");
    ROCPROFILER_CALL(rocprofiler_configure_buffer_dispatch_counting_service(
                         get_client_ctx(), get_buffer(), dispatch_callback, nullptr),
                     "Could not setup buffered service");
    rocprofiler_start_context(get_client_ctx());

    // no errors
    return 0;
}

void
tool_fini(void*)
{
    rocprofiler_flush_buffer(get_buffer());
    rocprofiler_stop_context(get_client_ctx());
    // Flush buffer isn't waiting....
    sleep(2);

    std::clog << "In tool fini\n";

    auto& cap   = *get_capture();
    auto  wlock = std::unique_lock{cap.m_mutex};

    // Print out errors in counters that were not collected or had differences in instance
    // count information.
    if(cap.captured.size() != cap.expected.size())
    {
        std::clog << "[ERROR] Expected " << cap.expected.size() << " counters collected but got "
                  << cap.captured.size() << "\n";
    }

    for(const auto& [counter_id, expected] : cap.expected)
    {
        std::string name = "UNKNOWN";
        if(auto pos = cap.expected_counter_names.find(counter_id);
           pos != cap.expected_counter_names.end())
        {
            name = pos->second;
        }

        std::optional<size_t> actual_size;

        if(auto pos = cap.captured.find(counter_id); pos != cap.captured.end())
        {
            actual_size = pos->second;
        }

        if(actual_size && *actual_size != expected)
        {
            std::clog << (*actual_size == expected ? "" : "[ERROR]") << "Counter ID: " << counter_id
                      << " (" << name << ")"
                      << " expected " << expected << " instances and got " << *actual_size << "\n";
        }
        else if(!actual_size)
        {
            std::clog << "[ERROR] Counter ID: " << counter_id << " (" << name
                      << ") is missing from output\n";
        }
        else
        {
            // Counter collected OK
            std::stringstream                                                           ss;
            std::vector<std::pair<rocprofiler_counter_record_dimension_info_t, size_t>> stack;
            bool passed = cap.expected_data_dims.at(counter_id).check_seen(ss, stack);
            if(!PRINT_ONLY_FAILING || !passed)
            {
                std::clog << (passed ? "[OK] " : "[ERROR] ") << "Counter ID: " << counter_id << " ("
                          << name << ")"
                          << " Expected: " << expected << " Got: " << *actual_size << "\n";
                std::clog << ss.str();
            }
        }
    }
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

    // create configure data
    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &tool_init,
                                            &tool_fini,
                                            static_cast<void*>(nullptr)};

    // return pointer to configure data
    return &cfg;
}
