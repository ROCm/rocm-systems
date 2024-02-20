#include "client.hpp"

#include <unistd.h>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <shared_mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#define PRINT_ONLY_FAILING true

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
    static rocprofiler_context_id_t ctx;
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

    void maybe_forward(const rocprofiler_record_dimension_info_t& dim)
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

    bool check_seen(std::stringstream&                                                   out,
                    std::vector<std::pair<rocprofiler_record_dimension_info_t, size_t>>& pos_stack)
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

    std::pair<rocprofiler_record_dimension_info_t, size_t> vector_pos;
    std::vector<std::unique_ptr<validate_dim_presence>>    sub_vectors;
    bool                                                   has_value{false};
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
        if(header->category == ROCPROFILER_BUFFER_CATEGORY_COUNTERS && header->kind == 0)
        {
            // Record the counters we have in the buffer and the number of instances of
            // the counter we have seen.
            rocprofiler_counter_id_t counter;
            auto* record = static_cast<rocprofiler_record_counter_t*>(header->payload);
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

void
dispatch_callback(rocprofiler_queue_id_t /*queue_id*/,
                  const rocprofiler_agent_t* agent,
                  rocprofiler_correlation_id_t /*correlation_id*/,
                  const hsa_kernel_dispatch_packet_t* /*dispatch_packet*/,
                  uint64_t /*kernel_id*/,
                  void* /*callback_data_args*/,
                  rocprofiler_profile_config_id_t* config)
{
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
                             agent->id,
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
            rocprofiler_counter_info_v0_t version;

            ROCPROFILER_CALL(rocprofiler_query_counter_info(found_counter,
                                                            ROCPROFILER_COUNTER_INFO_VERSION_0,
                                                            static_cast<void*>(&version)),
                             "Could not query counter_id");
            cap.expected_counter_names.emplace(found_counter.handle, std::string(version.name));
            size_t expected = 0;
            ROCPROFILER_CALL(
                rocprofiler_query_counter_instance_count(agent->id, found_counter, &expected),
                "COULD NOT QUERY INSTANCES");
            cap.remaining.push_back(found_counter);
            cap.expected.emplace(found_counter.handle, expected);

            auto& info_vector =
                cap.expected_data_dims.emplace(found_counter.handle, validate_dim_presence{})
                    .first->second;

            ROCPROFILER_CALL(rocprofiler_iterate_counter_dimensions(
                                 found_counter,
                                 [](rocprofiler_counter_id_t,
                                    const rocprofiler_record_dimension_info_t* dim_info,
                                    size_t                                     num_dims,
                                    void*                                      user_data) {
                                     validate_dim_presence* dim_presence =
                                         static_cast<validate_dim_presence*>(user_data);
                                     for(size_t i = 0; i < num_dims; i++)
                                     {
                                         dim_presence->maybe_forward(dim_info[i]);
                                     }
                                     return ROCPROFILER_STATUS_SUCCESS;
                                 },
                                 static_cast<void*>(&info_vector)),
                             "Could not fetch dimension info");
        }
        if(cap.expected.empty())
        {
            std::clog << "No counters found for agent - " << agent->name;
        }
    }
    if(cap.remaining.empty()) return;

    rocprofiler_profile_config_id_t profile;

    // Select the next counter to collect.
    ROCPROFILER_CALL(
        rocprofiler_create_profile_config(agent->id, &(cap.remaining.back()), 1, &profile),
        "Could not construct profile cfg");

    cap.remaining.pop_back();
    *config = profile;
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
            std::stringstream                                                   ss;
            std::vector<std::pair<rocprofiler_record_dimension_info_t, size_t>> stack;
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
    info << id->name << " is using rocprofiler-sdk v" << major << "." << minor << "." << patch
         << " (" << runtime_version << ")";

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
