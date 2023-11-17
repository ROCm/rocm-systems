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

#include <rocprofiler/registration.h>
#include <rocprofiler/rocprofiler.h>

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

struct CaptureRecords
{
    std::shared_mutex m_mutex{};
    // <counter id handle, expected instances>
    std::map<uint64_t, size_t>            expected{};
    std::map<uint64_t, std::string>       expected_counter_names{};
    std::vector<rocprofiler_counter_id_t> remaining{};
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
    auto&                      cap   = *get_capture();
    auto                       wlock = std::unique_lock{cap.m_mutex};
    std::map<uint64_t, size_t> seen_counters;
    for(size_t i = 0; i < num_headers; ++i)
    {
        auto* header = headers[i];
        if(header->category == ROCPROFILER_BUFFER_CATEGORY_COUNTERS && header->kind == 0)
        {
            rocprofiler_counter_id_t counter;
            auto* record = static_cast<rocprofiler_record_counter_t*>(header->payload);
            rocprofiler_query_record_counter_id(record->id, &counter);
            if(counter.handle == 517)
            {
                std::clog << "HERE";
            }
            seen_counters.emplace(counter.handle, 0).first->second++;
        }
    }

    for(const auto& [counter_id, instances] : seen_counters)
    {
        cap.captured.emplace(counter_id, 0).first->second += instances;
    }
}

void
dispatch_callback(rocprofiler_queue_id_t              queue_id,
                  const rocprofiler_agent_t*          agent,
                  rocprofiler_correlation_id_t        correlation_id,
                  const hsa_kernel_dispatch_packet_t* dispatch_packet,
                  void*                               callback_data_args,
                  rocprofiler_profile_config_id_t*    config)
{
    auto& cap   = *get_capture();
    auto  wlock = std::unique_lock{cap.m_mutex};

    if(cap.expected.empty())
    {
        std::vector<rocprofiler_counter_id_t> counters_needed;
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
                static_cast<void*>(&counters_needed)),
            "Could not fetch supported counters");

        for(auto& found_counter : counters_needed)
        {
            size_t expected = 0;
            rocprofiler_query_counter_instance_count(*agent, found_counter, &expected);
            cap.remaining.push_back(found_counter);
            cap.expected.emplace(found_counter.handle, expected);
            const char* name;
            size_t      name_size;
            ROCPROFILER_CALL(rocprofiler_query_counter_name(found_counter, &name, &name_size),
                             "Could not query name");
            cap.expected_counter_names.emplace(found_counter.handle, std::string(name));
        }
        if(cap.expected.empty())
        {
            std::clog << "No counters found for agent - " << agent->name;
        }
    }
    if(cap.remaining.empty()) return;

    rocprofiler_profile_config_id_t profile;

    ROCPROFILER_CALL(
        rocprofiler_create_profile_config(*agent, &(cap.remaining.back()), 1, &profile),
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

    auto  cap_ptr = get_capture();
    auto& cap     = *get_capture();
    auto  wlock   = std::unique_lock{cap.m_mutex};

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
        else
        {
            std::clog << "[ERROR] Counter ID: " << counter_id << " (" << name
                      << ") is missing from output\n";
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
