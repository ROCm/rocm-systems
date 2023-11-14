#include "client.hpp"

#include <set>
#include <sstream>  // std::stringstream
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

void
test_callback(rocprofiler_queue_id_t       queue_id,
              rocprofiler_agent_t          agent_id,
              rocprofiler_correlation_id_t corr_id,
              const hsa_kernel_dispatch_packet_t*,
              void*,
              rocprofiler_record_counter_t* out_counters,
              size_t                        out_size,
              rocprofiler_profile_config_id_t)
{
    static int enter_count = 0;
    enter_count++;
    // Limit output to avoid massive log size
    if(enter_count % 100 != 0) return;

    std::stringstream ss;
    for(size_t i = 0; i < out_size; i++)
    {
        ss << "(Id: " << out_counters[i].id << " Value [D]: " << out_counters[i].derived_counter
           << ", Value [I]: " << out_counters[i].hw_counter << "),";
    }
    // Callback containing counter data.
    std::clog << "[" << __FUNCTION__ << "] " << queue_id.handle << " | " << agent_id.id.handle
              << " | " << corr_id.internal << "|" << ss.str() << "\n";
}

int
tool_init(rocprofiler_client_finalize_t, void*)
{
    std::set<std::string> counters_to_collect = {"SQ_WAVES"};

    std::vector<rocprofiler_agent_t> gpu_agents;
    auto agent_query = [](const rocprofiler_agent_t** agents, size_t num_agents, void* user_data) {
        std::vector<rocprofiler_agent_t>* vec =
            static_cast<std::vector<rocprofiler_agent_t>*>(user_data);
        for(size_t i = 0; i < num_agents; i++)
        {
            const rocprofiler_agent_t* agent = agents[i];
            if(agent->type == ROCPROFILER_AGENT_TYPE_GPU)
            {
                vec->push_back(*agent);
            }
        }
        return ROCPROFILER_STATUS_SUCCESS;
    };

    ROCPROFILER_CALL(rocprofiler_query_available_agents(
                         agent_query, sizeof(rocprofiler_agent_t), static_cast<void*>(&gpu_agents)),
                     "Could not query agents");

    ROCPROFILER_CALL(rocprofiler_create_context(&get_client_ctx()), "context creation failed");

    std::vector<rocprofiler_profile_config_id_t> profile_configs;

    for(auto& agent : gpu_agents)
    {
        std::vector<rocprofiler_counter_id_t> collect_counters;
        std::vector<rocprofiler_counter_id_t> gpu_counters;
        std::clog << agent.name << "\n";
        ROCPROFILER_CALL(
            rocprofiler_iterate_agent_supported_counters(
                agent,
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
                             agent, collect_counters.data(), collect_counters.size(), &profile),
                         "Could not construct profile cfg");
        ROCPROFILER_CALL(rocprofiler_configure_dispatch_profile_counting_service(
                             get_client_ctx(), profile, test_callback, nullptr),
                         "Could not setup dispatch service");
        profile_configs.push_back(profile);
    }
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
