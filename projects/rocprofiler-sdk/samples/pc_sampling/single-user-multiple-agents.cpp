// Vladimir: The example that shows how a single user can use PC sampling
// on multiple GPU agents.

#include <rocprofiler/rocprofiler.h>
#include <string.h>
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "common.h"

namespace
{
// GPU agents supporting some kind of PC sampling
std::vector<rocprofiler_agent_t>      gpu_agents;
std::vector<rocprofiler_context_id_t> contexts;
std::vector<rocprofiler_buffer_id_t>  buffer_ids;

rocprofiler_status_t
find_all_gpu_agents_supporting_pc_sampling_impl(const rocprofiler_agent_t** agents,
                                                size_t                      num_agents,
                                                void*                       data)
{
    // data is required
    if(!data) return ROCPROFILER_STATUS_ERROR;

    auto* _out_agents = static_cast<std::vector<rocprofiler_agent_t>*>(data);
    // find the first GPU agent
    for(size_t i = 0; i < num_agents; i++)
    {
        if(agents[i]->type == ROCPROFILER_AGENT_TYPE_GPU)
        {
            // Skip GPU agents not supporting PC sampling
            // Vladimir: The assumption is that if a GPU agent does not support PC sampling,
            // the size is 0.
            if(agents[i]->pc_sampling_configs.size == 0) continue;

            _out_agents->push_back(*agents[i]);

            printf("[%s] %s :: id=%zu, type=%i, num pc sample configs=%zu\n",
                   __FUNCTION__,
                   agents[i]->name,
                   agents[i]->id.handle,
                   agents[i]->type,
                   agents[i]->pc_sampling_configs.size);
            return ROCPROFILER_STATUS_SUCCESS;
        }
        else
        {
            printf("[%s] %s :: id=%zu, type=%i, num pc sample configs=%zu\n",
                   __FUNCTION__,
                   agents[i]->name,
                   agents[i]->id.handle,
                   agents[i]->type,
                   agents[i]->pc_sampling_configs.size);
        }
    }

    return !_out_agents->empty() ? ROCPROFILER_STATUS_SUCCESS : ROCPROFILER_STATUS_ERROR;
}

void
find_all_gpu_agents_supporting_pc_sampling()
{
    // This function returns the all gpu agents supporting some kind of PC sampling
    ROCPROFILER_CALL(
        rocprofiler_query_available_agents(&find_all_gpu_agents_supporting_pc_sampling_impl,
                                           sizeof(rocprofiler_agent_t),
                                           static_cast<void*>(&gpu_agents)),
        "Failed to find GPU agents");
}
}  // namespace

void
configure_host_trap_sampling(rocprofiler_context_id_t context_id,
                             rocprofiler_buffer_id_t  buffer_id,
                             rocprofiler_agent_t      gpu_agent)
{
    // Vladimir: Does MI200 have only one configuration?
    assert(gpu_agent.pc_sampling_configs.size == 1);

    // Extract the configuration
    auto host_trap_config = gpu_agent.pc_sampling_configs.data[0];

    // The mean of min_interval and max_interval
    auto interval = (host_trap_config.min_interval + host_trap_config.max_interval) / 2;

    ROCPROFILER_CALL(rocprofiler_configure_pc_sampling_service(context_id,
                                                               gpu_agent,
                                                               host_trap_config.method,
                                                               host_trap_config.unit,
                                                               interval,
                                                               buffer_id),
                     "Cannot create host-trap PC sampling service");
}

rocprofiler_pc_sampling_configuration_t
extract_stochastic_config(rocprofiler_pc_sampling_config_array_t* configs)
{
    // Iterate over an array of configurations and return the first one
    // with stochasting method.
    for(size_t i = 0; i < configs->size; i++)
    {
        if(configs->data[i].method == ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC)
        {
            return configs->data[i];
        }
    }
    printf("Improper use of the `extract_stochastic_config` function.");
    exit(-1);
}

void
configure_stochastic_sampling(rocprofiler_context_id_t context_id,
                              rocprofiler_buffer_id_t  buffer_id,
                              rocprofiler_agent_t      gpu_agent)
{
    // Find the configuration matching stochastic sampling in cycles
    rocprofiler_pc_sampling_configuration_t stochastic_config =
        extract_stochastic_config(&gpu_agent.pc_sampling_configs);

    // The mean of min_interval and max_interval
    auto interval = (stochastic_config.min_interval + stochastic_config.max_interval) / 2;

    ROCPROFILER_CALL(rocprofiler_configure_pc_sampling_service(context_id,
                                                               gpu_agent,
                                                               stochastic_config.method,
                                                               stochastic_config.unit,
                                                               interval,
                                                               buffer_id),
                     "Cannot create stochastic PC sampling service");
}

int
main(int /*argc*/, char** /*argv*/)
{
    if(!find_first_gpu_agent())
    {
        fprintf(stderr, "no gpu agents were found\n");
        return EXIT_FAILURE;
    }

    find_all_gpu_agents_supporting_pc_sampling();

    if(gpu_agents.empty())
    {
        printf("No availabe gpu agents\n");
        exit(-1);
    }

    // Vladimir: The relations I assumed:
    // - a context per gpu agent
    // - a buffer per context
    // - a pc sampling service per buffer
    // How about the following: Single context with mulitple buffers and PC sampling services?
    // When starting the context, does it start all PC sampling services at once?

    for(auto gpu_agent : gpu_agents)
    {
        // creating a context
        rocprofiler_context_id_t context_id;
        ROCPROFILER_CALL(rocprofiler_create_context(&context_id), "Cannot create context\n");
        contexts.push_back(context_id);

        // creating a buffer that will hold pc sampling information
        rocprofiler_buffer_policy_t drop_buffer_action = ROCPROFILER_BUFFER_POLICY_DISCARD;
        rocprofiler_buffer_id_t     buffer_id;
        ROCPROFILER_CALL(rocprofiler_create_buffer(context_id,
                                                   BUFFER_SIZE_BYTES,
                                                   WATERMARK,
                                                   drop_buffer_action,
                                                   rocprofiler_pc_sampling_callback,
                                                   nullptr,
                                                   &buffer_id),
                         "Cannot create pc sampling buffer");
        buffer_ids.push_back(buffer_id);

        if(gpu_agent.name == MI200_NAME)
            configure_host_trap_sampling(context_id, buffer_id, gpu_agent);
        else
            configure_stochastic_sampling(context_id, buffer_id, gpu_agent);

        // Starting the context that should trigger PC sampling
        ROCPROFILER_CALL(rocprofiler_start_context(context_id), "Cannot start PC sampling context");
    }

    // Running the applicaiton
    run_HIP_app();

    for(size_t i = 0; i < gpu_agents.size(); i++)
    {
        // Stop the context that should stop PC sampling?
        ROCPROFILER_CALL(rocprofiler_stop_context(contexts[i]), "Cannot start PC sampling context");
        // Explicit buffer flush, before destroying it
        ROCPROFILER_CALL(rocprofiler_flush_buffer(buffer_ids[i]), "Cannot destroy buffer");
        // Destroying the buffer
        ROCPROFILER_CALL(rocprofiler_destroy_buffer(buffer_ids[i]), "Cannot destroy buffer");
    }

    return 0;
}
