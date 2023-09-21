// Vladimir: The example of using Host-trap PC sampling on a system with single MI200/300 by two
// users. The first user initiates Host-Trap sampling with the configuration A. The second user
// tries initiaiting stochastic sampling with configuration B and fails. Then it queries available
// configurations and observes only the configuration A. It accepts it and starts PC sampling.
// Vladimir: Currently, this example is written as a single-threaded program.
// Decide whether to move the second user to a separate thread or process

#include <rocprofiler/rocprofiler.h>

#include "common.h"

#include <cassert>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#define HOST_TRAP_INTERVAL 1000

rocprofiler_pc_sampling_method_t host_trap_sampling_method;
rocprofiler_pc_sampling_unit_t   host_trap_sampling_unit_time;
uint64_t                         host_trap_interval;

void
second_user()
{
    // creating a context
    rocprofiler_context_id_t context_id2;
    ROCPROFILER_CALL(rocprofiler_create_context(&context_id2),
                     "Cannot create context for the second user\n");

    auto gpu_agent = find_first_gpu_agent();
    if(!gpu_agent) throw std::runtime_error{"no gpu agents were found"};

    // creating a buffer that will hold pc sampling information
    rocprofiler_buffer_policy_t lossless_buffer_action = ROCPROFILER_BUFFER_POLICY_LOSSLESS;
    rocprofiler_buffer_id_t     buffer_id2;
    ROCPROFILER_CALL(rocprofiler_create_buffer(context_id2,
                                               BUFFER_SIZE_BYTES,
                                               WATERMARK,
                                               lossless_buffer_action,
                                               rocprofiler_pc_sampling_callback,
                                               nullptr,
                                               &buffer_id2),
                     "Cannot create pc sampling buffer for the second user");

    // The second user tries to create another pc sampling service with different configuration,
    // but the rocprofiler rejects it.
    rocprofiler_pc_sampling_method_t sampling_method2 = ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC;
    rocprofiler_pc_sampling_unit_t   sampling_unit2   = ROCPROFILER_PC_SAMPLING_UNIT_CYCLES;
    uint64_t interval2 = 2048;  // I assumed micro secs, so this should be 1ms
    // The following function returns an error code indicating the PC sampling has already been
    // configured.
    ROCPROFILER_CALL_FAILS(
        rocprofiler_configure_pc_sampling_service(
            context_id2, *gpu_agent, sampling_method2, sampling_unit2, interval2, buffer_id2),
        "Instantiation of the PC sampling service should fail");

    // After failure, the second user queries available configuration and observes the one chosen by
    // the first user.
    size_t                                               config_count = 10;
    std::vector<rocprofiler_pc_sampling_configuration_t> configs(config_count);
    ROCPROFILER_CALL(rocprofiler_query_pc_sampling_agent_configurations(
                         *gpu_agent, configs.data(), &config_count),
                     "The second user cannot query available configurations");

    // Only one configuration should be listed, and its parameters should match the parameters set
    // by the first user. Vladimir: Is it ok to use assertions? In the release mode, they might be
    // ignored.
    assert(config_count == 1);
    rocprofiler_pc_sampling_configuration_t first_user_config = configs[0];
    assert(first_user_config.method == host_trap_sampling_method);
    assert(first_user_config.unit == host_trap_sampling_unit_time);
    // Vladimir: Should the min_interval and max_interval have the same value at this point (the PC
    // sampling is alredy configured)??
    assert(first_user_config.min_interval == host_trap_interval &&
           first_user_config.min_interval == first_user_config.max_interval);

    // Reuse the same configuration set by the first user.
    // The second user is satisfied with the configuration chosen by the first user, so it
    // starts PC sampling.
    ROCPROFILER_CALL(rocprofiler_configure_pc_sampling_service(context_id2,
                                                               *gpu_agent,
                                                               first_user_config.method,
                                                               first_user_config.unit,
                                                               first_user_config.min_interval,
                                                               buffer_id2),
                     "The second user cannot share already created PC sampling configuration");

    // Starting the context that should trigger PC sampling?
    ROCPROFILER_CALL(rocprofiler_start_context(context_id2),
                     "Cannot start PC sampling context for the second user");

    // Running the applicaiton
    run_HIP_app();

    // Stop the context that should stop PC sampling?
    ROCPROFILER_CALL(rocprofiler_stop_context(context_id2),
                     "Cannot start PC sampling context for the second user");

    // Explicit buffer flush, before destroying it
    ROCPROFILER_CALL(rocprofiler_flush_buffer(buffer_id2),
                     "Cannot destroy the second user's buffer");
    // Destroying the buffer
    ROCPROFILER_CALL(rocprofiler_destroy_buffer(buffer_id2), "Cannot destroy the second user's");
}

int
main(int /*argc*/, char** /*argv*/)
{
    // creating a context
    rocprofiler_context_id_t context_id;
    ROCPROFILER_CALL(rocprofiler_create_context(&context_id), "Cannot create context\n");

    auto gpu_agent = find_first_gpu_agent();
    if(!gpu_agent)
    {
        fprintf(stderr, "no gpu agents were found\n");
        return EXIT_FAILURE;
    }

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

    // PC sampling service configuration
    host_trap_sampling_method    = ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP;
    host_trap_sampling_unit_time = ROCPROFILER_PC_SAMPLING_UNIT_TIME;
    // Vladimir: What units are we using for time? ms, micro secs, ns?
    host_trap_interval = HOST_TRAP_INTERVAL;
    // Instantiating the first PC sampling service succeeds.
    ROCPROFILER_CALL(rocprofiler_configure_pc_sampling_service(context_id,
                                                               *gpu_agent,
                                                               host_trap_sampling_method,
                                                               host_trap_sampling_unit_time,
                                                               host_trap_interval,
                                                               buffer_id),
                     "Cannot create PC sampling service");

    // Trigger the second user code.
    // Vladimir: Discuss whether this should be put in a separate thread/process.
    second_user();

    // Starting the context that should trigger PC sampling?
    ROCPROFILER_CALL(rocprofiler_start_context(context_id), "Cannot start PC sampling context");

    // Running the applicaiton
    run_HIP_app();

    // Stop the context that should stop PC sampling?
    ROCPROFILER_CALL(rocprofiler_stop_context(context_id), "Cannot start PC sampling context");

    // Explicit buffer flush, before destroying it
    ROCPROFILER_CALL(rocprofiler_flush_buffer(buffer_id), "Cannot destroy buffer");
    // Destroying the buffer
    ROCPROFILER_CALL(rocprofiler_destroy_buffer(buffer_id), "Cannot destroy buffer");

    // Vladimir: Do we need to destroy context or a service?

    return 0;
}
