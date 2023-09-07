// Vladimir: The example of using Host-trap PC sampling exclusively on the system with single MI200.
// If any of the rocprofiler calls returns status fail, we simply stop the application.

#include <rocprofiler/rocprofiler.h>
#include "common.h"

int
main(int /*argc*/, char** /*argv*/)
{
    // creating a context
    rocprofiler_context_id_t context_id;
    ROCPROFILER_CALL(rocprofiler_create_context(&context_id), "Cannot create context\n");

    rocprofiler_agent_t gpu_agent = find_first_gpu_agent();

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
    rocprofiler_pc_sampling_method_t sampling_method = ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP;
    rocprofiler_pc_sampling_unit_t   sampling_unit   = ROCPROFILER_PC_SAMPLING_UNIT_TIME;
    // What units are we using for time? ms, micro secs, ns?
    uint64_t interval = 1000;  // I assumed micro secs, so this should be 1ms
    // Instantiating the PC sampling service
    ROCPROFILER_CALL(
        rocprofiler_configure_pc_sampling_service(
            context_id, gpu_agent, sampling_method, sampling_unit, interval, buffer_id),
        "Cannot create PC sampling service");

    // Vladimir: Is this the place of retrying if someone already created the
    // configuration and the previous call fails?

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
