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

// Vladimir: The example of using Host-trap PC sampling exclusively on the system with single MI200.
// If any of the rocprofiler calls returns status fail, we simply stop the application.

#include <rocprofiler/rocprofiler.h>
#include <cstdlib>
#include "common.h"

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
    rocprofiler_pc_sampling_method_t sampling_method = ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP;
    rocprofiler_pc_sampling_unit_t   sampling_unit   = ROCPROFILER_PC_SAMPLING_UNIT_TIME;
    // What units are we using for time? ms, micro secs, ns?
    uint64_t interval = 1000;  // I assumed micro secs, so this should be 1ms
    // Instantiating the PC sampling service
    ROCPROFILER_CALL(
        rocprofiler_configure_pc_sampling_service(
            context_id, *gpu_agent, sampling_method, sampling_unit, interval, buffer_id),
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
