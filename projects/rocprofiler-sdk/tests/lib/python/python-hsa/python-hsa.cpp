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

// Direct-HSA workload module for the anytime-tool-config tests. Exercises the HSA core
// dispatch table (the table rocprofiler interposes for HSA_API tracing) WITHOUT going
// through HIP, so an HSA-only variation has a genuine HSA-driven workload. It deliberately
// avoids kernel dispatch (which would need a precompiled .hsaco): HSA_API tracing records
// every HSA call, so agent/queue/signal API traffic is plenty.
//
// C ABI (loaded via ctypes):
//   hsa_workload_init()        -> hsa_init + cache the first GPU agent. 0 on success.
//   hsa_workload_run(int reps) -> per rep: query agent info, create+destroy a queue,
//                                 create a signal, store/load/wait, destroy it. 0/-1.
//   hsa_workload_fini()        -> hsa_shut_down. 0 on success.
//
// hsa_workload_run is callable repeatedly so a driver can drive HSA activity both before
// and after a late tool registers.

#include <hsa/hsa.h>

#include <cstdio>

#define HSA_WORKLOAD_PUBLIC_API __attribute__((visibility("default")))

#define HSA_CHECK(call)                                                                            \
    do                                                                                             \
    {                                                                                              \
        hsa_status_t err = (call);                                                                 \
        if(err != HSA_STATUS_SUCCESS)                                                              \
        {                                                                                          \
            const char* msg = nullptr;                                                             \
            hsa_status_string(err, &msg);                                                          \
            fprintf(stderr,                                                                        \
                    "[python-hsa] HSA error %d (%s) at %s:%d\n",                                   \
                    (int) err,                                                                     \
                    msg ? msg : "?",                                                               \
                    __FILE__,                                                                      \
                    __LINE__);                                                                     \
            return -1;                                                                             \
        }                                                                                          \
    } while(0)

namespace
{
hsa_agent_t gpu_agent  = {};
bool        have_agent = false;

hsa_status_t
find_gpu_agent(hsa_agent_t agent, void* /*data*/)
{
    hsa_device_type_t type = {};
    if(hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type) != HSA_STATUS_SUCCESS)
        return HSA_STATUS_SUCCESS;
    if(type == HSA_DEVICE_TYPE_GPU && !have_agent)
    {
        gpu_agent  = agent;
        have_agent = true;
    }
    return HSA_STATUS_SUCCESS;
}
}  // namespace

extern "C" {

/// hsa_init + locate the first GPU agent. Returns 0 on success, -1 on error.
HSA_WORKLOAD_PUBLIC_API int
hsa_workload_init()
{
    HSA_CHECK(hsa_init());
    HSA_CHECK(hsa_iterate_agents(find_gpu_agent, nullptr));
    if(!have_agent)
    {
        fprintf(stderr, "[python-hsa] no GPU agent found\n");
        return -1;
    }
    fprintf(stderr, "[python-hsa] initialized; GPU agent handle %lu\n", gpu_agent.handle);
    return 0;
}

/// Run `reps` iterations of HSA core API activity. Each iteration queries agent info,
/// creates and destroys a queue, and creates/uses/destroys a signal. All of these are
/// HSA core API calls captured by HSA_API tracing. Returns 0 on success, -1 on error.
HSA_WORKLOAD_PUBLIC_API int
hsa_workload_run(int reps)
{
    if(!have_agent)
    {
        fprintf(stderr, "[python-hsa] hsa_workload_run called before init\n");
        return -1;
    }

    for(int i = 0; i < reps; ++i)
    {
        char name[64] = {};
        HSA_CHECK(hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_NAME, name));

        uint32_t queue_size = 0;
        HSA_CHECK(hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size));
        if(queue_size > 1024) queue_size = 1024;

        hsa_queue_t* queue = nullptr;
        HSA_CHECK(hsa_queue_create(
            gpu_agent, queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0, 0, &queue));

        hsa_signal_t signal = {};
        HSA_CHECK(hsa_signal_create(1, 0, nullptr, &signal));
        hsa_signal_store_relaxed(signal, 0);
        (void) hsa_signal_load_scacquire(signal);
        HSA_CHECK(hsa_signal_destroy(signal));

        HSA_CHECK(hsa_queue_destroy(queue));
    }
    return 0;
}

/// hsa_shut_down. Returns 0 on success, -1 on error.
HSA_WORKLOAD_PUBLIC_API int
hsa_workload_fini()
{
    HSA_CHECK(hsa_shut_down());
    have_agent = false;
    fprintf(stderr, "[python-hsa] shut down\n");
    return 0;
}

}  // extern "C"
