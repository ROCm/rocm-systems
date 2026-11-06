// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

// Dispatches kernels through an AQL packet with an application-owned completion signal and reads
// hsa_amd_profiling_get_dispatch_time() on that signal after waiting, the way libomptarget does.
// Each signal is then reused for a barrier packet, and for a kernel and a barrier submitted in one
// queue write. Exits with failure if any dispatch time is unavailable, if the kernel's time is
// shorter than the kernel can run, or if a barrier reports the kernel's time.

#include "hsa_code_object_app.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace
{
// large enough that the kernel runs far longer than the minimum duration below
constexpr uint32_t num_elements    = 1u << 26;
constexpr uint64_t min_duration_ns = 10000;
constexpr int      iterations      = 8;
}  // namespace

int
main()
{
    hsa_status_t     status = HSA_STATUS_SUCCESS;
    MQDependencyTest obj;
    if(!obj.device_discovery() || obj.gpu.empty())
    {
        fprintf(stderr, "No GPU agent found; cannot run test.\n");
        return EXIT_FAILURE;
    }

    auto agent = obj.gpu[0].agent;
    char agent_name[64];
    status = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, agent_name);
    RET_IF_HSA_ERR(status)

    auto code_object = MQDependencyTest::CodeObject{};
    auto kernel      = MQDependencyTest::Kernel{};
    auto hsaco_path  = std::string{HSACO_DIR} + "/" + agent_name + "_copy.hsaco";
    if(!obj.load_code_object(hsaco_path, agent, code_object) ||
       !obj.get_kernel(code_object, "copyA", agent, kernel))
    {
        fprintf(stderr, "failed to load kernel copyA from %s\n", hsaco_path.c_str());
        return EXIT_FAILURE;
    }

    struct args_t
    {
        uint32_t*                       a      = nullptr;
        uint32_t*                       b      = nullptr;
        MQDependencyTest::OCLHiddenArgs hidden = {};
    };

    auto* args = static_cast<args_t*>(obj.hsa_malloc(sizeof(args_t), obj.kernarg));
    *args      = {};
    args->a =
        static_cast<uint32_t*>(obj.hsa_malloc(num_elements * sizeof(uint32_t), obj.gpu[0], false));
    args->b =
        static_cast<uint32_t*>(obj.hsa_malloc(num_elements * sizeof(uint32_t), obj.gpu[0], false));

    hsa_queue_t* queue = nullptr;
    status             = hsa_queue_create(
        agent, 1024, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, UINT32_MAX, UINT32_MAX, &queue);
    RET_IF_HSA_ERR(status)
    status = hsa_amd_profiling_set_profiler_enabled(queue, 1);
    RET_IF_HSA_ERR(status)

    auto wait_and_get_time = [&](hsa_signal_t signal, hsa_amd_profiling_dispatch_time_t& time) {
        while(hsa_signal_wait_scacquire(
                  signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED) >= 1)
        {}
        return hsa_amd_profiling_get_dispatch_time(agent, signal, &time);
    };

    // writes both packets before ringing the doorbell once, so they arrive in one queue write
    auto submit_together = [queue](MQDependencyTest::Aql& first, MQDependencyTest::Aql& second) {
        auto*    ring  = static_cast<MQDependencyTest::Aql*>(queue->base_address);
        uint64_t write = hsa_queue_load_write_index_relaxed(queue);
        uint64_t index = write;
        for(auto* pkt : {&first, &second})
        {
            auto&    dst    = ring[index++ & (queue->size - 1)];
            uint16_t header = pkt->header.raw;
            pkt->header.raw = dst.header.raw;
            dst             = *pkt;
            __atomic_store_n(&dst.header.raw, header, __ATOMIC_RELEASE);
            pkt->header.raw = header;
        }
        hsa_queue_store_write_index_release(queue, index);
        hsa_signal_store_screlease(queue->doorbell_signal, index - 1);
    };

    int failures = 0;
    for(int i = 0; i < iterations; ++i)
    {
        auto signal = hsa_signal_t{};
        status      = hsa_signal_create(1, 0, nullptr, &signal);
        RET_IF_HSA_ERR(status)

        MQDependencyTest::Aql packet{};
        packet.header.type                   = HSA_PACKET_TYPE_KERNEL_DISPATCH;
        packet.header.barrier                = 1;
        packet.header.acquire                = HSA_FENCE_SCOPE_SYSTEM;
        packet.header.release                = HSA_FENCE_SCOPE_SYSTEM;
        packet.dispatch.setup                = 1;
        packet.dispatch.workgroup_size_x     = 256;
        packet.dispatch.workgroup_size_y     = 1;
        packet.dispatch.workgroup_size_z     = 1;
        packet.dispatch.grid_size_x          = num_elements;
        packet.dispatch.grid_size_y          = 1;
        packet.dispatch.grid_size_z          = 1;
        packet.dispatch.group_segment_size   = kernel.group;
        packet.dispatch.private_segment_size = kernel.scratch;
        packet.dispatch.kernel_object        = kernel.handle;
        packet.dispatch.kernarg_address      = args;
        packet.dispatch.completion_signal    = signal;
        obj.submit_packet(queue, packet);

        auto kernel_time = hsa_amd_profiling_dispatch_time_t{};
        status           = wait_and_get_time(signal, kernel_time);
        if(status != HSA_STATUS_SUCCESS || kernel_time.end < kernel_time.start + min_duration_ns)
        {
            fprintf(stderr,
                    "dispatch %d: hsa_amd_profiling_get_dispatch_time returned %d (start=%lu "
                    "end=%lu)\n",
                    i,
                    static_cast<int>(status),
                    static_cast<unsigned long>(kernel_time.start),
                    static_cast<unsigned long>(kernel_time.end));
            ++failures;
        }

        // reusing the signal for a barrier must not report the kernel's time
        hsa_signal_store_screlease(signal, 1);
        MQDependencyTest::Aql barrier{};
        barrier.header.type                   = HSA_PACKET_TYPE_BARRIER_AND;
        barrier.header.barrier                = 1;
        barrier.header.acquire                = HSA_FENCE_SCOPE_SYSTEM;
        barrier.header.release                = HSA_FENCE_SCOPE_SYSTEM;
        barrier.barrier_and.completion_signal = signal;
        obj.submit_packet(queue, barrier);

        auto barrier_time = hsa_amd_profiling_dispatch_time_t{};
        status            = wait_and_get_time(signal, barrier_time);
        if(status != HSA_STATUS_SUCCESS ||
           (barrier_time.start == kernel_time.start && barrier_time.end == kernel_time.end))
        {
            fprintf(stderr,
                    "barrier %d: hsa_amd_profiling_get_dispatch_time returned %d (start=%lu "
                    "end=%lu, kernel start=%lu end=%lu)\n",
                    i,
                    static_cast<int>(status),
                    static_cast<unsigned long>(barrier_time.start),
                    static_cast<unsigned long>(barrier_time.end),
                    static_cast<unsigned long>(kernel_time.start),
                    static_cast<unsigned long>(kernel_time.end));
            ++failures;
        }

        // a kernel and a barrier sharing the signal in one write must report the barrier's time
        hsa_signal_store_screlease(signal, 2);
        submit_together(packet, barrier);

        auto shared_time = hsa_amd_profiling_dispatch_time_t{};
        status           = wait_and_get_time(signal, shared_time);
        if(status != HSA_STATUS_SUCCESS || shared_time.end >= shared_time.start + min_duration_ns)
        {
            fprintf(stderr,
                    "kernel and barrier %d: hsa_amd_profiling_get_dispatch_time returned %d "
                    "(start=%lu end=%lu)\n",
                    i,
                    static_cast<int>(status),
                    static_cast<unsigned long>(shared_time.start),
                    static_cast<unsigned long>(shared_time.end));
            ++failures;
        }

        status = hsa_signal_destroy(signal);
        RET_IF_HSA_ERR(status)
    }

    status = hsa_queue_destroy(queue);
    RET_IF_HSA_ERR(status)
    status = hsa_amd_memory_pool_free(args->a);
    RET_IF_HSA_ERR(status)
    status = hsa_amd_memory_pool_free(args->b);
    RET_IF_HSA_ERR(status)
    status = hsa_amd_memory_pool_free(args);
    RET_IF_HSA_ERR(status)
    status = hsa_executable_destroy(code_object.executable);
    RET_IF_HSA_ERR(status)

    return (failures == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
