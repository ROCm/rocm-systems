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
#include <fstream>
#include <string>

namespace
{
// AMD interface 1.39 replaced the per-agent boolean HSA_AMD_AGENT_INFO_SIGNAL_HINTS_SUPPORTED
// (added in 1.36, never released) with the HSA_AMD_AQL_PROPERTY_SIGNAL_HINTS bit of
// HSA_AMD_AGENT_INFO_AQL_EXTENSIONS. These are enumerators and inline functions, not macros, so
// the interface version decides which query a header set provides; older headers provide neither
// and the test then runs without hinted handles.
#if !defined(HSA_AMD_INTERFACE_VERSION_MINOR)
#    define AMD_SIGNAL_HINTS_QUERY_AVAILABLE 0
#elif HSA_AMD_INTERFACE_VERSION_MINOR >= 39
#    define AMD_SIGNAL_HINTS_QUERY_AVAILABLE 1
bool
signal_hints_supported(hsa_agent_t agent)
{
    uint8_t extensions[8] = {};
    return hsa_agent_get_info(agent,
                              static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_AQL_EXTENSIONS),
                              extensions) == HSA_STATUS_SUCCESS &&
           hsa_flag_isset64(extensions, HSA_AMD_AQL_PROPERTY_SIGNAL_HINTS);
}
#elif HSA_AMD_INTERFACE_VERSION_MINOR >= 36
#    define AMD_SIGNAL_HINTS_QUERY_AVAILABLE 1
bool
signal_hints_supported(hsa_agent_t agent)
{
    auto supported = false;
    return hsa_agent_get_info(
               agent,
               static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_SIGNAL_HINTS_SUPPORTED),
               &supported) == HSA_STATUS_SUCCESS &&
           supported;
}
#else
#    define AMD_SIGNAL_HINTS_QUERY_AVAILABLE 0
#endif

// large enough that the kernel runs far longer than the minimum duration below
constexpr uint32_t num_elements    = 1u << 26;
constexpr uint64_t min_duration_ns = 10000;
constexpr int      iterations      = 8;
// Allow scheduling and runtime teardown overhead in addition to the shared drain budget.
constexpr auto max_shutdown_duration = std::chrono::seconds{45};
}  // namespace

int
main(int argc, char** argv)
{
    auto application_times = std::ofstream{};
    if(argc > 1)
    {
        application_times.open(argv[1]);
        if(!application_times) return EXIT_FAILURE;
        application_times << "Start,End\n";
    }

    hsa_status_t     status = HSA_STATUS_SUCCESS;
    MQDependencyTest obj;
    if(!obj.device_discovery() || obj.gpu.empty())
    {
        fprintf(stderr, "No GPU agent found; cannot run test.\n");
        return EXIT_FAILURE;
    }

    constexpr auto ns_per_second = uint64_t{1'000'000'000};
    auto           timestamp_hz  = uint64_t{0};
    status = hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &timestamp_hz);
    RET_IF_HSA_ERR(status)
    if(timestamp_hz == 0 || timestamp_hz > ns_per_second) return EXIT_FAILURE;
    // The trace fixture scales timestamps by the integer nanoseconds per tick.
    const auto ns_per_tick = ns_per_second / timestamp_hz;
    const auto min_duration_ticks =
        (min_duration_ns * timestamp_hz + ns_per_second - 1) / ns_per_second;

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
    // The packet layout is identical for every iteration, including the one left blocked at exit.
    auto packet = MQDependencyTest::Aql{};
    for(int i = 0; i < iterations; ++i)
    {
        auto signal = hsa_signal_t{};
        status      = hsa_signal_create(1, 0, nullptr, &signal);
        RET_IF_HSA_ERR(status)

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
#if AMD_SIGNAL_HINTS_QUERY_AVAILABLE
        if(signal_hints_supported(agent))
            packet.dispatch.completion_signal =
                hsa_amd_signal_set_hints(signal, AMD_SIGNAL_HINT_TS_ON);
#endif
        obj.submit_packet(queue, packet);

        auto kernel_time = hsa_amd_profiling_dispatch_time_t{};
        status           = wait_and_get_time(signal, kernel_time);
        if(status != HSA_STATUS_SUCCESS || kernel_time.end < kernel_time.start + min_duration_ticks)
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

        if(application_times.is_open())
            application_times << kernel_time.start * ns_per_tick << ','
                              << kernel_time.end * ns_per_tick << '\n';

        auto invalid_agent_time = hsa_amd_profiling_dispatch_time_t{};
        status = hsa_amd_profiling_get_dispatch_time(obj.cpu[0].agent, signal, &invalid_agent_time);
        if(status != HSA_STATUS_ERROR_INVALID_AGENT)
        {
            fprintf(
                stderr, "dispatch %d: CPU agent query returned %d\n", i, static_cast<int>(status));
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
        if(status != HSA_STATUS_SUCCESS ||
           shared_time.end >= shared_time.start + min_duration_ticks)
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

    // Final HSA shutdown retires the live queue after its completion handlers drain. With
    // ROCPROFILER_DISPATCH_TIME_BLOCKED set, the queue is instead left holding a kernel and an
    // intercepted copy behind a dependency that is never satisfied, so shutdown must abandon them
    // rather than wait for work the application never releases. The buffers those submissions name
    // stay allocated: the copy is never executed, and freeing them would not make it completable.
    if(std::getenv("ROCPROFILER_DISPATCH_TIME_BLOCKED") != nullptr)
    {
        auto dependency     = hsa_signal_t{};
        auto blocked_kernel = hsa_signal_t{};
        auto blocked_copy   = hsa_signal_t{};
        status              = hsa_signal_create(1, 0, nullptr, &dependency);
        RET_IF_HSA_ERR(status)
        status = hsa_signal_create(2, 0, nullptr, &blocked_kernel);
        RET_IF_HSA_ERR(status)
        status = hsa_signal_create(1, 0, nullptr, &blocked_copy);
        RET_IF_HSA_ERR(status)

        auto blocked_args = static_cast<args_t*>(obj.hsa_malloc(sizeof(args_t), obj.kernarg));
        *blocked_args     = {};
        blocked_args->a   = static_cast<uint32_t*>(
            obj.hsa_malloc(num_elements * sizeof(uint32_t), obj.gpu[0], false));
        blocked_args->b = static_cast<uint32_t*>(
            obj.hsa_malloc(num_elements * sizeof(uint32_t), obj.gpu[0], false));

        auto gate                      = MQDependencyTest::Aql{};
        gate.header.type               = HSA_PACKET_TYPE_BARRIER_AND;
        gate.header.barrier            = 1;
        gate.header.acquire            = HSA_FENCE_SCOPE_SYSTEM;
        gate.header.release            = HSA_FENCE_SCOPE_SYSTEM;
        gate.barrier_and.dep_signal[0] = dependency;

        auto blocked_packet                       = packet;
        blocked_packet.dispatch.kernarg_address   = blocked_args;
        blocked_packet.dispatch.completion_signal = blocked_kernel;
        submit_together(gate, blocked_packet);

        // Intercepted only when memory-copy tracing is active for this process.
        status = hsa_amd_memory_async_copy(blocked_args->a,
                                           agent,
                                           blocked_args->b,
                                           agent,
                                           num_elements * sizeof(uint32_t),
                                           1,
                                           &dependency,
                                           blocked_copy);
        RET_IF_HSA_ERR(status)
    }

    else
    {
        auto copy_done = hsa_signal_t{};
        status         = hsa_signal_create(1, 0, nullptr, &copy_done);
        RET_IF_HSA_ERR(status)
        status =
            hsa_amd_memory_async_copy(args->a, agent, args->b, agent, 4096, 0, nullptr, copy_done);
        RET_IF_HSA_ERR(status)
        while(hsa_signal_wait_scacquire(
                  copy_done, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED) >= 1)
        {}
        status = hsa_signal_destroy(copy_done);
        RET_IF_HSA_ERR(status)
    }

    status = hsa_amd_memory_pool_free(args->a);
    RET_IF_HSA_ERR(status)
    status = hsa_amd_memory_pool_free(args->b);
    RET_IF_HSA_ERR(status)
    status = hsa_amd_memory_pool_free(args);
    RET_IF_HSA_ERR(status)
    // A blocked submission still names this executable's kernel, so the executable is not destroyed
    // here: its lifetime ends with the runtime shutdown below.

    // Final shutdown must return even though the submissions above can never complete. The
    // destructor does not report the status, so shut the runtime down here and check it, then
    // keep the object from repeating the call.
    const auto shutdown_begin = std::chrono::steady_clock::now();
    status                    = hsa_shut_down();
    obj.shutdown_done         = true;
    if(status != HSA_STATUS_SUCCESS)
    {
        fprintf(stderr, "final hsa_shut_down returned %d\n", static_cast<int>(status));
        ++failures;
    }
    else if(std::chrono::steady_clock::now() - shutdown_begin > max_shutdown_duration)
    {
        fprintf(stderr, "final hsa_shut_down exceeded its drain budget\n");
        ++failures;
    }

    return (failures == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
