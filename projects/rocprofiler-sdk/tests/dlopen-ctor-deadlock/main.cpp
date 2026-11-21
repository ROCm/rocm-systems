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

// The plugin's constructor runs under dlopen, which holds glibc's dl_load_lock, and calls into
// HIP. If the profiler takes dl_load_lock (e.g. by registering a thread_local destructor) from
// inside a runtime path that HIP call needs, it deadlocks and the watchdog prints "DEADLOCK".
// Each mode puts the profiler inside a different runtime path at that moment:
//
//   code-object-load  A worker lazily loads this executable's code object: HIP holds its
//                     code-object lock across hsa_executable_freeze, which the profiler wraps.
//                     The plugin's own fat-binary registration needs that lock.
//   kernel-dispatch   A worker's first kernel launch: the profiler's queue interception runs
//                     under HIP's per-device execution lock. The constructor's hipMemset on the
//                     same device needs it.
//   scratch           The constructor launches a kernel that needs more scratch than is
//                     allocated. ROCr grants it on its own events thread, which calls the
//                     profiler's scratch-memory callback first.
//
// Like PyTorch's Kineto at `import torch`, the app is its own tool and starts only code-object
// tracing; the other modes run under rocprofv3 for the services they need.

#include <hip/hip_runtime.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <dlfcn.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

#define HIP_CHECK(...)                                                                             \
    do                                                                                             \
    {                                                                                              \
        hipError_t _err = (__VA_ARGS__);                                                           \
        if(_err != hipSuccess)                                                                     \
        {                                                                                          \
            fprintf(stderr, "%s failed: %s\n", #__VA_ARGS__, hipGetErrorString(_err));             \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while(0)

#define ROCPROFILER_CHECK(...)                                                                     \
    do                                                                                             \
    {                                                                                              \
        if((__VA_ARGS__) != ROCPROFILER_STATUS_SUCCESS)                                            \
        {                                                                                          \
            fprintf(stderr, "%s failed\n", #__VA_ARGS__);                                          \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while(0)

namespace
{
std::string       mode         = {};
float*            device_data  = nullptr;
std::atomic<bool> ctor_entered = {false};

__device__ int lazy_symbol[64];

__global__ void
add_one(float* data)
{
    data[threadIdx.x] += 1.0f;
}

__global__ void
use_scratch(float* data, int idx)
{
    volatile float buf[4096];
    for(int i = 0; i < 4096; ++i)
        buf[i] = data[threadIdx.x] + i;
    data[threadIdx.x] = buf[idx % 4096];
}

// HIP sets up its own per-thread state with a thread_local destructor, so a worker whose first
// HIP call races the constructor would block on dl_load_lock inside HIP, before the profiler.
void
initialize_hip_thread_state()
{
    int dev = -1;
    HIP_CHECK(hipGetDevice(&dev));
    HIP_CHECK(__hipPushCallConfiguration(dim3(1), dim3(1), 0, nullptr));
    auto        grid   = dim3{};
    auto        block  = dim3{};
    size_t      shmem  = 0;
    hipStream_t stream = nullptr;
    HIP_CHECK(__hipPopCallConfiguration(&grid, &block, &shmem, &stream));
    (void) hipStreamQuery(nullptr);
    (void) hipGetLastError();
}

int
tool_init(rocprofiler_client_finalize_t, void*)
{
    auto ctx = rocprofiler_context_id_t{};
    ROCPROFILER_CHECK(rocprofiler_create_context(&ctx));
    ROCPROFILER_CHECK(rocprofiler_configure_callback_tracing_service(
        ctx,
        ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
        nullptr,
        0,
        [](rocprofiler_callback_tracing_record_t, rocprofiler_user_data_t*, void*) {},
        nullptr));
    ROCPROFILER_CHECK(rocprofiler_start_context(ctx));
    return 0;
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t* id)
{
    id->name = "dlopen-ctor-deadlock";

    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, nullptr, nullptr};
    return &cfg;
}

// called by the plugin's constructor; gives the worker time to get inside its HIP call
extern "C" __attribute__((visibility("default"))) void
dlopen_ctor_deadlock_hook()
{
    ctor_entered.store(true);
    std::this_thread::sleep_for(std::chrono::milliseconds{500});

    if(mode == "kernel-dispatch")
    {
        HIP_CHECK(hipMemset(device_data, 0, 256));
        HIP_CHECK(hipDeviceSynchronize());
    }
    else if(mode == "scratch")
    {
        use_scratch<<<256, 256>>>(device_data, 3);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
    }
}

int
main(int argc, char** argv)
{
    if(argc != 3)
    {
        fprintf(stderr,
                "usage: %s <code-object-load|kernel-dispatch|scratch> <ctor-plugin>\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    mode = argv[1];
    if(mode != "code-object-load" && mode != "kernel-dispatch" && mode != "scratch")
    {
        fprintf(stderr, "unknown mode: %s\n", mode.c_str());
        return EXIT_FAILURE;
    }

    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipMalloc(&device_data, 1 << 20));
    HIP_CHECK(hipMemset(device_data, 0, 1 << 20));
    if(mode != "code-object-load")
    {
        // load this executable's code object now, so the only code-object load still to come
        // is the plugin's
        add_one<<<1, 64>>>(device_data);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
    }

    auto worker       = std::thread{};
    auto worker_ready = std::atomic<bool>{false};
    if(mode != "scratch")
    {
        worker = std::thread{[&worker_ready]() {
            initialize_hip_thread_state();
            worker_ready.store(true);

            while(!ctor_entered.load())
                std::this_thread::yield();

            if(mode == "code-object-load")
            {
                // first use of this executable's code object loads it
                int host[64] = {};
                HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(lazy_symbol), host, sizeof(host)));
            }
            else
            {
                add_one<<<1, 64>>>(device_data);
                HIP_CHECK(hipGetLastError());
            }
            HIP_CHECK(hipDeviceSynchronize());
        }};

        while(!worker_ready.load())
            std::this_thread::yield();
    }

    std::thread{[]() {
        std::this_thread::sleep_for(std::chrono::seconds{30});
        printf("DEADLOCK: no progress after 30 seconds in mode %s\n", mode.c_str());
        fflush(stdout);
        std::_Exit(EXIT_FAILURE);
    }}.detach();

    if(dlopen(argv[2], RTLD_NOW | RTLD_LOCAL) == nullptr)
    {
        fprintf(stderr, "dlopen(%s) failed: %s\n", argv[2], dlerror());
        return EXIT_FAILURE;
    }

    if(worker.joinable()) worker.join();

    printf("PASSED: %s\n", mode.c_str());
    fflush(stdout);
    return EXIT_SUCCESS;
}
