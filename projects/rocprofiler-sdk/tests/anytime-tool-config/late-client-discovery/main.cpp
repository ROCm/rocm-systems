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

/**
 * @file tests/anytime-tool-config/late-client-discovery/main.cpp
 *
 * @brief Test app for a tool library which becomes visible after rocprofiler-sdk loaded.
 *
 * The app links HIP but makes no HIP call until after it has dlopen'd the tool. That gap
 * is what the test needs: HIP registers its compiler API table while the process is being
 * loaded, and its runtime and HSA tables only on the first HIP call. With
 * rocprofiler-register told to load rocprofiler-sdk without waiting for a client, the
 * library is therefore brought in, finds no client and closes its configuration window
 * before main() runs -- then the tool arrives, and two more API table registrations follow.
 *
 * Nothing here calls rocprofiler_force_configure(). A tool which gates that call on
 * rocprofiler_is_initialized() reporting "not yet" never makes it, which is how a process
 * ends up with the SDK loaded, a tool loaded, and no profiling at all. Recovering from
 * that is rocprofiler-sdk's job and is what this exercises.
 *
 * Two variants run the same binary: one where the SDK is preloaded with no client (the
 * scenario) and one where it is not resident at all (the ordinary path, which must keep
 * working). The expected residency is an argument so a variant whose environment stops
 * producing its scenario fails instead of passing for the wrong reason.
 */

#include <hip/hip_runtime.h>

#include <dlfcn.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#define HIP_CALL(...)                                                                              \
    {                                                                                              \
        auto _status = (__VA_ARGS__);                                                              \
        if(_status != hipSuccess)                                                                  \
        {                                                                                          \
            std::cerr << "ERROR: " << #__VA_ARGS__ << " failed: " << hipGetErrorString(_status)    \
                      << "\n";                                                                     \
            return EXIT_FAILURE;                                                                   \
        }                                                                                          \
    }

namespace
{
constexpr int num_launches      = 8;
constexpr int threads_per_block = 64;

// Kernel completion and the dispatch record reaching the tool are different events, and
// there is no public call which waits for the second, so the count has to be polled.
constexpr auto poll_interval = std::chrono::milliseconds{10};
constexpr auto poll_deadline = std::chrono::seconds{10};

using dispatch_count_func_t = uint64_t (*)();
using is_initialized_func_t = int (*)(int*);

bool
symbol_visible(const char* name)
{
    return (dlsym(RTLD_DEFAULT, name) != nullptr);
}
}  // namespace

__global__ void
late_client_discovery_kernel(float* data, float value)
{
    data[threadIdx.x] = value;
}

int
main(int argc, char** argv)
{
    if(argc < 3)
    {
        std::cerr << "ERROR: usage: " << argv[0] << " <tool-library> <expect-sdk-resident 0|1>\n";
        return EXIT_FAILURE;
    }

    const auto* tool_library    = argv[1];
    const bool  expect_resident = (std::string{argv[2]} == "1");

    // Whether the library is already here is set by the environment, so check it rather
    // than assume it: a variant which stops producing its scenario must fail, not pass.
    const bool sdk_resident = symbol_visible("rocprofiler_force_configure");
    std::cout << "PREMISE sdk_resident=" << sdk_resident << " expected=" << expect_resident << "\n";

    if(sdk_resident != expect_resident)
    {
        std::cerr << "ERROR: rocprofiler-sdk residency before the tool loads is not what this "
                     "variant tests, so the scenario is no longer being exercised\n";
        return EXIT_FAILURE;
    }

    // Nothing may define rocprofiler_configure yet. If something does, client discovery
    // succeeds the ordinary way and the test passes without exercising anything.
    if(symbol_visible("rocprofiler_configure"))
    {
        std::cerr << "ERROR: rocprofiler_configure is resolvable before the tool is loaded. Find "
                     "what pulled a tool into the process rather than relaxing this check\n";
        return EXIT_FAILURE;
    }

    if(sdk_resident)
    {
        auto* is_initialized = reinterpret_cast<is_initialized_func_t>(
            dlsym(RTLD_DEFAULT, "rocprofiler_is_initialized"));
        int init_status = 0;

        if(is_initialized == nullptr || is_initialized(&init_status) != 0 || init_status != 1)
        {
            std::cerr << "ERROR: expected rocprofiler-sdk to have finished initializing with no "
                         "client before main(), init status is "
                      << init_status << "\n";
            return EXIT_FAILURE;
        }

        std::cout << "PREMISE init_status=" << init_status
                  << " (configuration window already closed)\n";
    }

    void* tool_handle = dlopen(tool_library, RTLD_NOW | RTLD_GLOBAL);
    if(tool_handle == nullptr)
    {
        std::cerr << "ERROR: failed to load " << tool_library << ": " << dlerror() << "\n";
        return EXIT_FAILURE;
    }

    if(!symbol_visible("rocprofiler_configure"))
    {
        std::cerr << "ERROR: " << tool_library
                  << " loaded but does not make rocprofiler_configure globally visible\n";
        return EXIT_FAILURE;
    }

    std::cout << "Tool loaded; rocprofiler_configure is now visible\n" << std::flush;

    // First HIP call of the process, and the point the remaining API tables register.
    float* data = nullptr;
    HIP_CALL(hipMalloc(&data, threads_per_block * sizeof(float)));

    for(int i = 0; i < num_launches; ++i)
        late_client_discovery_kernel<<<1, threads_per_block>>>(data, static_cast<float>(i + 1));

    HIP_CALL(hipDeviceSynchronize());

    // Establish that the kernels reached the GPU before reading the tool's count, because
    // a zero means something entirely different if no kernel ever ran.
    auto host = std::vector<float>(threads_per_block, 0.0F);
    HIP_CALL(
        hipMemcpy(host.data(), data, threads_per_block * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CALL(hipFree(data));

    if(host.front() != static_cast<float>(num_launches))
    {
        std::cerr << "ERROR: the workload did not run: the device holds " << host.front()
                  << " where the last of " << num_launches << " kernels should have written "
                  << num_launches << "\n";
        return EXIT_FAILURE;
    }

    auto* dispatch_count = reinterpret_cast<dispatch_count_func_t>(
        dlsym(tool_handle, "late_client_discovery_dispatch_count"));
    if(dispatch_count == nullptr)
    {
        std::cerr << "ERROR: " << tool_library << " does not export "
                  << "late_client_discovery_dispatch_count\n";
        return EXIT_FAILURE;
    }

    uint64_t   observed   = 0;
    const auto poll_start = std::chrono::steady_clock::now();
    while(std::chrono::steady_clock::now() - poll_start < poll_deadline)
    {
        observed = dispatch_count();
        if(observed >= static_cast<uint64_t>(num_launches)) break;
        std::this_thread::sleep_for(poll_interval);
    }

    std::cout << "RESULT dispatches=" << observed << " expected>=" << num_launches << "\n";

    // Not an equality: a device-to-host copy may itself be serviced by a blit kernel, and
    // how many dispatches that adds is not this test's subject. Zero against a workload
    // already proven to have run is, and that is what a tool nobody configured reports.
    if(observed < static_cast<uint64_t>(num_launches))
    {
        std::cerr << "ERROR: the tool recorded " << observed << " kernel dispatches, fewer than "
                  << "the " << num_launches << " launched. Zero means it was never configured\n";
        return EXIT_FAILURE;
    }

    std::cout << "Test PASSED: a tool loaded after rocprofiler-sdk recorded every kernel "
                 "dispatch\n";
    return EXIT_SUCCESS;
}
