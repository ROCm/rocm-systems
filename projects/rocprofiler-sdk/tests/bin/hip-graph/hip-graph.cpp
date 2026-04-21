/*
Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <libgen.h>
#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>

// hip header file
#include <hip/hip_runtime.h>

#include <stdio.h>
#include <unistd.h>
#include <regex>
#include <string>
#include <thread>
#include <vector>

namespace
{
using auto_lock_t    = std::unique_lock<std::mutex>;
auto print_mutex     = std::mutex{};
auto global_kern_num = std::atomic<uint64_t>{0};
}  // namespace

template <typename T>
void
check(T result, char const* const func, const char* const file, int const line)
{
    if(result)
    {
        fprintf(stderr,
                "Hip error at %s:%d code=%d(%s) \"%s\" \n",
                file,
                line,
                static_cast<unsigned int>(result),
                hipGetErrorName(result),
                func);
        exit(EXIT_FAILURE);
    }
}
#define checkHipErrors(val) check((val), #val, __FILE__, __LINE__)

__global__ void
kernel_foo(const int devid, const int kernid, const int kernid_global, const volatile int* streamid)
{
    printf("[hip-graph][device %2i][stream %2i] Kernel foo | %2i | %2i executing...\n",
           devid,
           *streamid,
           kernid,
           kernid_global);
}

__global__ void
kernel_bar(const int devid, const int kernid, const int kernid_global, const volatile int* streamid)
{
    printf("[hip-graph][device %2i][stream %2i] Kernel bar | %2i | %2i executing...\n",
           devid,
           *streamid,
           kernid,
           kernid_global);
}

void
run(uint64_t                        devid,
    uint64_t                        nstream,
    uint64_t                        nkernel_per_stream,
    std::atomic<uint64_t>*          progress,
    const std::shared_future<void>& future)
{
    auto prefix = [devid]() {
        auto ss = std::stringstream{};
        ss << "[hip-graph][device " << std::setw(2) << devid << "] ";
        return ss.str();
    }();

    auto log_message = [&prefix](const auto& msg) {
        auto _lk = auto_lock_t{print_mutex};
        std::cout << prefix << msg << "..." << std::endl;
    };

    log_message("setting device");
    checkHipErrors(hipSetDevice(devid));

    auto streams    = std::vector<hipStream_t>(nstream, nullptr);
    auto stream_num = std::vector<int*>(nstream, nullptr);

    log_message("creating streams");
    for(auto& itr : streams)
        checkHipErrors(hipStreamCreate(&itr));

    log_message("allocating data");
    for(uint64_t i = 0; i < nstream; ++i)
    {
        auto& itr = stream_num.at(i);
        auto* str = streams.at(i);
        auto  val = i;
        checkHipErrors(hipMallocAsync(&itr, sizeof(int), str));
        checkHipErrors(hipMemcpyAsync(itr, &val, sizeof(int), hipMemcpyHostToDevice, str));
    }

    auto graphs = std::vector<hipGraph_t>(nstream);
    auto execs  = std::vector<hipGraphExec_t>(nstream, nullptr);

    uint64_t kern_num = 0;
    for(uint64_t i = 0; i < nstream; ++i)
    {
        log_message("capture begin (stream=" + std::to_string(i) + ")");
        checkHipErrors(hipStreamBeginCapture(streams.at(i), hipStreamCaptureModeGlobal));

        for(uint64_t j = 0; j < nkernel_per_stream; ++j)
        {
            auto kern_num_v      = kern_num++;
            auto glob_kern_num_v = global_kern_num++;
            auto kernel          = (j % 2 == 0) ? kernel_foo : kernel_bar;
            if(j == 0 || j + 1 == nkernel_per_stream)
            {
                log_message("capture enqueue (stream=" + std::to_string(i) +
                            ", kernel_idx=" + std::to_string(j) +
                            ", kernel_id=" + std::to_string(kern_num_v) +
                            ", kernel_global=" + std::to_string(glob_kern_num_v) + ")");
            }
            hipLaunchKernelGGL(kernel,
                               dim3(1),
                               dim3(1),
                               0,
                               streams.at(i),
                               devid,
                               kern_num_v,
                               glob_kern_num_v,
                               stream_num.at(i));
            checkHipErrors(hipGetLastError());
        }

        log_message("capture end (stream=" + std::to_string(i) + ")");
        checkHipErrors(hipStreamEndCapture(streams.at(i), &graphs.at(i)));
        log_message("graph instantiate begin (stream=" + std::to_string(i) + ")");
        checkHipErrors(hipGraphInstantiate(&execs.at(i), graphs.at(i), nullptr, nullptr, 0));
        log_message("graph instantiate done (stream=" + std::to_string(i) +
                    ", graph=" + std::to_string(reinterpret_cast<uint64_t>(graphs.at(i))) +
                    ", exec=" + std::to_string(reinterpret_cast<uint64_t>(execs.at(i))) + ")");
    }

    if(progress)
    {
        auto ready_count = progress->fetch_add(1) + 1;
        log_message("barrier reached (ready=" + std::to_string(ready_count) + ")");
    }
    log_message("waiting on launch barrier");
    future.wait();
    log_message("launch barrier released");

    log_message("launching graph");
    for(uint64_t i = 0; i < nstream; ++i)
    {
        log_message("graph launch begin (stream=" + std::to_string(i) + ")");
        checkHipErrors(hipGraphLaunch(execs.at(i), streams.at(i)));
        log_message("graph launch done (stream=" + std::to_string(i) + ")");
    }

    for(uint64_t i = 0; i < nstream; ++i)
    {
        log_message("stream synchronize begin (stream=" + std::to_string(i) + ")");
        checkHipErrors(hipStreamSynchronize(streams.at(i)));
        log_message("stream synchronize done (stream=" + std::to_string(i) + ")");
    }

    log_message("destroying graph");
    for(uint64_t i = 0; i < nstream; ++i)
    {
        log_message("graph destroy (stream=" + std::to_string(i) +
                    ", graph=" + std::to_string(reinterpret_cast<uint64_t>(graphs.at(i))) +
                    ", exec=" + std::to_string(reinterpret_cast<uint64_t>(execs.at(i))) + ")");
        checkHipErrors(hipGraphDestroy(graphs.at(i)));
    }

    log_message("freeing data");
    for(uint64_t i = 0; i < stream_num.size(); ++i)
    {
        auto& itr = stream_num.at(i);
        log_message("free data (stream=" + std::to_string(i) +
                    ", ptr=" + std::to_string(reinterpret_cast<uint64_t>(itr)) + ")");
        checkHipErrors(hipFree(itr));
    }

    log_message("returning");
}

int
main(int argc, char* argv[])
{
    std::cout << "[" << basename(argv[0]) << "] executing..." << std::endl;

    int ndevice_real = 0;
    checkHipErrors(hipGetDeviceCount(&ndevice_real));

    uint64_t nstream            = 1;
    uint64_t nkernel_per_stream = 12;
    uint64_t ndevice            = ndevice_real;

    if(argc > 1) nstream = std::stoul(argv[1]);
    if(argc > 2) nkernel_per_stream = std::stoul(argv[2]);
    if(argc > 3) ndevice = std::stoul(argv[3]);

    ndevice = std::min<uint64_t>(ndevice, ndevice_real);

    std::cout << "[DIAG-HG-APP-CONFIG] pid=" << getpid() << " ndevice_real=" << ndevice_real
              << " ndevice=" << ndevice << " nstream=" << nstream
              << " nkernel_per_stream=" << nkernel_per_stream << std::endl;

    auto progress = std::atomic<uint64_t>{0};
    auto promise  = std::promise<void>{};
    auto future   = promise.get_future().share();
    auto threads  = std::vector<std::thread>{};
    threads.reserve(ndevice);

    for(uint64_t i = 0; i < ndevice; ++i)
        threads.emplace_back(run, i, nstream, nkernel_per_stream, &progress, future);

    // wait for all threads to reach designated progress point
    auto wait_start  = std::chrono::steady_clock::now();
    auto last_report = wait_start;
    while(progress < ndevice)
    {
        std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
        auto now = std::chrono::steady_clock::now();
        if(now - last_report > std::chrono::seconds{2})
        {
            std::cout << "[DIAG-HG-APP-PROGRESS] ready=" << progress.load()
                      << " expected=" << ndevice
                      << " wait_ms="
                      << std::chrono::duration_cast<std::chrono::milliseconds>(now - wait_start)
                             .count()
                      << std::endl;
            last_report = now;
        }
    }

    std::cout << "[DIAG-HG-APP-PROGRESS] all workers ready wait_ms="
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now() - wait_start)
                     .count()
              << std::endl;

    // release the threads
    promise.set_value();

    for(uint64_t i = 0; i < threads.size(); ++i)
    {
        std::cout << "[DIAG-HG-APP-JOIN] thread=" << i << " begin" << std::endl;
        auto& itr = threads.at(i);
        itr.join();
        std::cout << "[DIAG-HG-APP-JOIN] thread=" << i << " done" << std::endl;
    }

    for(uint64_t i = 0; i < ndevice; ++i)
    {
        std::cout << "[DIAG-HG-APP-SYNC] device=" << i << " begin" << std::endl;
        checkHipErrors(hipSetDevice(i));
        checkHipErrors(hipDeviceSynchronize());
        std::cout << "[DIAG-HG-APP-SYNC] device=" << i << " done" << std::endl;
    }

    std::cout << "[" << basename(argv[0]) << "] complete" << std::endl;
    return 0;
}
