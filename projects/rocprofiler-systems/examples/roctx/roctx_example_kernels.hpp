// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include <hip/hip_runtime.h>
#include <rocprofiler-sdk-roctx/roctx.h>

static std::mutex print_lock{};

#define HIP_CHECK(call)                                                                  \
    do                                                                                   \
    {                                                                                    \
        hipError_t err = call;                                                           \
        if(err != hipSuccess)                                                            \
        {                                                                                \
            std::lock_guard<std::mutex> _lk{ print_lock };                               \
            fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(err), __FILE__, \
                    __LINE__);                                                           \
            exit(1);                                                                     \
        }                                                                                \
    } while(0)

constexpr int KERNEL_ELEMENTS = 1024 * 1024;
constexpr int KERNEL_ITERS    = 1000;
constexpr int BLOCK_SIZE      = 256;
constexpr int GRID_SIZE       = (KERNEL_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE;

// Each code block launches the kernel this many times to produce enough
// sustained CPU+GPU work for the sampling profiler to capture thread timers.
constexpr int    KERNEL_LAUNCH_REPEATS = 100;
constexpr size_t DEFAULT_NUM_THREADS   = 2;

#define DEFINE_KERNEL(name, offset)                                                      \
    __global__ void name(float* data, int n)                                             \
    {                                                                                    \
        int idx = blockIdx.x * blockDim.x + threadIdx.x;                                 \
        if(idx < n)                                                                      \
            for(int i = 0; i < KERNEL_ITERS; ++i)                                        \
                data[idx] = __sinf(data[idx] + float(offset));                           \
    }

#define LAUNCH_KERNEL(name, data)                                                        \
    do                                                                                   \
    {                                                                                    \
        name<<<GRID_SIZE, BLOCK_SIZE>>>(data, KERNEL_ELEMENTS);                          \
        HIP_CHECK(hipDeviceSynchronize());                                               \
    } while(0)

#define LAUNCH_KERNEL_STREAM(name, stream, data)                                         \
    do                                                                                   \
    {                                                                                    \
        for(int _rep = 0; _rep < KERNEL_LAUNCH_REPEATS; ++_rep)                          \
        {                                                                                \
            name<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(data, KERNEL_ELEMENTS);           \
            HIP_CHECK(hipStreamSynchronize(stream));                                     \
        }                                                                                \
    } while(0)

struct gpu_buffer
{
    float* d_data = nullptr;

    gpu_buffer()
    {
        HIP_CHECK(hipMalloc(&d_data, KERNEL_ELEMENTS * sizeof(float)));
        HIP_CHECK(hipMemset(d_data, 0, KERNEL_ELEMENTS * sizeof(float)));
    }

    ~gpu_buffer() { (void) hipFree(d_data); }

    gpu_buffer(const gpu_buffer&)            = delete;
    gpu_buffer& operator=(const gpu_buffer&) = delete;

    float* get() { return d_data; }
};

// Reusable barrier for synchronizing N threads at region boundaries.
// Threads must all arrive before any can proceed past each wait() call.
class thread_barrier
{
public:
    explicit thread_barrier(size_t count)
    : m_threshold(count)
    , m_count(count)
    , m_generation(0)
    {}

    void wait()
    {
        std::unique_lock<std::mutex> lock{ m_mutex };
        auto                         gen = m_generation;
        if(--m_count == 0)
        {
            ++m_generation;
            m_count = m_threshold;
            lock.unlock();
            m_cv.notify_all();
        }
        else
        {
            m_cv.wait(lock, [this, gen] { return gen != m_generation; });
        }
    }

private:
    std::mutex              m_mutex;
    std::condition_variable m_cv;
    size_t                  m_threshold;
    size_t                  m_count;
    size_t                  m_generation;
};

inline void
run_on_threads(size_t nthreads, std::function<void(int, hipStream_t)> func)
{
    std::vector<hipStream_t> streams(nthreads);
    for(size_t i = 0; i < nthreads; ++i)
        HIP_CHECK(hipStreamCreate(&streams[i]));

    std::vector<std::thread> threads;
    for(size_t i = 1; i < nthreads; ++i)
        threads.emplace_back(func, static_cast<int>(i), streams[i]);

    func(0, streams[0]);

    for(auto& t : threads)
        t.join();

    for(size_t i = 0; i < nthreads; ++i)
        HIP_CHECK(hipStreamDestroy(streams[i]));
}
