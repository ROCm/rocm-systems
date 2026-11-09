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

// An order-dependent computation run as one range through rocprofiler::range_replay::host_run.
//
// Each step runs three kernels on two streams joined by events (an affine update, a table mix and
// a block reduction), copies the reduction to the host, and lets the host derive the next step's
// kernel arguments and mix table from it, copying the table back. Every kernel reads what the
// previous one wrote and the host computation feeds the next step, so a pass that did not start
// from the restored entry state would end elsewhere. The arithmetic is wrapping 32-bit unsigned, so
// the device result is checked bit for bit against a host reference of one execution.
//
// The checks: the body ran RR_API_EXPECT_PASSES times with passes 0..N-1, every pass and the state
// after the range equal one execution, the registered host state was rewound, once-only output ran
// on pass 0 only, and a C++ object in the body was destroyed on every pass. With
// RR_API_THROW_PASS=k the body throws on pass k instead, and the exception must reach main.

#include "range.hpp"

#include <rocprofiler-sdk/experimental/range_replay.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#define HIP_CHECK(call)                                                                            \
    do                                                                                             \
    {                                                                                              \
        hipError_t _err = (call);                                                                  \
        if(_err != hipSuccess)                                                                     \
        {                                                                                          \
            fprintf(                                                                               \
                stderr, "HIP error '%s' at %s:%d\n", hipGetErrorString(_err), __FILE__, __LINE__); \
            return EXIT_FAILURE;                                                                   \
        }                                                                                          \
    } while(0)

// The range body returns void, so a failed call inside it throws, which also closes the range.
#define HIP_THROW(call)                                                                            \
    do                                                                                             \
    {                                                                                              \
        hipError_t _err = (call);                                                                  \
        if(_err != hipSuccess) throw hip_error{_err, __LINE__};                                    \
    } while(0)

namespace
{
struct hip_error : std::runtime_error
{
    hip_error(hipError_t err, int line)
    : std::runtime_error{std::string{hipGetErrorString(err)} + " at line " + std::to_string(line)}
    {}
};

struct injected_failure : std::runtime_error
{
    using std::runtime_error::runtime_error;
};

__host__ __device__ inline uint32_t
affine(uint32_t value, uint32_t a, uint32_t b, uint32_t index)
{
    return value * a + b + index;
}

__host__ __device__ inline uint32_t
mix(uint32_t value, uint32_t table_entry)
{
    const uint32_t v = value ^ table_entry;
    return (v << 5) | (v >> 27);
}

__global__ void
rr_api_affine(uint32_t* x, uint32_t a, uint32_t b)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < kN) x[i] = affine(x[i], a, b, static_cast<uint32_t>(i));
}

__global__ void
rr_api_mix(uint32_t* x, const uint32_t* table)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < kN) x[i] = mix(x[i], table[i % kTable]);
}

// Wrapping sums are associative, so the reduction order does not change the result.
__global__ void
rr_api_reduce(const uint32_t* x, uint32_t* partials)
{
    __shared__ uint32_t s[kBlock];
    const int           t = threadIdx.x;
    s[t]                  = x[blockIdx.x * kBlock + t];
    __syncthreads();
    for(int half = kBlock / 2; half > 0; half >>= 1)
    {
        if(t < half) s[t] += s[t + half];
        __syncthreads();
    }
    if(t == 0) partials[blockIdx.x] = s[0];
}

// Host state the range reads and writes, registered so that every pass starts from its entry value.
struct host_state_t
{
    uint32_t a            = 0;
    uint32_t b            = 0;
    uint64_t steps        = 0;
    uint32_t sums[kSteps] = {};
};

void
init_host_state(host_state_t& h)
{
    h   = host_state_t{};
    h.a = 5;
    h.b = 7;
}

void
init_field(uint32_t* x)
{
    for(int i = 0; i < kN; ++i)
        x[i] = static_cast<uint32_t>(i) * 2654435761u;
}

void
init_table(uint32_t* table)
{
    for(int k = 0; k < kTable; ++k)
        table[k] = static_cast<uint32_t>(k + 1) * 16777619u;
}

// The host computation between steps: the next step's kernel arguments and mix table depend on this
// step's reduction.
void
advance(host_state_t& h, uint32_t sum, uint32_t* table)
{
    h.sums[h.steps % kSteps] = sum;
    h.a                      = (h.a * 1664525u + sum) | 1u;
    h.b                      = (h.b ^ (sum >> 3)) + 0x9E3779B9u;
    for(int k = 0; k < kTable; ++k)
        table[k] = h.a * static_cast<uint32_t>(k + 1) + (h.b >> (k % 31));
    ++h.steps;
}

bool
same_host_state(const host_state_t& lhs, const host_state_t& rhs)
{
    return lhs.a == rhs.a && lhs.b == rhs.b && lhs.steps == rhs.steps &&
           std::memcmp(lhs.sums, rhs.sums, sizeof(lhs.sums)) == 0;
}

uint64_t
fnv1a(const uint32_t* data, size_t count)
{
    auto hash = uint64_t{14695981039346656037ULL};
    for(size_t i = 0; i < count; ++i)
    {
        hash ^= data[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

struct reference_t
{
    std::vector<uint32_t> x    = std::vector<uint32_t>(kN);
    host_state_t          host = {};
};

// One execution of the range on the host.
reference_t
reference_run()
{
    auto ref   = reference_t{};
    auto table = std::vector<uint32_t>(kTable);
    init_field(ref.x.data());
    init_host_state(ref.host);
    init_table(table.data());
    for(int step = 0; step < kSteps; ++step)
    {
        for(int i = 0; i < kN; ++i)
            ref.x[i] = affine(ref.x[i], ref.host.a, ref.host.b, static_cast<uint32_t>(i));
        for(int i = 0; i < kN; ++i)
            ref.x[i] = mix(ref.x[i], table[i % kTable]);
        auto sum = uint32_t{0};
        for(auto value : ref.x)
            sum += value;
        advance(ref.host, sum, table.data());
    }
    return ref;
}

struct resources_t
{
    hipStream_t a           = nullptr;
    hipStream_t b           = nullptr;
    hipEvent_t  affine_done = nullptr;
    hipEvent_t  mix_done    = nullptr;
    uint32_t*   x           = nullptr;
    uint32_t*   table       = nullptr;
    uint32_t*   partials    = nullptr;
    uint32_t*   h_partials  = nullptr;
    uint32_t*   h_table     = nullptr;
    uint32_t*   h_x         = nullptr;
};

// Registered with the SDK, so rewound before every re-executed pass.
host_state_t g_host = {};

// Not registered: these accumulate across passes and record what each pass saw.
int          g_passes_run               = 0;
uint64_t     g_pass_ids[kMaxPasses]     = {};
uint64_t     g_checksums[kMaxPasses]    = {};
host_state_t g_host_at_exit[kMaxPasses] = {};
int          g_pass_number_mismatches   = 0;
int          g_pass0_log_lines          = 0;
int          g_pass_objects_destroyed   = 0;

void
run_step(resources_t& r, int step, uint64_t pass)
{
    rr_api_affine<<<kBlocks, kBlock, 0, r.a>>>(r.x, g_host.a, g_host.b);
    HIP_THROW(hipGetLastError());
    HIP_THROW(hipEventRecord(r.affine_done, r.a));
    HIP_THROW(hipStreamWaitEvent(r.b, r.affine_done, 0));
    rr_api_mix<<<kBlocks, kBlock, 0, r.b>>>(r.x, r.table);
    HIP_THROW(hipGetLastError());
    HIP_THROW(hipEventRecord(r.mix_done, r.b));
    HIP_THROW(hipStreamWaitEvent(r.a, r.mix_done, 0));
    rr_api_reduce<<<kBlocks, kBlock, 0, r.a>>>(r.x, r.partials);
    HIP_THROW(hipGetLastError());
    HIP_THROW(hipMemcpyAsync(
        r.h_partials, r.partials, kBlocks * sizeof(uint32_t), hipMemcpyDeviceToHost, r.a));
    HIP_THROW(hipStreamSynchronize(r.a));

    auto sum = uint32_t{0};
    for(int k = 0; k < kBlocks; ++k)
        sum += r.h_partials[k];
    if(pass == 0)
    {
        printf("[app] step %d sum=%u\n", step, sum);
        ++g_pass0_log_lines;
    }

    advance(g_host, sum, r.h_table);
    HIP_THROW(
        hipMemcpyAsync(r.table, r.h_table, kTable * sizeof(uint32_t), hipMemcpyHostToDevice, r.b));
}

// Destroyed at the end of every pass, including one left by an exception: the body is an ordinary
// call, so the macro path's rule against C++ objects with destructors in the range does not apply.
struct pass_object_t
{
    std::vector<uint32_t> scratch = std::vector<uint32_t>(kBlocks);
    ~pass_object_t() { ++g_pass_objects_destroyed; }
};

int
env_int(const char* name, int fallback)
{
    const char* value = getenv(name);
    return (value != nullptr && *value != '\0') ? atoi(value) : fallback;
}
}  // namespace

int
main()
{
    const int expect_passes = env_int("RR_API_EXPECT_PASSES", 4);
    const int throw_pass    = env_int("RR_API_THROW_PASS", -1);

    auto r = resources_t{};
    HIP_CHECK(hipStreamCreateWithFlags(&r.a, hipStreamNonBlocking));
    HIP_CHECK(hipStreamCreateWithFlags(&r.b, hipStreamNonBlocking));
    HIP_CHECK(hipEventCreateWithFlags(&r.affine_done, hipEventDisableTiming));
    HIP_CHECK(hipEventCreateWithFlags(&r.mix_done, hipEventDisableTiming));
    HIP_CHECK(hipMalloc(&r.x, kN * sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&r.table, kTable * sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&r.partials, kBlocks * sizeof(uint32_t)));
    HIP_CHECK(hipHostMalloc(&r.h_partials, kBlocks * sizeof(uint32_t), hipHostMallocDefault));
    HIP_CHECK(hipHostMalloc(&r.h_table, kTable * sizeof(uint32_t), hipHostMallocDefault));
    HIP_CHECK(hipHostMalloc(&r.h_x, kN * sizeof(uint32_t), hipHostMallocDefault));

    // Warm-up step outside the range: HIP loads the module on the first launch of one of its
    // kernels and sets up its copy paths lazily, and neither should first happen inside the range.
    try
    {
        init_host_state(g_host);
        run_step(r, 0, 1);
    } catch(const std::exception& e)
    {
        fprintf(stderr, "[app] FAIL: warm-up step: %s\n", e.what());
        return EXIT_FAILURE;
    }
    HIP_CHECK(hipDeviceSynchronize());

    // The range's real starting state.
    init_host_state(g_host);
    init_field(r.h_x);
    init_table(r.h_table);
    HIP_CHECK(hipMemcpy(r.x, r.h_x, kN * sizeof(uint32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(r.table, r.h_table, kTable * sizeof(uint32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    if(rocprofiler_range_replay_host_track_memory(&g_host, sizeof(g_host)) !=
       ROCPROFILER_STATUS_SUCCESS)
    {
        fprintf(stderr, "[app] FAIL: rocprofiler_range_replay_host_track_memory failed\n");
        return EXIT_FAILURE;
    }

    auto body = [&r, throw_pass](uint64_t pass) {
        const auto pass_object = pass_object_t{};

        auto sdk_pass = uint64_t{~0ULL};
        if(rocprofiler_range_replay_get_host_pass(&sdk_pass) != ROCPROFILER_STATUS_SUCCESS ||
           sdk_pass != pass)
            ++g_pass_number_mismatches;

        for(int step = 0; step < kSteps; ++step)
            run_step(r, step, pass);

        HIP_THROW(hipDeviceSynchronize());
        HIP_THROW(hipMemcpy(r.h_x, r.x, kN * sizeof(uint32_t), hipMemcpyDeviceToHost));
        HIP_THROW(hipDeviceSynchronize());

        const int slot       = (g_passes_run < kMaxPasses) ? g_passes_run : kMaxPasses - 1;
        g_pass_ids[slot]     = pass;
        g_checksums[slot]    = fnv1a(r.h_x, kN);
        g_host_at_exit[slot] = g_host;
        ++g_passes_run;

        if(throw_pass >= 0 && pass == static_cast<uint64_t>(throw_pass))
            throw injected_failure{"injected failure"};
    };

    auto status          = ROCPROFILER_STATUS_ERROR;
    auto caught_injected = false;
    try
    {
        status = rocprofiler::range_replay::host_run(kRangeId, body);
    } catch(const injected_failure&)
    {
        caught_injected = true;
    } catch(const std::exception& e)
    {
        fprintf(stderr, "[app] FAIL: the range body failed: %s\n", e.what());
        return EXIT_FAILURE;
    }

    // From here on the range is closed.
    const auto ref          = reference_run();
    const auto ref_checksum = fnv1a(ref.x.data(), ref.x.size());
    HIP_CHECK(hipMemcpy(r.h_x, r.x, kN * sizeof(uint32_t), hipMemcpyDeviceToHost));

    printf("[app] status=%d passes=%d destroyed=%d log_lines=%d caught=%d steps=%lu\n",
           static_cast<int>(status),
           g_passes_run,
           g_pass_objects_destroyed,
           g_pass0_log_lines,
           caught_injected ? 1 : 0,
           static_cast<unsigned long>(g_host.steps));

    auto ok   = true;
    auto fail = [&ok](const char* what) {
        fprintf(stderr, "[app] FAIL: %s\n", what);
        ok = false;
    };

    if(g_pass_objects_destroyed != g_passes_run)
        fail("a C++ object in the body was not destroyed on every pass");
    if(g_pass0_log_lines != kSteps) fail("once-only output ran on a pass other than pass 0");
    if(g_pass_number_mismatches != 0)
        fail("the body's pass differs from rocprofiler_range_replay_get_host_pass");
    for(int pass = 0; pass < g_passes_run && pass < kMaxPasses; ++pass)
        if(g_pass_ids[pass] != static_cast<uint64_t>(pass)) fail("passes ran out of order");

    if(throw_pass >= 0)
    {
        if(!caught_injected) fail("the body's exception did not reach the caller of host_run");
        if(g_passes_run != throw_pass + 1) fail("the range ran past the throwing pass");
    }
    else
    {
        if(status != ROCPROFILER_STATUS_SUCCESS) fail("host_run did not return success");
        if(g_passes_run != expect_passes)
        {
            fprintf(
                stderr, "[app] FAIL: ran %d passes, expected %d\n", g_passes_run, expect_passes);
            ok = false;
        }
        for(int pass = 0; pass < g_passes_run && pass < kMaxPasses; ++pass)
        {
            if(g_checksums[pass] != ref_checksum)
            {
                fprintf(stderr,
                        "[app] FAIL: pass %d ended with checksum %llx, one execution gives %llx\n",
                        pass,
                        static_cast<unsigned long long>(g_checksums[pass]),
                        static_cast<unsigned long long>(ref_checksum));
                ok = false;
            }
            if(!same_host_state(g_host_at_exit[pass], ref.host))
                fail("a pass ended with host state other than one execution's (not rewound)");
        }
        if(std::memcmp(r.h_x, ref.x.data(), kN * sizeof(uint32_t)) != 0)
            fail("the device state after the range is not the state one execution produces");
        if(!same_host_state(g_host, ref.host))
            fail("the host state after the range is not the state one execution produces");
    }

    HIP_CHECK(hipHostFree(r.h_x));
    HIP_CHECK(hipHostFree(r.h_table));
    HIP_CHECK(hipHostFree(r.h_partials));
    HIP_CHECK(hipFree(r.partials));
    HIP_CHECK(hipFree(r.table));
    HIP_CHECK(hipFree(r.x));
    HIP_CHECK(hipEventDestroy(r.mix_done));
    HIP_CHECK(hipEventDestroy(r.affine_done));
    HIP_CHECK(hipStreamDestroy(r.b));
    HIP_CHECK(hipStreamDestroy(r.a));

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
