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

// An MPI application whose time step is a halo exchange plus a reduction, run inside one
// host-replay range on every rank.
//
// Each step runs a kernel, copies a halo to the host, exchanges it around a ring with
// MPI_Sendrecv, copies the neighbor's halo back and mixes it into the field, and reduces a
// residual with MPI_Allreduce. The device field therefore depends on the neighbor's state at every
// step. If the ranks re-ran the range out of step, or one restored while its neighbor was still
// using its data, the MPI calls would not pair up (a hang) or the passes would end in different
// states (a failed check).
//
// Checks, per rank: every pass ends with the same device checksum and the same reduced value, the
// host state registered with rocprofiler_range_replay_host_track_memory shows one execution of the
// range rather than one per pass, and the number of passes and MPI calls is what the scenario
// expects. Results are also written to RR_MPI_OUT.app.rank<R>.txt for the validator.

#include "common.hpp"

#include <rocprofiler-sdk/experimental/range_replay.h>

#include <hip/hip_runtime.h>
#include <mpi.h>

#include <dlfcn.h>
#include <setjmp.h>

#include <cstring>
#include <fstream>

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

namespace
{
constexpr int kCells   = 1 << 14;
constexpr int kHalo    = 256;
constexpr int kMaxPass = 16;

__global__ void
rr_mpi_step(float* u, int n, float step)
{
    const int stride = blockDim.x * gridDim.x;
    for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += stride)
        u[i] = u[i] * 0.5f + 1.0f + step;
}

__global__ void
rr_mpi_apply_halo(float* u, const float* halo, int count)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < count) u[i] = 0.5f * (u[i] + halo[i]);
}

__global__ void
rr_mpi_init(float* u, int n, float seed)
{
    const int stride = blockDim.x * gridDim.x;
    for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += stride)
        u[i] = seed + static_cast<float>(i % 13);
}

// Host state the range mutates; registered so that it is rewound before every extra pass.
struct host_state_t
{
    uint64_t steps   = 0;
    double   reduced = 0.0;
};

host_state_t g_host{};

// Not registered, so they keep counting across passes: how many times the range ran, how many MPI
// calls this rank made, and what each pass ended with.
int      g_passes_run = 0;
uint64_t g_mpi_calls  = 0;
double   g_checksum[kMaxPass]{};
double   g_reduced[kMaxPass]{};
void*    g_leak = nullptr;

struct resources_t
{
    float* u      = nullptr;
    float* d_halo = nullptr;
    float* h_send = nullptr;
    float* h_recv = nullptr;
    int    rank   = 0;
    int    size   = 1;
    bool   leak   = false;

    decltype(&rocprofiler_range_replay_get_host_pass) get_pass = nullptr;
};

int
device_checksum(resources_t& r, double& out)
{
    static float h[kCells];
    HIP_CHECK(hipMemcpy(h, r.u, sizeof(h), hipMemcpyDeviceToHost));
    out = 0.0;
    for(float v : h)
        out += static_cast<double>(v);
    return EXIT_SUCCESS;
}

int
range_body(resources_t& r)
{
    uint64_t pass = 0;
    r.get_pass(&pass);

    const int next = (r.rank + 1) % r.size;
    const int prev = (r.rank + r.size - 1) % r.size;

    for(int step = 0; step < kSteps; ++step)
    {
        rr_mpi_step<<<64, 256>>>(r.u, kCells, static_cast<float>(step));
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        HIP_CHECK(hipMemcpy(r.h_send, r.u, kHalo * sizeof(float), hipMemcpyDeviceToHost));
        ++g_mpi_calls;
        MPI_Sendrecv(r.h_send,
                     kHalo,
                     MPI_FLOAT,
                     next,
                     7,
                     r.h_recv,
                     kHalo,
                     MPI_FLOAT,
                     prev,
                     7,
                     MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);

        HIP_CHECK(hipMemcpy(r.d_halo, r.h_recv, kHalo * sizeof(float), hipMemcpyHostToDevice));
        rr_mpi_apply_halo<<<(kHalo + 255) / 256, 256>>>(r.u, r.d_halo, kHalo);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        auto local = 0.0;
        for(int i = 0; i < kHalo; ++i)
            local += static_cast<double>(r.h_recv[i]);
        auto global = 0.0;
        ++g_mpi_calls;
        MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        g_host.reduced += global;
        ++g_host.steps;
    }

    // A device allocation made inside the range and still live at its end changes the allocation
    // set the entry snapshot was taken against: the range declines on this rank only.
    if(r.leak && pass == 0) HIP_CHECK(hipMalloc(&g_leak, 4096));

    HIP_CHECK(hipDeviceSynchronize());

    if(g_passes_run < kMaxPass)
    {
        if(device_checksum(r, g_checksum[g_passes_run]) != EXIT_SUCCESS) return EXIT_FAILURE;
        g_reduced[g_passes_run] = g_host.reduced;
    }
    ++g_passes_run;
    return EXIT_SUCCESS;
}
}  // namespace

int
main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    auto r = resources_t{};
    MPI_Comm_rank(MPI_COMM_WORLD, &r.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &r.size);

    auto* begin_fn = reinterpret_cast<decltype(&rocprofiler_range_replay_begin)>(
        dlsym(RTLD_DEFAULT, "rocprofiler_range_replay_begin"));
    auto* end_fn = reinterpret_cast<decltype(&rocprofiler_range_replay_end)>(
        dlsym(RTLD_DEFAULT, "rocprofiler_range_replay_end"));
    auto* jmpbuf_fn = reinterpret_cast<decltype(&rocprofiler_range_replay_host_jmpbuf)>(
        dlsym(RTLD_DEFAULT, "rocprofiler_range_replay_host_jmpbuf"));
    auto* track_fn = reinterpret_cast<decltype(&rocprofiler_range_replay_host_track_memory)>(
        dlsym(RTLD_DEFAULT, "rocprofiler_range_replay_host_track_memory"));
    r.get_pass = reinterpret_cast<decltype(&rocprofiler_range_replay_get_host_pass)>(
        dlsym(RTLD_DEFAULT, "rocprofiler_range_replay_get_host_pass"));

    if(begin_fn == nullptr || end_fn == nullptr || jmpbuf_fn == nullptr || track_fn == nullptr ||
       r.get_pass == nullptr)
    {
        fprintf(stderr, "[app] FAIL: the range replay host API is not in this process\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    const auto mode         = env_str("RR_MPI_MODE", "replay");
    const int  special_rank = env_int("RR_MPI_SPECIAL_RANK", 1);
    const int  expect       = env_int("RR_MPI_EXPECT_PASSES", 4);
    const auto out_prefix   = env_str("RR_MPI_OUT", "");
    const bool special      = (r.rank == special_rank);
    const bool want_leak    = special && mode == "leak";
    const bool arm          = !(special && mode == "unarmed");

    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    HIP_CHECK(hipSetDevice(r.rank % device_count));

    HIP_CHECK(hipMalloc(&r.u, kCells * sizeof(float)));
    HIP_CHECK(hipMalloc(&r.d_halo, kHalo * sizeof(float)));
    HIP_CHECK(hipHostMalloc(&r.h_send, kHalo * sizeof(float), hipHostMallocDefault));
    HIP_CHECK(hipHostMalloc(&r.h_recv, kHalo * sizeof(float), hipHostMallocDefault));

    // Load every module and warm HIP's copy path outside the range, as the single-process test
    // does. The warm-up also runs the MPI calls once so that the transport is connected before the
    // range's first exchange.
    rr_mpi_init<<<64, 256>>>(r.u, kCells, static_cast<float>(r.rank * 1000));
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    if(range_body(r) != EXIT_SUCCESS) return EXIT_FAILURE;

    rr_mpi_init<<<64, 256>>>(r.u, kCells, static_cast<float>(r.rank * 1000));
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    g_host       = host_state_t{};
    g_passes_run = 0;
    g_mpi_calls  = 0;
    r.leak       = want_leak;
    MPI_Barrier(MPI_COMM_WORLD);

    if(track_fn(&g_host, sizeof(g_host)) != ROCPROFILER_STATUS_SUCCESS)
    {
        fprintf(stderr, "[app] FAIL: rocprofiler_range_replay_host_track_memory failed\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    volatile auto begin_status = ROCPROFILER_STATUS_SUCCESS;
    if(arm)
    {
        // ROCPROFILER_RANGE_REPLAY_HOST_BEGIN, spelled out because the SDK is resolved via dlsym.
        if(sigsetjmp(*static_cast<sigjmp_buf*>(jmpbuf_fn()), 1) == 0)
            begin_status = begin_fn(kRangeId);
    }
    else
    {
        begin_status = begin_fn(kRangeId);
    }
    if(begin_status != ROCPROFILER_STATUS_SUCCESS)
    {
        fprintf(stderr, "[app] FAIL: begin returned status %d\n", static_cast<int>(begin_status));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if(range_body(r) != EXIT_SUCCESS) MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

    if(const auto status = end_fn(); status != ROCPROFILER_STATUS_SUCCESS)
    {
        fprintf(stderr, "[app] FAIL: end returned status %d\n", static_cast<int>(status));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // The range is closed and runs once from here on.
    auto ok = true;
    if(g_passes_run != expect)
    {
        fprintf(stderr,
                "[app] FAIL: rank %d ran %d passes, expected %d\n",
                r.rank,
                g_passes_run,
                expect);
        ok = false;
    }
    if(g_mpi_calls != static_cast<uint64_t>(g_passes_run) * kSteps * 2)
    {
        fprintf(stderr,
                "[app] FAIL: rank %d made %lu MPI calls for %d passes of %d steps\n",
                r.rank,
                static_cast<unsigned long>(g_mpi_calls),
                g_passes_run,
                kSteps);
        ok = false;
    }
    auto consistent = 1;
    for(int pass = 1; pass < g_passes_run && pass < kMaxPass; ++pass)
    {
        if(std::memcmp(&g_checksum[pass], &g_checksum[0], sizeof(double)) != 0 ||
           std::memcmp(&g_reduced[pass], &g_reduced[0], sizeof(double)) != 0)
        {
            fprintf(stderr,
                    "[app] FAIL: rank %d pass %d ended with checksum=%.17g reduced=%.17g, pass 0 "
                    "with checksum=%.17g reduced=%.17g\n",
                    r.rank,
                    pass,
                    g_checksum[pass],
                    g_reduced[pass],
                    g_checksum[0],
                    g_reduced[0]);
            consistent = 0;
            ok         = false;
        }
    }
    if(g_host.steps != static_cast<uint64_t>(kSteps))
    {
        fprintf(stderr,
                "[app] FAIL: rank %d host state counted %lu steps; it was not rewound between "
                "passes\n",
                r.rank,
                static_cast<unsigned long>(g_host.steps));
        ok = false;
    }

    if(!out_prefix.empty())
    {
        std::ofstream out{out_prefix + ".app.rank" + std::to_string(r.rank) + ".txt"};
        out << "passes_run=" << g_passes_run << "\n";
        out << "mpi_calls=" << g_mpi_calls << "\n";
        out << "host_steps=" << g_host.steps << "\n";
        out << "consistent=" << consistent << "\n";
        out << "result=" << (ok ? "PASS" : "FAIL") << "\n";
    }
    printf("[app] rank %d passes=%d mpi_calls=%lu host_steps=%lu %s\n",
           r.rank,
           g_passes_run,
           static_cast<unsigned long>(g_mpi_calls),
           static_cast<unsigned long>(g_host.steps),
           ok ? "PASS" : "FAIL");

    if(g_leak != nullptr) HIP_CHECK(hipFree(g_leak));
    HIP_CHECK(hipHostFree(r.h_recv));
    HIP_CHECK(hipHostFree(r.h_send));
    HIP_CHECK(hipFree(r.d_halo));
    HIP_CHECK(hipFree(r.u));

    auto world_ok = ok ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &world_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Finalize();
    return world_ok == 1 ? EXIT_SUCCESS : EXIT_FAILURE;
}
