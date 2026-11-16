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

// MPI + HIP application for the kernel-replay MPI integration test.
//
// Every rank launches two kernels per iteration and joins an MPI collective:
//
//   compute_kernel       rank-local, in-place update of a device buffer. This is the kernel the
//                        test asks kernel replay to profile, so it is replayed once per counter
//                        group with device memory restored between passes.
//   peer_visible_kernel  stands in for a kernel with effects outside its own process (an RCCL or
//                        GPU-aware MPI collective). The test excludes it with
//                        --kernel-exclude-regex, so it must run exactly once per launch.
//
// Both kernels bump a counter in pinned, device-mapped host memory. Replay snapshots only
// coarse-grained device allocations (host and fine-grained memory are out of scope), so that
// counter is not restored between passes and records how many times each kernel really executed.
// The app writes those counts to <prefix>.rank<R>.txt for the validator.
//
// The application also checks its own results on every rank, and the ranks cross-check each other
// with MPI_Allreduce: a replay that failed to restore a rank's device memory shows up as a
// mismatch here before any profiler output is read.

#include <hip/hip_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                                               \
        }                                                                                          \
    } while(0)

namespace
{
constexpr int kCheckedElems = 1024;

__global__ void
compute_kernel(float* y, int n, unsigned long long* exec_count)
{
    const int stride = blockDim.x * gridDim.x;
    for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += stride)
        y[i] = y[i] + 1.0f;

    if(blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(exec_count, 1ull);
}

__global__ void
peer_visible_kernel(float* buf, int n, float tag, unsigned long long* exec_count)
{
    const int stride = blockDim.x * gridDim.x;
    for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += stride)
        buf[i] = tag;

    if(blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(exec_count, 1ull);
}

bool
approx_equal(double got, double want)
{
    return std::fabs(got - want) <= 1e-3 * std::fabs(want) + 1e-3;
}
}  // namespace

int
main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int         n      = (argc > 1) ? std::atoi(argv[1]) : 65536;
    const int         iters  = (argc > 2) ? std::atoi(argv[2]) : 4;
    const std::string prefix = (argc > 3) ? argv[3] : "";

    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if(device_count < 1)
    {
        fprintf(stderr, "rank %d: no HIP device\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    HIP_CHECK(hipSetDevice(rank % device_count));

    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    // Distinct per-rank starting value so a restore that mixed up ranks would be visible.
    std::vector<float> h_y(n, static_cast<float>(rank));
    float*             d_y    = nullptr;
    float*             d_peer = nullptr;
    HIP_CHECK(hipMalloc(&d_y, bytes));
    HIP_CHECK(hipMalloc(&d_peer, bytes));
    HIP_CHECK(hipMemcpy(d_y, h_y.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_peer, 0, bytes));

    unsigned long long* h_counts = nullptr;
    unsigned long long* d_counts = nullptr;
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h_counts),
                            2 * sizeof(unsigned long long),
                            hipHostMallocMapped | hipHostMallocCoherent));
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&d_counts), h_counts, 0));
    h_counts[0] = 0;
    h_counts[1] = 0;

    const int checked = std::min(n, kCheckedElems);
    int       status  = EXIT_SUCCESS;

    for(int iter = 0; iter < iters; ++iter)
    {
        compute_kernel<<<64, 256>>>(d_y, n, d_counts + 0);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        peer_visible_kernel<<<64, 256>>>(d_peer, n, static_cast<float>(rank + 1), d_counts + 1);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Each app-observed launch advances every element by exactly one, however many times the
        // profiler replayed it.
        HIP_CHECK(hipMemcpy(h_y.data(), d_y, bytes, hipMemcpyDeviceToHost));
        const double want_local = static_cast<double>(rank) + static_cast<double>(iter + 1);
        double       local      = 0.0;
        for(int i = 0; i < checked; ++i)
        {
            if(!approx_equal(h_y[i], want_local))
            {
                fprintf(stderr,
                        "rank %d iter %d elem %d: %f != %f\n",
                        rank,
                        iter,
                        i,
                        h_y[i],
                        want_local);
                status = EXIT_FAILURE;
                break;
            }
            local += h_y[i];
        }

        // Host-side MPI collective after the kernels completed: every rank must agree on the
        // global result, which is only true if every rank's replay restored its own memory.
        double global = 0.0;
        MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double want_global = 0.0;
        for(int r = 0; r < size; ++r)
            want_global += (static_cast<double>(r) + static_cast<double>(iter + 1)) * checked;
        if(!approx_equal(global, want_global))
        {
            fprintf(
                stderr, "rank %d iter %d: allreduce %f != %f\n", rank, iter, global, want_global);
            status = EXIT_FAILURE;
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    HIP_CHECK(hipDeviceSynchronize());
    const unsigned long long compute_execs = h_counts[0];
    const unsigned long long peer_execs    = h_counts[1];

    printf("rank %d/%d: iters=%d compute_execs=%llu peer_execs=%llu status=%s\n",
           rank,
           size,
           iters,
           compute_execs,
           peer_execs,
           status == EXIT_SUCCESS ? "OK" : "FAIL");

    if(!prefix.empty())
    {
        const auto path = std::filesystem::path{prefix + ".rank" + std::to_string(rank) + ".txt"};
        if(path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
        std::ofstream out{path};
        out << "compute_execs=" << compute_execs << "\n";
        out << "peer_execs=" << peer_execs << "\n";
    }

    HIP_CHECK(hipHostFree(h_counts));
    HIP_CHECK(hipFree(d_y));
    HIP_CHECK(hipFree(d_peer));

    int world_status = status;
    MPI_Allreduce(&status, &world_status, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return world_status;
}
