/*
Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.

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

// Long-running MPI + HIP matrixTranspose workload for live process-attachment tests.
// Launched as: mpiexec -n <N> mpi-simple-attach [duration_sec] [host_sleep_ms]

#include <mpi.h>

#include <hip/hip_runtime.h>
#include <rocprofiler-sdk-roctx/roctx.h>

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <unistd.h>

#define WIDTH 1024
#define NUM (WIDTH * WIDTH)

#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4

#define HIP_API_CALL(CALL)                                                                         \
    {                                                                                              \
        const hipError_t error_ = (CALL);                                                          \
        if(error_ != hipSuccess)                                                                   \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "%s:%d :: HIP error : %s\n",                                                   \
                    __FILE__,                                                                      \
                    __LINE__,                                                                      \
                    hipGetErrorString(error_));                                                    \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    }

namespace
{
volatile sig_atomic_t g_stop = 0;

void
handle_stop(int)
{
    g_stop = 1;
}

int
parse_positive_int(const char* text, int fallback)
{
    if(!text || !*text) return fallback;
    char*             end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if(end == text || parsed <= 0) return fallback;
    return static_cast<int>(parsed);
}

}  // namespace

__global__ void
matrixTranspose(float* out, float* in, const int width)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    out[y * width + x] = in[x * width + y];
}

int
main(int argc, char** argv)
{
    const int duration_sec  = parse_positive_int(argc > 1 ? argv[1] : nullptr, 90);
    const int host_sleep_ms = parse_positive_int(argc > 2 ? argv[2] : nullptr, 100);

    std::signal(SIGINT, handle_stop);
    std::signal(SIGTERM, handle_stop);

    int rank = 0;
    int size = 1;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(rank == 0)
    {
        printf("mpi-simple-attach launcher pid=%d ranks=%d duration_sec=%d host_sleep_ms=%d\n",
               static_cast<int>(getpid()),
               size,
               duration_sec,
               host_sleep_ms);
        fflush(stdout);
    }

    int ndevice = 0;
    HIP_API_CALL(hipGetDeviceCount(&ndevice));
    if(ndevice == 0)
    {
        if(rank == 0) fprintf(stderr, "mpi-simple-attach: no HIP devices\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int device = rank % ndevice;
    HIP_API_CALL(hipSetDevice(device));

    hipDeviceProp_t devProp{};
    HIP_API_CALL(hipGetDeviceProperties(&devProp, device));
    printf("mpi-simple-attach rank=%d pid=%d device=%d name=%s\n",
           rank,
           static_cast<int>(getpid()),
           device,
           devProp.name);
    fflush(stdout);

    auto range_id = roctxRangeStart("mpi_simple_attach_main");

    float* Matrix             = static_cast<float*>(malloc(NUM * sizeof(float)));
    float* gpuMatrix          = nullptr;
    float* gpuTransposeMatrix = nullptr;
    if(!Matrix)
    {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for(int i = 0; i < NUM; ++i)
    {
        Matrix[i] = static_cast<float>(i) * 10.0f;
    }

    HIP_API_CALL(hipMalloc(reinterpret_cast<void**>(&gpuMatrix), NUM * sizeof(float)));
    HIP_API_CALL(hipMalloc(reinterpret_cast<void**>(&gpuTransposeMatrix), NUM * sizeof(float)));
    HIP_API_CALL(hipMemcpy(gpuMatrix, Matrix, NUM * sizeof(float), hipMemcpyHostToDevice));

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(duration_sec);
    int iteration = 0;

    while(!g_stop && std::chrono::steady_clock::now() < deadline)
    {
        ++iteration;
        hipLaunchKernelGGL(matrixTranspose,
                           dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
                           dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                           0,
                           0,
                           gpuTransposeMatrix,
                           gpuMatrix,
                           WIDTH);
        HIP_API_CALL(hipDeviceSynchronize());

        if(host_sleep_ms > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(host_sleep_ms));
    }

    roctxRangeStop(range_id);

    HIP_API_CALL(hipFree(gpuMatrix));
    HIP_API_CALL(hipFree(gpuTransposeMatrix));
    free(Matrix);

    printf("mpi-simple-attach rank=%d finished iterations=%d\n", rank, iteration);
    fflush(stdout);

    MPI_Finalize();
    return g_stop ? 130 : 0;
}
