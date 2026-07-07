/*
Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.

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

// Minimal rocSHMEM application used by rocprofiler-sdk integration tests. The
// goal is to invoke every host-stream API listed in rocshmem/src/api_trace.cc
// at least once so the tracing tool sees a full sample of the dispatch table:
//
//   barrier_all_on_stream      sync_all_on_stream
//   quiet_on_stream            alltoallmem_on_stream
//   broadcastmem_on_stream     getmem_on_stream
//   putmem_on_stream           putmem_signal_on_stream
//   signal_wait_until_on_stream
//
// The demo must be launched by an MPI/PMI launcher (mpirun / mpiexec / srun) that exposes
// the node-local rank via an environment variable; it uses that to pin each process to a
// GPU before rocshmem_init(). If launched without one, exit cleanly with status 0 so the
// validator interprets the trace as "unavailable" and skips rather than reporting a hard
// failure. rocSHMEM itself is not tied to Open MPI -- it bootstraps against whatever MPI it
// was built with (OpenMPI, MPICH, MVAPICH); the lookup below just covers common launchers.

#include <hip/hip_runtime.h>
#include <rocshmem/rocshmem.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

#define CHECK_HIP(cmd)                                                                             \
    do                                                                                             \
    {                                                                                              \
        hipError_t _err = (cmd);                                                                   \
        if(_err != hipSuccess)                                                                     \
        {                                                                                          \
            std::fprintf(stderr,                                                                   \
                         "[rocshmem-demo] HIP error '%s' at %s:%d\n",                              \
                         hipGetErrorString(_err),                                                  \
                         __FILE__,                                                                 \
                         __LINE__);                                                                \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while(0)

using namespace rocshmem;

int
main(int /*argc*/, char** /*argv*/)
{
    // The node-local rank env var name varies by launcher, so check the common ones instead
    // of tying the demo to a single MPI implementation.
    const char* local_rank = nullptr;
    for(const char* _var : {"OMPI_COMM_WORLD_LOCAL_RANK",  // Open MPI
                            "MV2_COMM_WORLD_LOCAL_RANK",   // MVAPICH2
                            "MPI_LOCALRANKID",             // MPICH / Hydra
                            "SLURM_LOCALID"})              // Slurm
    {
        local_rank = std::getenv(_var);
        if(local_rank != nullptr) break;
    }
    if(local_rank == nullptr)
    {
        std::fprintf(stderr,
                     "[rocshmem-demo] no node-local rank env var set (checked "
                     "OMPI_COMM_WORLD_LOCAL_RANK, MV2_COMM_WORLD_LOCAL_RANK, MPI_LOCALRANKID, "
                     "SLURM_LOCALID); this demo must be launched via an MPI/PMI launcher "
                     "(mpirun/mpiexec/srun). Exiting without exercising the rocSHMEM API so the "
                     "integration test reports tracing as unavailable.\n");
        return 0;
    }

    // Map the local MPI rank onto a visible GPU. On a typical multi-GPU box this
    // is the identity (rank 0 -> device 0, rank 1 -> device 1, ...). On CI runners
    // where HIP_VISIBLE_DEVICES limits visibility to a single device, the modulo
    // makes ranks share the GPU instead of failing with "invalid device ordinal".
    int device_count = 0;
    CHECK_HIP(hipGetDeviceCount(&device_count));
    if(device_count <= 0)
    {
        std::fprintf(stderr,
                     "[rocshmem-demo] no HIP devices visible; skipping rocSHMEM exercise.\n");
        return 0;
    }
    CHECK_HIP(hipSetDevice(std::atoi(local_rank) % device_count));

    rocshmem_init();

    const int my_pe = rocshmem_my_pe();
    const int n_pes = std::max(rocshmem_n_pes(), 1);
    const int peer  = (n_pes > 1) ? ((my_pe + 1) % n_pes) : my_pe;

    constexpr size_t kPerPeBytes = 64;
    const size_t     kTotalBytes = kPerPeBytes * static_cast<size_t>(n_pes);

    auto* src    = static_cast<char*>(rocshmem_malloc(kTotalBytes));
    auto* dst    = static_cast<char*>(rocshmem_malloc(kTotalBytes));
    auto* signal = static_cast<uint64_t*>(rocshmem_malloc(sizeof(uint64_t)));

    if(src == nullptr || dst == nullptr || signal == nullptr)
    {
        std::fprintf(stderr, "[rocshmem-demo] symmetric heap allocation failed\n");
        rocshmem_global_exit(1);
    }

    std::memset(src, static_cast<int>(my_pe & 0xFF), kTotalBytes);
    std::memset(dst, 0, kTotalBytes);
    *signal = 0;
    CHECK_HIP(hipDeviceSynchronize());

    hipStream_t stream{};
    CHECK_HIP(hipStreamCreate(&stream));

    // Run the full API set kIterations times so the tracer is exercised under
    // repetition (stress test). The signal is left at its post-iter-1 value
    // (>= 1) so subsequent signal_wait_until calls return immediately - they
    // still trace, which is what this test cares about. Resetting the signal
    // between iterations is a host/stream race because barrier_all_on_stream
    // and sync_all_on_stream only order on the stream, not on the host.
    constexpr int kIterations = 3;
    for(int iter = 0; iter < kIterations; ++iter)
    {
        rocshmem_putmem_on_stream(dst, src, kPerPeBytes, peer, stream);
        rocshmem_quiet_on_stream(stream);
        rocshmem_getmem_on_stream(dst, src, kPerPeBytes, peer, stream);

        rocshmem_putmem_signal_on_stream(dst,
                                         src,
                                         kPerPeBytes,
                                         signal,
                                         /*signal=*/1,
                                         ROCSHMEM_SIGNAL_SET,
                                         peer,
                                         stream);
        rocshmem_signal_wait_until_on_stream(signal, ROCSHMEM_CMP_GE, 1, stream);

        rocshmem_broadcastmem_on_stream(
            ROCSHMEM_TEAM_WORLD, dst, src, kPerPeBytes, /*pe_root=*/0, stream);

        rocshmem_alltoallmem_on_stream(ROCSHMEM_TEAM_WORLD, dst, src, kPerPeBytes, stream);

        rocshmem_barrier_all_on_stream(stream);
        rocshmem_sync_all_on_stream(stream);
    }

    CHECK_HIP(hipStreamSynchronize(stream));
    CHECK_HIP(hipStreamDestroy(stream));

    rocshmem_free(src);
    rocshmem_free(dst);
    rocshmem_free(signal);

    rocshmem_finalize();
    return 0;
}
