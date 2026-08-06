// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * Exercises all 9 rocSHMEM host-stream APIs in a minimal 2-PE program.
 *
 * Every PE calls every API so that rocprofiler-systems tracing captures all
 * nine rocm_rocshmem_api spans:
 *
 *   barrier_all_on_stream      quiet_on_stream          sync_all_on_stream
 *   putmem_on_stream           getmem_on_stream
 *   putmem_signal_on_stream    signal_wait_until_on_stream
 *   broadcastmem_on_stream     alltoallmem_on_stream
 *
 * Launch with: mpirun -np 2 rocshmem
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <hip/hip_runtime.h>
#include <rocshmem/rocshmem.hpp>
using namespace rocshmem;

namespace
{
bool
check_hip(hipError_t err, const char* file, int line)
{
    if(err == hipSuccess) return true;
    fprintf(stderr, "[HIP error] %s:%d — %s\n", file, line, hipGetErrorString(err));
    return false;
}
}  // namespace

#define CHECK_HIP(cmd)                                                                   \
    do                                                                                   \
    {                                                                                    \
        if(!check_hip((cmd), __FILE__, __LINE__)) return 1;                              \
    } while(0)

int
main()
{
    // Guard: verify at least one GPU is accessible before committing to
    // rocshmem_init().  On systems where rocSHMEM is installed but the GPU
    // does not support it, rocshmem_init() may call abort(); this early check
    // provides a clean exit (code 2) that the test framework converts to a
    // skip rather than a failure.
    int        num_devices = 0;
    hipError_t hip_dev_err = hipGetDeviceCount(&num_devices);
    if(hip_dev_err != hipSuccess || num_devices == 0)
    {
        fprintf(stderr, "rocshmem: no GPU available (%s) — skipping\n",
                hipGetErrorString(hip_dev_err != hipSuccess ? hip_dev_err
                                                            : hipErrorNoDevice));
        return 2;
    }

    // rocSHMEM requires hipSetDevice() to be called BEFORE rocshmem_init()
    // so that the MPI PE is bound to its GPU before the symmetric heap is
    // allocated on that device.  Use OMPI_COMM_WORLD_LOCAL_RANK (set by
    // Open MPI before any process code runs) to pick the right device.
    const char* local_rank_env = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    const int   local_rank     = local_rank_env ? atoi(local_rank_env) : 0;
    CHECK_HIP(hipSetDevice(local_rank % num_devices));

    rocshmem_init();

    const int me   = rocshmem_my_pe();
    const int npes = rocshmem_n_pes();
    const int peer = (me + 1) % npes;

    hipStream_t stream = nullptr;
    CHECK_HIP(hipStreamCreate(&stream));

    // Symmetric buffers:
    //   src  — DATA_BYTES bytes sent by this PE
    //   dst  — npes * DATA_BYTES bytes for receives (alltoall needs npes slots)
    //   sig  — one uint64_t signal word per PE
    constexpr size_t DATA_BYTES = 64;

    auto* src = static_cast<uint8_t*>(rocshmem_malloc(DATA_BYTES));
    auto* dst = static_cast<uint8_t*>(rocshmem_malloc(npes * DATA_BYTES));
    auto* sig = static_cast<uint64_t*>(rocshmem_malloc(sizeof(uint64_t)));

    if(src == nullptr || dst == nullptr || sig == nullptr)
    {
        fprintf(stderr, "rocshmem: symmetric heap allocation failed\n");
        return 1;
    }

    memset(src, me, DATA_BYTES);
    memset(dst, 0, npes * DATA_BYTES);
    *sig = 0;

    // ------------------------------------------------------------------
    // 1. barrier_all_on_stream — global fence before any transfers
    rocshmem_barrier_all_on_stream(stream);
    CHECK_HIP(hipStreamSynchronize(stream));

    // 2. sync_all_on_stream — lightweight global synchronisation
    rocshmem_sync_all_on_stream(stream);
    CHECK_HIP(hipStreamSynchronize(stream));

    // 3. putmem_on_stream — each PE writes DATA_BYTES into its peer's dst
    rocshmem_putmem_on_stream(dst, src, DATA_BYTES, peer, stream);
    rocshmem_barrier_all_on_stream(stream);
    CHECK_HIP(hipStreamSynchronize(stream));

    // 4. getmem_on_stream — each PE reads DATA_BYTES from its peer's src
    rocshmem_getmem_on_stream(dst, src, DATA_BYTES, peer, stream);
    rocshmem_barrier_all_on_stream(stream);
    CHECK_HIP(hipStreamSynchronize(stream));

    // 5. putmem_signal_on_stream — write data and set remote signal atomically.
    //    Each PE targets its peer: after the put the peer's sig word is 1.
    // 6. signal_wait_until_on_stream — each PE waits for its own sig word to
    //    be set to 1 by its peer.
    //    The stream sync after step 5 guarantees the put (and its network
    //    transmission) is initiated before the wait is enqueued on the stream.
    *sig = 0;
    rocshmem_barrier_all_on_stream(stream);
    CHECK_HIP(hipStreamSynchronize(stream));

    rocshmem_putmem_signal_on_stream(dst, src, DATA_BYTES, sig, /*signal_value=*/1,
                                     ROCSHMEM_SIGNAL_SET, peer, stream);
    CHECK_HIP(hipStreamSynchronize(stream));

    rocshmem_signal_wait_until_on_stream(sig, ROCSHMEM_CMP_EQ, /*cmp_value=*/1, stream);
    CHECK_HIP(hipStreamSynchronize(stream));

    // 7. broadcastmem_on_stream — PE 0 broadcasts DATA_BYTES to all PEs
    rocshmem_broadcastmem_on_stream(ROCSHMEM_TEAM_WORLD, dst, src, DATA_BYTES,
                                    /*pe_root=*/0, stream);
    CHECK_HIP(hipStreamSynchronize(stream));

    // 8. alltoallmem_on_stream — each PE contributes DATA_BYTES; dst must be
    //    at least npes * DATA_BYTES (allocated above)
    rocshmem_alltoallmem_on_stream(ROCSHMEM_TEAM_WORLD, dst, src, DATA_BYTES, stream);
    CHECK_HIP(hipStreamSynchronize(stream));

    // 9. quiet_on_stream — ensure all outstanding remote operations are complete
    rocshmem_quiet_on_stream(stream);
    CHECK_HIP(hipStreamSynchronize(stream));
    // ------------------------------------------------------------------

    printf("[PE %d/%d] rocshmem: all 9 host-stream APIs completed\n", me, npes);

    rocshmem_free(src);
    rocshmem_free(dst);
    rocshmem_free(sig);
    CHECK_HIP(hipStreamDestroy(stream));

    rocshmem_finalize();
    return 0;
}
