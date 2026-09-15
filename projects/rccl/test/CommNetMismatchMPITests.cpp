/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file CommNetMismatchMPITests.cpp
 * @brief MPI test for per-rank Net device-count mismatch handling (AICOMRCCL-1213).
 *
 * Feature under test: during communicator init, RCCL AllGathers each rank's
 * @c localNetDeviceCount (the number of NET devices reported by the active net
 * plugin, i.e. @c ncclNet->devices()) inside the @c allGatherInfo struct. Rank 0
 * compares the min/max across ranks and, on a mismatch, either continues with a
 * warning or fails init with @c ncclSystemError depending on
 * @c NCCL_IGNORE_NET_MISMATCH (default 1 = ignore). Multi-NIC nodes with an
 * unequal NIC count per rank previously failed later with obscure transport
 * errors; this guard turns that into an explicit signal (see src/init.cc).
 *
 * The test synthesizes a mismatch on a SINGLE node without special hardware by
 * forcing the built-in socket net (NCCL_IB_DISABLE=1, NCCL_NET=Socket) and
 * giving each rank a different NCCL_SOCKET_IFNAME list so the socket net's
 * devices() returns a different count per rank:
 *   - rank 0   -> two interfaces (localNetDeviceCount = 2)
 *   - rank !=0 -> one interface  (localNetDeviceCount = 1)
 *
 * The expected behavior is selected by NCCL_IGNORE_NET_MISMATCH in the
 * environment:
 *   - NCCL_IGNORE_NET_MISMATCH=1 (or unset): init SUCCEEDS on every rank.
 *   - NCCL_IGNORE_NET_MISMATCH=0            : init FAILS on rank 0 with
 *                                             ncclSystemError.
 *
 * IMPORTANT: NCCL_PARAM values (NCCL_IGNORE_NET_MISMATCH) and the socket net's
 * interface list are cached per process, so the two modes must each run in their
 * OWN process. The test_runner configs launch this filter twice (once per mode,
 * each a separate mpirun), which satisfies that; see
 * tools/scripts/test_runner/configs/*.json (net_device_count_mismatch).
 *
 * Init is done non-blocking (ncclConfig_t.blocking = 0) with a bounded wait so
 * that a rank which never receives the abort signal (the mismatch check runs only
 * on rank 0, mirroring upstream NCCL) times out and aborts locally instead of
 * hanging the job.
 */

#ifdef MPI_TESTS_ENABLED

#include "MPIEnvironment.hpp"
#include "TestChecks.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include <dirent.h>
#include <unistd.h>

namespace
{

// Bounded wait (seconds) for a non-blocking init to resolve before we treat the
// rank as stuck and abort it locally. Comfortably above bootstrap latency but
// short enough that a rank-0-only abort does not stall the job.
constexpr double kInitTimeoutSec = 25.0;

double nowSec()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

bool startsWithAny(const std::string& name, const std::vector<std::string>& prefixes)
{
    for(const auto& p : prefixes)
    {
        if(name.rfind(p, 0) == 0)
            return true;
    }
    return false;
}

// Discover usable network interfaces for the socket net: skip loopback and the
// usual virtual/bridge devices, and require operstate "up". Allows an override
// via RCCL_MISMATCH_IFACES ("eth0 eth1").
std::vector<std::string> discoverInterfaces()
{
    if(const char* override_ifaces = getenv("RCCL_MISMATCH_IFACES"))
    {
        std::vector<std::string> out;
        std::string              s(override_ifaces);
        size_t                   pos = 0;
        while(pos < s.size())
        {
            size_t sp = s.find_first_of(" ,", pos);
            if(sp == std::string::npos)
                sp = s.size();
            if(sp > pos)
                out.emplace_back(s.substr(pos, sp - pos));
            pos = sp + 1;
        }
        return out;
    }

    static const std::vector<std::string> kSkipPrefixes
        = {"lo", "docker", "veth", "virbr", "br-", "br0", "cni", "flannel", "tun", "tap"};

    std::vector<std::string> names;
    if(DIR* dir = opendir("/sys/class/net"))
    {
        while(struct dirent* ent = readdir(dir))
        {
            std::string name = ent->d_name;
            if(name == "." || name == "..")
                continue;
            if(startsWithAny(name, kSkipPrefixes))
                continue;
            names.push_back(name);
        }
        closedir(dir);
    }

    // Deterministic ordering so all ranks agree on which iface is shared.
    std::sort(names.begin(), names.end());

    std::vector<std::string> out;
    for(const auto& name : names)
    {
        std::string path  = "/sys/class/net/" + name + "/operstate";
        bool        is_up = false;
        if(FILE* f = fopen(path.c_str(), "r"))
        {
            char state[32] = {0};
            if(fgets(state, sizeof(state), f))
                is_up = (strncmp(state, "up", 2) == 0);
            fclose(f);
        }
        if(is_up)
            out.push_back(name);
    }
    return out;
}

const char* stateStr(ncclResult_t r)
{
    switch(r)
    {
    case ncclSuccess: return "ncclSuccess";
    case ncclInProgress: return "ncclInProgress(timeout)";
    case ncclSystemError: return "ncclSystemError";
    default: return ncclGetErrorString(r);
    }
}

} // namespace

// Forces a per-rank Net device-count mismatch over the socket net and verifies
// the NCCL_IGNORE_NET_MISMATCH behavior. Rank 0 (which runs the mismatch check)
// gates the result; peer ranks are informational because the abort is rank-0
// only (matches upstream NCCL). Run once per mode (separate process each).
TEST(CommNetMismatchMPITest, NetDeviceCountMismatch)
{
    const int world_rank = MPIEnvironment::world_rank;
    const int world_size = MPIEnvironment::world_size;

    if(world_size < 2)
        GTEST_SKIP() << "Net device-count mismatch test requires at least 2 ranks";

    // Force the built-in socket net so NCCL_SOCKET_IFNAME controls the per-rank
    // Net device count. Set before any RCCL entry point.
    setenv("NCCL_IB_DISABLE", "1", 1);
    setenv("NCCL_NET", "Socket", 1);

    const std::vector<std::string> ifaces = discoverInterfaces();
    if(ifaces.size() < 2)
        GTEST_SKIP() << "need >= 2 usable network interfaces to synthesize a mismatch";

    // rank 0 sees two interfaces (count 2); every other rank sees one (count 1),
    // sharing ifaces[0] so bootstrap can still connect. This creates the mismatch.
    const std::string ifname
        = (world_rank == 0) ? (ifaces[0] + "," + ifaces[1]) : ifaces[0];
    setenv("NCCL_SOCKET_IFNAME", ifname.c_str(), 1);

    const char* ignore_env = getenv("NCCL_IGNORE_NET_MISMATCH");
    // Mirror the RCCL default (IGNORE_NET_MISMATCH defaults to 1 = ignore).
    const bool expect_ignore = (ignore_env == nullptr) || (atoi(ignore_env) != 0);

    ncclUniqueId id;
    if(world_rank == 0)
        ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);
    ASSERT_EQ(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD), MPI_SUCCESS);

    // Non-blocking init so a rank that never receives the abort signal can time
    // out locally instead of hanging.
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking     = 0;

    ncclComm_t   comm  = nullptr;
    ncclResult_t res   = ncclCommInitRankConfig(&comm, world_size, id, world_rank, &config);
    ncclResult_t state = res;

    if((res == ncclSuccess || res == ncclInProgress) && comm != nullptr)
    {
        const double start = nowSec();
        while(true)
        {
            ncclResult_t async = ncclInProgress;
            ncclCommGetAsyncError(comm, &async);
            if(async != ncclInProgress)
            {
                state = async;
                break;
            }
            if(nowSec() - start > kInitTimeoutSec)
            {
                state = ncclInProgress; // treat as timeout
                break;
            }
            usleep(20 * 1000);
        }
    }

    // Clean up: a fully-initialized comm is destroyed; anything else is aborted.
    if(comm != nullptr)
    {
        if(state == ncclSuccess)
        {
            // Prove the tolerated comm is actually usable with a tiny AllReduce.
            int*        buf    = nullptr;
            hipStream_t stream = nullptr;
            if(hipMalloc(&buf, sizeof(int)) == hipSuccess
               && hipStreamCreate(&stream) == hipSuccess)
            {
                int one = 1;
                (void)hipMemcpy(buf, &one, sizeof(int), hipMemcpyHostToDevice);
                ncclResult_t ar = ncclAllReduce(buf, buf, 1, ncclInt, ncclSum, comm, stream);
                if(ar == ncclSuccess || ar == ncclInProgress)
                    (void)hipStreamSynchronize(stream);
                else
                    state = ar;
            }
            if(stream)
                (void)hipStreamDestroy(stream);
            if(buf)
                (void)hipFree(buf);
            ncclCommDestroy(comm);
        }
        else
        {
            ncclCommAbort(comm);
        }
    }

    // Re-synchronize before returning. In abort mode rank 0 fails fast while
    // peer ranks wait out the bounded init timeout (the abort is rank-0 only),
    // so without this the ranks reach the harness teardown badly out of sync and
    // its barrier times out. This barrier makes them leave the test together.
    MPI_Barrier(MPI_COMM_WORLD);

    // Only rank 0 runs the mismatch check, so it gates the verdict. Peer ranks
    // are informational (the abort is not broadcast in upstream NCCL either).
    if(world_rank == 0)
    {
        TEST_INFO("NetDeviceCountMismatch mode=%s ifname=%s state=%s",
                  expect_ignore ? "ignore" : "abort",
                  ifname.c_str(),
                  stateStr(state));
        if(expect_ignore)
            EXPECT_EQ(state, ncclSuccess)
                << "with NCCL_IGNORE_NET_MISMATCH=1 the mismatch must be tolerated";
        else
            EXPECT_EQ(state, ncclSystemError)
                << "with NCCL_IGNORE_NET_MISMATCH=0 rank 0 init must fail on the mismatch";
    }
    else
    {
        TEST_INFO("NetDeviceCountMismatch (peer) rank=%d ifname=%s state=%s",
                  world_rank,
                  ifname.c_str(),
                  stateStr(state));
    }
}

#endif // MPI_TESTS_ENABLED
