/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// End-to-end, multi-rank tests for the NCCL Inspector point-to-point (P2P)
// Prometheus metrics (ticket AICOMRCCL-1214 / NCCL v2.30 sync):
//   - NCCL_INSPECTOR_ENABLE_P2P   toggles Send/Recv tracking.
//   - NCCL_INSPECTOR_PROM_DUMP    switches output to the Prometheus
//     node-exporter textfile format (nccl_inspector_metrics_<uuid>.prom).
//
// The metrics validated are nccl_p2p_bus_bandwidth_gbs and
// nccl_p2p_exec_time_microseconds, each carrying a p2p_operation="Send"/"Recv"
// label. A ring sendrecv workload is used because it issues real P2P
// operations (each rank sends to its successor and receives from its
// predecessor), which is exactly what the P2P metrics describe.
//
// How the Inspector behaves in Prometheus mode (see plugins/profiler/inspector):
//   * The dump thread writes .prom files periodically; in Prometheus mode the
//     interval is floored at 30 s to match a node exporter's poll interval.
//     The workload therefore runs for more than 30 s so at least one periodic
//     dump lands (with data) before the test reads the files.
//   * At communicator teardown the plugin *deletes* its .prom files (a node
//     exporter is expected to have scraped them during the run). We therefore
//     validate the files while the communicator is still alive, i.e. inside the
//     test body before TearDown() destroys it.
//   * NCCL_INSPECTOR_REQUIRE_KERNEL_TIMING=0 makes the test independent of
//     whether the plugin build exposes GPU-based kernel timing (otherwise
//     events with CPU-measured timing are discarded and nothing is recorded).
//
// The Inspector plugin is a separate .so loaded at runtime. BUILD_PROFILER_INSPECTOR
// (on by default with BUILD_TESTS) builds it in-tree and CMake hands this binary its
// path, so no setup is needed; NCCL_INSPECTOR_PLUGIN_SO or an inspector-named
// NCCL_PROFILER_PLUGIN override it. These tests SKIP if none of those resolves.

#ifdef MPI_TESTS_ENABLED

#include "MPITestBase.hpp"
#include "TestChecks.hpp"
#include "ResourceGuards.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include <unistd.h>

#include <hip/hip_runtime.h>
#include <mpi.h>

namespace fs = std::filesystem;

using namespace MPITestConstants;
using namespace RCCLTestGuards;

namespace
{
// Prometheus dump interval. In Prometheus mode the plugin floors this at 30 s
// (MIN_PROM_INTERVAL) to match the node-exporter poll cadence, so we request
// exactly the floor value.
constexpr const char* kDumpIntervalUs = "30000000"; // 30 s

// P2P workload duration. Must exceed the 30 s Prometheus dump interval so at
// least one periodic dump fires (with data) during the run; the plugin deletes
// its .prom files at teardown, so a periodic (not teardown) dump is the only
// one the test can observe.
constexpr double kWorkloadSeconds = 35.0;

// 8 MiB messages (comfortably above the Inspector's min tracked size).
constexpr int kMsgElems = 2 * 1024 * 1024;

const char* kP2pBwMetric   = "nccl_p2p_bus_bandwidth_gbs{";
const char* kP2pTimeMetric = "nccl_p2p_exec_time_microseconds{";
} // namespace

class InspectorPromP2pMPITest : public MPITestBase
{
protected:
    std::string dump_dir_;
    std::string plugin_so_;

    // Resolve the Inspector plugin .so, preferring an explicit override from the
    // environment over the copy built alongside this binary. Returns "" if none
    // is usable, in which case the test skips.
    static std::string resolvePluginSo()
    {
        // NCCL_PROFILER_PLUGIN is generic NCCL configuration and may name some
        // other profiler, which would emit no nccl_p2p_* metrics, so it counts
        // only when it names the Inspector. NCCL_INSPECTOR_PLUGIN_SO is specific
        // to the Inspector and is taken as given.
        const char* generic = getenv("NCCL_PROFILER_PLUGIN");
        if(generic && std::string(generic).find("inspector") == std::string::npos)
            generic = nullptr;

        const char* candidates[] = {getenv("NCCL_INSPECTOR_PLUGIN_SO"),
                                    generic,
#ifdef RCCL_INSPECTOR_PLUGIN_SO
                                    RCCL_INSPECTOR_PLUGIN_SO,
#endif
        };
        for(const char* c : candidates)
        {
            if(c && c[0] != '\0' && access(c, R_OK) == 0)
                return c;
        }
        return "";
    }

    // Per-rank dump directory so each rank validates its own device file(s)
    // without cross-rank/filesystem races (works single- and multi-node).
    std::string makeDumpDir()
    {
        const char*     base = getenv("NCCL_INSPECTOR_TEST_DUMP_DIR");
        const fs::path  root = base && base[0] ? fs::path(base) : fs::temp_directory_path();
        const fs::path  dir  = root / ("rccl_inspector_prom_pid" + std::to_string(getpid()) +
                                     "_rank" + std::to_string(MPIEnvironment::world_rank));
        std::error_code ec;
        fs::create_directories(dir, ec);
        // Clear any stale .prom files.
        removePromFiles(dir.string());
        return dir.string();
    }

    static void removePromFiles(const std::string& dir)
    {
        std::error_code ec;
        for(const auto& entry : fs::directory_iterator(dir, ec))
        {
            if(entry.path().extension() == ".prom")
                fs::remove(entry.path(), ec);
        }
    }

    void setInspectorEnv(bool enableP2p)
    {
        setenv("NCCL_PROFILER_PLUGIN", plugin_so_.c_str(), 1);
        setenv("NCCL_INSPECTOR_ENABLE", "1", 1);
        setenv("NCCL_INSPECTOR_ENABLE_P2P", enableP2p ? "1" : "0", 1);
        setenv("NCCL_INSPECTOR_PROM_DUMP", "1", 1);
        setenv("NCCL_INSPECTOR_REQUIRE_KERNEL_TIMING", "0", 1);
        setenv("NCCL_INSPECTOR_DUMP_DIR", dump_dir_.c_str(), 1);
        setenv("NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS", kDumpIntervalUs, 1);
        setenv("NCCL_INSPECTOR_DUMP_MIN_SIZE_BYTES", "1024", 1);
    }

    // Bidirectional ring sendrecv for at least kWorkloadSeconds so a populated
    // periodic Prometheus dump fires. Every rank issues one Send and one Recv
    // per iteration.
    void runSendRecvRing()
    {
        const int   world_rank = MPIEnvironment::world_rank;
        const int   world_size = MPIEnvironment::world_size;
        ncclComm_t  comm       = getActiveCommunicator();
        hipStream_t stream     = getActiveStream();
        ASSERT_MPI_TRUE(comm != nullptr && stream != nullptr);

        const int next = (world_rank + 1) % world_size;
        const int prev = (world_rank - 1 + world_size) % world_size;

        float* send_buf = nullptr;
        float* recv_buf = nullptr;
        HIP_CHECK(hipMalloc(&send_buf, kMsgElems * sizeof(float)));
        auto send_guard = makeScopeGuard([&]() { (void)hipFree(send_buf); });
        HIP_CHECK(hipMalloc(&recv_buf, kMsgElems * sizeof(float)));
        auto recv_guard = makeScopeGuard([&]() { (void)hipFree(recv_buf); });
        HIP_CHECK(hipMemset(send_buf, 1, kMsgElems * sizeof(float)));

        // The stop decision must be COLLECTIVE: every rank has to execute the
        // exact same number of ring iterations. If each rank timed the loop with
        // its own clock, differing per-iteration latencies (especially across
        // nodes) would let some ranks exit while others enter one more
        // ncclSend/ncclRecv + MPI_Barrier, deadlocking the survivors on the
        // collective. So rank 0 owns the clock and broadcasts a keep-going flag
        // that all ranks branch on identically.
        auto      start = std::chrono::steady_clock::now();
        long long iters = 0;
        while(true)
        {
            int keep_going = 0;
            if(world_rank == 0)
            {
                const double elapsed =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                keep_going = (elapsed < kWorkloadSeconds) ? 1 : 0;
            }
            MPI_Bcast(&keep_going, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if(!keep_going)
                break;

            RCCL_TEST_CHECK_GTEST_FAIL(ncclGroupStart());
            RCCL_TEST_CHECK_GTEST_FAIL(ncclSend(send_buf, kMsgElems, ncclFloat, next, comm, stream));
            RCCL_TEST_CHECK_GTEST_FAIL(ncclRecv(recv_buf, kMsgElems, ncclFloat, prev, comm, stream));
            RCCL_TEST_CHECK_GTEST_FAIL(ncclGroupEnd());
            HIP_CHECK(hipStreamSynchronize(stream));
            MPI_Barrier(MPI_COMM_WORLD);
            ++iters;
        }

        if(world_rank == 0)
            TEST_INFO("Ran %lld ring-sendrecv iterations over %.1f s", iters, kWorkloadSeconds);

        // The 30 s periodic dump already fired during the >30 s workload above;
        // this short grace period just ensures the file write has completed
        // before we read it.
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

    // Count P2P Prometheus samples in this rank's dump dir. Only well-formed
    // samples (valid p2p_operation + expected nranks label) are counted.
    void countP2pSamples(int& sendSamples, int& recvSamples, int& malformed)
    {
        sendSamples = recvSamples = malformed = 0;
        const std::string nranksLabel = "nranks=\"" +
                                        std::to_string(MPIEnvironment::world_size) + "\"";

        std::error_code ec;
        for(const auto& entry : fs::directory_iterator(dump_dir_, ec))
        {
            if(entry.path().extension() != ".prom")
                continue;

            std::ifstream f(entry.path());
            std::string   line;
            while(std::getline(f, line))
            {
                bool isBw   = line.rfind(kP2pBwMetric, 0) == 0;
                bool isTime = line.rfind(kP2pTimeMetric, 0) == 0;
                if(!isBw && !isTime)
                    continue;
                // Count the bandwidth metric only, to count one sample per
                // (operation) bucket rather than double-counting bw+time.
                if(!isBw)
                    continue;

                bool isSend = line.find("p2p_operation=\"Send\"") != std::string::npos;
                bool isRecv = line.find("p2p_operation=\"Recv\"") != std::string::npos;
                bool nranksOk = line.find(nranksLabel) != std::string::npos;
                bool nnodesOk = line.find("n_nodes=\"") != std::string::npos;

                if((isSend || isRecv) && nranksOk && nnodesOk)
                {
                    if(isSend)
                        ++sendSamples;
                    else
                        ++recvSamples;
                }
                else
                {
                    ++malformed;
                }
            }
        }
    }

    void SetUp() override
    {
        MPITestBase::SetUp();
        plugin_so_ = resolvePluginSo();
    }

    void TearDown() override
    {
        MPITestBase::TearDown(); // destroys comm (Inspector deletes its .prom files)
        if(!dump_dir_.empty())
        {
            removePromFiles(dump_dir_);
            std::error_code ec;
            fs::remove(dump_dir_, ec);
        }
    }
};

// ---------------------------------------------------------------------------
// P2P enabled: the Inspector must emit well-formed nccl_p2p_* Prometheus
// metrics (both Send and Recv) for the ring sendrecv workload.
// ---------------------------------------------------------------------------
TEST_F(InspectorPromP2pMPITest, PromP2pMetricsEmittedWhenEnabled)
{
    ASSERT_MPI_TRUE(validateTestPrerequisites(kMinProcessesForMPI));

    if(plugin_so_.empty())
        GTEST_SKIP() << "Inspector plugin not available: build with "
                        "-DBUILD_PROFILER_INSPECTOR=ON (the default with BUILD_TESTS), or set "
                        "NCCL_INSPECTOR_PLUGIN_SO to a librccl-profiler-inspector.so.";

    dump_dir_ = makeDumpDir();
    setInspectorEnv(/*enableP2p=*/true);

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    runSendRecvRing();

    int sendSamples = 0, recvSamples = 0, malformed = 0;
    countP2pSamples(sendSamples, recvSamples, malformed);

    if(malformed > 0)
        TEST_WARN("Rank %d found %d malformed P2P sample(s)", MPIEnvironment::world_rank, malformed);

    // Every rank sends to its successor and receives from its predecessor, so
    // each rank's own device file must contain both Send and Recv P2P samples,
    // all well-formed.
    bool localOk = (sendSamples > 0) && (recvSamples > 0) && (malformed == 0);
    ASSERT_MPI_TRUE(localOk);
}

// ---------------------------------------------------------------------------
// P2P disabled: no nccl_p2p_* metrics may be emitted even though the workload
// is entirely point-to-point.
// ---------------------------------------------------------------------------
TEST_F(InspectorPromP2pMPITest, PromP2pMetricsSuppressedWhenDisabled)
{
    ASSERT_MPI_TRUE(validateTestPrerequisites(kMinProcessesForMPI));

    if(plugin_so_.empty())
        GTEST_SKIP() << "Inspector plugin not available: build with "
                        "-DBUILD_PROFILER_INSPECTOR=ON (the default with BUILD_TESTS), or set "
                        "NCCL_INSPECTOR_PLUGIN_SO to a librccl-profiler-inspector.so.";

    dump_dir_ = makeDumpDir();
    setInspectorEnv(/*enableP2p=*/false);

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    runSendRecvRing();

    int sendSamples = 0, recvSamples = 0, malformed = 0;
    countP2pSamples(sendSamples, recvSamples, malformed);

    bool localNoP2p = (sendSamples == 0) && (recvSamples == 0);
    ASSERT_MPI_TRUE(localNoP2p);
}

#endif // MPI_TESTS_ENABLED
