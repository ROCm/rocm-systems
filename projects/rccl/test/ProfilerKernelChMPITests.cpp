/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// KernelCh profiling on the dedicated per-communicator profiler thread (NCCL 2.31.2).
//
// The kernel writes a start/stop timestamp and a per-channel work counter into
// host-pinned memory; ncclProfilerPostPlanWork() posts one op per (task, channel)
// from the host task that launches the plan, and the profiler thread fires
// KernelCh start/stop once the counters show the work ran. Posting from the host
// task rather than from proxy progress is what reaches plans with no proxy ops
// (intra-node P2P and SHM) and re-posts on every graph replay.
//
// Every check here goes through the recorder plugin, which sees exactly what any
// external profiler plugin would. The invariant most of them lean on: a Coll or
// P2p event advertises nChannels, and exactly that many KernelCh children must
// arrive under it. Completion-gating plugins (inspector, example) refcount the
// parent by that number, so a missing child leaks the parent and an extra one
// frees it early.
//
// RCCL's DDA collectives return before the enqueue path and raise no profiler
// events at all, so the suite runs with RCCL_DDA_ENABLE=0; otherwise a full
// single-node communicator takes DDA for AllReduce and records nothing.

#include "MPITestBase.hpp"
#include "TestChecks.hpp"

#include "comm.h"
#include "transport.h"
#include "plugin/nccl_profiler.h"
#include "plugin/profiler_kernelch_recorder.h"

#include <dlfcn.h>
#include <hip/hip_runtime.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <vector>

#ifdef MPI_TESTS_ENABLED

using namespace MPITestConstants;

namespace {

constexpr int kDrainTimeoutMs = 20000;
constexpr size_t kPluginPathLimit = 255;

long currentTid() { return (long)syscall(SYS_gettid); }

bool isTask(const RcclKchRecord& r) { return r.type == ncclProfileColl || r.type == ncclProfileP2p; }

// True on every rank if it is true on any, so all ranks take the same branch.
bool onAnyRank(bool local)
{
    int flag = local ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &flag, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    return flag != 0;
}

bool envIsOne(const char* name)
{
    const char* v = getenv(name);
    return v && atoi(v) == 1;
}

// Outcome of checking one snapshot. Kept as data so a test can add its own
// expectations on top of the shared ones.
struct KernelChStats {
  size_t tasks = 0;
  size_t kernelCh = 0;
  size_t advertised = 0;  // sum of nChannels over tasks
  std::set<long> kernelChTids;
};

}  // namespace

class ProfilerKernelChMPITest : public MPITestBase
{
protected:
    using ResetFn     = void (*)();
    using SnapshotFn  = size_t (*)(RcclKchRecord*, size_t);
    using CounterFn   = uint64_t (*)();
    using ThreadsFn   = size_t (*)(long*, size_t);

    void SetUp() override
    {
        MPITestBase::SetUp();
        callerTid_ = currentTid();

        // A binary built without the recorder, or a path RCCL would not accept, is
        // a reason to skip. A recorder that is there but will not load is a broken
        // build or install, and fails.
        std::string skip, fail;
#ifdef RCCL_TEST_KCH_RECORDER_NAME
        // Built and installed beside this binary, so it resolves the same way from
        // a build tree and from an install.
        std::error_code ec;
        std::filesystem::path exe = std::filesystem::read_symlink("/proc/self/exe", ec);
        if(!ec) pluginPath_ = (exe.parent_path() / RCCL_TEST_KCH_RECORDER_NAME).string();
#endif
        if(pluginPath_.empty())
        {
            skip = "KernelCh recorder plugin was not built into this binary";
        }
        else if(pluginPath_.size() >= kPluginPathLimit)
        {
            // RCCL ignores a longer NCCL_PROFILER_PLUGIN without saying so.
            skip = "recorder plugin path exceeds RCCL's plugin path limit: " + pluginPath_;
        }
        else
        {
            handle_ = dlopen(pluginPath_.c_str(), RTLD_NOW | RTLD_LOCAL);
            if(handle_ == nullptr)
            {
                fail = std::string("cannot dlopen ") + pluginPath_ + ": " + dlerror();
            }
            else
            {
                reset_       = (ResetFn)dlsym(handle_, "rcclKchRecorderReset");
                snapshot_    = (SnapshotFn)dlsym(handle_, "rcclKchRecorderSnapshot");
                initCount_   = (CounterFn)dlsym(handle_, "rcclKchRecorderInitCount");
                proxyOps_    = (CounterFn)dlsym(handle_, "rcclKchRecorderProxyOpCount");
                anomalies_   = (CounterFn)dlsym(handle_, "rcclKchRecorderAnomalies");
                proxyThreads_ = (ThreadsFn)dlsym(handle_, "rcclKchRecorderProxyThreads");
                if(!reset_ || !snapshot_ || !initCount_ || !proxyOps_ || !anomalies_ || !proxyThreads_)
                    fail = "KernelCh recorder plugin " + pluginPath_ + " is missing its query symbols";
            }
        }
        if(onAnyRank(!fail.empty()))
        {
            failReason_ = fail.empty() ? "KernelCh recorder plugin failed to load on another rank" : fail;
            return;
        }
        skipReason_ = mpiCoordinatedSkipReason(!skip.empty(), skip.c_str());
        if(!skipReason_.empty()) return;

        const char* prev = getenv("NCCL_PROFILER_PLUGIN");
        hadPrevPlugin_   = prev != nullptr;
        if(prev) prevPlugin_ = prev;
        setenv("NCCL_PROFILER_PLUGIN", pluginPath_.c_str(), 1);
        reset_();
    }

    void TearDown() override
    {
        if(ownedStream_)
        {
            (void)hipStreamSynchronize(ownedStream_);
            (void)hipStreamDestroy(ownedStream_);
            ownedStream_ = nullptr;
        }
        freeBuffers();
        for(auto it = ownedComms_.rbegin(); it != ownedComms_.rend(); ++it) (void)ncclCommDestroy(*it);
        ownedComms_.clear();
        MPITestBase::TearDown();
        if(skipReason_.empty() && failReason_.empty())
        {
            if(hadPrevPlugin_)
                setenv("NCCL_PROFILER_PLUGIN", prevPlugin_.c_str(), 1);
            else
                unsetenv("NCCL_PROFILER_PLUGIN");
        }
        if(handle_) dlclose(handle_);
    }

    // Create the fixture communicator and confirm RCCL loaded the recorder. RCCL
    // remembers a failed profiler load for the life of the process and never
    // retries, so a communicator created earlier in this binary without the
    // plugin configured leaves this one without it.
    std::string createCommWithRecorder()
    {
        bool failed = createTestCommunicator() != ncclSuccess;
        if(onAnyRank(failed))
        {
            ADD_FAILURE() << (failed ? "createTestCommunicator failed" : "createTestCommunicator failed on another rank");
            return "no communicator";
        }
        return recorderNotLoadedReason();
    }

    // The same for a communicator built from a config the test controls, plus a
    // stream of its own; both are released in TearDown.
    std::string createConfiguredCommWithRecorder(ncclConfig_t* config, ncclComm_t* comm, hipStream_t* stream)
    {
        bool failed = createConfiguredComm(config, comm) != ncclSuccess ||
                      hipStreamCreate(&ownedStream_) != hipSuccess;
        if(onAnyRank(failed))
        {
            ADD_FAILURE() << (failed ? "communicator or stream creation failed"
                                     : "communicator or stream creation failed on another rank");
            return "no communicator";
        }
        *stream = ownedStream_;
        return recorderNotLoadedReason();
    }

    std::string recorderNotLoadedReason()
    {
        bool notLoaded = initCount_() == 0;
        return mpiCoordinatedSkipReason(
            notLoaded,
            "RCCL did not load the recorder plugin; an earlier communicator in this process "
            "already failed a profiler load. Run this suite with NCCL_PROFILER_PLUGIN set "
            "for the whole process (see the profiler_kernelch config)");
    }

    // A communicator built from a config the test controls, destroyed in TearDown
    // unless the test destroys it first through destroyOwnedComm().
    ncclResult_t createConfiguredComm(ncclConfig_t* config, ncclComm_t* out)
    {
        ncclUniqueId id{};
        ncclResult_t idRes = ncclSuccess;
        if(MPIEnvironment::world_rank == 0) idRes = ncclGetUniqueId(&id);
        MPI_Bcast(&idRes, sizeof(idRes), MPI_BYTE, 0, MPI_COMM_WORLD);
        if(idRes != ncclSuccess) return idRes;
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
        ncclResult_t res =
            ncclCommInitRankConfig(out, MPIEnvironment::world_size, id, MPIEnvironment::world_rank, config);
        if(res == ncclSuccess) ownedComms_.push_back(*out);
        return res;
    }

    ncclResult_t destroyOwnedComm(ncclComm_t comm)
    {
        ownedComms_.erase(std::remove(ownedComms_.begin(), ownedComms_.end(), comm), ownedComms_.end());
        return ncclCommDestroy(comm);
    }

    std::vector<long> proxyThreadIds()
    {
        std::vector<long> tids(proxyThreads_(nullptr, 0) + 16);
        tids.resize(std::min(proxyThreads_(tids.data(), tids.size()), tids.size()));
        return tids;
    }

    std::vector<RcclKchRecord> snapshot()
    {
        size_t total = snapshot_(nullptr, 0);
        std::vector<RcclKchRecord> recs(total + 1024);
        size_t got = snapshot_(recs.data(), recs.size());
        recs.resize(std::min(got, recs.size()));
        return recs;
    }

    // Wait for the profiler thread to deliver every KernelCh the recorded tasks
    // advertise. The stream being synchronized only means the kernels finished;
    // the thread observes them asynchronously.
    std::vector<RcclKchRecord> waitForDrain(size_t minTasks)
    {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kDrainTimeoutMs);
        std::vector<RcclKchRecord> recs;
        while(true)
        {
            recs = snapshot();
            size_t tasks = 0, advertised = 0, stopped = 0;
            for(const auto& r : recs)
            {
                if(isTask(r))
                {
                    ++tasks;
                    advertised += r.nChannels;
                }
                if(r.type == ncclProfileKernelCh && r.stopEvents > 0) ++stopped;
            }
            if(tasks >= minTasks && stopped >= advertised) break;
            if(std::chrono::steady_clock::now() > deadline)
            {
                ADD_FAILURE() << "profiler thread did not drain within " << kDrainTimeoutMs << " ms: " << tasks
                              << " of " << minTasks << " Coll/P2p recorded, " << stopped << " of " << advertised
                              << " advertised KernelCh stopped";
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return recs;
    }

    // The invariants every KernelCh-enabled workload must satisfy, whatever the
    // transport. Non-fatal so every rank reaches the closing barrier. Each
    // communicator has one profiler thread unless it shares its parent's, so the
    // expected thread count is the number of thread-owning communicators involved.
    KernelChStats checkBalanced(const std::vector<RcclKchRecord>& recs, const char* what,
                                size_t expectedThreads = 1)
    {
        KernelChStats s;
        std::map<int64_t, std::vector<int>> childChannels;  // task index -> channel ids
        for(size_t i = 0; i < recs.size(); i++)
        {
            const RcclKchRecord& r = recs[i];
            if(isTask(r))
            {
                ++s.tasks;
                s.advertised += r.nChannels;
                childChannels[(int64_t)i];  // a task with no children must still be checked
                continue;
            }
            if(r.type != ncclProfileKernelCh) continue;
            ++s.kernelCh;
            s.kernelChTids.insert(r.startTid);
            EXPECT_GE(r.parentIndex, 0) << what << ": KernelCh on channel " << r.channelId
                                        << " has no Coll/P2p parent";
            if(r.parentIndex >= 0)
            {
                EXPECT_TRUE(isTask(recs[r.parentIndex]))
                    << what << ": KernelCh parent is not a Coll/P2p event";
                childChannels[r.parentIndex].push_back(r.channelId);
            }
            EXPECT_EQ(1, r.stopEvents) << what << ": KernelCh on channel " << r.channelId << " stopped "
                                       << r.stopEvents << " times";
            EXPECT_EQ(1, r.stopStates) << what << ": KernelCh on channel " << r.channelId
                                       << " got " << r.stopStates << " stop states";
            EXPECT_NE(0u, r.startTimer) << what << ": KernelCh start timer is zero";
            EXPECT_GE(r.stopTimer, r.startTimer)
                << what << ": KernelCh on channel " << r.channelId << " stops before it starts";
        }

        for(auto& [taskIdx, channels] : childChannels)
        {
            const RcclKchRecord& t = recs[taskIdx];
            EXPECT_EQ((size_t)t.nChannels, channels.size())
                << what << ": " << t.func << " (seq " << t.seqNumber << ") advertised " << t.nChannels
                << " channels but got " << channels.size() << " KernelCh children";
            std::sort(channels.begin(), channels.end());
            EXPECT_TRUE(std::adjacent_find(channels.begin(), channels.end()) == channels.end())
                << what << ": " << t.func << " got two KernelCh children on the same channel";
        }

        EXPECT_GT(s.tasks, 0u) << what << ": no Coll/P2p events recorded. RCCL's DDA collectives bypass the "
                                  "profiler entirely; run with RCCL_DDA_ENABLE=0";
        EXPECT_GT(s.kernelCh, 0u) << what << ": no KernelCh events recorded";
        EXPECT_EQ(s.advertised, s.kernelCh) << what << ": advertised vs delivered KernelCh";
        // Dedicated profiler threads, never the thread that called RCCL.
        EXPECT_EQ(expectedThreads, s.kernelChTids.size())
            << what << ": KernelCh arrived on " << s.kernelChTids.size() << " threads";
        EXPECT_EQ(0u, s.kernelChTids.count(callerTid_))
            << what << ": KernelCh delivered on the thread that issued the collectives";
        for(long tid : proxyThreadIds())
            EXPECT_EQ(0u, s.kernelChTids.count(tid)) << what << ": KernelCh delivered on a proxy thread";
        EXPECT_EQ(0u, anomalies_()) << what << ": recorder saw unbalanced start/stop calls";
        return s;
    }

    // Consecutive work on one channel of one communicator is serialized on the
    // device, so in the order the host posted it each KernelCh must start no
    // earlier than its predecessor on that channel stopped. A timestamp read from
    // a slot the device already reused for later work breaks this.
    void checkPerChannelOrder(const std::vector<RcclKchRecord>& recs, const char* what)
    {
        std::map<std::pair<uint64_t, int>, std::vector<const RcclKchRecord*>> byChannel;
        for(const auto& r : recs)
            if(r.type == ncclProfileKernelCh && r.parentIndex >= 0)
                byChannel[{r.commId, r.channelId}].push_back(&r);
        size_t violations = 0;
        for(auto& [key, list] : byChannel)
        {
            std::sort(list.begin(), list.end(), [&](const RcclKchRecord* a, const RcclKchRecord* b) {
                return recs[a->parentIndex].order < recs[b->parentIndex].order;
            });
            for(size_t i = 1; i < list.size(); i++)
            {
                if(list[i]->startTimer < list[i - 1]->stopTimer)
                {
                    if(violations++ < 5)
                        ADD_FAILURE() << what << ": channel " << key.second << " work #" << i << " starts at "
                                      << list[i]->startTimer << ", before work #" << i - 1 << " stopped at "
                                      << list[i - 1]->stopTimer;
                }
            }
        }
        EXPECT_EQ(0u, violations) << what << ": KernelCh timestamps out of order on a channel";
    }

    void allocBuffers(size_t count)
    {
        count_ = count;
        ASSERT_EQ(hipSuccess, hipMalloc(&send_, count * sizeof(float)));
        ASSERT_EQ(hipSuccess, hipMalloc(&recv_, count * sizeof(float)));
        ASSERT_EQ(hipSuccess, hipMemset(send_, 0, count * sizeof(float)));
    }

    void freeBuffers()
    {
        if(send_) (void)hipFree(send_);
        if(recv_) (void)hipFree(recv_);
        send_ = recv_ = nullptr;
    }

    ncclResult_t allReduce(ncclComm_t comm, hipStream_t stream)
    {
        return ncclAllReduce(send_, recv_, count_, ncclFloat, ncclSum, comm, stream);
    }

    std::string pluginPath_;
    void* handle_ = nullptr;
    ResetFn reset_ = nullptr;
    SnapshotFn snapshot_ = nullptr;
    CounterFn initCount_ = nullptr;
    CounterFn proxyOps_ = nullptr;
    CounterFn anomalies_ = nullptr;
    ThreadsFn proxyThreads_ = nullptr;
    std::string skipReason_;
    std::string failReason_;
    bool hadPrevPlugin_ = false;
    std::string prevPlugin_;
    long callerTid_ = 0;
    std::vector<ncclComm_t> ownedComms_;
    hipStream_t ownedStream_ = nullptr;
    void* send_ = nullptr;
    void* recv_ = nullptr;
    size_t count_ = 0;
};

#define KCH_SKIP_IF_NEEDED(reason)                 \
    do {                                           \
        std::string r_ = (reason);                 \
        if(!r_.empty()) { GTEST_SKIP() << r_; }    \
    } while(0)

// Every test opens with this: SetUp's verdict on the recorder, agreed by all ranks.
#define KCH_REQUIRE_RECORDER()                              \
    do {                                                    \
        if(!failReason_.empty()) { FAIL() << failReason_; } \
        KCH_SKIP_IF_NEEDED(skipReason_);                    \
    } while(0)

// Eager collectives: every Coll gets exactly its advertised KernelCh children,
// all delivered by one dedicated thread.
TEST_F(ProfilerKernelChMPITest, EagerAllReduceIsBalanced)
{
    KCH_REQUIRE_RECORDER();
    if(!validateTestPrerequisites(2)) GTEST_SKIP() << "needs at least 2 ranks";
    KCH_SKIP_IF_NEEDED(createCommWithRecorder());
    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    ASSERT_NO_FATAL_FAILURE(allocBuffers(1 << 20));

    constexpr int kIters = 16;
    for(int i = 0; i < kIters; i++) EXPECT_EQ(ncclSuccess, allReduce(comm, stream));
    EXPECT_EQ(hipSuccess, hipStreamSynchronize(stream));

    auto recs = waitForDrain(kIters);
    KernelChStats s = checkBalanced(recs, "eager AllReduce");
    EXPECT_EQ((size_t)kIters, s.tasks);
    checkPerChannelOrder(recs, "eager AllReduce");
    freeBuffers();
}

// The point of moving off the proxy thread: an all-intra-node communicator has no
// proxy ops at all, and must still be timed.
TEST_F(ProfilerKernelChMPITest, ProxyLessIntraNodeIsTimed)
{
    KCH_REQUIRE_RECORDER();
    if(!validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired, 1, kRequireSingleNode))
        GTEST_SKIP() << "needs at least 2 ranks on a single node";
    KCH_SKIP_IF_NEEDED(createCommWithRecorder());
    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    ASSERT_NO_FATAL_FAILURE(allocBuffers(1 << 20));

    constexpr int kIters = 8;
    for(int i = 0; i < kIters; i++) EXPECT_EQ(ncclSuccess, allReduce(comm, stream));
    EXPECT_EQ(hipSuccess, hipStreamSynchronize(stream));

    auto recs = waitForDrain(kIters);
    // The premise first: had this path used the proxy, the old design covered it too.
    EXPECT_EQ(0u, proxyOps_()) << "intra-node AllReduce posted proxy ops, so this run does not "
                                  "exercise a proxy-less plan";
    KernelChStats s = checkBalanced(recs, "intra-node AllReduce");
    EXPECT_EQ((size_t)kIters, s.tasks);
    freeBuffers();
}

// A plan with proxy ops must still get its KernelCh from the profiler thread
// rather than from proxy progress, as it did before the move. Across nodes the
// network gives it proxy ops; on one node, NCCL_P2P_DISABLE=1 with
// NCCL_SHM_DISABLE=1 forces the network transport, which also shows the
// recorder's proxy counter can move, the premise the proxy-less tests rely on.
TEST_F(ProfilerKernelChMPITest, ProxyPathKernelChOffProxyThread)
{
    KCH_REQUIRE_RECORDER();
    if(!validateTestPrerequisites(2)) GTEST_SKIP() << "needs at least 2 ranks";
    if(detectNodeCount() < 2 && !(envIsOne("NCCL_P2P_DISABLE") && envIsOne("NCCL_SHM_DISABLE")))
        GTEST_SKIP() << "needs ranks on 2 nodes, or NCCL_P2P_DISABLE=1 and NCCL_SHM_DISABLE=1 on one";
    KCH_SKIP_IF_NEEDED(createCommWithRecorder());
    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    ASSERT_NO_FATAL_FAILURE(allocBuffers(1 << 20));

    constexpr int kIters = 8;
    for(int i = 0; i < kIters; i++) EXPECT_EQ(ncclSuccess, allReduce(comm, stream));
    EXPECT_EQ(hipSuccess, hipStreamSynchronize(stream));

    auto recs = waitForDrain(kIters);
    // The premise: without proxy ops there is no proxy thread to tell apart. Only
    // ranks that touch the network have them (a tree leaf may not), so count all.
    unsigned long long localProxyOps = proxyOps_(), allProxyOps = 0;
    MPI_Allreduce(&localProxyOps, &allProxyOps, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    EXPECT_GT(allProxyOps, 0u) << "network AllReduce posted no proxy ops on any rank";
    KernelChStats s = checkBalanced(recs, "network AllReduce");
    EXPECT_EQ((size_t)kIters, s.tasks);
    freeBuffers();
}

// SHM is the other proxy-less transport. NCCL_P2P_DISABLE is read once per process,
// so the runner config sets it for this test alone.
TEST_F(ProfilerKernelChMPITest, ShmTransportIsTimed)
{
    KCH_REQUIRE_RECORDER();
    const char* p2pOff = getenv("NCCL_P2P_DISABLE");
    if(!p2pOff || atoi(p2pOff) == 0) GTEST_SKIP() << "set NCCL_P2P_DISABLE=1 to force the SHM transport";
    if(!validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired, 1, kRequireSingleNode))
        GTEST_SKIP() << "needs at least 2 ranks on a single node";
    KCH_SKIP_IF_NEEDED(createCommWithRecorder());
    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    ASSERT_NO_FATAL_FAILURE(allocBuffers(1 << 20));

    constexpr int kIters = 8;
    for(int i = 0; i < kIters; i++) EXPECT_EQ(ncclSuccess, allReduce(comm, stream));
    EXPECT_EQ(hipSuccess, hipStreamSynchronize(stream));

    // Without this the test passes just as well over P2P and proves nothing.
    int rings = 0, shmRings = 0;
    for(int c = 0; c < comm->nChannels; c++)
    {
        const ncclConnector& conn = comm->channels[c].peers[comm->channels[c].ring.next]->send[0];
        if(!conn.connected) continue;
        ++rings;
        if(conn.transportComm == &shmTransport.send) ++shmRings;
    }
    EXPECT_GT(rings, 0) << "no connected ring channels";
    EXPECT_EQ(rings, shmRings) << "NCCL_P2P_DISABLE=1 left " << rings - shmRings << " of " << rings
                               << " ring channels off the SHM transport";

    auto recs = waitForDrain(kIters);
    EXPECT_EQ(0u, proxyOps_()) << "SHM AllReduce posted proxy ops";
    KernelChStats s = checkBalanced(recs, "SHM AllReduce");
    EXPECT_EQ((size_t)kIters, s.tasks);
    freeBuffers();
}

// Both halves of a send/recv pair advertise their own channel count and get that
// many KernelCh, reading the shared per-channel device slot.
TEST_F(ProfilerKernelChMPITest, SendRecvRingIsBalanced)
{
    KCH_REQUIRE_RECORDER();
    if(!validateTestPrerequisites(2)) GTEST_SKIP() << "needs at least 2 ranks";
    KCH_SKIP_IF_NEEDED(createCommWithRecorder());
    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    ASSERT_NO_FATAL_FAILURE(allocBuffers(1 << 18));
    const int rank = MPIEnvironment::world_rank, n = MPIEnvironment::world_size;

    constexpr int kIters = 8;
    for(int i = 0; i < kIters; i++)
    {
        EXPECT_EQ(ncclSuccess, ncclGroupStart());
        EXPECT_EQ(ncclSuccess, ncclSend(send_, count_, ncclFloat, (rank + 1) % n, comm, stream));
        EXPECT_EQ(ncclSuccess, ncclRecv(recv_, count_, ncclFloat, (rank + n - 1) % n, comm, stream));
        EXPECT_EQ(ncclSuccess, ncclGroupEnd());
    }
    EXPECT_EQ(hipSuccess, hipStreamSynchronize(stream));

    auto recs = waitForDrain(2 * kIters);
    KernelChStats s = checkBalanced(recs, "send/recv ring");
    EXPECT_EQ((size_t)(2 * kIters), s.tasks);
    freeBuffers();
}

// Graph replays re-run the host node that posts KernelCh work, so each replay is
// timed on its own. Before the move, ops were posted once at capture.
TEST_F(ProfilerKernelChMPITest, GraphReplaysAreEachTimed)
{
    KCH_REQUIRE_RECORDER();
    if(!validateTestPrerequisites(2)) GTEST_SKIP() << "needs at least 2 ranks";
    KCH_SKIP_IF_NEEDED(createCommWithRecorder());
    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    ASSERT_NO_FATAL_FAILURE(allocBuffers(1 << 20));

    constexpr int kPerGraph = 4, kReplays = 6;
    hipGraph_t graph = nullptr;
    hipGraphExec_t exec = nullptr;
    ASSERT_EQ(hipSuccess, hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
    for(int i = 0; i < kPerGraph; i++) EXPECT_EQ(ncclSuccess, allReduce(comm, stream));
    ASSERT_EQ(hipSuccess, hipStreamEndCapture(stream, &graph));
    ASSERT_EQ(hipSuccess, hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for(int i = 0; i < kReplays; i++) EXPECT_EQ(hipSuccess, hipGraphLaunch(exec, stream));
    EXPECT_EQ(hipSuccess, hipStreamSynchronize(stream));

    auto recs = waitForDrain(kPerGraph * kReplays);
    KernelChStats s = checkBalanced(recs, "graph replays");
    EXPECT_EQ((size_t)(kPerGraph * kReplays), s.tasks) << "expected one Coll per collective per replay";
    checkPerChannelOrder(recs, "graph replays");
    // Each replay is a new collective, and RAS and plugins tell them apart by seqNumber.
    std::vector<uint64_t> seqs;
    for(const auto& r : recs)
        if(r.type == ncclProfileColl) seqs.push_back(r.seqNumber);
    EXPECT_TRUE(std::adjacent_find(seqs.begin(), seqs.end(), std::greater_equal<uint64_t>()) == seqs.end())
        << "replayed collectives do not get increasing seqNumbers";

    EXPECT_EQ(hipSuccess, hipGraphExecDestroy(exec));
    EXPECT_EQ(hipSuccess, hipGraphDestroy(graph));
    freeBuffers();
}

// Eager plans and graph replays share one host workCounter per channel, so the
// host must post their work in the order the device runs it or timings land on
// the wrong collective. That is only promised with graph mixing support
// (graphUsageMode=2): without it, an eager collective issued while a graph launch
// is outstanding is unsupported. Holding the stream ahead of every replay delays
// the replay's host node past the eager call that follows, so a regression shows
// up on every run rather than by chance; the test checks that the hold did.
TEST_F(ProfilerKernelChMPITest, MixedEagerAndGraphPostInDeviceOrder)
{
    KCH_REQUIRE_RECORDER();
    if(!validateTestPrerequisites(2)) GTEST_SKIP() << "needs at least 2 ranks";
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.graphUsageMode = 2;
    ncclComm_t comm = nullptr;
    hipStream_t stream = nullptr;
    KCH_SKIP_IF_NEEDED(createConfiguredCommWithRecorder(&config, &comm, &stream));

    // Distinct sizes tell eager collectives and replayed ones apart in the record.
    constexpr size_t kEagerCount = 1 << 12, kGraphCount = 1 << 14;
    ASSERT_NO_FATAL_FAILURE(allocBuffers(kGraphCount));
    constexpr int kPerGraph = 2, kRounds = 8;
    hipGraph_t graph = nullptr;
    hipGraphExec_t exec = nullptr;
    ASSERT_EQ(hipSuccess, hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
    for(int i = 0; i < kPerGraph; i++)
        EXPECT_EQ(ncclSuccess, ncclAllReduce(send_, recv_, kGraphCount, ncclFloat, ncclSum, comm, stream));
    ASSERT_EQ(hipSuccess, hipStreamEndCapture(stream, &graph));
    ASSERT_EQ(hipSuccess, hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    std::atomic<int> holdsDone{0};
    auto holdStream = [](void* done) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        static_cast<std::atomic<int>*>(done)->fetch_add(1);
    };
    std::vector<size_t> expected;
    int heldBack = 0;  // rounds whose eager call returned while the replay was still held
    for(int i = 0; i < kRounds; i++)
    {
        EXPECT_EQ(hipSuccess, hipLaunchHostFunc(stream, holdStream, &holdsDone));
        EXPECT_EQ(hipSuccess, hipGraphLaunch(exec, stream));
        EXPECT_EQ(ncclSuccess, ncclAllReduce(send_, recv_, kEagerCount, ncclFloat, ncclSum, comm, stream));
        if(holdsDone.load() <= i) ++heldBack;
        expected.insert(expected.end(), kPerGraph, kGraphCount);
        expected.push_back(kEagerCount);
    }
    EXPECT_EQ(hipSuccess, hipStreamSynchronize(stream));
    // Otherwise the replay's host node ran first and the posting order below holds
    // whatever the core does.
    EXPECT_GT(heldBack, kRounds / 2) << "only " << heldBack << " of " << kRounds
                                     << " eager calls overtook the held replay";

    auto recs = waitForDrain(expected.size());
    KernelChStats s = checkBalanced(recs, "mixed eager/graph");
    EXPECT_EQ(expected.size(), s.tasks);
    checkPerChannelOrder(recs, "mixed eager/graph");

    std::vector<size_t> posted;
    for(const auto& r : recs)
        if(isTask(r)) posted.push_back(r.count);
    auto mismatch = std::mismatch(expected.begin(), expected.end(), posted.begin(), posted.end());
    EXPECT_TRUE(mismatch.first == expected.end() && mismatch.second == posted.end())
        << "host posted work out of stream order: collective #" << (mismatch.first - expected.begin())
        << " has count " << (mismatch.second == posted.end() ? 0 : *mismatch.second) << ", expected "
        << (mismatch.first == expected.end() ? 0 : *mismatch.first);

    EXPECT_EQ(hipSuccess, hipGraphExecDestroy(exec));
    EXPECT_EQ(hipSuccess, hipGraphDestroy(graph));
    freeBuffers();
}

// A burst of small collectives with no synchronization, well past the 64 slots
// the device keeps per channel. The core accepts that a thread that falls that
// far behind may time an op from a reused slot, so timeline order is not
// guaranteed here; every op must still be delivered exactly once, with nothing
// left pending or stopped twice.
TEST_F(ProfilerKernelChMPITest, BurstPastSlotRingIsBalanced)
{
    KCH_REQUIRE_RECORDER();
    if(!validateTestPrerequisites(2)) GTEST_SKIP() << "needs at least 2 ranks";
    KCH_SKIP_IF_NEEDED(createCommWithRecorder());
    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    ASSERT_NO_FATAL_FAILURE(allocBuffers(1024));

    constexpr int kIters = 512;
    for(int i = 0; i < kIters; i++) EXPECT_EQ(ncclSuccess, allReduce(comm, stream));
    EXPECT_EQ(hipSuccess, hipStreamSynchronize(stream));

    auto recs = waitForDrain(kIters);
    KernelChStats s = checkBalanced(recs, "burst");
    EXPECT_EQ((size_t)kIters, s.tasks);
    freeBuffers();
}

// The thread is per communicator, but a split child with shareResources reuses
// its parent's; a child without it gets its own.
class ProfilerKernelChSplitMPITest : public ProfilerKernelChMPITest,
                                     public ::testing::WithParamInterface<int>
{};

TEST_P(ProfilerKernelChSplitMPITest, SplitThreadOwnership)
{
    const int share = GetParam();
    KCH_REQUIRE_RECORDER();
    if(!validateTestPrerequisites(2)) GTEST_SKIP() << "needs at least 2 ranks";

    // ncclCommSplit takes splitShare from the parent's config, not the child's.
    ncclConfig_t parentConfig = NCCL_CONFIG_INITIALIZER;
    parentConfig.splitShare = share;
    ncclComm_t parent = nullptr;
    hipStream_t stream = nullptr;
    KCH_SKIP_IF_NEEDED(createConfiguredCommWithRecorder(&parentConfig, &parent, &stream));
    ASSERT_NO_FATAL_FAILURE(allocBuffers(1 << 18));

    ncclComm_t child = nullptr;
    ASSERT_EQ(ncclSuccess, ncclCommSplit(parent, 0, MPIEnvironment::world_rank, &child, nullptr));
    ownedComms_.push_back(child);

    constexpr int kIters = 4;
    for(int i = 0; i < kIters; i++)
    {
        EXPECT_EQ(ncclSuccess, allReduce(parent, stream));
        EXPECT_EQ(ncclSuccess, allReduce(child, stream));
    }
    EXPECT_EQ(hipSuccess, hipStreamSynchronize(stream));
    auto recs = waitForDrain(2 * kIters);
    KernelChStats both = checkBalanced(recs, "parent and child", share ? 1 : 2);
    EXPECT_EQ((size_t)(2 * kIters), both.tasks);

    std::map<uint64_t, std::set<long>> tidsByComm;
    for(const auto& r : recs)
        if(r.type == ncclProfileKernelCh) tidsByComm[r.commId].insert(r.startTid);
    EXPECT_EQ(2u, tidsByComm.size()) << "expected KernelCh from parent and child";
    std::set<long> all;
    for(auto& [comm, tids] : tidsByComm)
    {
        EXPECT_EQ(1u, tids.size()) << "one communicator's KernelCh arrived on several threads";
        all.insert(tids.begin(), tids.end());
    }
    if(share)
        EXPECT_EQ(1u, all.size()) << "splitShare=1 child did not reuse its parent's profiler thread";
    else
        EXPECT_EQ(2u, all.size()) << "splitShare=0 child shares its parent's profiler thread";

    // The child's teardown must not take the parent's thread with it when shared.
    // The parent is still alive, so judge only the records made from here on
    // rather than resetting the recorder under it.
    EXPECT_EQ(ncclSuccess, destroyOwnedComm(child));
    const size_t before = snapshot().size();
    for(int i = 0; i < kIters; i++) EXPECT_EQ(ncclSuccess, allReduce(parent, stream));
    EXPECT_EQ(hipSuccess, hipStreamSynchronize(stream));
    auto after = waitForDrain(2 * kIters + kIters);
    std::vector<RcclKchRecord> tail;
    for(size_t i = before; i < after.size(); i++)
    {
        RcclKchRecord r = after[i];
        r.parentIndex = (r.parentIndex >= (int64_t)before) ? r.parentIndex - (int64_t)before : -1;
        tail.push_back(r);
    }
    KernelChStats s = checkBalanced(tail, "parent after child destroyed");
    EXPECT_EQ((size_t)kIters, s.tasks);
    freeBuffers();
}

INSTANTIATE_TEST_SUITE_P(Share, ProfilerKernelChSplitMPITest, ::testing::Values(0, 1),
                         [](const ::testing::TestParamInfo<int>& i) {
                             return i.param ? std::string("Shared") : std::string("Separate");
                         });

// Aborting with work in flight must let the thread drain and exit: abort joins
// it, so a thread stuck waiting on counters that will never advance hangs abort.
TEST_F(ProfilerKernelChMPITest, AbortWithWorkInFlightReturns)
{
    KCH_REQUIRE_RECORDER();
    if(!validateTestPrerequisites(2)) GTEST_SKIP() << "needs at least 2 ranks";
    KCH_SKIP_IF_NEEDED(createCommWithRecorder());
    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    ASSERT_NO_FATAL_FAILURE(allocBuffers(1 << 22));

    for(int i = 0; i < 64; i++) EXPECT_EQ(ncclSuccess, allReduce(comm, stream));
    // Otherwise only an idle teardown is exercised.
    EXPECT_EQ(hipErrorNotReady, hipStreamQuery(stream)) << "the work drained before ncclCommAbort was called";
    auto t0 = std::chrono::steady_clock::now();
    EXPECT_EQ(ncclSuccess, ncclCommAbort(comm));
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    test_comm_ = nullptr;  // aborted; cleanup must not destroy it again
    EXPECT_LT(secs, 60.0) << "ncclCommAbort took " << secs << " s with KernelCh work in flight";
    EXPECT_EQ(0u, anomalies_()) << "recorder saw unbalanced calls around abort";
    (void)hipStreamSynchronize(stream);
    freeBuffers();
}

// KernelCh timers are raw device ticks. On AMD that is wall_clock64(), whose rate
// the runtime reports. Converted at that rate, the KernelCh span must fit inside
// a pair of stream events bracketing the same work, and fill most of it, so a
// wrong unit fails in either direction.
TEST_F(ProfilerKernelChMPITest, TimersAreWallClockTicks)
{
    KCH_REQUIRE_RECORDER();
    if(!validateTestPrerequisites(2)) GTEST_SKIP() << "needs at least 2 ranks";
    KCH_SKIP_IF_NEEDED(createCommWithRecorder());
    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    ASSERT_NO_FATAL_FAILURE(allocBuffers(1 << 22));

    int dev = 0, rateKHz = 0;
    ASSERT_EQ(hipSuccess, hipGetDevice(&dev));
    ASSERT_EQ(hipSuccess, hipDeviceGetAttribute(&rateKHz, hipDeviceAttributeWallClockRate, dev));
    ASSERT_GT(rateKHz, 0);
    const double nsPerTick = 1e6 / rateKHz;

    hipEvent_t begin = nullptr, end = nullptr;
    ASSERT_EQ(hipSuccess, hipEventCreate(&begin));
    ASSERT_EQ(hipSuccess, hipEventCreate(&end));
    EXPECT_EQ(hipSuccess, hipStreamSynchronize(stream));
    constexpr int kIters = 8;
    EXPECT_EQ(hipSuccess, hipEventRecord(begin, stream));
    for(int i = 0; i < kIters; i++) EXPECT_EQ(ncclSuccess, allReduce(comm, stream));
    EXPECT_EQ(hipSuccess, hipEventRecord(end, stream));
    EXPECT_EQ(hipSuccess, hipStreamSynchronize(stream));
    float eventMs = 0;
    EXPECT_EQ(hipSuccess, hipEventElapsedTime(&eventMs, begin, end));
    const double eventNs = eventMs * 1e6;
    (void)hipEventDestroy(begin);
    (void)hipEventDestroy(end);

    auto recs = waitForDrain(kIters);
    KernelChStats s = checkBalanced(recs, "timer units");
    EXPECT_EQ((size_t)kIters, s.tasks);
    uint64_t lo = UINT64_MAX, hi = 0;
    for(const auto& r : recs)
    {
        if(r.type != ncclProfileKernelCh) continue;
        lo = std::min(lo, r.startTimer);
        hi = std::max(hi, r.stopTimer);
    }
    ASSERT_GT(hi, lo);
    // Event timestamps and wall_clock64() are separate clocks, hence the slack.
    const double spanNs = (double)(hi - lo) * nsPerTick;
    EXPECT_LE(spanNs, eventNs * 1.05 + 20e3) << "KernelCh span " << spanNs << " ns at " << rateKHz
                                             << " kHz exceeds the " << eventNs << " ns between stream events";
    EXPECT_GE(spanNs, eventNs * 0.5) << "KernelCh span " << spanNs << " ns at " << rateKHz
                                     << " kHz covers under half of the " << eventNs << " ns between stream events";
    freeBuffers();
}

#endif  // MPI_TESTS_ENABLED
