/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"
#include <gtest/gtest-spi.h>
#include <hip/hip_runtime.h>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>

namespace RcclUnitTesting
{
  // Stress coverage for the parallelized TestBed::DestroyComms teardown
  // (ROCM-25953): DestroyComms now broadcasts the destroy command to every child
  // process first and collects the acknowledgements in a second pass, so the
  // children tear their communicators down concurrently instead of one at a time.
  // These tests drive that path repeatedly, in multi-process mode (one child per
  // GPU) where the two-pass ordering actually matters, and in both blocking and
  // non-blocking modes (the child-side DestroyComms branches on useBlocking).
  namespace
  {
    // Returns true if ncclFloat32 is available under the current UT_DATATYPES.
    // Checked once in each test body so the test can GTEST_SKIP() explicitly
    // rather than silently passing as a no-op when the datatype is excluded.
    bool float32Supported(TestBed& testBed)
    {
      std::vector<ncclDataType_t> dataTypes;
      testBed.GetSupportedDataTypes(dataTypes, {ncclFloat32});
      return !dataTypes.empty();
    }

    // Why this run cannot host a multi-process ncclFloat32 test on >= 2 GPUs
    // (optionally pooled, with warm-ahead), or empty if it can.
    std::string MultiProcessSkipReason(TestBed& testBed, bool const needPool = false,
                                       bool const needWarmAhead = false)
    {
      if (needPool && !testBed.poolMode) return "Requires communicator pooling (UT_COMM_POOL=1)";
      if (needWarmAhead && !testBed.ev.commWarmAhead) return "Requires UT_COMM_WARMAHEAD=1";
      if (testBed.ev.maxGpus < 2)
      {
        return "Teardown stress requires at least 2 GPUs (detected " + std::to_string(testBed.ev.maxGpus) + ")";
      }
      if (!(testBed.ev.processMask & (1 << 1))) return "Teardown stress requires multi-process mode (UT_PROCESS_MASK)";
      if (!float32Supported(testBed)) return "Teardown stress requires ncclFloat32 (excluded by UT_DATATYPES)";
      return "";
    }

    // Runs body with this thread's gtest failures redirected into failures; returns its duration.
    template <typename Body>
    std::chrono::steady_clock::duration CaptureFailures(::testing::TestPartResultArray& failures, Body&& body)
    {
      auto const start = std::chrono::steady_clock::now();
      ::testing::ScopedFakeTestPartResultReporter reporter(
        ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ONLY_CURRENT_THREAD, &failures);
      body();
      return std::chrono::steady_clock::now() - start;
    }

    // Run `iterations` init / collective / destroy cycles over `totalRanks`
    // ranks, one child process per rank. Each cycle ends in DestroyComms, the
    // path under test. A leaked pipe fd or child handle, or any mismatch between
    // the broadcast pass and the ack-collection pass, shows up as a hang or a
    // failure once enough cycles accumulate.
    void RunTeardownCycles(TestBed& testBed,
                           int  const totalRanks,
                           bool const useBlocking,
                           int  const iterations,
                           bool&      isCorrect)
    {
      size_t const numElements   = 32 * 1024;
      bool   const inPlace       = false;
      bool   const useManagedMem = false;
      int    const numProcesses  = totalRanks;  // one child process per rank
      const std::vector<int>& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();

      for (int iter = 0; iter < iterations && isCorrect; ++iter)
      {
        testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks,
                                                    gpuPriorityOrder),
                          1, 1, 1, useBlocking);

        OptionalColArgs options;
        options.redOp = ncclSum;
        testBed.SetCollectiveArgs(ncclCollAllReduce, ncclFloat32,
                                  numElements, numElements, options);
        testBed.AllocateMem(inPlace, useManagedMem);
        testBed.PrepareData();
        testBed.ExecuteCollectives();
        testBed.ValidateResults(isCorrect);
        testBed.DeallocateMem();
        testBed.DestroyComms();
      }
    }
  }

  // Repeated multi-process teardown in both blocking and non-blocking modes.
  TEST(Teardown, RepeatedDestroyComms)
  {
    TestBed testBed;
    std::string const skipReason = MultiProcessSkipReason(testBed);
    if (!skipReason.empty()) GTEST_SKIP() << skipReason;

    bool isCorrect = true;
    for (bool useBlocking : {true, false})
      RunTeardownCycles(testBed, testBed.ev.maxGpus, useBlocking,
                        /*iterations*/ 3, isCorrect);
    EXPECT_TRUE(isCorrect);
    testBed.Finalize();
  }

  // Teardown across a varying number of child processes within one test, so
  // DestroyComms is exercised with different numActiveChildren values back to
  // back. Catches assumptions that only hold for a fixed child count.
  TEST(Teardown, DestroyCommsVaryingChildCount)
  {
    TestBed testBed;
    std::string const skipReason = MultiProcessSkipReason(testBed);
    if (!skipReason.empty()) GTEST_SKIP() << skipReason;

    bool isCorrect = true;
    for (int ranks = testBed.ev.maxGpus; ranks >= 2 && isCorrect; ranks /= 2)
      RunTeardownCycles(testBed, ranks, /*useBlocking*/ true,
                        /*iterations*/ 1, isCorrect);
    EXPECT_TRUE(isCorrect);
    testBed.Finalize();
  }

  // A worker can exit before teardown after a HIP/RCCL failure. Its closed
  // command pipe must not prevent the parent from reaping it and resetting the
  // TestBed state.
  TEST(Teardown, AlreadyExitedChildCleanup)
  {
    TestBed testBed;
    testBed.poolMode = false;
    testBed.configUsedPool = false;

    TestBedChild* child = new TestBedChild(0, false, 0, false);
    ASSERT_EQ(child->InitPipes(), TEST_SUCCESS);
    child->pid = fork();
    ASSERT_GE(child->pid, 0);
    if (child->pid == 0)
    {
      close(child->parentWriteFd);
      close(child->parentReadFd);
      close(child->childWriteFd);
      close(child->childReadFd);
      _exit(0);
    }

    close(child->childWriteFd);
    close(child->childReadFd);
    child->childWriteFd = -1;
    child->childReadFd = -1;
    testBed.childList = {child};
    testBed.numActiveChildren = 1;
    testBed.numActiveRanks = 1;

    // Wait until the worker has exited, but leave it for TestBed to reap.
    siginfo_t childInfo{};
    ASSERT_EQ(waitid(P_PID, child->pid, &childInfo, WEXITED | WNOWAIT), 0);
    pid_t const childPid = child->pid;
    // Teardown deletes the child, so snapshot the parent-side descriptors first.
    int const parentWriteFd = child->parentWriteFd;
    int const parentReadFd = child->parentReadFd;

    testBed.TeardownOwnedChildList();

    EXPECT_TRUE(testBed.childList.empty());
    EXPECT_EQ(testBed.numActiveChildren, 0);
    EXPECT_EQ(testBed.numActiveRanks, 0);
    errno = 0;
    EXPECT_EQ(waitpid(childPid, nullptr, WNOHANG), -1);
    EXPECT_EQ(errno, ECHILD);
    for (int const fd : {parentWriteFd, parentReadFd})
    {
      errno = 0;
      EXPECT_EQ(fcntl(fd, F_GETFD), -1) << "parent pipe fd " << fd << " left open";
      EXPECT_EQ(errno, EBADF);
    }
  }

  namespace
  {
    void LimitParentWaitsToOneSecond(TestBed& testBed)
    {
      testBed.ev.debugPause = false;
      testBed.ev.timeoutUs = 0;
      testBed.ev.childTimeoutSec = 1;
    }

    // Forks a worker that keeps its pipes open but never answers, as a worker
    // wedged in a collective would.
    TestBedChild* ForkHungChild(int const childId)
    {
      TestBedChild* child = new TestBedChild(childId, false, 0, false);
      if (child->InitPipes() != TEST_SUCCESS)
      {
        child->ClosePipes();
        delete child;
        return nullptr;
      }
      child->pid = fork();
      if (child->pid == 0)
      {
        close(child->parentWriteFd);
        close(child->parentReadFd);
        while (true) pause();
      }
      if (child->pid < 0)
      {
        child->ClosePipes();
        delete child;
        return nullptr;
      }
      close(child->childWriteFd);
      close(child->childReadFd);
      child->childWriteFd = -1;
      child->childReadFd = -1;
      return child;
    }

    // Makes a hung worker testBed's only (non-pool) child, with 1 s parent waits.
    TestBedChild* InstallHungChild(TestBed& testBed)
    {
      testBed.poolMode = false;
      testBed.configUsedPool = false;
      LimitParentWaitsToOneSecond(testBed);
      TestBedChild* const child = ForkHungChild(0);
      if (child == nullptr) return nullptr;
      testBed.childList = {child};
      testBed.numActiveChildren = 1;
      testBed.numActiveRanks = 1;
      return child;
    }

    bool AnyFailureMentions(::testing::TestPartResultArray const& failures, char const* text)
    {
      for (int i = 0; i < failures.size(); ++i)
      {
        if (std::string(failures.GetTestPartResult(i).message()).find(text) != std::string::npos) return true;
      }
      return false;
    }
  }

  // A worker that never replies must fail the current test once
  // UT_CHILD_TIMEOUT_S expires, and be killed and reaped, instead of hanging
  // the whole binary.
  TEST(Teardown, SilentChildFailsAfterReplyTimeout)
  {
    TestBed testBed;
    TestBedChild* const child = InstallHungChild(testBed);
    ASSERT_NE(child, nullptr);
    pid_t const childPid = child->pid;

    ::testing::TestPartResultArray failures;
    auto const elapsed = CaptureFailures(failures, [&]() { testBed.DestroyComms(); });

    EXPECT_EQ(failures.size(), 1);
    EXPECT_TRUE(AnyFailureMentions(failures, "stopped responding"));
    EXPECT_LT(elapsed, std::chrono::seconds(30));
    EXPECT_TRUE(testBed.childList.empty());
    EXPECT_EQ(testBed.numActiveChildren, 0);
    EXPECT_FALSE(testBed.childFailed);
    errno = 0;
    EXPECT_EQ(waitpid(childPid, nullptr, WNOHANG), -1);
    EXPECT_EQ(errno, ECHILD);
  }

  // A worker that ignores CHILD_STOP (e.g. stuck in comm teardown) must be
  // killed at the teardown deadline rather than waited on forever.
  TEST(Teardown, HungChildKilledAtTeardownDeadline)
  {
    TestBed testBed;
    TestBedChild* const child = InstallHungChild(testBed);
    ASSERT_NE(child, nullptr);
    pid_t const childPid = child->pid;

    ::testing::TestPartResultArray failures;
    auto const elapsed = CaptureFailures(failures, [&]() { testBed.TeardownOwnedChildList(); });

    EXPECT_EQ(failures.size(), 1);
    EXPECT_TRUE(AnyFailureMentions(failures, "teardown deadline"));
    EXPECT_LT(elapsed, std::chrono::seconds(30));
    EXPECT_TRUE(testBed.childList.empty());
    EXPECT_EQ(testBed.numActiveChildren, 0);
    errno = 0;
    EXPECT_EQ(waitpid(childPid, nullptr, WNOHANG), -1);
    EXPECT_EQ(errno, ECHILD);
  }

  // Pool workers must start from a fresh process image even when an earlier
  // test initialized HIP in the parent process.
  TEST(Teardown, PoolAfterParentHipInitialization)
  {
    TestBed testBed;
    std::string const skipReason = MultiProcessSkipReason(testBed, /*needPool=*/true);
    if (!skipReason.empty()) GTEST_SKIP() << skipReason;

    // Initialize HIP only once the test will run, so a skipped run leaves the
    // parent HIP-clean for later tests in this binary.
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    void* parentAllocation = nullptr;
    ASSERT_EQ(hipMalloc(&parentAllocation, 1), hipSuccess);
    ASSERT_EQ(hipFree(parentAllocation), hipSuccess);

    bool isCorrect = true;
    RunTeardownCycles(testBed, testBed.ev.maxGpus, /*useBlocking*/ true,
                      /*iterations*/ 1, isCorrect);
    EXPECT_TRUE(isCorrect);
    // InitComms silently falls back to fork-fresh workers if the pool cannot
    // serve the config, which would make this test cover the wrong path.
    EXPECT_TRUE(testBed.configUsedPool) << "Config was not served from the communicator pool";
    testBed.Finalize();
  }

  namespace
  {
    // Creates the pool with a 1-process config, then starts warming the second
    // worker of a 2-process config. Returns that worker, or -1 on failure.
    int StartWarmingSecondWorker(TestBed& testBed, std::vector<std::vector<int>>& twoProcessIds)
    {
      bool isCorrect = true;
      RunTeardownCycles(testBed, 1, /*useBlocking*/ true, /*iterations*/ 1, isCorrect);
      if (!isCorrect || testBed.poolChildren.empty()) return -1;

      twoProcessIds = TestBed::GetDeviceIdsList(2, 2, testBed.ev.GetGpuPriorityOrder());
      int const worker = twoProcessIds[1][0];
      EXPECT_EQ(testBed.poolWorkerState[worker], TestBed::PoolWorkerState::Cold);
      testBed.WarmPoolWorkers(twoProcessIds);
      EXPECT_EQ(testBed.poolWorkerState[worker], TestBed::PoolWorkerState::Warming);
      return worker;
    }

    // Every reply is a TEST_SUCCESS int, so a reply the parent never read just
    // shifts later acks by one; only the leftover bytes in the pipe show it.
    bool HasUnreadReply(TestBedChild const* worker, int const waitMs)
    {
      pollfd fd = {worker->parentReadFd, POLLIN, 0};
      return poll(&fd, 1, waitMs) > 0 && (fd.revents & POLLIN);
    }

    // Makes only rank 1 expect a wrong first element, so only that rank fails ValidateResults.
    ErrCode WrongExpectedOnRank1(CollectiveArgs& collArgs)
    {
      ErrCode const status = DefaultPrepData_Broadcast(collArgs);
      if (status == TEST_SUCCESS && collArgs.globalRank == 1)
      {
        memset(collArgs.expected.ptr, 0xFF, DataTypeToBytes(collArgs.dataType));  // NaN for float32
      }
      return status;
    }
  }

  // Rank 0 gets an out-of-range root, so its ncclBroadcast fails at once, while
  // rank 1 runs a valid Broadcast that waits on rank 0. The parent must stop the
  // workers on rank 0's failure, not wait for rank 1's own UT_TIMEOUT_US.
  TEST(Teardown, FailedCollectiveStopsBlockedPeers)
  {
    TestBed testBed;
    std::string const skipReason = MultiProcessSkipReason(testBed);
    if (!skipReason.empty()) GTEST_SKIP() << skipReason;
    testBed.ev.childTimeoutSec = 60;
    testBed.ev.timeoutUs = 5000000;

    testBed.InitComms(TestBed::GetDeviceIdsList(2, 2, testBed.ev.GetGpuPriorityOrder()));
    ASSERT_FALSE(testing::Test::HasFailure());
    size_t const numElements = 1024 * 1024;
    OptionalColArgs options;
    options.root = 0;
    testBed.SetCollectiveArgs(ncclCollBroadcast, ncclFloat32, numElements, numElements, options,
                              /*collId*/ -1, /*groupId*/ 0, /*rank*/ 1);
    options.root = 7;
    testBed.SetCollectiveArgs(ncclCollBroadcast, ncclFloat32, numElements, numElements, options,
                              /*collId*/ -1, /*groupId*/ 0, /*rank*/ 0);
    testBed.AllocateMem(/*inPlace*/ false, /*useManagedMem*/ false);
    ASSERT_FALSE(testing::Test::HasFailure());

    ::testing::TestPartResultArray failures;
    auto const elapsed = CaptureFailures(failures, [&]() { testBed.ExecuteCollectives(); });

    EXPECT_EQ(failures.size(), 1);
    EXPECT_TRUE(AnyFailureMentions(failures, "Child 0 reported failure"));
    EXPECT_LT(elapsed, std::chrono::microseconds(testBed.ev.timeoutUs))
      << "Waited for rank 1 to time out instead of stopping the workers";
    EXPECT_TRUE(testBed.childList.empty());
    EXPECT_EQ(testBed.numActiveChildren, 0);
    testBed.Finalize();
  }

  // Child 1 fails InitComms before creating its comm (its GPU does not exist),
  // while child 0 waits inside ncclCommInitRank for it to join, which nothing on
  // the child side can end. The parent must stop the workers at once rather
  // than wait out the reply timeout.
  TEST(Teardown, FailedInitStopsWaitingPeers)
  {
    TestBed testBed;
    std::string const skipReason = MultiProcessSkipReason(testBed);
    if (!skipReason.empty()) GTEST_SKIP() << skipReason;
    int const replyTimeoutSec = 60;
    testBed.ev.childTimeoutSec = replyTimeoutSec;
    int const firstGpu = testBed.ev.GetGpuPriorityOrder()[0];

    ::testing::TestPartResultArray failures;
    auto const elapsed = CaptureFailures(failures, [&]() { testBed.InitComms({{firstGpu}, {9999}}); });

    EXPECT_EQ(failures.size(), 1);
    EXPECT_TRUE(AnyFailureMentions(failures, "Child 1 reported failure"));
    EXPECT_LT(elapsed, std::chrono::seconds(10));
    EXPECT_TRUE(testBed.childList.empty());
    EXPECT_EQ(testBed.numActiveChildren, 0);
    testBed.Finalize();
  }

  // The next InitComms that uses a warming worker collects its reply first.
  TEST(Teardown, WarmedPoolWorkerIsReused)
  {
    TestBed testBed;
    std::string const skipReason = MultiProcessSkipReason(testBed, /*needPool=*/true, /*needWarmAhead=*/true);
    if (!skipReason.empty()) GTEST_SKIP() << skipReason;

    std::vector<std::vector<int>> twoProcessIds;
    int const worker = StartWarmingSecondWorker(testBed, twoProcessIds);
    ASSERT_GE(worker, 0);

    bool isCorrect = true;
    RunTeardownCycles(testBed, 2, /*useBlocking*/ true, /*iterations*/ 1, isCorrect);
    EXPECT_TRUE(isCorrect);
    EXPECT_TRUE(testBed.configUsedPool) << "Config was not served from the communicator pool";
    EXPECT_EQ(testBed.poolWorkerState[worker], TestBed::PoolWorkerState::Warm);
    for (auto const& deviceIds : twoProcessIds)
    {
      EXPECT_FALSE(HasUnreadReply(testBed.poolChildren[deviceIds[0]], /*waitMs*/ 1000))
        << "Pool worker " << deviceIds[0] << " has a reply the parent never read";
    }
    testBed.Finalize();
  }

  // Finalize collects a pending warm-up reply, so a worker that died while
  // warming fails the test instead of being reaped silently.
  TEST(Teardown, FinalizeReportsLostWarmupWorker)
  {
    TestBed testBed;
    LimitParentWaitsToOneSecond(testBed);
    TestBedChild* const child = ForkHungChild(0);
    ASSERT_NE(child, nullptr);
    pid_t const childPid = child->pid;
    testBed.poolChildren = {child};
    testBed.poolWorkerState = {TestBed::PoolWorkerState::Warming};
    ASSERT_EQ(kill(childPid, SIGKILL), 0);

    ::testing::TestPartResultArray failures;
    CaptureFailures(failures, [&]() { testBed.Finalize(); });

    EXPECT_EQ(failures.size(), 1);
    EXPECT_TRUE(AnyFailureMentions(failures, "warm-up pipe closed"));
    EXPECT_TRUE(testBed.poolChildren.empty());
    EXPECT_TRUE(testBed.poolWorkerState.empty());
    errno = 0;
    EXPECT_EQ(waitpid(childPid, nullptr, WNOHANG), -1);
    EXPECT_EQ(errno, ECHILD);
  }

  // Warm-ups run in parallel, so hung warming workers share one reply deadline
  // instead of each waiting out its own UT_CHILD_TIMEOUT_S.
  TEST(Teardown, HungWarmupsShareOneDeadline)
  {
    TestBed testBed;
    LimitParentWaitsToOneSecond(testBed);
    int const numWorkers = 3;
    for (int worker = 0; worker < numWorkers; ++worker)
    {
      TestBedChild* const child = ForkHungChild(worker);
      ASSERT_NE(child, nullptr);
      testBed.poolChildren.push_back(child);
      testBed.poolWorkerState.push_back(TestBed::PoolWorkerState::Warming);
    }

    ::testing::TestPartResultArray failures;
    auto const elapsed = CaptureFailures(failures, [&]() { testBed.Finalize(); });

    EXPECT_EQ(failures.size(), numWorkers);
    EXPECT_TRUE(AnyFailureMentions(failures, "warm-up did not reply"));
    EXPECT_LT(elapsed, std::chrono::seconds(numWorkers - 1)) << "Each warm-up waited out its own reply timeout";
    EXPECT_TRUE(testBed.poolChildren.empty());
  }

  // When one rank fails ValidateResults it aborts its comms, so DestroyComms must
  // abort its healthy peers too: finalizing would leave them waiting on that rank.
  TEST(Teardown, DestroyCommsAbortsPeersAfterValidationFailure)
  {
    TestBed testBed;
    std::string const skipReason = MultiProcessSkipReason(testBed);
    if (!skipReason.empty()) GTEST_SKIP() << skipReason;
    testBed.ev.childTimeoutSec = 20;

    testBed.InitComms(TestBed::GetDeviceIdsList(2, 2, testBed.ev.GetGpuPriorityOrder()));
    ASSERT_FALSE(testing::Test::HasFailure());
    OptionalColArgs options;
    options.root = 0;
    testBed.SetCollectiveArgs(ncclCollBroadcast, ncclFloat32, 4096, 4096, options);
    testBed.AllocateMem(/*inPlace*/ false, /*useManagedMem*/ false);
    testBed.PrepareData(-1, -1, -1, WrongExpectedOnRank1);
    testBed.ExecuteCollectives();
    ASSERT_FALSE(testing::Test::HasFailure());

    bool isCorrect = true;
    ::testing::TestPartResultArray validateFailures;
    CaptureFailures(validateFailures, [&]() { testBed.ValidateResults(isCorrect); });
    ASSERT_FALSE(isCorrect) << "Rank 1 did not detect its corrupted expected value";
    ASSERT_TRUE(testBed.childFailed);
    testBed.DeallocateMem();
    ASSERT_FALSE(testing::Test::HasFailure());

    ::testing::TestPartResultArray failures;
    auto const elapsed = CaptureFailures(failures, [&]() { testBed.DestroyComms(); });

    EXPECT_EQ(failures.size(), 0) << (failures.size() > 0 ? failures.GetTestPartResult(0).message() : "");
    EXPECT_LT(elapsed, std::chrono::seconds(testBed.ev.childTimeoutSec));
    EXPECT_FALSE(testBed.childFailed);
    testBed.Finalize();
  }

  // A worker lost during warm-up fails the test, leaves the pool, and the
  // config still runs on freshly forked workers.
  TEST(Teardown, LostWarmupWorkerFallsBackToFreshWorkers)
  {
    TestBed testBed;
    std::string const skipReason = MultiProcessSkipReason(testBed, /*needPool=*/true, /*needWarmAhead=*/true);
    if (!skipReason.empty()) GTEST_SKIP() << skipReason;

    std::vector<std::vector<int>> twoProcessIds;
    int const worker = StartWarmingSecondWorker(testBed, twoProcessIds);
    ASSERT_GE(worker, 0);
    pid_t const workerPid = testBed.poolChildren[worker]->pid;
    ASSERT_EQ(kill(workerPid, SIGKILL), 0);

    ::testing::TestPartResultArray failures;
    CaptureFailures(failures, [&]() { testBed.InitComms(twoProcessIds); });
    EXPECT_EQ(failures.size(), 1);
    EXPECT_TRUE(AnyFailureMentions(failures, "warm-up"));
    EXPECT_EQ(testBed.poolChildren[worker], nullptr);
    EXPECT_FALSE(testBed.configUsedPool);
    EXPECT_EQ(testBed.numActiveChildren, 2);
    errno = 0;
    EXPECT_EQ(waitpid(workerPid, nullptr, WNOHANG), -1);
    EXPECT_EQ(errno, ECHILD);

    testBed.DestroyComms();
    testBed.Finalize();
  }
}
