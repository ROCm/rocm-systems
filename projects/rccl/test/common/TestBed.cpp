/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include <fcntl.h>
#include <poll.h>
#include <sys/wait.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <map>
#include <thread>
#include <unistd.h>
#include "TestBed.hpp"
#include "PipeUtils.hpp"
#include <rccl/rccl.h>

// A worker that stops answering takes every worker of this TestBed with it
// (AbandonChildren), so the current test fails and later tests start fresh.
#define PIPE_WRITE(childId, val)                                                        \
  do                                                                                    \
  {                                                                                     \
    if (RcclUnitTesting::safe_pipe_write(childList[childId]->parentWriteFd, &val, sizeof(val)) != \
        static_cast<ssize_t>(sizeof(val)))                                              \
    {                                                                                   \
      TEST_ERROR("Unable to write to child %d: Error %s", childId, strerror(errno));   \
      FAIL() << AbandonChildren(childId);                                               \
    }                                                                                   \
  } while (0)

#define PIPE_WRITE_VEC(childId, vec)                                                    \
  do                                                                                    \
  {                                                                                     \
    if (!RcclUnitTesting::pipe_write_vec(childList[childId]->parentWriteFd, vec))       \
    {                                                                                   \
      TEST_ERROR("Unable to write to child %d: Error %s", childId, strerror(errno));   \
      FAIL() << AbandonChildren(childId);                                               \
    }                                                                                   \
  } while (0)

#define PIPE_READ(childId, val)                                                         \
  {                                                                                     \
    if (ev.verbose) TEST_INFO("Calling PIPE_READ to Child %d", childId); \
    int const lostChildId = WaitForChild(childId);                                      \
    if (lostChildId == kChildWaitTimedOut)                                              \
    {                                                                                   \
      TEST_ERROR("Child %d did not reply within %d s (UT_CHILD_TIMEOUT_S)", childId,    \
                 ChildReplyTimeoutSec());                                               \
      FAIL() << AbandonChildren(childId, /*waitForExit=*/false);                        \
    }                                                                                   \
    if (lostChildId >= 0)                                                               \
    {                                                                                   \
      TEST_ERROR("Child %d pipe closed while waiting for child %d", lostChildId, childId); \
      FAIL() << AbandonChildren(lostChildId);                                           \
    }                                                                                   \
    ssize_t retval = RcclUnitTesting::safe_pipe_read(childList[childId]->parentReadFd, &val, sizeof(val)); \
    if (ev.verbose) TEST_INFO("Got PIPE_READ %ld from Child %d", retval, childId); \
    if (retval != static_cast<ssize_t>(sizeof(val)))                                    \
    {                                                                                   \
      ReportReadFailure(childId, retval, sizeof(val));                                  \
      FAIL() << AbandonChildren(childId);                                               \
    }                                                                                   \
  }
#define PIPE_CHECK(childId, command)                                                    \
  {                                                                                     \
    TestBedChild::Reply reply = {};                                                     \
    PIPE_READ(childId, reply);                                                          \
    std::string const mismatch = ReplyMismatch(*childList[childId], reply, command);    \
    if (!mismatch.empty())                                                              \
    {                                                                                   \
      TEST_ERROR("Child %d %s", childId, mismatch.c_str());                             \
      FAIL() << AbandonChildren(childId, /*waitForExit=*/false, kOutOfSequence);        \
    }                                                                                   \
    if (reply.status != TEST_SUCCESS)                                                   \
    {                                                                                   \
      TEST_ERROR("Child %d reports failure", childId);                                  \
      this->childFailed = true;                                                         \
      ASSERT_EQ(reply.status, TEST_SUCCESS);                                            \
      FAIL();                                                                           \
    }                                                                                   \
  }

namespace RcclUnitTesting
{
  namespace
  {
    // Matched by suite, not by test: one sweep's sample, so exact test names leave siblings exposed.
    char const* const kGfx95NoPoolSuites[] = {"AlltoAll", "ReduceScatter", "AllGather",
                                              "P2pChannelScaling", "Broadcast", "GroupCall"};

    // Reaps pid. A worker still alive at deadline (e.g. stuck in a collective
    // teardown) is SIGKILLed and the current test fails. With noDeadline the
    // wait is unbounded.
    pid_t ReapChild(pid_t const pid, int* status, bool const noDeadline,
                    std::chrono::steady_clock::time_point const deadline)
    {
      pid_t waitResult = -1;
      if (!noDeadline)
      {
        while (true)
        {
          waitResult = waitpid(pid, status, WNOHANG);
          if (waitResult == -1 && errno == EINTR) continue;
          if (waitResult != 0) return waitResult;
          if (std::chrono::steady_clock::now() >= deadline) break;
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        TEST_ERROR("Child pid %d did not exit before the teardown deadline (UT_CHILD_TIMEOUT_S); killing it", pid);
        ADD_FAILURE() << "Child pid " << pid << " did not exit before the teardown deadline (UT_CHILD_TIMEOUT_S)";
        kill(pid, SIGKILL);
      }
      do
      {
        waitResult = waitpid(pid, status, 0);
      } while (waitResult == -1 && errno == EINTR);
      return waitResult;
    }

    char const* const kOutOfSequence = "sent an out-of-sequence reply";

    // Consumes the next reply sequence number of child. Empty if reply answers
    // command in order; otherwise describes the mismatch. A mismatch means the
    // pipe is out of step and cannot be resynchronized.
    std::string ReplyMismatch(TestBedChild& child, TestBedChild::Reply const& reply, int const command)
    {
      uint32_t const expectedSeq = ++child.replySeq;
      if (reply.command == command && reply.seq == expectedSeq) return "";
      return "sent reply #" + std::to_string(reply.seq) + " to " + TestBedChild::CommandName(reply.command) +
             " where reply #" + std::to_string(expectedSeq) + " to " + TestBedChild::CommandName(command) +
             " was expected";
    }

    // poll() timeout for the time left until deadline; -1 (no limit) when timeoutSec is 0.
    int PollTimeoutMs(int const timeoutSec, std::chrono::steady_clock::time_point const deadline)
    {
      if (timeoutSec <= 0) return -1;
      auto const remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - std::chrono::steady_clock::now());
      return static_cast<int>(std::clamp<long long>(remaining.count(), 0, INT_MAX));
    }

    void ReportReadFailure(int const childId, ssize_t const retval, size_t const expectedBytes)
    {
      if (retval == -1)
      {
        TEST_ERROR("Unable to read from child %d: Error %s", childId, strerror(errno));
      }
      else if (retval == 0)
      {
        TEST_ERROR("Child %d pipe closed unexpectedly", childId);
      }
      else
      {
        TEST_ERROR("Child %d pipe read incomplete (%ld / %lu)", childId, retval, expectedBytes);
      }
    }
  }

  TestBed::TestBed() :
    numDevicesAvailable(0),
    numGroupCalls(0)
  {
    // Ignore SIGPIPE so a write to a dead pool worker fails with EPIPE instead
    // of killing the parent.
    signal(SIGPIPE, SIG_IGN);

    // Collect the number of GPUs
    this->numDevicesAvailable = ev.maxGpus;
    if (ev.verbose) TEST_INFO("Detected %d GPUs", this->numDevicesAvailable);

    // Communicator process pool: ON by default; set UT_COMM_POOL=0 to disable.
    // Parsed/registered centrally in EnvVars (shown in the config banner) like every UT_* var.
    this->poolMode = ev.commPool;

    // Unstable on gfx95 when comms/workers are reused across configs (AICOMRCCL-2275); unconditional,
    // so UT_COMM_POOL=1 cannot defeat it. Exact match excludes AlltoAllv, whose hang is AICOMRCCL-1900.
    if (this->poolMode && ev.isGfx95)
    {
      // TestBed is a local in each TEST body, so current_test_info() is null only outside one.
      ::testing::TestInfo const* testInfo = ::testing::UnitTest::GetInstance()->current_test_info();
      if (testInfo != nullptr && testInfo->test_suite_name() != nullptr)
      {
        for (char const* const suiteName : kGfx95NoPoolSuites)
        {
          if (strcmp(testInfo->test_suite_name(), suiteName) != 0)
          {
            continue;
          }
          this->poolMode = false;
          TEST_INFO("Comm pool (UT_COMM_POOL) forced off for %s.%s: pool reuse is unstable for this suite on gfx95",
                    testInfo->test_suite_name(), testInfo->name());
          break;
        }
      }
    }
    this->configUsedPool = false;
  }

  bool TestBed::SpawnChildProcess(TestBedChild* child, MemAllocType const memAllocType)
  {
    if (child == nullptr)
      return false;

    if (child->InitPipes() != TEST_SUCCESS)
    {
      child->ClosePipes();
      return false;
    }

    // Parent-side descriptors must not leak into subsequently exec'd workers.
    // Child-side descriptors are the only descriptors preserved across exec.
    if (fcntl(child->parentWriteFd, F_SETFD, FD_CLOEXEC) == -1 ||
        fcntl(child->parentReadFd, F_SETFD, FD_CLOEXEC) == -1 ||
        fcntl(child->childWriteFd, F_SETFD, 0) == -1 ||
        fcntl(child->childReadFd, F_SETFD, 0) == -1)
    {
      TEST_ERROR("Unable to configure pipe descriptors for child %d: %s",
                 child->childId, strerror(errno));
      child->ClosePipes();
      return false;
    }

    // Construct all arguments before fork. A HIP-initialized parent can have
    // background threads holding libc locks, so the child must do only
    // async-signal-safe work before replacing its process image.
    std::string const sChildId      = std::to_string(child->childId);
    std::string const sChildReadFd  = std::to_string(child->childReadFd);
    std::string const sChildWriteFd = std::to_string(child->childWriteFd);
    std::string const sVerbose      = std::to_string(ev.verbose ? 1 : 0);
    std::string const sPrintVal     = std::to_string(ev.printValues);
    std::string const sThreading    = std::to_string(ev.useMultithreading ? 1 : 0);
    std::string const sMemAllocType = std::to_string(static_cast<int>(memAllocType));

    char const* childArgv[TestBedChild::NUM_CHILD_ARGS + 1] = {};
    childArgv[0]                                      = program_invocation_name; // parent's argv[0]
    childArgv[TestBedChild::CHILD_ARG_FLAG]           = TestBedChild::kChildFlag;
    childArgv[TestBedChild::CHILD_ARG_ID]             = sChildId.c_str();
    childArgv[TestBedChild::CHILD_ARG_READ_FD]        = sChildReadFd.c_str();
    childArgv[TestBedChild::CHILD_ARG_WRITE_FD]       = sChildWriteFd.c_str();
    childArgv[TestBedChild::CHILD_ARG_VERBOSE]        = sVerbose.c_str();
    childArgv[TestBedChild::CHILD_ARG_PRINT_VALUES]   = sPrintVal.c_str();
    childArgv[TestBedChild::CHILD_ARG_RANK_THREADING] = sThreading.c_str();
    childArgv[TestBedChild::CHILD_ARG_MEM_ALLOC_TYPE] = sMemAllocType.c_str();

    fflush(nullptr);
    pid_t const pid = fork();
    if (pid == 0)
    {
      close(child->parentWriteFd);
      close(child->parentReadFd);
      execv("/proc/self/exe", const_cast<char* const*>(childArgv));
      _exit(127);
    }
    if (pid < 0)
    {
      TEST_ERROR("fork() failed for child %d: %s", child->childId, strerror(errno));
      child->ClosePipes();
      return false;
    }

    child->pid = pid;
    close(child->childWriteFd);
    close(child->childReadFd);
    child->childWriteFd = -1;
    child->childReadFd = -1;
    return true;
  }

  int TestBed::ChildReplyTimeoutSec() const
  {
    if (ev.debugPause || ev.childTimeoutSec <= 0) return 0;
    // ExecuteCollectives replies only after its own UT_TIMEOUT_US expires and
    // the timed-out comms are aborted.
    long long const collectiveTimeoutSec = 2 * ((static_cast<long long>(ev.timeoutUs) + 999999) / 1000000);
    return static_cast<int>(std::min<long long>(std::max<long long>(ev.childTimeoutSec, collectiveTimeoutSec), INT_MAX));
  }

  int TestBed::WaitForChild(int const childId)
  {
    // A peer that dies can leave childId blocked in a collective forever, so
    // watch every active child's pipe, not only the one being read. A child
    // that hangs without dying is caught by the reply timeout.
    int const timeoutSec = ChildReplyTimeoutSec();
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeoutSec);
    std::vector<pollfd> fds;
    std::vector<int> fdChildIds;
    for (int c = 0; c < static_cast<int>(childList.size()); ++c)
    {
      if (childList[c] == nullptr || childList[c]->parentReadFd < 0) continue;
      fds.push_back({childList[c]->parentReadFd, POLLIN, 0});
      fdChildIds.push_back(c);
    }
    if (std::find(fdChildIds.begin(), fdChildIds.end(), childId) == fdChildIds.end())
      return -1;  // no pipe to wait on; the read that follows reports it

    while (true)
    {
      int const pollResult = poll(fds.data(), fds.size(), PollTimeoutMs(timeoutSec, deadline));
      if (pollResult < 0)
      {
        if (errno == EINTR) continue;
        return -1;  // the read that follows reports the error
      }
      if (pollResult == 0) return kChildWaitTimedOut;
      for (size_t i = 0; i < fds.size(); ++i)
      {
        if (fds[i].revents == 0) continue;
        if (fdChildIds[i] == childId) return -1;
        if (fds[i].revents & (POLLHUP | POLLERR | POLLNVAL)) return fdChildIds[i];
        // Data queued by another child is read later. Stop polling it for input
        // (which would spin) but keep watching for its hangup.
        fds[i].events = 0;
      }
    }
  }

  std::string TestBed::AbandonChildren(int const lostChildId, bool const waitForExit, char const* what)
  {
    std::string reason = "Child " + std::to_string(lostChildId) + " " + what;
    TestBedChild* const lost = (lostChildId >= 0 && lostChildId < static_cast<int>(childList.size()))
                               ? childList[lostChildId] : nullptr;
    if (lost != nullptr && lost->pid > 0)
    {
      // The pipe closes before the process is gone (a core dump can take a
      // while), so give it time to finish dying and report the real cause.
      int status = 0;
      pid_t waitResult = 0;
      auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(waitForExit ? 30 : 0);
      while ((waitResult = waitpid(lost->pid, &status, WNOHANG)) == 0 &&
             std::chrono::steady_clock::now() < deadline)
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
      reason += " (pid " + std::to_string(lost->pid) + ")";
      if (waitResult == lost->pid)
      {
        if (WIFSIGNALED(status))
          reason += ": killed by signal " + std::to_string(WTERMSIG(status)) + " (" + strsignal(WTERMSIG(status)) + ")";
        else if (WIFEXITED(status))
          reason += ": exited with code " + std::to_string(WEXITSTATUS(status));
        lost->pid = -1;
      }
    }
    reason += "; stopping all workers of this TestBed";
    TEST_ERROR("%s", reason.c_str());

    // Surviving workers may be waiting on the lost one inside a collective, so
    // they cannot be asked to stop. Kill them all first so they die in
    // parallel; the destructors then reap them and close their pipes.
    for (auto const* workers : {&this->ownedChildren, &this->poolChildren})
    {
      for (auto const& c : *workers)
      {
        if (c && c->pid > 0) kill(c->pid, SIGKILL);
      }
    }
    this->ownedChildren.clear();
    this->poolChildren.clear();
    this->poolWorkerState.clear();
    this->configUsedPool = false;
    ResetActiveConfig();
    return reason;
  }

  void TestBed::ResetActiveConfig()
  {
    this->childList.clear();
    this->rankToChildMap.clear();
    this->childFailed = false;
  }

  TestBedChild* TestBed::AdoptChild(std::unique_ptr<TestBedChild> child)
  {
    this->poolMode = false;
    this->configUsedPool = false;
    this->rankToChildMap.push_back(NumActiveChildren());
    this->childList.push_back(child.get());
    this->ownedChildren.push_back(std::move(child));
    return this->childList.back();
  }

  bool TestBed::SelectRanks(int const rank, std::vector<int>& ranks) const
  {
    ranks.clear();
    int const numRanks = NumActiveRanks();
    if (rank != -1 && (rank < 0 || rank >= numRanks))
    {
      ADD_FAILURE() << "Rank " << rank << " is not active (" << numRanks << " active ranks)";
      return false;
    }
    for (int i = 0; i < numRanks; ++i)
      if (rank == -1 || rank == i) ranks.push_back(i);
    return true;
  }

  bool TestBed::SelectGroups(int const groupId, std::vector<int>& groups) const
  {
    groups.clear();
    if (groupId != -1 && (groupId < 0 || groupId >= this->numGroupCalls))
    {
      ADD_FAILURE() << "Group " << groupId << " is not active (" << this->numGroupCalls << " group calls)";
      return false;
    }
    for (int i = 0; i < this->numGroupCalls; ++i)
      if (groupId == -1 || groupId == i) groups.push_back(i);
    return true;
  }

  std::vector<std::vector<int>> TestBed::RanksPerChild(std::vector<int> const& ranks) const
  {
    std::vector<std::vector<int>> ranksPerChild(NumActiveChildren());
    for (int const rank : ranks)
      ranksPerChild[rankToChildMap[rank]].push_back(rank);
    return ranksPerChild;
  }

  std::vector<int> TestBed::AllChildIds() const
  {
    std::vector<int> childIds(NumActiveChildren());
    for (int childId = 0; childId < NumActiveChildren(); ++childId) childIds[childId] = childId;
    return childIds;
  }

  std::vector<int> TestBed::ChildrenWithRanks(std::vector<std::vector<int>> const& ranksPerChild)
  {
    std::vector<int> childIds;
    for (int childId = 0; childId < static_cast<int>(ranksPerChild.size()); ++childId)
      if (!ranksPerChild[childId].empty()) childIds.push_back(childId);
    return childIds;
  }

  bool TestBed::PoolWorkersFor(std::vector<std::vector<int>> const& deviceIdsPerChild,
                               std::vector<int>& workers) const
  {
    // Map each child onto a distinct pool worker by representative device.
    workers.clear();
    std::set<int> usedWorkers;
    for (auto const& deviceIds : deviceIdsPerChild)
    {
      int const worker = deviceIds.empty() ? -1 : deviceIds[0];
      if (worker < 0 || worker >= (int)this->poolChildren.size() ||
          this->poolChildren[worker] == nullptr || usedWorkers.count(worker))
      {
        return false;  // out-of-range, missing worker, or collision (multi-rank-per-GPU)
      }
      usedWorkers.insert(worker);
      workers.push_back(worker);
    }
    return true;
  }

  void TestBed::WarmPoolWorkers(std::vector<std::vector<int>> const& deviceIdsPerChild)
  {
    if (!this->poolMode || !ev.commWarmAhead) return;
    std::vector<int> workers;
    if (!PoolWorkersFor(deviceIdsPerChild, workers)) return;

    int const cmd = TestBedChild::CHILD_WARMUP;
    for (int const worker : workers)
    {
      if (this->poolWorkerState[worker] != PoolWorkerState::Cold) continue;
      // A failed write leaves the worker Cold; InitComms then reports the dead pipe.
      if (RcclUnitTesting::safe_pipe_write(this->poolChildren[worker]->parentWriteFd, &cmd, sizeof(cmd)) ==
          static_cast<ssize_t>(sizeof(cmd)))
      {
        this->poolWorkerState[worker] = PoolWorkerState::Warming;
        if (ev.verbose) TEST_INFO("Warming pool worker %d", worker);
      }
    }
  }

  bool TestBed::FinishWarmup(int const worker, std::chrono::steady_clock::time_point const deadline)
  {
    TestBedChild* const child = this->poolChildren[worker].get();
    this->poolWorkerState[worker] = PoolWorkerState::Warm;

    int const timeoutSec = ChildReplyTimeoutSec();
    pollfd fd = {child->parentReadFd, POLLIN, 0};
    int pollResult;
    do
    {
      pollResult = poll(&fd, 1, PollTimeoutMs(timeoutSec, deadline));
    } while (pollResult < 0 && errno == EINTR);

    std::string problem;
    TestBedChild::Reply reply = {};
    if (pollResult == 0)
    {
      problem = "did not reply within " + std::to_string(timeoutSec) + " s (UT_CHILD_TIMEOUT_S)";
    }
    else if (RcclUnitTesting::safe_pipe_read(child->parentReadFd, &reply, sizeof(reply)) !=
             static_cast<ssize_t>(sizeof(reply)))
    {
      problem = "pipe closed";
    }
    else
    {
      problem = ReplyMismatch(*child, reply, TestBedChild::CHILD_WARMUP);
      if (problem.empty() && reply.status != TEST_SUCCESS) problem = "reported failure";
    }
    if (problem.empty()) return true;

    std::string const reason = "Pool worker " + std::to_string(worker) + " (pid " +
                               std::to_string(child->pid) + ") warm-up " + problem;
    TEST_ERROR("%s; removing it from the pool", reason.c_str());
    ADD_FAILURE() << reason;
    this->poolChildren[worker].reset();
    return false;
  }

  void TestBed::InitComms(std::vector<std::vector<int>> const& deviceIdsPerProcess,
                          std::vector<int>              const& numCollectivesInGroup,
                          std::vector<int>              const& numStreamsPerGroup,
                          int                           const  numGroupCalls,
                          bool                          const  useBlocking,
                          MemAllocType                  const  memAllocType)
  {
    InteractiveWait("Starting InitComms");

    #ifndef ENABLE_OPENMP
    if (ev.useMultithreading)
    {
      FAIL() << "UT_MULTITHREAD=1 requires a unit-test build with OPENMP_TESTS_ENABLED=ON";
    }
    #endif

    // If children/comms from a previous test mode (e.g. SP) are still active,
    // ensure they are cleanly stopped before creating new ones for MP mode
    if (NumActiveChildren() > 0)
    {
      this->DestroyComms();
    }

    // DestroyComms() owns teardown for both pool and fork-fresh paths. A
    // non-empty childList here means teardown failed to restore the invariant.
    if (!childList.empty())
    {
      FAIL() << "DestroyComms failed to clear childList before InitComms";
    }

    // Track the owning child of each rank
    int const numChildren = static_cast<int>(deviceIdsPerProcess.size());
    this->memAllocType = memAllocType;
    this->numGroupCalls = numGroupCalls;
    this->numCollectivesInGroup = numCollectivesInGroup;
    this->useBlocking = useBlocking;
    this->numStreamsPerGroup = numStreamsPerGroup;
    this->rankToChildMap.clear();
    if (ev.verbose) TEST_INFO("Setting up %d active child processes", numChildren);

    for (int childId = 0; childId < numChildren; ++childId)
      this->rankToChildMap.insert(this->rankToChildMap.end(), deviceIdsPerProcess[childId].size(), childId);

    // Comm pool (UT_COMM_POOL): worker d is pinned to device d and keeps its
    // device-code object resident, so reuse skips the ~15-30s load per config.
    this->configUsedPool = false;
    if (this->poolMode)
    {
      // Each worker snapshots env at exec and NCCL_PARAM caches it; a test that
      // changes env must call Finalize() to re-exec the pool.
      if (this->poolChildren.empty())
      {
        int poolSize = this->numDevicesAvailable;
        int const detectedGpus = ev.GetNumDetectedGpus();
        if (detectedGpus > 0 && detectedGpus < poolSize)
        {
          poolSize = detectedGpus;
        }
        this->poolChildren.resize(poolSize);
        this->poolWorkerState.assign(poolSize, PoolWorkerState::Cold);
        for (int d = 0; d < poolSize; ++d)
        {
          this->poolChildren[d] = std::make_unique<TestBedChild>(d, ev.verbose, ev.printValues, ev.useMultithreading);
          if (!SpawnChildProcess(this->poolChildren[d].get(), memAllocType))
          {
            // Reap the half-built pool; FAIL() (not TEST_ERROR) so the sweep
            // stops instead of indexing an empty childList -> SEGV.
            TeardownPool();
            FAIL() << "Unable to start pool child process " << d;
          }
        }
        // Do NOT pre-warm the whole pool here: under the runner's --jobs N it
        // storms ncclCommInitAll across all GPUs and fails intermittently.
        // Sweeps warm only the next configs' workers (WarmPoolWorkers).
      }

      std::vector<int> workers;
      bool mappable = PoolWorkersFor(deviceIdsPerProcess, workers);
      // Workers warm in parallel, so they share one reply deadline.
      auto const warmupDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(ChildReplyTimeoutSec());
      for (int const worker : workers)
      {
        if (this->poolWorkerState[worker] == PoolWorkerState::Warming && !FinishWarmup(worker, warmupDeadline))
        {
          mappable = false;  // worker removed from the pool -> fork-fresh
        }
      }

      if (mappable)
      {
        // Reuse: childList borrows the selected pool workers (pool retains ownership).
        childList.clear();
        for (int const worker : workers)
        {
          childList.push_back(this->poolChildren[worker].get());
          this->poolWorkerState[worker] = PoolWorkerState::Warm;
        }
        this->configUsedPool = true;
      }
    }

    if (!this->configUsedPool)
    {
      // ---- Classic fork-fresh path (pool disabled, or an unmappable config) ----
      // The childList invariant was checked above after any prior teardown.
      for (int childId = 0; childId < numChildren; ++childId)
      {
        auto child = std::make_unique<TestBedChild>(childId, ev.verbose, ev.printValues, ev.useMultithreading);
        if (!SpawnChildProcess(child.get(), memAllocType))
        {
          // The never-started child is dropped here, so teardown does not write to or wait on it.
          // FAIL() only returns from InitComms and callers may carry on, so no
          // rank may still map to a child that was never started.
          this->rankToChildMap.clear();
          FAIL() << "Unable to start child process " << childId;
        }
        childList.push_back(child.get());
        this->ownedChildren.push_back(std::move(child));
      }
    }

    // If debugging is enabled, pause here to allow users to attach debugger
    if (ev.debugPause) {
      TEST_INFO("============================================================");
      TEST_INFO(" Pausing for debug attach: (e.g. sudo rocgdb -p <PID>)");
      TEST_INFO("============================================================");
      for (int childId = 0; childId < NumActiveChildren(); ++childId) {
        TEST_INFO(" Child %02d: processID: %d", childId, childList[childId]->pid);
      }
      TEST_INFO("============================================================");
      TEST_INFO("<Press enter to continue>");
      scanf("%*c");
    }

    // Use the MultiRank interfaces when some GPU hosts more than one rank.
    std::set<int> uniqueDevices;
    for (auto const& deviceIds : deviceIdsPerProcess)
      uniqueDevices.insert(deviceIds.begin(), deviceIds.end());
    bool const useMulti = static_cast<int>(uniqueDevices.size()) < NumActiveRanks();

    // Tell first rank to get ncclUniqueId
    int getIdCmd = TestBedChild::CHILD_GET_UNIQUE_ID;
    PIPE_WRITE(0, getIdCmd);
    PIPE_CHECK(0, TestBedChild::CHILD_GET_UNIQUE_ID);

    // Receive back unique ID from first rank
    ncclUniqueId id;
    PIPE_READ(0, id);

    // Send InitComms command to each active child process
    int const cmd = TestBedChild::CHILD_INIT_COMMS;
    int const numRanks = NumActiveRanks();
    int rankOffset = 0;
    for (int childId = 0; childId < NumActiveChildren(); ++childId)
    {
      if (ev.verbose) TEST_INFO("Sending InitComm event to child %d", childId);
      PIPE_WRITE(childId, cmd);

      // Send unique ID to child process
      PIPE_WRITE(childId, id);

      // Send total number of ranks to child process
      PIPE_WRITE(childId, numRanks);

      // Send the rank offset for this child process
      PIPE_WRITE(childId, rankOffset);

      // Send the total number of group calls for this child process
      PIPE_WRITE(childId, numGroupCalls);

      // Send the number of collectives to be run per group call
      PIPE_WRITE_VEC(childId, this->numCollectivesInGroup);

      // Send the RCCL communication with blocking or non-blocking option
      PIPE_WRITE(childId, useBlocking);

      // Send memAllocType to child process
      int const memAllocTypeVal = static_cast<int>(memAllocType);
      PIPE_WRITE(childId, memAllocTypeVal);

      // Send whether to use MultiRank interfaces or not.
      PIPE_WRITE(childId, useMulti);

      // Send how many streams to use per group call
      PIPE_WRITE_VEC(childId, this->numStreamsPerGroup);

      // Send the GPUs this child uses
      int const numGpus = deviceIdsPerProcess[childId].size();
      PIPE_WRITE(childId, numGpus);
      for (int i = 0; i < numGpus; i++)
        PIPE_WRITE(childId, deviceIdsPerProcess[childId][i]);

      rankOffset += numGpus;
    }

    // Wait for child acknowledgement
    // This is done after previous loop to avoid deadlock as every rank needs to enter ncclInitCommRank
    CollectAcks(TestBedChild::CHILD_INIT_COMMS, AllChildIds(), OnChildFailure::StopIfOwed);
    InteractiveWait("Finishing InitComms");
  }

  void TestBed::InitComms(std::vector<std::vector<int>> const& deviceIdsPerProcess,
                          int const numCollectivesInGroup, int const numStreamsPerGroup, int const numGroupCalls, bool const useBlocking, MemAllocType const memAllocType)
  {
    InitComms(deviceIdsPerProcess, TestBed::GetNumCollsPerGroup(numCollectivesInGroup, numGroupCalls), TestBed::GetNumStreamsPerGroup(numStreamsPerGroup, numGroupCalls), numGroupCalls, useBlocking, memAllocType);
  }

  void TestBed::InitComms(int const numGpus, int const numCollectivesInGroup, int const numStreamsPerGroup, int const numGroupCalls, bool const useBlocking, MemAllocType const memAllocType)
  {
     const std::vector<int>& gpuPriorityOrder = ev.GetGpuPriorityOrder();
     InitComms(GetDeviceIdsList(1, numGpus, gpuPriorityOrder), TestBed::GetNumCollsPerGroup(numCollectivesInGroup, numGroupCalls), TestBed::GetNumStreamsPerGroup(numStreamsPerGroup, numGroupCalls), numGroupCalls, useBlocking, memAllocType);
  }

  void TestBed::SetCollectiveArgs(ncclFunc_t      const funcType,
                                  ncclDataType_t  const dataType,
                                  size_t          const numInputElements,
                                  size_t          const numOutputElements,
                                  OptionalColArgs const &optionalArgs,
                                  int             const collId,
                                  int             const groupId,
                                  int             const rank,
                                  int             const streamIdx)
  {
    InteractiveWait("Starting SetCollectiveArgs");
    std::vector<int> rankList;
    if (!SelectRanks(rank, rankList)) return;

    if (groupId < 0 || groupId >= this->numGroupCalls ||
        groupId >= static_cast<int>(this->numStreamsPerGroup.size()))
    {
      FAIL() << "SetCollectiveArgs group " << groupId << " is not active (" << this->numGroupCalls
             << " group calls)";
    }
    if (streamIdx < 0 || streamIdx >= this->numStreamsPerGroup[groupId])
    {
      TEST_ERROR("StreamIdx for group %d collective %d is out of bounds (%d/%d):", groupId, collId, streamIdx, numStreamsPerGroup[groupId]);
      FAIL();
    }

    // An out-of-range value would make the worker exit, which costs this
    // TestBed all of its workers rather than just this collective.
    // redOp is not checked: custom scalar reductions use values >= ncclNumOps.
    if (funcType < 0 || funcType >= ncclNumFuncs)
    {
      TEST_ERROR("Invalid function type %d for group %d collective %d", funcType, groupId, collId);
      FAIL();
    }
    if (dataType < 0 || dataType >= ncclNumTypes)
    {
      TEST_ERROR("Invalid datatype %d for group %d collective %d", dataType, groupId, collId);
      FAIL();
    }

    // Send CollectiveArgs for every selected rank before waiting, so children work in parallel
    int const cmd = TestBedChild::CHILD_SET_COLL_ARGS;
    std::vector<int> ackChildIds;
    for (auto currRank : rankList)
    {
      int const childId = rankToChildMap[currRank];
      ackChildIds.push_back(childId);
      PIPE_WRITE(childId, cmd);
      PIPE_WRITE(childId, currRank);
      PIPE_WRITE(childId, collId);
      PIPE_WRITE(childId, groupId);
      PIPE_WRITE(childId, funcType);
      PIPE_WRITE(childId, dataType);
      PIPE_WRITE(childId, numInputElements);
      PIPE_WRITE(childId, numOutputElements);
      PIPE_WRITE(childId, streamIdx);
      PIPE_WRITE(childId, optionalArgs);
    }
    CollectAcks(TestBedChild::CHILD_SET_COLL_ARGS, ackChildIds);
    InteractiveWait("Finishing SetCollectiveArgs");
  }

  void TestBed::CollectAcks(int const command, std::vector<int> const& childIds,
                            OnChildFailure const onFailure, bool* allSucceeded)
  {
    if (allSucceeded) *allSucceeded = false;
    char const* const commandName = TestBedChild::CommandName(command);

    // Drain every ack before failing, or the next command reads a stale one. Arrival
    // order lets a failure be seen while an earlier child is blocked on the failed one.
    using Clock = std::chrono::steady_clock;
    std::map<int, int> pendingAcks;
    for (int childId : childIds) ++pendingAcks[childId];
    int const replyTimeoutSec = ChildReplyTimeoutSec();
    auto const replyDeadline = Clock::now() + std::chrono::seconds(replyTimeoutSec);
    int numFailed = 0;
    int firstFailedChild = -1;
    while (!pendingAcks.empty())
    {
      // Watch every active child for a hangup: a dead peer can leave a pending
      // child blocked forever. Only pending children are read.
      std::vector<pollfd> fds;
      std::vector<int> fdChildIds;
      for (int c = 0; c < static_cast<int>(childList.size()); ++c)
      {
        if (childList[c] == nullptr || childList[c]->parentReadFd < 0) continue;
        fds.push_back({childList[c]->parentReadFd, static_cast<short>(pendingAcks.count(c) ? POLLIN : 0), 0});
        fdChildIds.push_back(c);
      }
      for (auto const& pending : pendingAcks)
      {
        if (std::find(fdChildIds.begin(), fdChildIds.end(), pending.first) != fdChildIds.end()) continue;
        TEST_ERROR("Child %d has no pipe to read its acknowledgement from", pending.first);
        FAIL() << AbandonChildren(pending.first);
      }

      int const pollResult = poll(fds.data(), fds.size(), PollTimeoutMs(replyTimeoutSec, replyDeadline));
      if (pollResult < 0)
      {
        if (errno == EINTR) continue;
        TEST_ERROR("poll on child pipes failed: %s", strerror(errno));
        FAIL() << AbandonChildren(pendingAcks.begin()->first);
      }
      if (pollResult == 0)
      {
        int const stuckChild = pendingAcks.begin()->first;
        TEST_ERROR("Child %d did not reply within %d s (UT_CHILD_TIMEOUT_S)", stuckChild, replyTimeoutSec);
        FAIL() << AbandonChildren(stuckChild, /*waitForExit=*/false);
      }

      for (size_t i = 0; i < fds.size(); ++i)
      {
        if (fds[i].revents == 0) continue;
        int const childId = fdChildIds[i];
        if (!pendingAcks.count(childId))
        {
          TEST_ERROR("Child %d pipe closed while waiting for child %d", childId, pendingAcks.begin()->first);
          FAIL() << AbandonChildren(childId);
        }
        TestBedChild::Reply reply = {};
        ssize_t const retval = RcclUnitTesting::safe_pipe_read(fds[i].fd, &reply, sizeof(reply));
        if (retval != static_cast<ssize_t>(sizeof(reply)))
        {
          ReportReadFailure(childId, retval, sizeof(reply));
          FAIL() << AbandonChildren(childId);
        }
        std::string const mismatch = ReplyMismatch(*childList[childId], reply, command);
        if (!mismatch.empty())
        {
          TEST_ERROR("Child %d %s", childId, mismatch.c_str());
          FAIL() << AbandonChildren(childId, /*waitForExit=*/false, kOutOfSequence);
        }
        if (--pendingAcks[childId] == 0) pendingAcks.erase(childId);
        if (reply.status != TEST_SUCCESS)
        {
          TEST_ERROR("Child %d reports failure", childId);
          if (numFailed++ == 0) firstFailedChild = childId;
          // Children still owing an ack may be blocked on this one and cannot
          // read an abort request; stopping the workers is the only way out.
          if (onFailure != OnChildFailure::Report && !pendingAcks.empty())
          {
            std::string waiting;
            for (auto const& pending : pendingAcks)
            {
              waiting += (waiting.empty() ? "" : ", ") + std::to_string(pending.first);
            }
            TEST_ERROR("Not waiting for child(ren) %s after child %d failed", waiting.c_str(), childId);
            FAIL() << AbandonChildren(childId, /*waitForExit=*/false, "reported failure");
          }
        }
      }
    }
    if (numFailed > 0 && onFailure == OnChildFailure::StopAlways)
    {
      TEST_ERROR("%s: work queued on the other children waits on child %d", commandName, firstFailedChild);
      FAIL() << AbandonChildren(firstFailedChild, /*waitForExit=*/false, "reported failure");
    }
    if (numFailed > 0)
    {
      this->childFailed = true;
      FAIL() << commandName << ": " << numFailed << " of " << childIds.size()
             << " child acknowledgements reported failure";
    }
    if (allSucceeded) *allSucceeded = true;
  }

  bool TestBed::AllocateMemInternal(bool   const inPlace,
                                    bool   const useManagedMem,
                                    int    const groupId,
                                    int    const collId,
                                    int    const rank,
                                    bool   const userRegistered)
  {
    InteractiveWait("Starting AllocateMemInternal");
    std::vector<int> rankList, groupList;
    if (!SelectRanks(rank, rankList) || !SelectGroups(groupId, groupList)) return false;

    // Loop over all ranks and send allocation command to appropriate child process.
    // The lambda confines the early return of the ASSERT-based PIPE_WRITE.
    bool allocated = false;
    [&]()
    {
      int const cmd = TestBedChild::CHILD_ALLOCATE_MEM;
      std::vector<int> ackChildIds;
      for (auto currGroup : groupList) {
        for (auto currRank : rankList)
        {
          int const childId = rankToChildMap[currRank];
          PIPE_WRITE(childId, cmd);
          PIPE_WRITE(childId, currRank);
          PIPE_WRITE(childId, collId);
          PIPE_WRITE(childId, inPlace);
          PIPE_WRITE(childId, useManagedMem);
          PIPE_WRITE(childId, userRegistered);
          PIPE_WRITE(childId, currGroup);
          ackChildIds.push_back(childId);
        }
      }
      // Each CHILD_ALLOCATE_MEM command produces one acknowledgement. Read one
      // acknowledgement from the child that received that command.
      CollectAcks(TestBedChild::CHILD_ALLOCATE_MEM, ackChildIds, OnChildFailure::Report, &allocated);
    }();
    InteractiveWait("Finishing AllocateMemInternal");
    return allocated;
  }

  void TestBed::RegisterMemInternal(int    const groupId,
                                    int    const collId,
                                    int    const rank)
  {
    InteractiveWait("Starting RegisterMemInternal");
    std::vector<int> rankList, groupList;
    if (!SelectRanks(rank, rankList) || !SelectGroups(groupId, groupList)) return;

    // Group selected ranks by child so each child can register all of its
    // selected local ranks in one grouped RCCL call.
    std::vector<std::vector<int>> const ranksPerChild = RanksPerChild(rankList);
    std::vector<int> const childIds = ChildrenWithRanks(ranksPerChild);

    // Symmetric-window registration is collective over the comm, so peers wait on each other.
    OnChildFailure const registrationFailure = this->memAllocType == MEM_ALLOC_SYMMETRIC_WIN
                                               ? OnChildFailure::StopIfOwed : OnChildFailure::Report;
    int const regCmd = TestBedChild::CHILD_REGISTER_MEM;
    for (auto currGroup : groupList) {
      // Send to all participating children before waiting so collective
      // symmetric-window registration can make progress across processes.
      for (int const childId : childIds) {
        PIPE_WRITE(childId, regCmd);
        PIPE_WRITE(childId, currGroup);
        PIPE_WRITE(childId, collId);
        int const numRanks = static_cast<int>(ranksPerChild[childId].size());
        PIPE_WRITE(childId, numRanks);
        for (int currRank : ranksPerChild[childId])
          PIPE_WRITE(childId, currRank);
      }

      bool allAcked = false;
      CollectAcks(TestBedChild::CHILD_REGISTER_MEM, childIds, registrationFailure, &allAcked);
      if (!allAcked) return;
    }
    InteractiveWait("Finishing RegisterMemInternal");
  }

  void TestBed::AllocateMem(bool   const inPlace,
                            bool   const useManagedMem,
                            int    const groupId,
                            int    const collId,
                            int    const rank,
                            bool   const userRegistered)
  {
    // Symmetric window registration is collective over the whole communicator,
    // so registering a subset of ranks would leave the others outside the
    // barrier and hang.
    if (this->memAllocType == MEM_ALLOC_SYMMETRIC_WIN && rank != -1)
    {
      FAIL() << "Symmetric window registration requires all ranks (rank = -1), got rank " << rank;
    }
    // Symmetric windows need ncclMemAlloc buffers, which userRegistered selects.
    bool const useNcclMemAlloc = userRegistered || this->memAllocType == MEM_ALLOC_SYMMETRIC_WIN;
    if (!this->AllocateMemInternal(inPlace,useManagedMem,groupId,collId,rank,useNcclMemAlloc))
      return;
    if (useNcclMemAlloc)
      this->RegisterMemInternal(groupId,collId,rank);
  }

  void TestBed::PrepareData(int         const groupId,
                            int         const collId,
                            int         const rank,
                            CollFuncPtr const prepDataFunc)
  {
    InteractiveWait("Starting PrepareData");
    std::vector<int> rankList, groupList;
    if (!SelectRanks(rank, rankList) || !SelectGroups(groupId, groupList)) return;

    // Send every selected rank its prepare data command before waiting, so children work in parallel
    int const cmd = TestBedChild::CHILD_PREPARE_DATA;
    intptr_t const prepDataFuncOffset = CollFuncPtrToOffset(prepDataFunc);
    std::vector<int> ackChildIds;
    for (auto currGroup : groupList)
    {
      for (auto currRank : rankList)
      {
        int const childId = rankToChildMap[currRank];
        PIPE_WRITE(childId, cmd);
        PIPE_WRITE(childId, currRank);
        PIPE_WRITE(childId, currGroup);
        PIPE_WRITE(childId, collId);
        PIPE_WRITE(childId, prepDataFuncOffset);
        ackChildIds.push_back(childId);
      }
    }
    CollectAcks(TestBedChild::CHILD_PREPARE_DATA, ackChildIds);
    InteractiveWait("Finishing PrepareData");
  }

  void TestBed::ExecuteCollectives(std::vector<int> const &currentRanks, int const groupId,
                                   bool const useHipGraph)
  {
    InteractiveWait("Starting ExecuteCollectives");

    int const cmd = TestBedChild::CHILD_EXECUTE_COLL;
    ++TestBed::NumTestsRun();

    for (int const currRank : currentRanks)
    {
      if (currRank < 0 || currRank >= NumActiveRanks())
      {
        FAIL() << "ExecuteCollectives rank " << currRank << " is not active ("
               << NumActiveRanks() << " active ranks)";
      }
    }
    std::vector<int> groupList;
    if (!SelectGroups(groupId, groupList)) return;

    // An empty currentRanks runs every rank.
    std::vector<int> const childIds = currentRanks.empty() ? AllChildIds()
                                                           : ChildrenWithRanks(RanksPerChild(currentRanks));
    for (auto currGroup : groupList) {
      // Send ExecuteColl command to each participating child process
      for (int const childId : childIds)
      {
        InteractiveWait("Starting ExecuteCollectives for child " + std::to_string(childId));
        PIPE_WRITE(childId, cmd);
        PIPE_WRITE(childId, ev.timeoutUs);
        PIPE_WRITE(childId, currGroup);
        PIPE_WRITE(childId, useHipGraph);
        int tempCurrentRanks = currentRanks.size();
        PIPE_WRITE(childId, tempCurrentRanks);
        for (int rank = 0; rank < currentRanks.size(); ++rank){
          PIPE_WRITE(childId, currentRanks[rank]);
        }
      }

      bool allAcked = false;
      CollectAcks(TestBedChild::CHILD_EXECUTE_COLL, childIds, OnChildFailure::StopIfOwed, &allAcked);
      if (!allAcked) return;
    }

    InteractiveWait("Finishing ExecuteCollectives");
  }

  void TestBed::ValidateResults(bool& isCorrect, int const groupId, int const collId, int const rank)
  {
    InteractiveWait("Starting ValidateResults");
    isCorrect = false;
    std::vector<int> rankList, groupList;
    if (!SelectRanks(rank, rankList) || !SelectGroups(groupId, groupList)) return;

    // Send every selected rank its ValidateResults command before waiting, so children work in parallel
    int const cmd = TestBedChild::CHILD_VALIDATE_RESULTS;
    std::vector<int> ackChildIds;
    for (auto currGroup : groupList)
    {
      for (auto currRank : rankList)
      {
        int const childId = rankToChildMap[currRank];
        PIPE_WRITE(childId, cmd);
        PIPE_WRITE(childId, currRank);
        PIPE_WRITE(childId, currGroup);
        PIPE_WRITE(childId, collId);
        ackChildIds.push_back(childId);
      }
    }
    CollectAcks(TestBedChild::CHILD_VALIDATE_RESULTS, ackChildIds, OnChildFailure::Report, &isCorrect);

    InteractiveWait("Finishing ValidateResults");
  }

  void TestBed::LaunchGraphs(int const groupId)
  {
    InteractiveWait("Starting LaunchGraphs");
    std::vector<int> groupList;
    if (!SelectGroups(groupId, groupList)) return;

    // A launch only queues the graph, so the replies arrive without peers waiting on
    // each other. But the launched graphs wait on any child that failed to launch,
    // and no later command could complete, so a failed launch stops every worker.
    int const cmd = TestBedChild::CHILD_LAUNCH_GRAPHS;
    for (auto currGroup : groupList)
    {
      for (int childId = 0; childId < NumActiveChildren(); ++childId)
      {
        PIPE_WRITE(childId, cmd);
        PIPE_WRITE(childId, currGroup);
      }
      bool allLaunched = false;
      CollectAcks(TestBedChild::CHILD_LAUNCH_GRAPHS, AllChildIds(), OnChildFailure::StopAlways, &allLaunched);
      if (!allLaunched) return;
    }

    InteractiveWait("Finishing LaunchGraphs");
  }

  void TestBed::DeallocateMem(int const groupId, int const collId, int const rank)
  {
    InteractiveWait("Starting DeallocateMem");
    std::vector<int> rankList, groupList;
    if (!SelectRanks(rank, rankList) || !SelectGroups(groupId, groupList)) return;

    // Symmetric registration is grouped across every rank owned by a child.
    // Mirror that shape during teardown, and finish deregistration on every
    // child before any backing allocation is freed.
    std::vector<std::vector<int>> const ranksPerChild = RanksPerChild(rankList);
    std::vector<int> const childIds = ChildrenWithRanks(ranksPerChild);

    OnChildFailure const registrationFailure = this->memAllocType == MEM_ALLOC_SYMMETRIC_WIN
                                               ? OnChildFailure::StopIfOwed : OnChildFailure::Report;
    int const deregCmd = TestBedChild::CHILD_DEREGISTER_MEM;
    int const deallocCmd = TestBedChild::CHILD_DEALLOCATE_MEM;
    for (auto currGroup : groupList) {
      // Phase 1: deregister all selected windows/handles. Send every child its
      // command before waiting so any collective registration teardown can
      // make progress across processes.
      for (int const childId : childIds) {
        PIPE_WRITE(childId, deregCmd);
        PIPE_WRITE(childId, currGroup);
        PIPE_WRITE(childId, collId);
        int const numRanks = static_cast<int>(ranksPerChild[childId].size());
        PIPE_WRITE(childId, numRanks);
        for (int currRank : ranksPerChild[childId])
          PIPE_WRITE(childId, currRank);
      }

      bool allDeregistered = false;
      CollectAcks(TestBedChild::CHILD_DEREGISTER_MEM, childIds, registrationFailure, &allDeregistered);
      if (!allDeregistered) return;

      // Phase 2: now no selected communicator retains a window/registration
      // over these buffers, so their backing allocations can be released.
      std::vector<int> deallocAckChildIds;
      for (auto currRank : rankList)
      {
        int const childId = rankToChildMap[currRank];
        PIPE_WRITE(childId, deallocCmd);
        PIPE_WRITE(childId, currRank);
        PIPE_WRITE(childId, currGroup);
        PIPE_WRITE(childId, collId);
        deallocAckChildIds.push_back(childId);
      }

      // Each CHILD_DEALLOCATE_MEM command produces one acknowledgement.
      CollectAcks(TestBedChild::CHILD_DEALLOCATE_MEM, deallocAckChildIds);
    }

    InteractiveWait("Finishing DeallocateMem");
  }

  void TestBed::QuerySymmetricSupport(bool& isSupported)
  {
    isSupported = false;
    int const cmd = TestBedChild::CHILD_QUERY_SYMMETRIC;
    bool allSupported = true;
    for (int childId = 0; childId < NumActiveChildren(); ++childId)
    {
      PIPE_WRITE(childId, cmd);
      PIPE_CHECK(childId, TestBedChild::CHILD_QUERY_SYMMETRIC);
      int childSupported = 0;
      PIPE_READ(childId, childSupported);
      if (!childSupported) allSupported = false;
    }
    isSupported = allSupported && NumActiveChildren() > 0;
  }

  void TestBed::DestroyComms()
  {
    InteractiveWait("Starting DestroyComms");

    int const cmd = TestBedChild::CHILD_DESTROY_COMMS;
    // After any child failure, healthy ranks must abort too: finalizing would
    // block them in the intranode barrier waiting for the failed rank.
    int const abortComms = this->childFailed ? 1 : 0;
    if (abortComms && NumActiveChildren() > 0)
    {
      TEST_WARN("DestroyComms: aborting communicators on all %d children after a child failure",
                NumActiveChildren());
    }

    // Send DestroyComms command to all active child processes first so they can
    // work in parallel, then collect acknowledgements in a second pass. Run the
    // pipe traffic inside a lambda: the PIPE_WRITE/PIPE_CHECK macros call
    // gtest's ASSERT_*/FAIL() which return on failure, so a failed write to one
    // child would otherwise skip Finalize() and orphan the remaining children.
    // The lambda confines that early return, so Finalize() always runs.
    using Clock = std::chrono::steady_clock;
    [&]()
    {
      // Timestamps are only consumed when verbose, so skip the clock reads on
      // the common (non-verbose) path, which is hot in the test suite.
      Clock::time_point sendStart, waitStart;
      if (ev.verbose) sendStart = Clock::now();
      for (int childId = 0; childId < NumActiveChildren(); ++childId)
      {
        PIPE_WRITE(childId, cmd);
        PIPE_WRITE(childId, abortComms);
      }
      if (ev.verbose) waitStart = Clock::now();
      CollectAcks(TestBedChild::CHILD_DESTROY_COMMS, AllChildIds(), OnChildFailure::StopIfOwed);

      if (ev.verbose)
      {
        using std::chrono::duration_cast;
        using std::chrono::milliseconds;
        long long const sendMs = duration_cast<milliseconds>(waitStart - sendStart).count();
        long long const waitMs = duration_cast<milliseconds>(Clock::now() - waitStart).count();
        TEST_INFO("DestroyComms: %d children, send %lld ms, parallel teardown %lld ms",
                  NumActiveChildren(), sendMs, waitMs);
      }
    }();

    if (this->configUsedPool)
    {
      // Pool mode: comms/streams were freed on the workers, but keep the workers alive for
      // the next config (their device-code kernels stay resident). Drop the borrowed refs;
      // the pool keeps ownership. Finalize() (between sweeps / at teardown) reaps the pool.
      ResetActiveConfig();
    }
    else
    {
      // Classic / fork-fresh: reap the per-config children this call owns.
      TeardownOwnedChildList();
    }
    this->childFailed = false;

    InteractiveWait("Finishing DestroyComms");
  }

  void TestBed::DestroyGraphs()
  {
    InteractiveWait("Starting DestroyGraphs");

    int const cmd = TestBedChild::CHILD_DESTROY_GRAPHS;
    for (int currGroup = 0; currGroup < this->numGroupCalls; ++currGroup)
    {
      // Send DestroyGraphs command to all active child processes first so they
      // can work in parallel, then collect acknowledgements in a second pass.
      for (int childId = 0; childId < NumActiveChildren(); ++childId)
      {
        PIPE_WRITE(childId, cmd);
        PIPE_WRITE(childId, currGroup);
      }
      bool allAcked = false;
      CollectAcks(TestBedChild::CHILD_DESTROY_GRAPHS, AllChildIds(), OnChildFailure::Report, &allAcked);
      if (!allAcked) return;
    }

    InteractiveWait("Finishing DestroyGraphs");
  }

  void TestBed::Finalize()
  {
    // Reap per-config fork-fresh children (if any), then the persistent pool. Finalize()
    // is the pool-reset boundary: tests that change env between sweeps call it, so the
    // next sweep re-forks a pool that inherits the new env.
    TeardownOwnedChildList();
    TeardownPool();
  }

  void TestBed::TeardownOwnedChildList()
  {
    // Only reaps the fork-fresh children of the active config. A pool config
    // owns none (DestroyComms drops its borrowed refs), so this is a no-op.
    if (this->ownedChildren.empty())
      return;

    InteractiveWait("Starting Finalize");

    // Stop is best-effort: a worker may already have exited after reporting a
    // HIP/RCCL failure. Teardown must still close every pipe, reap every child,
    // and restore the empty-childList invariant.
    int const cmd = TestBedChild::CHILD_STOP;
    for (auto const& child : this->ownedChildren)
    {
      if (child->pid > 0 && child->parentWriteFd >= 0)
        (void)RcclUnitTesting::safe_pipe_write(child->parentWriteFd, &cmd, sizeof(cmd));
      if (child->parentWriteFd >= 0)
      {
        close(child->parentWriteFd);
        child->parentWriteFd = -1;
      }
    }

    // Wait for processes to stop. Children tear down in parallel, so they share one deadline.
    int const timeoutSec = ChildReplyTimeoutSec();
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeoutSec);
    for (int childId = 0; childId < static_cast<int>(this->ownedChildren.size()); ++childId)
    {
      TestBedChild* const child = this->ownedChildren[childId].get();
      int returnVal = 0;
      pid_t waitResult = -1;
      if (child->pid > 0)
      {
        waitResult = ReapChild(child->pid, &returnVal, timeoutSec == 0, deadline);
        child->pid = -1;
      }
      if (waitResult > 0 && WIFSIGNALED(returnVal))
      {
        TEST_ERROR("Child process %d killed by signal %d", childId, WTERMSIG(returnVal));
      }
      else if (waitResult > 0 && WIFEXITED(returnVal) && WEXITSTATUS(returnVal) != 0)
      {
        TEST_ERROR("Child process %d exited with code %d", childId, WEXITSTATUS(returnVal));
      }
    }
    // Only close the read ends AFTER the child processes are dead
    this->ownedChildren.clear();

    ResetActiveConfig();

    InteractiveWait("Finishing Finalize");
  }

  void TestBed::TeardownPool()
  {
    if (this->poolChildren.empty())
      return;

    // A warming worker writes its reply before it reads STOP; collect it so a
    // failed warm-up is reported and the reply does not hit a closed pipe.
    auto const warmupDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(ChildReplyTimeoutSec());
    for (int worker = 0; worker < (int)this->poolWorkerState.size(); ++worker)
    {
      if (this->poolChildren[worker] != nullptr &&
          this->poolWorkerState[worker] == PoolWorkerState::Warming)
      {
        (void)FinishWarmup(worker, warmupDeadline);
      }
    }

    // Best-effort stop (avoid the ASSERT-based PIPE_WRITE during teardown), then reap.
    int const cmd = TestBedChild::CHILD_STOP;
    for (auto const& c : this->poolChildren)
    {
      if (!c)
      {
        continue;
      }
      // Only a forked worker (pid > 0) has a reader on the pipe; skip the STOP write for a
      // never-forked entry.
      if (c->pid > 0 && c->parentWriteFd >= 0)
        (void)RcclUnitTesting::safe_pipe_write(c->parentWriteFd, &cmd, sizeof(cmd));
      c->ClosePipes();
    }
    int const timeoutSec = ChildReplyTimeoutSec();
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeoutSec);
    for (auto const& c : this->poolChildren)
    {
      if (!c)
      {
        continue;
      }
      // Reap only workers we actually forked; a never-forked entry has pid == -1.
      if (c->pid > 0)
      {
        int returnVal = 0;
        pid_t const waitResult = ReapChild(c->pid, &returnVal, timeoutSec == 0, deadline);
        c->pid = -1;
        if (waitResult > 0 && WIFSIGNALED(returnVal))
        {
          TEST_ERROR("Pool worker %d killed by signal %d", c->childId, WTERMSIG(returnVal));
        }
        else if (waitResult > 0 && WIFEXITED(returnVal) && WEXITSTATUS(returnVal) != 0)
        {
          TEST_ERROR("Pool worker %d exited with code %d", c->childId, WEXITSTATUS(returnVal));
        }
      }
    }
    this->poolChildren.clear();
    this->poolWorkerState.clear();
    this->configUsedPool = false;
    ResetActiveConfig();
  }

  TestBed::~TestBed()
  {
    Finalize();
  }

  std::vector<ncclRedOp_t> const& TestBed::GetAllSupportedRedOps()
  {
    return ev.GetAllSupportedRedOps();
  }

  std::vector<ncclDataType_t> const& TestBed::GetAllSupportedDataTypes()
  {
    return ev.GetAllSupportedDataTypes();
  }

  void TestBed::GetSupportedRedOps(std::vector<ncclRedOp_t>& redOps, const std::vector<ncclRedOp_t>& testRedOps)
  {
    // Filter out any unsupported reduction ops, in case only subset has been compiled for
    auto& supportedOps = ev.GetAllSupportedRedOps();
    for (auto redop : testRedOps)
    {
      for (int i = 0; i < supportedOps.size(); ++i)
      {
        if (supportedOps[i] == redop)
        {
          redOps.push_back(redop);
          break;
        }
      }
    }
  }

  void TestBed::GetSupportedDataTypes(std::vector<ncclDataType_t>& dataTypes, const std::vector<ncclDataType_t>& testDataTypes)
  {
    // Filter out any unsupported datatypes, in case only subset has been compiled for
    auto& supportedDataTypes = ev.GetAllSupportedDataTypes();
    for (auto dt : testDataTypes)
    {
      for (int i = 0; i < supportedDataTypes.size(); ++i)
      {
        if (supportedDataTypes[i] == dt)
        {
          dataTypes.push_back(dt);
          break;
        }
      }
    }
  }

  std::vector<int> const TestBed::GetNumCollsPerGroup(int numCollectivesInGroup,
                                                       int numGroupCalls)
  {
    return std::vector<int>(numGroupCalls, numCollectivesInGroup);
  }

  std::vector<int> const TestBed::GetNumStreamsPerGroup(int numStreamsPerGroup,
                                                         int numGroupCalls)
  {
    return std::vector<int>(numGroupCalls, numStreamsPerGroup);
  }

  std::vector<std::vector<int>> TestBed::GetDeviceIdsList(int const numProcesses,
                                                          int const numGpus,
                                                          const std::vector<int>& gpuPriorityOrder)
  {
    return GetDeviceIdsList(numProcesses, numGpus, 1, gpuPriorityOrder);
  }

  std::vector<std::vector<int>> TestBed::GetDeviceIdsList(int const numProcesses,
                                                          int const numGpus,
                                                          int const ranksPerGpu,
                                                          const std::vector<int>& gpuPriorityOrder)
  {
    std::vector<std::vector<int>> result(numProcesses);
    int ntasks = numProcesses == 1 ? numGpus : 1;
    int k=0;
    for (int i = 0; i < numProcesses; i++)
      for (int j = 0; j < ntasks * ranksPerGpu; j++) {
        result[i].push_back(gpuPriorityOrder[k%numGpus]);
        k++;
      }
    return result;
  }

  std::string TestBed::GetTestCaseName(int            const totalRanks,
                                       bool           const isMultiProcess,
                                       ncclFunc_t     const funcType,
                                       ncclDataType_t const dataType,
                                       ncclRedOp_t    const redOp,
                                       int            const root,
                                       bool           const inPlace,
                                       bool           const managedMem,
                                       bool           const useHipGraph,
                                       int            const ranksPerProc)
  {
    std::stringstream ss;
    ss << (isMultiProcess ? "MP" : "SP") <<  " ";
    ss << totalRanks;
    if (ranksPerProc > 1)
      ss << "(" << ranksPerProc << ") ";
    else
      ss << "    ";
    ss << "ranks ";
    ss << std::setfill(' ') << std::setw(20) << ncclFuncNames[funcType] << " ";
    ss << "(" << (inPlace ? "IP" : "OP") << ","
       << (managedMem ? "MM" : "GM") << ","
       << (useHipGraph ? "GL" : "NL") <<") ";
    ss << std::setfill(' ') << std::setw(15) << ncclDataTypeNames[dataType] << " ";
    if (CollectiveArgs::UsesReduce(funcType)) ss << std::setfill(' ') << std::setw(7) << ncclRedOpNames[redOp] << " ";
    if (CollectiveArgs::UsesRoot(funcType)) ss << "Root " << root << " ";
    return ss.str();
  }

  void TestBed::RunSimpleSweep(std::vector<ncclFunc_t>     const& funcTypes,
                               std::vector<ncclDataType_t> const& tmpDataTypes,
                               std::vector<ncclRedOp_t>    const& tmpRedOps,
                               std::vector<int>            const& roots,
                               std::vector<int>            const& numElements,
                               std::vector<bool>           const& inPlaceList,
                               std::vector<bool>           const& managedMemList,
                               std::vector<bool>           const& useHipGraphList,
                               bool                        const& enableSweep,
                               MemAllocType                memAllocType)
  {
    // Sort numElements in descending order to cut down on # of allocations
    std::vector<int> sortedN = numElements;
    std::sort(sortedN.rbegin(), sortedN.rend());
    OptionalColArgs optionalArgs;
    std::vector<ncclDataType_t> dataTypes;
    this->GetSupportedDataTypes(dataTypes, tmpDataTypes);
    if (dataTypes.empty()) {
      GTEST_SKIP() << "Skipping... test datatypes excluded by UT_DATATYPES.";
    }

    std::vector<ncclRedOp_t> redOps;
    this->GetSupportedRedOps(redOps, tmpRedOps);
    if (redOps.empty()) {
      GTEST_SKIP() << "Skipping... test reduction operations excluded by UT_REDOPS.";
    }

    if (memAllocType == MEM_ALLOC_SYMMETRIC_WIN &&
        std::find(managedMemList.begin(), managedMemList.end(), false) == managedMemList.end()) {
      GTEST_SKIP() << "Skipping... symmetric window allocation does not support managed memory.";
    }

    bool isCorrect = true;
    int numSymmetricConfigsRun = 0;
    int numSymmetricConfigsSkipped = 0;

    // Sweep over the number of ranks: either single process all GPUs, or 1 process per GPU
    struct RankConfig { int numGpus; int isMultiProcess; int ranksPerGpu; };
    std::vector<RankConfig> rankConfigs;
    for (int numGpus : ev.GetNumGpusList())
    for (int isMultiProcess : ev.GetIsMultiProcessList())
    for (int ranksPerGpu=1; ranksPerGpu <= ev.maxRanksPerGpu; ++ranksPerGpu)
    {
      if(enableSweep == false && (numGpus < 8 || numGpus*ranksPerGpu < 8)) {
        continue;
      }
      rankConfigs.push_back({numGpus, isMultiProcess, ranksPerGpu});
    }
    const std::vector<int>& gpuPriorityOrder = ev.GetGpuPriorityOrder();
    auto deviceIdsFor = [&](RankConfig const& rc) {
      return this->GetDeviceIdsList(rc.isMultiProcess ? rc.numGpus : 1, rc.numGpus, rc.ranksPerGpu, gpuPriorityOrder);
    };

    for (size_t rcIdx = 0; rcIdx < rankConfigs.size() && isCorrect; ++rcIdx)
    {
      int const numGpus        = rankConfigs[rcIdx].numGpus;
      int const isMultiProcess = rankConfigs[rcIdx].isMultiProcess;
      int const ranksPerGpu    = rankConfigs[rcIdx].ranksPerGpu;
      int const numRanks       = numGpus*ranksPerGpu;
      this->InitComms(deviceIdsFor(rankConfigs[rcIdx]),1,1,1,true,memAllocType);
      if (testing::Test::HasFailure())
      {
        isCorrect = false;
        this->DestroyComms();
        continue;
      }

      // Two configs ahead: the next one alone (often single-process on warm
      // worker 0) is too short to hide a worker's ~2 s device-code load.
      for (size_t ahead = rcIdx + 1; ahead < rankConfigs.size() && ahead <= rcIdx + 2; ++ahead)
      {
        this->WarmPoolWorkers(deviceIdsFor(rankConfigs[ahead]));
      }

      // Symmetric sweeps only cover configurations where RCCL can select symmetric kernels
      if (memAllocType == MEM_ALLOC_SYMMETRIC_WIN)
      {
        bool isSymmetricSupported = false;
        this->QuerySymmetricSupport(isSymmetricSupported);
        if (testing::Test::HasFailure())
        {
          isCorrect = false;
          this->DestroyComms();
          continue;
        }
        if (!isSymmetricSupported)
        {
          if (ev.verbose)
            TEST_INFO("Skipping %d ranks (%s): symmetric kernels unsupported on this comm",
                      numRanks, isMultiProcess ? "multi-process" : "single-process");
          ++numSymmetricConfigsSkipped;
          this->DestroyComms();
          continue;
        }
        ++numSymmetricConfigsRun;
      }

      for (int ftIdx = 0; ftIdx < funcTypes.size()      && isCorrect; ++ftIdx)
      for (int dtIdx = 0; dtIdx < dataTypes.size()      && isCorrect; ++dtIdx)
      {
      //Skipping AllReduce FP8 test on 9 to 16 ranks (gfx90a).
      if(ev.isGfx90 && numRanks > 8 && funcTypes[ftIdx] == ncclCollAllReduce
                    && (dataTypes[dtIdx] == ncclFloat8e4m3
                    || dataTypes[dtIdx] == ncclFloat8e5m2))
      {
            continue;
      }
      for (int rdIdx = 0; rdIdx < redOps.size()         && isCorrect; ++rdIdx)
      for (int rtIdx = 0; rtIdx < roots.size()          && isCorrect; ++rtIdx)
      for (int ipIdx = 0; ipIdx < inPlaceList.size()    && isCorrect; ++ipIdx)
      for (int mmIdx = 0; mmIdx < managedMemList.size() && isCorrect; ++mmIdx)
      {
        //  GUARD: Symmetric Memory is incompatible with Managed memory
        if (memAllocType == MEM_ALLOC_SYMMETRIC_WIN && managedMemList[mmIdx])
        {
          continue;
        }
        for (int neIdx = 0; neIdx < numElements.size() && isCorrect; ++neIdx)
        {
          int numInputElements, numOutputElements;
          CollectiveArgs::GetNumElementsForFuncType(funcTypes[ftIdx],
                                                    sortedN[neIdx],
                                                    numRanks,
                                                    &numInputElements,
                                                    &numOutputElements);
          optionalArgs.redOp = redOps[rdIdx];
          optionalArgs.root = roots[rtIdx] % numRanks;
          // Set biasNumElements if bias is enabled
          if (optionalArgs.useBias)
          {
            optionalArgs.biasNumElements = numOutputElements;
          }
          this->SetCollectiveArgs(funcTypes[ftIdx],
                                  dataTypes[dtIdx],
                                  numInputElements,
                                  numOutputElements,
                                  optionalArgs);
          if (testing::Test::HasFailure())
          {
            isCorrect = false;
            continue;
          }

          // Only allocate once for largest size
          if (neIdx == 0)
          {
            this->AllocateMem(inPlaceList[ipIdx], managedMemList[mmIdx]);
            if (testing::Test::HasFailure())
            {
              isCorrect = false;
              continue;
            }
          }

          for (int hgIdx = 0; hgIdx < useHipGraphList.size() && isCorrect; ++hgIdx)
          {
            // Re-prepare every sub-case, even when a prefix of the previous expected values is
            // still valid: PrepareData also clears the output, so a smaller collective that
            // writes nothing cannot pass on the output a larger run left behind.
            this->PrepareData();
            if (testing::Test::HasFailure())
            {
              isCorrect = false;
              continue;
            }

            std::string name = this->GetTestCaseName(numGpus, isMultiProcess,
                                                     funcTypes[ftIdx], dataTypes[dtIdx],
                                                     redOps[rdIdx], roots[rtIdx],
                                                     inPlaceList[ipIdx], managedMemList[mmIdx],
                                                     useHipGraphList[hgIdx], ranksPerGpu);

            if (ev.showNames)
            {
              TEST_INFO("%s [%9d elements]", name.c_str(), numInputElements);
            }

            std::vector<int> currentRanksEmpty = {};
            this->ExecuteCollectives(currentRanksEmpty, /*all groups*/ -1, useHipGraphList[hgIdx]);
            if (useHipGraphList[hgIdx]) {
              this->LaunchGraphs();
              this->DestroyGraphs();
            }
            if (testing::Test::HasFailure())
            {
              isCorrect = false;
              continue;
            }
            this->ValidateResults(isCorrect);
            if (!isCorrect)
            {
              TEST_ERROR("Incorrect output for %s", name.c_str());
            }
          }
        }
        this->DeallocateMem();
        if (testing::Test::HasFailure())
        {
          isCorrect = false;
        }
      }
    }
      this->DestroyComms();
    }

    if (numSymmetricConfigsSkipped > 0 && !testing::Test::HasFailure())
    {
      if (numSymmetricConfigsRun == 0)
      {
        GTEST_SKIP() << "Skipping... symmetric kernels unsupported for all "
                     << numSymmetricConfigsSkipped << " rank configurations (comm->symmetricSupport == 0).";
      }
      TEST_INFO("Symmetric sweep ran %d rank configurations, skipped %d without symmetric kernel support",
                numSymmetricConfigsRun, numSymmetricConfigsSkipped);
    }
  }

  void TestBed::InteractiveWait(std::string message)
  {
    if (ev.useInteractive)
    {
      TEST_INFO("%s", message.c_str());
      TEST_INFO("<Hit any key to continue>");
      scanf("%*c");
    }
  }

  int& TestBed::NumTestsRun()
  {
    static int numTestsRun = 0;
    return numTestsRun;
  }

}

#undef PIPE_WRITE
#undef PIPE_CHECK
