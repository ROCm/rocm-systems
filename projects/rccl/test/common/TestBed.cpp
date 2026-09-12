/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include <fcntl.h>
#include <csignal>
#include <unistd.h>
#include "TestBed.hpp"
#include <rccl/rccl.h>


/**
 * @brief Reads exactly 'count' bytes from a file descriptor, handling partial 
 * reads and signal interruptions (EINTR).
 * @return 'count' on success, or -1 on error / premature EOF.
 */
inline ssize_t safe_pipe_read(int fd, void* buf, size_t count) {
  char* ptr = static_cast<char*>(buf);
  size_t bytesLeft = count;

  while (bytesLeft > 0) {
    ssize_t const bytesRead = read(fd, ptr, bytesLeft);
    if (bytesRead < 0) {
      if (errno == EINTR) continue; // Interrupted by OS signal, retry
      return -1;                    // Read error
    }
    if (bytesRead == 0) {
      return -1;                    // EOF: Pipe closed prematurely
    }
    ptr += bytesRead;
    bytesLeft -= bytesRead;
  }
  return static_cast<ssize_t>(count); // Successfully read all requested bytes
}

/**
 * @brief Writes exactly 'count' bytes to a file descriptor, handling partial 
 * writes and signal interruptions (EINTR).
 * @return 'count' on success, or -1 on error.
 */
inline ssize_t safe_pipe_write(int fd, const void* buf, size_t count) {
  const char* ptr = static_cast<const char*>(buf);
  size_t bytesLeft = count;

  while (bytesLeft > 0) {
    ssize_t const bytesWritten = write(fd, ptr, bytesLeft);
    if (bytesWritten < 0) {
      if (errno == EINTR) continue; // Interrupted by OS signal, retry
      return -1;                    // Write error
    }
    ptr += bytesWritten;
    bytesLeft -= bytesWritten;
  }
  return static_cast<ssize_t>(count); // Successfully wrote all requested bytes
}

#define PIPE_WRITE(childId, val)                                        \
  ASSERT_EQ(safe_pipe_write(childList[childId]->parentWriteFd, &val, sizeof(val)), sizeof(val))


#define PIPE_READ(childId, val)                                                         \
  {                                                                                     \
    if (ev.verbose) TEST_INFO("Calling PIPE_READ to Child %d", childId); \
    ssize_t retval = safe_pipe_read(childList[childId]->parentReadFd, &val, sizeof(val)); \
    if (ev.verbose) TEST_INFO("Got PIPE_READ %ld from Child %d", retval, childId); \
    if (retval == -1)                                                                   \
    {                                                                                   \
      TEST_ERROR("Unable to read from child %d: Error %s", childId, strerror(errno));      \
      FAIL();                                                                           \
    }                                                                                   \
    else if (retval == 0)                                                               \
    {                                                                                   \
      TEST_ERROR("Child %d pipe closed unexpectedly", childId);                            \
      exit(1);                                                                          \
    }                                                                                   \
    else if (retval < sizeof(int))                                                      \
    {                                                                                   \
      TEST_ERROR("Child %d pipe read incomplete (%ld / %lu)", childId, retval, sizeof(val)); \
      exit(1);                                                                          \
    }                                                                                   \
  }

#define PIPE_CHECK(childId)                         \
  {                                                 \
    int response = 0;                               \
    PIPE_READ(childId, response);                   \
    if (response != TEST_SUCCESS)                   \
    {                                               \
      TEST_ERROR("Child %d reports failure", childId); \
      ASSERT_EQ(response, TEST_SUCCESS);            \
      FAIL();                                       \
    }                                               \
  }

namespace RcclUnitTesting
{
  TestBed::TestBed() :
    numDevicesAvailable(0),
    numActiveChildren(0),
    numActiveRanks(0)
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
    this->configUsedPool = false;
  }

  void TestBed::InitComms(std::vector<std::vector<int>> const& deviceIdsPerProcess,
                          std::vector<int>              const& numCollectivesInGroup,
                          std::vector<int>              const& numStreamsPerGroup,
                          int                           const  numGroupCalls,
                          bool                          const  useBlocking,
                          MemAllocType                  const  memAllocType)
  {
    InteractiveWait("Starting InitComms");

    // If children/comms from a previous test mode (e.g. SP) are still active,
    // ensure they are cleanly stopped before creating new ones for MP mode
    if (this->numActiveChildren > 0)
    {
      this->DestroyComms();
      for (int i = 0; i < this->numActiveChildren; ++i)
      {
        this->StopChild(i); // Gracefully stops and waits for child
      }
      this->numActiveChildren = 0;
    }
    // Count up the total number of GPUs to use and track child/deviceId per rank
    this->numActiveChildren = deviceIdsPerProcess.size();
    this->numActiveRanks = 0;
    this->numGroupCalls = numGroupCalls;
    this->numCollectivesInGroup = numCollectivesInGroup;
    this->useBlocking = useBlocking;
    this->numStreamsPerGroup = numStreamsPerGroup;
    this->rankToChildMap.clear();
    this->rankToDeviceMap.clear();
    if (ev.verbose) TEST_INFO("Setting up %d active child processes", this->numActiveChildren);

    for (int childId = 0; childId < this->numActiveChildren; ++childId)
    {
      for (auto i = 0; i < deviceIdsPerProcess[childId].size(); ++i)
      {
        this->rankToChildMap.push_back(childId);
        this->rankToDeviceMap.push_back(deviceIdsPerProcess[childId][i]);
        ++this->numActiveRanks;
      }
    }

    // Guards both paths: the pool-reuse branch below would silently overwrite
    // a non-empty childList.
    if (childList.size() > 0)
    {
      FAIL() << "DestroyComms must be called prior to subsequent call to InitComms";
    }

    // Comm pool (UT_COMM_POOL): worker d is pinned to device d and keeps its
    // device-code object resident, so reuse skips the ~15-30s load per config.
    this->configUsedPool = false;
    if (this->poolMode)
    {
      // Each worker snapshots env at fork and NCCL_PARAM caches it; a test that
      // changes env must call Finalize() to re-fork the pool.
      if (this->poolChildren.empty())
      {
        // CRITICAL: no HIP call in the parent before fork -- HIP state does not
        // survive fork() and workers SEGV. Use ev.GetNumDetectedGpus() instead.
        int poolSize = this->numDevicesAvailable;
        int const detectedGpus = ev.GetNumDetectedGpus();
        if (detectedGpus > 0 && detectedGpus < poolSize)
        {
          poolSize = detectedGpus;
        }
        this->poolChildren.assign(poolSize, nullptr);
        for (int d = 0; d < poolSize; ++d)
        {
          this->poolChildren[d] = new TestBedChild(d, ev.verbose, ev.printValues, ev.useMultithreading);
          if (this->poolChildren[d]->InitPipes() != TEST_SUCCESS)
          {
            // Reap the half-built pool; FAIL() (not TEST_ERROR) so the sweep
            // stops instead of indexing an empty childList -> SEGV.
            TeardownPool();
            FAIL() << "Unable to create pipes to pool child process " << d;
          }
          pid_t pid = fork();
          if (pid == 0)
          {
            this->poolChildren[d]->StartExecutionLoop();
            return;
          }
          if (pid < 0)
          {
            TeardownPool();
            FAIL() << "fork() failed for pool child process " << d;
          }
          this->poolChildren[d]->pid = pid;
          close(this->poolChildren[d]->childWriteFd);
          close(this->poolChildren[d]->childReadFd);
        }
        // Do NOT pre-warm workers: under the runner's --jobs N it storms
        // ncclCommInitAll across all GPUs and fails intermittently.
      }
      

      // Map this config's children onto distinct pool workers by representative device.
      bool mappable = true;
      std::vector<TestBedChild*> mapped(this->numActiveChildren, nullptr);
      std::set<int> usedWorkers;
      for (int c = 0; c < this->numActiveChildren && mappable; ++c)
      {
        int const worker = deviceIdsPerProcess[c].empty() ? -1 : deviceIdsPerProcess[c][0];
        if (worker < 0 || worker >= (int)this->poolChildren.size() ||
            this->poolChildren[worker] == nullptr || usedWorkers.count(worker))
        {
          mappable = false;  // out-of-range, missing worker, or collision (multi-rank-per-GPU) -> fork-fresh
          break;
        }
        usedWorkers.insert(worker);
        mapped[c] = this->poolChildren[worker];
      }

      if (mappable)
      {
        // Reuse: childList borrows the selected pool workers (pool retains ownership).
        childList = mapped;
        this->configUsedPool = true;
      }
    }

    if (!this->configUsedPool)
    {
      // ---- Classic fork-fresh path (pool disabled, or an unmappable config) ----
      // (The "DestroyComms must precede InitComms" guard is hoisted above, covering both paths.)
      childList.resize(this->numActiveChildren);
      for (int childId = 0; childId < this->numActiveChildren; ++childId)
      {
        childList[childId] = new TestBedChild(childId, ev.verbose, ev.printValues, ev.useMultithreading);
        if (childList[childId]->InitPipes() != TEST_SUCCESS)
        {
          TEST_ERROR("Unable to create pipes to child process %d", childId);
          return;
        }

        // Ensure child-side pipe descriptors remain open across execl()
        fcntl(childList[childId]->childReadFd, F_SETFD, 0);
        fcntl(childList[childId]->childWriteFd, F_SETFD, 0);

        pid_t pid = fork();
        if (pid == 0)
        {
          // Child process enters execution loop
          close(childList[childId]->parentWriteFd);
          close(childList[childId]->parentReadFd);

          // String arguments for execl
          std::string sChildId      = std::to_string(childId);
          std::string sChildReadFd  = std::to_string(childList[childId]->childReadFd);
          std::string sChildWriteFd = std::to_string(childList[childId]->childWriteFd);
          std::string sVerbose      = std::to_string(ev.verbose ? 1 : 0);
          std::string sPrintVal     = std::to_string(ev.printValues);
          std::string sThreading    = std::to_string(ev.useMultithreading ? 1 : 0);
          std::string sMemAllocType = std::to_string(static_cast<int>(memAllocType));

          // Re-execute binary to clear inherited HIP/HSA driver state
          execl("/proc/self/exe", "rccl_unit_test",
                "--child",
                sChildId.c_str(),
                sChildReadFd.c_str(),
                sChildWriteFd.c_str(),
                sVerbose.c_str(),
                sPrintVal.c_str(),
                sThreading.c_str(),
                sMemAllocType.c_str(),
                NULL);
          perror("execl failed");
          _exit(1);
        }
        else if (pid > 0)
        {
          // Parent records child process ID and closes unused ends of pipe
          childList[childId]->pid = pid;
          close(childList[childId]->childWriteFd);
          close(childList[childId]->childReadFd);
        }
        else 
        {
          TEST_ERROR("fork() failed when creating child process %d", childId);
        }
      }
    }

    // If debugging is enabled, pause here to allow users to attach debugger
    if (ev.debugPause) {
      TEST_INFO("============================================================");
      TEST_INFO(" Pausing for debug attach: (e.g. sudo rocgdb -p <PID>)");
      TEST_INFO("============================================================");
      for (int childId = 0; childId < this->numActiveChildren; ++childId) {
        TEST_INFO(" Child %02d: processID: %d", childId, childList[childId]->pid);
      }
      TEST_INFO("============================================================");
      TEST_INFO("<Press enter to continue>");
      scanf("%*c");
    }

    // Determine number of unique GPUs being used.
    std::set<int> unique_devices;
    for (auto a:  this->rankToDeviceMap)
      unique_devices.insert(a);
    bool useMulti = unique_devices.size() < this->rankToDeviceMap.size() ? true : false;

    // Tell first rank to get ncclUniqueId
    int getIdCmd = TestBedChild::CHILD_GET_UNIQUE_ID;
    PIPE_WRITE(0, getIdCmd);
    PIPE_CHECK(0);

    // Receive back unique ID from first rank
    ncclUniqueId id;
    PIPE_READ(0, id);

    // Send InitComms command to each active child process
    int const cmd = TestBedChild::CHILD_INIT_COMMS;
    int rankOffset = 0;
    for (int childId = 0; childId < this->numActiveChildren; ++childId)
    {
      if (ev.verbose) TEST_INFO("Sending InitComm event to child %d", childId);
      PIPE_WRITE(childId, cmd);

      // Send unique ID to child process
      PIPE_WRITE(childId, id);

      // Send total number of ranks to child process
      PIPE_WRITE(childId, this->numActiveRanks);

      // Send the rank offset for this child process
      PIPE_WRITE(childId, rankOffset);

      // Send the total number of group calls for this child process
      PIPE_WRITE(childId, numGroupCalls);

      // Send the number of collectives to be run per group call
      int const numCollSize = this->numCollectivesInGroup.size();
      PIPE_WRITE(childId, numCollSize);
      if (numCollSize > 0) {
        write(childList[childId]->parentWriteFd,
          this->numCollectivesInGroup.data(),
          numCollSize * sizeof(int));
      }

      // Send the RCCL communication with blocking or non-blocking option
      PIPE_WRITE(childId, useBlocking);

      // Send memAllocType to child process
      int const memAllocTypeVal = static_cast<int>(memAllocType);
      PIPE_WRITE(childId, memAllocTypeVal);

      // Send whether to use MultiRank interfaces or not.
      PIPE_WRITE(childId, useMulti);

      // Send how many streams to use per group call
      int const numStreamsSize = this->numStreamsPerGroup.size();
      PIPE_WRITE(childId, numStreamsSize);
      if (numStreamsSize > 0) {
        write(childList[childId]->parentWriteFd,
        this->numStreamsPerGroup.data(),
        numStreamsSize * sizeof(int));
      }

      // Send the GPUs this child uses
      int const numGpus = deviceIdsPerProcess[childId].size();
      PIPE_WRITE(childId, numGpus);
      for (int i = 0; i < numGpus; i++)
        PIPE_WRITE(childId, deviceIdsPerProcess[childId][i]);

      rankOffset += numGpus;
    }

    // Wait for child acknowledgement
    // This is done after previous loop to avoid deadlock as every rank needs to enter ncclInitCommRank
    for (int childId = 0; childId < this->numActiveChildren; ++childId)
    {
      PIPE_CHECK(childId);
    }
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
    // Build list of ranks this applies to (-1 for rank means to set for all)
    std::vector<int> rankList;
    for (int i = 0; i < this->numActiveRanks; ++i)
      if (rank == -1 || rank == i) rankList.push_back(i);

    if (streamIdx < 0 || streamIdx >= this->numStreamsPerGroup[groupId])
    {
      TEST_ERROR("StreamIdx for group %d collective %d is out of bounds (%d/%d):", groupId, collId, streamIdx, numStreamsPerGroup[groupId]);
      FAIL();
    }

    // Loop over all ranks and send CollectiveArgs to appropriate child process
    int const cmd = TestBedChild::CHILD_SET_COLL_ARGS;
    for (auto currRank : rankList)
    {
      int const childId = rankToChildMap[currRank];
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
      PIPE_CHECK(childId);
    }
    InteractiveWait("Finishing SetCollectiveArgs");
  }

  void TestBed::AllocateMemInternal(bool   const inPlace,
                                    bool   const useManagedMem,
                                    int    const groupId,
                                    int    const collId,
                                    int    const rank,
                                    bool   const userRegistered) 
  {
    InteractiveWait("Starting AllocateMemInternal");

    // Build list of ranks this applies to (-1 for rank means to set for all)
    std::vector<int> rankList;
    for (int i = 0; i < this->numActiveRanks; ++i)
      if (rank == -1 || rank == i) rankList.push_back(i);

    // Build list of groups this applies to (-1 for groupId means to set for all)
    std::vector<int> groupList;
    for (int i = 0; i < this->numGroupCalls; ++i)
      if (groupId == -1 || groupId == i) groupList.push_back(i);

    // Loop over all ranks and send allocation command to appropriate child process
    int const cmd = TestBedChild::CHILD_ALLOCATE_MEM;
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
      }
    }
    // Wait for all children to complete allocation before registering
    for (int childId = 0; childId < this->numActiveChildren; ++childId) {
      PIPE_CHECK(childId);
    }
    InteractiveWait("Finishing AllocateMemInternal");
  }

  void TestBed::RegisterMemInternal(int    const groupId,
                                    int    const collId,
                                    int    const rank)
  {
    InteractiveWait("Starting RegisterMemInternal");
    // Build list of ranks this applies to (-1 for rank means to set for all)
    std::vector<int> rankList;
    for (int i = 0; i < this->numActiveRanks; ++i)
      if (rank == -1 || rank == i) rankList.push_back(i);

    // Build list of groups this applies to (-1 for groupId means to set for all)
    std::vector<int> groupList;
    for (int i = 0; i < this->numGroupCalls; ++i)
      if (groupId == -1 || groupId == i) groupList.push_back(i);

    // Grouped RCCL Window Registration across all active children
    int const regCmd = TestBedChild::CHILD_REGISTER_MEM;
    for (auto currGroup : groupList) {
      // 1. Send CHILD_REGISTER_MEM command to ALL active children FIRST
      for (int childId = 0; childId < this->numActiveChildren; ++childId) {
        PIPE_WRITE(childId, regCmd);
        PIPE_WRITE(childId, currGroup);
        PIPE_WRITE(childId, collId);
      }
      // 2. THEN wait for all child ACKs after all children have entered registration
      for (int childId = 0; childId < this->numActiveChildren; ++childId) {
         PIPE_CHECK(childId);
      }
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
    this->AllocateMemInternal(inPlace,useManagedMem,groupId,collId,rank,userRegistered);
    this->RegisterMemInternal(groupId,collId,rank);
  }

  void TestBed::PrepareData(int         const groupId,
                            int         const collId,
                            int         const rank,
                            CollFuncPtr const prepDataFunc)
  {
    InteractiveWait("Starting PrepareData");
    // Build list of ranks this applies to (-1 for rank means to set for all)
    std::vector<int> rankList;
    for (int i = 0; i < this->numActiveRanks; ++i)
      if (rank == -1 || rank == i) rankList.push_back(i);

    // Build list of groups this applies to (-1 for groupId means to set for all)
    std::vector<int> groupList;
    for (int i = 0; i < this->numGroupCalls; ++i)
      if (groupId == -1 || groupId == i) groupList.push_back(i);

    // Loop over all ranks and send prepare data command to appropriate child process
    int const cmd = TestBedChild::CHILD_PREPARE_DATA;
    for (auto currGroup : groupList)
    {
      for (auto currRank : rankList)
      {
        int const childId = rankToChildMap[currRank];
        PIPE_WRITE(childId, cmd);
        PIPE_WRITE(childId, currRank);
        PIPE_WRITE(childId, currGroup);
        PIPE_WRITE(childId, collId);
        PIPE_WRITE(childId, prepDataFunc);
        PIPE_CHECK(childId);
      }
    }
    InteractiveWait("Finishing PrepareData");
  }

  void TestBed::ExecuteCollectives(std::vector<int> const &currentRanks, int const groupId,
                                   bool const useHipGraph)
  {
    InteractiveWait("Starting ExecuteCollectives");

    int const cmd = TestBedChild::CHILD_EXECUTE_COLL;
    ++TestBed::NumTestsRun();

    std::vector<std::vector<int>> ranksPerChild(this->numActiveChildren);
    for (int rank = 0; rank < currentRanks.size(); ++rank)
    {
      ranksPerChild[rankToChildMap[currentRanks[rank]]].push_back(rank);
    }

    // Build list of groups this applies to (-1 for groupId means to set for all)
    std::vector<int> groupList;
    for (int i = 0; i < this->numGroupCalls; ++i)
      if (groupId == -1 || groupId == i) groupList.push_back(i);

    for (auto currGroup : groupList) {
      // Send ExecuteColl command to each active child process
      for (int childId = 0; childId < this->numActiveChildren; ++childId)
      {
        if ((currentRanks.size() == 0) || (ranksPerChild[childId].size() > 0))
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
      }

      for (int childId = 0; childId < this->numActiveChildren; ++childId)
      {
        if ((currentRanks.size() == 0) || (ranksPerChild[childId].size() > 0)) PIPE_CHECK(childId);
      }
    }

    InteractiveWait("Finishing ExecuteCollectives");
  }

  void TestBed::ValidateResults(bool& isCorrect, int const groupId, int const collId, int const rank)
  {
    InteractiveWait("Starting ValidateResults");

    // Build list of ranks this applies to (-1 for rank means to set for all)
    std::vector<int> rankList;
    for (int i = 0; i < this->numActiveRanks; ++i)
      if (rank == -1 || rank == i) rankList.push_back(i);

    // Build list of groups this applies to (-1 for groupId means to set for all)
    std::vector<int> groupList;
    for (int i = 0; i < this->numGroupCalls; ++i)
      if (groupId == -1 || groupId == i) groupList.push_back(i);

    int const cmd = TestBedChild::CHILD_VALIDATE_RESULTS;

    isCorrect = true;
    for (auto currGroup : groupList)
    {
      // Send ValidateResults command to each active child process
      for (auto currRank : rankList)
      {
        int const childId = rankToChildMap[currRank];
        PIPE_WRITE(childId, cmd);
        PIPE_WRITE(childId, currRank);
        PIPE_WRITE(childId, currGroup);
        PIPE_WRITE(childId, collId);

        int response = 0;
        ASSERT_EQ(read(childList[childId]->parentReadFd, &response, sizeof(int)), sizeof(int));
        isCorrect &= (response == TEST_SUCCESS);
      }
    }

    ASSERT_EQ(isCorrect, true) << "Output does not match expected";

    InteractiveWait("Finishing ValidateResults");
  }

  void TestBed::LaunchGraphs(int const groupId)
  {
    InteractiveWait("Starting LaunchGraphs");

    // Build list of groups this applies to (-1 for groupId means to set for all)
    std::vector<int> groupList;
    for (int i = 0; i < this->numGroupCalls; ++i)
      if (groupId == -1 || groupId == i) groupList.push_back(i);

    int const cmd = TestBedChild::CHILD_LAUNCH_GRAPHS;
    for (auto currGroup : groupList)
    {
      for (int childId = 0; childId < this->numActiveChildren; ++childId)
      {
        // Send LaunchGraphs command to each active child process
        PIPE_WRITE(childId, cmd);
        PIPE_WRITE(childId, currGroup);

        // Wait for child acknowledgement
        PIPE_CHECK(childId);
      }
    }

    InteractiveWait("Finishing LaunchGraphs");
  }

  void TestBed::DeallocateMem(int const groupId, int const collId, int const rank)
  {
    InteractiveWait("Starting DeallocateMem");

    // Build list of ranks this applies to (-1 for rank means to set for all)
    std::vector<int> rankList;
    for (int i = 0; i < this->numActiveRanks; ++i)
      if (rank == -1 || rank == i) rankList.push_back(i);

    // Build list of groups this applies to (-1 for groupId means to set for all)
    std::vector<int> groupList;
    for (int i = 0; i < this->numGroupCalls; ++i)
      if (groupId == -1 || groupId == i) groupList.push_back(i);

    int const deallocCmd = TestBedChild::CHILD_DEALLOCATE_MEM;
    for (auto currGroup : groupList) {
      for (auto currRank : rankList)
      {
        int const childId = rankToChildMap[currRank];
        PIPE_WRITE(childId, deallocCmd);
        PIPE_WRITE(childId, currRank);
        PIPE_WRITE(childId, currGroup);
        PIPE_WRITE(childId, collId);
      }
    }

    // Wait for all physical deallocations to complete
    for (int childId = 0; childId < this->numActiveChildren; ++childId) {
      PIPE_CHECK(childId);
    }

    InteractiveWait("Finishing DeallocateMem");
  }

  void TestBed::DestroyComms()
  {
    InteractiveWait("Starting DestroyComms");

    int const cmd = TestBedChild::CHILD_DESTROY_COMMS;

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
      for (int childId = 0; childId < this->numActiveChildren; ++childId)
      {
        PIPE_WRITE(childId, cmd);
      }
      if (ev.verbose) waitStart = Clock::now();
      for (int childId = 0; childId < this->numActiveChildren; ++childId)
      {
        // Wait for child acknowledgement
        PIPE_CHECK(childId);
      }

      if (ev.verbose)
      {
        using std::chrono::duration_cast;
        using std::chrono::milliseconds;
        long long const sendMs = duration_cast<milliseconds>(waitStart - sendStart).count();
        long long const waitMs = duration_cast<milliseconds>(Clock::now() - waitStart).count();
        TEST_INFO("DestroyComms: %d children, send %lld ms, parallel teardown %lld ms",
                  this->numActiveChildren, sendMs, waitMs);
      }
    }();

    if (this->configUsedPool)
    {
      // Pool mode: comms/streams were freed on the workers, but keep the workers alive for
      // the next config (their device-code kernels stay resident). Drop the borrowed refs;
      // the pool keeps ownership. Finalize() (between sweeps / at teardown) reaps the pool.
      this->childList.clear();
      this->numActiveChildren = 0;
      this->numActiveRanks = 0;
    }
    else
    {
      // Classic / fork-fresh: reap the per-config children this call owns.
      TeardownOwnedChildList();
    }

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
      for (int childId = 0; childId < this->numActiveChildren; ++childId)
      {
        PIPE_WRITE(childId, cmd);
        PIPE_WRITE(childId, currGroup);
      }
      for (int childId = 0; childId < this->numActiveChildren; ++childId)
      {
        // Wait for child acknowledgement
        PIPE_CHECK(childId);
      }
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
    // Only reaps children that childList OWNS (classic / fork-fresh path). In pool mode
    // childList holds borrowed pool refs (cleared by DestroyComms), so this is a no-op.
    if (this->configUsedPool || this->numActiveChildren == 0)
      return;

    InteractiveWait("Starting Finalize");

    // Send Stop to all child processes
    int const cmd = TestBedChild::CHILD_STOP;
    for (int childId = 0; childId < this->numActiveChildren; ++childId)
    {
      PIPE_WRITE(childId, cmd);

      // Close pipes to child process
      close(childList[childId]->parentWriteFd);
    }

    // Wait for processes to stop
    for (int childId = 0; childId < this->numActiveChildren; ++childId)
    {
      int returnVal = 0;
      waitpid(childList[childId]->pid, &returnVal, 0);
      if (WIFSIGNALED(returnVal))
      {
        TEST_ERROR("Child process %d killed by signal %d", childId, WTERMSIG(returnVal));
      }
      else if (WEXITSTATUS(returnVal) != 0)
      {
        TEST_ERROR("Child process %d exited with code %d", childId, WEXITSTATUS(returnVal));
      }
      // Only close the read end AFTER the child process is dead
      close(childList[childId]->parentReadFd);
      delete(childList[childId]);
    }

    childList.clear();

    // Reset bookkeeping
    this->numActiveChildren = 0;
    this->numActiveRanks = 0;

    InteractiveWait("Finishing Finalize");
  }

  void TestBed::TeardownPool()
  {
    if (this->poolChildren.empty())
      return;

    // Best-effort stop (avoid the ASSERT-based PIPE_WRITE during teardown), then reap.
    int const cmd = TestBedChild::CHILD_STOP;
    for (TestBedChild* c : this->poolChildren)
    {
      if (!c)
      {
        continue;
      }
      // Only a forked worker (pid > 0) has a reader on the pipe; skip the STOP write for a
      // never-forked entry. close(-1) on an unopened fd is a harmless no-op.
      if (c->pid > 0)
      {
        ssize_t const w = write(c->parentWriteFd, &cmd, sizeof(cmd));
        (void)w;
      }
      close(c->parentWriteFd);
      close(c->parentReadFd);
    }
    for (TestBedChild* c : this->poolChildren)
    {
      if (!c)
      {
        continue;
      }
      // Reap only workers we actually forked; a never-forked entry has pid == -1.
      if (c->pid > 0)
      {
        int returnVal = 0;
        waitpid(c->pid, &returnVal, 0);
      }
      delete c;
    }
    this->poolChildren.clear();
    this->childList.clear();      // borrowed refs only; do not delete
    this->configUsedPool = false;
    this->numActiveChildren = 0;
    this->numActiveRanks = 0;
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

    bool isCorrect = true;

    // Sweep over the number of ranks
    for (int numGpus : ev.GetNumGpusList())
    for (int isMultiProcess : ev.GetIsMultiProcessList())
    for (int ranksPerGpu=1; ranksPerGpu <= ev.maxRanksPerGpu && isCorrect; ++ranksPerGpu)
    {
      // Test either single process all GPUs, or 1 process per GPU
      int const numChildren = isMultiProcess ? numGpus : 1;
      int const numRanks    = numGpus*ranksPerGpu;
      if(enableSweep == false && (numGpus < 8 || numRanks < 8)) {
        continue;
      }
      const std::vector<int>& gpuPriorityOrder = ev.GetGpuPriorityOrder();
      this->InitComms(this->GetDeviceIdsList(numChildren, numGpus, ranksPerGpu, gpuPriorityOrder),1,1,1,true,memAllocType);
      if (testing::Test::HasFailure())
      {
        isCorrect = false;
        this->DestroyComms();
        continue;
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
          optionalArgs.root = roots[rtIdx] % this->numActiveRanks;
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
            this->AllocateMem(inPlaceList[ipIdx], managedMemList[mmIdx],-1,-1,-1, (memAllocType == MEM_ALLOC_SYMMETRIC_WIN));
            if (testing::Test::HasFailure())
            {
              isCorrect = false;
              continue;
            }
          }

          for (int hgIdx = 0; hgIdx < useHipGraphList.size() && isCorrect; ++hgIdx)
          {
            // There are some cases when data does not need to be re-prepared
            // e.g. AllReduce subarray expected results are still valid
            bool canSkip = (neIdx != 0 && !inPlaceList[ipIdx] &&
                            (funcTypes[ftIdx] == ncclCollBroadcast ||
                             funcTypes[ftIdx] == ncclCollReduce    ||
                             funcTypes[ftIdx] == ncclCollAllReduce));
            if (!canSkip) this->PrepareData();
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
            // ====================================================================
            // DEBUG: Force GPU execution to complete & flush stdout print buffers
            // ====================================================================
            if (!isMultiProcess) {
              hipDeviceSynchronize(); 
              fflush(stdout);
            }
            this->ValidateResults(isCorrect);
            if (!isCorrect)
            {
              TEST_ERROR("Incorrect output for %s", name.c_str());
            }
          }
        }
        this->DeallocateMem();
      }
    }
      this->DestroyComms();
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

  void TestBed::StopChild(int const childId)
  {
    if (childId < 0 || childId >= this->childList.size()) return;
    TestBedChild* child = this->childList[childId];
    if (child != nullptr)
     {
      // 1. Send CHILD_STOP command to the child
       int const cmd = TestBedChild::CHILD_STOP;
       PIPE_WRITE(childId, cmd);
       PIPE_CHECK(childId);
       // 2. Wait for child process to exit cleanly before closing pipes
       int status;
       waitpid(child->pid, &status, 0); 
       // 3. Close pipes and delete object
       delete child;
       this->childList[childId] = nullptr;
     }
  }

}

#undef PIPE_WRITE
#undef PIPE_CHECK
