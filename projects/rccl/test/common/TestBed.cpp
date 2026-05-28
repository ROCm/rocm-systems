/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include <unistd.h>
#include "TestBed.hpp"
#include <rccl/rccl.h>

#define PIPE_WRITE(childId, val)                                        \
  ASSERT_EQ(write(childList[childId]->parentWriteFd, &val, sizeof(val)), sizeof(val))


#define PIPE_READ(childId, val)                                                         \
  {                                                                                     \
    if (ev.verbose) INFO("Calling PIPE_READ to Child %d\n", childId); \
    ssize_t retval = read(childList[childId]->parentReadFd, &val, sizeof(val)); \
    if (ev.verbose) INFO("Got PIPE_READ %ld from Child %d\n", retval, childId); \
    if (retval == -1)                                                                   \
    {                                                                                   \
      ERROR("Unable to read from child %d: Error %s\n", childId, strerror(errno));      \
      FAIL();                                                                           \
    }                                                                                   \
    else if (retval == 0)                                                               \
    {                                                                                   \
      ERROR("Child %d pipe closed unexpectedly\n", childId);                            \
      exit(1);                                                                          \
    }                                                                                   \
    else if (retval < sizeof(int))                                                      \
    {                                                                                   \
      ERROR("Child %d pipe read incomplete (%ld / %lu)\n", childId, retval, sizeof(val)); \
      exit(1);                                                                          \
    }                                                                                   \
  }

#define PIPE_CHECK(childId)                         \
  {                                                 \
    int response = 0;                               \
    PIPE_READ(childId, response);                   \
    if (response != TEST_SUCCESS)                   \
    {                                               \
      ERROR("Child %d reports failure\n", childId); \
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
    // Collect the number of GPUs
    this->numDevicesAvailable = ev.maxGpus;
    if (ev.verbose) INFO("Detected %d GPUs\n", this->numDevicesAvailable);
  }

  void TestBed::InitComms(std::vector<std::vector<int>> const& deviceIdsPerProcess,
                          std::vector<int>              const& numCollectivesInGroup,
                          std::vector<int>              const& numStreamsPerGroup,
                          int                           const  numGroupCalls,
                          bool                          const  useBlocking)
  {
    InteractiveWait("Starting InitComms");

    // Count up the total number of GPUs to use and track child/deviceId per rank
    this->numActiveChildren = deviceIdsPerProcess.size();
    this->numActiveRanks = 0;
    this->numGroupCalls = numGroupCalls;
    this->numCollectivesInGroup = numCollectivesInGroup;
    this->useBlocking = useBlocking;
    this->numStreamsPerGroup = numStreamsPerGroup;
    this->rankToChildMap.clear();
    this->rankToDeviceMap.clear();
    if (ev.verbose) INFO("Setting up %d active child processes\n", this->numActiveChildren);

    for (int childId = 0; childId < this->numActiveChildren; ++childId)
    {
      for (auto i = 0; i < deviceIdsPerProcess[childId].size(); ++i)
      {
        this->rankToChildMap.push_back(childId);
        this->rankToDeviceMap.push_back(deviceIdsPerProcess[childId][i]);
        ++this->numActiveRanks;
      }
    }

    // Check that no children currently exist
    if (childList.size() > 0)
    {
      ERROR("DestroyComms must be called prior to subsequent call to InitComms\n");
      return;
    }

    // Create child-processes
    childList.resize(this->numActiveChildren);
    for (int childId = 0; childId < this->numActiveChildren; ++childId)
    {
      childList[childId] = new TestBedChild(childId, ev.verbose, ev.printValues, ev.useMultithreading);
      if (childList[childId]->InitPipes() != TEST_SUCCESS)
      {
        ERROR("Unable to create pipes to child process\n");
        return;
      }

      pid_t pid = fork();
      if (pid == 0)
      {
        // Child process enters execution loop
        childList[childId]->StartExecutionLoop();
        return;
      }
      else
      {
        // Parent records child process ID and closes unused ends of pipe
        childList[childId]->pid = pid;
        close(childList[childId]->childWriteFd);
        close(childList[childId]->childReadFd);
      }
    }

    // If debugging is enabled, pause here to allow users to attach debugger
    if (ev.debugPause) {
      INFO("============================================================\n");
      INFO(" Pausing for debug attach: (e.g. sudo rocgdb -p <PID>)\n");
      INFO("============================================================\n");
      for (int childId = 0; childId < this->numActiveChildren; ++childId) {
        INFO(" Child %02d: processID: %d\n", childId, childList[childId]->pid);
      }
      INFO("============================================================\n");
      INFO("<Press enter to continue>\n");
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
      if (ev.verbose) INFO("Sending InitComm event to child %d\n", childId);
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
      PIPE_WRITE(childId, numCollectivesInGroup);

      // Send the RCCL communication with blocking or non-blocking option
      PIPE_WRITE(childId, useBlocking);

      // Send whether to use MultiRank interfaces or not.
      PIPE_WRITE(childId, useMulti);

      // Send how many streams to use per group call
      PIPE_WRITE(childId, numStreamsPerGroup);

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
                          int const numCollectivesInGroup, int const numStreamsPerGroup, int const numGroupCalls, bool const useBlocking)
  {
    InitComms(deviceIdsPerProcess, TestBed::GetNumCollsPerGroup(numCollectivesInGroup, numGroupCalls), TestBed::GetNumStreamsPerGroup(numStreamsPerGroup, numGroupCalls), numGroupCalls, useBlocking);
  }

  void TestBed::InitComms(int const numGpus, int const numCollectivesInGroup, int const numStreamsPerGroup, int const numGroupCalls, bool const useBlocking)
  {
     const std::vector<int>& gpuPriorityOrder = ev.GetGpuPriorityOrder();
     InitComms(GetDeviceIdsList(1, numGpus, gpuPriorityOrder), TestBed::GetNumCollsPerGroup(numCollectivesInGroup, numGroupCalls), TestBed::GetNumStreamsPerGroup(numStreamsPerGroup, numGroupCalls), numGroupCalls, useBlocking);
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
      ERROR("StreamIdx for group %d collective %d is out of bounds (%d/%d):\n", groupId, collId, streamIdx, numStreamsPerGroup[groupId]);
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

  void TestBed::AllocateMem(bool   const inPlace,
                            bool   const useManagedMem,
                            int    const groupId,
                            int    const collId,
                            int    const rank,
                            bool   const userRegistered)
  {
    InteractiveWait("Starting AllocateMem");

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
        PIPE_CHECK(childId);
      }
    }
    InteractiveWait("Finishing AllocateMem");
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
    }

    // Wait for child acknowledgement
    for (int childId = 0; childId < this->numActiveChildren; ++childId)
    {
      if ((currentRanks.size() == 0) || (ranksPerChild[childId].size() > 0)) PIPE_CHECK(childId);
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

    int const cmd = TestBedChild::CHILD_DEALLOCATE_MEM;

    for (auto currGroup : groupList)
    {
      for (auto currRank : rankList)
      {
        int const childId = rankToChildMap[currRank];
        PIPE_WRITE(childId, cmd);
        PIPE_WRITE(childId, currRank);
        PIPE_WRITE(childId, currGroup);
        PIPE_WRITE(childId, collId);
        PIPE_CHECK(childId);
      }
    }

    InteractiveWait("Finishing DeallocateMem");
  }

  void TestBed::DestroyComms()
  {
    InteractiveWait("Starting DestroyComms");

    int const cmd = TestBedChild::CHILD_DESTROY_COMMS;
    for (int childId = 0; childId < this->numActiveChildren; ++childId)
    {
      // Send DestroyComms command to each active child process
      PIPE_WRITE(childId, cmd);

      // Wait for child acknowledgement
      PIPE_CHECK(childId);
    }

    // Close any open child processes
    Finalize();

    InteractiveWait("Finishing DestroyComms");
  }

  void TestBed::DestroyGraphs()
  {
    InteractiveWait("Starting DestroyGraphs");

    int const cmd = TestBedChild::CHILD_DESTROY_GRAPHS;
    for (int currGroup = 0; currGroup < this->numGroupCalls; ++currGroup)
    {
      for (int childId = 0; childId < this->numActiveChildren; ++childId)
      {
        // Send DestroyGraphs command to each active child process
        PIPE_WRITE(childId, cmd);
        PIPE_WRITE(childId, currGroup);

        // Wait for child acknowledgement
        PIPE_CHECK(childId);
      }
    }

    InteractiveWait("Finishing DestroyGraphs");
  }

  void TestBed::Finalize()
  {
    if (this->numActiveChildren == 0)
      return;

    InteractiveWait("Starting Finalize");

    // Send Stop to all child processes
    int const cmd = TestBedChild::CHILD_STOP;
    for (int childId = 0; childId < this->numActiveChildren; ++childId)
    {
      PIPE_WRITE(childId, cmd);

      // Close pipes to child process
      close(childList[childId]->parentWriteFd);
      close(childList[childId]->parentReadFd);
    }

    // Wait for processes to stop
    for (int childId = 0; childId < this->numActiveChildren; ++childId)
    {
      int returnVal = 0;
      waitpid(childList[childId]->pid, &returnVal, 0);
      if (returnVal != 0)
      {
        ERROR("Child process %d exited with code %d\n", childId, returnVal);
      }
      delete(childList[childId]);
    }

    childList.clear();

    // Reset bookkeeping
    this->numActiveChildren = 0;
    this->numActiveRanks = 0;

    InteractiveWait("Finishing Finalize");
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
                               bool                        const& enableSweep)
  {
    // Sort numElements in descending order to cut down on # of allocations
    std::vector<int> sortedN = numElements;
    std::sort(sortedN.rbegin(), sortedN.rend());
    OptionalColArgs optionalArgs;
    // Filter out any unsupported datatypes, in case only subset has been compiled for
    std::vector<ncclDataType_t> const& supportedDataTypes = this->GetAllSupportedDataTypes();
    std::vector<ncclDataType_t> dataTypes;
    for (auto dt : tmpDataTypes)
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

    // Filter out any unsupported reduction ops, in case only subset has been compiled for
    std::vector<ncclRedOp_t> const& supportedOps = this->GetAllSupportedRedOps();
    std::vector<ncclRedOp_t> redOps;
    for (auto redop : tmpRedOps)
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
      this->InitComms(this->GetDeviceIdsList(numChildren, numGpus, ranksPerGpu, gpuPriorityOrder));
      if (testing::Test::HasFailure())
      {
        isCorrect = false;
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
              INFO("%s [%9d elements]\n", name.c_str(), numInputElements);
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
              ERROR("Incorrect output for %s\n", name.c_str());
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
      INFO("%s\n", message.c_str());
      INFO("<Hit any key to continue>\n");
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
