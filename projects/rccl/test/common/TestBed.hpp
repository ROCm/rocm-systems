/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once
#include <map>
#include "CollectiveArgs.hpp"
#include "TestBedChild.hpp"
#include "EnvVars.hpp"
#include <gtest/gtest.h>
#include <iomanip>
#include <chrono>

namespace RcclUnitTesting
{
  // This class facilitates testing RCCL collectives across various process / device configurations
  //
  class TestBed
  {
  public:
    int                        numDevicesAvailable;   // # of devices detected on node
    std::vector<TestBedChild*> childList;             // Active child processes (with usable RCCL comms)
    std::vector<int>           rankToChildMap;        // Child process of each active rank
    std::vector<int>           numCollectivesInGroup; // # of collectives to execute per group call
    std::vector<int>           numStreamsPerGroup;    // # of different streams available per group call
    int                        numGroupCalls;         // Total # of group calls to be executed
    bool                       useBlocking;           // RCCL communication with blocking or non-blocking option
    MemAllocType               memAllocType = MEM_ALLOC_HIP; // Allocation mode of the current config
    EnvVars                    ev;                    // Environment variables

    // Comm process pool: poolChildren[d] is pinned to device d; Finalize() is
    // the pool-reset boundary (re-forks the pool with the current env).
    bool                       poolMode;              // reuse workers across configs (UT_COMM_POOL != 0)
    std::vector<TestBedChild*> poolChildren;          // persistent pool, index == pinned device id
    // Cold: device code not loaded yet. Warming: a WARMUP reply is still unread,
    // so nothing else may be sent to or read from that worker. Warm: loaded.
    enum class PoolWorkerState { Cold, Warming, Warm };
    std::vector<PoolWorkerState> poolWorkerState;     // parallel to poolChildren
    bool                       configUsedPool;        // last InitComms served from the pool (skip Finalize in DestroyComms)
    bool                       childFailed = false;   // a child reported failure since the last DestroyComms (abort, not finalize)

    // Constructor - Creates one child process per detected GPU device that waits for further commands
    TestBed();

    int NumActiveChildren() const { return static_cast<int>(childList.size()); }
    int NumActiveRanks() const    { return static_cast<int>(rankToChildMap.size()); }

    // Prepare TestBed with multiple group call customization
    void InitComms(std::vector<std::vector<int>> const& deviceIdsPerChild,
                   std::vector<int>              const& numCollectivesInGroup,
                   std::vector<int>              const& numStreamsPerGroup,
                   int                           const  numGroupCalls = 1,
                   bool                          const  useBlocking   = true,
                   MemAllocType                  const  memAllocType  = MEM_ALLOC_HIP);

    // Prepare TestBed for use with GPUs across multiple child processes
    void InitComms(std::vector<std::vector<int>> const& deviceIdsPerChild,
                   int  const numCollectivesInGroup = 1,
                   int  const numStreamsPerGroup    = 1,
                   int  const numGroupCalls         = 1,
                   bool const useBlocking           = true,
                   MemAllocType const memAllocType  = MEM_ALLOC_HIP);

    // Prepare TestBed for use with GPUs on a single child process
    void InitComms(int  const numGpus,
                   int  const numCollectivesInGroup = 1,
                   int  const numStreamsPerGroup    = 1,
                   int  const numGroupCalls         = 1,
                   bool const useBlocking           = true,
                   MemAllocType const memAllocType  = MEM_ALLOC_HIP);

    // Set collectives arguments for specified collective / rank
    // Setting scalarsPerRank to non-null will create custom reduction operator
    // Using collId = -1 (default) applies settings to all collectives in group
    // Using rank = -1 (default) applies settings to all ranks
    void SetCollectiveArgs(ncclFunc_t      const funcType,
                           ncclDataType_t  const dataType,
                           size_t          const numInputElements,
                           size_t          const numOutputElements,
                           OptionalColArgs const &optionalArgs = {},
                           int             const collId        = -1,
                           int             const groupId       = 0,
                           int             const rank          = -1,
                           int             const streamIdx     = 0);

    // Allocate memory for specified collective / rank
    // - Requires SetCollectiveArgs to have been called already
    // Using collId = -1 (default) applies settings to all collectives in group
    // Using rank = -1 (default) applies settings to all ranks
    // Using groupIdx = -1 (default) applies setting to all groups
    // With MEM_ALLOC_SYMMETRIC_WIN, rank must be -1: window registration is
    // collective over the whole communicator.
    void AllocateMem(bool   const inPlace = false,
                     bool   const useManagedMem = false,
                     int    const groupId  = -1,
                     int    const collId   = -1,
                     int    const rank     = -1,
                     bool   const userRegistered = false);

    // Initialize input and compute expected results
    // - requires that SetCollectiveArgs and AllocateMemory have already been called
    // Setting groupId to -1 applies setting to all groups
    // Setting collId to -1 applies settings to all collectives in group
    // Setting rank to -1 applies settings to all ranks
    // Setting prepDataFunc to nullptr uses the default fill pattern routine
    void PrepareData(int         const groupId      = -1,
                     int         const collId       = -1,
                     int         const rank         = -1,
                     CollFuncPtr const prepDataFunc = nullptr);

    // Execute all collectives on all test children
    // Blocks until collective is completed
    void ExecuteCollectives(std::vector<int> const &currentRanks = {},
                            int              const groupId       = -1,
                            bool             const useHipGraph   = false);

    // Perform results validation - compare output to expected
    void ValidateResults(bool& isCorrect, int const groupId = -1, int const collId = -1, int const rank = -1);

    // Launch instantiated graphs
    void LaunchGraphs(int const groupId = -1);

    // Release allocated memory
    void DeallocateMem(int const groupId = -1, int const collId = -1, int const rank = -1);

    // Release the RCCL comms
    void DestroyComms();

    // Pool mode (UT_COMM_WARMAHEAD): starts loading device code on the cold pool
    // workers a later InitComms(deviceIdsPerChild) will use, without waiting.
    // That InitComms collects the replies.
    void WarmPoolWorkers(std::vector<std::vector<int>> const& deviceIdsPerChild);

    // Query whether every active rank's comm supports symmetric kernels (requires InitComms)
    void QuerySymmetricSupport(bool& isSupported);

    // Release created graphs
    void DestroyGraphs();

    // Explicit TestBed destructor that releases all child processes
    // No further calls to TestBed should be performed after this call
    void Finalize();

    // Teardown helpers. TeardownOwnedChildList reaps the per-config fork-fresh children
    // that childList owns (classic / pool-fallback path). TeardownPool reaps the
    // persistent pool workers. Finalize() is the pool-reset boundary and runs both.
    void TeardownOwnedChildList();
    void TeardownPool();

    // Destructor - Calls Finalize() to release all child processes
    ~TestBed();

    // Returns all the supported reduction operations based on build settings
    std::vector<ncclRedOp_t> const& GetAllSupportedRedOps();

    // Return all the supported data types based on build settings
    std::vector<ncclDataType_t> const& GetAllSupportedDataTypes();

    // Returns the intersection of testRedOps with supported reduction operations as redOps.
    void GetSupportedRedOps(std::vector<ncclRedOp_t>& redOps, const std::vector<ncclRedOp_t>& testRedOps);

    // Returns the intersection of testDataTypes with supported data types as dataTypes.
    void GetSupportedDataTypes(std::vector<ncclDataType_t>& dataTypes, const std::vector<ncclDataType_t>& testDataTypes);

    // Return a list for # of collectives per group
    std::vector<int> const GetNumCollsPerGroup(int const numCollectivesInGroup,
                                                int const numGroupCalls);

    // Return a list for # of streams per group
    std::vector<int> const GetNumStreamsPerGroup(int const numStreamsPerGroup,
                                                  int const numGroupCalls);

    // Helper function that splits up GPUs to the given number of processes
    static std::vector<std::vector<int>> GetDeviceIdsList(int const numProcesses,
                                                          int const numGpus,
                                                          int const ranksPerGpu,
                                                          const std::vector<int>& gpuPriorityOrder);

    static std::vector<std::vector<int>> GetDeviceIdsList(int const numProcesses,
                                                          int const numGpus,
                                                          const std::vector<int>& gpuPriorityOrder);

    // Generate a test case name
    static std::string GetTestCaseName(int            const totalRanks,
                                       bool           const isMultiProcess,
                                       ncclFunc_t     const funcType,
                                       ncclDataType_t const dataType,
                                       ncclRedOp_t    const redOp,
                                       int            const root,
                                       bool           const inPlace,
                                       bool           const managedMem,
                                       bool           const useHipGraph,
                                       int            const ranksPerProc=1);

    // Run a simple sweep
    void RunSimpleSweep(std::vector<ncclFunc_t>     const& funcTypes,
                        std::vector<ncclDataType_t> const& dataTypes,
                        std::vector<ncclRedOp_t>    const& redOps,
                        std::vector<int>            const& roots,
                        std::vector<int>            const& numElements,
                        std::vector<bool>           const& inPlaceList,
                        std::vector<bool>           const& managedMemList,
                        std::vector<bool>           const& useHipGraphList,
                        bool                        const& enableSweep = true,
                        MemAllocType                       memAllocType = MEM_ALLOC_HIP);

    // Wait for user-input if in interactive mode
    void InteractiveWait(std::string message);

    // Used to track total number of calls to ExecuteCollectives()
    static int& NumTestsRun();

  private:
    // Starts a worker in a fresh process image so it never inherits HIP/HSA
    // runtime state from the test parent.
    bool SpawnChildProcess(TestBedChild* child, MemAllocType memAllocType);

    // WaitForChild result when childId sent nothing within ChildReplyTimeoutSec().
    static constexpr int kChildWaitTimedOut = -2;

    // Seconds the parent waits for any single child reply (0 = no limit). Never
    // shorter than the child's own collective timeout, and off under UT_DEBUG_PAUSE.
    int ChildReplyTimeoutSec() const;

    // Blocks until childList[childId] has data (or EOF) to read. Returns the id
    // of another active child whose pipe closed first, kChildWaitTimedOut, or -1.
    int WaitForChild(int childId);

    // Kills and reaps every worker after lostChildId is lost (waitForExit: let it
    // report its exit status first); returns the failure report.
    std::string AbandonChildren(int lostChildId, bool waitForExit = true,
                                char const* what = "stopped responding");

    // Fills workers with the pool worker serving each child of deviceIdsPerChild.
    // False when some child has no distinct live worker (fork-fresh path).
    bool PoolWorkersFor(std::vector<std::vector<int>> const& deviceIdsPerChild,
                        std::vector<int>& workers) const;

    // Reads a Warming worker's reply by deadline; on failure removes it from the pool and fails the test.
    bool FinishWarmup(int worker, std::chrono::steady_clock::time_point deadline);

    // Reads one ack per listed child id and fails if any reported failure. peersInterlocked:
    // children wait on each other, so a failure with acks still owed stops every worker.
    void CollectAcks(std::vector<int> const& childIds, bool* allSucceeded = nullptr,
                     bool peersInterlocked = false);

    // Forgets the active config's children, ranks and failure flag; owned children
    // must already be reaped. configUsedPool still reports how that config was served.
    void ResetActiveConfig();

    // Fill ranks / groups with the selection (-1 = all active). False, with a test
    // failure, when a specific rank / group is not active, so a mistyped target
    // cannot turn the command into a silent no-op.
    bool SelectRanks(int rank, std::vector<int>& ranks) const;
    bool SelectGroups(int groupId, std::vector<int>& groups) const;

    // ranks grouped by owning child, indexed by child id.
    std::vector<std::vector<int>> RanksPerChild(std::vector<int> const& ranks) const;

    // Ids of all active children / of the children that own at least one rank, ascending.
    std::vector<int> AllChildIds() const;
    static std::vector<int> ChildrenWithRanks(std::vector<std::vector<int>> const& ranksPerChild);

    // AllocateMem is split into AllocateMemInternal + RegisterMemInternal to maintain
    // compatibility with existing tests, and extend registration for symmetric memory.
    // Returns false if any child failed, so registration can be skipped.
    bool AllocateMemInternal(bool   const inPlace = false,
                             bool   const useManagedMem = false,
                             int    const groupId  = -1,
                             int    const collId   = -1,
                             int    const rank     = -1,
                             bool   const userRegistered = false);

    void RegisterMemInternal(int    const groupId,
                             int    const collId,
                             int    const rank);
  };
}
