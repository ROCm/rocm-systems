/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "TestBedChild.hpp"
#include "PipeUtils.hpp"

#include <algorithm>
#include <functional>
#include <thread>
#include <execinfo.h>
#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

static int getThreadId()
{
  #ifdef ENABLE_OPENMP
  return (int)omp_get_thread_num();
  #else
  return -1;
  #endif
}

#define CHILD_NCCL_CALL_BASE(cmd, msg, RESULT, RESULT_ARGS...)          \
  do {                                                                  \
    if (this->verbose) printf("[ NCCL CALL] " #cmd "\n");               \
    ncclResult_t status = cmd;                                          \
    if (status != ncclSuccess)                                          \
    {                                                                   \
      TEST_ERROR("Child process %d fails NCCL call %s with code %d", this->childId, msg, status); \
      RESULT(TEST_FAIL, ##RESULT_ARGS);                                 \
    }                                                                   \
  } while (false)
#define CHILD_NCCL_CALL(cmd, msg) CHILD_NCCL_CALL_BASE(cmd, msg, RETURN_RESULT)

#define CHILD_NCCL_CALL_NON_BLOCKING_BASE(msg, localRank, RESULT, RESULT_ARGS...) \
  do {                                                                \
    unsigned long int loop_counter = 0;                               \
    ncclResult_t ncclAsyncErr;                                        \
    loop_counter = 0;                                                 \
    do                                                                \
    {                                                                 \
      loop_counter++;                                                 \
      if (loop_counter == MAX_LOOP_COUNTER) break;                    \
      ncclCommGetAsyncError(this->comms[localRank], &ncclAsyncErr);   \
    } while(ncclAsyncErr == ncclInProgress);                          \
    if (ncclAsyncErr != ncclSuccess)                                  \
    {                                                                 \
      TEST_ERROR("Child process %d fails NCCL call %s with code %d", this->childId, msg, ncclAsyncErr);  \
      RESULT(TEST_FAIL, ##RESULT_ARGS);                               \
    }                                                                 \
  } while (false)
#define CHILD_NCCL_CALL_NON_BLOCKING(msg, localRank) CHILD_NCCL_CALL_NON_BLOCKING_BASE(msg, localRank, RETURN_RESULT)
#define PIPE_READ(val) \
    if (RcclUnitTesting::detail::safe_pipe_read(childReadFd, &val, sizeof(val)) != sizeof(val)) return TEST_FAIL;

#ifdef ENABLE_OPENMP
#define CHILD_NCCL_CALL_RANK(errCode, cmd, msg) CHILD_NCCL_CALL_BASE(cmd, msg, OMP_CANCEL_FOR, errCode)
#define CHILD_NCCL_CALL_NON_BLOCKING_RANK(errCode, msg, localRank) CHILD_NCCL_CALL_NON_BLOCKING_BASE(msg, localRank, OMP_CANCEL_FOR, errCode)
#else
#define CHILD_NCCL_CALL_RANK(errCode, cmd, msg) CHILD_NCCL_CALL(cmd, msg)
#define CHILD_NCCL_CALL_NON_BLOCKING_RANK(errCode, msg, localRank) CHILD_NCCL_CALL_NON_BLOCKING(msg, localRank)
#endif

namespace RcclUnitTesting
{
  TestBedChild::TestBedChild(int const childId, bool const verbose, int const printValues, bool const useRankThreading)
  {
    this->childId = childId;
    this->verbose = verbose;
    this->printValues = printValues;
    this->useRankThreading = useRankThreading;
    // -1 sentinel: teardown skips waitpid()/close() for a never-forked child.
    this->pid          = -1;
    this->parentWriteFd = -1;
    this->parentReadFd  = -1;
    this->childWriteFd  = -1;
    this->childReadFd   = -1;
  }

  int TestBedChild::InitPipes()
  {
    // Prepare parent->child pipe
    int pipefd[2];
    if (pipe(pipefd) == -1)
    {
      TEST_ERROR("Unable to create parent->child pipe for child %d", this->childId);
      return TEST_FAIL;
    }
    this->childReadFd   = pipefd[0];
    this->parentWriteFd = pipefd[1];

    // Prepare child->parent pipe
    this->parentReadFd = -1;
    if (pipe(pipefd) == -1)
    {
      TEST_ERROR("Unable to create child->parent pipe for child %d", this->childId);
      return TEST_FAIL;
    }
    this->parentReadFd = pipefd[0];
    this->childWriteFd = pipefd[1];

    return TEST_SUCCESS;
  }

  void TestBedChild::StartExecutionLoop()
  {

    // Wait for commands from parent process
    if (verbose) TEST_INFO("Child %d enters execution loop", this->childId);
    #ifndef ENABLE_OPENMP
    if (useRankThreading)
    {
      TEST_ERROR("UT_MULTITHREAD=1 requires a unit-test build with OPENMP_TESTS_ENABLED=ON");
      exit(1);
    }
    #endif
    int command;
    while (true)
    {
      if (RcclUnitTesting::detail::safe_pipe_read(childReadFd, &command, sizeof(command)) != sizeof(command)) {
        break;
      }
      ErrCode status = TEST_SUCCESS;
      if (command < 0 || command >= NUM_CHILD_COMMANDS) {
        // The payload length is unknown, so the stream cannot be resynchronized:
        // report the failure to the parent, then stop.
        TEST_ERROR("Child %d received invalid command ID: %d", this->childId, command);
        status = TEST_FAIL;
        RcclUnitTesting::detail::safe_pipe_write(childWriteFd, &status, sizeof(status));
        goto stop;
      }

      if (verbose) TEST_INFO("Child %d received command [%s]:", this->childId, ChildCommandNames[command]);;
      std::vector<char> retValBuf;
      switch(command)
      {
      case CHILD_GET_UNIQUE_ID   : status = GetUniqueId(retValBuf); break;
      case CHILD_INIT_COMMS      : status = InitComms();            break;
      case CHILD_SET_COLL_ARGS   : status = SetCollectiveArgs();    break;
      case CHILD_ALLOCATE_MEM    : status = AllocateMem();          break;
      case CHILD_REGISTER_MEM    : status = RegisterMem();          break;
      case CHILD_PREPARE_DATA    : status = PrepareData();          break;
      case CHILD_EXECUTE_COLL    : status = ExecuteCollectives();   break;
      case CHILD_VALIDATE_RESULTS: status = ValidateResults();      break;
      case CHILD_LAUNCH_GRAPHS   : status = LaunchGraphs();         break;
      case CHILD_DEALLOCATE_MEM  : status = DeallocateMem();        break;
      case CHILD_DESTROY_COMMS   : status = DestroyComms();         break;
      case CHILD_DESTROY_GRAPHS  : status = DestroyGraphs();        break;
      case CHILD_STOP            : goto stop;
      default:
        TEST_ERROR("Child %d received unknown command ID: %d", this->childId, command);
        status = TEST_FAIL;
        RcclUnitTesting::detail::safe_pipe_write(childWriteFd, &status, sizeof(status));
        goto stop;
      }

      // Send back acknowledgement to parent
      if (status == TEST_FAIL)
        TEST_ERROR("Child %d failed on command [%s]:", this->childId, ChildCommandNames[command]);
      if (RcclUnitTesting::detail::safe_pipe_write(childWriteFd, &status, sizeof(status)) < 0) {
        TEST_ERROR("Child %d write to parent failed: %s", this->childId, strerror(errno));
        break;
      }
      if (retValBuf.size() > 0 &&
          RcclUnitTesting::detail::safe_pipe_write(childWriteFd, retValBuf.data(), retValBuf.size()) < 0) {
        TEST_ERROR("Child %d write return value to parent failed: %s", this->childId, strerror(errno));
        break;
      }
    }
  stop:
    // Ensure communicators are destroyed before child process exits
    if (!this->comms.empty()) DestroyComms();

    if (verbose) TEST_INFO("Child %d exiting execution loop", this->childId);

    fflush(stdout);
    fflush(stderr);
    // Close child ends of pipe
    close(this->childReadFd);
    close(this->childWriteFd);

    exit(0);
  }

  ErrCode TestBedChild::GetUniqueId(std::vector<char>& retValBuf)
  {
    if (this->verbose) TEST_INFO("Child %d begins GetUniqueId()", this->childId);

    // Get a unique ID and pass it back to parent process
    ncclUniqueId id;
    CHILD_NCCL_CALL(ncclGetUniqueId(&id), "ncclGetUniqueId");
    retValBuf.resize(sizeof(id));
    memcpy(retValBuf.data(), &id, sizeof(id));

    if (this->verbose) TEST_INFO("Child %d finishes GetUniqueId()", this->childId);
    return TEST_SUCCESS;
  }

  ErrCode TestBedChild::InitComms()
  {
    if (this->verbose) TEST_INFO("Child %d begins InitComms()", this->childId);

    // Read config sent by parent
    ncclUniqueId id;
    PIPE_READ(id);
    PIPE_READ(this->totalRanks);
    PIPE_READ(this->rankOffset);
    PIPE_READ(this->numGroupCalls);
    // --- Read numCollectivesInGroup ---
    int numCollSize = 0;
    PIPE_READ(numCollSize);
    if (numCollSize < 0)
    {
      TEST_ERROR("Child %d received invalid numCollectivesInGroup size %d", this->childId, numCollSize);
      return TEST_FAIL;
    }
    this->numCollectivesInGroup.resize(numCollSize);
    if (numCollSize > 0)
    {
      if (RcclUnitTesting::detail::safe_pipe_read(this->childReadFd,
                                                  this->numCollectivesInGroup.data(),
                                                  numCollSize * sizeof(int)) !=
          static_cast<ssize_t>(numCollSize * sizeof(int)))
        return TEST_FAIL;
    }
    PIPE_READ(this->useBlocking);
    int allocTypeInt = 0;
    PIPE_READ(allocTypeInt);
    this->memAllocType = static_cast<MemAllocType>(allocTypeInt);

    bool useMultiRankPerGpu;
    PIPE_READ(useMultiRankPerGpu);
    // --- Read numStreamsPerGroup ---
    int numStreamsSize = 0;
    PIPE_READ(numStreamsSize);
    if (numStreamsSize < 0)
    {
      TEST_ERROR("Child %d received invalid numStreamsPerGroup size %d", this->childId, numStreamsSize);
      return TEST_FAIL;
    }
    this->numStreamsPerGroup.resize(numStreamsSize);
    if (numStreamsSize > 0)
    {
      if (RcclUnitTesting::detail::safe_pipe_read(this->childReadFd,
                                                  this->numStreamsPerGroup.data(),
                                                  numStreamsSize * sizeof(int)) !=
          static_cast<ssize_t>(numStreamsSize * sizeof(int)))
        return TEST_FAIL;
    }

    // Read GPUs and prepare storage
    int numGpus;
    PIPE_READ(numGpus);
    if (numGpus < 0)
    {
      TEST_ERROR("Child %d received invalid GPU count %d", this->childId, numGpus);
      return TEST_FAIL;
    }
    std::vector<int> newDeviceIds(numGpus);
    for (int& deviceId : newDeviceIds) PIPE_READ(deviceId);

    if (this->numGroupCalls < 0 ||
        static_cast<int>(this->numCollectivesInGroup.size()) < this->numGroupCalls ||
        static_cast<int>(this->numStreamsPerGroup.size()) < this->numGroupCalls)
    {
      TEST_ERROR("Child %d received %d group calls but %zu collective counts and %zu stream counts",
                 this->childId, this->numGroupCalls, this->numCollectivesInGroup.size(),
                 this->numStreamsPerGroup.size());
      return TEST_FAIL;
    }

    // Destroy existing HIP streams before clearing vector to prevent hardware queue leak!
    // deviceIds still holds the previous config's devices here; each stream must be
    // destroyed with its owning device current.
    for (auto& groupStreams : this->streams)
    {
      for (size_t localRank = 0; localRank < groupStreams.size(); ++localRank)
      {
        for (hipStream_t& stream : groupStreams[localRank])
        {
          if (stream == nullptr) continue;
          if (localRank < this->deviceIds.size())
            CHECK_HIP(hipSetDevice(this->deviceIds[localRank]));
          CHECK_HIP(hipStreamDestroy(stream));
          stream = nullptr;
        }
      }
    }

    this->deviceIds = newDeviceIds;
    this->streams.clear();
    this->streams.resize(this->numGroupCalls);
    this->collArgs.resize(this->numGroupCalls);

    for (int i = 0; i < this->numGroupCalls; i++)
    {
      this->collArgs[i].resize(numGpus);
      this->streams[i].resize(numGpus);
      for (int j = 0; j < numGpus; j++)
      {
        this->collArgs[i][j].clear();
        this->collArgs[i][j].resize(numCollectivesInGroup[i]);
        this->streams[i][j].resize(numStreamsPerGroup[i], nullptr);
      }
    }

    // Initialize graph tracking
    this->graphs.resize(this->numGroupCalls);
    this->graphExecs.resize(this->numGroupCalls);
    this->graphEnabled.resize(this->numGroupCalls);

    // Initialize communicators
    comms.clear();
    comms.resize(numGpus);

    ErrCode status = TEST_SUCCESS;

    // Create HIP streams OUTSIDE of ncclGroupStart()
    for (int groupCallIdx = 0; groupCallIdx < this->numGroupCalls; ++groupCallIdx)
    {
      for (int localRank = 0; localRank < numGpus; ++localRank)
      {
        int const globalRank = this->rankOffset + localRank;
        int const currGpu = this->deviceIds[localRank];

        if (hipSetDevice(currGpu) != hipSuccess)
        {
          TEST_ERROR("Rank %d on child %d unable to switch to GPU %d", globalRank, this->childId, currGpu);
          status = TEST_FAIL;
          break;
        }

        for (int i = 0; i < this->numStreamsPerGroup[groupCallIdx]; i++)
        {
          hipError_t err = hipStreamCreate(&(this->streams[groupCallIdx][localRank][i]));
          if (err != hipSuccess)
          {
            TEST_ERROR("Rank %d on child %d unable to create stream %d for GPU %d in group %d. HIP Error: %s (%d)",
                       globalRank, this->childId, i, currGpu, groupCallIdx, hipGetErrorString(err), err);
            status = TEST_FAIL;
            break;
          }
        }
        if (status == TEST_FAIL) break;
      }
      // Properly break outer loop on error
      if (status == TEST_FAIL) break;
    }

    if (status == TEST_FAIL) return TEST_FAIL;

    // Initialize NCCL communicators within a group call to prevent deadlock
    CHILD_NCCL_CALL(ncclGroupStart(), "ncclGroupStart");

    for (int localRank = 0; localRank < numGpus; ++localRank)
    {
      int const globalRank = this->rankOffset + localRank;
      int const currGpu = this->deviceIds[localRank];

      if (hipSetDevice(currGpu) != hipSuccess)
      {
        TEST_ERROR("Rank %d on child %d unable to switch to GPU %d during comm init", globalRank, this->childId, currGpu);
        status = TEST_FAIL;
        break;
      }

      if (useMultiRankPerGpu)
      {
        TEST_ERROR("Rank %d on child %d: Multi-rank per GPU requested but not implemented", globalRank, this->childId);
        status = TEST_FAIL;
        break;
      }
      else if (this->useBlocking == false)
      {
        ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
        config.blocking = 0;
        ncclResult_t const initState =
          ncclCommInitRankConfig(&this->comms[localRank], this->totalRanks, id, globalRank, &config);
        if (initState != ncclSuccess && initState != ncclInProgress)
        {
          TEST_ERROR("Rank %d on child %d unable to call ncclCommInitRankConfig: error %d",
                     globalRank, this->childId, initState);
          status = TEST_FAIL;
          break;
        }
      }
      else
      {
        if (ncclCommInitRank(&this->comms[localRank], this->totalRanks, id, globalRank) != ncclSuccess)
        {
          TEST_ERROR("Rank %d on child %d unable to call ncclCommInitRank", globalRank, this->childId);
          status = TEST_FAIL;
          break;
        }
      }
    }

    // ALWAYS call ncclGroupEnd() once ncclGroupStart() has been executed!
    ncclResult_t const groupEndState = ncclGroupEnd();
    if (this->useBlocking == false)
    {
      if (groupEndState != ncclSuccess && groupEndState != ncclInProgress)
      {
        TEST_ERROR("Child %d ncclGroupEnd failed with error %d", this->childId, groupEndState);
        status = TEST_FAIL;
      }
      else
      {
        // Grouped non-blocking initialization starts at ncclGroupEnd(). Wait for
        // every communicator that was created, even when setup above already
        // failed, so DestroyComms never finalizes one that is still initializing.
        for (int localRank = 0; localRank < numGpus; ++localRank)
        {
          if (this->comms[localRank] == nullptr) continue;
          int const currGpu = this->deviceIds[localRank];
          if (hipSetDevice(currGpu) != hipSuccess)
          {
            TEST_ERROR("Child %d unable to switch to GPU %d while waiting for communicator initialization",
                       this->childId, currGpu);
            status = TEST_FAIL;
            continue;
          }
          ErrCode const pollStatus = [&]() -> ErrCode
          {
            CHILD_NCCL_CALL_NON_BLOCKING("ncclCommGetAsyncErrorInitRankConfig", localRank);
            return TEST_SUCCESS;
          }();
          if (pollStatus != TEST_SUCCESS) status = TEST_FAIL;
        }
      }
    }
    else if (groupEndState != ncclSuccess)
    {
      TEST_ERROR("Child %d ncclGroupEnd failed with error %d", this->childId, groupEndState);
      status = TEST_FAIL;
    }

    if (this->verbose)
    {
      TEST_INFO("Child %d finishes InitComms() [%s]", this->childId, status == TEST_SUCCESS ? "SUCCESS" : "FAIL");
    }
    return status;
  }

  ErrCode TestBedChild::SetCollectiveArgs()
  {
    if (this->verbose) TEST_INFO("Child %d begins SetCollectiveArgs()", this->childId);

    // Read values sent by parent [see TestBed::SetCollectiveArgs()]
    int             globalRank;
    int             collId;
    int             groupId;
    ncclFunc_t      funcType;
    ncclDataType_t  dataType;
    size_t          numInputElements;
    size_t          numOutputElements;
    int             streamIdx;
    OptionalColArgs options;

    PIPE_READ(globalRank);
    PIPE_READ(collId);
    PIPE_READ(groupId);
    PIPE_READ(funcType);
    PIPE_READ(dataType);
    PIPE_READ(numInputElements);
    PIPE_READ(numOutputElements);
    PIPE_READ(streamIdx);
    PIPE_READ(options);

    if (!this->IsValidGroupId(groupId, "SetCollectiveArgs")) return TEST_FAIL;
    if (globalRank < this->rankOffset || (this->rankOffset + comms.size() <= globalRank))
    {
      TEST_ERROR("Child %d does not contain rank %d", this->childId, globalRank);
      return TEST_FAIL;
    }
    int const localRank = globalRank - rankOffset;
    CHECK_HIP(hipSetDevice(this->deviceIds[localRank]));

    for (int collIdx = 0; collIdx < collArgs[groupId][localRank].size(); ++collIdx)
    {
      if (collId == -1 || collId == collIdx)
      {
        CollectiveArgs& collArg = this->collArgs[groupId][localRank][collIdx];
        CHECK_CALL(collArg.SetArgs(globalRank, this->totalRanks,
                                   this->deviceIds[localRank],
                                   funcType, dataType,
                                   numInputElements, numOutputElements,
                                   streamIdx,
                                   options));
        if (this->verbose) TEST_INFO("Rank %d on child %d sets collective %d in group %d [%s]",
                                globalRank, this->childId, collIdx, groupId,
                                collArg.GetDescription().c_str());

        // If pre-mult scalars are provided, then create a custom reduction operator
        if (options.scalarMode >= 0)
        {
          CHILD_NCCL_CALL(ncclRedOpCreatePreMulSum(&collArg.options.redOp,
                                                   collArg.localScalar.ptr,
                                                   dataType,
                                                   (ncclScalarResidence_t)options.scalarMode,
                                                   this->comms[localRank]),
                          "ncclRedOpCreatePreMulSum");
          if (verbose) TEST_INFO("Child %d created custom redop %d for group %d collective %d",
                            this->childId, collArg.options.redOp, groupId, collIdx);
        }
      }
    }
    if (this->verbose) TEST_INFO("Child %d finishes SetCollectiveArgs()", this->childId);
    return TEST_SUCCESS;
  }

  ErrCode TestBedChild::AllocateMem()
  {
    if (this->verbose) TEST_INFO("Child %d begins AllocateMem()", this->childId);

    // Read values sent by parent [see TestBed::AllocateMem()]
    int    globalRank;
    int    collId;
    bool   inPlace;
    bool   useManagedMem;
    bool   userRegistered;
    int    groupId;

    PIPE_READ(globalRank);
    PIPE_READ(collId);
    PIPE_READ(inPlace);
    PIPE_READ(useManagedMem);
    PIPE_READ(userRegistered);
    PIPE_READ(groupId);

    if (!this->IsValidGroupId(groupId, "AllocateMem")) return TEST_FAIL;
    if (globalRank < this->rankOffset || (this->rankOffset + comms.size() <= globalRank))
    {
      TEST_ERROR("Child %d does not contain rank %d", this->childId, globalRank);
      return TEST_FAIL;
    }
    int const localRank = globalRank - rankOffset;
    CHECK_HIP(hipSetDevice(this->deviceIds[localRank]));

    for (int collIdx = 0; collIdx < collArgs[groupId][localRank].size(); ++collIdx)
    {
      if (collId == -1 || collId == collIdx)
      {
        CollectiveArgs& collArg = this->collArgs[groupId][localRank][collIdx];
        CHECK_CALL(collArg.AllocateMem(inPlace, useManagedMem, userRegistered));
        if (this->verbose) TEST_INFO("Rank %d on child %d allocates memory for collective %d in group %d on device %d (%s,%s,%s) Input: %p Output %p",
                                globalRank, this->childId, collIdx, groupId, this->deviceIds[localRank],
                                inPlace ? "in-place" : "out-of-place",
                                useManagedMem ? "managed" : "unmanaged",
                                userRegistered ? "user registered buffer" : "internal copy",
                                collArg.inputGpu.ptr,
                                collArg.outputGpu.ptr);
      }
    }

    if (this->verbose) TEST_INFO("Child %d finishes AllocateMem()", this->childId);
    return TEST_SUCCESS;
  }

  // Fill input memory with pre-known patterned based on rank
  ErrCode TestBedChild::PrepareData()
  {
    if (this->verbose) TEST_INFO("Child %d begins PrepareData()", this->childId);

    // Read values sent by parent [see TestBed::PrepareData()]
    int globalRank;
    int collId;
    int groupId;
    intptr_t prepDataFuncOffset;

    PIPE_READ(globalRank);
    PIPE_READ(groupId);
    PIPE_READ(collId);
    PIPE_READ(prepDataFuncOffset);
    CollFuncPtr const prepDataFunc = CollFuncPtrFromOffset(prepDataFuncOffset);

    if (!this->IsValidGroupId(groupId, "PrepareData")) return TEST_FAIL;
    if (globalRank < this->rankOffset || (this->rankOffset + comms.size() <= globalRank))
    {
      TEST_ERROR("Child %d does not contain rank %d", this->childId, globalRank);
      return TEST_FAIL;
    }

    int const localRank = globalRank - rankOffset;
    CHECK_HIP(hipSetDevice(this->deviceIds[localRank]));

    for (int collIdx = 0; collIdx < collArgs[groupId][localRank].size(); ++collIdx)
    {
      if (collId != -1 && collId != collIdx) continue;

      if (this->verbose) TEST_INFO("Rank %d on child %d prepares data for collective %d in group %d",
                              globalRank, this->childId, collIdx, groupId);
      CHECK_CALL(this->collArgs[groupId][localRank][collIdx].PrepareData(prepDataFunc));
    }
    if (this->verbose) TEST_INFO("Child %d finishes PrepareData()", this->childId);
    return TEST_SUCCESS;
  }

  ErrCode TestBedChild::ExecuteCollectives()
  {
    int timeoutUs = 0;
    int groupId = 0;
    bool useHipGraph = false;

    PIPE_READ(timeoutUs);
    PIPE_READ(groupId);
    PIPE_READ(useHipGraph);

    int numRanksToExecute, tempRank;
    std::vector<int> ranksToExecute = {};
    PIPE_READ(numRanksToExecute);

    for (int rank = 0; rank < numRanksToExecute; ++rank){
      PIPE_READ(tempRank);
      ranksToExecute.push_back(tempRank - this->rankOffset);
    }
    if (!this->IsValidGroupId(groupId, "ExecuteCollectives")) return TEST_FAIL;
    if (this->verbose) TEST_INFO("Child %d begins ExecuteCollectives() %s with allocation type %d", this->childId, useHipGraph ? "(using hipGraphs)" : "", (int32_t)this->memAllocType);

    // Determine which local ranks to execute on
    std::vector<int> localRanksToExecute;
    for (int localRank = 0; localRank < this->deviceIds.size(); ++localRank)
    {
      // If ranksToExeute is empty, execute all local ranks belonging to this child
      if (!ranksToExecute.empty() &&
          (std::count(ranksToExecute.begin(), ranksToExecute.end(), localRank) == 0)) continue;
      localRanksToExecute.push_back(localRank);
    }

    numRanksToExecute = (int)localRanksToExecute.size();

    // =========================================================================
    // STAGE 1: PRE-COLLECTIVE DEBUG PRINTING (BEFORE ncclGroupStart)
    // =========================================================================
    if (this->printValues && !useHipGraph)
    {
      for (int collId = 0; collId < this->numCollectivesInGroup[groupId]; ++collId)
      {
        for (int localRank : localRanksToExecute)
        {
          CollectiveArgs& collArg = this->collArgs[groupId][localRank][collId];
          CHECK_HIP(hipSetDevice(this->deviceIds[localRank]));

          int const numInputElementsToPrint = (this->printValues < 0 ? collArg.numInputElements : this->printValues);
          PtrUnion inputCpu;
          size_t const numInputBytes = numInputElementsToPrint * DataTypeToBytes(collArg.dataType);
          inputCpu.AllocateCpuMem(numInputBytes);

          // Safe hipMemcpy BEFORE collective launch
          CHECK_HIP(hipMemcpy(inputCpu.ptr, collArg.inputGpu.ptr, numInputBytes, hipMemcpyDeviceToHost));
          printf("[ DEBUG    ] Rank %02d Group %d Coll %d %-10s: %s\n", collArg.globalRank, groupId, collId, "Input",
                 inputCpu.ToString(collArg.dataType, numInputElementsToPrint).c_str());
          inputCpu.FreeCpuMem();

          int const numOutputElementsToPrint = (this->printValues < 0 ? collArg.numOutputElements : this->printValues);
          size_t const numOutputBytes = numOutputElementsToPrint * DataTypeToBytes(collArg.dataType);
          CHECK_HIP(hipMemcpy(collArg.outputCpu.ptr, collArg.outputGpu.ptr, numOutputBytes, hipMemcpyDeviceToHost));
          printf("[ DEBUG    ] Rank %02d Group %d Coll %d %-10s: %s\n", collArg.globalRank, groupId, collId, "Pre-Output",
                 collArg.outputCpu.ToString(collArg.dataType, numOutputElementsToPrint).c_str());
        }
      }
    }

    // A previous execution may have left graphs behind (e.g. DestroyGraphs was
    // skipped after a failure); free them before their flags are reset below.
    CHECK_CALL(this->ReleaseGraphHandles(groupId));

    size_t const totalLocalDevices = this->deviceIds.size();
    this->graphs[groupId].resize(totalLocalDevices);
    this->graphExecs[groupId].resize(totalLocalDevices);
    this->graphEnabled[groupId].resize(totalLocalDevices);
    for (int i = 0; i < totalLocalDevices; i++)
    {
      this->graphs[groupId][i].resize(this->numStreamsPerGroup[groupId]);
      this->graphExecs[groupId][i].resize(this->numStreamsPerGroup[groupId]);
      this->graphEnabled[groupId][i].resize(this->numStreamsPerGroup[groupId]);
      // Reset graphEnabled state for all streams on this device
      for (int s = 0; s < this->numStreamsPerGroup[groupId]; s++)
      {
        this->graphEnabled[groupId][i][s] = false;
      }
    }

    // Once capture starts, every exit path must end it; otherwise the streams
    // stay in capture mode for the rest of the worker's life.
    bool captureActive = false;
    struct CaptureGuard
    {
      std::function<void()> onExit;
      ~CaptureGuard() { onExit(); }
    } captureGuard{[&]()
    {
      if (!captureActive) return;
      for (int localRank : localRanksToExecute)
      {
        if (hipSetDevice(this->deviceIds[localRank]) != hipSuccess) continue;
        for (hipStream_t stream : this->streams[groupId][localRank])
        {
          hipStreamCaptureStatus captureStatus = hipStreamCaptureStatusNone;
          if (hipStreamIsCapturing(stream, &captureStatus) != hipSuccess ||
              captureStatus == hipStreamCaptureStatusNone)
            continue;
          hipGraph_t graph = nullptr;
          if (hipStreamEndCapture(stream, &graph) == hipSuccess && graph != nullptr)
            (void)hipGraphDestroy(graph);
        }
      }
    }};

    // Start HIP graph stream capture if requested
    if (useHipGraph)
    {
      captureActive = true;
      for (int localRank : localRanksToExecute)
      {
        if (this->verbose) TEST_INFO("Capturing stream for group %d rank %d", groupId, localRank);
        CHECK_HIP(hipSetDevice(this->deviceIds[localRank]));
        for (int i = 0; i < this->numStreamsPerGroup[groupId]; i++)
        {
          CHECK_HIP(hipStreamBeginCapture(this->streams[groupId][localRank][i], hipStreamCaptureModeRelaxed));
        }
      }
    }

    int const numThreadsToUse = this->useRankThreading ? numRanksToExecute : 1;

    #ifdef ENABLE_OPENMP
    if (this->useRankThreading && numThreadsToUse > 1)
    {
      int observedThreads = 1;
      #pragma omp parallel num_threads(numThreadsToUse) reduction(max : observedThreads)
      {
        observedThreads = omp_get_num_threads();
      }
      if (observedThreads != numThreadsToUse)
      {
        TEST_ERROR("UT_MULTITHREAD requested %d rank threads, but OpenMP created %d",
                   numThreadsToUse, observedThreads);
        return TEST_FAIL;
      }
    }
    #endif

    // Submit one collective without changing NCCL group scope. The caller owns
    // group start/end so serial execution can group all local communicators,
    // while rank-threaded execution can keep one thread-local group per rank.
    auto submitCollective = [&](int const localRank, int const collId) -> ErrCode
    {
      CollectiveArgs& collArg = this->collArgs[groupId][localRank][collId];
      switch (collArg.funcType)
      {
      case ncclCollBroadcast:
        CHILD_NCCL_CALL(ncclBroadcast(
                                   collArg.inputGpu.ptr,
                                   collArg.outputGpu.ptr,
                                   collArg.numInputElements,
                                   collArg.dataType,
                                   collArg.options.root,
                                   this->comms[localRank],
                                   this->streams[groupId][localRank][collArg.streamIdx]),
                        "ncclBroadcast");
        break;
      case ncclCollReduce:
        CHILD_NCCL_CALL(ncclReduce(
                                collArg.inputGpu.ptr,
                                collArg.outputGpu.ptr,
                                collArg.numInputElements,
                                collArg.dataType,
                                collArg.options.redOp,
                                collArg.options.root,
                                this->comms[localRank],
                                this->streams[groupId][localRank][collArg.streamIdx]),
                        "ncclReduce");
        break;
      case ncclCollAllGather:
        CHILD_NCCL_CALL(ncclAllGather(
                                   collArg.inputGpu.ptr,
                                   collArg.outputGpu.ptr,
                                   collArg.numInputElements,
                                   collArg.dataType,
                                   this->comms[localRank],
                                   this->streams[groupId][localRank][collArg.streamIdx]),
                        "ncclAllGather");
        break;
      case ncclCollReduceScatter:
        CHILD_NCCL_CALL(ncclReduceScatter(
                                       collArg.inputGpu.ptr,
                                       collArg.outputGpu.ptr,
                                       collArg.numOutputElements,
                                       collArg.dataType,
                                       collArg.options.redOp,
                                       this->comms[localRank],
                                       this->streams[groupId][localRank][collArg.streamIdx]),
                        "ncclReduceScatter");
        break;
      case ncclCollAllReduce:
        if (collArg.options.useBias)
        {
          CHILD_NCCL_CALL(ncclAllReduceWithBias(
                                     collArg.inputGpu.ptr,
                                     collArg.outputGpu.ptr,
                                     collArg.numInputElements,
                                     collArg.dataType,
                                     collArg.options.redOp,
                                     this->comms[localRank],
                                     this->streams[groupId][localRank][collArg.streamIdx],
                                     collArg.options.biasPtr),
                          "ncclAllReduceWithBias");
        }
        else
        {
          CHILD_NCCL_CALL(ncclAllReduce(
                                     collArg.inputGpu.ptr,
                                     collArg.outputGpu.ptr,
                                     collArg.numInputElements,
                                     collArg.dataType,
                                     collArg.options.redOp,
                                     this->comms[localRank],
                                     this->streams[groupId][localRank][collArg.streamIdx]),
                          "ncclAllReduce");
        }
        break;
      case ncclCollGather:
        CHILD_NCCL_CALL(ncclGather(
                                collArg.inputGpu.ptr,
                                collArg.outputGpu.ptr,
                                collArg.numInputElements,
                                collArg.dataType,
                                collArg.options.root,
                                this->comms[localRank],
                                this->streams[groupId][localRank][collArg.streamIdx]),
                        "ncclGather");
        break;
      case ncclCollScatter:
        CHILD_NCCL_CALL(ncclScatter(
                                 collArg.inputGpu.ptr,
                                 collArg.outputGpu.ptr,
                                 collArg.numOutputElements,
                                 collArg.dataType,
                                 collArg.options.root,
                                 this->comms[localRank],
                                 this->streams[groupId][localRank][collArg.streamIdx]),
                        "ncclScatter");
        break;
      case ncclCollAlltoAll:
        CHILD_NCCL_CALL(ncclAlltoAll(
                                  collArg.inputGpu.ptr,
                                  collArg.outputGpu.ptr,
                                  collArg.numInputElements / collArg.totalRanks,
                                  collArg.dataType,
                                  this->comms[localRank],
                                  this->streams[groupId][localRank][collArg.streamIdx]),
                        "ncclAlltoAll");
        break;
      case ncclCollAlltoAllv:
        CHILD_NCCL_CALL(ncclAlltoAllv(
                                   collArg.inputGpu.ptr,
                                   collArg.options.sendcounts + (this->rankOffset + localRank)*this->totalRanks,
                                   collArg.options.sdispls + (this->rankOffset + localRank)*this->totalRanks,
                                   collArg.outputGpu.ptr,
                                   collArg.options.recvcounts + (this->rankOffset + localRank)*this->totalRanks,
                                   collArg.options.rdispls + (this->rankOffset + localRank)*this->totalRanks,
                                   collArg.dataType,
                                   this->comms[localRank],
                                   this->streams[groupId][localRank][collArg.streamIdx]),
                        "ncclAlltoAllv");
        break;
      case ncclCollSend:
        CHILD_NCCL_CALL(ncclSend(
                              collArg.inputGpu.ptr,
                              collArg.numInputElements,
                              collArg.dataType,
                              collArg.options.root,
                              this->comms[localRank],
                              this->streams[groupId][localRank][collArg.streamIdx]),
                        "ncclSend");
        break;
      case ncclCollRecv:
        CHILD_NCCL_CALL(ncclRecv(
                              collArg.outputGpu.ptr,
                              collArg.numOutputElements,
                              collArg.dataType,
                              collArg.options.root,
                              this->comms[localRank],
                              this->streams[groupId][localRank][collArg.streamIdx]),
                        "ncclRecv");
        break;
      default:
        TEST_ERROR("Unknown func type %d", collArg.funcType);
        return TEST_FAIL;
      }
      return TEST_SUCCESS;
    };

    if (this->useRankThreading)
    {
      // NCCL group state is thread-local. Keep each rank on one OpenMP thread
      // for the complete group so multi-collective and send/recv groups retain
      // their original boundaries.
      int errCode = TEST_SUCCESS;
      #ifdef ENABLE_OPENMP
      #pragma omp parallel for num_threads(numThreadsToUse) reduction(max : errCode)
      #endif
      for (int rankIdx = 0; rankIdx < numRanksToExecute; ++rankIdx)
      {
        int const localRank = localRanksToExecute[rankIdx];
        if (this->verbose)
          TEST_INFO("Group %d rank %d submitting %d collectives on thread %d",
                    groupId, localRank, this->numCollectivesInGroup[groupId], getThreadId());

        CHECK_HIP_RANK(errCode, hipSetDevice(this->deviceIds[localRank]));
        CHILD_NCCL_CALL_RANK(
          errCode, ncclGroupStart(), "ncclGroupStart ExecuteCollectives rank thread");

        for (int collId = 0; collId < this->numCollectivesInGroup[groupId]; ++collId)
        {
          if (this->verbose)
            TEST_INFO("Group %d collective %d running rank %d on thread %d",
                      groupId, collId, localRank, getThreadId());
          ErrCode const collStatus = submitCollective(localRank, collId);
          if (collStatus != TEST_SUCCESS)
          {
            errCode = collStatus;
            break;
          }
        }

        if (this->useBlocking == false)
        {
          ncclResult_t const groupEndState = ncclGroupEnd();
          if (groupEndState != ncclSuccess && groupEndState != ncclInProgress)
          {
            TEST_ERROR("Child process %d fails rank-threaded ncclGroupEnd with code %d",
                       this->childId, groupEndState);
            errCode = TEST_FAIL;
          }
          else
          {
            CHILD_NCCL_CALL_NON_BLOCKING_RANK(
              errCode, "ncclCommGetAsyncErrorExecuteCollectives rank thread", localRank);
          }
        }
        else
        {
          CHILD_NCCL_CALL_RANK(
            errCode, ncclGroupEnd(), "ncclGroupEnd ExecuteCollectives rank thread");
        }

        if (this->verbose)
          TEST_INFO("Group %d done rank %d on thread %d", groupId, localRank, getThreadId());
      }
      CHECK_CALL(static_cast<ErrCode>(errCode));
    }
    else
    {
      // Serial rank iteration needs one group spanning all local communicators.
      CHILD_NCCL_CALL(ncclGroupStart(), "ncclGroupStart ExecuteCollectives");
      ErrCode const submitStatus = [&]() -> ErrCode
      {
        for (int collId = 0; collId < this->numCollectivesInGroup[groupId]; ++collId)
        {
          for (int localRank : localRanksToExecute)
          {
            CHECK_HIP(hipSetDevice(this->deviceIds[localRank]));
            CHECK_CALL(submitCollective(localRank, collId));
            if (this->useBlocking == false)
              CHILD_NCCL_CALL_NON_BLOCKING("ncclCommGetAsyncErrorExecuteCollectives", localRank);
          }
        }
        return TEST_SUCCESS;
      }();

      CHECK_CALL(this->EndGroup(submitStatus, "ncclGroupEnd ExecuteCollectives"));
    }

    // Instantiate and launch HIP graph if requested
    if (useHipGraph)
    {
      for (int localRank : localRanksToExecute)
      {
        if (this->verbose) TEST_INFO("Ending stream capture for rank %d", localRank);
        CHECK_HIP(hipSetDevice(this->deviceIds[localRank]));
        for (int i = 0; i < this->numStreamsPerGroup[groupId]; i++)
        {
          CHECK_HIP(hipStreamEndCapture(this->streams[groupId][localRank][i], &this->graphs[groupId][localRank][i]));

          // if (this->verbose)
          // {
          //   size_t numNodes;
          //   hipGraphNode_t* nodes;
          //   CHECK_HIP(hipGraphGetNodes(graphs[localRank][i], nodes, &numNodes));
          //   TEST_INFO("Graph for rank %d stream %d has %lu nodes", localRank, i, numNodes);
          // }
        }

        if (this->verbose) TEST_INFO("Instantiating executable graph for group %d rank %d", groupId, localRank);
        for (int i = 0; i < this->numStreamsPerGroup[groupId]; i++)
        {
          CHECK_HIP(hipGraphInstantiate(&this->graphExecs[groupId][localRank][i], this->graphs[groupId][localRank][i], NULL, NULL, 0));
          graphEnabled[groupId][localRank][i] = true;
        }
      }
      captureActive = false;
    }
    else
    {
      if (this->verbose)
        TEST_INFO("Child %d submits group call.  Waiting for completion", this->childId);
    }

    // Synchronize
    std::vector<hipStream_t> streamsToComplete;
    for (int localRank : localRanksToExecute)
    {
      for (int i = 0; i < this->numStreamsPerGroup[groupId]; i++)
        streamsToComplete.push_back(this->streams[groupId][localRank][i]);
    }
    int usElapsed = 0, timedout = 0;
    using namespace std::chrono;
    using Clock = std::chrono::high_resolution_clock;
    if (this->verbose) TEST_INFO("Starting sychronization and timing");
    const auto start = Clock::now();
    while (!streamsToComplete.empty() && usElapsed < timeoutUs)
    {
      for (int i = 0; i < streamsToComplete.size(); i++)
      {
        if (hipStreamQuery(streamsToComplete[i]) == hipSuccess)
        {
          streamsToComplete.erase(streamsToComplete.begin() + i);
          i--;
        }
      }
      usElapsed = duration_cast<microseconds>(Clock::now() - start).count();
      if (!streamsToComplete.empty()) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
      }
    }

    // timed out
    if (!streamsToComplete.empty())
    {
      if (this->verbose) TEST_INFO("Collective timed out, aborting");
      for (int localRank : localRanksToExecute)
      {
        CHECK_HIP(hipSetDevice(this->deviceIds[localRank]));
        ncclCommAbort(this->comms[localRank]);
        timedout = 1;
      }
    }

    // extra sync to flush GPU cache for validation later
    // TODO: remove this after figuring out & fixing the exact behavior
    // of fencing between kernels and at hipStreamQuery
    for (int localRank : localRanksToExecute)
    {
      if (this->verbose) TEST_INFO("Starting synchronization for group %d rank %d", groupId, localRank);
      CHECK_HIP(hipSetDevice(this->deviceIds[localRank]));
      for (int i = 0; i < this->numStreamsPerGroup[groupId]; i++)
      {
        CHECK_HIP(hipStreamSynchronize(this->streams[groupId][localRank][i]));
      }
      CHECK_HIP(hipDeviceSynchronize());
    }

    if (this->printValues)
    {
      for (int collId = 0; collId < this->numCollectivesInGroup[groupId]; ++collId)
        for (int localRank : localRanksToExecute)
        {
          CollectiveArgs const& collArg = this->collArgs[groupId][localRank][collId];
          CHECK_HIP(hipSetDevice(this->deviceIds[localRank]));
          int numOutputElementsToPrint = (this->printValues < 0 ? collArg.numOutputElements : this->printValues);
          size_t const numOutputBytes = numOutputElementsToPrint * DataTypeToBytes(collArg.dataType);
          CHECK_HIP(hipMemcpy(collArg.outputCpu.ptr, collArg.outputGpu.ptr, numOutputBytes, hipMemcpyDeviceToHost));
          printf("[ DEBUG    ] Rank %02d Group %d Coll %d %-10s: %s\n", collArg.globalRank, groupId, collId, "Output",
                 collArg.outputCpu.ToString(collArg.dataType, numOutputElementsToPrint).c_str());

          // Device-data mode builds the reference in expectedGpu; the host 'expected'
          // buffer is unused there, so copy it back before printing.
          if (collArg.expectedOnDevice)
          {
            CHECK_HIP(hipMemcpy(collArg.expected.ptr, collArg.expectedGpu.ptr, numOutputBytes, hipMemcpyDeviceToHost));
          }

          printf("[ DEBUG    ] Rank %02d Group %d Coll %d %-10s: %s\n", collArg.globalRank, groupId, collId, "Expected",
                 collArg.expected.ToString(collArg.dataType, numOutputElementsToPrint).c_str());
        }
    }

    if (timedout)
    {
      TEST_ERROR("Child %d timed out and exceeded limit %d us in ExecuteCollectives()", this->childId, timeoutUs);
      return TEST_TIMEOUT;
    }

    if (this->verbose) TEST_INFO("Child %d finishes ExecuteCollectives()", this->childId);
    return TEST_SUCCESS;
  }

  ErrCode TestBedChild::ValidateResults()
  {
    // Read values sent by parent [see TestBed::ValidateResults()]
    int globalRank = -1;
    int groupId = -1;
    int collId = -1;
    PIPE_READ(globalRank);
    PIPE_READ(groupId);
    PIPE_READ(collId);

    if (this->verbose) TEST_INFO("Child %d begins ValidateResults()", this->childId);

    if (globalRank < this->rankOffset || (this->rankOffset +  static_cast<int>(comms.size()) <= globalRank))
    {
      TEST_ERROR("Child %d does not contain rank %d", this->childId, globalRank);
      return TEST_FAIL;
    }
    int const localRank = globalRank - rankOffset;
    if (!this->IsValidGroupId(groupId, "ValidateResults")) return TEST_FAIL;
    if (localRank >= static_cast<int>(this->collArgs[groupId].size()))
    {
      TEST_ERROR("Child %d Rank %d: localRank %d out of bounds for groupId %d (size: %zu)",
               this->childId, globalRank, localRank, groupId, this->collArgs[groupId].size());
      return TEST_FAIL;
    }
    CHECK_HIP(hipSetDevice(this->deviceIds[localRank]));
    CHECK_HIP(hipDeviceSynchronize());

    ErrCode status = TEST_SUCCESS;
    for (int collIdx = 0; collIdx < collArgs[groupId][localRank].size(); ++collIdx)
    {
      if (collId == -1 || collId == collIdx)
      {
        if (this->verbose) TEST_INFO("Rank %d on child %d validating collective %d in group %d results",
                                globalRank, this->childId, collIdx, groupId);
        if (this->collArgs[groupId][localRank][collIdx].ValidateResults() != TEST_SUCCESS)
        {
          TEST_ERROR("Rank %d Group %d Collective %d output does not match expected", globalRank, groupId, collIdx);
          status = TEST_FAIL;
        }
      }
    }
    if (this->verbose) TEST_INFO("Child %d finishes ValidateResults() with status %s", this->childId,
                            status == TEST_SUCCESS ? "SUCCESS" : "FAIL");
    return status;
  }

  ErrCode TestBedChild::LaunchGraphs()
  {
    int groupId;
    PIPE_READ(groupId);

    if (this->verbose) TEST_INFO("Child %d begins LaunchGraphs for group %d", this->childId, groupId);
    if (groupId < 0 ||
        groupId >= static_cast<int>(this->graphExecs.size()) ||
        groupId >= static_cast<int>(this->graphEnabled.size()) ||
        groupId >= static_cast<int>(this->streams.size()) ||
        groupId >= static_cast<int>(this->numStreamsPerGroup.size()))
    {
      TEST_ERROR("Child %d: Invalid groupId %d for LaunchGraphs", this->childId, groupId);
      return TEST_FAIL;
    }

    size_t const numLocalRanks = this->deviceIds.size();
    if (this->graphEnabled[groupId].size() != numLocalRanks ||
        this->graphExecs[groupId].size() != numLocalRanks ||
        this->streams[groupId].size() != numLocalRanks)
    {
      TEST_ERROR("Child %d: Graph state for group %d is not initialized for all %zu local ranks",
                 this->childId, groupId, numLocalRanks);
      return TEST_FAIL;
    }

    size_t const numStreams = this->numStreamsPerGroup[groupId];
    for (size_t localRank = 0; localRank < numLocalRanks; ++localRank)
    {
      if (this->graphEnabled[groupId][localRank].size() != numStreams ||
          this->graphExecs[groupId][localRank].size() != numStreams ||
          this->streams[groupId][localRank].size() != numStreams)
      {
        TEST_ERROR("Child %d: Graph state for group %d rank %zu is not initialized for all %zu streams",
                   this->childId, groupId, localRank, numStreams);
        return TEST_FAIL;
      }
    }

    for (size_t localRank = 0; localRank < numLocalRanks; ++localRank) {
      CHECK_HIP(hipSetDevice(this->deviceIds[localRank]));

      for (size_t streamIdx = 0; streamIdx < numStreams; ++streamIdx)
      {
        if (this->graphEnabled[groupId][localRank][streamIdx]){
          if (this->verbose) TEST_INFO("Launch graph for group %d rank %zu stream %zu", groupId, localRank, streamIdx);
          CHECK_HIP(hipGraphLaunch(this->graphExecs[groupId][localRank][streamIdx], this->streams[groupId][localRank][streamIdx]));
        }
      }
    }

    if (this->verbose) TEST_INFO("Child %d finishes LaunchGraphs for group %d", this->childId, groupId);
    return TEST_SUCCESS;
  }

  ErrCode TestBedChild::DeregisterMemInternal_impl(int groupId, int collId, int localRank)
  {
    if (this->verbose) TEST_INFO("Child %d begins DeregisterMemInternal", this->childId);
    CHECK_HIP(hipSetDevice(this->deviceIds[localRank]));

    // Enclose RCCL deregistration calls in a group for safety
    CHILD_NCCL_CALL(ncclGroupStart(), "ncclGroupStart DeregisterMem");

    ErrCode const deregisterStatus = [&]() -> ErrCode
    {
      for (size_t collIdx = 0; collIdx < collArgs[groupId][localRank].size(); ++collIdx)
      {
        CollectiveArgs& collArg = this->collArgs[groupId][localRank][collIdx];
        if (collId != -1 && collId != static_cast<int>(collIdx)) continue;

        if (this->verbose)
        {
           TEST_INFO("Child %d deregistering memory for collective %zu in group %d",
                  this->childId, collIdx, groupId);
        }
        // =====================================================================
        // 1. Deregister Symmetric Windows (ncclCommWindowDeregister)
        // =====================================================================
        if (this->memAllocType == MEM_ALLOC_SYMMETRIC_WIN)
        {
          if (collArg.inPlace)
          {
            // In-place mode: RegisterMem registers a single window into outputWin
            if (collArg.outputWin != nullptr)
            {
              CHILD_NCCL_CALL(
                ncclCommWindowDeregister(this->comms[localRank], collArg.outputWin),
                "ncclCommWindowDeregister (in-place)");
            }
            collArg.inputWin = nullptr;
            collArg.outputWin = nullptr;
          }
          else
          {
            // Out-of-place mode: Distinct window handles
            if (collArg.inputWin != nullptr)
            {
              CHILD_NCCL_CALL(
                ncclCommWindowDeregister(this->comms[localRank], collArg.inputWin),
                "ncclCommWindowDeregister (input)");
              collArg.inputWin = nullptr;
            }

            if (collArg.outputWin != nullptr)
            {
              CHILD_NCCL_CALL(
               ncclCommWindowDeregister(this->comms[localRank], collArg.outputWin),
               "ncclCommWindowDeregister (output)");
              collArg.outputWin = nullptr;
            }
          }
        }

        // =====================================================================
        // 2. Deregister Standard Buffers (ncclCommDeregister)
        // =====================================================================
        if (collArg.inputRegHandle != nullptr)
        {
          CHILD_NCCL_CALL(
            ncclCommDeregister(this->comms[localRank], collArg.inputRegHandle),
            "ncclCommDeregister");
          collArg.inputRegHandle = nullptr;
        }
        if (collArg.outputRegHandle != nullptr)
        {
          CHILD_NCCL_CALL(
            ncclCommDeregister(this->comms[localRank], collArg.outputRegHandle),
            "ncclCommDeregister");
          collArg.outputRegHandle = nullptr;
        }
      }
      return TEST_SUCCESS;
    }();
    CHECK_CALL(this->EndGroup(deregisterStatus, "ncclGroupEnd DeregisterMem"));
    if (this->verbose) TEST_INFO("Child %d finishes DeregisterMemInternal", this->childId);
    return TEST_SUCCESS;
  }

  ErrCode TestBedChild::DeallocateMemInternal_impl(int groupId, int collId, int localRank)
  {
    if (this->verbose) TEST_INFO("Child %d begins DeallocateMemInternal", this->childId);
    // Release every collective even if one fails; a pooled worker would otherwise
    // keep the remaining buffers alive into the next config.
    ErrCode status = TEST_SUCCESS;
    for (size_t collIdx = 0; collIdx < collArgs[groupId][localRank].size(); ++collIdx)
    {
      if (collId != -1 && collId != static_cast<int>(collIdx)) continue;

      CollectiveArgs& collArg = this->collArgs[groupId][localRank][collIdx];
      if (collArg.DeallocateMem() != TEST_SUCCESS) status = TEST_FAIL;

      if (collArg.options.scalarMode >= 0)
      {
        ncclResult_t const result = ncclRedOpDestroy(collArg.options.redOp, this->comms[localRank]);
        if (result != ncclSuccess)
        {
          TEST_ERROR("Child process %d fails NCCL call ncclRedOpDestroy with code %d", this->childId, result);
          status = TEST_FAIL;
        }
        else if (this->verbose)
        {
          TEST_INFO("Child %d destroys custom redop %d for collective %zu in group %d",
                  this->childId, collArg.options.redOp, collIdx, groupId);
        }
      }
    }
    if (this->verbose) TEST_INFO("Child %d finishes DeallocateMemInternal", this->childId);
    return status;
  }

  ErrCode TestBedChild::DeallocateMem()
  {
    if (this->verbose) TEST_INFO("Child %d begins DeallocateMem", this->childId);
    // Read values sent by parent [matches IPC pipe format]
    int globalRank, groupId, collId;
    PIPE_READ(globalRank);
    PIPE_READ(groupId);
    PIPE_READ(collId);

    if (!this->IsValidGroupId(groupId, "DeallocateMem")) return TEST_FAIL;
    if (globalRank < this->rankOffset || (this->rankOffset + static_cast<int>(comms.size()) <= globalRank))
    {
      TEST_ERROR("Child %d does not contain rank %d", this->childId, globalRank);
      return TEST_FAIL;
    }

    int const localRank = globalRank - rankOffset;
    // Free the buffers even if deregistration fails, so a pooled worker does not
    // carry them into the next config.
    ErrCode const deregisterStatus = this->DeregisterMemInternal_impl(groupId, collId, localRank);
    ErrCode const deallocateStatus = this->DeallocateMemInternal_impl(groupId, collId, localRank);
    return deregisterStatus != TEST_SUCCESS ? deregisterStatus : deallocateStatus;
  }

  ErrCode TestBedChild::DestroyComms()
  {
    if (this->verbose) TEST_INFO("Child %d begins DestroyComms", this->childId);

    // Graphs are captured on this config's streams; free any that DestroyGraphs
    // did not, before the streams go. A pooled worker would otherwise keep them
    // alive into the next config.
    ErrCode graphStatus = TEST_SUCCESS;
    for (int groupId = 0; groupId < static_cast<int>(this->graphs.size()); ++groupId)
    {
      if (this->ReleaseGraphHandles(groupId) != TEST_SUCCESS) graphStatus = TEST_FAIL;
    }
    this->graphs.clear();
    this->graphExecs.clear();
    this->graphEnabled.clear();

    // Release comms.  Finalize waits on an intra-node barrier with the other
    // host-local ranks, so when this child owns several ranks they must be
    // finalized inside one group call: finalizing them one at a time blocks the
    // first rank in the barrier before any of the others can enter it.
    // Every step below runs even if an earlier one fails, so a pooled worker does
    // not carry live comms or streams into the next config.
    ErrCode status = graphStatus;
    bool hasActiveComm = false;
    for (ncclComm_t comm : this->comms)
      hasActiveComm |= comm != nullptr;

    if (hasActiveComm)
    {
      ncclResult_t const groupStartState = ncclGroupStart();
      if (groupStartState != ncclSuccess)
      {
        TEST_ERROR("Child process %d fails NCCL call ncclGroupStart with code %d", this->childId, groupStartState);
        status = TEST_FAIL;
      }
      else
      {
        ErrCode const finalizeStatus = [&]() -> ErrCode
        {
          for (int i = 0; i < this->comms.size(); ++i)
          {
            if (this->comms[i] == nullptr) continue;
            CHILD_NCCL_CALL(ncclCommFinalize(this->comms[i]), "ncclCommFinalize");
          }
          return TEST_SUCCESS;
        }();

        if (this->useBlocking == false)
        {
          ncclResult_t const groupEndState = ncclGroupEnd();
          if (groupEndState != ncclSuccess && groupEndState != ncclInProgress)
          {
            TEST_ERROR("Child %d ncclGroupEnd failed during communicator finalization with error %d",
                       this->childId, groupEndState);
            status = TEST_FAIL;
          }
          else
          {
            if (finalizeStatus != TEST_SUCCESS) status = TEST_FAIL;
            for (int i = 0; i < this->comms.size(); ++i)
            {
              if (this->comms[i] == nullptr) continue;
              ErrCode const pollStatus = [&]() -> ErrCode
              {
                CHILD_NCCL_CALL_NON_BLOCKING("ncclCommGetAsyncErrorCommFinalize", i);
                return TEST_SUCCESS;
              }();
              if (pollStatus != TEST_SUCCESS) status = TEST_FAIL;
            }
          }
        }
        else if (this->EndGroup(finalizeStatus, "ncclGroupEnd") != TEST_SUCCESS)
        {
          status = TEST_FAIL;
        }
      }
    }

    for (int i = 0; i < this->comms.size(); ++i)
    {
      if (this->comms[i] == nullptr) continue;
      ncclResult_t const destroyState = ncclCommDestroy(this->comms[i]);
      if (destroyState != ncclSuccess)
      {
        TEST_ERROR("Child process %d fails NCCL call ncclCommDestroy with code %d", this->childId, destroyState);
        status = TEST_FAIL;
      }
      this->comms[i] = nullptr;
    }

    // 2. Safely release HIP streams with correct device context
    for (int i = 0; i < this->streams.size(); ++i)
    {
      for (int j = 0; j < this->streams[i].size(); ++j)
      {
        // Switch active GPU context to the device that owns this stream group
        if (hipSetDevice(this->deviceIds[j]) != hipSuccess)
        {
          TEST_ERROR("Child %d unable to switch to GPU %d to destroy streams", this->childId, this->deviceIds[j]);
          status = TEST_FAIL;
          continue;
        }

        for (int k = 0; k < this->streams[i][j].size(); ++k)
        {
          // Avoid destroying null handles or the default stream (0)
          if (this->streams[i][j][k] != nullptr)
          {
            if (hipStreamDestroy(this->streams[i][j][k]) != hipSuccess)
            {
              TEST_ERROR("Child %d unable to destroy stream %d for rank %d", this->childId, k, j);
              status = TEST_FAIL;
            }
            this->streams[i][j][k] = nullptr; // Avoid double-destruction
          }
        }
      }
    }

    this->comms.clear();
    this->streams.clear();
    if (this->verbose) TEST_INFO("Child %d finishes DestroyComms", this->childId);
    return status;
  }

  bool TestBedChild::IsValidGroupId(int const groupId, char const* handler) const
  {
    if (groupId >= 0 && groupId < static_cast<int>(this->collArgs.size())) return true;
    TEST_ERROR("Child %d %s received invalid groupId %d (config has %zu group calls)",
               this->childId, handler, groupId, this->collArgs.size());
    return false;
  }

  void TestBedChild::AbortComms()
  {
    for (int localRank = 0; localRank < static_cast<int>(this->comms.size()); ++localRank)
    {
      if (this->comms[localRank] == nullptr) continue;
      if (localRank < static_cast<int>(this->deviceIds.size()))
        (void)hipSetDevice(this->deviceIds[localRank]);
      ncclResult_t const abortState = ncclCommAbort(this->comms[localRank]);
      if (abortState != ncclSuccess)
        TEST_ERROR("Child %d fails ncclCommAbort for rank %d with code %d", this->childId, localRank, abortState);
      this->comms[localRank] = nullptr;
    }
  }

  ErrCode TestBedChild::EndGroup(ErrCode const bodyStatus, char const* msg)
  {
    ncclResult_t const groupEndState = ncclGroupEnd();
    if (groupEndState == ncclInProgress && !this->useBlocking)
    {
      // Non-blocking comms finish the group asynchronously; wait on every comm.
      ErrCode pollStatus = TEST_SUCCESS;
      for (int localRank = 0; localRank < static_cast<int>(this->comms.size()); ++localRank)
      {
        if (this->comms[localRank] == nullptr) continue;
        ErrCode const rankStatus = [&]() -> ErrCode
        {
          CHECK_HIP(hipSetDevice(this->deviceIds[localRank]));
          CHILD_NCCL_CALL_NON_BLOCKING(msg, localRank);
          return TEST_SUCCESS;
        }();
        if (rankStatus != TEST_SUCCESS) pollStatus = TEST_FAIL;
      }
      return pollStatus != TEST_SUCCESS ? pollStatus : bodyStatus;
    }
    if (groupEndState != ncclSuccess)
    {
      TEST_ERROR("Child process %d fails NCCL call %s with code %d", this->childId, msg, groupEndState);
      return TEST_FAIL;
    }
    return bodyStatus;
  }

  ErrCode TestBedChild::ReleaseGraphHandles(int const groupId)
  {
    if (groupId < 0 || groupId >= static_cast<int>(this->graphs.size()) ||
        groupId >= static_cast<int>(this->graphExecs.size()))
      return TEST_SUCCESS;

    auto& groupGraphs = this->graphs[groupId];
    auto& groupExecs  = this->graphExecs[groupId];
    for (size_t localRank = 0; localRank < groupGraphs.size() || localRank < groupExecs.size(); ++localRank)
    {
      auto isLive = [](auto handle) { return handle != nullptr; };
      bool const hasLiveGraph =
        (localRank < groupGraphs.size() &&
         std::any_of(groupGraphs[localRank].begin(), groupGraphs[localRank].end(), isLive)) ||
        (localRank < groupExecs.size() &&
         std::any_of(groupExecs[localRank].begin(), groupExecs[localRank].end(), isLive));
      if (!hasLiveGraph) continue;

      if (localRank >= this->deviceIds.size())
      {
        TEST_ERROR("Child %d: live graph for group %d rank %zu has no owning device", this->childId, groupId, localRank);
        return TEST_FAIL;
      }
      CHECK_HIP(hipSetDevice(this->deviceIds[localRank]));
      CHECK_HIP(hipDeviceSynchronize());
      if (localRank < groupExecs.size())
      {
        for (hipGraphExec_t& graphExec : groupExecs[localRank])
        {
          if (graphExec == nullptr) continue;
          CHECK_HIP(hipGraphExecDestroy(graphExec));
          graphExec = nullptr;
        }
      }
      if (localRank < groupGraphs.size())
      {
        for (hipGraph_t& graph : groupGraphs[localRank])
        {
          if (graph == nullptr) continue;
          CHECK_HIP(hipGraphDestroy(graph));
          graph = nullptr;
        }
      }
    }

    if (groupId < static_cast<int>(this->graphEnabled.size()))
    {
      for (auto& rankEnabled : this->graphEnabled[groupId])
        std::fill(rankEnabled.begin(), rankEnabled.end(), false);
    }
    return TEST_SUCCESS;
  }

  ErrCode TestBedChild::DestroyGraphs()
  {
    if (this->verbose) TEST_INFO("Child %d begins DestroyGraphs", this->childId);

    int groupId = -1;
    PIPE_READ(groupId);

    if (groupId < 0 ||
      groupId >= static_cast<int>(this->graphs.size()) ||
      this->graphs[groupId].empty())
      {
        if (this->verbose) TEST_INFO("Child %d: No graphs present to destroy for group %d", this->childId, groupId);
        return TEST_SUCCESS;
      }

    // MUST synchronize streams FIRST to ensure in-flight graphs finish execution!
    for (int localRank = 0; localRank < this->deviceIds.size(); ++localRank)
    {
      CHECK_HIP(hipSetDevice(this->deviceIds[localRank]));
      for (int i = 0; i < this->numStreamsPerGroup[groupId]; ++i)
      {
        CHECK_HIP(hipStreamSynchronize(this->streams[groupId][localRank][i]));
      }
      CHECK_HIP(hipDeviceSynchronize());
    }

    // Release every live handle, including a graph captured but never
    // instantiated (graphEnabled stays false when hipGraphInstantiate fails).
    CHECK_CALL(this->ReleaseGraphHandles(groupId));

    this->graphs[groupId].clear();
    this->graphExecs[groupId].clear();
    this->graphEnabled[groupId].clear();

    if (this->verbose) TEST_INFO("Child %d finishes DestroyGraphs", this->childId);
    return TEST_SUCCESS;
  }

  ErrCode TestBedChild::RegisterMem()
  {
    if (this->verbose) TEST_INFO("Child %d begins RegisterMem()", this->childId);

    int groupId;
    int collId;
    int numRanks;

    PIPE_READ(groupId);
    PIPE_READ(collId);
    PIPE_READ(numRanks);

    if (numRanks < 0)
    {
      TEST_ERROR("Child %d received invalid registration rank count %d", this->childId, numRanks);
      return TEST_FAIL;
    }

    // Consume the whole payload before validating it, so a rejected request does
    // not leave rank ids in the pipe to be decoded as the next command.
    std::vector<int> globalRanks(numRanks);
    for (int& globalRank : globalRanks) PIPE_READ(globalRank);
    if (!this->IsValidGroupId(groupId, "RegisterMem")) return TEST_FAIL;

    if (numRanks > this->totalRanks)
    {
      TEST_ERROR("Child %d received invalid registration rank count %d", this->childId, numRanks);
      return TEST_FAIL;
    }

    std::vector<int> localRanks;
    localRanks.reserve(numRanks);
    for (int globalRank : globalRanks)
    {
      if (globalRank < this->rankOffset ||
          this->rankOffset + static_cast<int>(this->comms.size()) <= globalRank)
      {
        TEST_ERROR("Child %d does not contain registration rank %d", this->childId, globalRank);
        return TEST_FAIL;
      }
      localRanks.push_back(globalRank - this->rankOffset);
    }

    // Group registration across the selected local ranks managed by this child.
    CHILD_NCCL_CALL(ncclGroupStart(), "ncclGroupStart RegisterMem");

    ErrCode const registerStatus = [&]() -> ErrCode
    {
      for (int localRank : localRanks)
      {
        CHECK_HIP(hipSetDevice(this->deviceIds[localRank]));

        for (size_t collIdx = 0; collIdx < collArgs[groupId][localRank].size(); ++collIdx)
        {
          if (collId != -1 && collId != static_cast<int>(collIdx)) continue;

          CollectiveArgs& collArg = this->collArgs[groupId][localRank][collIdx];
          if (this->memAllocType == MEM_ALLOC_SYMMETRIC_WIN)
            CHECK_CALL(this->RegisterMemSymmetric(localRank, collArg));
          else if (collArg.userRegistered)
            CHECK_CALL(this->RegisterMemLegacy(localRank, collArg));
        }
      }
      return TEST_SUCCESS;
    }();

    // Completes handle exchange for all selected local ranks simultaneously.
    ErrCode const groupStatus = this->EndGroup(registerStatus, "ncclGroupEnd RegisterMem");
    if (groupStatus != TEST_SUCCESS)
    {
      // Peers that registered every window wait in the comm-wide symmetric
      // registration barrier at their group end; abort so they fail instead of hanging.
      if (this->memAllocType == MEM_ALLOC_SYMMETRIC_WIN) this->AbortComms();
      return groupStatus;
    }
    // Ensure GPU memory mapping / TLB invalidations complete on selected ranks.
    for (int localRank : localRanks)
    {
      CHECK_HIP(hipSetDevice(this->deviceIds[localRank]));
      CHECK_HIP(hipDeviceSynchronize());
    }

    ErrCode status = TEST_SUCCESS;
    if (this->memAllocType == MEM_ALLOC_SYMMETRIC_WIN)
    {
      for (int localRank : localRanks)
      {
        for (size_t collIdx = 0; collIdx < collArgs[groupId][localRank].size(); ++collIdx)
        {
          if (collId != -1 && collId != static_cast<int>(collIdx)) continue;

          CollectiveArgs& collArg = this->collArgs[groupId][localRank][collIdx];
          if (collArg.inPlace)
          {
            // Window registration is deferred until ncclGroupEnd(), so only
            // now is the returned handle final.
            collArg.inputWin = collArg.outputWin;
            if (collArg.outputWin == nullptr)
            {
              TEST_ERROR("Child %d rank %d group %d collective %zu: "
                         "in-place symmetric window registration returned a null handle",
                         this->childId, localRank, groupId, collIdx);
              status = TEST_FAIL;
            }
          }
          else
          {
            if (collArg.inputGpu.ptr != nullptr &&
                collArg.numInputBytesAllocated > 0 &&
                collArg.inputWin == nullptr)
            {
              TEST_ERROR("Child %d rank %d group %d collective %zu: "
                         "input symmetric window registration returned a null handle",
                         this->childId, localRank, groupId, collIdx);
              status = TEST_FAIL;
            }
            if (collArg.outputGpu.ptr != nullptr &&
                collArg.numOutputBytesAllocated > 0 &&
                collArg.outputWin == nullptr)
            {
              TEST_ERROR("Child %d rank %d group %d collective %zu: "
                         "output symmetric window registration returned a null handle",
                         this->childId, localRank, groupId, collIdx);
              status = TEST_FAIL;
            }
          }
        }
      }
    }

    if (this->verbose) TEST_INFO("Child %d finishes RegisterMem()", this->childId);
    return status;
  }

  // In-place collectives share one base allocation; see CollectiveArgs::AllocateMem.
  static void InPlaceBaseAllocation(CollectiveArgs const& collArg, void** buff, size_t* bufSize)
  {
    ncclFunc_t const fn = collArg.funcType;
    if (fn == ncclCollScatter || fn == ncclCollReduceScatter)
    {
      *buff = collArg.inputGpu.ptr;
      *bufSize = collArg.numInputBytesAllocated;
    }
    else if (fn == ncclCollGather || fn == ncclCollAllGather)
    {
      *buff = collArg.outputGpu.ptr;
      *bufSize = collArg.numOutputBytesAllocated;
    }
    else
    {
      *buff = collArg.inputGpu.ptr;
      *bufSize = std::max(collArg.numInputBytesAllocated, collArg.numOutputBytesAllocated);
    }
  }

  // Must run inside the caller's ncclGroupStart/End: the window handles are only
  // final after ncclGroupEnd().
  ErrCode TestBedChild::RegisterMemSymmetric(int const localRank, CollectiveArgs& collArg)
  {
    if (collArg.inPlace)
    {
      void* buff = nullptr;
      size_t bufSize = 0;
      InPlaceBaseAllocation(collArg, &buff, &bufSize);
      if (buff == nullptr || bufSize == 0) return TEST_SUCCESS;

      if (this->verbose)
        TEST_INFO("Child %d rank %d registers in-place symmetric window buff=%p bufSize=%zu",
                  this->childId, localRank, buff, bufSize);
      CHILD_NCCL_CALL(
        ncclCommWindowRegister(this->comms[localRank], buff, bufSize,
                               &(collArg.outputWin), NCCL_WIN_COLL_SYMMETRIC),
        "ncclCommWindowRegister (output in-place)");
      return TEST_SUCCESS;
    }

    if (collArg.inputGpu.ptr && collArg.numInputBytesAllocated > 0)
    {
      CHILD_NCCL_CALL(
        ncclCommWindowRegister(this->comms[localRank], collArg.inputGpu.ptr,
                               collArg.numInputBytesAllocated,
                               &(collArg.inputWin), NCCL_WIN_COLL_SYMMETRIC),
        "ncclCommWindowRegister (input)");
    }
    if (collArg.outputGpu.ptr && collArg.numOutputBytesAllocated > 0)
    {
      CHILD_NCCL_CALL(
        ncclCommWindowRegister(this->comms[localRank], collArg.outputGpu.ptr,
                               collArg.numOutputBytesAllocated,
                               &(collArg.outputWin), NCCL_WIN_COLL_SYMMETRIC),
        "ncclCommWindowRegister (output)");
    }
    return TEST_SUCCESS;
  }

  ErrCode TestBedChild::RegisterMemLegacy(int const localRank, CollectiveArgs& collArg)
  {
    if (collArg.inPlace)
    {
      void* buff = nullptr;
      size_t bufSize = 0;
      InPlaceBaseAllocation(collArg, &buff, &bufSize);
      if (buff == nullptr || bufSize == 0) return TEST_SUCCESS;

      CHILD_NCCL_CALL(
        ncclCommRegister(this->comms[localRank], buff, bufSize, &(collArg.outputRegHandle)),
        "ncclCommRegister (in-place)");
      return TEST_SUCCESS;
    }

    if (collArg.inputGpu.ptr && collArg.numInputBytesAllocated > 0)
    {
      CHILD_NCCL_CALL(
        ncclCommRegister(this->comms[localRank], collArg.inputGpu.ptr,
                         collArg.numInputBytesAllocated, &(collArg.inputRegHandle)),
        "ncclCommRegister (input)");
    }
    if (collArg.outputGpu.ptr && collArg.numOutputBytesAllocated > 0)
    {
      CHILD_NCCL_CALL(
        ncclCommRegister(this->comms[localRank], collArg.outputGpu.ptr,
                         collArg.numOutputBytesAllocated, &(collArg.outputRegHandle)),
        "ncclCommRegister (output)");
    }
    return TEST_SUCCESS;
  }
}
