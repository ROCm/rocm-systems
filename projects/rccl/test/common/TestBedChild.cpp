/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "TestBedChild.hpp"

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
      ERROR("Child process %d fails NCCL call %s with code %d\n", this->childId, msg, status); \
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
      ERROR("Child process %d fails NCCL call %s with code %d\n", this->childId, msg, ncclAsyncErr);  \
      RESULT(TEST_FAIL, ##RESULT_ARGS);                               \
    }                                                                 \
  } while (false)
#define CHILD_NCCL_CALL_NON_BLOCKING(msg, localRank) CHILD_NCCL_CALL_NON_BLOCKING_BASE(msg, localRank, RETURN_RESULT)

#define PIPE_READ(val) \
  if (read(childReadFd, &val, sizeof(val)) != sizeof(val)) return TEST_FAIL;

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
  }

  int TestBedChild::InitPipes()
  {
    // Prepare parent->child pipe
    int pipefd[2];
    if (pipe(pipefd) == -1)
    {
      ERROR("Unable to create parent->child pipe for child %d\n", this->childId);
      return TEST_FAIL;
    }
    this->childReadFd   = pipefd[0];
    this->parentWriteFd = pipefd[1];

    // Prepare child->parent pipe
    this->parentReadFd = -1;
    if (pipe(pipefd) == -1)
    {
      ERROR("Unable to create parent->child pipe for child %d\n", this->childId);
      return TEST_FAIL;
    }
    this->parentReadFd = pipefd[0];
    this->childWriteFd = pipefd[1];

    return TEST_SUCCESS;
  }

  void TestBedChild::StartExecutionLoop()
  {
    // Close unused ends of pipes
    close(this->parentWriteFd);
    close(this->parentReadFd);

    // Wait for commands from parent process
    if (verbose) INFO("Child %d enters execution loop\n", this->childId);
    #ifndef ENABLE_OPENMP
    if (verbose && useRankThreading) WARN("Multi-threaded ranks requires ENABLE_OPENMP to be defined\n");
    #endif
    int command;
    while (read(childReadFd, &command, sizeof(command)) > 0)
    {
      if (verbose) INFO("Child %d received command [%s]:\n", this->childId, ChildCommandNames[command]);;
      ErrCode status = TEST_SUCCESS;
      std::vector<char> retValBuf;
      switch(command)
      {
      case CHILD_GET_UNIQUE_ID   : status = GetUniqueId(retValBuf); break;
      case CHILD_INIT_COMMS      : status = InitComms();            break;
      case CHILD_SET_COLL_ARGS   : status = SetCollectiveArgs();    break;
      case CHILD_ALLOCATE_MEM    : status = AllocateMem();          break;
      case CHILD_PREPARE_DATA    : status = PrepareData();          break;
      case CHILD_EXECUTE_COLL    : status = ExecuteCollectives();   break;
      case CHILD_VALIDATE_RESULTS: status = ValidateResults();      break;
      case CHILD_LAUNCH_GRAPHS   : status = LaunchGraphs();         break;
      case CHILD_DEALLOCATE_MEM  : status = DeallocateMem();        break;
      case CHILD_DESTROY_COMMS   : status = DestroyComms();         break;
      case CHILD_DESTROY_GRAPHS  : status = DestroyGraphs();        break;
      case CHILD_STOP            : goto stop;
      default: exit(0);
      }

      // Send back acknowledgement to parent
      if (status == TEST_FAIL)
        ERROR("Child %d failed on command [%s]:\n", this->childId, ChildCommandNames[command]);
      if (write(childWriteFd, &status, sizeof(status)) < 0)
      {
        ERROR("Child %d write to parent failed: %s\n", this->childId, strerror(errno));
        break;
      }
      if (retValBuf.size() > 0 && write(childWriteFd, retValBuf.data(), retValBuf.size()) < 0) {
        ERROR("Child %d write return value to parent failed: %s\n", this->childId, strerror(errno));
        break;
      }
    }
  stop:
    if (verbose) INFO("Child %d exiting execution loop\n", this->childId);

    // Close child ends of pipe
    close(this->childReadFd);
    close(this->childWriteFd);

    exit(0);
  }

  ErrCode TestBedChild::GetUniqueId(std::vector<char>& retValBuf)
  {
    if (this->verbose) INFO("Child %d begins GetUniqueId()\n", this->childId);

    // Get a unique ID and pass it back to parent process
    ncclUniqueId id;
    CHILD_NCCL_CALL(ncclGetUniqueId(&id), "ncclGetUniqueId");
    retValBuf.resize(sizeof(id));
    memcpy(retValBuf.data(), &id, sizeof(id));

    if (this->verbose) INFO("Child %d finishes GetUniqueId()\n", this->childId);
    return TEST_SUCCESS;
  }

  ErrCode TestBedChild::InitComms()
  {
    if (this->verbose) INFO("Child %d begins InitComms()\n", this->childId);

    // Read values sent by parent [see TestBed::InitComms()]
    ncclUniqueId id;
    PIPE_READ(id);
    PIPE_READ(this->totalRanks);
    PIPE_READ(this->rankOffset);
    PIPE_READ(this->numGroupCalls);
    PIPE_READ(this->numCollectivesInGroup);
    PIPE_READ(this->useBlocking);
    bool useMultiRankPerGpu;
    PIPE_READ(useMultiRankPerGpu);
    PIPE_READ(this->numStreamsPerGroup);

    // Read the GPUs this child uses and prepare storage for collective args / datasets
    int numGpus;
    PIPE_READ(numGpus);
    this->deviceIds.resize(numGpus);
    this->streams.clear();
    this->streams.resize(this->numGroupCalls);
    this->collArgs.resize(this->numGroupCalls);
    for (int i = 0; i < this->numGroupCalls; i++)
    {
      this->collArgs[i].resize(numGpus);
      this->streams[i].resize(numGpus);
      for (int j = 0; j < numGpus; j++)
      {
        //PIPE_READ(this->deviceIds[j]);
        this->collArgs[i][j].clear();
        this->collArgs[i][j].resize(numCollectivesInGroup[i]);
        this->streams[i][j].resize(numStreamsPerGroup[i]);
      }
    }

    for (int i = 0; i < numGpus; i++)
      PIPE_READ(this->deviceIds[i]);

    // Initialize graphs
    this->graphs.resize(this->numGroupCalls);
    this->graphExecs.resize(this->numGroupCalls);
    this->graphEnabled.resize(this->numGroupCalls);

    // Initialize communicators
    comms.clear();
    comms.resize(numGpus);

    // Initialize within a group call to avoid deadlock when using multiple ranks per child
    ErrCode status = TEST_SUCCESS;
    CHILD_NCCL_CALL(ncclGroupStart(), "ncclGroupStart");
    for (int groupCallIdx = 0; groupCallIdx < this->numGroupCalls; ++groupCallIdx)
    {
      for (int localRank = 0; localRank < numGpus; ++localRank)
      {
        int const globalRank = this->rankOffset + localRank;
        int const currGpu = this->deviceIds[localRank];

        if (hipSetDevice(currGpu) != hipSuccess)
        {
          ERROR("Rank %d on child %d unable to switch to GPU %d\n", globalRank, this->childId, currGpu);
          status = TEST_FAIL;
          break;
        }

        for (int i = 0; i < this->numStreamsPerGroup[groupCallIdx]; i++)
        {
          if (hipStreamCreate(&(this->streams[groupCallIdx][localRank][i])) != hipSuccess)
          {
            ERROR("Rank %d on child %d unable to create stream %d for GPU %d in group %d\n", globalRank, this->childId, i, currGpu, groupCallIdx);
            status = TEST_FAIL;
            break;
          }
        }

        if (groupCallIdx == 0) {
          if (useMultiRankPerGpu)
          {
            //if (ncclCommInitRankMulti(&this->comms[localRank], this->totalRanks, id, globalRank, globalRank) != ncclSuccess)
            {
              ERROR("Rank %d on child %d unable to call ncclCommInitRankMulti\n", globalRank, this->childId);
              status = TEST_FAIL;
              break;
            }
          }
          else if (this->useBlocking == false)
          {
            // When non-blocking communicator is desired call ncclCommInitRankConfig with appropriate flag
            ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
            config.blocking = 0;
            ncclCommInitRankConfig(&this->comms[localRank], this->totalRanks, id, globalRank, &config);
            CHILD_NCCL_CALL_NON_BLOCKING("ncclCommGetAsyncErrorInitRankConfig", localRank);
          }
          else
          {
            if (ncclCommInitRank(&this->comms[localRank], this->totalRanks, id, globalRank) != ncclSuccess)
            {
              ERROR("Rank %d on child %d unable to call ncclCommInitRank\n", globalRank, this->childId);
              status = TEST_FAIL;
              break;
            }
          }
        }
      }
    }

    if (status == TEST_SUCCESS)
    {
      // Check if the communicator is non-blocking
      if (this->useBlocking == false)
      {
        // handle the ncclGroupEnd in case of non-blocking communication
        ncclResult_t Group_End_state = ncclGroupEnd();
        if (Group_End_state != ncclSuccess)
        {
          for (int localRank = 0; localRank < numGpus; ++localRank)
          {
            CHILD_NCCL_CALL_NON_BLOCKING("ncclCommGetAsyncErrorGroupEnd", localRank);
          }
        }
      }
      else
      {
        // In case of blocking communication just call ncclGroupEnd
        CHILD_NCCL_CALL(ncclGroupEnd(), "ncclGroupEnd");
      }
    }

    if (this->verbose) INFO("Child %d finishes InitComms() [%s]\n",
                            this->childId, status == TEST_SUCCESS ? "SUCCESS" : "FAIL");
    return status;
  }

  ErrCode TestBedChild::SetCollectiveArgs()
  {
    if (this->verbose) INFO("Child %d begins SetCollectiveArgs()\n", this->childId);

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

    if (globalRank < this->rankOffset || (this->rankOffset + comms.size() <= globalRank))
    {
      ERROR("Child %d does not contain rank %d\n", this->childId, globalRank);
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
        if (this->verbose) INFO("Rank %d on child %d sets collective %d in group %d [%s]\n",
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
          if (verbose) INFO("Child %d created custom redop %d for group %d collective %d\n",
                            this->childId, collArg.options.redOp, groupId, collIdx);
        }
      }
    }
    if (this->verbose) INFO("Child %d finishes SetCollectiveArgs()\n", this->childId);
    return TEST_SUCCESS;
  }

  ErrCode TestBedChild::AllocateMem()
  {
    if (this->verbose) INFO("Child %d begins AllocateMem()\n", this->childId);

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

    if (globalRank < this->rankOffset || (this->rankOffset + comms.size() <= globalRank))
    {
      ERROR("Child %d does not contain rank %d\n", this->childId, globalRank);
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
        if (collArg.userRegistered && (collArg.funcType == ncclCollSend || collArg.funcType == ncclCollRecv))
          CHILD_NCCL_CALL(ncclCommRegister(this->comms[localRank], collArg.inputGpu.ptr, collArg.numInputBytesAllocated, &(collArg.commRegHandle)),"ncclCommRegister");
        if (this->verbose) INFO("Rank %d on child %d allocates memory for collective %d in group %d on device %d (%s,%s,%s) Input: %p Output %p\n",
                                globalRank, this->childId, collIdx, groupId, this->deviceIds[localRank],
                                inPlace ? "in-place" : "out-of-place",
                                useManagedMem ? "managed" : "unmanaged",
                                userRegistered ? "user registered buffer" : "internal copy",
                                collArg.inputGpu.ptr,
                                collArg.outputGpu.ptr);
      }
    }

    if (this->verbose) INFO("Child %d finishes AllocateMem()\n", this->childId);
    return TEST_SUCCESS;
  }

  // Fill input memory with pre-known patterned based on rank
  ErrCode TestBedChild::PrepareData()
  {
    if (this->verbose) INFO("Child %d begins PrepareData()\n", this->childId);

    // Read values sent by parent [see TestBed::PrepareData()]
    int globalRank;
    int collId;
    int groupId;
    CollFuncPtr prepDataFunc;

    PIPE_READ(globalRank);
    PIPE_READ(groupId);
    PIPE_READ(collId);
    PIPE_READ(prepDataFunc);

    if (globalRank < this->rankOffset || (this->rankOffset + comms.size() <= globalRank))
    {
      ERROR("Child %d does not contain rank %d\n", this->childId, globalRank);
      return TEST_FAIL;
    }

    int const localRank = globalRank - rankOffset;
    CHECK_HIP(hipSetDevice(this->deviceIds[localRank]));

    for (int collIdx = 0; collIdx < collArgs[groupId][localRank].size(); ++collIdx)
    {
      if (collId == -1 || collId == collIdx)
      {
        if (this->verbose) INFO("Rank %d on child %d prepares data for collective %d in group %d\n",
                                globalRank, this->childId, collIdx, groupId);
        CHECK_CALL(this->collArgs[groupId][localRank][collIdx].PrepareData(prepDataFunc));
      }
    }
    if (this->verbose) INFO("Child %d finishes PrepareData()\n", this->childId);
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
    if (this->verbose) INFO("Child %d begins ExecuteCollectives() %s\n", this->childId, useHipGraph ? "(using hipGraphs)" : "");

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
    this->graphs[groupId].resize(numRanksToExecute);
    this->graphExecs[groupId].resize(numRanksToExecute);
    this->graphEnabled[groupId].resize(numRanksToExecute);
    for (int i = 0; i < numRanksToExecute; i++)
    {
      this->graphs[groupId][i].resize(this->numStreamsPerGroup[groupId]);
      this->graphExecs[groupId][i].resize(this->numStreamsPerGroup[groupId]);
      this->graphEnabled[groupId][i].resize(this->numStreamsPerGroup[groupId]);
    }

    // Start HIP graph stream capture if requested
    if (useHipGraph)
    {
      for (int localRank : localRanksToExecute)
      {
        if (this->verbose) INFO("Capturing stream for group %d rank %d\n", groupId, localRank);
        CHECK_HIP(hipSetDevice(this->deviceIds[localRank]));
        for (int i = 0; i < this->numStreamsPerGroup[groupId]; i++)
        {
          CHECK_HIP(hipStreamBeginCapture(this->streams[groupId][localRank][i], hipStreamCaptureModeRelaxed));
        }
      }
    }

    int numThreadsToUse = this->useRankThreading ? numRanksToExecute : 1;

    // Start group call
    CHILD_NCCL_CALL(ncclGroupStart(), "ncclGroupStart");

    // Loop over all collectives to be executed in group call
    for (int collId = 0; collId < this->numCollectivesInGroup[groupId]; ++collId)
    {
      // Loop over all local ranks
      if (this->verbose && this->useRankThreading)
        INFO("Group %d collective %d running %d threads\n", groupId, collId, numThreadsToUse);
      ErrCode errCode = TEST_SUCCESS;
      auto& errCodeVal = reinterpret_cast<int&>(errCode);
      #pragma omp parallel for num_threads(numThreadsToUse) reduction(max : errCodeVal)
      for (int localRank : localRanksToExecute)
      {
        if (this->verbose && this->useRankThreading)
          INFO("Group %d collective %d running rank %d on thread %d\n", groupId, collId, localRank, getThreadId());

        CHECK_HIP_RANK(errCode, hipSetDevice(this->deviceIds[localRank]));

        CollectiveArgs& collArg = this->collArgs[groupId][localRank][collId];

        if (this->printValues && !useHipGraph)
        {
          int const numInputElementsToPrint = (this->printValues < 0 ? collArg.numInputElements : this->printValues);
          PtrUnion inputCpu;
          size_t const numInputBytes = numInputElementsToPrint * DataTypeToBytes(collArg.dataType);
          inputCpu.AllocateCpuMem(numInputBytes);
          CHECK_HIP_RANK(errCode, hipMemcpy(inputCpu.ptr, collArg.inputGpu.ptr, numInputBytes, hipMemcpyDeviceToHost));
          printf("[ DEBUG    ] Rank %02d Group %d Coll %d %-10s: %s\n", collArg.globalRank, groupId, collId, "Input",
                 inputCpu.ToString(collArg.dataType, numInputElementsToPrint).c_str());
          inputCpu.FreeCpuMem();

          int const numOutputElementsToPrint = (this->printValues < 0 ? collArg.numOutputElements : this->printValues);
          size_t const numOutputBytes = numOutputElementsToPrint * DataTypeToBytes(collArg.dataType);
          CHECK_HIP_RANK(errCode, hipMemcpy(collArg.outputCpu.ptr, collArg.outputGpu.ptr, numOutputBytes, hipMemcpyDeviceToHost));
          printf("[ DEBUG    ] Rank %02d Group %d Coll %d %-10s: %s\n", collArg.globalRank, groupId, collId, "Pre-Output",
                 collArg.outputCpu.ToString(collArg.dataType, numOutputElementsToPrint).c_str());
        }

        switch (collArg.funcType)
        {
        case ncclCollBroadcast:
          CHILD_NCCL_CALL_RANK(errCode, ncclBroadcast(
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
          CHILD_NCCL_CALL_RANK(errCode, ncclReduce(
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
          CHILD_NCCL_CALL_RANK(errCode, ncclAllGather(
                                        collArg.inputGpu.ptr,
                                        collArg.outputGpu.ptr,
                                        collArg.numInputElements,
                                        collArg.dataType,
                                        this->comms[localRank],
                                        this->streams[groupId][localRank][collArg.streamIdx]),
                          "ncclAllGather");
          break;
        case ncclCollReduceScatter:
          CHILD_NCCL_CALL_RANK(errCode, ncclReduceScatter(
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
          CHILD_NCCL_CALL_RANK(errCode, ncclAllReduce(
                                        collArg.inputGpu.ptr,
                                        collArg.outputGpu.ptr,
                                        collArg.numInputElements,
                                        collArg.dataType,
                                        collArg.options.redOp,
                                        this->comms[localRank],
                                        this->streams[groupId][localRank][collArg.streamIdx]),
                          "ncclAllReduce");
          break;
        case ncclCollGather:
          CHILD_NCCL_CALL_RANK(errCode, ncclGather(
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
          CHILD_NCCL_CALL_RANK(errCode, ncclScatter(
                                      collArg.inputGpu.ptr,
                                      collArg.outputGpu.ptr,
                                      collArg.numOutputElements,
                                      collArg.dataType,
                                      collArg.options.root,
                                      this->comms[localRank],
                                      this->streams[groupId][localRank][collArg.streamIdx]),
                          "ncclScatter");
          break;
        case ncclCollAllToAll:
          CHILD_NCCL_CALL_RANK(errCode, ncclAllToAll(
                                       collArg.inputGpu.ptr,
                                       collArg.outputGpu.ptr,
                                       collArg.numInputElements / collArg.totalRanks,
                                       collArg.dataType,
                                       this->comms[localRank],
                                       this->streams[groupId][localRank][collArg.streamIdx]),
                          "ncclAllToAll");
          break;
        case ncclCollAllToAllv:
          CHILD_NCCL_CALL_RANK(errCode, ncclAllToAllv(
                                        collArg.inputGpu.ptr,
                                        collArg.options.sendcounts + (this->rankOffset + localRank)*this->totalRanks,
                                        collArg.options.sdispls + (this->rankOffset + localRank)*this->totalRanks,
                                        collArg.outputGpu.ptr,
                                        collArg.options.recvcounts + (this->rankOffset + localRank)*this->totalRanks,
                                        collArg.options.rdispls + (this->rankOffset + localRank)*this->totalRanks,
                                        collArg.dataType,
                                        this->comms[localRank],
                                        this->streams[groupId][localRank][collArg.streamIdx]),
                          "ncclAllToAllv");
          break;
        case ncclCollSend:
          CHILD_NCCL_CALL_RANK(errCode, ncclSend(
                                   collArg.inputGpu.ptr,
                                   collArg.numInputElements,
                                   collArg.dataType,
                                   collArg.options.root,
                                   this->comms[localRank],
                                   this->streams[groupId][localRank][collArg.streamIdx]),
                          "ncclSend");
          break;
        case ncclCollRecv:
          CHILD_NCCL_CALL_RANK(errCode, ncclRecv(
                                   collArg.outputGpu.ptr,
                                   collArg.numOutputElements,
                                   collArg.dataType,
                                   collArg.options.root,
                                   this->comms[localRank],
                                   this->streams[groupId][localRank][collArg.streamIdx]),
                          "ncclRecv");
          break;
        default:
          ERROR("Unknown func type %d\n", collArg.funcType);
          RANK_RESULT(errCode, TEST_FAIL);
        }
        if (this->useBlocking == false)
        {
          CHILD_NCCL_CALL_NON_BLOCKING_RANK(errCode, "ncclCommGetAsyncErrorExecuteCollectives", localRank);
        }

        if (this->verbose && this->useRankThreading)
          INFO("Group %d collective %d done rank %d on thread %d\n", groupId, collId, localRank, getThreadId());
      }

      if (this->useRankThreading) CHECK_CALL(errCode);
    }
    // End group call
    if (this->useBlocking == false)
    {
      // handle the ncclGroupEnd in case of non-blocking communication
      ncclResult_t Group_End_state = ncclGroupEnd();
      if (Group_End_state != ncclSuccess)
      {
        for (int localRank = 0; localRank < this->comms.size(); ++localRank)
        {
          CHECK_HIP(hipSetDevice(this->deviceIds[localRank]));
          CHILD_NCCL_CALL_NON_BLOCKING("ncclCommGetAsyncErrorGroupEnd", localRank);
        }
      }
    }
    else
    {
      // In case of blocking communication just call ncclGroupEnd
      CHILD_NCCL_CALL(ncclGroupEnd(), "ncclGroupEnd");
    }

    // Instantiate and launch HIP graph if requested
    if (useHipGraph)
    {
      for (int localRank : localRanksToExecute)
      {
        if (this->verbose) INFO("Ending stream capture for rank %d\n", localRank);
        CHECK_HIP(hipSetDevice(this->deviceIds[localRank]));
        for (int i = 0; i < this->numStreamsPerGroup[groupId]; i++)
        {
          CHECK_HIP(hipStreamEndCapture(this->streams[groupId][localRank][i], &this->graphs[groupId][localRank][i]));

          // if (this->verbose)
          // {
          //   size_t numNodes;
          //   hipGraphNode_t* nodes;
          //   CHECK_HIP(hipGraphGetNodes(graphs[localRank][i], nodes, &numNodes));
          //   INFO("Graph for rank %d stream %d has %lu nodes\n", localRank, i, numNodes);
          // }
        }

        if (this->verbose) INFO("Instantiating executable graph for group %d rank %d\n", groupId, localRank);
        for (int i = 0; i < this->numStreamsPerGroup[groupId]; i++)
        {
          CHECK_HIP(hipGraphInstantiate(&this->graphExecs[groupId][localRank][i], this->graphs[groupId][localRank][i], NULL, NULL, 0));
          graphEnabled[groupId][localRank][i] = true;
        }
      }
    }
    else
    {
      if (this->verbose)
        INFO("Child %d submits group call.  Waiting for completion\n", this->childId);
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
    if (this->verbose) INFO("Starting sychronization and timing\n");
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
    }

    // timed out
    if (!streamsToComplete.empty())
    {
      if (this->verbose) INFO("Collective timed out, aborting\n");
      for (int localRank : localRanksToExecute)
      {
        ncclCommAbort(this->comms[localRank]); 
        timedout = 1;
      }
    }

    // extra sync to flush GPU cache for validation later
    // TODO: remove this after figuring out & fixing the exact behavior 
    // of fencing between kernels and at hipStreamQuery
    for (int localRank : localRanksToExecute)
    {
      if (this->verbose) INFO("Starting synchronization for group %d rank %d\n", groupId, localRank);
      for (int i = 0; i < this->numStreamsPerGroup[groupId]; i++)
        CHECK_HIP(hipStreamSynchronize(this->streams[groupId][localRank][i]));
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

          printf("[ DEBUG    ] Rank %02d Group %d Coll %d %-10s: %s\n", collArg.globalRank, groupId, collId, "Expected",
                 collArg.expected.ToString(collArg.dataType, numOutputElementsToPrint).c_str());
        }
    }

    if (timedout)
    {
      ERROR("Child %d timed out and exceeded limit %d us in ExecuteCollectives()\n", this->childId, timeoutUs);
      return TEST_TIMEOUT;
    }

    if (this->verbose) INFO("Child %d finishes ExecuteCollectives()\n", this->childId);
    return TEST_SUCCESS;
  }

  ErrCode TestBedChild::ValidateResults()
  {
    // Read values sent by parent [see TestBed::ValidateResults()]
    int globalRank, groupId, collId;
    PIPE_READ(globalRank);
    PIPE_READ(groupId);
    PIPE_READ(collId);

    if (this->verbose) INFO("Child %d begins ValidateResults()\n", this->childId);

    if (globalRank < this->rankOffset || (this->rankOffset + comms.size() <= globalRank))
    {
      ERROR("Child %d does not contain rank %d\n", this->childId, globalRank);
      return TEST_FAIL;
    }
    int const localRank = globalRank - rankOffset;
    CHECK_HIP(hipSetDevice(this->deviceIds[localRank]));

    ErrCode status = TEST_SUCCESS;
    for (int collIdx = 0; collIdx < collArgs[groupId][localRank].size(); ++collIdx)
    {
      if (collId == -1 || collId == collIdx)
      {
        if (this->verbose) INFO("Rank %d on child %d validating collective %d in group %d results\n",
                                globalRank, this->childId, collIdx, groupId);
        if (this->collArgs[groupId][localRank][collIdx].ValidateResults() != TEST_SUCCESS)
        {
          ERROR("Rank %d Group %d Collective %d output does not match expected\n", globalRank, groupId, collIdx);
          status = TEST_FAIL;
        }
      }
    }
    if (this->verbose) INFO("Child %d finishes ValidateResults() with status %s\n", this->childId,
                            status == TEST_SUCCESS ? "SUCCESS" : "FAIL");
    return status;
  }

  ErrCode TestBedChild::LaunchGraphs()
  {
    int groupId;
    PIPE_READ(groupId);

    if (this->verbose) INFO("Child %d begins LaunchGraphs for group %d\n", this->childId, groupId);

    for (int localRank = 0; localRank < this->deviceIds.size(); ++localRank) {
      CHECK_HIP(hipSetDevice(this->deviceIds[localRank]));

      for (int streamIdx = 0; streamIdx < this->numStreamsPerGroup[groupId]; ++streamIdx)
      {
        if (this->verbose) INFO("Launch graph for group %d rank %d stream %d\n", groupId, localRank, streamIdx);
        CHECK_HIP(hipGraphLaunch(this->graphExecs[groupId][localRank][streamIdx], this->streams[groupId][localRank][streamIdx]));
      }
    }

    if (this->verbose) INFO("Child %d finishes LaunchGraphs for group %d\n", this->childId, groupId);
    return TEST_SUCCESS;
  }

  ErrCode TestBedChild::DeallocateMem()
  {
    if (this->verbose) INFO("Child %d begins DeallocateMem\n", this->childId);

    // Read values sent by parent [see TestBed::DeallocateMem()]
    int globalRank, groupId, collId;
    PIPE_READ(globalRank);
    PIPE_READ(groupId);
    PIPE_READ(collId);

    if (globalRank < this->rankOffset || (this->rankOffset + comms.size() <= globalRank))
    {
      ERROR("Child %d does not contain rank %d\n", this->childId, globalRank);
      return TEST_FAIL;
    }
    int const localRank = globalRank - rankOffset;
    CHECK_HIP(hipSetDevice(this->deviceIds[localRank]));

    for (int collIdx = 0; collIdx < collArgs[groupId][localRank].size(); ++collIdx)
    {
      CollectiveArgs& collArg = this->collArgs[groupId][localRank][collIdx];
      if (collId == -1 || collId == collIdx)
      {
        if (this->verbose)
        {
          INFO("Child %d release memory for collective %d in group %d (Input: %p Output %p\n",
               this->childId, collIdx, groupId, collArg.inputGpu.ptr, collArg.outputGpu.ptr);
        }
        if (collArg.userRegistered && (collArg.funcType == ncclCollSend || collArg.funcType == ncclCollRecv))
        {
          CHILD_NCCL_CALL(ncclCommDeregister(this->comms[localRank], collArg.commRegHandle), "ncclCommDeregister");
        }

        CHECK_CALL(collArg.DeallocateMem());
      }
      if (collArg.options.scalarMode != -1)
      {
        CHILD_NCCL_CALL(ncclRedOpDestroy(collArg.options.redOp, this->comms[localRank]),
                        "ncclRedOpDestroy");
        if (verbose) INFO("Child %d destroys custom redop %d for collective %d in group %d\n",
                          this->childId, collArg.options.redOp, collIdx, groupId);
      }
    }
    if (this->verbose) INFO("Child %d finishes DeallocateMem\n", this->childId);
    return TEST_SUCCESS;
  }

  ErrCode TestBedChild::DestroyComms()
  {
    if (this->verbose) INFO("Child %d begins DestroyComms\n", this->childId);

    // Release comms
    for (int i = 0; i < this->comms.size(); ++i)
    {
      // Check if the communicator is non-blocking
      if (this->useBlocking == false)
      {
        // handle the non-blocking case
        ncclCommFinalize(this->comms[i]);
        CHILD_NCCL_CALL_NON_BLOCKING("ncclCommGetAsyncErrorCommFinalize", i);
      }
      else
      {
        // In case of blocking just call Finalize
        CHILD_NCCL_CALL(ncclCommFinalize(this->comms[i]), "ncclCommFinalize");
      }
    }

    for (int i = 0; i < this->comms.size(); ++i)
    {
      CHILD_NCCL_CALL(ncclCommDestroy(this->comms[i]), "ncclCommDestroy");
    }
    for (int i = 0; i < this->numGroupCalls; ++i)
    {
      for (int j = 0; j < this->streams[i].size(); ++j)
      {
        for (int k = 0; k < this->streams[i][j].size(); ++k)
        {
          CHECK_HIP(hipStreamDestroy(this->streams[i][j][k]));
        }
      }
    }
    this->comms.clear();
    this->streams.clear();
    if (this->verbose) INFO("Child %d finishes DestroyComms\n", this->childId);
    return TEST_SUCCESS;
  }

  ErrCode TestBedChild::DestroyGraphs()
  {
    if (this->verbose) INFO("Child %d begins DestroyGraphs\n", this->childId);

    int groupId;
    PIPE_READ(groupId);

    // Release graphs
    for (int localRank = 0; localRank < this->deviceIds.size(); ++localRank) 
    {
      CHECK_HIP(hipSetDevice(this->deviceIds[localRank]));
      for (int streamIdx = 0; streamIdx < this->numStreamsPerGroup[groupId]; ++streamIdx)
      {
        if (graphEnabled[groupId][localRank][streamIdx])
        {
          if (this->verbose) INFO("Destroying graphs for group %d rank %d stream %d\n", groupId, localRank, streamIdx);

          CHECK_HIP(hipGraphDestroy(this->graphs[groupId][localRank][streamIdx]));
          CHECK_HIP(hipGraphExecDestroy(this->graphExecs[groupId][localRank][streamIdx]));
        }
      }
    }

    for (int localRank = 0; localRank < this->deviceIds.size(); ++localRank)
    {
      for (int i = 0; i < this->numStreamsPerGroup[groupId]; ++i)
        CHECK_HIP(hipStreamSynchronize(this->streams[groupId][localRank][i]));
    }    

    this->graphs[groupId].clear();
    this->graphExecs[groupId].clear();
    this->graphEnabled[groupId].clear();

    if (this->verbose) INFO("Child %d finishes DestroyGraphs\n", this->childId);
    return TEST_SUCCESS;
  }
}
