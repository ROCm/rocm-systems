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

// #define PIPE_READ(val) \
//   if (read(childReadFd, &val, sizeof(val)) != sizeof(val)) return TEST_FAIL;

#undef PIPE_READ // Just in case it's defined elsewhere
#define PIPE_READ(val) \
    if (safe_pipe_read(childReadFd, &val, sizeof(val)) != sizeof(val)) return TEST_FAIL;

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
    if (verbose && useRankThreading) TEST_WARN("Multi-threaded ranks requires ENABLE_OPENMP to be defined");
    #endif
    int command;
    while (true)
    {
      if (safe_pipe_read(childReadFd, &command, sizeof(command)) <= 0) {
        break;
      }
      ErrCode status = TEST_SUCCESS;
      if (command < 0 || command >= NUM_CHILD_COMMANDS) {
        TEST_ERROR("Child %d received invalid command ID: %d", this->childId, command);
        status = TEST_FAIL;
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
        goto stop;
      }

      // Send back acknowledgement to parent
      if (status == TEST_FAIL)
        TEST_ERROR("Child %d failed on command [%s]:", this->childId, ChildCommandNames[command]);
      if (safe_pipe_write(childWriteFd, &status, sizeof(status)) < 0) {
        TEST_ERROR("Child %d write to parent failed: %s", this->childId, strerror(errno));
        break;
      }
      if (retValBuf.size() > 0 && safe_pipe_write(childWriteFd, retValBuf.data(), retValBuf.size()) < 0) {
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
    this->numCollectivesInGroup.resize(numCollSize);
    if (numCollSize > 0)
    {
      read(this->childReadFd, this->numCollectivesInGroup.data(),
      numCollSize * sizeof(int));
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
    this->numStreamsPerGroup.resize(numStreamsSize);
    if (numStreamsSize > 0)
    {
      read(this->childReadFd, this->numStreamsPerGroup.data(),
      numStreamsSize * sizeof(int));
    }

    // Read GPUs and prepare storage
    int numGpus;
    PIPE_READ(numGpus);

    // Destroy existing HIP streams before clearing vector to prevent hardware queue leak!
    for (auto& groupStreams : this->streams) 
    {
      for (auto& rankStreams : groupStreams) 
      {
        for (hipStream_t& stream : rankStreams) 
        {
          if (stream != nullptr) 
          {
            hipStreamDestroy(stream);
            stream = nullptr;
          }
        }
      }
    }

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
        this->collArgs[i][j].clear();
        this->collArgs[i][j].resize(numCollectivesInGroup[i]);
        this->streams[i][j].resize(numStreamsPerGroup[i], nullptr);
      }
    }

    for (int i = 0; i < numGpus; i++)
    {
      PIPE_READ(this->deviceIds[i]);
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
        ncclCommInitRankConfig(&this->comms[localRank], this->totalRanks, id, globalRank, &config);
        CHILD_NCCL_CALL_NON_BLOCKING("ncclCommGetAsyncErrorInitRankConfig", localRank);
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
    ncclResult_t groupEndErr = ncclGroupEnd();
    if (groupEndErr != ncclSuccess)
    {
      TEST_ERROR("Child %d ncclGroupEnd failed with error %d", this->childId, groupEndErr);
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
    CollFuncPtr prepDataFunc;

    PIPE_READ(globalRank);
    PIPE_READ(groupId);
    PIPE_READ(collId);
    PIPE_READ(prepDataFunc);

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
        if (this->verbose) TEST_INFO("Rank %d on child %d prepares data for collective %d in group %d",
                                globalRank, this->childId, collIdx, groupId);
        CHECK_CALL(this->collArgs[groupId][localRank][collIdx].PrepareData(prepDataFunc));
      }
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

    // Start HIP graph stream capture if requested
    if (useHipGraph)
    {
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

    // int numThreadsToUse = this->useRankThreading ? numRanksToExecute : 1;
    int numThreadsToUse = (this->useRankThreading && useHipGraph) ? numRanksToExecute : 1;

    // Start group call
    CHILD_NCCL_CALL(ncclGroupStart(), "ncclGroupStart ExecuteCollectives");

    // Loop over all collectives to be executed in group call
    for (int collId = 0; collId < this->numCollectivesInGroup[groupId]; ++collId)
    {
      // Loop over all local ranks
      if (this->verbose && this->useRankThreading)
        TEST_INFO("Group %d collective %d running %d threads", groupId, collId, numThreadsToUse);
      ErrCode errCode = TEST_SUCCESS;
      auto& errCodeVal = reinterpret_cast<int&>(errCode);
      // #pragma omp parallel for num_threads(numThreadsToUse) reduction(max : errCodeVal)
      for (int localRank : localRanksToExecute)
      {
        if (this->verbose && this->useRankThreading)
          TEST_INFO("Group %d collective %d running rank %d on thread %d", groupId, collId, localRank, getThreadId());

        CHECK_HIP_RANK(errCode, hipSetDevice(this->deviceIds[localRank]));

        CollectiveArgs& collArg = this->collArgs[groupId][localRank][collId];
        
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
          // Use ncclAllReduceWithBias if bias is enabled
          if (collArg.options.useBias)
          {
            CHILD_NCCL_CALL_RANK(errCode, ncclAllReduceWithBias(
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
            CHILD_NCCL_CALL_RANK(errCode, ncclAllReduce(
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
        case ncclCollAlltoAll:
          CHILD_NCCL_CALL_RANK(errCode, ncclAlltoAll(
                                       collArg.inputGpu.ptr,
                                       collArg.outputGpu.ptr,
                                       collArg.numInputElements / collArg.totalRanks,
                                       collArg.dataType,
                                       this->comms[localRank],
                                       this->streams[groupId][localRank][collArg.streamIdx]),
                          "ncclAlltoAll");
          break;
        case ncclCollAlltoAllv:
          CHILD_NCCL_CALL_RANK(errCode, ncclAlltoAllv(
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
          TEST_ERROR("Unknown func type %d", collArg.funcType);
          RANK_RESULT(errCode, TEST_FAIL);
        }
        if (this->useBlocking == false)
        {
          CHILD_NCCL_CALL_NON_BLOCKING_RANK(errCode, "ncclCommGetAsyncErrorExecuteCollectives", localRank);
        }

        if (this->verbose && this->useRankThreading)
          TEST_INFO("Group %d collective %d done rank %d on thread %d", groupId, collId, localRank, getThreadId());
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
      CHILD_NCCL_CALL(ncclGroupEnd(), "ncclGroupEnd ExecuteCollectives");
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
    if (groupId < 0 || groupId >= static_cast<int>(this->collArgs.size()))
    {
      TEST_ERROR("Child %d Rank %d: Invalid groupId %d (collArgs size: %zu)",
               this->childId, globalRank, groupId, this->collArgs.size());
      return TEST_FAIL;
    }
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
    if (groupId < 0 || groupId >= static_cast<int>(this->graphExecs.size()))
    {
      TEST_ERROR("Child %d: Invalid groupId %d for LaunchGraphs (graphExecs size: %zu)",
               this->childId, groupId, this->graphExecs.size());
               return TEST_FAIL;
    }

    for (size_t localRank = 0; localRank < this->deviceIds.size(); ++localRank) {
      CHECK_HIP(hipSetDevice(this->deviceIds[localRank]));

      for (int streamIdx = 0; streamIdx < this->numStreamsPerGroup[groupId]; ++streamIdx)
      {
        if (this->graphEnabled[groupId][localRank][streamIdx]){
          if (this->verbose) TEST_INFO("Launch graph for group %d rank %d stream %d", groupId, localRank, streamIdx);
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

    ErrCode errCode = TEST_SUCCESS;

    // Enclose RCCL deregistration calls in a group for safety
    CHILD_NCCL_CALL(ncclGroupStart(), "ncclGroupStart DeregisterMem");

    for (size_t collIdx = 0; collIdx < collArgs[groupId][localRank].size(); ++collIdx)
    {
      CollectiveArgs& collArg = this->collArgs[groupId][localRank][collIdx];
      if (collId == -1 || collId == static_cast<int>(collIdx))
      {
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
          if (collArg.inputWin == collArg.outputWin)
          {
            // In-place mode: Both pointers share the same window handle
            if (collArg.inputWin != nullptr)
            {
              CHILD_NCCL_CALL_RANK(errCode,
                ncclCommWindowDeregister(this->comms[localRank], collArg.inputWin),
                "ncclCommWindowDeregister (in-place)");
              collArg.inputWin = nullptr;
              collArg.outputWin = nullptr; // Explicitly clear outputWin to prevent dangling handle
            }
          }
          else
          {
            // Out-of-place mode: Distinct window handles
            if (collArg.inputWin != nullptr)
            {
              CHILD_NCCL_CALL_RANK(errCode,
                ncclCommWindowDeregister(this->comms[localRank], collArg.inputWin),
                "ncclCommWindowDeregister (input)");
              collArg.inputWin = nullptr;
            }

            if (collArg.outputWin != nullptr)
            {
              CHILD_NCCL_CALL_RANK(errCode,
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
          CHILD_NCCL_CALL_RANK(errCode,
            ncclCommDeregister(this->comms[localRank], collArg.inputRegHandle),
            "ncclCommDeregister");
          collArg.inputRegHandle = nullptr;
        }
         if (collArg.outputRegHandle != nullptr)
        {
          CHILD_NCCL_CALL_RANK(errCode,
            ncclCommDeregister(this->comms[localRank], collArg.outputRegHandle),
            "ncclCommDeregister");
          collArg.outputRegHandle = nullptr;
        }

        if (collArg.biasRegHandle != nullptr)
        {
          CHILD_NCCL_CALL_RANK(errCode,
            ncclCommDeregister(this->comms[localRank], collArg.biasRegHandle),
            "ncclCommDeregister (bias)");
          collArg.biasRegHandle = nullptr;
        }
      }
    }
    CHILD_NCCL_CALL(ncclGroupEnd(), "ncclGroupEnd DeregisterMem");
    if (this->verbose) TEST_INFO("Child %d finishes DeregisterMemInternal", this->childId);
    return errCode;
  }

  ErrCode TestBedChild::DeallocateMemInternal_impl(int groupId, int collId, int localRank)
  {
    if (this->verbose) TEST_INFO("Child %d begins DeallocateMemInternal", this->childId);
    ErrCode errCode = TEST_SUCCESS;
    for (size_t collIdx = 0; collIdx < collArgs[groupId][localRank].size(); ++collIdx)
    {
      CollectiveArgs& collArg = this->collArgs[groupId][localRank][collIdx];
      if (collId == -1 || collId == static_cast<int>(collIdx))
      {
        CHECK_CALL(collArg.DeallocateMem());

        if (collArg.options.scalarMode >= 0)
        {
          CHILD_NCCL_CALL_RANK(errCode,
            ncclRedOpDestroy(collArg.options.redOp, this->comms[localRank]),
            "ncclRedOpDestroy");
          if (this->verbose)
          {
            TEST_INFO("Child %d destroys custom redop %d for collective %zu in group %d",
                    this->childId, collArg.options.redOp, collIdx, groupId);
          }
        }
      }
    }
    if (this->verbose) TEST_INFO("Child %d finishes DeallocateMemInternal", this->childId);
    return errCode;
  }

  ErrCode TestBedChild::DeallocateMem()
  {
    if (this->verbose) TEST_INFO("Child %d begins DeallocateMem", this->childId);
    ErrCode errCode = TEST_SUCCESS;
    // Read values sent by parent [matches IPC pipe format]
    int globalRank, groupId, collId;
    PIPE_READ(globalRank);
    PIPE_READ(groupId);
    PIPE_READ(collId);

    if (globalRank < this->rankOffset || (this->rankOffset + static_cast<int>(comms.size()) <= globalRank))
    {
      TEST_ERROR("Child %d does not contain rank %d", this->childId, globalRank);
      return TEST_FAIL;
    }

    int const localRank = globalRank - rankOffset;
    CHECK_CALL(this->DeregisterMemInternal_impl(groupId,collId,localRank));
    CHECK_CALL(this->DeallocateMemInternal_impl(groupId,collId,localRank));
    return errCode;
  }

  ErrCode TestBedChild::DestroyComms()
  {
    if (this->verbose) TEST_INFO("Child %d begins DestroyComms", this->childId);
    
    // 1. Release NCCL communicators
    for (int i = 0; i < this->comms.size(); ++i)
    {
      if (this->comms[i] == nullptr) continue;
      if (this->useBlocking == false)
      {
        ncclCommFinalize(this->comms[i]);
        CHILD_NCCL_CALL_NON_BLOCKING("ncclCommGetAsyncErrorCommFinalize", i);
      }
      else
      {
        CHILD_NCCL_CALL(ncclCommFinalize(this->comms[i]), "ncclCommFinalize");
      }
    }
    
    for (int i = 0; i < this->comms.size(); ++i)
    {
      if (this->comms[i] != nullptr)
      {
        CHILD_NCCL_CALL(ncclCommDestroy(this->comms[i]), "ncclCommDestroy");
        this->comms[i] = nullptr;
      }
    }

    // 2. Safely release HIP streams with correct device context
    for (int i = 0; i < this->numGroupCalls; ++i)
    {
      for (int j = 0; j < this->streams[i].size(); ++j)
      {
        // Switch active GPU context to the device that owns this stream group
        CHECK_HIP(hipSetDevice(this->deviceIds[j]));

        for (int k = 0; k < this->streams[i][j].size(); ++k)
        {
          // Avoid destroying null handles or the default stream (0)
          if (this->streams[i][j][k] != nullptr)
          {
            CHECK_HIP(hipStreamDestroy(this->streams[i][j][k]));
            this->streams[i][j][k] = nullptr; // Avoid double-destruction
          }
        }
      }
    }

    this->comms.clear();
    this->streams.clear();
    if (this->verbose) TEST_INFO("Child %d finishes DestroyComms", this->childId);
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

    // Release graphs
    for (size_t localRank = 0; localRank < this->deviceIds.size(); ++localRank)
    {
      CHECK_HIP(hipSetDevice(this->deviceIds[localRank]));
      for (size_t streamIdx = 0; streamIdx < this->numStreamsPerGroup[groupId]; ++streamIdx)
      {
        if (streamIdx < this->graphEnabled[groupId][localRank].size() && this->graphEnabled[groupId][localRank][streamIdx])
        {
          if (this->verbose) TEST_INFO("Destroying graphs for group %d rank %d stream %d", groupId, localRank, streamIdx);
          if (this->graphExecs[groupId][localRank][streamIdx])
          {
            CHECK_HIP(hipGraphExecDestroy(this->graphExecs[groupId][localRank][streamIdx]));
            this->graphExecs[groupId][localRank][streamIdx] = nullptr;
          }
          if (this->graphs[groupId][localRank][streamIdx])
          {
            CHECK_HIP(hipGraphDestroy(this->graphs[groupId][localRank][streamIdx]));
            this->graphs[groupId][localRank][streamIdx] = nullptr;
          }
        }
      }
    }

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

    PIPE_READ(groupId);
    PIPE_READ(collId);

    ErrCode errCode = TEST_SUCCESS;

    // Grouped window registration across ALL local ranks managed by this child process
    CHILD_NCCL_CALL(ncclGroupStart(), "ncclGroupStart RegisterMem");

    for (size_t localRank = 0; localRank < this->comms.size(); ++localRank)
    {
      CHECK_HIP(hipSetDevice(this->deviceIds[localRank]));

      for (size_t collIdx = 0; collIdx < collArgs[groupId][localRank].size(); ++collIdx)
      {
        if (collId == -1 || collId == static_cast<int>(collIdx))
        {
          CollectiveArgs& collArg = this->collArgs[groupId][localRank][collIdx];

          // CASE 1: Symmetric Window Path (ncclCommWindowRegister)
          if (this->memAllocType == MEM_ALLOC_SYMMETRIC_WIN)
          {
            if (collArg.inPlace)
            {
              // For in-place collectives, register the FULL allocation first
              RcclUnitTesting::ncclFunc_t fn = collArg.funcType;
              void* buff = (fn == ncclCollScatter || fn ==ncclCollReduceScatter) ? collArg.inputGpu.ptr : collArg.outputGpu.ptr;
              size_t bufSize = (fn == ncclCollScatter || fn ==ncclCollReduceScatter) ? collArg.numInputBytesAllocated : collArg.numOutputBytesAllocated;

              if (this->verbose) TEST_INFO("Child %d calls ncclCommWindowRegister(this->comms[localRank = %d]) buff= %lu bufSize = %d &(collArg.outputWin) NCCL_WIN_COLL_SYMMETRIC", this->childId,localRank,buff,bufSize);
              CHILD_NCCL_CALL_RANK(errCode,
              ncclCommWindowRegister(this->comms[localRank],
                                     buff,
                                     bufSize,
                                     &(collArg.outputWin),
                                     NCCL_WIN_COLL_SYMMETRIC),
              "ncclCommWindowRegister (output in-place)");
              // Just to indicate windows are same, this assignment doesn't affect functionality.
              collArg.inputWin = collArg.outputWin;
           }
           else
           {
             // Out-of-place: Register input and output buffers independently
              if (collArg.inputGpu.ptr && collArg.numInputBytesAllocated > 0)
              {
                CHILD_NCCL_CALL_RANK(errCode,
                ncclCommWindowRegister(this->comms[localRank],
                                       collArg.inputGpu.ptr,
                                       collArg.numInputBytesAllocated,
                                       &(collArg.inputWin),
                                       NCCL_WIN_COLL_SYMMETRIC),
                "ncclCommWindowRegister (input)");
              }

              if (collArg.outputGpu.ptr && collArg.numOutputBytesAllocated > 0)
              {
                CHILD_NCCL_CALL_RANK(errCode,
                ncclCommWindowRegister(this->comms[localRank],
                                       collArg.outputGpu.ptr,
                                       collArg.numOutputBytesAllocated,
                                       &(collArg.outputWin),
                                       NCCL_WIN_COLL_SYMMETRIC),
                "ncclCommWindowRegister (output)");
              }
            }
          }
          // CASE 2: Legacy / Buffer Registration Path (ncclCommRegister)
          else if (collArg.userRegistered)
          {
            if (collArg.inPlace)
            {
              if (collArg.outputGpu.ptr && collArg.numOutputBytesAllocated > 0)
              {
                CHILD_NCCL_CALL_RANK(errCode,
                ncclCommRegister(this->comms[localRank],
                                 collArg.outputGpu.ptr,
                                 collArg.numOutputBytesAllocated,
                                 &(collArg.outputRegHandle)),
                "ncclCommRegister (output in-place)");
              }
           } 
           else
           {
              // Register BOTH input AND output buffers for out-of-place operations
              if (collArg.inputGpu.ptr && collArg.numInputBytesAllocated > 0)
              {
                CHILD_NCCL_CALL_RANK(errCode,
                ncclCommRegister(this->comms[localRank],
                                 collArg.inputGpu.ptr,
                                 collArg.numInputBytesAllocated,
                                 &(collArg.inputRegHandle)),
                "ncclCommRegister (input)");
              }

              if (collArg.outputGpu.ptr && collArg.numOutputBytesAllocated > 0)
              {
                CHILD_NCCL_CALL_RANK(errCode,
                ncclCommRegister(this->comms[localRank],
                                 collArg.outputGpu.ptr,
                                 collArg.numOutputBytesAllocated,
                                 &(collArg.outputRegHandle)),
                "ncclCommRegister (output)");
              }
            }
          }
        }
      }
    }

    // Completes handle exchange for all local ranks simultaneously
    CHILD_NCCL_CALL(ncclGroupEnd(), "ncclGroupEnd RegisterMem");
    // Ensure GPU memory mapping / TLB invalidations complete across all local ranks
    for (size_t localRank = 0; localRank < this->comms.size(); ++localRank)
     {
      CHECK_HIP(hipSetDevice(this->deviceIds[localRank]));
      CHECK_HIP(hipDeviceSynchronize());
    }

    if (this->verbose) TEST_INFO("Child %d finishes RegisterMem()", this->childId);
    return TEST_SUCCESS;
  }
}
