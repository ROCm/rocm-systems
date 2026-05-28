/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "CollectiveArgs.hpp"
#include "PrepDataFuncs.hpp"
#include <cstdio>
#include <hip/hip_runtime.h>

namespace RcclUnitTesting
{
  ErrCode DefaultPrepareDataFunc(CollectiveArgs &collArgs)
  {
    switch (collArgs.funcType)
    {
    case ncclCollBroadcast:     return DefaultPrepData_Broadcast(collArgs);
    case ncclCollReduce:        return DefaultPrepData_Reduce(collArgs, false);
    case ncclCollAllGather:     return DefaultPrepData_Gather(collArgs, true);
    case ncclCollReduceScatter: return DefaultPrepData_ReduceScatter(collArgs);
    case ncclCollAllReduce:     return DefaultPrepData_Reduce(collArgs, true);
    case ncclCollGather:        return DefaultPrepData_Gather(collArgs, false);
    case ncclCollScatter:       return DefaultPrepData_Scatter(collArgs);
    case ncclCollAllToAll:      return DefaultPrepData_AllToAll(collArgs);
    case ncclCollAllToAllv:     return DefaultPrepData_AllToAllv(collArgs);
    case ncclCollSend:          return DefaultPrepData_Send(collArgs);
    case ncclCollRecv:          return DefaultPrepData_Recv(collArgs);
    default:
      ERROR("Unknown func type %d\n", collArgs.funcType);
      return TEST_FAIL;
    }
  }

  ErrCode CheckAllocation(CollectiveArgs const& collArgs)
  {
    if (collArgs.numInputElements > collArgs.numInputElementsAllocated)
    {
      ERROR("Number of input elements (%lu) exceeds the number of allocated input elements (%lu)\n",
            collArgs.numInputElements, collArgs.numInputElementsAllocated);
      return TEST_FAIL;
    }

    if (collArgs.numOutputElements > collArgs.numOutputElementsAllocated)
    {
      ERROR("Number of output elements (%lu) exceeds the number of allocated output elements (%lu)\n",
            collArgs.numOutputElements, collArgs.numOutputElementsAllocated);
      return TEST_FAIL;
    }
    return TEST_SUCCESS;
  }

  ErrCode DefaultPrepData_Broadcast(CollectiveArgs &collArgs)
  {
    CHECK_CALL(CheckAllocation(collArgs));
    if (collArgs.numInputElements != collArgs.numOutputElements)
    {
      ERROR("Number of input elements must match number of output elements for Broadcast\n");
      return TEST_FAIL;
    }

    size_t const numBytes = collArgs.numInputElements * DataTypeToBytes(collArgs.dataType);

    // Clear output for all ranks (done before filling input in case of in-place)
    CHECK_CALL(collArgs.outputGpu.ClearGpuMem(numBytes));

    // Only root needs input pattern
    if (collArgs.globalRank == collArgs.options.root)
      CHECK_CALL(collArgs.inputGpu.FillPattern(collArgs.dataType,
                                               collArgs.numInputElements,
                                               collArgs.options.root, true));

    // Otherwise all other ranks expected output is the same as input of root
    return collArgs.expected.FillPattern(collArgs.dataType,
                                         collArgs.numInputElements,
                                         collArgs.options.root,
                                         false);
  }

  ErrCode DefaultPrepData_Reduce(CollectiveArgs &collArgs, bool const isAllReduce)
  {
    CHECK_CALL(CheckAllocation(collArgs));
    if (collArgs.numInputElements != collArgs.numOutputElements)
    {
      ERROR("Number of input elements must match number of output elements for Reduce\n");
      return TEST_FAIL;
    }

    size_t const numBytes = collArgs.numInputElements * DataTypeToBytes(collArgs.dataType);

    // Clear output for all ranks (done before filling input in case of in-place)
    CHECK_CALL(collArgs.outputGpu.ClearGpuMem(numBytes));

    // Clear expected buffer for holding reduction
    PtrUnion result;
    CHECK_CALL(result.Attach(collArgs.expected));
    CHECK_CALL(result.ClearCpuMem(numBytes));

    // If average or custom reduction operator is used, perform a summation instead
    ncclRedOp_t const tempOp = (collArgs.options.redOp >= ncclAvg ? ncclSum : collArgs.options.redOp);

    // Loop over each rank and generate their input into a temp buffer, then reduce
    PtrUnion scalarsPerRank;
    scalarsPerRank.Attach(collArgs.options.scalarTransport.ptr);

    PtrUnion tempInputCpu;
    CHECK_CALL(tempInputCpu.Attach(collArgs.outputCpu));
    for (int rank = 0; rank < collArgs.totalRanks; ++rank)
    {
      // Generate temporary input for this rank
      CHECK_CALL(tempInputCpu.FillPattern(collArgs.dataType, collArgs.numInputElements, rank, false));

      // Copy the pre-scaled input into GPU memory for the correct rank
      if (rank == collArgs.globalRank)
      {
        CHECK_HIP(hipMemcpy(collArgs.inputGpu.ptr, tempInputCpu.ptr, numBytes, hipMemcpyHostToDevice));
      }

      // Scale the temporary input by local scalar for this rank
      // (Used by custom reduction ops)
      if (collArgs.options.scalarMode >= 0)
      {
        CHECK_CALL(tempInputCpu.Scale(collArgs.dataType, collArgs.numInputElements,
                                      scalarsPerRank, rank));
      }

      // Any rank that requires output reduces the scaled-inputs
      if (isAllReduce || collArgs.options.root == collArgs.globalRank)
      {
        if (rank == 0)
        {
          memcpy(result.ptr, tempInputCpu.ptr, numBytes);
        }
        else
        {
          CHECK_CALL(result.Reduce(collArgs.dataType, collArgs.numInputElements,
                                   tempInputCpu, tempOp));
        }
      }
    }

    // Perform averaging if necessary
    if (collArgs.options.redOp == ncclAvg && (isAllReduce || collArgs.options.root == collArgs.globalRank))
    {
      CHECK_CALL(result.DivideByInt(collArgs.dataType, collArgs.numInputElements, collArgs.totalRanks));
    }
    return TEST_SUCCESS;
  }

  ErrCode DefaultPrepData_Gather(CollectiveArgs &collArgs, bool const isAllGather)
  {
    CHECK_CALL(CheckAllocation(collArgs));
    if (collArgs.totalRanks * collArgs.numInputElements != collArgs.numOutputElements)
    {
      ERROR("# of output elements must be total ranks * # input elements for AllGather\n");
      return TEST_FAIL;
    }

    // Clear output for all ranks (done before filling input in case of in-place)
    size_t const numInputBytes = collArgs.numInputElements * DataTypeToBytes(collArgs.dataType);
    size_t const numOutputBytes = collArgs.numOutputElements * DataTypeToBytes(collArgs.dataType);
    CHECK_CALL(collArgs.inputGpu.ClearGpuMem(numInputBytes));
    CHECK_CALL(collArgs.outputGpu.ClearGpuMem(numOutputBytes));

    PtrUnion result;
    CHECK_CALL(result.Attach(collArgs.expected.ptr));
    CHECK_CALL(result.ClearCpuMem(numOutputBytes));

    // Use outputCpu buffer to store temporary input
    PtrUnion tempInputCpu;
    CHECK_CALL(tempInputCpu.Attach(collArgs.outputCpu.ptr));

    for (int rank = 0; rank < collArgs.totalRanks; ++rank)
    {
      CHECK_CALL(tempInputCpu.FillPattern(collArgs.dataType, collArgs.numInputElements, rank, false));
      if (rank == collArgs.globalRank)
      {
        CHECK_HIP(hipMemcpy(collArgs.inputGpu.ptr, tempInputCpu.ptr, numInputBytes, hipMemcpyHostToDevice));
      }
      if (isAllGather || collArgs.options.root == collArgs.globalRank)
      {
        memcpy(result.I1 + (rank * numInputBytes), tempInputCpu.ptr, numInputBytes);
      }
    }
    return TEST_SUCCESS;
  }

  ErrCode DefaultPrepData_ReduceScatter(CollectiveArgs &collArgs)
  {
    CHECK_CALL(CheckAllocation(collArgs));
    if (collArgs.numInputElements != collArgs.numOutputElements * collArgs.totalRanks)
    {
      ERROR("# of input elements must be total ranks * # output elements for ReduceScatter\n");
      return TEST_FAIL;
    }

    size_t const numInputBytes  = collArgs.numInputElements * DataTypeToBytes(collArgs.dataType);
    size_t const numOutputBytes = collArgs.numOutputElements * DataTypeToBytes(collArgs.dataType);

    // Clear output for all ranks (done before filling input in case of in-place)
    CHECK_CALL(collArgs.outputGpu.ClearGpuMem(numOutputBytes));

    PtrUnion tempInputCpu;
    PtrUnion tempResultCpu;

    CHECK_CALL(tempInputCpu.AllocateCpuMem(numInputBytes));
    CHECK_CALL(tempResultCpu.AllocateCpuMem(numInputBytes));
    CHECK_CALL(tempResultCpu.ClearCpuMem(numInputBytes));

    // If average or custom reduction operator is used, perform a summation instead
    ncclRedOp_t const tempOp = (collArgs.options.redOp >= ncclAvg ? ncclSum : collArgs.options.redOp);

    // Loop over each rank and generate the input / scale / reduce
    PtrUnion scalarsPerRank;
    scalarsPerRank.Attach(collArgs.options.scalarTransport.ptr);
    for (int rank = 0; rank < collArgs.totalRanks; ++rank)
    {
      CHECK_CALL(tempInputCpu.FillPattern(collArgs.dataType, collArgs.numInputElements, rank, false));

      if (rank == collArgs.globalRank)
      {
        if (hipMemcpy(collArgs.inputGpu.ptr, tempInputCpu.ptr, numInputBytes, hipMemcpyHostToDevice) != hipSuccess)
        {
          ERROR("hipMemcpy to input failed\n");
          CHECK_CALL(tempInputCpu.FreeCpuMem());
          CHECK_CALL(tempResultCpu.FreeCpuMem());
          return TEST_FAIL;
        }
      }

      // Scale the temporary input by local scalar for this rank
      // (Used by custom reduction ops)
      if (collArgs.options.scalarMode >= 0)
      {
        CHECK_CALL(tempInputCpu.Scale(collArgs.dataType, collArgs.numInputElements,
                                      scalarsPerRank, rank));
      }

      if (rank == 0)
      {
        memcpy(tempResultCpu.ptr, tempInputCpu.ptr, numInputBytes);
      }
      else
      {
        CHECK_CALL(tempResultCpu.Reduce(collArgs.dataType, collArgs.numInputElements,
                                        tempInputCpu, tempOp));
      }
    }

    // Perform averaging if necessary
    if (collArgs.options.redOp == ncclAvg)
    {
      CHECK_CALL(tempResultCpu.DivideByInt(collArgs.dataType, collArgs.numInputElements, collArgs.totalRanks));
    }

    // Copy over portion of result
    memcpy(collArgs.expected.I1,
           tempResultCpu.I1 + collArgs.globalRank * numOutputBytes,
           numOutputBytes);
    CHECK_CALL(tempInputCpu.FreeCpuMem());
    CHECK_CALL(tempResultCpu.FreeCpuMem());
    return TEST_SUCCESS;
  }

  ErrCode DefaultPrepData_Scatter(CollectiveArgs &collArgs)
  {
    CHECK_CALL(CheckAllocation(collArgs));
    if (collArgs.numInputElements != collArgs.numOutputElements * collArgs.totalRanks)
    {
      ERROR("# of input elements must be total ranks * # output elements for Scatter\n");
      return TEST_FAIL;
    }

    size_t const numInputBytes = collArgs.numInputElements * DataTypeToBytes(collArgs.dataType);
    size_t const numOutputBytes = collArgs.numOutputElements * DataTypeToBytes(collArgs.dataType);

    // Clear outputs on all ranks (prior to input in case of in-place)
    collArgs.outputGpu.ClearGpuMem(numOutputBytes);

    // Generate input as if on root rank - each rank will receive a portion
    PtrUnion tempInput;
    tempInput.AllocateCpuMem(numInputBytes);
    tempInput.FillPattern(collArgs.dataType, collArgs.numInputElements, collArgs.options.root, false);

    // Copy input to root rank
    if (collArgs.globalRank == collArgs.options.root)
    {
      if (hipMemcpy(collArgs.inputGpu.ptr, tempInput.ptr, numInputBytes, hipMemcpyHostToDevice) != hipSuccess)
      {
        ERROR("hipMemcpy to input failed\n");
        tempInput.FreeCpuMem();
        return TEST_FAIL;
      }
    }
    else
    {
      collArgs.inputGpu.ClearGpuMem(numInputBytes);
    }

    // Each rank receive a portion of the input
    memcpy(collArgs.expected.U1, tempInput.U1 + (collArgs.globalRank * numOutputBytes), numOutputBytes);

    tempInput.FreeCpuMem();
    return TEST_SUCCESS;
  }

  ErrCode DefaultPrepData_AllToAll(CollectiveArgs &collArgs)
  {
    CHECK_CALL(CheckAllocation(collArgs));
    if (collArgs.numInputElements != collArgs.numOutputElements)
    {
      ERROR("Number of input elements must match number of output elements for AllToAll\n");
      return TEST_FAIL;
    }
    if (collArgs.numInputElements % collArgs.totalRanks)
    {
      ERROR("Input / Output size for AllToAll must be a multiple of %d\n", collArgs.totalRanks);
      return TEST_FAIL;
    }
    size_t const numInputBytes = collArgs.numInputElements * DataTypeToBytes(collArgs.dataType);
    size_t const numOutputBytes = collArgs.numOutputElements * DataTypeToBytes(collArgs.dataType);
    size_t const numBytes = numInputBytes / collArgs.totalRanks;

    // Clear outputs on all ranks (prior to input in case of in-place)
    collArgs.outputGpu.ClearGpuMem(numOutputBytes);

    // Generate input on root rank - each rank will receive a portion
    PtrUnion tempInput;
    tempInput.Attach(collArgs.outputCpu);

    for (int rank = 0; rank < collArgs.totalRanks; ++rank)
    {
      tempInput.FillPattern(collArgs.dataType, collArgs.numInputElements, rank, false);

      // Copy input
      if (rank == collArgs.globalRank)
      {
        CHECK_HIP(hipMemcpy(collArgs.inputGpu.ptr, tempInput.ptr, numInputBytes, hipMemcpyHostToDevice));
      }
      memcpy(collArgs.expected.U1 + (numBytes * rank), tempInput.U1 + (numBytes * collArgs.globalRank), numBytes);
    }
    return TEST_SUCCESS;
  }

  ErrCode DefaultPrepData_AllToAllv(CollectiveArgs &collArgs)
  {

    CHECK_CALL(CheckAllocation(collArgs));
    size_t const numInputBytes = collArgs.numInputElements * DataTypeToBytes(collArgs.dataType);
    size_t const numOutputBytes = collArgs.numOutputElements * DataTypeToBytes(collArgs.dataType);

    // calculating maxNumElements  as the maximum number of input bytes out of all the ranks
    size_t maxNumElements  = 0;
    for (int sendRank = 0; sendRank < collArgs.totalRanks; ++sendRank)
    for (int recvRank = 0; recvRank < collArgs.totalRanks; ++recvRank)
    {
      size_t rankSendCount = collArgs.options.sdispls[(sendRank)*collArgs.totalRanks+recvRank] + collArgs.options.sendcounts[(sendRank)*collArgs.totalRanks+recvRank];
      maxNumElements = std::max(maxNumElements, rankSendCount);
    }

    // Clear outputs on all ranks (prior to input in case of in-place)
    collArgs.outputGpu.ClearGpuMem(numOutputBytes);

    // Generate input on root rank - each rank will receive a portion
    PtrUnion tempInput;
    tempInput.AllocateCpuMem(maxNumElements*DataTypeToBytes(collArgs.dataType));

    for (int sendRank = 0; sendRank < collArgs.totalRanks; ++sendRank)
    {
      tempInput.FillPattern(collArgs.dataType, maxNumElements, sendRank, false);
      size_t recvDspls = collArgs.options.rdispls[collArgs.globalRank*collArgs.totalRanks + sendRank] * DataTypeToBytes(collArgs.dataType);
      size_t sendDspls = collArgs.options.sdispls[sendRank*collArgs.totalRanks + collArgs.globalRank] * DataTypeToBytes(collArgs.dataType);
      size_t numBytes = collArgs.options.recvcounts[collArgs.globalRank*collArgs.totalRanks + sendRank] * DataTypeToBytes(collArgs.dataType);
      memcpy(collArgs.expected.U1 + recvDspls, tempInput.U1 + sendDspls, numBytes);
    }
    tempInput.FillPattern(collArgs.dataType, collArgs.numInputElements, collArgs.globalRank, false);

    CHECK_HIP(hipMemcpy(collArgs.inputGpu.ptr, tempInput.ptr, numInputBytes, hipMemcpyHostToDevice));

    tempInput.FreeCpuMem();
    return TEST_SUCCESS;
  }

  ErrCode DefaultPrepData_Send(CollectiveArgs &collArgs)
  {
    CHECK_CALL(CheckAllocation(collArgs));
    return collArgs.inputGpu.FillPattern(collArgs.dataType,
                                               collArgs.numInputElements,
                                               collArgs.globalRank, true);
  }

  ErrCode DefaultPrepData_Recv(CollectiveArgs &collArgs)
  {
    CHECK_CALL(CheckAllocation(collArgs));
    return collArgs.expected.FillPattern(collArgs.dataType,
                                         collArgs.numOutputElements,
                                         collArgs.options.root,
                                         false);
  }
}
