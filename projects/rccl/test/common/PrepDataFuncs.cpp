/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "CollectiveArgs.hpp"
#include "PrepDataFuncs.hpp"
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>

namespace RcclUnitTesting
{

  class ScopedDevice {
  public:
    explicit ScopedDevice(int targetDeviceId) {
      // Save the caller's current device
      if (hipGetDevice(&savedDeviceId_) != hipSuccess) {
        savedDeviceId_ = 0;
      }
      // Only switch context if target is different from current
      if (savedDeviceId_ != targetDeviceId) {
        hipSetDevice(targetDeviceId);
      }
    }

    ~ScopedDevice() {
      // Automatically restore caller's original device on scope exit
      hipSetDevice(savedDeviceId_);
    }

    // Prevent copying/moving
    ScopedDevice(const ScopedDevice&) = delete;
    ScopedDevice& operator=(const ScopedDevice&) = delete;

  private:
    int savedDeviceId_ = 0;
  };
  // Byte written into expectedGpu[0] by the UT_DEVICE_DATA_FAULT negative control.
  static constexpr int kDeviceDataFaultByte = 0xFF;

  // Negative control: when UT_DEVICE_DATA_FAULT is set to a non-zero value, corrupt one
  // expected element so a correct collective output must mismatch (proves device validate
  // is not a no-op). No-op when the var is unset or "0".
  static ErrCode MaybeInjectDeviceDataFault(CollectiveArgs& collArgs)
  {
    char const* faultEnv = getenv("UT_DEVICE_DATA_FAULT");
    if (faultEnv != nullptr && atoi(faultEnv) != 0)
    {
      CHECK_HIP(hipMemset(collArgs.expectedGpu.ptr, kDeviceDataFaultByte,
                          DataTypeToBytes(collArgs.dataType)));
      fprintf(stdout, "[UT][device-data] FAULT injected into expectedGpu[0] (rank %d)\n",
              collArgs.globalRank);
      fflush(stdout);
    }
    return TEST_SUCCESS;
  }

  ErrCode DefaultPrepareDataFunc(CollectiveArgs &collArgs)
  {
    ScopedDevice dev(collArgs.deviceId);
    switch (collArgs.funcType)
    {
    case ncclCollBroadcast:     return DefaultPrepData_Broadcast(collArgs);
    case ncclCollReduce:        return DefaultPrepData_Reduce(collArgs, false);
    case ncclCollAllGather:     return DefaultPrepData_Gather(collArgs, true);
    case ncclCollReduceScatter: return DefaultPrepData_ReduceScatter(collArgs);
    case ncclCollAllReduce:     return DefaultPrepData_Reduce(collArgs, true);
    case ncclCollGather:        return DefaultPrepData_Gather(collArgs, false);
    case ncclCollScatter:       return DefaultPrepData_Scatter(collArgs);
    case ncclCollAlltoAll:      return DefaultPrepData_AllToAll(collArgs);
    case ncclCollAlltoAllv:     return DefaultPrepData_AllToAllv(collArgs);
    case ncclCollSend:          return DefaultPrepData_Send(collArgs);
    case ncclCollRecv:          return DefaultPrepData_Recv(collArgs);
    default:
      TEST_ERROR("Unknown func type %d", collArgs.funcType);
      return TEST_FAIL;
    }
  }

  ErrCode CheckAllocation(CollectiveArgs const& collArgs)
  {
    if (collArgs.numInputElements > collArgs.numInputElementsAllocated)
    {
      TEST_ERROR("Number of input elements (%lu) exceeds the number of allocated input elements (%lu)",
            collArgs.numInputElements, collArgs.numInputElementsAllocated);
      return TEST_FAIL;
    }

    if (collArgs.numOutputElements > collArgs.numOutputElementsAllocated)
    {
      TEST_ERROR("Number of output elements (%lu) exceeds the number of allocated output elements (%lu)",
            collArgs.numOutputElements, collArgs.numOutputElementsAllocated);
      return TEST_FAIL;
    }

    // Check bias allocation if bias is enabled
    if (collArgs.options.useBias)
    {
      if (collArgs.numBiasElements == 0 || collArgs.numBiasBytesAllocated == 0)
      {
        TEST_ERROR("Bias is enabled but bias buffers are not allocated");
        return TEST_FAIL;
      }
      if (collArgs.numBiasElements != collArgs.numOutputElements)
      {
        TEST_ERROR("Number of bias elements (%lu) must match number of output elements (%lu)",
              collArgs.numBiasElements, collArgs.numOutputElements);
        return TEST_FAIL;
      }
    }
    return TEST_SUCCESS;
  }

  ErrCode DefaultPrepData_Broadcast(CollectiveArgs &collArgs)
  {
    CHECK_CALL(CheckAllocation(collArgs));
    if (collArgs.numInputElements != collArgs.numOutputElements)
    {
      TEST_ERROR("Number of input elements must match number of output elements for Broadcast");
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
      TEST_ERROR("Number of input elements must match number of output elements for Reduce");
      return TEST_FAIL;
    }

    size_t const numBytes = collArgs.numInputElements * DataTypeToBytes(collArgs.dataType);

    // Device-data mode (AllReduce only, standard reduction): build this rank's input
    // and the all-ranks reduced expected entirely on the GPU. Custom-op / scalar /
    // bias / constant-input variants keep the host reference path.
    if (UtDeviceDataEnabled() && isAllReduce
        && UtDeviceDtypeSupported(collArgs.dataType)
        && collArgs.numInputElements >= UtDeviceDataMinElements()
        && collArgs.options.scalarMode < 0
        && !collArgs.options.useBias
        && collArgs.options.inputConstantValue < 0)
    {
      CHECK_CALL(collArgs.outputGpu.ClearGpuMem(numBytes));
      CHECK_CALL(collArgs.inputGpu.FillPatternDevice(collArgs.dataType, collArgs.numInputElements,
                                                     collArgs.globalRank, 0));
      CHECK_CALL(collArgs.expectedGpu.FillReducedPatternDevice(collArgs.dataType, collArgs.numInputElements,
                                                               collArgs.totalRanks, collArgs.options.redOp));
      CHECK_CALL(MaybeInjectDeviceDataFault(collArgs));
      collArgs.expectedOnDevice = true;
      return TEST_SUCCESS;
    }

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
      if (collArgs.options.inputConstantValue >= 0)
      {
        // Use constant value for all input elements across all ranks
        // This is useful for ncclProd at high rank counts to avoid factorial overflow
        for (size_t i = 0; i < collArgs.numInputElements; i++)
        {
          CHECK_CALL(tempInputCpu.Set(collArgs.dataType, i,
                                      collArgs.options.inputConstantValue,
                                      (double)collArgs.options.inputConstantValue));
        }
      }
      else
      {
        // Use rank-based pattern: value[rank][i] = (rank + i) % 256 (default behavior)
        CHECK_CALL(tempInputCpu.FillPattern(collArgs.dataType, collArgs.numInputElements, rank, false));
      }

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

    // Add bias to expected output if bias is enabled
    if (collArgs.options.useBias && (isAllReduce || collArgs.options.root == collArgs.globalRank))
    {
      // Initialize bias data on CPU
      if (collArgs.options.biasConstantValue >= 0)
      {
        // Use constant value for all bias elements (useful for ncclProd to avoid overflow)
        for (size_t i = 0; i < collArgs.numBiasElements; i++)
        {
          CHECK_CALL(collArgs.biasCpu.Set(collArgs.dataType, i,
                                          collArgs.options.biasConstantValue,
                                          (double)collArgs.options.biasConstantValue));
        }
      }
      else
      {
        // Use incremental pattern: bias[i] = i (default behavior)
        CHECK_CALL(collArgs.biasCpu.FillPattern(collArgs.dataType, collArgs.numBiasElements, 0, false));
      }

      // Copy bias data to GPU
      size_t const biasBytes = collArgs.numBiasBytesAllocated;
      CHECK_HIP(hipMemcpy(collArgs.biasGpu.ptr, collArgs.biasCpu.ptr, biasBytes, hipMemcpyHostToDevice));

      // Apply bias to expected output using the SAME reduction operation as AllReduce
      CHECK_CALL(result.Reduce(collArgs.dataType, collArgs.numInputElements, collArgs.biasCpu, tempOp));

      // Update the biasPtr in options to point to the GPU buffer
      collArgs.options.biasPtr = collArgs.biasGpu.ptr;
    }

    return TEST_SUCCESS;
  }

  ErrCode DefaultPrepData_Gather(CollectiveArgs &collArgs, bool const isAllGather)
  {
    CHECK_CALL(CheckAllocation(collArgs));
    if (collArgs.totalRanks * collArgs.numInputElements != collArgs.numOutputElements)
    {
      TEST_ERROR("# of output elements must be total ranks * # input elements for AllGather");
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
      TEST_ERROR("# of input elements must be total ranks * # output elements for ReduceScatter");
      return TEST_FAIL;
    }

    size_t const numInputBytes  = collArgs.numInputElements * DataTypeToBytes(collArgs.dataType);
    size_t const numOutputBytes = collArgs.numOutputElements * DataTypeToBytes(collArgs.dataType);

    // Device-data mode: build this rank's full input (totalRanks*N) on the GPU and the
    // reduced expected SLICE for this rank directly on the GPU, mirroring the host loop
    // below but with no host fill / reduce / D2H. ReduceScatter output[j] for this rank
    // is reduce-over-ranks of the pattern at global index globalRank*numOutput + j, so we
    // pass that startIdx to the reduced-pattern kernel. Standard reductions only; the
    // scalar/custom-op path keeps the host reference below.
    if (UtDeviceDataEnabled()
        && UtDeviceDtypeSupported(collArgs.dataType)
        && collArgs.numInputElements >= UtDeviceDataMinElements()
        && collArgs.options.scalarMode < 0)
    {
      CHECK_CALL(collArgs.outputGpu.ClearGpuMem(numOutputBytes));
      CHECK_CALL(collArgs.inputGpu.FillPatternDevice(collArgs.dataType, collArgs.numInputElements,
                                                     collArgs.globalRank, 0));
      CHECK_CALL(collArgs.expectedGpu.FillReducedPatternDevice(collArgs.dataType, collArgs.numOutputElements,
                                                               collArgs.totalRanks, collArgs.options.redOp,
                                                               (size_t)collArgs.globalRank * collArgs.numOutputElements));
      CHECK_CALL(MaybeInjectDeviceDataFault(collArgs));
      collArgs.expectedOnDevice = true;
      return TEST_SUCCESS;
    }

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
          TEST_ERROR("hipMemcpy to input failed");
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
      TEST_ERROR("# of input elements must be total ranks * # output elements for Scatter");
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
        TEST_ERROR("hipMemcpy to input failed");
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
      TEST_ERROR("Number of input elements must match number of output elements for AllToAll");
      return TEST_FAIL;
    }
    if (collArgs.numInputElements % collArgs.totalRanks)
    {
      TEST_ERROR("Input / Output size for AllToAll must be a multiple of %d", collArgs.totalRanks);
      return TEST_FAIL;
    }
    size_t const numInputBytes = collArgs.numInputElements * DataTypeToBytes(collArgs.dataType);
    size_t const numOutputBytes = collArgs.numOutputElements * DataTypeToBytes(collArgs.dataType);
    size_t const numBytes = numInputBytes / collArgs.totalRanks;

    // Device-data mode: build input and expected entirely on the GPU (no host fill,
    // no H2D copy). AllToAll is a pure permutation, so expected block `rank` is this
    // rank's slice of source rank `rank`'s pattern: FillPatternDevice with the source
    // rank and a start index of globalRank*elemsPerBlock reproduces it exactly.
    if (UtDeviceDataEnabled() && UtDeviceDtypeSupported(collArgs.dataType)
        && collArgs.numInputElements >= UtDeviceDataMinElements())
    {
      size_t const elemsPerBlock = collArgs.numInputElements / collArgs.totalRanks;
      CHECK_CALL(collArgs.outputGpu.ClearGpuMem(numOutputBytes));
      CHECK_CALL(collArgs.inputGpu.FillPatternDevice(collArgs.dataType,
                                                     collArgs.numInputElements,
                                                     collArgs.globalRank, 0));
      for (int rank = 0; rank < collArgs.totalRanks; ++rank)
      {
        PtrUnion blockDst;
        blockDst.Attach(collArgs.expectedGpu.U1 + (numBytes * rank));
        CHECK_CALL(blockDst.FillPatternDevice(collArgs.dataType, elemsPerBlock, rank,
                                              (size_t)collArgs.globalRank * elemsPerBlock));
      }
      CHECK_CALL(MaybeInjectDeviceDataFault(collArgs));
      collArgs.expectedOnDevice = true;
      return TEST_SUCCESS;
    }

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
