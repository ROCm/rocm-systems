/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "CollectiveArgs.hpp"
#include "VerifiableData.hpp"
#include "gtest/gtest.h"

namespace RcclUnitTesting
{
  ErrCode CollectiveArgs::SetArgs(int             const  globalRank,
                                  int             const  totalRanks,
                                  int             const  deviceId,
                                  ncclFunc_t      const  funcType,
                                  ncclDataType_t  const  dataType,
                                  size_t          const  numInputElements,
                                  size_t          const  numOutputElements,
                                  int             const  streamIdx,
                                  OptionalColArgs const  &optionalColArgs)
  {
    // Free scalar based on the previous scalarMode, including when the new
    // arguments disable scalar mode.
    if (this->localScalar.ptr != nullptr)
    {
      if (this->options.scalarMode == 0) CHECK_CALL(this->localScalar.FreeGpuMem());
      if (this->options.scalarMode == 1) CHECK_HIP(hipHostFree(this->localScalar.ptr));
      this->localScalar.Attach(nullptr);
    }

    this->globalRank        = globalRank;
    this->totalRanks        = totalRanks;
    this->deviceId          = deviceId;
    this->funcType          = funcType;
    this->dataType          = dataType;
    this->numInputElements  = numInputElements;
    this->numOutputElements = numOutputElements;
    if (this->inputGpu.ptr != nullptr || this->outputGpu.ptr != nullptr)
    {
      CHECK_CALL(this->AttachMem());
    }
    this->streamIdx         = streamIdx;
    this->options           = optionalColArgs;

    if (this->options.scalarMode != -1)
    {
      size_t const numBytes = DataTypeToBytes(dataType);
      if (this->options.scalarMode == ncclScalarDevice)
      {
        CHECK_CALL(this->localScalar.AllocateGpuMem(numBytes));
        CHECK_HIP(hipMemcpy(this->localScalar.ptr, optionalColArgs.scalarTransport.ptr + (globalRank * numBytes),
                            numBytes, hipMemcpyHostToDevice));
      }
      else if (this->options.scalarMode == ncclScalarHostImmediate)
      {
        CHECK_HIP(hipHostMalloc(&this->localScalar.ptr, numBytes, 0));
        memcpy(this->localScalar.ptr, optionalColArgs.scalarTransport.ptr + (globalRank * numBytes), numBytes);
      }
    }
    return TEST_SUCCESS;
  }

  ErrCode CollectiveArgs::AttachMem()
  {
    // Calculate the current active bytes based on this iteration's element count
    size_t currentInputBytes = this->numInputElements * DataTypeToBytes(this->dataType);
    size_t currentOutputBytes = this->numOutputElements * DataTypeToBytes(this->dataType);

    // For out-of-place, both pointers remain at the start of their respective base allocations.
    // No attachment/offsetting is necessary.
    if (!this->inPlace) return TEST_SUCCESS;

    size_t requiredBytes = 0;
    size_t allocatedBytes = 0;
    if (this->funcType == ncclCollScatter || this->funcType == ncclCollReduceScatter)
    {
      requiredBytes  = (this->globalRank + 1) * currentOutputBytes;
      allocatedBytes = this->numInputBytesAllocated;
    }
    else if (this->funcType == ncclCollGather || this->funcType == ncclCollAllGather)
    {
      requiredBytes  = (this->globalRank + 1) * currentInputBytes;
      allocatedBytes = this->numOutputBytesAllocated;
    }
    else
    {
      requiredBytes  = std::max(currentInputBytes, currentOutputBytes);
      allocatedBytes = std::max(this->numInputBytesAllocated, this->numOutputBytesAllocated);
    }
    if (requiredBytes > allocatedBytes)
    {
      TEST_ERROR("Rank %d in-place %s needs %zu bytes but only %zu were allocated",
            this->globalRank, ncclFuncNames[this->funcType], requiredBytes, allocatedBytes);
      return TEST_FAIL;
    }

    if (this->funcType == ncclCollScatter || this->funcType == ncclCollReduceScatter)
    {
      // inputGpu holds the base pointer. Offset outputGpu.
      this->outputGpu.Attach(this->inputGpu.U1 + (this->globalRank * currentOutputBytes));
    }
    else if (this->funcType == ncclCollGather || this->funcType == ncclCollAllGather)
    {
      // outputGpu holds the base pointer. Offset inputGpu.
      this->inputGpu.Attach(this->outputGpu.U1 + (this->globalRank * currentInputBytes));
    }
    else
    {
      // Both buffers share the exact same base pointer
      this->outputGpu.Attach(this->inputGpu.ptr);
    }
    return TEST_SUCCESS;
  }

  ErrCode CollectiveArgs::AllocateMem(bool   const inPlace,
                                      bool   const useManagedMem,
                                      bool   const userRegistered)
  {
    this->numInputBytesAllocated     = this->numInputElements * DataTypeToBytes(this->dataType);
    this->numOutputBytesAllocated    = this->numOutputElements * DataTypeToBytes(this->dataType);
    this->numInputElementsAllocated  = this->numInputElements;
    this->numOutputElementsAllocated = this->numOutputElements;
    this->inPlace                    = inPlace;
    this->useManagedMem              = useManagedMem;
    this->userRegistered             = userRegistered;

    CHECK_HIP(hipSetDevice(this->deviceId));

    if (inPlace)
    {
      if (this->funcType == ncclCollScatter || this->funcType == ncclCollReduceScatter)
      {
        CHECK_CALL(this->inputGpu.AllocateGpuMem(this->numInputBytesAllocated, useManagedMem, userRegistered));
      }
      else if (this->funcType == ncclCollGather || this->funcType == ncclCollAllGather)
      {
        CHECK_CALL(this->outputGpu.AllocateGpuMem(this->numOutputBytesAllocated, useManagedMem, userRegistered));
      }
      else
      {
        size_t const numBytes = std::max(this->numInputBytesAllocated, this->numOutputBytesAllocated);
        CHECK_CALL(this->inputGpu.AllocateGpuMem(numBytes, useManagedMem, userRegistered));
      }
      CHECK_CALL(this->AttachMem());
    }
    else
    {
      CHECK_CALL(this->inputGpu.AllocateGpuMem(this->numInputBytesAllocated, useManagedMem, userRegistered));
      CHECK_CALL(this->outputGpu.AllocateGpuMem(this->numOutputBytesAllocated, useManagedMem, userRegistered));
    }
    CHECK_CALL(this->expected.AllocateCpuMem(this->numOutputBytesAllocated));
    CHECK_CALL(this->outputCpu.AllocateCpuMem(this->numOutputBytesAllocated));

    // Device-data mode: a device-resident expected buffer for device-side validate.
    // Allocated only for collectives whose prep func builds expected on the GPU
    // (AllToAll, AllReduce, ReduceScatter), so no other collective pays an extra
    // device buffer.
    this->expectedOnDevice = false;
    if (UtDeviceDataEnabled() &&
        (this->funcType == ncclCollAlltoAll || this->funcType == ncclCollAllReduce
         || this->funcType == ncclCollReduceScatter))
    {
      // userRegistered must be passed, otherwise in the case of symmetric memory,
      // data validation failures show up with UT_DEVICE_DATA=1 but not with 0
      // it is verified that even expected [CPU data] !=  expectedGpu .
      // ncclMemAlloc() +  hipMallocManaged/hipMalloc is not compatible.
      CHECK_CALL(this->expectedGpu.AllocateGpuMem(this->numOutputBytesAllocated, useManagedMem, userRegistered));
    }

    // Allocate bias buffers if bias is enabled
    if (this->options.useBias)
    {
      this->numBiasElements = this->options.biasNumElements;
      this->numBiasBytesAllocated = this->numBiasElements * DataTypeToBytes(this->dataType);
      CHECK_CALL(this->biasGpu.AllocateGpuMem(this->numBiasBytesAllocated, useManagedMem, userRegistered));
      CHECK_CALL(this->biasCpu.AllocateCpuMem(this->numBiasBytesAllocated));
      this->biasRegHandle = nullptr;
    }

    return TEST_SUCCESS;
  }

  ErrCode CollectiveArgs::PrepareData(CollFuncPtr const prepareDataFunc)
  {
    // Reset per call: buffers are reused across sub-cases (AllocateMem is not re-run for
    // each), so a prior device sub-case must not leave this true for a later host-path
    // sub-case (which would validate against a stale expectedGpu). Device prep funcs set
    // it true only when they actually build expectedGpu.
    this->expectedOnDevice = false;
    this->usesVerifiableData = false;
    CollFuncPtr prepFunc = (prepareDataFunc == nullptr ? DefaultPrepareDataFunc : prepareDataFunc);
    return prepFunc(*this);
  }

  ErrCode CollectiveArgs::ValidateResults()
  {
    // Ignore non-root outputs for collectives with a root, except Broadcast/Scatter where
    // every rank receives a defined result and must be validated.
    if (CollectiveArgs::UsesRoot(this->funcType) &&
        this->funcType != ncclCollBroadcast && this->funcType != ncclCollScatter &&
        this->options.root != this->globalRank) return TEST_SUCCESS;
    if (this->funcType == ncclCollSend) return TEST_SUCCESS; // on the send receive pair only recv needs to be checked
    size_t const numOutputBytes = (this->numOutputElements * DataTypeToBytes(this->dataType));

    bool isMatch = true;

    if (this->usesVerifiableData) return VerifiableValidate(*this);

    // Device-data mode: compare outputGpu vs the device-built expectedGpu on the GPU
    // (no D2H copy, no host element loop), using the same per-type tolerances as IsEqual.
    if (UtDeviceDataEnabled() && this->expectedOnDevice)
    {
      CHECK_HIP(hipSetDevice(this->deviceId));
      size_t mismatches = 0;
      CHECK_CALL(PtrUnion::IsEqualDevice(this->dataType,
                                         this->numOutputElements,
                                         this->outputGpu.ptr,
                                         this->expectedGpu.ptr,
                                         mismatches));
      isMatch = (mismatches == 0);
      if (!isMatch)
      {
        TEST_ERROR("Mismatch (%zu elements) for %s", mismatches, this->GetDescription().c_str());
      }
      return isMatch ? TEST_SUCCESS : TEST_FAIL;
    }

    if (this->funcType == ncclCollRecv && numOutputBytes != 0)
    {
      // outputCpu comes from AllocateCpuMem, which is calloc, so the buffer is
      // pageable. Send returns above; only Recv reads a result. ExecuteCollectives
      // has already synchronized the collective streams, including the extra
      // stream sync that flushes the GPU cache before validation. That does not
      // cover this copy. On gfx1250, hipMemcpy into the pageable buffer can
      // still return success before the GPU has a stable mapping for the
      // destination. SendRecv.SinglePairs then faulted on the host heap or
      // compared a partial result. Copy into page-locked memory, wait for the
      // device, then memcpy into the comparison buffer. Other collectives did
      // not hit this and keep the direct hipMemcpy below.
      PtrUnion staging;
      CHECK_HIP(hipHostMalloc(&staging.ptr, numOutputBytes));
      // CHECK_HIP returns on failure, so free the staging buffer before that
      // return. Otherwise a failed copy or sync leaks the pinned allocation.
      hipError_t stagingCopy = hipMemcpy(staging.ptr, this->outputGpu.ptr, numOutputBytes, hipMemcpyDeviceToHost);
      hipError_t stagingSync = hipSuccess;
      if (stagingCopy == hipSuccess)
        stagingSync = hipDeviceSynchronize();
      if (stagingCopy != hipSuccess || stagingSync != hipSuccess)
      {
        (void)hipHostFree(staging.ptr);
        staging.ptr = nullptr;
        CHECK_HIP(stagingCopy != hipSuccess ? stagingCopy : stagingSync);
      }
      memcpy(this->outputCpu.ptr, staging.ptr, numOutputBytes);
      CHECK_HIP(hipHostFree(staging.ptr));
    }
    else
    {
      CHECK_HIP(hipMemcpy(this->outputCpu.ptr, this->outputGpu.ptr, numOutputBytes, hipMemcpyDeviceToHost));
    }

    CHECK_CALL(this->outputCpu.IsEqual(this->dataType,
                                       this->numOutputElements,
                                       this->expected,
                                       true,
                                       isMatch));
    if (!isMatch) TEST_ERROR("Mismatch for %s", this->GetDescription().c_str());
    return isMatch ? TEST_SUCCESS : TEST_FAIL;
  }

  ErrCode CollectiveArgs::DeallocateMem()
  {
    // Free everything even if one release fails, then report the failure.
    ErrCode status = TEST_SUCCESS;
    auto track = [&status](ErrCode const result) { if (result != TEST_SUCCESS) status = result; };

    // If in-place, either only inputGpu or outputGpu was allocated; the other
    // is an alias into it and is cleared without being freed.
    if (this->inPlace)
    {
      if (this->funcType == ncclCollGather || this->funcType == ncclCollAllGather)
      {
        track(this->outputGpu.FreeGpuMem(this->userRegistered));
        this->inputGpu.Attach(nullptr);
      }
      else
      {
        track(this->inputGpu.FreeGpuMem(this->userRegistered));
        this->outputGpu.Attach(nullptr);
      }
    }
    else
    {
      track(this->inputGpu.FreeGpuMem(this->userRegistered));
      track(this->outputGpu.FreeGpuMem(this->userRegistered));
    }

    this->outputCpu.FreeCpuMem();
    this->expected.FreeCpuMem();
    if (this->expectedGpu.ptr != nullptr)
    {
      track(this->expectedGpu.FreeGpuMem(this->userRegistered));
    }

    if (this->localScalar.ptr != nullptr)
    {
      if (this->options.scalarMode == 0) this->localScalar.FreeGpuMem();
      if (this->options.scalarMode == 1) CHECK_HIP(hipHostFree(this->localScalar.ptr));
      this->localScalar.Attach(nullptr);
    }

    // Deallocate bias buffers if they were allocated
    if (this->options.useBias && this->numBiasBytesAllocated > 0)
    {
      track(this->biasGpu.FreeGpuMem(this->userRegistered));
      this->biasCpu.FreeCpuMem();
      this->biasRegHandle = nullptr;
    }

    return status;
  }

  std::string CollectiveArgs::GetDescription() const
  {
    std::stringstream ss;

    ss << "(Rank " << this->globalRank << ") ";
    switch (this->funcType)
    {
    case ncclCollBroadcast:     ss << "ncclBroadcast";     break;
    case ncclCollReduce:        ss << "ncclReduce";        break;
    case ncclCollAllGather:     ss << "ncclAllGather";     break;
    case ncclCollReduceScatter: ss << "ncclReduceScatter"; break;
    case ncclCollAllReduce:     ss << "ncclAllReduce";     break;
    case ncclCollGather:        ss << "ncclGather";        break;
    case ncclCollScatter:       ss << "ncclScatter";       break;
    case ncclCollAlltoAll:      ss << "ncclAlltoAll";      break;
    case ncclCollAlltoAllv:     ss << "ncclAlltoAllv";     break;
    case ncclCollSend:          ss << "ncclSend";          break;
    case ncclCollRecv:          ss << "ncclRecv";          break;
    default:                    ss << "[Unknown]";         break;
    }

    ss << " " << ncclDataTypeNames[this->dataType] << " ";
    if (this->funcType == ncclCollReduce ||
        this->funcType == ncclCollReduceScatter ||
        this->funcType == ncclCollAllReduce)
    {
      if (this->options.redOp < ncclNumOps)
      {
        ss << ncclRedOpNames[this->options.redOp] << " ";
      }
      else
      {
        ss << "CustomScalar ";
        PtrUnion scalarsPerRank;
        scalarsPerRank.Attach(scalarsPerRank.ptr);
        switch (this->dataType)
        {
        case ncclInt8:       ss << scalarsPerRank.I1[this->globalRank]; break;
        case ncclUint8:      ss << scalarsPerRank.U1[this->globalRank]; break;
        case ncclInt32:      ss << scalarsPerRank.I4[this->globalRank]; break;
        case ncclUint32:     ss << scalarsPerRank.U4[this->globalRank]; break;
        case ncclInt64:      ss << scalarsPerRank.I8[this->globalRank]; break;
        case ncclUint64:     ss << scalarsPerRank.U8[this->globalRank]; break;
        case ncclFloat8e4m3: ss << (float)scalarsPerRank.F1[this->globalRank]; break;
        case ncclFloat32:    ss << scalarsPerRank.F4[this->globalRank]; break;
        case ncclFloat64:    ss << scalarsPerRank.F8[this->globalRank]; break;
        case ncclFloat8e5m2: ss << (float)scalarsPerRank.B1[this->globalRank]; break;
        case ncclBfloat16:   ss << (float)scalarsPerRank.B2[this->globalRank]; break;
        default:             ss << "(UNKNOWN)";
        }
        ss << " ";
      }
    }

    if (this->funcType == ncclCollBroadcast ||
        this->funcType == ncclCollReduce ||
        this->funcType == ncclCollGather ||
        this->funcType == ncclCollScatter)
    {
      ss << "Root " << this->options.root << " ";
    }

    if (this->funcType == ncclCollSend ||
        this->funcType == ncclCollRecv)
    {
      ss << "Peer " << this->options.root << " ";
    }

    ss << "#In: " << this->numInputElements;
    ss << " #Out: " << this->numOutputElements;

    return ss.str();
  }

  void CollectiveArgs::GetNumElementsForFuncType(ncclFunc_t const funcType,
                                                 int        const N,
                                                 int        const totalRanks,
                                                 int*             numInputElements,
                                                 int*             numOutputElements)
  {
    switch (funcType)
    {
    case ncclCollBroadcast:
    case ncclCollReduce:
    case ncclCollAllReduce:
      *numInputElements  = N;
      *numOutputElements = N;
      break;
    case ncclCollGather:
    case ncclCollAllGather:
      *numInputElements  = N;
      *numOutputElements = totalRanks * N;
      break;
    case ncclCollScatter:
    case ncclCollReduceScatter:
      *numInputElements  = totalRanks * N;
      *numOutputElements = N;
      break;
    case ncclCollAlltoAll:
      *numInputElements = totalRanks * N;
      *numOutputElements = totalRanks * N;
      break;
    default:
      *numInputElements = N;
      *numOutputElements = N;
      break;
    }
  }

  bool CollectiveArgs::UsesReduce(ncclFunc_t const funcType)
  {
    return (funcType == ncclCollReduce    ||
            funcType == ncclCollAllReduce ||
            funcType == ncclCollReduceScatter);
  }

  bool CollectiveArgs::UsesRoot(ncclFunc_t const funcType)
  {
    return (funcType == ncclCollBroadcast ||
            funcType == ncclCollReduce    ||
            funcType == ncclCollGather    ||
            funcType == ncclCollScatter);
  }
}
