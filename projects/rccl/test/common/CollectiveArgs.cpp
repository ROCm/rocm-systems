/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <algorithm>

#include "CollectiveArgs.hpp"
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
    // Free scalar based on previous scalarMode
    if (optionalColArgs.scalarMode != -1)
    {
      if (this->localScalar.ptr != nullptr)
      {
        if (this->options.scalarMode == 0) CHECK_CALL(this->localScalar.FreeGpuMem());
        if (this->options.scalarMode == 1) CHECK_HIP(hipHostFree(this->localScalar.ptr));
      }
    }

    this->globalRank        = globalRank;
    this->totalRanks        = totalRanks;
    this->deviceId          = deviceId;
    this->funcType          = funcType;
    this->dataType          = dataType;
    this->numInputElements  = numInputElements;
    this->numOutputElements = numOutputElements;
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
      if (this->funcType == ncclCollScatter)
      {
        CHECK_CALL(this->inputGpu.AllocateGpuMem(this->numInputBytesAllocated, useManagedMem, userRegistered));
        this->outputGpu.Attach(this->inputGpu.U1 + (this->globalRank  * this->numOutputBytesAllocated));
      }
      else if (this->funcType == ncclCollGather || this->funcType == ncclCollAllGather)
      {
        CHECK_CALL(this->outputGpu.AllocateGpuMem(this->numOutputBytesAllocated, useManagedMem, userRegistered));
        this->inputGpu.Attach(this->outputGpu.U1 + (this->globalRank * this->numInputBytesAllocated));
      }
      else
      {
        size_t const numBytes = std::max(this->numInputBytesAllocated, this->numOutputBytesAllocated);
        CHECK_CALL(this->inputGpu.AllocateGpuMem(numBytes, useManagedMem, userRegistered));
        this->outputGpu.Attach(this->inputGpu.ptr);
      }
      CHECK_CALL(this->expected.AllocateCpuMem(this->numOutputBytesAllocated));
    }
    else
    {
      CHECK_CALL(this->inputGpu.AllocateGpuMem(this->numInputBytesAllocated, useManagedMem, userRegistered));
      CHECK_CALL(this->outputGpu.AllocateGpuMem(this->numOutputBytesAllocated, useManagedMem, userRegistered));
      CHECK_CALL(this->expected.AllocateCpuMem(this->numOutputBytesAllocated));
    }
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
      CHECK_CALL(this->expectedGpu.AllocateGpuMem(this->numOutputBytesAllocated, useManagedMem, false));
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

    // Device-data mode: compare outputGpu vs the device-built expectedGpu on the GPU
    // (no D2H copy, no host element loop), using the same per-type tolerances as IsEqual.
    if (UtDeviceDataEnabled() && this->expectedOnDevice)
    {
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

    CHECK_HIP(hipMemcpy(this->outputCpu.ptr, this->outputGpu.ptr, numOutputBytes, hipMemcpyDeviceToHost));

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
    // Mitigation (AICOMRCCL-2275): 4 MiB of device writes at teardown clears the
    // pooled-worker corruption. Timing, not contents: prep rewrites these buffers.
    size_t const cap = 4u << 20;
    hipError_t errIn = hipSuccess, errOut = hipSuccess;
    // In-place Attaches one buffer as an interior alias of the other, so there is a
    // single allocation and its owner is the lower address; its own length fits inside.
    if (this->inPlace)
    {
      bool const inLower = this->inputGpu.ptr <= this->outputGpu.ptr;
      void* const  own   = inLower ? this->inputGpu.ptr : this->outputGpu.ptr;
      size_t const bytes = inLower ? this->numInputBytesAllocated : this->numOutputBytesAllocated;
      if (own && bytes)
      {
        errIn = hipMemset(own, 0, std::min(bytes, cap));
      }
    }
    else
    {
      if (this->inputGpu.ptr && this->numInputBytesAllocated)
      {
        errIn = hipMemset(this->inputGpu.ptr, 0, std::min(this->numInputBytesAllocated, cap));
      }
      if (this->outputGpu.ptr && this->numOutputBytesAllocated)
      {
        errOut = hipMemset(this->outputGpu.ptr, 0, std::min(this->numOutputBytesAllocated, cap));
      }
    }

    // If in-place, either only inputGpu or outputGpu was allocated
    if (this->inPlace)
    {
      if (this->funcType == ncclCollGather)
        this->outputGpu.FreeGpuMem();
      else
        this->inputGpu.FreeGpuMem(this->userRegistered);
    }
    else
    {
      this->inputGpu.FreeGpuMem(this->userRegistered);
      this->outputGpu.FreeGpuMem(this->userRegistered);
    }

    this->outputCpu.FreeCpuMem();
    this->expected.FreeCpuMem();
    if (this->expectedGpu.ptr != nullptr)
    {
      this->expectedGpu.FreeGpuMem();
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
      this->biasGpu.FreeGpuMem(this->userRegistered);
      this->biasCpu.FreeCpuMem();
      this->biasRegHandle = nullptr;
    }

    CHECK_HIP(errIn != hipSuccess ? errIn : errOut);
    return TEST_SUCCESS;
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
