/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

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
        if (this->options.scalarMode == 0) this->localScalar.FreeGpuMem();
        if (this->options.scalarMode == 1) hipHostFree(this->localScalar.ptr);
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

    if (hipSetDevice(this->deviceId) != hipSuccess)
    {
      ERROR("Unable to call hipSetDevice to set to GPU %d\n", this->deviceId);
      return TEST_FAIL;
    }

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
    return TEST_SUCCESS;
  }

  ErrCode CollectiveArgs::PrepareData(CollFuncPtr const prepareDataFunc)
  {
    CollFuncPtr prepFunc = (prepareDataFunc == nullptr ? DefaultPrepareDataFunc : prepareDataFunc);
    return prepFunc(*this);
  }

  ErrCode CollectiveArgs::ValidateResults()
  {
    // Ignore non-root outputs for collectives with a root
    if (CollectiveArgs::UsesRoot(this->funcType) && this->options.root != this->globalRank) return TEST_SUCCESS;
    if (this->funcType == ncclCollSend) return TEST_SUCCESS; // on the send receive pair only recv needs to be checked
    size_t const numOutputBytes = (this->numOutputElements * DataTypeToBytes(this->dataType));

    CHECK_HIP(hipMemcpy(this->outputCpu.ptr, this->outputGpu.ptr, numOutputBytes, hipMemcpyDeviceToHost));

    bool isMatch = true;
    CHECK_CALL(this->outputCpu.IsEqual(this->dataType,
                                       this->numOutputElements,
                                       this->expected,
                                       true,
                                       isMatch));
    if (!isMatch) ERROR("Mismatch for %s\n", this->GetDescription().c_str());
    return isMatch ? TEST_SUCCESS : TEST_FAIL;
  }

  ErrCode CollectiveArgs::DeallocateMem()
  {
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

    if (this->localScalar.ptr != nullptr)
    {
      if (this->options.scalarMode == 0) this->localScalar.FreeGpuMem();
      if (this->options.scalarMode == 1) CHECK_HIP(hipHostFree(this->localScalar.ptr));
      this->localScalar.Attach(nullptr);
    }
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
    case ncclCollAllToAll:      ss << "ncclAllToAll";      break;
    case ncclCollAllToAllv:     ss << "ncclAllToAllv";     break;
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
        case ncclBfloat16:   ss << scalarsPerRank.B2[this->globalRank]; break;
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
    case ncclCollAllToAll:
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
