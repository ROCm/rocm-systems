/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once
#include "ErrCode.hpp"
#include "rccl/rccl.h"
#include "rccl_float8.h"
#include <hip/hip_bfloat16.h>
#include "hip/hip_fp16.h"

namespace RcclUnitTesting
{
  // Performs the various basic reduction operations
  template <typename T>
  T ReduceOp(ncclRedOp_t const op, T const A, T const B)
  {
    switch (op)
    {
    case ncclSum:  return A + B;
    case ncclProd: return A * B;
    case ncclMax:  return std::max(A, B);
    case ncclMin:  return std::min(A, B);
    default:
      ERROR("Unsupported reduction operator (%d)\n", op);
      exit(0);
    }
  }

  size_t DataTypeToBytes(ncclDataType_t const dataType);

  // PtrUnion encapsulates a pointer of all the different supported datatypes
  // NOTE: Currently half-precision float tests are unsupported due to half
  //       being supported on GPU only and not host
  union PtrUnion
  {
    void*          ptr;
    int8_t*        I1; // ncclInt8
    uint8_t*       U1; // ncclUint8
    int32_t*       I4; // ncclInt32
    uint32_t*      U4; // ncclUint32
    int64_t*       I8; // ncclInt64
    uint64_t*      U8; // ncclUint64
    __half*        F2; // ncclFloat16
    rccl_float8*   F1; // ncclFloat8e4m3
    float*         F4; // ncclFloat32
    double*        F8; // ncclFloat64
    rccl_bfloat8*  B1; // ncclFloat8e5m2
    hip_bfloat16*  B2; // ncclBfloat16

    constexpr PtrUnion() : ptr(nullptr) {}

    ErrCode Attach(void *ptr);
    ErrCode Attach(PtrUnion ptrUnion);

    ErrCode AllocateGpuMem(size_t const numBytes, bool const useManagedMem = false, bool const userRegistered = false);
    ErrCode AllocateCpuMem(size_t const numBytes);

    ErrCode FreeGpuMem(bool const userRegistered = false);
    ErrCode FreeCpuMem();

    ErrCode ClearGpuMem(size_t const numBytes);
    ErrCode ClearCpuMem(size_t const numBytes);

    ErrCode FillPattern(ncclDataType_t const dataType,
                        size_t         const numElements,
                        int            const globalRank,
                        bool           const isGpuMem);

    ErrCode Set(ncclDataType_t const dataType, int const idx, int valueI, double valueF);
    ErrCode Get(ncclDataType_t const dataType, int const idx, int& valueI, double& valueF) const;

    // Multiplies in-place each element by scalarsPerRank[rank]
    ErrCode Scale(ncclDataType_t const  dataType,
                  size_t         const  numElements,
                  PtrUnion       const& scalarsPerRank,
                  int            const  rank);

    // Reduces input into this PtrUnion
    ErrCode Reduce(ncclDataType_t const  dataType,
                   size_t         const  numElements,
                   PtrUnion       const& inputCpu,
                   ncclRedOp_t    const  op);

    // Divide each element by a integer value
    ErrCode DivideByInt(ncclDataType_t const dataType,
                        size_t         const numElements,
                        int            const divisor);

    // Compares for equality (fuzzy comparision for floating point types)
    ErrCode IsEqual(ncclDataType_t const  dataType,
                    size_t         const  numElements,
                    PtrUnion       const& expected,
                    bool           const  verbose,
                    bool&                 isMatch);

    // Output to string (for debug)
    std::string ToString(ncclDataType_t const  dataType,
                         size_t         const  numElements) const;
  };
}
