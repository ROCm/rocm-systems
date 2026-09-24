/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once
#include "ErrCode.hpp"
#include "rccl/rccl.h"
#include "rccl_float8.h"
#if ROCM_VERSION >= 60000
  // hip_bf16.h should be used from ROCm 6.0
  #include <hip/hip_bf16.h>
  typedef __hip_bfloat16 hip_bfloat16;
#else
  #include <hip/hip_bfloat16.h>
#endif
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
      TEST_ERROR("Unsupported reduction operator (%d)", op);
      exit(0);
    }
  }

  size_t DataTypeToBytes(ncclDataType_t const dataType);

  // Device-data mode (ON by default; set UT_DEVICE_DATA=0 to disable): when enabled,
  // the unit-test data layer builds input/expected and validates on the GPU instead
  // of the host. Cached once. Disabling restores the host reference path byte-for-byte.
  bool UtDeviceDataEnabled();

  // Whether the device-data kernels support this dtype. All host-supported dtypes are
  // covered (fp8 via dedicated byte-based kernels). Kept as a hook for future gating.
  bool UtDeviceDtypeSupported(ncclDataType_t const dataType);

  // Minimum element count for the device-data path to engage. Below this, host prep is
  // cheap relative to per-sub-case overhead (fork/comm-init/launch), so the device path
  // gives no speedup and stays off. Default 1Mi; override with UT_DEVICE_DATA_MIN_ELEMS.
  size_t UtDeviceDataMinElements();

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

    // Device data-op layer (reusable by any collective/test). Fills this (device)
    // buffer with the shared pattern via a kernel; the pattern at position j uses
    // global index (startIdx + j). IsEqualDevice compares two device buffers with
    // the same per-type tolerance as IsEqual, returns the mismatch count, and on a
    // mismatch logs the first divergent index with its expected/actual value.
    ErrCode FillPatternDevice(ncclDataType_t const dataType,
                              size_t         const numElements,
                              int            const globalRank,
                              size_t         const startIdx = 0);

    static ErrCode IsEqualDevice(ncclDataType_t const dataType,
                                 size_t         const numElements,
                                 void*          const actualGpu,
                                 void*          const expectedGpu,
                                 size_t&              mismatches);

    // Device-build the all-ranks reduction of the pattern into this (device) buffer,
    // mirroring PtrUnion::Reduce + DivideByInt. Used for AllReduce's expected in
    // device-data mode. Handles ncclSum/Prod/Max/Min/Avg (no scalar/bias/const).
    // startIdx offsets the pattern's global element index (0 for AllReduce's full
    // buffer; globalRank*numOutput for ReduceScatter's per-rank scattered slice).
    ErrCode FillReducedPatternDevice(ncclDataType_t const dataType,
                                     size_t         const numElements,
                                     int            const totalRanks,
                                     ncclRedOp_t    const op,
                                     size_t         const startIdx = 0);

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
