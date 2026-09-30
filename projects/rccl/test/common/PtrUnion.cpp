/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "PtrUnion.hpp"
#include "api_trace.h"
#include "DeviceDataOps.hpp"
namespace RcclUnitTesting
{
  // Threads per block for the device data-op kernels.
  static constexpr size_t kDeviceKernelBlockSize = 256;
  // Default Ut device-data element-count threshold (1Mi) when UT_DEVICE_DATA_MIN_ELEMS is unset.
  static constexpr long long kDefaultDeviceDataMinElems = (1LL << 20);

  // Dispatch a device data-op over the concrete element type for a dtype.
  // ACTION is a statement using template type `T`.
  #define RCCL_UT_DTYPE_DISPATCH(dt, ACTION)                                   \
    switch (dt) {                                                              \
      case ncclInt8:      { using T = int8_t;       ACTION; break; }           \
      case ncclUint8:     { using T = uint8_t;      ACTION; break; }           \
      case ncclInt32:     { using T = int32_t;      ACTION; break; }           \
      case ncclUint32:    { using T = uint32_t;     ACTION; break; }           \
      case ncclInt64:     { using T = int64_t;      ACTION; break; }           \
      case ncclUint64:    { using T = uint64_t;     ACTION; break; }           \
      case ncclFloat16:   { using T = __half;       ACTION; break; }           \
      case ncclFloat32:   { using T = float;        ACTION; break; }           \
      case ncclFloat64:   { using T = double;       ACTION; break; }           \
      case ncclBfloat16:  { using T = hip_bfloat16; ACTION; break; }           \
      /* fp8 is never dispatched here: callers route it to the dedicated byte-based    */ \
      /* kernels (rccl_float8/rccl_bfloat8 alias different types in the host vs device */ \
      /* compile pass, so a templated kernel would be instantiated with a mismatched  */ \
      /* mangling). Fail loudly rather than instantiate the templated form for fp8.   */ \
      case ncclFloat8e4m3:                                                     \
      case ncclFloat8e5m2:                                                     \
        TEST_ERROR("fp8 must use the byte-based kernel path (%d)", dt);        \
        return TEST_FAIL;                                                      \
      default: TEST_ERROR("Unsupported datatype (%d)", dt); return TEST_FAIL;  \
    }

  ErrCode PtrUnion::FillPatternDevice(ncclDataType_t const dataType,
                                      size_t         const numElements,
                                      int            const globalRank,
                                      size_t         const startIdx)
  {
    // Wiring telltale: fire once per process so runs visibly confirm the device
    // data path is actually exercised (guards against a silent host fallback).
    static bool s_fillLogged = false;
    if (!s_fillLogged)
    {
      s_fillLogged = true;
      fprintf(stdout, "[UT][device-data] FillPatternDevice ACTIVE (first call: dtype=%d, n=%lu)\n",
              (int)dataType, (unsigned long)numElements);
      fflush(stdout);
    }
    if (numElements == 0) return TEST_SUCCESS;
    bool const fp8 = (dataType == ncclFloat8e4m3 || dataType == ncclFloat8e5m2);
    size_t const threads = kDeviceKernelBlockSize, blocks = (numElements + threads - 1) / threads;
    if (fp8)
    {
      hipLaunchKernelGGL(FillKernelFp8, dim3(blocks), dim3(threads), 0, 0,
                         (uint8_t*)this->ptr, numElements, globalRank, startIdx,
                         dataType == ncclFloat8e5m2);
    }
    else
    {
      RCCL_UT_DTYPE_DISPATCH(dataType,
        hipLaunchKernelGGL(FillKernel<T>, dim3(blocks), dim3(threads), 0, 0,
                           (T*)this->ptr, numElements, globalRank, startIdx, fp8));
    }
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    return TEST_SUCCESS;
  }

  ErrCode PtrUnion::IsEqualDevice(ncclDataType_t const dataType,
                                  size_t         const numElements,
                                  void*          const actualGpu,
                                  void*          const expectedGpu,
                                  size_t&              mismatches)
  {
    // Wiring telltale: fire once per process so runs visibly confirm the device
    // validate path is actually exercised (guards against a silent host fallback).
    static bool s_cmpLogged = false;
    if (!s_cmpLogged)
    {
      s_cmpLogged = true;
      fprintf(stdout, "[UT][device-data] IsEqualDevice ACTIVE (first call: dtype=%d, n=%lu)\n",
              (int)dataType, (unsigned long)numElements);
      fflush(stdout);
    }
    mismatches = 0;
    if (numElements == 0) return TEST_SUCCESS;

    // Scope-guarded device frees: every early return below (CHECK_HIP failures and the
    // dispatch default) runs these destructors, so nothing leaks on the error path.
    struct DevFree
    {
      void* p = nullptr;
      ~DevFree()
      {
        if (p) { (void)hipFree(p); }
      }
    } gScratch, gVals, gBits;

    // dScratch[0] = mismatch count, dScratch[1] = first (lowest) divergent index.
    unsigned long long* dScratch = nullptr;
    double*             dVals    = nullptr;   // [expected, actual] float view at first divergent index
    unsigned long long* dBits    = nullptr;   // [expected, actual] exact raw bits (integer dtypes)
    CHECK_HIP(hipMalloc(&dScratch, 2 * sizeof(unsigned long long))); gScratch.p = dScratch;
    CHECK_HIP(hipMalloc(&dVals,    2 * sizeof(double)));             gVals.p    = dVals;
    CHECK_HIP(hipMalloc(&dBits,    2 * sizeof(unsigned long long))); gBits.p    = dBits;
    unsigned long long hInit[2] = { 0ULL, (unsigned long long)numElements };  // idx init = n (= "none")
    CHECK_HIP(hipMemcpy(dScratch, hInit, sizeof(hInit), hipMemcpyHostToDevice));

    bool const fp8    = (dataType == ncclFloat8e4m3 || dataType == ncclFloat8e5m2);
    bool const isE5m2 = (dataType == ncclFloat8e5m2);
    size_t const threads = kDeviceKernelBlockSize, blocks = (numElements + threads - 1) / threads;
    if (fp8)
    {
      hipLaunchKernelGGL(MismatchReduceFp8, dim3(blocks), dim3(threads), 0, 0,
                         (const uint8_t*)actualGpu, (const uint8_t*)expectedGpu, numElements,
                         dScratch, dScratch + 1, isE5m2);
    }
    else
    {
      RCCL_UT_DTYPE_DISPATCH(dataType,
        hipLaunchKernelGGL(MismatchReduceKernel<T>, dim3(blocks), dim3(threads), 0, 0,
                           (const T*)actualGpu, (const T*)expectedGpu, numElements,
                           dScratch, dScratch + 1));
    }
    CHECK_HIP(hipGetLastError());
    unsigned long long hOut[2] = { 0ULL, 0ULL };
    CHECK_HIP(hipMemcpy(hOut, dScratch, sizeof(hOut), hipMemcpyDeviceToHost));
    mismatches = (size_t)hOut[0];

    // Diagnostic: with the host buffering path retired this is the only value dump, so
    // report the first divergent index with its expected/actual (dtype-aware). The test
    // pattern's values are small, so a double captures every dtype exactly.
    if (hOut[0] != 0 && hOut[1] < (unsigned long long)numElements)
    {
      size_t const fi = (size_t)hOut[1];
      if (fp8)
      {
        hipLaunchKernelGGL(CaptureElemFp8, dim3(1), dim3(1), 0, 0,
                           (const uint8_t*)actualGpu, (const uint8_t*)expectedGpu, fi, dVals, isE5m2);
      }
      else
      {
        RCCL_UT_DTYPE_DISPATCH(dataType,
          hipLaunchKernelGGL(CaptureElemKernel<T>, dim3(1), dim3(1), 0, 0,
                             (const T*)actualGpu, (const T*)expectedGpu, fi, dVals, dBits));
      }
      CHECK_HIP(hipGetLastError());
      double             hVals[2] = { 0.0, 0.0 };    // float view [expected, actual]
      unsigned long long hBits[2] = { 0ULL, 0ULL };  // exact raw bits [expected, actual]
      CHECK_HIP(hipMemcpy(hVals, dVals, sizeof(hVals), hipMemcpyDeviceToHost));
      CHECK_HIP(hipMemcpy(hBits, dBits, sizeof(hBits), hipMemcpyDeviceToHost));
      // Mirror the host IsEqual verbose format exactly, per dtype: integers print from the
      // exact bits (no double rounding), floats from the double view. fp8's dBits are unused
      // (it fills only dVals via CaptureElemFp8) and prints through the float default.
      switch (dataType)
      {
      case ncclInt8:
        TEST_ERROR("Expected output: %d.  Actual output: %d at index %zu",
                   (int)(int8_t)hBits[0], (int)(int8_t)hBits[1], fi); break;
      case ncclUint8:
        TEST_ERROR("Expected output: %u.  Actual output: %u at index %zu",
                   (unsigned)(uint8_t)hBits[0], (unsigned)(uint8_t)hBits[1], fi); break;
      case ncclInt32:
        TEST_ERROR("Expected output: %d.  Actual output: %d at index %zu",
                   (int32_t)hBits[0], (int32_t)hBits[1], fi); break;
      case ncclUint32:
        TEST_ERROR("Expected output: %u.  Actual output: %u at index %zu",
                   (uint32_t)hBits[0], (uint32_t)hBits[1], fi); break;
      case ncclInt64:
        TEST_ERROR("Expected output: %lld.  Actual output: %lld at index %zu",
                   (long long)(int64_t)hBits[0], (long long)(int64_t)hBits[1], fi); break;
      case ncclUint64:
        TEST_ERROR("Expected output: %llu.  Actual output: %llu at index %zu",
                   (unsigned long long)hBits[0], (unsigned long long)hBits[1], fi); break;
      default:  // floating-point dtypes (fp16/fp32/fp64/bf16/fp8) — exact via double
        TEST_ERROR("Expected output: %f.  Actual output: %f at index %zu",
                   hVals[0], hVals[1], fi); break;
      }
    }
    return TEST_SUCCESS;
  }

  ErrCode PtrUnion::FillReducedPatternDevice(ncclDataType_t const dataType,
                                             size_t         const numElements,
                                             int            const totalRanks,
                                             ncclRedOp_t    const op,
                                             size_t         const startIdx)
  {
    static bool s_redLogged = false;
    if (!s_redLogged)
    {
      s_redLogged = true;
      fprintf(stdout, "[UT][device-data] FillReducedPatternDevice ACTIVE (first call: dtype=%d, ranks=%d, op=%d)\n",
              (int)dataType, totalRanks, (int)op);
      fflush(stdout);
    }
    if (numElements == 0) return TEST_SUCCESS;
    bool const fp8    = (dataType == ncclFloat8e4m3 || dataType == ncclFloat8e5m2);
    bool const isAvg  = (op == ncclAvg);
    int  const tempOp = (op >= ncclAvg ? (int)ncclSum : (int)op);  // avg/custom reduce as sum
    size_t const threads = kDeviceKernelBlockSize, blocks = (numElements + threads - 1) / threads;
    if (fp8)
    {
      hipLaunchKernelGGL(ExpectedReduceFp8, dim3(blocks), dim3(threads), 0, 0,
                         (uint8_t*)this->ptr, numElements, totalRanks, tempOp, isAvg,
                         dataType == ncclFloat8e5m2, startIdx);
    }
    else
    {
      RCCL_UT_DTYPE_DISPATCH(dataType,
        hipLaunchKernelGGL(ExpectedReduceKernel<T>, dim3(blocks), dim3(threads), 0, 0,
                           (T*)this->ptr, numElements, totalRanks, fp8, tempOp, isAvg, startIdx));
    }
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    return TEST_SUCCESS;
  }

  size_t DataTypeToBytes(ncclDataType_t const dataType)
  {
    switch (dataType)
    {
    case ncclInt8:   return 1;
    case ncclUint8:  return 1;
    case ncclFloat8e4m3:return 1;
    case ncclFloat8e5m2:return 1;
    case ncclInt32:  return 4;
    case ncclUint32: return 4;
    case ncclInt64:  return 8;
    case ncclUint64: return 8;
    case ncclFloat16: return 2;
    case ncclFloat32: return 4;
    case ncclFloat64: return 8;
    case ncclBfloat16: return 2;
    default:
      TEST_ERROR("Unsupported datatype (%d)", dataType);
      exit(0);
    }
  }

  ErrCode PtrUnion::Attach(void *ptr)
  {
    this->ptr = ptr;
    return TEST_SUCCESS;
  }

  ErrCode PtrUnion::Attach(PtrUnion ptrUnion)
  {
    this->ptr = ptrUnion.ptr;
    return TEST_SUCCESS;
  }

  ErrCode PtrUnion::AllocateGpuMem(size_t const numBytes, bool const useManagedMem, bool const userRegistered)
  {
    if (numBytes)
    {
      if (userRegistered)
      {
        if (ncclMemAlloc((void**)&I1, numBytes) != ncclSuccess)
        {
          TEST_ERROR("Unable to allocate user managed GPU memory (%lu bytes)", numBytes);
          return TEST_FAIL;
        }
      }
      else
      {
        if (useManagedMem)
        {
          CHECK_HIP(hipMallocManaged(&I1, numBytes));
        }
        else
        {
          CHECK_HIP(hipMalloc(&I1, numBytes));
        }
      }

    }
    return TEST_SUCCESS;
  }

  ErrCode PtrUnion::AllocateCpuMem(size_t const numBytes)
  {
    if (numBytes)
    {
      this->ptr = calloc(numBytes, 1);
      if (!ptr)
      {
        TEST_ERROR("Unable to allocate memory (%lu bytes)", numBytes);
        return TEST_FAIL;
      }
    }
    return TEST_SUCCESS;
  }

  ErrCode PtrUnion::FreeGpuMem(bool const userRegistered)
  {
    if (this->ptr != nullptr)
    {
      if (userRegistered)
        CHECK_NCCL(ncclMemFree(this->ptr));
      else
        CHECK_HIP(hipFree(this->ptr));
      this->ptr = nullptr;
    }
    return TEST_SUCCESS;
  }

  ErrCode PtrUnion::FreeCpuMem()
  {
    if (this->ptr != nullptr)
    {
      free(this->ptr);
      this->ptr = nullptr;
    }
    return TEST_SUCCESS;
  }

  ErrCode PtrUnion::ClearGpuMem(size_t const numBytes)
  {
    CHECK_HIP(hipMemset(this->ptr, 0, numBytes));
    CHECK_HIP(hipStreamSynchronize(NULL));
    return TEST_SUCCESS;
  }

  ErrCode PtrUnion::ClearCpuMem(size_t const numBytes)
  {
    memset(this->ptr, 0, numBytes);
    return TEST_SUCCESS;
  }

  // Device-data mode (ON by default; UT_DEVICE_DATA=0 to disable). Per-collective prep
  // funcs consult this to build input/expected on the GPU and validate device-side, but
  // ONLY where that yields a measured speedup. Cached once for the process.
  bool UtDeviceDataEnabled()
  {
    // ON by default; set UT_DEVICE_DATA=0 to force the host reference path.
    static int cached = -1;
    if (cached < 0)
    {
      char const* e = getenv("UT_DEVICE_DATA");
      cached = (e && e[0] == '0') ? 0 : 1;
    }
    return cached == 1;
  }

  bool UtDeviceDtypeSupported(ncclDataType_t const /*dataType*/)
  {
    // All host-supported dtypes now have device kernels (fp8 via dedicated byte-based
    // kernels that dodge the host/device typedef-mangling mismatch). Kept as a hook.
    return true;
  }

  size_t UtDeviceDataMinElements()
  {
    static long long cached = -1;
    if (cached < 0)
    {
      char const* e = getenv("UT_DEVICE_DATA_MIN_ELEMS");
      long long v = e ? atoll(e) : 0;
      cached = (v > 0) ? v : kDefaultDeviceDataMinElems;
    }
    return (size_t)cached;
  }

  ErrCode PtrUnion::FillPattern(ncclDataType_t const dataType,
                                size_t         const numElements,
                                int            const globalRank,
                                bool           const isGpuMem)
  {
    // NOTE: device-data mode is opt-in PER COLLECTIVE (a prep func calls
    // FillPatternDevice / builds expectedGpu directly) so that only collectives with a
    // measured speedup change behavior. FillPattern itself stays on the host path.
    PtrUnion temp;
    size_t const numBytes = numElements * DataTypeToBytes(dataType);

    // If this is GPU memory, create a CPU temp buffer otherwise fill CPU memory directly
    if (isGpuMem)
      temp.AllocateCpuMem(numBytes);
    else
      temp.Attach(this->ptr);

    for (int i = 0; i < numElements; i++)
    {
      // Due to floating-point math not being commutative, the ordering in which ranks are added will matter.
      // For lower-precision data types, we initialize all ranks to the same value to avoid this
      int    valueI = (dataType == ncclFloat8e4m3 || dataType == ncclFloat8e5m2)? (i % 16) :(globalRank + i) % 256;
      double valueF = 1.0L/((double)valueI+1.0L);
      temp.Set(dataType, i, valueI, valueF);
    }

    // If this is GPU memory, copy from CPU temp buffer
    if (isGpuMem)
    {
      CHECK_HIP(hipMemcpy(this->ptr, temp.ptr, numBytes, hipMemcpyHostToDevice));
      temp.FreeCpuMem();
    }

    return TEST_SUCCESS;
  }

  ErrCode PtrUnion::Set(ncclDataType_t const dataType, int const idx, int valueI, double valueF)
  {
    switch (dataType)
    {
    case ncclInt8:     I1[idx] = valueI; break;
    case ncclUint8:    U1[idx] = valueI; break;
    case ncclInt32:    I4[idx] = valueI; break;
    case ncclUint32:   U4[idx] = valueI; break;
    case ncclInt64:    I8[idx] = valueI; break;
    case ncclUint64:   U8[idx] = valueI; break;
    case ncclFloat8e4m3:  F1[idx] = rccl_float8(valueF); break;
    case ncclFloat16:  F2[idx] = __float2half(static_cast<float>(valueF)); break;
    case ncclFloat32:  F4[idx] = valueF; break;
    case ncclFloat64:  F8[idx] = valueF; break;
    case ncclFloat8e5m2:  B1[idx] = rccl_bfloat8(valueF); break;
    case ncclBfloat16: B2[idx] = hip_bfloat16(static_cast<float>(valueF)); break;
    default:
      TEST_ERROR("Unsupported datatype");
      return TEST_FAIL;
    }
    return TEST_SUCCESS;
  }

  ErrCode PtrUnion::Get(ncclDataType_t const dataType, int const idx, int& valueI, double& valueF) const
  {
    switch (dataType)
    {
    case ncclInt8:     valueI = I1[idx]; break;
    case ncclUint8:    valueI = I1[idx]; break;
    case ncclInt32:    valueI = I4[idx]; break;
    case ncclUint32:   valueI = U4[idx]; break;
    case ncclInt64:    valueI = I8[idx]; break;
    case ncclUint64:   valueI = U8[idx]; break;
    case ncclFloat8e4m3:  valueF = float(F1[idx]); break;
    case ncclFloat16:  valueF = __half2float(F2[idx]); break;
    case ncclFloat32:  valueF = F4[idx]; break;
    case ncclFloat64:  valueF = F8[idx]; break;
    case ncclFloat8e5m2:  valueF = float(B1[idx]); break;
    case ncclBfloat16: valueF = B2[idx]; break;
    default:
      TEST_ERROR("Unsupported datatype");
      return TEST_FAIL;
    }
    return TEST_SUCCESS;
  }

  // Multiplies in-place each element by scalarsPerRank[rank]
  ErrCode PtrUnion::Scale(ncclDataType_t const  dataType,
                          size_t         const  numElements,
                          PtrUnion       const& scalarsPerRank,
                          int            const  rank)
  {
    // If no scalars are provided do nothing
    if (scalarsPerRank.ptr == nullptr) return TEST_SUCCESS;

    for (size_t idx = 0; idx < numElements; ++idx)
    {
      switch (dataType)
      {
      case ncclInt8:     I1[idx] *= scalarsPerRank.I1[rank]; break;
      case ncclUint8:    U1[idx] *= scalarsPerRank.U1[rank]; break;
      case ncclInt32:    I4[idx] *= scalarsPerRank.I4[rank]; break;
      case ncclUint32:   U4[idx] *= scalarsPerRank.U4[rank]; break;
      case ncclInt64:    I8[idx] *= scalarsPerRank.I8[rank]; break;
      case ncclUint64:   U8[idx] *= scalarsPerRank.U8[rank]; break;
      case ncclFloat8e4m3:  F1[idx]  = rccl_float8((float)F1[idx] * (float)scalarsPerRank.F1[rank]); break;
      case ncclFloat16:  F2[idx]  = __float2half(__half2float(F2[idx]) * __half2float(scalarsPerRank.F2[rank])); break;
      case ncclFloat32:  F4[idx] *= scalarsPerRank.F4[rank]; break;
      case ncclFloat64:  F8[idx] *= scalarsPerRank.F8[rank]; break;
      case ncclFloat8e5m2:  B1[idx]  = rccl_bfloat8((float)B1[idx] * (float)scalarsPerRank.B1[rank]); break;
      case ncclBfloat16: B2[idx] *= scalarsPerRank.B2[rank]; break;
      default:
        TEST_ERROR("Unsupported datatype");
        return TEST_FAIL;
      }
    }
    return TEST_SUCCESS;
  }

  ErrCode PtrUnion::Reduce(ncclDataType_t const  dataType,
                           size_t         const  numElements,
                           PtrUnion       const& inputCpu,
                           ncclRedOp_t    const  op)
  {
    if (inputCpu.ptr == nullptr)
    {
      TEST_ERROR("Input pointer to Reduce should not be nullptr");
      return TEST_FAIL;
    }

    for (size_t idx = 0; idx < numElements; ++idx)
    {
      switch (dataType)
      {
      case ncclInt8:     I1[idx] = ReduceOp(op, I1[idx], inputCpu.I1[idx]); break;
      case ncclUint8:    U1[idx] = ReduceOp(op, U1[idx], inputCpu.U1[idx]); break;
      case ncclInt32:    I4[idx] = ReduceOp(op, I4[idx], inputCpu.I4[idx]); break;
      case ncclUint32:   U4[idx] = ReduceOp(op, U4[idx], inputCpu.U4[idx]); break;
      case ncclInt64:    I8[idx] = ReduceOp(op, I8[idx], inputCpu.I8[idx]); break;
      case ncclUint64:   U8[idx] = ReduceOp(op, U8[idx], inputCpu.U8[idx]); break;
      case ncclFloat8e4m3:  F1[idx] = rccl_float8(ReduceOp(op, float(F1[idx]), float(inputCpu.F1[idx]))); break;
      case ncclFloat16:  F2[idx] = __float2half(ReduceOp(op, __half2float(F2[idx]), __half2float(inputCpu.F2[idx]))); break;
      case ncclFloat32:  F4[idx] = ReduceOp(op, F4[idx], inputCpu.F4[idx]); break;
      case ncclFloat64:  F8[idx] = ReduceOp(op, F8[idx], inputCpu.F8[idx]); break;
      case ncclFloat8e5m2:  B1[idx] = rccl_bfloat8(ReduceOp(op, float(B1[idx]), float(inputCpu.B1[idx]))); break;
      case ncclBfloat16: B2[idx] = hip_bfloat16(ReduceOp(op, float(B2[idx]), float(inputCpu.B2[idx]))); break;
      default:
        TEST_ERROR("Unsupported datatype");
        return TEST_FAIL;
      }
    }
    return TEST_SUCCESS;
  }


  ErrCode PtrUnion::DivideByInt(ncclDataType_t const dataType,
                                size_t         const numElements,
                                int            const divisor)
  {
    for (size_t idx = 0; idx < numElements; ++idx)
    {
      switch (dataType)
      {
      case ncclInt8:     I1[idx] /= divisor; break;
      case ncclUint8:    U1[idx] /= divisor; break;
      case ncclInt32:    I4[idx] /= divisor; break;
      case ncclUint32:   U4[idx] /= divisor; break;
      case ncclInt64:    I8[idx] /= divisor; break;
      case ncclUint64:   U8[idx] /= divisor; break;
      case ncclFloat8e4m3:  F1[idx] = (rccl_float8((float)(F1[idx]) / divisor)); break;
      case ncclFloat16:  F2[idx] = __float2half(__half2float(F2[idx])/divisor); break;
      case ncclFloat32:  F4[idx] /= divisor; break;
      case ncclFloat64:  F8[idx] /= divisor; break;
      case ncclFloat8e5m2:  B1[idx] = (rccl_bfloat8((float)(B1[idx]) / divisor)); break;
      case ncclBfloat16: B2[idx] = (hip_bfloat16((float)(B2[idx]) / divisor)); break;
      default:
        TEST_ERROR("Unsupported datatype");
        return TEST_FAIL;
      }
    }
    return TEST_SUCCESS;
  }

  ErrCode PtrUnion::IsEqual(ncclDataType_t const  dataType,
                            size_t         const  numElements,
                            PtrUnion       const& expected,
                            bool           const  verbose,
                            bool&                 isMatch)
  {
    isMatch = true;
    size_t idx = 0;
    for (idx = 0; idx < numElements; ++idx)
    {
      switch (dataType)
      {
      case ncclInt8:    isMatch = (I1[idx] == expected.I1[idx]); break;
      case ncclUint8:   isMatch = (U1[idx] == expected.U1[idx]); break;
      case ncclInt32:   isMatch = (I4[idx] == expected.I4[idx]); break;
      case ncclUint32:  isMatch = (U4[idx] == expected.U4[idx]); break;
      case ncclInt64:   isMatch = (I8[idx] == expected.I8[idx]); break;
      case ncclUint64:  isMatch = (U8[idx] == expected.U8[idx]); break;
      case ncclFloat8e4m3: isMatch = (fabs(float(F1[idx]) - float(expected.F1[idx])) < 9e-2); break;
      case ncclFloat16: isMatch = (fabs(__half2float(F2[idx]) - __half2float(expected.F2[idx])) < 9e-2); break;
      case ncclFloat32: isMatch = (fabs(F4[idx] - expected.F4[idx]) < 1e-5); break;
      case ncclFloat64: isMatch = (fabs(F8[idx] - expected.F8[idx]) < 1e-12); break;
      case ncclFloat8e5m2: isMatch = (fabs(float(B1[idx]) - float(expected.B1[idx])) < 9e-2); break;
      case ncclBfloat16: isMatch = (fabs((float)B2[idx] - (float)expected.B2[idx]) < 9e-2); break;
      default:
        TEST_ERROR("Unsupported datatype");
        return TEST_FAIL;
      }
      if (!isMatch) break;
    }

    if (verbose && !isMatch)
    {
      switch (dataType)
      {
      case ncclInt8:
        TEST_ERROR("Expected output: %d.  Actual output: %d at index %lu", expected.I1[idx], I1[idx], idx); break;
      case ncclUint8:
        TEST_ERROR("Expected output: %u.  Actual output: %u at index %lu", expected.U1[idx], U1[idx], idx); break;
      case ncclInt32:
        TEST_ERROR("Expected output: %d.  Actual output: %d at index %lu", expected.I4[idx], I4[idx], idx); break;
      case ncclUint32:
        TEST_ERROR("Expected output: %u.  Actual output: %u at index %lu", expected.U4[idx], U4[idx], idx); break;
      case ncclInt64:
        TEST_ERROR("Expected output: %ld.  Actual output: %ld at index %lu", expected.I8[idx], I8[idx], idx); break;
      case ncclUint64:
        TEST_ERROR("Expected output: %lu.  Actual output: %lu at index %lu", expected.U8[idx], U8[idx], idx); break;
      case ncclFloat8e4m3:
        TEST_ERROR("Expected output: %f.  Actual output: %f at index %lu", (float)expected.F1[idx], (float)F1[idx], idx); break;
      case ncclFloat16:
        TEST_ERROR("Expected output: %f.  Actual output: %f at index %lu", __half2float(expected.F2[idx]), __half2float(F2[idx]), idx); break;
      case ncclFloat32:
        TEST_ERROR("Expected output: %f.  Actual output: %f at index %lu", expected.F4[idx], F4[idx], idx); break;
      case ncclFloat64:
        TEST_ERROR("Expected output: %lf.  Actual output: %lf at index %lu", expected.F8[idx], F8[idx], idx); break;
      case ncclFloat8e5m2:
        TEST_ERROR("Expected output: %f.  Actual output: %f at index %lu", (float)expected.B1[idx], (float)B1[idx], idx); break;
      case ncclBfloat16:
        TEST_ERROR("Expected output: %f.  Actual output: %f at index %lu", (float)expected.B2[idx], (float)B2[idx], idx); break;
      default:
        break;
      }
    }
    return TEST_SUCCESS;
  }

  std::string PtrUnion::ToString(ncclDataType_t const  dataType,
                                 size_t         const  numElements) const
  {
    std::stringstream ss;
    for (int i = 0; i < numElements; i++)
    {
      if (i) ss <<  " ";
      switch (dataType)
      {
      case ncclInt8:     ss << I1[i]; break;
      case ncclUint8:    ss << U1[i]; break;
      case ncclInt32:    ss << I4[i]; break;
      case ncclUint32:   ss << U4[i]; break;
      case ncclInt64:    ss << I8[i]; break;
      case ncclUint64:   ss << U8[i]; break;
      case ncclFloat8e4m3:  ss << (float)F1[i]; break;
      case ncclFloat16:  ss << __half2float(F2[i]); break;
      case ncclFloat32:  ss << F4[i]; break;
      case ncclFloat64:  ss << F8[i]; break;
      case ncclFloat8e5m2:  ss << (float)B1[i]; break;
      case ncclBfloat16: ss << (float)B2[i]; break;
      default: break;
      }
    }
    return ss.str();
  }
}
