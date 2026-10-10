/*************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef RCCL_UT_DEVICE_DATA_OPS_HPP_
#define RCCL_UT_DEVICE_DATA_OPS_HPP_

// Reusable host/device data-operation layer for the unit-test framework.
//
// Single source of truth for (a) the deterministic per-element test pattern and
// (b) the per-type equality test, shared by BOTH the host path (PtrUnion::Set /
// IsEqual) and the device path (fill / mismatch-reduce kernels). Any collective's
// PrepareData/ValidateResults that use PtrUnion get device fill + device validate
// for free once switched into device-data mode; new tests need no kernel code.
//
// NOTE ON PRECISION: the pattern uses `double` (GPUs have no `long double`). In
// device-data mode BOTH input and expected are generated with this same functor,
// so they are self-consistent and correctness does not depend on reproducing the
// host `long double` bytes. Host mode is unchanged.

#include <hip/hip_runtime.h>
#include <type_traits>
#include "PtrUnion.hpp"   // dtype enum + typedefs (hip_bfloat16, rccl_float8, rccl_bfloat8, __half)

namespace RcclUnitTesting
{
  // ---- shared scalar pattern (host + device) --------------------------------
  __host__ __device__ inline int PatternValueI(bool fp8, int globalRank, size_t idx)
  {
    return fp8 ? (int)(idx % 16) : (int)((globalRank + idx) % 256);
  }
  __host__ __device__ inline double PatternValueF(int valueI)
  {
    return 1.0 / ((double)valueI + 1.0);
  }

  // ---- per-type element construction (mirrors PtrUnion::Set) -----------------
  template <typename T> __host__ __device__ inline T MakeVal(int vi, double vf);
  template <> __host__ __device__ inline int8_t   MakeVal<int8_t>  (int vi, double)  { return (int8_t)vi; }
  template <> __host__ __device__ inline uint8_t  MakeVal<uint8_t> (int vi, double)  { return (uint8_t)vi; }
  template <> __host__ __device__ inline int32_t  MakeVal<int32_t> (int vi, double)  { return (int32_t)vi; }
  template <> __host__ __device__ inline uint32_t MakeVal<uint32_t>(int vi, double)  { return (uint32_t)vi; }
  template <> __host__ __device__ inline int64_t  MakeVal<int64_t> (int vi, double)  { return (int64_t)vi; }
  template <> __host__ __device__ inline uint64_t MakeVal<uint64_t>(int vi, double)  { return (uint64_t)vi; }
  template <> __host__ __device__ inline float    MakeVal<float>   (int, double vf)  { return (float)vf; }
  template <> __host__ __device__ inline double   MakeVal<double>  (int, double vf)  { return vf; }
  template <> __host__ __device__ inline __half        MakeVal<__half>       (int, double vf) { return __float2half((float)vf); }
  template <> __host__ __device__ inline hip_bfloat16  MakeVal<hip_bfloat16> (int, double vf) { return hip_bfloat16((float)vf); }
  template <> __host__ __device__ inline rccl_float8   MakeVal<rccl_float8>  (int, double vf) { return rccl_float8((float)vf); }
  template <> __host__ __device__ inline rccl_bfloat8  MakeVal<rccl_bfloat8> (int, double vf) { return rccl_bfloat8((float)vf); }

  // ---- per-type equality (mirrors PtrUnion::IsEqual tolerances) --------------
  template <typename T> __host__ __device__ inline bool Matches(T a, T b);
  template <> __host__ __device__ inline bool Matches<int8_t>  (int8_t a,  int8_t b)  { return a == b; }
  template <> __host__ __device__ inline bool Matches<uint8_t> (uint8_t a, uint8_t b) { return a == b; }
  template <> __host__ __device__ inline bool Matches<int32_t> (int32_t a, int32_t b) { return a == b; }
  template <> __host__ __device__ inline bool Matches<uint32_t>(uint32_t a,uint32_t b){ return a == b; }
  template <> __host__ __device__ inline bool Matches<int64_t> (int64_t a, int64_t b) { return a == b; }
  template <> __host__ __device__ inline bool Matches<uint64_t>(uint64_t a,uint64_t b){ return a == b; }
  // FP8 compares exactly: its reductions are validated against exactly-reducible data
  // (VerifiableFp8.hpp). Do not use a magnitude-scaled tolerance here: at eight ranks
  // it can accept results with several missing contributions.
  // Non-FP8 tolerances use the SAME double literals as the host IsEqual (PtrUnion.cpp),
  // not float literals: 9e-2/1e-5 aren't exactly representable, so a float-literal bound
  // differs from the host's double bound by ~1e-9 and could flip a verdict at the
  // tolerance boundary. Keeping them identical guarantees host==device verdicts.
  template <> __host__ __device__ inline bool Matches<float>   (float a,  float b)  { return fabs((double)(a - b)) < 1e-5; }
  template <> __host__ __device__ inline bool Matches<double>  (double a, double b)  { return fabs(a - b) < 1e-12; }
  template <> __host__ __device__ inline bool Matches<__half>       (__half a, __half b)             { return fabs((double)(__half2float(a) - __half2float(b))) < 9e-2; }
  template <> __host__ __device__ inline bool Matches<hip_bfloat16> (hip_bfloat16 a, hip_bfloat16 b) { return fabs((double)((float)a - (float)b)) < 9e-2; }
  template <> __host__ __device__ inline bool Matches<rccl_float8>  (rccl_float8 a, rccl_float8 b)   { return (float)a == (float)b; }
  template <> __host__ __device__ inline bool Matches<rccl_bfloat8> (rccl_bfloat8 a, rccl_bfloat8 b) { return (float)a == (float)b; }

  // ---- per-type -> double (for the first-mismatch diagnostic) -----------------
  // Test-pattern values are small (mod 256, reduced over a handful of ranks), so a
  // double captures every dtype's value exactly, including fp8 converted with the
  // correct device (fnuz) type.
  template <typename T> __host__ __device__ inline double ToDoubleVal(T v)          { return (double)v; }
  template <> __host__ __device__ inline double ToDoubleVal<__half>      (__half v)       { return (double)__half2float(v); }
  template <> __host__ __device__ inline double ToDoubleVal<hip_bfloat16>(hip_bfloat16 v) { return (double)(float)v; }
  template <> __host__ __device__ inline double ToDoubleVal<rccl_float8> (rccl_float8 v)  { return (double)(float)v; }
  template <> __host__ __device__ inline double ToDoubleVal<rccl_bfloat8>(rccl_bfloat8 v) { return (double)(float)v; }

  // ---- reduction step (mirrors PtrUnion::Reduce / DivideByInt per-type) -------
  // op encoding matches ncclRedOp_t: 0=sum 1=prod 2=max 3=min (avg handled via sum+DivStep).
  __host__ __device__ inline float DevReduceF(int op, float a, float b)
  {
    switch (op) { case 1: return a * b; case 2: return a > b ? a : b; case 3: return a < b ? a : b; default: return a + b; }
  }
  // Native-type accumulate (int / float / double): reduce directly in T.
  template <typename T> __host__ __device__ inline T AccStep(int op, T a, T b)
  {
    switch (op) { case 1: return (T)(a * b); case 2: return a > b ? a : b; case 3: return a < b ? a : b; default: return (T)(a + b); }
  }
  // Low-precision accumulate: reduce in float then round back to T (matches host).
  template <> __host__ __device__ inline __half       AccStep<__half>      (int op, __half a, __half b)             { return __float2half(DevReduceF(op, __half2float(a), __half2float(b))); }
  template <> __host__ __device__ inline hip_bfloat16 AccStep<hip_bfloat16>(int op, hip_bfloat16 a, hip_bfloat16 b) { return hip_bfloat16(DevReduceF(op, (float)a, (float)b)); }
  template <> __host__ __device__ inline rccl_float8  AccStep<rccl_float8> (int op, rccl_float8 a, rccl_float8 b)   { return rccl_float8(DevReduceF(op, (float)a, (float)b)); }
  template <> __host__ __device__ inline rccl_bfloat8 AccStep<rccl_bfloat8>(int op, rccl_bfloat8 a, rccl_bfloat8 b) { return rccl_bfloat8(DevReduceF(op, (float)a, (float)b)); }

  // Divide-by-int for the average op (mirrors PtrUnion::DivideByInt).
  template <typename T> __host__ __device__ inline T DivStep(T a, int n) { return (T)(a / n); }
  template <> __host__ __device__ inline __half       DivStep<__half>      (__half a, int n)       { return __float2half(__half2float(a) / n); }
  template <> __host__ __device__ inline hip_bfloat16 DivStep<hip_bfloat16>(hip_bfloat16 a, int n) { return hip_bfloat16((float)a / n); }
  template <> __host__ __device__ inline rccl_float8  DivStep<rccl_float8> (rccl_float8 a, int n)  { return rccl_float8((float)a / n); }
  template <> __host__ __device__ inline rccl_bfloat8 DivStep<rccl_bfloat8>(rccl_bfloat8 a, int n) { return rccl_bfloat8((float)a / n); }

  // ---- FP8 kernel element type ------------------------------------------------
  // rccl_float8 / rccl_bfloat8 name different types in the host and device compile passes
  // (OCP on host, fnuz on gfx942 device), so a kernel templated on them gets a different
  // mangled name in each pass and fails to launch ("Cannot find Symbol"). Fp8Bits holds the
  // raw byte and mangles identically in both passes; its operations go through the native
  // type, so they give the collective's encoding only in device code.
  template <bool IsE5m2>
  struct Fp8Bits
  {
    using Native = typename std::conditional<IsE5m2, rccl_bfloat8, rccl_float8>::type;
    uint8_t bits;

    __host__ __device__ static Fp8Bits From(Native v) { return Fp8Bits{*reinterpret_cast<uint8_t const*>(&v)}; }
    __host__ __device__ Native Get() const { return *reinterpret_cast<Native const*>(&bits); }
  };
  using Fp8E4m3Bits = Fp8Bits<false>;
  using Fp8E5m2Bits = Fp8Bits<true>;

#define RCCL_UT_FP8_BITS_OPS(B)                                                                                     \
  template <> __host__ __device__ inline B      MakeVal<B>    (int vi, double vf) { return B::From(MakeVal<B::Native>(vi, vf)); }             \
  template <> __host__ __device__ inline bool   Matches<B>    (B a, B b)          { return Matches<B::Native>(a.Get(), b.Get()); }             \
  template <> __host__ __device__ inline double ToDoubleVal<B>(B v)               { return ToDoubleVal<B::Native>(v.Get()); }                  \
  template <> __host__ __device__ inline B      AccStep<B>    (int op, B a, B b)  { return B::From(AccStep<B::Native>(op, a.Get(), b.Get())); } \
  template <> __host__ __device__ inline B      DivStep<B>    (B a, int n)        { return B::From(DivStep<B::Native>(a.Get(), n)); }
  RCCL_UT_FP8_BITS_OPS(Fp8E4m3Bits)
  RCCL_UT_FP8_BITS_OPS(Fp8E5m2Bits)
#undef RCCL_UT_FP8_BITS_OPS

  // ---- kernels ---------------------------------------------------------------
  template <typename T>
  __global__ void FillKernel(T* p, size_t n, int globalRank, size_t startIdx, bool fp8)
  {
    size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    int vi = PatternValueI(fp8, globalRank, startIdx + j);
    p[j] = MakeVal<T>(vi, PatternValueF(vi));
  }

  template <typename T>
  __global__ void MismatchReduceKernel(const T* a, const T* b, size_t n,
                                       unsigned long long* mismatches,
                                       unsigned long long* firstIdx)
  {
    size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    if (!Matches<T>(a[j], b[j]))
    {
      atomicAdd(mismatches, 1ULL);
      atomicMin(firstIdx, (unsigned long long)j);   // remember earliest divergent index
    }
  }

  // Capture expected/actual at a single index for the diagnostic print. Launched with
  // one thread after the first divergent index is known. Emits BOTH a double view (used
  // for the float dtypes) and the exact raw element bits (used for the integer dtypes so
  // 64-bit values above 2^53 print without double rounding).
  template <typename T>
  __global__ void CaptureElemKernel(const T* actual, const T* expected, size_t idx,
                                    double* outF, unsigned long long* outBits)
  {
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
      outF[0] = ToDoubleVal<T>(expected[idx]);   // [0] = expected
      outF[1] = ToDoubleVal<T>(actual[idx]);     // [1] = actual
      unsigned long long e = 0, a = 0;           // zero-extended exact bits (sizeof(T) <= 8)
      memcpy(&e, &expected[idx], sizeof(T));
      memcpy(&a, &actual[idx],   sizeof(T));
      outBits[0] = e;
      outBits[1] = a;
    }
  }

  // Build the all-ranks reduction of the pattern for element idx, mirroring the host
  // loop: start from rank 0, accumulate ranks 1..totalRanks-1 with `op`, optional avg.
  // out[i] = reduce over ranks of the pattern at index i (AllReduce expected).
  // startIdx offsets the pattern's global element index so a caller can build the reduced
  // expected for a sub-range: out[idx] = reduce over ranks of pattern at (startIdx + idx).
  // AllReduce passes startIdx=0 (full buffer); ReduceScatter passes globalRank*numOutput
  // to build only this rank's scattered slice.
  template <typename T>
  __global__ void ExpectedReduceKernel(T* out, size_t n, int totalRanks, bool fp8, int op, bool isAvg, size_t startIdx)
  {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    size_t gidx = startIdx + idx;
    int vi = PatternValueI(fp8, 0, gidx);
    T acc = MakeVal<T>(vi, PatternValueF(vi));
    for (int r = 1; r < totalRanks; ++r)
    {
      int v = PatternValueI(fp8, r, gidx);
      acc = AccStep<T>(op, acc, MakeVal<T>(v, PatternValueF(v)));
    }
    if (isAvg) acc = DivStep<T>(acc, totalRanks);
    out[idx] = acc;
  }
}
#endif  // RCCL_UT_DEVICE_DATA_OPS_HPP_
