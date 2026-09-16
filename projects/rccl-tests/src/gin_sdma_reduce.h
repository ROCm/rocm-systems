/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Device-side reduction ops for the GIN-SDMA ReduceScatter (deviceImpl == 3).
//
// These MUST bit-match rccl-tests' correctness oracle (verifiable.cu). The
// verifier's compare tolerance is 0 for sum/prod/min/max on EVERY type (2 for
// floating avg, 1 for fp8_e5m2 sum with >1 rank -- see ncclVerifiableVerify), so
// each op here mirrors the corresponding Reduce* struct in verifiable.cu exactly:
//   * low-precision types (half / bf16 / fp8) convert to float, apply the op,
//     then narrow back to T on EVERY pairwise step (the verifier folds the same
//     way, so accumulating in float and narrowing once would NOT bit-match);
//   * the caller folds contributions in ascending source-rank order -- the same
//     order ncclVerifiablePrepareExpected uses;
//   * avg premultiplies each element by 1/rankN for floating T (postOp identity)
//     but sums-then-divides for integral T (preOp identity), exactly as
//     ReduceAvg_Base<T, integral> does.
//
// mulsum / PreMulSum is intentionally NOT supported (deferred); the dispatch
// macro (SPECIALIZE_REDUCE_KERNEL in common.h) returns nullptr for it, and fp8
// prod is excluded there too.
//
// This header is device-only: include it AFTER common.h (which pulls in rccl.h
// for __half / hip_bfloat16 and rccl_float8.h for the fp8 types) so all element
// types and the HAVE_BF16 / HAVE_FP8 guards are in scope.

#ifndef GIN_SDMA_REDUCE_H_
#define GIN_SDMA_REDUCE_H_

#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)

#include <type_traits>
#if HAVE_BF16
#include <hip/hip_bfloat16.h>  // hip_bfloat16 (same type verifiable.cu folds with)
#endif

namespace gin_sdma_reduce {

// ------------------------------- sum (a + b) -------------------------------
// Template covers all integer types (int8/uint8/int32/uint32/int64/uint64) plus
// float/double; the explicit overloads below are more specialized so they win
// for the low-precision types (matching verifiable.cu::ReduceSum).
template <class T> __device__ __forceinline__ T rsum(T a, T b) { return a + b; }
__device__ __forceinline__ __half rsum(__half a, __half b) {
  return __float2half(__half2float(a) + __half2float(b));
}
#if HAVE_BF16
__device__ __forceinline__ hip_bfloat16 rsum(hip_bfloat16 a, hip_bfloat16 b) {
  return hip_bfloat16(static_cast<float>(a) + static_cast<float>(b));
}
#endif
#if HAVE_FP8
__device__ __forceinline__ rccl_float8 rsum(rccl_float8 a, rccl_float8 b) {
  return rccl_float8(static_cast<float>(a) + static_cast<float>(b));
}
__device__ __forceinline__ rccl_bfloat8 rsum(rccl_bfloat8 a, rccl_bfloat8 b) {
  return rccl_bfloat8(static_cast<float>(a) + static_cast<float>(b));
}
#endif

// ------------------------------- prod (a * b) ------------------------------
template <class T> __device__ __forceinline__ T rprod(T a, T b) { return a * b; }
__device__ __forceinline__ __half rprod(__half a, __half b) {
  return __float2half(__half2float(a) * __half2float(b));
}
#if HAVE_BF16
__device__ __forceinline__ hip_bfloat16 rprod(hip_bfloat16 a, hip_bfloat16 b) {
  return hip_bfloat16(static_cast<float>(a) * static_cast<float>(b));
}
#endif
#if HAVE_FP8
__device__ __forceinline__ rccl_float8 rprod(rccl_float8 a, rccl_float8 b) {
  return static_cast<rccl_float8>(static_cast<float>(a) * static_cast<float>(b));
}
__device__ __forceinline__ rccl_bfloat8 rprod(rccl_bfloat8 a, rccl_bfloat8 b) {
  return static_cast<rccl_bfloat8>(static_cast<float>(a) * static_cast<float>(b));
}
#endif

// ------------------------------- min / max ---------------------------------
template <class T> __device__ __forceinline__ T rmin(T a, T b) { return a < b ? a : b; }
template <class T> __device__ __forceinline__ T rmax(T a, T b) { return a > b ? a : b; }
__device__ __forceinline__ __half rmin(__half a, __half b) {
  return __half2float(a) < __half2float(b) ? a : b;
}
__device__ __forceinline__ __half rmax(__half a, __half b) {
  return __half2float(a) > __half2float(b) ? a : b;
}
#if HAVE_BF16
__device__ __forceinline__ hip_bfloat16 rmin(hip_bfloat16 a, hip_bfloat16 b) {
  return static_cast<float>(a) < static_cast<float>(b) ? a : b;
}
__device__ __forceinline__ hip_bfloat16 rmax(hip_bfloat16 a, hip_bfloat16 b) {
  return static_cast<float>(a) > static_cast<float>(b) ? a : b;
}
#endif
#if HAVE_FP8
__device__ __forceinline__ rccl_float8 rmin(rccl_float8 a, rccl_float8 b) {
  return static_cast<float>(a) < static_cast<float>(b) ? a : b;
}
__device__ __forceinline__ rccl_float8 rmax(rccl_float8 a, rccl_float8 b) {
  return static_cast<float>(a) > static_cast<float>(b) ? a : b;
}
__device__ __forceinline__ rccl_bfloat8 rmin(rccl_bfloat8 a, rccl_bfloat8 b) {
  return static_cast<float>(a) < static_cast<float>(b) ? a : b;
}
__device__ __forceinline__ rccl_bfloat8 rmax(rccl_bfloat8 a, rccl_bfloat8 b) {
  return static_cast<float>(a) > static_cast<float>(b) ? a : b;
}
#endif

// ---------------------------- avg premul factor ----------------------------
// The 1/rankN factor, built in the same intermediate precision the verifier
// uses (ReduceAvg_Base<T,false>: T1 = float when sizeof(T) < sizeof(double),
// else double). Only used on the floating branch; integral avg divides in postOp.
template <class T> __device__ __forceinline__ T avgFactor(int rankN) {
  return (T)(1.0f / (float)rankN);
}
template <> __device__ __forceinline__ double avgFactor<double>(int rankN) {
  return 1.0 / (double)rankN;
}

// ------------------------ op-tagged pre / combine / post --------------------
// `op` is an ncclRedOp_t value (ncclSum=0, ncclProd=1, ncclMax=2, ncclMin=3,
// ncclAvg=4). PreMulSum is never dispatched here.

template <class T> __device__ __forceinline__ T preOp(int op, T x, int rankN) {
  // Floating avg premultiplies each element by 1/rankN before summing; integral
  // avg (and every other op) leaves the element untouched (divides in postOp).
  if (op == ncclAvg && !std::is_integral<T>::value) {
    return rprod(avgFactor<T>(rankN), x);
  }
  return x;
}

template <class T> __device__ __forceinline__ T combine(int op, T a, T b) {
  switch (op) {
    case ncclProd: return rprod(a, b);
    case ncclMax:  return rmax(a, b);
    case ncclMin:  return rmin(a, b);
    default:       return rsum(a, b);  // ncclSum and ncclAvg both accumulate by +
  }
}

// Integral avg finishes with an integer divide of the (wrapped) sum; floating T
// already divided in preOp. Split via SFINAE so `x / rankN` is only instantiated
// for integral T (fp8/bf16/half have no unambiguous `operator/(T,int)`).
template <class T> __device__ __forceinline__
typename std::enable_if<std::is_integral<T>::value, T>::type
postDivAvg(T x, int rankN) { return (T)(x / rankN); }
template <class T> __device__ __forceinline__
typename std::enable_if<!std::is_integral<T>::value, T>::type
postDivAvg(T x, int /*rankN*/) { return x; }

template <class T> __device__ __forceinline__ T postOp(int op, T x, int rankN) {
  if (op == ncclAvg) return postDivAvg<T>(x, rankN);
  return x;
}

}  // namespace gin_sdma_reduce

#endif  // ENABLE_DEVICE_API && >= 2.28
#endif  // GIN_SDMA_REDUCE_H_
