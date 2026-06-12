/*************************************************************************
 * Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Opt-in, compile-time-gated device-side NaN/Inf detector for RCCL reductions.
//
// Enable with `-DRCCL_ENABLE_NAN_CHECK=ON` (see top-level CMakeLists.txt and
// install.sh `--enable-nan-check`). When the macro RCCL_ENABLE_NAN_CHECK is NOT
// defined, every helper/macro here expands to nothing, so there is truly zero
// overhead in a normal build.
//
// The detector inspects the operands/results that flow through the reduction
// combine point in common_kernel.h (BytePack<N> packs). For floating-point
// element types it tests each lane with isnan()/isinf(); on a hit it bumps a
// global device counter and emits a single, rate-limited device printf with
// enough context (INPUT vs OUTPUT, the offending value, the element offset, and
// blockIdx/threadIdx so the rank+channel can be inferred host-side).
//
// Non-floating-point element types are a no-op (NaN/Inf only exist for floats).

#ifndef RCCL_NAN_CHECK_H_
#define RCCL_NAN_CHECK_H_

#ifdef RCCL_ENABLE_NAN_CHECK

#include "op128.h"
#include "reduce_kernel.h"   // for IsFloatingPoint<T>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "rccl_float8.h"

// What is being inspected at the reduction combine point.
enum rcclNanCheckKind {
  RCCL_NANCHECK_INPUT  = 0,   // an operand loaded into the reduction
  RCCL_NANCHECK_OUTPUT = 1    // the reduced result about to be stored
};

// Global device counters. The first counts every detected element (so the host
// can tell whether anything was seen at all). The second gates the printf so a
// pathological run cannot flood stdout.
//
// These are file-local (`static __device__`) so the header can be included by
// every device translation unit without causing duplicate-symbol errors at
// device-link time. Rate-limiting is therefore per-translation-unit, which is
// fine for a debugging aid.
static __device__ unsigned long long rcclNanCheckHitCount   = 0;
static __device__ unsigned int       rcclNanCheckPrintCount = 0;

// Maximum number of device printf lines emitted across the whole kernel launch.
#ifndef RCCL_NANCHECK_MAX_PRINTS
#define RCCL_NANCHECK_MAX_PRINTS 32
#endif

// Scalar test for one floating-point element. Promote sub-fp32 types to float so
// isnan/isinf are well-defined for all supported element types.
template<typename T>
__device__ __forceinline__ bool rcclNanCheckScalar(T v) {
  float f = (float)v;
  return isnan(f) || isinf(f);
}
template<>
__device__ __forceinline__ bool rcclNanCheckScalar<float>(float v) {
  return isnan(v) || isinf(v);
}
template<>
__device__ __forceinline__ bool rcclNanCheckScalar<double>(double v) {
  return isnan(v) || isinf(v);
}

// Report one offending element. Rate-limited and (optionally) aborts.
// MUST be noinline: it carries a device printf (varargs). If inlined into the
// heavily-unrolled hot reduce path at every check site (x lanes x Unroll), it
// makes amdclang -O3 compilation explode (single-kernel compiles ran >20 min).
// Out-of-line, each check site is just a cheap isnan branch + a call.
template<typename T>
__device__ __attribute__((noinline)) void rcclNanCheckReport(
    T v, int kind, long long eltOffset) {
  atomicAdd(&rcclNanCheckHitCount, 1ull);
  unsigned int idx = atomicAdd(&rcclNanCheckPrintCount, 1u);
  if (idx < RCCL_NANCHECK_MAX_PRINTS) {
    // %g works once we promote to float/double; rank+channel are recovered
    // host-side from the (block,thread) coordinates plus the launch grid.
    printf("[RCCL NaN/Inf] %s value=%g eltOffset=%lld block=(%d,%d,%d) thread=(%d,%d,%d)\n",
           kind == RCCL_NANCHECK_OUTPUT ? "OUTPUT" : "INPUT",
           (double)(float)v, eltOffset,
           blockIdx.x, blockIdx.y, blockIdx.z,
           threadIdx.x, threadIdx.y, threadIdx.z);
  }
#ifdef RCCL_NANCHECK_ABORT
  __trap();
#endif
}

// Extract element `e` (of type T) from a BytePack<Bytes> as a BytePack<sizeof(T)>.
// We copy the raw bytes via the always-present u8[] member, which works for every
// pack size (BytePack<0> is excluded by the EltPerPack>0 guard below) and avoids
// depending on differently-named sub-pack members across pack sizes.
template<typename T, int Bytes>
__device__ __forceinline__ BytePack<sizeof(T)> rcclNanCheckElt(
    BytePack<Bytes> pack, int e) {
  BytePack<sizeof(T)> out;
  #pragma unroll
  for (int i = 0; i < (int)sizeof(T); i++) {
    out.u8[i] = pack.u8[e*(int)sizeof(T) + i];
  }
  return out;
}

// Core: iterate the EltPerPack lanes of a BytePack and test each as type T.
// `eltBase` is the element-index of lane 0 (best-effort location hint).
template<typename T, int Bytes>
__device__ __forceinline__ void rcclNanCheckPack(
    BytePack<Bytes> pack, int kind, long long eltBase) {
  constexpr int EltPerPack = Bytes / sizeof(T);
  // Guard the empty-pack case (e.g. BytePack<0>) so the body, and therefore
  // rcclNanCheckElt's member access, is never instantiated for it.
  if constexpr (EltPerPack > 0) {
    #pragma unroll
    for (int e = 0; e < EltPerPack; e++) {
      T v = fromPack<T>(rcclNanCheckElt<T, Bytes>(pack, e));
      if (rcclNanCheckScalar<T>(v)) {
        rcclNanCheckReport<T>(v, kind, eltBase + e);
      }
    }
  }
}

// Dispatcher keyed on whether the reduction's element type is floating point.
// Non-float element types compile to nothing.
template<typename Fn, int Bytes, bool IsFloat>
struct rcclNanCheckDispatch {
  __device__ __forceinline__ static void run(BytePack<Bytes>, int, long long) {}
};
template<typename Fn, int Bytes>
struct rcclNanCheckDispatch<Fn, Bytes, true> {
  __device__ __forceinline__ static void run(
      BytePack<Bytes> pack, int kind, long long eltBase) {
    rcclNanCheckPack<typename Fn::EltType, Bytes>(pack, kind, eltBase);
  }
};

// Entry point used at the insertion site. `Fn` is the redop functor (FuncSum<T>
// etc.), so Fn::EltType gives us the scalar element type.
template<typename Fn, int Bytes>
__device__ __forceinline__ void rcclNanCheck(
    BytePack<Bytes> pack, int kind, long long eltBase) {
  rcclNanCheckDispatch<Fn, Bytes,
    IsFloatingPoint<typename Fn::EltType>::value>::run(pack, kind, eltBase);
}

// --- Word variant for the LL / LL128 reduce paths --------------------------
// LL (prims_ll.h) and LL128 (prims_ll128.h) hold reduced data in 64-bit words
// (uint64_t), not BytePack<>. Reinterpret the 8 bytes as a pack of T and test
// each lane. Non-float T compiles to nothing.
template<typename T, bool IsFloat>
struct rcclNanCheckWordDispatch {
  __device__ __forceinline__ static void run(unsigned long long, int, long long) {}
};
template<typename T>
struct rcclNanCheckWordDispatch<T, true> {
  __device__ __forceinline__ static void run(
      unsigned long long w, int kind, long long eltBase) {
    constexpr int n = (int)(sizeof(unsigned long long) / sizeof(T));
    if constexpr (n > 0) {
      T vals[n];
      __builtin_memcpy(vals, &w, n * sizeof(T));
      #pragma unroll
      for (int i = 0; i < n; i++) {
        if (rcclNanCheckScalar<T>(vals[i]))
          rcclNanCheckReport<T>(vals[i], kind, eltBase + i);
      }
    }
  }
};
template<typename T>
__device__ __forceinline__ void rcclNanCheckWord(
    unsigned long long w, int kind, long long eltBase) {
  rcclNanCheckWordDispatch<T, IsFloatingPoint<T>::value>::run(w, kind, eltBase);
}

// --- Reduce combine-point hook ---------------------------------------------
// Called from applyReduce() so Simple-path reduction operands are INPUT-checked
// without instrumenting each load site in common_kernel.h. eltOffset is -1 (not
// available here). Restricted to BytePack<N> (Simple + bias fold): the LL/LL128
// paths reduce raw words whose flag/shuffled slots would false-positive, so they
// keep their own flag-guarded RCCL_NANCHECK_WORD checks.
template<typename Fn, typename Pack>
__device__ __forceinline__ void rcclNanCheckReduceInput(Pack a, Pack b) {
  if constexpr (!std::is_integral<Pack>::value) {
    constexpr int Bytes = (int)BytePackOf<Pack>::Size;
    using Disp = rcclNanCheckDispatch<Fn, Bytes,
                   IsFloatingPoint<typename Fn::EltType>::value>;
    Disp::run(toPack(a), RCCL_NANCHECK_INPUT, (long long)-1);
    Disp::run(toPack(b), RCCL_NANCHECK_INPUT, (long long)-1);
  }
}

// Convenience macros for the insertion sites.
// RCCL_NANCHECK:      RedFn/BytePerPack in scope (common_kernel.h, Simple path).
// RCCL_NANCHECK_WORD: element type T in scope (prims_ll.h / prims_ll128.h).
#define RCCL_NANCHECK(pack, kind, eltBase) \
  rcclNanCheck<RedFn, BytePerPack>((pack), (kind), (long long)(eltBase))
#define RCCL_NANCHECK_WORD(word, kind, eltBase) \
  rcclNanCheckWord<T>((unsigned long long)(word), (kind), (long long)(eltBase))
// Used inside applyReduce(); `fn` is the redop functor in scope there.
#define RCCL_NANCHECK_REDUCE_INPUTS(fn, a, b) \
  rcclNanCheckReduceInput<decltype(fn)>((a), (b))

#else // !RCCL_ENABLE_NAN_CHECK

// Zero-overhead no-op when the detector is disabled.
#define RCCL_NANCHECK(pack, kind, eltBase) do {} while (0)
#define RCCL_NANCHECK_WORD(word, kind, eltBase) do {} while (0)
#define RCCL_NANCHECK_REDUCE_INPUTS(fn, a, b) do {} while (0)

#endif // RCCL_ENABLE_NAN_CHECK

#endif // RCCL_NAN_CHECK_H_
