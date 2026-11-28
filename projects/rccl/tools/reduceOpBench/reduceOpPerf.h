/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
*/

/**
 * @file reduceOpPerf.h
 *
 * Throughput benchmark for the FP8 reduction paths a collective would use.
 *
 * Each variant implements the same logical operation — upcast two FP8 operands,
 * add, downcast back to FP8 — using a different hardware path, so the timings
 * are directly comparable per element.
 *
 * The data stays in registers and the inner loop runs 4 independent chains, so
 * this measures ALU throughput of the conversion path, not memory bandwidth and
 * not single-chain latency.
 */

#ifndef REDUCE_OP_PERF_H_
#define REDUCE_OP_PERF_H_

#include <stdint.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>

typedef float float2_t __attribute__((ext_vector_type(2)));
typedef _Float16 half2_t __attribute__((ext_vector_type(2)));
typedef short shortx2_t __attribute__((ext_vector_type(2)));

// Independent chains per thread, enough to keep the conversion pipe busy.
#define REDUCE_OP_ILP_WAYS 4

__device__ inline uint32_t ropShortx2ToU32(shortx2_t v) {
  union {
    shortx2_t s;
    uint32_t u32;
  } u{v};
  return u.u32;
}

__device__ inline half2_t ropU32ToHalf2(uint32_t v) {
  union {
    uint32_t u32;
    half2_t h;
  } u{v};
  return u.h;
}

// ---------------------------------------------------------------------------
// Variants. Each exposes:
//   available() - is the path usable on this offload target
//   step(acc, src) - one reduce op over the packed FP8 lanes of acc and src
// ---------------------------------------------------------------------------

// e4m3 via f32: cvt_pk_f32_fp8 -> add -> cvt_pk_fp8_f32. 4 elements per step.
struct OpF32Pk2E4m3 {
#if __has_builtin(__builtin_amdgcn_cvt_pk_f32_fp8) && \
    __has_builtin(__builtin_amdgcn_cvt_pk_fp8_f32)
  static __device__ bool available() {
    return __builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_pk_f32_fp8) &&
           __builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_pk_fp8_f32);
  }
  static __device__ uint32_t step(uint32_t acc, uint32_t src) {
    float2_t alo = __builtin_amdgcn_cvt_pk_f32_fp8(acc, false);
    float2_t ahi = __builtin_amdgcn_cvt_pk_f32_fp8(acc, true);
    float2_t slo = __builtin_amdgcn_cvt_pk_f32_fp8(src, false);
    float2_t shi = __builtin_amdgcn_cvt_pk_f32_fp8(src, true);
    float2_t rlo = alo + slo;
    float2_t rhi = ahi + shi;
    uint32_t out = 0;
    out = __builtin_amdgcn_cvt_pk_fp8_f32(rlo[0], rlo[1], out, false);
    out = __builtin_amdgcn_cvt_pk_fp8_f32(rhi[0], rhi[1], out, true);
    return out;
  }
#else
  static __device__ bool available() { return false; }
  static __device__ uint32_t step(uint32_t acc, uint32_t) { return acc; }
#endif
};

// e5m2 via f32: cvt_pk_f32_bf8 -> add -> cvt_pk_bf8_f32. 4 elements per step.
struct OpF32Pk2E5m2 {
#if __has_builtin(__builtin_amdgcn_cvt_pk_f32_bf8) && \
    __has_builtin(__builtin_amdgcn_cvt_pk_bf8_f32)
  static __device__ bool available() {
    return __builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_pk_f32_bf8) &&
           __builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_pk_bf8_f32);
  }
  static __device__ uint32_t step(uint32_t acc, uint32_t src) {
    float2_t alo = __builtin_amdgcn_cvt_pk_f32_bf8(acc, false);
    float2_t ahi = __builtin_amdgcn_cvt_pk_f32_bf8(acc, true);
    float2_t slo = __builtin_amdgcn_cvt_pk_f32_bf8(src, false);
    float2_t shi = __builtin_amdgcn_cvt_pk_f32_bf8(src, true);
    float2_t rlo = alo + slo;
    float2_t rhi = ahi + shi;
    uint32_t out = 0;
    out = __builtin_amdgcn_cvt_pk_bf8_f32(rlo[0], rlo[1], out, false);
    out = __builtin_amdgcn_cvt_pk_bf8_f32(rhi[0], rhi[1], out, true);
    return out;
  }
#else
  static __device__ bool available() { return false; }
  static __device__ uint32_t step(uint32_t acc, uint32_t) { return acc; }
#endif
};

// e4m3 via f16 block-scale: cvt_scalef32_pk_f16_fp8 -> add -> cvt_scalef32_pk_fp8_f16.
struct OpF16ScaleE4m3 {
#if __has_builtin(__builtin_amdgcn_cvt_scalef32_pk_f16_fp8) && \
    __has_builtin(__builtin_amdgcn_cvt_scalef32_pk_fp8_f16)
  static __device__ bool available() {
    return __builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_f16_fp8) &&
           __builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_fp8_f16);
  }
  static __device__ uint32_t step(uint32_t acc, uint32_t src) {
    half2_t alo = __builtin_amdgcn_cvt_scalef32_pk_f16_fp8(acc, 1.f, false);
    half2_t ahi = __builtin_amdgcn_cvt_scalef32_pk_f16_fp8(acc, 1.f, true);
    half2_t slo = __builtin_amdgcn_cvt_scalef32_pk_f16_fp8(src, 1.f, false);
    half2_t shi = __builtin_amdgcn_cvt_scalef32_pk_f16_fp8(src, 1.f, true);
    half2_t rlo = alo + slo;
    half2_t rhi = ahi + shi;
    return ropShortx2ToU32(
        __builtin_amdgcn_cvt_scalef32_pk_fp8_f16(rlo, rhi, 1.f, false));
  }
#else
  static __device__ bool available() { return false; }
  static __device__ uint32_t step(uint32_t acc, uint32_t) { return acc; }
#endif
};

// e5m2 via f16 block-scale.
struct OpF16ScaleE5m2 {
#if __has_builtin(__builtin_amdgcn_cvt_scalef32_pk_f16_bf8) && \
    __has_builtin(__builtin_amdgcn_cvt_scalef32_pk_bf8_f16)
  static __device__ bool available() {
    return __builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_f16_bf8) &&
           __builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_bf8_f16);
  }
  static __device__ uint32_t step(uint32_t acc, uint32_t src) {
    half2_t alo = __builtin_amdgcn_cvt_scalef32_pk_f16_bf8(acc, 1.f, false);
    half2_t ahi = __builtin_amdgcn_cvt_scalef32_pk_f16_bf8(acc, 1.f, true);
    half2_t slo = __builtin_amdgcn_cvt_scalef32_pk_f16_bf8(src, 1.f, false);
    half2_t shi = __builtin_amdgcn_cvt_scalef32_pk_f16_bf8(src, 1.f, true);
    half2_t rlo = alo + slo;
    half2_t rhi = ahi + shi;
    return ropShortx2ToU32(
        __builtin_amdgcn_cvt_scalef32_pk_bf8_f16(rlo, rhi, 1.f, false));
  }
#else
  static __device__ bool available() { return false; }
  static __device__ uint32_t step(uint32_t acc, uint32_t) { return acc; }
#endif
};

// e4m3 through the HIP type, i.e. whatever the compiler emits without
// hand-picked intrinsics. Useful as the "write the obvious code" reference.
struct OpHipTypeE4m3 {
  static __device__ bool available() { return true; }
  static __device__ uint32_t step(uint32_t acc, uint32_t src) {
    uint32_t out = 0;
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      __hip_fp8_e4m3 a, s;
      a.__x = (unsigned char)((acc >> (8 * b)) & 0xffu);
      s.__x = (unsigned char)((src >> (8 * b)) & 0xffu);
      __hip_fp8_e4m3 r(static_cast<float>(a) + static_cast<float>(s));
      out |= (uint32_t)r.__x << (8 * b);
    }
    return out;
  }
};

// Reference ceiling: packed f16 add with no conversion at all. 2 elements per
// step, so it is normalised the same way and shows what the add alone costs.
struct OpF16AddOnly {
  static __device__ bool available() { return true; }
  static __device__ uint32_t step(uint32_t acc, uint32_t src) {
    half2_t a = ropU32ToHalf2(acc);
    half2_t s = ropU32ToHalf2(src);
    half2_t r = a + s;
    union {
      half2_t h;
      uint32_t u32;
    } u{r};
    return u.u32;
  }
};

// ---------------------------------------------------------------------------
// Registry. One list drives the device availability kernel and the host table,
// so the two can never drift out of order.
//   X(type, name, description, elements processed per step)
// ---------------------------------------------------------------------------

#define REDUCE_OP_VARIANTS(X)                                                  \
  X(OpF32Pk2E4m3, "e4m3 via f32 pk2",                                          \
    "cvt_pk_f32_fp8 -> v_pk_add_f32 -> cvt_pk_fp8_f32", 4)                     \
  X(OpF32Pk2E5m2, "e5m2 via f32 pk2",                                          \
    "cvt_pk_f32_bf8 -> v_pk_add_f32 -> cvt_pk_bf8_f32", 4)                     \
  X(OpF16ScaleE4m3, "e4m3 via f16 scalef32",                                   \
    "cvt_scalef32_pk_f16_fp8 -> v_pk_add_f16 -> cvt_scalef32_pk_fp8_f16", 4)   \
  X(OpF16ScaleE5m2, "e5m2 via f16 scalef32",                                   \
    "cvt_scalef32_pk_f16_bf8 -> v_pk_add_f16 -> cvt_scalef32_pk_bf8_f16", 4)   \
  X(OpHipTypeE4m3, "e4m3 via __hip_fp8_e4m3",                                  \
    "compiler-generated conversion, no explicit intrinsics", 4)                \
  X(OpF16AddOnly, "reference: f16 add only",                                   \
    "no conversion, shows the add cost alone", 2)

enum {
#define X(Op, name, desc, n) REDUCE_OP_IDX_##Op,
  REDUCE_OP_VARIANTS(X)
#undef X
      REDUCE_OP_NUM_VARIANTS
};

__global__ void reduceOpAvailKernel(int* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
#define X(Op, name, desc, n) out[REDUCE_OP_IDX_##Op] = Op::available() ? 1 : 0;
  REDUCE_OP_VARIANTS(X)
#undef X
}

// ---------------------------------------------------------------------------
// Timing kernel
// ---------------------------------------------------------------------------

template <typename Op>
__global__ __launch_bounds__(256) void reduceOpPerfKernel(
    const uint32_t* __restrict__ in, uint32_t* __restrict__ out, int iters) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t src = in[tid];

  // Independent chains: the accumulators never feed each other inside the loop,
  // so the measurement is throughput rather than dependent-issue latency.
  uint32_t v0 = src;
  uint32_t v1 = src ^ 0x01010101u;
  uint32_t v2 = src ^ 0x02020202u;
  uint32_t v3 = src ^ 0x03030303u;

  // Rotating the shared operand keeps it loop-variant. Without this the
  // compiler hoists its upcast out of the loop and only one of the two
  // operands gets converted per step, overstating every conversion path.
  uint32_t s = src;
  for (int i = 0; i < iters; ++i) {
    s = (s << 8) | (s >> 24);
    v0 = Op::step(v0, s);
    v1 = Op::step(v1, s);
    v2 = Op::step(v2, s);
    v3 = Op::step(v3, s);
  }

  // Consuming every chain keeps the loop from being optimised away.
  out[tid] = v0 ^ v1 ^ v2 ^ v3;
}

typedef void (*ReduceOpLaunchFn)(const uint32_t*, uint32_t*, int, int, int);

template <typename Op>
void reduceOpLaunch(const uint32_t* in, uint32_t* out, int iters, int blocks,
                    int threads) {
  hipLaunchKernelGGL(reduceOpPerfKernel<Op>, dim3(blocks), dim3(threads), 0, 0,
                     in, out, iters);
}

struct ReduceOpVariant {
  const char* name;
  const char* detail;
  int elemsPerStep;
  ReduceOpLaunchFn launch;
};

static const ReduceOpVariant kReduceOpVariants[REDUCE_OP_NUM_VARIANTS] = {
#define X(Op, name, desc, n) {name, desc, n, &reduceOpLaunch<Op>},
    REDUCE_OP_VARIANTS(X)
#undef X
};

#endif  // REDUCE_OP_PERF_H_
