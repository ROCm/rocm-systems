/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Golden-data generator for the narrow float (fp8/bf8) reduce paths.
//
// Purpose: freeze the answers a known-good tree returns, so a later change can
// be shown to preserve them. Run this once on the reference commit, keep the
// output, and diff against a run from the tree under review. A test that
// computes its expectation from the same headers it exercises cannot do this;
// it moves whenever the implementation moves.
//
// What is captured, per element type (fp8 = e4m3, bf8 = e5m2):
//
//   sum, prod, min, max   via Apply_Reduce<Fn, EltPerPack>
//   premul                via Apply_PreOp<FuncPreMulSum<T>, EltPerPack>
//
// each at EltPerPack=1 and EltPerPack=2. Going through Apply_* rather than the
// helpers in rccl_float8.h is deliberate: that is the surface the collective
// kernels dispatch through, and on a tree with no EltPerPack=2 specialization
// the generic template recurses into the 1-wide case, which is exactly the
// per-element reference a vectorized replacement has to reproduce.
//
// Every input space here is finite and swept exhaustively:
//
//   binary ops  all 256x256 ordered byte pairs
//   premul      all 256 inputs x all 256 scalars (FuncPreMulSum<rccl_float8>
//               decodes its scalar from one fp8 byte, so 256 is the whole space)
//
// For EltPerPack=2 the two lanes are fed (a,b) and (b,a), so each lane position
// independently sees all ordered pairs; a helper that transposes or drops a lane
// cannot hide.
//
// Results are recorded as raw bytes. Decoded floats would lose the distinction
// between -0 and +0 and between NaN encodings, which is where narrow-float
// conversion bugs tend to live.
//
// The output is only meaningful for the architecture that produced it: gfx942
// uses the fnuz encodings while gfx950 and gfx12xx use OCP, and the host
// software emulation differs from both. The arch name is recorded in the header
// and a consumer must refuse to compare across a mismatch.
//
// Usage:
//   rccl-narrow-fp-golden [--out FILE] [--device N] [--label STR] [--digest-only]
//
//   --label     provenance string, e.g. the git commit the tree was built from
//   --digest-only  emit only the per-section digests, not the byte tables

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <vector>

// Generated header: defines RCCL_FLOAT8 and RCCL_BFLOAT16, which gate the
// narrow-float specializations in reduce_kernel.h. Without it the generic
// template is instantiated instead and the sweep would measure the wrong code.
#include "nccl.h"

// reduce_kernel.h refers to hip_bfloat16, which src/include/device.h aliases to
// __hip_bfloat16 on ROCm 6 and newer. That alias is replicated here rather than
// including device.h, which would pull in the library's entire header chain.
#if !defined(_HIP_INCLUDE_HIP_AMD_DETAIL_HIP_BFLOAT16_H_) && !defined(_HIP_BFLOAT16_H_)
#define _HIP_INCLUDE_HIP_AMD_DETAIL_HIP_BFLOAT16_H_
#define _HIP_BFLOAT16_H_
#include <hip/hip_bf16.h>
typedef __hip_bfloat16 hip_bfloat16;
#endif

#include "reduce_kernel.h"

namespace {

// Each sweep is a 16-bit case index split into two bytes, so 65536 cases.
constexpr int kCases = 65536;

#define HIP_CHECK(expr)                                                                  \
  do {                                                                                   \
    hipError_t _e = (expr);                                                               \
    if (_e != hipSuccess) {                                                                \
      std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #expr,            \
                   hipGetErrorString(_e));                                                 \
      std::exit(1);                                                                        \
    }                                                                                      \
  } while (0)

// Raw byte <-> BytePack moves. Done with memcpy so this file does not depend on
// BytePack's field names.
template <int N>
__device__ __forceinline__ BytePack<N> packFromBytes(const uint8_t* src) {
  BytePack<N> p;
  __builtin_memcpy(&p, src, N);
  return p;
}

template <int N>
__device__ __forceinline__ void bytesFromPack(BytePack<N> p, uint8_t* dst) {
  __builtin_memcpy(dst, &p, N);
}

// Which element type and which op, as integers. The kernels below are
// parameterized by these and never by rccl_float8 itself, because on gfx942 that
// is a different type in the two compilation passes -- fnuz in the device pass,
// OCP in the host pass, since the fnuz encodings only exist on that target. A
// __global__ templated on it therefore mangles to a different name on each side
// and the launch cannot find its own kernel. Integers mangle the same everywhere.
enum : int { kFp8 = 0, kBf8 = 1 };
enum : int { kOpSum = 0, kOpProd = 1, kOpMinMax = 2 };

template <int Type>
using ElemType = std::conditional_t<Type == kBf8, rccl_bfloat8, rccl_float8>;

template <int Type, int Op>
using ReduceFn = std::conditional_t<
    Op == kOpSum, FuncSum<ElemType<Type>>,
    std::conditional_t<Op == kOpProd, FuncProd<ElemType<Type>>, FuncMinMax<ElemType<Type>>>>;

// Binary reduce sweep. opArg is passed to the Fn constructor; for FuncMinMax it
// selects min (bit 0 clear) or max (bit 0 set).
// The bodies are device-only: reduce_kernel.h is compiled by the library with
// --offload-device-only and its reduce specializations are not all visible in a
// host pass, where the generic template would then be instantiated and fail.
template <int Type, int Op, int EltPerPack>
__global__ void kReduce(uint64_t opArg, uint8_t* __restrict__ out) {
#if __HIP_DEVICE_COMPILE__
  using Fn = ReduceFn<Type, Op>;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= kCases) return;

  const uint8_t a = uint8_t(i >> 8);
  const uint8_t b = uint8_t(i & 0xFF);

  Fn fn(opArg);

  if constexpr (EltPerPack == 1) {
    uint8_t ab[1] = {a};
    uint8_t bb[1] = {b};
    BytePack<1> r = Apply_Reduce<Fn, 1>::reduce(fn, packFromBytes<1>(ab), packFromBytes<1>(bb));
    bytesFromPack<1>(r, out + i);
  } else {
    // lane 0 sees (a,b), lane 1 sees (b,a)
    uint8_t ab[2] = {a, b};
    uint8_t bb[2] = {b, a};
    BytePack<2> r = Apply_Reduce<Fn, 2>::reduce(fn, packFromBytes<2>(ab), packFromBytes<2>(bb));
    bytesFromPack<2>(r, out + 2 * i);
  }
#endif
}

// PreMulSum sweep: high byte is the input element, low byte is the scalar.
template <int Type, int EltPerPack>
__global__ void kPreMul(uint8_t* __restrict__ out) {
#if __HIP_DEVICE_COMPILE__
  using T = ElemType<Type>;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= kCases) return;

  const uint8_t x = uint8_t(i >> 8);
  const uint8_t s = uint8_t(i & 0xFF);

  // Named variable, not FuncPreMulSum<T> fn(uint64_t(s)): that parses as a
  // function declaration.
  const uint64_t opArg = s;
  FuncPreMulSum<T> fn(opArg);

  if constexpr (EltPerPack == 1) {
    uint8_t xb[1] = {x};
    BytePack<1> r = Apply_PreOp<FuncPreMulSum<T>, 1>::preOp(fn, packFromBytes<1>(xb));
    bytesFromPack<1>(r, out + i);
  } else {
    // Give the second lane a different value so both lanes are exercised
    // independently across the sweep.
    uint8_t xb[2] = {x, uint8_t(0xFF - x)};
    BytePack<2> r = Apply_PreOp<FuncPreMulSum<T>, 2>::preOp(fn, packFromBytes<2>(xb));
    bytesFromPack<2>(r, out + 2 * i);
  }
#endif
}

uint64_t fnv1a64(const uint8_t* p, size_t n) {
  uint64_t h = 0xcbf29ce484222325ull;
  for (size_t i = 0; i < n; ++i) {
    h ^= p[i];
    h *= 0x100000001b3ull;
  }
  return h;
}

struct Emitter {
  std::FILE* f;
  bool digestOnly;
  int sections = 0;

  void section(const char* name, const std::vector<uint8_t>& bytes, int bytesPerCase) {
    std::fprintf(f, "[%s] cases=%d bytes_per_case=%d digest=0x%016llx\n", name, kCases,
                 bytesPerCase, (unsigned long long)fnv1a64(bytes.data(), bytes.size()));
    if (!digestOnly) {
      for (size_t i = 0; i < bytes.size(); i += 32) {
        const size_t n = (bytes.size() - i < 32) ? bytes.size() - i : 32;
        for (size_t j = 0; j < n; ++j) std::fprintf(f, "%02x", bytes[i + j]);
        std::fputc('\n', f);
      }
    }
    ++sections;
  }
};

// Runs one kernel launch and hands the bytes to the emitter.
template <typename Launch>
void run(Emitter& em, const char* name, int bytesPerCase, uint8_t* dOut, Launch launch) {
  const size_t n = size_t(kCases) * bytesPerCase;
  HIP_CHECK(hipMemset(dOut, 0, n));
  launch();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<uint8_t> host(n);
  HIP_CHECK(hipMemcpy(host.data(), dOut, n, hipMemcpyDeviceToHost));
  em.section(name, host, bytesPerCase);
}

constexpr int kThreads = 256;
constexpr int kBlocks = (kCases + kThreads - 1) / kThreads;

// All sections for one element type. `tag` is "fp8" or "bf8".
template <int Type>
void emitType(Emitter& em, const char* tag, uint8_t* dOut) {
  char name[64];
  auto nm = [&](const char* op, int w) {
    std::snprintf(name, sizeof(name), "%s/%s/w%d", tag, op, w);
    return name;
  };

  run(em, nm("sum", 1), 1, dOut, [&] {
    hipLaunchKernelGGL((kReduce<Type, kOpSum, 1>), kBlocks, kThreads, 0, 0, 0ull, dOut);
  });
  run(em, nm("sum", 2), 2, dOut, [&] {
    hipLaunchKernelGGL((kReduce<Type, kOpSum, 2>), kBlocks, kThreads, 0, 0, 0ull, dOut);
  });

  run(em, nm("prod", 1), 1, dOut, [&] {
    hipLaunchKernelGGL((kReduce<Type, kOpProd, 1>), kBlocks, kThreads, 0, 0, 0ull, dOut);
  });
  run(em, nm("prod", 2), 2, dOut, [&] {
    hipLaunchKernelGGL((kReduce<Type, kOpProd, 2>), kBlocks, kThreads, 0, 0, 0ull, dOut);
  });

  // FuncMinMax: isMinNotMax = (opArg & 1) == 0
  run(em, nm("min", 1), 1, dOut, [&] {
    hipLaunchKernelGGL((kReduce<Type, kOpMinMax, 1>), kBlocks, kThreads, 0, 0, 0ull, dOut);
  });
  run(em, nm("min", 2), 2, dOut, [&] {
    hipLaunchKernelGGL((kReduce<Type, kOpMinMax, 2>), kBlocks, kThreads, 0, 0, 0ull, dOut);
  });
  run(em, nm("max", 1), 1, dOut, [&] {
    hipLaunchKernelGGL((kReduce<Type, kOpMinMax, 1>), kBlocks, kThreads, 0, 0, 1ull, dOut);
  });
  run(em, nm("max", 2), 2, dOut, [&] {
    hipLaunchKernelGGL((kReduce<Type, kOpMinMax, 2>), kBlocks, kThreads, 0, 0, 1ull, dOut);
  });

  run(em, nm("premul", 1), 1, dOut,
      [&] { hipLaunchKernelGGL((kPreMul<Type, 1>), kBlocks, kThreads, 0, 0, dOut); });
  run(em, nm("premul", 2), 2, dOut,
      [&] { hipLaunchKernelGGL((kPreMul<Type, 2>), kBlocks, kThreads, 0, 0, dOut); });
}

} // namespace

int main(int argc, char** argv) {
  const char* outPath = nullptr;
  const char* label = "";
  int device = 0;
  bool digestOnly = false;

  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--out") && i + 1 < argc) outPath = argv[++i];
    else if (!std::strcmp(argv[i], "--label") && i + 1 < argc) label = argv[++i];
    else if (!std::strcmp(argv[i], "--device") && i + 1 < argc) device = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--digest-only")) digestOnly = true;
    else {
      std::fprintf(stderr,
                   "usage: %s [--out FILE] [--device N] [--label STR] [--digest-only]\n", argv[0]);
      return 2;
    }
  }

  HIP_CHECK(hipSetDevice(device));
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, device));

  std::FILE* f = stdout;
  if (outPath != nullptr) {
    f = std::fopen(outPath, "w");
    if (f == nullptr) {
      std::fprintf(stderr, "cannot open %s for writing\n", outPath);
      return 1;
    }
  }

  // Provenance. arch and saturate_inf both change the answers, so a consumer
  // has to check them before comparing anything.
  std::fprintf(f, "# rccl-narrow-fp-golden v1\n");
  std::fprintf(f, "# arch: %s\n", prop.gcnArchName);
  std::fprintf(f, "# hip_version: %d\n", HIP_VERSION);
  // Only exists on trees that carry the saturation policy switch; a reference
  // tree predating it reports "absent" rather than failing to build.
#ifdef RCCL_NARROW_FP_SATURATE_INF
  std::fprintf(f, "# saturate_inf: %d\n", (int)RCCL_NARROW_FP_SATURATE_INF);
#else
  std::fprintf(f, "# saturate_inf: absent\n");
#endif
  std::fprintf(f, "# label: %s\n", label);
  std::fprintf(f, "# note: bytes are raw storage, element 0 first\n");

  uint8_t* dOut = nullptr;
  HIP_CHECK(hipMalloc(&dOut, size_t(kCases) * 2));

  Emitter em{f, digestOnly};
  emitType<kFp8>(em, "fp8", dOut);
  emitType<kBf8>(em, "bf8", dOut);

  std::fprintf(f, "# sections: %d\n", em.sections);

  HIP_CHECK(hipFree(dOut));
  if (f != stdout) std::fclose(f);
  return 0;
}
