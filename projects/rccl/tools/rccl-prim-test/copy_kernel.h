/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COMMON_KERNEL_H_
#define NCCL_COMMON_KERNEL_H_

#include <cstdio>
#include <cstdint>

// Define min for ssize_t
static __device__ int min(int a, ssize_t b) { return (a < b) ? a : b; }

typedef uint64_t PackType;


template<class FUNC, typename T>
struct MULTI {
  __device__ PackType operator()(const PackType x, const PackType y) const
  {
    return FUNC()(x, y);
  }
};


template<typename T> inline __device__
T vFetch(const volatile T* ptr) {
  return __builtin_nontemporal_load(ptr);
}

template<typename T> inline __device__
void vStore(volatile T* ptr, const T val) {
  __builtin_nontemporal_store(val, ptr);
}

template<typename T>
struct FuncPassA {
  __device__ T operator()(const T x, const T y) const {
    return x;
  }
};

template<typename T>
struct FuncSum {
  __device__ T operator()(const T x, const T y) const {
    return x + y;
  }
};

template<class FUNC>
struct MULTI<FUNC, float> {
  static_assert(sizeof(PackType) == 2 * sizeof(float),
      "PackType must be twice the size of float.");
  union converter {
    PackType storage;
    struct {
      float a, b;
    };
  };

  __device__ PackType operator()(const PackType x, const PackType y) const {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    cr.a = FUNC()(cx.a, cy.a);
    cr.b = FUNC()(cx.b, cy.b);

    return cr.storage;
  }
};


typedef ulong2 Pack128;

template<class FUNC, typename T>
struct MULTI128 {
  __device__ void operator()(Pack128& x, Pack128& y) {
    x.x = MULTI<FUNC, T>()(x.x, y.x);
    x.y = MULTI<FUNC, T>()(x.y, y.y);
  }
};

inline __device__ void Fetch128(Pack128& v, const Pack128* p) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HCC__) || defined(__HIPCC__)
  v.x = __builtin_nontemporal_load(&p->x);
  v.y = __builtin_nontemporal_load(&p->y);
#else
  asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p) : "memory");
#endif
}
inline __device__ void Store128(Pack128* p, Pack128& v) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HCC__) || defined(__HIPCC__)
  __builtin_nontemporal_store(v.x, &p->x);
  __builtin_nontemporal_store(v.y, &p->y);
#else
  asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" :: "l"(p), "l"(v.x), "l"(v.y) : "memory");
#endif
}

#define WARP_SIZE 64

template<class FUNC, typename T, int UNROLL, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS>
__device__ __forceinline__ void ReduceCopyMulti(const int w, const int nw, const int t,
    int nsrcs, const T** s, int ndsts, T** d, const int elemOffset, const int Nelem) {
  const int inc = nw * UNROLL * WARP_SIZE;
  int offset = w * UNROLL * WARP_SIZE + t;

  const T* srcs[MAXSRCS];
  for (int i=0; i<MAXSRCS; i++) srcs[i] = s[i]+elemOffset+offset;
  T* dsts[MAXDSTS];
  for (int i=0; i<MAXDSTS; i++) dsts[i] = d[i]+elemOffset+offset;

  while (offset < Nelem) {
    T vals[UNROLL];
    // Load and reduce
    for (int u = 0; u < UNROLL; ++u) vals[u] = vFetch(srcs[0]+u*WARP_SIZE);

    #pragma unroll
    for (int i=1; i<MINSRCS; i++) {
      T vals2[UNROLL];
      for (int u = 0; u < UNROLL; ++u) vals2[u] = vFetch(srcs[i]+u*WARP_SIZE);
      for (int u = 0; u < UNROLL; ++u) vals[u] = FUNC()(vals[u], vals2[u]);
    }
    #pragma unroll
    for (int i=MINSRCS; i<MAXSRCS; i++) {
      if (i<nsrcs) {
        T vals2[UNROLL];
        for (int u = 0; u < UNROLL; ++u) vals2[u] = vFetch(srcs[i]+u*WARP_SIZE);
        for (int u = 0; u < UNROLL; ++u) vals[u] = FUNC()(vals[u], vals2[u]);
      }
    }

    // Store
    #pragma unroll
    for (int i = 0; i < MINDSTS; i++) {
      for (int u = 0; u < UNROLL; ++u) vStore(dsts[i]+u*WARP_SIZE, vals[u]);
    }
    #pragma unroll
    for (int i=MINDSTS; i<MAXDSTS; i++) {
      if (i<ndsts) {
        for (int u = 0; u < UNROLL; ++u) vStore(dsts[i]+u*WARP_SIZE, vals[u]);
      }
    }
    for (int i=0; i<MAXSRCS; i++) srcs[i] += inc;
    for (int i=0; i<MAXDSTS; i++) dsts[i] += inc;
    offset += inc;
  }
}

template<class FUNC, typename T, int UNROLL, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS>
__device__ void ReduceCopy128bMulti(const int w, const int nw, const int t,
    int nsrcs, const T** s, int ndsts, T** d, const int elemOffset, const int Npack) {
  const int inc = nw * UNROLL * WARP_SIZE;
  int offset = w * UNROLL * WARP_SIZE + t;

  const Pack128* srcs[MAXSRCS];
  for (int i=0; i<MAXSRCS; i++) srcs[i] = ((const Pack128*)(s[i]+elemOffset))+offset;
  Pack128* dsts[MAXDSTS];
  for (int i=0; i<MAXDSTS; i++) dsts[i] = ((Pack128*)(d[i]+elemOffset))+offset;

  while (offset < Npack) {
    Pack128 vals[UNROLL];
    // Load and reduce
    for (int u = 0; u < UNROLL; ++u) Fetch128(vals[u], srcs[0]+u*WARP_SIZE);

    for (int i=1; i<MINSRCS; i++) {
      Pack128 vals2[UNROLL];
      for (int u = 0; u < UNROLL; ++u) Fetch128(vals2[u], srcs[i]+u*WARP_SIZE);
      for (int u = 0; u < UNROLL; ++u) MULTI128<FUNC, T>()(vals[u], vals2[u]);
    }
    for (int i=MINSRCS; i<MAXSRCS && i<nsrcs; i++) {
      Pack128 vals2[UNROLL];
      for (int u = 0; u < UNROLL; ++u) Fetch128(vals2[u], srcs[i]+u*WARP_SIZE);
      for (int u = 0; u < UNROLL; ++u) MULTI128<FUNC, T>()(vals[u], vals2[u]);
    }

    // Store
    for (int i = 0; i < MINDSTS; i++) {
      for (int u = 0; u < UNROLL; ++u) Store128(dsts[i]+u*WARP_SIZE, vals[u]);
    }
    for (int i=MINDSTS; i<MAXDSTS; i++) {
      if (i<ndsts) {
        for (int u = 0; u < UNROLL; ++u) Store128(dsts[i]+u*WARP_SIZE, vals[u]);
      }
    }
    for (int i=0; i<MAXSRCS; i++) srcs[i] += inc;
    for (int i=0; i<MAXDSTS; i++) dsts[i] += inc;
    offset += inc;
  }
}

template <typename T>
__device__ int ptrAlign128(T* ptr) { return (uint64_t)ptr % alignof(int32_t); }

#define PACKELEMS (sizeof(Pack128) / sizeof(T))

#if defined(__HIP_PLATFORM_AMD__) || defined(__HCC__) || defined(__HIPCC__)
// Multiply UNROLL by 2 if single source/single destination
#define AUTOUNROLL (UNROLL*((MINSRCS==1 && MINDSTS==1) ? 2 : 1))
#endif

template<int UNROLL, class FUNC, typename T, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS>
__device__ __forceinline__ void ReduceOrCopyMulti(const int tid, const int nthreads,
    int nsrcs, const T** srcs, int ndsts, T** dsts,
    int N) {
  int Nrem = N;
  if (Nrem <= 0) return;

  int w = tid / WARP_SIZE;       // Warp number
  int nw = nthreads / WARP_SIZE; // Number of warps
  int t = tid % WARP_SIZE;       // Thread (inside the warp)

  // Check that all is 16B aligned. If not don't use 16B load/stores.
  int align = 0;
  #pragma unroll
  for (int i=0; i<MINSRCS; i++) align |= ptrAlign128(srcs[i]);
  for (int i=MINSRCS; i<MAXSRCS && i<nsrcs; i++) align |= ptrAlign128(srcs[i]);
  #pragma unroll
  for (int i=0; i<MINDSTS; i++) align |= ptrAlign128(dsts[i]);
  for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) align |= ptrAlign128(dsts[i]);

  int offset = 0;
  if (align == 0) {
    // fast path: use 128b loads/stores to do the bulk of the work,
    // assuming the pointers we have are all 128-bit aligned.

    // main loop
    int Npack = (Nrem / (PACKELEMS*AUTOUNROLL*WARP_SIZE)) * (AUTOUNROLL*WARP_SIZE); // round down
    int Nelem = Npack * PACKELEMS;

    ReduceCopy128bMulti<FUNC, T, AUTOUNROLL, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(w, nw, t, nsrcs, srcs, ndsts, dsts, offset, Npack);

    Nrem -= Nelem;
    if (Nrem == 0) return;
    offset += Nelem;

    // slightly less optimized for section when we don't have full unrolling
    Npack = Nrem / PACKELEMS;
    Nelem = Npack * PACKELEMS;

    ReduceCopy128bMulti<FUNC, T, 1, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(w, nw, t, nsrcs, srcs, ndsts, dsts, offset, Npack);

    Nrem -= Nelem;
    if (Nrem == 0) return;
    offset += Nelem;
  }

  // unrolled, by-type (mostly for unaligned buffers)
  int Nelem = (Nrem / (UNROLL*PACKELEMS/2*WARP_SIZE)) * (UNROLL*PACKELEMS/2*WARP_SIZE); // round down

  ReduceCopyMulti<FUNC, T, UNROLL*PACKELEMS/2, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(w, nw, t, nsrcs, srcs, ndsts, dsts, offset, Nelem);

  Nrem -= Nelem;
  if (Nrem == 0) return;
  offset += Nelem;

  // no unroll, by type. Should finish what's remaining.
  ReduceCopyMulti<FUNC, T, 1, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(w, nw, t, nsrcs, srcs, ndsts, dsts, offset, Nrem);
}

#endif
