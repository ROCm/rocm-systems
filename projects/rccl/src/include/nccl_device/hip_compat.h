#ifndef _NCCL_DEVICE_HIP_COMPAT_H_
#define _NCCL_DEVICE_HIP_COMPAT_H_

/*
 * This header provides compatibility between CUDA and HIP for the nccl_device
 * headers. It defines unified macros and provides HIP implementations of
 * CUDA-specific constructs.
 *
 * Usage: Include this header FIRST in nccl_device headers, then use
 * NCCL_DEVICE_COMPILE instead of __CUDACC__ for device code guards.
 *
 * "NO HIP EQUIVALENT:" marks NVIDIA-only features (e.g. NVLINK multicast).
 */

////////////////////////////////////////////////////////////////////////////////
// Unified device compiler detection
//
// NCCL_DEVICE_COMPILE: defined when compiling device code (CUDA or HIP)
// NCCL_HIP_PLATFORM:   defined when targeting AMD/HIP
// NCCL_CUDA_PLATFORM:  defined when targeting NVIDIA/CUDA
////////////////////////////////////////////////////////////////////////////////

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
  #define NCCL_HIP_PLATFORM 1
  #define NCCL_DEVICE_COMPILE 1
#elif defined(__CUDACC__)
  #define NCCL_CUDA_PLATFORM 1
  #define NCCL_DEVICE_COMPILE 1
#else
  #define NCCL_DEVICE_COMPILE 0
#endif

////////////////////////////////////////////////////////////////////////////////
// Device function qualifiers
////////////////////////////////////////////////////////////////////////////////

#if NCCL_DEVICE_COMPILE
  #define NCCL_DEVICE_INLINE __device__ __forceinline__
  #define NCCL_HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#else
  #ifndef __host__
    #define __host__
  #endif
  #ifndef __device__
    #define __device__
  #endif
  #define NCCL_DEVICE_INLINE
  #define NCCL_HOST_DEVICE_INLINE inline __attribute__((always_inline))
#endif

////////////////////////////////////////////////////////////////////////////////
// Architecture detection
//
// NCCL_DEVICE_ARCH: Non-zero when compiling for device
// Use this instead of __CUDA_ARCH__ or __HIP_DEVICE_COMPILE__
////////////////////////////////////////////////////////////////////////////////

#if defined(__CUDA_ARCH__)
  #define NCCL_DEVICE_ARCH __CUDA_ARCH__
#elif defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
  // Map HIP GFX versions to a comparable value
  // MI200 (gfx90a) and MI300 (gfx942) are roughly Hopper-class
  #if defined(__gfx942__) || defined(__gfx950__)
    #define NCCL_DEVICE_ARCH 942  // MI300 class
  #elif defined(__gfx90a__)
    #define NCCL_DEVICE_ARCH 90   // MI200 class
  #elif defined(__gfx908__)
    #define NCCL_DEVICE_ARCH 80   // MI100 class
  #else
    #define NCCL_DEVICE_ARCH 70   // Generic GCN
  #endif
#else
  #define NCCL_DEVICE_ARCH 0
#endif

// Hopper+ features (multimem, etc.) - NVIDIA only
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  #define NCCL_ARCH_HAS_MULTIMEM 1
#else
  #define NCCL_ARCH_HAS_MULTIMEM 0
#endif

////////////////////////////////////////////////////////////////////////////////
// Wave/Warp size abstraction
////////////////////////////////////////////////////////////////////////////////

#if defined(NCCL_HIP_PLATFORM)
  // AMD GPUs use 64-wide waves (or 32 in wave32 mode)
  #if defined(__GFX10__) || defined(__GFX11__) || defined(__gfx1100__) || \
      defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1200__) || defined(__gfx1201__)
    #define NCCL_WARP_SIZE 32
  #else
    #define NCCL_WARP_SIZE 64
  #endif
#else
  #define NCCL_WARP_SIZE 32
#endif

////////////////////////////////////////////////////////////////////////////////
// Memory ordering types for atomic operations
//
// These provide a unified interface matching cuda::memory_order
////////////////////////////////////////////////////////////////////////////////

#if NCCL_DEVICE_COMPILE

#if defined(NCCL_HIP_PLATFORM)

namespace nccl_hip {

// Memory order enumeration matching cuda::memory_order
enum memory_order {
  memory_order_relaxed = __ATOMIC_RELAXED,
  memory_order_acquire = __ATOMIC_ACQUIRE,
  memory_order_release = __ATOMIC_RELEASE,
  memory_order_acq_rel = __ATOMIC_ACQ_REL,
  memory_order_seq_cst = __ATOMIC_SEQ_CST
};

// Thread scope enumeration
enum thread_scope {
  thread_scope_thread   = 0,
  thread_scope_block    = 1,
  thread_scope_device   = 2,
  thread_scope_system   = 3
};

// Map thread_scope to HIP memory scope
NCCL_DEVICE_INLINE constexpr int toHipMemoryScope(thread_scope scope) {
  switch (scope) {
    case thread_scope_thread: return __HIP_MEMORY_SCOPE_SINGLETHREAD;
    case thread_scope_block:  return __HIP_MEMORY_SCOPE_WORKGROUP;
    case thread_scope_device: return __HIP_MEMORY_SCOPE_AGENT;
    case thread_scope_system: return __HIP_MEMORY_SCOPE_SYSTEM;
    default:                  return __HIP_MEMORY_SCOPE_SYSTEM;
  }
}

////////////////////////////////////////////////////////////////////////////////
// atomic_ref implementation for HIP
//
// Provides cuda::atomic_ref-compatible interface using HIP atomics
////////////////////////////////////////////////////////////////////////////////

template<typename T, thread_scope Scope = thread_scope_system>
struct atomic_ref {
  T* ptr;

  NCCL_DEVICE_INLINE explicit atomic_ref(T& ref) : ptr(&ref) {}

  NCCL_DEVICE_INLINE void store(T val, memory_order order = memory_order_seq_cst) const {
    if constexpr (sizeof(T) == 4) {
      __hip_atomic_store(reinterpret_cast<unsigned int*>(ptr),
                         *reinterpret_cast<unsigned int*>(&val),
                         order, toHipMemoryScope(Scope));
    } else if constexpr (sizeof(T) == 8) {
      __hip_atomic_store(reinterpret_cast<unsigned long long*>(ptr),
                         *reinterpret_cast<unsigned long long*>(&val),
                         order, toHipMemoryScope(Scope));
    } else {
      __atomic_store_n(ptr, val, order);
    }
  }

  NCCL_DEVICE_INLINE T load(memory_order order = memory_order_seq_cst) const {
    T result;
    if constexpr (sizeof(T) == 4) {
      unsigned int tmp = __hip_atomic_load(reinterpret_cast<unsigned int*>(ptr),
                                           order, toHipMemoryScope(Scope));
      result = *reinterpret_cast<T*>(&tmp);
    } else if constexpr (sizeof(T) == 8) {
      unsigned long long tmp = __hip_atomic_load(reinterpret_cast<unsigned long long*>(ptr),
                                                  order, toHipMemoryScope(Scope));
      result = *reinterpret_cast<T*>(&tmp);
    } else {
      result = __atomic_load_n(ptr, order);
    }
    return result;
  }

  NCCL_DEVICE_INLINE T fetch_add(T val, memory_order order = memory_order_seq_cst) const {
    if constexpr (sizeof(T) == 4) {
      return __hip_atomic_fetch_add(ptr, val, order, toHipMemoryScope(Scope));
    } else if constexpr (sizeof(T) == 8) {
      return __hip_atomic_fetch_add(ptr, val, order, toHipMemoryScope(Scope));
    } else {
      return __atomic_fetch_add(ptr, val, order);
    }
  }
};

// __builtin_amdgcn_fence requires compile-time constant arguments, so we
// dispatch order x scope with a helper macro instead of a runtime switch.
#define NCCL_HIP_FENCE_SCOPE(ORD, scope) do { \
    switch (scope) { \
      case thread_scope_thread:  __atomic_signal_fence(ORD); break; \
      case thread_scope_block:   __builtin_amdgcn_fence(ORD, "workgroup"); break; \
      case thread_scope_device:  __builtin_amdgcn_fence(ORD, "agent"); break; \
      case thread_scope_system: default: __builtin_amdgcn_fence(ORD, ""); break; \
    } \
  } while(0)

NCCL_DEVICE_INLINE void atomic_thread_fence(memory_order order, thread_scope scope = thread_scope_device) {
  switch (order) {
    case memory_order_relaxed: break;
    case memory_order_acquire: NCCL_HIP_FENCE_SCOPE(__ATOMIC_ACQUIRE, scope); break;
    case memory_order_release: NCCL_HIP_FENCE_SCOPE(__ATOMIC_RELEASE, scope); break;
    case memory_order_acq_rel: NCCL_HIP_FENCE_SCOPE(__ATOMIC_ACQ_REL, scope); break;
    case memory_order_seq_cst: default: NCCL_HIP_FENCE_SCOPE(__ATOMIC_SEQ_CST, scope); break;
  }
}

#undef NCCL_HIP_FENCE_SCOPE

} // namespace nccl_hip

// Bring into cuda namespace for source compatibility
namespace cuda {
  using nccl_hip::memory_order;
  using nccl_hip::memory_order_relaxed;
  using nccl_hip::memory_order_acquire;
  using nccl_hip::memory_order_release;
  using nccl_hip::memory_order_acq_rel;
  using nccl_hip::memory_order_seq_cst;
  using nccl_hip::thread_scope;
  using nccl_hip::thread_scope_thread;
  using nccl_hip::thread_scope_block;
  using nccl_hip::thread_scope_device;
  using nccl_hip::thread_scope_system;
  using nccl_hip::atomic_ref;
  using nccl_hip::atomic_thread_fence;
}

#else // CUDA platform

// Include CUDA's atomic header when available
// #include <cuda/atomic>  // Uncomment when CUDA 11+ is required

#endif // NCCL_HIP_PLATFORM

#endif // NCCL_DEVICE_COMPILE

////////////////////////////////////////////////////////////////////////////////
// Warp/Wave synchronization primitives
////////////////////////////////////////////////////////////////////////////////

#if NCCL_DEVICE_COMPILE

#if defined(NCCL_HIP_PLATFORM)

// Lane ID - get the lane index within the wave
NCCL_DEVICE_INLINE int nccl_lane_id() {
  return __lane_id();
}

// Lane mask less than - lower 32 bits of the mask of lanes below current.
// For wave64 lanes 32-63, all lower 32 lanes are below -> 0xFFFFFFFF.
// A 64-bit variant would be needed for full wave64 cooperative groups.
NCCL_DEVICE_INLINE unsigned int nccl_lanemask_lt() {
  int lane = __lane_id();
  #if NCCL_WARP_SIZE == 64
    if (lane >= 32) return 0xFFFFFFFFu;
    return (1u << lane) - 1;
  #else
    return (1u << lane) - 1;
  #endif
}

// Warp sync - AMD CDNA waves execute in lockstep (no divergence-convergence
// like post-Volta NVIDIA), so explicit sync is unnecessary.  Matches the
// RCCL convention in common.h (#define __syncwarp()).
NCCL_DEVICE_INLINE void nccl_syncwarp(unsigned int mask = 0xffffffff) {
  (void)mask;
}

// Active mask - get mask of active lanes
NCCL_DEVICE_INLINE unsigned int nccl_activemask() {
  // __ballot(1) returns mask of lanes where predicate is true
  // Since we pass 1 (true), returns mask of all active lanes
  return __ballot(1);
}

// Named barrier with count.  CUDA has __barrier_sync_count(id, count) for
// sub-block synchronization.  AMD GCN only has s_barrier (full-block).
// __syncthreads() is an overly-conservative but safe fallback.  For a
// software multi-warp barrier, see barrier_generic in primitives.h.
NCCL_DEVICE_INLINE void nccl_barrier_sync_count(int id, int count) {
  (void)id;
  (void)count;
  __syncthreads();
}

// Population count - count set bits
NCCL_DEVICE_INLINE int nccl_popc(unsigned int x) {
  return __popc(x);
}

// Population count - 64-bit version
NCCL_DEVICE_INLINE int nccl_popcll(unsigned long long x) {
  return __popcll(x);
}

////////////////////////////////////////////////////////////////////////////////
// Wave64-compatible warp intrinsic wrappers
//
// CUDA warp intrinsics take an explicit lane mask; AMD CDNA waves execute in
// lockstep so the mask is implicit.  These wrappers let NCCL source use the
// CUDA signatures unchanged.
////////////////////////////////////////////////////////////////////////////////

NCCL_DEVICE_INLINE int __shfl_sync(unsigned int mask, int val, int srcLane, int width = NCCL_WARP_SIZE) {
  (void)mask;
  return __shfl(val, srcLane, width);
}
NCCL_DEVICE_INLINE unsigned int __shfl_sync(unsigned int mask, unsigned int val, int srcLane, int width = NCCL_WARP_SIZE) {
  (void)mask;
  return __shfl(val, srcLane, width);
}
NCCL_DEVICE_INLINE float __shfl_sync(unsigned int mask, float val, int srcLane, int width = NCCL_WARP_SIZE) {
  (void)mask;
  return __shfl(val, srcLane, width);
}

NCCL_DEVICE_INLINE unsigned long long __ballot_sync(unsigned int mask, int pred) {
  (void)mask;
  return __ballot(pred);
}

// __syncwarp(mask) — CUDA warp sync with lane mask.
// AMD CDNA waves are lockstep; this is a no-op.  common.h defines
// __syncwarp() (zero-arg macro) but coop.h needs the 1-arg form.
NCCL_DEVICE_INLINE void __syncwarp(unsigned int mask) {
  (void)mask;
}

// __activemask — HIP provides this natively (amd_warp_functions.h)
// __fns — HIP provides this natively (amd_device_functions.h)

// __barrier_sync_count — CUDA named barrier; HIP only has s_barrier.
NCCL_DEVICE_INLINE void __barrier_sync_count(int id, int count) {
  (void)id; (void)count;
  __syncthreads();
}

#else // CUDA platform

NCCL_DEVICE_INLINE int nccl_lane_id() {
  int ret;
  asm("mov.u32 %0, %%laneid;" : "=r"(ret));
  return ret;
}

NCCL_DEVICE_INLINE unsigned int nccl_lanemask_lt() {
  unsigned int ret;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(ret));
  return ret;
}

NCCL_DEVICE_INLINE void nccl_syncwarp(unsigned int mask = 0xffffffff) {
  __syncwarp(mask);
}

NCCL_DEVICE_INLINE unsigned int nccl_activemask() {
  return __activemask();
}

NCCL_DEVICE_INLINE void nccl_barrier_sync_count(int id, int count) {
  __barrier_sync_count(id, count);
}

NCCL_DEVICE_INLINE int nccl_popc(unsigned int x) {
  return __popc(x);
}

NCCL_DEVICE_INLINE int nccl_popcll(unsigned long long x) {
  return __popcll(x);
}

#endif // NCCL_HIP_PLATFORM

#endif // NCCL_DEVICE_COMPILE

////////////////////////////////////////////////////////////////////////////////
// Math intrinsics
////////////////////////////////////////////////////////////////////////////////

#if NCCL_DEVICE_COMPILE

#if defined(NCCL_HIP_PLATFORM)

NCCL_DEVICE_INLINE uint32_t nccl_umulhi(uint32_t a, uint32_t b) {
  return __umulhi(a, b);
}

NCCL_DEVICE_INLINE uint64_t nccl_umul64hi(uint64_t a, uint64_t b) {
  return __umul64hi(a, b);
}

#else // CUDA

NCCL_DEVICE_INLINE uint32_t nccl_umulhi(uint32_t a, uint32_t b) {
  return __umulhi(a, b);
}

NCCL_DEVICE_INLINE uint64_t nccl_umul64hi(uint64_t a, uint64_t b) {
  return __umul64hi(a, b);
}

#endif // NCCL_HIP_PLATFORM

#else // Host fallbacks

inline uint32_t nccl_umulhi(uint32_t a, uint32_t b) {
  return uint64_t(a) * b >> 32;
}

inline uint64_t nccl_umul64hi(uint64_t a, uint64_t b) {
  return (uint64_t)(((unsigned __int128)a) * b >> 64);
}

#endif // NCCL_DEVICE_COMPILE

////////////////////////////////////////////////////////////////////////////////
// Memory operations
////////////////////////////////////////////////////////////////////////////////

#if NCCL_DEVICE_COMPILE

#if defined(NCCL_HIP_PLATFORM)

// Load through constant cache (ldg) - HIP doesn't have direct equivalent
// Just use regular load
template<typename T>
NCCL_DEVICE_INLINE T nccl_ldg(const T* ptr) {
  return *ptr;
}

// 128-bit volatile vector loads/stores for LL protocol.
//
// DWORDX4 detection replicated from rccl_ptr.h (hip_compat.h must be
// self-contained and cannot include rccl_ptr.h).
#if defined(__HIP_DEVICE_COMPILE__)
  #if (defined(__gfx942__) || defined(__gfx950__)) \
      && __has_builtin(__builtin_amdgcn_global_load_b128) \
      && __has_builtin(__builtin_amdgcn_global_store_b128) \
      && !defined(DWORDX4_INTRINSICS_FORCE_OFF)
    #define NCCL_COMPAT_HAVE_DWORDX4 1
  #else
    #define NCCL_COMPAT_HAVE_DWORDX4 0
  #endif
#else
  #define NCCL_COMPAT_HAVE_DWORDX4 0
#endif

typedef __attribute__((__vector_size__(4 * sizeof(unsigned int)))) unsigned int nccl_compat_v4u;
typedef __attribute__((address_space(1))) nccl_compat_v4u* nccl_compat_v4u_gptr;

NCCL_DEVICE_INLINE void nccl_st_volatile_v4_u32(void* ptr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
  union { nccl_compat_v4u vec; uint32_t u32[4]; } u;
  u.u32[0] = v0; u.u32[1] = v1; u.u32[2] = v2; u.u32[3] = v3;
  #if NCCL_COMPAT_HAVE_DWORDX4
    __builtin_amdgcn_global_store_b128((nccl_compat_v4u_gptr)ptr, u.vec, "");
  #else
    typedef __attribute__((address_space(1))) uint64_t* u64_gptr_t;
    uint64_t* p64 = reinterpret_cast<uint64_t*>(ptr);
    __builtin_nontemporal_store(*reinterpret_cast<uint64_t*>(&u.u32[0]), (u64_gptr_t)p64);
    __builtin_nontemporal_store(*reinterpret_cast<uint64_t*>(&u.u32[2]), (u64_gptr_t)(p64 + 1));
  #endif
}

NCCL_DEVICE_INLINE void nccl_ld_volatile_v4_u32(const void* ptr, uint32_t& v0, uint32_t& v1, uint32_t& v2, uint32_t& v3) {
  union { nccl_compat_v4u vec; uint32_t u32[4]; } u;
  #if NCCL_COMPAT_HAVE_DWORDX4
    u.vec = __builtin_amdgcn_global_load_b128((nccl_compat_v4u_gptr)ptr, "");
  #else
    typedef __attribute__((address_space(1))) uint64_t* u64_gptr_t;
    const uint64_t* p64 = reinterpret_cast<const uint64_t*>(ptr);
    *reinterpret_cast<uint64_t*>(&u.u32[0]) = __builtin_nontemporal_load((u64_gptr_t)p64);
    *reinterpret_cast<uint64_t*>(&u.u32[2]) = __builtin_nontemporal_load((u64_gptr_t)(p64 + 1));
  #endif
  v0 = u.u32[0]; v1 = u.u32[1]; v2 = u.u32[2]; v3 = u.u32[3];
}

// GPU-scope (agent) acquire fence
NCCL_DEVICE_INLINE void nccl_fence_acq_gpu() {
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
}

// GPU-scope (agent) release fence
NCCL_DEVICE_INLINE void nccl_fence_rel_gpu() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
}

// Multimem reduction - NVLINK multicast, NO HIP EQUIVALENT
// These are stubs that will compile but do nothing
NCCL_DEVICE_INLINE void nccl_multimem_red_release_add_u32(void* ptr) {
  // NO HIP EQUIVALENT: Requires NVLINK multicast hardware
  (void)ptr;
}

NCCL_DEVICE_INLINE void nccl_multimem_red_relaxed_add_u32(void* ptr) {
  // NO HIP EQUIVALENT: Requires NVLINK multicast hardware
  (void)ptr;
}

#else // CUDA platform

template<typename T>
NCCL_DEVICE_INLINE T nccl_ldg(const T* ptr) {
  return __ldg(ptr);
}

NCCL_DEVICE_INLINE void nccl_st_volatile_v4_u32(void* ptr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
  asm volatile("st.volatile.v4.u32 [%0],{%1,%2,%3,%4};" ::
    "l"(ptr), "r"(v0), "r"(v1), "r"(v2), "r"(v3));
}

NCCL_DEVICE_INLINE void nccl_ld_volatile_v4_u32(const void* ptr, uint32_t& v0, uint32_t& v1, uint32_t& v2, uint32_t& v3) {
  asm volatile("ld.volatile.v4.u32 {%0,%1,%2,%3},[%4];"
    : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3)
    : "l"(ptr));
}

NCCL_DEVICE_INLINE void nccl_fence_acq_gpu() {
  static __device__ int dummy;
  int tmp;
  asm volatile("ld.acquire.gpu.s32 %0,[%1];" : "=r"(tmp) : "l"(&dummy) : "memory");
  dummy = tmp;
}

NCCL_DEVICE_INLINE void nccl_fence_rel_gpu() {
  cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_device);
}

NCCL_DEVICE_INLINE void nccl_multimem_red_release_add_u32(void* ptr) {
  #if __CUDA_ARCH__ >= 900
    asm volatile("multimem.red.release.sys.add.u32 [%0],1;" :: "l"(ptr));
  #else
    (void)ptr;
  #endif
}

NCCL_DEVICE_INLINE void nccl_multimem_red_relaxed_add_u32(void* ptr) {
  #if __CUDA_ARCH__ >= 900
    asm volatile("multimem.red.relaxed.sys.add.u32 [%0],1;" :: "l"(ptr));
  #else
    (void)ptr;
  #endif
}

#endif // NCCL_HIP_PLATFORM

#endif // NCCL_DEVICE_COMPILE

////////////////////////////////////////////////////////////////////////////////
// Diagnostic macros for NYI features
////////////////////////////////////////////////////////////////////////////////

#if defined(NCCL_HIP_PLATFORM) && NCCL_DEVICE_COMPILE
  // Mark features that are not yet implemented for HIP
  #define NCCL_HIP_NYI(feature) \
    do { /* NYI: feature */ } while(0)

  // Warning for features that have no HIP equivalent
  #define NCCL_HIP_NO_EQUIVALENT(feature) \
    do { /* NO HIP EQUIVALENT: feature */ } while(0)
#else
  #define NCCL_HIP_NYI(feature) do {} while(0)
  #define NCCL_HIP_NO_EQUIVALENT(feature) do {} while(0)
#endif

#endif // _NCCL_DEVICE_HIP_COMPAT_H_
