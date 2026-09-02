// AIRUNTIME-28: blit copy kernel variants for the non-temporal investigation.
//
// Every variant below is a faithful transcription of __amd_rocclr_copyBuffer
// (clr/rocclr/device/blitcl.cpp) -- same argument list, same grid-stride loop,
// same end_ptr termination, same scalar tail. The only things that change
// between variants are the element width and how the load/store is issued.
//
// Each variant also declares the instruction its store must lower to. That
// expectation is data here rather than a pattern in a shell script, so a compiler
// change that silently stops emitting a non-temporal hint fails the ISA check
// instead of turning the experiment into a comparison of two identical kernels.
#ifndef AIRUNTIME28_VARIANTS_H_
#define AIRUNTIME28_VARIANTS_H_

#include <hip/hip_runtime.h>

#include <cstdio>

#include "config.h"

namespace bench {

// ---------------------------------------------------------------------------
// Access modes
// ---------------------------------------------------------------------------
enum AccessMode {
  kPlain = 0,      // regular load, regular store             (production)
  kNtLoadStore,    // NT load, NT store                       (TH_LOAD_NT + TH_STORE_NT)
  kNtStoreOnly,    // regular load, NT store                  (TH_STORE_NT)
  kNtRtStoreOnly,  // regular load, NT-near/regular-far store (TH_STORE_NT_RT, inline asm)
  // Control for kNtRtStoreOnly: identical inline-asm mechanism, but carrying the
  // DEFAULT temporal hint. Any difference between this and kPlain is caused by
  // hand-writing the store rather than by the temporal hint. Without it, a
  // codegen artefact is indistinguishable from a real cache effect.
  kAsmRtStoreOnly,
};

template <AccessMode M, typename T>
__device__ __forceinline__ T blitLoad(const T* p) {
  if constexpr (M == kNtLoadStore) {
    return __builtin_nontemporal_load(p);
  } else {
    return *p;
  }
}

// TH_STORE_NT_RT is not reachable from __builtin_nontemporal_store, which only
// ever emits plain TH_NT, so the only way to measure it is to write the
// instruction by hand.
//
// In the two-part TH names the first part applies to the near caches and the
// second to the last level, so NT_RT is non-temporal in L0/GL1 and *regular in
// GL2* - the line still lands in GL2 and stays there for a later consumer. That
// is the opposite of what the shipped change does, and it is why this variant
// shows no cache benefit once the codegen control is subtracted: it never
// stopped polluting GL2 in the first place.
//
// (An earlier version of this comment said NT_RT avoids polluting L2. It does
// not, and the measurement agrees: -0.29% against asm-plain-128, i.e. nothing.)
__device__ __forceinline__ void storeB128NtRt(u64x2* p, u64x2 v) {
  asm volatile("global_store_b128 %0, %1, off th:TH_STORE_NT_RT" : : "v"(p), "v"(v) : "memory");
}
__device__ __forceinline__ void storeB64NtRt(u64* p, u64 v) {
  asm volatile("global_store_b64 %0, %1, off th:TH_STORE_NT_RT" : : "v"(p), "v"(v) : "memory");
}
__device__ __forceinline__ void storeB32NtRt(u32* p, u32 v) {
  asm volatile("global_store_b32 %0, %1, off th:TH_STORE_NT_RT" : : "v"(p), "v"(v) : "memory");
}

// Same asm shape, default temporal hint. The codegen control.
__device__ __forceinline__ void storeB128Rt(u64x2* p, u64x2 v) {
  asm volatile("global_store_b128 %0, %1, off th:TH_STORE_RT" : : "v"(p), "v"(v) : "memory");
}
__device__ __forceinline__ void storeB64Rt(u64* p, u64 v) {
  asm volatile("global_store_b64 %0, %1, off th:TH_STORE_RT" : : "v"(p), "v"(v) : "memory");
}
__device__ __forceinline__ void storeB32Rt(u32* p, u32 v) {
  asm volatile("global_store_b32 %0, %1, off th:TH_STORE_RT" : : "v"(p), "v"(v) : "memory");
}

template <AccessMode M, typename T>
__device__ __forceinline__ void blitStore(T* p, T v) {
  if constexpr (M == kNtLoadStore || M == kNtStoreOnly) {
    __builtin_nontemporal_store(v, p);
  } else if constexpr (M == kNtRtStoreOnly) {
    if constexpr (sizeof(T) == 16) {
      storeB128NtRt(reinterpret_cast<u64x2*>(p), *reinterpret_cast<u64x2*>(&v));
    } else if constexpr (sizeof(T) == 8) {
      storeB64NtRt(reinterpret_cast<u64*>(p), *reinterpret_cast<u64*>(&v));
    } else {
      storeB32NtRt(reinterpret_cast<u32*>(p), *reinterpret_cast<u32*>(&v));
    }
  } else if constexpr (M == kAsmRtStoreOnly) {
    if constexpr (sizeof(T) == 16) {
      storeB128Rt(reinterpret_cast<u64x2*>(p), *reinterpret_cast<u64x2*>(&v));
    } else if constexpr (sizeof(T) == 8) {
      storeB64Rt(reinterpret_cast<u64*>(p), *reinterpret_cast<u64*>(&v));
    } else {
      storeB32Rt(reinterpret_cast<u32*>(p), *reinterpret_cast<u32*>(&v));
    }
  } else {
    *p = v;
  }
}

// ---------------------------------------------------------------------------
// The blit kernel itself
// ---------------------------------------------------------------------------
// Transcribed from __amd_rocclr_copyBuffer. ElemT selects the wide path width
// (u64x2 == the production 128-bit path, u64 == PR 2616's "NT64" path). The
// narrow fallback path stays uint in every variant, exactly as production does.
//
// The aligned_size argument is deliberately kept as a runtime value rather than
// folded away, because which branch it selects is itself part of what is under
// test: PR 2616's kernel tests `aligned_size == sizeof(ulong)` while the host
// still passes kMaxAlignment (16), so it never takes its own wide path.
template <typename ElemT, AccessMode M>
__global__ void blitCopyBuffer(const u8* src, u8* dst, u64 size, u32 remainder, u32 aligned_size,
                               u64 end_ptr, u32 next_chunk, u32 workgroup_size) {
  u32 l = threadIdx.x;
  u32 g = blockIdx.x;
  u64 id = (static_cast<u64>(g) * workgroup_size + l);
  u64 id_remainder = id;

  if (aligned_size == sizeof(ElemT)) {
    const ElemT* srcD = reinterpret_cast<const ElemT*>(src);
    ElemT* dstD = reinterpret_cast<ElemT*>(dst);
    while (reinterpret_cast<u64>(&dstD[id]) < end_ptr) {
      blitStore<M, ElemT>(&dstD[id], blitLoad<M, ElemT>(&srcD[id]));
      id += next_chunk;
    }
  } else {
    const u32* srcD = reinterpret_cast<const u32*>(src);
    u32* dstD = reinterpret_cast<u32*>(dst);
    while (reinterpret_cast<u64>(&dstD[id]) < end_ptr) {
      blitStore<M, u32>(&dstD[id], blitLoad<M, u32>(&srcD[id]));
      id += next_chunk;
    }
  }
  if ((remainder != 0) && (id_remainder == 0)) {
    for (u64 i = size - remainder; i < size; ++i) {
      dst[i] = src[i];
    }
  }
}

// ---------------------------------------------------------------------------
// Variant table
// ---------------------------------------------------------------------------
enum VariantId {
  CopyPlain128 = 0,     // ulong2, plain load + plain store -- exactly __amd_rocclr_copyBuffer
  CopyNtBoth128,        // ulong2, NT load + NT store
  CopyNtStore128,       // ulong2, plain load + NT store            <- the shipped change
  CopyNtStore64,        // ulong,  plain load + NT store            -- PR 2616 as *intended*
  CopyPlain64,          // ulong,  plain load + plain store         -- width control for NtStore64
  CopyNtRtStore128,     // ulong2, plain load + TH_STORE_NT_RT store (inline asm)
  CopyPr2616Actual32,   // PR 2616 as *written*: kernel tests ==8, host passes 16, so uint runs
  CopyPlain32,          // uint,   plain load + plain store         -- width control for Pr2616Actual32
  CopyAsmPlain128,      // ulong2, inline-asm store w/ DEFAULT hint -- codegen control for NtRtStore128
  kNumVariants
};

struct VariantInfo {
  const char* name;
  const char* description;
  // Element size used to compute the dispatch geometry, i.e. what the host
  // believes the access width is. This is kMaxAlignment (16) in production.
  u32 geomElemBytes;
  // The aligned_size value actually handed to the kernel.
  u32 alignedSizeArg;

  // --- what this variant must compile to -----------------------------------
  // sizeof(ElemT) for the instantiation launchVariant() picks, and the access
  // mode. Together these identify the kernel in a disassembly: the mangled name
  // carries the element type and the AccessMode value.
  u32 templateElemBytes;
  AccessMode mode;
  // The store the hot loop actually executes, and its temporal hint. A null hint
  // means the store must carry NO th: modifier. remote/isa_check.sh reads these
  // from the binary rather than repeating them, so a compiler change that stops
  // emitting a hint fails the check instead of quietly turning the experiment
  // into a comparison of two identical kernels.
  const char* expectedStoreOpcode;
  const char* expectedStoreHint;
};

static const VariantInfo kVariants[kNumVariants] = {
    {"plain-128", "ulong2 plain/plain (production baseline)", 16, 16, 16, kPlain,
     "global_store_b128", nullptr},
    {"nt-both-128", "ulong2 NT load + NT store", 16, 16, 16, kNtLoadStore, "global_store_b128",
     "th:TH_STORE_NT"},
    {"nt-store-128", "ulong2 plain load + NT store (shipped)", 16, 16, 16, kNtStoreOnly,
     "global_store_b128", "th:TH_STORE_NT"},
    {"nt-store-64", "ulong NT store (PR 2616 as intended)", 8, 8, 8, kNtStoreOnly,
     "global_store_b64", "th:TH_STORE_NT"},
    {"plain-64", "ulong plain/plain (width control)", 8, 8, 8, kPlain, "global_store_b64", nullptr},
    {"ntrt-store-128", "ulong2 plain load + TH_STORE_NT_RT", 16, 16, 16, kNtRtStoreOnly,
     "global_store_b128", "th:TH_STORE_NT_RT"},
    // Same instantiation as nt-store-64; what differs is which branch runs, and
    // the branch that runs is the narrow one. That is the PR's defect.
    {"pr2616-actual-32", "PR 2616 as written: falls through to uint NT", 16, 16, 8, kNtStoreOnly,
     "global_store_b32", "th:TH_STORE_NT"},
    {"plain-32", "uint plain/plain (width control)", 4, 4, 4, kPlain, "global_store_b32", nullptr},
    {"asm-plain-128", "ulong2 asm store, default hint (codegen control)", 16, 16, 16,
     kAsmRtStoreOnly, "global_store_b128", "th:TH_STORE_RT"},
};

// Emits the table above in a form remote/isa_check.sh can read, so the ISA
// expectations have exactly one definition.
inline void printIsaExpectations() {
  std::printf("# variant\telem_bytes\tmode\topcode\thint\n");
  for (int v = 0; v < kNumVariants; ++v) {
    const VariantInfo& vi = kVariants[v];
    std::printf("ISA\t%s\t%u\t%d\t%s\t%s\n", vi.name, vi.templateElemBytes,
                static_cast<int>(vi.mode), vi.expectedStoreOpcode,
                vi.expectedStoreHint ? vi.expectedStoreHint : "-");
  }
}

inline void launchVariant(int v, dim3 grid, dim3 block, hipStream_t stream, const u8* src, u8* dst,
                          u64 size, u32 remainder, u32 alignedSize, u64 endPtr, u32 nextChunk,
                          u32 workgroupSize) {
#define NT_LAUNCH(ELEM, MODE)                                                                    \
  hipLaunchKernelGGL((blitCopyBuffer<ELEM, MODE>), grid, block, 0, stream, src, dst, size,        \
                     remainder, alignedSize, endPtr, nextChunk, workgroupSize)

  switch (v) {
    case CopyPlain128:       NT_LAUNCH(u64x2, kPlain);           break;
    case CopyNtBoth128:      NT_LAUNCH(u64x2, kNtLoadStore);     break;
    case CopyNtStore128:     NT_LAUNCH(u64x2, kNtStoreOnly);     break;
    case CopyNtStore64:      NT_LAUNCH(u64,   kNtStoreOnly);     break;
    case CopyPlain64:        NT_LAUNCH(u64,   kPlain);           break;
    case CopyNtRtStore128:   NT_LAUNCH(u64x2, kNtRtStoreOnly);   break;
    // ElemT is u64 so the `aligned_size == 8` test fails against the 16 the
    // host passes, and control lands in the uint branch. That is the bug.
    case CopyPr2616Actual32: NT_LAUNCH(u64,   kNtStoreOnly);     break;
    case CopyPlain32:        NT_LAUNCH(u32,   kPlain);           break;
    case CopyAsmPlain128:    NT_LAUNCH(u64x2, kAsmRtStoreOnly);  break;
    default: break;
  }
#undef NT_LAUNCH
}

}  // namespace bench

#endif  // AIRUNTIME28_VARIANTS_H_
