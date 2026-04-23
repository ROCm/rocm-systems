// ARCH: gfx1250
//
// Fixture for the `amdgpu-lds-size` function-attribute propagation in
// `raiser.cpp`.  Without this attribute, the AMDGPU backend's KD
// emitter sees no module-level `addrspace(3)` `GlobalVariable`s (our
// raised IR accesses LDS via `inttoptr i64 to ptr addrspace(3)`
// raw-pointer arithmetic, not GVs) and defaults `group_segment_fixed_
// size: 0` in the lifted HSACO.  Kernels that use static LDS then
// access an unallocated segment and every LDS read returns zero.
//
// This silently miscompiled every lifted kernel with a cross-thread
// LDS fragment shuffle.  The bug was caught via `quad_wmma_cross_lds`
// in the matmul_fp16 investigation (matrix-translation.md §12.4
// bisection): my-slot-only reads returned correct data (the LDS
// elision LLVM's forward-prop does for per-thread disjoint slots
// bypassed the 0-size segment), but cross-thread reads via LDS
// returned all zeros because they actually touched the unallocated
// segment.
//
// The fix propagates the source's `.group_segment_fixed_size` into
// an `amdgpu-lds-size` function attribute; `AMDGPUMachineFunctionInfo`
// reads that attr and plumbs the value through to the emitted KD.

#include <hip/hip_runtime.h>

extern "C" __global__ __launch_bounds__(32) void
group_segment_fixed_size_attr_kernel(int *out) {
  __shared__ int lds[1024];  // 4096 bytes of static LDS
  lds[threadIdx.x] = threadIdx.x;
  __syncthreads();
  out[threadIdx.x] = lds[(threadIdx.x + 1) & 31];
}
