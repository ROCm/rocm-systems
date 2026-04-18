// ARCH: gfx1250
//
// Fixture for the Class-1 "absolute lane-ID leak" obstruction
// (lit_tests/c1_lane_id_leak).
//
// SPE_DESIGN.md §3 Class 1 names the constructs through which the
// absolute target-hardware lane position leaks into observable
// values: v_mbcnt_hi_u32_b32 (the canonical abs-lane-id path on
// wave64), v_readlane / v_writelane with operand >= W_s, and the
// llvm.amdgcn.wavefrontsize / ballot intrinsics. A binary that was
// compiled for wave32 cannot contain these without tripping the
// wave-size-obliviousness theorem, because the raiser has no way to
// reconstruct "lane_id mod W_s" from the target-hardware mbcnt
// sequence without a class-specific rewrite.
//
// The ds_bpermute_b32 fix in lit_tests/ds_bpermute_b32 covers a
// different class (C2-bpermute / cross-lane rewritable). This
// fixture isolates the C1 case so the classifier's syntactic check
// has a dedicated pin.
//
// Implementation note. hipcc's normal codegen on gfx1250 uses
// v_mbcnt_lo alone (because wave32) — the v_mbcnt_hi instruction
// that actually triggers the classifier is not emitted from any HIP
// intrinsic on wave32. To pin the instruction, we force it via
// inline asm; the kernel body is otherwise an identity load/store.

#include <hip/hip_runtime.h>

extern "C" __global__ void c1_lane_id_leak_kernel(int *out) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int lane_id;
  // v_mbcnt_lo is benign on its own; v_mbcnt_hi is the C1 flag the
  // classifier matches on. Using both so the instruction sequence
  // looks like a real "compute absolute lane id" pattern.
  asm volatile(
      "v_mbcnt_lo_u32_b32 %[lane], -1, 0\n\t"
      "v_mbcnt_hi_u32_b32 %[lane], -1, %[lane]\n\t"
      : [lane] "=v"(lane_id));
  out[tid] = lane_id;
}
