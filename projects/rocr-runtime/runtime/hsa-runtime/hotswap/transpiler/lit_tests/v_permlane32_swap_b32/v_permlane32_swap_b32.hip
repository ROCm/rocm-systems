// ARCH: gfx950
//
// Lift fixture for the gfx950-only `v_permlane32_swap_b32` cross-
// lane swap — the XOR-32 sibling of `v_permlane16_swap_b32`.  The
// instruction's hardware semantics (swap VGPR contents between
// lanes `L` and `L XOR 32` for every lane in the wave64) have no
// native gfx942 equivalent: `FeaturePermlane32Swap` is gated on
// GFX950Insts in llvm/lib/Target/AMDGPU/AMDGPU.td; gfx940/gfx942
// do not enable it.  The raiser's
// `handle_valu_cross_lane.cpp::V_PERMLANE32_SWAP_B32` handler
// therefore emulates it via `ds_bpermute(laneId XOR 32, vdst_in)`
// / `ds_bpermute(laneId XOR 32, src0_in)` — structurally
// identical to the XOR-16 arm but with the partner mask widened
// to 32.  This fixture pins the emulation's lifted IR shape so a
// future refactor that breaks the XOR-32 pattern, the byte-
// address shift, or the two-bpermute convergence guarantee fails
// here rather than silently at an AITER end-to-end regression.
//
// Pinned via HIP inline asm against a gfx950 build because:
//   1. hipcc has no language-level spelling that lowers directly
//      to `v_permlane32_swap_b32` — the `__shfl_xor` intrinsic
//      family targets `ds_bpermute` / `ds_permute`, not the
//      scalar-permlane family.
//   2. The MCInst shape we need the handler to recognise is
//      exactly what the gfx950 assembler emits for the
//      `v_permlane32_swap_b32_e32 vA, vB` two-VGPR form (the
//      shape AITER's `fmha_v3_fwd/fwd_hd128_bf16*` kernels
//      issue — verified via `llvm-objdump -d --mcpu=gfx950` on
//      `fwd_hd128_bf16.co`).
//
// The `"+v"` in/out constraints bind %[a] and %[b] to VGPRs
// that are both read AND written (TableGen's VOP_PERMLANE_SWAP
// profile ties vdst↔vdst_in and src0↔src0_out — both VGPRs get
// swapped in-place).

#include <hip/hip_runtime.h>

extern "C" __global__ void v_permlane32_swap_b32_kernel(
    unsigned *out_a, unsigned *out_b) {
  // One workgroup, 64 lanes (one wave64 on gfx950).  Each lane
  // seeds its two VGPRs with identifiable values (a = lane,
  // b = lane + 1000) so the companion .ll can point at the
  // pls32_partner / pls32_addr identifiers without binding to
  // any numerical runtime result (which would require hardware
  // availability to verify).  The fixture is purely IR-shape
  // regression coverage; end-to-end numerical coverage for this
  // handler lives in the AITER batch-raise path
  // (tests/batch_raise_test.cpp::BatchRaise.AiterGfx950), which
  // lifts 3 real fmha_v3_fwd kernels that depend on this arm.
  unsigned lane = threadIdx.x;
  unsigned a = lane;
  unsigned b = lane + 1000;
  asm volatile(
      "v_permlane32_swap_b32 %[a], %[b]"
      : [a] "+v"(a), [b] "+v"(b));
  out_a[lane] = a;
  out_b[lane] = b;
}
