// ARCH: gfx1250
//
// Fixture for the Class-2 LaneGroupShuffle positive case — the
// non-swap `v_permlane16_b32` / `v_permlanex16_b32` primitives that
// the P2 row of hotswap/docs/wave-size-translation.md §5.3 covers.
// Companion to
// `lit_tests/c2_permlane_swap` which pins the still-pending P4
// (permlane*_swap) refusal path; the two fixtures together exercise
// both branches of the per-SemOp `rewriteImplemented` dispatch in
// `wave_size_obstruction.cpp`'s LaneGroupShuffle case.
//
// Implementation approach (see handle_valu_cross_lane.cpp's
// V_PERMLANE16_B32 / V_PERMLANEX16_B32 case for the full design):
// the raiser emulates both intrinsics via `ds_bpermute_b32` rather
// than emitting `llvm.amdgcn.permlane16` / `permlanex16` directly,
// because those intrinsics are RDNA/gfx10+ only and fail isel on
// CDNA (gfx942) targets with "Cannot select: intrinsic
// %llvm.amdgcn.permlanex16". The ds_bpermute emulation is target-
// independent and works for any AMDGPU generation with LDS
// (gfx8+).
//
// The fixture uses inline assembly with an identity-permute
// selector pattern (`0x76543210` for lanes 0-7, `0xFEDCBA98` for
// lanes 8-15) so the observed behaviour on the source hardware
// would be a no-op shuffle. The lit check asserts on the
// ds_bpermute call's presence, not on the selector values —
// selector computation happens in IR and is verified structurally.

#include <hip/hip_runtime.h>

extern "C" __global__ void c2_permlane16_kernel(int *a) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int v = a[tid];
  int out16, outx16;
  // The AMDGPU assembler allows at most one *unique* literal per
  // VOP3 instruction; passing two different i32 constants as both
  // selectors is rejected. We therefore use the same literal value
  // for both within-group halves — the specific selector bits are
  // not load-bearing for the structural lit check, which asserts on
  // the emitted `llvm.amdgcn.ds.bpermute` emulation, not on the
  // selector values themselves.
  //
  // `op_sel:[1, 0]` encodes fi=1, bc=0 — the only op_sel combination
  // the emulation in `handle_valu_cross_lane.cpp` supports today
  // (per the comment there, every GPT-OSS / softmax / bitmatrix
  // disassembly uses this encoding, and the other three
  // combinations refuse with `unsupportedShape` so a future corpus
  // kernel relying on fi=0 or bc=1 surfaces loudly). Omitting
  // `op_sel:` defaults to `[0, 0]`, which would trigger the refusal
  // path.
  asm volatile(
      "v_permlane16_b32  %[out16],  %[v], 0x76543210, 0x76543210 op_sel:[1,0]\n\t"
      "v_permlanex16_b32 %[outx16], %[v], 0x76543210, 0x76543210 op_sel:[1,0]\n\t"
      : [out16] "=v"(out16), [outx16] "=v"(outx16)
      : [v] "v"(v));
  a[tid] = out16 + outx16;
}
