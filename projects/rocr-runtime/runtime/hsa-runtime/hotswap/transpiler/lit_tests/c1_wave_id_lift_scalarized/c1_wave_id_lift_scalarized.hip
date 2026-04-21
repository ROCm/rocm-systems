// ARCH: gfx1250
//
// Refusal fixture for the Class-1 "canonical wave_id lift scalarised
// through a cross-lane primitive under WMMA" obstruction
// (lit_tests/c1_wave_id_lift_scalarized). Companion to the
// c1_ttmp_wave_id_lift rescue fixture — together they pin the two
// terminal outcomes for the canonical `s_bfe_u32 sDST, ttmp8,
// 0x50019` pattern under cross-widening:
//
//   * rescue  — no v_writelane / v_readlane + no WMMA (or just one
//               of those missing): the handle_sop2.cpp lift emits
//               `(workitem.id.x >> 5) & 0x1F` directly from the
//               divergent-leaf intrinsic and the downstream SGPR
//               consumers see per-source-wave divergence. Pinned by
//               c1_ttmp_wave_id_lift.
//
//   * refuse  — this fixture. The kernel contains all three:
//                 1. the canonical `s_bfe_u32 sDST, ttmp8, 0x50019`
//                    pattern (wave_id_in_workgroup extraction),
//                 2. at least one v_writelane_b32 or v_readlane_b32
//                    (cross-lane scalar primitive),
//                 3. at least one v_wmma_* (WMMA forecloses the
//                    ThreadLoopProjection escape hatch because
//                    wave-size-translation.md §5.2's matrix shape
//                    requires the full target wave simultaneously).
//
//               The handle_sop2.cpp lift still fires and produces a
//               divergent VGPR value for the BFE destination, but
//               when that SGPR-shaped value flows into a
//               v_writelane / v_readlane scalar source operand the
//               backend inserts an implicit v_readfirstlane on it
//               (the SSA shape is formally "scalar", so codegen
//               scalarises it). That collapse erases the per-
//               source-wave distinction — source_wave[0]'s wave_id=0
//               and source_wave[1]'s wave_id=1 both become 0 in the
//               target wave's scalar operand — and the resulting
//               tile-column address arithmetic writes to the wrong
//               target-wave half. Without ThreadLoopProjection as an
//               escape hatch there is no correct projection today,
//               so classifier emits a loud refusal rather than a
//               silent miscompile.
//
// Detection logic:
//   wave_size_obstruction.cpp::buildObstructionReport tracks each of
//   the three conditions in separate vectors
//   (`canonicalWaveIdBfeSites`, `crossLaneScalarSites`, `haveWMMA`).
//   Post-loop, if all three are non-empty it emits one
//   ObstructionKind::WaveIdLiftScalarized site per
//   v_writelane / v_readlane, each mapped to
//   RaiseFailure::crossWaveLaneIdLeak by selectFailureFromReport.
//
// Implementation note. The three instructions are forced via inline
// assembly (BFE + writelane) + the WMMA builtin. hipcc on gfx1250
// would emit v_mbcnt_lo (not v_mbcnt_hi) for any HIP intrinsic-based
// lane-id computation, and its register allocator never emits
// writelane/readlane for user code — so inline asm is the only way
// to pin the shape.

#include <hip/hip_runtime.h>

typedef __bf16 v16bf16 __attribute__((ext_vector_type(16)));
typedef float  v8f     __attribute__((ext_vector_type(8)));

extern "C" __global__ void
c1_wave_id_lift_scalarized_kernel(const v16bf16 *a_ptr,
                                  const v16bf16 *b_ptr,
                                  v8f *c_ptr,
                                  int *out) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // (1) Canonical wave_id_in_workgroup extraction — the same shape
  //     every Tensile / rocBLAS matmul uses and every gfx1250 kernel
  //     the command processor launches. Bits [29:25] of ttmp8.
  int wave_id;
  asm volatile("s_bfe_u32 %[wid], ttmp8, 0x50019\n\t"
               : [wid] "=s"(wave_id));

  // (2) Cross-lane scalar primitive consuming a scalar source
  //     derived from wave_id. v_writelane's scalar input is the
  //     spot where the backend inserts the implicit readfirstlane
  //     that collapses per-source-wave divergence. We tee the
  //     result into a VGPR so the instruction is not DCE'd and the
  //     output address keeps a data dependency on it.
  int lane_slot;
  asm volatile("v_writelane_b32 %[out], %[val], 0\n\t"
               : [out] "=v"(lane_slot)
               : [val] "s"(wave_id));

  // (3) WMMA. Forecloses ThreadLoopProjection (§5.2 requires the
  //     full target wave simultaneously, which TLP's one-source-
  //     wave-at-a-time iteration cannot provide). The concrete
  //     fragment values do not matter — the classifier only checks
  //     for the SemOp's presence.
  v16bf16 a = a_ptr[0];
  v16bf16 b = b_ptr[0];
  v8f c = c_ptr[0];
  c = __builtin_amdgcn_wmma_f32_16x16x32_bf16(
      /*a_neg=*/0, a,
      /*b_neg=*/0, b,
      /*c_mod=*/0, c,
      /*a_reuse=*/false,
      /*b_reuse=*/false);
  c_ptr[0] = c;

  // Keep the BFE + writelane results live so hipcc does not DCE
  // them out of the compiled .co. Without this consumer the front
  // end's dead-store elimination would strip the inline asm whose
  // outputs it cannot see being observed.
  out[tid] = wave_id ^ lane_slot;
}
