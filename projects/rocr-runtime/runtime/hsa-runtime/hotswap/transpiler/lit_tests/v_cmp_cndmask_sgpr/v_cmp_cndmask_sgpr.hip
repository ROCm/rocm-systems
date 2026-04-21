// ARCH: gfx1250
//
// Lift fixture for the canonical V_CMP_*_e64 -> SGPR -> V_CNDMASK_B32_e64
// per-lane-predication idiom. The compare writes a **single** wave32
// SGPR (s4), and the cndmask consumes it as a per-lane wave mask —
// lane i selects src1 iff bit i of s4 is set.
//
// This is the shape every libdevice math kernel (asin, acos, atan,
// log, exp, copysign) lowers to when it branches on a floating-point
// range, and the shape Triton epilogues reach for whenever the
// compiler chooses e64 over e32 to survive the compare across a
// basic-block boundary.
//
// Pre-fix salmon lowered the SGPR-source cndmask as
// `icmp ne <sgpr-as-integer>, 0`, collapsing the mask to one
// wave-uniform `i1` and making every lane pick the same side of the
// select — a silent miscompile that surfaces as ~50% wrong lanes on
// data-dependent predication (cndmask reducer) and ~87% wrong on
// two-stage polynomial branches (asin). The handler in
// `handle_valu_vop3p.cpp` was the asymmetric counterpart to the VCC
// consumer path (`loadVCC` -> per-lane i1): the SGPR path bypassed
// the projection's `extractLaneBitFromWaveMask` that the VCC path
// routes through via `readVCCAsWaveMask`.
//
// This fixture pins the post-fix IR shape — the per-lane extract
// chain (mask widen -> lshr by lane_id -> and 1 -> icmp ne 0) AND
// the NEGATIVE assertion that no wave-uniform `icmp ne <sgpr>, 0`
// survives. Under the wave-native projection (wave32 source on
// wave64 target) the widen step is a replication (`(v << 32) | v`);
// under modulo-replication (wave64 source on wave32 target) the
// widen is a no-op on matching widths. The fixture's RUN line
// exercises the wave-native path because that's the direction the
// Triton corpus lifts travel.
//
// Two inline-asm'd operations keep the pattern stable across hipcc
// versions: the compare is forced as e64 to an explicit SGPR (s4),
// and the cndmask is forced as e64 reading s4 rather than VCC. The
// `"+v"(val)` and the unmasked store at the end keep the result
// VGPR-live so the compiler doesn't DCE the sequence.

#include <hip/hip_runtime.h>

extern "C" __global__ void v_cmp_cndmask_sgpr_kernel(float *out,
                                                     const float *in) {
  int lane = (int)threadIdx.x;
  float val = in[lane];
  float result = 0.0f;
  asm volatile(
      // Wave32 source: V_CMP_GE_F32_e64 sdst is a single SGPR (s4).
      // abs modifier on src0 forces the VOP3 encoding so hipcc can't
      // substitute an e32 -> VCC form under us.
      "v_cmp_ge_f32_e64 s4, |%[val]|, 0.5\n\t"
      // Per-lane predicated select driven by s4. The e64 form is
      // what lets the mask source be an arbitrary SGPR; the e32
      // form would read VCC and exercise a different handler path.
      "v_cndmask_b32_e64 %[result], -1.0, 1.0, s4\n\t"
      : [result] "=v"(result)
      : [val] "v"(val)
      : "s4");
  out[lane] = result;
}
