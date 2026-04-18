#ifndef HOTSWAP_TRANSPILER_HANDLE_VALU_INTERNAL_HPP
#define HOTSWAP_TRANSPILER_HANDLE_VALU_INTERNAL_HPP

#include "raise_context.hpp"

// Sub-handlers that together make up `handleVALU`. Each returns
// `{handled=true}` when it recognised and lowered the SemOp, or an
// unhandled HandlerResult when the SemOp is out of its scope (so the
// top-level router can try the next sub-handler).
//
// Sub-handlers are private to the handle_valu.* translation units;
// they are not exposed to the format dispatcher in raiser.cpp.

namespace transpiler {

// Cross-lane primitives: V_READFIRSTLANE_B32, V_READLANE_B32,
// V_WRITELANE_B32, V_MBCNT_{LO,HI}_U32_B32, V_PERMLANE{16,X16,64}_B32,
// V_PERMLANE{16,32}_SWAP_B32. Isolated because the new cross-wave
// strategy (SPE_DESIGN.md §4 / CROSS_LANE_SURVEY.md) keeps iterating
// on exactly this surface.
HandlerResult handleVALU_CrossLane(RaiseContext &ctx, const DecodedInst &di,
                                    OpResolver &op);

// "Small ops": type conversions, F16 arith, 16-bit shifts / min /
// max, byte pack, V_BFREV_B32, V_NOT_B32, F32 single-src
// transcendentals. See `handle_valu_small_ops.cpp` for the exact list.
HandlerResult handleVALU_SmallOps(RaiseContext &ctx, const DecodedInst &di,
                                   OpResolver &op);

// Vector compares (V_CMP / V_CMPX collapsed onto two SemOps with
// VCmpMeta side-table lookup) including cross-wave projection of the
// ballot result back to source-EXEC width.
HandlerResult handleVALU_Vcmp(RaiseContext &ctx, const DecodedInst &di,
                               OpResolver &op);

// VOP3P packed ops (V_PK_*_F32, V_PK_MOV_B32), WMMA (V_WMMA_F32_*),
// v_fma_mix_f32, and v_cndmask_b32.
HandlerResult handleVALU_VOP3P(RaiseContext &ctx, const DecodedInst &di,
                                OpResolver &op);

} // namespace transpiler

#endif
