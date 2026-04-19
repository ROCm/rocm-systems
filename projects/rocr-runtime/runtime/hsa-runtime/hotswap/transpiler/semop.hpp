#ifndef HOTSWAP_TRANSPILER_SEMOP_HPP
#define HOTSWAP_TRANSPILER_SEMOP_HPP

#include <cstdint>

namespace transpiler {

// Architecture-neutral instruction identity used for dispatch in the raiser.
// Each entry maps to one or more MC opcodes via OpcodeMap.
enum class SemOp : uint16_t {
  Unknown = 0,

  // -- SOPP / control flow --
  S_ENDPGM, S_NOP, S_BRANCH, S_CODE_END,
  S_CBRANCH_SCC0, S_CBRANCH_SCC1,
  S_CBRANCH_VCCZ, S_CBRANCH_VCCNZ,
  S_CBRANCH_EXECZ, S_CBRANCH_EXECNZ,
  S_WAITCNT, S_WAIT_LOADCNT, S_WAIT_KMCNT, S_WAIT_DSCNT, S_WAIT_XCNT,
  S_WAIT_LOADCNT_DSCNT, S_WAIT_ALU,
  S_CLAUSE, S_DELAY_ALU, S_SET_GPR_IDX_ON, S_SET_GPR_IDX_OFF, S_SETVSKIP,
  // Barriers. GFX12+ splits s_barrier into signal + wait; earlier ISAs emit a
  // single s_barrier. Handlers model signal as a no-op and wait as a full
  // LLVM `amdgcn.s.barrier` call.
  S_BARRIER, S_BARRIER_WAIT, S_BARRIER_SIGNAL,

  // -- SMEM --
  S_LOAD_B32, S_LOAD_B64, S_LOAD_B96, S_LOAD_B128, S_LOAD_B256, S_LOAD_B512,
  S_STORE_B32, S_STORE_B64, S_STORE_B128,

  // -- SOPC --
  S_CMP_EQ_U32, S_CMP_LG_U32, S_CMP_GT_U32, S_CMP_GE_U32,
  S_CMP_LT_U32, S_CMP_LE_U32,
  // gfx8+ 64-bit unsigned scalar compares (SOPC_CMP_64). Only EQ and
  // LG (not equal) are defined in SOPInstructions.td; there are no
  // ordered/strict 64-bit SOPC compares on any AMDGPU generation
  // because the .td record `SOPC_CMP_64` is reserved for these two.
  S_CMP_EQ_U64, S_CMP_LG_U64,
  S_CMP_EQ_I32, S_CMP_LG_I32, S_CMP_GT_I32, S_CMP_GE_I32,
  S_CMP_LT_I32, S_CMP_LE_I32,
  S_CMP_EQ_F32, S_CMP_LG_F32, S_CMP_GT_F32, S_CMP_GE_F32,
  S_CMP_LT_F32, S_CMP_LE_F32, S_CMP_NEQ_F32,
  S_CMP_NGT_F32, S_CMP_NGE_F32, S_CMP_NLT_F32, S_CMP_NLE_F32, S_CMP_NLG_F32,
  S_CMP_EQ_F16, S_CMP_LG_F16, S_CMP_GT_F16, S_CMP_GE_F16,
  S_CMP_LT_F16, S_CMP_LE_F16, S_CMP_NEQ_F16,
  S_CMP_NGT_F16, S_CMP_NGE_F16, S_CMP_NLT_F16, S_CMP_NLE_F16, S_CMP_NLG_F16,

  // -- SOPK --
  S_MOVK_I32, S_ADDK_I32, S_MULK_I32,
  S_CMPK_GE_I32, S_CMPK_GT_I32, S_CMPK_LE_I32, S_CMPK_LT_I32,
  S_CMPK_GE_U32, S_CMPK_GT_U32, S_CMPK_LE_U32, S_CMPK_LT_U32,
  S_CMPK_EQ_I32, S_CMPK_EQ_U32, S_CMPK_LG_I32, S_CMPK_LG_U32,
  S_GETREG_B32, S_SETREG_B32, S_SETREG_IMM32_B32,

  // -- SOP1 --
  S_MOV_B32, S_MOV_B64, S_NOT_B32, S_NOT_B64,
  S_BREV_B32, S_FF1_I32_B32, S_FF1_I32_B64,
  // s_ff0_i32_b{32,64}: find first 0 bit (lowest position), returning
  // -1 when the source is all-ones. SOPInstructions.td:278-279 (no
  // LLVM ISel pattern is provided, so the instruction is only emitted
  // by hand-written asm / inline-asm — but the corpus contains it).
  // Lowers to `cttz(~src, is_zero_poison=false)` with a `cmov` to -1
  // on the all-ones input path, mirroring the V_FFBL_B32 / V_FFBH_U32
  // shape (the AMDGPU instruction returns 0xFFFFFFFF in the no-bit
  // case rather than the LLVM intrinsic's bitwidth-wide return).
  S_FF0_I32_B32, S_FF0_I32_B64,
  S_FLBIT_I32_B32, S_FLBIT_I32_B64,
  // s_flbit_i32 / s_flbit_i32_i64: signed find-leading-bit-not-equal-to-
  // sign-bit. Lowers to llvm.amdgcn.sffbh, the dedicated AMDGPU
  // intrinsic that selects directly back to v_ffbh_i32_e32 (or its
  // i64-split lowering for the 64-bit variant). See
  // SOPInstructions.td:296-298 / VOP1Instructions.td:373.
  S_FLBIT_I32, S_FLBIT_I32_I64,
  S_SEXT_I32_I8, S_SEXT_I32_I16,
  S_CVT_F32_U32, S_CVT_F32_I32, S_CVT_U32_F32, S_CVT_I32_F32,
  S_AND_SAVEEXEC_B32, S_OR_SAVEEXEC_B32, S_XOR_SAVEEXEC_B32,
  S_ANDN2_SAVEEXEC_B32, S_ORN2_SAVEEXEC_B32,
  S_GETPC_B64,
  // SOP1 indirect set-PC. gfx1250 asm rename for `S_SETPC_B64`
  // (SOPInstructions.td:323 declares `isBranch + isIndirectBranch`,
  // line 2208 renames the asm string to `s_set_pc_i64`). The source
  // SGPR pair holds an absolute 64-bit PC value. In our IR-on-LLVM
  // setting we model three principled lowerings (see setpc_analysis.{hpp,
  // cpp} for the static analysis that classifies each site):
  //   DirectA      — statically resolvable intra-kernel branch (the
  //                  source SGPR pair was produced by a local
  //                  `s_get_pc_i64 + s_add_co_u32 + s_add_co_ci_u32`
  //                  chain). Lowers to `br label %BB_target` since
  //                  the target is a known intra-function label.
  //   IndirectB    — subroutine return via an SGPR pair stashed at the
  //                  call site (the canonical s[30:31] return-PC
  //                  idiom). Lowers to `indirectbr ptr %ret_pc,
  //                  [list of resolved return targets]`. The
  //                  corresponding call-site
  //                  `s_get_pc_i64 + s_add*` chains are rewritten by
  //                  the raiser to materialise a
  //                  `blockaddress(@kernel, %ret_BB)` into the
  //                  ret-pair (via a post-handler hook in raiser.cpp),
  //                  so the i64 fed to indirectbr is a real LLVM
  //                  blockaddress constant rather than a binary PC.
  //   DispatchSet  — multi-target dispatch via inter-block PC-chain
  //                  dataflow: each predecessor block writes a
  //                  different chain target into the same SGPR pair,
  //                  then a join block consumes it through
  //                  `s_set_pc_i64`. The dataflow in setpc_analysis
  //                  enumerates the bounded set of targets reaching
  //                  the use site through distinct CFG paths. Lowers
  //                  to `indirectbr ptr %target, [list]`. Same chain-
  //                  terminator hook as IndirectB writes a
  //                  `blockaddress` constant on each contributing
  //                  predecessor path so the indirectbr's source is
  //                  a real LLVM constant.
  // Sites the analysis cannot resolve (incomplete dataflow,
  // unbounded fan-in past kMaxDispatchTargets, or pair killed by an
  // unmodelled write before the use site) refuse loudly via
  // RaiseFailure::unsupportedShape — never silently emit a stub.
  S_SET_PC_I64,
  // SOP1 branch-and-link. gfx1250 asm rename for `S_SWAPPC_B64`
  // (SOPInstructions.td:336 declares `isCall = 1`, line 2311 renames
  // the asm string to `s_swap_pc_i64`). Operands:
  //   sdst = sX:X+1 receives the return PC (i.e. the absolute kernel
  //          offset of the instruction immediately following the
  //          swap, swap.offset + swap.size).
  //   ssrc = sY:Y+1 holds the absolute call target PC.
  //   PC <- ssrc; sdst <- (return-PC)  (atomically)
  //
  // Three principled raisings, mirroring S_SET_PC_I64:
  //   DirectA      — call target ssrc was produced by a local
  //                  `s_get_pc_i64 + s_add_co_u32 + s_add_co_ci_u32`
  //                  chain that resolves intra-block. Lowering writes
  //                  `blockaddress(@kernel, %BB_returnAddr)` cast to
  //                  i64 into sdst and emits `br label %BB_callee`.
  //   DispatchSet  — call target reached via inter-block PC-chain
  //                  dataflow (the tensilelite "activation function
  //                  dispatcher" shape: each predecessor block
  //                  computes a distinct callee target into the same
  //                  pair via its own getpc+add chain, then a join
  //                  block executes `s_swap_pc_i64`). Lowering writes
  //                  the return-PC blockaddress into sdst as in
  //                  DirectA, then emits `indirectbr ptr ssrc, [list
  //                  of enumerated callee targets]`. The chain-
  //                  terminator hook in raiser.cpp rewrites ssrc to
  //                  hold a `blockaddress(@kernel, %BB_callee)` on
  //                  every contributing predecessor path.
  //   Unresolvable — call target cannot be statically enumerated
  //                  (incomplete dataflow, fan-in past
  //                  kMaxDispatchTargets, or runtime-derived value).
  //                  Refuse loudly via RaiseFailure::unsupportedShape
  //                  — never emit a stub branch.
  //
  // The analysis never produces IndirectB for a swap_pc site (a
  // swap_pc's source pair is the call target, not a return slot;
  // IndirectB describes the return-side use of such a pair).
  //
  // Independent of the call-target classification, the analysis
  // registers a synthetic chain-terminator at the swap site itself
  // (key = swap.offset, value = {sdst-low-reg, swap.offset+swap.size})
  // so any downstream IndirectB `s_set_pc_i64` reading sdst
  // enumerates the swap's return offset as one of its indirectbr
  // targets.
  S_SWAP_PC_I64,
  S_ABS_I32,
  S_SET_VGPR_MSB,
  // Read-modify-write bit set/clear on an SGPR. Tied src keeps the
  // un-touched bits of the destination register alive across the op.
  // B64 variants index into 64 bits (bit index is still an SReg_32).
  S_BITSET0_B32, S_BITSET1_B32,
  S_BITSET0_B64, S_BITSET1_B64,
  // Conditional move on SCC. `if (SCC) sdst = src; else sdst stays
  // unchanged.` The dst-on-SCC=0 read-modify is NOT modeled by LLVM
  // as a tied sdst_in operand on the MCInst (SOP1_32/SOP1_64 just
  // declares `(outs sdst), (ins src0)`), so the handler must
  // explicitly read the prior dst value via
  // `ctx.regs.readReg{32,64}(op.dst())`. SCC is read but not
  // written.
  S_CMOV_B32, S_CMOV_B64,

  // -- SOP2 --
  S_ADD_U32, S_ADDC_U32, S_SUB_U32, S_SUBB_U32, S_ADD_U64,
  S_AND_B32, S_AND_B64, S_OR_B32, S_OR_B64, S_XOR_B32, S_XOR_B64,
  S_ANDN2_B32, S_ANDN2_B64, S_ORN2_B32, S_ORN2_B64,
  // SOP2 negated bitops (gfx7+). SOPInstructions.td:789-803 — each
  // computes `dst = ~(src0 OP src1)` and sets SCC = (result != 0). These
  // are produced heavily by triton/tensilelite when constant-folding
  // bitfield masks (e.g. `s_nand_b32 sX, sY, 0xffff` to clear the low
  // 16 bits). All can target EXEC, so they must be marked
  // routesExecThroughStoreExec.
  S_NAND_B32, S_NAND_B64, S_NOR_B32, S_NOR_B64, S_XNOR_B32, S_XNOR_B64,
  // SOP2 absolute-difference (gfx7+). SOPInstructions.td:886-888 —
  // `dst = |src0 - src1|` on signed i32, SCC = (result != 0). Lower
  // through llvm.abs.i32 with is_int_min_poison=false: hardware wraps
  // for INT_MIN (the only value whose negation equals itself), so we
  // mustn't poison there. Heavily used by tensilelite for stride math.
  S_ABSDIFF_I32,
  S_LSHL_B32, S_LSHL_B64, S_LSHR_B32, S_LSHR_B64, S_ASHR_I32, S_ASHR_I64,
  S_MUL_I32, S_MUL_HI_U32, S_MUL_HI_I32, S_MUL_U64, S_MUL_F32, S_ADD_F32, S_SUB_F32,
  S_BFE_U32, S_BFE_I32, S_BFM_B32, S_BFM_B64,
  S_CSELECT_B32, S_CSELECT_B64,
  S_MIN_I32, S_MIN_U32, S_MAX_I32, S_MAX_U32,
  S_PACK_LL_B32_B16, S_PACK_LH_B32_B16,
  S_LSHL1_ADD_U32, S_LSHL2_ADD_U32, S_LSHL3_ADD_U32, S_LSHL4_ADD_U32,
  S_ADD_NC_U64, S_SUB_NC_U64,

  // -- VOP1 --
  V_MOV_B32, V_MOV_B64, V_NOP, V_NOT_B32, V_BFREV_B32,
  V_SWAP_B32,
  V_CVT_F32_I32, V_CVT_F32_U32, V_CVT_I32_F32, V_CVT_U32_F32,
  V_CVT_F16_F32, V_CVT_F32_F16, V_CVT_F32_BF16,
  V_CVT_F32_UBYTE0, V_CVT_F32_UBYTE1, V_CVT_F32_UBYTE2, V_CVT_F32_UBYTE3,
  V_CVT_F64_U32, V_CVT_F64_I32, V_CVT_U32_F64,
  V_RCP_IFLAG_F32, V_RCP_F32, V_RSQ_F32, V_SQRT_F32, V_EXP_F32, V_LOG_F32,
  V_LDEXP_F32,
  V_FLOOR_F32, V_CEIL_F32, V_TRUNC_F32, V_FRACT_F32,
  V_READFIRSTLANE_B32,
  // VOP1 packed FP8/BF8 → 2x F32 expansion (VOP1Instructions.td:652-
  // 653, profile VOPProfileCVT_PK_F32_F8). Reads 16 bits of the i32
  // src — the low half (bytes 0,1) when op_sel:[0,*] / SDWA WORD_0,
  // the high half (bytes 2,3) when op_sel:[1,*] / SDWA WORD_1 — and
  // expands the two FP8/BF8 lanes into a v2f32 written to a VGPR
  // pair starting at vdst. FP8 is the OCP E4M3FN format; BF8 is the
  // OCP E5M2 format. Lowering selects the matching
  // `llvm.amdgcn.cvt.pk.f32.{fp8,bf8}(i32 src, i1 word_sel)`
  // intrinsic and bitcasts the v2f32 result to i64 before
  // writeReg64. The op_sel-based word selector is parsed from the
  // disassembly text exactly as in V_ADD_NC_U16 / V_FMA_MIX_F32 (no
  // first-class "modifier" channel exists in our OperandView yet);
  // unparseable / out-of-range selectors fall through to word_sel=0
  // — never silently corrupted, the parser invariant is the same as
  // for the other op_sel handlers. The reverse direction
  // (V_CVT_PK_FP8_F32 / V_CVT_PK_BF8_F32) lives in the VOP3 block
  // below; this is the read-side companion.
  V_CVT_PK_F32_FP8, V_CVT_PK_F32_BF8,
  // VOP1 single-lane FP8/BF8 → F32 expansion (VOP1Instructions.td:650-
  // 651, profile VOPProfileCVT_F32_F8). Reads ONE 8-bit lane of the
  // i32 src — selected by SDWA src0_sel / e64 op_sel byte_sel — and
  // produces an f32. The SDWA encoding can pick any of the four bytes
  // (0..3); the e64 encoding's default (no op_sel printed) is byte 0
  // and is the only shape the gfx1250 corpus emits today (the LLVM
  // isel pattern in VOP1Instructions.td:670-680 maps non-zero
  // byte_sel through the SDWA pseudo, which we have not yet wired —
  // adding it would only widen this handler, not change its shape).
  // Lowering selects `llvm.amdgcn.cvt.f32.{fp8,bf8}(i32 src, i32
  // byte_sel)` and writeReg32 the result. SDWA / op_sel-bearing
  // encodings refuse loudly via RaiseFailure::unsupportedShape so a
  // future corpus drift surfaces immediately rather than silently
  // collapsing to byte 0.
  V_CVT_F32_FP8, V_CVT_F32_BF8,
  // VOP1 find-first-bit family (gfx7+, VOP1Instructions.td:371-373).
  // V_FFBH_U32  -> AMDGPUffbh_u32 = ctlz_zero_undef but returns -1 on
  //                input 0; lower with llvm.ctlz(x, false) — LLVM
  //                returns the bitwidth (32) for input 0, so we cmov
  //                to -1 explicitly to match hardware.
  // V_FFBL_B32  -> AMDGPUffbl_b32 = cttz_zero_undef but returns -1 on
  //                input 0; same pattern with llvm.cttz.
  // V_FFBH_I32  -> AMDGPUffbh_i32 = position of highest non-sign bit;
  //                returns -1 for input 0 or -1 (uniform sign). Lower
  //                via the dedicated llvm.amdgcn.sffbh intrinsic which
  //                selects directly back to v_ffbh_i32_e32.
  V_FFBH_U32, V_FFBL_B32, V_FFBH_I32,

  // -- VOP2 / VOP3 --
  V_ADD_F32, V_SUB_F32, V_SUBREV_F32, V_MUL_F32,
  V_FMAC_F32, V_FMA_F32, V_FMAMK_F32, V_FMAAK_F32,
  V_MAX_F32, V_MIN_F32,
  V_ADD_NC_U32, V_SUB_NC_U32, V_SUBREV_NC_U32,
  V_ADD_CO_U32, V_ADD_CO_CI_U32,
  V_SUB_CO_U32, V_SUBREV_CO_U32, V_SUB_CO_CI_U32, V_SUBREV_CO_CI_U32,
  V_AND_B32, V_OR_B32, V_XOR_B32, V_XNOR_B32,
  V_LSHLREV_B32, V_LSHRREV_B32, V_ASHRREV_I32,
  V_CNDMASK_B32,
  V_MUL_LO_U32, V_MUL_HI_U32, V_MUL_HI_I32,
  V_MUL_I32_I24, V_MUL_U32_U24, V_MUL_HI_U32_U24, V_MUL_HI_I32_I24,
  V_MAD_U32_U24, V_MAD_U32,
  V_ADD3_U32, V_LSHL_ADD_U32, V_ADD_LSHL_U32, V_LSHL_OR_B32, V_AND_OR_B32, V_OR3_B32, V_XAD_U32,
  // VOP3 funnel-shift right: dst = ((src0:src1) >> src2[4:0])[31:0].
  // .td uses the SDAG `fshr` node directly (VOP3Instructions.td:222),
  // which maps to `llvm.fshr.i32` in IR. src2 is masked to 5 bits
  // by hardware before the shift.
  V_ALIGNBIT_B32,
  // VOP3 ternary xor — gfx10+ only (VOP3Instructions.td:1348),
  // .td has no SDAG `umin3`-style node, the iselect pattern at
  // line 1350 directly matches `(xor (xor a, b), c)`. Lift is the
  // same shape as V_OR3_B32 above.
  V_XOR3_B32,
  // VOP3 16-bit no-carry add — gfx10+ (VOP3Instructions.td:1362).
  // Op_sel routes 16-bit halves of src0/src1 (lo or hi) and
  // selects which half of the 32-bit dst register receives the
  // result; the unselected half of dst is preserved per the
  // RDNA3+ ISA. The handler must read the prior dst value when
  // dst op_sel is set so the preserved half survives the
  // read-modify-write.
  V_ADD_NC_U16,
  V_BFE_U32, V_BFE_I32, V_PERM_B32,
  V_MBCNT_LO_U32_B32, V_MBCNT_HI_U32_B32,
  V_READLANE_B32, V_WRITELANE_B32,
  V_MED3_F32, V_MAX3_F32, V_MIN3_F32, V_MAX3_NUM_F32,
  // VOP3 IEEE-2019 ternary clamp `minnum(maxnum(s0, s1), s2)`.
  // gfx12 renamed gfx11's V_MINMAX_F32 (.td:1485, opcode 0x25f)
  // to V_MINMAX_NUM_F32 (.td:1696, opcode 0x268) when the .NUM
  // suffix was introduced to disambiguate from the IEEE-754
  // 2019 V_MINIMUMMAXIMUM_F32 (NaN-propagating, opcode 0x26c).
  // The opcode_map collapses both real names onto this SemOp.
  V_MINMAX_NUM_F32,
  // VOP3 integer 3-way max/min/median. The .td uses
  // AMDGPU{u,s}{max,min,med}3 SDAG nodes which the backend pattern-
  // matches; we lift them as the natural 2-step ICmp+Select chain
  // (no LLVM `*3` IR intrinsic exists). gfx11/gfx12 keep these
  // (VOP3Instructions.td:1792-1798).
  V_MAX3_U32,
  // VOP3 signed-integer median-of-three. Hardware semantic
  // (VOP3Instructions.td:1796 via AMDGPUsmed3 SDAG node):
  //   med3_i32(a, b, c) = smax(smin(a, b), smin(smax(a, b), c))
  // i.e. the middle of three signed i32 values. We lift it as a
  // pair of `llvm.smin`/`llvm.smax` intrinsics (matching the
  // `handle_vopd.cpp` style that already uses these intrinsics for
  // VOPD smin/smax/umin/umax pairs). The backend's
  // `AMDGPUISelDAGToDAG`/`AMDGPUISelLowering` pattern-matches the
  // `smax(smin(...), smin(smax(...), ...))` shape back to
  // V_MED3_I32, so the round-trip is structure-preserving and the
  // generated assembly recovers the original instruction without
  // codegen quality loss.
  V_MED3_I32,
  V_MAX_NUM_F32, V_MIN_NUM_F32,
  // IEEE-754 2019 maximum/minimum: propagate NaN (distinct from maxnum/minnum).
  V_MAXIMUM_F32, V_MINIMUM_F32,
  V_DIV_FIXUP_F32, V_DIV_FMAS_F32, V_DIV_SCALE_F32,
  V_FMA_MIX_F32,
  V_ADD_F16, V_MUL_F16, V_SUB_F16, V_SUBREV_F16, V_MAC_F16, V_FMAC_F16,
  // VOP2 F16 multiply-add-with-literal pseudos (mirror of
  // V_FMAMK_F32 / V_FMAAK_F32 for the f16 lane). Defined in
  // VOP2Instructions.td:1206-1210 — both take a 16-bit constant K
  // alongside two F16 sources and lower to llvm.fma.f16:
  //   v_madmk_f16 dst, src0, K, src2 -> dst = src0 * K + src2
  //   v_madak_f16 dst, src0, src1, K -> dst = src0 * src1 + K
  // Note: hardware uses the legacy "mad" name, but the lowered
  // semantics are fused-multiply-add (no rounding of the intermediate
  // product), matching the F32 FMAMK/FMAAK convention.
  V_MADMK_F16, V_MADAK_F16,
  V_MAX_F16, V_MIN_F16, V_LDEXP_F16, V_FLOOR_F16, V_CVT_F16_U16, V_CVT_U16_F16,
  V_ASHRREV_I16, V_LSHRREV_B16, V_LSHLREV_B16,
  V_MAX_U16, V_MIN_U16, V_MAX_I16, V_MIN_I16,
  // 16-bit integer arith (gfx8+, VOP2Instructions.td). Plain i16
  // add/sub/subrev with wrapping overflow (no carry-out — distinct
  // from the rarely-used v_add_co_u16). v_mul_lo_u16 returns the low
  // 16 bits of the multiply, naturally produced by `mul i16`.
  V_ADD_U16, V_SUB_U16, V_SUBREV_U16, V_MUL_LO_U16,
  V_DOT2C_I32_I16, V_DOT4C_I32_I8, V_DOT8C_I32_I4,
  V_PK_FMAC_F16,
  V_PACK_B32_F16,
  V_CVT_PK_BF16_F32, V_CVT_PK_BF8_F32, V_CVT_PK_FP8_F32,
  V_CVT_PKRTZ_F16_F32, V_CVT_PK_F16_F32,
  V_CVT_SCALEF32_PK_FP4_F32,
  V_BFM_B32,

  // -- VOP2/VOP3 FP64 --
  V_ADD_F64, V_MUL_F64, V_FMA_F64, V_FMAC_F64,
  // VOP1 FP64. v_rcp_f64 is a TRANS-class transcendental (see
  // VOP1Instructions.td: `let TRANS = 1, SchedRW = [WriteTrans64]`),
  // not a true reciprocal — hardware returns a ~26-bit accurate
  // approximation that the LLVM `int_amdgcn_rcp` intrinsic models
  // exactly. We deliberately lift to that intrinsic rather than to a
  // generic `fdiv 1.0, x` because (a) gfx942 isels the intrinsic
  // straight back to v_rcp_f64 (no Newton-Raphson refinement is
  // emitted), and (b) `fdiv` would lower to a software divide
  // sequence on gfx942 unless `arcp`/fast-math flags are set, which
  // would be a silent semantics change versus the source op.
  V_RCP_F64,

  V_MAX_U32, V_MIN_U32, V_MAX_I32, V_MIN_I32,
  V_PERMLANE16_B32, V_PERMLANEX16_B32, V_PERMLANE64_B32,
  V_PERMLANE16_SWAP_B32, V_PERMLANE32_SWAP_B32,

  // -- VOPC (V_CMP_* and V_CMPX_*) --
  //
  // All ~100 V_CMP_*_{U,I,F}{16,32,64} and V_CMPX_*_{U,I,F}{16,32} pseudos
  // collapse onto these two SemOps; the actual {predicate, element type,
  // width} triple is looked up from `VCmpMeta` keyed on the MC opcode.
  // `V_CMP`   writes an SGPR pair (or VCC, depending on the encoding).
  // `V_CMPX`  additionally ANDs the compare result into EXEC.
  V_CMP, V_CMPX,

  // -- VOP3P --
  V_PK_ADD_F32, V_PK_MUL_F32, V_PK_FMA_F32,
  V_PK_MAX_F32, V_PK_MIN_F32, V_PK_MOV_B32,

  V_BITOP3_B32, V_BITOP3_B16,

  // GFX9 VOP3-only v_add/sub_i32 — plain add/sub when clamp=0,
  // saddsat/ssubsat when clamp=1.
  V_ADD_I32, V_SUB_I32,

  // -- 64-bit vector ops --
  V_LSHLREV_B64,
  // gfx8+ VOP3 64-bit shifts. Same operand shape as V_LSHLREV_B64
  // (i64 dst, i32 shamt, i64 src1, reversed-operand convention:
  // `dst = src1 >> shamt`). Lower to LLVM `lshr` (logical right) and
  // `ashr` (arithmetic right) on the i64 src1, with the i32 shamt
  // zext'd to i64 — the AMDGPU hardware masks the count to 6 bits so
  // the LLVM behaviour matches as long as we feed a valid i32 (LLVM
  // shifts >= bitwidth are poison, the hardware masks; we don't paper
  // over the difference because corpus shifts always carry a finite
  // immediate or a producer that already masks).
  V_LSHRREV_B64, V_ASHRREV_I64,
  V_LSHL_ADD_U64, V_ADD_NC_U64,
  // gfx1250 VOP2 64-bit unsigned multiply (low 64 bits of s0 * s1).
  V_MUL_U64,
  V_MAD_U64_U32, V_MAD_CO_U64_U32,

  // -- FLAT / GLOBAL / SCRATCH memory --
  FLAT_LOAD_UBYTE, FLAT_LOAD_SBYTE, FLAT_LOAD_USHORT, FLAT_LOAD_SSHORT,
  FLAT_LOAD_DWORD, FLAT_LOAD_DWORDX2, FLAT_LOAD_DWORDX3, FLAT_LOAD_DWORDX4,
  FLAT_STORE_BYTE, FLAT_STORE_SHORT, FLAT_STORE_SHORT_D16_HI,
  FLAT_STORE_DWORD, FLAT_STORE_DWORDX2, FLAT_STORE_DWORDX3, FLAT_STORE_DWORDX4,
  GLOBAL_LOAD_UBYTE, GLOBAL_LOAD_SBYTE, GLOBAL_LOAD_USHORT, GLOBAL_LOAD_SSHORT,
  GLOBAL_LOAD_SHORT_D16_HI,
  GLOBAL_LOAD_DWORD, GLOBAL_LOAD_DWORDX2, GLOBAL_LOAD_DWORDX3, GLOBAL_LOAD_DWORDX4,
  GLOBAL_STORE_BYTE, GLOBAL_STORE_SHORT, GLOBAL_STORE_SHORT_D16_HI,
  GLOBAL_STORE_DWORD, GLOBAL_STORE_DWORDX2, GLOBAL_STORE_DWORDX3, GLOBAL_STORE_DWORDX4,

  // -- FLAT atomics --
  FLAT_ATOMIC_ADD, FLAT_ATOMIC_SUB,
  FLAT_ATOMIC_AND, FLAT_ATOMIC_OR, FLAT_ATOMIC_XOR,
  FLAT_ATOMIC_SMIN, FLAT_ATOMIC_SMAX, FLAT_ATOMIC_UMIN, FLAT_ATOMIC_UMAX,
  FLAT_ATOMIC_SWAP, FLAT_ATOMIC_CMPSWAP,
  FLAT_ATOMIC_ADD_F32,

  // -- GLOBAL atomics --
  GLOBAL_ATOMIC_ADD, GLOBAL_ATOMIC_SUB,
  GLOBAL_ATOMIC_AND, GLOBAL_ATOMIC_OR, GLOBAL_ATOMIC_XOR,
  GLOBAL_ATOMIC_SMIN, GLOBAL_ATOMIC_SMAX, GLOBAL_ATOMIC_UMIN, GLOBAL_ATOMIC_UMAX,
  GLOBAL_ATOMIC_SWAP, GLOBAL_ATOMIC_CMPSWAP,
  GLOBAL_ATOMIC_ADD_F32,
  GLOBAL_ATOMIC_PK_ADD_BF16, GLOBAL_ATOMIC_PK_ADD_F16,

  // -- SMEM atomics --
  S_ATOMIC_SWAP,

  // -- DS --
  DS_LOAD_TR16_B128,
  DS_READ_B64_TR_B16,
  DS_READ_B64_TR_B8,
  // gfx1250 spelling of the same 64-bit transposed LDS load that
  // gfx950 disassembles as `ds_read_b64_tr_b8`. The hardware
  // semantics are identical: each lane reads 64 bits (8 x i8) from
  // its LDS base, then the data is transposed across 8-lane groups
  // so each lane post-transpose holds 8 i8 values from 8 different
  // source lanes at the same intra-group element offset (v2i32
  // packed). The two SemOps are kept distinct because they are two
  // distinct LLVM MC opcodes (DS_LOAD_TR8_B64 vs DS_READ_B64_TR_B8)
  // with separate isel patterns and separate intrinsics
  // (`int_amdgcn_ds_load_tr8_b64` gated isGFX1250Plus,
  // `int_amdgcn_ds_read_tr8_b64` gated HasGFX950Insts); both lower
  // through the same hand-rolled bpermute-based emulation in
  // handle_ds.cpp because gfx942 (the transpiler's target ISA) has
  // neither isel pattern and no in-tree pre-isel emulation.
  DS_LOAD_TR8_B64,
  DS_READ_B32, DS_READ_B64,
  // 96-bit (3 x i32) LDS load. LLVM MC opcode `DS_READ_B96`; gfx11+
  // (gfx1100/gfx1200/gfx1250) renames the asm spelling to
  // `ds_load_b96` (DSInstructions.td:1578 declares
  // `defm DS_READ_B96 : DS_Real_gfx11_gfx12_gfx13<0x0fe,
  // "ds_load_b96">`). Hardware reads 96 bits from the lane's LDS
  // base; the lift is `load <3 x i32>` from addrspace(3). The
  // gfx942 backend lowers the 3-dword vector load to either a
  // native `ds_read_b96` (gfx9 inherits the `_vi` Real form) or
  // splits it into 3x `ds_read_b32` with the appropriate
  // increments — both are correct in-place lowerings.
  // Inserted between DS_READ_B64 and DS_READ_B128 deliberately so
  // the existing range checks (`sop >= DS_READ_B32 &&
  // sop <= DS_READ_I8` for reads, parallel for writes) continue to
  // cover it without a special case.
  DS_READ_B96,
  DS_READ_B128,
  DS_READ2_B32, DS_READ2_B64,
  DS_READ_U16, DS_READ_I16, DS_READ_U8, DS_READ_I8,
  DS_WRITE_B32, DS_WRITE_B64,
  // Symmetric write-side for `ds_load_b96`: gfx11+ asm spelling is
  // `ds_store_b96` (DSInstructions.td:1576); the LLVM MC opcode
  // remains `DS_WRITE_B96`. Lift is `store <3 x i32>` to
  // addrspace(3). Inserted between DS_WRITE_B64 and DS_WRITE_B128
  // for the same range-check reason as DS_READ_B96 above.
  DS_WRITE_B96,
  DS_WRITE_B128,
  DS_WRITE2_B32, DS_WRITE2_B64,
  DS_WRITE_B16, DS_WRITE_B8,
  // D16_HI partial-store family (gfx8+ HasD16LoadStore):
  // store the upper 16 bits (B16_D16_HI) or bits [23:16] (B8_D16_HI)
  // of the source VGPR to LDS. The "D16_HI" suffix names the
  // *source* register half being stored, not a dest-merge — these
  // are write-only and there is no tied dest_in operand. The
  // companion D16 reads (DS_READ_U/I8_D16{,_HI}, DS_READ_U16_D16{,_HI})
  // are not yet on the worklist; if they surface, add them here as
  // a separate set with their own tied-source dest_in handling.
  DS_WRITE_B16_D16_HI, DS_WRITE_B8_D16_HI,
  DS_BPERMUTE_B32,
  // SPE_DESIGN.md §3 Class 2 (DsSwizzle). Wave-width-specific
  // cross-lane shuffle. The handler refuses with `unsupportedShape`
  // until CROSS_LANE_SURVEY.md item P6 (lift through
  // llvm.amdgcn.ds.swizzle) lands; the wave-size classifier
  // (wave_size_obstruction.cpp) flags it before the handler is even
  // dispatched in the cross-wave case.
  DS_SWIZZLE_B32,

  // -- MUBUF --
  BUFFER_LOAD_DWORD, BUFFER_LOAD_DWORDX2, BUFFER_LOAD_DWORDX3, BUFFER_LOAD_DWORDX4,
  BUFFER_LOAD_UBYTE, BUFFER_LOAD_SBYTE, BUFFER_LOAD_USHORT, BUFFER_LOAD_SSHORT,
  BUFFER_LOAD_SHORT_D16, BUFFER_LOAD_SHORT_D16_HI,
  // D16 byte variants — gfx9+ partial-write loads. The 8-bit datum is
  // sign- or zero-extended to i16 and merged into the lo (`_D16`) or
  // hi (`_D16_HI`) half of the destination VGPR; the other 16 bits
  // are preserved (BUFInstructions.td:1155-1169, predicate
  // `D16PreservesUnusedBits`). Mnemonic on gfx11+/gfx1250 is
  // `buffer_load_d16_u8` / `_d16_i8` / `_d16_hi_u8` / `_d16_hi_i8`.
  BUFFER_LOAD_UBYTE_D16, BUFFER_LOAD_UBYTE_D16_HI,
  BUFFER_LOAD_SBYTE_D16, BUFFER_LOAD_SBYTE_D16_HI,
  BUFFER_LOAD_DWORD_LDS, BUFFER_LOAD_DWORDX2_LDS,
  BUFFER_LOAD_DWORDX4_LDS, BUFFER_STORE_DWORDX4_LDS,
  BUFFER_STORE_DWORD, BUFFER_STORE_DWORDX2, BUFFER_STORE_DWORDX3, BUFFER_STORE_DWORDX4,
  BUFFER_STORE_BYTE, BUFFER_STORE_SHORT,

  // -- MUBUF atomics --
  // Order is significant: handle_mubuf.cpp dispatches via the range
  // check `[BUFFER_ATOMIC_ADD, BUFFER_ATOMIC_PK_ADD_F16]`. New
  // BUFFER_ATOMIC_* SemOps must stay inside this range so the range
  // check picks them up; entries the handler does not explicitly
  // case-match are caught by the switch's default branch with a
  // `RaiseFailure::unsupportedShape("unsupported buffer atomic")`.
  BUFFER_ATOMIC_ADD, BUFFER_ATOMIC_SUB,
  BUFFER_ATOMIC_AND, BUFFER_ATOMIC_OR, BUFFER_ATOMIC_XOR,
  // SPE_DESIGN.md §3 Class 3 non-commutative atomics (NonCommutativeAtomic).
  // The wave-size classifier flags these in the cross-wave case;
  // commutative AtomicRMW dispatch in handle_mubuf.cpp does not
  // model the cmpxchg / xchg semantics, so the handler's default
  // branch refuses with `unsupportedShape`.
  BUFFER_ATOMIC_SWAP, BUFFER_ATOMIC_CMPSWAP,
  BUFFER_ATOMIC_ADD_F32,
  BUFFER_ATOMIC_PK_ADD_BF16, BUFFER_ATOMIC_PK_ADD_F16,

  // -- MFMA --
  // gfx950 scaled F8F6F4 variants share a per-shape intrinsic but take 9
  // src-format sub-variants each; those are collapsed onto these four SemOps
  // in kCanonTable.
  V_MFMA_F32_16x16x128_F8F6F4, V_MFMA_SCALE_F32_16x16x128_F8F6F4,
  V_MFMA_F32_32x32x64_F8F6F4, V_MFMA_SCALE_F32_32x32x64_F8F6F4,
  // F32 <- F16/F32 (gfx908+). Each covers its pseudo's _e64/_vgprcd_/_mac_
  // variants via pseudoAlias stripping in OpcodeMap::canonicalize.
  V_MFMA_F32_16x16x16_F16, V_MFMA_F32_32x32x8_F16,
  V_MFMA_F32_16x16x4_F32, V_MFMA_F32_32x32x1_F32, V_MFMA_F32_32x32x2_F32,
  V_MFMA_F32_4x4x1_F32, V_MFMA_F32_16x16x1_F32,
  V_MFMA_F32_32x32x4_F16, V_MFMA_F32_16x16x4_F16, V_MFMA_F32_4x4x4_F16,
  // I32 <- I8.
  V_MFMA_I32_16x16x32_I8, V_MFMA_I32_32x32x16_I8,
  V_MFMA_I32_32x32x4_I8, V_MFMA_I32_16x16x4_I8, V_MFMA_I32_4x4x4_I8,
  // F32 <- XF32 (gfx940+).
  V_MFMA_F32_16x16x8_XF32, V_MFMA_F32_32x32x4_XF32,
  // F32 <- BF16 (gfx908 2-byte variants).
  V_MFMA_F32_32x32x2_BF16, V_MFMA_F32_16x16x2_BF16, V_MFMA_F32_4x4x2_BF16,
  // F32 <- BF16 "1K" shapes (gfx90a+).
  V_MFMA_F32_16x16x16_BF16_1K, V_MFMA_F32_32x32x8_BF16_1K,
  // F32 <- BF16/F16 wide shapes (gfx950).
  V_MFMA_F32_16x16x32_BF16, V_MFMA_F32_32x32x16_BF16,
  V_MFMA_F32_16x16x32_F16,
  // F32 <- FP8/BF8 (gfx940+).
  V_MFMA_F32_16x16x32_FP8_FP8, V_MFMA_F32_16x16x32_FP8_BF8,
  V_MFMA_F32_16x16x32_BF8_FP8, V_MFMA_F32_16x16x32_BF8_BF8,
  V_MFMA_F32_32x32x16_FP8_FP8, V_MFMA_F32_32x32x16_FP8_BF8,
  V_MFMA_F32_32x32x16_BF8_FP8, V_MFMA_F32_32x32x16_BF8_BF8,

  // -- WMMA (gfx1250) --
  // 16x16x32 WMMA with f32 accumulator and 16-bit element types. Both
  // share the same per-lane fragment shape (A,B: <16 x t>, C/D:
  // <8 x f32>) and same K-decomposition path through the gfx942 MFMA
  // lowering — `emitWMMAtoMFMA` is parameterised on input element
  // type and routes to the matching CDNA3 MFMA intrinsic
  // (mfma_f32_16x16x16f16 vs mfma_f32_16x16x16bf16_1k).
  V_WMMA_F32_16x16x32_F16,
  V_WMMA_F32_16x16x32_BF16,
  // 16x16x4 WMMA with f32 accumulator and 32-bit f32 element types
  // for both A and B (gfx1250 RDNA4 VOP3P opcode 0x05D). Per-Wave32-
  // lane fragment shape is A,B: <2 x f32> (only 4 K-elements split
  // across 2 dwords per lane), C/D: <8 x f32>; this is structurally
  // distinct from the 16-bit (K=32, A/B = <16 x t>) and 8-bit
  // (K=64, A/B = <8 x i32>) families above and so does NOT share
  // the `emitWMMAtoMFMA` decomposition (which is parameterised on
  // 16-/8-bit element packing, not f32). The native intrinsic
  // `amdgcn_wmma_f32_16x16x4_f32` is declared inside
  // `AMDGPUWMMAIntrinsicsGFX1250` (gated by `isGFX125xOnly` in
  // IntrinsicsAMDGPU.td:4113-4114) and is NOT part of the gfx12
  // RDNA4-base WMMA family (`AMDGPUWMMAIntrinsicsGFX12`,
  // FeatureWMMA{128,256}bInsts), so the same-target lift gates on
  // `ISAProfile::hasTensorOps` (FeatureGFX1250Insts) — matching
  // the LLVM intrinsic's actual subtarget gating — rather than
  // `hasWMMA12`. Call shape is `AMDGPUWmmaIntrinsicModsAllReuse`,
  // 8 args: `(A_mod, A, B_mod, B, C_mod, C, reuse_a, reuse_b)`.
  // Cross-target lift to gfx942 would need a new K=4 MFMA
  // decomposition path (gfx942 has `mfma_f32_16x16x4f32`) that no
  // kernel in the current corpus exercises, so we refuse loudly
  // via `RaiseFailure::unsupportedShape` to surface the gap
  // immediately rather than silently degrade.
  V_WMMA_F32_16x16x4_F32,
  // 16x16x64 WMMA with f32 accumulator and 8-bit element types
  // (fp8/bf8). The four AB combinations are distinct opcodes (and
  // distinct CDNA3 MFMA intrinsics on gfx942) but share the same
  // per-lane fragment shape (A,B: <8 x i32> = 32 fp8/bf8 bytes per
  // Wave32 lane, C/D: <8 x f32>) and the same gfx942 MFMA decomposition
  // path through `emitWMMAtoMFMA`. The K=64 dimension splits into
  // 2 chained K=32 MFMAs per Wave32 group, mirroring the K=32→2×K=16
  // split used for the 16-bit variants. The lane-redistribution math
  // is byte-identical between the two K-families (32 bytes per lane
  // either way), so the only divergence inside `emitWMMAtoMFMA` is the
  // per-MFMA pack type (i64 vs <4 x half|i16>) and the dispatched
  // intrinsic ID. See `WMMAInputType` in `wmma_lowering.hpp` for the
  // full enumeration.
  V_WMMA_F32_16x16x64_FP8_FP8,
  V_WMMA_F32_16x16x64_FP8_BF8,
  V_WMMA_F32_16x16x64_BF8_FP8,
  V_WMMA_F32_16x16x64_BF8_BF8,
  // 16x16x64 WMMA with i32 accumulator and unsigned/signed 8-bit
  // integer inputs (the gfx1250 IU8 variant; the LLVM intrinsic
  // uses `iu8` to denote that the per-input sign extension is
  // selected at call site through the `neg_lo` modifier rather
  // than the opcode itself). Per-Wave32-lane fragment shape is
  // identical to the FP8 sibling (A,B: <8 x i32> = 32 packed i8
  // bytes per lane, C/D: <8 x i32> for integer accumulator). On
  // gfx942 we lower through the same `emitWMMAtoMFMA` helper,
  // dispatching the per-MFMA call to `mfma_i32_16x16x32_i8`
  // (i64 packed A/B, <4 x i32> accumulator). The handler must
  // also use a different native-WMMA12 intrinsic shape on gfx12
  // hardware: `AMDGPUWmmaIntrinsicModsABClamp` (8 args including
  // a trailing clamp flag), distinct from the 16-bit AllReuse
  // and the 8-bit FP8 ModsC shapes.
  V_WMMA_I32_16x16x64_IU8,

  // -- VOPD -- (handled via string parsing of fullText, not opcode)
  VOPD_GENERIC,

  // -- VIMAGE TENSOR (gfx1250-only) --
  // Tensor descriptor memory ops driven by the gfx1250 TENSOR cnt unit
  // (`MIMGInstructions.td:2049-2113`, `VIMAGE_TENSOR_Pseudo`,
  // `let SubtargetPredicate = isGFX125xOnly`). Each opcode encodes
  // up to four 128-/256-bit Tensor Descriptors (`D# group 0..3`),
  // a `R128A16:$r128` flag, and a `CPol:$cpol` cachepolicy immediate.
  // `_d2` is the up-to-2D form (passes NULL for D# group 2/3); `_d4`
  // is the up-to-4D form. Both share the same SemOp here because
  // their semantic intent is identical and their refusal contract is
  // identical too — the handler `handleVIMAGE` discriminates on
  // `di.mnemonic` only when shape differentiation matters (e.g.,
  // a future native-target intrinsic-emit path that fills the
  // 0-init D# operands for `_d2`).
  //
  // gfx942 has no equivalent hardware unit. The handler refuses
  // loudly via `RaiseFailure::unsupportedShape` with a precise
  // diagnostic explaining the cross-target gap, in line with the
  // user-rules (no silent fallbacks). The matching LLVM intrinsics
  // are `int_amdgcn_tensor_load_to_lds` /
  // `int_amdgcn_tensor_store_from_lds` (IntrinsicsAMDGPU.td:4213).
  TENSOR_LOAD_TO_LDS,
  TENSOR_STORE_FROM_LDS,

  // -- gfx1250 async global → LDS load --
  //
  // FLAT async global-to-LDS load, four widths per the b8 / b32 / b64 /
  // b128 family. Each width has both a plain VGPR_64 vaddr form and a
  // SADDR (SReg_64 base + VGPR_32 vaddr offset) form, both of which
  // collapse to the same SemOp per width; `handleFLAT` discriminates
  // shape on `op.nSrcs()` exactly the same way `tensor_load_to_lds`
  // discriminates `_d2` vs `_d4`. The pseudo InOperandList is
  // documented in `FLATInstructions.td:391-417`
  // (`FLAT_Global_Load_LDS_Pseudo<…, IsAsync=1>`):
  //
  //   plain : (vdst:VGPR_32, vaddr:VGPR_64,             offset, cpol)
  //   SADDR : (vdst:VGPR_32, saddr:SReg_64, vaddr:VGPR_32, offset, cpol)
  //
  // `vdst` here is the per-lane LDS i32 OFFSET (TableGen `vdst` slot
  // is in the *input* list because `IsAsync=1` enables `has_vdst`),
  // not a written register: each lane uses its own VGPR_32 value as
  // the LDS-base address for the burst write. The intrinsics
  // `int_amdgcn_global_load_async_to_lds_b{8,32,64,128}`
  // (IntrinsicsAMDGPU.td:3939-3946) all share signature
  // `AMDGPUAsyncGlobalLoadToLDS` (line 3904) and take the LDS
  // pointer as the second operand (`local_ptr_ty`); we materialise
  // it via `inttoptr i32 -> ptr addrspace(3)` from the per-lane
  // VGPR_32. The width is encoded only in the intrinsic ID — the
  // operand bank is identical across all four widths.
  //
  // Separate SemOps per width (rather than a single
  // `GLOBAL_LOAD_ASYNC_TO_LDS_BX` discriminated by mnemonic) so the
  // SemOp ↔ intrinsic mapping is direct and the handler is a small
  // switch instead of string parsing — the canonical opcode_map
  // collapses each `_gfx1250` real onto its width-specific pseudo.
  //
  // === Same-target gfx1250 → gfx1250 contract ===
  //
  // gfx1250 has the asynccnt unit and the native intrinsic; the
  // handler emits a direct call inside an `emitUnderExec` diamond
  // (per-lane operation: each lane fires its own LDS write, inactive
  // lanes do not). `IntrInaccessibleMemOrArgMemOnly` on the
  // intrinsic prevents downstream passes from CSEing or reordering
  // the asynchronous fetch across other memory sites — the
  // user-visible barrier semantics live in companion
  // `s_wait_asynccnt` instructions, not in this op. The intrinsic's
  // `offset` immediate corresponds to the FLAT instruction's
  // `flat_offset` slot; `cpol` is the gfx12+ cachepolicy bitfield
  // (th, scope) carried as the trailing immediate.
  //
  // === Cross-target (gfx942 and earlier) contract ===
  //
  // The asynccnt unit and `int_amdgcn_global_load_async_to_lds_b*`
  // are gfx1250-only (`SubtargetPredicate = isGFX1250Plus` on the
  // VFLAT reals, `FeatureGFX1250Insts`). gfx942 has no asynchronous
  // global→LDS DMA channel and no equivalent burst path. Refusing
  // loudly via `RaiseFailure::unsupportedShape` is the only honest
  // option; a synthesised synchronous global_load + ds_write pair
  // would change the wave's memory-ordering and asynccnt observable
  // state, which is exactly what gfx1250 producers (e.g. tensilelite
  // f8 / bf16 GEMMs and triton block-pipelined matmul kernels) rely
  // on for software pipelining.
  GLOBAL_LOAD_ASYNC_TO_LDS_B8,
  GLOBAL_LOAD_ASYNC_TO_LDS_B32,
  GLOBAL_LOAD_ASYNC_TO_LDS_B64,
  GLOBAL_LOAD_ASYNC_TO_LDS_B128,

  // -- AGPR --
  V_ACCVGPR_READ_B32, V_ACCVGPR_WRITE_B32,

  SemOp_COUNT
};

// Stable human-readable identifier for a SemOp (the enum's spelling,
// e.g. `"V_CMPX"` for `SemOp::V_CMPX`). Used in diagnostics — prefer
// this over `(int)sop` so errors name the instruction class rather
// than a raw enum position that drifts with enum edits.
const char *semOpName(SemOp op);

} // namespace transpiler

#endif
