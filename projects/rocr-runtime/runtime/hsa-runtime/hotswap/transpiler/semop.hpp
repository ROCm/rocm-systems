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
  S_FLBIT_I32_B32, S_FLBIT_I32_B64,
  S_SEXT_I32_I8, S_SEXT_I32_I16,
  S_CVT_F32_U32, S_CVT_F32_I32, S_CVT_U32_F32, S_CVT_I32_F32,
  S_AND_SAVEEXEC_B32, S_OR_SAVEEXEC_B32, S_XOR_SAVEEXEC_B32,
  S_ANDN2_SAVEEXEC_B32, S_ORN2_SAVEEXEC_B32,
  S_GETPC_B64,
  // SOP1 indirect set-PC. gfx1250 asm rename for `S_SETPC_B64`
  // (SOPInstructions.td:323 declares `isBranch + isIndirectBranch`,
  // line 2208 renames the asm string to `s_set_pc_i64`). The source
  // SGPR pair holds an absolute 64-bit PC value. In our IR-on-LLVM
  // setting we model two principled lowerings (see setpc_analysis.{hpp,
  // cpp} for the static analysis that classifies each site):
  //   Pattern A — statically resolvable intra-kernel branch (the source
  //               SGPR pair was produced by a local
  //               `s_get_pc_i64 + s_add_co_u32 + s_add_co_ci_u32` chain).
  //               Lowers to `br label %BB_target` since the target is a
  //               known intra-function label.
  //   Pattern B — subroutine return via an SGPR pair stashed at the call
  //               site (the canonical s[30:31] return-PC idiom). Lowers
  //               to `indirectbr ptr %ret_pc, [list of resolved return
  //               targets]`. The corresponding call-site
  //               `s_get_pc_i64 + s_add*` chains are rewritten by the
  //               raiser to materialise a `blockaddress(@kernel, %ret_BB)`
  //               into the ret-pair (via a post-handler hook in
  //               raiser.cpp), so the i64 fed to indirectbr is a real
  //               LLVM blockaddress constant rather than a binary PC.
  // Sites the analysis cannot resolve refuse loudly via
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
  // Two raisings, mirroring S_SET_PC_I64:
  //   Pattern A — call target ssrc was produced by a local
  //               `s_get_pc_i64 + s_add_co_u32 + s_add_co_ci_u32` chain
  //               that the SetPcAnalysis can resolve. Lowering writes
  //               `blockaddress(@kernel, %BB_returnAddr)` cast to i64
  //               into sdst and emits `br label %BB_callee`. The
  //               return PC the callee will eventually consume via a
  //               Pattern B `s_set_pc_i64 sdst` is therefore a real
  //               LLVM blockaddress constant.
  //   Otherwise — the call target is statically unresolvable
  //               (typical tensilelite pattern: chain addend is a
  //               runtime scalar derived from a kernarg, so the
  //               target is dynamic-dispatch). Refuse loudly via
  //               RaiseFailure::unsupportedShape — never emit a stub
  //               branch. Cross-block scalar / kernarg-derived call
  //               target resolution is tracked as a separate story
  //               (see semop dispatch site for the link).
  // Independent of the above, the analysis registers a synthetic
  // chain-terminator at the swap site itself (key = swap.offset,
  // value = {sdst-low-reg, swap.offset+swap.size}) so any downstream
  // Pattern B `s_set_pc_i64` reading sdst enumerates the swap's
  // return offset as one of its indirectbr targets.
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
  V_CVT_PK_F32_FP8,

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
  V_MAX_NUM_F32, V_MIN_NUM_F32,
  // IEEE-754 2019 maximum/minimum: propagate NaN (distinct from maxnum/minnum).
  V_MAXIMUM_F32, V_MINIMUM_F32,
  V_DIV_FIXUP_F32, V_DIV_FMAS_F32, V_DIV_SCALE_F32,
  V_FMA_MIX_F32,
  V_ADD_F16, V_MUL_F16, V_SUB_F16, V_SUBREV_F16, V_MAC_F16, V_FMAC_F16,
  V_MAX_F16, V_MIN_F16, V_LDEXP_F16, V_FLOOR_F16, V_CVT_F16_U16, V_CVT_U16_F16,
  V_ASHRREV_I16, V_LSHRREV_B16, V_LSHLREV_B16,
  V_MAX_U16, V_MIN_U16, V_MAX_I16, V_MIN_I16,
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
  V_LSHLREV_B64, V_LSHL_ADD_U64, V_ADD_NC_U64,
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
  V_WMMA_F32_16x16x32_F16,

  // -- VOPD -- (handled via string parsing of fullText, not opcode)
  VOPD_GENERIC,

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
