#include "opcode_map.hpp"

#include <cstdint>
#include <optional>
#include <string>

// AMDGPU target-private headers. They expose:
//   AMDGPU::getMCOpcode           (declared in Utils/AMDGPUBaseInfo.h)
//   AMDGPU::getVOPe64 / getVOPe32 / getDPPOp32 / getDPPOp64 /
//   getSDWAOp / getBasicFromSDWAOp / getGlobalVaddrOp
//                                  (declared in SIInstrInfo.h, implemented
//                                   in the TableGen-generated
//                                   AMDGPUGenInstrInfo.inc under
//                                   `#define GET_INSTRMAP_INFO`, linked from
//                                   libLLVMAMDGPUUtils.a).
//
// SIInstrInfo.h drags in the CodeGen TargetInstrInfo base, which we do not
// use at runtime, but pulling it in is preferable to hand-rolling forward
// declarations that would silently go stale if LLVM changes a signature.
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "SIInstrInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace transpiler {

namespace {

// Maps a canonical AMDGPU pseudo opcode to the SemOp the raiser dispatches on.
// The canonical form is what comes out of the canonicalization chain below:
//   MC opcode -> pseudo
//   pseudo    -> e64 (if VOP/VOPC has e32/e64 split)
//   pseudo    -> base (if SDWA/DPP)
//   pseudo    -> VADDR (if FLAT/GLOBAL SADDR)
//
// Using AMDGPU:: enum constants instead of strings gives us compile-time
// checking: if LLVM renames a pseudo, the build fails here rather than the
// lookup silently returning SemOp::Unknown at runtime.
struct Entry {
  unsigned opc;
  SemOp sem;
};

// Convenience macros for families with many addressing-mode variants that
// LLVM does not expose a single canonicalization helper for.
#define E(OP, SEM) Entry{AMDGPU::OP, SemOp::SEM}

// MUBUF: four addressing modes (OFFSET/OFFEN/IDXEN/BOTHEN) per base opcode,
// plus the gfx12+ VBUFFER fork that was introduced to separate the buffer
// descriptor source.
#define MUBUF4(BASE, SEM) \
  E(BASE##_OFFSET, SEM), \
  E(BASE##_OFFEN,  SEM), \
  E(BASE##_IDXEN,  SEM), \
  E(BASE##_BOTHEN, SEM)
#define VBUF4(BASE, SEM) \
  E(BASE##_VBUFFER_OFFSET, SEM), \
  E(BASE##_VBUFFER_OFFEN,  SEM), \
  E(BASE##_VBUFFER_IDXEN,  SEM), \
  E(BASE##_VBUFFER_BOTHEN, SEM)

// SMEM scalar loads: three operand-source forms (IMM / SGPR / SGPR_IMM).
#define SMEM3(BASE, SEM) \
  E(BASE##_IMM,      SEM), \
  E(BASE##_SGPR,     SEM), \
  E(BASE##_SGPR_IMM, SEM)

static const Entry kCanonTable[] = {
    // ---------------------------------------------------------------------
    // SOPP
    // ---------------------------------------------------------------------
    E(S_ENDPGM, S_ENDPGM),
    E(S_NOP, S_NOP),
    E(S_BRANCH, S_BRANCH),
    E(S_CODE_END, S_CODE_END),
    E(S_CBRANCH_SCC0, S_CBRANCH_SCC0),
    E(S_CBRANCH_SCC1, S_CBRANCH_SCC1),
    E(S_CBRANCH_VCCZ, S_CBRANCH_VCCZ),
    E(S_CBRANCH_VCCNZ, S_CBRANCH_VCCNZ),
    E(S_CBRANCH_EXECZ, S_CBRANCH_EXECZ),
    E(S_CBRANCH_EXECNZ, S_CBRANCH_EXECNZ),
    E(S_WAITCNT, S_WAITCNT),
    E(S_WAIT_LOADCNT, S_WAIT_LOADCNT),
    E(S_WAIT_KMCNT, S_WAIT_KMCNT),
    E(S_WAIT_DSCNT, S_WAIT_DSCNT),
    E(S_WAIT_XCNT, S_WAIT_XCNT),
    E(S_WAIT_LOADCNT_DSCNT, S_WAIT_LOADCNT_DSCNT),
    E(S_WAITCNT_DEPCTR, S_WAIT_ALU),
    E(S_CLAUSE, S_CLAUSE),
    E(S_DELAY_ALU, S_DELAY_ALU),
    E(S_SET_GPR_IDX_ON, S_SET_GPR_IDX_ON),
    E(S_SET_GPR_IDX_OFF, S_SET_GPR_IDX_OFF),
    E(S_SETVSKIP, S_SETVSKIP),

    // Barriers. GFX < 12 has a single SOPP `S_BARRIER` pseudo; GFX12+ splits
    // it into a SOPP wait and a SOP1 signal (both IMM and M0 forms). All
    // `_ISFIRST`/`_INIT`/`_JOIN`/`_LEAVE` variants are intentionally left
    // unmapped — the raiser does not model them yet.
    E(S_BARRIER, S_BARRIER),
    E(S_BARRIER_WAIT, S_BARRIER_WAIT),
    E(S_BARRIER_SIGNAL_IMM, S_BARRIER_SIGNAL),
    E(S_BARRIER_SIGNAL_M0, S_BARRIER_SIGNAL),

    // ---------------------------------------------------------------------
    // SMEM scalar loads
    // ---------------------------------------------------------------------
    SMEM3(S_LOAD_DWORD,    S_LOAD_B32),
    SMEM3(S_LOAD_DWORDX2,  S_LOAD_B64),
    SMEM3(S_LOAD_DWORDX3,  S_LOAD_B96),
    SMEM3(S_LOAD_DWORDX4,  S_LOAD_B128),
    SMEM3(S_LOAD_DWORDX8,  S_LOAD_B256),
    SMEM3(S_LOAD_DWORDX16, S_LOAD_B512),
    SMEM3(S_STORE_DWORD,   S_STORE_B32),
    SMEM3(S_STORE_DWORDX2, S_STORE_B64),
    SMEM3(S_STORE_DWORDX4, S_STORE_B128),

    // ---------------------------------------------------------------------
    // SOPC
    // ---------------------------------------------------------------------
    E(S_CMP_EQ_U32, S_CMP_EQ_U32), E(S_CMP_LG_U32, S_CMP_LG_U32),
    E(S_CMP_GT_U32, S_CMP_GT_U32), E(S_CMP_GE_U32, S_CMP_GE_U32),
    E(S_CMP_LT_U32, S_CMP_LT_U32), E(S_CMP_LE_U32, S_CMP_LE_U32),
    E(S_CMP_EQ_U64, S_CMP_EQ_U64), E(S_CMP_LG_U64, S_CMP_LG_U64),
    E(S_CMP_EQ_I32, S_CMP_EQ_I32), E(S_CMP_LG_I32, S_CMP_LG_I32),
    E(S_CMP_GT_I32, S_CMP_GT_I32), E(S_CMP_GE_I32, S_CMP_GE_I32),
    E(S_CMP_LT_I32, S_CMP_LT_I32), E(S_CMP_LE_I32, S_CMP_LE_I32),
    E(S_CMP_EQ_F32, S_CMP_EQ_F32), E(S_CMP_LG_F32, S_CMP_LG_F32),
    E(S_CMP_GT_F32, S_CMP_GT_F32), E(S_CMP_GE_F32, S_CMP_GE_F32),
    E(S_CMP_LT_F32, S_CMP_LT_F32), E(S_CMP_LE_F32, S_CMP_LE_F32),
    E(S_CMP_NEQ_F32, S_CMP_NEQ_F32),
    E(S_CMP_NGT_F32, S_CMP_NGT_F32), E(S_CMP_NGE_F32, S_CMP_NGE_F32),
    E(S_CMP_NLT_F32, S_CMP_NLT_F32), E(S_CMP_NLE_F32, S_CMP_NLE_F32),
    E(S_CMP_NLG_F32, S_CMP_NLG_F32),
    E(S_CMP_EQ_F16, S_CMP_EQ_F16), E(S_CMP_LG_F16, S_CMP_LG_F16),
    E(S_CMP_GT_F16, S_CMP_GT_F16), E(S_CMP_GE_F16, S_CMP_GE_F16),
    E(S_CMP_LT_F16, S_CMP_LT_F16), E(S_CMP_LE_F16, S_CMP_LE_F16),
    E(S_CMP_NEQ_F16, S_CMP_NEQ_F16),
    E(S_CMP_NGT_F16, S_CMP_NGT_F16), E(S_CMP_NGE_F16, S_CMP_NGE_F16),
    E(S_CMP_NLT_F16, S_CMP_NLT_F16), E(S_CMP_NLE_F16, S_CMP_NLE_F16),
    E(S_CMP_NLG_F16, S_CMP_NLG_F16),

    // ---------------------------------------------------------------------
    // SOPK
    // ---------------------------------------------------------------------
    E(S_MOVK_I32, S_MOVK_I32),
    E(S_ADDK_I32, S_ADDK_I32),
    E(S_MULK_I32, S_MULK_I32),
    E(S_CMPK_GE_I32, S_CMPK_GE_I32), E(S_CMPK_GT_I32, S_CMPK_GT_I32),
    E(S_CMPK_LE_I32, S_CMPK_LE_I32), E(S_CMPK_LT_I32, S_CMPK_LT_I32),
    E(S_CMPK_GE_U32, S_CMPK_GE_U32), E(S_CMPK_GT_U32, S_CMPK_GT_U32),
    E(S_CMPK_LE_U32, S_CMPK_LE_U32), E(S_CMPK_LT_U32, S_CMPK_LT_U32),
    E(S_CMPK_EQ_I32, S_CMPK_EQ_I32), E(S_CMPK_EQ_U32, S_CMPK_EQ_U32),
    E(S_CMPK_LG_I32, S_CMPK_LG_I32), E(S_CMPK_LG_U32, S_CMPK_LG_U32),
    E(S_GETREG_B32, S_GETREG_B32),
    E(S_SETREG_B32, S_SETREG_B32),
    E(S_SETREG_IMM32_B32, S_SETREG_IMM32_B32),

    // ---------------------------------------------------------------------
    // SOP1
    // ---------------------------------------------------------------------
    E(S_MOV_B32, S_MOV_B32), E(S_MOV_B64, S_MOV_B64),
    E(S_NOT_B32, S_NOT_B32), E(S_NOT_B64, S_NOT_B64),
    E(S_BREV_B32, S_BREV_B32),
    E(S_FF0_I32_B32, S_FF0_I32_B32), E(S_FF0_I32_B64, S_FF0_I32_B64),
    E(S_FF1_I32_B32, S_FF1_I32_B32), E(S_FF1_I32_B64, S_FF1_I32_B64),
    E(S_FLBIT_I32_B32, S_FLBIT_I32_B32), E(S_FLBIT_I32_B64, S_FLBIT_I32_B64),
    E(S_FLBIT_I32, S_FLBIT_I32), E(S_FLBIT_I32_I64, S_FLBIT_I32_I64),
    E(S_SEXT_I32_I8, S_SEXT_I32_I8), E(S_SEXT_I32_I16, S_SEXT_I32_I16),
    E(S_CVT_F32_U32, S_CVT_F32_U32), E(S_CVT_F32_I32, S_CVT_F32_I32),
    E(S_CVT_U32_F32, S_CVT_U32_F32), E(S_CVT_I32_F32, S_CVT_I32_F32),
    E(S_AND_SAVEEXEC_B32, S_AND_SAVEEXEC_B32),
    E(S_OR_SAVEEXEC_B32, S_OR_SAVEEXEC_B32),
    E(S_XOR_SAVEEXEC_B32, S_XOR_SAVEEXEC_B32),
    E(S_ANDN2_SAVEEXEC_B32, S_ANDN2_SAVEEXEC_B32),
    E(S_ORN2_SAVEEXEC_B32, S_ORN2_SAVEEXEC_B32),
    // GFX9 B64 aliases: handler uses EXEC-width-aware ops, same semantics.
    E(S_AND_SAVEEXEC_B64, S_AND_SAVEEXEC_B32),
    E(S_OR_SAVEEXEC_B64, S_OR_SAVEEXEC_B32),
    E(S_XOR_SAVEEXEC_B64, S_XOR_SAVEEXEC_B32),
    E(S_ANDN2_SAVEEXEC_B64, S_ANDN2_SAVEEXEC_B32),
    E(S_ORN2_SAVEEXEC_B64, S_ORN2_SAVEEXEC_B32),
    E(S_GETPC_B64, S_GETPC_B64),
    // gfx1250 asm renames `S_SETPC_B64` to `s_set_pc_i64`
    // (SOPInstructions.td:2208 declares the rename via
    // `SOP1_Real_gfx1250<0x048, "s_set_pc_i64">`); the LLVM MC opcode
    // remains `S_SETPC_B64`. We canonicalise both names onto
    // SemOp::S_SET_PC_I64 so the handler dispatches uniformly across
    // gfx versions; the new name is the SemOp because that is how the
    // corpus surfaces it (gfx1250 disasm) and what the SemOp enum
    // comment in semop.hpp documents.
    E(S_SETPC_B64, S_SET_PC_I64),
    // gfx1250 asm renames `S_SWAPPC_B64` to `s_swap_pc_i64`
    // (SOPInstructions.td:336 declares the SOP1_64 with isCall=1;
    // line 2311 renames the asm string via
    // `SOP1_Real_gfx1250<0x049, "s_swap_pc_i64">`); the LLVM MC
    // opcode remains `S_SWAPPC_B64`. We canonicalise both names
    // onto SemOp::S_SWAP_PC_I64 so the handler dispatches uniformly
    // across gfx versions; the new name is the SemOp because that
    // is how the corpus surfaces it (gfx1250 disasm) and what the
    // SemOp enum comment in semop.hpp documents.
    E(S_SWAPPC_B64, S_SWAP_PC_I64),
    E(S_ABS_I32, S_ABS_I32),
    E(S_SET_VGPR_MSB, S_SET_VGPR_MSB),
    E(S_BITSET0_B32, S_BITSET0_B32),
    E(S_BITSET1_B32, S_BITSET1_B32),
    E(S_BITSET0_B64, S_BITSET0_B64),
    E(S_BITSET1_B64, S_BITSET1_B64),
    // S_CMOV uses [SCC] but does not write SCC, and has no _e64 form
    // (SOP1 has only one encoding). The B64 alias is included even
    // though the corpus first surfaced only the B32 form, because
    // the B64 form ships in the same gfx1250 SOP1 table at adjacent
    // opcode (.td: 0x002 / 0x003) and the handler is symmetric.
    E(S_CMOV_B32, S_CMOV_B32),
    E(S_CMOV_B64, S_CMOV_B64),

    // ---------------------------------------------------------------------
    // SOP2
    // ---------------------------------------------------------------------
    E(S_ADD_U32, S_ADD_U32), E(S_ADD_I32, S_ADD_U32),
    E(S_ADDC_U32, S_ADDC_U32),
    E(S_SUB_U32, S_SUB_U32), E(S_SUB_I32, S_SUB_U32),
    E(S_SUBB_U32, S_SUBB_U32),
    E(S_ADD_U64, S_ADD_U64),
    E(S_AND_B32, S_AND_B32), E(S_AND_B64, S_AND_B64),
    E(S_OR_B32, S_OR_B32), E(S_OR_B64, S_OR_B64),
    E(S_XOR_B32, S_XOR_B32), E(S_XOR_B64, S_XOR_B64),
    E(S_ANDN2_B32, S_ANDN2_B32), E(S_ANDN2_B64, S_ANDN2_B64),
    E(S_ORN2_B32, S_ORN2_B32), E(S_ORN2_B64, S_ORN2_B64),
    E(S_NAND_B32, S_NAND_B32), E(S_NAND_B64, S_NAND_B64),
    E(S_NOR_B32, S_NOR_B32), E(S_NOR_B64, S_NOR_B64),
    E(S_XNOR_B32, S_XNOR_B32), E(S_XNOR_B64, S_XNOR_B64),
    E(S_ABSDIFF_I32, S_ABSDIFF_I32),
    E(S_LSHL_B32, S_LSHL_B32), E(S_LSHL_B64, S_LSHL_B64),
    E(S_LSHR_B32, S_LSHR_B32), E(S_LSHR_B64, S_LSHR_B64),
    E(S_ASHR_I32, S_ASHR_I32), E(S_ASHR_I64, S_ASHR_I64),
    E(S_MUL_I32, S_MUL_I32), E(S_MUL_HI_U32, S_MUL_HI_U32),
    E(S_MUL_HI_I32, S_MUL_HI_I32),
    E(S_MUL_U64, S_MUL_U64), E(S_MUL_F32, S_MUL_F32),
    E(S_ADD_F32, S_ADD_F32), E(S_SUB_F32, S_SUB_F32),
    E(S_BFE_U32, S_BFE_U32), E(S_BFE_I32, S_BFE_I32),
    E(S_BFM_B32, S_BFM_B32), E(S_BFM_B64, S_BFM_B64),
    E(S_CSELECT_B32, S_CSELECT_B32), E(S_CSELECT_B64, S_CSELECT_B64),
    E(S_MIN_I32, S_MIN_I32), E(S_MIN_U32, S_MIN_U32),
    E(S_MAX_I32, S_MAX_I32), E(S_MAX_U32, S_MAX_U32),
    E(S_PACK_LL_B32_B16, S_PACK_LL_B32_B16),
    E(S_PACK_LH_B32_B16, S_PACK_LH_B32_B16),
    E(S_LSHL1_ADD_U32, S_LSHL1_ADD_U32), E(S_LSHL2_ADD_U32, S_LSHL2_ADD_U32),
    E(S_LSHL3_ADD_U32, S_LSHL3_ADD_U32), E(S_LSHL4_ADD_U32, S_LSHL4_ADD_U32),
    // LLVM's pseudo is `S_ADD_U64`; we still surface it as `SemOp::S_ADD_NC_U64`
    // to keep handler parity with the no-carry mnemonic used downstream.
    E(S_ADD_U64, S_ADD_NC_U64),
    // gfx12 `s_sub_nc_u64` (renamed from `s_sub_u64` in the
    // assembler — see SOPInstructions.td 2311
    // `S_SUB_U64 ... "s_sub_nc_u64"`). LLVM's pseudo is still
    // `S_SUB_U64`; surface it as `SemOp::S_SUB_NC_U64` to mirror
    // the S_ADD_NC_U64 convention.
    E(S_SUB_U64, S_SUB_NC_U64),

    // ---------------------------------------------------------------------
    // VOP1 (canonical form is `_e64` after getVOPe64 collapses e32)
    // ---------------------------------------------------------------------
    E(V_MOV_B32_e64, V_MOV_B32),
    E(V_MOV_B64_e64, V_MOV_B64),
    E(V_MOV_B64_PSEUDO, V_MOV_B64),
    E(V_SWAP_B32, V_SWAP_B32),
    E(V_NOP_e64, V_NOP),
    E(V_NOT_B32_e64, V_NOT_B32),
    E(V_BFREV_B32_e64, V_BFREV_B32),
    E(V_FFBH_U32_e64, V_FFBH_U32),
    E(V_FFBL_B32_e64, V_FFBL_B32),
    E(V_FFBH_I32_e64, V_FFBH_I32),
    E(V_CVT_F32_I32_e64, V_CVT_F32_I32),
    E(V_CVT_F32_U32_e64, V_CVT_F32_U32),
    E(V_CVT_I32_F32_e64, V_CVT_I32_F32),
    E(V_CVT_U32_F32_e64, V_CVT_U32_F32),
    E(V_CVT_F16_F32_e64, V_CVT_F16_F32),
    E(V_CVT_F32_F16_e64, V_CVT_F32_F16),
    E(V_CVT_F32_BF16_e64, V_CVT_F32_BF16),
    E(V_CVT_PK_F32_FP8_e64, V_CVT_PK_F32_FP8),
    E(V_CVT_F32_UBYTE0_e64, V_CVT_F32_UBYTE0),
    E(V_CVT_F32_UBYTE1_e64, V_CVT_F32_UBYTE1),
    E(V_CVT_F32_UBYTE2_e64, V_CVT_F32_UBYTE2),
    E(V_CVT_F32_UBYTE3_e64, V_CVT_F32_UBYTE3),
    E(V_CVT_F64_U32_e64, V_CVT_F64_U32),
    E(V_CVT_F64_I32_e64, V_CVT_F64_I32),
    E(V_CVT_U32_F64_e64, V_CVT_U32_F64),
    E(V_RCP_IFLAG_F32_e64, V_RCP_IFLAG_F32),
    E(V_RCP_F32_e64, V_RCP_F32),
    E(V_RSQ_F32_e64, V_RSQ_F32),
    E(V_SQRT_F32_e64, V_SQRT_F32),
    E(V_EXP_F32_e64, V_EXP_F32),
    E(V_LOG_F32_e64, V_LOG_F32),
    E(V_FLOOR_F32_e64, V_FLOOR_F32),
    E(V_CEIL_F32_e64, V_CEIL_F32),
    E(V_TRUNC_F32_e64, V_TRUNC_F32),
    E(V_FRACT_F32_e64, V_FRACT_F32),
    E(V_READFIRSTLANE_B32, V_READFIRSTLANE_B32),
    E(V_FLOOR_F16_e64, V_FLOOR_F16),
    E(V_CVT_F16_U16_e64, V_CVT_F16_U16),
    E(V_CVT_U16_F16_e64, V_CVT_U16_F16),

    // ---------------------------------------------------------------------
    // VOP2 / VOP3
    // ---------------------------------------------------------------------
    E(V_ADD_F32_e64, V_ADD_F32),
    E(V_SUB_F32_e64, V_SUB_F32),
    E(V_SUBREV_F32_e64, V_SUBREV_F32),
    E(V_MUL_F32_e64, V_MUL_F32),
    E(V_FMAC_F32_e64, V_FMAC_F32),
    E(V_FMA_F32_e64, V_FMA_F32),
    E(V_FMAMK_F32, V_FMAMK_F32),
    E(V_FMAAK_F32, V_FMAAK_F32),
    E(V_MAX_F32_e64, V_MAX_F32),
    E(V_MIN_F32_e64, V_MIN_F32),
    // gfx11+ IEEE-754 2019 maximum/minimum. These differ from V_MAX_NUM_F32 /
    // V_MIN_NUM_F32 in their NaN semantics: maximum/minimum propagate NaN,
    // while maxnum/minnum return the non-NaN operand.
    E(V_MAXIMUM_F32_e64, V_MAXIMUM_F32),
    E(V_MINIMUM_F32_e64, V_MINIMUM_F32),
    E(V_DIV_FIXUP_F32_e64, V_DIV_FIXUP_F32),
    E(V_DIV_FMAS_F32_e64, V_DIV_FMAS_F32),
    E(V_DIV_SCALE_F32_e64, V_DIV_SCALE_F32),
    // LLVM's no-carry 32-bit add/sub pseudos are just `V_ADD_U32` etc.; the
    // carry-in-carry-out form is the older `V_ADDC_U32`/`V_SUBB_U32` family.
    E(V_ADD_U32_e64, V_ADD_NC_U32),
    E(V_SUB_U32_e64, V_SUB_NC_U32),
    E(V_SUBREV_U32_e64, V_SUBREV_NC_U32),
    // GFX9 VOP3-only signed add/sub (saddsat/ssubsat when clamp is set,
    // plain add/sub otherwise). Distinct from the legacy GFX10+ NC variants.
    E(V_ADD_I32_e64, V_ADD_I32),
    E(V_SUB_I32_e64, V_SUB_I32),
    E(V_ADD_CO_U32_e64, V_ADD_CO_U32),
    E(V_ADDC_U32_e64, V_ADD_CO_CI_U32),
    E(V_SUB_CO_U32_e64, V_SUB_CO_U32),
    E(V_SUBREV_CO_U32_e64, V_SUBREV_CO_U32),
    E(V_SUBB_U32_e64, V_SUB_CO_CI_U32),
    E(V_SUBBREV_U32_e64, V_SUBREV_CO_CI_U32),
    E(V_AND_B32_e64, V_AND_B32),
    E(V_OR_B32_e64, V_OR_B32),
    E(V_XOR_B32_e64, V_XOR_B32),
    E(V_XNOR_B32_e64, V_XNOR_B32),
    E(V_LSHLREV_B32_e64, V_LSHLREV_B32),
    E(V_LSHRREV_B32_e64, V_LSHRREV_B32),
    E(V_ASHRREV_I32_e64, V_ASHRREV_I32),
    E(V_CNDMASK_B32_e64, V_CNDMASK_B32),
    E(V_MUL_LO_U32_e64, V_MUL_LO_U32),
    E(V_MUL_HI_U32_e64, V_MUL_HI_U32),
    E(V_MUL_HI_I32_e64, V_MUL_HI_I32),
    E(V_MUL_I32_I24_e64, V_MUL_I32_I24),
    E(V_MUL_U32_U24_e64, V_MUL_U32_U24),
    E(V_MUL_HI_U32_U24_e64, V_MUL_HI_U32_U24),
    E(V_MUL_HI_I32_I24_e64, V_MUL_HI_I32_I24),
    E(V_MAD_U32_U24_e64, V_MAD_U32_U24),
    E(V_MAD_U32_e64, V_MAD_U32),
    E(V_ADD3_U32_e64, V_ADD3_U32),
    E(V_LSHL_ADD_U32_e64, V_LSHL_ADD_U32),
    E(V_ADD_LSHL_U32_e64, V_ADD_LSHL_U32),
    E(V_LSHL_OR_B32_e64, V_LSHL_OR_B32),
    E(V_AND_OR_B32_e64, V_AND_OR_B32),
    E(V_OR3_B32_e64, V_OR3_B32),
    // gfx9+ ternary xor+add (VOP3Instructions.td:684, iselect
    // pattern at :831 — `add(xor(a, b), c)`).
    E(V_XAD_U32_e64, V_XAD_U32),
    // gfx6+ funnel-shift right (VOP3Instructions.td:218,
    // gfx11/gfx12 opcode 0x216 at :1787). The `_t16` and
    // `_fake16` variants collapse onto V_ALIGNBIT_B32_e64 via
    // sameSemanticShape.
    E(V_ALIGNBIT_B32_e64, V_ALIGNBIT_B32),
    // gfx10+ ternary xor (VOP3Instructions.td:1348). VOP3
    // ternaries only have the e64 form.
    E(V_XOR3_B32_e64, V_XOR3_B32),
    // gfx10+ VOP3 16-bit no-carry add (VOP3Instructions.td:1362,
    // real opcode 0x303 at :1852/2016). The opcode_map's
    // sameSemanticShape canonicalizer collapses
    // V_ADD_NC_U16_t16_e64 and V_ADD_NC_U16_fake16_e64 onto this
    // base entry automatically, so a single mapping suffices.
    E(V_ADD_NC_U16_e64, V_ADD_NC_U16),
    E(V_BFE_U32_e64, V_BFE_U32),
    E(V_BFE_I32_e64, V_BFE_I32),
    E(V_PERM_B32_e64, V_PERM_B32),
    E(V_MBCNT_LO_U32_B32_e64, V_MBCNT_LO_U32_B32),
    E(V_MBCNT_HI_U32_B32_e64, V_MBCNT_HI_U32_B32),
    E(V_READLANE_B32, V_READLANE_B32),
    E(V_WRITELANE_B32, V_WRITELANE_B32),
    E(V_MED3_F32_e64, V_MED3_F32),
    E(V_MAX3_F32_e64, V_MAX3_F32),
    E(V_MIN3_F32_e64, V_MIN3_F32),
    // gfx11 V_MINMAX_F32 (opcode 0x25f) and the gfx12-renamed
    // V_MINMAX_NUM_F32 (opcode 0x268) share the same minnum-of-
    // maxnum semantics; both LLVM pseudos canonicalize onto
    // V_MINMAX_F32_e64 so a single mapping suffices.
    E(V_MINMAX_F32_e64, V_MINMAX_NUM_F32),
    // gfx11/gfx12 VOP3 integer 3-way max (opcode 0x21e, see
    // VOP3Instructions.td:1795). LLVM only emits the e64 form for
    // VOP3 ternaries; no DPP variant exists for v_max3_*.
    E(V_MAX3_U32_e64, V_MAX3_U32),
    // LLVM does not yet expose a `V_MAX3_MAXIMUM_F32` pseudo; leave the
    // `V_MAX3_NUM_F32` SemOp unmapped until it does.
    E(V_BITOP3_B32_e64, V_BITOP3_B32),
    E(V_BITOP3_B16_e64, V_BITOP3_B16),
    E(V_FMA_MIX_F32, V_FMA_MIX_F32),
    E(V_ADD_F16_e64, V_ADD_F16),
    E(V_MUL_F16_e64, V_MUL_F16),
    E(V_SUB_F16_e64, V_SUB_F16),
    E(V_SUBREV_F16_e64, V_SUBREV_F16),
    E(V_MAC_F16_e64, V_MAC_F16),
    E(V_MADMK_F16, V_MADMK_F16),
    E(V_MADAK_F16, V_MADAK_F16),
    E(V_FMAC_F16_e64, V_FMAC_F16),
    E(V_MAX_F16_e64, V_MAX_F16),
    E(V_MIN_F16_e64, V_MIN_F16),
    E(V_LDEXP_F16_e64, V_LDEXP_F16),
    E(V_LDEXP_F32_e64, V_LDEXP_F32),
    E(V_DOT2C_I32_I16_e64, V_DOT2C_I32_I16),
    E(V_DOT4C_I32_I8_e64, V_DOT4C_I32_I8),
    E(V_DOT8C_I32_I4_e64, V_DOT8C_I32_I4),
    E(V_PK_FMAC_F16_e64, V_PK_FMAC_F16),
    E(V_MAX_U16_e64, V_MAX_U16),
    E(V_MIN_U16_e64, V_MIN_U16),
    E(V_MAX_I16_e64, V_MAX_I16),
    E(V_MIN_I16_e64, V_MIN_I16),
    E(V_ADD_U16_e64, V_ADD_U16),
    E(V_SUB_U16_e64, V_SUB_U16),
    E(V_SUBREV_U16_e64, V_SUBREV_U16),
    E(V_MUL_LO_U16_e64, V_MUL_LO_U16),
    E(V_ASHRREV_I16_e64, V_ASHRREV_I16),
    E(V_LSHRREV_B16_e64, V_LSHRREV_B16),
    E(V_LSHLREV_B16_e64, V_LSHLREV_B16),
    E(V_PACK_B32_F16_e64, V_PACK_B32_F16),
    E(V_CVT_PK_BF16_F32_e64, V_CVT_PK_BF16_F32),
    E(V_CVT_PK_BF8_F32_e64, V_CVT_PK_BF8_F32),
    E(V_CVT_PK_FP8_F32_e64, V_CVT_PK_FP8_F32),
    E(V_CVT_PKRTZ_F16_F32_e64, V_CVT_PKRTZ_F16_F32),
    E(V_CVT_PK_F16_F32_e64, V_CVT_PK_F16_F32),
    E(V_CVT_SCALEF32_PK_FP4_F32_e64, V_CVT_SCALEF32_PK_FP4_F32),
    E(V_BFM_B32_e64, V_BFM_B32),

    // ---------------------------------------------------------------------
    // FP64
    // ---------------------------------------------------------------------
    E(V_ADD_F64_e64, V_ADD_F64),
    E(V_MUL_F64_e64, V_MUL_F64),
    E(V_FMA_F64_e64, V_FMA_F64),
    E(V_FMAC_F64_e64, V_FMAC_F64),
    // V_RCP_F64 has no DPP (`VOP1_Real_..._NO_DPP_...`); the e32
    // form is collapsed to e64 by getVOPe64 before lookup, so a
    // single e64 entry covers both encodings.
    E(V_RCP_F64_e64, V_RCP_F64),

    // ---------------------------------------------------------------------
    // More VOP3 32-bit int min/max and lane perms
    // ---------------------------------------------------------------------
    E(V_MAX_U32_e64, V_MAX_U32),
    E(V_MIN_U32_e64, V_MIN_U32),
    E(V_MAX_I32_e64, V_MAX_I32),
    E(V_MIN_I32_e64, V_MIN_I32),
    E(V_PERMLANE16_B32_e64, V_PERMLANE16_B32),
    E(V_PERMLANEX16_B32_e64, V_PERMLANEX16_B32),
    E(V_PERMLANE64_B32, V_PERMLANE64_B32),
    E(V_PERMLANE16_SWAP_B32_e32, V_PERMLANE16_SWAP_B32),
    E(V_PERMLANE16_SWAP_B32_e64, V_PERMLANE16_SWAP_B32),
    E(V_PERMLANE32_SWAP_B32_e32, V_PERMLANE32_SWAP_B32),
    E(V_PERMLANE32_SWAP_B32_e64, V_PERMLANE32_SWAP_B32),

    // ---------------------------------------------------------------------
    // VOPC (V_CMP_* and V_CMPX_*) are NOT enumerated here. They're picked up
    // by scanning canonical pseudo names in `OpcodeMap::build` and routed to
    // the two collapsed SemOps (V_CMP / V_CMPX) with metadata (predicate,
    // element width, int/float) in the `vcmp_` side table.
    //
    // ---------------------------------------------------------------------
    // VOP3P (canonical is bare pseudo name, no _e64)
    // ---------------------------------------------------------------------
    E(V_PK_ADD_F32, V_PK_ADD_F32),
    E(V_PK_MUL_F32, V_PK_MUL_F32),
    E(V_PK_FMA_F32, V_PK_FMA_F32),
    // LLVM has no `V_PK_MAX_F32`/`V_PK_MIN_F32` pseudo (only F16 variants);
    // leave the matching SemOps unmapped until one appears.
    E(V_PK_MOV_B32, V_PK_MOV_B32),

    // ---------------------------------------------------------------------
    // 64-bit vector
    // ---------------------------------------------------------------------
    E(V_LSHLREV_B64_e64, V_LSHLREV_B64),
    // V_LSHRREV_B64 / V_ASHRREV_I64 — gfx8+ 64-bit logical / arithmetic
    // right shift with reversed operands (`dst = src1 >> shamt`). Only
    // the plain `_e64` pseudo exists; per-ISA realtriples (gfx10 / gfx11
    // / gfx1250) all collapse back to it through the disassembler's
    // pseudo-alias step (TableGen treats the gfx1250 form as a `_e64`
    // realtriple, not a `_t16_e64` half-precision variant — see
    // VOP3Instructions.td:2341 vs the b16 family at 2336-2340).
    E(V_LSHRREV_B64_e64, V_LSHRREV_B64),
    E(V_ASHRREV_I64_e64, V_ASHRREV_I64),
    E(V_LSHL_ADD_U64_e64, V_LSHL_ADD_U64),
    // LLVM's 64-bit no-carry add pseudo is simply `V_ADD_U64_e64`.
    E(V_ADD_U64_e64, V_ADD_NC_U64),
    // gfx1250 VOP2 V_MUL_U64; canonicalizer collapses `_gfx1250` suffix
    // onto the bare `V_MUL_U64_e64` pseudo.
    E(V_MUL_U64_e64, V_MUL_U64),
    E(V_MAD_U64_U32_e64, V_MAD_U64_U32),
    // LLVM no longer exposes a distinct carry-out variant; `V_MAD_CO_U64_U32`
    // SemOp stays unmapped until one reappears.

    // ---------------------------------------------------------------------
    // FLAT
    // ---------------------------------------------------------------------
    E(FLAT_LOAD_UBYTE, FLAT_LOAD_UBYTE), E(FLAT_LOAD_SBYTE, FLAT_LOAD_SBYTE),
    E(FLAT_LOAD_USHORT, FLAT_LOAD_USHORT), E(FLAT_LOAD_SSHORT, FLAT_LOAD_SSHORT),
    E(FLAT_LOAD_DWORD, FLAT_LOAD_DWORD), E(FLAT_LOAD_DWORDX2, FLAT_LOAD_DWORDX2),
    E(FLAT_LOAD_DWORDX3, FLAT_LOAD_DWORDX3), E(FLAT_LOAD_DWORDX4, FLAT_LOAD_DWORDX4),
    E(FLAT_STORE_BYTE, FLAT_STORE_BYTE), E(FLAT_STORE_SHORT, FLAT_STORE_SHORT),
    E(FLAT_STORE_SHORT_D16_HI, FLAT_STORE_SHORT_D16_HI),
    E(FLAT_STORE_DWORD, FLAT_STORE_DWORD), E(FLAT_STORE_DWORDX2, FLAT_STORE_DWORDX2),
    E(FLAT_STORE_DWORDX3, FLAT_STORE_DWORDX3), E(FLAT_STORE_DWORDX4, FLAT_STORE_DWORDX4),
    E(GLOBAL_LOAD_UBYTE, GLOBAL_LOAD_UBYTE), E(GLOBAL_LOAD_SBYTE, GLOBAL_LOAD_SBYTE),
    E(GLOBAL_LOAD_USHORT, GLOBAL_LOAD_USHORT), E(GLOBAL_LOAD_SSHORT, GLOBAL_LOAD_SSHORT),
    E(GLOBAL_LOAD_SHORT_D16_HI, GLOBAL_LOAD_SHORT_D16_HI),
    E(GLOBAL_LOAD_DWORD, GLOBAL_LOAD_DWORD), E(GLOBAL_LOAD_DWORDX2, GLOBAL_LOAD_DWORDX2),
    E(GLOBAL_LOAD_DWORDX3, GLOBAL_LOAD_DWORDX3), E(GLOBAL_LOAD_DWORDX4, GLOBAL_LOAD_DWORDX4),
    E(GLOBAL_STORE_BYTE, GLOBAL_STORE_BYTE), E(GLOBAL_STORE_SHORT, GLOBAL_STORE_SHORT),
    E(GLOBAL_STORE_SHORT_D16_HI, GLOBAL_STORE_SHORT_D16_HI),
    E(GLOBAL_STORE_DWORD, GLOBAL_STORE_DWORD), E(GLOBAL_STORE_DWORDX2, GLOBAL_STORE_DWORDX2),
    E(GLOBAL_STORE_DWORDX3, GLOBAL_STORE_DWORDX3), E(GLOBAL_STORE_DWORDX4, GLOBAL_STORE_DWORDX4),

    // ---------------------------------------------------------------------
    // FLAT atomics (canonicalized to non-SADDR via getGlobalVaddrOp; no-return form)
    // ---------------------------------------------------------------------
    E(FLAT_ATOMIC_ADD, FLAT_ATOMIC_ADD),
    E(FLAT_ATOMIC_SUB, FLAT_ATOMIC_SUB),
    E(FLAT_ATOMIC_AND, FLAT_ATOMIC_AND),
    E(FLAT_ATOMIC_OR, FLAT_ATOMIC_OR),
    E(FLAT_ATOMIC_XOR, FLAT_ATOMIC_XOR),
    E(FLAT_ATOMIC_SMIN, FLAT_ATOMIC_SMIN),
    E(FLAT_ATOMIC_SMAX, FLAT_ATOMIC_SMAX),
    E(FLAT_ATOMIC_UMIN, FLAT_ATOMIC_UMIN),
    E(FLAT_ATOMIC_UMAX, FLAT_ATOMIC_UMAX),
    E(FLAT_ATOMIC_SWAP, FLAT_ATOMIC_SWAP),
    E(FLAT_ATOMIC_CMPSWAP, FLAT_ATOMIC_CMPSWAP),
    E(FLAT_ATOMIC_ADD_F32, FLAT_ATOMIC_ADD_F32),

    // ---------------------------------------------------------------------
    // GLOBAL atomics
    // ---------------------------------------------------------------------
    E(GLOBAL_ATOMIC_ADD, GLOBAL_ATOMIC_ADD),
    E(GLOBAL_ATOMIC_SUB, GLOBAL_ATOMIC_SUB),
    E(GLOBAL_ATOMIC_AND, GLOBAL_ATOMIC_AND),
    E(GLOBAL_ATOMIC_OR, GLOBAL_ATOMIC_OR),
    E(GLOBAL_ATOMIC_XOR, GLOBAL_ATOMIC_XOR),
    E(GLOBAL_ATOMIC_SMIN, GLOBAL_ATOMIC_SMIN),
    E(GLOBAL_ATOMIC_SMAX, GLOBAL_ATOMIC_SMAX),
    E(GLOBAL_ATOMIC_UMIN, GLOBAL_ATOMIC_UMIN),
    E(GLOBAL_ATOMIC_UMAX, GLOBAL_ATOMIC_UMAX),
    E(GLOBAL_ATOMIC_SWAP, GLOBAL_ATOMIC_SWAP),
    E(GLOBAL_ATOMIC_CMPSWAP, GLOBAL_ATOMIC_CMPSWAP),
    E(GLOBAL_ATOMIC_ADD_F32, GLOBAL_ATOMIC_ADD_F32),
    E(GLOBAL_ATOMIC_PK_ADD_BF16, GLOBAL_ATOMIC_PK_ADD_BF16),
    E(GLOBAL_ATOMIC_PK_ADD_F16, GLOBAL_ATOMIC_PK_ADD_F16),

    // ---------------------------------------------------------------------
    // SMEM atomics (enumerate addressing forms: IMM / SGPR / SGPR_IMM)
    // ---------------------------------------------------------------------
    E(S_ATOMIC_SWAP_IMM, S_ATOMIC_SWAP),
    E(S_ATOMIC_SWAP_SGPR, S_ATOMIC_SWAP),
    E(S_ATOMIC_SWAP_SGPR_IMM, S_ATOMIC_SWAP),

    // ---------------------------------------------------------------------
    // DS
    // ---------------------------------------------------------------------
    E(DS_LOAD_TR16_B128, DS_LOAD_TR16_B128),
    E(DS_READ_B64_TR_B16, DS_READ_B64_TR_B16),
    E(DS_READ_B64_TR_B8, DS_READ_B64_TR_B8),
    // gfx1250 64-bit transposed LDS load (i8 element, 8 elements per
    // lane post-transpose). Distinct LLVM MC pseudo from the gfx950
    // `DS_READ_B64_TR_B8` sibling — same semantics, different isel
    // gate (isGFX1250Plus vs HasGFX950Insts). The handler folds both
    // paths into a shared hand-rolled emulation; canonicalising the
    // SemOp on the gfx1250 spelling matches the disassembly the
    // raise_cli operator sees.
    E(DS_LOAD_TR8_B64, DS_LOAD_TR8_B64),
    E(DS_READ_B32, DS_READ_B32), E(DS_READ_B64, DS_READ_B64),
    // 96-bit LDS load. LLVM MC keeps the legacy `DS_READ_B96`
    // pseudo name for what gfx11+ disassembles as `ds_load_b96`
    // (DSInstructions.td:1578); we canonicalise on the gfx11+
    // spelling, mirroring the s_set_pc_i64 / DS_LOAD_TR8_B64
    // precedent. The handler dispatches via the existing generic
    // DS read range — see dsClassify in handle_ds.cpp for the
    // {dwords=3, loadBits=96} entry.
    E(DS_READ_B96, DS_READ_B96),
    E(DS_READ_B128, DS_READ_B128),
    E(DS_READ2_B32, DS_READ2_B32), E(DS_READ2_B64, DS_READ2_B64),
    E(DS_READ_U16, DS_READ_U16), E(DS_READ_I16, DS_READ_I16),
    E(DS_READ_U8, DS_READ_U8), E(DS_READ_I8, DS_READ_I8),
    E(DS_WRITE_B32, DS_WRITE_B32), E(DS_WRITE_B64, DS_WRITE_B64),
    // Symmetric 96-bit LDS store. gfx11+ asm spelling is
    // `ds_store_b96` (DSInstructions.td:1576); the LLVM MC opcode
    // remains `DS_WRITE_B96`.
    E(DS_WRITE_B96, DS_WRITE_B96),
    E(DS_WRITE_B128, DS_WRITE_B128),
    E(DS_WRITE2_B32, DS_WRITE2_B32), E(DS_WRITE2_B64, DS_WRITE2_B64),
    E(DS_WRITE_B16, DS_WRITE_B16), E(DS_WRITE_B8, DS_WRITE_B8),
    // gfx8+ HasD16LoadStore D16_HI store family (DSInstructions.td
    // §604-606). Stores bits [31:16] (B16_HI) or bits [23:16] (B8_HI)
    // of the source VGPR to LDS — same VGPR/i32 source operand
    // shape as their non-_HI siblings, so the canonical-table macro
    // routes both encoding forks (gfx10 m0-based vs gfx11+ no-m0)
    // through the same SemOp.
    E(DS_WRITE_B16_D16_HI, DS_WRITE_B16_D16_HI),
    E(DS_WRITE_B8_D16_HI, DS_WRITE_B8_D16_HI),
    E(DS_BPERMUTE_B32, DS_BPERMUTE_B32),
    // ds_swizzle_b32 — wave-width-specific cross-lane shuffle. The
    // handler refuses with `unsupportedShape` until CROSS_LANE_SURVEY
    // P6 lands; the wave-size classifier (Phase 1.4.5) flags it as
    // SPE_DESIGN.md §3 Class 2 in the cross-wave case.
    E(DS_SWIZZLE_B32, DS_SWIZZLE_B32),

    // ---------------------------------------------------------------------
    // MUBUF direct-to-LDS loads (distinct semantics from VGPR-dest loads)
    // ---------------------------------------------------------------------
    // LLVM only ships DWORD and DWORDX4 LDS pseudos — the DWORDX2_LDS SemOp
    // stays unmapped until an LLVM pseudo exists for it.
    MUBUF4(BUFFER_LOAD_DWORD_LDS, BUFFER_LOAD_DWORD_LDS),
    MUBUF4(BUFFER_LOAD_DWORDX4_LDS, BUFFER_LOAD_DWORDX4_LDS),

    // ---------------------------------------------------------------------
    // MUBUF loads
    // ---------------------------------------------------------------------
    // gfx10- uses `BUFFER_LOAD_DWORD_*`; gfx11+ (vbuffer) uses the same
    // pseudo name with a `_VBUFFER_` infix rather than a `B32` rename.
    MUBUF4(BUFFER_LOAD_DWORD, BUFFER_LOAD_DWORD),
    VBUF4(BUFFER_LOAD_DWORD, BUFFER_LOAD_DWORD),
    MUBUF4(BUFFER_LOAD_DWORDX2, BUFFER_LOAD_DWORDX2),
    MUBUF4(BUFFER_LOAD_DWORDX3, BUFFER_LOAD_DWORDX3),
    MUBUF4(BUFFER_LOAD_DWORDX4, BUFFER_LOAD_DWORDX4),
    MUBUF4(BUFFER_LOAD_UBYTE, BUFFER_LOAD_UBYTE),
    MUBUF4(BUFFER_LOAD_SBYTE, BUFFER_LOAD_SBYTE),
    MUBUF4(BUFFER_LOAD_USHORT, BUFFER_LOAD_USHORT),
    MUBUF4(BUFFER_LOAD_SSHORT, BUFFER_LOAD_SSHORT),
    // D16 partial-write loads: the 16-bit (or 8-bit, sign/zero-extended
    // to i16) datum lands in the lo half (`_D16`) or hi half
    // (`_D16_HI`) of the destination VGPR, preserving the other 16
    // bits (BUFInstructions.td:1155-1177, predicate
    // `D16PreservesUnusedBits`). gfx10- uses the legacy MUBUF
    // encodings; gfx11+/gfx1250 fork to VBUFFER (BUFInstructions.td
    // 2610-2620 maps `_d16_*` mnemonics to the VBUFFER pseudos). Both
    // encodings dispatch to the same partial-write handler in
    // handle_mubuf.cpp via the SemOp; the `D16PreservesUnusedBits`
    // semantics is enforced there by reading the prior dst and
    // merging.
    MUBUF4(BUFFER_LOAD_SHORT_D16, BUFFER_LOAD_SHORT_D16),
    VBUF4(BUFFER_LOAD_SHORT_D16, BUFFER_LOAD_SHORT_D16),
    MUBUF4(BUFFER_LOAD_SHORT_D16_HI, BUFFER_LOAD_SHORT_D16_HI),
    VBUF4(BUFFER_LOAD_SHORT_D16_HI, BUFFER_LOAD_SHORT_D16_HI),
    MUBUF4(BUFFER_LOAD_UBYTE_D16, BUFFER_LOAD_UBYTE_D16),
    VBUF4(BUFFER_LOAD_UBYTE_D16, BUFFER_LOAD_UBYTE_D16),
    MUBUF4(BUFFER_LOAD_UBYTE_D16_HI, BUFFER_LOAD_UBYTE_D16_HI),
    VBUF4(BUFFER_LOAD_UBYTE_D16_HI, BUFFER_LOAD_UBYTE_D16_HI),
    MUBUF4(BUFFER_LOAD_SBYTE_D16, BUFFER_LOAD_SBYTE_D16),
    VBUF4(BUFFER_LOAD_SBYTE_D16, BUFFER_LOAD_SBYTE_D16),
    MUBUF4(BUFFER_LOAD_SBYTE_D16_HI, BUFFER_LOAD_SBYTE_D16_HI),
    VBUF4(BUFFER_LOAD_SBYTE_D16_HI, BUFFER_LOAD_SBYTE_D16_HI),

    // ---------------------------------------------------------------------
    // MUBUF stores
    // ---------------------------------------------------------------------
    MUBUF4(BUFFER_STORE_DWORD, BUFFER_STORE_DWORD),
    VBUF4(BUFFER_STORE_DWORD, BUFFER_STORE_DWORD),
    MUBUF4(BUFFER_STORE_DWORDX2, BUFFER_STORE_DWORDX2),
    MUBUF4(BUFFER_STORE_DWORDX3, BUFFER_STORE_DWORDX3),
    MUBUF4(BUFFER_STORE_DWORDX4, BUFFER_STORE_DWORDX4),
    MUBUF4(BUFFER_STORE_BYTE, BUFFER_STORE_BYTE),
    MUBUF4(BUFFER_STORE_SHORT, BUFFER_STORE_SHORT),

    // ---------------------------------------------------------------------
    // MUBUF atomics (non-return; handlers don't distinguish RTN yet)
    // ---------------------------------------------------------------------
    MUBUF4(BUFFER_ATOMIC_ADD, BUFFER_ATOMIC_ADD),
    MUBUF4(BUFFER_ATOMIC_SUB, BUFFER_ATOMIC_SUB),
    MUBUF4(BUFFER_ATOMIC_AND, BUFFER_ATOMIC_AND),
    MUBUF4(BUFFER_ATOMIC_OR, BUFFER_ATOMIC_OR),
    MUBUF4(BUFFER_ATOMIC_XOR, BUFFER_ATOMIC_XOR),
    // SPE_DESIGN.md §3 Class 3 non-commutative atomics. The
    // wave-size classifier flags them in the cross-wave case before
    // dispatch ever reaches handle_mubuf.cpp's switch (which would
    // otherwise refuse them via the default branch since they don't
    // fit the AtomicRMW commutative model).
    MUBUF4(BUFFER_ATOMIC_SWAP, BUFFER_ATOMIC_SWAP),
    MUBUF4(BUFFER_ATOMIC_CMPSWAP, BUFFER_ATOMIC_CMPSWAP),
    MUBUF4(BUFFER_ATOMIC_ADD_F32, BUFFER_ATOMIC_ADD_F32),
    MUBUF4(BUFFER_ATOMIC_PK_ADD_BF16, BUFFER_ATOMIC_PK_ADD_BF16),
    MUBUF4(BUFFER_ATOMIC_PK_ADD_F16, BUFFER_ATOMIC_PK_ADD_F16),
    // gfx11+/gfx12 VBUFFER fork for the buffer atomics. The asm
    // spelling on gfx11+/gfx1250 renames `BUFFER_ATOMIC_ADD` to
    // `buffer_atomic_add_u32` (BUFInstructions.td:2789 declares
    // `defm BUFFER_ATOMIC_ADD : MUBUF_Real_Atomic_gfx11_gfx12<0x035,
    // "buffer_atomic_add_u32">`); LLVM MC keeps the legacy
    // `BUFFER_ATOMIC_*` pseudo names but suffixes them with
    // `_VBUFFER_<addressing>` to distinguish the gfx12 buffer-
    // descriptor encoding from the legacy MUBUF one. The decoder in
    // mubuf_addr.cpp explicitly recognises both encodings (keys on
    // ParsedReg::Kind rather than operand position) so the existing
    // BUFFER_ATOMIC_* SemOps and the AtomicRMW lowering in
    // handle_mubuf.cpp's `sop >= BUFFER_ATOMIC_ADD &&
    // sop <= BUFFER_ATOMIC_PK_ADD_F16` branch work unchanged for
    // VBUFFER atomics. Without these entries the gfx1250 corpus
    // (e.g. scope_discovery___sum_bitmatrix_rows refused on
    // `buffer_atomic_add_u32 v0, v1, s[4:7], null offen`) would
    // refuse with `unsupportedOpcode` despite the underlying SemOp
    // being present.
    //
    // PK_ADD_BF16 / PK_ADD_F16 are intentionally omitted — they have
    // no VBUFFER Real form in BUFInstructions.td (the gfx12 buffer
    // packed-add fork uses a different mnemonic family); a stray
    // VBUF4 entry would expand to AMDGPU enum values that don't
    // exist and fail to compile.
    VBUF4(BUFFER_ATOMIC_ADD, BUFFER_ATOMIC_ADD),
    VBUF4(BUFFER_ATOMIC_SUB, BUFFER_ATOMIC_SUB),
    VBUF4(BUFFER_ATOMIC_AND, BUFFER_ATOMIC_AND),
    VBUF4(BUFFER_ATOMIC_OR, BUFFER_ATOMIC_OR),
    VBUF4(BUFFER_ATOMIC_XOR, BUFFER_ATOMIC_XOR),
    VBUF4(BUFFER_ATOMIC_SWAP, BUFFER_ATOMIC_SWAP),
    VBUF4(BUFFER_ATOMIC_CMPSWAP, BUFFER_ATOMIC_CMPSWAP),
    VBUF4(BUFFER_ATOMIC_ADD_F32, BUFFER_ATOMIC_ADD_F32),

    // ---------------------------------------------------------------------
    // AGPR moves
    // ---------------------------------------------------------------------
    E(V_ACCVGPR_READ_B32_e64, V_ACCVGPR_READ_B32),
    E(V_ACCVGPR_WRITE_B32_e64, V_ACCVGPR_WRITE_B32),

    // ---------------------------------------------------------------------
    // MFMA shapes with a single TableGen pseudo per intrinsic.
    // Register-class variants (`_vgprcd_`, `_mac_`) collapse to the base
    // `_e64` via pseudoAlias, so we only list the base pseudo here.
    // ---------------------------------------------------------------------
    E(V_MFMA_F32_16X16X16F16_e64, V_MFMA_F32_16x16x16_F16),
    E(V_MFMA_F32_32X32X8F16_e64,  V_MFMA_F32_32x32x8_F16),
    E(V_MFMA_F32_16X16X4F32_e64,  V_MFMA_F32_16x16x4_F32),
    E(V_MFMA_F32_32X32X1F32_e64,  V_MFMA_F32_32x32x1_F32),
    E(V_MFMA_F32_32X32X2F32_e64,  V_MFMA_F32_32x32x2_F32),
    E(V_MFMA_F32_4X4X1F32_e64,    V_MFMA_F32_4x4x1_F32),
    E(V_MFMA_F32_16X16X1F32_e64,  V_MFMA_F32_16x16x1_F32),
    E(V_MFMA_F32_32X32X4F16_e64,  V_MFMA_F32_32x32x4_F16),
    E(V_MFMA_F32_16X16X4F16_e64,  V_MFMA_F32_16x16x4_F16),
    E(V_MFMA_F32_4X4X4F16_e64,    V_MFMA_F32_4x4x4_F16),
    E(V_MFMA_I32_16X16X32I8_e64,  V_MFMA_I32_16x16x32_I8),
    E(V_MFMA_I32_32X32X16I8_e64,  V_MFMA_I32_32x32x16_I8),
    E(V_MFMA_I32_32X32X4I8_e64,   V_MFMA_I32_32x32x4_I8),
    E(V_MFMA_I32_16X16X4I8_e64,   V_MFMA_I32_16x16x4_I8),
    E(V_MFMA_I32_4X4X4I8_e64,     V_MFMA_I32_4x4x4_I8),
    E(V_MFMA_F32_16X16X8XF32_e64, V_MFMA_F32_16x16x8_XF32),
    E(V_MFMA_F32_32X32X4XF32_e64, V_MFMA_F32_32x32x4_XF32),
    E(V_MFMA_F32_32X32X2BF16_e64, V_MFMA_F32_32x32x2_BF16),
    E(V_MFMA_F32_16X16X2BF16_e64, V_MFMA_F32_16x16x2_BF16),
    E(V_MFMA_F32_4X4X2BF16_e64,   V_MFMA_F32_4x4x2_BF16),
    // Only the 16x16x16 and 32x32x8 1K shapes were in the legacy mnemonic
    // table; the 4x4x4/16x16x4/32x32x4 1K variants are distinct intrinsics
    // that the raiser does not (yet) model and therefore stay unmapped.
    E(V_MFMA_F32_16X16X16BF16_1K_e64, V_MFMA_F32_16x16x16_BF16_1K),
    E(V_MFMA_F32_32X32X8BF16_1K_e64,  V_MFMA_F32_32x32x8_BF16_1K),
    E(V_MFMA_F32_16X16X32_BF16_e64, V_MFMA_F32_16x16x32_BF16),
    E(V_MFMA_F32_32X32X16_BF16_e64, V_MFMA_F32_32x32x16_BF16),
    E(V_MFMA_F32_16X16X32_F16_e64,  V_MFMA_F32_16x16x32_F16),
    E(V_MFMA_F32_16X16X32_FP8_FP8_e64, V_MFMA_F32_16x16x32_FP8_FP8),
    E(V_MFMA_F32_16X16X32_FP8_BF8_e64, V_MFMA_F32_16x16x32_FP8_BF8),
    E(V_MFMA_F32_16X16X32_BF8_FP8_e64, V_MFMA_F32_16x16x32_BF8_FP8),
    E(V_MFMA_F32_16X16X32_BF8_BF8_e64, V_MFMA_F32_16x16x32_BF8_BF8),
    E(V_MFMA_F32_32X32X16_FP8_FP8_e64, V_MFMA_F32_32x32x16_FP8_FP8),
    E(V_MFMA_F32_32X32X16_FP8_BF8_e64, V_MFMA_F32_32x32x16_FP8_BF8),
    E(V_MFMA_F32_32X32X16_BF8_FP8_e64, V_MFMA_F32_32x32x16_BF8_FP8),
    E(V_MFMA_F32_32X32X16_BF8_BF8_e64, V_MFMA_F32_32x32x16_BF8_BF8),

    // ---------------------------------------------------------------------
    // MFMA F8F6F4: LLVM enumerates all 9 src0/src1 mantissa-type pairs as
    // separate pseudos; the raiser handles them identically.
    // ---------------------------------------------------------------------
    E(V_MFMA_F32_16X16X128_F8F6F4_f4_f4_e64, V_MFMA_F32_16x16x128_F8F6F4),
    E(V_MFMA_F32_16X16X128_F8F6F4_f4_f6_e64, V_MFMA_F32_16x16x128_F8F6F4),
    E(V_MFMA_F32_16X16X128_F8F6F4_f4_f8_e64, V_MFMA_F32_16x16x128_F8F6F4),
    E(V_MFMA_F32_16X16X128_F8F6F4_f6_f4_e64, V_MFMA_F32_16x16x128_F8F6F4),
    E(V_MFMA_F32_16X16X128_F8F6F4_f6_f6_e64, V_MFMA_F32_16x16x128_F8F6F4),
    E(V_MFMA_F32_16X16X128_F8F6F4_f6_f8_e64, V_MFMA_F32_16x16x128_F8F6F4),
    E(V_MFMA_F32_16X16X128_F8F6F4_f8_f4_e64, V_MFMA_F32_16x16x128_F8F6F4),
    E(V_MFMA_F32_16X16X128_F8F6F4_f8_f6_e64, V_MFMA_F32_16x16x128_F8F6F4),
    E(V_MFMA_F32_16X16X128_F8F6F4_f8_f8_e64, V_MFMA_F32_16x16x128_F8F6F4),

    E(V_MFMA_SCALE_F32_16X16X128_F8F6F4_f4_f4_e64, V_MFMA_SCALE_F32_16x16x128_F8F6F4),
    E(V_MFMA_SCALE_F32_16X16X128_F8F6F4_f4_f6_e64, V_MFMA_SCALE_F32_16x16x128_F8F6F4),
    E(V_MFMA_SCALE_F32_16X16X128_F8F6F4_f4_f8_e64, V_MFMA_SCALE_F32_16x16x128_F8F6F4),
    E(V_MFMA_SCALE_F32_16X16X128_F8F6F4_f6_f4_e64, V_MFMA_SCALE_F32_16x16x128_F8F6F4),
    E(V_MFMA_SCALE_F32_16X16X128_F8F6F4_f6_f6_e64, V_MFMA_SCALE_F32_16x16x128_F8F6F4),
    E(V_MFMA_SCALE_F32_16X16X128_F8F6F4_f6_f8_e64, V_MFMA_SCALE_F32_16x16x128_F8F6F4),
    E(V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f4_e64, V_MFMA_SCALE_F32_16x16x128_F8F6F4),
    E(V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f6_e64, V_MFMA_SCALE_F32_16x16x128_F8F6F4),
    E(V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f8_e64, V_MFMA_SCALE_F32_16x16x128_F8F6F4),

    E(V_MFMA_F32_32X32X64_F8F6F4_f4_f4_e64, V_MFMA_F32_32x32x64_F8F6F4),
    E(V_MFMA_F32_32X32X64_F8F6F4_f4_f6_e64, V_MFMA_F32_32x32x64_F8F6F4),
    E(V_MFMA_F32_32X32X64_F8F6F4_f4_f8_e64, V_MFMA_F32_32x32x64_F8F6F4),
    E(V_MFMA_F32_32X32X64_F8F6F4_f6_f4_e64, V_MFMA_F32_32x32x64_F8F6F4),
    E(V_MFMA_F32_32X32X64_F8F6F4_f6_f6_e64, V_MFMA_F32_32x32x64_F8F6F4),
    E(V_MFMA_F32_32X32X64_F8F6F4_f6_f8_e64, V_MFMA_F32_32x32x64_F8F6F4),
    E(V_MFMA_F32_32X32X64_F8F6F4_f8_f4_e64, V_MFMA_F32_32x32x64_F8F6F4),
    E(V_MFMA_F32_32X32X64_F8F6F4_f8_f6_e64, V_MFMA_F32_32x32x64_F8F6F4),
    E(V_MFMA_F32_32X32X64_F8F6F4_f8_f8_e64, V_MFMA_F32_32x32x64_F8F6F4),

    E(V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f4_e64, V_MFMA_SCALE_F32_32x32x64_F8F6F4),
    E(V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f6_e64, V_MFMA_SCALE_F32_32x32x64_F8F6F4),
    E(V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f8_e64, V_MFMA_SCALE_F32_32x32x64_F8F6F4),
    E(V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f4_e64, V_MFMA_SCALE_F32_32x32x64_F8F6F4),
    E(V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f6_e64, V_MFMA_SCALE_F32_32x32x64_F8F6F4),
    E(V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f8_e64, V_MFMA_SCALE_F32_32x32x64_F8F6F4),
    E(V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f4_e64, V_MFMA_SCALE_F32_32x32x64_F8F6F4),
    E(V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f6_e64, V_MFMA_SCALE_F32_32x32x64_F8F6F4),
    E(V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f8_e64, V_MFMA_SCALE_F32_32x32x64_F8F6F4),

    // ---------------------------------------------------------------------
    // WMMA (gfx1250): twoaddr/threeaddr pseudo variants both mean the same
    // semantic op to our raiser.
    // ---------------------------------------------------------------------
    E(V_WMMA_F32_16X16X32_F16_w32_twoaddr, V_WMMA_F32_16x16x32_F16),
    E(V_WMMA_F32_16X16X32_F16_w32_threeaddr, V_WMMA_F32_16x16x32_F16),
    // 16x16x32 WMMA, BF16 inputs, F32 accumulator (gfx1250 RDNA4 VOP3P
    // opcode 0x062). Same fragment shape as the F16 variant — the only
    // delta is the input element type, which routes to the bf16 MFMA
    // intrinsic on gfx942 (mfma_f32_16x16x16bf16_1k) and to
    // amdgcn_wmma_f32_16x16x32_bf16 when the target supports WMMA12.
    E(V_WMMA_F32_16X16X32_BF16_w32_twoaddr, V_WMMA_F32_16x16x32_BF16),
    E(V_WMMA_F32_16X16X32_BF16_w32_threeaddr, V_WMMA_F32_16x16x32_BF16),
    // 16x16x64 WMMA, 8-bit element types (gfx1250 RDNA4 VOP3P opcodes
    // 0x06a..0x06d). All four AB combinations share the per-lane
    // fragment shape (A,B: <8 x i32>, C/D: <8 x f32>) and route through
    // the gfx942 MFMA lowering with a per-variant intrinsic dispatch
    // (see `runGroupPass` in wmma_lowering.cpp). Both `_twoaddr` and
    // `_threeaddr` MC pseudo variants represent the same semantic op,
    // mirroring the F16/BF16 mapping above.
    E(V_WMMA_F32_16X16X64_FP8_FP8_w32_twoaddr,   V_WMMA_F32_16x16x64_FP8_FP8),
    E(V_WMMA_F32_16X16X64_FP8_FP8_w32_threeaddr, V_WMMA_F32_16x16x64_FP8_FP8),
    E(V_WMMA_F32_16X16X64_FP8_BF8_w32_twoaddr,   V_WMMA_F32_16x16x64_FP8_BF8),
    E(V_WMMA_F32_16X16X64_FP8_BF8_w32_threeaddr, V_WMMA_F32_16x16x64_FP8_BF8),
    E(V_WMMA_F32_16X16X64_BF8_FP8_w32_twoaddr,   V_WMMA_F32_16x16x64_BF8_FP8),
    E(V_WMMA_F32_16X16X64_BF8_FP8_w32_threeaddr, V_WMMA_F32_16x16x64_BF8_FP8),
    E(V_WMMA_F32_16X16X64_BF8_BF8_w32_twoaddr,   V_WMMA_F32_16x16x64_BF8_BF8),
    E(V_WMMA_F32_16X16X64_BF8_BF8_w32_threeaddr, V_WMMA_F32_16x16x64_BF8_BF8),
    // 16x16x64 WMMA, IU8 (signed/unsigned 8-bit integer inputs, i32
    // accumulator; gfx1250 RDNA4 VOP3P opcode 0x072). The MC opcode
    // sign-extension knobs travel through the `neg_lo` operand bits
    // rather than the opcode itself, so there's a single SemOp for
    // both signed and unsigned interpretations. On gfx942 this
    // routes to mfma_i32_16x16x32_i8 (i64 packed A/B, <4 x i32>
    // accumulator). Both `_twoaddr` and `_threeaddr` MC pseudo
    // variants represent the same semantic op, mirroring the
    // F16/BF16 / FP8/BF8 mappings above.
    E(V_WMMA_I32_16X16X64_IU8_w32_twoaddr,   V_WMMA_I32_16x16x64_IU8),
    E(V_WMMA_I32_16X16X64_IU8_w32_threeaddr, V_WMMA_I32_16x16x64_IU8),
    // ---------------------------------------------------------------------
    // VIMAGE TENSOR (gfx1250 RDNA4 — VIMAGE 0xc4 / 0xc5).
    // The disassembler's MC opcodes are the `_gfx1250` reals
    // (`TENSOR_LOAD_TO_LDS_d{2,4}_gfx1250`,
    // `TENSOR_STORE_FROM_LDS_d{2,4}_gfx1250`,
    // `MIMGInstructions.td:2115`); the canonicalization chain in
    // `OpcodeMap::canonicalize` collapses each real onto its pseudo
    // (`TENSOR_LOAD_TO_LDS_d{2,4}` / `TENSOR_STORE_FROM_LDS_d{2,4}`)
    // via `buildMcToPseudoMap`. Both the `_d2` (up-to-2D) and `_d4`
    // (up-to-4D) operand-shape variants share a SemOp because their
    // semantic intent and their cross-target refusal contract are
    // identical; `handleVIMAGE` uses `di.mnemonic` directly when it
    // needs to discriminate (e.g., a future native-target intrinsic
    // lowering that zero-fills D# group 2/3 for the `_d2` form).
    // ---------------------------------------------------------------------
    E(TENSOR_LOAD_TO_LDS_d2,    TENSOR_LOAD_TO_LDS),
    E(TENSOR_LOAD_TO_LDS_d4,    TENSOR_LOAD_TO_LDS),
    E(TENSOR_STORE_FROM_LDS_d2, TENSOR_STORE_FROM_LDS),
    E(TENSOR_STORE_FROM_LDS_d4, TENSOR_STORE_FROM_LDS),
};

#undef SMEM3
#undef VBUF4
#undef MUBUF4
#undef E

// Iteration bound for SIEncodingFamily: the enum in SIDefines.h is a closed
// numeric set with GFX13 as the current maximum, so we scan [0, GFX13] when
// inverting the pseudo -> MC map.  If LLVM adds a new family the next
// enumerator value appears here automatically and the build still compiles;
// the static_assert keeps us honest if LLVM ever renames the sentinel we use.
static_assert(SIEncodingFamily::GFX13 >= SIEncodingFamily::SI,
              "SIEncodingFamily enum layout changed unexpectedly");
constexpr unsigned kNumEncodingFamilies =
    static_cast<unsigned>(SIEncodingFamily::GFX13) + 1;

// Build a reverse map MC-opcode -> canonical pseudo by scanning every pseudo
// opcode across all subtarget generations. This is ~O(N * 15) work at init
// time (N ~= 70k AMDGPU opcodes on recent LLVM), which is well under a
// millisecond on modern hardware and done once per raiser.
DenseMap<unsigned, unsigned>
buildMcToPseudoMap(unsigned numOpc) {
  DenseMap<unsigned, unsigned> result;
  for (unsigned p = 0; p < numOpc; ++p) {
    for (unsigned gen = 0; gen < kNumEncodingFamilies; ++gen) {
      int mc = AMDGPU::getMCOpcode(p, gen);
      if (mc > 0 && static_cast<unsigned>(mc) != p)
        result.try_emplace(static_cast<unsigned>(mc), p);
    }
  }
  return result;
}

// Rule predicates: an optional semantic invariant the alias must preserve.
// Every time an alias step is committed (source pseudo S collapses onto
// target pseudo T), the firing rule's predicate is evaluated against
// MCInstrDesc(S) and MCInstrDesc(T). A violation means LLVM renamed or
// repurposed a pseudo in a way that breaks our naming contract, and is
// reported as a fatal error at init time rather than silently producing
// wrong IR at runtime.
using RulePredicate = bool (*)(const MCInstrDesc &src, const MCInstrDesc &tgt);

// `_RTN` collapse: source must be an atomic with a return value; target must
// be the same atomic without one. The raiser uses `numDefs` as the
// "publishes old value" signal, so that invariant must also hold.
static bool atomicRetToNoRet(const MCInstrDesc &src, const MCInstrDesc &tgt) {
  constexpr uint64_t kRet = SIInstrFlags::IsAtomicRet;
  constexpr uint64_t kNoRet = SIInstrFlags::IsAtomicNoRet;
  return (src.TSFlags & kRet) && (tgt.TSFlags & kNoRet) &&
         src.getNumDefs() > 0 && tgt.getNumDefs() == 0;
}

// `_vgprcd_` / `_mac_` collapse: both source and target must be MFMA
// (matrix-accumulate) pseudos.
static bool bothAreMAI(const MCInstrDesc &src, const MCInstrDesc &tgt) {
  constexpr uint64_t kMAI = SIInstrFlags::IsMAI;
  return (src.TSFlags & kMAI) && (tgt.TSFlags & kMAI);
}

// `_nosdst_` collapse: starting with GFX11, VOPC CMPX instructions no longer
// write a scalar destination register (EXEC receives the mask directly) and
// LLVM represents this as a `_nosdst_` variant. The non-`_nosdst_` target
// form keeps the scalar dst (for older subtargets). Both forms share
// dispatch-relevant TSFlags; the raiser's CMPX handler only writes EXEC and
// ignores the optional sdst, so collapsing the variant onto the base is
// safe. The source has one fewer def when the base includes sdst (e64
// forms) and the same number of defs otherwise (e32, where both lack sdst).

// Bits we require to be identical between source and target for an alias
// collapse to be considered semantically safe. Deliberately excludes encoding
// variation flags like `VOP3_OPSEL` (set on `_t16_` op-sel encodings but not
// on the base `_e64`) and `renamedInGFX9` (set only on the subtarget-specific
// pseudo). Everything listed below represents *what the handler dispatches
// on*: instruction family (SOP/VOP/FLAT/DS/...), atomic kind, and MAI.
static constexpr uint64_t kSemanticShapeMask =
    // Instruction families.
    SIInstrFlags::SOP1 | SIInstrFlags::SOP2 | SIInstrFlags::SOPC |
    SIInstrFlags::SOPK | SIInstrFlags::SOPP | SIInstrFlags::VOP1 |
    SIInstrFlags::VOP2 | SIInstrFlags::VOPC | SIInstrFlags::VOP3 |
    SIInstrFlags::VOP3P | SIInstrFlags::SDWA | SIInstrFlags::DPP |
    SIInstrFlags::MUBUF | SIInstrFlags::MTBUF | SIInstrFlags::SMRD |
    SIInstrFlags::FLAT | SIInstrFlags::DS | SIInstrFlags::MIMG |
    // Semantic classification (atomic kind, MAI).
    SIInstrFlags::IsAtomicRet | SIInstrFlags::IsAtomicNoRet |
    SIInstrFlags::IsMAI;

// Subtarget-/operand-class variants (`_gfx9`, `_t16_`, `_fake16_`, `_agpr`,
// etc.) may legitimately toggle encoding flags such as `VOP3_OPSEL` or
// `renamedInGFX9` between source and target, but they must preserve the
// instruction's dispatch identity: same family, same atomic kind, same MAI
// classification, same def arity. A violation means LLVM renamed or
// repurposed a pseudo in a way our alias map cannot safely collapse.
static bool sameSemanticShape(const MCInstrDesc &src,
                              const MCInstrDesc &tgt) {
  return (src.TSFlags & kSemanticShapeMask) ==
             (tgt.TSFlags & kSemanticShapeMask) &&
         src.getNumDefs() == tgt.getNumDefs();
}

// `_nosdst_` collapse: same dispatch identity as the base, and the source
// never has more defs than the target (the scalar dst is either dropped
// entirely or added back on the target's e64 form).
static bool nosdstDropsScalarDef(const MCInstrDesc &src,
                                 const MCInstrDesc &tgt) {
  return (src.TSFlags & kSemanticShapeMask) ==
             (tgt.TSFlags & kSemanticShapeMask) &&
         tgt.getNumDefs() >= src.getNumDefs() &&
         tgt.getNumDefs() - src.getNumDefs() <= 1;
}

// Build an alias map that collapses "parallel" pseudos LLVM generates for the
// same semantic instruction into a single canonical pseudo. Examples:
//   DS_WRITE_B16_gfx9        -> DS_WRITE_B16
//   V_ADD_F16_t16_e64        -> V_ADD_F16_e64
//   V_ADD_F16_fake16_e64     -> V_ADD_F16_e64
// LLVM does not expose a helper for this collapse, so we match on pseudo name
// at init time. Name lookups are confined to this one-shot scan over
// `MCII.getNumOpcodes()`; runtime lookups remain pure DenseMap hits.
DenseMap<unsigned, unsigned>
buildPseudoAliasMap(const MCInstrInfo &MCII) {
  unsigned numOpc = MCII.getNumOpcodes();

  llvm::StringMap<unsigned> byName;
  for (unsigned p = 0; p < numOpc; ++p)
    byName.try_emplace(MCII.getName(p), p);

  struct Rule {
    llvm::StringRef needle;
    bool isSuffix;
    // Optional semantic check on (source, target) MCInstrDesc. A null
    // predicate means "no validation yet" (see the older subtarget/operand
    // markers below).
    RulePredicate pred;
  };
  // Subtarget-specific markers ("_gfx9", "_gfx1250", ...) and operand-size
  // markers ("_t16_", "_fake16_") that LLVM injects into the pseudo name.
  // A single pseudo can carry multiple markers (e.g.
  // `V_BITOP3_B16_gfx1250_fake16_e64`), so the outer loop below applies these
  // rules iteratively until the name stops shrinking.
  static const Rule rules[] = {
      // Subtarget-specific markers. LLVM emits a dedicated pseudo per
      // subtarget (e.g. `_gfx9`, `_gfx1250`, `_vi_gfx9`) with the same
      // TableGen class as the base; collapsing them is sound as long as
      // TSFlags and def arity match.
      {"_vi_gfx9", true, sameSemanticShape},
      {"_gfx9", true, sameSemanticShape},
      {"_gfx1250", true, sameSemanticShape},
      {"_gfx1250_", false, sameSemanticShape},
      {"_pseudo_", false, sameSemanticShape},
      // True16 / Fake16 mark the 16-bit operand encoding variant; LLVM has
      // no dedicated TSFlag bit for this (the distinction lives in
      // True16Predicate on the TableGen side), and the t16 encoding toggles
      // `VOP3_OPSEL`. We cross-check that dispatch-relevant TSFlags and def
      // arity are preserved, but tolerate encoding-bit drift.
      {"_t16_", false, sameSemanticShape},
      {"_fake16_", false, sameSemanticShape},
      // GFX11+ VOPC CMPX family drops the scalar destination register; the
      // raiser's CMPX handler only touches EXEC so the `_nosdst_` form
      // collapses cleanly onto the base pseudo of the same encoding width.
      {"_nosdst_", false, nosdstDropsScalarDef},
      // MFMA register-class modifiers.  `_vgprcd_` marks a VGPR destination
      // variant; `_mac_` marks a multiply-accumulate (tied dst/src2) variant.
      // Both keep the same TableGen intrinsic and semantic shape, so they
      // collapse onto the base `_e64` pseudo.
      {"_vgprcd_", false, bothAreMAI},
      {"_mac_", false, bothAreMAI},
      // Atomic return-value variants: LLVM emits distinct `_RTN` pseudos for
      // the forms that return the pre-modification value, plus `_agpr`
      // variants that just pick an AGPR destination register class. These
      // pseudos carry the same TableGen intrinsic and identical semantics;
      // the only difference is whether the handler should write the result
      // back, which the raiser already derives from `di.numDefs`
      // (MCInstrDesc::getNumDefs()). Collapse them onto the non-RTN pseudo
      // so both forms share a single SemOp.
      {"_agpr", true, sameSemanticShape},
      {"_RTN", true, atomicRetToNoRet},
  };

  // Returns the index of the firing rule or -1 if no rule applies.
  auto stripOnce = [&](llvm::StringRef name, std::string &out) -> int {
    for (size_t i = 0; i < std::size(rules); ++i) {
      const Rule &r = rules[i];
      if (r.isSuffix) {
        if (!name.ends_with(r.needle))
          continue;
        out = name.drop_back(r.needle.size()).str();
        return static_cast<int>(i);
      }
      size_t pos = name.find(r.needle);
      if (pos == llvm::StringRef::npos)
        continue;
      out = (name.substr(0, pos).str() + std::string("_") +
             name.substr(pos + r.needle.size()).str());
      return static_cast<int>(i);
    }
    return -1;
  };

  DenseMap<unsigned, unsigned> alias;
  for (const auto &kv : byName) {
    std::string cur = kv.first().str();
    unsigned curOpc = kv.second;
    unsigned finalOpc = kv.second;
    while (true) {
      std::string next;
      int ruleIdx = stripOnce(cur, next);
      if (ruleIdx < 0)
        break;
      auto it = byName.find(next);
      if (it != byName.end() && it->second != kv.second) {
        const Rule &r = rules[ruleIdx];
        if (r.pred && !r.pred(MCII.get(curOpc), MCII.get(it->second))) {
          report_fatal_error(
              Twine("opcode_map: alias rule '") + r.needle +
              "' broke its semantic invariant while collapsing '" + cur +
              "' -> '" + next +
              "'. LLVM likely renamed or repurposed a pseudo; update the "
              "alias rules or the predicate.");
        }
        curOpc = it->second;
        finalOpc = curOpc;
      }
      cur = std::move(next);
    }
    if (finalOpc != kv.second)
      alias.try_emplace(kv.second, finalOpc);
  }
  return alias;
}

// Build a reverse DPP map: DPP opcode -> base VOP opcode. LLVM only provides
// forward mappings (base -> DPP32 / DPP64), so we invert by scanning.
DenseMap<unsigned, unsigned>
buildDppToBaseMap(unsigned numOpc) {
  DenseMap<unsigned, unsigned> result;
  for (unsigned p = 0; p < numOpc; ++p) {
    int d32 = AMDGPU::getDPPOp32(p);
    if (d32 > 0)
      result.try_emplace(static_cast<unsigned>(d32), p);
    int d64 = AMDGPU::getDPPOp64(p);
    if (d64 > 0)
      result.try_emplace(static_cast<unsigned>(d64), p);
  }
  return result;
}

// Canonicalize any MC opcode `mc` to the pseudo form matched in kCanonTable.
// The chain is:
//   MC -> pseudo                (TableGen Subtarget map)
//   pseudo -> base VOP          (strip DPP / SDWA)
//   e32 -> e64                  (collapse VOP encoding variants)
//   SADDR -> VADDR              (FLAT/GLOBAL global-saddr table)
unsigned canonicalize(unsigned mc,
                      const MCInstrInfo &MCII,
                      const DenseMap<unsigned, unsigned> &mcToPseudo,
                      const DenseMap<unsigned, unsigned> &pseudoAlias,
                      const DenseMap<unsigned, unsigned> &dppToBase) {
  unsigned p = mc;

  // MC (subtarget-specific real) -> pseudo.
  if (auto it = mcToPseudo.find(p); it != mcToPseudo.end())
    p = it->second;

  // Parallel-pseudo alias -> base pseudo (strips _gfx9, _t16_, _fake16_).
  if (auto it = pseudoAlias.find(p); it != pseudoAlias.end())
    p = it->second;

  // DPP -> base. This handles both VOP2-like _dpp pseudos and VOP3-like
  // _e64_dpp pseudos; the reverse map was built from both getDPPOp32 and
  // getDPPOp64.
  if (auto it = dppToBase.find(p); it != dppToBase.end())
    p = it->second;

  // SDWA -> base. LLVM provides a forward helper for this direction.
  int base = AMDGPU::getBasicFromSDWAOp(p);
  if (base > 0)
    p = static_cast<unsigned>(base);

  // e32 -> e64.
  int e64 = AMDGPU::getVOPe64(p);
  if (e64 > 0)
    p = static_cast<unsigned>(e64);

  // Re-apply the pseudo-alias step. `getVOPe64` can resolve an `_e32` pseudo
  // (e.g. `V_LSHLREV_B64_pseudo_e32`) to an `_e64` pseudo with a parallel
  // variant marker (`V_LSHLREV_B64_pseudo_e64`) that only collapses once
  // both the `_pseudo_` infix and `_e64` suffix are visible together.
  if (auto it = pseudoAlias.find(p); it != pseudoAlias.end())
    p = it->second;

  // FLAT/GLOBAL SADDR -> VADDR. Only applicable to instructions tagged with
  // the FLAT format flag; the helper returns -1 for non-FLAT opcodes but
  // checking the flag first avoids the lookup for every non-FLAT opcode.
  if (p < MCII.getNumOpcodes() &&
      (MCII.get(p).TSFlags & SIInstrFlags::FLAT) != 0) {
    int vaddr = AMDGPU::getGlobalVaddrOp(p);
    if (vaddr > 0)
      p = static_cast<unsigned>(vaddr);
  }

  return p;
}

// Parse a canonical vector-compare pseudo name into (predicate, bits, kind).
// Accepted shape: `V_CMP_<PRED>_<TYPE><BITS>_e64` where
//   PRED  ∈ {EQ, NE, GT, GE, LT, LE, LG, NEQ, NLT, NLE, NGT, NGE, NLG, U, O}
//   TYPE  ∈ {U, I, F}
//   BITS  ∈ {16, 32, 64}
// and an optional `V_CMPX_` prefix plays the role of `V_CMP_`. Returns
// `std::nullopt` for anything else; caller is responsible for only passing
// compare-family pseudos.
//
// Rationale: LLVM exposes `AMDGPU::getVCMPXOpFromVCMP` as a V_CMP → V_CMPX
// mapping, but no public helper that hands back a CmpInst::Predicate or
// element width. Rather than hand-list 100 opcode→metadata pairs we parse
// the pseudo name once at init time; the same token grammar is already
// hard-coded in LLVM's TableGen for these instructions.
std::optional<VCmpMeta> parseVCmpPseudoName(llvm::StringRef name) {
  llvm::StringRef rest = name;
  if (!rest.consume_front("V_CMPX_") && !rest.consume_front("V_CMP_"))
    return std::nullopt;
  if (!rest.consume_back("_e64"))
    return std::nullopt;

  auto [predTok, typeTok] = rest.rsplit('_');
  if (predTok.empty() || typeTok.size() < 2)
    return std::nullopt;

  const char typeCh = typeTok[0];
  unsigned bits = 0;
  if (typeTok.drop_front().getAsInteger(10, bits))
    return std::nullopt;
  if (bits != 16 && bits != 32 && bits != 64)
    return std::nullopt;

  using llvm::CmpInst;
  VCmpMeta m{CmpInst::BAD_ICMP_PREDICATE, static_cast<uint8_t>(bits), false};

  if (typeCh == 'F') {
    m.isFloat = true;
    // Float predicates: ordered variants set the O-prefix predicates;
    // N-prefixed AMDGPU names select the "unordered-or-..." complements.
    if (predTok == "EQ")        m.pred = CmpInst::FCMP_OEQ;
    else if (predTok == "GT")   m.pred = CmpInst::FCMP_OGT;
    else if (predTok == "GE")   m.pred = CmpInst::FCMP_OGE;
    else if (predTok == "LT")   m.pred = CmpInst::FCMP_OLT;
    else if (predTok == "LE")   m.pred = CmpInst::FCMP_OLE;
    // LG ("less or greater"), NE, and NEQ all mean "ordered and !=" in
    // AMDGPU's model and all lower to FCMP_ONE.
    else if (predTok == "LG" || predTok == "NE" || predTok == "NEQ")
                                m.pred = CmpInst::FCMP_ONE;
    else if (predTok == "NLT")  m.pred = CmpInst::FCMP_UGE;
    else if (predTok == "NLE")  m.pred = CmpInst::FCMP_UGT;
    else if (predTok == "NGT")  m.pred = CmpInst::FCMP_ULE;
    else if (predTok == "NGE")  m.pred = CmpInst::FCMP_ULT;
    // NLG ("not (less or greater)") is the unordered-or-equal complement.
    else if (predTok == "NLG")  m.pred = CmpInst::FCMP_UEQ;
    else if (predTok == "U")    m.pred = CmpInst::FCMP_UNO;
    else if (predTok == "O")    m.pred = CmpInst::FCMP_ORD;
    else return std::nullopt;
  } else if (typeCh == 'U' || typeCh == 'I') {
    const bool isSigned = typeCh == 'I';
    if (predTok == "EQ")        m.pred = CmpInst::ICMP_EQ;
    else if (predTok == "NE")   m.pred = CmpInst::ICMP_NE;
    else if (predTok == "GT")   m.pred = isSigned ? CmpInst::ICMP_SGT
                                                   : CmpInst::ICMP_UGT;
    else if (predTok == "GE")   m.pred = isSigned ? CmpInst::ICMP_SGE
                                                   : CmpInst::ICMP_UGE;
    else if (predTok == "LT")   m.pred = isSigned ? CmpInst::ICMP_SLT
                                                   : CmpInst::ICMP_ULT;
    else if (predTok == "LE")   m.pred = isSigned ? CmpInst::ICMP_SLE
                                                   : CmpInst::ICMP_ULE;
    else return std::nullopt;
  } else {
    return std::nullopt;
  }

  return m;
}

} // namespace

SemOp OpcodeMap::lookup(unsigned opcode) const {
  auto it = map_.find(opcode);
  return it != map_.end() ? it->second : SemOp::Unknown;
}

const VCmpMeta *OpcodeMap::lookupVCmp(unsigned opcode) const {
  auto it = vcmp_.find(opcode);
  return it != vcmp_.end() ? &it->second : nullptr;
}

void OpcodeMap::build(const MCInstrInfo &MCII) {
  // Flatten the static kCanonTable into a DenseMap for O(1) lookups during the
  // subsequent scan over every MC opcode.
  DenseMap<unsigned, SemOp> canonToSem;
  canonToSem.reserve(std::size(kCanonTable));
  for (const Entry &e : kCanonTable)
    canonToSem.try_emplace(e.opc, e.sem);

  const unsigned numOpc = MCII.getNumOpcodes();
  const auto mcToPseudo  = buildMcToPseudoMap(numOpc);
  const auto pseudoAlias = buildPseudoAliasMap(MCII);
  const auto dppToBase   = buildDppToBaseMap(numOpc);

  map_.clear();
  vcmp_.clear();
  // Heuristic: roughly a quarter of MC opcodes carry a SemOp in practice;
  // resizing a few times is fine for a one-shot init.
  map_.reserve(numOpc / 4);
  for (unsigned mc = 0; mc < numOpc; ++mc) {
    const unsigned canon =
        canonicalize(mc, MCII, mcToPseudo, pseudoAlias, dppToBase);
    if (auto it = canonToSem.find(canon); it != canonToSem.end()) {
      map_[mc] = it->second;
      continue;
    }
    // The canonical pseudo was not enumerated in `kCanonTable`. Check if it
    // belongs to the V_CMP / V_CMPX family (which is handled via metadata
    // side-table rather than per-opcode enumeration). Use the canonical
    // pseudo's name so we don't have to re-canonicalize any DPP/SDWA
    // variants (those have already been folded by `canonicalize`).
    if (canon >= numOpc)
      continue;
    llvm::StringRef canonName = MCII.getName(canon);
    const bool isCmp  = canonName.starts_with("V_CMP_");
    const bool isCmpX = canonName.starts_with("V_CMPX_");
    if (!isCmp && !isCmpX)
      continue;
    if (auto meta = parseVCmpPseudoName(canonName)) {
      map_[mc] = isCmpX ? SemOp::V_CMPX : SemOp::V_CMP;
      vcmp_.try_emplace(mc, *meta);
    }
    // Names that start with V_CMP_ but don't parse (e.g. a hypothetical
    // future family) are left as SemOp::Unknown so the raiser reports them
    // loudly rather than silently producing wrong IR.
  }
}

} // namespace transpiler
