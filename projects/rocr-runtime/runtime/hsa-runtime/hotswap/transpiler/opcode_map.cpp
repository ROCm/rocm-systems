#include "opcode_map.hpp"

#include <cstdint>
#include <string>

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"

using namespace llvm;

// LLVM's AMDGPU target generates a family of instruction-mapping functions via
// TableGen (FilterClass/Maps in SIInstrInfo.td). They are emitted into
// AMDGPUGenInstrInfo.inc under `#define GET_INSTRMAP_INFO`, instantiated in
// llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp, and linked from
// libLLVMAMDGPUUtils.a. Their declarations live in SIInstrInfo.h, which pulls
// in a lot of CodeGen machinery we don't need; forward-declare them here.
namespace llvm::AMDGPU {
// MC (subtarget-specific real) <-> pseudo opcode.
int32_t getMCOpcode(uint32_t Opcode, unsigned Gen);

// Encoding variants: e32 <-> e64, base <-> DPP/SDWA.
int32_t getVOPe64(uint32_t Opcode);
int32_t getVOPe32(uint32_t Opcode);
int32_t getDPPOp32(uint32_t Opcode);
int32_t getDPPOp64(uint32_t Opcode);
int32_t getSDWAOp(uint32_t Opcode);
int32_t getBasicFromSDWAOp(uint32_t Opcode);

// FLAT/GLOBAL SADDR <-> VADDR (despite the name, also covers FLAT).
int32_t getGlobalVaddrOp(uint32_t Opcode);
} // namespace llvm::AMDGPU

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

    // ---------------------------------------------------------------------
    // SMEM scalar loads
    // ---------------------------------------------------------------------
    SMEM3(S_LOAD_DWORD,    S_LOAD_B32),
    SMEM3(S_LOAD_DWORDX2,  S_LOAD_B64),
    SMEM3(S_LOAD_DWORDX3,  S_LOAD_B96),
    SMEM3(S_LOAD_DWORDX4,  S_LOAD_B128),
    SMEM3(S_LOAD_DWORDX8,  S_LOAD_B256),

    // ---------------------------------------------------------------------
    // SOPC
    // ---------------------------------------------------------------------
    E(S_CMP_EQ_U32, S_CMP_EQ_U32), E(S_CMP_LG_U32, S_CMP_LG_U32),
    E(S_CMP_GT_U32, S_CMP_GT_U32), E(S_CMP_GE_U32, S_CMP_GE_U32),
    E(S_CMP_LT_U32, S_CMP_LT_U32), E(S_CMP_LE_U32, S_CMP_LE_U32),
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
    E(S_FF1_I32_B32, S_FF1_I32_B32), E(S_FF1_I32_B64, S_FF1_I32_B64),
    E(S_FLBIT_I32_B32, S_FLBIT_I32_B32), E(S_FLBIT_I32_B64, S_FLBIT_I32_B64),
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
    E(S_ABS_I32, S_ABS_I32),
    E(S_SET_VGPR_MSB, S_SET_VGPR_MSB),

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
    E(S_LSHL_B32, S_LSHL_B32), E(S_LSHL_B64, S_LSHL_B64),
    E(S_LSHR_B32, S_LSHR_B32), E(S_ASHR_I32, S_ASHR_I32),
    E(S_MUL_I32, S_MUL_I32), E(S_MUL_HI_U32, S_MUL_HI_U32),
    E(S_MUL_U64, S_MUL_U64), E(S_MUL_F32, S_MUL_F32),
    E(S_ADD_F32, S_ADD_F32),
    E(S_BFE_U32, S_BFE_U32), E(S_BFM_B32, S_BFM_B32), E(S_BFM_B64, S_BFM_B64),
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

    // ---------------------------------------------------------------------
    // VOP1 (canonical form is `_e64` after getVOPe64 collapses e32)
    // ---------------------------------------------------------------------
    E(V_MOV_B32_e64, V_MOV_B32),
    E(V_NOP_e64, V_NOP),
    E(V_NOT_B32_e64, V_NOT_B32),
    E(V_BFREV_B32_e64, V_BFREV_B32),
    E(V_CVT_F32_I32_e64, V_CVT_F32_I32),
    E(V_CVT_F32_U32_e64, V_CVT_F32_U32),
    E(V_CVT_I32_F32_e64, V_CVT_I32_F32),
    E(V_CVT_U32_F32_e64, V_CVT_U32_F32),
    E(V_CVT_F16_F32_e64, V_CVT_F16_F32),
    E(V_CVT_F32_F16_e64, V_CVT_F32_F16),
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
    // gfx11+ IEEE-754 max/min (called `V_MAXIMUM_F32`/`V_MINIMUM_F32` in LLVM,
    // surfaced here as the `_NUM_` SemOp the handlers already recognize).
    E(V_MAXIMUM_F32_e64, V_MAX_NUM_F32),
    E(V_MINIMUM_F32_e64, V_MIN_NUM_F32),
    E(V_DIV_FIXUP_F32_e64, V_DIV_FIXUP_F32),
    E(V_DIV_FMAS_F32_e64, V_DIV_FMAS_F32),
    E(V_DIV_SCALE_F32_e64, V_DIV_SCALE_F32),
    // LLVM's no-carry 32-bit add/sub pseudos are just `V_ADD_U32` etc.; the
    // carry-in-carry-out form is the older `V_ADDC_U32`/`V_SUBB_U32` family.
    E(V_ADD_U32_e64, V_ADD_NC_U32),
    E(V_SUB_U32_e64, V_SUB_NC_U32),
    E(V_SUBREV_U32_e64, V_SUBREV_NC_U32),
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
    E(V_LSHL_OR_B32_e64, V_LSHL_OR_B32),
    E(V_AND_OR_B32_e64, V_AND_OR_B32),
    E(V_OR3_B32_e64, V_OR3_B32),
    E(V_BFE_U32_e64, V_BFE_U32),
    E(V_PERM_B32_e64, V_PERM_B32),
    E(V_MBCNT_LO_U32_B32_e64, V_MBCNT_LO_U32_B32),
    E(V_MBCNT_HI_U32_B32_e64, V_MBCNT_HI_U32_B32),
    E(V_READLANE_B32, V_READLANE_B32),
    E(V_WRITELANE_B32, V_WRITELANE_B32),
    E(V_MED3_F32_e64, V_MED3_F32),
    E(V_MAX3_F32_e64, V_MAX3_F32),
    E(V_MIN3_F32_e64, V_MIN3_F32),
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
    E(V_ASHRREV_I16_e64, V_ASHRREV_I16),
    E(V_LSHRREV_B16_e64, V_LSHRREV_B16),
    E(V_LSHLREV_B16_e64, V_LSHLREV_B16),
    E(V_PACK_B32_F16_e64, V_PACK_B32_F16),
    E(V_CVT_PK_BF16_F32_e64, V_CVT_PK_BF16_F32),
    E(V_CVT_PK_BF8_F32_e64, V_CVT_PK_BF8_F32),
    E(V_CVT_PK_FP8_F32_e64, V_CVT_PK_FP8_F32),

    // ---------------------------------------------------------------------
    // FP64
    // ---------------------------------------------------------------------
    E(V_ADD_F64_e64, V_ADD_F64),
    E(V_MUL_F64_e64, V_MUL_F64),
    E(V_FMA_F64_e64, V_FMA_F64),
    E(V_FMAC_F64_e64, V_FMAC_F64),

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

    // ---------------------------------------------------------------------
    // VOPC integer (canonical is `_e64`)
    // ---------------------------------------------------------------------
    E(V_CMP_EQ_U32_e64, V_CMP_EQ_U32), E(V_CMP_NE_U32_e64, V_CMP_NE_U32),
    E(V_CMP_GT_U32_e64, V_CMP_GT_U32), E(V_CMP_GE_U32_e64, V_CMP_GE_U32),
    E(V_CMP_LT_U32_e64, V_CMP_LT_U32), E(V_CMP_LE_U32_e64, V_CMP_LE_U32),
    E(V_CMP_EQ_I32_e64, V_CMP_EQ_I32), E(V_CMP_NE_I32_e64, V_CMP_NE_I32),
    E(V_CMP_GT_I32_e64, V_CMP_GT_I32), E(V_CMP_GE_I32_e64, V_CMP_GE_I32),
    E(V_CMP_LT_I32_e64, V_CMP_LT_I32), E(V_CMP_LE_I32_e64, V_CMP_LE_I32),
    E(V_CMP_EQ_U64_e64, V_CMP_EQ_U64), E(V_CMP_NE_U64_e64, V_CMP_NE_U64),
    E(V_CMP_GT_U64_e64, V_CMP_GT_U64), E(V_CMP_GE_U64_e64, V_CMP_GE_U64),
    E(V_CMP_LT_U64_e64, V_CMP_LT_U64), E(V_CMP_LE_U64_e64, V_CMP_LE_U64),
    E(V_CMP_GT_I64_e64, V_CMP_GT_I64), E(V_CMP_GE_I64_e64, V_CMP_GE_I64),
    E(V_CMP_LT_I64_e64, V_CMP_LT_I64), E(V_CMP_LE_I64_e64, V_CMP_LE_I64),

    // ---------------------------------------------------------------------
    // VOPC float
    // ---------------------------------------------------------------------
    // LLVM names the ordered-not-equal float predicate `NEQ`, not `NE` (which
    // is integer-only). The legacy `V_CMP_NE_F*` SemOps therefore remain
    // unmapped; handlers already consume the `NEQ` SemOps.
    E(V_CMP_EQ_F32_e64, V_CMP_EQ_F32),
    E(V_CMP_GT_F32_e64, V_CMP_GT_F32), E(V_CMP_GE_F32_e64, V_CMP_GE_F32),
    E(V_CMP_LT_F32_e64, V_CMP_LT_F32), E(V_CMP_LE_F32_e64, V_CMP_LE_F32),
    E(V_CMP_LG_F32_e64, V_CMP_LG_F32),
    E(V_CMP_NEQ_F32_e64, V_CMP_NEQ_F32),
    E(V_CMP_NLT_F32_e64, V_CMP_NLT_F32), E(V_CMP_NLE_F32_e64, V_CMP_NLE_F32),
    E(V_CMP_NGT_F32_e64, V_CMP_NGT_F32), E(V_CMP_NGE_F32_e64, V_CMP_NGE_F32),
    E(V_CMP_NLG_F32_e64, V_CMP_NLG_F32),
    E(V_CMP_U_F32_e64, V_CMP_U_F32), E(V_CMP_O_F32_e64, V_CMP_O_F32),
    E(V_CMP_EQ_F16_e64, V_CMP_EQ_F16),
    E(V_CMP_GT_F16_e64, V_CMP_GT_F16), E(V_CMP_GE_F16_e64, V_CMP_GE_F16),
    E(V_CMP_LT_F16_e64, V_CMP_LT_F16), E(V_CMP_LE_F16_e64, V_CMP_LE_F16),
    E(V_CMP_LG_F16_e64, V_CMP_LG_F16),
    E(V_CMP_NEQ_F16_e64, V_CMP_NEQ_F16),
    E(V_CMP_NLT_F16_e64, V_CMP_NLT_F16), E(V_CMP_NLE_F16_e64, V_CMP_NLE_F16),
    E(V_CMP_NGT_F16_e64, V_CMP_NGT_F16), E(V_CMP_NGE_F16_e64, V_CMP_NGE_F16),
    E(V_CMP_U_F16_e64, V_CMP_U_F16), E(V_CMP_O_F16_e64, V_CMP_O_F16),
    E(V_CMP_EQ_F64_e64, V_CMP_EQ_F64),
    E(V_CMP_GT_F64_e64, V_CMP_GT_F64), E(V_CMP_GE_F64_e64, V_CMP_GE_F64),
    E(V_CMP_LT_F64_e64, V_CMP_LT_F64), E(V_CMP_LE_F64_e64, V_CMP_LE_F64),
    E(V_CMP_LG_F64_e64, V_CMP_LG_F64),
    E(V_CMP_NEQ_F64_e64, V_CMP_NEQ_F64),
    E(V_CMP_NLT_F64_e64, V_CMP_NLT_F64), E(V_CMP_NLE_F64_e64, V_CMP_NLE_F64),
    E(V_CMP_NGT_F64_e64, V_CMP_NGT_F64), E(V_CMP_NGE_F64_e64, V_CMP_NGE_F64),
    E(V_CMP_U_F64_e64, V_CMP_U_F64), E(V_CMP_O_F64_e64, V_CMP_O_F64),

    // ---------------------------------------------------------------------
    // CMPX integer
    // ---------------------------------------------------------------------
    E(V_CMPX_EQ_U32_e64, V_CMPX_EQ_U32), E(V_CMPX_NE_U32_e64, V_CMPX_NE_U32),
    E(V_CMPX_GT_U32_e64, V_CMPX_GT_U32), E(V_CMPX_GE_U32_e64, V_CMPX_GE_U32),
    E(V_CMPX_LT_U32_e64, V_CMPX_LT_U32), E(V_CMPX_LE_U32_e64, V_CMPX_LE_U32),
    E(V_CMPX_EQ_I32_e64, V_CMPX_EQ_I32), E(V_CMPX_NE_I32_e64, V_CMPX_NE_I32),
    E(V_CMPX_GT_I32_e64, V_CMPX_GT_I32), E(V_CMPX_GE_I32_e64, V_CMPX_GE_I32),
    E(V_CMPX_LT_I32_e64, V_CMPX_LT_I32), E(V_CMPX_LE_I32_e64, V_CMPX_LE_I32),

    // ---------------------------------------------------------------------
    // CMPX float
    // ---------------------------------------------------------------------
    E(V_CMPX_EQ_F32_e64, V_CMPX_EQ_F32),
    E(V_CMPX_GT_F32_e64, V_CMPX_GT_F32), E(V_CMPX_GE_F32_e64, V_CMPX_GE_F32),
    E(V_CMPX_LT_F32_e64, V_CMPX_LT_F32), E(V_CMPX_LE_F32_e64, V_CMPX_LE_F32),
    E(V_CMPX_LG_F32_e64, V_CMPX_LG_F32),
    E(V_CMPX_NEQ_F32_e64, V_CMPX_NEQ_F32),
    E(V_CMPX_NLT_F32_e64, V_CMPX_NLT_F32), E(V_CMPX_NLE_F32_e64, V_CMPX_NLE_F32),
    E(V_CMPX_NGT_F32_e64, V_CMPX_NGT_F32), E(V_CMPX_NGE_F32_e64, V_CMPX_NGE_F32),
    E(V_CMPX_EQ_F16_e64, V_CMPX_EQ_F16),
    E(V_CMPX_GT_F16_e64, V_CMPX_GT_F16), E(V_CMPX_GE_F16_e64, V_CMPX_GE_F16),
    E(V_CMPX_LT_F16_e64, V_CMPX_LT_F16), E(V_CMPX_LE_F16_e64, V_CMPX_LE_F16),
    E(V_CMPX_LG_F16_e64, V_CMPX_LG_F16),
    E(V_CMPX_NEQ_F16_e64, V_CMPX_NEQ_F16),

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
    E(V_LSHL_ADD_U64_e64, V_LSHL_ADD_U64),
    // LLVM's 64-bit no-carry add pseudo is simply `V_ADD_U64_e64`.
    E(V_ADD_U64_e64, V_ADD_NC_U64),
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
    E(DS_READ_B32, DS_READ_B32), E(DS_READ_B64, DS_READ_B64),
    E(DS_READ_B128, DS_READ_B128),
    E(DS_READ2_B32, DS_READ2_B32), E(DS_READ2_B64, DS_READ2_B64),
    E(DS_READ_U16, DS_READ_U16), E(DS_READ_I16, DS_READ_I16),
    E(DS_READ_U8, DS_READ_U8), E(DS_READ_I8, DS_READ_I8),
    E(DS_WRITE_B32, DS_WRITE_B32), E(DS_WRITE_B64, DS_WRITE_B64),
    E(DS_WRITE_B128, DS_WRITE_B128),
    E(DS_WRITE2_B32, DS_WRITE2_B32), E(DS_WRITE2_B64, DS_WRITE2_B64),
    E(DS_WRITE_B16, DS_WRITE_B16), E(DS_WRITE_B8, DS_WRITE_B8),
    E(DS_BPERMUTE_B32, DS_BPERMUTE_B32),

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
    MUBUF4(BUFFER_LOAD_SHORT_D16, BUFFER_LOAD_SHORT_D16),
    MUBUF4(BUFFER_LOAD_SHORT_D16_HI, BUFFER_LOAD_SHORT_D16_HI),

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
    MUBUF4(BUFFER_ATOMIC_ADD_F32, BUFFER_ATOMIC_ADD_F32),
    MUBUF4(BUFFER_ATOMIC_PK_ADD_BF16, BUFFER_ATOMIC_PK_ADD_BF16),
    MUBUF4(BUFFER_ATOMIC_PK_ADD_F16, BUFFER_ATOMIC_PK_ADD_F16),

    // ---------------------------------------------------------------------
    // AGPR moves
    // ---------------------------------------------------------------------
    E(V_ACCVGPR_READ_B32_e64, V_ACCVGPR_READ_B32),
    E(V_ACCVGPR_WRITE_B32_e64, V_ACCVGPR_WRITE_B32),

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
};

#undef SMEM3
#undef VBUF4
#undef MUBUF4
#undef E

// SIEncodingFamily from [SIDefines.h]: SI(0) .. GFX13(14), inclusive.
// getMCOpcode(pseudo, gen) returns the MC opcode used on that subtarget
// generation, or -1 if no real encoding exists.
constexpr unsigned kNumEncodingFamilies = 15;

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
  };
  // Subtarget-specific markers ("_gfx9", "_gfx1250", ...) and operand-size
  // markers ("_t16_", "_fake16_") that LLVM injects into the pseudo name.
  // A single pseudo can carry multiple markers (e.g.
  // `V_BITOP3_B16_gfx1250_fake16_e64`), so the outer loop below applies these
  // rules iteratively until the name stops shrinking.
  static const Rule rules[] = {
      {"_vi_gfx9", true},
      {"_gfx9", true},
      {"_gfx1250", true},
      {"_gfx1250_", false},
      {"_pseudo_", false},
      {"_t16_", false},
      {"_fake16_", false},
  };

  auto stripOnce = [&](llvm::StringRef name, std::string &out) -> bool {
    for (const Rule &r : rules) {
      if (r.isSuffix) {
        if (!name.ends_with(r.needle))
          continue;
        out = name.drop_back(r.needle.size()).str();
        return true;
      }
      size_t pos = name.find(r.needle);
      if (pos == llvm::StringRef::npos)
        continue;
      out = (name.substr(0, pos).str() + std::string("_") +
             name.substr(pos + r.needle.size()).str());
      return true;
    }
    return false;
  };

  DenseMap<unsigned, unsigned> alias;
  for (const auto &kv : byName) {
    std::string cur = kv.first().str();
    unsigned finalOpc = kv.second;
    while (true) {
      std::string next;
      if (!stripOnce(cur, next))
        break;
      auto it = byName.find(next);
      if (it != byName.end() && it->second != kv.second)
        finalOpc = it->second;
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

} // namespace

SemOp OpcodeMap::lookup(unsigned opcode) const {
  auto it = map.find(opcode);
  return it != map.end() ? it->second : SemOp::Unknown;
}

void OpcodeMap::build(const MCInstrInfo &MCII) {
  // Flatten the static table into a DenseMap for O(1) lookups during the
  // subsequent scan over every MC opcode.
  DenseMap<unsigned, SemOp> canonToSem;
  canonToSem.reserve(sizeof(kCanonTable) / sizeof(kCanonTable[0]));
  for (const Entry &e : kCanonTable)
    canonToSem.try_emplace(e.opc, e.sem);

  unsigned numOpc = MCII.getNumOpcodes();
  auto mcToPseudo = buildMcToPseudoMap(numOpc);
  auto pseudoAlias = buildPseudoAliasMap(MCII);
  auto dppToBase = buildDppToBaseMap(numOpc);

  map.clear();
  map.reserve(numOpc / 4);
  for (unsigned i = 0; i < numOpc; ++i) {
    unsigned canon = canonicalize(i, MCII, mcToPseudo, pseudoAlias, dppToBase);
    auto it = canonToSem.find(canon);
    if (it != canonToSem.end())
      map[i] = it->second;
  }
}

} // namespace transpiler
