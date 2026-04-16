#include "opcode_map.hpp"

#include <utility>

#include "llvm/ADT/StringMap.h"
#include "llvm/MC/MCInstrInfo.h"

using namespace llvm;

namespace transpiler {

SemOp OpcodeMap::lookup(unsigned opcode) const {
  auto it = map.find(opcode);
  return it != map.end() ? it->second : SemOp::Unknown;
}

void OpcodeMap::build(const MCInstrInfo &MCII) {
    // Static table: MCInstrInfo base name → SemOp
    static const std::pair<StringRef, SemOp> table[] = {
      // SOPP
      {"S_ENDPGM", SemOp::S_ENDPGM}, {"S_NOP", SemOp::S_NOP},
      {"S_BRANCH", SemOp::S_BRANCH}, {"S_CODE_END", SemOp::S_CODE_END},
      {"S_CBRANCH_SCC0", SemOp::S_CBRANCH_SCC0}, {"S_CBRANCH_SCC1", SemOp::S_CBRANCH_SCC1},
      {"S_CBRANCH_VCCZ", SemOp::S_CBRANCH_VCCZ}, {"S_CBRANCH_VCCNZ", SemOp::S_CBRANCH_VCCNZ},
      {"S_CBRANCH_EXECZ", SemOp::S_CBRANCH_EXECZ}, {"S_CBRANCH_EXECNZ", SemOp::S_CBRANCH_EXECNZ},
      {"S_WAITCNT", SemOp::S_WAITCNT},
      {"S_WAIT_LOADCNT", SemOp::S_WAIT_LOADCNT}, {"S_WAIT_KMCNT", SemOp::S_WAIT_KMCNT},
      {"S_WAIT_DSCNT", SemOp::S_WAIT_DSCNT}, {"S_WAIT_XCNT", SemOp::S_WAIT_XCNT},
      {"S_WAIT_LOADCNT_DSCNT", SemOp::S_WAIT_LOADCNT_DSCNT},
      {"S_WAITCNT_DEPCTR", SemOp::S_WAIT_ALU},
      {"S_CLAUSE", SemOp::S_CLAUSE}, {"S_DELAY_ALU", SemOp::S_DELAY_ALU},
      {"S_SET_GPR_IDX_ON", SemOp::S_SET_GPR_IDX_ON}, {"S_SET_GPR_IDX_OFF", SemOp::S_SET_GPR_IDX_OFF},
      {"S_SETVSKIP", SemOp::S_SETVSKIP},

      // SMEM (with and without _IMM suffix since stripMCIISuffix removes it)
      {"S_LOAD_B32_IMM", SemOp::S_LOAD_B32}, {"S_LOAD_B32", SemOp::S_LOAD_B32},
      {"S_LOAD_DWORD_IMM", SemOp::S_LOAD_B32}, {"S_LOAD_DWORD", SemOp::S_LOAD_B32},
      {"S_LOAD_B64_IMM", SemOp::S_LOAD_B64}, {"S_LOAD_B64", SemOp::S_LOAD_B64},
      {"S_LOAD_DWORDX2_IMM", SemOp::S_LOAD_B64}, {"S_LOAD_DWORDX2", SemOp::S_LOAD_B64},
      {"S_LOAD_B96_IMM", SemOp::S_LOAD_B96}, {"S_LOAD_B96", SemOp::S_LOAD_B96},
      {"S_LOAD_DWORDX3_IMM", SemOp::S_LOAD_B96}, {"S_LOAD_DWORDX3", SemOp::S_LOAD_B96},
      {"S_LOAD_B128_IMM", SemOp::S_LOAD_B128}, {"S_LOAD_B128", SemOp::S_LOAD_B128},
      {"S_LOAD_DWORDX4_IMM", SemOp::S_LOAD_B128}, {"S_LOAD_DWORDX4", SemOp::S_LOAD_B128},
      {"S_LOAD_B256_IMM", SemOp::S_LOAD_B256}, {"S_LOAD_B256", SemOp::S_LOAD_B256},
      {"S_LOAD_DWORDX8_IMM", SemOp::S_LOAD_B256}, {"S_LOAD_DWORDX8", SemOp::S_LOAD_B256},

      // SOPC
      {"S_CMP_EQ_U32", SemOp::S_CMP_EQ_U32}, {"S_CMP_LG_U32", SemOp::S_CMP_LG_U32},
      {"S_CMP_GT_U32", SemOp::S_CMP_GT_U32}, {"S_CMP_GE_U32", SemOp::S_CMP_GE_U32},
      {"S_CMP_LT_U32", SemOp::S_CMP_LT_U32}, {"S_CMP_LE_U32", SemOp::S_CMP_LE_U32},
      {"S_CMP_EQ_I32", SemOp::S_CMP_EQ_I32}, {"S_CMP_LG_I32", SemOp::S_CMP_LG_I32},
      {"S_CMP_GT_I32", SemOp::S_CMP_GT_I32}, {"S_CMP_GE_I32", SemOp::S_CMP_GE_I32},
      {"S_CMP_LT_I32", SemOp::S_CMP_LT_I32}, {"S_CMP_LE_I32", SemOp::S_CMP_LE_I32},
      {"S_CMP_EQ_F32", SemOp::S_CMP_EQ_F32}, {"S_CMP_LG_F32", SemOp::S_CMP_LG_F32},
      {"S_CMP_GT_F32", SemOp::S_CMP_GT_F32}, {"S_CMP_GE_F32", SemOp::S_CMP_GE_F32},
      {"S_CMP_LT_F32", SemOp::S_CMP_LT_F32}, {"S_CMP_LE_F32", SemOp::S_CMP_LE_F32},
      {"S_CMP_NEQ_F32", SemOp::S_CMP_NEQ_F32},
      {"S_CMP_NGT_F32", SemOp::S_CMP_NGT_F32}, {"S_CMP_NGE_F32", SemOp::S_CMP_NGE_F32},
      {"S_CMP_NLT_F32", SemOp::S_CMP_NLT_F32}, {"S_CMP_NLE_F32", SemOp::S_CMP_NLE_F32},
      {"S_CMP_NLG_F32", SemOp::S_CMP_NLG_F32},
      {"S_CMP_EQ_F16", SemOp::S_CMP_EQ_F16}, {"S_CMP_LG_F16", SemOp::S_CMP_LG_F16},
      {"S_CMP_GT_F16", SemOp::S_CMP_GT_F16}, {"S_CMP_GE_F16", SemOp::S_CMP_GE_F16},
      {"S_CMP_LT_F16", SemOp::S_CMP_LT_F16}, {"S_CMP_LE_F16", SemOp::S_CMP_LE_F16},
      {"S_CMP_NEQ_F16", SemOp::S_CMP_NEQ_F16},
      {"S_CMP_NGT_F16", SemOp::S_CMP_NGT_F16}, {"S_CMP_NGE_F16", SemOp::S_CMP_NGE_F16},
      {"S_CMP_NLT_F16", SemOp::S_CMP_NLT_F16}, {"S_CMP_NLE_F16", SemOp::S_CMP_NLE_F16},
      {"S_CMP_NLG_F16", SemOp::S_CMP_NLG_F16},

      // SOPK
      {"S_MOVK_I32", SemOp::S_MOVK_I32}, {"S_ADDK_I32", SemOp::S_ADDK_I32},
      {"S_ADDK_CO_I32", SemOp::S_ADDK_I32},
      {"S_MULK_I32", SemOp::S_MULK_I32},
      {"S_CMPK_GE_I32", SemOp::S_CMPK_GE_I32}, {"S_CMPK_GT_I32", SemOp::S_CMPK_GT_I32},
      {"S_CMPK_LE_I32", SemOp::S_CMPK_LE_I32}, {"S_CMPK_LT_I32", SemOp::S_CMPK_LT_I32},
      {"S_CMPK_GE_U32", SemOp::S_CMPK_GE_U32}, {"S_CMPK_GT_U32", SemOp::S_CMPK_GT_U32},
      {"S_CMPK_LE_U32", SemOp::S_CMPK_LE_U32}, {"S_CMPK_LT_U32", SemOp::S_CMPK_LT_U32},
      {"S_CMPK_EQ_I32", SemOp::S_CMPK_EQ_I32}, {"S_CMPK_EQ_U32", SemOp::S_CMPK_EQ_U32},
      {"S_CMPK_LG_I32", SemOp::S_CMPK_LG_I32}, {"S_CMPK_LG_U32", SemOp::S_CMPK_LG_U32},
      {"S_GETREG_B32", SemOp::S_GETREG_B32},
      {"S_SETREG_B32", SemOp::S_SETREG_B32}, {"S_SETREG_IMM32_B32", SemOp::S_SETREG_IMM32_B32},

      // SOP1
      {"S_MOV_B32", SemOp::S_MOV_B32}, {"S_MOV_B64", SemOp::S_MOV_B64},
      {"S_NOT_B32", SemOp::S_NOT_B32}, {"S_NOT_B64", SemOp::S_NOT_B64},
      {"S_BREV_B32", SemOp::S_BREV_B32},
      {"S_FF1_I32_B32", SemOp::S_FF1_I32_B32}, {"S_FF1_I32_B64", SemOp::S_FF1_I32_B64},
      {"S_FLBIT_I32_B32", SemOp::S_FLBIT_I32_B32}, {"S_FLBIT_I32_B64", SemOp::S_FLBIT_I32_B64},
      {"S_SEXT_I32_I8", SemOp::S_SEXT_I32_I8}, {"S_SEXT_I32_I16", SemOp::S_SEXT_I32_I16},
      {"S_CVT_F32_U32", SemOp::S_CVT_F32_U32}, {"S_CVT_F32_I32", SemOp::S_CVT_F32_I32},
      {"S_CVT_U32_F32", SemOp::S_CVT_U32_F32}, {"S_CVT_I32_F32", SemOp::S_CVT_I32_F32},
      {"S_AND_SAVEEXEC_B32", SemOp::S_AND_SAVEEXEC_B32},
      {"S_OR_SAVEEXEC_B32", SemOp::S_OR_SAVEEXEC_B32},
      {"S_XOR_SAVEEXEC_B32", SemOp::S_XOR_SAVEEXEC_B32},
      {"S_ANDN2_SAVEEXEC_B32", SemOp::S_ANDN2_SAVEEXEC_B32},
      {"S_ORN2_SAVEEXEC_B32", SemOp::S_ORN2_SAVEEXEC_B32},
      {"S_GETPC_B64", SemOp::S_GETPC_B64},
      {"S_ABS_I32", SemOp::S_ABS_I32},
      {"S_SET_VGPR_MSB", SemOp::S_SET_VGPR_MSB},

      // SOP2
      {"S_ADD_U32", SemOp::S_ADD_U32}, {"S_ADD_I32", SemOp::S_ADD_U32},
      {"S_ADDC_U32", SemOp::S_ADDC_U32},
      {"S_SUB_U32", SemOp::S_SUB_U32}, {"S_SUB_I32", SemOp::S_SUB_U32},
      {"S_SUBB_U32", SemOp::S_SUBB_U32},
      {"S_ADD_U64", SemOp::S_ADD_U64},
      {"S_AND_B32", SemOp::S_AND_B32}, {"S_AND_B64", SemOp::S_AND_B64},
      {"S_OR_B32", SemOp::S_OR_B32}, {"S_OR_B64", SemOp::S_OR_B64},
      {"S_XOR_B32", SemOp::S_XOR_B32}, {"S_XOR_B64", SemOp::S_XOR_B64},
      {"S_ANDN2_B32", SemOp::S_ANDN2_B32}, {"S_ANDN2_B64", SemOp::S_ANDN2_B64},
      {"S_ORN2_B32", SemOp::S_ORN2_B32}, {"S_ORN2_B64", SemOp::S_ORN2_B64},
      {"S_LSHL_B32", SemOp::S_LSHL_B32}, {"S_LSHL_B64", SemOp::S_LSHL_B64},
      {"S_LSHR_B32", SemOp::S_LSHR_B32}, {"S_ASHR_I32", SemOp::S_ASHR_I32},
      {"S_MUL_I32", SemOp::S_MUL_I32}, {"S_MUL_HI_U32", SemOp::S_MUL_HI_U32},
      {"S_MUL_U64", SemOp::S_MUL_U64}, {"S_MUL_F32", SemOp::S_MUL_F32},
      {"S_ADD_F32", SemOp::S_ADD_F32},
      {"S_BFE_U32", SemOp::S_BFE_U32}, {"S_BFM_B32", SemOp::S_BFM_B32}, {"S_BFM_B64", SemOp::S_BFM_B64},
      {"S_CSELECT_B32", SemOp::S_CSELECT_B32}, {"S_CSELECT_B64", SemOp::S_CSELECT_B64},
      {"S_MIN_I32", SemOp::S_MIN_I32}, {"S_MIN_U32", SemOp::S_MIN_U32},
      {"S_MAX_I32", SemOp::S_MAX_I32}, {"S_MAX_U32", SemOp::S_MAX_U32},
      {"S_PACK_LL_B32_B16", SemOp::S_PACK_LL_B32_B16},
      {"S_PACK_LH_B32_B16", SemOp::S_PACK_LH_B32_B16},
      {"S_LSHL1_ADD_U32", SemOp::S_LSHL1_ADD_U32}, {"S_LSHL2_ADD_U32", SemOp::S_LSHL2_ADD_U32},
      {"S_LSHL3_ADD_U32", SemOp::S_LSHL3_ADD_U32}, {"S_LSHL4_ADD_U32", SemOp::S_LSHL4_ADD_U32},
      {"S_ADD_NC_U64", SemOp::S_ADD_NC_U64},

      // VOP1
      {"V_MOV_B32", SemOp::V_MOV_B32}, {"V_NOP", SemOp::V_NOP},
      {"V_NOT_B32", SemOp::V_NOT_B32}, {"V_BFREV_B32", SemOp::V_BFREV_B32},
      {"V_CVT_F32_I32", SemOp::V_CVT_F32_I32}, {"V_CVT_F32_U32", SemOp::V_CVT_F32_U32},
      {"V_CVT_I32_F32", SemOp::V_CVT_I32_F32}, {"V_CVT_U32_F32", SemOp::V_CVT_U32_F32},
      {"V_CVT_F16_F32", SemOp::V_CVT_F16_F32}, {"V_CVT_F32_F16", SemOp::V_CVT_F32_F16},
      {"V_CVT_F32_UBYTE0", SemOp::V_CVT_F32_UBYTE0}, {"V_CVT_F32_UBYTE1", SemOp::V_CVT_F32_UBYTE1},
      {"V_CVT_F32_UBYTE2", SemOp::V_CVT_F32_UBYTE2}, {"V_CVT_F32_UBYTE3", SemOp::V_CVT_F32_UBYTE3},
      {"V_CVT_F64_U32", SemOp::V_CVT_F64_U32}, {"V_CVT_F64_I32", SemOp::V_CVT_F64_I32},
      {"V_CVT_U32_F64", SemOp::V_CVT_U32_F64},
      {"V_RCP_IFLAG_F32", SemOp::V_RCP_IFLAG_F32}, {"V_RCP_F32", SemOp::V_RCP_F32},
      {"V_RSQ_F32", SemOp::V_RSQ_F32},
      {"V_SQRT_F32", SemOp::V_SQRT_F32}, {"V_EXP_F32", SemOp::V_EXP_F32},
      {"V_LOG_F32", SemOp::V_LOG_F32},
      {"V_LDEXP_F32", SemOp::V_LDEXP_F32},
      {"V_FLOOR_F32", SemOp::V_FLOOR_F32}, {"V_CEIL_F32", SemOp::V_CEIL_F32},
      {"V_TRUNC_F32", SemOp::V_TRUNC_F32}, {"V_FRACT_F32", SemOp::V_FRACT_F32},
      {"V_READFIRSTLANE_B32", SemOp::V_READFIRSTLANE_B32},

      // VOP2 / VOP3
      {"V_ADD_F32", SemOp::V_ADD_F32}, {"V_SUB_F32", SemOp::V_SUB_F32},
      {"V_SUBREV_F32", SemOp::V_SUBREV_F32}, {"V_MUL_F32", SemOp::V_MUL_F32},
      {"V_FMAC_F32", SemOp::V_FMAC_F32}, {"V_FMA_F32", SemOp::V_FMA_F32},
      {"V_MAX_F32", SemOp::V_MAX_F32}, {"V_MIN_F32", SemOp::V_MIN_F32},
      {"V_MAX_NUM_F32", SemOp::V_MAX_NUM_F32}, {"V_MIN_NUM_F32", SemOp::V_MIN_NUM_F32},
      {"V_DIV_FIXUP_F32", SemOp::V_DIV_FIXUP_F32},
      {"V_DIV_FMAS_F32", SemOp::V_DIV_FMAS_F32},
      {"V_DIV_SCALE_F32", SemOp::V_DIV_SCALE_F32},
      {"V_ADD_NC_U32", SemOp::V_ADD_NC_U32}, {"V_SUB_NC_U32", SemOp::V_SUB_NC_U32},
      {"V_SUBREV_NC_U32", SemOp::V_SUBREV_NC_U32},
      {"V_ADD_CO_U32", SemOp::V_ADD_CO_U32}, {"V_ADD_CO_CI_U32", SemOp::V_ADD_CO_CI_U32},
      {"V_ADDC_CO_U32", SemOp::V_ADD_CO_CI_U32}, {"V_ADDC_U32", SemOp::V_ADD_CO_CI_U32},
      {"V_AND_B32", SemOp::V_AND_B32}, {"V_OR_B32", SemOp::V_OR_B32}, {"V_XOR_B32", SemOp::V_XOR_B32},
      {"V_LSHLREV_B32", SemOp::V_LSHLREV_B32}, {"V_LSHRREV_B32", SemOp::V_LSHRREV_B32},
      {"V_ASHRREV_I32", SemOp::V_ASHRREV_I32},
      {"V_CNDMASK_B32", SemOp::V_CNDMASK_B32},
      {"V_MUL_LO_U32", SemOp::V_MUL_LO_U32}, {"V_MUL_HI_U32", SemOp::V_MUL_HI_U32},
      {"V_MUL_HI_I32", SemOp::V_MUL_HI_I32},
      {"V_MUL_I32_I24", SemOp::V_MUL_I32_I24}, {"V_MUL_U32_U24", SemOp::V_MUL_U32_U24},
      {"V_MAD_U32_U24", SemOp::V_MAD_U32_U24}, {"V_MAD_U32", SemOp::V_MAD_U32},
      {"V_ADD3_U32", SemOp::V_ADD3_U32}, {"V_LSHL_ADD_U32", SemOp::V_LSHL_ADD_U32},
      {"V_LSHL_OR_B32", SemOp::V_LSHL_OR_B32},
      {"V_AND_OR_B32", SemOp::V_AND_OR_B32}, {"V_OR3_B32", SemOp::V_OR3_B32},
      {"V_BFE_U32", SemOp::V_BFE_U32}, {"V_PERM_B32", SemOp::V_PERM_B32},
      {"V_MBCNT_LO_U32_B32", SemOp::V_MBCNT_LO_U32_B32},
      {"V_MBCNT_HI_U32_B32", SemOp::V_MBCNT_HI_U32_B32},
      {"V_READLANE_B32", SemOp::V_READLANE_B32}, {"V_WRITELANE_B32", SemOp::V_WRITELANE_B32},
      {"V_MED3_F32", SemOp::V_MED3_F32}, {"V_MAX3_F32", SemOp::V_MAX3_F32},
      {"V_MIN3_F32", SemOp::V_MIN3_F32}, {"V_MAX3_NUM_F32", SemOp::V_MAX3_NUM_F32},
      {"V_BITOP3_B32", SemOp::V_BITOP3_B32}, {"V_BITOP3_B16", SemOp::V_BITOP3_B16},
      {"V_FMA_MIX_F32", SemOp::V_FMA_MIX_F32},
      {"V_ADD_F16", SemOp::V_ADD_F16}, {"V_MUL_F16", SemOp::V_MUL_F16},
      {"V_PACK_B32_F16", SemOp::V_PACK_B32_F16},
      {"V_CVT_PK_BF16_F32", SemOp::V_CVT_PK_BF16_F32},
      {"V_CVT_PK_BF8_F32", SemOp::V_CVT_PK_BF8_F32},
      {"V_CVT_PK_FP8_F32", SemOp::V_CVT_PK_FP8_F32},

      // FP64
      {"V_ADD_F64", SemOp::V_ADD_F64}, {"V_MUL_F64", SemOp::V_MUL_F64},
      {"V_FMA_F64", SemOp::V_FMA_F64},

      {"V_MAX_U32", SemOp::V_MAX_U32}, {"V_MIN_U32", SemOp::V_MIN_U32},
      {"V_MAX_I32", SemOp::V_MAX_I32}, {"V_MIN_I32", SemOp::V_MIN_I32},
      {"V_PERMLANE16_B32", SemOp::V_PERMLANE16_B32},
      {"V_PERMLANEX16_B32", SemOp::V_PERMLANEX16_B32},
      {"V_PERMLANE64_B32", SemOp::V_PERMLANE64_B32},

      // VOPC integer
      {"V_CMP_EQ_U32", SemOp::V_CMP_EQ_U32}, {"V_CMP_NE_U32", SemOp::V_CMP_NE_U32},
      {"V_CMP_GT_U32", SemOp::V_CMP_GT_U32}, {"V_CMP_GE_U32", SemOp::V_CMP_GE_U32},
      {"V_CMP_LT_U32", SemOp::V_CMP_LT_U32}, {"V_CMP_LE_U32", SemOp::V_CMP_LE_U32},
      {"V_CMP_LG_U32", SemOp::V_CMP_NE_U32},
      {"V_CMP_EQ_I32", SemOp::V_CMP_EQ_I32}, {"V_CMP_NE_I32", SemOp::V_CMP_NE_I32},
      {"V_CMP_GT_I32", SemOp::V_CMP_GT_I32}, {"V_CMP_GE_I32", SemOp::V_CMP_GE_I32},
      {"V_CMP_LT_I32", SemOp::V_CMP_LT_I32}, {"V_CMP_LE_I32", SemOp::V_CMP_LE_I32},
      {"V_CMP_LG_I32", SemOp::V_CMP_NE_I32},
      {"V_CMP_EQ_U64", SemOp::V_CMP_EQ_U64}, {"V_CMP_NE_U64", SemOp::V_CMP_NE_U64},
      {"V_CMP_GT_U64", SemOp::V_CMP_GT_U64}, {"V_CMP_GE_U64", SemOp::V_CMP_GE_U64},
      {"V_CMP_LT_U64", SemOp::V_CMP_LT_U64}, {"V_CMP_LE_U64", SemOp::V_CMP_LE_U64},
      {"V_CMP_GT_I64", SemOp::V_CMP_GT_I64}, {"V_CMP_GE_I64", SemOp::V_CMP_GE_I64},
      {"V_CMP_LT_I64", SemOp::V_CMP_LT_I64}, {"V_CMP_LE_I64", SemOp::V_CMP_LE_I64},

      // VOPC float
      {"V_CMP_EQ_F32", SemOp::V_CMP_EQ_F32}, {"V_CMP_NE_F32", SemOp::V_CMP_NE_F32},
      {"V_CMP_GT_F32", SemOp::V_CMP_GT_F32}, {"V_CMP_GE_F32", SemOp::V_CMP_GE_F32},
      {"V_CMP_LT_F32", SemOp::V_CMP_LT_F32}, {"V_CMP_LE_F32", SemOp::V_CMP_LE_F32},
      {"V_CMP_LG_F32", SemOp::V_CMP_LG_F32},
      {"V_CMP_NEQ_F32", SemOp::V_CMP_NEQ_F32},
      {"V_CMP_NLT_F32", SemOp::V_CMP_NLT_F32}, {"V_CMP_NLE_F32", SemOp::V_CMP_NLE_F32},
      {"V_CMP_NGT_F32", SemOp::V_CMP_NGT_F32}, {"V_CMP_NGE_F32", SemOp::V_CMP_NGE_F32},
      {"V_CMP_NLG_F32", SemOp::V_CMP_NLG_F32},
      {"V_CMP_U_F32", SemOp::V_CMP_U_F32}, {"V_CMP_O_F32", SemOp::V_CMP_O_F32},
      {"V_CMP_EQ_F16", SemOp::V_CMP_EQ_F16}, {"V_CMP_NE_F16", SemOp::V_CMP_NE_F16},
      {"V_CMP_GT_F16", SemOp::V_CMP_GT_F16}, {"V_CMP_GE_F16", SemOp::V_CMP_GE_F16},
      {"V_CMP_LT_F16", SemOp::V_CMP_LT_F16}, {"V_CMP_LE_F16", SemOp::V_CMP_LE_F16},
      {"V_CMP_LG_F16", SemOp::V_CMP_LG_F16},
      {"V_CMP_NEQ_F16", SemOp::V_CMP_NEQ_F16},
      {"V_CMP_NLT_F16", SemOp::V_CMP_NLT_F16}, {"V_CMP_NLE_F16", SemOp::V_CMP_NLE_F16},
      {"V_CMP_NGT_F16", SemOp::V_CMP_NGT_F16}, {"V_CMP_NGE_F16", SemOp::V_CMP_NGE_F16},
      {"V_CMP_U_F16", SemOp::V_CMP_U_F16}, {"V_CMP_O_F16", SemOp::V_CMP_O_F16},
      {"V_CMP_EQ_F64", SemOp::V_CMP_EQ_F64}, {"V_CMP_NE_F64", SemOp::V_CMP_NE_F64},
      {"V_CMP_GT_F64", SemOp::V_CMP_GT_F64}, {"V_CMP_GE_F64", SemOp::V_CMP_GE_F64},
      {"V_CMP_LT_F64", SemOp::V_CMP_LT_F64}, {"V_CMP_LE_F64", SemOp::V_CMP_LE_F64},
      {"V_CMP_LG_F64", SemOp::V_CMP_LG_F64},
      {"V_CMP_NEQ_F64", SemOp::V_CMP_NEQ_F64},
      {"V_CMP_NLT_F64", SemOp::V_CMP_NLT_F64}, {"V_CMP_NLE_F64", SemOp::V_CMP_NLE_F64},
      {"V_CMP_NGT_F64", SemOp::V_CMP_NGT_F64}, {"V_CMP_NGE_F64", SemOp::V_CMP_NGE_F64},
      {"V_CMP_U_F64", SemOp::V_CMP_U_F64}, {"V_CMP_O_F64", SemOp::V_CMP_O_F64},

      // CMPX integer
      {"V_CMPX_EQ_U32", SemOp::V_CMPX_EQ_U32}, {"V_CMPX_NE_U32", SemOp::V_CMPX_NE_U32},
      {"V_CMPX_GT_U32", SemOp::V_CMPX_GT_U32}, {"V_CMPX_GE_U32", SemOp::V_CMPX_GE_U32},
      {"V_CMPX_LT_U32", SemOp::V_CMPX_LT_U32}, {"V_CMPX_LE_U32", SemOp::V_CMPX_LE_U32},
      {"V_CMPX_LG_U32", SemOp::V_CMPX_NE_U32},
      {"V_CMPX_EQ_I32", SemOp::V_CMPX_EQ_I32}, {"V_CMPX_NE_I32", SemOp::V_CMPX_NE_I32},
      {"V_CMPX_GT_I32", SemOp::V_CMPX_GT_I32}, {"V_CMPX_GE_I32", SemOp::V_CMPX_GE_I32},
      {"V_CMPX_LT_I32", SemOp::V_CMPX_LT_I32}, {"V_CMPX_LE_I32", SemOp::V_CMPX_LE_I32},

      // CMPX float
      {"V_CMPX_EQ_F32", SemOp::V_CMPX_EQ_F32}, {"V_CMPX_NE_F32", SemOp::V_CMPX_NE_F32},
      {"V_CMPX_GT_F32", SemOp::V_CMPX_GT_F32}, {"V_CMPX_GE_F32", SemOp::V_CMPX_GE_F32},
      {"V_CMPX_LT_F32", SemOp::V_CMPX_LT_F32}, {"V_CMPX_LE_F32", SemOp::V_CMPX_LE_F32},
      {"V_CMPX_LG_F32", SemOp::V_CMPX_LG_F32},
      {"V_CMPX_NEQ_F32", SemOp::V_CMPX_NEQ_F32},
      {"V_CMPX_NLT_F32", SemOp::V_CMPX_NLT_F32}, {"V_CMPX_NLE_F32", SemOp::V_CMPX_NLE_F32},
      {"V_CMPX_NGT_F32", SemOp::V_CMPX_NGT_F32}, {"V_CMPX_NGE_F32", SemOp::V_CMPX_NGE_F32},
      {"V_CMPX_EQ_F16", SemOp::V_CMPX_EQ_F16}, {"V_CMPX_NE_F16", SemOp::V_CMPX_NE_F16},
      {"V_CMPX_GT_F16", SemOp::V_CMPX_GT_F16}, {"V_CMPX_GE_F16", SemOp::V_CMPX_GE_F16},
      {"V_CMPX_LT_F16", SemOp::V_CMPX_LT_F16}, {"V_CMPX_LE_F16", SemOp::V_CMPX_LE_F16},
      {"V_CMPX_LG_F16", SemOp::V_CMPX_LG_F16},
      {"V_CMPX_NEQ_F16", SemOp::V_CMPX_NEQ_F16},

      // VOP3P
      {"V_PK_ADD_F32", SemOp::V_PK_ADD_F32}, {"V_PK_MUL_F32", SemOp::V_PK_MUL_F32},
      {"V_PK_FMA_F32", SemOp::V_PK_FMA_F32},
      {"V_PK_MAX_F32", SemOp::V_PK_MAX_F32}, {"V_PK_MIN_F32", SemOp::V_PK_MIN_F32},
      {"V_PK_MOV_B32", SemOp::V_PK_MOV_B32},

      // 64-bit vector
      {"V_LSHLREV_B64", SemOp::V_LSHLREV_B64}, {"V_LSHL_ADD_U64", SemOp::V_LSHL_ADD_U64},
      {"V_ADD_NC_U64", SemOp::V_ADD_NC_U64}, {"V_ADD_U64", SemOp::V_ADD_NC_U64},
      {"V_MAD_U64_U32", SemOp::V_MAD_U64_U32}, {"V_MAD_CO_U64_U32", SemOp::V_MAD_CO_U64_U32},
      {"V_ADD_I32", SemOp::V_ADD_I32_legacy}, {"V_SUB_I32", SemOp::V_SUB_I32_legacy},

      // FLAT
      {"FLAT_LOAD_UBYTE", SemOp::FLAT_LOAD_UBYTE}, {"FLAT_LOAD_SBYTE", SemOp::FLAT_LOAD_SBYTE},
      {"FLAT_LOAD_USHORT", SemOp::FLAT_LOAD_USHORT}, {"FLAT_LOAD_SSHORT", SemOp::FLAT_LOAD_SSHORT},
      {"FLAT_LOAD_DWORD", SemOp::FLAT_LOAD_DWORD}, {"FLAT_LOAD_DWORDX2", SemOp::FLAT_LOAD_DWORDX2},
      {"FLAT_LOAD_DWORDX3", SemOp::FLAT_LOAD_DWORDX3}, {"FLAT_LOAD_DWORDX4", SemOp::FLAT_LOAD_DWORDX4},
      {"FLAT_STORE_BYTE", SemOp::FLAT_STORE_BYTE}, {"FLAT_STORE_SHORT", SemOp::FLAT_STORE_SHORT},
      {"FLAT_STORE_SHORT_D16_HI", SemOp::FLAT_STORE_SHORT_D16_HI},
      {"FLAT_STORE_DWORD", SemOp::FLAT_STORE_DWORD}, {"FLAT_STORE_DWORDX2", SemOp::FLAT_STORE_DWORDX2},
      {"FLAT_STORE_DWORDX3", SemOp::FLAT_STORE_DWORDX3}, {"FLAT_STORE_DWORDX4", SemOp::FLAT_STORE_DWORDX4},
      {"GLOBAL_LOAD_UBYTE", SemOp::GLOBAL_LOAD_UBYTE}, {"GLOBAL_LOAD_SBYTE", SemOp::GLOBAL_LOAD_SBYTE},
      {"GLOBAL_LOAD_USHORT", SemOp::GLOBAL_LOAD_USHORT}, {"GLOBAL_LOAD_SSHORT", SemOp::GLOBAL_LOAD_SSHORT},
      {"GLOBAL_LOAD_SHORT_D16_HI", SemOp::GLOBAL_LOAD_SHORT_D16_HI},
      {"GLOBAL_LOAD_DWORD", SemOp::GLOBAL_LOAD_DWORD}, {"GLOBAL_LOAD_DWORDX2", SemOp::GLOBAL_LOAD_DWORDX2},
      {"GLOBAL_LOAD_DWORDX3", SemOp::GLOBAL_LOAD_DWORDX3}, {"GLOBAL_LOAD_DWORDX4", SemOp::GLOBAL_LOAD_DWORDX4},
      {"GLOBAL_STORE_BYTE", SemOp::GLOBAL_STORE_BYTE}, {"GLOBAL_STORE_SHORT", SemOp::GLOBAL_STORE_SHORT},
      {"GLOBAL_STORE_SHORT_D16_HI", SemOp::GLOBAL_STORE_SHORT_D16_HI},
      {"GLOBAL_STORE_DWORD", SemOp::GLOBAL_STORE_DWORD}, {"GLOBAL_STORE_DWORDX2", SemOp::GLOBAL_STORE_DWORDX2},
      {"GLOBAL_STORE_DWORDX3", SemOp::GLOBAL_STORE_DWORDX3}, {"GLOBAL_STORE_DWORDX4", SemOp::GLOBAL_STORE_DWORDX4},

      // FLAT atomics
      {"FLAT_ATOMIC_ADD", SemOp::FLAT_ATOMIC_ADD},
      {"FLAT_ATOMIC_ADD_U32", SemOp::FLAT_ATOMIC_ADD},
      {"FLAT_ATOMIC_SUB", SemOp::FLAT_ATOMIC_SUB},
      {"FLAT_ATOMIC_SUB_U32", SemOp::FLAT_ATOMIC_SUB},
      {"FLAT_ATOMIC_AND", SemOp::FLAT_ATOMIC_AND},
      {"FLAT_ATOMIC_AND_B32", SemOp::FLAT_ATOMIC_AND},
      {"FLAT_ATOMIC_OR", SemOp::FLAT_ATOMIC_OR},
      {"FLAT_ATOMIC_OR_B32", SemOp::FLAT_ATOMIC_OR},
      {"FLAT_ATOMIC_XOR", SemOp::FLAT_ATOMIC_XOR},
      {"FLAT_ATOMIC_XOR_B32", SemOp::FLAT_ATOMIC_XOR},
      {"FLAT_ATOMIC_SMIN", SemOp::FLAT_ATOMIC_SMIN},
      {"FLAT_ATOMIC_SMIN_I32", SemOp::FLAT_ATOMIC_SMIN},
      {"FLAT_ATOMIC_SMAX", SemOp::FLAT_ATOMIC_SMAX},
      {"FLAT_ATOMIC_SMAX_I32", SemOp::FLAT_ATOMIC_SMAX},
      {"FLAT_ATOMIC_UMIN", SemOp::FLAT_ATOMIC_UMIN},
      {"FLAT_ATOMIC_UMIN_U32", SemOp::FLAT_ATOMIC_UMIN},
      {"FLAT_ATOMIC_UMAX", SemOp::FLAT_ATOMIC_UMAX},
      {"FLAT_ATOMIC_UMAX_U32", SemOp::FLAT_ATOMIC_UMAX},
      {"FLAT_ATOMIC_SWAP", SemOp::FLAT_ATOMIC_SWAP},
      {"FLAT_ATOMIC_SWAP_B32", SemOp::FLAT_ATOMIC_SWAP},
      {"FLAT_ATOMIC_CMPSWAP", SemOp::FLAT_ATOMIC_CMPSWAP},
      {"FLAT_ATOMIC_CMPSWAP_B32", SemOp::FLAT_ATOMIC_CMPSWAP},
      {"FLAT_ATOMIC_ADD_F32", SemOp::FLAT_ATOMIC_ADD_F32},

      // GLOBAL atomics
      {"GLOBAL_ATOMIC_ADD", SemOp::GLOBAL_ATOMIC_ADD},
      {"GLOBAL_ATOMIC_ADD_U32", SemOp::GLOBAL_ATOMIC_ADD},
      {"GLOBAL_ATOMIC_SUB", SemOp::GLOBAL_ATOMIC_SUB},
      {"GLOBAL_ATOMIC_SUB_U32", SemOp::GLOBAL_ATOMIC_SUB},
      {"GLOBAL_ATOMIC_AND", SemOp::GLOBAL_ATOMIC_AND},
      {"GLOBAL_ATOMIC_AND_B32", SemOp::GLOBAL_ATOMIC_AND},
      {"GLOBAL_ATOMIC_OR", SemOp::GLOBAL_ATOMIC_OR},
      {"GLOBAL_ATOMIC_OR_B32", SemOp::GLOBAL_ATOMIC_OR},
      {"GLOBAL_ATOMIC_XOR", SemOp::GLOBAL_ATOMIC_XOR},
      {"GLOBAL_ATOMIC_XOR_B32", SemOp::GLOBAL_ATOMIC_XOR},
      {"GLOBAL_ATOMIC_SMIN", SemOp::GLOBAL_ATOMIC_SMIN},
      {"GLOBAL_ATOMIC_SMIN_I32", SemOp::GLOBAL_ATOMIC_SMIN},
      {"GLOBAL_ATOMIC_SMAX", SemOp::GLOBAL_ATOMIC_SMAX},
      {"GLOBAL_ATOMIC_SMAX_I32", SemOp::GLOBAL_ATOMIC_SMAX},
      {"GLOBAL_ATOMIC_UMIN", SemOp::GLOBAL_ATOMIC_UMIN},
      {"GLOBAL_ATOMIC_UMIN_U32", SemOp::GLOBAL_ATOMIC_UMIN},
      {"GLOBAL_ATOMIC_UMAX", SemOp::GLOBAL_ATOMIC_UMAX},
      {"GLOBAL_ATOMIC_UMAX_U32", SemOp::GLOBAL_ATOMIC_UMAX},
      {"GLOBAL_ATOMIC_SWAP", SemOp::GLOBAL_ATOMIC_SWAP},
      {"GLOBAL_ATOMIC_SWAP_B32", SemOp::GLOBAL_ATOMIC_SWAP},
      {"GLOBAL_ATOMIC_CMPSWAP", SemOp::GLOBAL_ATOMIC_CMPSWAP},
      {"GLOBAL_ATOMIC_CMPSWAP_B32", SemOp::GLOBAL_ATOMIC_CMPSWAP},
      {"GLOBAL_ATOMIC_ADD_F32", SemOp::GLOBAL_ATOMIC_ADD_F32},
      {"GLOBAL_ATOMIC_PK_ADD_BF16", SemOp::GLOBAL_ATOMIC_PK_ADD_BF16},
      {"GLOBAL_ATOMIC_PK_ADD_F16", SemOp::GLOBAL_ATOMIC_PK_ADD_F16},

      // DS
      {"DS_LOAD_TR16_B128", SemOp::DS_LOAD_TR16_B128},
      {"DS_READ_B32", SemOp::DS_READ_B32}, {"DS_READ_B64", SemOp::DS_READ_B64},
      {"DS_READ_B128", SemOp::DS_READ_B128},
      {"DS_READ2_B32", SemOp::DS_READ2_B32}, {"DS_READ2_B64", SemOp::DS_READ2_B64},
      {"DS_READ_U16", SemOp::DS_READ_U16}, {"DS_READ_I16", SemOp::DS_READ_I16},
      {"DS_READ_U8", SemOp::DS_READ_U8}, {"DS_READ_I8", SemOp::DS_READ_I8},
      {"DS_WRITE_B32", SemOp::DS_WRITE_B32}, {"DS_WRITE_B64", SemOp::DS_WRITE_B64},
      {"DS_WRITE_B128", SemOp::DS_WRITE_B128},
      {"DS_WRITE2_B32", SemOp::DS_WRITE2_B32}, {"DS_WRITE2_B64", SemOp::DS_WRITE2_B64},
      {"DS_WRITE_B16", SemOp::DS_WRITE_B16}, {"DS_WRITE_B8", SemOp::DS_WRITE_B8},
      {"DS_BPERMUTE_B32", SemOp::DS_BPERMUTE_B32},

      // MUBUF
      {"BUFFER_LOAD_DWORD", SemOp::BUFFER_LOAD_DWORD},
      {"BUFFER_LOAD_B32", SemOp::BUFFER_LOAD_DWORD},
      {"BUFFER_LOAD_B32_VBUFFER_OFFSET", SemOp::BUFFER_LOAD_DWORD},
      {"BUFFER_LOAD_B32_VBUFFER_OFFEN", SemOp::BUFFER_LOAD_DWORD},
      {"BUFFER_LOAD_B32_VBUFFER_IDXEN", SemOp::BUFFER_LOAD_DWORD},
      {"BUFFER_LOAD_B32_VBUFFER_BOTHEN", SemOp::BUFFER_LOAD_DWORD},
      {"BUFFER_LOAD_DWORDX2", SemOp::BUFFER_LOAD_DWORDX2},
      {"BUFFER_LOAD_DWORDX3", SemOp::BUFFER_LOAD_DWORDX3},
      {"BUFFER_LOAD_DWORDX4", SemOp::BUFFER_LOAD_DWORDX4},
      {"BUFFER_LOAD_UBYTE", SemOp::BUFFER_LOAD_UBYTE},
      {"BUFFER_LOAD_SBYTE", SemOp::BUFFER_LOAD_SBYTE},
      {"BUFFER_LOAD_USHORT", SemOp::BUFFER_LOAD_USHORT},
      {"BUFFER_LOAD_SSHORT", SemOp::BUFFER_LOAD_SSHORT},
      {"BUFFER_LOAD_SHORT_D16_HI", SemOp::BUFFER_LOAD_SHORT_D16_HI},
      {"BUFFER_STORE_DWORD", SemOp::BUFFER_STORE_DWORD},
      {"BUFFER_STORE_B32", SemOp::BUFFER_STORE_DWORD},
      {"BUFFER_STORE_B32_VBUFFER_OFFSET", SemOp::BUFFER_STORE_DWORD},
      {"BUFFER_STORE_B32_VBUFFER_OFFEN", SemOp::BUFFER_STORE_DWORD},
      {"BUFFER_STORE_B32_VBUFFER_IDXEN", SemOp::BUFFER_STORE_DWORD},
      {"BUFFER_STORE_B32_VBUFFER_BOTHEN", SemOp::BUFFER_STORE_DWORD},
      {"BUFFER_STORE_DWORDX2", SemOp::BUFFER_STORE_DWORDX2},
      {"BUFFER_STORE_DWORDX3", SemOp::BUFFER_STORE_DWORDX3},
      {"BUFFER_STORE_DWORDX4", SemOp::BUFFER_STORE_DWORDX4},
      {"BUFFER_STORE_BYTE", SemOp::BUFFER_STORE_BYTE},
      {"BUFFER_STORE_SHORT", SemOp::BUFFER_STORE_SHORT},

      // MUBUF atomics
      {"BUFFER_ATOMIC_ADD", SemOp::BUFFER_ATOMIC_ADD},
      {"BUFFER_ATOMIC_ADD_U32", SemOp::BUFFER_ATOMIC_ADD},
      {"BUFFER_ATOMIC_SUB", SemOp::BUFFER_ATOMIC_SUB},
      {"BUFFER_ATOMIC_SUB_U32", SemOp::BUFFER_ATOMIC_SUB},
      {"BUFFER_ATOMIC_AND", SemOp::BUFFER_ATOMIC_AND},
      {"BUFFER_ATOMIC_AND_B32", SemOp::BUFFER_ATOMIC_AND},
      {"BUFFER_ATOMIC_OR", SemOp::BUFFER_ATOMIC_OR},
      {"BUFFER_ATOMIC_OR_B32", SemOp::BUFFER_ATOMIC_OR},
      {"BUFFER_ATOMIC_XOR", SemOp::BUFFER_ATOMIC_XOR},
      {"BUFFER_ATOMIC_XOR_B32", SemOp::BUFFER_ATOMIC_XOR},
      {"BUFFER_ATOMIC_ADD_F32", SemOp::BUFFER_ATOMIC_ADD_F32},
      {"BUFFER_ATOMIC_PK_ADD_BF16", SemOp::BUFFER_ATOMIC_PK_ADD_BF16},
      {"BUFFER_ATOMIC_PK_ADD_F16", SemOp::BUFFER_ATOMIC_PK_ADD_F16},

      // AGPR
      {"V_ACCVGPR_READ_B32", SemOp::V_ACCVGPR_READ_B32},
      {"V_ACCVGPR_WRITE_B32", SemOp::V_ACCVGPR_WRITE_B32},

      // MFMA
      {"V_MFMA_F32_16X16X128_F8F6F4", SemOp::V_MFMA_F32_16x16x128_F8F6F4},
      {"V_MFMA_SCALE_F32_16X16X128_F8F6F4", SemOp::V_MFMA_SCALE_F32_16x16x128_F8F6F4},
      {"V_MFMA_F32_32X32X64_F8F6F4", SemOp::V_MFMA_F32_32x32x64_F8F6F4},
      {"V_MFMA_SCALE_F32_32X32X64_F8F6F4", SemOp::V_MFMA_SCALE_F32_32x32x64_F8F6F4},

      // WMMA (gfx1250)
      {"V_WMMA_F32_16X16X32_F16", SemOp::V_WMMA_F32_16x16x32_F16},
    };

    // Build name → SemOp lookup
    StringMap<SemOp> nameMap;
    for (const auto &[name, sem] : table)
      nameMap[name] = sem;

    // Iterate all MCInstrInfo opcodes; strip suffixes and look up
    unsigned numOpc = MCII.getNumOpcodes();
    map.reserve(numOpc / 4);
    for (unsigned i = 0; i < numOpc; ++i) {
      StringRef full = MCII.getName(i);
      StringRef base = stripMCIISuffix(full);
      auto it = nameMap.find(base);
      if (it != nameMap.end())
        map[i] = it->second;
    }
  }
StringRef OpcodeMap::stripMCIISuffix(StringRef name) {
    // Iteratively strip suffixes until no more match, since they can nest:
    // e.g. V_BITOP3_B16_gfx1250_e64_dpp → _dpp → _e64 → _gfx1250 → V_BITOP3_B16
    bool changed = true;
    while (changed) {
      changed = false;
      for (StringRef suf : {"_dpp8", "_dpp16", "_dpp", "_sdwa",
                            "_e32", "_e64",
                            "_gfx1250", "_gfx12", "_gfx11", "_gfx10", "_gfx9",
                            "_gfx1170", "_gfx8", "_vi", "_si", "_ci",
                            "_SADDR", "_IMM",
                            "_VBUFFER_OFFSET", "_VBUFFER_OFFEN",
                            "_VBUFFER_IDXEN", "_VBUFFER_BOTHEN",
                            "_w32", "_w64", "_twoaddr", "_threeaddr",
                            "_fake16", "_t16"}) {
        if (name.ends_with(suf)) {
          name = name.drop_back(suf.size());
          changed = true;
        }
      }
    }
    // Some GFX12 names embed the mnemonic twice:
    // "V_CVT_F16_F32V_CVT_F16_F32" — take the first half
    // Detect by checking if the name has a repeated pattern
    if (name.size() > 8) {
      size_t half = name.size() / 2;
      if (name.substr(0, half) == name.substr(half))
        name = name.substr(0, half);
    }
    return name;
}

} // namespace transpiler

