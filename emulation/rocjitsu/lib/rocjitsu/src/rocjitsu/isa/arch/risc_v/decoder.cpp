// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/risc_v/decoder.h"
#include "rocjitsu/isa/arch/risc_v/insts.h"

#include <memory>

namespace rocjitsu {
namespace risc_v {

using detail::AddiInst;
using detail::AddInst;
using detail::AddiwInst;
using detail::AddwInst;
using detail::AmoaddDInst;
using detail::AmoaddWInst;
using detail::AmoandDInst;
using detail::AmoandWInst;
using detail::AmomaxDInst;
using detail::AmomaxuDInst;
using detail::AmomaxuWInst;
using detail::AmomaxWInst;
using detail::AmominDInst;
using detail::AmominuDInst;
using detail::AmominuWInst;
using detail::AmominWInst;
using detail::AmoorDInst;
using detail::AmoorWInst;
using detail::AmoswapDInst;
using detail::AmoswapWInst;
using detail::AmoxorDInst;
using detail::AmoxorWInst;
using detail::AndiInst;
using detail::AndInst;
using detail::AuipcInst;
using detail::BeqInst;
using detail::BgeInst;
using detail::BgeuInst;
using detail::BltInst;
using detail::BltuInst;
using detail::BneInst;
using detail::CsrrciInst;
using detail::CsrrcInst;
using detail::CsrrsiInst;
using detail::CsrrsInst;
using detail::CsrrwiInst;
using detail::CsrrwInst;
using detail::DivInst;
using detail::DivuInst;
using detail::DivuwInst;
using detail::DivwInst;
using detail::EbreakInst;
using detail::EcallInst;
using detail::FaddDInst;
using detail::FaddSInst;
using detail::FclassDInst;
using detail::FclassSInst;
using detail::FcvtDLInst;
using detail::FcvtDLuInst;
using detail::FcvtDSInst;
using detail::FcvtDWInst;
using detail::FcvtDWuInst;
using detail::FcvtLDInst;
using detail::FcvtLSInst;
using detail::FcvtLuDInst;
using detail::FcvtLuSInst;
using detail::FcvtSDInst;
using detail::FcvtSLInst;
using detail::FcvtSLuInst;
using detail::FcvtSWInst;
using detail::FcvtSWuInst;
using detail::FcvtWDInst;
using detail::FcvtWSInst;
using detail::FcvtWuDInst;
using detail::FcvtWuSInst;
using detail::FdivDInst;
using detail::FdivSInst;
using detail::FenceIInst;
using detail::FenceInst;
using detail::FeqDInst;
using detail::FeqSInst;
using detail::FldInst;
using detail::FleDInst;
using detail::FleSInst;
using detail::FltDInst;
using detail::FltSInst;
using detail::FlwInst;
using detail::FmaddDInst;
using detail::FmaddSInst;
using detail::FmaxDInst;
using detail::FmaxSInst;
using detail::FminDInst;
using detail::FminSInst;
using detail::FmsubDInst;
using detail::FmsubSInst;
using detail::FmulDInst;
using detail::FmulSInst;
using detail::FmvDXInst;
using detail::FmvWXInst;
using detail::FmvXDInst;
using detail::FmvXWInst;
using detail::FnmaddDInst;
using detail::FnmaddSInst;
using detail::FnmsubDInst;
using detail::FnmsubSInst;
using detail::FsdInst;
using detail::FsgnjDInst;
using detail::FsgnjnDInst;
using detail::FsgnjnSInst;
using detail::FsgnjSInst;
using detail::FsgnjxDInst;
using detail::FsgnjxSInst;
using detail::FsqrtDInst;
using detail::FsqrtSInst;
using detail::FsubDInst;
using detail::FsubSInst;
using detail::FswInst;
using detail::JalInst;
using detail::JalrInst;
using detail::LbInst;
using detail::LbuInst;
using detail::LdInst;
using detail::LhInst;
using detail::LhuInst;
using detail::LrDInst;
using detail::LrWInst;
using detail::LuiInst;
using detail::LwInst;
using detail::LwuInst;
using detail::MretInst;
using detail::MulhInst;
using detail::MulhsuInst;
using detail::MulhuInst;
using detail::MulInst;
using detail::MulwInst;
using detail::OriInst;
using detail::OrInst;
using detail::RemInst;
using detail::RemuInst;
using detail::RemuwInst;
using detail::RemwInst;
using detail::SbInst;
using detail::ScDInst;
using detail::ScWInst;
using detail::SdInst;
using detail::SfenceVmaInst;
using detail::ShInst;
using detail::SlliInst;
using detail::SllInst;
using detail::SlliwInst;
using detail::SllwInst;
using detail::SltiInst;
using detail::SltInst;
using detail::SltiuInst;
using detail::SltuInst;
using detail::SraiInst;
using detail::SraInst;
using detail::SraiwInst;
using detail::SrawInst;
using detail::SretInst;
using detail::SrliInst;
using detail::SrlInst;
using detail::SrliwInst;
using detail::SrlwInst;
using detail::SubInst;
using detail::SubwInst;
using detail::SwInst;
using detail::WfiInst;
using detail::XoriInst;
using detail::XorInst;

// Decode stub definitions - each leaf returns a concrete instruction object.
// illegal_decode returns nullptr.

std::unique_ptr<Instruction> Decoder::illegal_decode(uint32_t) { return nullptr; }

// RV64I Base Integer
std::unique_ptr<Instruction> Decoder::lui_decode(uint32_t instr) {
  return std::make_unique<LuiInst>(instr);
}
std::unique_ptr<Instruction> Decoder::auipc_decode(uint32_t instr) {
  return std::make_unique<AuipcInst>(instr);
}
std::unique_ptr<Instruction> Decoder::jal_decode(uint32_t instr) {
  return std::make_unique<JalInst>(instr);
}
std::unique_ptr<Instruction> Decoder::jalr_decode(uint32_t instr) {
  return std::make_unique<JalrInst>(instr);
}
std::unique_ptr<Instruction> Decoder::beq_decode(uint32_t instr) {
  return std::make_unique<BeqInst>(instr);
}
std::unique_ptr<Instruction> Decoder::bne_decode(uint32_t instr) {
  return std::make_unique<BneInst>(instr);
}
std::unique_ptr<Instruction> Decoder::blt_decode(uint32_t instr) {
  return std::make_unique<BltInst>(instr);
}
std::unique_ptr<Instruction> Decoder::bge_decode(uint32_t instr) {
  return std::make_unique<BgeInst>(instr);
}
std::unique_ptr<Instruction> Decoder::bltu_decode(uint32_t instr) {
  return std::make_unique<BltuInst>(instr);
}
std::unique_ptr<Instruction> Decoder::bgeu_decode(uint32_t instr) {
  return std::make_unique<BgeuInst>(instr);
}
std::unique_ptr<Instruction> Decoder::lb_decode(uint32_t instr) {
  return std::make_unique<LbInst>(instr);
}
std::unique_ptr<Instruction> Decoder::lh_decode(uint32_t instr) {
  return std::make_unique<LhInst>(instr);
}
std::unique_ptr<Instruction> Decoder::lw_decode(uint32_t instr) {
  return std::make_unique<LwInst>(instr);
}
std::unique_ptr<Instruction> Decoder::ld_decode(uint32_t instr) {
  return std::make_unique<LdInst>(instr);
}
std::unique_ptr<Instruction> Decoder::lbu_decode(uint32_t instr) {
  return std::make_unique<LbuInst>(instr);
}
std::unique_ptr<Instruction> Decoder::lhu_decode(uint32_t instr) {
  return std::make_unique<LhuInst>(instr);
}
std::unique_ptr<Instruction> Decoder::lwu_decode(uint32_t instr) {
  return std::make_unique<LwuInst>(instr);
}
std::unique_ptr<Instruction> Decoder::sb_decode(uint32_t instr) {
  return std::make_unique<SbInst>(instr);
}
std::unique_ptr<Instruction> Decoder::sh_decode(uint32_t instr) {
  return std::make_unique<ShInst>(instr);
}
std::unique_ptr<Instruction> Decoder::sw_decode(uint32_t instr) {
  return std::make_unique<SwInst>(instr);
}
std::unique_ptr<Instruction> Decoder::sd_decode(uint32_t instr) {
  return std::make_unique<SdInst>(instr);
}
std::unique_ptr<Instruction> Decoder::addi_decode(uint32_t instr) {
  return std::make_unique<AddiInst>(instr);
}
std::unique_ptr<Instruction> Decoder::slti_decode(uint32_t instr) {
  return std::make_unique<SltiInst>(instr);
}
std::unique_ptr<Instruction> Decoder::sltiu_decode(uint32_t instr) {
  return std::make_unique<SltiuInst>(instr);
}
std::unique_ptr<Instruction> Decoder::xori_decode(uint32_t instr) {
  return std::make_unique<XoriInst>(instr);
}
std::unique_ptr<Instruction> Decoder::ori_decode(uint32_t instr) {
  return std::make_unique<OriInst>(instr);
}
std::unique_ptr<Instruction> Decoder::andi_decode(uint32_t instr) {
  return std::make_unique<AndiInst>(instr);
}
std::unique_ptr<Instruction> Decoder::slli_decode(uint32_t instr) {
  return std::make_unique<SlliInst>(instr);
}
std::unique_ptr<Instruction> Decoder::srli_decode(uint32_t instr) {
  return std::make_unique<SrliInst>(instr);
}
std::unique_ptr<Instruction> Decoder::srai_decode(uint32_t instr) {
  return std::make_unique<SraiInst>(instr);
}
std::unique_ptr<Instruction> Decoder::add_decode(uint32_t instr) {
  return std::make_unique<AddInst>(instr);
}
std::unique_ptr<Instruction> Decoder::sub_decode(uint32_t instr) {
  return std::make_unique<SubInst>(instr);
}
std::unique_ptr<Instruction> Decoder::sll_decode(uint32_t instr) {
  return std::make_unique<SllInst>(instr);
}
std::unique_ptr<Instruction> Decoder::slt_decode(uint32_t instr) {
  return std::make_unique<SltInst>(instr);
}
std::unique_ptr<Instruction> Decoder::sltu_decode(uint32_t instr) {
  return std::make_unique<SltuInst>(instr);
}
std::unique_ptr<Instruction> Decoder::xor_decode(uint32_t instr) {
  return std::make_unique<XorInst>(instr);
}
std::unique_ptr<Instruction> Decoder::srl_decode(uint32_t instr) {
  return std::make_unique<SrlInst>(instr);
}
std::unique_ptr<Instruction> Decoder::sra_decode(uint32_t instr) {
  return std::make_unique<SraInst>(instr);
}
std::unique_ptr<Instruction> Decoder::or_decode(uint32_t instr) {
  return std::make_unique<OrInst>(instr);
}
std::unique_ptr<Instruction> Decoder::and_decode(uint32_t instr) {
  return std::make_unique<AndInst>(instr);
}
std::unique_ptr<Instruction> Decoder::addiw_decode(uint32_t instr) {
  return std::make_unique<AddiwInst>(instr);
}
std::unique_ptr<Instruction> Decoder::slliw_decode(uint32_t instr) {
  return std::make_unique<SlliwInst>(instr);
}
std::unique_ptr<Instruction> Decoder::srliw_decode(uint32_t instr) {
  return std::make_unique<SrliwInst>(instr);
}
std::unique_ptr<Instruction> Decoder::sraiw_decode(uint32_t instr) {
  return std::make_unique<SraiwInst>(instr);
}
std::unique_ptr<Instruction> Decoder::addw_decode(uint32_t instr) {
  return std::make_unique<AddwInst>(instr);
}
std::unique_ptr<Instruction> Decoder::subw_decode(uint32_t instr) {
  return std::make_unique<SubwInst>(instr);
}
std::unique_ptr<Instruction> Decoder::sllw_decode(uint32_t instr) {
  return std::make_unique<SllwInst>(instr);
}
std::unique_ptr<Instruction> Decoder::srlw_decode(uint32_t instr) {
  return std::make_unique<SrlwInst>(instr);
}
std::unique_ptr<Instruction> Decoder::sraw_decode(uint32_t instr) {
  return std::make_unique<SrawInst>(instr);
}
std::unique_ptr<Instruction> Decoder::fence_decode(uint32_t instr) {
  return std::make_unique<FenceInst>(instr);
}
std::unique_ptr<Instruction> Decoder::ecall_decode(uint32_t instr) {
  return std::make_unique<EcallInst>(instr);
}
std::unique_ptr<Instruction> Decoder::ebreak_decode(uint32_t instr) {
  return std::make_unique<EbreakInst>(instr);
}

// Privileged
std::unique_ptr<Instruction> Decoder::sret_decode(uint32_t instr) {
  return std::make_unique<SretInst>(instr);
}
std::unique_ptr<Instruction> Decoder::mret_decode(uint32_t instr) {
  return std::make_unique<MretInst>(instr);
}
std::unique_ptr<Instruction> Decoder::wfi_decode(uint32_t instr) {
  return std::make_unique<WfiInst>(instr);
}
std::unique_ptr<Instruction> Decoder::sfence_vma_decode(uint32_t instr) {
  return std::make_unique<SfenceVmaInst>(instr);
}

// Zicsr Extension
std::unique_ptr<Instruction> Decoder::zicsr_csrrw_decode(uint32_t instr) {
  return std::make_unique<CsrrwInst>(instr);
}
std::unique_ptr<Instruction> Decoder::zicsr_csrrs_decode(uint32_t instr) {
  return std::make_unique<CsrrsInst>(instr);
}
std::unique_ptr<Instruction> Decoder::zicsr_csrrc_decode(uint32_t instr) {
  return std::make_unique<CsrrcInst>(instr);
}
std::unique_ptr<Instruction> Decoder::zicsr_csrrwi_decode(uint32_t instr) {
  return std::make_unique<CsrrwiInst>(instr);
}
std::unique_ptr<Instruction> Decoder::zicsr_csrrsi_decode(uint32_t instr) {
  return std::make_unique<CsrrsiInst>(instr);
}
std::unique_ptr<Instruction> Decoder::zicsr_csrrci_decode(uint32_t instr) {
  return std::make_unique<CsrrciInst>(instr);
}

// Zifencei Extension
std::unique_ptr<Instruction> Decoder::zifencei_fence_i_decode(uint32_t instr) {
  return std::make_unique<FenceIInst>(instr);
}

// M Extension
std::unique_ptr<Instruction> Decoder::m_mul_decode(uint32_t instr) {
  return std::make_unique<MulInst>(instr);
}
std::unique_ptr<Instruction> Decoder::m_mulh_decode(uint32_t instr) {
  return std::make_unique<MulhInst>(instr);
}
std::unique_ptr<Instruction> Decoder::m_mulhsu_decode(uint32_t instr) {
  return std::make_unique<MulhsuInst>(instr);
}
std::unique_ptr<Instruction> Decoder::m_mulhu_decode(uint32_t instr) {
  return std::make_unique<MulhuInst>(instr);
}
std::unique_ptr<Instruction> Decoder::m_div_decode(uint32_t instr) {
  return std::make_unique<DivInst>(instr);
}
std::unique_ptr<Instruction> Decoder::m_divu_decode(uint32_t instr) {
  return std::make_unique<DivuInst>(instr);
}
std::unique_ptr<Instruction> Decoder::m_rem_decode(uint32_t instr) {
  return std::make_unique<RemInst>(instr);
}
std::unique_ptr<Instruction> Decoder::m_remu_decode(uint32_t instr) {
  return std::make_unique<RemuInst>(instr);
}
std::unique_ptr<Instruction> Decoder::m_mulw_decode(uint32_t instr) {
  return std::make_unique<MulwInst>(instr);
}
std::unique_ptr<Instruction> Decoder::m_divw_decode(uint32_t instr) {
  return std::make_unique<DivwInst>(instr);
}
std::unique_ptr<Instruction> Decoder::m_divuw_decode(uint32_t instr) {
  return std::make_unique<DivuwInst>(instr);
}
std::unique_ptr<Instruction> Decoder::m_remw_decode(uint32_t instr) {
  return std::make_unique<RemwInst>(instr);
}
std::unique_ptr<Instruction> Decoder::m_remuw_decode(uint32_t instr) {
  return std::make_unique<RemuwInst>(instr);
}

// A Extension
std::unique_ptr<Instruction> Decoder::a_lr_w_decode(uint32_t instr) {
  return std::make_unique<LrWInst>(instr);
}
std::unique_ptr<Instruction> Decoder::a_sc_w_decode(uint32_t instr) {
  return std::make_unique<ScWInst>(instr);
}
std::unique_ptr<Instruction> Decoder::a_amoswap_w_decode(uint32_t instr) {
  return std::make_unique<AmoswapWInst>(instr);
}
std::unique_ptr<Instruction> Decoder::a_amoadd_w_decode(uint32_t instr) {
  return std::make_unique<AmoaddWInst>(instr);
}
std::unique_ptr<Instruction> Decoder::a_amoxor_w_decode(uint32_t instr) {
  return std::make_unique<AmoxorWInst>(instr);
}
std::unique_ptr<Instruction> Decoder::a_amoand_w_decode(uint32_t instr) {
  return std::make_unique<AmoandWInst>(instr);
}
std::unique_ptr<Instruction> Decoder::a_amoor_w_decode(uint32_t instr) {
  return std::make_unique<AmoorWInst>(instr);
}
std::unique_ptr<Instruction> Decoder::a_amomin_w_decode(uint32_t instr) {
  return std::make_unique<AmominWInst>(instr);
}
std::unique_ptr<Instruction> Decoder::a_amomax_w_decode(uint32_t instr) {
  return std::make_unique<AmomaxWInst>(instr);
}
std::unique_ptr<Instruction> Decoder::a_amominu_w_decode(uint32_t instr) {
  return std::make_unique<AmominuWInst>(instr);
}
std::unique_ptr<Instruction> Decoder::a_amomaxu_w_decode(uint32_t instr) {
  return std::make_unique<AmomaxuWInst>(instr);
}
std::unique_ptr<Instruction> Decoder::a_lr_d_decode(uint32_t instr) {
  return std::make_unique<LrDInst>(instr);
}
std::unique_ptr<Instruction> Decoder::a_sc_d_decode(uint32_t instr) {
  return std::make_unique<ScDInst>(instr);
}
std::unique_ptr<Instruction> Decoder::a_amoswap_d_decode(uint32_t instr) {
  return std::make_unique<AmoswapDInst>(instr);
}
std::unique_ptr<Instruction> Decoder::a_amoadd_d_decode(uint32_t instr) {
  return std::make_unique<AmoaddDInst>(instr);
}
std::unique_ptr<Instruction> Decoder::a_amoxor_d_decode(uint32_t instr) {
  return std::make_unique<AmoxorDInst>(instr);
}
std::unique_ptr<Instruction> Decoder::a_amoand_d_decode(uint32_t instr) {
  return std::make_unique<AmoandDInst>(instr);
}
std::unique_ptr<Instruction> Decoder::a_amoor_d_decode(uint32_t instr) {
  return std::make_unique<AmoorDInst>(instr);
}
std::unique_ptr<Instruction> Decoder::a_amomin_d_decode(uint32_t instr) {
  return std::make_unique<AmominDInst>(instr);
}
std::unique_ptr<Instruction> Decoder::a_amomax_d_decode(uint32_t instr) {
  return std::make_unique<AmomaxDInst>(instr);
}
std::unique_ptr<Instruction> Decoder::a_amominu_d_decode(uint32_t instr) {
  return std::make_unique<AmominuDInst>(instr);
}
std::unique_ptr<Instruction> Decoder::a_amomaxu_d_decode(uint32_t instr) {
  return std::make_unique<AmomaxuDInst>(instr);
}

// F Extension
std::unique_ptr<Instruction> Decoder::f_flw_decode(uint32_t instr) {
  return std::make_unique<FlwInst>(instr);
}
std::unique_ptr<Instruction> Decoder::f_fsw_decode(uint32_t instr) {
  return std::make_unique<FswInst>(instr);
}
std::unique_ptr<Instruction> Decoder::f_fadd_s_decode(uint32_t instr) {
  return std::make_unique<FaddSInst>(instr);
}
std::unique_ptr<Instruction> Decoder::f_fsub_s_decode(uint32_t instr) {
  return std::make_unique<FsubSInst>(instr);
}
std::unique_ptr<Instruction> Decoder::f_fmul_s_decode(uint32_t instr) {
  return std::make_unique<FmulSInst>(instr);
}
std::unique_ptr<Instruction> Decoder::f_fdiv_s_decode(uint32_t instr) {
  return std::make_unique<FdivSInst>(instr);
}
std::unique_ptr<Instruction> Decoder::f_fsqrt_s_decode(uint32_t instr) {
  return std::make_unique<FsqrtSInst>(instr);
}
std::unique_ptr<Instruction> Decoder::f_fsgnj_s_decode(uint32_t instr) {
  return std::make_unique<FsgnjSInst>(instr);
}
std::unique_ptr<Instruction> Decoder::f_fsgnjn_s_decode(uint32_t instr) {
  return std::make_unique<FsgnjnSInst>(instr);
}
std::unique_ptr<Instruction> Decoder::f_fsgnjx_s_decode(uint32_t instr) {
  return std::make_unique<FsgnjxSInst>(instr);
}
std::unique_ptr<Instruction> Decoder::f_fmin_s_decode(uint32_t instr) {
  return std::make_unique<FminSInst>(instr);
}
std::unique_ptr<Instruction> Decoder::f_fmax_s_decode(uint32_t instr) {
  return std::make_unique<FmaxSInst>(instr);
}
std::unique_ptr<Instruction> Decoder::f_fcvt_w_s_decode(uint32_t instr) {
  return std::make_unique<FcvtWSInst>(instr);
}
std::unique_ptr<Instruction> Decoder::f_fcvt_wu_s_decode(uint32_t instr) {
  return std::make_unique<FcvtWuSInst>(instr);
}
std::unique_ptr<Instruction> Decoder::f_fcvt_l_s_decode(uint32_t instr) {
  return std::make_unique<FcvtLSInst>(instr);
}
std::unique_ptr<Instruction> Decoder::f_fcvt_lu_s_decode(uint32_t instr) {
  return std::make_unique<FcvtLuSInst>(instr);
}
std::unique_ptr<Instruction> Decoder::f_fcvt_s_w_decode(uint32_t instr) {
  return std::make_unique<FcvtSWInst>(instr);
}
std::unique_ptr<Instruction> Decoder::f_fcvt_s_wu_decode(uint32_t instr) {
  return std::make_unique<FcvtSWuInst>(instr);
}
std::unique_ptr<Instruction> Decoder::f_fcvt_s_l_decode(uint32_t instr) {
  return std::make_unique<FcvtSLInst>(instr);
}
std::unique_ptr<Instruction> Decoder::f_fcvt_s_lu_decode(uint32_t instr) {
  return std::make_unique<FcvtSLuInst>(instr);
}
std::unique_ptr<Instruction> Decoder::f_fmv_x_w_decode(uint32_t instr) {
  return std::make_unique<FmvXWInst>(instr);
}
std::unique_ptr<Instruction> Decoder::f_fclass_s_decode(uint32_t instr) {
  return std::make_unique<FclassSInst>(instr);
}
std::unique_ptr<Instruction> Decoder::f_fmv_w_x_decode(uint32_t instr) {
  return std::make_unique<FmvWXInst>(instr);
}
std::unique_ptr<Instruction> Decoder::f_feq_s_decode(uint32_t instr) {
  return std::make_unique<FeqSInst>(instr);
}
std::unique_ptr<Instruction> Decoder::f_flt_s_decode(uint32_t instr) {
  return std::make_unique<FltSInst>(instr);
}
std::unique_ptr<Instruction> Decoder::f_fle_s_decode(uint32_t instr) {
  return std::make_unique<FleSInst>(instr);
}
std::unique_ptr<Instruction> Decoder::f_fmadd_s_decode(uint32_t instr) {
  return std::make_unique<FmaddSInst>(instr);
}
std::unique_ptr<Instruction> Decoder::f_fmsub_s_decode(uint32_t instr) {
  return std::make_unique<FmsubSInst>(instr);
}
std::unique_ptr<Instruction> Decoder::f_fnmsub_s_decode(uint32_t instr) {
  return std::make_unique<FnmsubSInst>(instr);
}
std::unique_ptr<Instruction> Decoder::f_fnmadd_s_decode(uint32_t instr) {
  return std::make_unique<FnmaddSInst>(instr);
}

// D Extension
std::unique_ptr<Instruction> Decoder::d_fld_decode(uint32_t instr) {
  return std::make_unique<FldInst>(instr);
}
std::unique_ptr<Instruction> Decoder::d_fsd_decode(uint32_t instr) {
  return std::make_unique<FsdInst>(instr);
}
std::unique_ptr<Instruction> Decoder::d_fadd_d_decode(uint32_t instr) {
  return std::make_unique<FaddDInst>(instr);
}
std::unique_ptr<Instruction> Decoder::d_fsub_d_decode(uint32_t instr) {
  return std::make_unique<FsubDInst>(instr);
}
std::unique_ptr<Instruction> Decoder::d_fmul_d_decode(uint32_t instr) {
  return std::make_unique<FmulDInst>(instr);
}
std::unique_ptr<Instruction> Decoder::d_fdiv_d_decode(uint32_t instr) {
  return std::make_unique<FdivDInst>(instr);
}
std::unique_ptr<Instruction> Decoder::d_fsqrt_d_decode(uint32_t instr) {
  return std::make_unique<FsqrtDInst>(instr);
}
std::unique_ptr<Instruction> Decoder::d_fsgnj_d_decode(uint32_t instr) {
  return std::make_unique<FsgnjDInst>(instr);
}
std::unique_ptr<Instruction> Decoder::d_fsgnjn_d_decode(uint32_t instr) {
  return std::make_unique<FsgnjnDInst>(instr);
}
std::unique_ptr<Instruction> Decoder::d_fsgnjx_d_decode(uint32_t instr) {
  return std::make_unique<FsgnjxDInst>(instr);
}
std::unique_ptr<Instruction> Decoder::d_fmin_d_decode(uint32_t instr) {
  return std::make_unique<FminDInst>(instr);
}
std::unique_ptr<Instruction> Decoder::d_fmax_d_decode(uint32_t instr) {
  return std::make_unique<FmaxDInst>(instr);
}
std::unique_ptr<Instruction> Decoder::d_fcvt_w_d_decode(uint32_t instr) {
  return std::make_unique<FcvtWDInst>(instr);
}
std::unique_ptr<Instruction> Decoder::d_fcvt_wu_d_decode(uint32_t instr) {
  return std::make_unique<FcvtWuDInst>(instr);
}
std::unique_ptr<Instruction> Decoder::d_fcvt_l_d_decode(uint32_t instr) {
  return std::make_unique<FcvtLDInst>(instr);
}
std::unique_ptr<Instruction> Decoder::d_fcvt_lu_d_decode(uint32_t instr) {
  return std::make_unique<FcvtLuDInst>(instr);
}
std::unique_ptr<Instruction> Decoder::d_fcvt_d_w_decode(uint32_t instr) {
  return std::make_unique<FcvtDWInst>(instr);
}
std::unique_ptr<Instruction> Decoder::d_fcvt_d_wu_decode(uint32_t instr) {
  return std::make_unique<FcvtDWuInst>(instr);
}
std::unique_ptr<Instruction> Decoder::d_fcvt_d_l_decode(uint32_t instr) {
  return std::make_unique<FcvtDLInst>(instr);
}
std::unique_ptr<Instruction> Decoder::d_fcvt_d_lu_decode(uint32_t instr) {
  return std::make_unique<FcvtDLuInst>(instr);
}
std::unique_ptr<Instruction> Decoder::d_fcvt_s_d_decode(uint32_t instr) {
  return std::make_unique<FcvtSDInst>(instr);
}
std::unique_ptr<Instruction> Decoder::d_fcvt_d_s_decode(uint32_t instr) {
  return std::make_unique<FcvtDSInst>(instr);
}
std::unique_ptr<Instruction> Decoder::d_fmv_x_d_decode(uint32_t instr) {
  return std::make_unique<FmvXDInst>(instr);
}
std::unique_ptr<Instruction> Decoder::d_fclass_d_decode(uint32_t instr) {
  return std::make_unique<FclassDInst>(instr);
}
std::unique_ptr<Instruction> Decoder::d_fmv_d_x_decode(uint32_t instr) {
  return std::make_unique<FmvDXInst>(instr);
}
std::unique_ptr<Instruction> Decoder::d_feq_d_decode(uint32_t instr) {
  return std::make_unique<FeqDInst>(instr);
}
std::unique_ptr<Instruction> Decoder::d_flt_d_decode(uint32_t instr) {
  return std::make_unique<FltDInst>(instr);
}
std::unique_ptr<Instruction> Decoder::d_fle_d_decode(uint32_t instr) {
  return std::make_unique<FleDInst>(instr);
}
std::unique_ptr<Instruction> Decoder::d_fmadd_d_decode(uint32_t instr) {
  return std::make_unique<FmaddDInst>(instr);
}
std::unique_ptr<Instruction> Decoder::d_fmsub_d_decode(uint32_t instr) {
  return std::make_unique<FmsubDInst>(instr);
}
std::unique_ptr<Instruction> Decoder::d_fnmsub_d_decode(uint32_t instr) {
  return std::make_unique<FnmsubDInst>(instr);
}
std::unique_ptr<Instruction> Decoder::d_fnmadd_d_decode(uint32_t instr) {
  return std::make_unique<FnmaddDInst>(instr);
}

// Vendor Custom
std::unique_ptr<Instruction> Decoder::vendor0_ext_decode(uint32_t) { return nullptr; }
std::unique_ptr<Instruction> Decoder::vendor1_ext_decode(uint32_t) { return nullptr; }
std::unique_ptr<Instruction> Decoder::vendor2_ext_decode(uint32_t) { return nullptr; }
std::unique_ptr<Instruction> Decoder::vendor3_ext_decode(uint32_t) { return nullptr; }

// Decode method - dispatches through the opcode table

std::unique_ptr<Instruction> Decoder::decode(uint32_t instr) {
  auto fn = opcode_table_[(instr >> 2) & 0x1F];
  return fn ? fn(instr) : nullptr;
}

// Decode tables - directly initialized, bottom-up (leaves before branches).
// nullptr entries represent illegal/reserved encodings.
// Trailing array elements are zero-initialized (nullptr) by C++.

// Level 3: SYSTEM privileged rs2 tables (32 entries each)

const DecodeFn Decoder::system_priv_ecall_rs2_[32] = {
    &ecall_decode,
    &ebreak_decode,
};

const DecodeFn Decoder::system_priv_sret_rs2_[32] = {
    nullptr, nullptr, &sret_decode, nullptr, nullptr, &wfi_decode,
};

const DecodeFn Decoder::system_priv_mret_rs2_[32] = {
    nullptr,
    nullptr,
    &mret_decode,
};

// Level 2: OP-IMM / OP-IMM-32 shift-right bit30 (2 entries each)

const DecodeFn Decoder::op_imm_sri_bit30_[2] = {
    &srli_decode,
    &srai_decode,
};

const DecodeFn Decoder::op_imm32_sri_bit30_[2] = {
    &srliw_decode,
    &sraiw_decode,
};

// Level 2: AMO funct5[31:27] (32 entries per width)

const DecodeFn Decoder::amo_w_funct5_[32] = {
    &a_amoadd_w_decode,
    &a_amoswap_w_decode,
    &a_lr_w_decode,
    &a_sc_w_decode,
    &a_amoxor_w_decode,
    nullptr,
    nullptr,
    nullptr,
    &a_amoor_w_decode,
    nullptr,
    nullptr,
    nullptr,
    &a_amoand_w_decode,
    nullptr,
    nullptr,
    nullptr,
    &a_amomin_w_decode,
    nullptr,
    nullptr,
    nullptr,
    &a_amomax_w_decode,
    nullptr,
    nullptr,
    nullptr,
    &a_amominu_w_decode,
    nullptr,
    nullptr,
    nullptr,
    &a_amomaxu_w_decode,
};

const DecodeFn Decoder::amo_d_funct5_[32] = {
    &a_amoadd_d_decode,
    &a_amoswap_d_decode,
    &a_lr_d_decode,
    &a_sc_d_decode,
    &a_amoxor_d_decode,
    nullptr,
    nullptr,
    nullptr,
    &a_amoor_d_decode,
    nullptr,
    nullptr,
    nullptr,
    &a_amoand_d_decode,
    nullptr,
    nullptr,
    nullptr,
    &a_amomin_d_decode,
    nullptr,
    nullptr,
    nullptr,
    &a_amomax_d_decode,
    nullptr,
    nullptr,
    nullptr,
    &a_amominu_d_decode,
    nullptr,
    nullptr,
    nullptr,
    &a_amomaxu_d_decode,
};

// Level 2: OP funct7[31:25] (128 entries x 8 funct3 values)
// Valid at 0x00 (base), 0x01 (M-ext), 0x20 (alt). Rest are nullptr.

// funct3=000: ADD(0x00) MUL(0x01) SUB(0x20)
const DecodeFn Decoder::op_f3_000_funct7_[128] = {
    &add_decode, &m_mul_decode, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // 0x00
    nullptr,     nullptr,       nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // 0x08
    nullptr,     nullptr,       nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // 0x10
    nullptr,     nullptr,       nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // 0x18
    &sub_decode,                                                                      // 0x20
};

// funct3=001: SLL(0x00) MULH(0x01)
const DecodeFn Decoder::op_f3_001_funct7_[128] = {
    &sll_decode,
    &m_mulh_decode,
};

// funct3=010: SLT(0x00) MULHSU(0x01)
const DecodeFn Decoder::op_f3_010_funct7_[128] = {
    &slt_decode,
    &m_mulhsu_decode,
};

// funct3=011: SLTU(0x00) MULHU(0x01)
const DecodeFn Decoder::op_f3_011_funct7_[128] = {
    &sltu_decode,
    &m_mulhu_decode,
};

// funct3=100: XOR(0x00) DIV(0x01)
const DecodeFn Decoder::op_f3_100_funct7_[128] = {
    &xor_decode,
    &m_div_decode,
};

// funct3=101: SRL(0x00) DIVU(0x01) SRA(0x20)
const DecodeFn Decoder::op_f3_101_funct7_[128] = {
    &srl_decode, &m_divu_decode, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // 0x00
    nullptr,     nullptr,        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // 0x08
    nullptr,     nullptr,        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // 0x10
    nullptr,     nullptr,        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // 0x18
    &sra_decode,                                                                       // 0x20
};

// funct3=110: OR(0x00) REM(0x01)
const DecodeFn Decoder::op_f3_110_funct7_[128] = {
    &or_decode,
    &m_rem_decode,
};

// funct3=111: AND(0x00) REMU(0x01)
const DecodeFn Decoder::op_f3_111_funct7_[128] = {
    &and_decode,
    &m_remu_decode,
};

// Level 2: OP-32 funct7[31:25] (128 entries x 8 funct3 values)

// funct3=000: ADDW(0x00) MULW(0x01) SUBW(0x20)
const DecodeFn Decoder::op32_f3_000_funct7_[128] = {
    &addw_decode, &m_mulw_decode, nullptr, nullptr, nullptr,      nullptr, nullptr,
    nullptr,      nullptr,        nullptr, nullptr, nullptr,      nullptr, nullptr,
    nullptr,      nullptr,        nullptr, nullptr, nullptr,      nullptr, nullptr,
    nullptr,      nullptr,        nullptr, nullptr, nullptr,      nullptr, nullptr,
    nullptr,      nullptr,        nullptr, nullptr, &subw_decode,
};

// funct3=001: SLLW(0x00)
const DecodeFn Decoder::op32_f3_001_funct7_[128] = {
    &sllw_decode,
};

// funct3=010: all illegal
const DecodeFn Decoder::op32_f3_010_funct7_[128] = {};

// funct3=011: all illegal
const DecodeFn Decoder::op32_f3_011_funct7_[128] = {};

// funct3=100: DIVW(0x01)
const DecodeFn Decoder::op32_f3_100_funct7_[128] = {
    nullptr,
    &m_divw_decode,
};

// funct3=101: SRLW(0x00) DIVUW(0x01) SRAW(0x20)
const DecodeFn Decoder::op32_f3_101_funct7_[128] = {
    &srlw_decode, &m_divuw_decode, nullptr, nullptr, nullptr,      nullptr, nullptr,
    nullptr,      nullptr,         nullptr, nullptr, nullptr,      nullptr, nullptr,
    nullptr,      nullptr,         nullptr, nullptr, nullptr,      nullptr, nullptr,
    nullptr,      nullptr,         nullptr, nullptr, nullptr,      nullptr, nullptr,
    nullptr,      nullptr,         nullptr, nullptr, &sraw_decode,
};

// funct3=110: REMW(0x01)
const DecodeFn Decoder::op32_f3_110_funct7_[128] = {
    nullptr,
    &m_remw_decode,
};

// funct3=111: REMUW(0x01)
const DecodeFn Decoder::op32_f3_111_funct7_[128] = {
    nullptr,
    &m_remuw_decode,
};

// Level 2: OP-FP funct3 subtables (8 entries each)

const DecodeFn Decoder::fp_fsgnj_s_funct3_[8] = {
    &f_fsgnj_s_decode,
    &f_fsgnjn_s_decode,
    &f_fsgnjx_s_decode,
};

const DecodeFn Decoder::fp_fsgnj_d_funct3_[8] = {
    &d_fsgnj_d_decode,
    &d_fsgnjn_d_decode,
    &d_fsgnjx_d_decode,
};

const DecodeFn Decoder::fp_fminmax_s_funct3_[8] = {
    &f_fmin_s_decode,
    &f_fmax_s_decode,
};

const DecodeFn Decoder::fp_fminmax_d_funct3_[8] = {
    &d_fmin_d_decode,
    &d_fmax_d_decode,
};

const DecodeFn Decoder::fp_fcmp_s_funct3_[8] = {
    &f_fle_s_decode,
    &f_flt_s_decode,
    &f_feq_s_decode,
};

const DecodeFn Decoder::fp_fcmp_d_funct3_[8] = {
    &d_fle_d_decode,
    &d_flt_d_decode,
    &d_feq_d_decode,
};

const DecodeFn Decoder::fp_fmvclass_s_funct3_[8] = {
    &f_fmv_x_w_decode,
    &f_fclass_s_decode,
};

const DecodeFn Decoder::fp_fmvclass_d_funct3_[8] = {
    &d_fmv_x_d_decode,
    &d_fclass_d_decode,
};

// Level 2: OP-FP rs2 subtables (32 entries each)

const DecodeFn Decoder::fp_fcvt_s_d_rs2_[32] = {
    nullptr,
    &d_fcvt_s_d_decode,
};

const DecodeFn Decoder::fp_fcvt_d_s_rs2_[32] = {
    &d_fcvt_d_s_decode,
};

const DecodeFn Decoder::fp_fcvt_int_from_s_rs2_[32] = {
    &f_fcvt_w_s_decode,
    &f_fcvt_wu_s_decode,
    &f_fcvt_l_s_decode,
    &f_fcvt_lu_s_decode,
};

const DecodeFn Decoder::fp_fcvt_int_from_d_rs2_[32] = {
    &d_fcvt_w_d_decode,
    &d_fcvt_wu_d_decode,
    &d_fcvt_l_d_decode,
    &d_fcvt_lu_d_decode,
};

const DecodeFn Decoder::fp_fcvt_s_from_int_rs2_[32] = {
    &f_fcvt_s_w_decode,
    &f_fcvt_s_wu_decode,
    &f_fcvt_s_l_decode,
    &f_fcvt_s_lu_decode,
};

const DecodeFn Decoder::fp_fcvt_d_from_int_rs2_[32] = {
    &d_fcvt_d_w_decode,
    &d_fcvt_d_wu_decode,
    &d_fcvt_d_l_decode,
    &d_fcvt_d_lu_decode,
};

// Level 2: SYSTEM privileged funct7[31:25] (128 entries)

const DecodeFn Decoder::system_priv_funct7_[128] = {
    // 0x00-0x07
    &decode_field<system_priv_ecall_rs2_, 20, 5>, // 0x00 ECALL/EBREAK
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    // 0x08-0x0F
    &decode_field<system_priv_sret_rs2_, 20, 5>, // 0x08 SRET/WFI
    &sfence_vma_decode,                          // 0x09 SFENCE.VMA
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    // 0x10-0x17
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    // 0x18-0x1F
    &decode_field<system_priv_mret_rs2_, 20, 5>, // 0x18 MRET
};

// Level 1: OP-FP funct7[31:25] (128 entries)

const DecodeFn Decoder::op_fp_funct7_[128] = {
    // 0x00-0x07
    &f_fadd_s_decode, // 0x00 FADD.S
    &d_fadd_d_decode, // 0x01 FADD.D
    nullptr, nullptr,
    &f_fsub_s_decode, // 0x04 FSUB.S
    &d_fsub_d_decode, // 0x05 FSUB.D
    nullptr, nullptr,
    // 0x08-0x0F
    &f_fmul_s_decode, // 0x08 FMUL.S
    &d_fmul_d_decode, // 0x09 FMUL.D
    nullptr, nullptr,
    &f_fdiv_s_decode, // 0x0C FDIV.S
    &d_fdiv_d_decode, // 0x0D FDIV.D
    nullptr, nullptr,
    // 0x10-0x17
    &decode_field<fp_fsgnj_s_funct3_, 12, 3>,                     // 0x10 FSGNJ*.S
    &decode_field<fp_fsgnj_d_funct3_, 12, 3>,                     // 0x11 FSGNJ*.D
    nullptr, nullptr, &decode_field<fp_fminmax_s_funct3_, 12, 3>, // 0x14 FMIN/FMAX.S
    &decode_field<fp_fminmax_d_funct3_, 12, 3>,                   // 0x15 FMIN/FMAX.D
    nullptr, nullptr,
    // 0x18-0x1F
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    // 0x20-0x27
    &decode_field<fp_fcvt_s_d_rs2_, 20, 5>, // 0x20 FCVT.S.D
    &decode_field<fp_fcvt_d_s_rs2_, 20, 5>, // 0x21 FCVT.D.S
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    // 0x28-0x2F
    nullptr, nullptr, nullptr, nullptr,
    &f_fsqrt_s_decode, // 0x2C FSQRT.S
    &d_fsqrt_d_decode, // 0x2D FSQRT.D
    nullptr, nullptr,
    // 0x30-0x3F
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    // 0x40-0x4F
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    // 0x50-0x57
    &decode_field<fp_fcmp_s_funct3_, 12, 3>, // 0x50 FEQ/FLT/FLE.S
    &decode_field<fp_fcmp_d_funct3_, 12, 3>, // 0x51 FEQ/FLT/FLE.D
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    // 0x58-0x5F
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    // 0x60-0x67
    &decode_field<fp_fcvt_int_from_s_rs2_, 20, 5>, // 0x60 FCVT.W/WU/L/LU.S
    &decode_field<fp_fcvt_int_from_d_rs2_, 20, 5>, // 0x61 FCVT.W/WU/L/LU.D
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    // 0x68-0x6F
    &decode_field<fp_fcvt_s_from_int_rs2_, 20, 5>, // 0x68 FCVT.S.W/WU/L/LU
    &decode_field<fp_fcvt_d_from_int_rs2_, 20, 5>, // 0x69 FCVT.D.W/WU/L/LU
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    // 0x70-0x77
    &decode_field<fp_fmvclass_s_funct3_, 12, 3>, // 0x70 FMV.X.W / FCLASS.S
    &decode_field<fp_fmvclass_d_funct3_, 12, 3>, // 0x71 FMV.X.D / FCLASS.D
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    // 0x78-0x7F
    &f_fmv_w_x_decode, // 0x78 FMV.W.X
    &d_fmv_d_x_decode, // 0x79 FMV.D.X
};

// Level 1: funct3 tables (8 entries each)

const DecodeFn Decoder::load_funct3_[8] = {
    &lb_decode, &lh_decode, &lw_decode, &ld_decode, &lbu_decode, &lhu_decode, &lwu_decode, nullptr,
};

const DecodeFn Decoder::load_fp_funct3_[8] = {
    nullptr,
    nullptr,
    &f_flw_decode,
    &d_fld_decode,
};

const DecodeFn Decoder::misc_mem_funct3_[8] = {
    &fence_decode,
    &zifencei_fence_i_decode,
};

const DecodeFn Decoder::op_imm_funct3_[8] = {
    &addi_decode,                            // 000
    &slli_decode,                            // 001
    &slti_decode,                            // 010
    &sltiu_decode,                           // 011
    &xori_decode,                            // 100
    &decode_field<op_imm_sri_bit30_, 30, 1>, // 101 SRLI/SRAI
    &ori_decode,                             // 110
    &andi_decode,                            // 111
};

const DecodeFn Decoder::op_imm_32_funct3_[8] = {
    &addiw_decode, // 000
    &slliw_decode, // 001
    nullptr,
    nullptr,
    nullptr,                                   // 010-100
    &decode_field<op_imm32_sri_bit30_, 30, 1>, // 101 SRLIW/SRAIW
};

const DecodeFn Decoder::store_funct3_[8] = {
    &sb_decode,
    &sh_decode,
    &sw_decode,
    &sd_decode,
};

const DecodeFn Decoder::store_fp_funct3_[8] = {
    nullptr,
    nullptr,
    &f_fsw_decode,
    &d_fsd_decode,
};

const DecodeFn Decoder::amo_funct3_[8] = {
    nullptr, nullptr, &decode_field<amo_w_funct5_, 27, 5>, // 010 word
    &decode_field<amo_d_funct5_, 27, 5>,                   // 011 doubleword
};

const DecodeFn Decoder::op_funct3_[8] = {
    &decode_field<op_f3_000_funct7_, 25, 7>, &decode_field<op_f3_001_funct7_, 25, 7>,
    &decode_field<op_f3_010_funct7_, 25, 7>, &decode_field<op_f3_011_funct7_, 25, 7>,
    &decode_field<op_f3_100_funct7_, 25, 7>, &decode_field<op_f3_101_funct7_, 25, 7>,
    &decode_field<op_f3_110_funct7_, 25, 7>, &decode_field<op_f3_111_funct7_, 25, 7>,
};

const DecodeFn Decoder::op_32_funct3_[8] = {
    &decode_field<op32_f3_000_funct7_, 25, 7>, &decode_field<op32_f3_001_funct7_, 25, 7>,
    &decode_field<op32_f3_010_funct7_, 25, 7>, &decode_field<op32_f3_011_funct7_, 25, 7>,
    &decode_field<op32_f3_100_funct7_, 25, 7>, &decode_field<op32_f3_101_funct7_, 25, 7>,
    &decode_field<op32_f3_110_funct7_, 25, 7>, &decode_field<op32_f3_111_funct7_, 25, 7>,
};

const DecodeFn Decoder::branch_funct3_[8] = {
    &beq_decode, &bne_decode, nullptr,      nullptr,
    &blt_decode, &bge_decode, &bltu_decode, &bgeu_decode,
};

const DecodeFn Decoder::system_funct3_[8] = {
    &decode_field<system_priv_funct7_, 25, 7>, // 000 PRIV
    &zicsr_csrrw_decode,                       // 001
    &zicsr_csrrs_decode,                       // 010
    &zicsr_csrrc_decode,                       // 011
    nullptr,                                   // 100
    &zicsr_csrrwi_decode,                      // 101
    &zicsr_csrrsi_decode,                      // 110
    &zicsr_csrrci_decode,                      // 111
};

// Level 1: funct2 tables (4 entries each) - FP fused multiply-add

const DecodeFn Decoder::madd_fmt_[4] = {
    &f_fmadd_s_decode,
    &d_fmadd_d_decode,
    nullptr,
    nullptr,
};

const DecodeFn Decoder::msub_fmt_[4] = {
    &f_fmsub_s_decode,
    &d_fmsub_d_decode,
    nullptr,
    nullptr,
};

const DecodeFn Decoder::nmsub_fmt_[4] = {
    &f_fnmsub_s_decode,
    &d_fnmsub_d_decode,
    nullptr,
    nullptr,
};

const DecodeFn Decoder::nmadd_fmt_[4] = {
    &f_fnmadd_s_decode,
    &d_fnmadd_d_decode,
    nullptr,
    nullptr,
};

// Level 0: opcode[6:2], 32 entries

const DecodeFn Decoder::opcode_table_[32] = {
    &decode_field<load_funct3_, 12, 3>,      // [0]  LOAD
    &decode_field<load_fp_funct3_, 12, 3>,   // [1]  LOAD-FP
    &vendor0_ext_decode,                     // [2]  custom-0
    &decode_field<misc_mem_funct3_, 12, 3>,  // [3]  MISC-MEM
    &decode_field<op_imm_funct3_, 12, 3>,    // [4]  OP-IMM
    &auipc_decode,                           // [5]  AUIPC
    &decode_field<op_imm_32_funct3_, 12, 3>, // [6]  OP-IMM-32
    nullptr,                                 // [7]  48-bit prefix
    &decode_field<store_funct3_, 12, 3>,     // [8]  STORE
    &decode_field<store_fp_funct3_, 12, 3>,  // [9]  STORE-FP
    &vendor1_ext_decode,                     // [10] custom-1
    &decode_field<amo_funct3_, 12, 3>,       // [11] AMO
    &decode_field<op_funct3_, 12, 3>,        // [12] OP
    &lui_decode,                             // [13] LUI
    &decode_field<op_32_funct3_, 12, 3>,     // [14] OP-32
    nullptr,                                 // [15] 64-bit prefix
    &decode_field<madd_fmt_, 25, 2>,         // [16] MADD
    &decode_field<msub_fmt_, 25, 2>,         // [17] MSUB
    &decode_field<nmsub_fmt_, 25, 2>,        // [18] NMSUB
    &decode_field<nmadd_fmt_, 25, 2>,        // [19] NMADD
    &decode_field<op_fp_funct7_, 25, 7>,     // [20] OP-FP
    nullptr,                                 // [21] OP-V / reserved
    &vendor2_ext_decode,                     // [22] custom-2
    nullptr,                                 // [23] 48-bit prefix
    &decode_field<branch_funct3_, 12, 3>,    // [24] BRANCH
    &jalr_decode,                            // [25] JALR
    nullptr,                                 // [26] reserved
    &jal_decode,                             // [27] JAL
    &decode_field<system_funct3_, 12, 3>,    // [28] SYSTEM
    nullptr,                                 // [29] reserved
    &vendor3_ext_decode,                     // [30] custom-3
    nullptr,                                 // [31] >= 80-bit
};

} // namespace risc_v
} // namespace rocjitsu
