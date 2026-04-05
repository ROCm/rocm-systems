// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/cdna4/encodings.h"
#include <string>

namespace rocjitsu {
namespace cdna4 {

namespace {
std::string flat_mnemonic(const std::string &mnemonic, int seg) {
  // seg: 0=FLAT, 1=SCRATCH, 2=GLOBAL
  if (seg == 1) {
    if (mnemonic.substr(0, 5) == "flat_")
      return "scratch_" + mnemonic.substr(5);
  } else if (seg == 2) {
    if (mnemonic.substr(0, 5) == "flat_")
      return "global_" + mnemonic.substr(5);
  }
  return mnemonic;
}
} // namespace

Sop1::Sop1(const std::string &mnemonic, const Sop1MachineInst *inst)
    : IsaInstruction<Isa>(mnemonic), inst_(*inst) {
  size_ = sizeof(OpEncoding);
  if (!default_encoding())
    size_ += sizeof(MachineInst);
}

bool Sop1::default_encoding() { return inst_.ssrc0 != 255; }

bool Sop1::has_lit_0() { return inst_.ssrc0 == 255; }

Sopc::Sopc(const std::string &mnemonic, const SopcMachineInst *inst)
    : IsaInstruction<Isa>(mnemonic), inst_(*inst) {
  size_ = sizeof(OpEncoding);
  if (!default_encoding())
    size_ += sizeof(MachineInst);
}

bool Sopc::default_encoding() { return inst_.ssrc0 != 255 && inst_.ssrc1 != 255; }

bool Sopc::has_lit_0() { return inst_.ssrc0 == 255 && inst_.ssrc1 != 255; }

bool Sopc::has_lit_1() { return inst_.ssrc0 != 255 && inst_.ssrc1 == 255; }

bool Sopc::has_lit_0_has_lit_1() { return inst_.ssrc0 == 255 && inst_.ssrc1 == 255; }

Sopp::Sopp(const std::string &mnemonic, const SoppMachineInst *inst)
    : IsaInstruction<Isa>(mnemonic), inst_(*inst) {
  size_ = sizeof(OpEncoding);
}

Sopk::Sopk(const std::string &mnemonic, const SopkMachineInst *inst)
    : IsaInstruction<Isa>(mnemonic), inst_(*inst) {
  size_ = sizeof(OpEncoding);
  if (hasImpliedLiteral())
    size_ += sizeof(MachineInst);
}

bool Sopk::hasImpliedLiteral() { return inst_.op == 20; }

Sop2::Sop2(const std::string &mnemonic, const Sop2MachineInst *inst)
    : IsaInstruction<Isa>(mnemonic), inst_(*inst) {
  size_ = sizeof(OpEncoding);
  if (!default_encoding())
    size_ += sizeof(MachineInst);
}

bool Sop2::default_encoding() { return inst_.ssrc0 != 255 && inst_.ssrc1 != 255; }

bool Sop2::has_lit_0() { return inst_.ssrc0 == 255 && inst_.ssrc1 != 255; }

bool Sop2::has_lit_1() { return inst_.ssrc0 != 255 && inst_.ssrc1 == 255; }

bool Sop2::has_lit_0_has_lit_1() { return inst_.ssrc0 == 255 && inst_.ssrc1 == 255; }

Smem::Smem(const std::string &mnemonic, const SmemMachineInst *inst)
    : IsaInstruction<Isa>(mnemonic), inst_(*inst) {
  size_ = sizeof(OpEncoding);
  if (inst->soffset_en && inst->imm)
    modifiers_ += " offset:" + std::to_string(inst->offset);
  if (inst->glc)
    modifiers_ += " glc";
  if (inst->nv)
    modifiers_ += " nv";
}

Vop1::Vop1(const std::string &mnemonic, const Vop1MachineInst *inst)
    : IsaInstruction<Isa>(mnemonic + "_e32"), inst_(*inst) {
  size_ = sizeof(OpEncoding);
  if (!default_encoding())
    size_ += sizeof(MachineInst);
}

bool Vop1::default_encoding() {
  return inst_.src0 != 250 && inst_.src0 != 255 && inst_.src0 != 249;
}

bool Vop1::has_lit() { return inst_.src0 == 255; }

bool Vop1::has_dpp() { return inst_.src0 == 250; }

bool Vop1::has_sdwa() { return inst_.src0 == 249; }

Vopc::Vopc(const std::string &mnemonic, const VopcMachineInst *inst)
    : IsaInstruction<Isa>(mnemonic + "_e32"), inst_(*inst) {
  size_ = sizeof(OpEncoding);
  if (!default_encoding())
    size_ += sizeof(MachineInst);
}

bool Vopc::default_encoding() {
  return inst_.src0 != 250 && inst_.src0 != 255 && inst_.src0 != 249;
}

bool Vopc::has_lit() { return inst_.src0 == 255; }

bool Vopc::has_sdwa() { return inst_.src0 == 249; }

Vop2::Vop2(const std::string &mnemonic, const Vop2MachineInst *inst)
    : IsaInstruction<Isa>(mnemonic + "_e32"), inst_(*inst) {
  size_ = sizeof(OpEncoding);
  if (!default_encoding() || hasImpliedLiteral())
    size_ += sizeof(MachineInst);
}

bool Vop2::default_encoding() {
  return inst_.src0 != 250 && inst_.src0 != 255 && inst_.src0 != 249;
}

bool Vop2::has_lit() { return inst_.src0 == 255; }

bool Vop2::has_dpp() { return inst_.src0 == 250; }

bool Vop2::has_sdwa() { return inst_.src0 == 249; }

bool Vop2::hasImpliedLiteral() {
  return inst_.op == 23 || inst_.op == 24 || inst_.op == 36 || inst_.op == 37;
}

Vop3p::Vop3p(const std::string &mnemonic, const Vop3pMachineInst *inst)
    : IsaInstruction<Isa>(mnemonic), inst_(*inst) {
  size_ = sizeof(OpEncoding);
}

Vop3::Vop3(const std::string &mnemonic, const Vop3MachineInst *inst)
    : IsaInstruction<Isa>(mnemonic), inst_(*inst) {
  size_ = sizeof(OpEncoding);
}

Ds::Ds(const std::string &mnemonic, const DsMachineInst *inst)
    : IsaInstruction<Isa>(mnemonic), inst_(*inst) {
  size_ = sizeof(OpEncoding);
}

Mubuf::Mubuf(const std::string &mnemonic, const MubufMachineInst *inst)
    : IsaInstruction<Isa>(mnemonic), inst_(*inst) {
  size_ = sizeof(OpEncoding);
  if (inst->offen)
    modifiers_ += " offen";
  if (inst->idxen)
    modifiers_ += " idxen";
  if (inst->offset)
    modifiers_ += " offset:" + std::to_string(inst->offset);
  if (inst->sc0)
    modifiers_ += " sc0";
  if (inst->sc1)
    modifiers_ += " sc1";
  if (inst->nt)
    modifiers_ += " nt";
}

Mtbuf::Mtbuf(const std::string &mnemonic, const MtbufMachineInst *inst)
    : IsaInstruction<Isa>(mnemonic), inst_(*inst) {
  size_ = sizeof(OpEncoding);
  if (inst->offen)
    modifiers_ += " offen";
  if (inst->offset)
    modifiers_ += " offset:" + std::to_string(inst->offset);
  if (inst->sc0)
    modifiers_ += " sc0";
  if (inst->sc1)
    modifiers_ += " sc1";
  if (inst->nt)
    modifiers_ += " nt";
}

Flat::Flat(const std::string &mnemonic, const FlatMachineInst *inst)
    : IsaInstruction<Isa>(flat_mnemonic(mnemonic, inst->seg)), inst_(*inst) {
  size_ = sizeof(OpEncoding);
  int flat_offset = (inst->seg != 0) ? (inst->offset | (inst->pad_12 << 12)) : inst->offset;
  if (flat_offset)
    modifiers_ += " offset:" + std::to_string(flat_offset);
  if (inst->sc0)
    modifiers_ += " sc0";
  if (inst->sc1)
    modifiers_ += " sc1";
  if (inst->nt)
    modifiers_ += " nt";
}

Vop3SdstEnc::Vop3SdstEnc(const std::string &mnemonic, const Vop3SdstEncMachineInst *inst)
    : IsaInstruction<Isa>(mnemonic), inst_(*inst) {
  size_ = sizeof(OpEncoding);
}

Vop3pMfma::Vop3pMfma(const std::string &mnemonic, const Vop3pMfmaMachineInst *inst)
    : IsaInstruction<Isa>(mnemonic), inst_(*inst) {
  size_ = sizeof(OpEncoding);
}

} // namespace cdna4
} // namespace rocjitsu
