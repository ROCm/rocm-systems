// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/rdna1/encodings.h"
#include <string>

namespace rocjitsu {
namespace rdna1 {

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
  if (!default_encoding())
    size_ += sizeof(MachineInst);
}

bool Sopp::default_encoding() { return true; }

Sopk::Sopk(const std::string &mnemonic, const SopkMachineInst *inst)
    : IsaInstruction<Isa>(mnemonic), inst_(*inst) {
  size_ = sizeof(OpEncoding);
  if (!default_encoding() || hasImpliedLiteral())
    size_ += sizeof(MachineInst);
}

bool Sopk::default_encoding() { return true; }

bool Sopk::hasImpliedLiteral() { return inst_.op == 21; }

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
  if (inst->glc)
    modifiers_ += " glc";
  if (inst->dlc)
    modifiers_ += " dlc";
}

Vop1::Vop1(const std::string &mnemonic, const Vop1MachineInst *inst)
    : IsaInstruction<Isa>(mnemonic + "_e32"), inst_(*inst) {
  size_ = sizeof(OpEncoding);
  if (!default_encoding())
    size_ += sizeof(MachineInst);
}

bool Vop1::default_encoding() {
  return inst_.src0 != 250 && inst_.src0 != 233 && inst_.src0 != 234 && inst_.src0 != 255 &&
         inst_.src0 != 249;
}

bool Vop1::has_lit() { return inst_.src0 == 255; }

Vopc::Vopc(const std::string &mnemonic, const VopcMachineInst *inst)
    : IsaInstruction<Isa>(mnemonic + "_e32"), inst_(*inst) {
  size_ = sizeof(OpEncoding);
  if (!default_encoding())
    size_ += sizeof(MachineInst);
}

bool Vopc::default_encoding() {
  return inst_.src0 != 250 && inst_.src0 != 233 && inst_.src0 != 234 && inst_.src0 != 255 &&
         inst_.src0 != 249;
}

bool Vopc::has_lit() { return inst_.src0 == 255; }

Vop2::Vop2(const std::string &mnemonic, const Vop2MachineInst *inst)
    : IsaInstruction<Isa>(mnemonic + "_e32"), inst_(*inst) {
  size_ = sizeof(OpEncoding);
  if (!default_encoding() || hasImpliedLiteral())
    size_ += sizeof(MachineInst);
}

bool Vop2::default_encoding() {
  return inst_.src0 != 250 && inst_.src0 != 233 && inst_.src0 != 234 && inst_.src0 != 255 &&
         inst_.src0 != 249;
}

bool Vop2::has_lit() { return inst_.src0 == 255; }

bool Vop2::hasImpliedLiteral() {
  return inst_.op == 32 || inst_.op == 33 || inst_.op == 44 || inst_.op == 45 || inst_.op == 55 ||
         inst_.op == 56;
}

Vintrp::Vintrp(const std::string &mnemonic, const VintrpMachineInst *inst)
    : IsaInstruction<Isa>(mnemonic), inst_(*inst) {
  size_ = sizeof(OpEncoding);
  if (!default_encoding())
    size_ += sizeof(MachineInst);
}

bool Vintrp::default_encoding() { return true; }

Vop3::Vop3(const std::string &mnemonic, const Vop3MachineInst *inst)
    : IsaInstruction<Isa>(mnemonic), inst_(*inst) {
  size_ = sizeof(OpEncoding);
}

bool Vop3::has_lit_0() { return inst_.src0 == 255 && inst_.src1 != 255 && inst_.src2 != 255; }

bool Vop3::has_lit_1() { return inst_.src0 != 255 && inst_.src1 == 255 && inst_.src2 != 255; }

bool Vop3::has_lit_0_has_lit_1() {
  return inst_.src0 == 255 && inst_.src1 == 255 && inst_.src2 != 255;
}

bool Vop3::has_lit_2() { return inst_.src0 != 255 && inst_.src1 != 255 && inst_.src2 == 255; }

bool Vop3::has_lit_0_has_lit_2() {
  return inst_.src0 == 255 && inst_.src1 != 255 && inst_.src2 == 255;
}

bool Vop3::has_lit_1_has_lit_2() {
  return inst_.src0 != 255 && inst_.src1 == 255 && inst_.src2 == 255;
}

bool Vop3::has_lit_0_has_lit_1_has_lit_2() {
  return inst_.src0 == 255 && inst_.src1 == 255 && inst_.src2 == 255;
}

Vop3p::Vop3p(const std::string &mnemonic, const Vop3pMachineInst *inst)
    : IsaInstruction<Isa>(mnemonic), inst_(*inst) {
  size_ = sizeof(OpEncoding);
}

bool Vop3p::has_lit_0() { return inst_.src0 == 255 && inst_.src1 != 255 && inst_.src2 != 255; }

bool Vop3p::has_lit_1() { return inst_.src0 != 255 && inst_.src1 == 255 && inst_.src2 != 255; }

bool Vop3p::has_lit_0_has_lit_1() {
  return inst_.src0 == 255 && inst_.src1 == 255 && inst_.src2 != 255;
}

bool Vop3p::has_lit_2() { return inst_.src0 != 255 && inst_.src1 != 255 && inst_.src2 == 255; }

bool Vop3p::has_lit_0_has_lit_2() {
  return inst_.src0 == 255 && inst_.src1 != 255 && inst_.src2 == 255;
}

bool Vop3p::has_lit_1_has_lit_2() {
  return inst_.src0 != 255 && inst_.src1 == 255 && inst_.src2 == 255;
}

bool Vop3p::has_lit_0_has_lit_1_has_lit_2() {
  return inst_.src0 == 255 && inst_.src1 == 255 && inst_.src2 == 255;
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
  if (inst->glc)
    modifiers_ += " glc";
  if (inst->dlc)
    modifiers_ += " dlc";
  if (inst->slc)
    modifiers_ += " slc";
}

Mtbuf::Mtbuf(const std::string &mnemonic, const MtbufMachineInst *inst)
    : IsaInstruction<Isa>(mnemonic), inst_(*inst) {
  size_ = sizeof(OpEncoding);
  if (inst->offen)
    modifiers_ += " offen";
  if (inst->offset)
    modifiers_ += " offset:" + std::to_string(inst->offset);
  if (inst->glc)
    modifiers_ += " glc";
  if (inst->dlc)
    modifiers_ += " dlc";
  if (inst->slc)
    modifiers_ += " slc";
}

Mimg::Mimg(const std::string &mnemonic, const MimgMachineInst *inst)
    : IsaInstruction<Isa>(mnemonic), inst_(*inst) {
  size_ = sizeof(OpEncoding);
}

bool Mimg::has_nsa_1() { return inst_.nsa == 1; }

bool Mimg::has_nsa_2() { return inst_.nsa == 2; }

bool Mimg::has_nsa_3() { return inst_.nsa == 3; }

Exp::Exp(const std::string &mnemonic, const ExpMachineInst *inst)
    : IsaInstruction<Isa>(mnemonic), inst_(*inst) {
  size_ = sizeof(OpEncoding);
}

Flat::Flat(const std::string &mnemonic, const FlatMachineInst *inst)
    : IsaInstruction<Isa>(flat_mnemonic(mnemonic, inst->seg)), inst_(*inst) {
  size_ = sizeof(OpEncoding);
  if (inst->offset)
    modifiers_ += " offset:" + std::to_string(inst->offset);
  if (inst->glc)
    modifiers_ += " glc";
  if (inst->dlc)
    modifiers_ += " dlc";
  if (inst->slc)
    modifiers_ += " slc";
}

Vop3SdstEnc::Vop3SdstEnc(const std::string &mnemonic, const Vop3SdstEncMachineInst *inst)
    : IsaInstruction<Isa>(mnemonic), inst_(*inst) {
  size_ = sizeof(OpEncoding);
}

} // namespace rdna1
} // namespace rocjitsu
