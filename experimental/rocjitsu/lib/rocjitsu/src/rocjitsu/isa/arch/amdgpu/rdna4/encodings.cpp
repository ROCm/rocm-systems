// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/rdna4/encodings.h"
#include <string>

namespace rocjitsu {
namespace rdna4 {

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

bool Sopk::hasImpliedLiteral() { return inst_.op == 19; }

Sop2::Sop2(const std::string &mnemonic, const Sop2MachineInst *inst)
    : IsaInstruction<Isa>(mnemonic), inst_(*inst) {
  size_ = sizeof(OpEncoding);
  if (!default_encoding() || hasImpliedLiteral())
    size_ += sizeof(MachineInst);
}

bool Sop2::default_encoding() { return inst_.ssrc0 != 255 && inst_.ssrc1 != 255; }

bool Sop2::has_lit_0() { return inst_.ssrc0 == 255 && inst_.ssrc1 != 255; }

bool Sop2::has_lit_1() { return inst_.ssrc0 != 255 && inst_.ssrc1 == 255; }

bool Sop2::has_lit_0_has_lit_1() { return inst_.ssrc0 == 255 && inst_.ssrc1 == 255; }

bool Sop2::hasImpliedLiteral() { return inst_.op == 69 || inst_.op == 70; }

Smem::Smem(const std::string &mnemonic, const SmemMachineInst *inst)
    : IsaInstruction<Isa>(mnemonic), inst_(*inst) {
  size_ = sizeof(OpEncoding);
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
  return inst_.src0 != 250 && inst_.src0 != 233 && inst_.src0 != 234 && inst_.src0 != 255;
}

bool Vop1::has_lit() { return inst_.src0 == 255; }

Vopc::Vopc(const std::string &mnemonic, const VopcMachineInst *inst)
    : IsaInstruction<Isa>(mnemonic + "_e32"), inst_(*inst) {
  size_ = sizeof(OpEncoding);
  if (!default_encoding())
    size_ += sizeof(MachineInst);
}

bool Vopc::default_encoding() {
  return inst_.src0 != 250 && inst_.src0 != 233 && inst_.src0 != 234 && inst_.src0 != 255;
}

bool Vopc::has_lit() { return inst_.src0 == 255; }

Vop2::Vop2(const std::string &mnemonic, const Vop2MachineInst *inst)
    : IsaInstruction<Isa>(mnemonic + "_e32"), inst_(*inst) {
  size_ = sizeof(OpEncoding);
  if (!default_encoding() || hasImpliedLiteral())
    size_ += sizeof(MachineInst);
}

bool Vop2::default_encoding() {
  return inst_.src0 != 250 && inst_.src0 != 233 && inst_.src0 != 234 && inst_.src0 != 255;
}

bool Vop2::has_lit() { return inst_.src0 == 255; }

bool Vop2::hasImpliedLiteral() {
  return inst_.op == 44 || inst_.op == 45 || inst_.op == 55 || inst_.op == 56;
}

Vop3::Vop3(const std::string &mnemonic, const Vop3MachineInst *inst)
    : IsaInstruction<Isa>(mnemonic), inst_(*inst) {
  size_ = sizeof(OpEncoding);
}

bool Vop3::has_lit_0() { return inst_.src0 == 255 && inst_.src1 != 255 && inst_.src2 != 255; }

bool Vop3::has_lit_1() {
  return inst_.src0 != 250 && inst_.src0 != 233 && inst_.src0 != 234 && inst_.src0 != 255 &&
         inst_.src1 == 255 && inst_.src2 != 255;
}

bool Vop3::has_lit_0_has_lit_1() {
  return inst_.src0 == 255 && inst_.src1 == 255 && inst_.src2 != 255;
}

bool Vop3::has_lit_2() {
  return inst_.src0 != 250 && inst_.src0 != 233 && inst_.src0 != 234 && inst_.src0 != 255 &&
         inst_.src1 != 255 && inst_.src2 == 255;
}

bool Vop3::has_lit_0_has_lit_2() {
  return inst_.src0 == 255 && inst_.src1 != 255 && inst_.src2 == 255;
}

bool Vop3::has_lit_1_has_lit_2() {
  return inst_.src0 != 250 && inst_.src0 != 233 && inst_.src0 != 234 && inst_.src0 != 255 &&
         inst_.src1 == 255 && inst_.src2 == 255;
}

bool Vop3::has_lit_0_has_lit_1_has_lit_2() {
  return inst_.src0 == 255 && inst_.src1 == 255 && inst_.src2 == 255;
}

Vop3p::Vop3p(const std::string &mnemonic, const Vop3pMachineInst *inst)
    : IsaInstruction<Isa>(mnemonic), inst_(*inst) {
  size_ = sizeof(OpEncoding);
}

bool Vop3p::has_lit_0() { return inst_.src0 == 255 && inst_.src1 != 255 && inst_.src2 != 255; }

bool Vop3p::has_lit_1() {
  return inst_.src0 != 250 && inst_.src0 != 233 && inst_.src0 != 234 && inst_.src0 != 255 &&
         inst_.src1 == 255 && inst_.src2 != 255;
}

bool Vop3p::has_lit_0_has_lit_1() {
  return inst_.src0 == 255 && inst_.src1 == 255 && inst_.src2 != 255;
}

bool Vop3p::has_lit_2() {
  return inst_.src0 != 250 && inst_.src0 != 233 && inst_.src0 != 234 && inst_.src0 != 255 &&
         inst_.src1 != 255 && inst_.src2 == 255;
}

bool Vop3p::has_lit_0_has_lit_2() {
  return inst_.src0 == 255 && inst_.src1 != 255 && inst_.src2 == 255;
}

bool Vop3p::has_lit_1_has_lit_2() {
  return inst_.src0 != 250 && inst_.src0 != 233 && inst_.src0 != 234 && inst_.src0 != 255 &&
         inst_.src1 == 255 && inst_.src2 == 255;
}

bool Vop3p::has_lit_0_has_lit_1_has_lit_2() {
  return inst_.src0 == 255 && inst_.src1 == 255 && inst_.src2 == 255;
}

Vinterp::Vinterp(const std::string &mnemonic, const VinterpMachineInst *inst)
    : IsaInstruction<Isa>(mnemonic), inst_(*inst) {
  size_ = sizeof(OpEncoding);
}

Vdsdir::Vdsdir(const std::string &mnemonic, const VdsdirMachineInst *inst)
    : IsaInstruction<Isa>(mnemonic), inst_(*inst) {
  size_ = sizeof(OpEncoding);
  if (!default_encoding())
    size_ += sizeof(MachineInst);
}

bool Vdsdir::default_encoding() { return true; }

Vds::Vds(const std::string &mnemonic, const VdsMachineInst *inst)
    : IsaInstruction<Isa>(mnemonic), inst_(*inst) {
  size_ = sizeof(OpEncoding);
}

Vbuffer::Vbuffer(const std::string &mnemonic, const VbufferMachineInst *inst)
    : IsaInstruction<Isa>(mnemonic), inst_(*inst) {
  size_ = sizeof(OpEncoding);
  if (inst->offen)
    modifiers_ += " offen";
  if (inst->idxen)
    modifiers_ += " idxen";
  if (inst->ioffset)
    modifiers_ += " offset:" + std::to_string(inst->ioffset);
  if (inst->nv)
    modifiers_ += " nv";
}

Vimage::Vimage(const std::string &mnemonic, const VimageMachineInst *inst)
    : IsaInstruction<Isa>(mnemonic), inst_(*inst) {
  size_ = sizeof(OpEncoding);
}

Vsample::Vsample(const std::string &mnemonic, const VsampleMachineInst *inst)
    : IsaInstruction<Isa>(mnemonic), inst_(*inst) {
  size_ = sizeof(OpEncoding);
}

Vexport::Vexport(const std::string &mnemonic, const VexportMachineInst *inst)
    : IsaInstruction<Isa>(mnemonic), inst_(*inst) {
  size_ = sizeof(OpEncoding);
}

Vflat::Vflat(const std::string &mnemonic, const VflatMachineInst *inst)
    : IsaInstruction<Isa>(mnemonic), inst_(*inst) {
  size_ = sizeof(OpEncoding);
  if (inst->nv)
    modifiers_ += " nv";
}

Vscratch::Vscratch(const std::string &mnemonic, const VscratchMachineInst *inst)
    : IsaInstruction<Isa>(mnemonic), inst_(*inst) {
  size_ = sizeof(OpEncoding);
  if (inst->nv)
    modifiers_ += " nv";
}

Vglobal::Vglobal(const std::string &mnemonic, const VglobalMachineInst *inst)
    : IsaInstruction<Isa>(mnemonic), inst_(*inst) {
  size_ = sizeof(OpEncoding);
  if (inst->nv)
    modifiers_ += " nv";
}

Vop3SdstEnc::Vop3SdstEnc(const std::string &mnemonic, const Vop3SdstEncMachineInst *inst)
    : IsaInstruction<Isa>(mnemonic), inst_(*inst) {
  size_ = sizeof(OpEncoding);
}

} // namespace rdna4
} // namespace rocjitsu
