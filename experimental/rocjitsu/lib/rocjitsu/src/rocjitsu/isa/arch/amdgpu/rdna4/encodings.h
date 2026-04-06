// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_RDNA4_ENCODINGS_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_RDNA4_ENCODINGS_H_

#include "rocjitsu/isa/arch/amdgpu/rdna4/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/machine_insts.h"
#include "rocjitsu/isa/instruction.h"
#include <string>

namespace rocjitsu {
namespace rdna4 {

class Sop1 : public IsaInstruction<Isa> {
public:
  Sop1(const std::string &mnemonic, const Sop1MachineInst *inst);
  using OpEncoding = Sop1MachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
  bool default_encoding();
  bool has_lit_0();
};

class Sopc : public IsaInstruction<Isa> {
public:
  Sopc(const std::string &mnemonic, const SopcMachineInst *inst);
  using OpEncoding = SopcMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
  bool default_encoding();
  bool has_lit_0();
  bool has_lit_1();
  bool has_lit_0_has_lit_1();
};

class Sopp : public IsaInstruction<Isa> {
public:
  Sopp(const std::string &mnemonic, const SoppMachineInst *inst);
  using OpEncoding = SoppMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
  bool default_encoding();
};

class Sopk : public IsaInstruction<Isa> {
public:
  Sopk(const std::string &mnemonic, const SopkMachineInst *inst);
  using OpEncoding = SopkMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
  bool default_encoding();
  bool hasImpliedLiteral();
};

class Sop2 : public IsaInstruction<Isa> {
public:
  Sop2(const std::string &mnemonic, const Sop2MachineInst *inst);
  using OpEncoding = Sop2MachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
  bool default_encoding();
  bool has_lit_0();
  bool has_lit_1();
  bool has_lit_0_has_lit_1();
  bool hasImpliedLiteral();
};

class Smem : public IsaInstruction<Isa> {
public:
  Smem(const std::string &mnemonic, const SmemMachineInst *inst);
  using OpEncoding = SmemMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Vop1 : public IsaInstruction<Isa> {
public:
  Vop1(const std::string &mnemonic, const Vop1MachineInst *inst);
  using OpEncoding = Vop1MachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
  bool default_encoding();
  bool has_lit();
};

class Vopc : public IsaInstruction<Isa> {
public:
  Vopc(const std::string &mnemonic, const VopcMachineInst *inst);
  using OpEncoding = VopcMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
  bool default_encoding();
  bool has_lit();
};

class Vop2 : public IsaInstruction<Isa> {
public:
  Vop2(const std::string &mnemonic, const Vop2MachineInst *inst);
  using OpEncoding = Vop2MachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
  bool default_encoding();
  bool has_lit();
  bool hasImpliedLiteral();
};

class Vop3 : public IsaInstruction<Isa> {
public:
  Vop3(const std::string &mnemonic, const Vop3MachineInst *inst);
  using OpEncoding = Vop3MachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
  bool has_lit_0();
  bool has_lit_1();
  bool has_lit_0_has_lit_1();
  bool has_lit_2();
  bool has_lit_0_has_lit_2();
  bool has_lit_1_has_lit_2();
  bool has_lit_0_has_lit_1_has_lit_2();
};

class Vop3p : public IsaInstruction<Isa> {
public:
  Vop3p(const std::string &mnemonic, const Vop3pMachineInst *inst);
  using OpEncoding = Vop3pMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
  bool has_lit_0();
  bool has_lit_1();
  bool has_lit_0_has_lit_1();
  bool has_lit_2();
  bool has_lit_0_has_lit_2();
  bool has_lit_1_has_lit_2();
  bool has_lit_0_has_lit_1_has_lit_2();
};

class Vinterp : public IsaInstruction<Isa> {
public:
  Vinterp(const std::string &mnemonic, const VinterpMachineInst *inst);
  using OpEncoding = VinterpMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Vdsdir : public IsaInstruction<Isa> {
public:
  Vdsdir(const std::string &mnemonic, const VdsdirMachineInst *inst);
  using OpEncoding = VdsdirMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
  bool default_encoding();
};

class Vds : public IsaInstruction<Isa> {
public:
  Vds(const std::string &mnemonic, const VdsMachineInst *inst);
  using OpEncoding = VdsMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Vbuffer : public IsaInstruction<Isa> {
public:
  Vbuffer(const std::string &mnemonic, const VbufferMachineInst *inst);
  using OpEncoding = VbufferMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Vimage : public IsaInstruction<Isa> {
public:
  Vimage(const std::string &mnemonic, const VimageMachineInst *inst);
  using OpEncoding = VimageMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Vsample : public IsaInstruction<Isa> {
public:
  Vsample(const std::string &mnemonic, const VsampleMachineInst *inst);
  using OpEncoding = VsampleMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Vexport : public IsaInstruction<Isa> {
public:
  Vexport(const std::string &mnemonic, const VexportMachineInst *inst);
  using OpEncoding = VexportMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Vflat : public IsaInstruction<Isa> {
public:
  Vflat(const std::string &mnemonic, const VflatMachineInst *inst);
  using OpEncoding = VflatMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Vscratch : public IsaInstruction<Isa> {
public:
  Vscratch(const std::string &mnemonic, const VscratchMachineInst *inst);
  using OpEncoding = VscratchMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Vglobal : public IsaInstruction<Isa> {
public:
  Vglobal(const std::string &mnemonic, const VglobalMachineInst *inst);
  using OpEncoding = VglobalMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Vop3SdstEnc : public IsaInstruction<Isa> {
public:
  Vop3SdstEnc(const std::string &mnemonic, const Vop3SdstEncMachineInst *inst);
  using OpEncoding = Vop3SdstEncMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

} // namespace rdna4
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_RDNA4_ENCODINGS_H_
