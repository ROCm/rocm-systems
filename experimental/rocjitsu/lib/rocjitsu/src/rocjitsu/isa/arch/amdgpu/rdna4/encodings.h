// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_RDNA4_ENCODINGS_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_RDNA4_ENCODINGS_H_

#include "rocjitsu/isa/arch/amdgpu/rdna4/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/machine_insts.h"
#include "rocjitsu/isa/instruction.h"
#include <string>
#include <string_view>

namespace rocjitsu {
namespace rdna4 {

class Sop1 : public IsaInstruction<Isa> {
public:
  Sop1(std::string_view mnemonic, const Sop1MachineInst *inst, ExecuteFn exec_fn);
  using OpEncoding = Sop1MachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
  bool default_encoding();
  bool has_lit_0();
};

class Sopc : public IsaInstruction<Isa> {
public:
  Sopc(std::string_view mnemonic, const SopcMachineInst *inst, ExecuteFn exec_fn);
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
  Sopp(std::string_view mnemonic, const SoppMachineInst *inst, ExecuteFn exec_fn);
  using OpEncoding = SoppMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
  bool default_encoding();
};

class Sopk : public IsaInstruction<Isa> {
public:
  Sopk(std::string_view mnemonic, const SopkMachineInst *inst, ExecuteFn exec_fn);
  using OpEncoding = SopkMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
  bool default_encoding();
  bool hasImpliedLiteral();
};

class Sop2 : public IsaInstruction<Isa> {
public:
  Sop2(std::string_view mnemonic, const Sop2MachineInst *inst, ExecuteFn exec_fn);
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
  Smem(std::string_view mnemonic, const SmemMachineInst *inst, ExecuteFn exec_fn);
  void build_modifiers(std::string &out) const override;
  using OpEncoding = SmemMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Vop1 : public IsaInstruction<Isa> {
public:
  Vop1(std::string_view mnemonic, const Vop1MachineInst *inst, ExecuteFn exec_fn);
  using OpEncoding = Vop1MachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
  bool default_encoding();
  bool has_lit();
};

class Vopc : public IsaInstruction<Isa> {
public:
  Vopc(std::string_view mnemonic, const VopcMachineInst *inst, ExecuteFn exec_fn);
  using OpEncoding = VopcMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
  bool default_encoding();
  bool has_lit();
};

class Vop2 : public IsaInstruction<Isa> {
public:
  Vop2(std::string_view mnemonic, const Vop2MachineInst *inst, ExecuteFn exec_fn);
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
  Vop3(std::string_view mnemonic, const Vop3MachineInst *inst, ExecuteFn exec_fn);
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
  Vop3p(std::string_view mnemonic, const Vop3pMachineInst *inst, ExecuteFn exec_fn);
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
  Vinterp(std::string_view mnemonic, const VinterpMachineInst *inst, ExecuteFn exec_fn);
  using OpEncoding = VinterpMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Vdsdir : public IsaInstruction<Isa> {
public:
  Vdsdir(std::string_view mnemonic, const VdsdirMachineInst *inst, ExecuteFn exec_fn);
  using OpEncoding = VdsdirMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
  bool default_encoding();
};

class Vds : public IsaInstruction<Isa> {
public:
  Vds(std::string_view mnemonic, const VdsMachineInst *inst, ExecuteFn exec_fn);
  using OpEncoding = VdsMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Vbuffer : public IsaInstruction<Isa> {
public:
  Vbuffer(std::string_view mnemonic, const VbufferMachineInst *inst, ExecuteFn exec_fn);
  void build_modifiers(std::string &out) const override;
  using OpEncoding = VbufferMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Vimage : public IsaInstruction<Isa> {
public:
  Vimage(std::string_view mnemonic, const VimageMachineInst *inst, ExecuteFn exec_fn);
  using OpEncoding = VimageMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Vsample : public IsaInstruction<Isa> {
public:
  Vsample(std::string_view mnemonic, const VsampleMachineInst *inst, ExecuteFn exec_fn);
  using OpEncoding = VsampleMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Vexport : public IsaInstruction<Isa> {
public:
  Vexport(std::string_view mnemonic, const VexportMachineInst *inst, ExecuteFn exec_fn);
  using OpEncoding = VexportMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Vflat : public IsaInstruction<Isa> {
public:
  Vflat(std::string_view mnemonic, const VflatMachineInst *inst, ExecuteFn exec_fn);
  void build_modifiers(std::string &out) const override;
  using OpEncoding = VflatMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Vscratch : public IsaInstruction<Isa> {
public:
  Vscratch(std::string_view mnemonic, const VscratchMachineInst *inst, ExecuteFn exec_fn);
  void build_modifiers(std::string &out) const override;
  using OpEncoding = VscratchMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Vglobal : public IsaInstruction<Isa> {
public:
  Vglobal(std::string_view mnemonic, const VglobalMachineInst *inst, ExecuteFn exec_fn);
  void build_modifiers(std::string &out) const override;
  using OpEncoding = VglobalMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Vop3SdstEnc : public IsaInstruction<Isa> {
public:
  Vop3SdstEnc(std::string_view mnemonic, const Vop3SdstEncMachineInst *inst, ExecuteFn exec_fn);
  using OpEncoding = Vop3SdstEncMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

} // namespace rdna4
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_RDNA4_ENCODINGS_H_
