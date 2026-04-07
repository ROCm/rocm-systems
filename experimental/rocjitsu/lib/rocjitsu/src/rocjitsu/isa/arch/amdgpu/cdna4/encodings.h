// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_CDNA4_ENCODINGS_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_CDNA4_ENCODINGS_H_

#include "rocjitsu/isa/arch/amdgpu/cdna4/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/machine_insts.h"
#include "rocjitsu/isa/instruction.h"
#include <string>
#include <string_view>

namespace rocjitsu {
namespace cdna4 {

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
};

class Sopk : public IsaInstruction<Isa> {
public:
  Sopk(std::string_view mnemonic, const SopkMachineInst *inst, ExecuteFn exec_fn);
  using OpEncoding = SopkMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
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
  bool has_dpp();
  bool has_sdwa();
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
  bool has_sdwa();
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
  bool has_dpp();
  bool has_sdwa();
  bool hasImpliedLiteral();
};

class Vop3p : public IsaInstruction<Isa> {
public:
  Vop3p(std::string_view mnemonic, const Vop3pMachineInst *inst, ExecuteFn exec_fn);
  using OpEncoding = Vop3pMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Vop3 : public IsaInstruction<Isa> {
public:
  Vop3(std::string_view mnemonic, const Vop3MachineInst *inst, ExecuteFn exec_fn);
  using OpEncoding = Vop3MachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Ds : public IsaInstruction<Isa> {
public:
  Ds(std::string_view mnemonic, const DsMachineInst *inst, ExecuteFn exec_fn);
  using OpEncoding = DsMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Mubuf : public IsaInstruction<Isa> {
public:
  Mubuf(std::string_view mnemonic, const MubufMachineInst *inst, ExecuteFn exec_fn);
  void build_modifiers(std::string &out) const override;
  using OpEncoding = MubufMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Mtbuf : public IsaInstruction<Isa> {
public:
  Mtbuf(std::string_view mnemonic, const MtbufMachineInst *inst, ExecuteFn exec_fn);
  void build_modifiers(std::string &out) const override;
  using OpEncoding = MtbufMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Flat : public IsaInstruction<Isa> {
public:
  Flat(std::string_view mnemonic, const FlatMachineInst *inst, ExecuteFn exec_fn);
  void build_modifiers(std::string &out) const override;
  using OpEncoding = FlatMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;
  std::string owned_mnemonic_;

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

class Vop3pMfma : public IsaInstruction<Isa> {
public:
  Vop3pMfma(std::string_view mnemonic, const Vop3pMfmaMachineInst *inst, ExecuteFn exec_fn);
  using OpEncoding = Vop3pMfmaMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

} // namespace cdna4
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_CDNA4_ENCODINGS_H_
