// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_5_ENCODINGS_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_5_ENCODINGS_H_

#include "rocjitsu/isa/arch/amdgpu/rdna3_5/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3_5/machine_insts.h"
#include "rocjitsu/isa/instruction.h"
#include <string>

namespace rocjitsu {
namespace rdna3_5 {

class Sop1 : public IsaInstruction<Isa> {
public:
  Sop1(const std::string &mnemonic, const Sop1MachineInst *inst);
  using OpEncoding = Sop1MachineInst;

protected:
  [[maybe_unused]] const OpEncoding inst_;

private:
  bool default_encoding();
  bool has_lit_0();
};

class Sopc : public IsaInstruction<Isa> {
public:
  Sopc(const std::string &mnemonic, const SopcMachineInst *inst);
  using OpEncoding = SopcMachineInst;

protected:
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

protected:
  [[maybe_unused]] const OpEncoding inst_;

private:
  bool default_encoding();
};

class Sopk : public IsaInstruction<Isa> {
public:
  Sopk(const std::string &mnemonic, const SopkMachineInst *inst);
  using OpEncoding = SopkMachineInst;

protected:
  [[maybe_unused]] const OpEncoding inst_;

private:
  bool default_encoding();
  bool hasImpliedLiteral();
};

class Sop2 : public IsaInstruction<Isa> {
public:
  Sop2(const std::string &mnemonic, const Sop2MachineInst *inst);
  using OpEncoding = Sop2MachineInst;

protected:
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

protected:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Vop1 : public IsaInstruction<Isa> {
public:
  Vop1(const std::string &mnemonic, const Vop1MachineInst *inst);
  using OpEncoding = Vop1MachineInst;

protected:
  [[maybe_unused]] const OpEncoding inst_;

private:
  bool default_encoding();
  bool has_lit();
};

class Vopc : public IsaInstruction<Isa> {
public:
  Vopc(const std::string &mnemonic, const VopcMachineInst *inst);
  using OpEncoding = VopcMachineInst;

protected:
  [[maybe_unused]] const OpEncoding inst_;

private:
  bool default_encoding();
  bool has_lit();
};

class Vop2 : public IsaInstruction<Isa> {
public:
  Vop2(const std::string &mnemonic, const Vop2MachineInst *inst);
  using OpEncoding = Vop2MachineInst;

protected:
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

protected:
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

protected:
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

protected:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Ldsdir : public IsaInstruction<Isa> {
public:
  Ldsdir(const std::string &mnemonic, const LdsdirMachineInst *inst);
  using OpEncoding = LdsdirMachineInst;

protected:
  [[maybe_unused]] const OpEncoding inst_;

private:
  bool default_encoding();
};

class Ds : public IsaInstruction<Isa> {
public:
  Ds(const std::string &mnemonic, const DsMachineInst *inst);
  using OpEncoding = DsMachineInst;

protected:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Mubuf : public IsaInstruction<Isa> {
public:
  Mubuf(const std::string &mnemonic, const MubufMachineInst *inst);
  using OpEncoding = MubufMachineInst;

protected:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Mtbuf : public IsaInstruction<Isa> {
public:
  Mtbuf(const std::string &mnemonic, const MtbufMachineInst *inst);
  using OpEncoding = MtbufMachineInst;

protected:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Mimg : public IsaInstruction<Isa> {
public:
  Mimg(const std::string &mnemonic, const MimgMachineInst *inst);
  using OpEncoding = MimgMachineInst;

protected:
  [[maybe_unused]] const OpEncoding inst_;

private:
  bool has_nsa();
};

class Exp : public IsaInstruction<Isa> {
public:
  Exp(const std::string &mnemonic, const ExpMachineInst *inst);
  using OpEncoding = ExpMachineInst;

protected:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Flat : public IsaInstruction<Isa> {
public:
  Flat(const std::string &mnemonic, const FlatMachineInst *inst);
  using OpEncoding = FlatMachineInst;

protected:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Vop3SdstEnc : public IsaInstruction<Isa> {
public:
  Vop3SdstEnc(const std::string &mnemonic, const Vop3SdstEncMachineInst *inst);
  using OpEncoding = Vop3SdstEncMachineInst;

protected:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

} // namespace rdna3_5
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_5_ENCODINGS_H_
