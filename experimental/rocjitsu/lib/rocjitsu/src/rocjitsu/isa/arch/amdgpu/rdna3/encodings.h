// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_ENCODINGS_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_ENCODINGS_H_

#include "rocjitsu/isa/arch/amdgpu/rdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/machine_insts.h"
#include "rocjitsu/isa/instruction.h"
#include <string>
#include <string_view>

namespace rocjitsu {
namespace rdna3 {

class Sop1 : public IsaInstruction<Isa> {
public:
  Sop1(std::string_view mnemonic, const Sop1MachineInst *inst);
  using OpEncoding = Sop1MachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
  bool default_encoding();
  bool has_lit_0();
};

class Sopc : public IsaInstruction<Isa> {
public:
  Sopc(std::string_view mnemonic, const SopcMachineInst *inst);
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
  Sopp(std::string_view mnemonic, const SoppMachineInst *inst);
  using OpEncoding = SoppMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
  bool default_encoding();
};

class Sopk : public IsaInstruction<Isa> {
public:
  Sopk(std::string_view mnemonic, const SopkMachineInst *inst);
  using OpEncoding = SopkMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
  bool default_encoding();
  bool hasImpliedLiteral();
};

class Sop2 : public IsaInstruction<Isa> {
public:
  Sop2(std::string_view mnemonic, const Sop2MachineInst *inst);
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
  Smem(std::string_view mnemonic, const SmemMachineInst *inst);
  using OpEncoding = SmemMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Vop1 : public IsaInstruction<Isa> {
public:
  Vop1(std::string_view mnemonic, const Vop1MachineInst *inst);
  using OpEncoding = Vop1MachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
  bool default_encoding();
  bool has_lit();
};

class Vopc : public IsaInstruction<Isa> {
public:
  Vopc(std::string_view mnemonic, const VopcMachineInst *inst);
  using OpEncoding = VopcMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
  bool default_encoding();
  bool has_lit();
};

class Vop2 : public IsaInstruction<Isa> {
public:
  Vop2(std::string_view mnemonic, const Vop2MachineInst *inst);
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
  Vop3(std::string_view mnemonic, const Vop3MachineInst *inst);
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
  Vop3p(std::string_view mnemonic, const Vop3pMachineInst *inst);
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
  Vinterp(std::string_view mnemonic, const VinterpMachineInst *inst);
  using OpEncoding = VinterpMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Ldsdir : public IsaInstruction<Isa> {
public:
  Ldsdir(std::string_view mnemonic, const LdsdirMachineInst *inst);
  using OpEncoding = LdsdirMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
  bool default_encoding();
};

class Ds : public IsaInstruction<Isa> {
public:
  Ds(std::string_view mnemonic, const DsMachineInst *inst);
  using OpEncoding = DsMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Mubuf : public IsaInstruction<Isa> {
public:
  Mubuf(std::string_view mnemonic, const MubufMachineInst *inst);
  using OpEncoding = MubufMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Mtbuf : public IsaInstruction<Isa> {
public:
  Mtbuf(std::string_view mnemonic, const MtbufMachineInst *inst);
  using OpEncoding = MtbufMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Mimg : public IsaInstruction<Isa> {
public:
  Mimg(std::string_view mnemonic, const MimgMachineInst *inst);
  using OpEncoding = MimgMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
  bool has_nsa();
};

class Exp : public IsaInstruction<Isa> {
public:
  Exp(std::string_view mnemonic, const ExpMachineInst *inst);
  using OpEncoding = ExpMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Flat : public IsaInstruction<Isa> {
public:
  Flat(std::string_view mnemonic, const FlatMachineInst *inst);
  using OpEncoding = FlatMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;
  std::string owned_mnemonic_;

private:
};

class Vop3SdstEnc : public IsaInstruction<Isa> {
public:
  Vop3SdstEnc(std::string_view mnemonic, const Vop3SdstEncMachineInst *inst);
  using OpEncoding = Vop3SdstEncMachineInst;

public:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

} // namespace rdna3
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_ENCODINGS_H_
