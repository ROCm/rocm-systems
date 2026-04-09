// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_CDNA3_ENCODINGS_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_CDNA3_ENCODINGS_H_

#include "rocjitsu/isa/arch/amdgpu/cdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/machine_insts.h"
#include "rocjitsu/isa/instruction.h"
#include <string>

namespace rocjitsu {
namespace cdna3 {

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
  bool has_dpp();
  bool has_sdwa();
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
  bool has_sdwa();
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
  bool has_dpp();
  bool has_sdwa();
  bool hasImpliedLiteral();
};

class Vop3p : public IsaInstruction<Isa> {
public:
  Vop3p(const std::string &mnemonic, const Vop3pMachineInst *inst);
  using OpEncoding = Vop3pMachineInst;

protected:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

class Vop3 : public IsaInstruction<Isa> {
public:
  Vop3(const std::string &mnemonic, const Vop3MachineInst *inst);
  using OpEncoding = Vop3MachineInst;

protected:
  [[maybe_unused]] const OpEncoding inst_;

private:
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

class Vop3pMfma : public IsaInstruction<Isa> {
public:
  Vop3pMfma(const std::string &mnemonic, const Vop3pMfmaMachineInst *inst);
  using OpEncoding = Vop3pMfmaMachineInst;

protected:
  [[maybe_unused]] const OpEncoding inst_;

private:
};

} // namespace cdna3
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_CDNA3_ENCODINGS_H_
