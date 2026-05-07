// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_RISC_V_BASE_I_H_
#define ROCJITSU_ISA_ARCH_RISC_V_BASE_I_H_

#include "rocjitsu/isa/arch/risc_v/encodings.h"
#include "rocjitsu/isa/arch/risc_v/operand.h"

#include <cstdint>

namespace rocjitsu {
namespace risc_v {
namespace detail {

// U-type instructions

class LuiInst : public UType {
public:
  explicit LuiInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand imm_op;
};

class AuipcInst : public UType {
public:
  explicit AuipcInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand imm_op;
};

// J-type instructions

class JalInst : public JType {
public:
  explicit JalInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand offset;
};

// I-type jump instructions

class JalrInst : public IType {
public:
  explicit JalrInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1;
  Operand imm_op;
};

// B-type branch instructions

class BeqInst : public BType {
public:
  explicit BeqInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rs1_op;
  Operand rs2_op;
  Operand offset;
};

class BneInst : public BType {
public:
  explicit BneInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rs1_op;
  Operand rs2_op;
  Operand offset;
};

class BltInst : public BType {
public:
  explicit BltInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rs1_op;
  Operand rs2_op;
  Operand offset;
};

class BgeInst : public BType {
public:
  explicit BgeInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rs1_op;
  Operand rs2_op;
  Operand offset;
};

class BltuInst : public BType {
public:
  explicit BltuInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rs1_op;
  Operand rs2_op;
  Operand offset;
};

class BgeuInst : public BType {
public:
  explicit BgeuInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rs1_op;
  Operand rs2_op;
  Operand offset;
};

// I-type load instructions

class LbInst : public IType {
public:
  explicit LbInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1_op;
  Operand offset;
};

class LhInst : public IType {
public:
  explicit LhInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1_op;
  Operand offset;
};

class LwInst : public IType {
public:
  explicit LwInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1_op;
  Operand offset;
};

class LdInst : public IType {
public:
  explicit LdInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1_op;
  Operand offset;
};

class LbuInst : public IType {
public:
  explicit LbuInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1_op;
  Operand offset;
};

class LhuInst : public IType {
public:
  explicit LhuInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1_op;
  Operand offset;
};

class LwuInst : public IType {
public:
  explicit LwuInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1_op;
  Operand offset;
};

// S-type store instructions

class SbInst : public SType {
public:
  explicit SbInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rs2_op;
  Operand rs1_op;
  Operand offset;
};

class ShInst : public SType {
public:
  explicit ShInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rs2_op;
  Operand rs1_op;
  Operand offset;
};

class SwInst : public SType {
public:
  explicit SwInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rs2_op;
  Operand rs1_op;
  Operand offset;
};

class SdInst : public SType {
public:
  explicit SdInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rs2_op;
  Operand rs1_op;
  Operand offset;
};

// I-type ALU instructions

class AddiInst : public IType {
public:
  explicit AddiInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1;
  Operand imm_op;
};

class SltiInst : public IType {
public:
  explicit SltiInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1;
  Operand imm_op;
};

class SltiuInst : public IType {
public:
  explicit SltiuInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1;
  Operand imm_op;
};

class XoriInst : public IType {
public:
  explicit XoriInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1;
  Operand imm_op;
};

class OriInst : public IType {
public:
  explicit OriInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1;
  Operand imm_op;
};

class AndiInst : public IType {
public:
  explicit AndiInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1;
  Operand imm_op;
};

// I-type shift instructions

class SlliInst : public IType {
public:
  explicit SlliInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1;
  Operand imm_op;
};

class SrliInst : public IType {
public:
  explicit SrliInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1;
  Operand imm_op;
};

class SraiInst : public IType {
public:
  explicit SraiInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1;
  Operand imm_op;
};

// R-type ALU instructions

class AddInst : public RType {
public:
  explicit AddInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

class SubInst : public RType {
public:
  explicit SubInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

class SllInst : public RType {
public:
  explicit SllInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

class SltInst : public RType {
public:
  explicit SltInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

class SltuInst : public RType {
public:
  explicit SltuInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

class XorInst : public RType {
public:
  explicit XorInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

class SrlInst : public RType {
public:
  explicit SrlInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

class SraInst : public RType {
public:
  explicit SraInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

class OrInst : public RType {
public:
  explicit OrInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

class AndInst : public RType {
public:
  explicit AndInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

// I-type W-ALU instructions (RV64I)

class AddiwInst : public IType {
public:
  explicit AddiwInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1;
  Operand imm_op;
};

class SlliwInst : public IType {
public:
  explicit SlliwInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1;
  Operand imm_op;
};

class SrliwInst : public IType {
public:
  explicit SrliwInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1;
  Operand imm_op;
};

class SraiwInst : public IType {
public:
  explicit SraiwInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1;
  Operand imm_op;
};

// R-type W-ALU instructions (RV64I)

class AddwInst : public RType {
public:
  explicit AddwInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

class SubwInst : public RType {
public:
  explicit SubwInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

class SllwInst : public RType {
public:
  explicit SllwInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

class SrlwInst : public RType {
public:
  explicit SrlwInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

class SrawInst : public RType {
public:
  explicit SrawInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

// I-type system instructions

class FenceInst : public IType {
public:
  explicit FenceInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand imm_op;
};

// I-type environment instructions

class EcallInst : public IType {
public:
  explicit EcallInst(uint32_t raw);
  void execute_impl(HartState &ctx);
};

class EbreakInst : public IType {
public:
  explicit EbreakInst(uint32_t raw);
  void execute_impl(HartState &ctx);
};

} // namespace detail
} // namespace risc_v
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_RISC_V_BASE_I_H_
