#ifndef HOTSWAP_TRANSPILER_PARSED_REG_HPP
#define HOTSWAP_TRANSPILER_PARSED_REG_HPP

namespace transpiler {

struct ISAProfile;  // forward declaration

struct ParsedReg {
  // Compact-predicate "source-only" registers (SIRegisterInfo.td:198-200):
  //   SRC_VCCZ : i1 == (VCC == 0)   read as i32 in VOP/SOP src slots
  //   SRC_EXECZ: i1 == (EXEC == 0)
  //   SRC_SCC  : i1 == SCC
  // These cannot be written. We model them as their own kinds so
  // readOp32 / readOp64 can materialise the boolean result on demand
  // (see raise_context.cpp). Tensile gfx1250 emits them as F16 source
  // operands (e.g. `v_sub_f16 v64, src_vccz, v48`), so the dispatch
  // path must recognise them or the kernel crashes inside parseReg.
  enum Kind { SGPR, VGPR, AGPR, VCC, EXEC, SCC, MODE, M0, FLAT_SCR, TTMP,
              LDS_DIRECT, SRC_VCCZ, SRC_EXECZ, SRC_SCC, NOREG, OTHER };
  Kind kind = OTHER;
  int baseIdx = -1;
  int width = 1;
};

} // namespace transpiler

#endif
