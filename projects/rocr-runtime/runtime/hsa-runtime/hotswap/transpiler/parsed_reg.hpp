#ifndef HOTSWAP_TRANSPILER_PARSED_REG_HPP
#define HOTSWAP_TRANSPILER_PARSED_REG_HPP

namespace transpiler {

struct ISAProfile;  // forward declaration

struct ParsedReg {
  enum Kind { SGPR, VGPR, AGPR, VCC, EXEC, SCC, MODE, M0, FLAT_SCR, TTMP, LDS_DIRECT, NOREG, OTHER };
  Kind kind = OTHER;
  int baseIdx = -1;
  int width = 1;
};

} // namespace transpiler

#endif
