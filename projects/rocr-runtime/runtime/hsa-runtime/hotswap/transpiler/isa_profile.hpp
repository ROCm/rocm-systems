#ifndef HOTSWAP_TRANSPILER_ISA_PROFILE_HPP
#define HOTSWAP_TRANSPILER_ISA_PROFILE_HPP

#include "llvm/ADT/StringRef.h"
#include <string>

namespace transpiler {

struct ISAProfile {
  std::string target;
  unsigned waveSize = 64;
  bool hasAGPR = true;
  bool hasMFMA = true;
  bool hasVOPD = false;
  bool hasScalarFP = false;
  bool hasWMMA = false;

  bool isWave32() const { return waveSize == 32; }

  static ISAProfile fromTarget(llvm::StringRef isa) {
    ISAProfile p;
    p.target = isa.str();
    if (isa.starts_with("gfx10") || isa.starts_with("gfx11") ||
        isa.starts_with("gfx12")) {
      p.waveSize = 32;
      p.hasAGPR = false;
      p.hasMFMA = false;
    }
    if (isa.starts_with("gfx12")) {
      p.hasVOPD = true;
      p.hasScalarFP = true;
    }
    if (isa == "gfx1250") {
      p.hasWMMA = true;
    }
    return p;
  }
};

} // namespace transpiler

#endif
