#ifndef HOTSWAP_TRANSPILER_ISA_PROFILE_HPP
#define HOTSWAP_TRANSPILER_ISA_PROFILE_HPP

#include "MCTargetDesc/AMDGPUMCTargetDesc.h" // AMDGPU::Feature* enum
#include "Utils/AMDGPUBaseInfo.h"            // AMDGPU::hasMAIInsts
#include "llvm/MC/MCSubtargetInfo.h"

namespace transpiler {

// Snapshot of the capability bits the raiser actually branches on. Every
// field is derived directly from the MC subtarget feature bits that TableGen
// emits, so adding a new AMDGPU generation does not require touching this
// struct; we just read the already-defined FeatureFoo bit.
//
// This is a pure value snapshot — the factory copies bits out of the
// MCSubtargetInfo and does not retain any reference to it. Callers must
// construct via `fromSubtarget`; there is intentionally no default ctor.
struct ISAProfile {
  unsigned waveSize = 64;
  bool hasAGPR = false;
  bool hasMFMA = false;
  bool hasVOPD = false;
  bool hasScalarFP = false;
  // True iff the subtarget exposes the gfx12-era WMMA instructions
  // (FeatureWMMA{128,256}bInsts). gfx11 WMMA is encoded via FeatureGFX11Insts
  // + VOP3P patterns and is not covered here; the only WMMA source we lift
  // today is gfx1250.
  bool hasWMMA12 = false;

  bool isWave32() const { return waveSize == 32; }

  static ISAProfile fromSubtarget(const llvm::MCSubtargetInfo &STI) {
    ISAProfile p;
    p.waveSize = STI.hasFeature(llvm::AMDGPU::FeatureWavefrontSize32) ? 32 : 64;
    // AGPRs/MFMA share the mai-insts feature today; keep them as separate
    // fields so future divergence stays expressible without touching callers.
    p.hasMFMA = llvm::AMDGPU::hasMAIInsts(STI);
    p.hasAGPR = p.hasMFMA;
    p.hasVOPD = STI.hasFeature(llvm::AMDGPU::FeatureVOPDInsts);
    p.hasScalarFP = STI.hasFeature(llvm::AMDGPU::FeatureSALUFloatInsts);
    p.hasWMMA12 = STI.hasFeature(llvm::AMDGPU::FeatureWMMA128bInsts) ||
                  STI.hasFeature(llvm::AMDGPU::FeatureWMMA256bInsts);
    return p;
  }

 private:
  ISAProfile() = default; // constructible only via fromSubtarget()
};

} // namespace transpiler

#endif
