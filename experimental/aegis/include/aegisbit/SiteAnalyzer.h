//===-- aegisbit/SiteAnalyzer.h - Instrumentation Site Analysis ----*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Discovers VMEM and LDS memory instruction sites in a kernel's CFG and
/// computes per-site pre-spill drain values for AccVGPR spill safety.
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_SITE_ANALYZER_H
#define AEGISBIT_SITE_ANALYZER_H

#include "aegisbit/Types.h"
#include <cstdint>
#include <vector>

namespace aegisbit {

class Disassembler;

namespace SiteAnalyzer {

/// Find all memory sites (global_load/store, buffer_load/store, DS_READ/WRITE)
/// in a kernel's CFG. Pure disassembly walk -- no liveness or register analysis.
/// Extracts the per-lane address/offset VGPR from each instruction.
///
/// \param SupportsGPUAtomics If true, LDS sites are instrumented even on
///   AccVGPR kernels (the payload uses fire-and-forget atomics). Otherwise
///   LDS sites are skipped when AccVGPR spill is active.
std::vector<InstrumentationSite>
findMemorySites(const ControlFlowGraph &CFG, uint64_t BaseAddr,
                Disassembler &Disasm,
                const ScratchRegisters &Scratch = ScratchRegisters{},
                bool SupportsGPUAtomics = false);

/// For each site, walk backward through its basic block to compute the
/// minimum s_waitcnt vmcnt / lgkmcnt needed to drain only those in-flight
/// loads that target the victim VGPRs. Stores results in each site's
/// PreSpillVmWait / PreSpillLgkmWait fields.
void computePreSpillDrainValues(
    const ControlFlowGraph &CFG,
    std::vector<InstrumentationSite> &Sites,
    const ScratchRegisters &Scratch,
    Disassembler &Disasm);

} // namespace SiteAnalyzer

} // namespace aegisbit

#endif // AEGISBIT_SITE_ANALYZER_H
