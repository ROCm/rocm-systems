#ifndef HOTSWAP_TRANSPILER_SETPC_ANALYSIS_HPP
#define HOTSWAP_TRANSPILER_SETPC_ANALYSIS_HPP

#include "decoded_inst.hpp"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <map>
#include <set>
#include <string>

namespace transpiler {

struct MCState;

// Static analysis pass that classifies every `s_set_pc_i64` site in a
// decoded kernel. See semop.hpp's S_SET_PC_I64 doc for the two
// principled lowering shapes (Pattern A direct, Pattern B indirectbr).
//
// The analysis is conservative: any SGPR pair whose value path it
// cannot follow drops out of the symbolic-PC table immediately, so
// downstream `s_set_pc_i64` sites that read a dropped pair are
// classified Unresolvable rather than guessed. The handler refuses
// loudly on Unresolvable — never silently emit a stub branch.

struct SetPcSiteInfo {
  enum class Kind { DirectA, IndirectB, Unresolvable };
  Kind kind = Kind::Unresolvable;
  // Pattern A: the absolute kernel offset to branch to. The lowering
  // emits `br label %BB_<directTarget>`.
  uint64_t directTarget = 0;
  // Pattern B: the enumerated set of possible return targets (each is
  // the absolute kernel offset of a basic block following one of the
  // identified call sites). The lowering emits
  // `indirectbr ptr %ret_pc, [list]`. Order is deterministic
  // (ascending) so lit fixtures can pin shape.
  llvm::SmallVector<uint64_t, 4> indirectTargets;
  // Pattern B: the SGPR low index of the source pair (for diagnostics
  // and so the handler can read the right pair).
  unsigned indirectRetPairLowReg = 0;
  // Unresolvable: human-readable reason for the refusal diagnostic.
  std::string refusalReason;
};

struct SetPcCallSiteInfo {
  // Absolute kernel offset of the instruction immediately following
  // the call-site `s_branch` (i.e. the return target the call site
  // expected). The Pattern B lowering will list this offset's BB as
  // one of the indirectbr targets.
  uint64_t resolvedReturnAddr = 0;
  // SGPR low index of the ret-pair this call site populates. Pattern
  // B `s_set_pc_i64 sX:Y` enumerates call sites whose
  // retPairLowReg == X.
  unsigned retPairLowReg = 0;
};

struct SetPcAnalysis {
  // Keyed by the s_set_pc_i64 instruction's absolute kernel offset.
  std::map<uint64_t, SetPcSiteInfo> setpcSites;

  // Chain-terminator hooks: keyed by the absolute offset of the
  // s_add_co_ci_u32 (high-half add) that completes a call-site PC
  // chain. The raiser runs the SOP2 handler normally for that
  // instruction and then overwrites the ret-pair SGPR with
  // `blockaddress(@kernel, %BB_<resolvedReturnAddr>)` (cast to i64),
  // so the i64 value carried in the ret-pair across the call is a
  // real LLVM blockaddress constant rather than a binary PC. This
  // makes the downstream `indirectbr` target enumeration sound.
  std::map<uint64_t, SetPcCallSiteInfo> chainTerminators;

  // Block-start offsets newly discovered by this analysis: Pattern A
  // direct targets + Pattern B return targets. Caller merges these
  // into the overall blockStarts BEFORE basic-block layout.
  std::set<uint64_t> extraBlockStarts;
};

// Run the analysis. `blockStarts` is the set of leaders the decoder
// already discovered (used to reset the per-block symbolic-PC table
// at every BB boundary). `mc` provides MCRegisterInfo for SGPR
// classification.
SetPcAnalysis analyseSetPC(llvm::ArrayRef<DecodedInst> insts,
                           const std::set<uint64_t> &blockStarts,
                           const MCState &mc);

} // namespace transpiler

#endif
