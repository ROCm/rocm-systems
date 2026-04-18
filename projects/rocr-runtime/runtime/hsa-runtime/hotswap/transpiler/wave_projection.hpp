#ifndef HOTSWAP_TRANSPILER_WAVE_PROJECTION_HPP
#define HOTSWAP_TRANSPILER_WAVE_PROJECTION_HPP

#include "decoded_inst.hpp"
#include "isa_profile.hpp"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

namespace transpiler {

struct MCState;

// ============================================================================
// WaveProjection — the cross-wave translation policy surface.
//
// A *projection* maps a source-ISA wavefront onto a target-ISA wavefront
// when the two wave widths differ. This is an abstract base; see
// `ModuloReplicationProjection` below for the sole concrete policy in
// use today. SPE_DESIGN.md §7 catalogues the alternatives
// (thread-loop, scalarisation, half-wave-masking) that are not yet
// implemented but whose implementations would each be a new subclass.
//
// Every call site on the raiser side names this base class; the choice
// of projection is made exactly once in `raiseToIR` when we construct
// the concrete subclass. Adding a new projection is one new subclass +
// one line in `raiseToIR`.
//
// The class is stateless per call (it caches nothing itself); caches
// that depend on a raise-instance's IR emission position live in
// `RaiseContext`.
class WaveProjection {
public:
  WaveProjection(const ISAProfile &srcIsa, const ISAProfile &tgtIsa,
                 llvm::Type *i32Ty, llvm::Type *i64Ty)
      : src_(srcIsa), tgt_(tgtIsa),
        waveMaskTy_(tgtIsa.isWave32() ? i32Ty : i64Ty) {}

  virtual ~WaveProjection() = default;

  const ISAProfile &sourceIsa() const { return src_; }
  const ISAProfile &targetIsa() const { return tgt_; }
  // Hardware-width wave mask (i32 on wave32 target, i64 on wave64 target).
  // Distinct from the source-width EXEC storage in `AllocaRegFile::execTy`.
  llvm::Type *waveMaskTy() const { return waveMaskTy_; }

  // Emit the current lane's linear index within the wavefront. Uses
  // amdgcn.mbcnt.lo (+ mbcnt.hi on wave64) with an all-ones mask: mbcnt
  // counts set bits strictly below the current lane, which with `-1` as
  // the mask equals the lane id. Result type is i32. Wave-size choice is
  // based on the *target* ISA (waveMaskTy) because the lane id is a
  // runtime property of the hardware the raised IR runs on, independent
  // of the source ISA.
  //
  // Provided as a non-virtual method on the base because every known
  // projection derives lane id the same way (mbcnt against the target's
  // hardware wave). Override only if a future projection changes what
  // "the current lane" means (e.g. a thread-loop projection where a
  // single target lane iterates over multiple source lanes).
  virtual llvm::Value *emitLaneIdx(llvm::IRBuilder<> &B) const;

  // Given the current EXEC alloca value (source-width iN), return an i1
  // true iff the current lane is active. Concrete projections define
  // what "active" means — modulo-replication fans each target lane onto
  // bit `lane_id mod W_src` of the source EXEC mask.
  virtual llvm::Value *emitLaneActiveBit(llvm::IRBuilder<> &B,
                                          llvm::Value *execVal) const = 0;

  // Collect a per-lane i1 predicate into a wave-level bit-mask of width
  // `resultTy`. Invariant: the ballot MUST match the target wave width
  // (waveMaskTy) because the AMDGPU backend only has selection patterns
  // for ballot.i32 on wave32 hardware and ballot.i64 on wave64. Must be
  // emitted in "outer" / full-EXEC control flow so inactive lanes don't
  // silently contribute 0.
  //
  // Projections differ in how they reconcile a wave-mask of width
  // `waveMaskTy` with a caller-requested `resultTy` of a different
  // width. Modulo-replication truncates when narrowing; other
  // projections might refuse outright or redistribute bits.
  virtual llvm::Value *ballotI1ToWidth(llvm::IRBuilder<> &B, llvm::Value *pred,
                                        llvm::Type *resultTy,
                                        const llvm::Twine &name = "ballot")
      const = 0;

  // Project a wave-level bit-mask back onto the current lane's bit (i1).
  // Inverse direction of the ballot. Per-lane i1 inputs short-circuit
  // to a direct pass-through (some callers already produce the final
  // per-lane i1 and route through writeReg*(VCC, i1)); those must not
  // be reinterpreted as a one-bit wave mask.
  virtual llvm::Value *extractLaneBitFromWaveMask(llvm::IRBuilder<> &B,
                                                   llvm::Value *v) const = 0;

protected:
  ISAProfile src_;
  ISAProfile tgt_;
  llvm::Type *waveMaskTy_;
};

// ============================================================================
// ModuloReplicationProjection — today's sole concrete projection.
//
// Modulo-replication's bet is:
//
//   * Target lane L reads bit `L mod W_src` of the source EXEC mask
//     (`emitLaneActiveBit` / `MODREP`).
//   * A target ballot.iN truncated to source-width takes the lower
//     `W_src` lanes as canonical (`ballotI1ToWidth`).
//   * A wave-level mask projected back onto a per-lane bit indexes by
//     `lane_id mod W_src` (`extractLaneBitFromWaveMask`).
//
// None of that is a hardware fact — it is a *choice*. See SPE_DESIGN.md
// §2 for the correctness theorem (wave-size-obliviousness) and §7 for
// the alternatives.
class ModuloReplicationProjection final : public WaveProjection {
public:
  using WaveProjection::WaveProjection;

  llvm::Value *emitLaneActiveBit(llvm::IRBuilder<> &B,
                                  llvm::Value *execVal) const override;
  llvm::Value *ballotI1ToWidth(llvm::IRBuilder<> &B, llvm::Value *pred,
                                llvm::Type *resultTy,
                                const llvm::Twine &name = "ballot")
      const override;
  llvm::Value *extractLaneBitFromWaveMask(llvm::IRBuilder<> &B,
                                           llvm::Value *v) const override;
};

// ============================================================================
// ThreadLoopProjection — placeholder subclass for the SPE_DESIGN.md §7
// coverage ladder's second rung.
//
// The thread-loop rung wraps the raised IR body in `for iter in 0..R:`
// with R = W_tgt / W_src, using only the lower W_src target lanes per
// iteration. It naturally handles SPE_DESIGN.md §3 Class 3 (no
// replicas, no inter-replica races) and Class 4 (per-iteration EXEC),
// trading full-wave throughput for expanded coverage. It does NOT
// dissolve Class 2 cross-lane obstructions (see §7 "The hard wall that
// no projection crosses").
//
// Today no corpus kernel reaches this rung (gpt-oss-derisking.md §9.1
// reports every GPT-OSS / hipBLASLt / Gluon kernel is outcome (a) or
// (b) under modulo-replication). The class is declared here so §7's
// type hierarchy is visible in code; constructing it today is a
// principled `report_fatal_error` rather than silent acceptance,
// because the projection's emission semantics require a structural
// rewrite of the raiser's main loop that has not been implemented.
//
// MAINTENANCE. When implementing thread-loop (expected trigger: a
// corpus kernel in outcome (c) under modulo-replication that would be
// outcome (a) under thread-loop), the steps are:
//   1. Populate the overridden emitters with thread-loop semantics
//      (see SPE_DESIGN.md §7 "Projection 4").
//   2. Add the additional correctness obligations the thread-loop
//      projection introduces — barrier hoisting, LDS-aliasing — as
//      extra checks in `buildObstructionReport` gated on the current
//      projection choice.
//   3. Extend `decideProjection` to try thread-loop after modulo-
//      replication refuses, per the ladder in SPE_DESIGN.md §7.
class ThreadLoopProjection final : public WaveProjection {
public:
  // Placeholder ctor: aborts loudly so a typo that instantiates this
  // subclass before the implementation lands surfaces at raise time
  // rather than as a silent miscompile.
  ThreadLoopProjection(const ISAProfile &srcIsa, const ISAProfile &tgtIsa,
                       llvm::Type *i32Ty, llvm::Type *i64Ty);

  llvm::Value *emitLaneActiveBit(llvm::IRBuilder<> &B,
                                  llvm::Value *execVal) const override;
  llvm::Value *ballotI1ToWidth(llvm::IRBuilder<> &B, llvm::Value *pred,
                                llvm::Type *resultTy,
                                const llvm::Twine &name = "ballot")
      const override;
  llvm::Value *extractLaneBitFromWaveMask(llvm::IRBuilder<> &B,
                                           llvm::Value *v) const override;
};

// ============================================================================
// EXEC-writer detection — architecture-neutral, authoritative.
//
// Derivation strategy (no string matching, no per-opcode allow-lists):
//
//   (a) Implicit defs — canonical LLVM TableGen-derived source of truth.
//       `MCInstrDesc::implicit_defs()` is iterated during decoding and
//       each result is normalised through `AMDGPU::mc2PseudoReg` (strips
//       subtarget-variant suffixes) before classification. Results are
//       cached in `DecodedInst::defsEXEC`. Equivalent to
//       `desc.hasImplicitDefOfPhysReg(EXEC)` extended across the
//       EXEC/EXEC_LO/EXEC_HI canonical-alias family. Instructions like
//       `v_cmpx_*` and `s_*_saveexec_*` surface here.
//
//   (b) Explicit defs — required because LLVM models some EXEC writers
//       with EXEC as an *explicit* destination operand (e.g.
//       `s_mov_b64 exec, sN`, `s_and_b32 exec_lo, exec_lo, s2`).
//       `hasImplicitDefOfPhysReg` does NOT cover these by design. Walk
//       the first `desc.getNumDefs()` operands (TableGen convention:
//       defs always come first in the MCInst operand list) and classify
//       each through `AMDGPU::mc2PseudoReg`, same as (a).
//
// (a) ∪ (b) is exhaustive for AMDGPU: an MCInst either defines a
// register implicitly (via TableGen `let Defs = [...]`) or explicitly
// (as an `outs` operand). There is no third path. Both halves ground in
// MCInstrDesc; no mnemonic parsing and no per-opcode lists.
bool instructionWritesEXEC(const DecodedInst &di, const MCState &mc);

// ============================================================================
// Cross-wave safety warning (Phase 1.4) — legacy warn-only surface.
//
// Kept for the `lit_tests/cross_wave_warn` regression test, which
// validates the principle that a cross-wave translation with only
// lane-position-INDEPENDENT EXEC writers continues to raise under the
// new classifier gate. In production the structured decider
// `decideProjection` below is the primary surface and this function
// becomes a diagnostic logger only (routed through LLVM_DEBUG).
//
// Returns true iff a diagnostic was emitted.
bool emitCrossWaveWarning(const WaveProjection &proj, const MCState &mc,
                          llvm::ArrayRef<DecodedInst> insts,
                          llvm::StringRef sourceISA,
                          llvm::StringRef targetISA);

} // namespace transpiler

#endif
