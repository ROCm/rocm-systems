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
// use today. hotswap/docs/wave-size-translation.md §2.2 catalogues the
// alternatives (thread-loop, scalarisation, half-wave-masking) that
// are not yet implemented but whose implementations would each be a
// new subclass.
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
      : src_(srcIsa), tgt_(tgtIsa), i32Ty_(i32Ty), i64Ty_(i64Ty),
        waveMaskTy_(tgtIsa.isWave32() ? i32Ty : i64Ty) {}

  virtual ~WaveProjection() = default;

  const ISAProfile &sourceIsa() const { return src_; }
  const ISAProfile &targetIsa() const { return tgt_; }
  // Hardware-width wave mask (i32 on wave32 target, i64 on wave64 target).
  // Distinct from the EXEC alloca storage width returned by
  // `execStorageTy()`.
  llvm::Type *waveMaskTy() const { return waveMaskTy_; }

  // Source-width wave mask (i32 on wave32 source, i64 on wave64 source).
  // This is the width the source ISA observes when reading/writing EXEC
  // and SGPR wave masks through 32-bit or 64-bit scalar operations.
  // Modulo-replication keeps `execStorageTy()` equal to this, so EXEC
  // and the source author's scalar view share a representation; wave-
  // native cross-widening widens `execStorageTy()` to the target
  // hardware mask, leaving `sourceWaveMaskTy()` unchanged so source-
  // width operands (SGPR scalars, imm masks, save/restore SGPRs) keep
  // their native width at the boundary.
  llvm::Type *sourceWaveMaskTy() const {
    return src_.isWave32() ? i32Ty_ : i64Ty_;
  }

  // EXEC alloca storage width chosen by the projection. Modulo-
  // replication returns the source wave width (the long-standing
  // default, see hotswap/docs/wave-size-translation.md §5.1); wave-
  // native cross-widening returns the target hardware wave mask
  // width (`waveMaskTy_`) so a target-width ballot from a data-
  // dependent `v_cmpx` AND's directly into EXEC without losing the
  // upper half. Callers in `AllocaRegFile::init` use this to size
  // the alloca; operand read/write helpers in `RaiseContext` use
  // the same width to decide where a widen-by-replication or a
  // narrow-by-truncation is required on source-width scalars.
  virtual llvm::Type *execStorageTy() const { return sourceWaveMaskTy(); }

  // Emit the initial value to store into the EXEC alloca at kernel
  // entry. Default is all-ones (every source lane active on entry),
  // which matches the architectural boot state of a dispatched wave.
  //
  // Projections that decouple the HARDWARE EXEC (what the target GPU
  // applies to EXEC-gated writes) from the MODELED source EXEC (what
  // the transpiler's `emitUnderExec` diamonds read through the alloca)
  // override this to emit an entry-block side effect that captures the
  // hardware EXEC into the alloca while forcing hardware EXEC to all-
  // ones. Wave-native Wave32→Wave64 cross-widening is the canonical
  // case: the WMMA→MFMA redistribution pipeline in `wmma_lowering.cpp`
  // must run under hardware EXEC = -1 so lanes 32-63 participate in
  // the Wave64 MFMA (otherwise they never write their destination
  // VGPRs on a partial-wave launch and MFMA reads garbage), and
  // memory stores must STILL honour the original per-lane active mask.
  // `WaveNativeProjection` threads this by emitting
  // `@llvm.amdgcn.init_whole_wave` (sets HW EXEC=-1, returns the
  // original per-lane active bit) followed by a ballot that packs the
  // per-lane bit into a wave-width mask for the alloca. Downstream,
  // every VGPR write / memory store / LDS op already routes through
  // `RaiseContext::emitUnderExec`, which reads the alloca and
  // conditionally branches, so the hardware-vs-modeled EXEC split is
  // invisible to handlers.
  //
  // This replaces the earlier `@llvm.amdgcn.strict.wwm`-per-MFMA-output
  // strategy in the WMMA lowering, which crashed
  // `SIPreAllocateWWMRegs` on large matmul kernels: that pass requires
  // a DEDICATED physical VGPR per vreg defined inside a WWM bracket,
  // and the WWM def-chain from an MFMA-output marker walks back
  // through the entire accumulator initialisation (≈200 IMPLICIT_DEF
  // / AV_MOV_B32 0 defs in a 128×128 f16 matmul tile's entry region),
  // which cannot fit in gfx942's 256-VGPR pool once the kernel's
  // own computation has claimed its share. Moving the EXEC=-1
  // guarantee to kernel entry sidesteps the allocator pressure
  // entirely because no intermediate vreg is ever "inside WWM" —
  // the whole kernel body runs under HW EXEC=-1 and regalloc is
  // ordinary.
  virtual llvm::Value *emitInitialExec(llvm::IRBuilder<> &B) const;

  // True iff a 32-bit write to EXEC_LO carries "replicate across the
  // full widened EXEC" semantics rather than the source-architectural
  // "replace the low half, keep the high half" semantics. Only wave-
  // native cross-widening sets this: on wave32 source → wave64 target
  // the source author's `s_mov_b32 exec_lo, v` means "set the whole
  // wavefront's EXEC to v", and the wave-native projection models
  // each target lane as an independent source thread, so a
  // conceptually-whole-wave write must fan out to both halves of the
  // widened EXEC. Modulo-replication keeps this false because source
  // wave width matches EXEC storage width and no widening is needed.
  virtual bool broadcastNarrowExecLoWrite() const { return false; }

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

  // True iff this projection guarantees hardware EXEC = -1 between
  // `emitUnderExec` diamonds *kernel-wide*.  When this is true the
  // WMMA → MFMA redistribute / MFMA / collect pipeline in
  // `wmma_lowering.cpp` can run without any additional EXEC
  // scaffolding because every target-wave lane is already HW-active
  // for the entire kernel body.  When this is false the handler
  // must route the pipeline through `emitUnderFullWaveExec` below,
  // which emits per-dword `@llvm.amdgcn.strict.wwm` markers so the
  // backend inserts a scoped EXEC save/set-minus-one/restore around
  // the cross-lane collective.
  //
  // `ModuloReplicationProjection` returns false because it keeps
  // hardware EXEC at the source-wave active mask (see
  // `emitInitialExec`'s comment on why this is correct for the wave-
  // size-oblivious class of kernels — phantom target lanes stay
  // HW-inactive kernel-wide, which prevents undef-VGPR contamination
  // of the source author's own cross-lane ops but means the WMMA
  // lowering has to locally widen HW EXEC for its own synthesised
  // cross-lane ops).  `WaveNativeProjection` returns true because its
  // `emitInitialExec` explicitly calls `@llvm.amdgcn.init_whole_wave`
  // to set HW EXEC=-1 for the kernel body.
  virtual bool providesFullWaveExecInvariant() const { return false; }

  // Number of source waves whose per-lane fragment data is present in
  // each target wave under this projection's mapping.  Callers that
  // synthesise per-source-wave passes (most notably the WMMA → MFMA
  // redistribute / MFMA / collect pipeline in `wmma_lowering.cpp`)
  // iterate `groupBase ∈ {0, W_src, ..., (numSourceWavesPerTarget() -
  // 1) * W_src}` so that each pass covers exactly one source wave's
  // worth of data.
  //
  // `WaveNativeProjection` (wave32 → wave64 cross-widening) maps two
  // source waves into one target wave (source wave 0 → target lanes
  // 0..31, source wave 1 → target lanes 32..63) — returns 2.
  //
  // `ModuloReplicationProjection` in the cross-widening direction
  // always fires under the phantom-lane regime
  // (`max_flat_workgroup_size < targetWaveSize`, see `raiser.cpp`),
  // which by definition has a single source wave (the workgroup is
  // below one target wavefront, so the second "half" of the target
  // wave has no source workitem — those lanes are phantom).  Same-
  // wave MODREP instantiations (source == target wave width) also
  // carry exactly one source wave per target.  Returns 1 in either
  // case.  A future projection (`ThreadLoopProjection`) would answer
  // this based on its wrap count.
  //
  // Pure virtual so every new projection must answer the question
  // explicitly — a silent default would let a new cross-widening
  // projection pick the wrong pass count in `wmma_lowering.cpp` and
  // emit a bogus second-source-wave MFMA that read undef from
  // phantom lanes.
  virtual unsigned numSourceWavesPerTarget() const = 0;

  // Return `v` wrapped in `@llvm.amdgcn.strict.wwm` iff the current
  // projection does NOT already guarantee HW EXEC=-1 kernel-wide.
  // On projections that DO provide the invariant (e.g.
  // `WaveNativeProjection` via kernel-entry `init_whole_wave`),
  // this is an identity: returning the input unchanged avoids the
  // regalloc pressure that `SIPreAllocateWWMRegs` would impose if
  // we emitted a redundant marker (see `WaveProjection::emitInitial
  // Exec`'s block comment for the 128×128-matmul accumulator-ring
  // failure mode).
  //
  // The marker tells the AMDGPU backend's `SIWholeQuadMode` pass
  // that the producing instruction chain must execute under HW
  // EXEC=-1.  EXEC save/restore materialises as
  // `s_or_saveexec_b64 sN, -1` / `s_mov_b64 exec, sN` around the
  // backward-dataflow-minimal region SIWholeQuadMode computes.
  //
  // Typing: `strict.wwm`'s overload set is `llvm_any_ty`, so this
  // helper accepts any SSA type (scalar i32 for per-dword wrapping,
  // `<4 x float>` for an MFMA accumulator, `<2 x half>` for a
  // bitcast-ready operand pack, etc.).  WMMA→MFMA lowering uses
  // i32 for per-dword result wrapping and `<4 x float>` / `<4 x
  // i32>` for per-MFMA-output wrapping so the MFMA itself is
  // dragged into the WWM backward slice.
  //
  // Why per-MFMA-output wrapping (and not only per-result-dword):
  // SIWholeQuadMode's backward propagation from a `strict.wwm` on a
  // `ds_bpermute` result stops at the bpermute boundary because
  // `ds_bpermute`'s read is EXEC-independent — the backend sees
  // that the bpermute picks up source lanes 0..W_src-1 by address
  // and concludes that the MFMA output on lanes W_src..2*W_src-1
  // is "not consumed".  That's correct under a single-pass MFMA,
  // but WRONG for our cross-widening lowering: the MFMA OUTPUT
  // lives in the SAME destination VGPR across all 64 target lanes,
  // and the subsequent `collectResult` bpermute DOES read those
  // upper-half lanes (target lanes 16..31 collect from source
  // lanes 32..63's MFMA output to get rows 8..15 in WMMA layout).
  // If the MFMA runs under HW EXEC=source-active-mask, target
  // lanes 32..63 never write their MFMA destination VGPR, so the
  // collect reads stale data and rows 8..15 of the output are
  // corrupted.  Wrapping the MFMA output in `strict.wwm` forces
  // SIWholeQuadMode to mark the MFMA itself as WWM, so it writes
  // every lane's destination VGPR.
  llvm::Value *wrapAsWWMValue(llvm::IRBuilder<> &B, llvm::Value *v,
                               const llvm::Twine &name = "wwm") const;

protected:
  ISAProfile src_;
  ISAProfile tgt_;
  // Retained on the base so subclass overrides of `sourceWaveMaskTy()`
  // / `execStorageTy()` can return the canonical i32/i64 IR types
  // without re-deriving them from the current IRBuilder's context
  // (subclasses are constructed once per kernel and outlive any
  // particular builder).
  llvm::Type *i32Ty_;
  llvm::Type *i64Ty_;
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
// None of that is a hardware fact — it is a *choice*. See hotswap/
// docs/wave-size-translation.md §6 for the correctness theorem
// (wave-size-obliviousness) and §2.2 for the alternatives.
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

  // MODREP in the cross-widening direction only instantiates when
  // `raiser.cpp` routes phantom-lane kernels here as the fallback,
  // and phantom-lane by definition has exactly one source wave per
  // target wavefront.  Same-wave MODREP instantiations (source ==
  // target wave width) are also one-source-wave-per-target.
  unsigned numSourceWavesPerTarget() const override { return 1; }
};

// ============================================================================
// WaveNativeProjection — cross-widening (wave32 → wave64) projection
// that preserves the full target-hardware EXEC mask.
//
// Whereas `ModuloReplicationProjection` keeps the EXEC alloca sized to
// the *source* wave width and truncates a target ballot to that width
// (losing the upper half on wave32 → wave64 cross-widening), the wave-
// native projection widens the EXEC alloca to the *target* hardware
// wave-mask width. Each target lane is treated as an independent
// source-thread equivalent, so a data-dependent `v_cmpx` that
// naturally produces a different answer on target lanes 0..31 vs
// 32..63 keeps both halves distinct through the round trip:
//
//     cmp    = icmp ...                           ; per-target-lane i1
//     ballot = @llvm.amdgcn.ballot.i64(cmp)      ; 64-bit wave mask
//     curExec= load i64, ptr %exec               ; widened storage
//     newExec= and i64 curExec, ballot           ; no trunc needed
//     store  i64 newExec, ptr %exec
//
// The price: source-width EXEC writes (`s_mov_b32 exec_lo, v`, `s_*_
// saveexec_b32 sN, ...`) must be reconciled with the widened storage.
// This projection picks the symmetry that matches the source author's
// intent on a wave32 kernel — "the whole wave" — by replicating the
// 32-bit value into both halves of the widened EXEC. Symmetrically,
// reads that narrow EXEC to 32 bits (e.g. `s_mov_b32 sN, exec_lo`)
// take the low half; the save/restore round trip is lossless as long
// as the kernel author never observes the upper half of EXEC
// independently of the lower half, which wave32 source ISAs cannot
// express.
//
// Scope. This projection is correct only for wave32 → wave64 cross-
// widening. Instantiating it for same-wave or narrowing directions
// would make `broadcastNarrowExecLoWrite()` change EXEC semantics in
// directions the source author can disambiguate, so the constructor
// asserts. The ladder in hotswap/docs/wave-size-translation.md §2.2
// still reserves `ThreadLoopProjection` for higher-obligation
// rewrites; wave-native sits between the two as the first rung that
// handles data-dependent EXEC writes correctly without restructuring
// the raiser's main loop.
class WaveNativeProjection final : public WaveProjection {
public:
  WaveNativeProjection(const ISAProfile &srcIsa, const ISAProfile &tgtIsa,
                        llvm::Type *i32Ty, llvm::Type *i64Ty);

  llvm::Type *execStorageTy() const override { return waveMaskTy_; }
  bool broadcastNarrowExecLoWrite() const override { return true; }
  // `init_whole_wave` in `emitInitialExec` forces HW EXEC=-1 for the
  // kernel body, so this projection does provide the full-wave-EXEC
  // invariant that WMMA→MFMA lowering (and other
  // all-lanes-must-participate collectives) require.
  bool providesFullWaveExecInvariant() const override { return true; }
  // Wave32 → wave64 cross-widening: two source waves stack into one
  // target wave (source wave 0 → target lanes 0..31, source wave 1 →
  // target lanes 32..63).  Callers emitting per-source-wave passes
  // run two iterations under this projection.
  unsigned numSourceWavesPerTarget() const override { return 2; }

  llvm::Value *emitInitialExec(llvm::IRBuilder<> &B) const override;
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
// ThreadLoopProjection — placeholder subclass for the second rung of
// the coverage ladder described in hotswap/docs/wave-size-
// translation.md §2.2.
//
// The thread-loop rung wraps the raised IR body in `for iter in 0..R:`
// with R = W_tgt / W_src, using only the lower W_src target lanes per
// iteration. It naturally handles Class 3 obstructions (no replicas,
// no inter-replica races) and Class 4 (per-iteration EXEC) as
// catalogued in wave-size-translation.md §6, trading full-wave
// throughput for expanded coverage. It does NOT dissolve Class 2
// cross-lane obstructions (see wave-size-translation.md §7's
// unrewritable and pending tables).
//
// Today no corpus kernel reaches this rung (hotswap/docs/gpt-oss-
// derisking.md §9.1 reports every GPT-OSS / hipBLASLt / Gluon kernel
// is outcome (a) or (b) under modulo-replication). The class is
// declared here so the projection ladder described in wave-size-
// translation.md §2.2 has a type-level anchor in code; constructing
// it today is a principled `report_fatal_error` rather than silent
// acceptance, because the projection's emission semantics require a
// structural rewrite of the raiser's main loop that has not been
// implemented.
//
// MAINTENANCE. When implementing thread-loop (expected trigger: a
// corpus kernel in outcome (c) under modulo-replication that would be
// outcome (a) under thread-loop), the steps are:
//   1. Populate the overridden emitters with thread-loop semantics
//      (follow the projection sketch in wave-size-translation.md
//      §2.2's projections table).
//   2. Add the additional correctness obligations the thread-loop
//      projection introduces — barrier hoisting, LDS-aliasing — as
//      extra checks in `buildObstructionReport` gated on the current
//      projection choice.
//   3. Extend `decideProjection` to try thread-loop after modulo-
//      replication refuses, per the ladder in wave-size-
//      translation.md §2.2.
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

  // `report_fatal_error` placeholder: the projection is not
  // implemented (see the class docstring).  An override here keeps
  // the base class's pure-virtual contract satisfied so we don't
  // need to reintroduce a non-pure default — a value will never be
  // observed because the constructor aborts before any instance is
  // returned to a caller.
  unsigned numSourceWavesPerTarget() const override;
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
