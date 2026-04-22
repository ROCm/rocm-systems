// Unit tests for `WaveProjection::providesFullWaveExecInvariant()` —
// the virtual method introduced by the WMMA-refusal commit in the
// matmul_fp16 triage.  The method is the contract that
// `handle_valu_vop3p.cpp`'s WMMA → MFMA handlers consult before
// calling `emitWMMAtoMFMA*`; a regression that silently flips the
// return value for either concrete projection would silently
// unleash a WMMA miscompile on phantom-lane kernels (see the
// `wmma_phantom_lane_refuse/` lit fixture for the end-to-end
// regression fence).
//
// These tests cover:
//
//   * `ModuloReplicationProjection::providesFullWaveExecInvariant()`
//     inherits the base default of `false` — MODREP's
//     `emitInitialExec` returns source-width all-ones and does
//     NOT call `@llvm.amdgcn.init_whole_wave`, so hardware EXEC
//     stays at whatever the dispatcher set it to (the source-
//     wave active mask for a partial-wave launch).  Collective
//     lowerings that need all target-wave lanes active (WMMA →
//     MFMA redistribute-collect) must refuse when this is false.
//
//   * `WaveNativeProjection::providesFullWaveExecInvariant()`
//     overrides to `true` because
//     `WaveNativeProjection::emitInitialExec` explicitly emits
//     `@llvm.amdgcn.init_whole_wave` which sets HW EXEC = -1 for
//     the remainder of the kernel body.
//
//   * A base-class pointer dispatch works correctly — callers in
//     `handle_valu_vop3p.cpp` hold a `WaveProjection &` reference
//     and rely on virtual dispatch to pick the right answer per
//     kernel.
//
//   * ThreadLoopProjection is NOT tested here because the class is
//     a `report_fatal_error` placeholder today (see
//     `wave_projection.hpp` for the rationale); if a future commit
//     graduates it to a real projection, the author should add a
//     test here pinning the intended return value.

#include "../wave_projection.hpp"

#include "../isa_profile.hpp"

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"

#include "gtest/gtest.h"

using namespace llvm;
using transpiler::ISAProfile;
using transpiler::ModuloReplicationProjection;
using transpiler::WaveNativeProjection;
using transpiler::WaveProjection;

namespace {

// Canonical source / target ISA profiles for the cross-widening
// direction the two projections cover.  Matches the source
// (gfx1250, wave32) / target (gfx942, wave64) cross-widening the
// matmul_fp16 triage surfaced the bug on.
//
// Constructed via `ISAProfile::forTesting(waveSize)` rather than
// `fromSubtarget` because standing up an `MCSubtargetInfo` would
// require the full LLVM AMDGPU init chain (multiple
// `InitializeAllTarget*` calls + target-lookup dance) and buys
// us nothing for the contract check this file pins — only the
// `waveSize` dimension is consulted by `WaveNativeProjection`'s
// direction-gate assertion, and
// `providesFullWaveExecInvariant()`'s return value is independent
// of the other feature flags.  See the `forTesting` docstring in
// `isa_profile.hpp` for the test-only scope of this factory.
ISAProfile makeGfx1250Profile() { return ISAProfile::forTesting(32); }
ISAProfile makeGfx942Profile() { return ISAProfile::forTesting(64); }

} // namespace

// ----------------------------------------------------------------------------
// MODREP: the base-class default of `false` is inherited; the
// projection does NOT decouple HW EXEC from the source active
// mask.  Pins the default so a future refactor that accidentally
// promotes MODREP to full-wave-EXEC semantics (without actually
// emitting `init_whole_wave`) would be caught here rather than by
// the WMMA handler silently accepting MODREP kernels and
// miscompiling them.
// ----------------------------------------------------------------------------
TEST(WaveProjectionContract, ModuloReplicationDoesNotProvideFullWaveExec) {
  LLVMContext ctx;
  auto *i32Ty = Type::getInt32Ty(ctx);
  auto *i64Ty = Type::getInt64Ty(ctx);

  ISAProfile src = makeGfx1250Profile();
  ISAProfile tgt = makeGfx942Profile();

  ModuloReplicationProjection proj(src, tgt, i32Ty, i64Ty);
  EXPECT_FALSE(proj.providesFullWaveExecInvariant());

  // Cross-check through a base-class reference to confirm the
  // virtual dispatch resolves to the correct override (or lack
  // thereof — MODREP inherits the base's `false`).
  const WaveProjection &base = proj;
  EXPECT_FALSE(base.providesFullWaveExecInvariant());
}

// ----------------------------------------------------------------------------
// WaveNative: overrides to `true` because `emitInitialExec` emits
// `@llvm.amdgcn.init_whole_wave` which forces HW EXEC = -1 for the
// kernel body.  Pins the override so a future refactor that
// removes it (or moves the `init_whole_wave` emission site without
// updating the contract method) is caught here before it can
// silently gate off the WMMA handlers' acceptance of
// WaveNative-raised kernels.
// ----------------------------------------------------------------------------
TEST(WaveProjectionContract, WaveNativeProvidesFullWaveExec) {
  LLVMContext ctx;
  auto *i32Ty = Type::getInt32Ty(ctx);
  auto *i64Ty = Type::getInt64Ty(ctx);

  // WaveNativeProjection's constructor asserts `src.isWave32() &&
  // !tgt.isWave32()` (per its docstring — this projection is only
  // defined for wave32 → wave64 cross-widening), so we construct
  // with the canonical gfx1250 → gfx942 pair.  Any other direction
  // would fatal-error inside the constructor.
  ISAProfile src = makeGfx1250Profile();
  ISAProfile tgt = makeGfx942Profile();

  WaveNativeProjection proj(src, tgt, i32Ty, i64Ty);
  EXPECT_TRUE(proj.providesFullWaveExecInvariant());

  const WaveProjection &base = proj;
  EXPECT_TRUE(base.providesFullWaveExecInvariant());
}

// ----------------------------------------------------------------------------
// The base class's default return value is `false`.  Tested through
// `ModuloReplicationProjection` above (which inherits the default),
// but also pinned explicitly here via a minimal subclass that
// implements only the pure virtuals.  This protects against a
// future author flipping the default to `true` and thereby silently
// graduating un-audited projection classes to "allowed to run WMMA"
// status.
// ----------------------------------------------------------------------------
namespace {
// Minimal concrete `WaveProjection` that only satisfies the pure
// virtuals required to instantiate — it is not a real projection and
// is used SOLELY to assert the base class's default return values
// for the concrete-but-permissive contract methods.
class DefaultTestProjection final : public WaveProjection {
public:
  using WaveProjection::WaveProjection;
  llvm::Value *emitLaneActiveBit(llvm::IRBuilder<> &,
                                  llvm::Value *) const override {
    return nullptr;  // unused by these tests
  }
  llvm::Value *ballotI1ToWidth(llvm::IRBuilder<> &, llvm::Value *,
                                llvm::Type *,
                                const llvm::Twine &) const override {
    return nullptr;  // unused by these tests
  }
  llvm::Value *extractLaneBitFromWaveMask(llvm::IRBuilder<> &,
                                           llvm::Value *) const override {
    return nullptr;  // unused by these tests
  }
};
} // namespace

TEST(WaveProjectionContract, BaseDefaultIsNotFullWaveExec) {
  LLVMContext ctx;
  auto *i32Ty = Type::getInt32Ty(ctx);
  auto *i64Ty = Type::getInt64Ty(ctx);

  ISAProfile src = makeGfx1250Profile();
  ISAProfile tgt = makeGfx942Profile();

  DefaultTestProjection proj(src, tgt, i32Ty, i64Ty);
  EXPECT_FALSE(proj.providesFullWaveExecInvariant());
}
