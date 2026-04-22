#include "handle_valu_internal.hpp"

#include "semop.hpp"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h" // AMDGPU::OpName
#include "SIDefines.h"                        // SISrcMods::OP_SEL_0
#include "Utils/AMDGPUBaseInfo.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace transpiler {

// Cross-lane VALU primitives — the subset of VALU opcodes whose result
// in lane L depends on values held by lane L' != L. Isolated from the
// rest of handleVALU because this is exactly the surface the cross-
// wave strategy (hotswap/docs/wave-size-translation.md §§5.3 and 7)
// keeps iterating on: every rewrite from the "wave-size-baked cross-
// lane" rewrite table lands in this file, not scattered through the
// VALU arithmetic sections.
//
// Each branch MUST use a genuine cross-lane LLVM intrinsic
// (`llvm.amdgcn.readlane`, `writelane`, `readfirstlane`, `mbcnt.{lo,
// hi}`, etc.). A "same-lane" stub that ignores the source-lane
// selector is a silent miscompile for any kernel that feeds divergent
// operands into the primitive. Several permlane variants here are
// known broken (see the pending-rewrite table in wave-size-
// translation.md §7); they stay same-lane for now but any new cross-
// lane SemOp must be modelled correctly before landing.

// Shared `ds_bpermute`-based emulation of the `VOP_PERMLANE_SWAP`
// profile (see `llvm/lib/Target/AMDGPU/VOP1Instructions.td`) — two
// tied VGPR pairs (vdst↔vdst_in, src0↔src0_out) whose values are
// swapped across the lane partner `L XOR partnerXorMask`.  Called
// from both the XOR-16 and XOR-32 arms below; the only things that
// differ between the two SemOp cases are:
//
//   * `partnerXorMask` — 16 for `v_permlane16_swap_b32` (XOR-16 pair
//     stays within each 32-lane half), 32 for
//     `v_permlane32_swap_b32` (XOR-32 pair spans both halves of a
//     wave64).
//   * `ssaPrefix`       — `"pls16"` or `"pls32"`, stamped onto every
//     emitted SSA value's twine.  Lit fixtures pin on this prefix
//     (`lit_tests/{c2_permlane_swap,v_permlane32_swap_b32}/`), so it
//     IS a load-bearing contract; any future widener (XOR-64 on some
//     future ISA, say) would pick its own prefix.
//
// EXEC / fi / bc handling: the MCInst surfaces `fi` and `bound_ctrl`
// as named immediate operands ONLY in the e64 form (VOP3OpSel
// encoding); the e32 form (VOP1 encoding, which the GPT-OSS /
// AITER corpus exclusively emits) has no fi/bc operands and they
// default to 0.  The `ds_bpermute` emulation is observationally
// equivalent to the source's `fi=0, bc=0` semantics WHEN
// EXEC=all-active at the swap site — which is the invariant
// butterfly-reduction kernels (Triton / AITER fmha reduction cores)
// maintain (the kernel emits divergent EXEC writes only at
// iteration boundaries, not inside the reduction core).  For
// partial-EXEC sites:
//
//   * fi=0 source semantics: inactive lanes' contribution is 0;
//     ds_bpermute returns the stale VGPR value instead.  Divergence
//     only on inactive lanes.
//   * bc=0 source semantics: out-of-range source lane → return
//     %old.  For the XOR-16 swap every partner is in-range (XOR 16
//     stays within each 32-lane half); for the XOR-32 swap every
//     partner is in-range on wave64 (XOR 32 maps 0..31 ↔ 32..63
//     within the single wave).  `bc` is irrelevant for both masks.
//
// The corpus patterns (e32 form, EXEC=full at the swap site) are
// bit-exact correct under this emulation; fi/bc are accepted
// without inspection and the EXEC=full assumption is documented
// here.  P4.b future-hardening (wave-size-translation.md §10): a
// "true fi=0 emulation" would zero inactive lanes' VGPR
// contribution before the bpermute via `select EXEC[L], src, 0`,
// at the cost of two extra selects per swap; a static alternative
// is a classifier check that proves EXEC=full at the swap site
// and refuses otherwise.  Today's corpus invariant makes both
// deferrable.
static HandlerResult
emitPermLaneSwapEmulation(RaiseContext &ctx, const DecodedInst &di,
                           OpResolver &op, uint32_t partnerXorMask,
                           const char *ssaPrefix) {
  HandlerResult hr;

  // Operand-table contract (same for every `VOP_PERMLANE_SWAP`
  // profile instance): MCInst operand 0 is `vdst` (output, tied to
  // `vdst_in` input — the disassembler elides the tied input), and
  // the `OpName::src0_out` named operand is the second output (tied
  // to `src0` input, same elision).  Logical inputs are therefore
  // `readReg32(vdst)` (the pre-instruction value of the tied vdst
  // slot) and `op.src(0)` (which `buildSrcMap` keeps as src0 after
  // the `vdst_in` elision).  Two outputs, two tied inputs, both
  // carried on VGPRs.
  int src0OutIdx = AMDGPU::getNamedOperandIdx(di.inst.getOpcode(),
                                               AMDGPU::OpName::src0_out);
  if (src0OutIdx < 0 ||
      static_cast<unsigned>(src0OutIdx) >= di.inst.getNumOperands() ||
      !di.inst.getOperand(static_cast<unsigned>(src0OutIdx)).isReg()) {
    std::string msg = std::string(di.mnemonic) +
                      " missing OpName::src0_out register operand — "
                      "operand-table mismatch (expected the "
                      "VOP_PERMLANE_SWAP profile's second-output "
                      "operand-table slot to be a register)";
    hr.failure = RaiseFailure::unsupportedShape(di, "VALU", msg);
    return hr;
  }
  ParsedReg vdstReg = op.dst();
  ParsedReg src0OutReg =
      ctx.parseReg(di.getReg(static_cast<unsigned>(src0OutIdx)),
                    static_cast<unsigned>(src0OutIdx));

  // Snapshot BOTH inputs up-front (read-before-write: `vdst_in`
  // aliases `vdst`, so writing vdstReg first would shadow this
  // read; mirroring the snapshot for `src0_in` keeps the pair
  // structural rather than relying on the rest of the handler's
  // implicit ordering).
  Value *vdstIn = ctx.regs.readReg32(ctx.B, vdstReg);
  Value *src0In = op.src(0);

  // Partner lane: `L XOR partnerXorMask`.  Computed on the target
  // hardware lane id (mbcnt-derived), with per-BB memoisation via
  // `emitLaneIdx`.  Byte-address shift by 2 is the ds_bpermute
  // convention — each lane's selector is the byte offset of the
  // source lane's LDS slot (LDS slot size = 4 bytes for a 32-bit
  // dword).
  Value *laneId = ctx.emitLaneIdx();
  Value *partner = ctx.B.CreateXor(
      laneId, ctx.B.getInt32(partnerXorMask),
      Twine(ssaPrefix) + "_partner");
  Value *bpermIdx = ctx.B.CreateShl(partner, ctx.B.getInt32(2),
                                     Twine(ssaPrefix) + "_addr");

  // Two convergent ds_bpermute calls, cross-wired so the two tied
  // output VGPRs each pick up their partner's tied-input value:
  //
  //   new_vdst      = bperm(addr, src0_in)
  //   new_src0_out  = bperm(addr, vdst_in)
  //
  // Emitted OUTSIDE `emitUnderExec` so all hardware lanes
  // participate in the bpermute's LDS round-trip; `writeReg32`
  // below wraps the stores for EXEC masking on the target side.
  Function *bperm = Intrinsic::getOrInsertDeclaration(
      &ctx.M, Intrinsic::amdgcn_ds_bpermute);
  Value *newVdst = ctx.B.CreateCall(bperm, {bpermIdx, src0In},
                                     Twine(ssaPrefix) + "_new_vdst");
  Value *newSrc0Out = ctx.B.CreateCall(
      bperm, {bpermIdx, vdstIn}, Twine(ssaPrefix) + "_new_src0_out");
  ctx.writeReg32(vdstReg, newVdst);
  ctx.writeReg32(src0OutReg, newSrc0Out);
  hr.handled = true;
  return hr;
}
HandlerResult handleVALU_CrossLane(RaiseContext &ctx, const DecodedInst &di,
                                    OpResolver &op) {
  HandlerResult hr;
  SemOp sop = di.semOp;

  switch (sop) {

  // ---- v_permlane16_b32 / v_permlanex16_b32 ----
  // P2 lowering — see the permlane16 / permlanex16 row of hotswap/
  // docs/wave-size-translation.md §5.3. Target constraint: `v_permlane16`
  // and `v_permlanex16` are RDNA/gfx10+ instructions and DO NOT exist
  // on CDNA (gfx9/gfx94x). Emitting `llvm.amdgcn.permlane16` or
  // `permlanex16` directly fails isel on gfx942 with "Cannot select:
  // intrinsic %llvm.amdgcn.permlanex16". We therefore emulate both
  // via `ds_bpermute_b32`, which IS available on every AMDGPU
  // generation with LDS (gfx8+), so this lowering is target-
  // independent — it works for gfx1250 → gfx942, gfx1250 → gfx1250,
  // and any future target with ds_bpermute.
  //
  // MCInst operand layout (from VOP3_PERMLANE_Profile's InsVOP3OpSel):
  //
  //   [0] vdst (output)             [5] src2_modifiers (always 0)
  //   [1] src0_modifiers  <-- fi    [6] src2 (SSrc_b32)  = selector_2
  //   [2] src0 (VRegSrc_32) = val   [7] vdst_in (VGPR, tied) = %old
  //   [3] src1_modifiers  <-- bc    [8] op_sel (VOP3OpSel imm, unused
  //   [4] src1 (SSrc_b32)  = sel_1                                here)
  //
  // Selector encoding: src1 and src2 are each 32-bit scalar values
  // containing 8 × 4-bit per-lane selectors. src1 covers within-
  // group lanes 0..7, src2 covers within-group lanes 8..15. Each
  // 4-bit nibble selects a source lane within the 16-lane group.
  //
  // Per-lane emulation, L = `mbcnt`-derived absolute lane id (0..W_t):
  //
  //   group_base  = L & ~0xF           // 16, 32, 48 boundaries
  //   within      = L & 0xF            // 0..15
  //   within_lo   = within & 7         // 0..7 (nibble index)
  //   sel_word    = within < 8 ? src1 : src2
  //   nibble      = (sel_word >> (within_lo * 4)) & 0xF
  //
  //   permlane16  : src_group = group_base
  //   permlanex16 : src_group = group_base ^ 0x10  (swap adjacent groups)
  //
  //   src_lane_abs = src_group | nibble
  //   byte_addr    = src_lane_abs << 2
  //   result       = ds_bpermute(byte_addr, src0)
  //
  // Wave-width correctness under modulo-replication (hotswap/docs/
  // wave-size-translation.md §6's wave-size-obliviousness theorem):
  // the source gfx1250 kernel is wave32 so its selector values
  // encode a shuffle pattern over 2 × 16-lane groups. On wave64
  // target each modrep replica occupies 2 × 16-lane groups (R=2),
  // and the `group ^ 0x10` swap stays within a replica (0↔1 within
  // replica 0, 2↔3 within replica 1), so the modrep invariant is
  // preserved for permlanex16. permlane16 keeps every lane within
  // its own group, trivially within-replica.
  //
  // Handling of `fi` (fetch-invalid) and `bc` (bound_ctrl) — the two
  // i1 immediates encoded via `opsel_i1timm` in PermlanePat
  // (`SISrcMods::OP_SEL_0` bit of src0_modifiers / src1_modifiers):
  //
  //   - `fi=1`: on an EXEC-inactive source lane, the kernel still
  //     fetches that lane's VGPR value (possibly stale). This is
  //     exactly how `llvm.amdgcn.ds.bpermute` behaves naturally
  //     (the LDS-backed path reads the VGPR alloca regardless of
  //     EXEC), so `fi=1` is supported directly.
  //   - `bc=0`: on an "out-of-range" source lane, the target lane
  //     retains %old. For permlane16 the 4-bit selector nibble is
  //     always in [0, 16) so the source lane is always in-group;
  //     `bc=0` is the only case the emulation needs to support.
  //     Under SPE, `writeReg32`'s `emitUnderExec` already retains
  //     prior VDST values on EXEC-masked target lanes, covering the
  //     "target lane inactive" direction of `bc=0`.
  //   - `fi=0` and `bc=1` diverge from the above in ways the
  //     emulation does not model. Every GPT-OSS / softmax /
  //     bitmatrix disassembly we have examined uses `op_sel:[1, 0]`
  //     (fi=1, bc=0); refusing the other combinations keeps the
  //     classifier-gate's "no silent miscompile" invariant intact
  //     rather than emitting ds_bpermute with fi=0 semantics it
  //     does not provide.
  //
  // Future optimisation: on targets that DO support native
  // permlane16 (gfx10+), emit the intrinsic directly for lower
  // latency. Left as a profitability refinement — correctness-first
  // lands the ds_bpermute emulation.
  case SemOp::V_PERMLANE16_B32:
  case SemOp::V_PERMLANEX16_B32: {
    const bool isPermlaneX16 = (sop == SemOp::V_PERMLANEX16_B32);
    const bool fi = (op.srcMod(0) & SISrcMods::OP_SEL_0) != 0;
    const bool bc = (op.srcMod(1) & SISrcMods::OP_SEL_0) != 0;
    if (!fi || bc) {
      // Empirically the GPT-OSS / softmax / bitmatrix corpora emit
      // `op_sel:[1, 0]` exclusively (fi=1, bc=0). Refuse any other
      // encoding loudly so a future corpus kernel's extended
      // fi/bc use surfaces during classifier verification rather
      // than producing an approximation silently. Re-narrowing this
      // gate is the right place to extend the emulation.
      std::string detail;
      raw_string_ostream os(detail);
      os << "permlane16 / permlanex16 emulation supports only "
            "op_sel:[1,0] (fi=1, bc=0); saw fi="
         << (fi ? 1 : 0) << ", bc=" << (bc ? 1 : 0);
      hr.failure = RaiseFailure::unsupportedShape(di, "VALU", detail);
      return hr;
    }
    Value *src0 = op.src(0);
    Value *sel1 = op.src(1);
    Value *sel2 = op.src(2);

    // Target-hardware lane id, wave-width-aware via emitLaneIdx, with
    // per-BB memoisation. Multiple permlane16 sites in the same BB
    // (e.g. butterfly reductions) reuse the single cached i32 instead
    // of re-emitting the mbcnt_lo / mbcnt_hi chain at each site —
    // LLVM's CSE would converge to the same end state, but the
    // pre-mem2reg IR stays smaller and lit-test-friendlier.
    Value *laneId = ctx.emitLaneIdx();

    // Group base (lane & ~0xF) and within-group index (lane & 0xF).
    Value *groupBase = ctx.B.CreateAnd(laneId, ctx.B.getInt32(~0xF), "pl_group");
    Value *within = ctx.B.CreateAnd(laneId, ctx.B.getInt32(0xF), "pl_within");

    // Pick the right 32-bit selector word based on within's high bit.
    Value *isHiHalf = ctx.B.CreateICmpUGE(within, ctx.B.getInt32(8), "pl_hi");
    Value *selWord = ctx.B.CreateSelect(isHiHalf, sel2, sel1, "pl_sel");

    // Extract the 4-bit nibble at position (within & 7) * 4.
    Value *withinLo = ctx.B.CreateAnd(within, ctx.B.getInt32(7), "pl_lo");
    Value *shiftAmt = ctx.B.CreateShl(withinLo, ctx.B.getInt32(2), "pl_shift");
    Value *shifted = ctx.B.CreateLShr(selWord, shiftAmt, "pl_shifted");
    Value *nibble = ctx.B.CreateAnd(shifted, ctx.B.getInt32(0xF), "pl_nibble");

    // For permlanex16, XOR the group base by 0x10 to swap adjacent groups.
    Value *srcGroup = isPermlaneX16
        ? ctx.B.CreateXor(groupBase, ctx.B.getInt32(0x10), "plx_group")
        : groupBase;
    Value *srcLaneAbs = ctx.B.CreateOr(srcGroup, nibble, "pl_src_lane");
    Value *byteAddr = ctx.B.CreateShl(srcLaneAbs, ctx.B.getInt32(2), "pl_addr");

    // Convergent: emit the bpermute outside any emitUnderExec diamond
    // so all hardware lanes participate. writeReg32 below wraps the
    // store for EXEC masking.
    Function *bperm = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::amdgcn_ds_bpermute);
    Value *result = ctx.B.CreateCall(
        bperm, {byteAddr, src0},
        isPermlaneX16 ? "permlanex16_emu" : "permlane16_emu");
    ctx.writeReg32(op.dst(), result);
    hr.handled = true;
    return hr;
  }

  // ---- v_permlane64_b32 ----
  // KNOWN LIMITATION — see the v_permlane64_b32 row in the
  // unrewritable table of hotswap/docs/wave-size-translation.md §7:
  // no wave32 analogue, so
  // the Phase 1.4.5 classifier refuses this op in any cross-wave
  // lift (it is taxonomised as FullWaveRotate / unrewritable). The
  // same-lane fallback here only runs in same-wave (wave64 → wave64)
  // translation, where a gfx1250 binary would not contain the op
  // anyway (gfx942 and earlier do not emit it). Keeping the stub
  // prevents a silent raise failure on the theoretical case.
  case SemOp::V_PERMLANE64_B32: {
    if (di.numDefs >= 1 && di.numSrcs >= 1)
      ctx.writeReg32(op.dst(), op.src(0));
    hr.handled = true;
    return hr;
  }

  // ---- gfx950 lane-swap: v_permlane16_swap_b32 ----
  //
  // P4 lowering — see the permlane16_swap row of hotswap/docs/wave-
  // size-translation.md §5.3. Exchanges two VGPRs across
  // lanes 0..15 ↔ 16..31 within each 32-lane group. Two defs
  // (vdst, src0_out) and two tied uses (vdst_in tied to vdst,
  // src0 tied to src0_out). Effect:
  //
  //   new_vdst[L]      = src0_in[L XOR 16]
  //   new_src0_out[L]  = vdst_in[L XOR 16]
  //
  // Target constraint mirrors P2: `v_permlane16_swap_b32` exists
  // natively only on gfx950 (CDNA4) and gfx12+ (HasPermlane16Swap
  // subtarget feature). gfx942 (CDNA3) and earlier wave64 targets
  // lack native isel for `llvm.amdgcn.permlane16.swap` — upstream
  // LLVM's `test/CodeGen/AMDGPU/llvm.amdgcn.permlane16.swap.ll`
  // explicitly asserts the gfx942 isel failure ("LLVM ERROR: Cannot
  // select: intrinsic %llvm.amdgcn.permlane16.swap"). Emulate via
  // `ds_bpermute_b32`, available on every AMDGPU generation with
  // LDS (gfx8+), so this lowering is target-independent.
  //
  // Per-lane emulation, L = `mbcnt`-derived absolute lane id:
  //
  //   partner   = L XOR 16
  //   bperm_idx = partner << 2          // ds_bpermute byte address
  //   new_vdst       = ds_bpermute(bperm_idx, src0_in)
  //   new_src0_out   = ds_bpermute(bperm_idx, vdst_in)
  //
  // Wave-width correctness under modulo-replication: lane L XOR 16
  // is naturally per-32-lane-half-independent on wave64 (lane 0 ↔
  // lane 16 in lower half, lane 32 ↔ lane 48 in upper half — both
  // halves swap internally). The wave32 source kernel's two-VGPR
  // exchange therefore lifts to a wave64 instruction that performs
  // the same exchange independently in each half, the textbook
  // modulo-replication match.
  //
  // CI regression gate: the `Gfx1250Gpu.Permlane16Swap` GTest
  // (tests/gfx1250_gpu_test.cpp) lifts the committed
  // `test_data/gfx1250/permlane16_swap_gfx1250.hsaco` (built from
  // `permlane16_swap_kernel.hip` — a wave32 source kernel using
  // `v_permlane16_swap_b32_e32` via inline asm) and runs it on
  // gfx942 wave64 hardware, verifying per-lane outputs match the
  // expected XOR-16 partner pattern across all 64 lanes:
  //
  //   For every L in [0, 64): new_vdst[L]      == 1000 + (L XOR 16)
  //                           new_src0_out[L]  == (L XOR 16)
  //
  // (vdst_in seeded with L, src0_in with 1000+L.) The test verifies
  // BOTH the per-lane swap pattern AND per-32-lane-half
  // independence (lanes 32..63 produce the lower-half result + 32
  // for both VGPRs). A future change that breaks the XOR-16
  // partner, the byte-address shift, the convergence semantics, or
  // the per-half independence would fail this test.
  //
  // Native gfx942 isel for `llvm.amdgcn.permlane16.swap` would fail
  // (per upstream LLVM's permlane16.swap.ll ERR-SDAG assertion), so
  // we cannot directly compare emulation-vs-native on this target;
  // probing other targets (e.g. gfx950 which has the native
  // instruction) is left for hardware-availability work and is the
  // P4.b sub-item recorded in hotswap/docs/wave-size-translation.md
  // §10 (known gaps). The emulation
  // independently maps onto the published .td swap semantics
  // (VOP_PERMLANE_SWAP profile), so per-target hardware-vs-
  // emulation parity follows from emulation-correctness +
  // .td-semantics.
  //
  // Emission details (operand-table lookup, snapshot ordering,
  // convergence, EXEC/fi/bc handling, P4.b future-hardening) are
  // documented on the shared `emitPermLaneSwapEmulation` helper at
  // the top of this file.  Both arms below call the helper with
  // their mask-specific arguments; the only per-arm code here is
  // the XOR-32 precondition check on wave size.
  //
  // V_PERMLANE32_SWAP_B32 (the wider variant) is the wave64-native
  // XOR-32 sibling.  Source kernels that emit it come from the
  // gfx950-and-later wave64 ISAs where `FeaturePermlane32Swap`
  // is enabled (the GFX950Insts feature block in
  // llvm/lib/Target/AMDGPU/AMDGPU.td).  Same-wave lifts for those
  // sources (gfx950 → gfx942) need an emulation because the gfx942
  // target does NOT enable `FeaturePermlane32Swap` — gfx940 base
  // features do not include it, so the instruction is unavailable
  // natively on the compilation target.  The helper's
  // `ds_bpermute(lane_id XOR 32, src)` emulation covers this on
  // wave64 → wave64; refusal is preserved for the narrowing
  // direction (wave64 source → wave32 target, which has no 64-lane
  // neighbourhood to XOR against) and for the impossible wave32-
  // source case (no wave32 ISA enables the feature, so seeing it
  // in wave32 source bytes indicates either a corrupted disassembly
  // or a wave64 source mis-classified as wave32).
  //
  // CI regression gates:
  //   * `Gfx1250Gpu.Permlane16Swap` (tests/gfx1250_gpu_test.cpp)
  //     runs the XOR-16 emulation on gfx942 hardware and verifies
  //     per-lane outputs against the expected XOR-16 partner
  //     pattern across all 64 lanes.  A regression in the helper
  //     (wrong XOR mask, missing byte-address shift, cross-wiring
  //     error) fails this test before reaching a user.
  //   * `BatchRaise.AiterGfx950` (tests/batch_raise_test.cpp) lifts
  //     3 AITER `fmha_v3_fwd/fwd_hd128_bf16*` kernels whose
  //     reduction cores depend on `v_permlane32_swap_b32`; a
  //     regression on the helper surfaces here as a lift failure.
  //     IR-shape coverage lives in `lit_tests/c2_permlane_swap/`
  //     (XOR-16) and `lit_tests/v_permlane32_swap_b32/` (XOR-32).
  case SemOp::V_PERMLANE16_SWAP_B32:
    return emitPermLaneSwapEmulation(ctx, di, op, /*partnerXorMask=*/16,
                                      /*ssaPrefix=*/"pls16");
  case SemOp::V_PERMLANE32_SWAP_B32: {
    // Two precondition checks, both specific to the wider XOR-32
    // variant:
    //
    //   1. target wave32 → `laneId ^ 32` wraps past the target
    //      wave; `ds_bpermute` cannot deliver a lane index >=
    //      target wave size.
    //   2. source wave32 → no wave32 ISA enables
    //      `FeaturePermlane32Swap` (see the GFX950Insts block in
    //      `llvm/lib/Target/AMDGPU/AMDGPU.td`), so seeing the
    //      instruction in wave32 source bytes means corrupted
    //      disassembly or upstream wave-size mis-classification.
    //
    // Refusal in both cases preserves the "refuse when uncertain"
    // contract documented on the P4 pending row of
    // hotswap/docs/wave-size-translation.md §5.3; the wave64 →
    // wave64 path below is the positive case this handler now
    // lifts.  The XOR-16 sibling has no equivalent precondition
    // because its partner stays within each 32-lane half
    // regardless of wave size.
    if (ctx.targetIsa.isWave32()) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "VALU",
          "v_permlane32_swap_b32 lift refused: target is wave32, "
          "but the instruction's XOR-32 partner has no wave32 "
          "analogue (the partner index wraps past the target "
          "wave).  See the P4 permlane32_swap entry in the "
          "pending-rewrite table of "
          "hotswap/docs/wave-size-translation.md \u00a75.3.");
      return hr;
    }
    if (ctx.isa.isWave32()) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "VALU",
          "v_permlane32_swap_b32 in a wave32 source kernel — no "
          "wave32 ISA enables FeaturePermlane32Swap (see "
          "llvm/lib/Target/AMDGPU/AMDGPU.td), so this is either a "
          "corrupted disassembly or a wave64 source mis-classified "
          "as wave32 upstream.  Refusing rather than silently "
          "emitting an XOR-32 partner that cannot exist in the "
          "source's wave topology.");
      return hr;
    }
    return emitPermLaneSwapEmulation(ctx, di, op, /*partnerXorMask=*/32,
                                      /*ssaPrefix=*/"pls32");
  }

  // ---- v_readfirstlane_b32 sDST, vSRC ----
  // Broadcast the value of vSRC from the lowest-numbered active lane
  // (or lane 0 if EXEC==0) to sDST. A plain per-lane move would leave
  // each lane with its own vSRC, breaking the uniformity invariant
  // that downstream consumers (s_mov, s_load base, branch condition)
  // rely on. Emit the native `llvm.amdgcn.readfirstlane.i32`; the
  // SGPR dst is NOT wrapped in emitUnderExec (see
  // `RaiseContext::writeReg32` — SGPR writes bypass predication), so
  // the broadcast lands on all lanes.
  case SemOp::V_READFIRSTLANE_B32: {
    Function *rfl = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::amdgcn_readfirstlane, {ctx.i32Ty});
    Value *src = ctx.B.CreateZExtOrTrunc(op.src(0), ctx.i32Ty, "rfl_src");
    Value *val = ctx.B.CreateCall(rfl, {src}, "readfirstlane");
    ctx.writeReg32(op.dst(), val);
    hr.handled = true;
    return hr;
  }

  // ---- v_writelane_b32 ----
  // Write `val` into lane `lane` of vDst. Cross-lane: cannot be
  // emulated via per-thread private scratch nor via a single scalar
  // SSA value. `llvm.amdgcn.writelane(val, lane, old)` lowers to the
  // hardware primitive; the intrinsic returns the new per-lane scalar
  // (either `val` when lane_id==lane, else `old`), so the VGPR's
  // SSA slot carries the correct value for whichever lane we are.
  //
  // First-write pattern: if writelane is the first assignment to
  // vDst, non-selected lanes legitimately hold whatever vDst
  // contained before (hardware semantics). `readReg32` on the
  // never-stored alloca returns LLVM `undef`, which is the right
  // "unobservable" encoding — any downstream use of those lanes
  // before they are written is itself undefined on hardware.
  case SemOp::V_WRITELANE_B32: {
    ParsedReg dst = op.dst();
    Value *val = op.src(0);
    Value *lane = op.src(1);
    lane = ctx.B.CreateZExtOrTrunc(lane, ctx.i32Ty, "wrlane_idx");
    Value *oldVal = ctx.regs.readReg32(ctx.B, dst);
    Function *wl = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::amdgcn_writelane, {ctx.i32Ty});
    Value *newVal = ctx.B.CreateCall(wl, {val, lane, oldVal}, "writelane");
    ctx.writeReg32(dst, newVal);
    hr.handled = true;
    return hr;
  }

  // ---- v_readlane_b32 sDST, vSRC, lane ----
  // Read a specific lane of vSRC into an SGPR. Reverse of writelane;
  // cross-lane so must use the native intrinsic.
  case SemOp::V_READLANE_B32: {
    ParsedReg srcReg = op.srcReg(0);
    Value *lane = op.src(1);
    lane = ctx.B.CreateZExtOrTrunc(lane, ctx.i32Ty, "rdlane_idx");
    Value *src = ctx.regs.readReg32(ctx.B, srcReg);
    Function *rl = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::amdgcn_readlane, {ctx.i32Ty});
    Value *val = ctx.B.CreateCall(rl, {src, lane}, "readlane");
    ctx.writeReg32(op.dst(), val);
    hr.handled = true;
    return hr;
  }

  // ---- v_mbcnt_lo_u32_b32 / v_mbcnt_hi_u32_b32 ----
  // Count set bits in src0 below the current lane. Wave-size-aware
  // (mbcnt.hi only meaningful on wave64). These are the building
  // blocks for lane-id derivation that `WaveProjection::emitLaneIdx`
  // also uses; the raw intrinsics must be passed through exactly.
  case SemOp::V_MBCNT_LO_U32_B32: {
    Function *mbcnt = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::amdgcn_mbcnt_lo, {});
    ctx.writeReg32(op.dst(),
                   ctx.B.CreateCall(mbcnt, {op.src(0), op.src(1)},
                                    "mbcnt_lo"));
    hr.handled = true;
    return hr;
  }
  case SemOp::V_MBCNT_HI_U32_B32: {
    Function *mbcnt = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::amdgcn_mbcnt_hi, {});
    ctx.writeReg32(op.dst(),
                   ctx.B.CreateCall(mbcnt, {op.src(0), op.src(1)},
                                    "mbcnt_hi"));
    hr.handled = true;
    return hr;
  }

  default:
    break;
  }
  return hr;
}

} // namespace transpiler
