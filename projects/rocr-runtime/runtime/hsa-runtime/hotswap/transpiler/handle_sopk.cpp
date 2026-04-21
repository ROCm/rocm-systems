#include "handlers.hpp"
#include "pipeline.hpp" // isStrictMode()

#include "SIDefines.h" // AMDGPU::Hwreg::Id
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace transpiler {

// HWREG (s_getreg / s_setreg) policy — direction-aware.
//
// The simm16 encoding for these opcodes is
// `id[5:0] | offset[10:6] | size_minus_1[15:11]`; only the id field
// selects which hardware register is being touched. EXEC is not a HWREG
// id (confirmed in AMDGPU::Hwreg::Id), so s_setreg can never reach EXEC;
// the SPE abort-gate in raiser.cpp remains exhaustive for EXEC writers.
//
// Two axes of decision per id:
//
//  * READ policy (`HwregRead`):
//      Zero  — produce 0. Used for ids whose observable value is either
//              a hardware timer / state we can't reproduce but whose 0
//              is the architectural default (MODE) or a diagnostic read
//              the kernel's compute does not meaningfully act on
//              (STATUS, HW_ID, perf counters, shader cycles, ALLOC
//              descriptors, TBA/TMA/trap-handler bases, etc.).
//      Abort — refuse to lower. Used for ids whose value the kernel
//              *might* actually consume to make compute decisions we
//              cannot reproduce faithfully (FLAT_SCR_*, MEM_BASES,
//              XNACK_MASK). Zero would be a silent lie.
//
//  * WRITE policy (`HwregWrite`):
//      Drop     — silently discard. Used for ids whose observable
//                 state the transpiler does not model downstream
//                 (diagnostic / status / perf), so dropping a write
//                 leaves no gap in semantics SPE lifts.
//      WarnDrop — discard but emit a stderr warning. Used for ids
//                 where a write *could* change observable compute
//                 semantics in the general case but where silently
//                 aborting would break the current test corpus
//                 (compiler-emitted MODE writes for FP flags that
//                 downstream ops happen not to consume in
//                 practice). Symmetric with the cross-wave
//                 warn-only policy. Tightening to Abort requires
//                 either a same-wave regression corpus or a proof
//                 that the downstream FP ops are insensitive to the
//                 discarded bits.
//      Abort    — refuse to lower. Used for ids whose value *does*
//                 feed state we lift and where dropping would
//                 change lifted semantics: FLAT aperture
//                 (FLAT_SCR_*, MEM_BASES), XNACK retry semantics,
//                 trap handler bases.
//
// Unknown ids (not enumerated below) default to (Abort, Abort): the
// principled fail-closed choice. A future ISA adding a new load-
// bearing register is trapped loudly instead of silently miscompiling.
enum class HwregRead { Zero, Abort };
enum class HwregWrite { Drop, WarnDrop, Abort };

struct HwregPolicy {
  HwregRead read;
  HwregWrite write;
};

static HwregPolicy classifyHwreg(unsigned id) {
  using namespace AMDGPU::Hwreg;
  switch (id) {
  // --- Load-bearing: both directions abort ---------------------------------
  // FLAT_SCR is the aperture base for FLAT_* instructions; MEM_BASES sets
  // the scratch/private apertures; XNACK_MASK governs per-lane page-fault
  // retry. Reads cannot be faithfully reproduced (our IR does not carry
  // aperture config), writes must not be dropped (they would corrupt
  // addressing for subsequent FLAT ops we *do* lift).
  case ID_FLAT_SCR_LO:
  case ID_FLAT_SCR_HI:
  case ID_MEM_BASES:
  case ID_XNACK_MASK:
    return {HwregRead::Abort, HwregWrite::Abort};

  // --- MODE: read zero, write warn-and-drop -------------------------------
  // Reading MODE as 0 returns the architectural default (IEEE, round-
  // to-nearest-even, no FP exception flags, no FTZ). Writing MODE
  // mutates FP rounding / FTZ / clamp / IEEE / various graphics-
  // context bits. HIP-compiled compute kernels routinely emit
  // `s_setreg_imm32_b32 mode(offset, size), imm` as part of their
  // prologue (e.g. setting bit 25 to 1) where `offset` and `size`
  // target bits whose effect on downstream compute is practically
  // nil for the inputs tests use. Aborting on these would block the
  // current cross-wave regression corpus; silently dropping would be
  // a hidden latent miscompile for FP-mode-sensitive kernels. Emit a
  // loud warning and continue, matching the cross-wave warn-only
  // policy in raiser.cpp. Tightening to Abort is a follow-up.
  case ID_MODE:
    return {HwregRead::Zero, HwregWrite::WarnDrop};

  // --- Trap handler bases: read-as-zero, write-abort -----------------------
  // TBA_LO/HI and TMA_LO/HI set the trap-handler base / trap-memory
  // address. SPE doesn't model trap handlers, so a dropped write would
  // lose information we have no other channel to surface. Aborting
  // forces us to address this explicitly when a real kernel writes
  // them; reads return zero because compiler-emitted probes exist and
  // their result does not feed compute.
  case ID_TBA_LO:
  case ID_TBA_HI:
  case ID_TMA_LO:
  case ID_TMA_HI:
    return {HwregRead::Zero, HwregWrite::Abort};

  // --- Diagnostic / read-dominant: read zero, drop writes ------------------
  // Status, HW_ID*, allocation descriptors (set by the runtime at kernel
  // launch, not by compute), instruction-buffer status, performance
  // counters, shader cycle counters, packer / scheduling hints,
  // privileged status / exception / XCC ids. None of these feed compute
  // SPE lifts; reading zero is a defensible default; writes (if any
  // compiler ever emits one) are lost state we do not track.
  case ID_STATUS:
  case ID_TRAPSTS:
  case ID_HW_ID:
  case ID_HW_ID1:
  case ID_HW_ID2:
  case ID_GPR_ALLOC:
  case ID_LDS_ALLOC:
  case ID_IB_STS:
  case ID_IB_STS2:
  case ID_POPS_PACKER:
  case ID_SCHED_MODE:
  case ID_PERF_SNAPSHOT_DATA_gfx11:
  case ID_SHADER_CYCLES:
  case ID_SHADER_CYCLES_HI:
  case ID_DVGPR_ALLOC_LO:
  case ID_DVGPR_ALLOC_HI:
    return {HwregRead::Zero, HwregWrite::Drop};

  // --- Unknown: fail-closed in both directions -----------------------------
  default:
    return {HwregRead::Abort, HwregWrite::Abort};
  }
}

// Extract the hwreg id from a s_setreg/s_getreg simm16 operand.
static unsigned extractHwregId(int64_t simm16) {
  return static_cast<unsigned>(simm16 & 0x3f);
}

HandlerResult handleSOPK(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op) {
  HandlerResult hr;
  SemOp sop = di.semOp;

  // SOPK format: dst = SDST, src(0) = sign-extended 16-bit imm
  if (sop == SemOp::S_MOVK_I32) {
    ctx.regs.writeReg32(ctx.B, op.dst(), op.src(0));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_MULK_I32) {
    Value *dst = ctx.regs.readReg32(ctx.B, op.dst());
    ctx.regs.writeReg32(ctx.B, op.dst(),
                        ctx.B.CreateMul(dst, op.src(0), "mulk"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_ADDK_I32) {
    Value *dst = ctx.regs.readReg32(ctx.B, op.dst());
    Value *imm = op.src(0);
    Value *res = ctx.B.CreateAdd(dst, imm, "addk");
    ctx.regs.writeReg32(ctx.B, op.dst(), res);
    auto *ov = ctx.B.CreateIntrinsic(Intrinsic::uadd_with_overflow, {ctx.i32Ty},
                                     {dst, imm});
    ctx.regs.storeSCC(ctx.B, ctx.B.CreateExtractValue(ov, 1));
    hr.sccHandled = true;
    hr.handled = true;
    return hr;
  }
  // SOPK compares: s_cmpk_XX_i32 / s_cmpk_XX_u32
  if (sop == SemOp::S_CMPK_EQ_I32 || sop == SemOp::S_CMPK_EQ_U32 ||
      sop == SemOp::S_CMPK_LG_I32 || sop == SemOp::S_CMPK_LG_U32 ||
      (sop >= SemOp::S_CMPK_GE_I32 && sop <= SemOp::S_CMPK_LT_U32)) {
    Value *sdst = ctx.regs.readReg32(ctx.B, op.dst());
    Value *imm = op.src(0);
    Value *cmp = nullptr;
    if (sop == SemOp::S_CMPK_EQ_I32 || sop == SemOp::S_CMPK_EQ_U32)
      cmp = ctx.B.CreateICmpEQ(sdst, imm, "scmpk");
    else if (sop == SemOp::S_CMPK_LG_I32 || sop == SemOp::S_CMPK_LG_U32)
      cmp = ctx.B.CreateICmpNE(sdst, imm, "scmpk");
    else if (sop == SemOp::S_CMPK_GT_I32)
      cmp = ctx.B.CreateICmpSGT(sdst, imm, "scmpk");
    else if (sop == SemOp::S_CMPK_GE_I32)
      cmp = ctx.B.CreateICmpSGE(sdst, imm, "scmpk");
    else if (sop == SemOp::S_CMPK_LT_I32)
      cmp = ctx.B.CreateICmpSLT(sdst, imm, "scmpk");
    else if (sop == SemOp::S_CMPK_LE_I32)
      cmp = ctx.B.CreateICmpSLE(sdst, imm, "scmpk");
    else if (sop == SemOp::S_CMPK_GT_U32)
      cmp = ctx.B.CreateICmpUGT(sdst, imm, "scmpk");
    else if (sop == SemOp::S_CMPK_GE_U32)
      cmp = ctx.B.CreateICmpUGE(sdst, imm, "scmpk");
    else if (sop == SemOp::S_CMPK_LT_U32)
      cmp = ctx.B.CreateICmpULT(sdst, imm, "scmpk");
    else if (sop == SemOp::S_CMPK_LE_U32)
      cmp = ctx.B.CreateICmpULE(sdst, imm, "scmpk");
    if (cmp) {
      ctx.regs.storeSCC(ctx.B, cmp);
      hr.sccHandled = true;
      hr.handled = true;
      return hr;
    }
  }
  // s_getreg_b32 / s_setreg_*: read or write one field of a hardware
  // configuration register. Both encodings carry the HWREG selector as a
  // simm16 at MCInst operand index 1 (TableGen `ins` order is
  // `sdst, simm16` for GETREG/SETREG_B32 and `imm, simm16` for
  // SETREG_IMM32_B32). Policy is driven by `classifyHwreg` above —
  // direction-aware per id. See the classifier's docstring for the
  // rationale behind each entry.
  if (sop == SemOp::S_GETREG_B32 || sop == SemOp::S_SETREG_B32 ||
      sop == SemOp::S_SETREG_IMM32_B32) {
    const unsigned simm16OpIdx = 1;
    if (simm16OpIdx >= di.numOps() || !di.isImm(simm16OpIdx)) {
      errs() << "transpiler: " << di.mnemonic
             << " has unexpected operand layout (simm16 not at op 1) — "
                "refusing to model it silently.\n";
      return hr;
    }
    unsigned hwregId = extractHwregId(di.getImm(simm16OpIdx));
    HwregPolicy policy = classifyHwreg(hwregId);

    if (sop == SemOp::S_GETREG_B32) {
      if (policy.read == HwregRead::Abort) {
        errs() << "transpiler: " << di.mnemonic
               << " reads load-bearing or unknown HWREG id=" << hwregId
               << " — refusing to lower. The transpiler does not carry "
                  "this register's value and returning zero would be a "
                  "silent lie that the kernel's compute may act on.\n";
        return hr;
      }
      ctx.regs.writeReg32(ctx.B, op.dst(), ConstantInt::get(ctx.i32Ty, 0));
      hr.handled = true;
      return hr;
    }

    // S_SETREG path (both S_SETREG_B32 and S_SETREG_IMM32_B32).
    if (policy.write == HwregWrite::Abort) {
      errs() << "transpiler: " << di.mnemonic
             << " writes load-bearing or unknown HWREG id=" << hwregId
             << " — refusing to lower. Dropping this write would silently "
                "change compute semantics (FLAT aperture, trap handler, "
                "XNACK retry, …) that subsequent lifted instructions "
                "rely on.\n";
      return hr;
    }
    if (policy.write == HwregWrite::WarnDrop) {
      // Strict mode (`HSA_SALMON_STRICT=1`, see pipeline.hpp) refuses
      // these writes structurally instead of warn-and-continue. The
      // refusal is the honest answer for any caller (e.g. the corpus
      // runner) that wants UNSUPPORTED verdicts in place of latent
      // wrong-result hazards. The non-strict path stays unchanged so
      // existing GPU tests that emit `s_setreg_imm32_b32 mode, imm`
      // and pass bit-exactly continue to pass.
      if (isStrictMode()) {
        errs() << "transpiler: " << di.mnemonic
               << " writes HWREG id=" << hwregId
               << " (MODE / FP-state-bearing register) — refusing under "
                  "HSA_SALMON_STRICT. Dropping the write would silently "
                  "change FP rounding / denormal / IEEE / FTZ semantics if "
                  "downstream compute consumes those bits.\n";
        hr.failure = RaiseFailure::strictUnsafeLowering(
            di, "HWREG_MODE_write",
            "salmon (strict): MODE-register write would be silently "
            "dropped; kernel may rely on FP rounding / denormal / IEEE / "
            "FTZ bits being changed");
        return hr;
      }
      errs() << "transpiler: WARNING: " << di.mnemonic
             << " writes HWREG id=" << hwregId
             << " (MODE or similar FP-state-bearing register). Dropping "
                "the write to stay within the existing SPE lifting shape; "
                "this is observationally correct only when downstream "
                "compute is insensitive to the written bits. Tighten to "
                "Abort once the corpus allows it.\n";
    }
    hr.handled = true;
    return hr;
  }
  return hr;
}

} // namespace transpiler
