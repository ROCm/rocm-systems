#include "wave_size_obstruction.hpp"

#include "decoded_inst.hpp"
#include "isa_profile.hpp"
#include "mc_state.hpp"
#include "semop.hpp"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h" // AMDGPU::OpName, AMDGPU::TTMP_32RegClassID, AMDGPU::mc2PseudoReg
#include "Utils/AMDGPUBaseInfo.h"             // AMDGPU::getNamedOperandIdx
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

// SIInstrFlags::DPP / SDWA bits live in the AMDGPU target's SIDefines
// header. We only use the flag constant; no other target dependency.
#include "SIDefines.h"

namespace transpiler {

using namespace llvm;

// ----------------------------------------------------------------------------
// Taxonomy rendering. The text in each branch is the label that
// surfaces in the classifier trace and that lit tests assert on. The
// obstruction-class number (Class 1..4, see hotswap/docs/wave-size-
// translation.md §6) is included parenthetically so operators reading
// the trace can cross-reference the spec without mental translation.
// ----------------------------------------------------------------------------

const char *obstructionKindName(ObstructionKind k) {
  switch (k) {
  case ObstructionKind::None:
    return "None";
  case ObstructionKind::MbcntHiLaneIdLeak:
    return "MbcntHiLaneIdLeak (\u00a73 Class 1: absolute lane-ID leak via v_mbcnt_hi)";
  case ObstructionKind::OutOfRangeLaneOperand:
    return "OutOfRangeLaneOperand (\u00a73 Class 1: readlane/writelane operand >= W_s)";
  case ObstructionKind::TtmpWaveIdLeak:
    return "TtmpWaveIdLeak (\u00a73 Class 1: source read of ttmp8 under cross-widening — wave_id_in_wg field)";
  case ObstructionKind::WaveIdLiftScalarized:
    return "WaveIdLiftScalarized (\u00a73 Class 1: canonical wave_id BFE lift + v_writelane/v_readlane + WMMA — cross-lane primitive scalarises the divergent lift, collapsing per-source-wave distinction)";
  case ObstructionKind::FullWaveRotate:
    return "FullWaveRotate (\u00a73 Class 2: unrewritable v_permlane64)";
  case ObstructionKind::LaneGroupShuffle:
    return "LaneGroupShuffle (\u00a73 Class 2: permlane16 / permlanex16 / permlane*_swap)";
  case ObstructionKind::DsSwizzle:
    return "DsSwizzle (\u00a73 Class 2: ds_swizzle_b32)";
  case ObstructionKind::DppCrossLane:
    return "DppCrossLane (\u00a73 Class 2: DPP modifier)";
  case ObstructionKind::DsBpermuteGather:
    return "DsBpermuteGather (\u00a73 Class 2: ds_bpermute_b32)";
  case ObstructionKind::NonCommutativeAtomic:
    return "NonCommutativeAtomic (\u00a73 Class 3: cmpswap/swap/xchg, replica race)";
  case ObstructionKind::CmpxFromLaneId:
    return "CmpxFromLaneId (\u00a73 Class 4: lane-predicated v_cmpx)";
  case ObstructionKind::SaveExecFromLaneId:
    return "SaveExecFromLaneId (\u00a73 Class 4: lane-predicated s_*_saveexec_b32)";
  }
  return "UnknownObstructionKind";
}

const char *rewriteIdName(RewriteId r) {
  switch (r) {
  case RewriteId::None:
    return "none";
  case RewriteId::P1_DsBpermute:
    return "P1 (llvm.amdgcn.ds.bpermute)";
  case RewriteId::P2_PermLane16:
    return "P2 (llvm.amdgcn.permlane16)";
  case RewriteId::P3_PermLane64:
    return "P3 (reserved: v_permlane64 has no rewrite)";
  case RewriteId::P4_PermLaneSwap:
    return "P4 (permlane*_swap via LDS round-trip or permlane16 pair)";
  case RewriteId::P5_DppModifier:
    return "P5 (llvm.amdgcn.update.dpp)";
  case RewriteId::P6_DsSwizzle:
    return "P6 (llvm.amdgcn.ds.swizzle)";
  case RewriteId::LaneOpBoundsValidator:
    return "raise-time readlane/writelane bounds validator";
  case RewriteId::PostRaiseCrossLaneRewrite:
    return "post-raise cross-lane rewrite (writelane -> select, "
           "readlane -> ds.bpermute)";
  }
  return "UnknownRewriteId";
}

// ----------------------------------------------------------------------------
// ObstructionReport queries.
// ----------------------------------------------------------------------------

bool ObstructionReport::hasUnrewritable() const {
  for (const auto &s : sites)
    if (s.rewrite == RewriteId::None)
      return true;
  return false;
}

bool ObstructionReport::hasPendingRewrite() const {
  for (const auto &s : sites)
    if (s.rewrite != RewriteId::None && !s.rewriteImplemented)
      return true;
  return false;
}

bool ObstructionReport::isOblivious() const {
  for (const auto &s : sites)
    if (!s.rewriteImplemented)
      return false;
  return true;
}

const ObstructionSite *ObstructionReport::firstUnrewritable() const {
  for (const auto &s : sites)
    if (s.rewrite == RewriteId::None)
      return &s;
  return nullptr;
}

const ObstructionSite *ObstructionReport::firstPending() const {
  for (const auto &s : sites)
    if (s.rewrite != RewriteId::None && !s.rewriteImplemented)
      return &s;
  return nullptr;
}

// ----------------------------------------------------------------------------
// Classification primitives.
// ----------------------------------------------------------------------------

namespace {

// Extract the lane operand of a v_readlane / v_writelane instruction.
// Returns std::nullopt when the operand is register-typed (dynamic) or
// when LLVM's named-operand index lookup fails (e.g. on a future LLVM
// version that renames `src1`).
//
// We use `AMDGPU::getNamedOperandIdx(opcode, AMDGPU::OpName::src1)`
// which is the authoritative way to find a named MCInst operand and
// is robust to operand-layout reordering between LLVM versions.
// Implementation note: both v_readlane_b32 (dst, src0, src1) and
// v_writelane_b32 (vdst, src0, src1, vdst_in) name the lane operand
// `src1`, so a single getNamedOperandIdx call covers both.
std::optional<int64_t> extractLaneOperandImm(const DecodedInst &di) {
  const MCInst &inst = di.inst;
  int idx = AMDGPU::getNamedOperandIdx(inst.getOpcode(), AMDGPU::OpName::src1);
  if (idx < 0 || static_cast<unsigned>(idx) >= inst.getNumOperands())
    return std::nullopt;
  const MCOperand &op = inst.getOperand(idx);
  if (op.isImm())
    return op.getImm();
  return std::nullopt; // dynamic SGPR operand — cannot statically prove range
}

// Decide whether a `ds_swizzle_b32` immediate encodes a swizzle mode
// that is *structurally* wave-size-oblivious under modulo-replication
// (the projection-ladder's first rung, see hotswap/docs/wave-size-
// translation.md §2.2).
//
// The 16-bit imm encodes one of seven modes (SIDefines.h
// `Swizzle::Id`). Per AMDGPU SIDefines.h `Swizzle::EncBits`:
//
//   QUAD_PERM_ENC    == 0x8000, QUAD_PERM_ENC_MASK    == 0xFF00
//   BITMASK_PERM_ENC == 0x0000, BITMASK_PERM_ENC_MASK == 0x8000
//   FFT_MODE_ENC     == 0xE000  (FFT_ROTATE_MODE_MASK  == 0xF000)
//   ROTATE_MODE_ENC  == 0xC000  (FFT_ROTATE_MODE_MASK  == 0xF000)
//
// Within each envelope, only specific sub-encodings are *valid*:
//
//   QUAD_PERM     — bits 0..7  encode four 2-bit lane selectors.
//                   All 256 imms in [0x8000, 0x80FF] are valid.
//
//   BITMASK_PERM  — bits 0..4  AND mask (0..31)
//                   bits 5..9  OR mask  (0..31)
//                   bits 10..14 XOR mask (0..31)
//                   bit 15      = 0 (envelope discriminator)
//                   All 32K imms in [0x0000, 0x7FFF] are valid.
//
//   FFT_MODE      — bits 0..4   FFT_SWIZZLE_MASK (0..31)
//                   bits 5..11  reserved, MUST be 0
//                   bits 12..15 = 0xE (envelope discriminator)
//                   Valid imms: exactly the 32 in [0xE000, 0xE01F].
//
//   ROTATE_MODE   — bits 0..4   reserved, MUST be 0
//                   bits 5..9   ROTATE_SIZE_MASK (0..31)
//                   bit  10     ROTATE_DIR_MASK  (0|1)
//                   bit  11     reserved, MUST be 0
//                   bits 12..15 = 0xC (envelope discriminator)
//                   Valid imms: exactly the 64 with bits 0..4=0,
//                   bit 11=0, top nibble=0xC.
//
// All four valid-encoding sub-spaces are wave-size-oblivious under
// modulo-replication. The argument is the same for all four: wave64
// ds_swizzle hardware preserves bit 5 of the lane id, so each
// 32-lane half independently performs the same permutation.
// Preservation verified empirically on gfx942 (CDNA3) wave64 across
// the four envelopes (see the MODREP block in handle_ds.cpp for the
// per-envelope probe results and the LLVM-upstream-test cross-
// citation that lifts the property to the wave64 GPU family).
//
// REFUSED imms (any of):
//   * RESERVED top-nibble envelopes (top nibble in
//     {0x9, 0xA, 0xB, 0xD, 0xF}) — `Swizzle::EncBits` assigns no
//     semantics; hardware behavior is undefined.
//   * FFT_MODE imms with reserved bits 5..11 set — within the FFT
//     envelope but outside the valid sub-encoding space; hardware
//     behavior likewise undefined.
//   * ROTATE_MODE imms with reserved bits 0..4 or 11 set — same
//     argument.
//
// Refusing these (rather than silently passing them through to
// `llvm.amdgcn.ds.swizzle`, which would emit `ds_swizzle_b32
// offset:<imm>` and let the wave64 hardware do whatever it does
// for an undefined imm) matches the no-fallback contract.
//
// Check ordering note: the four valid-encoding checks are pairwise
// disjoint on the discriminator bits (bit 15 separates QUAD_PERM /
// BITMASK_PERM; the top nibble separates FFT_MODE / ROTATE_MODE
// from QUAD_PERM and from each other), so the order between them
// is irrelevant for correctness. The implicit "no match → return
// false" branch correctly catches all the REFUSED categories.
bool dsSwizzleSafeForModRep(uint16_t imm) {
  using namespace AMDGPU::Swizzle;
  if ((imm & QUAD_PERM_ENC_MASK) == QUAD_PERM_ENC)
    return true;
  if ((imm & BITMASK_PERM_ENC_MASK) == BITMASK_PERM_ENC)
    return true;
  // FFT_MODE: discriminator (top nibble = 0xE) AND every reserved
  // bit (5..11) clear. Equivalent to: imm & ~FFT_SWIZZLE_MASK ==
  // FFT_MODE_ENC, with the mask cast to uint16_t to keep the
  // bitwise-not within the 16-bit envelope (otherwise `~uint16_t`
  // promotes to int and would set bits 16..31 of the comparison
  // operand).
  if ((imm & static_cast<uint16_t>(~FFT_SWIZZLE_MASK)) == FFT_MODE_ENC)
    return true;
  // ROTATE_MODE: discriminator (top nibble = 0xC) AND only the size
  // (bits 5..9) and direction (bit 10) bits set. Variable-bit mask
  // is (ROTATE_SIZE_MASK << ROTATE_SIZE_SHIFT) | (ROTATE_DIR_MASK
  // << ROTATE_DIR_SHIFT) = 0x7E0; everything else (including
  // reserved bits 0..4 and 11) must match ROTATE_MODE_ENC exactly.
  constexpr uint16_t ROTATE_VAR_MASK =
      (ROTATE_SIZE_MASK << ROTATE_SIZE_SHIFT) |
      (ROTATE_DIR_MASK << ROTATE_DIR_SHIFT);
  if ((imm & static_cast<uint16_t>(~ROTATE_VAR_MASK)) == ROTATE_MODE_ENC)
    return true;
  // RESERVED top-nibble envelope, FFT/ROTATE with reserved bits
  // set, or any other non-valid encoding: refuse.
  return false;
}

// Return true iff any source-operand register of `di` covers the 32-bit
// TTMP8 lane — either as a bare `ttmp8` or as part of a larger tuple
// (e.g. `ttmp[8:9]`). Defs are skipped: only reads of TTMP8 constitute a
// wave_id leak. The surrounding `(src.waveSize != tgt.waveSize)` check
// in the caller gates this to cross-widening only.
//
// Detection strategy (TableGen-authoritative, no enum arithmetic):
//   1. Find the TTMP_32 register class — its register at position 8 IS
//      the generation-agnostic pseudo for `ttmp8` (the class is declared
//      `(add (sequence "TTMP%u", 0, 15))` in SIRegisterInfo.td so
//      position == index).
//   2. For every source reg operand (index >= di.numDefs), normalise
//      each 32-bit sub-register via `mc2PseudoReg` (strips subtarget
//      suffixes such as `TTMP8_gfx9plus` → `TTMP8`) and compare against
//      the pseudo from step 1. A tuple like `ttmp[8:9]` contributes
//      sub0 = TTMP8, sub1 = TTMP9 — we match on sub0.
bool readsTtmp8Source(const DecodedInst &di, const MCRegisterInfo &MRI) {
  const MCRegisterClass &TTMP32 =
      MRI.getRegClass(AMDGPU::TTMP_32RegClassID);
  MCRegister ttmp8Pseudo = TTMP32.getRegister(8);
  const llvm::MCInst &inst = di.inst;
  for (unsigned i = di.numDefs, e = inst.getNumOperands(); i < e; ++i) {
    const MCOperand &op = inst.getOperand(i);
    if (!op.isReg())
      continue;
    MCRegister reg = op.getReg();
    if (!reg)
      continue;
    // Walk 32-bit sub-registers. A 32-bit TTMP lane has no sub0 and
    // mc2PseudoReg normalises it directly; a TTMP pair (`ttmp[8:9]`)
    // has sub0 = TTMP8_aliased / sub1 = TTMP9_aliased and we match on
    // the sub0 lane.
    MCRegister lane = MRI.getSubReg(reg, AMDGPU::sub0);
    if (!lane)
      lane = reg;
    if (AMDGPU::mc2PseudoReg(lane) == ttmp8Pseudo)
      return true;
    // Also check sub1..subN in case TTMP8 appears in the upper half of a
    // pair that starts earlier (unusual but possible in tuple-aligned
    // encodings).
    const unsigned maxSubIdx = MRI.getNumSubRegIndices();
    for (unsigned subIdx = AMDGPU::sub1; subIdx < maxSubIdx; ++subIdx) {
      MCRegister s = MRI.getSubReg(reg, subIdx);
      if (!s)
        break;
      if (AMDGPU::mc2PseudoReg(s) == ttmp8Pseudo)
        return true;
    }
  }
  return false;
}

// Return true iff `di` is the canonical gfx1250 HIP-emitted wave_id
// extraction pattern: `s_bfe_u32 sDST, ttmp8, 0x50019`. The immediate
// encodes (offset=25, width=5), which extracts bits [29:25] of ttmp8
// — the command processor's `wave_id_in_workgroup` field.
//
// Pairs with the handle_sop2.cpp `S_BFE_U32` pattern-lift that emits
// `dst = (workitem.id.x >> log2(W_s)) & 0x1F` for this exact shape.
// The lift gives the BFE a principled meaning under cross-widening
// (each target lane gets its source-wave rank as a divergent VGPR),
// so we must NOT refuse on this shape here. Any OTHER ttmp8 read
// falls through to the `ttmp8ReadSites` path below.
bool isCanonicalWaveIdBfe(const DecodedInst &di,
                           const MCRegisterInfo &MRI) {
  if (di.semOp != SemOp::S_BFE_U32)
    return false;
  // S_BFE_U32 canonically has one destination (sDST), one source reg
  // (sSRC or ttmp), and one immediate control. Defs come first in the
  // MCInst operand list, sources follow. We want src0 = ttmp8 and
  // src1 = imm 0x50019.
  if (di.numSrcs < 2)
    return false;
  unsigned src0Idx = di.srcMap[0];
  unsigned src1Idx = di.srcMap[1];
  if (!di.isReg(src0Idx) || !di.isImm(src1Idx))
    return false;
  if (di.getImm(src1Idx) != 0x50019)
    return false;
  const MCRegisterClass &TTMP32 =
      MRI.getRegClass(AMDGPU::TTMP_32RegClassID);
  MCRegister ttmp8Pseudo = TTMP32.getRegister(8);
  MCRegister src0Reg = di.inst.getOperand(src0Idx).getReg();
  if (!src0Reg)
    return false;
  // Canonical pattern is a single `ttmp8` (not a tuple), so the
  // operand register itself should normalise to the TTMP8 pseudo.
  if (AMDGPU::mc2PseudoReg(src0Reg) == ttmp8Pseudo)
    return true;
  return false;
}

} // namespace

// ----------------------------------------------------------------------------
// buildObstructionReport — the main walk.
// ----------------------------------------------------------------------------

ObstructionReport buildObstructionReport(ArrayRef<DecodedInst> insts,
                                          const MCState &mc,
                                          const ISAProfile &src,
                                          const ISAProfile &tgt,
                                          bool enableWritelaneRewrite) {
  ObstructionReport report;
  if (src.waveSize == tgt.waveSize)
    return report;
  const MCRegisterInfo &MRI = *mc.regInfo;

  // First pass: tag the self-contained obstruction kinds (lane-id
  // leaks, cross-lane shuffles, replica races). Also collect
  // co-occurrence state for the lane-predicated EXEC check below.
  //
  // The walk matches purely on `SemOp` and `MCInstrDesc` TSFlags
  // bits; there is no string matching on `rawMnemonic`. New
  // obstruction triggers should be added by extending semop.hpp +
  // opcode_map.cpp (so the lookup is a single enum compare here),
  // not by adding `raw.contains(...)` substring tests.
  bool haveMbcnt = false;
  bool haveWMMA = false;
  struct PendingExecSite {
    const DecodedInst *inst;
    ObstructionKind kind; // CmpxFromLaneId or SaveExecFromLaneId.
  };
  llvm::SmallVector<PendingExecSite> pendingExecWriters;
  // Deferred TtmpWaveIdLeak site emission. The canonical shape —
  // `s_bfe_u32 sDST, ttmp8, 0x50019` — has a principled rescue in
  // `handle_sop2.cpp`'s `S_BFE_U32` pattern-lift, which emits
  // `dst = (workitem.id.x >> log2(W_s)) & 0x1F` directly from the
  // divergent-leaf intrinsic and sidesteps the backend's implicit
  // scalarisation of the formally-scalar BFE → SGPR path. The lift
  // preserves per-source-wave semantics across cross-widening, so
  // the canonical shape is NOT recorded here (filtered via
  // `isCanonicalWaveIdBfe`).
  //
  // Any OTHER ttmp8 source read (non-canonical immediates, AND /
  // LSHR / s_load offsets, trap-handler prologues, etc.) is still a
  // Class 1 leak: the raiser's ttmp8 init (`raiser.cpp` phase-4)
  // only models the `bits [29:25] = wave_id` field, so a consumer
  // that reads other bits or uses a different bitfield extract
  // semantics would silently miscompile. Those sites are collected
  // here; non-WMMA kernels have a future escape hatch through
  // `ThreadLoopProjection` (§2.2 — iterate the body R = W_t / W_s
  // times with a synthetic per-source-wave wave_id in ttmp8), and
  // WMMA kernels refuse because the §5.2 lane layout requires the
  // full target wave simultaneously and cannot be TLP-split.
  llvm::SmallVector<const DecodedInst *> ttmp8ReadSites;

  // Co-occurrence tracking for the WaveIdLiftScalarized refusal below.
  //
  //   - `canonicalWaveIdBfeSites`: every occurrence of the canonical
  //     `s_bfe_u32 sDST, ttmp8, 0x50019` pattern that the handle_sop2.cpp
  //     lift rescues by making sDST a per-lane divergent VGPR value
  //     (workitem.id.x >> log2(W_s)). That rescue is only semantically
  //     valid if the per-lane divergent value never feeds a source-ISA
  //     construct that enforces scalar-in-source semantics at the
  //     hardware level. `v_writelane_b32` / `v_readlane_b32` enforce
  //     exactly that scalar-in rule (the `src0` operand is an SGPR in
  //     the encoding), and the backend materialises that by inserting
  //     a readfirstlane on a divergent input — which collapses target
  //     lanes 0..31 (source_wave[0]) with 32..63 (source_wave[1]) back
  //     to a single value. That collapse silently miscompiles every
  //     wave_id-dependent tile address the matmul encoded.
  //
  //   - `crossLaneScalarSites`: every `v_writelane_b32` / `v_readlane_b32`
  //     in the kernel. The refusal fires once per site so the diagnostic
  //     trace points at the precise instructions where the collapse
  //     happens, not just at the BFE where the divergent value was
  //     manufactured.
  //
  // Both buffers are emptied into `ObstructionReport::sites` after the
  // walk completes, gated on `haveWMMA` — the non-WMMA case has a
  // future ThreadLoopProjection escape hatch (§2.2; iterate the body
  // R = W_t / W_s times with a synthetic per-source-wave wave_id in
  // ttmp8) and must not be refused preemptively here. WMMA kernels
  // cannot use TLP because §5.2 WMMA lane layout requires the full
  // target wave simultaneously, so the refusal is terminal.
  llvm::SmallVector<const DecodedInst *> canonicalWaveIdBfeSites;
  llvm::SmallVector<const DecodedInst *> crossLaneScalarSites;

  for (const DecodedInst &di : insts) {
    const SemOp sop = di.semOp;

    // --- §3 Class 1: wave_id leak via ttmp8 source read --------------
    // Under cross-widening, raiser.cpp seeds the transpiler's ttmp8
    // alloca from `workitem.id.x >> 5` shifted into bits [29:25] so
    // the per-lane value encodes the source's `wave_id_in_workgroup`.
    //
    // The canonical shape `s_bfe_u32 sDST, ttmp8, 0x50019` is rescued
    // inline by `handle_sop2.cpp`'s pattern-lift: it emits
    // `dst = (workitem.id.x >> log2(W_s)) & 0x1F` directly from the
    // divergent-leaf intrinsic, bypassing the backend's implicit
    // SGPR-class scalarisation. That shape is therefore NOT a
    // Class 1 refusal surface and is filtered out here.
    //
    // Any other source reference to ttmp8 (non-canonical BFE
    // immediates, `s_and_b32` / `s_lshr_b32` operating on ttmp8,
    // `s_load_dword` using ttmp8 as offset, trap-handler prologues
    // touching ttmp8..ttmp15, etc.) is still a leak: the raiser's
    // init only models the `[29:25] = wave_id` field, so consumers
    // of other bits read either zero or a garbage pattern. We defer
    // the site emission until after the loop has established whether
    // the kernel also contains WMMA (see below). Without WMMA the
    // leak is handled by ThreadLoopProjection; with WMMA it is
    // unrewritable (TLP and WMMA are mutually exclusive — §5.2 WMMA
    // lane layout requires the full target wave) and we refuse.
    if (readsTtmp8Source(di, MRI) && !isCanonicalWaveIdBfe(di, MRI))
      ttmp8ReadSites.push_back(&di);

    // Track canonical wave_id BFE sites for the WaveIdLiftScalarized
    // post-loop check. The lift in handle_sop2.cpp makes this BFE's
    // destination SGPR carry a per-lane divergent value (wave_id mod W_s)
    // instead of the backend's scalar BFE result; that rescue is
    // correct in isolation but collapses back to uniform when the
    // divergent value is consumed by any construct the source-ISA
    // encodes with scalar-in semantics (writelane src, readlane src).
    // The post-loop join below pairs these sites with the
    // crossLaneScalarSites + haveWMMA co-occurrence to decide the
    // refusal.
    if (isCanonicalWaveIdBfe(di, MRI))
      canonicalWaveIdBfeSites.push_back(&di);

    // WMMA-family detection. If any of these show up in the kernel,
    // the WMMA → MFMA lowering (matrix-translation.md) is going to be
    // invoked and the TLP escape hatch is not available — every
    // deferred ttmp8 site in this kernel becomes an unrewritable
    // refusal surface.
    switch (sop) {
    case SemOp::V_WMMA_F32_16x16x32_F16:
    case SemOp::V_WMMA_F32_16x16x32_BF16:
    case SemOp::V_WMMA_F32_16x16x4_F32:
    case SemOp::V_WMMA_F32_16x16x64_FP8_FP8:
    case SemOp::V_WMMA_F32_16x16x64_FP8_BF8:
    case SemOp::V_WMMA_F32_16x16x64_BF8_FP8:
    case SemOp::V_WMMA_F32_16x16x64_BF8_BF8:
    case SemOp::V_WMMA_I32_16x16x64_IU8:
    case SemOp::V_WMMA_SCALE_F32_16x16x128_F8F6F4:
      haveWMMA = true;
      break;
    default:
      break;
    }

    // --- §3 Class 1: absolute lane-ID leaks --------------------------
    if (sop == SemOp::V_MBCNT_HI_U32_B32) {
      ObstructionSite site;
      site.inst = &di;
      site.kind = ObstructionKind::MbcntHiLaneIdLeak;
      site.rewrite = RewriteId::None;
      site.rewriteImplemented = false;
      site.detail = "v_mbcnt_hi reads target exec_hi — no wave32 semantics "
                    "to preserve under modulo-replication";
      report.sites.push_back(std::move(site));
      haveMbcnt = true;
      continue;
    }
    if (sop == SemOp::V_MBCNT_LO_U32_B32) {
      // v_mbcnt_lo alone is not a leak by itself (wave32 sources use
      // it as the canonical lane-id probe and it's lane-position-
      // independent inside its wave). We track its presence only
      // for the lane-predicated-EXEC co-occurrence heuristic below.
      haveMbcnt = true;
      continue;
    }
    if (sop == SemOp::V_READLANE_B32 || sop == SemOp::V_WRITELANE_B32) {
      // Track every readlane/writelane — in-bounds or otherwise — for
      // the WaveIdLiftScalarized post-loop check. Out-of-range static
      // lane operands additionally emit an OutOfRangeLaneOperand site
      // in the block below; the two conditions are independent (a
      // kernel could have a well-formed writelane whose `val` operand
      // carries a wave_id-derived divergent value, and that refusal
      // must fire even when every lane operand is in-bounds).
      crossLaneScalarSites.push_back(&di);

      auto imm = extractLaneOperandImm(di);
      // Negative-value guard: an int64_t imm cast to uint64_t for the
      // bounds compare wraps around to a value > waveSize for any
      // negative value, which is the correct logical answer (negative
      // lane indices are never in [0, W_s)) but is implicit in the
      // cast. Make it explicit so the intent survives a refactor.
      if (imm.has_value() &&
          (*imm < 0 ||
           static_cast<uint64_t>(*imm) >= src.waveSize)) {
        // Static constant operand provably out of source wave range.
        // No rewrite preserves the semantics on a wider target wave.
        ObstructionSite site;
        site.inst = &di;
        site.kind = ObstructionKind::OutOfRangeLaneOperand;
        site.rewrite = RewriteId::None;
        site.rewriteImplemented = false;
        std::string det;
        raw_string_ostream os(det);
        os << "operand value " << *imm << " out of [0, W_s=" << src.waveSize
           << ")";
        site.detail = os.str();
        report.sites.push_back(std::move(site));
      }
      // Otherwise (static imm < W_s, or dynamic operand): do not
      // emit a site.
      //
      // - Static imm < W_s: provably in-bounds, safe by construction.
      // - Dynamic operand (SGPR): we cannot statically prove the
      //   runtime value is < W_s, BUT we also cannot prove it is out
      //   of bounds. Triton's softmax / matmul patterns (see
      //   gpt-oss-derisking.md §7.1) use `v_writelane_b32` with
      //   dynamic lane operands that are in-bounds at runtime but not
      //   statically provable. Flagging those as refusal would
      //   collapse coverage on every Gfx1250Gpu.* test that uses
      //   them.
      //
      // TODO(dataflow-upgrade): graduate dynamic operands from "not
      // flagged" to "proved via LLVM uniformity / value-range
      // analysis on the raised IR" once the post-raise dataflow
      // analysis lands. Today this is a sound-not-complete choice
      // toward false negatives on readlane/writelane specifically —
      // tracked in wave_size_obstruction.hpp's TODO block.
      continue;
    }

    // --- §3 Class 2: wave-width-specific cross-lane shuffles --------
    if (sop == SemOp::V_PERMLANE64_B32) {
      ObstructionSite site;
      site.inst = &di;
      site.kind = ObstructionKind::FullWaveRotate;
      site.rewrite = RewriteId::None;
      site.rewriteImplemented = false;
      site.detail = "v_permlane64 has no wave32 analogue";
      report.sites.push_back(std::move(site));
      continue;
    }
    if (sop == SemOp::V_PERMLANE16_B32 ||
        sop == SemOp::V_PERMLANEX16_B32 ||
        sop == SemOp::V_PERMLANE16_SWAP_B32 ||
        sop == SemOp::V_PERMLANE32_SWAP_B32) {
      ObstructionSite site;
      site.inst = &di;
      site.kind = ObstructionKind::LaneGroupShuffle;
      // P2 covers base permlane16/permlanex16; P4 covers the swap
      // variants. `rewriteImplemented` is set per-SemOp so the
      // decider emits a precise "cross-wave-shuffle-rewrite-pending"
      // diagnostic for the specific P-item still missing.
      //
      // P2 (base permlane16/permlanex16) landed:
      // `handle_valu_cross_lane.cpp` emits ds_bpermute-emulated
      // permlane16/permlanex16 with fi / bc extracted from the
      // VOP3 src{0,1}_modifiers' OP_SEL_0 bit.
      //
      // P4 (permlane16_swap_b32) landed: ds_bpermute-emulated
      // partner = lane_id XOR 16, two bpermutes for the two-VGPR
      // exchange. The wider permlane32_swap_b32 variant stays
      // unrewritable — its XOR-32 partner spans wave64's two
      // 32-lane halves, which has no wave32 analogue, so a wave32
      // source kernel cannot meaningfully encode it. Seeing it in
      // a wave32 source binary indicates either a corrupted
      // disassembly or a wave64 source mis-classified as wave32.
      if (sop == SemOp::V_PERMLANE32_SWAP_B32) {
        site.rewrite = RewriteId::P4_PermLaneSwap;
        site.rewriteImplemented = false;
        site.detail = "v_permlane32_swap_b32: XOR-32 partner spans "
                      "wave64 32-lane halves; no wave32 analogue";
      } else if (sop == SemOp::V_PERMLANE16_SWAP_B32) {
        site.rewrite = RewriteId::P4_PermLaneSwap;
        site.rewriteImplemented = true;
      } else {
        site.rewrite = RewriteId::P2_PermLane16;
        site.rewriteImplemented = true;
      }
      report.sites.push_back(std::move(site));
      continue;
    }
    if (sop == SemOp::DS_SWIZZLE_B32) {
      ObstructionSite site;
      site.inst = &di;
      site.kind = ObstructionKind::DsSwizzle;
      site.rewrite = RewriteId::P6_DsSwizzle;
      // P6 landed (see the DS_SWIZZLE_B32 row of hotswap/docs/wave-
      // size-translation.md §5.3): the handler in handle_ds.cpp emits
      // `llvm.amdgcn.ds.swizzle(value, offset)`
      // with the 16-bit immediate plumbed through. The lift is only
      // wave-size-oblivious for the QUAD_PERM and BITMASK_PERM
      // sub-modes (see `dsSwizzleSafeForModRep` above for the
      // structural argument); FFT_MODE / ROTATE_MODE / unknown-high-
      // nibble imms remain pending and refuse loudly via the same
      // CrossWaveShuffleRewritePending channel as before.
      //
      // The 16-bit imm is extracted at decode time into
      // `di.dsSwizzleImm` (see `decode.cpp::decodeDsSwizzleImm`).
      // `!di.hasDsSwizzleImm` here means the decoder rejected the
      // operand (missing, non-immediate, or out of 16-bit range);
      // we mirror that rejection as a P6-pending refusal with a
      // malformed-disassembly diagnostic so the kernel fails loudly
      // rather than the handler inventing a value.
      if (!di.hasDsSwizzleImm) {
        site.rewriteImplemented = false;
        site.detail = "ds_swizzle_b32 missing/invalid OpName::offset "
                      "immediate operand — disassembly malformed or "
                      "outside 16-bit range";
      } else {
        site.rewriteImplemented = dsSwizzleSafeForModRep(di.dsSwizzleImm);
        if (!site.rewriteImplemented) {
          std::string det;
          raw_string_ostream os(det);
          os << "ds_swizzle_b32 imm 0x"
             << format_hex_no_prefix(di.dsSwizzleImm, 4)
             << " is not a valid swizzle encoding (not QUAD_PERM, "
                "BITMASK_PERM, valid FFT_MODE, or valid ROTATE_MODE) "
                "— RESERVED top-nibble or FFT/ROTATE reserved bits "
                "set; AMDGPU hardware semantics undefined";
          site.detail = os.str();
        }
      }
      report.sites.push_back(std::move(site));
      continue;
    }
    // DPP detection via the MCInstrDesc TSFlags bit. opcode_map.cpp
    // canonicalises DPP variants down to their base SemOp, so the
    // SemOp alone cannot identify them, but `di.tsFlags` is captured
    // from the *original* MCInstrDesc (see decode.cpp) so the DPP
    // bit is still visible. Same for SDWA — though SDWA is same-lane
    // and not a cross-wave concern, so we don't flag it.
    if (di.tsFlags & SIInstrFlags::DPP) {
      ObstructionSite site;
      site.inst = &di;
      site.kind = ObstructionKind::DppCrossLane;
      site.rewrite = RewriteId::P5_DppModifier;
      // P5 (DPP16 lift via `llvm.amdgcn.update.dpp`) landed for the
      // DPP16 encoding family. `decodeDppModifiers` sets
      // `di.hasDpp = true` only after successfully extracting every
      // DPP16 modifier operand (dpp_ctrl / row_mask / bank_mask /
      // bound_ctrl); for DPP8 instructions it early-returns and leaves
      // `hasDpp` false. Gate `rewriteImplemented` on `hasDpp` so DPP16
      // accepts (outcome b) while DPP8 refuses loudly (outcome c
      // pending a P5 extension to `llvm.amdgcn.mov.dpp8`). The
      // `tsFlags & SIInstrFlags::DPP` check still fires for both
      // forms — both are a Class-2 cross-lane site by the hotswap/
      // docs/wave-size-translation.md §6 taxonomy; the flipped-by-
      // form `rewriteImplemented` bit is
      // what separates "handled" from "pending" without mucking with
      // the taxonomy.
      site.rewriteImplemented = di.hasDpp;
      if (!di.hasDpp)
        site.detail =
            "DPP8 lane-permutation form — P5 currently covers only the "
            "DPP16 encoding family via llvm.amdgcn.update.dpp; extending "
            "to DPP8 requires an llvm.amdgcn.mov.dpp8 lift path.";
      report.sites.push_back(std::move(site));
      continue;
    }
    if (sop == SemOp::DS_BPERMUTE_B32) {
      // P1 is IMPLEMENTED in handle_ds.cpp (see lit_tests/ds_bpermute_b32).
      // Record the site so the trace shows it, but mark as
      // `rewriteImplemented = true` so the decider treats it as
      // outcome (a)/(b) rather than refusal.
      ObstructionSite site;
      site.inst = &di;
      site.kind = ObstructionKind::DsBpermuteGather;
      site.rewrite = RewriteId::P1_DsBpermute;
      site.rewriteImplemented = true;
      report.sites.push_back(std::move(site));
      continue;
    }

    // --- §3 Class 3: replica races on shared state ------------------
    // The SemOp set here is the complete enumeration of
    // non-commutative atomics modeled in semop.hpp today. New
    // non-commutative atomic encodings (e.g. SCRATCH_ATOMIC_SWAP if
    // we ever model it) should be added by extending the enum +
    // opcode_map.cpp + semop.cpp, not by adding a substring check
    // here. Atomics not yet modeled refuse via the existing Phase 5
    // unsupportedOpcode path.
    // S_ATOMIC_DEC's wrap-at-zero decrement is non-commutative: two
    // lanes in opposite source waves racing on the same counter produce
    // different outcomes from the two possible orderings when `old`
    // crosses the `0` or `> src` boundary on only one of them.  Every
    // corpus producer (AITER split-k epilogue barriers) keys on the
    // pre-decrement value being 1, so an ordering flip silently picks
    // the wrong wave as the "last workgroup", corrupting the reduction.
    if (sop == SemOp::GLOBAL_ATOMIC_SWAP ||
        sop == SemOp::GLOBAL_ATOMIC_CMPSWAP ||
        sop == SemOp::FLAT_ATOMIC_SWAP ||
        sop == SemOp::FLAT_ATOMIC_CMPSWAP ||
        sop == SemOp::BUFFER_ATOMIC_SWAP ||
        sop == SemOp::BUFFER_ATOMIC_CMPSWAP ||
        sop == SemOp::S_ATOMIC_SWAP ||
        sop == SemOp::S_ATOMIC_DEC) {
      ObstructionSite site;
      site.inst = &di;
      site.kind = ObstructionKind::NonCommutativeAtomic;
      site.rewrite = RewriteId::None;
      site.rewriteImplemented = false;
      site.detail = "non-commutative atomic races target lanes "
                    "i and i+W_s under modulo-replication";
      report.sites.push_back(std::move(site));
      continue;
    }

    // --- §3 Class 4: lane-predicated EXEC writers -------------------
    // The principled check is "does the EXEC-writer's source operand
    // chain contain a value derived from mbcnt?". The syntactic
    // approximation: collect EXEC writers now, decide after the walk
    // based on kernel-level mbcnt presence.
    if (sop == SemOp::V_CMPX) {
      pendingExecWriters.push_back({&di, ObstructionKind::CmpxFromLaneId});
      continue;
    }
    if (sop == SemOp::S_AND_SAVEEXEC_B32 ||
        sop == SemOp::S_OR_SAVEEXEC_B32 ||
        sop == SemOp::S_XOR_SAVEEXEC_B32 ||
        sop == SemOp::S_ANDN2_SAVEEXEC_B32 ||
        sop == SemOp::S_ORN2_SAVEEXEC_B32) {
      pendingExecWriters.push_back({&di, ObstructionKind::SaveExecFromLaneId});
      continue;
    }
  }

  // Deferred TtmpWaveIdLeak emission. See the pre-loop comment on
  // `ttmp8ReadSites` for the rationale: the `s_bfe_u32 ttmp8, 0x50019`
  // wave_id extraction is clang/hip boilerplate in every non-trivial
  // gfx1250 kernel, so unconditionally refusing on it would collapse
  // coverage. We only refuse when the kernel also contains WMMA —
  // in which case ThreadLoopProjection (the §2.2 escape hatch for
  // class-4 wave_id leaks) cannot be applied because the §5.2 WMMA
  // lane layout requires the full target wave simultaneously. In the
  // non-WMMA case, fall through silently; the caller's projection
  // selector will pick TLP in raiser.cpp.
  if (haveWMMA) {
    for (const DecodedInst *di : ttmp8ReadSites) {
      ObstructionSite site;
      site.inst = di;
      site.kind = ObstructionKind::TtmpWaveIdLeak;
      site.rewrite = RewriteId::None;
      site.rewriteImplemented = false;
      site.detail =
          "source reads ttmp8 under cross-widening — bits [29:25] carry "
          "wave_id_in_workgroup, which is a function of the target's "
          "absolute lane position (not of lane_id mod W_s). Kernel also "
          "contains WMMA, so ThreadLoopProjection is not available — refuse.";
      report.sites.push_back(std::move(site));
    }
  }

  // WaveIdLiftScalarized — the canonical-BFE rescue collapses inside a
  // cross-lane scalar primitive under WMMA.
  //
  // This is the Matmul128x128 / `matmul_f16_large_gfx1250` pattern:
  //
  //     s_bfe_u32 s2, ttmp8, 0x50019          ; lifted → divergent wave_id
  //     s_and_b32 s73, s2, 3                  ; tainted (SGPR but divergent)
  //     s_lshl_b32 s18, s2, 5                 ; tainted
  //     v_writelane_b32 vgpr256, s18, 4       ; backend readfirstlane(s18)
  //                                           ; collapses target lanes
  //                                           ; 0..31 with 32..63 to a
  //                                           ; single value → per-
  //                                           ; source-wave tile offset
  //                                           ; LOST.
  //     …
  //     v_wmma_f32_16x16x32_f16 …             ; WMMA → TLP not available.
  //     …
  //     v_readlane_b32 sDST, vgpr256, 4       ; reads the collapsed value.
  //     v_or_b32 v_col, sDST, v_col_within    ; per-source-wave column
  //                                           ; base is now uniform,
  //                                           ; writing the wrong tile.
  //
  // The refusal is syntactic (three-way co-occurrence over the kernel)
  // and therefore a sound-not-complete over-approximation, same as the
  // CmpxFromLaneId / SaveExecFromLaneId co-occurrence heuristic above.
  // Kernels that happen to contain all three constructs but do NOT
  // route the wave_id value into the cross-lane scalar source would be
  // over-refused — benign (false positive). Kernels that are missing
  // any of the three would not be refused here but ALSO cannot express
  // the matmul-shaped wave_id-dependent tile column bug (no canonical
  // BFE means ttmp8 is only read via other shapes, which fall to the
  // ttmp8ReadSites + WMMA refusal above; no cross-lane scalar means
  // the divergent value has no scalar-semantics consumer to collapse
  // into; no WMMA means TLP is available for the class-4 escape). The
  // implications chain is safe by construction.
  //
  // TODO(dataflow-upgrade): replace the syntactic co-occurrence with a
  // precise check that the BFE's destination SGPR flows (through the
  // raised IR's SSA uses) into a v_writelane / v_readlane scalar
  // source operand. The LLVM Uniformity Analysis on the raised IR
  // (post-Phase-2) is the natural place to land this — see
  // wave_size_obstruction.hpp's TODO block.
  if (haveWMMA && !canonicalWaveIdBfeSites.empty() &&
      !crossLaneScalarSites.empty()) {
    for (const DecodedInst *di : crossLaneScalarSites) {
      ObstructionSite site;
      site.inst = di;
      site.kind = ObstructionKind::WaveIdLiftScalarized;
      // When `enableWritelaneRewrite` is on, the site has an
      // implemented rewrite (the post-mem2reg pass in
      // `rewrite_cross_lane_divergent.{hpp,cpp}` replaces the
      // collapsing cross-lane primitive with a per-source-wave
      // `select` / `ds.bpermute`). Tag it accordingly so the
      // pre-translation abort below does NOT fire — the rewrite
      // discharges the obstruction during Phase 6.5 of raiser.cpp.
      // Paired with a post-raise safety net in raiser.cpp that
      // verifies the rewrite pass actually rewrote at least one
      // site (guards against an oracle false-negative disagreeing
      // with this syntactic co-occurrence classifier).
      if (enableWritelaneRewrite) {
        site.rewrite = RewriteId::PostRaiseCrossLaneRewrite;
        site.rewriteImplemented = true;
      } else {
        site.rewrite = RewriteId::None;
        site.rewriteImplemented = false;
      }
      site.detail =
          "kernel also contains the canonical `s_bfe_u32 sDST, ttmp8, "
          "0x50019` wave_id lift and v_wmma_* — the lift's per-lane "
          "divergent result is scalarised by the backend on entry to "
          "this cross-lane primitive's scalar source operand, "
          "collapsing source_wave[0]'s and source_wave[1]'s distinct "
          "values into a single uniform. WMMA forecloses the "
          "ThreadLoopProjection escape hatch (§5.2 requires the full "
          "target wave simultaneously), so no correct projection is "
          "available.";
      report.sites.push_back(std::move(site));
    }
  }

  // Second pass: for each pending EXEC writer, apply the syntactic
  // co-occurrence heuristic. If the kernel contains ANY mbcnt (lo or
  // hi), treat the writer as lane-predicated and unrewritable.
  // Otherwise it is a lane-position-INDEPENDENT bounds-check style
  // writer (the overwhelming common case per gpt-oss-derisking.md
  // §7.6) and does NOT produce a site — the outer Phase 1.4 legacy
  // diagnostic (now LLVM_DEBUG-only) still logs it for auditability.
  //
  // TODO(dataflow-upgrade): replace this co-occurrence heuristic with
  // a precise dataflow check once the classifier runs post-raise on
  // the LLVM IR. See wave_size_obstruction.hpp's TODO block.
  if (haveMbcnt) {
    for (const auto &pw : pendingExecWriters) {
      ObstructionSite site;
      site.inst = pw.inst;
      site.kind = pw.kind;
      site.rewrite = RewriteId::None;
      site.rewriteImplemented = false;
      site.detail =
          "v_cmpx / saveexec co-occurs with v_mbcnt_* in the same kernel — "
          "syntactic over-approximation of 'gating expression flows from an "
          "absolute-lane-id value'. Dataflow upgrade may reclassify this as "
          "wave-size-oblivious (see TODO(dataflow-upgrade)).";
      report.sites.push_back(std::move(site));
    }
  }

  return report;
}

// ----------------------------------------------------------------------------
// Rendering — stable-enough-for-lit trace format.
// ----------------------------------------------------------------------------

std::string renderObstructionTrace(const ObstructionReport &report,
                                    StringRef kernelName, StringRef srcIsa,
                                    StringRef tgtIsa, unsigned srcWaveSize,
                                    unsigned tgtWaveSize) {
  std::string out;
  raw_string_ostream os(out);

  os << "transpiler: projection decision for kernel '" << kernelName << "':\n";
  os << "  source: " << srcIsa << " (wave" << srcWaveSize
     << ") -> target: " << tgtIsa << " (wave" << tgtWaveSize << "), R="
     << (srcWaveSize > 0 ? tgtWaveSize / srcWaveSize : 0) << "\n";

  if (report.sites.empty()) {
    os << "  obstructions found: none\n"
       << "  outcome: (a) wave-size-oblivious — emit modulo-replication\n";
    return out;
  }

  os << "  obstructions found:\n";
  for (const ObstructionSite &s : report.sites) {
    os << "    " << obstructionKindName(s.kind);
    if (s.inst) {
      os << " @ 0x" << format_hex_no_prefix(s.inst->offset, 4) << ": "
         << s.inst->rawMnemonic;
    }
    os << "\n      rewrite: " << rewriteIdName(s.rewrite);
    if (s.rewrite != RewriteId::None)
      os << " [" << (s.rewriteImplemented ? "implemented" : "pending") << "]";
    if (!s.detail.empty())
      os << "\n      detail: " << s.detail;
    os << "\n";
  }

  if (report.hasUnrewritable()) {
    os << "  outcome: (c) refuse — at least one obstruction has no rewrite "
          "in wave-size-translation.md \u00a77's unrewritable table\n";
  } else if (report.hasPendingRewrite()) {
    os << "  outcome: (c) refuse — rewrite(s) exist on paper but the "
          "matching handler(s) have not yet landed "
          "(wave-size-translation.md \u00a77's pending-rewrite table)\n";
  } else {
    os << "  outcome: (b) rewrite-then-emit — all obstruction sites have "
          "an implemented rewrite; emit modulo-replication\n";
  }
  return out;
}

// ----------------------------------------------------------------------------
// Failure selection — pick the first refusal-worthy site and package
// it as a RaiseFailure for raiser.cpp to propagate.
// ----------------------------------------------------------------------------

RaiseFailure selectFailureFromReport(const ObstructionReport &report) {
  // Prefer unrewritable over pending — the caller should see the
  // strongest refusal reason first. Ties broken by decoded order (the
  // `sites` vector is in decoded order, so `firstUnrewritable` /
  // `firstPending` both return the earliest match).
  //
  // Twine lifetime: each `Twine(...) + ... + ...` chain is built and
  // consumed in the SAME full-expression as the factory call below.
  // This is the LLVM-supported lifetime contract — Twine concat
  // results are temporaries that hold references into their operands
  // and *must not* be bound to a named variable (`const Twine x = a +
  // b + c` would leave `x` referencing temporaries that are destroyed
  // at the end of that statement). See the LLVM Programmer's Manual
  // on Twine.
  if (const ObstructionSite *site = report.firstUnrewritable()) {
    switch (site->kind) {
    case ObstructionKind::MbcntHiLaneIdLeak:
    case ObstructionKind::OutOfRangeLaneOperand:
    case ObstructionKind::TtmpWaveIdLeak:
    case ObstructionKind::WaveIdLiftScalarized:
      return RaiseFailure::crossWaveLaneIdLeak(
          *site->inst,
          Twine(obstructionKindName(site->kind)) + " [" + site->detail + "]");
    case ObstructionKind::FullWaveRotate:
      return RaiseFailure::crossWaveUnrewritableShuffle(
          *site->inst,
          Twine(obstructionKindName(site->kind)) + " [" + site->detail + "]");
    case ObstructionKind::NonCommutativeAtomic:
      return RaiseFailure::crossWaveReplicaRace(
          *site->inst,
          Twine(obstructionKindName(site->kind)) + " [" + site->detail + "]");
    case ObstructionKind::CmpxFromLaneId:
    case ObstructionKind::SaveExecFromLaneId:
      return RaiseFailure::crossWaveLanePredicatedExec(
          *site->inst,
          Twine(obstructionKindName(site->kind)) + " [" + site->detail + "]");
    // The kinds below never set `rewrite = None` under
    // buildObstructionReport, so they cannot reach firstUnrewritable().
    // If they do, our state is inconsistent — fail loudly rather than
    // silently fall through to the empty-RaiseFailure return below.
    case ObstructionKind::LaneGroupShuffle:
    case ObstructionKind::DsSwizzle:
    case ObstructionKind::DppCrossLane:
    case ObstructionKind::DsBpermuteGather:
    case ObstructionKind::None:
      llvm_unreachable("ObstructionKind classified as unrewritable but "
                       "buildObstructionReport never tags it that way");
    }
    llvm_unreachable("unhandled ObstructionKind in selectFailureFromReport "
                     "(unrewritable branch)");
  }
  if (const ObstructionSite *site = report.firstPending()) {
    return RaiseFailure::crossWaveShuffleRewritePending(
        *site->inst,
        Twine(obstructionKindName(site->kind)) + " [rewrite " +
            rewriteIdName(site->rewrite) + " pending]");
  }
  // Oblivious / fully-rewritten: no failure.
  return RaiseFailure();
}

} // namespace transpiler
