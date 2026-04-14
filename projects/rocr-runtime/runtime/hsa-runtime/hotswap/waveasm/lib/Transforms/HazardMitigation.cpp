// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Hazard Mitigation Pass - Insert s_nop instructions for hardware hazards
//
// This pass handles hardware-specific hazards that require NOP insertion:
// - VALU → v_readfirstlane hazard (gfx940+)
// - Trans → non-Trans VALU forwarding hazard (gfx940+)
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMAttrs.h"
#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Transforms/Liveness.h"
#include "waveasm/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace waveasm;

namespace waveasm {
#define GEN_PASS_DEF_WAVEASMHAZARDMITIGATION
#include "waveasm/Transforms/Passes.h.inc"
} // namespace waveasm

namespace {

//===----------------------------------------------------------------------===//
// Instruction Classification
//===----------------------------------------------------------------------===//

/// Check if an operation is a VALU instruction (writes VGPR, not memory)
bool isVALUOp(Operation *op) {
  // Must produce at least one VGPR result
  bool writesVGPR = false;
  for (Value result : op->getResults()) {
    if (isa<VRegType, PVRegType>(result.getType())) {
      writesVGPR = true;
      break;
    }
  }
  if (!writesVGPR)
    return false;

  // Exclude memory operations (VMEM, LDS, SMEM)
  if (isa<BUFFER_LOAD_DWORD, BUFFER_LOAD_DWORDX2, BUFFER_LOAD_DWORDX3,
          BUFFER_LOAD_DWORDX4, BUFFER_LOAD_UBYTE, BUFFER_LOAD_SBYTE,
          BUFFER_LOAD_USHORT, BUFFER_LOAD_SSHORT, GLOBAL_LOAD_DWORD,
          GLOBAL_LOAD_DWORDX2, GLOBAL_LOAD_DWORDX3, GLOBAL_LOAD_DWORDX4,
          GLOBAL_LOAD_UBYTE, GLOBAL_LOAD_SBYTE, GLOBAL_LOAD_USHORT,
          GLOBAL_LOAD_SSHORT, FLAT_LOAD_DWORD, FLAT_LOAD_DWORDX2,
          FLAT_LOAD_DWORDX3, FLAT_LOAD_DWORDX4, DS_READ_B32, DS_READ_B64,
          DS_READ_B128, DS_READ2_B32, DS_READ2_B64, DS_READ_U8, DS_READ_I8,
          DS_READ_U16, DS_READ_I16>(op))
    return false;

  // Exclude non-ALU ops that produce VGPRs
  if (isa<PrecoloredVRegOp, PackOp, ExtractOp>(op))
    return false;

  // Exclude v_readfirstlane (it's the consumer in the hazard, not the producer)
  if (isa<V_READFIRSTLANE_B32>(op))
    return false;

  return true;
}

/// Check if an operation is v_readfirstlane
bool isReadfirstlaneOp(Operation *op) { return isa<V_READFIRSTLANE_B32>(op); }

/// Check if an operation does NOT emit an assembly instruction.
/// These ops are lowered to register aliases or eliminated entirely,
/// so they don't create real inter-instruction delays.
bool isNonEmittingOp(Operation *op) {
  return op->hasTrait<OpTrait::NonEmittingOp>();
}

/// Walk backwards from position `start` in the flattened op list, skipping
/// non-emitting ops, and return the nearest emitting predecessor.
/// Because collectOpsRecursive flattens region bodies into the list,
/// this naturally looks through region boundaries -- yield/condition
/// terminators are non-emitting, so the scan skips them and finds the
/// last real instruction inside the region.
Operation *findPrecedingEmittingOp(ArrayRef<Operation *> ops, size_t start) {
  for (size_t j = start; j > 0; --j) {
    Operation *candidate = ops[j - 1];
    if (!isNonEmittingOp(candidate))
      return candidate;
  }
  return nullptr;
}

/// Check if an operation is a transcendental instruction (uses the Trans
/// pipeline which has different latency characteristics from the main VALU).
bool isTransOp(Operation *op) {
  return isa<V_RCP_F32, V_RCP_F64, V_RSQ_F32, V_RSQ_F64, V_SQRT_F32, V_SQRT_F64,
             V_EXP_F32, V_LOG_F32, V_SIN_F32, V_COS_F32>(op);
}

/// Check if `producer` writes a VGPR that `consumer` reads.
/// Both sets are tiny (1-3 elements each -- one VALU result, two-three
/// operands), so a plain nested loop beats any set/hashing overhead.
///
/// After register allocation all VGPRs should be PVRegType; virtual
/// VRegType pairs can only conflict via SSA identity since they carry
/// no physical index.
bool hasVGPRConflict(Operation *producer, Operation *consumer) {
  for (Value def : producer->getResults()) {
    auto defPVReg = dyn_cast<PVRegType>(def.getType());
    if (!defPVReg && !isa<VRegType>(def.getType()))
      continue;
    for (Value use : consumer->getOperands()) {
      if (use == def)
        return true;
      if (defPVReg)
        if (auto usePVReg = dyn_cast<PVRegType>(use.getType()))
          if (usePVReg.getIndex() == defPVReg.getIndex())
            return true;
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Target-Specific Hazard Rules
//===----------------------------------------------------------------------===//

/// Check if target requires VALU → readfirstlane hazard mitigation
static bool needsVALUReadFirstLaneHazard(TargetAttrInterface target) {
  // gfx940+ (CDNA3/4) architectures need this hazard mitigation
  return isa<GFX942TargetAttr, GFX950TargetAttr, GFX1250TargetAttr>(target);
}

//===----------------------------------------------------------------------===//
// Hazard Mitigation Pass
//===----------------------------------------------------------------------===//

struct HazardMitigationPass
    : public waveasm::impl::WAVEASMHazardMitigationBase<HazardMitigationPass> {
  using WAVEASMHazardMitigationBase::WAVEASMHazardMitigationBase;

  void runOnOperation() override {
    Operation *module = getOperation();

    // Parse target arch from option.
    std::optional<TargetKind> parsed = symbolizeTargetKind(targetArch);
    if (!parsed) {
      module->emitError() << "Invalid target architecture: '" << targetArch
                          << "'. Supported targets: gfx942, gfx950, gfx1250";
      return signalPassFailure();
    }
    targetKindEnum = *parsed;

    // Process each program.
    module->walk([&](ProgramOp program) { processProgram(program); });
  }

private:
  TargetKind targetKindEnum = TargetKind::GFX942;
  unsigned numNopsInserted = 0;

  void processProgram(ProgramOp program) {
    TargetAttrInterface targetKind;
    // Get target from program if available.
    if (auto targetAttr = program.getTarget()) {
      targetKind = targetAttr.getTargetKind();
    } else {
      targetKind = getTargetKindAttr(program.getContext(), targetKindEnum);
    }

    // Check if this target needs VALU → readfirstlane hazard mitigation
    bool needsVALUHazard = needsVALUReadFirstLaneHazard(targetKind);
    if (!needsVALUHazard)
      return;

    // Collect operations in order, recursively walking into while/if bodies
    llvm::SmallVector<Operation *> ops;
    collectOpsRecursive(program.getBodyBlock(), ops);

    // Scan for hazards and collect insertion points.
    // Non-emitting ops (constants, extracts, precolored refs) don't produce
    // assembly instructions, so we look past them to find the preceding
    // emitting instruction that would actually be adjacent in the output.
    llvm::SmallVector<Operation *> insertionPoints;

    for (size_t i = 0; i < ops.size(); ++i) {
      Operation *op = ops[i];

      if (isReadfirstlaneOp(op)) {
        Operation *pred = findPrecedingEmittingOp(ops, i);
        if (pred && isVALUOp(pred) && hasVGPRConflict(pred, op))
          insertionPoints.push_back(op);
      }

      // Transcendental instructions (v_rcp_f32, v_rsq_f32, etc.) have a
      // one-cycle forwarding hazard when a non-Trans VALU immediately
      // consumes the result. Insert s_nop 0 to cover the required wait
      // state. See LLVM GCNHazardRecognizer::checkVALUHazards,
      // TransDefWaitstates = 1.
      // The consumer must be a non-Trans VALU; Trans can forward to Trans
      // without penalty. See LLVM GCNHazardRecognizer::checkVALUHazards,
      // guard: !SIInstrInfo::isTRANS(*VALU).
      if (isVALUOp(op) && !isTransOp(op) && i > 0) {
        Operation *pred = findPrecedingEmittingOp(ops, i);
        if (pred && isTransOp(pred) && hasVGPRConflict(pred, op))
          insertionPoints.push_back(op);
      }
    }

    // Insert s_nop instructions
    for (Operation *insertBefore : insertionPoints) {
      OpBuilder builder(insertBefore);
      S_NOP::create(builder, insertBefore->getLoc(),
                    builder.getI32IntegerAttr(0));
      numNopsInserted++;
    }
  }
};

} // namespace
