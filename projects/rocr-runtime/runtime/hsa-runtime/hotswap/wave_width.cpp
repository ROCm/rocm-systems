////////////////////////////////////////////////////////////////////////////////
//
// Wave Width Translation: wave32 (RDNA/GFX12) -> wave64 (CDNA/GFX9)
//
////////////////////////////////////////////////////////////////////////////////

#include "wave_width.hpp"

#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/Builders.h>

using namespace mlir;

//===----------------------------------------------------------------------===//
// IR construction helpers
//===----------------------------------------------------------------------===//

namespace {

Value createPhysSReg(OpBuilder &builder, Location loc, MLIRContext *ctx,
                     int64_t index, int64_t size) {
  auto ty = waveasm::PSRegType::get(ctx, index, size);
  OperationState state(loc, "waveasm.precolored.sreg");
  state.addTypes({ty});
  state.addAttribute("index",
                     builder.getIntegerAttr(builder.getI64Type(), index));
  state.addAttribute("size",
                     builder.getIntegerAttr(builder.getI64Type(), size));
  return builder.create(state)->getResult(0);
}

Value createImm(OpBuilder &builder, Location loc, MLIRContext *ctx,
                int64_t value) {
  auto ty = waveasm::ImmType::get(ctx, value);
  OperationState state(loc, "waveasm.constant");
  state.addTypes({ty});
  state.addAttribute("value", builder.getI64IntegerAttr(value));
  return builder.create(state)->getResult(0);
}

Operation *createOp(OpBuilder &builder, Location loc, llvm::StringRef mnemonic,
                    TypeRange resultTypes, ValueRange operands,
                    ArrayRef<NamedAttribute> attrs = {}) {
  OperationState state(loc, ("waveasm." + mnemonic).str());
  state.addTypes(resultTypes);
  state.addOperands(operands);
  state.addAttributes(attrs);
  return builder.create(state);
}

void insertExecHiClear(OpBuilder &builder, Location loc, MLIRContext *ctx) {
  auto execHiType = waveasm::PSRegType::get(ctx, 127, 1);
  auto zero = createImm(builder, loc, ctx, 0);
  createOp(builder, loc, "s_mov_b32", {execHiType}, {zero});
}

void insertVccHiClear(OpBuilder &builder, Location loc, MLIRContext *ctx) {
  auto vccHiType = waveasm::PSRegType::get(ctx, 107, 1);
  auto zero = createImm(builder, loc, ctx, 0);
  createOp(builder, loc, "s_mov_b32", {vccHiType}, {zero});
}

//===----------------------------------------------------------------------===//
// Pattern detection
//===----------------------------------------------------------------------===//

bool isSaveexecB32(llvm::StringRef name) {
  return name.contains("saveexec_b32");
}

llvm::StringRef getSaveexecAluOp(llvm::StringRef name) {
  if (name.contains("s_and_not1_saveexec") ||
      name.contains("s_andn2_saveexec"))
    return "s_andn2_b32";
  if (name.contains("s_and_saveexec"))
    return "s_and_b32";
  if (name.contains("s_or_saveexec"))
    return "s_or_b32";
  if (name.contains("s_xor_saveexec"))
    return "s_xor_b32";
  if (name.contains("s_or_not1_saveexec") ||
      name.contains("s_orn2_saveexec"))
    return "s_orn2_b32";
  return "";
}

bool isVcmpx(llvm::StringRef mnem) {
  return mnem.starts_with("v_cmpx_");
}

bool isVcmp(llvm::StringRef mnem) {
  return mnem.starts_with("v_cmp_") && !mnem.starts_with("v_cmpx_");
}

bool writesExecLo(Operation *op) {
  if (op->getName().getStringRef().contains("precolored"))
    return false;
  for (auto result : op->getResults()) {
    if (auto ps = dyn_cast<waveasm::PSRegType>(result.getType()))
      if (ps.getIndex() == 126)
        return true;
  }
  return false;
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Main pass
//===----------------------------------------------------------------------===//

LogicalResult hotswap::widenExecMask(ModuleOp module) {
  auto *ctx = module.getContext();
  constexpr int64_t kTempSgpr = 16;

  module->walk([&](waveasm::ProgramOp program) {
    Block &body = program.getBody().front();
    OpBuilder builder(ctx);

    // Collect ops by category (cannot modify block during iteration).
    struct SaveexecInfo {
      Operation *op;
      llvm::StringRef aluOp;
    };
    llvm::SmallVector<SaveexecInfo> saveexecOps;
    llvm::SmallVector<Operation *> vcmpxOps;
    llvm::SmallVector<Operation *> vcmpOps;
    llvm::SmallVector<Operation *> execWriteOps;

    for (Operation &op : body) {
      auto name = op.getName().getStringRef();
      llvm::StringRef mnem =
          name.starts_with("waveasm.") ? name.drop_front(8) : name;

      if (isSaveexecB32(name)) {
        auto alu = getSaveexecAluOp(name);
        if (!alu.empty())
          saveexecOps.push_back({&op, alu});
      } else if (isVcmpx(mnem)) {
        vcmpxOps.push_back(&op);
      } else if (isVcmp(mnem)) {
        vcmpOps.push_back(&op);
      } else if (writesExecLo(&op)) {
        execWriteOps.push_back(&op);
      }
    }

    // ── 1. Expand saveexec_b32 ──────────────────────────────────────────────
    //
    //   s_and_saveexec_b32 dst, src
    //   →  s_mov_b32 dst, exec_lo        (save old exec)
    //      s_and_b32 exec_lo, exec_lo, src (mask exec)
    //      s_mov_b32 exec_hi, 0          (keep upper lanes disabled)
    //
    for (auto &info : saveexecOps) {
      Operation *op = info.op;
      builder.setInsertionPoint(op);
      auto loc = op->getLoc();

      Type dstType = op->getResult(0).getType();
      Value srcVal =
          (op->getNumOperands() > 0) ? op->getOperand(0) : Value();

      auto execLoType = waveasm::PSRegType::get(ctx, 126, 1);

      // s_mov_b32 dst, exec_lo
      auto execLo = createPhysSReg(builder, loc, ctx, 126, 1);
      auto *movOp = createOp(builder, loc, "s_mov_b32", {dstType}, {execLo});

      // s_<alu>_b32 exec_lo, exec_lo, src
      auto execLo2 = createPhysSReg(builder, loc, ctx, 126, 1);
      llvm::SmallVector<Value> aluOperands = {execLo2};
      if (srcVal)
        aluOperands.push_back(srcVal);
      createOp(builder, loc, info.aluOp, {execLoType}, aluOperands);

      // s_mov_b32 exec_hi, 0
      insertExecHiClear(builder, loc, ctx);

      op->getResult(0).replaceAllUsesWith(movOp->getResult(0));
      op->erase();
    }

    // ── 2. Convert v_cmpx → v_cmp + manual exec AND ────────────────────────
    //
    //   v_cmpx_<op> src0, src1
    //   →  s_mov_b32 s{temp}, vcc_lo     (save vcc)
    //      v_cmp_<op> src0, src1          (writes vcc_lo)
    //      s_and_b32 exec_lo, exec_lo, vcc_lo
    //      s_mov_b32 exec_hi, 0
    //      s_mov_b32 vcc_lo, s{temp}     (restore vcc)
    //
    for (Operation *op : vcmpxOps) {
      builder.setInsertionPoint(op);
      auto loc = op->getLoc();
      auto name = op->getName().getStringRef();
      llvm::StringRef mnem =
          name.starts_with("waveasm.") ? name.drop_front(8) : name;

      // v_cmpx_le_f32 → v_cmp_le_f32 (remove the 'x')
      std::string newMnem =
          ("v_cmp_" + mnem.drop_front(7) /* drop "v_cmpx_" */).str();

      auto tempType = waveasm::PSRegType::get(ctx, kTempSgpr, 1);
      auto vccLoType = waveasm::PSRegType::get(ctx, 106, 1);
      auto execLoType = waveasm::PSRegType::get(ctx, 126, 1);

      // Save vcc_lo to temp
      auto vccLo = createPhysSReg(builder, loc, ctx, 106, 1);
      createOp(builder, loc, "s_mov_b32", {tempType}, {vccLo});

      // v_cmp_<op> (same operands, writes vcc_lo)
      createOp(builder, loc, newMnem, op->getResultTypes(),
               op->getOperands());

      // s_and_b32 exec_lo, exec_lo, vcc_lo
      auto execLo = createPhysSReg(builder, loc, ctx, 126, 1);
      auto vccLo2 = createPhysSReg(builder, loc, ctx, 106, 1);
      createOp(builder, loc, "s_and_b32", {execLoType}, {execLo, vccLo2});

      // s_mov_b32 exec_hi, 0
      insertExecHiClear(builder, loc, ctx);

      // Restore vcc_lo from temp
      auto temp = createPhysSReg(builder, loc, ctx, kTempSgpr, 1);
      createOp(builder, loc, "s_mov_b32", {vccLoType}, {temp});

      // Replace uses and erase
      if (op->getNumResults() > 0) {
        for (unsigned i = 0; i < op->getNumResults(); ++i) {
          if (!op->getResult(i).use_empty())
            op->getResult(i).replaceAllUsesWith(
                createPhysSReg(builder, loc, ctx, 126, 1));
        }
      }
      op->erase();
    }

    // ── 3. Clear vcc_hi after v_cmp ─────────────────────────────────────────
    for (Operation *op : vcmpOps) {
      builder.setInsertionPointAfter(op);
      insertVccHiClear(builder, op->getLoc(), ctx);
    }

    // ── 4. Clear exec_hi after any other exec_lo write ──────────────────────
    for (Operation *op : execWriteOps) {
      builder.setInsertionPointAfter(op);
      insertExecHiClear(builder, op->getLoc(), ctx);
    }
  });

  return success();
}
