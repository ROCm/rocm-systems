////////////////////////////////////////////////////////////////////////////////
//
// SSA Construction: physical register refs -> def-use chains
//
// For each waveasm.program body block:
// 1. Walk ops in order, tracking the latest SSA value for each physical reg
// 2. Replace uses of precolored.{vreg,sreg,areg} with the SSA value from
//    the most recent definition of that register
// 3. Convert result types from physical (pvreg) to virtual (vreg)
// 4. Remove dead precolored ops
//
////////////////////////////////////////////////////////////////////////////////

#include "ssa_construction.hpp"

#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

using namespace mlir;

/// Key for register tracking: (register class, base index).
/// Register classes: 0 = VGPR, 1 = SGPR, 2 = AGPR, 3 = special
using RegKey = std::pair<int, int64_t>;

static RegKey getRegKey(Type ty) {
  if (auto pv = dyn_cast<waveasm::PVRegType>(ty))
    return {0, pv.getIndex()};
  if (auto ps = dyn_cast<waveasm::PSRegType>(ty))
    return {1, ps.getIndex()};
  if (auto pa = dyn_cast<waveasm::PARegType>(ty))
    return {2, pa.getIndex()};
  return {-1, 0};
}

/// Convert a physical register type to its virtual counterpart.
static Type physToVirtual(Type ty, MLIRContext *ctx) {
  if (auto pv = dyn_cast<waveasm::PVRegType>(ty))
    return waveasm::VRegType::get(ctx, pv.getSize());
  if (auto ps = dyn_cast<waveasm::PSRegType>(ty))
    return waveasm::SRegType::get(ctx, ps.getSize());
  if (auto pa = dyn_cast<waveasm::PARegType>(ty))
    return waveasm::ARegType::get(ctx, pa.getSize());
  return ty;
}

/// Check if an op is a precolored register materialization.
static bool isPrecoloredOp(Operation *op) {
  auto name = op->getName().getStringRef();
  return name == "waveasm.precolored.vreg" ||
         name == "waveasm.precolored.sreg" ||
         name == "waveasm.precolored.areg";
}

LogicalResult hotswap::constructSSA(ModuleOp module) {
  auto *ctx = module.getContext();

  module->walk([&](waveasm::ProgramOp program) {
    Block &body = program.getBody().front();

    // regDefs maps physical register key -> latest SSA value defining it.
    llvm::DenseMap<RegKey, Value> regDefs;
    llvm::SmallVector<Operation *> deadOps;

    for (Operation &op : llvm::make_early_inc_range(body)) {
      if (isPrecoloredOp(&op)) {
        // This op materializes a physical register. If we already have an
        // SSA value for this register, replace all uses and mark dead.
        if (op.getNumResults() == 0)
          continue;
        auto result = op.getResult(0);
        auto key = getRegKey(result.getType());
        if (key.first < 0)
          continue;

        auto it = regDefs.find(key);
        if (it != regDefs.end()) {
          // Replace uses of this precolored op with the existing def.
          // The types may differ (physical vs virtual), but MLIR allows
          // replacing with compatible types in our dialect.
          result.replaceAllUsesWith(it->second);
          deadOps.push_back(&op);
        } else {
          // First reference to this register. Keep the precolored op as
          // the initial definition (e.g., ABI register).
          // Convert to virtual type.
          auto newType = physToVirtual(result.getType(), ctx);
          result.setType(newType);
          regDefs[key] = result;
        }
        continue;
      }

      // For non-precolored ops: rewire operands and update result defs.

      // Rewire operands: if an operand comes from a precolored op that
      // was kept (i.e., an ABI initial def), it's already correct.
      // For operands coming from replaced precolored ops, RAUW handled it.

      // Update result types and register defs.
      for (unsigned i = 0; i < op.getNumResults(); ++i) {
        auto result = op.getResult(i);
        auto key = getRegKey(result.getType());
        if (key.first < 0)
          continue;

        auto newType = physToVirtual(result.getType(), ctx);
        result.setType(newType);
        regDefs[key] = result;
      }
    }

    // Erase dead precolored ops (in reverse to respect dominance).
    for (auto it = deadOps.rbegin(); it != deadOps.rend(); ++it) {
      if ((*it)->use_empty())
        (*it)->erase();
    }
  });

  return success();
}
