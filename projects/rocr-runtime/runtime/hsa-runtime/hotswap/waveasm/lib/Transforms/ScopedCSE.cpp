// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Scoped CSE Pass - Common Subexpression Elimination respecting control flow
//
// This pass implements scoped CSE for WAVEASM IR. Unlike global CSE, it
// respects structured control flow by maintaining a scope stack:
// - Global scope: Values that dominate all uses
// - Region scopes: Pushed/popped for structured regions (loops, ifs)
//
// Only operations within the same or dominating scope can be deduplicated.
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMInterfaces.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "waveasm-scoped-cse"

using namespace mlir;
using namespace waveasm;

namespace waveasm {
#define GEN_PASS_DEF_WAVEASMSCOPEDCSE
#include "waveasm/Transforms/Passes.h.inc"
} // namespace waveasm

namespace {

//===----------------------------------------------------------------------===//
// Expression Key for CSE
//===----------------------------------------------------------------------===//

/// Key for identifying equivalent operations for CSE purposes.
/// Two operations are equivalent if they have the same opcode and operands.
/// For commutative ops, operands are sorted to canonicalize ordering.
struct ExpressionKey {
  StringRef opName;
  SmallVector<Value, 4> operands;
  SmallVector<Attribute, 2> attrs;
  SmallVector<Type, 2> resultTypes;

  ExpressionKey() : opName("") {}

  ExpressionKey(Operation *op) : opName(op->getName().getStringRef()) {
    operands.append(op->operand_begin(), op->operand_end());
    // For commutative binary ops, sort operands so that
    // v_add_u32(a, b) and v_add_u32(b, a) have the same key.
    if (operands.size() == 2 && op->hasTrait<OpTrait::IsCommutative>()) {
      if (operands[0].getAsOpaquePointer() > operands[1].getAsOpaquePointer())
        std::swap(operands[0], operands[1]);
    }
    // Include relevant attributes (e.g., immediate values)
    for (auto attr : op->getAttrs()) {
      // Skip unimportant attributes
      if (attr.getName() == "sym_name")
        continue;
      attrs.push_back(attr.getValue());
    }
    // Include result types to prevent merging ops with different result sizes
    // (e.g., v_mov_b32 -> vreg vs v_mov_b32 -> vreg<4,4>)
    for (Type ty : op->getResultTypes()) {
      resultTypes.push_back(ty);
    }
  }

  bool operator==(const ExpressionKey &rhs) const {
    return opName == rhs.opName && operands == rhs.operands &&
           attrs == rhs.attrs && resultTypes == rhs.resultTypes;
  }

  bool isEmpty() const { return opName.empty(); }
};

} // namespace

namespace llvm {
template <> struct DenseMapInfo<ExpressionKey> {
  static ExpressionKey getEmptyKey() {
    ExpressionKey key;
    return key;
  }

  static ExpressionKey getTombstoneKey() {
    ExpressionKey key;
    key.opName = "<<TOMBSTONE>>";
    return key;
  }

  static unsigned getHashValue(const ExpressionKey &key) {
    unsigned hash = llvm::hash_value(key.opName);
    for (Value v : key.operands) {
      hash = llvm::hash_combine(hash, mlir::hash_value(v));
    }
    for (Attribute a : key.attrs) {
      hash = llvm::hash_combine(hash, mlir::hash_value(a));
    }
    for (Type t : key.resultTypes) {
      hash = llvm::hash_combine(hash, mlir::hash_value(t));
    }
    return hash;
  }

  static bool isEqual(const ExpressionKey &lhs, const ExpressionKey &rhs) {
    // Handle empty/tombstone keys
    if (lhs.opName.data() == nullptr && rhs.opName.data() == nullptr)
      return true;
    if (lhs.opName.data() == nullptr || rhs.opName.data() == nullptr)
      return false;
    return lhs == rhs;
  }
};
} // namespace llvm

namespace {

//===----------------------------------------------------------------------===//
// Scoped CSE Implementation
//===----------------------------------------------------------------------===//

/// Check if an operation is eligible for CSE (pure, no side effects)
/// Uses MLIR traits instead of string matching for type safety.
bool isCSEEligible(Operation *op) {
  // Exclude operations explicitly marked as non-CSE-able (e.g., accumulator
  // inits)
  if (op->hasAttr("no_cse"))
    return false;

  // Exclude operations with side effects
  if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
    return false;

  // Memory operations are not eligible (have side effects)
  if (op->hasTrait<OpTrait::MemoryOp>())
    return false;

  // Control flow operations are not eligible
  if (op->hasTrait<OpTrait::ControlFlowOp>())
    return false;

  // Special register operations are not eligible (M0, exec, vcc)
  if (op->hasTrait<OpTrait::SpecialRegOp>())
    return false;

  // SCC-reading ops are NOT CSE-eligible: their result depends on implicit
  // SCC state, so two ops with identical operands can produce different
  // results.
  if (op->hasTrait<OpTrait::SCCUse>())
    return false;

  // Precolored registers are not eligible (they're fixed)
  if (isa<PrecoloredSRegOp, PrecoloredVRegOp>(op))
    return false;

  // Constants are always eligible
  if (isa<ConstantOp>(op))
    return true;

  // Arithmetic operations are eligible (has ArithmeticOp trait)
  if (op->hasTrait<OpTrait::ArithmeticOp>())
    return true;

  // Pure operations without side effects are eligible
  if (isPure(op))
    return true;

  // Default: not eligible
  return false;
}

/// Scope-aware CSE using a manual scope stack
class ScopedCSEDriver {
public:
  ScopedCSEDriver() = default;

  /// Process a program, performing scoped CSE
  void processProgram(ProgramOp program) {
    opsRemoved = 0;

    // Start with empty scope stack
    scopeStack.clear();
    pushScope();

    // Process the program body
    processBlock(&program.getBody().front());

    popScope();
  }

  unsigned getOpsRemoved() const { return opsRemoved; }

private:
  using ScopeMap = llvm::DenseMap<ExpressionKey, Operation *>;
  SmallVector<ScopeMap, 8> scopeStack;
  unsigned opsRemoved = 0;

  void pushScope() { scopeStack.push_back(ScopeMap()); }

  void popScope() {
    if (!scopeStack.empty())
      scopeStack.pop_back();
  }

  /// Look up an expression in all active scopes (innermost to outermost)
  Operation *lookup(const ExpressionKey &key) {
    for (auto it = scopeStack.rbegin(); it != scopeStack.rend(); ++it) {
      auto found = it->find(key);
      if (found != it->end())
        return found->second;
    }
    return nullptr;
  }

  /// Insert an expression into the current (innermost) scope
  void insert(const ExpressionKey &key, Operation *op) {
    if (!scopeStack.empty())
      scopeStack.back()[key] = op;
  }

  void processBlock(Block *block) {
    SmallVector<Operation *, 16> opsToErase;

    for (Operation &op : *block) {
      // Process nested regions first (with new scope)
      for (Region &region : op.getRegions()) {
        pushScope();
        for (Block &nestedBlock : region) {
          processBlock(&nestedBlock);
        }
        popScope();
      }

      // Skip non-eligible operations
      if (!isCSEEligible(&op))
        continue;

      // Skip operations with no results
      if (op.getNumResults() == 0)
        continue;

      ExpressionKey key(&op);

      // Check if we've seen this expression before
      if (Operation *existing = lookup(key)) {
        // Replace all uses with the existing operation's results
        op.replaceAllUsesWith(existing);
        opsToErase.push_back(&op);
        opsRemoved++;
      } else {
        // Record this expression
        insert(key, &op);
      }
    }

    // Erase redundant operations
    for (Operation *op : opsToErase) {
      op->erase();
    }
  }
};

//===----------------------------------------------------------------------===//
// Scoped CSE Pass
//===----------------------------------------------------------------------===//

struct ScopedCSEPass
    : public waveasm::impl::WAVEASMScopedCSEBase<ScopedCSEPass> {
  using WAVEASMScopedCSEBase::WAVEASMScopedCSEBase;

  void runOnOperation() override {
    Operation *module = getOperation();
    unsigned totalRemoved = 0;

    // Process each program.
    module->walk([&](ProgramOp program) {
      ScopedCSEDriver driver;
      driver.processProgram(program);
      totalRemoved += driver.getOpsRemoved();
    });

    LLVM_DEBUG(if (totalRemoved > 0) {
      llvm::dbgs() << "ScopedCSE: removed " << totalRemoved
                   << " redundant operations\n";
    });
  }
};

} // namespace
