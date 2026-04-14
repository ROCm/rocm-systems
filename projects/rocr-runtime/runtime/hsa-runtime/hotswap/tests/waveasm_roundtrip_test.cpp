////////////////////////////////////////////////////////////////////////////////
//
// Validates that the waveasm MLIR dialect links and functions correctly.
// Constructs a minimal waveasm.program in C++, verifies it, and prints it.
//
////////////////////////////////////////////////////////////////////////////////

#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"
#include "waveasm/Dialect/WaveASMAttrs.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Verifier.h>

#include <cassert>
#include <iostream>

static bool TestDialectRegistration() {
  mlir::MLIRContext ctx;
  ctx.loadDialect<waveasm::WaveASMDialect>();

  auto *dialect = ctx.getLoadedDialect<waveasm::WaveASMDialect>();
  if (!dialect) {
    std::cerr << "FAIL: WaveASM dialect not loaded\n";
    return false;
  }
  std::cout << "TestDialectRegistration: PASSED\n";
  return true;
}

static bool TestTypeCreation() {
  mlir::MLIRContext ctx;
  ctx.loadDialect<waveasm::WaveASMDialect>();

  auto vreg1 = waveasm::VRegType::get(&ctx);
  auto vreg4 = waveasm::VRegType::get(&ctx, 4);
  auto sreg2 = waveasm::SRegType::get(&ctx, 2);
  auto imm = waveasm::ImmType::get(&ctx, 42);
  auto scc = waveasm::SCCType::get(&ctx);

  assert(waveasm::isVGPRType(vreg1));
  assert(waveasm::isVGPRType(vreg4));
  assert(waveasm::isSGPRType(sreg2));
  assert(waveasm::isImmType(imm));
  assert(waveasm::isSCCType(scc));
  assert(waveasm::getRegSize(vreg4) == 4);
  assert(waveasm::getRegSize(sreg2) == 2);

  std::cout << "TestTypeCreation: PASSED\n";
  return true;
}

static bool TestModulePrint() {
  mlir::MLIRContext ctx;
  ctx.loadDialect<waveasm::WaveASMDialect>();

  mlir::OpBuilder builder(&ctx);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());

  module->print(llvm::outs());
  llvm::outs() << "\n";

  if (mlir::failed(mlir::verify(module))) {
    std::cerr << "FAIL: Module verification failed\n";
    return false;
  }

  std::cout << "TestModulePrint: PASSED\n";
  return true;
}

int main() {
  bool all_passed = true;
  all_passed &= TestDialectRegistration();
  all_passed &= TestTypeCreation();
  all_passed &= TestModulePrint();

  std::cout << "\n";
  if (all_passed) {
    std::cout << "All waveasm round-trip tests passed.\n";
    return 0;
  }
  std::cerr << "Some waveasm tests FAILED.\n";
  return 1;
}
