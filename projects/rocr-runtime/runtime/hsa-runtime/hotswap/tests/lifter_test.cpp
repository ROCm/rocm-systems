////////////////////////////////////////////////////////////////////////////////
//
// Tests for the binary lifter (MCInst -> waveasm MLIR ops).
//
// Each test feeds known AMDGPU instruction bytes to the Lifter and
// verifies the resulting waveasm IR.
//
////////////////////////////////////////////////////////////////////////////////

#include "../lifter.hpp"
#include "../ssa_construction.hpp"
#include "../cross_target.hpp"
#include "../wave_width.hpp"
#include "../emit_assembly.hpp"
#include "../pipeline.hpp"

#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>

#include <cassert>
#include <cstring>
#include <iostream>

/// Check if the module contains an op with the given name (e.g., "waveasm.s_endpgm").
static bool hasOp(mlir::ModuleOp module, llvm::StringRef opName) {
  bool found = false;
  module->walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == opName)
      found = true;
  });
  return found;
}

/// Count ops with the given name.
static int countOps(mlir::ModuleOp module, llvm::StringRef opName) {
  int count = 0;
  module->walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == opName)
      ++count;
  });
  return count;
}

//===----------------------------------------------------------------------===//
// Test: Lifter produces a valid ModuleOp with waveasm.program
//===----------------------------------------------------------------------===//

static bool TestLifterBasicStructure() {
  mlir::MLIRContext ctx;
  hotswap::Lifter lifter(ctx);

  // s_endpgm = 0xBF810000 (little-endian)
  uint8_t bytes[] = {0x00, 0x00, 0x81, 0xBF};
  auto module = lifter.lift(bytes, "gfx942", "test_kernel");

  if (!module) {
    std::cerr << "FAIL: Lifter returned null\n";
    return false;
  }

  if (!hasOp(*module, "waveasm.program")) {
    std::cerr << "FAIL: No waveasm.program found\n";
    return false;
  }

  auto &stats = lifter.getStats();
  if (stats.totalInstructions == 0) {
    std::cerr << "FAIL: No instructions processed\n";
    return false;
  }

  std::cout << "TestLifterBasicStructure: PASSED"
            << " (total=" << stats.totalInstructions
            << " lifted=" << stats.liftedInstructions
            << " raw=" << stats.rawFallbacks
            << " failed=" << stats.failedDisassembly << ")\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Test: Lifter disassembles and lifts s_endpgm
//===----------------------------------------------------------------------===//

static bool TestLifterEndpgm() {
  mlir::MLIRContext ctx;
  hotswap::Lifter lifter(ctx);

  uint8_t bytes[] = {0x00, 0x00, 0x81, 0xBF};
  auto module = lifter.lift(bytes, "gfx942", "test_endpgm");
  if (!module) {
    std::cerr << "FAIL: Lifter returned null\n";
    return false;
  }

  if (!hasOp(*module, "waveasm.s_endpgm")) {
    std::cerr << "FAIL: No waveasm.s_endpgm op found. IR:\n";
    module->print(llvm::errs());
    return false;
  }

  std::cout << "TestLifterEndpgm: PASSED\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Test: Lifter handles multiple instructions
//===----------------------------------------------------------------------===//

static bool TestLifterMultipleInstructions() {
  mlir::MLIRContext ctx;
  hotswap::Lifter lifter(ctx);

  // s_nop 0 = 0xBF800000, s_endpgm = 0xBF810000
  uint8_t bytes[] = {
      0x00, 0x00, 0x80, 0xBF, // s_nop 0
      0x00, 0x00, 0x81, 0xBF, // s_endpgm
  };

  auto module = lifter.lift(bytes, "gfx942", "test_multi");
  if (!module) {
    std::cerr << "FAIL: Lifter returned null\n";
    return false;
  }

  auto &stats = lifter.getStats();
  if (stats.totalInstructions != 2) {
    std::cerr << "FAIL: Expected 2 instructions, got "
              << stats.totalInstructions << "\n";
    module->print(llvm::errs());
    return false;
  }

  std::cout << "TestLifterMultipleInstructions: PASSED"
            << " (lifted=" << stats.liftedInstructions
            << " raw=" << stats.rawFallbacks << ")\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Test: Lifter handles failed disassembly gracefully
//===----------------------------------------------------------------------===//

static bool TestLifterInvalidBytes() {
  mlir::MLIRContext ctx;
  hotswap::Lifter lifter(ctx);

  // Some bytes that may not decode + s_endpgm
  uint8_t bytes[] = {
      0xFF, 0xFF, 0xFF, 0xFF, // possibly invalid
      0x00, 0x00, 0x81, 0xBF, // s_endpgm
  };

  auto module = lifter.lift(bytes, "gfx942", "test_invalid");
  if (!module) {
    std::cerr << "FAIL: Lifter returned null\n";
    return false;
  }

  auto &stats = lifter.getStats();
  if (stats.totalInstructions < 1) {
    std::cerr << "FAIL: Expected at least 1 instruction\n";
    return false;
  }

  std::cout << "TestLifterInvalidBytes: PASSED"
            << " (total=" << stats.totalInstructions
            << " failed_disasm=" << stats.failedDisassembly
            << " raw=" << stats.rawFallbacks << ")\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Test: Lifter prints IR output
//===----------------------------------------------------------------------===//

static bool TestLifterPrintsIR() {
  mlir::MLIRContext ctx;
  hotswap::Lifter lifter(ctx);

  // s_nop 0, s_nop 1, s_barrier, s_endpgm
  uint8_t bytes[] = {
      0x00, 0x00, 0x80, 0xBF, // s_nop 0
      0x01, 0x00, 0x80, 0xBF, // s_nop 1
      0x00, 0x00, 0x81, 0xBF, // s_endpgm
  };

  auto module = lifter.lift(bytes, "gfx942", "test_print");
  if (!module) {
    std::cerr << "FAIL: Lifter returned null\n";
    return false;
  }

  std::cout << "--- Lifted IR ---\n";
  module->print(llvm::outs());
  llvm::outs() << "\n--- End IR ---\n";

  std::cout << "TestLifterPrintsIR: PASSED\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Test: Lifter targets different ISAs
//===----------------------------------------------------------------------===//

static bool TestLifterMultipleISAs() {
  uint8_t bytes[] = {0x00, 0x00, 0x81, 0xBF};

  for (const char *isa : {"gfx942", "gfx950"}) {
    mlir::MLIRContext ctx;
    hotswap::Lifter lifter(ctx);
    auto module = lifter.lift(bytes, isa, "test_isa");
    if (!module) {
      std::cerr << "FAIL: Lifter returned null for " << isa << "\n";
      return false;
    }
    auto &stats = lifter.getStats();
    if (stats.totalInstructions == 0) {
      std::cerr << "FAIL: No instructions for " << isa << "\n";
      return false;
    }
  }

  std::cout << "TestLifterMultipleISAs: PASSED\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Test: Lifter handles real VALU / SALU / memory / DS instruction sequences
//===----------------------------------------------------------------------===//

static bool TestLifterRealInstructions() {
  mlir::MLIRContext ctx;
  hotswap::Lifter lifter(ctx);

  // Real gfx942 instruction bytes from llvm-mc:
  //   v_add_f32_e32 v0, v1, v2          = 02 00 05 01 -> LE: 01 05 00 02
  //   s_mov_b32 s0, s1                   = BE 80 00 01 -> LE: 01 00 80 BE
  //   v_mov_b32_e32 v3, v4               = 7E 06 03 04 -> LE: 04 03 06 7E
  //   global_load_dword v5, v[6:7], off  = DC 50 80 00 05 7F 00 06
  //   ds_read_b32 v8, v9                 = D8 6C 00 00 08 00 00 09
  //   s_endpgm                           = BF 81 00 00
  uint8_t bytes[] = {
      0x01, 0x05, 0x00, 0x02,             // v_add_f32_e32 v0, v1, v2
      0x01, 0x00, 0x80, 0xBE,             // s_mov_b32 s0, s1
      0x04, 0x03, 0x06, 0x7E,             // v_mov_b32_e32 v3, v4
      0x00, 0x80, 0x50, 0xDC,             // global_load_dword v5, v[6:7], off (lo)
      0x06, 0x00, 0x7F, 0x05,             // (hi)
      0x00, 0x00, 0x6C, 0xD8,             // ds_read_b32 v8, v9 (lo)
      0x09, 0x00, 0x00, 0x08,             // (hi)
      0x00, 0x00, 0x81, 0xBF,             // s_endpgm
  };

  auto module = lifter.lift(bytes, "gfx942", "test_real");
  if (!module) {
    std::cerr << "FAIL: Lifter returned null\n";
    return false;
  }

  auto &stats = lifter.getStats();
  std::cout << "TestLifterRealInstructions: "
            << stats.totalInstructions << " total, "
            << stats.liftedInstructions << " lifted, "
            << stats.rawFallbacks << " raw\n";

  std::cout << "--- Real instruction IR ---\n";
  module->print(llvm::outs());
  llvm::outs() << "\n--- End IR ---\n";

  if (stats.totalInstructions < 4) {
    std::cerr << "FAIL: Expected at least 4 instructions\n";
    return false;
  }

  std::cout << "TestLifterRealInstructions: PASSED\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Test: SSA construction creates proper def-use chains
//===----------------------------------------------------------------------===//

static bool TestSSAConstruction() {
  mlir::MLIRContext ctx;
  hotswap::Lifter lifter(ctx);

  // v_add_f32 v0, v1, v2 ; defines v0 using v1 and v2
  // v_mov_b32 v3, v0     ; uses v0 (should chain to the add result)
  // s_endpgm
  uint8_t bytes[] = {
      0x01, 0x05, 0x00, 0x02, // v_add_f32_e32 v0, v1, v2
      0x00, 0x03, 0x06, 0x7E, // v_mov_b32_e32 v3, v0
      0x00, 0x00, 0x81, 0xBF, // s_endpgm
  };

  auto module = lifter.lift(bytes, "gfx942", "test_ssa");
  if (!module) {
    std::cerr << "FAIL: Lifter returned null\n";
    return false;
  }

  std::cout << "--- Before SSA construction ---\n";
  module->print(llvm::outs());
  llvm::outs() << "\n";

  auto result = hotswap::constructSSA(*module);
  if (mlir::failed(result)) {
    std::cerr << "FAIL: SSA construction failed\n";
    return false;
  }

  std::cout << "--- After SSA construction ---\n";
  module->print(llvm::outs());
  llvm::outs() << "\n";

  // Verify that:
  // 1. Virtual types are used (vreg instead of pvreg)
  bool hasVirtual = false;
  module->walk([&](mlir::Operation *op) {
    for (auto result : op->getResults()) {
      if (mlir::isa<waveasm::VRegType>(result.getType()))
        hasVirtual = true;
    }
  });

  if (!hasVirtual) {
    std::cerr << "FAIL: No virtual register types after SSA construction\n";
    return false;
  }

  // 2. The v_mov_b32 should use the v_add_f32's result, not a fresh precolored
  bool foundChain = false;
  module->walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == "waveasm.v_mov_b32_e32") {
      for (auto operand : op->getOperands()) {
        if (auto defOp = operand.getDefiningOp()) {
          if (defOp->getName().getStringRef() == "waveasm.v_add_f32_e32")
            foundChain = true;
        }
      }
    }
  });

  if (!foundChain) {
    std::cerr << "FAIL: v_mov_b32 does not use v_add_f32 result\n";
    return false;
  }

  std::cout << "TestSSAConstruction: PASSED\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Test: Mnemonic mapping table
//===----------------------------------------------------------------------===//

static bool TestMnemonicMapping() {
  struct Case {
    const char *input;
    const char *expected;
  };
  Case cases[] = {
      {"global_load_b32", "global_load_dword"},
      {"global_store_b64", "global_store_dwordx2"},
      {"flat_load_b32", "flat_load_dword"},
      {"ds_load_b32", "ds_read_b32"},
      {"ds_store_b32", "ds_write_b32"},
      {"s_load_b64", "s_load_dwordx2"},
      {"s_add_co_u32", "s_add_u32"},
      {"v_max_num_f32", "v_max_f32"},
      {"v_add_nc_u32", "v_add_u32"},
      {"global_atomic_add_u32", "global_atomic_add"},
      // Identity mappings (no rename needed)
      {"v_add_f32_e32", "v_add_f32_e32"},
      {"s_mov_b32", "s_mov_b32"},
      {"s_endpgm", "s_endpgm"},
  };

  bool allOK = true;
  for (const auto &tc : cases) {
    auto result = hotswap::mapMnemonic(tc.input, "gfx1250", "gfx942");
    if (result != tc.expected) {
      std::cerr << "FAIL: mapMnemonic(\"" << tc.input << "\") = \""
                << result << "\", expected \"" << tc.expected << "\"\n";
      allOK = false;
    }
  }

  if (!allOK)
    return false;
  std::cout << "TestMnemonicMapping: PASSED ("
            << (sizeof(cases) / sizeof(cases[0])) << " cases)\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Test: Cross-target module retargeting
//===----------------------------------------------------------------------===//

static bool TestCrossTargetMapping() {
  mlir::MLIRContext ctx;
  hotswap::Lifter lifter(ctx);

  // Lift gfx1250 instructions: use the same bytes but label them gfx1250.
  // s_nop 0, s_endpgm — these don't need remapping, but the target attr should change.
  uint8_t bytes[] = {
      0x00, 0x00, 0x80, 0xBF, // s_nop 0
      0x00, 0x00, 0x81, 0xBF, // s_endpgm
  };

  auto module = lifter.lift(bytes, "gfx1250", "test_retarget");
  if (!module) {
    std::cerr << "FAIL: Lifter returned null\n";
    return false;
  }

  std::cout << "--- Before retarget ---\n";
  module->print(llvm::outs());
  llvm::outs() << "\n";

  auto result = hotswap::retargetModule(*module, "gfx1250", "gfx942");
  if (mlir::failed(result)) {
    std::cerr << "FAIL: retargetModule failed\n";
    return false;
  }

  std::cout << "--- After retarget ---\n";
  module->print(llvm::outs());
  llvm::outs() << "\n";

  // Verify target attribute changed to gfx942
  bool foundGfx942 = false;
  module->walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == "waveasm.program") {
      std::string attrStr;
      llvm::raw_string_ostream rso(attrStr);
      op->print(rso);
      if (attrStr.find("gfx942") != std::string::npos)
        foundGfx942 = true;
    }
  });

  if (!foundGfx942) {
    std::cerr << "FAIL: Target not updated to gfx942\n";
    return false;
  }

  std::cout << "TestCrossTargetMapping: PASSED\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Test: Branch/label resolution in the lifter
//===----------------------------------------------------------------------===//

static bool TestBranchForward() {
  mlir::MLIRContext ctx;
  hotswap::Lifter lifter(ctx);

  // PC 0:  s_nop 0           (0xBF800000)
  // PC 4:  s_cbranch_scc0 +1 (0xBF840001) → target = 4+4+1*4 = 12
  // PC 8:  s_nop 1           (0xBF800001)
  // PC 12: s_endpgm          (0xBF810000) ← branch target
  uint8_t bytes[] = {
      0x00, 0x00, 0x80, 0xBF, // s_nop 0
      0x01, 0x00, 0x84, 0xBF, // s_cbranch_scc0 +1
      0x01, 0x00, 0x80, 0xBF, // s_nop 1
      0x00, 0x00, 0x81, 0xBF, // s_endpgm
  };

  auto module = lifter.lift(bytes, "gfx942", "test_branch_fwd");
  if (!module) {
    std::cerr << "FAIL: Lifter returned null\n";
    return false;
  }

  std::cout << "--- Branch forward IR ---\n";
  module->print(llvm::outs());
  llvm::outs() << "\n";

  bool foundLabel = hasOp(*module, "waveasm.label");
  bool foundBranch = hasOp(*module, "waveasm.s_cbranch_scc0");

  if (!foundLabel) {
    std::cerr << "FAIL: No waveasm.label found\n";
    return false;
  }
  if (!foundBranch) {
    std::cerr << "FAIL: No waveasm.s_cbranch_scc0 found\n";
    return false;
  }

  // Verify label and branch reference the same symbol
  std::string labelName, branchTarget;
  module->walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == "waveasm.label") {
      if (auto attr = op->getAttrOfType<mlir::StringAttr>("sym_name"))
        labelName = attr.getValue().str();
    }
    if (op->getName().getStringRef() == "waveasm.s_cbranch_scc0") {
      if (auto attr = op->getAttrOfType<mlir::FlatSymbolRefAttr>("target"))
        branchTarget = attr.getValue().str();
    }
  });

  if (labelName.empty() || branchTarget.empty()) {
    std::cerr << "FAIL: Could not extract label/target names\n";
    return false;
  }
  if (labelName != branchTarget) {
    std::cerr << "FAIL: Label '" << labelName << "' != target '"
              << branchTarget << "'\n";
    return false;
  }

  std::cout << "TestBranchForward: PASSED (label=" << labelName << ")\n";
  return true;
}

static bool TestBranchBackward() {
  mlir::MLIRContext ctx;
  hotswap::Lifter lifter(ctx);

  // PC 0:  s_nop 0              (0xBF800000) ← backward branch target
  // PC 4:  s_cbranch_scc1 -2    (0xBF85FFFE) → target = 4+4+(-2)*4 = 0
  // PC 8:  s_endpgm             (0xBF810000)
  uint8_t bytes[] = {
      0x00, 0x00, 0x80, 0xBF, // s_nop 0
      0xFE, 0xFF, 0x85, 0xBF, // s_cbranch_scc1 -2
      0x00, 0x00, 0x81, 0xBF, // s_endpgm
  };

  auto module = lifter.lift(bytes, "gfx942", "test_branch_back");
  if (!module) {
    std::cerr << "FAIL: Lifter returned null\n";
    return false;
  }

  std::cout << "--- Branch backward IR ---\n";
  module->print(llvm::outs());
  llvm::outs() << "\n";

  bool foundLabel = hasOp(*module, "waveasm.label");
  bool foundBranch = hasOp(*module, "waveasm.s_cbranch_scc1");

  if (!foundLabel) {
    std::cerr << "FAIL: No waveasm.label found for backward target\n";
    return false;
  }
  if (!foundBranch) {
    std::cerr << "FAIL: No waveasm.s_cbranch_scc1 found\n";
    return false;
  }

  std::cout << "TestBranchBackward: PASSED\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Test: Wave width saveexec_b32 expansion
//===----------------------------------------------------------------------===//

static bool TestWaveWidthSaveexec() {
  mlir::MLIRContext ctx;
  ctx.loadDialect<waveasm::WaveASMDialect>();
  ctx.allowUnregisteredDialects();

  mlir::OpBuilder builder(&ctx);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(module.getBody());
  auto loc = builder.getUnknownLoc();

  auto targetAttr = waveasm::TargetAttr::get(
      &ctx, waveasm::GFX1250TargetAttr::get(&ctx), 5);
  auto abiAttr = waveasm::KernelABIAttr::get(
      &ctx, 0, 0, std::optional<int64_t>(4), std::nullopt, std::nullopt);
  auto program = builder.create<waveasm::ProgramOp>(
      loc, "test_wave", targetAttr, abiAttr);

  auto &body = program.getBody().front();
  builder.setInsertionPointToEnd(&body);

  // Create vcc_lo reference
  {
    auto vccType = waveasm::PSRegType::get(&ctx, 106, 1);
    mlir::OperationState vccState(loc, "waveasm.precolored.sreg");
    vccState.addTypes({vccType});
    vccState.addAttribute("index", builder.getI64IntegerAttr(106));
    vccState.addAttribute("size", builder.getI64IntegerAttr(1));
    auto vcc = builder.create(vccState)->getResult(0);

    // s_and_saveexec_b32 s4, vcc_lo
    auto dstType = waveasm::PSRegType::get(&ctx, 4, 1);
    mlir::OperationState saveState(loc, "waveasm.s_and_saveexec_b32");
    saveState.addTypes({dstType});
    saveState.addOperands({vcc});
    builder.create(saveState);
  }

  // s_endpgm
  {
    mlir::OperationState endState(loc, "waveasm.s_endpgm");
    builder.create(endState);
  }

  std::cout << "--- Before wave width ---\n";
  module.print(llvm::outs());
  llvm::outs() << "\n";

  auto result = hotswap::widenExecMask(module);
  if (mlir::failed(result)) {
    std::cerr << "FAIL: widenExecMask failed\n";
    return false;
  }

  std::cout << "--- After wave width ---\n";
  module.print(llvm::outs());
  llvm::outs() << "\n";

  // Verify: no saveexec_b32 should remain
  bool foundSaveexec = false;
  int movCount = 0, andCount = 0;
  module->walk([&](mlir::Operation *op) {
    auto name = op->getName().getStringRef();
    if (name.contains("saveexec_b32"))
      foundSaveexec = true;
    if (name == "waveasm.s_mov_b32")
      movCount++;
    if (name == "waveasm.s_and_b32")
      andCount++;
  });

  if (foundSaveexec) {
    std::cerr << "FAIL: saveexec_b32 not expanded\n";
    return false;
  }
  // Expect >= 2 s_mov_b32: save exec + exec_hi = 0
  if (movCount < 2) {
    std::cerr << "FAIL: Expected >= 2 s_mov_b32, got " << movCount << "\n";
    return false;
  }
  // Expect >= 1 s_and_b32: exec_lo &= src
  if (andCount < 1) {
    std::cerr << "FAIL: Expected >= 1 s_and_b32, got " << andCount << "\n";
    return false;
  }

  std::cout << "TestWaveWidthSaveexec: PASSED (mov=" << movCount
            << " and=" << andCount << ")\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Test: Wave width VCC clearing after v_cmp
//===----------------------------------------------------------------------===//

static bool TestWaveWidthVccClear() {
  mlir::MLIRContext ctx;
  ctx.loadDialect<waveasm::WaveASMDialect>();
  ctx.allowUnregisteredDialects();

  mlir::OpBuilder builder(&ctx);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(module.getBody());
  auto loc = builder.getUnknownLoc();

  auto targetAttr = waveasm::TargetAttr::get(
      &ctx, waveasm::GFX942TargetAttr::get(&ctx), 5);
  auto abiAttr = waveasm::KernelABIAttr::get(
      &ctx, 0, 0, std::optional<int64_t>(4), std::nullopt, std::nullopt);
  auto program = builder.create<waveasm::ProgramOp>(
      loc, "test_vcc", targetAttr, abiAttr);

  auto &body = program.getBody().front();
  builder.setInsertionPointToEnd(&body);

  // Create two VGPRs
  auto v0Type = waveasm::PVRegType::get(&ctx, 0, 1);
  auto v1Type = waveasm::PVRegType::get(&ctx, 1, 1);
  mlir::OperationState v0State(loc, "waveasm.precolored.vreg");
  v0State.addTypes({v0Type});
  v0State.addAttribute("index", builder.getI64IntegerAttr(0));
  v0State.addAttribute("size", builder.getI64IntegerAttr(1));
  auto v0 = builder.create(v0State)->getResult(0);

  mlir::OperationState v1State(loc, "waveasm.precolored.vreg");
  v1State.addTypes({v1Type});
  v1State.addAttribute("index", builder.getI64IntegerAttr(1));
  v1State.addAttribute("size", builder.getI64IntegerAttr(1));
  auto v1 = builder.create(v1State)->getResult(0);

  // v_cmp_le_f32 (writes vcc implicitly)
  auto sccType = waveasm::SCCType::get(&ctx);
  mlir::OperationState cmpState(loc, "waveasm.v_cmp_le_f32_e32");
  cmpState.addTypes({sccType});
  cmpState.addOperands({v0, v1});
  builder.create(cmpState);

  // s_endpgm
  mlir::OperationState endState(loc, "waveasm.s_endpgm");
  builder.create(endState);

  auto result = hotswap::widenExecMask(module);
  if (mlir::failed(result)) {
    std::cerr << "FAIL: widenExecMask failed\n";
    return false;
  }

  std::cout << "--- After VCC width ---\n";
  module.print(llvm::outs());
  llvm::outs() << "\n";

  // Verify vcc_hi clear was inserted (s_mov_b32 with psreg<107, 1>)
  bool foundVccHiClear = false;
  module->walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == "waveasm.s_mov_b32") {
      for (auto result : op->getResults()) {
        if (auto ps = mlir::dyn_cast<waveasm::PSRegType>(result.getType()))
          if (ps.getIndex() == 107)
            foundVccHiClear = true;
      }
    }
  });

  if (!foundVccHiClear) {
    std::cerr << "FAIL: No vcc_hi clear found after v_cmp\n";
    return false;
  }

  std::cout << "TestWaveWidthVccClear: PASSED\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Test: Full pipeline: lift → retarget → widen → SSA
//===----------------------------------------------------------------------===//

static bool TestFullPipeline() {
  mlir::MLIRContext ctx;
  hotswap::Lifter lifter(ctx);

  // GFX942 instructions: v_add_f32, s_nop, s_endpgm
  uint8_t bytes[] = {
      0x01, 0x05, 0x00, 0x02, // v_add_f32_e32 v0, v1, v2
      0x00, 0x00, 0x80, 0xBF, // s_nop 0
      0x00, 0x00, 0x81, 0xBF, // s_endpgm
  };

  auto module = lifter.lift(bytes, "gfx942", "test_pipeline");
  if (!module) {
    std::cerr << "FAIL: Lifter returned null\n";
    return false;
  }

  std::cout << "--- Pipeline step 1: Lifted ---\n";
  module->print(llvm::outs());
  llvm::outs() << "\n";

  // Cross-target (identity for gfx942→gfx942, just verify it doesn't break)
  auto r1 = hotswap::retargetModule(*module, "gfx942", "gfx942");
  if (mlir::failed(r1)) {
    std::cerr << "FAIL: retargetModule failed\n";
    return false;
  }

  // Wave width (no-op for same-width, verify it doesn't break)
  auto r2 = hotswap::widenExecMask(*module);
  if (mlir::failed(r2)) {
    std::cerr << "FAIL: widenExecMask failed\n";
    return false;
  }

  // SSA construction
  auto r3 = hotswap::constructSSA(*module);
  if (mlir::failed(r3)) {
    std::cerr << "FAIL: constructSSA failed\n";
    return false;
  }

  std::cout << "--- Pipeline step 4: After SSA ---\n";
  module->print(llvm::outs());
  llvm::outs() << "\n";

  // Verify module is still valid
  if (!hasOp(*module, "waveasm.program")) {
    std::cerr << "FAIL: No waveasm.program after pipeline\n";
    return false;
  }

  std::cout << "TestFullPipeline: PASSED\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Test: Assembly emission from lifted IR
//===----------------------------------------------------------------------===//

static bool TestEmitAssembly() {
  mlir::MLIRContext ctx;
  hotswap::Lifter lifter(ctx);

  // v_add_f32_e32 v0, v1, v2; s_mov_b32 s0, s1; s_nop 0; s_endpgm
  uint8_t bytes[] = {
      0x01, 0x05, 0x00, 0x02, // v_add_f32_e32 v0, v1, v2
      0x01, 0x00, 0x80, 0xBE, // s_mov_b32 s0, s1
      0x00, 0x00, 0x80, 0xBF, // s_nop 0
      0x00, 0x00, 0x81, 0xBF, // s_endpgm
  };

  auto module = lifter.lift(bytes, "gfx942", "test_emit");
  if (!module) {
    std::cerr << "FAIL: Lifter returned null\n";
    return false;
  }

  std::string asmText = hotswap::emitAssembly(*module);
  std::cout << "--- Emitted assembly ---\n" << asmText << "--- End ---\n";

  // Verify key instructions appear in the output
  bool hasVadd = asmText.find("v_add_f32_e32") != std::string::npos;
  bool hasSmov = asmText.find("s_mov_b32") != std::string::npos;
  bool hasSnop = asmText.find("s_nop") != std::string::npos;
  bool hasEndpgm = asmText.find("s_endpgm") != std::string::npos;

  if (!hasVadd || !hasSmov || !hasSnop || !hasEndpgm) {
    std::cerr << "FAIL: Missing expected instructions in assembly\n";
    std::cerr << "  v_add_f32_e32: " << hasVadd << "\n";
    std::cerr << "  s_mov_b32: " << hasSmov << "\n";
    std::cerr << "  s_nop: " << hasSnop << "\n";
    std::cerr << "  s_endpgm: " << hasEndpgm << "\n";
    return false;
  }

  // Verify register names are present
  bool hasV0 = asmText.find("v0") != std::string::npos;
  bool hasS0 = asmText.find("s0") != std::string::npos;
  if (!hasV0 || !hasS0) {
    std::cerr << "FAIL: Missing register names in assembly\n";
    return false;
  }

  std::cout << "TestEmitAssembly: PASSED\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Test: Full MLIR pipeline end-to-end (lift → retarget → emit)
//===----------------------------------------------------------------------===//

static bool TestPipelineEndToEnd() {
  // Use simple GFX942 instructions that don't need remapping (identity test)
  uint8_t bytes[] = {
      0x00, 0x00, 0x80, 0xBF, // s_nop 0
      0x01, 0x05, 0x00, 0x02, // v_add_f32_e32 v0, v1, v2
      0x00, 0x00, 0x81, 0xBF, // s_endpgm
  };

  auto result = hotswap::runPipeline(bytes, sizeof(bytes), "gfx942", "gfx942",
                                     "test_e2e");

  if (!result.success) {
    std::cerr << "FAIL: Pipeline failed: " << result.errorMessage << "\n";
    return false;
  }

  std::cout << "--- Pipeline assembly output ---\n"
            << result.assemblyText << "--- End ---\n";

  if (result.assemblyText.empty()) {
    std::cerr << "FAIL: Empty assembly output\n";
    return false;
  }

  std::cout << "TestPipelineEndToEnd: PASSED (instructions="
            << result.stats.totalInstructions
            << " lifted=" << result.stats.liftedInstructions << ")\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Test: Assemble → disassemble round-trip
//===----------------------------------------------------------------------===//

static bool TestAssembleRoundtrip() {
  // Assemble simple GFX942 instructions
  std::string asmText = ".text\ns_nop 0\ns_endpgm\n";

  auto assembled = hotswap::assembleToBytes(asmText, "gfx942");

  if (assembled.empty()) {
    std::cerr << "FAIL: Assembly returned empty bytes\n";
    return false;
  }

  std::cout << "TestAssembleRoundtrip: assembled " << assembled.size()
            << " bytes\n";

  // Verify we got at least 8 bytes (2 instructions × 4 bytes)
  if (assembled.size() < 8) {
    std::cerr << "FAIL: Expected at least 8 bytes, got " << assembled.size()
              << "\n";
    return false;
  }

  // Verify the first instruction is s_nop 0 (0xBF800000)
  uint32_t first;
  std::memcpy(&first, assembled.data(), 4);
  if (first != 0xBF800000) {
    std::cerr << "FAIL: First instruction 0x" << std::hex << first
              << " != 0xBF800000 (s_nop 0)\n" << std::dec;
    return false;
  }

  // Verify the second instruction is s_endpgm (0xBF810000)
  uint32_t second;
  std::memcpy(&second, assembled.data() + 4, 4);
  if (second != 0xBF810000) {
    std::cerr << "FAIL: Second instruction 0x" << std::hex << second
              << " != 0xBF810000 (s_endpgm)\n" << std::dec;
    return false;
  }

  std::cout << "TestAssembleRoundtrip: PASSED\n";
  return true;
}

//===----------------------------------------------------------------------===//

int main() {
  bool allPassed = true;
  allPassed &= TestLifterBasicStructure();
  allPassed &= TestLifterEndpgm();
  allPassed &= TestLifterMultipleInstructions();
  allPassed &= TestLifterInvalidBytes();
  allPassed &= TestLifterPrintsIR();
  allPassed &= TestLifterMultipleISAs();
  allPassed &= TestLifterRealInstructions();
  allPassed &= TestSSAConstruction();
  allPassed &= TestCrossTargetMapping();
  allPassed &= TestMnemonicMapping();
  allPassed &= TestBranchForward();
  allPassed &= TestBranchBackward();
  allPassed &= TestWaveWidthSaveexec();
  allPassed &= TestWaveWidthVccClear();
  allPassed &= TestFullPipeline();
  allPassed &= TestEmitAssembly();
  allPassed &= TestPipelineEndToEnd();
  allPassed &= TestAssembleRoundtrip();

  std::cout << "\n";
  if (allPassed) {
    std::cout << "All lifter tests passed.\n";
    return 0;
  }
  std::cerr << "Some lifter tests FAILED.\n";
  return 1;
}
