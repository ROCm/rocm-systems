////////////////////////////////////////////////////////////////////////////////
//
// Assembly Text Emission from waveasm MLIR IR
//
////////////////////////////////////////////////////////////////////////////////

#include "emit_assembly.hpp"

#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>

using namespace mlir;

//===----------------------------------------------------------------------===//
// Register name formatting
//===----------------------------------------------------------------------===//

namespace {

/// Format a physical VGPR name from index and size.
std::string formatVGPR(int64_t index, int64_t size) {
  if (size <= 1)
    return "v" + std::to_string(index);
  return "v[" + std::to_string(index) + ":" +
         std::to_string(index + size - 1) + "]";
}

/// Format a physical SGPR name, handling special registers.
std::string formatSGPR(int64_t index, int64_t size) {
  if (index == 106 && size <= 1)
    return "vcc_lo";
  if (index == 106 && size == 2)
    return "vcc";
  if (index == 107 && size <= 1)
    return "vcc_hi";
  if (index == 124 && size <= 1)
    return "m0";
  if (index == 126 && size <= 1)
    return "exec_lo";
  if (index == 126 && size == 2)
    return "exec";
  if (index == 127 && size <= 1)
    return "exec_hi";
  if (size <= 1)
    return "s" + std::to_string(index);
  return "s[" + std::to_string(index) + ":" +
         std::to_string(index + size - 1) + "]";
}

/// Format an AGPR name.
std::string formatAGPR(int64_t index, int64_t size) {
  if (size <= 1)
    return "a" + std::to_string(index);
  return "a[" + std::to_string(index) + ":" +
         std::to_string(index + size - 1) + "]";
}

/// Format an MLIR Value as an assembly operand string.
/// Returns the register/immediate name based on the type.
std::string formatValue(Value val) {
  Type ty = val.getType();

  if (auto pv = dyn_cast<waveasm::PVRegType>(ty))
    return formatVGPR(pv.getIndex(), pv.getSize());
  if (auto ps = dyn_cast<waveasm::PSRegType>(ty))
    return formatSGPR(ps.getIndex(), ps.getSize());
  if (auto pa = dyn_cast<waveasm::PARegType>(ty))
    return formatAGPR(pa.getIndex(), pa.getSize());
  if (auto imm = dyn_cast<waveasm::ImmType>(ty))
    return std::to_string(imm.getValue());
  if (isa<waveasm::SCCType>(ty))
    return "scc";

  // Virtual register types (after SSA construction) — not directly emittable
  if (isa<waveasm::VRegType>(ty))
    return "/*vreg*/";
  if (isa<waveasm::SRegType>(ty))
    return "/*sreg*/";
  if (isa<waveasm::ARegType>(ty))
    return "/*areg*/";

  return "/*unknown*/";
}

/// Format a result register name (destination).
std::string formatResult(OpResult result) {
  return formatValue(result);
}

//===----------------------------------------------------------------------===//
// Per-operation emission
//===----------------------------------------------------------------------===//

/// Emit a single waveasm operation as an assembly line.
/// Returns empty string for ops that don't produce assembly (e.g., precolored).
std::string emitOp(Operation *op) {
  llvm::StringRef fullName = op->getName().getStringRef();

  // Skip ops that don't produce assembly.
  if (fullName == "waveasm.precolored.vreg" ||
      fullName == "waveasm.precolored.sreg" ||
      fullName == "waveasm.precolored.areg" ||
      fullName == "waveasm.constant" ||
      fullName == "waveasm.yield" ||
      fullName == "waveasm.program")
    return "";

  // Label ops
  if (fullName == "waveasm.label") {
    if (auto attr = op->getAttrOfType<StringAttr>("sym_name"))
      return attr.getValue().str() + ":";
    return "";
  }

  // Raw ops (pre-formatted assembly text)
  if (fullName == "waveasm.raw") {
    if (auto attr = op->getAttrOfType<StringAttr>("text"))
      return attr.getValue().str();
    return "";
  }

  // Extract mnemonic from op name (drop "waveasm." prefix)
  if (!fullName.starts_with("waveasm."))
    return "";
  llvm::StringRef mnemonic = fullName.drop_front(8);

  // SOPP: no register operands, use attributes
  if (mnemonic == "s_endpgm" || mnemonic == "s_barrier")
    return mnemonic.str();

  if (mnemonic == "s_nop") {
    int64_t count = 0;
    if (auto attr = op->getAttrOfType<IntegerAttr>("simm16"))
      count = attr.getInt();
    return "s_nop " + std::to_string(count);
  }

  if (mnemonic == "s_waitcnt") {
    if (auto attr = op->getAttrOfType<IntegerAttr>("count"))
      return "s_waitcnt " + std::to_string(attr.getInt());
    return "s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)";
  }

  // Branch ops: use target attribute
  if (mnemonic.starts_with("s_branch") || mnemonic.starts_with("s_cbranch_")) {
    if (auto target = op->getAttrOfType<FlatSymbolRefAttr>("target"))
      return mnemonic.str() + " " + target.getValue().str();
    return mnemonic.str();
  }

  // ── SMEM instructions: s_load_*, s_store_*, s_buffer_load_* ──
  // Format: mnemonic dst, base, offset
  // The lifter captures trailing flags operands (GLC/DLC) which must be omitted.
  if (mnemonic.starts_with("s_load_") || mnemonic.starts_with("s_store_") ||
      mnemonic.starts_with("s_buffer_load_")) {
    llvm::SmallVector<std::string> dstStrs;
    for (auto result : op->getResults()) {
      if (isa<waveasm::SCCType>(result.getType()))
        continue;
      dstStrs.push_back(formatResult(result));
    }
    llvm::SmallVector<std::string> srcStrs;
    unsigned srcCount = 0;
    for (auto operand : op->getOperands()) {
      if (srcCount >= 2)
        break;
      std::string val = formatValue(operand);
      srcStrs.push_back(val);
      ++srcCount;
    }
    std::string line = mnemonic.str();
    bool first = true;
    for (auto &s : dstStrs) {
      line += (first ? " " : ", ");
      line += s;
      first = false;
    }
    for (auto &s : srcStrs) {
      line += (first ? " " : ", ");
      line += s;
      first = false;
    }
    return line;
  }

  // ── Global memory instructions: global_load_*, global_store_* ──
  // Loads:  global_load_*  dst, addr, off
  // Stores: global_store_* addr, data, off
  // The "off" keyword replaces the offset when it is 0 (absent soffset).
  // Trailing flags operands must be omitted.
  if (mnemonic.starts_with("global_load_") ||
      mnemonic.starts_with("global_store_")) {
    bool isStore = mnemonic.starts_with("global_store_");
    llvm::SmallVector<std::string> dstStrs;
    for (auto result : op->getResults()) {
      if (isa<waveasm::SCCType>(result.getType()))
        continue;
      dstStrs.push_back(formatResult(result));
    }
    // For loads: sources are [addr_pair, soffset, flags...]
    // For stores: sources are [addr_pair, data, soffset, flags...]
    llvm::SmallVector<std::string> srcStrs;
    unsigned maxSrcs = isStore ? 3 : 2;
    unsigned srcCount = 0;
    unsigned srcIdx = 0;
    for (auto operand : op->getOperands()) {
      if (srcCount >= maxSrcs)
        break;
      std::string val = formatValue(operand);
      // The soffset operand (last relevant source): replace 0 with "off"
      bool isSoffset = (isStore && srcIdx == 2) || (!isStore && srcIdx == 1);
      if (isSoffset && val == "0")
        val = "off";
      srcStrs.push_back(val);
      ++srcCount;
      ++srcIdx;
    }
    std::string line = mnemonic.str();
    bool first = true;
    for (auto &s : dstStrs) {
      line += (first ? " " : ", ");
      line += s;
      first = false;
    }
    for (auto &s : srcStrs) {
      line += (first ? " " : ", ");
      line += s;
      first = false;
    }
    return line;
  }

  // ── VOPC instructions: v_cmp_*_e32 ──
  // These have VCC as an implicit destination not in the MLIR results.
  // Assembly format requires explicit "vcc, " before the source operands.
  if (mnemonic.starts_with("v_cmp_") && mnemonic.ends_with("_e32")) {
    std::string line = mnemonic.str() + " vcc";
    for (auto operand : op->getOperands()) {
      line += ", ";
      line += formatValue(operand);
    }
    return line;
  }

  // ── General register-based instruction ──
  // Format: mnemonic dst, src0, src1, ...
  std::string line = mnemonic.str();

  llvm::SmallVector<std::string> dstStrs;
  for (auto result : op->getResults()) {
    Type ty = result.getType();
    if (isa<waveasm::SCCType>(ty))
      continue;
    dstStrs.push_back(formatResult(result));
  }

  llvm::SmallVector<std::string> srcStrs;
  for (auto operand : op->getOperands())
    srcStrs.push_back(formatValue(operand));

  llvm::SmallVector<std::string> allOperands;
  allOperands.append(dstStrs.begin(), dstStrs.end());
  allOperands.append(srcStrs.begin(), srcStrs.end());

  if (!allOperands.empty()) {
    line += " ";
    for (size_t i = 0; i < allOperands.size(); ++i) {
      if (i > 0)
        line += ", ";
      line += allOperands[i];
    }
  }

  return line;
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

std::string hotswap::emitAssembly(ModuleOp module) {
  std::string result;
  llvm::raw_string_ostream os(result);

  module->walk([&](waveasm::ProgramOp program) {
    Block &body = program.getBody().front();
    for (Operation &op : body) {
      std::string line = emitOp(&op);
      if (!line.empty())
        os << line << "\n";
    }
  });

  os.flush();
  return result;
}
