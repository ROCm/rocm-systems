// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Kernel Generator - AMDGCN Assembly Code Generation
//
// Core code generation from WaveASM IR to AMDGCN assembly text.
// Handles value resolution, TypeSwitch-based op dispatch, and the main
// generate() loop. InstructionFormatter, MetadataEmitter, and literal
// materialization logic live in separate files.
//===----------------------------------------------------------------------===//

#include "waveasm/Transforms/AssemblyEmitter.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"
#include "waveasm/Target/AMDGCN/ABI.h"
#include "waveasm/Target/AMDGCN/RegisterInfo.h"
#include "waveasm/Transforms/RegAlloc.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <sstream>

using namespace mlir;

namespace waveasm {
using namespace abi;

//===----------------------------------------------------------------------===//
// Kernel Generator Implementation (Pure SSA)
//===----------------------------------------------------------------------===//

KernelGenerator::KernelGenerator(ProgramOp program,
                                 const PhysicalMapping &mapping,
                                 TargetAttrInterface target)
    : program(program), mapping(mapping), target(target) {}

std::string KernelGenerator::resolveValue(Value value) {
  Type ty = value.getType();

  if (auto pvreg = dyn_cast<PVRegType>(ty)) {
    return formatVGPRRange(pvreg.getIndex(), pvreg.getSize());
  }
  if (auto psreg = dyn_cast<PSRegType>(ty)) {
    return formatSGPRRange(psreg.getIndex(), psreg.getSize());
  }
  if (auto pareg = dyn_cast<PARegType>(ty)) {
    return formatAGPRRange(pareg.getIndex(), pareg.getSize());
  }

  if (isVirtualRegType(ty)) {
    int64_t physIdx = mapping.getPhysReg(value);
    if (physIdx >= 0) {
      int64_t size = getRegSize(ty);
      if (isVGPRType(ty)) {
        return formatVGPRRange(physIdx, size);
      } else if (isAGPRType(ty)) {
        return formatAGPRRange(physIdx, size);
      } else {
        return formatSGPRRange(physIdx, size);
      }
    }
    return "%<ssa>";
  }

  if (isa<ImmType>(ty)) {
    if (auto defOp = value.getDefiningOp()) {
      if (auto constOp = dyn_cast<ConstantOp>(defOp)) {
        int64_t val = static_cast<int64_t>(constOp.getValue());
        return std::to_string(val);
      }
    }
    return "0";
  }

  if (isa<SCCType>(ty))
    return "scc";

  return "<unknown>";
}

std::pair<bool, int64_t> KernelGenerator::getLiteralValue(Value value) {
  Type ty = value.getType();
  if (isa<ImmType>(ty)) {
    if (auto defOp = value.getDefiningOp()) {
      if (auto constOp = dyn_cast<ConstantOp>(defOp)) {
        int64_t val = static_cast<int64_t>(constOp.getValue());
        return {true, val};
      }
    }
  }
  return {false, 0};
}

KernelGenerator::MaterializedOperand
KernelGenerator::materializeLiteralOperand(Value operand, int scratchIdx) {
  auto [isLit, val] = getLiteralValue(operand);
  if (isLit && !isInlineConstant(val)) {
    std::string scratch = formatVGPRRange(scratchIdx, 1);
    peakVGPRs = std::max(peakVGPRs, static_cast<int64_t>(scratchIdx + 1));
    // Reuse cached value if kScratchVGPR already holds this literal.
    if (scratchIdx == kScratchVGPR && scratchVGPRValue == val)
      return {scratch, ""};
    std::string prefix =
        "  v_mov_b32 " + scratch + ", " + std::to_string(val) + "\n";
    if (scratchIdx == kScratchVGPR)
      scratchVGPRValue = val;
    return {scratch, prefix};
  }
  return {resolveValue(operand), ""};
}

//===----------------------------------------------------------------------===//
// TypeSwitch-based Operation Code Generation
//===----------------------------------------------------------------------===//

std::string KernelGenerator::emitBufferLoadLDS(Operation *op,
                                               llvm::StringRef mnemonic) {
  std::string result = "  " + mnemonic.str();
  // Use VMEMToLDSLoadOpInterface to access operands by name
  if (auto loadOp = dyn_cast<VMEMToLDSLoadOpInterface>(op)) {
    std::string voffset = resolveValue(loadOp.getVoffset());
    std::string srd = resolveValue(loadOp.getSrd());
    std::string soffset = resolveValue(loadOp.getSoffset());
    result += " " + voffset + ", " + srd + ", " + soffset + " offen";
    if (auto instOffsetAttr = op->getAttrOfType<IntegerAttr>("instOffset")) {
      int64_t offset = instOffsetAttr.getInt();
      if (offset > 0) {
        result += " offset:" + std::to_string(offset);
      }
    }
    result += " lds";
  }
  return result;
}

std::string KernelGenerator::emitBufferLoad(Operation *op,
                                            llvm::StringRef mnemonic) {
  std::string result = "  " + mnemonic.str();
  std::string vdata;
  for (Value res : op->getResults()) {
    vdata = resolveValue(res);
  }
  // Use VMEMLoadOpInterface to access operands by name
  if (auto loadOp = dyn_cast<VMEMLoadOpInterface>(op)) {
    std::string voffset = resolveValue(loadOp.getVoffset());
    std::string srd = resolveValue(loadOp.getSaddr());
    std::string soffset = resolveValue(loadOp.getSoffset());
    result +=
        " " + vdata + ", " + voffset + ", " + srd + ", " + soffset + " offen";
    if (auto instOffsetAttr = op->getAttrOfType<IntegerAttr>("instOffset")) {
      int64_t offset = instOffsetAttr.getInt();
      if (offset > 0) {
        result += " offset:" + std::to_string(offset);
      }
    }
  }
  return result;
}

std::string KernelGenerator::emitBufferStore(Operation *op,
                                             llvm::StringRef mnemonic) {
  std::string result = "  " + mnemonic.str();
  // Use VMEMStoreOpInterface to access operands by name
  if (auto storeOp = dyn_cast<VMEMStoreOpInterface>(op)) {
    std::string vdata = resolveValue(storeOp.getData());
    std::string voffset = resolveValue(storeOp.getVoffset());
    std::string srd = resolveValue(storeOp.getSaddr());
    result += " " + vdata + ", " + voffset + ", " + srd + ", 0 offen";
    if (auto instOffsetAttr = op->getAttrOfType<IntegerAttr>("instOffset")) {
      int64_t offset = instOffsetAttr.getInt();
      if (offset > 0) {
        result += " offset:" + std::to_string(offset);
      }
    }
  }
  return result;
}

std::string KernelGenerator::emitGlobalLoad(Operation *op,
                                            llvm::StringRef mnemonic) {
  std::string result = "  " + mnemonic.str();
  std::string vdata;
  for (Value res : op->getResults()) {
    vdata = resolveValue(res);
  }
  // Use VMEMLoadOpInterface to access operands by name
  if (auto loadOp = dyn_cast<VMEMLoadOpInterface>(op)) {
    std::string vaddr = resolveValue(loadOp.getVoffset());
    std::string saddr = resolveValue(loadOp.getSaddr());
    result += " " + vdata + ", " + vaddr + ", " + saddr;
  }
  return result;
}

std::string KernelGenerator::emitGlobalStore(Operation *op,
                                             llvm::StringRef mnemonic) {
  std::string result = "  " + mnemonic.str();
  // Use VMEMStoreOpInterface to access operands by name
  if (auto storeOp = dyn_cast<VMEMStoreOpInterface>(op)) {
    std::string vdata = resolveValue(storeOp.getData());
    std::string vaddr = resolveValue(storeOp.getVoffset());
    std::string saddr = resolveValue(storeOp.getSaddr());
    result += " " + vaddr + ", " + vdata + ", " + saddr;
  }
  return result;
}

std::string KernelGenerator::emitLDSRead(Operation *op,
                                         llvm::StringRef mnemonic) {
  std::string result = "  " + mnemonic.str();
  std::string vdst;
  for (Value res : op->getResults()) {
    vdst = resolveValue(res);
  }
  // Use LDSLoadOpInterface to access operands by name
  if (auto loadOp = dyn_cast<LDSLoadOpInterface>(op)) {
    std::string vaddr = resolveValue(loadOp.getVaddr());
    result += " " + vdst + ", " + vaddr;
  }
  if (auto offsetAttr = op->getAttrOfType<IntegerAttr>("offset")) {
    int64_t offset = offsetAttr.getInt();
    if (offset != 0) {
      result += " offset:" + std::to_string(offset);
    }
  }
  return result;
}

std::string KernelGenerator::emitLDSWrite(Operation *op,
                                          llvm::StringRef mnemonic) {
  std::string result = "  " + mnemonic.str();
  // Use LDSStoreOpInterface to access operands by name
  if (auto storeOp = dyn_cast<LDSStoreOpInterface>(op)) {
    std::string vaddr = resolveValue(storeOp.getVaddr());
    std::string vdata = resolveValue(storeOp.getData());
    result += " " + vaddr + ", " + vdata;
  }
  if (auto offsetAttr = op->getAttrOfType<IntegerAttr>("offset")) {
    int64_t offset = offsetAttr.getInt();
    if (offset != 0) {
      result += " offset:" + std::to_string(offset);
    }
  }
  return result;
}

std::string KernelGenerator::resolveScalarValue(Value value) {
  Type ty = value.getType();
  if (auto pvreg = dyn_cast<PVRegType>(ty)) {
    if (pvreg.getSize() > 1) {
      return formatVGPRRange(pvreg.getIndex(), 1);
    }
  }
  if (auto psreg = dyn_cast<PSRegType>(ty)) {
    if (psreg.getSize() > 1) {
      return formatSGPRRange(psreg.getIndex(), 1);
    }
  }
  // For multi-register physical AGPRs, use only the first register
  if (auto pareg = dyn_cast<PARegType>(ty)) {
    if (pareg.getSize() > 1) {
      return formatAGPRRange(pareg.getIndex(), 1);
    }
  }
  return resolveValue(value);
}

std::string KernelGenerator::emitDefaultFormat(Operation *op,
                                               llvm::StringRef mnemonic) {
  llvm::SmallVector<std::string> operands;

  bool isScalarOp = false;
  if (op->getNumResults() == 1) {
    Type resTy = op->getResult(0).getType();
    if (auto pvreg = dyn_cast<PVRegType>(resTy)) {
      isScalarOp = (pvreg.getSize() == 1);
    } else if (auto psreg = dyn_cast<PSRegType>(resTy)) {
      isScalarOp = (psreg.getSize() == 1);
    } else if (auto vreg = dyn_cast<VRegType>(resTy)) {
      isScalarOp = (vreg.getSize() == 1);
    } else if (auto sreg = dyn_cast<SRegType>(resTy)) {
      isScalarOp = (sreg.getSize() == 1);
    }
  }

  for (Value result : op->getResults()) {
    if (isa<SCCType>(result.getType()))
      continue;
    operands.push_back(resolveValue(result));
  }
  for (Value operand : op->getOperands()) {
    if (isa<SCCType>(operand.getType()))
      continue;
    if (isScalarOp) {
      operands.push_back(resolveScalarValue(operand));
    } else {
      operands.push_back(resolveValue(operand));
    }
  }
  return formatter.format(mnemonic, operands);
}

std::optional<std::string>
KernelGenerator::emitScaledMFMA(Operation *scaledOp, llvm::StringRef mnemonic) {
  std::string line = emitDefaultFormat(scaledOp, mnemonic);

  // Emit op_sel/op_sel_hi from scalesIdxA/scalesIdxB attributes.
  // op_sel[0] = scalesIdxA bit 0, op_sel[1] = scalesIdxB bit 0
  // op_sel_hi[0] = scalesIdxA bit 1, op_sel_hi[1] = scalesIdxB bit 1
  int32_t opSelA = 0, opSelB = 0;
  if (auto a = scaledOp->getAttrOfType<IntegerAttr>("op_sel_a"))
    opSelA = a.getInt();
  if (auto b = scaledOp->getAttrOfType<IntegerAttr>("op_sel_b"))
    opSelB = b.getInt();

  int32_t selLo0 = opSelA & 1;
  int32_t selLo1 = opSelB & 1;
  int32_t selHi0 = (opSelA >> 1) & 1;
  int32_t selHi1 = (opSelB >> 1) & 1;

  // Third element is always 0 (accumulator operand op_sel, which is unused
  // for scaled MFMA instructions on CDNA3/CDNA4).
  if (selLo0 || selLo1) {
    line += " op_sel:[" + std::to_string(selLo0) + "," +
            std::to_string(selLo1) + ",0]";
  }
  // Always emit op_sel_hi for scaled MFMA: the hardware default for
  // VOP3P-class instructions is [1,1,1], not [0,0,0]. Omitting it when
  // all zeros causes the wrong scale bytes to be selected.
  line += " op_sel_hi:[" + std::to_string(selHi0) + "," +
          std::to_string(selHi1) + ",0]";

  int32_t cbsz = 4;
  int32_t blgp = 4;
  if (auto cbszAttr = scaledOp->getAttrOfType<IntegerAttr>("cbsz"))
    cbsz = cbszAttr.getInt();
  if (auto blgpAttr = scaledOp->getAttrOfType<IntegerAttr>("blgp"))
    blgp = blgpAttr.getInt();
  line += " cbsz:" + std::to_string(cbsz) + " blgp:" + std::to_string(blgp);
  return line;
}

std::optional<std::string> KernelGenerator::generateOp(Operation *op) {
  return llvm::TypeSwitch<Operation *, std::optional<std::string>>(op)
      .Case<ProgramOp, LabelOp, CommentOp, RawOp, PrecoloredVRegOp,
            PrecoloredSRegOp, PrecoloredARegOp, ConstantOp, PackOp, ExtractOp,
            DCEProtectOp>([](auto) { return std::nullopt; })

      .Case<S_WAITCNT>([&](S_WAITCNT waitcntOp) {
        std::optional<int64_t> vmcnt, lgkmcnt, expcnt;
        if (auto vmAttr = waitcntOp.getVmcntAttr())
          vmcnt = vmAttr.getInt();
        if (auto lgkmAttr = waitcntOp.getLgkmcntAttr())
          lgkmcnt = lgkmAttr.getInt();
        if (auto expAttr = waitcntOp.getExpcntAttr())
          expcnt = expAttr.getInt();
        if (!vmcnt && !lgkmcnt && !expcnt) {
          vmcnt = 0;
          lgkmcnt = 0;
        }
        return formatter.formatWaitcnt(vmcnt, lgkmcnt, expcnt);
      })
      .Case<S_WAITCNT_VMCNT>([&](S_WAITCNT_VMCNT waitcntOp) {
        return formatter.formatWaitcnt(waitcntOp.getCount(), std::nullopt,
                                       std::nullopt);
      })
      .Case<S_WAITCNT_LGKMCNT>([&](S_WAITCNT_LGKMCNT waitcntOp) {
        return formatter.formatWaitcnt(std::nullopt, waitcntOp.getCount(),
                                       std::nullopt);
      })

      .Case<BUFFER_LOAD_DWORD>([&](auto loadOp) {
        return emitBufferLoad(loadOp, "buffer_load_dword");
      })
      .Case<BUFFER_LOAD_DWORDX2>([&](auto loadOp) {
        return emitBufferLoad(loadOp, "buffer_load_dwordx2");
      })
      .Case<BUFFER_LOAD_DWORDX3>([&](auto loadOp) {
        return emitBufferLoad(loadOp, "buffer_load_dwordx3");
      })
      .Case<BUFFER_LOAD_DWORDX4>([&](auto loadOp) {
        return emitBufferLoad(loadOp, "buffer_load_dwordx4");
      })

      .Case<BUFFER_LOAD_UBYTE>([&](auto loadOp) {
        return emitBufferLoad(loadOp, "buffer_load_ubyte");
      })
      .Case<BUFFER_LOAD_USHORT>([&](auto loadOp) {
        return emitBufferLoad(loadOp, "buffer_load_ushort");
      })

      // Buffer load to LDS (gather-to-LDS) -- shared helper avoids duplication
      .Case<BUFFER_LOAD_DWORD_LDS>([&](auto loadOp) {
        return emitBufferLoadLDS(loadOp, "buffer_load_dword");
      })
      .Case<BUFFER_LOAD_DWORDX4_LDS>([&](auto loadOp) {
        return emitBufferLoadLDS(loadOp, "buffer_load_dwordx4");
      })

      // Buffer atomic operations
      .Case<BUFFER_ATOMIC_PK_ADD_BF16>([&](auto atomicOp) {
        return emitBufferStore(atomicOp, "buffer_atomic_pk_add_bf16");
      })
      .Case<BUFFER_ATOMIC_ADD_F32>([&](auto atomicOp) {
        return emitBufferStore(atomicOp, "buffer_atomic_add_f32");
      })

      .Case<BUFFER_STORE_DWORD>([&](auto storeOp) {
        return emitBufferStore(storeOp, "buffer_store_dword");
      })
      .Case<BUFFER_STORE_DWORDX2>([&](auto storeOp) {
        return emitBufferStore(storeOp, "buffer_store_dwordx2");
      })
      .Case<BUFFER_STORE_DWORDX3>([&](auto storeOp) {
        return emitBufferStore(storeOp, "buffer_store_dwordx3");
      })
      .Case<BUFFER_STORE_DWORDX4>([&](auto storeOp) {
        return emitBufferStore(storeOp, "buffer_store_dwordx4");
      })
      .Case<BUFFER_STORE_SHORT>([&](auto storeOp) {
        return emitBufferStore(storeOp, "buffer_store_short");
      })
      .Case<BUFFER_STORE_BYTE>([&](auto storeOp) {
        return emitBufferStore(storeOp, "buffer_store_byte");
      })

      .Case<GLOBAL_LOAD_DWORD>([&](auto loadOp) {
        return emitGlobalLoad(loadOp, "global_load_dword");
      })
      .Case<GLOBAL_LOAD_DWORDX2>([&](auto loadOp) {
        return emitGlobalLoad(loadOp, "global_load_dwordx2");
      })
      .Case<GLOBAL_LOAD_DWORDX3>([&](auto loadOp) {
        return emitGlobalLoad(loadOp, "global_load_dwordx3");
      })
      .Case<GLOBAL_LOAD_DWORDX4>([&](auto loadOp) {
        return emitGlobalLoad(loadOp, "global_load_dwordx4");
      })

      .Case<GLOBAL_STORE_DWORD>([&](auto storeOp) {
        return emitGlobalStore(storeOp, "global_store_dword");
      })
      .Case<GLOBAL_STORE_DWORDX2>([&](auto storeOp) {
        return emitGlobalStore(storeOp, "global_store_dwordx2");
      })
      .Case<GLOBAL_STORE_DWORDX3>([&](auto storeOp) {
        return emitGlobalStore(storeOp, "global_store_dwordx3");
      })
      .Case<GLOBAL_STORE_DWORDX4>([&](auto storeOp) {
        return emitGlobalStore(storeOp, "global_store_dwordx4");
      })

      .Case<DS_READ_B32>(
          [&](auto readOp) { return emitLDSRead(readOp, "ds_read_b32"); })
      .Case<DS_READ_B64>(
          [&](auto readOp) { return emitLDSRead(readOp, "ds_read_b64"); })
      .Case<DS_READ_B128>(
          [&](auto readOp) { return emitLDSRead(readOp, "ds_read_b128"); })
      .Case<DS_READ_U8>(
          [&](auto readOp) { return emitLDSRead(readOp, "ds_read_u8"); })
      .Case<DS_READ_U16>(
          [&](auto readOp) { return emitLDSRead(readOp, "ds_read_u16"); })

      .Case<DS_WRITE_B32>(
          [&](auto writeOp) { return emitLDSWrite(writeOp, "ds_write_b32"); })
      .Case<DS_WRITE_B64>(
          [&](auto writeOp) { return emitLDSWrite(writeOp, "ds_write_b64"); })
      .Case<DS_WRITE_B128>(
          [&](auto writeOp) { return emitLDSWrite(writeOp, "ds_write_b128"); })
      .Case<DS_WRITE_B8>(
          [&](auto writeOp) { return emitLDSWrite(writeOp, "ds_write_b8"); })
      .Case<DS_WRITE_B16>(
          [&](auto writeOp) { return emitLDSWrite(writeOp, "ds_write_b16"); })

      .Case<S_MOV_B32_M0>([&](S_MOV_B32_M0 movOp) {
        std::string result = "  s_mov_b32 m0";
        result += ", " + resolveValue(movOp.getSrc());
        return result;
      })

      .Case<S_AND_SAVEEXEC_B64>(
          [&](S_AND_SAVEEXEC_B64 saveOp) -> std::optional<std::string> {
            std::string dst = resolveValue(saveOp.getDst());
            return "  s_and_saveexec_b64 " + dst + ", vcc";
          })

      .Case<S_MOV_B64_EXEC>(
          [&](S_MOV_B64_EXEC restoreOp) -> std::optional<std::string> {
            std::string src = resolveValue(restoreOp.getSrc());
            return "  s_mov_b64 exec, " + src;
          })

      .Case<S_BRANCH>([&](S_BRANCH branchOp) {
        return std::string("  s_branch ") +
               branchOp.getTarget().getRootReference().str();
      })
      .Case<S_CBRANCH_SCC0>([&](S_CBRANCH_SCC0 branchOp) {
        return std::string("  s_cbranch_scc0 ") +
               branchOp.getTarget().getRootReference().str();
      })
      .Case<S_CBRANCH_SCC1>([&](S_CBRANCH_SCC1 branchOp) {
        return std::string("  s_cbranch_scc1 ") +
               branchOp.getTarget().getRootReference().str();
      })

      .Case<S_NOP>([&](S_NOP nopOp) {
        return std::string("  s_nop ") + std::to_string(nopOp.getCount());
      })

      .Case<S_SETPRIO>([&](S_SETPRIO prioOp) {
        return std::string("  s_setprio ") + std::to_string(prioOp.getCount());
      })

      .Case<S_BARRIER>([](auto) { return std::string("  s_barrier"); })
      .Case<S_ENDPGM>([](auto) { return std::string("  s_endpgm"); })

      .Case<V_MOV_B32>([&](V_MOV_B32 movOp) -> std::optional<std::string> {
        Value result = movOp.getDst();
        Value srcVal = movOp.getSrc();
        int64_t size = getRegSize(result.getType());
        bool isAGPR = isAGPRType(result.getType());
        bool srcIsImm = isa<ImmType>(srcVal.getType());

        if (size > 1) {
          int64_t baseIdx = mapping.getPhysReg(result);
          if (baseIdx < 0) {
            if (auto pvreg = dyn_cast<PVRegType>(result.getType())) {
              baseIdx = pvreg.getIndex();
            } else if (auto pareg = dyn_cast<PARegType>(result.getType())) {
              baseIdx = pareg.getIndex();
            }
          }
          if (baseIdx >= 0) {
            std::string src = resolveValue(srcVal);
            std::string lines;
            if (isAGPR) {
              // v_accvgpr_write_b32 requires a VGPR source in this backend.
              // Materialize immediate sources into the reserved scratch VGPR.
              std::string writeSrc = src;
              if (srcIsImm) {
                lines += "  v_mov_b32 " + formatVGPRRange(kScratchVGPR, 1) +
                         ", " + src;
                writeSrc = formatVGPRRange(kScratchVGPR, 1);
                peakVGPRs = std::max(peakVGPRs, kScratchVGPR + 1);
                invalidateScratchCache();
              }
              for (int64_t i = 0; i < size; ++i) {
                if (!lines.empty())
                  lines += "\n";
                lines += "  v_accvgpr_write_b32 a" +
                         std::to_string(baseIdx + i) + ", " + writeSrc;
              }
            } else {
              for (int64_t i = 0; i < size; ++i) {
                if (i > 0)
                  lines += "\n";
                lines +=
                    "  v_mov_b32 v" + std::to_string(baseIdx + i) + ", " + src;
              }
            }
            return lines;
          }
        }
        if (isAGPR) {
          if (srcIsImm) {
            std::string src = resolveValue(srcVal);
            std::string scratch = formatVGPRRange(kScratchVGPR, 1);
            peakVGPRs = std::max(peakVGPRs, kScratchVGPR + 1);
            invalidateScratchCache();
            return "  v_mov_b32 " + scratch + ", " + src +
                   "\n  v_accvgpr_write_b32 " + resolveValue(result) + ", " +
                   scratch;
          }
          return emitDefaultFormat(movOp, "v_accvgpr_write_b32");
        }
        return emitDefaultFormat(movOp, "v_mov_b32");
      })

      .Case<V_MFMA_SCALE_F32_16X16X128_F8F6F4>(
          [&](auto scaledOp) -> std::optional<std::string> {
            return emitScaledMFMA(scaledOp,
                                  "v_mfma_scale_f32_16x16x128_f8f6f4");
          })
      .Case<V_MFMA_SCALE_F32_32X32X64_F8F6F4>(
          [&](auto scaledOp) -> std::optional<std::string> {
            return emitScaledMFMA(scaledOp, "v_mfma_scale_f32_32x32x64_f8f6f4");
          })

      .Case<LoopOp>([&](LoopOp loopOp) -> std::optional<std::string> {
        std::string labelName = "L_loop_" + std::to_string(loopLabelCounter++);

        std::string buf;
        llvm::raw_string_ostream os(buf);

        // Emit copies from init arg registers to block arg registers when
        // they differ (i.e. coalescing was broken because the init arg has
        // post-loop uses).
        Block &body = loopOp.getBodyBlock();
        for (unsigned i = 0; i < body.getNumArguments(); ++i) {
          if (i >= loopOp.getInitArgs().size())
            break;
          auto [srcPhys, isSGPR] = [&]() -> std::pair<int64_t, bool> {
            Type ty = loopOp.getInitArgs()[i].getType();
            if (auto psreg = dyn_cast<PSRegType>(ty))
              return {psreg.getIndex(), true};
            if (auto pvreg = dyn_cast<PVRegType>(ty))
              return {pvreg.getIndex(), false};
            return {-1, false};
          }();
          auto [dstPhys, dstIsSGPR] = [&]() -> std::pair<int64_t, bool> {
            Type ty = body.getArgument(i).getType();
            if (auto psreg = dyn_cast<PSRegType>(ty))
              return {psreg.getIndex(), true};
            if (auto pvreg = dyn_cast<PVRegType>(ty))
              return {pvreg.getIndex(), false};
            return {-1, false};
          }();
          if (srcPhys >= 0 && dstPhys >= 0 && srcPhys != dstPhys) {
            int64_t width = getRegSize(body.getArgument(i).getType());
            for (int64_t r = 0; r < width; ++r) {
              if (isSGPR)
                os << "  s_mov_b32 s" << (dstPhys + r) << ", s" << (srcPhys + r)
                   << "\n";
              else
                os << "  v_mov_b32 v" << (dstPhys + r) << ", v" << (srcPhys + r)
                   << "\n";
            }
          }
        }

        os << labelName << ":\n";
        // Back-edge merges with loop entry; cannot assume scratch contents.
        invalidateScratchCache();
        for (Operation &bodyOp : body) {
          if (auto condOp = dyn_cast<ConditionOp>(&bodyOp)) {
            {
              unsigned numArgs = body.getNumArguments();
              unsigned numIter = condOp.getIterArgs().size();
              assert(numIter == numArgs &&
                     "ConditionOp iter_args count must match body block "
                     "argument count for correct register rotation");

              struct CopyInfo {
                int64_t dst;
                int64_t src;
                bool isSGPR;
              };
              SmallVector<CopyInfo> pendingCopies;

              auto getPhysRegInfo = [&](Value val) -> std::pair<int64_t, bool> {
                Type ty = val.getType();
                if (auto psreg = dyn_cast<PSRegType>(ty))
                  return {psreg.getIndex(), true};
                if (auto pvreg = dyn_cast<PVRegType>(ty))
                  return {pvreg.getIndex(), false};
                if (isVirtualRegType(ty))
                  return {mapping.getPhysReg(val), isSGPRType(ty)};
                return {-1, false};
              };

              // Read pre-coercion physical register indices saved by
              // LinearScanPass (before iter_arg types were coerced to match
              // block arg types for LoopLikeOpInterface).
              auto origPhysRegsAttr =
                  condOp->getAttrOfType<DenseI64ArrayAttr>("_iterArgPhysRegs");

              for (unsigned i = 0; i < numIter; ++i) {
                auto [dstPhys, dstIsSGPR] = getPhysRegInfo(body.getArgument(i));
                int64_t srcPhys;
                bool isSGPR;
                if (origPhysRegsAttr &&
                    i < static_cast<unsigned>(origPhysRegsAttr.size()) &&
                    origPhysRegsAttr[i] >= 0) {
                  srcPhys = origPhysRegsAttr[i];
                  isSGPR = dstIsSGPR;
                } else {
                  std::tie(srcPhys, isSGPR) =
                      getPhysRegInfo(condOp.getIterArgs()[i]);
                }

                if (srcPhys >= 0 && dstPhys >= 0 && srcPhys != dstPhys) {
                  if (isSGPR != dstIsSGPR)
                    continue;
                  int64_t width = getRegSize(body.getArgument(i).getType());
                  if (dstPhys > srcPhys) {
                    for (int64_t r = width - 1; r >= 0; --r)
                      pendingCopies.push_back(
                          {dstPhys + r, srcPhys + r, isSGPR});
                  } else {
                    for (int64_t r = 0; r < width; ++r)
                      pendingCopies.push_back(
                          {dstPhys + r, srcPhys + r, isSGPR});
                  }
                }
              }

              SmallVector<bool> handled(pendingCopies.size(), false);

              // Allocate swap temps once and reuse across all swaps.
              // Swaps are emitted sequentially so the temp is dead after
              // each 3-instruction sequence and can be reused.
              int64_t vSwapTemp = -1;
              int64_t sSwapTemp = -1;

              for (size_t i = 0; i < pendingCopies.size(); ++i) {
                if (handled[i])
                  continue;
                for (size_t j = i + 1; j < pendingCopies.size(); ++j) {
                  if (handled[j])
                    continue;
                  if (pendingCopies[i].dst == pendingCopies[j].src &&
                      pendingCopies[j].dst == pendingCopies[i].src &&
                      pendingCopies[i].isSGPR == pendingCopies[j].isSGPR) {
                    if (pendingCopies[i].isSGPR) {
                      int64_t regA = pendingCopies[i].dst;
                      int64_t regB = pendingCopies[j].dst;
                      if (sSwapTemp < 0) {
                        sSwapTemp = peakSGPRs;
                        peakSGPRs = std::max(peakSGPRs, sSwapTemp + 1);
                      }
                      os << "  s_mov_b32 s" << sSwapTemp << ", s" << regA
                         << "\n";
                      os << "  s_mov_b32 s" << regA << ", s" << regB << "\n";
                      os << "  s_mov_b32 s" << regB << ", s" << sSwapTemp
                         << "\n";
                      handled[i] = true;
                      handled[j] = true;
                      break;
                    }
                    int64_t regA = pendingCopies[i].dst;
                    int64_t regB = pendingCopies[j].dst;
                    if (vSwapTemp < 0) {
                      vSwapTemp = peakVGPRs;
                      peakVGPRs = std::max(peakVGPRs, vSwapTemp + 1);
                    }
                    os << "  v_mov_b32 v" << vSwapTemp << ", v" << regA << "\n";
                    os << "  v_mov_b32 v" << regA << ", v" << regB << "\n";
                    os << "  v_mov_b32 v" << regB << ", v" << vSwapTemp << "\n";
                    handled[i] = true;
                    handled[j] = true;
                    break;
                  }
                }
              }

              for (size_t i = 0; i < pendingCopies.size(); ++i) {
                if (handled[i])
                  continue;
                const auto &copy = pendingCopies[i];
                if (copy.isSGPR) {
                  os << "  s_mov_b32 s" << copy.dst << ", s" << copy.src
                     << "\n";
                } else {
                  os << "  v_mov_b32 v" << copy.dst << ", v" << copy.src
                     << "\n";
                }
              }
            }

            os << "  s_cbranch_scc1 " << labelName;
            break;
          }

          if (auto rawOp = dyn_cast<RawOp>(&bodyOp)) {
            os << generateRaw(rawOp) << "\n";
            continue;
          }
          if (auto commentOp = dyn_cast<CommentOp>(&bodyOp)) {
            os << generateComment(commentOp) << "\n";
            continue;
          }
          auto instrLines = generateOpWithLiteralHandling(&bodyOp);
          for (const auto &line : instrLines) {
            os << line << "\n";
          }
        }

        return os.str();
      })
      .Case<IfOp>([&](IfOp ifOp) -> std::optional<std::string> {
        int labelId = loopLabelCounter++;
        std::string elseLabel = "L_if_else_" + std::to_string(labelId);
        std::string endLabel = "L_if_end_" + std::to_string(labelId);

        std::string buf;
        llvm::raw_string_ostream os(buf);

        if (ifOp.hasElse()) {
          os << "  s_cbranch_scc0 " << elseLabel << "\n";
        } else {
          os << "  s_cbranch_scc0 " << endLabel << "\n";
        }

        for (Operation &thenOp : ifOp.getThenBlock()) {
          if (isa<YieldOp>(&thenOp))
            continue;
          if (auto rawOp = dyn_cast<RawOp>(&thenOp)) {
            os << generateRaw(rawOp) << "\n";
            continue;
          }
          if (auto commentOp = dyn_cast<CommentOp>(&thenOp)) {
            os << generateComment(commentOp) << "\n";
            continue;
          }
          auto instrLines = generateOpWithLiteralHandling(&thenOp);
          for (const auto &line : instrLines) {
            os << line << "\n";
          }
        }

        if (ifOp.hasElse()) {
          os << "  s_branch " << endLabel << "\n";
          os << elseLabel << ":\n";
          // Else block is entered from the branch at the top, not from
          // the then block, so scratch cache from then-block is invalid.
          invalidateScratchCache();
          for (Operation &elseOp : *ifOp.getElseBlock()) {
            if (isa<YieldOp>(&elseOp))
              continue;
            if (auto rawOp = dyn_cast<RawOp>(&elseOp)) {
              os << generateRaw(rawOp) << "\n";
              continue;
            }
            if (auto commentOp = dyn_cast<CommentOp>(&elseOp)) {
              os << generateComment(commentOp) << "\n";
              continue;
            }
            auto instrLines = generateOpWithLiteralHandling(&elseOp);
            for (const auto &line : instrLines) {
              os << line << "\n";
            }
          }
        }

        os << endLabel << ":";
        // Then/else branches may leave different values in scratch VGPR.
        invalidateScratchCache();
        return os.str();
      })
      .Case<ConditionOp>([&](ConditionOp) -> std::optional<std::string> {
        return std::nullopt;
      })
      .Case<YieldOp>(
          [&](YieldOp) -> std::optional<std::string> { return std::nullopt; })
      .Case<S_CMP_LT_U32, S_CMP_EQ_U32, S_CMP_LE_U32, S_CMP_GT_U32,
            S_CMP_GE_U32, S_CMP_NE_U32, S_CMP_LT_I32, S_CMP_EQ_I32,
            S_CMP_LE_I32, S_CMP_GT_I32, S_CMP_GE_I32, S_CMP_NE_I32>(
          [&](auto cmpOp) -> std::optional<std::string> {
            llvm::StringRef opName = cmpOp->getName().getStringRef();
            llvm::StringRef mnemonic = opName;
            if (opName.starts_with("waveasm.")) {
              mnemonic = opName.drop_front(8);
            }
            std::string mnemStr;
            if (mnemonic.contains("_ne_")) {
              mnemStr = mnemonic.str();
              size_t pos = mnemStr.find("_ne_");
              mnemStr.replace(pos, 4, "_lg_");
              mnemonic = mnemStr;
            }
            llvm::SmallVector<std::string> operands;
            for (Value operand : cmpOp->getOperands()) {
              operands.push_back(resolveValue(operand));
            }
            return formatter.format(mnemonic, operands);
          })

      // SALU arithmetic ops that set SCC: emit dst and operands, skip scc.
      .Case<S_ADD_U32, S_ADDC_U32, S_ADD_I32, S_SUB_U32, S_SUB_I32>(
          [&](auto addOp) -> std::optional<std::string> {
            llvm::StringRef opName = addOp->getName().getStringRef();
            llvm::StringRef mnemonic = opName;
            if (opName.starts_with("waveasm.")) {
              mnemonic = opName.drop_front(8);
            }
            llvm::SmallVector<std::string> operands;
            // Only emit the first result (dst), not the second (scc)
            operands.push_back(resolveValue(addOp.getDst()));
            for (Value operand : addOp->getOperands()) {
              if (isa<SCCType>(operand.getType()))
                continue;
              operands.push_back(resolveValue(operand));
            }
            return formatter.format(mnemonic, operands);
          })

      // V_CMP_* operations: write to VCC implicitly, need explicit vcc in asm.
      .Case<V_CMP_EQ_U32, V_CMP_NE_U32, V_CMP_LT_U32, V_CMP_LE_U32,
            V_CMP_GT_U32, V_CMP_GE_U32, V_CMP_EQ_I32, V_CMP_NE_I32,
            V_CMP_LT_I32, V_CMP_LE_I32, V_CMP_GT_I32, V_CMP_GE_I32,
            V_CMP_EQ_F32, V_CMP_NE_F32, V_CMP_LT_F32, V_CMP_LE_F32,
            V_CMP_GT_F32, V_CMP_GE_F32>(
          [&](auto cmpOp) -> std::optional<std::string> {
            llvm::StringRef opName = cmpOp->getName().getStringRef();
            llvm::StringRef mnemonic = opName;
            if (opName.starts_with("waveasm."))
              mnemonic = opName.drop_front(8);
            std::string prefix;
            llvm::SmallVector<std::string> operands;
            operands.push_back("vcc");
            for (Value operand : cmpOp->getOperands()) {
              auto mat = materializeLiteralOperand(operand, kScratchVGPR);
              prefix += mat.prefix;
              operands.push_back(mat.operandStr);
            }
            return prefix + formatter.format(mnemonic, operands);
          })

      // V_ADD_U32: VOP2 commutative — literal must be in src0.
      .Case<V_ADD_U32>([&](V_ADD_U32 addOp) -> std::optional<std::string> {
        std::string dst = resolveValue(addOp.getDst());
        std::string src0 = resolveValue(addOp.getSrc0());
        std::string src1 = resolveValue(addOp.getSrc1());
        auto [isLit0, val0] = getLiteralValue(addOp.getSrc0());
        auto [isLit1, val1] = getLiteralValue(addOp.getSrc1());
        std::string prefix;
        if (isLit1 && !isInlineConstant(val1)) {
          if (isLit0 && !isInlineConstant(val0)) {
            std::string scratch = formatVGPRRange(kScratchVGPR, 1);
            prefix =
                "  v_mov_b32 " + scratch + ", " + std::to_string(val1) + "\n";
            src1 = scratch;
            peakVGPRs = std::max(peakVGPRs, kScratchVGPR + 1);
            scratchVGPRValue = val1;
          } else {
            std::swap(src0, src1);
          }
        }
        llvm::SmallVector<std::string> operands = {dst, src0, src1};
        return prefix + formatter.format("v_add_u32", operands);
      })

      // V_CNDMASK_B32: VOP2 requires src1 to be a VGPR. When either source
      // is a non-VGPR (inline constant, SGPR), emit explicit vcc to force
      // VOP3 encoding.
      .Case<V_CNDMASK_B32>(
          [&](V_CNDMASK_B32 cndOp) -> std::optional<std::string> {
            std::string dst = resolveValue(cndOp.getDst());
            auto mat0 =
                materializeLiteralOperand(cndOp.getSrc0(), kScratchVGPR);
            int nextScratch =
                mat0.prefix.empty() ? kScratchVGPR : kScratchVGPR + 1;
            auto mat1 = materializeLiteralOperand(cndOp.getSrc1(), nextScratch);
            std::string prefix = mat0.prefix + mat1.prefix;
            // After materialization, non-inline literals are in scratch VGPRs
            // (prefix non-empty). Inline constants and SGPRs pass through
            // unchanged and need VOP3 encoding.
            bool src0IsVGPR =
                !mat0.prefix.empty() || isVGPRType(cndOp.getSrc0().getType());
            bool src1IsVGPR =
                !mat1.prefix.empty() || isVGPRType(cndOp.getSrc1().getType());
            llvm::SmallVector<std::string> operands = {dst, mat0.operandStr,
                                                       mat1.operandStr};
            if (!src0IsVGPR || !src1IsVGPR)
              operands.push_back("vcc");
            return prefix + formatter.format("v_cndmask_b32", operands);
          })

      // V_CVT_BF16_F32: emit as v_cvt_pk_bf16_f32 dst, src, 0
      .Case<V_CVT_BF16_F32>(
          [&](V_CVT_BF16_F32 cvtOp) -> std::optional<std::string> {
            llvm::SmallVector<std::string> operands;
            operands.push_back(resolveValue(cvtOp.getDst()));
            operands.push_back(resolveValue(cvtOp.getSrc()));
            operands.push_back("0");
            return formatter.format("v_cvt_pk_bf16_f32", operands);
          })

      // Carry ops: on GFX9, carry-out is implicit VCC.
      // v_add_co_u32:  dst, vcc, src0, src1
      // v_addc_co_u32: dst, vcc, src0, src1, vcc  (carry-in).
      .Case<V_ADD_CO_U32, V_SUB_CO_U32, V_ADDC_CO_U32, V_SUBB_CO_U32>(
          [&](auto carryOp) -> std::optional<std::string> {
            auto mat0 =
                materializeLiteralOperand(carryOp.getSrc0(), kScratchVGPR);
            int nextScratch =
                mat0.prefix.empty() ? kScratchVGPR : kScratchVGPR + 1;
            auto mat1 =
                materializeLiteralOperand(carryOp.getSrc1(), nextScratch);
            std::string prefix = mat0.prefix + mat1.prefix;
            llvm::SmallVector<std::string> operands = {
                resolveValue(carryOp.getDst()), "vcc", mat0.operandStr,
                mat1.operandStr};
            bool hasCarryIn =
                isa<V_ADDC_CO_U32, V_SUBB_CO_U32>(carryOp.getOperation());
            if (hasCarryIn)
              operands.push_back("vcc");
            return prefix + formatter.format(carryOp->getName().stripDialect(),
                                             operands);
          })

      .Case<S_CSELECT_B32>(
          [&](S_CSELECT_B32 selOp) -> std::optional<std::string> {
            llvm::SmallVector<std::string> operands;
            operands.push_back(resolveValue(selOp.getDst()));
            operands.push_back(resolveScalarValue(selOp.getSrc0()));
            operands.push_back(resolveScalarValue(selOp.getSrc1()));
            return formatter.format("s_cselect_b32", operands);
          })

      .Default([&](Operation *defaultOp) -> std::optional<std::string> {
        llvm::StringRef opName = defaultOp->getName().getStringRef();
        llvm::StringRef mnemonic = opName;
        if (opName.starts_with("waveasm."))
          mnemonic = opName.drop_front(8);

        if (mnemonic.starts_with("v_cmp_")) {
          std::string mnem64 = (mnemonic + "_e64").str();
          llvm::SmallVector<std::string> operands;
          operands.push_back("vcc");
          for (Value operand : defaultOp->getOperands())
            operands.push_back(resolveValue(operand));
          return formatter.format(mnem64, operands);
        }

        return emitDefaultFormat(defaultOp, mnemonic);
      });
}

std::string KernelGenerator::generateLabel(LabelOp labelOp) {
  return formatter.formatLabel(labelOp.getName());
}

std::string KernelGenerator::generateComment(CommentOp commentOp) {
  return formatter.formatComment(commentOp.getText());
}

std::string KernelGenerator::generateRaw(RawOp rawOp) {
  return formatter.formatRaw(rawOp.getText());
}

llvm::SmallVector<std::string> KernelGenerator::generate() {
  llvm::SmallVector<std::string> lines;

  MetadataEmitter metaEmitter(program, target);
  auto prologue = metaEmitter.emitPrologue();
  lines.append(prologue.begin(), prologue.end());

  peakVGPRs = 0;
  peakSGPRs = 0;
  peakAGPRs = 0;
  program.walk([&](Operation *preOp) {
    for (Value result : preOp->getResults()) {
      Type ty = result.getType();
      if (auto pvreg = dyn_cast<PVRegType>(ty)) {
        peakVGPRs = std::max(peakVGPRs, pvreg.getIndex() + pvreg.getSize());
      } else if (auto psreg = dyn_cast<PSRegType>(ty)) {
        peakSGPRs = std::max(peakSGPRs, psreg.getIndex() + psreg.getSize());
      } else if (auto pareg = dyn_cast<PARegType>(ty)) {
        peakAGPRs = std::max(peakAGPRs, pareg.getIndex() + pareg.getSize());
      } else if (isVirtualRegType(ty)) {
        int64_t size = getRegSize(ty);
        int64_t physIdx = mapping.getPhysReg(result);
        if (physIdx >= 0) {
          if (isVGPRType(ty))
            peakVGPRs = std::max(peakVGPRs, physIdx + size);
          else if (isAGPRType(ty))
            peakAGPRs = std::max(peakAGPRs, physIdx + size);
          else if (isSGPRType(ty))
            peakSGPRs = std::max(peakSGPRs, physIdx + size);
        }
      }
    }
    for (Value operand : preOp->getOperands()) {
      Type ty = operand.getType();
      if (auto pvreg = dyn_cast<PVRegType>(ty))
        peakVGPRs = std::max(peakVGPRs, pvreg.getIndex() + pvreg.getSize());
      else if (auto psreg = dyn_cast<PSRegType>(ty))
        peakSGPRs = std::max(peakSGPRs, psreg.getIndex() + psreg.getSize());
      else if (auto pareg = dyn_cast<PARegType>(ty))
        peakAGPRs = std::max(peakAGPRs, pareg.getIndex() + pareg.getSize());
    }
  });
  peakVGPRs = std::max(peakVGPRs, int64_t(1));
  peakSGPRs = std::max(peakSGPRs, int64_t(2));
  if (auto minSgprs = program->getAttrOfType<IntegerAttr>("min_sgprs"))
    peakSGPRs = std::max(peakSGPRs, minSgprs.getInt());

  for (Operation &op : program.getBodyBlock()) {
    if (auto labelOp = dyn_cast<LabelOp>(op)) {
      lines.push_back(generateLabel(labelOp));
      invalidateScratchCache();
      continue;
    }
    if (auto commentOp = dyn_cast<CommentOp>(op)) {
      lines.push_back(generateComment(commentOp));
      continue;
    }
    if (auto rawOp = dyn_cast<RawOp>(op)) {
      lines.push_back(generateRaw(rawOp));
      continue;
    }

    // Generate instruction (with literal handling for VALU ops)
    auto instrLines = generateOpWithLiteralHandling(&op);
    lines.append(instrLines.begin(), instrLines.end());
  }

  int64_t ldsSize = program.getLdsSize().value_or(0);
  auto epilogue =
      metaEmitter.emitEpilogue(peakVGPRs, peakSGPRs, peakAGPRs, ldsSize);
  lines.append(epilogue.begin(), epilogue.end());

  return lines;
}

//===----------------------------------------------------------------------===//
// Assembly Output Functions
//===----------------------------------------------------------------------===//

LogicalResult writeAssembly(ProgramOp program, const PhysicalMapping &mapping,
                            llvm::StringRef outputPath) {
  std::error_code ec;
  llvm::raw_fd_ostream os(outputPath, ec);
  if (ec) {
    return program.emitError() << "Failed to open output file: " << outputPath;
  }

  return writeAssembly(program, mapping, os);
}

LogicalResult writeAssembly(ProgramOp program, const PhysicalMapping &mapping,
                            llvm::raw_ostream &os) {
  auto targetAttr = program.getTarget();
  if (!targetAttr) {
    return program.emitError() << "target attribute not specified";
  }

  auto targetKind = targetAttr.getTargetKind();
  KernelGenerator generator(program, mapping, targetKind);
  auto lines = generator.generate();

  // Check if generated assembly exceeds hardware register limits.
  // The emitter may allocate extra registers (e.g., swap temps for loop
  // back-edge copies) beyond what the register allocator budgeted.  Catch
  // the overflow here with a single clear diagnostic instead of letting
  // the assembler produce many "register index is out of range" errors.
  int64_t peakVGPRs = generator.getPeakVGPRs();
  int64_t peakAGPRs = generator.getPeakAGPRs();
  int64_t maxVGPRs = targetKind.getMaxVGPRs();
  int64_t maxAGPRs = targetKind.getMaxAGPRs();

  std::string overflowDetails;
  llvm::raw_string_ostream detail(overflowDetails);
  bool overflow = false;
  if (peakVGPRs > maxVGPRs) {
    detail << "VGPRs: " << peakVGPRs << " used, " << maxVGPRs << " limit";
    overflow = true;
  }
  if (maxAGPRs > 0 && peakAGPRs > maxAGPRs) {
    if (overflow)
      detail << "; ";
    detail << "AGPRs: " << peakAGPRs << " used, " << maxAGPRs << " limit";
    overflow = true;
  }

  if (overflow) {
    InFlightDiagnostic diag =
        program.emitError()
        << "kernel '" << getKernelName(program)
        << "' exceeds hardware register limit: " << detail.str();
    diag.attachNote(program.getLoc())
        << "Register spilling is not supported; reduce register pressure "
           "by optimizing the kernel (e.g., smaller tile sizes, fewer "
           "unrolled iterations).";
    return diag;
  }

  for (const auto &line : lines) {
    os << line << "\n";
  }

  return success();
}

} // namespace waveasm
