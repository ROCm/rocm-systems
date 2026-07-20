// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <cstdint>

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"

#include "SQTTConfig.h"

constexpr const char* SQTT_MARKER_HEADER_METADATA = "sqtt.marker_header";
constexpr const char* SQTT_RAW_PAYLOAD_METADATA = "sqtt.raw_payload";
constexpr const char* SQTT_PAYLOAD_GROUP_METADATA = "sqtt.payload_group";

// ============================================================================
// Architecture detection
// ============================================================================

enum class GfxGen
{
    GFX9,
    RDNA,
    GFX12,
    Unknown
};

inline GfxGen getGfxGen(const llvm::Function& F)
{
    llvm::Attribute A = F.getFnAttribute("target-cpu");
    if (!A.isValid()) return GfxGen::Unknown;
    llvm::StringRef CPU = A.getValueAsString();

    if (CPU.starts_with("gfx9")) return GfxGen::GFX9;
    if (CPU.starts_with("gfx12")) return GfxGen::GFX12;
    if (CPU.starts_with("gfx10") || CPU.starts_with("gfx11")) return GfxGen::RDNA;
    return GfxGen::Unknown;
}

// gfx1200 and gfx1201 use HW_REG_SHADER_CYCLES_LO through s_getreg; gfx1250
// and subsequent targets have a dedicated shader-cycle instruction.
inline bool hasShaderCyclesU64(const llvm::Function& F)
{
    llvm::Attribute A = F.getFnAttribute("target-cpu");
    if (!A.isValid()) return false;
    llvm::StringRef CPU = A.getValueAsString();
    if (!CPU.consume_front("gfx")) return false;
    unsigned target = 0;
    return !CPU.consumeInteger(10, target) && target > 1201;
}

// Does this GfxGen support s_ttracedata_imm?
inline bool supportsImmTrace(GfxGen gen)
{
    return gen == GfxGen::RDNA || gen == GfxGen::GFX12; // gfx10+
}

// Wave size for this architecture
inline unsigned getWaveSize(GfxGen gen) { return (gen == GfxGen::GFX9) ? 64 : 32; }

struct HwRegEncodings
{
    uint32_t wave, simd, cu, wg;
};

inline HwRegEncodings getHwRegEncodings(GfxGen gen)
{
    if (gen == GfxGen::GFX9) return {GFX9_HWREG_WAVE, GFX9_HWREG_SIMD, GFX9_HWREG_CU, GFX9_HWREG_WG};
    return {RDNA_HWREG_WAVE, RDNA_HWREG_SIMD, RDNA_HWREG_CU, RDNA_HWREG_WG};
}

inline llvm::Value* getMemoryPointer(llvm::Instruction* I)
{
    if (auto* LI = llvm::dyn_cast<llvm::LoadInst>(I)) return LI->getPointerOperand();
    if (auto* SI = llvm::dyn_cast<llvm::StoreInst>(I)) return SI->getPointerOperand();
    if (auto* AI = llvm::dyn_cast<llvm::AtomicRMWInst>(I)) return AI->getPointerOperand();
    if (auto* AX = llvm::dyn_cast<llvm::AtomicCmpXchgInst>(I)) return AX->getPointerOperand();
    return nullptr;
}

// ============================================================================
// Instruction cost model
// ============================================================================

inline bool isCountedInstruction(const llvm::Instruction& I)
{
    return !llvm::isa<llvm::PHINode>(I) && !llvm::isa<llvm::AllocaInst>(I) && !I.isDebugOrPseudoInst() &&
           !llvm::isa<llvm::UnreachableInst>(I);
}

inline unsigned instructionCost(const llvm::Instruction& I)
{
    if (!isCountedInstruction(I)) return 0;
    // Check for lifetime intrinsics
    if (auto* CI = llvm::dyn_cast<llvm::CallInst>(&I))
    {
        if (auto* F = CI->getCalledFunction())
        {
            llvm::StringRef Name = F->getName();
            if (Name.starts_with("llvm.lifetime.")) return 0;
            if (Name.starts_with("llvm.dbg.")) return 0;
            // Matrix ops
            if (Name.starts_with("llvm.amdgcn.mfma.") || Name.starts_with("llvm.amdgcn.wmma.")) return 16;
            // LDS intrinsics
            if (Name.starts_with("llvm.amdgcn.ds.")) return 4;
        }
    }
    // Memory operations
    if (auto* LI = llvm::dyn_cast<llvm::LoadInst>(&I))
    {
        unsigned AS = LI->getPointerAddressSpace();
        if (AS == 3) return 4; // LDS
        return 10;             // global/flat
    }
    if (auto* SI = llvm::dyn_cast<llvm::StoreInst>(&I))
    {
        unsigned AS = SI->getPointerAddressSpace();
        if (AS == 3) return 4;
        return 10;
    }
    return 1;
}

inline unsigned computeFunctionSize(const llvm::Function& F, CostMode mode)
{
    unsigned total = 0;
    for (auto& BB : F)
    {
        for (auto& I : BB)
        {
            // Pass-owned marker calls must not make a function appear large
            // enough to retain the instrumentation that introduced them.
            if (I.getMetadata(SQTT_MARKER_HEADER_METADATA) || I.getMetadata(SQTT_RAW_PAYLOAD_METADATA) ||
                I.getMetadata(SQTT_PAYLOAD_GROUP_METADATA))
                continue;
            total += mode == CostMode::WeightedCost ? instructionCost(I) : isCountedInstruction(I);
        }
    }
    return total;
}

enum class BufferOpKind : uint8_t
{
    None,
    Load,
    Store,
    Atomic
};

inline BufferOpKind classifyBufferOp(llvm::StringRef name)
{
    for (const char* prefix : {"llvm.amdgcn.raw.buffer.", "llvm.amdgcn.struct.buffer.",
                               "llvm.amdgcn.raw.ptr.buffer.", "llvm.amdgcn.struct.ptr.buffer."})
    {
        llvm::StringRef opcode = name;
        if (!opcode.consume_front(prefix)) continue;
        if (opcode.starts_with("load")) return BufferOpKind::Load;
        if (opcode.starts_with("store")) return BufferOpKind::Store;
        if (opcode.starts_with("atomic")) return BufferOpKind::Atomic;
        return BufferOpKind::None;
    }
    return BufferOpKind::None;
}

inline bool isStructBuffer(llvm::StringRef name) { return name.contains("struct"); }

inline bool isBufferCmpSwap(llvm::StringRef name) { return name.contains("cmpswap"); }
