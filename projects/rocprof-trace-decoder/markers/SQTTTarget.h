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
    switch (gen)
    {
        case GfxGen::GFX9: return {GFX9_HWREG_WAVE, GFX9_HWREG_SIMD, GFX9_HWREG_CU, GFX9_HWREG_WG};
        case GfxGen::RDNA: return {RDNA_HWREG_WAVE, RDNA_HWREG_SIMD, RDNA_HWREG_CU, RDNA_HWREG_WG};
        case GfxGen::GFX12: return {RDNA_HWREG_WAVE, RDNA_HWREG_SIMD, RDNA_HWREG_CU, RDNA_HWREG_WG};
        default: return {RDNA_HWREG_WAVE, RDNA_HWREG_SIMD, RDNA_HWREG_CU, RDNA_HWREG_WG};
    }
}

inline unsigned getShaderClockBits(const SQTTConfig& config, GfxGen gen)
{
    if (config.ShaderClockBits != SQTTConfig::AutoShaderClockBits) return config.ShaderClockBits;
    return gen == GfxGen::GFX12 ? 12 : 0;
}

inline bool usesShaderClockPacking(const SQTTConfig& config, GfxGen gen)
{
    return gen == GfxGen::GFX12 && getShaderClockBits(config, gen) != 0;
}

// ============================================================================
// Instruction cost model
// ============================================================================

inline unsigned instructionCost(const llvm::Instruction& I)
{
    if (llvm::isa<llvm::PHINode>(I) || llvm::isa<llvm::AllocaInst>(I)) return 0;
    if (I.isDebugOrPseudoInst()) return 0;
    if (llvm::isa<llvm::UnreachableInst>(I)) return 0;
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
            if (I.getMetadata("sqtt.marker_header") || I.getMetadata("sqtt.raw_payload")) continue;
            if (mode == CostMode::WeightedCost)
                total += instructionCost(I);
            else
            {
                // Instruction count: skip non-substantive
                if (!llvm::isa<llvm::PHINode>(I) && !llvm::isa<llvm::AllocaInst>(I) && !I.isDebugOrPseudoInst() &&
                    !llvm::isa<llvm::UnreachableInst>(I))
                    total += 1;
            }
        }
    }
    return total;
}

// ============================================================================
// Buffer intrinsic classification
// ============================================================================

inline bool isBufferLoad(llvm::StringRef Name)
{
    return Name.starts_with("llvm.amdgcn.raw.buffer.load") || Name.starts_with("llvm.amdgcn.struct.buffer.load") ||
           Name.starts_with("llvm.amdgcn.raw.ptr.buffer.load") ||
           Name.starts_with("llvm.amdgcn.struct.ptr.buffer.load");
}

inline bool isBufferStore(llvm::StringRef Name)
{
    return Name.starts_with("llvm.amdgcn.raw.buffer.store") || Name.starts_with("llvm.amdgcn.struct.buffer.store") ||
           Name.starts_with("llvm.amdgcn.raw.ptr.buffer.store") ||
           Name.starts_with("llvm.amdgcn.struct.ptr.buffer.store");
}

inline bool isBufferAtomic(llvm::StringRef Name)
{
    return Name.starts_with("llvm.amdgcn.raw.buffer.atomic") || Name.starts_with("llvm.amdgcn.struct.buffer.atomic") ||
           Name.starts_with("llvm.amdgcn.raw.ptr.buffer.atomic") ||
           Name.starts_with("llvm.amdgcn.struct.ptr.buffer.atomic");
}

inline bool isBufferOp(llvm::StringRef Name)
{
    return isBufferLoad(Name) || isBufferStore(Name) || isBufferAtomic(Name);
}

inline bool isStructBuffer(llvm::StringRef Name) { return Name.contains("struct"); }

inline bool isBufferCmpSwap(llvm::StringRef Name) { return Name.contains("cmpswap"); }
