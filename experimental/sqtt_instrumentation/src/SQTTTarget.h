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
    Unknown
};

inline GfxGen getGfxGen(const llvm::Function& F)
{
    llvm::Attribute A = F.getFnAttribute("target-cpu");
    if (!A.isValid()) return GfxGen::Unknown;
    llvm::StringRef CPU = A.getValueAsString();

    if (CPU.starts_with("gfx9")) return GfxGen::GFX9;
    if (CPU.starts_with("gfx10") || CPU.starts_with("gfx11") || CPU.starts_with("gfx12")) return GfxGen::RDNA;
    return GfxGen::Unknown;
}

// Does this GfxGen support s_ttracedata_imm?
inline bool supportsImmTrace(GfxGen gen)
{
    return gen == GfxGen::RDNA; // gfx10+ = all RDNA
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
        default: return {RDNA_HWREG_WAVE, RDNA_HWREG_SIMD, RDNA_HWREG_CU, RDNA_HWREG_WG};
    }
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
