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

#include "SQTTPass.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/Passes/PassBuilder.h"
#if __has_include("llvm/Plugins/PassPlugin.h")
#    include "llvm/Plugins/PassPlugin.h"
#else
#    include "llvm/Passes/PassPlugin.h"
#endif
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

PreservedAnalyses SQTTInstrumentPass::run(Module& M, ModuleAnalysisManager& MAM)
{
    if (!Triple(M.getTargetTriple()).isAMDGPU()) return PreservedAnalyses::all();

    if (PassMode == Mode::Early)
        return runEarly(M);
    else
        return runLate(M);
}

PreservedAnalyses SQTTInstrumentPass::runEarly(Module& M)
{
    bool Changed = false;
    LLVMContext& Ctx = M.getContext();

    // Force-inline all callers of the named marker sentinels.
    for (const char* Name : {"__sqtt_named_marker_enter", "__sqtt_named_marker_exit", "__sqtt_named_marker_point"})
    {
        Function* S = M.getFunction(Name);
        if (!S) continue;
        SmallVector<Function*, 4> Wrappers;
        for (User* U : S->users())
        {
            auto* CI = dyn_cast<CallInst>(U);
            if (!CI) continue;
            Function* Wrapper = CI->getFunction();
            if (Wrapper && Wrapper != S) Wrappers.push_back(Wrapper);
        }
        for (Function* W : Wrappers)
        {
            SmallVector<CallInst*, 8> CallSites;
            for (User* U : W->users())
            {
                if (auto* CI = dyn_cast<CallInst>(U)) CallSites.push_back(CI);
            }
            for (auto* CS : CallSites)
            {
                InlineFunctionInfo IFI;
                InlineFunction(*CS, IFI);
                Changed = true;
            }
        }
    }

    // Now resolve sentinel calls that are directly visible
    for (auto& F : M)
    {
        if (F.isDeclaration()) continue;
        GfxGen Gen = getGfxGen(F);
        if (Gen == GfxGen::Unknown) continue;
        Changed |= resolveNamedMarkersEarly(F, Gen);
    }

    // Clean up sentinel declarations
    for (const char* Name : {"__sqtt_named_marker_enter", "__sqtt_named_marker_exit", "__sqtt_named_marker_point"})
    {
        Function* S = M.getFunction(Name);
        if (!S) continue;
        if (S->use_empty()) S->eraseFromParent();
    }

    // Store user marker map in module metadata for late pass
    if (!UserMarkers.empty()) storeUserMarkerMetadata(M, Ctx);

    for (auto& F : M)
    {
        if (F.isDeclaration()) continue;
        GfxGen Gen = getGfxGen(F);
        if (Gen == GfxGen::Unknown) continue;

        if (F.getCallingConv() == CallingConv::AMDGPU_KERNEL) continue;

        if (Config.FunctionThreshold == 0) continue;

        uint32_t id = NextEventID++;

        insertBareMarkers(F, id, Gen);
        storeFuncMetadata(F, id, Ctx);

        Changed = true;
    }

    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

PreservedAnalyses SQTTInstrumentPass::runLate(Module& M)
{
    bool Changed = false;

    bool hadEarlyFuncInst = M.getNamedMetadata("sqtt.funcmap.early") != nullptr;
    bool hadEarlyMarkers = M.getNamedMetadata("sqtt.usermarkers.early") != nullptr;
    bool hadEarlyPass = hadEarlyFuncInst || hadEarlyMarkers;

    recoverUserMarkerMetadata(M);

    if (hadEarlyFuncInst)
    {
        Changed |= filterInstrumentedFunctions(M);
        NextEventID = compactFuncIDs(M);
    }

    if (Config.InstrumentBarriers)
    {
        BarrierSignalID = NextEventID++;
        BarrierWaitID = NextEventID++;
        BarrierFullID = NextEventID++;
    }
    if (Config.InstrumentMemory)
    {
        VmemLoadID = NextEventID++;
        VmemStoreID = NextEventID++;
    }

    for (auto& F : M)
    {
        if (F.isDeclaration()) continue;
        GfxGen Gen = getGfxGen(F);
        if (Gen == GfxGen::Unknown) continue;

        CurScopeCheck = nullptr; // reset per function

        bool IsKernel = F.getCallingConv() == CallingConv::AMDGPU_KERNEL;

        if (IsKernel) KernelNames.push_back({F.getName().str(), getFunctionSourceLoc(F)});

        if (Config.needsScopeCheck()) Changed |= wrapExistingMarkers(F, Gen);

        if (hadEarlyPass) Changed |= addBarriersToExistingMarkers(F);

        Changed |= processNamedMarkers(F, Gen);

        if (Config.InstrumentBarriers) Changed |= instrumentBarriers(F, Gen);

        if (Config.InstrumentMemory) Changed |= instrumentMemoryOps(F, Gen);

        if (Config.hasAddressTracing()) Changed |= instrumentAddressTraces(F, Gen);

        if (!hadEarlyPass && Config.FunctionThreshold > 0 && !IsKernel) Changed |= instrumentFunctionDirect(F, Gen);
    }

    // Clean up sentinel declarations
    for (const char* Name : {"__sqtt_named_marker_enter", "__sqtt_named_marker_exit", "__sqtt_named_marker_point"})
    {
        if (Function* S = M.getFunction(Name))
            if (S->use_empty()) S->eraseFromParent();
    }

    if (!FuncMap.empty() || !KernelNames.empty() || !UserMarkers.empty() || Config.InstrumentBarriers ||
        Config.InstrumentMemory || !AddrTraceEntries.empty())
    {
        emitFuncMap(M);
        Changed = true;
    }

    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

// ============================================================================
// Plugin entry point
// ============================================================================

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo()
{
    return {
        LLVM_PLUGIN_API_VERSION,
        "SQTTInstrument",
        "0.1",
        [](PassBuilder& PB)
        {
            using Mode = SQTTInstrumentPass::Mode;
            auto Cfg = SQTTConfig::fromEnvironment();

            PB.registerPipelineEarlySimplificationEPCallback(
                [Cfg](ModulePassManager& MPM, OptimizationLevel OL, ThinOrFullLTOPhase)
                {
                    if (OL != OptimizationLevel::O0) MPM.addPass(SQTTInstrumentPass(Cfg, Mode::Early));
                }
            );

            PB.registerOptimizerLastEPCallback(
                [Cfg](ModulePassManager& MPM, OptimizationLevel OL, ThinOrFullLTOPhase)
                {
                    if (OL != OptimizationLevel::O0) MPM.addPass(SQTTInstrumentPass(Cfg, Mode::Late));
                }
            );

            PB.registerPipelineStartEPCallback(
                [Cfg](ModulePassManager& MPM, OptimizationLevel OL)
                {
                    if (OL == OptimizationLevel::O0) MPM.addPass(SQTTInstrumentPass(Cfg, Mode::Late));
                }
            );
        }};
}
