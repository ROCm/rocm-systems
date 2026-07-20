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

#include "llvm/Passes/PassBuilder.h"
#if __has_include("llvm/Plugins/PassPlugin.h")
#    include "llvm/Plugins/PassPlugin.h"
#else
#    include "llvm/Passes/PassPlugin.h"
#endif
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <algorithm>

using namespace llvm;

namespace
{

constexpr const char* MarkerSentinelNames[] = {
    "__sqtt_named_marker_enter", "__sqtt_named_marker_exit", "__sqtt_named_marker_point", "__sqtt_named_marker_data"
};

void eraseUnusedMarkerSentinels(Module& M)
{
    for (const char* Name : MarkerSentinelNames)
        if (Function* F = M.getFunction(Name); F && F->use_empty()) F->eraseFromParent();
}

bool inlineMarkerWrappers(Module& M)
{
    SmallVector<Function*, 4> wrappers;
    for (const char* Name : MarkerSentinelNames)
    {
        Function* Sentinel = M.getFunction(Name);
        if (!Sentinel) continue;
        for (User* U : Sentinel->users())
        {
            auto* Call = dyn_cast<CallInst>(U);
            Function* Wrapper = Call ? Call->getFunction() : nullptr;
            if (Wrapper && Wrapper != Sentinel &&
                std::find(wrappers.begin(), wrappers.end(), Wrapper) == wrappers.end())
                wrappers.push_back(Wrapper);
        }
    }

    bool changed = false;
    for (Function* Wrapper : wrappers)
    {
        SmallVector<CallInst*, 8> callSites;
        for (User* U : Wrapper->users())
            if (auto* Call = dyn_cast<CallInst>(U)) callSites.push_back(Call);
        for (CallInst* Call : callSites)
        {
            InlineFunctionInfo IFI;
            InlineFunction(*Call, IFI);
            changed = true;
        }
    }
    return changed;
}

template <typename Visit>
bool visitTargetFunctions(Module& M, Visit&& visit)
{
    bool changed = false;
    for (Function& F : M)
    {
        if (F.isDeclaration()) continue;
        GfxGen gen = getGfxGen(F);
        if (gen != GfxGen::Unknown) changed |= visit(F, gen);
    }
    return changed;
}

} // namespace

PreservedAnalyses SQTTInstrumentPass::run(Module& M, ModuleAnalysisManager& MAM)
{
    if (!Triple(M.getTargetTriple()).isAMDGPU()) return PreservedAnalyses::all();
    return PassMode == Mode::Early ? runEarly(M) : runLate(M);
}

PreservedAnalyses SQTTInstrumentPass::runEarly(Module& M)
{
    bool Changed = false;
    LLVMContext& Ctx = M.getContext();

    // Force-inline all callers of the named marker sentinels.
    Changed |= inlineMarkerWrappers(M);

    // Now resolve sentinel calls that are directly visible.
    Changed |= visitTargetFunctions(M, [&](Function& F, GfxGen gen)
    {
        return processMarkerCalls(F, gen, /*useBareTrace=*/true);
    });

    eraseUnusedMarkerSentinels(M);

    Changed |= visitTargetFunctions(M, [&](Function& F, GfxGen gen)
    {
        if (F.getCallingConv() == CallingConv::AMDGPU_KERNEL || Config.FunctionThreshold == 0 ||
            hasMustTailCall(F))
            return false;
        uint32_t id = NextEventID++;
        Type* I32 = Type::getInt32Ty(Ctx);
        F.setMetadata("sqtt.func.id", MDNode::get(Ctx, {ConstantAsMetadata::get(ConstantInt::get(I32, id))}));
        Markers.push_back({id, MarkerKind::Function, F.getName().str(), getFunctionSourceLoc(F),
                           computeFunctionSize(F, Config.Mode)});
        insertFunctionMarkers(F, id, gen, /*useBareTrace=*/true);
        return true;
    });

    if (!Markers.empty()) storeEarlyMarkerMetadata(M, Ctx);

    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

PreservedAnalyses SQTTInstrumentPass::runLate(Module& M)
{
    bool Changed = false;

    bool hadEarlyFuncInst = false;
    bool hadEarlyPass = recoverEarlyMarkerMetadata(M, hadEarlyFuncInst);

    if (hadEarlyFuncInst)
        Changed |= finalizeEarlyFunctionMarkers(M);

    auto addSystemMarkers = [&](std::initializer_list<const char*> names)
    {
        uint32_t firstID = NextEventID;
        for (const char* name : names) Markers.push_back({NextEventID++, MarkerKind::SystemPoint, name});
        return firstID;
    };
    if (Config.InstrumentBarriers)
        FirstBarrierID = addSystemMarkers({"barrier_signal", "barrier_wait", "barrier"});
    if (Config.MemoryChunkSize)
        FirstVmemID = addSystemMarkers({"vmem_load", "vmem_store"});

    // A nonzero clock field must wait until every payload-producing protocol
    // has been discovered. The default no-clock path can lower each function
    // as soon as all of its markers have been inserted.
    const bool deferFullTraceFinalization = Config.ShaderClockBits != 0;
    Changed |= visitTargetFunctions(M, [&](Function& F, GfxGen gen)
    {
        CurScopeCheck = nullptr; // reset per function
        bool isKernel = F.getCallingConv() == CallingConv::AMDGPU_KERNEL;
        if (isKernel) Markers.push_back({0, MarkerKind::Kernel, F.getName().str(), getFunctionSourceLoc(F)});

        bool changed = finalizeExistingMarkers(F, gen);
        changed |= processMarkerCalls(F, gen, /*useBareTrace=*/false);
        if (Config.InstrumentBarriers) changed |= instrumentBarriers(F, gen);
        if (Config.MemoryChunkSize) changed |= instrumentMemoryOps(F, gen);
        if (Config.hasAddressTracing()) changed |= instrumentAddressTraces(F, gen);
        if (!hadEarlyPass && Config.FunctionThreshold > 0 && !isKernel)
            changed |= instrumentFunctionDirect(F, gen);
        if (!deferFullTraceFinalization) changed |= finalizeFullTraces(F, gen);
        return changed;
    });

    if (Config.ShaderClockBits != 0 && std::any_of(Markers.begin(), Markers.end(), [](const MarkerRecord& entry)
        { return entry.ExtraPayloadCount != 0; }))
        report_fatal_error("SQTT payload markers require SQTT_SHADER_CLOCK_BITS=0");

    if (deferFullTraceFinalization)
        Changed |= visitTargetFunctions(M, [&](Function& F, GfxGen gen) { return finalizeFullTraces(F, gen); });

    eraseUnusedMarkerSentinels(M);

    if (!Markers.empty() || ShaderClockBitsUsed > 0)
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
