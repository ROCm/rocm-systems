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

#include "llvm/IR/Constants.h"

using namespace llvm;

void SQTTInstrumentPass::emitFuncMap(Module& M)
{
    std::string mapData;
    // Instrumented device functions: "F:ID:name" (+ "@source_loc" if known)
    for (auto& entry : FuncMap)
    {
        mapData += "F:";
        mapData += std::to_string(entry.ID);
        mapData += ':';
        mapData += entry.Name;
        if (!entry.SourceLoc.empty())
        {
            mapData += '@';
            mapData += entry.SourceLoc;
        }
        mapData += '\n';
    }
    // Kernels (not instrumented, for name/vaddr lookup): "K:name" (+ "@loc")
    for (auto& kentry : KernelNames)
    {
        mapData += "K:";
        mapData += kentry.Name;
        if (!kentry.SourceLoc.empty())
        {
            mapData += '@';
            mapData += kentry.SourceLoc;
        }
        mapData += '\n';
    }
    // Named user markers: "U:ID:name" for scope, "P:ID:name" for points
    for (auto& entry : UserMarkers)
    {
        mapData += entry.IsPoint ? "P:" : "U:";
        mapData += std::to_string(entry.ID);
        mapData += ':';
        mapData += entry.Name;
        mapData += '\n';
    }
    // Point markers (barriers, memory ops): "P:ID:name"
    if (Config.InstrumentBarriers)
    {
        mapData += "P:" + std::to_string(BarrierSignalID) + ":barrier_signal\n";
        mapData += "P:" + std::to_string(BarrierWaitID) + ":barrier_wait\n";
        mapData += "P:" + std::to_string(BarrierFullID) + ":barrier\n";
    }
    if (Config.InstrumentMemory)
    {
        mapData += "P:" + std::to_string(VmemLoadID) + ":vmem_load\n";
        mapData += "P:" + std::to_string(VmemStoreID) + ":vmem_store\n";
    }
    // Address trace: wave size and per-op unique IDs
    if (AddrTraceWaveSize > 0)
    {
        mapData += "W:";
        mapData += std::to_string(AddrTraceWaveSize);
        mapData += '\n';
    }
    for (auto& entry : AddrTraceEntries)
    {
        mapData += "P:";
        mapData += std::to_string(entry.ID);
        mapData += ':';
        mapData += entry.Kind;
        if (!entry.SourceLoc.empty())
        {
            mapData += '@';
            mapData += entry.SourceLoc;
        }
        mapData += '\n';
        if (entry.ExtraPayloadCount > 0)
        {
            mapData += "R:";
            mapData += std::to_string(entry.ID);
            mapData += ":extra_payload_count=";
            mapData += std::to_string(entry.ExtraPayloadCount);
            mapData += '\n';
        }
    }

    LLVMContext& Ctx = M.getContext();
    Constant* StrConst = ConstantDataArray::getString(
        Ctx,
        mapData,
        /*AddNull=*/true
    );

    // Use addrspace(1) for AMDGPU global memory
    unsigned AS = M.getDataLayout().getDefaultGlobalsAddressSpace();
    auto* GV = new GlobalVariable(
        M,
        StrConst->getType(),
        /*isConstant=*/true,
        GlobalValue::InternalLinkage,
        StrConst,
        ".sqtt_func_id_map",
        /*InsertBefore=*/nullptr,
        GlobalVariable::NotThreadLocal,
        AS
    );
    GV->setSection(".sqtt_funcmap");
    GV->setAlignment(Align(1));

    // Append to llvm.used to prevent stripping
    SmallVector<Constant*, 1> UsedVals;
    Constant* GVPtr = ConstantExpr::getPointerBitCastOrAddrSpaceCast(GV, PointerType::getUnqual(Ctx));
    UsedVals.push_back(GVPtr);

    auto* UsedTy = ArrayType::get(PointerType::getUnqual(Ctx), 1);
    auto* UsedInit = ConstantArray::get(UsedTy, UsedVals);

    GlobalVariable* LLVMUsed = M.getGlobalVariable("llvm.used");
    if (LLVMUsed)
    {
        SmallVector<Constant*, 8> Ops;
        if (auto* Init = LLVMUsed->getInitializer())
        {
            if (auto* CA = dyn_cast<ConstantArray>(Init))
            {
                for (unsigned i = 0; i < CA->getNumOperands(); i++) Ops.push_back(CA->getOperand(i));
            }
        }
        Ops.push_back(GVPtr);
        auto* NewTy = ArrayType::get(PointerType::getUnqual(Ctx), Ops.size());
        auto* NewInit = ConstantArray::get(NewTy, Ops);
        LLVMUsed->eraseFromParent();
        auto* NewUsed = new GlobalVariable(M, NewTy, false, GlobalValue::AppendingLinkage, NewInit, "llvm.used");
        NewUsed->setSection("llvm.metadata");
    }
    else
    {
        auto* NewUsed = new GlobalVariable(M, UsedTy, false, GlobalValue::AppendingLinkage, UsedInit, "llvm.used");
        NewUsed->setSection("llvm.metadata");
    }
}
