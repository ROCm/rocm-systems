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
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <algorithm>

using namespace llvm;

void SQTTInstrumentPass::emitFuncMap(Module& M)
{
    // Bucket once, preserving insertion order within every non-function row
    // class. Only functions need an ID sort.
    auto rowFor = [](MarkerKind kind)
    {
        if (kind == MarkerKind::Function) return 0u;
        if (kind == MarkerKind::Kernel) return 1u;
        if (kind == MarkerKind::UserScope || kind == MarkerKind::Point) return 2u;
        if (kind == MarkerKind::SystemPoint) return 3u;
        if (kind == MarkerKind::AddressPoint) return 4u;
        llvm_unreachable("invalid marker kind");
    };
    SmallVector<const MarkerRecord*, 8> rows[5];
    for (const MarkerRecord& entry : Markers) rows[rowFor(entry.Kind)].push_back(&entry);
    std::sort(rows[0].begin(), rows[0].end(), [](const MarkerRecord* a, const MarkerRecord* b)
    {
        return a->ID < b->ID;
    });

    std::string mapData;
    if (ShaderClockBitsUsed > 0)
    {
        mapData += "M:shader_clock_bits=";
        mapData += std::to_string(ShaderClockBitsUsed);
        mapData += ";shader_clock_shift=";
        mapData += std::to_string(Config.ShaderClockShift);
        mapData += '\n';
    }

    for (unsigned row = 0; row < 5; ++row)
    {
        if (row == 4 && AddrTraceWaveSize > 0)
        {
            mapData += "W:";
            mapData += std::to_string(AddrTraceWaveSize);
            mapData += '\n';
        }
        for (const MarkerRecord* entry : rows[row])
        {
            char kind = entry->Kind == MarkerKind::Kernel   ? 'K'
                        : entry->Kind == MarkerKind::Function ? 'F'
                        : entry->Kind == MarkerKind::UserScope ? 'U'
                                                             : 'P';
            mapData += kind;
            mapData += ':';
            if (kind != 'K')
            {
                mapData += std::to_string(entry->ID);
                mapData += ':';
            }
            mapData += entry->Name;
            if (!entry->SourceLoc.empty())
            {
                mapData += '@';
                mapData += entry->SourceLoc;
            }
            mapData += '\n';
            if (kind != 'K' && entry->ExtraPayloadCount)
            {
                mapData += "R:";
                mapData += std::to_string(entry->ID);
                mapData += ":extra_payload_count=";
                mapData += std::to_string(entry->ExtraPayloadCount);
                mapData += '\n';
            }
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

    appendToUsed(M, {GV});
}
