// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// MI450 (gfx1250) architecture implementation.
// GFX12_VARIANT must be set to GFX12_VARIANT_1250 before including gfx12_def.h
// so that the correct register headers and block table entries are used.
#define GFX12_VARIANT 0x1250
#include "core/architectures/gfx12_architecture.hpp"
#include "def/gfx12_def.h"
#include "pm4/gfx12_cmd_builder.h"

namespace aql_profile {

Mi450Architecture::Mi450Architecture(const AgentInfo* agent_info)
    : Gfx12Architecture(agent_info) {
  config_.xcc_per_aid = 4;
  // Re-initialize the block table with the gfx1250-specific entries.
  // The base class constructor has already called InitializeBlockTable() for
  // gfx1200, so we override it here.
  InitializeBlockTable();
}

void Mi450Architecture::InitializeBlockTable() {
  static const GpuBlockInfo* table[LastCounterBlockId + 1]{};

  // AIGC blocks
  table[__BLOCK_ID(GCEA_SE)]   = &GceaSeCounterBlockInfo;
  table[__BLOCK_ID_HSA(GL2A)]  = &Gl2aCounterBlockInfo;
  table[__BLOCK_ID_HSA(GL2C)]  = &Gl2cCounterBlockInfo;
  table[__BLOCK_ID(GRBMA)]     = &GrbmaCounterBlockInfo;
  table[__BLOCK_ID_HSA(ATCL2)] = &Atcl2CounterBlockInfo;
  table[__BLOCK_ID(GC_UTCL2)]  = &GcUtcl2CounterBlockInfo;
  table[__BLOCK_ID(GC_VML2)]   = &GcVml2CounterBlockInfo;
  table[__BLOCK_ID(GC_FFBM)]   = &Gcutcl2FfbmCounterBlockInfo;
  table[__BLOCK_ID(GC_NHTTLB)] = &Gcutcl2NhttlbCounterBlockInfo;
  table[__BLOCK_ID(GC_L2TLB)]  = &GcL2tlbCounterBlockInfo;
  // Global blocks
  table[__BLOCK_ID(CHA)]       = &ChaCounterBlockInfo;
  table[__BLOCK_ID(CHC)]       = &ChcCounterBlockInfo;
  table[__BLOCK_ID_HSA(CPC)]   = &CpcCounterBlockInfo;
  table[__BLOCK_ID_HSA(CPF)]   = &CpfCounterBlockInfo;
  table[__BLOCK_ID(CPG)]       = &CpgCounterBlockInfo;
  table[__BLOCK_ID_HSA(GCR)]   = &GcrCounterBlockInfo;
  table[__BLOCK_ID(GC_CANE)]   = &GcCaneCounterBlockInfo;
  table[__BLOCK_ID(GLARBA)]    = &GlarbaCounterBlockInfo;
  table[__BLOCK_ID(GLARBC)]    = &GlarbcCounterBlockInfo;
  table[__BLOCK_ID_HSA(GRBM)]  = &GrbmCounterBlockInfo;
  table[__BLOCK_ID(RLC)]       = &RlcCounterBlockInfo;
  table[__BLOCK_ID_HSA(SDMA)]  = &SdmaCounterBlockInfo;
  // SE blocks
  table[__BLOCK_ID_HSA(GL1A)]  = &Gl1aCounterBlockInfo;
  table[__BLOCK_ID_HSA(GL1C)]  = &Gl1cCounterBlockInfo;
  table[__BLOCK_ID(GRBMH)]     = &GrbmhCounterBlockInfo;
  table[__BLOCK_ID_HSA(SPI)]   = &SpiCounterBlockInfo;
  table[__BLOCK_ID(SQG)]       = &SqgCounterBlockInfo;
  table[__BLOCK_ID(GC_UTCL1)]  = &GcUtcl1CounterBlockInfo;
  // WGP blocks
  table[__BLOCK_ID_HSA(SQ)]    = &SqcCounterBlockInfo;
  table[__BLOCK_ID_HSA(TCP)]   = &TcpCounterBlockInfo;

  block_table_ = table;
  block_count_ = LastCounterBlockId + 1;
}

}  // namespace aql_profile
