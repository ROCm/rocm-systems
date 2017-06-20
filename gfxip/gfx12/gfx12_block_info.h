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


#ifndef _GFX12_BLOCKINFO_H_
#define _GFX12_BLOCKINFO_H_

namespace gfxip {
namespace gfx12 {
#define __BLOCK_ID(block) HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_##block
// Private PMC Counter BlockId is defined here
// Pubclic PMC Counter BlockId is defined in hsa_ven_amd_aqlprofile.h
enum CounterBlockId {
  __BLOCK_ID(RLC) = HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER,
  __BLOCK_ID(CPG),
  __BLOCK_ID(GRBMH),
  __BLOCK_ID(GRBMA),
  __BLOCK_ID(SQG),

  // mem blocks
  __BLOCK_ID(CHA),
  __BLOCK_ID(CHC),
  __BLOCK_ID(GLARBA),
  __BLOCK_ID(GLARBC),
  __BLOCK_ID(GC_CANE),
  __BLOCK_ID(GC_FFBM),
  __BLOCK_ID(GC_NHTTLB),
  __BLOCK_ID(GC_L2TLB),
  __BLOCK_ID(GC_UTCL1),
  __BLOCK_ID(GC_UTCL2),
  __BLOCK_ID(GC_VML2),

  __BLOCK_ID(GCEA_SE),

  // New SDMA Perfmon interface, comparing to the original SDMA PerfCnt. gfx12
  // supports both and they should provide the same counter events. We might
  // remove SDMA PerfCnt support in aqlprofile in the future since it is easier
  // to program Perfmon
  __BLOCK_ID(SDMA_PM),

  // Counters retrieved by KFD
  IommuV2CounterBlockId,
  KernelDriverCounterBlockId,

  CpPipeStatsCounterBlockId,
  HwInfoCounterBlockId,

  LastCounterBlockId = HwInfoCounterBlockId,
};

// Define SPM Counter BlockId
enum SpmGlobalBlockId {
  SPM_GLOBAL_BLOCK_NAME_FIRST = 0,
  SPM_GLOBAL_BLOCK_NAME_CPG = SPM_GLOBAL_BLOCK_NAME_FIRST,
  SPM_GLOBAL_BLOCK_NAME_CPC,
  SPM_GLOBAL_BLOCK_NAME_CPF,
  SPM_GLOBAL_BLOCK_NAME_GDS,
  SPM_GLOBAL_BLOCK_NAME_GCR,
  SPM_GLOBAL_BLOCK_NAME_PH,
  SPM_GLOBAL_BLOCK_NAME_GE1,
  SPM_GLOBAL_BLOCK_NAME_GL2A,
  SPM_GLOBAL_BLOCK_NAME_GL2C,
  SPM_GLOBAL_BLOCK_NAME_SDMA,
  SPM_GLOBAL_BLOCK_NAME_GUS,
  SPM_GLOBAL_BLOCK_NAME_EA,
  SPM_GLOBAL_BLOCK_NAME_CHA,
  SPM_GLOBAL_BLOCK_NAME_CHC,
  SPM_GLOBAL_BLOCK_NAME_CHCG,
  SPM_GLOBAL_BLOCK_NAME_ATCL2,
  SPM_GLOBAL_BLOCK_NAME_VML2,
  SPM_GLOBAL_BLOCK_NAME_GE2_SE,
  SPM_GLOBAL_BLOCK_NAME_GE2_DIST,
  SPM_GLOBAL_BLOCK_NAME_FFBM,
  SPM_GLOBAL_BLOCK_NAME_CANE,
  SPM_GLOBAL_BLOCK_NAME_LAST = SPM_GLOBAL_BLOCK_NAME_CANE,
};

enum SpmSeBlockId {
  SPM_SE_BLOCK_NAME_FIRST = 0,
  SPM_SE_BLOCK_NAME_CB = SPM_SE_BLOCK_NAME_FIRST,
  SPM_SE_BLOCK_NAME_DB,
  SPM_SE_BLOCK_NAME_PA,
  SPM_SE_BLOCK_NAME_SX,
  SPM_SE_BLOCK_NAME_SC,
  SPM_SE_BLOCK_NAME_TA,
  SPM_SE_BLOCK_NAME_TD,
  SPM_SE_BLOCK_NAME_TCP,
  SPM_SE_BLOCK_NAME_SPI,
  SPM_SE_BLOCK_NAME_SQG,
  SPM_SE_BLOCK_NAME_GL1A,
  SPM_SE_BLOCK_NAME_RMI,
  SPM_SE_BLOCK_NAME_GL1C,
  SPM_SE_BLOCK_NAME_GL1CG,
  SPM_SE_BLOCK_NAME_CBR,
  SPM_SE_BLOCK_NAME_DBR,
  SPM_SE_BLOCK_NAME_GL1H,
  SPM_SE_BLOCK_NAME_SQC,
  SPM_SE_BLOCK_NAME_PC,
  SPM_SE_BLOCK_NAME_EA,
  SPM_SE_BLOCK_NAME_GE,
  SPM_SE_BLOCK_NAME_GL2A,
  SPM_SE_BLOCK_NAME_GL2C,
  SPM_SE_BLOCK_NAME_WGS,
  SPM_SE_BLOCK_NAME_GL1XA,
  SPM_SE_BLOCK_NAME_GL1XC,
  SPM_SE_BLOCK_NAME_UTCL1,
  SPM_SE_BLOCK_NAME_LAST = SPM_SE_BLOCK_NAME_UTCL1,
};

namespace gfx1201 {
// IP versions for Radeon RX 9070
// ip_block : gc_12_0_1
// ip_block : athub_4_1_0
// ip_block : umc_8_14_0
// ip_block : df_4_15_1
// ip_block : pcie_6_1_0

// Number of block instances
// Reference: global_features.h (from gfxip header file package)
//            rspm_config.pm (from design configuration files)
// The following default values are generated from Radeon RX 9070, the first product of the
// RDNA 4 lineup. It could change for other products, and the change will be made in
// [PRODUCT_NAME]_factory.h
//
static const uint32_t GrbmCounterBlockNumInstances = 1;
static const uint32_t RlcCounterBlockNumInstances = 1;
static const uint32_t CpgCounterBlockNumInstances = 1;
static const uint32_t CpcCounterBlockNumInstances = 1;
static const uint32_t CpfCounterBlockNumInstances = 1;
static const uint32_t GcrCounterBlockNumInstances = 1;
static const uint32_t Ge1CounterBlockNumInstances = 1;
static const uint32_t Gl2aCounterBlockNumInstances = 4;   // GFX_CPWD__NUM_GL2A_PER_CPWD
static const uint32_t Gl2cCounterBlockNumInstances = 32;  // GFX_CPWD__NUM_GL2C_PER_CPWD
static const uint32_t GceaCounterBlockNumInstances = 36;  // GFX_CPWD__NUM_EA_PER_CPWD
static const uint32_t ChaCounterBlockNumInstances = 1;
static const uint32_t ChcCounterBlockNumInstances = 4;  // GFX_CPWD__NUM_CHC
static const uint32_t Ge2DistCounterBlockNumInstances = 1;
static const uint32_t SdmaCounterBlockNumInstances = 2;  // GFX_CPWD__NUM_SDMA_PER_CPWD
static const uint32_t GcVml2CounterBlockNumInstances = 1;
static const uint32_t GcMcVml2CounterBlockNumInstances = 1;
static const uint32_t GcUtcl2CounterBlockNumInstances = 1;
static const uint32_t GrbmhCounterBlockNumInstances = 1;
static const uint32_t CbCounterBlockNumInstances = 2;  // GFX_SE__NUM_RB_PER_SA
static const uint32_t DbCounterBlockNumInstances = 2;  // GFX_SE__NUM_RB_PER_SA
static const uint32_t SuCounterBlockNumInstances = 1;  // GFX_SE__NUM_PA_PER_SE
static const uint32_t SxCounterBlockNumInstances = 1;
static const uint32_t ScCounterBlockNumInstances = 2;   // GFX_SE__NUM_PACKER_PER_SA
static const uint32_t TaCounterBlockNumInstances = 2;   // GFX_SE__NUM_ROWS_PER_WGP
static const uint32_t TdCounterBlockNumInstances = 2;   // GFX_SE__NUM_ROWS_PER_WGP
static const uint32_t TcpCounterBlockNumInstances = 2;  // GFX_SE__NUM_ROWS_PER_WGP
static const uint32_t SpiCounterBlockNumInstances = 1;
static const uint32_t SqgCounterBlockNumInstances = 1;
static const uint32_t Gl1aCounterBlockNumInstances = 1;
static const uint32_t RmiCounterBlockNumInstances = 2;   // GFX_SE__NUM_RMI_PER_SA
static const uint32_t Gl1cCounterBlockNumInstances = 4;  // GFX_SE__NUM_GL1C_PER_SA
static const uint32_t SqcCounterBlockNumInstances = 1;
static const uint32_t PcCounterBlockNumInstances = 1;
static const uint32_t GceaSeCounterBlockNumInstances = 4;
static const uint32_t GeCounterBlockNumInstances = 1;
static const uint32_t WgsCounterBlockNumInstances = 1;
static const uint32_t Gl1xaCounterBlockNumInstances = 1;
static const uint32_t Gl1xcCounterBlockNumInstances = 4;  // GFX_SE__NUM_GL1C_PER_SA
static const uint32_t GcUtcl1CounterBlockNumInstances = 2;

static const uint32_t SdmaCounterBlockMaxInstances = 8;
static const uint32_t UmcCounterBlockMaxInstances = 32;

// Number of block counter registers - Auto-generated from chip_offset_byte.h, edit with extra
// caution Reference: chip_offset_byte.h (from gfxip header file package) The following default
// values are generated from Radeon RX 9070, the first product of the RDNA 4 lineup. It could change
// for other products, and the change will be made in [PRODUCT_NAME]_factory.h
//
static const uint32_t GrbmCounterBlockNumCounters = 2;
static const uint32_t RlcCounterBlockNumCounters = 2;
static const uint32_t CpgCounterBlockNumCounters = 2;
static const uint32_t CpcCounterBlockNumCounters = 2;
static const uint32_t CpfCounterBlockNumCounters = 2;
static const uint32_t GcrCounterBlockNumCounters = 2;
static const uint32_t PhCounterBlockNumCounters = 8;
static const uint32_t Ge1CounterBlockNumCounters = 4;
static const uint32_t Gl2aCounterBlockNumCounters = 4;
static const uint32_t Gl2cCounterBlockNumCounters = 4;
static const uint32_t GceaCounterBlockNumCounters = 2;
static const uint32_t ChaCounterBlockNumCounters = 4;
static const uint32_t ChcCounterBlockNumCounters = 4;
static const uint32_t Ge2DistCounterBlockNumCounters = 4;
static const uint32_t SdmaCounterBlockNumCounters = 2;
static const uint32_t GcVml2CounterBlockNumCounters = 2;
static const uint32_t GcMcVml2CounterBlockNumCounters = 1;
static const uint32_t GcUtcl2CounterBlockNumCounters = 1;
static const uint32_t GrbmhCounterBlockNumCounters = 2;
static const uint32_t CbCounterBlockNumCounters = 4;
static const uint32_t DbCounterBlockNumCounters = 4;
static const uint32_t SuCounterBlockNumCounters = 4;
static const uint32_t SxCounterBlockNumCounters = 4;
static const uint32_t PaScCounterBlockNumCounters = 8;
static const uint32_t TaCounterBlockNumCounters = 2;
static const uint32_t TdCounterBlockNumCounters = 2;
static const uint32_t TcpCounterBlockNumCounters = 4;
static const uint32_t SpiCounterBlockNumCounters = 6;
static const uint32_t SqgCounterBlockNumCounters = 8;
static const uint32_t Gl1aCounterBlockNumCounters = 4;
static const uint32_t RmiCounterBlockNumCounters = 4;
static const uint32_t Gl1cCounterBlockNumCounters = 4;
static const uint32_t SqcCounterBlockNumCounters = 16;
static const uint32_t PcCounterBlockNumCounters = 4;
static const uint32_t GceaSeCounterBlockNumCounters = 2;
static const uint32_t GeCounterBlockNumCounters = 4;
static const uint32_t WgsCounterBlockNumCounters = 2;
static const uint32_t Gl1xaCounterBlockNumCounters = 4;
static const uint32_t Gl1xcCounterBlockNumCounters = 4;
static const uint32_t GcUtcl1CounterBlockNumCounters = 4;

// Block counters max event value - Auto-generated from chip_enum.h, edit with extra caution
// Reference: chip_enum.h (from gfxip header file package)
// The following default values are generated from Radeon RX 9070, the first product of the
// RDNA 4 lineup. It could change for other products, and the change will be made in
// [PRODUCT_NAME]_factory.h
//
static const uint32_t GrbmCounterBlockMaxEvent = 51;
static const uint32_t RlcCounterBlockMaxEvent = 6;
static const uint32_t CpgCounterBlockMaxEvent = 30;
static const uint32_t CpcCounterBlockMaxEvent = 55;
static const uint32_t CpfCounterBlockMaxEvent = 4;
static const uint32_t GcrCounterBlockMaxEvent = 151;
static const uint32_t PhCounterBlockMaxEvent = 1023;
static const uint32_t Ge1CounterBlockMaxEvent = 54;
static const uint32_t Gl2aCounterBlockMaxEvent = 114;
static const uint32_t Gl2cCounterBlockMaxEvent = 249;
static const uint32_t GceaCounterBlockMaxEvent = 32;
static const uint32_t ChaCounterBlockMaxEvent = 25;
static const uint32_t ChcCounterBlockMaxEvent = 94;
static const uint32_t Ge2DistCounterBlockMaxEvent = 188;
static const uint32_t SdmaCounterBlockMaxEvent = 125;
static const uint32_t GcVml2CounterBlockMaxEvent = 90;
static const uint32_t GcMcVml2CounterBlockMaxEvent =
    1;  // This is handled by GCMC_VM_L2_PERFCOUNTER0_CFG
static const uint32_t GcUtcl2CounterBlockMaxEvent = 36;
static const uint32_t GrbmhCounterBlockMaxEvent = 25;
static const uint32_t CbCounterBlockMaxEvent = 315;
static const uint32_t DbCounterBlockMaxEvent = 441;
static const uint32_t PaSuCounterBlockMaxEvent = 828;
static const uint32_t SxCounterBlockMaxEvent = 81;
static const uint32_t ScCounterBlockMaxEvent = 821;
static const uint32_t TaCounterBlockMaxEvent = 254;
static const uint32_t TdCounterBlockMaxEvent = 271;
static const uint32_t TcpCounterBlockMaxEvent = 99;
static const uint32_t SpiCounterBlockMaxEvent = 318;
static const uint32_t SqgCounterBlockMaxEvent = 45;
static const uint32_t Gl1aCounterBlockMaxEvent = 21;
static const uint32_t RmiCounterBlockMaxEvent = 138;
static const uint32_t Gl1cCounterBlockMaxEvent = 121;
static const uint32_t SqcCounterBlockMaxEvent = 511;
static const uint32_t PcCounterBlockMaxEvent = 164;
static const uint32_t GceaSeCounterBlockMaxEvent = 32;
static const uint32_t GeCounterBlockMaxEvent = 103;
static const uint32_t WgsCounterBlockMaxEvent = 4;
static const uint32_t Gl1xaCounterBlockMaxEvent = 21;
static const uint32_t Gl1xcCounterBlockMaxEvent = 109;
static const uint32_t GcUtcl1CounterBlockMaxEvent = 71;
}  // namespace gfx1201

}  // namespace gfx12
}  // namespace gfxip

#endif  //  _GFX12_BLOCKINFO_H_
