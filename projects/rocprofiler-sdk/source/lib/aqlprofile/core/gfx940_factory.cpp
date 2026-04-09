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

#include "core/gfx9_factory.h"
#include "def/gfx940_def.h"
#include "pm4/gfx9_cmd_builder.h"
#include "pm4/pmc_builder.h"
#include "pm4/sqtt_builder.h"

namespace aql_profile {

class Mi300Factory : public Mi100Factory {
 public:
  explicit Mi300Factory(const AgentInfo* agent_info, gpu_id_t gpu_id = MI300_GPU_ID)
      : Mi100Factory(agent_info) {
    InitSpmBlockDelayTable(gpu_id);
    for (unsigned blockname_id = 0; blockname_id < AQLPROFILE_BLOCKS_NUMBER;
         ++blockname_id) {
      const GpuBlockInfo* base_table_ptr = Gfx9Factory::block_table_[blockname_id];
      if (base_table_ptr == NULL) continue;
      GpuBlockInfo* block_info = nullptr;
      if (blockname_id == HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_RPB)
        block_info = new GpuBlockInfo(RpbCounterBlockInfo);
      else if (blockname_id == HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATC)
        block_info = new GpuBlockInfo(AtcCounterBlockInfo);
      else
        block_info = new GpuBlockInfo(*base_table_ptr);
      block_table_[blockname_id] = block_info;
      // overwrite block info for any update from gfx9 to mi300
      InitSpmBlockDelay(block_info);
      switch (block_info->id) {
        case SqCounterBlockId:
          block_info->event_id_max = 373;
          break;
        case TcpCounterBlockId:
          block_info->event_id_max = 84;
          assert(agent_info->se_num * block_info->instance_count == cu_block_delay_table_size);
          break;
        case TccCounterBlockId:
          block_info->instance_count = 16;
          block_info->event_id_max = 199;
          break;
        case TcaCounterBlockId:
          block_info->instance_count = 32;
          block_info->event_id_max = 34;
          break;
        case GceaCounterBlockId:
          block_info->instance_count = 32;
          block_info->event_id_max = 82;
          break;
        case SdmaCounterBlockId:
          block_info->instance_count = 4 * pm4_builder::MAX_AID;
          break;
        case UmcCounterBlockId:
          block_info->counter_count = 11;
          block_info->instance_count = 32 * pm4_builder::MAX_AID;
          break;
        case RpbCounterBlockId:
          block_info->instance_count = 4;
          break;
        case AtcCounterBlockId:
          block_info->instance_count = 4;
          break;
      }
    }
  }

  virtual int GetAccumLowID() const override { return 1; };
  virtual int GetAccumHiID() const override { return 184; };
  virtual uint32_t GetSpmSampleDelayMax() { return 0x27; };

 private:
  void InitSpmBlockDelayTable(gpu_id_t gpu_id);
};

namespace gfx940 {
static const uint32_t CpgBlockDelayValue[] = {0x21};
static const uint32_t CpcBlockDelayValue[] = {0x1f};
static const uint32_t CpfBlockDelayValue[] = {0x23};
static const uint32_t GdsBlockDelayValue[] = {0x23};
static const uint32_t TccBlockDelayValue[] = {0x0f, 0x0f, 0x0c, 0x0e, 0x0e, 0x13, 0x13, 0x19,
                                              0x13, 0x13, 0x12, 0x13, 0x13, 0x17, 0x17, 0x1d};
static const uint32_t TcaBlockDelayValue[] = {0x14, 0x18};
static const uint32_t SxBlockDelayValue[] = {0x00, 0x03, 0x07, 0x03};
static const uint32_t TaBlockDelayValue[] = {
  0x17, 0x15, 0x13, 0x11, 0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0, 0, 0, 0, 0, 0,  // se0
  0x18, 0x16, 0x14, 0x12, 0x10, 0x0e, 0x0c, 0x0a, 0x08, 0x06, 0, 0, 0, 0, 0, 0,  // se1
  0x1c, 0x1a, 0x18, 0x16, 0x14, 0x12, 0x10, 0x0e, 0x0c, 0x0a, 0, 0, 0, 0, 0, 0,  // se2
  0x1a, 0x18, 0x16, 0x14, 0x12, 0x10, 0x0e, 0x0c, 0x0a, 0x08, 0, 0, 0, 0, 0, 0}; // se3
static const uint32_t SpiBlockDelayValue[] = {0x10, 0x19, 0x1d, 0x13};
static const uint32_t SqBlockDelayValue[] = {0x10, 0x1d, 0x21, 0x12};
} // namespace gfx940

namespace gfx950 {
static const uint32_t CpgBlockDelayValue[] = {0x33};
static const uint32_t CpcBlockDelayValue[] = {0x31};
static const uint32_t CpfBlockDelayValue[] = {0x33};
static const uint32_t GdsBlockDelayValue[] = {0x2f};
static const uint32_t TccBlockDelayValue[] = {0x21, 0x23, 0x27, 0x22, 0x23, 0x25, 0x27, 0x29,
                                              0x24, 0x25, 0x29, 0x25, 0x27, 0x27, 0x29, 0x2b};
static const uint32_t TcaBlockDelayValue[] = {0x2b, 0x2d};
static const uint32_t SxBlockDelayValue[] = {0x00, 0x04, 0x07, 0x01};
static const uint32_t TaBlockDelayValue[] = {
  0x29, 0x25, 0x21, 0x1d, 0x19, 0x15, 0x11, 0x0d, 0x09, 0, 0, 0, 0, 0, 0, 0,  // se0
  0x2a, 0x26, 0x22, 0x1e, 0x1a, 0x16, 0x12, 0x0e, 0x0a, 0, 0, 0, 0, 0, 0, 0,  // se1
  0x2b, 0x28, 0x24, 0x20, 0x1c, 0x18, 0x14, 0x10, 0x0c, 0, 0, 0, 0, 0, 0, 0,  // se2
  0x2a, 0x26, 0x22, 0x1e, 0x1a, 0x16, 0x12, 0x0e, 0x0a, 0, 0, 0, 0, 0, 0, 0}; // se3
static const uint32_t TdBlockDelayValue[] = {
  0x29, 0x25, 0x21, 0x1d, 0x19, 0x15, 0x11, 0x0d, 0x09, 0, 0, 0, 0, 0, 0, 0,  // se0
  0x2a, 0x26, 0x22, 0x1e, 0x1a, 0x16, 0x12, 0x0e, 0x0a, 0, 0, 0, 0, 0, 0, 0,  // se1
  0x2b, 0x28, 0x24, 0x20, 0x1c, 0x18, 0x14, 0x10, 0x0c, 0, 0, 0, 0, 0, 0, 0,  // se2
  0x2a, 0x26, 0x22, 0x1e, 0x1a, 0x16, 0x12, 0x0e, 0x0a, 0, 0, 0, 0, 0, 0, 0}; // se3
static const uint32_t TcpBlockDelayValue[] = {
  0x29, 0x25, 0x21, 0x1d, 0x19, 0x15, 0x11, 0x0d, 0x09, 0, 0, 0, 0, 0, 0, 0,  // se0
  0x2a, 0x26, 0x22, 0x1e, 0x1a, 0x16, 0x12, 0x0e, 0x0a, 0, 0, 0, 0, 0, 0, 0,  // se1
  0x2a, 0x28, 0x24, 0x20, 0x1c, 0x18, 0x14, 0x10, 0x0c, 0, 0, 0, 0, 0, 0, 0,  // se2
  0x2a, 0x27, 0x23, 0x1f, 0x1b, 0x17, 0x13, 0x0f, 0x0b, 0, 0, 0, 0, 0, 0, 0}; // se3
static const uint32_t SpiBlockDelayValue[] = {0x25, 0x2d, 0x2f, 0x2b};
static const uint32_t SqBlockDelayValue[] = {0x25, 0x2d, 0x2f, 0x2b};
} // namespace gfx950

void Mi300Factory::InitSpmBlockDelayTable(gpu_id_t gpu_id) {
  const uint32_t** p;
  if (gpu_id == MI300_GPU_ID) {
    cu_block_delay_table_size = sizeof(gfx940::TaBlockDelayValue) / sizeof(gfx940::TaBlockDelayValue[0]);
    // Global Blocks
    p = spm_block_delay_global;
    *p++ = gfx940::CpgBlockDelayValue;  // CPG = 0
    *p++ = gfx940::CpcBlockDelayValue;  // CPC = 1
    *p++ = gfx940::CpfBlockDelayValue;  // CPF = 2
    *p++ = gfx940::GdsBlockDelayValue;  // GDS = 3
    *p++ = gfx940::TccBlockDelayValue;  // TCC = 4
    *p++ = gfx940::TcaBlockDelayValue;  // TCA = 5
    *p++ = NULL;                        // IA = 6
    *p++ = NULL;                        // TCS = 7
    // SE Blocks
    p = spm_block_delay_se;
    *p++ = NULL;                        // CB = 0
    *p++ = NULL;                        // DB = 1
    *p++ = NULL;                        // PA = 2
    *p++ = gfx940::SxBlockDelayValue;   // SSX = 3
    *p++ = NULL;                        // SC = 4
    *p++ = gfx940::TaBlockDelayValue;   // TA = 5
    *p++ = gfx940::TaBlockDelayValue;   // TD = 6  - Same as TA
    *p++ = gfx940::TaBlockDelayValue;   // TCP = 7 - Same as TA
    *p++ = gfx940::SpiBlockDelayValue;  // SPI = 8
    *p++ = gfx940::SqBlockDelayValue;   // SQG = 9
    *p++ = NULL;                        // VGT = 10
  } else if (gpu_id == MI350_GPU_ID) {
    cu_block_delay_table_size = sizeof(gfx950::TaBlockDelayValue) / sizeof(gfx950::TaBlockDelayValue[0]);
    // Global Blocks
    p = spm_block_delay_global;
    *p++ = gfx950::CpgBlockDelayValue;  // CPG = 0
    *p++ = gfx950::CpcBlockDelayValue;  // CPC = 1
    *p++ = gfx950::CpfBlockDelayValue;  // CPF = 2
    *p++ = gfx950::GdsBlockDelayValue;  // GDS = 3
    *p++ = gfx950::TccBlockDelayValue;  // TCC = 4
    *p++ = gfx950::TcaBlockDelayValue;  // TCA = 5
    *p++ = NULL;                        // IA = 6
    *p++ = NULL;                        // TCS = 7
    // SE Blocks
    p = spm_block_delay_se;
    *p++ = NULL;                        // CB = 0
    *p++ = NULL;                        // DB = 1
    *p++ = NULL;                        // PA = 2
    *p++ = gfx950::SxBlockDelayValue;   // SSX = 3
    *p++ = NULL;                        // SC = 4
    *p++ = gfx950::TaBlockDelayValue;   // TA = 5
    *p++ = gfx950::TdBlockDelayValue;   // TD = 6
    *p++ = gfx950::TcpBlockDelayValue;  // TCP = 7
    *p++ = gfx950::SpiBlockDelayValue;  // SPI = 8
    *p++ = gfx950::SqBlockDelayValue;   // SQG = 9
    *p++ = NULL;                        // VGT = 10
  }
}

Pm4Factory* Pm4Factory::Mi300Create(const AgentInfo* agent_info) {
  auto p = new Mi300Factory(agent_info);
  if (p == NULL) throw aql_profile_exc_msg("Mi300Factory allocation failed");
  return p;
}

class Mi350Factory : public Mi300Factory {
 public:
  // MI350 is a copy of Mi300
  explicit Mi350Factory(const AgentInfo* agent_info) : Mi300Factory(agent_info, MI350_GPU_ID) {}

  virtual int GetAccumLowID() const override { return 1; };
  virtual int GetAccumHiID() const override { return 200; };
  virtual uint32_t GetSpmSampleDelayMax() { return 0x33; };
};

Pm4Factory* Pm4Factory::Mi350Create(const AgentInfo* agent_info) {
  auto p = new Mi350Factory(agent_info);
  if (p == NULL) throw aql_profile_exc_msg("Mi350Factory allocation failed");
  return p;
}

}  // namespace aql_profile
