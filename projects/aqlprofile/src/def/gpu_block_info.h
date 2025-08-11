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

#ifndef SRC_DEF_GPU_BLOCK_INFO_H_
#define SRC_DEF_GPU_BLOCK_INFO_H_

#include <stdint.h>
#include "util/reg_offsets.h"

// Counter Block attributes
enum CounterBlockAttr {
  // Default block attribute
  CounterBlockDfltAttr = 1,
  // Per ShaderEngine blocks
  CounterBlockSeAttr = 2,
  // SQ blocks
  CounterBlockSqAttr = 4,
  // Need to clean counter registers
  CounterBlockCleanAttr = 8,
  // MC Block
  CounterBlockMcAttr = 0x10,
  // CP PERFMON controllable blocks
  CounterBlockCpmonAttr = 0x1f,
  // SDMA block
  CounterBlockSdmaAttr = 0x100,
  // Texture cache
  CounterBlockTcAttr = 0x400,
  // Explicitly indexed blocks
  CounterBlockExplInstAttr = 0x800,
  // SPM blocks
  CounterBlockSpmGlobalAttr = 0x1000,
  CounterBlockSpmSeAttr = 0x2000,
  // GUS block
  CounterBlockGusAttr = 0x4000,
  // GRBM block
  CounterBlockGRBMAttr = 0x8000,
  // UMC block
  CounterBlockUmcAttr = 0x10000,
  // SE and SA-dependent blocks
  CounterBlockSaAttr = 0x20000,
  // MI300 AID blocks
  CounterBlockAidAttr = 0x40000,
  // SPI special
  CounterBlockSPIAttr = 0x80000,
  // Blocks within WGP
  CounterBlockWgpAttr = 0x100000,
  // RPB block
  CounterBlockRpbAttr = 0x200000,
  // ATC block
  CounterBlockAtcAttr = 0x400000,
};


// Register address corresponding to each counter
struct CounterRegInfo {
  // counter select register address
  Register select_addr;
  // counter control register address
  Register control_addr;
  // counter register address low
  Register register_addr_lo;
  // counter register address high
  Register register_addr_hi;
};

struct BlockDelayInfo {
  Register reg;
  uint32_t val;
};

struct counter_des_t;

// GPU Block info definition
struct GpuBlockInfo {
  // Unique string identifier of the block.
  const char* name;
  // Block ID
  uint32_t id;
  // Maximum number of block instances in the group per shader array
  uint32_t instance_count;
  // Maximum counter event ID
  uint32_t event_id_max;
  // Maximum number of counters that can be enabled at once
  uint32_t counter_count;
  // Counter registers addresses
  const CounterRegInfo* counter_reg_info;
  // Counter select value function
  uint32_t (*select_value)(const counter_des_t&);
  // Block attributes mask
  uint32_t attr;
  // Block delay info
  const BlockDelayInfo* delay_info;
  // SPM block id
  uint32_t spm_block_id;
};

// Block descriptor
struct block_des_t {
  uint32_t id;
  uint32_t index;
};

// block_des_t less then functor
struct lt_block_des {
  bool operator()(const block_des_t& a1, const block_des_t& a2) const {
    return (a1.id < a2.id) || ((a1.id == a2.id) && (a1.index < a2.index));
  }
};

// Counter descriptor
struct counter_des_t {
  uint32_t id;
  uint32_t index;
  block_des_t block_des;
  const GpuBlockInfo* block_info;
};


#endif  // SRC_DEF_GPU_BLOCK_INFO_H_
