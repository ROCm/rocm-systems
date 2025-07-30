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

#ifndef _GFX9_FACTORY_H_
#define _GFX9_FACTORY_H_
#include "core/pm4_factory.h"

namespace aql_profile {

// Gfx9 factory class
class Gfx9Factory : public Pm4Factory {
 public:
  explicit Gfx9Factory(const AgentInfo* agent_info)
      : Pm4Factory(BlockInfoMap(block_table_, sizeof(block_table_))) {
    Init(agent_info);
  }
  Gfx9Factory(const GpuBlockInfo** table, const uint32_t& size, const AgentInfo* agent_info)
      : Pm4Factory(BlockInfoMap(table, size)) {
    Init(agent_info);
  }

  bool IsGFX9() const override { return true; }

 protected:
  void Init(const AgentInfo* agent_info);
  static const GpuBlockInfo* block_table_[AQLPROFILE_BLOCKS_NUMBER];

  static void Print(const GpuBlockInfo* block_info);
};

// Mi100 factory class
class Mi100Factory : public Gfx9Factory {
 public:
  explicit Mi100Factory(const AgentInfo* agent_info);
 
 virtual int GetAccumLowID() const override { return 1; }
 
 virtual int GetAccumHiID() const override { return 158; }

 virtual uint32_t GetSpmSampleDelayMax() { return 0x34; }

 protected:
  static const GpuBlockInfo* block_table_[AQLPROFILE_BLOCKS_NUMBER];
};

}  // namespace aql_profile

#endif  // _GFX9_FACTORY_H_
