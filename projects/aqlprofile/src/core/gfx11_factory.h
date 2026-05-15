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

#ifndef _GFX11_FACTORY_H_
#define _GFX11_FACTORY_H_
#include "core/pm4_factory.h"

namespace aql_profile {

// Gfx11 factory class
class Gfx11Factory : public Pm4Factory {
 public:
  explicit Gfx11Factory(const AgentInfo* agent_info)
      : Pm4Factory(BlockInfoMap(block_table_, sizeof(block_table_))) {
    Init(agent_info);
  }
  Gfx11Factory(const GpuBlockInfo** table, const uint32_t& size, const AgentInfo* agent_info)
      : Pm4Factory(BlockInfoMap(table, size)) {
    Init(agent_info);
  }
  bool IsGFX11() const override { return true; }

  virtual int GetAccumLowID() const override { return 1; }
  virtual int GetAccumHiID() const override { return 1; }

 protected:
  void Init(const AgentInfo* agent_info);
  static const GpuBlockInfo* block_table_[AQLPROFILE_BLOCKS_NUMBER];
};

// Gfx11.5 factory class
class Gfx115xFactory : public Gfx11Factory {
 public:
  explicit Gfx115xFactory(const AgentInfo* agent_info);
  virtual ~Gfx115xFactory();

 protected:
  static const GpuBlockInfo* block_table_[AQLPROFILE_BLOCKS_NUMBER];
};

}  // namespace aql_profile

#endif  // _GFX11_FACTORY_H_
