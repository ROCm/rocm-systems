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

#include "core/gfx11_factory.h"
#include "def/gfx11_def.h"

namespace aql_profile {

// Gfx11.5 factory class
const GpuBlockInfo* Gfx115xFactory::block_table_[AQLPROFILE_BLOCKS_NUMBER] = {};

Gfx115xFactory::Gfx115xFactory(const AgentInfo* agent_info)
    : Gfx11Factory(block_table_, sizeof(block_table_), agent_info) {
  for (unsigned i = 0; i < AQLPROFILE_BLOCKS_NUMBER; ++i) {
    const GpuBlockInfo* base_table_ptr = Gfx11Factory::block_table_[i];
    if (base_table_ptr == NULL) continue;
    GpuBlockInfo* block_info = new GpuBlockInfo(*base_table_ptr);
    block_table_[i] = block_info;

    // Overwrite block info for gfx11.5 specific changes
    switch (block_info->id) {
      case Gl1aCounterBlockId:
        block_info->instance_count = 4;
        break;
      case Gl1cCounterBlockId:
        block_info->instance_count = 4;
        break;
      case Gl2aCounterBlockId:
        block_info->instance_count = 4;
        break;
      case Gl2cCounterBlockId:
        block_info->instance_count = 4;
        break;
      case TcpCounterBlockId:
        block_info->instance_count = 2;
        break;
      case TaCounterBlockId:
        block_info->instance_count = 2;
        break;
      case TdCounterBlockId:
        block_info->instance_count = 2;
        break;
      default:
        break;
    }
  }
}

Gfx115xFactory::~Gfx115xFactory() {
  for (unsigned i = 0; i < AQLPROFILE_BLOCKS_NUMBER; ++i) {
    if (block_table_[i] != NULL) {
      delete block_table_[i];
      block_table_[i] = NULL;
    }
  }
}

Pm4Factory* Pm4Factory::Gfx115xCreate(const AgentInfo* agent_info) {
  auto p = new Gfx115xFactory(agent_info);
  if (p == NULL) throw aql_profile_exc_msg("Gfx115Factory allocation failed");
  return p;
}

}  // namespace aql_profile
