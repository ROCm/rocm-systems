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

// GFX11.5x (gfx1150, gfx1151) architecture implementation.
// Inherits GFX11 builders and config; overrides block instance counts that
// differ between RDNA 3 and RDNA 3.5.
#include "core/architectures/gfx11_architecture.hpp"
#include "aqlprofile-sdk/aql_profile_v2.h"
#include "def/gfx11_def.h"

#include <mutex>

namespace aql_profile {

Gfx115xArchitecture::Gfx115xArchitecture(const AgentInfo* agent_info)
    : Gfx11Architecture(agent_info) {
  // Re-initialize block table with gfx11.5x-specific instance counts.
  // The base constructor already called InitializeBlockTable() for plain GFX11,
  // so we override it here (same pattern as Mi450Architecture).
  InitializeBlockTable();
}

void Gfx115xArchitecture::InitializeBlockTable() {
  // Static table of block pointers shared across all GFX11.5x instances.
  static const GpuBlockInfo* table[AQLPROFILE_BLOCKS_NUMBER]{};
  static std::once_flag init_flag;

  // block_table_ was set by Gfx11Architecture::InitializeBlockTable() during
  // the base class constructor. Capture it before call_once; it always points
  // to the same GFX11 static table so this is safe across all instances.
  const GpuBlockInfo** gfx11_table = block_table_;

  std::call_once(init_flag, [gfx11_table]() {
    // Start from the GFX11 base table.
    for (unsigned i = 0; i < AQLPROFILE_BLOCKS_NUMBER; ++i)
      table[i] = gfx11_table[i];

    // Patch instance counts that differ on gfx11.5x hardware.
    auto patch = [&](unsigned id, uint32_t instance_count) {
      if (id < AQLPROFILE_BLOCKS_NUMBER && gfx11_table[id]) {
        GpuBlockInfo* copy = new GpuBlockInfo(*gfx11_table[id]);
        copy->instance_count = instance_count;
        table[id] = copy;
      }
    };

    patch(Gl1aCounterBlockId, 4);
    patch(Gl1cCounterBlockId, 4);
    patch(Gl2aCounterBlockId, 4);
    patch(Gl2cCounterBlockId, 8);
    patch(TcpCounterBlockId,  2);
    patch(TaCounterBlockId,   2);
    patch(TdCounterBlockId,   2);
  });

  block_table_ = table;
  block_count_ = AQLPROFILE_BLOCKS_NUMBER;
}

}  // namespace aql_profile
