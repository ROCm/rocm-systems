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

#include "core/hardware_architecture.hpp"
#include "def/gpu_block_info.h"

namespace aql_profile {

size_t HardwareArchitecture::GetNumEventsForBlock(uint32_t block_id) const {
  const GpuBlockInfo* block_info = GetBlockInfo(block_id);
  if (!block_info) return 0;

  const auto& config = GetConfig();
  size_t se_number = config.GetSEPerXCC();
  size_t sa_number = config.sa_per_se_count;
  size_t block_samples_count = 1;

  // Multiply by shader engine count if block is per-SE
  if (block_info->attr & CounterBlockSeAttr)
    block_samples_count *= se_number;

  // Multiply by shader array count if block is per-SA
  if (block_info->attr & CounterBlockSaAttr)
    block_samples_count *= sa_number;

  // Multiply by WGP count if block is per-WGP
  if (block_info->attr & CounterBlockWgpAttr)
    block_samples_count *= GetNumWGPs();

  return block_samples_count;
}

size_t HardwareArchitecture::GetBytesNeededForBlock(uint32_t block_id) const {
  const auto& config = GetConfig();
  return GetNumEventsForBlock(block_id) * config.xcc_count * sizeof(uint64_t);
}

}  // namespace aql_profile
