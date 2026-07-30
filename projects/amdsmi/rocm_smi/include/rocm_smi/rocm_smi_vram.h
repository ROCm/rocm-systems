/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef INCLUDE_ROCM_SMI_ROCM_SMI_VRAM_H_
#define INCLUDE_ROCM_SMI_ROCM_SMI_VRAM_H_

#include <cstdint>
#include <string>
#include <vector>

namespace amd::smi {

// A single KFD memory bank parsed from
// /sys/class/kfd/kfd/topology/nodes/<n>/mem_banks/<b>/properties.
struct KfdMemBank {
  uint32_t heap_type;
  uint64_t size_in_bytes;
};

// Sum only the public-framebuffer banks (HSA_HEAPTYPE_FB_PUBLIC). Private,
// GDS/LDS, and scratch heaps are not user-visible VRAM and would over-report the
// total on discrete GPUs that enumerate them. Falls back to summing every bank
// when none is FB_PUBLIC, so a node that exposes its framebuffer under another
// heap type still yields a non-zero total.
uint64_t sum_public_vram_bytes(const std::vector<KfdMemBank>& banks);

// Decide whether the KFD topology total (mem_banks) should override the sysfs
// mem_info_vram_total when reporting RSMI_MEM_TYPE_VRAM; returns true when the
// KFD value should win. This happens in three cases: sysfs is unusable (0 or a
// read failure, e.g. MI300A with no node); a multi-partition mode
// (CPX/DPX/TPX/QPX), where sysfs splits the whole device and ignores reserved
// memory; and APUs (e.g. gfx1151) that report only the small BIOS carveout
// instead of the unified pool.
bool vram_total_prefer_kfd(bool sysfs_read_ok, uint64_t sysfs_total,
                           const std::string& compute_partition, uint64_t kfd_total);

}  // namespace amd::smi

#endif  // INCLUDE_ROCM_SMI_ROCM_SMI_VRAM_H_
