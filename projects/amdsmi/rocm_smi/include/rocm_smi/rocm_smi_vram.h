// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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

// Sum only the public-framebuffer banks (HSA_HEAPTYPE_FB_PUBLIC), which are the
// user-visible VRAM. Falls back to summing every bank when no bank is FB_PUBLIC:
// amdkfd reports the whole framebuffer as a single FB_PRIVATE bank on small-BAR
// GPUs, and as FB_PRIVATE on APUs, so the fallback is what keeps those nodes
// from reporting zero.
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
