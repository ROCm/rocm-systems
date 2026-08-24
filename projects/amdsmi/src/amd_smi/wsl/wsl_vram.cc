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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 */

#include "amd_smi/impl/wsl/wsl_vram.h"

#include <ntstatus.h>

#include <sstream>

#include "amd_smi/impl/wsl/dxcore_loader.h"
#include "rocm_smi/rocm_smi_logger.h"

namespace wsl {

namespace {

bool FindSegmentId(const WslAdapterInfo& info, SegmentKind kind, uint32_t* segment_id) {
  for (const auto& seg : info.segments) {
    if (seg.kind == kind) {
      *segment_id = seg.segment_id;
      return true;
    }
  }
  return false;
}

bool QuerySegmentBytesResident(const WslAdapterInfo& info, uint32_t segment_id,
                               uint64_t* bytes_resident) {
  D3DKMT_QUERYSTATISTICS stats = {};
  stats.Type = D3DKMT_QUERYSTATISTICS_SEGMENT;
  stats.AdapterLuid = info.adapter_luid;
  stats.QuerySegment.SegmentId = segment_id;

  NTSTATUS status = DXCORE_CALL(D3DKMTQueryStatistics)(&stats);
  if (status != STATUS_SUCCESS) {
    std::ostringstream ss;
    ss << "[wsl_vram] D3DKMTQueryStatistics(SEGMENT " << segment_id << ") failed: 0x" << std::hex
       << status;
    LOG_ERROR(ss);
    return false;
  }

  *bytes_resident = stats.QueryResult.SegmentInformation.BytesResident;
  return true;
}

bool QuerySegmentGroupUsage(const WslAdapterInfo& info, uint32_t segment_group,
                            uint64_t* bytes_allocated) {
  // Only supported by drivers reporting WDDM 3.1+; older drivers reject the query type.
  if (info.wddm_version < KMT_DRIVERVERSION_WDDM_3_1) return false;

  D3DKMT_QUERYSTATISTICS stats = {};
  stats.Type = D3DKMT_QUERYSTATISTICS_SEGMENT_GROUP_USAGE;
  stats.AdapterLuid = info.adapter_luid;
  stats.QuerySegmentGroupUsage.PhysicalAdapterIndex = 0;
  stats.QuerySegmentGroupUsage.SegmentGroup = static_cast<UINT16>(segment_group);

  NTSTATUS status = DXCORE_CALL(D3DKMTQueryStatistics)(&stats);
  if (status != STATUS_SUCCESS) return false;

  *bytes_allocated = stats.QueryResult.SegmentGroupUsageInformation.AllocatedBytes;
  return true;
}

bool QueryLocalVramUsage(const WslAdapterInfo& info, uint64_t* usage_bytes) {
  *usage_bytes = 0;

  if (QuerySegmentGroupUsage(info, D3DKMT_MEMORY_SEGMENT_GROUP_LOCAL, usage_bytes)) return true;

  uint32_t visible_segment_id = 0;
  if (!FindSegmentId(info, SegmentKind::kLocalMemory, &visible_segment_id)) {
    std::ostringstream ss;
    ss << "[wsl_vram] No local memory segment found";
    LOG_ERROR(ss);
    return false;
  }

  if (!QuerySegmentBytesResident(info, visible_segment_id, usage_bytes)) return false;

  if (info.device_info.local_invisible_heap_size == 0) return true;

  uint32_t invisible_segment_id = 0;
  bool found_invisible = false;
  for (const auto& seg : info.segments) {
    if (seg.kind == SegmentKind::kLocalMemory && seg.segment_id > visible_segment_id) {
      invisible_segment_id = seg.segment_id;
      found_invisible = true;
      break;
    }
  }
  if (!found_invisible) {
    std::ostringstream ss;
    ss << "[wsl_vram] No invisible local memory segment found";
    LOG_ERROR(ss);
    return false;
  }

  uint64_t invisible_usage = 0;
  if (!QuerySegmentBytesResident(info, invisible_segment_id, &invisible_usage)) return false;

  *usage_bytes += invisible_usage;
  return true;
}

bool QueryNonLocalVramUsage(const WslAdapterInfo& info, uint64_t* usage_bytes) {
  *usage_bytes = 0;

  if (QuerySegmentGroupUsage(info, D3DKMT_MEMORY_SEGMENT_GROUP_NON_LOCAL, usage_bytes)) return true;

  bool found_segment = false;
  for (const auto& seg : info.segments) {
    if (!seg.is_aperture || !seg.is_system_memory) continue;

    found_segment = true;
    uint64_t segment_usage = 0;
    if (!QuerySegmentBytesResident(info, seg.segment_id, &segment_usage)) return false;
    *usage_bytes += segment_usage;
  }

  if (!found_segment) {
    std::ostringstream ss;
    ss << "[wsl_vram] No non-local (system memory aperture) segment found";
    LOG_ERROR(ss);
  }
  return found_segment;
}

bool CpuWait(D3DKMT_HANDLE device, D3DKMT_HANDLE sync_obj, uint64_t* value) {
  D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU args = {};
  args.hDevice = device;
  args.ObjectCount = 1;
  args.ObjectHandleArray = &sync_obj;
  args.FenceValueArray = value;
  args.Flags.WaitAny = false;

  NTSTATUS status = DXCORE_CALL(D3DKMTWaitForSynchronizationObjectFromCpu)(&args);
  if (status != STATUS_SUCCESS) {
    std::ostringstream ss;
    ss << "[wsl_vram] D3DKMTWaitForSynchronizationObjectFromCpu failed: 0x" << std::hex << status;
    LOG_ERROR(ss);
    return false;
  }
  return true;
}

}  // namespace

uint64_t VramTotal(const WslAdapterInfo& info) {
  uint64_t total =
      info.device_info.local_visible_heap_size + info.device_info.local_invisible_heap_size;
  if (!info.device_info.is_dgpu) total += info.device_info.non_local_heap_size;
  return total;
}

bool QueryVramUsage(const WslAdapterInfo& info, uint64_t* usage_bytes) {
  if (!usage_bytes) return false;
  *usage_bytes = 0;

  uint64_t local_usage = 0;
  if (!QueryLocalVramUsage(info, &local_usage)) return false;
  *usage_bytes = local_usage;

  if (info.device_info.is_dgpu) return true;

  uint64_t non_local_usage = 0;
  if (!QueryNonLocalVramUsage(info, &non_local_usage)) return false;

  *usage_bytes += non_local_usage;
  return true;
}

bool VramAvailable(const WslAdapterInfo& info, uint64_t* available_bytes) {
  if (!available_bytes) return false;
  *available_bytes = 0;

  // WslAdapterInfo has no tracked fence-value counter (amdsmi never submits paging
  // work like the original WDDMDevice did), so wait on the current value the driver
  // has already written to the mapped fence CPU VA rather than a locally-cached one.
  if (info.paging_fence_cpu_va) {
    uint64_t value = *info.paging_fence_cpu_va;
    if (!CpuWait(info.device, info.paging_queue_sync_obj, &value)) return false;
  } else {
    std::ostringstream ss;
    ss << "[wsl_vram] paging_fence_cpu_va is null, skipping paging fence wait";
    LOG_WARN(ss);
  }

  uint64_t used = 0;
  if (!QueryVramUsage(info, &used)) return false;

  const uint64_t total = VramTotal(info);
  *available_bytes = used >= total ? 0 : total - used;
  return true;
}

}  // namespace wsl
