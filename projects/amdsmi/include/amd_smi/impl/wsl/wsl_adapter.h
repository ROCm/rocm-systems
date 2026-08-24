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

#pragma once

#include <cstdint>
#include <vector>

#include "amd_smi/impl/wsl/wddm_dxg_types.h"
#include "wkmi.h"

namespace wsl {

enum class SegmentKind { kUnknown, kAperture, kLocalMemory, kSystemMemory };

struct SegmentInfo {
  uint32_t segment_id = 0;
  SegmentKind kind = SegmentKind::kUnknown;
  bool is_aperture = false;
  bool is_system_memory = false;
};

struct WslAdapterInfo {
  D3DKMT_HANDLE adapter = 0;
  D3DKMT_HANDLE device = 0;
  D3DKMT_HANDLE paging_queue = 0;
  D3DKMT_HANDLE paging_queue_sync_obj = 0;
  uint64_t* paging_fence_cpu_va = nullptr;  // mapped fence CPU VA, from D3DKMT_CREATEPAGINGQUEUE
  LUID adapter_luid = {};
  uint32_t vendor_id = 0;
  uint32_t device_id = 0;
  uint32_t sub_vendor_id = 0;
  uint32_t sub_system_id = 0;
  Wkmi::DeviceInfo device_info = {};
  uint32_t wddm_version = 0;  // this adapter's WDDM version (see QueryWddmVersion below)
  std::vector<SegmentInfo>
      segments;  // cached local/system/aperture classification, populated once at open time
  char board_product_name[256] = {};
  char driver_version[256] = {};
  char driver_desc[256] = {};
  bool power_opt_disabled = false;
};

// Enumerate WSL-capable AMD adapters (vendor filter + Wkmi::QueryAdapterSupported check).
// Returns one WslAdapterInfo per adapter with `adapter` (LUID/handle-identifying fields) populated,
// but NOT yet opened (no device/paging-queue/segments — those are filled by OpenWslAdapter).
NTSTATUS EnumerateWslAdapters(std::vector<WslAdapterInfo>* out);

// Open a device + paging queue on an already-enumerated adapter, disable power optimization,
// classify segments, and populate device_info/board_product_name/driver_version/driver_desc.
NTSTATUS OpenWslAdapter(WslAdapterInfo* info);

// Tear down device/paging-queue/sync-obj opened by OpenWslAdapter (safe to call on a
// not-fully-opened WslAdapterInfo; only tears down what's non-zero).
void CloseWslAdapter(WslAdapterInfo* info);

}  // namespace wsl
