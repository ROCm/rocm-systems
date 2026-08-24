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

#include "amd_smi/impl/wsl/wsl_adapter.h"

#include <ntstatus.h>

#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <memory>
#include <sstream>

#include "amd_smi/impl/wsl/dxcore_loader.h"
#include "rocm_smi/rocm_smi_logger.h"

namespace wsl {

namespace {

NTSTATUS QueryAdapterInfo(D3DKMT_HANDLE adapter, KMTQUERYADAPTERINFOTYPE type, void* data,
                          uint32_t size) {
  D3DKMT_QUERYADAPTERINFO args = {};
  args.hAdapter = adapter;
  args.Type = type;
  args.pPrivateDriverData = data;
  args.PrivateDriverDataSize = size;
  return DXCORE_CALL(D3DKMTQueryAdapterInfo)(&args);
}

uint32_t QueryWddmVersion(D3DKMT_HANDLE adapter) {
  D3DKMT_DRIVERVERSION version = static_cast<D3DKMT_DRIVERVERSION>(0);
  if (QueryAdapterInfo(adapter, KMTQAITYPE_DRIVERVERSION, &version, sizeof(version)) !=
      STATUS_SUCCESS) {
    std::ostringstream ss;
    ss << "[wsl_adapter] Failed to query WDDM version";
    LOG_ERROR(ss);
    return 0;
  }
  return static_cast<uint32_t>(version);
}

// Reads a REG_SZ value from the adapter's driver registry key (e.g. "ReleaseVersion",
// "DriverDesc"). Non-fatal on failure -- callers just leave the output empty.
bool QueryAdapterRegStr(D3DKMT_HANDLE adapter, const char* key_name, char* buf_out,
                        size_t buf_len) {
  static constexpr uint32_t kMaxOutputSize = 512;
  struct RegQuery {
    D3DDDI_QUERYREGISTRY_INFO info;
    wchar_t output[kMaxOutputSize];
  } q = {};
  q.info.QueryType = D3DDDI_QUERYREGISTRY_ADAPTERKEY;
  q.info.QueryFlags.TranslatePath = 0;
  q.info.ValueType = 1;  // REG_SZ
  if (mbstowcs(q.info.ValueName, key_name, MAX_PATH) == static_cast<size_t>(-1)) return false;

  D3DKMT_QUERYADAPTERINFO args = {};
  args.hAdapter = adapter;
  args.Type = KMTQAITYPE_QUERYREGISTRY;
  args.pPrivateDriverData = &q;
  args.PrivateDriverDataSize = sizeof(q);
  if (DXCORE_CALL(D3DKMTQueryAdapterInfo)(&args) != STATUS_SUCCESS) return false;
  if (q.info.Status != D3DDDI_QUERYREGISTRY_STATUS_SUCCESS) return false;

  if (buf_out && buf_len > 0) {
    wcstombs(buf_out, q.info.OutputString, buf_len - 1);
    buf_out[buf_len - 1] = '\0';
  }
  return true;
}

NTSTATUS CreateDevice(WslAdapterInfo* info) {
  D3DKMT_CREATEDEVICE args = {};
  args.hAdapter = info->adapter;

  NTSTATUS status = DXCORE_CALL(D3DKMTCreateDevice)(&args);
  if (status != STATUS_SUCCESS) {
    std::ostringstream ss;
    ss << "[wsl_adapter] D3DKMTCreateDevice failed: 0x" << std::hex << status;
    LOG_ERROR(ss);
    return status;
  }

  info->device = args.hDevice;
  return STATUS_SUCCESS;
}

void DestroyDevice(WslAdapterInfo* info) {
  D3DKMT_DESTROYDEVICE args = {};
  args.hDevice = info->device;

  NTSTATUS status = DXCORE_CALL(D3DKMTDestroyDevice)(&args);
  if (status != STATUS_SUCCESS) {
    std::ostringstream ss;
    ss << "[wsl_adapter] D3DKMTDestroyDevice failed: 0x" << std::hex << status;
    LOG_ERROR(ss);
  }
  info->device = 0;
}

NTSTATUS CreatePagingQueue(WslAdapterInfo* info) {
  D3DKMT_CREATEPAGINGQUEUE args = {};
  args.hDevice = info->device;
  args.Priority = D3DDDI_PAGINGQUEUE_PRIORITY_NORMAL;

  NTSTATUS status = DXCORE_CALL(D3DKMTCreatePagingQueue)(&args);
  if (status != STATUS_SUCCESS) {
    std::ostringstream ss;
    ss << "[wsl_adapter] D3DKMTCreatePagingQueue failed: 0x" << std::hex << status;
    LOG_ERROR(ss);
    return status;
  }

  info->paging_queue = args.hPagingQueue;
  info->paging_queue_sync_obj = args.hSyncObject;
  info->paging_fence_cpu_va = reinterpret_cast<uint64_t*>(args.FenceValueCPUVirtualAddress);
  return STATUS_SUCCESS;
}

void DestroyPagingQueue(WslAdapterInfo* info) {
  D3DDDI_DESTROYPAGINGQUEUE args = {};
  args.hPagingQueue = info->paging_queue;

  NTSTATUS status = DXCORE_CALL(D3DKMTDestroyPagingQueue)(&args);
  if (status != STATUS_SUCCESS) {
    std::ostringstream ss;
    ss << "[wsl_adapter] D3DKMTDestroyPagingQueue failed: 0x" << std::hex << status;
    LOG_ERROR(ss);
  }
  info->paging_queue = 0;
  info->paging_queue_sync_obj = 0;
  info->paging_fence_cpu_va = nullptr;
}

// Non-fatal: KMD treats a failed escape here as "leave power optimization as-is".
void SetPowerOptimization(WslAdapterInfo* info, bool restore) {
  int priv_size = Wkmi::GetPowerOptPrivDataSize();
  std::unique_ptr<char[]> priv_data(new char[priv_size]);
  memset(priv_data.get(), 0, priv_size);
  Wkmi::FillinPowerOptPrivData(priv_data.get(), restore);

  D3DKMT_ESCAPE escape = {};
  escape.hAdapter = info->adapter;
  escape.hDevice = info->device;
  escape.hContext = 0;  // KMD only uses device to identify the process
  escape.Type = D3DKMT_ESCAPE_DRIVERPRIVATE;
  escape.pPrivateDriverData = priv_data.get();
  escape.PrivateDriverDataSize = priv_size;
  escape.Flags.HardwareAccess = true;

  NTSTATUS status = DXCORE_CALL(D3DKMTEscape)(&escape);
  if (status != STATUS_SUCCESS) {
    std::ostringstream ss;
    ss << "[wsl_adapter] SetPowerOptimization(restore=" << restore << ") escape failed: 0x"
       << std::hex << status;
    LOG_ERROR(ss);
    return;
  }

  if (!restore) info->power_opt_disabled = true;
}

// Best-effort: leaves info->segments empty on failure rather than failing adapter open.
void QuerySegmentInfo(WslAdapterInfo* info) {
  info->segments.clear();

  D3DKMT_QUERYSTATISTICS adapter_query = {};
  adapter_query.Type = D3DKMT_QUERYSTATISTICS_ADAPTER;
  adapter_query.AdapterLuid = info->adapter_luid;

  NTSTATUS status = DXCORE_CALL(D3DKMTQueryStatistics)(&adapter_query);
  if (status != STATUS_SUCCESS) {
    std::ostringstream ss;
    ss << "[wsl_adapter] D3DKMTQueryStatistics(ADAPTER) failed: 0x" << std::hex << status;
    LOG_ERROR(ss);
    return;
  }

  uint32_t segment_count = adapter_query.QueryResult.AdapterInformation.NbSegments;
  for (uint32_t i = 0; i < segment_count; ++i) {
    D3DKMT_QUERYSTATISTICS seg_query = {};
    seg_query.Type = D3DKMT_QUERYSTATISTICS_SEGMENT;
    seg_query.AdapterLuid = info->adapter_luid;
    seg_query.QuerySegment.SegmentId = i;

    status = DXCORE_CALL(D3DKMTQueryStatistics)(&seg_query);
    if (status != STATUS_SUCCESS) {
      std::ostringstream ss;
      ss << "[wsl_adapter] D3DKMTQueryStatistics(SEGMENT " << i << ") failed: 0x" << std::hex
         << status;
      LOG_ERROR(ss);
      return;
    }

    const auto& seg = seg_query.QueryResult.SegmentInformation;

    SegmentInfo seg_info;
    seg_info.segment_id = i;
    seg_info.is_aperture = seg.Aperture;
    seg_info.is_system_memory = seg.SegmentProperties.SystemMemory;
    if (seg.Aperture)
      seg_info.kind = SegmentKind::kAperture;
    else
      seg_info.kind = seg.SegmentProperties.SystemMemory ? SegmentKind::kSystemMemory
                                                         : SegmentKind::kLocalMemory;

    info->segments.push_back(seg_info);
  }
}

void CloseAdapterHandle(D3DKMT_HANDLE adapter) {
  D3DKMT_CLOSEADAPTER args = {};
  args.hAdapter = adapter;

  NTSTATUS status = DXCORE_CALL(D3DKMTCloseAdapter)(&args);
  if (status != STATUS_SUCCESS) {
    std::ostringstream ss;
    ss << "[wsl_adapter] D3DKMTCloseAdapter failed: 0x" << std::hex << status;
    LOG_ERROR(ss);
  }
}

void QueryBoardProductName(WslAdapterInfo* info) {
  D3DKMT_ADAPTERREGISTRYINFO reg_info = {};
  NTSTATUS status =
      QueryAdapterInfo(info->adapter, KMTQAITYPE_ADAPTERREGISTRYINFO, &reg_info, sizeof(reg_info));
  char adapter_string[MAX_PATH] = {};
  if (status == STATUS_SUCCESS && reg_info.AdapterString[0])
    wcstombs(adapter_string, reg_info.AdapterString, sizeof(adapter_string) - 1);

  const char* product_name = adapter_string[0] ? adapter_string : info->device_info.product_name;
  strncpy(info->board_product_name, product_name, sizeof(info->board_product_name) - 1);
}

}  // namespace

NTSTATUS EnumerateWslAdapters(std::vector<WslAdapterInfo>* out) {
  D3DKMT_ENUMADAPTERS3 args = {};
  args.Filter.IncludeComputeOnly = true;

  // First pass with pAdapters == nullptr just returns NumAdapters.
  NTSTATUS status = DXCORE_CALL(D3DKMTEnumAdapters3)(&args);
  if (status != STATUS_SUCCESS) {
    std::ostringstream ss;
    ss << "[wsl_adapter] D3DKMTEnumAdapters3 (count) failed: 0x" << std::hex << status;
    LOG_ERROR(ss);
    return status;
  }

  if (args.NumAdapters == 0) return STATUS_SUCCESS;

  std::unique_ptr<D3DKMT_ADAPTERINFO[]> adapters(new D3DKMT_ADAPTERINFO[args.NumAdapters]);
  args.pAdapters = adapters.get();
  status = DXCORE_CALL(D3DKMTEnumAdapters3)(&args);
  if (status != STATUS_SUCCESS) {
    std::ostringstream ss;
    ss << "[wsl_adapter] D3DKMTEnumAdapters3 (fill) failed: 0x" << std::hex << status;
    LOG_ERROR(ss);
    return status;
  }

  for (uint32_t i = 0; i < args.NumAdapters; ++i) {
    D3DKMT_QUERY_DEVICE_IDS query = {};
    bool queried = QueryAdapterInfo(adapters[i].hAdapter, KMTQAITYPE_PHYSICALADAPTERDEVICEIDS,
                                    &query, sizeof(query)) == STATUS_SUCCESS;

    if (!queried || query.DeviceIds.VendorID != 0x1002 ||
        !Wkmi::QueryAdapterSupported(query.DeviceIds.DeviceID)) {
      CloseAdapterHandle(adapters[i].hAdapter);
      continue;
    }

    WslAdapterInfo info;
    info.adapter = adapters[i].hAdapter;
    info.adapter_luid = adapters[i].AdapterLuid;
    info.vendor_id = query.DeviceIds.VendorID;
    info.device_id = query.DeviceIds.DeviceID;
    info.sub_vendor_id = query.DeviceIds.SubVendorID;
    info.sub_system_id = query.DeviceIds.SubSystemID;
    info.wddm_version = QueryWddmVersion(adapters[i].hAdapter);
    out->push_back(info);
  }

  return STATUS_SUCCESS;
}

NTSTATUS OpenWslAdapter(WslAdapterInfo* info) {
  NTSTATUS status = CreateDevice(info);
  if (status != STATUS_SUCCESS) return status;

  SetPowerOptimization(info, /*restore=*/false);

  status = CreatePagingQueue(info);
  if (status != STATUS_SUCCESS) {
    CloseWslAdapter(info);
    return status;
  }

  status = Wkmi::ParseAdapterInfo(info->adapter, &info->device_info);
  if (status == STATUS_OBJECT_NAME_NOT_FOUND || status == STATUS_REVISION_MISMATCH) {
    // Adapter registry info absent or GFX IP unsupported -- tolerate and continue
    // with a partially-populated device_info, matching WDDMDevice's behavior.
    std::ostringstream ss;
    ss << "[wsl_adapter] Wkmi::ParseAdapterInfo returned 0x" << std::hex << status
       << " (non-fatal)";
    LOG_INFO(ss);
  } else if (status != STATUS_SUCCESS) {
    std::ostringstream ss;
    ss << "[wsl_adapter] Wkmi::ParseAdapterInfo failed: 0x" << std::hex << status;
    LOG_ERROR(ss);
    CloseWslAdapter(info);
    return status;
  }

  QuerySegmentInfo(info);

  QueryAdapterRegStr(info->adapter, "ReleaseVersion", info->driver_version,
                     sizeof(info->driver_version));
  QueryAdapterRegStr(info->adapter, "DriverDesc", info->driver_desc, sizeof(info->driver_desc));
  QueryBoardProductName(info);

  return STATUS_SUCCESS;
}

void CloseWslAdapter(WslAdapterInfo* info) {
  if (info->device != 0) SetPowerOptimization(info, /*restore=*/true);

  if (info->paging_queue != 0) DestroyPagingQueue(info);

  if (info->device != 0) DestroyDevice(info);

  if (info->device_info.adapter_info != nullptr) {
    free(info->device_info.adapter_info);
    info->device_info.adapter_info = nullptr;
  }

  if (info->adapter != 0) {
    CloseAdapterHandle(info->adapter);
    info->adapter = 0;
  }
}

}  // namespace wsl
