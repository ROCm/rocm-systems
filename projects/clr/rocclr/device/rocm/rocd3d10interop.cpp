/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "rocdevice.hpp"
#include "rocmemory.hpp"
#include "rocd3d10interop.hpp"

#ifdef _WIN32

#include <D3D10.h>
#include <dxgi.h>
#include <mutex>
#include <unordered_map>

#include "platform/DxxInteropExt.h"
#include "platform/interop_d3d10.hpp"

namespace amd {
namespace roc {
namespace D3D10Interop {

static std::unordered_map<const Device*, CachedExt> gExtCache;
static std::mutex gExtCacheLock;

bool associateD3D10Device(const Device* device, ID3D10Device* pd3d10Device) {
  if (!device || !pd3d10Device) {
    return false;
  }

  if (!device->hasValidLUID()) {
    LogError("ROCr device does not have valid LUID for D3D10 interop");
    return false;
  }

  IDXGIDevice* pDXGIDevice = nullptr;
  HRESULT hr = pd3d10Device->QueryInterface(__uuidof(IDXGIDevice), (void**)&pDXGIDevice);
  if (FAILED(hr) || !pDXGIDevice) {
    LogError("Failed to query IDXGIDevice from D3D10 device");
    return false;
  }

  IDXGIAdapter* pDXGIAdapter = nullptr;
  hr = pDXGIDevice->GetAdapter(&pDXGIAdapter);
  pDXGIDevice->Release();
  if (FAILED(hr) || !pDXGIAdapter) {
    LogError("Failed to get DXGI adapter from D3D10 device");
    return false;
  }

  DXGI_ADAPTER_DESC adapterDesc;
  hr = pDXGIAdapter->GetDesc(&adapterDesc);
  pDXGIAdapter->Release();

  bool canInteroperate = SUCCEEDED(hr) &&
      (device->getDeviceLUID().HighPart == adapterDesc.AdapterLuid.HighPart) &&
      (device->getDeviceLUID().LowPart == adapterDesc.AdapterLuid.LowPart);

  if (!canInteroperate) {
    LogError("D3D10 device and ROCr device cannot interoperate (LUID mismatch)");
    return false;
  }

  // Create and cache AMD DXX extension objects for this device
#if defined _WIN64
  static constexpr CHAR dxxModuleName[13] = "atidxx64.dll";
#else
  static constexpr CHAR dxxModuleName[13] = "atidxx32.dll";
#endif

  HMODULE hDLL = GetModuleHandle(dxxModuleName);
  if (!hDLL) {
    LogError("atidxx DLL not loaded; D3D10 SRD queries will be unavailable");
    return true;  // LUID matched; proceed without SRD support
  }

  PFNAmdDxExtCreate AmdDxExtCreate =
      reinterpret_cast<PFNAmdDxExtCreate>(GetProcAddress(hDLL, "AmdDxExtCreate"));
  if (!AmdDxExtCreate) {
    return true;
  }

  IAmdDxExt* pExt = nullptr;
  hr = AmdDxExtCreate(pd3d10Device, &pExt);
  if (FAILED(hr) || !pExt) {
    return true;
  }

  IAmdDxExtCLInterop* pCLExt =
      static_cast<IAmdDxExtCLInterop*>(pExt->GetExtInterface(AmdDxExtCLInteropID));
  if (!pCLExt) {
    pExt->Release();
    return true;
  }

  std::lock_guard<std::mutex> sl(gExtCacheLock);
  gExtCache[device] = {pExt, pCLExt};

  return true;
}

void dissociateD3D10Device(const Device* device) {
  std::lock_guard<std::mutex> sl(gExtCacheLock);
  auto it = gExtCache.find(device);
  if (it != gExtCache.end()) {
    if (it->second.pCLExt) it->second.pCLExt->Release();
    if (it->second.pExt) it->second.pExt->Release();
    gExtCache.erase(it);
  }
}

bool Export(const Memory* memory, ID3D10Resource* d3d10Resource,
            UINT subresource, hsa_handle_t* handle, int* offset,
            void* srd, UINT* srdSize, hsa_interop_map_flag_t* mapFlags) {
  if (!memory || !d3d10Resource || !handle || !offset || !mapFlags) {
    return false;
  }

  // Use cached extension to query SRD
  if (srd && srdSize) {
    std::lock_guard<std::mutex> sl(gExtCacheLock);
    auto it = gExtCache.find(&memory->dev());
    if (it != gExtCache.end() && it->second.pCLExt) {
      HRESULT hr = it->second.pCLExt->CLQueryResource(d3d10Resource, srd, srdSize);
      if (FAILED(hr)) {
        LogError("CLQueryResource failed for D3D10 resource");
      }
    }
  }

  IDXGIResource* pDxgiRes = nullptr;
  HRESULT hr = d3d10Resource->QueryInterface(__uuidof(IDXGIResource), (void**)&pDxgiRes);
  if (FAILED(hr) || !pDxgiRes) {
    LogError("Failed to query IDXGIResource from D3D10 resource");
    return false;
  }

  // Get legacy KMT shared handle (requires D3D10_RESOURCE_MISC_SHARED).
  HANDLE hShared = nullptr;
  hr = pDxgiRes->GetSharedHandle(&hShared);
  pDxgiRes->Release();

  if (FAILED(hr) || !hShared) {
    LogError("Failed to get KMT shared handle from D3D10 resource");
    return false;
  }

  *handle = reinterpret_cast<hsa_handle_t>(hShared);
  *offset = 0;
  *mapFlags = HSA_INTEROP_MAP_FLAG_KMT_HANDLE;
  return true;
}

}  // namespace D3D10Interop
}  // namespace roc
}  // namespace amd

#endif  // _WIN32
