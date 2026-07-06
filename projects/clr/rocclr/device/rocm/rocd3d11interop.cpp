/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "rocdevice.hpp"
#include "rocmemory.hpp"
#include "rocd3d11interop.hpp"

#ifdef _WIN32

#include <D3D11.h>
#include <dxgi.h>
#include <mutex>
#include <unordered_map>

#include "platform/DxxInteropExt.h"
#include "platform/interop_d3d11.hpp"

namespace amd {
namespace roc {
namespace D3D11Interop {

static std::unordered_map<const Device*, CachedExt> gExtCache;
static std::mutex gExtCacheLock;

bool associateD3D11Device(const Device* device, ID3D11Device* pd3d11Device) {
  if (!device || !pd3d11Device) {
    return false;
  }

  if (!device->hasValidLUID()) {
    LogError("ROCr device does not have valid LUID for D3D11 interop");
    return false;
  }

  IDXGIDevice* pDXGIDevice = nullptr;
  HRESULT hr = pd3d11Device->QueryInterface(__uuidof(IDXGIDevice), (void**)&pDXGIDevice);
  if (FAILED(hr) || !pDXGIDevice) {
    LogError("Failed to query IDXGIDevice from D3D11 device");
    return false;
  }

  IDXGIAdapter* pDXGIAdapter = nullptr;
  hr = pDXGIDevice->GetAdapter(&pDXGIAdapter);
  pDXGIDevice->Release();
  if (FAILED(hr) || !pDXGIAdapter) {
    LogError("Failed to get DXGI adapter from D3D11 device");
    return false;
  }

  DXGI_ADAPTER_DESC adapterDesc;
  hr = pDXGIAdapter->GetDesc(&adapterDesc);
  pDXGIAdapter->Release();

  bool canInteroperate = SUCCEEDED(hr) &&
      (device->getDeviceLUID().HighPart == adapterDesc.AdapterLuid.HighPart) &&
      (device->getDeviceLUID().LowPart == adapterDesc.AdapterLuid.LowPart);

  if (!canInteroperate) {
    LogError("D3D11 device and ROCr device cannot interoperate (LUID mismatch)");
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
    LogError("atidxx DLL not loaded; D3D11 SRD queries will be unavailable");
    return true;  // LUID matched; proceed without SRD support
  }

  PFNAmdDxExtCreate11 AmdDxExtCreate11 =
      reinterpret_cast<PFNAmdDxExtCreate11>(GetProcAddress(hDLL, "AmdDxExtCreate11"));
  if (!AmdDxExtCreate11) {
    return true;
  }

  IAmdDxExt* pExt = nullptr;
  hr = AmdDxExtCreate11(pd3d11Device, &pExt);
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

void dissociateD3D11Device(const Device* device) {
  std::lock_guard<std::mutex> sl(gExtCacheLock);
  auto it = gExtCache.find(device);
  if (it != gExtCache.end()) {
    if (it->second.pCLExt) it->second.pCLExt->Release();
    if (it->second.pExt) it->second.pExt->Release();
    gExtCache.erase(it);
  }
}

bool Export(const Memory* memory, ID3D11Resource* d3d11Resource,
            UINT subresource, hsa_handle_t* handle, int* offset,
            void* srd, UINT* srdSize, hsa_interop_map_flag_t* mapFlags,
            size_t* sizeHint) {
  if (!memory || !d3d11Resource || !handle || !offset || !mapFlags) {
    return false;
  }

  // Query buffer size for D3D11 buffers so libhsakmt can handle resources
  // whose KMD private data is opaque (no UMDKMDIF SharedHandleInfo).
  if (sizeHint) {
    *sizeHint = 0;
    ID3D11Buffer* buf = nullptr;
    if (SUCCEEDED(d3d11Resource->QueryInterface(__uuidof(ID3D11Buffer), (void**)&buf)) && buf) {
      D3D11_BUFFER_DESC desc = {};
      buf->GetDesc(&desc);
      *sizeHint = desc.ByteWidth;
      buf->Release();
    }
  }

  // Use cached extension to query SRD
  if (srd && srdSize) {
    std::lock_guard<std::mutex> sl(gExtCacheLock);
    auto it = gExtCache.find(&memory->dev());
    if (it != gExtCache.end() && it->second.pCLExt) {
      HRESULT hr = it->second.pCLExt->CLQueryResource11(d3d11Resource, srd, srdSize);
      if (FAILED(hr)) {
        LogError("CLQueryResource11 failed for D3D11 resource");
      }
    }
  }

  IDXGIResource* pDxgiRes = nullptr;
  HRESULT hr = d3d11Resource->QueryInterface(__uuidof(IDXGIResource), (void**)&pDxgiRes);
  if (FAILED(hr) || !pDxgiRes) {
    LogError("Failed to query IDXGIResource from D3D11 resource");
    return false;
  }

  // Get legacy KMT shared handle (requires D3D11_RESOURCE_MISC_SHARED).
  HANDLE hShared = nullptr;
  hr = pDxgiRes->GetSharedHandle(&hShared);
  pDxgiRes->Release();

  if (FAILED(hr) || !hShared) {
    LogError("Failed to get KMT shared handle from D3D11 resource");
    return false;
  }

  *handle = reinterpret_cast<hsa_handle_t>(hShared);
  *offset = 0;
  *mapFlags = HSA_INTEROP_MAP_FLAG_KMT_HANDLE;
  return true;
}

}  // namespace D3D11Interop
}  // namespace roc
}  // namespace amd

#endif  // _WIN32
