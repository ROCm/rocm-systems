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

#include "platform/context.hpp"
#include "platform/DxxInteropExt.h"
#include "platform/interop_d3d10.hpp"

namespace amd {
namespace roc {
namespace D3D10Interop {

static PFNAmdDxExtCreate gAmdDxExtCreate = nullptr;
static std::once_flag gAmdDxExtCreateFlag;

static PFNAmdDxExtCreate GetAmdDxExtCreate() {
  std::call_once(gAmdDxExtCreateFlag, []() {
#if defined _WIN64
    static constexpr CHAR dxxModuleName[13] = "atidxx64.dll";
#else
    static constexpr CHAR dxxModuleName[13] = "atidxx32.dll";
#endif
    HMODULE hDLL = GetModuleHandle(dxxModuleName);
    if (hDLL) {
      gAmdDxExtCreate = reinterpret_cast<PFNAmdDxExtCreate>(
          GetProcAddress(hDLL, "AmdDxExtCreate"));
    }
  });
  return gAmdDxExtCreate;
}

bool associateD3D10Device(const Device* device, ID3D10Device* pd3d10Device,
                          void* gfxContext, bool validateOnly) {
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

  if (validateOnly) {
    return true;
  }

  amd::Context* ctx = static_cast<amd::Context*>(gfxContext);
  if (ctx->getD3D10Ext(pd3d10Device)) {
    return true;
  }

  PFNAmdDxExtCreate AmdDxExtCreate = GetAmdDxExtCreate();
  if (AmdDxExtCreate) {
    IAmdDxExt* pExt = nullptr;
    HRESULT hr = AmdDxExtCreate(pd3d10Device, &pExt);
    if (SUCCEEDED(hr) && pExt) {
      IAmdDxExtCLInterop* pCLExt =
          static_cast<IAmdDxExtCLInterop*>(pExt->GetExtInterface(AmdDxExtCLInteropID));
      if (pCLExt) {
        ctx->setD3D10Ext(pd3d10Device, pExt, pCLExt);
      } else {
        pExt->Release();
      }
    }
  }

  return true;
}

void dissociateD3D10Device(const Device* device, void* const gfxDevice[], void* gfxContext) {
  if (!gfxDevice || !gfxContext) return;
  void* d3d10Device = gfxDevice[amd::Context::DeviceFlagIdx::D3D10DeviceKhrIdx];
  if (!d3d10Device) return;

  amd::Context* ctx = static_cast<amd::Context*>(gfxContext);
  void* pExt = nullptr;
  void* pCLExt = nullptr;
  if (ctx->removeD3D10Ext(d3d10Device, &pExt, &pCLExt)) {
    if (pCLExt) static_cast<IAmdDxExtCLInterop*>(pCLExt)->Release();
    if (pExt)   static_cast<IAmdDxExt*>(pExt)->Release();
  }
}

bool Export(const Memory* memory, D3D10Object* d3d10Obj,
            hsa_handle_t* handle, int* offset,
            void* srd, UINT* srdSize, hsa_interop_map_flag_t* mapFlags) {
  auto d3d10Resource = d3d10Obj->getD3D10Resource();
  if (!memory || !d3d10Resource || !handle || !offset || !mapFlags) {
    return false;
  }

  if (srd && srdSize) {
    ID3D10Device* pOwnerDevice = nullptr;
    d3d10Resource->GetDevice(&pOwnerDevice);
    if (pOwnerDevice) {
      amd::Context& ctx = memory->owner()->getContext();
      IAmdDxExtCLInterop* pCLExt =
          static_cast<IAmdDxExtCLInterop*>(ctx.getD3D10Ext(pOwnerDevice));
      pOwnerDevice->Release();

      if (!pCLExt) {
        LogError("D3D10 device not associated with context; call clCreateContext with D3D10 device");
        return false;
      }

      HRESULT hr = pCLExt->CLQueryResource(d3d10Resource, 0, srd, srdSize);
      if (FAILED(hr)) {
        LogPrintfError("CLQueryResource failed for D3D10 resource: 0x%xh", hr);
        return false;
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
