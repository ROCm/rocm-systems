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

#include "platform/context.hpp"
#include "platform/DxxInteropExt.h"
#include "platform/interop_d3d11.hpp"

namespace amd {
namespace roc {
namespace D3D11Interop {

// Extension objects are cached per-ID3D11Device in the OpenCL context to support
// multiple D3D11 devices on the same adapter across contexts/threads.
static PFNAmdDxExtCreate11 gAmdDxExtCreate11 = nullptr;
static std::once_flag gAmdDxExtCreate11Flag;

static PFNAmdDxExtCreate11 GetAmdDxExtCreate11() {
  std::call_once(gAmdDxExtCreate11Flag, []() {
#if defined _WIN64
    static constexpr CHAR dxxModuleName[13] = "atidxx64.dll";
#else
    static constexpr CHAR dxxModuleName[13] = "atidxx32.dll";
#endif
    HMODULE hDLL = GetModuleHandle(dxxModuleName);
    if (hDLL) {
      gAmdDxExtCreate11 = reinterpret_cast<PFNAmdDxExtCreate11>(
          GetProcAddress(hDLL, "AmdDxExtCreate11"));
    }
  });
  return gAmdDxExtCreate11;
}

bool associateD3D11Device(const Device* device, ID3D11Device* pd3d11Device,
                          void* gfxContext, bool validateOnly) {
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

  if (validateOnly) {
    return true;
  }

  amd::Context* ctx = static_cast<amd::Context*>(gfxContext);
  if (ctx->getD3D11Ext(pd3d11Device)) {
    return true;
  }

  // Create and cache IAmdDxExtCLInterop in the context, keyed by pd3d11Device.
  // This avoids per-query extension creation in multithread/multicontext scenarios.
  PFNAmdDxExtCreate11 AmdDxExtCreate11 = GetAmdDxExtCreate11();
  if (AmdDxExtCreate11) {
    IAmdDxExt* pExt = nullptr;
    HRESULT hr = AmdDxExtCreate11(pd3d11Device, &pExt);
    if (SUCCEEDED(hr) && pExt) {
      IAmdDxExtCLInterop* pCLExt =
          static_cast<IAmdDxExtCLInterop*>(pExt->GetExtInterface(AmdDxExtCLInteropID));
      if (pCLExt) {
        ctx->setD3D11Ext(pd3d11Device, pExt, pCLExt);
      } else {
        pExt->Release();
      }
    }
  }

  return true;
}

void dissociateD3D11Device(const Device* device, void* const gfxDevice[], void* gfxContext) {
  if (!gfxDevice || !gfxContext) return;
  void* d3d11Device = gfxDevice[amd::Context::DeviceFlagIdx::D3D11DeviceKhrIdx];
  if (!d3d11Device) return;

  amd::Context* ctx = static_cast<amd::Context*>(gfxContext);
  void* pExt = nullptr;
  void* pCLExt = nullptr;
  if (ctx->removeD3D11Ext(d3d11Device, &pExt, &pCLExt)) {
    if (pCLExt) static_cast<IAmdDxExtCLInterop*>(pCLExt)->Release();
    if (pExt)   static_cast<IAmdDxExt*>(pExt)->Release();
  }
}

bool Export(const Memory* memory, D3D11Object* d3d11Obj,
            hsa_handle_t* handle, int* offset,
            void* srd, UINT* srdSize, hsa_interop_map_flag_t* mapFlags,
            size_t* sizeHint) {
  auto d3d11Resource = d3d11Obj->getD3D11Resource();
  bool isYUV = (d3d11Obj->getDxgiFormat() == DXGI_FORMAT_NV12 ||
                d3d11Obj->getDxgiFormat() == DXGI_FORMAT_P010);
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

  // Query SRD using an extension cached in the OpenCL context, keyed by the resource's ownving
  // ID3D11Device. This supports multiple ID3D11Device instances on the same adapter across
  // contexts/threads.
  if (srd && srdSize) {
    // plane -1 means "full resource" which is no longer valid; treat as plane 0.
    // Because Pal in D3D doesn't support querying SRD of the full resource of types
    // DXGI_FORMAT_NV12 and DXGI_FORMAT_P010.
    UINT planeIndex = 0;
    if (isYUV) {
      switch (d3d11Obj->getPlane()) {
        case -1:
        case 0:
          planeIndex = 0;
          break;
        case 1:
          planeIndex = 1;
          break;
        default:
          LogPrintfError("Unexpected plane %d", d3d11Obj->getPlane());
          return false;
      }
    }

    ID3D11Device* pOwnerDevice = nullptr;
    d3d11Resource->GetDevice(&pOwnerDevice);
    if (pOwnerDevice) {
      amd::Context& ctx = memory->owner()->getContext();
      IAmdDxExtCLInterop* pCLExt =
          static_cast<IAmdDxExtCLInterop*>(ctx.getD3D11Ext(pOwnerDevice));
      pOwnerDevice->Release();

      if (!pCLExt) {
        LogError("D3D11 device not associated with context; call clCreateContext with D3D11 device");
        return false;
      }

      HRESULT hr = pCLExt->CLQueryResource11(d3d11Resource, planeIndex, srd, srdSize);
      if (FAILED(hr)) {
        LogPrintfError("CLQueryResource11 failed for D3D11 resource: 0x%xh", hr);
        return false;
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

  if (isYUV && d3d11Obj->getPlane() == 1) {
    uint32_t w = d3d11Obj->getWidth() * 2u;
    uint32_t h = d3d11Obj->getHeight() * 2u;
    const uint32_t pitchAlign = memory->dev().info().imagePitchAlignment_;
    // Y plane (plane 0) element size for pitch calculation
    const uint32_t elemBytes =
        static_cast<uint32_t>(d3d11Obj->getElementBytes(d3d11Obj->getDxgiFormat(), 0));
    *offset = amd::alignUp(w * elemBytes, pitchAlign) * h;
  }
  return true;
}

}  // namespace D3D11Interop
}  // namespace roc
}  // namespace amd

#endif  // _WIN32
