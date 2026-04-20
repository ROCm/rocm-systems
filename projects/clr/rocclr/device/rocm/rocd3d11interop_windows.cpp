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

#include "DxxOpenCLInteropExt.h"
#include "platform/interop_d3d11.hpp"

namespace amd {
namespace roc {
namespace D3D11Interop {

/**
 * @brief Query GPU mask from D3D11 device using AMD DXX extensions
 *
 * Loads atidxx64.dll (or atidxx32.dll) and uses AMD DirectX extension
 * interface to query which GPU(s) in a multi-GPU chain can interoperate
 * with the given D3D11 device.
 *
 * @param pd3d11Device D3D11 device
 * @param pd3d11DeviceGPUMask Output bitmask of GPU indices
 * @return true if GPU mask was successfully queried
 */
static bool queryD3D11DeviceGPUMask(ID3D11Device* pd3d11Device, UINT* pd3d11DeviceGPUMask) {
  IAmdDxExt* pExt = nullptr;
  IAmdDxExtCLInterop* pCLExt = nullptr;
  PFNAmdDxExtCreate11 AmdDxExtCreate11;
  HRESULT hr = S_OK;

// Get a handle to the DXX DLL with extension API support
#if defined _WIN64
  static constexpr CHAR dxxModuleName[13] = "atidxx64.dll";
#else
  static constexpr CHAR dxxModuleName[13] = "atidxx32.dll";
#endif

  HMODULE hDLL = GetModuleHandle(dxxModuleName);

  if (hDLL == nullptr) {
    hr = E_FAIL;
  }

  // Get the exported AmdDxExtCreate11() function pointer
  if (SUCCEEDED(hr)) {
    AmdDxExtCreate11 =
        reinterpret_cast<PFNAmdDxExtCreate11>(GetProcAddress(hDLL, "AmdDxExtCreate11"));
    if (AmdDxExtCreate11 == nullptr) {
      hr = E_FAIL;
    }
  }

  // Create the extension object
  if (SUCCEEDED(hr)) {
    hr = AmdDxExtCreate11(pd3d11Device, &pExt);
  }

  // Get the extension version information
  if (SUCCEEDED(hr)) {
    AmdDxExtVersion extVersion;
    hr = pExt->GetVersion(&extVersion);

    if (extVersion.majorVersion == 0) {
      hr = E_FAIL;
    }
  }

  // Get the OpenCL Interop interface
  if (SUCCEEDED(hr)) {
    pCLExt = static_cast<IAmdDxExtCLInterop*>(pExt->GetExtInterface(AmdDxExtCLInteropID));
    if (pCLExt != nullptr) {
      // Get the GPU mask using the CL Interop extension.
      pCLExt->QueryInteropGpuMask(pd3d11DeviceGPUMask);
    } else {
      hr = E_FAIL;
    }
  }

  if (pCLExt != nullptr) {
    pCLExt->Release();
  }

  if (pExt != nullptr) {
    pExt->Release();
  }

  return (SUCCEEDED(hr));
}

bool associateD3D11Device(const Device* device, ID3D11Device* pd3d11Device) {
  if (!device || !pd3d11Device) {
    return false;
  }

  // Verify device has valid LUID
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
  if (FAILED(hr) || !pDXGIAdapter) {
    pDXGIDevice->Release();
    LogError("Failed to get DXGI adapter from D3D11 device");
    return false;
  }

  DXGI_ADAPTER_DESC adapterDesc;
  hr = pDXGIAdapter->GetDesc(&adapterDesc);

  // Stage 1: match the adapter LUID
  bool canInteroperate = SUCCEEDED(hr) &&
      (device->getDeviceLUID().HighPart == adapterDesc.AdapterLuid.HighPart) &&
      (device->getDeviceLUID().LowPart == adapterDesc.AdapterLuid.LowPart);

  // Stage 2: match the GPU chain ID using DXX extension
  if (canInteroperate) {
    UINT d3d11DeviceGPUMask = 0;
    UINT chainBitMask = 1 << device->getGpuIndex();

    if (queryD3D11DeviceGPUMask(pd3d11Device, &d3d11DeviceGPUMask)) {
      canInteroperate = (chainBitMask & d3d11DeviceGPUMask) != 0;
    } else {
      // Special handling for Intel iGPU + AMD dGPU in LDA mode
      // (only occurs on a PX platform) where
      // the D3D11Device object is created on the Intel iGPU and
      // passed to AMD dGPU (secondary) to interoperate.
      if (chainBitMask > 1) {
        canInteroperate = false;
      }
      // If we're GPU 0 and can't query mask, assume it's OK
    }
  }

  pDXGIDevice->Release();
  pDXGIAdapter->Release();

  if (!canInteroperate) {
    LogError("D3D11 device and ROCr device cannot interoperate (LUID or GPU mask mismatch)");
  }

  return canInteroperate;
}

void dissociateD3D11Device(const Device* device) {
  // Currently no cleanup needed
  // Future: may need to track associated devices and release resources
}

bool Export(const Memory* memory, ID3D11Resource* d3d11Resource,
            UINT subresource, hsa_handle_t* handle, int* offset) {
  if (!memory || !d3d11Resource || !handle || !offset) {
    return false;
  }

  // Query DXGI resource interface to get shared handle
  IDXGIResource* pDxgiRes = nullptr;
  HRESULT hr = d3d11Resource->QueryInterface(__uuidof(IDXGIResource), (void**)&pDxgiRes);
  if (FAILED(hr) || !pDxgiRes) {
    LogError("Failed to query IDXGIResource from D3D11 resource");
    return false;
  }

  // Get shared handle
  HANDLE hShared = nullptr;
  hr = pDxgiRes->GetSharedHandle(&hShared);
  pDxgiRes->Release();

  if (FAILED(hr) || !hShared) {
    LogError("Failed to get shared handle from D3D11 resource");
    return false;
  }

  // Cast to HSA handle (platform-specific handle type)
  *handle = reinterpret_cast<hsa_handle_t>(hShared);
  *offset = 0;  // D3D resources are typically zero-offset

  return true;
}

}  // namespace D3D11Interop
}  // namespace roc
}  // namespace amd

#endif  // _WIN32
