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

#include "DxxOpenCLInteropExt.h"
#include "platform/interop_d3d10.hpp"

namespace amd {
namespace roc {
namespace D3D10Interop {

/**
 * @brief Query GPU mask from D3D10 device using AMD DXX extensions
 *
 * Loads atidxx64.dll (or atidxx32.dll) and uses AMD DirectX extension
 * interface to query which GPU(s) in a multi-GPU chain can interoperate
 * with the given D3D10 device.
 *
 * @param pd3d10Device D3D10 device
 * @param pd3d10DeviceGPUMask Output bitmask of GPU indices
 * @return true if GPU mask was successfully queried
 */
static bool queryD3D10DeviceGPUMask(ID3D10Device* pd3d10Device, UINT* pd3d10DeviceGPUMask) {
  IAmdDxExt* pExt = nullptr;
  IAmdDxExtCLInterop* pCLExt = nullptr;
  PFNAmdDxExtCreate AmdDxExtCreate;
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

  // Get the exported AmdDxExtCreate() function pointer
  if (SUCCEEDED(hr)) {
    AmdDxExtCreate =
        reinterpret_cast<PFNAmdDxExtCreate>(GetProcAddress(hDLL, "AmdDxExtCreate"));
    if (AmdDxExtCreate == nullptr) {
      hr = E_FAIL;
    }
  }

  // Create the extension object
  if (SUCCEEDED(hr)) {
    hr = AmdDxExtCreate(pd3d10Device, &pExt);
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
      pCLExt->QueryInteropGpuMask(pd3d10DeviceGPUMask);
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

bool associateD3D10Device(const Device* device, ID3D10Device* pd3d10Device) {
  if (!device || !pd3d10Device) {
    return false;
  }

  // Verify device has valid LUID
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
  if (FAILED(hr) || !pDXGIAdapter) {
    pDXGIDevice->Release();
    LogError("Failed to get DXGI adapter from D3D10 device");
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
    UINT d3d10DeviceGPUMask = 0;
    UINT chainBitMask = 1 << device->getGpuIndex();

    if (queryD3D10DeviceGPUMask(pd3d10Device, &d3d10DeviceGPUMask)) {
      canInteroperate = (chainBitMask & d3d10DeviceGPUMask) != 0;
    } else {
      // Special handling for Intel iGPU + AMD dGPU in LDA mode
      if (chainBitMask > 1) {
        canInteroperate = false;
      }
      // If we're GPU 0 and can't query mask, assume it's OK
    }
  }

  pDXGIDevice->Release();
  pDXGIAdapter->Release();

  if (!canInteroperate) {
    LogError("D3D10 device and ROCr device cannot interoperate (LUID or GPU mask mismatch)");
  }

  return canInteroperate;
}

void dissociateD3D10Device(const Device* device) {
  // Currently no cleanup needed
  // Future: may need to track associated devices and release resources
}

bool Export(const Memory* memory, ID3D10Resource* d3d10Resource,
            UINT subresource, hsa_handle_t* handle, int* offset) {
  if (!memory || !d3d10Resource || !handle || !offset) {
    return false;
  }

  // Query DXGI resource interface to get shared handle
  IDXGIResource* pDxgiRes = nullptr;
  HRESULT hr = d3d10Resource->QueryInterface(__uuidof(IDXGIResource), (void**)&pDxgiRes);
  if (FAILED(hr) || !pDxgiRes) {
    LogError("Failed to query IDXGIResource from D3D10 resource");
    return false;
  }

  // Get shared handle
  HANDLE hShared = nullptr;
  hr = pDxgiRes->GetSharedHandle(&hShared);
  pDxgiRes->Release();

  if (FAILED(hr) || !hShared) {
    LogError("Failed to get shared handle from D3D10 resource");
    return false;
  }

  // Cast to HSA handle (platform-specific handle type)
  *handle = reinterpret_cast<hsa_handle_t>(hShared);
  *offset = 0;  // D3D resources are typically zero-offset

  return true;
}

}  // namespace D3D10Interop
}  // namespace roc
}  // namespace amd

#endif  // _WIN32
