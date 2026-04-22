/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "paldevice.hpp"

#if defined(ATI_OS_LINUX)
namespace amd::pal {
bool Device::associateD3D10Device(void* d3d10Device) { return false; }
}  // namespace amd::pal
#else  // !ATI_OS_WIN

#include <D3D10_1.h>

/**************************************************************************************************************
 * Note: ideally the DXX extension interfaces should be mapped from the DXX perforce branch.
 * This means OCL client spec will need to change to include headers directly from the DXX perforce
 *tree.
 * However, OCL only cares about the DXX OpenCL extension interface class. The spec cannot change
 * without notification. So it is safe to use a local copy of the relevant DXX extension interface
 *classes.
 **************************************************************************************************************/
#include "DxxOpenCLInteropExt.h"

namespace amd::pal {

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
    AmdDxExtCreate = reinterpret_cast<PFNAmdDxExtCreate>(GetProcAddress(hDLL, "AmdDxExtCreate"));
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

bool Device::associateD3D10Device(void* d3d10Device) {
  ID3D10Device* pd3d10Device = static_cast<ID3D10Device*>(d3d10Device);

  IDXGIDevice* pDXGIDevice;
  pd3d10Device->QueryInterface(__uuidof(IDXGIDevice), (void**)&pDXGIDevice);

  IDXGIAdapter* pDXGIAdapter;
  pDXGIDevice->GetAdapter(&pDXGIAdapter);

  DXGI_ADAPTER_DESC adapterDesc;
  pDXGIAdapter->GetDesc(&adapterDesc);

  // match the adapter
  bool canInteroperate =
      (properties().osProperties.luidHighPart == adapterDesc.AdapterLuid.HighPart) &&
      (properties().osProperties.luidLowPart == adapterDesc.AdapterLuid.LowPart);

  UINT chainBitMask = 1 << properties().gpuIndex;

  // match the chain ID
  if (canInteroperate) {
    UINT d3d10DeviceGPUMask = 0;

    if (queryD3D10DeviceGPUMask(pd3d10Device, &d3d10DeviceGPUMask)) {
      canInteroperate = (chainBitMask & d3d10DeviceGPUMask) != 0;
    } else {
      // special handling for Intel iGPU + AMD dGPU in LDA mode
      // (only occurs on a PX platform) where
      // the D3D10Device object is created on the Intel iGPU and
      // passed to AMD dGPU (secondary) to interoperate.
      if (chainBitMask > 1) {
        canInteroperate = false;
      }
    }
  }

  pDXGIDevice->Release();
  pDXGIAdapter->Release();

  return canInteroperate;
}

}  // namespace amd::pal

#endif  // !ATI_OS_WIN
