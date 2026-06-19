/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "paldevice.hpp"

#if defined(ATI_OS_LINUX)
namespace amd::pal {
bool Device::associateD3D9Device(void* d3d9Device) { return false; }
}  // namespace amd::pal
#else  // !ATI_OS_LINUX

#include <d3d9.h>
#include <dxgi.h>

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

bool Device::associateD3D9Device(void* d3d9Device) {
  D3DCAPS9 pCaps;
  IDirect3D9* p3d9dev;
  LUID d3d9deviceLuid = {0, 0};

  IDirect3DDevice9* pd3d9Device = static_cast<IDirect3DDevice9*>(d3d9Device);

  // Get D3D9 Device caps
  pd3d9Device->GetDeviceCaps(&pCaps);
  // Get 3D9 Device
  pd3d9Device->GetDirect3D(&p3d9dev);

  IDirect3D9Ex* p3d9devEx = static_cast<IDirect3D9Ex*>(p3d9dev);
  p3d9devEx->GetAdapterLUID(pCaps.AdapterOrdinal, &d3d9deviceLuid);
  p3d9dev->Release();

  // match the adapter
  bool canInteroperate = (properties().osProperties.luidHighPart == d3d9deviceLuid.HighPart) &&
                         (properties().osProperties.luidLowPart == d3d9deviceLuid.LowPart);

  return canInteroperate;
}

}  // namespace amd::pal
#endif  // !ATI_OS_WIN
