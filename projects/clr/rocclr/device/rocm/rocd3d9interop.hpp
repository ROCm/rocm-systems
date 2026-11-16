/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#ifndef ROC_D3D9_INTEROP_HPP_
#define ROC_D3D9_INTEROP_HPP_

#include "top.hpp"

#ifdef _WIN32

#include <d3d9.h>

namespace amd {

class D3D9Object;

namespace roc {

// Forward declarations
class Device;
class Memory;

namespace D3D9Interop {

//! Which cl_dx9_media_adapter_type_khr a context slot was populated from. The three
//! adapter types hand clCreateContext three unrelated object types, so the caller must
//! say which one it holds rather than letting us guess from the pointer.
enum class AdapterKind {
  D3D9,    //!< CL_ADAPTER_D3D9_KHR   -> IDirect3DDevice9*
  D3D9Ex,  //!< CL_ADAPTER_D3D9EX_KHR -> IDirect3DDevice9Ex*
  DXVA     //!< CL_ADAPTER_DXVA_KHR   -> IDXVAHD_Device*
};

/**
 * @brief Validate a D3D9 adapter against a ROCr GPU device
 *
 * Performs LUID matching via IDirect3D9Ex::GetAdapterLUID. A DXVA adapter supplies an
 * IDXVAHD_Device, which exposes no adapter identity, so it is accepted without a LUID
 * check - the surfaces it produces still carry their own owning D3D9 device, which is
 * what the export path uses. This matches the PAL backend, which does not handle the
 * DXVA slot at all.
 *
 * @param device ROCr device to validate against
 * @param gfxDevice The device pointer stored in the OpenCL context
 * @param kind Which adapter type gfxDevice came from
 * @param gfxContext Owning amd::Context*, used to cache the DXX extension
 * @param validateOnly When true, only check adapter compatibility and skip
 *        creating/caching the DXX extension objects (used by query paths)
 * @return true if the devices can interoperate, false otherwise
 */
bool associateD3D9Device(
    const Device* device,
    void* gfxDevice,
    AdapterKind kind,
    void* gfxContext,
    bool validateOnly
);

/**
 * @brief Release every DXX extension object cached for D3D9 in this context
 *
 * Cleanup function called during context destruction.
 *
 * @param device ROCr device
 * @param gfxDevice The context's device-handle array
 * @param gfxContext Owning amd::Context*
 */
void dissociateD3D9Device(const Device* device, void* const gfxDevice[], void* gfxContext);

/**
 * @brief Export a D3D9 surface to an HSA handle for interop
 *
 * Returns the surface's KMT shared handle for memory mapping and, for image interop,
 * queries the driver SRD through the DXX CL interop extension.
 *
 * @param memory ROCr memory object
 * @param d3d9Obj D3D9 object to export
 * @param handle Output HSA handle
 * @param offset Output offset into the resource (non-zero for YUV sub-planes)
 * @param srd Output driver SRD, may be null for buffer interop
 * @param srdSize In/out SRD size in bytes, may be null for buffer interop
 * @param mapFlags Output interop map flags
 * @return true if export succeeded, false otherwise
 */
bool Export(
    const Memory* memory,
    D3D9Object* d3d9Obj,
    hsa_handle_t* handle,
    int* offset,
    void* srd,
    UINT* srdSize,
    hsa_interop_map_flag_t* mapFlags
);

}  // namespace D3D9Interop
}  // namespace roc
}  // namespace amd

#endif  // _WIN32

#endif  // ROC_D3D9_INTEROP_HPP_
