/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#ifndef ROC_D3D11_INTEROP_HPP_
#define ROC_D3D11_INTEROP_HPP_

#include "top.hpp"

#ifdef _WIN32

#include <d3d11.h>
#include <dxgi.h>

namespace amd {
namespace roc {

// Forward declarations
class Device;
class Memory;

namespace D3D11Interop {

/**
 * @brief Validate D3D11 device matches ROCr GPU device
 *
 * Performs validation: LUID matching via DXGI adapter
 *
 * @param device ROCr device to validate against
 * @param d3d11Device D3D11 device to associate
 * @return true if devices can interoperate, false otherwise
 */
bool associateD3D11Device(
    const Device* device,
    ID3D11Device* d3d11Device
);

/**
 * @brief Dissociate D3D11 device from ROCr device
 *
 * Cleanup function called during context destruction
 *
 * @param device ROCr device
 */
void dissociateD3D11Device(const Device* device);

/**
 * @brief Export D3D11 resource to HSA handle for interop
 *
 * Extracts shared handle from D3D11 resource and returns it
 * as HSA handle for memory mapping
 *
 * @param memory ROCr memory object
 * @param d3d11Resource D3D11 resource to export
 * @param subresource Subresource index (for textures with mips)
 * @param handle Output HSA handle
 * @param offset Output offset into resource
 * @return true if export succeeded, false otherwise
 */
bool Export(
    const Memory* memory,
    ID3D11Resource* d3d11Resource,
    UINT subresource,
    hsa_handle_t* handle,
    int* offset,
    void* srd,
    UINT* srdSize,
    hsa_interop_map_flag_t* mapFlags,
    size_t* sizeHint = nullptr
);

}  // namespace D3D11Interop
}  // namespace roc
}  // namespace amd

#endif  // _WIN32

#endif  // ROC_D3D11_INTEROP_HPP_
