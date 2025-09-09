/*

Copyright © 2025 Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#ifndef HSA_RUNTIME_IMAGE_INC_AMD_MIPMAP_IMPL_H_
#define HSA_RUNTIME_IMAGE_INC_AMD_MIPMAP_IMPL_H_

#include "inc/hsa_ext_amd.h"
#include "inc/hsa_amd_mipmap.h"
#include "core/inc/amd_gpu_agent.h"
#include "addrlib/inc/addrinterface.h"
#include "addrlib/inc/addrtypes.h"

namespace rocr { namespace image {

// Use the same metadata info type from public header
using hsa_amd_mipmap_metadata_info_t = struct hsa_amd_mipmap_metadata_info_s;

class ImageRuntime;

//  Mipmapped Array helper
// Address resource type from geometry.
AddrResourceType GetAddrResourceType(hsa_ext_image_geometry_t geometry);

// Translate HSA image format to AddrLib format.
AddrFormat HsaToAddrFormat(const hsa_ext_image_format_t& format);

// Compute bits-per-pixel from HSA format (channel order  channel type).
// Returns 0 on unsupported combination
uint32_t ComputeBitsPerPixel(const hsa_ext_image_format_t& format);

// Get swizzle mode using ADDR2_GET_PREFERRED_SURF_SETTING
AddrSwizzleMode GetOptimalSwizzleMode(
    ADDR_HANDLE addr_lib,
    const ADDR2_COMPUTE_SURFACE_INFO_INPUT& base_input);

// Populate AddrLib input from descriptor + requested mip levels
void PopulateAddrInput(
    ADDR2_COMPUTE_SURFACE_INFO_INPUT* pIn,
    const hsa_ext_image_descriptor_t* desc,
    uint32_t num_mipmap_levels,
    hsa_agent_t agent,
    hsa_access_permission_t access_permission,
    bool is_tiled);

// Compute surface info using Addr3 for GFX12+ architectures
#ifdef ADDR_GFX12_BUILD
ADDR_E_RETURNCODE ComputeSurfaceInfoAddr3(
    ADDR_HANDLE addr_lib,
    hsa_agent_t agent,
    const hsa_ext_image_descriptor_t* desc,
    uint32_t num_mipmap_levels,
    hsa_access_permission_t access_permission,
    bool is_tiled,
    ADDR3_COMPUTE_SURFACE_INFO_OUTPUT* addr_out);
#endif

// Architecture check for Addr3 usage
bool UseAddr3ForArchitecture(uint32_t gfx_major, uint32_t gfx_minor);
// Existing image extension dispatch retrieval.
hsa_status_t GetImageExtensionTable(hsa_ext_images_1_pfn_t* table);

 } 
} // namespace rocr::image
#endif  // HSA_RUNTIME_IMAGE_INC_AMD_MIPMAP_IMPL_H_
