////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2022, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////
#ifndef HSA_RUNTIME_CORE_INC_AMD_MIPMAP_H_
#define HSA_RUNTIME_CORE_INC_AMD_MIPMAP_H_

#include "hsa_ext_image.h"


#ifdef __cplusplus
extern "C" {
#endif

/**
 * Get the image extension function table (images major extension v1).
 */
hsa_status_t GetImageExtensionTable(hsa_ext_images_1_pfn_t* table);

// ----------------------------------------------------------------------------
// Mipmapped Array Query & Validated Create API (Issue 3)
// ----------------------------------------------------------------------------
// New query structure returned by hsa_amd_mipmap_array_get_info.
// size        : Total bytes required (surface size of all requested levels).
// alignment   : Required base address alignment (bytes).
// max_levels  : Maximum legal levels derivable from descriptor (power-of-two chain).
// levels_used : The level count validated (must match requested on success).
//
// NOTE (Issue 4): Bits-per-pixel, swizzle mode, and tiling parameters used
// internally to compute size/alignment are now derived from the descriptor
// and agent characteristics; callers no longer rely on fixed defaults.
typedef struct hsa_amd_mipmap_array_info_s {
  size_t    size;
  size_t    alignment;
  uint32_t  max_levels;
  uint32_t  levels_used;
} hsa_amd_mipmap_array_info_t;

/**
 * Query the total surface size and base address alignment for a mipmapped array.
 *
 * Required:
 *   agent              : GPU agent that supports images
 *   desc               : Non-null image descriptor (base level dimensions, format, geometry)
 *   requested_levels   : > 0 and <= max derivable mip levels (power-of-two chain)
 *   layout             : HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR or _TILED
 *   info               : Output pointer (non-null)
 *
 * On success:
 *   info->size         : Total bytes encompassing all requested mip levels
 *   info->alignment    : Required alignment for the base address
 *   info->max_levels   : Maximum legal mip levels for descriptor (independent of request)
 *   info->levels_used  : Echo of requested_levels (no clamping on success path)
 *
 * This call is optional: hsa_amd_mipmap_array_create re-validates and recomputes
 * authoritative size and alignment. Applications may use the query to size
 * their backing allocation before calling create.
 *
 * Implementation notes (Issue 4):
 *   - Bits-per-pixel is computed from (channel_order, channel_type).
 *   - Swizzle / tiling mode is heuristically chosen per agent and layout.
 *   - Conservative defaults are used when detailed ASIC data is unavailable.
 *
 * Returns:
 *   HSA_STATUS_SUCCESS
 *   HSA_STATUS_ERROR_INVALID_ARGUMENT              (null pointers, zero levels, levels > max)
 *   HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED  (unsupported format/geometry)
 *   HSA_EXT_STATUS_ERROR_IMAGE_SIZE_UNSUPPORTED    (dimensions exceed hardware limits)
 *   HSA_STATUS_ERROR_OUT_OF_RESOURCES              (AddrLib failure / internal allocation issues)
 */

hsa_status_t HSA_API
hsa_amd_mipmap_array_get_info(
  hsa_agent_t agent,
  const hsa_ext_image_descriptor_t* desc,
  uint32_t requested_levels,
  hsa_ext_image_data_layout_t layout,
  size_t row_pitch,
  size_t slice_pitch,
  hsa_amd_mipmap_array_info_t* info);

 /**
  * Create a mipmapped array handle referencing user-provided backing storage.
  *
  * Runtime steps:
  *   1. Recompute size & alignment (same logic as query) and validate num_mipmap_levels.
  *   2. Validate image_data pointer alignment.
  *   3. Validate image_data_size >= required surface size.
  *   4. Populate SRD and internal per-level metadata (ADDR2_MIP_INFO array).
  *
  * The runtime NEVER allocates pixel storage for the mip chain; ownership of
  * image_data remains with the caller. The handle encapsulates metadata/SRD only.
  *
  * Returns the same error codes as the query plus:
  *   HSA_STATUS_ERROR_INVALID_ALLOCATION (insufficient size or misaligned base)
  */
hsa_status_t HSA_API
hsa_amd_mipmap_array_create(
  hsa_agent_t agent,
  const hsa_ext_image_descriptor_t* desc,
  const void* image_data,
  hsa_access_permission_t access_permission,
  uint32_t num_mipmap_levels,
  hsa_ext_image_t* image_out);

 /**
  * Destroy a mipmapped array handle created via hsa_amd_mipmap_array_create.
  * (Does not free user pixel memory.)
  */
hsa_status_t HSA_API
hsa_amd_mipmap_array_destroy(const hsa_ext_image_t* image);
// ----------------------------------------------------------------------------
// Issue 5: Per-level view creation (New API)
// ----------------------------------------------------------------------------
/**
 * Create an image view for a specific mip level of a mipmapped array.
 *
 * The returned image handle is a lightweight view referencing the SAME backing
 * memory (no allocation/copy). Destroy with the normal image destroy API.
 *
 * Parameters:
 *   agent            : GPU agent
 *   mipmapped_array  : Pointer to the mipmapped array handle previously created
 *                      by hsa_amd_mipmap_array_create
 *   mip_level        : Level index (0 = base). Must be < array's num levels.
 *   level_image_out  : Output image handle for the level view
 *
 * Notes:
 *   - Dimensions are clamped to at least 1 when shifting (right shift per level).
 *   - Row/slice pitches follow underlying layout; for tiled images internal
 *     SRD setup derives pitches; for linear layout the base pitches may
 *     be adjusted if required per level (future enhancement).
 *   - The view inherits access permissions from the parent array.
 *
 * Returns:
 *   HSA_STATUS_SUCCESS
 *   HSA_STATUS_ERROR_INVALID_ARGUMENT (null pointers, bad level, bad handle)
 *   HSA_STATUS_ERROR_OUT_OF_RESOURCES (allocation of view metadata failed)
 */
hsa_status_t HSA_API
hsa_amd_mipmap_array_get_level(
    hsa_agent_t agent,
    const hsa_ext_image_t* mipmapped_array,
    uint32_t mip_level,
    hsa_ext_image_t* level_image_out);


#ifdef __cplusplus
}
#endif

#endif  // HSA_RUNTIME_CORE_INC_AMD_MIPMAP_H_
