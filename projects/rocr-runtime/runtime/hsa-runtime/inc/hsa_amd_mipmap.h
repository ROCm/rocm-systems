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
 * @brief Get the image extension function table (images major extension v1).
 */
hsa_status_t GetImageExtensionTable(hsa_ext_images_1_pfn_t* table);

/**
 * @brief Query structure for mipmapped array information.
 * 
 * @param size          Total bytes required (surface size of all requested levels).
 * @param alignment     Required base address alignment (bytes).
 * @param max_levels    Maximum legal levels derivable from descriptor (power-of-two chain).
 * @param levels_used   The level count validated (must match requested on success).
 */
typedef struct hsa_amd_mipmap_array_info_s {
  size_t    size;
  size_t    alignment;
  uint32_t  max_levels;
  uint32_t  levels_used;
} hsa_amd_mipmap_array_info_t;

/**
 * @brief Query the total surface size and base address alignment for a mipmapped array.
 *
 * @param[in] agent               : GPU agent
 * @param[in] desc                : Image descriptor for the base level
 * @param[in] requested_levels    : Number of mip levels requested
 * @param[in] layout              : Layout of the backing image data
 * @param[in] row_pitch           : Row pitch of the backing image data (bytes)
 * @param[in] slice_pitch         : Slice pitch of the backing image data (bytes)
 * @param[out] info               : Output info structure (see returns)
 *
 * @details On success, populates the info structure as follows:
 *   info->size         : Total bytes encompassing all requested mip levels
 *   info->alignment    : Required alignment for the base address
 *   info->max_levels   : Maximum legal mip levels for descriptor (independent of request)
 *   info->levels_used  : Echo of requested_levels (no clamping on success path)
 *
 * @remark This call is optional: hsa_amd_mipmap_array_create re-validates and
 * recomputes authoritative size and alignment. Applications may use the query to
 * size their backing allocation before calling create.
 *
 * @retval HSA_STATUS_SUCCESS
 * @retval HSA_STATUS_ERROR_INVALID_ARGUMENT              (null pointers, zero levels, levels > max)
 * @retval HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED  (unsupported format/geometry)
 * @retval HSA_EXT_STATUS_ERROR_IMAGE_SIZE_UNSUPPORTED    (dimensions exceed hardware limits)
 * @retval HSA_STATUS_ERROR_OUT_OF_RESOURCES              (AddrLib failure / internal allocation issues)
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
 * @brief Create a mipmapped array handle referencing user-provided backing storage.
 *
 * @param[in] agent                : GPU agent
 * @param[in] desc                 : Image descriptor for the base level
 * @param[in] image_data           : Pointer to user-allocated backing storage
 * @param[in] access_permission    : Access permission for the image
 * @param[in] num_mipmap_levels    : Number of mip levels in the array
 * @param[out] image_out           : Output handle to the created mipmapped array
 *
 * @details Runtime steps:
 *   1. Recompute size & alignment and validate num_mipmap_levels.
 *   2. Validate image_data pointer alignment.
 *   3. Validate image_data_size >= required surface size.
 *   4. Populate SRD and internal per-level metadata.
 *
 * @note The runtime NEVER allocates pixel storage for the mip chain; ownership of
 * image_data remains with the caller. The handle encapsulates metadata/SRD only.
 *
 * @retval HSA_STATUS_SUCCESS
 * @retval HSA_STATUS_ERROR_INVALID_ARGUMENT              (null pointers, zero levels, levels > max)
 * @retval HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED  (unsupported format/geometry)
 * @retval HSA_EXT_STATUS_ERROR_IMAGE_SIZE_UNSUPPORTED    (dimensions exceed hardware limits)
 * @retval HSA_STATUS_ERROR_INVALID_ALLOCATION            (insufficient size or misaligned base)
 * @retval HSA_STATUS_ERROR_OUT_OF_RESOURCES              (AddrLib failure / internal allocation issues)
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
 * @brief Destroy a mipmapped array handle created via hsa_amd_mipmap_array_create.
 * (Does not free user pixel memory.)
 * 
 * @param image : Pointer to the mipmapped array handle to destroy
 */
hsa_status_t HSA_API
hsa_amd_mipmap_array_destroy(const hsa_ext_image_t* image);

/**
 * @brief Create an image view for a specific mip level of a mipmapped array.
 *
 * @param[in] agent             : GPU agent
 * @param[in] mipmapped_array   : Pointer to the mipmapped array handle previously
 *                                created by hsa_amd_mipmap_array_create
 * @param[in] mip_level         : Level index (0 = base). Must be < array's num levels.
 * @param[out] level_image_out  : Output image handle for the level view
 *
 * @details
 *   - Dimensions are clamped to at least 1 when shifting (right shift per level).
 *   - Row/slice pitches follow underlying layout; for tiled images internal
 *     SRD setup derives pitches; for linear layout the base pitches may
 *     be adjusted if required per level (future enhancement).
 *   - The view inherits access permissions from the parent array.
 * 
 * @retval HSA_STATUS_SUCCESS
 * @retval HSA_STATUS_ERROR_INVALID_ARGUMENT (null pointers, bad level, bad handle)
 * @retval HSA_STATUS_ERROR_OUT_OF_RESOURCES (allocation of view metadata failed)
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
