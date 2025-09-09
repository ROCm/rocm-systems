/*

Copyright © 2025 Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#ifndef HSA_RUNTIME_CORE_INC_AMD_MIPMAP_H_
#define HSA_RUNTIME_CORE_INC_AMD_MIPMAP_H_

#include "hsa_ext_image.h"
#include <cstring>

#ifdef __cplusplus
extern "C" {
#endif

hsa_status_t GetImageExtensionTable(hsa_ext_images_1_pfn_t* table);

typedef struct hsa_amd_mipmap_array_info_s {
  size_t    size;
  size_t    alignment;
  uint32_t  max_levels;
  uint32_t  levels_used;
} hsa_amd_mipmap_array_info_t;

hsa_status_t HSA_API
hsa_amd_mipmap_array_get_info(
  hsa_agent_t agent,
  const hsa_ext_image_descriptor_t* desc,
  uint32_t requested_levels,
  hsa_ext_image_data_layout_t layout,
  size_t row_pitch,
  size_t slice_pitch,
  hsa_amd_mipmap_array_info_t* info);

hsa_status_t HSA_API
hsa_amd_mipmap_array_create(
  hsa_agent_t agent,
  const hsa_ext_image_descriptor_t* desc,
  const void* image_data,
  size_t image_data_size,
  hsa_access_permission_t access_permission,
  uint32_t num_mipmap_levels,
  hsa_ext_image_data_layout_t layout,
  size_t row_pitch,
  size_t slice_pitch,
  hsa_ext_image_t* image_out);

hsa_status_t HSA_API
hsa_amd_mipmap_array_destroy(const hsa_ext_image_t* image);

hsa_status_t HSA_API
hsa_amd_mipmap_array_get_level(
    hsa_agent_t agent,
    const hsa_ext_image_t* mipmapped_array,
    uint32_t mip_level,
    hsa_ext_image_t* level_image_out);

hsa_status_t HSA_API
hsa_amd_mipmap_get_level_info(
    hsa_agent_t agent,
    const hsa_ext_image_t* mipmapped_array,
    uint32_t mip_level,
    hsa_ext_image_descriptor_t* level_desc);

hsa_status_t HSA_API
hsa_amd_mipmap_get_level_pitch(
    hsa_agent_t agent,
    const hsa_ext_image_t* mipmapped_array,
    uint32_t mip_level,
    size_t* row_pitch,
    size_t* slice_pitch);

hsa_status_t HSA_API
hsa_amd_mipmap_coord_to_address(
    const hsa_ext_image_t* mipmapped_array,
    uint32_t mip_level,
    const uint32_t coords[3],
    void** address);

hsa_status_t HSA_API
hsa_amd_mipmap_get_all_level_info(
    hsa_agent_t agent,
    const hsa_ext_image_t* mipmapped_array,
    hsa_ext_image_descriptor_t* level_descriptors,
    size_t* row_pitches,
    size_t* slice_pitches,
    uint32_t max_levels);

typedef struct hsa_amd_mipmap_metadata_info_s {
    size_t metadata_size;
    size_t cmask_size;
    size_t cmask_alignment;
    bool has_metadata;
} hsa_amd_mipmap_metadata_info_t;

hsa_status_t HSA_API
hsa_amd_mipmap_compute_metadata_info(
    hsa_agent_t agent,
    const hsa_ext_image_t* mipmapped_array,
    uint32_t mip_level,
    hsa_amd_mipmap_metadata_info_t* metadata_info);

hsa_status_t HSA_API
hsa_amd_mipmap_get_metadata_info(
    hsa_agent_t agent,
    const hsa_ext_image_t* mipmapped_array,
    uint32_t mip_level,
    size_t* metadata_size);

#ifdef __cplusplus
}

static inline hsa_status_t hsa_amd_mipmap_array_create(
  hsa_agent_t agent,
  const hsa_ext_image_descriptor_t* desc,
  uint32_t num_mipmap_levels,
  hsa_access_permission_t access_permission,
  hsa_ext_image_t* image_out) {
    return hsa_amd_mipmap_array_create(
        agent, desc, /*image_data*/nullptr, /*image_data_size*/0,
        access_permission, num_mipmap_levels,
        HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR, 0, 0, image_out);
}

static inline hsa_status_t hsa_amd_mipmap_array_destroy(
  hsa_agent_t /*agent*/, hsa_ext_image_t* image) {
    if (!image) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    hsa_ext_image_t tmp = *image;
    hsa_status_t s = hsa_amd_mipmap_array_destroy(&tmp);
    if (s == HSA_STATUS_SUCCESS) {
      memset(image, 0, sizeof(*image));
    }
    return s;
}

static inline hsa_status_t hsa_amd_mipmap_get_level(
    hsa_agent_t agent,
    hsa_ext_image_t mipmapped_array,
    uint32_t mip_level,
    hsa_ext_image_t* level_image_out) {
    return hsa_amd_mipmap_array_get_level(agent, &mipmapped_array, mip_level, level_image_out);
}

static inline hsa_status_t hsa_amd_mipmap_array_get_info(
    hsa_agent_t agent,
    const hsa_ext_image_descriptor_t* desc,
    uint32_t requested_levels,
    hsa_amd_mipmap_array_info_t* info) {
  return hsa_amd_mipmap_array_get_info(
      agent, desc, requested_levels,
      HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR,
      /*row_pitch*/0, /*slice_pitch*/0, info);
}
#endif

#endif  // HSA_RUNTIME_CORE_INC_AMD_MIPMAP_H_
