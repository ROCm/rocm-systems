/*

Copyright © 2025 Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#ifndef HSA_RUNTIME_CORE_INC_AMD_MIPMAP_H_
#define HSA_RUNTIME_CORE_INC_AMD_MIPMAP_H_

#include "inc/hsa_ext_amd.h"
#include "inc/hsa_ext_image.h"
#include "core/inc/amd_gpu_agent.h"
#include "addrlib/inc/addrinterface.h"

namespace rocr {
namespace image {

//Mipmapped image array
struct MipmappedArray {
  hsa_ext_image_t image;                // Base level image handle
  void* data;                           // Pointer to the allocated memory
  size_t size;                          // Total size of the allocated memory
  hsa_ext_image_descriptor_t descriptor;// Image descriptor for the base level
  hsa_access_permission_t permission;   // Access permissions
  uint32_t num_levels;                  // Number of mipmap levels
  uint32_t flags;                       // Reserved for future use
  ADDR_HANDLE addr_handle;              // Address library handle
  ADDR2_COMPUTE_SURFACE_INFO_OUTPUT addr_output; // Cached surface info
};

typedef MipmappedArray* MipmappedArray_t;

//Metadata information for depth surfaces
typedef struct hsa_amd_mipmap_metadata_info_s {
  size_t htile_size;                    // HTile metadata size in bytes
  size_t htile_alignment;               // HTile alignment requirement
  size_t cmask_size;                    // CMask metadata size in bytes
  size_t cmask_alignment;               // CMask alignment requirement
  bool has_metadata;                    // Whether metadata is available
} hsa_amd_mipmap_metadata_info_t;

} // namespace image
} // namespace rocr


#ifdef __cplusplus
extern "C" {
#endif

hsa_status_t hsa_amd_mipmap_array_create(
    hsa_agent_t agent_handle,
    const hsa_ext_image_descriptor_t* mipmap_desc,
    uint32_t num_mipmap_levels,
    hsa_access_permission_t access_permission,
    MipmappedArray_t* mipmapped_array_ptr);

hsa_status_t hsa_amd_mipmap_array_destroy(
    hsa_agent_t agent,
    MipmappedArray_t mipmapped_array);

hsa_status_t hsa_amd_mipmap_get_level(
    hsa_agent_t agent,
    MipmappedArray_t mipmapped_array,
    uint32_t mip_level,
    hsa_ext_image_t* level_image_out);

hsa_status_t hsa_amd_mipmap_compute_metadata_info(
    hsa_agent_t agent,
    const MipmappedArray_t mipmapped_array,
    uint32_t mip_level,
    hsa_amd_mipmap_metadata_info_t* metadata_info);

#ifdef __cplusplus
}
#endif

#endif  // HSA_RUNTIME_CORE_INC_AMD_MIPMAP_H_
