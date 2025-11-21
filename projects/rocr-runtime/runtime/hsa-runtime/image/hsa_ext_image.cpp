////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
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

#include "image_runtime.h"
#include "image/inc/hsa_ext_image_impl.h"
#include "core/inc/exceptions.h"

namespace rocr {

namespace AMD {
hsa_status_t handleException();

template <class T> static __forceinline T handleExceptionT() {
  handleException();
  abort();
  return T();
}
}   // namespace amd

#define TRY try {
#define CATCH } catch(...) { return AMD::handleException(); }
#define CATCHRET(RETURN_TYPE) } catch(...) { return AMD::handleExceptionT<RETURN_TYPE>(); }

namespace image {

//---------------------------------------------------------------------------//
//  Utilty routines
//---------------------------------------------------------------------------//
static void enforceDefaultPitch(hsa_agent_t agent,
                                const hsa_ext_image_descriptor_t* image_descriptor,
                                size_t& image_data_row_pitch, size_t& image_data_slice_pitch) {
  // Set default pitch
  if (image_data_row_pitch == 0) {
    auto manager = ImageRuntime::instance()->image_manager(agent);
    assert((manager != nullptr) && "Image manager should already exit.");
    image_data_row_pitch = image_descriptor->width *
      manager->GetImageProperty(agent, image_descriptor->format, image_descriptor->geometry)
      .element_size;
  }

  // Set default slice pitch
  if ((image_data_slice_pitch == 0) &&
    ((image_descriptor->depth != 0) || (image_descriptor->array_size != 0))) {
      switch (image_descriptor->geometry) {
      case HSA_EXT_IMAGE_GEOMETRY_3D:
      case HSA_EXT_IMAGE_GEOMETRY_2DA:
      case HSA_EXT_IMAGE_GEOMETRY_2DADEPTH: {
        image_data_slice_pitch = image_data_row_pitch * image_descriptor->height;
        break;
                                            }
      case HSA_EXT_IMAGE_GEOMETRY_1DA: {
        image_data_slice_pitch = image_data_row_pitch;
        break;
                                       }
      default:
        fprintf(stderr, "Depth set on single layer image geometry.\n");
        //assert(false && "Depth set on single layer image geometry.");
      }
  }
}

//---------------------------------------------------------------------------//
//  APIs that implement Image functionality
//---------------------------------------------------------------------------//

hsa_status_t hsa_amd_image_get_info_max_dim(hsa_agent_t agent, hsa_agent_info_t attribute,
                                            void* value) {
  TRY;
  if (agent.handle == 0) {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }

  if (value == NULL) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  return ImageRuntime::instance()->GetImageInfoMaxDimension(agent, attribute, value);
  CATCH;
}

hsa_status_t hsa_ext_image_get_capability(hsa_agent_t agent,
                                          hsa_ext_image_geometry_t image_geometry,
                                          const hsa_ext_image_format_t* image_format,
                                          uint32_t* capability_mask) {
  TRY;
  if (agent.handle == 0) {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }

  if ((image_format == NULL) || (capability_mask == NULL) ||
      (image_geometry < HSA_EXT_IMAGE_GEOMETRY_1D) ||
      (image_geometry > HSA_EXT_IMAGE_GEOMETRY_2DADEPTH)) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  return ImageRuntime::instance()->GetImageCapability(agent, *image_format, image_geometry,
                                                      *capability_mask);
  CATCH;
}

hsa_status_t hsa_ext_image_data_get_info(hsa_agent_t agent,
                                         const hsa_ext_image_descriptor_t* image_descriptor,
                                         hsa_access_permission_t access_permission,
                                         hsa_ext_image_data_info_t* image_data_info) {
  TRY;
  if (agent.handle == 0) {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }

  if ((image_descriptor == NULL) || (image_data_info == NULL) ||
      (access_permission < HSA_ACCESS_PERMISSION_RO) ||
      (access_permission > HSA_ACCESS_PERMISSION_RW)) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  return ImageRuntime::instance()->GetImageSizeAndAlignment(
      agent, *image_descriptor, HSA_EXT_IMAGE_DATA_LAYOUT_OPAQUE, 0, 0, *image_data_info);
  CATCH;
}

hsa_status_t hsa_ext_image_create(hsa_agent_t agent,
                                  const hsa_ext_image_descriptor_t* image_descriptor,
                                  const void* image_data, hsa_access_permission_t access_permission,
                                  hsa_ext_image_t* image) {
  TRY;
  if (agent.handle == 0) {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }

  if (image_descriptor == NULL || image_data == NULL || image == NULL) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  return ImageRuntime::instance()->CreateImageHandle(
      agent, *image_descriptor, image_data, access_permission, HSA_EXT_IMAGE_DATA_LAYOUT_OPAQUE, 0,
      0, *image);
  CATCH;
}

hsa_status_t hsa_ext_image_destroy(hsa_agent_t agent, hsa_ext_image_t image) {
  TRY;
  if (agent.handle == 0) {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }

  return ImageRuntime::instance()->DestroyImageHandle(image);
  CATCH;
}

hsa_status_t hsa_ext_image_copy(hsa_agent_t agent, hsa_ext_image_t src_image,
                                const hsa_dim3_t* src_offset, hsa_ext_image_t dst_image,
                                const hsa_dim3_t* dst_offset, const hsa_dim3_t* range) {
  TRY;
  if (agent.handle == 0) {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }

  if (src_image.handle == 0 || dst_image.handle == 0 || src_offset == NULL ||
      dst_offset == NULL || range == NULL) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  return ImageRuntime::instance()->CopyImage(src_image, dst_image, *src_offset, *dst_offset,
                                             *range);
  CATCH;
}

hsa_status_t hsa_ext_image_import(hsa_agent_t agent, const void* src_memory, size_t src_row_pitch,
                                  size_t src_slice_pitch, hsa_ext_image_t dst_image,
                                  const hsa_ext_image_region_t* image_region) {
  TRY;
  if (agent.handle == 0) {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }

  if (src_memory == NULL || dst_image.handle == 0 || image_region == NULL) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  return ImageRuntime::instance()->CopyBufferToImage(src_memory, src_row_pitch, src_slice_pitch,
                                                     dst_image, *image_region);
  CATCH;
}

hsa_status_t hsa_ext_image_export(hsa_agent_t agent, hsa_ext_image_t src_image, void* dst_memory,
                                  size_t dst_row_pitch, size_t dst_slice_pitch,
                                  const hsa_ext_image_region_t* image_region) {
  TRY;
  if (agent.handle == 0) {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }

  if (dst_memory == NULL || src_image.handle == 0 || image_region == NULL) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  return ImageRuntime::instance()->CopyImageToBuffer(src_image, dst_memory, dst_row_pitch,
                                                     dst_slice_pitch, *image_region);
  CATCH;
}

hsa_status_t hsa_ext_image_clear(hsa_agent_t agent, hsa_ext_image_t image, const void* data,
                                 const hsa_ext_image_region_t* image_region) {
  TRY;
  if (agent.handle == 0) {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }

  if (image.handle == 0 || image_region == NULL || data == NULL) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  return ImageRuntime::instance()->FillImage(image, data, *image_region);
  CATCH;
};

hsa_status_t hsa_ext_sampler_create(hsa_agent_t agent,
                                    const hsa_ext_sampler_descriptor_t* sampler_descriptor,
                                    hsa_ext_sampler_t* sampler) {
  TRY;
  if (agent.handle == 0) {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }

  if (sampler_descriptor == NULL || sampler == NULL) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  hsa_ext_sampler_descriptor_v2_t sampler_descriptor_v2 = {
      sampler_descriptor->coordinate_mode,
      sampler_descriptor->filter_mode,
      {sampler_descriptor->address_mode,
          sampler_descriptor->address_mode, sampler_descriptor->address_mode}
  };
  return ImageRuntime::instance()->CreateSamplerHandle(agent, sampler_descriptor_v2, *sampler);
  CATCH;
}

hsa_status_t hsa_ext_sampler_create_v2(hsa_agent_t agent,
                                    const hsa_ext_sampler_descriptor_v2_t* sampler_descriptor,
                                    hsa_ext_sampler_t* sampler) {
  TRY;
  if (agent.handle == 0) {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }

  if (sampler_descriptor == NULL || sampler == NULL) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  return ImageRuntime::instance()->CreateSamplerHandle(agent, *sampler_descriptor, *sampler);
  CATCH;
}

hsa_status_t hsa_ext_sampler_destroy(hsa_agent_t agent, hsa_ext_sampler_t sampler) {
  TRY;
  if (agent.handle == 0) {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }

  return ImageRuntime::instance()->DestroySamplerHandle(sampler);
  CATCH;
}

hsa_status_t hsa_ext_image_get_capability_with_layout(hsa_agent_t agent,
                                                      hsa_ext_image_geometry_t image_geometry,
                                                      const hsa_ext_image_format_t* image_format,
                                                      hsa_ext_image_data_layout_t image_data_layout,
                                                      uint32_t* capability_mask) {
  TRY;
  if (agent.handle == 0) {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }

  if ((image_format == NULL) || (capability_mask == NULL) ||
      (image_geometry < HSA_EXT_IMAGE_GEOMETRY_1D) ||
      (image_geometry > HSA_EXT_IMAGE_GEOMETRY_2DADEPTH) ||
      (image_data_layout != HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR)) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  return ImageRuntime::instance()->GetImageCapability(agent, *image_format, image_geometry,
                                                      *capability_mask);
  CATCH;
}

hsa_status_t hsa_ext_image_data_get_info_with_layout(
    hsa_agent_t agent, const hsa_ext_image_descriptor_t* image_descriptor,
    hsa_access_permission_t access_permission, hsa_ext_image_data_layout_t image_data_layout,
    size_t image_data_row_pitch, size_t image_data_slice_pitch,
    hsa_ext_image_data_info_t* image_data_info) {
  TRY;
  if (agent.handle == 0) {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }

  if ((image_descriptor == NULL) || (image_data_info == NULL) ||
      (access_permission < HSA_ACCESS_PERMISSION_RO) ||
      (access_permission > HSA_ACCESS_PERMISSION_RW) ||
      (image_data_layout != HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR)) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  enforceDefaultPitch(agent, image_descriptor, image_data_row_pitch, image_data_slice_pitch);

  return ImageRuntime::instance()->GetImageSizeAndAlignment(
      agent, *image_descriptor, image_data_layout, image_data_row_pitch, image_data_slice_pitch,
      *image_data_info);
  CATCH;
}

hsa_status_t hsa_ext_image_create_with_layout(
    hsa_agent_t agent, const hsa_ext_image_descriptor_t* image_descriptor, const void* image_data,
    hsa_access_permission_t access_permission, hsa_ext_image_data_layout_t image_data_layout,
    size_t image_data_row_pitch, size_t image_data_slice_pitch, hsa_ext_image_t* image) {
  TRY;
  if (agent.handle == 0) {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }

  if (image_descriptor == NULL || image_data == NULL || image == NULL ||
      image_data_layout != HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  enforceDefaultPitch(agent, image_descriptor, image_data_row_pitch, image_data_slice_pitch);

  return ImageRuntime::instance()->CreateImageHandle(
      agent, *image_descriptor, image_data, access_permission, image_data_layout,
      image_data_row_pitch, image_data_slice_pitch, *image);
  CATCH;
}

hsa_status_t hsa_amd_image_create(hsa_agent_t agent,
                                  const hsa_ext_image_descriptor_t* image_descriptor,
                                  const hsa_amd_image_descriptor_t* image_layout,
                                  const void* image_data, hsa_access_permission_t access_permission,
                                  hsa_ext_image_t* image) {
  TRY;
  if (agent.handle == 0) {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }

  if (image_descriptor == NULL || image_data == NULL || image == NULL) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  return ImageRuntime::instance()->CreateImageHandleWithLayout(
      agent, *image_descriptor, image_layout, image_data, access_permission, *image);
  CATCH;
}

void LoadImage(core::ImageExtTableInternal* image_api,
               decltype(::hsa_amd_image_create)** interface_api) {
  image_api->hsa_ext_image_get_capability_fn = hsa_ext_image_get_capability;

  image_api->hsa_ext_image_data_get_info_fn = hsa_ext_image_data_get_info;

  image_api->hsa_ext_image_create_fn = hsa_ext_image_create;

  image_api->hsa_ext_image_import_fn = hsa_ext_image_import;

  image_api->hsa_ext_image_export_fn = hsa_ext_image_export;

  image_api->hsa_ext_image_copy_fn = hsa_ext_image_copy;

  image_api->hsa_ext_image_clear_fn = hsa_ext_image_clear;

  image_api->hsa_ext_image_destroy_fn = hsa_ext_image_destroy;

  image_api->hsa_ext_sampler_create_fn = hsa_ext_sampler_create;

  image_api->hsa_ext_sampler_destroy_fn = hsa_ext_sampler_destroy;

  image_api->hsa_ext_image_get_capability_with_layout_fn = hsa_ext_image_get_capability_with_layout;

  image_api->hsa_ext_image_data_get_info_with_layout_fn = hsa_ext_image_data_get_info_with_layout;

  image_api->hsa_ext_image_create_with_layout_fn = hsa_ext_image_create_with_layout;

  image_api->hsa_amd_image_get_info_max_dim_fn = hsa_amd_image_get_info_max_dim;

  image_api->hsa_ext_sampler_create_v2_fn = hsa_ext_sampler_create_v2;

  *interface_api = hsa_amd_image_create;
}

void ReleaseImageRsrcs() { ImageRuntime::DestroySingleton(); }

}  // namespace image
}  // namespace rocr
