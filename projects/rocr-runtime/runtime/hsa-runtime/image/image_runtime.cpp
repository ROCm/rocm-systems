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

#include <assert.h>
#include <climits>
#include <cstring>
#include <vector>
#include <mutex>
#include <algorithm>

#include "core/inc/runtime.h"
#include "core/inc/hsa_internal.h"
#include "core/inc/hsa_ext_amd_impl.h"
#include "core/inc/exceptions.h"
#include "resource.h"
#include "image_manager_kv.h"
#include "image_manager_ai.h"
#include "image_manager_nv.h"
#include "image_manager_gfx11.h"
#include "image_manager_gfx12.h"
#include "device_info.h"


#define SINGLE_MIP_LEVEL 1

namespace rocr {
namespace image {

  static inline uint32_t ComputeMaxMipLevels(const hsa_ext_image_descriptor_t& d) {
    uint32_t w = d.width  ? d.width  : 1;
    uint32_t h = d.height ? d.height : 1;
    uint32_t depth = d.depth ? d.depth : 1;
    uint32_t dim_max = w;
    switch (d.geometry) {
      case HSA_EXT_IMAGE_GEOMETRY_1D:
      case HSA_EXT_IMAGE_GEOMETRY_1DA:
      case HSA_EXT_IMAGE_GEOMETRY_1DB:
        dim_max = w; break;
      case HSA_EXT_IMAGE_GEOMETRY_2D:
      case HSA_EXT_IMAGE_GEOMETRY_2DA:
      case HSA_EXT_IMAGE_GEOMETRY_2DDEPTH:
      case HSA_EXT_IMAGE_GEOMETRY_2DADEPTH:
        dim_max = std::max(w, h); break;
      case HSA_EXT_IMAGE_GEOMETRY_3D:
        dim_max = std::max(std::max(w, h), depth); break;
      default:
        break;
    }
    uint32_t levels = 0;
    while (dim_max > 0) { ++levels; dim_max >>= 1; }
    return (levels == 0) ? 1 : levels;
  }

hsa_status_t ImageRuntime::GetMipmapArraySizeAndAlignment(
    hsa_agent_t component,
    const hsa_ext_image_descriptor_t& desc,
    uint32_t num_mipmap_levels,
    hsa_ext_image_data_layout_t layout,
    size_t row_pitch,
    size_t slice_pitch,
    size_t& size_out,
    size_t& alignment_out) {
  size_out = 0;
  alignment_out = 0;

  if (num_mipmap_levels == 0 || num_mipmap_levels > ComputeMaxMipLevels(desc))
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  // Validate the image format and geometry.
  uint32_t capability = 0;
  hsa_status_t status =
      GetImageCapability(component, desc.format, desc.geometry, capability);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }

  if (capability == 0) {
    return static_cast<hsa_status_t>(
        HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED);
  }

  const hsa_ext_image_geometry_t geometry = desc.geometry;
  uint32_t max_width = 0;
  uint32_t max_height = 0;
  uint32_t max_depth = 0;
  uint32_t max_array_size = 0;

  ImageManager* manager = image_manager(component);

  // Validate the image dimension.
  manager->GetImageInfoMaxDimension(component, geometry, max_width, max_height,
                                    max_depth, max_array_size);

  if (desc.width > max_width || desc.height > max_height ||
      desc.depth > max_depth || desc.array_size > max_array_size) {
    return static_cast<hsa_status_t>(
        HSA_EXT_STATUS_ERROR_IMAGE_SIZE_UNSUPPORTED);
  }

  hsa_ext_image_data_info_t mipmap_info = {0};
  status = manager->CalculateImageSizeAndAlignment(component, desc, layout,
                    num_mipmap_levels, row_pitch, slice_pitch, mipmap_info);
  if (HSA_STATUS_SUCCESS != status) {
    return status;
  }

  alignment_out = mipmap_info.alignment;
  size_out = mipmap_info.size;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t FindKernelArgPool(hsa_amd_memory_pool_t pool, void* data) {
  assert(data != nullptr);

  hsa_status_t err;
  hsa_amd_segment_t segment;
  uint32_t flag;
  size_t size;

  err = AMD::hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
  assert(err == HSA_STATUS_SUCCESS);

  if (segment != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;

  err = AMD::hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flag);
  assert(err == HSA_STATUS_SUCCESS);

  err = AMD::hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SIZE, &size);
  assert(err == HSA_STATUS_SUCCESS);

  if (((HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT & flag) == 1) && (size != 0)) {
    *(reinterpret_cast<hsa_amd_memory_pool_t*>(data)) = pool;
    // Found the kernarg pool, stop the iteration.
    return HSA_STATUS_INFO_BREAK;
    }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t ImageRuntime::CreateImageManager(hsa_agent_t agent, void* data) {
  ImageRuntime* runtime = reinterpret_cast<ImageRuntime*>(data);

  hsa_device_type_t hsa_device_type;
  hsa_status_t hsa_error_code =
      HSA::hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &hsa_device_type);
  if (hsa_error_code != HSA_STATUS_SUCCESS) {
    return hsa_error_code;
  }

  if (hsa_device_type == HSA_DEVICE_TYPE_GPU) {

    uint32_t chip_id;
    hsa_error_code = GetGPUAsicID(agent, &chip_id);
    uint32_t major_ver = MajorVerFromDevID(chip_id);

    ImageManager* image_manager;

    switch (major_ver) {
    case 12:
      image_manager = new ImageManagerGfx12();
      break;
    case 11:
      image_manager = new ImageManagerGfx11();
      break;
    case 10:
      image_manager = new ImageManagerNv();
      break;
    case  9:
      image_manager = new ImageManagerAi();
      break;
    default:
      image_manager = new ImageManagerKv();
      break;
    }
    hsa_error_code = image_manager->Initialize(agent);

    if (hsa_error_code != HSA_STATUS_SUCCESS) {
      delete image_manager;
      return hsa_error_code;
    }

    runtime->image_managers_[agent.handle] = image_manager;
  } else if (hsa_device_type == HSA_DEVICE_TYPE_CPU) {
    uint32_t caches[4] = {0};
    hsa_error_code = HSA::hsa_agent_get_info(agent, HSA_AGENT_INFO_CACHE_SIZE, caches);

    if (hsa_error_code != HSA_STATUS_SUCCESS) {
      return hsa_error_code;
    }

    runtime->cpu_l2_cache_size_ = caches[1];

    if (runtime->kernarg_pool_.handle == 0)
      hsa_amd_agent_iterate_memory_pools(agent, FindKernelArgPool, &runtime->kernarg_pool_);
  }

  return HSA_STATUS_SUCCESS;
}

ImageRuntime* ImageRuntime::instance() {
  ImageRuntime* instance = get_instance().load(std::memory_order_acquire);
  if (instance == NULL) {
    // Protect the initialization from multi threaded access.
    std::lock_guard<std::mutex> lock(instance_mutex());

    // Make sure we are not initializing it twice.
    instance = get_instance().load(std::memory_order_relaxed);
    if (instance != NULL) {
      return instance;
    }

    instance = CreateSingleton();

    // UnloadCallback = &ext_image::ImageRuntime::DestroySingleton;
  }

  return instance;
}

ImageRuntime* ImageRuntime::CreateSingleton() {
  ImageRuntime* instance = new ImageRuntime();

  if (HSA_STATUS_SUCCESS != instance->blit_kernel_.Initialize()) {
    instance->Cleanup();
    delete instance;
    throw AMD::hsa_exception(HSA_STATUS_ERROR_OUT_OF_RESOURCES, 
                             "ImageRuntime: Failed to initialize blit kernel");
  }

  if (HSA_STATUS_SUCCESS != HSA::hsa_iterate_agents(CreateImageManager, instance)) {
    instance->Cleanup();
    delete instance;
    throw AMD::hsa_exception(HSA_STATUS_ERROR_OUT_OF_RESOURCES,
                             "ImageRuntime: Failed to create image managers");
  }

  assert(instance->kernarg_pool_.handle != 0);
  assert(instance->image_managers_.size() != 0);

  get_instance().store(instance, std::memory_order_release);
  return instance;
}

void ImageRuntime::DestroySingleton() {
  ImageRuntime* instance = get_instance().load(std::memory_order_acquire);
  if (instance == NULL) {
    return;
  }

  instance->Cleanup();

  get_instance().store(NULL, std::memory_order_release);
  delete instance;
}

hsa_status_t ImageRuntime::GetImageInfoMaxDimension(hsa_agent_t component,
                                                    hsa_agent_info_t attribute,
                                                    void* value) {
  uint32_t* value_u32 = NULL;
  uint32_t* value_u32_v2 = NULL;
  uint32_t* value_u32_v3 = NULL;

  hsa_ext_image_geometry_t geometry;

  size_t image_attribute = static_cast<size_t>(attribute);
  switch (image_attribute) {
    case HSA_EXT_AGENT_INFO_IMAGE_1D_MAX_ELEMENTS:
      geometry = HSA_EXT_IMAGE_GEOMETRY_1D;
      value_u32 = static_cast<uint32_t*>(value);
      break;
    case HSA_EXT_AGENT_INFO_IMAGE_1DA_MAX_ELEMENTS:
      geometry = HSA_EXT_IMAGE_GEOMETRY_1DA;
      value_u32 = static_cast<uint32_t*>(value);
      break;
    case HSA_EXT_AGENT_INFO_IMAGE_1DB_MAX_ELEMENTS:
      geometry = HSA_EXT_IMAGE_GEOMETRY_1DB;
      value_u32 = static_cast<uint32_t*>(value);
      break;
    case HSA_EXT_AGENT_INFO_IMAGE_2D_MAX_ELEMENTS:
      geometry = HSA_EXT_IMAGE_GEOMETRY_2D;
      value_u32_v2 = static_cast<uint32_t*>(value);
      break;
    case HSA_EXT_AGENT_INFO_IMAGE_2DA_MAX_ELEMENTS:
      geometry = HSA_EXT_IMAGE_GEOMETRY_2DA;
      value_u32_v2 = static_cast<uint32_t*>(value);
      break;
    case HSA_EXT_AGENT_INFO_IMAGE_2DDEPTH_MAX_ELEMENTS:
      geometry = HSA_EXT_IMAGE_GEOMETRY_2DDEPTH;
      value_u32_v2 = static_cast<uint32_t*>(value);
      break;
    case HSA_EXT_AGENT_INFO_IMAGE_2DADEPTH_MAX_ELEMENTS:
      geometry = HSA_EXT_IMAGE_GEOMETRY_2DADEPTH;
      value_u32_v2 = static_cast<uint32_t*>(value);
      break;
    case HSA_EXT_AGENT_INFO_IMAGE_3D_MAX_ELEMENTS:
      geometry = HSA_EXT_IMAGE_GEOMETRY_3D;
      value_u32_v3 = static_cast<uint32_t*>(value);
      break;
    case HSA_EXT_AGENT_INFO_IMAGE_ARRAY_MAX_LAYERS:
      geometry = HSA_EXT_IMAGE_GEOMETRY_2DA;
      value_u32 = static_cast<uint32_t*>(value);
      break;
    default:
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t depth = 0;
  uint32_t array_size = 0;

  hsa_device_type_t device_type;
  hsa_status_t status = HSA::hsa_agent_get_info(component, HSA_AGENT_INFO_DEVICE, &device_type);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }

  // Image is only supported on a GPU device.

  if (device_type == HSA_DEVICE_TYPE_GPU) {
    image_manager(component)->GetImageInfoMaxDimension(
        component, geometry, width, height, depth, array_size);
  }

  if (value_u32_v3 != NULL) {
    value_u32_v3[0] = width;
    value_u32_v3[1] = height;
    value_u32_v3[2] = depth;
  } else if (value_u32_v2 != NULL) {
    value_u32_v2[0] = width;
    value_u32_v2[1] = height;
  } else {
    *value_u32 = (image_attribute == HSA_EXT_AGENT_INFO_IMAGE_ARRAY_MAX_LAYERS)
                     ? array_size
                     : width;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t ImageRuntime::GetImageCapability(
    hsa_agent_t component, const hsa_ext_image_format_t& format,
    hsa_ext_image_geometry_t geometry, uint32_t& capability) {
  hsa_device_type_t device_type;
  hsa_status_t status = HSA::hsa_agent_get_info(component, HSA_AGENT_INFO_DEVICE, &device_type);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }

  if (device_type == HSA_DEVICE_TYPE_GPU) {
    ImageManager* manager = image_manager(component);
    capability = manager->GetImageProperty(component, format, geometry).cap;
  } else {
    // Image is only supported on a GPU device.
    capability = 0;
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t ImageRuntime::GetImageSizeAndAlignment(
    hsa_agent_t component, const hsa_ext_image_descriptor_t& desc,
    hsa_ext_image_data_layout_t image_data_layout,
    size_t image_data_row_pitch,
    size_t image_data_slice_pitch,
    hsa_ext_image_data_info_t& image_info) {
  image_info.alignment = 0;
  image_info.size = 0;

  // Validate the image format and geometry.
  uint32_t capability = 0;
  hsa_status_t status =
      GetImageCapability(component, desc.format, desc.geometry, capability);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }

  if (capability == 0) {
    return static_cast<hsa_status_t>(
        HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED);
  }

  const hsa_ext_image_geometry_t geometry = desc.geometry;
  uint32_t max_width = 0;
  uint32_t max_height = 0;
  uint32_t max_depth = 0;
  uint32_t max_array_size = 0;

  ImageManager* manager = image_manager(component);

  // Validate the image dimension.
  manager->GetImageInfoMaxDimension(component, geometry, max_width, max_height,
                                    max_depth, max_array_size);

  if (desc.width > max_width || desc.height > max_height ||
      desc.depth > max_depth || desc.array_size > max_array_size) {
    return static_cast<hsa_status_t>(
        HSA_EXT_STATUS_ERROR_IMAGE_SIZE_UNSUPPORTED);
  }

  return manager->CalculateImageSizeAndAlignment(
      component, desc, image_data_layout, SINGLE_MIP_LEVEL,
      image_data_row_pitch, image_data_slice_pitch, image_info);
}

hsa_status_t ImageRuntime::CreateImageHandle(
    hsa_agent_t component, const hsa_ext_image_descriptor_t& image_descriptor,
    const void* image_data, const hsa_access_permission_t access_permission,
    hsa_ext_image_data_layout_t image_data_layout,
    size_t image_data_row_pitch,
    size_t image_data_slice_pitch,
    hsa_ext_image_t& image_handle) {
  image_handle.handle = 0;

  assert(image_data != NULL);

  // Validate image dimension.
  hsa_ext_image_data_info_t image_info = {0};
  hsa_status_t status =
      GetImageSizeAndAlignment(component, image_descriptor,
        image_data_layout, image_data_row_pitch, image_data_slice_pitch,
        image_info);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }

  // Validate image address alignment.
  if (!IsMultipleOf(reinterpret_cast<size_t>(image_data),
                    image_info.alignment)) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  Image* image = Image::Create(component);
  image->component = component;
  image->desc = image_descriptor;
  image->permission = access_permission;
  image->data = const_cast<void*>(image_data);
  image->row_pitch = image_data_row_pitch;
  image->slice_pitch = image_data_slice_pitch;
  hsa_profile_t profile;
  status = HSA::hsa_agent_get_info(component, HSA_AGENT_INFO_PROFILE, &profile);

  if (image_data_layout == HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR) {
    image->tile_mode = Image::TileMode::LINEAR;
  } else {
    Image::TileMode tileMode =
        (profile == HSA_PROFILE_BASE && image_descriptor.geometry != HSA_EXT_IMAGE_GEOMETRY_1DB)
        ? Image::TileMode::TILED
        : Image::TileMode::LINEAR;
    image->tile_mode = tileMode;
  }

  image_manager(component)->PopulateImageSrd(*image);

  if (core::Runtime::runtime_singleton_->flag().image_print_srd()) image->printSRD();

  image_handle.handle = image->Convert();

  return HSA_STATUS_SUCCESS;
}

hsa_status_t ImageRuntime::CreateImageHandleWithLayout(
  hsa_agent_t component, const hsa_ext_image_descriptor_t& image_descriptor,
  const hsa_amd_image_descriptor_t* image_layout,
  const void* image_data, const hsa_access_permission_t access_permission,
  hsa_ext_image_t& image_handle)
{
  if(!IsMultipleOf(image_data, 256))
    return HSA_STATUS_ERROR_INVALID_ALLOCATION;

  if(image_layout->version!=1)
    return (hsa_status_t)HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED;
  
  uint32_t id;
  HSA::hsa_agent_get_info(component, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_CHIP_ID, &id);

  if(image_layout->deviceID!=(0x1002<<16|id))
    return (hsa_status_t)HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED;

  const metadata_amd_t* desc = reinterpret_cast<const metadata_amd_t*>(image_layout);

  Image* image = Image::Create(component);
  image->component = component;
  image->desc = image_descriptor;
  image->permission = access_permission;
  image->data = const_cast<void*>(image_data);
  image->tile_mode = Image::TILED;
  hsa_status_t err=image_manager(component)->PopulateImageSrd(*image, desc);
  if(err!=HSA_STATUS_SUCCESS) {
    Image::Destroy(image);
    return err;
  }

  if (core::Runtime::runtime_singleton_->flag().image_print_srd()) image->printSRD();

  image_handle.handle = image->Convert();
  return HSA_STATUS_SUCCESS;
}

hsa_status_t ImageRuntime::CreateMipmapArrayHandleWithLayout(
    hsa_agent_t component, const hsa_ext_image_descriptor_t& mipmap_descriptor,
    const hsa_amd_image_descriptor_t* image_layout,
    const void* image_data, const hsa_access_permission_t access_permission,
    uint32_t num_mipmap_levels,
    hsa_ext_image_t& image_handle) {
  
  image_handle.handle = 0;
  
  if (!IsMultipleOf(image_data, 256)) {
    return HSA_STATUS_ERROR_INVALID_ALLOCATION;
  }

  if (image_layout->version != 1) {
    return (hsa_status_t)HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED;
  }

  uint32_t id;
  HSA::hsa_agent_get_info(component, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_CHIP_ID, &id);

  if (image_layout->deviceID != (0x1002 << 16 | id)) {
    return (hsa_status_t)HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED;
  }

  if (num_mipmap_levels == 0) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  const metadata_amd_t* desc = reinterpret_cast<const metadata_amd_t*>(image_layout);

  MipmappedArray* mipmap_array = MipmappedArray::Create(component);
  if (!mipmap_array) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  mipmap_array->component = component;
  mipmap_array->desc = mipmap_descriptor;
  mipmap_array->permission = access_permission;
  mipmap_array->num_levels = num_mipmap_levels;
  mipmap_array->data = const_cast<void*>(image_data);
  mipmap_array->flags = 0;

  ImageManager* manager = image_manager(component);
  if (!manager) {
    MipmappedArray::Destroy(mipmap_array);
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  hsa_status_t status = manager->PopulateMipmapSrd(*mipmap_array, desc);
  if (status != HSA_STATUS_SUCCESS) {
    MipmappedArray::Destroy(mipmap_array);
    return status;
  }

  image_handle.handle = mipmap_array->Convert();
  return HSA_STATUS_SUCCESS;
}

hsa_status_t ImageRuntime::DestroyImageHandle(
    const hsa_ext_image_t& image_handle) {
  const Image* image = Image::Convert(image_handle.handle);

  if (image == NULL) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  Image::Destroy(const_cast<Image*>(image));

  return HSA_STATUS_SUCCESS;
}

hsa_status_t ImageRuntime::CopyBufferToImage(
    const void* src_memory, size_t src_row_pitch, size_t src_slice_pitch,
    const hsa_ext_image_t& dst_image_handle,
    const hsa_ext_image_region_t& image_region) {
  const Image* dst_image = Image::Convert(dst_image_handle.handle);

  if (dst_image == NULL) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  ImageManager* manager = image_manager(dst_image->component);
  return manager->CopyBufferToImage(src_memory, src_row_pitch, src_slice_pitch,
                                    *dst_image, image_region);
}

hsa_status_t ImageRuntime::CopyImageToBuffer(
    const hsa_ext_image_t& src_image_handle, void* dst_memory,
    size_t dst_row_pitch, size_t dst_slice_pitch,
    const hsa_ext_image_region_t& image_region) {
  const Image* src_image = Image::Convert(src_image_handle.handle);

  if (src_image == NULL) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  ImageManager* manager = image_manager(src_image->component);
  return manager->CopyImageToBuffer(*src_image, dst_memory, dst_row_pitch,
                                    dst_slice_pitch, image_region);
}

hsa_status_t ImageRuntime::CopyImage(const hsa_ext_image_t& src_image_handle,
                                     const hsa_ext_image_t& dst_image_handle,
                                     const hsa_dim3_t& src_origin,
                                     const hsa_dim3_t& dst_origin,
                                     const hsa_dim3_t size) {
  const Image* src_image = Image::Convert(src_image_handle.handle);

  if (src_image == NULL) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  const Image* dst_image = Image::Convert(dst_image_handle.handle);

  if (dst_image == NULL) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  if (src_image->component.handle != dst_image->component.handle) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  ImageManager* manager = image_manager(src_image->component);
  return manager->CopyImage(*dst_image, *src_image, dst_origin, src_origin,
                            size);
}

hsa_status_t ImageRuntime::FillImage(
    const hsa_ext_image_t& image_handle, const void* pattern,
    const hsa_ext_image_region_t& image_region) {
  const Image* image = Image::Convert(image_handle.handle);

  if (image == NULL) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  ImageManager* manager = image_manager(image->component);
  return manager->FillImage(*image, pattern, image_region);
}

hsa_status_t ImageRuntime::CreateSamplerHandle(
    hsa_agent_t component,
    const hsa_ext_sampler_descriptor_v2_t& sampler_descriptor,
    hsa_ext_sampler_t& sampler_handle) {
  sampler_handle.handle = 0;

  hsa_device_type_t device_type;
  hsa_status_t status = HSA::hsa_agent_get_info(component, HSA_AGENT_INFO_DEVICE, &device_type);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }

  // Sampler is only supported on a GPU device.
  if (device_type != HSA_DEVICE_TYPE_GPU) {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }

  Sampler* sampler = Sampler::Create(component);
  if (sampler == NULL) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  sampler->component = component;
  sampler->desc = sampler_descriptor;

  image_manager(component)->PopulateSamplerSrd(*sampler);

  sampler_handle.handle = sampler->Convert();

  return HSA_STATUS_SUCCESS;
}

hsa_status_t ImageRuntime::DestroySamplerHandle(
    hsa_ext_sampler_t& sampler_handle) {
  const Sampler* sampler = Sampler::Convert(sampler_handle.handle);

  if (sampler == NULL) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  Sampler::Destroy(sampler);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t ImageRuntime::CreateMipmapArrayHandle(
    hsa_agent_t component, const hsa_ext_image_descriptor_t& mipmap_descriptor,
    const void* image_data, const hsa_access_permission_t access_permission,
    uint32_t num_mipmap_levels,
    const hsa_ext_image_data_layout_t mipmap_layout,
    size_t image_data_row_pitch, size_t image_data_slice_pitch,
    hsa_ext_image_t& image_handle) {
  image_handle.handle = 0;
  if (mipmap_descriptor.width == 0 || num_mipmap_levels == 0) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  ImageManager* manager = image_manager(component);
  if (!manager) return HSA_STATUS_ERROR_INVALID_AGENT;

  // Validate mipmap array size and alignment requirements
  size_t required_size = 0;
  size_t required_alignment = 0;
  hsa_status_t status = GetMipmapArraySizeAndAlignment(
      component, mipmap_descriptor, num_mipmap_levels, mipmap_layout, image_data_row_pitch,
      image_data_slice_pitch, required_size, required_alignment);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }

  // Verify image_data alignment
  assert(image_data != NULL);
  assert(IsMultipleOf(image_data, required_alignment));

  // Create a new mipmapped array object
  MipmappedArray* mipmap_array = MipmappedArray::Create(component);
  if (!mipmap_array) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  // Determine the tile mode
  // 1DB (1D buffered) geometry MUST always be LINEAR per HSA spec
  // LINEAR layout forces linear swizzle mode (required by API)
  // TILED allows AddrLib to use internal heuristics to select optimal swizzle mode
  if (mipmap_descriptor.geometry == HSA_EXT_IMAGE_GEOMETRY_1DB) {
    // 1DB always uses linear addressing per HSA specification
    mipmap_array->tile_mode = Image::TileMode::LINEAR;
  } else if (mipmap_layout == HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR) {
    // Explicit LINEAR layout forces linear swizzle mode
    mipmap_array->tile_mode = Image::TileMode::LINEAR;
  } else {
    // OPAQUE layout: Let AddrLib choose the best swizzle mode
    mipmap_array->tile_mode = Image::TileMode::TILED;
  }

  // Initialize the mipmapped array object
  mipmap_array->component = component;
  mipmap_array->data = const_cast<void*>(image_data);
  mipmap_array->desc = mipmap_descriptor;
  mipmap_array->permission = access_permission;
  mipmap_array->num_levels = num_mipmap_levels;
  mipmap_array->flags = 0;

  manager->PopulateMipmapSrd(*mipmap_array);

  // assert(mipmap_array->size == required_size);
  image_handle.handle = mipmap_array->Convert();

  if (core::Runtime::runtime_singleton_->flag().image_print_srd()) {
    debug_print("Tile mode = %u (0: LINEAR, 1: TILED)", mipmap_array->tile_mode);
    debug_print("Populating mipmapped array SRD...");
    mipmap_array->printSRD();
    manager->printSRDDetailed(mipmap_array->srd);
    debug_print("output handle = %lu", image_handle.handle);
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t ImageRuntime::DestroyMipmapArrayHandle(
    const hsa_ext_image_t& image_handle) {
  const MipmappedArray* mipmap_array = MipmappedArray::Convert(image_handle.handle);

  if (mipmap_array == NULL) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  MipmappedArray::Destroy(const_cast<MipmappedArray*>(mipmap_array));

  return HSA_STATUS_SUCCESS;
}

hsa_status_t ImageRuntime::GetMipmapArrayLevelHandle(
    hsa_agent_t component, const hsa_ext_image_t& mipmapped_array,
    uint32_t mip_level, const hsa_ext_image_descriptor_v2_t* image_descriptor,
    hsa_ext_image_t& level_image_out) {
  ImageManager * manager = image_manager(component);
  if (!manager) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  level_image_out.handle = 0;

  // Get GPU architecture version
  uint32_t chip_id;
  hsa_status_t status = GetGPUAsicID(component, &chip_id);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }
  uint32_t major_ver = MajorVerFromDevID(chip_id);
  if (major_ver < 9) {
    debug_print("ERROR: Mip level views not supported on GFX%u hardware\n", major_ver);
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  // Validate mip level
  if (mip_level < 0) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  // Convert handle to internal object and perform basic sanity.
  rocr::image::MipmappedArray* array =
          rocr::image::MipmappedArray::Convert(mipmapped_array.handle);
  if (!array || array->num_levels == 0 || mip_level >= array->num_levels) {
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  if (core::Runtime::runtime_singleton_->flag().image_print_srd()) {
    debug_print("Creating mip level %u view for %u level mipmap\n",
              mip_level, array->num_levels);
  }

  // Create a view that references the parent mipmap array
  MipmappedArray* level_view = MipmappedArray::Create(component);
  if (!level_view) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  auto format = image_descriptor ? &image_descriptor->format : nullptr;
   if (format &&
       (array->desc.format.channel_type != format->channel_type ||
        array->desc.format.channel_order != format->channel_order)) {
    MipmappedArray tempArray = *array;
    tempArray.desc.format.channel_type = format->channel_type;
    tempArray.desc.format.channel_order = format->channel_order;
    status = manager->PopulateMipmapSrd(tempArray);
    if (status == HSA_STATUS_SUCCESS) {
      status = manager->PopulateMipLevelSrd(*level_view, tempArray, mip_level);
    }
    else {
      debug_print("PopulateMipmapSrd() failed with status %d", status);
    }
  }
  else {
    status = manager->PopulateMipLevelSrd(*level_view, *array, mip_level);
  }
  if (status != HSA_STATUS_SUCCESS) {
    MipmappedArray::Destroy(level_view);
    return status;
  }

  if (core::Runtime::runtime_singleton_->flag().image_print_srd()) {
    level_view->printSRD();
    manager->printSRDDetailed(level_view->srd);
  }
  // Return handle
  level_image_out.handle = level_view->Convert();
  return HSA_STATUS_SUCCESS;
}

ImageRuntime::ImageRuntime()
    : cpu_l2_cache_size_(0), kernarg_pool_({0}) {}

ImageRuntime::~ImageRuntime() {}

void ImageRuntime::Cleanup() {
  std::map<uint64_t, ImageManager*>::iterator it;
  for (it = image_managers_.begin(); it != image_managers_.end(); ++it) {
    it->second->Cleanup();
    delete it->second;
  }

  blit_kernel_.Cleanup();
}

}  // namespace image
}  // namespace rocr
