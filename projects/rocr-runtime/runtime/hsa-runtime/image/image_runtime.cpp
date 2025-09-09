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

#define NOMINMAX
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
#include "image/inc/hsa_amd_mipmap_impl.h"
#include "resource.h"
#include "image_manager_kv.h"
#include "image_manager_ai.h"
#include "image_manager_nv.h"
#include "image_manager_gfx11.h"
#include "image_manager_gfx12.h"
#include "device_info.h"

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
    size_t& alignment_out,
    uint32_t& max_levels_out) {
  size_out = 0;
  alignment_out = 0;
  max_levels_out = ComputeMaxMipLevels(desc);

  if (num_mipmap_levels == 0 || num_mipmap_levels > max_levels_out)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  hsa_ext_image_data_info_t base_info = {0};
  hsa_status_t status = GetImageSizeAndAlignment(component, desc,
                          layout, row_pitch, slice_pitch, base_info);
  if (status != HSA_STATUS_SUCCESS) return status;

  ImageManager* manager = image_manager(component);
  if (!manager) return HSA_STATUS_ERROR_INVALID_AGENT;
  ADDR_HANDLE addr_lib = manager->GetAddrLib();
  if (!addr_lib) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  const core::Agent* agent_info = core::Agent::Convert(component);
  if (!agent_info) return HSA_STATUS_ERROR_INVALID_AGENT;

  bool use_addr3 = false;
  ADDR2_COMPUTE_SURFACE_INFO_INPUT addr2_in = {};
  ADDR2_COMPUTE_SURFACE_INFO_OUTPUT addr2_out = {};
  std::vector<ADDR2_MIP_INFO> addr2_mip_info(num_mipmap_levels);

#ifdef ADDR_GFX12_BUILD
  ADDR3_COMPUTE_SURFACE_INFO_OUTPUT addr3_out = {};

  use_addr3 = rocr::image::UseAddr3ForArchitecture(agent_info->properties().major(), agent_info->properties().minor());

  if (use_addr3) {
    ADDR_E_RETURNCODE addr3_result = rocr::image::ComputeSurfaceInfoAddr3(
        addr_lib, component, &desc, num_mipmap_levels,
        HSA_ACCESS_PERMISSION_RO, layout != HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR,
        &addr3_out);

    if (addr3_result != ADDR_OK) {
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }

    size_out = addr3_out.surfSize;
    alignment_out = addr3_out.baseAlign;
    return HSA_STATUS_SUCCESS;
  }
#endif

  if (!use_addr3) {
    // Use Addr2 for GFX9-GFX11 architectures
    rocr::image::PopulateAddrInput(&addr2_in, &desc, num_mipmap_levels,
                      component, HSA_ACCESS_PERMISSION_RO,
                      layout != HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR);

    // Apply optimal swizzle mode
    if (layout != HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR) {
      addr2_in.swizzleMode = rocr::image::GetOptimalSwizzleMode(addr_lib, addr2_in);
    } else {
      addr2_in.swizzleMode = ADDR_SW_LINEAR;
    }

    addr2_out.pMipInfo = addr2_mip_info.data();
    addr2_out.size = sizeof(ADDR2_COMPUTE_SURFACE_INFO_OUTPUT);

    if (Addr2ComputeSurfaceInfo(addr_lib, &addr2_in, &addr2_out) != ADDR_OK) {
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }

    size_out = addr2_out.surfSize;
    alignment_out = addr2_out.baseAlign;
  }
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
    if (instance == NULL) {
      return NULL;
    }

    // UnloadCallback = &ext_image::ImageRuntime::DestroySingleton;
  }

  return instance;
}

ImageRuntime* ImageRuntime::CreateSingleton() {
  ImageRuntime* instance = new ImageRuntime();

  if (HSA_STATUS_SUCCESS != instance->blit_kernel_.Initialize()) {
    instance->Cleanup();
    delete instance;
    return NULL;
  }

  if (HSA_STATUS_SUCCESS != HSA::hsa_iterate_agents(CreateImageManager, instance)) {
    instance->Cleanup();
    delete instance;
    return NULL;
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

  return manager->CalculateImageSizeAndAlignment(component, desc,
    image_data_layout, image_data_row_pitch, image_data_slice_pitch, image_info);
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
    const void* image_data, size_t image_data_size,
    const hsa_access_permission_t access_permission,
    uint32_t num_mipmap_levels,
    const hsa_ext_image_data_layout_t mipmap_layout,
    size_t image_data_row_pitch, size_t image_data_slice_pitch,
    hsa_ext_image_t& image_handle) {
  image_handle.handle = 0;
  if (!image_data || mipmap_descriptor.width == 0 || num_mipmap_levels == 0) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  // Calculate the required size and alignment for the mipmapped array
  size_t required_size = 0;
  size_t alignment = 0;
  uint32_t max_levels = 0;
  hsa_status_t status = GetMipmapArraySizeAndAlignment(
      component, mipmap_descriptor, num_mipmap_levels, mipmap_layout,
      image_data_row_pitch, image_data_slice_pitch,
      required_size, alignment, max_levels);
  if (status != HSA_STATUS_SUCCESS) return status;

  // Validate the image data size
  if (image_data_size < required_size)
    return HSA_STATUS_ERROR_INVALID_ALLOCATION;

  // Validate the image data alignment
  if (!IsMultipleOf(reinterpret_cast<size_t>(image_data), alignment))
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  ImageManager* manager = image_manager(component);
  if (!manager) return HSA_STATUS_ERROR_INVALID_AGENT;
  ADDR_HANDLE addr_lib = manager->GetAddrLib();
  if (!addr_lib) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  //Vompute AddrLib output for SRD population and per-level info
  const core::Agent* agent_info = core::Agent::Convert(component);
  if (!agent_info) return HSA_STATUS_ERROR_INVALID_AGENT;
 
  bool use_addr3 = false;
  ADDR2_COMPUTE_SURFACE_INFO_INPUT addr2_in = {};
  ADDR2_COMPUTE_SURFACE_INFO_OUTPUT addr2_out = {};
  std::vector<ADDR2_MIP_INFO> addr2_mip_info(num_mipmap_levels);

#ifdef ADDR_GFX12_BUILD
  ADDR3_COMPUTE_SURFACE_INFO_OUTPUT addr3_out = {};
  std::vector<ADDR3_MIP_INFO> addr3_mip_info(num_mipmap_levels);

  //Check if we should use Addr3 for GFX12+ architectures
  use_addr3 = rocr::image::UseAddr3ForArchitecture(agent_info->properties().major(), agent_info->properties().minor());

  if (use_addr3) {
    // Use Addr3 for GFX12+ architectures
    ADDR_E_RETURNCODE addr3_result = rocr::image::ComputeSurfaceInfoAddr3(
        addr_lib, component, &mipmap_descriptor, num_mipmap_levels,
        access_permission, mipmap_layout != HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR,
        &addr3_out);

    if (addr3_result != ADDR_OK) {
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }

    // Copy mip info from addr3_out to local storage
    if (addr3_out.pMipInfo) {
      addr3_mip_info.assign(addr3_out.pMipInfo, addr3_out.pMipInfo + num_mipmap_levels);
    }
  }
#endif

  if (!use_addr3) {
    // Use Addr2 for GFX9-GFX11 architectures
    rocr::image::PopulateAddrInput(&addr2_in, &mipmap_descriptor, num_mipmap_levels,
                      component, access_permission,
                      mipmap_layout != HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR);

    // Apply optimal swizzle mode for PAL parity performance
    if (mipmap_layout != HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR) {
      addr2_in.swizzleMode = rocr::image::GetOptimalSwizzleMode(addr_lib, addr2_in);
    } else {
      addr2_in.swizzleMode = ADDR_SW_LINEAR;
    }

    addr2_out.pMipInfo = addr2_mip_info.data();
    addr2_out.size = sizeof(ADDR2_COMPUTE_SURFACE_INFO_OUTPUT);

    if (Addr2ComputeSurfaceInfo(addr_lib, &addr2_in, &addr2_out) != ADDR_OK) {
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }
  }

  MipmappedArray* mipmap_array = MipmappedArray::Create(component, required_size);
  if (!mipmap_array) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  // Determine the tile mode
  hsa_profile_t profile;
  status = HSA::hsa_agent_get_info(component, HSA_AGENT_INFO_PROFILE, &profile);

  if (mipmap_layout == HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR) {
    mipmap_array->tile_mode = Image::TileMode::LINEAR;
  } else {
    Image::TileMode tileMode =
        (profile == HSA_PROFILE_BASE && mipmap_descriptor.geometry != HSA_EXT_IMAGE_GEOMETRY_1DB)
        ? Image::TileMode::TILED
        : Image::TileMode::LINEAR;
    mipmap_array->tile_mode = tileMode;
  }

  mipmap_array->component = component;
  mipmap_array->data = const_cast<void*>(image_data);
  mipmap_array->desc = mipmap_descriptor;
  mipmap_array->permission = access_permission;
  mipmap_array->num_levels = num_mipmap_levels;
  mipmap_array->levels_allocated = num_mipmap_levels;
  mipmap_array->flags = 0;
  mipmap_array->addr_handle = addr_lib;

  if (use_addr3) {
#ifdef ADDR_GFX12_BUILD
    mipmap_array->addr_output = {};
    mipmap_array->addr_output.size = sizeof(ADDR2_COMPUTE_SURFACE_INFO_OUTPUT);
    mipmap_array->addr_output.pMipInfo = new ADDR2_MIP_INFO[num_mipmap_levels];
    for (uint32_t i = 0; i < num_mipmap_levels; ++i) {
      const auto& addr3_mip = addr3_mip_info[i];
      auto& addr2_mip = mipmap_array->addr_output.pMipInfo[i];

      addr2_mip.pitch = addr3_mip.pitch;
      addr2_mip.height = addr3_mip.height;
      addr2_mip.depth = addr3_mip.depth;
      addr2_mip.pixelPitch = addr3_mip.pixelPitch;
      addr2_mip.pixelHeight = addr3_mip.pixelHeight;
      addr2_mip.equationIndex = addr3_mip.equationIndex;
      addr2_mip.offset = addr3_mip.offset;
      addr2_mip.macroBlockOffset = addr3_mip.macroBlockOffset;
      addr2_mip.mipTailOffset = addr3_mip.mipTailOffset;
      addr2_mip.mipTailCoordX = addr3_mip.mipTailCoordX;
      addr2_mip.mipTailCoordY = addr3_mip.mipTailCoordY;
      addr2_mip.mipTailCoordZ = addr3_mip.mipTailCoordZ;
    }

    // Copy other relevant Addr3 output fields
    mipmap_array->addr_output.surfSize = addr3_out.surfSize;
    mipmap_array->addr_output.baseAlign = addr3_out.baseAlign;
    mipmap_array->addr_output.pitch = addr3_out.pitch;
    mipmap_array->addr_output.height = addr3_out.height;
    mipmap_array->addr_output.numSlices = addr3_out.numSlices;
#endif
  } else {
    // Use Addr2 results directly
    mipmap_array->addr_output = addr2_out;
    mipmap_array->addr_output.pMipInfo = new ADDR2_MIP_INFO[num_mipmap_levels];

    // Copy computed mip info from address library
    for (uint32_t i = 0; i < num_mipmap_levels; ++i) {
      mipmap_array->addr_output.pMipInfo[i] = addr2_mip_info[i];
    }
  }

  manager->PopulateImageSrd(*mipmap_array);
  if (core::Runtime::runtime_singleton_->flag().image_print_srd())
    mipmap_array->printSRD();

  assert(mipmap_array->size == required_size);
  image_handle.handle = mipmap_array->Convert();
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
