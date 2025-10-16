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

#include "image/inc/hsa_amd_mipmap_impl.h"
#include "core/inc/runtime.h"
#include "core/util/utils.h"
#include "image/image_manager.h"
#include "image_runtime.h"
#include "core/inc/exceptions.h"
#include <algorithm>
#include <vector>
#include <cstring>
#include <cmath>

namespace rocr {

namespace AMD {
hsa_status_t handleException();

template <class T> static __forceinline T handleExceptionT() {
    handleException();
    abort();
    return T();
}
}   // namespace AMD

#define TRY try {
#define CATCH } catch(...) { return AMD::handleException(); }

namespace image {

// Retrieve the image extension dispatch table (version 1.0)
hsa_status_t GetImageExtensionTable(hsa_ext_images_1_pfn_t* table) {
    return hsa_system_get_major_extension_table(HSA_EXTENSION_IMAGES, 1, sizeof(*table), table);
}


// Public API implementations for mipmapped array query & creation
extern "C" {

hsa_status_t HSA_API
hsa_amd_mipmap_array_get_info(
    hsa_agent_t agent,
    const hsa_ext_image_descriptor_t* desc,
    uint32_t requested_levels,
    hsa_ext_image_data_layout_t layout,
    size_t row_pitch,
    size_t slice_pitch,
    hsa_amd_mipmap_array_info_t* info) {

    if (!desc || !info) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    info->size = 0;
    info->alignment = 0;
    info->max_levels = 0;
    info->levels_used = 0;

    auto* rt = rocr::image::ImageRuntime::instance();
    if (!rt) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

    size_t size_req = 0;
    size_t align_req = 0;
    uint32_t max_lvls = 0;
    hsa_status_t s = rt->GetMipmapArraySizeAndAlignment(
        agent, *desc, requested_levels, layout,
        row_pitch, slice_pitch,
        size_req, align_req, max_lvls);
    if (s != HSA_STATUS_SUCCESS) return s;

    info->size = size_req;
    info->alignment = align_req;
    info->max_levels = max_lvls;
    info->levels_used = requested_levels;
    return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API
hsa_amd_mipmap_array_create(
    hsa_agent_t agent,
    const hsa_ext_image_descriptor_t* desc,
    const void* image_data,
    hsa_access_permission_t access_permission,
    uint32_t num_mipmap_levels,
    hsa_ext_image_t* image_out) {

    TRY
        if (!desc || !image_out) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
        if (num_mipmap_levels == 0) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

        debug_print("Validated 'desc', 'image_out' and 'num_mipmap_levels'");

        uint32_t base_dim = desc->width;
        if (desc->geometry == HSA_EXT_IMAGE_GEOMETRY_2D ||
            desc->geometry == HSA_EXT_IMAGE_GEOMETRY_2DA ||
            desc->geometry == HSA_EXT_IMAGE_GEOMETRY_2DDEPTH ||
            desc->geometry == HSA_EXT_IMAGE_GEOMETRY_2DADEPTH) {
            base_dim = std::max<uint32_t>(desc->width, desc->height);
        } else if (desc->geometry == HSA_EXT_IMAGE_GEOMETRY_3D) {
            base_dim = std::max<uint32_t>
                (base_dim, std::max<uint32_t>(desc->height, desc->depth));
        }
        uint32_t max_levels = 1;
        if (base_dim > 1) {
            max_levels = 1 + static_cast<uint32_t>
                (std::floor(std::log2(static_cast<double>(base_dim))));
        }
        if (num_mipmap_levels > max_levels) {
            return HSA_STATUS_ERROR_INVALID_ARGUMENT;
        }

        debug_print("Asserted: num_mipmap_levels <= max_levels");

        auto* rt = rocr::image::ImageRuntime::instance();
        if (!rt) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

        return rt->CreateMipmapArrayHandle(agent, *desc,
            image_data, access_permission, num_mipmap_levels,
            HSA_EXT_IMAGE_DATA_LAYOUT_OPAQUE, 0, 0, *image_out);

    CATCH
}

hsa_status_t HSA_API
hsa_amd_mipmap_array_destroy(const hsa_ext_image_t* image) {

  TRY
  if (!image) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (image->handle == 0) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  auto* rt = rocr::image::ImageRuntime::instance();
  if (!rt) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  return rt->DestroyMipmapArrayHandle(*image);

  CATCH
}

// per-level view retrieval implementation
hsa_status_t HSA_API
hsa_amd_mipmap_array_get_level(
        hsa_agent_t agent,
        const hsa_ext_image_t* mipmapped_array,
        uint32_t mip_level,
        hsa_ext_image_t* level_image_out) {
    TRY
    if (!mipmapped_array || !level_image_out) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    auto* rt = rocr::image::ImageRuntime::instance();
    if (!rt) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

    return rt->GetMipmapArrayLevelHandle(agent, *mipmapped_array, mip_level, *level_image_out);

    CATCH
}

} // extern "C"

} // namespace image
} // namespace rocr
