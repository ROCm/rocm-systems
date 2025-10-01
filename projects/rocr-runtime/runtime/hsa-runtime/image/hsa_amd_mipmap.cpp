/*

Copyright © 2025 Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

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
#include <unordered_map>
#include <mutex>

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

class AddrLibCache {
private:
    static std::unordered_map<uint32_t, ADDR_HANDLE> cached_handles_;
    static std::mutex cache_mutex_;

public:
    static ADDR_HANDLE GetCachedHandle(uint32_t architecture) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        auto it = cached_handles_.find(architecture);
        return (it != cached_handles_.end()) ? it->second : nullptr;
    }

    static void CacheHandle(uint32_t architecture, ADDR_HANDLE handle) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        cached_handles_[architecture] = handle;
    }

    static void ClearCache() {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        cached_handles_.clear();
    }
};

std::unordered_map<uint32_t, ADDR_HANDLE> AddrLibCache::cached_handles_;
std::mutex AddrLibCache::cache_mutex_;

void InitializeArchitectureHandles() {
    static bool initialized = false;
    if (!initialized) {
        initialized = true;
    }
}

AddrSwizzleMode GetOptimalSwizzleMode(
    ADDR_HANDLE addr_lib,
    const ADDR2_COMPUTE_SURFACE_INFO_INPUT& base_input) {

    ADDR2_GET_PREFERRED_SURF_SETTING_INPUT pref_input = {0};
    ADDR2_GET_PREFERRED_SURF_SETTING_OUTPUT pref_output = {0};

    pref_input.size = sizeof(pref_input);
    pref_input.flags = base_input.flags;
    pref_input.resourceType = base_input.resourceType;
    pref_input.format = base_input.format;
    pref_input.bpp = base_input.bpp;
    pref_input.width = base_input.width;
    pref_input.height = base_input.height;
    pref_input.numSlices = base_input.numSlices;
    pref_input.numMipLevels = base_input.numMipLevels;
    pref_input.numSamples = base_input.numSamples;
    pref_input.numFrags = base_input.numFrags;

    pref_output.size = sizeof(pref_output);

    ADDR_E_RETURNCODE result = Addr2GetPreferredSurfaceSetting(addr_lib, &pref_input, &pref_output);
    if (result == ADDR_OK) {
        return pref_output.swizzleMode;
    }
    return ADDR_SW_64KB_R_X;
}

ADDR2_SURFACE_FLAGS DetermineSurfaceFlags(
    const hsa_ext_image_descriptor_t* desc,
    hsa_access_permission_t access_permission,
    bool is_tiled) {

    ADDR2_SURFACE_FLAGS flags = {0};

    flags.texture = 1;

    flags.color = (desc->format.channel_order != HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH &&
                   desc->format.channel_order != HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH_STENCIL) ? 1 : 0;

    flags.depth = (desc->format.channel_order == HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH ||
                   desc->format.channel_order == HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH_STENCIL) ? 1 : 0;

    flags.display = 0;

    flags.opt4space = 1;
    flags.allowExtEquation = 1;

    flags.needEquation = (access_permission & HSA_ACCESS_PERMISSION_RW) ? 1 : 0;

    return flags;
}

bool UseAddr3ForArchitecture(uint32_t gfx_major, uint32_t gfx_minor) {
    return (gfx_major >= 12);
}

AddrResourceType GetAddrResourceType(hsa_ext_image_geometry_t geometry) {
    switch (geometry) {
        case HSA_EXT_IMAGE_GEOMETRY_1D:
        case HSA_EXT_IMAGE_GEOMETRY_1DA:
        case HSA_EXT_IMAGE_GEOMETRY_1DB:
            return ADDR_RSRC_TEX_1D;
        case HSA_EXT_IMAGE_GEOMETRY_2D:
        case HSA_EXT_IMAGE_GEOMETRY_2DA:
        case HSA_EXT_IMAGE_GEOMETRY_2DDEPTH:
        case HSA_EXT_IMAGE_GEOMETRY_2DADEPTH:
            return ADDR_RSRC_TEX_2D;
        case HSA_EXT_IMAGE_GEOMETRY_3D:
            return ADDR_RSRC_TEX_3D;
        default:
            return ADDR_RSRC_MAX_TYPE;
    }
}

AddrFormat HsaToAddrFormat(const hsa_ext_image_format_t& format) {
    switch (format.channel_type) {
        case HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT8:
        case HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT8:
        case HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT8:
        case HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8:
            return ADDR_FMT_8;
        case HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT16:
        case HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT16:
        case HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT16:
        case HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16:
        case HSA_EXT_IMAGE_CHANNEL_TYPE_HALF_FLOAT:
            return ADDR_FMT_16;
        case HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT32:
        case HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32:
        case HSA_EXT_IMAGE_CHANNEL_TYPE_FLOAT:
            return ADDR_FMT_32;
        default:
            return ADDR_FMT_INVALID;
    }
}

uint32_t ComputeBitsPerPixel(const hsa_ext_image_format_t& format) {
    uint32_t comp_bits = 0;
    switch (format.channel_type) {
        case HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT8:
        case HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT8:
        case HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT8:
        case HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8:
            comp_bits = 8; break;
        case HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT16:
        case HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT16:
        case HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT16:
        case HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16:
        case HSA_EXT_IMAGE_CHANNEL_TYPE_HALF_FLOAT:
            comp_bits = 16; break;
        case HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT32:
        case HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32:
        case HSA_EXT_IMAGE_CHANNEL_TYPE_FLOAT:
            comp_bits = 32; break;
        case HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT24:
            comp_bits = 24; break;
        case HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555:
            comp_bits = 16; break;
        case HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565:
            comp_bits = 16; break;
        case HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_101010:
            comp_bits = 32; break;
        default:
            return 0;
    }

    uint32_t comps = 0;
    switch (format.channel_order) {
        case HSA_EXT_IMAGE_CHANNEL_ORDER_R:
        case HSA_EXT_IMAGE_CHANNEL_ORDER_A:
        case HSA_EXT_IMAGE_CHANNEL_ORDER_INTENSITY:
        case HSA_EXT_IMAGE_CHANNEL_ORDER_LUMINANCE:
        case HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH:
            comps = 1; break;
        case HSA_EXT_IMAGE_CHANNEL_ORDER_RG:
        case HSA_EXT_IMAGE_CHANNEL_ORDER_RA:
            comps = 2; break;
        case HSA_EXT_IMAGE_CHANNEL_ORDER_RGB:
        case HSA_EXT_IMAGE_CHANNEL_ORDER_SRGB:
            comps = 3; break;
        case HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA:
        case HSA_EXT_IMAGE_CHANNEL_ORDER_BGRA:
        case HSA_EXT_IMAGE_CHANNEL_ORDER_ARGB:
        case HSA_EXT_IMAGE_CHANNEL_ORDER_ABGR:
        case HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBA:
        case HSA_EXT_IMAGE_CHANNEL_ORDER_SBGRA:
        case HSA_EXT_IMAGE_CHANNEL_ORDER_RGBX:
        case HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBX:
            comps = 4; break;
        case HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH_STENCIL:
           if (comp_bits == 24) return 32;
            if (comp_bits == 16) return 16;
            return 32;
        default:
            return 0;
    }
    if (format.channel_type == HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555 ||
        format.channel_type == HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565 ||
        format.channel_type == HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_101010) {
        return comp_bits;
    }
    return comps * comp_bits;
}

bool FillTileInfo(
    hsa_agent_t agent,
    const hsa_ext_image_descriptor_t* desc,
    ADDR_TILEINFO& out_tile) {
    (void)agent;
    (void)desc;

    std::memset(&out_tile, 0, sizeof(out_tile));
    out_tile.banks = 4;
    out_tile.bankWidth = 2;
    out_tile.bankHeight = 2;
    out_tile.macroAspectRatio = 1;
    out_tile.tileSplitBytes = 0;
    out_tile.pipeConfig = ADDR_PIPECFG_P2;
    return true;
}

void PopulateAddrInput(
    ADDR2_COMPUTE_SURFACE_INFO_INPUT* pIn,
    const hsa_ext_image_descriptor_t* desc,
    uint32_t num_mipmap_levels,
    hsa_agent_t agent,
    hsa_access_permission_t access_permission,
    bool is_tiled) {
    std::memset(pIn, 0, sizeof(*pIn));
    pIn->size = sizeof(*pIn);

    // Use dynamic surface flags instead of hardcoded values
    pIn->flags = DetermineSurfaceFlags(desc, access_permission, is_tiled);

    pIn->resourceType = GetAddrResourceType(desc->geometry);
    pIn->format = HsaToAddrFormat(desc->format);

    uint32_t bpp = ComputeBitsPerPixel(desc->format);
    if (bpp == 0) {
          bpp = 32;
    }
    pIn->bpp = bpp;

    pIn->width = desc->width;
    pIn->height = desc->height;
    pIn->numSlices = (desc->array_size > 0) ? desc->array_size : 1;
    pIn->numMipLevels = num_mipmap_levels;
   static thread_local ADDR_TILEINFO tileInfo;
    FillTileInfo(agent, desc, tileInfo);
    pIn->pTileInfo = &tileInfo;

    pIn->pitchInElement = 0;
    pIn->sliceAlign = 0;
    pIn->numSamples = 1;
    pIn->numFrags = 0;

    // Note: swizzleMode will be set by GetOptimalSwizzleMode after this function
}

#ifdef ADDR_GFX12_BUILD
ADDR_E_RETURNCODE ComputeSurfaceInfoAddr3(
    ADDR_HANDLE addr_lib,
    hsa_agent_t agent,
    const hsa_ext_image_descriptor_t* desc,
    uint32_t num_mipmap_levels,
    hsa_access_permission_t access_permission,
    bool is_tiled,
    ADDR3_COMPUTE_SURFACE_INFO_OUTPUT* addr_out) {

    const core::Agent* agent_info = core::Agent::Convert(agent);
    if (!agent_info) return ADDR_INVALIDPARAMS;

    uint32_t gfx_major = agent_info->properties().major();
    uint32_t gfx_minor = agent_info->properties().minor();

    if (!UseAddr3ForArchitecture(gfx_major, gfx_minor)) {
        return ADDR_INVALIDPARAMS; // Should use Addr2
    }

    ADDR3_COMPUTE_SURFACE_INFO_INPUT addr_in = {0};
    addr_in.size = sizeof(addr_in);

    addr_in.flags = DetermineSurfaceFlags(desc, access_permission, is_tiled);
    addr_in.resourceType = GetAddrResourceType(desc->geometry);
    addr_in.format = HsaToAddrFormat(desc->format);
    addr_in.bpp = ComputeBitsPerPixel(desc->format);
    if (addr_in.bpp == 0) addr_in.bpp = 32;

    addr_in.width = desc->width;
    addr_in.height = desc->height;
    addr_in.numSlices = (desc->array_size > 0) ? desc->array_size : 1;
    addr_in.numMipLevels = num_mipmap_levels;
    addr_in.numSamples = 1;
    addr_in.numFrags = 0;

    if (is_tiled) {
        ADDR3_GET_PREFERRED_SURF_SETTING_INPUT pref_input = {0};
        ADDR3_GET_PREFERRED_SURF_SETTING_OUTPUT pref_output = {0};

        pref_input.size = sizeof(pref_input);
        pref_input.flags = addr_in.flags;
        pref_input.resourceType = addr_in.resourceType;
        pref_input.format = addr_in.format;
        pref_input.bpp = addr_in.bpp;
        pref_input.width = addr_in.width;
        pref_input.height = addr_in.height;
        pref_input.numSlices = addr_in.numSlices;
        pref_input.numMipLevels = addr_in.numMipLevels;

        pref_output.size = sizeof(pref_output);

        if (Addr3GetPreferredSurfaceSetting(addr_lib, &pref_input, &pref_output) == ADDR_OK) {
            addr_in.swizzleMode = pref_output.swizzleMode;
        } else {
            addr_in.swizzleMode = ADDR_SW_64KB_R_X; // fallback
        }
    } else {
        addr_in.swizzleMode = ADDR_SW_LINEAR;
    }

    return Addr3ComputeSurfaceInfo(addr_lib, &addr_in, addr_out);
}
#endif


hsa_status_t GetImageExtensionTable(hsa_ext_images_1_pfn_t* table) {
    return hsa_system_get_major_extension_table(HSA_EXTENSION_IMAGES, 1, sizeof(*table), table);
}

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
  size_t image_data_size,
  hsa_access_permission_t access_permission,
  uint32_t num_mipmap_levels,
  hsa_ext_image_data_layout_t layout,
  size_t row_pitch,
  size_t slice_pitch,
  hsa_ext_image_t* image_out) {

  TRY

  if (!desc || !image_out) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (num_mipmap_levels == 0) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  uint32_t base_dim = desc->width;
  if (desc->geometry == HSA_EXT_IMAGE_GEOMETRY_2D || desc->geometry == HSA_EXT_IMAGE_GEOMETRY_2DA ||
      desc->geometry == HSA_EXT_IMAGE_GEOMETRY_2DDEPTH || desc->geometry == HSA_EXT_IMAGE_GEOMETRY_2DADEPTH) {
      base_dim = std::max<uint32_t>(desc->width, desc->height);
  } else if (desc->geometry == HSA_EXT_IMAGE_GEOMETRY_3D) {
      base_dim = std::max<uint32_t>(base_dim, std::max<uint32_t>(desc->height, desc->depth));
  }
  uint32_t max_levels = 1;
  if (base_dim > 1) {
      max_levels = 1 + static_cast<uint32_t>(std::floor(std::log2(static_cast<double>(base_dim))));
  }
  if (num_mipmap_levels > max_levels) {
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  auto* rt = rocr::image::ImageRuntime::instance();
  if (!rt) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  return rt->CreateMipmapArrayHandle(agent, *desc, image_data, image_data_size,
                                   access_permission, num_mipmap_levels, layout,
                                   row_pitch, slice_pitch, *image_out);

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

    if (mipmapped_array->handle == 0) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    const rocr::image::MipmappedArray* array =
        rocr::image::MipmappedArray::Convert(mipmapped_array->handle);
    if (!array || mip_level >= array->num_levels) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    // Create individual level image descriptor using AddressLib mip info
    const ADDR2_MIP_INFO* mip_infos = array->addr_output.pMipInfo;
    if (!mip_infos) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    const ADDR2_MIP_INFO& lvl = mip_infos[mip_level];

    // Create level descriptor with AddressLib computed dimensions
    hsa_ext_image_descriptor_t level_desc = array->desc;
    level_desc.width = lvl.pixelPitch ? lvl.pixelPitch :
        std::max<uint32_t>(1u, array->desc.width >> mip_level);
    level_desc.height = lvl.pixelHeight ? lvl.pixelHeight :
        std::max<uint32_t>(1u, array->desc.height >> mip_level);

    if (array->desc.geometry == HSA_EXT_IMAGE_GEOMETRY_3D) {
        level_desc.depth = std::max<uint32_t>(1u, array->desc.depth >> mip_level);
    }

    // Calculate level data pointer using AddressLib offset
    uint8_t* level_data = static_cast<uint8_t*>(array->data);
    if (level_data && lvl.offset > 0) {
        level_data += lvl.offset;
        if (lvl.macroBlockOffset) level_data += lvl.macroBlockOffset;
    }

    // Use ImageRuntime to create single-level image handle
    auto* rt = rocr::image::ImageRuntime::instance();
    if (!rt) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

    return rt->CreateImageHandle(agent, level_desc, level_data, array->permission,
                               HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR, 0, 0, *level_image_out);

    CATCH
}

hsa_status_t HSA_API
hsa_amd_mipmap_get_level_info(
    hsa_agent_t agent,
    const hsa_ext_image_t* mipmapped_array,
    uint32_t mip_level,
    hsa_ext_image_descriptor_t* level_desc) {

    TRY

    if (!mipmapped_array || !level_desc) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    const rocr::image::MipmappedArray* array =
        rocr::image::MipmappedArray::Convert(mipmapped_array->handle);
    if (!array || mip_level >= array->num_levels) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    const ADDR2_MIP_INFO* mip_infos = array->addr_output.pMipInfo;
    if (!mip_infos) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    const ADDR2_MIP_INFO& lvl = mip_infos[mip_level];

    *level_desc = array->desc;
    level_desc->width = lvl.pixelPitch ? lvl.pixelPitch :
        std::max<uint32_t>(1u, array->desc.width >> mip_level);
    level_desc->height = lvl.pixelHeight ? lvl.pixelHeight :
        std::max<uint32_t>(1u, array->desc.height >> mip_level);

    if (array->desc.geometry == HSA_EXT_IMAGE_GEOMETRY_3D) {
        level_desc->depth = std::max<uint32_t>(1u, array->desc.depth >> mip_level);
    }

    return HSA_STATUS_SUCCESS;

    CATCH
}

hsa_status_t HSA_API
hsa_amd_mipmap_get_level_pitch(
    hsa_agent_t agent,
    const hsa_ext_image_t* mipmapped_array,
    uint32_t mip_level,
    size_t* row_pitch,
    size_t* slice_pitch) {

    if (!mipmapped_array || !row_pitch || !slice_pitch) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    const rocr::image::MipmappedArray* array =
        rocr::image::MipmappedArray::Convert(mipmapped_array->handle);
    if (!array || mip_level >= array->num_levels) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    const ADDR2_MIP_INFO* mip_infos = array->addr_output.pMipInfo;
    if (!mip_infos) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    const ADDR2_MIP_INFO& lvl = mip_infos[mip_level];

    uint32_t pixel_bits = array->addr_output.pixelBits ? array->addr_output.pixelBits : array->addr_output.bpp;
    if (pixel_bits == 0) pixel_bits = ComputeBitsPerPixel(array->desc.format);
    uint32_t bytes_per_pixel = (pixel_bits + 7) / 8;

    if (lvl.pixelPitch && bytes_per_pixel) {
        *row_pitch = static_cast<size_t>(lvl.pixelPitch) * bytes_per_pixel;
    } else if (lvl.pitch && array->addr_output.bpp) {
        *row_pitch = static_cast<size_t>(lvl.pitch) * (array->addr_output.bpp / 8u);
    } else {
        *row_pitch = array->row_pitch; // fallback
    }

    // Calculate slice pitch
    uint32_t level_height = lvl.pixelHeight ? lvl.pixelHeight :
        std::max<uint32_t>(1u, array->desc.height >> mip_level);
    *slice_pitch = level_height * (*row_pitch);

    if (array->desc.geometry == HSA_EXT_IMAGE_GEOMETRY_3D) {
        uint32_t level_depth = std::max<uint32_t>(1u, array->desc.depth >> mip_level);
        *slice_pitch *= level_depth;
    }

    return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API
hsa_amd_mipmap_coord_to_address(
    const hsa_ext_image_t* mipmapped_array,
    uint32_t mip_level,
    const uint32_t coords[3],
    void** address) {

    TRY

    if (!mipmapped_array || !coords || !address) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    const rocr::image::MipmappedArray* array =
        rocr::image::MipmappedArray::Convert(mipmapped_array->handle);
    if (!array || mip_level >= array->num_levels) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    const ADDR2_MIP_INFO* mip_infos = array->addr_output.pMipInfo;
    if (!mip_infos || !array->data) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    const ADDR2_MIP_INFO& lvl = mip_infos[mip_level];
      uint8_t* base_bytes = static_cast<uint8_t*>(array->data);

      uint64_t combined_offset = lvl.offset;
    if (lvl.macroBlockOffset) combined_offset += lvl.macroBlockOffset;
    if (array->addr_output.mipChainInTail && lvl.mipTailOffset) {
        combined_offset = lvl.mipTailOffset; // tail overrides
    }

    // Get AddressLib handle for coordinate translation
    rocr::image::ImageManager* manager = rocr::image::ImageRuntime::instance()->image_manager(array->component);
    if (!manager) return HSA_STATUS_ERROR_INVALID_AGENT;

    ADDR_HANDLE addr_lib = manager->GetAddrLib();
    if (!addr_lib) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

    if (array->tile_mode == rocr::image::Image::TileMode::LINEAR) {
        uint32_t pixel_bits = array->addr_output.pixelBits ? array->addr_output.pixelBits : array->addr_output.bpp;
        if (pixel_bits == 0) pixel_bits = ComputeBitsPerPixel(array->desc.format);
        uint32_t bytes_per_pixel = (pixel_bits + 7) / 8;

        size_t row_pitch = 0, slice_pitch = 0;
        hsa_status_t pitch_status = hsa_amd_mipmap_get_level_pitch(hsa_agent_t{}, mipmapped_array, mip_level, &row_pitch, &slice_pitch);
        if (pitch_status != HSA_STATUS_SUCCESS) {
            return pitch_status;
        }

        combined_offset += coords[2] * slice_pitch + coords[1] * row_pitch + coords[0] * bytes_per_pixel;
    } else {
        ADDR2_COMPUTE_SURFACE_ADDRFROMCOORD_INPUT addr_input = {0};
        ADDR2_COMPUTE_SURFACE_ADDRFROMCOORD_OUTPUT addr_output = {0};

        addr_input.size = sizeof(addr_input);
        addr_input.x = coords[0];
        addr_input.y = coords[1];
        addr_input.slice = coords[2];
        addr_input.mipId = mip_level;
        addr_input.unalignedWidth = array->desc.width;
        addr_input.unalignedHeight = array->desc.height;
        addr_input.numSlices = (array->desc.array_size > 0) ? array->desc.array_size : 1;
        addr_input.numMipLevels = array->num_levels;
        addr_input.resourceType = GetAddrResourceType(array->desc.geometry);
        addr_input.bpp = array->addr_output.bpp;
        addr_input.swizzleMode = ADDR_SW_64KB_R_X;
        addr_input.flags.texture = 1;

        addr_output.size = sizeof(addr_output);

        if (Addr2ComputeSurfaceAddrFromCoord(addr_lib, &addr_input, &addr_output) == ADDR_OK) {
            combined_offset = addr_output.addr;
        } else {
            return HSA_STATUS_ERROR_INVALID_ARGUMENT;
        }
    }

    *address = base_bytes + combined_offset;
    return HSA_STATUS_SUCCESS;

    CATCH
}

hsa_status_t HSA_API
hsa_amd_mipmap_get_all_level_info(
    hsa_agent_t agent,
    const hsa_ext_image_t* mipmapped_array,
    hsa_ext_image_descriptor_t* level_descriptors,
    size_t* row_pitches,
    size_t* slice_pitches,
    uint32_t max_levels) {

    TRY

    if (!mipmapped_array || !level_descriptors || !row_pitches || !slice_pitches) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    const rocr::image::MipmappedArray* array =
        rocr::image::MipmappedArray::Convert(mipmapped_array->handle);
    if (!array) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    uint32_t levels_to_process = std::min(max_levels, array->num_levels);
    const ADDR2_MIP_INFO* mip_infos = array->addr_output.pMipInfo;
    if (!mip_infos) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    uint32_t pixel_bits = array->addr_output.pixelBits ? array->addr_output.pixelBits : array->addr_output.bpp;
    if (pixel_bits == 0) pixel_bits = ComputeBitsPerPixel(array->desc.format);
    uint32_t bytes_per_pixel = (pixel_bits + 7) / 8;

    for (uint32_t level = 0; level < levels_to_process; ++level) {
        const ADDR2_MIP_INFO& lvl = mip_infos[level];

        level_descriptors[level] = array->desc;
        level_descriptors[level].width = lvl.pixelPitch ? lvl.pixelPitch :
            std::max<uint32_t>(1u, array->desc.width >> level);
        level_descriptors[level].height = lvl.pixelHeight ? lvl.pixelHeight :
            std::max<uint32_t>(1u, array->desc.height >> level);

        if (array->desc.geometry == HSA_EXT_IMAGE_GEOMETRY_3D) {
            level_descriptors[level].depth = std::max<uint32_t>(1u, array->desc.depth >> level);
        }

        // Calculate pitches
        if (lvl.pixelPitch && bytes_per_pixel) {
            row_pitches[level] = static_cast<size_t>(lvl.pixelPitch) * bytes_per_pixel;
        } else if (lvl.pitch && array->addr_output.bpp) {
            row_pitches[level] = static_cast<size_t>(lvl.pitch) * (array->addr_output.bpp / 8u);
        } else {
            row_pitches[level] = array->row_pitch;
        }

        uint32_t level_height = level_descriptors[level].height;
        slice_pitches[level] = level_height * row_pitches[level];

        if (array->desc.geometry == HSA_EXT_IMAGE_GEOMETRY_3D) {
            slice_pitches[level] *= level_descriptors[level].depth;
        }
    }

    return HSA_STATUS_SUCCESS;

    CATCH
}

hsa_status_t HSA_API
hsa_amd_mipmap_compute_metadata_info(
    hsa_agent_t agent,
    const hsa_ext_image_t* mipmapped_array,
    uint32_t mip_level,
    hsa_amd_mipmap_metadata_info_t* metadata_info) {

    TRY

    if (!mipmapped_array || !metadata_info) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    const rocr::image::MipmappedArray* array =
        rocr::image::MipmappedArray::Convert(mipmapped_array->handle);
    if (!array || mip_level >= array->num_levels) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    metadata_info->metadata_size = 0;
    metadata_info->cmask_size = 0;
    metadata_info->cmask_alignment = 0;
    metadata_info->has_metadata = false;

    rocr::image::ImageManager* manager = rocr::image::ImageRuntime::instance()->image_manager(agent);
    if (!manager) return HSA_STATUS_ERROR_INVALID_AGENT;

    ADDR_HANDLE addr_lib = manager->GetAddrLib();
    if (!addr_lib) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

    const ADDR2_MIP_INFO* mip_infos = array->addr_output.pMipInfo;
    if (!mip_infos) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

    const ADDR2_MIP_INFO& lvl = mip_infos[mip_level];

    // Compute HTile metadata for depth surfaces
    if (array->desc.format.channel_order == HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH ||
        array->desc.format.channel_order == HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH_STENCIL) {

        ADDR2_COMPUTE_HTILE_INFO_INPUT htile_input = {0};
        ADDR2_COMPUTE_HTILE_INFO_OUTPUT htile_output = {0};

        htile_input.size = sizeof(htile_input);
        htile_input.depthFlags.depth = 1;
        htile_input.swizzleMode = ADDR_SW_64KB_Z_X;
        htile_input.unalignedWidth = lvl.pixelPitch;
        htile_input.unalignedHeight = lvl.pixelHeight;
        htile_input.numSlices = (array->desc.array_size > 0) ? array->desc.array_size : 1;
        // HTile structure doesn't have mipId field, use mip level in other calculations

        htile_output.size = sizeof(htile_output);

        if (Addr2ComputeHtileInfo(addr_lib, &htile_input, &htile_output) == ADDR_OK) {
            metadata_info->metadata_size = htile_output.htileBytes;
            metadata_info->has_metadata = true;
        }
    }

    // Compute CMask metadata for color surfaces
    if (array->desc.format.channel_order != HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH &&
        array->desc.format.channel_order != HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH_STENCIL) {

        ADDR2_COMPUTE_CMASK_INFO_INPUT cmask_input = {0};
        ADDR2_COMPUTE_CMASK_INFO_OUTPUT cmask_output = {0};

        cmask_input.size = sizeof(cmask_input);
        cmask_input.colorFlags.color = 1;
        cmask_input.swizzleMode = ADDR_SW_64KB_R_X;
        cmask_input.unalignedWidth = lvl.pixelPitch;
        cmask_input.unalignedHeight = lvl.pixelHeight;
        cmask_input.numSlices = (array->desc.array_size > 0) ? array->desc.array_size : 1;

        cmask_output.size = sizeof(cmask_output);

        if (Addr2ComputeCmaskInfo(addr_lib, &cmask_input, &cmask_output) == ADDR_OK) {
            metadata_info->metadata_size = cmask_output.cmaskBytes;
            metadata_info->cmask_size = cmask_output.cmaskBytes;
            metadata_info->cmask_alignment = cmask_output.baseAlign;
            metadata_info->has_metadata = true;
        }
    }

    return HSA_STATUS_SUCCESS;

    CATCH
}

hsa_status_t HSA_API
hsa_amd_mipmap_get_metadata_info(
    hsa_agent_t agent,
    const hsa_ext_image_t* mipmapped_array,
    uint32_t mip_level,
    size_t* metadata_size) {

    TRY

    if (!mipmapped_array || !metadata_size) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    if (mipmapped_array->handle == 0) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    const rocr::image::MipmappedArray* array =
        reinterpret_cast<const rocr::image::MipmappedArray*>(mipmapped_array->handle);

    if (!array) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    if (mip_level >= array->levels_allocated) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    // Call the compute_metadata_info function and extract just the metadata_size
    hsa_amd_mipmap_metadata_info_t metadata_info = {};
    hsa_status_t status = hsa_amd_mipmap_compute_metadata_info(agent, mipmapped_array, mip_level, &metadata_info);

    if (status == HSA_STATUS_SUCCESS) {
        *metadata_size = metadata_info.metadata_size;
    }

    return status;

    CATCH
}

} // extern "C"

} // namespace image
} // namespace rocr
