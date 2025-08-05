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
#include "image_manager_kv.h"

#include <assert.h>

#include <algorithm>
#include <climits>

#include "core/inc/runtime.h"
#include "hsakmt/hsakmt.h"
#include "inc/hsa_ext_amd.h"
#include "core/inc/hsa_internal.h"
#include "core/inc/hsa_ext_amd_impl.h"
#include "core/inc/runtime.h"
#include "addrlib/inc/addrinterface.h"
#include "addrlib/src/core/addrlib.h"
#include "image_runtime.h"
#include "resource.h"
#include "resource_kv.h"
#include "util.h"
#include "device_info.h"

namespace rocr {
namespace image {

ASSERT_SIZE_UINT32(SQ_BUF_RSRC_WORD0)
ASSERT_SIZE_UINT32(SQ_BUF_RSRC_WORD1)
ASSERT_SIZE_UINT32(SQ_BUF_RSRC_WORD2)
ASSERT_SIZE_UINT32(SQ_BUF_RSRC_WORD3)

ASSERT_SIZE_UINT32(SQ_IMG_RSRC_WORD0)
ASSERT_SIZE_UINT32(SQ_IMG_RSRC_WORD1)
ASSERT_SIZE_UINT32(SQ_IMG_RSRC_WORD2)
ASSERT_SIZE_UINT32(SQ_IMG_RSRC_WORD3)
ASSERT_SIZE_UINT32(SQ_IMG_RSRC_WORD4)
ASSERT_SIZE_UINT32(SQ_IMG_RSRC_WORD5)
ASSERT_SIZE_UINT32(SQ_IMG_RSRC_WORD6)
ASSERT_SIZE_UINT32(SQ_IMG_RSRC_WORD7)

ASSERT_SIZE_UINT32(SQ_IMG_SAMP_WORD0)
ASSERT_SIZE_UINT32(SQ_IMG_SAMP_WORD1)
ASSERT_SIZE_UINT32(SQ_IMG_SAMP_WORD2)
ASSERT_SIZE_UINT32(SQ_IMG_SAMP_WORD3)

ImageManagerKv::ImageManagerKv() : ImageManager() {}

ImageManagerKv::~ImageManagerKv() {}

hsa_status_t ImageManagerKv::Initialize(hsa_agent_t agent_handle) {
  agent_ = agent_handle;

  hsa_status_t status = GetGPUAsicID(agent_, &chip_id_);
  uint32_t major_ver = MajorVerFromDevID(chip_id_);
  assert(status == HSA_STATUS_SUCCESS);

  status = HSA::hsa_agent_get_info(
      agent_, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_ASIC_FAMILY_ID), &family_type_);
  assert(status == HSA_STATUS_SUCCESS);

  HsaGpuTileConfig tileConfig = {0};
  unsigned int tc[40];
  unsigned int mtc[40];
  tileConfig.TileConfig = &tc[0];
  tileConfig.NumTileConfigs = 40;
  tileConfig.MacroTileConfig = &mtc[0];
  tileConfig.NumMacroTileConfigs = 40;
  uint32_t node_id = 0;
  status = HSA::hsa_agent_get_info(
      agent_, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DRIVER_NODE_ID), &node_id);
  assert(status == HSA_STATUS_SUCCESS);
  hsa_status_t stat = HSA::hsa_get_tile_config(agent_handle, &tileConfig);
  assert(stat == HSA_STATUS_SUCCESS);

  // Initialize address library.
  // TODO(bwicakso) hard coded based on UGL parameters.
  // Need to get this information from KMD.
  addr_lib_ = NULL;
  ADDR_CREATE_INPUT addr_create_input = {0};
  ADDR_CREATE_OUTPUT addr_create_output = {0};

  if (major_ver >= 9) {
    addr_create_input.chipEngine = CIASICIDGFXENGINE_ARCTICISLAND;
  } else {
    addr_create_input.chipEngine = CIASICIDGFXENGINE_SOUTHERNISLAND;
  }

  addr_create_input.chipFamily = family_type_;
  addr_create_input.chipRevision = 0;  // TODO(bwicakso): find how to get this.

  ADDR_CREATE_FLAGS create_flags = {};
  create_flags.value = 0;
  create_flags.useTileIndex = 1;
  addr_create_input.createFlags = create_flags;

  addr_create_input.callbacks.allocSysMem = AllocSysMem;
  addr_create_input.callbacks.freeSysMem = FreeSysMem;
  addr_create_input.callbacks.debugPrint = 0;

  ADDR_REGISTER_VALUE reg_val = {0};
  reg_val.gbAddrConfig = tileConfig.GbAddrConfig;
  reg_val.noOfBanks = tileConfig.NumBanks;
  reg_val.noOfRanks = tileConfig.NumRanks;
  reg_val.pTileConfig = tileConfig.TileConfig;
  reg_val.noOfEntries = tileConfig.NumTileConfigs;
  reg_val.noOfMacroEntries = tileConfig.NumMacroTileConfigs;
  reg_val.pMacroTileConfig = tileConfig.MacroTileConfig;

  addr_create_input.regValue = reg_val;

  addr_create_input.minPitchAlignPixels = 0;

  ADDR_E_RETURNCODE addr_ret =
      AddrCreate(&addr_create_input, &addr_create_output);

  if (addr_ret == ADDR_OK) {
    addr_lib_ = addr_create_output.hLib;
  } else {
    return HSA_STATUS_ERROR;
  }

  // The ImageManagerKv::Initialize is called on the first call to
  // hsa_ext_image_*, so checking the coherency mode here is fine as long as
  // the change to the coherency mode happens before a call to
  // hsa_ext_image_create.
  hsa_amd_coherency_type_t coherency_type;
  status = AMD::hsa_amd_coherency_get_type(agent_, &coherency_type);
  assert(status == HSA_STATUS_SUCCESS);
  mtype_ = (coherency_type == HSA_AMD_COHERENCY_TYPE_COHERENT) ? 3 : 1;

  // TODO: handle the case where the call to hsa_set_memory_type happens after
  // hsa_ext_image_create.

  hsa_region_t local_region = {0};
  status = HSA::hsa_agent_iterate_regions(agent_, GetLocalMemoryRegion, &local_region);
  assert(status == HSA_STATUS_SUCCESS);

  local_memory_base_address_ = 0;
  if (local_region.handle != 0) {
    status = HSA::hsa_region_get_info(local_region,
                                      static_cast<hsa_region_info_t>(HSA_AMD_REGION_INFO_BASE),
                                      &local_memory_base_address_);
    assert(status == HSA_STATUS_SUCCESS);
  }

  // Zeroed the queue object so it can be created on demand.
  blit_queue_.queue_ = NULL;
  blit_queue_.cached_index_ = 0;

  return HSA_STATUS_SUCCESS;
}

void ImageManagerKv::Cleanup() {
  if (blit_queue_.queue_ != NULL) {
    HSA::hsa_queue_destroy(blit_queue_.queue_);
  }

  if (addr_lib_ != NULL) {
    AddrDestroy(addr_lib_);
  }
}

ImageProperty ImageManagerKv::GetImageProperty(
    hsa_agent_t component, const hsa_ext_image_format_t& format,
    hsa_ext_image_geometry_t geometry) const {
  return ImageLut().MapFormat(format, geometry);
}

void ImageManagerKv::GetImageInfoMaxDimension(hsa_agent_t component,
                                              hsa_ext_image_geometry_t geometry,
                                              uint32_t& width, uint32_t& height,
                                              uint32_t& depth,
                                              uint32_t& array_size) const {
  width = ImageLut().GetMaxWidth(geometry);
  height = ImageLut().GetMaxHeight(geometry);
  depth = ImageLut().GetMaxDepth(geometry);
  array_size = ImageLut().GetMaxArraySize(geometry);
}

hsa_status_t ImageManagerKv::CalculateImageSizeAndAlignment(
    hsa_agent_t component, const hsa_ext_image_descriptor_t& desc,
    hsa_ext_image_data_layout_t image_data_layout,
    size_t image_data_row_pitch,
    size_t image_data_slice_pitch,
    hsa_ext_image_data_info_t& image_info) const {
  ADDR_COMPUTE_SURFACE_INFO_OUTPUT out = {0};
  hsa_profile_t profile;

  hsa_status_t status = HSA::hsa_agent_get_info(component, HSA_AGENT_INFO_PROFILE, &profile);
  if (status != HSA_STATUS_SUCCESS) return status;

  Image::TileMode tileMode = Image::TileMode::LINEAR;
  if (image_data_layout == HSA_EXT_IMAGE_DATA_LAYOUT_OPAQUE) {
    tileMode = (profile == HSA_PROFILE_BASE &&
                desc.geometry != HSA_EXT_IMAGE_GEOMETRY_1DB)?
      Image::TileMode::TILED : Image::TileMode::LINEAR;
  }
  if (!GetAddrlibSurfaceInfo(component, desc, tileMode,
        image_data_row_pitch, image_data_slice_pitch, out)) {
    return HSA_STATUS_ERROR;
  }

  size_t rowPitch   = (out.bpp >> 3) * out.pitch;
  size_t slicePitch = rowPitch * out.height;
  if (desc.geometry != HSA_EXT_IMAGE_GEOMETRY_1DB &&
      image_data_layout == HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR &&
      ((image_data_row_pitch && (rowPitch != image_data_row_pitch)) ||
       (image_data_slice_pitch && (slicePitch != image_data_slice_pitch)))) {
    return static_cast<hsa_status_t>(HSA_EXT_STATUS_ERROR_IMAGE_PITCH_UNSUPPORTED);
  }

  image_info.size = out.surfSize;
  assert(image_info.size != 0);
  image_info.alignment = out.baseAlign;
  assert(image_info.alignment != 0);

  return HSA_STATUS_SUCCESS;
}

static const uint64_t kLimitSystem = 1ULL << 48;

bool ImageManagerKv::IsLocalMemory(const void* address) const {
  uintptr_t u_address = reinterpret_cast<uintptr_t>(address);

  uint32_t major_ver = MajorVerFromDevID(chip_id_);

  if (major_ver >= 8) {
    return true;
  }
#ifdef HSA_LARGE_MODEL
  // Fast path without querying local memory region info.
  // User mode system memory addressable by CPU is 0 to 2^48.
  return (u_address >= kLimitSystem);
#else
  // No local memory on 32 bit.
  return false;
#endif
}

hsa_status_t ImageManagerKv::PopulateImageSrd(Image& image, const metadata_amd_t* descriptor) const {
  metadata_amd_ci_vi_t* desc = (metadata_amd_ci_vi_t*)descriptor;
  bool atc_access = true;
  uint32_t mtype = mtype_;
  const void* image_data_addr = image.data;

  ImageProperty image_prop = ImageLut().MapFormat(image.desc.format, image.desc.geometry);
  if((image_prop.cap == HSA_EXT_IMAGE_CAPABILITY_NOT_SUPPORTED) ||
     (image_prop.element_size == 0))
    return (hsa_status_t)HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED;

  uint32_t hwPixelSize =
      ImageLut().GetPixelSize(desc->word1.bitfields.data_format, desc->word1.bitfields.num_format);
  if(image_prop.element_size!=hwPixelSize)
    return (hsa_status_t)HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED;

  const Swizzle swizzle = ImageLut().MapSwizzle(image.desc.format.channel_order);

  if (IsLocalMemory(image.data)) {
    atc_access = false;
    mtype = 1;
    image_data_addr = reinterpret_cast<const void*>(
        reinterpret_cast<uintptr_t>(image.data) - local_memory_base_address_);
  }

  image.srd[0]=desc->word0.u32_all;
  image.srd[1]=desc->word1.u32_all;
  image.srd[2]=desc->word2.u32_all;
  image.srd[3]=desc->word3.u32_all;
  image.srd[4]=desc->word4.u32_all;
  image.srd[5]=desc->word5.u32_all;
  image.srd[6]=desc->word6.u32_all;
  image.srd[7]=desc->word7.u32_all;

  ((SQ_IMG_RSRC_WORD0*)(&image.srd[0]))->bits.base_address = PtrLow40Shift8(image_data_addr);
  ((SQ_IMG_RSRC_WORD1*)(&image.srd[1]))->bits.base_address_hi = PtrHigh64Shift40(image_data_addr);
  ((SQ_IMG_RSRC_WORD1*)(&image.srd[1]))->bits.data_format = image_prop.data_format;
  ((SQ_IMG_RSRC_WORD1*)(&image.srd[1]))->bits.num_format = image_prop.data_type;
  ((SQ_IMG_RSRC_WORD1*)(&image.srd[1]))->bits.mtype = mtype;
  ((SQ_IMG_RSRC_WORD3*)(&image.srd[3]))->bits.atc=atc_access;
  ((SQ_IMG_RSRC_WORD3*)(&image.srd[3]))->bits.dst_sel_x = swizzle.x;
  ((SQ_IMG_RSRC_WORD3*)(&image.srd[3]))->bits.dst_sel_y = swizzle.y;
  ((SQ_IMG_RSRC_WORD3*)(&image.srd[3]))->bits.dst_sel_z = swizzle.z;
  ((SQ_IMG_RSRC_WORD3*)(&image.srd[3]))->bits.dst_sel_w = swizzle.w;
  ((SQ_IMG_RSRC_WORD7*)(&image.srd[7]))->bits.meta_data_address += PtrLow40Shift8(image_data_addr);

  //Looks like this is only used for CPU copies.
  image.row_pitch = (desc->word4.bits.pitch+1)*image_prop.element_size;
  image.slice_pitch = image.row_pitch * (desc->word2.bits.height+1);

  //Used by HSAIL shader ABI
  image.srd[8] = image.desc.format.channel_type;
  image.srd[9] = image.desc.format.channel_order;
  image.srd[10] = static_cast<uint32_t>(image.desc.width);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t ImageManagerKv::PopulateImageSrd(Image& image) const {
  ImageProperty image_prop = ImageLut().MapFormat(image.desc.format, image.desc.geometry);
  assert(image_prop.cap != HSA_EXT_IMAGE_CAPABILITY_NOT_SUPPORTED);
  assert(image_prop.element_size != 0);

  bool atc_access = true;
  uint32_t mtype = mtype_;
  const void* image_data_addr = image.data;

  if (IsLocalMemory(image.data)) {
    atc_access = false;
    mtype = 1;
    image_data_addr = reinterpret_cast<const void*>(
        reinterpret_cast<uintptr_t>(image.data) - local_memory_base_address_);
  }

  if (image.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_1DB) {
    SQ_BUF_RSRC_WORD0 word0;
    SQ_BUF_RSRC_WORD1 word1;
    SQ_BUF_RSRC_WORD2 word2;
    SQ_BUF_RSRC_WORD3 word3;

    word0.u32_all = 0;
    word0.bits.base_address = PtrLow32(image_data_addr);

    word1.u32_all = 0;
    word1.bits.base_address_hi = PtrHigh32(image_data_addr);
    word1.bits.stride = image_prop.element_size;
    word1.bits.swizzle_enable = false;
    word1.bits.cache_swizzle = false;

    uint32_t major_ver = MajorVerFromDevID(chip_id_);
    word2.bits.num_records = (major_ver < 8) ?
                image.desc.width : image.desc.width * image_prop.element_size;

    const Swizzle swizzle = ImageLut().MapSwizzle(image.desc.format.channel_order);
    word3.u32_all = 0;
    word3.bits.dst_sel_x = swizzle.x;
    word3.bits.dst_sel_y = swizzle.y;
    word3.bits.dst_sel_z = swizzle.z;
    word3.bits.dst_sel_w = swizzle.w;
    word3.bits.num_format = image_prop.data_type;
    word3.bits.data_format = image_prop.data_format;
    word3.bits.atc = atc_access;
    word3.bits.element_size = image_prop.element_size;
    word3.bits.type = ImageLut().MapGeometry(image.desc.geometry);
    word3.bits.mtype = mtype;

    image.srd[0] = word0.u32_all;
    image.srd[1] = word1.u32_all;
    image.srd[2] = word2.u32_all;
    image.srd[3] = word3.u32_all;

    image.row_pitch = image.desc.width * image_prop.element_size;
    image.slice_pitch = image.row_pitch;
  } else {
    SQ_IMG_RSRC_WORD0 word0;
    SQ_IMG_RSRC_WORD1 word1;
    SQ_IMG_RSRC_WORD2 word2;
    SQ_IMG_RSRC_WORD3 word3;
    SQ_IMG_RSRC_WORD4 word4;
    SQ_IMG_RSRC_WORD5 word5;
    SQ_IMG_RSRC_WORD6 word6;
    SQ_IMG_RSRC_WORD7 word7;

    ADDR_COMPUTE_SURFACE_INFO_OUTPUT out = {0};
    if (!GetAddrlibSurfaceInfo(image.component, image.desc, image.tile_mode,
          image.row_pitch, image.slice_pitch, out)) {
      return HSA_STATUS_ERROR;
    }

    assert((out.bpp / 8) == image_prop.element_size);

    const size_t row_pitch_size = out.pitch * image_prop.element_size;

    word0.bits.base_address = PtrLow40Shift8(image_data_addr);

    word1.u32_all = 0;
    word1.bits.base_address_hi = PtrHigh64Shift40(image_data_addr);
    word1.bits.min_lod = 0;
    word1.bits.data_format = image_prop.data_format;
    word1.bits.num_format = image_prop.data_type;
    word1.bits.mtype = mtype;

    word2.u32_all = 0;
    word2.bits.width = image.desc.width - 1;
    word2.bits.height = image.desc.height - 1;
    word2.bits.perf_mod = 0;
    word2.bits.interlaced = 0;

    const Swizzle swizzle = ImageLut().MapSwizzle(image.desc.format.channel_order);
    word3.u32_all = 0;
    word3.bits.dst_sel_x = swizzle.x;
    word3.bits.dst_sel_y = swizzle.y;
    word3.bits.dst_sel_z = swizzle.z;
    word3.bits.dst_sel_w = swizzle.w;
    word3.bits.tiling_index = out.tileIndex;
    word3.bits.pow2_pad = (IsPowerOfTwo(row_pitch_size) && IsPowerOfTwo(image.desc.height)) ? 1 : 0;
    word3.bits.type = ImageLut().MapGeometry(image.desc.geometry);
    word3.bits.atc = atc_access;

    const bool image_array =
        (image.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_1DA ||
         image.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_2DA ||
         image.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_2DADEPTH);
    const bool image_3d = (image.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_3D);

    word4.u32_all = 0;
    word4.bits.depth =
        (image_array)
            ? std::max(image.desc.array_size, static_cast<size_t>(1)) - 1
            : (image_3d) ? image.desc.depth - 1 : 0;
    word4.bits.pitch = out.pitch - 1;

    word5.u32_all = 0;
    word5.bits.last_array =
        (image_array)
            ? (std::max(image.desc.array_size, static_cast<size_t>(1)) - 1)
            : 0;

    word6.u32_all = 0;
    word7.u32_all = 0;

    image.srd[0] = word0.u32_all;
    image.srd[1] = word1.u32_all;
    image.srd[2] = word2.u32_all;
    image.srd[3] = word3.u32_all;
    image.srd[4] = word4.u32_all;
    image.srd[5] = word5.u32_all;
    image.srd[6] = word6.u32_all;
    image.srd[7] = word7.u32_all;

    image.row_pitch = row_pitch_size;
    image.slice_pitch = out.sliceSize;
  }

  image.srd[8] = image.desc.format.channel_type;
  image.srd[9] = image.desc.format.channel_order;
  image.srd[10] = static_cast<uint32_t>(image.desc.width);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t ImageManagerKv::ModifyImageSrd(
    Image& image, hsa_ext_image_format_t& new_format) const {
  image.desc.format = new_format;

  ImageProperty image_prop = ImageLut().MapFormat(image.desc.format, image.desc.geometry);
  assert(image_prop.cap != HSA_EXT_IMAGE_CAPABILITY_NOT_SUPPORTED);
  assert(image_prop.element_size != 0);

  if (image.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_1DB) {
    const Swizzle swizzle = ImageLut().MapSwizzle(image.desc.format.channel_order);
    SQ_BUF_RSRC_WORD3* word3 =
        reinterpret_cast<SQ_BUF_RSRC_WORD3*>(&image.srd[3]);
    word3->bits.dst_sel_x = swizzle.x;
    word3->bits.dst_sel_y = swizzle.y;
    word3->bits.dst_sel_z = swizzle.z;
    word3->bits.dst_sel_w = swizzle.w;
    word3->bits.num_format = image_prop.data_type;
    word3->bits.data_format = image_prop.data_format;
  } else {
    SQ_IMG_RSRC_WORD1* word1 =
        reinterpret_cast<SQ_IMG_RSRC_WORD1*>(&image.srd[1]);
    word1->bits.data_format = image_prop.data_format;
    word1->bits.num_format = image_prop.data_type;

    const Swizzle swizzle = ImageLut().MapSwizzle(image.desc.format.channel_order);
    SQ_IMG_RSRC_WORD3* word3 =
        reinterpret_cast<SQ_IMG_RSRC_WORD3*>(&image.srd[3]);
    word3->bits.dst_sel_x = swizzle.x;
    word3->bits.dst_sel_y = swizzle.y;
    word3->bits.dst_sel_z = swizzle.z;
    word3->bits.dst_sel_w = swizzle.w;
  }

  image.srd[8] = image.desc.format.channel_type;
  image.srd[9] = image.desc.format.channel_order;
  image.srd[10] = static_cast<uint32_t>(image.desc.width);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t ImageManagerKv::PopulateSamplerSrd(Sampler& sampler) const {
  const hsa_ext_sampler_descriptor_v2_t &sampler_descriptor = sampler.desc;

  SQ_IMG_SAMP_WORD0 word0;
  SQ_IMG_SAMP_WORD1 word1;
  SQ_IMG_SAMP_WORD2 word2;
  SQ_IMG_SAMP_WORD3 word3;

  word0.u32_all = 0;
  hsa_status_t status = convertAddressMode<SQ_IMG_SAMP_WORD0, SQ_TEX_CLAMP>
                                         (word0, sampler_descriptor.address_modes);
  if (status != HSA_STATUS_SUCCESS) return status;
  word0.bits.force_unormalized = (sampler_descriptor.coordinate_mode ==
                                  HSA_EXT_SAMPLER_COORDINATE_MODE_UNNORMALIZED);

  word1.u32_all = 0;
  word1.bits.max_lod = 4095;

  word2.u32_all = 0;
  switch (sampler_descriptor.filter_mode) {
    case HSA_EXT_SAMPLER_FILTER_MODE_NEAREST:
      word2.bits.xy_mag_filter = static_cast<int>(SQ_TEX_XY_FILTER_POINT);
      break;
    case HSA_EXT_SAMPLER_FILTER_MODE_LINEAR:
      word2.bits.xy_mag_filter = static_cast<int>(SQ_TEX_XY_FILTER_BILINEAR);
      break;
    default:
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  word2.bits.xy_min_filter = word2.bits.xy_mag_filter;
  word2.bits.z_filter = SQ_TEX_Z_FILTER_NONE;
  word2.bits.mip_filter = SQ_TEX_MIP_FILTER_NONE;

  word3.u32_all = 0;

  // TODO: check this bit with HSAIL spec.
  word3.bits.border_color_type = SQ_TEX_BORDER_COLOR_TRANS_BLACK;

  sampler.srd[0] = word0.u32_all;
  sampler.srd[1] = word1.u32_all;
  sampler.srd[2] = word2.u32_all;
  sampler.srd[3] = word3.u32_all;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t ImageManagerKv::CopyBufferToImage(
    const void* src_memory, size_t src_row_pitch, size_t src_slice_pitch,
    const Image& dst_image, const hsa_ext_image_region_t& image_region) {
  if (BlitQueueInit().queue_ == NULL) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  return ImageRuntime::instance()->blit_kernel().CopyBufferToImage(
      blit_queue_, blit_code_catalog_, src_memory, src_row_pitch, src_slice_pitch, dst_image,
      image_region);
}

hsa_status_t ImageManagerKv::CopyImageToBuffer(
    const Image& src_image, void* dst_memory, size_t dst_row_pitch,
    size_t dst_slice_pitch, const hsa_ext_image_region_t& image_region) {
  if (BlitQueueInit().queue_ == NULL) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  return ImageRuntime::instance()->blit_kernel().CopyImageToBuffer(
      blit_queue_, blit_code_catalog_, src_image, dst_memory, dst_row_pitch, dst_slice_pitch,
      image_region);
}

hsa_status_t ImageManagerKv::CopyImage(const Image& dst_image,
                                       const Image& src_image,
                                       const hsa_dim3_t& dst_origin,
                                       const hsa_dim3_t& src_origin,
                                       const hsa_dim3_t size) {
  if (BlitQueueInit().queue_ == NULL) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  const hsa_ext_image_format_t src_format = src_image.desc.format;
  const hsa_ext_image_channel_order32_t src_order = src_format.channel_order;
  const hsa_ext_image_channel_type32_t src_type = src_format.channel_type;

  const hsa_ext_image_format_t dst_format = dst_image.desc.format;
  const hsa_ext_image_channel_order32_t dst_order = dst_format.channel_order;
  const hsa_ext_image_channel_type32_t dst_type = dst_format.channel_type;

  BlitKernel::KernelOp copy_type = BlitKernel::KERNEL_OP_COPY_IMAGE_DEFAULT;

  if ((src_order == dst_order) && (src_type == dst_type)) {
    return ImageRuntime::instance()->blit_kernel().CopyImage(blit_queue_, blit_code_catalog_,
                                                             dst_image, src_image, dst_origin,
                                                             src_origin, size, copy_type);
  }

  // Source and destination format must be the same, except for
  // SRGBA <--> RGBA images.
  if ((src_type == HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT8) &&
      (dst_type == HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT8)) {
    if ((src_order == HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBA) &&
        (dst_order == HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA)) {
      copy_type = BlitKernel::KERNEL_OP_COPY_IMAGE_STANDARD_TO_LINEAR;
    } else if ((src_order == HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA) &&
               (dst_order == HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBA)) {
      copy_type = BlitKernel::KERNEL_OP_COPY_IMAGE_LINEAR_TO_STANDARD;
    }

    if (copy_type != BlitKernel::KERNEL_OP_COPY_IMAGE_DEFAULT) {
      // KV and CZ don't have write support for SRGBA image, so treat the
      // destination image as RGBA image.
      SQ_IMG_RSRC_WORD1* word1 = reinterpret_cast<SQ_IMG_RSRC_WORD1*>(
          &const_cast<Image&>(dst_image).srd[1]);

      // Destination can be linear or standard, preserve the original value.
      uint32_t num_format_original = word1->bits.num_format;
      word1->bits.num_format = TYPE_UNORM;

      hsa_status_t status = ImageRuntime::instance()->blit_kernel().CopyImage(
          blit_queue_, blit_code_catalog_, dst_image, src_image, dst_origin, src_origin, size,
          copy_type);

      // Revert to the original format after the copy operation is finished.
      word1->bits.num_format = num_format_original;

      return status;
    }
  }

  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}

hsa_status_t ImageManagerKv::FillImage(const Image& image, const void* pattern,
                                       const hsa_ext_image_region_t& region) {
  if (BlitQueueInit().queue_ == NULL) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  Image* image_view = const_cast<Image*>(&image);

  SQ_BUF_RSRC_WORD3* word3_buff = NULL;
  SQ_IMG_RSRC_WORD3* word3_image = NULL;
  uint32_t dst_sel_w_original = 0;
  if (image_view->desc.format.channel_type ==
      HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_101010) {
    // Force GPU to ignore the last two bits (alpha bits).
    if (image_view->desc.geometry == HSA_EXT_IMAGE_GEOMETRY_1DB) {
      word3_buff = reinterpret_cast<SQ_BUF_RSRC_WORD3*>(&image_view->srd[3]);
      dst_sel_w_original = word3_buff->bits.dst_sel_w;
      word3_buff->bits.dst_sel_w = SEL_0;
    } else {
      word3_image = reinterpret_cast<SQ_IMG_RSRC_WORD3*>(&image_view->srd[3]);
      dst_sel_w_original = word3_image->bits.dst_sel_w;
      word3_image->bits.dst_sel_w = SEL_0;
    }
  }

  SQ_IMG_RSRC_WORD1* word1 = NULL;
  uint32_t num_format_original = 0;
  const void* new_pattern = pattern;
  float fill_value[4] = {0};
  switch (image_view->desc.format.channel_order) {
    case HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBA:
    case HSA_EXT_IMAGE_CHANNEL_ORDER_SRGB:
    case HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBX:
    case HSA_EXT_IMAGE_CHANNEL_ORDER_SBGRA: {
      // KV and CZ don't have write support for SRGBA image, so convert pattern
      // to standard form and treat the image as RGBA image.
      const float* pattern_f = reinterpret_cast<const float*>(pattern);
      fill_value[0] = LinearToStandardRGB(pattern_f[0]);
      fill_value[1] = LinearToStandardRGB(pattern_f[1]);
      fill_value[2] = LinearToStandardRGB(pattern_f[2]);
      fill_value[3] = pattern_f[3];
      new_pattern = fill_value;

      word1 = reinterpret_cast<SQ_IMG_RSRC_WORD1*>(&image_view->srd[1]);
      num_format_original = word1->bits.num_format;
      word1->bits.num_format = TYPE_UNORM;
    } break;
    default:
      break;
  }

  hsa_status_t status = ImageRuntime::instance()->blit_kernel().FillImage(
      blit_queue_, blit_code_catalog_, *image_view, new_pattern, region);

  // Revert back original configuration.
  if (word3_buff != NULL) {
    word3_buff->bits.dst_sel_w = dst_sel_w_original;
  }

  if (word3_image != NULL) {
    word3_image->bits.dst_sel_w = dst_sel_w_original;
  }

  if (word1 != NULL) {
    word1->bits.num_format = num_format_original;
  }

  return status;
}

hsa_status_t ImageManagerKv::GetLocalMemoryRegion(hsa_region_t region,
                                                  void* data) {
  if (data == NULL) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  hsa_region_segment_t segment;
  hsa_status_t stat = HSA::hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment);
  if (stat != HSA_STATUS_SUCCESS) {
    return stat;
  }

  if (segment != HSA_REGION_SEGMENT_GLOBAL) {
    return HSA_STATUS_SUCCESS;
  }

  uint32_t base = 0;
  stat = HSA::hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &base);
  if (stat != HSA_STATUS_SUCCESS) {
    return stat;
  }

  if ((base & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) != 0) {
    hsa_region_t* local_memory_region = (hsa_region_t*)data;
    *local_memory_region = region;
  }

  return HSA_STATUS_SUCCESS;
}

AddrFormat ImageManagerKv::GetAddrlibFormat(const ImageProperty& image_prop) {
  switch (image_prop.data_format) {
    case FMT_8:
      return ADDR_FMT_8;
      break;
    case FMT_16:
      return (image_prop.data_type != TYPE_FLOAT) ? ADDR_FMT_16
                                                  : ADDR_FMT_16_FLOAT;
      break;
    case FMT_8_8:
      return ADDR_FMT_8_8;
      break;
    case FMT_32:
      return (image_prop.data_type != TYPE_FLOAT) ? ADDR_FMT_32
                                                  : ADDR_FMT_32_FLOAT;
      break;
    case FMT_16_16:
      return (image_prop.data_type != TYPE_FLOAT) ? ADDR_FMT_16_16
                                                  : ADDR_FMT_16_16_FLOAT;
      break;
    case FMT_2_10_10_10:
      return ADDR_FMT_2_10_10_10;
      break;
    case FMT_8_8_8_8:
      return ADDR_FMT_8_8_8_8;
      break;
    case FMT_32_32:
      return (image_prop.data_type != TYPE_FLOAT) ? ADDR_FMT_32_32
                                                  : ADDR_FMT_32_32_FLOAT;
      break;
    case FMT_16_16_16_16:
      return (image_prop.data_type != TYPE_FLOAT) ? ADDR_FMT_16_16_16_16
                                                  : ADDR_FMT_16_16_16_16_FLOAT;
      break;
    case FMT_32_32_32_32:
      return (image_prop.data_type != TYPE_FLOAT) ? ADDR_FMT_32_32_32_32
                                                  : ADDR_FMT_32_32_32_32_FLOAT;
      break;
    case FMT_5_6_5:
      return ADDR_FMT_5_6_5;
      break;
    case FMT_1_5_5_5:
      return ADDR_FMT_1_5_5_5;
      break;
    case FMT_8_24:
      return ADDR_FMT_8_24;
      break;
    default:
      assert(false && "Should not reach here");
      return ADDR_FMT_INVALID;
      break;
  }

  assert(false && "Should not reach here");
  return ADDR_FMT_INVALID;
}

VOID* ADDR_API
    ImageManagerKv::AllocSysMem(const ADDR_ALLOCSYSMEM_INPUT* input) {
  return malloc(input->sizeInBytes);
}

ADDR_E_RETURNCODE ADDR_API
    ImageManagerKv::FreeSysMem(const ADDR_FREESYSMEM_INPUT* input) {
  free(input->pVirtAddr);

  return ADDR_OK;
}

bool ImageManagerKv::GetAddrlibSurfaceInfo(
    hsa_agent_t component, const hsa_ext_image_descriptor_t& desc,
    Image::TileMode tileMode,
    size_t image_data_row_pitch,
    size_t image_data_slice_pitch,
    ADDR_COMPUTE_SURFACE_INFO_OUTPUT& out) const {
  const ImageProperty image_prop =
      GetImageProperty(component, desc.format, desc.geometry);

  const AddrFormat addrlib_format = GetAddrlibFormat(image_prop);

  const uint32_t width = static_cast<uint32_t>(desc.width);
  const uint32_t height = static_cast<uint32_t>(desc.height);
  static const size_t kMinNumSlice = 1;
  const uint32_t num_slice = static_cast<uint32_t>(
      std::max(kMinNumSlice, std::max(desc.array_size, desc.depth)));

  uint32_t major_ver = MajorVerFromDevID(chip_id_);

  if (major_ver >= 9) {
    ADDR2_COMPUTE_SURFACE_INFO_INPUT in = {0};
    in.size = sizeof(ADDR2_COMPUTE_SURFACE_INFO_INPUT);
    in.format = addrlib_format;
    in.bpp = static_cast<unsigned int>(image_prop.element_size) * 8;
    in.width = width;
    in.height = height;
    in.numSlices = num_slice;
    in.pitchInElement = image_data_row_pitch / image_prop.element_size;
    switch(desc.geometry) {
    case HSA_EXT_IMAGE_GEOMETRY_1D:
    case HSA_EXT_IMAGE_GEOMETRY_1DB:
      in.resourceType = ADDR_RSRC_TEX_1D;
      break;
    case HSA_EXT_IMAGE_GEOMETRY_2D:
    case HSA_EXT_IMAGE_GEOMETRY_2DDEPTH:
    case HSA_EXT_IMAGE_GEOMETRY_1DA:
      in.resourceType = ADDR_RSRC_TEX_2D;
      break;
    case HSA_EXT_IMAGE_GEOMETRY_3D:
    case HSA_EXT_IMAGE_GEOMETRY_2DA:
    case HSA_EXT_IMAGE_GEOMETRY_2DADEPTH:
      {
	      in.resourceType = ADDR_RSRC_TEX_3D;
	      /*
	       * 3D swizzle modes enforce alignment
	       * of the number of slices  to the block depth.
	       * If numSlices = 3 then the 3 slices are
	       * interleaved for 3D locality among the 8 slices
	       * that make up each block. This causes the memory
	       * footprint to jump to a 3x size of the ideal size
	       * 'enable3DSwizzleMode' flag tests for env variable
	       * HSA_IMAGE_ENABLE_3D_SWIZZLE_DEBUG to enable or disable
	       * 3D swizzle:
	       * true: Keep view3dAs2dArray = 0 for real 3D interleaving.
	       * false: Use view3dAs2dArray = 1 to avoid the alignment
	       *       expansion.
	       * 2D swizzle modes can lower size overhead but may yield
	       * suboptimal cache behavior for fully 3D volumetric
	       * operations.
	       */
	      bool enable3DSwizzleMode = core::Runtime::runtime_singleton_->flag().enable_3d_swizzle();
	      if (enable3DSwizzleMode)
		      in.flags.view3dAs2dArray = 0;
	      else
		      in.flags.view3dAs2dArray = 1;

	      break;
      }
    }
    in.flags.texture = 1;

    ADDR2_GET_PREFERRED_SURF_SETTING_INPUT  prefSettingsInput = { 0 };
    ADDR2_GET_PREFERRED_SURF_SETTING_OUTPUT prefSettingsOutput = { 0 };

    prefSettingsInput.size            = sizeof(prefSettingsInput);
    prefSettingsInput.flags           = in.flags;
    prefSettingsInput.bpp             = in.bpp;
    prefSettingsInput.format          = in.format;
    prefSettingsInput.width           = in.width;
    prefSettingsInput.height          = in.height;
    prefSettingsInput.numFrags        = in.numFrags;
    prefSettingsInput.numSamples      = in.numSamples;
    prefSettingsInput.numMipLevels    = in.numMipLevels;
    prefSettingsInput.numSlices       = in.numSlices;
    prefSettingsInput.resourceLoction = ADDR_RSRC_LOC_UNDEF;
    prefSettingsInput.resourceType    = in.resourceType;

    // Disallow all swizzles but linear.
    if (tileMode == Image::TileMode::LINEAR)
    {
      prefSettingsInput.forbiddenBlock.macroThin4KB = 1;
      prefSettingsInput.forbiddenBlock.macroThick4KB = 1;
      prefSettingsInput.forbiddenBlock.macroThin64KB = 1;
      prefSettingsInput.forbiddenBlock.macroThick64KB = 1;
    }

    prefSettingsInput.forbiddenBlock.micro = 1; // but don't ever allow the 256b swizzle modes
    prefSettingsInput.forbiddenBlock.var = 1; // and don't allow variable-size block modes

    if (ADDR_OK != Addr2GetPreferredSurfaceSetting(addr_lib_, &prefSettingsInput, &prefSettingsOutput)) {
      return false;
    }

    in.swizzleMode = prefSettingsOutput.swizzleMode;

    ADDR2_COMPUTE_SURFACE_INFO_OUTPUT out2 = {0};
    out.size = sizeof(ADDR2_COMPUTE_SURFACE_INFO_OUTPUT);
    if (ADDR_OK != Addr2ComputeSurfaceInfo(addr_lib_, &in, &out2)) {
      return false;
    }
    out.pitch = out2.pitch;
    out.height = out2.height;
    out.surfSize = out2.surfSize;
    out.bpp = out2.bpp;
    out.baseAlign = out2.baseAlign;
    out.tileIndex = in.swizzleMode;
    out.sliceSize = out2.sliceSize;
    return true;
  }

  ADDR_COMPUTE_SURFACE_INFO_INPUT in = {0};
  in.size = sizeof(ADDR_COMPUTE_SURFACE_INFO_INPUT);
  in.tileMode = (tileMode == Image::TileMode::LINEAR)?
    ADDR_TM_LINEAR_ALIGNED : ADDR_TM_2D_TILED_THIN1;
  in.format = addrlib_format;
  in.bpp = static_cast<unsigned int>(image_prop.element_size) * 8;
  in.numSamples = 1;
  in.width = width;
  in.height = height;
  in.numSlices = num_slice;
  in.flags.texture = 1;
  in.flags.noStencil = 1;
  in.flags.opt4Space = 0;
  in.tileType = ADDR_NON_DISPLAYABLE;
  in.tileIndex = -1;

  if (image_data_row_pitch != 0) {
    in.width = image_data_row_pitch / image_prop.element_size;
//    in.pitchAlign  = image_data_row_pitch / image_prop.element_size;
//    in.heightAlign = image_data_slice_pitch / image_data_row_pitch;
  }

  if (ADDR_OK != AddrComputeSurfaceInfo(addr_lib_, &in, &out)) {
    return false;
  }

  assert(out.tileIndex != -1);

  return (out.tileIndex != -1) ? true : false;
}

size_t ImageManagerKv::CalWorkingSizeBytes(hsa_ext_image_geometry_t geometry,
                                           hsa_dim3_t size_pixel,
                                           uint32_t element_size) const {
  switch (geometry) {
    case HSA_EXT_IMAGE_GEOMETRY_1D:
    case HSA_EXT_IMAGE_GEOMETRY_1DB:
      return size_pixel.x * element_size;
    case HSA_EXT_IMAGE_GEOMETRY_2D:
    case HSA_EXT_IMAGE_GEOMETRY_2DDEPTH:
    case HSA_EXT_IMAGE_GEOMETRY_1DA:
      return size_pixel.x * size_pixel.y * element_size;
    default:
      return size_pixel.x * size_pixel.y * size_pixel.z * element_size;
  }
}

BlitQueue& ImageManagerKv::BlitQueueInit() {
  if (blit_queue_.queue_ == NULL) {
    // Queue is a precious resource, so only create it when it is needed.
    std::lock_guard<std::mutex> lock(lock_);
    if (blit_queue_.queue_ == NULL) {
      // Create the kernel queue.
      blit_queue_.cached_index_ = 0;

      uint32_t max_queue_size = 0;
      hsa_status_t status =
          HSA::hsa_agent_get_info(agent_, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &max_queue_size);

      status = HSA::hsa_queue_create(agent_, max_queue_size, HSA_QUEUE_TYPE_MULTI, NULL, NULL,
                                     UINT_MAX, UINT_MAX, &blit_queue_.queue_);

      if (HSA_STATUS_SUCCESS != status) {
        blit_queue_.queue_ = NULL;
        return blit_queue_;
      }

      // Get the kernel handles.
      status = ImageRuntime::instance()->blit_kernel().BuildBlitCode(agent_, blit_code_catalog_);

      if (HSA_STATUS_SUCCESS != status) {
        blit_code_catalog_.clear();
        HSA::hsa_queue_destroy(blit_queue_.queue_);
        blit_queue_.queue_ = NULL;
        return blit_queue_;
      }
    }
  }

  assert(blit_queue_.queue_ != NULL &&
         blit_code_catalog_.size() == BlitKernel::KERNEL_OP_COUNT);

  return blit_queue_;
}

}  // namespace image
}  // namespace rocr
