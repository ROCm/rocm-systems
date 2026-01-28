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

#include "image_manager_ai.h"

#include <assert.h>

#include <algorithm>
#include <climits>

#include "core/inc/runtime.h"
#include "hsakmt/hsakmt.h"
#include "inc/hsa_ext_amd.h"
#include "core/inc/hsa_internal.h"
#include "addrlib/src/core/addrlib.h"
#include "image_runtime.h"
#include "resource.h"
#include "resource_ai.h"
#include "util.h"
#include "device_info.h"

namespace rocr {
namespace image {

ImageManagerAi::ImageManagerAi() : ImageManagerKv() {}

ImageManagerAi::~ImageManagerAi() {}

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

hsa_status_t ImageManagerAi::CalculateImageSizeAndAlignment(
    hsa_agent_t component, const hsa_ext_image_descriptor_t& desc,
    hsa_ext_image_data_layout_t image_data_layout,
    uint32_t num_mipmap_levels,
    size_t image_data_row_pitch,
    size_t image_data_slice_pitch,
    hsa_ext_image_data_info_t& image_info) const {
  ADDR2_COMPUTE_SURFACE_INFO_OUTPUT out = {0};
  hsa_profile_t profile;

  hsa_status_t status = HSA::hsa_agent_get_info(component, HSA_AGENT_INFO_PROFILE, &profile);
  if (status != HSA_STATUS_SUCCESS) return status;

  Image::TileMode tileMode = Image::TileMode::LINEAR;
  if (image_data_layout == HSA_EXT_IMAGE_DATA_LAYOUT_OPAQUE) {
    tileMode = (profile == HSA_PROFILE_BASE &&
                desc.geometry != HSA_EXT_IMAGE_GEOMETRY_1DB)?
      Image::TileMode::TILED : Image::TileMode::LINEAR;
  }
  if (GetAddrlibSurfaceInfoAi(component, desc, num_mipmap_levels, tileMode,
      image_data_row_pitch, image_data_slice_pitch, out) == (uint32_t)(-1)) {
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

bool ImageManagerAi::IsLocalMemory(const void* address) const {
  return true;
}

hsa_status_t ImageManagerAi::PopulateImageSrd(Image& image, const metadata_amd_t* descriptor) const {
  metadata_amd_ai_t* desc = (metadata_amd_ai_t*)descriptor;
  const void* image_data_addr = image.data;

  ImageProperty image_prop = ImageLut().MapFormat(image.desc.format, image.desc.geometry);
  if((image_prop.cap == HSA_EXT_IMAGE_CAPABILITY_NOT_SUPPORTED) ||
     (image_prop.element_size == 0))
    return (hsa_status_t)HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED;

  const Swizzle swizzle = ImageLut().MapSwizzle(image.desc.format.channel_order);

  if (IsLocalMemory(image.data)) {
    image_data_addr = reinterpret_cast<const void*>(
        reinterpret_cast<uintptr_t>(image.data) - local_memory_base_address_);
  }

  image.srd[0]=desc->word0.u32All;
  image.srd[1]=desc->word1.u32All;
  image.srd[2]=desc->word2.u32All;
  image.srd[3]=desc->word3.u32All;
  image.srd[4]=desc->word4.u32All;
  image.srd[5]=desc->word5.u32All;
  image.srd[6]=desc->word6.u32All;
  image.srd[7]=desc->word7.u32All;

  if (image.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_1DB) {
    sq_buf_rsrc_word0_u word0;
    sq_buf_rsrc_word1_u word1;
    sq_buf_rsrc_word3_u word3;

    word0.val = 0;
    word0.f.base_address = PtrLow32(image_data_addr);

    word1.val = image.srd[1];
    word1.f.base_address_hi = PtrHigh32(image_data_addr);
    word1.f.stride = image_prop.element_size;

    word3.val = image.srd[3];
    word3.f.dst_sel_x = swizzle.x;
    word3.f.dst_sel_y = swizzle.y;
    word3.f.dst_sel_z = swizzle.z;
    word3.f.dst_sel_w = swizzle.w;
    word3.f.num_format = image_prop.data_type;
    word3.f.data_format = image_prop.data_format;
    word3.f.index_stride = image_prop.element_size;

    image.srd[0] = word0.val;
    image.srd[1] = word1.val;
    image.srd[3] = word3.val;
  } else {
    uint32_t hwPixelSize = ImageLut().GetPixelSize(desc->word1.bitfields.DATA_FORMAT,
                                                   desc->word1.bitfields.NUM_FORMAT);
    if(image_prop.element_size!=hwPixelSize)
      return (hsa_status_t)HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED;

    ((SQ_IMG_RSRC_WORD0*)(&image.srd[0]))->bits.BASE_ADDRESS = PtrLow40Shift8(image_data_addr);
    ((SQ_IMG_RSRC_WORD1*)(&image.srd[1]))->bits.BASE_ADDRESS_HI = PtrHigh64Shift40(image_data_addr);
    ((SQ_IMG_RSRC_WORD1*)(&image.srd[1]))->bits.DATA_FORMAT = image_prop.data_format;
    ((SQ_IMG_RSRC_WORD1*)(&image.srd[1]))->bits.NUM_FORMAT = image_prop.data_type;
    ((SQ_IMG_RSRC_WORD3*)(&image.srd[3]))->bits.DST_SEL_X = swizzle.x;
    ((SQ_IMG_RSRC_WORD3*)(&image.srd[3]))->bits.DST_SEL_Y = swizzle.y;
    ((SQ_IMG_RSRC_WORD3*)(&image.srd[3]))->bits.DST_SEL_Z = swizzle.z;
    ((SQ_IMG_RSRC_WORD3*)(&image.srd[3]))->bits.DST_SEL_W = swizzle.w;
    if (image.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_1DA ||
        image.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_1D) {
      ((SQ_IMG_RSRC_WORD3*)(&image.srd[3]))->bits.TYPE =
          ImageLut().MapGeometry(image.desc.geometry);
    }
    
    // Imported metadata holds the offset to metadata, add the image base address.
    uintptr_t meta = uintptr_t(((SQ_IMG_RSRC_WORD5*)(&image.srd[5]))->bits.META_DATA_ADDRESS_HI) << 40;
    meta |= uintptr_t(((SQ_IMG_RSRC_WORD7*)(&image.srd[7]))->bits.META_DATA_ADDRESS) << 8;
    meta += reinterpret_cast<uintptr_t>(image_data_addr);

    ((SQ_IMG_RSRC_WORD7*)(&image.srd[7]))->bits.META_DATA_ADDRESS = PtrLow40Shift8((void*)meta);
    ((SQ_IMG_RSRC_WORD5*)(&image.srd[5]))->bits.META_DATA_ADDRESS_HI =
        PtrHigh64Shift40((void*)meta);
  }
  //Looks like this is only used for CPU copies.
  image.row_pitch = 0;//desc->word4.bits.pitch+1*desc->word3.bits.element_size;
  image.slice_pitch = 0;//desc->;

  //Used by HSAIL shader ABI
  image.srd[8] = image.desc.format.channel_type;
  image.srd[9] = image.desc.format.channel_order;
  image.srd[10] = static_cast<uint32_t>(image.desc.width);

  return HSA_STATUS_SUCCESS;
}

static TEX_BC_SWIZZLE GetBcSwizzle(const Swizzle& swizzle) {
    SEL r = (SEL)swizzle.x;
    SEL g = (SEL)swizzle.y;
    SEL b = (SEL)swizzle.z;
    SEL a = (SEL)swizzle.w;

    TEX_BC_SWIZZLE bcSwizzle = TEX_BC_Swizzle_XYZW;

    if (a == SEL_X)
    {
        // Have to use either TEX_BC_Swizzle_WZYX or TEX_BC_Swizzle_WXYZ
        //
        // For the pre-defined border color values (white, opaque black, transparent black), the only thing that
        // matters is that the alpha channel winds up in the correct place (because the RGB channels are all the same)
        // so either of these TEX_BC_Swizzle enumerations will work.  Not sure what happens with border color palettes.
        if (b == SEL_Y)
        {
            // ABGR
            bcSwizzle = TEX_BC_Swizzle_WZYX;
        }
        else if ((r == SEL_X) && (g == SEL_X) && (b == SEL_X))
        {
            //RGBA
            bcSwizzle = TEX_BC_Swizzle_XYZW;
        }
        else
        {
            // ARGB
            bcSwizzle = TEX_BC_Swizzle_WXYZ;
        }
    }
    else if (r == SEL_X)
    {
        // Have to use either TEX_BC_Swizzle_XYZW or TEX_BC_Swizzle_XWYZ
        if (g == SEL_Y)
        {
            // RGBA
            bcSwizzle = TEX_BC_Swizzle_XYZW;
        }
        else if((g == SEL_X) && (b == SEL_X) && (a == SEL_W))
        {
            // RGBA
            bcSwizzle = TEX_BC_Swizzle_XYZW;
        }
        else
        {
            // RAGB
            bcSwizzle = TEX_BC_Swizzle_XWYZ;
        }
    }
    else if (g == SEL_X)
    {
        // GRAB, have to use TEX_BC_Swizzle_YXWZ
        bcSwizzle = TEX_BC_Swizzle_YXWZ;
    }
    else if (b == SEL_X)
    {
        // BGRA, have to use TEX_BC_Swizzle_ZYXW
        bcSwizzle = TEX_BC_Swizzle_ZYXW;
    }

    return bcSwizzle;
}


hsa_status_t ImageManagerAi::PopulateImageSrd(Image& image) const {
  ImageProperty image_prop = ImageLut().MapFormat(image.desc.format, image.desc.geometry);
  assert(image_prop.cap != HSA_EXT_IMAGE_CAPABILITY_NOT_SUPPORTED);
  assert(image_prop.element_size != 0);

  const void* image_data_addr = image.data;

  if (IsLocalMemory(image.data))
    image_data_addr = reinterpret_cast<const void*>(
        reinterpret_cast<uintptr_t>(image.data) - local_memory_base_address_);

  if (image.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_1DB) {
    sq_buf_rsrc_word0_u word0;
    sq_buf_rsrc_word1_u word1;
    sq_buf_rsrc_word2_u word2;
    sq_buf_rsrc_word3_u word3;

    word0.val = 0;
    word0.f.base_address = PtrLow32(image_data_addr);

    word1.val = 0;
    word1.f.base_address_hi = PtrHigh32(image_data_addr);
    word1.f.stride = image_prop.element_size;
    word1.f.swizzle_enable = false;
    word1.f.cache_swizzle = false;

    word2.f.num_records = image.desc.width * image_prop.element_size;

    const Swizzle swizzle = ImageLut().MapSwizzle(image.desc.format.channel_order);
    word3.val = 0;
    word3.f.dst_sel_x = swizzle.x;
    word3.f.dst_sel_y = swizzle.y;
    word3.f.dst_sel_z = swizzle.z;
    word3.f.dst_sel_w = swizzle.w;
    word3.f.num_format = image_prop.data_type;
    word3.f.data_format = image_prop.data_format;
    word3.f.index_stride = image_prop.element_size;
    word3.f.type = ImageLut().MapGeometry(image.desc.geometry);

    image.srd[0] = word0.val;
    image.srd[1] = word1.val;
    image.srd[2] = word2.val;
    image.srd[3] = word3.val;

    image.row_pitch = image.desc.width * image_prop.element_size;
    image.slice_pitch = image.row_pitch;
  } else {
    sq_img_rsrc_word0_u word0;
    sq_img_rsrc_word1_u word1;
    sq_img_rsrc_word2_u word2;
    sq_img_rsrc_word3_u word3;
    sq_img_rsrc_word4_u word4;
    sq_img_rsrc_word5_u word5;
    sq_img_rsrc_word6_u word6;
    sq_img_rsrc_word7_u word7;

    ADDR2_COMPUTE_SURFACE_INFO_OUTPUT out = {0};

    uint32_t swizzleMode = GetAddrlibSurfaceInfoAi(image.component, image.desc,
                  1, image.tile_mode, image.row_pitch, image.slice_pitch, out);
    if (swizzleMode == (uint32_t)(-1)) {
      return HSA_STATUS_ERROR;
    }

    assert((out.bpp / 8) == image_prop.element_size);

    const size_t row_pitch_size = out.pitch * image_prop.element_size;

    word0.f.base_address = PtrLow40Shift8(image_data_addr);

    word1.val = 0;
    word1.f.base_address_hi = PtrHigh64Shift40(image_data_addr);
    word1.f.min_lod = 0;
    word1.f.data_format = image_prop.data_format;
    word1.f.num_format = image_prop.data_type;

    word2.val = 0;
    word2.f.width = image.desc.width - 1;
    word2.f.height = image.desc.height - 1;
    word2.f.perf_mod = 0;

    const Swizzle swizzle = ImageLut().MapSwizzle(image.desc.format.channel_order);
    word3.val = 0;
    word3.f.dst_sel_x = swizzle.x;
    word3.f.dst_sel_y = swizzle.y;
    word3.f.dst_sel_z = swizzle.z;
    word3.f.dst_sel_w = swizzle.w;
    word3.f.sw_mode = swizzleMode;
    word3.f.type = ImageLut().MapGeometry(image.desc.geometry);

    const bool image_array =
        (image.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_1DA ||
         image.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_2DA ||
         image.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_2DADEPTH);
    const bool image_3d = (image.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_3D);

    word4.val = 0;
    word4.f.depth =
        (image_array)
            ? std::max(image.desc.array_size, static_cast<size_t>(1)) - 1
            : (image_3d) ? image.desc.depth - 1 : 0;
    word4.f.pitch = out.pitch - 1;
    word4.f.bc_swizzle = GetBcSwizzle(swizzle);

    word5.val = 0;
    word6.val = 0;
    word7.val = 0;

    image.srd[0] = word0.val;
    image.srd[1] = word1.val;
    image.srd[2] = word2.val;
    image.srd[3] = word3.val;
    image.srd[4] = word4.val;
    image.srd[5] = word5.val;
    image.srd[6] = word6.val;
    image.srd[7] = word7.val;

    image.row_pitch = row_pitch_size;
    image.slice_pitch = out.sliceSize;
  }

  image.srd[8] = image.desc.format.channel_type;
  image.srd[9] = image.desc.format.channel_order;
  image.srd[10] = static_cast<uint32_t>(image.desc.width);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t ImageManagerAi::ModifyImageSrd(
    Image& image, hsa_ext_image_format_t& new_format) const {
  image.desc.format = new_format;

  ImageProperty image_prop = ImageLut().MapFormat(image.desc.format, image.desc.geometry);
  assert(image_prop.cap != HSA_EXT_IMAGE_CAPABILITY_NOT_SUPPORTED);
  assert(image_prop.element_size != 0);

  if (image.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_1DB) {
    const Swizzle swizzle = ImageLut().MapSwizzle(image.desc.format.channel_order);
    SQ_BUF_RSRC_WORD3* word3 =
        reinterpret_cast<SQ_BUF_RSRC_WORD3*>(&image.srd[3]);
    word3->bits.DST_SEL_X = swizzle.x;
    word3->bits.DST_SEL_Y = swizzle.y;
    word3->bits.DST_SEL_Z = swizzle.z;
    word3->bits.DST_SEL_W = swizzle.w;
    word3->bits.NUM_FORMAT = image_prop.data_type;
    word3->bits.DATA_FORMAT = image_prop.data_format;
  } else {
    SQ_IMG_RSRC_WORD1* word1 =
        reinterpret_cast<SQ_IMG_RSRC_WORD1*>(&image.srd[1]);
    word1->bits.DATA_FORMAT = image_prop.data_format;
    word1->bits.NUM_FORMAT = image_prop.data_type;

    const Swizzle swizzle = ImageLut().MapSwizzle(image.desc.format.channel_order);
    SQ_IMG_RSRC_WORD3* word3 =
        reinterpret_cast<SQ_IMG_RSRC_WORD3*>(&image.srd[3]);
    word3->bits.DST_SEL_X = swizzle.x;
    word3->bits.DST_SEL_Y = swizzle.y;
    word3->bits.DST_SEL_Z = swizzle.z;
    word3->bits.DST_SEL_W = swizzle.w;
  }

  image.srd[8] = image.desc.format.channel_type;
  image.srd[9] = image.desc.format.channel_order;
  image.srd[10] = static_cast<uint32_t>(image.desc.width);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t ImageManagerAi::PopulateSamplerSrd(Sampler& sampler) const {
  const hsa_ext_sampler_descriptor_v2_t &sampler_descriptor = sampler.desc;

  SQ_IMG_SAMP_WORD0 word0;
  SQ_IMG_SAMP_WORD1 word1;
  SQ_IMG_SAMP_WORD2 word2;
  SQ_IMG_SAMP_WORD3 word3;

  word0.u32All = 0;
  hsa_status_t status = convertAddressMode<SQ_IMG_SAMP_WORD0, SQ_TEX_CLAMP>
                                       (word0, sampler_descriptor.address_modes);
  if (status != HSA_STATUS_SUCCESS) return status;
  word0.bits.FORCE_UNNORMALIZED = (sampler_descriptor.coordinate_mode ==
                                  HSA_EXT_SAMPLER_COORDINATE_MODE_UNNORMALIZED);

  word1.u32All = 0;
  word1.bits.MAX_LOD = 4095;

  word2.u32All = 0;
  switch (sampler_descriptor.filter_mode) {
    case HSA_EXT_SAMPLER_FILTER_MODE_NEAREST:
      word2.bits.XY_MAG_FILTER = static_cast<int>(SQ_TEX_XY_FILTER_POINT);
      break;
    case HSA_EXT_SAMPLER_FILTER_MODE_LINEAR:
      word2.bits.XY_MAG_FILTER = static_cast<int>(SQ_TEX_XY_FILTER_BILINEAR);
      break;
    default:
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  word2.bits.XY_MIN_FILTER = word2.bits.XY_MAG_FILTER;
  word2.bits.Z_FILTER = SQ_TEX_Z_FILTER_NONE;

  switch (sampler_descriptor.mipmap_filter_mode) {
    case HSA_EXT_SAMPLER_FILTER_MODE_NEAREST:
      word2.bits.MIP_FILTER = static_cast<int>(SQ_TEX_MIP_FILTER_POINT);
      break;
    case HSA_EXT_SAMPLER_FILTER_MODE_LINEAR:
      word2.bits.MIP_FILTER = static_cast<int>(SQ_TEX_MIP_FILTER_LINEAR);
      break;
    default:
      word2.bits.MIP_FILTER = static_cast<int>(SQ_TEX_MIP_FILTER_NONE);
  }

  word3.u32All = 0;

  // TODO: check this bit with HSAIL spec.
  word3.bits.BORDER_COLOR_TYPE = SQ_TEX_BORDER_COLOR_TRANS_BLACK;

  sampler.srd[0] = word0.u32All;
  sampler.srd[1] = word1.u32All;
  sampler.srd[2] = word2.u32All;
  sampler.srd[3] = word3.u32All;

  return HSA_STATUS_SUCCESS;
}

uint32_t ImageManagerAi::GetAddrlibSurfaceInfoAi(
    hsa_agent_t component, const hsa_ext_image_descriptor_t& desc,
    uint32_t num_mipmap_levels,
    Image::TileMode tileMode,
    size_t image_data_row_pitch,
    size_t image_data_slice_pitch,
    ADDR2_COMPUTE_SURFACE_INFO_OUTPUT& out) const {
  const ImageProperty image_prop =
      GetImageProperty(component, desc.format, desc.geometry);

  const AddrFormat addrlib_format = GetAddrlibFormat(image_prop);

  const uint32_t width = static_cast<uint32_t>(desc.width);
  const uint32_t height = static_cast<uint32_t>(desc.height);
  static const size_t kMinNumSlice = 1;
  const uint32_t num_slice = static_cast<uint32_t>(
      std::max(kMinNumSlice, std::max(desc.array_size, desc.depth)));

  ADDR2_COMPUTE_SURFACE_INFO_INPUT in = {0};
  in.size = sizeof(ADDR2_COMPUTE_SURFACE_INFO_INPUT);
  in.format = addrlib_format;
  in.bpp = static_cast<unsigned int>(image_prop.element_size) * 8;
  in.width = width;
  in.height = height;
  in.numSlices = num_slice;
  in.numMipLevels = num_mipmap_levels;

  switch(desc.geometry) {
  case HSA_EXT_IMAGE_GEOMETRY_1D:
  case HSA_EXT_IMAGE_GEOMETRY_1DB:
  case HSA_EXT_IMAGE_GEOMETRY_1DA:
    in.resourceType = ADDR_RSRC_TEX_1D;
    break;
  case HSA_EXT_IMAGE_GEOMETRY_2D:
  case HSA_EXT_IMAGE_GEOMETRY_2DDEPTH:
  case HSA_EXT_IMAGE_GEOMETRY_2DA:
  case HSA_EXT_IMAGE_GEOMETRY_2DADEPTH:
    in.resourceType = ADDR_RSRC_TEX_2D;
    break;
  case HSA_EXT_IMAGE_GEOMETRY_3D:
    {
	    in.resourceType = ADDR_RSRC_TEX_3D;
	    /*
	     * 3D swizzle modes enforce alignment
	     * of the number of slices  to the block depth.
	     * If numSlices = 3 then the 3 slices are
	     * interleaved for 3D locality among the 8 slices
	     * that make up each block. This causes the memory
	     * footprint to jump to a 3x size of the ideal size
	     *
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
    return (uint32_t)(-1);
  }

  in.swizzleMode = prefSettingsOutput.swizzleMode;

  out.size = sizeof(ADDR2_COMPUTE_SURFACE_INFO_OUTPUT);
  if (ADDR_OK != Addr2ComputeSurfaceInfo(addr_lib_, &in, &out)) {
    return (uint32_t)(-1);
  }
  if (out.surfSize == 0) {
    return (uint32_t)(-1);
  }

  return in.swizzleMode;
}

hsa_status_t ImageManagerAi::PopulateMipmapSrd(MipmappedArray& mipmap) const {
  ImageProperty mipmap_prop = ImageLut().MapFormat(mipmap.desc.format, mipmap.desc.geometry);
  assert(mipmap_prop.cap != HSA_EXT_IMAGE_CAPABILITY_NOT_SUPPORTED);
  assert(mipmap_prop.element_size != 0);
  assert(mipmap.num_levels >= 1);

  const void* mipmap_data_addr = mipmap.data;

  if (IsLocalMemory(mipmap.data))
    mipmap_data_addr = reinterpret_cast<const void*>(
        reinterpret_cast<uintptr_t>(mipmap.data) - local_memory_base_address_);

  if (mipmap.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_1DB) {
    sq_buf_rsrc_word0_u word0;
    sq_buf_rsrc_word1_u word1;
    sq_buf_rsrc_word2_u word2;
    sq_buf_rsrc_word3_u word3;

    word0.val = 0;
    word0.f.base_address = PtrLow32(mipmap_data_addr);

    word1.val = 0;
    word1.f.base_address_hi = PtrHigh32(mipmap_data_addr);
    word1.f.stride = mipmap_prop.element_size;
    word1.f.swizzle_enable = false;
    word1.f.cache_swizzle = false;

    word2.val = 0;
    word2.f.num_records = mipmap.desc.width * mipmap_prop.element_size;

    const Swizzle swizzle = ImageLut().MapSwizzle(mipmap.desc.format.channel_order);
    word3.val = 0;
    word3.f.dst_sel_x = swizzle.x;
    word3.f.dst_sel_y = swizzle.y;
    word3.f.dst_sel_z = swizzle.z;
    word3.f.dst_sel_w = swizzle.w;
    word3.f.num_format = mipmap_prop.data_type;
    word3.f.data_format = mipmap_prop.data_format;
    word3.f.index_stride = mipmap_prop.element_size;
    word3.f.type = ImageLut().MapGeometry(mipmap.desc.geometry);

    mipmap.srd[0] = word0.val;
    mipmap.srd[1] = word1.val;
    mipmap.srd[2] = word2.val;
    mipmap.srd[3] = word3.val;

    mipmap.row_pitch = mipmap.desc.width * mipmap_prop.element_size;
    mipmap.slice_pitch = mipmap.row_pitch;
  } else {
    sq_img_rsrc_word0_u word0;
    sq_img_rsrc_word1_u word1;
    sq_img_rsrc_word2_u word2;
    sq_img_rsrc_word3_u word3;
    sq_img_rsrc_word4_u word4;
    sq_img_rsrc_word5_u word5;
    sq_img_rsrc_word6_u word6;
    sq_img_rsrc_word7_u word7;

    ADDR2_COMPUTE_SURFACE_INFO_OUTPUT out = {0};

    // pMipInfo not needed - set to nullptr and AddrLib will ignore it
    out.pMipInfo = nullptr;

    uint32_t swizzleMode = GetAddrlibSurfaceInfoAi(
                        mipmap.component, mipmap.desc, mipmap.num_levels,
                        mipmap.tile_mode, mipmap.row_pitch, mipmap.slice_pitch, out);
    if (swizzleMode == (uint32_t)(-1)) {
      return HSA_STATUS_ERROR;
    }
    mipmap.addr_output.addr2 = out;
    mipmap.size = out.surfSize;

    assert((out.bpp / 8) == mipmap_prop.element_size);

    const size_t row_pitch_size = out.pitch * mipmap_prop.element_size;

    word0.f.base_address = PtrLow40Shift8(mipmap_data_addr);

    word1.val = 0;
    word1.f.base_address_hi = PtrHigh64Shift40(mipmap_data_addr);
    word1.f.min_lod = 0;
    word1.f.data_format = mipmap_prop.data_format;
    word1.f.num_format = mipmap_prop.data_type;

    word2.val = 0;
    word2.f.width = mipmap.desc.width - 1;
    word2.f.height = mipmap.desc.height - 1;
    word2.f.perf_mod = 0;

    const Swizzle swizzle = ImageLut().MapSwizzle(mipmap.desc.format.channel_order);
    word3.val = 0;
    word3.f.dst_sel_x = swizzle.x;
    word3.f.dst_sel_y = swizzle.y;
    word3.f.dst_sel_z = swizzle.z;
    word3.f.dst_sel_w = swizzle.w;
    word3.f.sw_mode = swizzleMode;
    word3.f.base_level = 0;
    word3.f.last_level = mipmap.num_levels - 1;
    word3.f.type = ImageLut().MapGeometry(mipmap.desc.geometry);

    const bool mipmap_array =
        (mipmap.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_1DA ||
         mipmap.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_2DA ||
         mipmap.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_2DADEPTH);
    const bool mipmap_3d = (mipmap.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_3D);

    word4.val = 0;
    word4.f.depth =
        (mipmap_array)
            ? std::max(mipmap.desc.array_size, static_cast<size_t>(1)) - 1
            : (mipmap_3d) ? mipmap.desc.depth - 1 : 0;
    word4.f.pitch = out.pitch - 1;
    word4.f.bc_swizzle = GetBcSwizzle(swizzle);

    word5.val = 0;
    word5.f.max_mip = mipmap.num_levels - 1;
    word6.val = 0;
    word7.val = 0;

    mipmap.srd[0] = word0.val;
    mipmap.srd[1] = word1.val;
    mipmap.srd[2] = word2.val;
    mipmap.srd[3] = word3.val;
    mipmap.srd[4] = word4.val;
    mipmap.srd[5] = word5.val;
    mipmap.srd[6] = word6.val;
    mipmap.srd[7] = word7.val;

    mipmap.row_pitch = row_pitch_size;
    mipmap.slice_pitch = out.sliceSize;
  }

  mipmap.srd[8] = mipmap.desc.format.channel_type;
  mipmap.srd[9] = mipmap.desc.format.channel_order;
  mipmap.srd[10] = static_cast<uint32_t>(mipmap.desc.width);

  // Mipmap-specific
  mipmap.srd[11] = mipmap.num_levels;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t ImageManagerAi::PopulateMipmapSrd(MipmappedArray& mipmap_array, const metadata_amd_t* desc) const {
  const metadata_amd_ai_t* desc_ai = reinterpret_cast<const metadata_amd_ai_t*>(desc);
  const void* mipmap_data_addr = mipmap_array.data;
  
  ImageProperty mipmap_prop = ImageLut().MapFormat(mipmap_array.desc.format, mipmap_array.desc.geometry);
  if (mipmap_prop.cap == HSA_EXT_IMAGE_CAPABILITY_NOT_SUPPORTED || mipmap_prop.element_size == 0) {
    return (hsa_status_t)HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED;
  }
  
  const Swizzle swizzle = ImageLut().MapSwizzle(mipmap_array.desc.format.channel_order);
  
  if (IsLocalMemory(mipmap_array.data)) {
    mipmap_data_addr = reinterpret_cast<const void*>(
        reinterpret_cast<uintptr_t>(mipmap_array.data) - local_memory_base_address_);
  }
  
  // Copy the pre-computed SRD words 0-7 from metadata
  mipmap_array.srd[0] = desc_ai->word0.u32All;
  mipmap_array.srd[1] = desc_ai->word1.u32All;
  mipmap_array.srd[2] = desc_ai->word2.u32All;
  mipmap_array.srd[3] = desc_ai->word3.u32All;
  mipmap_array.srd[4] = desc_ai->word4.u32All;
  mipmap_array.srd[5] = desc_ai->word5.u32All;
  mipmap_array.srd[6] = desc_ai->word6.u32All;
  mipmap_array.srd[7] = desc_ai->word7.u32All;
  
  // Override specific fields after copying
  uint32_t hwPixelSize = ImageLut().GetPixelSize(mipmap_prop.data_format, mipmap_prop.data_type);
  if (mipmap_prop.element_size != hwPixelSize) {
    return (hsa_status_t)HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED;
  }
  
  reinterpret_cast<SQ_IMG_RSRC_WORD0*>(&mipmap_array.srd[0])->bits.BASE_ADDRESS = PtrLow40Shift8(mipmap_data_addr);
  reinterpret_cast<SQ_IMG_RSRC_WORD1*>(&mipmap_array.srd[1])->bits.BASE_ADDRESS_HI = PtrHigh64Shift40(mipmap_data_addr);
  reinterpret_cast<SQ_IMG_RSRC_WORD1*>(&mipmap_array.srd[1])->bits.DATA_FORMAT = mipmap_prop.data_format;
  reinterpret_cast<SQ_IMG_RSRC_WORD1*>(&mipmap_array.srd[1])->bits.NUM_FORMAT = mipmap_prop.data_type;
  reinterpret_cast<SQ_IMG_RSRC_WORD3*>(&mipmap_array.srd[3])->bits.DST_SEL_X = swizzle.x;
  reinterpret_cast<SQ_IMG_RSRC_WORD3*>(&mipmap_array.srd[3])->bits.DST_SEL_Y = swizzle.y;
  reinterpret_cast<SQ_IMG_RSRC_WORD3*>(&mipmap_array.srd[3])->bits.DST_SEL_Z = swizzle.z;
  reinterpret_cast<SQ_IMG_RSRC_WORD3*>(&mipmap_array.srd[3])->bits.DST_SEL_W = swizzle.w;
  reinterpret_cast<SQ_IMG_RSRC_WORD5*>(&mipmap_array.srd[5])->bits.MAX_MIP = mipmap_array.num_levels - 1;
  
  if (mipmap_array.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_1DA ||
      mipmap_array.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_1D) {
    reinterpret_cast<SQ_IMG_RSRC_WORD3*>(&mipmap_array.srd[3])->bits.TYPE =
        ImageLut().MapGeometry(mipmap_array.desc.geometry);
  }
  
  // Looks like this is only used for CPU copies.
  mipmap_array.row_pitch = 0;
  mipmap_array.slice_pitch = 0;
  
  // Store mipmap-specific metadata
  mipmap_array.srd[8] = mipmap_array.desc.format.channel_type;
  mipmap_array.srd[9] = mipmap_array.desc.format.channel_order;
  mipmap_array.srd[10] = static_cast<uint32_t>(mipmap_array.desc.width);
  mipmap_array.srd[11] = mipmap_array.num_levels;
  
  // Allocate and populate pMipInfo from metadata mip_offsets (ADDR2 for Ai/GFX9)
  ADDR2_MIP_INFO* mip_info_storage = new ADDR2_MIP_INFO[mipmap_array.num_levels];
  memset(mip_info_storage, 0, sizeof(ADDR2_MIP_INFO) * mipmap_array.num_levels);
  
  // Extract per-level information from mip_offsets array
  for (uint32_t level = 0; level < mipmap_array.num_levels; level++) {
    // mip_offsets contains offset bits [39:8], shift left by 8 to get actual byte offset
    mip_info_storage[level].offset = static_cast<uint64_t>(desc_ai->mip_offsets[level]) << 8;
    
    // Calculate dimensions for this level (halve at each level)
    mip_info_storage[level].pitch = std::max(1u, static_cast<uint32_t>(mipmap_array.desc.width >> level));
    mip_info_storage[level].height = std::max(1u, static_cast<uint32_t>(mipmap_array.desc.height >> level));
    mip_info_storage[level].depth = std::max(1u, static_cast<uint32_t>(mipmap_array.desc.depth >> level));
  }
  
  // Store pMipInfo in addr_output for later use by PopulateMipLevelSrd
  mipmap_array.addr_output.addr2.pMipInfo = mip_info_storage;
  
  // Total size calculation from metadata
  uint32_t last_level = mipmap_array.num_levels - 1;
  uint64_t last_level_size = mip_info_storage[last_level].pitch * 
                             mip_info_storage[last_level].height * 
                             mip_info_storage[last_level].depth * 
                             mipmap_prop.element_size;
  mipmap_array.size = mip_info_storage[last_level].offset + last_level_size;
  
  return HSA_STATUS_SUCCESS;
}

void ImageManagerAi::printSRDDetailed(const uint32_t* srd) const {
  if (!srd) {
    printf("\n========== Image SRD (GFX9/AI) - Detailed ==========\n");
    printf("ERROR: No SRD data provided.\n");
    printf("===============================================\n\n");
    return;
  }

  printf("\n========== Image SRD (GFX9/AI) - Detailed ==========\n");

  // Print all 12 words with bit field annotations
  for (int i = 0; i < 12; i++) {
    printf("WORD %d: 0x%08x  ", i, srd[i]);

    // Binary representation
    printf("(");
    for (int bit = 31; bit >= 0; bit--) {
      printf("%d", (srd[i] >> bit) & 1);
      if (bit % 4 == 0 && bit != 0) printf("_");
    }
    printf(")\n");
  }
        
  // WORD 0: BASE_ADDRESS (bits 39:8)
  sq_img_rsrc_word0_u word0;
  word0.val = srd[0];
  printf("\nWORD 0: BASE_ADDRESS (bits 39:8) = 0x%08x\n", word0.f.base_address);
  
  // WORD 1: Contains BASE_ADDRESS_HI, MIN_LOD, DATA_FORMAT, NUM_FORMAT
  sq_img_rsrc_word1_u word1;
  word1.val = srd[1];
  printf("WORD 1: BASE_ADDRESS_HI        = 0x%02x\n", word1.f.base_address_hi);
  printf("        MIN_LOD                = %u\n", word1.f.min_lod);
  printf("        DATA_FORMAT            = %u\n", word1.f.data_format);
  printf("        NUM_FORMAT             = %u\n", word1.f.num_format);
  
  // Calculate full address (GFX9 uses 40-bit shifted by 8)
  uint64_t base_addr = ((uint64_t)word1.f.base_address_hi << 32) | ((uint64_t)word0.f.base_address << 8);
  printf("        → Full Base Address    = 0x%016lx\n", base_addr);
  
  // WORD 2: WIDTH, HEIGHT, PERF_MOD
  sq_img_rsrc_word2_u word2;
  word2.val = srd[2];
  printf("WORD 2: WIDTH                  = %u (actual: %u)\n", word2.f.width, word2.f.width + 1);
  printf("        HEIGHT                 = %u (actual: %u)\n", word2.f.height, word2.f.height + 1);
  printf("        PERF_MOD               = %u\n", word2.f.perf_mod);
  
  // WORD 3: Channel selectors, SW_MODE, BASE_LEVEL, LAST_LEVEL, TYPE
  sq_img_rsrc_word3_u word3;
  word3.val = srd[3];
  printf("WORD 3: DST_SEL_X              = %u ", word3.f.dst_sel_x);
  printChannelSelect(word3.f.dst_sel_x);
  printf("        DST_SEL_Y              = %u ", word3.f.dst_sel_y);
  printChannelSelect(word3.f.dst_sel_y);
  printf("        DST_SEL_Z              = %u ", word3.f.dst_sel_z);
  printChannelSelect(word3.f.dst_sel_z);
  printf("        DST_SEL_W              = %u ", word3.f.dst_sel_w);
  printChannelSelect(word3.f.dst_sel_w);
  printf("        BASE_LEVEL             = %u ◄──── Current base level\n", word3.f.base_level);
  printf("        LAST_LEVEL             = %u ◄──── Current last level\n", word3.f.last_level);
  printf("        SW_MODE                = %u ", word3.f.sw_mode);
  printSwizzleMode(word3.f.sw_mode);
  printf("        TYPE                   = %u ", word3.f.type);
  printResourceType(word3.f.type);
  
  // WORD 4: DEPTH, PITCH, BC_SWIZZLE
  sq_img_rsrc_word4_u word4;
  word4.val = srd[4];
  printf("WORD 4: DEPTH                  = %u\n", word4.f.depth);
  printf("        PITCH                  = %u (actual: %u)\n", word4.f.pitch, word4.f.pitch + 1);
  printf("        BC_SWIZZLE             = %u\n", word4.f.bc_swizzle);
  
  // Calculate effective depth based on geometry
  uint32_t type = word3.f.type;
  if (type == 10) { // 3D
    printf("        → 3D Depth             = %u (actual: %u)\n", word4.f.depth, word4.f.depth + 1);
  } else if (type == 13 || type == 12) { // Arrays
    printf("        → Array Size           = %u (actual: %u)\n", word4.f.depth, word4.f.depth + 1);
  }
  
  // WORD 5-7: Usually zero for basic images, but may contain metadata addresses
  printf("WORD 5: META_DATA_ADDRESS_HI   = 0x%08x\n", srd[5]);
  printf("WORD 6: Reserved               = 0x%08x\n", srd[6]);
  printf("WORD 7: META_DATA_ADDRESS      = 0x%08x\n", srd[7]);
  
  // Additional mipmap information
  printf("WORD 8: CHANNEL_TYPE           = 0x%08x\n", srd[8]);
  printf("WORD 9: CHANNEL_ORDER          = 0x%08x\n", srd[9]);
  printf("WORD 10: WIDTH_ORIGINAL        = 0x%08x\n", srd[10]);
  printf("WORD 11: NUM_LEVELS            = 0x%08x\n", srd[11]);
  
  // Mipmap analysis
  if (word3.f.last_level > word3.f.base_level || word3.f.last_level > 0) {
    printf("\nMIPMAP ANALYSIS:\n");
    printf("        Total Levels           = %u\n", srd[11]);
    printf("        Active Range           = [%u, %u]\n", word3.f.base_level, word3.f.last_level);
    if (word3.f.base_level == word3.f.last_level) {
      printf("        Mode                   = SINGLE LEVEL VIEW ◄──── Mip level view\n");
      uint32_t level = word3.f.base_level;
      uint32_t level_width = std::max(1u, static_cast<uint32_t>((word2.f.width + 1) >> level));
      uint32_t level_height = std::max(1u, static_cast<uint32_t>((word2.f.height + 1) >> level));
      printf("        Effective Dimensions   = %ux%u (level %u)\n", level_width, level_height, level);
    } else {
      printf("        Mode                   = FULL MIPMAP CHAIN\n");
    }
  }
  printf("===============================================\n\n");
}

void ImageManagerAi::printChannelSelect(uint32_t sel) const {
    switch(sel) {
        case 0: printf("(SEL_0)\n"); break;
        case 1: printf("(SEL_1)\n"); break;
        case 4: printf("(SEL_X/R)\n"); break;
        case 5: printf("(SEL_Y/G)\n"); break;
        case 6: printf("(SEL_Z/B)\n"); break;
        case 7: printf("(SEL_W/A)\n"); break;
        default: printf("(UNKNOWN)\n"); break;
    }
}

void ImageManagerAi::printResourceType(uint32_t type) const {
    switch(type) {
        case 8:  printf("(1D)\n"); break;
        case 9:  printf("(2D)\n"); break;
        case 10: printf("(3D)\n"); break;
        case 11: printf("(CUBE)\n"); break;
        case 12: printf("(1D_ARRAY/1DB)\n"); break;
        case 13: printf("(2D_ARRAY)\n"); break;
        case 14: printf("(2D_MSAA)\n"); break;
        case 15: printf("(2D_MSAA_ARRAY)\n"); break;
        default: printf("(UNKNOWN=%u)\n", type); break;
    }
}

void ImageManagerAi::printSwizzleMode(uint32_t sw_mode) const {
    // GFX9 swizzle modes
    if (sw_mode == 0) {
        printf("(LINEAR)\n");
    } else if (sw_mode < 5) {
        printf("(SW_256B_%u)\n", sw_mode);
    } else if (sw_mode < 9) {
        printf("(SW_4KB_%u)\n", sw_mode - 4);
    } else if (sw_mode < 13) {
        printf("(SW_64KB_%u)\n", sw_mode - 8);
    } else if (sw_mode < 22) {
        printf("(SW_VAR_%u)\n", sw_mode - 12);
    } else {
        printf("(UNKNOWN=%u)\n", sw_mode);
    }
}

hsa_status_t ImageManagerAi::PopulateMipLevelSrd(
    MipmappedArray& level_view,
    const MipmappedArray& mipmap_array,
    uint32_t mip_level) const {
  // Copy entire parent structure (srd is a fixed array, so it's deep-copied automatically)
  level_view = mipmap_array;

  // SRD already copied from parent, just modify BASE_LEVEL/LAST_LEVEL fields
  uint32_t* srd_words = reinterpret_cast<uint32_t*>(level_view.srd);

  // SRD WORD3 has BASE_LEVEL and LAST_LEVEL fields
  sq_img_rsrc_word3_u* word3 = reinterpret_cast<sq_img_rsrc_word3_u*>(&srd_words[3]);

  // Set both to same value - hardware samples only this level
  word3->f.base_level = mip_level;
  word3->f.last_level = mip_level;

  if (core::Runtime::runtime_singleton_->flag().image_print_srd()) {
    debug_print("Set SRD mip selection: BASE_LEVEL=%u, LAST_LEVEL=%u", mip_level, mip_level);
  }

  return HSA_STATUS_SUCCESS;
}

}  // namespace image
}  // namespace rocr
