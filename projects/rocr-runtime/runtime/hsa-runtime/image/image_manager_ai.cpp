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
  if (GetAddrlibSurfaceInfoAi(component, desc, tileMode,
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

    uint32_t swizzleMode = GetAddrlibSurfaceInfoAi(image.component, image.desc, image.tile_mode,
          image.row_pitch, image.slice_pitch, out);
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
  word2.bits.MIP_FILTER = SQ_TEX_MIP_FILTER_NONE;

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
  in.pitchInElement = image_data_row_pitch / image_prop.element_size;
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

}  // namespace image
}  // namespace rocr
