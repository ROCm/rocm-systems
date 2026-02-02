////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2021, Advanced Micro Devices, Inc. All rights reserved.
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

#include "image_manager_gfx11.h"

#include <assert.h>

#include <algorithm>
#include <climits>

#include "core/inc/runtime.h"
#include "inc/hsa_ext_amd.h"
#include "core/inc/hsa_internal.h"
#include "addrlib/src/core/addrlib.h"
#include "image_runtime.h"
#include "resource.h"
#include "resource_gfx11.h"
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

//-----------------------------------------------------------------------------
// Workaround switch to combined format/type codes and missing gfx11
// specific look up table.  Only covers types used in image_lut_gfx11.cpp.
//-----------------------------------------------------------------------------
struct formatconverstion_t {
  FMT fmt;
  type type;
  FORMAT format;
};

// Format/Type to combined format code table.
// Sorted and indexed to allow fast searches.
static const formatconverstion_t FormatLUT[] = {
    {FMT_1_5_5_5, TYPE_UNORM, CFMT_1_5_5_5_UNORM},              // 0
    {FMT_10_10_10_2, TYPE_UNORM, CFMT_10_10_10_2_UNORM},        // 1
    {FMT_10_10_10_2, TYPE_SNORM, CFMT_10_10_10_2_SNORM},        // 2
    {FMT_10_10_10_2, TYPE_UINT, CFMT_10_10_10_2_UINT},          // 3
    {FMT_10_10_10_2, TYPE_SINT, CFMT_10_10_10_2_SINT},          // 4
    {FMT_16, TYPE_UNORM, CFMT_16_UNORM},                        // 5
    {FMT_16, TYPE_SNORM, CFMT_16_SNORM},                        // 6
    {FMT_16, TYPE_UINT, CFMT_16_UINT},                          // 7
    {FMT_16, TYPE_SINT, CFMT_16_SINT},                          // 8
    {FMT_16, TYPE_FLOAT, CFMT_16_FLOAT},                        // 9
    {FMT_16, TYPE_USCALED, CFMT_16_USCALED},                    // 10
    {FMT_16, TYPE_SSCALED, CFMT_16_SSCALED},                    // 11
    {FMT_16_16, TYPE_UNORM, CFMT_16_16_UNORM},                  // 12
    {FMT_16_16, TYPE_SNORM, CFMT_16_16_SNORM},                  // 13
    {FMT_16_16, TYPE_UINT, CFMT_16_16_UINT},                    // 14
    {FMT_16_16, TYPE_SINT, CFMT_16_16_SINT},                    // 15
    {FMT_16_16, TYPE_FLOAT, CFMT_16_16_FLOAT},                  // 16
    {FMT_16_16, TYPE_USCALED, CFMT_16_16_USCALED},              // 17
    {FMT_16_16, TYPE_SSCALED, CFMT_16_16_SSCALED},              // 18
    {FMT_16_16_16_16, TYPE_UNORM, CFMT_16_16_16_16_UNORM},      // 19
    {FMT_16_16_16_16, TYPE_SNORM, CFMT_16_16_16_16_SNORM},      // 20
    {FMT_16_16_16_16, TYPE_UINT, CFMT_16_16_16_16_UINT},        // 21
    {FMT_16_16_16_16, TYPE_SINT, CFMT_16_16_16_16_SINT},        // 22
    {FMT_16_16_16_16, TYPE_FLOAT, CFMT_16_16_16_16_FLOAT},      // 23
    {FMT_16_16_16_16, TYPE_USCALED, CFMT_16_16_16_16_USCALED},  // 24
    {FMT_16_16_16_16, TYPE_SSCALED, CFMT_16_16_16_16_SSCALED},  // 25
    {FMT_2_10_10_10, TYPE_UNORM, CFMT_2_10_10_10_UNORM},        // 26
    {FMT_2_10_10_10, TYPE_SNORM, CFMT_2_10_10_10_SNORM},        // 27
    {FMT_2_10_10_10, TYPE_UINT, CFMT_2_10_10_10_UINT},          // 28
    {FMT_2_10_10_10, TYPE_SINT, CFMT_2_10_10_10_SINT},          // 29
    {FMT_2_10_10_10, TYPE_USCALED, CFMT_2_10_10_10_USCALED},    // 30
    {FMT_2_10_10_10, TYPE_SSCALED, CFMT_2_10_10_10_SSCALED},    // 31
    {FMT_24_8, TYPE_UNORM, CFMT_24_8_UNORM},                    // 32
    {FMT_24_8, TYPE_UINT, CFMT_24_8_UINT},                      // 33
    {FMT_32, TYPE_UINT, CFMT_32_UINT},                          // 34
    {FMT_32, TYPE_SINT, CFMT_32_SINT},                          // 35
    {FMT_32, TYPE_FLOAT, CFMT_32_FLOAT},                        // 36
    {FMT_32_32, TYPE_UINT, CFMT_32_32_UINT},                    // 37
    {FMT_32_32, TYPE_SINT, CFMT_32_32_SINT},                    // 38
    {FMT_32_32, TYPE_FLOAT, CFMT_32_32_FLOAT},                  // 39
    {FMT_32_32_32, TYPE_UINT, CFMT_32_32_32_UINT},              // 40
    {FMT_32_32_32, TYPE_SINT, CFMT_32_32_32_SINT},              // 41
    {FMT_32_32_32, TYPE_FLOAT, CFMT_32_32_32_FLOAT},            // 42
    {FMT_32_32_32_32, TYPE_UINT, CFMT_32_32_32_32_UINT},        // 43
    {FMT_32_32_32_32, TYPE_SINT, CFMT_32_32_32_32_SINT},        // 44
    {FMT_32_32_32_32, TYPE_FLOAT, CFMT_32_32_32_32_FLOAT},      // 45
    {FMT_5_5_5_1, TYPE_UNORM, CFMT_5_5_5_1_UNORM},              // 46
    {FMT_5_6_5, TYPE_UNORM, CFMT_5_6_5_UNORM},                  // 47
    {FMT_8, TYPE_UNORM, CFMT_8_UNORM},                          // 48
    {FMT_8, TYPE_SNORM, CFMT_8_SNORM},                          // 49
    {FMT_8, TYPE_UINT, CFMT_8_UINT},                            // 50
    {FMT_8, TYPE_SINT, CFMT_8_SINT},                            // 51
    {FMT_8, TYPE_SRGB, CFMT_8_SRGB},                            // 52
    {FMT_8, TYPE_USCALED, CFMT_8_USCALED},                      // 53
    {FMT_8, TYPE_SSCALED, CFMT_8_SSCALED},                      // 54
    {FMT_8_24, TYPE_UNORM, CFMT_8_24_UNORM},                    // 55
    {FMT_8_24, TYPE_UINT, CFMT_8_24_UINT},                      // 56
    {FMT_8_8, TYPE_UNORM, CFMT_8_8_UNORM},                      // 57
    {FMT_8_8, TYPE_SNORM, CFMT_8_8_SNORM},                      // 58
    {FMT_8_8, TYPE_UINT, CFMT_8_8_UINT},                        // 59
    {FMT_8_8, TYPE_SINT, CFMT_8_8_SINT},                        // 60
    {FMT_8_8, TYPE_SRGB, CFMT_8_8_SRGB},                        // 61
    {FMT_8_8, TYPE_USCALED, CFMT_8_8_USCALED},                  // 62
    {FMT_8_8, TYPE_SSCALED, CFMT_8_8_SSCALED},                  // 63
    {FMT_8_8_8_8, TYPE_UNORM, CFMT_8_8_8_8_UNORM},              // 64
    {FMT_8_8_8_8, TYPE_SNORM, CFMT_8_8_8_8_SNORM},              // 65
    {FMT_8_8_8_8, TYPE_UINT, CFMT_8_8_8_8_UINT},                // 66
    {FMT_8_8_8_8, TYPE_SINT, CFMT_8_8_8_8_SINT},                // 67
    {FMT_8_8_8_8, TYPE_SRGB, CFMT_8_8_8_8_SRGB},                // 68
    {FMT_8_8_8_8, TYPE_USCALED, CFMT_8_8_8_8_USCALED},          // 69
    {FMT_8_8_8_8, TYPE_SSCALED, CFMT_8_8_8_8_SSCALED}           // 70
};
static const int FormatLUTSize = sizeof(FormatLUT)/sizeof(formatconverstion_t);

//Index in FormatLUT to start search, indexed by FMT enum.
static const int FormatEntryPoint[] = {
  71, // FMT_INVALID
  48, // FMT_8
  5,  // FMT_16
  57, // FMT_8_8
  34, // FMT_32
  12, // FMT_16_16
  71, // FMT_10_11_11
  71, // FMT_11_11_10
  1,  // FMT_10_10_10_2
  26, // FMT_2_10_10_10
  64, // FMT_8_8_8_8
  37, // FMT_32_32
  19, // FMT_16_16_16_16
  40, // FMT_32_32_32
  43, // FMT_32_32_32_32
  71, // RESERVED
  47, // FMT_5_6_5
  0,  // FMT_1_5_5_5
  46, // FMT_5_5_5_1
  71, // FMT_4_4_4_4
  55, // FMT_8_24
  32  // FMT_24_8
};

static FORMAT GetCombinedFormat(uint8_t fmt, uint8_t type) {
  assert(fmt < sizeof(FormatEntryPoint)/sizeof(int) && "FMT out of range.");
  int start = FormatEntryPoint[fmt];
  int stop = std::min(start + 6, FormatLUTSize); // Only 6 types are used in image_kv_lut.cpp

  for(int i=start; i<stop; i++) {
    if((FormatLUT[i].fmt == fmt) && (FormatLUT[i].type == type))
      return FormatLUT[i].format;
  }
  return CFMT_INVALID;
};
//-----------------------------------------------------------------------------
// End workaround
//-----------------------------------------------------------------------------

ImageManagerGfx11::ImageManagerGfx11() : ImageManagerKv() {}

ImageManagerGfx11::~ImageManagerGfx11() {}

// TODO(cfreehil) remove from class, make it a utility function
hsa_status_t ImageManagerGfx11::CalculateImageSizeAndAlignment(
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
  if (GetAddrlibSurfaceInfoNv(component, desc, num_mipmap_levels, tileMode,
      image_data_row_pitch, image_data_slice_pitch, out) == (uint32_t)(-1)) {
    return HSA_STATUS_ERROR;
  }

  size_t rowPitch   = (out.bpp >> 3) * out.pitch;
  size_t slicePitch = rowPitch * out.height;
  if (desc.geometry != HSA_EXT_IMAGE_GEOMETRY_1DB &&
      image_data_layout == HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR &&
      ((image_data_row_pitch && (rowPitch != image_data_row_pitch)) ||
       (image_data_slice_pitch && (slicePitch != image_data_slice_pitch)))) {
    return static_cast<hsa_status_t>(
                                HSA_EXT_STATUS_ERROR_IMAGE_PITCH_UNSUPPORTED);
  }

  image_info.size = out.surfSize;
  assert(image_info.size != 0);
  image_info.alignment = out.baseAlign;
  assert(image_info.alignment != 0);

  return HSA_STATUS_SUCCESS;
}

bool ImageManagerGfx11::IsLocalMemory(const void* address) const {
  return true;
}

hsa_status_t ImageManagerGfx11::PopulateImageSrd(Image& image,
                                     const metadata_amd_t* descriptor) const {
  const metadata_amd_gfx11_t* desc = reinterpret_cast<const metadata_amd_gfx11_t*>(descriptor);
  const void* image_data_addr = image.data;

  ImageProperty image_prop = ImageLut().MapFormat(image.desc.format, image.desc.geometry);
  if ((image_prop.cap == HSA_EXT_IMAGE_CAPABILITY_NOT_SUPPORTED) ||
     (image_prop.element_size == 0))
    return (hsa_status_t)HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED;

  const Swizzle swizzle = ImageLut().MapSwizzle(image.desc.format.channel_order);

  if (IsLocalMemory(image.data)) {
    image_data_addr = reinterpret_cast<const void*>(
        reinterpret_cast<uintptr_t>(image.data) - local_memory_base_address_);
  }

  image.srd[0] = desc->word0.u32All;
  image.srd[1] = desc->word1.u32All;
  image.srd[2] = desc->word2.u32All;
  image.srd[3] = desc->word3.u32All;
  image.srd[4] = desc->word4.u32All;
  image.srd[5] = desc->word5.u32All;
  image.srd[6] = desc->word6.u32All;
  image.srd[7] = desc->word7.u32All;

  if (image.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_1DB) {
    SQ_BUF_RSRC_WORD0 word0;
    SQ_BUF_RSRC_WORD1 word1;
    SQ_BUF_RSRC_WORD3 word3;

    word0.val = 0;
    word0.f.BASE_ADDRESS = PtrLow32(image_data_addr);

    word1.val = image.srd[1];
    word1.f.BASE_ADDRESS_HI = PtrHigh32(image_data_addr);
    word1.f.STRIDE = image_prop.element_size;

    word3.val = image.srd[3];
    word3.f.DST_SEL_X = swizzle.x;
    word3.f.DST_SEL_Y = swizzle.y;
    word3.f.DST_SEL_Z = swizzle.z;
    word3.f.DST_SEL_W = swizzle.w;

    word3.f.FORMAT = GetCombinedFormat(image_prop.data_format, image_prop.data_type);
    word3.f.INDEX_STRIDE = image_prop.element_size;

    image.srd[0] = word0.val;
    image.srd[1] = word1.val;
    image.srd[3] = word3.val;
  } else {
    uint32_t hwPixelSize = ImageLut().GetPixelSize(image_prop.data_format, image_prop.data_type);

    if (image_prop.element_size != hwPixelSize) {
      return (hsa_status_t)HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED;
    }
    reinterpret_cast<SQ_IMG_RSRC_WORD0*>(&image.srd[0])->bits.BASE_ADDRESS =
        PtrLow40Shift8(image_data_addr);
    reinterpret_cast<SQ_IMG_RSRC_WORD1*>(&image.srd[1])->bits.BASE_ADDRESS_HI =
        PtrHigh64Shift40(image_data_addr);
    reinterpret_cast<SQ_IMG_RSRC_WORD1*>(&image.srd[1])->bits.FORMAT = GetCombinedFormat(image_prop.data_format, image_prop.data_type);
    reinterpret_cast<SQ_IMG_RSRC_WORD3*>(&image.srd[3])->bits.DST_SEL_X =
                                                                    swizzle.x;
    reinterpret_cast<SQ_IMG_RSRC_WORD3*>(&image.srd[3])->bits.DST_SEL_Y =
                                                                    swizzle.y;
    reinterpret_cast<SQ_IMG_RSRC_WORD3*>(&image.srd[3])->bits.DST_SEL_Z =
                                                                    swizzle.z;
    reinterpret_cast<SQ_IMG_RSRC_WORD3*>(&image.srd[3])->bits.DST_SEL_W =
                                                                    swizzle.w;
    if (image.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_1DA ||
        image.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_1D) {
      reinterpret_cast<SQ_IMG_RSRC_WORD3*>(&image.srd[3])->bits.TYPE =
          ImageLut().MapGeometry(image.desc.geometry);
    }
    
    // Imported metadata holds the offset to metadata, add the image base address.
    uintptr_t meta = uintptr_t(((SQ_IMG_RSRC_WORD7*)(&image.srd[7]))->bits.META_DATA_ADDRESS_HI) << 16;
    meta |= uintptr_t(((SQ_IMG_RSRC_WORD6*)(&image.srd[6]))->bits.META_DATA_ADDRESS) << 8;
    meta += reinterpret_cast<uintptr_t>(image_data_addr);

    ((SQ_IMG_RSRC_WORD6*)(&image.srd[6]))->bits.META_DATA_ADDRESS = PtrLow16Shift8((void*)meta);
    ((SQ_IMG_RSRC_WORD7*)(&image.srd[7]))->bits.META_DATA_ADDRESS_HI =
        PtrHigh64Shift16((void*)meta);
  }
  // Looks like this is only used for CPU copies.
  image.row_pitch = 0;
  image.slice_pitch = 0;

  // Used by HSAIL shader ABI
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

    if (a == SEL_X) {
        // Have to use either TEX_BC_Swizzle_WZYX or TEX_BC_Swizzle_WXYZ
        //
        // For the pre-defined border color values (white, opaque black,
        // transparent black), the only thing that matters is that the alpha
        // channel winds up in the correct place (because the RGB channels are
        // all the same) so either of these TEX_BC_Swizzle enumerations will
        // work.  Not sure what happens with border color palettes.
        if (b == SEL_Y) {
            // ABGR
            bcSwizzle = TEX_BC_Swizzle_WZYX;
        } else if ((r == SEL_X) && (g == SEL_X) && (b == SEL_X)) {
            // RGBA
            bcSwizzle = TEX_BC_Swizzle_XYZW;
        } else {
            // ARGB
            bcSwizzle = TEX_BC_Swizzle_WXYZ;
        }
    } else if (r == SEL_X) {
        // Have to use either TEX_BC_Swizzle_XYZW or TEX_BC_Swizzle_XWYZ
        if (g == SEL_Y) {
            // RGBA
            bcSwizzle = TEX_BC_Swizzle_XYZW;
        } else if ((g == SEL_X) && (b == SEL_X) && (a == SEL_W)) {
            // RGBA
            bcSwizzle = TEX_BC_Swizzle_XYZW;
        } else {
            // RAGB
            bcSwizzle = TEX_BC_Swizzle_XWYZ;
        }
    } else if (g == SEL_X) {
        // GRAB, have to use TEX_BC_Swizzle_YXWZ
        bcSwizzle = TEX_BC_Swizzle_YXWZ;
    } else if (b == SEL_X) {
        // BGRA, have to use TEX_BC_Swizzle_ZYXW
        bcSwizzle = TEX_BC_Swizzle_ZYXW;
    }

    return bcSwizzle;
}


hsa_status_t ImageManagerGfx11::PopulateImageSrd(Image& image) const {
  ImageProperty image_prop = ImageLut().MapFormat(image.desc.format, image.desc.geometry);
  assert(image_prop.cap != HSA_EXT_IMAGE_CAPABILITY_NOT_SUPPORTED);
  assert(image_prop.element_size != 0);

  const void* image_data_addr = image.data;

  if (IsLocalMemory(image.data))
    image_data_addr = reinterpret_cast<const void*>(
        reinterpret_cast<uintptr_t>(image.data) - local_memory_base_address_);

  if (image.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_1DB) {
    SQ_BUF_RSRC_WORD0 word0;
    SQ_BUF_RSRC_WORD1 word1;
    SQ_BUF_RSRC_WORD2 word2;
    SQ_BUF_RSRC_WORD3 word3;

    word0.val = 0;
    word0.f.BASE_ADDRESS = PtrLow32(image_data_addr);

    word1.val = 0;
    word1.f.BASE_ADDRESS_HI = PtrHigh32(image_data_addr);
    word1.f.STRIDE = image_prop.element_size;
    word1.f.SWIZZLE_ENABLE = 0;

    word2.f.NUM_RECORDS = image.desc.width * image_prop.element_size;

    const Swizzle swizzle = ImageLut().MapSwizzle(image.desc.format.channel_order);
    word3.val = 0;
    word3.f.DST_SEL_X = swizzle.x;
    word3.f.DST_SEL_Y = swizzle.y;
    word3.f.DST_SEL_Z = swizzle.z;
    word3.f.DST_SEL_W = swizzle.w;
    word3.f.FORMAT = GetCombinedFormat(image_prop.data_format, image_prop.data_type);
    word3.f.INDEX_STRIDE = image_prop.element_size;
    word3.f.TYPE = ImageLut().MapGeometry(image.desc.geometry);

    image.srd[0] = word0.val;
    image.srd[1] = word1.val;
    image.srd[2] = word2.val;
    image.srd[3] = word3.val;

    image.row_pitch = image.desc.width * image_prop.element_size;
    image.slice_pitch = image.row_pitch;
  } else {
    SQ_IMG_RSRC_WORD0 word0;
    SQ_IMG_RSRC_WORD1 word1;
    SQ_IMG_RSRC_WORD2 word2;
    SQ_IMG_RSRC_WORD3 word3;
    SQ_IMG_RSRC_WORD4 word4;
    SQ_IMG_RSRC_WORD5 word5;
    SQ_IMG_RSRC_WORD5 word6;
    SQ_IMG_RSRC_WORD5 word7;

    ADDR2_COMPUTE_SURFACE_INFO_OUTPUT out = {0};

    uint32_t swizzleMode = GetAddrlibSurfaceInfoNv(image.component, image.desc,
                  1, image.tile_mode, image.row_pitch, image.slice_pitch, out);
    if (swizzleMode == (uint32_t)(-1)) {
      return HSA_STATUS_ERROR;
    }

    assert((out.bpp / 8) == image_prop.element_size);

    const size_t row_pitch_size = out.pitch * image_prop.element_size;

    word0.f.BASE_ADDRESS = PtrLow40Shift8(image_data_addr);

    word1.val = 0;
    word1.f.BASE_ADDRESS_HI = PtrHigh64Shift40(image_data_addr);
    word1.f.FORMAT = GetCombinedFormat(image_prop.data_format, image_prop.data_type);
    // Only take the lowest 2 bits of (image.desc.width - 1)
    word1.f.WIDTH = BitSelect<0, 1>(image.desc.width - 1);

    word2.val = 0;
    // Take the high 12 bits of (image.desc.width - 1)
    word2.f.WIDTH_HI = BitSelect<2, 13>(image.desc.width - 1);
    word2.f.HEIGHT = image.desc.height ? image.desc.height - 1 : 0;

    const Swizzle swizzle = ImageLut().MapSwizzle(image.desc.format.channel_order);
    word3.val = 0;
    word3.f.DST_SEL_X = swizzle.x;
    word3.f.DST_SEL_Y = swizzle.y;
    word3.f.DST_SEL_Z = swizzle.z;
    word3.f.DST_SEL_W = swizzle.w;
    word3.f.SW_MODE = swizzleMode;
    word3.f.BC_SWIZZLE = GetBcSwizzle(swizzle);
    word3.f.TYPE = ImageLut().MapGeometry(image.desc.geometry);

    const bool image_array =
        (image.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_1DA ||
         image.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_2DA ||
         image.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_2DADEPTH);
    const bool image_3d = (image.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_3D);

    word4.val = 0;
    word4.f.DEPTH =
        (image_array) // Doesn't hurt but isn't array_size already >0?
            ? std::max(image.desc.array_size, static_cast<size_t>(1)) - 1
            : (image_3d) ? image.desc.depth - 1 : 0;

    // For 1d, 2d and 2d-msaa in gfx11 this is pitch-1
    if (!image_array && !image_3d) word4.f.PITCH = out.pitch - 1;

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

hsa_status_t ImageManagerGfx11::ModifyImageSrd(
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
    word3->bits.FORMAT = GetCombinedFormat(image_prop.data_format, image_prop.data_type);
  } else {
    SQ_IMG_RSRC_WORD1* word1 =
        reinterpret_cast<SQ_IMG_RSRC_WORD1*>(&image.srd[1]);
    word1->bits.FORMAT = GetCombinedFormat(image_prop.data_format, image_prop.data_type);

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

hsa_status_t ImageManagerGfx11::PopulateSamplerSrd(Sampler& sampler) const {
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

uint32_t ImageManagerGfx11::GetAddrlibSurfaceInfoNv(
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

  switch (desc.geometry) {
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
  if (tileMode == Image::TileMode::LINEAR) {
      prefSettingsInput.forbiddenBlock.macroThin4KB = 1;
      prefSettingsInput.forbiddenBlock.macroThick4KB = 1;
      prefSettingsInput.forbiddenBlock.macroThin64KB = 1;
      prefSettingsInput.forbiddenBlock.macroThick64KB = 1;
      prefSettingsInput.forbiddenBlock.micro = 1;
      prefSettingsInput.forbiddenBlock.var = 1;
  }

  // but don't ever allow the 256b swizzle modes
  //prefSettingsInput.forbiddenBlock.micro = 1;
  // and don't allow variable-size block modes
  //prefSettingsInput.forbiddenBlock.var = 1;

  if (ADDR_OK != Addr2GetPreferredSurfaceSetting(addr_lib_,
                                   &prefSettingsInput, &prefSettingsOutput)) {
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

hsa_status_t ImageManagerGfx11::FillImage(const Image& image, const void* pattern,
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
      dst_sel_w_original = word3_buff->bits.DST_SEL_W;
      word3_buff->bits.DST_SEL_W = SEL_0;
    } else {
      word3_image = reinterpret_cast<SQ_IMG_RSRC_WORD3*>(&image_view->srd[3]);
      dst_sel_w_original = word3_image->bits.DST_SEL_W;
      word3_image->bits.DST_SEL_W = SEL_0;
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
      // We do not have write support for SRGBA image, so convert pattern
      // to standard form and treat the image as RGBA image.
      const float* pattern_f = reinterpret_cast<const float*>(pattern);
      fill_value[0] = LinearToStandardRGB(pattern_f[0]);
      fill_value[1] = LinearToStandardRGB(pattern_f[1]);
      fill_value[2] = LinearToStandardRGB(pattern_f[2]);
      fill_value[3] = pattern_f[3];
      new_pattern = fill_value;

      ImageProperty image_prop = ImageLut().MapFormat(image.desc.format, image.desc.geometry);

      word1 = reinterpret_cast<SQ_IMG_RSRC_WORD1*>(&image_view->srd[1]);
      num_format_original = word1->bits.FORMAT;
      word1->bits.FORMAT = GetCombinedFormat(image_prop.data_format, TYPE_UNORM);
    } break;
    default:
      break;
  }

  hsa_status_t status = ImageRuntime::instance()->blit_kernel().FillImage(
      blit_queue_, blit_code_catalog_, *image_view, new_pattern, region);

  // Revert back original configuration.
  if (word3_buff != NULL) {
    word3_buff->bits.DST_SEL_W = dst_sel_w_original;
  }

  if (word3_image != NULL) {
    word3_image->bits.DST_SEL_W = dst_sel_w_original;
  }

  if (word1 != NULL) {
    word1->bits.FORMAT = num_format_original;
  }

  return status;
}

hsa_status_t ImageManagerGfx11::PopulateMipmapSrd(MipmappedArray& mipmap) const {
  ImageProperty mipmap_prop = ImageLut().MapFormat(mipmap.desc.format, mipmap.desc.geometry);
  assert(mipmap_prop.cap != HSA_EXT_IMAGE_CAPABILITY_NOT_SUPPORTED);
  assert(mipmap_prop.element_size != 0);
  assert(mipmap.num_levels >= 1);

  const void* mipmap_data_addr = mipmap.data;

  if (IsLocalMemory(mipmap.data))
    mipmap_data_addr = reinterpret_cast<const void*>(
        reinterpret_cast<uintptr_t>(mipmap.data) - local_memory_base_address_);

  if (mipmap.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_1DB) {
    SQ_BUF_RSRC_WORD0 word0;
    SQ_BUF_RSRC_WORD1 word1;
    SQ_BUF_RSRC_WORD2 word2;
    SQ_BUF_RSRC_WORD3 word3;

    word0.val = 0;
    word0.f.BASE_ADDRESS = PtrLow32(mipmap_data_addr);

    word1.val = 0;
    word1.f.BASE_ADDRESS_HI = PtrHigh32(mipmap_data_addr);
    word1.f.STRIDE = mipmap_prop.element_size;
    word1.f.SWIZZLE_ENABLE = 0;

    word2.f.NUM_RECORDS = mipmap.desc.width * mipmap_prop.element_size;

    const Swizzle swizzle = ImageLut().MapSwizzle(mipmap.desc.format.channel_order);
    word3.val = 0;
    word3.f.DST_SEL_X = swizzle.x;
    word3.f.DST_SEL_Y = swizzle.y;
    word3.f.DST_SEL_Z = swizzle.z;
    word3.f.DST_SEL_W = swizzle.w;
    word3.f.FORMAT = GetCombinedFormat(mipmap_prop.data_format, mipmap_prop.data_type);
    word3.f.INDEX_STRIDE = mipmap_prop.element_size;
    word3.f.TYPE = ImageLut().MapGeometry(mipmap.desc.geometry);

    mipmap.srd[0] = word0.val;
    mipmap.srd[1] = word1.val;
    mipmap.srd[2] = word2.val;
    mipmap.srd[3] = word3.val;

    mipmap.row_pitch = mipmap.desc.width * mipmap_prop.element_size;
    mipmap.slice_pitch = mipmap.row_pitch;
  } else {
    SQ_IMG_RSRC_WORD0 word0;
    SQ_IMG_RSRC_WORD1 word1;
    SQ_IMG_RSRC_WORD2 word2;
    SQ_IMG_RSRC_WORD3 word3;
    SQ_IMG_RSRC_WORD4 word4;
    SQ_IMG_RSRC_WORD5 word5;
    SQ_IMG_RSRC_WORD5 word6;
    SQ_IMG_RSRC_WORD5 word7;

    ADDR2_COMPUTE_SURFACE_INFO_OUTPUT out = {0};

    // pMipInfo not needed - set to nullptr and AddrLib will ignore it
    out.pMipInfo = nullptr;

    uint32_t swizzleMode = GetAddrlibSurfaceInfoNv(
                        mipmap.component, mipmap.desc, mipmap.num_levels,
                        mipmap.tile_mode, mipmap.row_pitch, mipmap.slice_pitch, out);
    if (swizzleMode == (uint32_t)(-1)) {
      return HSA_STATUS_ERROR;
    }
    mipmap.addr_output.addr2 = out;
    mipmap.size = out.surfSize;

    assert((out.bpp / 8) == mipmap_prop.element_size);

    const size_t row_pitch_size = out.pitch * mipmap_prop.element_size;

    word0.val = 0;
    word0.f.BASE_ADDRESS = PtrLow40Shift8(mipmap_data_addr);

    word1.val = 0;
    word1.f.BASE_ADDRESS_HI = PtrHigh64Shift40(mipmap_data_addr);
    word1.f.MAX_MIP = mipmap.num_levels - 1;
    word1.f.FORMAT = GetCombinedFormat(mipmap_prop.data_format, mipmap_prop.data_type);
    // Only take the lowest 2 bits of (mipmap.desc.width - 1)
    word1.f.WIDTH = BitSelect<0, 1>(mipmap.desc.width - 1);

    word2.val = 0;
    // Take the high 12 bits of (mipmap.desc.width - 1)
    word2.f.WIDTH_HI = BitSelect<2, 13>(mipmap.desc.width - 1);
    word2.f.HEIGHT = mipmap.desc.height ? mipmap.desc.height - 1 : 0;

    const Swizzle swizzle = ImageLut().MapSwizzle(mipmap.desc.format.channel_order);
    word3.val = 0;
    word3.f.DST_SEL_X = swizzle.x;
    word3.f.DST_SEL_Y = swizzle.y;
    word3.f.DST_SEL_Z = swizzle.z;
    word3.f.DST_SEL_W = swizzle.w;
    word3.f.SW_MODE = swizzleMode;
    word3.f.BASE_LEVEL = 0;
    word3.f.LAST_LEVEL = mipmap.num_levels - 1;
    word3.f.BC_SWIZZLE = GetBcSwizzle(swizzle);
    word3.f.TYPE = ImageLut().MapGeometry(mipmap.desc.geometry);

    const bool mipmap_array =
        (mipmap.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_1DA ||
         mipmap.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_2DA ||
         mipmap.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_2DADEPTH);
    const bool mipmap_3d = (mipmap.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_3D);

    word4.val = 0;
    word4.f.DEPTH =
        (mipmap_array) // Doesn't hurt but isn't array_size already >0?
            ? std::max(mipmap.desc.array_size, static_cast<size_t>(1)) - 1
            : (mipmap_3d) ? mipmap.desc.depth - 1 : 0;

    // For 1d, 2d and 2d-msaa in gfx11 this is pitch-1
    if (!mipmap_array && !mipmap_3d) {
      word4.f.PITCH = 0; // mipmap dosesn't support custom pitch, so set it as 0
    }
    word5.val = 0;
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

  // Mipmap-specific auxiliary fields
  mipmap.srd[11] = mipmap.num_levels;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t ImageManagerGfx11::PopulateMipmapSrd(MipmappedArray& mipmap_array, const metadata_amd_t* desc) const {
  const metadata_amd_gfx11_t* desc_gfx11 = reinterpret_cast<const metadata_amd_gfx11_t*>(desc);
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
  mipmap_array.srd[0] = desc_gfx11->word0.u32All;
  mipmap_array.srd[1] = desc_gfx11->word1.u32All;
  mipmap_array.srd[2] = desc_gfx11->word2.u32All;
  mipmap_array.srd[3] = desc_gfx11->word3.u32All;
  mipmap_array.srd[4] = desc_gfx11->word4.u32All;
  mipmap_array.srd[5] = desc_gfx11->word5.u32All;
  mipmap_array.srd[6] = desc_gfx11->word6.u32All;
  mipmap_array.srd[7] = desc_gfx11->word7.u32All;
  
  // Override specific fields after copying
  uint32_t hwPixelSize = ImageLut().GetPixelSize(mipmap_prop.data_format, mipmap_prop.data_type);
  if (mipmap_prop.element_size != hwPixelSize) {
    return (hsa_status_t)HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED;
  }
  
  reinterpret_cast<SQ_IMG_RSRC_WORD0*>(&mipmap_array.srd[0])->bits.BASE_ADDRESS = PtrLow40Shift8(mipmap_data_addr);
  reinterpret_cast<SQ_IMG_RSRC_WORD1*>(&mipmap_array.srd[1])->bits.BASE_ADDRESS_HI = PtrHigh64Shift40(mipmap_data_addr);
  reinterpret_cast<SQ_IMG_RSRC_WORD1*>(&mipmap_array.srd[1])->bits.FORMAT = GetCombinedFormat(mipmap_prop.data_format, mipmap_prop.data_type);
  reinterpret_cast<SQ_IMG_RSRC_WORD1*>(&mipmap_array.srd[1])->bits.MAX_MIP = mipmap_array.num_levels - 1;
  reinterpret_cast<SQ_IMG_RSRC_WORD3*>(&mipmap_array.srd[3])->bits.DST_SEL_X = swizzle.x;
  reinterpret_cast<SQ_IMG_RSRC_WORD3*>(&mipmap_array.srd[3])->bits.DST_SEL_Y = swizzle.y;
  reinterpret_cast<SQ_IMG_RSRC_WORD3*>(&mipmap_array.srd[3])->bits.DST_SEL_Z = swizzle.z;
  reinterpret_cast<SQ_IMG_RSRC_WORD3*>(&mipmap_array.srd[3])->bits.DST_SEL_W = swizzle.w;
  
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
  
  // Allocate and populate pMipInfo from metadata mip_offsets (ADDR2 for GFX11)
  ADDR2_MIP_INFO* mip_info_storage = new ADDR2_MIP_INFO[mipmap_array.num_levels];
  memset(mip_info_storage, 0, sizeof(ADDR2_MIP_INFO) * mipmap_array.num_levels);
  
  // Extract per-level information from mip_offsets array
  for (uint32_t level = 0; level < mipmap_array.num_levels; level++) {
    // mip_offsets contains offset bits [39:8], shift left by 8 to get actual byte offset
    mip_info_storage[level].offset = static_cast<uint64_t>(desc_gfx11->mip_offsets[level]) << 8;
    
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

void ImageManagerGfx11::printSRDDetailed(const uint32_t* srd) const {
  if (!srd) {
    printf("\n========== Image SRD (GFX11) - Detailed ==========\n");
    printf("ERROR: No SRD data provided.\n");
    printf("===============================================\n\n");
    return;
  }

  printf("\n========== Image SRD (GFX11) - Detailed ==========\n");

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
  SQ_IMG_RSRC_WORD0 word0;
  word0.val = srd[0];
  printf("\nWORD 0: BASE_ADDRESS (bits 39:8) = 0x%08x\n", word0.f.BASE_ADDRESS);
  
  // WORD 1: Contains BASE_ADDRESS_HI, FORMAT, WIDTH (bits 1:0)
  SQ_IMG_RSRC_WORD1 word1;
  word1.val = srd[1];
  printf("WORD 1: BASE_ADDRESS_HI        = 0x%02x\n", word1.f.BASE_ADDRESS_HI);
  printf("        FORMAT                 = %u\n", word1.f.FORMAT);
  printf("        WIDTH (bits 1:0)       = %u\n", word1.f.WIDTH);
  
  // Calculate full address (GFX11 uses 40-bit shifted by 8)
  uint64_t base_addr = ((uint64_t)word1.f.BASE_ADDRESS_HI << 32) | ((uint64_t)word0.f.BASE_ADDRESS << 8);
  printf("        → Full Base Address    = 0x%016lx\n", base_addr);
  
  // WORD 2: WIDTH_HI, HEIGHT
  SQ_IMG_RSRC_WORD2 word2;
  word2.val = srd[2];
  printf("WORD 2: WIDTH_HI (bits 13:2)   = %u\n", word2.f.WIDTH_HI);
  printf("        HEIGHT                 = %u\n", word2.f.HEIGHT);
  
  // Calculate full width (GFX11 uses 14 bits split: 2 in WORD1 + 12 in WORD2)
  uint32_t full_width = word1.f.WIDTH | (word2.f.WIDTH_HI << 2);
  printf("        → Full Width           = %u (actual: %u)\n", full_width, full_width + 1);
  printf("        → Full Height          = %u (actual: %u)\n", word2.f.HEIGHT, word2.f.HEIGHT + 1);
  
  // WORD 3: Channel selectors, SW_MODE, BASE_LEVEL, LAST_LEVEL, BC_SWIZZLE, TYPE
  SQ_IMG_RSRC_WORD3 word3;
  word3.val = srd[3];
  printf("WORD 3: DST_SEL_X              = %u ", word3.f.DST_SEL_X);
  printChannelSelect(word3.f.DST_SEL_X);
  printf("        DST_SEL_Y              = %u ", word3.f.DST_SEL_Y);
  printChannelSelect(word3.f.DST_SEL_Y);
  printf("        DST_SEL_Z              = %u ", word3.f.DST_SEL_Z);
  printChannelSelect(word3.f.DST_SEL_Z);
  printf("        DST_SEL_W              = %u ", word3.f.DST_SEL_W);
  printChannelSelect(word3.f.DST_SEL_W);
  printf("        BASE_LEVEL             = %u ◄──── Current base level\n", word3.f.BASE_LEVEL);
  printf("        LAST_LEVEL             = %u ◄──── Current last level\n", word3.f.LAST_LEVEL);
  printf("        SW_MODE                = %u ", word3.f.SW_MODE);
  printSwizzleMode(word3.f.SW_MODE);
  printf("        BC_SWIZZLE             = %u\n", word3.f.BC_SWIZZLE);
  printf("        TYPE                   = %u ", word3.f.TYPE);
  printResourceType(word3.f.TYPE);
  
  // WORD 4: DEPTH, PITCH
  SQ_IMG_RSRC_WORD4 word4;
  word4.val = srd[4];
  printf("WORD 4: DEPTH                  = %u\n", word4.f.DEPTH);
  printf("        PITCH                  = %u (actual: %u)\n", word4.f.PITCH, word4.f.PITCH + 1);
  
  // Calculate effective depth based on geometry
  uint32_t type = word3.f.TYPE;
  if (type == 10) { // 3D
    printf("        → 3D Depth             = %u (actual: %u)\n", word4.f.DEPTH, word4.f.DEPTH + 1);
  } else if (type == 13 || type == 12) { // Arrays
    printf("        → Array Size           = %u (actual: %u)\n", word4.f.DEPTH, word4.f.DEPTH + 1);
  }
  
  // WORD 5-7: Usually zero for basic images, but may contain metadata addresses
  printf("WORD 5: Reserved               = 0x%08x\n", srd[5]);
  printf("WORD 6: META_DATA_ADDRESS      = 0x%08x\n", srd[6]);
  printf("WORD 7: META_DATA_ADDRESS_HI   = 0x%08x\n", srd[7]);
  
  // Additional mipmap information
  printf("WORD 8: CHANNEL_TYPE           = 0x%08x\n", srd[8]);
  printf("WORD 9: CHANNEL_ORDER          = 0x%08x\n", srd[9]);
  printf("WORD 10: WIDTH_ORIGINAL        = 0x%08x\n", srd[10]);
  printf("WORD 11: NUM_LEVELS            = 0x%08x\n", srd[11]);
  
  // Mipmap analysis
  if (word3.f.LAST_LEVEL > word3.f.BASE_LEVEL || word3.f.LAST_LEVEL > 0) {
    printf("\nMIPMAP ANALYSIS:\n");
    printf("        Total Levels           = %u\n", srd[11]);
    printf("        Active Range           = [%u, %u]\n", word3.f.BASE_LEVEL, word3.f.LAST_LEVEL);
    if (word3.f.BASE_LEVEL == word3.f.LAST_LEVEL) {
      printf("        Mode                   = SINGLE LEVEL VIEW ◄──── Mip level view\n");
      uint32_t level = word3.f.BASE_LEVEL;
      uint32_t level_width = std::max(1u, (full_width + 1) >> level);
      uint32_t level_height = std::max(1u, static_cast<uint32_t>((word2.f.HEIGHT + 1) >> level));
      printf("        Effective Dimensions   = %ux%u (level %u)\n", level_width, level_height, level);
    } else {
      printf("        Mode                   = FULL MIPMAP CHAIN\n");
    }
  }
  printf("===============================================\n\n");
}

void ImageManagerGfx11::printChannelSelect(uint32_t sel) const {
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

void ImageManagerGfx11::printResourceType(uint32_t type) const {
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

void ImageManagerGfx11::printSwizzleMode(uint32_t sw_mode) const {
    // GFX11 swizzle modes (similar to GFX9/10)
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

hsa_status_t ImageManagerGfx11::PopulateMipLevelSrd(
    MipmappedArray& level_view,
    const MipmappedArray& mipmap_array,
    uint32_t mip_level) const {
  // Copy entire parent structure (srd is a fixed array, so it's deep-copied automatically)
  level_view = mipmap_array;

  // SRD already copied from parent, just modify BASE_LEVEL/LAST_LEVEL fields
  uint32_t* srd_words = reinterpret_cast<uint32_t*>(level_view.srd);

  // GFX11 SRD WORD3 has BASE_LEVEL and LAST_LEVEL fields
  SQ_IMG_RSRC_WORD3* word3 = reinterpret_cast<SQ_IMG_RSRC_WORD3*>(&srd_words[3]);

  // Set both to same value - hardware samples only this level
  word3->f.BASE_LEVEL = mip_level;
  word3->f.LAST_LEVEL = mip_level;

  if (core::Runtime::runtime_singleton_->flag().image_print_srd()) {
    debug_print("Set SRD mip selection: BASE_LEVEL=%u, LAST_LEVEL=%u", mip_level, mip_level);
  }

  return HSA_STATUS_SUCCESS;
}

}  // namespace image
}  // namespace rocr
