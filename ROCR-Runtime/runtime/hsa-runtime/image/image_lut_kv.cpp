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

#include "image_lut_kv.h"
#include "resource_kv.h"

namespace rocr {
namespace image {

const uint32_t ImageLutKv::kGeometryLut_[GEOMETRY_COUNT] = {
    SQ_RSRC_IMG_1D,        // HSA_EXT_IMAGE_GEOMETRY_1D
    SQ_RSRC_IMG_2D,        // HSA_EXT_IMAGE_GEOMETRY_2D
    SQ_RSRC_IMG_3D,        // HSA_EXT_IMAGE_GEOMETRY_3D
    SQ_RSRC_IMG_1D_ARRAY,  // HSA_EXT_IMAGE_GEOMETRY_1DA
    SQ_RSRC_IMG_2D_ARRAY,  // HSA_EXT_IMAGE_GEOMETRY_2DA
    0,                     // HSA_EXT_IMAGE_GEOMETRY_1DB
    SQ_RSRC_IMG_2D,        // HSA_EXT_IMAGE_GEOMETRY_2DDEPTH
    SQ_RSRC_IMG_2D_ARRAY   // HSA_EXT_IMAGE_GEOMETRY_2DADEPTH
};

const ImageProperty ImageLutKv::kPropLut_[ORDER_COUNT][TYPE_COUNT] = {
    {// HSA_EXT_IMAGE_CHANNEL_ORDER_A
     {RW, 1, FMT_8, TYPE_SNORM},
     {RW, 2, FMT_16, TYPE_SNORM},
     {RW, 1, FMT_8, TYPE_UNORM},
     {RW, 2, FMT_16, TYPE_UNORM},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {RW, 1, FMT_8, TYPE_SINT},
     {RW, 2, FMT_16, TYPE_SINT},
     {RW, 4, FMT_32, TYPE_SINT},
     {RW, 1, FMT_8, TYPE_UINT},
     {RW, 2, FMT_16, TYPE_UINT},
     {RW, 4, FMT_32, TYPE_UINT},
     {RW, 2, FMT_16, TYPE_FLOAT},
     {RW, 4, FMT_32, TYPE_FLOAT}},
    {// HSA_EXT_IMAGE_CHANNEL_ORDER_R
     {RW, 1, FMT_8, TYPE_SNORM},
     {RW, 2, FMT_16, TYPE_SNORM},
     {RW, 1, FMT_8, TYPE_UNORM},
     {RW, 2, FMT_16, TYPE_UNORM},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {RW, 1, FMT_8, TYPE_SINT},
     {RW, 2, FMT_16, TYPE_SINT},
     {RW, 4, FMT_32, TYPE_SINT},
     {RW, 1, FMT_8, TYPE_UINT},
     {RW, 2, FMT_16, TYPE_UINT},
     {RW, 4, FMT_32, TYPE_UINT},
     {RW, 2, FMT_16, TYPE_FLOAT},
     {RW, 4, FMT_32, TYPE_FLOAT}},
    {},  // HSA_EXT_IMAGE_CHANNEL_ORDER_RX
    {     // HSA_EXT_IMAGE_CHANNEL_ORDER_RG
     {RW, 2, FMT_8_8, TYPE_SNORM},
     {RW, 4, FMT_16_16, TYPE_SNORM},
     {RW, 2, FMT_8_8, TYPE_UNORM},
     {RW, 4, FMT_16_16, TYPE_UNORM},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {RW, 2, FMT_8_8, TYPE_SINT},
     {RW, 4, FMT_16_16, TYPE_SINT},
     {RW, 8, FMT_32_32, TYPE_SINT},
     {RW, 2, FMT_8_8, TYPE_UINT},
     {RW, 4, FMT_16_16, TYPE_UINT},
     {RW, 8, FMT_32_32, TYPE_UINT},
     {RW, 4, FMT_16_16, TYPE_FLOAT},
     {RW, 8, FMT_32_32, TYPE_FLOAT}},
    {},  // HSA_EXT_IMAGE_CHANNEL_ORDER_RGX
    {     // HSA_EXT_IMAGE_CHANNEL_ORDER_RA
     {RW, 2, FMT_8_8, TYPE_SNORM},
     {RW, 4, FMT_16_16, TYPE_SNORM},
     {RW, 2, FMT_8_8, TYPE_UNORM},
     {RW, 4, FMT_16_16, TYPE_UNORM},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {RW, 2, FMT_8_8, TYPE_SINT},
     {RW, 4, FMT_16_16, TYPE_SINT},
     {RW, 8, FMT_32_32, TYPE_SINT},
     {RW, 2, FMT_8_8, TYPE_UINT},
     {RW, 4, FMT_16_16, TYPE_UINT},
     {RW, 8, FMT_32_32, TYPE_UINT},
     {RW, 4, FMT_16_16, TYPE_FLOAT},
     {RW, 8, FMT_32_32, TYPE_FLOAT}},
    {// HSA_EXT_IMAGE_CHANNEL_ORDER_RGB
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {RW, 2, FMT_1_5_5_5, TYPE_UNORM},
     {RW, 2, FMT_5_6_5, TYPE_UNORM},
     {RW, 4, FMT_2_10_10_10, TYPE_UNORM},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0}},
    {},  // HSA_EXT_IMAGE_CHANNEL_ORDER_RGBX
    {     // HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA
     {RW, 4, FMT_8_8_8_8, TYPE_SNORM},
     {RW, 8, FMT_16_16_16_16, TYPE_SNORM},
     {RW, 4, FMT_8_8_8_8, TYPE_UNORM},
     {RW, 8, FMT_16_16_16_16, TYPE_UNORM},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {RW, 4, FMT_8_8_8_8, TYPE_SINT},
     {RW, 8, FMT_16_16_16_16, TYPE_SINT},
     {RW, 16, FMT_32_32_32_32, TYPE_SINT},
     {RW, 4, FMT_8_8_8_8, TYPE_UINT},
     {RW, 8, FMT_16_16_16_16, TYPE_UINT},
     {RW, 16, FMT_32_32_32_32, TYPE_UINT},
     {RW, 8, FMT_16_16_16_16, TYPE_FLOAT},
     {RW, 16, FMT_32_32_32_32, TYPE_FLOAT}},
    {// HSA_EXT_IMAGE_CHANNEL_ORDER_BGRA
     {RW, 4, FMT_8_8_8_8, TYPE_SNORM},
     {0, 0, 0, 0},
     {RW, 4, FMT_8_8_8_8, TYPE_UNORM},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {RW, 4, FMT_8_8_8_8, TYPE_SINT},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {RW, 4, FMT_8_8_8_8, TYPE_UINT},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0}},
    {// HSA_EXT_IMAGE_CHANNEL_ORDER_ARGB
     {RW, 4, FMT_8_8_8_8, TYPE_SNORM},
     {0, 0, 0, 0},
     {RW, 4, FMT_8_8_8_8, TYPE_UNORM},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {RW, 4, FMT_8_8_8_8, TYPE_SINT},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {RW, 4, FMT_8_8_8_8, TYPE_UINT},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0}},
    {},  // HSA_EXT_IMAGE_CHANNEL_ORDER_ABGR
    {},  // HSA_EXT_IMAGE_CHANNEL_ORDER_SRGB
    {},  // HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBX
    {     // HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBA
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {RO, 4, FMT_8_8_8_8, TYPE_SRGB},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0}},
    {},  // HSA_EXT_IMAGE_CHANNEL_ORDER_SBGRA
    {     // HSA_EXT_IMAGE_CHANNEL_ORDER_INTENSITY
     {RW, 1, FMT_8, TYPE_SNORM},
     {RW, 2, FMT_16, TYPE_SNORM},
     {RW, 1, FMT_8, TYPE_UNORM},
     {RW, 2, FMT_16, TYPE_UNORM},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {RW, 2, FMT_16, TYPE_FLOAT},
     {RW, 4, FMT_32, TYPE_FLOAT}},
    {// HSA_EXT_IMAGE_CHANNEL_ORDER_LUMINANCE
     {RW, 1, FMT_8, TYPE_SNORM},
     {RW, 2, FMT_16, TYPE_SNORM},
     {RW, 1, FMT_8, TYPE_UNORM},
     {RW, 2, FMT_16, TYPE_UNORM},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {RW, 2, FMT_16, TYPE_FLOAT},
     {RW, 4, FMT_32, TYPE_FLOAT}},
    {// HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {ROWO, 2, FMT_16, TYPE_UNORM},
     // TODO: 24 bit
     {0, 3, FMT_32, TYPE_UNORM},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {0, 0, 0, 0},
     {ROWO, 4, FMT_32, TYPE_FLOAT}},
    {}  // HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH_STENCIL
};

const Swizzle ImageLutKv::kSwizzleLut_[ORDER_COUNT] = {
    {SEL_0, SEL_0, SEL_0, SEL_X},  // HSA_EXT_IMAGE_CHANNEL_ORDER_A
    {SEL_X, SEL_0, SEL_0, SEL_1},  // HSA_EXT_IMAGE_CHANNEL_ORDER_R
    {SEL_X, SEL_0, SEL_0, SEL_1},  // HSA_EXT_IMAGE_CHANNEL_ORDER_RX
    {SEL_X, SEL_Y, SEL_0, SEL_1},  // HSA_EXT_IMAGE_CHANNEL_ORDER_RG
    {SEL_X, SEL_Y, SEL_0, SEL_1},  // HSA_EXT_IMAGE_CHANNEL_ORDER_RGX
    {SEL_X, SEL_0, SEL_0, SEL_Y},  // HSA_EXT_IMAGE_CHANNEL_ORDER_RA
    {SEL_Z, SEL_Y, SEL_X, SEL_1},  // HSA_EXT_IMAGE_CHANNEL_ORDER_RGB
    {SEL_Z, SEL_Y, SEL_X, SEL_1},  // HSA_EXT_IMAGE_CHANNEL_ORDER_RGBX
    {SEL_X, SEL_Y, SEL_Z, SEL_W},  // HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA
    {SEL_Z, SEL_Y, SEL_X, SEL_W},  // HSA_EXT_IMAGE_CHANNEL_ORDER_BGRA
    {SEL_Y, SEL_Z, SEL_W, SEL_X},  // HSA_EXT_IMAGE_CHANNEL_ORDER_ARGB
    {SEL_Y, SEL_X, SEL_W, SEL_Z},  // HSA_EXT_IMAGE_CHANNEL_ORDER_ABGR
    {SEL_X, SEL_Y, SEL_Z, SEL_1},  // HSA_EXT_IMAGE_CHANNEL_ORDER_SRGB
    {SEL_X, SEL_Y, SEL_Z, SEL_1},  // HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBX
    {SEL_X, SEL_Y, SEL_Z, SEL_W},  // HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBA
    {SEL_Z, SEL_Y, SEL_X, SEL_W},  // HSA_EXT_IMAGE_CHANNEL_ORDER_SBGRA
    {SEL_X, SEL_X, SEL_X, SEL_X},  // HSA_EXT_IMAGE_CHANNEL_ORDER_INTENSITY
    {SEL_X, SEL_X, SEL_X, SEL_1},  // HSA_EXT_IMAGE_CHANNEL_ORDER_LUMINANCE
    {SEL_X, SEL_0, SEL_0, SEL_0},  // HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH
    {SEL_Y, SEL_0, SEL_0, SEL_0}   // HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH_STENCIL
};

const uint32_t ImageLutKv::kMaxDimensionLut_[GEOMETRY_COUNT][4] = {
    {16384, 1, 1, 1},         // HSA_EXT_IMAGE_GEOMETRY_1D
    {16384, 16384, 1, 1},     // HSA_EXT_IMAGE_GEOMETRY_2D
    {16384, 16384, 8192, 1},  // HSA_EXT_IMAGE_GEOMETRY_3D
    {16384, 1, 1, 8192},      // HSA_EXT_IMAGE_GEOMETRY_1DA
    {16384, 16384, 1, 8192},  // HSA_EXT_IMAGE_GEOMETRY_2DA
    {4294967295, 1, 1, 1},    // HSA_EXT_IMAGE_GEOMETRY_1DB
    {16384, 16384, 1, 1},     // HSA_EXT_IMAGE_GEOMETRY_2DDEPTH
    {16384, 16384, 1, 8192}   // HSA_EXT_IMAGE_GEOMETRY_2DADEPTH
};

uint32_t ImageLutKv::MapGeometry(hsa_ext_image_geometry_t geometry) const {
  switch (geometry) {
    case HSA_EXT_IMAGE_GEOMETRY_1D:
    case HSA_EXT_IMAGE_GEOMETRY_2D:
    case HSA_EXT_IMAGE_GEOMETRY_3D:
    case HSA_EXT_IMAGE_GEOMETRY_1DA:
    case HSA_EXT_IMAGE_GEOMETRY_2DA:
    case HSA_EXT_IMAGE_GEOMETRY_1DB:
    case HSA_EXT_IMAGE_GEOMETRY_2DDEPTH:
    case HSA_EXT_IMAGE_GEOMETRY_2DADEPTH:
      return kGeometryLut_[geometry];
    default:
      assert(false && "Should not reach here");
      return static_cast<uint32_t>(-1);
  };
}

ImageProperty ImageLutKv::MapFormat(const hsa_ext_image_format_t& format,
                                    hsa_ext_image_geometry_t geometry) const {
  switch (geometry) {
    case HSA_EXT_IMAGE_GEOMETRY_1D:
    case HSA_EXT_IMAGE_GEOMETRY_2D:
    case HSA_EXT_IMAGE_GEOMETRY_3D:
    case HSA_EXT_IMAGE_GEOMETRY_1DA:
    case HSA_EXT_IMAGE_GEOMETRY_2DA:
      return kPropLut_[format.channel_order][format.channel_type];
    case HSA_EXT_IMAGE_GEOMETRY_1DB:
      switch (format.channel_order) {
        // Hardware does not support buffer access to srgb image.
        case HSA_EXT_IMAGE_CHANNEL_ORDER_SRGB:
        case HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBX:
        case HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBA:
        case HSA_EXT_IMAGE_CHANNEL_ORDER_SBGRA:
          break;
        default:
          switch (format.channel_type) {
            // Hardware does not support buffer access to 555/565 packed image.
            case HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555:
            case HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565:
              break;
            default:
              return kPropLut_[format.channel_order][format.channel_type];
          }
      }
      break;
    case HSA_EXT_IMAGE_GEOMETRY_2DDEPTH:
    case HSA_EXT_IMAGE_GEOMETRY_2DADEPTH:
      switch (format.channel_order) {
        case HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH:
        case HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH_STENCIL:
          return kPropLut_[format.channel_order][format.channel_type];
        default:
          break;
      }
      break;
    default:
      assert(false && "Should not reach here");
      break;
  }

  ImageProperty prop = {0};
  return prop;
}

Swizzle ImageLutKv::MapSwizzle(hsa_ext_image_channel_order32_t order) const {
  const Swizzle invalid_swizzle = {0xff, 0xff, 0xff, 0xff};
  switch (order) {
    case HSA_EXT_IMAGE_CHANNEL_ORDER_A:
    case HSA_EXT_IMAGE_CHANNEL_ORDER_R:
    case HSA_EXT_IMAGE_CHANNEL_ORDER_RX:
    case HSA_EXT_IMAGE_CHANNEL_ORDER_RG:
    case HSA_EXT_IMAGE_CHANNEL_ORDER_RGX:
    case HSA_EXT_IMAGE_CHANNEL_ORDER_RA:
    case HSA_EXT_IMAGE_CHANNEL_ORDER_RGB:
    case HSA_EXT_IMAGE_CHANNEL_ORDER_RGBX:
    case HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA:
    case HSA_EXT_IMAGE_CHANNEL_ORDER_BGRA:
    case HSA_EXT_IMAGE_CHANNEL_ORDER_ARGB:
    case HSA_EXT_IMAGE_CHANNEL_ORDER_ABGR:
    case HSA_EXT_IMAGE_CHANNEL_ORDER_SRGB:
    case HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBX:
    case HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBA:
    case HSA_EXT_IMAGE_CHANNEL_ORDER_SBGRA:
    case HSA_EXT_IMAGE_CHANNEL_ORDER_INTENSITY:
    case HSA_EXT_IMAGE_CHANNEL_ORDER_LUMINANCE:
    case HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH:
    case HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH_STENCIL:
      return kSwizzleLut_[order];
    default:
      assert(false && "Should not reach here");
      return invalid_swizzle;
  };
}

uint32_t ImageLutKv::GetMaxWidth(hsa_ext_image_geometry_t geometry) const {
  return kMaxDimensionLut_[geometry][0];
}

uint32_t ImageLutKv::GetMaxHeight(hsa_ext_image_geometry_t geometry) const {
  return kMaxDimensionLut_[geometry][1];
}

uint32_t ImageLutKv::GetMaxDepth(hsa_ext_image_geometry_t geometry) const {
  return kMaxDimensionLut_[geometry][2];
}

uint32_t ImageLutKv::GetMaxArraySize(hsa_ext_image_geometry_t geometry) const {
  return kMaxDimensionLut_[geometry][3];
}

uint32_t ImageLutKv::GetPixelSize(uint8_t data_format, uint8_t data_type) const {
  //Currently only supports formats that ROCr can create.
  switch(data_format) {
    case FMT_1_5_5_5: return 2;
    case FMT_16: return 2;
    case FMT_16_16: return 4;
    case FMT_16_16_16_16: return 8;
    case FMT_2_10_10_10: return 4;
    //SPK: Where is unorm returning 3?  Was this a Hawaii specific thing?
    case FMT_32: return (data_type==TYPE_UNORM) ? 3 : 4;
    case FMT_32_32: return 8;
    case FMT_32_32_32_32: return 16;
    case FMT_5_6_5: return 2;
    case FMT_8: return 1;
    case FMT_8_8: return 2;
    case FMT_8_8_8_8: return 4;
    default: return 0;
  }
}

}  // namespace image
}  // namespace rocr
