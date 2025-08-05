////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2022, Advanced Micro Devices, Inc. All rights reserved.
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

#include "image_lut_gfx11.h"
#include "resource_gfx11.h"

namespace rocr {
namespace image {

  /* 
   * The type table has changed for gfx11, so we need a separate instance for
   * the Property LUT
   */
  const ImageProperty ImageLutGfx11::kPropLutGfx11_[ORDER_COUNT][TYPE_COUNT] = {
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

ImageProperty ImageLutGfx11::MapFormat(const hsa_ext_image_format_t& format,
                                    hsa_ext_image_geometry_t geometry) const {
  switch (geometry) {
    case HSA_EXT_IMAGE_GEOMETRY_1D:
    case HSA_EXT_IMAGE_GEOMETRY_2D:
    case HSA_EXT_IMAGE_GEOMETRY_3D:
    case HSA_EXT_IMAGE_GEOMETRY_1DA:
    case HSA_EXT_IMAGE_GEOMETRY_2DA:
      return kPropLutGfx11_[format.channel_order][format.channel_type];
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
              return kPropLutGfx11_[format.channel_order][format.channel_type];
          }
      }
      break;
    case HSA_EXT_IMAGE_GEOMETRY_2DDEPTH:
    case HSA_EXT_IMAGE_GEOMETRY_2DADEPTH:
      switch (format.channel_order) {
        case HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH:
        case HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH_STENCIL:
          return kPropLutGfx11_[format.channel_order][format.channel_type];
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

}  // namespace image
}  // namespace rocr
