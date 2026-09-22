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

#ifndef AMD_HSA_EXT_IMAGE_IMAGE_LUT_KV_H
#define AMD_HSA_EXT_IMAGE_IMAGE_LUT_KV_H

#include "image_lut.h"

namespace rocr {
namespace image {

class ImageLutKv : public ImageLut {
 public:
  ImageLutKv() {}

  virtual ~ImageLutKv() {}

  virtual uint32_t MapGeometry(hsa_ext_image_geometry_t geometry) const;

  virtual ImageProperty MapFormat(const hsa_ext_image_format_t& format,
                                  hsa_ext_image_geometry_t geometry) const;

  virtual Swizzle MapSwizzle(hsa_ext_image_channel_order32_t order) const;

  virtual uint32_t GetMaxWidth(hsa_ext_image_geometry_t geometry) const;

  virtual uint32_t GetMaxHeight(hsa_ext_image_geometry_t geometry) const;

  virtual uint32_t GetMaxDepth(hsa_ext_image_geometry_t geometry) const;

  virtual uint32_t GetMaxArraySize(hsa_ext_image_geometry_t geometry) const;

  uint32_t GetPixelSize(uint8_t data_format, uint8_t data_type) const;

 private:
  // Lookup table of image geometry to device geometry enum.
  static const uint32_t kGeometryLut_[GEOMETRY_COUNT];

  // Lookup table of channel format property. Based on HSA Programmer's
  // Reference Manual 1.0P Table 9-4 Channel Order, Channel type and Image
  // Geometry Combinations.
  static const ImageProperty kPropLut_[ORDER_COUNT][TYPE_COUNT];

  // Lookup table of channel order swizzle.
  static const Swizzle kSwizzleLut_[ORDER_COUNT];

  // Lookup table of image geometry to max dimension.
  // Each record contains four values: widht, height, depth, array_size.
  static const uint32_t kMaxDimensionLut_[GEOMETRY_COUNT][4];

  DISALLOW_COPY_AND_ASSIGN(ImageLutKv);
};

}  // namespace image
}  // namespace rocr
#endif  // AMD_HSA_EXT_IMAGE_IMAGE_LUT_KV_H
