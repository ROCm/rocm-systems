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

#include "inc/hsa_ext_amd.h"
#include "inc/hsa_ext_image.h"
#include "core/inc/hsa_ext_amd_impl.h"
#include "image_manager.h"
#include "image_runtime.h"

#include <assert.h>

#include <algorithm>
#include <climits>
#include <cmath>

#if (defined(WIN32) || defined(_WIN32))
#define NOMINMAX
__inline long int lrintf(float f) { return _mm_cvtss_si32(_mm_load_ss(&f)); }
#endif

namespace rocr {
namespace image {

Image* Image::Create(hsa_agent_t agent) {
  hsa_amd_memory_pool_t pool = ImageRuntime::instance()->kernarg_pool();

  Image* image = NULL;

  hsa_status_t status =
      AMD::hsa_amd_memory_pool_allocate(pool, sizeof(Image), 0, reinterpret_cast<void**>(&image));
  assert(status == HSA_STATUS_SUCCESS);

  if (status != HSA_STATUS_SUCCESS) return NULL;

  new (image) Image();

  status = AMD::hsa_amd_agents_allow_access(1, &agent, NULL, image);

  if (status != HSA_STATUS_SUCCESS) {
    Image::Destroy(image);
    return NULL;
  }

  return image;
}

void Image::Destroy(const Image* image) {
  assert(image != NULL);
  image->~Image();

  hsa_status_t status = AMD::hsa_amd_memory_pool_free(const_cast<Image*>(image));

  assert(status == HSA_STATUS_SUCCESS);
}

Sampler* Sampler::Create(hsa_agent_t agent) {
  hsa_amd_memory_pool_t pool = ImageRuntime::instance()->kernarg_pool();

  Sampler* sampler = NULL;

  hsa_status_t status = AMD::hsa_amd_memory_pool_allocate(pool, sizeof(Sampler), 0,
                                                          reinterpret_cast<void**>(&sampler));

  if (status != HSA_STATUS_SUCCESS) return NULL;

  new (sampler) Sampler();

  status = AMD::hsa_amd_agents_allow_access(1, &agent, NULL, sampler);

  if (status != HSA_STATUS_SUCCESS) {
    Sampler::Destroy(sampler);
    return NULL;
  }

  return sampler;
}

void Sampler::Destroy(const Sampler* sampler) {
  assert(sampler != NULL);
  sampler->~Sampler();

  hsa_status_t status = AMD::hsa_amd_memory_pool_free(const_cast<Sampler*>(sampler));

  assert(status == HSA_STATUS_SUCCESS);
}

ImageManager::ImageManager() {}

ImageManager::~ImageManager() {}

hsa_status_t ImageManager::CopyBufferToImage(
    const void* src_memory, size_t src_row_pitch, size_t src_slice_pitch,
    const Image& dst_image, const hsa_ext_image_region_t& image_region) {
  Image* src_image = Image::Create(dst_image.component);

  src_image->component = dst_image.component;
  src_image->desc = dst_image.desc;
  src_image->data = const_cast<void*>(src_memory);
  src_image->permission = HSA_ACCESS_PERMISSION_RO;
  src_image->row_pitch = src_row_pitch;
  src_image->slice_pitch = src_slice_pitch;

  const hsa_dim3_t dst_origin = image_region.offset;
  const hsa_dim3_t src_origin = {0};
  const hsa_dim3_t copy_size = image_region.range;

  hsa_status_t status = ImageManager::CopyImage(
      dst_image, *src_image, dst_origin, src_origin, copy_size);

  Image::Destroy(src_image);

  return status;
}

hsa_status_t ImageManager::CopyImageToBuffer(
    const Image& src_image, void* dst_memory, size_t dst_row_pitch,
    size_t dst_slice_pitch, const hsa_ext_image_region_t& image_region) {
  // Treat buffer as image since we don't tile our image anyway.
  Image* dst_image = Image::Create(src_image.component);

  dst_image->component = src_image.component;
  dst_image->desc = src_image.desc;  // the width, height, depth is ignored.
  dst_image->data = dst_memory;
  dst_image->permission = HSA_ACCESS_PERMISSION_WO;
  dst_image->row_pitch = dst_row_pitch;
  dst_image->slice_pitch = dst_slice_pitch;

  const hsa_dim3_t dst_origin = {0};
  const hsa_dim3_t src_origin = image_region.offset;
  const hsa_dim3_t copy_size = image_region.range;

  hsa_status_t status = ImageManager::CopyImage(
      *dst_image, src_image, dst_origin, src_origin, copy_size);

  Image::Destroy(dst_image);

  return status;
}

hsa_status_t ImageManager::CopyImage(const Image& dst_image,
                                     const Image& src_image,
                                     const hsa_dim3_t& dst_origin,
                                     const hsa_dim3_t& src_origin,
                                     const hsa_dim3_t size) {
  ImageProperty dst_image_prop = GetImageProperty(
      dst_image.component, dst_image.desc.format, dst_image.desc.geometry);
  assert(dst_image_prop.cap != HSA_EXT_IMAGE_CAPABILITY_NOT_SUPPORTED);

  const size_t dst_element_size = dst_image_prop.element_size;
  assert(dst_element_size != 0);

  ImageProperty src_image_prop = GetImageProperty(
      src_image.component, src_image.desc.format, src_image.desc.geometry);
  assert(src_image_prop.cap != HSA_EXT_IMAGE_CAPABILITY_NOT_SUPPORTED);

  const size_t src_element_size = src_image_prop.element_size;
  assert(src_element_size != 0);

  const hsa_ext_image_format_t src_format = src_image.desc.format;
  const hsa_ext_image_channel_order32_t src_order = src_format.channel_order;
  const hsa_ext_image_channel_type32_t src_type = src_format.channel_type;

  const hsa_ext_image_format_t dst_format = dst_image.desc.format;
  const hsa_ext_image_channel_order32_t dst_order = dst_format.channel_order;
  const hsa_ext_image_channel_type32_t dst_type = dst_format.channel_type;

  bool linear_to_standard_rgb = false;
  bool standard_to_linear_rgb = false;

  if ((src_order != dst_order) || (src_type != dst_type)) {
    // Source and destination format must be the same, except for
    // SRGBA <--> RGBA images.
    if ((src_type == HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT8) &&
        (dst_type == HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT8)) {
      if ((src_order == HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBA) &&
          (dst_order == HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA)) {
        standard_to_linear_rgb = true;
      } else if ((src_order == HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA) &&
                 (dst_order == HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBA)) {
        linear_to_standard_rgb = true;
      } else {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      }
    } else {
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
  }

  // Source and destination format should be the same so the element size
  // should be same too.
  const size_t element_size = src_element_size;

  // row_pitch and slice_pitch in bytes.
  const size_t dst_row_pitch =
      std::max(dst_image.row_pitch, size.x * element_size);
  const size_t dst_slice_pitch = std::max(
      dst_image.slice_pitch,
      dst_row_pitch *
          (dst_image.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_1DA ? 1 : size.y));

  const size_t src_row_pitch =
      std::max(src_image.row_pitch, size.x * element_size);
  const size_t src_slice_pitch = std::max(
      src_image.slice_pitch,
      src_row_pitch *
          (src_image.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_1DA ? 1 : size.y));

  size_t src_offset = src_origin.x;
  size_t dst_offset = dst_origin.x;
  size_t copy_size = size.x;

  // Calculate source the offset in bytes.
  src_offset *= element_size;
  src_offset += src_row_pitch * src_origin.y;
  src_offset += src_slice_pitch * src_origin.z;

  // Calculate destination the offset in bytes.
  dst_offset *= element_size;
  dst_offset += dst_row_pitch * dst_origin.y;
  dst_offset += dst_slice_pitch * dst_origin.z;

  copy_size *= element_size;

  // Get destination and source memory.
  unsigned char* dst = static_cast<unsigned char*>(dst_image.data);
  const unsigned char* src = static_cast<const unsigned char*>(src_image.data);

  if (!linear_to_standard_rgb && !standard_to_linear_rgb) {
    // Copy the memory by row.
    for (size_t slice = 0; slice < size.z; ++slice) {
      size_t src_offset_temp = src_offset + slice * src_slice_pitch;
      size_t dst_offset_temp = dst_offset + slice * dst_slice_pitch;

      for (size_t rows = 0; rows < size.y; ++rows) {
        std::memcpy((dst + dst_offset_temp), (src + src_offset_temp),
                    copy_size);
        src_offset_temp += src_row_pitch;
        dst_offset_temp += dst_row_pitch;
      }
    }
  } else {
    // Copy per pixel between RGBA-SRGBA images.
    for (size_t slice = 0; slice < size.z; ++slice) {
      size_t src_offset_temp = src_offset + slice * src_slice_pitch;
      size_t dst_offset_temp = dst_offset + slice * dst_slice_pitch;

      for (size_t rows = 0; rows < size.y; ++rows) {
        const uint8_t* src_pixel = src + src_offset_temp;
        uint8_t* dst_pixel = dst + dst_offset_temp;

        if (linear_to_standard_rgb) {
          for (size_t cols = 0; cols < size.x; ++cols) {
            dst_pixel[0] =
                Denormalize(LinearToStandardRGB(Normalize(src_pixel[0])));  // R
            dst_pixel[1] =
                Denormalize(LinearToStandardRGB(Normalize(src_pixel[1])));  // G
            dst_pixel[2] =
                Denormalize(LinearToStandardRGB(Normalize(src_pixel[2])));  // B
            dst_pixel[3] = src_pixel[3];                                    // A

            src_pixel += element_size;
            dst_pixel += element_size;
          }
        } else {
          assert(standard_to_linear_rgb);
          for (size_t cols = 0; cols < size.x; ++cols) {
            dst_pixel[0] =
                Denormalize(StandardToLinearRGB(Normalize(src_pixel[0])));  // R
            dst_pixel[1] =
                Denormalize(StandardToLinearRGB(Normalize(src_pixel[1])));  // G
            dst_pixel[2] =
                Denormalize(StandardToLinearRGB(Normalize(src_pixel[2])));  // B
            dst_pixel[3] = src_pixel[3];                                    // A

            src_pixel += element_size;
            dst_pixel += element_size;
          }
        }

        src_offset_temp += src_row_pitch;
        dst_offset_temp += dst_row_pitch;
      }
    }
  }

  return HSA_STATUS_SUCCESS;
}

uint16_t ImageManager::FloatToHalf(float in) {
  volatile union {
    float f;
    uint32_t u;
  } fu;

  fu.f = in;

  const uint16_t sign_bit_16 = (fu.u >> 16) & 0x8000;

  const uint32_t exp_32 = (fu.u >> 23) & 0xff;

  const uint32_t mantissa_32 = (fu.u) & 0x7fffff;

  if (exp_32 == 0 && mantissa_32 == 0) {
    // Zero.
    return sign_bit_16;
  } else if (exp_32 == 0xff) {
    if (mantissa_32 == 0) {
      // Inf.
      return (sign_bit_16 | 0x7c00);
    } else if ((mantissa_32 & 0x400000)) {
      // Quiet NaN.
      return (sign_bit_16 | 0x7e00);
    } else {
      // Signal NaN.
      return (sign_bit_16 | 0x7c01);
    }
  } else {
    const uint32_t kMaxExpNormal = 0x477fe000 >> 23;     // 65504.
    const uint32_t kMinExpNormal = 0x38800000 >> 23;     // 2^-14;
    const uint32_t kMinExpSubnormal = 0x33800000 >> 23;  // 2^-24.
    if (exp_32 > kMaxExpNormal) {
      // Half overflow.
      // TODO: clamp it to max half float or +Inf.
      return (sign_bit_16 | 0x7bff);
    } else if (exp_32 < kMinExpSubnormal) {
      // Half underflow.
      return (sign_bit_16);
    } else if (exp_32 < kMinExpNormal) {
      // Half subnormal.
      return (sign_bit_16 |
              ((0x0400 | (mantissa_32 >> 13)) >> (127 - exp_32 - 14)));
    } else {
      // Half normal.
      return (sign_bit_16 |
              (((exp_32 - 127 + 15) << 10) | (mantissa_32 >> 13)));
    }
  }
}

float ImageManager::Normalize(uint8_t u_val) {
  if (u_val == 0) {
    return 0.0f;
  } else if (u_val == UINT8_MAX) {
    return 1.0f;
  } else {
    return std::min(
        std::max(static_cast<float>(u_val) / static_cast<float>(UINT8_MAX),
                 0.0f),
        1.0f);
  }
}

uint8_t ImageManager::Denormalize(float f_val) {
  const unsigned long kScale = UINT8_MAX;
  return std::min(
      static_cast<unsigned long>(std::max(lrintf(kScale * f_val), 0l)), kScale);
}

float ImageManager::StandardToLinearRGB(float s_val) {
  // Map SRGB value to RGB color space based on HSA Programmers Reference
  // Manual version 1.0 Provisional, chapter 7.1.4.1.2  Standard RGB (s-Form).
  double l_val = (double)s_val;

  l_val = (l_val <= 0.04045f) ? (l_val / 12.92f)
                              : pow(((l_val + 0.055f) / 1.055f), 2.4f);

  return l_val;
}

float ImageManager::LinearToStandardRGB(float l_val) {
  // Map RGB value to SRGB color space based on HSA Programmers Reference
  // Manual version 1.0 Provisional, chapter 7.1.4.1.2  Standard RGB (s-Form).
  double s_val = (double)l_val;

#if (defined(WIN32) || defined(_WIN32))
  if (_isnan(s_val)) s_val = 0.0;
#else
  if (std::isnan(s_val)) s_val = 0.0;
#endif

  if (s_val > 1.0) {
    s_val = 1.0;
  } else if (s_val < 0.0) {
    s_val = 0.0;
  } else if (s_val < 0.0031308) {
    s_val = 12.92 * s_val;
  } else {
    s_val = (1.055 * pow(s_val, 5.0 / 12.0)) - 0.055;
  }

  return s_val;
}

void ImageManager::FormatPattern(const hsa_ext_image_format_t& format,
                                 const void* pattern_in, void* pattern_out) {
  const int kR = 0;
  const int kG = 1;
  const int kB = 2;
  const int kA = 3;

  int index[4] = {0};
  int num_channel = 0;

  switch (format.channel_order) {
    case HSA_EXT_IMAGE_CHANNEL_ORDER_A:
      index[0] = kA;
      num_channel = 1;
      break;
    case HSA_EXT_IMAGE_CHANNEL_ORDER_R:
    case HSA_EXT_IMAGE_CHANNEL_ORDER_RX:
      index[0] = kR;
      num_channel = 1;
      break;
    case HSA_EXT_IMAGE_CHANNEL_ORDER_RG:
    case HSA_EXT_IMAGE_CHANNEL_ORDER_RGX:
      index[0] = kR;
      index[1] = kG;
      num_channel = 2;
      break;
    case HSA_EXT_IMAGE_CHANNEL_ORDER_RA:
      index[0] = kR;
      index[1] = kA;
      num_channel = 2;
      break;
    case HSA_EXT_IMAGE_CHANNEL_ORDER_RGB:
    case HSA_EXT_IMAGE_CHANNEL_ORDER_RGBX:
    case HSA_EXT_IMAGE_CHANNEL_ORDER_SRGB:
    case HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBX:
      index[0] = kR;
      index[1] = kG;
      index[2] = kB;
      num_channel = 3;
      break;
    case HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA:
    case HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBA:
      index[0] = kR;
      index[1] = kG;
      index[2] = kB;
      index[3] = kA;
      num_channel = 4;
      break;
    case HSA_EXT_IMAGE_CHANNEL_ORDER_BGRA:
    case HSA_EXT_IMAGE_CHANNEL_ORDER_SBGRA:
      index[0] = kB;
      index[1] = kG;
      index[2] = kR;
      index[3] = kA;
      num_channel = 4;
      break;
    case HSA_EXT_IMAGE_CHANNEL_ORDER_ARGB:
      index[0] = kA;
      index[1] = kR;
      index[2] = kG;
      index[3] = kB;
      num_channel = 4;
      break;
    case HSA_EXT_IMAGE_CHANNEL_ORDER_ABGR:
      index[0] = kA;
      index[1] = kB;
      index[2] = kG;
      index[3] = kR;
      num_channel = 4;
      break;
    case HSA_EXT_IMAGE_CHANNEL_ORDER_INTENSITY:
    case HSA_EXT_IMAGE_CHANNEL_ORDER_LUMINANCE:
    case HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH:
    case HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH_STENCIL:
      index[0] = kR;
      num_channel = 1;
      break;
    default:
      assert(false && "Should not reach here.");
      break;
  }

  const float* pattern_in_f = NULL;
  const int32_t* pattern_in_i32 = NULL;
  const uint32_t* pattern_in_ui32 = NULL;

  float new_pattern_in_f[4] = { 0 };
  if ((format.channel_order == HSA_EXT_IMAGE_CHANNEL_ORDER_SRGB) ||
      (format.channel_order == HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBX) ||
      (format.channel_order == HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBA) ||
      (format.channel_order == HSA_EXT_IMAGE_CHANNEL_ORDER_SBGRA)) {
    pattern_in_f = reinterpret_cast<const float*>(pattern_in);

    new_pattern_in_f[0] = LinearToStandardRGB(pattern_in_f[0]);
    new_pattern_in_f[1] = LinearToStandardRGB(pattern_in_f[1]);
    new_pattern_in_f[2] = LinearToStandardRGB(pattern_in_f[2]);
    new_pattern_in_f[3] = pattern_in_f[3];

    pattern_in_f = reinterpret_cast<const float*>(new_pattern_in_f);
  } else {
    pattern_in_f = reinterpret_cast<const float*>(pattern_in);
    pattern_in_i32 = reinterpret_cast<const int32_t*>(pattern_in);
    pattern_in_ui32 = reinterpret_cast<const uint32_t*>(pattern_in);
  }

  for (int c = 0; c < num_channel; ++c) {
    switch (format.channel_type) {
      case HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT8: {
        int8_t* pattern_out_i8 = reinterpret_cast<int8_t*>(pattern_out);
        const long kScale = INT8_MAX;
        const long conv = lrintf(kScale * pattern_in_f[index[c]]);
        pattern_out_i8[c] = std::min(std::max(conv, -kScale - 1l), kScale);
      } break;
      case HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT16: {
        int16_t* pattern_out_i16 = reinterpret_cast<int16_t*>(pattern_out);
        const long kScale = INT16_MAX;
        const long conv = lrintf(kScale * pattern_in_f[index[c]]);
        pattern_out_i16[c] = std::min(std::max(conv, -kScale - 1l), kScale);
      } break;
      case HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT8: {
        uint8_t* pattern_out_ui8 = reinterpret_cast<uint8_t*>(pattern_out);
        const unsigned long kScale = UINT8_MAX;
        const long conv = lrintf(kScale * pattern_in_f[index[c]]);
        pattern_out_ui8[c] =
            std::min(static_cast<unsigned long>(std::max(conv, 0l)), kScale);
      } break;
      case HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT16: {
        uint16_t* pattern_out_ui16 = reinterpret_cast<uint16_t*>(pattern_out);
        const unsigned long kScale = UINT16_MAX;
        const long conv = lrintf(kScale * pattern_in_f[index[c]]);
        pattern_out_ui16[c] =
            std::min(static_cast<unsigned long>(std::max(conv, 0l)), kScale);
      } break;
      case HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT24: {
        typedef struct Order24 { uint32_t r : 24; } Order24;

        Order24* pattern_out_u24 = reinterpret_cast<Order24*>(pattern_out);
        const unsigned long kScale = 0xffffff;
        const long conv = lrintf(kScale * pattern_in_f[index[c]]);
        pattern_out_u24[c].r =
            std::min(static_cast<unsigned long>(std::max(conv, 0l)), kScale);
      } break;
      case HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555: {
        typedef struct Order555 {
          uint32_t b : 5;
          uint32_t g : 5;
          uint32_t r : 5;
        } Order555;

        Order555* pattern_out_u555 = reinterpret_cast<Order555*>(pattern_out);
        const unsigned long kScale = 0x1f;
        long conv = lrintf(kScale * pattern_in_f[index[0]]);
        pattern_out_u555->r =
            std::min(static_cast<unsigned long>(std::max(conv, 0l)), kScale);

        conv = lrintf(kScale * pattern_in_f[index[1]]);
        pattern_out_u555->g =
            std::min(static_cast<unsigned long>(std::max(conv, 0l)), kScale);

        conv = lrintf(kScale * pattern_in_f[index[2]]);
        pattern_out_u555->b =
            std::min(static_cast<unsigned long>(std::max(conv, 0l)), kScale);
        return;
      } break;
      case HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565: {
        typedef struct Order565 {
          uint32_t b : 5;
          uint32_t g : 6;
          uint32_t r : 5;
        } Order565;

        Order565* pattern_out_u565 = reinterpret_cast<Order565*>(pattern_out);
        unsigned long scale = 0x1f;
        long conv = lrintf(scale * pattern_in_f[index[0]]);
        pattern_out_u565->r =
            std::min(static_cast<unsigned long>(std::max(conv, 0l)), scale);

        scale = 0x3f;
        conv = lrintf(scale * pattern_in_f[index[1]]);
        pattern_out_u565->g =
            std::min(static_cast<unsigned long>(std::max(conv, 0l)), scale);

        scale = 0x1f;
        conv = lrintf(scale * pattern_in_f[index[2]]);
        pattern_out_u565->b =
            std::min(static_cast<unsigned long>(std::max(conv, 0l)), scale);
        return;
      } break;
      case HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_101010: {
        typedef struct Order101010 {
          uint32_t b : 10;
          uint32_t g : 10;
          uint32_t r : 10;
        } Order101010;

        Order101010* pattern_out_u101010 =
            reinterpret_cast<Order101010*>(pattern_out);
        const unsigned long kScale = 0x3ff;
        long conv = lrintf(kScale * pattern_in_f[index[0]]);
        pattern_out_u101010->r =
            std::min(static_cast<unsigned long>(std::max(conv, 0l)), kScale);

        conv = lrintf(kScale * pattern_in_f[index[1]]);
        pattern_out_u101010->g =
            std::min(static_cast<unsigned long>(std::max(conv, 0l)), kScale);

        conv = lrintf(kScale * pattern_in_f[index[2]]);
        pattern_out_u101010->b =
            std::min(static_cast<unsigned long>(std::max(conv, 0l)), kScale);

        return;
      } break;
      case HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT8: {
        int8_t* pattern_out_i8 = reinterpret_cast<int8_t*>(pattern_out);
        pattern_out_i8[c] = pattern_in_i32[index[c]];
      } break;
      case HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT16: {
        int16_t* pattern_out_i16 = reinterpret_cast<int16_t*>(pattern_out);
        pattern_out_i16[c] = pattern_in_i32[index[c]];
      } break;
      case HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT32: {
        int32_t* pattern_out_i32 = reinterpret_cast<int32_t*>(pattern_out);
        pattern_out_i32[c] = pattern_in_i32[index[c]];
      } break;
      case HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8: {
        uint8_t* pattern_out_ui8 = reinterpret_cast<uint8_t*>(pattern_out);
        pattern_out_ui8[c] = pattern_in_ui32[index[c]];
      } break;
      case HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16: {
        uint16_t* pattern_out_ui16 = reinterpret_cast<uint16_t*>(pattern_out);
        pattern_out_ui16[c] = pattern_in_ui32[index[c]];
      } break;
      case HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32: {
        uint32_t* pattern_out_ui32 = reinterpret_cast<uint32_t*>(pattern_out);
        pattern_out_ui32[c] = pattern_in_ui32[index[c]];
      } break;
      case HSA_EXT_IMAGE_CHANNEL_TYPE_HALF_FLOAT: {
        // TODO: convert to f16
        uint16_t* pattern_out_ui16 = reinterpret_cast<uint16_t*>(pattern_out);
        pattern_out_ui16[c] = FloatToHalf(pattern_in_f[index[c]]);
        break;
      }
      case HSA_EXT_IMAGE_CHANNEL_TYPE_FLOAT: {
        float* pattern_out_f = reinterpret_cast<float*>(pattern_out);
        pattern_out_f[c] = pattern_in_f[index[c]];
      } break;
      default:
        assert(false && "Should not reach here.");
        break;
    }
  }
}

hsa_status_t ImageManager::FillImage(const Image& image, const void* pattern,
                                     const hsa_ext_image_region_t& region) {
  const hsa_dim3_t origin = region.offset;
  const hsa_dim3_t size = region.range;

  ImageProperty image_prop =
      GetImageProperty(image.component, image.desc.format, image.desc.geometry);
  assert(image_prop.cap != HSA_EXT_IMAGE_CAPABILITY_NOT_SUPPORTED);

  const size_t element_size = image_prop.element_size;
  assert(element_size != 0);

  const size_t row_pitch = image.row_pitch;
  const size_t slice_pitch = image.slice_pitch;

  // Map memory.
  unsigned char* fill_mem = static_cast<unsigned char*>(image.data);

  char fill_value[4 * sizeof(int)] = {0};
  FormatPattern(image.desc.format, pattern, fill_value);

  // Calculate offset.
  size_t offset = origin.x * element_size;
  offset += row_pitch * origin.y;
  offset += slice_pitch * origin.z;

  // Fill the image memory with the pattern.
  for (size_t slice = 0; slice < size.z; ++slice) {
    size_t offset_temp = offset + slice * slice_pitch;

    for (size_t rows = 0; rows < size.y; ++rows) {
      size_t pix_offset = offset_temp;

      // Copy pattern per pixel.
      for (size_t column = 0; column < size.x; ++column) {
        memcpy((fill_mem + pix_offset), fill_value, element_size);
        pix_offset += element_size;
      }

      offset_temp += row_pitch;
    }
  }

  return HSA_STATUS_SUCCESS;
}

}  // namespace image
}  // namespace rocr
