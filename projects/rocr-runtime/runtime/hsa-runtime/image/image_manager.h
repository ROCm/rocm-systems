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

#ifndef AMD_HSA_EXT_IMAGE_IMAGE_MANAGER_H
#define AMD_HSA_EXT_IMAGE_IMAGE_MANAGER_H

#include <cstring>
#include "inc/hsa.h"
#include "inc/hsa_ext_image.h"
#include "resource.h"
#include "util.h"
#include "image/addrlib/inc/addrinterface.h"

// Forward declaration for BuildMipmapSrd(); full definition in hsakmt/hsakmttypes.h (included by the
// image manager .cpp files). Used only by pointer here, so the declaration suffices.
typedef struct _HsaWddmSurfaceMetadata HsaWddmSurfaceMetadata;

namespace rocr {
namespace image {

/// @brief Abstract class for creating AMD agent specific image / sampler
/// resources and data transfer.
class ImageManager {
 public:
  explicit ImageManager();
  virtual ~ImageManager();

  virtual hsa_status_t Initialize(hsa_agent_t agent_handle) = 0;

  virtual void Cleanup() = 0;

  /// @brief Retrieve device specific image property of a certain format
  /// and geometry.
  virtual ImageProperty GetImageProperty(
      hsa_agent_t component, const hsa_ext_image_format_t& format,
      hsa_ext_image_geometry_t geometry) const = 0;

  /// @brief Retrieve device specific supported max width, height, depth,
  /// and array size of an image geometry.
  virtual void GetImageInfoMaxDimension(hsa_agent_t component,
                                        hsa_ext_image_geometry_t geometry,
                                        uint32_t& width, uint32_t& height,
                                        uint32_t& depth,
                                        uint32_t& array_size) const = 0;

  /// @brief Calculate the size and alignment of the backing storage of an
  /// image.
  virtual hsa_status_t CalculateImageSizeAndAlignment(
      hsa_agent_t component, const hsa_ext_image_descriptor_t& desc,
      hsa_ext_image_data_layout_t image_data_layout,
      uint32_t num_mipmap_levels,
      size_t image_data_row_pitch,
      size_t image_data_slice_pitch,
      hsa_ext_image_data_info_t& image_info) const = 0;

  /// @brief Fill image structure with device specific image object.
  virtual hsa_status_t PopulateImageSrd(Image& image) const = 0;

  /// @brief Fill image structure with device specific image object using the given format.
  virtual hsa_status_t PopulateImageSrd(Image& image, const metadata_amd_t* desc) const = 0;

  /// @brief Modify device specific image object according to the specified
  /// new format.
  virtual hsa_status_t ModifyImageSrd(
      Image& image, hsa_ext_image_format_t& new_format) const = 0;

  /// @brief Fill sampler structure with device specific sampler object.
  virtual hsa_status_t PopulateSamplerSrd(Sampler& sampler) const = 0;

  // @brief Copy the content of a linear memory to an image object.
  virtual hsa_status_t CopyBufferToImage(
      const void* src_memory, size_t src_row_pitch, size_t src_slice_pitch,
      const Image& dst_image, const hsa_ext_image_region_t& image_region);

  /// @brief Copy the content of an image object to a linear memory.
  virtual hsa_status_t CopyImageToBuffer(
      const Image& src_image, void* dst_memory, size_t dst_row_pitch,
      size_t dst_slice_pitch, const hsa_ext_image_region_t& image_region);

  /// @brief Transfer images backing storage.
  virtual hsa_status_t CopyImage(const Image& dst_image, const Image& src_image,
                                 const hsa_dim3_t& dst_origin,
                                 const hsa_dim3_t& src_origin,
                                 const hsa_dim3_t size);

  /// @brief Fill image backing storage using host copy.
  virtual hsa_status_t FillImage(const Image& image, const void* pattern,
                                 const hsa_ext_image_region_t& region);

  /// @brief Get the address library handle
  virtual ADDR_HANDLE GetAddrLib() const = 0;

  /// @brief Fill mipmap structure with device specific mipmapped array object.
  virtual hsa_status_t PopulateMipmapSrd(MipmappedArray& mipmap_array) const = 0;

  /// @brief Fill mipmap structure with pre-computed AMD metadata descriptor.
  virtual hsa_status_t PopulateMipmapSrd(MipmappedArray& mipmap_array, const metadata_amd_t* desc) const = 0;

  /// @brief Create mip level view using SRD BASE_LEVEL/LAST_LEVEL fields
  virtual hsa_status_t PopulateMipLevelSrd(MipmappedArray& level_view,
      const MipmappedArray& mipmap_array, uint32_t mip_level) const = 0;

  virtual void printSRDDetailed(const uint32_t* srd) const = 0;
  virtual void printChannelSelect(uint32_t sel) const = 0;
  virtual void printResourceType(uint32_t type) const = 0;
  virtual void printSwizzleMode(uint32_t sw_mode) const = 0;

 protected:
  static uint16_t FloatToHalf(float in);

  static inline float Normalize(uint8_t u_val);

  static inline uint8_t Denormalize(float f_val);

  static float StandardToLinearRGB(float s_val);

  static float LinearToStandardRGB(float l_val);

  static void FormatPattern(const hsa_ext_image_format_t& format,
                            const void* pattern_in, void* pattern_out);

  template <typename dstT, typename srcT>
  static inline hsa_status_t convertAddressMode(dstT &word,
                            const hsa_ext_sampler_addressing_mode32_t address_mode[3]) {
    srcT clamp[3];
    for (int i = 0; i < 3; i++) {
      switch (address_mode[i]) {
        case HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE:
          clamp[i] = srcT::SQ_TEX_CLAMP_LAST_TEXEL;
          break;
        case HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_BORDER:
          clamp[i] = srcT::SQ_TEX_CLAMP_BORDER;
          break;
        case HSA_EXT_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT:
          clamp[i] = srcT::SQ_TEX_MIRROR;
          break;
        case HSA_EXT_SAMPLER_ADDRESSING_MODE_UNDEFINED:
        case HSA_EXT_SAMPLER_ADDRESSING_MODE_REPEAT:
          clamp[i] = srcT::SQ_TEX_WRAP;
          break;
        default:
          return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      }
    }
    word.bits.CLAMP_X = static_cast<unsigned int>(clamp[0]);
    word.bits.CLAMP_Y = static_cast<unsigned int>(clamp[1]);
    word.bits.CLAMP_Z = static_cast<unsigned int>(clamp[2]);
    return HSA_STATUS_SUCCESS;
  }

  /// @brief Values of the 3-bit SQ_IMG_SAMP_WORD0.MAX_ANISO_RATIO field, which
  /// holds log2 of the ratio. Mirrors the hardware SQ_TEX_ANISO_RATIO enum,
  /// which is not reachable from here - the generated register headers in this
  /// directory define SQ_TEX_XY_FILTER and friends but not this one.
  enum SqTexAnisoRatio {
    kSqTexAnisoRatio1 = 0,
    kSqTexAnisoRatio2 = 1,
    kSqTexAnisoRatio4 = 2,
    kSqTexAnisoRatio8 = 3,
    kSqTexAnisoRatio16 = 4,
  };

  /// @brief Determine the appropriate anisotropic filtering ratio.
  ///
  /// Mirrors PAL's GetAnisoRatio():
  ///  - for values of anisotropy not natively supported by the hardware we
  ///    clamp to the closest value less than what was requested (3 -> 2:1,
  ///    7 -> 4:1, ...);
  ///  - 16 is the maximum supported ratio, and anything above it saturates
  ///    there rather than being rejected or falling back to isotropic.
  ///
  /// Anisotropy applies to both point and linear sampling - the hardware has
  /// SQ_TEX_XY_FILTER_ANISO_POINT as well as _ANISO_BILINEAR, matching PAL's
  /// IsAnisoEnabled() which treats XyFilterAnisotropicPoint as aniso-enabled.
  static inline uint32_t encodeMaxAnisoRatio(uint32_t max_anisotropy) {
    uint32_t aniso_ratio = kSqTexAnisoRatio1;

    if (max_anisotropy < 2) {
      // Nothing to do.
    } else if (max_anisotropy < 4) {
      aniso_ratio = kSqTexAnisoRatio2;
    } else if (max_anisotropy < 8) {
      aniso_ratio = kSqTexAnisoRatio4;
    } else if (max_anisotropy < 16) {
      aniso_ratio = kSqTexAnisoRatio8;
    } else {
      aniso_ratio = kSqTexAnisoRatio16;
    }

    return aniso_ratio;
  }

  /// @brief SQ_IMG_SAMP_WORD0.ANISO_THRESHOLD, as PAL's !preciseAniso path
  /// computes it: encoded ratio >> 1. Zero when anisotropy is off.
  static inline uint32_t encodeAnisoThreshold(uint32_t max_anisotropy) {
    return encodeMaxAnisoRatio(max_anisotropy) >> 1;
  }

  /// @brief SQ_IMG_SAMP_WORD0.ANISO_BIAS - PAL uses the encoded ratio itself.
  /// PAL also sets PERF_MIP to (ratio + 6); not mirrored, because it is nonzero
  /// even at ratio 0 and would change every existing isotropic SRD.
  static inline uint32_t encodeAnisoBias(uint32_t max_anisotropy) {
    return encodeMaxAnisoRatio(max_anisotropy);
  }

  /// @brief True when anisotropic filtering is in effect, in which case
  /// XY_MIN/MAG_FILTER must select an SQ_TEX_XY_FILTER_ANISO_* variant for the
  /// hardware to apply MAX_ANISO_RATIO at all. Keeping this in step with
  /// encodeMaxAnisoRatio() is what makes the programmed filter and ratio agree,
  /// the same invariant PAL's IsAnisoEnabled()/GetAnisoRatio() pair maintains.
  static inline bool useAnisoFilter(
      const hsa_ext_sampler_descriptor_v3_t& desc) {
    return encodeMaxAnisoRatio(desc.max_anisotropy) != kSqTexAnisoRatio1;
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(ImageManager);
};

}  // namespace image
}  // namespace rocr
#endif  // AMD_HSA_EXT_IMAGE_IMAGE_MANAGER_H
