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

#ifndef HSA_RUNTIME_EXT_IMAGE_IMAGE_MANAGER_AI_H
#define HSA_RUNTIME_EXT_IMAGE_IMAGE_MANAGER_AI_H

#include "addrlib/inc/addrinterface.h"
#include "image_manager_kv.h"

namespace rocr {
namespace image {

class ImageManagerAi : public ImageManagerKv {
 public:
  explicit ImageManagerAi();
  virtual ~ImageManagerAi();

  /// @brief Calculate the size and alignment of the backing storage of an
  /// image.
  virtual hsa_status_t CalculateImageSizeAndAlignment(
      hsa_agent_t component, const hsa_ext_image_descriptor_t& desc,
      hsa_ext_image_data_layout_t image_data_layout,
      size_t image_data_row_pitch, size_t image_data_slice_pitch,
      hsa_ext_image_data_info_t& image_info) const;

  /// @brief Fill image structure with device specific image object.
  virtual hsa_status_t PopulateImageSrd(Image& image) const;

  /// @brief Fill image structure with device specific image object using the given format.
  virtual hsa_status_t PopulateImageSrd(Image& image, const metadata_amd_t* desc) const;

  /// @brief Modify device specific image object according to the specified
  /// new format.
  virtual hsa_status_t ModifyImageSrd(Image& image,
                                      hsa_ext_image_format_t& new_format) const;

  /// @brief Fill sampler structure with device specific sampler object.
  virtual hsa_status_t PopulateSamplerSrd(Sampler& sampler) const;

 protected:
  uint32_t GetAddrlibSurfaceInfoAi(hsa_agent_t component,
                             const hsa_ext_image_descriptor_t& desc,
                             Image::TileMode tileMode,
                             size_t image_data_row_pitch,
                             size_t image_data_slice_pitch,
                             ADDR2_COMPUTE_SURFACE_INFO_OUTPUT& out) const;

  bool IsLocalMemory(const void* address) const;

 private:
  DISALLOW_COPY_AND_ASSIGN(ImageManagerAi);
};

}  // namespace image
}  // namespace rocr
#endif  // HSA_RUNTIME_EXT_IMAGE_IMAGE_MANAGER_AI_H
