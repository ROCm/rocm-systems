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

#ifndef HSA_RUNTIME_EXT_IMAGE_IMAGE_MANAGER_KV_H
#define HSA_RUNTIME_EXT_IMAGE_IMAGE_MANAGER_KV_H

#include "addrlib/inc/addrinterface.h"
#include "blit_kernel.h"
#include "image_lut_kv.h"
#include "image_manager.h"

namespace rocr {
namespace image {

class ImageManagerKv : public ImageManager {
 public:
  explicit ImageManagerKv();
  virtual ~ImageManagerKv();

  virtual hsa_status_t Initialize(hsa_agent_t agent_handle);

  virtual void Cleanup();

  /// @brief Retrieve device specific image property of a certain format
  /// and geometry.
  virtual ImageProperty GetImageProperty(
      hsa_agent_t component, const hsa_ext_image_format_t& format,
      hsa_ext_image_geometry_t geometry) const;

  /// @brief Retrieve device specific supported max width, height, depth,
  /// and array size of an image geometry.
  virtual void GetImageInfoMaxDimension(hsa_agent_t component,
                                        hsa_ext_image_geometry_t geometry,
                                        uint32_t& width, uint32_t& height,
                                        uint32_t& depth,
                                        uint32_t& array_size) const;

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

  // @brief Copy the content of a linear memory to an image object.
  virtual hsa_status_t CopyBufferToImage(
      const void* src_memory, size_t src_row_pitch, size_t src_slice_pitch,
      const Image& dst_image, const hsa_ext_image_region_t& image_region);

  /// @brief Copy the content of an image object to a linear memory.
  virtual hsa_status_t CopyImageToBuffer(
      const Image& src_image, void* dst_memory, size_t dst_row_pitch,
      size_t dst_slice_pitch, const hsa_ext_image_region_t& image_region);

  /// @brief Transfer images backing storage using agent copy.
  virtual hsa_status_t CopyImage(const Image& dst_image, const Image& src_image,
                                 const hsa_dim3_t& dst_origin,
                                 const hsa_dim3_t& src_origin,
                                 const hsa_dim3_t size);

  /// @brief Fill image backing storage using agent copy.
  virtual hsa_status_t FillImage(const Image& image, const void* pattern,
                                 const hsa_ext_image_region_t& region);

 protected:
  static hsa_status_t GetLocalMemoryRegion(hsa_region_t region, void* data);

  static AddrFormat GetAddrlibFormat(const ImageProperty& image_prop);

  static VOID* ADDR_API AllocSysMem(const ADDR_ALLOCSYSMEM_INPUT* input);

  static ADDR_E_RETURNCODE ADDR_API
      FreeSysMem(const ADDR_FREESYSMEM_INPUT* input);

  bool GetAddrlibSurfaceInfo(hsa_agent_t component,
                             const hsa_ext_image_descriptor_t& desc,
                             Image::TileMode tileMode,
                             size_t image_data_row_pitch,
                             size_t image_data_slice_pitch,
                             ADDR_COMPUTE_SURFACE_INFO_OUTPUT& out) const;

  size_t CalWorkingSizeBytes(hsa_ext_image_geometry_t geometry,
                             hsa_dim3_t size_pixel,
                             uint32_t element_size) const;

  virtual bool IsLocalMemory(const void* address) const;

  BlitQueue& BlitQueueInit();

  virtual const ImageLutKv& ImageLut() const { return image_lut_; };

  ADDR_HANDLE addr_lib_;

  hsa_agent_t agent_;

  uint32_t family_type_;

  uint32_t chip_id_;

  BlitQueue blit_queue_;

  std::vector<BlitCodeInfo> blit_code_catalog_;

  uint32_t mtype_;

  uintptr_t local_memory_base_address_;

  std::mutex lock_;

 private:
  ImageLutKv image_lut_;
  DISALLOW_COPY_AND_ASSIGN(ImageManagerKv);
};

}  // namespace image
}  // namespace rocr
#endif  // HSA_RUNTIME_EXT_IMAGE_IMAGE_MANAGER_KV_H
