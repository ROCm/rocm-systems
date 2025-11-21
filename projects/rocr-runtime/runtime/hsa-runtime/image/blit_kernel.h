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

#ifndef HSA_RUNTIME_EXT_IMAGE_BLIT_KERNEL_H
#define HSA_RUNTIME_EXT_IMAGE_BLIT_KERNEL_H
#include <assert.h>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "inc/hsa.h"
#include "resource.h"

namespace rocr {
namespace image {

typedef struct BlitQueue {
  hsa_queue_t* queue_;
  volatile std::atomic<uint64_t> cached_index_;
} BlitQueue;

typedef struct BlitCodeInfo {
  uint64_t code_handle_;
  uint32_t group_segment_size_;
  uint32_t private_segment_size_;
} BlitCodeInfo;

class BlitKernel {
 public:
  typedef enum KernelOp {
    KERNEL_OP_COPY_IMAGE_TO_BUFFER = 0,
    KERNEL_OP_COPY_BUFFER_TO_IMAGE = 1,
    KERNEL_OP_COPY_IMAGE_DEFAULT = 2,
    KERNEL_OP_COPY_IMAGE_LINEAR_TO_STANDARD = 3,
    KERNEL_OP_COPY_IMAGE_STANDARD_TO_LINEAR = 4,
    KERNEL_OP_COPY_IMAGE_1DB = 5,
    KERNEL_OP_COPY_IMAGE_1DB_TO_REG = 6,
    KERNEL_OP_COPY_IMAGE_REG_TO_1DB = 7,
    KERNEL_OP_CLEAR_IMAGE = 8,
    KERNEL_OP_CLEAR_IMAGE_1DB = 9,
    KERNEL_OP_COUNT = 10
  } KernelOp;

  explicit BlitKernel();
  ~BlitKernel();

  hsa_status_t Initialize();

  hsa_status_t Cleanup();

  hsa_status_t BuildBlitCode(hsa_agent_t agent,
                             std::vector<BlitCodeInfo>& blit_code_catalog);

  hsa_status_t CopyBufferToImage(
      BlitQueue& blit_queue,
      const std::vector<BlitCodeInfo>& blit_code_catalog,
      const void* src_memory, size_t src_row_pitch, size_t src_slice_pitch,
      const Image& dst_image, const hsa_ext_image_region_t& image_region);

  hsa_status_t CopyImageToBuffer(
      BlitQueue& blit_queue,
      const std::vector<BlitCodeInfo>& blit_code_catalog,
      const Image& src_image, void* dst_memory, size_t dst_row_pitch,
      size_t dst_slice_pitch, const hsa_ext_image_region_t& image_region);

  hsa_status_t CopyImage(BlitQueue& blit_queue,
                         const std::vector<BlitCodeInfo>& blit_code_catalog,
                         const Image& dst_image, const Image& src_image,
                         const hsa_dim3_t& dst_origin,
                         const hsa_dim3_t& src_origin, const hsa_dim3_t size,
                         KernelOp copy_type);

  hsa_status_t FillImage(BlitQueue& blit_queue,
                         const std::vector<BlitCodeInfo>& blit_code_catalog,
                         const Image& image, const void* pattern,
                         const hsa_ext_image_region_t& region);

 private:

  hsa_status_t PopulateKernelCode(
      hsa_agent_t agent, hsa_executable_t executable,
      std::vector<BlitCodeInfo>& blit_code_catalog);

  inline void CalcBufferRowSlicePitchesInPixel(
      hsa_ext_image_geometry_t geometry, uint32_t element_size,
      const hsa_dim3_t& copy_size, size_t in_row_pitch_byte,
      size_t in_slice_pitch_byte, unsigned long* out_pitch_pixel);

  inline uint32_t GetDimSize(const Image& image);

  inline uint32_t GetNumChannel(const Image& image);

  inline uint32_t GetImageAccessType(const Image& image);

  void CalcWorkingSize(const Image& image, const hsa_dim3_t& range,
                       hsa_kernel_dispatch_packet_t& packet);

  void CalcWorkingSize(const Image& src_image, const Image& dst_image,
                       const hsa_dim3_t& range,
                       hsa_kernel_dispatch_packet_t& packet);

  hsa_status_t ConvertImage(const Image& original_image,
                            const Image** new_image);

  hsa_status_t LaunchKernel(BlitQueue& queue,
                            hsa_kernel_dispatch_packet_t& packet);

  // The kernels' name.
  static const char* kernel_name_[KERNEL_OP_COUNT];
  static const char* ocl_kernel_name_[KERNEL_OP_COUNT];

  // Mapping of ISA and kernel object.
  std::unordered_map<uint64_t, hsa_code_object_t> code_object_map_;

  // Mapping of ISA and kernel executable.
  std::unordered_map<uint64_t, hsa_executable_t> code_executable_map_;

  std::mutex lock_;

  DISALLOW_COPY_AND_ASSIGN(BlitKernel);

  // Get the patched code object
  hsa_status_t GetPatchedBlitObject(const char* agent_name, uint8_t** code_object_handle);
};

}  // namespace image
}  // namespace rocr

#endif  // HSA_RUNTIME_EXT_IMAGE_BLIT_KERNEL_H
