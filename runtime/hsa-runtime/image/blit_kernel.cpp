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

#include "blit_kernel.h"

#if (defined(WIN32) || defined(_WIN32))
#define NOMINMAX
#endif

#include <algorithm>
#include <atomic>
#include <sstream>
#include <string>

#include "image_manager.h"
#include "image_runtime.h"
#include "util.h"

#undef HSA_ARGUMENT_ALIGN_BYTES
#define HSA_ARGUMENT_ALIGN_BYTES 16

#include "core/inc/hsa_internal.h"
#include "core/inc/hsa_ext_amd_impl.h"
#include "core/inc/hsa_table_interface.h"

namespace rocr {
namespace image {

extern uint8_t blit_object_gfx7xx[14608];
extern uint8_t blit_object_gfx8xx[15424];
extern uint8_t blit_object_gfx9xx[15432];

extern uint8_t ocl_blit_object_gfx700[];
extern uint8_t ocl_blit_object_gfx701[];
extern uint8_t ocl_blit_object_gfx702[];
extern uint8_t ocl_blit_object_gfx801[];
extern uint8_t ocl_blit_object_gfx802[];
extern uint8_t ocl_blit_object_gfx803[];
extern uint8_t ocl_blit_object_gfx805[];
extern uint8_t ocl_blit_object_gfx810[];
extern uint8_t ocl_blit_object_gfx900[];
extern uint8_t ocl_blit_object_gfx902[];
extern uint8_t ocl_blit_object_gfx904[];
extern uint8_t ocl_blit_object_gfx906[];
extern uint8_t ocl_blit_object_gfx908[];
extern uint8_t ocl_blit_object_gfx909[];
extern uint8_t ocl_blit_object_gfx90a[];
extern uint8_t ocl_blit_object_gfx90c[];
extern uint8_t ocl_blit_object_gfx942[];
extern uint8_t ocl_blit_object_gfx950[];
extern uint8_t ocl_blit_object_gfx1010[];
extern uint8_t ocl_blit_object_gfx1011[];
extern uint8_t ocl_blit_object_gfx1012[];
extern uint8_t ocl_blit_object_gfx1013[];
extern uint8_t ocl_blit_object_gfx1030[];
extern uint8_t ocl_blit_object_gfx1031[];
extern uint8_t ocl_blit_object_gfx1032[];
extern uint8_t ocl_blit_object_gfx1033[];
extern uint8_t ocl_blit_object_gfx1034[];
extern uint8_t ocl_blit_object_gfx1035[];
extern uint8_t ocl_blit_object_gfx1036[];
extern uint8_t ocl_blit_object_gfx1100[];
extern uint8_t ocl_blit_object_gfx1101[];
extern uint8_t ocl_blit_object_gfx1102[];
extern uint8_t ocl_blit_object_gfx1103[];
extern uint8_t ocl_blit_object_gfx1150[];
extern uint8_t ocl_blit_object_gfx1151[];
extern uint8_t ocl_blit_object_gfx1152[];
extern uint8_t ocl_blit_object_gfx1153[];
extern uint8_t ocl_blit_object_gfx1200[];
extern uint8_t ocl_blit_object_gfx1201[];

// Arguments inserted by OCL compiler, all zero here.
struct OCLHiddenArgs {
  uint64_t offset_x;
  uint64_t offset_y;
  uint64_t offset_z;
  void* printf_buffer;
  void* enqueue;
  void* enqueue2;
  void* multi_grid;
};

static void* Allocate(hsa_agent_t agent, size_t size) {
  //use the host accessible kernarg pool
  hsa_amd_memory_pool_t pool = ImageRuntime::instance()->kernarg_pool();

  void* ptr = NULL;

  hsa_status_t status = AMD::hsa_amd_memory_pool_allocate(pool, size, 0, &ptr);
  assert(status == HSA_STATUS_SUCCESS);

  if (status != HSA_STATUS_SUCCESS) return NULL;

  status = AMD::hsa_amd_agents_allow_access(1, &agent, NULL, ptr);
  assert(status == HSA_STATUS_SUCCESS);

  if (status != HSA_STATUS_SUCCESS) {
    AMD::hsa_amd_memory_pool_free(ptr);
    return NULL;
  }
  return ptr;
}

BlitKernel::BlitKernel() {
}

BlitKernel::~BlitKernel() {}

hsa_status_t BlitKernel::Initialize() { return HSA_STATUS_SUCCESS; }

hsa_status_t BlitKernel::Cleanup() {

  for (std::pair<const uint64_t, hsa_executable_t> pair :
       code_executable_map_) {
    HSA::hsa_executable_destroy(pair.second);
  }

  code_executable_map_.clear();

  code_object_map_.clear();

  return HSA_STATUS_SUCCESS;
}

hsa_status_t BlitKernel::BuildBlitCode(
    hsa_agent_t agent, std::vector<BlitCodeInfo>& blit_code_catalog) {
  // Find existing kernels in the list that have compatible ISA.
  hsa_isa_t agent_isa = {0};
  hsa_status_t status = HSA::hsa_agent_get_info(agent, HSA_AGENT_INFO_ISA, &agent_isa);
  if (HSA_STATUS_SUCCESS != status) {
    return status;
  }

  std::lock_guard<std::mutex> lock(lock_);

  for (std::pair<uint64_t, hsa_executable_t> pair : code_executable_map_) {
    bool isa_compatible = false;
    hsa_isa_t code_isa = {pair.first};

    status = HSA::hsa_isa_compatible(code_isa, agent_isa, &isa_compatible);
    if (HSA_STATUS_SUCCESS != status) {
      return status;
    }

    if (isa_compatible) {
      return PopulateKernelCode(agent, pair.second, blit_code_catalog);
    }
  }

  // No existing compatible kernels. Build new kernels.
  hsa_code_object_t code_object = {0};

  // Get the target name
  char agent_name[64] = {0};
  status = HSA::hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, &agent_name);
  if (HSA_STATUS_SUCCESS != status) {
    return status;
  }

  // Get the patched code object
  uint8_t* patched_code_object;
  status = BlitKernel::GetPatchedBlitObject(agent_name, &patched_code_object);
  if (HSA_STATUS_SUCCESS != status) {
    return status;
  }

  // Pass the patched code object
  code_object.handle = reinterpret_cast<uint64_t>(patched_code_object);

  code_object_map_[agent_isa.handle] = code_object;

  // Create executable.
  hsa_executable_t executable = {0};
  status =
      HSA::hsa_executable_create(HSA_PROFILE_FULL, HSA_EXECUTABLE_STATE_UNFROZEN, "", &executable);
  if (HSA_STATUS_SUCCESS != status) {
    return status;
  }

  code_executable_map_[agent_isa.handle] = executable;

  // Load code object.
  status = HSA::hsa_executable_load_code_object(executable, agent, code_object, "");
  if (HSA_STATUS_SUCCESS != status) {
    return status;
  }

  // Freeze executable.
  status = HSA::hsa_executable_freeze(executable, "");
  if (HSA_STATUS_SUCCESS != status) {
    return status;
  }

  return PopulateKernelCode(agent, executable, blit_code_catalog);
}

hsa_status_t BlitKernel::CopyBufferToImage(
    BlitQueue& blit_queue, const std::vector<BlitCodeInfo>& blit_code_catalog,
    const void* src_memory, size_t src_row_pitch, size_t src_slice_pitch,
    const Image& dst_image, const hsa_ext_image_region_t& image_region) {
  if (dst_image.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_1DB) {
    ImageManager* manager = ImageRuntime::instance()->image_manager(dst_image.component);

    const uint32_t element_size =
        manager->GetImageProperty(dst_image.component, dst_image.desc.format,
                                  dst_image.desc.geometry).element_size;

    const size_t dst_origin = image_region.offset.x * element_size;
    char* dst_memory = reinterpret_cast<char*>(dst_image.data) + dst_origin;
    const size_t size = image_region.range.x * element_size;

    return HSA::hsa_memory_copy(dst_memory, src_memory, size);
  }

  const Image* dst_image_view = NULL;

  hsa_status_t status = ConvertImage(dst_image, &dst_image_view);
  if (HSA_STATUS_SUCCESS != status) {
    return status;
  }

  assert(dst_image_view != NULL);

  hsa_kernel_dispatch_packet_t packet = { };

  const BlitCodeInfo& blit_code =
      blit_code_catalog.at(KERNEL_OP_COPY_BUFFER_TO_IMAGE);
  packet.kernel_object = blit_code.code_handle_;
  packet.group_segment_size = blit_code.group_segment_size_;
  packet.private_segment_size = blit_code.private_segment_size_;

  // Setup kernel argument.
  /*
  buffer is start of output pixel in destination buffer
  format.x is element count
  format.y is element size
  format.z is max(dword per pixel, 1)
  format.w is texture type.
  pixelOrigin is start pixel address.
  */
  struct KernelArgs {
    const void* buffer;
    uint64_t image[5];
    int32_t pixelOrigin[4];
    uint32_t format[4];
    uint64_t pitch;
    uint64_t slice_pitch;
    OCLHiddenArgs ocl;
  };

  KernelArgs* args = (KernelArgs*)Allocate(dst_image_view->component, sizeof(KernelArgs));
  assert(args != NULL);
  memset(args, 0, sizeof(KernelArgs));
  args->buffer = src_memory;
  for(auto& img : args->image)
    img = dst_image_view->Convert();
  args->pixelOrigin[0] = image_region.offset.x;
  args->pixelOrigin[1] = image_region.offset.y;
  args->pixelOrigin[2] = image_region.offset.z;

  ImageManager* manager = ImageRuntime::instance()->image_manager(dst_image_view->component);

  const uint32_t element_size =
      manager->GetImageProperty(dst_image_view->component,
                                dst_image_view->desc.format,
                                dst_image_view->desc.geometry).element_size;

  // Try to minimize the read operation to buffer by reading the buffer
  // up to one DWORD at a time.
  uint32_t buffer_read_count = element_size / sizeof(uint32_t);
  buffer_read_count = (buffer_read_count == 0) ? 1 : buffer_read_count;

  const uint32_t num_channel = GetNumChannel(*dst_image_view);
  const uint32_t size_per_channel = element_size / num_channel;

  args->format[0] = num_channel;
  args->format[1] = size_per_channel;
  args->format[2] = buffer_read_count;
  args->format[3] = dst_image_view->desc.geometry;

  unsigned long buffer_pitch[2] = {0, 0};
  CalcBufferRowSlicePitchesInPixel(dst_image_view->desc.geometry, element_size,
                                   image_region.range, src_row_pitch,
                                   src_slice_pitch, buffer_pitch);

  args->pitch = buffer_pitch[0];
  args->slice_pitch = buffer_pitch[1];

  packet.kernarg_address = args;

  // Setup packet dimension and working size.
  CalcWorkingSize(*dst_image_view, image_region.range, packet);

  status = LaunchKernel(blit_queue, packet);

  if (&dst_image != dst_image_view) {
    Image::Destroy(dst_image_view);
  }
  AMD::hsa_amd_memory_pool_free(args);

  return status;
}

hsa_status_t BlitKernel::CopyImageToBuffer(
    BlitQueue& blit_queue, const std::vector<BlitCodeInfo>& blit_code_catalog,
    const Image& src_image, void* dst_memory, size_t dst_row_pitch,
    size_t dst_slice_pitch, const hsa_ext_image_region_t& image_region) {
  if (src_image.desc.geometry == HSA_EXT_IMAGE_GEOMETRY_1DB) {
    ImageManager* manager = ImageRuntime::instance()->image_manager(src_image.component);

    const uint32_t element_size =
        manager->GetImageProperty(src_image.component, src_image.desc.format,
                                  src_image.desc.geometry).element_size;

    const size_t src_origin = image_region.offset.x * element_size;
    const char* src_memory =
        reinterpret_cast<const char*>(src_image.data) + src_origin;
    const size_t size = image_region.range.x * element_size;

    return HSA::hsa_memory_copy(dst_memory, src_memory, size);
  }

  const Image* src_image_view = NULL;

  hsa_status_t status = ConvertImage(src_image, &src_image_view);
  if (HSA_STATUS_SUCCESS != status) {
    return status;
  }

  assert(src_image_view != NULL);

  hsa_kernel_dispatch_packet_t packet = { };

  const BlitCodeInfo& blit_code =
      blit_code_catalog.at(KERNEL_OP_COPY_IMAGE_TO_BUFFER);
  packet.kernel_object = blit_code.code_handle_;
  packet.group_segment_size = blit_code.group_segment_size_;
  packet.private_segment_size = blit_code.private_segment_size_;

  // Setup kernel argument.
  /*
  buffer is start of output pixel in destination buffer
  format.x is element count
  format.y is element size
  format.z is max(dword per pixel, 1)
  format.w is texture type.
  pixelOrigin is start pixel address.
  */
  struct KernelArgs {
    uint64_t image[5];
    void* buffer;
    int32_t pixelOrigin[4];
    uint32_t format[4];
    uint64_t pitch;
    uint64_t slice_pitch;
    OCLHiddenArgs ocl;
  };

  KernelArgs* args = (KernelArgs*)Allocate(src_image_view->component, sizeof(KernelArgs));
  assert(args != NULL);
  memset(args, 0, sizeof(KernelArgs));
  for(auto &img : args->image)
    img = src_image_view->Convert();
  args->buffer = dst_memory;
  args->pixelOrigin[0] = image_region.offset.x;
  args->pixelOrigin[1] = image_region.offset.y;
  args->pixelOrigin[2] = image_region.offset.z;

  ImageManager* manager = ImageRuntime::instance()->image_manager(src_image_view->component);

  const uint32_t element_size =
      manager->GetImageProperty(src_image_view->component,
                                src_image_view->desc.format,
                                src_image_view->desc.geometry).element_size;

  // Try to minimize the write operation to buffer by reading the buffer
  // up to one DWORD at a time.
  uint32_t buffer_write_count = element_size / sizeof(uint32_t);
  buffer_write_count = (buffer_write_count == 0) ? 1 : buffer_write_count;

  const uint32_t num_channel = GetNumChannel(*src_image_view);
  const uint32_t size_per_channel = element_size / num_channel;

  args->format[0] = num_channel;
  args->format[1] = size_per_channel;
  args->format[2] = buffer_write_count;
  args->format[3] = src_image_view->desc.geometry;

  unsigned long buffer_pitch[2] = {0, 0};
  CalcBufferRowSlicePitchesInPixel(src_image_view->desc.geometry, element_size,
                                   image_region.range, dst_row_pitch,
                                   dst_slice_pitch, buffer_pitch);

  args->pitch = buffer_pitch[0];
  args->slice_pitch = buffer_pitch[1];

  packet.kernarg_address = args;

  // Setup packet dimension and working size.
  CalcWorkingSize(*src_image_view, image_region.range, packet);

  status = LaunchKernel(blit_queue, packet);

  if (&src_image != src_image_view) {
    Image::Destroy(src_image_view);
  }
  AMD::hsa_amd_memory_pool_free(args);

  return status;
}

hsa_status_t BlitKernel::CopyImage(
    BlitQueue& blit_queue, const std::vector<BlitCodeInfo>& blit_code_catalog,
    const Image& dst_image, const Image& src_image,
    const hsa_dim3_t& dst_origin, const hsa_dim3_t& src_origin,
    const hsa_dim3_t size, KernelOp copy_type) {
  assert(src_image.component.handle == dst_image.component.handle);

  const Image* src_image_view = &src_image;
  const Image* dst_image_view = &dst_image;
  const BlitCodeInfo* blit_code = NULL;

  if (copy_type == KERNEL_OP_COPY_IMAGE_DEFAULT) {
    // Linear to linear image copy.

    hsa_status_t status = ConvertImage(src_image, &src_image_view);
    if (HSA_STATUS_SUCCESS != status) {
      return status;
    }

    assert(src_image_view != NULL);

    status = ConvertImage(dst_image, &dst_image_view);
    if (HSA_STATUS_SUCCESS != status) {
      return status;
    }

    assert(dst_image_view != NULL);

    const hsa_ext_image_geometry_t src_geometry = src_image_view->desc.geometry;
    const hsa_ext_image_geometry_t dst_geometry = dst_image_view->desc.geometry;

    if (src_geometry != HSA_EXT_IMAGE_GEOMETRY_1DB &&
        dst_geometry != HSA_EXT_IMAGE_GEOMETRY_1DB) {
      blit_code = &blit_code_catalog.at(KERNEL_OP_COPY_IMAGE_DEFAULT);
    } else if (src_geometry == HSA_EXT_IMAGE_GEOMETRY_1DB &&
               dst_geometry != HSA_EXT_IMAGE_GEOMETRY_1DB) {
      blit_code = &blit_code_catalog.at(KERNEL_OP_COPY_IMAGE_1DB_TO_REG);
    } else if (src_geometry != HSA_EXT_IMAGE_GEOMETRY_1DB &&
               dst_geometry == HSA_EXT_IMAGE_GEOMETRY_1DB) {
      blit_code = &blit_code_catalog.at(KERNEL_OP_COPY_IMAGE_REG_TO_1DB);
    } else {
      blit_code = &blit_code_catalog.at(KERNEL_OP_COPY_IMAGE_1DB);
    }
  } else {
    blit_code = &blit_code_catalog.at(copy_type);
  }

  hsa_kernel_dispatch_packet_t packet = { };

  packet.kernel_object = blit_code->code_handle_;
  packet.group_segment_size = blit_code->group_segment_size_;
  packet.private_segment_size = blit_code->private_segment_size_;

  // Setup kernel argument.
  struct KernelArgs {
    uint64_t src[5];
    uint64_t dst[5];
    int32_t srcOrigin[4];
    int32_t dstOrigin[4];
    int32_t srcFormat;
    int32_t dstFormat;
    OCLHiddenArgs ocl;
  };

  KernelArgs* args = (KernelArgs*)Allocate(dst_image_view->component, sizeof(KernelArgs));
  assert(args != NULL);
  memset(args, 0, sizeof(KernelArgs));

  for(auto& img : args->src)
    img = src_image_view->Convert();
  args->srcFormat = src_image_view->desc.geometry;
  args->srcOrigin[0] = src_origin.x;
  args->srcOrigin[1] = src_origin.y;
  args->srcOrigin[2] = src_origin.z;

  for(auto& img : args->dst)
    img = dst_image_view->Convert();
  args->dstFormat = dst_image_view->desc.geometry;
  args->dstOrigin[0] = dst_origin.x;
  args->dstOrigin[1] = dst_origin.y;
  args->dstOrigin[2] = dst_origin.z;

  packet.kernarg_address = args;

  // Setup packet dimension and working size.
  CalcWorkingSize(*src_image_view, *dst_image_view, size, packet);

  hsa_status_t status = LaunchKernel(blit_queue, packet);

  if (&src_image != src_image_view) {
    Image::Destroy(src_image_view);
  }

  if (&dst_image != dst_image_view) {
    Image::Destroy(dst_image_view);
  }

  AMD::hsa_amd_memory_pool_free(args);

  return status;
}

hsa_status_t BlitKernel::FillImage(
    BlitQueue& blit_queue, const std::vector<BlitCodeInfo>& blit_code_catalog,
    const Image& image, const void* pattern,
    const hsa_ext_image_region_t& region) {
  hsa_kernel_dispatch_packet_t packet = { };

  const BlitCodeInfo& blit_code =
      (image.desc.geometry != HSA_EXT_IMAGE_GEOMETRY_1DB)
          ? blit_code_catalog.at(KERNEL_OP_CLEAR_IMAGE)
          : blit_code_catalog.at(KERNEL_OP_CLEAR_IMAGE_1DB);
  packet.kernel_object = blit_code.code_handle_;
  packet.group_segment_size = blit_code.group_segment_size_;
  packet.private_segment_size = blit_code.private_segment_size_;

  // Setup kernel argument.
  struct KernelArgs {
    uint64_t image[5];
    int32_t format;
    uint32_t type;
    uint32_t data[4];
    int32_t origin[4];
    OCLHiddenArgs ocl;
  };

  KernelArgs* args = (KernelArgs*)Allocate(image.component, sizeof(KernelArgs));
  assert(args != NULL);
  memset(args, 0, sizeof(KernelArgs));

  for(auto &img : args->image)
    img = image.Convert();
  args->format = image.desc.geometry;
  for(int i=0; i<4; i++)
    args->data[i] = ((const uint32_t*)pattern)[i];
  args->origin[0] = region.offset.x;
  args->origin[1] = region.offset.y;
  args->origin[2] = region.offset.z;
  args->type = GetImageAccessType(image);

  packet.kernarg_address = args;

  // Setup packet dimension and working size.
  CalcWorkingSize(image, region.range, packet);

  hsa_status_t status = LaunchKernel(blit_queue, packet);

  AMD::hsa_amd_memory_pool_free(args);

  return status;
}

const char *BlitKernel::kernel_name_[KERNEL_OP_COUNT] = {
      "&__copy_image_to_buffer_kernel",
      "&__copy_buffer_to_image_kernel",
      "&__copy_image_default_kernel",
      "&__copy_image_linear_to_standard_kernel",
      "&__copy_image_standard_to_linear_kernel",
      "&__copy_image_1db_kernel",
      "&__copy_image_1db_to_reg_kernel",
      "&__copy_image_reg_to_1db_kernel",
      "&__clear_image_kernel",
      "&__clear_image_1db_kernel"};

const char *BlitKernel::ocl_kernel_name_[KERNEL_OP_COUNT] = {
      "copy_image_to_buffer.kd",
      "copy_buffer_to_image.kd",
      "copy_image_default.kd",
      "copy_image_linear_to_standard.kd",
      "copy_image_standard_to_linear.kd",
      "copy_image_1db.kd",
      "copy_image_1db_to_reg.kd",
      "copy_image_reg_to_1db.kd",
      "clear_image.kd",
      "clear_image_1db.kd"};

hsa_status_t BlitKernel::PopulateKernelCode(
    hsa_agent_t agent, hsa_executable_t executable,
    std::vector<BlitCodeInfo>& blit_code_catalog) {
  blit_code_catalog.clear();

  for (int i = 0; i < KERNEL_OP_COUNT; ++i) {
    // Get symbol handle.
    hsa_executable_symbol_t kernel_symbol = {0};

    hsa_status_t status = HSA::hsa_executable_get_symbol_by_name(executable, ocl_kernel_name_[i],
                                                                 &agent, &kernel_symbol);
    if (HSA_STATUS_SUCCESS != status) {
      blit_code_catalog.clear();
      return status;
    }

    // Get code handle.
    BlitCodeInfo blit_code = {0};
    status = HSA::hsa_executable_symbol_get_info(
        kernel_symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &blit_code.code_handle_);
    if (HSA_STATUS_SUCCESS != status) {
      blit_code_catalog.clear();
      return status;
    }

    status = HSA::hsa_executable_symbol_get_info(
        kernel_symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
        &blit_code.group_segment_size_);
    if (HSA_STATUS_SUCCESS != status) {
      blit_code_catalog.clear();
      return status;
    }

    status = HSA::hsa_executable_symbol_get_info(
        kernel_symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
        &blit_code.private_segment_size_);
    if (HSA_STATUS_SUCCESS != status) {
      blit_code_catalog.clear();
      return status;
    }

    blit_code_catalog.push_back(blit_code);
  }

  assert(blit_code_catalog.size() == KERNEL_OP_COUNT);
  return HSA_STATUS_SUCCESS;
}

void BlitKernel::CalcBufferRowSlicePitchesInPixel(
    hsa_ext_image_geometry_t geometry, uint32_t element_size,
    const hsa_dim3_t& copy_size, size_t in_row_pitch_byte,
    size_t in_slice_pitch_byte, unsigned long* out_pitch_pixel) {
  const bool is_1d_array =
      (geometry == HSA_EXT_IMAGE_GEOMETRY_1DA) ? true : false;

  out_pitch_pixel[0] =
      std::max(static_cast<unsigned long>(copy_size.x),
               static_cast<unsigned long>(in_row_pitch_byte / element_size));

  out_pitch_pixel[1] =
      (is_1d_array)
          ? out_pitch_pixel[0]
          : (std::max(
                static_cast<unsigned long>(out_pitch_pixel[0] * copy_size.y),
                static_cast<unsigned long>(in_slice_pitch_byte /
                                           element_size)));

  assert((out_pitch_pixel[0] <= out_pitch_pixel[1]));
}

uint32_t BlitKernel::GetDimSize(const Image& image) {
  static const uint32_t kDimSizeTable[] = {
      1,  // HSA_EXT_IMAGE_GEOMETRY_1D
      2,  // HSA_EXT_IMAGE_GEOMETRY_2D
      3,  // HSA_EXT_IMAGE_GEOMETRY_3D
      2,  // HSA_EXT_IMAGE_GEOMETRY_1DA
      3,  // HSA_EXT_IMAGE_GEOMETRY_2DA
      1,  // HSA_EXT_IMAGE_GEOMETRY_1DB
      2,  // HSA_EXT_IMAGE_GEOMETRY_2DDEPTH
      3,  // HSA_EXT_IMAGE_GEOMETRY_2DADEPTH
  };

  return kDimSizeTable[image.desc.geometry];
}

uint32_t BlitKernel::GetNumChannel(const Image& image) {
  static const uint32_t kNumChannelTable[] = {
      1,  // HSA_EXT_IMAGE_CHANNEL_ORDER_A,
      1,  // HSA_EXT_IMAGE_CHANNEL_ORDER_R,
      1,  // HSA_EXT_IMAGE_CHANNEL_ORDER_RX,
      2,  // HSA_EXT_IMAGE_CHANNEL_ORDER_RG,
      2,  // HSA_EXT_IMAGE_CHANNEL_ORDER_RGX,
      2,  // HSA_EXT_IMAGE_CHANNEL_ORDER_RA,
      3,  // HSA_EXT_IMAGE_CHANNEL_ORDER_RGB,
      3,  // HSA_EXT_IMAGE_CHANNEL_ORDER_RGBX,
      4,  // HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA,
      4,  // HSA_EXT_IMAGE_CHANNEL_ORDER_BGRA,
      4,  // HSA_EXT_IMAGE_CHANNEL_ORDER_ARGB,
      4,  // HSA_EXT_IMAGE_CHANNEL_ORDER_ABGR,
      3,  // HSA_EXT_IMAGE_CHANNEL_ORDER_SRGB,
      3,  // HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBX,
      4,  // HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBA,
      4,  // HSA_EXT_IMAGE_CHANNEL_ORDER_SBGRA,
      1,  // HSA_EXT_IMAGE_CHANNEL_ORDER_INTENSITY,
      1,  // HSA_EXT_IMAGE_CHANNEL_ORDER_LUMINANCE,
      1,  // HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH,
      1,  // HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH_STENCIL
  };

  return kNumChannelTable[image.desc.format.channel_order];
}

uint32_t BlitKernel::GetImageAccessType(const Image& image) {
  enum AccessType {
    ACCESS_TYPE_F = 0,
    ACCESS_TYPE_I = 1,
    ACCESS_TYPE_UI = 2,
  };

  static const uint32_t kAccessType[] = {
      ACCESS_TYPE_F,   // HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT8
      ACCESS_TYPE_F,   // HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT16
      ACCESS_TYPE_F,   // HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT8
      ACCESS_TYPE_F,   // HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT16
      ACCESS_TYPE_F,   // HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT24
      ACCESS_TYPE_F,   // HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555
      ACCESS_TYPE_F,   // HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565
      ACCESS_TYPE_F,   // HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_101010
      ACCESS_TYPE_I,   // HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT8
      ACCESS_TYPE_I,   // HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT16
      ACCESS_TYPE_I,   // HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT32
      ACCESS_TYPE_UI,  // HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8
      ACCESS_TYPE_UI,  // HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16
      ACCESS_TYPE_UI,  // HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32
      ACCESS_TYPE_F,   // HSA_EXT_IMAGE_CHANNEL_TYPE_HALF_FLOAT
      ACCESS_TYPE_F    // HSA_EXT_IMAGE_CHANNEL_TYPE_FLOAT
  };

  return kAccessType[image.desc.format.channel_type];
}

void BlitKernel::CalcWorkingSize(const Image& image, const hsa_dim3_t& range,
                                 hsa_kernel_dispatch_packet_t& packet) {
  switch (image.desc.geometry) {
    case HSA_EXT_IMAGE_GEOMETRY_1D:
    case HSA_EXT_IMAGE_GEOMETRY_1DB:
    case HSA_EXT_IMAGE_GEOMETRY_1DA:
      packet.setup = 2;
      packet.grid_size_x = range.x;
      packet.grid_size_y = range.y;
      packet.grid_size_z = 1;
      packet.workgroup_size_x = 64;
      packet.workgroup_size_y = packet.workgroup_size_z = 1;
      break;
    case HSA_EXT_IMAGE_GEOMETRY_2D:
    case HSA_EXT_IMAGE_GEOMETRY_2DDEPTH:
    case HSA_EXT_IMAGE_GEOMETRY_2DADEPTH:
    case HSA_EXT_IMAGE_GEOMETRY_2DA:
      packet.setup = 3;
      packet.grid_size_x = range.x;
      packet.grid_size_y = range.y;
      packet.grid_size_z = range.z;
      packet.workgroup_size_x = packet.workgroup_size_y = 8;
      packet.workgroup_size_z = 1;
      break;
    case HSA_EXT_IMAGE_GEOMETRY_3D:
      packet.setup = 3;
      packet.grid_size_x = range.x;
      packet.grid_size_y = range.y;
      packet.grid_size_z = range.z;
      packet.workgroup_size_x = packet.workgroup_size_y = 4;
      packet.workgroup_size_z = 4;
      break;
  }
}

void BlitKernel::CalcWorkingSize(const Image& src_image, const Image& dst_image,
                                 const hsa_dim3_t& range,
                                 hsa_kernel_dispatch_packet_t& packet) {
  if (GetDimSize(src_image) < GetDimSize(dst_image)) {
    CalcWorkingSize(src_image, range, packet);
  } else {
    CalcWorkingSize(dst_image, range, packet);
  }
}

hsa_status_t BlitKernel::ConvertImage(const Image& original_image,
                                      const Image** new_image) {
  // To simplify the kernel, some particular image channel types are converted
  // to a new channel type, while preserving the actual per pixel size.
  // E.g.: a UNORM SIGNED INT8 is converted into UNSIGNED INT8. This way the
  // kernel can just use read_imageui on all images.

  static const uint32_t kTypeConvertTable[] = {
      HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8,  // HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT8
      HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16,  // HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT16
      HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8,  // HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT8
      HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16,  // HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT16
      HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT24,  // HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT24
      HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16,  // HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555
      HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16,  // HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565
      HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32,  // HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_101010
      HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8,  // HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT8
      HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16,  // HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT16
      HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32,  // HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT32
      HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8,  // HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8
      HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16,  // HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16
      HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32,  // HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32
      HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16,  // HSA_EXT_IMAGE_CHANNEL_TYPE_HALF_FLOAT
      HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32  // HSA_EXT_IMAGE_CHANNEL_TYPE_FLOAT
  };

  // To simplify the kernel, some particular image channel orders are converted
  // to a new channel order, while preserving the actual per pixel size.
  // E.g.: a CHANNEL ORDER A is converted into CHANNEL ORDER R. This way the
  // kernel can just read the first components of vector4 on all images.
  static const uint32_t kOrderConvertTable[] = {
      HSA_EXT_IMAGE_CHANNEL_ORDER_R,     // HSA_EXT_IMAGE_CHANNEL_ORDER_A
      HSA_EXT_IMAGE_CHANNEL_ORDER_R,     // HSA_EXT_IMAGE_CHANNEL_ORDER_R
      HSA_EXT_IMAGE_CHANNEL_ORDER_R,     // HSA_EXT_IMAGE_CHANNEL_ORDER_RX
      HSA_EXT_IMAGE_CHANNEL_ORDER_RG,    // HSA_EXT_IMAGE_CHANNEL_ORDER_RG
      HSA_EXT_IMAGE_CHANNEL_ORDER_RG,    // HSA_EXT_IMAGE_CHANNEL_ORDER_RGX
      HSA_EXT_IMAGE_CHANNEL_ORDER_RG,    // HSA_EXT_IMAGE_CHANNEL_ORDER_RA
      HSA_EXT_IMAGE_CHANNEL_ORDER_RGB,   // HSA_EXT_IMAGE_CHANNEL_ORDER_RGB
      HSA_EXT_IMAGE_CHANNEL_ORDER_RGB,   // HSA_EXT_IMAGE_CHANNEL_ORDER_RGBX
      HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA,  // HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA
      HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA,  // HSA_EXT_IMAGE_CHANNEL_ORDER_BGRA
      HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA,  // HSA_EXT_IMAGE_CHANNEL_ORDER_ARGB
      HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA,  // HSA_EXT_IMAGE_CHANNEL_ORDER_ABGR
      HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA,  // HSA_EXT_IMAGE_CHANNEL_ORDER_SRGB
      HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA,  // HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBX
      HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA,  // HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBA
      HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA,  // HSA_EXT_IMAGE_CHANNEL_ORDER_SBGRA
      HSA_EXT_IMAGE_CHANNEL_ORDER_R,  // HSA_EXT_IMAGE_CHANNEL_ORDER_INTENSITY
      HSA_EXT_IMAGE_CHANNEL_ORDER_R,  // HSA_EXT_IMAGE_CHANNEL_ORDER_LUMINANCE
      HSA_EXT_IMAGE_CHANNEL_ORDER_R,  // HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH
      HSA_EXT_IMAGE_CHANNEL_ORDER_RG  // HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH_STENCIL
  };

  const uint32_t current_type = original_image.desc.format.channel_type;
  uint32_t converted_type = kTypeConvertTable[current_type];
  const uint32_t current_order = original_image.desc.format.channel_order;
  uint32_t converted_order = kOrderConvertTable[current_order];

  if ((current_type == converted_type) && (current_order == converted_order)) {
    *new_image = &original_image;
    return HSA_STATUS_SUCCESS;
  }

  // Handle formats that drop channels on conversion, only usable with RGB(X)
  if((current_type == HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555) ||
     (current_type == HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565) ||
     (current_type == HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_101010)) {
    converted_order = HSA_EXT_IMAGE_CHANNEL_ORDER_R;
  }

  // For internal book keeping, depth isn't a HW type.
  const hsa_ext_image_geometry_t current_geometry =
      original_image.desc.geometry;
  hsa_ext_image_geometry_t converted_geometry = current_geometry;
  if (converted_geometry == HSA_EXT_IMAGE_GEOMETRY_2DDEPTH) {
    converted_geometry = HSA_EXT_IMAGE_GEOMETRY_2D;
  } else if (converted_geometry == HSA_EXT_IMAGE_GEOMETRY_2DADEPTH) {
    converted_geometry = HSA_EXT_IMAGE_GEOMETRY_2DA;
  }

  hsa_ext_image_format_t new_format = {
      static_cast<hsa_ext_image_channel_type_t>(converted_type),
      static_cast<hsa_ext_image_channel_order_t>(converted_order)};

  Image* new_image_handle = Image::Create(original_image.component);
  *new_image_handle=original_image;
  new_image_handle->desc.geometry = converted_geometry;

  hsa_status_t status = ImageRuntime::instance()
                            ->image_manager(new_image_handle->component)
                            ->ModifyImageSrd(*new_image_handle, new_format);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }

  *new_image = new_image_handle;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t BlitKernel::LaunchKernel(BlitQueue& blit_queue,
                                      hsa_kernel_dispatch_packet_t& packet) {
  static const uint16_t kInvalidPacketHeader = HSA_PACKET_TYPE_INVALID;

  static const uint16_t kDispatchPacketHeader =
      (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
      (0 << HSA_PACKET_HEADER_BARRIER) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);

  // Copying the packet content to the queue buffer is not atomic, so it is
  // possible that the packet has a valid packet type but invalid content.
  // To make sure packet processor does not read invalid packet, we first
  // initialized the packet type to invalid.
  packet.header = kInvalidPacketHeader;

  // Setup completion signal.
  hsa_signal_t kernel_signal = {0};
  hsa_status_t status = HSA::hsa_signal_create(1, 0, NULL, &kernel_signal);
  if (HSA_STATUS_SUCCESS != status) {
    return status;
  }
  packet.completion_signal = kernel_signal;

  // Populate the queue.
  hsa_queue_t* queue = blit_queue.queue_;
  const uint32_t bitmask = queue->size - 1;

  // Reserve write index.
  uint64_t write_index = HSA::hsa_queue_add_write_index_scacq_screl(queue, 1);

  while (true) {
    // Wait until we have room in the queue;
    const uint64_t read_index = HSA::hsa_queue_load_read_index_relaxed(queue);
    if ((write_index - read_index) < queue->size) {
      break;
    }
  }

  // Populate queue buffer with AQL packet.
  hsa_kernel_dispatch_packet_t* queue_buffer =
      reinterpret_cast<hsa_kernel_dispatch_packet_t*>(queue->base_address);
  queue_buffer[write_index & bitmask] = packet;

  std::atomic_thread_fence(std::memory_order_release);

  // Enable packet.
  queue_buffer[write_index & bitmask].header = kDispatchPacketHeader;

  // Update doorbel register.
  HSA::hsa_signal_store_screlease(queue->doorbell_signal, write_index);

  // Wait for the packet to finish.
  if (HSA::hsa_signal_wait_scacquire(kernel_signal, HSA_SIGNAL_CONDITION_LT, 1, uint64_t(-1),
                                     HSA_WAIT_STATE_ACTIVE) != 0) {
    status = HSA::hsa_signal_destroy(kernel_signal);
    assert(status == HSA_STATUS_SUCCESS);
    // Signal wait returned unexpected value.
    return HSA_STATUS_ERROR;
  }

  // Cleanup
  status = HSA::hsa_signal_destroy(kernel_signal);
  assert(status == HSA_STATUS_SUCCESS);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t BlitKernel::GetPatchedBlitObject(const char* agent_name,
                                              uint8_t** blit_code_object) {
  std::string sname(agent_name);

  if (sname == "gfx700") {
    *blit_code_object = ocl_blit_object_gfx700;
  } else if (sname == "gfx701") {
    *blit_code_object = ocl_blit_object_gfx701;
  } else if (sname == "gfx702") {
    *blit_code_object = ocl_blit_object_gfx702;
  } else if (sname == "gfx801") {
    *blit_code_object = ocl_blit_object_gfx801;
  } else if (sname == "gfx802") {
    *blit_code_object = ocl_blit_object_gfx802;
  } else if (sname == "gfx803") {
    *blit_code_object = ocl_blit_object_gfx803;
  } else if (sname == "gfx805") {
    *blit_code_object = ocl_blit_object_gfx805;
  } else if (sname == "gfx810") {
    *blit_code_object = ocl_blit_object_gfx810;
  } else if (sname == "gfx900") {
    *blit_code_object = ocl_blit_object_gfx900;
  } else if (sname == "gfx902") {
    *blit_code_object = ocl_blit_object_gfx902;
  } else if (sname == "gfx904") {
    *blit_code_object = ocl_blit_object_gfx904;
  } else if (sname == "gfx906") {
    *blit_code_object = ocl_blit_object_gfx906;
  } else if (sname == "gfx908") {
    *blit_code_object = ocl_blit_object_gfx908;
  } else if (sname == "gfx909") {
    *blit_code_object = ocl_blit_object_gfx909;
  } else if (sname == "gfx90a") {
    *blit_code_object = ocl_blit_object_gfx90a;
  } else if (sname == "gfx90c") {
    *blit_code_object = ocl_blit_object_gfx90c;
  } else if (sname == "gfx942") {
    *blit_code_object = ocl_blit_object_gfx942;
  } else if (sname == "gfx950") {
    *blit_code_object = ocl_blit_object_gfx950;
  } else if (sname == "gfx1010") {
    *blit_code_object = ocl_blit_object_gfx1010;
  } else if (sname == "gfx1011") {
    *blit_code_object = ocl_blit_object_gfx1011;
  } else if (sname == "gfx1012") {
    *blit_code_object = ocl_blit_object_gfx1012;
  } else if (sname == "gfx1013") {
    *blit_code_object = ocl_blit_object_gfx1013;
  } else if (sname == "gfx1030") {
    *blit_code_object = ocl_blit_object_gfx1030;
  } else if (sname == "gfx1031") {
    *blit_code_object = ocl_blit_object_gfx1031;
  } else if (sname == "gfx1032") {
    *blit_code_object = ocl_blit_object_gfx1032;
  } else if (sname == "gfx1033") {
    *blit_code_object = ocl_blit_object_gfx1033;
  } else if (sname == "gfx1034") {
    *blit_code_object = ocl_blit_object_gfx1034;
  } else if (sname == "gfx1035") {
    *blit_code_object = ocl_blit_object_gfx1035;
  } else if (sname == "gfx1036") {
    *blit_code_object = ocl_blit_object_gfx1036;
  } else if (sname == "gfx1100") {
    *blit_code_object = ocl_blit_object_gfx1100;
  } else if (sname == "gfx1101") {
    *blit_code_object = ocl_blit_object_gfx1101;
  } else if (sname == "gfx1102") {
    *blit_code_object = ocl_blit_object_gfx1102;
  } else if (sname == "gfx1103") {
    *blit_code_object = ocl_blit_object_gfx1103;
  } else if (sname == "gfx1150") {
    *blit_code_object = ocl_blit_object_gfx1150;
  } else if (sname == "gfx1151") {
    *blit_code_object = ocl_blit_object_gfx1151;
  } else if (sname == "gfx1152") {
    *blit_code_object = ocl_blit_object_gfx1152;
  } else if (sname == "gfx1153") {
    *blit_code_object = ocl_blit_object_gfx1153;
  } else if (sname == "gfx1200") {
    *blit_code_object = ocl_blit_object_gfx1200;
  } else if (sname == "gfx1201") {
    *blit_code_object = ocl_blit_object_gfx1201;
  } else {
    return HSA_STATUS_ERROR_INVALID_ISA_NAME;
  }

  return HSA_STATUS_SUCCESS;
}

}  // namespace image
}  // namespace rocr
#undef HSA_ARGUMENT_ALIGN_BYTES
