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
#ifndef HSA_RUNTIME_IMAGE_INC_AMD_MIPMAP_IMPL_H_
#define HSA_RUNTIME_IMAGE_INC_AMD_MIPMAP_IMPL_H_

#include "inc/hsa_ext_amd.h"
#include "inc/hsa_amd_mipmap.h"
#include "core/inc/amd_gpu_agent.h"
#include "addrlib/inc/addrinterface.h"

namespace rocr { namespace image {

class ImageRuntime; // forward declaration for runtime helper access

//---------------------------------------------------------------------------//
//  Mipmapped Array helper / AddrLib bridge declarations
//---------------------------------------------------------------------------//

hsa_status_t GetImageExtensionTable(hsa_ext_images_1_pfn_t* table);

extern "C" {

hsa_status_t HSA_API
hsa_amd_mipmap_array_get_info(
  hsa_agent_t agent,
  const hsa_ext_image_descriptor_t* desc,
  uint32_t requested_levels,
  hsa_ext_image_data_layout_t layout,
  size_t row_pitch,
  size_t slice_pitch,
  hsa_amd_mipmap_array_info_t* info);

hsa_status_t HSA_API
hsa_amd_mipmap_array_create(
  hsa_agent_t agent,
  const hsa_ext_image_descriptor_t* desc,
  const void* image_data,
  hsa_access_permission_t access_permission,
  uint32_t num_mipmap_levels,
  hsa_ext_image_t* image_out);

hsa_status_t HSA_API
hsa_amd_mipmap_array_destroy(const hsa_ext_image_t* image);


hsa_status_t HSA_API
hsa_amd_mipmap_array_get_level(
    hsa_agent_t agent,
    const hsa_ext_image_t* mipmapped_array,
    uint32_t mip_level,
    hsa_ext_image_t* level_image_out);

} // extern "C"
} } // namespace rocr::image
#endif  // HSA_RUNTIME_IMAGE_INC_AMD_MIPMAP_IMPL_H_
