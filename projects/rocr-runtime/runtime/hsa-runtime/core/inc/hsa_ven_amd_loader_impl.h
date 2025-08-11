////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
// 
// Copyright (c) 2020-2020, Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef HSA_RUNTME_CORE_INC_HSA_VEN_AMD_LOADER_IMPL_H_
#define HSA_RUNTME_CORE_INC_HSA_VEN_AMD_LOADER_IMPL_H_

#include "inc/hsa_ven_amd_loader.h"

namespace rocr {

  hsa_status_t hsa_ven_amd_loader_query_host_address(
    const void *device_address,
    const void **host_address);

  hsa_status_t hsa_ven_amd_loader_query_segment_descriptors(
    hsa_ven_amd_loader_segment_descriptor_t *segment_descriptors,
    size_t *num_segment_descriptors);

  hsa_status_t hsa_ven_amd_loader_query_executable(
    const void *device_address,
    hsa_executable_t *executable);

  hsa_status_t hsa_ven_amd_loader_executable_iterate_loaded_code_objects(
    hsa_executable_t executable,
    hsa_status_t (*callback)(
    hsa_executable_t executable,
    hsa_loaded_code_object_t loaded_code_object,
    void *data),
    void *data);

  hsa_status_t hsa_ven_amd_loader_loaded_code_object_get_info(
    hsa_loaded_code_object_t loaded_code_object,
    hsa_ven_amd_loader_loaded_code_object_info_t attribute,
    void *value);

  hsa_status_t
    hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size(
    hsa_file_t file,
    size_t offset,
    size_t size,
    hsa_code_object_reader_t *code_object_reader);

  hsa_status_t
    hsa_ven_amd_loader_iterate_executables(
    hsa_status_t (*callback)(
      hsa_executable_t executable,
      void *data),
    void *data);
}  // namespace rocr

#endif
