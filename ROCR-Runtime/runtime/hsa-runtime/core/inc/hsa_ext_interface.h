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

#ifndef HSA_RUNTME_CORE_INC_AMD_EXT_INTERFACE_H_
#define HSA_RUNTME_CORE_INC_AMD_EXT_INTERFACE_H_

#include <string>
#include <vector>

#include "core/inc/hsa_api_trace_int.h"

#include "core/util/os.h"
#include "core/util/utils.h"

namespace rocr {
namespace core {
struct ImageExtTableInternal : public ImageExtTable {
  decltype(::hsa_amd_image_get_info_max_dim)* hsa_amd_image_get_info_max_dim_fn;
};

struct PcSamplingExtTableInternal : public PcSamplingExtTable {};

class ExtensionEntryPoints {
 public:

  // Table of function pointers for Hsa Extension Image
  ImageExtTableInternal image_api;

  // Table of function pointers for Hsa vendor PC Sampling
  PcSamplingExtTableInternal pcs_api;

  // Table of function pointers for Hsa Extension Finalizer
  FinalizerExtTable finalizer_api;

  ExtensionEntryPoints();

  bool LoadFinalizer(std::string library_name);
  void Unload();

  // Update Image Api table with handles to implementation
  bool LoadImage();

  // Reset Api tables to point to null implementations
  void UnloadImage();

  // Update PC Sampling Api table with handles to implementation
  void LoadPcSampling();

  // Reset PC Sampling tables to point to null implementations
  void UnloadPcSampling();

 private:
  typedef void (*Load_t)(const ::HsaApiTable* table);
  typedef void (*Unload_t)();

  std::vector<os::LibHandle> libs_;

  // Initialize table for HSA Finalizer Extension Api's
  void InitFinalizerExtTable();

  // Initialize table for HSA Image Extension Api's
  void InitImageExtTable();

  // Initialize table for HSA PC Sampling Extension Api's
  void InitPcSamplingExtTable();

  // Initialize Amd Ext table for Api related to Images
  void InitAmdExtTable();

  // Update Amd Ext table for Api related to Images
  void UpdateAmdExtTable(decltype(::hsa_amd_image_create)* func_ptr);

  DISALLOW_COPY_AND_ASSIGN(ExtensionEntryPoints);
};
}   //  namespace core
}   //  namespace rocr

#endif
