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

#ifndef HSA_RUNTIME_CORE_INC_HSA_API_TRACE_INT_H
#define HSA_RUNTIME_CORE_INC_HSA_API_TRACE_INT_H

#include "inc/hsa_api_trace.h"
#include "core/inc/hsa_internal.h"

namespace rocr {
namespace core {
  struct HsaApiTable {

    static const uint32_t HSA_EXT_FINALIZER_API_TABLE_ID = 0;
    static const uint32_t HSA_EXT_IMAGE_API_TABLE_ID = 1;
    static const uint32_t HSA_EXT_AQLPROFILE_API_TABLE_ID = 2;
    static const uint32_t HSA_EXT_PC_SAMPLING_API_TABLE_ID = 3;

    ::HsaApiTable hsa_api;
    ::CoreApiTable core_api;
    ::AmdExtTable amd_ext_api;
    ::FinalizerExtTable finalizer_api;
    ::ImageExtTable image_api;
    ::ToolsApiTable tools_api;
    ::PcSamplingExtTable pcs_api;

    HsaApiTable();
    void Init();
    void UpdateCore();
    void UpdateAmdExts();
    void UpdateTools();
    void CloneExts(void* ptr, uint32_t table_id);
    void LinkExts(void* ptr, uint32_t table_id);
    void Reset();
  };

  extern HsaApiTable& hsa_api_table();
  extern HsaApiTable& hsa_internal_api_table();

  void LoadInitialHsaApiTable();
}   //  namespace core
}   //  namespace rocr

#endif
