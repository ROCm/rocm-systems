// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef INC_AQL_PROFILE_H_
#define INC_AQL_PROFILE_H_

#if 0
// Profiling parameters
// All parameters are generic and if not applicable for a specific
// profile configuration then error status will be returned.
typedef enum {
  // SQTT applicable parameters
  HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_COMPUTE_UNIT_TARGET = 0,
  HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_VM_ID_MASK = 1,
  HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_MASK = 2,
  HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK = 3,
  HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK2 = 4,
  HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SE_MASK = 5
} hsa_ven_amd_aqlprofile_parameter_name_t;

// Profile attributes
typedef enum {
  HSA_VEN_AMD_AQLPROFILE_INFO_COMMAND_BUFFER_SIZE = 0,  // get_info returns uint32_t value
  HSA_VEN_AMD_AQLPROFILE_INFO_PMC_DATA_SIZE = 1,        // get_info returns uint32_t value
  HSA_VEN_AMD_AQLPROFILE_INFO_PMC_DATA = 2,             // get_info returns PMC uint64_t value
                                                        // in info_data object
  HSA_VEN_AMD_AQLPROFILE_INFO_SQTT_DATA = 3,            // get_info returns SQTT buffer ptr/size
                                                        // in info_data object
                                                        //
  HSA_VEN_AMD_AQLPROFILE_INFO_BLOCK_COUNTERS = 4,       // get_info returns number of block counter
  HSA_VEN_AMD_AQLPROFILE_INFO_BLOCK_ID = 5,             // get_info returns block id, instances
                                                        // by name string using _id_query_t
                                                        //
  HSA_VEN_AMD_AQLPROFILE_INFO_ENABLE_CMD = 6,           // get_info returns size/pointer for
                                                        // counters enable command buffer
} hsa_ven_amd_aqlprofile_info_type_t;
#endif

#endif  // INC_AQL_PROFILE_H_
