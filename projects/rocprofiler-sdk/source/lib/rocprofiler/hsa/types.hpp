// Copyright (c) 2018-2023 Advanced Micro Devices, Inc.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include "rocprofiler/hsa.h"

#if defined(ROCPROFILER_CI) && ROCPROFILER_CI > 0
#    if HSA_API_TABLE_MAJOR_VERSION <= 0x01
static_assert(HSA_CORE_API_TABLE_MAJOR_VERSION == 0x01,
              "Change in the major version of HSA core API table");
static_assert(HSA_AMD_EXT_API_TABLE_MAJOR_VERSION == 0x01,
              "Change in the major version of HSA amd-extended API table");
static_assert(HSA_FINALIZER_API_TABLE_MAJOR_VERSION == 0x01,
              "Change in the major version of HSA finalizer API table");
static_assert(HSA_IMAGE_API_TABLE_MAJOR_VERSION == 0x01,
              "Change in the major version of HSA image API table");
static_assert(HSA_AQLPROFILE_API_TABLE_MAJOR_VERSION == 0x01,
              "Change in the major version of HSA aqlprofile API table");

static_assert(HSA_CORE_API_TABLE_STEP_VERSION == 0x00,
              "Change in the major version of HSA core API table");
static_assert(HSA_AMD_EXT_API_TABLE_STEP_VERSION == 0x00,
              "Change in the major version of HSA amd-extended API table");
static_assert(HSA_FINALIZER_API_TABLE_STEP_VERSION == 0x00,
              "Change in the major version of HSA finalizer API table");
static_assert(HSA_IMAGE_API_TABLE_STEP_VERSION == 0x00,
              "Change in the major version of HSA image API table");
static_assert(HSA_AQLPROFILE_API_TABLE_STEP_VERSION == 0x00,
              "Change in the major version of HSA aqlprofile API table");

// if you hit these static asserts, that means HSA added entries to the table but did not update the
// step numbers
static_assert(sizeof(FinalizerExtTable) == 64, "HSA finalizer API table size changed");
static_assert(sizeof(ImageExtTable) == 120, "HSA image-extended API table size changed");
static_assert(sizeof(AmdExtTable) == 552, "HSA amd-extended API table size changed");
static_assert(sizeof(CoreApiTable) == 1016, "HSA core API table size changed");
#    else
#        error "HSA_API_TABLE_MAJOR_VERSION not supported"
#    endif
#endif
