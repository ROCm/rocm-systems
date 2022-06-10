/* Copyright (c) 2018-2022 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#ifndef INC_ROCTRACER_HSA_H_
#define INC_ROCTRACER_HSA_H_

#include <roctracer.h>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa_prof_str.h>

#include <rocprofiler/activity.h>

// HSA OP ID enumeration
enum hsa_op_id_t {
  HSA_OP_ID_DISPATCH = 0,
  HSA_OP_ID_COPY = 1,
  HSA_OP_ID_BARRIER = 2,
  HSA_OP_ID_RESERVED1 = 3,
  HSA_OP_ID_NUMBER
};

struct hsa_ops_properties_t {
  void* table;
  void* reserved1[3];
};

#endif  // INC_ROCTRACER_HSA_H_
