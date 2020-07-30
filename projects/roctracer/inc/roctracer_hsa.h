/*
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef INC_ROCTRACER_HSA_H_
#define INC_ROCTRACER_HSA_H_

#include <hsa.h>
#include <hsa_ext_amd.h>

#include <roctracer.h>

// HSA OP ID enumeration
enum hsa_op_id_t {
  HSA_OP_ID_DISPATCH = 0,
  HSA_OP_ID_COPY = 1,
  HSA_OP_ID_BARRIER = 2,
  HSA_OP_ID_RESERVED1 = 3,
  HSA_OP_ID_NUMBER = 4
};

#ifdef __cplusplus
#include <iostream>
#include <hsa_api_trace.h>

namespace roctracer {
namespace hsa_support {
enum {
  HSA_OP_ID_async_copy = 0
};

extern CoreApiTable CoreApiTable_saved;
extern AmdExtTable AmdExtTable_saved;
extern ImageExtTable ImageExtTable_saved;

struct ops_properties_t {
  void* table;
  activity_async_callback_t async_copy_callback_fun;
  void* async_copy_callback_arg;
  const char* output_prefix;
};

}; // namespace hsa_support

typedef hsa_support::ops_properties_t hsa_ops_properties_t;
}; // namespace roctracer

#include "hsa_ostream_ops.h"

std::ostream& operator<<(std::ostream& out, const hsa_amd_memory_pool_t& v)
{
   roctracer::hsa_support::operator<<(out, v);
   return out;
}

std::ostream& operator<<(std::ostream& out, const hsa_ext_image_t& v)
{
   roctracer::hsa_support::operator<<(out, v);
   return out;
}

std::ostream& operator<<(std::ostream& out, const hsa_ext_sampler_t& v)
{
   roctracer::hsa_support::operator<<(out, v);
   return out;
}

#else // !__cplusplus
typedef void* hsa_amd_queue_intercept_handler;
typedef void* hsa_amd_runtime_queue_notifier;
#endif //! __cplusplus

#include <hsa_prof_str.h>
#endif // INC_ROCTRACER_HSA_H_
