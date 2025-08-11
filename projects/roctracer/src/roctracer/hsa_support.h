/* Copyright (c) 2022 Advanced Micro Devices, Inc.

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

#ifndef HSA_SUPPORT_H_
#define HSA_SUPPORT_H_

#include "roctracer.h"
#include "roctracer_hsa.h"

#include <hsa/hsa_api_trace.h>

namespace roctracer::hsa_support {

struct hsa_trace_data_t {
  hsa_api_data_t api_data;
  uint64_t phase_enter_timestamp;
  uint64_t phase_data;

  void (*phase_enter)(hsa_api_id_t operation_id, hsa_trace_data_t* data);
  void (*phase_exit)(hsa_api_id_t operation_id, hsa_trace_data_t* data);
};

void Initialize(HsaApiTable* table);
void Finalize();

const char* GetApiName(uint32_t id);
const char* GetEvtName(uint32_t id);
const char* GetOpsName(uint32_t id);
uint32_t GetApiCode(const char* str);

void RegisterTracerCallback(int (*function)(activity_domain_t domain, uint32_t operation_id,
                                            void* data));
uint64_t timestamp_ns();

}  // namespace roctracer::hsa_support

#endif  // HSA_SUPPORT_H_
