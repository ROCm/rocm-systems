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

void Initialize(HsaApiTable* table);
void Finalize();

const char* GetApiName(uint32_t id);
const char* GetEvtName(uint32_t id);
const char* GetOpsName(uint32_t id);
uint32_t GetApiCode(const char* str);

void EnableActivity(roctracer_domain_t domain, uint32_t op, roctracer_pool_t* pool);
void EnableCallback(roctracer_domain_t domain, uint32_t cid, roctracer_rtapi_callback_t callback,
                    void* user_data);

void DisableCallback(roctracer_domain_t domain, uint32_t cid);
void DisableActivity(roctracer_domain_t domain, uint32_t op);

uint64_t timestamp_ns();

}  // namespace roctracer::hsa_support

#endif  // HSA_SUPPORT_H_
