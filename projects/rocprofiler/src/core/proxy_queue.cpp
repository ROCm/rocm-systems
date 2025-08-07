/******************************************************************************
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
*******************************************************************************/

#include "core/proxy_queue.h"

#include "core/hsa_proxy_queue.h"
#include "core/simple_proxy_queue.h"

namespace rocprofiler {
void ProxyQueue::HsaIntercept(HsaApiTable* table) {
  if (rocp_type_) SimpleProxyQueue::HsaIntercept(table);
}

ProxyQueue* ProxyQueue::Create(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
                               void (*callback)(hsa_status_t status, hsa_queue_t* source,
                                                void* data),
                               void* data, uint32_t private_segment_size,
                               uint32_t group_segment_size, hsa_queue_t** queue,
                               hsa_status_t* status) {
  hsa_status_t suc = HSA_STATUS_ERROR;
  ProxyQueue* instance =
      (rocp_type_) ? (ProxyQueue*)new SimpleProxyQueue() : (ProxyQueue*)new HsaProxyQueue();
  if (instance != NULL) {
    suc = instance->Init(agent, size, type, callback, data, private_segment_size,
                         group_segment_size, queue);
    if (suc != HSA_STATUS_SUCCESS) {
      delete instance;
      instance = NULL;
    }
  }
  *status = suc;
  assert(*status == HSA_STATUS_SUCCESS);
  return instance;
}

hsa_status_t ProxyQueue::Destroy(const ProxyQueue* obj) {
  assert(obj != NULL);
  auto suc = obj->Cleanup();
  delete obj;
  return suc;
}

bool ProxyQueue::rocp_type_ = false;
}  // namespace rocprofiler
