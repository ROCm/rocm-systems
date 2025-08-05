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

#ifndef _SRC_CORE_HSA_PROXY_QUEUE_H
#define _SRC_CORE_HSA_PROXY_QUEUE_H

#include <hsa/hsa.h>
#include <atomic>
#include <map>
#include <mutex>

#include "core/proxy_queue.h"
#include "util/exception.h"
#include "util/hsa_rsrc_factory.h"

namespace rocprofiler {
extern decltype(::hsa_queue_destroy)* hsa_queue_destroy_fn;
extern decltype(::hsa_amd_queue_intercept_create)* hsa_amd_queue_intercept_create_fn;
extern decltype(::hsa_amd_queue_intercept_register)* hsa_amd_queue_intercept_register_fn;

class HsaProxyQueue : public ProxyQueue {
 public:
  hsa_status_t SetInterceptCB(on_submit_cb_t on_submit_cb, void* data) {
    return hsa_amd_queue_intercept_register_fn(queue_, on_submit_cb, data);
  }

  void Submit(const packet_t* packet) {
    EXC_RAISING(HSA_STATUS_ERROR, "HsaProxyQueue::Submit() is not supported");
  }

 private:
  hsa_status_t Init(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
                    void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data),
                    void* data, uint32_t private_segment_size, uint32_t group_segment_size,
                    hsa_queue_t** queue) {
    const auto status = hsa_amd_queue_intercept_create_fn(
        agent, size, type, callback, data, private_segment_size, group_segment_size, &queue_);
    *queue = queue_;
    return status;
  }

  hsa_status_t Cleanup() const { return hsa_queue_destroy_fn(queue_); }

  hsa_queue_t* queue_;
};

}  // namespace rocprofiler

#endif  // _SRC_CORE_HSA_PROXY_QUEUE_H
