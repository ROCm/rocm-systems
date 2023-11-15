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

#ifndef _SRC_CORE_PROXY_QUEUE_H
#define _SRC_CORE_PROXY_QUEUE_H

#include <hsa/hsa.h>
#include <string.h>
#include <atomic>
#include <map>
#include <mutex>

#include "src/core/queue.h"
#include "src/core/types.h"

struct HsaApiTable;

namespace rocprofiler {
typedef void (*hsa_amd_queue_intercept_packet_writer)(const void* packets, uint64_t count);
typedef void (*on_submit_cb_t)(const void* packet, uint64_t count, uint64_t que_idx, void* data,
                               hsa_amd_queue_intercept_packet_writer writer);

class ProxyQueue : public Queue {
 public:
  static void InitFactory() {
    const char* type = getenv("ROCP_PROXY_QUEUE");
    if (type != NULL) {
      if (strncmp(type, "rocp", 4) == 0) rocp_type_ = true;
    }
  }

  static void HsaIntercept(HsaApiTable* table);

  static ProxyQueue* Create(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
                            void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data),
                            void* data, uint32_t private_segment_size, uint32_t group_segment_size,
                            hsa_queue_t** queue, hsa_status_t* status);

  static hsa_status_t Destroy(const ProxyQueue* obj);

  virtual hsa_status_t Init(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
                            void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data),
                            void* data, uint32_t private_segment_size, uint32_t group_segment_size,
                            hsa_queue_t** queue) = 0;
  virtual hsa_status_t Cleanup() const = 0;
  virtual hsa_status_t SetInterceptCB(on_submit_cb_t on_submit_cb, void* data) = 0;
  virtual void Submit(const packet_t* packet) = 0;

 protected:
  virtual ~ProxyQueue(){};

 private:
  static bool rocp_type_;
};

}  // namespace rocprofiler

#endif  // _SRC_CORE_PROXY_QUEUE_H
