#ifndef _SRC_CORE_HSA_PROXY_QUEUE_H
#define _SRC_CORE_HSA_PROXY_QUEUE_H

#include <hsa.h>
#include <atomic>
#include <map>
#include <mutex>

#include "core/proxy_queue.h"
#include "util/exception.h"

namespace rocprofiler {
extern decltype(hsa_queue_destroy)* hsa_queue_destroy_fn;
extern decltype(hsa_amd_queue_intercept_create)* hsa_amd_queue_intercept_create_fn;
extern decltype(hsa_amd_queue_intercept_register)* hsa_amd_queue_intercept_register_fn;

class HsaProxyQueue : public ProxyQueue {
  public:
  hsa_status_t SetInterceptCB(on_submit_cb_t on_submit_cb, void* data) {
    return hsa_amd_queue_intercept_register_fn(queue_, on_submit_cb, data);
  }

  void Submit(const packet_t* packet) { EXC_RAISING(HSA_STATUS_ERROR, "HsaProxyQueue::Submit() is not supported"); }

  private:
  hsa_status_t Init(
    hsa_agent_t agent,
    uint32_t size,
    hsa_queue_type32_t type,
    void (*callback)(hsa_status_t status, hsa_queue_t *source, void *data),
    void *data,
    uint32_t private_segment_size,
    uint32_t group_segment_size,
    hsa_queue_t **queue)
  {
    printf("HsaProxyQueue::Init()\n");
    const auto status = hsa_amd_queue_intercept_create_fn(agent, size, type, callback, data, private_segment_size, group_segment_size, &queue_);
    *queue = queue_;
    return status;
  }

  hsa_status_t Cleanup() const { return hsa_queue_destroy_fn(queue_); }

  hsa_queue_t* queue_;
};

} // namespace rocprofiler

#endif // _SRC_CORE_HSA_PROXY_QUEUE_H
