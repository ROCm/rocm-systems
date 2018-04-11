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
      (rocp_type_) ? (ProxyQueue*) new SimpleProxyQueue() : (ProxyQueue*) new HsaProxyQueue();
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
