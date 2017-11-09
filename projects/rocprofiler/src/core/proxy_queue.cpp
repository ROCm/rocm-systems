#include "core/proxy_queue.h"

#ifdef ROCP_HSA_PROXY
#include "core/hsa_proxy_queue.h"
#endif
#include "core/simple_proxy_queue.h"

namespace rocprofiler {
void ProxyQueue::HsaIntercept(HsaApiTable* table) {
  if (rocp_type_) SimpleProxyQueue::HsaIntercept(table);
}

ProxyQueue* ProxyQueue::Create(
  hsa_agent_t agent,
  uint32_t size,
  hsa_queue_type32_t type,
  void (*callback)(hsa_status_t status, hsa_queue_t *source, void *data),
  void *data,
  uint32_t private_segment_size,
  uint32_t group_segment_size,
  hsa_queue_t **queue,
  hsa_status_t* status)
{
  hsa_status_t suc = HSA_STATUS_ERROR;
#ifdef ROCP_HSA_PROXY
  ProxyQueue* instance = (rocp_type_) ? (ProxyQueue*) new SimpleProxyQueue() : (ProxyQueue*) new HsaProxyQueue();
#else
  ProxyQueue* instance = new SimpleProxyQueue();
#endif
  if (instance != NULL) {
    const auto suc = instance->Init(agent, size, type, callback, data, private_segment_size, group_segment_size, queue);
    if (suc != HSA_STATUS_SUCCESS) {
      delete instance;
      instance = NULL;
    }
  }
  *status = suc;
  return instance;
}

hsa_status_t ProxyQueue::Destroy(const ProxyQueue* obj) {
  auto suc = obj->Cleanup();
  delete obj;
  return suc;
}

bool ProxyQueue::rocp_type_ = false;
} // namespace rocprofiler
