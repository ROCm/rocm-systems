#include "core/simple_proxy_queue.h"

namespace rocprofiler {
void SimpleProxyQueue::HsaIntercept(HsaApiTable* table) {
  table->core_->hsa_signal_store_relaxed_fn = rocprofiler::SimpleProxyQueue::SignalStore;
  table->core_->hsa_queue_load_write_index_relaxed_fn = rocprofiler::SimpleProxyQueue::LoadIndex;
  table->core_->hsa_queue_store_write_index_relaxed_fn = rocprofiler::SimpleProxyQueue::StoreIndex;
}

SimpleProxyQueue::queue_map_t* SimpleProxyQueue::queue_map_ = NULL;
}  // namespace rocprofiler
