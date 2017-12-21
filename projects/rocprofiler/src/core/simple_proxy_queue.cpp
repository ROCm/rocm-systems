#include "core/simple_proxy_queue.h"

namespace rocprofiler {
void SimpleProxyQueue::HsaIntercept(HsaApiTable* table) {
  table->core_->hsa_signal_store_relaxed_fn = rocprofiler::SimpleProxyQueue::SignalStore;
  table->core_->hsa_signal_store_screlease_fn = rocprofiler::SimpleProxyQueue::SignalStore;

  table->core_->hsa_queue_load_write_index_relaxed_fn = rocprofiler::SimpleProxyQueue::GetQueueIndex;
  table->core_->hsa_queue_store_write_index_relaxed_fn = rocprofiler::SimpleProxyQueue::SetQueueIndex;
  table->core_->hsa_queue_load_read_index_relaxed_fn = rocprofiler::SimpleProxyQueue::GetSubmitIndex;

  table->core_->hsa_queue_load_write_index_scacquire_fn = rocprofiler::SimpleProxyQueue::GetQueueIndex;
  table->core_->hsa_queue_store_write_index_screlease_fn = rocprofiler::SimpleProxyQueue::SetQueueIndex;
  table->core_->hsa_queue_load_read_index_scacquire_fn = rocprofiler::SimpleProxyQueue::GetSubmitIndex;
}

SimpleProxyQueue::queue_map_t* SimpleProxyQueue::queue_map_ = NULL;
}  // namespace rocprofiler
