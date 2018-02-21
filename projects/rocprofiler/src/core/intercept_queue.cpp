#include "core/intercept_queue.h"

namespace rocprofiler {
void InterceptQueue::HsaIntercept(HsaApiTable* table) {
  table->core_->hsa_queue_create_fn = rocprofiler::InterceptQueue::QueueCreate;
  table->core_->hsa_queue_destroy_fn = rocprofiler::InterceptQueue::QueueDestroy;
}

InterceptQueue::mutex_t InterceptQueue::mutex_;
rocprofiler_callback_t InterceptQueue::on_dispatch_cb_ = NULL;
void* InterceptQueue::on_dispatch_cb_data_ = NULL;
InterceptQueue::obj_map_t* InterceptQueue::obj_map_ = NULL;
const char* InterceptQueue::kernel_none_ = "";
}  // namespace rocprofiler
