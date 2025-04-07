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

#include "core/simple_proxy_queue.h"

namespace rocprofiler {
void SimpleProxyQueue::HsaIntercept(HsaApiTable* table) {
  table->core_->hsa_signal_store_relaxed_fn = rocprofiler::SimpleProxyQueue::SignalStore;
  table->core_->hsa_signal_store_screlease_fn = rocprofiler::SimpleProxyQueue::SignalStore;

  table->core_->hsa_queue_load_write_index_relaxed_fn =
      rocprofiler::SimpleProxyQueue::GetQueueIndex;
  table->core_->hsa_queue_store_write_index_relaxed_fn =
      rocprofiler::SimpleProxyQueue::SetQueueIndex;
  table->core_->hsa_queue_load_read_index_relaxed_fn =
      rocprofiler::SimpleProxyQueue::GetSubmitIndex;

  table->core_->hsa_queue_load_write_index_scacquire_fn =
      rocprofiler::SimpleProxyQueue::GetQueueIndex;
  table->core_->hsa_queue_store_write_index_screlease_fn =
      rocprofiler::SimpleProxyQueue::SetQueueIndex;
  table->core_->hsa_queue_load_read_index_scacquire_fn =
      rocprofiler::SimpleProxyQueue::GetSubmitIndex;
}

SimpleProxyQueue::queue_map_t* SimpleProxyQueue::queue_map_ = NULL;
}  // namespace rocprofiler
