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

#include "core/intercept_queue.h"

namespace rocprofiler {
void InterceptQueue::HsaIntercept(HsaApiTable* table) {
  table->core_->hsa_queue_create_fn = rocprofiler::InterceptQueue::QueueCreate;
  table->core_->hsa_queue_destroy_fn = rocprofiler::InterceptQueue::QueueDestroy;
}

std::atomic<rocprofiler_callback_t> InterceptQueue::dispatch_callback_{NULL};
const char* InterceptQueue::kernel_none_ = "";
Tracker* InterceptQueue::tracker_ = NULL;
bool InterceptQueue::tracker_on_ = false;
bool InterceptQueue::in_create_call_ = false;
InterceptQueue::queue_id_t InterceptQueue::current_queue_id = 0;

rocprofiler_hsa_callback_fun_t InterceptQueue::submit_callback_fun_ = NULL;
void* InterceptQueue::submit_callback_arg_ = NULL;

bool InterceptQueue::opt_mode_ = false;
uint32_t InterceptQueue::k_concurrent_ = K_CONC_OFF;
std::once_flag InterceptQueue::once_flag_;

}  // namespace rocprofiler
