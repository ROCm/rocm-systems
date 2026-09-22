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

#include <hsa/hsa.h>
#include <string.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <atomic>

#include "ctrl/run_kernel.h"
#include "ctrl/test_aql.h"
#include "ctrl/test_hsa.h"
#include "rocprofiler/rocprofiler.h"
#include "dummy_kernel/dummy_kernel.h"
#include "simple_convolution/simple_convolution.h"
#include "util/test_assert.h"

// Dispatch callbacks and context handlers synchronization
pthread_mutex_t mutex = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;

// Error handler
void fatal(const std::string msg) {
  fflush(stdout);
  fprintf(stderr, "%s\n\n", msg.c_str());
  fflush(stderr);
  abort();
}

// Check returned HSA API status
void check_status(hsa_status_t status) {
  if (status != HSA_STATUS_SUCCESS) {
    const char* error_string = NULL;
    rocprofiler_error_string(&error_string);
    fprintf(stderr, "ERROR: %s\n", error_string);
    abort();
  }
}

// Context stored entry type
struct context_entry_t {
  bool valid;
  hsa_agent_t agent;
  rocprofiler_group_t group;
  rocprofiler_callback_data_t data;
};

// Dump stored context entry
void dump_context_entry(context_entry_t* entry) {
  volatile std::atomic<bool>* valid = reinterpret_cast<std::atomic<bool>*>(&entry->valid);
  while (valid->load() == false) sched_yield();

  const std::string kernel_name = entry->data.kernel_name;
  const rocprofiler_dispatch_record_t* record = entry->data.record;

  fflush(stdout);
  fprintf(stdout, "kernel-object(0x%lx) name(\"%s\")", entry->data.kernel_object,
          kernel_name.c_str());
  if (record)
    fprintf(stdout, ", gpu-id(%u), time(%lu,%lu,%lu,%lu)",
            HsaRsrcFactory::Instance().GetAgentInfo(entry->agent)->dev_index, record->dispatch,
            record->begin, record->end, record->complete);
  fprintf(stdout, "\n");
  fflush(stdout);

  rocprofiler_group_t& group = entry->group;
  if (group.context == NULL) {
    fprintf(stderr, "tool error: context is NULL\n");
    abort();
  }

  rocprofiler_close(group.context);
}

// Profiling completion handler
// Dump and delete the context entry
// Return true if the context was dumped successfully
bool context_handler(rocprofiler_group_t group, void* arg) {
  context_entry_t* entry = reinterpret_cast<context_entry_t*>(arg);

  if (pthread_mutex_lock(&mutex) != 0) {
    perror("pthread_mutex_lock");
    abort();
  }

  dump_context_entry(entry);
  delete entry;

  if (pthread_mutex_unlock(&mutex) != 0) {
    perror("pthread_mutex_unlock");
    abort();
  }

  return false;
}

// Kernel disoatch callback
hsa_status_t dispatch_callback(const rocprofiler_callback_data_t* callback_data,
                               void* /*user_data*/, rocprofiler_group_t* group) {
  // HSA status
  hsa_status_t status = HSA_STATUS_ERROR;

  // Profiling context
  rocprofiler_t* context = NULL;

  // Context entry
  context_entry_t* entry = new context_entry_t();

  // context properties
  rocprofiler_properties_t properties{};
  properties.handler = context_handler;
  properties.handler_arg = (void*)entry;

  // Open profiling context
  status = rocprofiler_open(callback_data->agent, NULL, 0, &context,
                            0 /*ROCPROFILER_MODE_SINGLEGROUP*/, &properties);
  check_status(status);

  // Get group[0]
  status = rocprofiler_get_group(context, 0, group);
  check_status(status);

  // Fill profiling context entry
  entry->agent = callback_data->agent;
  entry->group = *group;
  entry->data = *callback_data;
  entry->data.kernel_name = strdup(callback_data->kernel_name);
  reinterpret_cast<std::atomic<bool>*>(&entry->valid)->store(true);

  return HSA_STATUS_SUCCESS;
}

int main() {
  bool ret_val = true;
  const char* kiter_s = getenv("ROCP_KITER");
  const char* diter_s = getenv("ROCP_DITER");
  const unsigned kiter = (kiter_s != NULL) ? atol(kiter_s) : 1;
  const unsigned diter = (diter_s != NULL) ? atol(diter_s) : 1;

  // Adding dispatch observer
  rocprofiler_queue_callbacks_t callbacks_ptrs{};
  callbacks_ptrs.dispatch = dispatch_callback;
  rocprofiler_set_queue_callbacks(callbacks_ptrs, NULL);

  // Instantiate HSA resources
  HsaRsrcFactory::Create();

  // Getting GPU device info
  const AgentInfo* agent_info = NULL;
  if (HsaRsrcFactory::Instance().GetGpuAgentInfo(0, &agent_info) == false) abort();

  // Creating the queue
  hsa_queue_t* queue = NULL;
  if (HsaRsrcFactory::Instance().CreateQueue(agent_info, 128, &queue) == false) abort();

  // Test initialization
  TestHsa::HsaInstantiate();

  for (unsigned ind = 0; ind < kiter; ++ind) {
    printf("Iteration %u:\n", ind);
    if ((ind & 1) == 0)
      rocprofiler_start_queue_callbacks();
    else
      rocprofiler_stop_queue_callbacks();
    ret_val = RunKernel<DummyKernel, TestAql>(0, NULL, agent_info, queue, diter);
    if (ret_val) ret_val = RunKernel<SimpleConvolution, TestAql>(0, NULL, agent_info, queue, diter);
  }

  TestHsa::HsaShutdown();

  return (ret_val) ? 0 : 1;
}
