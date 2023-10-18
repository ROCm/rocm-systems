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

#include "activity.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <atomic>
#include <string>

// Tracer messages protocol
#define USE_PROF_API
#include <prof_protocol.h>

#include "core/context.h"
#include "util/hsa_rsrc_factory.h"

#define PUBLIC_API __attribute__((visibility("default")))

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

// Activity primitives
namespace activity_prim {
// PC sampling callback data
struct pcsmp_callback_data_t {
  const char* kernel_name;  // sampled kernel name
  void* data_buffer;        // host buffer for tracing data
  uint64_t id;              // sample id
  uint64_t cycle;           // sample cycle
  uint64_t pc;              // sample PC
};

uint32_t activity_op = UINT32_MAX;
void* activity_arg = NULL;
std::atomic<activity_async_callback_t> activity_callback{NULL};
rocprofiler_t* context = NULL;

hsa_status_t trace_data_cb(hsa_ven_amd_aqlprofile_info_type_t info_type,
                           hsa_ven_amd_aqlprofile_info_data_t* info_data, void* data) {
  const pcsmp_callback_data_t* pcsmp_data = (pcsmp_callback_data_t*)data;

  activity_record_t record{};
  record.op = activity_op;
  record.pc_sample.se = pcsmp_data->id;
  record.pc_sample.cycle = pcsmp_data->cycle;
  record.pc_sample.pc = pcsmp_data->pc;
  activity_async_callback_t fun = activity_callback.load(std::memory_order_acquire);
  if (fun) {
    (fun)(activity_op, &record, activity_arg);
  } else {
    free((void*)(pcsmp_data->kernel_name));
  }
  return HSA_STATUS_SUCCESS;
}

bool context_handler(rocprofiler_group_t group, void* arg) {
  hsa_agent_t agent{};
  hsa_status_t status = rocprofiler_get_agent(group.context, &agent);
  check_status(status);
  const rocprofiler::util::AgentInfo* agent_info =
      rocprofiler::util::HsaRsrcFactory::Instance().GetAgentInfo(agent);

  pcsmp_callback_data_t pcsmp_data{};
  pcsmp_data.kernel_name = (const char*)arg;
  pcsmp_data.data_buffer = rocprofiler::util::HsaRsrcFactory::Instance().AllocateSysMemory(
      agent_info, rocprofiler::TraceProfile::GetSize());
  status = rocprofiler_iterate_trace_data(group.context, trace_data_cb, &pcsmp_data);
  check_status(status);
  return false;
}

// Kernel disoatch callback
hsa_status_t dispatch_callback(const rocprofiler_callback_data_t* callback_data, void* user_data,
                               rocprofiler_group_t* group) {
  // context features
  const rocprofiler_feature_kind_t trace_kind = (rocprofiler_feature_kind_t)(
      ROCPROFILER_FEATURE_KIND_TRACE | ROCPROFILER_FEATURE_KIND_PCSMP_MOD);
  const uint32_t feature_count = 1;
  const uint32_t parameter_count = 1;
  rocprofiler_feature_t* features = new rocprofiler_feature_t[feature_count];
  memset(features, 0, feature_count * sizeof(rocprofiler_feature_t));
  rocprofiler_parameter_t* parameters = new rocprofiler_parameter_t[parameter_count];
  memset(features, 0, parameter_count * sizeof(rocprofiler_parameter_t));
  parameters[0].parameter_name = HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_COMPUTE_UNIT_TARGET;
  parameters[0].value = 0;

  features[0].kind = trace_kind;
  features[0].parameters = parameters;
  features[0].parameter_count = parameter_count;

  // context properties
  rocprofiler_properties_t properties{};
  properties.handler = context_handler;
  properties.handler_arg = (void*)strdup(callback_data->kernel_name);

  // Open profiling context
  hsa_status_t status = rocprofiler_open(callback_data->agent, features, feature_count, &context,
                                         0 /*ROCPROFILER_MODE_SINGLEGROUP*/, &properties);
  check_status(status);

  // Get group[0]
  status = rocprofiler_get_group(context, 0, group);
  check_status(status);

  return status;
}
}  // namespace activity_prim

extern "C" {
PUBLIC_API const char* GetOpName(uint32_t op) { return strdup("PCSAMPLE"); }

PUBLIC_API bool RegisterApiCallback(uint32_t op, void* callback, void* arg) { return true; }

PUBLIC_API bool RemoveApiCallback(uint32_t op) { return true; }

PUBLIC_API bool InitActivityCallback(void* callback, void* arg) {
  activity_prim::activity_arg = arg;
  activity_prim::activity_callback.store((activity_async_callback_t)callback,
                                         std::memory_order_release);

  rocprofiler_queue_callbacks_t queue_callbacks{};
  queue_callbacks.dispatch = activity_prim::dispatch_callback;
  rocprofiler_set_queue_callbacks(queue_callbacks, NULL);

  return true;
}

PUBLIC_API bool EnableActivityCallback(uint32_t op, bool enable) {
  if (enable) {
    activity_prim::activity_op = op;
    rocprofiler_start_queue_callbacks();
  } else {
    rocprofiler_stop_queue_callbacks();
  }
  return true;
}

struct evt_cb_entry_t {
  typedef std::pair<void*, void*> data_t;
  data_t data;
  std::mutex mutex;

  void set(const data_t& in) {
    mutex.lock();
    data = in;
    mutex.unlock();
  }
  data_t get() {
    mutex.lock();
    const data_t out = data;
    mutex.unlock();
    return out;
  }
  evt_cb_entry_t() : data{} {}
};
evt_cb_entry_t evt_cb_table[HSA_EVT_ID_NUMBER];

hsa_status_t codeobj_evt_callback(rocprofiler_hsa_cb_id_t id,
                                  const rocprofiler_hsa_callback_data_t* cb_data, void* arg) {
  const auto evt = evt_cb_table[id].get();
  activity_rtapi_callback_t evt_callback = (activity_rtapi_callback_t)evt.first;
  if (evt_callback != NULL) evt_callback(ACTIVITY_DOMAIN_HSA_EVT, id, cb_data, evt.second);
  return HSA_STATUS_SUCCESS;
}

PUBLIC_API const char* GetEvtName(uint32_t op) { return strdup("CODEOBJ"); }

PUBLIC_API bool RegisterEvtCallback(uint32_t op, void* callback, void* arg) {
  evt_cb_table[op].set({callback, arg});

  rocprofiler_hsa_callbacks_t ocb{};
  switch (op) {
    case HSA_EVT_ID_ALLOCATE:
      ocb.allocate = codeobj_evt_callback;
      break;
    case HSA_EVT_ID_DEVICE:
      ocb.device = codeobj_evt_callback;
      break;
    case HSA_EVT_ID_MEMCOPY:
      ocb.memcopy = codeobj_evt_callback;
      break;
    case HSA_EVT_ID_SUBMIT:
      ocb.submit = codeobj_evt_callback;
      break;
    case HSA_EVT_ID_KSYMBOL:
      ocb.ksymbol = codeobj_evt_callback;
      break;
    case HSA_EVT_ID_CODEOBJ:
      ocb.codeobj = codeobj_evt_callback;
      break;
    default:
      fatal("invalid activity opcode");
  }
  rocprofiler_set_hsa_callbacks(ocb, NULL);

  return true;
}

PUBLIC_API bool RemoveEvtCallback(uint32_t op) {
  rocprofiler_hsa_callbacks_t ocb{};
  rocprofiler_set_hsa_callbacks(ocb, NULL);
  return true;
}
}  // extern "C"
