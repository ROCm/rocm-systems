/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

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
*/

#include <sstream>
#include <string>

#include <stdio.h>

#include <inc/roctracer_hsa.h>
#include <inc/roctracer_hip.h>
#include <inc/roctracer_hcc.h>
#include <inc/ext/hsa_rt_utils.hpp>

#define PUBLIC_API __attribute__((visibility("default")))
#define CONSTRUCTOR_API __attribute__((constructor))
#define DESTRUCTOR_API __attribute__((destructor))

// Macro to check ROC-tracer calls status
#define ROCTRACER_CALL(call)                                                                       \
  do {                                                                                             \
    int err = call;                                                                                \
    if (err != 0) {                                                                                \
      std::cerr << roctracer_error_string() << std::endl << std::flush;                            \
      abort();                                                                                     \
    }                                                                                              \
  } while (0)

typedef hsa_rt_utils::Timer::timestamp_t timestamp_t;
hsa_rt_utils::Timer timer(NULL);
thread_local timestamp_t hsa_begin_timestamp = 0;
thread_local timestamp_t hip_begin_timestamp = 0;

// HSA API callback function
void hsa_api_callback(
    uint32_t domain,
    uint32_t cid,
    const void* callback_data,
    void* arg)
{
  (void)arg;

  const hsa_api_data_t* data = reinterpret_cast<const hsa_api_data_t*>(callback_data);

  if (data->phase == ACTIVITY_API_PHASE_ENTER) {
    hsa_begin_timestamp = timer.timestamp_fn_ns();
  } else {
    const timestamp_t end_timestamp = (cid == HSA_API_ID_hsa_shut_down) ? hsa_begin_timestamp : timer.timestamp_fn_ns();
    std::ostringstream os;
    os << '(' << hsa_begin_timestamp << ":" << end_timestamp << ") " << hsa_api_data_pair_t(cid, *data);
    fprintf(stdout, "%s\n", os.str().c_str());
  }
}

void hip_api_callback(
    uint32_t domain,
    uint32_t cid,
    const void* callback_data,
    void* arg)
{
  (void)arg;
  const hip_api_data_t* data = reinterpret_cast<const hip_api_data_t*>(callback_data);
  if (data->phase == ACTIVITY_API_PHASE_ENTER) {
    hsa_begin_timestamp = timer.timestamp_fn_ns();
  } else {
    const timestamp_t end_timestamp = timer.timestamp_fn_ns();
    fprintf(stdout, "(%lu:%lu) %s(", hsa_begin_timestamp, end_timestamp, roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cid, 0));
    switch (cid) {
      case HIP_API_ID_hipMemcpy:
        fprintf(stdout, "dst(%p) src(%p) size(0x%x) kind(%u)",
          data->args.hipMemcpy.dst,
          data->args.hipMemcpy.src,
          (uint32_t)(data->args.hipMemcpy.sizeBytes),
          (uint32_t)(data->args.hipMemcpy.kind));
        break;
      case HIP_API_ID_hipMalloc:
        fprintf(stdout, "ptr(0x%p) size(0x%x)",
          *(data->args.hipMalloc.ptr),
          (uint32_t)(data->args.hipMalloc.size));
        break;
      case HIP_API_ID_hipFree:
        fprintf(stdout, "ptr(%p)",
          data->args.hipFree.ptr);
        break;
      case HIP_API_ID_hipModuleLaunchKernel:
        fprintf(stdout, "kernel(\"%s\") stream(%p)",
          hipKernelNameRef(data->args.hipModuleLaunchKernel.f),
          data->args.hipModuleLaunchKernel.stream);
        break;
      default:
        break;
    }
    fprintf(stdout, ")\n"); fflush(stdout);
  }
}

// Activity tracing callback
//   hipMalloc id(3) correlation_id(1): begin_ns(1525888652762640464) end_ns(1525888652762877067)
void activity_callback(const char* begin, const char* end, void* arg) {
  const roctracer_record_t* record = reinterpret_cast<const roctracer_record_t*>(begin);
  const roctracer_record_t* end_record = reinterpret_cast<const roctracer_record_t*>(end);
  fprintf(stdout, "\tActivity records:\n"); fflush(stdout);
  while (record < end_record) {
    const char * name = roctracer_op_string(record->domain, record->op, record->kind);
    fprintf(stdout, "\t%s\tcorrelation_id(%lu) time_ns(%lu:%lu)",
      name,
      record->correlation_id,
      record->begin_ns,
      record->end_ns
    );
    if (record->domain == ACTIVITY_DOMAIN_HIP_API) {
      fprintf(stdout, " process_id(%u) thread_id(%u)",
        record->process_id,
        record->thread_id
      );
    } else if (record->domain == ACTIVITY_DOMAIN_HCC_OPS) {
      fprintf(stdout, " device_id(%d) queue_id(%lu)",
        record->device_id,
        record->queue_id
      );
    } else {
      fprintf(stderr, "Bad domain %d\n", record->domain);
      abort();
    }
    if (record->op == hc::HSA_OP_ID_COPY) fprintf(stdout, " bytes(0x%zx)", record->bytes);
    fprintf(stdout, "\n");
    fflush(stdout);
    ROCTRACER_CALL(roctracer_next_record(record, &record));
  }
}

extern "C" {
// HSA-runtime tool on-load method
PUBLIC_API bool OnLoad(HsaApiTable* table, uint64_t runtime_version, uint64_t failed_tool_count,
                       const char* const* failed_tool_names) {
  timer.init(table->core_->hsa_system_get_info_fn);
  const char* trace_domain = getenv("ROCTRACER_DOMAIN");
  const bool trace_hsa = (trace_domain == NULL) || (strncmp(trace_domain, "hsa", 3) == 0);
  const bool trace_hip = (trace_domain == NULL) || (strncmp(trace_domain, "hip", 3) == 0);

  // Enable HSA API callbacks
  if (trace_hsa) {
    ROCTRACER_CALL(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HSA_API, hsa_api_callback, NULL));
  }

  // Enable HIP API callbacks/activity
  if (trace_hip) {
    // Allocating tracing pool
    roctracer_properties_t properties{};
    properties.buffer_size = 12;
    properties.buffer_callback_fun = activity_callback;
    ROCTRACER_CALL(roctracer_open_pool(&properties));
    ROCTRACER_CALL(roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HCC_OPS));
    ROCTRACER_CALL(roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HIP_API));
    ROCTRACER_CALL(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HIP_API, hip_api_callback, NULL));
  }

  return true;
}

// HSA-runtime tool on-unload method
PUBLIC_API void OnUnload() {
  ROCTRACER_CALL(roctracer_disable_callback());
  ROCTRACER_CALL(roctracer_disable_activity());
  ROCTRACER_CALL(roctracer_close_pool());
}

}
