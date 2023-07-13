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

#include "ctrl/run_kernel.h"
#include "ctrl/test_aql.h"
#include "ctrl/test_hsa.h"
#include "rocprofiler/rocprofiler.h"
#include "dummy_kernel/dummy_kernel.h"
#include "simple_convolution/simple_convolution.h"
#include "util/hsa_rsrc_factory.h"
#include "util/test_assert.h"

// print time
void print_sys_time(clockid_t clock_id, rocprofiler_time_id_t time_id) {
  HsaTimer::timestamp_t value_ns = 0;
  HsaTimer::timestamp_t error_ns = 0;
  HsaTimer::timestamp_t timestamp = 0;

  timespec tm_val;
  clock_gettime(clock_id, &tm_val);
  HsaTimer::timestamp_t tm_val_ns = HsaTimer::timespec_to_ns(tm_val);

  timestamp = HsaRsrcFactory::Instance().TimestampNs();
  hsa_status_t status = rocprofiler_get_time(time_id, timestamp, &value_ns, &error_ns);
  TEST_STATUS(status == HSA_STATUS_SUCCESS);

  HsaTimer::timestamp_t timestamp1 = timestamp;
  HsaTimer::timestamp_t value_ns1 = value_ns;

  printf("time-id(%d) ts_ns(%lu) orig_ns(%lu) time_ns(%lu) err_ns(%lu)\n", (int)time_id, timestamp,
         tm_val_ns, value_ns, error_ns);

  sleep(1);

  timestamp = HsaRsrcFactory::Instance().TimestampNs();
  status = rocprofiler_get_time(time_id, timestamp, &value_ns, NULL);
  TEST_STATUS(status == HSA_STATUS_SUCCESS);
  status = rocprofiler_get_time(time_id, timestamp, NULL, &error_ns);
  TEST_STATUS(status == HSA_STATUS_SUCCESS);
  status = rocprofiler_get_time(time_id, timestamp, NULL, NULL);
  TEST_STATUS(status == HSA_STATUS_SUCCESS);

  HsaTimer::timestamp_t timestamp2 = timestamp;
  HsaTimer::timestamp_t value_ns2 = value_ns;

  printf("time-id(%d) ts_ns(%lu) orig_ns(%lu) time_ns(%lu) err_ns(%lu)\n", (int)time_id, timestamp,
         tm_val_ns, value_ns, error_ns);
  printf("ts-diff(%lu) tm-diff(%lu)\n", timestamp2 - timestamp1, value_ns2 - value_ns1);
}

// print profiler features
void print_features(rocprofiler_feature_t* feature, uint32_t feature_count) {
  for (rocprofiler_feature_t* p = feature; p < feature + feature_count; ++p) {
    std::cout << (p - feature) << ": " << p->name;
    switch (p->data.kind) {
      case ROCPROFILER_DATA_KIND_INT64:
        std::cout << std::dec << " result64 (" << p->data.result_int64 << ")" << std::endl;
        break;
      case ROCPROFILER_DATA_KIND_DOUBLE:
        std::cout << " result64 (" << p->data.result_double << ")" << std::endl;
        break;
      case ROCPROFILER_DATA_KIND_BYTES: {
        const char* ptr = reinterpret_cast<const char*>(p->data.result_bytes.ptr);
        uint64_t size = 0;
        for (unsigned i = 0; i < p->data.result_bytes.instance_count; ++i) {
          size = *reinterpret_cast<const uint64_t*>(ptr);
          const char* data = ptr + sizeof(size);
          std::cout << std::endl;
          std::cout << std::hex << "  data (" << (void*)data << ")" << std::endl;
          std::cout << std::dec << "  size (" << size << ")" << std::endl;
          ptr = data + size;
        }
        break;
      }
      default:
        std::cout << "result kind (" << p->data.kind << ")" << std::endl;
        TEST_ASSERT(false);
    }
  }
}

void read_features(uint32_t n, rocprofiler_t* context, rocprofiler_feature_t* feature,
                   const unsigned feature_count) {
  std::cout << "read features" << std::endl;
  hsa_status_t status = rocprofiler_read(context, n);
  TEST_STATUS(status == HSA_STATUS_SUCCESS);
  std::cout << "read issue" << std::endl;
  status = rocprofiler_get_data(context, n);
  TEST_STATUS(status == HSA_STATUS_SUCCESS);
  status = rocprofiler_get_metrics(context);
  TEST_STATUS(status == HSA_STATUS_SUCCESS);
  print_features(feature, feature_count);
}

int main() {
  bool ret_val = false;
  // HSA status
  hsa_status_t status = HSA_STATUS_ERROR;
  // Profiling context
  rocprofiler_t* context = NULL;
  // Profiling properties
  rocprofiler_properties_t properties;

  // Profiling feature objects
  const unsigned feature_count = 6;
  rocprofiler_feature_t feature[feature_count];
  // PMC events
  memset(feature, 0, sizeof(feature));
  feature[0].kind = ROCPROFILER_FEATURE_KIND_METRIC;
  feature[0].name = "GRBM_COUNT";
  feature[1].kind = ROCPROFILER_FEATURE_KIND_METRIC;
  feature[1].name = "GRBM_GUI_ACTIVE";
  feature[2].kind = ROCPROFILER_FEATURE_KIND_METRIC;
  feature[2].name = "GPUBusy";
  feature[3].kind = ROCPROFILER_FEATURE_KIND_METRIC;
  feature[3].name = "SQ_WAVES";
  feature[4].kind = ROCPROFILER_FEATURE_KIND_METRIC;
  feature[4].name = "SQ_INSTS_VALU";
  feature[5].kind = ROCPROFILER_FEATURE_KIND_METRIC;
  feature[5].name = "VALUInsts";
//  feature[6].kind = ROCPROFILER_FEATURE_KIND_METRIC;
//  feature[6].name = "TCC_HIT_sum";
//  feature[7].kind = ROCPROFILER_FEATURE_KIND_METRIC;
//  feature[7].name = "TCC_MISS_sum";
//  feature[8].kind = ROCPROFILER_FEATURE_KIND_METRIC;
//  feature[8].name = "WRITE_SIZE";
//  feature[8].kind = ROCPROFILER_FEATURE_KIND_METRIC;
//  feature[8].name = "TCC_EA_WRREQ_sum";
//  feature[9].kind = ROCPROFILER_FEATURE_KIND_METRIC;
//  feature[9].name = "TCC_EA_WRREQ_64B_sum";
#if 0
  // Tracing parameters
  const unsigned parameter_count = 2;
  rocprofiler_parameter_t parameters[parameter_count];
  feature[2].name = "THREAD_TRACE";
  feature[2].kind = ROCPROFILER_FEATURE_KIND_TRACE;
  feature[2].parameters = parameters;
  feature[2].parameter_count = parameter_count;
  parameters[0].parameter_name = HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_MASK;
  parameters[0].value = 0;
  parameters[1].parameter_name = HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK;
  parameters[1].value = 0;
#endif

  // Instantiate HSA resources
  HsaRsrcFactory::Create();

  // Getting GPU device info
  const AgentInfo* agent_info = NULL;
  if (HsaRsrcFactory::Instance().GetGpuAgentInfo(0, &agent_info) == false) abort();

  // Creating the queues pool
  const unsigned queue_count = 16;
  hsa_queue_t* queue[queue_count];
  for (unsigned queue_ind = 0; queue_ind < queue_count; ++queue_ind) {
    if (HsaRsrcFactory::Instance().CreateQueue(agent_info, 128, &queue[queue_ind]) == false)
      abort();
  }
  hsa_queue_t* prof_queue = queue[0];

  // Creating profiling context
  properties = {};
  properties.queue = prof_queue;
  status = rocprofiler_open(agent_info->dev_id, feature, feature_count, &context,
                            ROCPROFILER_MODE_STANDALONE, &properties);
  TEST_STATUS(status == HSA_STATUS_SUCCESS);

  // Test initialization
  TestHsa::HsaInstantiate();

  // Dispatching profiled kernel n-times to collect all counter groups data
  const unsigned group_n = 0;
  status = rocprofiler_start(context, group_n);
  TEST_STATUS(status == HSA_STATUS_SUCCESS);
  std::cout << "start" << std::endl;

  for (unsigned ind = 0; ind < 3; ++ind) {
#if 1
    const unsigned queue_ind = ind % queue_count;
    hsa_queue_t* prof_queue = queue[queue_ind];
    // ret_val = RunKernel<DummyKernel, TestAql>(0, NULL, NULL, prof_queue);
    ret_val = RunKernel<SimpleConvolution, TestAql>(0, NULL, NULL, prof_queue);
    std::cout << "run kernel, queue " << queue_ind << std::endl;
#else
    sleep(3);
#endif
    read_features(group_n, context, feature, feature_count);
  }

  // Stop counters
  status = rocprofiler_stop(context, group_n);
  TEST_STATUS(status == HSA_STATUS_SUCCESS);
  std::cout << "stop" << std::endl;

  // Finishing cleanup
  // Deleting profiling context will delete all allocated resources
  status = rocprofiler_close(context);
  TEST_STATUS(status == HSA_STATUS_SUCCESS);

  print_sys_time(CLOCK_REALTIME, ROCPROFILER_TIME_ID_CLOCK_REALTIME);
  sleep(1);
  print_sys_time(CLOCK_MONOTONIC, ROCPROFILER_TIME_ID_CLOCK_MONOTONIC);

  return (ret_val) ? 0 : 1;
}
