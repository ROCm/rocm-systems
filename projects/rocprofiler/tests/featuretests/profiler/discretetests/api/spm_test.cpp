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
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <rocprofiler.h>

#include <cassert>
#include <functional>
#include <iostream>
#include <thread>
#include <vector>
#include <stdlib.h>

#include "utils/test_utils.h"

#ifdef NDEBUG
#define HIP_ASSERT(x) x
#else
#define HIP_ASSERT(x) (assert((x) == hipSuccess))
#endif


#define WIDTH 1024
#define HEIGHT 1024

#define NUM (WIDTH * HEIGHT)

#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16
#define THREADS_PER_BLOCK_Z 1


/** \mainpage ROC Profiler API Test
 *
 * \section introduction Introduction
 *
 * The goal of this test is to test ROCmTools APIs to collect SPM.
 *
 * A simple vectoradd_float kernel is launched and the SPM results are printed
 * as console output
 */


// function to check spm tracing API status
auto CheckApi = [](rocprofiler_status_t status) {
  if (status != ROCPROFILER_STATUS_SUCCESS) {
    std::cout << "ROCmTools API Error" << std::endl;
  }
  assert(status == ROCPROFILER_STATUS_SUCCESS);
};


void FlushCallback(const rocprofiler_record_header_t* record,
                   const rocprofiler_record_header_t* end_record, rocprofiler_session_id_t session_id,
                   rocprofiler_buffer_id_t buffer_id) {
  while (record < end_record) {
    if (!record)
      break;
    else if (record->kind == ROCPROFILER_SPM_RECORD) {
      const rocprofiler_record_spm_t* spm_record =
          reinterpret_cast<const rocprofiler_record_spm_t*>(record);
      size_t name_length;
      int se_num = 4;
      // iterate over each shader engine
      for (int i = 0; i < se_num; i++) {
        printf("\n\n-------------- shader_engine %d --------------\n\n", i);
        rocprofiler_record_se_spm_data_t se_spm = spm_record->shader_engine_data[i];
        for (int i = 0; i < 32; i++) {
          printf("%04x\n", se_spm.counters_data[i].value);
        }
      }
    }
    CheckApi(rocprofiler_next_record(record, &record, session_id, buffer_id));
  }
}

__global__ void vectoradd_float(float* __restrict__ a, const float* __restrict__ b,
                                const float* __restrict__ c, int width, int height) {
  int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

  int i = y * width + x;
  if (i < (width * height)) {
    a[i] = b[i] + c[i];
  }
}

int LaunchVectorAddKernel() {
  float* hostA;
  float* hostB;
  float* hostC;

  float* deviceA;
  float* deviceB;
  float* deviceC;

  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);
  std::cout << " System minor " << devProp.minor << std::endl;
  std::cout << " System major " << devProp.major << std::endl;
  std::cout << " agent prop name " << devProp.name << std::endl;


  std::cout << "hip Device prop succeeded " << std::endl;


  int i;
  int errors;

  hostA = (float*)malloc(NUM * sizeof(float));
  hostB = (float*)malloc(NUM * sizeof(float));
  hostC = (float*)malloc(NUM * sizeof(float));

  // initialize the input data
  for (i = 0; i < NUM; i++) {
    hostB[i] = (float)i;
    hostC[i] = (float)i * 100.0f;
  }

  HIP_ASSERT(hipMalloc((void**)&deviceA, NUM * sizeof(float)));
  HIP_ASSERT(hipMalloc((void**)&deviceB, NUM * sizeof(float)));
  HIP_ASSERT(hipMalloc((void**)&deviceC, NUM * sizeof(float)));

  HIP_ASSERT(hipMemcpy(deviceB, hostB, NUM * sizeof(float), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(deviceC, hostC, NUM * sizeof(float), hipMemcpyHostToDevice));


  for (int i = 0; i < 20; i++)
    hipLaunchKernelGGL(vectoradd_float,
                       dim3(WIDTH / THREADS_PER_BLOCK_X, HEIGHT / THREADS_PER_BLOCK_Y),
                       dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0, deviceA, deviceB,
                       deviceC, WIDTH, HEIGHT);


  HIP_ASSERT(hipMemcpy(hostA, deviceA, NUM * sizeof(float), hipMemcpyDeviceToHost));

  // verify the results
  errors = 0;
  for (i = 0; i < NUM; i++) {
    if (hostA[i] != (hostB[i] + hostC[i])) {
      errors++;
    }
  }
  if (errors != 0) {
    printf("FAILED: %d errors\n", errors);
  } else {
    printf("PASSED!\n");
  }

  HIP_ASSERT(hipFree(deviceA));
  HIP_ASSERT(hipFree(deviceB));
  HIP_ASSERT(hipFree(deviceC));

  free(hostA);
  free(hostB);
  free(hostC);

  // hipResetDefaultAccelerator();

  return errors;
}


int main(int argc, char** argv) {
  // inititalize rocmtools
  hsa_init();
  CheckApi(rocprofiler_initialize());

  // spm trace collection parameters
  rocprofiler_session_id_t session_id;
  rocprofiler_spm_parameter_t spm_parameters;
  const char* counter_name = "SQ_WAVES";
  spm_parameters.counters_names = &counter_name;
  spm_parameters.counters_count = 1;
  spm_parameters.gpu_agent_id = NULL;
  // spm_parameters.cpu_agent_id = NULL;
  spm_parameters.sampling_rate = 10000;
  // create a session
  CheckApi(rocprofiler_create_session(ROCPROFILER_KERNEL_REPLAY_MODE, &session_id));

  // create a buffer to hold spm trace records for each kernel launch
  rocprofiler_buffer_id_t buffer_id;
  CheckApi(rocprofiler_create_buffer(session_id, FlushCallback, 0x99999999, &buffer_id));

  // create a filter for collecting spm traces
  rocprofiler_filter_id_t filter_id;
  rocprofiler_filter_property_t property = {};
  CheckApi(rocprofiler_create_filter(session_id, ROCPROFILER_SPM_COLLECTION,
                                   rocprofiler_filter_data_t{.spm_parameters = &spm_parameters}, 1,
                                   &filter_id, property));

  // set buffer for the filter
  CheckApi(rocprofiler_set_filter_buffer(session_id, filter_id, buffer_id));

  // activating spm tracing session
  CheckApi(rocprofiler_start_session(session_id));

  // Launch a kernel
  LaunchVectorAddKernel();

  // deactivate spm tracing session
  // dump spm tracing data
  //
  CheckApi(rocprofiler_terminate_session(session_id));
  // CheckApi(rocprofiler_flush_data(session_id, buffer_id));

  // destroy session
  CheckApi(rocprofiler_destroy_session(session_id));

  // finalize spm tracing by destroying rocmtool object
  CheckApi(rocprofiler_finalize());
  hsa_shut_down();
  return 0;
}