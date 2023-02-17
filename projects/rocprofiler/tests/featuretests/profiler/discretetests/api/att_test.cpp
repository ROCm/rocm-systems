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
#include <rocprofiler.h>

#include <cassert>
#include <functional>
#include <iostream>
#include <thread>
#include <vector>

#include "utils/test_utils.h"

#ifdef NDEBUG
#define HIP_ASSERT(x) x
#else
#define HIP_ASSERT(x) (assert((x)==hipSuccess))
#endif


#define WIDTH     1024
#define HEIGHT    1024

#define NUM       (WIDTH*HEIGHT)

#define THREADS_PER_BLOCK_X  16
#define THREADS_PER_BLOCK_Y  16
#define THREADS_PER_BLOCK_Z  1



/** \mainpage ROC Profiler API Test
 *
 * \section introduction Introduction
 *
 * The goal of this test is to test ROCProfiler APIs to collect ATT traces.
 *
 * A simple vectoradd_float kernel is launched and the trace results are printed 
 * as console output
 */


// function to check att tracing API status
auto CheckApi = [](rocprofiler_status_t status) {
  if (status != ROCPROFILER_STATUS_SUCCESS) {
    std::cout << "ROCProfiler API Error" << std::endl;
  }
  assert(status == ROCPROFILER_STATUS_SUCCESS);
};


// callback function to dump att tracing data
void FlushCallback(const rocprofiler_record_header_t* record,
                   const rocprofiler_record_header_t* end_record, rocprofiler_session_id_t session_id,
                   rocprofiler_buffer_id_t buffer_id) {
  while (record < end_record) {
    if (!record) break;
    else if (record->kind == ROCPROFILER_ATT_TRACER_RECORD){
      const rocprofiler_record_att_tracer_t* att_tracer_record =
          reinterpret_cast<const rocprofiler_record_att_tracer_t*>(record);
      size_t name_length;
      CheckApi(rocprofiler_query_kernel_info_size(ROCPROFILER_KERNEL_NAME, att_tracer_record->kernel_id,
                                                &name_length));
      const char* kernel_name_c = static_cast<const char*>(malloc(name_length * sizeof(char)));
      CheckApi(rocprofiler_query_kernel_info(ROCPROFILER_KERNEL_NAME, att_tracer_record->kernel_id,
                                           &kernel_name_c));
      int gpu_index = att_tracer_record->gpu_id.handle;
      printf(
          "Kernel Info:\n\tGPU Index: %d\n\tKernel Name: %s\n",
          gpu_index, kernel_name_c);

      // Get the number of shader engine traces
      int se_num = att_tracer_record->shader_engine_data_count;

      // iterate over each shader engine att trace
      for (int i = 0; i < se_num; i++){

         printf("\n\n-------------- shader_engine %d --------------\n\n", i);
         rocprofiler_record_se_att_data_t* se_att_trace = &att_tracer_record->shader_engine_data[i];
         uint32_t size = se_att_trace->buffer_size;
         const unsigned short* data_buffer_ptr = reinterpret_cast<const unsigned short*>(se_att_trace->buffer_ptr);
        
        // Print the buffer in terms of shorts (16 bits)
        for (uint32_t j = 0; j < (size / sizeof(short)); j++)
          printf("%04x\n", data_buffer_ptr[j]);

      }
      
    }
    CheckApi(rocprofiler_next_record(record, &record, session_id, buffer_id));
  }
}




__global__ void 
vectoradd_float(float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ c, int width, int height) 

  {
 
      int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
      int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

      int i = y * width + x;
      if ( i < (width * height)) {
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



  std::cout << "hip Device prop succeeded " << std::endl ;


  int i;
  int errors;

  hostA = (float*)malloc(NUM * sizeof(float));
  hostB = (float*)malloc(NUM * sizeof(float));
  hostC = (float*)malloc(NUM * sizeof(float));
  
  // initialize the input data
  for (i = 0; i < NUM; i++) {
    hostB[i] = (float)i;
    hostC[i] = (float)i*100.0f;
  }
  
  HIP_ASSERT(hipMalloc((void**)&deviceA, NUM * sizeof(float)));
  HIP_ASSERT(hipMalloc((void**)&deviceB, NUM * sizeof(float)));
  HIP_ASSERT(hipMalloc((void**)&deviceC, NUM * sizeof(float)));
  
  HIP_ASSERT(hipMemcpy(deviceB, hostB, NUM*sizeof(float), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(deviceC, hostC, NUM*sizeof(float), hipMemcpyHostToDevice));


  hipLaunchKernelGGL(vectoradd_float, 
                  dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y),
                  dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                  0, 0,
                  deviceA ,deviceB ,deviceC ,WIDTH ,HEIGHT);


  HIP_ASSERT(hipMemcpy(hostA, deviceA, NUM*sizeof(float), hipMemcpyDeviceToHost));

  // verify the results
  errors = 0;
  for (i = 0; i < NUM; i++) {
    if (hostA[i] != (hostB[i] + hostC[i])) {
      errors++;
    }
  }
  if (errors!=0) {
    printf("FAILED: %d errors\n",errors);
  } else {
      printf ("PASSED!\n");
  }

  HIP_ASSERT(hipFree(deviceA));
  HIP_ASSERT(hipFree(deviceB));
  HIP_ASSERT(hipFree(deviceC));

  free(hostA);
  free(hostB);
  free(hostC);

  //hipResetDefaultAccelerator();

  return errors;
}


int main(int argc, char** argv) {

  // inititalize ROCProfiler
  CheckApi(rocprofiler_initialize());

  // Att trace collection parameters
  rocprofiler_session_id_t session_id;
  std::vector<rocprofiler_att_parameter_t> parameters;
  parameters.emplace_back(rocprofiler_att_parameter_t{ROCPROFILER_ATT_COMPUTE_UNIT_TARGET, 0});
  parameters.emplace_back(rocprofiler_att_parameter_t{ROCPROFILER_ATT_MASK, 0x0F00});
  parameters.emplace_back(rocprofiler_att_parameter_t{ROCPROFILER_ATT_TOKEN_MASK, 0x344B});
  parameters.emplace_back(rocprofiler_att_parameter_t{ROCPROFILER_ATT_TOKEN_MASK2, 0xFFFF});

  // create a session
  CheckApi(rocprofiler_create_session(ROCPROFILER_KERNEL_REPLAY_MODE, &session_id));

  // create a buffer to hold att trace records for each kernel launch
  rocprofiler_buffer_id_t buffer_id;
  CheckApi(rocprofiler_create_buffer(session_id, FlushCallback, 0x9999, &buffer_id));

  // create a filter for collecting att traces
  rocprofiler_filter_id_t filter_id;
  rocprofiler_filter_property_t property = {};
  CheckApi(rocprofiler_create_filter(session_id, ROCPROFILER_ATT_TRACE_COLLECTION,
                                   rocprofiler_filter_data_t{.att_parameters = &parameters[0]},
                                   parameters.size(), &filter_id, property));
  
  // set buffer for the filter
  CheckApi(rocprofiler_set_filter_buffer(session_id, filter_id, buffer_id));

  // activating att tracing session
  CheckApi(rocprofiler_start_session(session_id));

  // Launch a kernel
  LaunchVectorAddKernel();

  // deactivate att tracing session
  CheckApi(rocprofiler_terminate_session(session_id));

  // dump att tracing data
  CheckApi(rocprofiler_flush_data(session_id, buffer_id));

  // destroy session
  CheckApi(rocprofiler_destroy_session(session_id));

  // finalize att tracing by destroying rocprofiler object
  CheckApi(rocprofiler_finalize());
  return 0;
}
