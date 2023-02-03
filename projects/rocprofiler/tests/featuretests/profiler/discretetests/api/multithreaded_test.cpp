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

/** \mainpage ROC Profiler API Test
 *
 * \section introduction Introduction
 *
 * The goal of this test is to test ROCmTools APIs from multiple threads
 * and verify if each API succeeds and multiple contexts are collected and
 * printed.
 *
 * An empty kernel is launched on multiple threads and profiling context is
 * collected and printed from each thread.
 */


// function to check profiler API status
auto CheckApi = [](rocprofiler_status_t status) {
  if (status != ROCPROFILER_STATUS_SUCCESS) {
    std::cout << "ROCmTools API Error" << std::endl;
  }
  assert(status == ROCPROFILER_STATUS_SUCCESS);
};

// empty kernel
__global__ void kernel() { printf("empty kernel\n"); }

// callback function to dump profiler data
void FlushCallback(const rocprofiler_record_header_t* record,
                   const rocprofiler_record_header_t* end_record, rocprofiler_session_id_t session_id,
                   rocprofiler_buffer_id_t buffer_id) {
  while (record < end_record) {
    if (!record) break;
    if (record->kind == ROCPROFILER_PROFILER_RECORD) {
      const rocprofiler_record_profiler_t* profiler_record =
          reinterpret_cast<const rocprofiler_record_profiler_t*>(record);
      size_t name_length;
      CheckApi(rocprofiler_query_kernel_info_size(ROCPROFILER_KERNEL_NAME, profiler_record->kernel_id,
                                                &name_length));
      const char* kernel_name_c = static_cast<const char*>(malloc(name_length * sizeof(char)));
      CheckApi(rocprofiler_query_kernel_info(ROCPROFILER_KERNEL_NAME, profiler_record->kernel_id,
                                           &kernel_name_c));
      int gpu_index = profiler_record->gpu_id.handle;
      uint64_t start_time = profiler_record->timestamps.begin.value;
      uint64_t end_time = profiler_record->timestamps.end.value;
      printf(
          "Kernel Info:\n\tGPU Index: %d\n\tKernel Name: %s\n\tStart "
          "Time: "
          "%lu\n\tEnd Time: %lu\n",
          gpu_index, kernel_name_c, start_time, end_time);
    }
    CheckApi(rocprofiler_next_record(record, &record, session_id, buffer_id));
  }
}

// launches an empty kernel in profiler context
void KernelLaunch() {
  // run empty kernel
  kernel<<<1, 1>>>();
  hipDeviceSynchronize();
}

int main(int argc, char** argv) {
  // Get the system cores
  int num_cpu_cores = GetNumberOfCores();

  // create as many threads as number of cores in system
  std::vector<std::thread> threads(num_cpu_cores);

  // inititalize profiler by creating rocmtool object
  CheckApi(rocprofiler_initialize());

  // Counter Collection with timestamps
  rocprofiler_session_id_t session_id;
  std::vector<const char*> counters;
  counters.emplace_back("SQ_WAVES");

  CheckApi(rocprofiler_create_session(ROCPROFILER_KERNEL_REPLAY_MODE, &session_id));

  rocprofiler_buffer_id_t buffer_id;
  CheckApi(rocprofiler_create_buffer(session_id, FlushCallback, 0x9999, &buffer_id));

  rocprofiler_filter_id_t filter_id;
  rocprofiler_filter_property_t property = {};
  CheckApi(rocprofiler_create_filter(session_id, ROCPROFILER_COUNTERS_COLLECTION,
                                   rocprofiler_filter_data_t{.counters_names = &counters[0]},
                                   counters.size(), &filter_id, property));

  CheckApi(rocprofiler_set_filter_buffer(session_id, filter_id, buffer_id));

  // activating profiler session
  CheckApi(rocprofiler_start_session(session_id));

  // launch kernel on each thread
  for (int n = 0; n < num_cpu_cores; ++n) {
    threads[n] = std::thread(KernelLaunch);
  }

  // wait for all kernel launches to complete
  for (int n = 0; n < num_cpu_cores; ++n) {
    threads[n].join();
  }

  // deactivate session
  CheckApi(rocprofiler_terminate_session(session_id));

  // dump profiler data
  CheckApi(rocprofiler_flush_data(session_id, buffer_id));

  // destroy session
  CheckApi(rocprofiler_destroy_session(session_id));

  // finalize profiler by destroying rocmtool object
  CheckApi(rocprofiler_finalize());
  return 0;
}
