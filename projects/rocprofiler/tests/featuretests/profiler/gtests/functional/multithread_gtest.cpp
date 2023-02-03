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
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <rocprofiler.h>

#include <functional>
#include <iostream>
#include <thread>
#include <vector>

#include "src/utils/helper.h"
#include "utils/test_utils.h"

/** \mainpage ROC Profiler API Test
 *
 * \section introduction Introduction
 *
 * The goal of this test is to check ROCmTools APIs from multiple threads
 * and verify if each API succeeds and multiple contexts are collected and
 * verified.
 *
 * An empty kernel is launched on multiple threads and profiling context is
 * collected and verified from each thread.
 */

// empty kernel
__global__ void kernel() {}

class ProfilerAPITest : public ::testing::Test {
 protected:
  // function to check profiler API status
  static void CheckApi(rocprofiler_status_t status) {
    ASSERT_EQ(status, ROCPROFILER_STATUS_SUCCESS);
  };

  // launches an empty kernel in profiler context
  static void KernelLaunch() {
    // run empty kernel
    kernel<<<1, 1>>>();
    hipDeviceSynchronize();
  }

  // callback function to dump profiler data
  static void FlushCallback(const rocprofiler_record_header_t* record,
                            const rocprofiler_record_header_t* end_record,
                            int64_t session_id) {
    while (record < end_record) {
      if (!record) break;
      if (record->kind == ROCPROFILER_PROFILER_RECORD) {
        const rocprofiler_record_profiler_t* profiler_record =
            reinterpret_cast<const rocprofiler_record_profiler_t*>(record);
        size_t name_length;
        CheckApi(rocprofiler_query_kernel_info_size(
            ROCPROFILER_KERNEL_NAME, profiler_record->kernel_id, &name_length));
        const char* kernel_name_c =
            static_cast<const char*>(malloc(name_length * sizeof(char)));
        CheckApi(rocprofiler_query_kernel_info(ROCPROFILER_KERNEL_NAME,
                                             profiler_record->kernel_id,
                                             &kernel_name_c));
        int gpu_index = profiler_record->gpu_id.handle;
        uint64_t start_time = profiler_record->timestamps.begin.value;
        uint64_t end_time = profiler_record->timestamps.end.value;

        // Check for each kernel if endtime > starttime
        ASSERT_GT(end_time, start_time);

        // Check for each kernel name_length is +ve
        ASSERT_GT(name_length, 0);

        // Check kernel name
        ASSERT_EQ(
            rocmtools::truncate_name(rocmtools::cxx_demangle(kernel_name_c)),
            "kernel");
      }
      CheckApi(rocprofiler_next_record(record, &record));
    }
  }
};

TEST_F(ProfilerAPITest, WhenRunningMultipleThreadsProfilerAPIsWorkFine) {
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
  CheckApi(rocprofiler_create_session(ROCPROFILER_APPLICATION_REPLAY_MODE,
                                    &session_id));
  CheckApi(rocprofiler_add_session_mode(session_id, ROCPROFILER_ASYNC_FLUSH,
                                      ROCPROFILER_COUNTERS_COLLECTION));
  CheckApi(rocprofiler_set_session_async_callback(
      session_id, ROCPROFILER_COUNTERS_COLLECTION,
      rocprofiler_session_buffer_size_t{0x8000}, FlushCallback,
      rocprofiler_flush_buffer_interval_t{0}));
  rocprofiler_filter_t filter{ROCPROFILER_FILTER_PROFILER_COUNTER_NAMES,
                            &counters[0],
                            rocprofiler_filter_data_count_t{counters.size()}};
  CheckApi(rocprofiler_session_set_filters(ROCPROFILER_COUNTERS_COLLECTION,
                                         &filter, rocprofiler_filters_count_t{1},
                                         session_id));

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
  // dump profiler data
  CheckApi(rocprofiler_flush_data(session_id));

  // deactivate session
  CheckApi(rocprofiler_terminate_session(session_id));

  // destroy session
  CheckApi(rocprofiler_destroy_session(session_id));

  // finalize profiler by destroying rocmtool object
  CheckApi(rocprofiler_finalize());
}
