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

/** \mainpage ROC Profiler Multi-Threaded Test Application
 *
 * \section introduction Introduction
 *
 * Test application launches an empty kernel on multiple threads.
 *
 * In subsequent tests, ROC profiler is run against this applicaiton
 * to confirm if collected contexts are valid.
 *
 */

#include <hip/hip_runtime.h>

#include <functional>
#include <thread>
#include <vector>

#include "utils/test_utils.h"

/** \mainpage ROC Profiler Test APplication
 *
 * \section introduction Introduction
 *
 * The goal of this test application is to launch an empty kernel
 * on multiple threads and multiple gpu's.
 *
 * Number of threads are caluculated based on the cores in the system
 * Number of gpus's are calculated based on the gpu's in the system
 */

// empty kernel
__global__ void kernel() {}

// launches kernel on multiple gpu's
void KernelLaunch() {
  // Multi-GPU
  int gpu_count = 0;
  hipGetDeviceCount(&gpu_count);

  for (uint32_t gpu_id = 0; gpu_id < gpu_count; gpu_id++) {
    // run empty kernel
    kernel<<<1, 1>>>();
  }
}

int main(int argc, char** argv) {
  // create as many threads as number of cores in system
  int threads_count = GetNumberOfCores();

  // create a pool of thrads
  std::vector<std::thread> threads(threads_count);

  // launch kernel on each thread
  for (int n = 0; n < threads_count; ++n) {
    threads[n] = std::thread(KernelLaunch);
  }
  // wait for all kernel launches to complete
  for (int n = 0; n < threads_count; ++n) {
    threads[n].join();
  }
}
