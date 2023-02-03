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

/** \mainpage ROC Profiler Multi Process Binary Test
 *
 * \section introduction Introduction
 *
 * The goal of this test is to test ROC profiler as a binary against a
 * multiprocess application.Test application launches an empty kernel
 * on multiple threads from both parent and child process.
 *
 * The test then parses the csv and verifies if the nuber of context collected
 *  are equal to number of threads launched in test application.
 *
 * Test also does some basic verification if counter values are non-negative
 */

#include <hip/hip_runtime.h>
#include <sys/wait.h>
#include <unistd.h>

#include <iostream>
#include <thread>
#include <vector>

#include "utils/test_utils.h"

// empty kernel
__global__ void kernel() {}

void KernelLaunch() {
  // run empty kernel
  kernel<<<1, 1>>>();
  hipDeviceSynchronize();
}

int main(int argc, char **argv) {
  // create as many threads as number of cores in system
  int num_cpu_cores = GetNumberOfCores();

  pid_t childpid = fork();

  if (childpid > 0) {  // Parent
    // create a pool of thrads
    std::vector<std::thread> threads(num_cpu_cores);
    for (int n = 0; n < num_cpu_cores / 2; ++n) {
      threads[n] = std::thread(KernelLaunch);
    }

    for (int n = 0; n < num_cpu_cores / 2; ++n) {
      threads[n].join();
    }
    // wait for child exit
    wait(NULL);

  } else if (!childpid) {  // child
                           // create a pool of thrads
    std::vector<std::thread> threads(num_cpu_cores);
    for (int n = 0; n < num_cpu_cores / 2; ++n) {
      threads[n] = std::thread(KernelLaunch);
    }

    for (int n = 0; n < num_cpu_cores / 2; ++n) {
      threads[n].join();
    }
  } else {  // failure
    return -1;
  }
}
