/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

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

#include <iostream>
#include <cstdio>
#include <string>
#include <chrono>
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

#define HIP_CALL(cmd)                                                 \
  do {                                                                \
    hipError_t error = (cmd);                                         \
    if (error != hipSuccess)                                          \
    {                                                                   \
      std::cout << "Encountered HIP error (" << hipGetErrorString(error) << ") at line " \
                << __LINE__ << " in file " << __FILE__ << "\n";         \
      exit(-1);                                                         \
    }                                                                   \
  } while (0)

#define NCCL_CALL(cmd) \
  do { \
    ncclResult_t error = (cmd);                 \
    if (error != ncclSuccess)                   \
    {                                           \
      std::cout << "Encountered NCCL error (" << ncclGetErrorString(error) << ") at line " \
                << __LINE__ << " in file " << __FILE__ << "\n";         \
      exit(-1);                                                         \
    }                                                                   \
  } while (0)

int main(int argc, char **argv)
{
  int nranks;
  HIP_CALL(hipGetDeviceCount(&nranks));

  // Initialize communicators for each rank
  ncclComm_t comm[nranks];
  NCCL_CALL(ncclCommInitAll(comm, nranks, NULL));

  // Allocate GPU resources
  hipStream_t stream[nranks];
  int* iputCpu[nranks];
  int* iputGpu[nranks];
  int* oputGpu[nranks];
  int* oputCpu[nranks];
  int* pattern;
  int* expected;

  int maxN = (1<<24);

  expected = (int*)calloc(maxN, sizeof(int));
  for (int r = 0; r < nranks; r++)
  {
    HIP_CALL(hipSetDevice(r));
    HIP_CALL(hipStreamCreate(&stream[r]));
    HIP_CALL(hipMalloc((void **)&iputGpu[r], maxN * sizeof(int)));
    HIP_CALL(hipMalloc((void **)&oputGpu[r], maxN * sizeof(int)));

    iputCpu[r] = (int*) malloc(maxN * sizeof(int));
    oputCpu[r] = (int*) malloc(maxN * sizeof(int));
    pattern    = (int*) malloc(maxN * sizeof(int));

    for (int i = 0; i < maxN; i++)
    {
      iputCpu[r][i] = (r * 235 + i) % 2057;
      oputCpu[r][i] = 0;
      expected[i] += iputCpu[r][i];

      pattern[i] = -1 - i;
    }

    HIP_CALL(hipMemcpy(iputGpu[r], iputCpu[r], maxN * sizeof(int), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(oputGpu[r], oputCpu[r], maxN * sizeof(int), hipMemcpyHostToDevice));
  }

  int numWarmups    = 3;
  int numIterations = 5;

  hipGraph_t graphs[nranks];
  hipGraphExec_t graphExec[nranks];

  printf("%12s", "NumBytes");
  for (int usingGraphs = 0; usingGraphs <= 1; usingGraphs++)
  {
    printf("%12s", "Setup");
    for (int i = 1; i <= numIterations; ++i)
      printf("%11s%d", usingGraphs ? "Graph" : "NoGraph", i);
    printf("%12s", "Avg");
  }
  printf("%12s\n", "Speedup");

  for (int N = 1; N <= maxN; N *= 2)
  {
    printf("%12lu", N * sizeof(int));

    double average[2] = {};
    for (int usingGraphs = 0; usingGraphs <= 1; usingGraphs++)
    {
      auto setupStart = std::chrono::high_resolution_clock::now();
      if (usingGraphs)
      {
        for (int r = 0; r < nranks; ++r)
        {
          HIP_CALL(hipSetDevice(r));
          HIP_CALL(hipStreamBeginCapture(stream[r], hipStreamCaptureModeThreadLocal));
        }

        NCCL_CALL(ncclGroupStart());
        for (int r = 0; r < nranks; ++r)
        {
          HIP_CALL(hipSetDevice(r));
          NCCL_CALL(ncclAllReduce(iputGpu[r], oputGpu[r], N, ncclInt, ncclSum, comm[r], stream[r]));
        }
        NCCL_CALL(ncclGroupEnd());

        for (int r = 0; r < nranks; ++r)
        {
          //HIP_CALL(hipSetDevice(r));
          HIP_CALL(hipStreamEndCapture(stream[r], &graphs[r]));
        }

        // Instantiating graphs
        for (int r = 0; r < nranks; ++r)
        {
          HIP_CALL(hipSetDevice(r));
          HIP_CALL(hipGraphInstantiate(&graphExec[r], graphs[r], NULL, NULL, 0));
        }
      }
      auto setupDelta = std::chrono::high_resolution_clock::now() - setupStart;
      double setupTime = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(setupDelta).count();
      printf("%12.3f", setupTime);

      // Perform iterations
      average[usingGraphs] = 0;
      for (int iteration = -numWarmups; iteration < numIterations; ++iteration)
      {
        auto cpuStart = std::chrono::high_resolution_clock::now();
        if (usingGraphs)
        {
          for (int r = 0; r < nranks; r++)
          {
            HIP_CALL(hipSetDevice(r));
            HIP_CALL(hipGraphLaunch(graphExec[r], stream[r]));
          }
        }
        else
        {
          NCCL_CALL(ncclGroupStart());
          for (int r = 0; r < nranks; ++r)
          {
            HIP_CALL(hipSetDevice(r));
            NCCL_CALL(ncclAllReduce(iputGpu[r], oputGpu[r], N, ncclInt, ncclSum, comm[r], stream[r]));
          }
          NCCL_CALL(ncclGroupEnd());
        }

        for (int r = 0; r < nranks; r++)
          HIP_CALL(hipStreamSynchronize(stream[r]));

        auto cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
        double iterationTime = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(cpuDelta).count();

        // Check result and reset
        bool isCorrect = true;
        for (int r = 0; r < nranks && isCorrect; r++)
        {
          HIP_CALL(hipMemcpy(oputCpu[r], oputGpu[r], N * sizeof(int), hipMemcpyDeviceToHost));
          for (int i = 0; i < N; i++)
          {
            if (oputCpu[r][i] != expected[i])
            {
              isCorrect = false;
              printf("ERROR: Expected: %d Output %d at Index %d\n", expected[i], oputCpu[r][i], i);
              exit(1);
            }
          }
          // Fill output with input for testing reasons
          HIP_CALL(hipMemcpy(oputGpu[r], pattern, N * sizeof(int), hipMemcpyHostToDevice));
        }

        if (iteration >= 0)
        {
          printf("%12.3f", iterationTime); fflush(stdout);
          average[usingGraphs] += iterationTime;
        }
      }
      average[usingGraphs] /= numIterations;
      printf("%12.3f", average[usingGraphs]);

      for (int r = 0; r < nranks; r++)
      {
        HIP_CALL(hipSetDevice(r));
        HIP_CALL(hipGraphDestroy(graphs[r]));
        HIP_CALL(hipGraphExecDestroy(graphExec[r]));
      }
    }
    printf("%12.3f\n", average[0] / average[1]);
    fflush(stdout);
  }
  return 0;
}
