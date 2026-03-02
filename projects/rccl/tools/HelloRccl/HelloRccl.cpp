/*
Copyright (c) 2020 Advanced Micro Devices, Inc. All rights reserved.

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

#include <sys/socket.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <unistd.h>
#include <cstdio>
#include <string>
#include <chrono>
#include <vector>
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include "HelloRccl.hpp"


void Usage(char *argv0);
void ExecuteTest(int numIntraRank, int intraRankStartId, int numTotalRanks, ncclComm_t* comm);

int main(int argc, char **argv)
{
  if (getenv("NCCL_COMM_ID") && argc == 3) // Run in multi-process mode
  {
    int nranks   = atoi(argv[1]);
    int rank     = atoi(argv[2]);
    if (rank == 0) printf("Running in multi-process mode\n");

    // Create communicator for this rank
    ncclUniqueId commId;
    NCCL_CALL(ncclGetUniqueId(&commId));

    // Initialize communicator
    ncclComm_t comm;
    HIP_CALL(hipSetDevice(rank));
    NCCL_CALL(ncclCommInitRank(&comm, nranks, commId, rank));

    // Run the test
    ExecuteTest(1, rank, nranks, &comm);
  }
  else if (argc == 2) // Run in single-process mode
  {
    printf("Running in single-process mode\n");

    int nranks   = atoi(argv[1]);

    // Initialize communicators for each rank
    std::vector<ncclComm_t> comm(nranks);
    NCCL_CALL(ncclCommInitAll(comm.data(), nranks, NULL));

    // Run the test
    ExecuteTest(nranks, 0, nranks, comm.data());
  }
  else
  {
    Usage(argv[0]);
    return 1;
  }
  return 0;
}

void ExecuteTest(int numIntraRank, int intraRankStartId, int numTotalRanks, ncclComm_t* comm)
{
  // Test configuration settings
  int minPow        = 10;      // Starting power of 2 input size
  int maxPow        = 28;      // Ending power of 2 input size
  int numWarmups    =  3;      // Number of untimed warmup iterations
  int numIterations = 10;      // Number of timed iterations

  // Allocate GPU resources for this process
  std::vector<hipStream_t> stream(numIntraRank);
  std::vector<hipEvent_t>  startEvent(numIntraRank);
  std::vector<hipEvent_t>  stopEvent(numIntraRank);
  for (int i = 0; i < numIntraRank; i++)
  {
    HIP_CALL(hipSetDevice(intraRankStartId + i));
    HIP_CALL(hipStreamCreate(&stream[i]));
    HIP_CALL(hipEventCreate(&startEvent[i]));
    HIP_CALL(hipEventCreate(&stopEvent[i]));
  }

  if (intraRankStartId == 0)
  {
    printf("AllReduce Performance (sum of floats):\n");
    printf("%10s %10s %10s\n", "Bytes", "CpuTime(ms)", "GpuTime(ms)");
  }

  // Loop over power-of-two input sizes
  for (int power = minPow; power <= maxPow; power++)
  {
    int N = 1 << power;

    // Allocate GPU memory
    std::vector<float *>iputGpu(numIntraRank);
    std::vector<float *>oputGpu(numIntraRank);
    for (int r = 0; r < numIntraRank; r++)
    {
      HIP_CALL(hipSetDevice(intraRankStartId + r));
      HIP_CALL(hipMalloc((void **)&iputGpu[r], N * sizeof(float)));
      HIP_CALL(hipMalloc((void **)&oputGpu[r], N * sizeof(float)));
    }

    // Allocate CPU memory for input/output
    float *iputCpu = (float *)malloc(N * sizeof(float));
    float *oputCpu = (float *)malloc(N * sizeof(float));

    // Fill CPU memory with a simple pattern
    for (int i = 0; i < N; i++)
    {
      iputCpu[i] = 1.0f;
      oputCpu[i] = 0.0f;
    }

    // Copy the input from CPU memory to GPU memory
    for (int r = 0; r < numIntraRank; r++)
    {
      HIP_CALL(hipSetDevice(intraRankStartId + r));
      HIP_CALL(hipMemcpy(iputGpu[r], iputCpu, N * sizeof(float), hipMemcpyHostToDevice));
    }

    // Perform some untimed initial warmup iterations
    for (int iteration = 0; iteration < numWarmups; iteration++)
    {
      NCCL_CALL(ncclGroupStart());
      for (int r = 0; r < numIntraRank; r++)
      {
        HIP_CALL(hipSetDevice(intraRankStartId + r));
        NCCL_CALL(ncclAllReduce(iputGpu[r], oputGpu[r], N, ncclFloat, ncclSum, comm[r], stream[r]));
      }
      NCCL_CALL(ncclGroupEnd());
    }
    for (int r = 0; r < numIntraRank; r++)
      HIP_CALL(hipStreamSynchronize(stream[r]));

    // Perform timed iterations
    auto cpuStart = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < numIntraRank; r++)
      HIP_CALL(hipEventRecord(startEvent[r], stream[r]));

    for (int iteration = 0; iteration < numIterations; iteration++)
    {
      NCCL_CALL(ncclGroupStart());
      for (int r = 0; r < numIntraRank; r++)
      {
        HIP_CALL(hipSetDevice(intraRankStartId + r));
        NCCL_CALL(ncclAllReduce(iputGpu[r], oputGpu[r], N, ncclFloat, ncclSum, comm[r], stream[r]));
      }
      NCCL_CALL(ncclGroupEnd());
    }

    for (int r = 0; r < numIntraRank; r++)
      HIP_CALL(hipEventRecord(stopEvent[r], stream[r]));

    for (int r = 0; r < numIntraRank; r++)
      HIP_CALL(hipStreamSynchronize(stream[r]));

    auto cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
    double totalCpuTime = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(cpuDelta).count();

    float totalGpuTime;
    HIP_CALL(hipEventElapsedTime(&totalGpuTime, startEvent[0], stopEvent[0]));

    if (intraRankStartId == 0) printf("%10lu %10.3f %10.3f\n", N * sizeof(float), (totalCpuTime / numIterations), (totalGpuTime / numIterations));

    // Validate results
    for (int r = 0; r < numIntraRank; r++)
    {
      HIP_CALL(hipMemcpy(oputCpu, oputGpu[r], N * sizeof(float), hipMemcpyDeviceToHost));
      bool isOK = true;
      int expected = numTotalRanks;
      for (int i = 0; i < N; i++)
      {
        isOK &= (oputCpu[i] == expected);
      }
      if (!isOK)
      {
        printf("[ERROR] Rank %d Incorrect results for N = %d\n", intraRankStartId + r, N);
        NCCL_CALL(ncclCommDestroy(comm[r]));
        exit(1);
      }
    }

    // Release GPU resources
    for (int r = 0; r < numIntraRank; r++)
    {
      HIP_CALL(hipFree(oputGpu[r]));
      HIP_CALL(hipFree(iputGpu[r]));
    }
    free(iputCpu);
    free(oputCpu);
  }

  for (int r = 0; r < numIntraRank; r++)
  {
    HIP_CALL(hipStreamDestroy(stream[r]));
    HIP_CALL(hipEventDestroy(startEvent[r]));
    HIP_CALL(hipEventDestroy(stopEvent[r]));
    NCCL_CALL(ncclCommDestroy(comm[r]));
  }
}

void Usage(char *argv0)
{
  printf("Single Process Usage: %s numRanks\n", argv0);
  printf("\n");
  printf("Multi Process Usage: %s numRanks rank\n", argv0);
  printf(" - NCCL_COMM_ID must be set in order to use this\n\n");
  printf(" - To use this process as the root process you may use any of the following:\n");

  char hostname[256];
  gethostname(hostname, 256);
  printf("    export NCCL_COMM_ID=%s:12345\n", hostname);

  // Loop over linked list of interfaces
  struct ifaddrs *ifaddr;
  getifaddrs(&ifaddr);
  for (struct ifaddrs* ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next)
  {
    // Skip anything not based on IPv4 / IPv6
    int family = ifa->ifa_addr->sa_family;
    if (family != AF_INET && family != AF_INET6) continue;

    // Skip iPv6 loopback interface
    if (family == AF_INET6)
    {
      struct sockaddr_in6* sa = (struct sockaddr_in6*)(ifa->ifa_addr);
      if (IN6_IS_ADDR_LOOPBACK(&sa->sin6_addr)) continue;
    }

    socklen_t saLen = (family == AF_INET ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6));
    char host[NI_MAXHOST];
    char service[NI_MAXSERV];

    getnameinfo(ifa->ifa_addr, saLen, host, NI_MAXHOST, service, NI_MAXSERV,
                NI_NUMERICHOST|NI_NUMERICSERV);

    std::string result = std::string(host) + ":12345";
    printf("    export NCCL_COMM_ID=%s\n", result.c_str());
  }
  freeifaddrs(ifaddr);
}
