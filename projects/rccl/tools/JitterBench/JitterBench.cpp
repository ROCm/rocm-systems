/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#include <chrono>
#include <cstdio>
#include <iostream>
#include <thread>
#include <vector>
#include <numa.h>
#include <omp.h>
#include <unistd.h>

#include "Common.hpp"
#include "Compatibility.hpp"
#include "GetClosestNumaNode.hpp"
#include "Timeline.hpp"

#ifdef MPI_SUPPORT
#include <mpi.h>
#endif

struct SyncData
{
  uint64_t cpuStart;
  uint64_t cpuStop;
  int32_t  xccId;
  uint64_t gpuStart;
  uint64_t gpuStop;
};

enum
{
  HOST_START      = 0,
  HOST_RETURN     = 1,
  DEV_START       = 2,
  HOST_ABORT      = 3,
  DEV_STOP        = 4,
  HOST_STOP       = 5,
  KERNEL_CPUTIME  = 6,
  KERNEL_GPUTIME  = 7,
  KERNEL_TIMEDIFF = 8,
  NUM_COLUMNS     = 9
} Columns;

bool printCol[NUM_COLUMNS] =
{
  false,
  false,
  true,
  false,
  true,
  false,
  true,
  true,
  true
};

#define LOAD(VAR)       __atomic_load_n((VAR),         __ATOMIC_ACQUIRE)
#define STORE(DST, SRC) __atomic_store_n((DST), (SRC), __ATOMIC_RELEASE)

__global__ void SyncKernel(uint64_t* cpuTime, uint32_t* abortFlag, SyncData* syncData)
{
  SyncData sd;
  // Only first thread in threadblock participates
  if (threadIdx.x != 0) return;

  // Collect timestamp upon kernel entry
  sd.cpuStart = LOAD(cpuTime);
  sd.gpuStart = wall_clock64();

  // Wait for abort flag to be modified
  while (!LOAD(abortFlag));

  // Collect timestamps after abort flag
  sd.cpuStop = LOAD(cpuTime);
  sd.gpuStop = wall_clock64();

  // Save timestamps
  GetXccId(sd.xccId);
  syncData[blockIdx.x] = sd;
}

void SetNumaNode(int numaId)
{
  // Move CPU thread to targeted NUMA node
  if (numa_run_on_node(numaId))
  {
    printf("[ERROR] Unable to migrate to NUMA node %d\n", numaId);
    exit(1);
  }

  // Set memory to allocate on targeted NUMA node
  numa_set_preferred(numaId);
}

void UpdateCpuTime(int const useNuma, int const numaId, uint64_t* cpuTimestamp, bool* abortThread)
{
  if (useNuma) SetNumaNode(numaId);
  while (!LOAD(abortThread))
  {
    // Unroll to increase update vs abort check ratio
    #pragma unroll
    for (int i = 0; i < 64; i++)
      STORE(cpuTimestamp, std::chrono::steady_clock::now().time_since_epoch().count());
  }
}

void HostMalloc(void** pinnedHostPtr, size_t size)
{
#if !defined(__NVCC__)
  HIP_CALL(hipHostMalloc(pinnedHostPtr, size, hipHostMallocNumaUser));
#else
  HIP_CALL(hipHostMalloc(pinnedHostPtr, size));
#endif
  memset(*pinnedHostPtr, 0, size);
}

int main(int argc, char **argv)
{
  // Initialize MPI (if supported) and check for NUMA support
#ifdef MPI_SUPPORT
  MPI_Init(&argc, &argv);
#endif
  if (numa_available() == -1)
  {
    printf("[ERROR] NUMA library not supported. Check to see if libnuma has been installed on this system\n");
    exit(1);
  }

  int numAvailableGpus;
  HIP_CALL(hipGetDeviceCount(&numAvailableGpus));

  // Figure out how many GPUs total / which GPU this process is responsible for
  int numUsedGpus, numTotalGpus, rank;
#ifdef MPI_SUPPORT
  numUsedGpus = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numTotalGpus);
  if (numTotalGpus > numAvailableGpus)
  {
    if (rank == 0) printf("[ERROR] Machine only has %d devices but %d ranks requested\n", numAvailableGpus, numTotalGpus);
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
  if (rank == 0)
  {
    printf("Running MPI version with %d ranks\n", numTotalGpus);
  }
#else
  numUsedGpus = numTotalGpus = numAvailableGpus;
  printf("Running OMP version with %d ranks\n", numTotalGpus);
  rank = 0;
#endif

  // Collect arguments from commandline or environment variable
  #define GETARG(IDX, STR, DEFAULT) \
    (argc > IDX ? atoi(argv[IDX]) : (getenv(STR) ? atoi(getenv(STR)) : DEFAULT))

  int  numBlocks        = GETARG(1, "NUM_BLOCKS",        4);
  int  blockSize        = GETARG(2, "BLOCKSIZE",        64);
  int  numUpdateThreads = GETARG(3, "NUM_UPDATERS",      0);
  int  useNuma          = GETARG(4, "USE_NUMA",          1);
  int  numIterations    = GETARG(5, "NUM_ITERATIONS",   10);
  int  numWarmups       = GETARG(6, "NUM_WARMUPS",    1000);
  int  numSleepUsec     = GETARG(7, "SLEEP_USEC",      100);
  int  totalIterations  = numWarmups + numIterations;

  int  verbose          = (getenv("VERBOSE"    ) ? atoi(getenv("VERBOSE"))     : 1);
  int  launchMode       = (getenv("LAUNCH_MODE") ? atoi(getenv("LAUNCH_MODE")) : 1);

  if (numUpdateThreads == 0) numUpdateThreads = numUsedGpus;

  // Print off configuration and machine information
  if (rank == 0)
  {
    printf("NUM_BLOCKS     = %8d\n", numBlocks);
    printf("BLOCKSIZE      = %8d\n", blockSize);
    printf("NUM_UPDATERS   = %8d\n", numUpdateThreads);
    printf("USE_NUMA       = %8d\n", useNuma);
    printf("NUM_ITERATIONS = %8d\n", numIterations);
    printf("NUM_WARMUPS    = %8d\n", numWarmups);
    printf("SLEEP_USEC     = %8d\n", numSleepUsec);
  }

  char archName[100];
  std::vector<double> uSecPerCycle(numUsedGpus);
  for (int i = 0; i < numUsedGpus; i++)
  {
    hipDeviceProp_t prop;
    HIP_CALL(hipGetDeviceProperties(&prop, i + rank));
    sscanf(prop.gcnArchName, "%[^:]", archName);
    int wallClockMhz;
    HIP_CALL(hipDeviceGetAttribute(&wallClockMhz, hipDeviceAttributeWallClockRate, i));
    uSecPerCycle[i] = 1000.0 / wallClockMhz;
    if (verbose) printf("GPU %02d: %s: Closest NUMA: %d usecPerWallClockCycle %g\n", i + rank, archName, GetClosestNumaNode(i + rank), uSecPerCycle[i]);
  }

  typedef typename std::ratio_multiply<std::chrono::steady_clock::period,std::mega>::type MicroSec;
  //printf("std::chrono::steady_clock precision: %8.3f usec\n", static_cast<double>(MicroSec::num)/MicroSec::den);

  // Allocate per-update-thread resources and start update threads
  bool abortUpdateThreads = false;
  std::vector<uint64_t*> cpuTimestamps(numUpdateThreads);
  std::vector<std::thread> updateThreads;
  for (int i = 0; i < numUpdateThreads; i++)
  {
    int numaId = GetClosestNumaNode(i + rank);
    HIP_CALL(hipSetDevice(i + rank));
    if (useNuma) SetNumaNode(numaId);

    HostMalloc((void**)&cpuTimestamps[i], 256);  // Allocate larger buffer to avoid multiple timestamps on same cacheline

    // Launch update thread
    updateThreads.push_back(std::thread(UpdateCpuTime, useNuma, numaId, cpuTimestamps[i], &abortUpdateThreads));
  }

  // Allocate per-GPU resources
  std::vector<SyncData*>   syncDataGpu(numUsedGpus);
  std::vector<uint32_t*>   abortFlags(numUsedGpus);
  std::vector<hipStream_t> streams(numUsedGpus);
  for (int i = 0; i < numUsedGpus; i++)
  {
    HIP_CALL(hipSetDevice(i + rank));
    if (useNuma) SetNumaNode(GetClosestNumaNode(i + rank));

    HIP_CALL(hipMalloc((void**)&syncDataGpu[i], totalIterations * numBlocks * sizeof(SyncData)));
    HostMalloc((void**)&abortFlags[i], 256); // Allocate larger buffer to avoid multiple abort flags on same cacheline

    HIP_CALL(hipStreamCreate(&streams[i]));
  }

  // Allocate per-iteration resources
  std::vector<std::vector<uint64_t>>  hostStartTimes(numTotalGpus, std::vector<uint64_t>(totalIterations));
  std::vector<std::vector<uint64_t>> hostReturnTimes(numTotalGpus, std::vector<uint64_t>(totalIterations));
  std::vector<std::vector<uint64_t>>  hostAbortTimes(numTotalGpus, std::vector<uint64_t>(totalIterations));
  std::vector<std::vector<uint64_t>>   hostStopTimes(numTotalGpus, std::vector<uint64_t>(totalIterations));

#ifndef MPI_SUPPORT
  #pragma omp parallel num_threads(numTotalGpus)
#endif
  {
#ifdef MPI_SUPPORT
    int deviceId = rank;
    int localIdx = 0;
#else
    int deviceId = omp_get_thread_num();
    int localIdx = deviceId;
#endif
    HIP_CALL(hipSetDevice(deviceId));
    if (useNuma) SetNumaNode(GetClosestNumaNode(deviceId));

    uint64_t* cpuTimestamp = cpuTimestamps[localIdx % numUpdateThreads];
    uint32_t* abortFlag    = abortFlags[localIdx];

    for (int iteration = 0; iteration < totalIterations; iteration++)
    {
      // Prepare for this iteration
      // Clear abort flag
      STORE(abortFlag, 0);
      SyncData* syncData = syncDataGpu[localIdx] + (iteration * numBlocks);

      // Wait for all threads to arrive before launching all kernels
#ifdef MPI_SUPPORT
      MPI_Barrier(MPI_COMM_WORLD);
#else
      #pragma omp barrier
#endif

      // Launch kernel
      uint64_t cpuStart = std::chrono::steady_clock::now().time_since_epoch().count();
      if (launchMode == 0)
      {
        SyncKernel<<<numBlocks, blockSize, 0, streams[localIdx]>>>(cpuTimestamp, abortFlag, syncData);
      }
      else
      {
        hipLaunchKernelGGL(SyncKernel, numBlocks, blockSize, 0, streams[localIdx], cpuTimestamp, abortFlag, syncData);
      }
      uint64_t cpuReturn = std::chrono::steady_clock::now().time_since_epoch().count();

      // Busy wait performs more accurately than usleep / sleep_for
      while (std::chrono::steady_clock::now().time_since_epoch().count() - cpuStart < numSleepUsec * 1000);
      STORE(abortFlag, 1);
      uint64_t cpuAbort = std::chrono::steady_clock::now().time_since_epoch().count();

      // Wait for kernel to finish
      HIP_CALL(hipStreamSynchronize(streams[localIdx]));
      uint64_t cpuStop = std::chrono::steady_clock::now().time_since_epoch().count();

      // Store values (after all timings to avoid false sharing)
      hostStartTimes [deviceId][iteration] = cpuStart;
      hostReturnTimes[deviceId][iteration] = cpuReturn;
      hostAbortTimes [deviceId][iteration] = cpuAbort;
      hostStopTimes  [deviceId][iteration] = cpuStop;

#ifdef MPI_SUPPORT
      MPI_Barrier(MPI_COMM_WORLD);
#else
      #pragma omp barrier
#endif
    }
  }

  // Stop all the update threads
  STORE(&abortUpdateThreads, true);
  for (auto& t : updateThreads)
    t.join();

  std::vector<std::vector<SyncData>> syncDataCpu(numTotalGpus, std::vector<SyncData>(totalIterations * numBlocks));
  for (int i = 0; i < numUsedGpus; i++)
  {
    HIP_CALL(hipMemcpy(syncDataCpu[i+rank].data(), syncDataGpu[i], totalIterations * numBlocks * sizeof(SyncData), hipMemcpyDeviceToHost));
  }

  std::vector<std::vector<double>> singleMinDiff(numTotalGpus, std::vector<double>(NUM_COLUMNS, std::numeric_limits<double>::max()));
  std::vector<std::vector<double>> singleSumDiff(numTotalGpus, std::vector<double>(NUM_COLUMNS, 0));
  std::vector<std::vector<double>> singleMaxDiff(numTotalGpus, std::vector<double>(NUM_COLUMNS, std::numeric_limits<double>::min()));
  std::vector<double> multiMinDiff(NUM_COLUMNS, std::numeric_limits<double>::max());
  std::vector<double> multiSumDiff(NUM_COLUMNS, 0);
  std::vector<double> multiMaxDiff(NUM_COLUMNS, std::numeric_limits<double>::min());
  std::vector<TimelineData> timelineData;
  char buff[1000];

#ifdef MPI_SUPPORT
  // Collect results from every rank
  if (rank == 0)
  {
    for (int deviceId = 1; deviceId < numTotalGpus; deviceId++)
    {
      MPI_Recv( hostStartTimes[deviceId].data(), totalIterations * sizeof(uint64_t), MPI_BYTE, deviceId, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(hostReturnTimes[deviceId].data(), totalIterations * sizeof(uint64_t), MPI_BYTE, deviceId, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv( hostAbortTimes[deviceId].data(), totalIterations * sizeof(uint64_t), MPI_BYTE, deviceId, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(  hostStopTimes[deviceId].data(), totalIterations * sizeof(uint64_t), MPI_BYTE, deviceId, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(syncDataCpu[deviceId].data(), totalIterations * numBlocks * sizeof(SyncData), MPI_BYTE, deviceId, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
  else
  {
    MPI_Send( hostStartTimes[rank].data(), totalIterations * sizeof(uint64_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
    MPI_Send(hostReturnTimes[rank].data(), totalIterations * sizeof(uint64_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
    MPI_Send( hostAbortTimes[rank].data(), totalIterations * sizeof(uint64_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
    MPI_Send(  hostStopTimes[rank].data(), totalIterations * sizeof(uint64_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
    MPI_Send(syncDataCpu[rank].data(), totalIterations * numBlocks * sizeof(SyncData), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
    goto end;
  }
#endif

  for (int iteration = 1; iteration <= numIterations; iteration++)
  {
    // Ignore warmup iterations
    int iter = iteration + numWarmups - 1;

    // Figure out which timestamp is "earliest" to use as origin for this iteration
    uint64_t origin = hostStartTimes[0][iter];
    for (int gpu = 1; gpu < numTotalGpus; gpu++)
      origin = std::min(origin, hostStartTimes[gpu][iter]);

    if (verbose)
    {
      printf("Iteration %d: (All times in usec)\n", iteration);
      printf("------------------------------------------------------------------------------------------------------------------------------------------\n");
      printf("| GPU | BLOCK | XCC | START(CPU) | RETURN(CPU)| START(GPU) | ABORT(CPU) | STOP (GPU) | STOP (CPU) | Kernel(CPU)| Kernel(GPU)|   AbsDiff  |\n");
    }

    std::vector<double>  multiMinTime(NUM_COLUMNS, std::numeric_limits<double>::max());
    std::vector<double>  multiMaxTime(NUM_COLUMNS, std::numeric_limits<double>::min());

    for (int gpu = 0; gpu < numTotalGpus; gpu++)
    {
      std::vector<double> times(NUM_COLUMNS);
      times[HOST_START]  = ( hostStartTimes[gpu][iter] - origin) / 1000.0;
      times[HOST_RETURN] = (hostReturnTimes[gpu][iter] - origin) / 1000.0;
      times[HOST_ABORT]  = ( hostAbortTimes[gpu][iter] - origin) / 1000.0;
      times[HOST_STOP]   = (  hostStopTimes[gpu][iter] - origin) / 1000.0;

      TimelineData td;
      sprintf(buff, "Iteration %d GPU %02d (CPU)", iteration, gpu); td.rowLabel = buff;
      td.barLabel  = "Launch (";
      sprintf(buff, "%.3f to %.3f", times[HOST_START], times[HOST_RETURN]); td.toolTip = buff;
      td.startTime = times[HOST_START];
      td.stopTime  = times[HOST_RETURN];
      timelineData.push_back(td);

      td.barLabel  = "Pause";
      sprintf(buff, "%.3f to %.3f", times[HOST_RETURN], times[HOST_ABORT]); td.toolTip = buff;
      td.startTime = times[HOST_RETURN];
      td.stopTime  = times[HOST_ABORT];
      timelineData.push_back(td);

      td.barLabel  = "Sync";
      sprintf(buff, "%.3f to %.3f", times[HOST_ABORT], times[HOST_STOP]); td.toolTip = buff;
      td.startTime = times[HOST_ABORT];
      td.stopTime  = times[HOST_STOP];
      timelineData.push_back(td);

      std::vector<double> singleMinTime(NUM_COLUMNS, std::numeric_limits<double>::max());
      std::vector<double> singleMaxTime(NUM_COLUMNS, std::numeric_limits<double>::min());
      for (int block = 0; block < numBlocks; block++)
      {
        int blockIdx = iter * numBlocks + block;
        int xccId    = syncDataCpu[gpu][blockIdx].xccId;

        times[DEV_START]       = (syncDataCpu[gpu][blockIdx].cpuStart - origin) / 1000.0;
        times[DEV_STOP]        = (syncDataCpu[gpu][blockIdx].cpuStop  - origin) / 1000.0;
        times[KERNEL_CPUTIME]  = times[DEV_STOP] - times[DEV_START];
        times[KERNEL_GPUTIME]  = (syncDataCpu[gpu][blockIdx].gpuStop - syncDataCpu[gpu][blockIdx].gpuStart) * uSecPerCycle[gpu];
        times[KERNEL_TIMEDIFF] = fabs(times[KERNEL_CPUTIME] - times[KERNEL_GPUTIME]);

        for (int col = 0; col < NUM_COLUMNS; col++)
        {
          singleMinTime[col] = std::min(singleMinTime[col], times[col]);
          singleMaxTime[col] = std::max(singleMaxTime[col], times[col]);
           multiMinTime[col] = std::min( multiMinTime[col], times[col]);
           multiMaxTime[col] = std::max( multiMaxTime[col], times[col]);
        }

        if (verbose)
        {
          printf("| %3d |  %3d  | %3d |", gpu, block, xccId);
          for (auto x : times) printf(" %10.3f |", x);
          printf("\n");
        }

        sprintf(buff, "Iteration %d GPU %02d (GPU)", iteration, gpu); td.rowLabel = buff;
        sprintf(buff, "Block %02d", block); td.barLabel = buff;
        sprintf(buff, "%.3f to %.3f", times[DEV_START], times[DEV_STOP]); td.toolTip = buff;
        td.startTime = times[DEV_START];
        td.stopTime  = times[DEV_STOP];
        timelineData.push_back(td);
      }

      for (int col = 0; col < NUM_COLUMNS; col++)
      {
        double const diff = singleMaxTime[col] - singleMinTime[col];
        singleMinDiff[gpu][col] = std::min(singleMinDiff[gpu][col], diff);
        singleSumDiff[gpu][col] += diff;
        singleMaxDiff[gpu][col] = std::max(singleMaxDiff[gpu][col], diff);
      }

      if (verbose)
      {
        printf("| %3d | MAX ABS DIFF|", gpu);
        for (int col = 0; col < NUM_COLUMNS; col++)
          printCol[col] ? printf(" %10.3f |", singleMaxTime[col] - singleMinTime[col]) : printf("            |");
        printf("\n");
      }
    }
    for (int col = 0; col < NUM_COLUMNS; col++)
    {
      double const diff = multiMaxTime[col] - multiMinTime[col];
      multiMinDiff[col] = std::min(multiMinDiff[col], diff);
      multiSumDiff[col] += diff;
      multiMaxDiff[col] = std::max(multiMaxDiff[col], diff);
    }

    if (verbose)
    {
      printf("------------------------------------------------------------------------------------------------------------------------------------------\n");
      printf("| ALL |         MIN |"); for (auto x : multiMinTime) printf(" %10.3f |", x); printf("\n");
      printf("| ALL |         MAX |"); for (auto x : multiMaxTime) printf(" %10.3f |", x); printf("\n");
      printf("| ALL |        DIFF |"); for (int col = 0; col < NUM_COLUMNS; col++) printf(" %10.3f |", multiMaxTime[col] - multiMinTime[col]); printf("\n");
    }
  }

  printf("==========================================================================================================================================\n");
  printf("| SUMMARY (All iter)| START(CPU) | RETURN(CPU)| START(GPU) | ABORT(CPU) | STOP (GPU) | STOP (CPU) | Kernel(CPU)| Kernel(GPU)|   AbsDiff  |\n");
  printf("==========================================================================================================================================\n");
  for (int gpu = 0; gpu < numTotalGpus; gpu++)
  {
    printf("|   GPU %02d DIFF MIN |", gpu);
    for (int col = 0; col < NUM_COLUMNS; col++)
      printCol[col] ? printf(" %10.3f |", singleMinDiff[gpu][col]) : printf("            |");
    printf("\n");
  }
  for (int gpu = 0; gpu < numTotalGpus; gpu++)
  {
    printf("|   GPU %02d DIFF AVG |", gpu);
    for (int col = 0; col < NUM_COLUMNS; col++)
      printCol[col] ? printf(" %10.3f |", singleSumDiff[gpu][col] / numIterations) : printf("            |");
    printf("\n");
  }
  for (int gpu = 0; gpu < numTotalGpus; gpu++)
  {
    printf("|   GPU %02d DIFF MAX |", gpu);
    for (int col = 0; col < NUM_COLUMNS; col++)
      printCol[col] ? printf(" %10.3f |", singleMaxDiff[gpu][col]) : printf("            |");
    printf("\n");
  }
  printf("==========================================================================================================================================\n");
  printf("| ALL GPUs DIFF MIN |"); for (auto x : multiMinDiff) printf(" %10.3f |", x); printf("\n");
  printf("| ALL GPUs DIFF AVG |"); for (auto x : multiSumDiff) printf(" %10.3f |", x / numIterations); printf("\n");
  printf("| ALL GPUs DIFF MAX |"); for (auto x : multiMaxDiff) printf(" %10.3f |", x); printf("\n");

  sprintf(buff, "timeline_%dx%s_%dx%dblockSize_%dCUTs_Numa%d_Sleep%d.html", numTotalGpus, archName, numBlocks, blockSize, numUpdateThreads, useNuma, numSleepUsec);
  printf("Timeline exported to %s\n", buff);
  ExportToTimeLine(buff, "Device", "Call", timelineData);

#ifdef MPI_SUPPORT
end:
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
#endif
  return 0;
}
