#include <cstdio>
#include <iostream>
#include <map>
#include <stack>
#include <vector>
#include <hip/hip_runtime.h>
#define NUM_LOOPS_WARMUP 10
#define NUM_LOOPS_RUN 50

#define HIP_CALL(cmd)                                                                   \
    do {                                                                                \
        hipError_t error = (cmd);                                                       \
        if (error != hipSuccess)                                                        \
        {                                                                               \
            std::cerr << "Encountered HIP error (" << hipGetErrorString(error)          \
                      << ") at line " << __LINE__ << " in file " << __FILE__ << "\n";   \
            exit(-1);                                                                   \
        }                                                                               \
    } while (0)


void EnablePeerAccess(int const deviceId, int const peerDeviceId)
{
  int canAccess;
  HIP_CALL(hipDeviceCanAccessPeer(&canAccess, deviceId, peerDeviceId));
  if (!canAccess)
  {
    printf("[ERROR] Unable to enable peer access from GPU devices %d to %d\n", peerDeviceId, deviceId);
    exit(1);
  }

  HIP_CALL(hipSetDevice(deviceId));
  hipError_t error = hipDeviceEnablePeerAccess(peerDeviceId, 0);
  if (error != hipSuccess && error != hipErrorPeerAccessAlreadyEnabled)
  {
    printf("[ERROR] Unable to enable peer to peer access from %d to %d (%s)\n",
           deviceId, peerDeviceId, hipGetErrorString(error));
    exit(1);
  }
}

typedef struct
{
  int pingIdx;
  int pingMemType;
  int pingMethod;

  int pongIdx;
  int pongMemType;
  int pongMethod;
} LatencyTest;

typedef struct
{
  int32_t   direction;
  int32_t   method;
  uint64_t  numCycles;

  uint64_t* localFlag;
  uint64_t* remoteFlag;

  uint64_t* pingTime;
  uint64_t* pongTime;
} BlockParam;

enum
{
  PING = 0,
  PONG = 1
};

enum
{
  GPU = 0,
  CPU = 1
};

enum
{
  LRRW = 0, //Local Read Remote Write
  LWRR = 1  //Local Write Remote Read
};

__global__ void LatencyKernel(BlockParam* params)
{
  BlockParam& param = params[blockIdx.x];

  // TODO
  if (param.direction == PING)
  {
    #pragma unroll
    for (uint32_t i = 1; i <= NUM_LOOPS_WARMUP; i++) {
      __atomic_store_n(param.remoteFlag, i, __ATOMIC_RELAXED);
      while (__atomic_load_n(param.localFlag, __ATOMIC_RELAXED) != i);
    }
    uint64_t start_time = wall_clock64();
    #pragma unroll
    for (uint32_t i = NUM_LOOPS_WARMUP + 1; i <= NUM_LOOPS_WARMUP + NUM_LOOPS_RUN; i++) {
      __atomic_store_n(param.remoteFlag, i, __ATOMIC_RELAXED);
      while (__atomic_load_n(param.localFlag, __ATOMIC_RELAXED) != i);
    }
    uint64_t end_time = wall_clock64();
    *param.pingTime = end_time - start_time;

    //printf("PING Time = %u !\n", *param.pingTime);
  }
  else
  {
    #pragma unroll
    for (uint32_t i = 1; i <= NUM_LOOPS_WARMUP; i++) {
      while (__atomic_load_n(param.localFlag, __ATOMIC_RELAXED) != i);
      __atomic_store_n(param.remoteFlag, i, __ATOMIC_RELAXED);
    }
    uint64_t start_time = wall_clock64();
    #pragma unroll
    for (uint32_t i = NUM_LOOPS_WARMUP + 1; i <= NUM_LOOPS_WARMUP + NUM_LOOPS_RUN; i++) {
      while (__atomic_load_n(param.localFlag, __ATOMIC_RELAXED) != i);
      __atomic_store_n(param.remoteFlag, i, __ATOMIC_RELAXED);
    }
    uint64_t end_time = wall_clock64();
    *param.pongTime = end_time - start_time;
    //printf("PONG Time = %u !\n", *param.pongTime);
  }
}


void RunKernel(int deviceId, int numBlocks, BlockParam* paramsGpu)
{
  HIP_CALL(hipSetDevice(deviceId));
  LatencyKernel<<<numBlocks, 1>>>(paramsGpu);
  HIP_CALL(hipDeviceSynchronize());
}

void RunLatencyTests(std::vector<LatencyTest> const& latencyTests, int numGpus)
{
  hipDeviceProp_t prop[2];
  // Prepare CPU copy of parameters
  std::map<int32_t, std::vector<BlockParam>> paramsCpuMap;
  for (auto test : latencyTests)
  {
    // Create flags for both devices
    uint64_t* pingFlag;
    uint64_t* pongFlag;
    uint64_t* pingTime;
    uint64_t* pongTime;

    HIP_CALL(hipSetDevice(test.pingIdx));
    HIP_CALL(hipGetDeviceProperties(&prop[0], test.pingIdx));
    HIP_CALL(hipExtMallocWithFlags((void**)&pingFlag, sizeof(int64_t), hipDeviceMallocUncached));
    HIP_CALL(hipExtMallocWithFlags((void**)&pingTime, sizeof(int64_t), hipDeviceMallocUncached));
    HIP_CALL(hipSetDevice(test.pongIdx));
    HIP_CALL(hipExtMallocWithFlags((void**)&pongFlag, sizeof(int64_t), hipDeviceMallocUncached));
    HIP_CALL(hipExtMallocWithFlags((void**)&pongTime, sizeof(int64_t), hipDeviceMallocUncached));

    // Setup PING parameters
    BlockParam p;
    p.direction  = PING;
    p.method     = test.pingMethod;
    p.localFlag  = pingFlag;
    p.remoteFlag = pongFlag;
    p.pingTime = pingTime;    
    paramsCpuMap[test.pingIdx].push_back(p);

    // Setup PONG parameters
    p.direction  = PONG;
    p.method     = test.pongMethod;
    p.localFlag  = pongFlag;
    p.remoteFlag = pingFlag;
    p.pongTime = pongTime;
    paramsCpuMap[test.pongIdx].push_back(p);
  }

  // Create GPU copy of parameters
  std::map<int32_t, std::pair<int32_t, BlockParam*>> paramsGpuMap;
  for (auto const& paramPair : paramsCpuMap)
  {
    int deviceId = paramPair.first;
    std::vector<BlockParam> const& paramCpu = paramPair.second;

    HIP_CALL(hipSetDevice(deviceId));

    BlockParam* paramGpu;
    HIP_CALL(hipMalloc((void**)&paramGpu, paramCpu.size() * sizeof(BlockParam)));
    HIP_CALL(hipMemcpy(paramGpu, paramCpu.data(), paramCpu.size() * sizeof(BlockParam), hipMemcpyHostToDevice));
    HIP_CALL(hipDeviceSynchronize());

    paramsGpuMap[deviceId] = std::make_pair(paramCpu.size(), paramGpu);
  }

  // Launch kernels in separate threads
  std::stack<std::thread> threads;
  for (auto const& paramPair : paramsGpuMap)
  {
    int deviceId         = paramPair.first;
    int numBlocks        = paramPair.second.first;
    BlockParam* paramGpu = paramPair.second.second;

    threads.push(std::thread(RunKernel, deviceId, numBlocks, paramGpu));
  }
  while(!threads.empty())
  {
    threads.top().join();
    threads.pop();
  }

  int testCount = 0;
  double vega_gpu_rtc_freq;
  // Print results
  std::cout << "Test:  Dev <-> Dev     PingMethod     PongMethod        PingTime       PongTime" <<std::endl;
  for (auto test : latencyTests)
  //for (int i = 0; i < numGpus; i++)
  {
    int pingId = test.pingIdx;
    int pongId = test.pongIdx;

    vega_gpu_rtc_freq = (strncmp(prop[0].gcnArchName, "gfx942", 6) == 0 || strncmp(prop[0].gcnArchName, "gfx950", 6) == 0) ? 1.0E8 : 2.5E7;
    std::vector<BlockParam> paramsCpuPing = paramsCpuMap[pingId];
    std::vector<BlockParam> paramsCpuPong = paramsCpuMap[pongId];
    BlockParam pi = paramsCpuPing.back();
    BlockParam po = paramsCpuPong.back();
    double pingTime = double(*pi.pingTime) * 1e6 / NUM_LOOPS_RUN / vega_gpu_rtc_freq / 2;
    double pongTime = double(*po.pongTime) * 1e6 / NUM_LOOPS_RUN / vega_gpu_rtc_freq / 2;
    std::cout << testCount++ << "     G" << pingId <<  "     G" << pongId << "           " << pi.method << "           " << po.method << "             "<<  pingTime << "             " << pongTime << std::endl;
    paramsCpuPing.pop_back();
    paramsCpuPong.pop_back();    

  }

  for (auto const& paramPair : paramsGpuMap)
  {
    int deviceId         = paramPair.first;
    int numBlocks        = paramPair.second.first;
    BlockParam* paramGpu = paramPair.second.second;
    HIP_CALL(hipSetDevice(deviceId));
    HIP_CALL(hipFree(paramGpu));
  }
}

int main(int argc, char **argv)
{
  int numGpus;
  HIP_CALL(hipGetDeviceCount(&numGpus));

  // Enable peer to peer for each GPU
  for (int i = 0; i < numGpus; i++)
    for (int j = 0; j < numGpus; j++)
      if (i != j) EnablePeerAccess(i, j);


  std::vector<LatencyTest> latencyTests;

  for (int i = 0; i < numGpus; i++)
    for (int j = 0; j < numGpus; j++)
    {
      LatencyTest t;
      t.pingIdx = i;
      t.pingMethod = 0;
      t.pongIdx = j;
      t.pongMethod = 0;
      latencyTests.push_back(t);
    }

  RunLatencyTests(latencyTests, numGpus);

  return 0;
}

