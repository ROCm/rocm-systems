/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 ************************************************************************/

#include "cpu_device_guard.h"
#include "cpu_kernel_launcher.h"
#include "cpu_kernel_internal.h"
#include "cpu_dev_comm_mirror.h"

#include "checks.h"
#include "debug.h"
#include "param.h"
#include "profiler.h"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace {
constexpr int kRcclCpuMaxDevices = 64;
std::mutex g_rcclCpuDevMutex[kRcclCpuMaxDevices];
}

RCCL_PARAM(CpuKernelEnable, "CPU_KERNEL_ENABLE", 0);
RCCL_PARAM(CpuKernelThreads, "CPU_KERNEL_THREADS", 0);
RCCL_PARAM(CpuKernelMultiRank, "CPU_KERNEL_MULTI_RANK", 0);

bool rcclCpuKernelEnabled() {
  return rcclParamCpuKernelEnable() != 0;
}

bool rcclCpuKernelPlanSupported(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  if (plan == nullptr) return false;
  if (comm != nullptr && comm->nRanks > 1 && rcclParamCpuKernelMultiRank() == 0) {
    return false;
  }
  if (plan->isCeColl || plan->isSymColl) return false;
  if (plan->persistent) return false;
  if (plan->workStorageType == ncclDevWorkStorageTypePersistent) return false;
  return true;
}

struct rcclCpuLaunchContext {
  struct ncclComm* comm;
  struct ncclKernelPlan* plan;
  std::vector<char> argsBuffer;
  struct rcclCpuCommMirrorState mirror;
  unsigned gridDimX;
  unsigned blockDimX;
};

static ncclResult_t rcclCpuPrepareArgs(struct ncclComm* comm, struct ncclKernelPlan* plan,
                                       struct rcclCpuLaunchContext* ctx) {
  ctx->argsBuffer.resize(plan->kernelArgsSize);
  std::memcpy(ctx->argsBuffer.data(), plan->kernelArgs, plan->kernelArgsSize);

  auto* args = reinterpret_cast<struct ncclDevKernelArgs*>(ctx->argsBuffer.data());
  if (plan->workStorageType == ncclDevWorkStorageTypeFifo) {
    args->workBuf = comm->workFifoBuf;
  }
  return ncclSuccess;
}

static ncclResult_t rcclCpuKernelExecuteSync(struct rcclCpuLaunchContext* ctx) {
  struct ncclComm* comm = ctx->comm;
  rcclCpuDeviceGuard deviceGuard(comm->cudaDev);

  struct ncclKernelComm* hostComm = nullptr;
  NCCLCHECK(rcclCpuMirrorDevComm(comm, &ctx->mirror, &hostComm));

  int maxParallel = static_cast<int>(rcclParamCpuKernelThreads());
  if (maxParallel <= 0) maxParallel = 1;

  std::vector<std::thread> threads;
  threads.reserve(ctx->gridDimX);
  std::vector<ncclResult_t> results(ctx->gridDimX, ncclSuccess);
  auto* args = reinterpret_cast<struct ncclDevKernelArgs*>(ctx->argsBuffer.data());

  auto runBlock = [&](int blockId) {
    rcclCpuDeviceGuard blockGuard(comm->cudaDev);
    results[blockId] = rcclCpuExecuteBlock(comm, &ctx->mirror, hostComm, args, blockId,
                                           static_cast<int>(ctx->blockDimX), comm->WarpSize);
  };

  // Execute on the host-callback thread when possible. Spawning std::threads that
  // read the thread_local comm mirror from another thread's TLS is unsafe on some
  // platforms; multi-block parallelism is only used when RCCL_CPU_KERNEL_THREADS>1.
  if (maxParallel <= 1 || ctx->gridDimX <= 1) {
    for (unsigned blockId = 0; blockId < ctx->gridDimX; blockId++) {
      runBlock(static_cast<int>(blockId));
    }
  } else {
    int block = 0;
    while (block < static_cast<int>(ctx->gridDimX)) {
      int wave = std::min(maxParallel, static_cast<int>(ctx->gridDimX) - block);
      threads.clear();
      for (int i = 0; i < wave; i++) {
        threads.emplace_back(runBlock, block + i);
      }
      for (auto& th : threads) th.join();
      block += wave;
    }
  }

  for (int i = 0; i < static_cast<int>(ctx->gridDimX); i++) {
    if (results[i] != ncclSuccess) return results[i];
  }
  return ncclSuccess;
}

static void HIPRT_CB rcclCpuKernelHostCallback(void* userData) {
  auto* ctx = static_cast<rcclCpuLaunchContext*>(userData);
  int dev = ctx->comm->cudaDev;
  if (dev < 0 || dev >= kRcclCpuMaxDevices) dev = 0;
  std::lock_guard<std::mutex> cpuLock(g_rcclCpuDevMutex[dev]);
  rcclCpuDeviceGuard deviceGuard(ctx->comm->cudaDev);
  ncclResult_t result = rcclCpuKernelExecuteSync(ctx);
  if (result != ncclSuccess) {
    WARN("rcclCpuKernelHostCallback failed: %s", ncclGetErrorString(result));
  }
  delete ctx;
}

ncclResult_t rcclLaunchKernelCpu(
    struct ncclComm* comm,
    struct ncclKernelPlan* plan,
    unsigned gridDimX,
    unsigned blockDimX,
    hipStream_t launchStream) {
  if (comm->rank == 0) {
    INFO(NCCL_COLL,
         "Launching RCCL collective kernel on CPU (grid=%u block=%u nWorkBatches=%d stream=%p)",
         gridDimX, blockDimX, plan->nWorkBatches, launchStream);
  }

  NCCLCHECK(ncclProfilerStartKernelLaunchEvent(plan, launchStream));

  auto* ctx = new rcclCpuLaunchContext();
  ctx->comm = comm;
  ctx->plan = plan;
  ctx->gridDimX = gridDimX;
  ctx->blockDimX = blockDimX;
  NCCLCHECK(rcclCpuPrepareArgs(comm, plan, ctx));

  struct ncclKernelComm* hostComm = nullptr;
  NCCLCHECK(rcclCpuMirrorDevComm(comm, &ctx->mirror, &hostComm));
  auto* args = reinterpret_cast<struct ncclDevKernelArgs*>(ctx->argsBuffer.data());
  if (!rcclCpuMirrorChannelsReady(comm, &ctx->mirror, args)) {
    delete ctx;
    NCCLCHECK(ncclProfilerStopKernelLaunchEvent(plan));
    return ncclInternalError;
  }

  CUDACHECK(cudaLaunchHostFunc(launchStream, rcclCpuKernelHostCallback, ctx));

  comm->lastStream = launchStream;
  comm->lastStreamValid = true;

  NCCLCHECK(ncclProfilerStopKernelLaunchEvent(plan));
  return ncclSuccess;
}
