/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 ************************************************************************/

#include "cpu_kernel_launcher.h"
#include "cpu_kernel_internal.h"
#include "cpu_dev_comm_mirror.h"

#include "checks.h"
#include "param.h"
#include "profiler.h"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstring>
#include <thread>
#include <vector>

RCCL_PARAM(CpuKernelEnable, "CPU_KERNEL_ENABLE", 0);
RCCL_PARAM(CpuKernelThreads, "CPU_KERNEL_THREADS", 0);

bool rcclCpuKernelEnabled() {
  return rcclParamCpuKernelEnable() != 0;
}

bool rcclCpuKernelPlanSupported(struct ncclKernelPlan* plan) {
  if (plan == nullptr) return false;
  if (plan->isCeColl || plan->isSymColl) return false;
  if (plan->persistent) return false;
  if (plan->workStorageType == ncclDevWorkStorageTypePersistent) return false;
  return true;
}

struct rcclCpuLaunchContext {
  struct ncclComm* comm;
  struct ncclKernelPlan* plan;
  std::vector<char> argsBuffer;
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
  struct ncclKernelComm* hostComm = nullptr;
  NCCLCHECK(rcclCpuMirrorDevComm(comm, &hostComm));

  int maxParallel = static_cast<int>(rcclParamCpuKernelThreads());
  if (maxParallel <= 0) maxParallel = static_cast<int>(ctx->gridDimX);

  std::vector<std::thread> threads;
  threads.reserve(ctx->gridDimX);
  std::vector<ncclResult_t> results(ctx->gridDimX, ncclSuccess);
  auto* args = reinterpret_cast<struct ncclDevKernelArgs*>(ctx->argsBuffer.data());

  auto runBlock = [&](int blockId) {
    results[blockId] = rcclCpuExecuteBlock(comm, hostComm, args, blockId,
                                           static_cast<int>(ctx->blockDimX), comm->WarpSize);
  };

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

  rcclCpuReleaseCommMirror();

  for (int i = 0; i < static_cast<int>(ctx->gridDimX); i++) {
    if (results[i] != ncclSuccess) return results[i];
  }
  return ncclSuccess;
}

static void HIPRT_CB rcclCpuKernelHostCallback(void* userData) {
  auto* ctx = static_cast<rcclCpuLaunchContext*>(userData);
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
  NCCLCHECK(ncclProfilerStartKernelLaunchEvent(plan, launchStream));

  auto* ctx = new rcclCpuLaunchContext();
  ctx->comm = comm;
  ctx->plan = plan;
  ctx->gridDimX = gridDimX;
  ctx->blockDimX = blockDimX;
  NCCLCHECK(rcclCpuPrepareArgs(comm, plan, ctx));

  CUDACHECK(cudaLaunchHostFunc(launchStream, rcclCpuKernelHostCallback, ctx));

  comm->lastStream = launchStream;
  comm->lastStreamValid = true;

  NCCLCHECK(ncclProfilerStopKernelLaunchEvent(plan));
  return ncclSuccess;
}
