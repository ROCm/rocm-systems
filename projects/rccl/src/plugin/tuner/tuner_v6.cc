/*************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2023, Meta Platforms, Inc. and affiliates.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <dlfcn.h>
#include <stdlib.h>
#include "debug.h"
#include "nccl_tuner.h"
#include "checks.h"

static ncclTuner_v6_t* ncclTuner_v6;
static ncclTuner_t ncclTuner;

// v6 plugins may keep the constants pointer init() gets until finalize(), as they could when v6
// was the core ABI. So the v6 copy lives with the communicator's tuner context, not on the stack.
struct ncclTunerContext_v6 {
  void* pluginContext;
  ncclTunerConstants_v6_t constants;
};

static ncclResult_t ncclTuner_getCollInfo(void* context, ncclFunc_t collType, size_t nBytes, int numPipeOps,
                                          float** collCostTable, int numAlgo, int numProto, int regBuff,
                                          int* nChannels) {
  return ncclTuner_v6->getCollInfo(((ncclTunerContext_v6*)context)->pluginContext, collType, nBytes, numPipeOps,
                                   collCostTable, numAlgo, numProto, regBuff, nChannels);
}

static ncclResult_t ncclTuner_getChunkSize(void* context, ncclFunc_t collType, size_t nBytes, int algo, int proto,
                                           int nChannels, size_t* chunkSize) {
  return ncclTuner_v6->getChunkSize(((ncclTunerContext_v6*)context)->pluginContext, collType, nBytes, algo, proto,
                                    nChannels, chunkSize);
}

static ncclResult_t ncclTuner_finalize(void* context) {
  ncclTunerContext_v6* ctx = (ncclTunerContext_v6*)context;
  if (ctx == nullptr) return ncclTuner_v6->finalize(nullptr); // init() failed or never ran
  ncclResult_t ret = ncclTuner_v6->finalize(ctx->pluginContext);
  free(ctx);
  return ret;
}

static ncclResult_t ncclTuner_init(void** context, uint64_t commId, size_t nRanks, size_t nNodes,
                                   ncclDebugLogger_t logfn, ncclNvlDomainInfo_v5_t* nvlDomainInfo,
                                   ncclTunerConstants_t* constants) {
  ncclTunerContext_v6* ctx = (ncclTunerContext_v6*)calloc(1, sizeof(*ctx));
  if (ctx == nullptr) return ncclSystemError;
  ncclTunerConstants_v7_to_v5(&ctx->constants, constants);
  ncclResult_t ret =
    ncclTuner_v6->init(&ctx->pluginContext, commId, nRanks, nNodes, logfn, nvlDomainInfo, &ctx->constants);
  if (ret != ncclSuccess) {
    free(ctx);
    return ret;
  }
  ncclTunerConstants_v5_to_v7(constants, &ctx->constants);
  *context = ctx;
  return ncclSuccess;
}

ncclTuner_t* getNcclTuner_v6(void* lib) {
  ncclTuner_v6 = (ncclTuner_v6_t*)dlsym(lib, "ncclTunerPlugin_v6");
  if (ncclTuner_v6) {
    ncclTuner.name = ncclTuner_v6->name;
    ncclTuner.init = ncclTuner_init;
    ncclTuner.getCollInfo = ncclTuner_getCollInfo;
    ncclTuner.getChunkSize = ncclTuner_v6->getChunkSize ? ncclTuner_getChunkSize : nullptr;
    ncclTuner.finalize = ncclTuner_finalize;
    INFO(NCCL_INIT | NCCL_TUNING, "TUNER/Plugin: Using %s (v6)", ncclTuner_v6->name);
    return &ncclTuner;
  }
  return NULL;
}
