/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2023, Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: Apache-2.0 and BSD-3
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <stdlib.h>
#include "os.h"
#include "debug.h"
#include "nccl_tuner.h"
#include "checks.h"

static ncclTuner_v5_t* ncclTuner_v5;
static ncclTuner_t ncclTuner;

// v5 plugins may keep the constants pointer init() gets until finalize(), as they could when v5
// constants were the core's own. So the v5 copy lives with the communicator's tuner context.
struct ncclTunerContext_v5 {
  void* pluginContext;
  ncclTunerConstants_v5_t constants;
};

static ncclResult_t ncclTuner_getCollInfo(void* context, ncclFunc_t collType, size_t nBytes, int numPipeOps,
                                          float** collCostTable, int numAlgo, int numProto, int regBuff,
                                          int* nChannels) {
  NCCLCHECK(ncclTuner_v5->getCollInfo(((ncclTunerContext_v5*)context)->pluginContext, collType, nBytes, numPipeOps,
                                      collCostTable, numAlgo, numProto, regBuff, nChannels));
  return ncclSuccess;
}

static ncclResult_t ncclTuner_finalize(void* context) {
  ncclTunerContext_v5* ctx = (ncclTunerContext_v5*)context;
  if (ctx == nullptr) return ncclTuner_v5->finalize(nullptr); // init() never ran
  ncclResult_t ret = ncclTuner_v5->finalize(ctx->pluginContext);
  free(ctx);
  return ret;
}

static ncclResult_t ncclTuner_init(void** context, uint64_t commId, size_t nRanks, size_t nNodes,
                                   ncclDebugLogger_t logfn, ncclNvlDomainInfo_v5_t* nvlDomainInfo,
                                   ncclTunerConstants_t* constants) {
  ncclTunerContext_v5* ctx = (ncclTunerContext_v5*)calloc(1, sizeof(*ctx));
  if (ctx == nullptr) return ncclSystemError;
  ncclTunerConstants_v7_to_v5(&ctx->constants, constants);
  // Set even when init() fails, so finalize() gets back whatever context the plugin stored.
  *context = ctx;
  NCCLCHECK(
    ncclTuner_v5->init(&ctx->pluginContext, commId, nRanks, nNodes, logfn, nvlDomainInfo, &ctx->constants));
  ncclTunerConstants_v5_to_v7(constants, &ctx->constants);
  return ncclSuccess;
}

ncclTuner_t* getNcclTuner_v5(void* lib) {
  ncclTuner_v5 = (ncclTuner_v5_t*)ncclOsDlsym(lib, "ncclTunerPlugin_v5");
  if (ncclTuner_v5) {
    ncclTuner.name = ncclTuner_v5->name;
    ncclTuner.init = ncclTuner_init;
    ncclTuner.getCollInfo = ncclTuner_getCollInfo;
    ncclTuner.getChunkSize = NULL;
    ncclTuner.finalize = ncclTuner_finalize;
    INFO(NCCL_INIT | NCCL_TUNING, "TUNER/Plugin: Using %s (v5)", ncclTuner_v5->name);
    return &ncclTuner;
  }
  return NULL;
}
