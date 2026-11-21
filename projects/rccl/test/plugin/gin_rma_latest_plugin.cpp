/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// GIN/RMA plugin stub for GinRmaPluginVersionTests. It exports only the newest
// plugin symbols, ncclGinPlugin_v15 and ncclRmaPlugin_v17 (NCCL 2.32.6), so RCCL
// loads it only if its loaders probe those versions.
//
// v15 has the same vtable shape as v14 and v17 the same as v16, so the stub is
// typed with the v14/v16 headers. devices() reports no adapters: RCCL then only
// calls init(), devices() and finalize(), and each init()/finalize() appends a
// line to the file named in RCCL_TEST_{GIN,RMA}_LATEST_{INIT,FINALIZE}_FILE.

#include <stdio.h>
#include <stdlib.h>
#include "nccl_net.h"
#include "gin/gin_v14.h"
#include "rma/rma_v16.h"

#define __hidden __attribute__((visibility("hidden")))

static void recordLine(const char* envVar) {
  const char* path = getenv(envVar);
  if (path == nullptr) return;
  FILE* f = fopen(path, "a");
  if (f == nullptr) return;
  fputs("1\n", f);
  fclose(f);
}

__hidden ncclResult_t ginInit(void** ctx, uint64_t, ncclDebugLogger_t) {
  recordLine("RCCL_TEST_GIN_LATEST_INIT_FILE");
  *ctx = calloc(1, 1);
  return *ctx ? ncclSuccess : ncclSystemError;
}

__hidden ncclResult_t ginFinalize(void* ctx) {
  recordLine("RCCL_TEST_GIN_LATEST_FINALIZE_FILE");
  free(ctx);
  return ncclSuccess;
}

__hidden ncclResult_t rmaInit(void** ctx, uint64_t, ncclDebugLogger_t) {
  recordLine("RCCL_TEST_RMA_LATEST_INIT_FILE");
  *ctx = calloc(1, 1);
  return *ctx ? ncclSuccess : ncclSystemError;
}

__hidden ncclResult_t rmaFinalize(void* ctx) {
  recordLine("RCCL_TEST_RMA_LATEST_FINALIZE_FILE");
  free(ctx);
  return ncclSuccess;
}

__hidden ncclResult_t noDevices(int* ndev) {
  *ndev = 0;
  return ncclSuccess;
}

extern "C" __attribute__((visibility("default"))) const ncclGin_v14_t ncclGinPlugin_v15 = {
  .name = "GinLatest",
  .init = ginInit,
  .devices = noDevices,
  .finalize = ginFinalize,
};

extern "C" __attribute__((visibility("default"))) const ncclRma_v16_t ncclRmaPlugin_v17 = {
  .name = "RmaLatest",
  .init = rmaInit,
  .devices = noDevices,
  .finalize = rmaFinalize,
};
