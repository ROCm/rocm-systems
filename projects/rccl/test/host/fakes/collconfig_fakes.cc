/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fakes for the symbols defined by src/config/collconfig.cc. No state, so no
// reset. collTaskAppend reads a per-call config through these two, so a target
// that includes enqueue.cc and covers collTaskAppend references them.
//
// The isolation predicate is reproduced rather than stubbed: it is pure
// arithmetic over the config header, so a copy cannot drift in behaviour, only
// in the fields it knows about. The mask query is faithful for the
// no-selection case and fails loud otherwise, because honoring a selection
// needs the real parser (ncclAlgParse / ncclAlgValidForFuncMask, src/config/
// algorithm_{parser,registry}.cc), which this binary does not link: a test that
// scripts config->algSelection must add those oracle TUs to the target rather
// than silently read back "automatic".

#include <cstdint>

#include "nccl.h"
#include "config/collconfig.h"

#include "fail_loud.h"

bool ncclCollConfigHasAlgSelection(const ncclCollConfig_t* config) {
  if (config == NULL) return false;
  return config->algSelection != NCCL_CONFIG_UNDEF_PTR && config->algSelection[0] != '\0';
}

bool ncclCollConfigNeedAggIsolate(const ncclCollConfig_t* config) {
  if (config->size == 0) return false;  // no user config passed
  if (config->minCTAs != NCCL_CONFIG_UNDEF_INT || config->maxCTAs != NCCL_CONFIG_UNDEF_INT) return true;
  if (config->nvlsCTAs != NCCL_CONFIG_UNDEF_INT) return true;
  if (ncclCollConfigHasAlgSelection(config)) return true;
  return false;
}

ncclResult_t ncclCollConfigGetAlgMask(const ncclCollConfig_t* config, ncclFunc_t, uint64_t* outMask) {
  if (ncclCollConfigHasAlgSelection(config)) {
    FailLoud("collconfig_fakes", "ncclCollConfigGetAlgMask: config->algSelection needs the real parser");
  }
  *outMask = 0;  // no selection -> automatic
  return ncclSuccess;
}
