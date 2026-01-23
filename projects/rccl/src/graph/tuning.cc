
/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "device.h"
#include "comm.h"
#include "topo.h"
#include "nccl_tuner.h"

NCCL_PARAM(Nthreads, "NTHREADS", -2);
NCCL_PARAM(Ll128Nthreads, "LL128_NTHREADS", -2);

static int getNthreads(const char* name, int env, int min, int max, int def, int WarpSize) {
  int nt = env;
  if (nt > 0) {
    if (nt % WarpSize != 0) {
      INFO(NCCL_GRAPH|NCCL_ENV, "Invalid %s %d (must be a multiple of %d)", name, nt, WarpSize);
      nt = max;
    } else if (nt > max) {
      INFO(NCCL_GRAPH|NCCL_ENV, "Invalid %s %d (maximum %d).", name, nt, max);
      nt = max;
    } else if (nt < min) {
      INFO(NCCL_GRAPH|NCCL_ENV, "Invalid %s %d (minimum %d).", name, nt, min);
      nt = min;
    }
  } else {
    nt = def;
  }
  return nt;
}

// Parse a map of prefixes to a list of elements. The first prefix is
// optional and, if not present, the list of elements will be applied
// to all prefixes. Only the first list of elements can lack a
// prefix. Prefixes (if present) are followed by a colon. Lists of
// elements are comma delimited. Mappings of prefix to the lists of
// elements are semi-colon delimited.
//
// For example:
//
//     NCCL_ALGO="ring,collnetdirect;allreduce:tree,collnetdirect;broadcast:ring"
// Enable ring and collnetdirect for all functions, then select tree
// and collnetdirect for allreduce and ring for broadcast.
//
//     NCCL_PROTO="LL,Simple;allreduce:^LL"
// Enable LL and Simple for all functions, but everything except LL
// for allreduce.
//
//     NCCL_PROTO="^LL128;allreduce:LL128"
// Enable everything but LL128, but only LL128 for allreduce.
ncclResult_t parseList(const char* str, const char* prefixElems[], int nprefixes, const char* elems[], int nelems, int* list) {
  ncclResult_t ret = ncclSuccess;
  char* fullStr = strdup(str);
  char* tmpFullStr;
  char* fullToken = strtok_r(fullStr, ";", &tmpFullStr);
  char* subToken = nullptr;
  char* tokStr = nullptr;
  while (fullToken) {
    subToken = strdup(fullToken);
    char* tmpSubStr;
    char* prefix = strtok_r(subToken, ":", &tmpSubStr);
    char* elemList = strtok_r(NULL, ":", &tmpSubStr);
    if (elemList == NULL) {
      if (fullToken != fullStr) {
        // It makes no sense for any entry other than the first to not have a prefix,
        // because then all the prefixes before the prefix-less entry would be
        // overwritten.
        WARN("All entries except the first must have a prefix: \"%s\"", str);
        ret = ncclInvalidUsage;
        goto fail;
      }
      elemList = prefix;
      prefix = NULL;
    }

    int unset, set;
    if (elemList[0] == '^') {
      unset = 1; set = 0; elemList++;
    } else {
      unset = 0; set = 1;
    }

    bool foundPrefix = false;
    for (int p=0; p<nprefixes; p++) {
      if (prefix && strcasecmp(prefix, prefixElems[p]) != 0) continue;
      foundPrefix = true;
      for (int e=0; e<nelems; e++) list[p*nelems+e] = unset;

      tokStr = strdup(elemList);
      char* tmpStr;
      char* elem = strtok_r(tokStr, ",", &tmpStr);
      while (elem) {
        int e;
        for (e=0; e<nelems; e++) {
          if (strcasecmp(elem, elems[e]) == 0) {
            list[p*nelems+e] = set;
            break;
          }
        }
        if (e==nelems) {
          WARN("Unrecognized element token \"%s\" when parsing \"%s\"", elem, str);
          ret = ncclInvalidUsage;
          goto fail;
        }
        elem = strtok_r(NULL, ",", &tmpStr);
      }
      free(tokStr);
      tokStr = nullptr;
    }
    if (!foundPrefix) {
      WARN("Unrecognized prefix token \"%s\" when parsing \"%s\"", prefix, str);
      ret = ncclInvalidUsage;
      goto fail;
    }
    free(subToken);
    subToken = nullptr;

    fullToken = strtok_r(NULL, ";", &tmpFullStr);
  }

exit:
  free(tokStr);
  free(subToken);
  free(fullStr);
  return ret;
fail:
  goto exit;
}

// Latencies in us, Bandwidths in GB/s
// Tree { LL, LL128, Simple } , Ring { LL, LL128, Simple }
static const float baseLat  [NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = {
      { 12.0, 12.0, 17.0 }, { 12.0, 12.0, 17.0 },   // Tree, Ring
      { 12.0, 12.0, 17.0 }, { 12.0, 12.0, 17.0 },   // Collnet Direct, Chain
      {    0,    0,    0 }, {    0,    0,    0 }};  // NVLS, NVLS Tree

// NVLink, PCI, Network
#define NCCL_HW_NVLINK 0
#define NCCL_HW_PCI 1
#define NCCL_HW_NET 2

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#define RCCL_FACTOR_TABLE_SIZE 27
#define RCCL_FACTOR_TABLE_MAX_INDEX (RCCL_FACTOR_TABLE_SIZE - 1)
#endif
struct tuningModel {
  float hwLat [3][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float bwRatio [2][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  float treeCorrectionFactor[NCCL_NUM_PROTOCOLS][RCCL_FACTOR_TABLE_SIZE];
  float ringCorrectionFactor[NCCL_NUM_PROTOCOLS][RCCL_FACTOR_TABLE_SIZE];
#else
  float treeCorrectionFactor[NCCL_NUM_PROTOCOLS][27];
  float ringCorrectionFactor[NCCL_NUM_PROTOCOLS][27];
#endif
  uint64_t llProtoRanges[RCCL_TUNABLE_COLLS][NCCL_NUM_PROTOCOLS - 1][RCCL_PROTOCOL_ENTRY_SIZE];
  uint64_t channelThresholds[RCCL_TUNABLE_COLLS][RCCL_CHANNELS_TUNABLE_ENTRIES][3]; //for each collective, set for 5 channel-counts: 2,4,8,16,32,40,48,56,64, {min,max,nchannels}
};


static struct tuningModel tuning_model_0 {
  .hwLat = {
    /* NVLINK */
    { /* Tree (LL/LL128/Simple)*/ { 0.8, 1.4, 2.5 }, /* Ring (LL/LL128/Simple)*/ { 0.8, 2.2, 3.6 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 0.8 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 1.4 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 }, /* PAT */ { 0, 0, 3.6} },
    /* PCI */
    { /* Tree (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* Ring (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 5.7 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 5.7 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 }, /* PAT */ { 0, 0, 5.7} },
    /* NET */
    { /* Tree (LL/LL128/Simple)*/ { 11.8, 18.2, 20.8 }, /* Ring (LL/LL128/Simple)*/ { 9.5, 19.8, 15.1 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 11.8 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 18.2 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 }, /* PAT */ { 0, 0, 15.1} },
  },

  .bwRatio = {
    /* 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.04, 0.22, 0.91 }, /* Ring (LL/LL128/Simple)*/ { 0.04, 0.34, 1.00 }, /* CollNetDirect (Simple)*/ { 0.00, 0.00, 1.00 }, /* CollNetChain (Simple)*/ { 0.00, 0.00, 1.00 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 }, /* PAT */ { 0, 0, 0} },
    /* more than 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.04, 0.22, 0.95 }, /* Ring (LL/LL128/Simple)*/ { 0.04, 0.34, 1.00 }, /* CollNetDirect (Simple)*/ { 0.00, 0.00, 1.00 }, /* CollNetChain (Simple)*/ { 0.00, 0.00, 1.00 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 }, /* PAT */ { 0, 0, 1.00} },
  },

  .treeCorrectionFactor = {
    { 0.1, 0.2, 0.1, 0.1, 0.9, 0.3, 0.4, 0.1, 0.2, 0.4, 0.2, 0.1, 0.3, 0.3, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, },
    { 0.1, 0.3, 1.0, 0.1, 0.5, 1.0, 0.9, 1.0, 1.0, 1.0, 0.3, 0.1, 0.4, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, },
    { 0.2, 1.0, 0.1, 0.1, 0.7, 0.2, 0.4, 0.1, 0.1, 0.3, 0.4, 0.3, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9, },
  },

  .ringCorrectionFactor = {
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.4, 0.2, 0.3, 0.5, 0.3, 0.1, 0.5, 0.5, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, },
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.7, 0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, },
    { 1.0, 0.8, 0.2, 1.0, 1.0, 0.3, 1.0, 0.1, 0.1, 0.2, 0.2, 0.1, 0.5, 1.0, 0.8, 0.8, 1.0, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, },
  },

  .llProtoRanges = {{{RCCL_LL_LIMITS_UNDEFINED}}},
  .channelThresholds  = {{{CHAN_THRESHOLDS_UNDEFINED}}},
};

static struct tuningModel tuning_model_1 {
  .hwLat =
  { /* NVLINK */
    { /* Tree (LL/LL128/Simple)*/ { 1.5, 1.5, 4.5 }, /* Ring (LL/LL128/Simple)*/ { 1.5, 1.5, 4.5 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 4.5 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 4.5 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 }, /* PAT */ { 0, 0, 0} },
    /* PCI */
    { /* Tree (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* Ring (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 5.7 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 5.7 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 }, /* PAT */ { 0, 0, 0} },
    /* NET */
    { /* Tree (LL/LL128/Simple)*/ { 33.0, 33.0, 15.8 }, /* Ring (LL/LL128/Simple)*/ { 5.1, 5.1, 68.8 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 15.8 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 15.8 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 }, /* PAT */ { 0, 0, 0} },
  },

  .bwRatio =
  { /* 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.12, 1.00, 0.99 }, /* Ring (LL/LL128/Simple)*/ { 0.12, 1.00, 1.00 }, /* CollNetDirect (Simple)*/ { 0.00, 0.00, 1.00 }, /* CollNetChain (Simple)*/ { 0.00, 0.00, 1.00 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 }, /* PAT */ { 0, 0, 0} },
    /* more than 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.15, 1.00, 0.42 }, /* Ring (LL/LL128/Simple)*/ { 0.20, 1.00, 1.00 }, /* CollNetDirect (Simple)*/ { 0.00, 0.00, 1.00 }, /* CollNetChain (Simple)*/ { 0.00, 0.00, 1.00 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 }, /* PAT */ { 0, 0, 0} },
  },

  .treeCorrectionFactor = {
    { 0.5, 0.4, 0.7, 0.6, 1.0, 1.0, 0.5, 0.4, 0.1, 0.5, 0.4, 0.6, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.5, 0.4, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1, },
    { 0.5, 0.4, 0.7, 0.6, 1.0, 1.0, 0.5, 0.4, 0.1, 0.5, 0.4, 0.6, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.5, 0.4, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1, },
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.4, 0.5, 0.1, 0.6, 1.0, 1.0, 1.0, 0.6, 0.5, 0.7, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.5, 0.3, 0.3, },
  },

  .ringCorrectionFactor = {
    { 1.0, 0.5, 1.0, 1.0, 0.6, 0.7, 1.0, 1.0, 0.2, 1.0, 0.9, 0.7, 1.0, 1.0, 1.0, 0.9, 0.9, 0.8, 0.8, 0.7, 0.6, 0.5, 0.5, 0.3, 0.2, 0.1, 0.1, },
    { 1.0, 0.5, 1.0, 1.0, 0.6, 0.7, 1.0, 1.0, 0.2, 1.0, 0.9, 0.7, 1.0, 1.0, 1.0, 0.9, 0.9, 0.8, 0.8, 0.7, 0.6, 0.5, 0.5, 0.3, 0.2, 0.1, 0.1, },
    { 0.3, 1.0, 0.3, 0.1, 0.1, 0.1, 0.3, 0.7, 1.0, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.5, 0.9, 1.0, 1.0, 1.0, 1.0, },
  },

  .llProtoRanges = {{{RCCL_LL_LIMITS_UNDEFINED}}},
  .channelThresholds  = {{{CHAN_THRESHOLDS_UNDEFINED}}},
};

static struct tuningModel tuning_model_2 {
  .hwLat = {
    /* NVLINK */
    { /* Tree (LL/LL128/Simple)*/ { 1.5, 1.5, 4.5 }, /* Ring (LL/LL128/Simple)*/ { 1.5, 1.5, 4.5 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 4.5 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 4.5 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 }, /* PAT */ { 0, 0, 0} },
    /* PCI */
    { /* Tree (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* Ring (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 5.7 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 5.7 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 }, /* PAT */ { 0, 0, 0} },
    /* NET */
    { /* Tree (LL/LL128/Simple)*/ { 27.9, 27.9, 15.8 }, /* Ring (LL/LL128/Simple)*/ { 12.1, 12.1, 68.8 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 15.8 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 15.8 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 }, /* PAT */ { 0, 0, 0} },
  },

  .bwRatio = {
    /* 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.07, 1.00, 0.99 }, /* Ring (LL/LL128/Simple)*/ { 0.08, 1.00, 1.00 }, /* CollNetDirect (Simple)*/ { 0.00, 0.00, 1.00 }, /* CollNetChain (Simple)*/ { 0.00, 0.00, 1.00 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 }, /* PAT */ { 0, 0, 0} },
    /* more than 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.07, 1.00, 0.42 }, /* Ring (LL/LL128/Simple)*/ { 0.08, 1.00, 1.00 }, /* CollNetDirect (Simple)*/ { 0.00, 0.00, 1.00 }, /* CollNetChain (Simple)*/ { 0.00, 0.00, 1.00 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 }, /* PAT */ { 0, 0, 0} },
  },

  .treeCorrectionFactor = {
    { 0.1, 0.4, 0.3, 0.3, 0.2, 0.4, 0.5, 0.1, 0.1, 0.6, 0.7, 0.7, 0.8, 1.0, 0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, },
    { 0.1, 0.4, 0.3, 0.3, 0.2, 0.4, 0.5, 0.1, 0.1, 0.6, 0.7, 0.7, 0.8, 1.0, 0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, },
    { 1.0, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.5, 0.1, 0.6, 0.9, 0.8, 0.7, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.9, 0.9, 1.0, 1.0, 1.0, },
  },

  .ringCorrectionFactor = {
    { 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.4, 1.0, 1.0, 1.0, 1.0, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, },
    { 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.4, 1.0, 1.0, 1.0, 1.0, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, },
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.4, 0.5, 0.6, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, },
  },

  .llProtoRanges = {{{RCCL_LL_LIMITS_UNDEFINED}}},
  .channelThresholds  = {{{CHAN_THRESHOLDS_UNDEFINED}}},
};

static struct tuningModel tuning_model_3 {
  .hwLat = {
    /* NVLINK */
    { /* Tree (LL/LL128/Simple)*/ { 0.8, 0.0, 2.5 }, /* Ring (LL/LL128/Simple)*/ { 0.8, 0.0, 3.6 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 0.8 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 0.0 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 }, /* PAT */ { 0, 0, 0} },
    /* PCI */
    { /* Tree (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* Ring (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 5.7 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 5.7 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 }, /* PAT */ { 0, 0, 0} },
    /* NET */
    { /* Tree (LL/LL128/Simple)*/ { 12.5, 0.0, 22.4 }, /* Ring (LL/LL128/Simple)*/ { 9.5, 0.0, 19.8 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 12.5 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 0.0 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 }, /* PAT */ { 0, 0, 0} },
  },

  .bwRatio = {
    /* 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.20, 0.00, 1.75 }, /* Ring (LL/LL128/Simple)*/ { 0.20, 0.00, 1.00 }, /* CollNetDirect (Simple)*/ { 0.00, 0.00, 1.00 }, /* CollNetChain (Simple)*/ { 0.00, 0.00, 1.00 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 }, /* PAT */ { 0, 0, 0} },
    /* more than 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.20, 0.00, 0.96 }, /* Ring (LL/LL128/Simple)*/ { 0.20, 0.00, 1.00 }, /* CollNetDirect (Simple)*/ { 0.00, 0.00, 1.00 }, /* CollNetChain (Simple)*/ { 0.00, 0.00, 1.00 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 }, /* PAT */ { 0, 0, 0} },
  },

  .treeCorrectionFactor = {
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 0.2, 1.0, 0.9, 1.0, 0.6, 0.4, 0.6, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, },
    { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, },
    { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.2, 1.0, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.7, 0.8, 0.9, 0.7, 0.7, },
  },

  .ringCorrectionFactor = {
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.1, 0.2, 0.1, 0.4, 0.4, 0.2, 0.2, 0.3, 0.7, 0.5, 0.4, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, },
    { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, },
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 1.0, 0.1, 0.3, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.4, 0.7, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, },
  },

  .llProtoRanges = {{{RCCL_LL_LIMITS_UNDEFINED}}},
  .channelThresholds  = {{{CHAN_THRESHOLDS_UNDEFINED}}},
};

static struct tuningModel tuning_model_4 {
  .hwLat = {
    /* NVLINK */
    { /* Tree (LL/LL128/Simple)*/ { 0.8, 1.4, 2.5 }, /* Ring (LL/LL128/Simple)*/ { 0.8, 2.2, 3.6 }, /* CollNetDirect (Simple)*/ { 0.8, 1.4, 2.5 }, /* CollNetChain (Simple)*/ { 0.8, 1.4, 2.5 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 }, /* PAT */ { 0, 0, 0} },
    /* PCI */
    { /* Tree (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* Ring (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 5.7 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 5.7 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 }, /* PAT */ { 0, 0, 0} },
    /* NET */
    { /* Tree (LL/LL128/Simple)*/ { 32.2, 34.4, 47.6 }, /* Ring (LL/LL128/Simple)*/ { 35.4, 87.8, 209.2 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 47.6 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 47.6 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 }, /* PAT */ { 0, 0, 0} },
  },

  .bwRatio = {
    /* 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.16, 1.09, 1.61 }, /* Ring (LL/LL128/Simple)*/ { 0.15, 0.41, 1.00 }, /* CollNetDirect (Simple)*/ { 0.00, 0.00, 1.00 }, /* CollNetChain (Simple)*/ { 0.00, 0.00, 1.00 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 }, /* PAT */ { 0, 0, 0} },
    /* more than 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.16, 1.09, 1.08 }, /* Ring (LL/LL128/Simple)*/ { 0.15, 0.41, 1.00 }, /* CollNetDirect (Simple)*/ { 0.00, 0.00, 1.00 }, /* CollNetChain (Simple)*/ { 0.00, 0.00, 1.00 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 }, /* PAT */ { 0, 0, 0} },
  },

  .treeCorrectionFactor = {
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.2, 0.4, 0.6, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, },
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 1.0, 0.5, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, },
    { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.4, 0.3, 0.3, 0.1, 0.1, 1.0, 1.0, 0.7, 0.5, 0.6, 0.5, 0.6, 0.6, 0.5, 0.6, 0.6, 0.6, 0.7, },
  },

  .ringCorrectionFactor = {
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.3, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, },
    { 0.4, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.1, 0.3, 1.0, 1.0, 0.7, 0.8, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.5, 0.4, 0.3, 0.3, },
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 0.8, 0.5, 0.1, 0.7, 0.2, 0.4, 0.4, 0.6, 0.7, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, },
  },

  .llProtoRanges = {{{RCCL_LL_LIMITS_UNDEFINED}}},
  .channelThresholds  = {{{CHAN_THRESHOLDS_UNDEFINED}}},
};

static struct tuningModel tuning_model_5 {
  .hwLat = {
    /* NVLINK */
    { /* Tree (LL/LL128/Simple)*/ { 0.9, 0.9, 2.3 }, /* Ring (LL/LL128/Simple)*/ { 0.8, 0.8, 2.1 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 0.9 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 0.0 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 }, /* PAT */ { 0, 0, 0} },
    /* PCI */
    { /* Tree (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* Ring (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 5.7 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 5.7 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 }, /* PAT */ { 0, 0, 0} },
    /* NET */
    { /* Tree (LL/LL128/Simple)*/ { 10.5, 10.5, 25.0 }, /* Ring (LL/LL128/Simple)*/ { 9.5, 9.5, 320.0 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 10.5 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 0.0 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 }, /* PAT */ { 0, 0, 0} },
  },

  .bwRatio = {
    /* 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.06, 0.06, 0.11 }, /* Ring (LL/LL128/Simple)*/ { 0.08, 0.08, 1.00 }, /* CollNetDirect (Simple)*/ { 0.00, 0.00, 1.00 }, /* CollNetChain (Simple)*/ { 0.00, 0.00, 1.00 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 }, /* PAT */ { 0, 0, 0} },
    /* more than 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.06, 0.06, 0.59 }, /* Ring (LL/LL128/Simple)*/ { 0.08, 0.08, 1.00 }, /* CollNetDirect (Simple)*/ { 0.00, 0.00, 1.00 }, /* CollNetChain (Simple)*/ { 0.00, 0.00, 1.00 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 }, /* PAT */ { 0, 0, 0} },
  },

  .treeCorrectionFactor = {                                                                    /*16M 32M  64M  128M 256M 512M  1G   2G  4G */
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 1.0, 0.9, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, },
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 1.0, 0.9, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, },
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.7, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.7, 0.5, 0.6, 0.6, 0.6, },
  },

  .ringCorrectionFactor = {
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.2, 1.0, 0.4, 0.4, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, },
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.2, 1.0, 0.4, 0.4, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, },
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.7, 0.2, 1.0, 1.0, 1.0, },
  },
  // Follow order in RcclTunableColls
  .llProtoRanges = {
    /*ReduceScatter*/
    {/*LL (min/max/factor/thread_threshold)*/ {0, 655360, 1, 16}, /*LL64/128 (min/max/factor/thread_threshold)*/ {131072, 3211264, 1, 64}},
    /*AllGather*/
    {/*LL (min/max/factor/thread_threshold)*/ {0, 98304,  1, 16}, /*LL64/128 (min/max/factor/thread_threshold)*/ {98304, 5046272, 1, 64}},
    /*AllReduce*/
    {/*LL (min/max/factor/thread_threshold)*/ {0, 524288, 1, 0},/*LL64/128 (min/max/factor/thread_threshold)*/ {524288, 4415057, 3145728, 0}},
    /*Reduce*/
    {/*LL (min/max/factor/thread_threshold)*/ {0, 4096, 1, 0},/*LL64/128 (min/max/factor/thread_threshold)*/ {4096, 16777216, 1, 0}},
    /*Broadcast*/
    {/*LL (min/max/factor/thread_threshold)*/ {0, 8192, 1, 0},/*LL64/128 (min/max/factor/thread_threshold)*/ {8192, 33554432, 1, 0}},
  },

  .channelThresholds  = {{{CHAN_THRESHOLDS_UNDEFINED}}},

};

static struct tuningModel tuning_model_6 {
  .hwLat = {
    /* NVLINK */
    { /* Tree (LL/LL128/Simple)*/ { 0.9, 0.9, 2.3 }, /* Ring (LL/LL128/Simple)*/ { 0.8, 0.8, 2.1 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 0.9 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 0.0 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
    /* PCI */
    { /* Tree (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* Ring (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 5.7 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 5.7 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
    /* NET */
    { /* Tree (LL/LL128/Simple)*/ { 10.5, 10.5, 25.0 }, /* Ring (LL/LL128/Simple)*/ { 9.5, 9.5, 320.0 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 10.5 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 0.0 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
  },

  .bwRatio = {
    /* 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.06, 0.06, 0.11 }, /* Ring (LL/LL128/Simple)*/ { 0.08, 0.08, 1.00 }, /* CollNetDirect (Simple)*/ { 0.00, 0.00, 1.00 }, /* CollNetChain (Simple)*/ { 0.00, 0.00, 1.00 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
    /* more than 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.06, 0.06, 0.59 }, /* Ring (LL/LL128/Simple)*/ { 0.08, 0.08, 1.00 }, /* CollNetDirect (Simple)*/ { 0.00, 0.00, 1.00 }, /* CollNetChain (Simple)*/ { 0.00, 0.00, 1.00 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
  },

  .treeCorrectionFactor = {                                                                    /*16M 32M  64M  128M 256M 512M  1G   2G  4G */
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 1.0, 0.9, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, },
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 1.0, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 0.1, 0.9, 0.9, 0.1, 0.1, },
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.7, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.7, 0.15, 0.6, 0.1, 0.6, },
  },

  .ringCorrectionFactor = {
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.2, 1.0, 0.4, 0.4, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, },
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.2, 1.0, 0.4, 0.4, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 5.5, 0.1, 0.1, 1.0, 1.0, },
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 1.0, 0.1, 0.6, 1.0, 1.0, },
  },
  // Follow order in RcclTunableColls
  .llProtoRanges = {
    /*ReduceScatter*/
    {/*LL (min/max/factor/thread_threshold)*/ {0, 65536, 1, 16}, /*LL64/128 (min/max/factor/thread_threshold)*/ {65536, 8388608, 1, 64}},
    /*AllGather*/
    {/*LL (min/max/factor/thread_threshold)*/ {0, 65536,  1, 16}, /*LL64/128 (min/max/factor/thread_threshold)*/ {65536, 8388608, 1, 64}},
    /*AllReduce*/
    {/*LL (min/max/factor/thread_threshold)*/ {0, 262144, 1, 0},/*LL64/128 (min/max/factor/thread_threshold)*/ {262144, 70640910, 3145728, 0}},
    /*Reduce*/
    {/*LL (min/max/factor/thread_threshold)*/ {0, 16383, 1, 0},/*LL64/128 (min/max/factor/thread_threshold)*/ {16383, 16777216, 1, 0}},
    /*Broadcast*/
    {/*LL (min/max/factor/thread_threshold)*/ {0, 2048, 1, 0},/*LL64/128 (min/max/factor/thread_threshold)*/ {2048, 16777216, 1, 0}},
  },

    .channelThresholds  = {
    // For each collective, define minMax per-rank size threshold for 32,40,48,56,64 channels
    /*ReduceScatter*/ {{512, 1024, 2},{1024, 2048, 4},{2048, 4096, 8},{4096, 65536, 16}, {65536, 262144, 32}, {262144, 524288, 40}, {1,1, 48}, {524288, 1048576, 56}, {1048576, 268435457, 64}},
    /*AllGather*/     {{2048, 4096, 2},{4096, 8192, 4},{8192, 16384, 8},{16384, 262144, 16},{262144, 524288, 32}, {524288, 1048576, 40}, {1,1, 48}, {1048576, 4194304, 56}, {4194304, 268435457, 64}},
    /*AllReduce*/     {{0,0,0},{0,0,0},{0,0,0},{0,0,0},{0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}},
  },
};

static struct tuningModel tuning_model_7 {
     .hwLat = {
    /* NVLINK */
    { /* Tree (LL/LL128/Simple)*/ { 0.8, 1.4, 2.5 }, /* Ring (LL/LL128/Simple)*/ { 0.8, 2.2, 3.6 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 0.8 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 1.4 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 }, /* PAT */ { 0, 0, 3.6} },
    /* PCI */
    { /* Tree (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* Ring (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 5.7 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 5.7 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 }, /* PAT */ { 0, 0, 5.7} },
    /* NET */
    { /* Tree (LL/LL128/Simple)*/ { 11.8, 18.2, 20.8 }, /* Ring (LL/LL128/Simple)*/ { 9.5, 19.8, 15.1 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 11.8 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 18.2 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 }, /* PAT */ { 0, 0, 15.1} },
  },

  .bwRatio = {
    /* 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.051, 0.22, 0.64 }, /* Ring (LL/LL128/Simple)*/ { 0.74, 0.34, 1.00 }, /* CollNetDirect (Simple)*/ { 0.00, 0.00, 1.00 }, /* CollNetChain (Simple)*/ { 0.00, 0.00, 1.00 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 }, /* PAT */ { 0, 0, 0} },
    /* more than 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.051, 0.22, 0.64 }, /* Ring (LL/LL128/Simple)*/ { 0.74, 0.34, 1.00 }, /* CollNetDirect (Simple)*/ { 0.00, 0.00, 1.00 }, /* CollNetChain (Simple)*/ { 0.00, 0.00, 1.00 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 }, /* PAT */ { 0, 0, 1.00} },
  },

  .treeCorrectionFactor = {
    { 0.1, 0.2, 0.1, 0.1, 0.9, 0.3, 0.4, 0.1, 0.2, 0.4, 0.2, 0.1, 0.3, 0.3, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, },
    { 0.1, 0.3, 1.0, 0.1, 0.5, 1.0, 0.9, 1.0, 1.0, 1.0, 0.3, 0.1, 0.4, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, },
    { 0.2, 1.0, 0.1, 0.1, 0.7, 0.2, 0.4, 0.1, 0.1, 0.3, 0.4, 0.3, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9, },
  },

  .ringCorrectionFactor = {
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.4, 0.2, 0.3, 0.5, 0.3, 0.1, 0.5, 0.5, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, },
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.7, 0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, },
    { 1.0, 0.8, 0.2, 1.0, 1.0, 0.3, 1.0, 0.1, 0.1, 0.2, 0.2, 0.1, 0.5, 1.0, 0.8, 0.8, 1.0, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, },
  },
  // Follow order in RcclTunableColls
  .llProtoRanges = {
    /*ReduceScatter*/
    {/*LL (min/max/factor/thread_threshold)*/ {0, 65536, 1, 16}, /*LL64/128 (min/max/factor/thread_threshold)*/ {65536, 8388608, 1, 64}},
    /*AllGather*/
    {/*LL (min/max/factor/thread_threshold)*/ {0, 65536,  1, 16}, /*LL64/128 (min/max/factor/thread_threshold)*/ {65536, 8388608, 1, 64}},
    /*AllReduce*/
    {/*LL (min/max/factor/thread_threshold)*/ {0, 1048576, 1, 0},/*LL64/128 (min/max/factor/thread_threshold)*/ {1048576, 70640910, 3145728, 0}},
    /*Reduce*/
    {/*LL (min/max/factor/thread_threshold)*/ {0, 16383, 1, 0},/*LL64/128 (min/max/factor/thread_threshold)*/ {16383, 16777216, 1, 0}},
    /*Broadcast*/
    {/*LL (min/max/factor/thread_threshold)*/ {0, 2048, 1, 0},/*LL64/128 (min/max/factor/thread_threshold)*/ {2048, 16777216, 1, 0}},
  },

    .channelThresholds  = {
    // For each collective, define minMax per-rank size threshold for 32,40,48,56,64 channels
    /*ReduceScatter*/ {{512, 1024, 2},{1024, 2048, 4},{2048, 4096, 8},{4096, 65536, 16}, {65536, 262144, 32}, {262144, 524288, 40}, {1,1, 48}, {524288, 1048576, 56}, {1048576, 268435457, 64}},
    /*AllGather*/     {{2048, 4096, 2},{4096, 8192, 4},{8192, 16384, 8},{16384, 262144, 16},{262144, 524288, 32}, {524288, 1048576, 40}, {1,1, 48}, {1048576, 4194304, 56}, {4194304, 268435457, 64}},
    /*AllReduce*/     {{0,0,0},{0,0,0},{0,0,0},{0,0,0},{0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}},
  },
};

static struct tuningModel rcclTuningModel[] = {
  tuning_model_0,
  tuning_model_1,
  tuning_model_2,
  tuning_model_3,
  tuning_model_4,
  tuning_model_5,
  tuning_model_6,
  tuning_model_7,
};

// NVLS efficiency factor.
static const float nvlsEfficiency[NCCL_NUM_COMPCAPS] = {
  0.0f, // Volta
  0.0f, // Ampere
  0.85f, // Hopper
  0.74f, // Blackwell
};

// Default tuner constants
static const ncclTunerConstants_t ncclTunerConstantsDefaults = {
  .baseLatencies = {
    {  6.8, 14.0,  8.4 }, {  6.6, 14.0,  8.4 },  // Tree, Ring
    {    0,    0,    0 }, {    0,    0,    0 },  // Collnet Direct, Chain
    {    0,    0,    0 }, {    0,    0,    0 },  // NVLS, NVLS Tree
    {  8.0,  8.0,  8.0 }                         // PAT
    },
  .hwLatencies = {
  /* NVLINK */
  { { .6, 1.25, 4.0 }, { .6, 1.9, 3.4 }, /* Tree (LL/LL128/Simple), Ring (LL/LL128/Simple)*/
    {  0,    0, 3.7 }, {  0,   0,  2.8 }, /* CollNetDirect (LL/LL128/Simple), CollNetChain (LL/LL128/Simple)*/
    {  0,    0,  25 }, {  0,   0,  25 }, /* NVLS (LL/LL128/Simple), NVLSTree (LL/LL128/Simple)*/
    {  0,    0, 4.0 } /* PAT (LL/LL128/Simple)*/
    },
  /* PCI */
  { { 1.0, 1.9, 4.0 }, { 1.0, 2.5, 5.7 }, /* Tree (LL/LL128/Simple), Ring (LL/LL128/Simple)*/
    {  0,    0, 3.7 }, {  0,   0,  2.8 }, /* CollNetDirect (LL/LL128/Simple), CollNetChain (LL/LL128/Simple)*/
    {  0,    0,   0 }, {  0,   0,    0 }, /* NVLS (LL/LL128/Simple), NVLSTree (LL/LL128/Simple)*/
    {  0,    0, 4.0 } /* PAT (LL/LL128/Simple)*/
    },
  /* NET */
  { { 5.0, 8.5, 14 }, { 2.7, 4.0, 14.0 }, /* Tree (LL/LL128/Simple), Ring (LL/LL128/Simple)*/
    {   0,   0, 31 }, {   0,   0,   30 }, /* CollNetDirect (LL/LL128/Simple), CollNetChain (LL/LL128/Simple)*/
    {   0,   0, 18 }, {   0,   0,   14 }, /* NVLS (LL/LL128/Simple), NVLSTree (LL/LL128/Simple)*/
    {   0,   0, 14 } /* PAT (LL/LL128/Simple)*/
    },
  },
  .llMaxBws = {
     {39.0, 39.0, 20.4}, /* Volta-N1/Intel-N2/Intel-N4) */
     {87.7, 22.5 /*avg of ring & tree*/, 19.0}, /* Ampere-N1/AMD-N2/AMD-N4) */
     {141.0, 45.0 /*avg of ring & tree*/, 35.0}, /* Hopper-N1/AMD-N2/AMD-N4) */
     {2*141.0, 2*45.0 /*avg of ring & tree*/, 2*35.0}, /* Blackwell-N1/AMD-N2/AMD-N4) */
  },
  .perChMaxRingLL128Bws = {
    {20.0, 20.0, 20.0}, /* Volta (N1/N2/N4) */
    {20.0, 20.0, 20.0}, /* Ampere (N1/N2/N4) */
    {36.7, 36.7, 36.7}, /* Hopper (N1/N2/N4) */
    {2*36.7, 2*36.7, 2*36.7}, /* Blackwell (N1/N2/N4) */
  },
  .perChMaxTreeLL128Bws = {
    {20.0, 20.0, 20.0}, /* Volta (N1/N2/N4) */
    {20.0, 20.0, 20.0}, /* Ampere (N1/N2/N4) */
    {36.7, 36.7, 29.0}, /* Hopper (N1/N2/N4) */
    {55.6, 31.67, 20.0}, /* Blackwell (N1/N2/N4) */
  },
  .perChMaxTreeBws = {
    {26.5, 18.5, 10.0}, /* Volta (N1/N2/N4) */
    {24.0, 23.6, 17.8}, /* Ampere (N1/N2/N4) */
    {38.7, 41.4, 36.0}, /* Hopper (N1/N2/N4) */
    {70.0, 42.8, 24.0}, /* Blackwell (N1/N2/N4) */
  },
  .perChMaxNVLSTreeBws = {
    {26.5, 18.5, 10.0}, /* Volta (N1/N2/N4) */
    {24.0, 23.6, 17.8}, /* Ampere (N1/N2/N4) */
    {0.0, 57.7, 45.5}, /* Hopper (N1/N2/N4) */
    {0.0, 96.0, 43.1} /* Blackwell (N1/N2/N4) */
  }
};

NCCL_PARAM(PatEnable, "PAT_ENABLE", 0);
static int ncclPatEnable(struct ncclComm* comm) {
  if (!ncclParamPatEnable()) return 0;
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIPCC__)
  if (comm->minCompCap < 60) return 0; // Need SM60 or higher for CUDA atomics
#endif
  if (comm->nNodes != comm->nRanks) return 0; // PAT only supports 1 GPU per node
  if (comm->netDeviceType != NCCL_NET_DEVICE_HOST) return 0;   // PAT doesn't support net device offload
  return 1;
}

// Network post overhead in ns (1000 = 1 us)
NCCL_PARAM(NetOverhead, "NET_OVERHEAD", -2);

static float getNetOverhead(struct ncclComm* comm) {
  if (ncclParamNetOverhead() != -2) return ncclParamNetOverhead() * .001;
  if (comm->cpuArch == NCCL_TOPO_CPU_ARCH_X86 && comm->cpuVendor == NCCL_TOPO_CPU_VENDOR_INTEL) return 1.0;
  if (comm->cpuArch == NCCL_TOPO_CPU_ARCH_X86 && comm->cpuVendor == NCCL_TOPO_CPU_VENDOR_AMD) return 2.0;
  return 1.0;
}

NCCL_PARAM(Ll128C2c, "LL128_C2C", 1);

ncclResult_t ncclTopoInitTunerConstants(struct ncclComm* comm) {

  comm->tunerConstants = ncclTunerConstantsDefaults;

  return ncclSuccess;
}

ncclResult_t ncclTopoTuneModel(struct ncclComm* comm, int minCompCap, int maxCompCap, struct ncclTopoGraph** graphs) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  static int rcclMaxThreads[NCCL_NUM_PROTOCOLS] = {0};
  if (rcclMaxThreads[NCCL_PROTO_SIMPLE] == 0) rcclGetMaxNthreads(comm, rcclMaxThreads);
  static int maxNthreads      = rcclMaxThreads[NCCL_PROTO_SIMPLE];
  static int maxLL128Nthreads = rcclMaxThreads[NCCL_PROTO_LL128];
  static int maxLLThreads     = rcclMaxThreads[NCCL_PROTO_LL];
  int simpleDefaultThreads = (graphs[NCCL_ALGO_RING]->bwIntra*graphs[NCCL_ALGO_RING]->nChannels <= PCI_BW) ? 256 : maxNthreads;
    comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 4*comm->WarpSize, maxNthreads, simpleDefaultThreads, comm->WarpSize);
  comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] = comm->maxThreads[NCCL_ALGO_COLLNET_DIRECT][NCCL_PROTO_SIMPLE] =
    getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 4*comm->WarpSize, maxNthreads, maxNthreads, comm->WarpSize);
  comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_LL] = comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_LL] = comm->maxThreads[NCCL_ALGO_COLLNET_DIRECT][NCCL_PROTO_LL] =
    getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 4*comm->WarpSize, maxNthreads, maxLLThreads, comm->WarpSize);
  comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_LL128] = comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_LL128] =
    getNthreads("NCCL_LL128_NTHREADS", ncclParamLl128Nthreads(), 4*comm->WarpSize, maxLL128Nthreads, maxLL128Nthreads, comm->WarpSize);
#else
  int simpleDefaultThreads = (graphs[NCCL_ALGO_RING]->bwIntra*graphs[NCCL_ALGO_RING]->nChannels <= PCI_BW) ? 256 : NCCL_MAX_NTHREADS;
    comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 2*WARP_SIZE, NCCL_SIMPLE_MAX_NTHREADS, simpleDefaultThreads);
  comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] =
    getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 2*WARP_SIZE, NCCL_SIMPLE_MAX_NTHREADS, NCCL_SIMPLE_MAX_NTHREADS);
  comm->maxThreads[NCCL_ALGO_COLLNET_DIRECT][NCCL_PROTO_SIMPLE] =
    comm->maxThreads[NCCL_ALGO_COLLNET_CHAIN][NCCL_PROTO_SIMPLE] =
    comm->maxThreads[NCCL_ALGO_NVLS][NCCL_PROTO_SIMPLE] =
    comm->maxThreads[NCCL_ALGO_NVLS_TREE][NCCL_PROTO_SIMPLE] = NCCL_MAX_NTHREADS;
  comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_LL] = comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_LL] =
    getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 2*WARP_SIZE, NCCL_LL_MAX_NTHREADS, NCCL_LL_MAX_NTHREADS);
  comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_LL128] = comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_LL128] =
    getNthreads("NCCL_LL128_NTHREADS", ncclParamLl128Nthreads(), NCCL_LL128_MAX_NTHREADS/4, NCCL_LL128_MAX_NTHREADS, NCCL_LL128_MAX_NTHREADS);
#endif

  int nNodes = comm->nNodes;
  int nRanks = comm->nRanks;
  if (nRanks <= 1) return ncclSuccess;
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIPCC__)
  int compCapIndex = minCompCap >= 100 ? NCCL_BLACKWELL_COMPCAP_IDX : (minCompCap >= 90 ? NCCL_HOPPER_COMPCAP_IDX : minCompCap >= 80 ? NCCL_AMPERE_COMPCAP_IDX : NCCL_VOLTA_COMPCAP_IDX);
  int index2 = nNodes <= 2 ? nNodes-1 : 2;
  // LL: for single node, we look at GPU type; for multi-node, we look at CPU type
  int index1 = nNodes == 1 ? compCapIndex :
               (comm->cpuVendor == NCCL_TOPO_CPU_VENDOR_AMD || comm->cpuVendor == NCCL_TOPO_CPU_VENDOR_MIXED) ? 1 : 0;
  double llMaxBw = comm->tunerConstants.llMaxBws[index1][index2];
  double perChMaxTreeBw = comm->tunerConstants.perChMaxTreeBws[compCapIndex][index2];
  double perChMaxRingLL128Bw = comm->tunerConstants.perChMaxRingLL128Bws[compCapIndex][index2];
  double perChMaxTreeLL128Bw = comm->tunerConstants.perChMaxTreeLL128Bws[compCapIndex][index2];
  double perChMaxNVLSTreeBw = comm->tunerConstants.perChMaxNVLSTreeBws[compCapIndex][index2];
  // De-penalize Tree/Simple latency on Power systems to favor Tree than Ring
  if (comm->cpuArch == NCCL_TOPO_CPU_ARCH_POWER) comm->tunerConstants.hwLatencies[NCCL_HW_PCI][NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] = comm->tunerConstants.hwLatencies[NCCL_HW_PCI][NCCL_ALGO_RING][NCCL_PROTO_SIMPLE];
#endif
  float ppn = (float)nRanks / nNodes;

  int intraHw[NCCL_NUM_ALGORITHMS], hw[NCCL_NUM_ALGORITHMS];
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) intraHw[a] = graphs[a]->typeIntra == LINK_NVL ? NCCL_HW_NVLINK : NCCL_HW_PCI;
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) hw[a] = nNodes == 1 ? intraHw[a] : NCCL_HW_NET;
  INFO(NCCL_INIT,"RCCL Tuning index:%d",comm->topo->tuning);
  memcpy(comm->minMaxLLRange,
        rcclTuningModel[comm->topo->tuning].llProtoRanges,
        sizeof(rcclTuningModel[comm->topo->tuning].llProtoRanges));

  memcpy(comm->minMaxChannelThresholds,
        rcclTuningModel[comm->topo->tuning].channelThresholds,
        sizeof(rcclTuningModel[comm->topo->tuning].channelThresholds));

  for (int coll=0; coll<NCCL_NUM_FUNCTIONS; coll++) {
    int nsteps = coll == ncclFuncAllReduce ? 2*(nRanks-1) :
      coll == ncclFuncReduceScatter || coll == ncclFuncAllGather ? nRanks-1 :
      nRanks;

    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      if ((coll == ncclFuncBroadcast || coll == ncclFuncReduce) && a != NCCL_ALGO_RING) continue;
      if ((coll == ncclFuncReduceScatter || coll == ncclFuncAllGather)
          && a != NCCL_ALGO_PAT && a != NCCL_ALGO_RING
          && a != NCCL_ALGO_NVLS && a != NCCL_ALGO_COLLNET_DIRECT) continue;
      if (coll == ncclFuncAllReduce && a == NCCL_ALGO_PAT) continue;

      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        if (a == NCCL_ALGO_TREE && p == NCCL_PROTO_SIMPLE && (IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx942") || IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950")) && comm->topo->nodes[GPU].count == comm->topo->nRanks) continue;
        if ((a == NCCL_ALGO_NVLS || a == NCCL_ALGO_NVLS_TREE) && p != NCCL_PROTO_SIMPLE) continue;
        if ((coll == ncclFuncReduceScatter || coll == ncclFuncAllGather)
            && a == NCCL_ALGO_PAT && (p != NCCL_PROTO_SIMPLE || ncclPatEnable(comm) == 0)) continue;
        int collnet = (a == NCCL_ALGO_COLLNET_DIRECT || a == NCCL_ALGO_COLLNET_CHAIN) ? 1 : 0;
        float bw = nNodes <= 2 || collnet ? graphs[a]->bwIntra : graphs[a]->bwInter;
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIPCC__)
        if (a == NCCL_ALGO_NVLS_TREE || a == NCCL_ALGO_NVLS)
        {
          // NVLS/NVLStree needs at least 2 channels
          if (graphs[a]->nChannels < 2 ) continue;
          // Convert to NVLS busBW/channel
          float intraBw = graphs[a]->bwIntra * nvlsEfficiency[compCapIndex] * (graphs[a]->nChannels - 1) / graphs[a]->nChannels;
	        // AllReduce pipelines two operations.
          if (coll == ncclFuncAllReduce) {
            intraBw *= 2.0f;
          } else {
            intraBw *= (ppn - 1) / ppn;
          }
          // Handle 2 node case of NVLSTree
          float interBw = graphs[a]->bwInter * ((nNodes <= 2 && a == NCCL_ALGO_NVLS_TREE) ? 2 : 1);
          bw = std::min( {intraBw, interBw, a == NCCL_ALGO_NVLS_TREE ? (float)perChMaxNVLSTreeBw : std::numeric_limits<float>::max()} );
        };
#endif
        float busBw = graphs[a]->nChannels * bw;

        // Various model refinements
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        if (nNodes <= 2)
          busBw *= rcclTuningModel[comm->topo->tuning].bwRatio[0][a][p];
        else
          busBw *= rcclTuningModel[comm->topo->tuning].bwRatio[1][a][p];
        if (a == NCCL_ALGO_RING && p == NCCL_PROTO_LL && (coll == ncclFuncBroadcast || coll == ncclFuncReduce) && (IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx942") || IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950")) && comm->topo->nodes[GPU].count == comm->topo->nRanks) { busBw = busBw * 1.65; }
#else
        if (a == NCCL_ALGO_RING && p == NCCL_PROTO_LL) { busBw = std::min(llMaxBw, busBw * .5); }
        if (a == NCCL_ALGO_RING && p == NCCL_PROTO_LL128) busBw = std::min(busBw * (0.92 /*120.0/128.0*/), graphs[a]->nChannels*perChMaxRingLL128Bw);
        if (a == NCCL_ALGO_TREE && coll == ncclFuncAllReduce) busBw = std::min(busBw*.92, graphs[a]->nChannels*perChMaxTreeBw);
        if (a == NCCL_ALGO_TREE && p == NCCL_PROTO_LL) busBw = std::min(busBw*1.0/3.8, llMaxBw);
        if (a == NCCL_ALGO_TREE && p == NCCL_PROTO_LL128) busBw = std::min(busBw * (nNodes == 1 ? 7.0/9.0 : 120.0/128.0), graphs[a]->nChannels*perChMaxTreeLL128Bw);
        if (a == NCCL_ALGO_TREE && comm->maxTreePattern == NCCL_TOPO_PATTERN_TREE) busBw *= .85;
        if (a == NCCL_ALGO_PAT) busBw *= .75;
        if (a == NCCL_ALGO_COLLNET_DIRECT && p != NCCL_PROTO_SIMPLE) busBw = 0;  // Not used
        if (a == NCCL_ALGO_COLLNET_CHAIN && p != NCCL_PROTO_SIMPLE) busBw = 0;  // Not used
        if (a == NCCL_ALGO_COLLNET_DIRECT && p == NCCL_PROTO_SIMPLE) {
          if (coll == ncclFuncAllGather || coll == ncclFuncReduceScatter) {
            busBw = ppn * std::min(graphs[a]->bwIntra, graphs[a]->bwInter * 0.9f);
          } else {
            // Collnet+Direct requires all GPUs to have a local NIC to work at full speed
            float factor = ppn / (1.0*graphs[a]->nChannels); // GPU/NIC ratio
            factor -= (factor-1)/2;
            busBw /= factor;
            if (minCompCap >= 90) busBw *= .85;
          }
        }
        // disable collnet for allgather/reducescatter if #localranks > #heads
        // AllGather/ReduceScatter requires 1:1 GPU:NIC
        if ((a == NCCL_ALGO_NVLS || a == NCCL_ALGO_COLLNET_DIRECT) && p == NCCL_PROTO_SIMPLE && (coll == ncclFuncAllGather || coll == ncclFuncReduceScatter) && comm->nNodes > 1) {
          int nHeads = 0;
          if (coll == ncclFuncAllGather && comm->nNodes > 1 && (!comm->ncclCollNet || !comm->ncclCollNet->iallgather)) busBw = 0.0f;
          if (coll == ncclFuncReduceScatter && comm->nNodes > 1 && (!comm->ncclCollNet || !comm->ncclCollNet->ireducescatter)) busBw = 0.0f;
          if (comm->config.collnetEnable)
            nHeads = comm->collNetHeadsNum;
          else
            busBw = 0.0f;
          if (busBw > 0.0f) {
            for (int r = 0; r < comm->nRanks; r++) {
              int node = comm->rankToNode[r];
              if (comm->nodeRanks[node].localRanks > nHeads) {
                busBw = 0.0f;
                break;
              }
            }
          }
        }
#endif
        // Convert bus BW to algorithm BW
        if (!(a != NCCL_ALGO_RING && (coll == ncclFuncAllGather || coll == ncclFuncReduceScatter))) {
          float ratio = 1.0f;
          if (a == NCCL_ALGO_RING || a == NCCL_ALGO_NVLS || a == NCCL_ALGO_NVLS_TREE) ratio *= (1.0 * nRanks) / nsteps;
          else ratio *= .5;
          busBw *= ratio;
        }
        comm->bandwidths[coll][a][p] = busBw;
        comm->latencies[coll][a][p] = baseLat[a][p];
        float intraLat = rcclTuningModel[comm->topo->tuning].hwLat[intraHw[a]][a][p];
        float interLat =  ppn == 1 ? rcclTuningModel[comm->topo->tuning].hwLat[NCCL_HW_NET][NCCL_ALGO_TREE][p] : rcclTuningModel[comm->topo->tuning].hwLat[NCCL_HW_NET][a][p];
        interLat += graphs[a]->latencyInter;
        // Also add the flush extra latency
        if (p == NCCL_PROTO_SIMPLE) interLat += graphs[a]->latencyInter;

        if (a == NCCL_ALGO_RING) {
          float lat = rcclTuningModel[comm->topo->tuning].hwLat[hw[a]][a][p];
          if ((coll == ncclFuncReduce || coll == ncclFuncBroadcast)) {
            if (graphs[a]->sameChannels) {
              comm->latencies[coll][a][p] += lat;
            } else {
              if (p == NCCL_PROTO_SIMPLE) lat = rcclTuningModel[comm->topo->tuning].hwLat[hw[a]][NCCL_ALGO_TREE][p]; // Add some chunk latency, waiting for proper chunk modeling
              comm->latencies[coll][a][p] += nsteps*lat;
            }
          } else {
            // Inter-node rings still have to launch nsteps * net overhead.
            float netOverhead = 0.0;
            if (nNodes > 1) {
              netOverhead = getNetOverhead(comm);
              if (p == NCCL_PROTO_SIMPLE) netOverhead *= 3;
            }
            intraLat = std::max(intraLat, netOverhead);
            int nInterSteps = nNodes == 1 ? 0 : coll == ncclFuncAllReduce ? 2*(nNodes-1) : nNodes-1;
            comm->latencies[coll][a][p] += (nsteps-nInterSteps)*intraLat + nInterSteps*interLat;
          }
        } else if (a == NCCL_ALGO_TREE) {
          if (coll == ncclFuncAllReduce) {
            comm->latencies[coll][a][p] +=
              2 * ((nRanks/nNodes-1) * intraLat + log2i(nNodes) * interLat);
          }
        } else if (a == NCCL_ALGO_COLLNET_DIRECT) {
          comm->latencies[coll][a][p] +=
            2 * (std::min(1, (nRanks/nNodes-1)) * intraLat + (nRanks/nNodes-1) * 0.4) + interLat;  // Add 0.4 us arity serialization latency
        } else if (a == NCCL_ALGO_COLLNET_CHAIN) {
          comm->latencies[coll][a][p] += 2 * (nRanks/nNodes-1) * intraLat + interLat;
        } else if (a == NCCL_ALGO_NVLS) {
          if (nNodes > 1) comm->latencies[coll][a][p] += rcclTuningModel[comm->topo->tuning].hwLat[NCCL_HW_NET][a][p];
        } else if (a == NCCL_ALGO_NVLS_TREE) {
          comm->latencies[coll][a][p] += 2*(nNodes-1)*rcclTuningModel[comm->topo->tuning].hwLat[NCCL_HW_NET][a][p];
        } else if (a == NCCL_ALGO_PAT) {
          if (coll == ncclFuncAllGather || coll == ncclFuncReduceScatter) {
            comm->latencies[coll][a][p] += log2i(nNodes) * (interLat/3.5) // Log latency
              + nRanks * 2.8; // Still a linear part; hopefully we'll manage to remove it at some point.
          }
        }
      }
    }
  }

  // Protocols/Algorithms enable/disable, and user overrides.
  // All are enabled except ll128 which is enabled by default only in certain cases.
  int protoEnable[NCCL_NUM_FUNCTIONS*NCCL_NUM_PROTOCOLS];
  int algoEnable[NCCL_NUM_FUNCTIONS*NCCL_NUM_ALGORITHMS];
  for (int f=0; f<NCCL_NUM_FUNCTIONS; f++) {
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      protoEnable[f*NCCL_NUM_PROTOCOLS+p] = p == NCCL_PROTO_LL128 ? 2 : 1;
    }
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      algoEnable[f*NCCL_NUM_ALGORITHMS+a] = 1;
    }
  }

  const char *protoStr = ncclGetEnv("NCCL_PROTO");
  if (protoStr) {
    INFO(NCCL_ENV, "NCCL_PROTO set by environment to %s", protoStr);
    NCCLCHECK(parseList(protoStr, ncclFuncStr, NCCL_NUM_FUNCTIONS, ncclProtoStr, NCCL_NUM_PROTOCOLS, protoEnable));
  }
  const char *algoStr = ncclGetEnv("NCCL_ALGO");
  if (algoStr) {
    INFO(NCCL_ENV, "NCCL_ALGO set by environment to %s", algoStr);
    NCCLCHECK(parseList(algoStr, ncclFuncStr, NCCL_NUM_FUNCTIONS, ncclAlgoStr, NCCL_NUM_ALGORITHMS, algoEnable));
  }

  if (comm->rank == 0 && (algoStr||protoStr)) {
    constexpr int strLength = 1024;
    char funcAlgoProtoTuningStr[strLength];
    int offset = 0;
    offset += snprintf(funcAlgoProtoTuningStr+offset, std::max(0, strLength-offset), "\n     Function | ");
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      offset += snprintf(funcAlgoProtoTuningStr+offset, std::max(0, strLength-offset), "%8s  ", ncclProtoStr[p]);
    }
    offset += snprintf(funcAlgoProtoTuningStr+offset, std::max(0, strLength-offset), " | ");
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      offset += snprintf(funcAlgoProtoTuningStr+offset, std::max(0, strLength-offset), "%13s  ", ncclAlgoStr[a]);
    }
    offset += snprintf(funcAlgoProtoTuningStr+offset, std::max(0, strLength-offset), "\n");

    for (int f=0; f<NCCL_NUM_FUNCTIONS; f++) {
      offset += snprintf(funcAlgoProtoTuningStr+offset, std::max(0, strLength-offset), "%13s | ", ncclFuncStr[f]);
      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        offset += snprintf(funcAlgoProtoTuningStr+offset, std::max(0, strLength-offset), "%8d  ", protoEnable[f*NCCL_NUM_PROTOCOLS+p]);
      }
      offset += snprintf(funcAlgoProtoTuningStr+offset, std::max(0, strLength-offset), " | ");
      for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
        offset += snprintf(funcAlgoProtoTuningStr+offset, std::max(0, strLength-offset), "%13d  ", algoEnable[f*NCCL_NUM_ALGORITHMS+a]);
      }
      offset += snprintf(funcAlgoProtoTuningStr+offset, std::max(0, strLength-offset), "\n");
    }

    INFO(NCCL_ENV, "Enabled NCCL Func/Proto/Algo Matrix:%s", funcAlgoProtoTuningStr);
  }

  int nvsCount = 0;
  NCCLCHECK(ncclTopoGetNvsCount(comm->topo, &nvsCount));

  for (int f=0; f<NCCL_NUM_FUNCTIONS; f++) {
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      int disable = 0;
      // Disable NVLS Tree on a single node
      if (comm->nNodes == 1 && a == NCCL_ALGO_NVLS_TREE) disable = 1;
      // Disable Collnet+Direct, Collnet+Chain or Collnet+NVLS if collnet is not supported.
      if (comm->config.collnetEnable == 0 &&
          (a == NCCL_ALGO_COLLNET_DIRECT ||
           a == NCCL_ALGO_COLLNET_CHAIN ||
           (a == NCCL_ALGO_NVLS && comm->nNodes > 1))) disable = 1;
      // Disable CollNet+Direct if not on an NVSwitch system
      if (nvsCount == 0 && a == NCCL_ALGO_COLLNET_DIRECT) disable = 1;
      if (disable) algoEnable[f*NCCL_NUM_ALGORITHMS+a] = 0;
    }
  }

  for (int c=0; c<NCCL_NUM_FUNCTIONS; c++) for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    int pEnable = protoEnable[c*NCCL_NUM_PROTOCOLS+p];
    if (pEnable != 0 && p == NCCL_PROTO_LL128) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#if defined(ENABLE_LL128)
      // Enable LL128 by default only on gfx90a with available tuning table
      pEnable = (graphs[a]->typeInter <= PATH_PXB) && graphs[a]->typeIntra <= PATH_NVL &&
        ((IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx90a") ||
          IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx942") ||
          IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950")) && comm->topo->ll128Enabled) ? 1 : 0;
#else
      pEnable = 0;
#endif
#else
      pEnable = 1;
      if (ncclParamLl128C2c() && minCompCap >= 90) {
        // Enable LL128 by default only on Hopper/Blackwell for all connections up to P2C and PXN.
        pEnable &= (graphs[a]->typeInter <= PATH_PXN);
      } else {
        // Enable LL128 only up to PXB. Don't enable LL128 over PxN because PxN can encapsulate PxB or P2C links.
        pEnable &= (graphs[a]->typeInter <= PATH_PXB);
        if (!ncclParamLl128C2c() && minCompCap >= 90)
          INFO(NCCL_GRAPH, "Disabling LL128 over all PxN connections (PXB and C2C). This ensures that no C2C link will be used by LL128.");
      }
      pEnable &= (graphs[a]->typeIntra <= PATH_NVB);
      pEnable &= (minCompCap == maxCompCap);
      pEnable &= !(minCompCap < 70 || (minCompCap == 90 && CUDART_VERSION == 11080 && c == ncclFuncAllReduce && a == NCCL_ALGO_RING && comm->nRanks == 2));
#endif
    }
    if (pEnable == 0) comm->bandwidths[c][a][p] = 0;
    if (algoEnable[c*NCCL_NUM_ALGORITHMS+a] == 0) comm->bandwidths[c][a][p] = 0;
  }

  if (comm->rank == 0) {
    constexpr int lineLen = 1024;
    char line[lineLen];
    int offset = 0;
    for (int block=0; block<DIVUP(NCCL_NUM_ALGORITHMS, 3); block++) {
      offset = snprintf(line, lineLen, "  Algorithm   |");
      for (int ba=0; ba<3; ba++) {
        int a = block*3+ba;
        if (a >= NCCL_NUM_ALGORITHMS) continue;
        offset += snprintf(line+offset, std::max(0, lineLen-offset), " %14s   %14s   %14s |", "", ncclAlgoStr[a], "");
      }
      INFO(NCCL_TUNING, "%s", line);
      offset = snprintf(line, lineLen, "  Protocol    |");
      for (int ba=0; ba<3; ba++) {
        for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
          offset += snprintf(line+offset, std::max(0, lineLen-offset), " %14s |", ncclProtoStr[p]);
        }
      }
      INFO(NCCL_TUNING, "%s", line);
      offset = snprintf(line, lineLen, " Max NThreads |");
      for (int ba=0; ba<3; ba++) {
        int a = block*3+ba;
        if (a >= NCCL_NUM_ALGORITHMS) continue;
        for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
          offset += snprintf(line+offset, std::max(0, lineLen-offset), " %14d |", comm->maxThreads[a][p]);
        }
      }
      INFO(NCCL_TUNING, "%s", line);
      for (int c=0; c<NCCL_NUM_FUNCTIONS; c++) {
        offset = snprintf(line, lineLen, "%13s |", ncclFuncStr[c]);
        for (int ba=0; ba<3; ba++) {
          int a = block*3+ba;
          if (a >= NCCL_NUM_ALGORITHMS) continue;
          for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
            offset += snprintf(line+offset, std::max(0, lineLen-offset), "%8.1f/%6.1f |", comm->latencies[c][a][p], comm->bandwidths[c][a][p]);
          }
        }
        INFO(NCCL_TUNING, "%s", line);
      }
    }
  }

  // Set per-thread amount of work before we increase nThreads and nChannels
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
    comm->threadThresholds[a][NCCL_PROTO_LL] = NCCL_LL_THREAD_THRESHOLD;
    comm->threadThresholds[a][NCCL_PROTO_LL128] = NCCL_LL128_THREAD_THRESHOLD;
    comm->threadThresholds[a][NCCL_PROTO_SIMPLE] = NCCL_SIMPLE_THREAD_THRESHOLD;
  }
  comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_LL] *= nRanks;
  comm->threadThresholds[NCCL_ALGO_COLLNET_DIRECT][NCCL_PROTO_SIMPLE] = 256;
  comm->threadThresholds[NCCL_ALGO_COLLNET_CHAIN][NCCL_PROTO_SIMPLE] = 256;

  // Override defaults with user env
  const char* str = ncclGetEnv("NCCL_THREAD_THRESHOLDS");
  if (str) {
    INFO(NCCL_ENV, "NCCL_THREAD_THRESHOLDS set by environment to %s", str);
    ssize_t t[2][NCCL_NUM_PROTOCOLS] = {{ -2, -2, -2 }, { -2, -2, -2 }};
    sscanf(str, "%ld %ld %ld %ld %ld %ld", t[0], t[0]+1, t[0]+2, t[1], t[1]+1, t[1]+2);
    for (int a=0; a<2; a++) {
      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        if (t[a][p] >= 0) comm->threadThresholds[a][p] = t[a][p];
      }
    }
  }

  INFO(NCCL_INIT, "threadThresholds %ld/%ld/%ld | %ld/%ld/%ld | %ld | %ld",
      comm->threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_LL],
      comm->threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_LL128],
      comm->threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE],
      comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_LL],
      comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_LL128],
      comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE],
      comm->threadThresholds[NCCL_ALGO_COLLNET_DIRECT][NCCL_PROTO_SIMPLE],
      comm->threadThresholds[NCCL_ALGO_COLLNET_CHAIN][NCCL_PROTO_SIMPLE]);
  return ncclSuccess;
}

// Trees are not perfectly sticking to the model for medium sizes. Applying a static correction
// factor is not ideal but works quite well. Powers of two, 64 B to 256MB.
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIPCC__)
static float treeCorrectionFactor[NCCL_NUM_PROTOCOLS][23] = {
  { 1.0, 1.0, 1.0, 1.0,  .9,  .8,  .7,  .7,  .7,  .7,  .6,  .5,  .4,  .4,  .5,  .6,  .7,  .8,  .9, 1.0, 1.0, 1.0, 1.0 },
  { 1.0, 1.0, 1.0, 1.0, 1.0,  .9,  .8,  .8,  .8,  .7,  .6,  .6,  .6,  .6,  .6,  .6,  .8,  .9,  .9,  .9,  .9, 1.0, 1.0 },
  {  .9,  .9,  .9,  .9,  .9,  .9,  .9,  .8,  .7,  .6,  .6,  .5,  .5,  .5,  .5,  .6,  .7,  .8,  .7,  .7,  .8,  .9,  .9 }
};
#endif

ncclResult_t ncclTopoGetAlgoTime(struct ncclComm* comm, int coll, int algorithm, int protocol, size_t nBytes, int numPipeOps, float* time) {
  float bw = comm->bandwidths[coll][algorithm][protocol];
  float lat = comm->latencies[coll][algorithm][protocol];

  if (bw == 0) {
    *time = -1.0; return ncclSuccess;
  }
  int logSize = log2i(nBytes>>6);
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  logSize = std::max(0, std::min(RCCL_FACTOR_TABLE_MAX_INDEX, logSize));
  if (algorithm == NCCL_ALGO_TREE) {
    bw *= rcclTuningModel[comm->topo->tuning].treeCorrectionFactor[protocol][logSize];
  }
  else if (algorithm == NCCL_ALGO_RING && comm->nNodes > 1) {
    bw *= rcclTuningModel[comm->topo->tuning].ringCorrectionFactor[protocol][logSize];
  }

#else
  if (algorithm == NCCL_ALGO_TREE && coll == ncclFuncAllReduce && logSize >= 0 && logSize < 23) bw *= treeCorrectionFactor[protocol][logSize];
  if (algorithm == NCCL_ALGO_RING && protocol == NCCL_PROTO_SIMPLE && comm->nNodes > 1
      && coll == ncclFuncAllReduce && nBytes/(comm->nChannels*comm->nRanks) >= 64) {
    lat *= comm->minCompCap < 80 ? 1.9 : 1.4; // Plateau effect of ring
  }
#endif
  // Tree pipelining saves latency in aggregation cases
  int latCount = algorithm == NCCL_ALGO_RING ? numPipeOps : DIVUP(numPipeOps, NCCL_MAX_DEV_WORK_BATCH_COLLS);
  *time = lat * latCount + nBytes / (1000 * bw);
  return ncclSuccess;
}

/**
 * takes gfx arch name as C-style string and returns a tuning index to
 */
int rcclGetTuningIndexForArch(const char* gfxarch) {
  static const std::vector<std::pair<std::string, int>> tuningIndexMap = {
    {"gfx906", 0}, {"gfx908", 0}, {"gfx90a", 0}, {"gfx942", 5},
    {"gfx950", 6}, {"gfx1030", 0}, {"gfx1100", 0}, {"gfx1102", 0},
    {"gfx1200", 7}, {"gfx1201", 7}
  };
  if (gfxarch == nullptr) return 0;
  std::string arch(gfxarch);
  for (const auto& p : tuningIndexMap) {
    const std::string& prefix = p.first;
    if (arch.size() >= prefix.size() &&
        arch.compare(0, prefix.size(), prefix) == 0) {
      return p.second;
    }
  }
  return 0;
}